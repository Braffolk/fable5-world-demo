/**
 * Froxel volumetrics (Phase 6) — canopy light shafts + valley fog
 * (spec GPU-systems #10).
 *
 * A camera-frustum-aligned 160×90×64 froxel grid is rebuilt every frame:
 *
 *   SCATTER (per froxel): fog density = ground-hugging layer + altitude
 *   layer, boosted by hydrology moisture, broken up by wind-advected fbm,
 *   thicker at low sun. Sun visibility = terrain horizon march (5 nearest
 *   height fetches) × canopy crown-band occlusion (the sun ray pierces the
 *   crown slab at a horizontal offset — the canopy map's gaps become real
 *   dappled shafts) × the Phase-2 cloud shadow map. In-scatter = sun ×
 *   Henyey–Greenstein(g=0.5) + zenith-sky ambient.
 *
 *   INTEGRATE (per screen column): front-to-back walk of the 64 slices
 *   with closed-form per-slice integration (S/σ·(1−e^(−σ·dz))), storing
 *   accumulated in-scatter radiance + transmittance per slice.
 *
 * The post stack samples the integrated texture trilinearly at each
 * fragment's (screenUV, depth-slice) — applied to the beauty BEFORE the
 * Hillaire aerial perspective (local volumetrics under the km-scale haze).
 * Slices are exponential (2 m → 480 m); beyond that the aerial term owns
 * the ray.
 */

import { HalfFloatType, Matrix4, Vector3 } from 'three';
import type { PerspectiveCamera } from 'three';
import type { ComputeNode, Renderer, StorageTexture } from 'three/webgpu';
import { Storage3DTexture } from 'three/webgpu';
import {
  Fn,
  If,
  Loop,
  Return,
  clamp,
  exp,
  float,
  instanceIndex,
  mat4,
  smoothstep,
  texture,
  texture3D,
  textureStore,
  time,
  uniform,
  uvec3,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import type { Clouds } from '../../sky/Clouds';
import type { Atmosphere } from '../../sky/Atmosphere';
import type { Heightfield } from '../../world/Heightfield';
import { sunU } from '../../render/VegMaterials';
import { windU } from '../../render/Wind';
import { WORLD_SIZE } from '../../world/WorldConst';
import { PERIOD_FBM } from './NoiseBake';
import { canopyAt } from './Scatter';
import { hash13 } from '../noise/NoiseTSL';
import type { NF, NI, NV2, NV3, NV4 } from '../TSLTypes';

const FX = 160;
const FY = 90;
const FZ = 64;
const NEAR = 2;
const FAR = 480;

export class Froxels {
  /** integrated per-slice (accumulated in-scatter rgb, transmittance a) */
  readonly integTex: Storage3DTexture;
  private readonly scatterTex: Storage3DTexture;
  private readonly scatterK: ComputeNode;
  private readonly integK: ComputeNode;
  private readonly uCamPos = uniform(new Vector3());
  private readonly uProjInv = uniform(new Matrix4());
  private readonly uCamWorld = uniform(new Matrix4());
  /** base fog density scale (?fog=N) */
  readonly fogK = uniform(1.0);

  constructor(
    hf: Heightfield,
    atm: Atmosphere,
    canopyTex: StorageTexture | null,
    clouds: Clouds | null,
  ) {
    const mk = (): Storage3DTexture => {
      const t = new Storage3DTexture(FX, FY, FZ);
      t.type = HalfFloatType;
      return t;
    };
    this.scatterTex = mk();
    this.integTex = mk();

    const noiseA = hf.noiseA;
    const fieldsTex = hf.fieldsTex;
    if (!noiseA || !fieldsTex) throw new Error('froxels need noise + fields');

    /** exponential slice parameter (0..1) → view distance (m) */
    const sliceDist = (u: NF): NF => float(NEAR).mul(float(FAR / NEAR).pow(u));

    // ---------------- scatter: source + extinction per froxel ----------------
    this.scatterK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(FX * FY * FZ), () => {
        Return();
      });
      const x = i.mod(FX);
      const y = i.div(FX).mod(FY);
      const z = i.div(FX * FY);

      // world ray through the froxel column (mirrors getViewPosition's
      // WebGPU branch: screen-uv y flips into NDC)
      const su = vec2(float(x).add(0.5).div(FX), float(y).add(0.5).div(FY));
      const ndc = vec2(su.x, su.y.oneMinus()).mul(2).sub(1);
      const clip = vec4(ndc.x, ndc.y, 0.5, 1);
      const viewP = mat4(this.uProjInv).mul(clip);
      const dirV = viewP.xyz.div(viewP.w).normalize();
      const dirW = mat4(this.uCamWorld).mul(vec4(dirV, 0)).xyz.normalize().toVar();
      const camPos = vec3(this.uCamPos);

      // jittered sample depth inside the slice hides exp-slice banding
      const jit = hash13(vec3(float(x), float(y), float(z)));
      const dist = sliceDist(float(z).add(jit.mul(0.8).add(0.1)).div(FZ));
      const p = camPos.add(dirW.mul(dist)).toVar();

      const groundY = hf.sampleHeightNearest(p.xz);
      const hAbove = p.y.sub(groundY).max(0);
      const sunDirN = vec3(sunU.dir as unknown as NV3).normalize().toVar();

      // --- density (extinction, 1/m) -------------------------------------------
      const drift = vec2(windU.dir as unknown as NV2).mul(time.mul(3.2));
      const billow = (texture(noiseA, p.xz.add(drift).div(38 * PERIOD_FBM), 0) as unknown as NV4)
        .y.mul(0.85)
        .add(0.45);
      const uvW = clamp(p.xz.div(WORLD_SIZE).add(0.5), 0, 1);
      const moisture = (texture(fieldsTex, uvW, 0) as unknown as NV4).x;
      // dawn/dusk fog is the look; noon stays thin so shafts read, not soup
      const todK = smoothstep(0.55, 0.08, sunDirN.y).mul(1.6).add(0.35);
      const rhoGround = exp(hAbove.div(-32));
      const rhoAlt = exp(p.y.sub(120).max(0).div(-140));
      const rho = (this.fogK as unknown as NF)
        .mul(todK)
        .mul(billow)
        .mul(rhoGround.mul(0.62).add(rhoAlt.mul(0.38)))
        .mul(moisture.mul(1.1).add(0.55))
        .mul(0.0095)
        .toVar();

      // --- sun visibility --------------------------------------------------------
      const vis = float(1).toVar();
      // terrain horizon: log-spaced probes along the sun ray
      for (const dSun of [12, 30, 75, 180, 420]) {
        const q = p.add(sunDirN.mul(dSun));
        vis.mulAssign(smoothstep(-10, 2, q.y.sub(hf.sampleHeightNearest(q.xz))));
      }
      if (canopyTex) {
        // crown slab pierce point: gaps in the canopy map become shafts
        const dy = groundY.add(13).sub(p.y);
        const off = sunDirN.xz.mul(dy.max(0).div(sunDirN.y.max(0.08)));
        const cov = canopyAt(canopyTex, p.xz.add(off));
        vis.mulAssign(
          dy.greaterThan(0).select(cov.mul(0.88).oneMinus(), float(1)),
        );
      }
      if (clouds) vis.mulAssign(clouds.shadowAt(p.xz));

      // --- in-scatter source -----------------------------------------------------
      const g = 0.5; // forward-leaning HG: shafts bloom toward the sun
      const cosT = dirW.dot(sunDirN);
      const phase = float((1 - g * g) / (4 * Math.PI)).div(
        float(1 + g * g).sub(cosT.mul(2 * g)).pow(1.5),
      );
      const sunCol = (sunU.color as unknown as NV3).mul(sunU.intensity as unknown as NF);
      const amb = atm.skyColor(vec3(0, 1, 0)).mul(0.045);
      const src = sunCol.mul(phase).mul(vis).add(amb).mul(rho);
      textureStore(
        this.scatterTex,
        uvec3(x.toUint(), y.toUint(), z.toUint()),
        vec4(src, rho),
      ).toWriteOnly();
    })().compute(FX * FY * FZ);
    this.scatterK.setName('froxelScatter');

    // ---------------- integrate: front-to-back per screen column ----------------
    this.integK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(FX * FY), () => {
        Return();
      });
      const x = i.mod(FX);
      const y = i.div(FX);
      const T = float(1).toVar();
      const L = vec3(0, 0, 0).toVar();
      Loop(FZ, ({ i: k }: { readonly i: NI }) => {
        const u0 = float(k).div(FZ);
        const u1 = float(k).add(1).div(FZ);
        const dz = sliceDist(u1).sub(sliceDist(u0));
        const uvw = vec3(
          float(x).add(0.5).div(FX),
          float(y).add(0.5).div(FY),
          float(k).add(0.5).div(FZ),
        );
        const s = texture3D(this.scatterTex, uvw, 0) as unknown as NV4;
        const Ts = exp(s.a.mul(dz).negate());
        // closed-form slice integral: S/σ·(1−e^(−σ·dz)) — exact for a
        // homogeneous slice, no thin-slice bias at the far coarse slices
        const Li = s.rgb.div(s.a.max(1e-6)).mul(float(1).sub(Ts));
        L.addAssign(Li.mul(T));
        T.mulAssign(Ts);
        textureStore(
          this.integTex,
          uvec3(x.toUint(), y.toUint(), k.toUint()),
          vec4(L, T),
        ).toWriteOnly();
      });
    })().compute(FX * FY);
    this.integK.setName('froxelIntegrate');
  }

  update(renderer: Renderer, camera: PerspectiveCamera): void {
    this.uCamPos.value.copy(camera.position);
    this.uProjInv.value.copy(camera.projectionMatrixInverse);
    this.uCamWorld.value.copy(camera.matrixWorld);
    renderer.compute(this.scatterK);
    renderer.compute(this.integK);
  }

  /** composite: fog applied to a fragment color (dist in meters) */
  apply(col: NV3, dist: NF, screenUV: NV2): NV3 {
    const w = dist
      .max(NEAR)
      .div(NEAR)
      .log2()
      .div(Math.log2(FAR / NEAR))
      .clamp(0, 1);
    const fr = texture3D(
      this.integTex,
      vec3(screenUV.x, screenUV.y, w),
      0,
    ) as unknown as NV4;
    return col.mul(fr.a).add(fr.rgb);
  }
}
