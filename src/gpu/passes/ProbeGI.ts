/**
 * Irradiance probe field (Phase 3 GI) — replaces the hemisphere-light ambient.
 *
 * Terrain-only world ⇒ probes gather by RAY-MARCHING THE HEIGHTFIELD in
 * compute (no per-probe scene renders):
 *  - 256×256 horizontal probes over the 4 km world (16 m spacing),
 *    6 TERRAIN-RELATIVE layers (1.5…105 m above ground) — the slabs follow
 *    the surface, so the field is world-static (no clipmap motion).
 *  - Per probe, D jittered fibonacci directions; each ray either hits the
 *    heightfield (→ albedo proxy × sun N·L × horizon-test visibility, plus a
 *    small second-bounce fudge) or escapes to the sky-view LUT.
 *  - Radiance is projected to SH-L1 and EMA'd into storage buffers
 *    (time-sliced: PROBES_PER_FRAME per frame ⇒ full refresh < 1 s, so
 *    time-of-day changes wash through automatically).
 *  - A publish kernel copies updated probes into three rgba16f 3D textures
 *    (one per color channel, xyzw = SH c0,c1x,c1y,c1z) for hardware
 *    trilinear sampling in materials via `probeIrradiance()`.
 */

import { HalfFloatType } from 'three';
import type { Renderer, StorageTexture } from 'three/webgpu';
import { Storage3DTexture, type StorageBufferNode } from 'three/webgpu';
import {
  Fn,
  If,
  Loop,
  Return,
  clamp,
  dot,
  exp,
  float,
  instanceIndex,
  instancedArray,
  log2,
  max,
  min,
  mix,
  smoothstep,
  texture,
  texture3D,
  textureStore,
  uniform,
  uvec3,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { WORLD_SIZE } from '../../world/WorldConst';
import type { Heightfield } from '../../world/Heightfield';
import type { Atmosphere } from '../../sky/Atmosphere';
import { SUN_E } from '../../sky/Atmosphere';
import { hash12 } from '../noise/NoiseTSL';
import type { NF, NI, NV2, NV3 } from '../TSLTypes';

export const PROBE_XZ = 256;
export const PROBE_LAYERS = 6;
/** layer i sits LAYER_BASE·LAYER_RATIO^i meters above ground */
const LAYER_BASE = 1.5;
const LAYER_RATIO = 2.36;
const TOTAL = PROBE_XZ * PROBE_XZ * PROBE_LAYERS;
const PROBES_PER_FRAME = 3072;
const DIRS = 16;
const MARCH_STEPS = 16;

/**
 * Canopy slab for probe rays: tree crowns approximated as a terrain-relative
 * Beer-Lambert slab (CAN_BOT…CAN_TOP m above ground) whose extinction scales
 * with the splatted coverage map. This is what makes forest interiors DIM in
 * the probe field — without it probes see the naked heightfield, gather the
 * full sky dome + sun bounce under dense forest, and noon interiors wash out
 * to a flat ambient (no dapple contrast against the direct-sun pools).
 */
const CAN_BOT = 5;
const CAN_TOP = 23;
/**
 * Chromatic extinction per meter at coverage 1 (vertical crossing ⇒
 * T ≈ 0.10/0.21/0.09 rgb). Green passes ~2× more than red/blue — leaf
 * transmission — so everything under the canopy is bathed in the green
 * light of the references, instead of a neutral gray dim.
 */
const CAN_SIGMA = [0.128, 0.087, 0.134] as const;
const CAN_SIGMA_Y = 0.11;

export class ProbeGI {
  /** SH-L1 per color channel: xyzw = c0, c1x, c1y, c1z */
  readonly texR: Storage3DTexture;
  readonly texG: Storage3DTexture;
  readonly texB: Storage3DTexture;
  private shR: StorageBufferNode<'vec4'>;
  private shG: StorageBufferNode<'vec4'>;
  private shB: StorageBufferNode<'vec4'>;
  private gatherK: Parameters<Renderer['compute']>[0] | null = null;
  private publishK: Parameters<Renderer['compute']>[0] | null = null;
  private frameBase = uniform(0);
  private blend = uniform(0.22);
  private rot = uniform(0);
  /** frames of boosted blend after a ToD jump */
  private boost = 0;

  constructor(
    private hf: Heightfield,
    private atmosphere: Atmosphere,
    private canopyTex: StorageTexture | null = null,
  ) {
    const mk = (): Storage3DTexture => {
      const t = new Storage3DTexture(PROBE_XZ, PROBE_XZ, PROBE_LAYERS);
      t.type = HalfFloatType;
      t.generateMipmaps = false;
      return t;
    };
    this.texR = mk();
    this.texG = mk();
    this.texB = mk();
    this.shR = instancedArray(TOTAL, 'vec4');
    this.shG = instancedArray(TOTAL, 'vec4');
    this.shB = instancedArray(TOTAL, 'vec4');
  }

  async init(renderer: Renderer): Promise<void> {
    const hf = this.hf;

    const heightAt = (p: NV2): NF => hf.sampleHeight(p);

    // canopy coverage 0..1 at a world xz (0 when no canopy map is wired)
    const canopy = this.canopyTex;
    const covAt = (wxz: NV2): NF => {
      if (!canopy) return float(0);
      const uv = wxz.div(WORLD_SIZE).add(0.5);
      return (texture(canopy, uv, 0) as unknown as { x: NF }).x;
    };

    /**
     * Path-integrated coverage (m of cov=1 equivalent) of the crown slab
     * along a ray (pure expression). Slab is terrain-relative at the RAY
     * ORIGIN's ground height — crowns follow the surface closely enough at
     * probe scales. Grazing paths cap at 90 m (crown fields are patchy; one
     * mid-point coverage sample).
     */
    const canopyPath = (origin: NV3, dir: NV3, ground: NF, tCap?: NF): NF => {
      if (!canopy) return float(0);
      const ady = dir.y.abs().max(0.025);
      const dy = dir.y.greaterThanEqual(0).select(ady, ady.negate());
      const h = origin.y.sub(ground);
      const t1 = float(CAN_BOT).sub(h).div(dy);
      const t2 = float(CAN_TOP).sub(h).div(dy);
      const tEnter = min(t1, t2).max(0);
      let tExit: NF = max(t1, t2).clamp(0, 90);
      if (tCap) tExit = tExit.min(tCap);
      const len = tExit.sub(tEnter).max(0);
      const mid = origin.xz.add(dir.xz.mul(tEnter.add(tExit).mul(0.5)));
      return covAt(mid).mul(len);
    };

    /** chromatic slab transmittance for a path-integrated coverage */
    const canopyT = (path: NF): NV3 =>
      vec3(
        exp(path.mul(-CAN_SIGMA[0])),
        exp(path.mul(-CAN_SIGMA[1])),
        exp(path.mul(-CAN_SIGMA[2])),
      ) as unknown as NV3;

    // ground-hit radiance proxy: biome palette × (sun + sky fudge)
    const sunDir = this.atmosphere.sunDir;
    const hitRadiance = (hp: NV3): NV3 => {
      const uv = hp.xz.div(WORLD_SIZE).add(0.5);
      const bio = texture(hf.biomeTex as NonNullable<typeof hf.biomeTex>, uv, 0);
      const nrm = texture(hf.normalTex, uv, 0).xyz;
      const grass = vec3(0.16, 0.2, 0.09);
      const rock = vec3(0.3, 0.28, 0.25);
      const snow = vec3(0.8, 0.82, 0.88);
      const litter = vec3(0.1, 0.075, 0.045);
      let albedo: NV3 = mix(rock, grass, bio.b);
      albedo = mix(albedo, snow, bio.g);
      const cov = covAt(hp.xz);
      // forest floor is leaf litter / moss, not meadow grass — and the
      // crowns overhead shadow both the sun and the sky on the bounce source
      albedo = mix(albedo, litter, cov.mul(0.75)) as NV3;
      // sun horizon test from the hit (short march)
      const sVis = float(1).toVar();
      const sunXZ = vec2(sunDir.x, sunDir.z);
      for (let s = 1; s <= 6; s++) {
        const t = 14 * s * s;
        const sp = hp.xz.add(sunXZ.mul(t));
        const sy = hp.y.add(sunDir.y.mul(t));
        sVis.mulAssign(heightAt(sp).lessThan(sy.add(1)).select(float(1), float(0.0)));
      }
      const sunSlab = canopyT(
        cov.mul(float(CAN_TOP - CAN_BOT).div(clamp(sunDir.y, 0.25, 1))),
      );
      const ndl = clamp(dot(nrm, sunDir), 0, 1);
      const sun = this.atmosphere
        .sampleTransmittance(float(6360.35), clamp(sunDir.y, -1, 1))
        .mul(SUN_E)
        .mul(ndl)
        .mul(sVis)
        .mul(sunSlab)
        .div(Math.PI);
      // skylight on the hit (cheap: zenith sky × upness) — second bounce fudge
      const skyUp = this.atmosphere
        .skyColor(vec3(0, 1, 0))
        .mul(0.25)
        .mul(canopyT(cov.mul(11)));
      return albedo.mul(sun.add(skyUp.mul(clamp(nrm.y, 0, 1))));
    };

    this.gatherK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(PROBES_PER_FRAME), () => {
        Return();
      });
      // ground-level sun color (transmittance × irradiance) for crown glow
      const sunCol = this.atmosphere
        .sampleTransmittance(float(6360.35), clamp(sunDir.y, -1, 1))
        .mul(SUN_E)
        .div(Math.PI);
      const pid = float(this.frameBase).add(float(i)).mod(TOTAL).toInt();
      const lay = pid.div(PROBE_XZ * PROBE_XZ);
      const rem = pid.mod(PROBE_XZ * PROBE_XZ);
      const px = rem.mod(PROBE_XZ);
      const pz = rem.div(PROBE_XZ);
      const wx = float(px).add(0.5).div(PROBE_XZ).sub(0.5).mul(WORLD_SIZE);
      const wz = float(pz).add(0.5).div(PROBE_XZ).sub(0.5).mul(WORLD_SIZE);
      const ground = heightAt(vec2(wx, wz));
      const layerH = float(LAYER_BASE).mul(
        float(LAYER_RATIO).pow(float(lay)),
      );
      const ppos = vec3(wx, ground.add(layerH), wz).toVar();

      const c0R = float(0).toVar();
      const c0G = float(0).toVar();
      const c0B = float(0).toVar();
      const c1R = vec3(0).toVar();
      const c1G = vec3(0).toVar();
      const c1B = vec3(0).toVar();

      Loop(DIRS, ({ i: di }: { readonly i: NI }) => {
        // jittered fibonacci sphere (rotates per refresh pass)
        const fi = float(di).add(hash12(vec2(float(pid), float(this.rot))).mul(0.8));
        const phi = fi.mul(2.39996323).add(float(this.rot));
        const y = float(1).sub(fi.add(0.5).mul(2 / DIRS));
        const r = float(1).sub(y.mul(y)).max(0).sqrt();
        const dir = vec3(phi.cos().mul(r), y, phi.sin().mul(r)).toVar();

        // march the heightfield
        const hitT = float(-1).toVar();
        const t = float(6).toVar();
        Loop(MARCH_STEPS, () => {
          const sp = ppos.add(dir.mul(t));
          If(sp.y.lessThan(heightAt(sp.xz)).and(hitT.lessThan(0)), () => {
            hitT.assign(t);
          });
          t.mulAssign(1.6);
        });

        const L = vec3(0).toVar();
        const path = float(0).toVar();
        If(hitT.greaterThan(0), () => {
          // bounce seen THROUGH the slab (probe above canopy looking down,
          // or across a crown field) is extinguished like the sky is
          path.assign(canopyPath(ppos, dir, ground, hitT));
          L.assign(hitRadiance(ppos.add(dir.mul(hitT))).mul(canopyT(path)));
        }).Else(() => {
          path.assign(canopyPath(ppos, dir, ground));
          L.assign(this.atmosphere.skyColor(dir).mul(canopyT(path)));
        });
        // translucent crown glow: whatever the slab swallowed is re-emitted
        // as leaf-filtered green (the "lit canopy from below" of the refs) —
        // this is what keeps interiors COLORFUL instead of neutral-dark
        L.addAssign(
          sunCol
            .mul(vec3(0.04, 0.085, 0.022))
            .mul(float(1).sub(path.mul(CAN_SIGMA_Y).negate().exp())),
        );

        // SH-L1 projection (radiance)
        const w = 4 / DIRS; // Σ ≈ 4π·(1/D)·… folded into eval constants
        c0R.addAssign(L.x.mul(w));
        c0G.addAssign(L.y.mul(w));
        c0B.addAssign(L.z.mul(w));
        c1R.addAssign(dir.mul(L.x.mul(w)));
        c1G.addAssign(dir.mul(L.y.mul(w)));
        c1B.addAssign(dir.mul(L.z.mul(w)));
      });

      const blend = float(this.blend);
      const prevR = this.shR.element(pid);
      const prevG = this.shG.element(pid);
      const prevB = this.shB.element(pid);
      prevR.assign(mix(prevR, vec4(c0R, c1R.x, c1R.y, c1R.z), blend));
      prevG.assign(mix(prevG, vec4(c0G, c1G.x, c1G.y, c1G.z), blend));
      prevB.assign(mix(prevB, vec4(c0B, c1B.x, c1B.y, c1B.z), blend));
    })().compute(PROBES_PER_FRAME);
    this.gatherK.setName('probeGather');

    this.publishK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(PROBES_PER_FRAME), () => {
        Return();
      });
      const pid = float(this.frameBase).add(float(i)).mod(TOTAL).toInt();
      const lay = pid.div(PROBE_XZ * PROBE_XZ);
      const rem = pid.mod(PROBE_XZ * PROBE_XZ);
      const px = rem.mod(PROBE_XZ);
      const pz = rem.div(PROBE_XZ);
      const xyz = uvec3(px.toUint(), pz.toUint(), lay.toUint());
      textureStore(this.texR, xyz, this.shR.element(pid)).toWriteOnly();
      textureStore(this.texG, xyz, this.shG.element(pid)).toWriteOnly();
      textureStore(this.texB, xyz, this.shB.element(pid)).toWriteOnly();
    })().compute(PROBES_PER_FRAME);
    this.publishK.setName('probePublish');

    // warm the whole field once (batched: uniform updates must land per
    // dispatch, so submit pairs, awaiting only every 16 batches)
    this.blend.value = 1;
    const batches = Math.ceil(TOTAL / PROBES_PER_FRAME);
    for (let n = 0; n < batches; n++) {
      const wait = n % 16 === 15 || n === batches - 1;
      if (wait) await renderer.computeAsync([this.gatherK, this.publishK]);
      else {
        renderer.compute(this.gatherK);
        renderer.compute(this.publishK);
      }
      this.frameBase.value = (this.frameBase.value + PROBES_PER_FRAME) % TOTAL;
    }
    this.blend.value = 0.22;
  }

  /** one time slice per frame (sync submit, no readback) */
  tick(renderer: Renderer): void {
    if (!this.gatherK || !this.publishK) return;
    renderer.compute(this.gatherK);
    renderer.compute(this.publishK);
    this.frameBase.value = (this.frameBase.value + PROBES_PER_FRAME) % TOTAL;
    this.rot.value = (this.rot.value + 1.61803) % 6.2831;
    if (this.boost > 0) {
      this.boost--;
      if (this.boost === 0) this.blend.value = 0.22;
    }
  }

  /** call after a time-of-day jump: converge faster for a full cycle */
  invalidate(): void {
    this.blend.value = 0.6;
    this.boost = Math.ceil(TOTAL / PROBES_PER_FRAME) + 2;
  }

  /**
   * Irradiance at a world position/normal (SH-L1 cosine-lobe evaluation).
   * Sample point is pushed up by `lift` meters (normal offset of the caller).
   */
  /** `groundY`: optional pre-sampled ground height. The vis-buffer resolve
   *  (F9, ≤10 storage buffers/stage) passes the height from heightTex (a
   *  TEXTURE — "plentiful") so it need not bind the height storage buffer;
   *  other callers omit it and use the buffer sampler. */
  irradiance(wp: NV3, n: NV3, lift = 2.0, groundY?: NF): NV3 {
    const gY = groundY ?? this.hf.sampleHeight(wp.xz);
    const hAbove = max(wp.y.sub(gY).add(lift), 0.0);
    // invert layerH = BASE·RATIO^i  →  i = log2(h/BASE)/log2(RATIO)
    const li = clamp(
      log2(hAbove.div(LAYER_BASE).max(1)).div(Math.log2(LAYER_RATIO)),
      0,
      PROBE_LAYERS - 1,
    );
    const uvw = vec3(
      wp.x.div(WORLD_SIZE).add(0.5),
      wp.z.div(WORLD_SIZE).add(0.5),
      li.add(0.5).div(PROBE_LAYERS),
    );
    const R = texture3D(this.texR, uvw, 0);
    const G = texture3D(this.texG, uvw, 0);
    const B = texture3D(this.texB, uvw, 0);
    // L1 irradiance: E(n) ≈ c0·a0 + (c1·n)·a1   (constants folded/eye-calibrated)
    const a0 = 0.6;
    const a1 = 0.7;
    const e = vec3(
      R.x.mul(a0).add(dot(R.yzw, n).mul(a1)),
      G.x.mul(a0).add(dot(G.yzw, n).mul(a1)),
      B.x.mul(a0).add(dot(B.yzw, n).mul(a1)),
    );
    return max(e, vec3(0));
  }
}

/** smooth fallback ambient used outside the probe domain (world edge) */
export function edgeAmbient(atmo: Atmosphere, n: NV3): NV3 {
  const sky = atmo.skyColor(vec3(0, 1, 0));
  const horizon = atmo.skyColor(vec3(0.7, 0.12, 0.7).normalize());
  const ground = horizon.mul(0.35).mul(vec3(1.0, 0.92, 0.8));
  const up = clamp(n.y.mul(0.5).add(0.5), 0, 1);
  return mix(ground, mix(horizon, sky, smoothstep(0.4, 1, up)), up);
}
