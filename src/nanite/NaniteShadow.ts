/**
 * N5 — cluster-driven CSM shadows (D-N28). R0: depth-only compute SW raster.
 *
 * The PROPER Nanite path (research-grounded — see D-N28): reuse the visibility
 * rasterizer DEPTH-ONLY to render shadow depth, into OWN r32 buffers, sampled by
 * the resolve's own PCSS. NOT three's CSM shadow map (a WebGPU compute shader
 * cannot write a DepthTexture), NOT the HW vertex-pulling caster (D-N26/D-N27,
 * measured 14–31 ms/cascade — deleted). 64-bit atomics are NOT needed: depth-only
 * is a single u32 atomicMin, which the Option-C depth pass already does.
 *
 * Per cascade:
 *   - a NaniteCull chain fed the cascade ORTHO frustum (off-screen casters still
 *     cast → sphereOccluded=null, F5; coneCull=false; LOD by the MAIN camera so
 *     casters match the lit surface's LOD, no peter-pan),
 *   - a buildNaniteRaster instance REUSED depth-only: its SW depth scanline + HW
 *     big-tri path, projecting through cam.vp = the cascade LIGHT VP, writing the
 *     cascade's own vis-depth buffer (atomicMin). We run clearVis → cull →
 *     depth1 → hwDepth (NO payload pass, NO material resolve — < half main-view),
 *   - a copy kernel: vis-depth (u32 f32-bits; 0xffffffff = empty → far 1.0) → an
 *     r32float StorageTexture the resolve SAMPLES (textures are outside the
 *     10-storage-buffer/stage cap — F9/D-N23, so depth reaches the resolve as a
 *     texture, not an 11th storage buffer).
 *
 * The resolve's shadow factor = shadowFactor(worldPos, normal): cascade-select
 * (nearest covering cascade) + PCSS (blocker search + world-metric penumbra +
 * Vogel PCF), sampling our textures. LOCKSTEP is automatic: raster and sample use
 * the SAME csm.lights[c].shadow.camera VP (cascVP) — three stays the matrix +
 * texel-snap authority, we never recompute ortho frusta.
 *
 * R1 (landed) gates the per-cascade raster on a light-VP change (CsmCached freezes
 * the pose between refreshes → bit-identical VP → skip + keep the cached texture):
 * ~0 cost on a static camera, the [1,2,3,6] cadence moving. R3 will add the
 * static/dynamic split so cached cascades still get moving wind shadows. Nanite
 * shadows are DEFAULT-ON; ?nanshadow=0 disables the whole system (the C1 HW caster
 * is retired — D-N28).
 */

import { FloatType, Frustum, Matrix4, NearestFilter, RedFormat, Vector4 } from 'three';
import type { PerspectiveCamera, Texture } from 'three';
import { StorageTexture, type Renderer } from 'three/webgpu';
import {
  Fn,
  If,
  float,
  instanceIndex,
  int,
  interleavedGradientNoise,
  normalize,
  screenCoordinate,
  textureLoad,
  textureStore,
  uint,
  uvec2,
  vec3,
  vec4,
} from 'three/tsl';
import { vogelDiskSample } from 'three/tsl';
import type { NB, NF, NV2, NV3 } from '../gpu/TSLTypes';
import type { RegistryGpu } from './GeometryRegistry';
import { makeNaniteCam, type NaniteCam } from './NaniteCommon';
import { buildNaniteCull, type NaniteCullChain } from './NaniteCull';
import { buildNaniteHzb, type NaniteHzb } from './NaniteHzb';
import type { TerrainDisp, TrunkWindOpt } from './NaniteFetch';
import {
  buildNaniteRaster,
  makeVisBuffers,
  type NaniteRasterHandles,
  type NaniteVisBuffers,
} from './NaniteRaster';
import { bcU2F, dispatch, elemU, minU, uniformArrV4, uniformF, uniformMat4 } from './Tsl';
import type { UniformArrV4, UniformMat4 } from './Tsl';
import { sunU } from '../render/VegMaterials';

/** CSM default cascade count (csmcasc can lower it — we guard per-cascade) */
export const SHADOW_CASCADES = 4;
const SHADOW_MAP = 2048;
const SHADOW_PIX = SHADOW_MAP * SHADOW_MAP;

// PCSS (mirrors ShadowSetup.ts — world-metric penumbra)
const BLOCKER_TAPS = 6;
const PCF_TAPS = 9;
const SUN_TAN = 0.011;
const MIN_PENUMBRA_M = 0.05;
const MAX_PENUMBRA_M = 3.0;
/** world-space normal offset on the receiver (acne suppression) */
const NORMAL_BIAS_M = 0.12;
/** WORLD-SPACE depth bias (metres). Converted to [0,1] cascade depth per cascade
 *  via depthRange (cascParam.y) — the cascade near/far span the full lightMargin+
 *  maxFar range, so a constant [0,1] bias would be metres of slop; world-metric
 *  keeps it a fixed ~0.3 m regardless of cascade depth range. */
const DEPTH_BIAS_M = 0.35;
const TAU = 6.28318530718;

interface CascadeLightCam {
  projectionMatrix: Matrix4;
  matrixWorldInverse: Matrix4;
  left?: number;
  right?: number;
  top?: number;
  bottom?: number;
  near?: number;
  far?: number;
}

interface Cascade {
  cam: NaniteCam;
  cull: NaniteCullChain;
  vis: NaniteVisBuffers;
  raster: NaniteRasterHandles;
  depthTex: StorageTexture;
  kCopy: unknown;
  /** S2-OCCL: per-cascade light HZB for the two-phase occlusion cull. null when
   *  ?shadowoccl is off (single-phase, no occlusion — the pre-S2-OCCL behaviour). */
  hzb: NaniteHzb | null;
  count: number;
  /** false until runPhase1 has dispatched at least once — its GPU buffers do not
   *  exist before then, so readCounts must skip it (CSM cascades init lazily). */
  ran: boolean;
  /** R1 CADENCE: the light VP that rastered the depth currently in depthTex. We
   *  re-raster a cascade only when its VP differs from this (CsmCached freezes the
   *  light pose between refreshes → bit-identical VP → exact-equality skip). Init
   *  identity; `ran` forces the first raster regardless. */
  lastVP: Matrix4;
}

interface NamedKernel {
  setName(n: string): unknown;
}

export interface NaniteShadow {
  /** per-cascade: refresh cascade VP/planes, cull, depth-raster, copy → texture.
   *  Call BEFORE post.render() (the resolve samples the textures that frame). */
  run(renderer: Renderer, csm: object | null, mainCamera: PerspectiveCamera): void;
  /** TSL shadow factor in [0,1] for the resolve: nearest-covering-cascade select
   *  + PCSS over our own per-cascade depth textures. worldPos+normal world-space. */
  /** pix: the pixel coord for the IGN sample-rotation noise. Defaults to
   *  screenCoordinate (fragment use); a COMPUTE caller (S0 half-res) MUST pass its
   *  own coord — screenCoordinate/fragCoord is undefined in a compute stage. */
  shadowFactor(worldPos: NV3, normal: NV3, pix?: NV2): NF;
  /** ?nandbg=shadowc debug: which cascade covers a world pos (color tint) */
  cascadeTint(worldPos: NV3): NV3;
  /** ?nandbg=shadowd debug: the stored cascade depth at a world pos (1=empty) */
  debugDepth(worldPos: NV3): NF;
  /** per-cascade visible-cluster counts (HUD) */
  readCounts(renderer: Renderer): Promise<number[]>;
  /** R1 validation: bitmask of cascades RE-RASTERED on the last run() (bit c = 1 ⇒
   *  cascade c re-rastered; 0 ⇒ served from cache). Static camera → 0 after warmup. */
  rasteredMask(): number;
  cascades: number;
}

export function buildNaniteShadow(
  gpu: RegistryGpu,
  instanceCount: number,
  heightTex: Texture,
  /** terrain micro-displacement — MUST match the camera raster's makeFetch */
  disp?: TerrainDisp,
  /** trunk wind — MUST match the camera raster's makeFetch */
  wind?: TrunkWindOpt,
): NaniteShadow {
  const cascades: Cascade[] = [];
  const cascVP: UniformMat4[] = [];
  // per-cascade (span_m, depthRange_m, texel, radius) for the world-metric PCSS
  const cascParam: UniformArrV4 = uniformArrV4(
    Array.from({ length: SHADOW_CASCADES }, () => new Vector4(1, 1, 1 / SHADOW_MAP, 1.15)),
  );

  // S4 (D-N29): DAG-DECOUPLED caster LOD — cast shadows from a COARSER DAG cut than
  // the camera view. The lit surface is at camera LOD; penumbra hides caster
  // silhouette error (more so the farther + softer the cascade). The shadow cull's
  // projK already runs ~2× FINER than the perspective camera (projK = uH·½ = 1024 at
  // cotHalfFov 1 vs ~468), so shadows were OVER-detailed — a coarser per-cascade τ
  // removes that waste AND sheds the moving raster's dominant cost. τ grows with the
  // cascade's distance band (c0 near = sharpest contact, c3 far = coarsest). The LOD
  // distance is camera-relative (cam.camPos = main camera), so the bands line up.
  // ?shadowtau=N scales the base px (default 4); =1 ≈ near the old camera-LOD detail.
  const shParams = new URLSearchParams(window.location.search);
  const shTauBase = Math.max(0.25, Number(shParams.get('shadowtau') ?? 4));
  const CASCADE_TAU_MUL = [1, 1.7, 2.7, 4];
  // S4 (cont): the DOMINANT shadow casters here are NOT DAG (terrain clipmap +
  // explicit vegetation — dagClusters is ~6% of the view), so the τ cut above only
  // trims the rock/bark minority. The cross-class lever is the min-screen-size cull
  // (NaniteCull "applies to all classes"): drop any shadow caster whose sphere
  // projects sub-shadowmap-pixel. Epic ships exactly this (small foliage dropped
  // from shadow maps → contact shadows). Grows per cascade (coarser far). projK is
  // ~1024 for the 2048 map, so minPx px ≈ that many shadow texels. ?shadowminpx=N.
  const shMinPxBase = Math.max(0, Number(shParams.get('shadowminpx') ?? 0));
  // S2-OCCL (D-N29, the log-av redirect): the headline moving-raster lever. The
  // shadow cascades carry ~38× the camera's clusters because they had NO occlusion
  // cull (sphereOccluded=null) — a per-cascade LIGHT HZB + two-phase cull skips
  // casters HIDDEN FROM THE SUN (zero shadow-quality loss). ORTHO occlusion (the
  // perspective test assumes a finite eye). Behind ?shadowoccl=1 until validated.
  const occlOn = shParams.get('shadowoccl') === '1';
  const sunDirNode = normalize(vec3(sunU.dir)) as unknown as NV3;

  for (let c = 0; c < SHADOW_CASCADES; c++) {
    const cam = makeNaniteCam(SHADOW_MAP, SHADOW_MAP);
    const vis = makeVisBuffers(SHADOW_PIX);
    // light HZB over THIS cascade's vis depth + the ORTHO occlusion test (span =
    // cascParam[c].x, the ortho width in m, live each frame). Built only when on.
    const hzb = occlOn ? buildNaniteHzb(vis.depthV.ro, cam) : null;
    const occl = hzb
      ? hzb.makeOrthoOccluded(sunDirNode, cascParam.element(int(c)).x as unknown as NF)
      : null;
    const cullTau = uniformF(shTauBase * (CASCADE_TAU_MUL[c] ?? 4));
    const cullMinPx = uniformF(shMinPxBase * (CASCADE_TAU_MUL[c] ?? 4));
    const cull = buildNaniteCull(gpu, instanceCount, cam, occl, {
      coneCull: false,
      tau: cullTau,
      minPx: cullMinPx,
    });
    // REUSE the raster depth-only: clearVis/depth1/hwDepth (+ depth2 when occl on).
    const raster = buildNaniteRaster(gpu, heightTex, cam, cull, vis, 'flat', false, disp, wind);

    const depthTex = new StorageTexture(SHADOW_MAP, SHADOW_MAP);
    depthTex.type = FloatType;
    depthTex.format = RedFormat;
    depthTex.magFilter = NearestFilter;
    depthTex.minFilter = NearestFilter;
    depthTex.generateMipmaps = false;
    depthTex.name = `nanShadowDepth${c}`;

    // copy vis-depth (u32 f32-bits) → r32f texture; clear sentinel → far (1.0)
    const kCopy = Fn(() => {
      const px = instanceIndex;
      If(px.lessThan(uint(SHADOW_PIX)), () => {
        const raw = elemU(vis.depthV.ro, px).toVar();
        const d = raw.equal(uint(0xffffffff)).select(float(1), bcU2F(raw));
        const x = px.mod(uint(SHADOW_MAP));
        const y = px.div(uint(SHADOW_MAP));
        textureStore(depthTex, uvec2(x, y), vec4(d, 0, 0, 1)).toWriteOnly();
      });
    })().compute(SHADOW_PIX, [256]);
    (kCopy as unknown as NamedKernel).setName(`nanShadowCopy${c}`);

    cascVP.push(uniformMat4(new Matrix4()));
    cascades.push({
      cam,
      cull,
      vis,
      raster,
      depthTex,
      kCopy,
      hzb,
      count: -1,
      ran: false,
      lastVP: new Matrix4(),
    });
  }

  const cascM = new Matrix4();
  const cascFrustum = new Frustum();
  let lastRasterMask = 0;

  const run = (renderer: Renderer, csm: object | null, mainCamera: PerspectiveCamera): void => {
    const lights = (csm as { lights?: { shadow?: { camera?: CascadeLightCam } }[] } | null)
      ?.lights;
    if (!lights) return;
    let mask = 0;
    for (let c = 0; c < SHADOW_CASCADES; c++) {
      const lcam = lights[c]?.shadow?.camera;
      if (!lcam || !Number.isFinite(lcam.left ?? NaN)) continue;
      const cc = cascades[c]!;
      // the cascade light VP (three is the matrix + texel-snap authority) — used
      // for BOTH the raster projection (cam.vp) and the resolve sample (cascVP).
      cascM.multiplyMatrices(lcam.projectionMatrix, lcam.matrixWorldInverse);
      // R1 CADENCE: re-raster ONLY when this cascade's VP changed since its last
      // raster. CsmCached freezes the light pose between refreshes (CsmCached.ts:294
      // — a cached cascade `continue`s without moving lwLight), so the recomputed VP
      // is BIT-IDENTICAL and exact equality is a robust gate (no epsilon). On skip,
      // the depthTex StorageTexture retains last refresh's depth and cascVP[c]/
      // cascParam[c] are left untouched → raster/sample lockstep holds (D-N28). A
      // fully static camera caches all four cascades ⇒ ~0 shadow cost; a moving
      // camera re-rasters c0 every frame and c1/2/3 on the [2,3,6] cadence + drift.
      if (cc.ran && cascM.equals(cc.lastVP)) continue;
      mask |= 1 << c;
      cascFrustum.setFromProjectionMatrix(cascM);
      // S2-OCCL: prevVp = this cascade's LAST raster VP — the light HZB still holds
      // depth from that raster, so phase-1 occlusion tests new clusters against it.
      cc.cam.prevVp.value.copy(cc.cam.vp.value);
      cc.cam.vp.value.copy(cascM);
      cascVP[c]!.value.copy(cascM);
      for (let p = 0; p < 6; p++) {
        const pl = cascFrustum.planes[p];
        if (pl) cc.cam.planes.array[p]?.set(pl.normal.x, pl.normal.y, pl.normal.z, pl.constant);
      }
      cc.cam.camPos.value.copy(mainCamera.position); // LOD by the camera, not the light
      // world-metric PCSS params from the ortho extents
      const span = Math.max((lcam.right ?? 1) - (lcam.left ?? 0), 1);
      const depthRange = Math.max((lcam.far ?? 1) - (lcam.near ?? 0), 1);
      (cascParam.array[c] as Vector4).set(span, depthRange, 1 / SHADOW_MAP, 1.15);
      // depth-only raster of the cascade, then copy to the sampled texture
      cc.raster.clearVis(renderer);
      cc.cull.runPhase1(renderer);
      cc.raster.depth1(renderer);
      cc.raster.hwDepth(renderer, mainCamera); // camera arg unused (HW vertexNode uses cam.vp)
      if (cc.hzb) {
        // two-phase: build a fresh light HZB from phase-1 depth, then re-cull —
        // casters whose nearest-to-sun point is behind the recorded depth are hidden
        // from the sun and skipped (depth2 rasters only the survivors).
        cc.hzb.build(renderer);
        cc.cull.runPhase2(renderer);
        cc.raster.depth2(renderer);
        cc.raster.hwDepth(renderer, mainCamera);
      }
      dispatch(renderer, cc.kCopy);
      cc.lastVP.copy(cascM);
      cc.ran = true;
    }
    lastRasterMask = mask;
  };

  // ---- resolve-side PCSS over our own textures --------------------------------
  const depthAt = (c: number, uv: NV2): NF => {
    const u = (uv as unknown as { clamp(a: number, b: number): NV2 }).clamp(0, 1);
    const tx = minU(uint((u as unknown as { x: NF }).x.mul(SHADOW_MAP)), uint(SHADOW_MAP - 1));
    const ty = minU(uint((u as unknown as { y: NF }).y.mul(SHADOW_MAP)), uint(SHADOW_MAP - 1));
    return (textureLoad(cascades[c]!.depthTex, uvec2(tx, ty)) as unknown as { x: NF }).x;
  };

  /** PCSS over cascade c at uv with receiver depth (mirrors ShadowSetup.pcssFilter
   *  but reads our r32 texture and does a manual depth compare). Returns lit∈[0,1]. */
  const pcss = (c: number, uv: NV2, receiver: NF, pix: NV2): NF =>
    Fn(() => {
      const param = cascParam.element(int(c));
      const span = (param as unknown as { x: NF }).x.max(1);
      const depthRange = (param as unknown as { y: NF }).y.max(1);
      const texel = (param as unknown as { z: NF }).z;
      const radius = (param as unknown as { w: NF }).w.max(1);
      const phi = interleavedGradientNoise(pix).mul(TAU);
      // world-metric depth bias → [0,1] depth via this cascade's depthRange
      const dBias = float(DEPTH_BIAS_M).div(depthRange);

      // blocker search (raw depth reads)
      const searchR = texel.mul(6).mul(radius);
      const blockerSum = float(0).toVar();
      const blockerCount = float(0).toVar();
      for (let i = 0; i < BLOCKER_TAPS; i++) {
        const tap = vogelDiskSample(float(i), float(BLOCKER_TAPS), phi) as unknown as NV2;
        const uvT = (uv as unknown as { add(o: unknown): NV2 }).add(
          (tap as unknown as { mul(o: unknown): NV2 }).mul(searchR),
        );
        const d = depthAt(c, uvT);
        const isBlk = d.lessThan(receiver.sub(dBias));
        blockerSum.addAssign(isBlk.select(d, float(0)));
        blockerCount.addAssign(isBlk.select(float(1), float(0)));
      }

      const result = float(1).toVar();
      If(blockerCount.greaterThan(0.5), () => {
        const avgBlocker = blockerSum.div(blockerCount);
        const gapM = receiver.sub(avgBlocker).mul(depthRange);
        const penumbraM = gapM.mul(SUN_TAN).clamp(MIN_PENUMBRA_M, MAX_PENUMBRA_M);
        const penumbra = penumbraM.div(span).max(texel.mul(0.75)).mul(radius);
        const sum = float(0).toVar();
        for (let i = 0; i < PCF_TAPS; i++) {
          const tap = vogelDiskSample(float(i), float(PCF_TAPS), phi) as unknown as NV2;
          const uvT = (uv as unknown as { add(o: unknown): NV2 }).add(
            (tap as unknown as { mul(o: unknown): NV2 }).mul(penumbra),
          );
          const lit = receiver.lessThanEqual(depthAt(c, uvT).add(dBias));
          sum.addAssign(lit.select(float(1), float(0)));
        }
        result.assign(sum.div(PCF_TAPS));
      });
      return result;
    })() as unknown as NF;

  // shadowCoord (uv ∈ [0,1], z ∈ [0,1]) + coverage test for cascade c at wp.
  const cascadeCoord = (
    c: number,
    wp: NV3,
  ): { uv: NV2; z: NF; inside: NB } => {
    const sc = cascVP[c]!.mul(vec4(wp, 1));
    const ndc = (sc as unknown as { xyz: NV3 }).xyz; // ortho → w == 1
    const uv = (ndc as unknown as { xy: NV2 }).xy.mul(0.5).add(0.5) as unknown as NV2;
    const z = (ndc as unknown as { z: NF }).z;
    const ux = (uv as unknown as { x: NF }).x;
    const uy = (uv as unknown as { y: NF }).y;
    const inside = ux
      .greaterThanEqual(0)
      .and(ux.lessThanEqual(1))
      .and(uy.greaterThanEqual(0))
      .and(uy.lessThanEqual(1))
      .and(z.greaterThanEqual(0))
      .and(z.lessThanEqual(1)) as unknown as NB;
    return { uv, z, inside };
  };

  const shadowFactor = (worldPos: NV3, normal: NV3, pix?: NV2): NF =>
    Fn(() => {
      // IGN rotation source: caller-supplied pixel coord (compute) or fragCoord.
      const pc = (pix ?? (screenCoordinate.xy as unknown as NV2)) as NV2;
      const wp = (worldPos as unknown as { add(o: unknown): NV3 }).add(
        (normal as unknown as { mul(o: number): NV3 }).mul(NORMAL_BIAS_M),
      ).toVar();
      const sf = float(1).toVar();
      const found = float(0).toVar();
      for (let c = 0; c < SHADOW_CASCADES; c++) {
        If(found.equal(0), () => {
          const { uv, z, inside } = cascadeCoord(c, wp as unknown as NV3);
          If(inside, () => {
            found.assign(1);
            sf.assign(pcss(c, uv, z, pc));
          });
        });
      }
      return sf;
    })() as unknown as NF;

  // ?nandbg=shadowc — DIAG: cascade-0 raw uv (r=uv.x, g=uv.y) — gradient in
  // [0,1] means coverage maths are right; saturated/uniform means cascVP×wp wrong.
  const cascadeTint = (worldPos: NV3): NV3 =>
    Fn(() => {
      const tints = [vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1), vec3(1, 1, 0)];
      const col = vec3(0).toVar();
      const found = float(0).toVar();
      for (let c = 0; c < SHADOW_CASCADES; c++) {
        If(found.equal(0), () => {
          const { inside } = cascadeCoord(c, worldPos);
          If(inside, () => {
            found.assign(1);
            col.assign(tints[c]!);
          });
        });
      }
      return col;
    })() as unknown as NV3;

  // ?nandbg=shadowd — the stored cascade depth at the selected cascade (1=empty)
  const debugDepth = (worldPos: NV3): NF =>
    Fn(() => {
      const d = float(1).toVar();
      const found = float(0).toVar();
      for (let c = 0; c < SHADOW_CASCADES; c++) {
        If(found.equal(0), () => {
          const { uv, inside } = cascadeCoord(c, worldPos);
          If(inside, () => {
            found.assign(1);
            d.assign(depthAt(c, uv));
          });
        });
      }
      return d;
    })() as unknown as NF;

  const readCounts = async (renderer: Renderer): Promise<number[]> => {
    const out = await Promise.all(
      cascades.map(async (cc) => {
        if (!cc.ran) return -1; // GPU buffers not created yet
        const counts = await cc.cull.readCounts(renderer);
        cc.count = counts.visClusters;
        return counts.visClusters;
      }),
    );
    return out;
  };

  const rasteredMask = (): number => lastRasterMask;

  return {
    run,
    shadowFactor,
    cascadeTint,
    debugDepth,
    readCounts,
    rasteredMask,
    cascades: SHADOW_CASCADES,
  };
}
