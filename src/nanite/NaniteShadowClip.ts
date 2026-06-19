/**
 * S3 — SCREEN-DENSITY SHADOW CLIPMAP (D-N29, the resolved sun-shadow rethink).
 *
 * Replaces the 4 fixed CSM cascades (NaniteShadow.ts) with ONE camera-centred
 * clipmap: L concentric ortho levels along the sun, each at the SAME texel
 * resolution but DOUBLING world half-extent (E_k = E_0·2^k). Texel density
 * therefore HALVES per level outward → ~constant shadow-texels-per-screen-pixel
 * (the level whose outer edge sits at distance d has texel size ∝ d, and a screen
 * pixel's world footprint is also ∝ d — so the ratio is constant across levels;
 * pick E_0/T so it lands near 1). The near level is crisp (BEAUTY); the far
 * levels are coarse, so the per-level min-screen-size cull + the natural snap
 * cadence shed their cluster count (PERF). This is D-N29 point (1): "RESOLUTION
 * REALLOCATED by screen-pixel density (clipmap, not fixed 4-cascade splits)".
 *
 * WHY a clipmap beats the cascades here (measured: the cascades carry ~38× the
 * camera's clusters = WIDE-ORTHO × 4-CASCADE × fine-geo, S2-OCCL/S4 both weak):
 *  - HOLLOW RINGS: each caster rasters into EXACTLY ONE level (the finest whose
 *    box reaches it; coarser levels reject it via NaniteCull innerReject). Kills
 *    the 4-cascade re-raster of the near field. Gap-free because shadows project
 *    along the sun ⇒ a caster and everything it shadows share light-XY, so the
 *    finer level covering the caster also covers its shadow.
 *  - SNAP CADENCE: each level's centre snaps to its own texel grid (anti-crawl,
 *    like CSM texel-snap). A coarse level's texel is huge ⇒ its snapped VP rarely
 *    changes ⇒ the R1 exact-VP-equality gate caches it for many frames. The fine
 *    level (small texel) re-rasters often, but is small. Screen-density caching
 *    falls out of the geometry — no separate cadence table.
 *
 * Memory: ONE shared vis buffer (rastered → copied to the level texture, then
 * reused for the next level — levels are processed sequentially), plus one r32f
 * texture per level the resolve samples. Cheaper than 4 cascades' 4 vis buffers.
 *
 * Same NaniteShadow interface as the cascade path ⇒ the resolve is unchanged and
 * ?shadowclip=0 A/Bs back to NaniteShadow.ts. CSM is dropped for shadow GEOMETRY;
 * world.csm survives only as the cloud-gate carrier in the resolve (severed in a
 * later cleanup). run()'s csm arg is ignored — VPs come from the sun + camera.
 */

import {
  FloatType,
  Frustum,
  Matrix4,
  NearestFilter,
  OrthographicCamera,
  RedFormat,
  Vector3,
  Vector4,
  WebGPUCoordinateSystem,
} from 'three';
import type { PerspectiveCamera, Texture } from 'three';
import { StorageTexture, type Renderer } from 'three/webgpu';
import {
  Fn,
  If,
  float,
  instanceIndex,
  int,
  interleavedGradientNoise,
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
import { buildClipCull, type ClipCull } from './NaniteClipCull';
import type { TerrainDisp, TrunkWindOpt } from './NaniteFetch';
import {
  buildNaniteRaster,
  makeVisBuffers,
  type NaniteRasterHandles,
  type NaniteVisBuffers,
} from './NaniteRaster';
import { bcU2F, dispatch, elemU, minU, uniformArrV4, uniformF, uniformMat4 } from './Tsl';
import type { UniformArrV4, UniformF, UniformMat4 } from './Tsl';
import { sunU } from '../render/VegMaterials';
import type { NaniteShadow } from './NaniteShadow';

// PCSS (mirrors NaniteShadow.ts / ShadowSetup.ts — world-metric penumbra)
const BLOCKER_TAPS = 6;
const PCF_TAPS = 9;
const SUN_TAN = 0.011;
const MIN_PENUMBRA_M = 0.05;
const MAX_PENUMBRA_M = 3.0;
const NORMAL_BIAS_M = 0.12;
const DEPTH_BIAS_M = 0.35;
const TAU = 6.28318530718;

/** hier BFS frontier capacity per shadow cull chain (SHADOW-HIER): shadow cuts are
 *  bounded (clipmap ring / cascade box), far below the camera's QRASTER_CAP flood, so
 *  the N×2 frontier buffers can be small. 2M entries = 16 MB/buffer (vs 64 MB at 8M). */
const SHADOW_FRONTIER_CAP = 1 << 21; // 2,097,152

interface NamedKernel {
  setName(n: string): unknown;
}

interface Level {
  cam: NaniteCam;
  /** this level's hollow uniform (1/E_k; 0 for level 0) */
  innerReject: UniformF;
  raster: NaniteRasterHandles;
  depthTex: StorageTexture;
  kCopy: unknown;
  /** ortho half-extent E_k (light-XY), world metres */
  half: number;
  /** three OrthographicCamera that produces this level's VP each frame */
  ortho: OrthographicCamera;
  /** R1 cadence: the VP that rastered the depth currently in depthTex. Re-raster
   *  only when the freshly-snapped VP differs (the snap makes static/slow camera
   *  bit-identical → exact-equality gate, no epsilon). */
  lastVP: Matrix4;
  ran: boolean;
  count: number;
}

export interface ShadowClipParams {
  levels: number;
  base: number;
  minPx: number;
  res: number;
}

function readClipParams(): ShadowClipParams {
  const q = new URLSearchParams(window.location.search);
  const levels = Math.max(2, Math.min(10, Math.round(Number(q.get('shadowcliplevels') ?? 6))));
  const base = Math.max(2, Number(q.get('shadowclipbase') ?? 12));
  // per-level min-screen-size cull (projected RADIUS in shadow texels) — drops
  // sub-texel casters. Far clipmap levels are coarse, so a fixed world cluster
  // projects to FEW texels there ⇒ this bites hard on the far field (where the
  // 4-cascade fine LOD made it useless). 0 = isolate the hollow win. ?shadowclipminpx
  const minPx = Math.max(0, Number(q.get('shadowclipminpx') ?? 0));
  // per-level map resolution. Fill (texels rastered) scales T² and is the moving
  // raster's dominant cost; soft PCSS penumbra tolerates < 2048. ?shadowclipres
  const res = Math.max(256, Math.min(2048, Math.round(Number(q.get('shadowclipres') ?? 1024))));
  return { levels, base, minPx, res };
}

export function buildNaniteShadowClip(
  gpu: RegistryGpu,
  instanceCount: number,
  heightTex: Texture,
  disp?: TerrainDisp,
  wind?: TrunkWindOpt,
  /** item 6: the measured deepest DAG anchor-chain (registry.maxDagDepth + margin) — the
   *  BFS pass count for the SHARED cut (paid once/frame when a level re-rasters). Omitted ⇒
   *  the cull's legacy default. ?hierdepth still overrides. */
  hierDepth?: number,
): NaniteShadow {
  const cfg = readClipParams();
  const LEVELS = cfg.levels;
  const SHADOW_MAP = cfg.res;
  const SHADOW_PIX = SHADOW_MAP * SHADOW_MAP;

  // ONE shared vis buffer: raster level k → copy to depthTex_k → reuse for k+1.
  const vis: NaniteVisBuffers = makeVisBuffers(SHADOW_PIX);

  const levels: Level[] = [];
  const levelVP: UniformMat4[] = [];
  // per-level (span_m, depthRange_m, texel, radius) for the world-metric PCSS
  const levelParam: UniformArrV4 = uniformArrV4(
    Array.from({ length: LEVELS }, () => new Vector4(1, 1, 1 / SHADOW_MAP, 1.15)),
  );

  // shared min-screen-size cull uniform (all levels share cfg.minPx) + the hier
  // SHARED-cut cam (S3-perf): the cut is identical across levels, so ONE cull walks
  // the DAG and per-level FILTERS apply each level's frustum + hollow (NaniteClipCull).
  const minPxU: UniformF = uniformF(cfg.minPx);
  const cutCam = makeNaniteCam(SHADOW_MAP, SHADOW_MAP);
  const levelCams: NaniteCam[] = [];
  const innerRejects: UniformF[] = [];

  // pass 1: cams, hollow uniforms, depth textures, copy kernels, orthos (no cull/raster)
  for (let k = 0; k < LEVELS; k++) {
    const half = cfg.base * 2 ** k;
    const cam = makeNaniteCam(SHADOW_MAP, SHADOW_MAP);
    // hollow: levels ≥1 reject clusters fully inside the next-finer box. The
    // uniform carries 1/E_k (radius→clip). Level 0 is the innermost ⇒ no hollow.
    const innerReject: UniformF = uniformF(k === 0 ? 0 : 1 / half);
    levelCams.push(cam);
    innerRejects.push(innerReject);

    const depthTex = new StorageTexture(SHADOW_MAP, SHADOW_MAP);
    depthTex.type = FloatType;
    depthTex.format = RedFormat;
    depthTex.magFilter = NearestFilter;
    depthTex.minFilter = NearestFilter;
    depthTex.generateMipmaps = false;
    depthTex.name = `nanClipDepth${k}`;

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
    (kCopy as unknown as NamedKernel).setName(`nanClipCopy${k}`);

    const ortho = new OrthographicCamera(-half, half, half, -half, 0, 1);
    // CRITICAL: a standalone three camera defaults to WebGLCoordinateSystem
    // (z∈[-1,1]); the raster + the resolve sample expect the WebGPU NDC (z∈[0,1])
    // the engine camera and the CSM cascades use. Without this the stored depth
    // and the sample-side z disagree → no shadows / inverted depth.
    ortho.coordinateSystem = WebGPUCoordinateSystem;

    levelVP.push(uniformMat4(new Matrix4()));
    levels.push({
      cam,
      innerReject,
      raster: null as unknown as NaniteRasterHandles,
      depthTex,
      kCopy,
      half,
      ortho,
      lastVP: new Matrix4(),
      ran: false,
      count: -1,
    });
  }

  // ONE shared cross-level cull (S3-perf): the cut is identical across levels, so a
  // single hier traverse + cheap per-level filters serve every level. Each level's
  // raster consumes the shared ClipCull queue (refilled per level by the filter;
  // levels raster sequentially over the one vis buffer).
  const clipCull: ClipCull = buildClipCull(gpu, instanceCount, cutCam, levelCams, innerRejects, {
    minPx: minPxU,
    // the cut is bounded by the OUTER ring (~384 m), far below the camera's 8M
    // far-field flood ⇒ a small frontier (16 MB/buf vs 64 MB at 8M) is ample.
    frontierCap: SHADOW_FRONTIER_CAP,
    hierDepth,
  });
  for (let k = 0; k < LEVELS; k++) {
    const lv = levels[k]!;
    lv.raster = buildNaniteRaster(gpu, heightTex, lv.cam, clipCull.queue, vis, 'flat', false, disp, wind);
  }

  // ---- per-frame clipmap fit + raster -----------------------------------------
  const forward = new Vector3();
  const right = new Vector3();
  const up = new Vector3();
  const center = new Vector3();
  const eye = new Vector3();
  const worldUpY = new Vector3(0, 1, 0);
  const worldUpZ = new Vector3(0, 0, 1);
  const vp = new Matrix4();
  const frustum = new Frustum();
  // S3-perf: the shared cut's ortho — covers the OUTER ring every frame (unsnapped,
  // centred on the camera) so the one traverse spans all levels. Extents are constant
  // (outer half is fixed); only the pose + far (sun elevation) change per frame.
  const cutOrtho = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
  cutOrtho.coordinateSystem = WebGPUCoordinateSystem;
  const reRaster: boolean[] = new Array(LEVELS).fill(false);
  let lastRasterMask = 0;
  // item 4 (CLIP camera||shadow overlap): when cullPrepass() ran the CPU fit + folded the
  // shared cut into the camera-cull submit this frame, it stores the mask here so the
  // following run() skips the re-fit + the cut dispatch (already done) and only runs the
  // per-level filter+raster. -1 = no prepass this frame (run() does the full thing itself).
  let prepassMask = -1;

  // PASS A (CPU, no GPU) — fit every level (snap + R1 cadence gate) → reRaster[] + mask.
  // For a re-rastering level set its cam (filter frustum) + resolve uniforms + lastVP. A
  // CACHED level keeps its old VP + levelVP so its depthTex sample stays aligned.
  const fitLevels = (mainCamera: PerspectiveCamera): { mask: number; sinElev: number } => {
    // sun "L" points surface→sun; the shadow view looks the other way.
    forward.copy(sunU.dir.value).normalize().multiplyScalar(-1);
    const sinElev = Math.max(0.12, sunU.dir.value.y); // sun elevation (≈ -forward.y)
    // light-plane basis ⟂ forward (worldUp swap near the pole so the cross is
    // well-conditioned). These are three's lookAt grid axes (up to a sign flip on
    // right, which round-snapping ignores), so the texel snap aligns.
    const worldUp = Math.abs(forward.y) > 0.99 ? worldUpZ : worldUpY;
    right.crossVectors(worldUp, forward).normalize();
    up.crossVectors(forward, right).normalize();

    const cp = mainCamera.position;
    const cz = cp.dot(forward); // camera depth along the sun axis

    let mask = 0;
    for (let k = 0; k < LEVELS; k++) {
      const lv = levels[k]!;
      reRaster[k] = false;
      const texelWorld = (2 * lv.half) / SHADOW_MAP;
      // snap the centre onto THIS level's texel grid (anti-crawl + cadence)
      const cx = cp.dot(right);
      const cy = cp.dot(up);
      const scx = Math.round(cx / texelWorld) * texelWorld;
      const scy = Math.round(cy / texelWorld) * texelWorld;
      center
        .copy(right)
        .multiplyScalar(scx)
        .addScaledVector(up, scy)
        .addScaledVector(forward, cz);
      // depth half-span: generous along the sun (covers terrain relief + canopy +
      // the grazing reach across the level). f32 depth precision ⇒ a big range is
      // fine. Scales with the level so per-level precision is consistent.
      const dHalf = lv.half / sinElev + 100;
      eye.copy(center).addScaledVector(forward, -dHalf); // toward the sun
      lv.ortho.position.copy(eye);
      lv.ortho.up.copy(up);
      lv.ortho.lookAt(center);
      lv.ortho.near = 0;
      lv.ortho.far = 2 * dHalf;
      lv.ortho.updateMatrixWorld(true);
      lv.ortho.updateProjectionMatrix();
      vp.multiplyMatrices(lv.ortho.projectionMatrix, lv.ortho.matrixWorldInverse);

      // R1 cadence: skip if the snapped VP is bit-identical to the cached one.
      if (lv.ran && vp.equals(lv.lastVP)) continue;
      reRaster[k] = true;
      mask |= 1 << k;

      const cam = lv.cam;
      cam.vp.value.copy(vp);
      cam.prevVp.value.copy(vp);
      cam.camPos.value.copy(cp); // LOD by the MAIN camera (match the lit surface)
      cam.prevCamPos.value.copy(cp);
      // frustum planes from the VP (the cull/filter's frustumVisible reads cam.planes).
      // Mirrors NaniteShadow.ts / NaniteCommon exactly (default coordinateSystem
      // arg — the L/R/T/B planes are convention-independent and do the culling;
      // near/far slack is absorbed by the generous depth range, as in the cascades).
      frustum.setFromProjectionMatrix(vp);
      for (let p = 0; p < 6; p++) {
        const pl = frustum.planes[p];
        if (pl) cam.planes.array[p]?.set(pl.normal.x, pl.normal.y, pl.normal.z, pl.constant);
      }
      levelVP[k]!.value.copy(vp);
      (levelParam.array[k] as Vector4).set(2 * lv.half, 2 * dHalf, 1 / SHADOW_MAP, 1.15);
      lv.lastVP.copy(vp);
      lv.ran = true;
    }
    return { mask, sinElev };
  };

  // PASS B fit (CPU, no GPU) — set cutCam to span the re-rastering levels. Must be called
  // with mask != 0. Mirrors the cut-ortho fit from the original run().
  const fitCut = (mainCamera: PerspectiveCamera, sinElev: number): void => {
    const cp = mainCamera.position;
    const cz = cp.dot(forward);
    // size the cut to the LARGEST re-rastering level so slow drift (only near levels tick)
    // traverses a small region, not the whole 384 m disc.
    let maxHalf = 0;
    for (let k = 0; k < LEVELS; k++) if (reRaster[k]) maxHalf = Math.max(maxHalf, levels[k]!.half);
    const cutDHalf = maxHalf / sinElev + 100;
    const cutHalf = maxHalf + 2 * ((2 * maxHalf) / SHADOW_MAP);
    center
      .copy(right)
      .multiplyScalar(cp.dot(right))
      .addScaledVector(up, cp.dot(up))
      .addScaledVector(forward, cz);
    eye.copy(center).addScaledVector(forward, -cutDHalf);
    cutOrtho.left = -cutHalf;
    cutOrtho.right = cutHalf;
    cutOrtho.top = cutHalf;
    cutOrtho.bottom = -cutHalf;
    cutOrtho.far = 2 * cutDHalf;
    cutOrtho.position.copy(eye);
    cutOrtho.up.copy(up);
    cutOrtho.lookAt(center);
    cutOrtho.updateMatrixWorld(true);
    cutOrtho.updateProjectionMatrix();
    vp.multiplyMatrices(cutOrtho.projectionMatrix, cutOrtho.matrixWorldInverse);
    cutCam.vp.value.copy(vp);
    cutCam.prevVp.value.copy(vp);
    cutCam.camPos.value.copy(cp);
    cutCam.prevCamPos.value.copy(cp);
    frustum.setFromProjectionMatrix(vp);
    for (let p = 0; p < 6; p++) {
      const pl = frustum.planes[p];
      if (pl) cutCam.planes.array[p]?.set(pl.normal.x, pl.normal.y, pl.normal.z, pl.constant);
    }
  };

  // the per-level filter + raster (after the shared cut is on the GPU). Reused by run()'s
  // default path and the overlap path (where the cut was folded into the camera submit).
  const rasterLevels = (renderer: Renderer, mainCamera: PerspectiveCamera): void => {
    for (let k = 0; k < LEVELS; k++) {
      if (!reRaster[k]) continue;
      const lv = levels[k]!;
      clipCull.runLevelFilter(renderer, k); // cut → level-k frustum+hollow → queue
      lv.raster.clearVis(renderer);
      lv.raster.depth1(renderer);
      lv.raster.hwDepth(renderer, mainCamera);
      dispatch(renderer, lv.kCopy);
    }
  };

  // item 4 (camera||shadow overlap, opt-in): do the CPU fit NOW and RETURN the shadow-cut
  // cull batch (camera-disjoint) without dispatching, so the caller folds it into the
  // camera-cull submit. Stores prepassMask so the following run() skips its own cut.
  // Returns null when no level re-rasters (the caller then dispatches nothing extra).
  const cullPrepass = (
    _renderer: Renderer,
    mainCamera: PerspectiveCamera,
  ): readonly unknown[] | null => {
    const { mask, sinElev } = fitLevels(mainCamera);
    prepassMask = mask;
    if (mask === 0) return null;
    fitCut(mainCamera, sinElev);
    return clipCull.sharedCutBatch();
  };

  const run = (renderer: Renderer, _csm: object | null, mainCamera: PerspectiveCamera): void => {
    // OVERLAP PATH: cullPrepass() already fit + dispatched the shared cut (folded into the
    // camera-cull submit). Consume that mask, skip the re-fit + runSharedCut, raster levels.
    if (prepassMask >= 0) {
      const mask = prepassMask;
      prepassMask = -1; // consume (so a run() without a prepass next frame refits itself)
      if (mask !== 0) rasterLevels(renderer, mainCamera);
      lastRasterMask = mask;
      return;
    }
    // DEFAULT PATH (no prepass): fit + cut + raster in this submit-set, as before.
    const { mask, sinElev } = fitLevels(mainCamera);
    if (mask !== 0) {
      fitCut(mainCamera, sinElev);
      clipCull.runSharedCut(renderer);
      rasterLevels(renderer, mainCamera);
    }
    lastRasterMask = mask;
  };

  // ---- resolve-side PCSS over our own textures (level-select = finest cover) ---
  const depthAt = (k: number, uv: NV2): NF => {
    const u = (uv as unknown as { clamp(a: number, b: number): NV2 }).clamp(0, 1);
    const tx = minU(uint((u as unknown as { x: NF }).x.mul(SHADOW_MAP)), uint(SHADOW_MAP - 1));
    const ty = minU(uint((u as unknown as { y: NF }).y.mul(SHADOW_MAP)), uint(SHADOW_MAP - 1));
    return (textureLoad(levels[k]!.depthTex, uvec2(tx, ty)) as unknown as { x: NF }).x;
  };

  const pcss = (k: number, uv: NV2, receiver: NF, pix: NV2): NF =>
    Fn(() => {
      const param = levelParam.element(int(k));
      const span = (param as unknown as { x: NF }).x.max(1);
      const depthRange = (param as unknown as { y: NF }).y.max(1);
      const texel = (param as unknown as { z: NF }).z;
      const radius = (param as unknown as { w: NF }).w.max(1);
      const phi = interleavedGradientNoise(pix).mul(TAU);
      const dBias = float(DEPTH_BIAS_M).div(depthRange);

      const searchR = texel.mul(6).mul(radius);
      const blockerSum = float(0).toVar();
      const blockerCount = float(0).toVar();
      for (let i = 0; i < BLOCKER_TAPS; i++) {
        const tap = vogelDiskSample(float(i), float(BLOCKER_TAPS), phi) as unknown as NV2;
        const uvT = (uv as unknown as { add(o: unknown): NV2 }).add(
          (tap as unknown as { mul(o: unknown): NV2 }).mul(searchR),
        );
        const d = depthAt(k, uvT);
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
          const lit = receiver.lessThanEqual(depthAt(k, uvT).add(dBias));
          sum.addAssign(lit.select(float(1), float(0)));
        }
        result.assign(sum.div(PCF_TAPS));
      });
      return result;
    })() as unknown as NF;

  const levelCoord = (k: number, wp: NV3): { uv: NV2; z: NF; inside: NB } => {
    const sc = levelVP[k]!.mul(vec4(wp, 1));
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
      const pc = (pix ?? (screenCoordinate.xy as unknown as NV2)) as NV2;
      const wp = (worldPos as unknown as { add(o: unknown): NV3 }).add(
        (normal as unknown as { mul(o: number): NV3 }).mul(NORMAL_BIAS_M),
      ).toVar();
      const sf = float(1).toVar();
      const found = float(0).toVar();
      for (let k = 0; k < LEVELS; k++) {
        If(found.equal(0), () => {
          const { uv, z, inside } = levelCoord(k, wp as unknown as NV3);
          If(inside, () => {
            found.assign(1);
            sf.assign(pcss(k, uv, z, pc));
          });
        });
      }
      return sf;
    })() as unknown as NF;

  // ?nandbg=shadowc — which clipmap level covers each pixel (cycling tint)
  const cascadeTint = (worldPos: NV3): NV3 =>
    Fn(() => {
      const tints = [
        vec3(1, 0, 0),
        vec3(0, 1, 0),
        vec3(0, 0, 1),
        vec3(1, 1, 0),
        vec3(1, 0, 1),
        vec3(0, 1, 1),
      ];
      const col = vec3(0).toVar();
      const found = float(0).toVar();
      for (let k = 0; k < LEVELS; k++) {
        If(found.equal(0), () => {
          const { inside } = levelCoord(k, worldPos);
          If(inside, () => {
            found.assign(1);
            col.assign(tints[k % tints.length]!);
          });
        });
      }
      return col;
    })() as unknown as NV3;

  const debugDepth = (worldPos: NV3): NF =>
    Fn(() => {
      const d = float(1).toVar();
      const found = float(0).toVar();
      for (let k = 0; k < LEVELS; k++) {
        If(found.equal(0), () => {
          const { uv, inside } = levelCoord(k, worldPos);
          If(inside, () => {
            found.assign(1);
            d.assign(depthAt(k, uv));
          });
        });
      }
      return d;
    })() as unknown as NF;

  const readCounts = async (renderer: Renderer): Promise<number[]> => {
    // per-level survivor counts from the shared cut's filters
    const { perLevel } = await clipCull.readCounts(renderer);
    return levels.map((lv, k) => {
      if (!lv.ran) return -1;
      lv.count = perLevel[k] ?? 0;
      return lv.count;
    });
  };

  const rasteredMask = (): number => lastRasterMask;

  return {
    run,
    cullPrepass,
    shadowFactor,
    cascadeTint,
    debugDepth,
    readCounts,
    rasteredMask,
    cascades: LEVELS,
  };
}
