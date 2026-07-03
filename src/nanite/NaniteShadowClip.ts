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
  atomicMin,
  dot,
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
import type { NB, NF, NU, NV2, NV3, NV4 } from '../gpu/TSLTypes';
import { CLUSTER_WORDS, MESH_WORDS, readCluster } from './GeometryRegistry';
import type { RegistryGpu } from './GeometryRegistry';
import {
  DISPATCH_ROW,
  QRASTER_CAP,
  instSphereRadius,
  instTransformPoint,
  instYaw,
  makeNaniteCam,
  type NaniteCam,
} from './NaniteCommon';
import { buildClipCull, type ClipCull } from './NaniteClipCull';
import type { TerrainDisp, TrunkWindOpt } from './NaniteFetch';
import {
  buildNaniteRaster,
  makeVisBuffers,
  type NaniteRasterHandles,
  type NaniteVisBuffers,
} from './NaniteRaster';
import { BRICK_HALF, BRICK_POS_X, BRICK_WORDS } from './VoxelBrick';
import {
  bcF2U,
  bcU2F,
  dispatch,
  dispatchIndirect,
  elemU,
  localX,
  loopU,
  minU,
  returnIf,
  toF,
  uniformArrV4,
  uniformF,
  uniformMat4,
  wgLinear,
} from './Tsl';
import type { UniformArrV4, UniformF, UniformMat4 } from './Tsl';
import { sunU } from '../render/VegMaterials';
import type { NaniteShadow } from './NaniteShadow';

// P5 TOROIDAL CLIPMAP (shadow arc 2026-07-03): depth is stored as GLOBAL-normalized
// sun-axis distance z_g = (dot(p, fwd) + D_OFF)/D_RANGE — a texel's value is a pure
// world property (independent of the camera), so a level's stored content survives
// ANY camera translation. On a snap shift the level rasters ONLY the newly-exposed
// texel strips (outer leading edges + the hollow-reveal trailing edges) into the
// shared vis buffer, and a strip-scoped kCopy publishes them into the PERSISTENT
// per-level texture at toroidally-wrapped addresses. Full re-rasters remain only for
// sun-direction changes. Math note: PCSS is EXACTLY equivalent — the blocker gap in
// metres is (receiver−blocker)·D_RANGE, identical to the old per-level depthRange
// formulation; the bias constants are metres either way. Precision: f32 over 16 km
// ≈ 1 mm steps ≪ the 0.35 m depth bias. ?shtoro=0 = legacy full-level re-raster on
// VP change (the strips become one full-window rect, gate = VP equality).
const D_OFF = 8192;
const D_RANGE = 16384;

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
  /** P5: snapped centre in integer texel units (light-plane axes) last frame */
  prevSx: number;
  prevSy: number;
  /** P5: toroidal origin (texels) — window texel (0,0) lives at texture
   *  ((originX)%RES, (originY)%RES); advanced by the snap delta each shift. */
  originX: number;
  originY: number;
  /** P7: fitLevels frame of this level's last strip update (staleness pick) */
  lastStrip: number;
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
  /** P3 vox-shadow-splat (shadow arc 2026-07-03, ARC GOAL): when the registry holds
   *  voxel bricks, splat class-7 clusters' bricks into the shadow depth so the
   *  60-280 m voxel band + fartiles CAST shadows (they cast NOTHING before — the
   *  tri depth raster discards class-7 per-thread). ?shvox=0 disables. */
  voxSplat?: boolean,
): NaniteShadow {
  const cfg = readClipParams();
  const LEVELS = cfg.levels;
  const SHADOW_MAP = cfg.res;
  const SHADOW_PIX = SHADOW_MAP * SHADOW_MAP;
  // P5: ?shtoro=0 — legacy full-level re-raster on VP change (A/B escape)
  const qs = new URLSearchParams(window.location.search);
  const toro = qs.get('shtoro') !== '0';
  // P8 acne fix (user report: terrain self-shadowing from mid distances, worse
  // further out, gone where shadows end): the fixed world-space biases (0.35 m
  // depth / 0.12 m normal) are correct for L0's 0.023 m texels but FAR too small
  // for the coarse levels (texel_5 = 0.75 m) — one texel of a steep grazing slope
  // spans metres of depth ⇒ blocky self-shadow acne exactly in the 50-380 m band
  // (attribution: nanshadow=0 shows clean rock). Scale BOTH biases per level by
  // the texel size: normal-offset (slope-aware, moves the receiver off the
  // surface) + depth bias. ?shnb / ?shdb tune the per-texel factors.
  const nbTexelK = Number(qs.get('shnb') ?? 1.5) || 0;
  const dbTexelK = Number(qs.get('shdb') ?? 1.0) || 0;

  // ONE shared vis buffer: raster level k → copy to depthTex_k → reuse for k+1.
  const vis: NaniteVisBuffers = makeVisBuffers(SHADOW_PIX);

  const levels: Level[] = [];
  const levelVP: UniformMat4[] = [];
  // per-level (span_m, depthRange_m, texel, radius) for the world-metric PCSS.
  // P5: depthRange is the GLOBAL D_RANGE (the metres-per-z_g factor).
  const levelParam: UniformArrV4 = uniformArrV4(
    Array.from({ length: LEVELS }, () => new Vector4(1, D_RANGE, 1 / SHADOW_MAP, 1.15)),
  );
  // P5 per-level (originX texels, originY texels, zScale, zBias): toroidal origin for
  // the wrap addressing + the affine level-z → global-z remap the kCopy applies
  // (z_g = z·zScale + zBias with zScale = 2·dHalf/D_RANGE, zBias = (scz − dHalf +
  // D_OFF)/D_RANGE — monotone-affine, so the raster's atomicMin election is unchanged).
  const levelOrigin: UniformArrV4 = uniformArrV4(
    Array.from({ length: LEVELS }, () => new Vector4(0, 0, 1, 0)),
  );
  // P5 per-level active strip rects (4 × [x0,y0,x1,y1] in window UV; empty = x1<=x0):
  // shared by the cull filter (drop clusters missing every rect) and kCopy (publish
  // only strip texels). A full update = one (0,0,1,1) rect.
  const strips: UniformArrV4[] = Array.from({ length: LEVELS }, () =>
    uniformArrV4(Array.from({ length: 4 }, () => new Vector4(0, 0, 0, 0))),
  );
  const setStripRects = (k: number, rects: [number, number, number, number][]): void => {
    const u = strips[k]!;
    for (let r = 0; r < 4; r++) {
      const v = u.array[r] as Vector4;
      const rect = rects[r];
      if (rect) v.set(rect[0], rect[1], rect[2], rect[3]);
      else v.set(0, 0, 0, 0);
    }
  };

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

    // P5 strip-scoped publish: window texel → (in an active rect?) → remap the level
    // z to GLOBAL z_g → store at the toroidally-wrapped texture address. Empty vis
    // texels inside a rect store 1 (far) — a strip with no caster must still
    // overwrite the stale world content that scrolled out.
    const kCopy = Fn(() => {
      const px = instanceIndex;
      If(px.lessThan(uint(SHADOW_PIX)), () => {
        const x = px.mod(uint(SHADOW_MAP));
        const y = px.div(uint(SHADOW_MAP));
        const ux = toF(x).add(0.5).div(SHADOW_MAP);
        const uy = toF(y).add(0.5).div(SHADOW_MAP);
        const rectsU = strips[k]!;
        const hit = float(0).toVar();
        for (let r = 0; r < 4; r++) {
          const rect = rectsU.element(int(r));
          If(
            ux
              .greaterThanEqual(rect.x)
              .and(ux.lessThan(rect.z))
              .and(uy.greaterThanEqual(rect.y))
              .and(uy.lessThan(rect.w)),
            () => {
              hit.assign(1);
            },
          );
        }
        If(hit.greaterThan(0.5), () => {
          const raw = elemU(vis.depthV.ro, px).toVar();
          const org = levelOrigin.element(int(k));
          const zg = raw
            .equal(uint(0xffffffff))
            .select(float(1), bcU2F(raw).mul(org.z).add(org.w));
          const tx = x.add(uint(org.x)).mod(uint(SHADOW_MAP));
          const ty = y.add(uint(org.y)).mod(uint(SHADOW_MAP));
          textureStore(depthTex, uvec2(tx, ty), vec4(zg, 0, 0, 1)).toWriteOnly();
        });
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
      prevSx: 0,
      prevSy: 0,
      originX: 0,
      originY: 0,
      lastStrip: 0,
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
    // P5: the per-level strip rects — the filter drops clusters missing every rect
    strips,
    levelHalves: levels.map((lv) => lv.half),
  });
  for (let k = 0; k < LEVELS; k++) {
    const lv = levels[k]!;
    lv.raster = buildNaniteRaster(gpu, heightTex, lv.cam, clipCull.queue, vis, 'flat', false, disp, wind);
  }

  // ---- P3 vox-shadow-splat (ARC GOAL: the mid-field casts) -----------------------
  // The shared cut emits class-7 (voxel/fartile) clusters into the level queues; the
  // tri depth raster discards them per-thread (NaniteRaster returnIf mcVox==7), so
  // the 60-280 m voxel band + all fartiles cast NOTHING. Per level: one workgroup
  // per queue item (the SAME indirect args the tri raster consumes), one lane per
  // brick — project the brick centre through the level ortho (w≡1, no near-plane
  // blowup class), take the SUN-FACING face depth (centre − worldHalf along z), and
  // atomicMin the few-texel footprint into the shared vis depth. kCopy then
  // publishes strip texels exactly like tri depth. Footprint is naturally ~1-3
  // texels (brick size and texel size both scale with distance); extent is capped
  // defensively. ?shvox=0 disables.
  const shVox =
    voxSplat === true && new URLSearchParams(window.location.search).get('shvox') !== '0';
  const voxSplatKernels: unknown[] = [];
  if (shVox) {
    for (let k = 0; k < LEVELS; k++) {
      const lv = levels[k]!;
      const texelWorld = (2 * lv.half) / SHADOW_MAP;
      const kSplat = Fn(() => {
        const itemIdx = wgLinear(DISPATCH_ROW).toVar();
        const qCount = minU(
          (clipCull.queue.qRasterRO.element(0) as unknown as { x: NU }).x,
          uint(QRASTER_CAP),
        );
        returnIf(itemIdx.greaterThanEqual(qCount));
        const item = clipCull.queue.qRasterRO.element(itemIdx.add(uint(1)));
        const instId = (item as unknown as { x: NU }).x.toVar();
        const ci = (item as unknown as { y: NU }).y.toVar();
        const c = readCluster(gpu.clusters, ci);
        const mcVox = elemU(gpu.meshes, c.meshId.mul(uint(MESH_WORDS)).add(uint(6)))
          .bitAnd(uint(0xff))
          .toVar();
        returnIf(mcVox.notEqual(uint(7)));
        const cBase = ci.mul(uint(CLUSTER_WORDS));
        const brickBase = elemU(gpu.clusters, cBase.add(uint(6))).toVar();
        const brickCount = elemU(gpu.clusters, cBase.add(uint(7))).bitAnd(uint(0xff)).toVar();
        const brickLocal = localX().toVar();
        returnIf(brickLocal.greaterThanEqual(brickCount));
        const A = gpu.instances.element(instId.mul(uint(2))).toVar() as unknown as NV4;
        const B = gpu.instances.element(instId.mul(uint(2)).add(uint(1))).toVar() as unknown as NV4;
        const yawSc = instYaw(B);
        const bw = brickBase.add(brickLocal).mul(uint(BRICK_WORDS));
        const brLocal = vec3(
          bcU2F(elemU(gpu.voxelBricks, bw.add(uint(BRICK_POS_X)))),
          bcU2F(elemU(gpu.voxelBricks, bw.add(uint(BRICK_POS_X + 1)))),
          bcU2F(elemU(gpu.voxelBricks, bw.add(uint(BRICK_POS_X + 2)))),
        ) as unknown as NV3;
        const brHalfL = bcU2F(elemU(gpu.voxelBricks, bw.add(uint(BRICK_HALF)))).toVar();
        const wc = instTransformPoint(A, B, yawSc, brLocal);
        const wHalf = (instSphereRadius(A, B, brHalfL as unknown as NF, float(0)) as unknown as NF).toVar();
        const clip = (lv.cam.vp.mul(vec4(wc, 1)) as unknown as NV4).toVar();
        // sun-facing face depth (clip z is metres/(2·dHalf) along the sun) — the
        // receiver-side DEPTH_BIAS_M covers residual brick-scale acne.
        const org = levelOrigin.element(int(k));
        const zNear = clip.z
          .sub(wHalf.div((org as unknown as { z: NF }).z.mul(D_RANGE)))
          .clamp(0, 1)
          .toVar();
        const bits = bcF2U(zNear as unknown as NF).toVar();
        // texel footprint around the projected centre, capped defensively
        const cx = (clip.x.mul(0.5).add(0.5) as unknown as NF).mul(SHADOW_MAP).toVar();
        const cy = (clip.y.mul(0.5).add(0.5) as unknown as NF).mul(SHADOW_MAP).toVar();
        const rpx = wHalf.div(texelWorld).clamp(0, 8).toVar();
        // off-window guard (filter passes whole clusters; edge bricks may poke out)
        returnIf(
          cx.add(rpx).lessThan(0).or(cx.sub(rpx).greaterThanEqual(SHADOW_MAP))
            .or(cy.add(rpx).lessThan(0))
            .or(cy.sub(rpx).greaterThanEqual(SHADOW_MAP)) as unknown as NB,
        );
        const x0 = uint(cx.sub(rpx).max(0)).toVar();
        const y0 = uint(cy.sub(rpx).max(0)).toVar();
        const x1 = minU(uint(cx.add(rpx).max(0)), uint(SHADOW_MAP - 1)).toVar();
        const y1 = minU(uint(cy.add(rpx).max(0)), uint(SHADOW_MAP - 1)).toVar();
        loopU(y0, y1.add(uint(1)), (ty) => {
          loopU(x0, x1.add(uint(1)), (tx) => {
            const px = ty.mul(uint(SHADOW_MAP)).add(tx);
            atomicMin(vis.depthV.atomic.element(px), bits);
          });
        });
      })().compute(DISPATCH_ROW * 128, [128]);
      (kSplat as unknown as NamedKernel).setName(`nanClipVoxSplat${k}`);
      voxSplatKernels.push(kSplat);
    }
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

  // P5: sun-direction change detection — the ONLY remaining full invalidation
  // (stored z_g is a world property; camera translation never invalidates texels).
  const lastSunFwd = new Vector3(0, 0, 0);
  // P7 coarse-strip budget (?shbudget=0 disables): frames where BOTH coarse levels
  // (k ≥ LEVELS−2) update strips carry the tail (measured med 23.6 ms vs 16.8 —
  // each coarse update sizes the shared cut to its 192/384 m disc). Allow at most
  // ONE coarse level per frame, picked by staleness (no starvation); a deferred
  // level stays FULLY frozen (VP/origin/prevS untouched) and its shift accumulates
  // into next frame's strips — deferral error is sub-texel at any speed that
  // didn't already force a full update. Sun-change full invalidates are exempt.
  const budgetOn = new URLSearchParams(window.location.search).get('shbudget') !== '0';
  const COARSE_K = Math.max(2, LEVELS - 2);
  let fitFrame = 0;

  // PASS A (CPU, no GPU) — fit every level → strip rects + reRaster[] + mask.
  // TOROIDAL (default): a level re-rasters ONLY its newly-exposed window strips
  // (outer leading edges from the snap shift + the hollow-reveal trailing edges
  // around the central hole); dx=dy=0 ⇒ fully cached, ANY camera motion along the
  // sun axis included (stored depth is global — cz affects only the raster slab).
  // LEGACY (?shtoro=0): full-window rect on any VP change (the old R1 gate).
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

    // sun moved ⇒ the light basis (and every stored z_g) is stale ⇒ full re-raster
    const sunMoved = forward.distanceToSquared(lastSunFwd) > 1e-12;
    if (sunMoved) lastSunFwd.copy(forward);

    // P7 pre-pass: pick the ONE coarse level allowed to update strips this frame —
    // the stalest of those with a pending shift (round-robins under load, so
    // neither coarse level starves; a starved level's shift just accumulates).
    fitFrame++;
    let coarsePick = -1;
    if (toro && budgetOn && !sunMoved) {
      let bestAge = -1;
      for (let k = COARSE_K; k < LEVELS; k++) {
        const lv = levels[k]!;
        if (!lv.ran) continue; // full path is exempt from the budget
        const texelK = (2 * lv.half) / SHADOW_MAP;
        const dx = Math.round(cp.dot(right) / texelK) - lv.prevSx;
        const dy = Math.round(cp.dot(up) / texelK) - lv.prevSy;
        if (dx === 0 && dy === 0) continue;
        const age = fitFrame - lv.lastStrip;
        if (age > bestAge) {
          bestAge = age;
          coarsePick = k;
        }
      }
    }

    let mask = 0;
    for (let k = 0; k < LEVELS; k++) {
      const lv = levels[k]!;
      reRaster[k] = false;
      const texelWorld = (2 * lv.half) / SHADOW_MAP;
      // snap the centre onto THIS level's texel grid (anti-crawl + toroidal shift).
      // ALL THREE axes: scx/scy is the classic anti-crawl; the SUN-AXIS depth snaps
      // too so the raster slab is stable between texel crossings (P1; an unsnapped
      // cz invalidated the old VP-equality cache on ANY motion — measured 598/600
      // frames re-rastering ALL 6 levels, the whole +17.6 ms moving shadow bill).
      const cx = cp.dot(right);
      const cy = cp.dot(up);
      const sx = Math.round(cx / texelWorld); // integer texel units
      const sy = Math.round(cy / texelWorld);
      const scx = sx * texelWorld;
      const scy = sy * texelWorld;
      const scz = Math.round(cz / texelWorld) * texelWorld;
      center
        .copy(right)
        .multiplyScalar(scx)
        .addScaledVector(up, scy)
        .addScaledVector(forward, scz);
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

      // ---- strip decision -------------------------------------------------------
      const R = SHADOW_MAP;
      let rects: [number, number, number, number][] | null = null;
      let full = false;
      if (!toro) {
        // legacy: full-window re-raster on any VP change (R1 exact-equality gate)
        if (!lv.ran || !vp.equals(lv.lastVP)) full = true;
      } else if (!lv.ran || sunMoved) {
        full = true;
      } else {
        const dx = sx - lv.prevSx;
        const dy = sy - lv.prevSy;
        // P7 coarse budget: a deferred coarse level stays FULLY frozen this frame —
        // no prevS/origin/VP/uniform mutation, so its stored window keeps sampling
        // correctly and the pending shift accumulates into next frame's strips.
        if (
          budgetOn &&
          k >= COARSE_K &&
          (dx !== 0 || dy !== 0) &&
          k !== coarsePick
        ) {
          continue;
        }
        if (dx !== 0 || dy !== 0) {
          if (k >= COARSE_K) lv.lastStrip = fitFrame;
          if (Math.abs(dx) >= R || Math.abs(dy) >= R) {
            full = true; // teleport — nothing survives
          } else {
            // CONTENT shift in WINDOW texels. three's lookAt makes the ortho x-axis
            // −right (xAxis = up×(−fwd) = −right) and y-axis +up — verified
            // empirically: ndc(center+10·right).x < 0. Camera +dx along `right`
            // ⇒ static world content moves +dx in uv.x; camera +dy along `up` ⇒
            // content moves −dy in uv.y.
            const sxShift = dx;
            const syShift = -dy;
            const ax = Math.abs(sxShift);
            const ay = Math.abs(syShift);
            // toroidal origin: address (w + origin) mod R stays fixed for a world
            // texel ⇒ origin advances OPPOSITE the content shift.
            lv.originX = ((lv.originX - sxShift) % R + R) % R;
            lv.originY = ((lv.originY - syShift) % R + R) % R;
            rects = [];
            // outer exposure strips — new world enters on the side content moves
            // AWAY from (shift>0 ⇒ low edge; shift<0 ⇒ high edge).
            if (sxShift > 0) rects.push([0, 0, ax / R, 1]);
            else if (sxShift < 0) rects.push([(R - ax) / R, 0, 1, 1]);
            if (syShift > 0) rects.push([0, 0, 1, ay / R]);
            else if (syShift < 0) rects.push([0, (R - ay) / R, 1, 1]);
            // hollow-reveal strips (k≥1: world content exiting the central hole was
            // never rastered at THIS level — the finer level owned it). Content
            // exits the hole [0.25,0.75)² on the side it moves TOWARD, padded by
            // the cross-axis delta (overlap is harmless).
            if (k > 0) {
              const py0 = Math.max(0, 0.25 - ay / R);
              const py1 = Math.min(1, 0.75 + ay / R);
              if (sxShift > 0) rects.push([0.75, py0, 0.75 + ax / R, py1]);
              else if (sxShift < 0) rects.push([0.25 - ax / R, py0, 0.25, py1]);
              const px0 = Math.max(0, 0.25 - ax / R);
              const px1 = Math.min(1, 0.75 + ax / R);
              if (syShift > 0) rects.push([px0, 0.75, px1, 0.75 + ay / R]);
              else if (syShift < 0) rects.push([px0, 0.25 - ay / R, px1, 0.25]);
            }
          }
        }
      }
      lv.prevSx = sx;
      lv.prevSy = sy;
      if (full) {
        lv.originX = 0;
        lv.originY = 0;
        rects = [[0, 0, 1, 1]];
      }
      // sampling must track the CURRENT window every frame (uv mapping follows the
      // snap even on cached frames; stored z_g is window-independent).
      levelVP[k]!.value.copy(vp);
      (levelParam.array[k] as Vector4).set(2 * lv.half, D_RANGE, 1 / SHADOW_MAP, 1.15);
      const org = levelOrigin.array[k] as Vector4;
      org.set(lv.originX, lv.originY, (2 * dHalf) / D_RANGE, (scz - dHalf + D_OFF) / D_RANGE);
      lv.lastVP.copy(vp);
      lv.ran = true;

      if (!rects || rects.length === 0) continue; // cached — no strips this frame
      if (rects.length > 4) rects.length = 4; // 4 slots (outer 2 + reveal 2)
      setStripRects(k, rects);
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
      clipCull.runLevelFilter(renderer, k); // cut → level-k frustum+hollow+strips → queue
      lv.raster.clearVis(renderer);
      lv.raster.depth1(renderer);
      lv.raster.hwDepth(renderer, mainCamera);
      // P3: brick depth splat for the class-7 clusters the queue carries (same
      // one-wg-per-item indirect args the tri raster consumes)
      if (shVox) dispatchIndirect(renderer, voxSplatKernels[k] as never, clipCull.queue.rasterDispatchAttr);
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
  // P5: window uv → toroidal texture address (clamp to the window FIRST — an
  // out-of-window tap must clamp to the window edge, never wrap to the far side).
  const depthAt = (k: number, uv: NV2): NF => {
    const u = (uv as unknown as { clamp(a: number, b: number): NV2 }).clamp(0, 1);
    const wx = minU(uint((u as unknown as { x: NF }).x.mul(SHADOW_MAP)), uint(SHADOW_MAP - 1));
    const wy = minU(uint((u as unknown as { y: NF }).y.mul(SHADOW_MAP)), uint(SHADOW_MAP - 1));
    const org = levelOrigin.element(int(k));
    const tx = wx.add(uint(org.x)).mod(uint(SHADOW_MAP));
    const ty = wy.add(uint(org.y)).mod(uint(SHADOW_MAP));
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
      // P8: depth bias scales with THIS level's world texel (coarse texels span
      // metres of slope depth — the fixed 0.35 m was L0-only thinking)
      const texelWorldK = (2 * levels[k]!.half) / SHADOW_MAP;
      const dBias = float(DEPTH_BIAS_M + texelWorldK * dbTexelK).div(depthRange);

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
      // P5: the receiver depth is GLOBAL z_g = (dot(p, fwd) + D_OFF)/D_RANGE —
      // computed from the sun uniform (fwd = −sunDir), compared against the
      // stored global texel values. levelCoord's z stays the per-window slab
      // coordinate and is used only for the inside test.
      const fwdN = (normalize(vec3(sunU.dir)) as unknown as { mul(o: number): NV3 }).mul(-1);
      const sf = float(1).toVar();
      const found = float(0).toVar();
      for (let k = 0; k < LEVELS; k++) {
        // P8: normal-offset scaled by THIS level's texel — slope-aware acne fix
        // (moving the receiver off the surface by ~a texel is the standard cure;
        // depth-only bias can't cover a steep slope's span within one coarse texel).
        const texelWorldK = (2 * levels[k]!.half) / SHADOW_MAP;
        If(found.equal(0), () => {
          const wpK = (worldPos as unknown as { add(o: unknown): NV3 })
            .add(
              (normal as unknown as { mul(o: number): NV3 }).mul(
                NORMAL_BIAS_M + texelWorldK * nbTexelK,
              ),
            )
            .toVar();
          const { uv, inside } = levelCoord(k, wpK as unknown as NV3);
          If(inside, () => {
            found.assign(1);
            const zg = (dot(wpK as unknown as NV3, fwdN as unknown as NV3) as unknown as NF)
              .add(D_OFF)
              .div(D_RANGE);
            sf.assign(pcss(k, uv, zg as unknown as NF, pc));
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
