/**
 * N2 vis-buffer raster on REGISTRY buffers — the spike pipeline (SpikeRaster)
 * ported onto GeometryRegistry mega-buffers + the world heightfield texture,
 * consuming the N2 cull chain's work queue. Option C only (D-N5): pass 1
 * atomicMin(f32-depth-bits), pass 2 equality payload store; near-crossing and
 * oversized triangles route to the HW queue whose fragment stage writes the
 * SAME buffers (depthWrite off — one resolve, one convention).
 *
 * Payload = (workItemIdx << CLUSTER_TRI_BITS) | localTri — qRaster doubles as the
 * visible-cluster list, so the resolve recovers (instance, cluster) in one
 * indirection. CLUSTER_TRI_BITS = log2(MAX_CLUSTER_TRIS) (7 @128, 8 @256 — the
 * ?clustertris A/B); itemIdx has 23 bits (QRASTER_CAP=2^23) ⇒ 23+8 = 31/32 at the
 * 256 cap, 1 spare.
 *
 * Scanline core (N3a): FIXED-POINT integer edge functions — verts snapped to
 * a 1/256-px grid (8 subpixel bits, HW convention), coverage + top-left rule
 * in exact i32 math (watertight; replaces the example's float −1e-5 bias).
 * Flat resolve = matClass palette or cluster-hash tint (the deferred N1
 * checkpoint), face-normal lambert, real f32 depth out.
 */

import { DoubleSide, Mesh, Scene, Vector3 } from 'three';
import {
  BufferGeometry,
  Float32BufferAttribute,
  RenderTarget,
  Sphere,
} from 'three';
import type { PerspectiveCamera, Texture } from 'three';
import {
  IndirectStorageBufferAttribute,
  NodeMaterial,
  StorageBufferAttribute,
  type Renderer,
  type StorageBufferNode,
} from 'three/webgpu';
import {
  Discard,
  Fn,
  If,
  atomicAdd,
  atomicMax,
  atomicMin,
  atomicStore,
  cross,
  dot,
  float,
  instanceIndex,
  max,
  normalize,
  positionGeometry,
  screenCoordinate,
  uint,
  varyingProperty,
  vec2,
  vec3,
  vec4,
  vertexIndex,
  workgroupArray,
  workgroupBarrier,
} from 'three/tsl';
import type { NB, NF, NI, NU, NV2, NV3, NV4 } from '../gpu/TSLTypes';
import { markFragmentWritable } from '../render/ThreePatches';
import {
  CLUSTER_TRI_BITS,
  CLUSTER_TRI_MASK,
  CLUSTER_WORDS,
  MAX_CLUSTER_TRIS,
  MESH_WORDS,
} from './GeometryRegistry';
import type { RegistryGpu } from './GeometryRegistry';
import {
  DISPATCH_ROW,
  QRASTER_CAP,
  hashColor,
  type NaniteCam,
} from './NaniteCommon';
import {
  makeFetch,
  type TerrainDisp,
  type TrunkWindOpt,
  type VertCtx,
} from './NaniteFetch';
import { clusterHwClass } from './NaniteHwClass';
import { makeVertexCache } from './NaniteVertexCache';
import {
  buildNaniteVoxelRaster,
  type VoxelRasterHandles,
} from './NaniteVoxelRaster';
import {
  aLoadU,
  bcF2U,
  bcU2F,
  bcU2I,
  dispatch,
  dispatchBatchMixed,
  dispatchIndirect,
  elemU,
  localX,
  loopI,
  loopU,
  loopUN,
  maxI,
  maxU,
  minI,
  minU,
  readBuffer,
  returnIf,
  sU32Views,
  setIndirectDispatch,
  toF,
  toI,
  uniformF,
  wgLinear,
} from './Tsl';
import type { BufOf, UV2 } from './Tsl';

// N9-C0: leaf needles are long, thin, near-camera tris — their bbox exceeds the
// SW raster's 16 px i32-safe limit, so a dense crown routes hundreds of thousands
// of them to the HW vertex-pulling path. The old 262k cap overflowed (clamp →
// dropped tris → black holes in foliage; measured 876k hwTris in a dense stand).
// Sized for that load; the queue only costs memory + the HW pass only pays for the
// tris actually present. The real reduction (halving the dup, shedding far crowns)
// is the two-sided raster + the aggregate DAG (N9-C2).
const HW_CAP = 2_097_152;
// ?swmax=N shrinks the SW/HW split (bbox extent ≤N px stays SW; larger → HW).
// Diagnostic: the relect ablation proved the SW loop cost is intrinsic
// per-covered-pixel walk+interp, so the last untested lever is letting the HW
// rasterizer eat the mid-size (5-16px) tris. Default 16 = shipped behavior.
const MAX_RASTER_SIZE = (() => {
  const v = Number(
    new URLSearchParams(window.location.search).get('swmax') ?? '16',
  );
  return Number.isFinite(v) && v >= 2 ? Math.floor(v) : 16;
})();
// SWCOOP small-bin threshold: clamped bbox EXTENT ≤4 px both axes (≤5×5=25 candidate
// pixels) stays lane-private; larger (up to MAX_RASTER_SIZE) goes cooperative.
const SW_SMALL_EXT = 4;
const NEAR_EPS = 1e-4;

/**
 * N9-C2 two-sided raster orientation (D-N43 Stage 0). The integer scanline core
 * below is built for POSITIVE-area (CCW) triangles — `area2`, `rcpArea`, the edge
 * walk and the top-left coverage rule all assume it. A single-sided cluster keeps
 * the classic back-face cull (accept front-faces, areaNdc > 0). A TWO-SIDED cluster
 * (leaf crowns) instead RE-WINDS a back-face to CCW in place — swap v1↔v2, which
 * negates the signed area (edgeFn = cross(v1−v0, v2−v0)) — so the SAME core rasters
 * whichever side faces the camera, exactly once. This replaces the leaf geometry's
 * reversed-winding triangle duplicate: half the leaf triangles, half the leaf
 * clusters, identical pixels (the resolve flips the leaf normal camera-ward, so
 * shading is unaffected by which winding rastered). `ndc1`/`ndc2` are the caller's
 * toVar()'d apex-relative corners, mutated in place; ndc0 (the shared apex) is
 * unchanged. Returns the accept gate — a degenerate area==0 still falls out at the
 * downstream integer `area2 > 0` test.
 */
export function orientForRaster(
  ndc1: NV3,
  ndc2: NV3,
  areaNdc: NF,
  twoSided: NB,
): NB {
  const flip = twoSided.and(areaNdc.lessThan(0)).toVar();
  const keep1 = vec3(ndc1).toVar(); // snapshot v1 before the in-place swap
  ndc1.assign(flip.select(ndc2, ndc1));
  ndc2.assign(flip.select(keep1, ndc2));
  return twoSided.select(
    areaNdc.notEqual(0),
    areaNdc.greaterThan(0),
  ) as unknown as NB;
}

interface ComputeKernel {
  setName(name: string): unknown;
}

export interface NaniteRasterHandles {
  /** fullscreen resolve mesh in its own scene — render with the main camera */
  resolveScene: Scene;
  /** clear vis buffers + hw queue (every frame, before any pass) */
  clearVis(renderer: Renderer): void;
  /** SW depth-only pass (shadow cascades/clipmap; atomicMin Z) */
  depth1(renderer: Renderer): void;
  /** P9 (shadow submit coalescing): the depth-only kernel pre-tagged with its
   *  indirect args, as a batch element for dispatchBatchMixed. */
  depth1Batch(): readonly unknown[];
  /** P9: the hw-queue/audit counter reset as a batch element — REQUIRED before
   *  every depth raster when the full kVisClear is replaced by a scoped clear. */
  hwQueueClearBatch(): readonly unknown[];
  /** HW big/near-tri depth render (re-runs hwArgs; full-queue redraw is
   *  idempotent atomicMin) */
  hwDepth(renderer: Renderer, camera: PerspectiveCamera): void;
  /** NaniteView debug single-pass packed Z+id (idLo/idHi split) — SW + HW in one go */
  combined(renderer: Renderer, camera: PerspectiveCamera): void;
  /** PERF-VB4 WORLD single pass: 24-bit Z election + full-id side buffer (visBV).
   *  hzbTail (SUBMIT-COALESCE ?coalesce=1): the HZB kernel chain to fold into the
   *  voxel raster's submit (or its own single submit when no voxRaster). */
  world1(
    renderer: Renderer,
    camera: PerspectiveCamera,
    hzbTail?: readonly unknown[],
  ): void;
  readHwCount(renderer: Renderer): Promise<number>;
  /** count covered/orphan pixels (NaniteView ?audit=1) */
  audit(renderer: Renderer): void;
  readAudit(renderer: Renderer): Promise<{ orphans: number; covered: number }>;
  /** 0a SCAR calibration (?scar=1): run the per-pixel covered-pixel counter for the
   *  mid/far foliage band AFTER world1. No-op unless ?scar=1. (The per-fragment band
   *  + total counters are accumulated inside the world1 raster itself.) */
  scar(renderer: Renderer): void;
  /** read the SCAR band counters back: band fragments (numerator), band covered
   *  pixels (denominator), total frame fragments, band cluster-emit count. */
  readScar(
    renderer: Renderer,
  ): Promise<{
    bandFrags: number;
    bandPx: number;
    totalFrags: number;
    bandClusters: number;
  }>;
  /** Stage-2 voxel BRICK-WRITE count (the occlusion-skip overlay number, §A2) — total
   *  per-pixel wgElect wins across all tiles this frame. null when no voxel raster. */
  readVoxWrites(renderer: Renderer): Promise<number | null>;
  /** W2 ?trihzb: the pyramid→hwQueue-tail mirror kernel, to append to the HZB
   *  build batch (null when the flag is off). */
  triHzbCopyKernel: unknown | null;
}

/** Option C vis buffers — created OUTSIDE the raster so the HZB (which the
 *  cull consumes) can view the depth buffer without a builder cycle */
export interface NaniteVisBuffers {
  depthAttr: StorageBufferAttribute;
  depthV: ReturnType<typeof sU32Views>;
  payloadAttr: StorageBufferAttribute;
  payloadV: ReturnType<typeof sU32Views>;
  /** PACKED-mode 2nd id buffer (idHi). visA(payloadV) holds (dk16<<16|idLo16), visB
   *  holds (dk16<<16|idHi16); both atomicMax'd so the nearest fragment wins BOTH and
   *  the depth+id can't desync (the race-free fix for the payload "branch-through-trunk"). */
  visBAttr: StorageBufferAttribute;
  visBV: ReturnType<typeof sU32Views>;
}

export function makeVisBuffers(pixelCount: number): NaniteVisBuffers {
  const depthAttr = new StorageBufferAttribute(new Uint32Array(pixelCount), 1);
  depthAttr.name = 'nanVisDepth';
  markFragmentWritable(depthAttr);
  const payloadAttr = new StorageBufferAttribute(
    new Uint32Array(pixelCount),
    1,
  );
  payloadAttr.name = 'nanVisPayload';
  markFragmentWritable(payloadAttr);
  const visBAttr = new StorageBufferAttribute(new Uint32Array(pixelCount), 1);
  visBAttr.name = 'nanVisB';
  markFragmentWritable(visBAttr);
  return {
    depthAttr,
    depthV: sU32Views(depthAttr, pixelCount),
    payloadAttr,
    payloadV: sU32Views(payloadAttr, pixelCount),
    visBAttr,
    visBV: sU32Views(visBAttr, pixelCount),
  };
}

export function buildNaniteRaster(
  gpu: RegistryGpu,
  heightTex: Texture,
  cam: NaniteCam,
  cull: {
    qRasterRO: BufOf<UV2>;
    rasterDispatchAttr: IndirectStorageBufferAttribute;
    rasterDispatch2Attr: IndirectStorageBufferAttribute;
    rasterDispatchFullAttr: IndirectStorageBufferAttribute;
    /** voxel-foliage (Stage 2 §6): the voxel-raster work queue + its 2D-split dispatch
     *  args (cull §A1). Present only when ?voxreg/?forcevox is active; when present the
     *  scatter voxel-brick raster runs in world1() after hwRender. */
    qVoxRasterRO?: BufOf<UV2>;
    voxRasterDispatchAttr?: IndirectStorageBufferAttribute;
    /** DEPTH-BUCKET F2B (?voxf2b): per-bucket (base,count) + per-bucket indirect args +
     *  the build-time K + the on/off flag (cull.voxBucketRangeRO / voxBucketDispatchAttr /
     *  voxF2bK / voxF2bEnabled). Present whenever the voxel queue is. */
    voxBucketRangeRO?: BufOf<UV2>;
    voxBucketDispatchAttr?: IndirectStorageBufferAttribute[];
    voxF2bK?: number;
    voxF2bEnabled?: boolean;
    voxPrevEnabled?: boolean;
    /** ?clhw per-cluster SW/HW split (NaniteCull). qHwRasterRO = HW-cluster qRaster
     *  indices; hwClusterDrawAttr = the instanced draw args. Present (non-null) only
     *  when ?clhw is on; the SW world1 kernel skips these clusters + the instanced HW
     *  draw paints them properly. */
    clhwEnabled?: boolean;
    qHwRasterRO?: StorageBufferNode<'uint'> | null;
    hwClusterDrawAttr?: IndirectStorageBufferAttribute | null;
  },
  vis: NaniteVisBuffers,
  tint: 'flat' | 'cluster' | 'lod',
  /** false (?shade=0): pure matClass color, no lambert — the parity gate's
   *  shading-free mode (coverage/structure compare only) */
  shade = true,
  /** terrain micro-displacement (frame mode; dbg views omit — NaniteFetch) */
  disp?: TerrainDisp,
  /** trunk wind (frame mode) — MUST match the resolve's makeFetch so the
   *  rastered geometry and the resolve's barycentric corners agree */
  wind?: TrunkWindOpt,
  /** PACKED vis-buffer (race-free combined path): the combined kernels write the id
   *  split across visA(payloadV)+visB via depth-keyed atomicMax, and the resolve decodes
   *  it from those two. Only valid with the hier `combined()` pass. Default false keeps
   *  the legacy depthV/payloadV (two-pass + brute) encoding for the world/shadow paths. */
  packed = false,
  /** WORLD single-pass (?vb=single / PERF-VB4 D-N45): ONE SW+HW pass writes EXACT f32
   *  depth (visDepthV via atomicMin — so the HZB, shadows, and the resolve's wp
   *  reconstruction stay byte-identical + exact) AND a COHERENT id (depth16-keyed
   *  atomicMax election into visPayloadV, no branch-through-trunk) whose winner plain-
   *  stores the FULL 25-bit id into ONE side buffer (visBV). Drops the 2nd (payload)
   *  raster pass. DEVIATES from D-N45's "recompute depth in the resolve": keeping the
   *  exact depthV is cheaper (folds into pass-1's atomicMin) and leaves every depth
   *  consumer unchanged. The residual is the same artifact class — at a rare depth16
   *  tie the id can pick the bucket-neighbour (a valid wrong-material pixel, never a
   *  torn/garbage fetch). Differs from `packed` (NaniteView's 16-bit-depth combined). */
  singlePass = false,
  /** voxel-foliage (Stage 2 §6): build + dispatch the scatter voxel-brick
   *  raster. Requires cull.qVoxRasterRO + cull.voxRasterDispatchAttr. OFF by
   *  default so a pure-triangle world never binds the voxel permutation. */
  voxActive = false,
  /** procedural grass (NaniteGrass, grass rethink 2026-07-03): its cull+raster
   *  kernels ride world1's batched submit (after kVisClear so the elections land
   *  on cleared buffers, before kHwArgs — the queues are disjoint); its blade HW
   *  pass (own depth target, hardware early-z) runs right after hwRender so the
   *  depth prime sees the mesh SW+HW election. */
  grass?: {
    batch: readonly unknown[];
    renderHw(renderer: Renderer, camera: PerspectiveCamera): void;
    enabled(): boolean;
  },
  /** 90fps-arc W2 EXPERIMENT (?trihzb=1): per-TRIANGLE occlusion reject in the
   *  world1 SW path — the tri's nearest ndc.z vs the PREV-frame HZB farthest over
   *  its pixel bbox (2×2 window at a fixed 32-px-pitch level). Kills the whole
   *  scanline walk of triangles hidden behind the canopy wall that the CLUSTER
   *  cull can't reject (cluster bboxes poke through gaps). Same staleness class
   *  as the cluster occlusion cull (1-frame disocclusion holes possible — gate
   *  with stills before any default flip). +1 storage binding, world1-only,
   *  build-time gated. */
  triHzb?: {
    ro: import('./Tsl').BufOf<NF>;
    levels: { offset: number; w: number; h: number }[];
  },
): NaniteRasterHandles {
  const { width, height } = cam;
  // Mirror pyramid levels 0..TRI_HZB_MAX into the hwQueue tail (level k texel
  // pitch = 2^(k+1) full-res px); the per-tri test picks the FINEST level whose
  // 2×2 window covers the tri's bbox — fine levels carry ~all the rejection
  // power in porous foliage (a coarse window almost always contains a gap).
  const TRI_HZB_MAX = 4;
  const triLvls =
    new URLSearchParams(window.location.search).get('trihzb') === '1' &&
    triHzb != null &&
    triHzb.levels.length > TRI_HZB_MAX
      ? triHzb.levels.slice(0, TRI_HZB_MAX + 1)
      : null;
  // dest offsets within the tail (source offsets differ — pyramid layout)
  const triDst: number[] = [];
  {
    let acc = 0;
    for (const l of triLvls ?? []) {
      triDst.push(acc);
      acc += l.w * l.h;
    }
  }
  const triHzbRO = triHzb?.ro;
  // single-pass clears the id buffers like `packed` (election anchor → 0, side id →
  // 0) but keeps the exact-depth sentinel (depthV → 0xffffffff, atomicMin).
  const packedClear = packed || singlePass;
  const pixelCount = width * height;
  // PERF-3 raster ablation (?rdbg): BUILD-TIME gate — 0 / absent emits the kernel
  // UNCHANGED (production pristine, zero instrumentation). Two independent stage
  // splits live behind it, dispatched by which kernel actually runs in a config:
  //
  //  • DEPTH kernel (shadow-only): rdbg 1/2/3 attribute nanRasterDepth's per-triangle
  //    setup (resolution scaling proved it ~70% setup-bound) — the three depth sinks.
  //
  //  • WORLD1 kernel (the full-pipe camera raster, the PERF LEVER #1 measure-first
  //    gate): rdbg 4→1→2→3 are nested STOP points for the launch-vs-setup-vs-pixel
  //    split — 4 = raw launch (before makeCtx), 1 = +makeCtx/wgcache FLOOR, 2 = +the
  //    per-triangle edge-setup (skip per-pixel), 3 = full. See the four world1 sinks.
  //    FINDING (2026-06-16 LEVER #1 run): at the forest operating point the per-PIXEL
  //    coverage/election loop is ≈90% of world1; launch+makeCtx+edge-setup together
  //    are < the 8.3 ms vsync floor even at 3.3M clusters / 2× res, and pure launch is
  //    < 2.5 ns/wg (≤ ~1.2 ms at 477k cl) ⇒ triangle-granular re-dispatch (#1) is
  //    bounded-marginal and was NOT implemented. CAVEAT for re-measuring: the gutted
  //    variants run at the 120 fps rAF cap, where the c.nanRasterWorld1 GPU timestamp
  //    reads a BOGUS constant ~15.2 ms (cross-frame pipelining span, not active GPU
  //    time) — only GPU-BOUND (frameMs ≫ 8.3) readings of these variants are valid.
  const rdbg = Number(
    new URLSearchParams(window.location.search).get('rdbg') ?? '0',
  );
  // PERF-3 (LOG az/ba): one workgroup == one cluster (128 threads share instId/ci),
  // but makeCtx ran PER THREAD (128×/cluster, incl. the trunk gust texture samples) =
  // 0.46 ms / 16% of nanRasterDepth, 100% redundant. Compute it ONCE (thread 0) and
  // broadcast via workgroup shared memory. DEFAULT ON (A/B-validated bit-identical —
  // f32 round-trip is exact, so the raster still agrees with the resolve's own
  // makeCtx); `?wgcache=0` opts out. Applies to the camera raster AND every shadow-
  // clipmap level + the HW vertex stage (all share buildNaniteRaster).
  const wgcache =
    new URLSearchParams(window.location.search).get('wgcache') !== '0';
  // SWCOOP (90fps arc, pixel-loop structural rewrite; WORLD1 kernel only) — MODES:
  //   0 = old scanline loop VERBATIM (the revert path).
  //   1 = two-bin with a COOPERATIVE large bin: small tris (bbox extent ≤4 px both
  //       axes) lane-private full walk; large (5..16 px) appended to workgroup shared
  //       memory + all 128 lanes stride each tri's pixels (the kVoxScatter Phase A/B
  //       pattern). MEASURED STRONGLY NEGATIVE 2026-07-04 (eye 23.2→40.8, oblique
  //       34.5→45.0 ms grass-off dpr2 even after thread-0 list compaction; 50.7/59.3
  //       uncompacted): tri footprints (≤289 px) are too small to amortize the two
  //       barriers + serial compaction scan + ~7 KB extra workgroup memory (on the
  //       kernel's existing ~8 KB → occupancy), and a cluster's 128 tris are size-
  //       coherent so the lane-divergence "union of bboxes" cost the audit targeted
  //       is small in practice. Kept build-time-gated as the measured A/B evidence.
  //   2 = two-bin split, both bins LANE-PRIVATE (small = full walk dropping the
  //       per-row x-span solve, large = the old row-solve scanline; no shared
  //       memory, no barriers, voxel returnIf preserved). MEASURED NEUTRAL
  //       (eye 23.4 vs 23.2, oblique 34.2 vs 34.1 — the fp32 row solve was never
  //       the wall-time driver).
  // DEFAULT 0 (the shipped loop verbatim): both alternatives are neutral-or-worse,
  // so the old loop stays the default and 1/2 remain as measured diagnostics.
  // BIT-IDENTITY LAW (all modes): same edge coefficients, same coverage rule
  // (biased cw ≥ 0 at the same pixel centres), same depth interp op-order, same
  // election — only work DISTRIBUTION changes. Gates 2026-07-04: scarTotalFrags
  // bit-equal across modes at a deterministic pose (6871171 oblique), shot diffs at
  // the cross-boot floor. Forced 0 under ?rdbg: its mid-kernel returnIf stops would
  // precede mode 1's barrier (non-uniform barrier ⇒ naga validation failure).
  const swcoopQ = Number(
    new URLSearchParams(window.location.search).get('swcoop') ?? '0',
  );
  const swcoop = rdbg !== 0 ? 0 : swcoopQ === 1 || swcoopQ === 2 ? swcoopQ : 0;
  // ?relect — WORLD1 ELECTION ABLATION (90fps-arc pixel-loop split, 2026-07-04).
  // BUILD-TIME gate. The world1 per-covered-pixel ELECTION is: relaxed atomicLoad
  // guard of visPayloadV → If(cand>prev) → atomicMax(visPayloadV) → winner-conditional
  // atomicStore of visBV. This param replaces it (in BOTH the emitW1 small-bin closure
  // AND the inline large-bin/scanline site — default swcoop=2 runs both) to measure the
  // election's share of the ~7.8/18.4 ms pixel loop vs the coverage-walk+interp compute:
  //   1 (DEFAULT) = shipped election, UNTOUCHED (production pristine — no elecSink, no
  //       kernel-end sink; the `relect !== 1` gates below all fall through to the verbatim
  //       atomicMax/atomicStore, so the emitted world1 kernel is byte-for-byte the shipped one).
  //   0 = NO election. The full loop still runs (row solve, coverage, depth interp, 24-bit
  //       key pack) but each covered fragment ACCUMULATES `cand` into a per-thread register
  //       (elecSink, free); ONE atomicMax sink write per raster thread at loop end keeps the
  //       whole chain live (the compiler cannot DCE the loop). default − relect0 = ELECTION TOTAL.
  //   2 = KEEP the atomicLoad guard + `cand>prev` compare; REPLACE only the atomicMax RMW +
  //       winner atomicStore with the register accumulate. default − relect2 = RMW/STORE side;
  //       relect2 − relect0 = the guard-load/compare side.
  // The kernel-end sink (visBV, already bound; unread under relect≠1 since the election that
  // wrote it is gone) is one atomic/thread at a hashed px — the same low-contention idiom the
  // rdbg=1/2 sinks use (NOT one-per-fragment ⇒ no artificial atomic-contention mirage). Frame
  // renders wrong/black in the nanite region under relect 0/2 — DIAGNOSTIC ONLY.
  const relect = Number(
    new URLSearchParams(window.location.search).get('relect') ?? '1',
  );
  // ?clhw per-cluster SW/HW split (see docs/mobile-gpu-perf/SW-HW-CLUSTER-AUDIT.md). The SW
  // world1 kernel skips clusters the cull classified HW (clusterHwClass, bit-identical to
  // kHwPartition ⇒ no holes/double), and an instanced HW draw paints those clusters properly.
  // Requires cull.qHwRasterRO + cull.hwClusterDrawAttr (present only when the cull saw ?clhw).
  const clhw =
    new URLSearchParams(window.location.search).get('clhw') === '1' &&
    !!cull.qHwRasterRO &&
    !!cull.hwClusterDrawAttr;
  const clhwMax = Math.max(
    2,
    Number(new URLSearchParams(window.location.search).get('clhwmax') ?? '16') || 16,
  );
  // projK = px per world-unit at unit depth (matches NaniteCull's cut projection exactly).
  const projK = cam.cotHalfFov.mul(cam.uH).mul(0.5) as unknown as NF;
  // (removed 2026-07-02 cleanup: ?noguard naive-FreePipe diagnostic — atomic-contention
  // hypothesis refuted long ago; ?f2b per-pixel early-out — measured null §5o)
  // HW vertex-pull pass: by DEFAULT drop the redundant per-frame full-res color CLEAR. The HW
  // fragment stage has colorWrite=false (it writes ONLY the vis storage buffers, never
  // the color target — capture confirms the full-res rgba8 is a dead clear/store, never
  // read downstream). The color attachment MUST stay full-res: r184 derives the HW
  // render pass viewport/scissor from RenderTarget.viewport/.scissor (Renderer.js:1599 ->
  // WebGPUBackend.js:1024-1034), so a 1x1 target would force a 1x1 viewport that squashes
  // ALL HW geometry into pixel (0,0) and clips the rest -- the fragment stage would run
  // for ~1 fragment, STARVING the vis-buffer writes (NOT byte-identical). Instead we keep
  // the target full-res (coverage byte-identical) and skip the redundant clear of the
  // dead rgba8 via autoClear=false around hwRender (loadOp Clear->Load). The store of the
  // unchanged texture remains (the backend hardcodes storeOp=Store), but the per-frame
  // full-res CLEAR -- the addressable waste -- is gone. Byte-identical: viewport/coverage
  // unchanged, vis buffers unchanged, color target never read. DEFAULT skips the clear (the
  // byte-exact win, A/B-validated); ?hwrt=1 RESTORES it (the A/B control).
  const hwrt = new URLSearchParams(window.location.search).get('hwrt') === '1';
  // ?scar=1 — 0a SCAR CALIBRATION (voxel-foliage spec §11 Stage 0a / §6.0). A
  // lightweight, BUILD-TIME-gated overdraw counter on the EXISTING world1 triangle
  // raster (production pristine when off — zero atomics injected). It measures, for
  // the MID/FAR FOLIAGE BAND (leaf clusters whose instance sits in [scarNear, scarFar],
  // ≈ the voxelizable distance window from transitionDist out to the instMinPx cull
  // edge): (1) the OVERDRAW factor = band fragments / band covered-pixels, and (2) the
  // band's share of frame fragments. This is the cheapest disproof of the SCAR (§6.0):
  // voxels only win if the band overdraws enough that (covered-px + brick-bin cost) <
  // the triangle fragments this counter reports. NOTHING about raster behaviour changes
  // — only atomic counters are added, gated on `scar`. The band is classified per
  // CLUSTER on instance distance (== the same A.xyz the cull uses), matching the
  // covered-pixel post-pass so numerator and denominator share one band definition.
  const scarParams = new URLSearchParams(window.location.search);
  const scar = scarParams.get('scar') === '1';
  // band bounds (m), tuneable: DEFAULT ≈ the spec's transitionDist (~40 m) out to the
  // instMinPx ~110 px cull edge (~90 m for R≈3.5 m). ?scarnear= / ?scarfar= override.
  const scarNear = uniformF(Number(scarParams.get('scarnear') ?? '40'));
  const scarFar = uniformF(Number(scarParams.get('scarfar') ?? '90'));
  const qRasterRO = cull.qRasterRO;
  const visDepthV = vis.depthV;
  const visPayloadV = vis.payloadV;
  const visBV = vis.visBV;
  // depth → 16-bit key, INVERTED so nearer (smaller cz) ⇒ LARGER key ⇒ wins atomicMax.
  const depthKey16 = (cz: NF): NU =>
    uint(float(1).sub(cz).mul(65535).clamp(0, 65535)) as unknown as NU;
  // PERF-VB4 world1: a 24-BIT key (8-bit id tiebreak below it). The full id is in the
  // side buffer, so the election word only needs depth + a coarse tiebreak — 24 bits of
  // depth (256× finer than the combined path's 16) makes the wp-reconstruction banding
  // sub-pixel. The resolve/shadowHalf decode cz = 1 − (key>>8)/16777215; the HZB reads
  // the top 16 bits (key>>16) which still decode as a valid coarse occluder.
  const depthKey24 = (cz: NF): NU =>
    uint(float(1).sub(cz).mul(16777215).clamp(0, 16777215)) as unknown as NU;

  // hwQueue: [0] = atomic count, then (payload, instId) pairs.
  // ⚠ SCAR FOLD (?scar=1): world1's compute stage is ALREADY at the WebGPU 10-storage-
  // buffer per-stage ceiling, so a SEPARATE scar buffer makes it 11 (validation error →
  // invalid pipeline → black frame, no counters). The scar counters (4 u32) are therefore
  // CARVED INTO THE TAIL of this already-bound hwQueue buffer at [SCAR_BASE..+4) — they
  // never overlap the queue's [0 .. 1+HW_CAP*2) range, so no new binding, stays at 10.
  const SCAR_BASE = 1 + HW_CAP * 2;
  // W2 ?trihzb TAIL FOLD (same binding-budget law as scar above): the per-tri
  // occlusion test needs the prev-frame HZB, but binding the pyramid buffer in
  // world1 is an 11th storage buffer (measured: validation error, dead pipeline).
  // So a tiny copy kernel (own 2-buffer pipeline, runs right after the pyramid
  // build) mirrors ONE fixed level into this buffer's tail at [TRIHZB_BASE..).
  // Tail is pre-filled with 1.0f bits (far) so frame-0 rejects nothing; clears
  // never touch it ([0] + scar slots only).
  const TRIHZB_BASE = SCAR_BASE + 4;
  const triTailN = (triLvls ?? []).reduce((a, l) => a + l.w * l.h, 0);
  const hwQueueInit = new Uint32Array(TRIHZB_BASE + triTailN);
  if (triTailN > 0) hwQueueInit.fill(0x3f800000, TRIHZB_BASE); // 1.0f = far
  const hwQueueAttr = new StorageBufferAttribute(hwQueueInit, 1);
  hwQueueAttr.name = 'nanHwQueue';
  const hwQueueV = sU32Views(hwQueueAttr, TRIHZB_BASE + triTailN);
  const hwDrawAttr = new IndirectStorageBufferAttribute(new Uint32Array(4), 4);
  hwDrawAttr.name = 'nanHwDraw';
  const hwDrawBuf = sU32Views(
    hwDrawAttr as unknown as StorageBufferAttribute,
    4,
  ).rw;

  // consistency audit (?audit=1): [0] = orphans (depth written but payload
  // never matched it — ANY pass disagreement, SW or HW, shows up here),
  // [1] = covered pixels
  const auditAttr = new StorageBufferAttribute(new Uint32Array(4), 1);
  auditAttr.name = 'nanRasterAudit';
  const auditV = sU32Views(auditAttr, 4);

  // 0a SCAR band counters (?scar=1). [0] = band fragments (the OVERDRAW numerator —
  // every covered fragment a band leaf cluster rasterizes, incl. occluded ones),
  // [1] = band covered PIXELS (the denominator — filled by the kScarCovered post-pass:
  // exactly 1 per pixel whose FINAL winner is a band leaf cluster), [2] = total frame
  // fragments (all clusters → the band's fragment share), [3] = band cluster-emit count
  // (diagnostic). Overdraw = [0]/[1]; band share = [0]/[2].
  // ⚠ BINDING BUDGET (see hwQueue above): world1's compute stage is already at the WebGPU
  // 10-storage-buffer ceiling, so scar's 4 counters are FOLDED into the hwQueue buffer at
  // [SCAR_BASE..+4) — same buffer object = same binding = no 11th slot. `scarV` aliases
  // `hwQueueV`; every scar access below is offset by SCAR_BASE (helper `scarEl`). The
  // readback reads hwQueueAttr at byte offset SCAR_BASE*4.
  const scarAttr = hwQueueAttr;
  const scarV = hwQueueV;
  // scar atomic element [i] lives at hwQueue slot SCAR_BASE+i. Returns the atomic ref
  // (what atomicStore/atomicAdd take), so call sites read like the old scarV.atomic.element.
  const scarEl = (i: number): ReturnType<typeof scarV.atomic.element> =>
    scarV.atomic.element(uint(SCAR_BASE + i));

  // ---- shared fetch helpers (NaniteFetch.ts — also the resolve's decode) ----------
  const nfetch = makeFetch(gpu, heightTex, disp, wind);
  const { makeCtx, fetchWorldVert, fetchWorldVertDyn } = nfetch;
  // M2l hw1fetch: HW vertex stage reconstructs ONE corner (runtime-selected) instead
  // of fetching all 3 and selecting — same selected vertex by construction.
  const hw1fetch =
    new URLSearchParams(window.location.search).get('hw1fetch') === '1';
  // PERF-3 win #2 — the cooperative vertex-transform cache lives in its own module
  // (default OFF, ?vcompact=1; measured marginal/conditional — see NaniteVertexCache).
  const vcache = makeVertexCache(gpu, nfetch);

  const edgeFn = (a: NV2, b: NV2, p: NV2): NF =>
    p.y
      .sub(a.y)
      .mul(b.x.sub(a.x))
      .sub(p.x.sub(a.x).mul(b.y.sub(a.y))) as unknown as NF;

  // ?wgcache bool→uint for packing isHF/isDAG into the shared-memory uint array
  const b2u = (b: NB): NU =>
    (b as unknown as { select(a: NU, c: NU): NU }).select(uint(1), uint(0));

  // SWCOOP workgroup-shared element write (the NaniteVoxelRaster wgSet idiom —
  // .element() is typed as a bare Node in @types; ONE cast here, call sites clean).
  const wgW = (
    arr: ReturnType<typeof workgroupArray>,
    i: NU,
    v: unknown,
  ): void => {
    (arr.element(i) as unknown as { assign(x: unknown): unknown }).assign(v);
  };

  // ---- kVisClear ------------------------------------------------------------------
  // W1 (?dvclear=0, spec-orchestration-submit-folds §Stage-2): in the world single-pass
  // path NOTHING reads or writes visDepthV (exhaustive consumer table in the spec — the
  // election lives in visPayloadV; HZB/resolve/shadowHalf read payload, never depthV), so
  // its 3.3M-pixel 0xffffffff clear is pure store traffic. Build-time gated + auto-kept
  // under every debug flag that DOES read it (nanprobe/audit/rdbg). Shadow/View raster
  // instances have singlePass=false ⇒ never gated. DEFAULT ON — clear SKIPPED (gate
  // passed 2026-07-02: shot-diff at D0 band, medians sub-noise). ?dvclear=1 = keep clear.
  const dvParams = new URLSearchParams(window.location.search);
  const skipDepthClear =
    singlePass &&
    dvParams.get('dvclear') !== '1' &&
    dvParams.get('nanprobe') !== '1' && // probe reads vis.depthV.ro
    dvParams.get('audit') !== '1' && // kAudit reads visDepthV.ro
    rdbg === 0; // rdbg sinks atomicMin depthV
  // ?visclear=0 DEBUG (measurement only): A/B whether the ~11ms c.nanVisClear timestamp is
  // real store cost or a render‖compute WAR-stall artifact — skips the 2 hot full-screen
  // vis clears (payloadV + visBV). OFF (default / flag absent) = byte-identical to today.
  const skipVisClear = dvParams.get('visclear') === '0';
  // P9 (shadow strip clear): the tiny tail of kVisClear — ONLY the hw-queue +
  // audit counters. The strip-scoped shadow clear covers the depth texels itself
  // but MUST still reset this counter or the HW depth pass renders an
  // ever-growing stale triangle list (the P9 regression: +8ms moving).
  const kHwQueueClear = Fn(() => {
    If(instanceIndex.equal(uint(0)), () => {
      atomicStore(hwQueueV.atomic.element(0), uint(0));
      atomicStore(auditV.atomic.element(0), uint(0));
      atomicStore(auditV.atomic.element(1), uint(0));
    });
  })().compute(1, [1]);
  (kHwQueueClear as unknown as ComputeKernel).setName('nanHwQueueClear');

  const kVisClear = Fn(() => {
    If(instanceIndex.lessThan(uint(pixelCount)), () => {
      if (!skipDepthClear)
        atomicStore(visDepthV.atomic.element(instanceIndex), uint(0xffffffff));
      // packed/single-pass: payload(+visB) are atomicMax/side targets ⇒ clear to 0 (the
      // smallest, "no fragment"). legacy: payload keeps the 0xffffffff orphan sentinel.
      // ?visclear=0 DEBUG (measurement only): skip the 2 hot vis clears to A/B the WAR-stall hypothesis; produces a dirty render.
      if (!skipVisClear) {
        atomicStore(
          visPayloadV.atomic.element(instanceIndex),
          uint(packedClear ? 0 : 0xffffffff),
        );
        if (packedClear)
          atomicStore(visBV.atomic.element(instanceIndex), uint(0));
      }
    });
    If(instanceIndex.equal(uint(0)), () => {
      atomicStore(hwQueueV.atomic.element(0), uint(0));
      atomicStore(auditV.atomic.element(0), uint(0));
      atomicStore(auditV.atomic.element(1), uint(0));
      if (scar) {
        atomicStore(scarEl(0), uint(0));
        atomicStore(scarEl(1), uint(0));
        atomicStore(scarEl(2), uint(0));
        atomicStore(scarEl(3), uint(0));
      }
    });
  })().compute(pixelCount, [256]);
  (kVisClear as unknown as ComputeKernel).setName('nanVisClear');

  // ---- SW raster kernels (fixed-point integer scanline) ------------------------------
  // mode 'depth' = atomicMin Z only (the SHADOW depth-only path). 'combined' = single-pass
  // packed Z+id for the NaniteView debug view (16-bit depth split idLo/idHi). 'world1' =
  // the WORLD single pass (PERF-VB4): a 24-bit depth election (visPayloadV) whose winner
  // stores the full 25-bit id into the side buffer visBV; the resolve reconstructs depth
  // from the election key. No exact depthV (a 3rd hot-loop atomic buffer = a 3× cliff).
  const rasterKernel = (mode: 'depth' | 'combined' | 'world1'): unknown => {
    // SWCOOP applies to the world single pass only; depth/combined keep the old
    // scanline loop byte-identical (build-time — the flag picks WHICH loop is emitted).
    const coopMode = mode === 'world1' ? swcoop : 0;
    const coop = coopMode === 1; // cooperative large bin (shared memory + barriers)
    const split = coopMode === 2; // two-bin, both lane-private (no shared memory)
    const kn = Fn(() => {
      const itemIdx = wgLinear(DISPATCH_ROW).toVar();
      const localTri = localX().toVar();
      const itemCount = qRasterRO.element(0).x;
      returnIf(itemIdx.greaterThanEqual(itemCount));
      const item = qRasterRO.element(itemIdx.add(uint(1)));
      const instId = item.x.toVar();
      const ci = item.y.toVar();
      // ?clhw: skip clusters the cull routed to the HW instanced draw. The decision is
      // clusterHwClass — BIT-IDENTICAL to kHwPartition's — so the SW-skip and the HW-append
      // can never disagree (no holes, no double-raster) without touching qRaster. UNIFORM
      // across the workgroup (instId/ci are per-cluster) and BEFORE any barrier ⇒ every live
      // thread returns together, no barrier deadlock. Built only under ?clhw&&world1.
      // ?clhw SW-skip: HW clusters are painted by the instanced HW draw, so the SW kernel
      // returns them here. With wgcache ON (default) the classify is computed ONCE on thread 0
      // and broadcast (slot 10, below) — NOT 128×/cluster on the SW hot path. The ?wgcache=0
      // fallback classifies per-thread (the result is uniform ⇒ all threads return together).
      if (clhw && mode === 'world1' && !wgcache) {
        returnIf(
          clusterHwClass(gpu, cam.camPos as unknown as NV3, projK, instId, ci, clhwMax),
        );
      }
      // ?relect (build-time, world1 only): per-thread election-key accumulator + the
      // election emitter shared by BOTH world1 sites (emitW1 small bin + inline scanline).
      // null / verbatim election when relect===1 (default) ⇒ world1 kernel stays pristine.
      const elecSink =
        mode === 'world1' && relect !== 1 ? uint(0).toVar() : null;
      const doElection = (px: NU, cand: NU, idStore: NU): void => {
        if (mode === 'world1' && relect === 0) {
          // NO election: fold the key into the per-thread register (one sink at loop end).
          (elecSink as NonNullable<typeof elecSink>).addAssign(cand);
        } else if (mode === 'world1' && relect === 2) {
          // guard-load + compare KEPT; RMW + winner-store replaced by the register sink.
          const prevE = aLoadU(visPayloadV.atomic.element(px));
          If(cand.greaterThan(prevE), () => {
            (elecSink as NonNullable<typeof elecSink>).addAssign(cand);
          });
        } else {
          // relect===1 (default) — the SHIPPED election, verbatim.
          const prevE = aLoadU(visPayloadV.atomic.element(px));
          If(cand.greaterThan(prevE), () => {
            const wonE = atomicMax(
              visPayloadV.atomic.element(px),
              cand,
            ) as unknown as NU;
            If(cand.greaterThan(wonE), () => {
              atomicStore(visBV.atomic.element(px), idStore);
            });
          });
        }
      };
      if (mode === 'world1' && rdbg === 4) {
        // RAW LAUNCH floor (?rdbg=4): return BEFORE makeCtx + the wgcache broadcast —
        // isolates the pure workgroup-launch cost of the QRASTER_CAP×128 grid (one
        // wg/cluster) from makeCtx. (rdbg1 − rdbg4) = makeCtx + the wgcache barrier;
        // rdbg4 = the launch overhead the triangle-granular re-shape directly attacks
        // (it dispatches a flat ~visTris grid instead of one near-empty wg/cluster).
        // Thread-0-only sink (1 atomic/wg) consumes item so the work-item read stays.
        If(localTri.equal(uint(0)), () => {
          atomicMin(
            visDepthV.atomic.element(instId.add(ci).mod(uint(pixelCount))),
            instId,
          );
        });
        returnIf(itemCount.greaterThanEqual(uint(0)));
      }
      let ctx: VertCtx;
      // matClass (F): broadcast from the wgcache thread-0 decode; null when wgcache
      // is off, so the voxel-skip below falls back to its per-thread mesh reload.
      // Bit-identical — the broadcast is the SAME mesh word6 extract, computed once.
      let matClassBroadcast: NU | null = null;
      if (wgcache) {
        // Compute makeCtx ONCE (thread 0), broadcast through workgroup shared
        // memory, so the per-thread hot path never re-decodes the cluster. The f32
        // round-trip is exact, so the cached ctx is bit-identical to a per-thread
        // makeCtx ⇒ the raster still agrees with the resolve. The only prior
        // early-out (itemIdx ≥ itemCount) is UNIFORM across the workgroup, so every
        // live thread reaches the barrier (no deadlock).
        //   shU: 0..9 = ctx (isHF,isDAG,triStart,triCount,meshId,channel,gx,gz,qxw,
        //     twoSided); 10 = matClass (voxel-skip, was a per-thread mesh reload);
        //     11 = ?clhw SW/HW classify (clhw+world1 only).
        //   shF: 0..10 = A.xyzw,B.xyzw,oX,oZ,cell; 11..20 = wind (when on); 21,22 =
        //     yawSc.cy/.sy (cos/sin, was recomputed on all 128 threads per cluster).
        // NOTE: TrunkWindFields is serialized field-by-field here — adding a wind
        // field (e.g. swayXPhase, slot 20) MUST extend shF + both halves below.
        const shU = workgroupArray('uint', clhw && mode === 'world1' ? 12 : 11);
        const shF = workgroupArray('float', 23);
        // .element() is typed as a bare Node here — cast to the fluent TSL types
        const setU = (i: number, v: NU): void =>
          void (
            shU.element(uint(i)) as unknown as { assign(x: NU): unknown }
          ).assign(v);
        const setF = (i: number, v: NF): void =>
          void (
            shF.element(uint(i)) as unknown as { assign(x: NF): unknown }
          ).assign(v);
        const getU = (i: number): NU => shU.element(uint(i)) as unknown as NU;
        const getF = (i: number): NF => shF.element(uint(i)) as unknown as NF;
        If(localTri.equal(uint(0)), () => {
          const c = makeCtx(instId, ci);
          setU(0, b2u(c.isHF));
          setU(1, b2u(c.isDAG));
          setU(2, c.triStart);
          setU(3, c.triCount);
          setU(4, c.meshId);
          setU(5, c.channel);
          setU(6, c.gx);
          setU(7, c.gz);
          setU(8, c.qxw);
          setU(9, b2u(c.twoSided)); // N9-C2 two-sided flag (slot 9)
          // matClass (slot 10, F) — broadcast the mesh word6 extract so the voxel-skip
          // reads ONE value instead of reloading gpu.meshes on all 128 threads. Same
          // expression as the per-thread read below ⇒ bit-identical.
          setU(
            10,
            elemU(gpu.meshes, c.meshId.mul(uint(MESH_WORDS)).add(uint(6)))
              .shiftRight(uint(8))
              .bitAnd(uint(0xff)),
          );
          if (clhw && mode === 'world1') {
            // ?clhw: broadcast the SW/HW classify in slot 11 — computed ONCE here on thread 0,
            // read after the barrier for a uniform early-out (vs 128×/cluster per-thread).
            setU(
              11,
              b2u(
                clusterHwClass(gpu, cam.camPos as unknown as NV3, projK, instId, ci, clhwMax),
              ),
            );
          }
          setF(0, c.A.x as unknown as NF);
          setF(1, c.A.y as unknown as NF);
          setF(2, c.A.z as unknown as NF);
          setF(3, c.A.w as unknown as NF);
          setF(4, c.B.x as unknown as NF);
          setF(5, c.B.y as unknown as NF);
          setF(6, c.B.z as unknown as NF);
          setF(7, c.B.w as unknown as NF);
          setF(8, c.oX);
          setF(9, c.oZ);
          setF(10, c.cell);
          if (wind) {
            const w = c.wind as NonNullable<VertCtx['wind']>;
            setF(11, w.h0);
            setF(12, w.dirX);
            setF(13, w.dirY);
            setF(14, w.leanBase);
            setF(15, w.swayABase);
            setF(16, w.swayPhase); // hoisted sin (was natW)
            setF(17, w.ph);
            setF(18, w.branchBase);
            setF(19, w.flutBase); // N9-C0 leaf flutter
            setF(20, w.swayXPhase); // hoisted cross-sway sin
          }
          // yawSc (E): broadcast the instance-yaw cos/sin makeCtx already computed,
          // so the per-thread reconstruction stops recomputing sin+cos 128×/cluster.
          setF(21, c.yawSc.cy);
          setF(22, c.yawSc.sy);
        });
        workgroupBarrier();
        // matClass (F): capture the broadcast for the voxel-skip below (always-on).
        matClassBroadcast = getU(10).toVar() as unknown as NU;
        // ?clhw: skip clusters the cull routed to the HW instanced draw (decision broadcast
        // from thread 0 above). Uniform across the workgroup ⇒ all threads return together.
        if (clhw && mode === 'world1') {
          returnIf(getU(11).toVar().equal(uint(1)));
        }
        const cB = vec4(
          getF(4).toVar(),
          getF(5).toVar(),
          getF(6).toVar(),
          getF(7).toVar(),
        ) as unknown as NV4;
        ctx = {
          isHF: getU(0).toVar().equal(uint(1)),
          isDAG: getU(1).toVar().equal(uint(1)),
          A: vec4(
            getF(0).toVar(),
            getF(1).toVar(),
            getF(2).toVar(),
            getF(3).toVar(),
          ) as unknown as NV4,
          B: cB,
          yawSc: { cy: getF(21).toVar(), sy: getF(22).toVar() },
          triStart: getU(2).toVar(),
          triCount: getU(3).toVar(),
          meshId: getU(4).toVar(),
          channel: getU(5).toVar(),
          twoSided: getU(9).toVar().equal(uint(1)),
          wind: wind
            ? {
                h0: getF(11).toVar(),
                dirX: getF(12).toVar(),
                dirY: getF(13).toVar(),
                leanBase: getF(14).toVar(),
                swayABase: getF(15).toVar(),
                swayPhase: getF(16).toVar(),
                ph: getF(17).toVar(),
                branchBase: getF(18).toVar(),
                flutBase: getF(19).toVar(), // N9-C0 leaf flutter
                swayXPhase: getF(20).toVar(),
              }
            : null,
          gx: getU(6).toVar(),
          gz: getU(7).toVar(),
          qxw: getU(8).toVar(),
          oX: getF(8).toVar(),
          oZ: getF(9).toVar(),
          cell: getF(10).toVar(),
        } as unknown as VertCtx;
      } else {
        ctx = makeCtx(instId, ci);
      }

      // PERF-3 win #2 — cooperative vertex-transform cache (own module; ?vcompact=1).
      // MUST run BEFORE the voxel returnIf below: prime() emits a workgroupBarrier, and
      // a barrier after a storage-derived returnIf is non-uniform control flow to naga
      // ⇒ WGSL validation failure ⇒ the whole world1 pipeline silently dies (the
      // "?vcompact renders an empty scene" bitrot, found 2026-07-02). Voxel clusters
      // have vcCount=0 (no compact range) so their populate no-ops — they pay only the
      // barrier before bailing.
      const corner = vcache.prime(ctx, ci, localTri);

      // voxel-foliage (spec §4.1 / §A1): SKIP voxel(7) clusters in the TRIANGLE raster.
      // The cut emits voxel clusters into the SAME qRaster as triangles (§4.6); the
      // post-traverse fan-out copies them to qVoxRaster for the Stage-2 voxel bin, but
      // they REMAIN in qRaster. A voxel cluster's word6/word7 point at BRICKS, not tris
      // (registerVoxelHead), so fetchWorldVert would read garbage triangle data —
      // bail before any vertex work. UNIFORM across the workgroup (matClass is per
      // cluster, broadcast via ctx.meshId), so every live thread returns (no barrier
      // deadlock; the wgcache + vcache barriers above already ran). The bricks render
      // via the scatter voxel raster (Stage 2) into the same vis buffers.
      // SWCOOP: the coop variant may NOT return here — its Phase A/B workgroupBarrier
      // sits at kernel top scope BELOW, and a barrier after a storage-derived return is
      // non-uniform control flow to naga (the ?vcompact bitrot, 2026-07-02). The skip
      // becomes a per-lane GUARD on the triangle work instead (execution-identical:
      // voxel clusters do no triangle work either way; their Phase-B slots stay empty).
      let swVoxGuard: NB | null = null;
      {
        // matClass (F): reuse the wgcache broadcast when present; else per-thread reload.
        const mcVox =
          matClassBroadcast ??
          elemU(gpu.meshes, ctx.meshId.mul(uint(MESH_WORDS)).add(uint(6)))
            .shiftRight(uint(8))
            .bitAnd(uint(0xff));
        if (coop) {
          swVoxGuard = mcVox.notEqual(uint(7)).toVar() as unknown as NB;
        } else {
          returnIf(mcVox.equal(uint(7)));
        }
      }

      // 0a SCAR per-cluster band classification (?scar=1, world1 only). Computed ONCE
      // per cluster (hoisted out of the per-fragment loop): is this a LEAF (matClass==4)
      // cluster whose instance sits in the voxelizable mid/far band [scarNear, scarFar]?
      // Distance is instance-A → camera (the SAME center the cull screen-sizes on,
      // NaniteCull.ts:393), so this matches the covered-pixel post-pass band exactly.
      // `bandFlag` is the per-fragment gate; null when scar is off (production pristine).
      let bandFlag: NB | null = null;
      if (scar && mode === 'world1') {
        const mc = elemU(
          gpu.meshes,
          ctx.meshId.mul(uint(MESH_WORDS)).add(uint(6)),
        )
          .shiftRight(uint(8))
          .bitAnd(uint(0xff));
        const isLeaf = mc.equal(uint(4));
        const distC = vec3(cam.camPos).sub(ctx.A.xyz).length();
        bandFlag = isLeaf
          .and(distC.greaterThanEqual(scarNear))
          .and(distC.lessThanEqual(scarFar))
          .toVar() as unknown as NB;
        // count band cluster emits (diagnostic), once per cluster (thread 0).
        If(localTri.equal(uint(0)).and(bandFlag), () => {
          atomicAdd(scarEl(3), uint(1));
        });
      }

      if (mode === 'depth' && rdbg === 3) {
        // ?rdbg=3 — stop right after makeCtx (the per-(instance,cluster) decode:
        // cluster/mesh/instance reads + the trunk-wind gust TEXTURE samples that
        // run once per THREAD = 128×/cluster). The sink folds the wind scalars so
        // those gust samples can't be DCE'd. (rdbg1 − rdbg3) = the 3 fetchWorldVert.
        const wSink = ctx.wind
          ? ctx.wind.leanBase.add(ctx.wind.swayPhase).add(ctx.wind.branchBase)
          : float(0);
        const sinkV = ctx.A.x.add(toF(ctx.triCount)).add(wSink).clamp(0, 1);
        const sinkPx = itemIdx
          .mul(uint(2654435761))
          .add(localTri)
          .mod(uint(pixelCount));
        atomicMin(
          visDepthV.atomic.element(sinkPx),
          bcF2U(sinkV as unknown as NF),
        );
        returnIf(itemCount.greaterThanEqual(uint(0)));
      }

      // PERF LEVER #1 STAGE-SPLIT (?rdbg on the WORLD single-pass kernel) — the
      // measure-first STOP gate that sizes the launch+ctx FLOOR (killable by the
      // triangle-granular re-shape) vs the per-triangle edge-setup (NOT killable —
      // same triangles either way) vs the per-pixel coverage loop. Build-time gate,
      // production pristine when rdbg==0. Numbering is world1-LOCAL (independent of
      // the depth kernel's 1/2/3):
      //   rdbg=1 → STOP right after makeCtx/wgcache broadcast = launch+ctx FLOOR
      //            (workgroup launch + barrier + the per-cluster makeCtx broadcast;
      //            skips ALL per-triangle work). The per-cluster slice of this floor
      //            is the ~4.3 ms "fixed term" candidate the re-shape can reclaim.
      //   rdbg=2 → ALSO do the per-triangle work (3× fetchWorldVert + vp transform +
      //            ndc + fixed-point snap + bbox + edge setup) but SKIP the per-pixel
      //            coverage loop + atomics (sink at the loop entry — see below).
      //   rdbg=3 → full kernel (== rdbg unset for world1).
      // (rdbg2 − rdbg1) = per-triangle edge-setup; (rdbg3 − rdbg2) = per-pixel loop.
      if (mode === 'world1' && rdbg === 1) {
        // launch+ctx FLOOR: consume the broadcast ctx so makeCtx + the wgcache
        // barrier survive DCE, then unconditionally return BEFORE any corner fetch
        // or triangle work. CRITICAL — the sink fires only on thread 0 (ONE atomic
        // per workgroup, not 128): the idle lanes (localTri ≥ triCount) do NOTHING
        // in the REAL kernel, so making all 128 lanes atomicMin here would inject
        // ~61M atomics of artificial contention that the production floor never
        // pays (the mirage trap). One atomic/workgroup matches the real coverage
        // floor while still pinning makeCtx + the wgcache barrier (all 128 lanes
        // reach the barrier, so the launch+barrier cost is fully timed). The sink
        // folds A.x + triCount + wind so the per-cluster gust samples can't be DCE'd.
        If(localTri.equal(uint(0)), () => {
          const wSink = ctx.wind
            ? ctx.wind.leanBase.add(ctx.wind.swayPhase).add(ctx.wind.branchBase)
            : float(0);
          const sinkV = ctx.A.x.add(toF(ctx.triCount)).add(wSink).clamp(0, 1);
          const sinkPx = itemIdx.mod(uint(pixelCount));
          atomicMin(
            visDepthV.atomic.element(sinkPx),
            bcF2U(sinkV as unknown as NF),
          );
        });
        returnIf(itemCount.greaterThanEqual(uint(0)));
      }

      // ── SWCOOP shared LARGE-bin records (Phase A → Phase B), slot = localTri.
      // 3 i32 edge values at the clamped bbox origin + 3 PACKED per-unit edge steps
      // (ex|ey as i16 pairs — a ≤16 px bbox bounds vert spread to <17·256 = 4352
      // subpixel units, i16-safe; Phase B sign-extends and ×256 exactly as the old
      // setup did) + packed bbox origin + packed w/h/bias-bits + payload + 4 f32
      // depth-interp constants = 13 words/tri × 128 tris = 6.5 KB workgroup memory.
      // ⚠️ BUDGET: the kernel already carries ~8.3 KB (wgcache broadcast + vcache);
      // the 16-word layout (unpacked steps) hit 16528 B > the 16384 B device default
      // — the packing is what keeps the pipeline valid.
      const swSh = coop
        ? {
            rw0: workgroupArray('int', MAX_CLUSTER_TRIS),
            rw1: workgroupArray('int', MAX_CLUSTER_TRIS),
            rw2: workgroupArray('int', MAX_CLUSTER_TRIS),
            exy0: workgroupArray('uint', MAX_CLUSTER_TRIS),
            exy1: workgroupArray('uint', MAX_CLUSTER_TRIS),
            exy2: workgroupArray('uint', MAX_CLUSTER_TRIS),
            xy: workgroupArray('uint', MAX_CLUSTER_TRIS),
            wh: workgroupArray('uint', MAX_CLUSTER_TRIS),
            pay: workgroupArray('uint', MAX_CLUSTER_TRIS),
            z0: workgroupArray('float', MAX_CLUSTER_TRIS),
            z1: workgroupArray('float', MAX_CLUSTER_TRIS),
            z2: workgroupArray('float', MAX_CLUSTER_TRIS),
            rcp: workgroupArray('float', MAX_CLUSTER_TRIS),
            // COMPACTED slot list + count (thread-0 scan between the two barriers).
            // Without it Phase B scanned ALL ≤128 slots on ALL 128 lanes in EVERY
            // workgroup (incl. the ~20k voxel workgroups) — ~1e9 shared-mem probes
            // per frame, measured 23.2 → 50.7 ms eye (the first-cut regression).
            list: workgroupArray('uint', MAX_CLUSTER_TRIS),
            count: workgroupArray('uint', 1),
          }
        : null;
      if (coop && swSh) {
        // EVERY lane seeds its slot EMPTY (wh=0 ⇒ area 0 ⇒ zero-trip Phase-B loop) so
        // small-bin/HW-routed/rejected/voxel/idle lanes contribute nothing (the voxel
        // kernel's seeding idiom, NaniteVoxelRaster.ts:739).
        wgW(swSh.wh, localTri, uint(0));
      }
      // SWCOOP world1 fragment emission — VERBATIM the swcoop=0 loop's depth interp
      // (unbiased integer weights, same f32 op order — see the bias rationale in the
      // old loop below), [0,1] gate, scar counters and 24-bit election. Shared by the
      // small-bin lane-private walk (Phase A) and the cooperative Phase B, so the two
      // bins cannot drift apart. `cand` is built INSIDE the consuming subtree (the TSL
      // hoist-pathology rule, NaniteVoxelRaster.ts:1264).
      const emitW1 =
        coop || split
          ? (
              px: NU,
              uw0: NI,
              uw1: NI,
              uw2: NI,
              z0: NF,
              z1: NF,
              z2: NF,
              rcp: NF,
              pay: NU,
            ): void => {
              const cz = toF(uw0)
                .mul(z0)
                .add(toF(uw1).mul(z1))
                .add(toF(uw2).mul(z2))
                .mul(rcp)
                .toVar();
              If(cz.greaterThanEqual(0).and(cz.lessThanEqual(1)), () => {
                if (scar) {
                  atomicAdd(scarEl(2), uint(1));
                  if (bandFlag) {
                    If(bandFlag, () => {
                      atomicAdd(scarEl(0), uint(1));
                    });
                  }
                }
                const cand = depthKey24(cz as unknown as NF)
                  .shiftLeft(uint(8))
                  .bitOr(pay.bitAnd(uint(0xff)))
                  .toVar();
                doElection(px, cand, pay);
              });
            }
          : null;

      // (vcache.prime moved ABOVE the voxel returnIf — its barrier must precede any
      // storage-derived return; see the note there.)
      const swTriLive =
        coop && swVoxGuard
          ? (swVoxGuard as unknown as { and(o: NB): NB }).and(
              localTri.lessThan(ctx.triCount),
            )
          : localTri.lessThan(ctx.triCount);
      If(swTriLive, () => {
        const w0 = corner(localTri, 0);
        const w1 = corner(localTri, 1);
        const w2 = corner(localTri, 2);

        const p0 = cam.vp.mul(vec4(w0, 1)).toVar();
        const p1 = cam.vp.mul(vec4(w1, 1)).toVar();
        const p2 = cam.vp.mul(vec4(w2, 1)).toVar();

        if (mode === 'depth' && rdbg === 1) {
          // ?rdbg=1 — stop right after the 3 vertex fetch+transforms. The sink
          // atomicMin consumes ALL THREE clip positions BEFORE the early-out, so
          // the compiler cannot sink the fetchWorldVert/transform work past the
          // return (it would be a side-effecting use). returnIf's guard is a
          // runtime-true compare (uint ≥ 0) ⇒ always returns here, yet the
          // downstream stays REACHABLE (no unreachable-code error). Isolates:
          // work-item fetch + ctx + 3× fetchWorldVert + 3× vp transform.
          const sinkV = p0.z
            .div(p0.w.max(NEAR_EPS))
            .add(p1.z.div(p1.w.max(NEAR_EPS)))
            .add(p2.z.div(p2.w.max(NEAR_EPS)))
            .mul(1 / 3)
            .clamp(0, 1);
          const sinkPx = itemIdx
            .mul(uint(2654435761))
            .add(localTri)
            .mod(uint(pixelCount));
          atomicMin(
            visDepthV.atomic.element(sinkPx),
            bcF2U(sinkV as unknown as NF),
          );
          returnIf(itemCount.greaterThanEqual(uint(0)));
        }

        const payload = itemIdx
          .shiftLeft(uint(CLUSTER_TRI_BITS))
          .bitOr(localTri)
          .toVar();

        const nearOK = p0.w
          .greaterThan(NEAR_EPS)
          .and(p1.w.greaterThan(NEAR_EPS))
          .and(p2.w.greaterThan(NEAR_EPS));

        If(nearOK.not(), () => {
          // near-plane crossing → HW path clips it (never drop, F10c)
          const slot = atomicAdd(
            hwQueueV.atomic.element(0),
            uint(1),
          ) as unknown as NU;
          If(slot.lessThan(uint(HW_CAP)), () => {
            const base = slot.mul(uint(2)).add(uint(1));
            atomicStore(hwQueueV.atomic.element(base), payload);
            atomicStore(hwQueueV.atomic.element(base.add(uint(1))), instId);
          });
        }).Else(() => {
          const ndc0 = p0.xyz.div(p0.w).toVar();
          const ndc1 = p1.xyz.div(p1.w).toVar();
          const ndc2 = p2.xyz.div(p2.w).toVar();

          const areaNdc = edgeFn(
            ndc0.xy as unknown as NV2,
            ndc1.xy as unknown as NV2,
            ndc2.xy as unknown as NV2,
          );
          // N9-C2: front-faces pass; a two-sided (leaf) back-face is re-wound to CCW
          // in place so the positive-area core below rasters it once (orientForRaster).
          const accept = orientForRaster(
            ndc1 as unknown as NV3,
            ndc2 as unknown as NV3,
            areaNdc as unknown as NF,
            ctx.twoSided,
          );
          If(accept, () => {
            const W = float(cam.uW);
            const H = float(cam.uH);
            const s0 = ndc0.xy.add(1).mul(0.5).mul(vec2(W, H)).toVar();
            const s1 = ndc1.xy.add(1).mul(0.5).mul(vec2(W, H)).toVar();
            const s2 = ndc2.xy.add(1).mul(0.5).mul(vec2(W, H)).toVar();
            // FIXED-POINT snap (N3a): 1/256-px integer grid — 8 subpixel bits,
            // the D3D HW convention. All coverage below is exact i32 math:
            // watertight at shared edges and bit-identical between the depth
            // and payload kernels by construction. WGSL f32→i32 SATURATES, so
            // far off-screen verts read as a huge extent → HW route.
            const xi0 = toI(s0.x.mul(256).round()).toVar();
            const yi0 = toI(s0.y.mul(256).round()).toVar();
            const xi1 = toI(s1.x.mul(256).round()).toVar();
            const yi1 = toI(s1.y.mul(256).round()).toVar();
            const xi2 = toI(s2.x.mul(256).round()).toVar();
            const yi2 = toI(s2.y.mul(256).round()).toVar();

            // whole-pixel bbox (trunc-div ≠ floor only below 0, where start
            // clamps to 0 / validBB rejects — harmless)
            const bbMinX = minI(xi0, minI(xi1, xi2)).div(toI(256)).toVar();
            const bbMaxX = maxI(xi0, maxI(xi1, xi2)).div(toI(256)).toVar();
            const bbMinY = minI(yi0, minI(yi1, yi2)).div(toI(256)).toVar();
            const bbMaxY = maxI(yi0, maxI(yi1, yi2)).div(toI(256)).toVar();

            // SW only when the UNCLAMPED extent is small (the float core sized
            // the clamped box — a screen-spanning tri with a 16-px on-screen
            // sliver ran SW with unbounded edge terms; now it routes HW).
            // Bound: deltas ≤ ~17 px ⇒ every edge term < 2^26, i32-safe.
            const smallEnough = bbMaxX
              .sub(bbMinX)
              .lessThanEqual(toI(MAX_RASTER_SIZE))
              .and(bbMaxY.sub(bbMinY).lessThanEqual(toI(MAX_RASTER_SIZE)));

            const startX = maxI(toI(0), bbMinX).toVar();
            const endX = minI(toI(width - 1), bbMaxX).toVar();
            const startY = maxI(toI(0), bbMinY).toVar();
            const endY = minI(toI(height - 1), bbMaxY).toVar();
            const validBB = startX
              .lessThanEqual(endX)
              .and(startY.lessThanEqual(endY));

            If(
              (smallEnough as unknown as { and: (o: unknown) => NB }).and(
                validBB,
              ),
              () => {
                // integer twice-area; snapping can collapse/flip a sub-1/256-px
                // sliver → skip (also guards the reciprocal — the float core
                // divided blindly)
                const area2 = yi2
                  .sub(yi0)
                  .mul(xi1.sub(xi0))
                  .sub(xi2.sub(xi0).mul(yi1.sub(yi0)))
                  .toVar();
                // SAMPLE-MISS cull (CuRast): skip a small tri whose snapped extent contains NO pixel
                // CENTRE (k·256+128 grid) on x OR y ⇒ it covers no sample ⇒ the scanline rasters
                // nothing. Conservative + exact (the scanline samples at the same centres) ⇒ zero
                // visible quality loss. Float ceil handles negative/off-screen xiMin. (Kept as
                // leverage: ~neutral on world1's already-≤1px cut, a building block / helps the tiled.)
                const xiMin = minI(xi0, minI(xi1, xi2));
                const xiMax = maxI(xi0, maxI(xi1, xi2));
                const yiMin = minI(yi0, minI(yi1, yi2));
                const yiMax = maxI(yi0, maxI(yi1, yi2));
                const firstCx = toF(xiMin.sub(toI(128)) as unknown as NI)
                  .div(256)
                  .ceil()
                  .mul(256)
                  .add(128);
                const firstCy = toF(yiMin.sub(toI(128)) as unknown as NI)
                  .div(256)
                  .ceil()
                  .mul(256)
                  .add(128);
                const coversSample = firstCx
                  .lessThanEqual(toF(xiMax as unknown as NI))
                  .and(firstCy.lessThanEqual(toF(yiMax as unknown as NI)));
                // W2 ?trihzb: conservative per-tri occlusion — nearest tri z vs the
                // prev-frame HZB farthest over the bbox (2×2 window, 32-px texels
                // cover the ≤17-px bbox), read from the hwQueue TAIL MIRROR (binding
                // budget — see TRIHZB_BASE). Positive-f32 bits compare monotonically
                // as u32, so no bitcast needed. Tail 1.0f-filled ⇒ frame-0 and
                // sky-backed tris never reject. world1 only (depth/shadow untouched).
                let triVis: NB | null = null;
                if (mode === 'world1' && triLvls) {
                  const nearBits = bcF2U(
                    ndc0.z.min(ndc1.z).min(ndc2.z).clamp(0, 1) as unknown as NF,
                  ).toVar();
                  // finest mirrored level whose 2×2 window covers the bbox:
                  // pitch(k) = 2^(k+1) px; need extent < 2·pitch ⇒ k = max(0,
                  // ceil(log2(e+1)) − 2), clamped to TRI_HZB_MAX.
                  const ext = maxI(endX.sub(startX), endY.sub(startY)).toVar();
                  const lvl = uint(
                    toF(ext as unknown as NI)
                      .add(1)
                      .max(1)
                      .log2()
                      .ceil()
                      .sub(2)
                      .clamp(0, TRI_HZB_MAX),
                  ).toVar();
                  // per-level (dstOff, width) via select chains (5 static levels)
                  let offSel: NU = uint(
                    TRIHZB_BASE + (triDst[0] ?? 0),
                  ) as unknown as NU;
                  let wSel: NU = uint(triLvls[0]?.w ?? 1) as unknown as NU;
                  for (let k = 1; k <= TRI_HZB_MAX; k++) {
                    const isK = lvl.equal(uint(k));
                    offSel = isK.select(
                      uint(TRIHZB_BASE + (triDst[k] ?? 0)),
                      offSel,
                    ) as unknown as NU;
                    wSel = isK.select(
                      uint(triLvls[k]?.w ?? 1),
                      wSel,
                    ) as unknown as NU;
                  }
                  const shift = lvl.add(uint(1));
                  const lo = offSel;
                  const lw = wSel;
                  const tx0 = uint(startX).shiftRight(shift).toVar();
                  const ty0 = uint(startY).shiftRight(shift).toVar();
                  const tx1 = uint(endX).shiftRight(shift).toVar();
                  const ty1 = uint(endY).shiftRight(shift).toVar();
                  const z00 = aLoadU(
                    hwQueueV.atomic.element(lo.add(ty0.mul(lw)).add(tx0)),
                  );
                  const z01 = aLoadU(
                    hwQueueV.atomic.element(lo.add(ty0.mul(lw)).add(tx1)),
                  );
                  const z10 = aLoadU(
                    hwQueueV.atomic.element(lo.add(ty1.mul(lw)).add(tx0)),
                  );
                  const z11 = aLoadU(
                    hwQueueV.atomic.element(lo.add(ty1.mul(lw)).add(tx1)),
                  );
                  const farBits = maxU(maxU(z00, z01), maxU(z10, z11));
                  triVis = nearBits.lessThanEqual(farBits) as unknown as NB;
                }
                const rasterGate = triVis
                  ? area2.greaterThan(toI(0)).and(coversSample).and(triVis)
                  : area2.greaterThan(toI(0)).and(coversSample);
                If(rasterGate, () => {
                  // edge i is opposite vertex i; ex/ey = dE per +1 UNIT (1/256 px)
                  const ex0 = yi1.sub(yi2).toVar();
                  const ey0 = xi2.sub(xi1).toVar();
                  const ex1 = yi2.sub(yi0).toVar();
                  const ey1 = xi0.sub(xi2).toVar();
                  const ex2 = yi0.sub(yi1).toVar();
                  const ey2 = xi1.sub(xi0).toVar();

                  // top-left rule, same orientation convention as the float core:
                  // the boundary E == 0 is owned iff dE/dx < 0, or dE/dx == 0 and
                  // dE/dy > 0; the -1 bias turns ≥0 into >0 on unowned edges —
                  // exact and scale-free (the -1e-5 float bias competed with ulp
                  // at edge-term scale ~1e6)
                  const tlBias = (ex: NI, ey: NI): NI =>
                    ex
                      .lessThan(toI(0))
                      .or(ex.equal(toI(0)).and(ey.greaterThan(toI(0))))
                      .select(toI(0), toI(-1)) as unknown as NI;
                  const bias0 = tlBias(
                    ex0 as unknown as NI,
                    ey0 as unknown as NI,
                  );
                  const bias1 = tlBias(
                    ex1 as unknown as NI,
                    ey1 as unknown as NI,
                  );
                  const bias2 = tlBias(
                    ex2 as unknown as NI,
                    ey2 as unknown as NI,
                  );

                  // E at the center of (startX, startY), in units²
                  const pcx = startX.mul(toI(256)).add(toI(128)).toVar();
                  const pcy = startY.mul(toI(256)).add(toI(128)).toVar();
                  const rw0 = pcy
                    .sub(yi1)
                    .mul(xi2.sub(xi1))
                    .sub(pcx.sub(xi1).mul(yi2.sub(yi1)))
                    .add(bias0)
                    .toVar();
                  const rw1 = pcy
                    .sub(yi2)
                    .mul(xi0.sub(xi2))
                    .sub(pcx.sub(xi2).mul(yi0.sub(yi2)))
                    .add(bias1)
                    .toVar();
                  const rw2 = pcy
                    .sub(yi0)
                    .mul(xi1.sub(xi0))
                    .sub(pcx.sub(xi0).mul(yi1.sub(yi0)))
                    .add(bias2)
                    .toVar();
                  // per-PIXEL steps = ex/ey × 256 units
                  const sx0 = ex0.mul(toI(256)).toVar();
                  const sx1 = ex1.mul(toI(256)).toVar();
                  const sx2 = ex2.mul(toI(256)).toVar();
                  const sy0 = ey0.mul(toI(256)).toVar();
                  const sy1 = ey1.mul(toI(256)).toVar();
                  const sy2 = ey2.mul(toI(256)).toVar();
                  const rcpArea = float(1)
                    .div(toF(area2 as unknown as NI))
                    .toVar();

                  if ((mode === 'depth' || mode === 'world1') && rdbg === 2) {
                    // ?rdbg=2 — stop right before the scanline loop. The sink folds
                    // the edge-setup vars (rcpArea, rw0..2, the per-pixel steps) so
                    // none are sunk past / DCE'd, forcing the full per-triangle
                    // setup to be timed. (rdbg2 − rdbg1) = near/backface + ndc +
                    // fixed-point snap + bbox + edge setup + the 3× fetchWorldVert
                    // (world1's rdbg=1 stops BEFORE any corner fetch); (rdbg3 − rdbg2)
                    // = the per-pixel loop (coverage + depth interp + the election
                    // atomicMax/atomicStore). Shared by the depth + world1 kernels —
                    // only the rdbg-gated CONFIG that's actually dispatched matters.
                    const sinkV = rcpArea
                      .add(toF(rw0 as unknown as NI))
                      .add(toF(rw1 as unknown as NI))
                      .add(toF(rw2 as unknown as NI))
                      .add(toF(sx0 as unknown as NI))
                      .add(toF(sy0 as unknown as NI))
                      .clamp(0, 1);
                    const sinkPx = uint(startY)
                      .mul(uint(cam.uW))
                      .add(uint(startX));
                    atomicMin(
                      visDepthV.atomic.element(sinkPx),
                      bcF2U(sinkV as unknown as NF),
                    );
                    returnIf(itemCount.greaterThanEqual(uint(0)));
                  }

                  // OLD scanline (verbatim) as an emit closure: swcoop=0 emits it alone;
                  // swcoop=2 emits it as the LARGE-bin branch. Same code, same point.
                  const oldScanline = (): void => {
                    loopI(
                      'sy',
                      startY as unknown as NI,
                      endY as unknown as NI,
                      (y) => {
                        const cw0 = rw0.toVar();
                        const cw1 = rw1.toVar();
                        const cw2 = rw2.toVar();
                        // PERF (UE5-gap win 1) — SCANLINE x-span. The edge value at column x is
                        // cw_i(x) = cw_i + (x−startX)·sx_i; the row is covered where all 3 are ≥0.
                        // Solve each edge for the crossing x = startX − cw_i/sx_i (÷-by-0 guarded),
                        // floor/ceil + ±1 pad so the span is a guaranteed SUPERSET (f32-safe). The
                        // EXACT per-pixel cw≥0 test below still decides coverage — this only fast-
                        // skips the ~half of the AABB that is provably outside the triangle. Bit-
                        // identical (skipped pixels all fail the test).
                        const den0 = sx0
                          .equal(toI(0))
                          .select(
                            toI(1),
                            sx0 as unknown as NI,
                          ) as unknown as NI;
                        const den1 = sx1
                          .equal(toI(0))
                          .select(
                            toI(1),
                            sx1 as unknown as NI,
                          ) as unknown as NI;
                        const den2 = sx2
                          .equal(toI(0))
                          .select(
                            toI(1),
                            sx2 as unknown as NI,
                          ) as unknown as NI;
                        const xc0 = toF(startX).sub(
                          toF(cw0 as unknown as NI).div(toF(den0)),
                        );
                        const xc1 = toF(startX).sub(
                          toF(cw1 as unknown as NI).div(toF(den1)),
                        );
                        const xc2 = toF(startX).sub(
                          toF(cw2 as unknown as NI).div(toF(den2)),
                        );
                        const lo0 = sx0
                          .greaterThan(toI(0))
                          .select(
                            toI(xc0.floor().sub(float(1))),
                            startX,
                          ) as unknown as NI;
                        const lo1 = sx1
                          .greaterThan(toI(0))
                          .select(
                            toI(xc1.floor().sub(float(1))),
                            startX,
                          ) as unknown as NI;
                        const lo2 = sx2
                          .greaterThan(toI(0))
                          .select(
                            toI(xc2.floor().sub(float(1))),
                            startX,
                          ) as unknown as NI;
                        const hi0 = sx0
                          .lessThan(toI(0))
                          .select(
                            toI(xc0.ceil().add(float(1))),
                            endX,
                          ) as unknown as NI;
                        const hi1 = sx1
                          .lessThan(toI(0))
                          .select(
                            toI(xc1.ceil().add(float(1))),
                            endX,
                          ) as unknown as NI;
                        const hi2 = sx2
                          .lessThan(toI(0))
                          .select(
                            toI(xc2.ceil().add(float(1))),
                            endX,
                          ) as unknown as NI;
                        // a zero-slope edge that is already negative ⇒ the whole row is empty
                        const emptyRow = sx0
                          .equal(toI(0))
                          .and(cw0.lessThan(toI(0)))
                          .or(sx1.equal(toI(0)).and(cw1.lessThan(toI(0))))
                          .or(sx2.equal(toI(0)).and(cw2.lessThan(toI(0))));
                        const xLo = maxI(
                          maxI(maxI(startX, lo0), lo1),
                          lo2,
                        ).toVar();
                        const xHi = minI(
                          minI(minI(endX, hi0), hi1),
                          hi2,
                        ).toVar();
                        xHi.assign(
                          emptyRow.select(
                            xLo.sub(toI(1)),
                            xHi,
                          ) as unknown as NI,
                        );
                        // advance the incremental edge values from startX to xLo
                        const dxL = xLo.sub(startX).toVar();
                        cw0.addAssign(dxL.mul(sx0));
                        cw1.addAssign(dxL.mul(sx1));
                        cw2.addAssign(dxL.mul(sx2));
                        loopI(
                          'sx',
                          xLo as unknown as NI,
                          xHi as unknown as NI,
                          (x) => {
                            If(
                              cw0
                                .greaterThanEqual(toI(0))
                                .and(cw1.greaterThanEqual(toI(0)))
                                .and(cw2.greaterThanEqual(toI(0))),
                              () => {
                                // depth from the UNBIASED integer weights: the top-left
                                // −1 biases belong to COVERAGE only. Folding them into
                                // the weights divides by area2 while the weights sum to
                                // area2−(1..3) — a RELATIVE error of ~bias/area2 that is
                                // ulp-level on big triangles but ~5e-4 on sub-pixel far
                                // slivers (area2 ~10³ units²) ⇒ far depth biased NEARER
                                // by hundreds of meters. Self-consistent across passes
                                // (audit/parity blind) — found at N4-C0 when WATER
                                // depth-tested against the buffer. Unbiased weights sum
                                // to area2 exactly (integer identity): cz is exact, and
                                // both passes still compute identical bits.
                                const emitFrag = (): void => {
                                  const uw0 = cw0.sub(bias0).toVar();
                                  const uw1 = cw1.sub(bias1).toVar();
                                  const uw2 = cw2.sub(bias2).toVar();
                                  const cz = toF(uw0 as unknown as NI)
                                    .mul(ndc0.z)
                                    .add(toF(uw1 as unknown as NI).mul(ndc1.z))
                                    .add(toF(uw2 as unknown as NI).mul(ndc2.z))
                                    .mul(rcpArea)
                                    .toVar();
                                  If(
                                    cz
                                      .greaterThanEqual(0)
                                      .and(cz.lessThanEqual(1)),
                                    () => {
                                      const px = uint(y)
                                        .mul(uint(cam.uW))
                                        .add(uint(x));
                                      const bits = bcF2U(cz as unknown as NF);
                                      if (mode === 'depth') {
                                        const cur = aLoadU(
                                          visDepthV.atomic.element(px),
                                        );
                                        If(bits.lessThan(cur), () => {
                                          atomicMin(
                                            visDepthV.atomic.element(px),
                                            bits,
                                          );
                                        });
                                      } else if (mode === 'combined') {
                                        // RACE-FREE packed vis buffer, NO separate depthV (stays under
                                        // the storage-buffer ceiling): id packed as (depthKey16<<16 |
                                        // idPart) into visA(payload)+visB via atomicMax. Depth lives in
                                        // the high bits ⇒ the nearest fragment wins BOTH buffers ⇒
                                        // depth+id can NEVER desync (kills "branch through trunk" for
                                        // any separated depths; only truly equal-Z coplanar tris can
                                        // still hybridise — rare). The HZB reads visA's key (packed).
                                        const myDk = depthKey16(
                                          cz as unknown as NF,
                                        ).toVar();
                                        const stored = aLoadU(
                                          visPayloadV.atomic.element(px),
                                        );
                                        If(
                                          myDk.greaterThan(
                                            stored.shiftRight(uint(16)),
                                          ),
                                          () => {
                                            const dk = myDk.shiftLeft(uint(16));
                                            atomicMax(
                                              visPayloadV.atomic.element(px),
                                              dk.bitOr(
                                                payload.bitAnd(uint(0xffff)),
                                              ),
                                            );
                                            atomicMax(
                                              visBV.atomic.element(px),
                                              dk.bitOr(
                                                payload
                                                  .shiftRight(uint(16))
                                                  .bitAnd(uint(0xffff)),
                                              ),
                                            );
                                          },
                                        );
                                      } else if (mode === 'world1') {
                                        // 0a SCAR (?scar=1): count this covered fragment. [2] = ALL
                                        // covered fragments (the band's fragment-share denominator);
                                        // [0] = band leaf fragments (the OVERDRAW numerator, gated on
                                        // the runtime per-cluster `bandFlag`). Both are pure atomic adds
                                        // gated on `scar` — the election below is BYTE-IDENTICAL whether
                                        // or not scar is on. This is exactly the overdraw signal §6.0
                                        // needs disproven: band fragments / band covered-pixels (the
                                        // latter from kScarCovered) is the per-pixel triangle overdraw
                                        // the voxel bin must undercut.
                                        if (scar) {
                                          atomicAdd(scarEl(2), uint(1));
                                          if (bandFlag) {
                                            If(bandFlag, () => {
                                              atomicAdd(scarEl(0), uint(1));
                                            });
                                          }
                                        }
                                        // PERF-VB4 single pass: EXACT f32 depth (atomicMin → HZB +
                                        // shadows + the resolve's wp reconstruction stay exact and
                                        // unchanged) AND a coherent id election. Depth16 in the high
                                        // bits of an atomicMax election (visPayloadV) ⇒ the nearest
                                        // fragment wins with no branch-through-trunk; the election
                                        // WINNER plain-stores the FULL 25-bit id into ONE side buffer
                                        // (visBV). A depth16 tie degrades to a valid-but-maybe-wrong
                                        // cluster (sparse wrong-material speckle), never a torn/garbage
                                        // id (the franksteining the idLo/idHi split would cause).
                                        const cand = depthKey24(
                                          cz as unknown as NF,
                                        )
                                          .shiftLeft(uint(8))
                                          .bitOr(payload.bitAnd(uint(0xff)))
                                          .toVar();
                                        // election WINNER stores the FULL 25-bit id into the single side
                                        // buffer (no franksteining). NO depthV write: a 3rd atomic storage
                                        // buffer in this kernel is the 3× cliff (15-17 ms) AND breaks its
                                        // writes — so the resolve takes the 16-bit depth from the election
                                        // key (visPayloadV high bits) instead of an exact depthV. (?relect
                                        // routes this through doElection — pristine when relect===1.)
                                        doElection(px, cand, payload);
                                      }
                                    },
                                  );
                                };
                                emitFrag();
                              },
                            );
                            cw0.addAssign(sx0);
                            cw1.addAssign(sx1);
                            cw2.addAssign(sx2);
                          },
                        );
                        rw0.addAssign(sy0);
                        rw1.addAssign(sy1);
                        rw2.addAssign(sy2);
                      },
                    );
                  };
                  if (coopMode === 0) {
                    oldScanline();
                  } else {
                    // ── SWCOOP two-bin split (replaces the unconditional row-span
                    // scanline). Coverage/depth/election are decided by the SAME math
                    // as the old loop (bit-identity law) — only work distribution
                    // changes. Small bin is lane-private in BOTH modes; the large bin
                    // is cooperative (mode 1) or the old scanline (mode 2).
                    const extW = endX.sub(startX).toVar();
                    const extH = endY.sub(startY).toVar();
                    If(
                      extW
                        .lessThanEqual(toI(SW_SMALL_EXT))
                        .and(extH.lessThanEqual(toI(SW_SMALL_EXT))),
                      () => {
                        // SMALL bin (≤5×5 candidates): lane-private full-bbox walk with
                        // the incremental fixed-point edge tests. The old per-row x-span
                        // solve (3 fp32 divides + ~50 ALU ≈ 10 walked pixels PER ROW) is
                        // dropped — at this size it costs more than it skips; the exact
                        // per-pixel cw≥0 test below is what decides emission either way.
                        loopI(
                          'cy',
                          startY as unknown as NI,
                          endY as unknown as NI,
                          (y) => {
                            const cw0 = rw0.toVar();
                            const cw1 = rw1.toVar();
                            const cw2 = rw2.toVar();
                            loopI(
                              'cx',
                              startX as unknown as NI,
                              endX as unknown as NI,
                              (x) => {
                                If(
                                  cw0
                                    .greaterThanEqual(toI(0))
                                    .and(cw1.greaterThanEqual(toI(0)))
                                    .and(cw2.greaterThanEqual(toI(0))),
                                  () => {
                                    const uw0 = cw0.sub(bias0).toVar();
                                    const uw1 = cw1.sub(bias1).toVar();
                                    const uw2 = cw2.sub(bias2).toVar();
                                    const px = uint(y)
                                      .mul(uint(cam.uW))
                                      .add(uint(x))
                                      .toVar();
                                    emitW1!(
                                      px,
                                      uw0,
                                      uw1,
                                      uw2,
                                      ndc0.z as unknown as NF,
                                      ndc1.z as unknown as NF,
                                      ndc2.z as unknown as NF,
                                      rcpArea,
                                      payload,
                                    );
                                  },
                                );
                                cw0.addAssign(sx0);
                                cw1.addAssign(sx1);
                                cw2.addAssign(sx2);
                              },
                            );
                            rw0.addAssign(sy0);
                            rw1.addAssign(sy1);
                            rw2.addAssign(sy2);
                          },
                        );
                      },
                    ).Else(() => {
                      if (split) {
                        // mode 2: the LARGE bin keeps the OLD lane-private row-solve
                        // scanline verbatim — no shared memory, no barriers.
                        oldScanline();
                        return;
                      }
                      // mode 1 LARGE bin (extent 5..16 px): append this tri's raster
                      // record into shared slot [localTri]; Phase B (below, after the
                      // barrier) strides its pixels across all 128 lanes.
                      const s = swSh!;
                      wgW(s.rw0, localTri, rw0);
                      wgW(s.rw1, localTri, rw1);
                      wgW(s.rw2, localTri, rw2);
                      // per-UNIT steps packed as (ey<<16 | ex&0xffff) i16 pairs
                      const packEdge = (ex: NI, ey: NI): NU =>
                        uint(ex.bitAnd(toI(0xffff))).bitOr(
                          uint(ey.bitAnd(toI(0xffff))).shiftLeft(uint(16)),
                        );
                      wgW(
                        s.exy0,
                        localTri,
                        packEdge(ex0 as unknown as NI, ey0 as unknown as NI),
                      );
                      wgW(
                        s.exy1,
                        localTri,
                        packEdge(ex1 as unknown as NI, ey1 as unknown as NI),
                      );
                      wgW(
                        s.exy2,
                        localTri,
                        packEdge(ex2 as unknown as NI, ey2 as unknown as NI),
                      );
                      wgW(
                        s.xy,
                        localTri,
                        uint(startX).bitOr(uint(startY).shiftLeft(uint(16))),
                      );
                      // bias bits (1 ⇔ old bias −1; Phase B unbiases via uw = e + bit)
                      const bb0 = bias0
                        .lessThan(toI(0))
                        .select(uint(1), uint(0)) as unknown as NU;
                      const bb1 = bias1
                        .lessThan(toI(0))
                        .select(uint(1), uint(0)) as unknown as NU;
                      const bb2 = bias2
                        .lessThan(toI(0))
                        .select(uint(1), uint(0)) as unknown as NU;
                      wgW(
                        s.wh,
                        localTri,
                        uint(extW.add(toI(1)))
                          .bitOr(uint(extH.add(toI(1))).shiftLeft(uint(8)))
                          .bitOr(bb0.shiftLeft(uint(16)))
                          .bitOr(bb1.shiftLeft(uint(17)))
                          .bitOr(bb2.shiftLeft(uint(18))),
                      );
                      wgW(s.pay, localTri, payload);
                      wgW(s.z0, localTri, ndc0.z);
                      wgW(s.z1, localTri, ndc1.z);
                      wgW(s.z2, localTri, ndc2.z);
                      wgW(s.rcp, localTri, rcpArea);
                    });
                  }
                });
              },
            ).Else(() => {
              If(validBB, () => {
                // big triangle → HW queue
                const slot = atomicAdd(
                  hwQueueV.atomic.element(0),
                  uint(1),
                ) as unknown as NU;
                If(slot.lessThan(uint(HW_CAP)), () => {
                  const base = slot.mul(uint(2)).add(uint(1));
                  atomicStore(hwQueueV.atomic.element(base), payload);
                  atomicStore(
                    hwQueueV.atomic.element(base.add(uint(1))),
                    instId,
                  );
                });
              });
            });
          });
        });
      });

      if (coop && swSh && emitW1) {
        // ── SWCOOP Phase B (the kVoxScatter cooperative-footprint idiom,
        // NaniteVoxelRaster.ts:1121): loop the shared LARGE-bin slots; WITHIN each
        // tri all 128 lanes stride its bbox pixels together (pixelIndex = lane,
        // lane+128, …; worst case 17×17 = 289 px = 3 strides/lane), each lane
        // evaluating the 3 edge functions DIRECTLY from the stored coefficients
        // (3 i32 mul-adds each — no incremental state). No lane-private nested walk
        // ⇒ a subgroup never runs the UNION of its worst lanes' bboxes. Empty slots
        // (small/HW/rejected/voxel/idle) have wh=0 ⇒ area 0 ⇒ zero-trip inner loop.
        // The barrier is at KERNEL TOP SCOPE: every returnIf above it is driven by
        // read-only-storage values (uniform to naga — the wgcache-barrier precedent);
        // the storage-derived voxel skip became a guard, not a return.
        workgroupBarrier(); // Phase A records complete before the compaction scan
        // thread-0 COMPACTION: serial scan of the sparse slots into a dense list +
        // count (no workgroup atomics in r184 TSL — WorkgroupInfoNode has no atomic
        // element type, so an atomicAdd append can't be expressed; a single-lane
        // scan is ≤128 shared loads per workgroup, latency-hidden by resident
        // neighbours). Both barriers are at kernel top scope (uniform).
        If(localTri.equal(uint(0)), () => {
          const n = uint(0).toVar();
          loopUN(
            'swsc',
            uint(0),
            minU(ctx.triCount, uint(MAX_CLUSTER_TRIS)),
            (slot) => {
              const wv = swSh.wh.element(slot) as unknown as NU;
              If(wv.bitAnd(uint(0xff)).notEqual(uint(0)), () => {
                wgW(swSh.list, n, slot);
                n.addAssign(uint(1));
              });
            },
          );
          wgW(swSh.count, uint(0), n);
        });
        workgroupBarrier(); // compacted list visible to all lanes
        const laneB = localX();
        const nLarge = (swSh.count.element(uint(0)) as unknown as NU).toVar();
        loopUN('swct', uint(0), nLarge, (li) => {
          const slot = (swSh.list.element(li) as unknown as NU).toVar();
          const whW = (swSh.wh.element(slot) as unknown as NU).toVar();
          const w = whW.bitAnd(uint(0xff)).toVar();
          {
            const h = whW.shiftRight(uint(8)).bitAnd(uint(0xff)).toVar();
            const area = w.mul(h).toVar();
            const xyW = (swSh.xy.element(slot) as unknown as NU).toVar();
            const x0 = xyW.bitAnd(uint(0xffff)).toVar();
            const y0 = xyW.shiftRight(uint(16)).toVar();
            const bRw0 = (swSh.rw0.element(slot) as unknown as NI).toVar();
            const bRw1 = (swSh.rw1.element(slot) as unknown as NI).toVar();
            const bRw2 = (swSh.rw2.element(slot) as unknown as NI).toVar();
            // unpack the i16 (ex, ey) pair and rebuild the PER-PIXEL steps ×256 —
            // identical i32 values to the old setup's sx/sy (sign-extend is exact).
            const exyW0 = (swSh.exy0.element(slot) as unknown as NU).toVar();
            const exyW1 = (swSh.exy1.element(slot) as unknown as NU).toVar();
            const exyW2 = (swSh.exy2.element(slot) as unknown as NU).toVar();
            const sExt = (w16: NU): NI =>
              bcU2I(w16.shiftLeft(uint(16))).shiftRight(
                toI(16),
              ) as unknown as NI;
            const bSx0 = sExt(exyW0.bitAnd(uint(0xffff)))
              .mul(toI(256))
              .toVar();
            const bSx1 = sExt(exyW1.bitAnd(uint(0xffff)))
              .mul(toI(256))
              .toVar();
            const bSx2 = sExt(exyW2.bitAnd(uint(0xffff)))
              .mul(toI(256))
              .toVar();
            const bSy0 = sExt(exyW0.shiftRight(uint(16)))
              .mul(toI(256))
              .toVar();
            const bSy1 = sExt(exyW1.shiftRight(uint(16)))
              .mul(toI(256))
              .toVar();
            const bSy2 = sExt(exyW2.shiftRight(uint(16)))
              .mul(toI(256))
              .toVar();
            const bPay = (swSh.pay.element(slot) as unknown as NU).toVar();
            const bZ0 = (swSh.z0.element(slot) as unknown as NF).toVar();
            const bZ1 = (swSh.z1.element(slot) as unknown as NF).toVar();
            const bZ2 = (swSh.z2.element(slot) as unknown as NF).toVar();
            const bRcp = (swSh.rcp.element(slot) as unknown as NF).toVar();
            // bias bits (1 ⇔ old bias −1): uw = cw − bias = e + bit
            const bB0 = toI(
              whW.shiftRight(uint(16)).bitAnd(uint(1)) as unknown as NF,
            ).toVar();
            const bB1 = toI(
              whW.shiftRight(uint(17)).bitAnd(uint(1)) as unknown as NF,
            ).toVar();
            const bB2 = toI(
              whW.shiftRight(uint(18)).bitAnd(uint(1)) as unknown as NF,
            ).toVar();
            loopU(
              laneB,
              area,
              (lp) => {
                const ly = lp.div(w).toVar();
                const lx = lp.sub(ly.mul(w)).toVar();
                const dx = toI(lx as unknown as NF).toVar();
                const dy = toI(ly as unknown as NF).toVar();
                // edge values DIRECTLY from coefficients — identical i32 results to
                // the old loop's incremental adds at the same pixel (exact integers).
                const e0 = bRw0.add(dx.mul(bSx0)).add(dy.mul(bSy0)).toVar();
                const e1 = bRw1.add(dx.mul(bSx1)).add(dy.mul(bSy1)).toVar();
                const e2 = bRw2.add(dx.mul(bSx2)).add(dy.mul(bSy2)).toVar();
                If(
                  e0
                    .greaterThanEqual(toI(0))
                    .and(e1.greaterThanEqual(toI(0)))
                    .and(e2.greaterThanEqual(toI(0))),
                  () => {
                    const uw0 = e0.add(bB0).toVar();
                    const uw1 = e1.add(bB1).toVar();
                    const uw2 = e2.add(bB2).toVar();
                    const px = y0
                      .add(ly)
                      .mul(uint(cam.uW))
                      .add(x0.add(lx))
                      .toVar();
                    emitW1(px, uw0, uw1, uw2, bZ0, bZ1, bZ2, bRcp, bPay);
                  },
                );
              },
              MAX_CLUSTER_TRIS,
            );
          }
        });
      }

      // ?relect (world1 0/2) — kernel-end election-key sink: ONE atomicMax per raster
      // thread (lanes with a real triangle) at a hashed px into visBV (already bound;
      // unread under relect≠1 since the election that fed it is gone). This keeps the
      // whole pixel loop live (elecSink ← every covered fragment's key ⇒ no DCE) with
      // NO per-fragment election atomics — the same one-atomic-per-thread cost the rdbg
      // sinks pay, so the split reads clean against the rdbg2 floor.
      if (mode === 'world1' && relect !== 1 && elecSink) {
        If(localTri.lessThan(ctx.triCount), () => {
          const sinkPx = itemIdx
            .mul(uint(2654435761))
            .add(localTri)
            .mod(uint(pixelCount));
          atomicMax(visBV.atomic.element(sinkPx), elecSink);
        });
      }
    })().compute(QRASTER_CAP * MAX_CLUSTER_TRIS, [MAX_CLUSTER_TRIS]);
    return kn;
  };

  // SHADOW depth-only pass (atomicMin Z into the shadow vis buffer).
  const kRasterDepth = rasterKernel('depth');
  (kRasterDepth as ComputeKernel).setName('nanRasterDepth');
  // NaniteView debug single-pass (packed 16-bit Z + idLo/idHi split).
  const kRasterCombined = rasterKernel('combined');
  (kRasterCombined as ComputeKernel).setName('nanRasterCombined');
  // PERF-VB4 WORLD single pass: 24-bit Z election + full-id side buffer (visBV).
  const kRasterWorld1 = rasterKernel('world1');
  (kRasterWorld1 as ComputeKernel).setName('nanRasterWorld1');
  // SUBMIT-COALESCE enabler (item 3): tag kRasterWorld1 with its full-range indirect
  // attr so the batched world1 below dispatches it at its tight rasterDispatchFull size
  // instead of the baked QRASTER_CAP×MAX_CLUSTER_TRIS (~1B-thread) grid.
  setIndirectDispatch(kRasterWorld1, cull.rasterDispatchFullAttr);

  // ---- kHwArgs ----------------------------------------------------------------------
  const kHwArgs = Fn(() => {
    const n = minU(aLoadU(hwQueueV.atomic.element(0)), uint(HW_CAP));
    hwDrawBuf.element(0).assign(n.mul(uint(3)));
    hwDrawBuf.element(1).assign(uint(1));
    hwDrawBuf.element(2).assign(uint(0));
    hwDrawBuf.element(3).assign(uint(0));
  })().compute(1, [1]);
  (kHwArgs as unknown as ComputeKernel).setName('nanHwArgs');

  // ---- kTriHzbCopy (W2 ?trihzb) — mirror one pyramid level into the hwQueue tail.
  // Own 2-buffer pipeline (pyramid ro + hwQueue) dispatched right after the HZB
  // build; world1 then reads the mirror through its EXISTING hwQueue binding.
  const kTriHzbCopy =
    triLvls && triHzbRO
      ? (() => {
          const kn = Fn(() => {
            If(instanceIndex.lessThan(uint(triTailN)), () => {
              // map flat tail index → (level, src pyramid index) via static ranges
              const idx = instanceIndex;
              const src = uint(0).toVar();
              for (let k = 0; k <= TRI_HZB_MAX; k++) {
                const l = triLvls[k];
                const d0 = triDst[k] ?? 0;
                if (!l) continue;
                const inK = idx
                  .greaterThanEqual(uint(d0))
                  .and(idx.lessThan(uint(d0 + l.w * l.h)));
                src.assign(
                  inK.select(uint(l.offset).add(idx.sub(uint(d0))), src),
                );
              }
              const v = triHzbRO.element(src);
              atomicStore(
                hwQueueV.atomic.element(uint(TRIHZB_BASE).add(idx)),
                bcF2U(v as unknown as NF),
              );
            });
          })().compute(triTailN, [64]);
          (kn as unknown as ComputeKernel).setName('nanTriHzbCopy');
          return kn;
        })()
      : null;

  // ---- kAudit (?audit=1) — run AFTER the payload passes: a covered pixel
  // whose payload is still the clear sentinel means no pass-2 writer ever
  // reproduced the stored depth (raster pass inconsistency). Gate: 0.
  const kAudit = Fn(() => {
    returnIf(instanceIndex.greaterThanEqual(uint(pixelCount)));
    const d = elemU(visDepthV.ro, instanceIndex);
    If(d.notEqual(uint(0xffffffff)), () => {
      atomicAdd(auditV.atomic.element(1), uint(1));
      const p = elemU(visPayloadV.ro, instanceIndex);
      If(p.equal(uint(0xffffffff)), () => {
        atomicAdd(auditV.atomic.element(0), uint(1));
      });
    });
  })().compute(pixelCount, [256]);
  (kAudit as unknown as ComputeKernel).setName('nanAudit');

  // ---- kScarCovered (?scar=1) — the 0a OVERDRAW DENOMINATOR -------------------------
  // Runs AFTER world1: one thread per pixel. Decodes the pixel's FINAL winner (the same
  // visBV→qRaster→cluster→mesh chain the resolve uses, NaniteResolve.ts:283-290) and, if
  // it is a BAND leaf cluster (matClass==4 + its instance in [scarNear, scarFar]), counts
  // exactly ONE covered pixel into scarV[1]. So scarV[1] is the EXACT count of unique
  // pixels finally owned by the mid/far foliage band, and overdraw = scarV[0]/scarV[1]
  // (band fragments rasterized per band pixel covered). The band test mirrors the raster
  // side (instance-A distance), so numerator and denominator share one definition.
  const kScarCovered = Fn(() => {
    returnIf(instanceIndex.greaterThanEqual(uint(pixelCount)));
    // world1/single-pass covered-test: election anchor 0 = cleared (no fragment).
    const elect = elemU(visPayloadV.ro, instanceIndex);
    If(elect.notEqual(uint(0)), () => {
      const pRaw = elemU(visBV.ro, instanceIndex);
      const itemIdx = pRaw.shiftRight(uint(CLUSTER_TRI_BITS));
      const item = qRasterRO.element(itemIdx.add(uint(1)));
      const instId = item.x;
      const ci = item.y;
      const meshId = elemU(
        gpu.clusters,
        ci.mul(uint(CLUSTER_WORDS)).add(uint(7)),
      ).shiftRight(uint(16));
      const matClass = elemU(
        gpu.meshes,
        meshId.mul(uint(MESH_WORDS)).add(uint(6)),
      )
        .shiftRight(uint(8))
        .bitAnd(uint(0xff));
      const A = gpu.instances.element(instId.mul(uint(2))) as unknown as NV4;
      const distC = vec3(cam.camPos)
        .sub(A.xyz as unknown as NV3)
        .length();
      If(
        matClass
          .equal(uint(4))
          .and(distC.greaterThanEqual(scarNear))
          .and(distC.lessThanEqual(scarFar)),
        () => {
          atomicAdd(scarEl(1), uint(1));
        },
      );
    });
  })().compute(pixelCount, [256]);
  (kScarCovered as unknown as ComputeKernel).setName('nanScarCovered');

  // ---- HW big/near-triangle passes (vertex pulling; fragment writes vis bufs) --------
  const hwGeometry = new BufferGeometry();
  hwGeometry.setAttribute(
    'position',
    new Float32BufferAttribute(new Float32Array(3), 3),
  );
  hwGeometry.setIndirect(hwDrawAttr, 0);
  hwGeometry.boundingSphere = new Sphere(
    new Vector3(),
    Number.POSITIVE_INFINITY,
  );

  const buildHwMaterial = (
    pass: 'depth' | 'combined' | 'world1',
    // ?clhw INSTANCED per-cluster draw: one instance per HW cluster, verts pulled from
    // qHwRaster instead of the per-tri soup queue. Same FS election ⇒ resolve unchanged.
    instanced = false,
  ): NodeMaterial => {
    const sfx = instanced ? `${pass}_cl` : pass;
    const mat = new NodeMaterial();
    mat.name = `nanRaster_${sfx}`;
    const vPayLo = varyingProperty('float', `nanPayLo_${sfx}`) as unknown as NF;
    const vPayHi = varyingProperty('float', `nanPayHi_${sfx}`) as unknown as NF;
    const vZ = varyingProperty('float', `nanZ_${sfx}`) as unknown as NF;
    const vW = varyingProperty('float', `nanW_${sfx}`) as unknown as NF;

    mat.vertexNode = Fn(() => {
      const corner = vertexIndex.mod(3) as unknown as NU;
      const setVaryings = (payload: NU, clip: NV4): void => {
        (vPayLo as unknown as { assign: (v: unknown) => void }).assign(
          toF(payload.bitAnd(uint(0xffff))),
        );
        (vPayHi as unknown as { assign: (v: unknown) => void }).assign(
          toF(payload.shiftRight(uint(16))),
        );
        (vZ as unknown as { assign: (v: unknown) => void }).assign(clip.z);
        (vW as unknown as { assign: (v: unknown) => void }).assign(clip.w);
      };
      const fetchWorld = (ctx: VertCtx, localTri: NU): NV3 => {
        if (hw1fetch) return fetchWorldVertDyn(ctx, localTri, corner);
        const w0 = fetchWorldVert(ctx, localTri, 0);
        const w1 = fetchWorldVert(ctx, localTri, 1);
        const w2 = fetchWorldVert(ctx, localTri, 2);
        return corner
          .equal(uint(1))
          .select(w1, corner.equal(uint(2)).select(w2, w0)) as unknown as NV3;
      };
      if (instanced) {
        // ?clhw INSTANCED cluster draw: one instance per HW cluster. qHwRaster[instanceIndex]
        // = the cluster's qRaster INDEX (tid). localTri = vertexIndex/3 (partial-cluster tail
        // clips out). payload = tid<<bits|localTri EXACTLY like the soup ⇒ resolve unchanged.
        const localTri = (vertexIndex.div(3) as unknown as NU).toVar();
        const tid = elemU(
          cull.qHwRasterRO as StorageBufferNode<'uint'>,
          instanceIndex,
        ).toVar();
        const item = qRasterRO.element(tid.add(uint(1)));
        const instId = item.x.toVar();
        const ci = item.y.toVar();
        const payload = tid.shiftLeft(uint(CLUSTER_TRI_BITS)).bitOr(localTri).toVar();
        const ctx = makeCtx(instId, ci);
        const world = fetchWorld(ctx, localTri);
        const clip = cam.vp.mul(vec4(world, 1)).toVar();
        // D (?clhw seam): snap clip.xy onto the SAME 1/256-px screen grid the SW
        // rasterizer quantizes to (s = (ndc+1)·0.5·(W,H) then ×256 round, ~:1175).
        // A shared SW↔HW cluster edge then lands on identical subpixel positions, so
        // no pixel near the seam is dropped by both paths. This is the exact inverse
        // of the SW screen transform with the quantize in the middle; the Y-up↔Y-down
        // framebuffer flip preserves the grid (H integer ⇒ H·256 integer). Per-component
        // scalar rounds match the SW path exactly.
        {
          const W = float(cam.uW);
          const H = float(cam.uH);
          const sx = clip.x.div(clip.w).add(1).mul(0.5).mul(W).mul(256).round().div(256);
          const sy = clip.y.div(clip.w).add(1).mul(0.5).mul(H).mul(256).round().div(256);
          clip.assign(
            vec4(
              sx.div(W).mul(2).sub(1).mul(clip.w),
              sy.div(H).mul(2).sub(1).mul(clip.w),
              clip.z,
              clip.w,
            ) as unknown as NV4,
          );
        }
        // partial-cluster tail: fixed MAX_CLUSTER_TRIS*3 verts over-cover a <128-tri cluster
        // ⇒ collapse those verts to a clipped point (z/w=2).
        clip.assign(
          (
            localTri.lessThan(ctx.triCount) as unknown as {
              select(a: NV4, b: NV4): NV4;
            }
          ).select(clip as unknown as NV4, vec4(0, 0, 2, 1) as unknown as NV4),
        );
        setVaryings(payload as unknown as NU, clip as unknown as NV4);
        return clip;
      }
      // SOUP (default, ?clhw off — byte-identical to the pre-clhw shader): per-tri from hwQueue.
      const triIndex = vertexIndex.div(3) as unknown as NU;
      const base = triIndex.mul(uint(2)).add(uint(1));
      const payload = elemU(hwQueueV.ro, base).toVar();
      const instId = elemU(hwQueueV.ro, base.add(uint(1))).toVar();
      const itemIdx = payload.shiftRight(uint(CLUSTER_TRI_BITS));
      const localTri = payload.bitAnd(uint(CLUSTER_TRI_MASK));
      const item = qRasterRO.element(itemIdx.add(uint(1)));
      const ci = item.y.toVar();
      const ctx = makeCtx(instId, ci);
      const world = fetchWorld(ctx, localTri as unknown as NU);
      const clip = cam.vp.mul(vec4(world, 1)).toVar();
      setVaryings(payload as unknown as NU, clip as unknown as NV4);
      return clip;
    })() as unknown as typeof mat.vertexNode;

    mat.fragmentNode = Fn(() => {
      const z = vZ.div(vW).toVar();
      const pay = uint(vPayLo.round())
        .bitOr(uint(vPayHi.round()).shiftLeft(uint(16)))
        .toVar();
      const fy = float(cam.uH).sub(screenCoordinate.y);
      const px = uint(fy).mul(uint(cam.uW)).add(uint(screenCoordinate.x));
      If(z.greaterThanEqual(0).and(z.lessThanEqual(1)), () => {
        const bits = bcF2U(z as unknown as NF);
        if (pass === 'depth') {
          atomicMin(visDepthV.atomic.element(px), bits);
        } else if (pass === 'combined') {
          // RACE-FREE packed (mirrors the SW combined path): id packed by depth-keyed
          // atomicMax into visA(payload)+visB, no separate depthV.
          const myDk = depthKey16(z as unknown as NF).toVar();
          const stored = aLoadU(visPayloadV.atomic.element(px));
          If(myDk.greaterThan(stored.shiftRight(uint(16))), () => {
            const dk = myDk.shiftLeft(uint(16));
            atomicMax(
              visPayloadV.atomic.element(px),
              dk.bitOr(pay.bitAnd(uint(0xffff))),
            );
            atomicMax(
              visBV.atomic.element(px),
              dk.bitOr(pay.shiftRight(uint(16)).bitAnd(uint(0xffff))),
            );
          });
        } else if (pass === 'world1') {
          // 0a SCAR (?scar=1): count this HW fragment into the whole-frame total [2] so
          // the band-share denominator includes the large/near tris the HW vertex-pull
          // path carries (terrain, trunks, near-plane-crossing leaf edges). The band
          // NUMERATOR [0] stays SW-only by design — mid/far foliage leaves rasterize on
          // the SW path; the HW path holds non-band near geometry. No band classify here.
          if (scar) atomicAdd(scarEl(2), uint(1));
          // PERF-VB4 single-pass WORLD (mirrors the SW world1 path): a depth16-keyed
          // atomicMax election whose WINNER write-stores the full id + its exact depth.
          const cand = depthKey24(z as unknown as NF)
            .shiftLeft(uint(8))
            .bitOr(pay.bitAnd(uint(0xff)))
            .toVar();
          // Mirror the SW world1 election's relaxed-load guard (the SW path at the
          // .Else()-branch above has had this; the HW vertex-pull path was missing
          // it — it RMW-contended visPayloadV on EVERY covered fragment). Bit-identical:
          // atomicMax is monotone, so skipping it when cand<=prevE leaves the buffer
          // value unchanged, and the store is already gated by cand>wonE which is false
          // whenever cand<=prevE (the buffer only grows). The only effect is dropping
          // the RMW for fragments behind the current front — the bulk of HW overdraw.
          const prevE = aLoadU(visPayloadV.atomic.element(px));
          If(cand.greaterThan(prevE), () => {
            const wonE = atomicMax(
              visPayloadV.atomic.element(px),
              cand,
            ) as unknown as NU;
            If(cand.greaterThan(wonE), () => {
              atomicStore(visBV.atomic.element(px), pay);
            });
          });
        }
      });
      return vec4(0, 0, 0, 0);
    })() as unknown as typeof mat.fragmentNode;
    mat.depthTest = false;
    mat.depthWrite = false;
    mat.colorWrite = false;
    mat.fog = false;
    mat.lights = false;
    // N9-C2: don't HW back-face-cull. Two-sided leaf tris that cross the near plane
    // or oversize the i32 SW path land here with a single winding (the reversed dup
    // is gone), so culling would hole them. Safe for opaque solids: the nearer face
    // wins atomicMin and only the depth-match writes payload, so back-faces never win.
    mat.side = DoubleSide;
    return mat;
  };

  const hwDepthMat = buildHwMaterial('depth');
  const hwCombinedMat = buildHwMaterial('combined');
  const hwWorld1Mat = buildHwMaterial('world1');
  const hwScene = new Scene();
  const hwMesh = new Mesh(hwGeometry, hwDepthMat);
  hwMesh.frustumCulled = false;
  hwScene.add(hwMesh);

  // ?clhw: the INSTANCED per-cluster HW draw — one instance per big/near cluster, drawn
  // MAX_CLUSTER_TRIS*3 verts each (partial-cluster tail clips out). Own geometry + scene;
  // shares hwRT + the world1 election (buildHwMaterial('world1', instanced=true)). Rendered
  // in world1() right after the soup hwRender. Built only when the cull supplied the queue.
  const hwClusterScene = clhw ? new Scene() : null;
  if (clhw && hwClusterScene && cull.hwClusterDrawAttr) {
    const g = new BufferGeometry();
    g.setAttribute('position', new Float32BufferAttribute(new Float32Array(3), 3));
    g.setIndirect(cull.hwClusterDrawAttr, 0);
    g.boundingSphere = new Sphere(new Vector3(), Number.POSITIVE_INFINITY);
    const m = new Mesh(g, buildHwMaterial('world1', true));
    m.frustumCulled = false;
    hwClusterScene.add(m);
  }
  // The HW pass renders into this dead full-res rgba8 (colorWrite=false -> never read).
  // It stays full-res unconditionally: r184 derives the render-pass viewport from
  // RenderTarget.viewport (= texture size), so shrinking it would clip HW coverage and
  // starve the vis buffers. ?hwrt=0 instead drops only the per-frame CLEAR (see hwRender).
  const hwRT = new RenderTarget(width, height, { depthBuffer: false });
  hwRT.texture.name = 'nanHwPass';

  // ---- flat resolve -----------------------------------------------------------------
  const resolveGeometry = new BufferGeometry();
  resolveGeometry.setAttribute(
    'position',
    new Float32BufferAttribute(
      new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]),
      3,
    ),
  );
  resolveGeometry.boundingSphere = new Sphere(
    new Vector3(),
    Number.POSITIVE_INFINITY,
  );

  const resolveMat = new NodeMaterial();
  resolveMat.name = 'nanHwResolveMat';
  resolveMat.vertexNode = vec4(
    positionGeometry.xy,
    0,
    1,
  ) as unknown as typeof resolveMat.vertexNode;
  resolveMat.fragmentNode = Fn(() => {
    const fy = float(cam.uH).sub(screenCoordinate.y);
    const pixelIndex = uint(fy).mul(uint(cam.uW)).add(uint(screenCoordinate.x));
    let itemIdx: NU;
    let localTri: NU;
    if (packed) {
      // PACKED (combined): id is split across visA(payload)+visB, depth in the high
      // bits. visA==0 ⇒ atomicMax never ran ⇒ empty pixel. (depthV is NOT read here —
      // keeps the resolve at 2 vis buffers under the 10-storage Metal ceiling.)
      const aRaw = elemU(visPayloadV.ro, pixelIndex);
      const bRaw = elemU(visBV.ro, pixelIndex);
      If(aRaw.equal(uint(0)), () => {
        Discard();
      });
      const id = bRaw
        .bitAnd(uint(0xffff))
        .shiftLeft(uint(16))
        .bitOr(aRaw.bitAnd(uint(0xffff)));
      itemIdx = id.shiftRight(uint(CLUSTER_TRI_BITS)) as unknown as NU;
      localTri = id.bitAnd(uint(CLUSTER_TRI_MASK)) as unknown as NU;
    } else {
      const dRaw = elemU(visDepthV.ro, pixelIndex);
      const pRaw = elemU(visPayloadV.ro, pixelIndex);
      If(dRaw.equal(uint(0xffffffff)), () => {
        Discard();
      });
      // orphan (pass-2 never matched pass-1's depth): background, not a garbage
      // payload decode — black-pixel probes then see it as a hole
      If(pRaw.equal(uint(0xffffffff)), () => {
        Discard();
      });
      itemIdx = pRaw.shiftRight(uint(CLUSTER_TRI_BITS)) as unknown as NU;
      localTri = pRaw.bitAnd(uint(CLUSTER_TRI_MASK)) as unknown as NU;
    }
    const item = qRasterRO.element(itemIdx.add(uint(1)));
    const instId = item.x.toVar();
    const ci = item.y.toVar();
    const ctx = makeCtx(instId, ci);

    let lambert: NF;
    if (shade) {
      const w0 = fetchWorldVert(ctx, localTri, 0);
      const w1 = fetchWorldVert(ctx, localTri, 1);
      const w2 = fetchWorldVert(ctx, localTri, 2);
      const faceN = normalize(
        cross(w1.sub(w0) as unknown as NV3, w2.sub(w0) as unknown as NV3),
      ) as unknown as NV3;
      const L = normalize(vec3(0.55, 0.8, 0.25)) as unknown as NV3;
      lambert = max(dot(faceN, L), 0).mul(0.85).add(0.18) as unknown as NF;
    } else {
      lambert = float(1) as unknown as NF;
    }

    // matClass palette (flat) — terrain/rock/bark/deadwood/leaf/grass/debris
    const matClass = elemU(
      gpu.meshes,
      ctx.meshId.mul(uint(MESH_WORDS)).add(uint(6)),
    )
      .shiftRight(uint(8))
      .bitAnd(uint(0xff))
      .toVar();
    const isT = matClass.equal(uint(0));
    const isR = matClass.equal(uint(1));
    const isB = matClass.equal(uint(2));
    const isD = matClass.equal(uint(3));
    const albedo = isT
      .select(
        vec3(0.3, 0.36, 0.22),
        isR.select(
          vec3(0.42, 0.41, 0.4),
          isB.select(
            vec3(0.36, 0.27, 0.19),
            isD.select(vec3(0.33, 0.28, 0.22), vec3(0.35, 0.33, 0.3)),
          ),
        ),
      )
      .toVar();
    let col: NV3;
    if (tint === 'cluster') {
      col = hashColor(ci).mul(lambert) as unknown as NV3;
    } else if (tint === 'lod') {
      // color by the cluster's LOD LEVEL (packed in cluster word7 bits 10-15) — a
      // per-level hue so the cut's distance falloff reads as colour bands (fine→coarse).
      // The hue is a NON-wrapping red→violet sweep: level/MAXLVL maps to t∈[0,4.8] rad
      // (< 2π), so the rainbow never completes a cycle and high levels can't loop back
      // to salmon (level 0). Leaf DAGs reach ~14 levels; MAXLVL=16 keeps them in range.
      const lvl = elemU(gpu.clusters, ci.mul(uint(CLUSTER_WORDS)).add(uint(7)))
        .shiftRight(uint(10))
        .bitAnd(uint(0x3f));
      const t = toF(lvl).div(16).min(1).mul(4.8);
      const c3 = vec3(
        t.cos().mul(0.5).add(0.5),
        t.add(2.094).cos().mul(0.5).add(0.5),
        t.add(4.188).cos().mul(0.5).add(0.5),
      ) as unknown as NV3;
      col = c3.mul(lambert) as unknown as NV3;
    } else {
      col = (albedo as unknown as NV3).mul(lambert) as unknown as NV3;
    }
    return vec4(col, 1);
  })() as unknown as typeof resolveMat.fragmentNode;
  resolveMat.depthNode = Fn(() => {
    const fy = float(cam.uH).sub(screenCoordinate.y);
    const pixelIndex = uint(fy).mul(uint(cam.uW)).add(uint(screenCoordinate.x));
    if (packed) {
      // reconstruct cz from visA's 16-bit depth key (dk = (1−cz)·65535). Approximate
      // (16-bit) — exact enough for the isolated debug view; depthV stays out of the
      // resolve's binding set. (A future world-mode packed path can recompute exact
      // cz from the reconstructed triangle instead.)
      const dk = toF(elemU(visPayloadV.ro, pixelIndex).shiftRight(uint(16)));
      return float(1).sub(
        dk.div(65535),
      ) as unknown as typeof resolveMat.depthNode;
    }
    return bcU2F(elemU(visDepthV.ro, pixelIndex));
  })() as unknown as typeof resolveMat.depthNode;
  resolveMat.depthTest = false;
  resolveMat.depthWrite = true;
  resolveMat.fog = false;
  resolveMat.lights = false;

  const resolveMesh = new Mesh(resolveGeometry, resolveMat);
  resolveMesh.frustumCulled = false;
  const resolveScene = new Scene();
  resolveScene.add(resolveMesh);

  // ---- per-frame passes --------------------------------------------------------------
  const clearVis = (renderer: Renderer): void => {
    dispatch(renderer, kVisClear);
  };
  const depth1 = (renderer: Renderer): void => {
    dispatchIndirect(renderer, kRasterDepth, cull.rasterDispatchAttr);
  };
  const depth1Batch = (): readonly unknown[] => [
    setIndirectDispatch(kRasterDepth, cull.rasterDispatchAttr),
  ];
  const hwQueueClearBatch = (): readonly unknown[] => [kHwQueueClear];
  const hwRender = (
    renderer: Renderer,
    camera: PerspectiveCamera,
    mat: NodeMaterial,
  ): void => {
    const prevRT = renderer.getRenderTarget();
    renderer.setRenderTarget(hwRT);
    hwMesh.material = mat;
    // ?hwrt=0: skip the per-frame full-res CLEAR of the dead rgba8 color target. With
    // autoClear=false the backend uses loadOp=Load (Background.js:209-217 -> the
    // descriptor's loadOp becomes Load not Clear). The HW fragment has colorWrite=false,
    // so it never writes the target; nothing downstream reads it; the only effect is the
    // clear no longer runs. The viewport/coverage (full-res hwRT) and every vis-buffer
    // write are unchanged -> byte-identical. hwScene has no .background, so forceClear
    // stays false and autoClear=false is honored. RESTORED immediately after the render.
    const prevAutoClear = renderer.autoClear;
    if (!hwrt) renderer.autoClear = false;
    renderer.render(hwScene, camera);
    if (!hwrt) renderer.autoClear = prevAutoClear;
    renderer.setRenderTarget(prevRT);
  };
  // ?clhw: the instanced per-cluster HW draw into the same hwRT/vis buffers (never clears —
  // the election is atomicMax, order-independent; it just adds the big/near cluster winners).
  const hwRenderCluster = (renderer: Renderer, camera: PerspectiveCamera): void => {
    if (!hwClusterScene) return;
    const prevRT = renderer.getRenderTarget();
    renderer.setRenderTarget(hwRT);
    const prevAutoClear = renderer.autoClear;
    renderer.autoClear = false;
    renderer.render(hwClusterScene, camera);
    renderer.autoClear = prevAutoClear;
    renderer.setRenderTarget(prevRT);
  };
  const hwDepth = (renderer: Renderer, camera: PerspectiveCamera): void => {
    dispatch(renderer, kHwArgs);
    hwRender(renderer, camera, hwDepthMat);
  };
  // single-pass Z+payload over the full set (hier path): replaces depth1 + the late
  // hwDepth + payload re-raster with ONE SW pass + ONE HW pass.
  const combined = (renderer: Renderer, camera: PerspectiveCamera): void => {
    dispatchIndirect(renderer, kRasterCombined, cull.rasterDispatchFullAttr);
    dispatch(renderer, kHwArgs); // SW pass filled hwQueue → build the indirect draw args
    hwRender(renderer, camera, hwCombinedMat);
  };
  // voxel-foliage (Stage 2 §6): the SCATTER voxel-brick raster (kVoxScatter).
  // Built only when the cull supplied the voxel work queue (?voxreg/
  // ?forcevox active); a separate permutation that DROPS tri-only verts/indices/hfVerts
  // (≤10 buffers, §4.6) and reuses the SAME 32-bit depthKey24 election + visPayloadV/
  // visBV. Dispatched in world1() AFTER hwRender so the on-chip election pre-seeds from
  // the global near-field TRIANGLE winners (§6.2/§6.6).
  const voxRaster: VoxelRasterHandles | null =
    voxActive && cull.qVoxRasterRO && cull.voxRasterDispatchAttr
      ? buildNaniteVoxelRaster({
          gpu,
          cam,
          qVoxRasterRO: cull.qVoxRasterRO,
          voxRasterDispatchAttr: cull.voxRasterDispatchAttr,
          // DEPTH-BUCKET F2B: the cull always publishes these alongside the voxel queue.
          voxBucketRangeRO: cull.voxBucketRangeRO as BufOf<UV2>,
          voxBucketDispatchAttr: cull.voxBucketDispatchAttr ?? [],
          voxF2bK: cull.voxF2bK ?? 1,
          voxF2bEnabled: cull.voxF2bEnabled ?? false,
          voxPrevEnabled: cull.voxPrevEnabled ?? false,
          depthKey24,
          visPayloadV,
          visBV,
          width,
          height,
          // ?voxwind rigid crown sway — presence gates it + supplies the mesh-matched camPos
          wind,
        })
      : null;

  // PERF-VB4 (D-N45) WORLD single pass over the full set (replaced depth1 → hwDepth →
  // payload). ONE SW + ONE HW pass: a 24-bit depth election (visPayloadV) whose winner
  // stores the full id into visBV; the resolve/shadows/HZB read depth from the election
  // key (no exact depthV — a 3rd hot-loop atomic buffer was measured a 3× cliff).
  //
  // SUBMIT-COALESCE (item 3): the vis CLEAR + the SW world1 raster + kHwArgs run in ONE
  // batched submit (was 3 separate queue.submit drains). Order is load-bearing and
  // preserved: [kVisClear, kRasterWorld1, kHwArgs].
  //   - kVisClear WAW-before kRasterWorld1 on vis* (depth→0xffffffff, payload/visB→0,
  //     hwQueue[0]→0) — the raster's atomicMin/atomicMax election + hwQueue appends MUST
  //     see the cleared sentinels;
  //   - kRasterWorld1 is INDIRECT over rasterDispatchFull (tagged via setIndirectDispatch
  //     above, so it keeps its tight size in this dispatchSize=null batch);
  //   - kHwArgs RAW-reads hwQueue[0] (the SW pass's near/big-tri appends) → hwDraw args.
  // hwRender STAYS its own renderer.render submit (a render pass + a compute pass cannot
  // share one command encoder). Because world1 now OWNS the clear, the frame must NOT call
  // clearVis() separately for the world path (it does not — NaniteFrame updated).
  const world1 = (
    renderer: Renderer,
    camera: PerspectiveCamera,
    hzbTail: readonly unknown[] = [],
  ): void => {
    // procedural grass rides this submit: after kVisClear (elections need the
    // cleared sentinels), before kHwArgs is irrelevant to it (disjoint queues) —
    // but BEFORE hwRender so its near-blade HW queue is filled when the shared
    // HW pass draws it. Its fragments then pre-seed the voxel raster's election
    // exactly like the mesh winners do.
    dispatchBatchMixed(renderer, [
      kVisClear,
      kRasterWorld1,
      ...(grass?.batch ?? []),
      kHwArgs,
    ]);
    hwRender(renderer, camera, hwWorld1Mat);
    // ?clhw: paint the big/near CLUSTERS the SW pass skipped, as proper instanced HW draws
    // (one instance/cluster, single transform). Same election ⇒ folds into the vis buffers
    // alongside the SW + soup winners, before grass/voxel pre-seed from them.
    if (clhw) hwRenderCluster(renderer, camera);
    // grass blade HW pass (own depth target, early-z primed from the election —
    // which now holds the mesh SW+HW winners + the grass SW slivers).
    if (grass?.enabled()) grass.renderHw(renderer, camera);
    // scatter voxel-brick raster (§6.6 insertion point: right after hwRender so the
    // SW+HW near-field triangle election is already in global visPayloadV to pre-seed).
    // SUBMIT-COALESCE §1c: the HZB chain rides as the TAIL of the voxel submit (it must
    // run strictly after kVoxScatter — in-pass UAV sync gives that). noleaves/no-vox:
    // dispatch the tail as its own single submit, same 1 submit as legacy hzb.build().
    if (voxRaster) voxRaster.dispatchVoxel(renderer, hzbTail);
    else if (hzbTail.length > 0) dispatchBatchMixed(renderer, hzbTail);
  };

  const readHwCount = async (renderer: Renderer): Promise<number> => {
    const buf = await readBuffer(renderer, hwQueueAttr, 0, 4);
    return new Uint32Array(buf)[0] ?? 0;
  };
  const audit = (renderer: Renderer): void => {
    dispatch(renderer, kAudit);
  };
  const readAudit = async (
    renderer: Renderer,
  ): Promise<{ orphans: number; covered: number }> => {
    const buf = await readBuffer(renderer, auditAttr, 0, 8);
    const u = new Uint32Array(buf);
    return { orphans: u[0] ?? 0, covered: u[1] ?? 0 };
  };
  // 0a SCAR: the per-fragment band/total counters are accumulated INSIDE world1; this
  // runs the covered-pixel denominator post-pass. No-op unless ?scar=1 (so production
  // pays nothing — neither the dispatch nor the world1 atomics fire).
  const scarRun = (renderer: Renderer): void => {
    if (scar) dispatch(renderer, kScarCovered);
  };
  const readScar = async (
    renderer: Renderer,
  ): Promise<{
    bandFrags: number;
    bandPx: number;
    totalFrags: number;
    bandClusters: number;
  }> => {
    // scar counters live in the hwQueue tail at slot SCAR_BASE (byte offset SCAR_BASE*4).
    const buf = await readBuffer(renderer, scarAttr, SCAR_BASE * 4, 16);
    const u = new Uint32Array(buf);
    return {
      bandFrags: u[0] ?? 0,
      bandPx: u[1] ?? 0,
      totalFrags: u[2] ?? 0,
      bandClusters: u[3] ?? 0,
    };
  };

  const readVoxWrites = async (renderer: Renderer): Promise<number | null> =>
    voxRaster ? voxRaster.readWriteCount(renderer) : null;

  return {
    resolveScene,
    clearVis,
    depth1,
    depth1Batch,
    hwQueueClearBatch,
    hwDepth,
    combined,
    world1,
    readHwCount,
    audit,
    readAudit,
    scar: scarRun,
    readScar,
    readVoxWrites,
    triHzbCopyKernel: kTriHzbCopy,
  };
}
