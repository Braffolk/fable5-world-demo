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

import { Mesh, Scene, Vector3 } from 'three';
import { BufferGeometry, Float32BufferAttribute, Sphere } from 'three';
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
  vec2,
  vec3,
  vec4,
  workgroupArray,
  workgroupBarrier,
} from 'three/tsl';
import type { NB, NF, NI, NU, NV3, NV4 } from '../gpu/TSLTypes';
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
  bcI2U,
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
// ─── extracted raster/ modules (task #76 vis-buffer rewrite, Step 1) ──────────────
import { CTX_STRIDE, CTX_U, buildClusterCtx } from './raster/ClusterCtx';
import { buildHw } from './raster/Hw';
import { buildMid } from './raster/Mid';
import {
  HW_CAP,
  MID_CAP,
  MID_STRIDE,
  SPLAT_CAP,
  buildQueues,
} from './raster/Queues';
import { makeScanline } from './raster/Scanline';
import { buildSplat } from './raster/Splat';
import {
  buildVisClear,
  depthKey16,
  depthKey24,
  makeElect,
  type NaniteVisBuffers,
} from './raster/VisBuffer';
export { makeVisBuffers } from './raster/VisBuffer';
export type { NaniteVisBuffers } from './raster/VisBuffer';

// HW_CAP / SPLAT_CAP / MID_CAP / MID_STRIDE now live in ./raster/Queues (imported).
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

// NaniteVisBuffers + makeVisBuffers now live in ./raster/VisBuffer (imported +
// re-exported above so existing callers keep importing them from './NaniteRaster').

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
  // ?ctxsm — keep the wgcache broadcast ctx SHARED-MEMORY-resident instead of hoisting all
  // ~30 fields into registers (.toVar()). world1 is register-bound (116 temp regs, spilling →
  // occupancy floor → the election-atomic latency can't hide); the ctx already lives in shU/shF,
  // so reading fields on-demand from on-chip threadgroup memory (cheap on Apple) trades the
  // scarce resource (registers) for the spare one (shared reads) with NO new buffer/bandwidth/
  // dispatch. Byte-identical (shared mem is read-only after the broadcast barrier). Default OFF
  // (A/B via `?ctxsm=1`); if tint re-hoists or the peak is makeCtx not the per-tri fetch, no win.
  const ctxsm =
    new URLSearchParams(window.location.search).get('ctxsm') === '1';
  // ?coopv — RADICAL restructure (de-risk build): split the world1 kernel into
  //   PHASE 1 (cooperative per-corner vertex fetch+transform → clip, written to shared)
  //   → barrier → PHASE 2 (per-triangle raster reads clip from shared).
  // Goal: the AGX allocator reuses phase-1 transform registers for phase-2 raster, so the
  // 116-reg peak (→ 27% occupancy) drops toward the raster-core size, lifting occupancy to
  // fill the measured ALU headroom. THIS BUILD IS A HYPOTHESIS TEST: corner-indexed (no
  // dedup, 384 clip slots = 6 KB shared) so it works for ALL cluster types with zero
  // enumeration — its ONLY job is to read the compiled register count. If registers drop,
  // the dedup + DAG-enumeration + shared-shrink follow-up is worth building. Byte-identical
  // (same transform, same clip). Default OFF (?coopv=1).
  const coopv =
    new URLSearchParams(window.location.search).get('coopv') === '1';
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
  // depthKey16/depthKey24 (nearer ⇒ larger key ⇒ wins atomicMax) live in ./raster/VisBuffer.
  // The shipped depth-keyed election bound to these vis buffers — shared by the Splat, Mid
  // and Hw consumers (byte-identical to the inline copies it replaces).
  const elect = makeElect(vis);

  // ─── Splat-election (task #76) — THE world1 sub-pixel path, not a flag ───────────
  // The sub-pixel majority is not rasterized as triangles: world1 APPENDS (px,cand,payload)
  // to splatQueue and the lean nanSplatElect kernel (24 regs measured) runs the depthKey24
  // (<<8|id) atomicMax election into visPayloadV/visBV at full occupancy. splatQueue[0] is
  // the RAW append count (atomicAdd keeps counting past the cap): count−CAP = sub-pixel
  // fragments DROPPED (overflow → holes) — read it to size the cap / gauge the true fragment
  // count N. INDIRECT dispatch (kSplatArgs→nanSplatElect, the cull BFS kArgs→kTraverse
  // in-batch pattern) launches only ceil(N/64) workgroups. 1-D grid ⇒ ceil(N/64) ≤ 65535 ⇒
  // SPLAT_CAP ≤ 65535·64. World1 (singlePass) only — it adds ONE storage buffer to world1 ⇒
  // 9→10, at the ceiling (the ctx pre-pass already shed gpu.clusters/instances/meshes → 9).
  // nanSplatElect binds only visPayloadV/visBV/splatQueue = 3 ⇒ lean.
  const splatElect = singlePass;
  // ── DIAGNOSTIC path isolation (task #76 de-über debug; TEMPORARY — remove once the blink
  // is pinned). Independently DROP each of world1's routed classes to see one in isolation:
  // ?nospl = ≤1px splat, ?nomid = 2..swmax mid, ?nohw = big/near HW (world1's per-tri append).
  // e.g. ?clhw=0&nospl=1&nohw=1 = ONLY the mid-band SW tris render (isolates nanMidRaster).
  const dbgQ = new URLSearchParams(window.location.search);
  const dbgNoSpl = dbgQ.get('nospl') === '1';
  const dbgNoMid = dbgQ.get('nomid') === '1';
  const dbgNoHw = dbgQ.get('nohw') === '1';
  // hw/splat/mid work queues + their indirect-args kernels (./raster/Queues). The splat/mid
  // queues + args exist only on the world1 path (splatElect); the ?scar counters + the
  // ?trihzb prev-frame HZB mirror fold into the hwQueue tail (SCAR_BASE/TRIHZB_BASE). triLvls
  // sizes the mirror. Caps (HW_CAP/SPLAT_CAP/MID_CAP/MID_STRIDE) are imported from Queues.
  const queues = buildQueues({ splatElect, triLvls });
  const {
    hwQueueAttr,
    hwQueueV,
    hwDrawAttr,
    hwDrawBuf,
    SCAR_BASE,
    TRIHZB_BASE,
    triTailN,
    scarEl,
    splatQueueV,
    splatDrawAttr,
    midQueueV,
    midDrawAttr,
    kSplatArgs,
    kMidArgs,
  } = queues;

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
  // scar counters live in the hwQueue tail at [SCAR_BASE..+4); scarEl (from buildQueues)
  // aliases hwQueueV. The readback reads hwQueueAttr at byte offset SCAR_BASE*4.
  const scarAttr = hwQueueAttr;

  // ---- shared fetch helpers (NaniteFetch.ts — also the resolve's decode) ----------
  const nfetch = makeFetch(gpu, heightTex, disp, wind);
  // fetchWorldVertDyn is used only by the HW vertex stage (./raster/Hw, via nfetch).
  const { makeCtx, fetchWorldVert } = nfetch;
  // M2l hw1fetch: HW vertex stage reconstructs ONE corner (runtime-selected) instead
  // of fetching all 3 and selecting — same selected vertex by construction.
  const hw1fetch =
    new URLSearchParams(window.location.search).get('hw1fetch') === '1';
  // PERF-3 win #2 — the cooperative vertex-transform cache lives in its own module
  // (default OFF, ?vcompact=1; measured marginal/conditional — see NaniteVertexCache).
  const vcache = makeVertexCache(gpu, nfetch);

  // ?ksplit (PERF task #76): build world1 as TWO class-specialized kernels —
  // 'explicit' (leaf/trunk/rock) + 'terrain' (heightfield) — each compiling ONLY its
  // own fetch arm (makeFetch variant), so it reserves ONLY its own registers (the leaf
  // kernel sheds terrainDispAt; the terrain kernel sheds the transform+wind set = the
  // branch-union that pinned world1's occupancy floor). MEASUREMENT stage: both dispatch
  // over the WHOLE qRaster and route by a uniform isHF early-out — byte-identical output,
  // ~1ms doubled launches — so Xcode can read each kernel's registers/occupancy BEFORE
  // the cull-side queue partition (Stage 2) removes the doubled launches. world1 only.
  const ksplit =
    new URLSearchParams(window.location.search).get('ksplit') === '1';

  // ─── Task 1 (radical): per-cluster ctx PRE-PASS ──────────────────────────────────
  // The wgcache thread-0 makeCtx broadcast is 1 ACTIVE LANE / 127 masked doing the
  // metadata unpack + up-to-six gust/disp samples + wind setup + the clhw classify,
  // behind a workgroupBarrier — a divergence + barrier stall the raster occupancy cannot
  // hide, and the counter bills every masked lane. Move it OUT to a ONE-THREAD-PER-CLUSTER
  // pre-pass (kClusterCtx, built below) that computes makeCtx once and writes the 35-word
  // ctx (12 u32 + 23 f32-as-bits) to a global buffer; the raster then reads it with
  // cache-coherent loads (itemIdx is UNIFORM per workgroup ⇒ one L1 line for all 128
  // lanes) — no thread-0 block, no barrier. BONUS: makeCtx leaving the raster SHEDS
  // gpu.clusters/instances/meshes from its bindings (the fetch only needs verts/indices/
  // height) ⇒ 9 storage buffers, under the 10 ceiling. WORLD1 (singlePass) only — depth/
  // combined/view keep the broadcast and never allocate the buffer. Buffer = QRASTER_CAP ×
  // 35 u32 (world QRASTER_CAP = 1M ⇒ 140 MB); every visible cluster is rewritten each frame
  // so the initial contents are irrelevant.
  const ctxPrepass = singlePass;
  // ?wgcache bool→uint for packing isHF/isDAG into the shared-memory uint array (shared by
  // the world1 broadcast AND the ClusterCtx pre-pass, kept identical so both pack the same).
  const b2u = (b: NB): NU =>
    (b as unknown as { select(a: NU, c: NU): NU }).select(uint(1), uint(0));
  // per-cluster ctx PRE-PASS (./raster/ClusterCtx): CTX_U/CTX_STRIDE (imported) + the 35-word
  // buffer + nanClusterCtxPrepass. WORLD1 (singlePass) only; the world1 raster reads
  // clusterCtxV with cache-coherent loads (no thread-0 makeCtx, no barrier).
  const { clusterCtxV, kClusterCtx } = buildClusterCtx({
    ctxPrepass,
    gpu,
    cam,
    qRasterRO,
    makeCtx,
    projK,
    clhw,
    clhwMax,
    wind,
    b2u,
  });

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
  // kVisClear (full-screen) + kHwQueueClear (the counter-only tail, reused by the scoped
  // shadow clear) live in ./raster/VisBuffer — both reset the queue + audit + scar counters.
  const { kVisClear, kHwQueueClear } = buildVisClear({
    pixelCount,
    vis,
    hwQueueV,
    auditV,
    scar,
    scarEl,
    splatQueueV,
    midQueueV,
    packedClear,
    skipDepthClear,
    skipVisClear,
  });

  // ---- SW raster kernels (fixed-point integer scanline) ------------------------------
  // mode 'depth' = atomicMin Z only (the SHADOW depth-only path). 'combined' = single-pass
  // packed Z+id for the NaniteView debug view (16-bit depth split idLo/idHi). 'world1' =
  // the WORLD single pass (PERF-VB4): a 24-bit depth election (visPayloadV) whose winner
  // stores the full 25-bit id into the side buffer visBV; the resolve reconstructs depth
  // from the election key. No exact depthV (a 3rd hot-loop atomic buffer = a 3× cliff).
  // swScanline (./raster/Scanline): THE single fixed-point coverage loop, shared by the
  // inline depth/combined/world1 paths AND nanMidRaster — no duplicate rasterisation.
  const swScanline = makeScanline(cam);

  const rasterKernel = (
    mode: 'depth' | 'combined' | 'world1',
    // ?ksplit: 'explicit'|'terrain' builds a class-specialized world1 kernel whose fetch
    // compiles ONLY that arm; undefined = the unified kernel (byte-identical default).
    splitVariant?: 'explicit' | 'terrain',
  ): unknown => {
    // Variant-specific fetch + vertex cache: the specialized kernel decodes and
    // transforms through the arm-selected makeFetch so it never reserves the other
    // class's registers. splitVariant===undefined ⇒ the module nfetch/vcache (identical
    // node graph to the pre-split kernel).
    const kFetch = splitVariant
      ? makeFetch(gpu, heightTex, disp, wind, true, splitVariant)
      : nfetch;
    const kMakeCtx = kFetch.makeCtx;
    const kVcache = splitVariant ? makeVertexCache(gpu, kFetch) : vcache;
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
      if (ctxPrepass && mode === 'world1' && clusterCtxV) {
        // Task 1: read the per-cluster ctx the PRE-PASS (kClusterCtx) wrote to global
        // memory — cache-coherent loads (itemIdx uniform per workgroup ⇒ one L1 line for
        // all 128 lanes), NO thread-0 makeCtx, NO workgroup barrier. Same 35-word layout as
        // the old broadcast; makeCtx no longer runs here so gpu.clusters/instances/meshes
        // are not bound to this kernel.
        const base = itemIdx.mul(uint(CTX_STRIDE)).toVar();
        const rU = (i: number): NU =>
          elemU(clusterCtxV.ro, base.add(uint(i))).toVar() as unknown as NU;
        const rF = (i: number): NF =>
          bcU2F(elemU(clusterCtxV.ro, base.add(uint(CTX_U + i)))).toVar() as unknown as NF;
        const iHF = rU(0);
        matClassBroadcast = rU(10);
        // ?clhw skip (slot 11) — uniform per cluster ⇒ all lanes return together.
        if (clhw && mode === 'world1') {
          returnIf(rU(11).equal(uint(1)));
        }
        // ?ksplit class routing (slot 0 = isHF), uniform per cluster.
        if (splitVariant) {
          returnIf(
            splitVariant === 'explicit'
              ? iHF.equal(uint(1))
              : iHF.notEqual(uint(1)),
          );
        }
        ctx = {
          isHF: iHF.equal(uint(1)),
          isDAG: rU(1).equal(uint(1)),
          A: vec4(rF(0), rF(1), rF(2), rF(3)) as unknown as NV4,
          B: vec4(rF(4), rF(5), rF(6), rF(7)) as unknown as NV4,
          yawSc: { cy: rF(21), sy: rF(22) },
          triStart: rU(2),
          triCount: rU(3),
          meshId: rU(4),
          channel: rU(5),
          twoSided: rU(9).equal(uint(1)),
          wind:
            wind && splitVariant !== 'terrain'
              ? {
                  h0: rF(11),
                  dirX: rF(12),
                  dirY: rF(13),
                  leanBase: rF(14),
                  swayABase: rF(15),
                  swayPhase: rF(16),
                  ph: rF(17),
                  branchBase: rF(18),
                  flutBase: rF(19),
                  swayXPhase: rF(20),
                }
              : null,
          gx: rU(6),
          gz: rU(7),
          qxw: rU(8),
          oX: rF(8),
          oZ: rF(9),
          cell: rF(10),
        } as unknown as VertCtx;
      } else if (wgcache) {
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
        // ?ctxsm: shared-resident ctx — return the raw shared-mem read (re-read on each use)
        // instead of pinning it into a register via .toVar(). Baseline (ctxsm off) is byte-identical.
        const tU = (i: number): NU =>
          ctxsm ? getU(i) : (getU(i).toVar() as unknown as NU);
        const tF = (i: number): NF =>
          ctxsm ? getF(i) : (getF(i).toVar() as unknown as NF);
        If(localTri.equal(uint(0)), () => {
          const c = kMakeCtx(instId, ci);
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
          // 'terrain' variant has no wind (its makeCtx returns wind=null) — skip the
          // broadcast so the terrain kernel neither writes nor reserves the wind slots.
          if (wind && splitVariant !== 'terrain') {
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
        // ?ksplit routing: the explicit kernel skips heightfield clusters, the terrain
        // kernel skips non-heightfield — class = the broadcast isHF slot (0). Uniform per
        // cluster ⇒ every live thread returns together (post-barrier, before vcache's
        // barrier) — no partial-workgroup barrier, same shape as the clhw skip above.
        if (splitVariant) {
          returnIf(
            splitVariant === 'explicit'
              ? getU(0).toVar().equal(uint(1))
              : getU(0).toVar().notEqual(uint(1)),
          );
        }
        const cB = vec4(tF(4), tF(5), tF(6), tF(7)) as unknown as NV4;
        ctx = {
          isHF: tU(0).equal(uint(1)),
          isDAG: tU(1).equal(uint(1)),
          A: vec4(tF(0), tF(1), tF(2), tF(3)) as unknown as NV4,
          B: cB,
          yawSc: { cy: tF(21), sy: tF(22) },
          triStart: tU(2),
          triCount: tU(3),
          meshId: tU(4),
          channel: tU(5),
          twoSided: tU(9).equal(uint(1)),
          wind:
            wind && splitVariant !== 'terrain'
            ? {
                h0: tF(11),
                dirX: tF(12),
                dirY: tF(13),
                leanBase: tF(14),
                swayABase: tF(15),
                swayPhase: tF(16),
                ph: tF(17),
                branchBase: tF(18),
                flutBase: tF(19), // N9-C0 leaf flutter
                swayXPhase: tF(20),
              }
            : null,
          gx: tU(6),
          gz: tU(7),
          qxw: tU(8),
          oX: tF(8),
          oZ: tF(9),
          cell: tF(10),
        } as unknown as VertCtx;
      } else {
        ctx = kMakeCtx(instId, ci);
        // ?ksplit routing (non-wgcache path): same class gate, per-thread isHF (uniform
        // per cluster). REQUIRED for correctness — without it the explicit kernel would
        // rasterize heightfield clusters through the explicit fetch (garbage), so ksplit
        // is only sound with the early-out on whichever ctx path runs.
        if (splitVariant) {
          const isHFb = ctx.isHF as unknown as { not(): NB };
          returnIf(
            (splitVariant === 'explicit' ? ctx.isHF : isHFb.not()) as unknown as NB,
          );
        }
      }

      // PERF-3 win #2 — cooperative vertex-transform cache (own module; ?vcompact=1).
      // MUST run BEFORE the voxel returnIf below: prime() emits a workgroupBarrier, and
      // a barrier after a storage-derived returnIf is non-uniform control flow to naga
      // ⇒ WGSL validation failure ⇒ the whole world1 pipeline silently dies (the
      // "?vcompact renders an empty scene" bitrot, found 2026-07-02). Voxel clusters
      // have vcCount=0 (no compact range) so their populate no-ops — they pay only the
      // barrier before bailing.
      const corner = kVcache.prime(ctx, ci, localTri);

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
        if (coop || coopv) {
          // ?coopv also needs the guard form: its Phase-1 barrier is at kernel top scope,
          // so voxel-cluster threads must reach it (a storage-derived return before a
          // barrier is non-uniform to naga — the same ?vcompact bitrot).
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
        const payload = itemIdx
          .shiftLeft(uint(CLUSTER_TRI_BITS))
          .bitOr(localTri)
          .toVar();
        // ── streaming vertex assembly (task #76 register-crest cut, R2): each corner
        // flows clip→ndc→snap and its p/ndc/screen intermediates DIE before the next is
        // built, so the crest never holds three vertices in four representations at once.
        // nearOK folds in incrementally (no held 3× clip vec4). A near-plane-crossing
        // vertex snaps to a saturated/garbage i32 that is DISCARDED — nearOK routes the
        // whole triangle HW before any fixed-point value is trusted (same as before, only
        // the snap now runs eagerly; the garbage is never read on the HW path).
        const xiA: NI[] = [];
        const yiA: NI[] = [];
        const dzA: NF[] = [];
        const W = float(cam.uW);
        const H = float(cam.uH);
        let nearAcc: NB | null = null;
        for (const v of [0, 1, 2] as const) {
          const wv = corner(localTri, v);
          const pv = cam.vp.mul(vec4(wv, 1)).toVar();
          const okv = pv.w.greaterThan(NEAR_EPS) as unknown as NB;
          nearAcc = nearAcc ? nearAcc.and(okv) : okv;
          const ndcv = pv.xyz.div(pv.w).toVar();
          dzA[v] = ndcv.z.toVar() as unknown as NF;
          const sv = ndcv.xy.add(1).mul(0.5).mul(vec2(W, H)).toVar();
          xiA[v] = toI(sv.x.mul(256).round()).toVar() as unknown as NI;
          yiA[v] = toI(sv.y.mul(256).round()).toVar() as unknown as NI;
        }

        if (mode === 'world1' && rdbg === 5) {
          // ?rdbg=5 (world1) — STOP right after the 3-corner projection (fetch + vp
          // transform + ndc + snap), BEFORE winding/bbox/edge-setup. Bisects the
          // rdbg2 (=76) setup peak: rdbg5 ≈ 76 ⇒ the PROJECTION is the register wall;
          // rdbg5 ≪ 76 ⇒ winding/edge-setup adds it. Sinks all 9 fixed-point coords +
          // 3 dz so fetch/transform/project/snap cannot be DCE'd past the return.
          const sinkV = (dzA[0] as NF)
            .add(dzA[1] as NF)
            .add(dzA[2] as NF)
            .add(toF(xiA[0] as unknown as NI))
            .add(toF(yiA[0] as unknown as NI))
            .add(toF(xiA[1] as unknown as NI))
            .add(toF(yiA[1] as unknown as NI))
            .add(toF(xiA[2] as unknown as NI))
            .add(toF(yiA[2] as unknown as NI))
            .mul(1 / 9)
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

        if (mode === 'depth' && rdbg === 1) {
          // ?rdbg=1 — stop right after the streaming vertex assembly. The sink atomicMin
          // consumes all three dz (= p.z/p.w for each corner) BEFORE the early-out, so the
          // compiler cannot sink the fetch/transform/ndc/snap work past the return (it
          // would be a side-effecting use). returnIf's guard is a runtime-true compare
          // (uint ≥ 0) ⇒ always returns here, yet downstream stays REACHABLE. Isolates:
          // work-item fetch + ctx + 3× fetchWorldVert + 3× vp transform + ndc (+ snap).
          const sinkV = (dzA[0] as NF)
            .add(dzA[1] as NF)
            .add(dzA[2] as NF)
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

        const nearOK = (nearAcc as NB).toVar();

        If(nearOK.not(), () => {
          // near-plane crossing → HW path clips it (never drop, F10c)
          if (!dbgNoHw) {
            const slot = atomicAdd(
              hwQueueV.atomic.element(0),
              uint(1),
            ) as unknown as NU;
            If(slot.lessThan(uint(HW_CAP)), () => {
              const base = slot.mul(uint(2)).add(uint(1));
              atomicStore(hwQueueV.atomic.element(base), payload);
              atomicStore(hwQueueV.atomic.element(base.add(uint(1))), instId);
            });
          }
        }).Else(() => {
          // R1 — winding + two-sided re-wind decided on the INTEGER twice-area the
          // scanline already trusts. Kills the redundant float areaNdc (and edgeFn) AND the
          // reason the three ndc had to coexist — they were kept only for edgeFn + the
          // ndc-space swap. Snapping then swapping vert1↔2 == swapping ndc then snapping
          // (snap is per-vertex) ⇒ the swapped xi/yi/dz are bit-identical to the old
          // post-orientForRaster corners. The accept/flip DECISION now reads sign(area2raw)
          // instead of sign(areaNdc): identical for every triangle whose |area| clears ~1
          // quantum, and self-consistent with the integer coverage below (the old float
          // winding / integer coverage split was NOT). The only difference is a
          // sub-1/256-px sliver where float and snapped winding disagree — it covers ≤1
          // sample (rasterGate = area2>0 AND coversSample), i.e. invisible.
          const area2raw = yiA[2]
            .sub(yiA[0])
            .mul(xiA[1].sub(xiA[0]))
            .sub(xiA[2].sub(xiA[0]).mul(yiA[1].sub(yiA[0])))
            .toVar();
          const flip = ctx.twoSided.and(area2raw.lessThan(toI(0))).toVar();
          const accept = ctx.twoSided.select(
            area2raw.notEqual(toI(0)),
            area2raw.greaterThan(toI(0)),
          );
          If(accept, () => {
            // camera-facing winding: swap vert1↔vert2 (+depth) so the positive-area core
            // rasters exactly once; the re-wound twice-area is |area2raw|. These swapped
            // xi/yi/dz equal the old post-orientForRaster snapped corners by construction
            // (snap is per-vertex ⇒ swap∘snap == snap∘swap). Fixed-point verts were snapped
            // in the streaming pass above (1/256-px grid, watertight i32; f32→i32 saturates
            // ⇒ far verts read as a huge extent → HW route via smallEnough below).
            const xi0 = xiA[0];
            const yi0 = yiA[0];
            const dz0 = dzA[0];
            const xi1 = flip.select(xiA[2], xiA[1]).toVar();
            const xi2 = flip.select(xiA[1], xiA[2]).toVar();
            const yi1 = flip.select(yiA[2], yiA[1]).toVar();
            const yi2 = flip.select(yiA[1], yiA[2]).toVar();
            const dz1 = flip.select(dzA[2], dzA[1]).toVar();
            const dz2 = flip.select(dzA[1], dzA[2]).toVar();
            const area2 = flip.select(area2raw.mul(toI(-1)), area2raw).toVar();

            // whole-pixel bbox via arithmetic >>8 (Task 3): a SIGNED i32 >> is
            // sign-extending = floor(x/256), ONE shift instead of the signed /256 the
            // compiler emitted as tint_div_i32 (real integer cost, buffer-indexing math is
            // everywhere in that 47%). The operand stays i32 (uint() is NOT wrapped around
            // it — that would force a logical shift and break negative off-screen bbMin),
            // so floor holds for negatives too; vs the old trunc it can shift the SW/HW
            // split ≤1px at an off-screen bbox edge, but both paths draw identical pixels.
            const bbMinX = minI(xi0, minI(xi1, xi2)).shiftRight(uint(8)).toVar();
            const bbMaxX = maxI(xi0, maxI(xi1, xi2)).shiftRight(uint(8)).toVar();
            const bbMinY = minI(yi0, minI(yi1, yi2)).shiftRight(uint(8)).toVar();
            const bbMaxY = maxI(yi0, maxI(yi1, yi2)).shiftRight(uint(8)).toVar();

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
                // area2 (twice-area, re-wound positive) was computed at winding time
                // above — one integer area now drives orient AND the raster. A snapped
                // sub-1/256-px sliver collapses to ≤0 and is dropped by the rasterGate
                // (area2>0) below, which also guards the reciprocal (the float core
                // divided blindly).
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
                    dz0.min(dz1).min(dz2).clamp(0, 1) as unknown as NF,
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
                  // R3: edge deltas kept LAZY (no toVar) so they never pin 6 registers
                  // across the edge-setup crest — the bias (sign-only) and sx/sy (×256)
                  // consume them inline; the swcoop path (default-off) recomputes them for
                  // packEdge. Identical i32 values either way.
                  const ex0 = yi1.sub(yi2);
                  const ey0 = xi2.sub(xi1);
                  const ex1 = yi2.sub(yi0);
                  const ey1 = xi0.sub(xi2);
                  const ex2 = yi0.sub(yi1);
                  const ey2 = xi1.sub(xi0);

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
                    // per-mode emit — swScanline computes coverage + the UNBIASED-weight
                    // depth cz and hands us (px, cz); this only does the mode-specific write.
                    // (depth atomicMin Z / combined packed idLo|idHi / world1 24-bit election)
                    const emit = (px: NU, cz: NF): void => {
                      if (mode === 'depth') {
                        const bits = bcF2U(cz as unknown as NF);
                        const cur = aLoadU(visDepthV.atomic.element(px));
                        If(bits.lessThan(cur), () => {
                          atomicMin(visDepthV.atomic.element(px), bits);
                        });
                      } else if (mode === 'combined') {
                        const myDk = depthKey16(cz as unknown as NF).toVar();
                        const stored = aLoadU(visPayloadV.atomic.element(px));
                        If(myDk.greaterThan(stored.shiftRight(uint(16))), () => {
                          const dk = myDk.shiftLeft(uint(16));
                          atomicMax(
                            visPayloadV.atomic.element(px),
                            dk.bitOr(payload.bitAnd(uint(0xffff))),
                          );
                          atomicMax(
                            visBV.atomic.element(px),
                            dk.bitOr(
                              payload.shiftRight(uint(16)).bitAnd(uint(0xffff)),
                            ),
                          );
                        });
                      } else if (mode === 'world1') {
                        // 0a SCAR (?scar=1): count this covered fragment — election below is
                        // BYTE-IDENTICAL whether or not scar is on.
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
                          .bitOr(payload.bitAnd(uint(0xff)))
                          .toVar();
                        doElection(px, cand, payload);
                      }
                    };
                    swScanline(
                      rw0 as unknown as NI,
                      rw1 as unknown as NI,
                      rw2 as unknown as NI,
                      sx0 as unknown as NI,
                      sx1 as unknown as NI,
                      sx2 as unknown as NI,
                      sy0 as unknown as NI,
                      sy1 as unknown as NI,
                      sy2 as unknown as NI,
                      bias0 as unknown as NI,
                      bias1 as unknown as NI,
                      bias2 as unknown as NI,
                      dz0 as unknown as NF,
                      dz1 as unknown as NF,
                      dz2 as unknown as NF,
                      rcpArea as unknown as NF,
                      startX as unknown as NI,
                      startY as unknown as NI,
                      endX as unknown as NI,
                      endY as unknown as NI,
                      emit,
                    );
                  };
                  if (coopMode === 0) {
                    if (mode === 'world1' && !scar) {
                      // task #76 PERF: a single-pixel-bbox tri is EXACTLY one scanline
                      // iteration at (startX,startY) where cw==rw ⇒ point-sample it
                      // directly and skip the whole coverage loop (no trip-count
                      // divergence, no walk). rw0..2 already IS the biased edge value
                      // at that pixel centre ⇒ bit-identical to the 1-iteration emit
                      // (coverage rw≥0, unbiased uw=rw−bias, same cz, same px, same
                      // election). No register change (peak is the setup) — pure ALU +
                      // divergence win for the sub-pixel majority.
                      If(
                        endX
                          .lessThanEqual(startX)
                          .and(endY.lessThanEqual(startY)),
                        () => {
                          If(
                            rw0
                              .greaterThanEqual(toI(0))
                              .and(rw1.greaterThanEqual(toI(0)))
                              .and(rw2.greaterThanEqual(toI(0))),
                            () => {
                              const uw0 = rw0.sub(bias0);
                              const uw1 = rw1.sub(bias1);
                              const uw2 = rw2.sub(bias2);
                              const cz = toF(uw0 as unknown as NI)
                                .mul(dz0)
                                .add(toF(uw1 as unknown as NI).mul(dz1))
                                .add(toF(uw2 as unknown as NI).mul(dz2))
                                .mul(rcpArea)
                                .toVar();
                              If(
                                cz
                                  .greaterThanEqual(0)
                                  .and(cz.lessThanEqual(1)),
                                () => {
                                  const px = uint(startY)
                                    .mul(uint(cam.uW))
                                    .add(uint(startX));
                                  const cand = depthKey24(cz as unknown as NF)
                                    .shiftLeft(uint(8))
                                    .bitOr(payload.bitAnd(uint(0xff)))
                                    .toVar();
                                  // APPEND (px,cand,payload) to splatQueue — the lean
                                  // nanSplatElect kernel elects it. splatQueueV is non-null
                                  // on every world1 (singlePass) kernel; the guard is only
                                  // TS null-safety for the unused non-singlePass build.
                                  if (splatQueueV && !dbgNoSpl) {
                                    const slot = atomicAdd(
                                      splatQueueV.atomic.element(0),
                                      uint(1),
                                    ) as unknown as NU;
                                    If(slot.lessThan(uint(SPLAT_CAP)), () => {
                                      const b = uint(1).add(slot.mul(uint(3)));
                                      atomicStore(
                                        splatQueueV.atomic.element(b),
                                        px,
                                      );
                                      atomicStore(
                                        splatQueueV.atomic.element(
                                          b.add(uint(1)),
                                        ),
                                        cand as unknown as NU,
                                      );
                                      atomicStore(
                                        splatQueueV.atomic.element(
                                          b.add(uint(2)),
                                        ),
                                        payload,
                                      );
                                    });
                                  }
                                },
                              );
                            },
                          );
                        },
                      ).Else(() => {
                        // mid band (2..MAX_RASTER_SIZE; big/near already went to HW above) →
                        // append the pre-projected, pre-wound corners to midQueue. nanMidRaster
                        // rasterizes them ONCE via the shared swScanline. world1 itself now runs
                        // NO coverage loop — it only projects + routes.
                        if (midQueueV && !dbgNoMid) {
                          const slot = atomicAdd(
                            midQueueV.atomic.element(0),
                            uint(1),
                          ) as unknown as NU;
                          If(slot.lessThan(uint(MID_CAP)), () => {
                            const mb = uint(1).add(slot.mul(uint(MID_STRIDE)));
                            atomicStore(
                              midQueueV.atomic.element(mb),
                              bcI2U(xi0 as unknown as NI),
                            );
                            atomicStore(
                              midQueueV.atomic.element(mb.add(uint(1))),
                              bcI2U(yi0 as unknown as NI),
                            );
                            atomicStore(
                              midQueueV.atomic.element(mb.add(uint(2))),
                              bcI2U(xi1 as unknown as NI),
                            );
                            atomicStore(
                              midQueueV.atomic.element(mb.add(uint(3))),
                              bcI2U(yi1 as unknown as NI),
                            );
                            atomicStore(
                              midQueueV.atomic.element(mb.add(uint(4))),
                              bcI2U(xi2 as unknown as NI),
                            );
                            atomicStore(
                              midQueueV.atomic.element(mb.add(uint(5))),
                              bcI2U(yi2 as unknown as NI),
                            );
                            atomicStore(
                              midQueueV.atomic.element(mb.add(uint(6))),
                              bcF2U(dz0 as unknown as NF),
                            );
                            atomicStore(
                              midQueueV.atomic.element(mb.add(uint(7))),
                              bcF2U(dz1 as unknown as NF),
                            );
                            atomicStore(
                              midQueueV.atomic.element(mb.add(uint(8))),
                              bcF2U(dz2 as unknown as NF),
                            );
                            atomicStore(
                              midQueueV.atomic.element(mb.add(uint(9))),
                              payload,
                            );
                          });
                        }
                      });
                    } else {
                      oldScanline();
                    }
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
                                      dz0,
                                      dz1,
                                      dz2,
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
                      wgW(s.z0, localTri, dz0);
                      wgW(s.z1, localTri, dz1);
                      wgW(s.z2, localTri, dz2);
                      wgW(s.rcp, localTri, rcpArea);
                    });
                  }
                });
              },
            ).Else(() => {
              if (!dbgNoHw) {
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
              }
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
        //workgroupBarrier(); // Phase A records complete before the compaction scan
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
        //workgroupBarrier(); // compacted list visible to all lanes
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
  // Task 1: the per-cluster ctx PRE-PASS kernel (kClusterCtx) is now built by
  // buildClusterCtx (./raster/ClusterCtx) at the top of buildNaniteRaster.
  // ?ksplit (PERF task #76): the two class-specialized world1 kernels. Named per variant
  // so a capture reads the split cleanly (nanRasterWorld1Explicit = the leaf/trunk whale
  // shed of terrainDispAt; nanRasterWorld1Terrain = heightfield, no transform/wind). Both
  // dispatch over the FULL queue (rasterDispatchFull) and route by the isHF early-out —
  // the Stage-1 measurement form (byte-identical output; Stage 2 adds the cull-side
  // partition that drops the doubled launches). Built only under ?ksplit (0 cost off).
  const kRasterWorld1Explicit = ksplit ? rasterKernel('world1', 'explicit') : null;
  const kRasterWorld1Terrain = ksplit ? rasterKernel('world1', 'terrain') : null;
  if (kRasterWorld1Explicit) {
    (kRasterWorld1Explicit as ComputeKernel).setName('nanRasterWorld1Explicit');
    setIndirectDispatch(kRasterWorld1Explicit, cull.rasterDispatchFullAttr);
  }
  if (kRasterWorld1Terrain) {
    (kRasterWorld1Terrain as ComputeKernel).setName('nanRasterWorld1Terrain');
    setIndirectDispatch(kRasterWorld1Terrain, cull.rasterDispatchFullAttr);
  }

  // ---- deferred-raster consumers (task #76) ----------------------------------------
  // kHwArgs is built by buildHw (below); kSplatArgs/kMidArgs by buildQueues (above).
  // nanSplatElect (sub-pixel election, ./raster/Splat) + nanMidRaster (mid-band scanline,
  // ./raster/Mid) are the two lean consumers — they read the queues world1 filled and run
  // the shared swScanline + election. null on the non-world1 (non-singlePass) paths.
  const kSplatElect = buildSplat({ splatQueueV, splatDrawAttr, elect });
  const kMidRaster = buildMid({
    midQueueV,
    midDrawAttr,
    swScanline,
    elect,
    width,
    height,
  });

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

  // ---- HW big/near-triangle path (./raster/Hw): the vertex-pulling materials + scenes +
  // render wrappers + kHwArgs. `clhw` preserved exactly (the instanced per-cluster draw is
  // built only when the cull supplied qHwRaster/hwClusterDraw). The fragment stage writes
  // the SAME vis buffers via the shared election ⇒ resolve unchanged.
  const hw = buildHw({
    cam,
    width,
    height,
    nfetch,
    hw1fetch,
    qRasterRO,
    vis,
    hwQueueV,
    hwDrawAttr,
    hwDrawBuf,
    elect,
    scar,
    scarEl,
    hwrt,
    clhw,
    qHwRasterRO: cull.qHwRasterRO ?? null,
    hwClusterDrawAttr: cull.hwClusterDrawAttr ?? null,
  });
  const {
    kHwArgs,
    hwDepthMat,
    hwCombinedMat,
    hwWorld1Mat,
    hwRender,
    hwRenderCluster,
  } = hw;

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
  // hwRender / hwRenderCluster now come from buildHw (./raster/Hw), destructured above.
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
      // Task 1: the ctx PRE-PASS runs right after the clear, BEFORE the raster — its
      // global ctx writes are visible to the raster's cache-coherent reads via the same
      // in-pass storage sync that carries kVisClear→raster. Removes the raster's thread-0
      // makeCtx divergence + barrier. Only present on the world1 (singlePass) instance.
      ...(kClusterCtx ? [kClusterCtx as unknown] : []),
      // ?ksplit: the two class kernels REPLACE the unified world1 in the SAME compute
      // pass — no barrier between them, so the order-independent atomic election merges
      // both, and (because they share vis via atomics, not disjoint buffers) they are
      // overlap-eligible on the GPU. Stage-3 measures whether the GPU actually overlaps.
      ...(ksplit
        ? [kRasterWorld1Explicit as unknown, kRasterWorld1Terrain as unknown]
        : [kRasterWorld1]),
      ...(grass?.batch ?? []),
      kHwArgs,
      // ?splat: build the sub-pixel append indirect args, then run the lean election.
      // After the raster (RAW on splatQueue); before the HZB tail reads visPayloadV.
      // Order vs kHwArgs/grass is irrelevant (disjoint buffers). In-pass auto-sync
      // carries raster→kSplatArgs→kSplatElect exactly as the cull BFS does.
      ...(kSplatArgs ? [kSplatArgs as unknown] : []),
      ...(kSplatElect ? [kSplatElect as unknown] : []),
      // ...and the mid-band (2..swmax): build its indirect args, then rasterize the queued
      // corners via the shared swScanline. After the raster (RAW on midQueue), before the
      // HZB tail reads visPayloadV; elects order-independently alongside splat/HW.
      ...(kMidArgs ? [kMidArgs as unknown] : []),
      ...(kMidRaster ? [kMidRaster as unknown] : []),
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
