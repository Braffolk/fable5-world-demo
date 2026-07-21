/**
 * N2 vis-buffer raster on REGISTRY buffers — GeometryRegistry mega-buffers +
 * the world heightfield texture, consuming the N2 cull chain's work queue.
 * Option C only (D-N5): pass 1
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
import type { PerspectiveCamera } from 'three';
import type { TerrainField } from '../world/TerrainField';
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
import type { NB, NF, NI, NU, NV3, NV4 } from '../../gpu/TSLTypes';
import {
  CLUSTER_TRI_BITS,
  CLUSTER_TRI_MASK,
  CLUSTER_WORDS,
  MAX_CLUSTER_TRIS,
  MESH_WORDS,
} from '../world/GeometryRegistry';
import type { RegistryGpu } from '../world/GeometryRegistry';
import {
  CLHW_MAX,
  DISPATCH_ROW,
  QRASTER_CAP,
  hashColor,
  type NaniteCam,
} from '../NaniteCommon';
import {
  makeFetch,
  type TrunkWindOpt,
  type VertCtx,
} from './NaniteFetch';
import { clusterHwClass, HWPROJ } from '../cull/NaniteHwClass';
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
  maxI,
  minI,
  readBuffer,
  returnIf,
  sU32Views,
  setIndirectDispatch,
  toF,
  toI,
  wgLinear,
} from '../Tsl';
import type { BufOf, UniformV3, UV2 } from '../Tsl';
// ─── extracted raster/ modules (task #76 vis-buffer rewrite, Step 1) ──────────────
import { CTX_PROJ_BASE, CTX_STRIDE, CTX_U, buildClusterCtx } from './ClusterCtx';
import { buildHw } from './Hw';
import { buildMid } from './Mid';
import {
  NEAR_SENTINEL,
  PROJ_CLUSTER_CAP,
  PROJ_INVALID_BASE,
  buildProject,
  canonVertSlot,
} from './Project';
import {
  HW_CAP,
  MID_CAP,
  MID_STRIDE,
  SPLAT_CAP,
  buildQueues,
} from './Queues';
import { makeScanline } from './Scanline';
import { buildSplat } from './Splat';
import {
  buildVisClear,
  depthKey16,
  depthKey24,
  makeElect,
  type NaniteVisBuffers,
} from './VisBuffer';
export { makeVisBuffers } from './VisBuffer';
export type { NaniteVisBuffers } from './VisBuffer';

// HW_CAP / SPLAT_CAP / MID_CAP / MID_STRIDE now live in ./raster/Queues (imported).
// ?swmax=N shrinks the SW/HW split (bbox extent ≤N px stays SW; larger → HW):
// lets the HW rasterizer eat the mid-size (5-16px) tris. Default 16 = shipped behavior.
const MAX_RASTER_SIZE = (() => {
  const v = Number(
    new URLSearchParams(window.location.search).get('swmax') ?? '16',
  );
  return Number.isFinite(v) && v >= 2 ? Math.floor(v) : 16;
})();
const NEAR_EPS = 1e-4;

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
  /** F3 triangle-budget: RAW splat/mid queue append counts at slot [0] (pre-cap).
   *  null when the queues don't exist (single-pass world1 only). */
  readSplatCount(renderer: Renderer): Promise<number | null>;
  readMidCount(renderer: Renderer): Promise<number | null>;
  /** Raw compact projected-record demand. Values above PROJ_RECORD_CAP mean clusters
   *  were deliberately dropped fail-closed rather than indexing outside the pool. */
  readProjectRecordCount(renderer: Renderer): Promise<number | null>;
  /** count covered/orphan pixels (NaniteView ?audit=1) */
  audit(renderer: Renderer): void;
  readAudit(renderer: Renderer): Promise<{ orphans: number; covered: number }>;
  /** Stage-2 voxel BRICK-WRITE count (the occlusion-skip overlay number, §A2) — total
   *  per-pixel wgElect wins across all tiles this frame. null when no voxel raster. */
  readVoxWrites(renderer: Renderer): Promise<number | null>;
}

// NaniteVisBuffers + makeVisBuffers now live in ./raster/VisBuffer (imported +
// re-exported above so existing callers keep importing them from './NaniteRaster').

export function buildNaniteRaster(
  gpu: RegistryGpu,
  /** terrain height source: the TerrainField plane pyramid */
  heightSrc: TerrainField,
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
    /** bucketed scatter (?voxprev): per-bucket (base,count) + per-bucket indirect args +
     *  the build-time K + the on/off flag (cull.voxBucketRangeRO / voxBucketDispatchAttr /
     *  voxF2bK / voxF2bEnabled). Present whenever the voxel queue is. */
    voxBucketRangeRO?: BufOf<UV2>;
    voxBucketDispatchAttr?: IndirectStorageBufferAttribute[];
    voxF2bK?: number;
    voxF2bEnabled?: boolean;
    voxPrevEnabled?: boolean;
    /** per-cluster SW/HW split (NaniteCull). qHwRasterRO = HW-cluster qRaster indices;
     *  hwClusterDrawAttr = the instanced draw args. Present on every buildNaniteCull cull
     *  (the split is the permanent default); the SW world1 kernel skips these clusters +
     *  the instanced HW draw paints them properly. ABSENT only on the shadow-clipmap queue
     *  (ClipLevelQueue, depth-only) which never runs the HW-cluster instanced draw. */
    qHwRasterRO?: StorageBufferNode<'uint'> | null;
    hwClusterDrawAttr?: IndirectStorageBufferAttribute | null;
    /** A2 (2026-07-09 HW vertex-prepass): the terrain-class `_cl` draw args + qHwRaster cap.
     *  Present on every full cull; consumed only by the singlePass camera world1 `_cl` split. */
    hwClusterDrawTerrainAttr?: IndirectStorageBufferAttribute | null;
    hwRasterCap?: number;
  },
  vis: NaniteVisBuffers,
  tint: 'flat' | 'cluster' | 'lod',
  /** false (?shade=0): pure matClass color, no lambert — the parity gate's
   *  shading-free mode (coverage/structure compare only) */
  shade = true,
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
    /** Non-null only when this grass graph consumes a rasterized outer shell. */
    shellHeight: number | null;
  },
  /** S6e: render anchor A for the terrain FIELD sampling in makeFetch — the S6d
   *  anchor-relative vert coords are re-absoluted to hit the world-anchored field
   *  planes. Streamed only; omitted ⇒ the verbatim absolute (generated) build. */
  fieldAnchor?: UniformV3,
): NaniteRasterHandles {
  const { width, height } = cam;
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
  // per-cluster SW/HW split (see docs/mobile-gpu-perf/SW-HW-CLUSTER-AUDIT.md). The SW world1
  // kernel skips clusters the cull classified HW (clusterHwClass, bit-identical to
  // kHwPartition ⇒ no holes/double), and an instanced HW draw paints those clusters properly.
  // The split is the PERMANENT DEFAULT — it runs whenever the cull carries the HW partition
  // (every buildNaniteCull cull: camera/view/shadow). The ONLY cull without it is the
  // shadow-clipmap queue (ClipLevelQueue, depth-only), which never runs the world1 path or
  // the HW-cluster draw — so this presence check keeps that path untouched.
  const clhw = !!cull.qHwRasterRO && !!cull.hwClusterDrawAttr;
  const clhwMax = CLHW_MAX;
  // projK = px per world-unit at unit depth (matches NaniteCull's cut projection exactly).
  const projK = cam.cotHalfFov.mul(float(cam.uH)).mul(0.5) as unknown as NF;
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
  // unchanged, vis buffers unchanged, color target never read.
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
  // hw/splat/mid work queues + their indirect-args kernels (./raster/Queues). The splat/mid
  // queues + args exist only on the world1 path (splatElect). Caps (HW_CAP/SPLAT_CAP/
  // MID_CAP/MID_STRIDE) are imported from Queues.
  const queues = buildQueues({ splatElect });
  const {
    hwQueueAttr,
    hwQueueV,
    hwDrawAttr,
    hwDrawBuf,
    splatQueueAttr,
    splatQueueV,
    splatDrawAttr,
    midQueueAttr,
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

  // ---- shared fetch helpers (NaniteFetch.ts — also the resolve's decode) ----------
  const nfetch = makeFetch(gpu, heightSrc, wind, true, 'both', fieldAnchor);
  // fetchWorldVertDyn is used only by the HW vertex stage (./raster/Hw, via nfetch).
  const { makeCtx, fetchWorldVert } = nfetch;
  // A2 (2026-07-09 HW vertex-prepass): CLASS-SPLIT fetch variants for the two `_cl` world1
  // draws — 'explicit' (leaf/trunk/rock) compiles only explicitWorldByIndex, 'terrain' (isHF)
  // only the heightfield arm — so each `_cl` vertex shader sheds the OTHER class's fetch-union
  // registers (same trick ?ksplit uses for the SW kernel). Consumed by buildHw ONLY on the
  // world1 (ctxPrepass / clusterCtxV != null) camera path; the flat ctx read replaces makeCtx.
  const nfetchExplicit = makeFetch(gpu, heightSrc, wind, true, 'explicit', fieldAnchor);
  const nfetchTerrain = makeFetch(gpu, heightSrc, wind, true, 'terrain', fieldAnchor);
  // HW vertex-prepass (2026-07-09, DEFAULT ON): the `_clE` mesh HW draw reads its verts
  // pre-projected from projVertBuf (w=1 screen-linear clip) instead of re-running the
  // compute-fetch + wind + vp + snap path. `?hwproj=0` is the disable-only escape back to
  // the compute-fetch vertex — selected at BUILD time in buildHw (the other body is never
  // compiled), never a runtime shader branch.
  const hwproj = HWPROJ; // opt-in flagship — single source of truth in NaniteHwClass

  // ?ksplit (PERF task #76): build world1 as TWO class-specialized kernels —
  // 'explicit' (leaf/trunk/rock) + 'terrain' (heightfield) — each compiling ONLY its
  // own fetch arm (makeFetch variant), so it reserves ONLY its own registers (the leaf
  // kernel sheds the explicit-mesh decode; the terrain kernel sheds the transform+wind set = the
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
  // pre-pass (kClusterCtx, built below) that computes makeCtx once and writes the 36-word
  // ctx (13 u32 + 23 f32-as-bits) to a global buffer; the raster then reads it with
  // cache-coherent loads (itemIdx is UNIFORM per workgroup ⇒ one L1 line for all 128
  // lanes) — no thread-0 block, no barrier. BONUS: makeCtx leaving the raster SHEDS
  // gpu.clusters/instances/meshes from its bindings (the fetch only needs verts/indices/
  // height) ⇒ 9 storage buffers, under the 10 ceiling. WORLD1 (singlePass) only — depth/
  // combined/view keep the broadcast and never allocate the buffer. Buffer = 96 Ki clusters ×
  // 36 u32 = 13.5 MiB; every visible cluster is rewritten each frame
  // so the initial contents are irrelevant.
  const ctxPrepass = singlePass;
  // ?wgcache bool→uint for packing isHF/isDAG into the shared-memory uint array (shared by
  // the world1 broadcast AND the ClusterCtx pre-pass, kept identical so both pack the same).
  const b2u = (b: NB): NU =>
    (b as unknown as { select(a: NU, c: NU): NU }).select(uint(1), uint(0));
  // per-cluster ctx PRE-PASS (./raster/ClusterCtx): CTX_U/CTX_STRIDE (imported) + the
  // buffer + nanClusterCtxPrepass. WORLD1 (singlePass) only; the world1 raster reads
  // clusterCtxV with cache-coherent loads (no thread-0 makeCtx, no barrier).
  const { clusterCtxV, kClusterCtx } = buildClusterCtx({
    ctxPrepass,
    gpu,
    cam,
    qRasterRO,
    makeCtx,
    projK,
    clhwMax,
    wind,
    b2u,
  });

  // ─── Step 2 (vis-buffer rewrite): the per-VERTEX projection PRE-PASS (./raster/Project).
  // Runs the SAME transform + wind + clip → NDC → 1/256-px snap world1 did inline per tri-
  // corner, ONCE per corner, into projVertBuf; the world1 raster below READS the 3 pre-
  // projected corners by slot instead of projecting (its 64-reg projection floor deleted).
  // Bit-identical: same ctx (from clusterCtxV), same nfetch.fetchWorldVert, VERBATIM
  // projectVert(). WORLD1 (singlePass = ctxPrepass) only; null everywhere else. See the file
  // header for the per-CORNER slotting rationale (terrain has no universal dedup key).
  const {
    projVertAttr,
    projAllocAttr,
    projVertV,
    kProjectClear,
    kProjectVerts,
  } = buildProject({
    ctxPrepass,
    cam,
    qRasterRO,
    clusterCtxV,
    nfetch,
    hasWind: !!wind,
    rasterDispatchFullAttr: cull.rasterDispatchFullAttr,
    indices: gpu.indices,
    vcompact: gpu.vcompact,
  });

  // ---- kVisClear ------------------------------------------------------------------
  // In the world single-pass path NOTHING reads or writes visDepthV (the election lives
  // in visPayloadV; HZB/resolve/shadowHalf read payload, never depthV), so its 3.3M-pixel
  // 0xffffffff clear is pure store traffic and is SKIPPED. Auto-kept under every debug
  // flag that DOES read it (nanprobe/audit/rdbg). Shadow/View raster instances have
  // singlePass=false ⇒ never skipped.
  const dvParams = new URLSearchParams(window.location.search);
  const skipDepthClear =
    singlePass &&
    (grass?.shellHeight ?? null) === null &&
    dvParams.get('nanprobe') !== '1' && // probe reads vis.depthV.ro
    dvParams.get('audit') !== '1' && // kAudit reads visDepthV.ro
    rdbg === 0; // rdbg sinks atomicMin depthV
  // kVisClear (full-screen) + kHwQueueClear (the counter-only tail, reused by the scoped
  // shadow clear) live in ./raster/VisBuffer — both reset the queue + audit counters.
  const { kVisClear, kHwQueueClear } = buildVisClear({
    pixelCount,
    vis,
    hwQueueV,
    auditV,
    splatQueueV,
    midQueueV,
    packedClear,
    skipDepthClear,
  });

  // ---- SW raster kernels (fixed-point integer scanline) ------------------------------
  // mode 'depth' = atomicMin Z only (the SHADOW depth-only path). 'combined' = single-pass
  // packed Z+id for the NaniteView debug view (16-bit depth split idLo/idHi). 'world1' =
  // the WORLD single pass (PERF-VB4): a 24-bit depth election (visPayloadV) whose winner
  // stores the full 25-bit id into the side buffer visBV; the resolve reconstructs depth
  // from the election key. No exact depthV (a 3rd hot-loop atomic buffer = a 3× cliff).
  // swScanline (./raster/Scanline): THE single fixed-point coverage loop, shared by the
  // inline depth/combined/world1 paths — no duplicate rasterisation. Pass NO opts here so
  // this instance is BYTE-IDENTICAL to the task-#76 extraction (world1's 56-reg path).
  const swScanline = makeScanline(cam);
  // nanMidRaster gets its OWN instance with the mid-only crest cut: A.3 packed bias
  // (bit-identical) always on. world1/depth/combined keep the plain `swScanline` above
  // untouched.
  const swScanlineMid = makeScanline(cam, {
    packBias: true,
  });

  const rasterKernel = (
    mode: 'depth' | 'combined' | 'world1',
    // ?ksplit: 'explicit'|'terrain' builds a class-specialized world1 kernel whose fetch
    // compiles ONLY that arm; undefined = the unified kernel (byte-identical default).
    splitVariant?: 'explicit' | 'terrain',
  ): unknown => {
    // Variant-specific fetch: the specialized kernel decodes and transforms through the
    // arm-selected makeFetch so it never reserves the other class's registers.
    // splitVariant===undefined ⇒ the module nfetch (identical node graph to the
    // pre-split kernel).
    const kFetch = splitVariant
      ? makeFetch(gpu, heightSrc, wind, true, splitVariant, fieldAnchor)
      : nfetch;
    const kMakeCtx = kFetch.makeCtx;
    const kn = Fn(() => {
      const itemIdx = wgLinear(DISPATCH_ROW).toVar();
      const localTri = localX().toVar();
      const itemCount = qRasterRO.element(0).x;
      returnIf(itemIdx.greaterThanEqual(itemCount));
      // The compact ctx/projection chain owns only PROJ_CLUSTER_CAP cluster entries. A cluster
      // past the cap has neither ctx nor projected records, so skip it before either is read —
      // a uniform per-workgroup early-out (before any barrier).
      // Parity requires the frame's visible-cluster count ≤ PROJ_CLUSTER_CAP (see Project.ts).
      if (mode === 'world1' && ctxPrepass) {
        returnIf(itemIdx.greaterThanEqual(uint(PROJ_CLUSTER_CAP)));
      }
      const item = qRasterRO.element(itemIdx.add(uint(1)));
      const instId = item.x.toVar();
      const ci = item.y.toVar();
      // SW/HW split: skip clusters the cull routed to the HW instanced draw. The decision is
      // clusterHwClass — BIT-IDENTICAL to kHwPartition's — so the SW-skip and the HW-append
      // can never disagree (no holes, no double-raster) without touching qRaster. UNIFORM
      // across the workgroup (instId/ci are per-cluster) and BEFORE any barrier ⇒ every live
      // thread returns together, no barrier deadlock. `clhw` = the cull carries the partition
      // (true on every world1 = camera raster; false only on the partition-less shadow-clipmap
      // world1 kernel, which is never dispatched). HW clusters are painted by the instanced HW
      // draw, so the SW kernel returns them here. With wgcache ON (default) the classify is
      // computed ONCE on thread 0 and broadcast (slot 11, below) — NOT 128×/cluster on the SW
      // hot path. The ?wgcache=0 fallback classifies per-thread (uniform ⇒ all threads return).
      if (clhw && mode === 'world1' && !wgcache) {
        returnIf(clusterHwClass(gpu, cam, projK, instId, ci, clhwMax));
      }
      // world1 per-covered-pixel election: relaxed atomicLoad guard of visPayloadV →
      // If(cand>prev) → atomicMax(visPayloadV) → winner-conditional atomicStore of visBV.
      const doElection = (px: NU, cand: NU, idStore: NU): void => {
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
      let projectedBase: NU | null = null;
      // matClass (F): broadcast from the wgcache thread-0 decode; null when wgcache
      // is off, so the voxel-skip below falls back to its per-thread mesh reload.
      // Bit-identical — the broadcast is the SAME mesh word6 extract, computed once.
      let matClassBroadcast: NU | null = null;
      if (ctxPrepass && mode === 'world1' && clusterCtxV) {
        // Task 1: read the per-cluster ctx the PRE-PASS (kClusterCtx) wrote to global
        // memory — cache-coherent loads (itemIdx uniform per workgroup ⇒ one L1 line for
        // all 128 lanes), NO thread-0 makeCtx, NO workgroup barrier. Same layout as
        // the old broadcast; makeCtx no longer runs here so gpu.clusters/instances/meshes
        // are not bound to this kernel.
        const base = itemIdx.mul(uint(CTX_STRIDE)).toVar();
        const rU = (i: number): NU =>
          elemU(clusterCtxV.rw, base.add(uint(i))).toVar() as unknown as NU;
        const rF = (i: number): NF =>
          bcU2F(elemU(clusterCtxV.rw, base.add(uint(CTX_U + i)))).toVar() as unknown as NF;
        const iHF = rU(0);
        projectedBase = rU(CTX_PROJ_BASE);
        returnIf(projectedBase.equal(uint(PROJ_INVALID_BASE)));
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
        // shared-resident ctx: read fields on-demand from on-chip threadgroup memory (cheap on
        // Apple) rather than pinning ~30 of them into registers — world1 is register-bound, and
        // the ctx already lives in shU/shF, so this trades the scarce resource (registers) for
        // the spare one (shared reads). Read-only after the broadcast barrier.
        const tU = (i: number): NU => getU(i);
        const tF = (i: number): NF => getF(i);
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
              b2u(clusterHwClass(gpu, cam, projK, instId, ci, clhwMax)),
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
        // cluster ⇒ every live thread returns together (post-barrier) — no
        // partial-workgroup barrier, same shape as the clhw skip above.
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

      const corner = (lt: NU, v: 0 | 1 | 2): NV3 => kFetch.fetchWorldVert(ctx, lt, v);

      // voxel-foliage (spec §4.1 / §A1): SKIP voxel(7) clusters in the TRIANGLE raster.
      // The cut emits voxel clusters into the SAME qRaster as triangles (§4.6); the
      // post-traverse fan-out copies them to qVoxRaster for the Stage-2 voxel bin, but
      // they REMAIN in qRaster. A voxel cluster's word6/word7 point at BRICKS, not tris
      // (registerVoxelHead), so fetchWorldVert would read garbage triangle data —
      // bail before any vertex work. UNIFORM across the workgroup (matClass is per
      // cluster, broadcast via ctx.meshId), so every live thread returns (no barrier
      // deadlock; the wgcache barrier above already ran). The bricks render via the
      // scatter voxel raster (Stage 2) into the same vis buffers.
      {
        // matClass (F): reuse the wgcache broadcast when present; else per-thread reload.
        const mcVox =
          matClassBroadcast ??
          elemU(gpu.meshes, ctx.meshId.mul(uint(MESH_WORDS)).add(uint(6)))
            .shiftRight(uint(8))
            .bitAnd(uint(0xff));
        returnIf(mcVox.equal(uint(7)));
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

      If(localTri.lessThan(ctx.triCount), () => {
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
        let nearAcc: NB | null = null;
        if (mode === 'world1' && ctxPrepass && projVertV) {
          // ── Step 2 (vis-buffer rewrite): world1 no longer PROJECTS here — it READS the 3
          // pre-projected corners nanProjectVerts wrote to projVertBuf THIS frame. That
          // deletes the inline fetch + vp transform + ndc + snap = world1's 64-reg projection
          // floor (the register win). Slot = canonVertSlot (DEDUPED: mesh reads gpu.indices for
          // vi−vBase, terrain per-corner); record = xi(i32) | yi(i32) | dz(f32). Bit-identical:
          // nanProjectVerts ran the VERBATIM projectVert() on the SAME fetchWorldVert(ctx,
          // localTri,v) world (same ctx) into the SAME canonical slot, so these xi/yi/dz
          // equal world1's old inline values exactly. A near-crossing corner carries NEAR_SENTINEL
          // in its dz word ⇒ nearOK = AND(dz ≠ sentinel) reproduces the old w>NEAR_EPS gate EXACTLY
          // (raw-u32 compare; the garbage dz is unread when nearOK is false — the tri routes HW).
          // world1 re-binds gpu.indices ONLY (for the mesh dedup key) ⇒ 7 → 8 storage buffers.
          const recCluster = (projectedBase as NU).toVar();
          for (const v of [0, 1, 2] as const) {
            const rb = canonVertSlot(
              recCluster,
              ctx.triStart as unknown as NU,
              ctx.isHF as unknown as NB,
              localTri,
              v,
              gpu.indices,
            ).toVar();
            const zw = elemU(projVertV.rw, rb.add(uint(2))).toVar();
            xiA[v] = bcU2I(
              elemU(projVertV.rw, rb),
            ).toVar() as unknown as NI;
            yiA[v] = bcU2I(
              elemU(projVertV.rw, rb.add(uint(1))),
            ).toVar() as unknown as NI;
            dzA[v] = bcU2F(zw).toVar() as unknown as NF;
            const okv = zw.notEqual(uint(NEAR_SENTINEL)) as unknown as NB;
            nearAcc = nearAcc ? nearAcc.and(okv) : okv;
          }
        } else {
          // ── streaming vertex assembly (depth/combined — inline projection, VERBATIM): each
          // corner flows clip→ndc→snap and its p/ndc/screen intermediates DIE before the next
          // is built, so the crest never holds three vertices in four representations at once.
          // nearOK folds in incrementally. A near-plane-crossing vertex snaps to a saturated/
          // garbage i32 that is DISCARDED — nearOK routes the whole triangle HW before any
          // fixed-point value is trusted (the garbage is never read on the HW path).
          const W = float(cam.uW);
          const H = float(cam.uH);
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
        }

        const nearOK = (nearAcc as NB).toVar();

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
                const rasterGate = area2.greaterThan(toI(0)).and(coversSample);
                If(rasterGate, () => {
                  // edge i is opposite vertex i; ex/ey = dE per +1 UNIT (1/256 px)
                  // R3: edge deltas kept LAZY (no toVar) so they never pin 6 registers
                  // across the edge-setup crest — the bias (sign-only) and sx/sy (×256)
                  // consume them inline.
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

                  // Row-span scanline as an emit closure (the mid/large-tri path).
                  const scanline = (): void => {
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
                  if (mode === 'world1') {
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
                                  if (splatQueueV) {
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
                        // append ONLY the 1-u32 tri id (payload). nanMidRaster re-reads the tri's
                        // 3 corners from projVertBuf (via canonVertSlot, the SAME deduped slot the classifier
                        // read), re-derives the identical winding + edge-setup, and rasterizes ONCE
                        // via the shared swScanline. Storing just the id (was 10 u32 of already-
                        // projected/wound corners) is the ~10× midQueue shrink — the projected
                        // verts already live in projVertBuf, so re-reading them is free of any new
                        // projection. world1 itself runs NO coverage loop — it only projects + routes.
                        if (midQueueV) {
                          const slot = atomicAdd(
                            midQueueV.atomic.element(0),
                            uint(1),
                          ) as unknown as NU;
                          If(slot.lessThan(uint(MID_CAP)), () => {
                            // MID_STRIDE===1 ⇒ no dead `*1u` in the hot append index.
                            const mOff = (
                              MID_STRIDE === 1 ? slot : slot.mul(uint(MID_STRIDE))
                            ) as unknown as NU;
                            const mb = uint(1).add(mOff);
                            atomicStore(midQueueV.atomic.element(mb), payload);
                          });
                        }
                      });
                    } else {
                      scanline();
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
  // shed of the terrain field taps; nanRasterWorld1Terrain = heightfield, no transform/wind). Both
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
    projVertV,
    clusterCtxV,
    indices: gpu.indices,
    swScanline: swScanlineMid,
    elect,
    width,
    height,
  });

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

  // ---- HW big/near-triangle path (./raster/Hw): the vertex-pulling materials + scenes +
  // render wrappers + kHwArgs. The instanced per-cluster draw is built whenever the cull
  // supplied qHwRaster/hwClusterDraw (every camera/view/shadow cull; absent only on the
  // shadow-clipmap queue). The fragment stage writes the SAME vis buffers via the shared
  // election ⇒ resolve unchanged.
  const hw = buildHw({
    cam,
    width,
    height,
    nfetch,
    qRasterRO,
    vis,
    hwQueueV,
    hwDrawAttr,
    hwDrawBuf,
    elect,
    qHwRasterRO: cull.qHwRasterRO ?? null,
    hwClusterDrawAttr: cull.hwClusterDrawAttr ?? null,
    // A2 (2026-07-09): flat per-cluster ctx read (Option A) + the class-split `_cl` draws.
    // clusterCtxV non-null ONLY on the world1 (singlePass) camera path ⇒ the split renders
    // there; every other raster keeps the single 'both' makeCtx `_cl` material (dormant).
    clusterCtxV,
    hasWind: !!wind,
    nfetchExplicit,
    nfetchTerrain,
    hwClusterDrawTerrainAttr: cull.hwClusterDrawTerrainAttr ?? null,
    hwRasterCap: cull.hwRasterCap ?? QRASTER_CAP,
    // HW vertex-prepass: the `_clE` mesh draw reads the SAME projected-vert records the SW
    // classifier + Mid consume (non-null only on the world1 / ctxPrepass path, like clusterCtxV).
    projVertV,
    projBaseSlot: CTX_PROJ_BASE,
    indices: gpu.indices,
    hwproj,
    grassShellHeight: grass?.shellHeight ?? null,
  });
  const {
    kHwArgs,
    kGrassShellArgs,
    hwDepthMat,
    hwCombinedMat,
    hwWorld1Mat,
    hwRender,
    hwRenderCluster,
    hwRenderGrassShell,
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
          // bucketed scatter (?voxprev): the cull always publishes these alongside the voxel queue.
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
      ...(kProjectClear ? [kProjectClear as unknown] : []),
      // Step 2 (vis-buffer rewrite): the per-VERTEX projection PRE-PASS runs after the ctx
      // pre-pass, BEFORE the raster/classifier — project → classify order. Its projVertBuf
      // writes are visible to the raster's reads via the same in-pass storage sync the batch
      // gives clear→ctx→raster. Only present on the world1 (singlePass) instance.
      ...(kProjectVerts ? [kProjectVerts as unknown] : []),
      ...(kGrassShellArgs ? [kGrassShellArgs as unknown] : []),
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
    // paint the big/near CLUSTERS the SW pass skipped, as proper instanced HW draws (one
    // instance/cluster, single transform). Same election ⇒ folds into the vis buffers
    // alongside the SW + soup winners, before grass/voxel pre-seed from them. (world1 is
    // the camera path ⇒ the cull always carries the partition; hwRenderCluster self-guards.)
    hwRenderCluster(renderer, camera);
    // The source method starts at the rasterized outer shell. Its depth lands
    // in the otherwise-unused vis.depthV before grassRay consumes it.
    if (grass?.enabled()) hwRenderGrassShell(renderer, camera);
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
  // F3 triangle-budget: the RAW per-layer append counts at queue slot [0] (pre-cap;
  // may exceed the cap ⇒ overflow/drops). splat = sub-pixel fragment appends (≈1px each),
  // mid = 2..swmax SW-raster tri records. null when the queues don't exist (!splatElect).
  const readSplatCount = async (renderer: Renderer): Promise<number | null> => {
    if (!splatQueueAttr) return null;
    const buf = await readBuffer(renderer, splatQueueAttr, 0, 4);
    return new Uint32Array(buf)[0] ?? 0;
  };
  const readMidCount = async (renderer: Renderer): Promise<number | null> => {
    if (!midQueueAttr) return null;
    const buf = await readBuffer(renderer, midQueueAttr, 0, 4);
    return new Uint32Array(buf)[0] ?? 0;
  };
  const readProjectRecordCount = async (
    renderer: Renderer,
  ): Promise<number | null> => {
    if (!projVertAttr || !projAllocAttr) return null;
    const buf = await readBuffer(renderer, projAllocAttr, 0, 4);
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
    readSplatCount,
    readMidCount,
    readProjectRecordCount,
    audit,
    readAudit,
    readVoxWrites,
  };
}
