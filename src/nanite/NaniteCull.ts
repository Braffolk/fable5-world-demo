/**
 * N2 culling chain (NANITE-SPEC.md "Culling (N2)") — registry-fed HIERARCHICAL
 * DAG-BFS cull (N8-HIC). One pass-set walks the cluster DAG top-down instead of
 * brute-forcing every cluster of every instance (the legacy two-level two-phase
 * path — kInstCull→chunk→kClusterCull + reject re-tests — was deleted once the
 * world camera AND the shadow culls all moved to hier: PERF-VB3 / SHADOW-HIER):
 *
 *   kClearHier   counters + both frontier counts → 0
 *   kSeedRoots   one thread per instance: world-sphere frustum (mesh sphere ×
 *                instance transform + swayPad) + the per-mesh DRAW ENVELOPE
 *                (lodDist; the hier far bound) + per-instance min-size → append
 *                the mesh's ROOTS to frontier A (skips rootCount==0 meshes)
 *   kTraverse ×N (ping-pong A↔B, N = hierdepth passes): per frontier item project
 *                ownError; CUT (project ≤ τ) ⇒ emit (frustum + cone + min-size +
 *                clipmap hollow + prev-HZB occlusion) to qRaster, else ⇒ enqueue
 *                its DAG children to the other frontier (kArgs sizes each pass)
 *   kRasterArgs  clamp → qRaster[0] = (count, 0), 2D-split args for the raster
 *   kRasterArgs2 (syncFullArgs) full-range payload-pass args (single-phase: the
 *                BFS already produced the whole cut, so "phase 2" is just this)
 *
 * Queue law (F14): explicit caps, clamp on overflow (HUD flag via readCounts),
 * counters-in-queue slot 0, 2D-split indirect dispatch. Crack-free: the cut is a
 * watertight DAG frontier (bit-exact sibling error spheres).
 */

import { IndirectStorageBufferAttribute, StorageBufferAttribute } from 'three/webgpu';
import type { Renderer } from 'three/webgpu';
import {
  Fn,
  If,
  Loop,
  abs,
  atomicAdd,
  atomicMax,
  atomicMin,
  atomicStore,
  ceil,
  cross,
  dot,
  exp2,
  float,
  instanceIndex,
  log2,
  normalize,
  uint,
  vec3,
  vec4,
} from 'three/tsl';
import { sunU } from '../render/VegMaterials';
import type { NB, NF, NU, NV3, NV4 } from '../gpu/TSLTypes';
import {
  CLUSTER_WORDS,
  LOD_NONE,
  MESH_FLAG_FARTILE,
  MESH_FLAG_HEIGHTFIELD,
  MESH_WORDS,
  TRANSFORM_CHANNEL,
  readCluster,
  readDag,
  readMesh,
} from './GeometryRegistry';
import type { RegistryGpu } from './GeometryRegistry';
import {
  CONE_SLACK,
  DISPATCH_ROW,
  QCHUNK_CAP,
  QRASTER_CAP,
  QVOX_CAP,
  instRotateDir,
  instSphereRadius,
  instTransformPoint,
  instYaw,
  noteQueueHwRenderer,
  queueCapParam,
  registerQueueHw,
  type NaniteCam,
} from './NaniteCommon';
import {
  aLoadU,
  bcF2U,
  bcU2F,
  dispatch,
  dispatchBatchMixed,
  dispatchIndirect,
  elemU,
  elemUW,
  localX,
  loopU,
  maxU,
  minU,
  setIndirectDispatch,
  readBuffer,
  returnIf,
  sU32Views,
  sUvec2,
  toI,
  uniformF,
  uv2,
  wgLinear,
  type UniformF,
  type UniformMat4,
  type UniformV3,
} from './Tsl';
import type { BufOf, UV2 } from './Tsl';

/** occlusion-reject list caps (F14). Overflow is graceful: victims miss
 *  phase 2 THIS frame only (next frame's phase 1 re-tests everything) —
 *  measured at ?stress=5 bm3: 3.8M inst rejects, flag fired, image intact. */
const REJ_INST_CAP = 1_048_576;
const REJ_CLUST_CAP = 1_048_576;

/** queue high-water diag: default label sequence for unlabeled chains */
let chainSeq = 0;

/** D-N43 Stage 0.5 SIM — distance-banded τ. The cross-instance-AGGREGATION win is
 *  *simulated* (before its builder exists) by coarsening the LOD cut with camera
 *  distance: τ_eff = τ·(1 + d/bandD), so the near field stays at full detail while
 *  far clusters collapse toward their per-mesh ROOT — the ≥1-cluster/instance floor
 *  the real Stage-1 merge must break. bandD ≤ 0 ⇒ DISABLED (τ_eff = τ, the shipped
 *  path is bit-identical). HOLE-FREE: the DAG cut stays watertight at any single
 *  distance; only the LOD-transition distance shifts outward, so the rastered
 *  geometry + tris/px stay REAL (unlike a drop-based region budget, which would
 *  under-count). Minor LOD-transition cracks from the spatial τ gradient are
 *  sub-pixel and acceptable for a perf BOUND — this knob never ships. */
function lodWarp(tau: NF, dist: NF, scale: NF, near: NF, pow: NF): NF {
  // τ_eff = τ · (1 + (max(0, dist − near) / scale)^pow).
  //   near  = full-detail plateau radius (≤ near m ⇒ τ_eff = τ, level 0)
  //   scale = distance PAST `near` at which τ doubles (smaller ⇒ coarsens sooner)
  //   pow   < 1 ⇒ detail drops FAST just past `near`, SLOW far (the "concentrate
  //             detail near the player" curve); = 1 ⇒ linear; > 1 ⇒ slow-near-fast-far
  // scale ≤ 0 ⇒ disabled (τ_eff = τ). CAVEAT: τ_eff varies with each cluster's own
  // distance, so neighbouring DAG levels can use slightly different τ — the cut is
  // watertight at a single point but the transition SHELL wiggles, which can open
  // sub-cluster LOD-seam cracks. Worst right at the plateau edge with pow<1 (the
  // gradient is vertical there). Inside the plateau (dist ≤ near) τ_eff = τ for all
  // clusters ⇒ exact. Foliage hides this well; soften with larger near/scale.
  const d = dist.sub(near).max(float(0)) as unknown as NF;
  const grow = float(1).add(d.div(scale.max(float(1e-3))).pow(pow)) as unknown as NF;
  return scale.greaterThan(0).select(tau.mul(grow), tau) as unknown as NF;
}

interface ComputeKernel {
  setName(name: string): unknown;
}

export type SphereOccludedFn = (
  center: NV3,
  radius: NF,
  vp: UniformMat4,
  camPos: UniformV3,
) => NB;

export interface NaniteCullCounts {
  /** chunk items pushed in phase 1 (pre-clamp) */
  chunks: number;
  /** raster items = visible clusters across both phases (pre-clamp) */
  visClusters: number;
  /** phase-2 inputs: occlusion-rejected instances / clusters from phase 1 */
  rejInst: number;
  rejClust: number;
  /** clusters phase 2 brought back (appended past the phase-1 base) */
  p2Appends: number;
  /** N8-D1: emitted clusters carrying a DAG record (the rock screen-error cut) */
  dagClusters: number;
  /** D-N43 Stage 0.5: exact per-frame SW rastered triangles (Σ cluster triCount) */
  visTris: number;
  /** D-N43 Stage 0.5: DAG-only rastered triangles (isolates leaf/aggregate tris) */
  dagTris: number;
  /** non-null when a queue clamped this frame */
  overflow: string | null;
}

export interface NaniteCullChain {
  /** raster work queue: [0] = (count, phase2Base), items at 1.. */
  qRasterRO: BufOf<UV2>;
  qRasterAttr: StorageBufferAttribute;
  /** phase-1 items (one wg per item) */
  rasterDispatchAttr: IndirectStorageBufferAttribute;
  /** phase-2 appended items only */
  rasterDispatch2Attr: IndirectStorageBufferAttribute;
  /** all items (payload passes) */
  rasterDispatchFullAttr: IndirectStorageBufferAttribute;
  /** voxel-foliage (spec §6.2): the fanned voxel-cluster work queue [0]=(count,0),
   *  items at 1.. (same (instId, ci) uvec2 as qRaster). The voxel raster reads it. */
  qVoxRasterRO: BufOf<UV2>;
  qVoxRasterAttr: StorageBufferAttribute;
  /** 2D-split dispatch args over the fanned voxel-cluster count (the voxel raster). */
  voxRasterDispatchAttr: IndirectStorageBufferAttribute;
  /** DEPTH-BUCKET F2B (?voxf2b): per-bucket (base,count) the K near→far scatter
   *  dispatches read (STORAGE, not a CPU uniform — all K share ONE submit). */
  voxBucketRangeRO: BufOf<UV2>;
  /** DEPTH-BUCKET F2B: per-bucket 2D-split indirect dispatch args (length = voxF2bK),
   *  ordered NEAR→FAR (index 0 = nearest). The voxel raster issues one dispatch each. */
  voxBucketDispatchAttr: IndirectStorageBufferAttribute[];
  /** DEPTH-BUCKET F2B: the BUILD-TIME bucket count K (the voxel raster bakes the SAME K
   *  kernel instances). 1 when ?voxf2b=0 effectively (the K dispatches are still built
   *  but the cull only fills one contiguous list — see runVoxFanout). */
  voxF2bK: number;
  /** DEPTH-BUCKET F2B: whether front-to-back ordering is ON (?voxf2b, DEFAULT TRUE —
   *  loss-exact + feeds the per-block occlusion cull the near depths it needs to skip
   *  whole far blocks; ?voxf2b=0 disables for the A/B). When false the cull runs today's
   *  single unordered append + the raster issues ONE dispatch over voxRasterDispatchAttr. */
  voxF2bEnabled: boolean;
  /** ?voxprev two-pass visibility partition active (implies voxF2bEnabled, K=2;
   *  spec-prev-frame-occlusion §2.4). Gates the voxB0/voxB1 meter counters. */
  voxPrevEnabled: boolean;
  /** phase 1: clear → instance cull → cluster cull → raster args */
  runPhase1(renderer: Renderer): void;
  /** CAMERA||SHADOW OVERLAP (item 4): the EXACT ordered kernel list runPhase1 would
   *  batch-submit (one element per dispatch), so a caller can CONCATENATE two culls'
   *  batches into ONE submit and let Dawn overlap them (they write DISJOINT buffers).
   *  The internal order/RAW correctness is identical to runPhase1; concatenation is
   *  safe ONLY between culls that share no writable buffer (verified: the camera cull
   *  and the shadow shared-cut cull each own fresh counters/qRaster/qFrontier). */
  phase1Batch(): readonly unknown[];
  /** write full-range args WITHOUT re-testing (?phase2=0 A/B + no-occl path) */
  syncFullArgs(renderer: Renderer): void;
  /** SUBMIT-COALESCE (?coalesce=1): the syncFullArgs kernel as a batchable list
   *  ([kRasterArgs2]) so the frame folds it into the ONE cull submit. RAW-safe:
   *  it reads qRaster[0]/counters written by phase1Batch's kRasterArgs, and in-pass
   *  dispatch order == queue order (Tsl dispatchBatchMixed). */
  fullArgsBatch(): readonly unknown[];
  /** SUBMIT-COALESCE (?coalesce=1): the exact ordered kernel list runVoxFanout would
   *  submit (F2B batch, or [kVoxFanoutArgs, kVoxFanout, kVoxRasterArgs] when voxf2b=0)
   *  for folding into the same cull submit. Order is load-bearing (RAW chain). */
  voxFanoutBatch(): readonly unknown[];
  /** voxel-foliage (spec §4.6): post-traverse fan-out — qRaster → qVoxRaster by
   *  matClass. Call AFTER runPhase1. Publishes voxRasterDispatchAttr for Stage-2. */
  runVoxFanout(renderer: Renderer): void;
  /** voxel-foliage: readback of the fanned voxel-cluster count (HUD/overflow). */
  readVoxCount(renderer: Renderer): Promise<number>;
  /** F2B/voxprev: per-bucket cluster counts (voxBucketRange[b].count, length = voxF2bK).
   *  Under ?voxprev: [0] = probably-visible (pass A), [1] = probably-occluded (pass B). */
  readVoxBuckets(renderer: Renderer): Promise<number[]>;
  readCounts(renderer: Renderer): Promise<NaniteCullCounts>;
}

export function buildNaniteCull(
  gpu: RegistryGpu,
  instanceCount: number,
  cam: NaniteCam,
  /** prev/fresh-HZB occlusion test (NaniteHzb.sphereOccluded); null = ?occl=0 */
  sphereOccluded: SphereOccludedFn | null,
  /** N5: shadow casters disable cone backface — the cone axis is camera-
   *  relative (instRotateDir vs cam.camPos), but a cluster facing away from the
   *  CAMERA still casts toward the LIGHT, so cone-culling it punches shadow
   *  holes. Default true (camera path). */
  opts?: {
    coneCull?: boolean;
    /** ?crownlod0 (user mandate 2026-07-04): leaf crowns are FULL DETAIL (LOD0)
     *  or VOXEL, never a simplified aggregate level. Camera path only — shadow /
     *  secondary culls omit it so caster leaf LOD stays coarse. */
    crownLod0?: boolean;
    /** shadow-only voxel-τ COARSEN multiplier (>1 ⇒ voxel clusters cut at a coarser
     *  DAG level = fewer/bigger bricks; the "less detailed crown in the shadow"). Voxel
     *  matClass only — trunks keep their τ. When set it REPLACES voxTauCap's fine cap. */
    voxCoarsen?: number;
    tau?: UniformF;
    minPx?: UniformF;
    innerReject?: UniformF;
    /** lodWarp: distance-band scale (m) — τ doubles `scale` m past lodNear; 0 = off */
    simBandD?: UniformF;
    /** lodWarp: full-detail plateau radius (m) — ≤ this, level 0 */
    lodNear?: UniformF;
    /** lodWarp: falloff exponent — <1 fast-near/slow-far, 1 linear, >1 slow-near */
    lodPow?: UniformF;
    /** N9-IMP: per-INSTANCE min screen SIZE (px, diameter). Below it the whole
     *  instance is too small for geometry — dropped at kInstCull so the imposter
     *  far-field owns it (UE5 model). 0 = off. */
    instMinPx?: UniformF;
    /** frontier-buffer capacity (items). Default = ?qfrontier (falls back to
     *  QRASTER_CAP). SHADOW culls run many chains with small cuts and pass a far
     *  smaller cap so N×2 frontier buffers don't waste GBs. */
    frontierCap?: number;
    /** qRaster capacity (items) for THIS chain. Default = the global QRASTER_CAP
     *  (?qrcap). The shadow shared-cut chain passes its own smaller cap — its cut
     *  is ring-bounded, far below the camera's far-field demand. Every append is
     *  slot<cap guarded; overflow = dropped clusters this frame + readCounts flag. */
    qRasterCap?: number;
    /** qVoxRaster capacity (items) for THIS chain. Default = the global QVOX_CAP
     *  (?qvcap). Chains that never run the vox fan-out (the shadow cuts — only the
     *  camera chain calls runVoxFanout/voxFanoutBatch) pass a tiny cap so the
     *  buffer isn't 16.8 MB of dead weight. */
    qVoxCap?: number;
    /** queue high-water diag label (window.__qHW); default cull<seq> */
    label?: string;
    /** item 6: the BFS pass count (≥ the deepest DAG anchor-chain or leaves never emit =
     *  holes). Callers pass registry.maxDagDepth (+ a small margin). ?hierdepth overrides.
     *  Omitted ⇒ the legacy constant. Each pass is paid TWICE/frame (camera + shadow cut). */
    hierDepth?: number;
    /** ?voxprev two-pass partition (spec-prev-frame-occlusion §2.4): the LIBERAL
     *  prev-frame visibility classifier (NaniteHzb.sphereProbablyOccluded). MUST NOT be
     *  the conservative emit-test closure — every qRaster entry already passed that exact
     *  test this frame, so partitioning by it yields a provably-empty bucket 1. Verdicts
     *  only ROUTE work (never drop); null (?occl=0 / shadow culls) ⇒ voxprev inert. */
    voxPrevTest?: SphereOccludedFn | null;
    /** P10 (shadow arc): RING-SNAPPED LOD distance for the SHADOW cut. The camera
     *  path's continuous camera-distance LOD makes toroidal strip content age (a
     *  region's stored casters keep the LOD from when it was written; the target
     *  drifts with every camera step ⇒ shadows visibly appear/att fade with motion
     *  direction). With this set (= the clipmap base E_0 in metres), the LOD
     *  distance becomes the CHEBYSHEV light-plane distance snapped UP to the ring
     *  ladder {E_0·2^k} — constant within a ring, changing EXACTLY when a region's
     *  owning level changes, i.e. when the hollow-reveal/outer strips rewrite it
     *  anyway. LOD age is impossible by construction, and caster detail matches
     *  the owning level's texel density. */
    lodRingSnap?: number;
  },
): NaniteCullChain {
  // N8-HIC: the cull is HIERARCHICAL — seed each mesh's roots + BFS-descend the DAG.
  // (The legacy brute-force two-phase path was deleted once the world + shadow culls
  // all moved to hier — PERF-VB3 / SHADOW-HIER.)
  // Per-chain queue caps (memory sizing 2026-07-04): qCap sizes THIS chain's qRaster
  // (+ every clamp/guard over it); vCap its vox fan-out queue. Defaults are the global
  // URL-overridable caps; the shadow shared-cut passes its own (NaniteClipCull). The
  // ceilings are the STRUCTURAL limits (payload bit budget / resolve mask / 128 MB
  // binding), NOT the global defaults — an explicit opts cap may exceed the world
  // default (e.g. the shadow cut's 262k vs the camera's 131k).
  const qCap = Math.min(8_388_608, Math.max(1_024, Math.round(opts?.qRasterCap ?? QRASTER_CAP)));
  const vCap = Math.min(2_097_152, Math.max(64, Math.round(opts?.qVoxCap ?? QVOX_CAP)));
  const frontierCap = Math.min(
    8_388_608,
    Math.max(1, Math.round(opts?.frontierCap ?? queueCapParam('qfrontier', QRASTER_CAP, 4_096, 8_388_608))),
  );
  // hier BFS pass count — a CPU-fixed loop (the GPU frontier count isn't visible to
  // the CPU mid-frame). Must be ≥ the deepest DAG anchor-chain or the tail clusters
  // never emit (holes). PRIORITY: ?hierdepth=N override > opts.hierDepth (the MEASURED
  // registry.maxDagDepth, item 6 — exact, no holes, sheds the empty-tail passes the old
  // constant 18 paid twice/frame) > the legacy constant 18 (safe fallback when no measured
  // depth is threaded). Each pass is paid TWICE/frame (camera + shadow shared-cut).
  const hierDepthOverride = new URLSearchParams(window.location.search).get('hierdepth');
  const hierDepth = Math.max(
    1,
    hierDepthOverride != null && Number.isFinite(Number(hierDepthOverride))
      ? Math.round(Number(hierDepthOverride))
      : opts?.hierDepth != null && Number.isFinite(opts.hierDepth)
        ? Math.round(opts.hierDepth)
        : 18,
  );
  // ── DEPTH-BUCKET FRONT-TO-BACK voxel scatter (?voxf2b) ───────────────────────
  // Order the fanned voxel clusters NEAR→FAR so the per-pixel early-Z gate in
  // kVoxScatter (NaniteVoxelRaster: If(candL>prevE) … elect) actually FIRES: the
  // nearest opaque brick writes first, farther bricks at that pixel then read the
  // now-near depth and SKIP their atomicMax+atomicStore (UE's "depth buckets,
  // rasterized front-to-back"). The fanout quantizes each cluster's LINEAR VIEW
  // DEPTH (cam.vp·centre .w — NOT NDC z, which perspective-collapses the far field)
  // into K contiguous buckets sliced across the live voxel depth range [dMin,dMax];
  // the voxel raster then issues K near→far dispatches over those slices in ONE
  // submit. LOSS-EXACT (atomicMax is monotone+commutative ⇒ reordering cannot
  // change the winner; ordering only skips redundant stores). ?voxf2b=0 reverts
  // EXACTLY to today's single-atomicAdd unordered append + single dispatch.
  //
  // DEFAULT ON (?voxf2b=0 disables for the A/B). The reorder is REAL and loss-exact
  // (image byte-identical, voxBrickWrites −38..−45 % monotone in K). The FRAME WIN it
  // exists to UNLOCK is the per-BLOCK FOOTPRINT-OCCLUSION early-out that ALREADY lives
  // in the scatter body (?voxoccl, default ON, NaniteVoxelRaster ~342-374): thread 0
  // reads the GLOBAL visPayloadV at the block-centre pixel and, if the block's NEAREST-
  // possible key can't beat the winner there, skips the WHOLE block — eliding its entire
  // Phase-A project + Phase-B per-pixel footprint loop (not just the minority losing
  // atomicMax+store). That cull only BITES across voxels when the near voxel depths are
  // ALREADY in visPayloadV by the time a far block runs — which is EXACTLY what F2B
  // ordering guarantees: the K near→far bucket dispatches ride ONE submit and the in-pass
  // UAV barriers serialize bucket b's writes before bucket b+1's occlusion-cull reads.
  // So near voxel blocks pre-seed the far blocks' whole-block skip — the iter-1/2 store-
  // saving PLUS the per-block VISIT-saving the occlusion cull now harvests. The kVox-
  // Scatter ELECTION body is UNCHANGED — F2B only (a) reorders the input into K linear-
  // view-depth slabs and (b) issues K near→far dispatches; the occlusion cull is the
  // pre-existing separate gate F2B feeds, not an election edit.
  const voxParams = new URLSearchParams(window.location.search);
  // DEFAULT FLIPPED TO OFF (2026-06-26, measured). The K near→far bucket dispatches ride ONE
  // submit with an in-pass UAV barrier BETWEEN each bucket — K serialized drains. The intended
  // payoff (near blocks pre-seed far blocks' WHOLE-BLOCK occlusion skip) is NOT realized: the
  // per-block cull reads voxOccPyr, a pyramid built ONCE in dispatchVoxel BEFORE any bucket and
  // never rebuilt between buckets, so bucket b+1's cull sees the STALE pre-scatter pyramid (trunk
  // triangles only under forcevox=all), NOT bucket b's fresh voxel depth. So F2B pays K barriers
  // for ~zero block-cull benefit + only the marginal per-pixel write-drop the unordered gate
  // already gives. MEASURED single voxelised tree, worst pose (camera below/beside crown looking
  // up): gpuWall 25.4 ms (K16 default) → 15.8 (K4) → 10.3 (K1) → 10.8 (unordered, F2B off); LIVE
  // 25 ms/42 fps → 8.3 ms/121 fps. The cost scales ~linearly with K = the bucket-barrier
  // serialization floor (≈2 clusters/bucket ⇒ catastrophic GPU occupancy), the SAME K-pass floor
  // the depth-bucketed BIN path was REFUTED + REMOVED for (file header §6.0). LOSS-EXACT: atomicMax
  // is order-independent, so the unordered path is image-identical (verified). ?voxf2b=1 restores
  // the old K-bucket path (the A/B control / opt-in for any future large-batch retune).
  // ?voxprev=1 — TWO-PASS vox scatter (spec-prev-frame-occlusion §2.4): partition
  // qVoxRaster by the LIBERAL prev-frame visibility classifier (bucket 0 = probably-
  // visible last frame, bucket 1 = probably-occluded), scattered as 2 waves with a
  // voxOccPyr rebuild between (the raster forces ?voxwaves=2), so pass B's per-block +
  // per-brick culls fire against REAL same-frame vox occluders (pass A's elections).
  // Rides the F2B plumbing at K=2 (bucket = visibility bit instead of depth slab).
  // Gated on opts.voxPrevTest, NOT the emit-test sphereOccluded — the emit test already
  // filtered qRaster with that exact closure, so partitioning by it would yield an
  // exactly-empty bucket 1. Under ?occl=0 voxPrevTest is null ⇒ fully INERT. Precedence:
  // voxprev=1 forces voxf2b on + K=2 (+ waves=2 in the raster); explicit ?voxf2b/
  // ?voxf2bk/?voxwaves values are ignored while it is set. NOTHING is ever dropped by
  // the prev-frame verdict — it only picks the pass; both passes end at the same
  // exact-conservative same-frame culls (quality-law argument: spec §3).
  // DEFAULT ON (2026-07-02 gates: oblique −4.6/−3.0 two sessions, eye −1.6, aerial +0.4;
  // live p50 15.8→10.1 p95 =; shots at D0 band; B1 share eye 71.7%/obl 23.6%/aer 9.7%).
  // ?voxprev=0 reverts to the single-pass unordered scatter (the permanent A/B control).
  const voxPrevTest = opts?.voxPrevTest ?? null;
  const voxPrev = voxParams.get('voxprev') !== '0' && voxPrevTest !== null;
  const voxf2b = voxPrev || (voxParams.get('voxf2b') ?? '0') !== '0';
  // ?voxtaucap — voxel-cluster τ_eff clamp (px); see the traverse cut note. DEFAULT 12
  // (2026-07-02c): pre-?fartiles this cap was ruinous — it fought the lodWarp across
  // THOUSANDS of per-tree crowns (~8× bricks per forced level; aerial 36→81 ms at 8px).
  // With ?fartiles the far field is ~1 head per 64 m TILE, so descending a level or two
  // costs a bounded few hundred bricks per visible tile — and WITHOUT the cap the warp
  // selects 12-24 m tile bricks from ~200 m whose occupancy masks are FULL (dense canopy
  // union), so ?voxcell paints them as solid tree-sized cubes (the user-reported "cube
  // landscape"). 12 px caps the emitted brick size so far tiles stay at 3-6 m bricks whose
  // carved cells read as foliage. ?voxcell still handles the silhouette; 0 disables.
  // 12→4 (2026-07-02 beautification): 12 px let the warp emit ~15-24 px bricks from ~40 m
  // out — the user-reported mid-field "giant voxels". 4 px forces ~1-2 finer pyramid
  // levels (bricks ~5-6 px on retina); perf cost user-accepted for the beauty arc.
  const voxTauCapRaw = Number(voxParams.get('voxtaucap') ?? '4');
  const voxTauCap = Number.isFinite(voxTauCapRaw) && voxTauCapRaw >= 0 ? voxTauCapRaw : 4;
  // K is a BUILD-TIME constant: it bakes K bucket counters + K indirect attrs and
  // (in the voxel raster) K kernel instances. ?voxf2bk default 16 (iter-2 NET-BEST: the
  // canopy write-drop SATURATES at K16 over the tight linear-view-depth [dMin,dMax]
  // range, so K16 captures the full separation while paying the fewest in-pass barriers;
  // K24 keeps separating only in the tall canopyFar column at +1 barrier-cost each. With
  // F2B feeding the per-block occlusion cull, finer K also means a far block's near
  // occluders are published in an earlier bucket ⇒ more whole-block skips. Try 8/12/24
  // via ?voxf2bk). Clamped [1,32]. buildNaniteVoxelRaster reads the SAME K.
  const voxF2bKraw = parseInt(voxParams.get('voxf2bk') ?? '16', 10);
  const VOX_F2B_K = voxPrev
    ? 2 // visibility partition: bucket 0 = probably-visible, bucket 1 = probably-occluded
    : Math.min(32, Math.max(1, Number.isFinite(voxF2bKraw) ? voxF2bKraw : 16));
  const coneCull = opts?.coneCull !== false;
  const crownLod0 = opts?.crownLod0 === true; // ?crownlod0 — leaf LOD0-or-descend (camera only)
  // shadow-only voxel COARSEN (>1): emit voxel clusters at a coarser DAG level for the
  // shvox2 caster (the "less detailed crown in the shadow"). ≤1 ⇒ inactive.
  const voxCoarsenRaw = opts?.voxCoarsen ?? 0;
  const voxCoarsen = Number.isFinite(voxCoarsenRaw) && voxCoarsenRaw > 1 ? voxCoarsenRaw : 0;
  // S3 SHADOW CLIPMAP hollow (D-N29): a clipmap level rasters only the RING
  // outside the next-finer level — a cluster whose light-space clip bbox lies
  // ENTIRELY within [±0.5] (the finer level's box, since extents double) is
  // covered by that finer level and is dropped here. The uniform carries 1/E
  // (the inverse light-XY half-extent of THIS level) so radius→clip is r·(1/E);
  // 0 = disabled (the camera path + the finest level, which has no finer level).
  // Shadows project along the sun ⇒ caster and its shadow share light-XY, so the
  // finer level that covers the caster also covers everything it shadows — the
  // hollow is gap-free. The radius margin keeps boundary-straddling clusters in
  // BOTH levels (no seam). Cheaper than per-cascade redundancy: each caster
  // rasters into exactly one level.
  const innerInvHalf = opts?.innerReject ?? uniformF(0);
  // N8-D1e min-screen-size cull threshold (projected sphere RADIUS in px). 0 =
  // disabled (exact pre-D1e behaviour). When > 0: any cluster projecting smaller
  // is culled, and DAG'd meshes drop their finite hybrid envelope for this bound.
  const minPx = opts?.minPx ?? uniformF(0);
  // N8-D1 continuous-LOD cut threshold (screen-error px). The camera path wires
  // ?loderr into this uniform; shadow cascades take the default. projK =
  // (screenH/2)·cot(fovY/2) — for the ortho shadow cams (cotHalfFov stays 1,
  // uH = map size) it lands ~the camera's, so casters track the lit-surface LOD
  // (proper DAG-decoupled caster LOD is S4). Mirrors probe-dag.project exactly.
  const tau = opts?.tau ?? uniformF(1);
  // D-N43 Stage 0.5 SIM: distance-band scale for the region-collapse sim (0 = off,
  // the default ⇒ τ_eff = τ). Only the camera path wires it (NaniteFrame); the
  // shadow culls leave it 0, so casters are untouched (separation honoured).
  const simBandD = opts?.simBandD ?? uniformF(0);
  // lodWarp shape: full-detail plateau radius + falloff exponent (1 = linear)
  const lodNear = opts?.lodNear ?? uniformF(0);
  const lodPow = opts?.lodPow ?? uniformF(1);
  // N9-IMP per-instance min screen-SIZE (px diameter); below it the instance is
  // dropped whole (imposter far-field territory). 0 = off (the camera default).
  const instMinPx = opts?.instMinPx ?? uniformF(0);
  const projK = cam.cotHalfFov.mul(cam.uH).mul(0.5) as unknown as NF;
  // ---- buffers ---------------------------------------------------------------
  // counters: [1] raster pushes (emitted clusters), [5] DAG clusters, [6] visTris,
  // [7] DAG tris (HUD/SIM). The hier BFS reuses [0]/[4] as the two frontier counts.
  const countersAttr = new StorageBufferAttribute(new Uint32Array(8), 1);
  const counters = sU32Views(countersAttr, 8).atomic;

  const qRasterAttr = new StorageBufferAttribute(new Uint32Array((qCap + 1) * 2), 2);
  const qRasterV = sUvec2(qRasterAttr, qCap + 1);

  // voxel-foliage (spec §4.6 / §6.2): the VOXEL raster work queue. The cut emits voxel
  // clusters into the SAME qRaster as triangles (kTraverse is pinned ≤10 buffers — no
  // room for a 2nd queue there, §4.6); a TINY post-traverse FAN-OUT pass (kVoxFanout
  // below) then re-scans the emitted qRaster, tests each cluster's mesh matClass, and
  // fans the voxel(7) entries here. qVoxRaster[0] = (count, 0); items at 1.. are the
  // SAME (instId, ci) uvec2 as qRaster (the voxel raster reads bricks via ci's word6/7).
  const qVoxRasterAttr = new StorageBufferAttribute(new Uint32Array((vCap + 1) * 2), 2);
  const qVoxRasterV = sUvec2(qVoxRasterAttr, vCap + 1);
  // voxel fan-out cursor (its OWN atomic counter so it doesn't contend the BFS counters).
  const voxCountAttr = new StorageBufferAttribute(new Uint32Array(1), 1);
  const voxCount = sU32Views(voxCountAttr, 1).atomic;
  // 2D-split dispatch args for the voxel BIN/raster (over qVoxRaster count, Stage 2).
  const voxRasterDispatchAttr = new IndirectStorageBufferAttribute(new Uint32Array(3), 3);
  const voxRasterDispatch = sU32Views(voxRasterDispatchAttr as unknown as StorageBufferAttribute, 3).rw;
  // ONE-THREAD-PER-qRaster-ENTRY fan-out dispatch args (ceil(qRaster/64) workgroups).
  const voxFanoutDispatchAttr = new IndirectStorageBufferAttribute(new Uint32Array(3), 3);
  const voxFanoutDispatch = sU32Views(voxFanoutDispatchAttr as unknown as StorageBufferAttribute, 3).rw;

  // ── DEPTH-BUCKET F2B build-time buffers (?voxf2b) ────────────────────────────
  // Per-bucket atomic count (kVoxCount), exclusive-prefix base (kVoxPrefix), the
  // published (base,count) pair per bucket the K scatter kernels read (STORAGE, not
  // a CPU uniform — all K dispatches share ONE submit so a single uniform would
  // collapse to the last value written), and a per-bucket scatter cursor.
  const K = VOX_F2B_K;
  const voxBucketCountAttr = new StorageBufferAttribute(new Uint32Array(K), 1);
  const voxBucketCount = sU32Views(voxBucketCountAttr, K).atomic;
  const voxBucketRangeAttr = new StorageBufferAttribute(new Uint32Array(2 * K), 2);
  const voxBucketRangeV = sUvec2(voxBucketRangeAttr, K); // [b] = (base_b, count_b)
  const voxCursorAttr = new StorageBufferAttribute(new Uint32Array(K), 1);
  const voxCursor = sU32Views(voxCursorAttr, K).atomic;
  // 2-word atomic depth RANGE (kVoxRange): [0]=min(d) bits, [1]=max(d) bits. View
  // depth d>0 ⇒ raw IEEE-754 bits are monotone as uint, so bcF2U feeds atomicMin/Max
  // directly. Seeded +inf/0 so the first cluster sets a real range; self-tightening,
  // exactly brackets the live voxel depths (the anti-NDC fix the prior attempt lacked).
  const voxRangeAttr = new StorageBufferAttribute(new Uint32Array(2), 1);
  const voxRange = sU32Views(voxRangeAttr, 2).atomic;
  // K per-bucket 2D-split INDIRECT dispatch args (one bucket = one near→far dispatch).
  const voxBucketDispatchAttr: IndirectStorageBufferAttribute[] = [];
  const voxBucketDispatch: ReturnType<typeof sU32Views>['rw'][] = [];
  for (let b = 0; b < K; b++) {
    const a = new IndirectStorageBufferAttribute(new Uint32Array(3), 3);
    voxBucketDispatchAttr.push(a);
    voxBucketDispatch.push(sU32Views(a as unknown as StorageBufferAttribute, 3).rw);
  }

  const rasterDispatchAttr = new IndirectStorageBufferAttribute(new Uint32Array(3), 3);
  const rasterDispatch = sU32Views(rasterDispatchAttr as unknown as StorageBufferAttribute, 3).rw;
  const rasterDispatch2Attr = new IndirectStorageBufferAttribute(new Uint32Array(3), 3);
  const rasterDispatch2 = sU32Views(rasterDispatch2Attr as unknown as StorageBufferAttribute, 3).rw;
  const rasterDispatchFullAttr = new IndirectStorageBufferAttribute(new Uint32Array(3), 3);
  const rasterDispatchFull = sU32Views(
    rasterDispatchFullAttr as unknown as StorageBufferAttribute,
    3,
  ).rw;

  const split2D = (
    args: ReturnType<typeof sU32Views>['rw'],
    n: NU,
  ): void => {
    const rows = n.add(uint(DISPATCH_ROW - 1)).div(uint(DISPATCH_ROW));
    elemUW(args, 0).assign(minU(n, uint(DISPATCH_ROW)));
    elemUW(args, 1).assign(maxU(rows, uint(1)));
    elemUW(args, 2).assign(uint(1));
  };

  // ---- frustum test helper (shared TSL) ------------------------------------------
  const frustumVisible = (center: NV3, radius: NF): NF => {
    const visible = float(1).toVar();
    Loop(6, ({ i: pi }) => {
      const plane = cam.planes.element(pi);
      const d = dot(plane.xyz, center).add(plane.w) as unknown as NF;
      If(d.lessThan(radius.negate()), () => {
        visible.assign(0);
      });
    });
    return visible as unknown as NF;
  };

  /** instance world sphere (heightfield = world-space already) */
  const instWorldSphere = (
    A: NV4,
    B: NV4,
    isHF: NB,
    localSphere: NV4,
    swayPad: NF,
  ): { center: NV3; radius: NF } => {
    const yawSc = instYaw(B);
    const centerW = vec3(0).toVar();
    const radiusW = float(0).toVar();
    If(isHF, () => {
      centerW.assign(localSphere.xyz);
      radiusW.assign(localSphere.w);
    }).Else(() => {
      centerW.assign(instTransformPoint(A, B, yawSc, localSphere.xyz as unknown as NV3));
      radiusW.assign(instSphereRadius(A, B, localSphere.w, swayPad));
    });
    return { center: centerW as unknown as NV3, radius: radiusW as unknown as NF };
  };

  // ---- queue high-water diag (memory sizing 2026-07-04) ---------------------------
  // Persistent (never frame-cleared) atomicMax'd RAW cursor maxima: [0] qRaster
  // emits, [1] BFS frontier in-count (max over passes), [2] vox fan-out count.
  // Raw = pre-clamp, so the value is the true demand even when a cap clamps.
  // Read/reset via window.__qHW (registerQueueHw below); cost = one atomicMax in
  // the tiny 1-thread args kernels, nothing in the hot traverse/emit kernels.
  const hwAttr = new StorageBufferAttribute(new Uint32Array(4), 1);
  const hwV = sU32Views(hwAttr, 4).atomic;
  const kHwReset = Fn(() => {
    If(instanceIndex.lessThan(uint(4)), () => {
      atomicStore(hwV.element(instanceIndex), uint(0));
    });
  })().compute(4, [4]);
  (kHwReset as unknown as ComputeKernel).setName('nanQHwReset');

  // ---- kRasterArgs (phase 1) -------------------------------------------------------
  const kRasterArgs = Fn(() => {
    atomicMax(hwV.element(0), aLoadU(counters.element(1)));
    const n = minU(aLoadU(counters.element(1)), uint(qCap));
    qRasterV.rw.element(0).assign(uv2(n, 0));
    split2D(rasterDispatch, n);
  })().compute(1, [1]);
  (kRasterArgs as unknown as ComputeKernel).setName('nanRasterArgs');

  // kRasterArgs2: total count + appended-range and full-range dispatch args.
  // The base read goes through the SAME rw view that writes slot 0 — mixing
  // the ro view into this dispatch is a same-scope usage violation (N0 law).
  const kRasterArgs2 = Fn(() => {
    const nT = minU(aLoadU(counters.element(1)), uint(qCap)).toVar();
    const base = qRasterV.rw.element(0).y.toVar();
    qRasterV.rw.element(0).assign(uv2(nT, base));
    split2D(rasterDispatch2, nT.sub(base));
    split2D(rasterDispatchFull, nT);
  })().compute(1, [1]);
  (kRasterArgs2 as unknown as ComputeKernel).setName('nanRasterArgs2');

  // ──────────────────────────────────────────────────────────────────────────
  // voxel-foliage (spec §4.6 / §6.2): the POST-TRAVERSE FAN-OUT. The cut emits voxel
  // clusters into the SAME qRaster as triangles (no room for a 2nd queue inside the
  // ≤10-buffer kTraverse, §4.6). This tiny pass re-scans the emitted qRaster ONCE,
  // tests each cluster's mesh matClass, and fans the voxel(7) entries into qVoxRaster.
  // The voxel clusters STAY in qRaster too — the triangle world1 raster SKIPS them by
  // the same matClass test (NaniteRaster), so they don't rasterize as garbage tris.
  // Cost = one re-scan of up to qRaster-count entries (a Stage-0a/3 perf line item).
  // ──────────────────────────────────────────────────────────────────────────
  const VOXEL_MATCLASS = 7; // MATERIAL_CLASS.voxel (spec §4.1)
  const LEAF_MATCLASS = 4; // MATERIAL_CLASS.leaf — crown clusters (?crownlod0)

  // kVoxFanoutArgs: clear the voxel cursor + size the one-thread-per-entry dispatch
  // over the live qRaster count. Runs BEFORE kVoxFanout (its cursor + dispatch args).
  const kVoxFanoutArgs = Fn(() => {
    atomicStore(voxCount.element(0), uint(0));
    const n = minU(aLoadU(counters.element(1)), uint(qCap));
    // one workgroup (64 threads) per 64 qRaster entries
    split2D(voxFanoutDispatch, n.add(uint(63)).div(uint(64)));
  })().compute(1, [1]);
  (kVoxFanoutArgs as unknown as ComputeKernel).setName('nanVoxFanoutArgs');

  // kVoxFanout: one thread per qRaster entry → matClass==voxel ⇒ append to qVoxRaster.
  const kVoxFanout = Fn(() => {
    const tid = wgLinear(DISPATCH_ROW).mul(uint(64)).add(localX()).toVar();
    const itemCount = minU(aLoadU(counters.element(1)), uint(qCap));
    returnIf(tid.greaterThanEqual(itemCount));
    const item = qRasterV.ro.element(tid.add(uint(1)));
    const instId = item.x.toVar();
    const ci = item.y.toVar();
    // mesh matClass = byte 1 of mesh word6 ((w6>>8)&0xff), reached via the cluster's
    // meshId (cluster word7>>16). readCluster gives meshId; read the mesh word6 byte.
    const meshId = readCluster(gpu.clusters, ci).meshId.toVar();
    const matClass = elemU(gpu.meshes, meshId.mul(uint(MESH_WORDS)).add(uint(6)))
      .shiftRight(uint(8))
      .bitAnd(uint(0xff));
    If(matClass.equal(uint(VOXEL_MATCLASS)), () => {
      const slot = atomicAdd(voxCount.element(0), uint(1)) as unknown as NU;
      If(slot.lessThan(uint(vCap)), () => {
        qVoxRasterV.rw.element(slot.add(uint(1))).assign(uv2(instId, ci));
      });
    });
  })().compute(qCap, [64]);
  (kVoxFanout as unknown as ComputeKernel).setName('nanVoxFanout');

  // kVoxRasterArgs: publish qVoxRaster[0] = (count, 0) + the Stage-2 voxel-raster
  // 2D-split dispatch args (over the fanned voxel-cluster count). The count read goes
  // through the SAME rw view that writes slot 0 (N0 same-scope law, like kRasterArgs).
  const kVoxRasterArgs = Fn(() => {
    atomicMax(hwV.element(2), aLoadU(voxCount.element(0)));
    const n = minU(aLoadU(voxCount.element(0)), uint(vCap)).toVar();
    qVoxRasterV.rw.element(0).assign(uv2(n, 0));
    split2D(voxRasterDispatch, n);
  })().compute(1, [1]);
  (kVoxRasterArgs as unknown as ComputeKernel).setName('nanVoxRasterArgs');

  // ──────────────────────────────────────────────────────────────────────────
  // DEPTH-BUCKET FRONT-TO-BACK fan-out (?voxf2b). Replaces the single-atomicAdd
  // append above with a RANGE → COUNT → PREFIX → SCATTER pipeline that partitions
  // the voxel clusters into K contiguous depth slabs of qVoxRaster, ordered NEAR→FAR.
  // All passes are one-thread-per-qRaster-entry like kVoxFanout (cheap; the voxel cut
  // is ≪ the tri cut). Each cluster is assigned to EXACTLY ONE bucket and appended
  // ONCE (count==scatter via the prefix) ⇒ the SET of work-items is unchanged, only
  // partitioned + reordered ⇒ byte-exact image, only redundant atomicMax+stores skipped.
  // ──────────────────────────────────────────────────────────────────────────

  // LINEAR VIEW DEPTH of a voxel cluster: transform its LOCAL sphere centre (cluster
  // word0-2) by the instance transform → world, then cam.vp·world .w = clip-w = the
  // camera-space distance along the view axis (NOT the perspective NDC z = clip.z/clip.w,
  // which collapses the far field into one bucket — the prior attempt's bug). Always > 0
  // for in-front clusters (a behind-camera cluster gets a finite small/neg w → clamped to
  // bucket 0; still loss-exact). Returns the f32 depth.
  const voxClusterDepth = (instId: NU, ci: NU): NF => {
    const A = gpu.instances.element(instId.mul(uint(2))).toVar() as unknown as NV4;
    const B = gpu.instances.element(instId.mul(uint(2)).add(uint(1))).toVar() as unknown as NV4;
    const yawSc = instYaw(B);
    const c = readCluster(gpu.clusters, ci);
    const wc = instTransformPoint(A, B, yawSc, c.sphere.xyz as unknown as NV3);
    const d = (cam.vp.mul(vec4(wc, 1)) as unknown as NV4).w as unknown as NF;
    return d.max(float(1e-3)) as unknown as NF; // keep strictly positive (bit-monotone)
  };

  // bucket = clamp(floor((d-dMin)/(dMax-dMin)·K), 0, K-1) using the live [dMin,dMax]
  // read from voxRange (kVoxRange's atomicMin/atomicMax). Degenerate range (dMax≤dMin,
  // e.g. ≤1 cluster) ⇒ bucket 0 (still loss-exact, just one slab). t is clamped to
  // [0,1) BEFORE the floor so the uint cast never sees a negative (no wrap).
  const voxDepthBucket = (d: NF): NU => {
    const dMin = bcU2F(aLoadU(voxRange.element(0))).toVar();
    const dMax = bcU2F(aLoadU(voxRange.element(1))).toVar();
    const span = dMax.sub(dMin).toVar();
    const tRaw = span.greaterThan(float(1e-6)).select(d.sub(dMin).div(span), float(0)) as unknown as NF;
    const t = tRaw.clamp(float(0), float(0.999999)).toVar();
    const bIdx = (uint(toI(t.mul(float(K)).floor())) as unknown as NU).toVar();
    return minU(bIdx, uint(K - 1));
  };

  // ?voxprev TWO-PASS partition bit: 1 ⇔ the cluster's world sphere classifies
  // PROBABLY-OCCLUDED against the PREV-frame full-content HZB via the LIBERAL centre
  // test (prevVp/prevCamPos — the same uniform pair the emit test uses; the classifier
  // is deliberately NOT the emit closure, see the flag comment). Sphere math mirrors
  // the emit site exactly (instWorldSphere + swayPad) so the partition classifies the
  // same bound the raster will paint. Self-contained: fetches the A/B instance words
  // itself (mirroring voxClusterDepth) — it is only ever called INSIDE the
  // If(matClass==7) branch, so every node it builds stays in the conditional subtree
  // (TSL hoist discipline; hoisting A/B above the If would pay 2 dead instance loads
  // per non-voxel entry).
  const voxPrevBucket = (instId: NU, ci: NU): NU => {
    const A = gpu.instances.element(instId.mul(uint(2))).toVar() as unknown as NV4;
    const B = gpu.instances.element(instId.mul(uint(2)).add(uint(1))).toVar() as unknown as NV4;
    const c = readCluster(gpu.clusters, ci);
    const isHF = c.flags.bitAnd(uint(1)).notEqual(uint(0));
    const swayPad = bcU2F(elemU(gpu.meshes, c.meshId.mul(uint(MESH_WORDS)).add(uint(11))));
    const s = instWorldSphere(A, B, isHF as unknown as NB, c.sphere, swayPad);
    const occ = voxPrevTest!(s.center, s.radius, cam.prevVp, cam.prevCamPos);
    return (occ as unknown as { select(a: NU, b: NU): NU }).select(uint(1), uint(0));
  };

  // kVoxRangeArgs: clear voxRange (min=+inf, max=0) + the K bucket counters → 0, and
  // size the one-thread-per-entry dispatch over the live qRaster count (shared
  // voxFanoutDispatch — kVoxRange/kVoxCount/kVoxScatterFan all run this same grid).
  const kVoxRangeArgs = Fn(() => {
    atomicStore(voxRange.element(0), bcF2U(float(3.4e38))); // +inf-ish for atomicMin
    atomicStore(voxRange.element(1), uint(0)); // 0 bits for atomicMax (smallest +float)
    for (let b = 0; b < K; b++) {
      atomicStore(voxBucketCount.element(uint(b)), uint(0));
    }
    const n = minU(aLoadU(counters.element(1)), uint(qCap));
    split2D(voxFanoutDispatch, n.add(uint(63)).div(uint(64)));
  })().compute(1, [1]);
  (kVoxRangeArgs as unknown as ComputeKernel).setName('nanVoxRangeArgs');

  // kVoxRange: one thread per qRaster entry → voxel ⇒ atomicMin/atomicMax its linear
  // view depth into voxRange. Self-tightening, exactly brackets the live voxel depths
  // (anti-NDC: linear, so a 60 m and a 90 m crown land in DIFFERENT buckets).
  const kVoxRange = Fn(() => {
    const tid = wgLinear(DISPATCH_ROW).mul(uint(64)).add(localX()).toVar();
    const itemCount = minU(aLoadU(counters.element(1)), uint(qCap));
    returnIf(tid.greaterThanEqual(itemCount));
    const item = qRasterV.ro.element(tid.add(uint(1)));
    const instId = item.x.toVar();
    const ci = item.y.toVar();
    const meshId = readCluster(gpu.clusters, ci).meshId.toVar();
    const matClass = elemU(gpu.meshes, meshId.mul(uint(MESH_WORDS)).add(uint(6)))
      .shiftRight(uint(8))
      .bitAnd(uint(0xff));
    If(matClass.equal(uint(VOXEL_MATCLASS)), () => {
      const dBits = bcF2U(voxClusterDepth(instId, ci));
      atomicMin(voxRange.element(0), dBits);
      atomicMax(voxRange.element(1), dBits);
    });
  })().compute(qCap, [64]);
  (kVoxRange as unknown as ComputeKernel).setName('nanVoxRange');

  // kVoxCount: one thread per qRaster entry → voxel ⇒ atomicAdd the cluster's bucket
  // counter (the histogram). Range is final (kVoxRange ran first in the batch).
  const kVoxCount = Fn(() => {
    const tid = wgLinear(DISPATCH_ROW).mul(uint(64)).add(localX()).toVar();
    const itemCount = minU(aLoadU(counters.element(1)), uint(qCap));
    returnIf(tid.greaterThanEqual(itemCount));
    const item = qRasterV.ro.element(tid.add(uint(1)));
    const instId = item.x.toVar();
    const ci = item.y.toVar();
    const meshId = readCluster(gpu.clusters, ci).meshId.toVar();
    const matClass = elemU(gpu.meshes, meshId.mul(uint(MESH_WORDS)).add(uint(6)))
      .shiftRight(uint(8))
      .bitAnd(uint(0xff));
    If(matClass.equal(uint(VOXEL_MATCLASS)), () => {
      // build-time ternary: visibility bit (?voxprev) or depth slab (?voxf2b legacy) —
      // the unused branch (and its bindings: hzb pyramid vs voxRange) is compiled out.
      const bIdx = voxPrev
        ? voxPrevBucket(instId, ci)
        : voxDepthBucket(voxClusterDepth(instId, ci));
      atomicAdd(voxBucketCount.element(bIdx), uint(1));
    });
  })().compute(qCap, [64]);
  (kVoxCount as unknown as ComputeKernel).setName('nanVoxCount');

  // kVoxPrefix (1 thread): exclusive prefix-sum the K bucket counts → per-bucket base;
  // publish voxBucketRange[b]=(base_b,count_b); qVoxRaster[0]=(total,0); clear voxCursor.
  // Unrolled over K on the CPU (K is a build-time constant). Bases clamped to QVOX_CAP so
  // the scatter never writes past the queue (graceful overflow). The per-bucket 2D-split
  // INDIRECT args are written by the K tiny kVoxBucketArgs kernels below (NOT here): each
  // dispatch attr is its OWN storage buffer, so writing K of them in one kernel would bind
  // K+ buffers and blow the per-stage storage-buffer limit at K=16/24. Splitting it keeps
  // each kernel at ≤2 buffers; the K args kernels are 1-thread direct dispatches.
  const kVoxPrefix = Fn(() => {
    const acc = uint(0).toVar();
    const accRaw = uint(0).toVar(); // pre-clamp demand (queue high-water diag)
    for (let b = 0; b < K; b++) {
      const cntRaw = aLoadU(voxBucketCount.element(uint(b))).toVar();
      accRaw.assign(accRaw.add(cntRaw));
      const cnt = minU(cntRaw, uint(vCap)).toVar();
      const base = minU(acc, uint(vCap)).toVar();
      const room = uint(vCap).sub(base).toVar();
      const cClamp = minU(cnt, room).toVar();
      voxBucketRangeV.rw.element(uint(b)).assign(uv2(base, cClamp));
      atomicStore(voxCursor.element(uint(b)), uint(0));
      acc.assign(acc.add(cClamp));
    }
    atomicMax(hwV.element(2), accRaw);
    const total = minU(acc, uint(vCap)).toVar();
    qVoxRasterV.rw.element(0).assign(uv2(total, 0));
    atomicStore(voxCount.element(0), total); // keep readVoxCount (HUD/overflow) correct in F2B
    split2D(voxRasterDispatch, total); // legacy full-range args still published (HUD/A-B)
  })().compute(1, [1]);
  (kVoxPrefix as unknown as ComputeKernel).setName('nanVoxPrefix');

  // kVoxBucketArgs[b] (1 thread each): read this bucket's count from voxBucketRange[b].y →
  // 2D-split into its OWN indirect dispatch attr (one workgroup per cluster in the bucket).
  // Each binds only {voxBucketRange.ro, voxBucketDispatch[b]} = 2 storage buffers, so K of
  // them never exceed the per-stage limit (vs cramming all K attrs into kVoxPrefix). They
  // run AFTER kVoxPrefix in the batch (RAW on voxBucketRange). voxBucketRange.ro reads the
  // value kVoxPrefix wrote via .rw — a cross-dispatch RAW, auto-synced inside the pass.
  const kVoxBucketArgs: unknown[] = [];
  for (let b = 0; b < K; b++) {
    const kb = Fn(() => {
      const cnt = voxBucketRangeV.ro.element(uint(b)).y.toVar();
      split2D(voxBucketDispatch[b], cnt);
    })().compute(1, [1]);
    (kb as unknown as ComputeKernel).setName(`nanVoxBucketArgs${b}`);
    kVoxBucketArgs.push(kb);
  }

  // kVoxScatterFan: one thread per qRaster entry → voxel ⇒ slot = base_b +
  // atomicAdd(cursor_b) → qVoxRaster[slot+1]=(instId,ci). Recomputes d+bucket (cheap).
  // The bucket's contiguous slice [base_b, base_b+count_b) is filled here.
  const kVoxScatterFan = Fn(() => {
    const tid = wgLinear(DISPATCH_ROW).mul(uint(64)).add(localX()).toVar();
    const itemCount = minU(aLoadU(counters.element(1)), uint(qCap));
    returnIf(tid.greaterThanEqual(itemCount));
    const item = qRasterV.ro.element(tid.add(uint(1)));
    const instId = item.x.toVar();
    const ci = item.y.toVar();
    const meshId = readCluster(gpu.clusters, ci).meshId.toVar();
    const matClass = elemU(gpu.meshes, meshId.mul(uint(MESH_WORDS)).add(uint(6)))
      .shiftRight(uint(8))
      .bitAnd(uint(0xff));
    If(matClass.equal(uint(VOXEL_MATCLASS)), () => {
      // MUST match kVoxCount's bucket exactly (count==scatter invariant; Q1 in the spec —
      // identical expression trees against identical pyramid bytes, the shipped F2B idiom).
      const bIdx = (voxPrev
        ? voxPrevBucket(instId, ci)
        : voxDepthBucket(voxClusterDepth(instId, ci))
      ).toVar();
      const range = voxBucketRangeV.ro.element(bIdx);
      const base = range.x.toVar();
      const cnt = range.y.toVar();
      const off = atomicAdd(voxCursor.element(bIdx), uint(1)) as unknown as NU;
      // guard the (clamped-bucket) overflow tail: drop entries past count_b.
      If(off.lessThan(cnt), () => {
        const slot = base.add(off).toVar();
        qVoxRasterV.rw.element(slot.add(uint(1))).assign(uv2(instId, ci));
      });
    });
  })().compute(qCap, [64]);
  (kVoxScatterFan as unknown as ComputeKernel).setName('nanVoxScatterFan');

  // ──────────────────────────────────────────────────────────────────────────

  // N8-HIC: HIERARCHICAL traversal (opts.hier). Instead of dispatching every
  // cluster of every visible instance, seed each hierarchical mesh's ROOTS into a
  // frontier and BFS-descend: project(ownError) ≤ τ ⇒ emit (frustum + cone + size
  // culls), else ⇒ enqueue children (gpu.dagLinks). Ping-pong frontiers (read A /
  // write B, then swap) so no buffer is read+written in one pass (N0 law). Counts
  // in frontierCount[0]=A,[1]=B. v1 single-phase, no occlusion (validate at occl=0
  // vs the brute-force; the cut it produces is the SAME, just reached top-down).
  // ──────────────────────────────────────────────────────────────────────────
  let runPhase1: (renderer: Renderer) => void;
  // item 4: the exact ordered BFS batch (assigned inside the block below) — exposed so a
  // caller can concatenate disjoint culls into one submit (phase1Batch()).
  let phase1BatchList: readonly unknown[] = [];
  {
    const HIER_MAX_DEPTH = hierDepth; // ≥ the deepest DAG (measured ~13) + margin
    // Frontier capacity (items in flight during the BFS). The CAMERA needs the full
    // QRASTER_CAP (dense far forest floods the frontier); SHADOW culls run 6-10 cull
    // chains (clipmap levels / cascades) and each cut is ≤ a few hundred K clusters,
    // so a full 8M-entry frontier × 2 × N-chains is wasted GBs — they pass a smaller
    // frontierCap. Overflow drops frontier items (⇒ missing casters), so keep margin.
    const fcap = frontierCap;
    const qFrontierAAttr = new StorageBufferAttribute(new Uint32Array(fcap * 2), 2);
    const qFrontierA = sUvec2(qFrontierAAttr, fcap);
    const qFrontierBAttr = new StorageBufferAttribute(new Uint32Array(fcap * 2), 2);
    const qFrontierB = sUvec2(qFrontierBAttr, fcap);
    // frontier counts A/B reuse counters slots 2/3 (rejInst/rejClust — unused in
    // hier mode) so kTraverse stays ≤10 storage buffers WITH the HZB occlusion read
    // (the HZB pyramid is itself a storage buffer, NaniteHzb).
    const frontierCount = counters;
    // frontier counts use counters slots 0/4 (chunk-push / chunk-snapshot — unused
    // in hier mode), leaving slot 3 (rejClust count) + the rejClustV buffer free for
    // the TWO-PHASE occlusion re-test (record-not-drop, then kClusterCull2b vs fresh HZB).
    const FA = 0;
    const FB = 4;
    const traverseDispatchAttr = new IndirectStorageBufferAttribute(new Uint32Array(3), 3);
    const traverseDispatch = sU32Views(traverseDispatchAttr as unknown as StorageBufferAttribute, 3).rw;

    // kClearHier: counters + both frontier counts → 0
    const kClearHier = Fn(() => {
      If(instanceIndex.lessThan(uint(8)), () => {
        atomicStore(counters.element(instanceIndex), uint(0)); // includes the FA/FB slots
      });
    })().compute(8, [8]);
    (kClearHier as unknown as ComputeKernel).setName('nanClearHier');

    // kSeedRoots: per instance → frustum + size cull → append the mesh's ROOTS to
    // frontier A (skips non-hierarchical meshes: rootCount 0 = terrain/discrete).
    const kSeedRoots = Fn(() => {
      returnIf(instanceIndex.greaterThanEqual(uint(instanceCount)));
      const A = gpu.instances.element(instanceIndex.mul(uint(2))).toVar() as unknown as NV4;
      const B = gpu.instances
        .element(instanceIndex.mul(uint(2)).add(uint(1)))
        .toVar() as unknown as NV4;
      const headId = elemU(gpu.instanceMesh, instanceIndex).toVar();
      const headBase = headId.mul(uint(MESH_WORDS)).toVar();
      const head = readMesh(gpu.meshes, headId);
      const rootCount = head.rootCount.toVar();
      returnIf(rootCount.equal(uint(0)));
      // PERF-VB3: the per-mesh DRAW ENVELOPE — drop an instance beyond its mesh's max
      // draw distance (lodNext==LOD_NONE && lodDist>0 && dist>lodDist), EXACTLY as the
      // brute lodSelectAndPush does. This is the hier far-field bound that was missing:
      // without it every visible instance to the horizon seeds a root → the per-instance
      // flood (trees the brute path culls at lodDist). Terrain lodDist=0 ⇒ unlimited.
      const lodNext = elemU(gpu.meshes, headBase.add(uint(4)));
      const lodDist = bcU2F(elemU(gpu.meshes, headBase.add(uint(5))));
      const instDist = cam.camPos.sub(A.xyz).length();
      returnIf(lodNext.equal(uint(LOD_NONE)).and(lodDist.greaterThan(0)).and(instDist.greaterThan(lodDist)));
      // voxel-foliage (spec §3 / Stage 3a) — the mesh→voxel HANDOFF, NEAR side. The mesh's
      // word-8 nearDist is the per-mesh NEAR draw envelope: drop an instance NEARER than it
      // (the VOXEL sibling sets nearDist=transitionDist so it seeds only BEYOND the handoff,
      // while the LEAF head's lodDist=transitionDist keeps it nearer → a clean hard switch,
      // either mesh OR voxel at a distance, no double-render, no gap). 0 = unlimited near
      // (every non-voxel mesh). The cull picks tier by distance — NOT a per-cluster math path.
      const nearDist = bcU2F(elemU(gpu.meshes, headBase.add(uint(8))));
      returnIf(nearDist.greaterThan(0).and(instDist.lessThan(nearDist)));
      const isHF = head.flags.bitAnd(uint(MESH_FLAG_HEIGHTFIELD)).notEqual(uint(0));
      const s = instWorldSphere(A, B, isHF as unknown as NB, head.sphere, head.swayPad);
      returnIf(frustumVisible(s.center, s.radius).lessThan(0.5));
      const sizePx = projK
        .mul(s.radius)
        .mul(2)
        .div(cam.camPos.sub(s.center).length().max(float(1e-3))) as unknown as NF;
      // MESH_FLAG_FARTILE exemption (2026-07-02 beautification): a far TILE is the far-field
      // representation itself — size-culling it deletes whole 64 m chunks once its ~100 m
      // sphere projects under instMinPx (~1.2 km at retina), the user-visible aerial holes.
      const isFartile = head.flags.bitAnd(uint(MESH_FLAG_FARTILE)).notEqual(uint(0));
      // GRASS exemption (S2, 2026-07-03): the ~110 px default instMinPx deleted
      // whole 4 m grass patches beyond ~140 m (measured: ZERO grass clusters at
      // 150 m) — the sward's far band is the patch DAG's own coarse levels, so
      // size-culling the instance removes the far-field representation itself
      // (the FARTILE argument exactly). Channel byte is already loaded — free.
      const isGrassCh = head.channel.equal(uint(TRANSFORM_CHANNEL.grass));
      returnIf(
        instMinPx
          .greaterThan(0)
          .and(sizePx.lessThan(instMinPx))
          .and(isFartile.not() as unknown as NB)
          .and(isGrassCh.not() as unknown as NB),
      );
      const rootBase = head.rootBase.toVar();
      const slotBase = (atomicAdd(frontierCount.element(FA), rootCount) as unknown as NU).toVar();
      loopU(uint(0), rootCount, (k) => {
        const slot = slotBase.add(k);
        If(slot.lessThan(uint(fcap)), () => {
          const ci = elemU(gpu.dagLinks, rootBase.add(k));
          qFrontierA.rw.element(slot).assign(uv2(instanceIndex, ci));
        });
      });
    })().compute(instanceCount, [64]);
    (kSeedRoots as unknown as ComputeKernel).setName('nanSeedRoots');

    // one BFS pass: read inV[0..inCount), for each (instId, cluster) EMIT if the
    // cut resolves, else enqueue its children to outV. Shared by the A→B / B→A
    // ping-pong (read + write are different buffers).
    const makeTraverse = (
      inV: ReturnType<typeof sUvec2>,
      outV: ReturnType<typeof sUvec2>,
      inIdx: number,
      outIdx: number,
    ): unknown => {
      const kn = Fn(() => {
        // ONE thread per frontier item: global id = wg·64 + localX (kInstCull2 form)
        const tid = wgLinear(DISPATCH_ROW).mul(uint(64)).add(localX()).toVar();
        returnIf(tid.greaterThanEqual(aLoadU(frontierCount.element(inIdx))));
        const item = inV.ro.element(tid);
        const instId = item.x.toVar();
        const ci = item.y.toVar();
        const A = gpu.instances.element(instId.mul(uint(2))).toVar() as unknown as NV4;
        const B = gpu.instances
          .element(instId.mul(uint(2)).add(uint(1)))
          .toVar() as unknown as NV4;
        const yawSc = instYaw(B);
        const rec = readDag(gpu.dag, ci);
        // project(ownError) — the cut's pOwn (perspective, instance-scaled)
        const ownC = instTransformPoint(A, B, yawSc, rec.ownSphere.xyz as unknown as NV3);
        const ownR = instSphereRadius(A, B, rec.ownSphere.w as unknown as NF, float(0));
        const dvo = cam.camPos.sub(ownC) as unknown as NV3;
        let denO = dot(dvo, dvo).sub(ownR.mul(ownR)).max(float(1e-6)).sqrt() as unknown as NF;
        let warpDist = cam.camPos.sub(ownC).length() as unknown as NF;
        if (opts?.lodRingSnap) {
          // P10: Chebyshev light-plane distance, snapped UP to the ring ladder —
          // basis derived in-shader from the SAME sun formulas fitLevels uses.
          const base = opts.lodRingSnap;
          const fwdN = (normalize(vec3(sunU.dir)) as unknown as { mul(o: number): NV3 }).mul(-1);
          const wUp = (abs((fwdN as unknown as { y: NF }).y).greaterThan(0.99) as unknown as {
            select(a: unknown, b: unknown): NV3;
          }).select(vec3(0, 0, 1), vec3(0, 1, 0));
          const rgt = normalize(cross(wUp, fwdN as unknown as NV3)) as unknown as NV3;
          const upA = cross(fwdN as unknown as NV3, rgt) as unknown as NV3;
          const dr = (abs(dot(dvo, rgt)) as unknown as NF);
          const du = (abs(dot(dvo, upA)) as unknown as NF);
          const cheb = dr.max(du).max(float(base));
          const snapped = float(base).mul(exp2(ceil(log2(cheb.div(base))))) as unknown as NF;
          denO = snapped;
          warpDist = snapped;
        }
        const pOwn = projK.mul(A.w).mul(rec.ownError).div(denO);
        const tauEff = lodWarp(
          tau,
          warpDist,
          simBandD,
          lodNear,
          lodPow,
        ).toVar();
        // ?voxtaucap (DEFAULT 8 px, 0 = off): the lodWarp balloons τ_eff to ~27-45 px at
        // 200-500 m (its own header says "this knob never ships", yet it is the production
        // default). A VOXEL cluster emitted at that error paints its bricks as solid screen
        // rects — the user-visible "far crown becomes one massive square". Clamp τ_eff for
        // brick-backed clusters only (trunk/mesh clusters keep the warp: their coarse
        // triangles still have real silhouettes), so no brick can be emitted coarser than
        // ~cap px — a far crown then always descends to a multi-brick level. Costs bricks
        // only in the warped band (60-300 m); painted-pixel area is invariant to brick size,
        // so the added cost is per-brick setup, not fill.
        const forceDescend = float(0).toVar();
        if (voxTauCap > 0 || crownLod0 || voxCoarsen > 0) {
          const mid7 = elemU(gpu.clusters, ci.mul(uint(CLUSTER_WORDS)).add(uint(7))).shiftRight(uint(16));
          const mc = elemU(gpu.meshes, mid7.mul(uint(MESH_WORDS)).add(uint(6)))
            .shiftRight(uint(8))
            .bitAnd(uint(0xff));
          if (voxCoarsen > 0) {
            // shadow "less detailed" crown: COARSEN voxel τ_eff (bigger τ ⇒ pOwn ≤ τ is met
            // at a coarser DAG level ⇒ fewer/bigger bricks). Replaces the camera fine cap;
            // trunk/mesh clusters are untouched, so trunk shadows keep their silhouette.
            If(mc.equal(uint(VOXEL_MATCLASS)), () => {
              tauEff.assign(tauEff.mul(float(voxCoarsen)));
            });
          } else if (voxTauCap > 0) {
            If(mc.equal(uint(VOXEL_MATCLASS)), () => {
              tauEff.assign(tauEff.min(float(voxTauCap)));
            });
          }
          if (crownLod0) {
            // ?crownlod0 (DEFAULT ON — user mandate): leaf crowns render at LOD0 (full
            // detail) or VOXEL, never a simplified aggregate level. The leaf head's draw
            // envelope = voxnear, so every leaf cluster reaching this CAMERA traverse is
            // inside the mesh band. Force any leaf cluster that still has DAG children to
            // DESCEND regardless of screen-error; only true LOD0 leaves (childCount==0 ⇒
            // ownError 0 ⇒ pOwn 0 ≤ τ) emit — watertight, no hole. Shadow culls never pass
            // crownLod0 (caster leaf LOD stays coarse — LOD0 casters would blow the raster).
            If(mc.equal(uint(LEAF_MATCLASS)).and(rec.childCount.greaterThan(uint(0))), () => {
              forceDescend.assign(1);
            });
          }
        }
        If(pOwn.lessThanEqual(tauEff).and(forceDescend.equal(0)), () => {
          // ── CUT: this cluster is the right LOD here → emit (with culls) ──────
          const c = readCluster(gpu.clusters, ci);
          const isHF = c.flags.bitAnd(uint(1)).notEqual(uint(0)).toVar();
          const swayPad = bcU2F(
            elemU(gpu.meshes, c.meshId.mul(uint(MESH_WORDS)).add(uint(11))),
          );
          const s = instWorldSphere(A, B, isHF as unknown as NB, c.sphere, swayPad);
          const visible = frustumVisible(s.center, s.radius).toVar();
          If(visible.greaterThan(0.5).and(minPx.greaterThan(0)), () => {
            const toC = s.center.sub(cam.camPos) as unknown as NV3;
            const distC = dot(toC, toC).max(float(1e-6)).sqrt();
            If(projK.mul(s.radius).div(distC).lessThan(minPx), () => {
              visible.assign(0);
            });
          });
          // S3 SHADOW CLIPMAP hollow (mirrors kClusterCull): drop clusters lying
          // ENTIRELY inside the next-finer level's box so each caster rasters into
          // exactly one clipmap level (the hollow ring). cam.vp here is the level's
          // ORTHO VP (w==1 → clip.xy ∈ [-1,1] over [±E]); rClip = radius·(1/E). The
          // radius margin keeps boundary-straddlers in BOTH levels (no seam). No-op
          // when innerInvHalf==0 (camera path + the finest level).
          If(visible.greaterThan(0.5).and(innerInvHalf.greaterThan(0)), () => {
            const clip = cam.vp.mul(vec4(s.center, 1)) as unknown as NV4;
            const rClip = s.radius.mul(innerInvHalf);
            If(
              abs(clip.x).add(rClip).lessThan(0.5).and(abs(clip.y).add(rClip).lessThan(0.5)),
              () => {
                visible.assign(0);
              },
            );
          });
          if (coneCull) {
            If(
              visible.greaterThan(0.5).and(c.coneCos.greaterThan(-0.99)).and(isHF.not()),
              () => {
                const sinTest = float(1)
                  .sub(c.coneCos.mul(c.coneCos))
                  .max(0)
                  .sqrt()
                  .add(CONE_SLACK)
                  .toVar();
                If(sinTest.lessThan(1), () => {
                  const axisW = instRotateDir(yawSc, c.coneAxis);
                  const toC = s.center.sub(cam.camPos).toVar();
                  const d = toC.length();
                  If(
                    dot(toC as unknown as NV3, axisW).greaterThan(d.mul(sinTest).add(s.radius)),
                    () => {
                      visible.assign(0);
                    },
                  );
                });
              },
            );
          }
          // N8-HIC occlusion: single-phase, prev-frame HZB at emit (the two-phase
          // record/re-test is a follow-up — it needs a buffer-budget rework). null at occl=0.
          if (sphereOccluded) {
            If(visible.greaterThan(0.5), () => {
              If(sphereOccluded(s.center, s.radius, cam.prevVp, cam.prevCamPos), () => {
                visible.assign(0);
              });
            });
          }
          If(visible.greaterThan(0.5), () => {
            const slot = atomicAdd(counters.element(1), uint(1)) as unknown as NU;
            If(slot.lessThan(uint(qCap)), () => {
              qRasterV.rw.element(slot.add(uint(1))).assign(uv2(instId, ci));
            });
            atomicAdd(counters.element(6), c.triCount);
            atomicAdd(counters.element(5), uint(1));
            atomicAdd(counters.element(7), c.triCount);
          });
        }).Else(() => {
          // ── too coarse → enqueue children (only an OWNER carries them) ───────
          const childBase = rec.childBase.toVar();
          const childCount = rec.childCount.toVar();
          const base = (atomicAdd(frontierCount.element(outIdx), childCount) as unknown as NU).toVar();
          loopU(uint(0), childCount, (k) => {
            const slot = base.add(k);
            If(slot.lessThan(uint(fcap)), () => {
              const child = elemU(gpu.dagLinks, childBase.add(k));
              outV.rw.element(slot).assign(uv2(instId, child));
            });
          });
        });
      })().compute(fcap, [64]);
      return kn;
    };
    const kTraverseAB = makeTraverse(qFrontierA, qFrontierB, FA, FB);
    (kTraverseAB as ComputeKernel).setName('nanTraverseAB');
    const kTraverseBA = makeTraverse(qFrontierB, qFrontierA, FB, FA);
    (kTraverseBA as ComputeKernel).setName('nanTraverseBA');
    // SUBMIT-COALESCE enabler: attach each traverse's per-node INDIRECT dispatch attr
    // so the batched BFS (below) dispatches them at their tight traverseDispatch size
    // (NOT the baked QRASTER_CAP grid — the ~1B-thread over-dispatch trap). Both share
    // the one traverseDispatchAttr; the strict [kArgs(p), kTraverse(p)] order makes the
    // shared-buffer WAW/WAR between passes serialize correctly inside the single pass.
    setIndirectDispatch(kTraverseAB, traverseDispatchAttr);
    setIndirectDispatch(kTraverseBA, traverseDispatchAttr);

    // args before each pass: reset the OUTPUT count, dispatch over the INPUT count
    const makeArgs = (inIdx: number, outIdx: number): unknown => {
      const kn = Fn(() => {
        atomicStore(frontierCount.element(outIdx), uint(0));
        // queue high-water diag: RAW input frontier count (covers every BFS pass
        // including the seed output; pre-clamp so overflow demand is visible)
        atomicMax(hwV.element(1), aLoadU(frontierCount.element(inIdx)));
        // one workgroup (64 threads) per 64 frontier items
        const n = minU(aLoadU(frontierCount.element(inIdx)), uint(fcap));
        split2D(traverseDispatch, n.add(uint(63)).div(uint(64)));
      })().compute(1, [1]);
      return kn;
    };
    const kArgsAB = makeArgs(FA, FB);
    (kArgsAB as ComputeKernel).setName('nanArgsAB');
    const kArgsBA = makeArgs(FB, FA);
    (kArgsBA as ComputeKernel).setName('nanArgsBA');

    // SUBMIT-COALESCE: the WHOLE cull BFS in ONE batched submit (was kClearHier,
    // kSeedRoots, then 2×HIER_MAX_DEPTH separate dispatches + kRasterArgs = up to ~38
    // queue.submit drains/cull). The exact ORDER is load-bearing and PRESERVED here:
    //   [kClearHier, kSeedRoots, (kArgsAB, kTraverseAB, kArgsBA, kTraverseBA)×depth/2,
    //    kRasterArgs]
    // RAW/WAW correctness inside the single compute pass (inter-dispatch UAV auto-sync,
    // each dispatch its own usage scope):
    //   - kSeedRoots AFTER kClearHier (reads cleared counters / FA frontier count);
    //   - kArgs(p) writes traverseDispatch + zeroes the OUTPUT frontier count, then
    //     kTraverse(p) reads traverseDispatch INDIRECT (RAW on the STORAGE|INDIRECT
    //     buffer) and reads/writes the ping-pong frontiers (read inV.ro, write outV.rw
    //     — disjoint, N0 law). NEVER reorder kArgs(p) after kTraverse(p).
    //   - the single shared traverseDispatch is rewritten each kArgs: the strict
    //     interleave makes kArgsBA's write WAR-ordered after kTraverseAB's indirect read.
    //   - kRasterArgs LAST reads counters[1] (the BFS emit cursor, RAW) → qRaster[0] +
    //     rasterDispatch. (kRasterArgs2/syncFullArgs RAW-depends on qRaster[0] written
    //     here; the frame invokes it right after this batch — separately on the legacy
    //     path, or folded in via fullArgsBatch() under ?coalesce=1. Either way it runs
    //     strictly after kRasterArgs, so the RAW holds.)
    // The traverse kernels were setIndirectDispatch-tagged above so they keep their
    // tight indirect size in this batched (single dispatchSize=null) path.
    const bfsBatch: unknown[] = [kClearHier, kSeedRoots];
    for (let p = 0; p < HIER_MAX_DEPTH; p++) {
      if (p % 2 === 0) {
        bfsBatch.push(kArgsAB, kTraverseAB);
      } else {
        bfsBatch.push(kArgsBA, kTraverseBA);
      }
    }
    bfsBatch.push(kRasterArgs);
    phase1BatchList = bfsBatch;
    runPhase1 = (renderer: Renderer): void => {
      noteQueueHwRenderer(renderer); // queue high-water diag: stash for window.__qHW
      dispatchBatchMixed(renderer, bfsBatch);
    };
  }

  // queue high-water diag registration (window.__qHW): raw per-boot maxima for
  // THIS chain's qRaster / BFS frontier / vox fan-out cursors + the built caps.
  registerQueueHw({
    label: opts?.label ?? `cull${chainSeq++}`,
    caps: { qRaster: qCap, frontier: frontierCap, qVox: vCap },
    read: async (renderer: unknown): Promise<Record<string, number>> => {
      const u = new Uint32Array(await readBuffer(renderer as Renderer, hwAttr, 0, 16));
      return { qRaster: u[0] ?? 0, frontier: u[1] ?? 0, qVox: u[2] ?? 0 };
    },
    reset: (renderer: unknown): void => {
      dispatch(renderer as Renderer, kHwReset);
    },
  });

  const syncFullArgs = (renderer: Renderer): void => {
    dispatch(renderer, kRasterArgs2);
  };

  // voxel-foliage (spec §4.6 / §6.2): run the post-traverse FAN-OUT — clear+size args,
  // scan qRaster → matClass==voxel → qVoxRaster, publish the voxel-raster dispatch args.
  // Call AFTER runPhase1 (qRaster + counters[1] are live). The voxel raster then
  // dispatches over voxRasterDispatchAttr. No-op-safe if no voxel clusters were emitted
  // (qVoxRaster[0]=(0,0) ⇒ the raster dispatches 0 workgroups).
  // F2B indirect kernels keep their tight one-thread-per-qRaster-entry grid in the
  // single batched submit (sized once by kVoxRangeArgs into voxFanoutDispatch).
  setIndirectDispatch(kVoxRange, voxFanoutDispatchAttr);
  setIndirectDispatch(kVoxCount, voxFanoutDispatchAttr);
  setIndirectDispatch(kVoxScatterFan, voxFanoutDispatchAttr);
  // SUBMIT-COALESCE (?coalesce=1): tag the non-F2B fan-out too so voxFanoutBatch() keeps
  // its tight indirect grid inside the frame's single folded submit. Harmless to the
  // legacy explicit dispatchIndirect path (the outer arg short-circuits the node tag).
  setIndirectDispatch(kVoxFanout, voxFanoutDispatchAttr);
  // The whole F2B fan-out in ONE submit. Order is load-bearing (RAW chain): args →
  // range(min/max) → count(reads range) → prefix(reads counts, writes bases/ranges/
  // qVoxRaster[0]/cursor + per-bucket dispatch args) → scatter(reads bases/ranges).
  const voxF2bBatch: unknown[] = [
    kVoxRangeArgs, // still needed under voxprev: clears bucket counters + sizes the grid
    // voxprev partitions by VISIBILITY, not depth — the depth-range pass only feeds
    // voxDepthBucket (compiled out), so drop it (its voxRange clears in kVoxRangeArgs
    // become dead writes — harmless, 2 words).
    ...(voxPrev ? [] : [kVoxRange]),
    kVoxCount,
    kVoxPrefix,
    ...kVoxBucketArgs, // K tiny 1-thread args kernels (per-bucket indirect dispatch sizes)
    kVoxScatterFan,
  ];
  const runVoxFanout = (renderer: Renderer): void => {
    if (voxf2b) {
      dispatchBatchMixed(renderer, voxF2bBatch);
    } else {
      // ?voxf2b=0 — EXACTLY today's path: single-atomicAdd unordered append + single
      // dispatch. Byte-identical to the pre-F2B behaviour for the A/B control.
      dispatch(renderer, kVoxFanoutArgs);
      dispatchIndirect(renderer, kVoxFanout as never, voxFanoutDispatchAttr);
      dispatch(renderer, kVoxRasterArgs);
    }
  };

  const readVoxCount = async (renderer: Renderer): Promise<number> => {
    const buf = await readBuffer(renderer, voxCountAttr, 0, 4);
    return new Uint32Array(buf)[0] ?? 0;
  };

  // ?voxprev gate instrumentation (spec §6 R9, MANDATORY): per-bucket cluster counts from
  // voxBucketRange[b] = (base, count). B1 ≈ 0 at oblique ⇒ the classifier is degenerate and
  // NO perf number may be interpreted (a perf null with B1≈0 is a classifier failure, not
  // evidence about occlusion). CPU-only readback; called from meterRead (drained queue).
  const readVoxBuckets = async (renderer: Renderer): Promise<number[]> => {
    const buf = await readBuffer(renderer, voxBucketRangeAttr, 0, 8 * K);
    const u = new Uint32Array(buf);
    return Array.from({ length: K }, (_, b) => u[2 * b + 1] ?? 0);
  };

  const readCounts = async (renderer: Renderer): Promise<NaniteCullCounts> => {
    noteQueueHwRenderer(renderer); // queue high-water diag: stash for window.__qHW
    const [buf, head] = await Promise.all([
      readBuffer(renderer, countersAttr, 0, 32),
      readBuffer(renderer, qRasterAttr, 0, 8),
    ]);
    const u = new Uint32Array(buf);
    const q = new Uint32Array(head);
    // with occlusion, [0] holds the phase-2 re-expansion — phase 1 is in [4]
    const chunks = sphereOccluded ? (u[4] ?? 0) : (u[0] ?? 0);
    const visClusters = u[1] ?? 0;
    const rejInst = u[2] ?? 0;
    const rejClust = u[3] ?? 0;
    const dagClusters = u[5] ?? 0;
    const visTris = u[6] ?? 0;
    const dagTris = u[7] ?? 0;
    const p2Appends = Math.max(0, (q[0] ?? 0) - (q[1] ?? 0));
    let overflow: string | null = null;
    const over = (label: string, n: number, cap: number): void => {
      if (n > cap) overflow = `${overflow ? `${overflow}; ` : ''}${label} ${n} > ${cap}`;
    };
    over('qChunks', chunks, QCHUNK_CAP);
    over('qRaster', visClusters, qCap);
    over('rejInst', rejInst, REJ_INST_CAP);
    over('rejClust', rejClust, REJ_CLUST_CAP);
    return {
      chunks,
      visClusters,
      rejInst,
      rejClust,
      dagClusters,
      visTris,
      dagTris,
      p2Appends,
      overflow,
    };
  };

  return {
    qRasterRO: qRasterV.ro,
    qRasterAttr,
    rasterDispatchAttr,
    rasterDispatch2Attr,
    rasterDispatchFullAttr,
    qVoxRasterRO: qVoxRasterV.ro,
    qVoxRasterAttr,
    voxRasterDispatchAttr,
    voxBucketRangeRO: voxBucketRangeV.ro,
    voxBucketDispatchAttr,
    voxF2bK: K,
    voxF2bEnabled: voxf2b,
    voxPrevEnabled: voxPrev,
    runPhase1,
    phase1Batch: () => phase1BatchList,
    syncFullArgs,
    fullArgsBatch: () => [kRasterArgs2],
    voxFanoutBatch: () =>
      voxf2b ? voxF2bBatch : [kVoxFanoutArgs, kVoxFanout, kVoxRasterArgs],
    runVoxFanout,
    readVoxCount,
    readVoxBuckets,
    readCounts,
  };
}
