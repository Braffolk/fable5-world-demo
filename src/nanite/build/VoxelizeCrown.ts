/**
 * VoxelizeCrown — the OFFLINE (build-time) palette voxelizer for the 2-tier
 * voxel-foliage subsystem (docs/perf-runs/nanite-voxel-foliage-spec.md §5).
 *
 * Stage 1 scope (§11): voxelize ONE crown's foliage mesh (an ExplicitSource) into
 * COARSE one-sample-per-brick bricks (§6.4 — the HARD default) at the per-crown
 * `voxelGridDim` knob (§3.2.bis, default ~180 effective cells / crown edge), with
 * FRACTIONAL/coverage density (§5.4.3 — mandatory for thin conifer needles so they
 * become a low-density smear, not a hole or a blob) and a density-weighted per-brick
 * MEAN normal + MEAN color. Cost-tolerant: this runs at boot, not per frame.
 *
 * Output = a flat array of `BrickCPU` records (VoxelBrick.ts codec) + the per-brick
 * grid origin/extent so a caller can place/raster them. The grid is laid out in
 * BRICKS of 4×4×4 cells: brickGridDim = ceil(voxelGridDim / BRICK_DIM) per axis.
 *
 * METHOD (offline, accuracy over speed):
 *  - One cubic voxel grid over the crown's local AABB (padded), edge = voxelGridDim
 *    CELLS. Cells group into 4×4×4 BRICKS.
 *  - Per triangle: rasterize into the cell grid via a conservative triangle/box SAT
 *    overlap (Akenine-Möller 13-axis SAT), supersampled S³ per cell for FRACTIONAL
 *    coverage. A needle quad ~0.018 m wide spans < 1 cell, so the supersample makes
 *    it a partial-coverage cell (density in (0,1)) instead of a hard 0/1 (§5.4.3).
 *  - Per cell accumulate: coverage Σ, normal·coverage Σ (the BENT/source normals —
 *    they are the MEAN per §5.4 caution), color·coverage Σ.
 *  - Per BRICK aggregate the 64 cells: occupancy bits (cell coverage ≥ occThresh),
 *    density = mean cell coverage, mean normal = normalize(Σ normal·cov), mean color
 *    = Σ color·cov / Σ cov, spread = 1 − |mean-vector-length| (RAW variance proxy,
 *    carried for the SGGX fallback, ignored by the mean-normal shade path, §4.3/§7).
 *
 * The triangle SAT is conservative (no holes — Risk #7 / §5.4.2); the supersample is
 * the fractional-density layer on top (§5.4.3). This is the COARSE path: per-cell
 * occupancy is build/raster-only; the resolve reads only the per-brick mean normal.
 */

import type { DagBuild } from './BuildDag';
import { type Sphere, mergeSpheres, partitionClusters } from './DagCommon';
import { type DagHierarchy, validateDagHierarchy } from './DagHierarchy';
import type { ExplicitSource, GeometryRegistry, MeshHandle } from '../world/GeometryRegistry';
// VoxelBrickCore (NOT VoxelBrick): keeps this module's import chain THREE-FREE
// so prepareVoxelCrown can run inside the DagWorker module Worker (2026-07-04).
import {
  type BrickCPU,
  BRICK_DIM,
  BRICK_WORDS,
  brickCellIndex,
  brickCenterHalf,
  MAX_BRICKS_PER_CLUSTER,
  writeBrick,
} from '../voxel/VoxelBrickCore';
import type { PackedLevel, PackedPreparedCrown } from './CrownPack';

/** voxlod: number of MIP-pyramid levels (L0 finest … coarsest), each level 2x coarser (bricks 2x
 *  wider, the brick grid halved per axis). DEFAULT 7 = ceil(log2(maxDist/transitionDist))+1, the
 *  count needed for the BAND-ANCHORED ladder (below) to span [35,2000] m with one real 2x level per
 *  octave. The pyramid tail is cheap (Sum_{L>=1} 8^-L ~ 0.14x the L0 brick count worst-case, far
 *  less for a sparse crown shell), and L4-L6 are sub-cluster, so 7 vs 4 levels is ~free. voxlod=0
 *  ignores all of this (single-level degenerate always-cut DAG).
 *
 *  HOW THE CUT SELECTS A LEVEL. NaniteCull emits cluster at level L when pOwn=projK*A.w*ownError(L)/d
 *  <= tau AND its parent (L+1) wants finer (pOwn>tau), i.e. for camera distance d in [projK*A.w*
 *  ownError(L)/tau, projK*A.w*ownError(L+1)/tau]. With the anchored ownError(L)=anchorL0*2^L the
 *  band is [transitionDist*2^L, transitionDist*2^(L+1)] — the ladder, NOT a per-pixel cost crossover,
 *  is what gives the true far dropoff. (The earlier errorK*cellSize "floor-crossover/sparseK/shell"
 *  theory under-coarsened — it pinned every transition below the 35 m handoff so the whole band sat
 *  at ONE coarsest level — and is SUPERSEDED by the anchor + the Rank-2 occupancy gate; see the
 *  VOXLOD_CFG block.) RUNTIME-TUNABLE A/B levers: ?voxlodk= (anchor multiplier), ?voxlodlevels=,
 *  ?voxlodsparse= (1=off), ?voxlodshell= (0=off) via setVoxlodConfig. */
// ANCHORED LADDER (the decisive fix — premise-audit correction 1+2). ownError is NO LONGER
// tied to cellSize (that pinned every transition BELOW the 35 m mesh→voxel handoff, so the whole
// >=35 m band rendered at ONE coarsest level — the repeated "no-true-dropoff" botch). Instead the
// ladder is ANCHORED to the band: ownError(L0)=anchorL0=transitionDist*tau/projK (local metres,
// reference instScale 1.0 — A.w scales it per-instance at runtime exactly like the mesh qemErr),
// ownError(L)=anchorL0*errorK*2^L for L>=1, L0=0 (always-emit finest, hole-safe). Then the level
// L is SELECTED for camera distance d in [transitionDist*2^L, transitionDist*2^(L+1)] — every
// doubling of distance descends EXACTLY one octave, so the ladder SPANS [35, 2240] m with one real
// 2x level per octave (L0 35-70, L1 70-140 … L6 2240+). This keeps every emitted brick a BOUNDED
// 2.5-5 px across the WHOLE band (px at a level's near edge = projK*A.w*brickHalf0/transitionDist,
// CONSTANT across L) — never sub-px (no waste), never a solid blob (Rank-2 occupancy gate paints
// only the silhouette anyway). levels: 7 = ceil(log2(2000/35))+1 spans the band (4 saturated).
// errorK: anchor MULTIPLIER (1 = the band anchor; ?voxlodk sweeps the ladder farther/nearer).
// sparseK: 1 = the per-block sparsity multiplier DISABLED (it perturbed the clean per-octave ladder
// correction-3 needs; the old footprint-cost theory is superseded by the anchor + Rank-2 gate).
// shell: 0 = no interior-brick removal (removes geometry => hole risk on concave/thin crowns; the
// HARD "no holes" constraint wins, Rank-2's occupancy silhouette handles the far-overdraw instead).
// errorK 1 (2026-07-02b, was 3→2→1 through the day): the geometric "one brick ≈ τ px" anchor.
// errorK=3 made L0 own the band to ~380 m (no coarsening anywhere visible); =2 still left L0
// owning 60-250 m. Rested 200k A/B for =1 vs =2: oblique 55.9→38.5 ms (−31%), aerial −4, eye −2.
// The old "=1 collapses far crowns to blobs/squares" objection is now handled STRUCTURALLY:
// K_FLOOR=3 keeps a multi-brick far shell, ?voxbn shades per-brick, and ?voxcell renders big
// bricks as carved rotated cubes with 4³ sub-cell detail — the 125-250 m band reads chunkier
// than =2 but stays crown-shaped (screenshot-gated). ?voxlodk=2/3 = finer ladders for eyeball A/B.
const VOXLOD_CFG = { levels: 7, errorK: 1, sparseK: 1, shell: 0, anchorL0: 0 };

/** voxlod: number of MIP-pyramid levels (see VOXLOD_CFG). Read via the getter so a runtime
 *  ?voxlodlevels= override (set before build) takes effect without re-threading signatures. */
export function voxlodLevels(): number { return VOXLOD_CFG.levels; }

/** voxlod: anchor MULTIPLIER on the band-anchored ladder (DEFAULT 1). ownError(L>=1) = anchorL0 *
 *  errorK * 2^L, where anchorL0 = transitionDist*tau/projK (computeVoxlodAnchorL0). Units = LOCAL-
 *  space metres — the SAME unit + projection the mesh DAG uses (NaniteCull pOwn = projK*A.w*ownError
 *  /denO; BuildDag qemErr is object-space metres), so makeTraverse compares voxel + mesh errors
 *  against ONE tau with no rescale, and A.w scales ownError per-instance just like qemErr. errorK>1
 *  pushes the whole ladder farther (more detail), <1 nearer; 1 = the exact band anchor. L0 keeps
 *  ownError=0 (always emit, no children) so the finest level never holes. */
export function voxlodErrorK(): number { return VOXLOD_CFG.errorK; }

/** voxlod: cap on the per-block SPARSITY MULTIPLIER (see buildVoxelPyramid / the cut-lever doc).
 *  A block's ownError is BASE * clamp(solidArea/childArea, 1, sparseK). 1 => the multiplier is
 *  disabled (pure geometric error — the clean per-octave ladder correction-3 requires). Higher =>
 *  sparse blocks defer coarsening farther, but that PERTURBS the clean ladder (some same-level blocks
 *  transition at different distances), so DEFAULT 1 (OFF). Kept only as a ?voxlodsparse= A/B lever. */
export function voxlodSparseK(): number { return VOXLOD_CFG.sparseK; }

/** voxlod ANCHOR (premise-audit correction 1): ownError(L0) = transitionDist*tau/projK, the LOCAL-
 *  space geometric error whose per-octave ladder anchorL0*2^L lands level-L's cut at distance
 *  transitionDist*2^L (one octave per level, spanning the whole [35,2000] m voxel band). 0 = unset
 *  (no call-site provided it) => buildVoxelPyramid falls back to the legacy cellSize ladder (the
 *  under-coarsening regime — only hit by a direct caller that skips the WorldRegistry/ForestScene
 *  anchor compute). Both production paths set it via computeVoxlodAnchorL0() before build(). */
export function voxlodAnchorL0(): number { return VOXLOD_CFG.anchorL0; }

/** voxlod ANCHOR helper: the LOCAL-space ownError(L0) so the FINEST level's cut-transition lands at
 *  the mesh→voxel handoff distance `transitionDist` for a reference instance (A.w=1.0). Mirrors the
 *  cull's projK exactly (NaniteCull: projK = cot(fovY/2)*renderHeight*0.5; pOwn = projK*A.w*ownError
 *  /d <= tau). ownError(L)=anchorL0*2^L then gives transition T(L)=projK*A.w*ownError(L)/tau, which
 *  at A.w=1,tau=1 is transitionDist*2^L. The ladder SHAPE (octave spacing) is projK-independent — a
 *  rough projK only shifts the absolute anchor by <1 level, so the band is always spanned. */
export function computeVoxlodAnchorL0(transitionDist: number, renderHeight: number, fovDeg: number, tau = 1): number {
  // CLAMP the height/fov to sane floors: at build the canvas may not be sized yet (height 0) —
  // a degenerate projK would zero the anchor and drop us to the buggy cellSize ladder. The ladder
  // SHAPE is projK-robust (a wrong height only shifts the absolute anchor <1 octave), so a 1080
  // floor keeps the band spanned. Runtime re-sizing never rebuilds the bake, so this is a one-shot.
  const h = renderHeight > 0 && Number.isFinite(renderHeight) ? renderHeight : 1080;
  const fov = fovDeg > 1 && fovDeg < 179 ? fovDeg : 55;
  const projK = (1 / Math.tan(((fov * Math.PI) / 180) / 2)) * h * 0.5;
  if (!(projK > 0) || !(transitionDist > 0) || !Number.isFinite(projK)) return 0;
  return (transitionDist * tau) / projK;
}

/** voxlod FAR-CHEAPER FIX (iteration-6, the GO-UP-A-LEVEL root cause). The prior 5 iterations all
 *  stayed INSIDE "make the brick footprint cheaper" (errorK floor-crossover, sparseK, the iter-4
 *  occ-gate, the iter-5 dither) and ALL failed: measured vl1 writes > vl0 at every distance. The
 *  CONTEXT they never put on trial: the cost is NOT silhouette area, it is DEPTH OVERDRAW through
 *  the crown VOLUME — a single far tree writes ~37k elections into a ~few-hundred-px silhouette
 *  (~95x overdraw). OR-downsampling keeps the FULL crown volume at every coarse level, so the
 *  coarse crown is still ~10 bricks deep and overdraws just as much (worse, because its opaque
 *  boxes fill the fine level's see-through gaps). The fine level is "cheap" at far only because it
 *  UNDERSAMPLES into gaps (fillFrac ~0.17) — an artifact, not real sparsity.
 *
 *  THE FIX (build-time, raster UNTOUCHED per G3): SHELL the coarse levels. A coarse brick that is
 *  fully ENCLOSED (all 6 face-neighbours in the coarse brick grid occupied) is INTERIOR — it can
 *  never win a single election from any exterior view (its opaque neighbours' footprints occlude
 *  it on every side), so it contributes pure depth-overdraw and ZERO visible pixels. Dropping it
 *  (density->0, excluded from `occupied` like any empty brick) collapses the coarse crown from a
 *  ~10-deep VOLUME to a ~2-deep SHELL => far depth-overdraw falls below the fine level it replaces
 *  (farCheaper) with NO silhouette/surface change (no holes, no popping — only invisible interior
 *  is removed) and FEWER bricks (eases the budget, G4). L0 is NEVER shelled (shell starts at L1)
 *  so the NEAR crown keeps its full opaque volume (nearDenseCorrect, fine-end rule intact). The
 *  spatial parent->child vote + coverage-repair + the HARD validateDagHierarchy gate guarantee
 *  every finer block still reaches a coarse parent after shelling (fail-loud, never a hole).
 *  ?voxlodshell=0 disables it for the A/B (= the refuted iter-2 OR-volume behaviour). DEFAULT 1. */
export function voxlodShell(): boolean { return VOXLOD_CFG.shell !== 0; }

/** voxlod (verify lever): override the pyramid tuning at runtime BEFORE build(). Called from
 *  WorldRegistry/ForestScene with ?voxlodk= / ?voxlodlevels=. Out-of-range/NaN args are ignored
 *  so a bad query string can never break the build. levels clamped to [1,8] (default 7), errorK to
 *  (0, 32] (the ANCHOR MULTIPLIER, default 1 = the exact band anchor — NOT the old 6=2*tau
 *  floor-crossover framing), sparseK to [1, 64] (default 1 = OFF). */
export function setVoxlodConfig(cfg: { levels?: number; errorK?: number; sparseK?: number; shell?: number; anchorL0?: number }): void {
  if (cfg.levels !== undefined && Number.isFinite(cfg.levels)) {
    // [1,8] — 7 (default) = ceil(log2(2000/35))+1 spans the band; headroom to 8 for a sweep.
    VOXLOD_CFG.levels = Math.max(1, Math.min(8, Math.round(cfg.levels)));
  }
  if (cfg.errorK !== undefined && Number.isFinite(cfg.errorK) && cfg.errorK > 0) {
    // anchor MULTIPLIER now (default 1 = the band anchor). >1 pushes the ladder farther.
    VOXLOD_CFG.errorK = Math.min(32, cfg.errorK);
  }
  if (cfg.sparseK !== undefined && Number.isFinite(cfg.sparseK) && cfg.sparseK >= 1) {
    VOXLOD_CFG.sparseK = Math.min(64, cfg.sparseK);
  }
  // ?voxlodshell= — 1 RE-ENABLES the coarse-level interior SHELL (default 0 = off; removes geometry
  // => hole risk on concave/thin crowns, so off by default and Rank-2's occupancy gate handles the
  // far-overdraw the shell used to chase). Any non-zero turns it on for an A/B experiment.
  if (cfg.shell !== undefined && Number.isFinite(cfg.shell)) {
    VOXLOD_CFG.shell = cfg.shell !== 0 ? 1 : 0;
  }
  // anchorL0 — the LOCAL-space ownError(L0) (computeVoxlodAnchorL0). Set by the call site BEFORE
  // build() so buildVoxelPyramid lays the band-anchored octave ladder instead of the cellSize one.
  if (cfg.anchorL0 !== undefined && Number.isFinite(cfg.anchorL0) && cfg.anchorL0 > 0) {
    VOXLOD_CFG.anchorL0 = cfg.anchorL0;
  }
}

/** default per-crown voxel DETAIL knob (§3.2.bis): effective CELL edge count over the
 *  whole crown. A build-time uniform (?voxgrid=). 256 (2026-07-03, was 180 — the
 *  2026-07-02 beautification pick, now SHARED forest+world): L0 bricks ~0.156 m ≈
 *  4.4 px at the 60 m handoff — near-mid voxels read close to real geometry (user
 *  pivot: fine voxels beat more mesh distance; crown prep 36→48 s cold, boot-cached). */
export const DEFAULT_VOXEL_GRID_DIM = 256;

/** supersamples per cell per axis for fractional coverage (§5.4.3). S=3 ⇒ 27 sub-
 *  samples/cell — enough to resolve a sub-cell needle as partial density. Offline. */
const COVERAGE_SUPERSAMPLE = 3;

/** OCC_SOLID — a cell is occupied OUTRIGHT once its fractional coverage clears this.
 *  RAISED 0.02 → 0.15 (2026-07-09, user report "voxels too thicc at the finest level —
 *  the voxelizer too easily voxelizes on a barely-got-a-hit"). The COVERAGE PHYSICS
 *  (S³ = 27 supersamples, per-sub-sample weight subW = 1/27), verified against the raster
 *  math below:
 *    • a leaf PLANE fully crossing a cell lights one ~S² sheet of sub-cells ⇒ cov ≈ 1/S ≈
 *      0.33 (the histogram's single-leaf peak) — this is the MAIN SHAPE; always ≥ OCC_SOLID
 *      ⇒ it survives, silhouette intact.
 *    • a leaf/needle TIP grazing a corner lights 1-3 sub-cells ⇒ cov ≈ 0.037-0.11 — a
 *      "barely got a hit" speck. The old 0.02 threshold minted a SOLID cube for EVERY such
 *      graze (even a single sub-sample = 1/27 = 0.037 ≥ 0.02), so crowns read THICKER than
 *      the source mesh. 0.15 sits between the graze band (≤ ~0.11) and the plane peak (0.33).
 *  ⚠️ The 2026-07-02 "threshold sweeps were a visual no-op" record is STALE — it predates the
 *  occupancy-GATED ray-band renderer (NaniteVoxelRaster occ-gate mask + cOccLo/cOccHi DDA,
 *  ~:1092 / :1349) that now PROJECTS and DDA-marches the per-cell occupancy bits. Clearing a
 *  cell's bit removes it from the silhouette the raster paints ⇒ this erosion GEOMETRICALLY
 *  thins the crown (NOT a render-side dither; the vetoed ?voxalpha stipple stays default-off).
 *  Occupancy is build/raster-only in the coarse path (§4.4); density (word4) carries the real
 *  weight and is UNCHANGED (true mean coverage over all covered cells).
 *  TUNABLE via ?voxocc / setVoxOccThreshold (same clamps). If thin sparse conifer needles lose
 *  too much body this is the per-species tuning lever — LOWER OCC_SOLID for conifers (no
 *  species split unless the kept/dropped numbers demand it). */
let OCC_COVERAGE_THRESHOLD = 0.15;

/** OCC_GRAZE — the SUPPORT-HYSTERESIS floor (fixed constant, no URL knob). A BORDERLINE cell
 *  with coverage in [OCC_GRAZE, OCC_SOLID) is the shape-preserver's domain: it survives IFF
 *  ≥ OCC_SUPPORT_MIN of its 6 face-neighbours are SOLID (cov ≥ OCC_SOLID) — so a rim cell
 *  hugging the solid crown surface stays (the main shape keeps its skin), while an ISOLATED
 *  graze speck with no solid support is eroded. Below OCC_GRAZE a cell is ALWAYS dropped
 *  (pure "barely got a hit" corner-clip). 0.06 sits inside the graze band so the faintest
 *  clips go regardless of support, while the mid-graze band [0.06,0.15) is support-gated. */
const OCC_GRAZE = 0.06;
/** hysteresis: min SOLID face-neighbours (of 6) for a borderline cell to survive. */
const OCC_SUPPORT_MIN = 2;

/** ?voxocc= — override OCC_SOLID, the outright-occupancy coverage threshold (build-time; call
 *  BEFORE prepareVoxelCrown). Clamped to [0.005, 0.95]; NaN/out-of-range ignored. */
export function setVoxOccThreshold(t: number): void {
  if (Number.isFinite(t) && t >= 0.005 && t <= 0.95) OCC_COVERAGE_THRESHOLD = t;
}
export function voxOccThreshold(): number { return OCC_COVERAGE_THRESHOLD; }
let occHistoLogged = false;
/** per-crown erosion census: crowns are voxelized ONCE PER SPECIES POOL (not per instance —
 *  ForestScene/WorldRegistry loop the pools), so the first ~24 census lines ARE the per-species
 *  kept/eroded/dropped breakdown. Capped so a many-pool world can't spam the console. */
let occCensusCount = 0;
const OCC_CENSUS_CAP = 24;

/** voxlod: one resolved BLOCK of a pyramid level — a contiguous (<=MAX_BRICKS_PER_CLUSTER)
 *  run of that level's occupied bricks that becomes ONE voxel cluster. Carries the DAG cut
 *  metadata the multi-level attach needs: an OWN (error, sphere) pair for the block and a copy
 *  of its ONE SPATIAL parent block's pair (each finer block has exactly one coarse parent it
 *  spatially belongs to — a strict tree, so no double-emit and parent contains own stays crack-free). */
export interface VoxelBlock {
  /** index into THIS level's `occupied[]` where the block's bricks start. */
  start: number;
  /** number of occupied bricks in the block (<= MAX_BRICKS_PER_CLUSTER). */
  count: number;
  /** OWN sphere = bound CONTAINING all the block's brick footprints (local space). */
  own: Sphere;
  /** OWN geometric error (local-space metres). BASE = the BAND-ANCHORED ladder
   *  anchorL0*voxlodErrorK()*2^L for L>=1, L0=0 (anchorL0 = transitionDist*tau/projK — see
   *  VOXLOD_CFG); NOT cellSize-tied (the superseded cellSize ladder pinned every transition below
   *  the mesh->voxel handoff). RAISED per-block by the sparsity multiplier ONLY when ?voxlodsparse>1
   *  (default 1 = OFF). Monotone child<parent is enforced bottom-up. */
  ownError: number;
  /** Sum over this block's bricks of (2*half)^2 — the SOLID screen-area PROXY the raster paints
   *  (Phase B strides each brick's full [center+-half] AABB rect, occupancy IGNORED, so a brick's
   *  cost ~ its projected box area ~ half^2). Drives the sparsity multiplier: coarsening is a
   *  pixel WIN only when a coarse block's solidArea < the summed solidArea of the fine children
   *  it replaces; the multiplier raises ownError when it is NOT, so the cut keeps it fine. */
  solidArea: number;
  /** PARENT sphere = the coarser block that contains this block (null on the COARSEST
   *  level => this block is a DAG root: parentError +inf). */
  parent: Sphere | null;
  /** PARENT error (the coarser level's ownError) — Infinity when `parent` is null. */
  parentError: number;
  /** child links: the finer (level-1) blocks this coarse block is the SPATIAL parent of (by
   *  grid containment). EMPTY only on the finest level (L0). Every coarse block has >=1 child
   *  (built bottom-up + a coverage-repair pass) so it can always refine. Mesh-local to the
   *  appended voxel cluster set — the registry resolves them to global cluster ids. */
  childBlocks: number[];
}

/** voxlod: one MIP level of the crown pyramid. levels[0] is the FINEST (= the single-level
 *  voxlod=0 grid). Each coarser level halves the brick grid per axis (cellSize 2x). */
export interface VoxelLevel {
  level: number;
  /** flat BrickCPU records in this level's brick-grid order. */
  bricks: BrickCPU[];
  /** indices (into this level's `bricks`) of the NON-EMPTY bricks (density > 0). */
  occupied: number[];
  /** brick-grid edge counts for this level. */
  brickGrid: { x: number; y: number; z: number };
  /** world size of ONE cell at this level (= cellSize(0) * 2^level). */
  cellSize: number;
  /** geometric error for this level = the band-anchored ownError anchorL0*voxlodErrorK()*2^level
   *  (L0=0), local-space metres — NOT voxlodErrorK()*cellSize (the superseded cellSize ladder). */
  geomError: number;
  /** the level's blocks (one voxel cluster each), in `occupied[]` order. */
  blocks: VoxelBlock[];
}

export interface CrownVoxelization {
  /** flat BrickCPU records in brick-grid order (x fastest, then y, then z). */
  bricks: BrickCPU[];
  /** indices (into `bricks`) of the bricks that are NON-EMPTY (density > 0). The
   *  voxelizer keeps the full grid for addressing but a caller only uploads/draws
   *  the occupied ones. */
  occupied: number[];
  /** brick-grid edge counts (ceil(voxelGridDim / BRICK_DIM) per axis). */
  brickGrid: { x: number; y: number; z: number };
  /** cell-grid edge counts (the requested voxelGridDim rounded UP to a brick multiple). */
  cellGrid: { x: number; y: number; z: number };
  /** local-space grid origin (min corner of the padded crown AABB). */
  origin: [number, number, number];
  /** world size of ONE cell (uniform; cubic grid). */
  cellSize: number;
  /** voxlod: the MIP pyramid (levels[0] = finest = the single-level grid above). Present
   *  ONLY when voxelizeCrown was called with voxlod=true; undefined => single-resolution
   *  (voxlod=0) path, byte-identical to today (the `bricks`/`occupied` fields above stand). */
  levels?: VoxelLevel[];
  /** stats for the §5.3 budget print. */
  stats: {
    triangles: number;
    cellsTouched: number;
    occupiedBricks: number;
    totalBricks: number;
    meanDensity: number;
    voxelizeMs: number;
  };
}

// ---------------------------------------------------------------------------
// Triangle / axis-aligned-box overlap — Akenine-Möller 13-axis SAT.
// Box is centered at `c` with half-extent `h` (uniform). Triangle verts v0/v1/v2.
// Returns true if they intersect (conservative — used to gate sub-sample tests).
// ---------------------------------------------------------------------------

function triBoxOverlap(
  cx: number, cy: number, cz: number,
  hx: number, hy: number, hz: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  ccx: number, ccy: number, ccz: number,
): boolean {
  // move triangle into box-centered space
  const v0x = ax - cx, v0y = ay - cy, v0z = az - cz;
  const v1x = bx - cx, v1y = by - cy, v1z = bz - cz;
  const v2x = ccx - cx, v2y = ccy - cy, v2z = ccz - cz;
  // edges
  const e0x = v1x - v0x, e0y = v1y - v0y, e0z = v1z - v0z;
  const e1x = v2x - v1x, e1y = v2y - v1y, e1z = v2z - v1z;
  const e2x = v0x - v2x, e2y = v0y - v2y, e2z = v0z - v2z;

  // 9 cross-product axes (edge × box axis): reject if the triangle's projection
  // interval [min,max] clears the box radius r (separating axis found).
  const sep = (p0: number, p1: number, r: number): boolean =>
    Math.min(p0, p1) > r || Math.max(p0, p1) < -r;

  // a00..a02: e0
  let p0: number, p1: number, rad: number;
  // e0 x X = (0, -e0z, e0y)
  p0 = e0z * v0y - e0y * v0z;
  p1 = e0z * v2y - e0y * v2z;
  rad = hy * Math.abs(e0z) + hz * Math.abs(e0y);
  if (sep(p0, p1, rad)) return false;
  // e0 x Y = (e0z, 0, -e0x)
  p0 = -e0z * v0x + e0x * v0z;
  p1 = -e0z * v2x + e0x * v2z;
  rad = hx * Math.abs(e0z) + hz * Math.abs(e0x);
  if (sep(p0, p1, rad)) return false;
  // e0 x Z = (-e0y, e0x, 0)
  p0 = e0y * v1x - e0x * v1y;
  p1 = e0y * v2x - e0x * v2y;
  rad = hx * Math.abs(e0y) + hy * Math.abs(e0x);
  if (sep(p0, p1, rad)) return false;

  // e1
  p0 = e1z * v0y - e1y * v0z;
  p1 = e1z * v2y - e1y * v2z;
  rad = hy * Math.abs(e1z) + hz * Math.abs(e1y);
  if (sep(p0, p1, rad)) return false;
  p0 = -e1z * v0x + e1x * v0z;
  p1 = -e1z * v2x + e1x * v2z;
  rad = hx * Math.abs(e1z) + hz * Math.abs(e1x);
  if (sep(p0, p1, rad)) return false;
  p0 = e1y * v0x - e1x * v0y;
  p1 = e1y * v1x - e1x * v1y;
  rad = hx * Math.abs(e1y) + hy * Math.abs(e1x);
  if (sep(p0, p1, rad)) return false;

  // e2
  p0 = e2z * v0y - e2y * v0z;
  p1 = e2z * v1y - e2y * v1z;
  rad = hy * Math.abs(e2z) + hz * Math.abs(e2y);
  if (sep(p0, p1, rad)) return false;
  p0 = -e2z * v0x + e2x * v0z;
  p1 = -e2z * v1x + e2x * v1z;
  rad = hx * Math.abs(e2z) + hz * Math.abs(e2x);
  if (sep(p0, p1, rad)) return false;
  p0 = e2y * v0x - e2x * v0y;
  p1 = e2y * v1x - e2x * v1y;
  rad = hx * Math.abs(e2y) + hy * Math.abs(e2x);
  if (sep(p0, p1, rad)) return false;

  // 3 box-face axes: triangle AABB vs box
  if (Math.min(v0x, v1x, v2x) > hx || Math.max(v0x, v1x, v2x) < -hx) return false;
  if (Math.min(v0y, v1y, v2y) > hy || Math.max(v0y, v1y, v2y) < -hy) return false;
  if (Math.min(v0z, v1z, v2z) > hz || Math.max(v0z, v1z, v2z) < -hz) return false;

  // triangle-plane axis
  const nx = e0y * e1z - e0z * e1y;
  const ny = e0z * e1x - e0x * e1z;
  const nz = e0x * e1y - e0y * e1x;
  const d = nx * v0x + ny * v0y + nz * v0z;
  const r = hx * Math.abs(nx) + hy * Math.abs(ny) + hz * Math.abs(nz);
  return Math.abs(d) <= r;
}

// scratch accumulators per cell (parallel flat arrays indexed by cell linear index)
interface CellAccum {
  cov: Float32Array;   // Σ coverage
  nx: Float32Array;    // Σ normal.x · cov
  ny: Float32Array;
  nz: Float32Array;
  cr: Float32Array;    // Σ color.r · cov
  cg: Float32Array;
  cb: Float32Array;
}

// C (2026-07-04 memory arc): PERSISTENT per-worker cell accumulator, reused across crown
// jobs. Was 7×Float32Array(cellTotal) freshly allocated per crown (~490 MB at grid 256) —
// with up to 8 workers voxelizing concurrently that churned several GB of transient worker
// heap. One set per module (⇒ per worker; JS is single-threaded so a job fully owns it),
// grown to the largest cellTotal seen and zeroed [0,cellTotal) each call.
let accScratchCells = 0;
let accScratch: CellAccum | null = null;
function getCellAccum(cellTotal: number): CellAccum {
  if (!accScratch || accScratchCells < cellTotal) {
    accScratchCells = cellTotal;
    accScratch = {
      cov: new Float32Array(cellTotal),
      nx: new Float32Array(cellTotal),
      ny: new Float32Array(cellTotal),
      nz: new Float32Array(cellTotal),
      cr: new Float32Array(cellTotal),
      cg: new Float32Array(cellTotal),
      cb: new Float32Array(cellTotal),
    };
  } else {
    const a = accScratch;
    a.cov.fill(0, 0, cellTotal); a.nx.fill(0, 0, cellTotal); a.ny.fill(0, 0, cellTotal);
    a.nz.fill(0, 0, cellTotal); a.cr.fill(0, 0, cellTotal); a.cg.fill(0, 0, cellTotal);
    a.cb.fill(0, 0, cellTotal);
  }
  return accScratch;
}

/** free the persistent cell-accumulator scratch. Crown workers free it implicitly on
 *  terminate; the MAIN-thread inline builders (ForestScene cold, world's rare inline
 *  fallback) call this after their crown loop so the ~490 MB set isn't resident all session. */
export function releaseVoxelizerScratch(): void {
  accScratch = null;
  accScratchCells = 0;
}

/**
 * Voxelize one crown foliage mesh (LOCAL space) into coarse bricks.
 *
 * @param src         the crown foliage ExplicitSource (positions/normals/indices).
 * @param albedo      the per-species foliage tint (0..1 linear) — the density-weighted
 *                    color falls back to this when the source carries no per-vert color
 *                    (our leaves are flat-tinted via matParam, §4.4 / §7.2.6). Used as
 *                    a uniform per-vertex color so the brick mean color is meaningful.
 * @param voxelGridDim effective CELL edge count over the crown (DEFAULT ~180).
 * @param voxlod       voxlod: build the MIP pyramid (levels[]) when true. FALSE (default)
 *                     emits exactly today's single grid — byte-identical, no extra work.
 */
function voxelizeCrown(
  src: ExplicitSource,
  albedo: { r: number; g: number; b: number; hueVar?: number },
  voxelGridDim: number = DEFAULT_VOXEL_GRID_DIM,
  voxlod = false,
): CrownVoxelization {
  const t0 = performance.now();
  const pos = src.positions;
  const nrm = src.normals;
  const idx = src.indices;
  const triCount = idx.length / 3;

  // ---- crown AABB (local) + padded cubic grid ----------------------------
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < pos.length; i += 3) {
    const x = pos[i] as number, y = pos[i + 1] as number, z = pos[i + 2] as number;
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  // cubic grid spanning the largest extent (keeps cells cubic — §3.2 cellWorld math)
  const extX = Math.max(maxX - minX, 1e-3);
  const extY = Math.max(maxY - minY, 1e-3);
  const extZ = Math.max(maxZ - minZ, 1e-3);
  const maxExt = Math.max(extX, extY, extZ);
  // round the cell count UP to a whole brick (4) so the brick grid tiles exactly
  const cellEdge = Math.max(BRICK_DIM, Math.ceil(voxelGridDim / BRICK_DIM) * BRICK_DIM);
  const cellSize = maxExt / cellEdge;
  // pad half a cell so boundary geometry isn't clipped, and center the crown
  const pad = cellSize * 0.5;
  const originX = minX - (maxExt - extX) * 0.5 - pad;
  const originY = minY - (maxExt - extY) * 0.5 - pad;
  const originZ = minZ - (maxExt - extZ) * 0.5 - pad;
  // one extra cell of headroom each side absorbs the centering + pad
  const cellGridX = cellEdge + 2;
  const cellGridY = cellEdge + 2;
  const cellGridZ = cellEdge + 2;
  // grow brick grid to cover the padded cell grid
  const brickGridX = Math.ceil(cellGridX / BRICK_DIM);
  const brickGridY = Math.ceil(cellGridY / BRICK_DIM);
  const brickGridZ = Math.ceil(cellGridZ / BRICK_DIM);
  const cgX = brickGridX * BRICK_DIM;
  const cgY = brickGridY * BRICK_DIM;
  const cgZ = brickGridZ * BRICK_DIM;
  const cellTotal = cgX * cgY * cgZ;

  const acc = getCellAccum(cellTotal);
  const cellLin = (cx: number, cy: number, cz: number): number =>
    cx + cy * cgX + cz * cgX * cgY;

  // ---- rasterize each triangle into the cell grid (SAT + supersample) -----
  const half = cellSize * 0.5;
  const S = COVERAGE_SUPERSAMPLE;
  const subW = 1 / (S * S * S); // per sub-sample coverage weight
  let cellsTouched = 0;
  for (let t = 0; t < triCount; t++) {
    const i0 = (idx[t * 3] as number) * 3;
    const i1 = (idx[t * 3 + 1] as number) * 3;
    const i2 = (idx[t * 3 + 2] as number) * 3;
    const ax = pos[i0] as number, ay = pos[i0 + 1] as number, az = pos[i0 + 2] as number;
    const bx = pos[i1] as number, by = pos[i1 + 1] as number, bz = pos[i1 + 2] as number;
    const ccx = pos[i2] as number, ccy = pos[i2 + 1] as number, ccz = pos[i2 + 2] as number;
    // face normal (geometric) — used for the per-cell normal accumulation. The
    // source vert normals are the BENT crown normals (§5.4 caution); we average the
    // face normal of the covering tris, which equals averaging the bent normals.
    const fnx0 = nrm[i0] as number, fny0 = nrm[i0 + 1] as number, fnz0 = nrm[i0 + 2] as number;
    const fnx1 = nrm[i1] as number, fny1 = nrm[i1 + 1] as number, fnz1 = nrm[i1 + 2] as number;
    const fnx2 = nrm[i2] as number, fny2 = nrm[i2 + 1] as number, fnz2 = nrm[i2 + 2] as number;
    let mnx = fnx0 + fnx1 + fnx2;
    let mny = fny0 + fny1 + fny2;
    let mnz = fnz0 + fnz1 + fnz2;
    const ml = Math.hypot(mnx, mny, mnz) || 1;
    mnx /= ml; mny /= ml; mnz /= ml;

    // tri AABB → cell range (clamped to grid)
    const tMinX = Math.min(ax, bx, ccx), tMaxX = Math.max(ax, bx, ccx);
    const tMinY = Math.min(ay, by, ccy), tMaxY = Math.max(ay, by, ccy);
    const tMinZ = Math.min(az, bz, ccz), tMaxZ = Math.max(az, bz, ccz);
    const c0x = Math.max(0, Math.floor((tMinX - originX) / cellSize) - 1);
    const c1x = Math.min(cgX - 1, Math.floor((tMaxX - originX) / cellSize) + 1);
    const c0y = Math.max(0, Math.floor((tMinY - originY) / cellSize) - 1);
    const c1y = Math.min(cgY - 1, Math.floor((tMaxY - originY) / cellSize) + 1);
    const c0z = Math.max(0, Math.floor((tMinZ - originZ) / cellSize) - 1);
    const c1z = Math.min(cgZ - 1, Math.floor((tMaxZ - originZ) / cellSize) + 1);

    for (let cz = c0z; cz <= c1z; cz++) {
      const ccz0 = originZ + (cz + 0.5) * cellSize;
      for (let cy = c0y; cy <= c1y; cy++) {
        const ccy0 = originY + (cy + 0.5) * cellSize;
        for (let cx = c0x; cx <= c1x; cx++) {
          const ccx0 = originX + (cx + 0.5) * cellSize;
          // conservative gate: does the triangle touch this cell at all?
          if (!triBoxOverlap(ccx0, ccy0, ccz0, half, half, half, ax, ay, az, bx, by, bz, ccx, ccy, ccz)) {
            continue;
          }
          // fractional coverage via S³ sub-cells, each tested as a tiny box
          const subHalf = half / S;
          let cov = 0;
          for (let sz = 0; sz < S; sz++) {
            const scz = ccz0 + (sz - (S - 1) * 0.5) * (cellSize / S);
            for (let sy = 0; sy < S; sy++) {
              const scy = ccy0 + (sy - (S - 1) * 0.5) * (cellSize / S);
              for (let sx = 0; sx < S; sx++) {
                const scx = ccx0 + (sx - (S - 1) * 0.5) * (cellSize / S);
                if (triBoxOverlap(scx, scy, scz, subHalf, subHalf, subHalf, ax, ay, az, bx, by, bz, ccx, ccy, ccz)) {
                  cov += subW;
                }
              }
            }
          }
          if (cov <= 0) continue;
          const li = cellLin(cx, cy, cz);
          if (acc.cov[li] === 0) cellsTouched++;
          acc.cov[li] = (acc.cov[li] as number) + cov;
          acc.nx[li] = (acc.nx[li] as number) + mnx * cov;
          acc.ny[li] = (acc.ny[li] as number) + mny * cov;
          acc.nz[li] = (acc.nz[li] as number) + mnz * cov;
          acc.cr[li] = (acc.cr[li] as number) + albedo.r * cov;
          acc.cg[li] = (acc.cg[li] as number) + albedo.g * cov;
          acc.cb[li] = (acc.cb[li] as number) + albedo.b * cov;
        }
      }
    }
  }

  // ---- SUPPORT-HYSTERESIS EROSION (2026-07-09 crown de-fattening) ---------
  // Decide per-cell occupancy on the FULL crown cell grid BEFORE brick packing. Face-neighbour
  // lookups cross brick boundaries, so this MUST run on the grid (cellLin is global), not
  // per-brick. Rule (constants OCC_SOLID / OCC_GRAZE / OCC_SUPPORT_MIN above):
  //   cov ≥ OCC_SOLID                → SOLID   (kept — the plane-crossing main shape)
  //   OCC_GRAZE ≤ cov < OCC_SOLID    → kept IFF ≥ OCC_SUPPORT_MIN solid face-neighbours
  //                                    (rim cells hugging the surface survive; lone specks erode)
  //   0 < cov < OCC_GRAZE            → dropped (pure "barely got a hit" graze)
  // "SOLID" is read from the coverage SNAPSHOT acc.cov (never mutated), so every borderline
  // cell sees the SIMULTANEOUS solid state — no order-dependent cascade that could eat past the
  // true surface. Density/normal/colour accumulation stays UNCHANGED (word4 density is still the
  // true mean coverage over ALL covered cells — the erosion only clears occupancy BITS). A cell
  // whose acc.cov > 1 (many tris) is solid (OCC_SOLID ≤ 0.95 < the clamp). keptMask drives the
  // brick occLo/occHi below; a brick with zero kept cells is emitted EMPTY (density 0, skipped).
  const OCC_SOLID = OCC_COVERAGE_THRESHOLD;
  const keptMask = new Uint8Array(cellTotal);
  let cellsSolid = 0, cellsSupported = 0, cellsEroded = 0, cellsDropped = 0;
  const solidAt = (x: number, y: number, z: number): boolean =>
    x >= 0 && x < cgX && y >= 0 && y < cgY && z >= 0 && z < cgZ &&
    (acc.cov[cellLin(x, y, z)] as number) >= OCC_SOLID;
  for (let cz = 0; cz < cgZ; cz++) {
    for (let cy = 0; cy < cgY; cy++) {
      for (let cx = 0; cx < cgX; cx++) {
        const li = cellLin(cx, cy, cz);
        const cov = acc.cov[li] as number;
        if (cov <= 0) continue;
        if (cov >= OCC_SOLID) { keptMask[li] = 1; cellsSolid++; continue; }
        if (cov < OCC_GRAZE) { cellsDropped++; continue; }
        // borderline [OCC_GRAZE, OCC_SOLID): keep iff supported by enough solid neighbours
        let solidN = 0;
        if (solidAt(cx - 1, cy, cz)) solidN++;
        if (solidAt(cx + 1, cy, cz)) solidN++;
        if (solidAt(cx, cy - 1, cz)) solidN++;
        if (solidAt(cx, cy + 1, cz)) solidN++;
        if (solidAt(cx, cy, cz - 1)) solidN++;
        if (solidAt(cx, cy, cz + 1)) solidN++;
        if (solidN >= OCC_SUPPORT_MIN) { keptMask[li] = 1; cellsSupported++; }
        else cellsEroded++;
      }
    }
  }

  // ---- aggregate cells → bricks (the COARSE one-sample-per-brick step) ----
  // erosion census (?voxocc calibration): logs the kept/eroded/dropped split so the crown
  // de-fattening is measurable at boot. The FIRST crown also prints the full cell-coverage
  // percentile histogram; the next OCC_CENSUS_CAP crowns print the compact per-species split
  // (one prepareVoxelCrown call PER SPECIES POOL ⇒ these lines are the per-species stats).
  // NOTE the 2026-07-02 "sweeps a no-op" claim is STALE (the occupancy-gated raster now renders
  // per-cell bits — see OCC_SOLID doc); this census is how the effect is now read.
  if (occCensusCount < OCC_CENSUS_CAP) {
    occCensusCount++;
    const covered = cellsSolid + cellsSupported + cellsEroded + cellsDropped;
    const kept = cellsSolid + cellsSupported;
    const frac = (a: number): string => (covered > 0 ? (a / covered).toFixed(2) : '0');
    if (!occHistoLogged) {
      occHistoLogged = true;
      const nz: number[] = [];
      for (let i = 0; i < cellTotal; i++) {
        const c = acc.cov[i] as number;
        if (c > 0) nz.push(Math.min(1, c));
      }
      nz.sort((a, b) => a - b);
      const pct = (p: number): string =>
        nz.length ? (nz[Math.min(nz.length - 1, Math.floor(p * nz.length))] as number).toFixed(3) : 'n/a';
      // eslint-disable-next-line no-console
      console.log(
        `[voxocc] first-crown cell coverage: n=${nz.length} p10=${pct(0.1)} p25=${pct(0.25)} ` +
          `p50=${pct(0.5)} p75=${pct(0.75)} p90=${pct(0.9)} solid=${OCC_SOLID} graze=${OCC_GRAZE}`,
      );
    }
    // eslint-disable-next-line no-console
    console.log(
      `[voxocc] crown#${occCensusCount} tris=${triCount} cells=${covered} ` +
        `kept=${kept}(${frac(kept)}) solid=${cellsSolid}(${frac(cellsSolid)}) ` +
        `supported=${cellsSupported}(${frac(cellsSupported)}) eroded=${cellsEroded}(${frac(cellsEroded)}) ` +
        `dropped=${cellsDropped}(${frac(cellsDropped)})`,
    );
  }
  const bricks: BrickCPU[] = [];
  const occupied: number[] = [];
  let densitySum = 0;
  // per-brick LOCAL-space size/half-extent (§6.2): every brick is a BRICK_DIM-cell cube,
  // so its world edge is constant; the raster paints each brick's OWN footprint (not the
  // whole block) so it MUST know each brick's center+half — stored in the brick record.
  const brickWorld = BRICK_DIM * cellSize;
  const brickHalf = brickWorld * 0.5;
  for (let bz = 0; bz < brickGridZ; bz++) {
    for (let by = 0; by < brickGridY; by++) {
      for (let bx = 0; bx < brickGridX; bx++) {
        let occLo = 0, occHi = 0;
        let bcov = 0;     // Σ over the 64 cells of cell coverage (clamped 0..1)
        let bnx = 0, bny = 0, bnz = 0;
        let bcr = 0, bcg = 0, bcb = 0;
        let bcolW = 0;
        for (let lz = 0; lz < BRICK_DIM; lz++) {
          const cz = bz * BRICK_DIM + lz;
          for (let ly = 0; ly < BRICK_DIM; ly++) {
            const cy = by * BRICK_DIM + ly;
            for (let lx = 0; lx < BRICK_DIM; lx++) {
              const cx = bx * BRICK_DIM + lx;
              const li = cellLin(cx, cy, cz);
              // a cell may have been hit by several tris ⇒ clamp coverage to 1
              const cellCov = Math.min(1, acc.cov[li] as number);
              if (cellCov <= 0) continue;
              // occupancy bit follows the erosion pass (keptMask), NOT the raw threshold — a
              // graze speck that survived accumulation may have been eroded above.
              if (keptMask[li]) {
                const cell = brickCellIndex(lx, ly, lz);
                if (cell < 32) occLo |= (1 << cell);
                else occHi |= (1 << (cell - 32));
              }
              bcov += cellCov;
              // normal/color accumulators are already coverage-weighted Σ; weight by
              // the cell's TOTAL Σcov so a cell hit by many tris dominates correctly
              const w = acc.cov[li] as number;
              bnx += acc.nx[li] as number;
              bny += acc.ny[li] as number;
              bnz += acc.nz[li] as number;
              bcr += acc.cr[li] as number;
              bcg += acc.cg[li] as number;
              bcb += acc.cb[li] as number;
              bcolW += w;
            }
          }
        }
        const bi = bricks.length;
        // brick LOCAL-space center (crown-local): the min corner is origin + brick·brickWorld;
        // center = + half a brick. This is the per-brick footprint origin the raster projects.
        const bCenter: [number, number, number] = [
          originX + (bx + 0.5) * brickWorld,
          originY + (by + 0.5) * brickWorld,
          originZ + (bz + 0.5) * brickWorld,
        ];
        if (bcov <= 0 || bcolW <= 0 || (occLo === 0 && occHi === 0)) {
          // empty brick — still emit a record so the grid addresses linearly, but
          // mark it density 0 (callers skip via `occupied`). ALSO empty when the erosion
          // pass cleared every occupancy bit (zero-bit brick): it would paint nothing under
          // the occupancy-gated raster, so don't emit it to `occupied[]`/the pyramid — the
          // downsample keys off density>0, so this thinning propagates to the coarse levels.
          bricks.push({
            occLo: 0, occHi: 0, normal: [0, 1, 0], spread: 0, albedo: [0, 0, 0], density: 0,
            center: bCenter, half: brickHalf,
          });
          continue;
        }
        // brick MEAN normal (bent) = normalize of the coverage-weighted sum
        const nlen = Math.hypot(bnx, bny, bnz);
        const nrmLen = nlen / bcolW; // mean-vector length in [0,1] → coherence
        const normal: [number, number, number] = nlen > 1e-8
          ? [bnx / nlen, bny / nlen, bnz / nlen]
          : [0, 1, 0];
        // spread = 1 − |mean| (0 = all normals coherent, →1 = isotropic). RAW proxy
        // (§4.3 word3) — the mean-normal shade path ignores it; SGGX fallback reads it.
        const spread = Math.max(0, Math.min(1, 1 - nrmLen));
        // density = mean cell coverage over the brick's 64 cells (Σcov / 64, §5.4.3)
        const density = Math.min(1, bcov / (BRICK_DIM * BRICK_DIM * BRICK_DIM));
        const color: [number, number, number] = [bcr / bcolW, bcg / bcolW, bcb / bcolW];
        bricks.push({ occLo, occHi, normal, spread, albedo: color, density, center: bCenter, half: brickHalf });
        occupied.push(bi);
        densitySum += density;
      }
    }
  }

  // BAKED within-crown LOOK (2026-07-03, user report "far trees blend into one mush"):
  // the L0 albedo above is the FLAT species tint (zero per-brick variation), while the
  // MESH leaf path gets per-leaf hueVar jitter + crown-depth AO (vdata.w·0.8+0.2) at
  // runtime — so the voxel band loses exactly those two channels and distant crowns
  // read as one flat tone. Bake the SAME statistics into the bricks (free at runtime;
  // the dominant-child pyramid albedo then carries them to every coarse level).
  bakeCrownBrickLook(bricks, occupied, albedo.hueVar ?? 0);

  // voxlod (G1): build the MIP pyramid ONLY when asked. levels[0] is the finest grid
  // (the bricks/occupied just produced); coarser levels 2x-downsample the brick grid.
  // voxlod=0 leaves `levels` undefined => the single-resolution path is byte-identical.
  const levels = voxlod
    ? buildVoxelPyramid(
        bricks,
        { x: brickGridX, y: brickGridY, z: brickGridZ },
        cellSize,
        [originX, originY, originZ],
        { blobNormals: true },
      )
    : undefined;

  const voxelizeMs = performance.now() - t0;
  return {
    bricks,
    occupied,
    brickGrid: { x: brickGridX, y: brickGridY, z: brickGridZ },
    cellGrid: { x: cgX, y: cgY, z: cgZ },
    origin: [originX, originY, originZ],
    cellSize,
    levels,
    stats: {
      triangles: triCount,
      cellsTouched,
      occupiedBricks: occupied.length,
      totalBricks: bricks.length,
      meanDensity: occupied.length > 0 ? densitySum / occupied.length : 0,
      voxelizeMs,
    },
  };
}

// ---------------------------------------------------------------------------
// voxlod — MIP pyramid (G1 voxlod=1 only). Re-aggregate the L0 brick grid into
// voxlodLevels() levels (L0 fine … coarsest), each 2x coarser, then build a REAL
// SPATIAL parent->child TREE (NOT the refuted level-uniform anchor-chain).
//
// WHY a tree, not an anchor-chain (iteration-2 root cause): the GPU cull's makeTraverse
// evaluates the cut PER BLOCK using each block's OWN sphere/distance, then either EMITs
// (pOwn<=tau) or DESCENDS by enqueuing ONLY that block's own children. The anchor-chain
// gave the WHOLE finer level to ONE coarse "anchor" block and EMPTY child lists to every
// other coarse block — so a non-anchor coarse block that wanted to refine (pOwn>tau)
// enqueued ZERO children and was not emitted => it VANISHED (a hole), and descent only ever
// flowed through the single anchor. Because per-block pOwn varies across a level (different
// sphere positions/distances), a level does NOT cut/descend together as the anchor-chain
// assumed. validateDagHierarchy passed only because it tests abstract error-cut CONSISTENCY,
// not that every block has a refine path under per-block projection.
//
// THE FIX (this file): every coarse block gets its OWN finer children by SPATIAL
// containment, built BOTTOM-UP so the tree is exact:
//   - L0 (finest) blocks = a deterministic spatial median-bisection partition of the
//     occupied bricks into <=MAX_BRICKS_PER_CLUSTER groups (DagCommon.partitionClusters,
//     the SAME partitioner the mesh DAG uses). L0 ownError = 0 => L0 ALWAYS emits (it has
//     no children; matches the mesh DAG's LOD0 own=0, so point-blank never holes).
//   - level L+1 blocks = formed by GROUPING level-L blocks spatially: each coarse block is
//     the parent of a contiguous spatial run of finer blocks. Each finer block therefore
//     has EXACTLY ONE parent (a strict tree => no double-emit, every block reachable from a
//     root => no orphan => no hole), and every coarse block has children by construction (it
//     was BUILT from them => a coarse block can never want-to-refine-but-have-none).
//   - roots = the coarsest level's blocks. Per-level ownError(L>=1) = anchorL0 * voxlodErrorK() *
//     2^L (L0=0; band-anchored, local-space metres, the mesh-DAG scale), monotone across levels.
// validateDagHierarchy still GATES the result (exact-cut at every threshold + reach + once).
// ---------------------------------------------------------------------------

/** the SOLID half-extent of one occupied brick's footprint (= the per-brick `half`). */
function brickSphereOf(b: BrickCPU): Sphere {
  return { x: b.center[0], y: b.center[1], z: b.center[2], r: b.half * Math.sqrt(3) };
}

/** union (provably-containing) sphere of a set of occupied bricks, given as indices INTO
 *  `occupied` (so callers pass a partitioned subgroup). Empty => a zero sphere. */
function unionBrickSphere(bricks: BrickCPU[], occupied: number[], members: number[]): Sphere {
  let own: Sphere | null = null;
  for (const m of members) {
    const b = bricks[occupied[m] as number] as BrickCPU;
    const s = brickSphereOf(b);
    own = own ? mergeSpheres(own, s) : s;
  }
  return own ?? { x: 0, y: 0, z: 0, r: 0 };
}

/** down-sample one full brick grid by 2x per axis: parent brick (X,Y,Z) aggregates the
 *  <=8 finer bricks (2X+dx, 2Y+dy, 2Z+dz). occupancy OR, density MEAN, normal density-
 *  weighted MEAN, albedo density-weighted MEAN. Empty parents get density 0 (skipped).
 *
 *  voxlod FAR-CHEAPER FIX (iteration-3 root cause): the GPU raster strides each brick's FULL
 *  projected AABB box [center +/- half] per pixel, IGNORING the occupancy bitmask. With `half` =
 *  the whole 2x2x2-child grid cube, a coarse brick whose occupied children only fill a CORNER
 *  overdraws the empty rest of the cube — its screen footprint (bbW*bbH) is (2^L)^2 per level,
 *  so at far the coarsest single block's footprints OVERDRAW the empty regions and the per-pixel
 *  cost ~DOUBLED vs the fine level even though the brick COUNT collapsed (19->1). The count
 *  dropped but the pixel work went up.
 *
 *  THE FIX (build-time, raster untouched per G3): set each coarse brick's `center`/`half` to
 *  the TIGHT cube bounding ONLY its occupied finer bricks' actual footprints ([fb.center +/-
 *  fb.half], which are THEMSELVES already tightened one level down — the union composes up the
 *  pyramid). A corner-occupied coarse brick now projects a corner-sized box, not a full-cube
 *  box, so its bbW*bbH shrinks to cover real geometry only. The footprint STILL contains every
 *  occupied finer brick (it is their union bound) => NO holes; `half` is render-only metadata
 *  (the resolve reads the per-brick mean normal/density, never re-tests occupancy, VoxelBrick.ts)
 *  => NO popping/quality change. Net: far emits a few coarse bricks whose footprints sum
 *  BELOW the fine level's (no empty-corner overdraw) => farCheaper holds. */
function downsampleBrickGrid(
  fine: BrickCPU[],
  fineGrid: { x: number; y: number; z: number },
  coarseCellSize: number,
  origin: [number, number, number],
  /** ellipsoid-normal blend for this OUTPUT level (crowns only): blend the mean-of-
   *  children normal toward (brickCenter − crown centroid) by k — restores the
   *  sun-side/shade-side gradient a distant blob-like crown must have (the plain
   *  mean converges to ~one direction per crown ⇒ flat, directionless far shading). */
  blob?: { cx: number; cy: number; cz: number; k: number },
  /** FARTILES ONLY (2026-07-04 axis-stripe fix): keep `half = coarseHalf` (full grid
   *  pitch) instead of the tight occupied-children bound. On the GLOBAL fartile grid
   *  every brick in a depth column shares the same screen column when viewed along a
   *  world axis, so tight-half gaps ALIGN in depth and read as coherent vertical
   *  see-through stripes (invisible at oblique yaws — bboxes stagger and overlap).
   *  Full-pitch halves tile adjacent bricks edge-to-edge; the occupancy re-bin below
   *  still carves see-through where children are absent. Crowns keep the tight bound
   *  (isolated blobs — the shrink IS their oversized-square fix). */
  fullPitchHalf?: boolean,
): { bricks: BrickCPU[]; grid: { x: number; y: number; z: number } } {
  const gx = Math.max(1, Math.ceil(fineGrid.x / 2));
  const gy = Math.max(1, Math.ceil(fineGrid.y / 2));
  const gz = Math.max(1, Math.ceil(fineGrid.z / 2));
  const coarseWorld = BRICK_DIM * coarseCellSize; // = 2 * fine brick world
  const fineLin = (x: number, y: number, z: number): number =>
    x + y * fineGrid.x + z * fineGrid.x * fineGrid.y;
  const bricks: BrickCPU[] = [];
  for (let cz = 0; cz < gz; cz++) {
    for (let cy = 0; cy < gy; cy++) {
      for (let cx = 0; cx < gx; cx++) {
        let densSum = 0, n = 0;
        let nx = 0, ny = 0, nz = 0;
        let ar = 0, ag = 0, ab = 0, aw = 0;
        // REPRESENTATIVE-child albedo (2026-07-03, user report "far tree ≈ one color"):
        // the density-weighted MEAN albedo re-averages every level, so within-crown color
        // variance dies exponentially up the pyramid and a distant crown flattens to a
        // single tone. Track the DOMINANT child (max density = what you'd actually see in
        // this cell) and inherit ITS albedo — real leaf-clump shades compose up the whole
        // ladder (retained DETAIL, not injected noise). Normals keep the weighted mean
        // (they vary systematically across the crown surface, so N·L survives averaging).
        let repW = -1;
        let repA: [number, number, number] | null = null;
        // TIGHT FOOTPRINT bound: union of the occupied finer bricks' actual [center +/- half]
        // boxes (NOT the full grid cube). Stays empty (±Inf) until a child contributes.
        let fMinX = Infinity, fMinY = Infinity, fMinZ = Infinity;
        let fMaxX = -Infinity, fMaxY = -Infinity, fMaxZ = -Infinity;
        // collect each occupied child's actual world box so the coarse OCCUPANCY can be
        // RE-BINNED into the coarse brick's TIGHT [center ± half] frame below (the far-cheaper
        // fix). The raster gates each footprint pixel by this occupancy bitmask, so it MUST be
        // expressed in the SAME cube the raster projects (= [center ± half], NOT the full grid
        // cube the children's raw occLo/occHi bits address). A simple OR of the children bits
        // would describe the full coarse cube while center/half are tightened — a frame mismatch.
        const childBoxes: { x0: number; y0: number; z0: number; x1: number; y1: number; z1: number }[] = [];
        for (let dz = 0; dz < 2; dz++) {
          const fz = cz * 2 + dz; if (fz >= fineGrid.z) continue;
          for (let dy = 0; dy < 2; dy++) {
            const fy = cy * 2 + dy; if (fy >= fineGrid.y) continue;
            for (let dx = 0; dx < 2; dx++) {
              const fx = cx * 2 + dx; if (fx >= fineGrid.x) continue;
              const fb = fine[fineLin(fx, fy, fz)];
              if (!fb || fb.density <= 0) continue;
              const w = fb.density;
              densSum += fb.density; n++;
              nx += fb.normal[0] * w; ny += fb.normal[1] * w; nz += fb.normal[2] * w;
              ar += fb.albedo[0] * w; ag += fb.albedo[1] * w; ab += fb.albedo[2] * w; aw += w;
              if (w > repW) { repW = w; repA = fb.albedo; }
              // grow the tight footprint by this child's actual (already-tight) box
              const bx0 = fb.center[0] - fb.half, bx1 = fb.center[0] + fb.half;
              const by0 = fb.center[1] - fb.half, by1 = fb.center[1] + fb.half;
              const bz0 = fb.center[2] - fb.half, bz1 = fb.center[2] + fb.half;
              fMinX = Math.min(fMinX, bx0); fMaxX = Math.max(fMaxX, bx1);
              fMinY = Math.min(fMinY, by0); fMaxY = Math.max(fMaxY, by1);
              fMinZ = Math.min(fMinZ, bz0); fMaxZ = Math.max(fMaxZ, bz1);
              childBoxes.push({ x0: bx0, y0: by0, z0: bz0, x1: bx1, y1: by1, z1: bz1 });
            }
          }
        }
        // GRID center (full-cube) is the fallback for an EMPTY cell so the grid still
        // addresses linearly; an occupied cell uses the TIGHT center/half from its children.
        const gridCenter: [number, number, number] = [
          origin[0] + (cx + 0.5) * coarseWorld,
          origin[1] + (cy + 0.5) * coarseWorld,
          origin[2] + (cz + 0.5) * coarseWorld,
        ];
        if (n === 0 || aw <= 0) {
          bricks.push({ occLo: 0, occHi: 0, normal: [0, 1, 0], spread: 0, albedo: [0, 0, 0], density: 0, center: gridCenter, half: coarseWorld * 0.5 });
          continue;
        }
        const nlen = Math.hypot(nx, ny, nz);
        const normal: [number, number, number] = nlen > 1e-8 ? [nx / nlen, ny / nlen, nz / nlen] : [0, 1, 0];
        if (blob) {
          // ellipsoid blend (see param doc): outward direction from the crown centroid
          // through THIS brick's grid center, mixed in by k (grows with level).
          const dx = origin[0] + (cx + 0.5) * coarseWorld - blob.cx;
          const dy = origin[1] + (cy + 0.5) * coarseWorld - blob.cy;
          const dz = origin[2] + (cz + 0.5) * coarseWorld - blob.cz;
          const dl = Math.hypot(dx, dy, dz);
          if (dl > 1e-6) {
            const bx2 = normal[0] * (1 - blob.k) + (dx / dl) * blob.k;
            const by2 = normal[1] * (1 - blob.k) + (dy / dl) * blob.k;
            const bz2 = normal[2] * (1 - blob.k) + (dz / dl) * blob.k;
            const bl = Math.hypot(bx2, by2, bz2);
            if (bl > 1e-6) { normal[0] = bx2 / bl; normal[1] = by2 / bl; normal[2] = bz2 / bl; }
          }
        }
        const spread = Math.max(0, Math.min(1, 1 - nlen / aw));
        // density MEAN over the contributing finer bricks (a coarse brick that subsumes
        // a sparse smear stays sparse — keeps the see-through look at distance, §5.4.3).
        const density = Math.min(1, densSum / n);
        // TIGHT cube over the occupied-children union, but CLAMPED INTO this brick's full grid
        // cube [gridMin, gridMax] first. The clamp is the correctness fix: a SCALAR `half` =
        // max-axis-span/2 over the RAW union can EXCEED the full-cube half when occupied children
        // sit in opposite corners (the span ≈ the cube diagonal). Clamping the union box to the
        // grid cube before taking the max span guarantees half ≤ coarseHalf — the footprint only
        // ever SHRINKS, never grows past the full cube (no footprint blow-up, still contains all kids).
        const coarseHalf = coarseWorld * 0.5;
        const gMinX = gridCenter[0] - coarseHalf, gMaxX = gridCenter[0] + coarseHalf;
        const gMinY = gridCenter[1] - coarseHalf, gMaxY = gridCenter[1] + coarseHalf;
        const gMinZ = gridCenter[2] - coarseHalf, gMaxZ = gridCenter[2] + coarseHalf;
        const tMinX = Math.max(fMinX, gMinX), tMaxX = Math.min(fMaxX, gMaxX);
        const tMinY = Math.max(fMinY, gMinY), tMaxY = Math.min(fMaxY, gMaxY);
        const tMinZ = Math.max(fMinZ, gMinZ), tMaxZ = Math.min(fMaxZ, gMaxZ);
        // FIX A (offset): the coarse brick CENTER is the SYMMETRIC grid-cube center `gridCenter`
        // (origin + (c+0.5)*coarseWorld) — tiling EXACTLY like L0 (originX+(bx+0.5)*brickWorld) — NOT
        // the occupancy-centroid (tMin+tMax)*0.5 of the clamped child box, which slides the brick off
        // the trunk axis (the lopsided crown drift). `half` is SYMMETRIZED about gridCenter: the max
        // over axes of the LARGER side-distance from gridCenter out to the clamped child box. The cube
        // [gridCenter ± half] still CONTAINS every clamped child (no holes — each child sits within
        // [tMin,tMax] which is within [gridCenter ± half]) while staying centered. ≤ coarseHalf since
        // tMin/tMax are already clamped into the full grid cube, so it never exceeds the full cube.
        const center: [number, number, number] = gridCenter;
        // fartiles: full-pitch half (see param doc) — adjacent bricks tile edge-to-edge,
        // no aligned axis gaps; occupancy (re-binned into the same full cube) still carves.
        const half = fullPitchHalf
          ? coarseHalf
          : Math.min(
              coarseHalf,
              Math.max(
                Math.max(gridCenter[0] - tMinX, tMaxX - gridCenter[0]),
                Math.max(gridCenter[1] - tMinY, tMaxY - gridCenter[1]),
                Math.max(gridCenter[2] - tMinZ, tMaxZ - gridCenter[2]),
              ),
            );
        // RE-BIN occupancy into THIS coarse brick's TIGHT [center ± half] cube (the far-cheaper
        // fix). The 4×4×4 occLo/occHi bitmask must describe the SAME cube the raster projects so
        // the raster's per-pixel occupancy gate skips the EMPTY interior between sparse children.
        // A cell is occupied (conservatively, no holes) iff any occupied child box overlaps it.
        let occLo = 0, occHi = 0;
        const cMinX = center[0] - half, cMinY = center[1] - half, cMinZ = center[2] - half;
        const occCell = half > 1e-12 ? (2 * half) / BRICK_DIM : 1; // edge of one re-binned sub-cell
        for (const cb of childBoxes) {
          // child box → inclusive sub-cell index range within [center ± half], clamped to [0,4).
          const lx0 = Math.max(0, Math.min(BRICK_DIM - 1, Math.floor((cb.x0 - cMinX) / occCell)));
          const lx1 = Math.max(0, Math.min(BRICK_DIM - 1, Math.floor((cb.x1 - cMinX) / occCell)));
          const ly0 = Math.max(0, Math.min(BRICK_DIM - 1, Math.floor((cb.y0 - cMinY) / occCell)));
          const ly1 = Math.max(0, Math.min(BRICK_DIM - 1, Math.floor((cb.y1 - cMinY) / occCell)));
          const lz0 = Math.max(0, Math.min(BRICK_DIM - 1, Math.floor((cb.z0 - cMinZ) / occCell)));
          const lz1 = Math.max(0, Math.min(BRICK_DIM - 1, Math.floor((cb.z1 - cMinZ) / occCell)));
          for (let lz = lz0; lz <= lz1; lz++)
            for (let ly = ly0; ly <= ly1; ly++)
              for (let lx = lx0; lx <= lx1; lx++) {
                const cell = brickCellIndex(lx, ly, lz);
                if (cell < 32) occLo |= (1 << cell); else occHi |= (1 << (cell - 32));
              }
        }
        // albedo: DOMINANT child, not the weighted mean — mean-of-means decays within-crown
        // color variance by ~½ per level (a distant crown flattens to ONE tone, user report);
        // dominant-child is mode/nearest downsampling for categorical data (leaf-clump shades)
        // and retains the full real palette at every level. Composes up the ladder unchanged.
        const albedo: [number, number, number] = repA ? [repA[0], repA[1], repA[2]] : [ar / aw, ag / aw, ab / aw];
        bricks.push({ occLo, occHi, normal, spread, albedo, density, center, half });
      }
    }
  }
  return { bricks, grid: { x: gx, y: gy, z: gz } };
}

/** voxlod FAR-CHEAPER (iteration-6): SHELL a coarse level's brick grid in place. A brick is
 *  INTERIOR iff it is occupied AND all 6 of its axis face-neighbours in the brick grid are also
 *  occupied — then it is enclosed on every side by opaque neighbour bricks whose projected
 *  footprints occlude it from ANY exterior view, so it can never win a single election (zero
 *  visible pixels) and only adds depth-overdraw. Mutate such bricks to density 0 so they drop out
 *  of `occupied` exactly like an empty brick (no separate path; the partitioner/DAG/validate all
 *  key off density>0). A brick on the grid boundary (a missing neighbour off the grid edge) is
 *  treated as a SURFACE brick (kept) — the crown's outer hull is always retained. Grid-occupancy
 *  is read from a SNAPSHOT taken before any mutation so the test is the simultaneous 6-neighbour
 *  state (no order-dependent cascade that could carve past the true surface). Conservative: it
 *  only ever removes a brick PROVABLY hidden by occupied neighbours on all sides ⇒ no holes, the
 *  silhouette + visible surface are byte-identical, density/normal of the kept surface unchanged
 *  ⇒ no popping. Returns the count removed (for the budget/debug print). NEVER called on L0. */
function shellCoarseBricks(
  bricks: BrickCPU[],
  grid: { x: number; y: number; z: number },
): number {
  const { x: gx, y: gy, z: gz } = grid;
  const lin = (x: number, y: number, z: number): number => x + y * gx + z * gx * gy;
  // SNAPSHOT the occupancy (density>0) so neighbour tests see the pre-shell state.
  const occ = new Uint8Array(bricks.length);
  for (let i = 0; i < bricks.length; i++) occ[i] = (bricks[i] as BrickCPU).density > 0 ? 1 : 0;
  let removed = 0;
  for (let z = 0; z < gz; z++) {
    for (let y = 0; y < gy; y++) {
      for (let x = 0; x < gx; x++) {
        const i = lin(x, y, z);
        if (!occ[i]) continue;
        // a boundary brick (any neighbour off the grid) is SURFACE — keep it (the outer hull).
        if (x === 0 || x === gx - 1 || y === 0 || y === gy - 1 || z === 0 || z === gz - 1) continue;
        // INTERIOR iff all 6 face-neighbours are occupied in the snapshot.
        const enclosed =
          occ[lin(x - 1, y, z)] === 1 && occ[lin(x + 1, y, z)] === 1 &&
          occ[lin(x, y - 1, z)] === 1 && occ[lin(x, y + 1, z)] === 1 &&
          occ[lin(x, y, z - 1)] === 1 && occ[lin(x, y, z + 1)] === 1;
        if (enclosed) {
          (bricks[i] as BrickCPU).density = 0; // drop from `occupied` (invisible interior)
          removed++;
        }
      }
    }
  }
  return removed;
}

/** brick-grid (x,y,z) of an occupied-brick index within a level's brick grid. */
function brickGridXYZ(idx: number, grid: { x: number; y: number; z: number }): [number, number, number] {
  const x = idx % grid.x;
  const y = Math.floor(idx / grid.x) % grid.y;
  const z = Math.floor(idx / (grid.x * grid.y));
  return [x, y, z];
}

/** Partition one level's occupied bricks into spatially-coherent blocks of
 *  <=MAX_BRICKS_PER_CLUSTER via median-axis bisection (DagCommon.partitionClusters — the
 *  SAME partitioner the mesh DAG uses). Each block carries its OWN containing sphere (the
 *  union of its bricks' footprints) and its `members` = the indices INTO `occupied` it owns
 *  (so a caller can map bricks→block and emit a CONTIGUOUS appended brick run via `order`).
 *  Returns { blocks, order } where `order` lists occupied-indices grouped block-by-block —
 *  the append/AABB order so each block's bricks are contiguous in the GPU brick buffer. */
function partitionLevelBlocks(
  bricks: BrickCPU[],
  occupied: number[],
  ownError: number,
): { blocks: VoxelBlock[]; order: number[] } {
  // build per-occupied-brick centroids for the partitioner (it keys on sx/sy/sz)
  const pts = occupied.map((bi) => {
    const b = bricks[bi] as BrickCPU;
    return { sx: b.center[0], sy: b.center[1], sz: b.center[2] };
  });
  const ids = occupied.map((_, i) => i); // indices INTO occupied[]
  const groups = partitionClusters(ids, pts, MAX_BRICKS_PER_CLUSTER);
  const blocks: VoxelBlock[] = [];
  const order: number[] = [];
  for (const grp of groups) {
    const start = order.length;
    for (const m of grp) order.push(m);
    // solidArea = Sum of (2*half)^2 over the block's bricks — the LOCAL-space proxy for the
    // SOLID screen rectangle the raster paints (it projects each brick's full [center +- half]
    // AABB; projected box area ~ (2*half)^2 * projK^2 / d^2, and the projK^2/d^2 factor is
    // common to a coarse block and the fine children it replaces, so it cancels in the ratio).
    let solidArea = 0;
    for (const m of grp) {
      const h = (bricks[occupied[m] as number] as BrickCPU).half;
      solidArea += (2 * h) * (2 * h);
    }
    blocks.push({
      start, count: grp.length,
      own: unionBrickSphere(bricks, occupied, grp),
      ownError,
      solidArea,
      parent: null,
      parentError: Infinity,
      childBlocks: [],
    });
  }
  return { blocks, order };
}

/** Build the crown's voxel MIP pyramid + a SPATIAL parent->child TREE (the iteration-2 fix —
 *  see the header). Each level's occupied bricks are partitioned into spatially-coherent
 *  blocks; level-L blocks are then attached to their level-(L+1) parent by GRID CONTAINMENT
 *  (a finer brick (fx,fy,fz) maps to coarse brick (fx>>1,fy>>1,fz>>1)). Single-parent voting +
 *  a coverage repair guarantee EXACTLY one parent per finer block AND >=1 child per coarse
 *  block, so the GPU per-block cut descends to the WHOLE finer level with no holes/double-emit. */
/** BAKE the mesh-leaf look channels into the L0 bricks (2026-07-03, "far-tree mush").
 *  Post-pass over the occupied bricks of one crown, crown-LOCAL space:
 *
 *  1. HUE JITTER — a deterministic hash on a ~0.75 m CLUMP lattice of the brick center,
 *     scaled by the species hueVar and applied with the EXACT warm/cool palette the
 *     mesh resolve uses per leaf (NaniteResolve fullLeaf: base·[1.18,1.0,0.55]·max(k,0)
 *     + base·[0.7,0.95,1.25]·max(−k,0) + base·(1−|k|)) — so the 60 m seam keeps one
 *     color statistic across representations.
 *  2. CROWN-DEPTH AO — the mesh path multiplies vdata.w·0.8+0.2 (baked crown-depth per
 *     vertex); bricks get the analogue: radial surface factor (interior → dark; mostly
 *     hidden by occupancy, but it darkens inner-surface pixels) × a vertical gradient
 *     (canopy self-shadowing: bottom ~0.78). GTAO fades out by 240 m (aofade), so this
 *     baked term is what keeps distant crowns from reading uniformly BRIGHT.
 *
 *  Bake-time only — zero runtime cost; composes up the pyramid via the dominant-child
 *  albedo. Deterministic (integer hash), so the boot cache stays reproducible. */
function bakeCrownBrickLook(bricks: BrickCPU[], occupied: number[], hueVar: number): void {
  if (occupied.length === 0) return;
  // density-weighted centroid + extents of the OCCUPIED crown
  let cx = 0, cy = 0, cz = 0, wSum = 0;
  let minY = Infinity, maxY = -Infinity;
  for (const oi of occupied) {
    const b = bricks[oi] as BrickCPU;
    const w = Math.max(1e-4, b.density);
    cx += b.center[0] * w; cy += b.center[1] * w; cz += b.center[2] * w; wSum += w;
    if (b.center[1] < minY) minY = b.center[1];
    if (b.center[1] > maxY) maxY = b.center[1];
  }
  cx /= wSum; cy /= wSum; cz /= wSum;
  let maxR = 1e-4;
  for (const oi of occupied) {
    const b = bricks[oi] as BrickCPU;
    const r = Math.hypot(b.center[0] - cx, b.center[1] - cy, b.center[2] - cz);
    if (r > maxR) maxR = r;
  }
  const ySpan = Math.max(1e-4, maxY - minY);
  const CLUMP = 0.75; // hue-jitter lattice (m) — foliage-clump scale, not per-voxel noise
  for (const oi of occupied) {
    const b = bricks[oi] as BrickCPU;
    const a = b.albedo;
    // 1. clump hue jitter (k ∈ [−1,1] × hueVar), mesh-palette warm/cool
    if (hueVar > 0) {
      const ix = Math.floor(b.center[0] / CLUMP) | 0;
      const iy = Math.floor(b.center[1] / CLUMP) | 0;
      const iz = Math.floor(b.center[2] / CLUMP) | 0;
      const h = ((Math.imul(ix, 73856093) ^ Math.imul(iy, 19349663) ^ Math.imul(iz, 83492791)) >>> 0) & 0xffff;
      const k = (h / 0xffff * 2 - 1) * hueVar;
      const kw = Math.max(0, k), kc = Math.max(0, -k), kb = 1 - Math.abs(k);
      const r0 = a[0], g0 = a[1], b0 = a[2];
      a[0] = r0 * 1.18 * kw + r0 * 0.7 * kc + r0 * kb;
      a[1] = g0 * 1.0 * kw + g0 * 0.95 * kc + g0 * kb;
      a[2] = b0 * 0.55 * kw + b0 * 1.25 * kc + b0 * kb;
    }
    // 2. crown-depth AO: radial (surface 1 → deep interior 0.2, mesh's dv.w·0.8+0.2
    //    analogue) × vertical canopy self-shadow (top 1 → bottom 0.78)
    const r = Math.hypot(b.center[0] - cx, b.center[1] - cy, b.center[2] - cz) / maxR;
    const radial = 0.2 + 0.8 * Math.max(0, Math.min(1, (r - 0.25) / 0.6));
    const vertical = 0.78 + 0.22 * (b.center[1] - minY) / ySpan;
    const ao = radial * vertical;
    a[0] *= ao; a[1] *= ao; a[2] *= ao;
  }
}

export function buildVoxelPyramid(
  l0Bricks: BrickCPU[],
  l0Grid: { x: number; y: number; z: number },
  l0CellSize: number,
  origin: [number, number, number] = [0, 0, 0],
  opts?: {
    /** blend coarse-brick normals toward "outward from the crown centroid" as levels
     *  coarsen (a distant crown shades like an ellipsoid: bright sun side, dark shade
     *  side — the mean-of-children normal converges to one direction and kills that).
     *  CROWNS ONLY — a 64 m far-TILE is many trees, not one blob (leave unset there). */
    blobNormals?: boolean;
    /** FARTILES ONLY: coarse bricks keep the full grid-pitch half-extent instead of the
     *  tight occupied-children bound — kills the axis-aligned see-through stripe gaps on
     *  the global tile grid (see downsampleBrickGrid.fullPitchHalf). */
    fullPitchHalves?: boolean;
  },
): VoxelLevel[] {
  const levels: VoxelLevel[] = [];
  // TRUST the passed `origin` — do NOT re-derive it from l0Bricks[0].center. The old
  // "recovery" (origin = b0.center − 0.5·brickWorld) silently CORRUPTED the fartile
  // pyramids (#69, root-caused 2026-07-09): FarTiles.emitTile pads empty slots with a
  // shared EMPTY_BRICK whose center is [0,0,0], so whenever a tile's corner brick was
  // empty the recovered origin was off by ~+30.5 m ⇒ every coarse re-bin clamped its
  // children to sub-cell 0 ⇒ one lit cell layer per coarse brick, phase-locked across
  // the tile grid = the axis-aligned see-through stripes (period = one coarse-brick
  // pitch; healed at oblique yaws; bimodal per tile on corner-brick emptiness). Crowns
  // were immune (their empty bricks carry true centers ⇒ recovery == origin). Node A/B
  // over the shipping splat: corrupted origin −1.455 vs true −32.0; L1 see-through
  // columns 33/44 → 0/44 with the origin trusted. Both callers pass the exact grid
  // origin (grep-verified). Full evidence: docs/tasks/2026-07-09/FARTILE-GAPS-DIAGNOSIS.md.
  const gridOrigin = origin;

  // -- build the level pyramid (bricks + per-level blocks) -------------------
  // L0 ownError = 0 (the finest level ALWAYS emits — it has no children, so a near block can
  // never want-to-refine-but-have-none; matches the mesh DAG's LOD0 own=0). Coarser levels carry
  // the REAL brick-world-size error curCell*BRICK_DIM*0.5*errorK (local-space metres) — see below.
  let curBricks = l0Bricks;
  let curGrid = l0Grid;
  let curCell = l0CellSize;
  // crown centroid for the ellipsoid-normal blend (opts.blobNormals — crowns only)
  let blobC: { cx: number; cy: number; cz: number } | null = null;
  if (opts?.blobNormals) {
    let sx = 0, sy = 0, sz = 0, sw = 0;
    for (const b of l0Bricks) {
      if (b.density <= 0) continue;
      const w = b.density;
      sx += b.center[0] * w; sy += b.center[1] * w; sz += b.center[2] * w; sw += w;
    }
    if (sw > 0) blobC = { cx: sx / sw, cy: sy / sw, cz: sz / sw };
  }
  // per-level: which level-L block each level-L OCCUPIED-brick belongs to (occ-index -> block).
  const blockOfOccByLevel: number[][] = [];
  const N_LEVELS = voxlodLevels();
  const ERR_K = voxlodErrorK();        // ladder MULTIPLIER (default 1; ?voxlodk= sweeps it)
  for (let L = 0; L < N_LEVELS; L++) {
    const occupied: number[] = [];
    for (let i = 0; i < curBricks.length; i++) if ((curBricks[i] as BrickCPU).density > 0) occupied.push(i);
    // FIX B (too-coarse-near): ownError(L) = the REAL local-space brick WORLD HALF-extent at this
    // level, curCell*BRICK_DIM*0.5 (= l0CellSize*BRICK_DIM*0.5 * 2^L, since curCell = l0CellSize*2^L),
    // times errorK. This is projK-INDEPENDENT and dimensionally the SAME unit (object-space metres)
    // the mesh DAG qemErr uses, so the cull's pOwn = projK*A.w*ownError/d selects level L EXACTLY
    // when its brick subtends ~tau px (projK*A.w*brickHalf_L/d <= tau) — fine near, coarsening one
    // octave per distance-doubling. This dissolves the prior build-time-projK anchor (computed from
    // a 1080-height floor decoupled from BOTH the runtime/retina projK AND the L0 brick world-size,
    // which made coarse levels engage far too NEAR — the massive flat squares at ~10 m). L0 = 0
    // (always-emit finest, hole-safe). errorK scales the whole ladder farther (>1) / nearer (<1).
    const ownError = L === 0 ? 0 : curCell * BRICK_DIM * 0.5 * ERR_K;
    const { blocks, order } = partitionLevelBlocks(curBricks, occupied, ownError);
    // reorder `occupied` so each block's bricks are CONTIGUOUS (block.start/count index it);
    // also record, per (reordered) occupied slot, which block owns it.
    const reordered = order.map((m) => occupied[m] as number);
    const blockOfOcc = new Array<number>(reordered.length);
    for (let bi = 0; bi < blocks.length; bi++) {
      const blk = blocks[bi] as VoxelBlock;
      for (let k = 0; k < blk.count; k++) blockOfOcc[blk.start + k] = bi;
    }
    blockOfOccByLevel.push(blockOfOcc);
    levels.push({ level: L, bricks: curBricks, occupied: reordered, brickGrid: curGrid, cellSize: curCell, geomError: ownError, blocks });
    // FIX C (coarseness CAP — a crown must NEVER be one square): stop coarsening once the LARGEST
    // brick-grid axis reaches K_FLOOR bricks, so the coarsest level stays a MULTI-brick shell that
    // still reads as a crown SHAPE (up to ~3×3×3 → a ~26-brick occupancy shell after the Rank-2
    // gate), never collapsing to a single 1×1×1 brick. MAX axis (not all-≤1) so thin/flat crowns
    // still coarsen on their large axes while keeping ≥K_FLOOR bricks there. (occupied≤1 kept as a
    // safety net for degenerate crowns.)
    const K_FLOOR = 3;
    if (L === N_LEVELS - 1 || occupied.length <= 1 || Math.max(curGrid.x, curGrid.y, curGrid.z) <= K_FLOOR) break;
    // ellipsoid-normal strength for the level being PRODUCED (L+1): none at the fine
    // levels (their mean normals still carry real variation), ramping to 0.7 by L3+.
    const next = downsampleBrickGrid(
      curBricks,
      curGrid,
      curCell * 2,
      gridOrigin,
      blobC ? { ...blobC, k: Math.min(0.7, 0.28 * (L + 1)) } : undefined,
      opts?.fullPitchHalves === true,
    );
    // voxlod FAR-CHEAPER (iteration-6): SHELL the coarse grid we just produced — drop bricks fully
    // enclosed by occupied neighbours (invisible interior) so this coarser level renders a ~2-deep
    // SHELL, not a ~10-deep VOLUME ⇒ far depth-overdraw falls below the fine level it replaces
    // (farCheaper) with no silhouette/surface change. `next` always feeds a COARSE level (L+1 ≥ 1);
    // L0 (l0Bricks) never passes through here, so the near crown keeps its full opaque volume.
    if (voxlodShell()) shellCoarseBricks(next.bricks, next.grid);
    curBricks = next.bricks;
    curGrid = next.grid;
    curCell = curCell * 2;
  }

  // -- wire the SPATIAL tree (coarse->fine) + roots --------------------------
  const coarsest = levels.length - 1;
  for (let L = coarsest; L >= 0; L--) {
    const lvl = levels[L] as VoxelLevel;
    if (L === coarsest) {
      for (const blk of lvl.blocks) { blk.parent = null; blk.parentError = Infinity; } // roots
      continue;
    }
    const coarser = levels[L + 1] as VoxelLevel;
    const coarserBlockOfOcc = blockOfOccByLevel[L + 1] as number[];
    // map a coarse brick GRID index -> the coarse occupied-slot (so a finer brick can find its
    // parent coarse BLOCK by grid containment). Only occupied coarse bricks are in the map.
    const coarseGridToOcc = new Map<number, number>();
    for (let oi = 0; oi < coarser.occupied.length; oi++) {
      coarseGridToOcc.set(coarser.occupied[oi] as number, oi);
    }
    const coarseLin = (x: number, y: number, z: number): number =>
      x + y * coarser.brickGrid.x + z * coarser.brickGrid.x * coarser.brickGrid.y;
    // each finer BLOCK votes for the coarse BLOCK that owns the MOST of its bricks' coarse
    // parents (grid containment fx>>1 etc.); ties -> lowest coarse-block index (deterministic).
    // Then attach the finer block to that ONE coarse parent (a strict tree => single-visit).
    for (let fb = 0; fb < lvl.blocks.length; fb++) {
      const blk = lvl.blocks[fb] as VoxelBlock;
      const votes = new Map<number, number>();
      for (let k = 0; k < blk.count; k++) {
        const fineGridIdx = lvl.occupied[blk.start + k] as number;
        const [fx, fy, fz] = brickGridXYZ(fineGridIdx, lvl.brickGrid);
        const cGrid = coarseLin(fx >> 1, fy >> 1, fz >> 1);
        const cOcc = coarseGridToOcc.get(cGrid);
        if (cOcc === undefined) continue; // coarse brick shelled-away as interior (iteration-6) ⇒ no vote from it
        const cBlk = coarserBlockOfOcc[cOcc] as number;
        votes.set(cBlk, (votes.get(cBlk) ?? 0) + 1);
      }
      let best = -1, bestVotes = -1;
      for (const [cBlk, v] of votes) {
        if (v > bestVotes || (v === bestVotes && cBlk < best)) { best = cBlk; bestVotes = v; }
      }
      // EMPTY-VOTE FALLBACK. Pre-shell this was a degenerate single-coarse-block case (best=0).
      // With the iteration-6 SHELL a finer block can legitimately have ALL its bricks map to coarse
      // bricks that were dropped as interior (so coarseGridToOcc misses every one ⇒ no votes). Then
      // pick the SPATIALLY-NEAREST coarse block by `own`-sphere centroid (NOT block 0, which could
      // be across the crown) so the parent still spatially contains the child after mergeSpheres ⇒
      // a coherent tree + a tight parent bound. validateDagHierarchy below still HARD-proves the cut.
      if (best < 0) {
        let nd = Infinity;
        for (let cb = 0; cb < coarser.blocks.length; cb++) {
          const co = (coarser.blocks[cb] as VoxelBlock).own;
          const dx = co.x - blk.own.x, dy = co.y - blk.own.y, dz = co.z - blk.own.z;
          const d = dx * dx + dy * dy + dz * dz;
          if (d < nd) { nd = d; best = cb; }
        }
        if (best < 0) best = 0; // (no coarse blocks at all — cannot happen, the level has ≥1)
      }
      (coarser.blocks[best] as VoxelBlock).childBlocks.push(fb);
    }
    // COVERAGE REPAIR: a coarse block with occupied bricks but ZERO children would, if it ever
    // wants to refine (pOwn>tau), descend to nothing => a hole. Re-home a SPATIALLY-NEAREST
    // finer block to it, STEALING from a donor parent that keeps >=1 child (so we never just
    // shift the hole to the donor). Among finer blocks whose current parent has >=2 children,
    // pick the nearest; only if none qualify do we take from a 1-child donor (then the validate
    // gate would catch any resulting orphan — fail loud, never ship a hole). Rare with spatially
    // coherent partitions, but it makes the >=1-child-per-coarse-block invariant HARD.
    {
      const parentOf = new Int32Array(lvl.blocks.length).fill(-1);
      for (let pb = 0; pb < coarser.blocks.length; pb++) {
        for (const fb of (coarser.blocks[pb] as VoxelBlock).childBlocks) parentOf[fb] = pb;
      }
      for (let cb = 0; cb < coarser.blocks.length; cb++) {
        const cblk = coarser.blocks[cb] as VoxelBlock;
        if (cblk.childBlocks.length > 0) continue;
        let pick = -1, pickD = Infinity, pickSafe = false;
        for (let fb = 0; fb < lvl.blocks.length; fb++) {
          const donor = parentOf[fb] as number;
          if (donor < 0) continue; // unparented (shouldn't happen) — skip
          const safe = (coarser.blocks[donor] as VoxelBlock).childBlocks.length >= 2;
          const fblk = lvl.blocks[fb] as VoxelBlock;
          const dx = fblk.own.x - cblk.own.x, dy = fblk.own.y - cblk.own.y, dz = fblk.own.z - cblk.own.z;
          const d = dx * dx + dy * dy + dz * dz;
          // prefer a SAFE donor (>=2 kids); within the same safety class pick the nearest.
          if ((safe && !pickSafe) || ((safe === pickSafe) && d < pickD)) { pick = fb; pickD = d; pickSafe = safe; }
        }
        if (pick < 0) continue; // (no finer blocks at all) — validate gate will catch if needed
        const donor = parentOf[pick] as number;
        const donorKids = (coarser.blocks[donor] as VoxelBlock).childBlocks;
        const j = donorKids.indexOf(pick);
        if (j >= 0) donorKids.splice(j, 1);
        cblk.childBlocks.push(pick);
        parentOf[pick] = cb;
      }
      // POST-REPAIR GUARD: the single-pass steal above can empty an already-processed donor
      // (it steals the donor's last child after that donor was repaired), leaving a childless
      // coarse block that refines to nothing. Reviewer-verified NOT a hole (the region renders
      // via the finer blocks' real parents, and validateVoxelPyramid still HARD-gates reachability),
      // but the donor-emptying path is non-obvious — warn LOUD so it surfaces rather than relying
      // purely on the argued invariant. Non-throwing: a throw here would block build on a benign case.
      for (let cb = 0; cb < coarser.blocks.length; cb++) {
        if ((coarser.blocks[cb] as VoxelBlock).childBlocks.length === 0) {
          // eslint-disable-next-line no-console
          console.warn(`[voxlod] coarse block L${L + 1} #${cb} has no children after coverage-repair — refine would descend to nothing; validateVoxelPyramid gates reachability`);
        }
      }
    }
    // parent SPHERE only here (provably contains child). parentERROR is set by the bottom-up
    // sparsity pass below — it must read the parent's PER-BLOCK effective ownError, which is not
    // finalized until that pass runs (coarse blocks get a sparsity-raised error), so deferring
    // it keeps child.parentError == parent.ownError exactly (the crack-safe-cut invariant).
    for (let cb = 0; cb < coarser.blocks.length; cb++) {
      const cblk = coarser.blocks[cb] as VoxelBlock;
      for (const fb of cblk.childBlocks) {
        const blk = lvl.blocks[fb] as VoxelBlock;
        blk.parent = mergeSpheres(cblk.own, blk.own); // guarantees parent contains own
      }
    }
  }

  // -- PER-BLOCK SPARSITY MULTIPLIER on ownError (the iteration-2 far-cheaper / no-pop fix) ----
  // The raster paints each brick's full [center +- half] AABB SOLID (occupancy ignored, G3 keeps
  // it that way), so coarsening a SPARSE block (its solid box fills the see-through gaps between
  // scattered children) paints MORE pixels than the fine children it replaces — coarsening is a
  // pixel LOSS there. We RAISE such a block's ownError so the screen-error cut keeps DESCENDING
  // (stays fine) until the crown is far enough that even the sparse coarse box is screen-tiny.
  // A DENSE block (solid box ~= its children's coverage) keeps multiplier ~1 and coarsens early.
  //
  // Processed FINEST->COARSEST so each parent can floor its error strictly ABOVE its children's
  // (the crack-safe monotone-cut invariant child.ownError < parent.ownError). L0 stays 0 (always
  // emits, no parent). The cut machinery (makeTraverse) is UNCHANGED — it just reads ownError.
  const SPARSE_K = voxlodSparseK();
  const MONO_EPS = 1e-3; // parent strictly above max child (relative bump; metres are O(0.1..10))
  for (let L = 1; L < levels.length; L++) {
    const lvl = levels[L] as VoxelLevel;
    const finer = levels[L - 1] as VoxelLevel;
    const base = lvl.geomError; // = band-anchored ownError anchorL0*ERR_K*2^L (L0=0), NOT ERR_K*cellSize
    for (const blk of lvl.blocks) {
      // childArea = summed SOLID area of the fine blocks this coarse block replaces.
      let childArea = 0;
      let maxChildOwn = 0;
      for (const fb of blk.childBlocks) {
        const c = finer.blocks[fb] as VoxelBlock;
        childArea += c.solidArea;
        if (c.ownError > maxChildOwn) maxChildOwn = c.ownError;
      }
      // sparsity = how much MORE solid area coarsening paints. <=1 (dense) => no deferral;
      // >1 (sparse) => defer, capped at SPARSE_K. childArea 0 (no kids — cannot happen post
      // coverage-repair) => treat as dense (sp=1) so a stray childless block still coarsens.
      const sp = childArea > 1e-9 ? Math.max(1, Math.min(SPARSE_K, blk.solidArea / childArea)) : 1;
      // floor strictly above every child so the per-block monotone cut stays crack-free even
      // when a sparse CHILD was itself deferred above this (dense) parent's base*sp.
      blk.ownError = Math.max(base * sp, maxChildOwn * (1 + MONO_EPS));
    }
  }
  // now wire each child's parentError to its PARENT's finalized per-block ownError (== the
  // crack-safe-cut invariant parentError(child) == ownError(parent)). Roots keep parentError +inf.
  for (let L = levels.length - 1; L >= 1; L--) {
    const lvl = levels[L] as VoxelLevel;
    const finer = levels[L - 1] as VoxelLevel;
    for (const blk of lvl.blocks) {
      for (const fb of blk.childBlocks) {
        (finer.blocks[fb] as VoxelBlock).parentError = blk.ownError;
      }
    }
  }
  // HARD build-time crack-safety assert: flatten the per-level blocks into a global-cluster view
  // + the (now SPATIAL-TREE) child links, then run validateDagHierarchy — it proves the BFS
  // traversal reproduces the per-cluster cut EXACTLY at every threshold (no holes/double-emit).
  // With L0 ownError=0 it also proves every leaf is REACHED (every coarse block has children).
  validateVoxelPyramid(levels);
  return levels;
}

/** Build the global-cluster hierarchy view of a voxel pyramid + assert crack-safety via
 *  validateDagHierarchy. The spatial tree gives each finer block exactly ONE parent, so the
 *  traversal visits every block once; throws if the traversal cut != the per-cluster cut. */
function validateVoxelPyramid(levels: VoxelLevel[]): void {
  // map (level, level-local block idx) → global cluster id (flatten fine→coarse by level index,
  // blocks in order — the SAME order appendVoxelCrownPyramid/registerVoxelHead use).
  const globalOf: number[][] = levels.map(() => []);
  let g = 0;
  for (let L = 0; L < levels.length; L++) {
    const lvl = levels[L] as VoxelLevel;
    for (let bi = 0; bi < lvl.blocks.length; bi++) globalOf[L]![bi] = g++;
  }
  const total = g;
  const clusters: { ownError: number; parentError: number }[] = new Array(total);
  const childStart = new Uint32Array(total);
  const childCount = new Uint32Array(total);
  const childList: number[] = [];
  const roots: number[] = [];
  const coarsest = levels.length - 1;
  for (let L = 0; L < levels.length; L++) {
    const lvl = levels[L] as VoxelLevel;
    for (let bi = 0; bi < lvl.blocks.length; bi++) {
      const blk = lvl.blocks[bi] as VoxelBlock;
      const gid = globalOf[L]![bi] as number;
      clusters[gid] = { ownError: blk.ownError, parentError: blk.parent ? blk.parentError : Infinity };
      if (L === coarsest) roots.push(gid);
      childStart[gid] = childList.length;
      childCount[gid] = blk.childBlocks.length;
      for (const cbi of blk.childBlocks) childList.push(globalOf[L - 1]![cbi] as number);
    }
  }
  const dag = { clusters } as unknown as DagBuild;
  const hier: DagHierarchy = {
    childStart,
    childCount,
    childIndices: Uint32Array.from(childList),
    rootIndices: Uint32Array.from(roots),
  };
  const v = validateDagHierarchy(dag, hier);
  if (!v.ok) {
    throw new Error(`voxlod: voxel pyramid DAG is not crack-safe — ${v.msg}`);
  }
}

// ---------------------------------------------------------------------------
// REGISTRY WIRING (§5.2/§5.3) — the offline-voxelizer ↔ GeometryRegistry path:
// reserve bricks BEFORE build() (addLate freezes caps), then append + register a
// voxel:7 sibling head AFTER build(). Stage 1 exercises this whole path so the
// addLate.bricks precondition and appendBricks are proven before Stage 2's raster.
// ---------------------------------------------------------------------------

export interface PreparedVoxelCrown {
  vox: CrownVoxelization;
  /** number of OCCUPIED bricks to RESERVE/append (the §5.3 brick budget). voxlod=0: the
   *  single grid's occupied count. voxlod=1: Σ occupied bricks over ALL pyramid levels. */
  brickCount: number;
  /** number of voxel CLUSTERS this crown authors (the addLate.clusters reservation). voxlod=0:
   *  ceil(occupied/128). voxlod=1: Σ over all levels of ceil(level.occupied/128). */
  clusterCount: number;
  /** number of dagLinks entries this crown needs (roots + child links). voxlod=0: one root
   *  per cluster, 0 children. voxlod=1: roots(coarsest blocks) + Σ spatial-tree child links.
   *  NOTE: NOT consumed by callers — dagLinks usage == clusterCount exactly (every non-root block
   *  has exactly one parent => one child link; roots + nonRoots = clusters), so the clusters
   *  reservation already covers it and it is intentionally not reserved separately. Kept as a
   *  diagnostic/self-check field. */
  dagLinkCount: number;
}

/** count the ≤128-brick blocks an occupied-brick list splits into. */
function blockCountOf(occupied: number): number {
  return Math.max(0, Math.ceil(occupied / MAX_BRICKS_PER_CLUSTER));
}

/**
 * Voxelize a crown and report the brick/cluster/dagLink counts to reserve. Call BEFORE
 * `reg.build()` and sum the counts into `reg.addLate({ bricks, clusters })` across crowns —
 * the registry freezes the buffer sizes at build (§5.3). `voxlod` builds the MIP pyramid
 * (G1): false (default) reserves exactly today's single grid; true reserves all levels.
 */
export function prepareVoxelCrown(
  src: ExplicitSource,
  albedo: { r: number; g: number; b: number; hueVar?: number },
  voxelGridDim: number = DEFAULT_VOXEL_GRID_DIM,
  voxlod = false,
): PreparedVoxelCrown {
  const vox = voxelizeCrown(src, albedo, voxelGridDim, voxlod);
  if (!voxlod || !vox.levels) {
    const brickCount = vox.occupied.length;
    return { vox, brickCount, clusterCount: blockCountOf(brickCount), dagLinkCount: blockCountOf(brickCount) };
  }
  // voxlod: Σ over all levels. dagLinks = roots(coarsest level's blocks) + Σ spatial-tree child links.
  let brickCount = 0;
  let clusterCount = 0;
  let dagLinkCount = 0;
  const coarsest = vox.levels.length - 1;
  for (let L = 0; L < vox.levels.length; L++) {
    const lvl = vox.levels[L] as VoxelLevel;
    brickCount += lvl.occupied.length;
    clusterCount += lvl.blocks.length;
    if (L === coarsest) dagLinkCount += lvl.blocks.length; // roots
    for (const blk of lvl.blocks) dagLinkCount += blk.childBlocks.length; // child links
  }
  return { vox, brickCount, clusterCount, dagLinkCount };
}

/**
 * Append a prepared crown's OCCUPIED bricks to the registry post-build and register a
 * `voxel:7` sibling mesh head over the SAME instances (§5.2 / Stage-2 A1). Returns the
 * head + the appended brick range. The head carries NO triangle geometry — its clusters
 * point at BRICKS (word6=brickBase, word7-lowbyte=brickCount) and its matClass=voxel
 * keeps them out of the triangle raster (§4.1). Re-clusterizing the full leaf crown
 * OVERFLOWED the late caps (§A1), so the head is authored DIRECTLY via registerVoxelHead
 * (0 verts/tris, only `blocks.length` clusters — within the `?voxreg` reservation).
 *
 * The occupied bricks (appended in grid order) are split into ≤MAX_BRICKS_PER_CLUSTER
 * contiguous BLOCKS (the §5.3 per-coarse-cluster fit), one cluster each, with a per-block
 * brick-AABB bound (the voxel raster's AABB projection, §6.2). `matParam` should be the same
 * packed leaf tint as the leaf head (§7.2.6). `leafSource` is unused (kept for signature
 * stability — the head no longer carries a placeholder tri from it).
 */
/** one block handed to registerVoxelHead — a ≤128-brick voxel cluster + its AABB bound,
 *  plus (voxlod) the DAG cut metadata (own/parent error+sphere) and child cluster links. */
interface VoxelHeadBlock {
  brickBase: number;
  brickCount: number;
  aabb: { min: [number, number, number]; max: [number, number, number] };
  /** voxlod DAG metadata. ABSENT => the degenerate single-root path (voxlod=0, unchanged). */
  dag?: {
    ownError: number;
    ownSphere: [number, number, number, number];
    /** undefined => this block is a DAG ROOT (parentError +inf). */
    parentError?: number;
    parentSphere?: [number, number, number, number];
    /** finer child cluster ids (GLOBAL block indices within THIS head's block list). */
    childClusterIdx: number[];
    /** is this block a coarsest-level root (seeds the traversal)? */
    isRoot: boolean;
    /** PYRAMID level L (0 = finest/L0, coarser = higher): the ?nanitedbg=lod tint (word7
     *  bits 10-15). The raster is UNCHANGED (G3) — it just rides along for debug colouring. */
    dagLevel: number;
  };
}

/** shared append opts + result across the BrickCPU and packed-words entry points. */
type VoxAppendOpts = { matParam: number; swayPad?: number; maxDist: number; nearDist?: number; label?: string };
type VoxAppendResult = { head: MeshHandle; brickBase: number; brickCount: number; clusters: number };

/**
 * A LevelBricks abstracts "N occupied bricks of one grid level": copy their gpu records
 * into the brick buffer + report each brick's LOCAL center+half for the block AABBs. Two
 * adapters feed the ONE append core so both paths stay byte-identical on the GPU:
 *  - cpuLevel   — ForestScene's fresh builds (writeBrick per BrickCPU).
 *  - wordsLevel — world / bootcache path: a straight typed-array copy of the packed 9×u32
 *    records, NO per-brick BrickCPU objects (the toVoxel/fartile object-graph was the CPU
 *    heap slab this arc removes, 2026-07-04).
 */
interface LevelBricks {
  count: number;
  cellSize: number;
  fill: (dst: Uint32Array, base: number) => void;
  centerHalf: (i: number) => readonly [number, number, number, number];
  blocks?: VoxelBlock[];
}

function cpuLevel(bricks: BrickCPU[], occupied: number[], cellSize: number, blocks?: VoxelBlock[]): LevelBricks {
  return {
    count: occupied.length,
    cellSize,
    fill: (dst, base) => {
      for (let i = 0; i < occupied.length; i++) {
        const b = bricks[occupied[i] as number];
        if (b) writeBrick(dst, base + i, b);
      }
    },
    // b.center/b.half are the brick's local-space center ± (BRICK_DIM·cellSize·0.5),
    // both computed from origin + (bx+0.5)·brickWorld — so the AABBs are unchanged.
    centerHalf: (i) => {
      const b = bricks[occupied[i] as number] as BrickCPU;
      return [b.center[0], b.center[1], b.center[2], b.half];
    },
    ...(blocks ? { blocks } : {}),
  };
}

function wordsLevel(words: Uint32Array, count: number, cellSize: number, blocks?: VoxelBlock[]): LevelBricks {
  return {
    count,
    cellSize,
    // `words` is exactly count·BRICK_WORDS u32 in occupied order == the append order.
    fill: (dst, base) => { dst.set(words, base * BRICK_WORDS); },
    centerHalf: (i) => brickCenterHalf(words, i),
    ...(blocks ? { blocks } : {}),
  };
}

function aabbOverBricks(
  lb: LevelBricks,
  start: number,
  count: number,
): { min: [number, number, number]; max: [number, number, number] } {
  let mnX = Infinity, mnY = Infinity, mnZ = Infinity;
  let mxX = -Infinity, mxY = -Infinity, mxZ = -Infinity;
  for (let i = 0; i < count; i++) {
    const [cx, cy, cz, h] = lb.centerHalf(start + i);
    mnX = Math.min(mnX, cx - h); mxX = Math.max(mxX, cx + h);
    mnY = Math.min(mnY, cy - h); mxY = Math.max(mxY, cy + h);
    mnZ = Math.min(mnZ, cz - h); mxZ = Math.max(mxZ, cz + h);
  }
  return { min: [mnX, mnY, mnZ], max: [mxX, mxY, mxZ] };
}

// -- voxlod=0: single-resolution degenerate DAG. Split into ≤128-brick blocks + per-block AABB.
function appendCrownSingle(reg: GeometryRegistry, lb: LevelBricks, opts: VoxAppendOpts): VoxAppendResult {
  const brickBase = reg.appendBricks(lb.count, lb.fill);
  const blocks: VoxelHeadBlock[] = [];
  for (let start = 0; start < lb.count; start += MAX_BRICKS_PER_CLUSTER) {
    const count = Math.min(MAX_BRICKS_PER_CLUSTER, lb.count - start);
    blocks.push({ brickBase: brickBase + start, brickCount: count, aabb: aabbOverBricks(lb, start, count) });
  }
  const head = reg.registerVoxelHead(opts.matParam, blocks, {
    swayPad: opts.swayPad ?? 3.8,
    maxDist: opts.maxDist,
    label: opts.label ?? 'voxel',
  });
  // voxel-foliage (spec §3 / Stage 3a): the NEAR side of the mesh→voxel handoff — the voxel
  // head seeds ONLY beyond nearDist. 0/undefined = voxel everywhere (?forcevox debug route).
  if (opts.nearDist && opts.nearDist > 0) reg.setNearDistance(head, opts.nearDist);
  return { head, brickBase, brickCount: lb.count, clusters: blocks.length };
}

/**
 * voxlod=1 (DEFAULT world): append ALL pyramid levels' occupied bricks + author a REAL multi-
 * level DAG over them. Blocks are flattened across levels into ONE cluster list; their DAG cut
 * metadata (per-level ownError + own/parent spheres + spatial-tree child links) is resolved to
 * GLOBAL block indices. The cull's makeTraverse then auto-cuts: far => coarse level emits.
 */
function appendCrownPyramid(reg: GeometryRegistry, levels: LevelBricks[], opts: VoxAppendOpts): VoxAppendResult {
  // append level by level so each block addresses a CONTIGUOUS brick sub-range.
  let total = 0;
  for (const l of levels) total += l.count;
  const levelBrickBase: number[] = [];
  const brickBase = reg.appendBricks(total, (bricks, base) => {
    let cursor = base;
    for (const l of levels) { levelBrickBase.push(cursor); l.fill(bricks, cursor); cursor += l.count; }
  });
  const firstBase = brickBase;

  // flatten blocks across levels; map (level, level-local block idx) → GLOBAL block index.
  const globalOf: number[][] = levels.map(() => []);
  let g = 0;
  for (let L = 0; L < levels.length; L++) {
    const bl = (levels[L] as LevelBricks).blocks as VoxelBlock[];
    for (let bi = 0; bi < bl.length; bi++) globalOf[L]![bi] = g++;
  }
  const coarsest = levels.length - 1;
  const blocks: VoxelHeadBlock[] = [];
  for (let L = 0; L < levels.length; L++) {
    const lb = levels[L] as LevelBricks;
    const bl = lb.blocks as VoxelBlock[];
    const lvlBase = levelBrickBase[L] as number;
    for (let bi = 0; bi < bl.length; bi++) {
      const blk = bl[bi] as VoxelBlock;
      const aabb = aabbOverBricks(lb, blk.start, blk.count);
      // this block lives at level L => its children are level L-1 → GLOBAL cluster ids.
      const childClusterIdx = blk.childBlocks.map((cbi) => globalOf[L - 1]![cbi] as number);
      blocks.push({
        brickBase: lvlBase + blk.start,
        brickCount: blk.count,
        aabb,
        dag: {
          ownError: blk.ownError,
          ownSphere: [blk.own.x, blk.own.y, blk.own.z, blk.own.r],
          ...(blk.parent
            ? { parentError: blk.parentError, parentSphere: [blk.parent.x, blk.parent.y, blk.parent.z, blk.parent.r] as [number, number, number, number] }
            : {}),
          childClusterIdx,
          isRoot: L === coarsest,
          // PYRAMID level L directly (L0 finest = 0) — the ?nanitedbg=lod tint (word7 bits 10-15).
          dagLevel: L,
        },
      });
    }
  }
  const head = reg.registerVoxelHead(opts.matParam, blocks, {
    swayPad: opts.swayPad ?? 3.8,
    maxDist: opts.maxDist,
    label: opts.label ?? 'voxel',
  });
  if (opts.nearDist && opts.nearDist > 0) reg.setNearDistance(head, opts.nearDist);
  return { head, brickBase: firstBase, brickCount: total, clusters: blocks.length };
}

/** append a fresh (BrickCPU) crown — ForestScene inline builds + the fartile BrickCPU path. */
export function appendVoxelCrown(
  reg: GeometryRegistry,
  prep: PreparedVoxelCrown,
  _leafSource: ExplicitSource,
  opts: VoxAppendOpts,
): VoxAppendResult {
  const { vox } = prep;
  if (vox.levels && vox.levels.length > 0) {
    return appendCrownPyramid(reg, vox.levels.map((l) => cpuLevel(l.bricks, l.occupied, l.cellSize, l.blocks)), opts);
  }
  return appendCrownSingle(reg, cpuLevel(vox.bricks, vox.occupied, vox.cellSize), opts);
}

/** append a PACKED crown straight from its 9×u32 words — the world / bootcache path. No
 *  per-brick BrickCPU objects are materialized (the memory-arc win). GPU output is identical
 *  to appendVoxelCrown on the same crown (words ARE the writeBrick records); block AABBs use
 *  the f32 packed centers (sub-µm vs the fresh double centers — invisible cull bounds). */
export function appendPackedCrown(
  reg: GeometryRegistry,
  packed: PackedPreparedCrown,
  opts: VoxAppendOpts,
): VoxAppendResult {
  const { vox } = packed;
  if (vox.levels && vox.levels.length > 0) {
    return appendCrownPyramid(
      reg,
      vox.levels.map((l: PackedLevel) => wordsLevel(l.words, l.occupied.length, l.cellSize, l.blocks)),
      opts,
    );
  }
  return appendCrownSingle(reg, wordsLevel(vox.grid.words, vox.grid.occupied.length, vox.cellSize), opts);
}
