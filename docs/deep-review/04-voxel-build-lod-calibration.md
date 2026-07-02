# 04 — Voxel build + LOD ladder + calibration (VoxelizeCrown / BuildAggregateDag / VoxelBrick)

Reader 04 slice. All ms from the 2026-07-02 measured table or scratchpad `fresh-*.json`
(post-voxbocc effective baseline eye/oblique/aerial = 18.9 / 37.2 / 16.5). Sign convention in
this doc: negative Δms = faster.

## 1. TL;DR

- The real voxel tau is **12 px on the brick HALF-extent** (τ base 3 × lodWarp, clamped by
  `?voxtaucap=12` for voxel matclass over the whole band ≥41.4 m) — NOT the "one brick ≈ τ px /
  2.5–5 px" the VoxelizeCrown header still claims. Emitted bricks are 12–24 px wide, carved to
  3–6 px cells by ?voxcell.
- Transitions-1.5×-farther (the user demand) = `?voxtaucap=8` (runtime) or `?voxlodk=1.5`
  (build); honest extrapolation from the measured voxlodk curve: **+8–14 ms oblique** pre-voxbocc
  math, post-voxbocc unknown → probe P1. Unaffordable naked; must be funded.
- Funding levers found in the build: (a) mid-ring merged tiles at MATCHED τ (est −4..−8 oblique,
  Class R-controllable), (b) cell-accurate coarse occupancy re-bin — coarse masks today are
  box-dilated ~8×/level and mostly bypass the carve gate (est −2..−5 oblique, Class Q),
  (c) FarTiles/per-tree 94–140 m double-render from the tile nearDist margin (est −1..−3, Class I),
  (d) voxnear 45→60 with leaflodk=0.4 — quality UP and historically NEGATIVE cost (est −2..−4).
- Brick dedup/encode-shrink are dead as frame levers (bricks are per-species already; not
  fetch-bound). K_FLOOR=3 is load-bearing for small crowns — keep.
- Per-tree pyramids currently render exactly TWO levels in the 45–140 m band (L2+L3 for an 8 m
  crown); L0/L1/L4 are built+uploaded dead weight (memory only, ~0 frame ms).

## 2. How it works today

### 2.1 Build pipeline (offline, per SPECIES, not per instance)

`voxelizeCrown` (src/nanite/VoxelizeCrown.ts:398) SAT-rasterizes one crown mesh into a cubic
cell grid (`voxelGridDim` default 180 cells/edge, VoxelizeCrown.ts:192), 3³ supersampled coverage
(:196), then aggregates 4×4×4 cells → bricks: occupancy bit at cell coverage ≥ 0.02
(OCC_COVERAGE_THRESHOLD, :201, applied :560), density = mean cell coverage (:605), mean
normal/albedo density-weighted. A brick is emitted "occupied" whenever `bcov > 0` (:586–608) —
**there is no occupancy or featherweight gate** (contrast FarTiles.ts:220, which prunes
`w ≤ 0.15 || occ==0`). ForestScene voxelizes ONE crown per species (src/debug/ForestScene.ts:175,
202) — bricks are shared by all ~200k instances via instancing, so dedup has no frame-cost angle.

`buildVoxelPyramid` (:949) then 2×-downsamples the brick grid per level. Coarse brick
center = symmetric grid center, half = tight symmetric bound over occupied-children boxes clamped
into the grid cube (:799–814, the 6a93dfb centering fix), and occupancy is RE-BINNED into that
tight frame by stamping each occupied CHILD BRICK's whole box into the 4³ mask (:744, :758–764,
:819–836). Levels stop at `K_FLOOR = 3` bricks on the max axis (:1006–1007) — a crown can never
collapse to one square. Blocks = ≤128-brick spatial partitions (partitionClusters, DagCommon.ts:39),
wired into a strict spatial parent tree with coverage repair and `validateDagHierarchy` hard gate
(:1020–1186). FarTiles reuses the same `buildVoxelPyramid` on 64 m tile grids (cellSize 0.5 m,
FarTiles.ts:246, grid spec FarTiles.ts:114–115).

### 2.2 The ladder that actually ships (the anchor is dead code)

`VOXLOD_CFG = { levels: 7, errorK: 1, sparseK: 1, shell: 0, anchorL0: 0 }` (VoxelizeCrown.ts:87).
The shipped ownError is **`ownError(L) = brickHalf(L) · errorK`, L0 = 0**
(`curCell * BRICK_DIM * 0.5 * ERR_K`, VoxelizeCrown.ts:988). The `anchorL0` machinery
(computeVoxlodAnchorL0 :123–133, set by ForestScene.ts:111–128 and WorldRegistry.ts:402–413) is
**computed but no longer read by the ladder** — line 988's comment says it "dissolves the prior
build-time-projK anchor". The big header comments (:63–99, "bounded 2.5–5 px", "anchorL0 = 
transitionDist·tau/projK") describe the SUPERSEDED anchor era and contradict the code one page
down. Stale-doc hazard for every future reader; flag for cleanup.

### 2.3 τ derivation — what "one brick ≈ τ px" actually evaluates to

Cull cut (NaniteCull.ts:835): `pOwn = projK · A.w · ownError / denO ≤ τ_eff` emits, else descend.

- `projK = cotHalfFov · uH · 0.5` (NaniteCull.ts:362). fov = 55° (Engine.ts:51–56), canonical
  renderHeight = 1473 (dpr 1.5) ⇒ **projK = cot(27.5°)·736.5 = 1414.8 px·m⁻¹·m**.
- base τ = **3 px** (`?loderr` default 3, NaniteFrame.ts:157–158).
- lodWarp (NaniteCull.ts:111–126) with production defaults simband=6, lodnear=4, lodpow=0.6
  (NaniteFrame.ts:172–177): `τ_eff = 3·(1 + ((d−4)/6)^0.6)`.
- Voxel-matclass clamp `τ_eff ≤ ?voxtaucap = 12` (NaniteCull.ts:317–318, applied :852–859).
  The warp crosses 12 at d = 4 + 6·3^(1/0.6) ≈ **41.4 m** — i.e. for the ENTIRE voxel band
  (voxnear = 45, ForestScene.ts:102) the effective voxel τ is a **constant 12 px**, applied to
  the brick HALF-extent.

So level L is selected for d ∈ [projK·A.w·half_L/12, 2×), and within its band a brick's
half-extent projects 12 px (band entry) → 6 px (band exit): **brick edge 12–24 px on screen,
everywhere in the band**. ?voxcell (NaniteVoxelRaster.ts:286–302) carves bricks with footprint
area > 64 px² into 4³ cells ⇒ **effective cell granularity 3–6 px**. This is the true current
quality calibration. The header's "2.5–5 px" (VoxelizeCrown.ts:73) is off by ~5×.

### 2.4 Transition distances for a representative crown (maxExt 8 m, A.w = 1)

cellSize = 8/180 = 0.044 m; brick half_L = 0.089·2^L m; T(L) = 1414.8·half_L/12 = 117.9·half_L:

| level | brick edge (m) | band (m) | in the 45–140 band? |
|---|---|---|---|
| L0 | 0.18 | < 21 | never (below voxnear) |
| L1 | 0.36 | 21–42 | never (below voxnear) |
| L2 | 0.71 | 42–84 | **YES — band entry level** (edge 22 px at 45 m) |
| L3 | 1.42 | 84–168 | **YES** (84–140; edge 14–24 px) |
| L4 | 2.84 | 168+ | never (per-tree head ends at aggdist=140, ForestScene.ts:376) |

Grid ladder 46→23→12→6→3 bricks/axis ⇒ 5 levels built (L0..L4). Only TWO render. A.w spread
shifts bands proportionally; smaller crowns (4 m) shift T down 2× so their L4 does engage
in-band (K_FLOOR=3 then guarantees ≥3 bricks on the max axis — load-bearing, keep).

FarTiles (half_L = 1·2^L m): T(L1) = 236 m, T(L2) = 471 m ⇒ tiles render **L0 (2 m bricks)**
from their nearDist ≈ 94 m ('aggDist − 46', ForestScene.ts:390, FarTiles.ts:26) out to 236 m;
tile L0 bricks at 94 m project ~30 px edge (carved 7.5 px cells). Tile levels L3/L4 are
unreachable in a ~2 km world — dead build weight, memory only.

### 2.5 Leaf ladder (BuildAggregateDag) — the near side of the seam

Area-preserving island thin+grow; per-level `groupErr = growError · AGG_LOD_CFG.errorK` with
errorK baked 0.4 (BuildAggregateDag.ts:77, 504; ForestScene.ts:150). Leaf clusters take the FULL
uncapped lodWarp (the cap is voxel-only), so mesh-leaf τ_eff ramps 8.4→12.5 px over 20–45 m —
coarsening engages in-band (the 2026-07-01 LOD0-everywhere disease is fixed). History on the
seam: voxnear 35→60 measured −8.9 eye / −12.3 oblique (ForestScene.ts:93–97, pre-fartiles era),
then 60→45 chosen for QUALITY (0.25-ladder spiky crowns) at perf-neutral 45/0.4 vs 60/0.25
(ForestScene.ts:98–101). The 60/**0.4** combo was never measured.

### 2.6 bead-v2 interaction with coarser far bricks

Resolve shades the winning brick's baked mean normal (?voxbn, NaniteVoxelRaster.ts:259–265),
blended 60% toward a crown field = 0.75·crown-radial + 0.25·(wp−brickCenter)
(?voxbead=0.6, NaniteResolve.ts:259, 804–831). The dominant crown-radial term uses the INSTANCE
origin, so it is brick-size independent — shading is by-design robust to coarser bricks; only the
25% within-brick term facets coarser. One real coarse-brick artifact channel: FarTiles bricks ride
identity-ish instances so crownDir is per-TILE radial (NaniteResolve.ts:810–812 comment) — a 64 m
"mega-crown" shading field; invisible at 140 m+ today, would NOT be acceptable if merged heads
move to ~70 m (lever 4 must feed per-tree origins or a splatted crown-center field).

## 3. Waste map

| # | Waste | Mechanism (cite) | Scales with | est ms (eye/obl/aer) |
|---|---|---|---|---|
| W1 | Per-tree vox ring 45–140 m holds the oblique gap | 9188 voxClusters oblique vs 659 aerial (fresh-voxbocc.json); aggdist=60 diagnostic −10.7 obl pre-voxbocc | visible ring trees × blocks × footprint px | ring total ≈ 8–13 obl (from voxlodk curve, §5.1) |
| W2 | Coarse occupancy box-dilation ⇒ carve gate bypass | L≥1 masks stamp whole CHILD-BRICK boxes (VoxelizeCrown.ts:744,819–836): 1-cell child ⇒ 8/64 parent cells; two levels up masks trend full ⇒ popcount > OCC_MASK_FULL=48 skips the mask (NaniteVoxelRaster.ts:227,948–951) and voxcell DDA hits immediately ⇒ near-solid paint | coarse-brick count × footprint area | phantom fill 0.25–0.6 Mpx ⇒ 2–5 obl (slope 8.1 ms/Mpx) |
| W3 | FarTiles/per-tree DOUBLE-RENDER band 94–140 m | tile nearDist = aggDist−46 tests TILE-CENTER distance (kSeedRoots per-instance, NaniteCull.ts:786–787); trees in those tiles also render per-tree (maxDist 140, ForestScene.ts:376) — both heads elect the same crowns; overlap is a deliberate seam margin (FarTiles.ts:26) at instance granularity | tiles with center ∈ [94,185] in frustum | 1–3 obl (needs counter) |
| W4 | occ==0 / featherweight L0 bricks painted OPAQUE | all-cells-below-2%-coverage bricks still emit (VoxelizeCrown.ts:586–608, no FarTiles-style prune :FarTiles.ts:220); opaque mode paints the full rect regardless of density (NaniteVoxelRaster.ts:181–194) | crown thin-needle fraction | 0–1.5 (unmeasured; add build counter) |
| W5 | Dead pyramid tails (crown L0/L1/L4; tile L3/L4) | §2.4 — built, uploaded, never selected at current τ | brick memory only | ~0 frame ms — NOT a lever |
| W6 | Anchor machinery + stale calibration comments | §2.2 | maintenance risk | 0 ms, correctness-of-understanding |

## 4. Levers (ranked)

### 4.1 Mid-ring merged voxel tiles at MATCHED τ (structural — attacks W1)
Mechanism: extend the FarTiles machinery inward: second tile ring ~70–140 m, tileSize 32 m,
cellSize 0.25 m (τ-matched: at 70 m a 12 px half-extent = 0.59 m ⇒ 1 m L0 bricks need 0.25 m
cells). Same `buildVoxelPyramid`, same τ cut ⇒ projected brick error IDENTICAL to what per-tree
heads emit at those distances — this is what separates it from the red-listed `aggdist=60`
(which reused 0.5 m cells and WAS visibly coarse). Collapses ring clusters/instances (the
premise-audit §3.4 prime suspect; UE5 has no per-instance floor — one continuous cluster DAG).
Files: FarTiles.ts (second grid spec + ring nearDist/maxDist), ForestScene.ts:130–138 wiring,
per-tree maxDist → 70. Expected: obl −4..−8, eye −0.5..−2, aerial ~0, live: fewer clusters ⇒
cpu.submit down. Quality: **R** (per-tree sway + yaw/scale variation lost in ring; bead crownDir
per-tile §2.6 — must splat per-tree crown origins or accept). Gate: side-by-side crops 70–140 m
+ sway A/B video + shotdiff outside the ring. Effort: L. UE5 analogue: cross-instance
aggregation/HLOD-style merged Nanite geometry.

### 4.2 Cell-accurate coarse occupancy re-bin (build-side — attacks W2)
Mechanism: in `downsampleBrickGrid`, stamp each occupied child's OCCUPIED CELLS (their world
boxes) into the parent 4³ frame instead of the child's whole brick box (replace childBoxes
:744/:758–764 content with per-cell boxes; same conservative overlap stamping :822–836).
Composes up the pyramid; masks get ~8× tighter per level; more coarse bricks pass
`popcount ≤ 48` ⇒ voxcell carves them to true silhouette; fewer phantom elected px.
Offline cost ~64× more stamp boxes — boot-absorbed (workers splat, main-thread pyramid).
Expected: obl −2..−5, eye −0.5..−1, aerial −0.5..−2 (tiles benefit most — their canopy-union
masks are the fullest). Quality: **Q** (pixels change TOWARD ground truth: far crowns get the
see-through the mesh has; still a conservative cover of real geometry). Gate: far-crown crops
(user sign-off) + a build histogram of per-level mask popcounts + voxClusters/elected-px counter.
Effort: M. UE5: their brick path marches REAL occupancy (2026-06-25 doc Rank-2 finding —
this lever finishes what the occupancy-gate work started).

### 4.3 voxnear 45→60 with leaflodk=0.4 (quality UP, likely negative cost — the unmeasured combo)
Mechanism: push the mesh→voxel handoff out 15 m; the milder 0.4 leaf ladder (which fixed the
spiky-crown rejection at 45) keeps the 45–60 m mesh ring cheap. Voxels start farther (a direct
user demand), the most expensive vox bricks (nearest = biggest px) disappear, replaced by mesh
tris that are measured cheap (noleaves oblique carries 2.1M tris at 15.4 ms). Prior trail:
35→60 measured −8.9/−12.3 (different era/config, sign trusted, magnitude not — ForestScene.ts:93–97);
60/0.25 spiky-quality-rejected; 45/0.4 shipped; **60/0.4 never measured**. Files:
ForestScene.ts:102 default only. Expected: obl −2..−4, eye −1..−2 (post-voxbocc eye is already
canopy-culled). Quality: **Q** (band-end crown shape must pass the same eyeball that rejected
60/0.25). Gate: probe P2 + crops at 55–60 m. Effort: S.

### 4.4 Per-cluster near-envelope cull for tile heads (Class I — attacks W3)
Mechanism: today nearDist culls at INSTANCE granularity (NaniteCull.ts:786–787), so a tile
whose center is ≥94 m renders ALL its blocks, including those wholly covering trees < 140 m that
per-tree heads already render. Add a cluster-granular test in makeTraverse for voxel heads:
drop a block when `dist(cam, blockSphereCenter) + blockSphereRadius < aggDist − treeReach`
(conservative: such a block only contains geometry from trees that are per-tree-rendered).
No new storage buffer (meshes + dag already bound; Metal 10-buffer ceiling respected — the test
reads existing cluster sphere + mesh word 8). Expected: obl −1..−3, eye −0.5, aerial ~0
(aerial has no per-tree ring). Quality: **I** (conservative-cull-equal; seam shots at 135–145 m
must be bit-clean). Gate: shotdiff 5 poses + voxClusters delta at oblique. Effort: S–M
(cull-side; hand to reader 05/cull owner with this spec).

### 4.5 Transitions 1.5× farther: voxtaucap 12→8 (== voxlodk 1.5) — the paid quality lever
Mechanism: since τ_eff is cap-bound across the whole band (§2.3), voxtaucap 8 multiplies every
transition distance by 1.5 (runtime knob, no rebuild; NaniteCull.ts:317). Band entry at 45 m
becomes L1 (edge ~10 px finer), cells 2–4 px instead of 3–6. Identical effect to `?voxlodk=1.5`
baked. Cost model: ring brick count ×k² (shell) to ×k³; measured inverse curve §5.1 ⇒ naive
**+8..+14 obl, +2..+4 eye** pre-voxbocc math. Post-voxbocc+fartiles this is genuinely unknown
(the old "taucap 8 ruinous, aerial 36→81" was PRE-fartiles, NaniteCull.ts:308–316) and finer
bricks cull BETTER under per-brick occlusion — probe P1 pins it. Quality: **Q** (strictly finer
everywhere; the direct answer to "each level farther than now"). Ship rule: only after 4.1–4.4
land and the pose budget (obl ≤ 21) still holds. Effort: S (flag flip + probes).

### 4.6 (minor) occ==0 brick prune at crown build — attacks W4
Match FarTiles' occupancy prune (`occLo==0 && occHi==0` ⇒ drop) in voxelizeCrown's brick emit
(:586). Do NOT copy the `w ≤ 0.15` featherweight prune wholesale — crown bricks paint opaque, so
density-based pruning is a visible-pixel change (Class R); occ==0-only is near-I (the only
painted content is sub-2%-coverage phantom). Add a build-stats counter first; if occ==0 bricks
are <1% of occupied, close this. Effort: S. Expected: 0..−1.5 obl.

## 5. Refuted / rejected for this stage

- **5.1 The voxlodk curve (kept as the cost model, red-listed as a lever):** voxlodk=0.7 ⇒
  32.7/36.4/17.1 (−3.4/−6.8 vs pre-voxbocc base 36.1/43.2, fresh-voxlodk07.json); 0.85 ⇒
  34.1/41.1 (−2.0/−2.1, fresh-voxlodk085.json). Both user-rejected on quality. Implied ring
  totals 7.5–13.3 ms obl (inconsistency ≈ bimodality ±2 ms + voxcell area effects) — the basis
  of §4.5's +8..+14 extrapolation. Do not re-propose k<1.
- **5.2 Brick dedup / DAG sharing across instances:** already maximal — ONE voxelization per
  species shared by all instances (ForestScene.ts:175,202). Only FarTiles bricks are unique, and
  that is memory, not frame time. Dead as a perf lever.
- **5.3 Cheaper brick encode (<9 words):** no evidence of fetch-bound raster (leafcheap=all ≈
  −0.9/−0.8; premise audit §3.3). Reopen only on new fetch-bound evidence.
- **5.4 Trimming K_FLOOR or pyramid levels:** K_FLOOR=3 is the "never one square" guarantee and
  DOES engage for small crowns in-band (§2.4); dead tails cost ~0 frame ms (W5). No lever here.
- **5.5 sparseK / shell:** both default-off for documented reasons (ladder perturbation
  :102–106; hole risk :176–178). The shell's goal (kill interior overdraw) is now served by
  voxbocc + F2B-off reality; do not resurrect without new evidence.
- **5.6 Contradiction with prior docs, stated:** memory `nanite-voxel-vs-ue5-structural-gap`
  says "errorK=3 + K_FLOOR=3 fixed 683cdcc". Shipped today is errorK=**1** (VoxelizeCrown.ts:87,
  2026-07-02b note :80–86, rested A/B obl 55.9→38.5 for 1 vs 2) with the blob-guards moved into
  voxbn/voxcell/K_FLOOR. Also the 2026-06-25 "occupancy mask dead at raster" finding is
  PARTIALLY fixed (gate + voxcell exist since; NaniteVoxelRaster.ts:196–227) — the surviving
  form is W2 (dilated masks bypass the gate).

## 6. Open questions + serial GPU probes requested

- **P1 — voxtaucap=8 A/B (post-voxbocc cost of transitions-1.5×-farther).**
  `?voxtaucap=8` vs default, 3 poses, same session interleaved, + crops at 60/100/140 m.
  Decision: obl cost ≤ +4 ⇒ §4.5 affordable after funding levers; ≥ +8 ⇒ park §4.5 behind §4.1.
- **P2 — voxnear=60&leaflodk=0.4 A/B** + crops at the 55–60 m band end.
  Decision: obl Δ ≤ 0 AND crowns pass eyeball ⇒ promote defaults (quality up, free); spiky ⇒
  try leaflodk=0.5–0.6 once more, else close.
- **P3 — voxlodk=1.25 A/B** (build-side finer, half-step). Decision: agreement with P1's
  per-step cost within 2 ms ⇒ treat runtime voxtaucap as THE shipping knob (no rebuilds);
  disagreement ⇒ the ladder and cap are not equivalent (block-partition granularity effect) —
  investigate before shipping either.
- **P4 — FarTiles overlap bound.** Add `?ftnear=` override for the appendFarTiles nearDist
  (ForestScene.ts:390) and run `?ftnear=140` vs default at oblique + seam shots ~135–145 m
  (EXPECT holes at the seam — diagnostic only, bounds W3). Decision: obl Δ ≥ 2 ⇒ build §4.4;
  < 1 ⇒ close W3.
- **P5 — engagement counters for W2/W4** (boot-log, ride any probe above): per-level histogram
  of coarse-mask popcounts + occ==0-brick count + voxcell carve-rate (bricks carved / eligible).
  Decision: median L2/L3 popcount > 48 confirms W2 headroom ⇒ build §4.2; occ==0 < 1% ⇒ drop §4.6.

Open questions: (a) crown extents per species (my §2.4 table assumes 8 m — print maxExt at build
to pin the real band map); (b) does A.w spread (instance scale) smear the L2/L3 seam enough to
matter for §4.5's cost; (c) if §4.1 lands, what replaces per-instance sway in the ring (splatted
sway phase per crown column?) — quality gate owns this.
