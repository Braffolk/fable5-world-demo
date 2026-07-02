# SPEC — QUALITY-PARITY tiered fine-cell ring tiles for the 60–140 m band (`?ringtiles=1`)

Status: SPEC (buildable, not built). Author context: deep-review docs 06 + 15 + 90 + 20;
baseline `fresh-voxbocc.json` (eye 18.9 / oblique 37.2 / aerial 16.5 med gpuWall, 200k trees,
2268×1473, nanite.mb 1353). Mission share: oblique −11 ms, ZERO quality sacrifice (user law).

This spec is self-contained: every mechanism it relies on is cited to code, and every number
to a measurement JSON in
`/private/tmp/claude-501/-Users-sebastian-IdeaProjects-fable-demo2/cc111c9f-86e4-4c1a-be01-9817f021c312/scratchpad/`.

---

## 0. STOP rules the implementing engineer MUST honor (read first)

1. **Go up a level before you grind.** On ANY negative/disappointing result (perf null, quality
   fail, memory overrun), do NOT vary the method first. Put the setup on trial: the params
   (cell size, OCC_COVER, ring bounds), the metric (med gpuWall vs slot histogram vs
   projected-CELL-size), and the structure one level up (the tier seams themselves,
   doc `90-premise-audit.md` §3.4). Only after the setup is explicitly cleared may you conclude
   "doesn't work". A null result may be recorded ONLY with an engagement counter proving the
   mechanism fired (doc 90 §6 trap 2) — here: the `nanite.voxClusters` restructuring delta (§7.2).
2. **Never park silently.** If a gate fails, surface the decision (revert / retune / abandon)
   to the user; do not quietly leave the flag off and move on.
3. **The gate is the user-observable output** (does the 60–140 m band look IDENTICAL at the
   poses the user checks?), not "machinery correct" internals.
4. **Quality class R** (doc 90 §4): this lever resamples a representation ⇒ pixels WILL differ.
   The implementing agent may NOT self-certify "you can't see it" (aggdist=60 and voxlodk=0.7
   both looked fine to their authors and were user-rejected). Default-flip requires explicit
   user sign-off on the shotdiff evidence (§7.3). Until then everything stays behind flags.

---

## 1. Problem + measured motivation

### 1.1 The ring is the oblique whale's home

The three-tier foliage stack (per species): triangle leaf head to 45 m
(`?voxnear=45`, src/debug/ForestScene.ts:102), per-tree voxel sibling head 45→140 m
(appendVoxelCrown `maxDist: farTilesOn ? aggDist : 2000`, ForestScene.ts:371–379), and merged
64 m FarTiles beyond `nearDist = max(10, aggDist−46)` = 94 m (ForestScene.ts:390) with
0.75 m cells (`ftcell`, ForestScene.ts:317). The 45–140 m PER-TREE band dominates the oblique
pose: **voxClusters 9185 at oblique vs 661 aerial** (`fresh-voxbocc.json` counters).

The `?aggdist=60` diagnostic (merged tiles pulled to 60 m) measured the ring pool: oblique
43.1 → 32.3 (−10.8), voxClusters 9188 → 4429 (`fresh-bead-v2-base.json` → `fresh-aggdist60.json`,
both pre-voxbocc era). Doc `06-ring-and-fartiles.md` §3.1 decomposes the −10.8:

| share | ms | capturable at zero quality loss? |
|---|---|---|
| (b) brick-size coarsening (0.75 m cells at 14–60 m = 17–76 px voxels) | ≈ −6.8 | **NO — quality-rejected** ("dogshit, massive voxels", red list doc 90 §4) |
| far-trunk-mesh removal (bark clamped to 60) | ≈ −1.3 | NO in this lever (bark stays; belongs to the trunk-DAG lever) |
| (c) cross-tree occupancy dedup + double-draw removal, AT 0.75 m cells | ≈ −2.5..−3 | **YES — this spec's target**, IF the merged rep is equal-or-finer |

So the honest target of a quality-parity merge is the **≈2.5–3 ms cross-tree dedup share plus a
slice of ring self-overdraw**, NOT the 10 ms headline. Two caveats sized by probes before any
build (§7.1): the decomposition is pre-voxbocc (the post-voxbocc oblique ring residual is
unmeasured — doc 06 P-A), and the tile-wave-occlusion lever (doc 06 L1 / doc 15 L1, a separate
spec) overlaps the double-draw slice.

### 1.2 Why the naive version failed and what "parity" must mean

`aggdist=60` failed because a FIXED 0.75 m cell was dragged to 14–60 m where it projects
17–76 px (projK = cot(27.5°)·1473/2 ≈ **1414.8** at 2268×1473; doc 06 §2.3), plus floaters
(isolated low-coverage cells clearing OCC_COVER=0.22, src/nanite/FarTilesSplat.ts:22,205–219,
surviving the `w ≤ 0.15` brick prune, src/nanite/FarTiles.ts:220) and merged/offset crowns.
Parity version = cell size matched to what the per-tree path actually shows at ring distance
(§3.1), floaters fixed at the ROOT (per-ring OCC_COVER + connectivity, §3.6), crown identity
preserved through geometry AND shading (§3.7), and a shotdiff + user gate before any default
flip (§7.3).

### 1.3 Why tiles must be STREAMED, not static

Brick count scales ≈ cell^−2.39 (measured pair: 0.75 m → 4.03 M bricks full-world
(`fresh-voxbocc.json` console: "812 tiles, 4033553 bricks (138.5 MB)"), 0.5 m → ~10.7 M
extrapolated, ForestScene.ts:313–316). Full-world 0.25 m tiles ⇒ ~56 M bricks ≈ 2.0 GB — dead
(doc 06 §3.5). The ring annulus is only ~1.6% of the plantation ⇒ a resident ring of fine
tiles is ~40 MB (§4.3). Therefore: **a fixed-size streamed slot pool** — exactly the proven
terrain-tile-pool pattern (`reserveTilePool`/`attachHeightDagTile`/`evictHeightDagTile`,
src/nanite/GeometryRegistry.ts:1817–2043).

Note on the "256 MB cliff": review docs treat 256 MB as the brick-buffer ceiling (cited to the
ForestScene.ts:314–316 comment), but src/core/Diagnostics.ts:36–52 requests
maxBufferSize/maxStorageBufferBindingSize ≈ 4.29 GB and the adapter grants ~4 GB
(`fresh-aggdist60.json` console dump). The design below fits under the CONSERVATIVE 256 MB
reading regardless (§4.3), so nothing depends on resolving the discrepancy — but log the
device-granted limits once at boot when landing Stage R1.

---

## 2. Design overview

Replace the PER-TREE voxel representation in the **60–140 m ring** with streamed, 16 m,
fine-cell merged tiles whose visible granularity is bounded by the SAME τ machinery the
per-tree path uses:

- Tier 1 (unchanged): mesh leaves 0–45 m; per-tree voxel heads now **45–60 m** (maxDist 140→60).
- Tier 2 (NEW): **16 m ring tiles, cellSize 0.25 m** (`?ringcell`), crowns only (no trunk
  columns — real bark meshes still render to 140 m, ForestScene.ts:388), engaged
  [ringNear−12, aggDist+12] = [48, 152] by tile-center distance (§3.3).
- Tier 3 (unchanged): 64 m / 0.75 m FarTiles beyond 94 m as today (they own 140 m+; the
  94–152 m overlap vs ring tiles replaces today's 94–140 m overlap vs per-tree).

Everything rides the EXISTING machinery: `splatTiles` (pure, worker-hosted,
src/nanite/FarTilesSplat.ts:79–224), `buildVoxelPyramid` multi-level DAG
(src/nanite/VoxelizeCrown.ts:949–1018), voxel heads with brick-range clusters
(`registerVoxelHead`, src/nanite/GeometryRegistry.ts:1041+), the vox fanout + scatter + per-brick
occlusion (src/nanite/NaniteCull.ts:512–533; src/nanite/NaniteVoxelRaster.ts:588–920), and the
resolve voxel branch (src/nanite/NaniteResolve.ts:795–848). **No new GPU storage buffers, no new
kernels** (one optional 5-line gate inside kVoxFanout, §3.5). The new code is CPU-side: a slot
pool in GeometryRegistry + a streamer module + splat/emit parameterization.

### 2.1 Why 16 m tiles and one cell size (the "tiered" part is the pyramid)

- 16 m tile ⇒ half-diagonal 11.31 m ⇒ the hole-free seam overlap shrinks from 46 m (64 m tiles)
  to 12 m — the mechanism behind the FarTiles nearDist construction (FarTiles.ts:24–28,
  ForestScene.ts:390) reused verbatim at the new seam.
- ONE ring cell size (0.25 m) suffices because each tile carries its OWN voxel MIP pyramid
  under the SAME screen-error cut: ownError(L) = curCell·BRICK_DIM·0.5·errorK
  (VoxelizeCrown.ts:988), τ_eff clamped to 12 px for voxel clusters (`?voxtaucap=12`,
  NaniteCull.ts:308–318, 852–860). A ring tile emits L0 (1 m bricks, 0.25 m cells) to
  ≈ projK·1.0/12 ≈ 118 m, then L1 (2 m bricks, 0.5 m cells) — i.e. the distance-tiering the
  earlier "two ring bands" sketch (doc 06 L3) hand-rolled falls out of the pyramid for free.
  Grid: 16 m/0.25 m ⇒ cellsXZ=64, cellsY=192 (FarTiles.ts:114–115 formulas), bricks
  16×48×16 = 12288 slots; pyramid levels L0..L4 (K_FLOOR=3 stop, VoxelizeCrown.ts:1006–1007).

---

## 3. Detailed design + exact edit sites

### 3.1 The quality-parity cell size (the load-bearing constant)

What the user actually SEES per vox pixel is the **cell**, not the brick: emitted bricks
subtend ≤ 24 px (τ_eff=12 clamps projected brick HALF-extent, NaniteCull.ts:852–860) and any
footprint > 64 px² is carved per-cell by the voxcell DDA
(NaniteVoxelRaster.ts:903–919 eligibility + Phase-B carve) ⇒ **visible element = brick/4 =
cell ≤ 6 px**, in BOTH representations, always. The per-tree side at 60 m concretely: species
crown L0 cell ≈ 0.061 m (grid 180 over ~11 m crowns, ForestScene.ts:92, DEFAULT_VOXEL_GRID_DIM
VoxelizeCrown.ts:192) ⇒ per-tree emitted level at 60 m is L2 (brick ≈ 0.98·A.w m, cell
≈ 0.244·A.w m ≈ 5.8 px at A.w=1).

**Rule: ringcell = the species-mean per-tree L0 BRICK world size = 0.25 m** (default;
`?ringcell` sweeps). Consequences:

- At the 60 m seam, ring cells (0.25 m ≈ 5.9 px) equal the mean per-tree emitted cell
  (0.244·A.w m). Deeper in the band both reps coarsen under the SAME τ ceiling ⇒ projected
  cells stay in (1.5, 6] px on both sides.
- Honest caveat (do not hide it from the user): equal-or-finer is exact vs the SPECIES-MEAN
  tree. Small instances (A.w=0.8) emit ~0.195 m cells at 60 m — up to ~25% finer than the
  ring's 0.25 m. Octave PHASE also differs (tile ladder vs tree ladder), so locally the ring
  can be up to ~2× coarser-or-finer within the shared ≤6 px bound. This is exactly what the
  shotdiff + user gate (§7.3) exists to judge; `?ringcell=0.1875` is the pre-planned fallback
  (strictly ≤ every tree's minimum, ~2× memory/boot per the ^−2.39 law).

### 3.2 Splat + emit changes (pure CPU, behavior-exact for legacy callers)

FarTilesSplat.ts is contractually bit-identical for the existing 64 m path
(header, FarTilesSplat.ts:5–8) — ALL changes are additive and gated on new optional fields.

1. **`SplatGridSpec` gains `trunks: 0|1` (default 1) and `origins: 0|1` (default 0)**
   (FarTilesSplat.ts:46–54). In `splatTiles`:
   - trunk column loop (FarTilesSplat.ts:196–202) runs only when `trunks` — ring tiles pass 0
     (real bark meshes cover 60–140 m; splatting columns would double-draw trunks AND resurrect
     the antenna class, doc 06 §3.6).
   - when `origins`: two extra per-brick accumulators `accOX, accOZ` (Float32Array(nBricks),
     alongside FarTilesSplat.ts:100–108) accumulating `w·(treeX − brickCenterX)` — wait, at
     accumulate time the brick center is derivable from `bi`; accumulate `w·treeLocalX/Z`
     (`lx0, lz0` at FarTilesSplat.ts:153–154) and let emit subtract the brick center. Extend
     `splatCell` (FarTilesSplat.ts:124–139) with the two writes; two extra arrays in
     `TileSplatOut` (FarTilesSplat.ts:63–74) + the worker transfer list
     (FarTiles.worker.ts:29–42). Purpose: §3.7 crown-identity shading.
2. **`planFarTiles`** (FarTiles.ts:91–198): already parameterized by `tileSize`/`cellSize`
   via `FarTileOpts`. Export it (currently module-private) plus `emitTile` (FarTiles.ts:201–283)
   so the ring streamer plans/emits per tile. Add to `emitTile`: when `origins`, write
   `brick.originPacked = packOriginOffsetXZ(meanOX − brickCx, meanOZ − brickCz)` (§3.7) and
   set `spread` unused; when building for the ring also apply §3.6's occupancy post-filter.
3. **Worker pools cache** (FarTiles.worker.ts:11–16): `FtReq.pools` becomes optional; the
   worker retains the last-received pools per `id` session (`let cachedPools`). The boot
   FarTiles path always sends pools (byte-identical behavior); the ring streamer sends pools
   once per worker then omits them (~4 MB structured-clone per message saved).

### 3.3 Seam construction (hole-free by the proven overlap rule)

All gating uses EXISTING per-mesh words — no cull changes:

- Per-tree voxel heads: `maxDist` 140 → **ringNear = 60** (`?ringnear`) — the ForestScene
  append loop already computes maxDist conditionally (ForestScene.ts:371–379); it becomes
  `ringOn ? ringNear : (farTilesOn ? aggDist : 2000)`. Enforced per INSTANCE distance at seed
  (NaniteCull.ts:776–779).
- Ring tile heads: ONE identity instance at tile center (appendFarTiles pattern,
  FarTiles.ts:361–381), `nearDist = ringNear − 12` = 48 (word-8 gate, NaniteCull.ts:786–787;
  setNearDistance GeometryRegistry.ts:1322–1327), `maxDist = aggDist + 12` = 152 (lodDist).
  Overlap = tile half-diagonal (11.31 ≤ 12) ⇒ a tree dropped by one tier is ALWAYS covered by
  the other: tree at instDist ≥ 60 ⇒ its tile center ≥ 60 − 11.31 > 48 ⇒ tile seeded; tree at
  instDist ≤ 140 covered per… (ring tile ON through 152 ≥ 140 + 11.31). Same algebra as the
  shipped 94 m seam.
- FarTiles (64 m) heads: UNCHANGED (nearDist 94, maxDist 100000). Bark: UNCHANGED (maxDist 140).
- Boundary crowns: instances bucket into EVERY tile their reach touches
  (FarTiles.ts:153–189) — reused verbatim ⇒ no tile-border holes.

Double-draw bands after this spec: [48,60] ring-vs-per-tree (killed by §3.5's near-gate),
[94,152] ring-vs-FarTiles (pre-existing class; the tile-wave lever and/or a mirrored far-gate
own it — out of scope here, noted in §8).

### 3.4 The streamed slot pool (GeometryRegistry) — mirror of the terrain tile pool

New methods, modeled line-for-line on the proven trio (reserveTilePool
GeometryRegistry.ts:1817–1846, attachHeightDagTile :1895–2017, evictHeightDagTile :2026–2043):

1. **`reserveVoxTilePool(slots, brickCap, clusterCap, opts)`** — PRE-build:
   `addLate({ bricks: slots·brickCap, clusters: slots·clusterCap, meshes: slots,
   instances: slots })` (addLate, GeometryRegistry.ts:987–996). Defaults (§4.3):
   slots=320, brickCap=6144 (`?ringcap`), clusterCap=96. dagLinks need = clusters exactly
   (roots + one child-link per non-root, VoxelizeCrown.ts:1271–1276) and the dagLinks buffer
   auto-sizes 2× clusters (see :1829–1831 comment) — no extra reservation.
2. **`initVoxTilePool()`** — POST-build, once: claim the contiguous brick band
   (advance `brickCursor` by slots·brickCap — bypasses appendBricks, which is cursor-append-only,
   GeometryRegistry.ts:1008–1022), claim cluster + dagLinks bands likewise, then for each slot
   register an EMPTY voxel head: `newEntry(handle,'voxel',…)` exactly as registerVoxelHead does
   (GeometryRegistry.ts:1101–1120) but with clusterCount=0, rootCount=0, sphere parked at
   `TILE_EVICTED_FAR` (:135–139) — kSeedRoots skips rootCount==0 (NaniteCull.ts:770). Bind one
   identity instance per slot (bindInstances precedent FarTiles.ts:376–378). Set
   **`MESH_FLAG_RINGTILE = 32`** (new constant next to :166–177; byte 2 of mesh word 6 has
   bits 32/64/128 free) on every slot head.
3. **`attachVoxTile(slot, blocks, center, nearDist, maxDist)`** — the in-place rewrite:
   validate `Σbricks ≤ brickCap`, `blocks.length ≤ clusterCap` (SKIP-oversized + counter,
   the tilePoolCap precedent :1854–1861 — an oversized tile stays unresident, which is safe
   because the global handoff (§3.8) then keeps per-tree rendering); write brick records into
   the slot's fixed brick band (`writeBrick`, src/nanite/VoxelBrick.ts:130–141); author cluster
   records + DAG records + dagLinks EXACTLY as registerVoxelHead's loop does
   (GeometryRegistry.ts:1122–1230) but addressed at the slot bases; update the mesh entry
   (clusterBase/Count, rootBase/Count, sphere from block union, lodDist=maxDist,
   word-8 nearDist) + `writeMeshRecord`; rewrite the identity instance A=(cx,0,cz,1) in the
   instances array; `pushRange` voxelBricks/clusters/dag/dagLinks/mesh/instances — the exact
   partial-upload set attachHeightDagTile pushes (:2009–2014) plus voxelBricks (:1020) and
   instances.
4. **`evictVoxTile(slot)`** — clusterCount=0, rootCount=0, park sphere, push mesh record, free
   slot (mirror :2026–2043).

Blocks arrive from the UNCHANGED `buildVoxelPyramid` output via the same flatten
`appendVoxelCrownPyramid` performs (VoxelizeCrown.ts:1436–1504) — extract its
"levels → VoxelHeadBlock[] (+ level brick bases)" section into a shared pure helper
`pyramidToBlocks(levels, brickBase)` so append (boot FarTiles) and attach (ring) share one
authoring path.

### 3.5 Runtime near-gate (Stage R3, `?ringgate=1`) — kills the [48,60] double-draw

Edit site: inside kVoxFanout's `If(matClass == VOXEL_MATCLASS)` (NaniteCull.ts:526–531).
Read mesh word 6 flags byte (idiom :523–525 reads the matClass byte of the same word); if
`MESH_FLAG_RINGTILE` set, read the cluster sphere (readCluster, already used at :522) and the
identity instance A (gpu.instances — new read in THIS kernel; binding count goes ~5→6, well
under the 10-buffer Metal cliff), and skip the qVoxRaster append when

    dist(camPos, sphereCenter + A.xyz) + sphereRadius < ringNear − reachMax

with `reachMax` = max over species of `reachOf` (FarTiles.ts:153–165) baked into a uniform
(≈ 12–16 m). Every tree contributing content to such a cluster has instDist < ringNear ⇒ its
per-tree head is live (per-tree maxDist = ringNear) ⇒ provably hole-free (the doc-06 L2
argument at the new seam). TSL note: build the whole test inside the existing `If` closure —
do NOT hoist the elemU reads to Fn scope (r184 hoist hazard; precedent comment
NaniteVoxelRaster.ts:1194-area and doc 90 §3.1). This also removes the only place where a
coarser ring brick could WIN an election against a live finer per-tree brick (the doc-15 §waste-2
"coarse bleed" class) ⇒ strictly quality-improving.

### 3.6 Per-ring OCC_COVER + the floater root-cause fix (bake-time, Stage R1)

Root cause of floaters (doc 06 §3.6): at 0.75 m cells an isolated fringe cell where two crown
fringes overlap clears OCC_COVER=0.22 and its brick survives the `w ≤ 0.15` prune with 1–2
occupied cells. At 0.25 m cells the fringe population GROWS (a single source brick can trip a
cell alone: overlap fraction up to (0.244/0.25)³ ≈ 0.93 vs ≤0.35 at 0.75 m), so the threshold
must be retuned per ring, not inherited:

- **`?ringocc` (default 0.35)**: ring tiles pass their own OCC_COVER via `SplatGridSpec`
  (make the constant a spec field defaulting to 0.22 — legacy path unchanged,
  FarTilesSplat.ts:22, 205–219).
- **Connected-component prune (`?ringcc=1`, default on for ring)** in the ring emit step
  (after FarTiles.ts:216–243's dense loop): drop occupied bricks whose 26-connected component
  has < 3 bricks AND component mean w < 0.5 (keeps legitimate saplings, kills 1–2-brick
  specks). Pure deterministic CPU, ~O(occupied).
- **Re-pin the w-scale constants**: brick `w ≤ 0.15` prune (FarTiles.ts:220) and
  `density = min(1, w/8)` (FarTiles.ts:235) were tuned at 0.75 m contribution counts; scale
  both by (ringcell/0.75)³ as a starting point and validate against the §7.2 brick-count gate.
- Trunk antennas cannot arise (no trunk columns in ring tiles, §3.2). The `crownMinY = 2`
  initializer-as-cap bug (ForestScene.ts:336–337, flagged in doc 15) is therefore untouched
  here — leave it to the FarTiles quality lever.

### 3.7 Crown-identity shading (Stage R2, `?ringbead=1`)

The resolve's default round-normal field (`?voxbead=0.6`, NaniteResolve.ts:253–260) computes
`crownDir` from `ctrW − vA.xyz` — the INSTANCE origin (NaniteResolve.ts:812–824). Per-tree
heads: per-crown radial field. Ring tiles ride ONE identity instance at the tile center ⇒
without a fix, crown shading flattens to a near-constant per-tile field (the comment at
:810–811 documents exactly this for 64 m tiles — acceptable at 140 m+, NOT at 60 m). Fix, with
NO layout change and NO new buffers:

- **Bake**: per ring brick, the w-weighted mean source-tree XZ offset (accOX/accOZ from §3.2)
  minus the brick center = `(dx, dz)`, |d| ≤ halfDiag + reach ≈ 23.3 m. Pack snorm2x16 at
  scale **RING_ORIGIN_R = 24 m** into brick word 3 (BRICK_SPREAD, VoxelBrick.ts:69,77).
  Safe: no runtime kernel reads BRICK_SPREAD today (grep: only VoxelBrick.ts's CPU
  readBrick mirror), the mean-normal resolve path ignores spread by design
  (VoxelBrick.ts:38–44). Add `packOriginOffsetXZ`/`BrickCPU.originPacked?` to VoxelBrick.ts
  (writeBrick :130–141 writes it raw to word 3 when present — document in the layout header,
  "change it HERE only" rule).
- **Resolve**: in the bead block (NaniteResolve.ts:812–824), when the winning brick's mesh has
  `MESH_FLAG_RINGTILE` (mesh word 6 flags byte via the existing fetch path), reconstruct
  `treeOriginW = (ctrW.xz − unpackSnorm2x16(word3)·24, y=0)` and feed the UNCHANGED formula
  `d0 = ctrW − treeOriginW`. The 25%-weight `beadPix` term keeps using ctrW (brick centers
  differ ≤ half a cell between representations — sub-element). Result: the ring's shading
  field radiates from the SAME per-tree origins as the per-tree rep. Where crowns
  interpenetrate, the weighted-mean origin blends — those pixels were already a cross-crown
  election mix per-tree.
- `?voxjit` value jitter is world-anchored (hash of floor(wp·1.4), NaniteResolve.ts:261–265)
  ⇒ identical across representations for free. Albedo: per-brick w-weighted mean of source
  brick albedos (`?voxbn` path, NaniteResolve.ts:838–848) — equals the source albedo wherever
  one crown owns a cell.

### 3.8 The streamer (`src/nanite/RingTiles.ts`, new) + fallback handoff

- **Plan once at boot** (reuses exported planFarTiles with tileSize=16, cellSize=ringcell,
  trunks=0, origins=1): the per-tile member map for the WHOLE world is cheap (it's the same
  bucketing pass the 64 m build already does in <1 s); tile GRIDS are not built — only jobs.
- **Residency manager** (Engine.onUpdate hook, Engine.ts:122–123; wired in ForestScene like
  the other scene hooks): required set = tiles intersecting the CURRENT-pose annulus
  [ringNear−12, aggDist+12]; prefetch set = annulus of pose + 2 s velocity lead. LRU-evict
  outside prefetch. Load rate: annulus ≈ 65 k m² ⇒ ~255 resident tiles (+ partials → 320
  slots); translation sweeps ≈ 2(152+48) m²/m ⇒ ~1.6 tiles/m ⇒ ~16 tiles/s at 10 m/s.
- **Persistent 2-worker pool** (FarTiles.worker.ts reused with the §3.2 pools-cache): per-tile
  splat ≈ 20–50 ms (measured 44–52 ms per 64 m tile at 0.75 m — doc 06 L3; ring tiles have
  ~10× fewer members × ~3× more source bricks). 16 tiles/s ≈ 0.3–0.8 worker-s/s — fits.
  Main-thread emit+pyramid ≈ 2–5 ms/tile (64 m emit measured 3477 ms/812 ≈ 4.3 ms,
  `fresh-aggdist60.json` console), budgeted **≤1 tile per frame** (doc 20: live CPU is
  1.15 ms/frame with ~15 ms headroom; 0 longtasks must stay 0).
- **Boot prefetch**: initial annulus ≈ 300 tiles ≈ 6–15 worker-s ≈ 1–2 s wall on the 8-worker
  boot pool (run it alongside the existing 64 m build; boot budget +≤3 s on 72 s,
  `fresh-voxbocc.json` bootS).
- **Global conservative handoff (the teleport/fallback story)**: per-tree vox heads keep
  TODAY's maxDist (140) until the required set is FULLY resident; then one
  `setMaxDistance(perTreeVoxHead, ringNear)` flip per species (post-build rewrite is
  supported: GeometryRegistry.ts:1307–1313 rewrites + uploads the mesh record). If any
  required tile goes missing (teleport, eviction race, oversized-skip), flip back up the same
  frame. Hysteresis: flip down only after N=30 consecutive fully-resident frames; count flips
  in a `nanite.ringFlips` counter (flapping gate §7.4). Consequence: the WORST failure mode is
  today's exact renderer — never a hole, never a coarser-than-today pixel. Probe poses:
  static poses settle ≫ prefetch time; the probe already waits for ready + settle
  (tools/probe-fresh-stutter.ts:76–80 + settle in the shot path).

### 3.9 What is deliberately NOT in this lever

- No bark/trunk changes (the −1.3 ms far-trunk share is the trunk-DAG lever's).
- No voxOccPyr/tile-wave changes (doc 06 L1 / doc 15 L1 — separate spec; ring tiles are
  matClass voxel:7 and inherit whatever the vox path ships, incl. per-brick `?voxbocc`,
  NaniteVoxelRaster.ts:838–884).
- No change to the 64 m FarTiles layer, the ≥140 m look, `?voxtaucap`, the voxlod ladder,
  errorK, K_FLOOR, dpr, TRAA (red list, doc 90 §4).

---

## 4. Budgets

### 4.1 GPU frame model (why this wins at oblique)

Ring per-tree clusters at oblique ≈ 9185 − 4429 ≈ 4.7 k (the aggdist60 run's 4429 ≈ non-ring
share, `fresh-aggdist60.json`). Ring tiles replace them with ~40–60 visible tile heads ×
~70 bricks/cluster of MERGED occupancy: cross-tree interpenetration dedup (trees at 4 m
spacing, ~11 m crowns) collapses inter-crown depth overdraw — the measured ≈2.5–3 ms share at
0.75 m cells (§1.1); at 0.25 m cells the dedup factor on ELECTIONS is the same (occupancy
union is resolution-independent in coverage terms) while granularity stays parity. Plus the
[48,60] near-gate removes the seam double-draw. **Expected: oblique −2..−5 ms (mid 3), eye ~0
(the eye ring is behind the <45 m mesh canopy — already voxbocc-culled,
doc 06 §3.1 eye note), aerial 0** (aerial is 100% 64 m tiles, doc 15 PA2 — untouched).

### 4.2 CPU/live

Splat on workers (persistent 2-pool), emit ≤1 tile/frame ≈ ≤5 ms on a thread with 15 ms
headroom (doc 20 PA3), message traffic ~2/s. Live gate: slot histogram + 0 longtasks (§7.4).

### 4.3 Memory (nanite.mb 1353 today)

Brick density at 0.25 m ≈ 1.22·3^2.39 ≈ 16.8 bricks/m² (all pyramid levels) ⇒ resident ring
≈ 65 k m² × 16.8 ≈ **1.1 M bricks ≈ 40 MB actual**; pool reserve 320 slots × 6144 =
1.97 M bricks = **71 MB** (+ 320×96 clusters ≈ 2.5 MB records) ⇒ brick buffer
4.03 M + 1.97 M = 6.0 M × 36 B ≈ **216 MB** — under even the conservative 256 MB reading
(§1.3), and nanite.mb ≈ 1353 → ~1430. Per-slot cap 6144 ≈ 1.4× the 4.3 k/tile mean; measured
p100 re-sizes it in Stage R1 validation (`?ringcap`). `?ringcell=0.1875` fallback: ×~2 bricks
⇒ reserve ~140 MB ⇒ ~285 MB total — needs the 4 GB limit reading confirmed OR outer-annulus
trim; state this to the user if that fallback is ever proposed.

---

## 5. Correctness + quality-equivalence argument

1. **Coverage/holes**: seams are the proven center-distance overlap construction (§3.3) — the
   same algebra that ships today at 94 m; boundary crowns duplicated by reach-bucketing
   (FarTiles.ts:153–189). The near-gate drops only provably per-tree-covered clusters (§3.5).
   Streaming failure degrades to TODAY'S renderer exactly (§3.8) — conservative by
   construction.
2. **Cull correctness**: ring heads are ordinary voxel heads (matClass 7, MESH_FLAG_HASDAG,
   identity instance) — kSeedRoots (NaniteCull.ts:760–806), traverse cut (τ_eff cap includes
   them via the matClass test :852–860), fanout, per-block + per-brick occlusion, voxcell are
   all UNCHANGED code paths already exercised by the 812 shipped 64 m tiles. In-place slot
   rewrites follow the attachHeightDagTile upload discipline (§3.4) — records + pushRange, no
   partial-state frames (mesh record is pushed LAST; until then rootCount=0 hides the slot).
3. **Granularity parity**: visible element (cell) ≤ 6 px in both representations under the
   same τ machinery; ring cell = species-mean per-tree L0 brick at the seam (§3.1). No pixel
   can become coarser than the accepted per-tree look bound; the residual difference class is
   octave-phase + sub-6px resampling, not "massive voxels".
4. **Shading identity**: per-brick albedo = source-brick mean; `?voxjit` world-anchored;
   crownDir field reconstructed per SOURCE TREE via the word-3 origin bake (§3.7); vox has no
   sway in either rep (identity contract transform — NaniteCommon.ts:137–148 contract, cited
   in doc 06 §2.2).
5. **What is NOT bit-equal, stated honestly**: brick boundaries (axis-aligned tile grid vs
   yaw-rotated crown grids) ⇒ elected depth keys shift within a brick, silhouettes re-dither
   at the ≤6 px cell scale; election tie-order differs (id8 tiebreak,
   NaniteRaster.ts:952–953 idiom). Therefore this is **Class R** (doc 90 §4): the gate is the
   §7.3 shotdiff protocol + explicit user sign-off — never self-certified, no default flip
   before it.

---

## 6. Staged landing plan (every stage flag-gated + independently measurable)

- **R0 — size the pool (no renderer change).** Run probes G0a/G0b (§7.1). STOP if the
  post-voxbocc ring residual < 2 ms (surface to user; the lever's premise would be dead).
- **R1 — `?ringtiles=1`**: splat/emit parameterization (§3.2), slot pool (§3.4), streamer +
  handoff (§3.8), ring OCC_COVER + component prune (§3.6; `?ringocc`, `?ringcc`). Sub-flags
  `?ringcell`, `?ringnear`, `?ringcap`. Measurable: boot log
  `[ringtiles] N slots, cap B, resident R, skipped S`, counters, gates G1/G2.
- **R2 — `?ringbead=1`** (origin bake + resolve crownDir, §3.7). Measurable: shotdiff delta
  at oblique/eye vs R1 (expect strictly smaller diff); zero perf cost expected (same ALU path).
- **R3 — `?ringgate=1`** (fanout near-gate, §3.5). Measurable: voxClusters delta at eye/oblique
  + gate G1 rerun; quality strictly improving (removes coarse bleed).
- **R4 — default-flip proposal**: only after G1+G2+G3 pass AND user sign-off on the shotdiff
  set. Flip = `ringtiles` default-on in ForestScene; keep `?ringtiles=0` as the A/B reverter.

---

## 7. Measurement gates (exact commands + decision rules)

All A/Bs same-session interleaved, reversed-order repeated (thermal trap, doc 90 §6.3); no
delta < 2 ms believed without two agreeing runs (apparatus floor, doc 90 §3.7). Judge oblique
with mode structure/p25–p75, not bare medians (doc 90 §6.6).

### 7.1 G0 — pool sizing (BEFORE building anything)

    # G0a: post-voxbocc ring residual (doc 06 P-A)
    CONFIG=default EXTRA=aggdist=60 LABEL=g0a-aggdist60 TREES=200000 TICKS=0 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts
    CONFIG=default LABEL=g0a-base TREES=200000 TICKS=0 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts

Decision: Δoblique(base − aggdist60) ≥ 4 ms ⇒ proceed; 2–4 ⇒ proceed but re-rank vs other
levers; < 2 ⇒ STOP (§0 rule 2). If the tile-wave lever (doc 06 L1) has landed by then, run G0a
on top of it — the ring estimate in §4.1 assumes the post-L1 world.

    # G0b (optional, sharper): tile-side share via the 5-line ?ftskip diagnostic (doc 06 P-B)
    CONFIG=default EXTRA=ftskip=1 LABEL=g0b-ftskip TREES=200000 TICKS=0 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts

### 7.2 G1 — perf A/B (after R1, rerun after R3)

    CONFIG=default LABEL=g1-off TREES=200000 TICKS=0 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts
    CONFIG=default EXTRA=ringtiles=1 LABEL=g1-on TREES=200000 TICKS=0 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts
    # then the same pair in reversed order

Decision (ALL required):
- oblique med gpuWall Δ ≤ −2.0 ms in both orderings (target −3);
- eye and aerial Δ within +0.5 ms;
- **engagement counters**: oblique `nanite.voxClusters` drops from ~9.2 k toward
  ~4.5–6 k (per-tree ring clusters gone, ring-tile clusters added) — if unchanged, the
  mechanism never fired ⇒ null is INVALID (doc 90 §6.2); `nanite.visTris` ~unchanged
  (bark untouched — the control that no triangle-side confound entered);
- boot Δ ≤ +3 s; nanite.mb ≤ 1500; oversized-skip counter < 2% of tiles.

### 7.3 G2 — the shotdiff + sign-off protocol (quality, blocking for R4)

New `tools/probe-ringtiles-shotdiff.ts`, cloned from tools/probe-treeconnect-shotdiff.ts
(boot/pose/screenshot/diff machinery :25–70): boot OFF (default) vs ON (`ringtiles=1`,
then `ringtiles=1,ringbead=1`), same seed, settle(120) + settle(40) per pose
(:38–47), poses = 3 canonical (:19–23) + 2 stress poses:
`graze {p:[0,3,0], yaw:0.6, pitch:0.0}` (tree-line silhouette through the ring) and
`midring {p:[0,20,60], yaw:0, pitch:-0.15}` (ring fills the frame). Report per pose: full-frame
mean abs diff, ring-band crop diff (oblique: middle 30–70% rows), max 32×32-block mean diff
(localized artifact detector = floaters/holes/seams), + amplified diff PNGs + side-by-side crops
at 60 m and 140 m seams.

Decision: auto-REJECT (fix before showing the user) if any pose's ring-band mean abs diff
> 2.0/255, or any 32×32 block mean > 12/255, or any visible floater/hole in the amplified
diff. Passing thresholds is NOT acceptance — package the shots and get explicit user sign-off
(Class R, doc 90 §4). Sweep ladder if the user rejects: `ringbead=1` (if not already) →
`ringocc` 0.30/0.40 → `ringcell=0.1875` (+memory note §4.3) → `ringnear=75` (shrinks the win;
re-run G1) → surface "lever quality-rejected" with the evidence.

### 7.4 G3 — live/streaming gate (after R1)

    CONFIG=default EXTRA=ringtiles=1 LABEL=g3-live TREES=200000 TICKS=2200 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts

Decision: 0 longtasks; slot histogram (fraction of frames ≤2/≤3 slots) no worse than the
paired default run (slot math, doc 90 §2.1/§6.5); `nanite.ringFlips` = 0 after warm-up during
the glide (hysteresis works); per-frame array shows no NEW mode (bimodality interaction check,
doc 90 §7 P3 context).

---

## 8. Risks + fallbacks

| risk | detection | fallback |
|---|---|---|
| Post-voxbocc/post-L1 ring residual already small | G0a < 2 ms | STOP before building (§0.2) |
| Emit hitches (main-thread pyramid) | G3 longtasks / slot regressions | ≤1 emit/frame budget already; then extract buildVoxelPyramid three-free (its imports are pure — doc 06 L3) into the worker |
| Per-slot cap overflow on dense clumps | skip counter in G1 | raise `?ringcap` (memory re-check §4.3) — safe meanwhile (slot skipped ⇒ per-tree fallback holds) |
| Handoff flip visible as a field "pop" | G2 graze/midring shots at the flip; ringFlips counter | flip only when resident (already) + hysteresis; if still visible, per-SPECIES staggered flip over 4 frames |
| Coarse ring brick wins vs finer per-tree brick in [48,60] | G2 midring crop | R3 near-gate removes the overlap band entirely |
| Fine-cell floater regrowth | G2 block-diff + skyline crop | `?ringocc` up, `?ringcc` component prune is the root-cause fix (§3.6) |
| Shading identity insufficient (crowns read flat/merged) | G2 with vs without `?ringbead` | ringbead is the designed fix; then `ringcell=0.1875`; then user call |
| Oblique win < 2 ms with counters proving engagement | G1 | genuine null: the dedup share was over-estimated — revert default, keep flag, record in the review corpus + memory (do NOT retry aggdist/coarsen variants — red-listed) |
| Worker/pool races (slot rewritten while visible) | flicker in G3 | mesh record pushed last (§5.2); evict→attach reuses a slot only via the free-stack (single-threaded residency manager on the main thread) |

Expected net at the canonical poses if all gates pass: **oblique −2..−5 ms (mid −3), eye ~0,
aerial 0**, memory +~75 MB reserved, boot +≤3 s, live slot histogram unchanged-or-better.
