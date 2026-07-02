# lod-cut deep review (2026-07-02)

Area: LOD ladders + cut math for both foliage tiers and the mesh DAG — `VoxelizeCrown.ts`
(voxel pyramid + errorK + K_FLOOR), `FarTiles.ts` (tile ladder), `BuildAggregateDag.ts`
(leaf ladder), the cut evaluation in `NaniteCull.ts` (makeTraverse), and the τ plumbing in
`NaniteFrame.ts`. All numbers below at the canonical config: dpr 1.5 → 2268×1473, fov 55°
(`src/core/Engine.ts:51`), so **projK = cot(27.5°)·1473/2 ≈ 1414.8 px** — verified identical
to the cull's formula (`NaniteCull.ts:362`).

## Premise audit

1. **The fact pack's description of the vox ladder is one generation stale.** The focus
   brief names "anchorL0/tau/projK derivation" as the live mechanism. It is NOT: the
   band-anchored anchorL0 ladder was superseded by "FIX B" — `buildVoxelPyramid` sets
   `ownError(L) = curCell·BRICK_DIM·0.5·errorK` (the real local-space brick HALF-extent,
   projK-independent, `VoxelizeCrown.ts:988`). `VOXLOD_CFG.anchorL0`, the
   `computeVoxlodAnchorL0()` computation, and the anchor blocks in `ForestScene.ts:110-129`
   / `WorldRegistry.ts:399-414` are **dead configuration** — `voxlodAnchorL0()` has zero
   consumers in the pyramid build. ~8 header comments in VoxelizeCrown.ts (lines 63-79,
   93-99, 109-115, 216-218, 251-252, 672-673, 1151) still document the dead ladder. Doc-rot
   that actively misled this review's own brief.

2. **"errorK=1 calibrated one brick ≈ τ px" is off by up to 2×.** pOwn projects the brick
   HALF-extent; a level is used for d ∈ [T(L), 2T(L)), so an emitted brick subtends
   **12-24 px** (τ_vox=12), sawtoothing across each octave band — 2τ at band entry, τ at
   the far edge. The mean emitted brick is ~17 px wide, not 12.

3. **The effective vox τ is voxTauCap=12, not loderr=3.** The lodWarp
   (τ_eff = 3·(1+((d−4)/6)^0.6), `NaniteCull.ts:111-126`, defaults `NaniteFrame.ts:172-177`)
   exceeds 12 at d ≥ 41.4 m; since the vox band starts at 45 m, **every voxel cluster in the
   scene is cut against the constant cap 12** (`NaniteCull.ts:317-318, 852-859`). The base
   τ=3 and the warp shape are irrelevant to voxels; they only shape the mesh (leaf <45 m,
   bark <140 m) cut.

4. **The user's "mid-band too coarse" is NOT a cut-selection failure — it is the tile
   ladder's missing bottom.** Every level that has a finer sibling is τ-bounded ≤24 px. But
   a ladder's FINEST level is unbounded from below: far tiles' L0 bricks are 3 m
   (ftcell 0.75 × 4, `ForestScene.ts:317`), and tiles engage from nearDist = 94 m
   (`ForestScene.ts:390`). A 3 m brick projects **45 px at 94 m, 30 px at 140 m**, and only
   falls to the 24 px τ-envelope at ~177 m. The 94-354 m band renders tile-L0 everywhere
   (T_tile(1) = 354 m). So the worst blockiness band (140-354 m) **cannot be improved by any
   ladder/errorK/τ knob** — those only shift which level is picked, and L0 is already the
   finest that exists. The fix must add finer tile content (tiered tiles) — see Levers.

5. **Second artifact axis: shading granularity ≠ geometry granularity.** The cut metric
   bounds geometric error; ?voxcell carves silhouettes to brick/4 cells (3-7.5 px). But
   shading is per-brick constant (voxbn: one normal+albedo per brick) → the perceived block
   size is the full 12-45 px brick. The §3 far-band diagnosis (attribution doc) confirmed
   flat FILL, not silhouettes, reads as blocks. ?voxbead/?voxjit (shipped, default-on)
   attack exactly this for ~free — they should be eyeballed before any octave of ladder
   engagement is funded, because an octave buys a 2× shading-granularity improvement for
   +6..15 ms while bead/jit buys an approximation of it for ~0 ms.

6. Metric check: pOwn uses denO = √(d²−r²) (distance to sphere edge, conservative,
   `NaniteCull.ts:834`) and A.w scales ownError per instance — both correct and consistent
   with the mesh DAG's qemErr units (object-space metres).

## How it works today

### The cut (one mechanism for everything)

`NaniteCull.ts` makeTraverse (`:811-948`): BFS from per-mesh roots; per frontier item
compute `pOwn = projK·A.w·ownError/denO` (`:835`); emit iff `pOwn ≤ τ_eff` (then frustum /
minPx / cone / shadow-hollow / prev-frame-HZB tests, `:861-934`), else enqueue children with
NO gating (`:935-947`). τ_eff = lodWarp(τ=3, d) (`:836-842`), clamped to `voxTauCap=12` for
VOXEL_MATCLASS clusters (`:852-859`). Instance-level gates in kSeedRoots: frustum, draw
envelope (`:771-779`), per-mesh nearDist (`:786-787` — the 45 m mesh→vox and 94 m tile
handoffs), instMinPx (default 0.075·1473 ≈ 110 px, `NaniteFrame.ts:187-191`).

### Ladder 1 — per-tree voxel crowns (45-140 m)

Built by `buildVoxelPyramid` (`VoxelizeCrown.ts:949-1187`): L0 = the 180-cell crown grid
(`cellSize c = maxExt/180`, `:426`), each level 2× coarser via `downsampleBrickGrid`
(`:715-842`, tight symmetric-half footprints + re-binned occupancy), stop at
`K_FLOOR = 3` bricks on the largest axis (`:1006-1007`). `ownError(L) = 2c·2^L` for L≥1,
L0 = 0 (always-emit, `:988`). Sparsity multiplier default OFF (`:107`), shell default OFF
(`:176-181`). Blocks = ≤128-brick clusters (`partitionLevelBlocks :904-941`), spatial
parent tree + coverage repair, gated by validateDagHierarchy (`:1181-1186`). Blocks carry
ownError into GPU DAG records via `appendVoxelCrownPyramid` (`:1436-1504`).

Level transition distance: **T(L) = projK·A.w·2c·2^L / 12**. For a typical crown
(maxExt ≈ 10 m → c ≈ 0.055, A.w = 1.1): T(1) ≈ 28.5 m, T(2) ≈ 57 m, T(3) ≈ 114 m,
T(4) ≈ 228 m. A.w spread 0.8-1.4 smears each boundary ±27%.

Consequences:
- **L0 never renders** for typical crowns (T(1) ≈ 21-36 m < the 45 m band start) — the
  finest per-tree level is resident-but-dead ammunition (~60-70% of the 124,417 per-species
  bricks, ~3 MB; trivial memory, but it means a ring-octave spend needs NO rebuild).
- The 45-140 band renders L1 (to ~57 m), L2 (~57-114), L3 (~114-140), mixed ±1 by A.w.
- L4 (the K_FLOOR shell) never engages (T(4) ≈ 166-290 m > 140 m maxDist,
  `ForestScene.ts:376`). K_FLOOR is a no-op for per-tree crowns in the forest config.

### Ladder 2 — far tiles (94 m → horizon)

`FarTiles.ts`: 64 m tiles, 0.75 m cells → L0 bricks 3 m (`emitTile :214-242`,
half = brickWorld/2); pyramid via the same `buildVoxelPyramid` (`:246-251`) → grid
22×16×22 halves to L3 (3×2×3, K_FLOOR stop). ownError(L) = 1.5·2^L m; identity instance
A.w = 1 (`appendFarTiles :361-380`, maxDist 100000). **T_tile(L) = 176.9·2^L m: L0 owns
94-354 m, L1 354-707, L2 707-1415, L3 1415+.** The 200k forest half-extent is ~896 m
(corner 1267 m), so in-scene tiles only ever show L0-L2 — K_FLOOR/L3 is unreachable, and
the "never one square" outcome at the visible horizon is provided by L2 (12 m bricks, 6×4×6
grid), not by K_FLOOR. K_FLOOR would only matter in a >1.4 km world, where tile L3 = 24 m
bricks in a 3×2×3 grid — multi-brick, but the Y axis bottoms at 2 (max-axis flooring), which
is where the far-skyline "trunk antenna" thin-pillar risk lives.

### Ladder 3 — aggregate leaf mesh (0-45 m)

`BuildAggregateDag.ts`: per level remove ~half the leaf-island area, grow survivors
(area-preserving, growMax 2.5); `growError = mean kept island radius × grow` (`:297-307`),
reported error = `growError × leaflodk(0.4)` (`:504`, set `ForestScene.ts:150`). Cut against
the UNCAPPED warp τ_eff (3→12.5 px across the band). Solving pOwn = τ_eff with the measured
L1 growError ≈ 0.5 m: L1 engages ≈ 30 m, L2+ mostly beyond the 45 m handoff — the leaf band
is L0 to ~30 m. Perf-irrelevant post-voxbocc (eye foliage total ≈ 2.1 ms; leaf decode
measured ~free), so leaflodk is a quality knob only; its known pose-trade (0.25 → oblique
+7) makes LOWER values doubly dead.

### Quantified: what the cut picks and what one brick projects (dpr 1.5, A.w=1.1, c≈0.055)

| d (m) | rep + level | brick world (m) | brick px | ?voxcell cell px | shading-block px |
|---|---|---|---|---|---|
| 60  | tree L2 | 0.97 | 22.8 | 5.7 | 22.8 |
| 100 | tree L2 | 0.97 | 13.7 | 3.4 | 13.7 |
| 140 | tree L3 **and** tile L0 (overlap) | 1.94 / 3.0 | 19.6 / 30.3 | 4.9 / 7.6 | 19.6 / 30.3 |
| 200 | tile L0 | 3.0 | 21.2 | 5.3 | 21.2 |
| 400 | tile L1 | 6.0 | 21.2 | 5.3 | 21.2 |
| (94-177) | tile L0 | 3.0 | 24-45 | 6-11.3 | 24-45 |

(45 m band entry: tree L1, 0.48 m, 15.2 px. Every brick ≥ 8 px across passes the
voxcellmin=64 px² carve gate, `NaniteVoxelRaster.ts:301-302`.)

The table makes the artifact geography exact: granularity is a flat ~12-24 px everywhere
EXCEPT the 94-177 m tile-L0 shelf (24-45 px bricks) — the handoff at 140 m jumps shading
blocks 19.6 → 30.3 px and cells 4.9 → 7.6 px mid-screen at oblique. That is the band the
eye lands on in the oblique money shot.

### Cut hysteresis / TRAA interplay

There is none, and none is needed: the cut has no temporal state; pOwn depends only on
camPos/A.w/ownError (all jitter-independent — the TRAA mirror jitters the projection matrix
only, `NaniteFrame.ts:383-440`), so level selection at a static pose is bit-stable. Jitter
enters ONLY the frustum planes + prev-frame HZB tests (sub-pixel wiggle on borderline
occlusion). The oblique bimodality already survived TAA-off (jitter frozen) → jitter
refuted. One cut-adjacent oscillator remains untested: **prev-frame-HZB feedback**
(frame A's emitted set builds the pyramid that culls frame B → a 2-state limit cycle is
structurally possible). `?occl=0` at a static pose discriminates it (probe §Open).

## Work model

- Vox cluster count(pose) ≈ Σ_trees blocks(level(d_tree)) + Σ_tiles blocks(level(d_tile));
  occupied bricks per level fall ~4×/level (shell-ish crown), blocks = ceil(occ/128).
  Model check at oblique: ~725 visible ring trees × ~4-10 blocks + ~170 visible tiles ×
  ~30 blocks ≈ 9k ↔ measured 9188 vox clusters. Eye 5893 (ring occluded → voxbocc kills
  the bricks but blocks still traverse), aerial 659 (steep → few tiles in frustum).
- Painted-pixel area is ~invariant to the level picked (τ cap bounds brick px; silhouette
  is silhouette) → cost = coverage(Mpx) + per-brick setup·brickCount + carve cost. This is
  why errorK sweeps move oblique strongly (brick count × ~4/octave in the affected band)
  while the pixel-law slope (8.1 ms/Mpx oblique) stays put.
- Measured ladder slope: errorK 1→2 direct A/B oblique 38.5→55.9 (+45%/octave, pre-fartiles
  era); voxlodk 0.7 inversion on the vox coverage share (27.8 ms) gives ×1.71/octave.
  Consistent.
- Band shares (pre-voxbocc, oblique 43.2): ring 60-140 ≈ 10.7 ms (aggdist=60 A/B), the rest
  of vox ≈ 17, base+post ≈ 15.4. Post-voxbocc oblique 37.2 → vox share ≈ 21.8 (scale ≈0.78).

## Waste inventory

1. **94-140(-185) m DOUBLE-DRAW ring** (the only real recoverable waste found): tile
   nearDist = aggdist−46 = 94 m because the seed gate tests the tile CENTER
   (`NaniteCull.ts:786-787`) and a >140 m tree can sit in a tile whose center is at 94.75 m
   — the overlap is geometrically required AT INSTANCE granularity. But at BLOCK granularity
   it is not: any tile block whose own-sphere lies entirely nearer than
   (aggdist − maxCrownReach ≈ 133 m) can only contain splat from trees the per-tree path
   already draws. Today those blocks traverse, occlusion-test, and raster; ~4-5 visible ring
   tiles × ~3.3k L0 bricks ≈ 13-16k bricks/frame at oblique fight elections against the
   finer per-tree rep (and sometimes win = a quality DEFECT). Est. 1-3 ms oblique, ~0-0.5
   eye (voxbocc already kills the occluded share), 0.5-1 aerial.
2. **Dead ladder rungs**: per-tree L0 (never emitted, T(1) < 45 m for typical crowns) and
   L4 (T(4) > 140 m); tile L3 (T(3) = 1415 m > scene extent). Memory-only (~3-4 MB total);
   L0's deadness is what makes the ring-octave spend build-free.
3. **lodWarp pow() per vox BFS item** — always clamps to 12 in-band; ~10-60k items/frame ×
   a few ALU ≈ sub-0.1 ms. Not worth a lever (a const-τ shortcut is not even strictly
   identical for crown blocks poking nearer than 41 m).
4. **Ungated descent** (`:935-947`): refused clusters enqueue children without frustum/HZB
   tests. Bounded by measured cull cost ~0.2 ms — not a lever.
5. **The 12→24 px sawtooth**: at any distance, crowns just past a T(L) boundary render 2×
   coarser than crowns just before it. Not removable within an octave pyramid (√2 levels
   don't exist for grids); UE5-style continuous error is a mesh luxury. Listed for
   completeness, no lever.

## Levers

Sign convention: positive expectedMs = saved; NEGATIVE = deliberate quality spend (the user
ruling asks these to be costed and funded by wins elsewhere).

### lod-cut:tile-ring-block-nearcull — kill the double-draw ring at block granularity
- **Mechanism**: per-cluster near cull for fartile heads at emit time: drop when
  `distC + s.radius < blockNearDist` with blockNearDist = aggdist − maxCrownReach (a new
  per-mesh word or a uniform gated on the fartile matclass). Complement of the existing
  far envelope; exact same conservative idiom.
- **Quality class**: improving — removed pixels are places the 3 m coarse rep currently
  outbids the 0.5-2 m per-tree rep inside its own band; A/B flips show fine-rep pixels.
- **Expected**: eye +0.3 / oblique +2 (1-3) / aerial +0.7 ms.
- **Discriminator**: diagnostic `?ftnear=140` (raise tile nearDist; corner holes accepted
  for measurement only) upper-bounds the win before any build.
- **Effort**: M (S for the diagnostic flag). **Risks**: reach constant must be conservative
  (derive from planFarTiles' reachOf, `FarTiles.ts:153-165`) or corner crowns hole;
  shotdiff-gate the ring.

### lod-cut:pertree-ring-octave — SPEND: engage every per-tree level one octave farther
- **Mechanism**: scoped errorK=2 for per-tree crown pyramids only (set before
  `prepareVoxelCrown`, reset before the FarTiles pyramid build — both read `voxlodErrorK()`
  at build). Band becomes L0(45-57)/L1(57-114)/L2(114-140): bricks 6-12 px instead of
  12-24, shading blocks halve, and the 140 m handoff jump shrinks (L2 0.97 m vs tile 3 m
  instead of 1.94 vs 3). Uses ALREADY-RESIDENT bricks (dead L0) — no memory, no rebuild.
- **Quality class**: improving (strictly finer; the user-ruling direction).
- **Expected COST**: eye −1 / oblique −6 (±2) / aerial −0.5 ms. Derivation: ring share
  10.7 ms (aggdist60 A/B) × 0.71/octave × 0.78 voxbocc-era scale; eye ring is
  canopy-occluded post-voxbocc so it pays little.
- **Discriminator**: new `?voxlodkring=2` flag; A/B vs default, thermal-ordered; verify
  tiles unchanged (cluster counters) + far-band shots.
- **Effort**: S. **Risks**: slope may run hotter than 1.71 (L0 is the volume-dense level —
  blocks/tree jump ~4→10+); if oblique cost >8 ms it must wait for funding.
- NOTE: **global** errorK=2 (`?voxlodk=2`) is measured at +17.4 ms oblique and additionally
  extends tile L0 to 707 m — the wrong shape; do not fund the global knob.

### lod-cut:tiered-mid-tiles — SPEND: the only fix for the real artifact band (140-354 m)
- **Mechanism**: second tile tier for the ~140-300 m ring: 32 m tiles at ftcell 0.375-0.5
  (worker splat has boot headroom), per-ring OCC_COVER; 64 m/0.75 tiles keep 300 m+. Ring
  L0 bricks 1.5-2 m → 15-20 px at 140 m (inside the τ envelope) instead of today's 30 px
  shelf; cells 3.8-5 px.
- **Quality class**: improving — directly removes the worst user-visible coarseness band,
  which NO ladder/τ knob can reach (premise-audit #4).
- **Expected COST**: eye −0.5 / oblique −3 / aerial −1 ms (ring brick count ×4-8 on ~69
  tiles; carve/setup-bound, painted area invariant).
- **Discriminator**: before building — `TREES=100000 EXTRA=ftcell=0.5` (smaller extent fits
  memory) for look+cost scaling of finer cells at the handoff.
- **Effort**: L. **Risks**: memory (global 0.5 m cells = 386 MB > 256 MB cliff — the ring
  scope is mandatory); low-coverage edge floaters (needs the per-ring OCC_COVER); double
  handoff seams at 300 m.

### lod-cut:cut-telemetry — make the ladder observable (enables exact calibration)
- **Mechanism**: boot print per species: maxExt, cellSize, per-level occupied/blocks, T(L)
  at A.w=1; plus one `?nanitedbg=lod` screenshot set at the canonical poses (dagLevel tint
  already plumbed, `VoxelizeCrown.ts:1490-1492`). Zero runtime cost, no output change.
- **Quality class**: identical (instrumentation; no pixel change in the shipped path).
- **Expected**: 0 ms — it de-risks the two spends above (T(L) currently carries ±30%
  species/A.w uncertainty; large crowns may already show L0 at 45-59 m).
- **Effort**: S. **Risks**: none.

### REJECTED-BY-POLICY (quality-trading; attribution instruments only — listed for completeness)
- `voxlodk<1` (0.85/0.7 measured −2/−6.8 oblique) — look rejected by user.
- `aggdist<140` (60 → −10.7 oblique) — "dogshit, massive voxels".
- `voxtaucap>12` (bigger far bricks) / `loderr>3` / warp reshaping toward coarser.
- `leaflodk<0.4` — spiky crowns + measured oblique pose-trade.
- `ftcell>0.75`, `instminpx` raises, dpr — all banned quality knobs.

## What UE5/prior art does here

- **Nanite meshes**: continuous cluster DAG with ~1 px screen-error τ at full res — no
  octave sawtooth (levels are ~1.3-1.5× steps, cut per cluster-group), and traversal culls
  (frustum+HZB) at every BVH NODE during descent, not only at emit. Their error metric is
  also purely geometric; shading granularity is never coarser than a pixel because clusters
  are triangles, so the "per-brick flat fill" artifact class doesn't exist for them.
- **Voxel far-field**: closest analogs are SVO-DAG LOD chains (Kämpe et al.) and brickmap
  renderers — all octave ladders like ours; nobody has sub-octave voxel LOD. The standard
  answer to our tile-L0 shelf is a finer-brick mid tier (= tiered tiles / cascaded brick
  grids, same shape as virtual-shadow-map clipmaps).
- **HLOD** (UE's cross-instance merge) = FarTiles' ancestor: UE builds per-cell merged
  proxies with a fixed transition RADIUS per cell and dither-crossfades the handoff over
  ~0.5 s to hide the rep jump; the per-instance/per-proxy double-render during the fade is
  accepted and bounded by the fade window, not left permanently on like our 46 m overlap.
- **Foliage**: UE 5.4+ Nanite foliage uses the same "preserve area" leaf thinning our
  aggregate DAG copies; their voxel-imposter experiments (Fortnite biomes talk) keep
  per-cell albedo/normal (not per-brick), i.e. their shading granularity = geometry
  granularity — the direction our voxbead/voxjit approximates cheaply, and a per-CELL
  attribute store would complete (raster/resolve area, not cut area).

## Open questions + proposed serial probes (exact command lines, GPU-serialized outside)

1. **Double-draw upper bound** (needs the S diagnostic flag `?ftnear=` in appendFarTiles):
   `CONFIG=default EXTRA=ftnear=140 LABEL=ftnear140 TREES=200000 TICKS=0 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts`
   vs same-session default. Gap = lever 1's ceiling. (Diagnostic only — corner holes.)
2. **Ring-octave true cost** (needs `?voxlodkring=`):
   `CONFIG=default EXTRA=voxlodkring=2 LABEL=ring-oct TREES=200000 TICKS=0 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts`
   — run AFTER a fresh default baseline; gates: oblique delta vs the −6 estimate, far-band
   shots, unchanged tile cluster counts.
3. **Occlusion-feedback bimodality** (cut-adjacent, unresolved):
   `CONFIG=default EXTRA=occl=0 LABEL=occl0-bimodal TREES=200000 TICKS=0 FRAMES=120 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts`
   — if the ~4-frame oblique runs vanish, the prev-frame HZB/cut feedback loop is the
   oscillator (worth 4-9 ms at the two worst poses).
4. **Level-band ground truth**:
   `CONFIG=default EXTRA=nanitedbg=lod LABEL=lodtint TREES=200000 TICKS=0 COOLDOWN_S=30 npx tsx tools/probe-fresh-stutter.ts`
   — shots confirm the computed T(L) bands + the A.w smear (checks the c≈0.055 assumption).
5. **Tiered-tile look/cost scaling** (pre-build de-risk):
   `CONFIG=default EXTRA=ftcell=0.5 LABEL=ftcell05 TREES=100000 TICKS=0 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts`
   (+ ftcell=0.375 if memory holds at 100k) — does the 140-250 m band read acceptably at
   1.5-2 m bricks, and what does the brick-count increase cost?
6. Open: per-species cellSize/maxExt spread (telemetry lever); post-voxbocc band-resolved
   oblique split (ring vs tiles) — rerun `EXTRA=aggdist=60` once, post-voxbocc, to re-pin
   the ring share before spending against it.
