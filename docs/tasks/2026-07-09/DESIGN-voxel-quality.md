# DESIGN — voxel quality: over-fill (too-thick crowns) + flat-splat plane forms

Task pair (user 2026-07-09), diagnosed from code by the coordinator (Fable). Implementation gated on this
design. Shadows/voxelizer DATA untouched — both fixes are **render-side** in `NaniteVoxelRaster.ts` kVoxScatter.

## Symptom 1 — "voxels too easily created for fine detail" (crowns read THICKER than mesh)

**User hypothesis check:** "voxel size isn't the problem" — CORRECT (L0 cell ≈ 0.156 m ≈ 4.4 px at 60 m is
fine). "Some threshold too easily fills voxels" — half right: it's not `OCC_COVERAGE_THRESHOLD` (0.02) —
raising it was TRIED 2026-07-02 and refuted (VoxelizeCrown.ts:208-214: the coverage histogram is bimodal —
interior cells saturate at 1.0, single-leaf cells sit at ~0.33; thresholds 0.12–0.5 bite nothing visible, and
0.12 cost +3.5 ms by pushing bricks off the full-mask fast path). **The real mechanism is the paint semantics:
a ~33%-covered cell renders 100% OPAQUE.** One leaf plane through a cell ⇒ bit set ⇒ solid cube face. The
coverage/density that would fix it is ALREADY BAKED per brick (VoxelBrick word 4 alpha = brick mean coverage,
density-MEAN-inherited up the coarse levels) — the renderer ignores it by default.

**Fix 1 — density-driven coverage, default ON.** The dormant `?voxalpha` prototype (NaniteVoxelRaster.ts:325-341,
1287-1296, 1370-1382) is the right mechanism: an election raster cannot alpha-blend, so partial opacity =
keep a **hash-stable screen-anchored fraction** of the brick's footprint pixels (no temporal shimmer). Changes
vs the prototype:
- Default ON (drop-old-behavior law; git is the undo).
- Ramp: prototype used `alpha = 0.4 + 0.6·d/MAXD` for d < MAXD. Replace with the honest mapping
  `alpha ≈ clamp(density / DENS_SOLID, αmin, 1)` with `DENS_SOLID ≈ 0.85` (interior bricks at ~1.0 stay fully
  solid — fast path preserved) and `αmin ≈ 0.3` (never balder than the ~0.33 single-leaf coverage; no holes).
  A 0.33-coverage edge brick then paints ~35-40% of its pixels — the aggregate thins toward the mesh density.
- Coarse levels inherit density-MEAN (VoxelizeCrown.ts:774-777) ⇒ far-field thinning falls out for free.
- ⚠️ USER-LAW tension to eyeball: beautification law says "no noise tricks." This stipple is hash-STABLE
  (screen-anchored, not temporal). If the user's eye rejects the dither texture, αmin rises / DENS_SOLID
  falls until acceptable — table constants, not knobs.
- Interaction with the mesh-side width ramp (bb5b4db): the rung-5 ×1.3 widthBoost compensated the mesh UP
  toward fat voxels. With voxels thinning DOWN, the boost may need retuning (possibly back toward 1.15) —
  eyeball the 55-65 m handoff band BOTH ways; the meet-in-the-middle is the goal.

## Symptom 2 — flat-splat plane forms (near-field "1 vertical + 3 horizontal planes"; far #69 grid-gaps)

**Mechanism (memory `voxel-brick-flat-splat-bug` + code):** the per-pixel RAY→BRICK path (`?voxcell`) is
DEFAULT ON but gated to bricks with **footprint area > `?voxcellmin` = 64 px²** AND non-straddling AND (with
alpha) not alpha-active (NaniteVoxelRaster.ts:295-334, 1309-1330). An L0 brick at 60 m is ~4.4 px ⇒ ~20 px²
⇒ **most visible bricks take the FLAT path**: the screen AABB of the 8 projected corners painted at ONE depth
(bbNearZ). Flat rects at single depths ⇒ the intersecting-plane forms near, gaps/billboard reads far (#69).

**Why voxcellmin can't just be lowered (the recorded failure):** `voxcellmin=16/0` made far bricks INVISIBLE
(Quality.ts:52 note). Root cause: at small footprints the PIXEL-CENTER ray misses a box that genuinely
overlaps the pixel (classic conservative-rasterization failure) ⇒ dropped pixels ⇒ invisible/gappy. The 64 px²
gate is a workaround for a fixable bug, not a law.

**Fix 2 — conservative small-brick ray.** Extend the ray path down by making the test conservative:
- Per pixel: ray vs the brick AABB **dilated by ~half the pixel's world-space footprint at brick depth**
  (one madd on the half-extent: `hf' = hf + 0.5·pixelWorld(depth)`), depth = clamped slab entry t.
  Hit ⇒ elect at the true per-pixel cube depth (kills single-depth planes AND tiles adjacent bricks — heals
  #69's gaps); miss ⇒ skip (true silhouette instead of the AABB-rect overpaint).
- Banding for cost control (the +3.5 ms lesson): area ≤ ~6 px² → keep the flat splat (a ~2 px brick IS a
  splat; cube-ness invisible); 6–64 px² → the NEW conservative ray; > 64 px² → existing ray path unchanged.
  Constants fixed in code (no new knobs); `voxcellmin` keeps meaning "the existing-path threshold".
- Straddlers (near-plane-crossing bricks, currently excluded ⇒ the worst ?voxnear forms): clamp the ray
  origin to the near plane and slab-test the remainder; if genuinely degenerate, keep the flat fallback for
  straddlers only.
- Alpha-active bricks NO LONGER skip the ray path (the prototype's exclusion at :1317-1319 was "front-slab is
  fine for a soft edge" — but near-field edge bricks are exactly where the plane forms show). Accept =
  conservativeRayHit AND hash<alpha; depth = ray entry.

## Validation plan
1. `?voxnear=1` close-up: bricks read as CUBES (3 visible faces, stable under camera motion), no
   vertical/horizontal plane forms. This was the original bug report's repro.
2. The 55-65 m handoff walk (the bb5b4db check, re-done): mesh→voxel density step with the voxels now
   thinner — retune rung-5 widthBoost if the meet-point moved.
3. Far-field #69 check: the on-axis fartile grid-with-gaps should visibly heal (same root).
4. Census re-run (`?census=1`) — expect voxel-side counters stable (this is per-pixel paint, not brick
   count); p95 must not regress (the flat→ray band adds slab ALU on 6-64 px² bricks; the honest-silhouette
   paint REMOVES overpainted AABB-corner pixels — plausibly a wash or win; MEASURE).
5. Fresh Xcode profile only after the eyeball passes (standing rule).

## Explicitly out of scope
- Voxelizer data/format changes (density is already baked; occupancy semantics stay).
- The voxel far-field LOD-continuum disease (voxel-lod-only-two-levels) — separate arc.
- Shadow voxel path (?shvox opt-in, never worked) — untouched.
