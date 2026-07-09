# FARTILE see-through stripe gaps — verified diagnosis (task #69)

**Status:** mechanism VERIFIED by a node run over the shipping code (no GPU). Root cause is a
single **grid-origin corruption in `buildVoxelPyramid`**, triggered only by `FarTiles.emitTile`.
Fix is a 4-line deletion. **No source was edited (read-only diagnosis).**

---

## TL;DR

`buildVoxelPyramid` (`src/nanite/VoxelizeCrown.ts:1252-1255`) throws away the correct `origin`
it was passed and instead **recovers the grid origin from `l0Bricks[0].center`**:

```ts
let gridOrigin = origin;
if (l0Bricks.length > 0) {
  const b0 = l0Bricks[0] as BrickCPU;
  gridOrigin = [b0.center[0] - 0.5 * l0World, b0.center[1] - 0.5 * l0World, b0.center[2] - 0.5 * l0World];
}
```

This assumes brick index 0 (the tile-cube corner brick, grid cell `(0,0,0)`) carries a valid
grid-center. That holds for `voxelizeCrown` (its empty bricks get their true `bCenter`,
`VoxelizeCrown.ts:749-752`) but **NOT for `FarTiles.emitTile`**, whose empty bricks are the shared
`EMPTY_BRICK` with `center: [0, 0, 0]` (`FarTiles.ts:78-87`, assigned at `:270`).

The tile-cube corner brick `(0,0,0)` — bottom-back-left, at ground/trunk height in the tile corner —
is **empty in most tiles**. When it is, `buildVoxelPyramid` recovers
`gridOrigin ≈ [-1.455, -1.455, -1.455]` instead of the true `[-32, 0, -32]` — an offset of
**+30.5 m** (≈ half a 64 m tile). Two consequences at every COARSE pyramid level (L≥1):

1. **Occupancy collapse → axis-aligned see-through stripes.** Every occupied child box, re-expressed
   in the mis-placed `[center ± half]` frame, lands far outside `[0, 2·half)` and **clamps to sub-cell
   index 0** (`downsampleBrickGrid`, `VoxelizeCrown.ts:1037-1048`). So each coarse brick lights only
   its cell-0 layer; that empty pattern is identical and **phase-locked across every brick**, giving
   coherent see-through corridors at **one coarse-brick pitch**, worst dead-on a world axis, healed
   oblique. This is exactly the reported symptom.
2. **+30.5 m spatial offset** of the coarse LOD bricks' render positions (`center = gridCenter`,
   `:1015`) — a secondary artifact (the "mis-angled billboards" the prior fix comment names).

L0 is unaffected (it uses the dense brick centers directly; only occupied bricks render, and their
centers are correct), which is why the **near/crown band (60-280 m) is fine** — the same reason the
crown voxel path is fine (crowns never feed `EMPTY_BRICK`).

This is neither H1 (splat quantization) nor H2 (a splat post-pass) nor plain H3 (real tree rows): it
is a **pyramid re-bin origin bug**. The splat itself is clean.

---

## Discriminator run (the decisive evidence)

Node harness ran the **real** `splatTiles` + `buildVoxelPyramid` on a faithful jittered forest patch
(world defaults: tile 64 m, `ftcell 0.75` → 88³ cells, cell 0.727 m, L0 brick 2.909 m; planting = 4 m
grid, ±1.7 m jitter, scale 0.8-1.4; crown = filled ball of ~0.625 m crown-L0 bricks). It reconstructs
per-cell occupancy from the shipping `occLo/occHi` and renders the dead-on "looking along +X" view
(rows = height cy, cols = cz). A vertical stripe = a `cz` column see-through through the canopy band.

### L0 splat occupancy — CLEAN (no stripes, refutes H1/H2)
Top-down canopy is a solid wall; dead-on view is a solid `########` wall at every crown radius (only
the ragged canopy-top row and the sparse single-cell **trunk band** below crown base show gaps —
those are physical, not the bug). `see-through cz columns: 0/88`. The splat manufactures **no**
axis-aligned corridors.

### Coarse level L1 — the bug (controlled A/B, `crownR = 2.5` and `5.0` m)

| variant | `brick[0]` center | recovered gridOrigin.x | L1 lit cells | L1 see-through cols | stripe period |
|---|---|---|---|---|---|
| **A — production** (`EMPTY_BRICK`=`[0,0,0]`) | `0.000` | **−1.455 (CORRUPTED, true −32)** | 1204 | **33/44 (75%)** | **lag 4 cells = 5.82 m = 1 L1 brick pitch** |
| **B — fix** (empty brick → true grid center) | `−30.545` | −32.000 (ok) | 19827 | **0/44** | none |

Dead-on view, variant A (period-4 collapse — one lit column, three see-through, phase-locked):
```
|#   #   #   #   #   #   #   #   #   #   #   |
|#   #   #   #   #   #   #   #   #   #   #   |   ← "vertical stripes, dead-on axis"
|#   #   #   #   #   #   #   #   #   #   #   |
```
Variant B (same trees, corner brick given its true center) — solid:
```
|############################################|
|############################################|
```

The result is **bimodal on whether `brick[0]` is empty**, not smooth in crown size (crownR 4 happened
to occupy the corner → solid; crownR 2.5 and 5 left it empty → striped). That bimodality — a 16×
swing in lit cells flipping on one corner brick — is the fingerprint of the origin recovery, and it
explains why only **some tiles / spots** stripe (those whose corner brick is empty) while the symptom
is otherwise position-independent.

**Predicted vs observed period:** H1 would predict ≈ cellSize (0.5-0.7 m); H2 ≈ brick/tile pitch
(2.9 m / 64 m); H3 ≈ tree spacing (4 m). Observed = **exactly the coarse-brick pitch of the rendered
pyramid level** (5.82 m at L1; scales with level) — the signature of a per-coarse-brick occupancy
collapse, matching none of H1/H2/H3 and confirming the re-bin-origin mechanism.

### Why the prior fix (`fullPitchHalves`) only half-worked
The 2026-07-04 fix (`FarTiles.ts:302-308`) correctly closed the inter-brick **world-tiling** gaps
(tight-half → full-pitch, edge-to-edge). Its own comment concedes "the re-binned occupancy still
carves see-through" — that residual carve **is this bug**. `fullPitchHalves` makes the footprints
tile watertight, but the occupancy mask inside them is still collapsed to cell-0 stripes by the
corrupted origin, so the gate paints see-through anyway.

### Render-band note (one supported assumption)
The stripes live at COARSE levels (L≥1). Direct evidence that coarse levels render inside the visible
fartile band (>~280 m): the prior `fullPitchHalves` axis-stripe fix targeted exactly these coarse
levels and measurably helped, and the tile head's LOD ladder (`ownError(L)=curCell·BRICK_DIM·0.5`,
`NaniteCull` cut `pOwn=projK·A.w·ownError/d ≤ τ_eff`, voxel `τ_eff` capped ~4-8 px) descends to
coarse levels with distance. The exact L0→L1 handoff distance depends on live `projK`/`voxtaucap`;
the mechanism holds for whichever coarse level is on screen.

---

## Fix design

### Recommended — trust the passed `origin` (delete the recovery). Lowest risk.
`VoxelizeCrown.ts:1251-1255`: drop the `l0Bricks[0].center` override; keep `gridOrigin = origin`.

- **Correctness:** both real callers already pass the exact grid origin —
  `voxelizeCrown` passes `[originX, originY, originZ]` (`:786-792`); `emitTile` passes
  `[-tileSize/2, 0, -tileSize/2]` (`:297-309`). For each, `origin` equals what the recovery computes
  in the *non-corrupted* case (`brick(0,0,0).center = origin + 0.5·brickWorld` ⇒ recovery returns
  `origin`), so the crown path is **byte-identical** and the fartile path is **fixed**.
- **Cost:** −4 lines. **Risk:** minimal — only these two callers exist (grep-verified); neither
  relies on the default `origin=[0,0,0]`. No GPU/layout change.

### Defense-in-depth (optional, additive) — fix `FarTiles.emitTile` empties.
Give `EMPTY_BRICK` its true grid center per brick (as `voxelizeCrown` already does), instead of the
shared `[0,0,0]` singleton. This makes the recovery harmless even if it stays. Slightly more code
(per-brick empty center) and leaves the fragile recovery in place, so prefer it *only* alongside the
recommended deletion, not instead of it.

### Not the fix
Do **not** dilate/close occupancy or disable the occupancy gate for fartiles — that would paper over
a correct mechanism (the gate and re-bin are right; the origin they run against is wrong) and would
re-fatten the far field the occupancy gate exists to thin. The +30.5 m spatial offset would also
remain. Fix the origin, not the symptom.

---

## Verification suggested post-fix
1. Re-run the discriminator with the recovery deleted → expect L1 `see-through 0/44` for corner-empty
   tiles (matches variant B).
2. Eyeball the world scene `?scene=world … aggdist=280` dead-on a world axis at ~300-500 m — stripes
   should be gone; confirm no far-tile pop/offset at the L0→coarse handoff (the +30.5 m offset should
   also vanish). `?fartiles=0` remains the A/B reference.

## Files
- Bug: `src/nanite/VoxelizeCrown.ts:1251-1255` (gridOrigin recovery) + `:1037-1048` (clamp that
  collapses to cell 0).
- Trigger: `src/nanite/FarTiles.ts:78-87` (`EMPTY_BRICK` center `[0,0,0]`), `:270` (assignment),
  `:297-309` (the `buildVoxelPyramid` call that passes the correct-but-ignored origin).
- Contrast (correct): `src/nanite/VoxelizeCrown.ts:749-752` (crown empties keep true center).
