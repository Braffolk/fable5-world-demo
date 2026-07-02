# 2026-07-03 — WORLD-SCENE VOXEL-FOLIAGE HOOKUP (arc 1 of the post-beauty roadmap)

User directive: bring the voxelised-foliage stack to the standard world scene
(`?scene=world&nanite=1`); **defaults shared, coming from what the forest picked**.
Pre-arc state: the nanite-only world (old geometry hard-disabled) rendered **leafless
trees** — leaf heads (`?naniteleaf`) and the voxel transition (`?voxreg`) were both
opt-in and OFF; fartiles + BootCache were ForestScene-only.

## Shared defaults (the forest picks, promoted to engine defaults)

| knob | old | new (shared) | lives in |
|---|---|---|---|
| mesh→voxel handoff (`voxnear`) | 35 | **60** | `WorldRegistry.DEFAULT_TRANSITION_DIST` |
| crown voxel grid (`voxgrid`) | 180 | **256** | `VoxelizeCrown.DEFAULT_VOXEL_GRID_DIM` |
| fartile cell (`ftcell`) | — | **0.6** | `FarTiles.DEFAULT_FT_CELL` |
| tile size | 64 (inline) | **64** | `FarTiles.FT_TILE_SIZE` |
| aggregation distance (`aggdist`) | — | **140** | `FarTiles.DEFAULT_AGG_DIST` |
| leaf ladder K (`leaflodk`) | 1 (scene-set 0.4) | **0.4** | `BuildAggregateDag.AGG_LOD_CFG` |

ForestScene consumes the constants (no behavior change there). Engine-level beauty
defaults (voxtaucap 4, lodnear 20, simband 25, aofade 90/240, ftnrm 0.65) already
applied globally.

**⚠️ Two knobs the world could NOT share (measured, booked in WorldRegistry comments):**
- **ftcell 0.75 world** (forest 0.6): the 4 km world splats ~3510 tiles vs forest ~841;
  0.6 ⇒ ~12-13M far bricks ≈ 450 MB brick buffer (over the ~256 MB cliff) and the
  BrickCPU object graph OOM-crashed the tab at ~8M. 0.75 lands 8.0M bricks / 275 MB —
  the forest-proven scale. Revisit if the brick payload slims.
- **aggdist 280 world** (forest 140): on real terrain an oblique view puts whole
  HILLSIDES into the 94-280 m band; at 140 the fartile takeover painted them as a
  washed-out wall of 2.9 m plates (user report). 280 keeps per-tree grid-256 voxels on
  everything you look at. Perf: oblique 47→58 fps (per-tree is CHEAPER than the tile
  overlap there), far 74→62, aerial 96→67 (detail buys it back — user law), high 109→107
  (the far collapse is intact). Forest keeps 140 (flat ground only shows tile TOPS).

## World wiring (WorldRegistry / TerrainScene)

- `?naniteleaf` default **ON** (full-frame mode) — leaf crown heads + aggregate DAG.
- `?voxreg` default **ON** — per-species crown voxelization (grid 256, 7L pyramid),
  voxel:7 sibling heads, 60 m handoff. 20 crowns, 244k bricks, ~178k voxel instances.
- **Fartiles in the world**: built pre-`reg.build()` from the same perId streams
  (species level picked ≈ ftCell, forest's algorithm), appended post-build; per-tree
  bark + voxel heads clamped to aggDist. 3417 tiles, 8.0M bricks (275 MB), 95k clusters,
  9 s splat (8 workers).
- **BootCache in the world** (same IDB store, `scene:'world'` key marker + resolved
  params): crowns + dags (QEM+aggregate, one array in [toDag...,toAggregate...] order)
  + fartiles. Cold boot ~103 s → warm ~35-40 s (crowns 36 ms, dags 1 s, fartiles 0 ms
  reads). ⚠️ WorldRegistry.ts is NOT in SRC_HASH — default changes there must be
  key-visible (resolved values in params).

## FarTiles: terrain-aware + streaming (engine-level, forest-parity at y=0)

1. **Per-tile baseY** (FarTilesSplat): instance ground y (`a[1]`) is honoured — each
   tile grid floors at its members' min ground y (cell-snapped); crown bricks splat at
   `y − baseY`, trunk columns rise from each tree's own ground; the tile head's identity
   instance sits at y = baseY. Flat ground ⇒ baseY 0, bit-identical legacy path.
2. **Per-tile cellsY**: each tile's Y extent = its own relief + 48 m (brick-multiple),
   NOT a global worst-case (one alpine tile must not size every tile's dense grid —
   the first OOM). World: cellsY ≤ 392 (worst relief 235 m); forest: exactly the legacy 84.
3. **Streaming emit** (FarTiles + worker): ONE message per tile (transferred), pyramid
   built + dense arrays released as each tile lands (the old all-at-the-end transfer
   held the whole map's accumulators — OOM #2). Deterministic job-order output.
4. **Packed end-to-end on the world path**: `buildFarTilesAsync(opts, map)` hook packs
   each build to the bootcache form AS IT EMITS; reservation reads counts from packed;
   append unpacks ONE tile at a time. Peak heap = packed array + one live tile (the
   whole-map BrickCPU object graph ≈ 250 B/brick OOM'd twice).
   ⚠️ do NOT release packed slots in place — the fire-and-forget `bootCache.put`'s
   structured clone runs AFTER append (nulling slots stored a poisoned cache entry;
   the load path now `filter(Boolean)`s to heal any such entry).

## Look fixes driven by user feedback (w6 → wagg280 → wvar galleries)

- **Slope-aware ftnrm** (NaniteResolve): the fartile normal up-blend is now weighted by
  `clamp(n.y,0,1)` — flat-forest canopy TOPS keep the full banding fix, hillside tile
  WALLS keep their directional normals instead of washing into uniformly-lit plates.
- **Dominant-child albedo in the voxel pyramid** (VoxelizeCrown.downsampleBrickGrid,
  GLOBAL — forest too): coarse-brick albedo = the max-density child's albedo, not the
  density-weighted mean. Mean-of-means killed within-crown color variance ~½ per level
  ⇒ "a far tree is almost a single color" (user report). Mode/nearest downsampling for
  categorical data (leaf-clump shades) retains the full real palette at every level —
  retained DETAIL, not injected noise; zero runtime cost; fps unchanged (61/59).
- **BAKED crown look (round 2, commit da2ed8d)** — user follow-up: still mush (less AO
  darkness + NO sun/shade side gradient far out). Root causes: **L0 brick albedo was
  the FLAT species tint** (zero variation existed for dominant-child to preserve!);
  the mesh leaf path gets per-leaf hueVar jitter + crown-depth AO (vdata.w·0.8+0.2)
  AT RUNTIME — the voxel band never had those channels; coarse mean-normals converge
  to one direction per crown; GTAO fades by 240 m. Bake-time package (free at runtime):
  `bakeCrownBrickLook` L0 post-pass = clump-lattice (~0.75 m) hue jitter × species
  hueVar with the mesh resolve's EXACT warm/cool palette + radial×vertical crown-depth
  AO; ellipsoid normal blend at coarse pyramid levels (crowns only, k=0.28·L cap 0.7)
  restores the sun-side/shade-side form. This is the voxgrad LOOK that was killed for
  its +4.9 ms runtime cost — baked, it costs nothing. Verified wbake-* gallery.
  Remaining wash: the >aggDist fartile band reads paler (splat per-cell mean + ftnrm
  up-normal bias + fog) — visible brightness step at the 280 m handoff; follow-up
  candidates: dominant-source tracking in FarTilesSplat accumulators, ftnrm strength
  re-tune now that tile sources carry baked AO.

## Verified (wvar gallery = the shipped default config)

eye: real mesh crowns, debris, terrain — 47 fps. oblique: granular per-tree voxel
crowns across the whole visible band, tonal clump structure, tiles only in the far
background — 59 fps. far: individual conifers to ~280 m, textured canopy beyond — 61 fps.
aerial (150 m): distinct crowns + cast shadows — 67 fps. high (420 m): terrain-following
canopy to the horizon, no floating/buried tiles, no seams — 107 fps.
Forest regression: fregress gallery matches the SHIP3 look (booted pre-albedo-fix;
the dominant-albedo change adds far-field variation there too — intended, global).

## Hazards / notes

- Two probes / shot runs must NEVER overlap (beauty-profile lock + GPU contention);
  stale `SingletonLock` after a crashed run must be removed (pkill first, then rm).
- A crashed/killed page leaves the node parent alive — pkill the script, not just Chrome.
- `| tail` on a background shot run buffers everything — write to a log file and poll.
- World shots: use scratchpad `shot-world.mjs` (ground-relative poses via
  `heightAtCpu`, heartbeat distinguishes busy vs dead main thread).
