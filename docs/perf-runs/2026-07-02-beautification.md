# 2026-07-02 — BEAUTIFICATION arc (forest scene)

User directive: maximize looks; perf regressions accepted (forest = stress test, re-optimize later).
The 5 reported issues and their resolutions. Shots: session scratchpad `shots/` (baseline `base-0-*`,
sweeps `s1..s3`, verify `v1-*`, discriminators `t1..t3`, near-field pivot `a/b/c-*`, `d/e-*`).

## Issue → root cause → fix

1. **Near field too close** (real geometry ends ~45 m) — REFRAMED by user mid-arc: pushing the
   mesh band to 90 m exposed the DAG simplifier's crown pathology ("MASSIVE leaves, long spiky
   pieces") and cost heavily (eye overlay 88→45 fps). PIVOT (user): mesh band back to ~60 m,
   near-mid owned by MUCH finer voxels (UE5-style). Shipped: `voxnear` default 45→(90)→**60**?
   [pending final pick], `voxgrid` 180→**256** candidate (L0 bricks 0.156 m ≈ 4.4 px at 45 m).

2. **Mid-field voxels ~15 px** — root cause: `lodWarp` (perf-era, "this knob never ships" —
   it shipped) τ doubling every 6 m past 4 m, clamped only at `voxtaucap=12` ⇒ ~12 px error
   budget ⇒ 15-24 px bricks everywhere past ~40 m. Shipped: **voxtaucap 12→4** (NaniteCull),
   **lodnear 4→20, simband 6→25** (NaniteFrame; softened curve also fixes mesh-crown mush
   in the 20-60 m band). Verified S2/S3/C shots: mid band reads granular (~5-6 px bricks).

3. **Far chunk repeating pattern, one side darker** — voxao=0 A/B proved 100% normal-driven;
   band pitch ≈ tile pitch (autocorr 226 px ≈ 64 m at 420 m alt). Splat-averaged tile brick
   mean normals carry a tile-anchored bias → N·L turns it into repeating dark bands.
   Shipped: **MESH_FLAG_FARTILE (=32)** set for label 'fartile' heads + **?ftnrm=0.65**
   (NaniteResolve): far-TILE pixels blend brick normal 65% toward up (real canopy >140 m is
   near-lambertian anyway). Per-tree crowns keep full baked normals. VERIFIED: bands GONE
   (c-vn60g256-high/high2 vs v1-defaults-high).

4. **Chunks disappearing in high aerial** — TWO causes found: (a) proven-by-math: per-instance
   min-screen-size cull (instMinPx ≈110 px) deletes whole tiles once their ~100 m sphere
   projects under it (~1.2 km slant) — fartiles ARE the far representation; never size-cull
   them. Shipped: MESH_FLAG_FARTILE exemption in kSeedRoots. (b) one unreproduced 420 m
   sighting (s1-vn90-high, voxtaucap=12 runtime): whole-tile hole with ground+chips; 0/7
   recurrences at cap4 across later runs — cut-path-coarse related, WATCH, not closed-proven.

5. **Z-fighting at chunk borders** — the tile cell grid was ceil-rounded to a brick multiple:
   88·0.75 = 66 m span per 64 m pitch ⇒ every tile's +X/+Z 2 m band re-splatted the neighbor's
   trees ⇒ coincident duplicate bricks (z-fight) + the always-same-side seam stripe. Shipped
   (FarTiles.planFarTiles): **cellSize snapped DOWN to tileSize/cellsXZ** (0.75→0.727) — grid
   spans exactly 64 m, watertight, no duplicates. VERIFIED: hard seam lines gone (v1 high).

## Remaining (this arc)
- **140 m voxel→fartile harsh jump** (user report #2): per-tree ~0.5 m bricks → fartile 3 m
  bricks in one step. Sweep E: ftcell 0.5 + aggdist 160 (finer tiles, farther seam) — pending.
- leaflodk 0.25 (softer crown simplification ≤60 m) — sweep D pending.
- Final defaults pick + cold verify + **perf snapshot** (canonical MF_COOLDOWN=5 probe) vs
  cleanup-gate 12.9/16.7/9.3 — REQUIRED booking (perf cost quantified, user-accepted).
- WATCH: one transient white glow blob (b-vn45g256-band, bloomed bright pixel, gone on
  same-build re-shoot — TAA/warm-up transient, low prio).

## Hazards / notes
- Boot cache: SRC_HASH covers FarTiles/VoxelizeCrown etc → every src edit = full cold rebuild
  (~75-95 s at voxgrid 256). Param combos each get their own cache entry (no eviction yet).
- Persistent-profile shot tool: scratchpad `shot.mjs` (`node shot.mjs <label> [query] [posesJson]`)
  — warm boots ~16 s; do NOT run two at once (profile lock).
- fps overlays in stills are settle-phase numbers — directional only, not bookable.
