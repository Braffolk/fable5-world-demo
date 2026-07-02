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

## Final picks (SHIPPED, commit 0751267)
- voxnear **60**, voxgrid **256** (forest-scoped), voxtaucap **4**, lodnear **20**,
  simband **25**, ftcell **0.6**, ftnrm **0.65** + MESH_FLAG_FARTILE fixes + exact tiling.
- 140 m jump: **F (ftcell 0.6) beat G (aggdist 200)** on looks — whole far field uniformly
  finer (2.4 m bricks) vs merely delaying 3 m slabs. ftcell **0.5 REJECTED**: ~3.4× splat,
  cold boot >8 min (run E timed out) — not shippable. aggdist stays 140.
- leaflodk 0.25 (run D): visually indistinguishable from 0.4 at the band pose → keep 0.4
  (user's perf-first tiebreaker).
- SHIP verify gallery: scratchpad shots/SHIP-* (8 poses) — no banding, no seams, no holes,
  no harsh band transition. Boot (cold, grid256+ftcell0.6): ~103 s; warm (cache) ~16 s.
- BOOTCACHE FIX (in commit): cache key now stores RESOLVED ftCell — a raw-null knob let a
  code-default change silently HIT the stale entry (first SHIP run rendered 0.75 fartiles).
  ⚠️ ForestScene.ts is NOT in SRC_HASH: default changes there must be key-visible (resolved
  values in params), or they will not invalidate the cache.

## Perf booking (canonical MF_COOLDOWN=5 iso, full clock)
| config | eye | oblique | aerial |
|---|---|---|---|
| pre-beauty (cleanup gate) | 12.9 | 16.7 | 9.3 |
| **SHIP** (fresh-beauty-ship.json) | **25.6** | **29.1** | **10.1** |
| SHIP + OLD lodWarp curve (attr) | 18.4 | 22.7 | 9.8 |

Attribution: the softened lodWarp holds **eye −7.2 / obl −6.4 ms** of the regression (mesh
band tris at fine τ near); the voxel-fidelity trio (voxgrid 256 + voxtaucap 4 + ftcell 0.6)
holds the remaining ~+5.5/+6.0. Aerial is essentially free (+0.8). User accepted the cost
("stress test, we'll look at perf again after"); the future perf arc starts from this table.
⚠️ probe EXTRA is COMMA-separated ('a=1,b=2') — '&' gets URL-encoded and silently corrupts
the config (first attr run invalid, caught via the printed URL).

## WATCH (open, low prio)
- One transient white glow blob (b-vn45g256-band; bloomed bright pixel, gone on same-build
  re-shoot — TAA/warm-up transient). If seen live: suspect a single-frame NaN in lighting.
- One unreproduced 420 m whole-tile hole at voxtaucap=12 (s1-vn90-high); 0 recurrences in
  ~10 high-pose samples at cap4 + instMinPx exemption. Coarse-cut related if it returns.

## Hazards / notes
- Boot cache: SRC_HASH covers FarTiles/VoxelizeCrown etc → every src edit = full cold rebuild
  (~75-95 s at voxgrid 256). Param combos each get their own cache entry (no eviction yet).
- Persistent-profile shot tool: scratchpad `shot.mjs` (`node shot.mjs <label> [query] [posesJson]`)
  — warm boots ~16 s; do NOT run two at once (profile lock).
- fps overlays in stills are settle-phase numbers — directional only, not bookable.
