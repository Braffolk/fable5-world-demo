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

## ROUND 2 (user feedback on the shipped arc, 2026-07-02 evening)

User reports: (a) mesh→voxel transition jarring — "voxels overcompensate, larger than the
crown itself" (forcevox same-tree A/B CONFIRMED: solid blob 2-3x the mesh crown's visual
mass); (b) AO+lighting emphasise the cube shape — "fluffy/blobby not built-from-cubes";
normals alone can't fix it (user-confirmed after voxbead already ON at 0.6).

**SHIPPED (round 2):**
- **GTAO distance fade 700/1800 → 90/240 (?aofade)**: the horizon march read every brick
  step as a crevice and OUTLINED the cubes. Mesh band (<60 m) keeps full AO; voxel band
  fades out by 240 m. Visually kills the crevice-emphasis AND measured ~−2.6 ms oblique.

**BUILT, MEASURED, KILLED (defaults off — booked so nobody retries blind):**
- **?voxocc occupancy-threshold slimming**: sweeps 0.12/0.2/0.35/0.5 visually NO-OP
  (histogram: interior cells saturate 1.0, single-leaf cells exactly ~0.33) and 0.12 COST
  +3.5 ms (thin masks push bricks off the free full-mask path). Default back to 0.02.
  Fatness is representational (leaf planes → solid cells), not a threshold problem.
- **?voxalpha density-driven stochastic opacity** (UE5-style dissolve, prototype):
  mechanically works (trunk visible through crown, rims dissolve) but USER-REJECTED —
  the stipple reads as noise in stills; also violates their perf philosophy (perf budget
  buys DETAIL, not screen-door tricks). Default off; DELETE in next cleanup if unused.
- **?voxgrad crown-gradient + ?voxjit2 clump albedo**: subtle wins, measured **+4.9 eye /
  +4.6 obl COMBINED** (per-voxel-pixel instance fetch + hash in resolve) — fails the
  looks-per-ms bar. Defaults 0; revisit only with a batched/cheap formulation.

**FINAL SHIP (fresh-beauty-occ002.json, cd5): eye 25.2 / oblique 26.5 / aerial 9.9** —
better than the round-1 booking (25.6/29.1/10.1) thanks to the AO fade savings.
⚠️ HARNESS: probe EXTRA is comma-separated → aofade's OWN comma arg can't ride EXTRA
(aofade=700-1800 silently no-ops → falls to default). Two probes must NEVER overlap
(the interrupted-probe rerun overlapped its zombie → 55-79 ms garbage numbers).

**OPEN (structural, next beauty arc):** the voxel crown's solid-mass look vs the mesh's
airy fronds is inherent to opaque-cell voxelization. Real candidates: finer cells near
the handoff (voxgrid ↑ costs boot+mem), UE5-style temporal dissolve BETWEEN
representations at the 60 m seam, or per-cell (not per-brick) albedo/normal payloads.

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
