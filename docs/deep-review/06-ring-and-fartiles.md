# 06 — The 60–140 m ring + FarTiles aggregation

Reader 06. Scope: `src/nanite/FarTiles.ts`, `FarTilesSplat.ts`, `FarTiles.worker.ts`, the
appendVoxelCrown path (`VoxelizeCrown.ts`, `GeometryRegistry.ts`, `WorldRegistry.ts:925`),
`ForestScene.ts` wiring (tileSize 64, TILE_CELL 0.75, aggdist 140, OCC_COVER 0.22).
Baseline for all lever estimates: post-voxbocc effective baseline **eye 18.9 / oblique 37.2 /
aerial 16.5** (fresh-voxbocc.json). Premise-audit targets: eye ≤18 / oblique ≤21 / aerial ≤16.

## 1. TL;DR

- The aggdist=60 "ring = ~10 ms" decomposes as ≈ **6.8 ms brick-size coarsening (quality-DEAD)**
  + ≈ **1.3 ms far-trunk-mesh removal** + ≈ **2.5–3 ms cross-tree dedup / cluster collapse** —
  the voxlodk=0.7 control reproduces 63% of it at nearly the SAME cluster count (4905 vs 4429).
- The prompt's "(a) per-cluster fixed cost 9188 → ~dozens" premise is FALSE: aggdist=60 oblique
  voxClusters = **4429**, not dozens (fresh-aggdist60.json) — the <60 m band + tile clusters remain.
- Cheapest real capture is NOT tiles-nearer: it is **tile-phase voxOccPyr refresh** (render per-tree
  vox, rebuild the min-pool pyramid once, then raster tile clusters under the existing per-brick
  cull) — Class IDENTICAL, kills the 94–140 m double-draw band AND all tiles behind the ring canopy
  at oblique. Expected oblique −2..−5.
- Static equal-or-finer ring tiles are **memory-infeasible** (≈56 M bricks ≈ 2 GB vs the 256 MB
  buffer cliff, ForestScene.ts:314–317); the quality-matched tiered-tile design only survives as
  RUNTIME-STREAMED 16 m tiles (~30–60 MB, ~0.4 worker-s/s at 10 m/s) — Class RISK, effort L,
  gated on probe P-A showing ≥5 ms residual ring pool after the Class-I levers.
- Floaters/antennas root cause: isolated cells clearing OCC_COVER=0.22 + 1-cell trunk columns
  unioned up the pyramid; fix = connected-component prune + trunk-top clamp at bake (Class Q, ~0 ms).

## 2. How it works today (mechanism walk)

### 2.1 The three-tier foliage stack and its seams

Per species: mesh leaf head to 45 m (`?voxnear=45`, ForestScene.ts:102), per-tree voxel sibling
head 45→140 m (`appendVoxelCrown` with `maxDist: farTilesOn ? aggDist : 2000`,
ForestScene.ts:371–379), FarTiles merged heads beyond `nearDist = max(10, aggDist − 46)` = 94 m
(ForestScene.ts:390). The cull enforces tiers per INSTANCE distance: `kSeedRoots` drops an instance
beyond `lodDist` (NaniteCull.ts:776–779) and nearer than word-8 `nearDist` (NaniteCull.ts:786–787).
Bark maxDist is clamped to aggDist too when fartiles is on (ForestScene.ts:388).

### 2.2 FarTiles boot pipeline

1. **Plan** (FarTiles.ts:91–198): plantation AABB → 64 m tile grid; species crown-pyramid level
   picked to best match TILE_CELL (ForestScene.ts:324–333: at 0.75 m it picks the crown level with
   ~0.75 m bricks); instances bucketed into EVERY tile their crown reach touches
   (FarTiles.ts:153–189 — the boundary-hole fix).
2. **Splat** (FarTilesSplat.ts:79–224, worker pool FarTiles.worker.ts): per tile, VOLUME-splat each
   member crown brick with per-cell overlap fractions (FarTilesSplat.ts:159–195) + a 1-cell trunk
   column ground→`crownMinY·s` with radial normals (FarTilesSplat.ts:196–202). Per-cell coverage
   `cellW` accumulates; the POST-pass sets occupancy bits only where `cellW ≥ OCC_COVER = 0.22`
   (FarTilesSplat.ts:22, 205–219) — the gap-preserving mask that makes far tiles read as trees.
3. **Emit + pyramid** (FarTiles.ts:201–283, main thread): bricks with `w ≤ 0.15` or zero occupancy
   pruned (FarTiles.ts:220); dense grid → `buildVoxelPyramid` (VoxelizeCrown.ts:949) — the SAME
   multi-level DAG machinery as per-tree crowns (K_FLOOR=3 cap, VoxelizeCrown.ts:1006).
4. **Append** (FarTiles.ts:361–381): one `appendVoxelCrown` per tile with `maxDist: 100000`,
   `nearDist: aggDist−46`, ONE identity instance at the tile center (`swayPad 0` — tiles never sway;
   per-tree vox bricks don't either: `instTransformPoint` is the contract transform, "no wind",
   NaniteCommon.ts:137–148).

Scale at 200k: **812 tiles / ~4.06 M bricks / ~140 MB** (CONTINUATION §3 wave 3; 36 B/brick,
BRICK_WORDS=9, GeometryRegistry.ts:91). Worker splat = boot 36–42 s → ~6 s (attribution doc §4.6).
Doc-rot note: the FarTiles.ts:13–16 header still says "default 0.5 m cells"; the ACTUAL default is
0.75 m (ForestScene.ts:317) because 0.5 m extrapolated to ~10.7 M bricks = 386 MB, over the 256 MB
max-buffer cliff (ForestScene.ts:314–316).

### 2.3 What the ring renders per frame

The per-tree band 45–140 m holds the oblique whale: **voxClusters 9188 at oblique vs 659 aerial**
(fresh-bead-v2-base.json). Each vox cluster = one 128-lane workgroup: per-cluster item fetch +
block-sphere occl test on lane 0 (NaniteVoxelRaster.ts:588–673), Phase A = one lane per brick
projecting 8 AABB corners → bbox record (NaniteVoxelRaster.ts:675–920), `?voxbocc` per-brick
front-slab test vs voxOccPyr (NaniteVoxelRaster.ts:838–884, DEFAULT ON), Phase B = cooperative
per-pixel atomicMax elections over each bbox. **voxOccPyr is built from visPayloadV BEFORE
dispatchVoxel and contains ONLY mesh content** (NaniteVoxelRaster.ts:389, 1454–1461) — no vox
election ever occludes another vox brick in the default single-dispatch path
(NaniteVoxelRaster.ts:1487–1491).

Granularity math (projK = cot(27.5°)·736.5 ≈ 1415 at 2268×1473): per-tree bricks hold ~5–6 px
through the band (entry L0 ~0.25 m at 60 m ≈ 5.9 px, ForestScene.ts:96–97 comment; L1 ~0.5 m at
140 m ≈ 5.1 px, FarTiles.ts:14–16). Tile cells: 0.75 m = 7.6 px at 140 m (accepted look), 11.3 px
at 94 m, **17.7 px at 60 m and ~76 px at the aggdist=60 nearDist of 14 m — the "massive voxels"**.
`?voxtaucap=12` (NaniteCull.ts:308–318, 852–859) is what keeps far crowns multi-brick at all.

## 3. Waste map

### 3.1 Attribution of the aggdist=60 "~10 ms" (the slice's core question)

Same-era rows (pre-voxbocc baseline bead-v2-base; med gpuWall; oblique voxClusters):

| run | eye | oblique | voxClu-obl | visTris-obl | source |
|---|---|---|---|---|---|
| default | 36.0 | 43.1 | 9188 | 1.270 M | fresh-bead-v2-base.json |
| voxlodk=0.85 | 34.0 | 41.1 | 6955 | 1.072 M | fresh-voxlodk085.json |
| voxlodk=0.7 | 32.7 | 36.3 | 4905 | 0.897 M | fresh-voxlodk07.json |
| aggdist=60 | 26.1 | 32.3 | 4429 | 0.730 M | fresh-aggdist60.json |
| voxbocc=1 | 18.8 | 37.2 | 9185 | 1.271 M | fresh-voxbocc.json |

Decomposition at oblique (−10.8 total):

- **(b) brick-size / election count ≈ −6.8 ms (63%).** voxlodk=0.7 is the brick-size-only control:
  per-tree crowns intact, bricks ~1.43× coarser ⇒ crown depth-in-bricks ÷1.43 ⇒ Σ projected brick
  area (the election count driver) ÷~1.43. This entire term is **quality-REJECTED** (red list,
  premise audit §4): capturing it while equal-or-finer is *by definition impossible via coarsening*.
- **(a) per-cluster fixed cost ≈ small, NOT the "9188→dozens" story.** aggdist=60 lands at 4429
  clusters — voxlodk=0.7 reaches 4905 for 4.0 ms LESS win. At near-equal cluster counts the two runs
  differ by 4.0 ms ⇒ cluster-count fixed cost cannot dominate. Contradicts the workflow prompt's
  premise (a); cite fresh-aggdist60.json. Ms-per-cluster is also non-constant across the sweep
  (0.9 µs at k085, 1.6 µs at k07, 2.3 µs at aggdist60) — clusters proxy bricks+elections, they are
  not themselves the cost.
- **(c) cross-tree dedup + trunk removal + coarser-than-k0.7 cells ≈ −4.0 ms.** Of this,
  far-trunk-mesh removal ≈ −1.3 ms (aggdist=60 clamps bark maxDist to 60, ForestScene.ts:388;
  visTris 0.897→0.730 M vs the k07 control; base oblique ≈ 10.3 ms / 1.27 M tris ≈ 8 ms/Mtri).
  Leaving **≈ 2.5–3 ms genuine cross-tree occupancy dedup** (merged tiles collapse inter-crown
  depth overdraw — trees at 4 m spacing with ~11 m crowns interpenetrate heavily) *at 0.75 m cells*.
- **Eye is a different story:** voxbocc (−17.2) SUBSUMES most of the eye ring pool (−9.9) — both
  remove behind-near-canopy elections. At eye the ring is occluded by the <45 m MESH canopy, which
  IS in voxOccPyr; at oblique the occluders are the 45–140 m VOX crowns, which are NOT. This is why
  voxbocc won 17.2 at eye and only 5.9 at oblique. The aggdist=60 row is PRE-voxbocc: the
  post-voxbocc residual ring pool at oblique is UNMEASURED (probe P-A).

### 3.2 The 94–140 m double-draw band (Class-I waste)

Ranges deliberately overlap by the tile half-diagonal (FarTiles.ts:24–28): tile nearDist =
aggDist − 46 (ForestScene.ts:390; 64 m tile half-diagonal = 45.25 m). In [94, 140] m the SAME trees
render as per-tree vox heads AND inside tile heads. Tile bricks there are 3 m L0 boxes (30–45 px)
carved by voxcell — real Phase A/B work whose elections then LOSE to the per-tree bricks in front
(same content, nearer keys) — atomicMax attempts and bbox walks are paid regardless. Scales with the
annulus screen share at oblique. Est 1–3 ms oblique (bounded by probe P-B). Correctness footnote:
reach-bucketed boundary crowns can have fringe cells in a NEIGHBOR tile that stays culled while the
tree just crossed aggDist (tile-center distance vs tree distance mismatch, FarTiles.ts:166–189 +
NaniteCull.ts:786–787) — a few-cell fringe hole at 140 m, cosmetic, but worth one shotdiff pose.

### 3.3 Tiles behind the ring canopy at oblique (Class-I waste)

Beyond 140 m, most tile footprints at oblique sit BEHIND the 45–140 m crown field. Because
voxOccPyr is mesh-only (§2.3), neither the per-block test nor per-brick voxbocc can cull them.
Aerial doesn't suffer (top-down, little vox-on-vox stacking: aggdist=60 aerial 16.6 ≈ default 16.8).
This + §3.2 is the tile share of oblique cost. Est 2–5 ms oblique.

### 3.4 Ring crown-behind-crown overdraw (Class-I-capturable share unknown)

Within 45–140 m, per-tree crowns occlude each other; elections scale with silhouette ×
depth-in-bricks. The voxwaves block-level null (41.1/48.8/28.1, table) does NOT refute
brick-granularity capture (premise audit trap #2), and voxwaves was never run WITH voxbocc
(voxbocc landed after). F2B's chain cost (+6 oblique for K=16) is the barrier; a K=2 split may not
pay. Est 0–4 ms; probe P-E decides.

### 3.5 Memory ceiling blocks all "finer tiles" designs (structural)

Brick count scales ~cell^−2.39 (measured pair: 0.75 m→4.06 M, 0.5 m→10.7 M ⇒ exponent
ln2.64/ln1.5). Equal-perceptual ring cells (0.25 m, matching per-tree 0.25 m entry bricks at 60 m)
⇒ ~56 M bricks ≈ 2.0 GB static — 8× past the 256 MB cliff, and splitting the brick buffer costs the
11th storage buffer (Metal cliff, NaniteRaster.ts:339 precedent). Static fine tiles are DEAD; only
streaming (§4.3) or a brick-record diet (tile L0 bricks are grid-aligned — center derivable from
grid index — but coarse levels are tightened per-brick, VoxelizeCrown.ts:770–814, so a second codec
is shader surgery; park unless P-C shows fetch/setup dominance).

### 3.6 Floaters + trunk antennas (quality debt, not ms)

- **Floating edge bricks** (user-rejected look at aggdist=60, still present in principle ≥140 m):
  an isolated cell clears OCC_COVER=0.22 where two crown fringes overlap, its brick survives the
  `w ≤ 0.15` prune (FarTiles.ts:220) with 1–2 occupied cells, disconnected from the crown body.
- **Trunk antennas** (CONTINUATION §4.5): the 1-cell trunk column splats at full coverage 1
  (FarTilesSplat.ts:200–201), always survives OCC_COVER, and unions up the pyramid into thin tall
  bricks poking above the far canopy where neighboring crown cells got threshold-pruned.

## 4. Levers (ranked)

### L1 — Tile-phase voxOccPyr refresh (two-phase vox dispatch)  ⭐

- **Mechanism:** split `dispatchVoxel` into phase 1 = per-tree vox clusters (whole-queue dispatch,
  guard skips tile items), ONE voxOccPyr rebuild (`dispatchBatch(renderer, voxPyrKernels)`,
  NaniteVoxelRaster.ts:1461 — the chain already exists), phase 2 = tile clusters. Tile bricks in
  the 94–140 band (behind the SAME trees rendered per-tree) and tiles behind the ring canopy now
  fail the EXISTING per-brick front-slab test (NaniteVoxelRaster.ts:838–884). No new storage buffer:
  tag tile items with instId bit 31 at kVoxFanout (NaniteCull.ts:512–529 writes `(instId, ci)`;
  gate on a MESH_FLAG_FARTILE set by `registerVoxelHead` — flags precedent GeometryRegistry.ts:1118).
  This is the same structure as UE5's two-pass occlusion (CULLING_PASS_OCCLUSION_MAIN/POST,
  docs/perf-runs/Nanite-UE5-shaders/NaniteCulling.ush:10–11) applied at the tier seam, and it
  implements the F2B postmortem's "pre-seed never realized" fix WITHOUT the K=16 bucket chain
  (+6 ms) — one rebuild, two dispatches.
- **Files:** NaniteVoxelRaster.ts (dispatchVoxel + scatter guard), NaniteCull.ts (fanout tag),
  GeometryRegistry.ts (flag), FarTiles.ts (opts.label→flag).
- **Expected ms:** eye −0.5 (tiles at eye are mostly behind MESH canopy = already voxbocc-culled);
  oblique **−2..−5**; aerial ~0 (little stacking); live: helps exactly the 25.0 ms slot frames.
- **Quality: IDENTICAL** (conservative cull; tie-order-only differences per the packing,
  NaniteRaster.ts:952–953 idiom). Gate: shotdiff maxDiff=0 at 3 canon + 2 stress poses; engagement
  counter = per-brick-cull reject delta for tile-flagged items (>0 required — trap #2).
- **Cost to beat:** one extra pyramid chain (~0.5–1 ms — voxwaves paid ~1 chain/wave and still
  showed −4.7 aerial at 3 extra chains) + a second dispatch walking guarded-out items.
- **Effort:** M. **UE5 analog:** two-pass occlusion + HLOD proxies tested against fresh HZB.

### L2 — Post-voxbocc ring/tile pool re-pin (measurement lever, do FIRST)

- **Mechanism:** the entire ring theater sizes itself off PRE-voxbocc data. Two diagnostics:
  (i) `?aggdist=60` with voxbocc default-on (P-A) → post-voxbocc ring residual;
  (ii) a 5-line `?ftskip=1` flag (build tiles, skip `appendFarTiles`, keep per-tree maxDist=140;
  far field empty — diagnostic only) → the tile-side total at each pose (P-B) = L1's ceiling.
- **Expected ms:** 0 directly; prevents building L3 (effort L) against a pool that may have shrunk
  to <3 ms. **Quality:** diagnostics never ship. **Effort:** S.

### L3 — Streamed tiered ring tiles (the quality-matched capture design, full spec)

- **Design:** 16 m tiles for the ring, cell size matched to the per-tree BRICK size at each band's
  near edge (the perceptual bar; the per-tree ladder holds bricks ~5–6 px): band [60, 90] m →
  0.25 m cells (5.9 px at 60 m ≤ per-tree's 5.9 px), band [90, 140] m → 0.375 m cells. 16 m tiles
  shrink the double-draw overlap from 46 m to half-diagonal 11.3 m (+ crown reach). Splat/emit
  reuse `splatTiles` verbatim (it is already pure + worker-hosted); `buildVoxelPyramid` needs a
  three-free extraction (its imports — partitionClusters, validateDagHierarchy — are already pure).
  **Per-ring OCC_COVER:** at cell ≈ source-brick size, interior cells saturate to cover 1.0 and
  fringe cells thin out — re-tune the threshold per band (start ~0.35 for 0.25 m cells) + the L4
  component filter, or floaters return worse. **Per-tree crown identity** is preserved by
  construction at these cell sizes (the splat is a resample of the same crown bricks; adjacent-crown
  merging only occurs where crowns genuinely interpenetrate).
- **Memory:** ring annulus [60,140] = π(140²−60²) ≈ 50.3 k m² = 1.57% of the 1789² m² plantation ⇒
  ~880 k bricks ≈ 32 MB at 0.25 m (fits trivially IF streamed; static full-world coverage is the
  2 GB dead end of §3.5). Needs a reserved brick ring-buffer + free-list over `appendBricks`
  (GeometryRegistry.ts:1008 — caps frozen at build ⇒ reserve up front) + runtime
  `registerVoxelHead`/`bindInstances`.
- **Boot/runtime cost:** per-16 m-tile splat ≈ 40 ms at 0.25 m (measured 44–52 ms per 64 m tile at
  0.75 m; members ÷16, source-brick count ×~14). Ring prefetch ≈ 196 tiles ≈ 8 worker-s ≈ ~1 s on
  the 8-worker pool. Moving at 10 m/s the frontier admits ~11 tiles/s ≈ 0.43 worker-s/s — feasible;
  teleports (probe poses!) leave the ring empty until splat completes ⇒ needs the current 0.75 m
  far-tile layer as fallback (coarse→fine pop = visible).
- **Expected ms:** captures ONLY the dedup + double-draw share at equal resolution — the −6.8
  brick-size term is forfeited by the quality bar, and L1 already takes the double-draw/behind-canopy
  slice. Expected oblique **−2..−5 beyond L1** (cross-tree depth-dedup factor ~1.5–2.5× on ring
  elections), eye ~0. 
- **Quality: RISK** (a resampled representation is never bit-equal; bead-v2 shading also changes —
  the vox "round normal" field radiates from instance origin A.xyz, and merged heads ride identity
  instances at tile centers ⇒ per-brick shading normal shifts). Gate: shotdiff + explicit user
  sign-off; the implementing agent may NOT self-certify (aggdist=60 precedent).
- **Effort: L.** Build ONLY if P-A/P-B show ≥5 ms residual after L1 lands.
- **UE5 analog:** World Partition HLOD — tiered layers (HLOD0 merges actors, HLOD1 merges HLOD0
  cells farther out), merge fidelity chosen so proxy screen-space error stays sub-perceptual at the
  layer's engagement distance, cells STREAMED not resident. UE5 never engages a merged proxy at a
  distance where its texel/geometry error exceeds the source — exactly the rule the fixed 0.75 m
  cell violates when pulled to 60 m. Nanite itself never cross-instance merges; its per-instance
  floor is a ~128-tri root cluster, cheap enough that UE's "ring" equivalent costs almost nothing —
  our 128-brick vox cluster floor is why the ring is expensive here and why merging is even on the
  table.

### L4 — Floaters + antennas bake fixes

- **Mechanism:** (i) connected-component filter in the emit step (FarTiles.ts:216–243): drop
  occupied bricks whose 26-connected component is < 3 bricks AND mean w < ~0.5 (kills isolated
  fringe bricks, keeps legitimate small saplings); (ii) trunk-column clamp: splat trunk cells at
  cover 0.6 above 60% of column height (they then need crown corroboration to clear OCC_COVER —
  antennas die, trunks under canopy survive); alternative: clip column top to the lowest occupied
  crown cell in the column's 3×3 XZ neighborhood.
- **Expected ms:** ~0 (slightly fewer far bricks). **Quality: IMPROVING** — pixels change; crops +
  user sign-off (Class Q gate). **Effort:** S (pure CPU bake, worker-hosted, deterministic).
  Do it regardless of L3 — it upgrades the SHIPPED ≥140 m look.

### L5 — Brick-granularity vox-behind-vox inside the ring (conditional)

- **Mechanism:** L1's phase split extended to 3 phases (per-tree near-half / far-half / tiles) needs
  per-tree depth ordering = the F2B machinery; K=2 waves + voxbocc was never measured together
  (waves nulled at BLOCK granularity pre-voxbocc — trap #2 says that null doesn't transfer).
- **Expected ms:** oblique −1..−4 if ring self-occlusion survives L1; F2B chain overhead (+5/+6 at
  K=16) is the risk — K=2 must come in under ~1.5 ms of chain cost.
- **Quality: IDENTICAL** (conservative cull). **Effort:** M (mostly exists behind flags).
  Decision by probe P-E; drop permanently if net ≥ −1 ms.

## 5. Refuted / rejected for this stage (do not retry)

- **aggdist=60 / any fixed-cell tiles pulled nearer** — quality-rejected verbatim ("dogshot,
  massive voxels"); 0.75 m cells are 17.7 px at 60 m vs the accepted 5–6 px per-tree granularity.
  Diagnostic value only (ring pool sizing).
- **voxlodk 0.7/0.85** — user-rejected; ONLY the control-curve lesson survives (§3.1).
- **"Ring win = cluster-count fixed cost"** — refuted by the 4905-vs-4429 cluster near-tie with a
  4.0 ms gap (§3.1); contradicts the workflow prompt's candidate (a).
- **Block-level vox-behind-vox (voxwaves) as-is** — measured ≈0 at eye/oblique (table row
  voxf2b=1,voxwaves=4); does NOT refute brick-granularity or the L1 tier-seam variant (trap #2).
- **Static fine-cell tile layers (any tier)** — memory-dead on the 256 MB cliff (§3.5); do not
  re-derive; only streamed variants are admissible.
- **"Ring costs ~10 ms at eye" as a live pool** — superseded: voxbocc captured the eye share
  (36.1→18.9); the eye and aggdist60 pools overlap ~fully at eye (§3.1). 5c's table row stands but
  its eye implication is stale.

## 6. Open questions + serial GPU probes wanted

- **P-A (first): `CONFIG=default EXTRA=aggdist=60 TICKS=0`** (voxbocc now default). Post-voxbocc
  ring residual at oblique. Decision: Δoblique vs fresh default ≥4 ms ⇒ ring theater stays open
  (L1 then L3/L5 sized by it); <2 ms ⇒ close L3+L5, keep only L1 (double-draw is orthogonal) + L4.
- **P-B: `?ftskip=1` diagnostic flag** (5-line ForestScene change: build tiles, skip appendFarTiles,
  keep per-tree maxDist=aggDist; far field empty). TICKS=0. Tile-side cost per pose = L1's ceiling.
  Decision: oblique tile share ≥3 ms ⇒ L1 proceeds; <1.5 ms ⇒ L1 demoted to S-effort cleanup.
- **P-C: `EXTRA=voxrdbg=2 TICKS=0`** (stop before Phase-B election, NaniteVoxelRaster.ts:243).
  Splits per-cluster+Phase-A setup from election cost at oblique. Decision: setup >3 ms ⇒ cluster/
  brick-record diets (§3.5 codec) gain standing; else elections own the pool and cull levers rule.
- **P-D (after L1 lands): default vs `?voxtilewave=0` interleaved same-session A/B** + shotdiff at
  3 canon + 2 stress poses + tile-item reject counter >0. Decision: net oblique ≤ −2 ms and
  maxDiff=0 ⇒ ship default-on; else revert, record engagement numbers.
- **P-E: `EXTRA=voxf2b=1,voxwaves=2 TICKS=0`** (first-ever waves×voxbocc combination). Decision:
  net oblique ≤ −2 ms ⇒ build the K=2 band split (L5); ≥ −1 ms ⇒ close vox-behind-vox-via-F2B
  permanently (with the reject-counter evidence attached).

Open questions: (1) live pose-mix (audit P2) — if live worst-frames are eye-in-canopy, L1's oblique
focus over-weights; (2) does the bimodality interact with the voxOccPyr chain (an extra rebuild in
L1 could shift mode structure — record per-frame arrays in P-D); (3) the ε-band fringe hole (§3.2)
needs one boundary-crossing shotdiff pose before L1 ships (it changes which band double-draws).
