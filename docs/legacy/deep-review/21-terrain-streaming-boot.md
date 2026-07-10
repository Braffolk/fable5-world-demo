> Reconciled into 20-live-loop-and-cpu.md §Reconciliation (2026-07-02); this doc kept for detail.

# terrain, streaming & boot deep review (2026-07-02)

Scope: the terrain stack (`src/nanite/TerrainClipmap.ts`, `TerrainStreamer.ts`,
`BuildHeightGrid.ts`, `DagCache.ts`, `DagWorkerClient.ts`/`DagWorker.worker.ts`, the terrain
arm of `WorldRegistry.ts`), the BOOT profile of the canonical forest scene
(`src/debug/ForestScene.ts`, `VegLibrary.ts`, `VoxelizeCrown.ts`, `FarTiles.ts`,
`GeometryRegistry.ts` build/flush, `src/main.ts`), registry growth paths + memory, and the
streaming-hitch story. Companion to `20-live-loop-and-cpu.md` (read it first — this doc does
NOT re-cover the live loop, the measurement model, or the 5.5 GB staging-array heap floor; it
covers what streams, what boots, and what neither does). All numbers computed from the session
JSONs (`fresh-voxbocc-milestone.json`, `fresh-voxbocc.json`, `fresh-noleaves-now.json`,
`fresh-base-*.json`, `fresh-cellsmoke.json`, `fresh-ft-smoke.json`, `fresh-final-rested.json`,
…) in the session scratchpad, plus code reads. No GPU probes were run for this doc.

---

## Premise audit

**PA1 — "noleaves 15.4 includes terrain" is FALSE. Terrain contributes 0.00 ms to EVERY
canonical number.** The forest scene has NO terrain clusters: "the terrain itself is not in
the registry, so not rendered" (ForestScene.ts:437-441). `Heightfield.generate` runs at boot
purely so the resolve's hf bindings are valid (ForestScene.ts:444-446); the resolve's terrain
shading is gated per-pixel on `isT` and no terrain pixels ever exist (NaniteResolve.ts:424-431).
`TerrainStreamer` is never constructed in the forest path — its only wiring is
WorldRegistry.ts:668-689 (built under `inSet('terrain')`) driven per frame by
TerrainScene.ts:307-313; ForestScene builds its own registry and never calls
`buildWorldRegistry` (ForestScene.ts:169, doc 20 PA3 concurs). So: **noleaves 15.4 oblique =
trunks + cull + resolve + post, zero terrain** — and per doc 12 it also OVERSTATES the trunk
share (noleaves keeps bark maxDist 2000 vs 140 under the default fartiles config,
ForestScene.ts:388). Every "terrain frame cost share" question dissolves; what remains is the
UNMEASURED world-scene terrain cost (probe 3).

**PA2 — the boot number is era-poisoned; compare within eras only.** "70-73 s today" is right
for the current code (milestone 73.6 s, voxbocc 72.7 s) but the same scratchpad holds 60.3-74.4 s
defaults AND 98.8-106.4 s runs — the ~100 s runs are pre-worker-splat (fresh-final-rested:
fartiles "built in 41672 ms" single-threaded vs 9915-10609 ms after wave 4). Any boot lever must
be judged against a same-commit control, not the folk "70 s".

**PA3 — boot is per-SPECIES work, not per-TREE work.** 4 000-tree boots cost the same as
200 000-tree boots in the same era (fresh-cellsmoke 61.3 s @ 4k vs fresh-ab200-off 62.9 s @ 200k;
fresh-ft-smoke 60.3 s @ 4k with fartiles at only 0.52 s). Planting, instance binding and
instance upload are ~1-2 s of the 73.6. The whale is 20 crowns × (voxelize + DAG) + 812 tiles —
deterministic recomputation of pure functions of (seed, config, code version). One level up:
**boot is a build system running as runtime**; the fix class is caching/fan-out (two in-repo
precedents already exist: DagCache for terrain DAGs, the FarTiles worker pool), not making any
single stage cleverer.

**PA4 — "0 live stutters" certifies a STATIC scene, not the streaming machinery.** In the
forest live loop nothing streams: FarTiles workers are boot-only and terminated
(FarTiles.ts:324-334), the DagWorker never exists, no buffer grows post-flush (doc 20 PA3).
The streaming-hitch question ("what keeps live at 0 stutters as budgets shrink") is therefore
about machinery that has never run under the milestone protocol — TerrainStreamer runs only in
`?scene=terrain`, and the doc-06 L3 streamed brick tiles don't exist yet. §Streaming below
audits the design + names the two missing enablers instead of claiming a measured guarantee.

**PA5 — the <15 s target needs a definition before it's chased.** `bootS` is probe wall-time
from `page.goto` to `__laas.ready` (probe-fresh-stutter.ts:74-87), which includes headless-
chromium page load + vite dev transforms (~2-4 s that a production bundle/user doesn't see or
sees differently) and `engine.settle(6)` — the first 6 real frames, i.e. the lazy TSL→WGSL→Metal
pipeline compilation (main.ts:175-180). And "interactive" ≠ "fully built": the append-after-
build architecture (appendVoxelCrown/appendFarTiles/flush are all post-`build()`,
ForestScene.ts:362-401) already supports going interactive before the far field lands. Target
proposal: **<15 s cold to first flyable frame on a production build, warm-cache <10 s, far
field allowed to complete within +10 s (user sign-off on that transient — quality law).**

---

## How it works today

### Terrain stack (the world/terrain scenes — NOT the forest)

- **Geometry**: `clipmapTiles` (TerrainClipmap.ts:55-97) — L concentric levels of same-gridN
  tiles at doubling stride, hollow rings, camera-snapped to 2× tile grid so levels nest exactly
  (no straddle ⇒ no z-fight/double-draw); `clipmapMaxTiles` = M² + (L−1)(M² − (M/2)²)
  (TerrainClipmap.ts:100-105). Defaults: gridN=128 (TerrainScene.ts:258), M=4, levels =
  ceil(log2(2·res/(M·gridN)))+1 (TerrainStreamer.ts:219) → res 4096: 5 levels, maxTiles 64.
- **Builder**: `buildHeightGrid` (BuildHeightGrid.ts:1-37) — heightmap-native regular-grid
  levels (stride 2^ℓ), tile-uniform cut, skirts; pure typed-array CPU, node-runnable, ~170 ms
  per gridN=128 tile cache-miss (DagWorkerClient.ts:97-99), run off-thread on a 2-4 worker
  `DagWorkerPool` (WorldRegistry.ts:646-653), IndexedDB-cached (`DagCache.ts` — key
  seed+gridN+suffix+version, DagCache.ts:39-41; **the store is terrain-only**, and
  DagWorker.worker.ts:2-3 says "(and later buildDag for explicit pools)" — never built).
- **Streamer**: `TerrainStreamer` — boot: `buildBootSet` bakes the spawn ring concurrently
  across the pool then `attachBootSet` post-build (TerrainStreamer.ts:267-316,
  WorldRegistry.ts:672, :964); live: `update(camXZ)` per frame → single-flight `runDiff`
  (TerrainStreamer.ts:319-337): ≤4 arrivals/batch baked concurrently (MAX_LOADS_PER_DIFF=4,
  :43), attach with no await between alloc and write (:434-445), **lazy eviction** — departed
  tiles keep rendering until full coverage returns or pool pressure reclaims the farthest
  (:355-370, :459-466, :476-496). Invariants: build-before-alloc, cap-pre-check-before-alloc,
  coarser-ring backstop (never a hole).
- **Registry side**: `reserveTilePool` freezes a fixed slot region at build
  (GeometryRegistry.ts:1817-1834, :1443-1459); `attachHeightDagTile` packs verts/indices/
  clusters/DAG/hier-links into the slot and pushes partial-upload ranges
  (GeometryRegistry.ts:1895-2017); `evictHeightDagTile` zeroes the mesh draw + parks the
  sphere — no tombstones (:2026-2043). This is the ONLY recycling allocator in the registry.

### Boot sequence of the canonical forest scene (ForestScene.ts:59-485, main.ts:123-180)

1. `buildVegLibrary` (progress 0.1→0.5): foliage-atlas GPU captures per species, bark bakes +
   `bakeBarkArray`, then 6 species × 4 variants × 3 LODs = 72 synchronous `buildTree` calls,
   PLUS understory/fern/flower/log/stump/rock/stone/branch pools (VegLibrary.ts:177-209,
   :253-322, :371-703) — of which ForestScene keeps only the 20 canopy pools
   (`p.cls <= 4 && p.leaf`, ForestScene.ts:164). Impostor bake already skipped
   (`impostors:false`, :160-162).
2. Register 40 meshes (bark+leaf per pool; synchronous `clusterize` inside `registerMesh` —
   32 109 boot clusters per the vcompact console line) + `prepareVoxelCrown` × 20 at grid 180,
   voxlod 7L (ForestScene.ts:202; pure main-thread CPU: `voxelizeCrown` + pyramid,
   VoxelizeCrown.ts:1291-1315).
3. Plant 200k (trivial), bind streams (:240-268).
4. 40 synchronous main-thread DAG builds — 20 QEM bark (`buildDag`) + 20 aggregate leaf
   (`buildAggregateDag`), "~0.8 s/crown @ 4000" per the comment (ForestScene.ts:270-285).
5. FarTiles: worker splat (8 workers) + MAIN-THREAD emit/pyramid (ForestScene.ts:346,
   FarTiles.ts:303-358) — measured 6.55 s + 4.01 s (milestone console).
6. `reg.build` (allocates + uploads the frozen-cap buffers, GeometryRegistry.ts:1360-1470) →
   `attachDag` × 40 → `appendVoxelCrown` × 20 → `appendFarTiles` (812 heads) → `flush` —
   the whole append+flush block measured **304 ms** (ForestScene.ts:362-401).
7. `Heightfield.generate` (bindings only — erosion 500-900 GPU iters + rivers + biome +
   readbacks, Heightfield.ts:108-192, WorldConst.ts:58-62) + SunSky + PostStack +
   `buildNaniteFrame` (ForestScene.ts:443-462).
8. `engine.start(); settle(6)` — first frames = pipeline compile — then ready (main.ts:175-180).

---

## Work model

### Boot decomposition (milestone run, 73.6 s cold; MEASURED vs DELTA vs EST)

| stage | cost | basis |
|---|---|---|
| browser + vite + module eval + WebGPU init | ~2-4 s | EST (probe-side; PA5) |
| veg library (GPU bakes + 72 buildTree + non-tree pools) | ~8-12 s | EST within the noleaves floor |
| registerMesh clusterize (32 109 clusters) | ~1-2 s | EST |
| **prepareVoxelCrown × 20 (grid 180, 7L)** | **~30-34 s** | DELTA: (default − fartiles) − noleaves = (73.6−10.6) − 29.5 ≈ 33.5; 4k-era gives ≈ 30.5 |
| 40 sync DAG builds (leaf aggregate dominates) | ~10-16 s | EST (comment ~0.8 s/crown ×20 + bark) |
| **FarTiles splat + emit/pyramid** | **10.6 s** (6.55 wkr + 4.01 main) | MEASURED (console) |
| reg.build + attach/append + flush + uploads | ~1-3 s (append+flush = 0.30 s measured) | MEASURED + EST |
| Heightfield.generate + sky + post + NaniteFrame | ~2-5 s | EST |
| settle(6) incl. pipeline compile | ~1-3 s | EST |

Cross-checks: noleaves floor (skips voxelize + fartiles, keeps everything else including the
20 leaf DAG builds — noLeaves only gates `toVoxel`/fartiles, ForestScene.ts:200-209, :312) =
**29.4-30.2 s** over 5 runs ✓; boot is tree-count-flat (PA3) ✓; pre-worker-splat era ≈ same
+31 s of single-threaded splat ✓. **~75 % of the 73.6 s (voxelize 33 + DAG ~13 + fartiles 10.6)
is deterministic recomputation** of (seed, config, version)-pure artifacts. The per-stage EST
bands need one instrumented run to become facts (probe 1).

Warm-boot model with a generalized artifact cache (lever B1): skip voxelize (−33), tree DAGs
(−13), fartiles (−10.6), pay IndexedDB reads (~0.5-2 s for a few hundred MB) → **~17-19 s**;
add trees-only veglib (B4, −3-6) → **~12-16 s ≈ the <15 s target**, without touching cold-path
algorithms. Cold boot with worker fan-out (B2+B3): 33/8 + 13/8 ≈ 6 s instead of 46 → cold
~30-35 s.

### Terrain frame-cost model (world/terrain scene — forest is 0 by PA1)

- CPU: `streamer.update` per frame = `clipmapTiles` (≤~80 tile records, integer math) + a Map
  diff; runDiff's bake path only fires on ring-boundary crossings. ≈ 0.02-0.1 ms + allocation
  dust. Invisible next to the 1.1 ms encode budget (doc 20).
- GPU: ≤64 resident tile meshes ride the SAME BFS cull + SW raster as vegetation
  (hier links packed per slot, GeometryRegistry.ts:1968-1993). Tile-uniform cut bounds emitted
  terrain tris to a τ-controlled screen density — order 10⁵ tris ≈ a few % of the forest's
  4.4 M visTris; expected ~0.5-2 ms gpuWall. **UNMEASURED — no terrain-scene run exists in the
  session data; probe 3 pins it before the world scene ever enters the mission ledger.**
- Streaming steady-state: cache-hit tiles attach in the batch loop (pack ≤4 tiles ≈ ~100k
  words + index rebase ≈ sub-ms each) + partial `writeBuffer` ranges next frame (100s of KB)
  — no identified >2 ms main-thread step. Cache-miss bakes are off-thread (~170 ms each,
  4 concurrent).

### Memory (nanite.mb 1353) + growth paths

`bytes()` (GeometryRegistry.ts:2096-2113) over used cursors: bricks 4 157 970 × 36 B
(BRICK_WORDS=9, VoxelBrick.ts:12) = **149.7 MB** (96 % fartiles, doc 15); instances 600 812 ×
36 B = **21.6 MB**; clusters ≈ 159 k × 32 B ≈ 5 MB; remainder ≈ **~1 176 MB = tree verts +
indices across all DAG levels** — the mesh pipe, not the voxel pipe, owns the GPU footprint.
Growth paths are all FROZEN at `build()` (caps from cursors + `addLate`,
GeometryRegistry.ts:1360-1376): post-build growth only consumes pre-reserved late space
(attachDag/appendBricks — appendBricks throws at cap, :1008-1019); the tile pool is the only
slot-recycling allocator and it is terrain-only. **For the doc-06 L3 tiered-tile design:** the
brick buffer has headroom to ~256 MB (the docs-06/15 cliff — but NOTE the adapter advertises
maxStorageBufferBindingSize = 4 GB in the boot console; the "cliff" is unverified on this
device — probe 4), and doc 06's streamed-ring math (~32 MB rings) fits trivially — what is
MISSING is machinery, not memory: (a) a brick slot pool/free-list (appendBricks is append-only),
(b) a three-free emit/pyramid so tiles can bake off-thread (VoxelizeCrown drags the three
import chain, FarTiles.ts:33-36). CPU-side the real pressure is the 5.5 GB heap floor
(live.heap = 5520.6 MB, milestone) — doc 20's release-staging-arrays lever, not re-covered here.

---

## Waste inventory

| item | cost | evidence |
|---|---|---|
| prepareVoxelCrown × 20 on the main thread, every boot | ~30-34 s/boot | delta table above; ForestScene.ts:202 |
| 40 tree DAG builds, main thread, uncached, every boot | ~10-16 s/boot | ForestScene.ts:270-285; DagCache is terrain-only (DagCache.ts:19-26) |
| FarTiles emit/pyramid on the main thread (splat already worker-fanned) | 4.0 s/boot | console; FarTiles.ts:345-356 |
| FarTiles splat recomputed every boot (deterministic) | 6.5 s/boot | doc 15 L5; console |
| veg library builds ~40 non-tree pools + atlases the forest discards | ~3-6 s/boot | VegLibrary.ts:371-703 vs ForestScene.ts:164 |
| GPU idle during the ~45 s CPU chunk (heightfield runs LAST, serial) | ~2-4 s wall | ForestScene.ts:444 after all CPU stages |
| terrain frame cost in canonical numbers | **0** (system absent) | PA1 |
| DagCache old-version rows never evicted | MBs of IndexedDB, unbounded across version bumps | DagCache.ts:23-26 (no GC) |
| boot tax on the measurement pipeline itself | ~70 s × every probe run | PA5; probes re-boot per run |

Honest summary: there is no frame-time waste in this area at the canonical config (terrain is
absent, streaming is idle); the waste is ~55 s of deterministic recomputation per boot and a
missing streaming enabler pair for the doc-06 tile design.

---

## Levers

Quality classes per the ABSOLUTE constraint. Boot levers touch no pixels ⇒ IDENTICAL, except
B5 (a user-visible transient ⇒ RISK, sign-off required).

### B1 — boot:artifact-cache — generalize DagCache to all boot artifacts (M, IDENTICAL, −55 s warm)
Mechanism: one IndexedDB store (DagCache pattern, DagCache.ts:43-171) for (i) tree DAG builds
(serialize DagBuild verts/indices/clusters — the terrain serializer is the template), (ii)
prepared voxel crowns (BrickCPU levels + blocks), (iii) fartile splat grids (doc 15 L5 —
subsumed here). Keys: seed + species/variant + leafdensity + clustertris/fill + voxgrid +
voxlod-config + code-version (bump-to-invalidate, DAG_CACHE_VERSION precedent). Expected:
73.6 → ~17-19 s warm; probe throughput +40 %/run. Risk: stale-cache bugs — version discipline;
large blobs (100s of MB) — chunk per-species/per-tile rows.

### B2 — boot:workerize-crown-voxelize (M, IDENTICAL, −27 s cold)
Extract the pure core of `voxelizeCrown`+`buildVoxelPyramid` into a three-free module
(FarTilesSplat.ts is the exact precedent for this surgery, FarTiles.ts:30-36) and fan the 20
crowns across the 8-worker pool: ~33 s → ~4-5 s. Doubles as the enabler for off-thread tile
emit (S1). Risk: determinism across worker boundaries (transfer typed arrays, keep emit order
— the FarTiles slot-ordering idiom, FarTiles.ts:309-329).

### B3 — boot:workerize-tree-dags (S-M, IDENTICAL, −10-14 s cold)
`buildDag`/`buildAggregateDag`/`clusterize` are ALREADY three-free (BuildDag.ts:35-36,
BuildAggregateDag.ts:33-40, Clusterize.ts imports nothing) — add the 'dag' request kind that
DagWorker.worker.ts:2-3 always intended, fan 40 builds over the pool. Combined with B2, cold
boot ≈ 30-35 s; combined with B1, warm rebuilds after a version bump stay tolerable.

### B4 — boot:veglib-trees-only (S, IDENTICAL in the forest scene, −3-6 s)
`opts.treesOnly` skipping understory/fern/flower/deadfall/rock/stone/branch pools + their
atlas captures when the caller filters to canopy pools anyway (ForestScene.ts:164) — the exact
`impostors:false` move (VegLibrary.ts:160-171) applied one ring wider. World scene unchanged.

### B5 — boot:progressive-far-field (M, RISK-transient / IDENTICAL steady-state, −10 s to interactive)
Go interactive after `reg.build`+per-tree flush; run FarTiles (or its B1 cache load) in the
background and `appendFarTiles`+`flush` when ready — the post-build append path is already the
architecture (ForestScene.ts:362-401, flush is incremental by design, GeometryRegistry.ts:33-35).
The far field pops in seconds after control is granted. NEEDS user sign-off (quality law: a
visible transient, even at boot). Surfaced, not assumed.

### B6 — boot:overlap-gpu-cpu (S, IDENTICAL, −2-4 s)
Kick `Heightfield.generate` (pure GPU) off at scene start concurrent with the CPU stages
instead of after them (today: ForestScene.ts:444 awaits serially, the GPU sits idle through
the ~45 s CPU chunk). Await it just before `buildNaniteFrame`. Zero interaction: heightfield
touches no registry state.

### S1 — streaming:brick-slot-pool + three-free tile emit (M, IDENTICAL, 0 frame-ms — ENABLER)
The two missing pieces for doc 06 L3 (streamed tiered ring tiles) and any future brick-budget
shrink: (a) mirror `reserveTilePool`/`attachHeightDagTile`/`evictHeightDagTile`
(GeometryRegistry.ts:1817-2043) for BRICK ranges + voxel heads — the evict idiom (clusterCount
0 + parked sphere) transfers verbatim; appendBricks today can never reuse (:1008-1019);
(b) B2's extraction so per-tile emit/pyramid (today 4.0 s/812 tiles ≈ ~5 ms per 64 m tile on
the MAIN thread — a guaranteed hitch source at exactly streaming cadence) runs on workers.
Adopt the TerrainStreamer invariants wholesale (single-flight, build-before-alloc, coalesce,
lazy-evict-backstop — TerrainStreamer.ts:17-30): they are the audited hitch-free template.
Guardrail: a per-frame main-thread streaming budget (≤2 ms) + `terrain.stream.*`-style
counters (TerrainStreamer.ts:499-508) + a live roam probe asserting 0 longtasks (probe 5).

### Cross-references
- doc 15 L5 (fartiles boot cache) ⊂ B1. doc 06 L3 memory/bake math confirmed compatible with
  S1 (§Work model). doc 20 release-staging-arrays owns the 5.5 GB heap floor. The doc-06/15
  "256 MB cliff" needs probe 4 before it constrains any design.

---

## What UE5/prior art does here

- **Derived-data cache (DDC):** UE5 never rebuilds deterministic artifacts at boot — Nanite
  cluster builds, HLOD proxies, texture compiles all land in a content-addressed local/shared
  DDC keyed by asset hash + build version. B1 is exactly a browser-side DDC; DagCache already
  proved the pattern in-repo for terrain.
- **Async loading thread + streaming pools:** UE5 streams World Partition cells / HLOD tiles
  through fixed-budget pools with time-sliced main-thread finalization (our attach batches) and
  never grows GPU allocations mid-session — the frozen-caps + slot-pool design matches; the
  brick pool (S1) is the missing analog for merged proxies.
- **Geometry clipmaps / CDLOD:** the terrain stack is textbook (camera-snapped nested rings,
  tile-uniform LOD, skirts); UE5's Landscape streams heightfield sections the same way. The
  ~170 ms off-thread tile bake + backstop-ring policy is the standard "never hole, pop coarser"
  contract.
- **Boot contract:** shipping titles hide residual streaming behind a controlled reveal
  (fade/vista) rather than blocking to full residency — B5 with an explicit reveal is the
  industry-standard answer to the quality-law transient objection.

---

## Open questions + proposed serial probes

Small `tools/` edits first (vite-ignored, safe while GPU jobs queue): (i) `[boot]`
performance.now stamps around each ForestScene/main stage (console lines already ride
`consoleLines`, probe-fresh-stutter.ts:61-64); (ii) a `?bootstamp=1` no-op guard if noise is a
concern.

1. **Instrumented boot split (turn the EST rows into facts).** After edit (i):
   `CONFIG=default LABEL=boot-stamps TREES=200000 TICKS=0 FRAMES=0 npx tsx tools/probe-fresh-stutter.ts`
   and `CONFIG=noleaves LABEL=boot-stamps-nl …`. Decision: confirms the ~33 s voxelize / ~13 s
   DAG / ~10 s veglib split; any stage >2× its EST band re-ranks B1-B6.
2. **Warm-cache A/B (after B1).** Two consecutive runs same config: `LABEL=boot-cold` then
   `LABEL=boot-warm`. Gate: warm ≤20 s (B1 alone) / ≤15 s (B1+B4); byte-identical first-frame
   shot vs cold (IDENTICAL class proof).
3. **Pin the world-scene terrain share (the number PA1 removed).**
   `SCENE=terrain LABEL=terrain-iso TICKS=300 FRAMES=48 npx tsx tools/probe-fresh-stutter.ts`
   (needs the probe's scene param plumbed; TerrainScene already drives the streamer). Read
   isolated meds + `terrain.stream.*` counters at a roam that crosses ring boundaries.
   Decision: terrain ≤2 ms ⇒ the world-scene mission ledger inherits forest levers unchanged;
   >2 ms ⇒ terrain gets its own review.
4. **Is the 256 MB brick cliff real on this device?** Boot-only sweep `EXTRA=ftcell=0.6`
   (~240 MB) then `ftcell=0.5` (~386 MB), TICKS=0 FRAMES=12: validation error vs graceful vs
   perf cliff at each. The adapter advertises 4 GB max binding (boot console) — if 386 MB just
   works, doc 06 §3.5's "static fine tiles are DEAD" needs re-deriving and S1's design space
   widens.
5. **Streaming hitch guard (once any streamer runs live).** Extend doc 20's POSE_PATH live
   probe with a terrain-scene roam segment crossing ≥2 clipmap boundaries at 10 m/s; gate on
   0 longtasks + no slot-histogram degradation vs a static hold — the "0 stutters under
   streaming" certificate PA4 says we don't yet have.
6. **Open question — production-build boot:** all bootS data is vite-dev; a `vite build`+
   preview run would separate transform/serve cost (~2-4 s?) from real work before anyone
   optimizes the wrong 4 seconds.
