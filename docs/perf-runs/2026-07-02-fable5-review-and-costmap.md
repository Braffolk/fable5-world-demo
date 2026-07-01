# Fable 5 fresh-eyes review + cost map — nanite & voxel renderer (2026-07-02)

Branch `nanite-raster` (HEAD `6a93dfb` + this session's changes). Canonical config: `scene=forest`,
200k trees, dpr=1.5 (2268×1473). Method: 6-agent read-only code review (frame orchestration / SW
raster / cull+HZB / DAG build / voxel path / boot+resolve) + fresh serial-GPU measurements
(`tools/probe-fresh-stutter.ts` — live-rAF moving capture + isolated `measureFrames` at the three
canonical poses, 45 s cooldown before isolated phases, per-pose screenshots).

## 1. The measured frame (before this session's changes)

Isolated gpuWall (median, ms) at 200k/dpr1.5 — two independent boots agreed within thermal drift:

| pose    | default (foliage on) | `?noleaves` (trunks+cull+resolve) | foliage share |
|---------|----------------------|-----------------------------------|---------------|
| eye     | 40.5 / 44.4          | 14.2                              | ~26-30 ms     |
| oblique | 58.7 / 59.2          | 12.5                              | ~46 ms        |
| aerial  | 37.1 / 38.2          | 10.2                              | ~27 ms        |

Live moving loop (rAF, eye-level glide): default ~41 ms steady GPU-bound; `?noleaves` **8.3 ms**
(p95 9.0). CPU per frame ≤ 4 ms, zero longtasks, flat heap → **no CPU-side stutter source exists
in the forest path**; per-frame CPU orchestration is O(1) in tree count.

Counters (eye): visTris 12.44M (hw only 0.96M), of which **leaf-mesh band (<35 m) = ~10.3M** and
trunks = 2.09M. Oblique: voxel band dominant (30k voxel clusters). Same visTris at 4k trees as at
200k — the emission load is near/mid-field, not far-field.

### The user-reported "avg 8 ms, p95 120 ms constant stutters"
Not reproduced headless (steady 41 ms, no spikes, no longtasks, no GC). Best-supported theory: the
frame is ~40 ms GPU-bound; under a 120 Hz vsync'd interactive session Chrome's deep GPU pipelining
reports fast rAF deltas until queue backpressure saturates → periodic ~120 ms stalls (the classic
pipelined-producer sawtooth). I.e. the stutter *is* the throughput problem wearing a trench coat.
Falsifiable: land the throughput fixes, re-check interactively; if hitches persist, next suspects
(from review) are `meter()`'s per-readback buffer create/submit/destroy churn and the every-frame
`resolveTimestampsAsync` (both measurement plumbing, both gateable).

## 2. Root causes found (review × measurement)

1. **Voxel LOD ladder never coarsens where you can see it.** `VOXLOD_CFG.errorK=3`
   (`VoxelizeCrown.ts`) makes L0 (finest bricks) own the band out to ~380 m at τ=3 — the entire
   visible forest. The 6a93dfb crown fix bought "never a single square" twice (errorK=3 AND
   K_FLOOR=3); K_FLOOR alone already guarantees the far shell. Measured: `?voxlodk=1` →
   **oblique 58.7→38.5 ms (−34%), aerial 37.1→17.7 ms (−52%)**, eye unchanged.
2. **Leaf-mesh band (<35 m) renders LOD0 everywhere.** The aggregate ladder's L1 cut lands at
   ~57 m — beyond the 35 m mesh→voxel handoff — so no aggregate level is ever selected in-band:
   ~10.3M tris at eye. Fix: `?leaflodk=` (new) scales the aggregate error so coarsening engages
   in-band (K<1 pulls it nearer; 0.25 ≈ L1 at 14 m).
3. **Per-BLOCK flat voxel shading** (one normal+tint for up to 128 bricks) fused adjacent ~10 px
   bricks into giant single-color plates and near-black crowns — most of the "Minecraft" look was
   *shading*, not geometry. Fix: `?voxbn` (new, default ON) packs the winning brick index into
   visBV bits 21-27; resolve shades with that brick's baked normal + albedo.
4. **The user's "voxels start too close / too large" bug** = the 35 m hard switch (no blend) +
   ~11 px solid L0 squares + (3). `voxdither=0`'s own correctness note requires sub-pixel bricks;
   the defaults violated it. Addressed via (3) + the freed budget for a farther handoff
   (`?voxnear`), pending visual pass.
5. **Boot bakes octahedral impostors that the forest never samples** (6 species × 64 views × 3
   passes + readbacks + dilate floods — count-independent). Fixed: `buildVegLibrary` gains
   `impostors:false`, ForestScene passes it. (`?noimpostors=1` already existed but nothing set it.)
6. **`?naniteleaf=1` is a NO-OP in scene=forest** (parsed only by TerrainScene). Prior canonical
   URLs carried it; it toggled nothing.

## 3. Review findings worth acting on later (ranked backlog)

- **Two-pass occlusion** (NaniteCull): we test against LAST frame's HZB with no re-test pass →
  disocclusion holes/flicker on motion (UE5 does main+post pass). Scaffolding (reject slots,
  `kRasterArgs2`) already exists but is vestigial. Correctness fix more than perf; measure
  `?occl=0` vs `1` first to size occlusion's value at all.
- **Election memory traffic** (NaniteRaster F1): 3 memory ops across 2 buffers per winning
  fragment (relaxed load + atomicMax + atomicStore) vs UE5's single 64-bit atomic — the never-
  isolated sub-bucket of the old "per-pixel loop 90%" claim. Measure with a single-buffer variant.
- **Two-word election desync race** (F3): `visPayloadV` winner and `visBV` id can desync under
  racing winners → wrong-id speckle (visible in screenshots on trunk close-ups). Bounded, real.
- **No small-tri rect fast path** (F2): every triangle pays a 3-divide-per-row scanline solve;
  UE5 splits rect≤4px/scanline. Cheap A/B; may be Tint-hoisted already.
- **No cross-triangle vertex dedup** (F4): 384 transforms per 128-tri cluster vs UE5's ~66 into
  LDS; the existing `vcache` was written off at the wrong operating point (720p/40k) — re-measure.
- **DAG structural floor** (BuildDag): disjoint interpenetrating bark tubes can never QEM-merge
  → whole-tree 8-32-tri roots unreachable by tuning. The registry already supports sibling
  far-representation heads with distance handoff (exactly how leaf→voxel works) — a bark far-rep
  (voxel column / cluster billboard) is the structural answer if trunks ever dominate again.
- **`meter()` readback churn**: fresh GPUBuffer create + submit + mapAsync + destroy ~5× every
  15 frames; every-frame `resolveTimestampsAsync` for the HUD. Gate behind HUD/measure.
- **Memory floor**: cull frontier/queue backing arrays ~200+ MB CPU-side retained forever
  (`frontierCap = QRASTER_CAP` ×2), registry keeps full CPU staging copies of all mega-buffers;
  multi-GB boot garbage (whole understory/rock/fern library built then discarded by the forest
  filter; crown geometry replicated ~5-6×). GC-pressure suspect for interactive hitches.
- **Aggregate grow-clamp**: growMax=2.5 under-restores area when a level drops >84% — balding
  risk at deep coarse levels (relevant once leaflodk engages them; watch the visual gate).

## 4. What shipped in this session

- `?leaflodk=` knob (BuildAggregateDag `AGG_LOD_CFG` + ForestScene wiring) — leaf in-band LOD.
- `?voxbn` per-brick election payload (NaniteVoxelRaster bits 21-27) + per-brick normal/albedo
  shading in NaniteResolve. Default ON; `=0` restores legacy bytes.
- `impostors:false` boot skip for ForestScene.
- `tools/probe-fresh-stutter.ts` — the reusable live+isolated probe (TREES/DPR/EXTRA/LABEL/
  COOLDOWN_S env knobs, per-pose screenshots).
- Defaults changed after visual+perf gates: see §5.

## 5. Final A/B numbers + shipped defaults

**Thermal caveat:** the machine drifted ~10-15% slower over the session (chassis heat, per the
user's warning); only same-session ordered ratios are honest. The confirmation sequence ran
coolest-first, so candidate wins below are LOWER bounds.

Confirmation sequence (200k, dpr1.5, pose-only isolated gpuWall median ms, all with voxbn on):

| config                                    | eye  | oblique | aerial |
|-------------------------------------------|------|---------|--------|
| default (errorK=3 bake, voxnear=35)       | 48.4 | 63.5    | 42.3   |
| + leaflodk=0.25 + voxlodk=1               | 45.9 | 52.3    | 22.2   |
| + voxnear=60                              | 36.9 | 40.0    | 18.9   |

Net vs same-session default: **eye −24%, oblique −37%, aerial −55%.** Cooler-session pairs
measured the individual levers larger (leaflodk eye −17% alone; voxlodk oblique −34% / aerial
−52% alone), so expect better absolutes on a cool machine. Eye visTris 12.44M → ~4.2M.

**Shipped defaults (this commit):**
- `leaflodk` default **0.25** in ForestScene (eye screenshot indistinguishable from LOD0).
- `VOXLOD_CFG.errorK` **3 → 2** (compromise; `?voxlodk=1` = full perf, `=3` = legacy fine ladder
  — the errorK=1 mid-field is visibly coarser at canopy poses, user should eyeball).
- ForestScene `voxnear` default **35 → 60 m** (fixes the too-close/too-large entry band: ~6 px
  entry bricks vs ~11 px; net perf positive because the deeper mesh ring is LODed now).
- `?voxbn` per-brick shading **ON** (strict visual improvement, ~0 measured cost).
- ForestScene boot skips the impostor bake (−~7 s; remaining boot = tree gen + 48 sync DAG builds).

**Status vs the 16.6 ms goal:** not yet reached — on the hot machine the best config reads
eye ~37 / oblique ~40; cool-machine estimate ~28-33. Roughly half the original gap closed with
zero eye-level visual change. The remaining eye/oblique cost is (a) the L0 voxel shell 60-250 m
(election/overdraw-bound — coarsening in that band is visual-risky), (b) ~3M remaining leaf tris,
(c) the 14 ms trunks+resolve base. Next levers in §3 order: election-traffic experiment (F1),
L0 occupancy gating / silhouette carve in the near voxel shell, leaflodk 0.15 sweep, rect fast
path (F2), then the structural bark-far-rep if trunks re-dominate.
