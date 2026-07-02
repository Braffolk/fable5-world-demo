# ue5-prior-art deep review (2026-07-02)

Area: UE5 Nanite + industry prior art, mapped subproblem-by-subproblem onto OUR pipeline.
Post-voxbocc baseline: eye 18.9 / oblique 37.2 / aerial 16.5 (isolated gpuWall med, 200k @
2268×1473). Money pose = oblique, gap ≈ −11..13 ms, quality-neutral only.

Unique asset of this area: the repo carries **UE 5.8's actual Nanite shader sources**
(`docs/perf-runs/Nanite-UE5-shaders/`, 61 files incl. the **`Voxel/` experimental voxel-brick
rasterizer**) plus 14 prior-art briefs (`docs/perf-runs/prior-art/`). Everything below is
grounded in those files or the measured fact pack, not in talk-slide folklore.

---

## Premise audit

1. **The prior-art briefs are calibrated to a DEAD cost model — re-rank, don't re-read.**
   All 14 briefs (IDEAS.md and the per-source briefs) attack the June-era model: "60%
   per-pixel coverage loop + 40% transform/launch, 36M visible tris". The 2026-07-02
   attribution (fact pack + `2026-07-02-attribution-and-waves.md` §2) replaced that model:
   leaf decode ≈ FREE, lighting/GI ≈ FREE, shadows ≈ absent (see #4), post ≈ 5-6 ms, and the
   foliage pool is **COUNT-bound, not fill-bound** (13-vox-raster premise §2: ~75-80% of
   oblique vox cost scales with cluster/brick count at equal coverage). Consequences:
   - The briefs' #1 convergent recommendation (shared-memory vertex transform, 7 sources)
     attacks a pool that is now ~2-4 ms at oblique (12-base-raster §work-model) and was
     already built (`vcompact`) and measured marginal. **Demoted.**
   - The briefs' inner-loop micro-opts (Q2/Q3 hoists, incremental z) attack the per-pixel
     term measured at ~1.2-1.6 ns/px ≈ 4-5 ms irreducible-ish at oblique. **Demoted.**
   - The briefs' "tiled raster" big bet (B1/B2/B3) was REFUTED in vivo (+11.7/+18.1 ms,
     SPEC D-N46, removed). Scatter is committed. **Stays dead without new evidence.**
   What survives re-ranking is the ORDERING/OCCLUSION family and per-brick footprint work —
   exactly where UE5's own voxel path concentrates (below).

2. **The fact pack's system description contains a stale claim: "resolve … CSM+clip
   shadows".** 17-shadows-gi premise §1-2 proved the canonical forest scene wires
   `csm: null, gi: null` (`src/debug/ForestScene.ts:457-459`) — the resolve compiles NO
   shadow sampling and no GI. Every VSM implication in this doc is therefore **world-scene
   contingent**; on the measured scene the whole shadow subsystem contributes 0.0 ms.

3. **"UE5" must mean TWO systems for us, not one.** The 2021 deep-dive (mesh micropoly
   Nanite) is the right comparison for our base/mesh path only. For the foliage pool (21.8 ms
   at oblique — THE gap) the right comparison is **Nanite Foliage's voxel path (UE 5.6-5.8)**,
   whose experimental source is in-repo (`Nanite-UE5-shaders/Voxel/*`) and whose shipped form
   is documented: crowns become clusters of **at most 128 4×4×4 voxel bricks** — the *same
   shape as ours* (`MAX_BRICKS_PER_CLUSTER=128`, `BRICK_DIM=4`, VoxelBrick.ts:61-75) — and
   "the clusters are **sorted into depth buckets and rasterized front to back** to gain some
   benefits of early Z testing" (UE 5.7 docs, sources at end). Our architecture is not a
   poor cousin; it is the same design point. The deltas are enumerable (below) and small in
   number.

4. **The voxf2b K16 verdict measured our BARRIER implementation, not UE5's idea**
   (tripwire #2, non-engaging mechanism). UE5's front-to-back is a **bucket-SORTED list
   consumed by an ordinary dispatch** — ordering comes from workgroup launch order, with the
   per-pixel depth test (their 64-bit `InterlockedMax` + `EarlyDepthTest()` pre-read,
   NaniteWritePixel.ush:48-65) converting "mostly-ordered" into "mostly 1-load early-outs".
   Our `?voxf2b=1` instead issues **K=16 barrier-serialized sub-dispatches**
   (NaniteVoxelRaster.ts:1482-1486, in-pass UAV auto-sync between buckets) — the measured
   +5/+6/+16 ms is the price of 16 pipeline drains, not of ordering. The barrier-free,
   UE5-shaped variant (ONE dispatch over the already-bucket-sorted list) has **never been
   measured** and all machinery for it exists (lever L1).

5. **Metric checks.** Expected-ms figures below are same-session isolated gpuWall med deltas
   and must clear the ±2 ms apparatus floor (90-premise-audit §3.7); oblique/aerial numbers
   carry the bimodality caveat (report mode structure). Two-pass-occlusion levers pay in the
   LIVE metric that isolated static poses cannot see (11-cull-and-hzb premise §6) — they are
   ranked on live p95 slot-histogram grounds, not med deltas.

---

## How it works today (ours ↔ UE5, subproblem by subproblem)

| Subproblem | OURS | UE5 (file:line, in-repo) | Verdict |
|---|---|---|---|
| Visibility election | 32-bit `atomicMax(depthKey24<<8\|id8)` + winner `atomicStore` side id (NaniteRaster.ts:952-975; NaniteVoxelRaster.ts:1398-1402) | 64-bit `ImageInterlockedMaxUInt64(depth32\|payload32)` when available, **depth-only 32-bit `InterlockedMax` otherwise** (NaniteWritePixel.ush:27-34). Their own voxel-scatter experiment uses **exactly our idiom**: `(DepthInt & 0xffffff00) \| (Id & 0xff)` into a 32-bit `InterlockedMax` (Voxel/ScatterBricks.usf:10-23) | **No gap.** Epic's own no-64-bit code is our packing. Election re-architecture permanently closed (also: atomic contention refuted). |
| Pre-write early-out | relaxed load + compare before RMW at both vox sites (NaniteVoxelRaster.ts:1266-1267, 1398) and mesh site (NaniteRaster.ts:952-956) | `EarlyDepthTest()` — plain read of the visbuffer word, bail if losing (NaniteWritePixel.ush:48-65) | **No gap.** Identical pattern. Its VALUE scales with front-to-back ordering — the gap is ordering, not the test (L1). |
| SW/HW raster split | tris >16 px or near-crossing → HW queue (NaniteRaster.ts:668-675, 729-731) | per-cluster binning, <32 px/edge → SW (deep-dive brief; NaniteRasterBinning.usf) | Same idea. One parity bug: our HW vertex stage fetches all 3 corners then selects (NaniteRaster.ts:1123-1128); UE5 fetches one (L5). |
| Scanline inner loop | shipped x-span solve (NaniteRaster.ts:848-873) | RasterizeTri_Rect/Scanline/Adaptive (NaniteRasterizer.ush) with >4px gate | **No gap**; further inner-loop work is measured-dead (12-base-raster §premise 6). |
| Cluster cull / HZB | single-phase, PREV-frame full-content HZB tested at emit only (NaniteCull.ts:919-925; NaniteHzb.ts:11-13) | **two-pass**: MAIN tests prev-HZB, occluded candidates are RECORDED (`WriteOccludedInstance`, NaniteInstanceCulling.usf:169-174; `GroupOccludedBitmask` re-enqueue, NaniteClusterCulling.usf:563-600) and POST re-tests them vs the fresh this-frame HZB (CULLING_PASS_MAIN/POST, NaniteCulling.ush:10-13) | **Real gap, live-metric.** Ours trades disocclusion holes + stale-cut inflation under motion. 11-doc L2/L5 are the ports; endorsed as L3/L7 here. |
| Voxel far foliage | per-tree brick clusters (45-140 m) + merged FarTiles (>140 m); one WG per cluster, Phase A projects **full brick cube**, Phase B rect-fill / ray-DDA with occupancy mask (NaniteVoxelRaster.ts:717-772, 1288-1381) | same 128×4³ brick clusters; **depth-bucket F2B ordering** (5.7 docs); PS ray-cast per brick with 2-level 64-bit occupancy DDA (Voxel/Voxel.ush:500-588); **bounds shrunk to the occupied-cell sub-box** before raster (`BlockBounds`, Voxel/Voxel.ush:10-31; RasterizeBricks.usf:62-90); HW path uses conservative depth (`SV_DepthLessEqual`+ReZ, RasterizeBricks.usf:12-24,149) for early-Z | Two portable deltas: **ordering without barriers** (L1) and **occupied-bounds shrink** (L2). Conservative-depth HW early-Z is NOT portable (WGSL has no `SV_DepthLessEqual`; `frag_depth` forces late-Z). |
| Voxel tiling | none (scatter committed) | experimental `TileBricks.usf`: 8×8-px tile linked lists → per-tile **insertion sort by depth** → per-pixel march with shrinking `Ray.Time[1]` + sorted-order `break` (TileBricks.usf:111-175, 199-299) | The per-tile form re-opens the REFUTED tiled-raster family — not proposed. The sorted-traversal PRINCIPLE ships barrier-free as L1/L6. |
| Material resolve | per-pixel decode in one resolve; measured ~FREE (leafcheap ceiling <1 ms) | shade binning per material (NaniteShadeBinning.usf), material-depth HiZ gating | **Anti-lever.** UE5's shade binning solves a cost we measured at ~0. Do not build. Confirmed dead by attribution. |
| Shadows | clipmap of 6 ortho levels, whole-level VP-equality caching (NaniteShadowClip.ts:271-339); voxel clusters never cast (NaniteRaster.ts:541-549). ABSENT in canonical scene | **Virtual Shadow Maps**: 16k virtual clipmaps, 128² pages, per-page cache/invalidation; Nanite rasters **depth-only 32-bit `InterlockedMax` into the physical page array** (NaniteWritePixel.ush:27-28, 87) via multi-view one-pass culling (ViewId plumbed through NaniteInstanceCulling.usf) | Our per-level cache = VSM at page-granularity ∞. World-scene levers live in 17-doc (storm budget = step toward pages). Note UE5 shadow raster being depth-only 32-bit confirms our shadow path needs no election tricks. |
| LOD transition stability | leaf-group (error,sphere) monotone cut, boundary-locked QEM — same reference construction (12-doc §UE5); but cut/occlusion tests consume the JITTERED VP (NaniteCommon.ts:108-115) | same DAG invariant guarantees crack-free, pop-free cuts with NO temporal hysteresis; culling views are separate `FNaniteView`s — TAA jitter belongs to raster/velocity, not to cull decisions (grep: `Jitter` appears only in velocity export, NaniteDepthExport.usf:130, NaniteExportGBuffer.usf:99) | **Real micro-gap:** our aerial ±9 ms period-3 oscillation is jitter-linked borderline cull flips on 661 huge FarTile clusters (11-doc premise §2). UE5 structurally cannot have this. Lever L6. |
| Streaming / assemblies | all-resident; FarTiles merges instances at boot | page streaming (NaniteTranscode.usf); **Assemblies** = instanced sub-hierarchies inside one resource (`AssemblyTransformIndex`, NaniteCulling.ush:21-27) — memory dedup for foliage, not a runtime raster saver | No action. FarTiles is our (stronger) runtime aggregation; UE5 has no cross-instance merge — they eat per-instance cluster counts and win via ordering + early-Z instead. |

**One-paragraph net read:** our pipeline is a faithful 32-bit port of the Nanite design and
in several places (FarTiles aggregation, brick-granular vox occlusion vs mesh) goes beyond
what UE5 ships. The portable residual is narrow and specific: (1) UE5 orders its voxel work
front-to-back **without barriers** and lets the per-pixel pre-read eat buried work — we
either don't order (default) or pay barriers for it (voxf2b); (2) UE5 never rasters a brick's
empty margin — occupied-cell bounds shrink; (3) UE5's culling is two-pass and
jitter-invariant — ours is single-phase and jittered; (4) UE5 splits raster work into
homogeneous bins per shader permutation — our vox kernel is a divergent monolith. Items
(1),(2) are S/M-effort in-area levers; (3),(4) are endorsed levers owned by 11-/13-docs.

---

## Work model

What the UE5 comparison explains about numbers we already have:

- **voxf2b=1 control +5/+6/+16 ms** = the cost of K=16 in-pass UAV barriers (16 pipeline
  drains at aerial's huge per-bucket variance), NOT of ordering. UE5 pays ~0 for ordering
  (list order + launch order). Model: barrier cost ≈ K × (drain + ramp) — at K=16, aerial's
  661 clusters cannot fill the machine between drains, hence +16 there vs +5 at eye.
- **voxbocc −17.2 eye / −6.0 obl** = what UE5's `EarlyDepthTest`+ordering would have gotten
  for free at eye (vox field behind near mesh canopy). The oblique residual is
  vox-behind-vox — reachable only by ordering (L1) or vox-inclusive two-pass (L3).
- **Count-bound slope 1.6-2.2 µs/vox-cluster** (13-doc): per-cluster prologue + per-brick
  Phase A + Phase-B setup ×4 SIMD groups. UE5's equivalents amortize differently (HW VS per
  brick, or persistent bins); the portable cut is footprint shrink (L2) + kernel split (L4),
  both of which shrink the per-brick constants, not the pixel term.
- **Election ≈ free, decode ≈ free** — matches Karis's own cost map ("inner loop does barely
  anything"; aggregates = overdraw-bound, deep-dive brief §cost-map). Nothing in UE5's
  per-pixel machinery would make our per-pixel term cheaper; their voxel PS DDA budget
  (`MaxTests = 3*16+3*4`, Voxel.ush:521) is BIGGER than our 6-step cap.

---

## Waste inventory (deltas vs UE5 practice, oblique-centric)

1. **Unordered vox dispatch** → buried bricks pay full ray/flat cost until a nearer key
   happens to land first. 13-doc sizes the phenomenon at ~1-3 ms oblique (ln(stack-depth)
   temporary-winner model). UE5 practice: depth-bucket order, no barriers. → L1.
2. **Full-cube brick footprints and slabs** → Phase-B rect covers the whole projected cube
   even when only a corner of the 4³ mask is occupied; ray misses + leading empty DDA steps
   + inflated `ceil(area/128)` rounds. UE5 shrinks to `BlockBounds` (both the drawn bounds
   AND the local ray slab). Sized with 13-doc waste #4 (bbox-vs-silhouette ~0.5-1 ms) plus a
   share of the Phase-B rounds term → est. **1-2 ms oblique**. → L2.
3. **Vox-behind-vox invisible to every same-frame cull** (voxOccPyr is mesh-only at scatter
   time). UE5's post-pass re-test makes far occludees testable against near occluders in the
   same frame. Brick-vs-vox is the unmeasured cell of the 2×2 matrix (11-doc). Best guess
   2-6 ms oblique upper pool, shared with L1. → L3.
4. **Monolith vox kernel divergence/occupancy** — flat path, ray path, mask build, dither all
   in one kernel; naive ALU model under-predicts measured slope 3-5× (13-doc open Q5). UE5
   raster-bins into homogeneous dispatches per path (NaniteRasterBinning.usf). Unmeasured;
   could be the multiplier on ALL per-brick terms. → L4 (probe first).
5. **Jitter-coupled cull flips** on FarTile-scale clusters: aerial ±9 ms period-3 swing
   (median impact ~1-2 ms, p95 4+). UE5 cull is jitter-free. → L6.
6. **HW vertex 3×-fetch** (eye-pose only, ~0.3-1.0 ms) — UE5 parity gap. → L5.
7. **NON-waste (UE5 solves problems we don't have):** shade binning (decode ≈ free);
   64-bit election (our 32-bit split-buffer is Epic's own fallback idiom); page streaming
   (all-resident is fine at our scale); assemblies (we already instance + merge).

---

## Levers

Ranked. Expected ms = same-session isolated gpuWall med savings (eye/oblique/aerial).
All must clear the 2 ms floor or use interleaved A/B; all shots-gated per doctrine.

### L1 — Barrier-free front-to-back: bucket-sorted list, ONE dispatch (UE5's shipped idiom)
- **Mechanism:** with `?voxf2b=1` the fanout already scatters qVoxRaster into K contiguous
  near→far depth slabs (NaniteCull.ts:553-710; NaniteVoxelRaster.ts:1431-1446 reads
  per-bucket ranges). Instead of K barrier-separated dispatches
  (NaniteVoxelRaster.ts:1482-1486), issue the EXISTING whole-list kernel `kVoxScatter`
  (:1424-1428) ONCE over the reordered list. Workgroup launch order ≈ list order on
  Metal/M1 ⇒ near clusters' elections mostly land before far clusters run ⇒ the existing
  `cand > prevE` guards (:1266-1267) collapse buried bricks to 1 load + compare. This is
  bit-for-bit UE5 Nanite Foliage's "sorted into depth buckets, rasterized front to back"
  with our 32-bit pre-read as the early-Z stand-in. No shader changes; one new flag
  (`?voxf2bone=1`) selecting the dispatch shape.
- **Quality class:** identical (atomicMax election is order-free; ordering only changes WHO
  pays, never who wins).
- **Expected:** 0.3 / 1.5 / 0.5 — bounded above by the buried-brick pool (~1-3 obl); cost
  side is the histogram+prefix+scatter chain (~5 small dispatches, est +0.2-0.5). Wide bars;
  the probe is nearly free.
- **Discriminator:** `CONFIG=default LABEL=ua-base TICKS=0 COOLDOWN_S=45 TREES=200000 npx
  tsx tools/probe-fresh-stutter.ts` then `CONFIG=default EXTRA=voxf2b=1,voxf2bone=1
  LABEL=ua-f2bone …` (after the S-effort flag lands). Null result requires the engagement
  check: confirm qVoxRaster really is slab-ordered in the f2b path (log bucket ranges).
- **Effort:** S. **Risks:** launch order is a scheduling *tendency*, not a guarantee —
  benefit may be partial (it can never be negative beyond the chain cost); the f2b fanout
  must leave qVoxRaster[0].x intact for the whole-list guard (verify).

### L2 — Occupied-bounds shrink (port of UE5 `BlockBounds`)
- **Mechanism:** every brick carries its 64-bit occupancy (words 0-1, VoxelBrick.ts:66-67).
  Phase A currently projects the full cube [center ± brWR] (NaniteVoxelRaster.ts:717-772);
  Phase B slab-tests the full cube (:1288-1298). Compute the occupied-cell AABB from
  occLo/occHi (UE5's `BlockBounds`, Voxel/Voxel.ush:10-31 — WGSL: `firstTrailingBit`/
  `firstLeadingBit`/`countOneBits` on 2 words, ~20 ALU once per brick in Phase A), shrink
  the projected bbox and the Phase-B slab to it, exactly as RasterizeBricks.usf:62-90 and
  TileBricks.usf:57-83 do. Cuts rect area (Phase-B rounds), ray misses, and leading empty
  DDA steps. Apply to RAY-eligible bricks; keep the flat path's bbox unshrunk in v1 (its
  painted rect is output-visible).
- **Quality class:** improving (needs sign-off): removing leading empty cells reduces the
  6-step-exhaustion **false fills** (:1375-1381) — some far sparse-brick pixels that today
  get conservative fill will correctly miss. That is a correctness gain but IS visible in a
  flip; a strictly-identical variant does not exist because exhaustion-fill is
  entry-point-dependent. Present far-band crops to the user with the perf number.
- **Expected:** 0.3 / 1.5 / 0.3.
- **Discriminator:** flag `?voxoshr=0` reverts; A/B all poses + far-band crops
  (`shots/*-oblique.png` zooms); watch the 13-doc waste-#4 pool.
- **Effort:** M. **Risks:** off-by-one in the 2-bit bounds pack = clipped brick edges
  (UE5 packs min/max in 12 bits — copy their encoding); slight thinning of far crowns must
  pass the user bar (it is the same artifact family the errorK calibration was done on).

### L3 — Vox two-pass deferral (UE5 MAIN/POST mapped onto the vox stack) — endorse 11-doc L2
- **Mechanism:** partition qVoxRaster by a prev-frame full-content-HZB test (that pyramid
  already includes last frame's vox, NaniteFrame.ts:481): "probably visible" scatters first,
  voxOccPyr rebuilds (now vox-inclusive), then the "probably occluded" list scatters with
  brick culls firing against real same-frame vox occluders. Deferral-not-drop ⇒
  byte-identical even under motion. This is UE5's CULLING_PASS_OCCLUSION_MAIN/POST
  (NaniteCulling.ush:10-13; NaniteInstanceCulling.usf:169-174) with "re-test" replaced by
  "defer", which is cheaper on our stack (no reject recording).
- **Quality class:** identical. **Expected:** 1 / 3 / 1 (11-doc band: obl −2..6, minus one
  pyramid rebuild ~0.1-0.3 + one submit).
- **Discriminator:** run the zero-code L1 probes first (`voxf2b=1,voxf2bk=2[,voxwaves=2]`,
  11-doc P2/P3) — if brick-vs-vox occlusion shows signal but the K-chain eats it, build
  this. Direct A/B: `?voxtwopass=0/1`.
- **Effort:** M. **Risks:** partition quality under fast motion (bad partition = no gain,
  no harm); one extra sync in dispatchVoxel.

### L4 — Raster-bin the vox monolith: flat-only + ray-only kernels (UE5 NaniteRasterBinning)
- **Mechanism:** UE5 never runs heterogeneous raster paths in one dispatch — clusters are
  binned per rasterizer bin and each bin gets a homogeneous indirect dispatch
  (NaniteRasterBinning.usf; deep-dive brief T4). Our vox kernel carries flat path + ray path
  + occ-mask build + dither in one register-fat monolith; 13-doc open-Q5 flags a 3-5×
  model-vs-measured multiplier suggesting occupancy/register pressure. Split: Phase A also
  routes brick records into two compacted lists (flat/small vs ray-eligible); two slim
  kernels consume them. Elections unchanged.
- **Quality class:** identical (same election set). **Expected:** 0.5 / 2.5 / 0.3 IF the
  occupancy multiplier is real — gate on the probe below.
- **Discriminator (before building):** 13-doc probe #5 — `CONFIG=default EXTRA=voxcell=0
  LABEL=ua-noray TICKS=0 COOLDOWN_S=45 …` (DIAGNOSTIC, quality-changing instrument): if the
  per-cluster slope collapses without the ray path compiled in, register pressure is the
  multiplier and the split pays. Also needs the `nanite.voxBricks` counter (13-doc probe #4).
- **Effort:** L. **Risks:** 10-buffer cliff (two more compacted lists — must fold into
  existing buffers per the scar precedent, NaniteRaster.ts:339-372); Phase A/B barrier
  restructure.

### L5 — HW vertex 1-fetch parity (endorse 12-doc L1)
- **Mechanism:** our HW path fetches 3 corners per vertex and selects
  (NaniteRaster.ts:1123-1128); UE5's HW path fetches one. Bit-identical output.
- **Quality class:** identical. **Expected:** 0.7 / 0.1 / 0 (eye-skewed; H=1.10M at eye).
- **Discriminator:** `EXTRA=hw1fetch=1` A/B, eye pose. **Effort:** S. **Risks:** TSL codegen
  for the dynamic corner index.

### L6 — Jitter-invariant cull decisions (UE5 parity; endorse 11-doc L6)
- **Mechanism:** UE5's cull views carry no TAA jitter (jitter only enters velocity export,
  NaniteDepthExport.usf:130). Our emit-time occlusion + cut tests consume the jittered VP ⇒
  borderline decisions on 661 huge FarTile clusters flip with Halton phase ⇒ aerial ±9 ms
  period-3 swing. Fix: evaluate occlusion (and optionally τ) with the unjittered VP or pad
  nearestZ by one HZB-texel — padding only ever KEEPS more.
- **Quality class:** identical (conservative padding). **Expected:** 0 / 0 / ~1 med, aerial
  p95 −2..4 (variance win, live-relevant).
- **Discriminator:** 11-doc P6 first (`EXTRA=ablate=taa FRAMES=64` at aerial — confirms
  linkage), then `?occleps` A/B. **Effort:** S-M. **Risks:** mis-padding loses culls
  (perf, never quality).

### L7 — Mesh two-pass occlusion (UE5 MAIN/POST for the mesh cut) — endorse 11-doc L5
- **Mechanism:** record occlusion-rejects, re-test vs fresh HZB, raster survivors
  (scaffolding exists: rejClust, NaniteCull.ts:97-99, 419-425). Kills disocclusion holes and
  licenses tighter phase-1 culling; the ~9-11 ms stale-HZB motion inflation (2026-06-26) is
  the pool, and live p95 is the metric.
- **Quality class:** improving (fixes a motion artifact; static identical). **Expected:**
  isolated 0 / 0 / 0; live p95 is the payoff — rank via the live slot histogram.
- **Effort:** M. **Risks:** frame-graph reshuffle (HZB build currently after everything).

### L8 — Within-cluster brick order by front-slab key (UE5 TileBricks sort, cluster-scoped)
- **Mechanism:** Phase A already computes each brick's front-slab key (`cand`,
  NaniteVoxelRaster.ts:891-902). Phase B walks bricks in memory order; sort the ≤128 shared
  records by key descending (bitonic in shared memory) so intra-cluster overlap resolves
  near-first and the guard eats the rest. Analog of TileBricks' per-tile insertion sort
  (TileBricks.usf:160-174) at cluster scope.
- **Quality class:** identical. **Expected:** 0.2 / 0.8 / 0.2 — only worth building if L1
  shows ordering pays at all.
- **Effort:** M. **Risks:** sort cost (~7 bitonic stages × 128 lanes) may exceed the gain on
  low-overlap clusters.

### REJECTED-BY-POLICY (listed for completeness; do not ship)
- **Visbuffer/billboard impostors for distant instances** (deep-dive T7, Scthe T5, IDEAS B4):
  Karis himself flags visible pop; SpeedTree crossfade is the same trade. Distance pop =
  Class R, red-listed. (SpeedTree's *shading* trick — crown-normal bending — is already
  shipped quality-side as `?voxbead`.)
- **Per-tile brick raster (TileBricks port), sort-middle tiling (IDEAS B1/B2/B3):** the
  tiled family was measured +11.7/+18.1 ms here and REMOVED; UE5's own tile path is
  experimental, not shipped. Re-litigation requires new evidence, none found.
- **aggdist<140 / voxlodk<1 / dpr<1.5 / coarser τ or K_FLOOR:** red-listed quality trades.
- **Shade-binning resolve:** premise measured dead (decode ≈ free) — an anti-lever we would
  copy from UE5 only to solve a problem we do not have.
- **64-bit-election emulation chase (F1 single-word experiments):** below noise; Epic's own
  32-bit fallback (NaniteWritePixel.ush:27-28) and voxel WritePixel (ScatterBricks.usf:10-23)
  are our exact scheme. Closed.

**Double-counting warning for the master plan:** L1, L3, L8 and the shipped voxbocc all
feed on the SAME buried-vox pool (~2-6 ms oblique); L2 and L4 both shrink the per-brick
constants. Realistic combined in-family ceiling ≈ **4-8 ms at oblique**, not the sum of the
table. The remaining oblique gap must come from the mid-ring cluster-count restructure
(premise-audit §3.4 — the 45-140 m per-tree ring, ~10 ms diagnostic ceiling, owned by the
foliage-aggregation area) and the post pool (~5.1 ms, owned by 16-resolve-and-post).

---

## What UE5/prior art does here (the explicit focus questions)

### How UE5 avoids paying for hidden geometry (two-pass occlusion)
Frame N: (1) MAIN pass culls instances → BVH nodes → clusters against the **previous
frame's HZB** reprojected to the current view; anything that fails is not dropped but
**recorded** (occluded-instance queue: NaniteInstanceCulling.usf:169-174; per-node bitmask:
NaniteClusterCulling.usf:563-600). (2) Visible survivors raster; the fresh HZB is built.
(3) POST pass re-tests ONLY the recorded rejects against the fresh HZB and rasters the
newly-visible remainder (CULLING_PASS_OCCLUSION_POST, NaniteCulling.ush:10-13). Net effect:
occlusion is always tested against REAL same-frame occluders, with prev-frame data used only
as a scheduling hint — no disocclusion holes, no stale-cull inflation, and hidden geometry
pays at most a cull test. Ours is single-phase prev-frame (emit-time only,
NaniteCull.ts:919-925): correct-ish when static, leaky under motion, and its vox-granular
analog (voxOccPyr) sees only mesh. L3/L7 are the ports; the deferral variant fits our stack
better than literal re-test (no reject storage, no 10-buffer pressure).

### How UE5 keeps LOD transitions stable
Structurally, not temporally: (a) cluster-group cut invariant — parent/child groups share
identical (error, bounding sphere) so the cut decision is IDENTICAL on both sides of every
boundary regardless of which node evaluates it (we implement the same pairing,
BuildDag.ts); (b) the error is projected with a monotone, view-dependent but
JITTER-FREE metric — TAA jitter never enters cull/LOD views (only velocity export references
jitter, NaniteDepthExport.usf:130); (c) no hysteresis, no crossfade — sub-pixel error makes
switches invisible by construction; TAA merely mops up the sub-pixel shimmer. Where WE
deviate: τ is warp-coarsened (lodWarp defaults, 11-doc premise §3) so switches are NOT
sub-pixel in the mid band (user-visible, the "too coarse" complaint), and our cull/cut runs
on the jittered VP (L6). SpeedTree-practice contrast: discrete LODs + dissolve crossfade +
billboard finale — every step of that ladder is a Class-R trade we've rejected; the one
SpeedTree idea worth keeping (crown-normal shading for volumetric read) already shipped as
`?voxbead`.

### What Virtual Shadow Maps imply for our shadow-clip design
VSM = one 16k virtual clipmap per light, 128²-texel pages, page table + physical pool;
pages are CACHED and re-rastered only when invalidated (caster moved, receiver page newly
requested, light moved). Nanite rasters shadow depth **depth-only via 32-bit
`InterlockedMax` into the physical page array** (NaniteWritePixel.ush:27-28, 87) with all
clipmap levels culled in ONE multi-view Nanite pass (ViewId through the same culling code).
Implications for us, in order of relevance:
1. **Our clipmap-with-VP-equality-cache (NaniteShadowClip.ts:271-339) is VSM at
   page-granularity = whole-level.** The delta is invalidation granularity, and it only
   matters on the world scene while MOVING (E[levels/frame] ≈ 2.7-5.1, 17-doc work model).
   The cheap step toward pages is 17-doc's storm-budget lever (cap levels/frame,
   finest-first), not a page-table build.
2. **Depth-only 32-bit atomics suffice for shadows** — no election problem exists in the
   shadow path on any platform; keep depth1 as-is.
3. **One-pass multi-view culling**: our shared-cut across clip levels (fitCut,
   NaniteShadowClip.ts:343-378) is the same amortization; `?culloverlap=1` (built,
   default-off) is the remaining submit-overlap step.
4. **Coverage gap UE5 does NOT have:** their voxel bricks render into VSM; our voxel
   clusters are skipped by every shadow raster (NaniteRaster.ts:541-549) ⇒ crowns beyond
   45 m and ALL FarTiles cast nothing. Quality-improving backlog item (17-doc lever 7),
   costs perf, world-scene only.
5. Small-caster dropout + screen-space contact shadows is how Epic bounds VSM cost — that is
   a quality trade we keep OFF (shadowminpx=0); do not import.

### Nanite Foliage's voxel path (UE 5.6-5.8), vs ours
Shipped shape (5.7 docs + in-repo experimental source): voxel-enabled meshes simplify into
clusters of ≤128 4×4×4 bricks; clusters sort into depth buckets and raster front-to-back;
per-pixel brick ray-cast with two-level 64-bit occupancy DDA (Voxel.ush RayCastBrick_L2*,
budget 3·16+3·4 tests); brick bounds shrunk to occupied cells (`BlockBounds`); HW path uses
conservative-depth PS (`SV_DepthLessEqual` + ReZ) so early-Z kills buried brick pixels;
an experimental compute scatter (ScatterBricks) projects each occupied CELL to one pixel
with our exact 32-bit packed InterlockedMax; an experimental tile path (TileBricks) does
8×8-px linked-list binning + per-tile depth insertion sort + shrinking-tmax march. Leaves
are modeled as opaque voxelized volume — no alpha-tested cards (their old worst case) —
which is precisely our two-tier choice. **Deltas we can port: the barrier-free F2B (L1) and
BlockBounds (L2). Deltas we cannot: conservative-depth early-Z (no WGSL equivalent),
64-bit fused election (irrelevant — they fall back to our scheme themselves).** Deltas in
OUR favor: FarTiles cross-instance merging (they pay per-instance cluster counts at
distance), brick-vs-MESH occlusion pyramid (voxbocc — their early-Z equivalent, which we got
without a depth target), gap-preserving coverage masks at aggregate scale.

### Other prior art, one line each (full briefs in docs/perf-runs/prior-art/)
- **CudaRaster / CuRast / FreePipe / LucidRaster / ComputeRaster / paraLLEl-GS:** the
  sort-middle tiling lineage — measured refuted on our stack; their remaining live gifts
  (sample-miss cull, incremental DDA, hoisted setup) are already in the kernel
  (NaniteRaster.ts:748-761, 848-873).
- **Scthe nanite-webgpu / Bevy meshlet:** the two WebGPU peers; both confirm the constraint
  (Bevy panics on M1; Scthe eats 16-bit depth) and neither has an inner-loop trick we lack;
  both reach for impostors/HW-routing = policy-rejected or already present.
- **Schütz point rasterizer:** batch-amortized setup + relaxed-load-then-atomic — we match;
  its 64-bit election is non-portable.
- **Tellusim compute-raster:** M1 ground truth that compute SW raster beats HW 1.68× for
  tiny prims — our architecture bet, confirmed.
- **Granite mesh-shader pipeline:** micro-poly bbox reject + exact FP32 orientation guard —
  the reject is in; the FP32-guard trick is available if any orientation math ever needs
  int64-free determinism.

---

## Open questions + proposed serial probes

GPU probes are serialized outside this workflow. All TICKS=0, COOLDOWN_S=45, TREES=200000,
baselines before candidates, same session, shots archived.

1. **P-A (zero-code, runs first — shared with 11-doc P2/P3): does vox ordering pay at all?**
   - `CONFIG=default LABEL=ua-base npx tsx tools/probe-fresh-stutter.ts`
   - `CONFIG=default EXTRA=voxf2b=1,voxf2bk=2 LABEL=ua-k2ctl npx tsx tools/probe-fresh-stutter.ts`
   - `CONFIG=default EXTRA=voxf2b=1,voxf2bk=2,voxwaves=2 LABEL=ua-k2w2 npx tsx tools/probe-fresh-stutter.ts`
   (P-A2−P-A1 = 2-bucket barrier tax; P-A3−P-A2 = brick-vs-vox occlusion gain.)
2. **P-B (after the S-effort `?voxf2bone` flag lands): L1 the UE5 idiom itself.**
   - `CONFIG=default EXTRA=voxf2b=1,voxf2bone=1 LABEL=ua-f2bone npx tsx tools/probe-fresh-stutter.ts`
   Engagement check on null: dump voxBucketRange (bucket bases/counts) once to confirm the
   list really is slab-ordered; shots must be pixel-equivalent to ua-base.
3. **P-C (before L4): is register pressure the 3-5× multiplier?**
   - `CONFIG=default EXTRA=voxcell=0 LABEL=ua-noray npx tsx tools/probe-fresh-stutter.ts`
   (DIAGNOSTIC only — quality-changing instrument; slope collapse ⇒ build the kernel split.)
4. **P-D (L6 confirmation, shared with 11-doc P6):**
   - `CONFIG=default EXTRA=ablate=taa FRAMES=64 LABEL=ua-taaoff npx tsx tools/probe-fresh-stutter.ts`
   (aerial period-3 oscillation must vanish per fresh-ablate-post3.)
5. **P-E (L2 after build):** `CONFIG=default EXTRA=voxoshr=0 LABEL=ua-oshr-ctl …` vs default
   ordering reversed (candidate first is fine here — bias against reverts), plus far-band
   oblique crops for the user sign-off (Class Q).
6. **Open question for the master plan:** does the product ship the forest scene or the
   world scene? (17-doc Q1 — decides whether VSM-side levers and the vox-shadow quality gap
   enter the budget at all.)
7. **Open question:** Metal workgroup launch-order fidelity — if L1 under-delivers with a
   confirmed-sorted list, capture one frame with WebGPU-Inspector and check election-order
   statistics before concluding ordering is dead (tripwire #2).

## Sources (external)
- [UE 5.7 documentation — Nanite Foliage](https://dev.epicgames.com/documentation/en-us/unreal-engine/nanite-foliage) (≤128 4×4×4 bricks/cluster; depth-bucket front-to-back; voxels beat triangles at distance)
- [The Future of Nanite Foliage — Unreal Fest Stockholm 2025](https://www.youtube.com/watch?v=aZr-mWAzoTg)
- In-repo UE 5.8 shader study copy: `docs/perf-runs/Nanite-UE5-shaders/` (README = file origins)
- Prior-art briefs + synthesis: `docs/perf-runs/prior-art/*.md` (14 briefs; IDEAS.md portfolio — re-ranked by this doc's premise audit)

---

## Reconciliation & verification (2026-07-02, post-limit continuation)

Adversarial verify pass over this doc + siblings `01-ue5-nanite-culling-raster.md` and
`02-ue5-materials-shading.md` (the duplicate-fleet coverage of this area). Every load-bearing
file:line below was re-read against the actual sources; every ≥1 ms number re-derived from the
scratchpad JSONs (`fresh-*.json`, medians recomputed from the raw `gpu` arrays). Sibling docs
carry a pointer blockquote; this section is the merged truth for the area.

### Measured anchors re-verified (recomputed from raw JSONs)

| Claim | Recomputed | Verdict |
|---|---|---|
| voxf2b=1 control +5/+6/+16 | `fresh-voxf2b-ctl` 41.0/49.4/32.8 vs `fresh-bead-v2-base` 36.1/43.2/16.8 = **+4.9/+6.2/+16.0** | CONFIRMED |
| voxwaves=4 vs f2b ~0/−0.6/−4.7 | `fresh-voxwaves4` 41.1/48.8/28.1 vs f2b-ctl = **+0.1/−0.6/−4.7** | CONFIRMED |
| voxbocc −17.2 eye / −6.0 obl | `fresh-voxbocc` 18.9/37.2/16.5 vs bead-v2-base 36.1/43.2/16.8 | CONFIRMED |
| leaf decode ≈ free (leafcheap=all ceiling <1 ms) | `fresh-leafcheap` 35.2/42.4/16.3 vs same-session bead-v2-base = −0.9/−0.8/−0.5 | CONFIRMED |
| lighting/shadows ≈ free at canonical poses | `fresh-attr-flat` 29.9/39.3/17.6 and `fresh-attr-noshadow` 32.6/37.0/15.6 vs SAME-SESSION `fresh-final-rested` 31.2/38.8/14.6 — all deltas within the ±2 ms floor. (Compared against the wrong-session bead-v2-base they'd look like 4-6 ms; session pairing matters.) | CONFIRMED |

### Corrections

1. **The "UE5 32-bit fallback" story — doc 02 §2.1 KILLED, doc 01 §2.6 confirmed, this doc's
   table row 1 rephrased.** NaniteWritePixel.ush verified: `DEPTH_ONLY` → 32-bit
   `InterlockedMax(OutDepthBuffer…)` (line 28); `COMPILER_SUPPORTS_UINT64_IMAGE_ATOMICS` →
   `ImageInterlockedMaxUInt64` (line 31); **otherwise `#error UNKNOWN_ATOMIC_PLATFORM` (line
   33)**. There is NO shipping 32-bit visibility-buffer fallback and NO "separate payload
   store" anywhere in the file — doc 02's "the fallback is depth-only InterlockedMax +
   separate payload store, i.e. OUR architecture" is false. The depth-only path is the
   shadow/VSM *target mode*, not a vis-buffer fallback. The correct validation of our
   `depthKey24<<8|id8` idiom is the **experimental** Voxel/ScatterBricks.usf:10-23 (verified:
   `(DepthInt & 0xffffff00) | (TriIndex & 0xff)` into a 32-bit `InterlockedMax`). Net verdict
   unchanged — election re-architecture stays closed — but for the corrected reason: Epic
   *refused* to solve 32-bit for meshes and only reached for our exact scheme in the voxel
   experiment.
2. **L1's open risk is CLEARED — effort confirmed S.** kVoxPrefix writes
   `qVoxRaster[0]=(total,0)` (NaniteCull.ts:661, verified) AND still publishes the whole-list
   indirect args (`split2D(voxRasterDispatch, total)`, NaniteCull.ts:663, verified); the
   whole-list `kVoxScatter` guards on `qVoxRaster[0].x` (NaniteVoxelRaster.ts:1426, verified).
   So `?voxf2bone` really is "fanout on + one whole-list dispatch", no new kernels. Bucket
   overflow clamps identically in both paths (QVOX_CAP at NaniteCull.ts:652-655, drop-guard
   :704) — loss parity holds.
3. **L2's quality class reclassified: improving → RISK.** The change is only reachable through
   removing 6-step-exhaustion false fills (verified NaniteVoxelRaster.ts:1375-1381) — far
   sparse-brick pixels flip fill→miss, i.e. far crowns thin slightly. Arguably more correct,
   but it is a visible far-band change in exactly the artifact family the user calibrated
   errorK on. Under the user law that is RISK: gate = `?voxoshr=0` revert flag + far-band
   oblique crops + explicit user sign-off + shotdiff report. No strictly-identical variant
   exists (entry-point-dependent exhaustion), as the lever text already admitted.
4. **Two levers from sibling 01 were missing from this doc's table — added** (see updated
   table below): **L9** = 4×4-texel occlusion windows one-two mips FINER (UE5
   `MipLevelForRect`/`GetMinDepthFromHZB(bSample4x4)`, NaniteHZBCull.ush:40-43/101/135-172,
   consumed with footprint=4 at NaniteCullingCommon.ush:599-602 — all verified) vs our 2×2 at
   diameter-fits-one-texel (NaniteHzb.ts:175-176/187-198; same idiom NaniteVoxelRaster.ts:
   636-659 block + 846-879 voxbocc brick — all verified). **L10** = zero-coverage cluster cull
   at emit (`bOverlapsPixelCenter`, NaniteHZBCull.ush:91, consumed NaniteCullingCommon.ush:604
   — verified; "~5% fewer clusters" per Epic's comment). Both Class IDENTICAL with shotdiff-0
   gates. Note L9's oblique estimate in doc 01 (−1.5..−4) double-counts with L1/L3's
   buried-vox pool — this doc's work model says the oblique vox residual is vox-behind-vox,
   which a finer WINDOW alone cannot see (the pyramid is mesh-only at scatter time); banded
   down to 0.5-2 obl and gated on the brick-skip counter.
5. **Doc 01 L5 (adaptive rect walk for ≤4 px tris) demoted, resolving the sibling
   contradiction in this doc's favor.** Mechanism real (RasterizeTri_Adaptive verified,
   NaniteRasterizer.ush:292-299) but its own estimate (0.3-1 ms eye) is under the 2 ms
   apparatus floor and 12-base-raster measured the inner-loop family dead. Not in the
   surviving table; revisit only if a future change makes the base per-row term dominant.
6. **`?reskeep` (sibling 02's one perf lever) verified still LIVE** — the default is still
   keep-ON (`keepOn = q.get('reskeep') !== '0'`, NaniteResolve.ts:286-291; runtime gate
   keepFullU at :951). The memory-note "reskeep shipped" refers to the flag/instrument, not a
   default flip. Doc 02's P-A within-boot A/B stands, owned by 16-resolve-and-post; expected
   0-1.5 ms, Class IDENTICAL (keep≡1 empty-map invariant) + shotdiff-0.
7. **"Main HZB is full-content (built after dispatchVoxel)" — mechanism verified with the
   missing link.** NaniteFrame.ts:481 alone doesn't show it; dispatchVoxel is nested INSIDE
   `raster.world1` (NaniteRaster.ts:1423, verified), which runs at NaniteFrame.ts:475, before
   `hzb.build` at :481. Claim stands.
8. **L6 quality nuance tightened.** Only the pad-nearestZ-by-one-texel variant is Class
   IDENTICAL (padding only ever keeps more). The "evaluate with the unjittered VP" variant is
   NOT strictly conservative on its own (cull VP ≠ raster VP by a sub-pixel offset) — ship it
   only WITH the padding. Jitter-free UE5 cull confirmed by grep: `Jitter` appears only in
   velocity-export comments (NaniteDepthExport.usf:130, NaniteExportGBuffer.usf:99).
9. **Cite audit:** all other load-bearing cites in this doc re-verified true, including
   BlockBounds (Voxel/Voxel.ush:10-31), RayCastBrick_L2 + `MaxTests = 3*16+3*4`
   (Voxel/Voxel.ush:500-588, :521-522), conservative-depth HW path
   (RasterizeBricks.usf:12-24 `SV_DepthLessEqual`, :149 `ENABLE_RE_Z`), 12-bit occupied-bounds
   pack (RasterizeBricks.usf:62-90; TileBricks.usf:57-83), per-tile insertion sort
   (TileBricks.usf:160-174), depth-bucket-major bin allocation (NaniteRasterBinning.usf:241,
   :354-390, :461-472), two-pass enum + occluded-instance recording + bitmask re-enqueue
   (NaniteCulling.ush:10-11; NaniteInstanceCulling.usf:169/400/361;
   NaniteClusterCulling.usf:563-600; NaniteCullingCommon.ush:620-662), DepthToBucket
   (NaniteClusterCulling.usf:314/854), SGGX voxel shading (NaniteVertexFactory.ush:902/917),
   our K-barrier dispatch shape (NaniteVoxelRaster.ts:1482-1486), pre-write guards
   (:1266-1267, :1398-1402; NaniteRaster.ts:963-975), full-cube Phase A/B footprints
   (:717-772, :1288-1298), BRICK_MAX_EXT=64 (:113), HW 3-corner fetch
   (NaniteRaster.ts:1123-1128), shadow vox-skip (:541-549), emit-time single-phase occlusion
   (NaniteCull.ts:917-925), phase-2 scaffolding (REJ_CLUST_CAP NaniteCull.ts:99, kRasterArgs2
   :482-488), `gi:null`/`csm:null` (ForestScene.ts:457/459), MAX_BRICKS_PER_CLUSTER=128 +
   BRICK_DIM=4 (VoxelBrick.ts:81/:61).

### Surviving lever table (merged, post-verification)

| # | Lever (mechanism 1-liner) | eye/obl/aerial ms | Quality | Probe (discriminator) | Effort | Conf |
|---|---|---|---|---|---|---|
| L1 | Barrier-free F2B: f2b fanout + ONE whole-list dispatch (`?voxf2bone`) | 0-0.5 / 1-3 / 0-0.5 | IDENTICAL | P-B; engagement: `nanite.voxBrickWrites` −≥25% + bucket-range dump | S | med |
| L3 | Vox two-pass deferral (MAIN/POST on the vox stack, defer-not-drop) | 1 / 3 / 1 | IDENTICAL | P-A (`voxf2b=1,voxf2bk=2[,voxwaves=2]`) first, then `?voxtwopass` A/B | M | med |
| L9 | 4×4 occlusion windows 1-2 mips finer (HZB + voxOccPyr block/brick) | 0.5 / 0.5-2 / 0.3 | IDENTICAL | `?occwin=4`; brick-skip counter +≥20% AND −≥2 ms obl | S-M | med |
| L2 | Occupied-bounds shrink (BlockBounds port), ray path only | 0.3 / 1.5 / 0.3 | **RISK** (far crowns thin; crops + user sign-off + `?voxoshr=0`) | P-E | M | med |
| L4 | Raster-bin the vox monolith (flat/ray kernel split) | 0.5 / 2.5 / 0.3 | IDENTICAL | P-C (`voxcell=0` slope diagnostic) BEFORE building | L | low-med |
| L6 | Jitter-invariant cull: pad nearestZ one HZB texel (± unjittered VP) | 0 / 0 / ~1 med, aerial p95 −2..4 | IDENTICAL (padding variant ONLY) | P-D (`ablate=taa`) then `?occleps` A/B | S-M | med |
| L7 | Mesh two-pass occlusion (record rejects, re-test vs fresh HZB) | ~0 iso; live p95 lever | IMPROVING (fixes disocclusion; static shotdiff-0) | doc-01 P-L1a counter-only build, then live slot histogram | L | med |
| L5 | HW vertex 1-fetch parity | 0.7 / 0.1 / 0 | IDENTICAL | `?hw1fetch` A/B at eye | S | med |
| L10 | Zero-coverage cluster cull at emit (pixel-center rect) | 0.3-1 / 0.3-0.8 / ~0 | IDENTICAL (outward-rounded only) | new flag A/B + shotdiff-0 | M | low |
| L8 | Within-cluster brick sort by front-slab key | 0.2 / 0.8 / 0.2 | IDENTICAL | only if L1 shows ordering pays | M | low |
| — | `?reskeep=0` default flip (owned by 16-doc) | 0-1.5 / 0-1.5 / ~0 | IDENTICAL (keep≡1) | doc-02 P-A within-boot `setKeepFull` A/B | S | med |
| — | SGGX stochastic voxel normal from SPREAD (owned by resolve/quality) | ~0 perf | IMPROVING (crops + user sign-off; jitter-stable noise) | doc-02 P-B crops | M | med |

Double-counting warning unchanged: L1/L3/L8/voxbocc share the buried-vox pool (~2-6 ms obl);
L2/L4 share the per-brick constants; **L9 also overlaps the buried-vox pool** at oblique.
Realistic combined in-family ceiling stays ≈ 4-8 ms at oblique.

### Killed claims

- **Doc 02 §2.1:** "UE5's 64-bit fallback is depth-only InterlockedMax + separate payload
  store, i.e. OUR architecture" — no such path exists; `#error UNKNOWN_ATOMIC_PLATFORM`
  (NaniteWritePixel.ush:33). The payload-store idiom exists only in experimental
  Voxel/ScatterBricks.usf.
- **This doc, table row 1 phrasing:** "depth-only 32-bit InterlockedMax otherwise" as a
  vis-buffer fallback — depth-only is a target MODE (shadows/VSM), not a fallback tier.
- **Doc 01 L5 as a standalone lever:** adaptive rect walk — real mechanism, sub-floor
  expected ms, family already measured dead (12-doc). Demoted to drawer.
- **Doc 01 L2's oblique upper band (−4 ms):** over-counts — the finer window cannot see
  vox-behind-vox (pyramid is mesh-only at scatter); banded to 0.5-2 obl pending the
  brick-skip-counter probe.
