# 01 — UE5 Nanite reference: culling + rasterization, mapped onto ours

Reader: deep-review 01. Sources: the 54 real UE5 shaders in `docs/perf-runs/Nanite-UE5-shaders/`
(read: NaniteCulling.ush, NaniteHierarchyTraversal.ush, NaniteInstanceCulling.usf,
NaniteClusterCulling.usf, NaniteCullingCommon.ush, NaniteHZBCull.ush, NaniteWritePixel.ush,
NaniteRasterizer.ush/.usf incl. the NANITE_VOXELS section, NaniteRasterizationCommon.ush,
NaniteRasterBinning.usf, NaniteDepthExport.usf, NaniteFastClear.usf, Voxel/Voxel.ush) and our
`src/nanite/{NaniteCull,NaniteHzb,NaniteRaster,NaniteVoxelRaster,Tsl}.ts` (read completely).
All measured numbers from the 2026-07-02 table / premise audit. Targets (premise audit §2.3):
eye ≤18, oblique ≤21, aerial ≤16 iso med; current effective baseline 18.9 / 37.2 / 16.5.

## 1. TL;DR

- Biggest UE5 mechanism we lack: **two-pass occlusion** (prev-HZB main pass + fresh-HZB post
  pass that re-tests the rejects). Ours is single-phase prev-HZB → the measured moving-camera
  cut inflation (~9–11 ms, 2026-06-26) and the aerial pose-arrival over-cull ramp are exactly
  the disease this cures. Live-p95 lever, ~0 at static iso.
- UE5's HZB/occlusion tests use a **screen-space RECT with a 4×4-texel footprint one mip finer**;
  ours use a sphere with a 2×2 window one mip coarser — our bound is systematically looser in
  BOTH the cluster cull and the voxOccPyr block/brick cull. S-effort, Class-IDENTICAL tightening.
- UE5 orders SW raster front-to-back by a **depth-bucket counting sort inside ONE dispatch**
  (no barriers). Our voxf2b null (+5/+6/+16 ms) measured K barrier-separated dispatches — the
  UE5-shaped variant (sorted queue, single dispatch) is UNTESTED and the machinery already exists.
- Their 32-bit story: **there is no 32-bit vis-buffer fallback** — `#error UNKNOWN_ATOMIC_PLATFORM`
  (NaniteWritePixel.ush:33). 32-bit InterlockedMax exists only for DEPTH_ONLY targets. Our
  depthKey24|id8 election is a solved problem they refused to solve; no action.
- Dispatch overhead for many tiny clusters: no gap — UE5 also runs 1 workgroup/cluster from one
  indirect dispatch; their persistent-threads trick is for the CULL, which for us is ~0.2 ms (refuted).

## 2. How UE5 does it (operational playbook, with cites)

### 2.1 Two-pass occlusion — the frame skeleton
Pass enum: `CULLING_PASS_OCCLUSION_MAIN=1 / _POST=2` (NaniteCulling.ush:9-12). The frame is:

1. **Main instance cull**: frustum + distance + HZB test against the **previous frame's** HZB
   using **previous-frame transforms** (NaniteCullingCommon.ush:620-645; the prev-pair keeps the
   test self-consistent). Visible instances seed the node queue; instances that fail ONLY the HZB
   test are **recorded, not dropped** — `WriteOccludedInstance` appends (ViewId, InstanceId) to
   `OutOccludedInstances` (NaniteInstanceCulling.usf:397-402, writer at 169-175).
2. **Main node+cluster cull** (persistent threads, §2.2): nodes that fail the prev-HZB test set a
   per-child bit in `GroupOccludedBitmask`; child 0 re-packs the node with that `EnabledBitmask`
   into the **post-pass node queue** `PassState[1]` (NaniteClusterCulling.usf:561-606). Clusters
   that fail HZB are appended to the far end of `CandidateClusters` for the post pass, keeping
   their computed SW/HW flag (NaniteClusterCulling.usf:941-962).
3. **Main raster** (SW+HW) of the surviving clusters → **build fresh HZB** from the result.
4. **Post pass**: re-run instance cull over the occluded-instance list with `bSkipCullFrustum=true`
   (already frustum-tested; NaniteInstanceCulling.usf:359-363), re-run node/cluster cull over the
   recorded rejects, but now the HZB test is against the **current frame's** HZB with current
   transforms (NaniteCullingCommon.ush:646-662, `CULLING_PASS_OCCLUSION_POST`). Survivors raster
   as an appended range (`OffsetClustersArgsSWHW`, NaniteClusterCulling.usf:697-698, 1004-1007).
5. Final HZB rebuild feeds next frame's main pass.

Net effect: the main pass may over- or under-cull under motion; the post pass **rescues every
false reject within the same frame** (zero popping) while keeping the main pass maximally
aggressive. One subtle motion detail: a box that was OUTSIDE the prev frustum is still HZB-tested
with a clamped rect — "a better guess for occlusion than assuming true… improves overdraw at
edges of screen for moving cameras" (NaniteCullingCommon.ush:628-631).

### 2.2 Persistent-threads hierarchy traversal
One dispatch of persistent workers drains an MPMC queue of BVH nodes and cluster batches
(NaniteHierarchyTraversal.ush:216-364): workers prefer nodes (critical path), fall back to
cluster batches while waiting; `QueueState.NodeCount` is incremented conservatively before
children are written and decremented after (NaniteHierarchyTraversal.ush:103-105, 180-185);
readiness is detected by writing 0xFFFFFFFF-cleared slots (`IsNodeDataReady`,
NaniteClusterCulling.usf:645-661). This exists because tree depth × dependent dispatches would
underutilize the GPU. **Mapping to us**: our BFS ping-pong (NaniteCull.ts:751-1010) is
2×hierDepth≈36 in-pass-serialized dispatches, but batched into ONE submit (NaniteCull.ts:998-1010)
and the whole cull is ~0.2 ms (refuted as a cost, base-raster review 2026-06-26). No lever here.

### 2.3 Cluster cull details worth copying
- **LOD cut**: `ShouldVisitChildInternal` = projected edge scale vs `MaxParentLODError`
  (NaniteClusterCulling.usf:258-285), `SmallEnoughToDraw` for emit (287-312) — same shape as our
  `pOwn ≤ τ_eff` cut (NaniteCull.ts:829-861).
- **SW/HW split decided at CULL time, cluster granularity**: `bUseHWRaster |= ProjectedEdgeScale <
  HWEdgeScale·|EdgeLength|·LODScaleHW` (NaniteClusterCulling.usf:301-309) plus `bNeedsClipping`
  (near/global-plane crossings, 860) ⇒ a cluster goes to exactly ONE raster path; SW clusters are
  written from the bottom of the visible-cluster buffer, HW from the top
  (`EmitVisibleCluster`, 688-720), and `CalculateSafeRasterizerArgs` (988-1035) trims overlap.
- **Depth bucket stashed per cluster at cull time**: `DepthToBucket(dot(ViewForward, center))`
  with a log2 distribution over [MinZ,MaxZ] (NaniteClusterCulling.usf:314-327, 852-855) — consumed
  by binning (§2.5), NOT by extra passes.

### 2.4 HZB test mechanics (tighter than ours in three ways)
`GetScreenRect` projects the full AABB (8 corners, `BoxCullFrustumPerspective`,
NaniteHZBCull.ush:444-540) to an NDC rect, then converts to a pixel rect counting only pixels
whose **CENTER** is covered — "typically ~5% fewer clusters drawn" (NaniteHZBCull.ush:80-91);
`bOverlapsPixelCenter=false` ⇒ the cluster rasters nothing and is dropped (zero-coverage cull,
consumed at NaniteCullingCommon.ush:604). `MipLevelForRect` picks the level via `firstbithigh`
so a **4-texel-wide footprint** fits (DesiredFootprintPixels=4 at NaniteCullingCommon.ush:599-602),
i.e. the test samples a 4×4 window **one-two mips finer** than a 2×2 scheme
(`GetMinDepthFromHZB` bSample4x4, NaniteHZBCull.ush:135-172). Finer mip = tighter depth bound =
more culled, still conservative.

**Ours**: sphere, nearest-point z, 2×2 window at the level where the sphere DIAMETER fits one
texel (NaniteHzb.ts:163-205, level pick at 175-176) — one-two mips coarser than UE5 for the same
footprint, and a sphere rect is looser than the true AABB rect. The same 2×2-coarse idiom is
copied in the voxel occlusion pyramid tests (block: NaniteVoxelRaster.ts:636-659; per-brick
?voxbocc: 846-879).

### 2.5 SW rasterizer
- **Per-cluster vertex staging**: transform up to 256 verts into LDS once (SoA
  `GroupVertsX/Y/Z`, NaniteRasterizer.usf:307-324, loop at 782-821), then 128 threads × triangles
  read 3 verts each from LDS. Ours re-fetches + re-transforms 3 verts per triangle thread
  (NaniteRaster.ts:634-640; `?vcompact` cache exists, measured marginal — consistent with the
  LEVER-#1 finding that the per-PIXEL loop is ≈90% of world1).
- **Triangle setup**: verts snapped to 1/256 px (NANITE_SUBPIXEL_BITS=8 — identical to our snap,
  NaniteRaster.ts:710-715), edge functions in float with a top-left saturate bias
  (NaniteRasterizer.ush:28-130). SW bbox clamp = MinPixel+63 (line 73); ours = 16 px
  (NaniteRaster.ts:102).
- **Adaptive inner loop**: `RasterizeTri_Adaptive` — if the wave has any tri wider than 4 px use
  the scanline (exact x-span from edge crossings), else the plain rect walk with NO span math
  (NaniteRasterizer.ush:291-300). Ours always pays the span solve — 3 f32 divides + floor/ceil
  per ROW (NaniteRaster.ts:848-868) even for 1-2 px tris, and our cut is 97% sub-pixel tris
  (base-raster memory).
- **Pixel write**: `TNaniteWritePixel` does `EarlyDepthTest()` (a PLAIN read of the vis buffer
  before the atomic) **only when the material is pixel-programmable**
  (ENABLE_EARLY_Z_TEST, NaniteRasterizationCommon.ush:33, 344-352); fixed-function pixels go
  straight to the 64-bit `ImageInterlockedMaxUInt64` (NaniteWritePixel.ush:29-31). Rule they
  encode: pre-test only when the per-pixel work being skipped is expensive. Our relaxed-load
  guard before atomicMax (NaniteRaster.ts:962-975) is the same idea applied to the cheap path
  too — fine on our HW (guard measured a win; `?noguard` exists as control).
- **HW raster**: mesh/prim shaders with dedup'd verts; PS does `QuadActiveAnyTrue(EarlyDepthTest)`
  then the same 64-bit write (NaniteRasterizer.usf:3358-3364, 3463-3467). Ours mirrors this with
  the prevE guard in the HW fragment (NaniteRaster.ts:1177-1183).

### 2.6 The 32-bit fallback, exactly
NaniteWritePixel.ush:20-35: `DEPTH_ONLY` ⇒ 32-bit `InterlockedMax(OutDepthBuffer, asuint(depth))`
— depth-only targets (shadow/VSM) never need a payload. Otherwise
`COMPILER_SUPPORTS_UINT64_IMAGE_ATOMICS` ⇒ `ImageInterlockedMaxUInt64(vis, (PixelValue,DepthInt))`.
Otherwise **`#error UNKNOWN_ATOMIC_PLATFORM`** — shipping UE5 has NO 32-bit visibility-buffer
election; 64-bit image atomics are a hardware requirement. (This CORRECTS the premise audit §3.2's
"falls back to depth-only InterlockedMax + separate payload write" — the depth-only path is not a
vis-buffer fallback, it is the shadow path.) Consequence: our `depthKey24<<8|id8` atomicMax + visBV
side-store (NaniteRaster.ts:335-336, 952-975; NaniteVoxelRaster.ts:1196-1215) is not a degraded
copy of UE5 — it is the solution to a constraint UE5 dodged. The known residual (equal-key24 tie ⇒
id8 tiebreak picks a valid neighbour) is already the accepted Class-I exception. Do NOT reopen the
election without P4 evidence (premise audit §7).

### 2.7 UE5's voxel-brick raster (NANITE_VOXELS) vs ours
`ClusterTraceBricks` (NaniteRasterizer.usf:1479-1782), one workgroup per voxel cluster:
- Per brick: decode, project AABB (with a CONSTANT_DIR_RECT sheared-near-face conservative rect,
  1391-1477), clamp rect to 30/128 px (1605) — cousin of our BRICK_MAX_EXT=64
  (NaniteVoxelRaster.ts:113).
- **Per-PIXEL occlusion pre-test against the LIVE vis buffer** at the rect's near z
  (`OcclusionTestPixel` → `EarlyDepthTest`, 1362-1382, applied at 1673-1679) BEFORE any DDA work
  is queued. No pyramid, no snapshot: because every pixel tests the live UAV, vox-behind-vox
  occlusion emerges automatically inside one dispatch, at pixel granularity, without barriers.
- **Wave work redistribution** (BRICK_TRACE_WORK_REDISTRIBUTION, 1610-1718): surviving pixels
  from ALL bricks in the wave are packed via `WavePrefixSum`/ballot into a 64-slot ring; batches
  of 32 are processed cooperatively, each lane pulling (sourceLane, pixel) with `WaveReadLaneAt`
  (1281-1360). This is cross-brick balancing; our Phase A/B (NaniteVoxelRaster.ts:488-1419)
  balances within a brick by striding its footprint across 128 lanes — same goal, coarser grain,
  no subgroup ops needed.
- **Per-pixel DDA depth always**: the elected depth is the DDA hit time (mid of entry/next,
  1263-1269; DDA in Voxel/Voxel.ush:53-132, ≤3·3+1 steps at 1248). Ours: flat path elects a
  brick-CONSTANT front-slab key (NaniteVoxelRaster.ts:886-894); only `?voxcell` big bricks get
  ray-exact depth (1383-1404).
- Their bricks sit in the same DAG/LOD stream (`Cluster.LODError` = voxel size, 1487): no
  distance-tier switch — consistent with the premise audit §3.4 (tier seams are OUR choice).

### 2.8 Front-to-back WITHOUT barriers (the depth-bucket consumption)
NaniteRasterBinning.usf: each raster bin's cluster array is allocated **depth-bucket-major** —
`RasterGroupDepthBlock` prefix-sums the 64 per-bucket counts into slot bases (240-261), and
clusters are scattered into their bucket's contiguous slice (`AllocateRasterGroupCluster`,
354-390, `FlatDepthBucket` from the cluster's stashed DepthBucket, 461-468). The rasterizer then
runs ONE indirect dispatch over the bin; near clusters occupy low group indices, and because GPUs
launch groups roughly in index order, the early-Z tests (§2.5, §2.7) statistically see near depth
first. **No inter-bucket barrier anywhere.** Correctness never depends on the order (InterlockedMax
is order-free) — order is purely a throughput heuristic.

**Contrast with our refuted ?voxf2b**: we built the identical counting sort
(kVoxRange/kVoxCount/kVoxPrefix/kVoxScatterFan, NaniteCull.ts:545-710) but then consumed it as
K=16 SEPARATE dispatches with in-pass UAV barriers between buckets (NaniteVoxelRaster.ts:1437-1452,
1481-1486) — K serialized drains, measured +5/+6/+16 ms. The barrier chain, not the ordering, is
what lost. The UE5-shaped consumption — sorted queue + ONE dispatch — was never measured, and
kVoxPrefix already publishes the whole-list dispatch args for it (NaniteCull.ts:663).

### 2.9 Depth export / fast clear
`DepthExport` converts vis-buffer depth to SceneDepth/HTILE with wave min/max per 8×8 tile
(NaniteDepthExport.usf:70-224) — console-specific HTILE plumbing, not portable and not needed
(our resolve writes depthNode directly, NaniteRaster.ts:1317-1329). NaniteFastClear.usf is only
a cmask visualization. No levers.

## 3. Waste map (ours, through the UE5 lens)

| # | Waste | Mechanism | Scales with | Est. cost (measured anchor) |
|---|---|---|---|---|
| W1 | Stale-HZB single-phase occlusion under motion | prev-VP×prev-HZB test at emit only (NaniteCull.ts:917-925; NaniteHzb.ts:44-51); no post-pass rescue | camera velocity × depth complexity | ~9–11 ms moving-camera cut inflation on base (2026-06-26 handoff); aerial arrival ramp 6.5–11.5 ms over-cull transient; foliage motion inflation unmeasured. Static iso: ~0 — this is why the canonical poses under-measure it (premise audit §3.10) |
| W2 | Loose occlusion bounds | sphere + 2×2 window at diameter-fits-one-texel level (NaniteHzb.ts:175-198); same idiom in voxOccPyr block (NaniteVoxelRaster.ts:636-659) and ?voxbocc brick (846-879) tests | occludee count (11.6k clusters, 9188 vox clusters oblique) | voxbocc with the LOOSE bound already took eye 36.1→18.9 but oblique only 43.2→37.2 — the oblique residual is partly bound-looseness (canopy gaps drive min-pool toward keep; a finer-mip window shrinks the window's reach into gaps) |
| W3 | Vox scatter runs unordered | qVoxRaster appended in BFS emit order (NaniteCull.ts:526-531); prevE guards + per-brick culls see empty/far pixels first at oblique | vox coverage × overdraw depth | old F2B data: −38..−45% brick writes when ordered; the win the barriers ate. Bounded by oblique vox share of 21.8 ms foliage pool |
| W4 | Zero-coverage clusters emitted | no `bOverlapsPixelCenter` equivalent at emit; sub-pixel clusters between pixel centers still enter qRaster → 128-lane workgroup each | far-field cluster count | UE5 comment: ~5% fewer clusters (NaniteHZBCull.ush:83). At visTris 4.4M eye / 1.27M oblique, order ~0.5–1 ms |
| W5 | Span solve on 1–2 px tris | 3 f32 divides + 6 floor/ceil/selects per bbox ROW (NaniteRaster.ts:848-868) vs UE5's ≤4px rect walk (NaniteRasterizer.ush:291-300) | rows × tri count (97% sub-pixel) | fraction of the per-pixel-loop 90%; est 0.3–1 ms eye |
| W6 | Vertex re-fetch ×3/tri | no per-cluster vert LDS staging (UE5: NaniteRasterizer.usf:782-821) | tri count | bounded-marginal (LEVER #1: setup < vsync floor even at 3.3M clusters); do not build |
| W7 | Flat-path brick depth is brick-constant | front-slab key for all non-?voxcell bricks (NaniteVoxelRaster.ts:886-894) vs UE5 per-pixel DDA depth | — | not a ms cost — a CORRECTNESS looseness: intra-brick occlusion + HZB slightly wrong-near; feeds W2's occluder quality |

## 4. Levers (ranked)

### L1 — Two-pass occlusion: record HZB rejects, rescue vs fresh HZB (UE5 main/post)
**Mechanism**: in kTraverse's emit branch, when `sphereOccluded` fires, append (instId, ci) to a
reject queue instead of dropping (cap exists: REJ_CLUST_CAP=1M, NaniteCull.ts:99; slot-3 counter +
rejClustV noted free at NaniteCull.ts:742-744). Frame order becomes: cull(main) → world1 →
hzb.build (already runs post-raster, NaniteHzb.ts:44-45) → kRetest (one thread/reject, CURRENT
vp/camPos vs the fresh pyramid — `sphereOccluded` is already parameterized for exactly this,
NaniteHzb.ts:47-51) → append survivors to qRaster → world1 phase-2 over the appended range (the
rasterDispatch2Attr / kRasterArgs2 / qRaster[0].y phase-2-base machinery is ALREADY BUILT,
NaniteCull.ts:482-489, NaniteRaster.ts:203). Then main-pass phase-1 can stay (or become more)
aggressive with zero popping. Voxel rescues re-run runVoxFanout over the appended range.
**Files**: NaniteCull.ts (reject append + kRetest), NaniteFrame.ts (pass order), NaniteRaster.ts
(phase-2 world1 dispatch), NaniteShadow untouched.
**Expected ms**: eye/oblique/aerial ISOLATED ≈ 0 (static poses have ~converged HZB; small + from
the retest dispatch). LIVE: attacks the measured ~9–11 ms moving-base inflation + the unmeasured
foliage equivalent; the aerial 2-3-frame arrival ramp (6.5–11.5 ms over-cull) should collapse.
This is the lever for "underlying live p95 ≤ 15.5" that static-pose probes cannot see.
**Quality**: IDENTICAL-or-better (it only ADDS geometry a conservative test wrongly dropped;
fixes 1-frame disocclusion artifacts). Shotdiff at static poses must be exactly 0.
**Gate**: engagement counter (rescued clusters/frame > 0 while moving); live slot histogram
(fraction ≤2 slots) before/after; iso pose-arrival ramp gone. **Effort**: L.
**UE5**: NaniteInstanceCulling.usf:397-402, NaniteClusterCulling.usf:561-606/941-962,
NaniteCullingCommon.ush:646-662.

### L2 — Finer-mip 4×4 occlusion windows (HZB + voxOccPyr block/brick tests)
**Mechanism**: replace "2×2 at level ceil(log2(footprint))" with UE5's "4×4 at MipLevelForRect
(footprint fits 4 texels)" — 2 mips finer for the same coverage, min/max over 16 texels instead
of 4. Apply in three places sharing the idiom: NaniteHzb.sphereOccluded (NaniteHzb.ts:175-198),
the per-block cull (NaniteVoxelRaster.ts:636-659), the ?voxbocc per-brick cull (846-879).
Optionally rect-not-sphere for the cluster test (project the cluster AABB's 8 corners — the
cull already has instance transform + local sphere; a local AABB is in cluster words). 16 loads
vs 4 per test; tests run per cluster / per brick-lane, well off the per-pixel hot path.
**Expected ms**: oblique −1.5..−4 (9188 vox clusters × bricks; voxbocc's oblique residual is the
target — canopy-gap texels currently drag a WIDE coarse window's min to "keep"); eye −0.5;
aerial −0.3. **Quality**: IDENTICAL (finer window over the same covered footprint is still
conservative: min-pool keeps any see-through texel; max-pool HZB still bounds the true max).
Shotdiff 0 required. **Gate**: brick-cull skip counter delta ≥ +20% at oblique AND gpuWall
−≥2 ms interleaved A/B, else discard. **Effort**: S–M.
**UE5**: NaniteHZBCull.ush:40-69, 135-172; NaniteCullingCommon.ush:599-602.

### L3 — Depth-sorted vox queue consumed by ONE dispatch (UE5 counting sort, no barriers)
**Mechanism**: run the existing F2B fanout (kVoxRange→kVoxCount→kVoxPrefix→kVoxScatterFan,
NaniteCull.ts:588-710 — qVoxRaster becomes near→far contiguous; kVoxPrefix already publishes
whole-list args at NaniteCull.ts:663) but dispatch the SINGLE whole-list kVoxScatter over it
(NaniteVoxelRaster.ts:1424-1429) instead of K barrier-separated bucket kernels. Near bricks land
in low workgroup indices → launch roughly in order → prevE guards (1196-1215), the per-block cull
and ?voxbocc see near canopy depth earlier. Zero barriers; ordering is a heuristic, correctness
is the order-free atomicMax (byte-identical already verified in the F2B work).
**Files**: NaniteCull.ts (flag: fanout-sorted but publish/use full-range args),
NaniteVoxelRaster.ts (dispatch path), ~30 lines.
**Expected ms**: oblique −1..−3 (old F2B measured −38..−45% brick writes before barriers ate it);
eye −0..−0.5 (vox pool at eye is only ~2.1 ms post-voxbocc); aerial ~0 (659 clusters).
**Quality**: IDENTICAL (reorder of an order-free election; shot-gated before).
**Gate**: interleaved A/B vs voxbocc baseline; accept only if oblique med −≥2 ms; engagement:
?voxwrites counter must drop ≥25% (proves the ordering engaged — tripwire 2, premise audit §6).
**Effort**: S. **UE5**: NaniteClusterCulling.usf:314-327, NaniteRasterBinning.usf:240-261,354-390.

### L4 — Zero-coverage cluster cull at emit (bOverlapsPixelCenter)
**Mechanism**: at the cut's emit (NaniteCull.ts:861-934), compute the projected rect in pixels and
drop clusters whose rect covers no pixel CENTER (UE5: NaniteHZBCull.ush:80-91 — "~5% fewer
clusters"). We already do this per-TRIANGLE (`coversSample`, NaniteRaster.ts:753-761); lifting it
to cluster level kills the whole 128-lane workgroup + qRaster/fanout traffic for between-center
slivers. Needs the cluster AABB (or sphere-rect approx — slightly conservative, still valid).
**Expected ms**: eye −0.3..−1 (visTris 4.4M, many sub-pixel), oblique −0.3..−0.8, aerial small.
**Quality**: IDENTICAL only if implemented exactly (a covered-center cluster must never drop) —
sphere-rect must round OUTWARD; shotdiff 0 gate. **Effort**: M (careful; wrongness = holes).

### L5 — Adaptive rect walk for narrow triangles (skip the span solve)
**Mechanism**: mirror `RasterizeTri_Adaptive` (NaniteRasterizer.ush:291-300): if bbox width ≤4 px,
skip the per-row x-span solve (3 divides + selects, NaniteRaster.ts:848-868) and walk the row
directly — the exact per-pixel cw≥0 test already decides coverage, so pixels are identical.
**Expected ms**: eye −0.3..−1, oblique −0.2..−0.6 (base pool; 97% sub-pixel tris).
**Quality**: IDENTICAL by construction (the span is only a fast-skip superset).
**Gate**: interleaved A/B; must clear 2 ms or record as neutral. **Effort**: S.

### Non-levers (explicit, so nobody builds them)
- **Persistent-threads cull**: our cull ≈0.2 ms; UE5's motivation (traversal latency) doesn't bind.
- **Vertex LDS staging (W6)**: bounded-marginal per LEVER #1 stage split; revisit only if a future
  change makes setup dominant.
- **64-bit-style election rework**: no UE5 32-bit recipe exists to copy (§2.6); election already
  refuted as bottleneck. Only P4 (voxrdbg=2 oblique delta >3 ms) reopens.
- **Per-pixel-vs-live occlusion instead of voxOccPyr**: UE5's per-pixel EarlyDepthTest vs the live
  UAV is what our prevE guard already is; the pyramid ADDS whole-brick/block skips they lack.
  Keep both; do not delete the pyramid chasing UE5 purity.

## 5. Refuted / rejected paths for this stage
- **K barrier-separated near→far bucket dispatches** (?voxf2b=1 / ?voxwaves) — measured +5/+6/+16
  and +0/−0.6/−4.7 vs f2b; the null does NOT cover the sorted-single-dispatch variant (L3). The
  wave-rebuild variant's block-level null also does NOT refute brick-granularity ordering effects
  (tripwire 2 precedent: voxwaves null at block grain while voxbocc at brick grain was the whale).
- **Atomic contention / election rearchitecture** — refuted (capture-as-ground-truth memory;
  coverage-bound), and UE5 offers no portable alternative (§2.6).
- **Tiled/binned raster** — REFUTED+REMOVED for triangles (+11.7/+18.1 ms, SPEC D-N46) and the
  voxel bin equally (NaniteVoxelRaster header, lines 2-12). UE5's binning is by MATERIAL for
  shading, not a screen-space tile raster; do not re-read §2.8 as license to re-tile.
- **Copying UE5's SW 63-px triangle bound** — our 16-px bound exists for i32 edge-term safety
  (NaniteRaster.ts:719-731); raising it moves work SW without evidence SW is cheaper there.

## 6. Open questions + serial GPU probes wanted

1. **P-L3 (?voxsort=1, new flag)**: F2B counting-sort fanout + single whole-list scatter (no
   bucket kernels). Interleaved A/B vs current default at the 3 poses. Decision: oblique med
   −≥2 ms AND ?voxwrites −≥25% ⇒ ship Class I with shots; writes drop but gpuWall flat ⇒ record
   "ordering engages but M1 scheduler doesn't reward it" and close.
2. **P-L2 (?occwin=4, new flag)**: 4×4-one-mip-finer windows in voxOccPyr block+brick tests (and
   optionally NaniteHzb.sphereOccluded). Decision: brick-skip counter +≥20% at oblique AND gpuWall
   −≥2 ms ⇒ ship; counter moves but wall doesn't ⇒ the residual oblique cost is not cull-bound —
   escalate to the foliage-structure reader (mid-ring merge).
3. **P-L1a (counter-only build)**: count per-frame emits that would be rescued/killed differently
   by a fresh-HZB retest (run kRetest as counter-only after hzb.build, no raster change), logged
   during a LIVE moving run. Decision: rescue+kill counts quantify the two-pass win's upper bound
   before committing effort L; <3% of emits ⇒ deprioritize L1 for iso, keep for correctness only.
4. **P-L5 (?rectnarrow=1)**: skip span solve when bbox width ≤4 px. Decision: ≥2 ms eye ⇒ ship;
   else record neutral (cheap probe, S effort).
5. **P1 rebaseline (premise audit)** remains the highest-value serial probe (post-voxbocc live
   slot histogram) — L1's value is priced entirely in it; request it runs FIRST.
