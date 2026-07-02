# 05 — Voxel raster runtime: the scatter/election engine

Scope: `src/nanite/NaniteVoxelRaster.ts` (whole file — Phase A brick record + voxbocc cull, Phase B
pixel spread, F2B scaffolding, wave path, voxOccPyr kernels, dispatch structure) + `src/nanite/Tsl.ts`
dispatch helpers. All numbers from the 2026-07-02 measured table or scratchpad `fresh-*.json`.
Targets inherited from the premise audit (90-premise-audit.md §2.3): oblique ≤21, eye ≤18, aerial ≤16.

## 1. TL;DR

- The engine is one workgroup per voxel BLOCK (128 lanes), Phase A = 1 lane/brick record + cull,
  Phase B = cooperative footprint spread with a UE5-style read-before-atomic guard ALREADY in place.
- Biggest open lever: **brick-granularity vox-behind-vox** (2-slab scatter + one voxOccPyr rebuild) —
  the table's "waves ≈ zero" null was measured at BLOCK granularity pre-voxbocc; est. −2 to −5 oblique.
- Second: **exact-rect mip pick for the brick cull window** (UE5 NaniteHZBCull idiom) — the current
  centre±1 window pools up to 4× the footprint area ⇒ missed culls; est. −1 to −3 oblique, free loads.
- Housekeeping: single-submit `dispatchVoxel` + dead `kClearBins` removal (−0.3 to −0.8 everywhere);
  shared clip-basis Phase A projection (−1 to −2.5 oblique); survivor compaction is the L-effort rock.
- The read-before-atomic lever from the task brief is ALREADY IMPLEMENTED (ts:1202-1205); atomic
  traffic is not the pool. The pool = per-brick fixed work + losing per-pixel visits; probes P2/P3 split it.

## 2. How it works today (mechanism walk)

### 2.1 Dispatch shape
`buildNaniteVoxelRaster` (NaniteVoxelRaster.ts:175) builds ONE scatter kernel via the
`makeVoxScatter` factory (ts:488). Grid: one WORKGROUP per qVoxRaster item (= one ≤128-brick voxel
BLOCK/cluster), `WG_RASTER = MAX_BRICKS_PER_CLUSTER = 128` lanes (ts:480, VoxelBrick.ts:81), baked
`.compute(DISPATCH_ROW*WG_RASTER,[WG_RASTER])` with the real workgroup count from the cull's 2D-split
indirect args (ts:1419, ts:1490). The queue is fed by NaniteCull's kVoxFanout (NaniteCull.ts:512-529);
`nanite.voxClusters` = that queue count (NaniteFrame.ts:524). Measured queue sizes (fresh-voxbocc.json):
eye 5935, oblique 9185, aerial 661 blocks.

`dispatchVoxel` (ts:1454-1492), default path (?voxf2b=0), issues **three submits** per frame:
1. `dispatchBatch(voxPyrKernels)` — the voxOccPyr min-pool chain (12 dependent dispatches, one pass).
2. `dispatch(kClearBins)` — zeroes the 1-word debug write counter (ts:309-315).
3. `dispatchIndirect(kVoxScatter, voxRasterDispatchAttr)` — the scatter itself.

### 2.2 voxOccPyr — the occluder structure and its polarity idiom
Built EVERY frame from THIS-frame `visPayloadV` (mesh SW+HW winners; vox runs after hwRender,
NaniteRaster.ts:1423). MIN-pooled 2×2 reduction chain (ts:429-474): election keys pack
`depthKey24<<8|id8` with NEARER=LARGER (NaniteRaster.ts:335-336), so the conservative occluder over a
window is the MIN = farthest/most-see-through pixel; any EMPTY texel (key 0) drives the min to 0 ⇒
KEEP (no holes, ts:371-398). At 2268×1473 the pyramid is half-res based: 1134×737 → … → 1×1 = 12
levels, ~1.12M texels (~4.5MB); L0 alone reads 4 taps/texel of visPayloadV (~3.34M loads). Real work
is small; the chain is 12 serialized (UAV-auto-synced) dispatches inside one pass (Tsl.ts:241-243).

### 2.3 Phase A (1 lane = 1 brick)
Per block: qVoxRaster item + instance A/B + cluster words 0-7 + block sphere transform (ts:551-582).
Block-level occlusion test on lane 0 only (?voxoccl, default on): project the block sphere, pick the
pyramid level from `rPx = ceil(log2(max(1, projected radius)))`, 2×2 window min, cull iff
`bNearKey ≤ occK` (ts:614-665), broadcast via workgroup flag + barrier.

Per live brick lane (`brickLocal < brickCount`, ts:690):
- 5 brick-word loads (POS_XYZ/HALF; +OCC_LO/HI under voxcell/gate; BRICK_WORDS=9, 36B — VoxelBrick.ts:61-81);
- project the brick AABB's **8 corners**, each = `instTransformPoint` + full `cam.vp` mat4·vec4 +
  perspective divide + bbox min/max (ts:737-772), with the near-plane straddle/NDC-explode
  classification (ts:727-736, NDC_EXPLODE=3);
- non-straddler → raw clamped bbox + span cap 2·BRICK_MAX_EXT=128px (ts:803-812); straddler → fixed
  BRICK_MAX_EXT centre box, i.e. up to 129×129 ≈ 16.6K px regardless of true brick size (ts:813-836);
- **?voxbocc per-BRICK cull (default ON since 2026-07-02)**: the brick's own clamped bbox + front-slab
  key vs the SAME pyramid, level from `rPx = max(bbW,bbH)/2`, centre-anchored 2×2 window, keep-on-tie
  `|0xff` (ts:838-884). Culled ⇒ record never stored ⇒ Phase B zero-trip. Straddlers exempt (ts:844).
- survivors store bbox+key records into workgroup arrays (ts:895-902), voxcell local-AABB/occupancy
  records (ts:908-920), and — for coarse (dagLevel>0), area≥16, popcount≤48 bricks — build the 4×4
  screen occupancy mask by projecting up to 64 occupied cells × 8 corners = **≤512 extra projections
  per armed brick** (ts:921-1060).

### 2.4 Phase B (all lanes, cooperative per-brick)
Barrier (ts:1064), then loop `b` over `nBricks ≤ 128` (ts:1140): each iteration reads the 5-record +
gate/cell words from workgroup memory and computes two float reciprocals (?voxrecip, ts:1161-1162),
then strides the brick's footprint `[0, area)` across 128 lanes (ts:1183). Per pixel visit:
- addressing ≈ 8-12 ALU (reciprocal ly/lx, ts:1187-1193);
- **read-before-atomic guard**: relaxed `aLoadU(visPayloadV)` + `cand > prevE` gates the atomicMax;
  winner plain-stores `visBV` (ts:1196-1215) — byte-for-byte the world1 election idiom
  (NaniteRaster.ts:962-967) and functionally UE5's `EarlyDepthTest()`
  (Nanite-UE5-shaders/NaniteWritePixel.ush:48-66: pre-read `OutDepthBuffer[Position] < DepthInt`
  before `InterlockedMax`). **A losing visit costs 1 global load + compare, no atomic.**
- occupancy-gated carved bricks add a 4×4 bucket bit test (~8 ALU, ts:1240-1252);
- ?voxcell (default on) for non-straddler bricks with area>64: after the same prevE early-out
  (ts:1266-1267), ray-vs-local-AABB slab test + ≤6-step DDA through the 4³ mask + exact per-pixel
  depth key ≈ 60-150 ALU per surviving pixel (ts:1269-1405).

### 2.5 F2B / waves scaffolding
?voxf2b=1 builds K (=?voxf2bk, default 16, clamped [1,32] — NaniteCull.ts:327-328) per-bucket kernel
instances closing over `voxBucketRange[b]` (ts:1437-1452), dispatched near→far in ONE submit via
`dispatchBatchMixed` — in-pass UAV auto-sync serializes bucket b's visPayloadV writes before b+1
(Tsl.ts:296-306). ?voxwaves=N chunks the K buckets into N waves with a full voxOccPyr rebuild between
waves (ts:1466-1480). The `setIndirectDispatch` tagging trick that lets batched kernels keep tight
indirect grids is Tsl.ts:274-289.

## 3. Waste map (cost model per unit, anchored to measured deltas)

Unit costs (from the mechanism walk):

| unit | cost |
|---|---|
| block (workgroup) fixed | ~12 global loads + ~80 ALU uniform + block-occl 4 pyramid loads + 2 barriers; voxcell ray basis ~150 ALU (ts:1098-1136) |
| brick, Phase A | 5-7 loads + 8 corners × ~40 ALU ≈ 320 ALU + voxbocc 4 pyramid loads; armed mask build ≤18K ALU (≤512 projections) |
| brick, Phase B setup | (5-12 wg reads + ~30 ALU incl. 2 recips) × all lanes, per nBricks iteration |
| pixel visit, losing | ~10 ALU + 1 relaxed global load |
| pixel visit, winning | + atomicMax + atomicStore (contention refuted — spread addresses) |
| pixel visit, voxcell | + 60-150 ALU after the prevE early-out |

Measured anchors:

- **W1 — the buried field pays ~everything up to the cull point.** voxbocc: eye 36.0→18.8
  (fresh-bead-v2-base vs fresh-voxbocc), i.e. **−17.2ms** was Phase A full projection + mask builds +
  Phase B losing visits for a ~fully-mesh-buried field of 5935 blocks. Post-cull eye foliage residual
  ≈ 2.1ms (90-premise-audit.md §2.4). Note the cull fires AFTER the 8-corner projection (it needs the
  bbox), so even culled bricks keep paying ~320 ALU + loads each — see L4.
- **W2 — brick-level vox-behind-vox is unculled.** The pyramid holds MESH content only (built once,
  before the scatter). Upstream, kTraverse HZB-culls whole BLOCKS against the prev-frame HZB, which
  does include last frame's vox depth (NaniteCull.ts:919-921) — that is why BLOCK-level waves measured
  ~zero at eye/oblique (fresh-voxwaves4 41.1/48.8 vs fresh-voxf2b-ctl 40.9/49.2). But those runs
  predate the voxbocc default (extra: no voxbocc; eye 41 ≈ pre-voxbocc 36+chain, not 19+chain), so
  **brick-granularity vox-behind-vox has NEVER been measured** — exactly premise-audit trap #2. The
  oblique ring (9185 blocks, rows of crowns 45-140m deep) is where the buried-brick fraction lives;
  the ring holds ~10ms at oblique (aggdist=60 diagnostic lesson, quality-rejected but the cost stands).
- **W3 — the cull window over-pools.** Both cull tests pick `level = ceil(log2(rPx))` from the MAX
  half-extent, then take a centre-anchored 2×2 window (ts:851-879): window edge = 2^(ℓ+2) full-res px
  ≥ 2× the footprint diameter, per axis ⇒ up to **4× the needed area pooled** ⇒ the min absorbs
  unrelated far/empty texels ⇒ KEEP verdicts that a tight window would cull. Elongated bboxes
  (bbW≫bbH) are worst (level from max extent). UE5 picks the level per-axis from the EXACT inclusive
  rect — `MipLevelXY = firstbithigh(RectPixels.zw - RectPixels.xy)` — and samples a rect-aligned 4×4
  (NaniteHZBCull.ush:43-57, 144-154).
- **W4 — dispatch-chain serialization floor (F2B).** +4.9/+6.1/+16.0 (fresh-voxf2b-ctl vs
  fresh-bead-v2-base). Cost model: K in-pass UAV barriers, each a full drain of the previous bucket's
  long-tail workgroups. Aerial is worst because 659 blocks/16 ≈ 41 workgroups per bucket (GPU has 32
  cores — near-zero occupancy) and FarTiles blocks run LONG (big footprints) ⇒ ~1ms per bucket
  boundary. Waves at aerial recovered −4.8 (28.0 vs 32.8) — proof vox-behind-vox gain exists where
  the granularity matches.
- **W5 — dead work in the default dispatch path.** `kClearBins` clears a counter only read under
  ?voxwrites (off; ts:241) yet is dispatched as its own submit every frame (ts:1489); dispatchVoxel is
  3 submits where 1 suffices (§2.1). cpu.submit med ~1.3ms in the voxbocc run.
- **W6 — 128-lane workgroup quantization.** Blocks carry brickCount ≤128 but always launch 128 lanes;
  Phase B iterates ALL nBricks serially per workgroup with 128-lane strides over areas that are often
  a few px ⇒ most lanes idle most iterations. The aggdist=60 lesson (~10ms over ~8.5k ring blocks ≈
  ~1.2µs per block) says cost scales with BLOCK COUNT, not final pixels — consistent with fixed
  per-block/per-brick machinery + scheduling, not visit ALU. Needs the P3 counters to split.
- **W7 — straddler fallback is size-blind.** Any near-plane-straddling brick paints a fixed
  129×129-px centre box (ts:813-836). Zero at canonical poses (vox starts at 45m), but a live camera
  clipping a crown can hit it — a live-p95 spike hazard, not a median cost.

## 4. Levers (ranked; all Class I unless stated)

### L1 — Brick-granularity vox-behind-vox: 2-slab scatter + one pyramid rebuild
**Mechanism:** split the scatter into 2 depth slabs (reuse the F2B bucket machinery at K=2, or a
dedicated near/far split), rebuild voxOccPyr once between them; the far slab's ?voxbocc then culls
bricks buried behind near VOX crowns, not just mesh. This is the voxbocc win re-aimed at the ring's
self-occlusion. Existing flags already compose the experiment: `?voxf2b=1&voxf2bk=2&voxwaves=2`
(voxbocc default on) — K=2 means ONE extra barrier + ONE pyramid rebuild (~0.3-0.8ms), not the
16-bucket floor.
**Files:** NaniteVoxelRaster.ts:1454-1492 (productize the 2-slab path outside ?voxf2b), NaniteCull.ts
F2B block (K=2 args).
**Expected:** oblique −2 to −5 (buried ring fraction × the ~10ms ring pool), eye −0 to −1 (already
mesh-culled), aerial −2 to −4 (waves already showed −4.8 there at heavy chain cost). Live: helps p95
(moving camera keeps deep rows).
**Quality:** IDENTICAL (conservative cull, same polarity idiom; atomicMax is order-independent —
image-identity of ordered vs unordered already verified, NaniteCull.ts:304-305). Shotdiff gate anyway.
**Gate:** probe P1; engagement counter = far-slab culled-brick count > 0 (else the null is void).
**Effort:** S for the probe (flags exist), M to productize.
**UE5:** two-pass occlusion (CULLING_PASS_OCCLUSION_MAIN/POST, NaniteCulling.ush:10-11) — same
principle (render near/confident set, re-test the rest against the fresh occluder), applied intra-frame.

### L2 — Exact-rect mip pick + rect-aligned window for both cull tests
**Mechanism:** replace `rPx=max(extent)/2; level=ceil(log2(rPx)); centre±1 window` (ts:847-874,
636-654) with the UE5 idiom: level per rect from `firstbithigh((endX>>1)-(startX>>1))` (pyramid is
half-res) so a 2×2 window EXACTLY covers the shifted rect, or one level lower with a 4×4 window (16
loads) for 4× tighter pooling. Strictly more culls, still pools every texel under the footprint ⇒
conservativeness unchanged.
**Files:** NaniteVoxelRaster.ts:614-665 (block test), 838-884 (brick test).
**Expected:** oblique −1 to −3 (missed-cull recovery on the ring; magnitude = f(buried fraction ×
current miss rate) — needs the P3 cull counter), eye −0 to −1 (residual 2.1ms bound), aerial ~0.
**Quality:** IDENTICAL (conservative cull; shotdiff maxDiff=0 at 5 poses).
**Gate:** culled-brick counter delta + gpuWall, same session A/B. **Effort:** S.
**UE5:** NaniteHZBCull.ush:43-57 (firstbithigh level pick), :144-154 (4×4 gather).

### L3 — Single-submit dispatchVoxel + remove the dead kClearBins
**Mechanism:** fold pyramid chain + scatter into ONE `dispatchBatchMixed` submit (tag kVoxScatter via
`setIndirectDispatch`, Tsl.ts:283-289 — the enabler already exists); skip kClearBins entirely when
!voxWrites (its counter is only read under ?voxwrites, ts:241-247). In-pass auto-sync preserves
pyramid→scatter ordering (same idiom as the chain itself, Tsl.ts:241-243).
**Files:** NaniteVoxelRaster.ts:1454-1492.
**Expected:** −0.2 to −0.8 all poses (2 fewer submits + inter-submit bubbles) + cpu.submit down.
Small but free and compounding with L1 (which adds a submit boundary back).
**Quality:** IDENTICAL (no math change). **Gate:** gpuWall + cpuSubmit A/B (probe P4). **Effort:** S.
**Metal note:** no binding-count change — same kernels, same buffers.

### L4 — Shared clip-basis projection in Phase A (corners + occ-mask cells)
**Mechanism:** brick AABBs are axis-aligned in instance-LOCAL space and the instance transform is
linear, so precompute per cluster the clip-space images of the local axes (3 mat4·vec4), then every
corner clip = `C ± h·Bx ± h·By ± h·Bz` (adds only; perspective divide stays per corner). Cuts the
8-corner path from ~320 to ~120 ALU/brick and the ≤512-projection mask build ~3× (cell corners are
linear in cell index: `base + i·Dx + j·Dy + k·Dz`). Same trick the voxcell ray basis already uses for
rays (ts:1078-1136) — extend it to Phase A. This also cheapens the bricks voxbocc culls (W1: they pay
projection BEFORE the cull).
**Files:** NaniteVoxelRaster.ts:717-772, 988-1019. TSL hoist care (build inside the per-brick flow).
**Expected:** scales with Phase A share (unknown until P2): if Phase A ≈ 30-40% of the ring,
oblique −1 to −2.5; eye −0.2 to −0.5; aerial −0.3 to −1 (FarTiles blocks are brick-heavy).
**Quality:** IDENTICAL-with-gate: ulp-level float reassociation can move a floor/ceil by 1px in rare
cases — shotdiff maxDiff=0 expected but must be verified, else classify and fix (widen by the known
ulp bound). **Gate:** P2 first (sizes the pool), then shotdiff + gpuWall. **Effort:** M.
**UE5:** equivalent spirit — Nanite transforms cluster bounds once and derives screen rects from
basis math, not 8 independent full transforms per primitive.

### L5 — Survivor compaction: Phase A → global brick queue → packed indirect Phase B
**Mechanism:** split the kernel. Phase A (one lane/brick, tight indirect over blocks) writes surviving
bricks' packed records (bbox 2 words, cand, voxIdB, occLo/occHi, local AABB 4 words, instId ≈ 11
words) to a global queue + atomic count; Phase B dispatches indirect over the QUEUE — workgroups
packed with live bricks only, no 128-lane quantization, no per-block Phase A/B barrier, block-tail
imbalance gone. Directly attacks W6's ~1.2µs/block fixed cost.
**Buffers (Metal 10-cliff, premise audit §3.9):** Phase A binds qVoxRaster, clusters, instances,
voxelBricks, voxOccPyr, queue(+count folded in word 0), args ≈ 7 — OK. Phase B binds queue,
instances, visPayloadV, visBV ≈ 4 — OK. Queue sizing: cap at QVOX_CAP×avg-bricks budget or 1.5M
records (~66MB at 11 words) — needs a cap + overflow-safe drop-to-legacy.
**Expected:** oblique −2 to −5 IF P3 shows per-brick fixed cost (not pixel visits) dominates the
ring; eye −0 to −1; aerial −0.5 to −1.5.
**Quality:** IDENTICAL (same records, same election; scheduling only).
**Gate:** MUST be preceded by P2+P3 (do not build blind). **Effort:** L.
**Scar note:** this is NOT the deleted workgroup flat-domain scheme (ts:1066-1075 header: deleted for
a duplicate-prefix CORRECTNESS hazard + 99.2% loss, not for measured slowness) — Phase B still strides
whole bricks internally, exactly today's inner loop; only the brick→workgroup assignment changes.
**UE5:** Nanite's whole pipeline is compaction-shaped — rasterizer bins consume compacted cluster
lists via indirect args (NaniteRasterBinning.usf), never lane-masked block loops.

### L6 — Straddler footprint tightening (live-spike guard)
**Mechanism:** the straddler centre box is fixed 129×129 px (ts:828-831); tighten to
`min(BRICK_MAX_EXT, ceil(projected rPx at clamped w)+2)` so a small clipped brick paints its true
scale, not 16.6K px. **Expected:** ~0 at canonical poses (no vox straddlers at ≥45m); removes a live
worst-frame hazard when the camera clips a crown. **Quality:** Q (pixels change only in the
already-approximate straddler fallback; needs stress-pose crops — camera inside crown — for sign-off).
**Effort:** S. **Gate:** forcevox close-up pose gpuWall + crops.

Honest sum for this slice at oblique: −4 to −8ms realistic (optimistic −11). The remaining oblique
gap must come from the tier-seam/FarTiles/mesh-leaf/base/post readers — consistent with the premise
audit's "foliage-only cannot carry −16.2 alone" (§2.4).

## 5. Refuted / rejected paths for this stage (do not retry)

- **K≥8 bucket chains** (?voxf2b default): +4.9/+6.1/+16.0 measured; per-bucket UAV-barrier drain
  floor, same disease as the deleted bin path (file header ts:1-14, NaniteCull.ts:293-306). Slab
  counts must be ≤2-3 (L1), never 16.
- **Block-granularity vox-behind-vox** (waves as-measured): ~zero at eye/oblique
  (fresh-voxwaves4.json) — AND structurally explained: kTraverse already HZB-culls blocks against the
  prev-frame HZB which contains vox depth (NaniteCull.ts:919-921). The null does NOT transfer to
  brick granularity (runs predate voxbocc; premise-audit trap #2).
- **More read-guarding of the election**: `prevE` gate already present (ts:1202-1205), mirrors UE5
  EarlyDepthTest (NaniteWritePixel.ush:48-66); atomic contention refuted (capture-method memory).
  Re-open only via premise-audit probe P4 (?voxrdbg=2 delta >3ms).
- **Per-win global atomicAdd counters** (?voxwrites): single-cache-line serialization, measured as the
  close-up cliff driver (ts:236-241). Any new counter must be workgroup-aggregated (P3 design).
- **voxdither see-through stipple** as default: see-through ⇒ no occlusion ⇒ election explosion
  (+18.7ms regression note, ts:186-193). A/B control only.
- **Workgroup flat-domain prefix rebalancing** of Phase B: deleted for duplicate-prefix correctness
  hazard (ts:1066-1075). Global compaction (L5) is the safe shape.
- **aggdist=60 / voxlodk≥0.7 / dpr<1.5**: red-listed quality rejects (premise audit §4); only their
  cost LESSONS are cited here.

## 6. Open questions + serial GPU probes wanted (decision rules)

- **Q1:** What fraction of the oblique ring's ~10ms is per-brick fixed work vs per-pixel visits?
  (Decides L4/L5 vs L1/L2 priority.)
- **Q2:** What is the brick-level buried fraction at oblique once the pyramid holds near-vox depth?
  (Sizes L1.)
- **Q3:** Does the current window's over-pooling actually miss culls in practice? (Sizes L2 — needs a
  cull counter, not just gpuWall.)

Probes (serial, same thermal era, interleaved A/B):

- **P1 (L1 value):** `?voxf2b=1&voxf2bk=2&voxwaves=2` vs default, 3 poses, interleaved. Decision:
  oblique med −≥2ms AND aerial not worse ⇒ productize the dedicated 2-slab path; between −1 and −2 ⇒
  try slab split at the 60% depth quantile; ≤−1 with a nonzero far-slab cull counter ⇒ buried
  fraction too small, close L1.
- **P2 (Phase split):** `?voxrdbg=2` vs default at oblique + eye, AND `?voxbocc=0&voxrdbg=2` vs
  `?voxbocc=0` at eye. (baseline − rdbg2) = Phase B share. Decision: Phase B ≥60% of the pose's
  foliage pool ⇒ L1/L2 first; Phase A ≥40% ⇒ promote L4 and gate L5 on P3.
- **P3 (counter build, diagnostic branch):** add `?voxstats=1` workgroup-aggregated counters (live
  bricks post-cull, voxbocc-culled bricks, footprint visits, election wins; one atomicAdd per
  WORKGROUP, not per event) and read at 3 poses. Decision: visits <10M at oblique ⇒ per-visit work is
  NOT the pool ⇒ L5 promoted; visits >40M ⇒ overdraw-dominated ⇒ L1/L2 carry; also yields the true
  overdraw factor for every future estimate.
- **P4 (L3 quick win):** implement single-submit + no-dead-clear, A/B gpuWall + cpuSubmit, 3 poses.
  Decision: any pose −≥0.3ms with shotdiff 0 ⇒ ship; else keep (still fewer submits) unless regression.
- **P5 (L2 verification):** after L2 lands behind a flag: A/B gpuWall + culled-brick counter delta +
  shotdiff maxDiff=0 at the 5 gate poses. Decision: counter up AND oblique −≥1ms ⇒ default-on.
