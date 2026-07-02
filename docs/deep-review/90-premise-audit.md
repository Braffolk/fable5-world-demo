> Shadow/GI claims reconciled into 17-shadows-gi.md §Reconciliation (2026-07-02); this doc kept for detail. Corrections: §7-P3's "ProbeGI 128-frame cycle = strongest surviving bimodality suspect" is KILLED (ProbeGI never runs in the forest scene — TerrainScene.ts:113/:120 only — and its per-frame work is constant); §5's "shadows measured ~FREE (nanshadow=0)" is not evidence (flag gated an absent system).

# 90 — Premise audit: the review framing itself, one level up

Scope: audits the FRAMING the 12 specialist readers will inherit — budget math, the assumed-fixed
context, the quality gates, and the known cognitive traps. No GPU probes were run for this doc;
every number is from the 2026-07-02 measured table or a JSON in the session scratchpad
(`/private/tmp/claude-501/-Users-sebastian-IdeaProjects-fable-demo2/cc111c9f-86e4-4c1a-be01-9817f021c312/scratchpad/`).

## 1. TL;DR

- "Locked 60 LIVE" on the 120Hz panel is SLOT math: live rAF deltas are hard-quantized to 8.33ms
  vsync slots (`fresh-final-rested.json`: 600 deltas, mean residual 0.28ms, slots {2:18, 3:398, 4:183}),
  so the table's "live ~0.6–0.7× isolated" is an INTERVAL, not a constant: r ∈ (0.65, 0.86].
- Honest isolated targets: **oblique ≤ 21, eye ≤ 18, aerial ≤ 16** (med gpuWall). The sanctioned
  "24–26ms worst-pose" is the optimistic edge (r=0.65, zero slack) and is only valid if a
  post-voxbocc live probe pins r ≤ 0.65 — that probe is the single highest-value measurement.
- Oblique needs ~−16ms; the pools are foliage ≈ 21.8, post ≈ 5.1, base ≈ 10.3. Foliage-only means a
  73% cut at zero quality loss — the plan must draft levers in all three pools.
- Prime architectural suspect: the hard tier seams (45m mesh→vox, 140m/64m FarTiles) — UE5 has no
  distance-tier switches; mid-ring merged heads sized to a screen-space error bound are
  quality-CONTROLLABLE, unlike the rejected naive `aggdist=60`.
- Tripwires: classify quality (I/Q/R) BEFORE measuring; a null result needs an engagement counter;
  no cross-thermal-era A/B; live wins judged on slot histograms, not p50/p95 deltas.

## 2. Budget math, done honestly

### 2.1 What the live numbers actually are

The live phase of `tools/probe-fresh-stutter.ts` records per-tick rAF deltas on the real loop
(header, tools/probe-fresh-stutter.ts:5–9). On this machine those deltas are vsync-quantized at
120Hz: in `fresh-final-rested.json` all 600 deltas sit within 0.28ms (mean) of a multiple of
8.333ms; the distribution is {1 slot: 1, 2: 18, 3: 398, 4: 183}. Therefore:

- "live p50 25.0" = "the median frame took 3 slots", i.e. underlying cost ∈ (16.7, 25.0].
- "live p95 33.4" = "the 95th-percentile frame took 4 slots", i.e. underlying cost ∈ (25.0, 33.3].
- Only 18/600 frames (3%) currently make the 60fps budget of ≤2 slots.

This REFINES a prior belief: the table's "Live ~ 0.6–0.7x isolated" (also in
docs/perf-runs/2026-07-02-attribution-and-waves.md) treats quantized slot counts as continuous
costs. The honest ratio from the one clean same-run pair (`fresh-final-rested.json`: iso med
31.1 / 38.7 / 14.5, live p95 = 4 slots) is an interval:

    r = underlying_live_p95 / iso_worst_med ∈ (25.0/38.7, 33.3/38.7] = (0.65, 0.86]

Two further unknowns corrupt r: (a) the live path's pose mix is unverified against the canonical
poses (only `endPose` is recorded); (b) the pair is PRE-voxbocc — voxbocc cut cpu.submit 5.3→1.4ms
(`fresh-voxbocc.json` cpuSubmit med ~1.3), which changes CPU/GPU pipelining and hence r.

### 2.2 What "locked 60" requires

Target: p95 rAF ≤ 16.7ms at 2268×1473, 200k trees, moving camera. On a 120Hz panel that means
≥95% of frames complete within 2 slots. There is NO partial credit: a 17ms frame displays as
25.0ms. Sitting exactly on the boundary produces 16.7↔25.0 flapping (visible judder), so the plan
must target **underlying live p95 ≈ 15.5ms** (≥1ms slack under the slot edge).

### 2.3 Per-pose isolated targets (what the master plan must sum to)

iso_worst target = 15.5 / r:

| r assumption | iso worst-pose target |
|---|---|
| 0.86 (pessimistic, upper bound of measured band) | 18.0 |
| 0.75 (mid-band) | 20.7 |
| 0.65 (optimistic, lower bound) | 23.8 |

The user-framed "worst-pose ~24–26ms" equals 16.7/0.65 = 25.7 — it assumes BOTH the most
favorable ratio AND zero slack. It may turn out true, but the plan may not treat it as
sufficient-by-construction. **Plan targets (mid-band, med gpuWall, to be re-pinned by probe P1 in §7):**

- **oblique ≤ 21** (from 37.2 ⇒ −16.2; bare minimum −13.2 to reach 24, valid only if r ≤ 0.65 is proven)
- **eye ≤ 18** (from 18.9 ⇒ −0.9; hold the line while oblique work lands)
- **aerial ≤ 16** (from 16.5; AND resolve its bimodality — see §6.6: its upper mode ~17–23 in
  `fresh-voxbocc.json` would flap slots live even though the median passes)

### 2.4 The pools per pose (post-voxbocc effective baseline)

Derived by cross-run subtraction (±1ms; noleaves and ablate runs don't touch the vox path so
voxbocc doesn't move them): base = noleaves − post; post = baseline − ablate(clouds+ao+bounce+bloom+taa);
foliage = voxbocc-baseline − noleaves. Sources: `fresh-noleaves-now.json` (16.8/15.2/11.1),
`fresh-ablate-post3.json` (29.3/37.9/16.9), `fresh-bead-v2-base.json` (36.0/43.1/16.8),
`fresh-voxbocc.json` (18.9/37.2/16.5).

| pose | total | base (trunks+terrain) | post stack | foliage (mesh leaves + vox + FarTiles) |
|---|---|---|---|---|
| eye | 18.9 | ~10.0 | ~6.8 | ~2.1 |
| oblique | 37.2 | ~10.3 | ~5.1 | ~21.8 |
| aerial | 16.5 | ~11.1 | ~0 | ~5.4 |

Consequence: the −16.2 at oblique cannot realistically come out of foliage alone (21.8 → 5.6 is a
73% cut at zero quality loss). The master plan must treat post (5.1ms at oblique: clouds, GTAO,
bounce, bloom, TAA) and base (10.3ms of trunk/terrain raster) as secondary theaters, each with its
own reader. The pixel-scaling law (oblique ≈ 12 + 8.1·Mpx; 3.34Mpx ⇒ 39.1, matching pre-voxbocc
43.2 within thermal noise) says oblique is slope-dominated: coverage × per-pixel work, consistent
with the foliage attribution.

## 3. The assumed-fixed context, on trial

For each item: FIXED (physics/user decree), CHOICE (revisable), or CHOICE-WITH-COST.

### 3.1 three.js r184 TSL codegen → WGSL — CHOICE
The hot kernels (raster election, vox scatter) are TSL with documented hoist hazards
(e.g. NaniteVoxelRaster.ts:1194 building the emit closure inside the per-pixel flow to keep the
atomicMax candidate from being hoisted). Nothing forces the hottest 2–3 kernels to stay TSL — raw
WGSL compute sharing the same buffers is available. Notably, WebGPU subgroups (shipped in Chrome)
are used NOWHERE in src (grep: zero subgroup ops), while UE5's rasterizer leans on wave ops
(WavePrefixSum, docs/perf-runs/Nanite-UE5-shaders/NaniteRasterizer.usf:1616). Verdict: keep TSL
globally, but a per-kernel escape hatch is a legitimate lever class; effort M–L per kernel.

### 3.2 Single vis-buffer + 32-bit atomicMax election — half FIXED, half CHOICE
No 64-bit atomics is genuinely FIXED on WebGPU/Dawn/Metal. UE5 uses
`ImageInterlockedMaxUInt64` when available and falls back to depth-only `InterlockedMax` +
separate payload write when not (docs/perf-runs/Nanite-UE5-shaders/NaniteWritePixel.ush:28–31).
Our packing depthKey24<<8|id8 with visBV side-store (NaniteRaster.ts:335, 952–953;
NaniteVoxelRaster.ts:348–354) is a CHOICE within the fixed constraint. But: atomic contention was
already refuted as the bottleneck (memory: coverage-bound), so re-architecting the election has
low expected value. Readers should only reopen this if they produce fresh evidence election is
hot (probe P4, §7).

### 3.3 Brick format BRICK_WORDS=9 — CHOICE, low priority
GeometryRegistry.ts:91. 36B/brick fetch. No measurement shows the vox raster is fetch-bound
(leaf DECODE measured ~free; `leafcheap=all` −0.9/−0.8/−0.5). Only revisit on evidence.

### 3.4 Two-tier foliage: 45m handoff + 140m/64m FarTiles — CHOICE, prime suspect
Defaults: ?voxnear=45 (ForestScene.ts:102), ?aggdist=140 with 64m tiles (ForestScene.ts:131,138).
The 45–140m ring renders PER-TREE voxel heads (oblique voxClusters 9185 vs aerial 661,
`fresh-voxbocc.json` counters); the aggdist=60 diagnostic showed the ring holds ~10ms at oblique
(26.1/32.5/16.6 vs pre-voxbocc 36.1/43.2/16.8) — quality-REJECTED, but the LESSON stands. One
level up: the hard tier seams are the non-UE5 part of this architecture. UE5 has one continuous
cluster DAG with no distance-tier switches — granularity degrades continuously, never by fiat at
a boundary. `aggdist=60` was rejected because its merged heads were TOO COARSE ("massive voxels"),
not because merging per se is visible; a mid-ring restructure (multi-tree merged heads from
~60–90m, or finer far tiles, sized so projected brick error stays under the same tau the per-tree
ladder uses) is quality-CONTROLLABLE and attacks the 21.8ms pool structurally. This is the
premise the foliage readers should interrogate hardest.

### 3.5 Voxel ladder calibration (errorK=1, K_FLOOR=3) — CHOICE
VoxelizeCrown.ts:87 (`VOXLOD_CFG = { levels: 7, errorK: 1, ... }`), VoxelizeCrown.ts:1006
(`K_FLOOR = 3`). The ladder only recently became a real continuum (memory: voxel-lod-only-two-levels).
On trial: is tau ("one brick ~ tau px") pinned by a perceptual test or inherited? Quality doctrine
allows moving transitions FARTHER only; any coarsening rung is Class R (§4). voxlodk=0.7/0.85
already user-rejected — red-listed.

### 3.6 dpr 1.5 + TRAA — FIXED by user decree
dpr lowering is DEAD ("unacceptable levels of blurry"); 1080p60 + naive upscale is sanctioned LAST
RESORT only. Note honestly why it keeps tempting: the scaling law gives oblique at dpr 1.0 ≈
12 + 8.1×1.49 ≈ 24ms — dpr alone would hit the optimistic target. Readers must not spend pages
rediscovering this. NOT forbidden: making an internal pass cheaper at bit-identical or
user-approved output (that's Class I/Q on its own merits). TRAA as bimodality suspect is already
refuted (persists with TAA off).

### 3.7 The measurement apparatus — SOUND for ≥2ms effects, NOT below ~1.5ms
`measureFrames` drives frames manually, isolated and GPU-bound; whole-frame gpuWall is ground
truth, per-pass timestamps are not (src/core/Hooks.ts:66–74). Trial findings: (a) capSuspect
rejection discarded ~30 samples per 32 kept in `fresh-voxbocc.json` (`capRejects: 30`) — a
selection filter that can bias medians; (b) 32-frame medians on BIMODAL distributions (§6.6)
carry ±1–2ms; (c) the table is built from cross-run subtraction, valid only within a thermal era
— `fresh-rested-def.json` (hot) reads 42.1/55.8/29.7 vs `fresh-final-rested.json` (rested, same
build) 31.1/38.7/14.5: +35–100% from thermal state alone. Verdict: apparatus stays, but every
claimed lever must clear 2ms or use interleaved same-session A/B (COOLDOWN_S honored,
tools/probe-fresh-stutter.ts:33–35).

### 3.8 The pose canon itself — UNVERIFIED premise
Poses at tools/probe-fresh-stutter.ts:37–41. Oblique [0,40,40] pitch −0.35 is THE gap pose — but
nobody has verified the live path's worst 5% of frames resembles it. If live spikes occur in
eye-in-dense-canopy instead, the plan optimizes the wrong pose. Probe P2 (§7) closes this.

### 3.9 Metal 10-storage-buffer cliff — genuinely FIXED
NaniteRaster.ts:339,363 (scar counters folded into hwQueue to stay at the ceiling),
NaniteCull.ts:739, NaniteRaster.ts:1233. Consequence for readers: any lever that ADDS a storage
buffer to a full stage must fold into an existing buffer (scar-fold precedent) or it silently
kills the pipeline. State this in every lever writeup that touches bindings.

### 3.10 Single-phase occlusion with prev-frame HZB — CHOICE, live-p95-relevant
NaniteHzb.ts:11–13: occlusion tests use PREV-frame VP. UE5 runs two-pass occlusion
(CULLING_PASS_OCCLUSION_MAIN/POST, docs/perf-runs/Nanite-UE5-shaders/NaniteCulling.ush:10–11):
pass 1 vs last frame's HZB, pass 2 re-tests the rejects vs the fresh HZB. Here the stale-HZB
disease is measured: aerial pose-arrival ramp (2–3 frames at 6.5–11.5ms over-cull then settle
~17) and the 2026-06-26 finding that MOVING-camera cut inflation was the real mission gap
(~9–11ms, docs/perf-runs/2026-06-26-base-raster-review-handoff.md). Since the TARGET is a moving
camera, isolated static poses systematically under-measure this. A reader should own
"live-vs-isolated divergence under motion" explicitly.

### 3.11 GTAO normals-from-depth — CHOICE inherited silently
src/render/Gtao.ts:122 (`getNormalFromDepth`) — material normals never feed AO. Affects what the
post reader may assume about AO cost/quality coupling.

## 4. Quality doctrine, operationalized

Classify every lever BEFORE measuring it, in the lever writeup. Classes and gates:

- **Class I — quality-IDENTICAL.** Bit-equal output. Gate: shotdiff maxDiff=0 at the 3 canonical
  poses + 2 stress poses (grazing tree-line silhouette, dense mid-ring), TAA jitter index pinned,
  same seed/time-of-day, `settle()` before capture. Conservative-cull levers (remove provably
  invisible work) belong HERE — if any pixel changes, the conservativeness claim is false;
  reclassify. Sole tolerated exception: election tie-order nondeterminism (equal depthKey24,
  different id8) — must be argued from the packing (NaniteRaster.ts:952–953) and bounded <0.01%
  px, else fail. Implementing agent may self-certify WITH archived shots. Precedent: voxbocc
  shipped shot-gated IDENTICAL.
- **Class Q — quality-IMPROVING.** Pixels change, claimed better (finer LOD for longer, rounder
  normals). Gate: side-by-side crops + explicit USER sign-off before default-on. An improvement is
  still a change; it does not skip review.
- **Class R — QUALITY-RISK.** Any mechanism that reduces information: coarser/earlier LOD, lower
  internal resolution, temporal reuse, earlier tier handoff. Gate: shotdiff + explicit user
  sign-off; the implementing agent may NOT self-certify "you can't see it" — aggdist=60 and
  voxlodk=0.7/0.85 all looked fine to their authors and were user-rejected.
- **Red list (permanently rejected as perf levers; do not re-propose):** aggdist=60,
  voxlodk ≥ 0.7, dpr < 1.5. (1080p60 + naive upscale exists only as the user's sanctioned last
  resort, not a reader recommendation.)

Rule of order: a lever measured first and classified after is how quality trades get smuggled.
Every result row in every reader doc carries its class and its shots.

## 5. Refuted / rejected framings (do not retry in this review)

- "Live = 0.65× isolated" as a constant — refined to the interval (0.65, 0.86] (§2.1; contradicts
  the flat ratio in the 2026-07-02 table, cited there and here).
- Atomic-contention-bound raster loop — refuted (capture-as-ground-truth memory); coverage-bound.
- Shade-binning as the frame model — half-wrong; resolve lighting+shadows measured ~FREE
  (nandbg=flat, nanshadow=0 ≈ baseline; docs/perf-runs/2026-07-02-attribution-and-waves.md).
- TRAA jitter as the oblique bimodality cause — refuted (persists with TAA off).
- Per-pixel-loop and cull as base bottleneck — refuted 2026-06-26 (triangle-emit-bound).
- aggdist=60 / voxlodk≥0.7 as levers — quality-rejected; keep only their diagnostic lessons.

## 6. Cognitive traps from this project's history — tripwires for all 12 readers

1. **Premise rot (wrong metric).** The front-to-back "refutation" was NDC-z bucketing compressing
   the far field. Tripwire: before recording any refutation, restate the metric and its space/units,
   and confirm with an independent counter.
2. **Non-engaging mechanism read as refutation.** The voxel DAG "coarsening doesn't pay" era was
   measured on a system that never coarsened; F2B pre-seed never actually seeded. LIVE example in
   today's table: voxwaves (block-level vox-behind-vox) ≈ zero gain, while per-BRICK cull vs the
   mesh pyramid took eye 36.1→18.9 — the wave null was measured at the WRONG granularity and does
   NOT refute brick-granularity vox-behind-vox. Tripwire: a null result may be recorded only with
   an engagement counter delta (e.g. rejected-unit count > 0) proving the mechanism fired.
3. **Thermal ordering bias.** Same build: oblique 55.8 hot vs 38.7 rested (§3.7). Tripwire: no
   cross-era comparisons; interleave A/B in one session; re-run suspicious wins in reversed order.
4. **Quality trades smuggled as wins.** aggdist=60 "won" ~10ms and was dogshit. Tripwire: §4
   classification-before-measurement, shots attached to every result row.
5. **Slot-quantization illusions (new, this doc).** Live rAF stats are 8.33ms-quantized: a real
   3ms live win can show ZERO p50 change (stays in slot) or a fake 8.33ms jump (crosses one).
   Tripwire: judge live effects by the slot histogram (fraction of frames ≤2 / ≤3 slots), never by
   raw live p50/p95 deltas; judge isolated effects by gpuWall.
6. **Bimodal medians.** `fresh-voxbocc.json` aerial alternates ~11–13 vs ~17–23 within one run;
   oblique historically alternates runs-of-4 (33–38 vs 42–47). A 32-frame median can sit on either
   mode. Tripwire: report the mode structure (or p25/p75) with every oblique/aerial number;
   believe no delta <3ms at those poses without two agreeing runs.

## 7. Open questions + serial GPU probes requested (with decision rules)

- **P1 (highest value): post-voxbocc live re-baseline.** `probe-fresh-stutter` CONFIG=default (voxbocc
  now default), full live phase. Decision: recompute the slot histogram and r-band against iso
  37.2; if r ≤ 0.65 is pinned, relax oblique target 21→24; if r ≥ 0.8, tighten toward 18 and say
  so to the user immediately (budget honesty).
- **P2: live pose-mix audit.** Record camera pose per live tick (or at least per spike frame >2
  slots). Decision: if spike poses are NOT oblique-like, promote the observed worst pose into the
  canon and re-derive §2.3.
- **P3: oblique/aerial bimodality isolation.** 128-frame isolated runs with the GI probe cycle
  frozen (ProbeGI 3072/frame, 128-frame cycle — the strongest surviving suspect) vs default.
  Decision: modes collapse ⇒ GI cycle owns the gap between modes and its cost enters the post
  pool; modes persist ⇒ escalate to voxOccPyr/HZB interaction and Dawn/Metal pipelining readers.
- **P4 (conditional): election cost check.** Only if a reader claims the 32-bit election is hot:
  ?voxrdbg=2 stops before the Phase-B per-pixel election (NaniteVoxelRaster.ts:1137). Decision:
  oblique delta >3ms ⇒ open §3.2; else close the election trial permanently.
- **P5: post pool itemization at oblique.** Single session, interleaved: ablate clouds / ao /
  bounce / bloom / taa one at a time. Decision: any single item >1.5ms at oblique becomes a named
  Class Q/R candidate with its own quality review; items <1ms are declared closed.
