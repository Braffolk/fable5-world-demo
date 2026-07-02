# live-loop & CPU deep review (2026-07-02)

Scope: the LIVE frame loop vs isolated measurement — rAF pacing, cpu.update/cpu.submit,
worker traffic, GC, per-frame uploads, the live ≈ 0.6–0.7× isolated relationship, and the
post-voxbocc live re-milestone plan. Files: `src/core/Engine.ts`, `src/core/MeasureHarness.ts`,
`src/debug/ForestScene.ts`, `src/nanite/NaniteFrame.ts`, `src/nanite/DagWorkerClient.ts`,
`src/nanite/TerrainStreamer.ts`, `src/nanite/FarTiles.ts`, `tools/probe-fresh-stutter.ts`.
All numbers computed from the session JSONs (`fresh-final-rested.json`,
`fresh-voxbocc-milestone.json`, `fresh-voxbocc.json`, `fresh-ablate-post3.json`) in the
session scratchpad. No GPU probes were run for this doc.

---

## Premise audit

**PA1 — The live milestone NEVER visits the worst pose.** Phase A of
`tools/probe-fresh-stutter.ts:129-167` is an eye-level glide only: `y = ground + 1.8`,
`pitch −0.02`, forward 8 m/s under a sweeping yaw. The oblique pose ([0,40,40], pitch −0.35 —
isolated 36.3–37.2 ms, the whole remaining gap) and aerial are visited ONLY in isolated
phase B. Every live verdict quoted in the mission ("p50 16.7, 0 stutters", "35% of frames at
25.0") is an **eye-level statement**. The live cost at oblique — the number locked-60 actually
turns on — has never been measured. This is the single most consequential gap in the
apparatus; probe 1 below closes it.

**PA2 — "live ≈ 0.6–0.7× isolated" is a quantization interval + a power-state effect, not a
law.** Live rAF deltas are hard-quantized to 8.33 ms vsync slots at 120 Hz: over the 600
milestone ticks the mean |residual| from the 8.333 grid is **0.181 ms**
(fresh-voxbocc-milestone). A "live p50 = 16.7" only bounds true per-frame GPU busy to
(8.33, 16.7]. Fitting the slot mixture (28×1-slot, 451×2-slot, 121×3-slot) puts the true live
eye cost at μ ≈ 14–15 ms, σ ≈ 3.5–4 → r = live/isolated ≈ 14.5/21.0 ≈ **0.69 at eye, this run**
— consistent with the folk 0.6–0.7 but derived, not assumed, and **unknown at oblique**
(PA1). Doc `90-premise-audit.md` §2.1 reaches the same interval conclusion independently;
this doc adds the mechanism decomposition (§Work model) and the per-pose unknowns.

**PA3 — The CPU-side threat model in my focus mostly does not exist in this scene.** Measured
live (600 ticks, both milestones): `cpu.update` avg **0.02 ms** (the only registered updateFn
is fly-camera, `src/main.ts:102`; probe runs `hud=false` so `HUD.ts:57` never registers);
`cpu.submit` avg **1.05–1.12 ms**, p95 1.5, max 2.2; **0 longtasks**; heap flat to the
counter's resolution for 600 frames. Worker traffic: `DagWorkerClient`/`TerrainStreamer` are
terrain-only and the forest scene has no terrain streamer (ForestScene builds its own registry;
`Heightfield.generate` only, ForestScene.ts:444); FarTiles workers are boot-only and terminated
after the build (`FarTiles.ts:324,333`). Per-frame buffer re-uploads: none (`needsUpdate`
appears only in build/attach paths, not per frame). **The live loop is ≥93% GPU; CPU is not a
lever for frame rate.** The area's value is measurement-model corrections, not CPU cuts.

**PA4 — The probe's "spike" counter is mislabeled at the 4-slot boundary.** The spike
threshold `d > 33.4` (probe-fresh-stutter.ts:158) is exactly the 4-slot edge, so the 51
"spikes" in fresh-final-rested are all ordinary 33.4–34.3 ms 4-slot frames — the high mode of
the frame-cost distribution, not stutter events. Real stutters (>100 ms, longtask-correlated)
are genuinely gone: 0 in both milestones.

**PA5 — capSuspect rejection is dead** (corroborates `10-frame-orchestration.md` P2):
capRejects = 30–32 of 32 on every pose of every run checked; the probe then uses all frames
(probe-fresh-stutter.ts:211-213). Medians remain honest (they are gpuWall), but no outlier
filtering exists.

**PA6 — heap flatness is weak evidence; longtasks are the real GC signal.** The probe samples
legacy `performance.memory.usedJSHeapSize` (quantized/cached by Chrome). Flat 5275.1 →
5275.1 MB over 600 frames may be an API artifact. The trustworthy no-GC-hitch evidence is 0
longtasks + the 0.18 ms mean vsync residual. Separately: **the JS heap floor is 5.3–5.5 GB**
(registry CPU staging arrays are kept forever — known backlog, CONTINUATION §4.8). Not a
frame-time item, but it is most of the machine's memory for one tab.

---

## How it works today

**rAF pacing.** `Engine.start()` → `renderer.setAnimationLoop(frame)` (Engine.ts:133-134).
`frame(timeMs)` computes rawDt from the rAF timestamp, clamps dt to [0, 0.1 s]
(Engine.ts:177-181), calls `renderStep(dt)` then `collectStats(rawDt)`. One rAF, no
requestVideoFrameCallback, vsync at 120 Hz on this panel — hence the 8.33 ms slot grid.

**renderStep** (Engine.ts:145-165): updateFns (fly only) → `post.meter(renderer)` →
`post.render()`, bracketed by `performance.now()` into `cpu.updateMs100` / `cpu.submitMs100`.
For the forest full pipe, `engine.post` is the NaniteFrame handle pair (ForestScene.ts:463),
so:

- `meter` = NaniteFrame.ts:499-572: PostStack exposure kernel (one `renderer.compute` submit
  per frame, PostStack.ts:678-681 — GPU histogram-free metering, **no readback**), TRAA
  jitter-index counter mirror, and **every 15th frame** a `Promise.all` of 4–6 async readbacks
  (`readCounts`, `readHwCount`, `readVoxCount`, `readVoxWrites`; NaniteFrame.ts:509-519 →
  `readBuffer` → `renderer.getArrayBufferAsync`, Tsl.ts:309-317). These are async
  staging-buffer copies; they never stall the queue.
- `render` = NaniteFrame.ts:425-497: `cam.update(jitteredCamera())` (pure CPU matrix math +
  ~9 small uniform writes, NaniteCommon.ts:108-124) → cull BFS submit → `syncFullArgs` →
  `runVoxFanout` → `raster.world1` (owns the vis clear) → `hzb.build` → (shadow: **never built
  in forest** — `world.csm === null`, NaniteFrame.ts:244, see doc 10 P1) → `post.render()`
  (scene pass with the two resolve meshes + post chain).

**collectStats** (Engine.ts:193-239): 120-entry ring sort for p95, fps EMA, `renderer.info`
memory counters, and **every frame** `resolveTimestampsAsync(RENDER) + (COMPUTE)` feeding
GpuProfiler. `trackTimestamp: true` is set at renderer construction (Engine.ts:74), so every
render/compute context (~100 per frame per the Engine.ts:219-222 comment) carries a
begin/end timestamp-query pair, live, always — even though nothing consumes them when the HUD
is off.

**Isolated protocol** (MeasureHarness.ts:122-184): stop rAF → 20 warmup renderSteps each
followed by a full drain → per sample: 50 ms sleep → drain → `renderStep(1/120)` → drain;
`gpuWallMs = tSubmit→tDone` where tSubmit is after ALL of the frame's submits are queued.
So isolated gpuWall = serial GPU execution of a fully-queued frame + fence/event-loop
resolution latency − the encode-window head start (the GPU starts submit 1 while the CPU
encodes the rest; that overlap is excluded).

**Live pipelining.** WebGPU has ONE queue; passes execute serially. In steady state the CPU
(1.1 ms) finishes encoding frame N+1 long before the GPU finishes N, so the queue never
starves; live throughput = per-frame GPU busy time, rounded UP to vsync slots by rAF. Two
consequences: (a) **submit-count coalescing is live-neutral** — the queue is never empty
between a live frame's submits, so doc 10's `coalesce-submits` lever should be expected to
clean ISOLATED numbers only (~0 live ms); (b) the rAF delta is a ceiling function of GPU busy,
never an average — all live wins must be judged on slot histograms (doc 90 §4 agrees).

---

## Work model

**Live:** `delta = 8.333 · ceil(gpuBusy / 8.333)` (residual 0.18 ms). CPU total
(update+encode) ≈ 1.15 ms/frame runs concurrently with the previous frame's GPU tail and never
gates. Frame cost scales with the same things isolated cost does — live visTris p50 4.38M ≈
isolated eye 4.44M (fresh-voxbocc-milestone counters), so the eye glide is workload-equivalent
to the eye pose.

**What the milestone data says (post-voxbocc, eye glide, 600 ticks):**
- Slots {1: 28, 2: 451, 3: 121}. The 121 3-slot frames are **isolated singletons** (run-length
  histogram {1: 119, 2: 1}) and are **not workload spikes** (their visTris avg 4.295M vs
  4.317M for 2-slot frames — slightly LOWER). They ramp over the capture: 0 per 50 ticks at
  the start → ~18 per 50 ticks at the end (thermal/clock drift across the 35 s run).
- Interpretation: true live eye cost sits at ~14–15 ms and **straddles the 16.7 boundary**
  under thermal drift + noise. A ~2 ms quality-identical GPU shave (from any other reviewer's
  lever) locks the eye-level path at 60. The eye-level mission is a boundary problem, not a
  mode problem.

**Live/isolated ratio decomposition (eye, same run):** isolated 21.0 vs fitted live ~14.5 →
gap ≈ 6.5 ms ≈ 31%. Candidate shares: (1) DVFS/duty — isolated runs at ~25% GPU duty
(50 ms sleep + ~20 ms work per sample) vs ~85%+ sustained live; Apple GPUs downclock at low
duty, and Metal microbenchmark lore says bursty measurement reads 20–40% slow — this is the
prime suspect for most of the gap; (2) fence/event-loop latency inside gpuWall (+0.5–1.5 ms
per isolated sample, MeasureHarness.ts:150-160); (3) minus the encode-overlap exclusion
(−1–3 ms, works the other way). Probe 2 separates (1) from (2)/(3) by sweeping duty.

**The bimodality, seen from this area (autocorrelation of the isolated gpu arrays):**
- fresh-voxbocc aerial: ACF lag-3 = +0.28, lag-6 = +0.35 → clean period-3 cycle
  (12 → 22 → 17 ms repeating). Per-sample wall time ≈ 50 + 16 + drain ≈ ~70 ms → ~210 ms
  oscillation.
- fresh-final-rested oblique: ACF lag-1 +0.36, lag-3 −0.54, lag-6/7 +0.41/+0.40 → period ~6–7
  (the "runs of ~4" banding). Per-sample wall ≈ ~95 ms → ~600 ms oscillation.
- fresh-voxbocc-milestone (warm) oblique/aerial: structure mostly gone (|ACF| ≤ 0.2).
- Not workload-correlated; period varies with pose/run/thermal era. Consistent with a
  power-governor limit cycle riding the 50 ms-sleep protocol (doc 10 P4 reached the same
  suspect); the run-to-run period drift means the wall-period is load-dependent, so the
  cooldown-sweep discriminator (probe 2) is required before ANY renderer-side "periodic cost"
  hunt. **In live data the same "bimodality" is just boundary straddle**: final-rested live
  4-slot frames are 158 singletons + 11 pairs, spread evenly with a mild ramp — no periodic
  banding on the real loop.

**What locked-60 actually requires (slot math):** every pose's live GPU busy < 16.7 ms.
With r ∈ [0.65, 0.80] (unpinned at oblique), the required isolated oblique med is
16.7/r ≈ **21–26 ms**. The fact pack's "−11..13 ms at oblique" is the r≈0.67–0.70 case; if
r at oblique is 0.80, the required cut is −16 ms. Pinning r at oblique (probe 1) is therefore
worth up to ~5 ms of mission ledger by itself.

---

## Waste inventory

Quantified per live frame (16.7 ms budget), best-effort:

| item | cost | evidence |
|---|---|---|
| CPU total (update + encode) | ~1.15 ms, max 2.2 | live counters both milestones |
| — redundant matrix math | ~0 (cam.update is ~30 flops + 9 uniform writes) | NaniteCommon.ts:108-124 |
| — allocations in the hot path | dust: `new Vector2()` per render (NaniteFrame.ts:426), `params.get` per frame (NaniteFrame.ts:398,480,495), ring sort of 120 (Engine.ts:198) | code read; heap flat, 0 longtasks |
| static-data re-uploads | none found (no per-frame `needsUpdate`; instance streams uploaded once) | grep + build-path read |
| timestamp machinery live | ~100 contexts × 2 query writes + 2 resolve passes + promise churn per frame, consumed by NOTHING when hud=false | Engine.ts:74,219-239 comment; unmeasured, bounded ~0–0.5 ms GPU |
| meter readbacks | 4–6 staging copies every 15 frames, async | NaniteFrame.ts:509-519; amortized ≪0.1 ms |
| exposure kernel | 1 small compute submit/frame (needed for auto-exposure) | PostStack.ts:678-681 |
| worker traffic | 0 (boot-only; FarTiles workers terminated) | FarTiles.ts:324,333 |
| GC | 0 observable | 0 longtasks, 600-tick residual 0.18 ms |
| JS heap floor | 5.3–5.5 GB held forever (registry staging arrays) | live heap counter; CONTINUATION §4.8 |

Honest summary: **there is no material CPU-side waste**. The measurable waste in this area is
in the measurement apparatus (isolated protocol duty-cycle artifact, missing live-oblique
coverage, dead capSuspect filter), which distorts the mission ledger by multiple ms.

---

## Levers

Quality classes per the ABSOLUTE constraint. No quality-trading lever exists in this area
(nothing here touches pixels), so there is no REJECTED-BY-POLICY section.

### live-loop:pose-complete-live-milestone — measure live at oblique/aerial (S, IDENTICAL)
Mechanism: extend phase A with a `POSE_PATH` env: segments that hold/slow-orbit each canonical
pose (eye glide as today → oblique [0,40,40] pitch −0.35 slow yaw sweep → aerial hold),
recording per-segment slot histograms. `tools/` is vite-ignored (safe to edit while probes are
queued). Expected direct ms: 0 — it re-aims the mission: pins r at the worst pose, converting
the oblique target from a 21–26 ms interval to a number (worth up to ~5 ms of ledger), and
gives the honest "how far from locked-60 is the user experience at the drone view" answer.
Discriminator: the probe IS the discriminator; decision rule — required iso-oblique =
16.7 × (iso-oblique-same-run / fitted-live-oblique). Risk: none (measurement only).

### live-loop:dvfs-duty-discriminator — cooldown sweep in phase B (S, IDENTICAL)
Mechanism: pass `cooldownMs` through `measureFrames` (MeasureOptions already supports it,
MeasureHarness.ts:78; the probe just needs to forward an env). Sweep 0/10/50/200 ms at fixed
pose. Decision rules: (a) if med gpuWall moves >10% with duty → a chunk of the live/isolated
gap is the governor, and isolated numbers get an era-correction; (b) if the band period in
FRAMES scales inversely with per-sample wall time (constant wall-period) → the oblique/aerial
bimodality is a power-state limit cycle, and the "4–9 ms periodic cost" line leaves the
renderer's ledger entirely. Expected direct ms: 0 (measurement; potentially removes a phantom
4–9 ms target). Risk: none.

### live-loop:gate-timestamps-off-live — ?prof flag for trackTimestamp + per-frame resolves (S, IDENTICAL)
Mechanism: default `trackTimestamp:false` + skip the per-frame
`resolveTimestampsAsync` pair (Engine.ts:74, 219-239) unless `?prof=1`; probes add `prof=1`
automatically (MeasureHarness.create returns null without the profiler, MeasureHarness.ts:112
— phase B needs it, so the probe URL must carry the flag; alternatively let MeasureHarness
fall back to gpuWall-only). Removes ~200 timestamp query writes + 2 resolve passes + promise
churn per live frame. Expected: 0–0.5 ms per pose (unmeasured; the ~100-pass count is real but
per-write cost is small). Quality: identical trivially (no pixel path touched).
Discriminator: live A/B slot histogram `EXTRA=prof=0` vs default, 600 ticks — judged on the
3-slot fraction at eye. Risk: probe/HUD workflows silently losing per-pass stats if a caller
forgets the flag; mitigate by warning once when gpuPasses is read while off.

### live-loop:meter-readback-gate — counters only when consumed (S, IDENTICAL, dust)
Mechanism: gate the every-15-frame readback bundle (NaniteFrame.ts:509-519) behind
hud-visible/probe flag; keep the exposure kernel. Expected ≤0.05 ms/pose — bundle with the
timestamps lever, not worth its own probe. Risk: HUD counters go stale if gating is wrong;
probes read counters, so the probe URL must keep them on (they already sample
`stats.counters` per tick in phase A).

### live-loop:release-staging-arrays — drop the 5 GB heap floor (M, IDENTICAL, 0 frame-ms)
Mechanism: after `reg.build()`+`flush()` the CPU-side staging arrays (verts/indices/clusters/
bricks and the FarTiles pyramids — partially released at ForestScene.ts:393) are never needed
again in the forest path; null them (keep behind a `debug()` opt-in — `?vrange=1` and
`registry.debug()` at NaniteFrame.ts:93 are the only consumers). Expected frame ms: 0 (GC is
already silent); the win is −4–5 GB resident, faster tab lifecycle, and removing the
"could GC-spike someday" tail risk. Risk: any late `attachDag`/`appendVoxelCrown`/debug path
touching freed arrays — needs an audit of GeometryRegistry consumers; M effort for that audit.

### Cross-reference: frame-orchestration:coalesce-submits is live-NEUTRAL
Per the §Work model queue analysis, live frames never starve the queue between submits
(CPU 1.1 ms vs GPU 14+ ms), so submit coalescing improves ISOLATED hygiene only. The master
plan should not book live ms for it. (Isolated hygiene is still valuable: it shrinks the
isolated↔live modeling gap that PA2 complains about.)

---

## What UE5/prior art does here

- **UE5** decouples measurement from presentation: `stat unit` separates Game/Draw/GPU
  threads; GPU benchmarking guidance is to measure under SUSTAINED load precisely because
  desktop/mobile DVFS makes bursty microbenchmarks read slow — consoles pin clocks; Apple
  Silicon cannot. Our MeasureHarness's 50 ms-sleep protocol is the textbook bursty case.
- **Frame pacing**: UE5's RHI frame pacer targets slot multiples explicitly (the same ceil()
  math as PA2) and reports "hitches" only above a pacing-aware threshold — our probe's 33.4 ms
  "spike" label is the same idea, but the threshold happens to sit exactly ON the 4-slot edge
  (PA4).
- **Timestamp overhead**: Chrome/Dawn guidance is to keep `timestamp-query` off in production
  loops; the passes-carry-queries-always pattern (three's `trackTimestamp`) is a known dev-mode
  tax. UE5 compiles out GPU stats in Test/Shipping configs — the `?prof` gate is the
  equivalent.
- **CPU-side**: UE5's render thread costs are dominated by draw setup — irrelevant here; a
  GPU-driven single-queue WebGPU pipe with 1.1 ms encode is already past UE5's own CPU story.

---

## Open questions + proposed serial probes

Small probe edits first (all in `tools/`, vite-ignored, safe while GPU jobs queue):
(i) `POSE_PATH` env for phase A segments; (ii) forward `MEASURE_CD_MS` to
`measureFrames({cooldownMs})`; (iii) label the >33.4 counter "4-slot frames" and add a real
stutter counter at >50 ms.

1. **Pose-complete live milestone (the missing number).** After edit (i):
   `CONFIG=default LABEL=live-poses TREES=200000 TICKS=900 POSE_PATH=eye,oblique,aerial COOLDOWN_S=60 npx tsx tools/probe-fresh-stutter.ts`
   Output: per-segment slot histograms + same-run isolated meds. Decision: required
   iso-oblique target = 16.7 × iso/live-fit ratio at oblique; if the oblique segment already
   shows ≥50% 2-slot frames, the remaining gap is smaller than the ledger says.
2. **DVFS/duty discriminator.** After edit (ii), fixed default config, TICKS=0, one pose per
   run at oblique first:
   `CONFIG=default LABEL=duty-cd0 TICKS=0 FRAMES=48 WARMUP=20 MEASURE_CD_MS=0 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts`
   then `LABEL=duty-cd10 MEASURE_CD_MS=10`, `LABEL=duty-cd200 MEASURE_CD_MS=200`.
   Decision rules in the lever above. Order the sweep low→high cooldown so thermal drift biases
   AGAINST the low-duty-reads-slow hypothesis.
3. **Timestamp tax A/B (live only).** After the `?prof` gate lands:
   `CONFIG=default LABEL=live-noprof TICKS=600 FRAMES=0 EXTRA=prof=0 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts`
   vs `LABEL=live-prof TICKS=600 FRAMES=0` control. Judge on the eye 3-slot fraction; if
   Δ < 2 percentage points, book it as dust and keep timestamps on for convenience.
4. **Open question — aerial pose-arrival transient in phase B**: fresh-ablate-post3 aerial
   frames 0–2 are 6.5/11.5/9.5 ms (stale-HZB over-cull) DESPITE settle(20)+warmup(20); either
   warmup isn't converging the HZB at teleport or the DVFS ramp masquerades as convergence.
   The duty sweep (probe 2) at aerial disambiguates: a governor effect scales with cooldown,
   a stale-HZB effect does not.
5. **Open question — presentation mode**: is the panel locked at 120 Hz for the probe's
   headless chromium (slot grid says yes, 8.333 exactly)? If a future machine/display runs
   60 Hz, all slot math (and "p50 25.0 = 3 quanta") re-derives; the probe should record the
   measured refresh (MeasureHarness.measureRefresh exists, MeasureHarness.ts:188-202 — surface
   it in the JSON).
