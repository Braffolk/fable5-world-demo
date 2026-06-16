# Perf harness hardening (2026-06-17)

Hardened the perf-measurement harness so nanite raster numbers are HONEST and robust. Motivated by the
2026-06-16 finding that per-pass GPU timestamps read a bogus ~constant (~15.2 ms) when the frame is
sub-vsync. Numbers below are **subagent-reported, pending independent spot-check** (`probe-motion.ts` re-run).

## The fix (two artifacts)
1. **Vsync / cross-frame pipelining artifact** — under rAF/`setAnimationLoop` the GPU idles for vsync and
   the pass-begin timestamp can fire while a prior frame is still draining ⇒ begin→end captures pipelining
   latency, not active time. **Fix:** `src/core/MeasureHarness.ts` (`__laas.measureFrames`) stops rAF,
   drives `engine.renderStep(dt)` back-to-back with no vsync wait, and **isolates** each frame with
   `device.queue.onSubmittedWorkDone()` before+after so the queue is empty ⇒ begin→end = real active GPU
   time. A `capSuspect` guard rejects any per-pass span exceeding its wall-clock GPU span (a pipelining ghost).
2. **Thermal throttling** — the drained flat-out loop heats the GPU; without cooldown, samples swing wildly
   (cooldown 0 → 27.7 ms/±24; 50 ms → 18.9 ms/±1.6). **Fix:** ~50 ms idle between samples + steady-state
   convergence (`measureActiveGpuSteady`) ⇒ absolute numbers are thermal-history-independent.

## New honest numbers (subagent-reported)
- **world1 (full pipe `?scene=forest&nanite=1`) @ 1280×720 ≈ 17.5 ms** @ ~347k visClusters (the artifact
  had been hiding this behind a flat ~15.2 ms).
- **combined (lean `?nanitedbg=flat`) @ 1280×720 ≈ 15.2 ms** @ ~349k visClusters.
- Scaling: world1 ~26.6 ms/Mcl, combined ~14.4 ms/Mcl; both LINEAR in visClusters, ~quadratic in resolution
  (2× res → +34.7 ms). **Vsync-immunity proof:** world1 now 12.98→32.11 ms across 167k→887k clusters
  (was flat ~15.2 ms). The number tracks work again.

## Motion track (the "during movement" target, finally measurable)
- `buildTrack` = a fixed 97-pose deterministic path (linear keyframe interp, no lib) through the worst
  angles: long alley, diagonal corridor, horizon-grazing, canopy dive-through. Booted `freeze=1`.
- Distribution of honest world1: **p50 17.56 / p95 26.80 / worst 29.03 ms** (worst = canopy dive, ~423k cl).
- Replay noise floor: fixed-pose adjacent-rep |Δ| ~0.66 ms; sparse per-pose on the long track ~1.4 ms
  (residual = slow thermal cooling drift, cancelled by a bracketed A/B).

## Files (the reusable harness)
- `src/core/MeasureHarness.ts` (new) — in-browser honest-timing core.
- `src/core/Engine.ts` — extracted `renderStep(dt)` (shared by rAF + harness) + `device`/`gpuProfiler` getters.
- `src/core/Hooks.ts` + `src/main.ts` — `__laas.measureFrames` hook (lazy, single-in-flight guard).
- **`tools/measure.ts` (new — the SHARED module probes import):** `measureActiveGpu` /
  `measureActiveGpuSteady` (honest timing), `buildTrack` / `replayMotionTrack` (motion), `abAcceptance`
  (bracketed/alternated boots + Hodges-Lehmann + bootstrap diff-CI), `bootPage`. Metric key is a parameter
  (`c.nanRasterWorld1` | `c.nanRasterCombined`).
- `tools/probe-motion.ts` (new) — the self-proof (blocks A–F).
- **Usage:** `import { bootPage, measureActiveGpu, abAcceptance } from './measure'` — never read
  `stats.gpuPasses` off the live rAF loop again.

## Validation
- Negative control: 2× res → world1 +34.67 ms, 95% CI [32.70, 36.44] (excludes 0). PASS.
- Known signal: simband 3→60 → world1 16.97→28.64 ms, cluster count tracking 233k→859k. PASS.
- No regression: rAF loop resumes after `measureFrames`; `settle()` works; forest renders; `tsc` clean, zero `any`. No commit.

## CAVEATS (how it can still be fooled)
1. **Thermal throttling is the dominant residual.** Cooldown + steady-state handle short bursts / fixed-pose;
   a long sequential motion track still has slow cooling drift. ⇒ **For acceptance use the bracketed/
   alternated A/B (`abAcceptance`), NOT raw track aggregates** — it cancels drift.
2. Headless refresh measured ~60 Hz (not 120); harness measures it at runtime.
3. `capSuspect` catches pipelining ghosts (span > wall) but NOT throttle-inflation (slow-but-real-looking);
   steady-state convergence + the noise floor are the second line of defense.

## Implication for past numbers
Any earlier number read off `stats.gpuPasses` on the live rAF loop (old probe-diag / probe-forestfull, and
parts of the 2026-06-16 diagnosis) was vsync-contaminated. Re-baseline through `tools/measure.ts`.
