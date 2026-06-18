/**
 * tools/measure.ts — the SHARED, reusable perf-measurement harness.
 *
 * Probes IMPORT this module (do not copy-paste). It provides HONEST active-GPU
 * time for the nanite raster kernels, robust to the vsync cap, plus a
 * deterministic motion track and an A/B acceptance protocol.
 *
 * WHY a shared module: the old probes each read st.gpuPasses['c.nanRasterWorld1']
 * straight off the live rAF loop — which, when the frame is sub-vsync, is the
 * BOGUS ~15.2 ms cross-frame pipelining span, NOT active GPU time (proven: flat
 * across 242k→3.3M clusters and 2× res). Everything here routes through
 * __laas.measureFrames (MeasureHarness in-browser) which drives frames manually,
 * GPU-bound + isolated, so the timestamp is real active GPU time.
 *
 * THREE PUBLIC SURFACES:
 *   measureActiveGpu(page, metricKey, opts)  — Req 1: honest per-pass time + distribution
 *   replayMotionTrack(page, track, metricKey) — Req 2: deterministic camera path
 *   abAcceptance(...)                         — Req 3: bracketed alternated A/B, diff-CI
 *
 * Metric keys are PARAMETERIZED so the same harness serves both raster kernels:
 *   c.nanRasterWorld1   — full pipe   (?scene=forest&nanite=1)
 *   c.nanRasterCombined — lean debug  (?scene=forest&nanitedbg=flat)
 *
 * Serialize ALL boots — one page / one GPU at a time.
 */

import type { Page } from "playwright";
import { laasUrl, launchWebGPU, type LaasPageOptions } from "./launch";

// ---------------------------------------------------------------------------
// wire types (mirror src/core/Hooks.ts — kept in sync; serializable over the
// Playwright page.evaluate boundary)
// ---------------------------------------------------------------------------
export interface MeasuredFrameWire {
  passes: Record<string, number>;
  gpuWallMs: number;
  cpuSubmitMs: number;
  counters: Record<string, number>;
  capSuspect: boolean;
  refreshMs: number;
}

export interface MeasureOpts {
  /** isolated frames to MEASURE (after warmup) */
  frames?: number;
  /** warmup frames (drained, not measured) — lets streaming/TRAA converge */
  warmup?: number;
  /** per-frame dt (sec) fed to renderStep; default 1/120 */
  dt?: number;
  /** idle cooldown (ms) between samples — defeats the flat-out thermal drift (default 50, calibrated) */
  cooldownMs?: number;
}

// ---------------------------------------------------------------------------
// stats primitives
// ---------------------------------------------------------------------------
export function median(a: number[]): number {
  const s = a.filter((x) => Number.isFinite(x)).sort((x, y) => x - y);
  return s.length === 0 ? 0 : (s[Math.floor(s.length / 2)] as number);
}

export function percentile(a: number[], p: number): number {
  const s = a.filter((x) => Number.isFinite(x)).sort((x, y) => x - y);
  if (s.length === 0) return 0;
  const idx = Math.min(s.length - 1, Math.max(0, Math.floor(s.length * p)));
  return s[idx] as number;
}

/** Hodges-Lehmann location estimate: median of all pairwise averages (robust). */
export function hodgesLehmann(a: number[]): number {
  const x = a.filter(Number.isFinite);
  if (x.length === 0) return 0;
  const avgs: number[] = [];
  for (let i = 0; i < x.length; i++) {
    for (let j = i; j < x.length; j++) avgs.push((x[i]! + x[j]!) / 2);
  }
  return median(avgs);
}

/**
 * Difference of two independent samples with a bootstrap CI on the median
 * difference (B resamples). Returns Δ = median(b) − median(a) and a 95% CI.
 */
export function bootstrapDiffCI(
  a: number[],
  b: number[],
  B = 2000,
): { delta: number; lo: number; hi: number } {
  const ca = a.filter(Number.isFinite);
  const cb = b.filter(Number.isFinite);
  const delta = median(cb) - median(ca);
  if (ca.length === 0 || cb.length === 0)
    return { delta, lo: delta, hi: delta };
  const draws: number[] = [];
  for (let k = 0; k < B; k++) {
    const ra: number[] = [];
    const rb: number[] = [];
    for (let i = 0; i < ca.length; i++)
      ra.push(ca[(Math.random() * ca.length) | 0]!);
    for (let i = 0; i < cb.length; i++)
      rb.push(cb[(Math.random() * cb.length) | 0]!);
    draws.push(median(rb) - median(ra));
  }
  draws.sort((x, y) => x - y);
  return {
    delta,
    lo: draws[Math.floor(B * 0.025)] ?? delta,
    hi: draws[Math.floor(B * 0.975)] ?? delta,
  };
}

// ---------------------------------------------------------------------------
// Requirement 1 — honest active-GPU time for a parameterized metric key
// ---------------------------------------------------------------------------
export interface ActiveGpuResult {
  /** metric key measured (e.g. c.nanRasterWorld1) */
  key: string;
  /** per-frame honest active-GPU samples of `key` (cap-suspect frames excluded) */
  samples: number[];
  /** distribution of `key` */
  p50: number;
  p95: number;
  worst: number;
  /** measured display refresh interval (ms) — the cap we are immune to */
  refreshMs: number;
  /** how many measured frames were rejected as cap/pipelining-suspect */
  rejected: number;
  /** companion metrics at the same frames (median): frame counters + wall */
  visClusters: number;
  gpuWallMs: number;
  /** the raw frames (for paired A/B + audits) */
  frames: MeasuredFrameWire[];
}

/**
 * Drive `opts.frames` ISOLATED GPU-bound frames in the browser and return the
 * honest active-GPU-time distribution for `metricKey`. Cap-suspect frames (where
 * isolation looked incomplete) are REJECTED, never averaged in.
 */
export async function measureActiveGpu(
  page: Page,
  metricKey: string,
  opts: MeasureOpts = {},
): Promise<ActiveGpuResult> {
  const frames = (await page.evaluate(
    async (o) => {
      if (!window.__laas.measureFrames) return null;
      return window.__laas.measureFrames(o);
    },
    {
      frames: opts.frames ?? 30,
      warmup: opts.warmup ?? 12,
      dt: opts.dt,
      cooldownMs: opts.cooldownMs,
    },
  )) as MeasuredFrameWire[] | null;
  if (!frames || frames.length === 0) {
    throw new Error(
      "__laas.measureFrames returned nothing — timestamp-query unavailable or harness not wired",
    );
  }
  const refreshMs = frames[0]!.refreshMs;
  const good = frames.filter(
    (f) => !f.capSuspect && Number.isFinite(f.passes[metricKey]),
  );
  const rejected = frames.length - good.length;
  const samples = good.map((f) => f.passes[metricKey] ?? 0);
  const vis = good.map((f) => f.counters["nanite.visClusters"] ?? -1);
  const wall = good.map((f) => f.gpuWallMs);
  return {
    key: metricKey,
    samples,
    p50: median(samples),
    p95: percentile(samples, 0.95),
    worst: samples.length ? Math.max(...samples) : 0,
    refreshMs,
    rejected,
    visClusters: median(vis),
    gpuWallMs: median(wall),
    frames,
  };
}

/**
 * THERMAL-STEADY-STATE measurement. A single measure right after boot reads HIGH
 * because boot+settle+warmup left the GPU warm (calibration: 32.8 ms warm vs the
 * 18.9 ms cold steady-state at the same pose/clusters). Throttling only ever
 * INFLATES active time, so the honest number is the converged (lowest-stable)
 * one. This repeats `measureActiveGpu` (each with cooldown) until two consecutive
 * reps agree within `tolMs`, then returns that steady result — making the
 * absolute number independent of thermal history.
 */
export async function measureActiveGpuSteady(
  page: Page,
  metricKey: string,
  opts: MeasureOpts & { maxReps?: number; tolMs?: number } = {},
): Promise<ActiveGpuResult & { reps: number; converged: boolean }> {
  const maxReps = opts.maxReps ?? 6;
  const tolMs = opts.tolMs ?? 1.5;
  let prev: ActiveGpuResult | null = null;
  let best: ActiveGpuResult | null = null; // lowest-p50 rep = least throttled = most honest
  for (let i = 0; i < maxReps; i++) {
    const r = await measureActiveGpu(page, metricKey, opts);
    if (!best || r.p50 < best.p50) best = r;
    if (prev && Math.abs(r.p50 - prev.p50) <= tolMs) {
      // two consecutive reps agree → steady state reached; report the lower of them
      const conv = r.p50 <= prev.p50 ? r : prev;
      return { ...conv, reps: i + 1, converged: true };
    }
    prev = r;
  }
  // didn't converge — return the LOWEST-p50 rep (least throttled) and flag it
  return { ...best!, reps: maxReps, converged: false };
}

// ---------------------------------------------------------------------------
// boot helpers
// ---------------------------------------------------------------------------
export interface BootOpts extends LaasPageOptions {
  /** time of day (applied via setTimeOfDay after boot) */
  T?: number;
  /** warmup settle frames after boot (streaming/exposure) */
  settle?: number;
  /** forward console lines with this prefix */
  logPrefix?: string;
}

/** boot a page and wait until __laas is ready (or throw on fatal). */
export async function bootPage(
  browser: Awaited<ReturnType<typeof launchWebGPU>>["browser"],
  opts: BootOpts,
): Promise<Page> {
  const width = opts.width ?? 1280;
  const height = opts.height ?? 720;
  const page = await browser.newPage({
    viewport: { width, height },
    deviceScaleFactor: 1,
  });
  page.on("pageerror", (e) => console.error("[pageerror]", e.message));
  if (opts.logPrefix) {
    const pre = opts.logPrefix;
    page.on("console", (m) => {
      const t = m.text();
      if (t.startsWith(pre)) console.log(`   · ${t}`);
    });
  }
  const url = laasUrl({ ...opts, width, height, freeze: opts.freeze ?? false });
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error != null),
    undefined,
    { timeout: 300000, polling: 250 },
  );
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);
  if (opts.T !== undefined)
    await page.evaluate((t) => window.__laas.setTimeOfDay?.(t), opts.T);
  if (opts.settle)
    await page.evaluate(async (n) => window.__laas.settle?.(n), opts.settle);
  return page;
}

// ---------------------------------------------------------------------------
// Requirement 2 — deterministic, replayable MOTION track
// ---------------------------------------------------------------------------
export interface TrackPose {
  p: [number, number, number];
  yaw: number;
  pitch: number;
  fov?: number;
}

/**
 * A fixed camera path. Built from keyframes by Catmull-Rom-free linear interp
 * (deterministic, no library, bit-identical every replay). The worst angles are
 * encoded as keyframes (long alley, horizon-grazing, canopy dive).
 */
export interface MotionTrack {
  name: string;
  /** absolute per-frame poses (already expanded); replay drives these in order */
  poses: TrackPose[];
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** expand keyframes into `framesPerSeg` poses per segment (linear, deterministic). */
export function buildTrack(
  name: string,
  keys: TrackPose[],
  framesPerSeg: number,
): MotionTrack {
  const poses: TrackPose[] = [];
  for (let s = 0; s < keys.length - 1; s++) {
    const a = keys[s]!;
    const b = keys[s + 1]!;
    for (let f = 0; f < framesPerSeg; f++) {
      const t = f / framesPerSeg;
      poses.push({
        p: [
          lerp(a.p[0], b.p[0], t),
          lerp(a.p[1], b.p[1], t),
          lerp(a.p[2], b.p[2], t),
        ],
        yaw: lerp(a.yaw, b.yaw, t),
        pitch: lerp(a.pitch, b.pitch, t),
        fov:
          a.fov !== undefined && b.fov !== undefined
            ? lerp(a.fov, b.fov, t)
            : a.fov,
      });
    }
  }
  poses.push(keys[keys.length - 1]!);
  return { name, poses };
}

export interface MotionResult {
  key: string;
  /** per-pose honest active-GPU sample (one isolated frame per pose) */
  perPose: number[];
  /** companion per-pose visClusters */
  perPoseVis: number[];
  /**
   * PAIRED per-pose noise: |Δ| between two back-to-back measurements of the SAME
   * pose (thermally matched ⇒ the true harness noise floor, immune to the slow
   * GPU heat drift a long flat-out track accumulates). Empty unless paired=true.
   */
  pairedNoise: number[];
  p50: number;
  p95: number;
  worst: number;
  worstPoseIndex: number;
  refreshMs: number;
  rejected: number;
}

/**
 * Replay a motion track BIT-IDENTICALLY: set each pose, then measure ONE isolated
 * GPU-bound frame for that exact pose. Same seed + same per-frame pose ⇒ baseline
 * and candidate see identical workloads frame-for-frame (paired A/B downstream).
 *
 * framesPerPose>1 measures several isolated frames per pose and takes the median
 * (kills per-frame noise without changing the workload).
 */
export async function replayMotionTrack(
  page: Page,
  track: MotionTrack,
  metricKey: string,
  framesPerPose = 1,
  /**
   * warmup frames per pose. The two-phase occlusion cull seeds from the PREVIOUS
   * frame's HZB, so a fresh pose under-/over-culls for a frame or two until the
   * HZB occluder converges. ≥6 makes each pose's measured frame independent of
   * how we arrived at it ⇒ bit-identical replay (validate via the noise floor).
   * REQUIRES the track to be booted with freeze=1 (static geometry) for a true
   * zero-noise replay — otherwise wind/time advance between replays.
   */
  warmupPerPose = 8,
  /**
   * paired noise floor: at every Nth pose, measure TWICE back-to-back and record
   * the |Δ| — the thermally-matched noise floor. 0 = off. A SPARSE subset (e.g.
   * every 12th pose) keeps the extra load small so it doesn't itself heat-drift
   * the run; a whole-track re-run is NOT a valid noise estimate on a throttling
   * GPU (it measures thermal drift, not harness noise).
   */
  noiseEvery = 0,
): Promise<MotionResult> {
  const perPose: number[] = [];
  const perPoseVis: number[] = [];
  const pairedNoise: number[] = [];
  let refreshMs = 1000 / 120;
  let rejected = 0;
  for (let pi = 0; pi < track.poses.length; pi++) {
    const pose = track.poses[pi]!;
    await page.evaluate((pp) => window.__laas.setPose?.(pp), pose);
    const r = await measureActiveGpu(page, metricKey, {
      frames: framesPerPose,
      warmup: warmupPerPose,
    });
    refreshMs = r.refreshMs;
    rejected += r.rejected;
    perPose.push(r.p50);
    perPoseVis.push(r.visClusters);
    if (noiseEvery > 0 && pi % noiseEvery === 0) {
      // immediate second measurement at the SAME pose, SAME warmup + frames (HZB
      // already converged, cooldown holds both reps at steady state) ⇒ honest floor
      const r2 = await measureActiveGpu(page, metricKey, {
        frames: framesPerPose,
        warmup: warmupPerPose,
      });
      pairedNoise.push(Math.abs(r2.p50 - r.p50));
    }
  }
  const worstPoseIndex = perPose.reduce(
    (mi, v, i, arr) => (v > arr[mi]! ? i : mi),
    0,
  );
  return {
    key: metricKey,
    perPose,
    perPoseVis,
    pairedNoise,
    p50: median(perPose),
    p95: percentile(perPose, 0.95),
    worst: perPose.length ? Math.max(...perPose) : 0,
    worstPoseIndex,
    refreshMs,
    rejected,
  };
}

// ---------------------------------------------------------------------------
// Requirement 3 — A/B acceptance: bracketed, alternated separate boots,
// time-interpolated baseline, robust estimator, difference-CI.
// ---------------------------------------------------------------------------
export interface AbSide {
  label: string;
  /** extra URL params for this side's boot */
  extra: Record<string, string>;
}

export interface AbResult {
  baseLabel: string;
  candLabel: string;
  /** robust per-side location (Hodges-Lehmann of the pooled samples) */
  baseHL: number;
  candHL: number;
  /** median-difference + bootstrap 95% CI (cand − base); CI excluding 0 = significant */
  delta: number;
  lo: number;
  hi: number;
  significant: boolean;
  /** noise floor: base-vs-base replicate Δ (≈0 expected) */
  noiseDelta: number;
}

/**
 * Bracketed, ALTERNATED A/B across SEPARATE boots, modeled on probe-postablate:
 * order = base, cand, base, cand, … , base — so cross-boot thermal drift cancels
 * (each side is measured at interleaved times). Each boot drives the SAME isolated
 * measurement (or motion track via `measureSide`). Reports a robust diff-CI.
 *
 * `measureSide(page)` returns the per-side sample array (e.g. the motion track's
 * perPose array, or measureActiveGpu().samples). It must be DETERMINISTIC across
 * boots (same pose/track) so the comparison is paired-by-workload.
 */
export async function abAcceptance(
  bootSide: (side: AbSide) => Promise<Page>,
  measureSide: (page: Page) => Promise<number[]>,
  base: AbSide,
  cand: AbSide,
  rounds = 3,
): Promise<AbResult> {
  const baseSamples: number[][] = [];
  const candSamples: number[][] = [];
  // bracket: base, (cand, base) × rounds  — base book-ends + interleaves
  const order: AbSide[] = [base];
  for (let r = 0; r < rounds; r++) order.push(cand, base);
  for (const side of order) {
    const page = await bootSide(side);
    const s = await measureSide(page);
    await page.close();
    if (side.label === base.label) baseSamples.push(s);
    else candSamples.push(s);
  }
  // pool (paired-by-workload: each replicate measures the same deterministic track)
  const basePool = baseSamples.flat();
  const candPool = candSamples.flat();
  const { delta, lo, hi } = bootstrapDiffCI(basePool, candPool);
  // noise floor = first base replicate vs last base replicate
  const noise =
    baseSamples.length >= 2
      ? bootstrapDiffCI(baseSamples[0]!, baseSamples[baseSamples.length - 1]!)
          .delta
      : 0;
  return {
    baseLabel: base.label,
    candLabel: cand.label,
    baseHL: hodgesLehmann(basePool),
    candHL: hodgesLehmann(candPool),
    delta,
    lo,
    hi,
    significant: lo > 0 || hi < 0,
    noiseDelta: noise,
  };
}
