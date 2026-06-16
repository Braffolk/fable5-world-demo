/**
 * MeasureHarness — HONEST, vsync-artifact-proof per-pass GPU timing.
 *
 * THE PROBLEM IT FIXES (a proven measurement artifact, see NaniteRaster.ts:229
 * and the PERF-1 investigation): under the engine's rAF / setAnimationLoop the
 * frame is vsync-capped (~120 fps ⇒ frameMs ≈ 8.3 ms). When the GPU finishes a
 * frame early it sits IDLE until the next vsync. The per-pass WebGPU timestamp
 * query (writeTimestamp at pass begin/end) then reads a BOGUS ~constant span
 * (~15.2 ms for `c.nanRasterWorld1`) — a cross-frame PIPELINING span, NOT the
 * pass's active GPU time. Proof: that "floor" stayed pinned at ~15.2 ms across
 * 242k → 3.3M visible clusters and at 2× resolution (slope ≈ 0; real work would
 * scale). Only GPU-BOUND frames (frameMs ≫ refresh interval) give honest active
 * GPU time.
 *
 * THE FIX — decouple measurement from rAF:
 *   1. STOP the rAF loop (renderer.setAnimationLoop(null)).
 *   2. Drive engine.renderStep(dt) in a tight BACK-TO-BACK loop with NO vsync
 *      wait, so the GPU is always the bottleneck.
 *   3. ISOLATE each measured frame on the GPU timeline: drain the queue
 *      (device.queue.onSubmittedWorkDone()) BEFORE the frame so the queue is
 *      empty, submit the one frame, drain AGAIN so it fully completes, THEN
 *      resolve timestamps. With no overlapping work the writeTimestamp begin
 *      query fires at the true pass start ⇒ begin→end == real active GPU time.
 *   4. Read the GpuProfiler per-pass spans for that isolated frame.
 *
 * WHY THIS DEFEATS THE ARTIFACT: the ~15.2 ms artifact is cross-frame
 * pipelining latency. Draining before AND after the measured frame removes ALL
 * cross-frame overlap, so the pass's begin timestamp can only fire once the GPU
 * actually starts the pass. The reported span then SCALES with workload (the
 * self-validation requirement) instead of pinning to a constant floor.
 *
 * GUARD (never silently report a sub-vsync timestamp as real): even in manual
 * mode we record each frame's wall-clock step time and the measured refresh
 * interval; `capSuspect` flags any frame whose isolation looks incomplete. The
 * harness REJECTS (does not average) frames whose per-pass total exceeds the
 * wall-clock GPU span — those are pipelining ghosts, not active time.
 */

import { TimestampQuery } from 'three/webgpu';
import type { Engine } from './Engine';

/** one isolated measured frame: per-pass GPU spans (ms) + wall-clock attribution */
export interface MeasuredFrame {
  /** per-pass GPU timings (ms), keyed exactly as GpuProfiler emits (e.g. c.nanRasterWorld1) */
  passes: Record<string, number>;
  /** wall-clock CPU+GPU span of the isolated frame (submit → onSubmittedWorkDone), ms */
  gpuWallMs: number;
  /** CPU submit span (encode time only, excludes GPU), ms */
  cpuSubmitMs: number;
  /** per-frame counters snapshot (e.g. nanite.visClusters) */
  counters: Record<string, number>;
  /** true if this frame's isolation looked suspect (rejected from honest stats) */
  capSuspect: boolean;
}

export interface MeasureOptions {
  /** number of ISOLATED frames to measure */
  frames: number;
  /** warmup frames driven (drained) before measuring — lets streaming/TRAA settle */
  warmup?: number;
  /** per-frame dt fed to renderStep (sec); default a fixed 1/120 for determinism */
  dt?: number;
  /**
   * idle COOLDOWN (ms) between isolated measured frames. The drained back-to-back
   * loop runs the GPU flat-out with ZERO idle (unlike rAF, which has vsync idle to
   * cool) — so a long run THERMALLY THROTTLES and the late frames inflate. A short
   * cooldown returns the GPU to steady-state thermal between samples WITHOUT
   * reintroducing cross-frame pipelining (the drain already isolated the frame).
   * The artifact this fixes is the dual of the vsync one: isolation kills the
   * pipelining ghost, cooldown kills the thermal drift.
   *
   * CALIBRATION (this machine, apple/metal-3 headless, 40k-tree forest, fixed
   * pose, 8 repeats of a 15-frame median): cooldown 0 → median 27.7 ms / spread
   * 24 ms (throttled, useless); 6 → 24.9 / 7.3; 20 → 19.9 / 5.5 (still cooling);
   * 50 → 18.9 / spread 1.6 (STABLE, reproducible). So 50 ms is the steady-state
   * floor where consecutive samples agree to <2 ms. Default 50 ms.
   */
  cooldownMs?: number;
}

const RAW_DT = 1 / 120;

/** drain the GPU queue: resolves once all submitted work to this point completes */
function drain(device: GPUDevice): Promise<void> {
  return device.queue.onSubmittedWorkDone();
}

/** idle the event loop (and the GPU) for ~ms — a thermal cooldown between samples */
function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

export class MeasureHarness {
  private readonly engine: Engine;
  private readonly device: GPUDevice;
  /** measured wall-clock refresh interval (ms) — the cap to be immune to */
  refreshMs = 1000 / 120;

  private constructor(engine: Engine, device: GPUDevice, refreshMs: number) {
    this.engine = engine;
    this.device = device;
    this.refreshMs = refreshMs;
  }

  /**
   * Build a harness, measuring the display refresh interval first (so cap
   * detection has a real number). Returns null if timestamps/device are
   * unavailable — the caller must then fall back / flag.
   */
  static async create(engine: Engine): Promise<MeasureHarness | null> {
    const device = engine.device;
    if (!device || !engine.gpuProfiler) return null;
    const refreshMs = await measureRefresh();
    return new MeasureHarness(engine, device, refreshMs);
  }

  /**
   * Drive `opts.frames` ISOLATED, GPU-bound frames and return each frame's
   * honest per-pass spans. The rAF loop is stopped for the duration and
   * restored after. Serialize callers — the GPU is a single shared resource.
   */
  async measure(opts: MeasureOptions): Promise<MeasuredFrame[]> {
    const { engine, device } = this;
    const dt = opts.dt ?? RAW_DT;
    const warmup = opts.warmup ?? 8;
    const cooldownMs = opts.cooldownMs ?? 50;
    const profiler = engine.gpuProfiler;
    if (!profiler) return [];

    // freeze the rAF loop so nothing else touches the GPU between our isolated frames
    engine.renderer.setAnimationLoop(null);
    try {
      // warmup: advance + fully drain so streaming/exposure/TRAA history converge
      for (let i = 0; i < warmup; i++) {
        engine.renderStep(dt);
        await drain(device);
      }

      const out: MeasuredFrame[] = [];
      for (let i = 0; i < opts.frames; i++) {
        // thermal cooldown BEFORE the timed frame (skip before the first — warmup
        // already drained) so each sample is taken from the same steady thermal state
        if (cooldownMs > 0) await sleep(cooldownMs);
        // 1. queue is empty (last drain completed) → submit exactly one frame
        await drain(device); // belt-and-braces: ensure truly idle before timing
        const t0 = performance.now();
        engine.renderStep(dt);
        const tSubmit = performance.now();
        // 2. fully complete THIS frame's GPU work in isolation
        await drain(device);
        const tDone = performance.now();
        // 3. resolve the isolated frame's timestamp queries
        await Promise.all([
          engine.renderer.resolveTimestampsAsync(TimestampQuery.RENDER),
          engine.renderer.resolveTimestampsAsync(TimestampQuery.COMPUTE),
        ]);
        const passes: Record<string, number> = {};
        profiler.collect(passes);

        const gpuWallMs = tDone - tSubmit;
        // INTEGRITY GUARD: a per-pass span larger than the whole isolated
        // GPU wall span is physically impossible for active time ⇒ it is a
        // pipelining ghost. Flag the frame as suspect (rejected downstream).
        let maxPass = 0;
        for (const [k, v] of Object.entries(passes)) {
          if ((k === 'render' || k === 'compute') && v > maxPass) maxPass = v;
        }
        // small slack for resolve/clock granularity
        const capSuspect = maxPass > gpuWallMs * 1.5 + 1.0;

        out.push({
          passes,
          gpuWallMs,
          cpuSubmitMs: tSubmit - t0,
          counters: { ...engine.stats.counters },
          capSuspect,
        });
      }
      return out;
    } finally {
      // restore the live loop
      engine.start();
    }
  }
}

/** measure the display refresh interval via rAF deltas (median of a short burst) */
async function measureRefresh(): Promise<number> {
  const samples: number[] = [];
  let last = -1;
  await new Promise<void>((resolve) => {
    const tick = (t: number): void => {
      if (last >= 0) samples.push(t - last);
      last = t;
      if (samples.length < 16) requestAnimationFrame(tick);
      else resolve();
    };
    requestAnimationFrame(tick);
  });
  samples.sort((a, b) => a - b);
  return samples[Math.floor(samples.length / 2)] ?? 1000 / 120;
}
