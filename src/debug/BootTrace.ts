/**
 * BootTrace — tiny boot-phase timer + main-thread-stall estimator (cold-boot
 * work, 2026-07-04). `phase(name)` closes the previous phase and opens the
 * next (the boot is a flat serial timeline); `span(name)` tracks CONCURRENT
 * work (worker fan-outs, overlapped builds) without disturbing the phase
 * timeline; `finish()` prints one console.table at first frame and exposes
 * `window.__bootTrace` for tooling (bootbench).
 *
 * Stall estimation: a self-rescheduling 50 ms setTimeout heartbeat — any
 * main-thread task longer than the interval shows up as a beat gap, which is
 * attributed to the phase active when the beat finally fires. Cheap (~20
 * beats/s), no PerformanceObserver dependency.
 *
 * `yieldIfDue()` is the shared cooperative-yield helper for long boot loops:
 * a macrotask hop (scheduler.yield when available) every ~250 ms of work —
 * the time-check pattern, so hot loops call it per item at near-zero cost.
 */

interface PhaseRec {
  name: string;
  t0: number;
  ms: number;
  worstStallMs: number;
}
interface SpanRec {
  name: string;
  t0: number;
  ms: number;
}

const HB_MS = 50;
const t0Boot = performance.now();
const phases: PhaseRec[] = [];
const spans: SpanRec[] = [];
let cur: PhaseRec | null = null;
let hbLast = 0;
let hbTimer: ReturnType<typeof setTimeout> | null = null;
let worstStallBoot = 0;

function beat(): void {
  const now = performance.now();
  const gap = now - hbLast - HB_MS;
  if (gap > 0) {
    if (cur && gap > cur.worstStallMs) cur.worstStallMs = gap;
    if (gap > worstStallBoot) worstStallBoot = gap;
  }
  hbLast = now;
  hbTimer = setTimeout(beat, HB_MS);
}

export const BootTrace = {
  /** close the previous phase, open the next (starts the heartbeat on first use) */
  phase(name: string): void {
    const now = performance.now();
    if (!hbTimer) {
      hbLast = now;
      hbTimer = setTimeout(beat, HB_MS);
    }
    if (cur) cur.ms = now - cur.t0;
    cur = { name, t0: now, ms: 0, worstStallMs: 0 };
    phases.push(cur);
  },

  /** concurrent work: `const end = BootTrace.span('crown workers'); …; end();` */
  span(name: string): () => void {
    const rec: SpanRec = { name, t0: performance.now(), ms: 0 };
    spans.push(rec);
    return (): void => {
      rec.ms = performance.now() - rec.t0;
    };
  },

  /** close out + print the summary table (call once, at/near the first frame) */
  finish(): void {
    const now = performance.now();
    if (cur) {
      cur.ms = now - cur.t0;
      cur = null;
    }
    if (hbTimer) {
      clearTimeout(hbTimer);
      hbTimer = null;
    }
    if (phases.length === 0 && spans.length === 0) return;
    const rows = [
      ...phases.map((p) => ({
        phase: p.name,
        at_s: Number(((p.t0 - t0Boot) / 1000).toFixed(1)),
        ms: Math.round(p.ms),
        worstStallMs: Math.round(p.worstStallMs),
      })),
      // ∥ rows run CONCURRENTLY with the phases above (not part of the serial sum)
      ...spans.map((s) => ({
        phase: `= ${s.name}`,
        at_s: Number(((s.t0 - t0Boot) / 1000).toFixed(1)),
        ms: Math.round(s.ms),
        worstStallMs: 0,
      })),
    ];
    // eslint-disable-next-line no-console
    console.table(rows);
    // eslint-disable-next-line no-console
    console.log(
      `[boottrace] total ${((now - t0Boot) / 1000).toFixed(1)} s since module load, ` +
        `worst main-thread stall ${Math.round(worstStallBoot)} ms`,
    );
    (window as unknown as { __bootTrace?: unknown }).__bootTrace = {
      rows,
      totalMs: now - t0Boot,
      worstStallMs: worstStallBoot,
    };
    finishResolve();
  },

  /** resolves when finish() runs (≈ first frame) — deferred best-effort work
   *  (BootCache stores) waits on this so its main-thread serialization cost
   *  never competes with the boot itself. Safety-resolved after 3 min so a
   *  scene that never calls finish() still gets its cache writes. */
  whenFinished(): Promise<void> {
    if (!finishTimer) finishTimer = setTimeout(finishResolve, 180_000);
    return finishPromise;
  },
};

let finishResolve!: () => void;
const finishPromise = new Promise<void>((r) => {
  finishResolve = r;
});
let finishTimer: ReturnType<typeof setTimeout> | null = null;

// ── cooperative yield ─────────────────────────────────────────────────────────

let lastYield = performance.now();

/**
 * Yield the main thread if ~budgetMs of work has run since the last yield —
 * call per item inside long boot loops (near-free when not due). Keeps every
 * boot slab under the frozen-tab threshold even where workers don't apply.
 * Deliberately a setTimeout macrotask, NOT scheduler.yield: the prioritized
 * continuation starves other timers (incl. the stall heartbeat above) for the
 * whole loop — a plain timer keeps timer-queue fairness during boot.
 */
export async function yieldIfDue(budgetMs = 250): Promise<void> {
  const now = performance.now();
  if (now - lastYield < budgetMs) return;
  await new Promise<void>((r) => setTimeout(r, 0));
  lastYield = performance.now();
}
