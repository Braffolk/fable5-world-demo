/**
 * FRESH COST-MAP + STUTTER CHARACTERIZATION (Fable 5, 2026-07-01).
 *
 * Two phases per config, one config per process run (CONFIG=default|noleaves):
 *   A. LIVE moving-camera capture on the real rAF loop — per-tick rAF delta,
 *      cpu.update/cpu.submit, JS heap, key nanite counters, longtasks, full
 *      counter snapshots on spike frames (>33.4ms). This is what the user
 *      experiences (avg ~8ms, p95 ~120ms stutters per report).
 *   B. Isolated GPU cost-map via __laas.measureFrames at the 3 canonical poses
 *      (eye/oblique/aerial) — comparable to the 2026-06 handoff numbers.
 *
 * Canonical: 200k trees, dpr=1.5 (CSS 1512x982 → backing 2268x1473). SERIAL.
 *   CONFIG=default npx tsx tools/probe-fresh-stutter.ts
 *   CONFIG=noleaves npx tsx tools/probe-fresh-stutter.ts
 */
import { launchWebGPU, laasUrl } from './launch';
import { writeFileSync, mkdirSync } from 'node:fs';

const BASE = process.env.BASE ?? 'http://localhost:5173/';
const CONFIG = (process.env.CONFIG ?? 'default') as 'default' | 'noleaves';
const OUT_DIR =
  process.env.OUT_DIR ??
  '/private/tmp/claude-501/-Users-sebastian-IdeaProjects-fable-demo2/cc111c9f-86e4-4c1a-be01-9817f021c312/scratchpad';
const TICKS = Number(process.env.TICKS ?? '2200');
const FRAMES = Number(process.env.FRAMES ?? '32');
const WARMUP = Number(process.env.WARMUP ?? '20');
// per-sample idle cooldown (ms) inside measureFrames — MF_COOLDOWN sweeps the
// DVFS suspicion: 50ms idle may downclock the GPU before each isolated sample
// (harness default 50 when unset).
const MF_COOLDOWN = process.env.MF_COOLDOWN ? Number(process.env.MF_COOLDOWN) : undefined;
// thermal guidance (user): iterate at reduced tree count; 200k only for the
// one-time ground-truth characterization run.
const TREES = process.env.TREES ?? '200000';
const DPR = process.env.DPR ?? '1.5';
// EXTRA='forcevox=1,voxreg=0' — arbitrary URL flags for A/B splits
const EXTRA = process.env.EXTRA ?? '';
// idle seconds before phase B so isolated numbers aren't thermally inflated
// by phase A's flat-out live loop
const COOLDOWN_S = Number(process.env.COOLDOWN_S ?? '45');
const LABEL = process.env.LABEL ?? (process.env.CONFIG ?? 'default');

type Pose = { name: string; p: [number, number, number]; yaw: number; pitch: number };
const POSES: Pose[] = [
  { name: 'eye', p: [0, 2, 0], yaw: 0.6, pitch: -0.02 },
  { name: 'oblique', p: [0, 40, 40], yaw: 0, pitch: -0.35 },
  { name: 'aerial', p: [0, 150, 0], yaw: 0, pitch: -1.45 },
];

const pct = (a: number[], q: number): number => {
  const s = a.filter(Number.isFinite).sort((x, y) => x - y);
  return s.length ? (s[Math.min(s.length - 1, Math.floor(s.length * q))] as number) : NaN;
};

async function main(): Promise<void> {
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({
    viewport: { width: 1512, height: 982 },
    deviceScaleFactor: 1.5,
  });
  const consoleLines: string[] = [];
  page.on('pageerror', (e) => {
    consoleLines.push(`[pageerror] ${e.message}`);
    console.error('[pageerror]', e.message);
  });
  page.on('console', (m) => {
    const t = m.text();
    if (/\[(forest|nanite|laas|vox|fartiles)/i.test(t)) consoleLines.push(t);
  });

  const extra: Record<string, string> = { trees: TREES, nanite: '1', dpr: DPR };
  if (CONFIG === 'noleaves') extra.noleaves = '1';
  for (const kv of EXTRA.split(',').filter(Boolean)) {
    const [k, v] = kv.split('=');
    if (k) extra[k] = v ?? '1';
  }
  const url = laasUrl({ scene: 'forest', freeze: false, hud: false, extra }, BASE);
  console.log(`[probe] CONFIG=${CONFIG} url=${url}`);
  const t0 = Date.now();
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error != null),
    undefined,
    { timeout: 480000, polling: 250 },
  );
  // tsx/esbuild injects __name() helper calls into serialized evaluate bodies;
  // define it in the page so inner named functions don't throw.
  await page.evaluate('globalThis.__name = globalThis.__name || ((f) => f)');
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);
  const bootS = (Date.now() - t0) / 1000;
  console.log(`[probe] boot ${bootS.toFixed(1)}s`);
  await page.evaluate(async () => {
    if (window.__laas.settle) await window.__laas.settle(120);
  });

  // Force a major GC before any capture: ~2GB of dead boot intermediates otherwise
  // sit uncollected for the whole run (live loop allocates too little to trigger V8's
  // major GC), making heapMB bimodal across sessions (5.5GB dirty vs 3.5GB clean —
  // verified by forced-GC forensics 2026-07-02: 5513→3516MB, zero live workers).
  {
    const cdp = await page.context().newCDPSession(page);
    await cdp.send('HeapProfiler.enable');
    await cdp.send('HeapProfiler.collectGarbage');
    await cdp.send('HeapProfiler.collectGarbage');
    await cdp.detach();
    const heapNow = await page.evaluate(
      () => Math.round((performance as unknown as { memory: { usedJSHeapSize: number } }).memory.usedJSHeapSize / 1e6),
    );
    console.log(`[probe] post-boot forced GC — heap ${heapNow}MB`);
  }

  mkdirSync(`${OUT_DIR}/shots`, { recursive: true });
  // static eye shot (visual reference for the voxel-band-too-close bug)
  await page.evaluate(async () => {
    window.__laas.setPose!({ p: [0, 2, 0], yaw: 0.6, pitch: -0.02 });
    if (window.__laas.settle) await window.__laas.settle(30);
  });
  await page.screenshot({ path: `${OUT_DIR}/shots/${LABEL}-eye-static.png` });

  // ── Phase A: live moving-camera stutter capture ─────────────────────────
  console.log(`[probe] phase A: live moving capture (${TICKS} ticks)…`);
  const live = TICKS === 0 ? null : (await page.evaluate(
    async ({ TICKS }) => {
      const h = window.__laas;
      const stats = h.stats!;
      const mem = (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory;
      const longtasks: { t: number; d: number }[] = [];
      try {
        const po = new PerformanceObserver((list) => {
          for (const e of list.getEntries()) longtasks.push({ t: e.startTime, d: e.duration });
        });
        po.observe({ entryTypes: ['longtask'] });
      } catch {
        /* longtask unsupported */
      }
      const delta: number[] = [];
      const cpuUp: number[] = [];
      const cpuSub: number[] = [];
      const heap: number[] = [];
      const visTris: number[] = [];
      const visCl: number[] = [];
      const gpuBuf: number[] = [];
      const tAbs: number[] = [];
      const spikes: { i: number; t: number; delta: number; counters: Record<string, number> }[] = [];
      const p0 = h.getPose!();
      let x = p0.p[0];
      let z = p0.p[2];
      let elapsed = 0;
      await new Promise<void>((resolve) => {
        let last = -1;
        const tick = (t: number): void => {
          if (last >= 0) {
            const d = t - last;
            const dt = Math.min(d / 1000, 0.1);
            elapsed += dt;
            // eye-level glide: forward 8 m/s along a slowly sweeping yaw
            const yaw = 0.6 + 0.35 * Math.sin(elapsed * 0.35);
            x += Math.sin(yaw) * 8 * dt;
            z += Math.cos(yaw) * 8 * dt;
            let y = 2;
            if (h.groundProbe) {
              try {
                y = h.groundProbe(x, z).ground + 1.8;
              } catch {
                y = 2;
              }
            }
            h.setPose!({ p: [x, y, z], yaw, pitch: -0.02 });
            const c = stats.counters;
            delta.push(Math.round(d * 100) / 100);
            tAbs.push(Math.round(t));
            cpuUp.push(c['cpu.updateMs100'] ?? -1);
            cpuSub.push(c['cpu.submitMs100'] ?? -1);
            heap.push(mem ? Math.round(mem.usedJSHeapSize / 1e5) / 10 : -1);
            visTris.push(c['nanite.visTris'] ?? -1);
            visCl.push(c['nanite.visClusters'] ?? -1);
            gpuBuf.push(c['gpu.buffers'] ?? -1);
            if (d > 33.4 && spikes.length < 80) {
              spikes.push({ i: delta.length - 1, t, delta: d, counters: { ...c } });
            }
          }
          last = t;
          if (delta.length < TICKS && t < (tAbs[0] ?? t) + 60000) requestAnimationFrame(tick);
          else resolve();
        };
        requestAnimationFrame(tick);
      });
      return { delta, tAbs, cpuUp, cpuSub, heap, visTris, visCl, gpuBuf, spikes, longtasks, endPose: h.getPose!() };
    },
    { TICKS },
  )) as {
    delta: number[];
    tAbs: number[];
    cpuUp: number[];
    cpuSub: number[];
    heap: number[];
    visTris: number[];
    visCl: number[];
    gpuBuf: number[];
    spikes: { i: number; t: number; delta: number; counters: Record<string, number> }[];
    longtasks: { t: number; d: number }[];
    endPose: unknown;
  };
  await page.screenshot({ path: `${OUT_DIR}/shots/${LABEL}-end-of-motion.png` });

  // ── Phase B: isolated GPU cost-map at canonical poses ───────────────────
  if (COOLDOWN_S > 0) {
    console.log(`[probe] cooling ${COOLDOWN_S}s before phase B…`);
    await new Promise((r) => setTimeout(r, COOLDOWN_S * 1000));
  }
  console.log('[probe] phase B: isolated measureFrames at canonical poses…');
  const posesOut: Record<
    string,
    {
      gpu: number[];
      cpuSubmit: number[];
      capRejects: number;
      counters: Record<string, number>;
      frameCounters: Record<string, number>[];
    }
  > = {};
  for (const pose of POSES) {
    const frames = (await page.evaluate(
      async ({ pose, FRAMES, WARMUP, MF_COOLDOWN }) => {
        window.__laas.setPose!({ p: pose.p, yaw: pose.yaw, pitch: pose.pitch });
        if (window.__laas.settle) await window.__laas.settle(20);
        const fs = await window.__laas.measureFrames!({
          frames: FRAMES,
          warmup: WARMUP,
          ...(MF_COOLDOWN !== undefined ? { cooldownMs: MF_COOLDOWN } : {}),
        });
        return fs.map((f) => ({
          gpu: f.gpuWallMs,
          cpu: f.cpuSubmitMs,
          cap: f.capSuspect ? 1 : 0,
          c: f.counters,
        }));
      },
      { pose, FRAMES, WARMUP, MF_COOLDOWN },
    )) as { gpu: number; cpu: number; cap: number; c: Record<string, number> }[];
    const good = frames.filter((f) => f.cap === 0);
    const use = good.length >= frames.length / 2 ? good : frames;
    // measure-infra 4c: capSuspect is now the event-loop-lag guard (rare) — if the
    // >half fallback still fires, something is systematically stalling the loop.
    if (use === frames && good.length < frames.length)
      console.warn(
        `  [${pose.name}] WARNING: ${frames.length - good.length}/${frames.length} frames ` +
          `capSuspect — outlier filter bypassed (event-loop stalls during drain)`,
      );
    posesOut[pose.name] = {
      gpu: use.map((f) => f.gpu),
      cpuSubmit: use.map((f) => f.cpu),
      capRejects: frames.length - good.length,
      counters: frames[frames.length - 1]!.c,
      // per-frame counter series (measure-infra 4b) — the bimodality discriminator:
      // fast-vs-slow frames with EQUAL visTris/visClusters ⇒ machine state (DVFS),
      // unequal ⇒ jitter-phase-dependent workload (cull/voxprev routing).
      frameCounters: use.map((f) => f.c),
    };
    console.log(
      `  [${pose.name}] gpuWall med=${pct(posesOut[pose.name]!.gpu, 0.5).toFixed(2)} ` +
        `p95=${pct(posesOut[pose.name]!.gpu, 0.95).toFixed(2)} cpuSubmit med=${pct(posesOut[pose.name]!.cpuSubmit, 0.5).toFixed(2)} ` +
        `capRejects=${frames.length - good.length}/${frames.length}`,
    );
    await page.screenshot({ path: `${OUT_DIR}/shots/${LABEL}-${pose.name}.png` });
  }

  const out = { config: CONFIG, label: LABEL, extra, bootS, live, poses: posesOut, consoleLines };
  const path = `${OUT_DIR}/fresh-${LABEL}.json`;
  writeFileSync(path, JSON.stringify(out));
  console.log(`[probe] wrote ${path}`);

  // ── human summary ────────────────────────────────────────────────────────
  if (!live) {
    await browser.close();
    return;
  }
  const d = live.delta;
  console.log(`\n===== LIVE MOVING (${CONFIG}) — ${d.length} ticks =====`);
  console.log(
    `delta ms: avg=${(d.reduce((a, b) => a + b, 0) / d.length).toFixed(2)} p50=${pct(d, 0.5).toFixed(1)} ` +
      `p90=${pct(d, 0.9).toFixed(1)} p95=${pct(d, 0.95).toFixed(1)} p99=${pct(d, 0.99).toFixed(1)} max=${pct(d, 1).toFixed(1)}`,
  );
  console.log(
    `frames >16.7ms: ${d.filter((x) => x > 16.7).length}  >33ms: ${d.filter((x) => x > 33.4).length}  >100ms: ${d.filter((x) => x > 100).length}`,
  );
  console.log(`longtasks: ${live.longtasks.length} (total ${live.longtasks.reduce((a, b) => a + b.d, 0).toFixed(0)}ms)`);
  console.log(`heapMB first/last: ${live.heap[0]} / ${live.heap[live.heap.length - 1]}`);
  for (const s of live.spikes.slice(0, 25)) {
    const lt = live.longtasks.find((l) => l.t + l.d > s.t - s.delta && l.t < s.t);
    console.log(
      `  spike i=${s.i} ${s.delta.toFixed(1)}ms cpuUp=${((s.counters['cpu.updateMs100'] ?? 0) / 100).toFixed(1)} ` +
        `cpuSub=${((s.counters['cpu.submitMs100'] ?? 0) / 100).toFixed(1)} ` +
        `visTris=${s.counters['nanite.visTris'] ?? -1} longtask=${lt ? `${lt.d.toFixed(0)}ms` : 'none'}`,
    );
  }
  await browser.close();
}

main().catch((e) => {
  console.error('[probe] FAILED:', e instanceof Error ? (e.stack ?? e.message) : e);
  process.exit(1);
});
