#!/usr/bin/env node
/**
 * interleaved_ab.mjs — throttle-resistant A/B frame-time comparison.
 *
 * The M1 Max thermally drifts during perf work (same build + pose measured
 * 30→51ms p95 across sequential runs). Sequential A…A B…B comparisons are
 * therefore confounded by drift. This harness INTERLEAVES the arms
 * (A,B,A,B,…) so slow thermal drift is shared by both, and reports per-arm
 * medians of p95 plus per-cycle deltas (B−A within the same cycle), which
 * cancel drift to first order.
 *
 * Usage:
 *   node tools/perf/interleaved_ab.mjs --base "<urlA>" --alt "<urlB>" \
 *     [--cycles 3] [--warmup 8] [--sample 10] [--port 5173] [--headed]
 *
 * Frame times are collected IN-PAGE via a requestAnimationFrame delta
 * sampler (per-frame ms), percentiles computed in Node. Thermal state
 * (`pmset -g therm` → CPU_Speed_Limit) is recorded around every reading;
 * readings taken under a speed limit <100 are flagged.
 *
 * Requires the Vite dev server on --port (default 5173). WebGPU needs a
 * secure context + real-GPU headless: chromium channel:'chromium' headless
 * works on this machine (falls back per tools/launch.ts findings); pass
 * --headed if headless WebGPU fails.
 *
 * Output: table on stdout + JSON at tools/perf/last_ab.json.
 */

import { execFileSync } from 'node:child_process';
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const HERE = dirname(fileURLToPath(import.meta.url));
const JSON_OUT = join(HERE, 'last_ab.json');

// ---------------------------------------------------------------- CLI args
function parseArgs(argv) {
  const out = {
    base: null,
    alt: null,
    cycles: 3,
    warmup: 8,
    sample: 10,
    port: 5173,
    headed: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--headed') out.headed = true;
    else if (a.startsWith('--')) {
      const key = a.slice(2);
      const val = argv[++i];
      if (val === undefined) fail(`missing value for --${key}`);
      if (key === 'base' || key === 'alt') out[key] = val;
      else if (key === 'cycles' || key === 'warmup' || key === 'sample' || key === 'port') {
        const n = Number(val);
        if (!Number.isFinite(n) || n <= 0) fail(`--${key} must be a positive number, got "${val}"`);
        out[key] = n;
      } else fail(`unknown flag --${key}`);
    } else fail(`unexpected argument "${a}"`);
  }
  if (!out.base || !out.alt) {
    fail('--base "<urlA>" and --alt "<urlB>" are required');
  }
  return out;
}

function fail(msg) {
  console.error(`[interleaved_ab] ERROR: ${msg}`);
  console.error(
    'usage: node tools/perf/interleaved_ab.mjs --base "<urlA>" --alt "<urlB>" ' +
      '[--cycles 3] [--warmup 8] [--sample 10] [--port 5173] [--headed]',
  );
  process.exit(1);
}

// ---------------------------------------------------------------- helpers
function percentile(sorted, p) {
  if (sorted.length === 0) return NaN;
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function median(values) {
  const s = [...values].sort((a, b) => a - b);
  return percentile(s, 0.5);
}

/** `pmset -g therm` → { limit, throttled }. Missing CPU_Speed_Limit = 100. */
function readTherm() {
  try {
    const out = execFileSync('pmset', ['-g', 'therm'], { encoding: 'utf8', timeout: 5000 });
    const m = out.match(/CPU_Speed_Limit\s*=\s*(\d+)/);
    const limit = m ? Number(m[1]) : 100;
    return { limit, throttled: limit < 100, at: new Date().toISOString() };
  } catch (err) {
    return { limit: null, throttled: false, at: new Date().toISOString(), error: String(err) };
  }
}

async function checkDevServer(port) {
  try {
    const res = await fetch(`http://localhost:${port}/`, { signal: AbortSignal.timeout(5000) });
    return res.ok || res.status < 500;
  } catch {
    return false;
  }
}

// ------------------------------------------------------------ browser side
/**
 * Launch a WebGPU-capable Chromium. Recipe from tools/launch.ts (empirical on
 * this machine): channel:'chromium' new-headless gets a real Metal adapter;
 * Playwright's default headless shell does NOT. Secure context required —
 * probe against the dev server origin, never about:blank.
 */
async function launchBrowser(port, headed) {
  const recipes = headed
    ? [{ headless: false, args: [] }]
    : [
        { headless: true, channel: 'chromium', args: [] },
        { headless: true, channel: 'chromium', args: ['--enable-unsafe-webgpu'] },
      ];
  for (const recipe of recipes) {
    let browser = null;
    try {
      const opts = { headless: recipe.headless, args: recipe.args };
      if (recipe.channel) opts.channel = recipe.channel;
      browser = await chromium.launch(opts);
      const page = await browser.newPage();
      await page.goto(`http://localhost:${port}/__webgpu_probe__`, {
        waitUntil: 'domcontentloaded',
      });
      const ok = await page.evaluate(async () => {
        const gpu = navigator.gpu;
        if (!gpu) return false;
        const adapter = await gpu.requestAdapter();
        return adapter !== null;
      });
      await page.close();
      if (ok) {
        console.log(
          `[launch] WebGPU OK — headless=${recipe.headless} channel=${recipe.channel ?? 'default'}`,
        );
        return browser;
      }
      await browser.close();
    } catch {
      if (browser) await browser.close().catch(() => {});
    }
  }
  throw new Error(
    'no launch recipe produced a WebGPU adapter' +
      (headed ? '' : ' — retry with --headed (headless WebGPU can fail on some setups)'),
  );
}

/**
 * One reading: navigate, wait ready, warm up, sample per-frame times via an
 * in-page rAF delta collector. Returns { p50, p95, frames, meanFps }.
 */
async function takeReading(page, url, warmupS, sampleS) {
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  // NOTE playwright gotcha: waitForFunction(fn, arg, OPTIONS) — options is 3rd.
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready === true || window.__laas.error !== null),
    undefined,
    { timeout: 300_000 },
  );
  const err = await page.evaluate(() => window.__laas.error);
  if (err) throw new Error(`page reported fatal error: ${err}`);

  await page.waitForTimeout(warmupS * 1000);

  const deltas = await page.evaluate(
    (ms) =>
      new Promise((resolve) => {
        const out = [];
        let prev = -1;
        let start = -1;
        function loop(t) {
          if (prev >= 0) out.push(t - prev);
          else start = t;
          prev = t;
          if (t - start < ms) requestAnimationFrame(loop);
          else resolve(out);
        }
        requestAnimationFrame(loop);
      }),
    sampleS * 1000,
  );
  if (!Array.isArray(deltas) || deltas.length < 5) {
    throw new Error(`too few frame samples (${deltas?.length ?? 0}) — page stalled?`);
  }
  const sorted = [...deltas].sort((a, b) => a - b);
  return {
    p50: percentile(sorted, 0.5),
    p95: percentile(sorted, 0.95),
    frames: deltas.length,
    meanFps: 1000 / (deltas.reduce((a, b) => a + b, 0) / deltas.length),
  };
}

// ------------------------------------------------------------------- main
async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (!(await checkDevServer(args.port))) {
    fail(
      `dev server not responding on http://localhost:${args.port} — start it with \`npm run dev\``,
    );
  }

  const browser = await launchBrowser(args.port, args.headed);
  const context = await browser.newContext({ viewport: { width: 1600, height: 900 } });
  let page = await context.newPage();

  const arms = [
    { name: 'A', url: args.base },
    { name: 'B', url: args.alt },
  ];
  const readings = []; // { index, cycle, arm, url, p50, p95, frames, therm, failed }

  console.log(
    `[interleaved_ab] ${args.cycles} cycle(s) × 2 arms — warmup ${args.warmup}s, sample ${args.sample}s each`,
  );
  console.log(`  A = ${args.base}`);
  console.log(`  B = ${args.alt}`);

  let index = 0;
  for (let cycle = 0; cycle < args.cycles; cycle++) {
    for (const arm of arms) {
      const thermBefore = readTherm();
      let result = null;
      let error = null;
      for (let attempt = 0; attempt < 2 && !result; attempt++) {
        try {
          if (page.isClosed()) page = await context.newPage();
          result = await takeReading(page, arm.url, args.warmup, args.sample);
        } catch (e) {
          error = e;
          console.error(
            `  [cycle ${cycle} arm ${arm.name}] attempt ${attempt + 1} failed: ${e.message}`,
          );
          // recycle the page — GPU loss / crash leaves it unusable
          await page.close().catch(() => {});
          page = await context.newPage();
        }
      }
      const thermAfter = readTherm();
      const throttled =
        thermBefore.throttled || thermAfter.throttled
          ? Math.min(thermBefore.limit ?? 100, thermAfter.limit ?? 100)
          : null;
      const reading = {
        index: index++,
        cycle,
        arm: arm.name,
        url: arm.url,
        failed: !result,
        p50: result?.p50 ?? null,
        p95: result?.p95 ?? null,
        frames: result?.frames ?? 0,
        meanFps: result?.meanFps ?? null,
        thermBefore,
        thermAfter,
        throttledAt: throttled,
        error: result ? null : String(error?.message ?? error),
      };
      readings.push(reading);
      const flag = reading.failed
        ? 'FAILED'
        : throttled !== null
          ? `THROTTLED(${throttled})`
          : 'ok';
      console.log(
        `  [${reading.index}] cycle ${cycle} arm ${arm.name}: ` +
          (reading.failed
            ? `FAILED (${reading.error})`
            : `p50 ${reading.p50.toFixed(2)}ms  p95 ${reading.p95.toFixed(2)}ms  (${reading.frames} frames, ${flag})`),
      );
    }
  }

  await browser.close();

  // ------------------------------------------------------------- summary
  const ok = readings.filter((r) => !r.failed);
  const a = ok.filter((r) => r.arm === 'A');
  const b = ok.filter((r) => r.arm === 'B');
  const deltas = [];
  for (let c = 0; c < args.cycles; c++) {
    const ra = readings.find((r) => r.cycle === c && r.arm === 'A' && !r.failed);
    const rb = readings.find((r) => r.cycle === c && r.arm === 'B' && !r.failed);
    if (ra && rb) {
      deltas.push({
        cycle: c,
        p95Delta: rb.p95 - ra.p95,
        p50Delta: rb.p50 - ra.p50,
        throttled: ra.throttledAt !== null || rb.throttledAt !== null,
      });
    }
  }

  console.log('');
  console.log('idx  cycle  arm  p50(ms)   p95(ms)   thermal');
  for (const r of readings) {
    const therm = r.throttledAt !== null ? `THROTTLED(${r.throttledAt})` : 'ok';
    if (r.failed) console.log(`${String(r.index).padEnd(5)}${String(r.cycle).padEnd(7)}${r.arm}    —         —         FAILED`);
    else
      console.log(
        `${String(r.index).padEnd(5)}${String(r.cycle).padEnd(7)}${r.arm}    ` +
          `${r.p50.toFixed(2).padEnd(10)}${r.p95.toFixed(2).padEnd(10)}${therm}`,
      );
  }

  console.log('');
  const medA = a.length ? median(a.map((r) => r.p95)) : NaN;
  const medB = b.length ? median(b.map((r) => r.p95)) : NaN;
  const medDelta = deltas.length ? median(deltas.map((d) => d.p95Delta)) : NaN;
  console.log(`arm A: median p95 = ${medA.toFixed(2)}ms   (${a.length}/${args.cycles} readings)`);
  console.log(`arm B: median p95 = ${medB.toFixed(2)}ms   (${b.length}/${args.cycles} readings)`);
  console.log(
    `per-cycle p95 delta (B−A): ${deltas.map((d) => `${d.p95Delta >= 0 ? '+' : ''}${d.p95Delta.toFixed(2)}`).join(', ')} ms` +
      `  → median ${medDelta >= 0 ? '+' : ''}${medDelta.toFixed(2)}ms`,
  );

  let verdict;
  if (!deltas.length) verdict = 'NO VERDICT — no complete cycles';
  else {
    const pct = (medDelta / medA) * 100;
    const consistent = deltas.every((d) => Math.sign(d.p95Delta) === Math.sign(medDelta));
    const dir = medDelta > 0 ? 'SLOWER' : 'FASTER';
    if (Math.abs(pct) < 3) verdict = `WASH — B within ±3% of A on median per-cycle p95 delta (${pct.toFixed(1)}%)`;
    else
      verdict =
        `B is ${dir} than A by ${Math.abs(medDelta).toFixed(2)}ms p95 (${Math.abs(pct).toFixed(1)}%)` +
        (consistent ? ', consistent sign across all cycles' : ' — ⚠️ sign flips across cycles, treat as noisy');
  }
  const anyThrottled = readings.some((r) => r.throttledAt !== null);
  console.log(`VERDICT: ${verdict}`);
  if (anyThrottled)
    console.log('⚠️  some readings ran under CPU speed limit <100 — interleaving shares the drift, but absolute numbers are depressed.');

  mkdirSync(HERE, { recursive: true });
  writeFileSync(
    JSON_OUT,
    JSON.stringify(
      {
        at: new Date().toISOString(),
        args,
        readings,
        perCycleDeltas: deltas,
        summary: { medianP95A: medA, medianP95B: medB, medianP95Delta: medDelta, verdict, anyThrottled },
      },
      null,
      2,
    ),
  );
  console.log(`[interleaved_ab] JSON written to ${JSON_OUT}`);
}

main().catch((e) => {
  console.error(`[interleaved_ab] FATAL: ${e?.stack ?? e}`);
  process.exit(1);
});
