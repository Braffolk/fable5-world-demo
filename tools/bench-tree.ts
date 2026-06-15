/**
 * bench-tree.ts — geometry-vs-renderer experiment harness.
 *
 * Loads reference/ref-tree.html (a copy of the three.js compute-rasterizer
 * example with ONLY the source geometry swapped) in headless WebGPU, for each
 * of: helmet (baseline control), tree (our FULL tree: trunk + leaf crown),
 * crown (our volumetric leaf needles), bark (our trunk manifold control).
 * Warms up, then samples the on-screen rAF frame time (median over a window),
 * captures the per-LOD tri counts and a screenshot, and prints a table.
 *
 * Run with vsync uncapped so the rAF loop runs as fast as the GPU allows.
 * Requires the vite dev server on :5173 (npm run dev).
 *
 * Usage: npx tsx tools/bench-tree.ts            (all geos)
 *        npx tsx tools/bench-tree.ts tree helmet
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { chromium, type Browser, type Page } from 'playwright';

const BASE = 'http://localhost:5173';
const URL = `${BASE}/reference/ref-tree.html`;
const W = 1600;
const H = 900;
const WARMUP_MS = 4000;   // let shaders compile + the scene settle
const SAMPLE_MS = 4000;   // collect frame-time samples over this window

// vsync uncapped — otherwise headless Chromium clamps rAF to the display refresh.
const FLAGS = [
  '--enable-unsafe-webgpu',
  '--disable-gpu-vsync',
  '--disable-frame-rate-limit',
  '--disable-features=CalculateNativeWinOcclusion',
];

interface LodInfo { lod: number; tris: number; verts: number; error: number }
interface Result {
  geo: string;
  perLod: LodInfo[];
  lodSumTris: number;
  lod0Tris: number;
  medianMs: number;
  medianFps: number;
  p95Ms: number;
  samples: number;
  boundingRadius: number;
  ok: boolean;
  error?: string;
}

async function launch(): Promise<Browser> {
  const browser = await chromium.launch({ headless: true, channel: 'chromium', args: FLAGS });
  const page = await browser.newPage();
  await page.goto(`${BASE}/`, { waitUntil: 'domcontentloaded' }).catch(() => undefined);
  const ok = await page.evaluate(async () => {
    const gpu = (navigator as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
    if (!gpu) return false;
    return (await gpu.requestAdapter()) !== null;
  });
  await page.close();
  if (!ok) {
    await browser.close();
    throw new Error('No WebGPU adapter under channel:chromium headless — is the dev server up on :5173?');
  }
  return browser;
}

async function benchOne(browser: Browser, geo: string, outDir: string): Promise<Result> {
  const page: Page = await browser.newPage({ viewport: { width: W, height: H } });
  const errors: string[] = [];
  page.on('console', (m) => {
    const t = m.text();
    if (m.type() === 'error' && !t.includes('Failed to load resource')) errors.push(t);
    if (t.includes('[ref-tree]')) console.log(`    ${t}`);
  });
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));

  const res: Result = {
    geo, perLod: [], lodSumTris: 0, lod0Tris: 0,
    medianMs: 0, medianFps: 0, p95Ms: 0, samples: 0, boundingRadius: 0, ok: false,
  };

  try {
    await page.goto(`${URL}?geo=${geo}`, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.waitForFunction(() => (window as any).__refLods !== undefined, { timeout: 60000 });
    const lods = await page.evaluate(() => (window as any).__refLods);
    res.perLod = lods.perLod;
    res.lodSumTris = lods.lodSumTris;
    res.lod0Tris = lods.perLod[0].tris;
    res.boundingRadius = lods.boundingRadiusLod0;

    // warm up, then sample raw per-frame deltas over a window (cool, GPU-uncapped)
    await page.waitForTimeout(WARMUP_MS);
    const frames: number[] = await page.evaluate(async (durMs) => {
      const samples: number[] = [];
      let last = performance.now();
      const end = last + durMs;
      return await new Promise<number[]>((resolve) => {
        const tick = () => {
          const now = performance.now();
          samples.push(now - last);
          last = now;
          if (now < end) requestAnimationFrame(tick);
          else resolve(samples);
        };
        requestAnimationFrame(tick);
      });
    }, SAMPLE_MS);

    const sorted = frames.slice().sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)] || 0;
    const p95 = sorted[Math.floor(sorted.length * 0.95)] || 0;
    res.medianMs = median;
    res.medianFps = median > 0 ? 1000 / median : 0;
    res.p95Ms = p95;
    res.samples = frames.length;

    const shot = `${outDir}/ref-${geo}.png`;
    await page.screenshot({ path: shot });
    res.ok = errors.length === 0;
    if (errors.length) res.error = errors.slice(0, 3).join(' | ');
    console.log(
      `  ${geo.padEnd(8)} LOD0=${res.lod0Tris}t perLOD=[${res.perLod.map((l) => l.tris).join(',')}]  ` +
        `median=${res.medianMs.toFixed(2)}ms (${res.medianFps.toFixed(0)} fps)  p95=${res.p95Ms.toFixed(2)}ms  ` +
        `${res.ok ? 'OK' : 'ERR: ' + res.error}  -> ${shot}`,
    );
  } catch (e) {
    res.error = `${(e as Error).message}${errors.length ? ' | ' + errors.slice(0, 2).join(' | ') : ''}`;
    console.log(`  ${geo.padEnd(8)} FAILED: ${res.error}`);
    try { await page.screenshot({ path: `${outDir}/ref-${geo}-FAIL.png` }); } catch { /* */ }
  } finally {
    await page.close();
  }
  return res;
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2).filter((a) => !a.startsWith('--'));
  const geos = argv.length ? argv : ['helmet', 'tree', 'crown', 'bark'];
  const outDir = 'tools/geo/shots';
  mkdirSync(outDir, { recursive: true });

  console.log(`[bench] viewport ${W}x${H}, warmup ${WARMUP_MS}ms, sample ${SAMPLE_MS}ms, geos: ${geos.join(', ')}`);
  const browser = await launch();
  const results: Result[] = [];
  try {
    for (const geo of geos) results.push(await benchOne(browser, geo, outDir));
  } finally {
    await browser.close();
  }

  console.log('\n================ RESULTS (forest-interior camera, 129600 instances) ================');
  const helmet = results.find((r) => r.geo === 'helmet');
  for (const r of results) {
    const rel = helmet && helmet.medianMs > 0 && r.medianMs > 0
      ? ` (${(r.medianMs / helmet.medianMs).toFixed(2)}x helmet)` : '';
    console.log(
      `${r.geo.padEnd(8)} | LOD0 ${String(r.lod0Tris).padStart(6)} t | ` +
        `LODs [${r.perLod.map((l) => l.tris).join(', ')}] | ` +
        `${r.medianMs.toFixed(2)} ms  ${r.medianFps.toFixed(0)} fps${rel} | ${r.ok ? 'ok' : 'ERR'}`,
    );
  }
  writeFileSync(`${outDir}/results.json`, JSON.stringify(results, null, 2));
  console.log(`\n[bench] results.json + screenshots in ${outDir}/`);
}

main().catch((e) => { console.error(e); process.exit(1); });
