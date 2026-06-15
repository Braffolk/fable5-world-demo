/**
 * N8-HIC — the decisive cull-vs-raster split. Boots the flood frame (shot 7 forest
 * interior, `?naniteleaf=1`, occl ON = the default path) and dumps the per-pass GPU
 * ledger (`stats.gpuPasses`) sorted. Decides the architecture:
 *
 *   nanClusterCull (the TEST cost: flat cut tests every cluster of every DAG) dominates
 *     ⇒ D-N31's deferred HIERARCHICAL CUT TRAVERSAL (general, whole-geometry) is the fix.
 *   nanRasterDepth/Payload/HW (the EMIT cost: rastering 1.17M visible clusters) dominates
 *     ⇒ the per-instance floor must be broken by CROSS-INSTANCE aggregation (fewer emitted).
 *
 *   LEAFDENSITY=800 npx tsx tools/probe-hicsplit.ts
 */

import { launchWebGPU, laasUrl } from './launch';

const SHOT = process.env.SHOT ?? '7';
const LEAFDENSITY = process.env.LEAFDENSITY ?? '800';

async function main(): Promise<void> {
  const width = 1280;
  const height = 720;
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));
  const url = laasUrl({
    scene: 'world',
    width,
    height,
    freeze: false,
    extra: {
      nanite: '1',
      naniteleaf: '1',
      naniteleafdensity: LEAFDENSITY,
      nanitedterrain: '0',
      nanshadow: '0',
      occl: '1',
      shot: SHOT,
      loderr: '1',
    },
  });
  console.log(`[hicsplit] SHOT=${SHOT} LEAFDENSITY=${LEAFDENSITY} occl ON`);
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error != null),
    undefined,
    { timeout: 240000, polling: 250 },
  );
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);
  await page.evaluate(() => window.__laasNanite?.setTau?.(1));

  // median per-pass over many settles (timestamps are noisy)
  const acc = new Map<string, number[]>();
  let frameMs = 0;
  let visClusters = 0;
  let chunks = 0;
  for (let i = 0; i < 40; i++) {
    const f = (await page.evaluate(async () => {
      if (window.__laas.settle) await window.__laas.settle(6);
      const st = window.__laas.stats;
      return {
        g: st?.gpuPasses ?? {},
        frameMs: st?.frameMs ?? 0,
        vis: st?.counters?.['nanite.visClusters'] ?? -1,
        chunks: st?.counters?.['nanite.chunks'] ?? -1,
      };
    })) as { g: Record<string, number>; frameMs: number; vis: number; chunks: number };
    if (i >= 10) {
      for (const [k, v] of Object.entries(f.g)) {
        if (!acc.has(k)) acc.set(k, []);
        acc.get(k)?.push(v);
      }
      frameMs = f.frameMs;
      visClusters = f.vis;
      chunks = f.chunks;
    }
  }
  const median = (a: number[]): number => {
    const s = [...a].sort((x, y) => x - y);
    return s[Math.floor(s.length / 2)] ?? 0;
  };
  const rows = [...acc.entries()]
    .map(([k, a]) => [k, median(a)] as [string, number])
    .filter(([, m]) => m >= 0.03)
    .sort((a, b) => b[1] - a[1]);
  console.log(`[hicsplit] frame ${frameMs.toFixed(1)}ms | visClusters ${visClusters} | chunks ${chunks}`);
  console.log('[hicsplit] per-pass GPU ledger (median ms, ≥0.03):');
  for (const [k, m] of rows) console.log(`  ${m.toFixed(2).padStart(7)} ms  ${k}`);
  await browser.close();
}

main().catch((e) => {
  console.error('[hicsplit] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
