/**
 * Smoke + perf for the forest scene's NEW full-frame pipe (NaniteFrame resolve + post on
 * isolated trees). Boots ?scene=forest&nanite=1 (no nanitedbg ⇒ the standard whole pipe),
 * warms, screenshots, and reports the world1 raster + scene/post GPU pass times.
 *
 *   TREES=40000 npx tsx tools/probe-forestfull.ts
 */
import { laasUrl, launchWebGPU } from './launch';

const W = 1280;
const H = 720;
const TREES = process.env.TREES ?? '40000';
const median = (a: number[]): number => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)] ?? 0;

async function main(): Promise<void> {
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));
  page.on('console', (m) => {
    const t = m.text();
    if (t.startsWith('[forest]')) console.log(`   · ${t}`);
  });
  const url = laasUrl({
    scene: 'forest',
    width: W,
    height: H,
    freeze: false,
    extra: { nanite: '1', trees: TREES, lodnear: '4', simband: '6', lodpow: '0.6', instminpx: '128' },
  });
  console.log(`[forestfull] → ${url}`);
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__laas && (window.__laas.ready || window.__laas.error != null), undefined, {
    timeout: 300000,
    polling: 250,
  });
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);
  await page.evaluate(async () => {
    if (window.__laas.settle) await window.__laas.settle(120);
  });
  const acc = new Map<string, number[]>();
  for (let i = 0; i < 30; i++) {
    const r = (await page.evaluate(async () => {
      if (window.__laas.settle) await window.__laas.settle(4);
      const st = window.__laas.stats;
      const g = (st?.gpuPasses ?? {}) as Record<string, number>;
      const c = (st?.counters ?? {}) as Record<string, number>;
      return {
        fps: st?.fps ?? 0,
        frameMs: st?.frameMs ?? 0,
        w1: g['c.nanRasterWorld1'] ?? 0,
        scene: g['r.scene'] ?? 0,
        compute: g['compute'] ?? 0,
        render: g['render'] ?? 0,
        vis: c['nanite.visClusters'] ?? -1,
      };
    })) as Record<string, number>;
    if (i >= 8) for (const [k, v] of Object.entries(r)) (acc.get(k) ?? acc.set(k, []).get(k))?.push(v);
  }
  await page.screenshot({ path: 'shots/wip/forestfull.png' });
  const m = (k: string): number => median(acc.get(k) ?? [0]);
  console.log(
    `[forestfull] trees=${TREES} vis=${m('vis')} → fps ${m('fps').toFixed(0)} frameMs ${m('frameMs').toFixed(1)} | ` +
      `world1 ${m('w1').toFixed(2)} scene ${m('scene').toFixed(2)} | compute ${m('compute').toFixed(2)} render ${m('render').toFixed(2)}  (shot: shots/wip/forestfull.png)`,
  );
  await browser.close();
}
main().catch((e) => {
  console.error('[forestfull] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
