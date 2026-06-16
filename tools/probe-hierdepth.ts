/**
 * SHADOW-HIER tuning: find the TRUE max hier BFS depth. The traverse is a CPU-fixed
 * pass loop (HIER_MAX_DEPTH); too few passes ⇒ the deepest DAG anchor-chains never
 * emit their tail (holes), too many ⇒ wasted empty passes that dominate the 6-10×
 * shadow cull chains' dispatch count. Boots the forest interior (deepest veg DAG +
 * terrain) at a sweep of ?hierdepth and reports the camera cut (visClusters) + the
 * shadow total (shTotal). The cut is FLAT above the true depth and DROPS below it —
 * the largest depth where it drops is the min safe value; pick that + a margin.
 *
 *   npx tsx tools/probe-hierdepth.ts
 */
import { launchWebGPU, laasUrl } from './launch';

const WIDTH = Number(process.env.WIDTH ?? 1280);
const HEIGHT = Number(process.env.HEIGHT ?? 720);
const DEPTHS = (process.env.DEPTHS ?? '18,13,12,11,10,9,8,7').split(',').map((s) => Number(s.trim()));

async function boot(depth: number): Promise<{ vis: number; sh: number; err: string | null }> {
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error(`[pageerror depth=${depth}]`, e.message));
  const url = laasUrl({
    scene: 'world',
    width: WIDTH,
    height: HEIGHT,
    freeze: false,
    extra: { nanite: '1', shot: process.env.SHOT ?? '7', hierdepth: String(depth) },
  });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__laas && (window.__laas.ready || window.__laas.error != null), undefined, {
    timeout: 180000,
    polling: 250,
  });
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) {
    await browser.close();
    return { vis: -1, sh: -1, err };
  }
  // settle long enough for the readback counters (meter reads every 15 frames) + the
  // shadow cadence to populate shTotal
  await page.evaluate(async () => window.__laas.settle && (await window.__laas.settle(60)));
  const m = (await page.evaluate(() => {
    const c = window.__laas.stats?.counters ?? {};
    return {
      vis: c['nanite.visClusters'] ?? -1,
      sh: c['nanite.shTotal'] ?? -1,
    };
  })) as { vis: number; sh: number };
  await browser.close();
  return { vis: m.vis, sh: m.sh, err: null };
}

async function main(): Promise<void> {
  console.log(`[hierdepth] bm${process.env.SHOT ?? '7'} — sweep ?hierdepth = ${DEPTHS.join(', ')}`);
  const rows: { d: number; vis: number; sh: number }[] = [];
  for (const d of DEPTHS) {
    const r = await boot(d);
    if (r.err) {
      console.log(`  depth ${String(d).padStart(2)}: BOOT ERROR — ${r.err}`);
      continue;
    }
    rows.push({ d, vis: r.vis, sh: r.sh });
    console.log(`  depth ${String(d).padStart(2)}: visClusters ${r.vis.toLocaleString()}  shTotal ${r.sh.toLocaleString()}`);
  }
  // the reference cut = the deepest sweep value (assumed ≥ true max)
  const ref = rows.find((x) => x.d === Math.max(...DEPTHS));
  if (ref) {
    console.log(`\n[hierdepth] reference (depth ${ref.d}): vis ${ref.vis.toLocaleString()} sh ${ref.sh.toLocaleString()}`);
    for (const r of rows) {
      const okVis = r.vis === ref.vis ? 'OK ' : `DROP(${(((ref.vis - r.vis) / Math.max(1, ref.vis)) * 100).toFixed(2)}%)`;
      const okSh = r.sh === ref.sh ? 'OK ' : `DROP(${(((ref.sh - r.sh) / Math.max(1, ref.sh)) * 100).toFixed(2)}%)`;
      console.log(`  depth ${String(r.d).padStart(2)}: vis ${okVis}  sh ${okSh}`);
    }
  }
}
main().catch((e) => {
  console.error('[hierdepth] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
