/**
 * N8-D1d terrain-DAG persistent cache validation (D-N30). Boots twice in the
 * SAME browser context (IndexedDB persists across navigations): boot 1 builds +
 * caches the DAG ([worker]/[sync]); boot 2 must load it from the cache ([cache],
 * ~0 ms build) and render the IDENTICAL cut (same dagClusters) — proving boot
 * renders the DAG from frame 1 with no rebuild and no window fallback.
 *
 *   npx tsx tools/probe-dagcache.ts [gridN=256]   # needs the dev server on :5173
 */
import { laasUrl, launchWebGPU } from './launch';
import type { Page } from 'playwright';

const GRID = Math.max(2, Math.floor(Number(process.argv[2] ?? '256')));

async function boot(page: Page, label: string): Promise<{ via: string; ms: number; dag: number }> {
  let via = '?';
  let ms = -1;
  const onConsole = (m: { text(): string }): void => {
    const mt = m.text().match(/terrain DAG \[([\w-]+)\]:.*?(\d+)\s*ms/);
    if (mt) {
      via = mt[1] ?? '?';
      ms = Number(mt[2]);
    }
  };
  page.on('console', onConsole);
  const url = laasUrl({
    scene: 'world',
    width: 1280,
    height: 720,
    freeze: true,
    extra: { nanite: '1', nanshadow: '0', occl: '1', nanitedterrain: String(GRID) },
  });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error != null),
    undefined,
    { timeout: 180000, polling: 250 },
  );
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`boot ${label}: ${err}`);
  let dag = -1;
  for (let i = 0; i < 12; i++) {
    dag = await page.evaluate(async () => {
      if (window.__laas.settle) await window.__laas.settle(16);
      return window.__laas.stats?.counters['nanite.dagClusters'] ?? -1;
    });
    if (dag >= 0) break;
  }
  page.off('console', onConsole);
  return { via, ms, dag };
}

async function main(): Promise<void> {
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
  const b1 = await boot(page, 'first');
  console.log(`[dagcache] boot 1 (gridN ${GRID}): via=${b1.via} build=${b1.ms}ms dag=${b1.dag}  (builds + caches)`);
  // let the fire-and-forget IndexedDB put commit before reloading
  await page.waitForTimeout(3000);
  const b2 = await boot(page, 'second');
  console.log(`[dagcache] boot 2 (gridN ${GRID}): via=${b2.via} build=${b2.ms}ms dag=${b2.dag}  (expect CACHE)`);
  await browser.close();
  const ok = b2.via === 'cache' && b2.dag > 0 && b1.dag === b2.dag;
  console.log(
    `[dagcache] ${ok ? 'PASS' : 'FAIL'}: boot2 via=${b2.via} (want cache), dag ${b1.dag}${b1.dag === b2.dag ? '==' : '!='}${b2.dag}, ` +
      `build ${b1.ms}ms → ${b2.ms}ms`,
  );
  if (!ok) process.exit(1);
}
main().catch((e) => {
  console.error('[dagcache] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
