/**
 * N9-C0 GATE — real mesh-leaf crowns through the nanite path (MATERIAL_CLASS.leaf
 * + the 'leaf' flutter channel + hero-ring registration). Boots the world twice
 * at a hero-tree framing (bm7, trunks ≤26 m) — leaf OFF (today's bare trunks) vs
 * leaf ON (`?naniteleaf=1`) — and proves:
 *
 *   REGISTRATION landed: leaf ON adds meshes (+24 leaf heads = 6 species × 4
 *   variants) and instances (the crown bound to every tree) vs leaf OFF, and
 *   emits visible clusters. A flat count = the leaf head never registered/bound.
 *
 *   CLEAN: no fatal boot, no console/page errors, leaf clusters actually raster
 *   (visClusters rises).
 *
 * Screenshots → shots/wip/leaf-* for the eyeball pass (crowns present, lit,
 * fluttering, not black) + ?nandbg=cls (leaf = bright green) to see coverage.
 * The MOTION (flutter + crown sway-with-trunk) is the user's in-motion call.
 *
 *   npx tsx tools/probe-leaf.ts            # needs the dev server on :5173
 */

import { launchWebGPU, laasUrl } from './launch';

const COUNTERS = [
  'nanite.meshes',
  'nanite.inst',
  'nanite.clusters',
  'nanite.visClusters',
  'nanite.trisK',
  'nanite.covered',
] as const;

let failures = 0;
const fail = (m: string): void => {
  failures++;
  console.error(`  FAIL ${m}`);
};

async function boot(
  leaf: boolean,
  shot: string,
  dbg?: string,
): Promise<{ counters: Record<string, number>; errs: string[]; shotPath: string }> {
  const width = 1280;
  const height = 720;
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  const errs: string[] = [];
  page.on('console', (m) => {
    if (m.type() === 'error') errs.push(m.text());
  });
  page.on('pageerror', (e) => errs.push('PAGEERR ' + e.message));

  const extra: Record<string, string> = {
    nanite: '1',
    nanitedterrain: '0',
    nanshadow: '0',
    occl: '0',
    shot,
  };
  if (leaf) extra.naniteleaf = '1';
  if (dbg) extra.nandbg = dbg;
  const url = laasUrl({ scene: 'world', width, height, freeze: true, extra });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error != null),
    undefined,
    { timeout: 180000, polling: 250 },
  );
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot (leaf=${leaf} shot=${shot}): ${err}`);

  const counters = await page.evaluate(async (keys: readonly string[]) => {
    if (window.__laas.settle) await window.__laas.settle(12);
    const c = window.__laas.stats?.counters ?? {};
    const out: Record<string, number> = {};
    for (const k of keys) out[k] = c[k] ?? -1;
    return out;
  }, COUNTERS);

  const tag = `${leaf ? 'on' : 'off'}${dbg ? '-' + dbg : ''}-s${shot}`;
  const shotPath = `shots/wip/leaf-${tag}.png`;
  await page.screenshot({ path: shotPath });
  if (errs.length) fail(`leaf=${leaf} shot=${shot}${dbg ? ' dbg=' + dbg : ''}: ${errs.length} errors: ${errs.slice(0, 3).join(' | ')}`);
  await browser.close();
  return { counters, errs, shotPath };
}

function dump(label: string, c: Record<string, number>): void {
  console.log(`  ${label.padEnd(16)} ${COUNTERS.map((k) => `${k.replace('nanite.', '')}=${c[k]}`).join('  ')}`);
}

async function main(): Promise<void> {
  console.log('[leaf] N9-C0 — hero-tree framing (bm7), leaf OFF vs ON');
  const off = await boot(false, '7');
  const on = await boot(true, '7');
  dump('leaf OFF', off.counters);
  dump('leaf ON', on.counters);

  // REGISTRATION proof: leaf heads (+24) + their instances must appear, and they
  // must actually rasterise (visClusters up). Otherwise the head never bound.
  const dMesh = (on.counters['nanite.meshes'] ?? 0) - (off.counters['nanite.meshes'] ?? 0);
  const dInst = (on.counters['nanite.inst'] ?? 0) - (off.counters['nanite.inst'] ?? 0);
  const dVis = (on.counters['nanite.visClusters'] ?? 0) - (off.counters['nanite.visClusters'] ?? 0);
  console.log(`  Δmeshes=${dMesh}  Δinst=${dInst}  ΔvisClusters=${dVis}`);
  // tree species × 4 variants (5 species today = 20; the 6th foliageColor is understory)
  if (dMesh <= 0 || dMesh % 4 !== 0)
    fail(`leaf heads missing/odd: Δmeshes ${dMesh} (expect a positive multiple of 4 variants)`);
  if (dInst <= 0) fail(`leaf instances missing: Δinst ${dInst} ≤ 0 (crown not bound to trees)`);
  if (dVis <= 0) fail(`leaf clusters not rastered: ΔvisClusters ${dVis} ≤ 0 (crown invisible)`);

  // cls-debug coverage shot (leaf = bright green) + a forest framing (bm4)
  console.log('[leaf] cls-debug + forest framing');
  const cls = await boot(true, '7', 'cls');
  const forest = await boot(true, '4');
  dump('forest ON', forest.counters);
  console.log(`  shots: ${off.shotPath} | ${on.shotPath} | ${cls.shotPath} | ${forest.shotPath}`);

  if (failures > 0) {
    console.error(`[leaf] ${failures} FAILURES`);
    process.exit(1);
  }
  console.log('[leaf] N9-C0 OK — leaf heads registered + bound + rastering; clean boot, no errors');
}

main().catch((e) => {
  console.error('[leaf] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
