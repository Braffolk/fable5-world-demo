/**
 * probe-tileflick: DIAGNOSE the tiled-raster cluster flicker. Boots ?tileproto=1 in the dense
 * forest canopy (spawn 0,2,0) via setPose (the shoot tool mis-places the forest cam; setPose
 * works — see probe-worstpos), settles, then per frame reads:
 *   tileStat() = [tileOverflow, clusterOverflow, hwEnqueues, _]  (accumulated across batches)
 *   counters   = visClusters, hwTris
 * If tileOverflow>0 ⇒ TILE_CAP drop. If hwEnqueues≈HW_CAP ⇒ HW drop. If visClusters WOBBLES
 * frame-to-frame ⇒ the cut itself oscillates (occlusion feedback). That isolates the cause.
 *
 *   npx tsx tools/probe-tileflick.ts            # needs dev server on :5173
 */
import { launchWebGPU, laasUrl } from './launch';

const X = Number(process.env.X ?? 0);
const Y = Number(process.env.Y ?? 2);
const Z = Number(process.env.Z ?? 0);
const YAW = Number(process.env.YAW ?? 0.6);
const PITCH = Number(process.env.PITCH ?? -0.02);
const W = Number(process.env.W ?? 1280);
const H = Number(process.env.H ?? 800);
const FRAMES = Number(process.env.FRAMES ?? 16);

interface LaasWin {
  __laas: {
    ready?: boolean;
    error?: string | null;
    setPose?: (p: { p: number[]; yaw: number; pitch: number }) => void;
    settle?: (n: number) => Promise<void>;
    stats?: { counters?: Record<string, number> };
  };
  __laasNanite?: { tileStat?: () => Promise<number[]> };
}
declare const window: LaasWin;

async function main(): Promise<void> {
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));
  const extra: Record<string, string> = { nanite: '1' };
  if ((process.env.TILEPROTO ?? '1') === '1') extra.tileproto = '1';
  if (process.env.F2B === '1') extra.f2b = '1';
  if (process.env.OCCL === '0') extra.occl = '0';
  const url = laasUrl({ scene: 'forest', width: W, height: H, freeze: false, extra });
  console.log(`[tileflick] booting ${url}`);
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error != null),
    undefined,
    { timeout: 180000, polling: 250 },
  );
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);
  await page.evaluate(
    (p) => window.__laas.setPose?.({ p: [p.x, p.y, p.z], yaw: p.yaw, pitch: p.pitch }),
    { x: X, y: Y, z: Z, yaw: YAW, pitch: PITCH },
  );
  await page.evaluate(async () => window.__laas.settle && (await window.__laas.settle(20)));

  console.log(`[tileflick] pose (${X},${Y},${Z}) yaw=${YAW} — ${FRAMES} frames @${W}×${H}:`);
  console.log('  frame | visClusters  hwTris | tileOvf clusterOvf hwEnq');
  const visSeen: number[] = [];
  for (let i = 0; i < FRAMES; i++) {
    await page.evaluate(async () => window.__laas.settle && (await window.__laas.settle(1)));
    const row = await page.evaluate(async () => {
      const st = (await window.__laasNanite?.tileStat?.()) ?? [];
      const c = window.__laas.stats?.counters ?? {};
      return { st, vis: c['nanite.visClusters'] ?? -1, hw: c['nanite.hwTris'] ?? -1 };
    });
    visSeen.push(row.vis);
    console.log(
      `  ${String(i).padStart(5)} | ${String(row.vis).padStart(11)} ${String(row.hw).padStart(7)} | ` +
        `${String(row.st[0] ?? 0).padStart(7)} ${String(row.st[1] ?? 0).padStart(10)} ${String(row.st[2] ?? 0).padStart(8)}`,
    );
  }
  const min = Math.min(...visSeen), max = Math.max(...visSeen);
  console.log(`\n[tileflick] visClusters range ${min}..${max} (Δ=${max - min}) — ${max - min > 200 ? 'OSCILLATES (cut/occlusion feedback)' : 'STABLE (drop is tri-level: TILE_CAP/HW)'}`);
  const tag = (process.env.TILEPROTO ?? '1') === '1' ? 'tiled' : 'base';
  await page.screenshot({ path: `shots/b1-proto/flick_${tag}.png` });
  console.log(`[tileflick] wrote shots/b1-proto/flick_${tag}.png`);
  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
