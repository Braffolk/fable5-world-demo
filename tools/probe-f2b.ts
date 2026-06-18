/**
 * probe-f2b: HONEST A/B of the world1 SCATTER raster WITH vs WITHOUT the
 * front-to-back per-pixel early-out (?f2b=1), at the user's real worst-case pose
 * (world scene, (-4.2,303.1,-1.4) T=11, 2592×1676). Uses measureActiveGpu
 * (GPU-bound isolated frames) — NOT the vsync-capped live gpuPasses.
 *
 * f2b is loss-exact ⇒ visClusters MUST match between A and B (same depth → same
 * HZB → same cut). The only thing that should move is c.nanRasterWorld1 time.
 *
 *   npx tsx tools/probe-f2b.ts            # needs dev server on :5173
 *   YAW=2.1 npx tsx tools/probe-f2b.ts
 */
import { launchWebGPU, laasUrl } from './launch';
import { measureActiveGpu } from './measure';
import type { Browser } from 'playwright';

const X = Number(process.env.X ?? -4.2);
const Y = Number(process.env.Y ?? 303.1);
const Z = Number(process.env.Z ?? -1.4);
const T = Number(process.env.T ?? 11);
const PITCH = Number(process.env.PITCH ?? -0.12);
const WIDTH = Number(process.env.WIDTH ?? 2592);
const HEIGHT = Number(process.env.HEIGHT ?? 1676);
// sweep a few yaws; the worst (longest sightline) is the peak SW-raster load.
const YAWS = (process.env.YAW ? [Number(process.env.YAW)] : [0, 1.05, 2.1, 3.14, 4.19, 5.24]);

interface LaasWin {
  __laas: {
    ready?: boolean;
    error?: string | null;
    setTimeOfDay?: (t: number) => void;
    setPose?: (p: { p: number[]; yaw: number; pitch: number }) => void;
    settle?: (n: number) => Promise<void>;
  };
}
declare const window: LaasWin;

async function boot(browser: Browser, f2b: boolean) {
  const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));
  const extra: Record<string, string> = { nanite: '1', vcompact: '1' };
  if (f2b) extra.f2b = '1';
  const url = laasUrl({ scene: 'world', width: WIDTH, height: HEIGHT, freeze: false, extra });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error != null),
    undefined,
    { timeout: 180000, polling: 250 },
  );
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);
  await page.evaluate((t) => window.__laas.setTimeOfDay?.(t), T);
  return page;
}

async function main(): Promise<void> {
  const { browser } = await launchWebGPU();
  for (const f2b of [false, true]) {
    const page = await boot(browser, f2b);
    console.log(`\n=== world1 ${f2b ? '+ f2b' : 'BASELINE'} (worst pose, ${WIDTH}×${HEIGHT}) ===`);
    for (const yaw of YAWS) {
      await page.evaluate(
        (p) => window.__laas.setPose?.({ p: [p.x, p.y, p.z], yaw: p.yaw, pitch: p.pitch }),
        { x: X, y: Y, z: Z, yaw, pitch: PITCH },
      );
      await page.evaluate(async () => window.__laas.settle && (await window.__laas.settle(12)));
      const r = await measureActiveGpu(page, 'c.nanRasterWorld1', { frames: 24, warmup: 10 });
      console.log(
        `  yaw=${yaw.toFixed(2)}  raster p50=${r.p50.toFixed(2)} p95=${r.p95.toFixed(2)}  gpuWall=${r.gpuWallMs.toFixed(2)}  visCl=${r.visClusters}  rej=${r.rejected}`,
      );
    }
    await page.close();
  }
  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
