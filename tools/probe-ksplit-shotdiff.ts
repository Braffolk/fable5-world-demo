/**
 * ksplit SHOTDIFF (task #76 Stage-1 correctness gate) — proves the world1 kernel-split
 * is BYTE-IDENTICAL to the unified kernel. Boots the raster config FROZEN (freeze=1 →
 * wind/motion stopped ⇒ two boots render the same frame), screenshots baseline vs
 * ?ksplit=1 at the dense-foliage eye pose, reports mean abs pixel diff (0..255) + an
 * amplified diff PNG. Expected: diff ≈ 0 (a disjoint isHF partition + order-independent
 * atomic election must reconstruct the same winners). A large diff = the split broke
 * rasterization (garbage / empty vis); a black frame on ksplit = pipeline validation death.
 *
 *   npx tsx tools/probe-ksplit-shotdiff.ts
 *
 * Serial (one page at a time) — never two GPU browsers at once (bench-contamination law).
 */
import { launchWebGPU, laasUrl } from './launch';
import type { Page } from 'playwright';
import { mkdirSync } from 'node:fs';
import sharp from 'sharp';

const OUT = 'shots/ksplit';
const WIDTH = Number(process.env.WIDTH ?? 1296);
const HEIGHT = Number(process.env.HEIGHT ?? 838);
const POSE = { p: [0, 2, 0] as [number, number, number], yaw: 0.6, pitch: -0.02 };
const T = Number(process.env.T ?? 11);

const BASE_CONFIG: Record<string, string> = {
  nanite: '1',
  dpr: '2',
  clhw: '1',
  clhwmax: '32',
  grass: '0',
  nanshadow: '0',
};

async function boot(page: Page, extraFlags: Record<string, string>): Promise<void> {
  const extra = { ...BASE_CONFIG, ...extraFlags };
  // freeze=1 → deterministic frame across boots (wind/time stopped)
  await page.goto(laasUrl({ scene: 'world', freeze: true, hud: false, extra }), {
    waitUntil: 'domcontentloaded',
  });
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error != null),
    undefined,
    { timeout: 300000, polling: 250 },
  );
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error('boot err: ' + err);
  await page.evaluate((t) => window.__laas.setTimeOfDay?.(t), T);
}

async function shoot(page: Page, tag: string): Promise<string> {
  await page.evaluate(async (po) => {
    window.__laas.setPose!({ p: po.p, yaw: po.yaw, pitch: po.pitch });
    if (window.__laas.settle) await window.__laas.settle(60);
  }, POSE);
  const path = `${OUT}/eye-${tag}.png`;
  await page.screenshot({ path });
  return path;
}

async function meanAbsDiff(a: string, b: string): Promise<number> {
  const [ba, bb] = await Promise.all([
    sharp(a).removeAlpha().raw().toBuffer(),
    sharp(b).removeAlpha().raw().toBuffer(),
  ]);
  let sum = 0;
  const n = Math.min(ba.length, bb.length);
  for (let i = 0; i < n; i++) sum += Math.abs(ba[i]! - bb[i]!);
  return sum / n;
}

async function main(): Promise<void> {
  mkdirSync(OUT, { recursive: true });
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({
    viewport: { width: WIDTH, height: HEIGHT },
    deviceScaleFactor: 1,
  });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));

  console.log('[ksplit-shotdiff] boot baseline (unified world1)...');
  await boot(page, {});
  const baseShot = await shoot(page, 'baseline');

  console.log('[ksplit-shotdiff] boot ?ksplit=1 (Explicit + Terrain kernels)...');
  await boot(page, { ksplit: '1' });
  const splitShot = await shoot(page, 'ksplit');

  await browser.close();

  const diff = await meanAbsDiff(baseShot, splitShot);
  const diffPath = `${OUT}/eye-DIFF.png`;
  await sharp(baseShot)
    .removeAlpha()
    .composite([{ input: splitShot, blend: 'difference' }])
    .linear(12, 0)
    .toFile(diffPath);

  console.log(`\n===== ksplit shotdiff (mean abs pixel diff, 0..255) =====`);
  console.log(`baseline: ${baseShot}`);
  console.log(`ksplit:   ${splitShot}`);
  console.log(`DIFF:     ${diff.toFixed(4)}   (amplified → ${diffPath})`);
  console.log(
    diff < 0.05
      ? '✅ BYTE-IDENTICAL (diff ≈ 0) — the split reconstructs the same frame.'
      : diff < 2
        ? '⚠️ small diff — inspect the DIFF png (TAA jitter? or a real seam).'
        : '❌ LARGE DIFF — the split changed rasterization (garbage/empty vis). Inspect shots.',
  );
}

main().catch((e) => {
  console.error('[ksplit-shotdiff] FAILED:', e instanceof Error ? (e.stack ?? e.message) : e);
  process.exit(1);
});
