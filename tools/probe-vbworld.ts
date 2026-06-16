/**
 * PERF-VB4 (D-N45): measure + screenshot the WORLD single-pass raster (now the default —
 * one SW+HW pass: 24-bit depth election into visPayloadV + full-id side buffer visBV).
 * Boots scene=world&nanite=1 at a configurable pose, thermal-warms, reports the median
 * nanRasterWorld1 GPU pass time, and shoots the frame (eyeball depth-precision banding /
 * the rare wrong-cluster speckle).
 *
 *   PITCH=-0.12 DY=25 npx tsx tools/probe-vbworld.ts          # grazing slope
 *   NANDBG=cluster npx tsx tools/probe-vbworld.ts             # per-cluster hash tint
 */
import type { CamPose } from '../src/core/Hooks';
import { laasUrl, launchWebGPU } from './launch';

const W = 1280;
const H = 720;
const NANDBG = process.env.NANDBG ?? '';
const PITCH = Number(process.env.PITCH ?? '-0.4');
const DY = Number(process.env.DY ?? '150');
const median = (a: number[]): number => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)] ?? 0;

async function main(): Promise<void> {
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));
  const extra: Record<string, string> = { nanite: '1' };
  if (NANDBG) extra.nandbg = NANDBG;
  const url = laasUrl({ scene: 'world', width: W, height: H, freeze: true, extra });
  console.log(`[vbworld] → ${url}`);
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__laas && (window.__laas.ready || window.__laas.error != null), undefined, {
    timeout: 240000,
    polling: 250,
  });
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);
  const base = (await page.evaluate(() => window.__laas.getPose?.() ?? null)) as CamPose | null;
  if (base) {
    await page.evaluate(
      (a) => window.__laas.setPose?.({ yaw: a.p.yaw, pitch: a.pitch, p: [a.p.p[0], a.p.p[1] + a.dy, a.p.p[2]] }),
      { p: base, pitch: PITCH, dy: DY },
    );
  }
  await page.evaluate(async () => {
    if (window.__laas.settle) await window.__laas.settle(220);
  });
  const acc: number[] = [];
  let vis = -1;
  for (let i = 0; i < 36; i++) {
    const r = (await page.evaluate(async () => {
      if (window.__laas.settle) await window.__laas.settle(4);
      const g = (window.__laas.stats?.gpuPasses ?? {}) as Record<string, number>;
      const c = (window.__laas.stats?.counters ?? {}) as Record<string, number>;
      return { w1: g['c.nanRasterWorld1'] ?? 0, vis: c['nanite.visClusters'] ?? -1 };
    })) as { w1: number; vis: number };
    if (i >= 10) acc.push(r.w1);
    vis = r.vis;
  }
  const tag = NANDBG ? `world-${NANDBG}` : 'world';
  await page.screenshot({ path: `shots/wip/vbworld-${tag}.png` });
  console.log(`[vbworld] vis=${vis} → nanRasterWorld1 = ${median(acc).toFixed(2)} ms  (shot: shots/wip/vbworld-${tag}.png)`);
  await browser.close();
}
main().catch((e) => {
  console.error('[vbworld] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
