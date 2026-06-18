/**
 * probe-b3perf: HONEST GPU-time A/B of the tiled raster at the dense forest canopy.
 * Uses measureActiveGpu (GPU-bound isolated frames → real gpuWall, NOT vsync-capped live
 * gpuPasses). occl=0 ⇒ a FROZEN deterministic cut (same geometry every frame, no wind/occlusion
 * variance) so the A/B is fair. setPose places the camera IN the canopy (the shoot tool mis-places it).
 *
 * Run the SAME probe against B3-on (HEAD) and B3-off (git checkout fed6771 -- the tiled file):
 *   npx tsx tools/probe-b3perf.ts                  # tileproto, occl=0
 *   TILEPROTO=0 npx tsx tools/probe-b3perf.ts       # world1 scatter baseline
 */
import { launchWebGPU, laasUrl } from './launch';
import { measureActiveGpu } from './measure';

const X = Number(process.env.X ?? 0), Y = Number(process.env.Y ?? 2), Z = Number(process.env.Z ?? 0);
const YAW = Number(process.env.YAW ?? 0.6), PITCH = Number(process.env.PITCH ?? -0.02);
const W = Number(process.env.W ?? 1280), H = Number(process.env.H ?? 800);
const TILEPROTO = (process.env.TILEPROTO ?? '1') === '1';
const METRIC = process.env.METRIC ?? (TILEPROTO ? 'c.nanRasterTiled' : 'c.nanRasterWorld1');

interface LaasWin {
  __laas: { ready?: boolean; error?: string | null;
    setPose?: (p: { p: number[]; yaw: number; pitch: number }) => void;
    settle?: (n: number) => Promise<void>; stats?: { counters?: Record<string, number> }; };
}
declare const window: LaasWin;

async function main(): Promise<void> {
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));
  const extra: Record<string, string> = { nanite: '1' };
  if ((process.env.OCCL ?? '0') === '0') extra.occl = '0'; // default deterministic; OCCL=1 ⇒ occlusion ON (realistic)
  if ((process.env.VCOMPACT ?? '1') === '1') extra.vcompact = '1';
  if (process.env.F2B === '1') extra.f2b = '1';
  if (TILEPROTO) extra.tileproto = '1';
  const url = laasUrl({ scene: 'forest', width: W, height: H, freeze: false, extra });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__laas && (window.__laas.ready || window.__laas.error != null), undefined, { timeout: 180000, polling: 250 });
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);
  await page.evaluate((p) => window.__laas.setPose?.({ p: [p.x, p.y, p.z], yaw: p.yaw, pitch: p.pitch }), { x: X, y: Y, z: Z, yaw: YAW, pitch: PITCH });
  await page.evaluate(async () => window.__laas.settle && (await window.__laas.settle(20)));
  const vis = await page.evaluate(() => window.__laas.stats?.counters?.['nanite.visClusters'] ?? -1);
  const r = await measureActiveGpu(page, METRIC, { frames: 30, warmup: 15 });
  console.log(`\n[b3perf] ${TILEPROTO ? 'TILED' : 'world1'} occl=0 canopy ${W}x${H} visClusters=${vis}`);
  console.log(`  gpuWall  p50=${r.gpuWallMs.toFixed(2)} ms   (whole-frame GPU time — the honest metric)`);
  console.log(`  ${METRIC}  p50=${r.p50.toFixed(2)} p95=${r.p95.toFixed(2)} ms   rejected=${r.rejected}/${r.samples.length + r.rejected}`);
  // per-pass breakdown (median over good frames) — decomposes the frame to target optimization
  const good = r.frames.filter((f) => !f.capSuspect);
  const keys = new Set<string>();
  good.forEach((f) => Object.keys(f.passes).forEach((k) => keys.add(k)));
  const med = (a: number[]): number => { const s = a.filter(Number.isFinite).sort((x, y) => x - y); return s.length ? (s[s.length >> 1] as number) : 0; };
  const rows = [...keys].map((k) => [k, med(good.map((f) => f.passes[k] ?? 0))] as [string, number]).filter((rr) => rr[1] > 0.05).sort((a, b) => b[1] - a[1]);
  console.log('  per-pass breakdown (median ms):');
  rows.forEach(([k, v]) => console.log('   ', k.padEnd(26), v.toFixed(2)));
  await browser.close();
}
main().catch((e) => { console.error(e); process.exit(1); });
