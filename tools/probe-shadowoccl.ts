/**
 * N5 shadow S2-OCCL gate: the per-cascade light-HZB occlusion cull only activates
 * under a MOVING camera (a static camera caches the cascades via CsmCached/R1 — they
 * raster once at boot with an all-far prev-HZB, then cache, so occlusion never reaches
 * the frame-2+ where the prev-HZB is populated). This MOVES the camera each frame so
 * the cascades re-raster with a populated prev-HZB, then medians shTotal (the shadow
 * visible-cluster count = the moving-raster proxy) over the moving window.
 *
 *   npx tsx tools/probe-shadowoccl.ts        # bm7; compares shadowoccl 0 vs 1
 *   SHOT=3 npx tsx tools/probe-shadowoccl.ts # vista
 */
import { launchWebGPU, laasUrl } from './launch';

async function boot(occl: '0' | '1'): Promise<{ shTotal: number; shot: string }> {
  const shot = process.env.SHOT ?? '7';
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
  page.on('pageerror', (err) => console.error('[pageerror]', err.message));
  const url = laasUrl({
    scene: 'world',
    width: 1280,
    height: 720,
    freeze: false, // shadow.run must execute every frame to populate the HZB
    extra: { nanite: '1', shot, shadowtau: '1', shadowoccl: occl },
  });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__laas && (window.__laas.ready || window.__laas.error != null), undefined, {
    timeout: 180000,
    polling: 250,
  });
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);
  await page.evaluate(async () => window.__laas.settle && (await window.__laas.settle(16)));

  // MOVE the camera in a slow orbit-ish drift so every cascade re-rasters each frame
  // with a populated prev-HZB; sample shTotal across the moving window.
  const samples: number[] = [];
  for (let i = 0; i < 40; i++) {
    const sh = await page.evaluate(async (step) => {
      const h = window.__laas;
      if (h.getPose && h.setPose) {
        const p = h.getPose();
        // forward + slight yaw drift — keeps the cascades moving
        h.setPose({ ...p, yaw: p.yaw + 0.012, p: [p.p[0] + step, p.p[1], p.p[2] + step * 0.6] });
      }
      if (h.settle) await h.settle(1);
      return h.stats?.counters['nanite.shTotal'] ?? -1;
    }, 1.2);
    if (i >= 10 && sh > 0) samples.push(sh); // skip warmup
  }
  await page.screenshot({ path: `shots/wip/occlmove-${occl}.png` });
  await browser.close();
  samples.sort((a, b) => a - b);
  return { shTotal: samples[Math.floor(samples.length / 2)] ?? -1, shot };
}

async function main(): Promise<void> {
  console.log('[shadowoccl] booting shadowoccl=0 (no occlusion) …');
  const off = await boot('0');
  console.log('[shadowoccl] booting shadowoccl=1 (light-HZB occlusion) …');
  const on = await boot('1');
  const ratio = off.shTotal > 0 ? (on.shTotal / off.shTotal) : 0;
  console.log(`\n[shadowoccl] bm${off.shot} MOVING, median shTotal (shadow cluster count):`);
  console.log(`  occl OFF: ${off.shTotal.toLocaleString()}`);
  console.log(`  occl ON:  ${on.shTotal.toLocaleString()}  (×${ratio.toFixed(2)} = ${((1 - ratio) * 100).toFixed(0)}% culled)`);
}
main().catch((e) => {
  console.error('[shadowoccl] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
