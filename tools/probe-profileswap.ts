/**
 * Stage-0 smoke test for the ?profile=1 two-device swap (core/ProfileBoot).
 *
 * Confirms the LOAD-BEARING assumption: after swapping to a fresh 'laas-render'
 * device post-load, the nanite geometry re-uploads and renders on the new device
 * (non-zero visClusters/hwTris, non-black frame). Terrain/bark/canopy textures are
 * intentionally BLANK at this stage (Stage 1 transfers them) — we are ONLY checking
 * that geometry survives the device swap.
 *
 * Runs under ?pure=1 so the (not-yet-re-pointed) post stack is out of the path.
 *
 *   npx tsx tools/probe-profileswap.ts
 */
import { laasUrl, launchWebGPU } from './launch';

async function main(): Promise<void> {
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: 800, height: 500 }, deviceScaleFactor: 1 });

  const logs: string[] = [];
  page.on('console', (m) => {
    const t = m.text();
    if (/profile|uncaptured|webgpu|error|laas|swap/i.test(t)) logs.push(`[${m.type()}] ${t}`);
  });
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));

  const url = laasUrl({
    scene: 'world',
    width: 800,
    height: 500,
    freeze: true,
    extra: { nanite: '1', profile: '1', pure: '1' },
  });
  console.log(`[profileswap] ${url}`);
  await page.goto(url, { waitUntil: 'domcontentloaded' });

  let readyErr: string | null = null;
  try {
    await page.waitForFunction(
      () => window.__laas && (window.__laas.ready || window.__laas.error != null),
      undefined,
      { timeout: 240000, polling: 250 },
    );
  } catch {
    readyErr = 'never became ready (timeout) — the swap likely threw or hung';
  }
  const bootErr = await page.evaluate(() => window.__laas?.error ?? null).catch(() => null);

  // let the render loop run so counters populate on the NEW device
  await page.evaluate(async () => {
    if (window.__laas?.settle) await window.__laas.settle(12);
  }).catch(() => undefined);

  const info = await page
    .evaluate(() => {
      const c = window.__laas?.stats?.counters ?? {};
      return {
        ready: window.__laas?.ready ?? false,
        visClusters: c['nanite.visClusters'] ?? -1,
        hwTris: c['nanite.hwTris'] ?? -1,
        dagClusters: c['nanite.dagClusters'] ?? -1,
        gpuBuffers: c['gpu.buffers'] ?? -1,
        gpuTextures: c['gpu.textures'] ?? -1,
        frame: window.__laas?.stats?.frame ?? -1,
      };
    })
    .catch(() => null);

  await page.screenshot({ path: 'shots/wip/profileswap-stage0.png' });
  await browser.close();

  console.log('=== console (filtered) ===');
  for (const l of logs) console.log(l);
  console.log('=== result ===');
  console.log('readyErr :', readyErr);
  console.log('bootErr  :', bootErr);
  console.log('info     :', JSON.stringify(info));
  console.log('screenshot: shots/wip/profileswap-stage0.png');

  const ok =
    !!info && info.ready && Number(info.visClusters) > 0 && Number(info.hwTris) > 0 && !bootErr;
  console.log(
    ok
      ? '\n✅ STAGE 0 PASS: nanite geometry re-uploaded + rendered on laas-render (device swap survives)'
      : '\n❌ STAGE 0 FAIL: geometry did NOT render on the swapped device — inspect logs/screenshot above',
  );
  process.exit(ok ? 0 : 1);
}
main().catch((e) => {
  console.error('[profileswap] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
