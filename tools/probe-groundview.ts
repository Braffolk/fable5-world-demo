/**
 * N8-D2 Stage 2e diagnostic: WHY is the clip-DAG terrain flat at the WALK SPAWN
 * (ground level) but correct from altitude (probe-seams)? Boots the default, logs
 * the spawn pose + ground height + render counters, then screenshots at the spawn
 * AND at the same XZ elevated 150 m looking down.
 *
 *   npx tsx tools/probe-groundview.ts
 */
import type { CamPose } from '../src/core/Hooks';
import { laasUrl, launchWebGPU } from './launch';

async function main(): Promise<void> {
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: 640, height: 400 }, deviceScaleFactor: 1 });
  const extra: Record<string, string> = { nanite: '1' };
  if (process.env.NANSHADOW === '0') extra.nanshadow = '0';
  if (process.env.OCCL) extra.occl = process.env.OCCL;
  if (process.env.ABLATE) extra.ablate = process.env.ABLATE;
  if (process.env.NANSKIRT) extra.nanitedskirt = process.env.NANSKIRT;
  const url = laasUrl({ scene: 'world', width: 640, height: 400, freeze: true, extra });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__laas && (window.__laas.ready || window.__laas.error != null), undefined, {
    timeout: 180000,
    polling: 250,
  });
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);
  await page.evaluate(async () => {
    if (window.__laas.settle) await window.__laas.settle(16);
  });

  const snap = async (label: string): Promise<void> => {
    await page.evaluate(async () => {
      if (window.__laas.settle) await window.__laas.settle(8);
    });
    const info = await page.evaluate(() => {
      const c = window.__laas.stats?.counters ?? {};
      const pose = window.__laas.getPose?.() ?? null;
      const gp = window.__laas.groundProbe;
      const ground = pose && gp ? gp(pose.p[0], pose.p[2]).ground : null;
      return {
        pose,
        ground,
        resident: c['terrain.stream.resident'] ?? -1,
        visClusters: c['nanite.visClusters'] ?? -1,
        dagClusters: c['nanite.dagClusters'] ?? -1,
        hwTris: c['nanite.hwTris'] ?? -1,
        rejClust: c['nanite.rejClust'] ?? -1,
      };
    });
    console.log(`  [${label}] ${JSON.stringify(info)}`);
    await page.screenshot({ path: `shots/wip/groundview-${label}.png` });
  };

  await snap('spawn');

  // same XZ, +150 m, pitch down — the probe-seams-equivalent view
  const base = (await page.evaluate(() => window.__laas.getPose?.() ?? null)) as CamPose | null;
  if (base) {
    await page.evaluate(
      (p) => window.__laas.setPose?.({ yaw: p.yaw, pitch: -0.5, p: [p.x, p.y + 150, p.z] as [number, number, number] }),
      { yaw: base.yaw, x: base.p[0], y: base.p[1], z: base.p[2] },
    );
    await snap('elevated');
  }
  await browser.close();
  console.log('[groundview] inspect shots/wip/groundview-spawn.png vs groundview-elevated.png');
}
main().catch((e) => {
  console.error('[groundview] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
