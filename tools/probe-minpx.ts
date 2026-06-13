/**
 * N8-D1e min-screen-size cull + unbounded envelope (D-N33) validation.
 * A/Bs the SAME far pose with the feature off (minPx=0 ⇒ must equal the pre-D1e
 * finite-envelope behaviour) vs on (?nanitemin=1 ⇒ DAG'd meshes persist past their
 * finite envelope until sub-pixel; sub-pixel clusters culled). Captures dagClusters
 * + frameMs + screenshots at a near pose and a far (−700 m) vista for both.
 *
 *   npx tsx tools/probe-minpx.ts          # needs the dev server on :5173
 */
import type { CamPose } from '../src/core/Hooks';
import { launchWebGPU, laasUrl } from './launch';

async function run(minpx: number, tag: string): Promise<{ near: number; far: number; nearMs: number; farMs: number }> {
  const width = 1280;
  const height = 720;
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  const url = laasUrl({
    scene: 'world',
    width,
    height,
    freeze: true,
    extra: { nanite: '1', nanitedag: 'all', nanshadow: '0', occl: '1', shot: '4', nanitemin: String(minpx) },
  });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__laas && (window.__laas.ready || window.__laas.error != null), undefined, {
    timeout: 180000,
    polling: 250,
  });
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);
  const sample = async (): Promise<{ dag: number; ms: number }> => {
    for (let t = 0; t < 12; t++) {
      const v = await page.evaluate(async () => {
        if (window.__laas.settle) await window.__laas.settle(16);
        return { dag: window.__laas.stats?.counters['nanite.dagClusters'] ?? -1, ms: window.__laas.stats?.frameMs ?? -1 };
      });
      if (v.dag >= 0) return v;
    }
    return { dag: -1, ms: -1 };
  };
  const near = await sample();
  await page.screenshot({ path: `shots/wip/minpx-${tag}-near.png` });
  const pose = (await page.evaluate(() => window.__laas.getPose?.() ?? null)) as CamPose | null;
  if (!pose) throw new Error('getPose missing');
  const fwd: [number, number, number] = [
    Math.sin(pose.yaw) * Math.cos(pose.pitch),
    Math.sin(pose.pitch),
    -Math.cos(pose.yaw) * Math.cos(pose.pitch),
  ];
  const back = 700;
  await page.evaluate(
    ({ pose, fwd, back }) => {
      window.__laas.setPose?.({ ...pose, p: [pose.p[0] - fwd[0] * back, pose.p[1] - fwd[1] * back, pose.p[2] - fwd[2] * back] });
    },
    { pose, fwd, back },
  );
  const far = await sample();
  await page.screenshot({ path: `shots/wip/minpx-${tag}-far.png` });
  await browser.close();
  return { near: near.dag, far: far.dag, nearMs: near.ms, farMs: far.ms };
}

async function main(): Promise<void> {
  const off = await run(0, 'off');
  console.log(`[minpx] OFF (minPx=0)  near ${off.near} cl ${off.nearMs.toFixed(1)}ms | far ${off.far} cl ${off.farMs.toFixed(1)}ms`);
  const on = await run(1, 'on');
  console.log(`[minpx] ON  (minPx=1)  near ${on.near} cl ${on.nearMs.toFixed(1)}ms | far ${on.far} cl ${on.farMs.toFixed(1)}ms`);
  console.log(`[minpx] near Δ ${on.near - off.near} cl (≈0 expected — minPx barely culls up close)`);
  console.log(`[minpx] far  Δ ${on.far - off.far} cl (unbounded envelope adds distant clusters, sub-pixel cull removes tiny ones)`);
  console.log('[minpx] inspect shots/wip/minpx-{off,on}-{near,far}.png');
}
main().catch((e) => {
  console.error('[minpx] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
