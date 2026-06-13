/**
 * N8-D1 envelope PERF sanity — the unlimited DAG envelope makes the whole 4 km
 * field eligible to draw, so the HZB occlusion cull is now what keeps the frame
 * sane. probe-envelope runs occl OFF (deterministic worst case: ~3.7M clusters);
 * this measures the REAL path (occl ON, the user's default) at a few poses so we
 * know what a re-test actually costs vs the occl-off ceiling.
 *
 *   npx tsx tools/probe-envperf.ts          # needs the dev server on :5173
 */

import type { CamPose } from '../src/core/Hooks';
import { launchWebGPU, laasUrl } from './launch';

async function boot(occl: '0' | '1'): Promise<{ near: Sample; far: Sample }> {
  const width = 1280;
  const height = 720;
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  const url = laasUrl({
    scene: 'world',
    width,
    height,
    freeze: true,
    extra: { nanite: '1', nanitedag: 'all', nandbg: 'cluster', nanshadow: '0', occl, shot: '4' },
  });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__laas && (window.__laas.ready || window.__laas.error != null), undefined, {
    timeout: 180000,
    polling: 250,
  });
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);

  const sample = async (): Promise<Sample> => {
    // average frameMs over a few settles so the reading is not a single spike
    let dag = -1;
    let vis = -1;
    const ms: number[] = [];
    for (let i = 0; i < 8; i++) {
      const v = await page.evaluate(async () => {
        if (window.__laas.settle) await window.__laas.settle(16);
        return {
          dag: window.__laas.stats?.counters['nanite.dagClusters'] ?? -1,
          vis: window.__laas.stats?.counters['nanite.visClusters'] ?? -1,
          frameMs: window.__laas.stats?.frameMs ?? -1,
        };
      });
      if (v.dag >= 0) {
        dag = v.dag;
        vis = v.vis;
        ms.push(v.frameMs);
      }
    }
    ms.sort((a, b) => a - b);
    return { dag, vis, frameMs: ms[Math.floor(ms.length / 2)] ?? -1 };
  };

  const near = await sample();
  const pose = (await page.evaluate(() => window.__laas.getPose?.() ?? null)) as CamPose | null;
  if (pose) {
    const fwd: [number, number, number] = [
      Math.sin(pose.yaw) * Math.cos(pose.pitch),
      Math.sin(pose.pitch),
      -Math.cos(pose.yaw) * Math.cos(pose.pitch),
    ];
    await page.evaluate(
      ({ pose, fwd }) => {
        window.__laas.setPose?.({
          ...pose,
          p: [pose.p[0] - fwd[0] * 300, pose.p[1] - fwd[1] * 300, pose.p[2] - fwd[2] * 300],
        });
      },
      { pose, fwd },
    );
  }
  const far = await sample();
  await browser.close();
  return { near, far };
}

interface Sample {
  dag: number;
  vis: number;
  frameMs: number;
}

async function main(): Promise<void> {
  console.log('[envperf] measuring nanitedag=all at bm4 (near) and −300 m (far):');
  for (const occl of ['1', '0'] as const) {
    const { near, far } = await boot(occl);
    const tag = occl === '1' ? 'occl ON  (real path)' : 'occl OFF (worst case)';
    console.log(
      `  ${tag}  near: ${(near.dag / 1e6).toFixed(2)}M dag / ${(near.vis / 1e6).toFixed(2)}M vis / ${near.frameMs.toFixed(1)} ms` +
        `   far: ${(far.dag / 1e6).toFixed(2)}M dag / ${far.frameMs.toFixed(1)} ms`,
    );
  }
}

main().catch((e) => {
  console.error('[envperf] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
