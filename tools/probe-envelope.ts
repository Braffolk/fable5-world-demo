/**
 * N8-D1 ENVELOPE regression (end-to-end GPU) — proves a DAG'd mesh draws out to
 * its full INTENDED draw envelope (the chain's max distance: trees 496 m, rocks
 * clsMaxDist), instead of being dropped at the stale chain-SWITCH distance that
 * attachDag used to leave behind, which made whole instances wink out.
 *
 * The bug: attachDag retired the LOD chain (lodNext=NONE) but kept the head's
 * switch distance (tree R0_FAR=26 m, rock EX_R1_FAR=120 m). Since the head is now
 * the chain tail, the cull envelope rule `lodNext==NONE && lodDist>0 && dist>lodDist`
 * dropped the ENTIRE instance past that — trees vanishing at ~26 m, rocks ~120 m.
 * The fix makes attachDag inherit the chain's MAX distance (setMaxDistance value).
 *
 * Decisive pose: boot the boulder field (?nanitedag=all, occl OFF for a
 * deterministic count), then PULL THE CAMERA BACK 300 m along its view ray so
 * every visible object sits well beyond the old 26/120 m bug envelope (but inside
 * trees' 496 m intended envelope). Pre-fix that frame collapses to ~0 DAG clusters
 * (all envelope-dropped); post-fix the field still draws (the per-cluster cut just
 * thins each object to its coarse frontier — trees dominate the far count).
 *
 * Frames → shots/wip/envelope-{near,far}.png. Reports frameMs (occl-OFF worst
 * case — the real HZB frame is far lighter).
 *
 *   npx tsx tools/probe-envelope.ts          # needs the dev server on :5173
 */

import type { CamPose } from '../src/core/Hooks';
import { launchWebGPU, laasUrl } from './launch';

declare global {
  interface Window {
    __laasNanite?: { setTau?(v: number): void; tau?(): number };
  }
}

let failures = 0;
const fail = (m: string): void => {
  failures++;
  console.error(`  FAIL ${m}`);
};

async function main(): Promise<void> {
  const width = 1280;
  const height = 720;
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  const errs: string[] = [];
  page.on('console', (m) => {
    if (m.type() === 'error') errs.push(m.text());
  });
  page.on('pageerror', (e) => errs.push('PAGEERR ' + e.message));

  const url = laasUrl({
    scene: 'world',
    width,
    height,
    freeze: true,
    extra: { nanite: '1', nanitedag: 'all', nandbg: 'cluster', nanshadow: '0', occl: '0', shot: '4' },
  });
  console.log(`[envelope] ${url}`);
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__laas && (window.__laas.ready || window.__laas.error != null), undefined, {
    timeout: 180000,
    polling: 250,
  });
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);

  // settle + read the DAG-cluster counter and frame cost (async readback — retry
  // until the counter lands).
  const sample = async (): Promise<{ dag: number; frameMs: number }> => {
    for (let tries = 0; tries < 12; tries++) {
      const v = await page.evaluate(async () => {
        if (window.__laas.settle) await window.__laas.settle(16);
        return {
          dag: window.__laas.stats?.counters['nanite.dagClusters'] ?? -1,
          frameMs: window.__laas.stats?.frameMs ?? -1,
        };
      });
      if (v.dag >= 0) return v;
    }
    return { dag: -1, frameMs: -1 };
  };

  // --- NEAR (bm4 boulder pose) ---
  const near = await sample();
  await page.screenshot({ path: 'shots/wip/envelope-near.png' });
  console.log(`[envelope] near (bm4)        dagClusters ${near.dag}  frameMs ${near.frameMs.toFixed(1)}`);

  // --- FAR: pull the camera back 300 m so everything is beyond the old envelope ---
  const pose = (await page.evaluate(() => window.__laas.getPose?.() ?? null)) as CamPose | null;
  if (!pose) throw new Error('getPose missing — cannot run the far-pose regression');
  const fwd: [number, number, number] = [
    Math.sin(pose.yaw) * Math.cos(pose.pitch),
    Math.sin(pose.pitch),
    -Math.cos(pose.yaw) * Math.cos(pose.pitch),
  ];
  const back = 300;
  await page.evaluate(
    ({ pose, fwd, back }) => {
      window.__laas.setPose?.({
        ...pose,
        p: [pose.p[0] - fwd[0] * back, pose.p[1] - fwd[1] * back, pose.p[2] - fwd[2] * back],
      });
    },
    { pose, fwd, back },
  );
  const far = await sample();
  await page.screenshot({ path: 'shots/wip/envelope-far.png' });
  console.log(`[envelope] far (−300 m back)  dagClusters ${far.dag}  frameMs ${far.frameMs.toFixed(1)}`);

  // DECISIVE: every object is now >120 m away. Pre-fix the envelope rule dropped
  // them all → ~0. Post-fix the field still draws (thinned to coarse frontiers).
  if (far.dag < 1000)
    fail(`far DAG clusters collapsed to ${far.dag} (<1000) — instances envelope-dropped past their stale switch distance`);

  if (errs.length) fail(`${errs.length} console/page errors: ${errs.slice(0, 3).join(' | ')}`);
  await browser.close();

  if (failures > 0) {
    console.error(`[envelope] ${failures} FAILURES`);
    process.exit(1);
  }
  console.log('[envelope] DAG draws past the old 26/120 m bug envelope — intended max envelope holds');
}

main().catch((e) => {
  console.error('[envelope] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
