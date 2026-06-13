/**
 * N8-D1c GATE — continuous-LOD DAG cut under MOTION (no pop, no cracks, stable
 * counts). Boots the world with the rock pools DAG'd (?nanitedag=rock) at the
 * bm4 boulder framing (a big foreground rock = the DAG'd hero), occlusion off
 * for a deterministic count, then exercises the screen-error cut two ways:
 *
 *   τ-SWEEP (fixed pose): tighten τ from coarse→fine via the live setTau hook.
 *   nanite.dagClusters (emitted rock-DAG clusters ONLY — terrain/bark are
 *   flag-0) must rise MONOTONICALLY as τ tightens (the cut frontier walks toward
 *   LOD0) and actually span a range (the cut is doing something).
 *
 *   ZOOM-SWEEP (τ=1): dolly the camera along its view ray; dagClusters must
 *   move SMOOTHLY frame to frame — a big jump = a LOD pop, the thing the
 *   continuous cut exists to avoid.
 *
 * Frames → shots/wip/zoom-* for an eyeball pass (cracks = holes in the rock
 * silhouette between LOD bands). Crack-FREENESS is proven in probe-dag (locked
 * boundaries + bit-exact sibling pairs); this gate confirms the GPU cut realises
 * it without pops/holes/errors.
 *
 *   npx tsx tools/probe-zoom.ts            # needs the dev server on :5173
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
    extra: { nanite: '1', nanitedag: 'rock', nanshadow: '0', occl: '0', shot: '4', loderr: '1' },
  });
  console.log(`[zoom] ${url}`);
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__laas && (window.__laas.ready || window.__laas.error != null), undefined, {
    timeout: 180000,
    polling: 250,
  });
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);
  const hasTau = await page.evaluate(() => typeof window.__laasNanite?.setTau === 'function');
  if (!hasTau) throw new Error('window.__laasNanite.setTau missing — DAG cut not wired');

  // settle + read the rock-DAG cluster counter (15-frame async readback — retry
  // until it lands).
  const readDagCount = async (): Promise<number> => {
    for (let tries = 0; tries < 12; tries++) {
      const v = await page.evaluate(async () => {
        if (window.__laas.settle) await window.__laas.settle(16);
        return window.__laas.stats?.counters['nanite.dagClusters'] ?? -1;
      });
      if (v >= 0) return v;
    }
    return -1;
  };

  // --- τ-SWEEP (fixed pose) ---
  console.log('[zoom] τ-sweep (fixed bm4 boulder pose, occl off):');
  const taus = [32, 16, 8, 4, 2, 1, 0.5, 0.25];
  const dagByTau: number[] = [];
  for (const t of taus) {
    await page.evaluate((tau) => window.__laasNanite?.setTau?.(tau), t);
    const n = await readDagCount();
    dagByTau.push(n);
    await page.screenshot({ path: `shots/wip/zoom-tau-${String(t).replace('.', 'p')}.png` });
    console.log(`  τ=${String(t).padStart(5)}  dagClusters ${n}`);
  }
  for (let i = 1; i < dagByTau.length; i++) {
    const prev = dagByTau[i - 1] as number;
    const cur = dagByTau[i] as number;
    if (cur < prev - 1) fail(`τ-sweep non-monotonic at τ=${taus[i]}: ${cur} < ${prev} (tighter τ shed clusters)`);
  }
  const span = (dagByTau[dagByTau.length - 1] as number) - (dagByTau[0] as number);
  if (span <= 0) fail(`τ-sweep flat: cut never refined (${dagByTau[0]} → ${dagByTau[dagByTau.length - 1]})`);

  // --- ZOOM-SWEEP (τ=1, dolly along the view ray) ---
  await page.evaluate(() => window.__laasNanite?.setTau?.(1));
  const pose0 = (await page.evaluate(() => window.__laas.getPose?.() ?? null)) as CamPose | null;
  if (pose0) {
    console.log('[zoom] zoom-sweep (dolly along view ray, τ=1):');
    const fwd: [number, number, number] = [
      Math.sin(pose0.yaw) * Math.cos(pose0.pitch),
      Math.sin(pose0.pitch),
      -Math.cos(pose0.yaw) * Math.cos(pose0.pitch),
    ];
    const dagByStep: number[] = [];
    for (let s = 0; s < 7; s++) {
      const d = s * 1.8; // metres along the ray
      await page.evaluate(
        ({ pose, fwd, d }) => {
          window.__laas.setPose?.({
            ...pose,
            p: [pose.p[0] + fwd[0] * d, pose.p[1] + fwd[1] * d, pose.p[2] + fwd[2] * d],
          });
        },
        { pose: pose0, fwd, d },
      );
      const n = await readDagCount();
      dagByStep.push(n);
      await page.screenshot({ path: `shots/wip/zoom-step-${s}.png` });
      console.log(`  +${d.toFixed(1)}m  dagClusters ${n}`);
    }
    // SMOOTHNESS: no LOD pop — consecutive steps change by < 60% of the larger.
    // (Direction-agnostic: dolly sign doesn't matter, only that it's continuous.)
    for (let i = 1; i < dagByStep.length; i++) {
      const a = dagByStep[i - 1] as number;
      const b = dagByStep[i] as number;
      const jump = Math.abs(b - a) / Math.max(1, a, b);
      if (jump > 0.6) fail(`zoom-sweep POP at step ${i}: ${a} → ${b} (${(jump * 100).toFixed(0)}% jump)`);
    }
  }

  if (errs.length) fail(`${errs.length} console/page errors: ${errs.slice(0, 3).join(' | ')}`);
  await browser.close();

  if (failures > 0) {
    console.error(`[zoom] ${failures} FAILURES`);
    process.exit(1);
  }
  console.log('[zoom] continuous-LOD cut OK — monotonic refine, smooth zoom, no errors');
}

main().catch((e) => {
  console.error('[zoom] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
