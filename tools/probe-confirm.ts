/**
 * probe-confirm — ADVERSARIAL CONFIRMER battery.
 *
 * The headline ("~86-92% per-covered-pixel coverage+election; compute-bound;
 * ~linear in pixels & visClusters") is already triangulated by probe-harvest
 * S2-S5. This battery does NOT re-derive it; it runs the CHEAPEST DECISIVE
 * falsifiers for the ONE thing the corpus did NOT separate: the sub-mechanism
 * split WITHIN the per-pixel half — atomic-election vs depth-interp ALU vs the
 * residual coverage-loop. That split decides which class of reshape pays off.
 *
 * Uses ONLY the hardened harness (tools/measure.ts → __laas.measureFrames),
 * the proven S2 methodology (occl=0 ⇒ IDENTICAL cluster set across rdbg levels,
 * dpr2 ⇒ GPU-bound so gpuWall is honest active time, isolated drain→submit→drain).
 *
 *   C0  LOCK     — world1 + combined steady @1280×720 + motion WORST (this thermal state).
 *   C1  SPLIT    — world1 rdbg 1/2/3/5/6 @ dpr2 occl=0 matched set, isolated gpuWall:
 *                    rdbg1 = launch+ctx FLOOR
 *                    rdbg2 = + per-triangle edge-setup
 *                    rdbg3 = full (per-pixel loop ON)
 *                    rdbg5 = full coverage+cz, election → NON-ATOMIC plain store
 *                            ⇒ (rdbg3 − rdbg5) = ATOMIC-ELECTION share
 *                    rdbg6 = full loop+atomics, cz = CONSTANT (skip depth interp)
 *                            ⇒ (rdbg3 − rdbg6) = depth-interp ALU share
 *                  residual coverage-loop = (rdbg5 ∩ rdbg6 floor) − rdbg2.
 *
 * ALL boots serialized — one GPU. Needs the dev server on :5173.
 *   npx tsx tools/probe-confirm.ts
 *   STAGES=C1 npx tsx tools/probe-confirm.ts
 */
import {
  bootPage,
  buildTrack,
  measureActiveGpuSteady,
  median,
  replayMotionTrack,
  type TrackPose,
} from './measure';
import { launchWebGPU } from './launch';
import type { Browser } from 'playwright';

const W = 1280;
const H = 720;
const FRAMES = Number(process.env.FRAMES ?? '30');
const STAGES = (process.env.STAGES ?? 'C0,C1').split(',').map((s) => s.trim());
const ALLEY: TrackPose = { p: [0, 2, 120], yaw: 1.05, pitch: -0.02 };

function forestExtra(kind: 'world1' | 'combined', over: Record<string, string> = {}): Record<string, string> {
  const base: Record<string, string> = {
    nanite: '1',
    trees: process.env.TREES ?? '40000',
    lodnear: '4',
    simband: '6',
    lodpow: '0.6',
    instminpx: '128',
  };
  if (kind === 'combined') base['nanitedbg'] = 'flat';
  return { ...base, ...over };
}

function forestTrack(): ReturnType<typeof buildTrack> {
  const keys: TrackPose[] = [
    { p: [0, 2, 0], yaw: 0.0, pitch: -0.02 },
    { p: [0, 2, 120], yaw: 0.0, pitch: -0.02 },
    { p: [0, 2, 120], yaw: 1.05, pitch: -0.02 },
    { p: [60, 2, 180], yaw: 1.05, pitch: 0.0 },
    { p: [60, 2, 180], yaw: 2.4, pitch: 0.0 },
    { p: [40, 14, 120], yaw: 2.4, pitch: -0.25 },
    { p: [40, 28, 60], yaw: 2.4, pitch: -0.7 },
    { p: [0, 28, 0], yaw: 3.6, pitch: -0.7 },
    { p: [0, 2, -40], yaw: 3.6, pitch: -0.02 },
  ];
  const fps = Number(process.env.TRACK_FPS ?? '12');
  return buildTrack('forest-worst', keys, fps);
}

const medL = (a: number[]): number => {
  const s = a.filter(Number.isFinite).sort((x, y) => x - y);
  return s.length ? (s[s.length >> 1] as number) : 0;
};

function hr(s: string): void {
  console.log(`\n========== ${s} ==========`);
}

const ROWS: Record<string, unknown>[] = [];

async function c0(browser: Browser): Promise<void> {
  hr('C0. LOCK FOUNDATION — steady world1+combined + motion WORST (this thermal state)');
  for (const kind of ['world1', 'combined'] as const) {
    const key = kind === 'combined' ? 'c.nanRasterCombined' : 'c.nanRasterWorld1';
    const page = await bootPage(browser, {
      scene: 'forest', width: W, height: H, extra: forestExtra(kind), settle: 60, logPrefix: '[forest]',
    });
    await page.evaluate((pp) => window.__laas.setPose?.(pp), ALLEY);
    const r = await measureActiveGpuSteady(page, key, { frames: FRAMES, warmup: 12 });
    console.log(
      `   ${kind.padEnd(9)} p50 ${r.p50.toFixed(2)} p95 ${r.p95.toFixed(2)} worst ${r.worst.toFixed(2)} ms ` +
        `| visCl ${r.visClusters} gpuWall ${r.gpuWallMs.toFixed(2)} reps ${r.reps}${r.converged ? '✓' : '✗'} rej ${r.rejected}`,
    );
    ROWS.push({ stage: 'C0', kind, p50: +r.p50.toFixed(2), p95: +r.p95.toFixed(2), worst: +r.worst.toFixed(2), visClusters: r.visClusters });
    await page.close();
  }
  // motion WORST
  const track = forestTrack();
  const page = await bootPage(browser, {
    scene: 'forest', width: W, height: H, freeze: true, extra: forestExtra('world1'), settle: 60, logPrefix: '[forest]',
  });
  await page.evaluate((pp) => window.__laas.setPose?.(pp), track.poses[0]!);
  await measureActiveGpuSteady(page, 'c.nanRasterWorld1', { frames: 12, warmup: 8, maxReps: 5 });
  const r = await replayMotionTrack(page, track, 'c.nanRasterWorld1', 3, 6, 12);
  const noiseMed = median(r.pairedNoise);
  console.log(
    `   motion: p50 ${r.p50.toFixed(2)} p95 ${r.p95.toFixed(2)} WORST ${r.worst.toFixed(2)} ms @pose#${r.worstPoseIndex} ` +
      `(visCl ${r.perPoseVis[r.worstPoseIndex]}) | paired-noise ${noiseMed.toFixed(3)} ms`,
  );
  ROWS.push({ stage: 'C0', kind: 'motion', p50: +r.p50.toFixed(2), worst: +r.worst.toFixed(2), worstPose: r.worstPoseIndex, worstVisCl: r.perPoseVis[r.worstPoseIndex], pairedNoiseMs: +noiseMed.toFixed(3) });
  await page.close();
}

async function c1(browser: Browser): Promise<void> {
  hr('C1. PER-PIXEL-HALF SPLIT — rdbg 1/2/3/5/6 @ dpr2 occl=0 matched set, isolated gpuWall');
  const dpr = process.env.DPR ?? '2';
  const levels = [1, 2, 3, 5, 6];
  const out: Record<number, { ms: number; vis: number }> = {};
  for (const rdbg of levels) {
    const page = await bootPage(browser, {
      scene: 'forest', width: W, height: H,
      extra: forestExtra('world1', { rdbg: String(rdbg), dpr, occl: '0' }), settle: 50, logPrefix: '[forest]',
    });
    await page.evaluate((pp) => window.__laas.setPose?.(pp), ALLEY);
    // a couple of reps; take the lowest-stable median (least throttled = most honest)
    const reps: number[] = [];
    let vis = -1;
    for (let k = 0; k < 3; k++) {
      const fr = (await page.evaluate(
        async (o) => window.__laas.measureFrames?.(o),
        { frames: 14, warmup: 16 },
      )) as { gpuWallMs: number; counters: Record<string, number> }[];
      reps.push(medL(fr.map((f) => f.gpuWallMs)));
      vis = medL(fr.map((f) => f.counters['nanite.visClusters'] ?? -1));
    }
    out[rdbg] = { ms: Math.min(...reps), vis };
    console.log(`   rdbg=${rdbg}: gpuWall ${out[rdbg]!.ms.toFixed(2)} ms  (reps ${reps.map((x) => x.toFixed(1)).join('/')}, visCl ${vis})`);
    await page.close();
  }
  const floor = out[1]!.ms;
  const setup = out[2]!.ms - out[1]!.ms;
  const perPixel = out[3]!.ms - out[2]!.ms;
  const total = out[3]!.ms;
  // (rdbg3 − rdbg5) = atomic-election share of the per-pixel half (rdbg5 keeps coverage+cz, drops atomics)
  const atomicShare = out[3]!.ms - out[5]!.ms;
  // (rdbg3 − rdbg6) = depth-interp ALU share (rdbg6 keeps atomics, drops per-pixel cz interp)
  const depthAluShare = out[3]!.ms - out[6]!.ms;
  // residual coverage-loop (edge-walk + key pack + the aLoadU pre-check that survives both) =
  // per-pixel − atomic − depthALU (lower bound; overlaps possible)
  const residual = perPixel - atomicShare - depthAluShare;
  console.log('\n   ── HEADLINE PARTITION (matched set, isolated gpuWall) ──');
  console.log(`   launch+ctx FLOOR ........ ${floor.toFixed(2)} ms  (${((floor / total) * 100).toFixed(0)}% of ${total.toFixed(1)} ms frame)`);
  console.log(`   per-triangle edge-setup . ${setup.toFixed(2)} ms  (${((setup / total) * 100).toFixed(0)}%)`);
  console.log(`   PER-PIXEL loop .......... ${perPixel.toFixed(2)} ms  (${((perPixel / total) * 100).toFixed(0)}%)`);
  console.log('   ── SPLIT OF THE PER-PIXEL HALF ──');
  console.log(`   atomic-election (rdbg3−rdbg5) ... ${atomicShare.toFixed(2)} ms  (${((atomicShare / perPixel) * 100).toFixed(0)}% of per-pixel, ${((atomicShare / total) * 100).toFixed(0)}% of frame)`);
  console.log(`   depth-interp ALU (rdbg3−rdbg6) .. ${depthAluShare.toFixed(2)} ms  (${((depthAluShare / perPixel) * 100).toFixed(0)}% of per-pixel, ${((depthAluShare / total) * 100).toFixed(0)}% of frame)`);
  console.log(`   residual coverage-loop ......... ${residual.toFixed(2)} ms  (${((residual / perPixel) * 100).toFixed(0)}% of per-pixel)`);
  console.log(`   [rdbg5 ${out[5]!.ms.toFixed(2)} ms = floor+setup+coverage+cz, NO atomics] [rdbg6 ${out[6]!.ms.toFixed(2)} ms = floor+setup+coverage+atomics, const cz]`);
  ROWS.push({
    stage: 'C1', dpr, visClusters: out[3]!.vis,
    floor_ms: +floor.toFixed(2), setup_ms: +setup.toFixed(2), perPixel_ms: +perPixel.toFixed(2), frame_ms: +total.toFixed(2),
    rdbg5_ms: +out[5]!.ms.toFixed(2), rdbg6_ms: +out[6]!.ms.toFixed(2),
    atomicElection_ms: +atomicShare.toFixed(2), depthAlu_ms: +depthAluShare.toFixed(2), residualCoverage_ms: +residual.toFixed(2),
    atomicPctOfPerPixel: +((atomicShare / perPixel) * 100).toFixed(0),
    depthAluPctOfPerPixel: +((depthAluShare / perPixel) * 100).toFixed(0),
    residualPctOfPerPixel: +((residual / perPixel) * 100).toFixed(0),
  });
}

async function main(): Promise<void> {
  const { browser } = await launchWebGPU();
  console.log(`[confirm] stages=${STAGES.join(',')} frames=${FRAMES} @ ${W}×${H} trees=${process.env.TREES ?? '40000'}`);
  if (STAGES.includes('C0')) await c0(browser);
  if (STAGES.includes('C1')) await c1(browser);
  await browser.close();
  console.log('\n========== CONFIRM CORPUS (json) ==========');
  console.log(JSON.stringify(ROWS, null, 2));
  console.log('\n[confirm] done.');
}
main().catch((e: unknown) => {
  console.error('[confirm] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
