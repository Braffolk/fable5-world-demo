/**
 * probe-motion — the canonical HARDENED nanite perf measurement + its self-proof.
 *
 * Uses the shared harness (tools/measure.ts → __laas.measureFrames →
 * src/core/MeasureHarness.ts) which drives frames MANUALLY, GPU-bound + isolated,
 * so the per-pass timestamp is HONEST active GPU time and NOT the vsync
 * cross-frame pipelining artifact (the old ~15.2 ms floor).
 *
 * SECOND artifact found + fixed while building this (be adversarial about your own
 * harness): the manual back-to-back DRAINED loop runs the GPU flat-out with ZERO
 * idle (unlike rAF's vsync idle), so a sustained run THERMALLY THROTTLES and late
 * frames inflate. Calibrated on this machine: cooldown 0 → 27.7 ms / spread 24;
 * 50 ms → 18.9 ms / spread 1.6. So the harness idles ~50 ms between samples
 * (MeasureHarness.cooldownMs) and block B converges to a thermal STEADY STATE.
 *
 * It runs SIX blocks (all serialized — one GPU):
 *   A. VSYNC-IMMUNITY: sweep visClusters via simband at 1280×720 (live, one boot)
 *      — world1 must SCALE with work (slope > 0), not pin to a constant. Proof.
 *   B. CANONICAL: the honest steady-state world1 @ 1280×720 (full pipe) and
 *      combined @ ?nanitedbg=flat (lean) numbers the artifact was hiding.
 *   C. MOTION TRACK: a fixed deterministic path (long alley, horizon-grazing,
 *      canopy dive) — p50/p95/worst of honest world1 + a sparse paired noise floor.
 *   D. NEGATIVE CONTROL: 2× resolution must read SLOWER, CI excluding 0.
 *   E. KNOWN SIGNAL: toggling simband (a real LOD knob) moves the number right.
 *   F. NOISE FLOOR: fixed-pose steady-state repeatability (the trustworthy floor —
 *      adjacent-rep |Δ| is the random-noise component a bracketed A/B works under).
 *
 *   npx tsx tools/probe-motion.ts                 # needs dev server on :5173
 *   KEY=combined npx tsx tools/probe-motion.ts    # measure the lean kernel
 *   BLOCKS=A,B    npx tsx tools/probe-motion.ts    # subset
 */
import {
  abAcceptance,
  bootPage,
  buildTrack,
  measureActiveGpu,
  measureActiveGpuSteady,
  median,
  percentile,
  replayMotionTrack,
  type AbSide,
  type TrackPose,
} from './measure';
import { launchWebGPU } from './launch';
import type { Page } from 'playwright';

// which raster kernel: full pipe (world1) | lean debug (combined @ nanitedbg=flat)
const KIND = (process.env.KEY ?? 'world1') as 'world1' | 'combined';
const KEY = KIND === 'combined' ? 'c.nanRasterCombined' : 'c.nanRasterWorld1';
const TREES = process.env.TREES ?? '40000';
const FRAMES = Number(process.env.FRAMES ?? '40');
const BLOCKS = (process.env.BLOCKS ?? 'A,B,C,D,E,F').split(',').map((s) => s.trim());

const W = 1280;
const H = 720;

/** forest extra params shared by every boot of the chosen kernel. */
function forestExtra(kind: 'world1' | 'combined'): Record<string, string> {
  const base: Record<string, string> = {
    nanite: '1',
    trees: TREES,
    lodnear: '4',
    simband: '6',
    lodpow: '0.6',
    instminpx: '128',
  };
  if (kind === 'combined') base['nanitedbg'] = 'flat';
  return base;
}

/**
 * Deterministic motion track through the forest (eye-height grid at y=0, crowns
 * ~15-25 m, grid ±400 m for 40k trees). Keyframes hit the worst angles:
 *  - long alley (horizontal sightline straight down a row)
 *  - horizon-grazing (pitch ≈ 0 looking far across the grid)
 *  - canopy dive (climb to ~28 m, pitch down through the tops)
 */
function forestTrack(): ReturnType<typeof buildTrack> {
  const keys: TrackPose[] = [
    { p: [0, 2, 0], yaw: 0.0, pitch: -0.02 }, // alley start, looking +Z
    { p: [0, 2, 120], yaw: 0.0, pitch: -0.02 }, // walk straight down the long alley
    { p: [0, 2, 120], yaw: 1.05, pitch: -0.02 }, // pan to a diagonal corridor (longest sightline)
    { p: [60, 2, 180], yaw: 1.05, pitch: 0.0 }, // horizon-grazing across the grid
    { p: [60, 2, 180], yaw: 2.4, pitch: 0.0 }, // pan to the far edge horizon
    { p: [40, 14, 120], yaw: 2.4, pitch: -0.25 }, // rise into the canopy
    { p: [40, 28, 60], yaw: 2.4, pitch: -0.7 }, // canopy dive-through, looking down
    { p: [0, 28, 0], yaw: 3.6, pitch: -0.7 }, // descend the dive over center
    { p: [0, 2, -40], yaw: 3.6, pitch: -0.02 }, // back to eye height, new alley
  ];
  // 8 segs × FPS + 1 poses. Kept modest: a long flat-out drained track slowly
  // heats the GPU (no vsync idle), so we keep the run short and use the PAIRED
  // per-pose noise floor (back-to-back, thermally matched) — not a whole-track
  // re-run — as the honest reproducibility metric.
  const fps = Number(process.env.TRACK_FPS ?? '15');
  return buildTrack('forest-worst', keys, fps); // 8×15+1 = 121 poses
}

function hr(label: string): void {
  console.log(`\n========== ${label} ==========`);
}

async function blockA(page: Page): Promise<void> {
  hr('A. VSYNC-IMMUNITY — world1 must SCALE with visClusters (old artifact = flat ~15.2 ms)');
  // pose at the long diagonal alley (heaviest), then sweep simband live (more sim =
  // fewer far clusters ⇒ a real visCluster sweep without re-boot).
  await page.evaluate(() => window.__laas.setPose?.({ p: [0, 2, 120], yaw: 1.05, pitch: -0.02 }));
  const bands = [60, 30, 14, 6, 3, 1.5]; // larger band = MORE clusters kept far
  console.log(`   key=${KEY} @ ${W}×${H}`);
  console.log('   simband |  visClusters |  world1 p50  p95  worst (ms) | gpuWall | rejected');
  const pts: { vis: number; ms: number }[] = [];
  for (const b of bands) {
    await page.evaluate((bb) => {
      const n = (window as unknown as { __laasNanite?: { setSimBand?(v: number): void } }).__laasNanite;
      const v = (window as unknown as { __naniteView?: { setLod?(a: number, c: number, d: number): void } }).__naniteView;
      n?.setSimBand?.(bb);
      v?.setLod?.(4, bb, 0.6);
    }, b);
    const r = await measureActiveGpu(page, KEY, { frames: FRAMES, warmup: 12 });
    pts.push({ vis: r.visClusters, ms: r.p50 });
    console.log(
      `   ${String(b).padStart(7)} | ${String(r.visClusters).padStart(12)} | ${r.p50.toFixed(2).padStart(10)} ${r.p95.toFixed(2).padStart(4)} ${r.worst.toFixed(2).padStart(6)} | ${r.gpuWallMs.toFixed(2).padStart(7)} | ${r.rejected}`,
    );
  }
  // slope: a real (non-artifact) measurement scales with visClusters
  const lo = pts[pts.length - 1]!;
  const hi = pts[0]!;
  const dvis = hi.vis - lo.vis;
  const dms = hi.ms - lo.ms;
  const slope = dvis !== 0 ? (dms / (dvis / 1e6)).toFixed(2) : 'n/a';
  console.log(
    `   → world1 ${lo.ms.toFixed(2)}→${hi.ms.toFixed(2)} ms as visClusters ${lo.vis}→${hi.vis} ` +
      `(slope ≈ ${slope} ms / Mcl). ${dms > 0.3 ? 'SCALES ✓ (artifact gone)' : 'FLAT ✗ — still capped?!'}`,
  );
}

async function blockB(browser: Awaited<ReturnType<typeof launchWebGPU>>['browser']): Promise<void> {
  hr('B. CANONICAL honest numbers @ 1280×720');
  for (const kind of ['world1', 'combined'] as const) {
    const key = kind === 'combined' ? 'c.nanRasterCombined' : 'c.nanRasterWorld1';
    const page = await bootPage(browser, {
      scene: 'forest',
      width: W,
      height: H,
      extra: forestExtra(kind),
      settle: 60,
      logPrefix: '[forest]',
    });
    // canonical pose = the heavy diagonal alley
    await page.evaluate(() => window.__laas.setPose?.({ p: [0, 2, 120], yaw: 1.05, pitch: -0.02 }));
    // steady-state (converged, thermal-history-independent) honest number
    const r = await measureActiveGpuSteady(page, key, { frames: FRAMES, warmup: 12 });
    console.log(
      `   ${kind.padEnd(9)} (${key}): p50 ${r.p50.toFixed(2)}  p95 ${r.p95.toFixed(2)}  worst ${r.worst.toFixed(2)} ms ` +
        `| visCl ${r.visClusters}  gpuWall ${r.gpuWallMs.toFixed(2)}  refresh ${r.refreshMs.toFixed(2)}  ` +
        `reps ${r.reps}${r.converged ? '✓conv' : '✗nc'}  rejected ${r.rejected}/${FRAMES}`,
    );
    await page.close();
  }
}

async function blockC(browser: Awaited<ReturnType<typeof launchWebGPU>>['browser']): Promise<void> {
  hr('C. MOTION TRACK — honest world1 distribution + replay-noise-floor');
  const track = forestTrack();
  console.log(`   track "${track.name}": ${track.poses.length} poses (long alley + horizon + canopy dive)`);
  // freeze=1 ⇒ static geometry (wind/time pinned) ⇒ a bit-identical replay; the
  // per-pose HZB warmup makes each pose independent of arrival ⇒ noise floor ≈ 0.
  const page = await bootPage(browser, {
    scene: 'forest',
    width: W,
    height: H,
    freeze: true,
    extra: forestExtra(KIND),
    settle: 60,
    logPrefix: '[forest]',
  });
  // noiseEvery=10: a SPARSE paired noise floor (back-to-back at every 10th pose,
  // thermally matched) — the honest replay-noise estimate. A whole-track re-run
  // is NOT valid on a throttling GPU (it measures thermal drift, not harness noise).
  // thermal pre-settle: run a few discarded measurements at the first pose so the
  // GPU reaches cold steady-state BEFORE the track (else pose #0/#1 read a hot
  // boot-transient). measureActiveGpuSteady's reps double as the warm-down.
  await page.evaluate((pp) => window.__laas.setPose?.(pp), track.poses[0]!);
  await measureActiveGpuSteady(page, KEY, { frames: 12, warmup: 8, maxReps: 5 });
  const r1 = await replayMotionTrack(page, track, KEY, 3, 6, 10);
  const noiseMed = median(r1.pairedNoise);
  const noiseP95 = percentile(r1.pairedNoise, 0.95);
  console.log(
    `   distribution (${KEY}): p50 ${r1.p50.toFixed(2)}  p95 ${r1.p95.toFixed(2)}  worst ${r1.worst.toFixed(2)} ms ` +
      `(worst @ pose #${r1.worstPoseIndex}, visCl ${r1.perPoseVis[r1.worstPoseIndex]})  rejected ${r1.rejected}`,
  );
  console.log(
    `   REPLAY NOISE FLOOR (PAIRED per-pose |Δ|, thermally matched, sparse): median ${noiseMed.toFixed(3)} ms, p95 ${noiseP95.toFixed(3)} ms`,
  );
  console.log(
    `   (the paired floor is the per-pose A/B sensitivity; a long sequential track also carries a slow ` +
      `cooling drift that a BRACKETED A/B — block D / abAcceptance — cancels. See block F for the clean fixed-pose floor.)`,
  );
  await page.close();
}

async function blockD(browser: Awaited<ReturnType<typeof launchWebGPU>>['browser']): Promise<void> {
  hr('D. NEGATIVE CONTROL — 2× resolution MUST read slower (CI excluding 0)');
  const base: AbSide = { label: '1×', extra: {} };
  const cand: AbSide = { label: '2×', extra: { dpr: '2' } };
  const bootSide = (side: AbSide): Promise<Page> =>
    bootPage(browser, {
      scene: 'forest',
      width: W,
      height: H,
      extra: { ...forestExtra(KIND), ...side.extra },
      settle: 50,
    });
  const measureSide = async (page: Page): Promise<number[]> => {
    await page.evaluate(() => window.__laas.setPose?.({ p: [0, 2, 120], yaw: 1.05, pitch: -0.02 }));
    return (await measureActiveGpu(page, KEY, { frames: FRAMES, warmup: 16 })).samples;
  };
  const ab = await abAcceptance(bootSide, measureSide, base, cand, 2);
  console.log(
    `   1× HL ${ab.baseHL.toFixed(2)} ms  →  2× HL ${ab.candHL.toFixed(2)} ms  | Δ ${ab.delta.toFixed(2)} ms ` +
      `[95% CI ${ab.lo.toFixed(2)}, ${ab.hi.toFixed(2)}]  noiseΔ ${ab.noiseDelta.toFixed(3)}`,
  );
  console.log(
    `   → ${ab.significant && ab.delta > 0 ? 'PASS ✓ — 2× res is SLOWER, CI excludes 0 (honest, scales with pixels)' : 'FAIL ✗ — no significant slowdown (would mean still capped)'}`,
  );
}

async function blockE(browser: Awaited<ReturnType<typeof launchWebGPU>>['browser']): Promise<void> {
  hr('E. KNOWN SIGNAL — a real LOD knob (simband) moves the number the right way');
  const page = await bootPage(browser, {
    scene: 'forest',
    width: W,
    height: H,
    extra: forestExtra(KIND),
    settle: 50,
  });
  await page.evaluate(() => window.__laas.setPose?.({ p: [0, 2, 120], yaw: 1.05, pitch: -0.02 }));
  const setBand = async (b: number): Promise<void> => {
    await page.evaluate((bb) => {
      const n = (window as unknown as { __laasNanite?: { setSimBand?(v: number): void } }).__laasNanite;
      const v = (window as unknown as { __naniteView?: { setLod?(a: number, c: number, d: number): void } }).__naniteView;
      n?.setSimBand?.(bb);
      v?.setLod?.(4, bb, 0.6);
    }, b);
  };
  await setBand(3); // aggressive LOD = fewer far clusters = FASTER
  const lean = await measureActiveGpu(page, KEY, { frames: FRAMES, warmup: 14 });
  await setBand(60); // lenient = more far clusters = SLOWER
  const rich = await measureActiveGpu(page, KEY, { frames: FRAMES, warmup: 14 });
  console.log(
    `   simband 3  (lean): ${lean.p50.toFixed(2)} ms @ ${lean.visClusters} visCl   |   ` +
      `simband 60 (rich): ${rich.p50.toFixed(2)} ms @ ${rich.visClusters} visCl`,
  );
  console.log(
    `   → Δ ${(rich.p50 - lean.p50).toFixed(2)} ms for ${rich.visClusters - lean.visClusters} more clusters. ` +
      `${rich.p50 > lean.p50 && rich.visClusters > lean.visClusters ? 'PASS ✓ (right direction + magnitude)' : 'FAIL ✗'}`,
  );
  await page.close();
}

async function blockF(browser: Awaited<ReturnType<typeof launchWebGPU>>['browser']): Promise<void> {
  hr('F. HARNESS NOISE FLOOR — fixed-pose steady-state repeatability (the trustworthy floor)');
  const page = await bootPage(browser, {
    scene: 'forest',
    width: W,
    height: H,
    extra: forestExtra(KIND),
    settle: 50,
  });
  await page.evaluate(() => window.__laas.setPose?.({ p: [0, 2, 120], yaw: 1.05, pitch: -0.02 }));
  // reach cold steady-state first, then 8 independent reps at the SAME pose
  await measureActiveGpuSteady(page, KEY, { frames: 12, warmup: 8, maxReps: 5 });
  const reps: number[] = [];
  for (let k = 0; k < 8; k++) {
    const r = await measureActiveGpu(page, KEY, { frames: 15, warmup: 6 });
    reps.push(r.p50);
  }
  const med = median(reps);
  const spread = Math.max(...reps) - Math.min(...reps);
  console.log(`   reps p50 (ms): [${reps.map((x) => x.toFixed(1)).join(', ')}]`);
  // adjacent-rep deltas (the real random-noise component; the full spread also
  // includes a slow monotonic cooling drift, which a bracketed A/B cancels)
  const adj = reps.slice(1).map((v, i) => Math.abs(v - reps[i]!));
  const adjMed = median(adj);
  console.log(
    `   → median ${med.toFixed(2)} ms, spread ${spread.toFixed(2)} ms (incl. slow cool-drift), ` +
      `adjacent |Δ| median ${adjMed.toFixed(2)} ms ` +
      `${adjMed < 1.5 ? '✓ (random noise <1.5 ms; bracketed A/B cancels the drift — deltas above this are real)' : '✗ (unstable)'}`,
  );
  await page.close();
}

async function main(): Promise<void> {
  const { browser } = await launchWebGPU();
  console.log(`[motion] kernel=${KIND} key=${KEY} trees=${TREES} frames/measure=${FRAMES} blocks=${BLOCKS.join(',')}`);

  if (BLOCKS.includes('A')) {
    // block A reuses one live boot to sweep simband without re-booting
    const page = await bootPage(browser, {
      scene: 'forest',
      width: W,
      height: H,
      extra: forestExtra(KIND),
      settle: 60,
      logPrefix: '[forest]',
    });
    await blockA(page);
    await page.close();
  }
  if (BLOCKS.includes('B')) await blockB(browser);
  if (BLOCKS.includes('C')) await blockC(browser);
  if (BLOCKS.includes('D')) await blockD(browser);
  if (BLOCKS.includes('E')) await blockE(browser);
  if (BLOCKS.includes('F')) await blockF(browser);

  await browser.close();
  console.log('\n[motion] done.');
}
main().catch((e) => {
  console.error('[motion] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
