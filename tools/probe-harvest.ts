/**
 * probe-harvest — MEASUREMENT-HARVEST battery for the nanite forest raster.
 *
 * Uses ONLY the hardened harness (tools/measure.ts → __laas.measureFrames). It
 * does NOT re-run the old contaminated probe-motion battery wholesale; it LOCKS
 * the honest foundation, then runs axis-SEPARATING sweeps to characterize WHERE
 * the per-pixel time goes:
 *
 *   S0  BASELINE        — honest steady-state world1 + combined @ 1280×720 (proves harness).
 *   S1  MOTION DIST     — replayMotionTrack p50/p95/WORST (the target is the worst frame).
 *   S2  RDBG SPLIT      — world1 ?rdbg=1/2/3 at 2× res (GPU-bound) ⇒ fixed-per-cluster
 *                         (launch+ctx) vs per-triangle edge-setup vs per-pixel loop.
 *                         CONFIRMS the per-pixel ~90% claim on honest readings.
 *   S3  RES SWEEP       — fixed work, vary resolution 0.75×/1×/1.41×/2× ⇒ is the cost
 *                         ∝ pixels (per-pixel/compute-bound) or sub-quadratic (fixed/bw)?
 *   S4  VISCL SWEEP     — simband sweep ⇒ fixed-cost intercept vs per-cluster slope
 *                         (linear fit: ms = a + b·visCl). The intercept = the irreducible
 *                         fixed half; the slope·visCl = the scaling per-cluster half.
 *   S5  OVERDRAW        — ?audit=1 covered-fragment volume vs lit pixels ⇒ the overdraw
 *                         factor (how much of the per-pixel 90% is occluded/redundant).
 *
 * ALL boots serialized — one GPU. Each sweep prints rows + a one-line TREND.
 *
 *   npx tsx tools/probe-harvest.ts                 # needs dev server on :5173
 *   STAGES=S0,S1 npx tsx tools/probe-harvest.ts    # subset
 */
import {
  bootPage,
  buildTrack,
  measureActiveGpu,
  measureActiveGpuSteady,
  median,
  replayMotionTrack,
  type TrackPose,
} from './measure';
import { launchWebGPU } from './launch';
import type { Browser } from 'playwright';
import type { Page } from 'playwright';

const W = 1280;
const H = 720;
const FRAMES = Number(process.env.FRAMES ?? '30');
const STAGES = (process.env.STAGES ?? 'S0,S1,S2,S3,S4,S5').split(',').map((s) => s.trim());

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

interface Row {
  config: string;
  metrics: Record<string, number>;
  notes: string;
}
const ROWS: Row[] = [];
function row(config: string, metrics: Record<string, number>, notes = ''): void {
  ROWS.push({ config, metrics, notes });
}

function hr(s: string): void {
  console.log(`\n========== ${s} ==========`);
}

async function setBand(page: Page, b: number): Promise<void> {
  await page.evaluate((bb) => {
    const n = (window as unknown as { __laasNanite?: { setSimBand?(v: number): void } }).__laasNanite;
    const v = (window as unknown as { __naniteView?: { setLod?(a: number, c: number, d: number): void } }).__naniteView;
    n?.setSimBand?.(bb);
    v?.setLod?.(4, bb, 0.6);
  }, b);
}

// least-squares fit y = a + b·x over (x,y) points
function linfit(xs: number[], ys: number[]): { a: number; b: number; r2: number } {
  const n = xs.length;
  const sx = xs.reduce((p, c) => p + c, 0);
  const sy = ys.reduce((p, c) => p + c, 0);
  const sxx = xs.reduce((p, c) => p + c * c, 0);
  const sxy = xs.reduce((p, c, i) => p + c * ys[i]!, 0);
  const d = n * sxx - sx * sx;
  const b = d !== 0 ? (n * sxy - sx * sy) / d : 0;
  const a = (sy - b * sx) / n;
  const my = sy / n;
  const ssTot = ys.reduce((p, c) => p + (c - my) * (c - my), 0);
  const ssRes = ys.reduce((p, c, i) => p + (c - (a + b * xs[i]!)) * (c - (a + b * xs[i]!)), 0);
  const r2 = ssTot !== 0 ? 1 - ssRes / ssTot : 0;
  return { a, b, r2 };
}

// ---------------------------------------------------------------------------
async function s0(browser: Browser): Promise<void> {
  hr('S0. BASELINE — honest steady-state @ 1280×720 (proves the harness)');
  for (const kind of ['world1', 'combined'] as const) {
    const key = kind === 'combined' ? 'c.nanRasterCombined' : 'c.nanRasterWorld1';
    const page = await bootPage(browser, {
      scene: 'forest', width: W, height: H, extra: forestExtra(kind), settle: 60, logPrefix: '[forest]',
    });
    await page.evaluate((pp) => window.__laas.setPose?.(pp), ALLEY);
    const r = await measureActiveGpuSteady(page, key, { frames: FRAMES, warmup: 12 });
    console.log(
      `   ${kind.padEnd(9)} (${key}): p50 ${r.p50.toFixed(2)} p95 ${r.p95.toFixed(2)} worst ${r.worst.toFixed(2)} ms ` +
        `| visCl ${r.visClusters} gpuWall ${r.gpuWallMs.toFixed(2)} refresh ${r.refreshMs.toFixed(1)} ` +
        `reps ${r.reps}${r.converged ? '✓' : '✗nc'} rej ${r.rejected}`,
    );
    row(`S0/${kind}@1280x720/alley`, {
      p50: +r.p50.toFixed(2), p95: +r.p95.toFixed(2), worst: +r.worst.toFixed(2),
      visClusters: r.visClusters, gpuWallMs: +r.gpuWallMs.toFixed(2), refreshMs: +r.refreshMs.toFixed(1),
    }, `steady reps=${r.reps} conv=${r.converged} rej=${r.rejected}`);
    await page.close();
  }
}

async function s1(browser: Browser): Promise<void> {
  hr('S1. MOTION DISTRIBUTION — honest world1 p50/p95/WORST over the worst-angle track');
  const track = forestTrack();
  const page = await bootPage(browser, {
    scene: 'forest', width: W, height: H, freeze: true, extra: forestExtra('world1'), settle: 60, logPrefix: '[forest]',
  });
  await page.evaluate((pp) => window.__laas.setPose?.(pp), track.poses[0]!);
  await measureActiveGpuSteady(page, 'c.nanRasterWorld1', { frames: 12, warmup: 8, maxReps: 5 });
  const r = await replayMotionTrack(page, track, 'c.nanRasterWorld1', 3, 6, 12);
  const noiseMed = median(r.pairedNoise);
  console.log(
    `   track "${track.name}" ${track.poses.length} poses: p50 ${r.p50.toFixed(2)} p95 ${r.p95.toFixed(2)} ` +
      `WORST ${r.worst.toFixed(2)} ms @pose#${r.worstPoseIndex} (visCl ${r.perPoseVis[r.worstPoseIndex]}) ` +
      `| paired-noise med ${noiseMed.toFixed(3)} ms | rej ${r.rejected}`,
  );
  // worst vs p50 visCluster comparison
  const p50vis = median(r.perPoseVis);
  console.log(
    `   → worst/p50 ratio ${(r.worst / r.p50).toFixed(2)}× ; worst-frame visCl ${r.perPoseVis[r.worstPoseIndex]} ` +
      `vs p50 visCl ${p50vis} (${(r.perPoseVis[r.worstPoseIndex]! / Math.max(1, p50vis)).toFixed(2)}× more clusters)`,
  );
  row('S1/world1/motion-track', {
    p50: +r.p50.toFixed(2), p95: +r.p95.toFixed(2), worst: +r.worst.toFixed(2),
    worstVisCl: r.perPoseVis[r.worstPoseIndex] ?? -1, p50VisCl: p50vis,
    pairedNoiseMs: +noiseMed.toFixed(3),
  }, `worst@pose#${r.worstPoseIndex}; >60fps needs worst<16.7ms (now ${r.worst.toFixed(1)})`);
  await page.close();
}

async function s2(browser: Browser): Promise<void> {
  hr('S2. RDBG STAGE-SPLIT — world1 ?rdbg=1/2/3 ⇒ non-raster floor vs per-triangle vs PER-PIXEL');
  // IMPORTANT: rdbg=1/2 make the WHOLE frame so cheap it is NOT GPU-bound, so the
  // per-pass `c.nanRasterWorld1` TIMESTAMP reads the bogus ~15.4 ms vsync pipelining
  // floor (capSuspect REJECTS it → p50=0). The honest signal is the ISOLATED
  // whole-frame gpuWall (drain→submit→drain), which is real active wall time.
  // occl=0 ⇒ NO HZB occlusion feedback ⇒ the cluster set is IDENTICAL across rdbg
  // levels, so the gpuWall DELTA between levels is the clean per-stage raster cost.
  //   rdbg1 = non-raster constant (cull+hzb+resolve+launch+ctx) ; rdbg2−rdbg1 = per-triangle ;
  //   rdbg3−rdbg2 = per-pixel coverage/election loop.
  const medL = (a: number[]): number => {
    const s = a.filter(Number.isFinite).sort((x, y) => x - y);
    return s.length ? (s[s.length >> 1] as number) : 0;
  };
  for (const dpr of ['1', '2'] as const) {
    const out: Record<number, { ms: number; vis: number }> = {};
    for (const rdbg of [1, 2, 3]) {
      const page = await bootPage(browser, {
        scene: 'forest', width: W, height: H,
        extra: forestExtra('world1', { rdbg: String(rdbg), dpr, occl: '0' }), settle: 50, logPrefix: '[forest]',
      });
      await page.evaluate((pp) => window.__laas.setPose?.(pp), ALLEY);
      const fr = (await page.evaluate(
        async (o) => window.__laas.measureFrames?.(o),
        { frames: 14, warmup: 16 },
      )) as { gpuWallMs: number; counters: Record<string, number> }[];
      out[rdbg] = {
        ms: medL(fr.map((f) => f.gpuWallMs)),
        vis: medL(fr.map((f) => f.counters['nanite.visClusters'] ?? -1)),
      };
      console.log(`   dpr${dpr} rdbg=${rdbg}: gpuWall ${out[rdbg]!.ms.toFixed(2)} ms (visCl ${out[rdbg]!.vis})`);
      await page.close();
    }
    const floor = out[1]!.ms;
    const tri = out[2]!.ms - out[1]!.ms;
    const pix = out[3]!.ms - out[2]!.ms;
    const total = out[3]!.ms;
    console.log(
      `   → dpr${dpr}: non-raster+launch FLOOR ${floor.toFixed(2)} (${((floor / total) * 100).toFixed(0)}%) | ` +
        `per-triangle edge-setup ${tri.toFixed(2)} (${((tri / total) * 100).toFixed(0)}%) | ` +
        `PER-PIXEL loop ${pix.toFixed(2)} (${((pix / total) * 100).toFixed(0)}%) of ${total.toFixed(2)} ms frame`,
    );
    row(`S2/world1/rdbg-split@dpr${dpr}`, {
      floorMs: +floor.toFixed(2), triMs: +tri.toFixed(2), pixelMs: +pix.toFixed(2), frameMs: +total.toFixed(2),
      pixelPct: +((pix / total) * 100).toFixed(0), triPct: +((tri / total) * 100).toFixed(0), visClusters: out[3]!.vis,
    }, 'occl=0 matched cluster set; gpuWall deltas; rdbg1=floor, rdbg2=+tri, rdbg3=+per-pixel');
  }
}

async function s3(browser: Browser): Promise<void> {
  hr('S3. RESOLUTION SWEEP (fixed work) — is cost ∝ pixels (per-pixel/compute) or sub-quadratic (fixed/bw)?');
  const dprs = [0.75, 1.0, 1.414, 2.0];
  const pts: { px: number; ms: number; vis: number; dpr: number }[] = [];
  for (const dpr of dprs) {
    const page = await bootPage(browser, {
      scene: 'forest', width: W, height: H,
      extra: forestExtra('world1', { dpr: String(dpr) }), settle: 50, logPrefix: '[forest]',
    });
    await page.evaluate((pp) => window.__laas.setPose?.(pp), ALLEY);
    const r = await measureActiveGpuSteady(page, 'c.nanRasterWorld1', { frames: FRAMES, warmup: 14 });
    const px = Math.round(W * dpr) * Math.round(H * dpr);
    pts.push({ px, ms: r.p50, vis: r.visClusters, dpr });
    console.log(`   dpr ${dpr.toFixed(3)} (${Math.round(W * dpr)}×${Math.round(H * dpr)}, ${(px / 1e6).toFixed(2)} Mpx): ${r.p50.toFixed(2)} ms (visCl ${r.visClusters})`);
    await page.close();
  }
  // fit ms = a + b·Mpx ; if a≈0 and high r² ⇒ purely per-pixel; a>0 ⇒ a fixed floor
  const xs = pts.map((p) => p.px / 1e6);
  const ys = pts.map((p) => p.ms);
  const { a, b, r2 } = linfit(xs, ys);
  // also the exponent: log-log slope (2.0 = perfectly quadratic in dpr)
  const lo = pts[0]!;
  const hi = pts[pts.length - 1]!;
  const expo = Math.log(hi.ms / lo.ms) / Math.log(hi.px / lo.px);
  console.log(
    `   → linear fit ms = ${a.toFixed(2)} + ${b.toFixed(2)}·Mpx (r²=${r2.toFixed(3)}); ` +
      `log-log px-exponent ${expo.toFixed(2)} (1.0 = ∝pixels). ` +
      `fixed-floor a=${a.toFixed(2)} ms = ${((a / hi.ms) * 100).toFixed(0)}% of the 2× number`,
  );
  row('S3/world1/res-sweep', {
    fixedFloorMs: +a.toFixed(2), msPerMpx: +b.toFixed(2), r2: +r2.toFixed(3), pxExponent: +expo.toFixed(2),
    ms_0p75x: +pts[0]!.ms.toFixed(2), ms_1x: +pts[1]!.ms.toFixed(2), ms_2x: +pts[3]!.ms.toFixed(2),
  }, 'a≈fixed/cluster floor (res-independent); b·Mpx = per-pixel term');
}

async function s4(browser: Browser): Promise<void> {
  hr('S4. VISCLUSTER SWEEP — intercept (fixed) vs slope (per-cluster) at fixed 1× res');
  const page = await bootPage(browser, {
    scene: 'forest', width: W, height: H, extra: forestExtra('world1'), settle: 60, logPrefix: '[forest]',
  });
  await page.evaluate((pp) => window.__laas.setPose?.(pp), ALLEY);
  const bands = [60, 30, 14, 6, 3, 1.5];
  const pts: { vis: number; ms: number }[] = [];
  for (const band of bands) {
    await setBand(page, band);
    const r = await measureActiveGpu(page, 'c.nanRasterWorld1', { frames: FRAMES, warmup: 12 });
    pts.push({ vis: r.visClusters, ms: r.p50 });
    console.log(`   simband ${String(band).padStart(4)}: visCl ${String(r.visClusters).padStart(7)} → ${r.p50.toFixed(2)} ms`);
  }
  await page.close();
  const xs = pts.map((p) => p.vis / 1e6);
  const ys = pts.map((p) => p.ms);
  const { a, b, r2 } = linfit(xs, ys);
  console.log(
    `   → ms = ${a.toFixed(2)} (fixed intercept) + ${b.toFixed(2)}·Mcl (r²=${r2.toFixed(3)}). ` +
      `At the baseline ~347k cl the per-cluster term ≈ ${(b * 0.347).toFixed(2)} ms, fixed ≈ ${a.toFixed(2)} ms`,
  );
  row('S4/world1/viscl-sweep', {
    interceptMs: +a.toFixed(2), msPerMcl: +b.toFixed(2), r2: +r2.toFixed(3),
    visClLo: pts[pts.length - 1]!.vis, visClHi: pts[0]!.vis,
    msLo: +pts[pts.length - 1]!.ms.toFixed(2), msHi: +pts[0]!.ms.toFixed(2),
  }, 'intercept = fixed floor (launch/dispatch/resolve); slope·visCl = scaling half');
}

async function s5(browser: Browser): Promise<void> {
  hr('S5. OVERDRAW (instrument-free) — leaf-DENSITY sweep at fixed screen footprint');
  // The in-kernel covered-fragment counter (?audit/?ovd on the packed path) reads 0
  // (a pre-existing packed-path readback gap — visDepthV is unused in the packed
  // election, and the readback returns stale on this NaniteView config). So size the
  // overdraw EMPIRICALLY: hold the camera + tree grid + resolution FIXED and sweep
  // ?leafdensity (anchors per crown). More leaves on the SAME crowns = more sub-pixel
  // triangles STACKED per pixel = more depth-election competition = more overdraw.
  // If world1 ms rises ~linearly with leaf density at a fixed screen footprint, the
  // cost IS the per-pixel coverage volume (overdraw), not the lit-pixel count.
  // stay UNDER the HW_CAP=2.1M overflow (8000 anchors drops geometry → invalid 0 ms).
  const densities = [500, 1000, 2000, 4000];
  const pts: { d: number; ms: number; vis: number; hwTris: number }[] = [];
  for (const d of densities) {
    const page = await bootPage(browser, {
      scene: 'forest', width: W, height: H,
      extra: forestExtra('world1', { leafdensity: String(d) }), settle: 60, logPrefix: '[forest]',
    });
    await page.evaluate((pp) => window.__laas.setPose?.(pp), ALLEY);
    const r = await measureActiveGpuSteady(page, 'c.nanRasterWorld1', { frames: FRAMES, warmup: 12 });
    const hwTris = (await page.evaluate(
      () => window.__laas.stats?.counters?.['nanite.hwTris'] ?? -1,
    )) as number;
    pts.push({ d, ms: r.p50, vis: r.visClusters, hwTris });
    console.log(
      `   leafdensity ${String(d).padStart(5)}: world1 ${r.p50.toFixed(2)} ms | visCl ${r.visClusters} | hwTris ${hwTris}` +
        (hwTris <= 0 ? '  ⚠ HW overflow — INVALID' : ''),
    );
    await page.close();
  }
  const lo = pts[0]!;
  const hi = pts[pts.length - 1]!;
  const dDens = hi.d / lo.d;
  const dMs = hi.ms / lo.ms;
  console.log(
    `   → leaf density ${lo.d}→${hi.d} (${dDens.toFixed(1)}×) ⇒ world1 ${lo.ms.toFixed(2)}→${hi.ms.toFixed(2)} ms (${dMs.toFixed(2)}×). ` +
      `${dMs > 1.3 ? 'COST RISES WITH LEAF DENSITY ⇒ the per-pixel work IS overdraw volume (sub-pixel tris stacked per pixel)' : 'flat ⇒ NOT density-bound'} ` +
      `at ~constant screen footprint.`,
  );
  row('S5/world1/leafdensity-sweep', {
    dens_lo: lo.d, dens_hi: hi.d, ms_lo: +lo.ms.toFixed(2), ms_hi: +hi.ms.toFixed(2),
    msRatio: +dMs.toFixed(2), densRatio: +dDens.toFixed(1),
    visCl_lo: lo.vis, visCl_hi: hi.vis,
  }, 'fixed pose+grid+res; rising ms with leaf anchors ⇒ overdraw-bound (stacked sub-pixel tris)');
  console.log(
    '   NOTE: in-kernel covered-fragment volume counter not available on the packed path; ' +
      'overdraw sized via this density sweep + the established 0.147 tris/px sub-pixel floor.',
  );
}

async function main(): Promise<void> {
  const { browser } = await launchWebGPU();
  console.log(`[harvest] stages=${STAGES.join(',')} frames/measure=${FRAMES} @ ${W}×${H} trees=${process.env.TREES ?? '40000'}`);
  if (STAGES.includes('S0')) await s0(browser);
  if (STAGES.includes('S1')) await s1(browser);
  if (STAGES.includes('S2')) await s2(browser);
  if (STAGES.includes('S3')) await s3(browser);
  if (STAGES.includes('S4')) await s4(browser);
  if (STAGES.includes('S5')) await s5(browser);
  await browser.close();
  console.log('\n========== CORPUS (json) ==========');
  console.log(JSON.stringify(ROWS, null, 2));
  console.log('\n[harvest] done.');
}
main().catch((e) => {
  console.error('[harvest] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
