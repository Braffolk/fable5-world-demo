/**
 * probe-raster-ab: whole-frame gpuWall A/B for the RASTER cluster path (task #76).
 *
 * Boots the raster-only config (grass+shadows OFF, ?clhw per-cluster split — the arc's
 * canonical capture URL) at a DENSE-FOLIAGE pose (eye, standing in the forest, where
 * nanRasterWorld1 = 52.6% of GPU) and measures the whole-frame delta of each variant flag
 * vs baseline. Whole-frame (frameMs) + compute/render timestamps only — per-pass timestamps
 * are NOT additive on Apple (render‖compute overlap), so we trust the whole-frame number and
 * the compute total (nanRasterWorld1 is compute). SPEC methodology: "verify with wall + ablation".
 *
 * Each variant is a SEPARATE BOOT (build-time flags read once at scene ctor), so we BRACKET:
 * baseline runs FIRST and LAST and each variant is compared to the time-interpolated baseline
 * (cancels cross-boot DVFS/thermal drift — the "iso gap was DVFS" hazard).
 *
 *   npx tsx tools/probe-raster-ab.ts
 *   VARIANTS='hw1fetch=1' npx tsx tools/probe-raster-ab.ts
 *   VARIANTS='hw1fetch=1;wgcache=0;clhwmax=16' POSE=eye SAMPLES=20 npx tsx tools/probe-raster-ab.ts
 *   POSE=oblique npx tsx tools/probe-raster-ab.ts
 *
 * VARIANTS = ';'-separated flag sets; each set is '&'-joined key=value (e.g. 'hw1fetch=1&relect=2').
 * Needs the dev server on :5173.
 */
import { launchWebGPU, laasUrl } from './launch';

type Pose = { name: string; p: [number, number, number]; yaw: number; pitch: number };
const POSES: Record<string, Pose> = {
  // eye = standing IN the forest — the dense-foliage scenario where nanRasterWorld1 dominates.
  eye: { name: 'eye', p: [0, 2, 0], yaw: 0.6, pitch: -0.02 },
  oblique: { name: 'oblique', p: [0, 40, 40], yaw: 0, pitch: -0.35 },
  aerial: { name: 'aerial', p: [0, 150, 0], yaw: 0, pitch: -1.45 },
};
const POSE = POSES[process.env.POSE ?? 'eye'] ?? POSES.eye!;
const T = Number(process.env.T ?? 11);
const SAMPLES = Number(process.env.SAMPLES ?? 16);
// dpr=2 to match the profile capture; viewport ×2 = internal render res.
const WIDTH = Number(process.env.WIDTH ?? 1296);
const HEIGHT = Number(process.env.HEIGHT ?? 838);
// the arc's raster-only + cluster-split config (matches the profiled capture URL)
const BASE_CONFIG: Record<string, string> = {
  nanite: '1',
  dpr: '2',
  clhw: '1',
  clhwmax: '32',
  grass: '0',
  nanshadow: '0',
};
const VARIANTS = (process.env.VARIANTS ?? 'hw1fetch=1').split(';').filter(Boolean);

function median(a: number[]): number {
  const s = a.filter((x) => Number.isFinite(x)).sort((x, y) => x - y);
  return s.length === 0 ? 0 : (s[Math.floor(s.length / 2)] as number);
}

interface Sample {
  frameMs: number;
  fps: number;
  render: number;
  compute: number;
}

async function measureBoot(
  browser: Awaited<ReturnType<typeof launchWebGPU>>['browser'],
  variant: string,
): Promise<Sample> {
  const page = await browser.newPage({
    viewport: { width: WIDTH, height: HEIGHT },
    deviceScaleFactor: 1,
  });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));
  const extra: Record<string, string> = { ...BASE_CONFIG };
  for (const part of variant.split('&').filter(Boolean)) {
    const eq = part.indexOf('=');
    if (eq >= 0) extra[part.slice(0, eq)] = part.slice(eq + 1);
    else extra[part] = '1';
  }
  const url = laasUrl({ scene: 'world', width: WIDTH, height: HEIGHT, freeze: false, extra });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error != null),
    undefined,
    { timeout: 180000, polling: 250 },
  );
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);
  await page.evaluate((t) => window.__laas.setTimeOfDay?.(t), T);
  await page.evaluate(
    (p) => window.__laas.setPose?.({ p: [p.x, p.y, p.z], yaw: p.yaw, pitch: p.pitch }),
    { x: POSE.p[0], y: POSE.p[1], z: POSE.p[2], yaw: POSE.yaw, pitch: POSE.pitch },
  );
  // settle streaming + let the GPU clock reach steady state (DVFS warm-up)
  await page.evaluate(async () => window.__laas.settle && (await window.__laas.settle(60)));
  const acc = { frameMs: [] as number[], fps: [] as number[], render: [] as number[], compute: [] as number[] };
  for (let i = 0; i < SAMPLES; i++) {
    await page.evaluate(async () => window.__laas.settle && (await window.__laas.settle(3)));
    const f = (await page.evaluate(() => {
      const st = window.__laas.stats;
      return {
        frameMs: st?.frameMs ?? 0,
        fps: st?.fps ?? 0,
        render: st?.gpuPasses?.['render'] ?? 0,
        compute: st?.gpuPasses?.['compute'] ?? 0,
      };
    })) as Sample;
    acc.frameMs.push(f.frameMs);
    acc.fps.push(f.fps);
    acc.render.push(f.render);
    acc.compute.push(f.compute);
  }
  await page.close();
  return {
    frameMs: median(acc.frameMs),
    fps: median(acc.fps),
    render: median(acc.render),
    compute: median(acc.compute),
  };
}

async function main(): Promise<void> {
  const { browser } = await launchWebGPU();
  console.log(
    `[raster-ab] pose=${POSE.name} (${POSE.p.join(',')}) yaw=${POSE.yaw} T=${T} @${WIDTH}×${HEIGHT}·dpr2, ${SAMPLES} samples/boot\n` +
      `[raster-ab] config: ${Object.entries(BASE_CONFIG).map(([k, v]) => `${k}=${v}`).join(' ')}\n` +
      `[raster-ab] order: baseline, ${VARIANTS.join(', ')}, baseline2 (bracketed for drift)`,
  );

  const order = ['', ...VARIANTS, ''];
  const results: Sample[] = [];
  for (const v of order) {
    process.stdout.write(`[raster-ab] booting "${v || 'baseline'}" ... `);
    const s = await measureBoot(browser, v);
    results.push(s);
    console.log(
      `frameMs ${s.frameMs.toFixed(2)}  fps ${s.fps.toFixed(0)}  compute ${s.compute.toFixed(2)}  render ${s.render.toFixed(2)}`,
    );
  }
  await browser.close();

  const base0 = results[0]!;
  const base1 = results[results.length - 1]!;
  const N = VARIANTS.length;
  const fmt = (n: number, w = 7): string => n.toFixed(2).padStart(w);
  console.log('\n[raster-ab] Δ vs time-interpolated baseline (POSITIVE = variant is FASTER):');
  console.log('  variant                      frameMs   Δframe    compute  Δcompute   render   fps');
  console.log(
    `  baseline                    ${fmt(base0.frameMs)}      —    ${fmt(base0.compute)}      —    ${fmt(base0.render)} ${base0.fps.toFixed(0).padStart(4)}`,
  );
  for (let i = 0; i < N; i++) {
    const s = results[i + 1]!;
    const frac = (i + 1) / (N + 1);
    const baseFrame = base0.frameMs + (base1.frameMs - base0.frameMs) * frac;
    const baseCompute = base0.compute + (base1.compute - base0.compute) * frac;
    const dF = baseFrame - s.frameMs;
    const dC = baseCompute - s.compute;
    console.log(
      `  ${(VARIANTS[i] ?? '').padEnd(26)} ${fmt(s.frameMs)} ${fmt(dF)}  ${fmt(s.compute)} ${fmt(dC)}  ${fmt(s.render)} ${s.fps.toFixed(0).padStart(4)}`,
    );
  }
  console.log(
    `  baseline2                   ${fmt(base1.frameMs)}      —    ${fmt(base1.compute)}      —    ${fmt(base1.render)} ${base1.fps.toFixed(0).padStart(4)}`,
  );
  console.log(
    '\n[raster-ab] Δframe = honest whole-frame win. compute = the pass nanRasterWorld1 lives in.\n' +
      '[raster-ab] baseline vs baseline2 gap = residual drift (trust deltas only if it is small).',
  );
}
main().catch((e) => {
  console.error('[raster-ab] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
