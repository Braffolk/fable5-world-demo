/**
 * probe-postablate: measure the TRUE MARGINAL cost of each post-chain effect by
 * ABLATION, not by trusting per-pass GPU timestamps. Back-to-back post passes
 * make the first passes after a heavy one (TRAA/scene) ABSORB drain latency —
 * e.g. bloom's `bright`+`h0` read ~7 ms each at HALF res while same-res `v0`
 * reads 0.07 ms, which is impossible as real work. So the only trustworthy
 * "what does effect X cost" is: boot with `?ablate=X`, measure the whole-frame
 * delta vs baseline. (SPEC PERF METHODOLOGY: "verify with wall fps + ablation".)
 *
 * Each config is a SEPARATE BOOT (ablate is read once in the PostStack ctor), so
 * to cancel cross-boot thermal drift we bracket: baseline runs FIRST and LAST and
 * each ablation is compared to the time-interpolated baseline.
 *
 *   npx tsx tools/probe-postablate.ts
 *   YAW=1.05 SAMPLES=16 npx tsx tools/probe-postablate.ts
 *   CONFIGS=bloom,taa,clouds,ao npx tsx tools/probe-postablate.ts
 *
 * Needs the dev server on :5173.
 */
import { launchWebGPU, laasUrl } from './launch';

const X = Number(process.env.X ?? -4.2);
const Y = Number(process.env.Y ?? 303.1);
const Z = Number(process.env.Z ?? -1.4);
const T = Number(process.env.T ?? 11);
const YAW = Number(process.env.YAW ?? 1.05); // ~60°, the heavy "long alley" orientation
const PITCH = Number(process.env.PITCH ?? -0.12);
const SAMPLES = Number(process.env.SAMPLES ?? 16);
const WIDTH = Number(process.env.WIDTH ?? 2592);
const HEIGHT = Number(process.env.HEIGHT ?? 1676);
// the effects to ablate one-at-a-time (PostStack ?ablate= set)
const CONFIGS = (process.env.CONFIGS ?? 'bloom,taa,clouds,ao,bounce,contact').split(',').filter(Boolean);

const KEYS = [
  'r.scene',
  'r.half.mrt',
  'r.rt#16(2592x1676)',
  'r.TRAANode.resolve',
  'r.UnrealBloomPass.bright',
  'r.UnrealBloomPass.h0',
  'r.nanHwPass',
];

function median(a: number[]): number {
  const s = a.filter((x) => Number.isFinite(x)).sort((x, y) => x - y);
  return s.length === 0 ? 0 : (s[Math.floor(s.length / 2)] as number);
}

interface Sample {
  frameMs: number;
  fps: number;
  render: number;
  compute: number;
  g: Record<string, number>;
}

async function measureBoot(
  browser: Awaited<ReturnType<typeof launchWebGPU>>['browser'],
  cfg: string,
): Promise<Sample> {
  const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));
  const extra: Record<string, string> = { nanite: '1' };
  // cfg is '&'-joined "key=value" URL params (e.g. aocheap=0&aosamples=8) or a
  // bare ablate stage name (e.g. bloom → ?ablate=bloom). '' = baseline.
  if (cfg) {
    for (const part of cfg.split('&')) {
      const eq = part.indexOf('=');
      if (eq >= 0) extra[part.slice(0, eq)] = part.slice(eq + 1);
      else extra['ablate'] = part;
    }
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
    { x: X, y: Y, z: Z, yaw: YAW, pitch: PITCH },
  );
  // settle streaming (T-ticks) + let exposure/TRAA history converge
  await page.evaluate(async () => window.__laas.settle && (await window.__laas.settle(40)));
  const acc = new Map<string, number[]>();
  const push = (k: string, v: number): void => {
    if (!acc.has(k)) acc.set(k, []);
    acc.get(k)!.push(v);
  };
  for (let i = 0; i < SAMPLES; i++) {
    await page.evaluate(async () => window.__laas.settle && (await window.__laas.settle(3)));
    const f = (await page.evaluate(() => {
      const st = window.__laas.stats;
      return { g: st?.gpuPasses ?? {}, frameMs: st?.frameMs ?? 0, fps: st?.fps ?? 0 };
    })) as { g: Record<string, number>; frameMs: number; fps: number };
    push('frameMs', f.frameMs);
    push('fps', f.fps);
    push('render', f.g['render'] ?? 0);
    push('compute', f.g['compute'] ?? 0);
    for (const k of KEYS) push(k, f.g[k] ?? 0);
  }
  await page.close();
  const g: Record<string, number> = {};
  for (const k of KEYS) g[k] = median(acc.get(k) ?? []);
  return {
    frameMs: median(acc.get('frameMs') ?? []),
    fps: median(acc.get('fps') ?? []),
    render: median(acc.get('render') ?? []),
    compute: median(acc.get('compute') ?? []),
    g,
  };
}

async function main(): Promise<void> {
  const { browser } = await launchWebGPU();
  console.log(
    `[postablate] pos (${X},${Y},${Z}) yaw=${YAW} T=${T} @${WIDTH}×${HEIGHT}, ${SAMPLES} samples/config\n` +
      `[postablate] order: baseline, ${CONFIGS.join(', ')}, baseline2 (bracketed for drift)`,
  );

  const order = ['', ...CONFIGS, ''];
  const results: Sample[] = [];
  for (const cfg of order) {
    process.stdout.write(`[postablate] booting ablate="${cfg || 'none'}" ... `);
    const s = await measureBoot(browser, cfg);
    results.push(s);
    console.log(`frameMs ${s.frameMs.toFixed(1)}  fps ${s.fps.toFixed(0)}  render ${s.render.toFixed(2)}`);
  }
  await browser.close();

  const base0 = results[0]!;
  const base1 = results[results.length - 1]!;
  // time-interpolated baseline render for config i (i in 1..N): linear blend
  const N = CONFIGS.length;
  console.log('\n[postablate] MARGINAL cost (Δrender vs interpolated baseline; >0 = effect costs that much):');
  console.log('  config       frameMs   fps   render   Δrender   Δframe   bright   h0     traa   rt#16  half.mrt');
  const fmt = (n: number, w = 6): string => n.toFixed(2).padStart(w);
  const bl = (s: Sample): string =>
    `${fmt(s.frameMs)} ${s.fps.toFixed(0).padStart(4)} ${fmt(s.render)}     —        —    ` +
    `${fmt(s.g['r.UnrealBloomPass.bright'] ?? 0)} ${fmt(s.g['r.UnrealBloomPass.h0'] ?? 0)} ` +
    `${fmt(s.g['r.TRAANode.resolve'] ?? 0)} ${fmt(s.g['r.rt#16(2592x1676)'] ?? 0)} ${fmt(s.g['r.half.mrt'] ?? 0)}`;
  console.log(`  baseline   ${bl(base0)}`);
  for (let i = 0; i < N; i++) {
    const s = results[i + 1]!;
    const frac = (i + 1) / (N + 1);
    const baseRender = base0.render + (base1.render - base0.render) * frac;
    const baseFrame = base0.frameMs + (base1.frameMs - base0.frameMs) * frac;
    const dR = baseRender - s.render;
    const dF = baseFrame - s.frameMs;
    console.log(
      `  -${(CONFIGS[i] ?? '').padEnd(9)} ${fmt(s.frameMs)} ${s.fps.toFixed(0).padStart(4)} ${fmt(s.render)} ` +
        `${fmt(dR, 7)}  ${fmt(dF, 6)}  ` +
        `${fmt(s.g['r.UnrealBloomPass.bright'] ?? 0)} ${fmt(s.g['r.UnrealBloomPass.h0'] ?? 0)} ` +
        `${fmt(s.g['r.TRAANode.resolve'] ?? 0)} ${fmt(s.g['r.rt#16(2592x1676)'] ?? 0)} ${fmt(s.g['r.half.mrt'] ?? 0)}`,
    );
  }
  console.log(`  baseline2  ${bl(base1)}`);
  console.log(
    '\n[postablate] Δframe is the honest whole-frame win from removing that effect (includes un-throttling).',
  );
}
main().catch((e) => {
  console.error('[postablate] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
