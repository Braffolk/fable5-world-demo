/**
 * N5 shadow S0 (D-N29) perf ledger: the resolve-side PCSS SAMPLE cost, half-res
 * vs full-res. Boots bm7 (forest interior, dense shadow receivers) STATIC/frozen,
 * settles, then medians specific gpuPasses keys directly — bypassing shoot.ts's
 * render+compute>0 gate, which drops every sample under shadows-on because three's
 * CSM keep-alive map (r.shadow.c0) reports a garbage NEGATIVE timestamp that
 * poisons the 'render' total.
 *
 * shalfres=1 (S0): r.scene holds only the cheap bilateral upsample; the PCSS moves
 * to the half-res compute pass c.nanShadowHalf (¼ the pixels).
 * shalfres=0 (full): r.scene holds the full-res per-pixel PCSS.
 * The SAMPLE win = r.scene(full) − [r.scene(half) + c.nanShadowHalf].
 *
 *   npx tsx tools/probe-shadowcost.ts          # needs the dev server on :5173
 *   SHOT=3 npx tsx tools/probe-shadowcost.ts   # vista instead of forest interior
 */
import { launchWebGPU, laasUrl } from './launch';

const KEYS = ['r.scene', 'c.nanShadowHalf', 'c.nanRasterPayload', 'c.nanRasterDepth'] as const;
type Key = (typeof KEYS)[number];
type Row = Record<Key, number>;

function median(a: number[]): number {
  const s = a.filter((x) => x > 0).sort((x, y) => x - y);
  return s.length === 0 ? 0 : (s[Math.floor(s.length / 2)] as number);
}

const WIDTH = Number(process.env.WIDTH ?? 1280);
const HEIGHT = Number(process.env.HEIGHT ?? 720);

async function boot(extra: Record<string, string>, samples: number): Promise<Row> {
  const shot = process.env.SHOT ?? '7';
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT }, deviceScaleFactor: 1 });
  page.on('pageerror', (err) => console.error('[pageerror]', err.message));
  const url = laasUrl({ scene: 'world', width: WIDTH, height: HEIGHT, freeze: true, extra: { nanite: '1', shot, ...extra } });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__laas && (window.__laas.ready || window.__laas.error != null), undefined, {
    timeout: 180000,
    polling: 250,
  });
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);
  await page.evaluate(async () => window.__laas.settle && (await window.__laas.settle(50)));

  const acc: Record<string, number[]> = {};
  for (const k of KEYS) acc[k] = [];
  for (let i = 0; i < samples; i++) {
    await page.evaluate(async () => window.__laas.settle && (await window.__laas.settle(10)));
    const g = (await page.evaluate(() => window.__laas.stats?.gpuPasses ?? {})) as Record<string, number>;
    for (const k of KEYS) acc[k]!.push(g[k] ?? 0);
  }
  await browser.close();
  const out = {} as Row;
  for (const k of KEYS) out[k] = median(acc[k]!);
  return out;
}

async function main(): Promise<void> {
  const samples = Number(process.env.SAMPLES ?? 18);
  console.log(`[shadowcost] bm${process.env.SHOT ?? '7'} static, ${samples} samples, ${WIDTH}×${HEIGHT}`);
  console.log('[shadowcost] booting shalfres=1 (S0 half-res) …');
  const half = await boot({}, samples);
  console.log('[shadowcost] booting shalfres=0 (full-res) …');
  const full = await boot({ shalfres: '0' }, samples);

  const f = (n: number): string => `${n.toFixed(3)} ms`;
  console.log('\n[shadowcost] median per-pass GPU time:');
  console.log(`  r.scene (resolve)      full ${f(full['r.scene'])}   |  half ${f(half['r.scene'])}`);
  console.log(`  c.nanShadowHalf        full ${f(full['c.nanShadowHalf'])}   |  half ${f(half['c.nanShadowHalf'])}`);
  console.log(`  c.nanRasterPayload     full ${f(full['c.nanRasterPayload'])}   |  half ${f(half['c.nanRasterPayload'])}  (ref, unchanged)`);
  const fullSample = full['r.scene'];
  const halfTotal = half['r.scene'] + half['c.nanShadowHalf'];
  const win = fullSample - halfTotal;
  console.log(
    `\n[shadowcost] shadow-pass cost: full-res r.scene ${f(fullSample)}  vs  ` +
      `half-res (r.scene+half) ${f(halfTotal)}  →  Δ ${f(win)} ` +
      `(${fullSample > 0 ? ((win / fullSample) * 100).toFixed(0) : '0'}% of resolve)`,
  );
}
main().catch((e) => {
  console.error('[shadowcost] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
