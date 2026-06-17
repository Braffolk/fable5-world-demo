/**
 * probe-diag: parametric single-config diagnostic for the forest vis-buffer hot
 * path. Boots ?scene=forest&nanite=1 (full pipe) OR with ?nanitedbg=<mode> (lean
 * vis-buffer only), warms ~120 frames via __laas.settle, then prints ONE JSON
 * line of medians for that config. Built for serial sweep batteries — the GPU is
 * a single resource, so this script measures exactly one config per invocation.
 *
 * Env vars (all optional, defaults = canonical operating point):
 *   TREES=40000  INSTMINPX=128  SIMBAND=6  LODNEAR=4  LODPOW=0.6
 *   W=1280  H=720  NANITEDBG=<unset=full pipe | flat | cluster>
 *   SAMPLES=22 (frames measured after warm) WARM=120
 *
 * Emits to stdout: a single line beginning "RESULT " followed by JSON:
 *   {config, frameMs, world1Ms, computeMs, renderMs, cpuSubmitMs,
 *    visClusters, hwTris, visTris, fps, mode}
 *
 * Needs the dev server on :5173.
 */
import { laasUrl, launchWebGPU } from './launch';

const W = Number(process.env.W ?? 1280);
const H = Number(process.env.H ?? 720);
const TREES = process.env.TREES ?? '40000';
const INSTMINPX = process.env.INSTMINPX ?? '128';
const SIMBAND = process.env.SIMBAND ?? '6';
const LODNEAR = process.env.LODNEAR ?? '4';
const LODPOW = process.env.LODPOW ?? '0.6';
const LODERR = process.env.LODERR ?? ''; // '' = default tau (1px); >1 = coarser screen-error cut
const NANITEDBG = process.env.NANITEDBG ?? ''; // '' = full pipe
const SAMPLES = Number(process.env.SAMPLES ?? 22);
const WARM = Number(process.env.WARM ?? 120);

function median(a: number[]): number {
  const s = a.filter((x) => Number.isFinite(x)).sort((x, y) => x - y);
  return s.length === 0 ? 0 : (s[Math.floor(s.length / 2)] as number);
}

async function main(): Promise<void> {
  const extra: Record<string, string> = {
    nanite: '1',
    trees: TREES,
    lodnear: LODNEAR,
    simband: SIMBAND,
    lodpow: LODPOW,
    instminpx: INSTMINPX,
  };
  if (LODERR) extra['loderr'] = LODERR;
  if (NANITEDBG) extra['nanitedbg'] = NANITEDBG;
  // EXTRA="wgcache=0,dag=0,occl=0" — arbitrary passthrough knobs for ablations
  for (const kv of (process.env.EXTRA ?? '').split(',')) {
    if (!kv) continue;
    const [k, v] = kv.split('=');
    if (k) extra[k] = v ?? '1';
  }

  const configStr =
    `trees=${TREES} instminpx=${INSTMINPX} simband=${SIMBAND} lodnear=${LODNEAR} ` +
    `lodpow=${LODPOW} loderr=${LODERR || '1'} W=${W} H=${H} dbg=${NANITEDBG || 'full'}`;

  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));
  const url = laasUrl({ scene: 'forest', width: W, height: H, freeze: false, extra });
  console.error(`[diag] booting ${url}`);

  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__laas && (window.__laas.ready || window.__laas.error != null), undefined, {
    timeout: 300000,
    polling: 250,
  });
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);

  await page.evaluate(async (warm) => {
    if (window.__laas.settle) await window.__laas.settle(warm);
  }, WARM);

  const acc = new Map<string, number[]>();
  for (let i = 0; i < SAMPLES; i++) {
    const r = (await page.evaluate(async () => {
      if (window.__laas.settle) await window.__laas.settle(4);
      const st = window.__laas.stats;
      const g = (st?.gpuPasses ?? {}) as Record<string, number>;
      const c = (st?.counters ?? {}) as Record<string, number>;
      return {
        fps: st?.fps ?? 0,
        frameMs: st?.frameMs ?? 0,
        // full pipe = nanRasterWorld1; lean nanitedbg view = nanRasterCombined.
        // 'raster' = whichever SW vis-buffer kernel actually ran this config.
        world1: g['c.nanRasterWorld1'] ?? 0,
        combined: g['c.nanRasterCombined'] ?? 0,
        raster: (g['c.nanRasterWorld1'] ?? 0) || (g['c.nanRasterCombined'] ?? 0),
        compute: g['compute'] ?? 0,
        render: g['render'] ?? 0,
        // cpu.submitMs100 is stored ×100 as an integer counter
        cpuSubmit: (c['cpu.submitMs100'] ?? 0) / 100,
        cpuUpdate: (c['cpu.updateMs100'] ?? 0) / 100,
        visClusters: c['nanite.visClusters'] ?? -1,
        hwTris: c['nanite.hwTris'] ?? -1,
        visTris: c['nanite.visTris'] ?? -1,
        dagClusters: c['nanite.dagClusters'] ?? -1,
        chunks: c['nanite.chunks'] ?? -1,
        rejInst: c['nanite.rejInst'] ?? -1,
        rejClust: c['nanite.rejClust'] ?? -1,
        p2: c['nanite.p2'] ?? -1,
      };
    })) as Record<string, number>;
    for (const [k, v] of Object.entries(r)) {
      if (!acc.has(k)) acc.set(k, []);
      acc.get(k)!.push(v);
    }
  }
  const m = (k: string): number => median(acc.get(k) ?? [0]);

  const out = {
    config: configStr,
    mode: NANITEDBG || 'full',
    fps: +m('fps').toFixed(1),
    frameMs: +m('frameMs').toFixed(2),
    world1Ms: +m('world1').toFixed(2),
    combinedMs: +m('combined').toFixed(2),
    rasterMs: +m('raster').toFixed(2),
    computeMs: +m('compute').toFixed(2),
    renderMs: +m('render').toFixed(2),
    cpuSubmitMs: +m('cpuSubmit').toFixed(2),
    cpuUpdateMs: +m('cpuUpdate').toFixed(2),
    visClusters: Math.round(m('visClusters')),
    hwTris: Math.round(m('hwTris')),
    visTris: Math.round(m('visTris')),
    dagClusters: Math.round(m('dagClusters')),
    chunks: Math.round(m('chunks')),
    rejInst: Math.round(m('rejInst')),
    rejClust: Math.round(m('rejClust')),
    p2: Math.round(m('p2')),
    nsPerCluster: m('visClusters') > 0 ? +((m('raster') * 1e6) / m('visClusters')).toFixed(1) : -1,
  };
  console.log('RESULT ' + JSON.stringify(out));
  await browser.close();
}
main().catch((e) => {
  console.error('[diag] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
