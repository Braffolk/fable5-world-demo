/**
 * probe-pcfixed: per-cluster-fixed-overhead attribution probe. Same boot/warm/
 * median flow as probe-diag, but injects arbitrary URL flags via EXTRA="k=v,k2=v2"
 * (notably wgcache, vcompact, rdbg) so the wgcache A/B isolates the per-cluster
 * launch+makeCtx+broadcast fixed cost from the per-pixel coverage cost.
 *
 * Env: TREES INSTMINPX SIMBAND LODNEAR LODPOW W H NANITEDBG SAMPLES WARM
 *      EXTRA="wgcache=0"  (comma-separated k=v injected straight into the URL)
 */
import { laasUrl, launchWebGPU } from './launch';

const W = Number(process.env.W ?? 1280);
const H = Number(process.env.H ?? 720);
const TREES = process.env.TREES ?? '40000';
const INSTMINPX = process.env.INSTMINPX ?? '128';
const SIMBAND = process.env.SIMBAND ?? '6';
const LODNEAR = process.env.LODNEAR ?? '4';
const LODPOW = process.env.LODPOW ?? '0.6';
const NANITEDBG = process.env.NANITEDBG ?? '';
const SAMPLES = Number(process.env.SAMPLES ?? 22);
const WARM = Number(process.env.WARM ?? 120);
const EXTRA = process.env.EXTRA ?? '';

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
  if (NANITEDBG) extra['nanitedbg'] = NANITEDBG;
  for (const kv of EXTRA.split(',')) {
    const t = kv.trim();
    if (!t) continue;
    const eq = t.indexOf('=');
    if (eq > 0) extra[t.slice(0, eq)] = t.slice(eq + 1);
  }

  const configStr =
    `trees=${TREES} instminpx=${INSTMINPX} simband=${SIMBAND} lodnear=${LODNEAR} ` +
    `lodpow=${LODPOW} W=${W} H=${H} dbg=${NANITEDBG || 'full'} EXTRA=[${EXTRA}]`;

  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));
  const url = laasUrl({ scene: 'forest', width: W, height: H, freeze: false, extra });
  console.error(`[pcfixed] booting ${url}`);

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
        world1: g['c.nanRasterWorld1'] ?? 0,
        combined: g['c.nanRasterCombined'] ?? 0,
        raster: (g['c.nanRasterWorld1'] ?? 0) || (g['c.nanRasterCombined'] ?? 0),
        compute: g['compute'] ?? 0,
        render: g['render'] ?? 0,
        cpuSubmit: (c['cpu.submitMs100'] ?? 0) / 100,
        visClusters: c['nanite.visClusters'] ?? -1,
        hwTris: c['nanite.hwTris'] ?? -1,
        visTris: c['nanite.visTris'] ?? -1,
      };
    })) as Record<string, number>;
    for (const [k, v] of Object.entries(r)) {
      if (!acc.has(k)) acc.set(k, []);
      acc.get(k)!.push(v);
    }
  }
  const m = (k: string): number => median(acc.get(k) ?? [0]);
  const vc = m('visClusters');
  const vt = m('visTris');
  const out = {
    config: configStr,
    fps: +m('fps').toFixed(1),
    frameMs: +m('frameMs').toFixed(2),
    rasterMs: +m('raster').toFixed(2),
    computeMs: +m('compute').toFixed(2),
    renderMs: +m('render').toFixed(2),
    cpuSubmitMs: +m('cpuSubmit').toFixed(2),
    visClusters: Math.round(vc),
    hwTris: Math.round(m('hwTris')),
    visTris: Math.round(vt),
    trisPerCluster: vc > 0 ? +(vt / vc).toFixed(1) : -1,
    nsPerCluster: vc > 0 ? +((m('raster') * 1e6) / vc).toFixed(1) : -1,
    nsPerTri: vt > 0 ? +((m('raster') * 1e6) / vt).toFixed(2) : -1,
  };
  console.log('RESULT ' + JSON.stringify(out));
  await browser.close();
}
main().catch((e) => {
  console.error('[pcfixed] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
