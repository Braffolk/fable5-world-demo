/**
 * Measure the ?scene=forest testbed — our cull→raster→resolve on N instanced
 * trees, isolated. The harness we iterate against while closing the gap to the
 * reference compute-rasterizer. Reads the live HUD (frameMs + nanite counters)
 * and grabs a screenshot.
 *
 *   TREES=200000 LEAFDENSITY=4000 npx tsx tools/probe-forest.ts   # dev server :5173
 *   TREES=50000 DAG=1 MODE=cluster npx tsx tools/probe-forest.ts
 */
import { launchWebGPU, laasUrl } from './launch';

const TREES = process.env.TREES ?? '200000';
const LEAFDENSITY = process.env.LEAFDENSITY ?? '4000';
const DAG = process.env.DAG ?? '1';
const MODE = process.env.MODE ?? 'cluster';
const OCCL = process.env.OCCL ?? '1';
const SHOTNAME = process.env.SHOTNAME ?? `forest-${TREES}-d${LEAFDENSITY}-dag${DAG}`;

const median = (a: number[]): number => {
  const s = [...a].sort((x, y) => x - y);
  return s[Math.floor(s.length / 2)] ?? 0;
};

async function main(): Promise<void> {
  const width = 1280;
  const height = 720;
  const px = width * height;
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));
  page.on('console', (m) => {
    const t = m.text();
    if (t.startsWith('[forest]')) console.log(`   · ${t}`);
  });
  const url = laasUrl({
    scene: 'forest',
    width,
    height,
    freeze: false,
    extra: {
      trees: TREES,
      leafdensity: LEAFDENSITY,
      dag: DAG,
      occl: OCCL,
      nanitedbg: MODE,
      hier: process.env.HIER ?? '0',
      simband: process.env.SIMBAND ?? '0',
      lodnear: process.env.LODNEAR ?? '0',
      lodpow: process.env.LODPOW ?? '1',
    },
  });
  console.log(`[forest] trees=${TREES} density=${LEAFDENSITY} dag=${DAG} occl=${OCCL} mode=${MODE} — px=${px}`);
  const tBoot = Date.now();
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error != null),
    undefined,
    { timeout: 300000, polling: 250 },
  );
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);
  console.log(`[forest] booted in ${((Date.now() - tBoot) / 1000).toFixed(1)} s`);

  // sweep the imposter min-screen-size cull in ONE boot (0 = baseline geometry).
  // GPU PASS times (not frameMs — that's wall-clock): cull + depth + payload = the
  // real compute cost the HUD shows.
  const sweep = (process.env.INSTMINPX ?? '0').split(',');
  for (const v of sweep) {
    await page.evaluate((px) => {
      const n = (window as unknown as { __naniteView?: { setInstMinPx?(x: number): void } }).__naniteView;
      n?.setInstMinPx?.(Number(px));
    }, v);
    // collect EVERY frame stat + EVERY gpu pass into one flat record per frame
    const acc = new Map<string, number[]>();
    for (let i = 0; i < 36; i++) {
      const f = (await page.evaluate(async () => {
        if (window.__laas.settle) await window.__laas.settle(4);
        const st = window.__laas.stats;
        const c = (st?.counters ?? {}) as Record<string, number>;
        const g = (st?.gpuPasses ?? {}) as Record<string, number>;
        const rec: Record<string, number> = {
          fps: st?.fps ?? 0,
          frameMs: st?.frameMs ?? 0,
          cpuUpd: (c['cpu.updateMs100'] ?? 0) / 100,
          cpuSub: (c['cpu.submitMs100'] ?? 0) / 100,
          vis: c['nanite.visClusters'] ?? -1,
          chunks: c['nanite.chunks'] ?? -1,
          hwTris: c['nanite.hwTris'] ?? -1,
        };
        for (const [k, val] of Object.entries(g)) rec[`g:${k}`] = val; // all passes
        return rec;
      })) as Record<string, number>;
      if (i >= 10) for (const [k, val] of Object.entries(f)) (acc.get(k) ?? acc.set(k, []).get(k))?.push(val);
    }
    const r: Record<string, number> = {};
    for (const [k, a] of acc) r[k] = median(a);
    const gpuTot = (r['g:compute'] ?? 0) + (r['g:render'] ?? 0);
    const shot = `shots/forest/${SHOTNAME}-imp${v}.png`;
    await page.screenshot({ path: shot });

    // ── frame accounting: where do the milliseconds actually go? ───────────────
    console.log(`\n━━━ instMinPx=${v} · vis=${r.vis} chunks=${r.chunks} hwTris=${r.hwTris} → ${shot}`);
    console.log(
      `  WALL frameMs=${(r.frameMs ?? 0).toFixed(1)} (fps ${(r.fps ?? 0).toFixed(1)})  │  ` +
        `CPU update=${(r.cpuUpd ?? 0).toFixed(1)} submit=${(r.cpuSub ?? 0).toFixed(1)}  │  ` +
        `GPU compute=${(r['g:compute'] ?? 0).toFixed(1)} render=${(r['g:render'] ?? 0).toFixed(1)} (tot ${gpuTot.toFixed(1)})`,
    );
    const gap = (r.frameMs ?? 0) - Math.max(r.cpuSub ?? 0, gpuTot);
    const bound = (r.cpuSub ?? 0) > gpuTot * 1.2 ? 'CPU-ENCODE' : gpuTot > (r.cpuSub ?? 0) * 1.2 ? 'GPU' : 'BALANCED';
    console.log(`  → bound: ${bound}   unaccounted gap (stall/present/pipeline): ${gap.toFixed(1)} ms`);
    // per-pass GPU breakdown, biggest first
    const passes = Object.entries(r)
      .filter(([k]) => (k.startsWith('g:c.') || k.startsWith('g:r.')) && (r[k] ?? 0) > 0.02)
      .sort((a, b) => b[1] - a[1]);
    console.log('   GPU passes (ms):');
    for (const [k, ms] of passes) console.log(`     ${ms.toFixed(2).padStart(6)}  ${k.slice(2)}`);
  }
  await browser.close();
}

main().catch((e) => {
  console.error('[forest] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
