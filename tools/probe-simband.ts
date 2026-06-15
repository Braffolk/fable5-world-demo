/**
 * D-N43 Stage 0.5 SIM — BOUND the cross-instance-aggregation win BEFORE building it.
 *
 * Boots the flood frame (shot 7 forest interior, ?naniteleaf=1, occl ON = default,
 * density 4000 — the headline capture) and sweeps the DISTANCE-BANDED τ knob
 * (simBandD): τ_eff = τ·(1 + d/bandD). Near field stays sharp; far crowns collapse
 * toward their per-mesh ROOT — the ≥1-cluster/instance FLOOR the real Stage-1 merge
 * must break. This is the user's own reframe ("use the least-detailed DAG levels far
 * away"), measured: the residual cluster count + tris/px at the plateau IS the floor.
 *
 * Reads the NEW per-frame visTris counter (exact Σ cluster.triCount) ⇒ tris/px is
 * MEASURED, not estimated. The question this answers:
 *   does pure per-mesh LOD (coarsest levels far) approach ~1 tri/px  ⇒ merge alone may suffice
 *   or does tris/px PLATEAU well above 1 at the cluster floor          ⇒ voxel far-field MANDATORY
 *
 * Plus a FLOOR anchor (uniform huge τ, band off) = every DAG cluster forced to root
 * = the absolute per-mesh floor (visible crowns × root clusters), for reference.
 *
 *   LEAFDENSITY=4000 npx tsx tools/probe-simband.ts   # needs dev server on :5173
 */

import { launchWebGPU, laasUrl } from './launch';

const SHOT = process.env.SHOT ?? '7';
const LEAFDENSITY = process.env.LEAFDENSITY ?? '4000';

// { tau, band }: band=0 ⇒ uniform τ. Baseline = the flood; banded sweep walks
// toward the floor keeping the near field sharp; FLOOR = uniform huge τ (all roots).
const CONFIGS: { label: string; tau: number; band: number }[] = [
  { label: 'baseline', tau: 1, band: 0 },
  { label: 'band 4000', tau: 1, band: 4000 },
  { label: 'band 2000', tau: 1, band: 2000 },
  { label: 'band 1000', tau: 1, band: 1000 },
  { label: 'band  500', tau: 1, band: 500 },
  { label: 'band  250', tau: 1, band: 250 },
  { label: 'band  120', tau: 1, band: 120 },
  { label: 'band   60', tau: 1, band: 60 },
  { label: 'band   30', tau: 1, band: 30 },
  { label: 'FLOOR τ1e4', tau: 1e4, band: 0 },
];

// NOTE: probe-floodtau.ts already `declare global`s a narrower __laasNanite shape;
// re-declaring it here with extra members collides (TS2717). Each page.evaluate
// callback runs in the BROWSER (can't close over a Node helper), so the cast to
// reach setSimBand is inlined per-callback below.

async function main(): Promise<void> {
  const width = 1280;
  const height = 720;
  const px = width * height;
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));
  const url = laasUrl({
    scene: 'world',
    width,
    height,
    freeze: false,
    extra: {
      nanite: '1',
      naniteleaf: '1',
      naniteleafdensity: LEAFDENSITY,
      nanitedterrain: '0', // terrain DAG off ⇒ dagClusters isolates the leaf/bark floor
      nanshadow: '0',
      occl: '1',
      shot: SHOT,
      loderr: '1',
    },
  });
  console.log(
    `[simband] SHOT=${SHOT} LEAFDENSITY=${LEAFDENSITY} occl ON — px=${px} (${width}×${height})`,
  );
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error != null),
    undefined,
    { timeout: 240000, polling: 250 },
  );
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);
  // confirm the sim hook landed (else the sweep is a no-op)
  const hasHook = await page.evaluate(
    () =>
      typeof (window as unknown as { __laasNanite?: { setSimBand?: unknown } }).__laasNanite
        ?.setSimBand === 'function',
  );
  if (!hasHook) throw new Error('__laasNanite.setSimBand missing — sim knob not wired');

  const median = (a: number[]): number => {
    const s = [...a].sort((x, y) => x - y);
    return s[Math.floor(s.length / 2)] ?? 0;
  };

  const sampleAt = async (tau: number, band: number): Promise<Record<string, number>> => {
    await page.evaluate(
      ([t, b]) => {
        const n = (
          window as unknown as {
            __laasNanite?: { setTau?(v: number): void; setSimBand?(v: number): void };
          }
        ).__laasNanite;
        n?.setTau?.(t);
        n?.setSimBand?.(b);
      },
      [tau, band] as [number, number],
    );
    const acc = new Map<string, number[]>();
    for (let i = 0; i < 24; i++) {
      const f = (await page.evaluate(async () => {
        if (window.__laas.settle) await window.__laas.settle(6);
        const st = window.__laas.stats;
        const g = st?.gpuPasses ?? {};
        const c = st?.counters ?? {};
        return {
          frameMs: st?.frameMs ?? 0,
          payload: g['c.nanRasterPayload'] ?? 0,
          depth: g['c.nanRasterDepth'] ?? 0,
          clusterCull: g['c.nanClusterCull'] ?? 0,
          vis: c['nanite.visClusters'] ?? -1,
          dag: c['nanite.dagClusters'] ?? -1,
          visTris: c['nanite.visTris'] ?? -1,
          dagTris: c['nanite.dagTris'] ?? -1,
          hwTris: c['nanite.hwTris'] ?? -1,
        };
      })) as Record<string, number>;
      if (i >= 8) for (const [k, v] of Object.entries(f)) (acc.get(k) ?? acc.set(k, []).get(k))?.push(v);
    }
    const out: Record<string, number> = {};
    for (const [k, a] of acc) out[k] = median(a);
    return out;
  };

  // tris/px split: ALL = total SW (terrain base inflates it when terrain DAG off);
  // LEAF = DAG-only (the foliage flood, the real target — answers the voxel question).
  console.log(
    '   config    | frameMs | visClusters | dagClusters |   tris/px(all) | tris/px(leaf) | SWraster | frame/Mcl',
  );
  const rows: { label: string; r: Record<string, number> }[] = [];
  for (const cfg of CONFIGS) {
    const r = await sampleAt(cfg.tau, cfg.band);
    rows.push({ label: cfg.label, r });
    const total = (r.visTris >= 0 ? r.visTris : 0) + (r.hwTris >= 0 ? r.hwTris : 0);
    const trisPx = (total / px).toFixed(2);
    const leafPx = (r.dagTris >= 0 ? r.dagTris / px : 0).toFixed(2);
    const swRaster = (r.payload + r.depth).toFixed(2);
    const perMcl = r.vis > 0 ? (r.frameMs / (r.vis / 1e6)).toFixed(1) : '—';
    console.log(
      `  ${cfg.label.padEnd(10)} | ${r.frameMs.toFixed(1).padStart(7)} | ${String(r.vis).padStart(11)} | ${String(r.dag).padStart(11)} | ${trisPx.padStart(14)} | ${leafPx.padStart(13)} | ${swRaster.padStart(8)} | ${perMcl}`,
    );
  }

  // VERDICT: the LEAF (DAG-only) numbers answer Stage 1's two questions:
  //   (1) is the flood cluster-count or triangle-density bound? → leaf clusters vs tris/px
  //   (2) does per-mesh LOD bottom out above screen density? → leaf floor tris/px
  const base = rows[0]?.r;
  const floor = rows[rows.length - 1]?.r;
  const tightBand = rows[rows.length - 2]?.r; // band 30 — the most aggressive realistic
  const leafPxOf = (r?: Record<string, number>): number => (r && r.dagTris >= 0 ? r.dagTris / px : 0);
  if (base && floor && tightBand) {
    console.log('');
    console.log(
      `[simband] BASELINE (flood):  ${base.dag} leaf-clusters, leaf ${leafPxOf(base).toFixed(2)} tris/px, ${base.frameMs.toFixed(1)} ms`,
    );
    console.log(
      `[simband] FLOOR (all roots): ${floor.dag} leaf-clusters, leaf ${leafPxOf(floor).toFixed(3)} tris/px, ${floor.frameMs.toFixed(1)} ms`,
    );
    const clusterDrop = floor.dag > 0 ? (base.dag / Math.max(1, floor.dag)).toFixed(0) : '—';
    console.log(
      `[simband] LEAF cluster floor: ${base.dag} → ${floor.dag} (${clusterDrop}× LOD range); the per-mesh ≥1-cl/instance floor.`,
    );
    const leafFloor = leafPxOf(floor);
    console.log(
      `[simband] VERDICT: leaf floor = ${leafFloor.toFixed(3)} tris/px (${leafFloor < 1 ? 'SUB-PIXEL' : 'over screen density'}) at ${floor.dag} clusters. ${
        leafFloor < 1
          ? 'The flood is per-cluster OVERHEAD, not triangle density — the aggregate already thins leaves below 1 tri/px at the coarse end. ⇒ cross-instance MERGE (fewer shared clusters) is the lever and likely SUFFICES ALONE for foliage; voxels = far-tail nicety, NOT mandatory for this hero.'
          : 'Per-mesh roots stay above screen density ⇒ voxel far-field mandatory.'
      }`,
    );
  }
  await browser.close();
}

main().catch((e) => {
  console.error('[simband] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
