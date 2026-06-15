/**
 * N8-HIC measurement — the architectural fork. The N9-C2 flood ledger was measured
 * with `occl=0` (occlusion OFF), so the EXISTING per-instance + per-cluster two-phase
 * occlusion was never applied to it. This probe boots the SAME flood frame (shot 7
 * forest interior, `?naniteleaf=1`, terrain-DAG/shadow off) with occlusion OFF vs ON
 * and dumps the visible-cluster / chunk / frame-ms deltas, to decide:
 *
 *   occl ON cuts visClusters a LOT  ⇒ the flood is OCCLUDED geometry → HIC = make the
 *     occlusion HIERARCHICAL (cull spatial regions at once, cheaper than per-instance).
 *   occl ON barely moves visClusters ⇒ the flood is GENUINELY VISIBLE crowns → HIC must
 *     AGGREGATE distant regions (region-LOD proxy), not just cull them.
 *
 * `chunks` ≈ Σ ceil(clusterCount/64) over instances that survived the instance cull
 * (the per-instance dispatch floor D-N41 named); `inst` = total bound instances.
 *
 *   LEAFDENSITY=800  npx tsx tools/probe-hic.ts   # fast build, the RATIO
 *   LEAFDENSITY=4000 npx tsx tools/probe-hic.ts   # the real flood density
 */

import { launchWebGPU, laasUrl } from './launch';

const SHOT = process.env.SHOT ?? '7';
const LEAFDENSITY = process.env.LEAFDENSITY ?? '800';

interface Snap {
  frameMs: number;
  visClusters: number;
  dagClusters: number;
  chunks: number;
  hwTris: number;
  trisK: number;
  inst: number;
  rejInst: number;
  rejClust: number;
}

async function boot(occl: boolean): Promise<{ snap: Snap; errs: string[] }> {
  const width = 1280;
  const height = 720;
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  const errs: string[] = [];
  page.on('console', (m) => {
    if (m.type() === 'error') errs.push(m.text());
  });
  page.on('pageerror', (e) => errs.push('PAGEERR ' + e.message));

  const url = laasUrl({
    scene: 'world',
    width,
    height,
    freeze: true,
    extra: {
      nanite: '1',
      naniteleaf: '1',
      naniteleafdensity: LEAFDENSITY,
      nanitedterrain: '0',
      nanshadow: '0',
      occl: occl ? '1' : '0',
      shot: SHOT,
      loderr: '1',
    },
  });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error != null),
    undefined,
    { timeout: 240000, polling: 250 },
  );
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot (occl=${occl}): ${err}`);
  await page.evaluate(() => window.__laasNanite?.setTau?.(1));
  const snap = (await page.evaluate(async () => {
    for (let i = 0; i < 30; i++) if (window.__laas.settle) await window.__laas.settle(8);
    const c = window.__laas.stats?.counters ?? {};
    return {
      frameMs: window.__laas.stats?.frameMs ?? -1,
      visClusters: c['nanite.visClusters'] ?? -1,
      dagClusters: c['nanite.dagClusters'] ?? -1,
      chunks: c['nanite.chunks'] ?? -1,
      hwTris: c['nanite.hwTris'] ?? -1,
      trisK: c['nanite.trisK'] ?? -1,
      inst: c['nanite.inst'] ?? -1,
      rejInst: c['nanite.rejInst'] ?? -1,
      rejClust: c['nanite.rejClust'] ?? -1,
    };
  })) as Snap;
  await page.screenshot({ path: `shots/wip/hic-occl${occl ? '1' : '0'}-d${LEAFDENSITY}.png` });
  await browser.close();
  return { snap, errs };
}

function show(label: string, s: Snap): void {
  console.log(
    `  ${label.padEnd(10)} frame ${Number(s.frameMs).toFixed(1)}ms | vis ${s.visClusters} | leafDag ${s.dagClusters} | chunks ${s.chunks} | hwTris ${s.hwTris} | trisK ${s.trisK} | inst ${s.inst} | rejI ${s.rejInst} | rejC ${s.rejClust}`,
  );
}

async function main(): Promise<void> {
  console.log(`[hic] flood occl OFF vs ON — SHOT=${SHOT} LEAFDENSITY=${LEAFDENSITY}`);
  const off = await boot(false);
  const on = await boot(true);
  show('occl OFF', off.snap);
  show('occl ON', on.snap);
  const dVis = off.snap.visClusters - on.snap.visClusters;
  const pct = off.snap.visClusters > 0 ? (100 * dVis) / off.snap.visClusters : 0;
  console.log(
    `[hic] occlusion culls ${dVis} visClusters (${pct.toFixed(1)}%); frame ${Number(off.snap.frameMs).toFixed(1)} → ${Number(on.snap.frameMs).toFixed(1)} ms`,
  );
  console.log(
    `[hic] VERDICT: ${pct > 40 ? 'occlusion dominates → HIC = hierarchical OCCLUSION cull' : 'occlusion weak → flood is VISIBLE → HIC = region-LOD AGGREGATION'}`,
  );
  const allErrs = [...off.errs, ...on.errs];
  if (allErrs.length) console.error(`[hic] ${allErrs.length} console/page errors: ${allErrs.slice(0, 4).join(' | ')}`);
}

main().catch((e) => {
  console.error('[hic] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
