/**
 * N9-C2 GATE — the leaf AGGREGATE DAG cut realised on the GPU (continuous crown
 * LOD, no pop). Boots the world with `?naniteleaf=1` (the area-preserving leaf
 * aggregate attached + the envelope extended to TREE_GEO_FAR) at a hero-tree
 * framing, terrain-DAG/shadows/occlusion off so the ONLY DAG-flagged clusters are
 * the leaf crowns ⇒ `nanite.dagClusters` is the isolated LEAF cut signal. Then:
 *
 *   BUILD: prints the boot's `[worldreg] leaf aggregate DAG … ms` (the boot-budget
 *   number that decides whether the build must move to the Worker path).
 *
 *   τ-SWEEP (fixed pose): tighten τ coarse→fine via setTau; the leaf dagClusters
 *   must rise MONOTONICALLY (the cut frontier walks toward hero-detail) and span a
 *   range — proof the aggregate's own/parent error bands drive a real cut.
 *
 *   ZOOM-SWEEP (τ=1): dolly AWAY from the trees; the leaf cut must shed clusters
 *   SMOOTHLY (no pop = no jump > 60%) as crowns taper to fewer/bigger leaves.
 *
 * Frames → shots/wip/leafzoom-* for the eyeball pass (crowns present + thinning,
 * never black/holey). Area-preservation (no balding) is proven in probe-aggregate;
 * this gate confirms the GPU cut realises it.
 *
 *   npx tsx tools/probe-leafzoom.ts                  # density 800 (fast build)
 *   LEAFDENSITY=4000 npx tsx tools/probe-leafzoom.ts # the real boot-budget number
 *   SHOT=3 npx tsx tools/probe-leafzoom.ts           # needs the dev server on :5173
 */

import type { CamPose } from '../src/core/Hooks';
import { launchWebGPU, laasUrl } from './launch';

const SHOT = process.env.SHOT ?? '7';
const LEAFDENSITY = process.env.LEAFDENSITY ?? '800';

declare global {
  interface Window {
    __laasNanite?: { setTau?(v: number): void; tau?(): number };
  }
}

let failures = 0;
const fail = (m: string): void => {
  failures++;
  console.error(`  FAIL ${m}`);
};

async function main(): Promise<void> {
  const width = 1280;
  const height = 720;
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  const errs: string[] = [];
  let aggLog = '';
  page.on('console', (m) => {
    const t = m.text();
    if (t.includes('leaf aggregate DAG')) aggLog = t;
    if (m.type() === 'error') errs.push(t);
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
      occl: '0',
      shot: SHOT,
      loderr: '1',
    },
  });
  console.log(`[leafzoom] SHOT=${SHOT} LEAFDENSITY=${LEAFDENSITY}`);
  console.log(`[leafzoom] ${url}`);
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__laas && (window.__laas.ready || window.__laas.error != null), undefined, {
    timeout: 240000,
    polling: 250,
  });
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);
  if (aggLog) console.log(`[leafzoom] ${aggLog}`);
  else fail('no "[worldreg] leaf aggregate DAG" boot log — leaf aggregate never built');
  const hasTau = await page.evaluate(() => typeof window.__laasNanite?.setTau === 'function');
  if (!hasTau) throw new Error('window.__laasNanite.setTau missing — DAG cut not wired');

  // perf snapshot at the boot pose, τ=1 — the per-instance-floor evidence (the
  // N8-HIC question the SPEC says C2 re-measures at leaf density)
  await page.evaluate(() => window.__laasNanite?.setTau?.(1));
  const snap = await page.evaluate(async () => {
    for (let i = 0; i < 30; i++) if (window.__laas.settle) await window.__laas.settle(8);
    const c = window.__laas.stats?.counters ?? {};
    return {
      frameMs: window.__laas.stats?.frameMs ?? -1,
      visClusters: c['nanite.visClusters'] ?? -1,
      dagClusters: c['nanite.dagClusters'] ?? -1,
      hwTris: c['nanite.hwTris'] ?? -1,
      trisK: c['nanite.trisK'] ?? -1,
      inst: c['nanite.inst'] ?? -1,
    };
  });
  console.log(
    `[leafzoom] PERF @boot τ=1: frame ${Number(snap.frameMs).toFixed(1)}ms | vis ${snap.visClusters} | leafDag ${snap.dagClusters} | hwTris ${snap.hwTris} | trisK ${snap.trisK} | inst ${snap.inst}`,
  );
  if (process.env.SNAPONLY) {
    await page.screenshot({ path: `shots/wip/leafzoom-snap-d${LEAFDENSITY}.png` });
    if (errs.length) fail(`${errs.length} console/page errors: ${errs.slice(0, 3).join(' | ')}`);
    await browser.close();
    if (failures > 0) {
      console.error(`[leafzoom] ${failures} FAILURES`);
      process.exit(1);
    }
    console.log('[leafzoom] snapshot-only OK');
    return;
  }

  const readDagCount = async (): Promise<number> => {
    const reads: number[] = [];
    for (let tries = 0; tries < 18 && reads.length < 9; tries++) {
      const v = await page.evaluate(async () => {
        if (window.__laas.settle) await window.__laas.settle(8);
        return window.__laas.stats?.counters['nanite.dagClusters'] ?? -1;
      });
      if (v >= 0) reads.push(v);
    }
    if (reads.length === 0) return -1;
    reads.sort((a, b) => a - b);
    return reads[Math.floor(reads.length / 2)] as number;
  };

  // --- τ-SWEEP (fixed pose) — leaf dagClusters must refine monotonically ---
  console.log('[leafzoom] τ-sweep (leaf cut, fixed pose):');
  const taus = [32, 16, 8, 4, 2, 1, 0.5, 0.25];
  const byTau: number[] = [];
  for (const t of taus) {
    await page.evaluate((tau) => window.__laasNanite?.setTau?.(tau), t);
    const n = await readDagCount();
    byTau.push(n);
    await page.screenshot({ path: `shots/wip/leafzoom-tau-${String(t).replace('.', 'p')}.png` });
    console.log(`  τ=${String(t).padStart(5)}  leaf dagClusters ${n}`);
  }
  for (let i = 1; i < byTau.length; i++) {
    const prev = byTau[i - 1] as number;
    const cur = byTau[i] as number;
    const noise = Math.max(2, Math.round(prev * 0.01));
    if (cur < prev - noise)
      fail(`τ-sweep non-monotonic at τ=${taus[i]}: ${cur} < ${prev} (tighter τ shed >${noise} leaf clusters)`);
  }
  const span = (byTau[byTau.length - 1] as number) - (byTau[0] as number);
  if (span <= 0) fail(`τ-sweep flat: leaf cut never refined (${byTau[0]} → ${byTau[byTau.length - 1]})`);

  // --- ZOOM-SWEEP (τ=1, dolly AWAY) — crowns must taper smoothly, no pop ---
  await page.evaluate(() => window.__laasNanite?.setTau?.(1));
  const pose0 = (await page.evaluate(() => window.__laas.getPose?.() ?? null)) as CamPose | null;
  if (pose0) {
    console.log('[leafzoom] zoom-sweep (dolly back along view ray, τ=1):');
    const fwd: [number, number, number] = [
      Math.sin(pose0.yaw) * Math.cos(pose0.pitch),
      Math.sin(pose0.pitch),
      -Math.cos(pose0.yaw) * Math.cos(pose0.pitch),
    ];
    const byStep: { d: number; n: number }[] = [];
    for (const d of [0, -12, -28, -55, -100, -180]) {
      await page.evaluate(
        ({ pose, fwd, d }) => {
          window.__laas.setPose?.({
            ...pose,
            p: [pose.p[0] + fwd[0] * d, pose.p[1] + fwd[1] * d, pose.p[2] + fwd[2] * d],
          });
        },
        { pose: pose0, fwd, d },
      );
      const n = await readDagCount();
      byStep.push({ d, n });
      await page.screenshot({ path: `shots/wip/leafzoom-step-${Math.abs(d)}.png` });
      console.log(`  ${d.toString().padStart(5)}m  leaf dagClusters ${n}`);
    }
    for (let i = 1; i < byStep.length; i++) {
      const a = (byStep[i - 1] as { n: number }).n;
      const b = (byStep[i] as { n: number }).n;
      const jump = Math.abs(b - a) / Math.max(1, a, b);
      if (jump > 0.6) fail(`zoom-sweep POP at step ${i}: ${a} → ${b} (${(jump * 100).toFixed(0)}% jump)`);
    }
  }

  if (errs.length) fail(`${errs.length} console/page errors: ${errs.slice(0, 3).join(' | ')}`);
  await browser.close();

  if (failures > 0) {
    console.error(`[leafzoom] ${failures} FAILURES`);
    process.exit(1);
  }
  console.log('[leafzoom] leaf aggregate cut OK — monotonic refine, smooth taper, no errors');
}

main().catch((e) => {
  console.error('[leafzoom] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
