/**
 * N8-D2b terrain LOD-DAG wiring validation. A/Bs the discrete WINDOW terrain
 * (?nanitedterrain=0) against the adaptive DAG terrain (?nanitedterrain=N) at a
 * near ground pose + an elevated vista, in lit and per-cluster-tint (?nandbg=
 * cluster) modes. The DAG path must:
 *   - render CRACK-FREE (no sky/background holes), near and far;
 *   - match the window terrain up close (DECODE correct — same heights/grid);
 *   - show ADAPTIVE cluster sizes in the tint view (big patches on plains / at
 *     distance from the cut, fine speckle on cliffs) vs the window's uniform grid.
 *
 *   npx tsx tools/probe-dterrain.ts [gridN=512]   # needs the dev server on :5173
 */
import type { CamPose } from '../src/core/Hooks';
import { laasUrl, launchWebGPU } from './launch';

const W = 1280;
const H = 720;
const DAG_N = Math.max(2, Math.floor(Number(process.argv[2] ?? '512')));
const TILES = Math.max(1, Math.floor(Number(process.argv[3] ?? '1')));

interface Sample {
  dag: number;
  ms: number;
}

const POOL = process.env.POOL === '1'; // route terrain tiles through the streaming pool (2b-1)

async function capture(gridN: number, nandbg: string): Promise<{ near: Sample; vista: Sample }> {
  const tag = `g${gridN}-t${TILES}-${POOL ? 'pool-' : ''}${nandbg || 'lit'}`;
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  page.on('console', (m) => {
    const t = m.text();
    if (t.includes('terrain DAG')) console.log(`  [${tag}] ${t.replace(/^\[laas]\s*/, '')}`);
  });
  const extra: Record<string, string> = {
    nanite: '1',
    nanshadow: '0',
    occl: '1',
    nanitedterrain: String(gridN),
  };
  if (TILES > 1) extra.nanitedtiles = String(TILES);
  if (POOL) extra.nanitedpool = '1';
  if (nandbg) extra.nandbg = nandbg;
  const url = laasUrl({ scene: 'world', width: W, height: H, freeze: true, extra });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error != null),
    undefined,
    { timeout: 180000, polling: 250 },
  );
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot (${tag}): ${err}`);
  const sample = async (): Promise<Sample> => {
    for (let t = 0; t < 12; t++) {
      const v = await page.evaluate(async () => {
        if (window.__laas.settle) await window.__laas.settle(16);
        return {
          dag: window.__laas.stats?.counters['nanite.dagClusters'] ?? -1,
          ms: window.__laas.stats?.frameMs ?? -1,
        };
      });
      if (v.dag >= 0) return v;
    }
    return { dag: -1, ms: -1 };
  };
  const near = await sample();
  await page.screenshot({ path: `shots/wip/dterrain-${tag}-near.png` });
  // elevated vista — lift + pitch down to expose the near→far cluster gradient
  const pose = (await page.evaluate(() => window.__laas.getPose?.() ?? null)) as CamPose | null;
  if (!pose) throw new Error('getPose missing');
  await page.evaluate((p) => {
    window.__laas.setPose?.({ yaw: p.yaw, pitch: -0.38, p: [p.p[0], p.p[1] + 220, p.p[2]] });
  }, pose);
  const vista = await sample();
  await page.screenshot({ path: `shots/wip/dterrain-${tag}-vista.png` });
  await browser.close();
  return { near, vista };
}

async function main(): Promise<void> {
  const configs: Array<{ gridN: number; nandbg: string }> = [
    { gridN: 0, nandbg: '' },
    { gridN: DAG_N, nandbg: '' },
    { gridN: 0, nandbg: 'cluster' },
    { gridN: DAG_N, nandbg: 'cluster' },
  ];
  for (const c of configs) {
    const r = await capture(c.gridN, c.nandbg);
    const label = `g${c.gridN}/${c.nandbg || 'lit'}`;
    console.log(
      `[dterrain] ${label.padEnd(14)} near ${String(r.near.dag).padStart(7)} cl ${r.near.ms.toFixed(1)}ms | ` +
        `vista ${String(r.vista.dag).padStart(7)} cl ${r.vista.ms.toFixed(1)}ms`,
    );
  }
  console.log(`[dterrain] inspect shots/wip/dterrain-g{0,${DAG_N}}-{lit,cluster}-{near,vista}.png`);
  console.log(
    '[dterrain] expect: gN lit ≈ g0 lit up close (decode) + crack-free; gN cluster shows VARYING ' +
      'cluster sizes (adaptive) vs g0 uniform window grid; dagClusters drops from near→vista (cut).',
  );
}
main().catch((e) => {
  console.error('[dterrain] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
