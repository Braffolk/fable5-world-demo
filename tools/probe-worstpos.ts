/**
 * probe-worstpos: decompose the per-pass GPU cost at a user-supplied WORST-PERF
 * camera POSITION by sweeping yaw to find the heaviest orientation — "looking
 * down the long alley" = the longest sightline = the most visible clusters at
 * every LOD = the peak SW-raster load. Dumps the full per-pass ledger (sorted,
 * median) + a screenshot at the worst yaw so the framing can be confirmed.
 *
 * Default position = the user's reported worst: cam (-4.2, 303.1, -1.4) @ T=11,
 * which read 43 fps / GPU compute 17.17 ms in-browser (≈2× the bm3 vista). The
 * "no guessing" instrument: every number here is a real per-pass timestamp at a
 * REPRESENTATIVE wide view, not a fast close-up.
 *
 *   npx tsx tools/probe-worstpos.ts                       # full beauty
 *   EXTRA=pure=1 npx tsx tools/probe-worstpos.ts          # isolate pure nanite
 *   X=-4.2 Y=303.1 Z=-1.4 T=11 PITCH=-0.12 npx tsx tools/probe-worstpos.ts
 *
 * Needs the dev server on :5173.
 */
import { launchWebGPU, laasUrl } from './launch';

const X = Number(process.env.X ?? -4.2);
const Y = Number(process.env.Y ?? 303.1);
const Z = Number(process.env.Z ?? -1.4);
const T = Number(process.env.T ?? 11);
const PITCH = Number(process.env.PITCH ?? -0.12);
const YAWS = Number(process.env.YAWS ?? 24);
const SAMPLES = Number(process.env.SAMPLES ?? 12);
const WIDTH = Number(process.env.WIDTH ?? 2592);
const HEIGHT = Number(process.env.HEIGHT ?? 1676);

function median(a: number[]): number {
  const s = a.filter((x) => Number.isFinite(x)).sort((x, y) => x - y);
  return s.length === 0 ? 0 : (s[Math.floor(s.length / 2)] as number);
}

interface Frame {
  g: Record<string, number>;
  frameMs: number;
  fps: number;
  c: Record<string, number>;
}

interface YawRow {
  yaw: number;
  compute: number;
  render: number;
  frameMs: number;
  fps: number;
  rDepth: number;
  rPayload: number;
  hwPass: number;
  scene: number;
  visClusters: number;
  hwTris: number;
}

async function main(): Promise<void> {
  const extra: Record<string, string> = { nanite: '1' };
  for (const kv of (process.env.EXTRA ?? '').split(',')) {
    if (!kv) continue;
    const [k, v] = kv.split('=');
    if (k) extra[k] = v ?? '1';
  }

  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));
  const url = laasUrl({ scene: 'world', width: WIDTH, height: HEIGHT, freeze: false, extra });
  console.log(`[worstpos] booting ${url}`);
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__laas && (window.__laas.ready || window.__laas.error != null), undefined, {
    timeout: 180000,
    polling: 250,
  });
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);
  await page.evaluate((t) => window.__laas.setTimeOfDay?.(t), T);

  const sampleAt = async (yaw: number): Promise<Record<string, number>> => {
    await page.evaluate(
      (p) => window.__laas.setPose?.({ p: [p.x, p.y, p.z], yaw: p.yaw, pitch: p.pitch }),
      { x: X, y: Y, z: Z, yaw, pitch: PITCH },
    );
    await page.evaluate(async () => window.__laas.settle && (await window.__laas.settle(8)));
    const acc = new Map<string, number[]>();
    for (let i = 0; i < SAMPLES; i++) {
      await page.evaluate(async () => window.__laas.settle && (await window.__laas.settle(3)));
      const f = (await page.evaluate(() => {
        const st = window.__laas.stats;
        return { g: st?.gpuPasses ?? {}, frameMs: st?.frameMs ?? 0, fps: st?.fps ?? 0, c: st?.counters ?? {} };
      })) as Frame;
      const merged: Record<string, number> = { ...f.g, frameMs: f.frameMs, fps: f.fps };
      merged['visClusters'] = f.c['nanite.visClusters'] ?? -1;
      merged['hwTris'] = f.c['nanite.hwTris'] ?? -1;
      for (const [k, v] of Object.entries(merged)) {
        if (!acc.has(k)) acc.set(k, []);
        acc.get(k)!.push(v);
      }
    }
    const out: Record<string, number> = {};
    for (const [k, a] of acc) out[k] = median(a);
    return out;
  };

  console.log(
    `[worstpos] pos (${X}, ${Y}, ${Z}) T=${T} pitch=${PITCH} — sweeping ${YAWS} yaws @${WIDTH}×${HEIGHT} (${SAMPLES} samples each)`,
  );
  const rows: YawRow[] = [];
  const ledgers: Record<string, number>[] = [];
  for (let i = 0; i < YAWS; i++) {
    const yaw = (i / YAWS) * Math.PI * 2;
    const g = await sampleAt(yaw);
    ledgers.push(g);
    rows.push({
      yaw,
      compute: g['compute'] ?? 0,
      render: g['render'] ?? 0,
      frameMs: g['frameMs'] ?? 0,
      fps: g['fps'] ?? 0,
      rDepth: g['c.nanRasterDepth'] ?? 0,
      rPayload: g['c.nanRasterPayload'] ?? 0,
      hwPass: g['r.nanHwPass'] ?? 0,
      scene: g['r.scene'] ?? 0,
      visClusters: g['visClusters'] ?? -1,
      hwTris: g['hwTris'] ?? -1,
    });
    process.stdout.write('.');
  }
  process.stdout.write('\n');

  // worst = highest GPU compute (the user's in-browser signal: compute 17 ms)
  const sorted = [...rows].sort((a, b) => b.compute - a.compute);
  console.log('\n[worstpos] per-yaw (sorted by GPU compute, worst first):');
  console.log('   yaw°   fps  frameMs  compute  render | rDepth rPayload  hwPass  r.scene | visCl   hwTris');
  for (const r of sorted) {
    const deg = ((r.yaw * 180) / Math.PI).toFixed(0).padStart(4);
    console.log(
      `  ${deg}  ${r.fps.toFixed(0).padStart(4)}  ${r.frameMs.toFixed(1).padStart(6)}  ` +
        `${r.compute.toFixed(2).padStart(6)}  ${r.render.toFixed(2).padStart(6)} | ` +
        `${r.rDepth.toFixed(2).padStart(6)} ${r.rPayload.toFixed(2).padStart(7)} ${r.hwPass.toFixed(2).padStart(7)} ${r.scene.toFixed(2).padStart(7)} | ` +
        `${String(r.visClusters).padStart(6)} ${String(r.hwTris).padStart(8)}`,
    );
  }

  const worst = sorted[0];
  if (!worst) throw new Error('no measurements');
  const worstLedger = ledgers[rows.indexOf(worst)] ?? {};
  console.log(
    `\n[worstpos] WORST yaw = ${((worst.yaw * 180) / Math.PI).toFixed(1)}° ` +
      `(cam "${X},${Y},${Z},${worst.yaw.toFixed(3)},${PITCH},60") — compute ${worst.compute.toFixed(2)} ms, ` +
      `${worst.fps.toFixed(0)} fps, ${worst.visClusters} visClusters, ${worst.hwTris} hwTris`,
  );
  console.log('[worstpos] FULL per-pass ledger at the worst yaw (median, ms, ≥0.05):');
  const ledRows = Object.entries(worstLedger)
    .filter(([k, m]) => (k.startsWith('r.') || k.startsWith('c.')) && !k.includes('__garbage') && m >= 0.05)
    .sort((a, b) => b[1] - a[1]);
  for (const [k, m] of ledRows) console.log(`  ${m.toFixed(2).padStart(7)} ms  ${k}`);

  // re-pose to worst + screenshot so the framing ("the long alley") is verifiable
  await page.evaluate(
    (p) => window.__laas.setPose?.({ p: [p.x, p.y, p.z], yaw: p.yaw, pitch: p.pitch }),
    { x: X, y: Y, z: Z, yaw: worst.yaw, pitch: PITCH },
  );
  await page.evaluate(async () => window.__laas.settle && (await window.__laas.settle(16)));
  await page.screenshot({ path: 'shots/perf/worstpos.png' });
  console.log('[worstpos] screenshot → shots/perf/worstpos.png');
  await browser.close();
}
main().catch((e) => {
  console.error('[worstpos] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
