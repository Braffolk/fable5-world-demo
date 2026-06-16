/**
 * SHADOW-HIER (#54) gate: the shadow culls now run the hierarchical DAG-BFS
 * (?shadowhier=1, default) instead of brute-forcing every cluster (?shadowhier=0).
 * Boots a shadow-heavy bookmark for BOTH shadow paths (clipmap default + cascades
 * via ?shadowclip=0), captures boot errors, screenshots hier vs brute for a visual
 * PARITY A/B, and reports fps + per-level/per-cascade cluster counts STATIC and
 * MOVING (moving shTotal = the per-frame raster proxy). Parity = the shadows look
 * identical; the win is hier dispatches the cut top-down (no per-instance chunk
 * expansion) so cluster counts should match or drop.
 *
 *   npx tsx tools/probe-shadowhier.ts            # bm7 forest interior, clipmap
 *   SHOT=3 npx tsx tools/probe-shadowhier.ts     # bm3 vista
 *   CLIP=0 npx tsx tools/probe-shadowhier.ts     # the 4-cascade path instead
 */
import { launchWebGPU, laasUrl } from './launch';

interface Result {
  err: string | null;
  fpsStatic: number;
  fpsMoving: number;
  shStatic: number;
  shMoving: number;
  counts: number[];
}

function median(a: number[]): number {
  const s = a.filter((x) => x > 0).sort((x, y) => x - y);
  return s.length === 0 ? 0 : (s[Math.floor(s.length / 2)] as number);
}

const WIDTH = Number(process.env.WIDTH ?? 1280);
const HEIGHT = Number(process.env.HEIGHT ?? 720);
const CLIP = (process.env.CLIP ?? '1') as '0' | '1';

async function boot(hier: '0' | '1'): Promise<Result> {
  const shot = process.env.SHOT ?? '7';
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error(`[pageerror hier=${hier}]`, e.message));
  const extra: Record<string, string> = { nanite: '1', shot, shadowclip: CLIP, shadowhier: hier };
  // optional ?hierdepth override (brute ignores it) — quantify the BFS-depth lever
  if (process.env.HIERDEPTH) extra.hierdepth = process.env.HIERDEPTH;
  const url = laasUrl({
    scene: 'world',
    width: WIDTH,
    height: HEIGHT,
    freeze: false, // shadow.run must execute every frame
    extra,
  });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__laas && (window.__laas.ready || window.__laas.error != null), undefined, {
    timeout: 180000,
    polling: 250,
  });
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) {
    await browser.close();
    return { err, fpsStatic: 0, fpsMoving: 0, shStatic: 0, shMoving: 0, counts: [] };
  }
  await page.evaluate(async () => window.__laas.settle && (await window.__laas.settle(40)));

  // STATIC window: fps + shadow cluster total + per-level counts
  const fpsS: number[] = [];
  const shS: number[] = [];
  for (let i = 0; i < 20; i++) {
    await page.evaluate(async () => window.__laas.settle && (await window.__laas.settle(4)));
    const m = (await page.evaluate(() => ({
      fps: window.__laas.stats?.fps ?? 0,
      sh: window.__laas.stats?.counters['nanite.shTotal'] ?? -1,
    }))) as { fps: number; sh: number };
    fpsS.push(m.fps);
    if (m.sh > 0) shS.push(m.sh);
  }
  const counts = (await page.evaluate(() => {
    const c = window.__laas.stats?.counters ?? {};
    const out: number[] = [];
    for (let i = 0; i < 10; i++) {
      const v = c[`nanite.shC${i}`];
      if (typeof v === 'number' && v >= 0) out.push(v);
    }
    return out;
  })) as number[];
  await page.screenshot({ path: `shots/wip/shier-clip${CLIP}-${hier}-static.png` });

  // MOVING window: drift the camera so levels re-raster; median shTotal + fps
  const shM: number[] = [];
  const fpsM: number[] = [];
  for (let i = 0; i < 40; i++) {
    const m = (await page.evaluate(async (step) => {
      const h = window.__laas;
      if (h.getPose && h.setPose) {
        const p = h.getPose();
        h.setPose({ ...p, yaw: p.yaw + 0.01, p: [p.p[0] + step, p.p[1], p.p[2] + step * 0.6] });
      }
      if (h.settle) await h.settle(1);
      return {
        sh: h.stats?.counters['nanite.shTotal'] ?? -1,
        fps: h.stats?.fps ?? 0,
      };
    }, 1.0)) as { sh: number; fps: number };
    if (i >= 10) {
      if (m.sh > 0) shM.push(m.sh);
      if (m.fps > 0) fpsM.push(m.fps);
    }
  }
  await page.screenshot({ path: `shots/wip/shier-clip${CLIP}-${hier}-moving.png` });
  await browser.close();
  return {
    err: null,
    fpsStatic: median(fpsS),
    fpsMoving: median(fpsM),
    shStatic: median(shS),
    shMoving: median(shM),
    counts,
  };
}

async function main(): Promise<void> {
  const shot = process.env.SHOT ?? '7';
  const path = CLIP === '1' ? 'clipmap' : 'cascades';
  console.log(`[shadowhier] bm${shot} ${path} — hier (=1, default) vs brute (=0)`);
  console.log('[shadowhier] booting hier …');
  const hier = await boot('1');
  console.log('[shadowhier] booting brute …');
  const brute = await boot('0');

  const row = (label: string, r: Result): void => {
    if (r.err) {
      console.log(`  ${label}: BOOT ERROR — ${r.err}`);
      return;
    }
    console.log(
      `  ${label}: fps static ${r.fpsStatic.toFixed(1)} moving ${r.fpsMoving.toFixed(1)}  |  shTotal static ${r.shStatic.toLocaleString()} moving ${r.shMoving.toLocaleString()}  |  per-band [${r.counts.join(', ')}]`,
    );
  };
  console.log('\n[shadowhier] RESULTS:');
  row('hier ', hier);
  row('brute', brute);
  if (!hier.err && !brute.err && brute.shMoving > 0) {
    console.log(
      `\n[shadowhier] MOVING shTotal: hier ${hier.shMoving.toLocaleString()} vs brute ${brute.shMoving.toLocaleString()} = ×${(hier.shMoving / brute.shMoving).toFixed(2)}`,
    );
  }
  console.log(`[shadowhier] screenshots → shots/wip/shier-clip${CLIP}-{0,1}-{static,moving}.png`);
}
main().catch((e) => {
  console.error('[shadowhier] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
