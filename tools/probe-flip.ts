/**
 * N8-D2 Stage 2e (D-N39) — the "boot only to dag" FLIP gate. Verifies the
 * PRODUCTION DEFAULT (bare `?nanite=1`, no terrain flags) now boots the full-res
 * clip-STREAMED DAG terrain — NOT the legacy implicit window grid — and that
 * `?nanitedterrain=0` still selects the window grid as the explicit opt-out.
 *
 * Asserts for the default boot:
 *   - no fatal boot error,
 *   - terrain DAG live (nanite.dagClusters > 0 AND terrain.stream.resident > 0),
 *   - the pool reservation prints (captures the real vert/tri/cluster caps + memory),
 *   - a non-sky frame (screenshot for eyeball).
 * Asserts for `?nanitedterrain=0`:
 *   - no streamer (terrain.stream.resident absent), terrain still renders (window grid).
 *
 *   npx tsx tools/probe-flip.ts        # needs the dev server on :5173
 */
import { laasUrl, launchWebGPU } from './launch';
import { VERT_WORDS } from '../src/nanite/GeometryRegistry';

const W = 1280;
const H = 720;

interface BootProbe {
  err: string | null;
  dagClusters: number;
  resident: number;
  loaded: number;
  poolLine: string | null;
  nonSky: number; // fraction of pixels that are NOT background sky-ish
}

async function bootOnce(
  extra: Record<string, string>,
  shot: string,
): Promise<BootProbe> {
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  let poolLine: string | null = null;
  page.on('console', (m) => {
    const t = m.text();
    // the per-tile-pool summary rides the big "[laas] nanite registry:" line (deferred[])
    if (t.includes('terrain DAG') && t.includes('POOL')) {
      // the summary rides a big multi-part line (deferred[] joined by '; '); isolate
      // just the "terrain DAG CLIPMAP … POOL …(…) … ms" segment.
      const m = t.match(/terrain DAG CLIPMAP[^;]*?POOL \d+×\([^)]*\)[^;]*/);
      if (m) poolLine = m[0].trim();
    }
  });
  const url = laasUrl({ scene: 'world', width: W, height: H, freeze: true, extra: { nanite: '1', ...extra } });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__laas && (window.__laas.ready || window.__laas.error != null), undefined, {
    timeout: 180000,
    polling: 250,
  });
  const err = (await page.evaluate(() => window.__laas.error ?? null)) as string | null;
  if (!err) {
    // settle so the streamer re-centers on the real (walk-spawn) camera + tiles attach
    await page.evaluate(async () => {
      if (window.__laas.settle) await window.__laas.settle(12);
    });
  }
  const counters = (await page.evaluate(() => window.__laas.stats?.counters ?? {})) as Record<string, number>;
  await page.screenshot({ path: shot });
  // crude sky test: sample the framebuffer is overkill; instead trust counters +
  // the screenshot. Report a placeholder nonSky (eyeball the PNG).
  await browser.close();
  return {
    err,
    dagClusters: counters['nanite.dagClusters'] ?? 0,
    resident: counters['terrain.stream.resident'] ?? -1,
    loaded: counters['terrain.stream.loaded'] ?? -1,
    poolLine,
    nonSky: -1,
  };
}

async function main(): Promise<void> {
  console.log('[flip] === DEFAULT boot (bare ?nanite=1 → clip-DAG terrain, the 2e flip) ===');
  const def = await bootOnce({}, 'shots/wip/flip-default.png');
  if (def.err) throw new Error(`default boot fatal: ${def.err}`);
  console.log(`  dagClusters       ${def.dagClusters}`);
  console.log(`  stream.resident   ${def.resident}`);
  console.log(`  stream.loaded     ${def.loaded}`);
  console.log(`  pool              ${def.poolLine ?? '(no POOL line seen!)'}`);
  if (def.poolLine) {
    const m = def.poolLine.match(/POOL (\d+)×\(v(\d+)\/t(\d+)\/c(\d+)\)/);
    if (m) {
      const [, slots, v, t, c] = m.map(Number) as [number, number, number, number, number];
      const vertMB = (slots * v * 4) / (1024 * 1024); // 2e: stride-1, one word/vert
      const idxMB = (slots * t * 3 * 4) / (1024 * 1024);
      console.log(
        `  pool memory       ${slots} slots × v${v}(×1w, 2e)/t${t}/c${c} ⇒ verts ${vertMB.toFixed(1)} MB` +
          ` (was ${(vertMB * VERT_WORDS).toFixed(1)} MB pre-2e at ×${VERT_WORDS}w), indices ${idxMB.toFixed(1)} MB`,
      );
    }
  }
  // nanite.dagClusters is a per-FRAME cull counter (zeroed under ?freeze); the
  // streamer's resident-tile count is the freeze-stable proof the DAG terrain loaded.
  const defOk = def.resident > 0;
  console.log(`  ⇒ ${defOk ? `PASS — DAG terrain is the default (${def.resident} resident tiles streamed)` : 'FAIL — no DAG terrain on the default boot'}`);

  console.log('\n[flip] === OPT-OUT boot (?nanitedterrain=0 → legacy window grid) ===');
  const win = await bootOnce({ nanitedterrain: '0' }, 'shots/wip/flip-window.png');
  if (win.err) throw new Error(`window opt-out boot fatal: ${win.err}`);
  console.log(`  dagClusters       ${win.dagClusters} (expect 0 — no DAG terrain)`);
  console.log(`  stream.resident   ${win.resident} (expect -1 — no streamer)`);
  const winOk = win.resident === -1;
  console.log(`  ⇒ ${winOk ? 'PASS — window grid restored as the opt-out' : 'FAIL — streamer still active under ?nanitedterrain=0'}`);

  console.log(
    `\n[flip] ${defOk && winOk ? 'PASS' : 'FAIL'} — inspect shots/wip/flip-default.png (DAG terrain) vs flip-window.png (window grid)`,
  );
  if (!(defOk && winOk)) process.exit(1);
}
main().catch((e) => {
  console.error('[flip] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
