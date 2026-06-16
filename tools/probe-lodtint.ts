/**
 * LOD-tint terrain gradient check: ?nanitedbg=lod colours each cluster by its DAG
 * LOD level (word7 bits 10-15). Terrain clusters previously left those bits 0 (all
 * salmon = level 0); the fix packs dc.level in attachHeightDagTile + carries it
 * through the DagCache. Boots the terrain debug view, waits for the streamer to
 * settle (DAG_CACHE_VERSION bumped ⇒ a one-time re-bake), and shoots an elevated
 * across-terrain vista where near→far spans several LOD bands.
 *
 *   npx tsx tools/probe-lodtint.ts
 */
import type { CamPose } from '../src/core/Hooks';
import { laasUrl, launchWebGPU } from './launch';

const W = 1280;
const H = 720;

async function main(): Promise<void> {
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));
  const url = laasUrl({
    scene: 'terrain',
    width: W,
    height: H,
    freeze: false,
    // loderr=8 coarsens the cut so the terrain spans HIGH LOD levels (distinct
    // purple/blue/green hues) — proves the level is packed+read, not just subtle reds.
    extra: { nanite: '1', nanitedbg: 'lod', loderr: '8' },
  });
  console.log(`[lodtint] → ${url}`);
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__laas && (window.__laas.ready || window.__laas.error != null), undefined, {
    timeout: 300000,
    polling: 250,
  });
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);
  const base = (await page.evaluate(() => window.__laas.getPose?.() ?? null)) as CamPose | null;
  if (base) {
    // high vista looking shallowly toward the horizon → terrain recedes far, so the
    // tile-uniform cut spans many LOD bands (near level 0 → far coarse) = a clear gradient
    await page.evaluate(
      (p) => window.__laas.setPose?.({ yaw: p.yaw, pitch: -0.3, p: [p.p[0], p.p[1] + 160, p.p[2]] }),
      base,
    );
  }
  // settle for streaming (re-bake on the cache-version bump) — poll resident/built steady
  let key = '';
  let steady = 0;
  for (let i = 0; i < 80; i++) {
    await page.evaluate(async () => window.__laas.settle && (await window.__laas.settle(15)));
    const k = (await page.evaluate(() => {
      const c = (window.__laas.stats?.counters ?? {}) as Record<string, number>;
      return `${c['terrain.stream.resident'] ?? -1}/${c['terrain.stream.built'] ?? -1}`;
    })) as string;
    if (k === key) {
      if (++steady >= 4) break;
    } else {
      key = k;
      steady = 0;
    }
  }
  await page.screenshot({ path: 'shots/wip/lodtint-terrain.png' });
  console.log(`[lodtint] stream=${key} → shots/wip/lodtint-terrain.png`);
  await browser.close();
}
main().catch((e) => {
  console.error('[lodtint] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
