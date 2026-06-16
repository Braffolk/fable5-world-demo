/**
 * TERRAIN-RW ground truth — capture the CURRENT coarse-terrain artifacts the user
 * reported (holes, spurious upward walls, random slopes, hidden trees) so the
 * heightmap-native rework has a concrete before/after. Boots the real world scene
 * (?scene=world&nanite=1), flies to a sequence of vistas, and shoots each in lit
 * + per-cluster-LOD-tint so the coarse cluster shapes are visible.
 *
 *   npx tsx tools/probe-terrainshape.ts            # needs dev server on :5173
 *   TAG=before npx tsx tools/probe-terrainshape.ts
 */
import type { CamPose } from '../src/core/Hooks';
import { laasUrl, launchWebGPU } from './launch';

const W = 1280;
const H = 720;
const SEED = process.env.SEED ?? '1';

interface Shot {
  name: string;
  pose: (base: CamPose) => CamPose;
}

// poses that exercise the coarse-terrain LOD: elevated, looking ACROSS mountain
// slopes where the cut bites, plus a re-streaming fly. Regression shots for the
// heightmap-native regular-grid terrain (the QEM fans are gone).
const SHOTS: Shot[] = [
  { name: 'mtn-across', pose: (b) => ({ yaw: b.yaw, pitch: -0.18, p: [b.p[0], b.p[1] + 80, b.p[2]] }) },
  { name: 'vista-low', pose: (b) => ({ yaw: b.yaw, pitch: -0.34, p: [b.p[0], b.p[1] + 120, b.p[2]] }) },
  { name: 'vista-high', pose: (b) => ({ yaw: b.yaw, pitch: -0.5, p: [b.p[0], b.p[1] + 420, b.p[2]] }) },
  // FLY to a different region (XZ moves ~1400 m) — forces a full clipmap re-stream so
  // the settle-poll is genuinely exercised (the user's "shot before tiles ready?" case).
  { name: 'fly-region', pose: (b) => ({ yaw: b.yaw + 1.2, pitch: -0.3, p: [b.p[0] + 1400, b.p[1] + 180, b.p[2] - 900] }) },
];

async function shoot(): Promise<void> {
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));
  page.on('console', (m) => {
    const t = m.text();
    if (t.includes('terrain DAG')) console.log(`  ${t.replace(/^\[laas]\s*/, '')}`);
  });
  const extra: Record<string, string> = { nanite: '1', nanshadow: '0' };
  // ?nandbg=cluster — per-cluster hash tint (exposes the regular-grid cluster blocks).
  const nandbg = process.env.NANDBG ?? '';
  if (nandbg) extra.nandbg = nandbg;
  const tagSuffix = nandbg ? `-${nandbg}` : '';
  const url = laasUrl({ scene: 'world', seed: Number(SEED), width: W, height: H, freeze: false, extra });
  console.log(`[terrainshape] → ${url}`);
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__laas && (window.__laas.ready || window.__laas.error != null), undefined, {
    timeout: 300000,
    polling: 250,
  });
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);
  const base = (await page.evaluate(() => window.__laas.getPose?.() ?? null)) as CamPose | null;
  if (!base) throw new Error('getPose missing');
  console.log(`[terrainshape] spawn pose p=${base.p.map((n) => n.toFixed(0)).join(',')} yaw=${base.yaw.toFixed(2)}`);

  for (const s of SHOTS) {
    const p = s.pose(base);
    await page.evaluate((pose) => window.__laas.setPose?.(pose), p);
    // WAIT FOR THE STREAMER TO SETTLE before shooting — terrain tiles bake async on
    // workers, so a naive fixed settle can catch the coarse backstop mid-stream. Poll
    // the stream counters (resident/loaded/built) and only shoot once they hold steady
    // across consecutive checks (all desired tiles arrived, no bake in flight).
    let stableKey = '';
    let stableFor = 0;
    for (let it = 0; it < 60; it++) {
      await page.evaluate(async () => {
        if (window.__laas.settle) await window.__laas.settle(15);
      });
      const key = (await page.evaluate(() => {
        const c = (window.__laas.stats?.counters ?? {}) as Record<string, number>;
        return `${c['terrain.stream.resident'] ?? -1}/${c['terrain.stream.loaded'] ?? -1}/${c['terrain.stream.built'] ?? -1}`;
      })) as string;
      if (key === stableKey) {
        if (++stableFor >= 3) break; // steady for 3×15 frames ⇒ fully streamed
      } else {
        stableKey = key;
        stableFor = 0;
      }
    }
    const stats = (await page.evaluate(() => {
      const st = window.__laas.stats;
      const c = (st?.counters ?? {}) as Record<string, number>;
      return {
        resident: c['terrain.stream.resident'] ?? -1,
        skipped: c['terrain.stream.skipped'] ?? -1,
        visClusters: c['nanite.visClusters'] ?? -1,
        tris: st?.triangles ?? -1,
        fps: st?.fps ?? 0,
      };
    })) as Record<string, number>;
    const path = `shots/terrain/${s.name}${tagSuffix}.png`;
    await page.screenshot({ path });
    console.log(
      `  ${s.name.padEnd(11)} resident=${stats.resident} skipped=${stats.skipped} ` +
        `vis=${stats.visClusters} tris=${stats.tris} fps=${stats.fps.toFixed(0)} → ${path}`,
    );
  }
  await browser.close();
}

async function main(): Promise<void> {
  await shoot();
}

main().catch((e) => {
  console.error('[terrainshape] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
