/**
 * N8-D2 Stage 2b-3 GPU validation: the terrain CLIPMAP streamer FOLLOWS the
 * camera. Boots ?nanitedclip=1 (frame-1 terrain, no fallback), then hops the
 * camera across the field from an elevated vista, asserting at each pose:
 *   - terrain still RENDERS (nanite.dagClusters > 0 — no hole, the no-fallback floor)
 *   - the streamer is LIVE: resident bounded & > 0; loaded + evicted GROW as the
 *     camera roams (detail streams in / far detail drops out)
 *   - returning to spawn restores a comparable resident count (residency re-centers)
 * Screenshots each pose so holes (sky where ground should be) are eyeball-checkable.
 *
 *   npx tsx tools/probe-streammove.ts [gridN=128]   # needs the dev server on :5173
 */
import type { CamPose } from '../src/core/Hooks';
import { laasUrl, launchWebGPU } from './launch';

const W = 1280;
const H = 720;
const GRID_N = Math.max(2, Math.floor(Number(process.argv[2] ?? '128')));

interface Snap {
  dag: number;
  resident: number;
  loaded: number;
  evicted: number;
  skipped: number;
  built: number;
  ms: number;
}

let failures = 0;
const fail = (m: string): void => {
  failures++;
  console.error(`  FAIL ${m}`);
};
const expect = (c: boolean, m: string): void => {
  if (!c) fail(m);
};

async function main(): Promise<void> {
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  page.on('console', (m) => {
    const t = m.text();
    if (t.includes('terrain DAG') || t.includes('stream')) console.log(`  [boot] ${t.replace(/^\[laas]\s*/, '')}`);
  });
  const url = laasUrl({
    scene: 'world',
    width: W,
    height: H,
    freeze: true,
    extra: { nanite: '1', nanshadow: '0', occl: '1', nanitedterrain: String(GRID_N), nanitedclip: '1' },
  });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error != null),
    undefined,
    { timeout: 180000, polling: 250 },
  );
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);

  const readCounters = async (): Promise<Snap> =>
    page.evaluate(async () => {
      if (window.__laas.settle) await window.__laas.settle(8);
      const c = window.__laas.stats?.counters ?? {};
      return {
        dag: c['nanite.dagClusters'] ?? -1,
        resident: c['terrain.stream.resident'] ?? -1,
        loaded: c['terrain.stream.loaded'] ?? -1,
        evicted: c['terrain.stream.evicted'] ?? -1,
        skipped: c['terrain.stream.skipped'] ?? -1,
        built: c['terrain.stream.built'] ?? -1,
        ms: window.__laas.stats?.frameMs ?? -1,
      };
    });

  // settle to CONVERGENCE: keep rendering until the resident set stops changing
  // for `stableFor` consecutive reads (the async streamer has drained its load
  // queue for this pose), or a hard round cap. Measures STEADY STATE, not mid-load.
  const snap = async (cap = 60, stableFor = 3): Promise<Snap> => {
    let last = await readCounters();
    let stable = 0;
    for (let r = 0; r < cap; r++) {
      const now = await readCounters();
      stable = now.resident === last.resident && now.loaded === last.loaded ? stable + 1 : 0;
      last = now;
      if (stable >= stableFor) break;
    }
    return last;
  };

  // lift to an elevated vista so a hole (sky-through-ground) is visible, then keep
  // that yaw/pitch while we translate the footprint across the field.
  const base = (await page.evaluate(() => window.__laas.getPose?.() ?? null)) as CamPose | null;
  if (!base) throw new Error('getPose missing');
  const setXZ = async (x: number, z: number, y: number): Promise<void> => {
    await page.evaluate((p) => window.__laas.setPose?.({ yaw: p.yaw, pitch: -0.42, p: p.xyz }), {
      yaw: base.yaw,
      xyz: [x, y, z] as [number, number, number],
    });
  };

  const vistaY = base.p[1] + 240;
  // a CONTINUOUS-ish traverse in moderate hops (each well within the coarse ring's
  // field-spanning reach) so lazy eviction keeps the old LOD covering through the
  // bake window — mid-bake must stay hole-free. Plus a final hard TELEPORT to the
  // far rim (the 2b-4 stress) which need only CONVERGE hole-free.
  const X0 = base.p[0];
  const Z0 = base.p[2];
  const path: Array<[string, number, number, boolean]> = [
    ['spawn', X0, Z0, false],
    ['e-350', X0 + 350, Z0, true],
    ['e-700', X0 + 700, Z0, true],
    ['ne-1050', X0 + 700, Z0 + 350, true],
    ['ne-1400', X0 + 1050, Z0 + 700, true],
    ['back-spawn', X0, Z0, true],
    ['teleport-rim', -1700, -1600, false], // hard jump — convergence-only
  ];

  const snaps: Array<{ label: string; mid: Snap; s: Snap }> = [];
  for (const [label, x, z, midMatters] of path) {
    await setXZ(x, z, vistaY);
    const mid = await readCounters(); // ONE settle(8) — mid-bake (before convergence)
    await page.screenshot({ path: `shots/wip/streammove-${label}-mid.png` }); // mid-bake frame
    const s = await snap(); // settle to steady state
    snaps.push({ label, mid, s });
    await page.screenshot({ path: `shots/wip/streammove-${label}.png` });
    console.log(
      `  ${label.padEnd(13)} mid dag ${String(mid.dag).padStart(6)} (res ${String(mid.resident).padStart(3)}) → ` +
        `settled dag ${String(s.dag).padStart(6)} (res ${String(s.resident).padStart(3)}) | ` +
        `loaded ${s.loaded} evicted ${s.evicted} skip ${s.skipped}${midMatters ? '' : '  [convergence-only]'}`,
    );
  }
  await browser.close();

  // ---- assertions ----
  const spawn = snaps[0]!.s;
  expect(spawn.resident > 0, `boot resident ${spawn.resident} — streamer not live`);
  for (const { label, mid, s } of snaps) {
    // steady state: terrain renders everywhere (the no-fallback floor)
    expect(s.dag > 0, `${label}: settled dagClusters ${s.dag} — terrain HOLE`);
    expect(s.resident > 0, `${label}: resident ${s.resident} — terrain went empty`);
    // THE FIX: a moderate hop must stay hole-free DURING the bake (old LOD lingers)
    const midMatters = path.find((p) => p[0] === label)![3];
    if (midMatters) {
      expect(mid.dag > 0, `${label}: MID-BAKE dagClusters ${mid.dag} — old LOD evicted before replacement baked (the bug)`);
    }
  }
  const last = snaps[snaps.length - 1]!.s;
  expect(last.loaded > spawn.loaded, `streamer did not LOAD new tiles as the camera moved (${spawn.loaded}→${last.loaded})`);
  expect(last.evicted > 0, `streamer did not EVICT far tiles (${last.evicted})`);
  const back = snaps.find((x) => x.label === 'back-spawn')!.s;
  expect(Math.abs(back.resident - spawn.resident) <= 4, `back-spawn resident ${back.resident} far from boot ${spawn.resident} (re-center failed)`);

  const midMin = Math.min(...snaps.filter((x) => path.find((p) => p[0] === x.label)![3]).map((x) => x.mid.dag));
  console.log(
    `[streammove] gridN ${GRID_N}: ${path.length} poses; worst MID-BAKE dag over moderate hops = ${midMin} cl (must be >0 = no transient hole); ` +
      `total loaded ${last.loaded} / evicted ${last.evicted} / skipped ${last.skipped}`,
  );
  console.log('[streammove] inspect shots/wip/streammove-*.png — terrain fills the frame at every pose');
  if (failures > 0) {
    console.error(`[streammove] ${failures} FAILURES`);
    process.exit(1);
  }
  console.log('[streammove] PASS — detail follows the camera; old LOD survives the bake window (no transient hole on moderate motion)');
}
main().catch((e) => {
  console.error('[streammove] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
