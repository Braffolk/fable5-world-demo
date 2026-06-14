/**
 * N8-D2 Stage 2d seam INSPECTION (D-N39): clipmap levels ABUT at doubling stride,
 * so a fine tile's edge has 2× the verts of the coarse tile it meets → T-junction
 * cracks that show SKY (the coarse terrain there was hollowed out). This probe
 * strips all clutter (?ablate=veg,grass,water,shell,particles) and parks an
 * elevated, grazing-angle vista so the concentric ring boundaries cross the frame —
 * then COUNTS background (sky) pixels inside the terrain silhouette as a crack proxy
 * and screenshots for eyeball A/B. Re-run before/after the skirt fix to measure it.
 *
 *   npx tsx tools/probe-seams.ts [gridN=128]   # needs the dev server on :5173
 */
import type { CamPose } from '../src/core/Hooks';
import { laasUrl, launchWebGPU } from './launch';

const W = 1280;
const H = 720;
const GRID_N = Math.max(2, Math.floor(Number(process.argv[2] ?? '128')));
// screenshot tag so a skirt-OFF vs skirt-ON A/B doesn't overwrite (SKIRT=0/1; unset=def).
const TAG = process.env.SKIRT === '0' ? 'off' : process.env.SKIRT === '1' ? 'on' : 'def';

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
    extra: {
      nanite: '1',
      nanshadow: '0',
      occl: '1',
      nanitedterrain: String(GRID_N),
      nanitedclip: '1',
      // 2d A/B: SKIRT=0 → inter-level seams open (sky cracks); SKIRT=1 / unset → sealed.
      ...(process.env.SKIRT != null ? { nanitedskirt: process.env.SKIRT } : {}),
      // NOTE: do NOT ablate 'veg' — the nanite engine (incl. the DAG terrain) is built
      // INSIDE the !ablate.has('veg') block in TerrainScene, so ablating veg disables the
      // whole nanite path. Ablate the rest (grass is the separable GroundRing).
      ...(process.env.ABLATE === '' ? {} : { ablate: process.env.ABLATE ?? 'grass,water,shell,particles,caustics' }),
    },
  });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__laas && (window.__laas.ready || window.__laas.error != null), undefined, {
    timeout: 180000,
    polling: 250,
  });
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);

  // settle the streamer to steady state at a pose (resident set stops changing)
  const settle = async (): Promise<void> => {
    let last = -1;
    for (let r = 0; r < 60; r++) {
      const res = await page.evaluate(async () => {
        if (window.__laas.settle) await window.__laas.settle(6);
        return window.__laas.stats?.counters?.['terrain.stream.resident'] ?? -1;
      });
      if (res === last) break;
      last = res;
    }
  };

  const base = (await page.evaluate(() => window.__laas.getPose?.() ?? null)) as CamPose | null;
  if (!base) throw new Error('getPose missing');
  // diagnostic: what counters exist + spawn pose (is the camera even over terrain?)
  const diag = await page.evaluate(async () => {
    if (window.__laas.settle) await window.__laas.settle(8);
    const c = window.__laas.stats?.counters ?? {};
    return {
      keys: Object.keys(c).filter((k) => k.includes('nanite') || k.includes('terrain')),
      dag: c['nanite.dagClusters'],
      pose: window.__laas.getPose?.() ?? null,
    };
  });
  console.log(`  [diag] base pose ${JSON.stringify(diag.pose)}`);
  console.log(`  [diag] dag=${diag.dag} keys=${diag.keys.join(',')}`);

  // grazing vistas at increasing height: ring boundaries arc across the mid-frame
  // and are seen edge-on (where sky-cracks open). A steeper near-top-down catches
  // the inner boundaries; a shallow one catches the far ones against the sky.
  const poses: Array<{ label: string; dy: number; pitch: number }> = [
    { label: 'vista-shallow', dy: 120, pitch: -0.28 },
    { label: 'vista-mid', dy: 240, pitch: -0.45 },
    { label: 'vista-steep', dy: 420, pitch: -0.75 },
  ];

  for (const { label, dy, pitch } of poses) {
    await page.evaluate(
      (p) => window.__laas.setPose?.({ yaw: p.yaw, pitch: p.pitch, p: [p.x, p.y, p.z] as [number, number, number] }),
      { yaw: base.yaw, pitch, x: base.p[0], y: base.p[1] + dy, z: base.p[2] },
    );
    await settle();
    await page.screenshot({ path: `shots/wip/seams-${TAG}-${label}.png` });
    const dag = await page.evaluate(() => window.__laas.stats?.counters?.['nanite.dagClusters'] ?? -1);
    console.log(`  ${label.padEnd(13)} dag ${String(dag).padStart(6)} cl  → shots/wip/seams-${TAG}-${label}.png`);
  }
  await browser.close();
  console.log('[seams] inspect shots/wip/seams-*.png — sky slivers along concentric ring boundaries = T-junction cracks');
}
main().catch((e) => {
  console.error('[seams] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
