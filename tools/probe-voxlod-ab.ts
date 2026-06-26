/**
 * voxlod A/B harness — ?voxlod=0 vs ?voxlod=1 on the SAME forced-voxel tree at NEAR/FAR,
 * plus a 200k deep-canopy pose. Reads nanite.voxBrickWrites / nanite.voxClusters from
 * window.__laas.stats.counters + grabs PNGs. Serves the worktree dev server on :5344.
 */
import { mkdirSync } from 'node:fs';
import { launchWebGPU, laasUrl } from './launch';

const BASE = 'http://localhost:5344/';
const SHOTDIR =
  '/Users/sebastian/IdeaProjects/fable-demo2/.claude/worktrees/nanite-voxdaglod/docs/perf-runs/shots/voxlod/iter4';

type Counters = Record<string, number>;
interface PoseSpec { name: string; pose: { p: [number, number, number]; yaw: number; pitch: number }; }

async function boot(page: import('playwright').Page, url: string): Promise<void> {
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error != null),
    undefined, { timeout: 300000, polling: 250 });
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error('fatal boot: ' + err);
}
async function readCounters(page: import('playwright').Page): Promise<Counters> {
  return page.evaluate(() => (window.__laas.stats?.counters ?? {}) as Record<string, number>);
}
async function settle(page: import('playwright').Page, n: number): Promise<void> {
  await page.evaluate(async (k) => { if (window.__laas.settle) await window.__laas.settle(k); }, n);
}
function medOf(samples: Counters[], key: string): number {
  const v = samples.map((s) => s[key] ?? -1).filter((x) => x >= 0).sort((a, b) => a - b);
  return v.length ? (v[Math.floor(v.length / 2)] ?? -1) : -1;
}
async function poseAndShoot(page: import('playwright').Page, spec: PoseSpec, tag: string) {
  await page.evaluate((pp) => window.__laas.setPose?.(pp), spec.pose);
  await settle(page, 30);
  const samples: Counters[] = [];
  for (let i = 0; i < 6; i++) { await settle(page, 4); samples.push(await readCounters(page)); }
  const clusters = medOf(samples, 'nanite.voxClusters');
  const writes = medOf(samples, 'nanite.voxBrickWrites');
  const vis = medOf(samples, 'nanite.visClusters');
  await page.screenshot({ path: SHOTDIR + '/' + tag + '.png' });
  console.log('   [' + tag + '] voxClusters=' + clusters + ' voxBrickWrites=' + writes + ' visClusters=' + vis);
  return { vis, clusters, writes };
}
async function dollySweep(page: import('playwright').Page, tree: [number, number, number], tag: string) {
  const dists = [120, 90, 70, 55, 42, 32, 24, 16, 11, 8];
  for (const d of dists) {
    const pose = { p: [tree[0], tree[1] + 4, tree[2] + d] as [number, number, number], yaw: 0, pitch: -0.06 };
    await page.evaluate((pp) => window.__laas.setPose?.(pp), pose);
    await settle(page, 16);
    const c = await readCounters(page);
    await page.screenshot({ path: SHOTDIR + '/' + tag + '-d' + d + '.png' });
    console.log('      dolly d=' + d + 'm  voxClusters=' + (c['nanite.voxClusters'] ?? -1) +
      ' voxBrickWrites=' + (c['nanite.voxBrickWrites'] ?? -1));
  }
}
function findTree(): [number, number, number] { return [0, 6, 0]; }

async function runForcedTree(page: import('playwright').Page, voxlod: 0 | 1, extra: Record<string, string>) {
  const url = laasUrl({ scene: 'forest', width: 1512, height: 982, freeze: false,
    extra: { trees: '1', nanite: '1', naniteleaf: '1', forcevox: 'all', voxdither: '0',
      voxlod: String(voxlod), dpr: '1.5', ...extra } }, BASE);
  console.log('\n=== forced-tree voxlod=' + voxlod + ' ' + JSON.stringify(extra) + ' ===\n   ' + url);
  const tBoot = Date.now();
  await boot(page, url);
  console.log('   booted in ' + ((Date.now() - tBoot) / 1000).toFixed(1) + 's');
  const dims = await page.evaluate(() => {
    const c = document.querySelector('canvas') as HTMLCanvasElement | null;
    return c ? { w: c.width, h: c.height } : null; });
  console.log('   canvas ' + dims?.w + 'x' + dims?.h);
  const tree = findTree();
  const tail = voxlod === 1 ? 'vl1' : 'vl0';
  const keysfx = extra.voxlodshell !== undefined ? '-shell' + extra.voxlodshell : '';
  const nearPose: PoseSpec = { name: 'near', pose: { p: [tree[0], tree[1] + 1, tree[2] + 9], yaw: 0, pitch: -0.08 } };
  const farPose: PoseSpec = { name: 'far', pose: { p: [tree[0], tree[1] + 8, tree[2] + 90], yaw: 0, pitch: -0.06 } };
  const near = await poseAndShoot(page, nearPose, 'tree-' + tail + keysfx + '-near');
  const far = await poseAndShoot(page, farPose, 'tree-' + tail + keysfx + '-far');
  if (voxlod === 1 && extra.voxlodshell === undefined) await dollySweep(page, tree, 'tree-' + tail);
  return { near: { clusters: near.clusters, writes: near.writes }, far: { clusters: far.clusters, writes: far.writes } };
}

async function runForest(page: import('playwright').Page, voxlod: 0 | 1) {
  const url = laasUrl({ scene: 'forest', width: 1512, height: 982, freeze: false,
    extra: { trees: '200000', nanite: '1', naniteleaf: '1', voxlod: String(voxlod), forcevox: 'all',
      voxdither: '0', dpr: '1.5', lodnear: '4', simband: '6', lodpow: '0.6', instminpx: '128' } }, BASE);
  console.log('\n=== 200k forest voxlod=' + voxlod + ' ===\n   ' + url);
  const tBoot = Date.now();
  await boot(page, url);
  console.log('   booted in ' + ((Date.now() - tBoot) / 1000).toFixed(1) + 's');
  const pose = { p: [0, 9, 60] as [number, number, number], yaw: 0.3, pitch: -0.05 };
  await page.evaluate((pp) => window.__laas.setPose?.(pp), pose);
  await settle(page, 40);
  const samples: Counters[] = [];
  let fps = 0;
  for (let i = 0; i < 8; i++) { await settle(page, 4); samples.push(await readCounters(page));
    fps = await page.evaluate(() => window.__laas.stats?.fps ?? 0); }
  const r = { clusters: medOf(samples, 'nanite.voxClusters'), writes: medOf(samples, 'nanite.voxBrickWrites'),
    vis: medOf(samples, 'nanite.visClusters'), fps };
  await page.screenshot({ path: SHOTDIR + '/forest-vl' + voxlod + '-deep.png' });
  console.log('   [forest vl' + voxlod + '] voxClusters=' + r.clusters + ' voxBrickWrites=' + r.writes +
    ' visClusters=' + r.vis + ' fps=' + fps.toFixed(1));
  return r;
}

async function main(): Promise<void> {
  mkdirSync(SHOTDIR, { recursive: true });
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: 1512, height: 982 }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));
  const badLines: string[] = [];
  page.on('console', (m) => {
    const t = m.text();
    if (t.includes('Invalid generated code') || t.includes('exceeds') || t.includes('Invalid RenderPipeline')) {
      badLines.push(t); console.log('   !! console: ' + t);
    }
    if (t.startsWith('[forest]') || t.startsWith('[worldreg]')) console.log('   · ' + t);
  });
  const results: Record<string, unknown> = {};
  results.vl0 = await runForcedTree(page, 0, {});
  results.vl1 = await runForcedTree(page, 1, {});
  results.vl1_shell0 = await runForcedTree(page, 1, { voxlodshell: '0' });
  results.forest_vl0 = await runForest(page, 0);
  results.forest_vl1 = await runForest(page, 1);
  console.log('\n================ SUMMARY ================');
  console.log(JSON.stringify(results, null, 2));
  console.log('\nbad console lines: ' + badLines.length);
  for (const l of badLines.slice(0, 20)) console.log('  ' + l);
  await browser.close();
}
main().catch((e) => { console.error(e); process.exit(1); });
