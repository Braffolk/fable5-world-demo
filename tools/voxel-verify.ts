/**
 * THROWAWAY Stage-1 verify driver (voxel-foliage). Boots a URL on :5193, settles,
 * reads window.__laas.stats.counters, screenshots. NOT a perf harness — no GPU timing.
 *
 *   npx tsx tools/voxel-verify.ts <task>
 *   task = voxdbg-spruce | voxdbg-pine | scar
 */
import { launchWebGPU } from './launch';

// agent threads reset cwd between calls; pin to the worktree root so launch.ts's
// relative .cache path resolves to the symlinked main-repo cache.
process.chdir('/Users/sebastian/IdeaProjects/fable-demo2/.claude/worktrees/nanite-voxel-foliage');

const BASE = 'http://localhost:5193/';
const SHOTS =
  '/Users/sebastian/IdeaProjects/fable-demo2/.claude/worktrees/nanite-voxel-foliage/docs/perf-runs/shots/stage1';

interface Job {
  name: string;
  url: string;
  width: number;
  height: number;
  dsf: number;
  settle: number;
  /** console-line prefixes to surface (e.g. [voxdbg]) */
  logPrefix?: string;
  counterKeys: string[];
}

function jobFor(task: string): Job {
  if (task === 'voxdbg-spruce' || task === 'voxdbg-pine') {
    const species = task === 'voxdbg-pine' ? 'pine' : 'spruce';
    return {
      name: task,
      url:
        `${BASE}?scene=voxdbg&voxspecies=${species}&freeze=1&hud=0`,
      width: 1512,
      height: 982,
      dsf: 1.5,
      settle: 24,
      logPrefix: '[voxdbg]',
      counterKeys: [
        'voxdbg.occBricks',
        'voxdbg.totalBricks',
        'voxdbg.bytes',
        'voxdbg.tris',
      ],
    };
  }
  if (task === 'scar' || task === 'scar-wide') {
    const wide = task === 'scar-wide';
    // canonical forest: scene=forest trees=200000 1512x982 dpr=1.5 freeze=1 settle=60
    // full post (default). scar=1 + the canonical LOD knobs from the cycle prompt.
    const q = new URLSearchParams({
      scene: 'forest',
      // nanite=1 + NO nanitedbg ⇒ the FULL world1 pipe (NaniteFrame), required for the
      // scar atomics which are gated on mode==='world1'. Without it ForestScene falls to
      // the lean cluster-debug NaniteView and the scar counters never fire.
      nanite: '1',
      trees: '200000',
      dpr: '1.5',
      freeze: '1',
      hud: '0',
      lodnear: '4',
      simband: '6',
      lodpow: '0.6',
      instminpx: '128',
      scar: '1',
    });
    if (wide) {
      // widen the band to the full voxelizable envelope: ~the spec transition (~40 m,
      // but probe nearer at 28 m to catch the cross-fade start) out to the instMinPx
      // ~110 px cull edge (~90-110 m). Tests how sensitive the 40%-of-frame gate is.
      q.set('scarnear', '28');
      q.set('scarfar', '110');
    }
    return {
      name: task,
      url: `${BASE}?${q.toString()}`,
      width: 1512,
      height: 982,
      dsf: 1.5,
      settle: 60,
      counterKeys: [
        'nanite.scarBandFrags',
        'nanite.scarBandPx',
        'nanite.scarTotalFrags',
        'nanite.scarBandClusters',
        'nanite.scarOverdrawX100',
        'nanite.scarBandShareX1000',
        'nanite.visClusters',
        'nanite.hwTris',
      ],
    };
  }
  throw new Error(`unknown task ${task}`);
}

async function main(): Promise<void> {
  const task = process.argv[2] ?? '';
  const job = jobFor(task);
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({
    viewport: { width: job.width, height: job.height },
    deviceScaleFactor: job.dsf,
  });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));
  page.on('console', (m) => {
    const t = m.text();
    if (job.logPrefix && t.includes(job.logPrefix)) console.log('   CONSOLE', t);
    if (t.toLowerCase().includes('error') || t.toLowerCase().includes('fail'))
      console.log('   CONSOLE', t);
  });

  console.log(`[verify] ${job.name} → ${job.url}`);
  await page.goto(job.url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error != null),
    undefined,
    { timeout: 300000, polling: 250 },
  );
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) {
    console.error(`[verify] FATAL boot: ${err}`);
    await browser.close();
    process.exit(1);
  }
  // settle (TAA / streaming / the 15-frame counter readback cadence)
  await page.evaluate(async (n) => window.__laas.settle?.(n), job.settle);
  // the scar/voxdbg counters surface on the frame%15 readback — settle more to be safe
  await page.evaluate(async () => window.__laas.settle?.(40));

  const canvas = await page.evaluate(() => {
    const c = document.querySelector('canvas');
    return c ? { w: c.width, h: c.height } : null;
  });
  console.log(`[verify] canvas backbuffer = ${canvas?.w}×${canvas?.h} (expect ${Math.round(job.width * job.dsf)}×${Math.round(job.height * job.dsf)})`);

  const counters = await page.evaluate(
    (keys) => {
      const cs = window.__laas.stats?.counters ?? {};
      const out: Record<string, number> = {};
      for (const k of keys) out[k] = cs[k] ?? -999999;
      // also dump every counter that matches scar/voxdbg for completeness
      for (const k of Object.keys(cs)) {
        if (k.includes('scar') || k.includes('voxdbg') || k.includes('band'))
          out[k] = cs[k] as number;
      }
      return out;
    },
    job.counterKeys,
  );
  console.log(`[verify] COUNTERS ${job.name}:`);
  for (const [k, v] of Object.entries(counters)) console.log(`   ${k} = ${v}`);

  const shotPath = `${SHOTS}/${job.name}.png`;
  await page.screenshot({ path: shotPath });
  console.log(`[verify] SHOT ${shotPath}`);

  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
