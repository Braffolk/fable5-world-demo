/**
 * R1 validation: nanite shadow-raster CADENCE under a MOVING camera.
 *
 * Boots bm7 with the nanite depth-only shadows (?nanshadow2=1), warms the cache,
 * then steps the camera forward one frame at a time (world FROZEN so wind/time do
 * not confound — pure camera motion) and reads `nanite.shRaster` each frame: the
 * bitmask of cascades RE-RASTERED that frame (bit c set ⇒ cascade c re-rastered;
 * 0 ⇒ served from cache).
 *
 * Expected (D-N28 / CsmCached PERIODS=[1,2,3,6]): cascade 0 re-rasters ~every
 * frame (it tracks the camera), cascades 1/2/3 only on their cadence. A static
 * camera (the --static flag) → all-zero after warmup. This is the "moving → only
 * changed cascades" half of the R1 gate (the static half is the shoot.ts perf row).
 *
 *   npx tsx tools/probe-shadowcadence.ts            # moving, bm7, 30 steps
 *   npx tsx tools/probe-shadowcadence.ts --static   # static control (expect all 0)
 *   npx tsx tools/probe-shadowcadence.ts --shot 3 --steps 24 --step 1.5
 */

import { launchWebGPU, laasUrl } from './launch';

interface Args {
  [k: string]: string | boolean;
}
function parseArgs(argv: string[]): Args {
  const out: Args = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i] as string;
    if (a.startsWith('--')) {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next !== undefined && !next.startsWith('--')) {
        out[key] = next;
        i++;
      } else out[key] = true;
    }
  }
  return out;
}
const str = (v: string | boolean | undefined): string | undefined =>
  typeof v === 'string' ? v : undefined;

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const shot = str(args['shot']) ?? '7';
  const steps = Number(str(args['steps']) ?? 30);
  const step = Number(str(args['step']) ?? 1.5); // metres/frame in x+z
  const isStatic = args['static'] === true;
  const width = 1280;
  const height = 720;

  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  page.on('console', (msg) => {
    const t = msg.text();
    if (t.startsWith('[laas]') || msg.type() === 'error') console.log(`[page:${msg.type()}] ${t}`);
  });
  page.on('pageerror', (err) => console.error('[pageerror]', err.message));

  const url = laasUrl({
    scene: 'world',
    width,
    height,
    freeze: true, // world frozen — isolate camera motion from wind/time
    extra: { nanite: '1', nanshadow2: '1', shot },
  });
  console.log(`[cadence] ${url} (${isStatic ? 'STATIC control' : 'MOVING'}, ${steps} steps)`);
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error !== null),
    undefined,
    { timeout: 180000, polling: 250 },
  );
  const err = await page.evaluate(() => window.__laas.error);
  if (err) throw new Error(`fatal: ${err}`);

  // warm the cache (all cascades raster at least once, then settle to steady state)
  await page.evaluate(async () => window.__laas.settle && (await window.__laas.settle(48)));

  const masks: number[] = [];
  for (let i = 0; i < steps; i++) {
    masks.push(
      await page.evaluate(
        async ({ dx, dz, isStatic }) => {
          const h = window.__laas;
          if (!isStatic && h.getPose && h.setPose) {
            const p = h.getPose();
            h.setPose({ ...p, p: [p.p[0] + dx, p.p[1], p.p[2] + dz] });
          }
          if (h.settle) await h.settle(1);
          return h.stats?.counters['nanite.shRaster'] ?? -1;
        },
        { dx: step, dz: step, isStatic },
      ),
    );
  }

  // report
  const bit = (m: number, c: number): number => (m >> c) & 1;
  const seq = masks.map((m) => (m < 0 ? '?' : m.toString(2).padStart(4, '0'))).join(' ');
  console.log(`[cadence] shRaster per frame (c3 c2 c1 c0):\n  ${seq}`);
  const n = masks.filter((m) => m >= 0).length || 1;
  const freq = [0, 1, 2, 3].map((c) => masks.filter((m) => m >= 0 && bit(m, c) === 1).length);
  console.log(`[cadence] per-cascade raster count / ${n} frames:`);
  for (let c = 0; c < 4; c++) {
    const period = [1, 2, 3, 6][c];
    console.log(
      `  c${c}: ${freq[c]}/${n}  (${((freq[c]! / n) * 100).toFixed(0)}%)  ` +
        `expect ~${isStatic ? 0 : Math.round((n / period!) * 10) / 10} (period ${period})`,
    );
  }
  const total = masks.reduce((s, m) => s + (m >= 0 ? popcount(m) : 0), 0);
  console.log(
    `[cadence] total cascade-rasters ${total} over ${n} frames = ${(total / n).toFixed(2)}/frame ` +
      `(R0 = 4.00/frame always; lower is the R1 win)`,
  );
  await browser.close();
}
function popcount(m: number): number {
  let c = 0;
  for (let i = 0; i < 4; i++) c += (m >> i) & 1;
  return c;
}
main().catch((e) => {
  console.error('[cadence] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
