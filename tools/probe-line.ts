/**
 * Prints heightAtCpu / waterYAtCpu along a world-space line — artifact
 * triage for the water/terrain fields.
 *
 *   npx tsx tools/probe-line.ts --x0 11 --z0 1338 --x1 -200 --z1 1255 --n 40
 */

import { launchWebGPU, laasUrl } from './launch';

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const get = (k: string, d: string): string => {
    const i = args.indexOf(`--${k}`);
    return i >= 0 ? (args[i + 1] ?? d) : d;
  };
  const x0 = Number(get('x0', '0'));
  const z0 = Number(get('z0', '0'));
  const x1 = Number(get('x1', '100'));
  const z1 = Number(get('z1', '100'));
  const n = Number(get('n', '30'));

  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: 320, height: 200 } });
  await page.goto(laasUrl({ scene: 'world', width: 320, height: 200 }), {
    waitUntil: 'domcontentloaded',
  });
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error !== null),
    undefined,
    { timeout: 180000, polling: 250 },
  );
  const rows = await page.evaluate(
    ({ x0, z0, x1, z1, n }) => {
      (globalThis as unknown as { __name?: unknown }).__name ??= (t: unknown): unknown => t;
      const hf = (window as unknown as {
        __laasDbg?: { engine?: { heightfield?: unknown } };
      }).__laasDbg?.engine?.heightfield as {
        heightAtCpu(x: number, z: number): number;
        waterYAtCpu(x: number, z: number): number;
      };
      const out: string[] = [];
      for (let k = 0; k <= n; k++) {
        const t = k / n;
        const x = x0 + (x1 - x0) * t;
        const z = z0 + (z1 - z0) * t;
        const h = hf.heightAtCpu(x, z);
        const w = hf.waterYAtCpu(x, z);
        out.push(
          `${x.toFixed(0).padStart(6)} ${z.toFixed(0).padStart(6)}  h=${h.toFixed(2).padStart(8)}  w=${w.toFixed(2).padStart(8)}  ${w > h ? 'WET d=' + (w - h).toFixed(2) : ''}`,
        );
      }
      return out;
    },
    { x0, z0, x1, z1, n },
  );
  for (const r of rows) console.log(r);
  await browser.close();
}

void main().catch((e: unknown) => {
  console.error(e);
  process.exit(1);
});
