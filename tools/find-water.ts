/**
 * Scans the booted world's CPU hydrology mirrors for good water framings:
 * shallow wet cells (caustic band) with a dry bank a few meters away to
 * stand on. Prints candidate `--x --z --yaw` shot args, best first.
 *
 *   npx tsx tools/find-water.ts [--seed 1] [--top 12] [--minDepth 0.08]
 */

import { launchWebGPU, laasUrl } from './launch';

interface Cand {
  x: number;
  z: number;
  yaw: number;
  depth: number;
  wet: number; // local wet fraction (pool size proxy)
  level: number;
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const get = (k: string, d: string): string => {
    const i = args.indexOf(`--${k}`);
    return i >= 0 ? (args[i + 1] ?? d) : d;
  };
  const seed = Number(get('seed', '1'));
  const top = Number(get('top', '12'));
  const minDepth = Number(get('minDepth', '0.08'));
  const maxDepth = Number(get('maxDepth', '0.7'));

  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: 320, height: 200 } });
  page.on('pageerror', (err) => console.error('[pageerror]', err.message));
  await page.goto(laasUrl({ scene: 'world', seed, width: 320, height: 200 }), {
    waitUntil: 'domcontentloaded',
  });
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error !== null),
    undefined,
    { timeout: 180000, polling: 250 },
  );

  const cands = await page.evaluate(
    ({ minDepth, maxDepth }) => {
      // tsx/esbuild --keep-names wraps inner functions in __name(...) calls;
      // the helper doesn't exist inside the page — shim it first.
      (globalThis as unknown as { __name?: unknown }).__name ??= (t: unknown): unknown => t;
      const dbg = (window as unknown as {
        __laasDbg?: { engine?: { heightfield?: unknown } };
      }).__laasDbg;
      const hf = dbg?.engine?.heightfield as {
        heightAtCpu(x: number, z: number): number;
        waterYAtCpu(x: number, z: number): number;
      };
      if (!hf) return [] as Cand[];
      const out: Cand[] = [];
      const STEP = 6;
      const HALF = 2040;
      const depthAt = (x: number, z: number): number =>
        hf.waterYAtCpu(x, z) - hf.heightAtCpu(x, z);
      for (let z = -HALF; z <= HALF; z += STEP) {
        for (let x = -HALF; x <= HALF; x += STEP) {
          const d = depthAt(x, z);
          if (d < minDepth || d > maxDepth) continue;
          // local pool size: wet fraction in a 24 m disc
          let wet = 0;
          let n = 0;
          for (let dz = -24; dz <= 24; dz += 8) {
            for (let dx = -24; dx <= 24; dx += 8) {
              n++;
              if (depthAt(x + dx, z + dz) > 0.05) wet++;
            }
          }
          const wetFrac = wet / n;
          if (wetFrac < 0.3) continue;
          // bank: nearest dry+low spot 5–9 m away in 8 directions
          let best: { bx: number; bz: number } | null = null;
          for (let a = 0; a < 8; a++) {
            const ang = (a / 8) * Math.PI * 2;
            for (const r of [5, 7, 9]) {
              const bx = x + Math.cos(ang) * r;
              const bz = z + Math.sin(ang) * r;
              const bd = depthAt(bx, bz);
              const rise = hf.heightAtCpu(bx, bz) - hf.waterYAtCpu(x, z);
              if (bd < -0.2 && rise > 0.1 && rise < 2.5) {
                best = { bx, bz };
                break;
              }
            }
            if (best) break;
          }
          if (!best) continue;
          // yaw: camera at bank looking at the wet cell. forward=(−sin,−cos)
          const fx = x - best.bx;
          const fz = z - best.bz;
          const yaw = Math.atan2(-fx, -fz);
          out.push({
            x: Math.round(best.bx),
            z: Math.round(best.bz),
            yaw: Math.round(yaw * 100) / 100,
            depth: Math.round(d * 100) / 100,
            wet: Math.round(wetFrac * 100) / 100,
            level: Math.round(hf.waterYAtCpu(x, z)),
          });
        }
      }
      // dedup on a 60 m lattice, prefer bigger pools then mid depths
      out.sort((a, b) => b.wet - a.wet || Math.abs(a.depth - 0.3) - Math.abs(b.depth - 0.3));
      const seen = new Set<string>();
      const dedup: Cand[] = [];
      for (const c of out) {
        const k = `${Math.round(c.x / 60)},${Math.round(c.z / 60)}`;
        if (seen.has(k)) continue;
        seen.add(k);
        dedup.push(c);
      }
      return dedup;
      interface Cand {
        x: number;
        z: number;
        yaw: number;
        depth: number;
        wet: number;
        level: number;
      }
    },
    { minDepth, maxDepth },
  );

  for (const c of (cands as Cand[]).slice(0, top)) {
    console.log(
      `--x ${c.x} --z ${c.z} --yaw ${c.yaw}  # depth ${c.depth}m wet ${c.wet} level ${c.level}m`,
    );
  }
  await browser.close();
}

void main().catch((e: unknown) => {
  console.error(e);
  process.exit(1);
});
