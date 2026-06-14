/**
 * N8-D2 Stage 2d skirt-DEPTH SUFFICIENCY proof (D-N39). The inter-level clipmap
 * crack at a transition between fine stride S and coarse stride 2S is the vertical
 * deviation the coarse edge introduces by OMITTING the fine midpoints:
 *     gap(p) = | h(p) − ½(h(p−S) + h(p+S)) |   at p = odd multiples of S.
 * The fine tile (clipmap level k, S = baseStride·2^k) hangs a skirt of depth
 * SKIRT_BASE_WORLD·2^k from its edge; it seals the crack iff that depth ≥ the gap.
 * So skirts provably seal EVERY inter-level crack on this field iff, for every
 * transition k, SKIRT_BASE_WORLD·2^k ≥ max-over-field gap at stride 2^k.
 *
 * This reads the REAL field via __laas.groundProbe (the walk-collision height
 * sampler — no GPU readback, no trees, no TAA) along dense full-resolution
 * scanlines in BOTH axes (edges run along x AND z), takes the worst gap per stride,
 * and reports the margin. GPU/jitter/vegetation-free ⇒ a clean PASS/FAIL on whether
 * SKIRT_BASE_WORLD is deep enough (bump it if any transition is RED).
 *
 *   npx tsx tools/probe-skirtgap.ts        # needs the dev server on :5173
 */
import { laasUrl, launchWebGPU } from './launch';
import { SKIRT_DEPTH_A, SKIRT_DEPTH_B } from '../src/nanite/BuildHeightDag';
import { WORLD_SIZE, HEIGHT_RES } from '../src/world/WorldConst';

// clipmap config mirrors TerrainStreamer: baseStride 1, M 4, levels sized so the
// coarsest ring spans the field. levels = ceil(log2(2·res/(M·gridN))) + 1.
const GRID_N = 128;
const M = 4;
const BASE_STRIDE = 1;
const LEVELS = Math.max(1, Math.ceil(Math.log2((2 * HEIGHT_RES) / (M * GRID_N))) + 1);

async function main(): Promise<void> {
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: 320, height: 200 }, deviceScaleFactor: 1 });
  // a plain world boot is enough — groundProbe samples hf.cpuHeights, identical to
  // what the clip streamer subsamples. No nanite needed (faster boot).
  const url = laasUrl({ scene: 'world', width: 320, height: 200, freeze: true });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__laas && (window.__laas.ready || window.__laas.error != null), undefined, {
    timeout: 180000,
    polling: 250,
  });
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);

  // tsx/esbuild keep-names wraps the inner const helpers in __name(); the browser
  // context has no __name. Inject a no-op shim as a STRING (string args bypass
  // esbuild) so the serialized evaluate body resolves it.
  await page.evaluate('globalThis.__name = globalThis.__name || function (f) { return f; };');

  const result = await page.evaluate(
    (cfg: { res: number; world: number; base: number; levels: number; nLines: number }) => {
      const gp = window.__laas.groundProbe;
      if (!gp) return { error: 'groundProbe missing' };
      const { res, world, base, levels, nLines } = cfg;
      const cell = world / res;
      const origin = cell / 2 - world / 2; // texel t → world = t*cell + origin
      const wx = (t: number): number => t * cell + origin;
      // sample a full-resolution height row/col into an array via groundProbe
      const rowAt = (fixed: number, axis: 'x' | 'z'): Float64Array => {
        const h = new Float64Array(res);
        for (let t = 0; t < res; t++) {
          const p = axis === 'x' ? gp(wx(t), wx(fixed)) : gp(wx(fixed), wx(t));
          h[t] = p.ground;
        }
        return h;
      };
      // worst |h(p) − ½(h(p−S)+h(p+S))| over a line, p at odd multiples of S
      const lineGap = (h: Float64Array, S: number): number => {
        let mx = 0;
        for (let p = S; p + S < h.length; p += S) {
          if (((p / S) & 1) === 0) continue; // only the midpoints the coarse omits
          const g = Math.abs((h[p] as number) - 0.5 * ((h[p - S] as number) + (h[p + S] as number)));
          if (g > mx) mx = g;
        }
        return mx;
      };
      // per transition k (fine stride base·2^k), worst gap over nLines scanlines each axis
      const maxGap = new Array<number>(levels - 1).fill(0);
      for (let li = 0; li < nLines; li++) {
        const fixed = Math.floor(((li + 0.5) / nLines) * res);
        const rx = rowAt(fixed, 'x');
        const rz = rowAt(fixed, 'z');
        for (let k = 0; k < levels - 1; k++) {
          const S = base * (1 << k);
          const g = Math.max(lineGap(rx, S), lineGap(rz, S));
          if (g > (maxGap[k] as number)) maxGap[k] = g;
        }
      }
      return { maxGap };
    },
    { res: HEIGHT_RES, world: WORLD_SIZE, base: BASE_STRIDE, levels: LEVELS, nLines: 256 },
  );

  if ('error' in result && result.error) throw new Error(result.error);
  const maxGap = (result as { maxGap: number[] }).maxGap;

  console.log(`[skirtgap] depth = ${SKIRT_DEPTH_A} + ${SKIRT_DEPTH_B}·level m, ${LEVELS} levels, gridN ${GRID_N}, 256 scanlines/axis`);
  console.log('  transition   fineStride   maxCrack(m)   skirtDepth(m)   margin   verdict');
  let worst = Infinity;
  for (let k = 0; k < maxGap.length; k++) {
    const S = BASE_STRIDE * (1 << k);
    const crack = maxGap[k] as number;
    const depth = SKIRT_DEPTH_A + SKIRT_DEPTH_B * k;
    const margin = depth - crack;
    if (margin < worst) worst = margin;
    const verdict = margin >= 0 ? 'SEAL' : 'CRACK';
    console.log(
      `  L${k}→L${k + 1}        ${String(S).padStart(6)}      ${crack.toFixed(2).padStart(10)}    ${depth.toFixed(1).padStart(10)}    ${margin >= 0 ? '+' : ''}${margin.toFixed(1).padStart(6)}   ${verdict}`,
    );
  }
  await browser.close();
  console.log(
    worst >= 0
      ? `[skirtgap] PASS — every transition sealed (worst margin +${worst.toFixed(1)} m)`
      : `[skirtgap] FAIL — raise SKIRT_DEPTH_A/_B by ≥${(-worst).toFixed(1)} m at the tightest transition`,
  );
}
main().catch((e) => {
  console.error('[skirtgap] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
