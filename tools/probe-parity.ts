/**
 * N3(b) silhouette parity gate (F12) — boots the SAME framing through the
 * nanite flat view (`?nanitedbg=flat`) and the hardware reference
 * (`?nanitedbg=hwref`), then pixel-diffs the screenshots.
 *
 * Gate per framing: diff ≤ 0.05% of pixels (per-channel tolerance absorbs
 * sub-LSB shading float differences; what remains is silhouette/coverage —
 * the raster correctness signal). A red-overlay diff PNG is written per
 * framing for the "no structural breaks" eyeball.
 *
 *   npx tsx tools/probe-parity.ts [framings...]   (default: spawn 1 3 4 7)
 *   (dev server on :5173; outputs → /tmp/parity-*.png)
 */

import { launchWebGPU, laasUrl } from './launch';
import sharp from 'sharp';

const W = 1280;
const H = 720;
const TOL = 3; // per-channel 8-bit tolerance (shading float noise)
/** SILHOUETTE gate (F12's literal target): pixels where exactly one side
 *  shows background = coverage/silhouette disagreement */
const GATE_COVER_PCT = 0.05;
/** backstop on interior class flips (both sides lit, different matClass):
 *  depth-ownership at near-coplanar surface INTERSECTIONS (stones/trunks
 *  sunk into terrain) — cross-rasterizer tie-breaks, not raster defects;
 *  bounded so a real regression still fails */
const GATE_FLIP_PCT = 0.2;
const BG = 24; // r+g+b below this = background (#06080a page bg)

async function shoot(mode: 'flat' | 'hwref', framing: string): Promise<string> {
  const { browser } = await launchWebGPU();
  const path = `/tmp/parity-${framing}-${mode}.png`;
  try {
    const page = await browser.newPage();
    await page.setViewportSize({ width: W, height: H });
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    // shade=0: pure matClass color both sides — the diff measures coverage/
    // structure only (lambert differs by shading MODEL at sub-pixel tris:
    // the resolve fetches the pixel's exact triangle, HW derivative normals
    // average across quads — not a raster defect; humans judge lambert shots)
    const extra: Record<string, string> = { nanite: '1', nanitedbg: mode, shade: '0' };
    if (framing !== 'spawn') extra['shot'] = framing;
    await page.goto(laasUrl({ scene: 'world', hud: false, extra }), {
      waitUntil: 'domcontentloaded',
    });
    await page.waitForFunction(
      () => (window as unknown as { __laas?: { ready?: boolean } }).__laas?.ready === true,
      undefined,
      { timeout: 240_000 },
    );
    await new Promise((r) => setTimeout(r, 2500)); // settle (two-phase + hwref pose freeze)
    await page.evaluate(() => {
      // the always-on fps chip + boot overlay are DOM — chip text varies
      // between runs; the overlay's fade can linger into early shots
      for (const id of ['hud-fps', 'boot']) {
        const el = document.getElementById(id);
        if (el) el.style.display = 'none';
      }
    });
    await page.screenshot({ path });
    if (errors.length) throw new Error(`page errors in ${mode}/${framing}: ${errors[0]}`);
  } finally {
    await browser.close();
  }
  return path;
}

async function diff(framing: string, a: string, b: string): Promise<{ cover: number; flips: number }> {
  const [ia, ib] = await Promise.all([
    sharp(a).raw().toBuffer({ resolveWithObject: true }),
    sharp(b).raw().toBuffer({ resolveWithObject: true }),
  ]);
  if (ia.info.width !== ib.info.width || ia.info.height !== ib.info.height) {
    throw new Error('parity shots differ in size');
  }
  const ca = ia.info.channels;
  const cb = ib.info.channels;
  const n = ia.info.width * ia.info.height;
  const overlay = Buffer.alloc(n * 3);
  let cover = 0;
  let flips = 0;
  for (let p = 0; p < n; p++) {
    const da = Math.abs((ia.data[p * ca] ?? 0) - (ib.data[p * cb] ?? 0));
    const dg = Math.abs((ia.data[p * ca + 1] ?? 0) - (ib.data[p * cb + 1] ?? 0));
    const db = Math.abs((ia.data[p * ca + 2] ?? 0) - (ib.data[p * cb + 2] ?? 0));
    const d = Math.max(da, dg, db);
    if (d > TOL) {
      const la = (ia.data[p * ca] ?? 0) + (ia.data[p * ca + 1] ?? 0) + (ia.data[p * ca + 2] ?? 0);
      const lb = (ib.data[p * cb] ?? 0) + (ib.data[p * cb + 1] ?? 0) + (ib.data[p * cb + 2] ?? 0);
      const isCover = la <= BG !== lb <= BG;
      if (isCover) {
        cover++;
        overlay[p * 3] = 255; // red = silhouette/coverage disagreement
      } else {
        flips++;
        overlay[p * 3] = 255; // yellow = interior class flip (depth tie)
        overlay[p * 3 + 1] = 220;
      }
    } else {
      const g = (ia.data[p * ca] ?? 0) >> 1;
      overlay[p * 3] = g;
      overlay[p * 3 + 1] = g;
      overlay[p * 3 + 2] = g;
    }
  }
  await sharp(overlay, { raw: { width: ia.info.width, height: ia.info.height, channels: 3 } })
    .png()
    .toFile(`/tmp/parity-${framing}-diff.png`);
  return { cover, flips };
}

async function main(): Promise<void> {
  const framings = process.argv.slice(2).length ? process.argv.slice(2) : ['spawn', '1', '3', '4', '7'];
  let fail = false;
  for (const f of framings) {
    const a = await shoot('flat', f);
    const b = await shoot('hwref', f);
    const { cover, flips } = await diff(f, a, b);
    const coverPct = (cover / (W * H)) * 100;
    const flipPct = (flips / (W * H)) * 100;
    const ok = coverPct <= GATE_COVER_PCT && flipPct <= GATE_FLIP_PCT;
    if (!ok) fail = true;
    console.log(
      `[parity] ${f}: silhouette ${cover} px (${coverPct.toFixed(4)}%), intersection flips ${flips} px (${flipPct.toFixed(4)}%) ${ok ? 'OK' : 'OVER GATE'} — /tmp/parity-${f}-diff.png`,
    );
  }
  console.log(
    `[probe-parity] ${fail ? 'FAIL' : 'PASS'} (silhouette ≤${GATE_COVER_PCT}%, flips ≤${GATE_FLIP_PCT}%, tol ${TOL}/255)`,
  );
  if (fail) process.exit(1);
}

void main();
