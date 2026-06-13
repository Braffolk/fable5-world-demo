/**
 * N4-C4 — BARK + DEADWOOD shadow-receive + no-black-shadows gate.
 *
 * For each framing it boots three captures of the SAME pose (freeze=1):
 *   beauty   = ?nanite=1                 (full shading the user sees)
 *   cls      = ?nanite=1&nandbg=cls&postmin=1   (matClass mask: bark blue,
 *              deadwood cyan, terrain green, rock red — NaniteResolve cls view)
 *   noshadow = ?nanite=1&nanshadow=0     (CSM term removed from the resolve)
 *
 * The cls mask selects bark/deadwood pixels (eroded to INTERIOR only — cls and
 * beauty are separate boots with different TRAA jitter, so silhouette edge px
 * are unreliable). The "no black shadows" gate measures the SHADOWED-SUNLIT
 * subset: bark that is clearly sunlit when unshadowed (noshadow luma ≥ LIT_MIN)
 * AND meaningfully darkened by the CSM (noshadow − beauty ≥ SHADOW_DELTA). That
 * subset is exactly where the gate applies — real cast/self shadows on lit bark
 * — and it EXCLUDES bark that is dark in BOTH passes (distant/aerial deep
 * forest, correctly dark, NOT a shadow; measuring it would read the tonemap toe
 * through the canvas — the GOTCHA that PNG quantity probes are sRGB-poisoned).
 * A non-empty shadowed-sunlit subset also witnesses CSM RECEIVE (a different
 * producer — the nanshadow A/B — than the floor it checks).
 *
 * Runs with oldgeo=1: CSM casters (ShadowProxy + Forests per-cascade siblings)
 * are old-path until N5, so BLACK SLATE HAS AN EMPTY SHADOW MAP — bark there
 * receives nothing and nanshadow on/off differs only by cross-boot jitter (the
 * trap this probe first fell into). oldgeo restores the casters; the migrated
 * tree CAMERA draws stay hidden (suppressMigrated) so nanite bark still owns
 * its pixels and RECEIVES the casters' shadows (the N4 hybrid path). wind=0 +
 * lockexp=1 + framealign make beauty/noshadow comparable.
 *
 *   npx tsx tools/probe-barkshadow.ts          (dev server on :5173)
 *
 * Gates:
 *   shadowed-sunlit bark warm-albedo fraction ≥ CHROMA_MIN (no black shadows —
 *     a zero-ambient bug zeroes albedo→flat black; correct dim shadow keeps the
 *     warm tint; an absolute luma floor would fight the energy-correct dark
 *     forest + the bark's own cavity-AO crevices, so it is NOT used)
 *   ≥ SHADOWED_MIN shadowed-sunlit bark px at ≥1 framing (CSM receive is live)
 *   deadwood shares bark's isBD resolve branch (same CSM + ambient floor, only
 *     albedo differs and is dimmer) → covered by bark; gated where it has px.
 */

import { launchWebGPU, laasUrl } from './launch';
import sharp from 'sharp';

const W = 1280;
const H = 720;
/** ≥ this fraction of shadowed-sunlit bark must RETAIN warm albedo (not flat
 *  grey-black) — the no-black-shadows gate (zero-ambient bug → chroma→0). */
const CHROMA_MIN = 0.45;
const BARK_MIN_PX = 2000; // bark must cover this many ERODED px to be gated
const DEAD_MIN_PX = 300; // deadwood is sparse (thin ground logs) — lower bar
const SHADOW_DELTA = 8; // noshadow−beauty luma drop that counts as shadowed
const LIT_MIN = 55; // noshadow luma above which a surface is clearly sunlit —
//   the shadowed-sunlit subset (sunlit-when-unshadowed AND now shadowed) is
//   where "no black shadows" actually applies; dark-in-both distant forest
//   is excluded (it is correctly dark, not a black shadow).
const SHADOWED_MIN = 400; // a class needs this many shadowed-sunlit px to gate
//   (also the CSM-receive witness: a non-empty shadowed subset ⇒ receive live)
const ERODE = 2; // Chebyshev radius: drop px within ERODE of a class boundary —
//   kills silhouette/jitter false-black (cls & beauty are separate boots with
//   different TRAA jitter phase, so edge px land on the bg behind them). What
//   survives is INTERIOR surface, which is what no-black-shadows is about.

type Cls = 'bark' | 'dead' | 'other';

function classify(r: number, g: number, b: number): Cls {
  // ratio classification on the cls debug tint (robust to sRGB encoding):
  //   bark     (0.1,0.1,0.95) → B max, G low
  //   deadwood (0.1,0.7,0.8)  → B max-ish, G ALSO high, R low
  if (b > r * 1.3 && b > g * 1.3) return 'bark';
  if (b > r * 1.4 && g > r * 1.4) return 'dead';
  return 'other';
}

function luma(r: number, g: number, b: number): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

async function shoot(extra: Record<string, string>, tag: string): Promise<Buffer> {
  const { browser } = await launchWebGPU();
  try {
    const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    page.on('console', (m) => {
      if (m.type() === 'error') errors.push(m.text());
    });
    await page.goto(laasUrl({ scene: 'world', hud: false, extra }), { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => (window as unknown as { __laas?: { ready?: boolean; error?: unknown } }).__laas?.ready === true,
      undefined,
      { timeout: 240_000 },
    );
    await page.evaluate(
      async () =>
        (window as unknown as { __laas: { settle?: (n: number) => Promise<void> } }).__laas.settle?.(24),
    );
    // frame-align to a FIXED jitter phase (shoot.ts logic) so beauty/noshadow/
    // cls — three SEPARATE boots — share the TRAA jitter offset; without it the
    // beauty−noshadow diff is dominated by sub-pixel jitter on busy bark texture
    // (10.9% of px at bm7) not the CSM shadow term (the documented cross-boot
    // trap). With lockexp=1 + wind=0 + freeze the captures are then comparable.
    await page.evaluate(async () => {
      const s = (window as unknown as { __laas: { settle?: (n: number) => Promise<void>; stats?: { frame: number } } }).__laas;
      if (!s.settle || !s.stats) return;
      for (let guard = 0; guard < 1100; guard++) {
        if (s.stats.frame % 1024 === 0) break;
        await s.settle(1);
      }
    });
    const png = await page.screenshot({ type: 'png' });
    if (errors.length) throw new Error(`[${tag}] page errors: ${errors.slice(0, 3).join(' | ')}`);
    return png;
  } finally {
    await browser.close();
  }
}

async function raw(png: Buffer): Promise<Buffer> {
  const { data } = await sharp(png).raw().toBuffer({ resolveWithObject: true });
  return data;
}

interface ClassStat {
  /** total eroded (interior) class px */
  px: number;
  /** px in the SHADOWED-SUNLIT subset (would-be-sunlit, now CSM-shadowed) */
  shadowed: number;
  /** median beauty luma over the shadowed-sunlit subset (context) */
  sp50: number;
  /** fraction of the subset that RETAINS warm albedo (r−b ≥ 2, luma ≥ 3) —
   *  the no-black-shadows signal: a zero-ambient bug zeroes albedo→flat black
   *  (no chroma); correct dim shadow keeps albedo×ambient = a dark warm tint */
  chromaFrac: number;
  /** fraction pure-black (luma < 3): tonemap toe + sub-feature bark crevices */
  voidFrac: number;
}

/** classMap: 1=bark, 2=dead, 0=other; fps chip region forced to 0 */
function buildClassMap(cls: Buffer): Uint8Array {
  const m = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) {
    const x = i % W;
    const y = (i / W) | 0;
    if (y < 26 && x < 96) continue; // always-on fps DOM chip (top-left)
    const o = i * 3;
    const c = classify(cls[o] as number, cls[o + 1] as number, cls[o + 2] as number);
    m[i] = c === 'bark' ? 1 : c === 'dead' ? 2 : 0;
  }
  return m;
}

/** true iff every px within Chebyshev radius ERODE equals `v` (interior only) */
function eroded(m: Uint8Array, x: number, y: number, v: number): boolean {
  for (let dy = -ERODE; dy <= ERODE; dy++) {
    const yy = y + dy;
    if (yy < 0 || yy >= H) return false;
    for (let dx = -ERODE; dx <= ERODE; dx++) {
      const xx = x + dx;
      if (xx < 0 || xx >= W) return false;
      if (m[yy * W + xx] !== v) return false;
    }
  }
  return true;
}

function analyse(beauty: Buffer, map: Uint8Array, noshadow: Buffer, want: Cls): ClassStat {
  const wv = want === 'bark' ? 1 : 2;
  let total = 0;
  const sl: number[] = []; // shadowed-sunlit beauty lumas
  let chroma = 0;
  let voids = 0;
  for (let i = 0; i < W * H; i++) {
    if (map[i] !== wv) continue;
    const x = i % W;
    const y = (i / W) | 0;
    if (!eroded(map, x, y, wv)) continue; // interior only — no silhouette edges
    total++;
    const o = i * 3;
    const br = beauty[o] as number;
    const bg = beauty[o + 1] as number;
    const bb = beauty[o + 2] as number;
    const lb = luma(br, bg, bb);
    const ln = luma(noshadow[o] as number, noshadow[o + 1] as number, noshadow[o + 2] as number);
    // the shadowed-sunlit subset: clearly sunlit when unshadowed (noshadow ≥
    // LIT_MIN) AND meaningfully darkened by the CSM. That isolates real cast/
    // self shadows from surfaces dark in BOTH passes (distant deep forest —
    // correctly dark, not a shadow). oldgeo casters + framealign make the
    // beauty−noshadow diff the pure shadow term, not cross-boot jitter.
    if (ln < LIT_MIN || ln - lb < SHADOW_DELTA) continue;
    sl.push(lb);
    // no-black SIGNAL = warm-albedo retention. A zero-ambient bug zeroes
    // albedo → flat grey-black (no chroma); correct dim shadow keeps the
    // warm bark/deadwood albedo × ambient (a dark warm tint). Pure-black is
    // the tonemap toe + sub-feature bark crevices (cavity AO) — a tail.
    if (lb < 3) voids++;
    else if (br - bb >= 2) chroma++;
  }
  sl.sort((a, b) => a - b);
  const at = (q: number): number => sl[Math.min(sl.length - 1, Math.floor(q * sl.length))] ?? 0;
  return {
    px: total,
    shadowed: sl.length,
    sp50: Math.round(at(0.5)),
    chromaFrac: sl.length ? chroma / sl.length : 0,
    voidFrac: sl.length ? voids / sl.length : 0,
  };
}

interface Framing {
  tag: string;
  shot?: string;
}
const FRAMINGS: Framing[] = [
  { tag: 'bm7-forest-midday', shot: '7' }, // dense trunks: rich bark + canopy shadow
  { tag: 'bm4-meadow-lowsun', shot: '4' }, // low sun: long cast shadows
];
// CSM casters (ShadowProxy + Forests per-cascade siblings) are old-path until
// N5, so black slate has an EMPTY shadow map. Run with oldgeo=1 to restore them
// (the migrated tree CAMERA draws stay hidden via suppressMigrated, so nanite
// bark still owns those pixels and RECEIVES the old casters' shadows — exactly
// the N4 hybrid shadow path, as terrain's receive was proven in PROGRESS (s)).
const BASE_EXTRA: Record<string, string> = { nanite: '1', oldgeo: '1', wind: '0', lockexp: '1' };

async function main(): Promise<void> {
  let fail = false;
  // bark and deadwood share the resolve's isBD lighting branch (same CSM
  // receive + ambient floor; only albedo differs). So bark's receive proof
  // covers deadwood — deadwood-receive is informational, no-black is its real
  // (and more conservative, since its albedo is dimmer) gate.
  let barkReceived = false;
  for (const f of FRAMINGS) {
    const base: Record<string, string> = { ...BASE_EXTRA };
    if (f.shot) base['shot'] = f.shot;
    const beauty = await raw(await shoot({ ...base }, `${f.tag}/beauty`));
    const cls = await raw(await shoot({ ...base, nandbg: 'cls', postmin: '1' }, `${f.tag}/cls`));
    const noshadow = await raw(await shoot({ ...base, nanshadow: '0' }, `${f.tag}/noshadow`));
    const map = buildClassMap(cls);
    for (const c of ['bark', 'dead'] as const) {
      const minPx = c === 'bark' ? BARK_MIN_PX : DEAD_MIN_PX;
      const s = analyse(beauty, map, noshadow, c);
      if (s.px < minPx || s.shadowed < SHADOWED_MIN) {
        console.log(
          `[${f.tag}] ${c}: ${s.px} eroded px, ${s.shadowed} shadowed-sunlit ` +
            `(< gate ${minPx}px/${SHADOWED_MIN}sh — not gated here)`,
        );
        continue;
      }
      // gate ALBEDO RETENTION, not an absolute luma floor: deep forest shadow
      // is correctly dark (D-N22 energy-correct) and a luma floor fights both
      // the tonemap toe and the bark's own cavity-AO fissure detail. A zero-
      // ambient bug would zero the albedo → flat grey-black (chromaFrac→0);
      // correct dim shadow keeps the warm bark tint (chromaFrac high). voidFrac
      // (pure-black tail) is reported for context, not gated.
      const blackOk = s.chromaFrac >= CHROMA_MIN;
      if (c === 'bark') barkReceived = true; // non-empty shadowed subset ⇒ receive
      if (!blackOk) fail = true;
      const recv = c === 'dead' ? ' (informational — shares bark isBD path)' : '';
      console.log(
        `[${f.tag}] ${c}: ${s.px} eroded px, ${s.shadowed} shadowed-sunlit | ` +
          `p50 ${s.sp50} | warm-albedo ${(s.chromaFrac * 100).toFixed(0)}% void ${(s.voidFrac * 100).toFixed(0)}%${recv} ` +
          `→ ${blackOk ? 'no-black ✓' : 'BLACK CRUSH ✗'}`,
      );
    }
  }
  if (!barkReceived) {
    fail = true;
    console.log(`[gate] bark: CSM receive NOT detected (no framing reached ${SHADOWED_MIN} shadowed-sunlit px) ✗`);
  } else {
    console.log('[gate] bark: CSM receive live ✓ (covers deadwood — shared isBD resolve branch)');
  }
  console.log(fail ? '[probe-barkshadow] FAIL' : '[probe-barkshadow] PASS');
  if (fail) process.exitCode = 1;
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
