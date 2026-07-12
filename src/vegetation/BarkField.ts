/**
 * BarkField — one CPU field per bark species that drives BOTH the real mesh
 * displacement (TubeMesh macro furrows/ridges) AND a compact baked albedo+normal
 * texture. Pure functions, no three.js imports (three-free ⇒ deterministic,
 * worker-safe, no GLSL/JS drift with a separate GPU bake).
 *
 * Frequency-band contract (no double-count):
 *   • MACRO (≥ ~6 cm: furrows, ridges, plates) → REAL vertex displacement.
 *     barkMacro() returns a signed radial offset + finite-diff derivatives;
 *     TubeMesh cuts it into the trunk tube and recomputes normals from the
 *     displaced surface. This is where the silhouette relief comes from.
 *   • MESO+MICRO (< ~6 cm: crack walls, flake edges, grain) → NORMAL MAP ONLY.
 *     Honest sub-triangle detail; it never fakes macro relief (no POM, no
 *     parallax — the macro band is real geometry).
 *   • ALBEDO tone + cavity AO → baked texture channels (aligned to the macro
 *     field at the same uv, so the dark furrow albedo sits IN the geometric
 *     furrow).
 *
 * Periodicity: the field is periodic in BOTH u (one tile) and v (one tile), so
 * a) the baked texture tiles seamlessly under RepeatWrapping, and b) TubeMesh's
 * closed-manifold u-seam (k=0 vs k=segsAround) gets bit-identical displacement
 * whenever uRepeats is an integer (enforced by the world-proportional UV remap).
 * The `twist` grain spiral is applied as u += twist·v (NOT v += twist·u), which
 * shears the AROUND coordinate as the trunk rises — a real spiral that keeps the
 * u-seam periodic (both seam samples receive the same twist·v offset).
 */

// ─────────────────────────── small math ────────────────────────────
const clamp01 = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x);
const smoothstep = (a: number, b: number, x: number): number => {
  if (a === b) return x < a ? 0 : 1;
  const t = clamp01((x - a) / (b - a));
  return t * t * (3 - 2 * t);
};
const mix = (a: number, b: number, t: number): number => a + (b - a) * t;
const wrap = (c: number, p: number): number => {
  const r = c % p;
  return r < 0 ? r + p : r;
};

/** deterministic hash of an integer lattice cell → [0,1). */
function hashCell(cx: number, cy: number, seed: number): number {
  let h = Math.imul((cx | 0) + 0x7ed55d16, 0x85ebca6b);
  h = Math.imul(h ^ ((cy | 0) + 0x165667b1), 0xc2b2ae35);
  h = Math.imul(h ^ (seed | 0), 0x27d4eb2f);
  h ^= h >>> 15;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  return (h >>> 0) / 4294967296;
}

/** value noise on a wrapped integer lattice → periodic at (perX, perY). */
function vnoise(x: number, y: number, perX: number, perY: number, seed: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const h = (ox: number, oy: number): number =>
    hashCell(wrap(ix + ox, perX), wrap(iy + oy, perY), seed);
  const a = h(0, 0);
  const b = h(1, 0);
  const c = h(0, 1);
  const d = h(1, 1);
  return mix(mix(a, b, ux), mix(c, d, ux), uy);
}

/** periodic fbm (period must be an integer count of the base lattice). */
function fbm2(x: number, y: number, per: number, seed: number, oct: number): number {
  let sum = 0;
  let amp = 0.5;
  let sc = 1;
  for (let i = 0; i < oct; i++) {
    sum += vnoise(x * sc, y * sc, per * sc, per * sc, seed + i * 37) * amp;
    amp *= 0.5;
    sc *= 2;
  }
  return sum;
}

/**
 * Periodic, anisotropic Worley. cellsU/cellsV cells per tile; the metric
 * d² = (Δu·aspect)² + (Δv)² so aspect > 1 elongates cells vertically ⇒ crack
 * boundaries chain VERTICALLY (vertical furrows); aspect < 1 ⇒ horizontal.
 * Returns F1 (nearest) and edge = F2−F1 (small near a cell boundary = a crack).
 */
function worley(
  u: number,
  v: number,
  cellsU: number,
  cellsV: number,
  aspect: number,
  seed: number,
): { f1: number; edge: number } {
  const qu = u * cellsU;
  const qv = v * cellsV;
  const cu = Math.floor(qu);
  const cv = Math.floor(qv);
  const fu = qu - cu;
  const fv = qv - cv;
  let f1 = 1e9;
  let f2 = 1e9;
  for (let oy = -1; oy <= 1; oy++) {
    for (let ox = -1; ox <= 1; ox++) {
      const wx = wrap(cu + ox, cellsU);
      const wy = wrap(cv + oy, cellsV);
      const hx = hashCell(wx, wy, seed);
      const hy = hashCell(wx, wy, seed + 911);
      const dx = (ox + hx - fu) * aspect;
      const dy = oy + hy - fv;
      const d = dx * dx + dy * dy;
      if (d < f1) {
        f2 = f1;
        f1 = d;
      } else if (d < f2) {
        f2 = d;
      }
    }
  }
  const f1s = Math.sqrt(f1);
  return { f1: f1s, edge: Math.sqrt(f2) - f1s };
}

// ─────────────────────────── params ────────────────────────────
export interface BarkFieldParams {
  /** species id, for logging */
  id: string;
  /** fixed per-layer seed (world-independent ⇒ same bark recipe in both worlds) */
  seed: number;
  /** world metres per texture/field tile (both axes at base radius) */
  tileW: number;
  /** fracture cells per tile, around / along */
  cellsU: number;
  cellsV: number;
  /** worley metric anisotropy (>1 = vertical furrows, <1 = horizontal bands) */
  aspect: number;
  /** domain-warp amplitude (breaks the lattice grid) */
  warp: number;
  /** F2−F1 width of the crack band (small = narrow deep cracks) */
  fissureW: number;
  /** metres — geometric furrow DEPTH at the reference radius */
  macroAmp: number;
  /** plate-interior doming */
  plateRound: number;
  /** ridge-interior fine detail (meso, normal-only) */
  ridge: number;
  /** micro grain amplitude (normal-only) */
  micro: number;
  /** 0/1 — birch lenticel dashes */
  lenticels: 0 | 1;
  /** grain spiral: u += twist·v (seam-safe). snag / karst weathered grain */
  twist: number;
  /** 0..1 — damp the fracture network toward smooth (beech / birch papery) */
  smooth: number;
  deep: [number, number, number];
  high: [number, number, number];
  mottle: number;
}

/** index == barkLayer (Species.barkLayer, Understory, deadwood share 2 & 5).
 *  deep/high/mottle carried over from the old BARK_TABLE — the palette was never
 *  the problem. tileW/cells/aspect/amp per the bark spec (§3), tuned for real
 *  furrow topology: spruce narrow vertical ridges, pine deep-crevice plate
 *  mosaic, beech near-smooth muscle, birch horizontal peel + lenticels, karst
 *  twisted deep ridges, snag long spiral splits. */
export const BARK_FIELDS: readonly BarkFieldParams[] = [
  { // 0 spruce — narrow vertical fissured ridges
    id: 'spruce', seed: 17, tileW: 1.4, cellsU: 10, cellsV: 2, aspect: 5.0,
    warp: 0.35, fissureW: 0.26, macroAmp: 0.044, plateRound: 0.25, ridge: 0.3,
    micro: 0.30, lenticels: 0, twist: 0.0, smooth: 0.0,
    deep: [0.045, 0.032, 0.026], high: [0.21, 0.155, 0.115], mottle: 0.25,
  },
  { // 1 pine — big flaky plates, deep crevices, orange
    id: 'pine', seed: 149, tileW: 1.8, cellsU: 5, cellsV: 3, aspect: 1.6,
    warp: 0.45, fissureW: 0.38, macroAmp: 0.055, plateRound: 0.42, ridge: 0.35,
    micro: 0.22, lenticels: 0, twist: 0.0, smooth: 0.0,
    deep: [0.05, 0.027, 0.016], high: [0.30, 0.155, 0.075], mottle: 0.35,
  },
  { // 2 beech — near-smooth pale grey "muscle" undulation, no fissure net
    id: 'beech', seed: 281, tileW: 2.0, cellsU: 3, cellsV: 3, aspect: 1.0,
    warp: 0.30, fissureW: 0.85, macroAmp: 0.006, plateRound: 0.1, ridge: 0.08,
    micro: 0.12, lenticels: 0, twist: 0.0, smooth: 0.85,
    deep: [0.16, 0.15, 0.135], high: [0.30, 0.285, 0.25], mottle: 0.5,
  },
  { // 3 birch — smooth white paper + horizontal peel bands + dark lenticels
    id: 'birch', seed: 397, tileW: 1.6, cellsU: 3, cellsV: 6, aspect: 0.25,
    warp: 0.25, fissureW: 0.55, macroAmp: 0.008, plateRound: 0.05, ridge: 0.06,
    micro: 0.10, lenticels: 1, twist: 0.0, smooth: 0.7,
    deep: [0.46, 0.44, 0.42], high: [0.80, 0.79, 0.76], mottle: 0.22,
  },
  { // 4 karst gnarl — twisted deep ridges
    id: 'karst', seed: 523, tileW: 1.2, cellsU: 8, cellsV: 2, aspect: 4.0,
    warp: 0.90, fissureW: 0.32, macroAmp: 0.072, plateRound: 0.3, ridge: 0.34,
    micro: 0.34, lenticels: 0, twist: 0.15, smooth: 0.0,
    deep: [0.05, 0.043, 0.036], high: [0.205, 0.18, 0.15], mottle: 0.3,
  },
  { // 5 snag — weathered silver-grey, long spiral splits
    id: 'snag', seed: 661, tileW: 1.5, cellsU: 9, cellsV: 1, aspect: 8.0,
    warp: 0.30, fissureW: 0.23, macroAmp: 0.052, plateRound: 0.15, ridge: 0.26,
    micro: 0.26, lenticels: 0, twist: 0.20, smooth: 0.0,
    deep: [0.07, 0.065, 0.06], high: [0.26, 0.25, 0.23], mottle: 0.2,
  },
];

/** baked texture resolution (per layer slice). 512²×6 rgba8 + mips ≈ 8.4 MB. */
export const BARK_TEX_RES = 512;

// ─────────────────────────── field ────────────────────────────

/** domain-warped tile coords (seam-safe: twist shears the AROUND axis by v). */
function warpedCoords(p: BarkFieldParams, u: number, v: number): [number, number] {
  const su = u + p.twist * v; // spiral grain — periodic in u (twist·v added to both seam samples)
  const WC = 3; // warp fbm cells per tile (integer ⇒ periodic)
  const wu = fbm2(su * WC, v * WC, WC, p.seed + 31, 2) - 0.5;
  const wv = fbm2(su * WC, v * WC, WC, p.seed + 67, 2) - 0.5;
  return [su + (wu * p.warp) / p.cellsU, v + (wv * p.warp) / p.cellsV];
}

/** macro height H ∈ [0,1] at (u,v): plates ≈ 0.5+ (nominal radius), cracks → ~0
 *  (cut in). The trunk's outer surface is the ridges; furrows recede by macroAmp. */
function macroH(p: BarkFieldParams, u: number, v: number): number {
  const [uu, vv] = warpedCoords(p, u, v);
  const low = vnoise(uu * 2, vv * 2, 2, 2, p.seed + 7) - 0.5; // buttress-scale undulation
  const w = worley(uu, vv, p.cellsU, p.cellsV, p.aspect, p.seed);
  const plate = smoothstep(0, p.fissureW, w.edge); // 1 on plate interiors, 0 in cracks
  const crack = 1 - plate;
  const bulge = (1 - clamp01(w.f1)) * p.plateRound; // gentle dome toward the plate centre
  const net = 1 - p.smooth; // fracture-network strength (0 for beech/birch-smooth)
  let H = 0.5 + low * 0.12;
  H -= crack * 0.5 * net; // furrows cut in
  H += plate * bulge * 0.25 * net; // ridges dome out slightly
  return clamp01(H);
}

/** meso+micro fine height at (u,v) — the NORMAL-MAP band only (never displaced). */
function mesoH(p: BarkFieldParams, u: number, v: number): number {
  const [uu, vv] = warpedCoords(p, u, v);
  const w = worley(uu, vv, p.cellsU, p.cellsV, p.aspect, p.seed);
  const plate = smoothstep(0, p.fissureW, w.edge);
  // ridge-interior detail: higher-freq fbm gated to plates (not cracks)
  const cu2 = Math.max(2, Math.round(p.cellsU * 2));
  const cv2 = Math.max(2, Math.round(p.cellsV * 2));
  const ridge = (fbm2(uu * cu2, vv * cv2, cu2, p.seed + 41, 2) - 0.5) * plate * p.ridge;
  // micro grain: fine high-freq, everywhere
  const grain = (fbm2(uu * 24, vv * 24, 24, p.seed + 91, 2) - 0.5) * p.micro;
  return ridge + grain;
}

export interface BarkMacro {
  /** signed radial offset factor ∈ ~[-1,+0.4]; TubeMesh scales it by macroAmp. */
  d: number;
  /** ∂d/∂u, ∂d/∂v (per tile-unit) — for the displaced-surface normal. */
  ddu: number;
  ddv: number;
}

/** macro displacement + derivatives (geometry). Periodic in u & v; for integer
 *  uRepeats the u-seam (u=0 vs u=uRepeats) is bit-identical. */
export function barkMacro(p: BarkFieldParams, u: number, v: number): BarkMacro {
  const eu = 1 / (p.cellsU * 4);
  const ev = 1 / (p.cellsV * 4);
  const c = (macroH(p, u, v) - 0.5) * 2;
  const du = ((macroH(p, u + eu, v) - 0.5) * 2 - (macroH(p, u - eu, v) - 0.5) * 2) / (2 * eu);
  const dv = ((macroH(p, u, v + ev) - 0.5) * 2 - (macroH(p, u, v - ev) - 0.5) * 2) / (2 * ev);
  return { d: c, ddu: du, ddv: dv };
}

export interface BarkTexel {
  /** tangent-space micro-grain normal (xy in [-1,1], z implied 1) */
  nx: number;
  ny: number;
  /** albedo mix factor: albedo = mix(deep, high, tone). Low in furrows. */
  tone: number;
  /** cavity AO (0.3 dark .. 1 open) */
  cavity: number;
}

/** one baked texel: micro-normal + albedo tone + cavity AO. */
export function barkSample(p: BarkFieldParams, u: number, v: number): BarkTexel {
  const H = macroH(p, u, v);
  const meso = mesoH(p, u, v);

  // micro/meso normal from the fine-height gradient (NOT the macro band)
  const eu = 1 / (p.cellsU * 16);
  const ev = 1 / (p.cellsV * 16);
  const dMu = (mesoH(p, u + eu, v) - mesoH(p, u - eu, v)) / (2 * eu);
  const dMv = (mesoH(p, u, v + ev) - mesoH(p, u, v - ev)) / (2 * ev);
  const K = 0.06; // grain normal strength (tuned so grain reads sub-triangle, not macro)
  let nx = clamp01(-dMu * K * 0.5 + 0.5) * 2 - 1;
  let ny = clamp01(-dMv * K * 0.5 + 0.5) * 2 - 1;

  // albedo tone: high on ridges (H high), dark in furrows; meso adds fine variation
  let tone = clamp01(H * 0.85 + meso * 0.6 + 0.05);

  // cavity AO: base darkening by depth + a small horizon term over the macro field
  const ao =
    Math.max(0, macroH(p, u + 2 * eu, v) - H) +
    Math.max(0, macroH(p, u - 2 * eu, v) - H) +
    Math.max(0, macroH(p, u, v + 2 * ev) - H) +
    Math.max(0, macroH(p, u, v - 2 * ev) - H);
  let cavity = clamp01((0.4 + H * 0.6) * (1 - ao * 0.5));

  // birch lenticels: dark horizontal dashes (physically motivated cavity + tone drop)
  if (p.lenticels) {
    const [uu, vv] = warpedCoords(p, u, v);
    const lw = worley(uu, vv, 5, 22, 0.35, p.seed + 77);
    const dash = 1 - smoothstep(0.16, 0.42, lw.f1);
    tone = mix(tone, 0.1, dash * 0.85);
    cavity = mix(cavity, 0.45, dash * 0.7);
    ny += dash * 0.25; // slight lip on the dash
  }

  return { nx: clamp01(nx * 0.5 + 0.5) * 2 - 1, ny: clamp01(ny * 0.5 + 0.5) * 2 - 1, tone, cavity };
}

/** flat level-0 byte size for a DataArrayTexture (width*height*depth*4). */
export function barkTexByteSize(res = BARK_TEX_RES): number {
  return res * res * BARK_FIELDS.length * 4;
}

/**
 * Fill rows [y0,y1) of one layer into a pre-allocated rgba8 array (the caller
 * chunks + yields so a cold boot never freezes). rgba8 layout:
 *   R,G = micro-grain normal.xy (0..255 ⇒ −1..1)   B = albedo tone   A = cavity AO
 * No POM height channel — the macro relief is real geometry.
 */
export function fillBarkRows(
  out: Uint8Array,
  layer: number,
  y0: number,
  y1: number,
  res = BARK_TEX_RES,
): void {
  const p = BARK_FIELDS[layer] as BarkFieldParams;
  const base = layer * res * res * 4;
  for (let y = y0; y < y1; y++) {
    const v = (y + 0.5) / res;
    for (let x = 0; x < res; x++) {
      const u = (x + 0.5) / res;
      const t = barkSample(p, u, v);
      const o = base + (y * res + x) * 4;
      out[o] = Math.round(clamp01(t.nx * 0.5 + 0.5) * 255);
      out[o + 1] = Math.round(clamp01(t.ny * 0.5 + 0.5) * 255);
      out[o + 2] = Math.round(clamp01(t.tone) * 255);
      out[o + 3] = Math.round(clamp01(t.cavity) * 255);
    }
  }
}
