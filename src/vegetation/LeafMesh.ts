/**
 * Foliage geometry — REAL meshes, no alpha cards at LOD0 (cards come from
 * captures of these meshes at LOD1+). A leaf is a folded/curled strip with a
 * parametric outline; a needle spray is a stem with dozens of single-quad
 * needles in comb or brush arrangement. Everything appends into a MeshGrower
 * in the anchor's local frame (+z outward, +y up) via a supplied transform.
 *
 * vdata: x hue, y sway flex, z sway phase, w AO (crown depth applied later).
 */

import { Matrix4, Quaternion, Vector3 } from 'three';
import type { Rng } from '../core/Seed';
import type { MeshGrower } from './TubeMesh';
import type { LeafAnchor, LeafShapeParams } from './VegTypes';

const _p = new Vector3();
const _n = new Vector3();
const _m = new Matrix4();
const _q = new Quaternion();

function pushXf(
  g: MeshGrower,
  m: Matrix4,
  px: number, py: number, pz: number,
  nx: number, ny: number, nz: number,
  u: number, v: number,
  d0: number, d1: number, d2: number, d3: number,
): number {
  _p.set(px, py, pz).applyMatrix4(m);
  _n.set(nx, ny, nz).transformDirection(m);
  return g.vertex(_p.x, _p.y, _p.z, _n.x, _n.y, _n.z, u, v, d0, d1, d2, d3);
}

/** Per-survivor crown-LOD shaping for a broadleaf leaf: area-preserving growth
 *  (§3.3) + connected-strip simplification (§5, "used in conjunction with a
 *  surface-filtering method"). Both are no-ops at the defaults, so rung 0 stays
 *  byte-identical to LOD0. */
export interface LeafLod {
  /** area-preserving survivor growth: leaf WIDTH ×(1/λ). Cook Eq 3 gives area
   *  scale s = 1/λ; Eq 7's binomial shows s ≈ 1/λ exactly once λ ≫ r (r = one
   *  leaf's area / crown visible area ≈ 5e-4, λ ≥ 0.7 here ⇒ λ/r ≳ 1400), so we
   *  use full-strength s = 1/λ and skip the Eq-6 visible-area estimate (needless
   *  machinery in this regime). Realized by scaling ONE dimension — the WIDTH —
   *  with length fixed, exactly the paper's Fig 2c/d ("the leaf widths are
   *  scaled"). widthMul = 1/λ ⇒ quad area ×(1/λ). 1 = native. */
  widthMul?: number;
  /** connected-shape simplification: blade rows along the length (LOD0 = 4).
   *  Fewer rows = coarser fold/curl resolution at the SAME len/width envelope, so
   *  silhouette coverage holds and the tri cut comes from element DETAIL, not from
   *  removing whole leaves. It is a single connected strip (no soup, no weld).
   *  Clamped ≥ 2 so the mid-leaf width peak (sin(πs) at s≈0.5) is always sampled —
   *  a 1-row leaf would sample only the narrow ends and thin the silhouette. */
  rows?: number;
}

/**
 * One leaf: N-row strip along +z, 3 verts per row (−w, mid, +w), folded along
 * the midrib and curled toward the tip. ~18 tris at ROWS=4. Local: base at
 * origin, blade along +z, face up +y.
 */
export function buildLeaf(
  g: MeshGrower,
  m: Matrix4,
  shape: LeafShapeParams,
  hue: number,
  flex: number,
  phase: number,
  ao: number,
  lod?: LeafLod,
): void {
  const ROWS = Math.max(2, Math.round(lod?.rows ?? 4));
  const L = shape.len;
  const W = shape.width * (lod?.widthMul ?? 1);
  const rows: number[][] = [];
  // tiny petiole
  const stem = L * 0.14;
  for (let i = 0; i <= ROWS; i++) {
    const s = i / ROWS;
    const w = W * Math.pow(Math.sin(Math.PI * Math.min(1, s * 0.86 + 0.07)), shape.shapePow);
    const z = stem + s * (L - stem);
    const curlY = -shape.curl * s * s * L;
    const foldY = shape.fold * w;
    // normal tilts with fold; rough but cheap (verts re-lit by bendNormals)
    const r: number[] = [];
    r.push(pushXf(g, m, -w, curlY + foldY * 0.0 - foldY, z, -shape.fold * 0.8, 1, 0, 0, s, hue, flex, phase, ao * 0.92));
    r.push(pushXf(g, m, 0, curlY + foldY * 0.35, z, 0, 1, shape.curl * s, 0.5, s, hue, flex, phase, ao));
    r.push(pushXf(g, m, w, curlY - foldY, z, shape.fold * 0.8, 1, 0, 1, s, hue, flex, phase, ao * 0.92));
    rows.push(r);
  }
  for (let i = 0; i < ROWS; i++) {
    const a = rows[i] as number[];
    const b = rows[i + 1] as number[];
    g.quad(a[0] as number, b[0] as number, b[1] as number, a[1] as number);
    g.quad(a[1] as number, b[1] as number, b[2] as number, a[2] as number);
  }
  // petiole quad
  const p0 = pushXf(g, m, -W * 0.06, 0, 0, 0, 1, 0, 0.45, 0, hue, flex * 0.7, phase, ao);
  const p1 = pushXf(g, m, W * 0.06, 0, 0, 0, 1, 0, 0.55, 0, hue, flex * 0.7, phase, ao);
  const r0 = rows[0] as number[];
  g.quad(p0, r0[0] as number, r0[1] as number, p1);
  g.tri(p1, r0[1] as number, r0[2] as number);
}

/** Per-survivor crown-LOD shaping for a conifer needle spray. Every field is a
 *  no-op at its default, so rung 0 (widthMul 1, no needleKeep, stemSegs 4) is
 *  byte-identical to LOD0. The spray transform and needle LENGTH are NEVER
 *  scaled here (elongation = the historical spruce-spike); growth is WIDTH-only. */
export interface SprayLod {
  /** ELEMENT-level (whole-spray) area-preserving needle-WIDTH growth ×(1/λ). Same
   *  Cook Eq 3 / Eq 7 reasoning as LeafLod.widthMul: a needle quad's area ∝ width,
   *  so width ×(1/λ) = area ×(1/λ); λ ≫ r ⇒ s = 1/λ exactly. 1 = native. */
  widthMul?: number;
  /** INTRA-element (within-spray) needle pruning: keep fraction μ of the needles
   *  in this spray, de-correlated + nested (low-hash prefix of hash(needleIdx ⊕
   *  seed)). Survivors are area-compensated INSIDE the spray by width ×(count/kept)
   *  (so total needle area is preserved and the spray never sparsens macroscopically
   *  — spray placement/count is untouched). This is an intra-element tri lever that
   *  keeps the ELEMENT-pruning λ gentle. `seed` is per-spray (stable across rungs ⇒
   *  nested keep-sets). Absent / μ≥1 ⇒ all needles kept. */
  needleKeep?: { mu: number; seed: number };
  /** stem strip RENDER segments (LOD0 = 4). The sag CURVE is always integrated at
   *  full resolution and needle bases sample it, so reducing this straightens the
   *  thin twig strip WITHOUT moving any needle (no pop). Clamped to [2,4]. */
  stemSegs?: number;
}

/** De-correlated + nested needle keep-mask for intra-spray pruning (Cook §3.2:
 *  a random priority order, keep the low-key prefix). Keying on needle INDEX (not
 *  position) is fine — the comb/brush layout already de-correlates index from
 *  space, and the per-spray `seed` varies which subset each spray drops. Lower μ
 *  keeps a strict subset of higher μ (same sort, shorter prefix) ⇒ nested rungs. */
function needleKeepMask(count: number, mu: number, seed: number): boolean[] {
  const keep = new Array<boolean>(count).fill(false);
  if (count === 0) return keep;
  const k = Math.max(0, Math.min(count, Math.round(Math.max(0, Math.min(1, mu)) * count)));
  if (k >= count) return keep.fill(true);
  if (k === 0) return keep;
  const hashOf = (i: number): number =>
    (Math.imul(i + 1, 2246822519) ^ Math.imul(seed | 0, 3266489917)) >>> 0;
  const order = Array.from({ length: count }, (_, i) => i);
  order.sort((a, b) => {
    const ha = hashOf(a);
    const hb = hashOf(b);
    return ha !== hb ? ha - hb : a - b;
  });
  for (let i = 0; i < k; i++) keep[order[i] as number] = true;
  return keep;
}

/**
 * Needle spray: drooping stem polyline + `needleCount` single-quad needles,
 * comb (flat, ±row) or brush (radial) arrangement. Local: along +z.
 */
export function buildNeedleSpray(
  g: MeshGrower,
  m: Matrix4,
  shape: LeafShapeParams,
  scale: number,
  rng: Rng,
  hue: number,
  flex: number,
  phase: number,
  ao: number,
  lod?: SprayLod,
): void {
  // sag curve is ALWAYS integrated at full resolution (needle bases sample it);
  // the rendered strip may use fewer of these points (stemSegs) — see below.
  const CURVE_SEGS = 4;
  const L = scale;
  // stem: thin two-sided strip (cheaper than a tube, reads as twig)
  const stemPts: Vector3[] = [];
  let dz = 1;
  let dy = 0;
  let z = 0;
  let y = 0;
  for (let i = 0; i <= CURVE_SEGS; i++) {
    stemPts.push(new Vector3(0, y, z));
    const step = L / CURVE_SEGS;
    dy -= 0.16 * (i / CURVE_SEGS); // sag
    const dl = Math.hypot(dy, dz);
    z += (dz / dl) * step;
    y += (dy / dl) * step;
  }
  const sw = L * 0.012 + 0.002;
  // render the strip over a subsampled point set (endpoints + taper kept via the
  // ORIGINAL index j, so stemSegs=4 is byte-identical to LOD0 and coarser levels
  // just straighten the twig; needle bases below still use the full stemPts).
  const renderSegs = Math.max(2, Math.min(CURVE_SEGS, Math.round(lod?.stemSegs ?? CURVE_SEGS)));
  const stemRows: number[][] = [];
  for (let s = 0; s <= renderSegs; s++) {
    const j = Math.round((s * CURVE_SEGS) / renderSegs);
    const p = stemPts[j] as Vector3;
    const w = sw * (1 - (j / CURVE_SEGS) * 0.7);
    stemRows.push([
      pushXf(g, m, p.x - w, p.y, p.z, 0, 1, 0, 0.48, j / CURVE_SEGS, hue, flex, phase, ao * 0.85),
      pushXf(g, m, p.x + w, p.y, p.z, 0, 1, 0, 0.52, j / CURVE_SEGS, hue, flex, phase, ao * 0.85),
    ]);
  }
  for (let i = 0; i < stemRows.length - 1; i++) {
    const a = stemRows[i] as number[];
    const b = stemRows[i + 1] as number[];
    g.quad(a[0] as number, b[0] as number, b[1] as number, a[1] as number);
  }

  // needles
  const count = shape.needleCount;
  const nl = shape.len;
  const keep = lod?.needleKeep && lod.needleKeep.mu < 1
    ? needleKeepMask(count, lod.needleKeep.mu, lod.needleKeep.seed)
    : null;
  let kept = count;
  if (keep) {
    kept = 0;
    for (let i = 0; i < count; i++) if (keep[i]) kept++;
  }
  // width growth = (element 1/λ) × (intra-spray count/kept). The intra factor uses
  // the ACTUAL kept count (not 1/μ) so within-spray area is preserved EXACTLY.
  const nw = shape.width * (lod?.widthMul ?? 1) * (count / Math.max(1, kept));
  for (let i = 0; i < count; i++) {
    const s = (i + 0.5) / count;
    // comb: two layered rows ±x (fills the bough plane); brush: radial. ALL rng
    // draws happen for EVERY needle (survivor or pruned) so a survivor's placement
    // is bit-identical regardless of μ — only the quad emit is skipped when pruned.
    const side = i % 2 === 0 ? 1 : -1;
    const layer = i % 4 < 2 ? 1 : 0;
    const az = shape.brush > 0.5
      ? rng.float() * Math.PI * 2
      : side * (1.05 + (rng.float() - 0.5) * 0.85);
    const elev = shape.brush > 0.5
      ? (rng.float() - 0.2) * 1.1
      : (layer === 1 ? 0.42 : 0.02) + (rng.float() - 0.5) * 0.3;
    const swing = (rng.float() - 0.5) * 0.3 + s * 0.55; // sweep toward tip
    const lenRand = rng.float();
    const hueRand = rng.float();
    if (keep && !keep[i]) continue; // pruned: rng fully consumed above, no emit
    const idxF = s * CURVE_SEGS;
    const i0 = Math.min(CURVE_SEGS - 1, Math.floor(idxF));
    const f = idxF - i0;
    const base = _p
      .copy(stemPts[i0] as Vector3)
      .lerp(stemPts[i0 + 1] as Vector3, f)
      .clone();
    const dir = new Vector3(
      Math.sin(az) * Math.cos(elev),
      Math.sin(elev),
      Math.cos(az) * Math.cos(elev) * 0.35 + swing,
    ).normalize();
    const lenJ = nl * (0.75 + lenRand * 0.5) * (0.65 + 0.35 * Math.sin(Math.PI * Math.min(1, s * 1.18)));
    const tip = base.clone().addScaledVector(dir, lenJ);
    // quad across the needle, normal ≈ up-out blend
    const acrossDir = new Vector3(-dir.z, 0, dir.x).normalize().multiplyScalar(nw * 0.5);
    const nrm = new Vector3(0, 1, 0).addScaledVector(dir, -0.25).normalize();
    const hueN = hue + (hueRand - 0.5) * 0.5;
    const a0 = pushXf(g, m, base.x - acrossDir.x, base.y, base.z - acrossDir.z, nrm.x, nrm.y, nrm.z, 0, 0, hueN, flex, phase, ao * 0.9);
    const a1 = pushXf(g, m, base.x + acrossDir.x, base.y, base.z + acrossDir.z, nrm.x, nrm.y, nrm.z, 1, 0, hueN, flex, phase, ao * 0.9);
    const b0 = pushXf(g, m, tip.x - acrossDir.x * 0.25, tip.y, tip.z - acrossDir.z * 0.25, nrm.x, nrm.y, nrm.z, 0.4, 1, hueN, flex * 1.15, phase, ao);
    const b1 = pushXf(g, m, tip.x + acrossDir.x * 0.25, tip.y, tip.z + acrossDir.z * 0.25, nrm.x, nrm.y, nrm.z, 0.6, 1, hueN, flex * 1.15, phase, ao);
    g.quad(a0, b0, b1, a1);
  }
}

/** leaf cluster: `n` leaves fanned around the anchor's +z
 *
 * `lod` (crown-LOD) shapes each survivor leaf: WIDTH growth ×(1/λ) about its
 * rooted base (`anchor.pos`) to preserve area when the crown is thinned — a
 * WIDER leaf, never a merged one (pruning is per whole anchor, so no leaf is
 * ever welded to another) — plus a rows count for connected-strip coarsening.
 * `lod` fields do not touch the RNG stream, so a survivor's placement is
 * independent of which siblings were pruned; the default (widthMul 1, rows 4)
 * reproduces the LOD0 vertices bit-for-bit. */
export function buildLeafCluster(
  g: MeshGrower,
  anchor: LeafAnchor,
  shape: LeafShapeParams,
  clusterSize: [number, number],
  rng: Rng,
  lod?: LeafLod,
): void {
  const n = Math.round(clusterSize[0] + rng.float() * (clusterSize[1] - clusterSize[0]));
  const flex = 0.55 + rng.float() * 0.3;
  for (let i = 0; i < n; i++) {
    const az = (i / n) * Math.PI * 2 + rng.float() * 0.9;
    const pitch = -0.5 - rng.float() * 0.6; // droop down-out
    _q.setFromAxisAngle(new Vector3(0, 1, 0), az);
    const qp = new Quaternion().setFromAxisAngle(new Vector3(1, 0, 0), -pitch);
    const qr = anchor.quat.clone().multiply(_q).multiply(qp);
    const s = anchor.scale * (0.8 + rng.float() * 0.45);
    _m.compose(anchor.pos, qr, new Vector3(s, s, s));
    buildLeaf(
      g, _m, shape,
      anchor.hue + (rng.float() - 0.5) * 0.4,
      flex,
      rng.float() * Math.PI * 2,
      1,
      lod,
    );
  }
}

/** needle spray at an anchor
 *
 * `lod` shapes the survivor spray: needle-WIDTH growth (element 1/λ + intra-spray
 * count/kept), intra-spray needle pruning, and stem-strip coarsening — the spray
 * transform stays unit-scaled and the needle LENGTH is never touched, so a
 * survivor spray can never elongate (the spruce-spike failure). `lod` fields do
 * not add or remove RNG draws (the needle loop always runs `count` times), so the
 * default (no lod) → bit-identical to LOD0. */
export function buildSprayAt(
  g: MeshGrower,
  anchor: LeafAnchor,
  shape: LeafShapeParams,
  rng: Rng,
  lod?: SprayLod,
): void {
  _m.compose(anchor.pos, anchor.quat, new Vector3(1, 1, 1));
  buildNeedleSpray(
    g, _m, shape, anchor.scale, rng,
    anchor.hue, 0.5 + rng.float() * 0.3, rng.float() * Math.PI * 2, 1,
    lod,
  );
}
