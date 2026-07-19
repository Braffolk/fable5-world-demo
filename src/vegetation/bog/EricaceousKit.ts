/**
 * EricaceousKit — low-level procedural primitives shared by the three bog
 * dwarf-shrub modules (Heather, LabradorTea, BogRosemary). NOT a pipeline file:
 * it only assembles BufferGeometry via MeshGrower + tubeForBranch. Each species
 * module owns its own habit/leaf/flower parameters and calls these helpers.
 *
 * Design (grounded in real reference — see each species module header):
 *   • growStem  — a wiry ascending dwarf-shrub stem as a tapered N-sided tube,
 *                 built as a hand-made SkelBranch fed through the engine's own
 *                 tubeForBranch (parallel-transport frames, taper, tip cap). It
 *                 returns the centerline samples so leaves/flowers attach along it.
 *   • leafBlade — a leathery lanceolate/elliptic leaf with a raised keel and
 *                 revolute (rolled-under) margins → real 3-D volume, not a flat card.
 *   • scaleLeaf — a tiny appressed scale leaf (heather) hugging the stem.
 *   • urnBell   — a small pendulous urn/bell flower (heather floret, bog-rosemary bell).
 *   • starFloret— a flat 5-petal star flower (Labrador-tea corymb floret).
 *
 * vdata convention on the CROWN geometry (leaves + flowers), mirroring
 * buildFlower / flowerMaterial (VegMaterials.ts): x = part id (0 leaf/stem,
 * 0.5 flower centre, 1 petal), y = sway flex, z = sway phase, w = baked AO.
 * The bark geometry keeps tubeForBranch's own vdata (x = per-branch hue jitter).
 */

import { Vector3 } from 'three';
import type { BufferGeometry } from 'three';
import type { Rng } from '../../core/Seed';
import { MeshGrower, tubeForBranch } from '../TubeMesh';
import type { SkelBranch } from '../VegTypes';

const UP = new Vector3(0, 1, 0);

export interface StemSample {
  p: Vector3;
  dir: Vector3;
  r: number;
  t: number;
}

export interface StemOpts {
  origin: Vector3;
  azimuth: number;
  /** 0 = upright, 1 = splayed flat outward (initial lean of the stem) */
  lean: number;
  /** stem length along its arc (m) */
  height: number;
  baseR: number;
  tipR: number;
  segs: number;
  /** random direction jitter per segment (rad-ish) — wiriness */
  wander: number;
  /** pull back toward vertical as the stem rises (ascending habit, 0..1) */
  ascend: number;
  /** tip nod outward+down (0..1) */
  recurve: number;
  swayPhase: number;
  hue: number;
}

/** Grow one wiry ascending stem into `bark`; return centerline samples. */
export function growStem(bark: MeshGrower, o: StemOpts, rng: Rng): StemSample[] {
  const outward = new Vector3(Math.cos(o.azimuth), 0, Math.sin(o.azimuth));
  const d = new Vector3().copy(UP).multiplyScalar(1 - o.lean).addScaledVector(outward, o.lean).normalize();
  const segLen = o.height / o.segs;
  const pts: Vector3[] = [o.origin.clone()];
  const cur = o.origin.clone();
  for (let i = 0; i < o.segs; i++) {
    const t = (i + 1) / o.segs;
    // wiry wander
    d.x += (rng.float() - 0.5) * o.wander;
    d.y += (rng.float() - 0.5) * o.wander * 0.5;
    d.z += (rng.float() - 0.5) * o.wander;
    // ascending: blend the vertical component back in
    const dotUp = d.dot(UP);
    const towardUp = new Vector3().copy(UP).addScaledVector(d, -dotUp);
    if (towardUp.lengthSq() > 1e-8) d.addScaledVector(towardUp.normalize(), o.ascend * segLen * 2.2);
    // tip nod
    if (t > 0.62) d.addScaledVector(outward, o.recurve * 0.14).addScaledVector(UP, -o.recurve * 0.05);
    d.normalize();
    cur.addScaledVector(d, segLen);
    pts.push(cur.clone());
  }

  const n = pts.length;
  const dirs: Vector3[] = [];
  const radii: number[] = [];
  for (let i = 0; i < n; i++) {
    const a = pts[Math.max(0, i)] as Vector3;
    const b = pts[Math.min(n - 1, i + 1)] as Vector3;
    const dir = i < n - 1 ? new Vector3().subVectors(b, a).normalize() : (dirs[i - 1] as Vector3).clone();
    dirs.push(dir);
    const tt = i / (n - 1);
    radii.push(o.baseR + (o.tipR - o.baseR) * tt);
  }
  let len = 0;
  for (let i = 1; i < n; i++) len += (pts[i] as Vector3).distanceTo(pts[i - 1] as Vector3);

  const br: SkelBranch = {
    level: 1, // level ≥ 1 → no root flare
    pts,
    radii,
    dirs,
    len,
    tParent: 0,
    broken: false,
    parentIdx: -1,
  };
  tubeForBranch(
    bark,
    br,
    {
      ringSegs: 5,
      uRepeats: 1,
      vScale: 1,
      swayPhase: o.swayPhase,
      swayFlexBase: 0.08,
      swayFlexTip: 0.6,
      hue: o.hue,
    },
    rng,
  );

  const samples: StemSample[] = [];
  for (let i = 0; i < n; i++) {
    samples.push({ p: (pts[i] as Vector3).clone(), dir: (dirs[i] as Vector3).clone(), r: radii[i] as number, t: i / (n - 1) });
  }
  return samples;
}

/**
 * Walk a stem's centerline in real arc-length steps of `spacing` (m), starting at
 * fraction `tStart` of the total length, invoking `cb` with the interpolated
 * point, unit tangent, and normalized position t at each station.
 */
export function walkStem(
  samples: StemSample[],
  spacing: number,
  tStart: number,
  cb: (p: Vector3, dir: Vector3, t: number, k: number) => void,
): void {
  const n = samples.length;
  if (n < 2) return;
  const seg: number[] = [0];
  let total = 0;
  for (let i = 1; i < n; i++) {
    total += (samples[i] as StemSample).p.distanceTo((samples[i - 1] as StemSample).p);
    seg.push(total);
  }
  if (total < 1e-6) return;
  let k = 0;
  const p = new Vector3();
  const dir = new Vector3();
  for (let s = tStart * total; s < total; s += spacing) {
    // locate segment containing arc-length s
    let i = 1;
    while (i < n && (seg[i] as number) < s) i++;
    const i0 = Math.min(n - 1, i);
    const s0 = seg[i0 - 1] as number;
    const s1 = seg[i0] as number;
    const f = s1 > s0 ? (s - s0) / (s1 - s0) : 0;
    p.copy((samples[i0 - 1] as StemSample).p).lerp((samples[i0] as StemSample).p, f);
    dir.copy((samples[i0 - 1] as StemSample).dir).lerp((samples[i0] as StemSample).dir, f).normalize();
    cb(p.clone(), dir.clone(), s / total, k++);
  }
}

/** two unit vectors perpendicular to `dir` (and to each other). */
export function perpFrame(dir: Vector3, outU: Vector3, outV: Vector3): void {
  const ref = Math.abs(dir.y) < 0.94 ? UP : new Vector3(1, 0, 0);
  outU.crossVectors(ref, dir).normalize();
  outV.crossVectors(dir, outU).normalize();
}

/**
 * A leathery leaf blade: raised keel (V midrib) + revolute margins rolled under,
 * lanceolate/elliptic width profile (w0 base → w1 mid → w2 tip). `axis` runs the
 * length, `side` is across the blade, both unit; `keel` lifts the midrib, `revolute`
 * rolls the margins under. Real 3-D volume, per-vertex tilted normals for shading.
 */
export function leafBlade(
  g: MeshGrower,
  base: Vector3,
  axis: Vector3,
  side: Vector3,
  len: number,
  w0: number,
  w1: number,
  w2: number,
  keel: number,
  revolute: number,
  hue: number,
  vdx: number,
  aoBase: number,
  aoTip: number,
  segs = 3,
): void {
  const n = new Vector3().crossVectors(axis, side).normalize(); // face normal (up)
  const nL = new Vector3().copy(n).addScaledVector(side, -0.4).normalize();
  const nR = new Vector3().copy(n).addScaledVector(side, 0.4).normalize();
  const widthAt = (t: number): number => (t < 0.5 ? w0 + (w1 - w0) * (t / 0.5) : w1 + (w2 - w1) * ((t - 0.5) / 0.5));
  const L: number[] = [];
  const M: number[] = [];
  const R: number[] = [];
  const c = new Vector3();
  const pL = new Vector3();
  const pR = new Vector3();
  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    const w = widthAt(t);
    c.copy(axis).multiplyScalar(len * t).add(base);
    const mid = new Vector3().copy(c).addScaledVector(n, keel * w);
    pL.copy(c).addScaledVector(side, -w).addScaledVector(n, -revolute * w);
    pR.copy(c).addScaledVector(side, w).addScaledVector(n, -revolute * w);
    const ao = aoBase + (aoTip - aoBase) * t;
    L.push(g.vertex(pL.x, pL.y, pL.z, nL.x, nL.y, nL.z, 0, t, vdx, 0.4, hue, ao));
    M.push(g.vertex(mid.x, mid.y, mid.z, n.x, n.y, n.z, 0.5, t, vdx, 0.5, hue, ao));
    R.push(g.vertex(pR.x, pR.y, pR.z, nR.x, nR.y, nR.z, 1, t, vdx, 0.4, hue, ao));
  }
  for (let i = 0; i < segs; i++) {
    g.quad(L[i] as number, M[i] as number, M[i + 1] as number, L[i + 1] as number);
    g.quad(M[i] as number, R[i] as number, R[i + 1] as number, M[i + 1] as number);
  }
}

/** Tiny appressed scale leaf (heather): a small 2-tri leaf hugging the stem. */
export function scaleLeaf(
  g: MeshGrower,
  base: Vector3,
  axis: Vector3,
  side: Vector3,
  outN: Vector3,
  len: number,
  wid: number,
  hue: number,
  vdx: number,
  ao: number,
): void {
  const tip = new Vector3().copy(base).addScaledVector(axis, len).addScaledVector(outN, len * 0.25);
  const a0 = new Vector3().copy(base).addScaledVector(side, -wid);
  const a1 = new Vector3().copy(base).addScaledVector(side, wid);
  const v0 = g.vertex(a0.x, a0.y, a0.z, outN.x, outN.y, outN.z, 0, 0, vdx, 0.5, hue, ao);
  const v1 = g.vertex(a1.x, a1.y, a1.z, outN.x, outN.y, outN.z, 1, 0, vdx, 0.5, hue, ao);
  const v2 = g.vertex(tip.x, tip.y, tip.z, outN.x, outN.y, outN.z, 0.5, 1, vdx, 0.7, hue, ao);
  g.tri(v0, v1, v2);
}

/**
 * A small pendulous urn/bell flower along `dir` (the direction it hangs/points).
 * Lathe of a narrow-neck → round-belly → constricted-mouth profile. `size` is the
 * bell length (m). vdata.x = 1 (petal). Used for heather florets + bog-rosemary bells.
 */
export function urnBell(
  g: MeshGrower,
  attach: Vector3,
  dir: Vector3,
  size: number,
  sides: number,
  swayPhase: number,
): void {
  const u = new Vector3();
  const v = new Vector3();
  perpFrame(dir, u, v);
  // profile: [t along dir, radiusFrac]
  const prof: [number, number][] = [
    [0.0, 0.2],
    [0.52, 0.5],
    [1.0, 0.34],
  ];
  const rings: number[][] = [];
  const c = new Vector3();
  const rad = new Vector3();
  const nrm = new Vector3();
  for (let ri = 0; ri < prof.length; ri++) {
    const [tt, rf] = prof[ri] as [number, number];
    c.copy(attach).addScaledVector(dir, size * tt);
    const rr = rf * size;
    const ring: number[] = [];
    const ao = 0.55 + 0.45 * tt; // mouth brighter
    for (let k = 0; k <= sides; k++) {
      const a = (k / sides) * Math.PI * 2;
      rad.copy(u).multiplyScalar(Math.cos(a)).addScaledVector(v, Math.sin(a));
      nrm.copy(rad).normalize();
      const px = c.x + rad.x * rr;
      const py = c.y + rad.y * rr;
      const pz = c.z + rad.z * rr;
      ring.push(g.vertex(px, py, pz, nrm.x, nrm.y, nrm.z, k / sides, tt, 1, 0.6, swayPhase, ao));
    }
    rings.push(ring);
  }
  for (let ri = 0; ri < rings.length - 1; ri++) {
    const a = rings[ri] as number[];
    const b = rings[ri + 1] as number[];
    for (let k = 0; k < sides; k++) {
      g.quad(a[k] as number, a[k + 1] as number, b[k + 1] as number, b[k] as number);
    }
  }
}

/** A short thin pedicel (flower-coloured) — bog-rosemary's nodding pink stalks. */
export function pedicel(g: MeshGrower, a: Vector3, b: Vector3, hw: number, swayPhase: number): void {
  const dir = new Vector3().subVectors(b, a).normalize();
  const u = new Vector3();
  const v = new Vector3();
  perpFrame(dir, u, v);
  const mk = (p: Vector3, off: Vector3, vv: number): number =>
    g.vertex(p.x + off.x, p.y + off.y, p.z + off.z, u.x, u.y, u.z, 0, vv, 1, 0.5, swayPhase, 0.8);
  const uL = new Vector3().copy(u).multiplyScalar(-hw);
  const uR = new Vector3().copy(u).multiplyScalar(hw);
  const a0 = mk(a, uL, 0);
  const a1 = mk(a, uR, 0);
  const b0 = mk(b, uL, 1);
  const b1 = mk(b, uR, 1);
  g.quad(a0, a1, b1, b0);
}

/**
 * A flat 5-petal white star flower (Labrador-tea corymb floret) in the plane
 * whose normal is `up`. `size` is the flower radius. Petals vdata.x = 1, tiny
 * centre vdata.x = 0.5.
 */
export function starFloret(g: MeshGrower, center: Vector3, up: Vector3, size: number, swayPhase: number): void {
  const u = new Vector3();
  const v = new Vector3();
  perpFrame(up, u, v);
  const petals = 5;
  const dir = new Vector3();
  const perp = new Vector3();
  const tip = new Vector3();
  const s0 = new Vector3();
  const s1 = new Vector3();
  for (let i = 0; i < petals; i++) {
    const a = (i / petals) * Math.PI * 2;
    dir.copy(u).multiplyScalar(Math.cos(a)).addScaledVector(v, Math.sin(a));
    perp.copy(u).multiplyScalar(-Math.sin(a)).addScaledVector(v, Math.cos(a));
    const baseR = size * 0.16;
    const w = size * 0.32;
    tip.copy(center).addScaledVector(dir, size).addScaledVector(up, size * 0.08);
    s0.copy(center).addScaledVector(dir, baseR).addScaledVector(perp, -w);
    s1.copy(center).addScaledVector(dir, baseR).addScaledVector(perp, w);
    const c = g.vertex(center.x, center.y, center.z, up.x, up.y, up.z, 0.5, 0, 0.5, 0.4, swayPhase, 0.7);
    const p0 = g.vertex(s0.x, s0.y, s0.z, up.x, up.y, up.z, 0, 0.5, 1, 0.5, swayPhase, 0.85);
    const p1 = g.vertex(s1.x, s1.y, s1.z, up.x, up.y, up.z, 1, 0.5, 1, 0.5, swayPhase, 0.85);
    const pt = g.vertex(tip.x, tip.y, tip.z, up.x, up.y, up.z, 0.5, 1, 1, 0.6, swayPhase, 1);
    g.tri(c, p0, pt);
    g.tri(c, pt, p1);
  }
}

/** Weld several built geometries into one (identity transform), preserving all
 *  four vertex streams + index. Used to merge leaf + flower growers into the
 *  single `crown` geometry the pipeline consumes. */
export function mergeGeo(geos: BufferGeometry[]): BufferGeometry {
  const g = new MeshGrower();
  for (const src of geos) {
    const pos = src.getAttribute('position');
    const nrm = src.getAttribute('normal');
    const uv = src.getAttribute('uv');
    const dat = src.getAttribute('vdata');
    const idx = src.getIndex();
    const base = g.vertCount;
    for (let i = 0; i < pos.count; i++) {
      g.vertex(
        pos.getX(i), pos.getY(i), pos.getZ(i),
        nrm ? nrm.getX(i) : 0, nrm ? nrm.getY(i) : 1, nrm ? nrm.getZ(i) : 0,
        uv ? uv.getX(i) : 0, uv ? uv.getY(i) : 0,
        dat ? dat.getX(i) : 0, dat ? dat.getY(i) : 0, dat ? dat.getZ(i) : 0, dat ? dat.getW(i) : 1,
      );
    }
    if (idx) {
      for (let i = 0; i < idx.count; i += 3) {
        g.tri(base + idx.getX(i), base + idx.getX(i + 1), base + idx.getX(i + 2));
      }
    } else {
      for (let i = 0; i < pos.count; i += 3) g.tri(base + i, base + i + 1, base + i + 2);
    }
  }
  return g.build();
}
