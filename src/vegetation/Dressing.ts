/**
 * Dressing generators (Pillar C): hanging vines for overhangs, mushrooms
 * (cap + shelf) for tree bases and dead logs. Leaf cards on vines reuse a
 * captured foliage atlas via the standard card material.
 */

import { Quaternion, Vector3 } from 'three';
import type { BufferGeometry } from 'three';
import type { Rng } from '../core/Seed';
import { MeshGrower } from './TubeMesh';
import { buildFoliageCards } from './FoliageCards';
import type { LeafAnchor } from './VegTypes';

/**
 * Hanging vine curtain: `count` strands sagging from a top anchor line
 * (local origin = top center, strands hang −y). Returns stem geometry and
 * leaf-card geometry (texture via any broadleaf atlas).
 */
export function buildVines(
  rng: Rng,
  width: number,
  drop: number,
  count: number,
): { stems: BufferGeometry; leaves: BufferGeometry } {
  const stems = new MeshGrower();
  const leaves = new MeshGrower();
  const anchors: LeafAnchor[] = [];
  const q = new Quaternion();
  const qt = new Quaternion();
  const Y = new Vector3(0, 1, 0);
  const X = new Vector3(1, 0, 0);
  for (let s = 0; s < count; s++) {
    const x0 = (s / (count - 1) - 0.5) * width + (rng.float() - 0.5) * 0.3;
    const z0 = (rng.float() - 0.5) * 0.25;
    const len = drop * (0.45 + rng.float() * 0.55);
    const segs = 7;
    const sway = (rng.float() - 0.5) * 0.35;
    const pts: Vector3[] = [];
    for (let i = 0; i <= segs; i++) {
      const t = i / segs;
      pts.push(
        new Vector3(
          x0 + Math.sin(t * 2.2 + s) * 0.07 + sway * t * t,
          -t * len,
          z0 + Math.cos(t * 1.7 + s * 2) * 0.06 + t * 0.18,
        ),
      );
    }
    // thin strand tube (4-sided)
    const hue = rng.float() * 2 - 1;
    const ringN = 4;
    const ringIds: number[][] = [];
    const N = new Vector3();
    const B = new Vector3();
    for (let i = 0; i <= segs; i++) {
      const d =
        i === 0
          ? (pts[1] as Vector3).clone().sub(pts[0] as Vector3).normalize()
          : (pts[i] as Vector3).clone().sub(pts[i - 1] as Vector3).normalize();
      const ref = Math.abs(d.y) < 0.9 ? new Vector3(0, 1, 0) : new Vector3(1, 0, 0);
      N.crossVectors(ref, d).normalize();
      B.crossVectors(d, N).normalize();
      const rr = 0.012 * (1 - (i / segs) * 0.5);
      const ring: number[] = [];
      for (let k = 0; k <= ringN; k++) {
        const a = (k / ringN) * Math.PI * 2;
        const nx = N.x * Math.cos(a) + B.x * Math.sin(a);
        const ny = N.y * Math.cos(a) + B.y * Math.sin(a);
        const nz = N.z * Math.cos(a) + B.z * Math.sin(a);
        const pt = pts[i] as Vector3;
        ring.push(
          stems.vertex(pt.x + nx * rr, pt.y + ny * rr, pt.z + nz * rr, nx, ny, nz, k / ringN, i / segs, hue, 0.6, s, 0.8),
        );
      }
      ringIds.push(ring);
    }
    for (let i = 0; i < segs; i++) {
      const a = ringIds[i] as number[];
      const b = ringIds[i + 1] as number[];
      for (let k = 0; k < ringN; k++) {
        stems.quad(a[k] as number, b[k] as number, b[k + 1] as number, a[k + 1] as number);
      }
    }
    // leaf cards along the strand
    const nl = Math.round(len / 0.22);
    for (let i = 0; i < nl; i++) {
      const t = (i + 0.5) / nl;
      const idxF = t * segs;
      const i0 = Math.min(segs - 1, Math.floor(idxF));
      const f = idxF - i0;
      const pos = (pts[i0] as Vector3).clone().lerp(pts[i0 + 1] as Vector3, f);
      q.setFromAxisAngle(Y, rng.float() * Math.PI * 2);
      qt.setFromAxisAngle(X, -Math.PI / 2 + 0.35 + (rng.float() - 0.5) * 0.6);
      q.multiply(qt);
      anchors.push({
        pos,
        quat: q.clone(),
        scale: 0.1 + rng.float() * 0.07,
        hue: rng.float() * 2 - 1,
        age: rng.float() * 0.5,
      });
    }
  }
  buildFoliageCards(leaves, anchors, { mode: 'cross', sizeK: 2.0 }, rng);
  return { stems: stems.build(), leaves: leaves.build() };
}

/**
 * Mushroom: cap (lathed dome w/ gill underside) + stem. kind 'cap' stands
 * on the ground; 'shelf' is a half-cap for log/trunk sides.
 * vdata.x: 0 = stem, 1 = cap top, 0.5 = gills.
 */
export function buildMushroom(rng: Rng, kind: 'cap' | 'shelf'): BufferGeometry {
  const g = new MeshGrower();
  const capR = kind === 'cap' ? 0.035 + rng.float() * 0.05 : 0.05 + rng.float() * 0.07;
  const capH = capR * (kind === 'cap' ? 0.55 + rng.float() * 0.45 : 0.3);
  const stemH = kind === 'cap' ? capR * (1.2 + rng.float() * 1.4) : 0;
  const SEG = 9;
  const arc = kind === 'cap' ? Math.PI * 2 : Math.PI;
  // cap: dome rows
  const rows = 4;
  const rowIds: number[][] = [];
  for (let i = 0; i <= rows; i++) {
    const t = i / rows;
    const ang = t * Math.PI * 0.5;
    const r = capR * Math.cos(ang) * (1 + (rng.float() - 0.5) * 0.06);
    const y = stemH + Math.sin(ang) * capH;
    const ring: number[] = [];
    for (let k = 0; k <= SEG; k++) {
      const a = (k / SEG) * arc;
      const nx = Math.cos(a) * Math.cos(ang);
      const nz = Math.sin(a) * Math.cos(ang);
      const ny = Math.sin(ang) * 0.8 + 0.2;
      ring.push(
        g.vertex(
          Math.cos(a) * r, y, Math.sin(a) * r,
          nx, ny, nz,
          k / SEG, t,
          1, 0, 0, 0.85 + t * 0.15,
        ),
      );
    }
    rowIds.push(ring);
  }
  for (let i = 0; i < rows; i++) {
    const a = rowIds[i] as number[];
    const b = rowIds[i + 1] as number[];
    for (let k = 0; k < SEG; k++) {
      g.quad(a[k] as number, a[k + 1] as number, b[k + 1] as number, b[k] as number);
    }
  }
  // gill underside disc
  const center = g.vertex(0, stemH * 0.98, 0, 0, -1, 0, 0.5, 0.5, 0.5, 0, 0, 0.6);
  const rim = rowIds[0] as number[];
  for (let k = 0; k < SEG; k++) {
    g.tri(center, rim[k] as number, rim[k + 1] as number);
  }
  // stem
  if (stemH > 0) {
    const sr = capR * 0.28;
    const sIds: number[][] = [];
    for (let i = 0; i <= 2; i++) {
      const t = i / 2;
      const ring: number[] = [];
      for (let k = 0; k <= 6; k++) {
        const a = (k / 6) * Math.PI * 2;
        ring.push(
          g.vertex(
            Math.cos(a) * sr * (1.25 - t * 0.25), t * stemH, Math.sin(a) * sr * (1.25 - t * 0.25),
            Math.cos(a), 0.1, Math.sin(a),
            k / 6, t, 0, 0, 0, 0.7 + t * 0.3,
          ),
        );
      }
      sIds.push(ring);
    }
    for (let i = 0; i < 2; i++) {
      const a = sIds[i] as number[];
      const b = sIds[i + 1] as number[];
      for (let k = 0; k < 6; k++) {
        g.quad(a[k] as number, b[k] as number, b[k + 1] as number, a[k + 1] as number);
      }
    }
  }
  return g.build();
}
