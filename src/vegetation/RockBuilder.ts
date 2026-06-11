/**
 * Rock/boulder generator — welded icosphere displaced by a layered field:
 * ellipsoid squash → macro warp → tilted strata ledges (hardness story from
 * the terrain) → ridged creases → planar fracture cuts (craggy silhouettes,
 * spec §2) → micro grain. Hero LOD0 ≥ 200k tris; the same field evaluated at
 * lower subdivision gives consistent-silhouette LODs.
 *
 * vdata: x hue, y strataT (drives albedo banding), z moss/lichen openness,
 * w cavity AO (concavity proxy via field band).
 */

import { BufferAttribute, BufferGeometry, Vector3 } from 'three';
import { fbm3, ridged3 } from '../core/NoiseJS';
import type { Rng } from '../core/Seed';

export interface RockParams {
  /** overall radius (m) */
  radius: number;
  /** ellipsoid squash (y typically < 1) */
  squash: [number, number, number];
  /** macro shape warp amplitude (fraction of radius) */
  macro: number;
  /** strata band amplitude + frequency (0 = none) */
  strata: number;
  strataFreq: number;
  /** strata axis tilt (rad from up) */
  strataTilt: number;
  /** ridged crease amplitude */
  ridged: number;
  /** number of planar fracture cuts */
  cuts: number;
  /** cut depth (how far planes bite, fraction of radius) */
  cutBite: number;
  /** micro grain amplitude */
  micro: number;
}

export const ROCK_PRESETS = {
  hero: {
    radius: 3.4, squash: [1, 0.82, 0.9] as [number, number, number],
    macro: 0.38, strata: 0.16, strataFreq: 3.1, strataTilt: 0.3,
    ridged: 0.17, cuts: 8, cutBite: 0.34, micro: 0.022,
  },
  boulder: {
    radius: 1.1, squash: [1, 0.74, 0.92] as [number, number, number],
    macro: 0.24, strata: 0.045, strataFreq: 3.2, strataTilt: 0.2,
    ridged: 0.05, cuts: 2, cutBite: 0.1, micro: 0.012,
  },
  angular: {
    radius: 0.85, squash: [1, 0.85, 0.95] as [number, number, number],
    macro: 0.22, strata: 0.04, strataFreq: 4, strataTilt: 0.5,
    ridged: 0.12, cuts: 10, cutBite: 0.5, micro: 0.016,
  },
  slab: {
    radius: 1.7, squash: [1, 0.42, 0.78] as [number, number, number],
    macro: 0.16, strata: 0.08, strataFreq: 5, strataTilt: 0.12,
    ridged: 0.05, cuts: 5, cutBite: 0.28, micro: 0.012,
  },
  cobble: {
    radius: 0.16, squash: [1, 0.72, 0.88] as [number, number, number],
    macro: 0.14, strata: 0, strataFreq: 1, strataTilt: 0,
    ridged: 0.02, cuts: 0, cutBite: 0, micro: 0.01,
  },
  cliffFace: {
    radius: 3.0, squash: [1, 1.7, 0.5] as [number, number, number],
    macro: 0.22, strata: 0.13, strataFreq: 5.5, strataTilt: 0.1,
    ridged: 0.12, cuts: 6, cutBite: 0.4, micro: 0.02,
  },
} as const;

export type RockPreset = keyof typeof ROCK_PRESETS;

interface CutPlane {
  n: Vector3;
  off: number;
}

/** welded icosphere (edge-midpoint cache subdivision) */
function icosphere(detail: number): { pos: Float32Array; idx: Uint32Array } {
  const t = (1 + Math.sqrt(5)) / 2;
  const verts: number[][] = [
    [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
    [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
    [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
  ].map((v) => {
    const l = Math.hypot(v[0] as number, v[1] as number, v[2] as number);
    return [(v[0] as number) / l, (v[1] as number) / l, (v[2] as number) / l];
  });
  let faces: number[][] = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
  ];
  const midCache = new Map<number, number>();
  const midpoint = (a: number, b: number): number => {
    const key = a < b ? a * 16777216 + b : b * 16777216 + a;
    const hit = midCache.get(key);
    if (hit !== undefined) return hit;
    const va = verts[a] as number[];
    const vb = verts[b] as number[];
    const m = [
      ((va[0] as number) + (vb[0] as number)) / 2,
      ((va[1] as number) + (vb[1] as number)) / 2,
      ((va[2] as number) + (vb[2] as number)) / 2,
    ];
    const l = Math.hypot(m[0] as number, m[1] as number, m[2] as number);
    verts.push([(m[0] as number) / l, (m[1] as number) / l, (m[2] as number) / l]);
    const id = verts.length - 1;
    midCache.set(key, id);
    return id;
  };
  for (let d = 0; d < detail; d++) {
    const next: number[][] = [];
    for (const f of faces) {
      const [a, b, c] = f as [number, number, number];
      const ab = midpoint(a, b);
      const bc = midpoint(b, c);
      const ca = midpoint(c, a);
      next.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
    }
    faces = next;
  }
  const pos = new Float32Array(verts.length * 3);
  for (let i = 0; i < verts.length; i++) {
    const v = verts[i] as number[];
    pos[i * 3] = v[0] as number;
    pos[i * 3 + 1] = v[1] as number;
    pos[i * 3 + 2] = v[2] as number;
  }
  const idx = new Uint32Array(faces.length * 3);
  for (let i = 0; i < faces.length; i++) {
    const f = faces[i] as number[];
    idx[i * 3] = f[0] as number;
    idx[i * 3 + 1] = f[1] as number;
    idx[i * 3 + 2] = f[2] as number;
  }
  return { pos, idx };
}

export interface BuiltRock {
  geometry: BufferGeometry;
  stats: { tris: number };
}

/**
 * detail: icosphere subdivisions — 5 ≈ 20k tris, 6 ≈ 82k, 7 ≈ 327k.
 * Hero uses 7, gallery boulders 5–6, cobbles 3.
 */
export function buildRock(
  preset: RockPreset,
  rng: Rng,
  detail: number,
): BuiltRock {
  const p: RockParams = ROCK_PRESETS[preset];
  const seedA = rng.u32() & 0x7fffffff;
  const seedB = rng.u32() & 0x7fffffff;
  const seedC = rng.u32() & 0x7fffffff;

  // fracture planes
  const cuts: CutPlane[] = [];
  for (let i = 0; i < p.cuts; i++) {
    const n = new Vector3(rng.gauss(), rng.gauss() * 0.7, rng.gauss()).normalize();
    cuts.push({ n, off: 1 - p.cutBite * (0.4 + rng.float() * 0.6) });
  }
  const strataAxis = new Vector3(
    Math.sin(p.strataTilt) * Math.cos(rng.float() * 6.28),
    Math.cos(p.strataTilt),
    Math.sin(p.strataTilt) * Math.sin(rng.float() * 6.28),
  ).normalize();
  const strataPhase = rng.float() * 10;
  // per-band hardness offsets (layer-cake irregularity)
  const bandAmp: number[] = [];
  for (let i = 0; i < 24; i++) bandAmp.push(0.55 + rng.float() * 0.9);

  const { pos, idx } = icosphere(detail);
  const n = pos.length / 3;
  const dat = new Float32Array(n * 4);
  const dir = new Vector3();

  // radial displacement field along unit dir
  const fieldR = (d: Vector3): { r: number; strataT: number; cav: number } => {
    // macro warp: low-freq fbm of direction
    const macro = fbm3(d.x * 1.4, d.y * 1.4, d.z * 1.4, seedA, 3) * p.macro;
    // strata: banding along tilted axis with per-band hardness
    const s = d.dot(strataAxis) * p.strataFreq + strataPhase
      + fbm3(d.x * 2.3, d.y * 2.3, d.z * 2.3, seedB, 2) * 0.5;
    const band = Math.floor(s);
    const f = s - band;
    const amp = bandAmp[((band % 24) + 24) % 24] as number;
    // ledge profile: quick rise, slow fall → overhang-ish steps
    const ledge = (Math.min(f * 4.2, 1) - f * 0.62) * p.strata * amp;
    const rid = ridged3(d.x * 3.1, d.y * 3.1, d.z * 3.1, seedC, 3) * p.ridged;
    const micro = fbm3(d.x * 14, d.y * 14, d.z * 14, seedB ^ 0x55aa, 2) * p.micro;
    let r = 1 + macro + ledge + rid + micro;
    // fracture cuts: clamp radius against planes (flat facets), smoothed
    let cav = 0;
    for (const c of cuts) {
      const dn = d.dot(c.n);
      if (dn > 0.001) {
        const rCut = c.off / dn;
        if (rCut < r) {
          const depth = Math.min((r - rCut) * 3, 1);
          r = rCut + (r - rCut) * 0.035; // slight rounding on the facet
          cav = Math.max(cav, depth * 0.4);
        }
      }
    }
    return { r, strataT: f, cav: Math.min(1, cav + Math.max(0, -macro - ledge) * 2.2) };
  };

  for (let i = 0; i < n; i++) {
    dir.set(pos[i * 3] as number, pos[i * 3 + 1] as number, pos[i * 3 + 2] as number);
    const f = fieldR(dir);
    const rr = f.r * p.radius;
    pos[i * 3] = dir.x * rr * (p.squash[0] ?? 1);
    pos[i * 3 + 1] = dir.y * rr * (p.squash[1] ?? 1);
    pos[i * 3 + 2] = dir.z * rr * (p.squash[2] ?? 1);
    dat[i * 4] = 0; // hue jitter applied per-instance later
    dat[i * 4 + 1] = f.strataT;
    dat[i * 4 + 2] = Math.max(0, dir.y); // upness before squash ≈ moss openness
    dat[i * 4 + 3] = 1 - f.cav * 0.85;
  }

  const g = new BufferGeometry();
  g.setAttribute('position', new BufferAttribute(pos, 3));
  g.setAttribute('vdata', new BufferAttribute(dat, 4));
  g.setIndex(new BufferAttribute(idx, 1));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  return { geometry: g, stats: { tris: idx.length / 3 } };
}
