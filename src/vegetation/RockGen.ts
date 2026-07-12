/**
 * RockGen — seeded SDF-composed rock library generator (SPEC-ROCKS §B–§D, §G).
 *
 * ⛔ USER LAW: rocks are MESHES. The SDF exists ONLY inside this generator —
 * the output is ordinary indexed typed-array mesh data (positions/normals/
 * vdata/indices) for the standard registerMesh + DAG path. Pure CPU f64,
 * zero three.js imports, node-runnable like Clusterize/BuildDag (validated
 * by tools/probe-rockgen.ts).
 *
 * Pipeline per variant, deterministic from one Rng stream per
 * (archetype[:mod], variant):
 *   §C field recipe (smin ellipsoid base → cut planes → IQ fbm-SDF →
 *   worley-crack / plateau-strata relief → bottom flatten)
 *   → §G narrow band: ¼-res coarse pass, refine only blocks with |d| near 0,
 *     trilinear fill elsewhere; outermost shell clamped ≥ +ε (closed surface)
 *   → §D naive Surface Nets: one vertex per sign-change cell at the
 *     field-zero centroid (edge-crossing average), a quad per sign-change
 *     grid edge → 2 tris wound so face normals follow the field gradient
 *   → vertex normals = normalized field gradient (central differences)
 *   → per-vertex vdata 4×u8: curvature / flow coord / AO×upness / cavity AO
 *   → validation (throws loud): every edge shared by exactly 2 tris with
 *     opposite winding (open-address edge audit mirroring Clusterize),
 *     zero-area tris dropped, < 65k verts, positive enclosed volume.
 */

import { WorldSeed, mix32 } from '../core/Seed';
import type { Rng } from '../core/Seed';

export type RockArchetype =
  | 'graniteErratic'
  | 'fieldstone'
  | 'angularShard'
  | 'flatCobble'
  | 'pebble';

/** recipe modifier: 'low' = squashed erratic (Slab), 'large' = big flat
 *  cobble (Slab), 'hero' = EtakErratic stream (recipe unchanged; detail
 *  comes from the 96³ grid). */
export type RockMod = 'hero' | 'low' | 'large';

export interface RockStats {
  verts: number;
  tris: number;
  quads: number;
  /** degenerate tris removed before the audit (expected 0) */
  droppedTris: number;
  /** closed non-manifold pinch edges (≤1% tolerated; retired downstream) */
  pinchEdges: number;
  /** exact field evaluations (narrow band; compare vs (N+1)³ full grid) */
  fieldEvals: number;
  /** max |vertex| — measured nominal radius for R1 instance scaling */
  boundRadius: number;
  gridMs: number;
  meshMs: number;
  bakeMs: number;
  totalMs: number;
}

export interface RockMesh {
  positions: Float32Array;
  normals: Float32Array;
  /** 4 × u8 per vertex: x curvature, y flow coord, z AO×upness, w cavity AO */
  vdata: Uint8Array;
  indices: Uint32Array;
  stats: RockStats;
}

// ---------------------------------------------------------------------------
// scalar SDF ops (§C)

/** polynomial smooth min (IQ) */
function smin(a: number, b: number, k: number): number {
  if (k <= 0) return Math.min(a, b);
  const h = Math.max(k - Math.abs(a - b), 0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
}

function smax(a: number, b: number, k: number): number {
  return -smin(-a, -b, k);
}

function sstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/** IQ ellipsoid distance approximation (exact enough for field composition) */
function sdEllipsoid(
  px: number, py: number, pz: number,
  cx: number, cy: number, cz: number,
  rx: number, ry: number, rz: number,
): number {
  const ux = (px - cx) / rx;
  const uy = (py - cy) / ry;
  const uz = (pz - cz) / rz;
  const k0 = Math.sqrt(ux * ux + uy * uy + uz * uz);
  if (k0 < 1e-9) return -Math.min(rx, Math.min(ry, rz));
  const vx = ux / rx;
  const vy = uy / ry;
  const vz = uz / rz;
  const k1 = Math.sqrt(vx * vx + vy * vy + vz * vz);
  return (k0 * (k0 - 1)) / k1;
}

/** deterministic per-cell hash → [0,1) */
function hash01(x: number, y: number, z: number, salt: number): number {
  const h = (Math.imul(x | 0, 0x9e3779b1) ^ Math.imul(y | 0, 0x85ebca77) ^ Math.imul(z | 0, 0xc2b2ae3d) ^ (salt | 0)) >>> 0;
  return mix32(h) / 4294967296;
}

// ---------------------------------------------------------------------------
// noise fields (all shape the FIELD — no shading noise here)

/** min distance to 8 spheres on the surrounding unit-grid corners, radius
 *  0.5·hash(corner) — the IQ fbm-SDF base primitive */
function sdSphereGrid(x: number, y: number, z: number, salt: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const iz = Math.floor(z);
  let best = Infinity;
  for (let dx = 0; dx <= 1; dx++) {
    for (let dy = 0; dy <= 1; dy++) {
      for (let dz = 0; dz <= 1; dz++) {
        const cx = ix + dx;
        const cy = iy + dy;
        const cz = iz + dz;
        const r = 0.5 * hash01(cx, cy, cz, salt);
        const ex = x - cx;
        const ey = y - cy;
        const ez = z - cz;
        const d = Math.sqrt(ex * ex + ey * ey + ez * ez) - r;
        if (d < best) best = d;
      }
    }
  }
  return best;
}

/** 3D Worley F2−F1 (0 on cell borders → crack lines) */
function worleyF2mF1(x: number, y: number, z: number, salt: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const iz = Math.floor(z);
  let f1 = Infinity;
  let f2 = Infinity;
  for (let dx = -1; dx <= 1; dx++) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dz = -1; dz <= 1; dz++) {
        const cx = ix + dx;
        const cy = iy + dy;
        const cz = iz + dz;
        const ex = cx + hash01(cx, cy, cz, salt) - x;
        const ey = cy + hash01(cx, cy, cz, salt ^ 0x68bc21eb) - y;
        const ez = cz + hash01(cx, cy, cz, salt ^ 0x02e5be93) - z;
        const d = Math.sqrt(ex * ex + ey * ey + ez * ez);
        if (d < f1) {
          f2 = f1;
          f1 = d;
        } else if (d < f2) {
          f2 = d;
        }
      }
    }
  }
  return f2 - f1;
}

/** periodic wave with flat tops/bottoms (clamped scaled cosine) → stepped
 *  strata ledges when added to the field */
function plateauWave(t: number): number {
  const c = Math.cos(t * Math.PI * 2) * 2.5;
  return c < -1 ? -1 : c > 1 ? 1 : c;
}

// ---------------------------------------------------------------------------
// fbm-SDF (§C pseudocode, verbatim structure): corner-sphere grid per octave,
// clipped against the inflated host then smooth-unioned — lumps AND bites,
// stays a valid SDF. Octave scale starts at 2·amp so first-octave lumps have
// amplitude ≈ amp (sphere radius ≤ 0.5·s), halving per octave; domain is
// rotated between octaves to decorrelate the lattices.

type FieldFn = (x: number, y: number, z: number) => number;

function makeFbm(rng: Rng, octaves: number, amp: number): (x: number, y: number, z: number, d: number) => number {
  const salt = rng.u32() | 0;
  const rots = new Float64Array(octaves * 9);
  for (let o = 0; o < octaves; o++) {
    // Rodrigues axis-angle rotation matrix
    let ax = rng.gauss();
    let ay = rng.gauss();
    let az = rng.gauss();
    const al = Math.sqrt(ax * ax + ay * ay + az * az) || 1;
    ax /= al;
    ay /= al;
    az /= al;
    const ang = rng.range(0.5, Math.PI * 2 - 0.5);
    const c = Math.cos(ang);
    const s = Math.sin(ang);
    const t = 1 - c;
    const m = o * 9;
    rots[m] = t * ax * ax + c;
    rots[m + 1] = t * ax * ay - s * az;
    rots[m + 2] = t * ax * az + s * ay;
    rots[m + 3] = t * ax * ay + s * az;
    rots[m + 4] = t * ay * ay + c;
    rots[m + 5] = t * ay * az - s * ax;
    rots[m + 6] = t * ax * az - s * ay;
    rots[m + 7] = t * ay * az + s * ax;
    rots[m + 8] = t * az * az + c;
  }
  return (x: number, y: number, z: number, d: number): number => {
    let s = amp * 2;
    let px = x;
    let py = y;
    let pz = z;
    for (let o = 0; o < octaves; o++) {
      let n = sdSphereGrid(px / s, py / s, pz / s, salt + o * 0x9e37) * s;
      n = smax(n, d - 0.1 * s, 0.3 * s); // clip against inflated host
      d = smin(n, d, 0.3 * s); // union: lumps AND bites
      const m = o * 9;
      const qx = rots[m] * px + rots[m + 1] * py + rots[m + 2] * pz;
      const qy = rots[m + 3] * px + rots[m + 4] * py + rots[m + 5] * pz;
      const qz = rots[m + 6] * px + rots[m + 7] * py + rots[m + 8] * pz;
      px = qx;
      py = qy;
      pz = qz;
      s *= 0.5;
    }
    return d;
  };
}

// ---------------------------------------------------------------------------
// archetype field recipes (§C table). Unit nominal radius, object space.
// `dom` = half-extent of the (cubic) sample domain — the tri-target tuning
// lever: tris ≈ 2 × sign-change edges ∝ (surface area)/(2·dom/N)².

interface FieldDef {
  f: FieldFn;
  /** vdata.y flow coord (strata phase / crack proximity), → [0,1] */
  flow: FieldFn;
  /** sample-domain half-extent (before per-library-entry domainScale) */
  dom: number;
}

const FLOW_NONE: FieldFn = (_x: number, _y: number, _z: number) => 0.5;

/** smin-union of n random ellipsoids (centers ±cr, radii [rLo,rHi] per axis) */
function makeEllipsoidBase(
  rng: Rng, n: number, cr: number, rLo: number, rHi: number, k: number, ySquash: number,
): FieldFn {
  const e = new Float64Array(n * 6);
  for (let i = 0; i < n; i++) {
    e[i * 6] = rng.range(-cr, cr);
    e[i * 6 + 1] = rng.range(-cr * 0.7, cr * 0.7) * ySquash;
    e[i * 6 + 2] = rng.range(-cr, cr);
    e[i * 6 + 3] = rng.range(rLo, rHi);
    e[i * 6 + 4] = rng.range(rLo, rHi) * ySquash;
    e[i * 6 + 5] = rng.range(rLo, rHi);
  }
  return (x, y, z) => {
    let d = sdEllipsoid(x, y, z, e[0], e[1], e[2], e[3], e[4], e[5]);
    for (let i = 1; i < n; i++) {
      const m = i * 6;
      d = smin(d, sdEllipsoid(x, y, z, e[m], e[m + 1], e[m + 2], e[m + 3], e[m + 4], e[m + 5]), k);
    }
    return d;
  };
}

/** random unit normals + offsets for cut planes / half-space chains */
function makePlanes(rng: Rng, n: number, offLo: number, offHi: number): Float64Array {
  const p = new Float64Array(n * 4);
  for (let i = 0; i < n; i++) {
    let nx = rng.gauss();
    let ny = rng.gauss();
    let nz = rng.gauss();
    const l = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
    nx /= l;
    ny /= l;
    nz /= l;
    p[i * 4] = nx;
    p[i * 4 + 1] = ny;
    p[i * 4 + 2] = nz;
    p[i * 4 + 3] = rng.range(offLo, offHi);
  }
  return p;
}

function makeField(archetype: RockArchetype, mod: RockMod | undefined, rng: Rng): FieldDef {
  switch (archetype) {
    case 'graniteErratic': {
      // base 3-4 ellipsoids; 2-4 weathered cut facets (LARGE kEdge); fbm 3 oct
      // 0.10r; worley cracks; part-buried flat bottom.
      const low = mod === 'low';
      const base = makeEllipsoidBase(rng, 3 + rng.int(2), 0.35, 0.6, 0.9, 0.25, low ? 0.55 : 1);
      const cuts = makePlanes(rng, 2 + rng.int(3), 0.5, 0.85);
      const nCut = cuts.length / 4;
      const fbm = makeFbm(rng, 3, 0.1);
      const crackSalt = rng.u32() | 0;
      const crackFreq = 2.5;
      const crackDepth = 0.03;
      const crackW = 0.08;
      const flattenY = low ? 0.3 : 0.55;
      const f: FieldFn = (x, y, z) => {
        let d = base(x, y, z);
        for (let i = 0; i < nCut; i++) {
          const m = i * 4;
          d = smax(d, x * cuts[m] + y * cuts[m + 1] + z * cuts[m + 2] - cuts[m + 3], 0.12);
        }
        d = fbm(x, y, z, d);
        d += crackDepth * (1 - sstep(0, crackW, worleyF2mF1(x * crackFreq, y * crackFreq, z * crackFreq, crackSalt)));
        return smax(d, -(y + flattenY), 0.2);
      };
      const flow: FieldFn = (x, y, z) =>
        1 - sstep(0, crackW * 2, worleyF2mF1(x * crackFreq, y * crackFreq, z * crackFreq, crackSalt));
      return { f, flow, dom: 1.6 };
    }
    case 'fieldstone': {
      // very rounded glacial fieldstone: 2-3 ellipsoids k=0.35, gentle fbm.
      const base = makeEllipsoidBase(rng, 2 + rng.int(2), 0.3, 0.65, 0.9, 0.35, 1);
      const fbm = makeFbm(rng, 2, 0.05);
      return { f: (x, y, z) => fbm(x, y, z, base(x, y, z)), flow: FLOW_NONE, dom: 1.4 };
    }
    case 'angularShard': {
      // talus block: ellipsoid ∩ 5-8 half-spaces (hard smax chain), light fbm
      // so facets aren't dead-flat.
      const rx = rng.range(0.85, 1.0);
      const ry = rng.range(0.55, 0.75);
      const rz = rng.range(0.7, 0.9);
      const planes = makePlanes(rng, 5 + rng.int(4), 0.35, 0.65);
      const nPl = planes.length / 4;
      const fbm = makeFbm(rng, 2, 0.035);
      const f: FieldFn = (x, y, z) => {
        let d = sdEllipsoid(x, y, z, 0, 0, 0, rx, ry, rz);
        for (let i = 0; i < nPl; i++) {
          const m = i * 4;
          d = smax(d, x * planes[m] + y * planes[m + 1] + z * planes[m + 2] - planes[m + 3], 0.03);
        }
        return fbm(x, y, z, d);
      };
      return { f, flow: FLOW_NONE, dom: 1.05 };
    }
    case 'flatCobble': {
      // squashed ellipsoid + one side-lobe; tilted plateau strata; 1-oct fbm.
      const large = mod === 'large';
      const sc = large ? 1.25 : 1;
      const rx = rng.range(0.8, 1.0) * sc;
      const ry = rng.range(0.35, 0.5) * sc;
      const rz = rng.range(0.75, 0.95) * sc;
      const lobeA = rng.range(0, Math.PI * 2);
      const lx = Math.cos(lobeA) * rx * 0.6;
      const lz = Math.sin(lobeA) * rz * 0.6;
      const lr = rng.range(0.45, 0.6) * sc;
      const tilt = rng.range(0.1, 0.2);
      const tiltA = rng.range(0, Math.PI * 2);
      const axX = Math.sin(tilt) * Math.cos(tiltA);
      const axY = Math.cos(tilt);
      const axZ = Math.sin(tilt) * Math.sin(tiltA);
      const strataFreq = rng.range(4, 7);
      const strataPhase = rng.float();
      const strataAmp = 0.02;
      const fbm = makeFbm(rng, 1, 0.03);
      const f: FieldFn = (x, y, z) => {
        let d = sdEllipsoid(x, y, z, 0, 0, 0, rx, ry, rz);
        d = smin(d, sdEllipsoid(x, y, z, lx, ry * 0.15, lz, lr, ry * 0.7, lr), 0.25 * sc);
        d += strataAmp * plateauWave((x * axX + y * axY + z * axZ) * strataFreq + strataPhase);
        return fbm(x, y, z, d);
      };
      const flow: FieldFn = (x, y, z) => {
        const t = (x * axX + y * axY + z * axZ) * strataFreq + strataPhase;
        return t - Math.floor(t);
      };
      return { f, flow, dom: large ? 1.6 : 1.4 };
    }
    case 'pebble': {
      // single ellipsoid + 1-oct fbm.
      const rx = rng.range(0.75, 1.0);
      const ry = rng.range(0.55, 0.8);
      const rz = rng.range(0.7, 0.95);
      const fbm = makeFbm(rng, 1, 0.025);
      return {
        f: (x, y, z) => fbm(x, y, z, sdEllipsoid(x, y, z, 0, 0, 0, rx, ry, rz)),
        flow: FLOW_NONE,
        dom: 1.45,
      };
    }
  }
}

// ---------------------------------------------------------------------------
// §G narrow-band grid sampling: coarse pass at ¼ res, refine 4³ blocks whose
// coarse corners come near the surface, trilinear-fill the rest (far-field
// sign is all the mesher needs there). The refine threshold carries a slack
// term because crack/strata relief locally exceeds Lipschitz-1.

const BORDER_EPS = 1e-3;

function sampleGrid(f: FieldFn, N: number, dom: number): Float64Array {
  const M = N + 1;
  const h = (2 * dom) / N;
  const Nc = N >> 2;
  const Mc = Nc + 1;
  const d = new Float64Array(M * M * M);
  const known = new Uint8Array(M * M * M);
  const cd = new Float64Array(Mc * Mc * Mc);

  // coarse pass (every 4th fine corner)
  for (let ci = 0; ci < Mc; ci++) {
    for (let cj = 0; cj < Mc; cj++) {
      for (let ck = 0; ck < Mc; ck++) {
        const gi = ci << 2;
        const gj = cj << 2;
        const gk = ck << 2;
        const v = f(-dom + gi * h, -dom + gj * h, -dom + gk * h);
        cd[(ci * Mc + cj) * Mc + ck] = v;
        const fi = (gi * M + gj) * M + gk;
        d[fi] = v;
        known[fi] = 1;
      }
    }
  }

  // surface can pass through a block only if some corner has |d| ≤ blockDiag
  // (field ≈ SDF); + slack for the sub-Lipschitz crack/strata relief
  const refineT = 4 * h * Math.sqrt(3) * 1.25 + 0.1;
  for (let ci = 0; ci < Nc; ci++) {
    for (let cj = 0; cj < Nc; cj++) {
      for (let ck = 0; ck < Nc; ck++) {
        const c000 = cd[(ci * Mc + cj) * Mc + ck];
        const c100 = cd[((ci + 1) * Mc + cj) * Mc + ck];
        const c010 = cd[(ci * Mc + cj + 1) * Mc + ck];
        const c110 = cd[((ci + 1) * Mc + cj + 1) * Mc + ck];
        const c001 = cd[(ci * Mc + cj) * Mc + ck + 1];
        const c101 = cd[((ci + 1) * Mc + cj) * Mc + ck + 1];
        const c011 = cd[(ci * Mc + cj + 1) * Mc + ck + 1];
        const c111 = cd[((ci + 1) * Mc + cj + 1) * Mc + ck + 1];
        let m = Math.abs(c000);
        m = Math.min(m, Math.abs(c100), Math.abs(c010), Math.abs(c110));
        m = Math.min(m, Math.abs(c001), Math.abs(c101), Math.abs(c011), Math.abs(c111));
        const refine = m < refineT;
        const bi = ci << 2;
        const bj = cj << 2;
        const bk = ck << 2;
        for (let u = 0; u <= 4; u++) {
          for (let v = 0; v <= 4; v++) {
            for (let w = 0; w <= 4; w++) {
              const fi = ((bi + u) * M + bj + v) * M + bk + w;
              if (known[fi]) continue;
              if (refine) {
                d[fi] = f(-dom + (bi + u) * h, -dom + (bj + v) * h, -dom + (bk + w) * h);
              } else {
                const fu = u * 0.25;
                const fv = v * 0.25;
                const fw = w * 0.25;
                const a00 = c000 + (c100 - c000) * fu;
                const a10 = c010 + (c110 - c010) * fu;
                const a01 = c001 + (c101 - c001) * fu;
                const a11 = c011 + (c111 - c011) * fu;
                const b0 = a00 + (a10 - a00) * fv;
                const b1 = a01 + (a11 - a01) * fv;
                d[fi] = b0 + (b1 - b0) * fw;
              }
              known[fi] = 1;
            }
          }
        }
      }
    }
  }

  // border guarantee: outermost shell ≥ +ε → surface always closed
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < M; j++) {
      for (let k = 0; k < M; k++) {
        if (i > 0 && i < N && j > 0 && j < N && k > 0 && k < N) continue;
        const fi = (i * M + j) * M + k;
        if (d[fi] < BORDER_EPS) d[fi] = BORDER_EPS;
      }
    }
  }
  return d;
}

// ---------------------------------------------------------------------------
// §D naive Surface Nets

/** cell-corner offsets (x,y,z) and the 12 cell edges as corner-index pairs */
const CORNER = [
  [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
  [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
] as const;
const CELL_EDGE = [
  [0, 1], [2, 3], [4, 5], [6, 7],
  [0, 2], [1, 3], [4, 6], [5, 7],
  [0, 4], [1, 5], [2, 6], [3, 7],
] as const;

interface NetsMesh {
  vx: number[];
  vy: number[];
  vz: number[];
  indices: number[];
  quads: number;
  droppedTris: number;
}

function surfaceNets(d: Float64Array, N: number, dom: number): NetsMesh {
  const M = N + 1;
  const h = (2 * dom) / N;
  const cellVert = new Int32Array(N * N * N).fill(-1);
  const vx: number[] = [];
  const vy: number[] = [];
  const vz: number[] = [];
  const corner = new Float64Array(8);

  // pass 1: one vertex per sign-change cell at the edge-crossing centroid
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      for (let k = 0; k < N; k++) {
        let inside = 0;
        for (let c = 0; c < 8; c++) {
          const o = CORNER[c] as readonly [number, number, number];
          const v = d[((i + o[0]) * M + j + o[1]) * M + k + o[2]];
          corner[c] = v;
          if (v < 0) inside++;
        }
        if (inside === 0 || inside === 8) continue;
        let sx = 0;
        let sy = 0;
        let sz = 0;
        let cnt = 0;
        for (let e = 0; e < 12; e++) {
          const pair = CELL_EDGE[e] as readonly [number, number];
          const d0 = corner[pair[0]];
          const d1 = corner[pair[1]];
          if (d0 < 0 === d1 < 0) continue;
          const t = d0 / (d0 - d1);
          const o0 = CORNER[pair[0]] as readonly [number, number, number];
          const o1 = CORNER[pair[1]] as readonly [number, number, number];
          sx += o0[0] + (o1[0] - o0[0]) * t;
          sy += o0[1] + (o1[1] - o0[1]) * t;
          sz += o0[2] + (o1[2] - o0[2]) * t;
          cnt++;
        }
        cellVert[(i * N + j) * N + k] = vx.length;
        vx.push(-dom + (i + sx / cnt) * h);
        vy.push(-dom + (j + sy / cnt) * h);
        vz.push(-dom + (k + sz / cnt) * h);
      }
    }
  }

  // pass 2: a quad around every sign-change grid edge, wound so the face
  // normal follows the field gradient (inside → outside). Border clamp
  // guarantees the 4 adjacent cells exist for every crossing edge.
  const indices: number[] = [];
  let quads = 0;
  let droppedTris = 0;
  const quad = new Int32Array(4);

  const emitTri = (a: number, b: number, c: number): void => {
    const e1x = vx[b] - vx[a];
    const e1y = vy[b] - vy[a];
    const e1z = vz[b] - vz[a];
    const e2x = vx[c] - vx[a];
    const e2y = vy[c] - vy[a];
    const e2z = vz[c] - vz[a];
    const fx = e1y * e2z - e1z * e2y;
    const fy = e1z * e2x - e1x * e2z;
    const fz = e1x * e2y - e1y * e2x;
    if (fx * fx + fy * fy + fz * fz < 1e-28) {
      droppedTris++;
      return;
    }
    indices.push(a, b, c);
  };

  for (let axis = 0; axis < 3; axis++) {
    const b = (axis + 1) % 3;
    const c = (axis + 2) % 3;
    const lim = [0, 0, 0];
    lim[axis] = N; // edge start along the axis: 0..N-1
    lim[b] = N; // transverse: 1..N-1
    lim[c] = N;
    const gs = [M * M, M, 1] as const; // corner strides (x-major)
    const cs = [N * N, N, 1] as const; // cell strides
    const p = [0, 0, 0];
    for (p[axis] = 0; p[axis] < lim[axis]; p[axis]++) {
      for (p[b] = 1; p[b] < lim[b]; p[b]++) {
        for (p[c] = 1; p[c] < lim[c]; p[c]++) {
          const gi = p[0] * gs[0] + p[1] * gs[1] + p[2] * gs[2];
          const d0 = d[gi];
          const d1 = d[gi + gs[axis]];
          if (d0 < 0 === d1 < 0) continue;
          // 4 cells around the edge, CCW as seen from +axis
          const cellBase = p[0] * cs[0] + p[1] * cs[1] + p[2] * cs[2];
          quad[0] = cellVert[cellBase - cs[b] - cs[c]];
          quad[1] = cellVert[cellBase - cs[c]];
          quad[2] = cellVert[cellBase];
          quad[3] = cellVert[cellBase - cs[b]];
          if (quad[0] < 0 || quad[1] < 0 || quad[2] < 0 || quad[3] < 0) {
            throw new Error('RockGen: crossing edge with missing cell vertex');
          }
          if (d0 >= 0) {
            // gradient points −axis: reverse the loop
            const t = quad[1];
            quad[1] = quad[3];
            quad[3] = t;
          }
          // split on the shorter diagonal (better-shaped, deterministic)
          const q0 = quad[0];
          const q1 = quad[1];
          const q2 = quad[2];
          const q3 = quad[3];
          const dax = vx[q0] - vx[q2];
          const day = vy[q0] - vy[q2];
          const daz = vz[q0] - vz[q2];
          const dbx = vx[q1] - vx[q3];
          const dby = vy[q1] - vy[q3];
          const dbz = vz[q1] - vz[q3];
          if (dax * dax + day * day + daz * daz <= dbx * dbx + dby * dby + dbz * dbz) {
            emitTri(q0, q1, q2);
            emitTri(q0, q2, q3);
          } else {
            emitTri(q1, q2, q3);
            emitTri(q1, q3, q0);
          }
          quads++;
        }
      }
    }
  }
  return { vx, vy, vz, indices, quads, droppedTris };
}

// ---------------------------------------------------------------------------
// validation (§D): open-address edge audit mirroring Clusterize.ts:44-81 —
// every undirected edge must be used by exactly 2 tris, once per direction
// (manifold + consistently wound). Naive-Surface-Nets PINCH edges (the same
// vertex pair connected through TWO sign-change grid edges at thin sub-cell
// features → count 4, both directions, still closed) are tolerated under a
// 1% cap and reported — §D: Clusterize retires 3rd+ tris on an edge and
// BuildDag position-welds, so they are handled downstream. Open / odd /
// same-direction edges are REAL failures (holes, flipped winding) → throw.
// Plus a positive-enclosed-volume check (normals outward) and the 65k cap.

function auditManifold(indices: number[], vx: number[], vy: number[], vz: number[]): number {
  const triCount = indices.length / 3;
  const edgeCount = triCount * 3;
  let cap = 1;
  while (cap < edgeCount * 2) cap <<= 1;
  const mask = cap - 1;
  const keyLo = new Uint32Array(cap);
  const keyHi = new Uint32Array(cap);
  const count = new Int32Array(cap);
  const dirs = new Uint8Array(cap);

  for (let t = 0; t < triCount; t++) {
    for (let e = 0; e < 3; e++) {
      const a = indices[t * 3 + e];
      const b = indices[t * 3 + ((e + 1) % 3)];
      const lo = a < b ? a : b;
      const hi = a < b ? b : a;
      let h = (lo * 0x85ebca6b) ^ (hi * 0xc2b2ae35);
      h = (h ^ (h >>> 13)) >>> 0;
      let s = h & mask;
      for (;;) {
        if (count[s] === 0) {
          keyLo[s] = lo;
          keyHi[s] = hi;
          count[s] = 1;
          dirs[s] = a < b ? 1 : 2;
          break;
        }
        if (keyLo[s] === lo && keyHi[s] === hi) {
          count[s]++;
          dirs[s] |= a < b ? 1 : 2;
          break;
        }
        s = (s + 1) & mask;
      }
    }
  }
  let open = 0;
  let bad = 0;
  let pinch = 0;
  let edges = 0;
  let example = '';
  for (let s = 0; s < cap; s++) {
    const n = count[s];
    if (n === 0) continue;
    edges++;
    if (n === 2 && dirs[s] === 3) continue;
    if (n % 2 === 0 && dirs[s] === 3) {
      pinch++;
      continue;
    }
    if (n === 1) open++;
    else bad++;
    if (example === '') example = `edge ${keyLo[s]}-${keyHi[s]} count ${n} dirs ${dirs[s]}`;
  }
  if (open + bad > 0) {
    throw new Error(
      `RockGen: manifold audit FAILED — ${open} open edges, ${bad} bad-share/winding ` +
        `(first: ${example})`,
    );
  }
  if (pinch > edges * 0.01) {
    throw new Error(`RockGen: pinch edges ${pinch}/${edges} exceed the 1% tolerance cap`);
  }
  // orientation: enclosed volume must be positive (outward CCW winding)
  let vol6 = 0;
  for (let t = 0; t < triCount; t++) {
    const a = indices[t * 3];
    const b = indices[t * 3 + 1];
    const c = indices[t * 3 + 2];
    vol6 +=
      vx[a] * (vy[b] * vz[c] - vz[b] * vy[c]) -
      vy[a] * (vx[b] * vz[c] - vz[b] * vx[c]) +
      vz[a] * (vx[b] * vy[c] - vy[b] * vx[c]);
  }
  if (vol6 <= 0) throw new Error(`RockGen: orientation audit FAILED — enclosed volume ${vol6 / 6}`);
  return pinch;
}

// ---------------------------------------------------------------------------
// per-vertex bakes (§C): normal = field gradient; vdata x curvature (6-tap
// Laplacian remap), y flow, z AO×upness (occlusion taps along +n), w cavity AO.

/** curvature remap half-range, 1/r units (radius-of-curvature 1/CURV_C saturates) */
const CURV_C = 8;
const AO_TAPS = [0.15, 0.4, 1.0] as const;
const CAVITY_TAPS = [0.05, 0.11, 0.22] as const;

function bakeVertices(
  f: FieldFn, flow: FieldFn,
  vx: number[], vy: number[], vz: number[], h: number,
): { normals: Float32Array; vdata: Uint8Array } {
  const n = vx.length;
  const normals = new Float32Array(n * 3);
  const vdata = new Uint8Array(n * 4);
  const eps = h * 0.6;
  for (let i = 0; i < n; i++) {
    const x = vx[i];
    const y = vy[i];
    const z = vz[i];
    const c = f(x, y, z);
    const dpx = f(x + eps, y, z);
    const dmx = f(x - eps, y, z);
    const dpy = f(x, y + eps, z);
    const dmy = f(x, y - eps, z);
    const dpz = f(x, y, z + eps);
    const dmz = f(x, y, z - eps);
    let nx = dpx - dmx;
    let ny = dpy - dmy;
    let nz = dpz - dmz;
    const nl = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (nl > 1e-12) {
      nx /= nl;
      ny /= nl;
      nz /= nl;
    } else {
      nx = 0;
      ny = 1;
      nz = 0;
    }
    normals[i * 3] = nx;
    normals[i * 3 + 1] = ny;
    normals[i * 3 + 2] = nz;

    // x: mean-curvature proxy — SDF Laplacian is + on convex, − in hollows
    const lap = (dpx + dmx + dpy + dmy + dpz + dmz - 6 * c) / (eps * eps);
    const curv = clamp01(0.5 + lap / (2 * CURV_C));

    // z/w: field-occlusion ratio d(p+n·t)/t — 1 on open convex, <1 in hollows
    let ao = 0;
    for (const t of AO_TAPS) ao += clamp01(f(x + nx * t, y + ny * t, z + nz * t) / t);
    ao /= AO_TAPS.length;
    let cav = 0;
    for (const t of CAVITY_TAPS) cav += clamp01(f(x + nx * t, y + ny * t, z + nz * t) / t);
    cav /= CAVITY_TAPS.length;

    vdata[i * 4] = Math.round(curv * 255);
    vdata[i * 4 + 1] = Math.round(clamp01(flow(x, y, z)) * 255);
    vdata[i * 4 + 2] = Math.round(ao * Math.max(ny, 0) * 255);
    vdata[i * 4 + 3] = Math.round(cav * 255);
  }
  return { normals, vdata };
}

// ---------------------------------------------------------------------------
// public API

export function generateRock(
  archetype: RockArchetype,
  variant: number,
  seed: number,
  gridRes: number,
  mod?: RockMod,
  domainScale = 1,
): RockMesh {
  if (gridRes < 16 || gridRes % 4 !== 0) {
    throw new Error(`RockGen: gridRes must be ≥16 and divisible by 4 (got ${gridRes})`);
  }
  const t0 = performance.now();
  const rng = new WorldSeed(seed >>> 0).rng(`rock/${archetype}${mod ? ':' + mod : ''}/${variant}`);
  const def = makeField(archetype, mod, rng);
  let fieldEvals = 0;
  const f: FieldFn = (x, y, z) => {
    fieldEvals++;
    return def.f(x, y, z);
  };
  const dom = def.dom * domainScale;
  const N = gridRes;

  const grid = sampleGrid(f, N, dom);
  const t1 = performance.now();

  const nets = surfaceNets(grid, N, dom);
  const verts = nets.vx.length;
  if (verts === 0) throw new Error(`RockGen: ${archetype}/${variant} produced an empty field`);
  if (verts >= 65536) {
    throw new Error(`RockGen: ${archetype}/${variant} at ${N}³ → ${verts} verts (cap 65k)`);
  }
  const pinchEdges = auditManifold(nets.indices, nets.vx, nets.vy, nets.vz);
  const t2 = performance.now();

  const baked = bakeVertices(f, def.flow, nets.vx, nets.vy, nets.vz, (2 * dom) / N);
  const t3 = performance.now();

  const positions = new Float32Array(verts * 3);
  let boundR2 = 0;
  for (let i = 0; i < verts; i++) {
    const x = nets.vx[i];
    const y = nets.vy[i];
    const z = nets.vz[i];
    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;
    const r2 = x * x + y * y + z * z;
    if (r2 > boundR2) boundR2 = r2;
  }
  return {
    positions,
    normals: baked.normals,
    vdata: baked.vdata,
    indices: new Uint32Array(nets.indices),
    stats: {
      verts,
      tris: nets.indices.length / 3,
      quads: nets.quads,
      droppedTris: nets.droppedTris,
      pinchEdges,
      fieldEvals,
      boundRadius: Math.sqrt(boundR2),
      gridMs: t1 - t0,
      meshMs: t2 - t1,
      bakeMs: t3 - t2,
      totalMs: performance.now() - t0,
    },
  };
}

// ---------------------------------------------------------------------------
// §B library table. classId literals MUST mirror the Scatter.ts VegClass enum
// (RockGen is a zero-three.js-imports worker module, so it can't import the enum;
// the reserved-block layout keeps these stable): Boulder 25 / Slab 26 / StoneL 27 /
// StoneM 28 / StoneS 29 (extras+stones tail). EtakErratic is the ETAK-only hero head
// — class id 31 (the erratic slot after the reserved tree/understory/extras/stones
// blocks). It has NO scatter kernel: instances come from ETAK boulder records at R3
// (0 until then). Variant semantics preserved: v0/1 = pale/talus context, v2/3 =
// dark/mossy context. domainScale is the per-entry tri-target tuning lever (same
// archetype serves multiple classes at different grid res).

/** VegClass id of the ETAK-only hero-erratic head (Scatter.ts reserved-block erratic slot) */
export const ETAK_ERRATIC_CLASS = 31;

export interface RockVariantSpec {
  archetype: RockArchetype;
  mod?: RockMod;
  domainScale?: number;
}

export interface RockClassSpec {
  name: string;
  /** VegClass id — mirrors the Scatter.ts enum (Boulder/Slab/StoneL/StoneM/StoneS =
   *  25–29; ETAK_ERRATIC_CLASS = 31 for the hero head) */
  classId: number;
  gridRes: number;
  /** LOD0 tri target per variant (SPEC-ROCKS §B, gate ±25%) */
  targetTris: number;
  variants: RockVariantSpec[];
}

export const ROCK_LIBRARY: RockClassSpec[] = [
  {
    name: 'Boulder', classId: 25, gridRes: 64, targetTris: 14000,
    variants: [
      { archetype: 'graniteErratic', domainScale: 0.93 },
      { archetype: 'graniteErratic', domainScale: 0.93 },
      { archetype: 'fieldstone' },
      { archetype: 'fieldstone' },
    ],
  },
  {
    name: 'Slab', classId: 26, gridRes: 64, targetTris: 12000,
    variants: [
      { archetype: 'flatCobble', mod: 'large' },
      { archetype: 'flatCobble', mod: 'large' },
      { archetype: 'graniteErratic', mod: 'low', domainScale: 0.9 },
      { archetype: 'graniteErratic', mod: 'low', domainScale: 0.9 },
    ],
  },
  {
    name: 'StoneL', classId: 27, gridRes: 48, targetTris: 7000,
    variants: [
      { archetype: 'angularShard', domainScale: 1.15 },
      { archetype: 'angularShard', domainScale: 1.15 },
      { archetype: 'fieldstone', domainScale: 1.06 },
      { archetype: 'fieldstone', domainScale: 1.06 },
    ],
  },
  {
    name: 'StoneM', classId: 28, gridRes: 32, targetTris: 2500,
    variants: [
      { archetype: 'flatCobble' },
      { archetype: 'flatCobble' },
      { archetype: 'pebble' },
      { archetype: 'pebble' },
    ],
  },
  {
    name: 'StoneS', classId: 29, gridRes: 24, targetTris: 1000,
    variants: [
      { archetype: 'pebble', domainScale: 1.22 },
      { archetype: 'pebble', domainScale: 1.22 },
      { archetype: 'pebble', domainScale: 1.22 },
      { archetype: 'pebble', domainScale: 1.22 },
    ],
  },
  {
    name: 'EtakErratic', classId: ETAK_ERRATIC_CLASS, gridRes: 96, targetTris: 30000,
    variants: [
      { archetype: 'graniteErratic', mod: 'hero' },
      { archetype: 'graniteErratic', mod: 'hero' },
    ],
  },
];
