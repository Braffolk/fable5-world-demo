/**
 * Nanite LOD DAG build (NANITE.md N8-D0) — hand-rolled, CPU/typed-array,
 * node-runnable (no three.js, no GPU) for the validation probe (tools/probe-dag.ts).
 *
 * Given LOD0 clusters (Clusterize.ts), build a continuous-LOD cluster DAG:
 * per level k → k+1, group adjacent clusters, lock the group boundary,
 * QEM-simplify the interior to ~half the triangles, re-clusterize the result
 * into the PARENTS of the whole group, and record the crack-free runtime-cut
 * metadata — per-cluster (ownError, ownSphere) and (parentError, parentSphere)
 * pairs with containment, max-monotonicity, and EXACT sibling-pair equality
 * (the boundary of any screen-error cut always falls between groups, where
 * vertices were locked → no cracks; jglrxavpok / zeux meshoptimizer #750).
 *
 * Deviations from the implementation-ready spec, documented honestly:
 *  - Group partition is SPATIAL median bisection of cluster centres (compact
 *    groups ⇒ short locked boundary ⇒ good reduction), a deterministic METIS
 *    substitute. Crack-freeness does NOT depend on the partitioner (it falls
 *    out of boundary-locking + shared parent pairs for ANY partition); only the
 *    reduction EFFICIENCY does. Adjacency-graph boundary-min refinement is a
 *    future quality lever if a level's reduction is poor.
 *  - QEM is POSITION-driven (Garland-Heckbert area-weighted quadrics); per-vertex
 *    attributes (normal/uv/vdata) ride along by linear interpolation on each
 *    collapse (normal re-normalised). Full attribute-space quadrics are a later
 *    quality lever. Error is reported as an approximate object-space RMS
 *    deviation in metres (quadric value / accumulated area, square-rooted) —
 *    consistent and monotonic; absolute τ calibration is a D1 concern.
 *  - Position-WELD per group (not id-weld): coincident-but-split vertices merge,
 *    which is what lets a locked boundary re-simplify once the NEXT level
 *    re-groups across it (the locked verts are bit-identical in both groups, so
 *    they weld). Un-welded seams are the #1 "simplification stuck" cause.
 *
 * The build is RNG-free → deterministic by construction (given input geometry).
 */

import { type BuiltClusters, clusterize } from './Clusterize';

export interface DagOpts {
  /** cluster triangle cap (matches Clusterize), default 128 */
  maxTris?: number;
  /** simplify each group's interior to this fraction of its triangles, default 0.5 */
  targetRatio?: number;
  /** max clusters per group (recursive bisection stops at ≤ this), default 24 */
  groupMax?: number;
  /** a group that reduces fewer than this fraction of its tris is "stuck" → its
   *  clusters become roots, default 0.15 */
  stuckFrac?: number;
  /** safety cap on DAG levels, default 24 */
  maxLevels?: number;
  /** float offset of the normal within the vertex record (renormalised after
   *  interpolation); -1 = no normal to renormalise, default -1 */
  normalOffset?: number;
  /** position weld grid (world units), default 1e-5 */
  weldEps?: number;
}

/** one cluster at one LOD level, across the whole DAG (LOD0 first) */
export interface DagCluster {
  level: number;
  /** triangle range into DagBuild.indices */
  triStart: number;
  triCount: number;
  /** geometric bounding sphere (xyzr) + backface cone (axis xyz + cos) — culling */
  sx: number;
  sy: number;
  sz: number;
  sr: number;
  cax: number;
  cay: number;
  caz: number;
  ccos: number;
  /** DAG runtime-cut metadata: this cluster's OWN simplification error + sphere */
  ownError: number;
  oex: number;
  oey: number;
  oez: number;
  oer: number;
  /** the error+sphere at which this cluster is REPLACED by its parents
   *  (parentError = +∞ ⇒ this is a root) */
  parentError: number;
  pex: number;
  pey: number;
  pez: number;
  per: number;
  /** group that consumed this cluster as input (defines its parent pair); -1 = root */
  groupAsInput: number;
  /** group that produced this cluster as a parent; -1 = LOD0 */
  groupAsParent: number;
}

/** a group: input clusters (level k) → simplified → parent clusters (level k+1),
 *  all sharing one (error, sphere) pair bit-for-bit */
export interface DagGroup {
  id: number;
  level: number;
  error: number;
  sx: number;
  sy: number;
  sz: number;
  sr: number;
  inputs: number[];
  parents: number[];
  /** false ⇒ stuck (no parents; inputs became roots) */
  reduced: boolean;
}

export interface DagLevelStats {
  level: number;
  inClusters: number;
  inTris: number;
  outClusters: number;
  outTris: number;
  groups: number;
  stuckGroups: number;
  /** fraction of triangles removed this level (1 - outTris/inTris over non-stuck) */
  triReduction: number;
}

export interface DagStats {
  buildMs: number;
  levels: number;
  lod0Clusters: number;
  lod0Tris: number;
  totalClusters: number;
  totalTris: number;
  roots: number;
  maxError: number;
}

export interface DagBuild {
  /** interleaved vertex pool (vertStride floats/vertex; position at offset 0) */
  verts: Float32Array;
  vertStride: number;
  /** global triangle indices into `verts`; cluster c owns [triStart, +triCount) */
  indices: Uint32Array;
  clusters: DagCluster[];
  groups: DagGroup[];
  levelStats: DagLevelStats[];
  /** count of LOD0 clusters (clusters[0 .. lod0Count) are level 0) */
  lod0Count: number;
  stats: DagStats;
}

// ---------------------------------------------------------------------------
// quadric math (Garland-Heckbert): 10 floats per symmetric 4×4
//   [ q0 q1 q2 q3 ]
//   [ q1 q4 q5 q6 ]
//   [ q2 q5 q7 q8 ]
//   [ q3 q6 q8 q9 ]
// ---------------------------------------------------------------------------

function addPlane(Q: Float64Array, base: number, a: number, b: number, c: number, d: number, w: number): void {
  Q[base] += w * a * a;
  Q[base + 1] += w * a * b;
  Q[base + 2] += w * a * c;
  Q[base + 3] += w * a * d;
  Q[base + 4] += w * b * b;
  Q[base + 5] += w * b * c;
  Q[base + 6] += w * b * d;
  Q[base + 7] += w * c * c;
  Q[base + 8] += w * c * d;
  Q[base + 9] += w * d * d;
}

function addQuadric(Q: Float64Array, dst: number, src: number): void {
  for (let k = 0; k < 10; k++) Q[dst + k] += Q[src + k] as number;
}

/** vᵀ Q v for v = (x,y,z,1) */
function evalQuadric(Q: Float64Array, base: number, x: number, y: number, z: number): number {
  const q0 = Q[base] as number;
  const q1 = Q[base + 1] as number;
  const q2 = Q[base + 2] as number;
  const q3 = Q[base + 3] as number;
  const q4 = Q[base + 4] as number;
  const q5 = Q[base + 5] as number;
  const q6 = Q[base + 6] as number;
  const q7 = Q[base + 7] as number;
  const q8 = Q[base + 8] as number;
  const q9 = Q[base + 9] as number;
  return (
    q0 * x * x +
    2 * q1 * x * y +
    2 * q2 * x * z +
    2 * q3 * x +
    q4 * y * y +
    2 * q5 * y * z +
    2 * q6 * y +
    q7 * z * z +
    2 * q8 * z +
    q9
  );
}

/** minimise vᵀ Q v over (x,y,z): solve the 3×3 [A]p = -[b]; null if near-singular */
function solveQuadric(Q: Float64Array, base: number): [number, number, number] | null {
  const a00 = Q[base] as number;
  const a01 = Q[base + 1] as number;
  const a02 = Q[base + 2] as number;
  const a11 = Q[base + 4] as number;
  const a12 = Q[base + 5] as number;
  const a22 = Q[base + 7] as number;
  const bx = -(Q[base + 3] as number);
  const by = -(Q[base + 6] as number);
  const bz = -(Q[base + 8] as number);
  // det of the symmetric 3×3
  const det =
    a00 * (a11 * a22 - a12 * a12) - a01 * (a01 * a22 - a12 * a02) + a02 * (a01 * a12 - a11 * a02);
  // scale-relative singularity guard
  const scale = Math.abs(a00) + Math.abs(a11) + Math.abs(a22) + 1e-20;
  if (Math.abs(det) < 1e-10 * scale * scale * scale) return null;
  const inv = 1 / det;
  // adjugate · b
  const x = inv * (bx * (a11 * a22 - a12 * a12) - by * (a01 * a22 - a02 * a12) + bz * (a01 * a12 - a02 * a11));
  const y = inv * (-bx * (a01 * a22 - a12 * a02) + by * (a00 * a22 - a02 * a02) - bz * (a00 * a12 - a01 * a02));
  const z = inv * (bx * (a01 * a12 - a11 * a02) - by * (a00 * a12 - a01 * a02) + bz * (a00 * a11 - a01 * a01));
  return [x, y, z];
}

// ---------------------------------------------------------------------------
// bounding-sphere combine (fold) — containment guaranteed, minimality not
// ---------------------------------------------------------------------------

interface Sphere {
  x: number;
  y: number;
  z: number;
  r: number;
}

/** smallest-ish sphere that provably contains both s0 and s1 */
function mergeSpheres(s0: Sphere, s1: Sphere): Sphere {
  const dx = s1.x - s0.x;
  const dy = s1.y - s0.y;
  const dz = s1.z - s0.z;
  const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (d + s1.r <= s0.r + 1e-9) return { x: s0.x, y: s0.y, z: s0.z, r: s0.r }; // s0 ⊇ s1
  if (d + s0.r <= s1.r + 1e-9) return { x: s1.x, y: s1.y, z: s1.z, r: s1.r }; // s1 ⊇ s0
  const r = (s0.r + s1.r + d) * 0.5;
  const t = d > 1e-12 ? (r - s0.r) / d : 0;
  return { x: s0.x + dx * t, y: s0.y + dy * t, z: s0.z + dz * t, r };
}

// ---------------------------------------------------------------------------
// spatial median group partition (recursive bisection)
// ---------------------------------------------------------------------------

/** split a set of cluster ids into groups of ≤ groupMax, by recursive
 *  longest-axis median bisection of their sphere centres (deterministic) */
function partitionClusters(ids: number[], clusters: DagCluster[], groupMax: number): number[][] {
  if (ids.length <= groupMax) return [ids];
  // bbox of centres
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (const id of ids) {
    const c = clusters[id] as DagCluster;
    if (c.sx < minX) minX = c.sx;
    if (c.sy < minY) minY = c.sy;
    if (c.sz < minZ) minZ = c.sz;
    if (c.sx > maxX) maxX = c.sx;
    if (c.sy > maxY) maxY = c.sy;
    if (c.sz > maxZ) maxZ = c.sz;
  }
  const ex = maxX - minX;
  const ey = maxY - minY;
  const ez = maxZ - minZ;
  const axis = ex >= ey && ex >= ez ? 0 : ey >= ez ? 1 : 2;
  const key = (id: number): number => {
    const c = clusters[id] as DagCluster;
    return axis === 0 ? c.sx : axis === 1 ? c.sy : c.sz;
  };
  // sort by axis, tie-break by id for determinism
  const sorted = ids.slice().sort((a, b) => {
    const ka = key(a);
    const kb = key(b);
    return ka !== kb ? ka - kb : a - b;
  });
  const mid = sorted.length >> 1;
  const left = sorted.slice(0, mid);
  const right = sorted.slice(mid);
  return partitionClusters(left, clusters, groupMax).concat(partitionClusters(right, clusters, groupMax));
}

// ---------------------------------------------------------------------------
// per-group simplification
// ---------------------------------------------------------------------------

interface SimplifyResult {
  /** compacted interleaved vertices (vertStride floats each) */
  verts: Float32Array;
  /** 0-based triangle indices into `verts` */
  indices: Uint32Array;
  soupTris: number;
  simpTris: number;
  /** approximate object-space RMS error (metres) introduced this group */
  qemErr: number;
}

/** min-heap entry for a candidate collapse a→keep (other removed) */
interface Collapse {
  cost: number;
  a: number;
  b: number;
  va: number;
  vb: number;
  keep: number; // survivor id (a or b)
  tx: number;
  ty: number;
  tz: number;
}

function heapPush(h: Collapse[], e: Collapse): void {
  h.push(e);
  let i = h.length - 1;
  while (i > 0) {
    const p = (i - 1) >> 1;
    const hp = h[p] as Collapse;
    const hi = h[i] as Collapse;
    if (hp.cost <= hi.cost) break;
    h[p] = hi;
    h[i] = hp;
    i = p;
  }
}

function heapPop(h: Collapse[]): Collapse {
  const top = h[0] as Collapse;
  const last = h.pop() as Collapse;
  if (h.length > 0) {
    h[0] = last;
    let i = 0;
    for (;;) {
      const l = i * 2 + 1;
      const r = l + 1;
      let m = i;
      if (l < h.length && (h[l] as Collapse).cost < (h[m] as Collapse).cost) m = l;
      if (r < h.length && (h[r] as Collapse).cost < (h[m] as Collapse).cost) m = r;
      if (m === i) break;
      const hm = h[m] as Collapse;
      h[m] = h[i] as Collapse;
      h[i] = hm;
      i = m;
    }
  }
  return top;
}

/**
 * Weld the group's clusters into one soup, lock the group boundary, and
 * QEM-collapse the interior down to ~targetRatio of its triangles.
 */
function simplifyGroup(
  pool: ArrayLike<number>,
  poolIdx: ArrayLike<number>,
  stride: number,
  groupClusters: DagCluster[],
  opts: Required<DagOpts>,
): SimplifyResult {
  // ---- weld group triangles into a local soup (by quantised position) -------
  //  spatial-hash buckets (qx,qy,qz,localId quads) — no string keys. Locked
  //  boundary verts are copied bit-identical across groups ⇒ identical quants ⇒
  //  they re-weld at the next level (the mechanism that re-simplifies boundaries).
  const inv = 1 / opts.weldEps;
  const buckets = new Map<number, number[]>();
  const localPos: number[] = []; // interleaved, stride floats
  const triLocal: number[] = []; // local id triples
  const localOf = (vid: number): number => {
    const p = vid * stride;
    const qx = Math.round((pool[p] as number) * inv);
    const qy = Math.round((pool[p + 1] as number) * inv);
    const qz = Math.round((pool[p + 2] as number) * inv);
    const h = (Math.imul(qx, 73856093) ^ Math.imul(qy, 19349663) ^ Math.imul(qz, 83492791)) | 0;
    let bucket = buckets.get(h);
    if (bucket) {
      for (let i = 0; i < bucket.length; i += 4) {
        if (bucket[i] === qx && bucket[i + 1] === qy && bucket[i + 2] === qz) return bucket[i + 3] as number;
      }
    } else {
      bucket = [];
      buckets.set(h, bucket);
    }
    const id = localPos.length / stride;
    for (let s = 0; s < stride; s++) localPos.push(pool[p + s] as number);
    bucket.push(qx, qy, qz, id);
    return id;
  };
  let soupTris = 0;
  for (const c of groupClusters) {
    for (let t = c.triStart; t < c.triStart + c.triCount; t++) {
      const a = localOf(poolIdx[t * 3] as number);
      const b = localOf(poolIdx[t * 3 + 1] as number);
      const cc = localOf(poolIdx[t * 3 + 2] as number);
      if (a === b || b === cc || a === cc) continue; // welded-degenerate
      triLocal.push(a, b, cc);
      soupTris++;
    }
  }
  const nv = localPos.length / stride;
  const verts = new Float64Array(localPos.length);
  for (let i = 0; i < localPos.length; i++) verts[i] = localPos[i] as number;

  // ---- triangle + incidence structures --------------------------------------
  const triA = new Int32Array(soupTris);
  const triB = new Int32Array(soupTris);
  const triC = new Int32Array(soupTris);
  const triAlive = new Uint8Array(soupTris).fill(1);
  for (let t = 0; t < soupTris; t++) {
    triA[t] = triLocal[t * 3] as number;
    triB[t] = triLocal[t * 3 + 1] as number;
    triC[t] = triLocal[t * 3 + 2] as number;
  }
  const vtri: number[][] = Array.from({ length: nv }, () => []);
  for (let t = 0; t < soupTris; t++) {
    (vtri[triA[t] as number] as number[]).push(t);
    (vtri[triB[t] as number] as number[]).push(t);
    (vtri[triC[t] as number] as number[]).push(t);
  }

  // ---- lock the GROUP boundary: edges used by ≠2 soup triangles --------------
  //  intra-group edge use-count == 1 ⇔ shared with another group OR open mesh
  //  boundary; either way it must not move (crack-free / no silhouette shrink).
  const edgeUse = new Map<number, number>();
  const ekey = (u: number, v: number): number => (u < v ? u * nv + v : v * nv + u);
  for (let t = 0; t < soupTris; t++) {
    const a = triA[t] as number;
    const b = triB[t] as number;
    const c = triC[t] as number;
    edgeUse.set(ekey(a, b), (edgeUse.get(ekey(a, b)) ?? 0) + 1);
    edgeUse.set(ekey(b, c), (edgeUse.get(ekey(b, c)) ?? 0) + 1);
    edgeUse.set(ekey(c, a), (edgeUse.get(ekey(c, a)) ?? 0) + 1);
  }
  const locked = new Uint8Array(nv);
  for (let t = 0; t < soupTris; t++) {
    const a = triA[t] as number;
    const b = triB[t] as number;
    const c = triC[t] as number;
    const pairs: [number, number][] = [
      [a, b],
      [b, c],
      [c, a],
    ];
    for (const [u, v] of pairs) {
      if ((edgeUse.get(ekey(u, v)) as number) !== 2) {
        locked[u] = 1;
        locked[v] = 1;
      }
    }
  }

  // ---- per-vertex quadrics (area-weighted) + accumulated area ---------------
  const Q = new Float64Array(nv * 10);
  const warea = new Float64Array(nv);
  for (let t = 0; t < soupTris; t++) {
    const a = triA[t] as number;
    const b = triB[t] as number;
    const c = triC[t] as number;
    const ax = verts[a * stride] as number;
    const ay = verts[a * stride + 1] as number;
    const az = verts[a * stride + 2] as number;
    const e1x = (verts[b * stride] as number) - ax;
    const e1y = (verts[b * stride + 1] as number) - ay;
    const e1z = (verts[b * stride + 2] as number) - az;
    const e2x = (verts[c * stride] as number) - ax;
    const e2y = (verts[c * stride + 1] as number) - ay;
    const e2z = (verts[c * stride + 2] as number) - az;
    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * e2x - e1x * e2z;
    let nz = e1x * e2y - e1y * e2x;
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (len < 1e-20) continue;
    const area = 0.5 * len;
    nx /= len;
    ny /= len;
    nz /= len;
    const d = -(nx * ax + ny * ay + nz * az);
    addPlane(Q, a * 10, nx, ny, nz, d, area);
    addPlane(Q, b * 10, nx, ny, nz, d, area);
    addPlane(Q, c * 10, nx, ny, nz, d, area);
    warea[a] = (warea[a] as number) + area;
    warea[b] = (warea[b] as number) + area;
    warea[c] = (warea[c] as number) + area;
  }

  // ---- candidate-collapse cost + optimal target -----------------------------
  const version = new Int32Array(nv);
  const vAlive = new Uint8Array(nv).fill(1);
  const tmp = new Float64Array(10);
  const evalCollapse = (a: number, b: number): Collapse | null => {
    if (locked[a] === 1 && locked[b] === 1) return null; // both fixed: cannot collapse
    for (let k = 0; k < 10; k++) tmp[k] = (Q[a * 10 + k] as number) + (Q[b * 10 + k] as number);
    let keep = a;
    let tx: number;
    let ty: number;
    let tz: number;
    if (locked[a] === 1) {
      keep = a;
      tx = verts[a * stride] as number;
      ty = verts[a * stride + 1] as number;
      tz = verts[a * stride + 2] as number;
    } else if (locked[b] === 1) {
      keep = b;
      tx = verts[b * stride] as number;
      ty = verts[b * stride + 1] as number;
      tz = verts[b * stride + 2] as number;
    } else {
      const opt = solveQuadric(tmp, 0);
      if (opt) {
        tx = opt[0];
        ty = opt[1];
        tz = opt[2];
      } else {
        // singular: best of a, b, midpoint
        const cand: [number, number, number][] = [
          [verts[a * stride] as number, verts[a * stride + 1] as number, verts[a * stride + 2] as number],
          [verts[b * stride] as number, verts[b * stride + 1] as number, verts[b * stride + 2] as number],
          [
            ((verts[a * stride] as number) + (verts[b * stride] as number)) * 0.5,
            ((verts[a * stride + 1] as number) + (verts[b * stride + 1] as number)) * 0.5,
            ((verts[a * stride + 2] as number) + (verts[b * stride + 2] as number)) * 0.5,
          ],
        ];
        let best = 0;
        let bestC = Infinity;
        for (let i = 0; i < cand.length; i++) {
          const cc = cand[i] as [number, number, number];
          const e = evalQuadric(tmp, 0, cc[0], cc[1], cc[2]);
          if (e < bestC) {
            bestC = e;
            best = i;
          }
        }
        const cc = cand[best] as [number, number, number];
        tx = cc[0];
        ty = cc[1];
        tz = cc[2];
      }
    }
    const cost = Math.max(0, evalQuadric(tmp, 0, tx, ty, tz));
    return { cost, a, b, va: version[a] as number, vb: version[b] as number, keep, tx, ty, tz };
  };

  // ---- seed the heap with every soup edge -----------------------------------
  const heap: Collapse[] = [];
  const seen = new Set<number>();
  for (let t = 0; t < soupTris; t++) {
    const a = triA[t] as number;
    const b = triB[t] as number;
    const c = triC[t] as number;
    for (const [u, v] of [
      [a, b],
      [b, c],
      [c, a],
    ] as [number, number][]) {
      const k = ekey(u, v);
      if (seen.has(k)) continue;
      seen.add(k);
      const e = evalCollapse(u, v);
      if (e) heapPush(heap, e);
    }
  }

  // ---- would moving keep→(tx,ty,tz) flip any surviving incident triangle? ----
  //  compares RAW cross products (sign of the dot ⇒ flip; no normalisation).
  const cb = [0, 0, 0]; // scratch: cross before
  const ca = [0, 0, 0]; // scratch: cross after
  const rawCross = (
    out: number[],
    i0: number,
    i1: number,
    i2: number,
    sx: number,
    sy: number,
    sz: number,
  ): void => {
    const gx = (id: number, c: number): number =>
      id < 0 ? (c === 0 ? sx : c === 1 ? sy : sz) : (verts[id * stride + c] as number);
    const ax = gx(i0, 0);
    const ay = gx(i0, 1);
    const az = gx(i0, 2);
    const e1x = gx(i1, 0) - ax;
    const e1y = gx(i1, 1) - ay;
    const e1z = gx(i1, 2) - az;
    const e2x = gx(i2, 0) - ax;
    const e2y = gx(i2, 1) - ay;
    const e2z = gx(i2, 2) - az;
    out[0] = e1y * e2z - e1z * e2y;
    out[1] = e1z * e2x - e1x * e2z;
    out[2] = e1x * e2y - e1y * e2x;
  };
  const wouldFlip = (a: number, b: number, tx: number, ty: number, tz: number): boolean => {
    for (let vi = 0; vi < 2; vi++) {
      const v = vi === 0 ? a : b;
      for (const t of vtri[v] as number[]) {
        if (triAlive[t] !== 1) continue;
        const i0 = triA[t] as number;
        const i1 = triB[t] as number;
        const i2 = triC[t] as number;
        // tris containing the collapsed edge degenerate away — skip
        const hasA = i0 === a || i1 === a || i2 === a;
        const hasB = i0 === b || i1 === b || i2 === b;
        if (hasA && hasB) continue;
        rawCross(cb, i0, i1, i2, 0, 0, 0); // no -1 here
        const s0 = i0 === a || i0 === b ? -1 : i0;
        const s1 = i1 === a || i1 === b ? -1 : i1;
        const s2 = i2 === a || i2 === b ? -1 : i2;
        rawCross(ca, s0, s1, s2, tx, ty, tz);
        if ((cb[0] as number) * (ca[0] as number) + (cb[1] as number) * (ca[1] as number) + (cb[2] as number) * (ca[2] as number) < 0) {
          return true;
        }
      }
    }
    return false;
  };

  // ---- collapse loop --------------------------------------------------------
  const targetTris = Math.max(1, Math.floor(soupTris * opts.targetRatio));
  let aliveTris = soupTris;
  let maxCost = 0;
  while (aliveTris > targetTris && heap.length > 0) {
    const e = heapPop(heap);
    if (vAlive[e.a] !== 1 || vAlive[e.b] !== 1) continue;
    if ((version[e.a] as number) !== e.va || (version[e.b] as number) !== e.vb) continue; // stale
    if (wouldFlip(e.a, e.b, e.tx, e.ty, e.tz)) continue;
    const keep = e.keep;
    const drop = keep === e.a ? e.b : e.a;
    // attribute update on the survivor (locked survivor keeps its record verbatim)
    if (locked[keep] !== 1) {
      const t = segParam(verts, stride, keep, drop, e.tx, e.ty, e.tz);
      const kb = keep * stride;
      const db = drop * stride;
      for (let s = 0; s < stride; s++) {
        verts[kb + s] = (verts[kb + s] as number) * (1 - t) + (verts[db + s] as number) * t;
      }
      verts[kb] = e.tx;
      verts[kb + 1] = e.ty;
      verts[kb + 2] = e.tz;
      if (opts.normalOffset >= 0) renormalize(verts, kb + opts.normalOffset);
    }
    // accumulate quadric + area onto the survivor
    addQuadric(Q, keep * 10, drop * 10);
    warea[keep] = (warea[keep] as number) + (warea[drop] as number);
    // re-point dropped vertex's triangles to the survivor; kill degenerates
    for (const t of vtri[drop] as number[]) {
      if (triAlive[t] !== 1) continue;
      if ((triA[t] as number) === drop) triA[t] = keep;
      if ((triB[t] as number) === drop) triB[t] = keep;
      if ((triC[t] as number) === drop) triC[t] = keep;
      const a0 = triA[t] as number;
      const b0 = triB[t] as number;
      const c0 = triC[t] as number;
      if (a0 === b0 || b0 === c0 || a0 === c0) {
        triAlive[t] = 0;
        aliveTris--;
      } else {
        (vtri[keep] as number[]).push(t);
      }
    }
    vAlive[drop] = 0;
    version[keep] = (version[keep] as number) + 1;
    version[drop] = (version[drop] as number) + 1;
    if (e.cost > maxCost) maxCost = e.cost;
    // compact survivor incidence + recompute edges to its neighbours
    const live: number[] = [];
    const nbr = new Set<number>();
    for (const t of vtri[keep] as number[]) {
      if (triAlive[t] !== 1) continue;
      live.push(t);
      const a0 = triA[t] as number;
      const b0 = triB[t] as number;
      const c0 = triC[t] as number;
      if (a0 !== keep) nbr.add(a0);
      if (b0 !== keep) nbr.add(b0);
      if (c0 !== keep) nbr.add(c0);
    }
    vtri[keep] = live;
    for (const w of nbr) {
      if (vAlive[w] !== 1) continue;
      const ne = evalCollapse(keep, w);
      if (ne) heapPush(heap, ne);
    }
  }

  // ---- compact survivors → fresh local mesh ---------------------------------
  const remap = new Int32Array(nv).fill(-1);
  const outPos: number[] = [];
  const outIdx: number[] = [];
  let simpTris = 0;
  for (let t = 0; t < soupTris; t++) {
    if (triAlive[t] !== 1) continue;
    const ids = [triA[t] as number, triB[t] as number, triC[t] as number];
    for (let v = 0; v < 3; v++) {
      const id = ids[v] as number;
      if ((remap[id] as number) < 0) {
        remap[id] = outPos.length / stride;
        const p = id * stride;
        for (let s = 0; s < stride; s++) outPos.push(verts[p + s] as number);
      }
      outIdx.push(remap[id] as number);
    }
    simpTris++;
  }
  const meanArea = warea.length ? warea.reduce((s, a) => s + a, 0) / Math.max(1, nv) : 1;
  const qemErr = Math.sqrt(maxCost / Math.max(meanArea, 1e-12));
  return {
    verts: Float32Array.from(outPos),
    indices: Uint32Array.from(outIdx),
    soupTris,
    simpTris,
    qemErr,
  };
}

/** clamped projection parameter of (tx,ty,tz) onto segment keep→drop */
function segParam(
  verts: Float64Array,
  stride: number,
  keep: number,
  drop: number,
  tx: number,
  ty: number,
  tz: number,
): number {
  const ax = verts[keep * stride] as number;
  const ay = verts[keep * stride + 1] as number;
  const az = verts[keep * stride + 2] as number;
  const bx = (verts[drop * stride] as number) - ax;
  const by = (verts[drop * stride + 1] as number) - ay;
  const bz = (verts[drop * stride + 2] as number) - az;
  const len2 = bx * bx + by * by + bz * bz;
  if (len2 < 1e-20) return 0;
  const t = ((tx - ax) * bx + (ty - ay) * by + (tz - az) * bz) / len2;
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

function renormalize(verts: Float64Array, base: number): void {
  const x = verts[base] as number;
  const y = verts[base + 1] as number;
  const z = verts[base + 2] as number;
  const len = Math.sqrt(x * x + y * y + z * z);
  if (len < 1e-12) return;
  verts[base] = x / len;
  verts[base + 1] = y / len;
  verts[base + 2] = z / len;
}

// ---------------------------------------------------------------------------
// orchestrator
// ---------------------------------------------------------------------------

/**
 * Build the LOD DAG for one mesh.
 *
 * @param verts   interleaved vertex pool (vertStride floats/vertex, position at offset 0)
 * @param vertStride floats per vertex
 * @param indices triangle indices (0-based into verts)
 * @param opts    see DagOpts
 * @param lod0    optional precomputed LOD0 clusters (else clusterize internally)
 */
export function buildDag(
  verts: Float32Array,
  vertStride: number,
  indices: Uint32Array,
  opts: DagOpts = {},
  lod0?: BuiltClusters,
): DagBuild {
  const t0 = now();
  const o: Required<DagOpts> = {
    maxTris: opts.maxTris ?? 128,
    targetRatio: opts.targetRatio ?? 0.5,
    groupMax: opts.groupMax ?? 24,
    stuckFrac: opts.stuckFrac ?? 0.15,
    maxLevels: opts.maxLevels ?? 24,
    normalOffset: opts.normalOffset ?? -1,
    weldEps: opts.weldEps ?? 1e-5,
  };

  const built = lod0 ?? clusterize(verts, vertStride, indices, o.maxTris);

  // growing geometry pool (LOD0 verts as-is; LOD0 indices = the permuted set)
  const poolVerts: number[] = Array.from(verts);
  const poolIdx: number[] = Array.from(built.indices);
  const clusters: DagCluster[] = [];
  const groups: DagGroup[] = [];
  const levelStats: DagLevelStats[] = [];

  // -- LOD0 clusters: ownError 0, ownSphere = geometric sphere ---------------
  for (let c = 0; c < built.clusterCount; c++) {
    const sx = built.sphere[c * 4] as number;
    const sy = built.sphere[c * 4 + 1] as number;
    const sz = built.sphere[c * 4 + 2] as number;
    const sr = built.sphere[c * 4 + 3] as number;
    clusters.push({
      level: 0,
      triStart: built.triStart[c] as number,
      triCount: built.triCount[c] as number,
      sx,
      sy,
      sz,
      sr,
      cax: built.cone[c * 4] as number,
      cay: built.cone[c * 4 + 1] as number,
      caz: built.cone[c * 4 + 2] as number,
      ccos: built.cone[c * 4 + 3] as number,
      ownError: 0,
      oex: sx,
      oey: sy,
      oez: sz,
      oer: sr,
      parentError: Infinity,
      pex: sx,
      pey: sy,
      pez: sz,
      per: sr,
      groupAsInput: -1,
      groupAsParent: -1,
    });
  }
  const lod0Count = clusters.length;

  // -- build levels -----------------------------------------------------------
  let active: number[] = clusters.map((_, i) => i);
  let maxError = 0;
  for (let level = 0; level < o.maxLevels; level++) {
    if (active.length < 2) break; // nothing left to merge
    const partition = partitionClusters(active, clusters, o.groupMax);
    let outClusters = 0;
    let outTris = 0;
    let inTris = 0;
    let stuckGroups = 0;
    const nextActive: number[] = [];
    for (const member of active) inTris += (clusters[member] as DagCluster).triCount;

    for (const groupIds of partition) {
      const gid = groups.length;
      const groupClusters = groupIds.map((id) => clusters[id] as DagCluster);
      // simplifyGroup only READS the pool; pass the growing arrays directly
      // (appending parents below never moves existing cluster tri ranges).
      const res = simplifyGroup(poolVerts, poolIdx, vertStride, groupClusters, o);

      // child own-error max + own-sphere union (containment)
      let childErr = 0;
      let sphere: Sphere | null = null;
      for (const id of groupIds) {
        const c = clusters[id] as DagCluster;
        if (c.ownError > childErr) childErr = c.ownError;
        const s: Sphere = { x: c.oex, y: c.oey, z: c.oez, r: c.oer };
        sphere = sphere ? mergeSpheres(sphere, s) : s;
      }
      const gs = sphere ?? { x: 0, y: 0, z: 0, r: 0 };

      const reduction = res.soupTris > 0 ? 1 - res.simpTris / res.soupTris : 0;
      const stuck = reduction < o.stuckFrac || res.simpTris < 1;

      if (stuck) {
        // group cannot reduce → its inputs become roots (parentError stays +∞)
        stuckGroups++;
        groups.push({
          id: gid,
          level,
          error: childErr,
          sx: gs.x,
          sy: gs.y,
          sz: gs.z,
          sr: gs.r,
          inputs: groupIds.slice(),
          parents: [],
          reduced: false,
        });
        continue;
      }

      // group error: strictly monotone above every child's own error
      const groupErr = Math.max(res.qemErr, childErr * (1 + 1e-6) + 1e-7);
      if (groupErr > maxError) maxError = groupErr;

      // append the simplified soup to the pool, re-clusterize into parents
      const vertBase = poolVerts.length / vertStride;
      for (let i = 0; i < res.verts.length; i++) poolVerts.push(res.verts[i] as number);
      const parentBuilt = clusterize(res.verts, vertStride, res.indices, o.maxTris);
      const triBase = poolIdx.length / 3;
      for (let i = 0; i < parentBuilt.indices.length; i++) {
        poolIdx.push((parentBuilt.indices[i] as number) + vertBase);
      }

      const parentIds: number[] = [];
      for (let c = 0; c < parentBuilt.clusterCount; c++) {
        const id = clusters.length;
        clusters.push({
          level: level + 1,
          triStart: triBase + (parentBuilt.triStart[c] as number),
          triCount: parentBuilt.triCount[c] as number,
          sx: parentBuilt.sphere[c * 4] as number,
          sy: parentBuilt.sphere[c * 4 + 1] as number,
          sz: parentBuilt.sphere[c * 4 + 2] as number,
          sr: parentBuilt.sphere[c * 4 + 3] as number,
          cax: parentBuilt.cone[c * 4] as number,
          cay: parentBuilt.cone[c * 4 + 1] as number,
          caz: parentBuilt.cone[c * 4 + 2] as number,
          ccos: parentBuilt.cone[c * 4 + 3] as number,
          // parent OWN pair == group pair (exact); parent of the parent is TBD
          ownError: groupErr,
          oex: gs.x,
          oey: gs.y,
          oez: gs.z,
          oer: gs.r,
          parentError: Infinity,
          pex: gs.x,
          pey: gs.y,
          pez: gs.z,
          per: gs.r,
          groupAsInput: -1,
          groupAsParent: gid,
        });
        parentIds.push(id);
        nextActive.push(id);
        outClusters++;
        outTris += parentBuilt.triCount[c] as number;
      }

      // every input child shares this group's parent pair, bit-for-bit
      for (const id of groupIds) {
        const c = clusters[id] as DagCluster;
        c.parentError = groupErr;
        c.pex = gs.x;
        c.pey = gs.y;
        c.pez = gs.z;
        c.per = gs.r;
        c.groupAsInput = gid;
      }

      groups.push({
        id: gid,
        level,
        error: groupErr,
        sx: gs.x,
        sy: gs.y,
        sz: gs.z,
        sr: gs.r,
        inputs: groupIds.slice(),
        parents: parentIds,
        reduced: true,
      });
    }

    levelStats.push({
      level,
      inClusters: active.length,
      inTris,
      outClusters,
      outTris,
      groups: partition.length,
      stuckGroups,
      triReduction: inTris > 0 ? 1 - outTris / inTris : 0,
    });

    if (nextActive.length === 0) break; // all groups stuck → done
    if (nextActive.length >= active.length) break; // no progress → stop (roots stand)
    active = nextActive;
  }

  // -- pack -------------------------------------------------------------------
  const outVerts = Float32Array.from(poolVerts);
  const outIdx = Uint32Array.from(poolIdx);
  let roots = 0;
  let totalTris = 0;
  for (const c of clusters) {
    if (!Number.isFinite(c.parentError)) roots++;
    totalTris += c.triCount;
  }

  return {
    verts: outVerts,
    vertStride,
    indices: outIdx,
    clusters,
    groups,
    levelStats,
    lod0Count,
    stats: {
      buildMs: now() - t0,
      levels: levelStats.length,
      lod0Clusters: lod0Count,
      lod0Tris: built.indices.length / 3,
      totalClusters: clusters.length,
      totalTris,
      roots,
      maxError,
    },
  };
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}
