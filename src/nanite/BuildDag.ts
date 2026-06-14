/**
 * Nanite LOD DAG build (NANITE-SPEC.md N8-D0) — hand-rolled, CPU/typed-array,
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
  /** GRID mode (terrain heightfield): restrict every interior edge-collapse to
   *  the lower-cost ENDPOINT instead of the off-grid QEM-optimal point, so all
   *  survivors stay on the heightfield grid (F4 — positions reconstruct from the
   *  heights buffer, never baked off-grid). QEM cost on a heightfield IS the
   *  vertical deviation, so flat regions (≈0 cost) collapse first and cliffs
   *  (high cost) survive → the adaptive "plains cheap / cliffs dense" decimation.
   *  Default false (explicit meshes use the free QEM-optimal target). */
  gridEndpoint?: boolean;
  /** terrain GRID mode: per grid-vertex VERTICAL error (the martini error
   *  pyramid), sampled at a survivor's world (x,z). When set, an interior
   *  collapse DROPS the lower-error vertex and its cost IS that vertical error
   *  (metres) — so the decimation is ordered + bounded by true vertical
   *  deviation, not the area-weighted QEM scalar. */
  gridErrAt?: (x: number, z: number) => number;
  /** terrain GRID mode: per-level vertical-error BUDGET (metres). A level only
   *  runs collapses with error ≤ budget(level); costlier ones DEFER to the next
   *  (higher-budget) level. Doubling per level ⇒ progressive LOD bands: flat
   *  plains collapse at the lowest band, cliffs survive until the budget grows
   *  to their detail → smooth ±1 cut, no big jumps (crack-free). */
  levelBudget?: (level: number) => number;
}

/** resolved options: every scalar defaulted, the terrain callbacks stay optional */
type FullDagOpts = Required<Omit<DagOpts, 'gridErrAt' | 'levelBudget'>> &
  Pick<DagOpts, 'gridErrAt' | 'levelBudget'>;

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
  opts: FullDagOpts,
  /** terrain error-bounded mode: collapse only while the cheapest remaining
   *  drop costs ≤ this vertical-error budget (Infinity ⇒ ratio mode). */
  budget = Infinity,
): SimplifyResult {
  // ---- weld group triangles into a local soup (by quantised position) -------
  //  Open-addressing typed hash (was a Map<hash, number[]> + per-vertex number[]
  //  push — the group's heaviest GC source). Local ids are still assigned in first-
  //  ENCOUNTER order, so the welded soup is BIT-IDENTICAL; the hash only changes the
  //  internal probe order. Locked boundary verts quantise identically across groups
  //  ⇒ they re-weld next level (the boundary re-simplification mechanism). The soup
  //  is bounded (≤ 3·inputTris verts), so everything is sized once, no growth.
  const inv = 1 / opts.weldEps;
  let totalInputTris = 0;
  for (const c of groupClusters) totalInputTris += c.triCount;
  const maxVerts = Math.max(totalInputTris * 3, 1);
  let hcap = 16;
  while (hcap < maxVerts * 2) hcap *= 2;
  const hmask = hcap - 1;
  const hslot = new Int32Array(hcap).fill(-1); // slot → localId (-1 empty)
  const localQX = new Int32Array(maxVerts);
  const localQY = new Int32Array(maxVerts);
  const localQZ = new Int32Array(maxVerts);
  const verts = new Float64Array(maxVerts * stride);
  const triA = new Int32Array(totalInputTris);
  const triB = new Int32Array(totalInputTris);
  const triC = new Int32Array(totalInputTris);
  let nLocal = 0;
  const localOf = (vid: number): number => {
    const p = vid * stride;
    const qx = Math.round((pool[p] as number) * inv);
    const qy = Math.round((pool[p + 1] as number) * inv);
    const qz = Math.round((pool[p + 2] as number) * inv);
    let h = (Math.imul(qx, 73856093) ^ Math.imul(qy, 19349663) ^ Math.imul(qz, 83492791)) & hmask;
    for (;;) {
      const lid = hslot[h] as number;
      if (lid === -1) {
        const id = nLocal++;
        const b = id * stride;
        for (let s = 0; s < stride; s++) verts[b + s] = pool[p + s] as number;
        localQX[id] = qx;
        localQY[id] = qy;
        localQZ[id] = qz;
        hslot[h] = id;
        return id;
      }
      if ((localQX[lid] as number) === qx && (localQY[lid] as number) === qy && (localQZ[lid] as number) === qz) {
        return lid;
      }
      h = (h + 1) & hmask;
    }
  };
  let soupTris = 0;
  for (const c of groupClusters) {
    for (let t = c.triStart; t < c.triStart + c.triCount; t++) {
      const a = localOf(poolIdx[t * 3] as number);
      const b = localOf(poolIdx[t * 3 + 1] as number);
      const cc = localOf(poolIdx[t * 3 + 2] as number);
      if (a === b || b === cc || a === cc) continue; // welded-degenerate
      triA[soupTris] = a;
      triB[soupTris] = b;
      triC[soupTris] = cc;
      soupTris++;
    }
  }
  const nv = nLocal;

  // ---- triangle incidence ---------------------------------------------------
  const triAlive = new Uint8Array(soupTris).fill(1);
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

  // ---- terrain FAST path: cost = precomputed martini vertical error ----------
  //  In gridEndpoint+gridErrAt mode the collapse cost of edge (u,v) is a FIXED
  //  function of the ORIGINAL heightfield — drop the lower-error endpoint, cost =
  //  min(err[u],err[v]) — NOT a quadric that mutates as the mesh simplifies. So we
  //  precompute per-vertex error ONCE and skip the entire QEM machinery (quadric
  //  accumulation over every triangle, the tmp solve, the per-collapse reseed
  //  cost). Output is BIT-IDENTICAL to the quadric path (which, in this mode,
  //  computed Q but never read it for cost/target) — just far cheaper.
  const terrainFast = opts.gridEndpoint && opts.gridErrAt != null;
  const gErr = opts.gridErrAt;
  const localErr = terrainFast ? new Float64Array(nv) : null;
  if (terrainFast && localErr && gErr) {
    for (let v = 0; v < nv; v++) {
      localErr[v] = gErr(verts[v * stride] as number, verts[v * stride + 2] as number);
    }
  }

  // ---- version-stamp membership scratch -------------------------------------
  //  Replaces the per-call Sets in linkOk + the reseed dedup (which dominated the
  //  build — native Set alloc/add/has). mark[v]===markGen ⇔ v is "in the set";
  //  bumping markGen clears the whole set in O(1) with no allocation. markGen is
  //  local to this group and monotonic, so stamps never collide across collapses.
  const mark = new Int32Array(nv);
  let markGen = 0;

  // ---- per-vertex quadrics (area-weighted) + accumulated area ---------------
  //  (skipped in terrainFast — the precomputed martini error replaces the quadric)
  const Q = new Float64Array(terrainFast ? 0 : nv * 10);
  const warea = new Float64Array(terrainFast ? 0 : nv);
  if (!terrainFast)
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
    if (terrainFast && localErr) {
      // keep the higher-error endpoint (stay where vertical detail is) and drop the
      // lower; the cost IS the dropped vertex's martini error (metres), precomputed.
      const keep =
        locked[a] === 1 ? a : locked[b] === 1 ? b : (localErr[a] as number) >= (localErr[b] as number) ? a : b;
      const drop = keep === a ? b : a;
      return {
        cost: localErr[drop] as number,
        a,
        b,
        va: version[a] as number,
        vb: version[b] as number,
        keep,
        tx: verts[keep * stride] as number,
        ty: verts[keep * stride + 1] as number,
        tz: verts[keep * stride + 2] as number,
      };
    }
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
    } else if (opts.gridEndpoint) {
      // terrain GRID mode: KEEP one endpoint (stay on-grid, F4); DROP the lower-
      // detail one. With the martini error field, "detail" = vertical error
      // (keep the higher-error vertex); else fall back to the lower-QEM endpoint.
      let keepA: boolean;
      if (opts.gridErrAt) {
        const errA = opts.gridErrAt(verts[a * stride] as number, verts[a * stride + 2] as number);
        const errB = opts.gridErrAt(verts[b * stride] as number, verts[b * stride + 2] as number);
        keepA = errA >= errB;
      } else {
        const ea = evalQuadric(tmp, 0, verts[a * stride] as number, verts[a * stride + 1] as number, verts[a * stride + 2] as number);
        const eb = evalQuadric(tmp, 0, verts[b * stride] as number, verts[b * stride + 1] as number, verts[b * stride + 2] as number);
        keepA = ea <= eb;
      }
      keep = keepA ? a : b;
      tx = verts[keep * stride] as number;
      ty = verts[keep * stride + 1] as number;
      tz = verts[keep * stride + 2] as number;
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
    // martini mode: cost = the DROPPED vertex's true vertical error (metres), so
    // the heap orders + the budget bounds by real height deviation. Else: the
    // area-weighted QEM scalar (explicit meshes, ratio-bounded).
    let cost: number;
    if (opts.gridErrAt) {
      const drop = keep === a ? b : a;
      cost = opts.gridErrAt(verts[drop * stride] as number, verts[drop * stride + 2] as number);
    } else {
      cost = Math.max(0, evalQuadric(tmp, 0, tx, ty, tz));
    }
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
  //  Compares RAW cross products (sign of the dot ⇒ flip; no normalisation). Fully
  //  INLINED — no rawCross / nested gx closure: this was the build's single hottest
  //  spot (the per-call closure ALLOCATION + indirect component fetches, ~50% of
  //  the build). Verts equal to a or b move to (tx,ty,tz); others keep position.
  const wouldFlip = (a: number, b: number, tx: number, ty: number, tz: number, skip: number): boolean => {
    for (let vi = 0; vi < 2; vi++) {
      const v = vi === 0 ? a : b;
      if (v === skip) continue; // this vertex doesn't move ⇒ its star can't flip
      const tris = vtri[v] as number[];
      for (let ti = 0; ti < tris.length; ti++) {
        const t = tris[ti] as number;
        if (triAlive[t] !== 1) continue;
        const i0 = triA[t] as number;
        const i1 = triB[t] as number;
        const i2 = triC[t] as number;
        // tris containing the collapsed edge degenerate away — skip
        if ((i0 === a || i1 === a || i2 === a) && (i0 === b || i1 === b || i2 === b)) continue;
        const o0 = i0 * stride;
        const o1 = i1 * stride;
        const o2 = i2 * stride;
        const x0 = verts[o0] as number;
        const y0 = verts[o0 + 1] as number;
        const z0 = verts[o0 + 2] as number;
        const x1 = verts[o1] as number;
        const y1 = verts[o1 + 1] as number;
        const z1 = verts[o1 + 2] as number;
        const x2 = verts[o2] as number;
        const y2 = verts[o2 + 1] as number;
        const z2 = verts[o2 + 2] as number;
        // BEFORE cross (original positions)
        const be1x = x1 - x0;
        const be1y = y1 - y0;
        const be1z = z1 - z0;
        const be2x = x2 - x0;
        const be2y = y2 - y0;
        const be2z = z2 - z0;
        const cbx = be1y * be2z - be1z * be2y;
        const cby = be1z * be2x - be1x * be2z;
        const cbz = be1x * be2y - be1y * be2x;
        // AFTER cross (verts == a or b snap to the target)
        const X0 = i0 === a || i0 === b ? tx : x0;
        const Y0 = i0 === a || i0 === b ? ty : y0;
        const Z0 = i0 === a || i0 === b ? tz : z0;
        const X1 = i1 === a || i1 === b ? tx : x1;
        const Y1 = i1 === a || i1 === b ? ty : y1;
        const Z1 = i1 === a || i1 === b ? tz : z1;
        const X2 = i2 === a || i2 === b ? tx : x2;
        const Y2 = i2 === a || i2 === b ? ty : y2;
        const Z2 = i2 === a || i2 === b ? tz : z2;
        const ae1x = X1 - X0;
        const ae1y = Y1 - Y0;
        const ae1z = Z1 - Z0;
        const ae2x = X2 - X0;
        const ae2y = Y2 - Y0;
        const ae2z = Z2 - Z0;
        const cax = ae1y * ae2z - ae1z * ae2y;
        const cay = ae1z * ae2x - ae1x * ae2z;
        const caz = ae1x * ae2y - ae1y * ae2x;
        if (cbx * cax + cby * cay + cbz * caz < 0) return true;
      }
    }
    return false;
  };

  // ---- link condition (manifold safety) -------------------------------------
  //  A half-edge collapse a→b is manifold-safe iff a and b share NO common
  //  neighbour other than the apex(es) of the triangle(s) on edge (a,b). A
  //  third shared neighbour ⇒ the collapse folds 3+ triangles onto one edge
  //  (non-manifold). QEM-optimal collapses on irregular meshes rarely trip this,
  //  but REGULAR grid endpoint-collapses do — so terrain checks it explicitly.
  const linkOk = (a: number, b: number): boolean => {
    // stamp a's neighbours (nbrA) with a fresh generation; track the apex(es) —
    // the third verts of the triangle(s) on edge a-b — inline (a manifold edge has
    // ≤2 incident tris, so ≤2 apexes; if the soup were already non-manifold here a
    // 3rd apex only makes this stricter, which is crack-safe).
    markGen++;
    const gen = markGen;
    let apex0 = -1;
    let apex1 = -1;
    for (const t of vtri[a] as number[]) {
      if (triAlive[t] !== 1) continue;
      const i0 = triA[t] as number;
      const i1 = triB[t] as number;
      const i2 = triC[t] as number;
      if (i0 !== a) mark[i0] = gen;
      if (i1 !== a) mark[i1] = gen;
      if (i2 !== a) mark[i2] = gen;
      if (i0 === b || i1 === b || i2 === b) {
        if (i0 !== a && i0 !== b) apex0 < 0 ? (apex0 = i0) : (apex1 = i0);
        if (i1 !== a && i1 !== b) apex0 < 0 ? (apex0 = i1) : (apex1 = i1);
        if (i2 !== a && i2 !== b) apex0 < 0 ? (apex0 = i2) : (apex1 = i2);
      }
    }
    for (const t of vtri[b] as number[]) {
      if (triAlive[t] !== 1) continue;
      const i0 = triA[t] as number;
      const i1 = triB[t] as number;
      const i2 = triC[t] as number;
      // shared non-apex neighbour ⇒ collapse folds 3+ tris on edge a-b (non-manifold)
      if (i0 !== b && mark[i0] === gen && i0 !== apex0 && i0 !== apex1) return false;
      if (i1 !== b && mark[i1] === gen && i1 !== apex0 && i1 !== apex1) return false;
      if (i2 !== b && mark[i2] === gen && i2 !== apex0 && i2 !== apex1) return false;
    }
    return true;
  };

  // ---- degeneracy guard (terrain) -------------------------------------------
  //  Reject a collapse that would leave any surviving incident triangle near-
  //  COLLINEAR. On a flat boundary row, endpoint-collapsing the interior can
  //  fold a triangle onto 3 collinear grid points (zero area). wouldFlip only
  //  catches sign FLIPS, not degeneracies, so the zero-area sliver survives and —
  //  produced identically by both groups sharing that boundary — reads as a
  //  duplicated, non-manifold triangle. Scale-invariant test: sin²θ < 1e-8.
  const triDegenerates = (a: number, b: number, tx: number, ty: number, tz: number, skip: number): boolean => {
    for (let vi = 0; vi < 2; vi++) {
      const v = vi === 0 ? a : b;
      if (v === skip) continue; // this vertex doesn't move ⇒ its star can't degenerate
      const tris = vtri[v] as number[];
      for (let ti = 0; ti < tris.length; ti++) {
        const t = tris[ti] as number;
        if (triAlive[t] !== 1) continue;
        const i0 = triA[t] as number;
        const i1 = triB[t] as number;
        const i2 = triC[t] as number;
        if ((i0 === a || i1 === a || i2 === a) && (i0 === b || i1 === b || i2 === b)) continue; // degenerates away
        // inlined position fetch (verts == a or b snap to the target) — no closure
        const o0 = i0 * stride;
        const o1 = i1 * stride;
        const o2 = i2 * stride;
        const ax = i0 === a || i0 === b ? tx : (verts[o0] as number);
        const ay = i0 === a || i0 === b ? ty : (verts[o0 + 1] as number);
        const az = i0 === a || i0 === b ? tz : (verts[o0 + 2] as number);
        const e1x = (i1 === a || i1 === b ? tx : (verts[o1] as number)) - ax;
        const e1y = (i1 === a || i1 === b ? ty : (verts[o1 + 1] as number)) - ay;
        const e1z = (i1 === a || i1 === b ? tz : (verts[o1 + 2] as number)) - az;
        const e2x = (i2 === a || i2 === b ? tx : (verts[o2] as number)) - ax;
        const e2y = (i2 === a || i2 === b ? ty : (verts[o2 + 1] as number)) - ay;
        const e2z = (i2 === a || i2 === b ? tz : (verts[o2 + 2] as number)) - az;
        const cx = e1y * e2z - e1z * e2y;
        const cy = e1z * e2x - e1x * e2z;
        const cz = e1x * e2y - e1y * e2x;
        const m2 = cx * cx + cy * cy + cz * cz;
        const d = (e1x * e1x + e1y * e1y + e1z * e1z) * (e2x * e2x + e2y * e2y + e2z * e2z);
        if (d > 0 && m2 < 1e-8 * d) return true; // sin²θ < 1e-8 ⇒ near-collinear
      }
    }
    return false;
  };

  // ---- shared-boundary guard (terrain) --------------------------------------
  //  Reject a collapse that would leave a surviving triangle whose THREE verts
  //  are all LOCKED (all on the group boundary). Such a triangle lies entirely on
  //  the seam shared with the neighbouring group, which — locking the identical
  //  verts — re-creates the SAME triangle, so it appears in BOTH clusters (a
  //  non-manifold duplicate). Keeping an interior vertex makes the triangle owned
  //  by exactly one group. (Rejecting a collapse never opens a hole.)
  const makesBoundaryTri = (a: number, b: number, keep: number): boolean => {
    for (let vi = 0; vi < 2; vi++) {
      const v = vi === 0 ? a : b;
      for (const t of vtri[v] as number[]) {
        if (triAlive[t] !== 1) continue;
        const i0 = triA[t] as number;
        const i1 = triB[t] as number;
        const i2 = triC[t] as number;
        if ((i0 === a || i1 === a || i2 === a) && (i0 === b || i1 === b || i2 === b)) continue; // degenerates away
        const m0 = i0 === a || i0 === b ? keep : i0;
        const m1 = i1 === a || i1 === b ? keep : i1;
        const m2 = i2 === a || i2 === b ? keep : i2;
        if (locked[m0] === 1 && locked[m1] === 1 && locked[m2] === 1) return true;
      }
    }
    return false;
  };

  // ---- collapse loop --------------------------------------------------------
  //  ratio mode (explicit): collapse cheapest-first until ≤ targetRatio tris.
  //  budget mode (terrain): collapse every drop whose vertical error ≤ budget;
  //  stop once the cheapest remaining exceeds the band (the rest defer upward).
  const useBudget = Number.isFinite(budget);
  const targetTris = Math.max(1, Math.floor(soupTris * opts.targetRatio));
  let aliveTris = soupTris;
  let maxCost = 0;
  while (heap.length > 0) {
    if (useBudget) {
      if ((heap[0] as Collapse).cost > budget) break; // min-heap: nothing else fits the band
    } else if (aliveTris <= targetTris) {
      break;
    }
    const e = heapPop(heap);
    if (vAlive[e.a] !== 1 || vAlive[e.b] !== 1) continue;
    if ((version[e.a] as number) !== e.va || (version[e.b] as number) !== e.vb) continue; // stale
    // gridEndpoint: target == keep's own position ⇒ KEEP doesn't move, so only the
    // DROP star can flip/degenerate — skip keep's star (bit-identical: keep's tris
    // are unchanged and the live mesh has no pre-existing flipped/degenerate tri).
    if (wouldFlip(e.a, e.b, e.tx, e.ty, e.tz, opts.gridEndpoint ? e.keep : -1)) continue;
    if (opts.gridEndpoint && !linkOk(e.a, e.b)) continue; // terrain: keep the soup manifold
    if (opts.gridEndpoint && triDegenerates(e.a, e.b, e.tx, e.ty, e.tz, e.keep)) continue; // no collinear slivers
    if (opts.gridEndpoint && makesBoundaryTri(e.a, e.b, e.keep)) continue; // no seam-duplicate tris
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
    // accumulate quadric + area onto the survivor (skipped in terrainFast — no QEM)
    if (!terrainFast) {
      addQuadric(Q, keep * 10, drop * 10);
      warea[keep] = (warea[keep] as number) + (warea[drop] as number);
    }
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
    // compact survivor incidence + recompute edges to its neighbours. Stamp-dedup
    // the neighbour list (was a per-collapse Set — the build's hottest allocation);
    // push order is preserved, so the heap-seed order is bit-identical.
    const live: number[] = [];
    markGen++;
    const gen = markGen;
    const nbrList: number[] = [];
    for (const t of vtri[keep] as number[]) {
      if (triAlive[t] !== 1) continue;
      live.push(t);
      const a0 = triA[t] as number;
      const b0 = triB[t] as number;
      const c0 = triC[t] as number;
      if (a0 !== keep && mark[a0] !== gen) (mark[a0] = gen), nbrList.push(a0);
      if (b0 !== keep && mark[b0] !== gen) (mark[b0] = gen), nbrList.push(b0);
      if (c0 !== keep && mark[c0] !== gen) (mark[c0] = gen), nbrList.push(c0);
    }
    vtri[keep] = live;
    for (const w of nbrList) {
      if (vAlive[w] !== 1) continue;
      const ne = evalCollapse(keep, w);
      if (ne) heapPush(heap, ne);
    }
  }

  // ---- compact survivors → fresh local mesh (typed scratch, no number[].push) -
  //  remap assigns new ids in first-encounter order (bit-identical to the old
  //  outPos.length/stride); the result is returned as exact-length subarray views.
  const remap = new Int32Array(nv).fill(-1);
  const outPos = new Float32Array(nv * stride); // ≤ nv survivor verts
  const outIdx = new Uint32Array(soupTris * 3); // ≤ soupTris survivor tris
  let outV = 0;
  let outI = 0;
  let simpTris = 0;
  for (let t = 0; t < soupTris; t++) {
    if (triAlive[t] !== 1) continue;
    const ta = triA[t] as number;
    const tb = triB[t] as number;
    const tc = triC[t] as number;
    for (let v = 0; v < 3; v++) {
      const id = v === 0 ? ta : v === 1 ? tb : tc;
      let r = remap[id] as number;
      if (r < 0) {
        r = outV++;
        remap[id] = r;
        const p = id * stride;
        const o = r * stride;
        for (let s = 0; s < stride; s++) outPos[o + s] = verts[p + s] as number;
      }
      outIdx[outI++] = r;
    }
    simpTris++;
  }
  const meanArea = warea.length ? warea.reduce((s, a) => s + a, 0) / Math.max(1, nv) : 1;
  // budget mode: maxCost IS the worst vertical error in metres (already the cut's
  // unit). ratio mode: convert the area-weighted QEM scalar to an RMS metre.
  const qemErr = useBudget ? maxCost : Math.sqrt(maxCost / Math.max(meanArea, 1e-12));
  return {
    verts: outPos.subarray(0, outV * stride),
    indices: outIdx.subarray(0, outI),
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
  const o: FullDagOpts = {
    maxTris: opts.maxTris ?? 128,
    targetRatio: opts.targetRatio ?? 0.5,
    groupMax: opts.groupMax ?? 24,
    stuckFrac: opts.stuckFrac ?? 0.15,
    maxLevels: opts.maxLevels ?? 24,
    normalOffset: opts.normalOffset ?? -1,
    weldEps: opts.weldEps ?? 1e-5,
    gridEndpoint: opts.gridEndpoint ?? false,
    gridErrAt: opts.gridErrAt,
    levelBudget: opts.levelBudget,
  };

  const built = lod0 ?? clusterize(verts, vertStride, indices, o.maxTris);

  // growing geometry pools (LOD0 verts as-is; LOD0 indices = the permuted set),
  // TYPED-ARRAY backed: number[].push over millions of floats was O(n) realloc +
  // boxing + a final Float32Array.from over a ~400 MB JS array at 33.5M tris. This
  // is amortised-O(1) append (capacity-doubling) + an exact slice to finalise.
  let pv = new Float32Array(Math.max(verts.length * 2, 1024));
  pv.set(verts);
  let pvLen = verts.length;
  let pi = new Uint32Array(Math.max(built.indices.length * 2, 1024));
  pi.set(built.indices);
  let piLen = built.indices.length;
  const growV = (extra: number): void => {
    if (pvLen + extra <= pv.length) return;
    let cap = pv.length;
    while (cap < pvLen + extra) cap *= 2;
    const next = new Float32Array(cap);
    next.set(pv.subarray(0, pvLen));
    pv = next;
  };
  const growI = (extra: number): void => {
    if (piLen + extra <= pi.length) return;
    let cap = pi.length;
    while (cap < piLen + extra) cap *= 2;
    const next = new Uint32Array(cap);
    next.set(pi.subarray(0, piLen));
    pi = next;
  };
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
  let noProgress = 0; // terrain: consecutive levels with zero reduction (budget too low / all-locked)
  for (let level = 0; level < o.maxLevels; level++) {
    if (active.length < 2) break; // nothing left to merge
    // terrain: this level's vertical-error budget (Infinity ⇒ explicit ratio mode)
    const budget = o.levelBudget ? o.levelBudget(level) : Infinity;
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
      // simplifyGroup only READS the pool; pass the current backing arrays
      // (appending parents below grows them, but only AFTER this returns).
      const res = simplifyGroup(pv, pi, vertStride, groupClusters, o, budget);

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
        // group cannot reduce at this budget. Explicit mesh: its inputs become
        // roots (parentError stays +∞). Terrain (budget mode): DEFER the inputs
        // to the next, higher-budget level instead — a cliff that can't simplify
        // at e_ℓ may at 2·e_ℓ, so it earns intermediate LODs (smooth cut) rather
        // than freezing at LOD0. If nothing anywhere reduces, the loop's
        // no-progress guard ends it and the un-collapsed clusters stay roots.
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
        if (o.gridEndpoint) for (const id of groupIds) nextActive.push(id);
        continue;
      }

      // group error: strictly monotone above every child's own error
      const groupErr = Math.max(res.qemErr, childErr * (1 + 1e-6) + 1e-7);
      if (groupErr > maxError) maxError = groupErr;

      // append the simplified soup to the pool, re-clusterize into parents
      const vertBase = pvLen / vertStride;
      growV(res.verts.length);
      pv.set(res.verts, pvLen);
      pvLen += res.verts.length;
      const parentBuilt = clusterize(res.verts, vertStride, res.indices, o.maxTris);
      const triBase = piLen / 3;
      growI(parentBuilt.indices.length);
      for (let i = 0; i < parentBuilt.indices.length; i++) {
        pi[piLen + i] = (parentBuilt.indices[i] as number) + vertBase;
      }
      piLen += parentBuilt.indices.length;

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

    if (nextActive.length === 0) break; // all groups stuck (non-deferring) → done
    if (o.gridEndpoint) {
      // terrain: a no-reduction level isn't terminal — the budget doubles next
      // level and a cliff may then simplify. Only stop after 2 consecutive dead
      // levels (4× budget, still nothing ⇒ the remainder is locked-boundary).
      if (outClusters === 0) {
        if (++noProgress >= 2) break;
      } else {
        noProgress = 0;
      }
    } else if (nextActive.length >= active.length) {
      break; // explicit ratio mode: no progress → stop (roots stand)
    }
    active = nextActive;
  }

  // -- pack -------------------------------------------------------------------
  const outVerts = pv.slice(0, pvLen);
  const outIdx = pi.slice(0, piLen);
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
