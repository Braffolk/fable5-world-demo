/**
 * Shared primitives for the LOD-DAG builders (NANITE-SPEC.md N8/N9).
 *
 * Both the QEM cluster DAG (BuildDag.ts) and the foliage AGGREGATE DAG
 * (BuildAggregateDag.ts) need the same low-level pieces: a provably-containing
 * bounding-sphere fold (containment ⇒ projection monotonicity, the crack-free
 * cut requirement) and a deterministic spatial group partition. They live here
 * so neither builder re-implements them — extending the DAG with a new
 * simplification strategy must not mean copy-pasting the orchestration glue.
 *
 * Pure CPU/typed-array code — no three.js, runs under node for the probes.
 */

export interface Sphere {
  x: number;
  y: number;
  z: number;
  r: number;
}

/** smallest-ish sphere that provably CONTAINS both s0 and s1 (zeux: union, not
 *  a distance heuristic — containment is required for projection monotonicity). */
export function mergeSpheres(s0: Sphere, s1: Sphere): Sphere {
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

/** Split a set of cluster ids into groups of ≤ groupMax by recursive longest-axis
 *  median bisection of their sphere centres (deterministic METIS substitute —
 *  zeux meshoptimizer#750). Generic over any record exposing a sphere centre, so
 *  both builders share it without a circular type dependency on DagCluster. */
export function partitionClusters<T extends { sx: number; sy: number; sz: number }>(
  ids: number[],
  clusters: ArrayLike<T>,
  groupMax: number,
): number[][] {
  if (ids.length <= groupMax) return [ids];
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (const id of ids) {
    const c = clusters[id] as T;
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
    const c = clusters[id] as T;
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

/** monotonic wall clock (browser perf or node Date) for build-cost stats */
export function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}
