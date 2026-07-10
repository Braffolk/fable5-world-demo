/**
 * Foliage AGGREGATE LOD DAG (NANITE-SPEC.md N9-C1; DAG-section "Aggregates").
 *
 * A leaf/needle crown is THOUSANDS of DISCONNECTED quads/strips. QEM edge-collapse
 * (BuildDag.ts) DEGENERATES on disconnected geometry — there are no interior edges
 * to collapse, so it gets "stuck" immediately and every leaf becomes a root. The
 * foliage simplification Epic shipped for Fortnite ("Preserve Area") is different:
 * per LOD level, stochastically REMOVE whole leaves and GROW the survivors so the
 * crown keeps its silhouette MASS (area) instead of balding. Far crowns become
 * "fewer, bigger leaf shapes", never a thinning skeleton.
 *
 * This builder mirrors BuildDag's CRACK-FREE cut machinery exactly — it emits the
 * SAME (ownError, ownSphere)/(parentError, parentSphere) records with bit-exact
 * sibling pairs, so a leaf aggregate rides the existing flat kClusterCull cut
 * (D-N31) UNCHANGED at C2. The ONLY thing that differs from the QEM DAG is the
 * per-level simplification STRATEGY (remove+grow vs edge-collapse) — so the
 * orchestration glue (LOD0 setup, spatial grouping, pair assignment, packing) is
 * the same shape, and the genuinely-shared primitives (mergeSpheres,
 * partitionClusters) are imported from DagCommon rather than duplicated.
 *
 * Crack-freeness here is STRONGER and simpler than the QEM case: islands are
 * connected components, so GLOBAL connected-component detection reunites a leaf
 * even if clusterize split it across clusters, and a per-island keep/drop + a
 * per-island affine (scale about the island centroid) is applied IDENTICALLY to
 * every fragment. A leaf is therefore present-or-absent as one atomic unit, never
 * half-dropped, and a shared vertex transforms bit-identically from both sides.
 *
 * Determinism: no RNG — the keep/drop order is a hash of the (seeded) island
 * centroid, so `?seed=N` thins differently but reproducibly. Pure CPU/typed-array,
 * node-runnable for tools/probe-aggregate.ts (no three.js, no GPU).
 */

import {
  type DagBuild,
  type DagCluster,
  type DagGroup,
  type DagLevelStats,
} from './BuildDag';
import { clusterize } from './Clusterize';
import { type Sphere, mergeSpheres, now, partitionClusters } from './DagCommon';

export interface AggregateDagOpts {
  /** cluster triangle cap (matches Clusterize), default 128 */
  maxTris?: number;
  /** keep ~this fraction of leaf AREA each level (the rest is removed + its area
   *  redistributed onto survivors), default 0.5 */
  targetRatio?: number;
  /** max clusters per spatial group (recursive bisection stops at ≤ this), default 24 */
  groupMax?: number;
  /** a group that removes fewer than this fraction of its tris is "stuck" → its
   *  clusters become roots (multiple roots are legal), default 0.15 */
  stuckFrac?: number;
  /** safety cap on DAG levels, default 24 */
  maxLevels?: number;
  /** position weld grid (world units) for connected-component island detection,
   *  default 1e-5 */
  weldEps?: number;
  /** deterministic thinning seed (folded into the per-island hash), default 0 */
  seed?: number;
  /** clamp on the per-level linear grow factor (area = factor²), so a group that
   *  collapses to a couple of islands can't blow a survivor up unboundedly,
   *  default 2.5 */
  growMax?: number;
  /** survivor growth shape (grass arc S2):
   *  'uniform' (default) — scale the island uniformly about its centroid (leaf
   *    crowns: quads legitimately get bigger in every axis).
   *  'widen' — scale ONLY x/z about the centroid, y untouched (grass blades:
   *    the ring's thin×widen law widens survivors to conserve COVERAGE; uniform
   *    growth stacked ~×2^L into multi-metre blade monsters at coarse levels —
   *    the ravine-wall columns, user 2026-07-03). Area conservation then rides
   *    the width factor alone (= g², clamped at growMax²). */
  growMode?: 'uniform' | 'widen';
}

type FullOpts = Required<AggregateDagOpts>;

/** ?leaflodk= — SCALE on the aggregate ladder's reported error (growError). The measured
 *  2026-07-01 disease: growError for L1 is small enough that its cut distance (a node EMITS
 *  when projK·err/d ≤ τ, i.e. beyond d₁ = projK·err/τ ≈ 57 m at retina) lands BEYOND the
 *  35 m mesh→voxel handoff, so the leaf-mesh band renders LOD0 everywhere → ~10.3M of the
 *  12.4M eye-pose visTris. errorK<1 SHRINKS the reported error, pulling each level's
 *  engagement K× nearer (L1 at ~57·K m — e.g. K=0.25 → ~14 m), engaging real coarsening
 *  INSIDE the band; K>1 pushes detail farther (more tris). Baked at DAG build; set BEFORE
 *  buildAggregateDag (the scenes wire ?leaflodk=). DEFAULT 0.4 (2026-07-03, was 1 — the
 *  2026-07-02c forest pick, now SHARED forest+world): hides the user-reported spiky
 *  coarse crowns at the band end at neutral perf (deeper 0.25 traded oblique +7 ms). */
const AGG_LOD_CFG = { errorK: 0.4 };
export function setAggLodErrorK(k: number): void {
  if (Number.isFinite(k) && k > 0) AGG_LOD_CFG.errorK = k;
}

/** the per-level decision + grown geometry the group loop consumes. Islands are
 *  pre-grown into `gverts`; `keepTri` already folds in each tri's island keep
 *  flag — so the group loop never touches island bookkeeping. */
interface LevelSoup {
  /** grown welded soup vertices (stride floats each), survivors transformed in place */
  gverts: Float64Array;
  nLocal: number;
  /** per soup-tri local vertex ids */
  triA: Int32Array;
  triB: Int32Array;
  triC: Int32Array;
  /** per soup-tri: is its island kept this level? */
  keepTri: Uint8Array;
  /** cluster id → [first soup tri, count] in the arrays above (-1 = not active) */
  soupStart: Int32Array;
  soupCount: Int32Array;
  /** representative grown-leaf radius (metres) = this level's geometric error proxy */
  growError: number;
}

const HASH_QUANT = 1000; // 1 mm centroid quantum for the stable per-island hash

/**
 * One LOD level: weld ALL active clusters into one global soup, find leaf islands
 * (connected components), stochastically remove ~(1-targetRatio) of the leaf AREA
 * (seed-deterministic), and GROW every survivor about its own centroid so the
 * total surviving area is restored (area-preserving). Returns the grown soup +
 * per-cluster tri ranges for the group loop to re-cluster into parents.
 */
function aggregateLevel(
  active: number[],
  clusters: DagCluster[],
  pool: Float32Array,
  poolIdx: Uint32Array,
  stride: number,
  o: FullOpts,
): LevelSoup {
  // ---- weld every active triangle into one position-keyed soup ---------------
  const inv = 1 / o.weldEps;
  let totalTris = 0;
  for (const id of active) totalTris += (clusters[id] as DagCluster).triCount;
  const maxVerts = Math.max(totalTris * 3, 1);
  let hcap = 16;
  while (hcap < maxVerts * 2) hcap *= 2;
  const hmask = hcap - 1;
  const hslot = new Int32Array(hcap).fill(-1);
  const qx = new Int32Array(maxVerts);
  const qy = new Int32Array(maxVerts);
  const qz = new Int32Array(maxVerts);
  const gverts = new Float64Array(maxVerts * stride);
  const triA = new Int32Array(totalTris);
  const triB = new Int32Array(totalTris);
  const triC = new Int32Array(totalTris);
  let nLocal = 0;
  const localOf = (vid: number): number => {
    const p = vid * stride;
    const ix = Math.round((pool[p] as number) * inv);
    const iy = Math.round((pool[p + 1] as number) * inv);
    const iz = Math.round((pool[p + 2] as number) * inv);
    let h = (Math.imul(ix, 73856093) ^ Math.imul(iy, 19349663) ^ Math.imul(iz, 83492791)) & hmask;
    for (;;) {
      const lid = hslot[h] as number;
      if (lid === -1) {
        const id = nLocal++;
        const b = id * stride;
        for (let s = 0; s < stride; s++) gverts[b + s] = pool[p + s] as number;
        qx[id] = ix;
        qy[id] = iy;
        qz[id] = iz;
        hslot[h] = id;
        return id;
      }
      if ((qx[lid] as number) === ix && (qy[lid] as number) === iy && (qz[lid] as number) === iz) return lid;
      h = (h + 1) & hmask;
    }
  };

  // soup tris, and the contiguous soup-tri range each active cluster contributes
  const soupStart = new Int32Array(clusters.length).fill(-1);
  const soupCount = new Int32Array(clusters.length);
  let nTris = 0;
  for (const id of active) {
    const c = clusters[id] as DagCluster;
    soupStart[id] = nTris;
    for (let t = c.triStart; t < c.triStart + c.triCount; t++) {
      const a = localOf(poolIdx[t * 3] as number);
      const b = localOf(poolIdx[t * 3 + 1] as number);
      const cc = localOf(poolIdx[t * 3 + 2] as number);
      triA[nTris] = a;
      triB[nTris] = b;
      triC[nTris] = cc;
      nTris++;
    }
    soupCount[id] = nTris - (soupStart[id] as number);
  }

  // ---- connected-component islands (union-find over the soup edges) -----------
  const parent = new Int32Array(nLocal);
  for (let i = 0; i < nLocal; i++) parent[i] = i;
  const find = (x: number): number => {
    let r = x;
    while ((parent[r] as number) !== r) r = parent[r] = parent[parent[r] as number] as number;
    return r;
  };
  const union = (a: number, b: number): void => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[ra] = rb;
  };
  for (let t = 0; t < nTris; t++) {
    const a = triA[t] as number;
    const b = triB[t] as number;
    const c = triC[t] as number;
    union(a, b);
    union(b, c);
  }
  // dense island ids + per-vertex island lookup
  const islandOfVert = new Int32Array(nLocal).fill(-1);
  const rootToIsland = new Map<number, number>();
  let nIslands = 0;
  for (let v = 0; v < nLocal; v++) {
    const r = find(v);
    let isl = rootToIsland.get(r);
    if (isl === undefined) {
      isl = nIslands++;
      rootToIsland.set(r, isl);
    }
    islandOfVert[v] = isl;
  }

  // ---- per-island area + area-weighted centroid ------------------------------
  const area = new Float64Array(nIslands);
  const cenX = new Float64Array(nIslands);
  const cenY = new Float64Array(nIslands);
  const cenZ = new Float64Array(nIslands);
  for (let t = 0; t < nTris; t++) {
    const a = triA[t] as number;
    const b = triB[t] as number;
    const c = triC[t] as number;
    const ax = gverts[a * stride] as number;
    const ay = gverts[a * stride + 1] as number;
    const az = gverts[a * stride + 2] as number;
    const e1x = (gverts[b * stride] as number) - ax;
    const e1y = (gverts[b * stride + 1] as number) - ay;
    const e1z = (gverts[b * stride + 2] as number) - az;
    const e2x = (gverts[c * stride] as number) - ax;
    const e2y = (gverts[c * stride + 1] as number) - ay;
    const e2z = (gverts[c * stride + 2] as number) - az;
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    const triArea = 0.5 * Math.sqrt(nx * nx + ny * ny + nz * nz);
    const isl = islandOfVert[a] as number;
    const cx = (ax + (gverts[b * stride] as number) + (gverts[c * stride] as number)) / 3;
    const cy = (ay + (gverts[b * stride + 1] as number) + (gverts[c * stride + 1] as number)) / 3;
    const cz = (az + (gverts[b * stride + 2] as number) + (gverts[c * stride + 2] as number)) / 3;
    area[isl] = (area[isl] as number) + triArea;
    cenX[isl] = (cenX[isl] as number) + triArea * cx;
    cenY[isl] = (cenY[isl] as number) + triArea * cy;
    cenZ[isl] = (cenZ[isl] as number) + triArea * cz;
  }
  let totalArea = 0;
  for (let i = 0; i < nIslands; i++) {
    const w = (area[i] as number) || 1e-20;
    cenX[i] = (cenX[i] as number) / w;
    cenY[i] = (cenY[i] as number) / w;
    cenZ[i] = (cenZ[i] as number) / w;
    totalArea += area[i] as number;
  }
  // per-island radius (max centroid→vertex distance) for the error proxy
  const radius = new Float64Array(nIslands);
  for (let v = 0; v < nLocal; v++) {
    const isl = islandOfVert[v] as number;
    const dx = (gverts[v * stride] as number) - (cenX[isl] as number);
    const dy = (gverts[v * stride + 1] as number) - (cenY[isl] as number);
    const dz = (gverts[v * stride + 2] as number) - (cenZ[isl] as number);
    const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (d > (radius[isl] as number)) radius[isl] = d;
  }

  // ---- seed-deterministic removal: keep the low-hash prefix until kept area
  //      reaches targetRatio·total; the suffix is removed (uniform spatial thin) -
  const order = Array.from({ length: nIslands }, (_, i) => i);
  const hashOf = (isl: number): number => {
    const ix = Math.round((cenX[isl] as number) * HASH_QUANT);
    const iy = Math.round((cenY[isl] as number) * HASH_QUANT);
    const iz = Math.round((cenZ[isl] as number) * HASH_QUANT);
    const h =
      Math.imul(ix, 73856093) ^
      Math.imul(iy, 19349663) ^
      Math.imul(iz, 83492791) ^
      Math.imul(o.seed | 0, 2654435761);
    return h >>> 0;
  };
  order.sort((a, b) => {
    const ha = hashOf(a);
    const hb = hashOf(b);
    return ha !== hb ? ha - hb : a - b;
  });
  const keepIsland = new Uint8Array(nIslands);
  const target = o.targetRatio * totalArea;
  let keptArea = 0;
  for (const isl of order) {
    keepIsland[isl] = 1;
    keptArea += area[isl] as number;
    if (keptArea >= target) break; // enough silhouette mass retained; drop the tail
  }

  // ---- grow survivors about their centroid to restore total area -------------
  //  uniform linear factor g with g²·keptArea = totalArea ⇒ area preserved. Clamp
  //  so a near-empty group can't explode a survivor; growError = the resulting
  //  representative leaf radius (the feature size that can no longer be resolved).
  let g = keptArea > 1e-20 ? Math.sqrt(totalArea / keptArea) : 1;
  if (g < 1) g = 1;
  if (g > o.growMax) g = o.growMax;
  let radiusSum = 0;
  let keptCount = 0;
  for (let i = 0; i < nIslands; i++) {
    if (keepIsland[i] !== 1) continue;
    radiusSum += radius[i] as number;
    keptCount++;
  }
  const growError = keptCount > 0 ? (radiusSum / keptCount) * g : 0;

  // 'widen' (grass): y untouched; the full area compensation rides x/z (factor
  // g² = area ratio, clamped at growMax²) — the ring's thin×widen conservation.
  const widen = o.growMode === 'widen';
  const gxz = widen ? Math.min(g * g, o.growMax * o.growMax) : g;
  const gy = widen ? 1 : g;
  for (let v = 0; v < nLocal; v++) {
    const isl = islandOfVert[v] as number;
    if (keepIsland[isl] !== 1) continue; // dropped island — verts never emitted
    const b = v * stride;
    gverts[b] = (cenX[isl] as number) + ((gverts[b] as number) - (cenX[isl] as number)) * gxz;
    gverts[b + 1] = (cenY[isl] as number) + ((gverts[b + 1] as number) - (cenY[isl] as number)) * gy;
    gverts[b + 2] = (cenZ[isl] as number) + ((gverts[b + 2] as number) - (cenZ[isl] as number)) * gxz;
  }

  // per-tri keep flag (folds in the island decision once, for the group loop)
  const keepTri = new Uint8Array(nTris);
  for (let t = 0; t < nTris; t++) keepTri[t] = keepIsland[islandOfVert[triA[t] as number] as number] as number;

  return { gverts, nLocal, triA, triB, triC, keepTri, soupStart, soupCount, growError };
}

/**
 * Build the area-preserving aggregate LOD DAG for one foliage mesh. Returns the
 * SAME DagBuild contract as buildDag (verts/indices/clusters/groups/levelStats),
 * so the registry attaches it identically at N9-C2.
 */
export function buildAggregateDag(
  verts: Float32Array,
  vertStride: number,
  indices: Uint32Array,
  opts: AggregateDagOpts = {},
): DagBuild {
  const t0 = now();
  const o: FullOpts = {
    maxTris: opts.maxTris ?? 128,
    targetRatio: opts.targetRatio ?? 0.5,
    groupMax: opts.groupMax ?? 24,
    stuckFrac: opts.stuckFrac ?? 0.15,
    maxLevels: opts.maxLevels ?? 24,
    weldEps: opts.weldEps ?? 1e-5,
    seed: opts.seed ?? 0,
    growMax: opts.growMax ?? 2.5,
    growMode: opts.growMode ?? 'uniform',
  };

  const built = clusterize(verts, vertStride, indices, o.maxTris);

  // growing geometry pools (capacity-doubling typed arrays; exact slice to finish)
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

  // -- LOD0 clusters: ownError 0, ownSphere = geometric sphere -----------------
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
  // per-group vertex remap (version-stamped to avoid clearing nLocal each group)
  let vstamp = new Int32Array(0);
  let vremap = new Int32Array(0);
  let stampGen = 0;

  for (let level = 0; level < o.maxLevels; level++) {
    if (active.length < 2) break;
    const soup = aggregateLevel(active, clusters, pv, pi, vertStride, o);
    if (vstamp.length < soup.nLocal) {
      vstamp = new Int32Array(soup.nLocal);
      vremap = new Int32Array(soup.nLocal);
    }
    const partition = partitionClusters(active, clusters, o.groupMax);

    let outClusters = 0;
    let outTris = 0;
    let inTris = 0;
    let stuckGroups = 0;
    const nextActive: number[] = [];
    for (const member of active) inTris += (clusters[member] as DagCluster).triCount;

    for (const groupIds of partition) {
      const gid = groups.length;

      // child own-error max + own-sphere union (containment)
      let childErr = 0;
      let inGroupTris = 0;
      let sphere: Sphere | null = null;
      for (const id of groupIds) {
        const c = clusters[id] as DagCluster;
        if (c.ownError > childErr) childErr = c.ownError;
        inGroupTris += c.triCount;
        const s: Sphere = { x: c.oex, y: c.oey, z: c.oez, r: c.oer };
        sphere = sphere ? mergeSpheres(sphere, s) : s;
      }
      let gs = sphere ?? { x: 0, y: 0, z: 0, r: 0 };

      // gather this group's KEPT (grown) soup tris into a compact local mesh
      stampGen++;
      const groupVerts = new Float32Array(inGroupTris * 3 * vertStride);
      const groupIdx = new Uint32Array(inGroupTris * 3);
      let gv = 0;
      let giN = 0;
      const emit = (lv: number): void => {
        if ((vstamp[lv] as number) !== stampGen) {
          vstamp[lv] = stampGen;
          vremap[lv] = gv;
          const o2 = gv * vertStride;
          const s2 = lv * vertStride;
          for (let s = 0; s < vertStride; s++) groupVerts[o2 + s] = soup.gverts[s2 + s] as number;
          gv++;
        }
        groupIdx[giN++] = vremap[lv] as number;
      };
      for (const id of groupIds) {
        const s0 = soup.soupStart[id] as number;
        const n0 = soup.soupCount[id] as number;
        for (let k = 0; k < n0; k++) {
          const t = s0 + k;
          if (soup.keepTri[t] !== 1) continue;
          emit(soup.triA[t] as number);
          emit(soup.triB[t] as number);
          emit(soup.triC[t] as number);
        }
      }
      const outSoupTris = giN / 3;

      const reduction = inGroupTris > 0 ? 1 - outSoupTris / inGroupTris : 0;
      const stuck = reduction < o.stuckFrac || outSoupTris < 1;
      if (stuck) {
        // group can't thin at this ratio → its inputs become roots (parentError +∞)
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

      // group error: strictly monotone above every child's own error. ?leaflodk= scales the
      // geometric term so the cut engages K× nearer (in-band leaf coarsening — see AGG_LOD_CFG).
      const groupErr = Math.max(soup.growError * AGG_LOD_CFG.errorK, childErr * (1 + 1e-6) + 1e-7);
      if (groupErr > maxError) maxError = groupErr;

      // append the grown soup, re-clusterize into PARENTS
      const vertBase = pvLen / vertStride;
      const gvFloats = gv * vertStride;
      growV(gvFloats);
      pv.set(groupVerts.subarray(0, gvFloats), pvLen);
      pvLen += gvFloats;
      const parentBuilt = clusterize(groupVerts.subarray(0, gvFloats), vertStride, groupIdx.subarray(0, giN), o.maxTris);

      // grow can push survivors OUTSIDE the input union sphere → expand gs to
      // contain the actual parent geometry, preserving projection-monotone
      // containment (parentSphere ⊇ ownSphere for every input AND every parent).
      for (let c = 0; c < parentBuilt.clusterCount; c++) {
        gs = mergeSpheres(gs, {
          x: parentBuilt.sphere[c * 4] as number,
          y: parentBuilt.sphere[c * 4 + 1] as number,
          z: parentBuilt.sphere[c * 4 + 2] as number,
          r: parentBuilt.sphere[c * 4 + 3] as number,
        });
      }

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
          // parent OWN pair == group pair (exact); parent's own parent is TBD
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

    if (nextActive.length === 0) break; // all groups stuck → roots stand
    if (nextActive.length >= active.length) break; // no progress → stop
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
