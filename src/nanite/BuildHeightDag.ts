/**
 * Terrain heightfield LOD DAG (NANITE.md N8-D2, D-N32) — heightfield-NATIVE
 * adaptive builder. Hand-rolled, CPU/typed-array, node-runnable (no three.js,
 * no GPU) for the validation probe (tools/probe-heightdag.ts).
 *
 * Terrain is a REGULAR grid: connectivity is implicit and per-vertex error is
 * pure VERTICAL deviation, so the right decimation is grid-native — keep every
 * survivor ON the grid (F4: positions reconstruct from the heights buffer; we
 * store compact grid-coord connectivity, never baked floats). We get there by
 * driving BuildDag's PROVEN crack-free cluster-DAG scaffolding (group → lock the
 * group boundary → simplify interior → re-cluster parents → exact sibling
 * (error,sphere) sharing → flat runtime cut) with its QEM collapse restricted to
 * grid ENDPOINTS (`gridEndpoint`). On a heightfield the QEM cost equals the
 * vertical error, so a flat plain (≈0 cost) collapses to a few big triangles
 * while a cliff/erosion-channel (high cost) stays dense — the "non-naive
 * optimizer" mandate, adaptive for free. Crack-freeness is INHERITED bit-for-bit
 * from BuildDag (siblings share the parent pair exactly ⇒ the cut falls between
 * groups where boundary verts were locked), so terrain rides the SAME
 * kClusterCull cut as rock/bark — unified runtime, specialised build.
 *
 * WHY this and not a pure martini/RTIN getMesh per error band: martini is
 * crack-free only under a per-FRAME bintree traversal (ROAM). Baked into a flat
 * per-cluster cut (D-N31 — terrain must use the same cut, no per-frame
 * traversal) independent per-band meshes T-junction at the cut frontier where
 * adjacent regions at different distances pick different bands. Feeding the
 * vertical-error metric into BuildDag's locked-boundary scaffolding is the only
 * construction that is BOTH crack-free under the flat cut AND adaptive.
 *
 * Build cost: this reuses BuildDag's iterative QEM heap (per-collapse quadric
 * reseed). That is fine for the node probe and modest fields; for the full
 * 4096² field the O(n) martini error PYRAMID (precomputed per-vertex vertical
 * error, no heap reseed) is the speed path and the background Worker (D-N30)
 * keeps it off the boot critical path — both are D2b concerns. D2a proves
 * correctness (invariants), F4 (on-grid survivors), and adaptivity here.
 *
 * RNG-free → deterministic by construction (given the heights).
 */

import { type DagCluster, type DagGroup, type DagLevelStats, buildDag } from './BuildDag';

/** the heightfield to DAG: row-major (gridN+1)² heights over a regular grid. */
export interface HeightField {
  /** (gridN+1)*(gridN+1) row-major heights (metres); vertex (gx,gz) at gz*(gridN+1)+gx */
  heights: Float32Array;
  /** quads per axis (vertices = gridN+1 per axis) */
  gridN: number;
  /** world units per quad */
  cellSize: number;
  /** world X of grid vertex (0,0) */
  originX: number;
  /** world Z of grid vertex (0,0) */
  originZ: number;
}

export interface HeightDagOpts {
  /** cluster triangle cap (matches Clusterize / GPU), default 128 */
  maxTris?: number;
  /** max clusters per group, default 24 */
  groupMax?: number;
  /** a group reducing fewer than this fraction of its tris (at the current error
   *  band) is stuck → DEFERS to the next band, default 0.15 */
  stuckFrac?: number;
  /** safety cap on DAG levels, default 24 */
  maxLevels?: number;
  /** the FINEST error band (metres of vertical deviation). Level ℓ's budget is
   *  baseError·2^ℓ, so a vertex is removed at the band where the surface error of
   *  omitting it first fits. Smaller ⇒ LOD0 stays finer near the camera. Default
   *  0.05 m. */
  baseError?: number;
}

/**
 * Martini-style per-grid-vertex VERTICAL error pyramid (Mapbox `martini`,
 * Evans/Kirkpatrick/Sloan RTIN), bottom-up O(n): each hypotenuse-midpoint vertex
 * gets max(|its own interpolation error|, its two children's errors). errs[v] =
 * the height deviation introduced by OMITTING v — exactly the decimation cost the
 * cut projects. Requires a (2^k+1)² grid (gridN a power of two).
 */
function martiniErrors(heights: Float32Array, gridN: number): Float64Array {
  const size = gridN + 1;
  const errs = new Float64Array(size * size);
  const numSmallest = gridN * gridN;
  const numTris = numSmallest * 2 - 2;
  const numParent = numTris - numSmallest;
  for (let i = numTris - 1; i >= 0; i--) {
    let id = i + 2;
    let ax = 0;
    let ay = 0;
    let bx = 0;
    let by = 0;
    let cx = 0;
    let cy = 0;
    if (id & 1) {
      bx = by = cx = gridN; // bottom-right root triangle (a=0,0; c=gridN,0)
    } else {
      ax = ay = cx = gridN; // top-left root triangle
    }
    while ((id >>= 1) > 1) {
      const mx = (ax + bx) >> 1;
      const my = (ay + by) >> 1;
      if (id & 1) {
        bx = ax;
        by = ay;
        ax = cx;
        ay = cy;
      } else {
        ax = bx;
        ay = by;
        bx = cx;
        by = cy;
      }
      cx = mx;
      cy = my;
    }
    const interp = ((heights[ay * size + ax] as number) + (heights[by * size + bx] as number)) / 2;
    const mi = ((ay + by) >> 1) * size + ((ax + bx) >> 1);
    const mErr = Math.abs(interp - (heights[mi] as number));
    let e = errs[mi] as number;
    if (mErr > e) e = mErr;
    if (i < numParent) {
      const lc = ((ay + cy) >> 1) * size + ((ax + cx) >> 1);
      const rc = ((by + cy) >> 1) * size + ((bx + cx) >> 1);
      const le = errs[lc] as number;
      const re = errs[rc] as number;
      if (le > e) e = le;
      if (re > e) e = re;
    }
    errs[mi] = e;
  }
  return errs;
}

export interface HeightDagStats {
  buildMs: number;
  levels: number;
  lod0Clusters: number;
  lod0Tris: number;
  totalClusters: number;
  totalTris: number;
  roots: number;
  /** max own-error across the DAG (metres, vertical) */
  maxError: number;
  /** vertices that round-tripped OFF the grid (must be 0 — F4 / on-grid invariant) */
  offGridVerts: number;
  /** worst |recovered - exact| grid-coord round-trip residual (in cells) */
  maxGridResidual: number;
}

/** the terrain DAG: same cluster/group cut metadata as BuildDag, but the vertex
 *  pool is packed GRID COORDS (gx | gz<<16), not floats — heights fetch on the GPU. */
export interface HeightDagBuild {
  /** packed (gx | (gz<<16)) per output vertex; world pos = (gx*cell+oX, height(gx,gz), gz*cell+oZ) */
  gridVerts: Uint32Array;
  /** triangle indices into gridVerts; cluster c owns tris [triStart, +triCount) */
  indices: Uint32Array;
  clusters: DagCluster[];
  groups: DagGroup[];
  levelStats: DagLevelStats[];
  lod0Count: number;
  stats: HeightDagStats;
  /** grid params needed to reconstruct world positions / register the mesh */
  gridN: number;
  cellSize: number;
  originX: number;
  originZ: number;
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

/** index of grid vertex (gx,gz) into a row-major (gridN+1)² array */
function vidx(gx: number, gz: number, gridN: number): number {
  return gz * (gridN + 1) + gx;
}

/**
 * Build the full-resolution LOD0 grid mesh (every cell → 2 right triangles,
 * winding matched to the existing GPU heightfield path: even (0,0)(0,1)(1,1),
 * odd (0,0)(1,1)(1,0), up-facing +Y). Positions are interleaved stride-3 floats.
 */
function lod0Grid(hf: HeightField): { verts: Float32Array; indices: Uint32Array } {
  const { gridN, cellSize, originX, originZ, heights } = hf;
  const vpa = gridN + 1; // vertices per axis
  const verts = new Float32Array(vpa * vpa * 3);
  for (let gz = 0; gz <= gridN; gz++) {
    for (let gx = 0; gx <= gridN; gx++) {
      const i = vidx(gx, gz, gridN);
      verts[i * 3] = gx * cellSize + originX;
      verts[i * 3 + 1] = heights[i] as number;
      verts[i * 3 + 2] = gz * cellSize + originZ;
    }
  }
  const indices = new Uint32Array(gridN * gridN * 6);
  let o = 0;
  for (let gz = 0; gz < gridN; gz++) {
    for (let gx = 0; gx < gridN; gx++) {
      const v00 = vidx(gx, gz, gridN);
      const v10 = vidx(gx + 1, gz, gridN);
      const v01 = vidx(gx, gz + 1, gridN);
      const v11 = vidx(gx + 1, gz + 1, gridN);
      // even tri (0,0)(0,1)(1,1)
      indices[o++] = v00;
      indices[o++] = v01;
      indices[o++] = v11;
      // odd tri (0,0)(1,1)(1,0)
      indices[o++] = v00;
      indices[o++] = v11;
      indices[o++] = v10;
    }
  }
  return { verts, indices };
}

export function buildHeightDag(hf: HeightField, opts: HeightDagOpts = {}): HeightDagBuild {
  const t0 = now();
  const { gridN, cellSize, originX, originZ } = hf;
  if (gridN < 2 || (gridN & (gridN - 1)) !== 0) {
    throw new Error(`buildHeightDag: gridN must be a power of two (RTIN error pyramid), got ${gridN}`);
  }
  const baseError = opts.baseError ?? 0.05;

  // O(n) martini vertical-error pyramid: errs[v] = the height error of OMITTING v.
  const errs = martiniErrors(hf.heights, gridN);
  const size = gridN + 1;
  const invCell = 1 / cellSize;
  const gridErrAt = (x: number, z: number): number => {
    let gx = Math.round((x - originX) * invCell);
    let gz = Math.round((z - originZ) * invCell);
    gx = gx < 0 ? 0 : gx > gridN ? gridN : gx;
    gz = gz < 0 ? 0 : gz > gridN ? gridN : gz;
    return errs[gz * size + gx] as number;
  };
  // doubling error bands: a vertex collapses at the first level whose budget ≥ its
  // omission error → flat plains at the lowest band, cliffs progressively after.
  const levelBudget = (level: number): number => baseError * 2 ** level;

  // LOD0 = full grid (ownError 0 ⇒ no holes up close, even on cliffs). The DAG
  // decimates upward error-bounded: flat groups collapse at low bands, cliffs
  // defer to higher bands → adaptive, smooth ±1 cut.
  const { verts, indices } = lod0Grid(hf);

  const dag = buildDag(verts, 3, indices, {
    maxTris: opts.maxTris ?? 128,
    groupMax: opts.groupMax ?? 24,
    stuckFrac: opts.stuckFrac ?? 0.15,
    maxLevels: opts.maxLevels ?? 24,
    normalOffset: -1, // terrain stride-3: no per-vertex normal (normalTex on GPU)
    gridEndpoint: true, // F4: every survivor stays on the heightfield grid
    gridErrAt, // martini vertical error drives + bounds the decimation
    levelBudget,
  });

  // -- recover grid coords from the (on-grid) survivor positions → pack --------
  const nv = dag.verts.length / dag.vertStride;
  const gridVerts = new Uint32Array(nv);
  let offGridVerts = 0;
  let maxGridResidual = 0;
  for (let i = 0; i < nv; i++) {
    const x = dag.verts[i * dag.vertStride] as number;
    const z = dag.verts[i * dag.vertStride + 2] as number;
    const fx = (x - originX) * invCell;
    const fz = (z - originZ) * invCell;
    const gx = Math.round(fx);
    const gz = Math.round(fz);
    const res = Math.max(Math.abs(fx - gx), Math.abs(fz - gz));
    if (res > maxGridResidual) maxGridResidual = res;
    if (res > 1e-3 || gx < 0 || gx > gridN || gz < 0 || gz > gridN) offGridVerts++;
    const cgx = gx < 0 ? 0 : gx > gridN ? gridN : gx;
    const cgz = gz < 0 ? 0 : gz > gridN ? gridN : gz;
    gridVerts[i] = (cgx & 0xffff) | ((cgz & 0xffff) << 16);
  }

  return {
    gridVerts,
    indices: dag.indices,
    clusters: dag.clusters,
    groups: dag.groups,
    levelStats: dag.levelStats,
    lod0Count: dag.lod0Count,
    stats: {
      buildMs: now() - t0,
      levels: dag.stats.levels,
      lod0Clusters: dag.stats.lod0Clusters,
      lod0Tris: dag.stats.lod0Tris,
      totalClusters: dag.stats.totalClusters,
      totalTris: dag.stats.totalTris,
      roots: dag.stats.roots,
      maxError: dag.stats.maxError,
      offGridVerts,
      maxGridResidual,
    },
    gridN,
    cellSize,
    originX,
    originZ,
  };
}
