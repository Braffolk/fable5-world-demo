/**
 * Terrain heightfield LOD DAG (NANITE-SPEC.md N8-D2, D-N32) — heightfield-NATIVE
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

/**
 * N8-D2 Stage 2d — skirt depth in WORLD metres for a clipmap level-k tile:
 *     depth(k) = SKIRT_DEPTH_A + SKIRT_DEPTH_B · k.
 * The inter-level T-junction crack is the coarse-neighbour's vertical deviation at
 * stride 2^k; on real eroded terrain (cliffs) that grows with k but SATURATES
 * (measured ~16→26→30→44 m for k=0..3, tools/probe-skirtgap.ts), so a LINEAR depth
 * hugs it far better than ∝2^k would (which starves the fine transitions and wildly
 * over-covers the coarse ones → skirt walls). These coefficients seal every measured
 * transition with a 7-18 m margin; bump them if probe-skirtgap reports a CRACK (a
 * steeper seed). Both the BUILD (skirt bounding sphere) and the GPU (NaniteFetch
 * drops a flagged skirt vert) read them, so they agree. The depth-LEVEL is encoded
 * as a 3-bit code (level+1) in bits 13-15 of a skirt vert's word0 — free because a
 * tile texel coord ≤4095 uses only bits 0-11. */
export const SKIRT_DEPTH_A = 24;
export const SKIRT_DEPTH_B = 12;
/** the 3-bit skirt code (level+1) caps the encodable clipmap level at 6 (code ≤7). */
const SKIRT_MAX_LEVEL = 6;

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
  /** N8-D2 Stage 2d: when ≥0, append a DOUBLE-SIDED perimeter SKIRT at this clipmap
   *  level (0 = finest) to seal inter-level T-junction cracks. The curtain drops by
   *  SKIRT_BASE_WORLD·2^level (∝ the ring's crack size). Undefined/<0 = no skirt
   *  (default) ⇒ the headless probe + uniform-tile path stay byte-for-byte unchanged. */
  skirtLevel?: number;
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

/** one always-on skirt cluster: level-0, ownError 0, parentError +∞ (→ drawn at
 *  EVERY runtime cut), cone disabled (ccos −1 — heightfields skip cone cull anyway,
 *  NaniteCull). Sphere == own/parent sphere (the root branch in attachHeightDagTile
 *  copies own→parent). */
function skirtCluster(
  cx: number,
  cy: number,
  cz: number,
  sr: number,
  triStart: number,
  triCount: number,
): DagCluster {
  return {
    level: 0,
    triStart,
    triCount,
    sx: cx,
    sy: cy,
    sz: cz,
    sr,
    cax: 0,
    cay: 1,
    caz: 0,
    ccos: -1,
    ownError: 0,
    oex: cx,
    oey: cy,
    oez: cz,
    oer: sr,
    parentError: Infinity,
    pex: cx,
    pey: cy,
    pez: cz,
    per: sr,
    groupAsInput: -1,
    groupAsParent: -1,
  };
}

/**
 * N8-D2 Stage 2d — build a self-contained, ALWAYS-ON perimeter SKIRT for one tile.
 * Clipmap levels ABUT at 2× stride, so a fine tile's edge has 2× the verts of the
 * coarse tile it meets ⇒ T-junction cracks that show SKY. We hang a vertical curtain
 * from every BASE-STRIDE perimeter vertex (buildDag locks tile borders at all LODs,
 * so the surface edge passes through them at every cut): the TOP verts (code 0) sit
 * exactly on the surface edge; the BOTTOM verts carry a 3-bit depth code in bits
 * 13-15 of word0 so the GPU (NaniteFetch) drops them by SKIRT_BASE_WORLD·2^level.
 * DOUBLE-SIDED — the SW raster backface-culls by winding (NaniteRaster) and the
 * camera can graze a boundary from either side, so each segment emits 4 tris and the
 * area test renders whichever pair faces the camera (≈0 raster cost; the back pair is
 * always culled). Verts/tris index into the COMBINED tile arrays: `vertBase` =
 * surface vert count, `triBase` = surface tri count. Spheres are WORLD-space (match
 * buildDag's). */
function buildPerimeterSkirt(
  hf: HeightField,
  level: number,
  maxTris: number,
  vertBase: number,
  triBase: number,
): { packed: number[]; tris: number[]; clusters: DagCluster[] } {
  const { gridN, cellSize, originX, originZ, heights } = hf;
  const lv = level < 0 ? 0 : level > SKIRT_MAX_LEVEL ? SKIRT_MAX_LEVEL : level;
  const depth = SKIRT_DEPTH_A + SKIRT_DEPTH_B * lv;
  const code = (lv + 1) & 0x7; // bits 13-15; 0 means "surface vert" (no drop)
  const size = gridN + 1;
  const packed: number[] = [];
  const tris: number[] = [];
  const clusters: DagCluster[] = [];
  const maxSeg = Math.max(1, maxTris >> 2); // 4 tris per double-sided segment

  // the in-progress cluster's WORLD AABB (top at h, bottom at h−depth) → sphere
  let cStart = triBase;
  let cSeg = 0;
  let nx = Infinity;
  let ny = Infinity;
  let nz = Infinity;
  let xx = -Infinity;
  let xy = -Infinity;
  let xz = -Infinity;
  const grow = (x: number, h: number, z: number): void => {
    if (x < nx) nx = x;
    if (x > xx) xx = x;
    if (z < nz) nz = z;
    if (z > xz) xz = z;
    if (h > xy) xy = h;
    if (h - depth < ny) ny = h - depth;
  };
  const flush = (): void => {
    if (cSeg === 0) return;
    const cx = (nx + xx) / 2;
    const cy = (ny + xy) / 2;
    const cz = (nz + xz) / 2;
    const sr = Math.hypot(xx - cx, xy - cy, xz - cz);
    const end = triBase + tris.length / 3;
    clusters.push(skirtCluster(cx, cy, cz, sr, cStart, end - cStart));
    cStart = end;
    cSeg = 0;
    nx = ny = nz = Infinity;
    xx = xy = xz = -Infinity;
  };

  // walk the 4 edges; corner verts are duplicated across adjacent edges (a coincident
  // overlap, never a gap). Winding is irrelevant (double-sided), so all step the same.
  const edges: ReadonlyArray<readonly [number, number, number, number]> = [
    [0, 0, 1, 0], // south:  (0,0)→(gridN,0)
    [gridN, 0, 0, 1], // east:   (gridN,0)→(gridN,gridN)
    [gridN, gridN, -1, 0], // north:  (gridN,gridN)→(0,gridN)
    [0, gridN, 0, -1], // west:   (0,gridN)→(0,0)
  ];
  for (const [ex, ez, dx, dz] of edges) {
    let pTop = -1;
    let pBot = -1;
    let pwx = 0;
    let pwz = 0;
    let ph = 0;
    for (let i = 0; i <= gridN; i++) {
      const gx = ex + dx * i;
      const gz = ez + dz * i;
      const h = heights[gz * size + gx] as number;
      const wx = gx * cellSize + originX;
      const wz = gz * cellSize + originZ;
      const top = vertBase + packed.length;
      packed.push(((gx & 0x1fff) | ((gz & 0xffff) << 16)) >>> 0);
      const bot = vertBase + packed.length;
      packed.push(((gx & 0x1fff) | (code << 13) | ((gz & 0xffff) << 16)) >>> 0);
      if (i > 0) {
        grow(pwx, ph, pwz);
        grow(wx, h, wz);
        // quad (pTop,pBot,bot,top): 2 front tris + 2 reversed back tris
        tris.push(pTop, pBot, bot, pTop, bot, top);
        tris.push(pTop, bot, pBot, pTop, top, bot);
        cSeg++;
        if (cSeg >= maxSeg) flush();
      }
      pTop = top;
      pBot = bot;
      pwx = wx;
      pwz = wz;
      ph = h;
    }
    flush(); // close at the edge end so clusters stay edge-local
  }
  return { packed, tris, clusters };
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

  // N8-D2 Stage 2d: append a self-contained perimeter skirt (seals inter-level
  // clipmap T-junctions). Off (skirtLevel undef/<0) ⇒ the arrays are unchanged, so
  // the headless probe + uniform path are byte-identical to before.
  let outVerts = gridVerts;
  let outIndices = dag.indices;
  let outClusters = dag.clusters;
  let skirtClusters = 0;
  let skirtTris = 0;
  if (opts.skirtLevel != null && opts.skirtLevel >= 0) {
    const sk = buildPerimeterSkirt(hf, opts.skirtLevel, opts.maxTris ?? 128, nv, dag.indices.length / 3);
    outVerts = new Uint32Array(nv + sk.packed.length);
    outVerts.set(gridVerts, 0);
    outVerts.set(sk.packed, nv);
    outIndices = new Uint32Array(dag.indices.length + sk.tris.length);
    outIndices.set(dag.indices, 0);
    outIndices.set(sk.tris, dag.indices.length);
    outClusters = dag.clusters.concat(sk.clusters);
    skirtClusters = sk.clusters.length;
    skirtTris = sk.tris.length / 3;
  }

  return {
    gridVerts: outVerts,
    indices: outIndices,
    clusters: outClusters,
    groups: dag.groups,
    levelStats: dag.levelStats,
    lod0Count: dag.lod0Count,
    stats: {
      buildMs: now() - t0,
      levels: dag.stats.levels,
      lod0Clusters: dag.stats.lod0Clusters,
      lod0Tris: dag.stats.lod0Tris,
      totalClusters: dag.stats.totalClusters + skirtClusters,
      totalTris: dag.stats.totalTris + skirtTris,
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
