/**
 * Terrain heightfield LOD — the HEIGHTMAP-NATIVE regular-grid builder (TERRAIN-RW,
 * D-N44). The SOLE terrain-LOD builder. Hand-rolled, CPU/typed-array, node-runnable
 * (no three.js, no GPU) for the validation probe (tools/probe-heightgrid.ts). Emits
 * `HeightDagBuild` (packed grid-coord verts + indices + DagCluster cut records) →
 * the streamer / `attachHeightDagTile` / GPU fetch / `kClusterCull` consume it.
 *
 * WHY it replaced the QEM approach (the old N8-D2 RTIN/QEM build was BROKEN, user
 * 2026-06-16): RTIN/QEM decimation on the built mesh produces TRIANGLE FANS (many
 * grid verts collapse onto one surviving endpoint ⇒ long slivers radiating from a
 * point — "a flap going into the center")
 * and HUGE SPANNING TRIANGLES whose 3 corner-heights don't represent the floor
 * between them ⇒ the coarse surface "flaps" off the real terrain. It is also slow
 * (an iterative quadric heap). The user's insight: we HAVE a heightmap — coarsen
 * it DIRECTLY (regular downsample) instead of running a geometry-domain optimiser.
 *
 * THE REPRESENTATION (geometry-clipmap / CDLOD, expressed through the cluster cut):
 *  - LOD level ℓ is the tile sampled at REGULAR stride 2^ℓ. Every triangle covers
 *    exactly one stride-cell and every vertex sits on a real heightmap texel
 *    (height fetched on the GPU) ⇒ a fan or a spanning triangle is STRUCTURALLY
 *    impossible. The only error is bilinear deviation across a cell — bounded,
 *    measured per level (E[ℓ]).
 *  - TILE-UNIFORM cut: every cluster's ERROR sphere (oe.../pe..., what kClusterCull
 *    projects ownError/parentError through — D-N31 cut) is the whole-TILE sphere,
 *    so all clusters of a level select together ⇒ the tile renders ONE level ⇒ NO
 *    intra-tile T-junction cracks. The per-cluster geometric sphere (sx..sr, used
 *    for frustum/occlusion) stays the real block bounds, so off-screen blocks
 *    still cull individually. Adjacent tiles differ by ≤1 effective stride (the
 *    clipmap's 2× nesting), sealed by per-level perimeter SKIRTS.
 *  - ADAPTIVE at TILE granularity for free: a flat tile has E[ℓ]≈0 at every level,
 *    so equal-error level COALESCING emits only the coarsest (root) level ⇒ a flat
 *    tile is ~2 triangles; a cliff tile keeps every level. The cut then picks the
 *    coarsest level whose screen error ≤ τ. (Per-CLUSTER adaptivity is deliberately
 *    dropped — it was the source of the QEM fans.)
 *
 * RNG-free → deterministic by construction (given the heights).
 */

import type { DagCluster, DagGroup, DagLevelStats } from './BuildDag';

/** Inter-level/inter-tile skirt curtain depth in WORLD metres for skirt-code k
 *  (1-7, encoded in bits 13-15 of a skirt vert): depth = SKIRT_DEPTH_A + SKIRT_DEPTH_B·(k-1).
 *  Both the BUILD (skirt sphere) and the GPU (NaniteFetch drops a flagged vert) read
 *  these so they agree. The inter-tile crack SATURATES with coarseness on real eroded
 *  terrain, so a LINEAR depth hugs it (probe-skirtgap calibrates). */
export const SKIRT_DEPTH_A = 24;
export const SKIRT_DEPTH_B = 12;

/** the heightfield to LOD: row-major (gridN+1)² heights over a regular grid. */
export interface HeightField {
  /** (gridN+1)*(gridN+1) row-major heights (metres); vertex (gx,gz) at gz*(gridN+1)+gx */
  heights: Float32Array;
  /** quads per axis (vertices = gridN+1 per axis); must be a power of two */
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
  /** when ≥0, append per-level DOUBLE-SIDED perimeter SKIRTS (curtain depth from the
   *  skirt code) sealing the inter-tile T-junction at this clipmap level (0 = finest).
   *  Undefined/<0 = no skirts (the headless probe / uniform-tile path). */
  skirtLevel?: number;
}

export interface HeightDagStats {
  buildMs: number;
  levels: number;
  lod0Clusters: number;
  lod0Tris: number;
  totalClusters: number;
  totalTris: number;
  roots: number;
  /** max own-error across the build (metres, vertical) — the coarsest-level deviation */
  maxError: number;
  /** vertices off the grid (always 0 here — every vert is a grid texel by construction) */
  offGridVerts: number;
  /** worst grid-coord round-trip residual in cells (always 0 here) */
  maxGridResidual: number;
}

/** the terrain LOD build: cluster/cut metadata + a packed GRID-COORD vertex pool
 *  (gx | gz<<16, NOT floats — heights fetch on the GPU from the resident texture). */
export interface HeightDagBuild {
  /** packed (gx | (gz<<16) | skirtCode<<13) per vertex; world = (gx*cell+oX, height, gz*cell+oZ) */
  gridVerts: Uint32Array;
  /** triangle indices into gridVerts; cluster c owns tris [triStart, +triCount) */
  indices: Uint32Array;
  clusters: DagCluster[];
  groups: DagGroup[];
  levelStats: DagLevelStats[];
  lod0Count: number;
  stats: HeightDagStats;
  gridN: number;
  cellSize: number;
  originX: number;
  originZ: number;
}

/** cluster block edge in CELLS (at the level's own stride): 8×8 cells = 2·64 =
 *  128 triangles = the GPU MAX_CLUSTER_TRIS cap, one full cluster per block. */
const CLUSTER_BLOCK = 8;
/** the skirt code is 3 bits (level+1, 1-7) ⇒ encodable coarseness caps at 6. */
const SKIRT_MAX_CODE = 7;

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

/** linear height at fractional cell position (u,v)∈[0,1]² under the SAME diagonal
 *  split the grid mesh rasterises (even tri (0,0)(0,1)(1,1), odd (0,0)(1,1)(1,0) —
 *  diagonal v=u). Matches what the GPU draws, so E[ℓ] is the exact rendered error. */
function triHeight(u: number, v: number, h00: number, h10: number, h01: number, h11: number): number {
  return v >= u
    ? h00 + (v - u) * (h01 - h00) + u * (h11 - h00) // upper-left triangle (v≥u)
    : h00 + v * (h11 - h00) + (u - v) * (h10 - h00); // lower-right triangle (v<u)
}

/**
 * Max vertical deviation (metres) of the level-`stride` regular grid surface from
 * the full LOD0 heights, over the whole tile — the error OMITTING the finer detail
 * introduces, exactly the metric kClusterCull projects to screen pixels.
 */
function levelError(heights: Float32Array, gridN: number, stride: number): number {
  const size = gridN + 1;
  let maxE = 0;
  for (let gz = 0; gz <= gridN; gz++) {
    const cz0 = gz - (gz % stride);
    const cz1 = cz0 + stride;
    if (cz1 > gridN) continue; // boundary grid line — coincides with a level vertex
    const v = (gz - cz0) / stride;
    for (let gx = 0; gx <= gridN; gx++) {
      const cx0 = gx - (gx % stride);
      const cx1 = cx0 + stride;
      if (cx1 > gridN) continue;
      const u = (gx - cx0) / stride;
      const h00 = heights[cz0 * size + cx0] as number;
      const h10 = heights[cz0 * size + cx1] as number;
      const h01 = heights[cz1 * size + cx0] as number;
      const h11 = heights[cz1 * size + cx1] as number;
      const e = Math.abs((heights[gz * size + gx] as number) - triHeight(u, v, h00, h10, h01, h11));
      if (e > maxE) maxE = e;
    }
  }
  return maxE;
}

/** world-space bounding sphere of an [x0,x1]×[z0,z1] block over height [yMin,yMax]. */
function blockSphere(
  x0: number,
  x1: number,
  z0: number,
  z1: number,
  yMin: number,
  yMax: number,
): { cx: number; cy: number; cz: number; r: number } {
  const cx = (x0 + x1) / 2;
  const cy = (yMin + yMax) / 2;
  const cz = (z0 + z1) / 2;
  return { cx, cy, cz, r: Math.hypot(x1 - cx, yMax - cy, z1 - cz) };
}

/** a terrain cluster: cone disabled (heightfields skip cone cull), the ERROR
 *  spheres set to the shared TILE sphere (tile-uniform cut), the GEOMETRIC sphere
 *  to the block bounds. parentError +∞ ⇒ root (coarsest level). */
function makeCluster(
  level: number,
  triStart: number,
  triCount: number,
  geo: { cx: number; cy: number; cz: number; r: number },
  ownError: number,
  parentError: number,
  tile: { cx: number; cy: number; cz: number; r: number },
): DagCluster {
  return {
    level,
    triStart,
    triCount,
    sx: geo.cx,
    sy: geo.cy,
    sz: geo.cz,
    sr: geo.r,
    cax: 0,
    cay: 1,
    caz: 0,
    ccos: -1,
    ownError,
    oex: tile.cx,
    oey: tile.cy,
    oez: tile.cz,
    oer: tile.r,
    parentError,
    pex: tile.cx,
    pey: tile.cy,
    pez: tile.cz,
    per: tile.r,
    groupAsInput: -1,
    groupAsParent: -1,
  };
}

/**
 * Heightmap-native regular-grid LOD build → `HeightDagBuild`.
 *
 *  1. measure E[ℓ] for ℓ = 0..log2(gridN) (vertical error of stride 2^ℓ);
 *  2. coalesce equal-error levels (coarse→fine: keep a finer level only if it
 *     meaningfully reduces error) so a flat tile collapses to its root;
 *  3. emit each kept level as regular CLUSTER_BLOCK² clusters + an optional
 *     per-level perimeter skirt, chaining ownError/parentError into a monotone
 *     cut and sharing the tile error-sphere (tile-uniform LOD).
 */
export function buildHeightGrid(hf: HeightField, opts: HeightDagOpts = {}): HeightDagBuild {
  const t0 = now();
  const { gridN, cellSize, originX, originZ, heights } = hf;
  if (gridN < 2 || (gridN & (gridN - 1)) !== 0) {
    throw new Error(`buildHeightGrid: gridN must be a power of two, got ${gridN}`);
  }
  const size = gridN + 1;
  const maxTris = opts.maxTris ?? 128;
  if (CLUSTER_BLOCK * CLUSTER_BLOCK * 2 > maxTris) {
    throw new Error(`buildHeightGrid: CLUSTER_BLOCK² exceeds maxTris ${maxTris}`);
  }

  // tile AABB → shared ERROR sphere (the tile-uniform cut anchor)
  let yMin = Infinity;
  let yMax = -Infinity;
  for (let i = 0; i < size * size; i++) {
    const h = heights[i] as number;
    if (h < yMin) yMin = h;
    if (h > yMax) yMax = h;
  }
  const tileSphere = blockSphere(originX, originX + gridN * cellSize, originZ, originZ + gridN * cellSize, yMin, yMax);

  // -- 1. per-level error + 2. coalesce equal-error levels (coarse→fine) --------
  const maxLevel = Math.log2(gridN); // coarsest = stride gridN = one cell = 2 tris
  const errAt: number[] = [];
  for (let l = 0; l <= maxLevel; l++) errAt[l] = l === 0 ? 0 : levelError(heights, gridN, 1 << l);
  // walk coarse→fine; keep the root, then each finer level only if it cuts error
  // by a meaningful margin (≥2 cm) — flat regions collapse, cliffs keep every level.
  const REDUCE = 0.02;
  const kept: number[] = [maxLevel];
  for (let l = maxLevel - 1; l >= 0; l--) {
    if ((errAt[l] as number) < (errAt[kept[kept.length - 1] as number] as number) - REDUCE) kept.push(l);
  }
  kept.reverse(); // now fine→coarse: [ℓ0, ℓ1, ..., maxLevel]

  // -- 3. emit clusters per kept level ------------------------------------------
  const vmap = new Map<number, number>(); // packed grid word → output vertex index
  const gridVerts: number[] = [];
  const indices: number[] = [];
  const clusters: DagCluster[] = [];
  const levelStats: DagLevelStats[] = [];
  const getVert = (gx: number, gz: number, code: number): number => {
    const key = ((gx & 0x1fff) | (code << 13) | ((gz & 0xffff) << 16)) >>> 0;
    let idx = vmap.get(key);
    if (idx === undefined) {
      idx = gridVerts.length;
      gridVerts.push(key);
      vmap.set(key, idx);
    }
    return idx;
  };
  const skirtLevel = opts.skirtLevel != null && opts.skirtLevel >= 0 ? opts.skirtLevel : -1;

  let lod0Clusters = 0;
  let lod0Tris = 0;
  for (let k = 0; k < kept.length; k++) {
    const l = kept[k] as number;
    const stride = 1 << l;
    const cells = gridN >> l; // cells per axis at this level
    // LEAF CONTRACT (the vanishing-tile fix): the FINEST KEPT level is this
    // tile's leaf representation — there is NOTHING finer to descend to, so its
    // ownError MUST be 0 or the cut refuses it near the camera and emits NOTHING
    // (pOwn = projK·err/denO > τ within ~772·err/τ metres: a collapsed-but-kept
    // level-ℓ>0 tile — a genuinely calm bog lawn, err ≤ REDUCE — VANISHED inside
    // ~5-8 m at eye level while rendering fine from altitude; the persistent
    // near-camera terrain holes). Coalescing keeps its ≤ REDUCE honesty bound;
    // rendered heights live-sample the field anyway, so drawing the finest kept
    // lattice is always strictly better than a hole. Coarser levels keep their
    // real errors (the anchor-chain descent still retires them near the camera).
    const ownError = k === 0 ? 0 : (errAt[l] as number);
    const isRoot = k === kept.length - 1;
    const parentError = isRoot ? Infinity : (errAt[kept[k + 1] as number] as number);
    const blocks = Math.ceil(cells / CLUSTER_BLOCK);
    let lvlClusters = 0;
    let lvlInTris = 0;
    let lvlTris = 0;

    for (let bz = 0; bz < blocks; bz++) {
      for (let bx = 0; bx < blocks; bx++) {
        const cx0 = bx * CLUSTER_BLOCK;
        const cz0 = bz * CLUSTER_BLOCK;
        const cx1 = Math.min(cx0 + CLUSTER_BLOCK, cells);
        const cz1 = Math.min(cz0 + CLUSTER_BLOCK, cells);
        const triStart = indices.length / 3;
        let bMinX = Infinity;
        let bMaxX = -Infinity;
        let bMinZ = Infinity;
        let bMaxZ = -Infinity;
        let bMinY = Infinity;
        let bMaxY = -Infinity;
        for (let cz = cz0; cz < cz1; cz++) {
          for (let cx = cx0; cx < cx1; cx++) {
            const gx0 = cx * stride;
            const gz0 = cz * stride;
            const gx1 = gx0 + stride;
            const gz1 = gz0 + stride;
            const v00 = getVert(gx0, gz0, 0);
            const v10 = getVert(gx1, gz0, 0);
            const v01 = getVert(gx0, gz1, 0);
            const v11 = getVert(gx1, gz1, 0);
            indices.push(v00, v01, v11, v00, v11, v10); // winding == GPU heightfield path
            // grow block AABB (world) over the cell's 4 heights
            for (const [ix, iz] of [
              [gx0, gz0],
              [gx1, gz0],
              [gx0, gz1],
              [gx1, gz1],
            ] as const) {
              const wx = ix * cellSize + originX;
              const wz = iz * cellSize + originZ;
              const wy = heights[iz * size + ix] as number;
              if (wx < bMinX) bMinX = wx;
              if (wx > bMaxX) bMaxX = wx;
              if (wz < bMinZ) bMinZ = wz;
              if (wz > bMaxZ) bMaxZ = wz;
              if (wy < bMinY) bMinY = wy;
              if (wy > bMaxY) bMaxY = wy;
            }
          }
        }
        const triCount = indices.length / 3 - triStart;
        if (triCount === 0) continue;
        const geo = blockSphere(bMinX, bMaxX, bMinZ, bMaxZ, bMinY, bMaxY);
        clusters.push(makeCluster(l, triStart, triCount, geo, ownError, parentError, tileSphere));
        lvlClusters++;
        lvlTris += triCount;
        lvlInTris += triCount;
      }
    }

    // per-level perimeter skirt (top verts on THIS level's coarse edge, gated to
    // this level by the shared ownError/parentError ⇒ drawn only when level l is
    // active). Seals the inter-tile T-junction crack vs a neighbour at a coarser
    // effective stride. code grows with total coarseness (clipmap + intra level).
    if (skirtLevel >= 0) {
      const code = Math.min(skirtLevel + l + 1, SKIRT_MAX_CODE);
      const sk = appendLevelSkirt(hf, stride, cells, code, maxTris, getVert, indices);
      for (const s of sk) {
        clusters.push(makeCluster(l, s.triStart, s.triCount, s.geo, ownError, parentError, tileSphere));
        lvlClusters++;
        lvlTris += s.triCount;
      }
    }

    levelStats.push({
      level: l,
      inClusters: lvlClusters,
      inTris: lvlInTris,
      outClusters: lvlClusters,
      outTris: lvlTris,
      groups: 0,
      stuckGroups: 0,
      triReduction: 0,
    });
    if (k === 0) {
      lod0Clusters = lvlClusters;
      lod0Tris = lvlTris;
    }
  }

  const roots = clusters.filter((c) => !Number.isFinite(c.parentError)).length;
  const totalTris = indices.length / 3;
  const groups: DagGroup[] = []; // regular-grid build has no QEM groups
  return {
    gridVerts: Uint32Array.from(gridVerts),
    indices: Uint32Array.from(indices),
    clusters,
    groups,
    levelStats,
    lod0Count: lod0Clusters,
    stats: {
      buildMs: now() - t0,
      levels: kept.length,
      lod0Clusters,
      lod0Tris,
      totalClusters: clusters.length,
      totalTris,
      roots,
      maxError: errAt[maxLevel] as number,
      offGridVerts: 0, // every vert is a grid texel by construction
      maxGridResidual: 0,
    },
    gridN,
    cellSize,
    originX,
    originZ,
  };
}

/**
 * Append a DOUBLE-SIDED perimeter skirt for level `stride` (coarse-edge verts at
 * the level's own stride). Top verts (code 0) sit on the surface edge; bottom
 * verts carry the depth `code` (bits 13-15) so the GPU drops them below to seal
 * the inter-level/inter-tile T-junction. Returns one cluster span per edge run
 * (≤ maxSeg segments) so the spheres stay edge-local. The skirt vert encoding
 * (code in bits 13-15, depth = SKIRT_DEPTH_A + SKIRT_DEPTH_B·(code-1)) matches the
 * GPU fetch (NaniteFetch `dagWorldByIndex`).
 */
function appendLevelSkirt(
  hf: HeightField,
  stride: number,
  cells: number,
  code: number,
  maxTris: number,
  getVert: (gx: number, gz: number, code: number) => number,
  indices: number[],
): Array<{ triStart: number; triCount: number; geo: { cx: number; cy: number; cz: number; r: number } }> {
  const { cellSize, originX, originZ, heights } = hf;
  const size = hf.gridN + 1;
  const depth = SKIRT_DEPTH_A + SKIRT_DEPTH_B * (code - 1);
  const maxSeg = Math.max(1, maxTris >> 2); // 4 tris (double-sided) per segment
  const out: Array<{ triStart: number; triCount: number; geo: { cx: number; cy: number; cz: number; r: number } }> = [];
  // walk the 4 edges in level-grid steps (texel coord = cell·stride), splitting each
  // into ≤maxSeg-segment clusters so no skirt cluster exceeds the GPU tri cap.
  const edges: ReadonlyArray<readonly [number, number, number, number]> = [
    [0, 0, 1, 0], // south
    [cells, 0, 0, 1], // east
    [cells, cells, -1, 0], // north
    [0, cells, 0, -1], // west
  ];
  for (const [ecx, ecz, dcx, dcz] of edges) {
    let triStart = indices.length / 3;
    let seg = 0;
    let minX = Infinity;
    let maxX = -Infinity;
    let minZ = Infinity;
    let maxZ = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    const flush = (): void => {
      const triCount = indices.length / 3 - triStart;
      if (triCount > 0) out.push({ triStart, triCount, geo: blockSphere(minX, maxX, minZ, maxZ, minY, maxY) });
      triStart = indices.length / 3;
      seg = 0;
      minX = minZ = minY = Infinity;
      maxX = maxZ = maxY = -Infinity;
    };
    let pTop = -1;
    let pBot = -1;
    const grow = (gx: number, gz: number, h: number): void => {
      const wx = gx * cellSize + originX;
      const wz = gz * cellSize + originZ;
      if (wx < minX) minX = wx;
      if (wx > maxX) maxX = wx;
      if (wz < minZ) minZ = wz;
      if (wz > maxZ) maxZ = wz;
      if (h > maxY) maxY = h;
      if (h - depth < minY) minY = h - depth;
    };
    for (let i = 0; i <= cells; i++) {
      const gx = (ecx + dcx * i) * stride;
      const gz = (ecz + dcz * i) * stride;
      const h = heights[gz * size + gx] as number;
      const top = getVert(gx, gz, 0);
      const bot = getVert(gx, gz, code);
      if (i > 0) {
        // quad (pTop,pBot,bot,top): 2 front + 2 reversed back tris (double-sided)
        indices.push(pTop, pBot, bot, pTop, bot, top);
        indices.push(pTop, bot, pBot, pTop, top, bot);
        grow((ecx + dcx * (i - 1)) * stride, (ecz + dcz * (i - 1)) * stride, heights[((ecz + dcz * (i - 1)) * stride) * size + (ecx + dcx * (i - 1)) * stride] as number);
        grow(gx, gz, h);
        if (++seg >= maxSeg) flush();
      }
      pTop = top;
      pBot = bot;
    }
    flush(); // close the edge so clusters stay edge-local
  }
  return out;
}
