/**
 * FarTiles — cross-instance FAR-FIELD AGGREGATION (the UE5-HLOD-equivalent move,
 * 2026-07-02 wave 3, ?fartiles=1).
 *
 * WHY: beyond the aggregation distance the per-tree representation is structurally
 * wasteful — thousands of per-tree voxel heads (each a DAG cut + blocks + bricks) and
 * trunk meshes, all separately culled and rastered; worse, the per-instance size cull
 * (instMinPx ≈ 110 px) simply DELETES trees beyond ~300 m in the forest scene, so the
 * forest visibly ends. Aggregating MANY trees into ONE per-tile voxel head collapses
 * the far cluster/brick count by orders of magnitude and extends the forest to the
 * horizon for the cost of ~1 head per 64 m tile.
 *
 * HOW: at boot (after planting, BEFORE reg.build()), each tile splats its member
 * trees into a tile-local cell grid (default 0.5 m cells → 2 m L0 bricks — matched to
 * the per-tree fidelity at the default 140 m handoff, where the per-tree path shows
 * ~0.5 m crown-L1 bricks):
 *   - crown: the species' prepared crown pyramid level whose brick size best matches
 *     the tile cell size, each source brick transformed by the instance (scale, yaw,
 *     translate — the forest plants with zero lean) and splatted into its tile cell
 *     with albedo/normal/density accumulation;
 *   - trunk: a vertical column of bark-colored cells from ground to the crown base
 *     (radial horizontal normals so far trunks shade like cylinders, not flat).
 * Each tile grid then runs the SAME buildVoxelPyramid → blocks → appendVoxelCrown
 * machinery as a single crown, bound to ONE identity instance at the tile center.
 * The per-tree heads (bark/leaf/voxel) get maxDist = aggDist; the tile head gets
 * nearDist = aggDist − tileRadius (ranges OVERLAP by the tile radius — a tree is
 * never dropped before its tile is guaranteed on; the overlap band double-draws,
 * which is cheap at 140 m and hole-free by construction).
 */

import type { GeometryRegistry } from './GeometryRegistry';
import { appendVoxelCrown, buildVoxelPyramid, type PreparedVoxelCrown, type VoxelLevel } from './VoxelizeCrown';
import { BRICK_DIM, type BrickCPU } from './VoxelBrick';

export interface FarTileSpecies {
  /** occupied bricks of the chosen coarse crown pyramid level (crown-LOCAL space). */
  bricks: BrickCPU[];
  /** crown-local min Y of the crown (trunk column top). */
  crownMinY: number;
  /** bark albedo for the trunk column. */
  bark: { r: number; g: number; b: number };
}

export interface FarTileBuild {
  center: [number, number, number];
  prep: PreparedVoxelCrown;
}

const EMPTY_BRICK: BrickCPU = {
  occLo: 0,
  occHi: 0,
  normal: [0, 1, 0],
  spread: 0.5,
  albedo: [0, 0, 0],
  density: 0,
  center: [0, 0, 0],
  half: 0,
};

/** Splat all instances into tile grids and build one voxel pyramid per tile. */
export function buildFarTiles(opts: {
  tileSize: number; // world meters (default 64)
  cellSize: number; // world meters (default 0.5)
  /** per-pool instance streams (the SAME arrays bound to the tree heads):
   *  a = [x, 0, z, scale]×n, b = [yaw, …]×n. */
  pools: { a: Float32Array; b: Float32Array; species: FarTileSpecies }[];
}): FarTileBuild[] {
  const { tileSize, cellSize, pools } = opts;
  // world extent of the plantation
  let mnX = Infinity;
  let mnZ = Infinity;
  let mxX = -Infinity;
  let mxZ = -Infinity;
  for (const p of pools) {
    for (let i = 0; i < p.a.length; i += 4) {
      const x = p.a[i] as number;
      const z = p.a[i + 2] as number;
      if (x < mnX) mnX = x;
      if (x > mxX) mxX = x;
      if (z < mnZ) mnZ = z;
      if (z > mxZ) mxZ = z;
    }
  }
  if (!Number.isFinite(mnX)) return [];
  const tilesX = Math.max(1, Math.ceil((mxX - mnX + 1) / tileSize));
  const tilesZ = Math.max(1, Math.ceil((mxZ - mnZ + 1) / tileSize));

  // cubic-ish tile grid: XZ = tileSize, Y sized to cover the tallest content (trees are
  // ≤ ~40 m world; cap the cell grid to a brick multiple).
  const cellsXZ = Math.ceil(tileSize / cellSize / BRICK_DIM) * BRICK_DIM;
  const cellsY = Math.ceil(48 / cellSize / BRICK_DIM) * BRICK_DIM;
  const bricksX = cellsXZ / BRICK_DIM;
  const bricksY = cellsY / BRICK_DIM;
  const nBricks = bricksX * bricksY * bricksX;
  const nCellsPerBrickRow = BRICK_DIM;

  // flat accumulators, reused across tiles (typed arrays — no per-cell objects)
  const occLo = new Uint32Array(nBricks);
  const occHi = new Uint32Array(nBricks);
  const accW = new Float32Array(nBricks);
  const accR = new Float32Array(nBricks);
  const accG = new Float32Array(nBricks);
  const accB = new Float32Array(nBricks);
  const accNX = new Float32Array(nBricks);
  const accNY = new Float32Array(nBricks);
  const accNZ = new Float32Array(nBricks);

  // bucket instances into EVERY tile their crown can reach (boundary-crossing crowns
  // were silently clipped when bucketed by trunk position only — the user-visible HOLES
  // and white slabs at tile borders). reach = per-species max XZ brick extent × max scale.
  const reachOf = pools.map((p) => {
    let r = 1;
    for (const b of p.species.bricks) {
      r = Math.max(r, Math.abs(b.center[0]) + b.half, Math.abs(b.center[2]) + b.half);
    }
    return r * 1.45; // max instance scale ≈ 1.4
  });
  const tileOf = new Map<number, [number, number][]>();
  for (let pi = 0; pi < pools.length; pi++) {
    const a = (pools[pi] as { a: Float32Array }).a;
    const reach = reachOf[pi] as number;
    for (let ii = 0; ii * 4 < a.length; ii++) {
      const x = a[ii * 4] as number;
      const z = a[ii * 4 + 2] as number;
      const tx0 = Math.min(tilesX - 1, Math.max(0, Math.floor((x - reach - mnX) / tileSize)));
      const tx1 = Math.min(tilesX - 1, Math.max(0, Math.floor((x + reach - mnX) / tileSize)));
      const tz0 = Math.min(tilesZ - 1, Math.max(0, Math.floor((z - reach - mnZ) / tileSize)));
      const tz1 = Math.min(tilesZ - 1, Math.max(0, Math.floor((z + reach - mnZ) / tileSize)));
      for (let tz = tz0; tz <= tz1; tz++) {
        for (let tx = tx0; tx <= tx1; tx++) {
          const key = tz * tilesX + tx;
          let list = tileOf.get(key);
          if (!list) {
            list = [];
            tileOf.set(key, list);
          }
          list.push([pi, ii]);
        }
      }
    }
  }

  const splatCell = (
    cx: number,
    cy: number,
    cz: number,
    alb: [number, number, number],
    nrm: [number, number, number],
    w: number,
  ): void => {
    if (cx < 0 || cy < 0 || cz < 0 || cx >= cellsXZ || cy >= cellsY || cz >= cellsXZ) return;
    const bx = (cx / nCellsPerBrickRow) | 0;
    const by = (cy / nCellsPerBrickRow) | 0;
    const bz = (cz / nCellsPerBrickRow) | 0;
    const bi = bx + by * bricksX + bz * bricksX * bricksY;
    const cellBit = (cx & 3) + (cy & 3) * 4 + (cz & 3) * 16;
    if (cellBit < 32) occLo[bi] = ((occLo[bi] as number) | (1 << cellBit)) >>> 0;
    else occHi[bi] = ((occHi[bi] as number) | (1 << (cellBit - 32))) >>> 0;
    accW[bi] = (accW[bi] as number) + w;
    accR[bi] = (accR[bi] as number) + alb[0] * w;
    accG[bi] = (accG[bi] as number) + alb[1] * w;
    accB[bi] = (accB[bi] as number) + alb[2] * w;
    accNX[bi] = (accNX[bi] as number) + nrm[0] * w;
    accNY[bi] = (accNY[bi] as number) + nrm[1] * w;
    accNZ[bi] = (accNZ[bi] as number) + nrm[2] * w;
  };
  // VOLUME splat: cover every cell the source brick's AABB overlaps. Point-splatting the
  // center produced a sampling BEAT against the tile grid (0.53 m source bricks vs 0.75 m
  // cells → regular missed columns = the user-visible "wireframe" stripe artifact).
  const splatBox = (
    lx: number,
    ly: number,
    lz: number,
    half: number,
    alb: [number, number, number],
    nrm: [number, number, number],
    w: number,
  ): void => {
    const x0 = Math.floor((lx - half) / cellSize);
    const x1 = Math.floor((lx + half) / cellSize);
    const y0 = Math.floor((ly - half) / cellSize);
    const y1 = Math.floor((ly + half) / cellSize);
    const z0 = Math.floor((lz - half) / cellSize);
    const z1 = Math.floor((lz + half) / cellSize);
    for (let cz = z0; cz <= z1; cz++)
      for (let cy = y0; cy <= y1; cy++)
        for (let cx = x0; cx <= x1; cx++) splatCell(cx, cy, cz, alb, nrm, w);
  };

  const out: FarTileBuild[] = [];
  for (const [key, members] of tileOf) {
    const tx = key % tilesX;
    const tz = (key / tilesX) | 0;
    // tile-local origin = min corner; the identity instance sits at the tile CENTER,
    // so bricks are authored in tile-local coords with the center subtracted.
    const originX = mnX + tx * tileSize;
    const originZ = mnZ + tz * tileSize;
    const centerX = originX + tileSize * 0.5;
    const centerZ = originZ + tileSize * 0.5;

    occLo.fill(0);
    occHi.fill(0);
    accW.fill(0);
    accR.fill(0);
    accG.fill(0);
    accB.fill(0);
    accNX.fill(0);
    accNY.fill(0);
    accNZ.fill(0);

    for (const [pi, ii] of members) {
      const pool = pools[pi] as { a: Float32Array; b: Float32Array; species: FarTileSpecies };
      const x = pool.a[ii * 4] as number;
      const z = pool.a[ii * 4 + 2] as number;
      const s = pool.a[ii * 4 + 3] as number;
      const yaw = pool.b[ii * 4] as number;
      const cy = Math.cos(yaw);
      const sy = Math.sin(yaw);
      const lx0 = x - originX;
      const lz0 = z - originZ;
      const sp = pool.species;
      // crown bricks (instance transform: scale, yaw, translate — zero lean in forest)
      for (const b of sp.bricks) {
        const px = (b.center[0] as number) * s;
        const py = (b.center[1] as number) * s;
        const pz = (b.center[2] as number) * s;
        const wx = px * cy + pz * sy + lx0;
        const wz = pz * cy - px * sy + lz0;
        // rotate the baked normal by yaw
        const nx = (b.normal[0] as number) * cy + (b.normal[2] as number) * sy;
        const nz = (b.normal[2] as number) * cy - (b.normal[0] as number) * sy;
        splatBox(wx, py, wz, b.half * s, b.albedo as [number, number, number], [nx, b.normal[1] as number, nz], Math.max(0.05, b.density));
      }
      // trunk column: ground → crown base, radial horizontal normals
      const topY = Math.max(cellSize, sp.crownMinY * s);
      for (let y = cellSize * 0.5; y < topY; y += cellSize) {
        splatCell(Math.floor(lx0 / cellSize), Math.floor(y / cellSize), Math.floor(lz0 / cellSize), [sp.bark.r, sp.bark.g, sp.bark.b], [cy, 0.15, -sy], 1);
      }
    }

    // emit the dense BrickCPU grid for this tile (transient; pyramid copies what it needs)
    const dense: BrickCPU[] = new Array(nBricks);
    const brickWorld = cellSize * BRICK_DIM;
    let occupied = 0;
    for (let bi = 0; bi < nBricks; bi++) {
      const w = accW[bi] as number;
      // prune featherweight bricks (a lone grazing splat) — they add memory, not shape
      if (w <= 0.15) {
        dense[bi] = EMPTY_BRICK;
        continue;
      }
      occupied++;
      const bx = bi % bricksX;
      const by = ((bi / bricksX) | 0) % bricksY;
      const bz = (bi / (bricksX * bricksY)) | 0;
      const nl = Math.hypot(accNX[bi] as number, accNY[bi] as number, accNZ[bi] as number) || 1;
      dense[bi] = {
        occLo: occLo[bi] as number,
        occHi: occHi[bi] as number,
        normal: [(accNX[bi] as number) / nl, (accNY[bi] as number) / nl, (accNZ[bi] as number) / nl],
        spread: 0.6,
        albedo: [(accR[bi] as number) / w, (accG[bi] as number) / w, (accB[bi] as number) / w],
        density: Math.min(1, w / 8),
        center: [
          bx * brickWorld + brickWorld * 0.5 - tileSize * 0.5,
          by * brickWorld + brickWorld * 0.5,
          bz * brickWorld + brickWorld * 0.5 - tileSize * 0.5,
        ],
        half: brickWorld * 0.5,
      };
    }
    if (occupied === 0) continue;

    const levels: VoxelLevel[] = buildVoxelPyramid(
      dense,
      { x: bricksX, y: bricksY, z: bricksX },
      cellSize,
      [-tileSize * 0.5, 0, -tileSize * 0.5],
    );
    let brickCount = 0;
    let clusterCount = 0;
    for (const lvl of levels) {
      brickCount += lvl.occupied.length;
      clusterCount += lvl.blocks.length;
    }
    out.push({
      center: [centerX, 0, centerZ],
      prep: {
        vox: {
          bricks: [],
          occupied: [],
          brickGrid: { x: bricksX, y: bricksY, z: bricksX },
          cellGrid: { x: cellsXZ, y: cellsY, z: cellsXZ },
          origin: [-tileSize * 0.5, 0, -tileSize * 0.5],
          cellSize,
          levels,
          stats: {
            triangles: 0,
            cellsTouched: 0,
            occupiedBricks: occupied,
            totalBricks: nBricks,
            meanDensity: 0,
            voxelizeMs: 0,
          },
        },
        brickCount,
        clusterCount,
        dagLinkCount: clusterCount,
      },
    });
  }
  return out;
}

/** Append built tiles post-reg.build(): one voxel head + one identity instance each. */
export function appendFarTiles(
  reg: GeometryRegistry,
  tiles: FarTileBuild[],
  opts: { nearDist: number; matParam: number },
): number {
  let bricks = 0;
  for (const t of tiles) {
    const r = appendVoxelCrown(reg, t.prep, undefined as never, {
      matParam: opts.matParam,
      swayPad: 0,
      maxDist: 100000,
      nearDist: opts.nearDist,
      label: 'fartile',
    });
    bricks += r.brickCount;
    const a = new Float32Array([t.center[0], t.center[1], t.center[2], 1]);
    const b = new Float32Array([0, 0, 0, 0]);
    reg.bindInstances(r.head, { a, b });
  }
  return bricks;
}
