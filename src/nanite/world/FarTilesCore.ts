/**
 * FarTilesCore — the far-tile aggregation's PURE core (SPEC-STREAMING-WORLD §4,
 * the A3 named refactor, S8): plan → (splat: FarTilesSplat) → dense-emit →
 * buildVoxelPyramid → pack, all THREE-FREE (VoxelizeCrown + VoxelBrickCore +
 * FarTilesSplat + CrownPack — no three/tsl/GeometryRegistry in the import
 * chain), so the ENTIRE per-tile fartile job can run inside the StreamBrain
 * worker. Main-thread receives ready-to-upload packed words only.
 *
 * Moved VERBATIM from FarTiles.ts (plan/emit — the 2026-07-02/03/04 behavior:
 * exact XZ tiling, per-tile relief grids, crown-reach bucketing, trusted grid
 * origin, full-pitch coarse halves) and from BootCache.ts (the PackedFarTile
 * codec). FarTiles.ts keeps the main-thread wrappers (worker-pool boot build +
 * registry appends) for ForestScene's boot path.
 */

import { buildVoxelPyramid, type PreparedVoxelCrown, type VoxelLevel } from '../build/VoxelizeCrown';
import { packPreparedCrown, unpackPreparedCrown, type PackedPreparedCrown } from '../build/CrownPack';
import { BRICK_DIM, type BrickCPU } from '../voxel/VoxelBrickCore';
import {
  SPECIES_BRICK_STRIDE,
  type SplatGridSpec,
  type SplatPoolFlat,
  type TileSplatJob,
  type TileSplatOut,
} from './FarTilesSplat';

/** SHARED far-tile defaults (2026-07-03 — the 2026-07-02 beautification picks, consumed
 *  by ForestScene AND the world path so both scenes ride the same far field):
 *  - tile size 64 m (one merged voxel head per tile);
 *  - cell 0.6 m (L0 bricks 2.4 m — far field reads uniformly finer than 0.75's 3 m slabs
 *    and the 140 m voxel→fartile handoff stops being a harsh jump; 0.5 REJECTED: ~3.4×
 *    splat ⇒ 8+ min cold boot);
 *  - aggregation distance 140 m (per-tree heads end here, the tile head owns beyond;
 *    ranges overlap by the tile radius so a tree is never dropped before its tile is on). */
export const FT_TILE_SIZE = 64;
export const DEFAULT_FT_CELL = 0.6;
export const DEFAULT_AGG_DIST = 140;

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

export interface FarTileOpts {
  tileSize: number; // world meters (default 64)
  cellSize: number; // world meters (default 0.5)
  /** per-pool instance streams (the SAME arrays bound to the tree heads):
   *  a = [x, groundY, z, scale]×n, b = [yaw, …]×n. groundY is honoured since
   *  2026-07-03 (terrain-aware tiles); the forest plants at y=0. */
  pools: { a: Float32Array; b: Float32Array; species: FarTileSpecies }[];
  /** S8: pin the tile lattice to an EXPLICIT world box (chunk-scoped runtime
   *  bakes — tiles snap to the chunk corner so neighboring chunks' lattices
   *  meet watertight). Omitted ⇒ derived from the pools' tree extent (the boot
   *  whole-map path, unchanged). */
  bounds?: { mnX: number; mnZ: number; mxX: number; mxZ: number };
}

/** derived grid + per-tile jobs + flattened pools — shared by the sync and async paths. */
export interface FarTilePlan {
  grid: SplatGridSpec;
  jobs: TileSplatJob[];
  poolsFlat: SplatPoolFlat[];
}

export function planFarTiles(opts: FarTileOpts): FarTilePlan | null {
  const { tileSize, pools } = opts;
  let { cellSize } = opts;
  // world extent of the plantation (or the caller's pinned box — S8 chunk bakes)
  let mnX = Infinity;
  let mnZ = Infinity;
  let mxX = -Infinity;
  let mxZ = -Infinity;
  if (opts.bounds) {
    mnX = opts.bounds.mnX;
    mnZ = opts.bounds.mnZ;
    mxX = opts.bounds.mxX;
    mxZ = opts.bounds.mxZ;
  } else {
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
  }
  if (!Number.isFinite(mnX)) return null;
  const tilesX = Math.max(1, Math.ceil((mxX - mnX + 1) / tileSize));
  const tilesZ = Math.max(1, Math.ceil((mxZ - mnZ + 1) / tileSize));

  // cubic-ish tile grid: XZ = tileSize, Y sized to cover the tallest content (trees are
  // ≤ ~40 m world; cap the cell grid to a brick multiple).
  // EXACT XZ TILING (2026-07-02 beautification): the grid must span EXACTLY tileSize —
  // the old ceil-rounded grid spanned cellsXZ·cellSize > tileSize (e.g. 88·0.75 = 66 m
  // per 64 m pitch), so every tile's +X/+Z 2 m band re-splatted the neighbor's trees:
  // coincident duplicate bricks = the user-visible Z-FIGHT stripes + the repeating
  // one-side-darker seam at every tile border. Snap cellSize DOWN so cellsXZ·cellSize
  // == tileSize (0.75 → 64/88 ≈ 0.727) — watertight, overlap-free.
  const cellsXZ = Math.ceil(tileSize / cellSize / BRICK_DIM) * BRICK_DIM;
  cellSize = tileSize / cellsXZ;

  // flatten species brick sets to the worker-transportable stream (splat uses ONLY
  // center/half/albedo/normal/density of the source bricks)
  const poolsFlat: SplatPoolFlat[] = pools.map((p) => {
    const src = p.species.bricks;
    const flat = new Float32Array(src.length * SPECIES_BRICK_STRIDE);
    for (let i = 0; i < src.length; i++) {
      const b = src[i] as BrickCPU;
      const o = i * SPECIES_BRICK_STRIDE;
      flat[o] = b.center[0];
      flat[o + 1] = b.center[1];
      flat[o + 2] = b.center[2];
      flat[o + 3] = b.half;
      flat[o + 4] = b.albedo[0];
      flat[o + 5] = b.albedo[1];
      flat[o + 6] = b.albedo[2];
      flat[o + 7] = b.normal[0];
      flat[o + 8] = b.normal[1];
      flat[o + 9] = b.normal[2];
      flat[o + 10] = b.density;
    }
    return {
      a: p.a,
      b: p.b,
      species: {
        bricks: flat,
        crownMinY: p.species.crownMinY,
        barkR: p.species.bark.r,
        barkG: p.species.bark.g,
        barkB: p.species.bark.b,
      },
    };
  });

  // bucket instances into EVERY tile their crown can reach (boundary-crossing crowns
  // were silently clipped when bucketed by trunk position only — the user-visible HOLES
  // and white slabs at tile borders). reach = per-species max XZ brick extent × max scale.
  const reachOf = pools.map((p) => {
    let maxR = 0;
    for (const b of p.species.bricks) {
      const r = Math.max(Math.abs(b.center[0]), Math.abs(b.center[2])) + b.half;
      if (r > maxR) maxR = r;
    }
    let maxS = 0;
    for (let i = 3; i < p.a.length; i += 4) {
      const s = p.a[i] as number;
      if (s > maxS) maxS = s;
    }
    return maxR * maxS * 1.45;
  });
  const tileOf = new Map<number, number[]>();
  // per-tile member ground-height range (terrain-aware, 2026-07-03): the tile grid's
  // world floor = min member y (baseY), and the shared cellsY must cover the WORST
  // per-tile relief + the tallest tree. Flat ground (forest, all y=0) ⇒ baseY=0 and
  // cellsY = the legacy 48 m — bit-identical tiles.
  const tileMinY = new Map<number, number>();
  const tileMaxY = new Map<number, number>();
  for (let pi = 0; pi < pools.length; pi++) {
    const a = (pools[pi] as { a: Float32Array }).a;
    const reach = reachOf[pi] as number;
    for (let ii = 0; ii * 4 < a.length; ii++) {
      const x = a[ii * 4] as number;
      const y = a[ii * 4 + 1] as number;
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
          list.push(pi, ii);
          if (y < (tileMinY.get(key) ?? Infinity)) tileMinY.set(key, y);
          if (y > (tileMaxY.get(key) ?? -Infinity)) tileMaxY.set(key, y);
        }
      }
    }
  }
  // PER-TILE grid height: THIS tile's relief (maxY − baseY) + 48 m of tree, snapped to
  // a brick multiple. baseY snaps DOWN to a cell multiple so flat-ground tiles get
  // exactly baseY=0 and the legacy 48 m extent (forest parity). Per-tile (not a global
  // worst-case) — one alpine tile must not size every tile's dense grid (world OOM).
  let worstRelief = 0;
  let maxCellsY = 0;
  const jobs: TileSplatJob[] = [];
  for (const [key, members] of tileOf) {
    const baseY = Math.floor((tileMinY.get(key) ?? 0) / cellSize) * cellSize;
    const relief = (tileMaxY.get(key) ?? 0) - baseY;
    if (relief > worstRelief) worstRelief = relief;
    const cellsY = Math.ceil((48 + relief) / cellSize / BRICK_DIM) * BRICK_DIM;
    if (cellsY > maxCellsY) maxCellsY = cellsY;
    jobs.push({ key, baseY, cellsY, members: Uint32Array.from(members) });
  }
  // eslint-disable-next-line no-console
  console.log(
    `[fartiles] plan: ${jobs.length} tiles (${tilesX}×${tilesZ} grid), cell ${cellSize.toFixed(3)} m, ` +
      `cellsXZ ${cellsXZ}, cellsY ≤ ${maxCellsY} (worst relief ${worstRelief.toFixed(1)} m)`,
  );

  return {
    grid: { tileSize, cellSize, cellsXZ, tilesX, mnX, mnZ },
    jobs,
    poolsFlat,
  };
}

/** dense-emit + pyramid + prep for ONE splatted tile (pure — worker-runnable). */
export function emitTile(grid: SplatGridSpec, res: TileSplatOut): FarTileBuild | null {
  const { tileSize, cellSize, cellsXZ, tilesX, mnX, mnZ } = grid;
  const cellsY = res.cellsY;
  const bricksX = cellsXZ / BRICK_DIM;
  const bricksY = cellsY / BRICK_DIM;
  const nBricks = bricksX * bricksY * bricksX;
  const tx = res.key % tilesX;
  const tz = (res.key / tilesX) | 0;
  const centerX = mnX + tx * tileSize + tileSize * 0.5;
  const centerZ = mnZ + tz * tileSize + tileSize * 0.5;

  const { occLo, occHi, accW, accR, accG, accB, accNX, accNY, accNZ } = res;
  // emit the dense BrickCPU grid for this tile (transient; pyramid copies what it needs)
  const dense: BrickCPU[] = new Array(nBricks);
  const brickWorld = cellSize * BRICK_DIM;
  let occupied = 0;
  for (let bi = 0; bi < nBricks; bi++) {
    const w = accW[bi] as number;
    // prune featherweight bricks AND bricks whose every cell fell below the coverage
    // threshold (no occupancy bits ⇒ nothing for the raster to paint or carve)
    if (w <= 0.15 || ((occLo[bi] as number) === 0 && (occHi[bi] as number) === 0)) {
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
  if (occupied === 0) return null;
  // terrain-aware (2026-07-03): brick Y is tile-LOCAL (measured from the tile's world
  // floor); the identity instance at center y = baseY translates it back to world.

  const levels: VoxelLevel[] = buildVoxelPyramid(
    dense,
    { x: bricksX, y: bricksY, z: bricksX },
    cellSize,
    [-tileSize * 0.5, 0, -tileSize * 0.5],
    // full-pitch coarse halves (2026-07-04 axis-stripe fix): on this GLOBAL grid the
    // tight-half gaps align in depth along world axes → coherent vertical see-through
    // stripes ("mis-angled billboards", worst dead-on-axis, gone oblique). Edge-to-edge
    // tiling closes them; the re-binned occupancy still carves see-through. Crowns
    // (VoxelizeCrown's own pyramid call) keep the tight bound — their shrink is the
    // oversized-square fix for isolated blobs.
    { fullPitchHalves: true },
  );
  let brickCount = 0;
  let clusterCount = 0;
  for (const lvl of levels) {
    brickCount += lvl.occupied.length;
    clusterCount += lvl.blocks.length;
  }
  return {
    center: [centerX, res.baseY, centerZ],
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
  };
}

// ── packed far-tile codec (moved from BootCache.ts — CrownPack carries the
// crown-level pack; these 2 wrappers are the fartile-specific form, shared by
// ForestScene's boot cache and the S8 brain→main attach packets) ─────────────

export interface PackedFarTile {
  center: [number, number, number];
  prep: PackedPreparedCrown;
}

export function packFarTiles(tiles: FarTileBuild[]): PackedFarTile[] {
  return tiles.map((t) => ({ center: t.center, prep: packPreparedCrown(t.prep) }));
}

export function unpackFarTiles(packed: PackedFarTile[]): FarTileBuild[] {
  return packed.map((t) => ({ center: t.center, prep: unpackPreparedCrown(t.prep) }));
}
