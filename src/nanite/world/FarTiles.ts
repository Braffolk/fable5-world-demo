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
 *
 * SPLAT CORE (2026-07-02 wave 4): the per-tile splat + occupancy post-pass live in
 * FarTilesSplat.ts (pure, three-free) so buildFarTilesAsync can fan tiles across a
 * module-Worker pool (FarTiles.worker.ts) — the splat is ~1.5B cell accumulations at
 * 200k trees (36-42 s single-threaded). The dense-emit + buildVoxelPyramid + prep
 * stay HERE on the main thread (VoxelizeCrown drags the three-import chain workers
 * must avoid). buildFarTiles (sync) runs the SAME core on the caller thread — the
 * worker-failure fallback and the behavior-exactness reference.
 */

import type { GeometryRegistry } from './GeometryRegistry';
import type { PackedFarTile } from './BootCache';
import { appendPackedCrown, appendVoxelCrown, buildVoxelPyramid, type PreparedVoxelCrown, type VoxelLevel } from '../build/VoxelizeCrown';
import { BRICK_DIM, type BrickCPU } from '../voxel/VoxelBrick';
import {
  splatTiles,
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
}

/** derived grid + per-tile jobs + flattened pools — shared by the sync and async paths. */
interface FarTilePlan {
  grid: SplatGridSpec;
  jobs: TileSplatJob[];
  poolsFlat: SplatPoolFlat[];
}

function planFarTiles(opts: FarTileOpts): FarTilePlan | null {
  const { tileSize, pools } = opts;
  let { cellSize } = opts;
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

/** dense-emit + pyramid + prep for ONE splatted tile (main thread — VoxelizeCrown). */
function emitTile(grid: SplatGridSpec, res: TileSplatOut): FarTileBuild | null {
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

/** Splat all instances into tile grids and build one voxel pyramid per tile (SYNC —
 *  the single-thread reference + worker-failure fallback). STREAMS one tile at a time
 *  (splat → pyramid → release the dense arrays) so peak memory is one tile's working
 *  set, not the whole map's (the world-scene OOM, 2026-07-03). `map` (optional)
 *  transforms each build AS IT EMITS — pass a packer to keep only the compact form
 *  in memory (the accumulated BrickCPU objects for a whole map are heap-fatal). */
function buildFarTiles<T = FarTileBuild>(
  opts: FarTileOpts,
  map?: (b: FarTileBuild, jobIndex: number) => T,
): T[] {
  const plan = planFarTiles(opts);
  if (!plan) return [];
  const out: T[] = [];
  for (let j = 0; j < plan.jobs.length; j++) {
    const results = splatTiles(plan.grid, plan.poolsFlat, [plan.jobs[j] as TileSplatJob]);
    const r = results[0];
    if (!r) continue;
    const t = emitTile(plan.grid, r);
    if (t) out.push(map ? map(t, j) : (t as unknown as T));
  }
  return out;
}

/** Worker-pool splat (wave 4): fans the per-tile splat across module Workers; the
 *  dense-emit + pyramid runs on the main thread AS EACH TILE ARRIVES (one message per
 *  tile since 2026-07-03 — the dense accumulator arrays are released tile-by-tile, so
 *  peak memory is the in-flight tiles, not the whole map; the old all-at-the-end
 *  transfer OOM'd the world scene). Deterministic: builds land in job order regardless
 *  of worker completion order. Falls back to buildFarTiles on any worker error. */
export async function buildFarTilesAsync<T = FarTileBuild>(
  opts: FarTileOpts,
  map?: (b: FarTileBuild, jobIndex: number) => T,
): Promise<T[]> {
  const plan = planFarTiles(opts);
  if (!plan) return [];
  const t0 = performance.now();
  const hw = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 4 : 4;
  const workerCount = Math.max(1, Math.min(8, hw - 2, plan.jobs.length));
  const builds = new Array<T | null>(plan.jobs.length).fill(null);
  let emitMs = 0;
  let emitted = 0;
  let runningBricks = 0;
  // chunk jobs round-robin-contiguous: worker w takes jobs [w·per, (w+1)·per)
  const per = Math.ceil(plan.jobs.length / workerCount);
  try {
    await Promise.all(
      Array.from({ length: workerCount }, (_, w) => {
        const jobs = plan.jobs.slice(w * per, (w + 1) * per);
        if (jobs.length === 0) return Promise.resolve();
        return new Promise<void>((resolve, reject) => {
          const worker = new Worker(new URL('./FarTiles.worker.ts', import.meta.url), {
            type: 'module',
            name: `fartiles-${w}`,
          });
          worker.onmessage = (e: MessageEvent<import('./FarTiles.worker').FtRes>): void => {
            const res = e.data;
            if (!res.ok) {
              worker.terminate();
              reject(new Error(res.error));
              return;
            }
            if (res.done) {
              worker.terminate();
              resolve();
              return;
            }
            // one tile — pyramid it now, release the dense arrays with this message;
            // `map` (e.g. the bootcache packer) immediately compacts the build so the
            // per-brick JS objects never accumulate across the whole map (heap-fatal).
            const tE0 = performance.now();
            const jobIndex = w * per + res.i;
            const built = emitTile(plan.grid, res.tile);
            builds[jobIndex] = built ? (map ? map(built, jobIndex) : (built as unknown as T)) : null;
            emitMs += performance.now() - tE0;
            emitted++;
            if (built) runningBricks += built.prep.brickCount;
            if (emitted % 256 === 0) {
              // eslint-disable-next-line no-console
              console.log(
                `[fartiles] progress ${emitted}/${plan.jobs.length} tiles, ${runningBricks} bricks, ` +
                  `${((performance.now() - t0) / 1000).toFixed(0)} s (emit ${(emitMs / 1000).toFixed(1)} s)`,
              );
            }
          };
          worker.onerror = (e: ErrorEvent): void => {
            worker.terminate();
            reject(new Error(`FarTiles worker crashed: ${e.message}`));
          };
          worker.postMessage({ id: w, grid: plan.grid, pools: plan.poolsFlat, jobs });
        });
      }),
    );
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn('[fartiles] worker splat failed — falling back to sync build:', err);
    return buildFarTiles(opts, map);
  }
  const out: T[] = [];
  for (const b of builds) {
    if (b) out.push(b);
  }
  // eslint-disable-next-line no-console
  console.log(
    `[fartiles] worker splat ${(performance.now() - t0).toFixed(0)} ms (${workerCount} workers, ` +
      `${plan.jobs.length} tiles, interleaved emit/pyramid ${emitMs.toFixed(0)} ms)`,
  );
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

/** Append PACKED tiles post-reg.build() — the world path. Reads straight from each tile's
 *  9×u32 words (appendPackedCrown), no BrickCPU materialization; `tick` yields between tiles
 *  (thousands of tiles = a long slab otherwise). Identical GPU output to appendFarTiles. */
export async function appendPackedFarTiles(
  reg: GeometryRegistry,
  tiles: PackedFarTile[],
  opts: { nearDist: number; matParam: number },
  tick?: () => Promise<void>,
): Promise<number> {
  let bricks = 0;
  for (const t of tiles) {
    if (tick) await tick();
    const r = appendPackedCrown(reg, t.prep, {
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
