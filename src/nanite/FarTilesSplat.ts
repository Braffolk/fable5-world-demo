/**
 * FarTilesSplat — the PURE per-tile splat core of the far-tile aggregation
 * (FarTiles.ts), extracted THREE-FREE so it can run in a module Worker
 * (FarTiles.worker.ts) as well as synchronously on the main thread (the
 * fallback path). This is a behavior-exact port of the original in-file splat:
 * same accumulation order (pool-major, instance-minor, brick-minor), same
 * volume-splat overlap fractions, same trunk columns, same gap-preserving
 * OCC_COVER post-pass — outputs are bit-identical to the pre-extraction build.
 *
 * The worker boundary is TYPED ARRAYS ONLY: species brick sets ride a flat
 * Float32Array stream (stride 11), per-tile members ride packed (pi,ii) pairs
 * in a Uint32Array, and each tile's result is 9 transferable arrays.
 *
 * WHY: the splat is ~1.5B cell accumulations at 200k trees (36-42 s of boot,
 * single-threaded). Tiles are independent ⇒ a worker pool cuts wall time by
 * ~core-count; the pyramid/prep stay on the main thread (they need
 * VoxelizeCrown, which drags the three-import chain the workers must avoid).
 */

/** per-cell coverage threshold — a cell below it contributes NO occupancy bit
 *  (gap-preserving masks; see FarTiles.ts POST-pass comment). */
export const OCC_COVER = 0.22;

/** flat species brick stream stride: cx,cy,cz,half, r,g,b, nx,ny,nz, density */
export const SPECIES_BRICK_STRIDE = 11;

export interface SplatSpeciesFlat {
  /** flat brick stream, SPECIES_BRICK_STRIDE floats per brick (crown-LOCAL space). */
  bricks: Float32Array;
  /** crown-local min Y (trunk column top, pre-scale). */
  crownMinY: number;
  barkR: number;
  barkG: number;
  barkB: number;
}

export interface SplatPoolFlat {
  /** [x, y, z, scale]×n — the SAME instance stream bound to the tree heads. y is the
   *  tree's GROUND height (terrain-aware since 2026-07-03; the forest plants at y=0). */
  a: Float32Array;
  /** [yaw, …]×n. */
  b: Float32Array;
  species: SplatSpeciesFlat;
}

/** grid constants derived once in FarTiles.buildGrid (main thread). cellsY is PER-TILE
 *  (TileSplatJob) since 2026-07-03 — a single global Y extent sized to the worst alpine
 *  tile made every tile pay for it (the world-scene OOM). */
export interface SplatGridSpec {
  tileSize: number;
  cellSize: number;
  cellsXZ: number;
  tilesX: number;
  mnX: number;
  mnZ: number;
}

/** one tile's work order: its key (tz·tilesX+tx) + packed (pi,ii) member pairs.
 *  baseY = the tile grid's WORLD floor (min member ground y, snapped down to a cell
 *  multiple) — instances splat at (y − baseY) so a sloped tile's content stays in-grid;
 *  the tile head's identity instance then sits at y = baseY. 0 on flat ground (forest).
 *  cellsY = THIS tile's grid height in cells (its members' relief + the tallest tree,
 *  brick-multiple) — flat ground yields the legacy 48 m extent. */
export interface TileSplatJob {
  key: number;
  baseY: number;
  cellsY: number;
  members: Uint32Array;
}

/** one tile's splat result — per-BRICK accumulators + post-passed occupancy. */
export interface TileSplatOut {
  key: number;
  /** echoed from the job — emitTile places the tile head at this world height. */
  baseY: number;
  /** echoed from the job — emitTile derives this tile's brick-grid Y from it. */
  cellsY: number;
  occLo: Uint32Array;
  occHi: Uint32Array;
  accW: Float32Array;
  accR: Float32Array;
  accG: Float32Array;
  accB: Float32Array;
  accNX: Float32Array;
  accNY: Float32Array;
  accNZ: Float32Array;
}

const BRICK_DIM = 4; // VoxelBrick.BRICK_DIM — inlined so this module stays import-free

/** splat the given tiles; allocates fresh result arrays per tile (transferable).
 *  Per-tile Y extents (job.cellsY) ⇒ per-tile allocations; the cell-coverage working
 *  buffer grows to the largest tile seen and is reused. */
export function splatTiles(
  grid: SplatGridSpec,
  pools: SplatPoolFlat[],
  jobs: TileSplatJob[],
): TileSplatOut[] {
  const { tileSize, cellSize, cellsXZ, tilesX, mnX, mnZ } = grid;
  const bricksX = cellsXZ / BRICK_DIM;
  // per-CELL coverage accumulator — working memory reused across this call's tiles
  let cellW = new Float32Array(0);

  const out: TileSplatOut[] = [];
  for (const job of jobs) {
    const key = job.key;
    const baseY = job.baseY;
    const cellsY = job.cellsY;
    const bricksY = cellsY / BRICK_DIM;
    const nBricks = bricksX * bricksY * bricksX;
    const nCells = cellsXZ * cellsY * cellsXZ;
    const tx = key % tilesX;
    const tz = (key / tilesX) | 0;
    const originX = mnX + tx * tileSize;
    const originZ = mnZ + tz * tileSize;

    const occLo = new Uint32Array(nBricks);
    const occHi = new Uint32Array(nBricks);
    const accW = new Float32Array(nBricks);
    const accR = new Float32Array(nBricks);
    const accG = new Float32Array(nBricks);
    const accB = new Float32Array(nBricks);
    const accNX = new Float32Array(nBricks);
    const accNY = new Float32Array(nBricks);
    const accNZ = new Float32Array(nBricks);
    if (cellW.length < nCells) cellW = new Float32Array(nCells);
    else cellW.fill(0, 0, nCells);

    // splatCell — exact port (bounds check, brick index, coverage + 7 accumulators)
    const splatCell = (
      cx: number,
      cy: number,
      cz: number,
      ar: number,
      ag: number,
      ab: number,
      nx: number,
      ny: number,
      nz: number,
      w: number,
      cover: number,
    ): void => {
      if (cx < 0 || cy < 0 || cz < 0 || cx >= cellsXZ || cy >= cellsY || cz >= cellsXZ) return;
      const bx = (cx / BRICK_DIM) | 0;
      const by = (cy / BRICK_DIM) | 0;
      const bz = (cz / BRICK_DIM) | 0;
      const bi = bx + by * bricksX + bz * bricksX * bricksY;
      const ci = cx + cy * cellsXZ + cz * cellsXZ * cellsY;
      cellW[ci] = (cellW[ci] as number) + cover;
      accW[bi] = (accW[bi] as number) + w;
      accR[bi] = (accR[bi] as number) + ar * w;
      accG[bi] = (accG[bi] as number) + ag * w;
      accB[bi] = (accB[bi] as number) + ab * w;
      accNX[bi] = (accNX[bi] as number) + nx * w;
      accNY[bi] = (accNY[bi] as number) + ny * w;
      accNZ[bi] = (accNZ[bi] as number) + nz * w;
    };

    // members: pool-major, instance-minor packed pairs — same order as the original
    const m = job.members;
    for (let k = 0; k + 1 < m.length; k += 2) {
      const pi = m[k] as number;
      const ii = m[k + 1] as number;
      const pool = pools[pi] as SplatPoolFlat;
      const x = pool.a[ii * 4] as number;
      // tile-LOCAL ground height of this tree (terrain-aware): its crown/trunk splat
      // shifted up by ly0; 0 everywhere on flat ground (forest — bit-identical path).
      const ly0 = (pool.a[ii * 4 + 1] as number) - baseY;
      const z = pool.a[ii * 4 + 2] as number;
      const s = pool.a[ii * 4 + 3] as number;
      const yaw = pool.b[ii * 4] as number;
      const cy = Math.cos(yaw);
      const sy = Math.sin(yaw);
      const lx0 = x - originX;
      const lz0 = z - originZ;
      const sp = pool.species;
      const br = sp.bricks;
      // crown bricks (scale, yaw, translate — zero lean in forest); VOLUME splat with
      // per-cell overlap fractions (exact port of splatBox)
      for (let bo = 0; bo < br.length; bo += SPECIES_BRICK_STRIDE) {
        const px = (br[bo] as number) * s;
        const py = (br[bo + 1] as number) * s + ly0;
        const pz = (br[bo + 2] as number) * s;
        const half = (br[bo + 3] as number) * s;
        const ar = br[bo + 4] as number;
        const ag = br[bo + 5] as number;
        const ab = br[bo + 6] as number;
        const bnx = br[bo + 7] as number;
        const bny = br[bo + 8] as number;
        const bnz = br[bo + 9] as number;
        const w = Math.max(0.05, br[bo + 10] as number);
        const lx = px * cy + pz * sy + lx0;
        const lz = pz * cy - px * sy + lz0;
        const nx = bnx * cy + bnz * sy;
        const nz = bnz * cy - bnx * sy;
        const x0 = Math.floor((lx - half) / cellSize);
        const x1 = Math.floor((lx + half) / cellSize);
        const y0 = Math.floor((py - half) / cellSize);
        const y1 = Math.floor((py + half) / cellSize);
        const z0 = Math.floor((lz - half) / cellSize);
        const z1 = Math.floor((lz + half) / cellSize);
        const inv = 1 / cellSize;
        for (let cz2 = z0; cz2 <= z1; cz2++) {
          const fz =
            Math.max(0, Math.min((cz2 + 1) * cellSize, lz + half) - Math.max(cz2 * cellSize, lz - half)) * inv;
          for (let cy2 = y0; cy2 <= y1; cy2++) {
            const fy =
              Math.max(0, Math.min((cy2 + 1) * cellSize, py + half) - Math.max(cy2 * cellSize, py - half)) * inv;
            for (let cx2 = x0; cx2 <= x1; cx2++) {
              const fx =
                Math.max(0, Math.min((cx2 + 1) * cellSize, lx + half) - Math.max(cx2 * cellSize, lx - half)) * inv;
              splatCell(cx2, cy2, cz2, ar, ag, ab, nx, bny, nz, w, fx * fy * fz);
            }
          }
        }
      }
      // trunk column: the tree's OWN ground (ly0) → crown base, radial horizontal
      // normals, full coverage. ly0=0 on flat ground — the legacy loop exactly.
      const topY = Math.max(cellSize, sp.crownMinY * s);
      const tcx = Math.floor(lx0 / cellSize);
      const tcz = Math.floor(lz0 / cellSize);
      for (let y = cellSize * 0.5; y < topY; y += cellSize) {
        splatCell(tcx, Math.floor((y + ly0) / cellSize), tcz, sp.barkR, sp.barkG, sp.barkB, cy, 0.15, -sy, 1, 1);
      }
    }

    // POST-pass: occupancy bits from per-cell coverage (gap-preserving threshold)
    for (let cz2 = 0; cz2 < cellsXZ; cz2++) {
      for (let cy2 = 0; cy2 < cellsY; cy2++) {
        for (let cx2 = 0; cx2 < cellsXZ; cx2++) {
          if ((cellW[cx2 + cy2 * cellsXZ + cz2 * cellsXZ * cellsY] as number) < OCC_COVER) continue;
          const bi =
            ((cx2 / BRICK_DIM) | 0) +
            ((cy2 / BRICK_DIM) | 0) * bricksX +
            ((cz2 / BRICK_DIM) | 0) * bricksX * bricksY;
          const cellBit = (cx2 & 3) + (cy2 & 3) * 4 + (cz2 & 3) * 16;
          if (cellBit < 32) occLo[bi] = ((occLo[bi] as number) | (1 << cellBit)) >>> 0;
          else occHi[bi] = ((occHi[bi] as number) | (1 << (cellBit - 32))) >>> 0;
        }
      }
    }

    out.push({ key, baseY, cellsY, occLo, occHi, accW, accR, accG, accB, accNX, accNY, accNZ });
  }
  return out;
}
