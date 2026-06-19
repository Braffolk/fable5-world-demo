/**
 * VoxelizeCrown — the OFFLINE (build-time) palette voxelizer for the 2-tier
 * voxel-foliage subsystem (docs/perf-runs/nanite-voxel-foliage-spec.md §5).
 *
 * Stage 1 scope (§11): voxelize ONE crown's foliage mesh (an ExplicitSource) into
 * COARSE one-sample-per-brick bricks (§6.4 — the HARD default) at the per-crown
 * `voxelGridDim` knob (§3.2.bis, default ~180 effective cells / crown edge), with
 * FRACTIONAL/coverage density (§5.4.3 — mandatory for thin conifer needles so they
 * become a low-density smear, not a hole or a blob) and a density-weighted per-brick
 * MEAN normal + MEAN color. Cost-tolerant: this runs at boot, not per frame.
 *
 * Output = a flat array of `BrickCPU` records (VoxelBrick.ts codec) + the per-brick
 * grid origin/extent so a caller can place/raster them. The grid is laid out in
 * BRICKS of 4×4×4 cells: brickGridDim = ceil(voxelGridDim / BRICK_DIM) per axis.
 *
 * METHOD (offline, accuracy over speed):
 *  - One cubic voxel grid over the crown's local AABB (padded), edge = voxelGridDim
 *    CELLS. Cells group into 4×4×4 BRICKS.
 *  - Per triangle: rasterize into the cell grid via a conservative triangle/box SAT
 *    overlap (Akenine-Möller 13-axis SAT), supersampled S³ per cell for FRACTIONAL
 *    coverage. A needle quad ~0.018 m wide spans < 1 cell, so the supersample makes
 *    it a partial-coverage cell (density in (0,1)) instead of a hard 0/1 (§5.4.3).
 *  - Per cell accumulate: coverage Σ, normal·coverage Σ (the BENT/source normals —
 *    they are the MEAN per §5.4 caution), color·coverage Σ.
 *  - Per BRICK aggregate the 64 cells: occupancy bits (cell coverage ≥ occThresh),
 *    density = mean cell coverage, mean normal = normalize(Σ normal·cov), mean color
 *    = Σ color·cov / Σ cov, spread = 1 − |mean-vector-length| (RAW variance proxy,
 *    carried for the SGGX fallback, ignored by the mean-normal shade path, §4.3/§7).
 *
 * The triangle SAT is conservative (no holes — Risk #7 / §5.4.2); the supersample is
 * the fractional-density layer on top (§5.4.3). This is the COARSE path: per-cell
 * occupancy is build/raster-only; the resolve reads only the per-brick mean normal.
 */

import type { ExplicitSource, GeometryRegistry, MeshHandle } from './GeometryRegistry';
import {
  type BrickCPU,
  BRICK_DIM,
  brickCellIndex,
  MAX_BRICKS_PER_CLUSTER,
  writeBrick,
} from './VoxelBrick';

/** default per-crown voxel DETAIL knob (§3.2.bis): effective CELL edge count over the
 *  whole crown. ~180 ⇒ a leaf spans ≥2-3 cells at the near transition (§3.2). A
 *  build-time uniform (?voxgrid=), ratcheted COARSER in Stage 5 until KG-0b breaks. */
export const DEFAULT_VOXEL_GRID_DIM = 180;

/** supersamples per cell per axis for fractional coverage (§5.4.3). S=3 ⇒ 27 sub-
 *  samples/cell — enough to resolve a sub-cell needle as partial density. Offline. */
const COVERAGE_SUPERSAMPLE = 3;

/** a cell counts as OCCUPIED (occupancy bit set) once its fractional coverage clears
 *  this — low so a thin-needle smear still sets the bit (no bald spots). Occupancy is
 *  build/raster-only in the coarse path (§4.4); density carries the real weight. */
const OCC_COVERAGE_THRESHOLD = 0.02;

export interface CrownVoxelization {
  /** flat BrickCPU records in brick-grid order (x fastest, then y, then z). */
  bricks: BrickCPU[];
  /** indices (into `bricks`) of the bricks that are NON-EMPTY (density > 0). The
   *  voxelizer keeps the full grid for addressing but a caller only uploads/draws
   *  the occupied ones. */
  occupied: number[];
  /** brick-grid edge counts (ceil(voxelGridDim / BRICK_DIM) per axis). */
  brickGrid: { x: number; y: number; z: number };
  /** cell-grid edge counts (the requested voxelGridDim rounded UP to a brick multiple). */
  cellGrid: { x: number; y: number; z: number };
  /** local-space grid origin (min corner of the padded crown AABB). */
  origin: [number, number, number];
  /** world size of ONE cell (uniform; cubic grid). */
  cellSize: number;
  /** stats for the §5.3 budget print. */
  stats: {
    triangles: number;
    cellsTouched: number;
    occupiedBricks: number;
    totalBricks: number;
    meanDensity: number;
    voxelizeMs: number;
  };
}

// ---------------------------------------------------------------------------
// Triangle / axis-aligned-box overlap — Akenine-Möller 13-axis SAT.
// Box is centered at `c` with half-extent `h` (uniform). Triangle verts v0/v1/v2.
// Returns true if they intersect (conservative — used to gate sub-sample tests).
// ---------------------------------------------------------------------------

function triBoxOverlap(
  cx: number, cy: number, cz: number,
  hx: number, hy: number, hz: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  ccx: number, ccy: number, ccz: number,
): boolean {
  // move triangle into box-centered space
  const v0x = ax - cx, v0y = ay - cy, v0z = az - cz;
  const v1x = bx - cx, v1y = by - cy, v1z = bz - cz;
  const v2x = ccx - cx, v2y = ccy - cy, v2z = ccz - cz;
  // edges
  const e0x = v1x - v0x, e0y = v1y - v0y, e0z = v1z - v0z;
  const e1x = v2x - v1x, e1y = v2y - v1y, e1z = v2z - v1z;
  const e2x = v0x - v2x, e2y = v0y - v2y, e2z = v0z - v2z;

  // 9 cross-product axes (edge × box axis): reject if the triangle's projection
  // interval [min,max] clears the box radius r (separating axis found).
  const sep = (p0: number, p1: number, r: number): boolean =>
    Math.min(p0, p1) > r || Math.max(p0, p1) < -r;

  // a00..a02: e0
  let p0: number, p1: number, rad: number;
  // e0 x X = (0, -e0z, e0y)
  p0 = e0z * v0y - e0y * v0z;
  p1 = e0z * v2y - e0y * v2z;
  rad = hy * Math.abs(e0z) + hz * Math.abs(e0y);
  if (sep(p0, p1, rad)) return false;
  // e0 x Y = (e0z, 0, -e0x)
  p0 = -e0z * v0x + e0x * v0z;
  p1 = -e0z * v2x + e0x * v2z;
  rad = hx * Math.abs(e0z) + hz * Math.abs(e0x);
  if (sep(p0, p1, rad)) return false;
  // e0 x Z = (-e0y, e0x, 0)
  p0 = e0y * v1x - e0x * v1y;
  p1 = e0y * v2x - e0x * v2y;
  rad = hx * Math.abs(e0y) + hy * Math.abs(e0x);
  if (sep(p0, p1, rad)) return false;

  // e1
  p0 = e1z * v0y - e1y * v0z;
  p1 = e1z * v2y - e1y * v2z;
  rad = hy * Math.abs(e1z) + hz * Math.abs(e1y);
  if (sep(p0, p1, rad)) return false;
  p0 = -e1z * v0x + e1x * v0z;
  p1 = -e1z * v2x + e1x * v2z;
  rad = hx * Math.abs(e1z) + hz * Math.abs(e1x);
  if (sep(p0, p1, rad)) return false;
  p0 = e1y * v0x - e1x * v0y;
  p1 = e1y * v1x - e1x * v1y;
  rad = hx * Math.abs(e1y) + hy * Math.abs(e1x);
  if (sep(p0, p1, rad)) return false;

  // e2
  p0 = e2z * v0y - e2y * v0z;
  p1 = e2z * v1y - e2y * v1z;
  rad = hy * Math.abs(e2z) + hz * Math.abs(e2y);
  if (sep(p0, p1, rad)) return false;
  p0 = -e2z * v0x + e2x * v0z;
  p1 = -e2z * v1x + e2x * v1z;
  rad = hx * Math.abs(e2z) + hz * Math.abs(e2x);
  if (sep(p0, p1, rad)) return false;
  p0 = e2y * v0x - e2x * v0y;
  p1 = e2y * v1x - e2x * v1y;
  rad = hx * Math.abs(e2y) + hy * Math.abs(e2x);
  if (sep(p0, p1, rad)) return false;

  // 3 box-face axes: triangle AABB vs box
  if (Math.min(v0x, v1x, v2x) > hx || Math.max(v0x, v1x, v2x) < -hx) return false;
  if (Math.min(v0y, v1y, v2y) > hy || Math.max(v0y, v1y, v2y) < -hy) return false;
  if (Math.min(v0z, v1z, v2z) > hz || Math.max(v0z, v1z, v2z) < -hz) return false;

  // triangle-plane axis
  const nx = e0y * e1z - e0z * e1y;
  const ny = e0z * e1x - e0x * e1z;
  const nz = e0x * e1y - e0y * e1x;
  const d = nx * v0x + ny * v0y + nz * v0z;
  const r = hx * Math.abs(nx) + hy * Math.abs(ny) + hz * Math.abs(nz);
  return Math.abs(d) <= r;
}

// scratch accumulators per cell (parallel flat arrays indexed by cell linear index)
interface CellAccum {
  cov: Float32Array;   // Σ coverage
  nx: Float32Array;    // Σ normal.x · cov
  ny: Float32Array;
  nz: Float32Array;
  cr: Float32Array;    // Σ color.r · cov
  cg: Float32Array;
  cb: Float32Array;
}

/**
 * Voxelize one crown foliage mesh (LOCAL space) into coarse bricks.
 *
 * @param src         the crown foliage ExplicitSource (positions/normals/indices).
 * @param albedo      the per-species foliage tint (0..1 linear) — the density-weighted
 *                    color falls back to this when the source carries no per-vert color
 *                    (our leaves are flat-tinted via matParam, §4.4 / §7.2.6). Used as
 *                    a uniform per-vertex color so the brick mean color is meaningful.
 * @param voxelGridDim effective CELL edge count over the crown (DEFAULT ~180).
 */
export function voxelizeCrown(
  src: ExplicitSource,
  albedo: { r: number; g: number; b: number },
  voxelGridDim: number = DEFAULT_VOXEL_GRID_DIM,
): CrownVoxelization {
  const t0 = performance.now();
  const pos = src.positions;
  const nrm = src.normals;
  const idx = src.indices;
  const triCount = idx.length / 3;

  // ---- crown AABB (local) + padded cubic grid ----------------------------
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < pos.length; i += 3) {
    const x = pos[i] as number, y = pos[i + 1] as number, z = pos[i + 2] as number;
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  // cubic grid spanning the largest extent (keeps cells cubic — §3.2 cellWorld math)
  const extX = Math.max(maxX - minX, 1e-3);
  const extY = Math.max(maxY - minY, 1e-3);
  const extZ = Math.max(maxZ - minZ, 1e-3);
  const maxExt = Math.max(extX, extY, extZ);
  // round the cell count UP to a whole brick (4) so the brick grid tiles exactly
  const cellEdge = Math.max(BRICK_DIM, Math.ceil(voxelGridDim / BRICK_DIM) * BRICK_DIM);
  const cellSize = maxExt / cellEdge;
  // pad half a cell so boundary geometry isn't clipped, and center the crown
  const pad = cellSize * 0.5;
  const originX = minX - (maxExt - extX) * 0.5 - pad;
  const originY = minY - (maxExt - extY) * 0.5 - pad;
  const originZ = minZ - (maxExt - extZ) * 0.5 - pad;
  // one extra cell of headroom each side absorbs the centering + pad
  const cellGridX = cellEdge + 2;
  const cellGridY = cellEdge + 2;
  const cellGridZ = cellEdge + 2;
  // grow brick grid to cover the padded cell grid
  const brickGridX = Math.ceil(cellGridX / BRICK_DIM);
  const brickGridY = Math.ceil(cellGridY / BRICK_DIM);
  const brickGridZ = Math.ceil(cellGridZ / BRICK_DIM);
  const cgX = brickGridX * BRICK_DIM;
  const cgY = brickGridY * BRICK_DIM;
  const cgZ = brickGridZ * BRICK_DIM;
  const cellTotal = cgX * cgY * cgZ;

  const acc: CellAccum = {
    cov: new Float32Array(cellTotal),
    nx: new Float32Array(cellTotal),
    ny: new Float32Array(cellTotal),
    nz: new Float32Array(cellTotal),
    cr: new Float32Array(cellTotal),
    cg: new Float32Array(cellTotal),
    cb: new Float32Array(cellTotal),
  };
  const cellLin = (cx: number, cy: number, cz: number): number =>
    cx + cy * cgX + cz * cgX * cgY;

  // ---- rasterize each triangle into the cell grid (SAT + supersample) -----
  const half = cellSize * 0.5;
  const S = COVERAGE_SUPERSAMPLE;
  const subW = 1 / (S * S * S); // per sub-sample coverage weight
  let cellsTouched = 0;
  for (let t = 0; t < triCount; t++) {
    const i0 = (idx[t * 3] as number) * 3;
    const i1 = (idx[t * 3 + 1] as number) * 3;
    const i2 = (idx[t * 3 + 2] as number) * 3;
    const ax = pos[i0] as number, ay = pos[i0 + 1] as number, az = pos[i0 + 2] as number;
    const bx = pos[i1] as number, by = pos[i1 + 1] as number, bz = pos[i1 + 2] as number;
    const ccx = pos[i2] as number, ccy = pos[i2 + 1] as number, ccz = pos[i2 + 2] as number;
    // face normal (geometric) — used for the per-cell normal accumulation. The
    // source vert normals are the BENT crown normals (§5.4 caution); we average the
    // face normal of the covering tris, which equals averaging the bent normals.
    const fnx0 = nrm[i0] as number, fny0 = nrm[i0 + 1] as number, fnz0 = nrm[i0 + 2] as number;
    const fnx1 = nrm[i1] as number, fny1 = nrm[i1 + 1] as number, fnz1 = nrm[i1 + 2] as number;
    const fnx2 = nrm[i2] as number, fny2 = nrm[i2 + 1] as number, fnz2 = nrm[i2 + 2] as number;
    let mnx = fnx0 + fnx1 + fnx2;
    let mny = fny0 + fny1 + fny2;
    let mnz = fnz0 + fnz1 + fnz2;
    const ml = Math.hypot(mnx, mny, mnz) || 1;
    mnx /= ml; mny /= ml; mnz /= ml;

    // tri AABB → cell range (clamped to grid)
    const tMinX = Math.min(ax, bx, ccx), tMaxX = Math.max(ax, bx, ccx);
    const tMinY = Math.min(ay, by, ccy), tMaxY = Math.max(ay, by, ccy);
    const tMinZ = Math.min(az, bz, ccz), tMaxZ = Math.max(az, bz, ccz);
    const c0x = Math.max(0, Math.floor((tMinX - originX) / cellSize) - 1);
    const c1x = Math.min(cgX - 1, Math.floor((tMaxX - originX) / cellSize) + 1);
    const c0y = Math.max(0, Math.floor((tMinY - originY) / cellSize) - 1);
    const c1y = Math.min(cgY - 1, Math.floor((tMaxY - originY) / cellSize) + 1);
    const c0z = Math.max(0, Math.floor((tMinZ - originZ) / cellSize) - 1);
    const c1z = Math.min(cgZ - 1, Math.floor((tMaxZ - originZ) / cellSize) + 1);

    for (let cz = c0z; cz <= c1z; cz++) {
      const ccz0 = originZ + (cz + 0.5) * cellSize;
      for (let cy = c0y; cy <= c1y; cy++) {
        const ccy0 = originY + (cy + 0.5) * cellSize;
        for (let cx = c0x; cx <= c1x; cx++) {
          const ccx0 = originX + (cx + 0.5) * cellSize;
          // conservative gate: does the triangle touch this cell at all?
          if (!triBoxOverlap(ccx0, ccy0, ccz0, half, half, half, ax, ay, az, bx, by, bz, ccx, ccy, ccz)) {
            continue;
          }
          // fractional coverage via S³ sub-cells, each tested as a tiny box
          const subHalf = half / S;
          let cov = 0;
          for (let sz = 0; sz < S; sz++) {
            const scz = ccz0 + (sz - (S - 1) * 0.5) * (cellSize / S);
            for (let sy = 0; sy < S; sy++) {
              const scy = ccy0 + (sy - (S - 1) * 0.5) * (cellSize / S);
              for (let sx = 0; sx < S; sx++) {
                const scx = ccx0 + (sx - (S - 1) * 0.5) * (cellSize / S);
                if (triBoxOverlap(scx, scy, scz, subHalf, subHalf, subHalf, ax, ay, az, bx, by, bz, ccx, ccy, ccz)) {
                  cov += subW;
                }
              }
            }
          }
          if (cov <= 0) continue;
          const li = cellLin(cx, cy, cz);
          if (acc.cov[li] === 0) cellsTouched++;
          acc.cov[li] = (acc.cov[li] as number) + cov;
          acc.nx[li] = (acc.nx[li] as number) + mnx * cov;
          acc.ny[li] = (acc.ny[li] as number) + mny * cov;
          acc.nz[li] = (acc.nz[li] as number) + mnz * cov;
          acc.cr[li] = (acc.cr[li] as number) + albedo.r * cov;
          acc.cg[li] = (acc.cg[li] as number) + albedo.g * cov;
          acc.cb[li] = (acc.cb[li] as number) + albedo.b * cov;
        }
      }
    }
  }

  // ---- aggregate cells → bricks (the COARSE one-sample-per-brick step) ----
  const bricks: BrickCPU[] = [];
  const occupied: number[] = [];
  let densitySum = 0;
  for (let bz = 0; bz < brickGridZ; bz++) {
    for (let by = 0; by < brickGridY; by++) {
      for (let bx = 0; bx < brickGridX; bx++) {
        let occLo = 0, occHi = 0;
        let bcov = 0;     // Σ over the 64 cells of cell coverage (clamped 0..1)
        let bnx = 0, bny = 0, bnz = 0;
        let bcr = 0, bcg = 0, bcb = 0;
        let bcolW = 0;
        for (let lz = 0; lz < BRICK_DIM; lz++) {
          const cz = bz * BRICK_DIM + lz;
          for (let ly = 0; ly < BRICK_DIM; ly++) {
            const cy = by * BRICK_DIM + ly;
            for (let lx = 0; lx < BRICK_DIM; lx++) {
              const cx = bx * BRICK_DIM + lx;
              const li = cellLin(cx, cy, cz);
              // a cell may have been hit by several tris ⇒ clamp coverage to 1
              const cellCov = Math.min(1, acc.cov[li] as number);
              if (cellCov <= 0) continue;
              const cell = brickCellIndex(lx, ly, lz);
              if (cellCov >= OCC_COVERAGE_THRESHOLD) {
                if (cell < 32) occLo |= (1 << cell);
                else occHi |= (1 << (cell - 32));
              }
              bcov += cellCov;
              // normal/color accumulators are already coverage-weighted Σ; weight by
              // the cell's TOTAL Σcov so a cell hit by many tris dominates correctly
              const w = acc.cov[li] as number;
              bnx += acc.nx[li] as number;
              bny += acc.ny[li] as number;
              bnz += acc.nz[li] as number;
              bcr += acc.cr[li] as number;
              bcg += acc.cg[li] as number;
              bcb += acc.cb[li] as number;
              bcolW += w;
            }
          }
        }
        const bi = bricks.length;
        if (bcov <= 0 || bcolW <= 0) {
          // empty brick — still emit a record so the grid addresses linearly, but
          // mark it density 0 (callers skip via `occupied`)
          bricks.push({
            occLo: 0, occHi: 0, normal: [0, 1, 0], spread: 0, albedo: [0, 0, 0], density: 0,
          });
          continue;
        }
        // brick MEAN normal (bent) = normalize of the coverage-weighted sum
        const nlen = Math.hypot(bnx, bny, bnz);
        const nrmLen = nlen / bcolW; // mean-vector length in [0,1] → coherence
        const normal: [number, number, number] = nlen > 1e-8
          ? [bnx / nlen, bny / nlen, bnz / nlen]
          : [0, 1, 0];
        // spread = 1 − |mean| (0 = all normals coherent, →1 = isotropic). RAW proxy
        // (§4.3 word3) — the mean-normal shade path ignores it; SGGX fallback reads it.
        const spread = Math.max(0, Math.min(1, 1 - nrmLen));
        // density = mean cell coverage over the brick's 64 cells (Σcov / 64, §5.4.3)
        const density = Math.min(1, bcov / (BRICK_DIM * BRICK_DIM * BRICK_DIM));
        const color: [number, number, number] = [bcr / bcolW, bcg / bcolW, bcb / bcolW];
        bricks.push({ occLo, occHi, normal, spread, albedo: color, density });
        occupied.push(bi);
        densitySum += density;
      }
    }
  }

  const voxelizeMs = performance.now() - t0;
  return {
    bricks,
    occupied,
    brickGrid: { x: brickGridX, y: brickGridY, z: brickGridZ },
    cellGrid: { x: cgX, y: cgY, z: cgZ },
    origin: [originX, originY, originZ],
    cellSize,
    stats: {
      triangles: triCount,
      cellsTouched,
      occupiedBricks: occupied.length,
      totalBricks: bricks.length,
      meanDensity: occupied.length > 0 ? densitySum / occupied.length : 0,
      voxelizeMs,
    },
  };
}

/** world-space center of a brick in the grid (local crown space) — for the throwaway
 *  debug render's instanced-box placement. */
export function brickCenterLocal(
  vox: CrownVoxelization,
  brickIndex: number,
): [number, number, number] {
  const { x: bgx, y: bgy } = vox.brickGrid;
  const bx = brickIndex % bgx;
  const by = Math.floor(brickIndex / bgx) % bgy;
  const bz = Math.floor(brickIndex / (bgx * bgy));
  const bs = BRICK_DIM * vox.cellSize;
  return [
    vox.origin[0] + (bx + 0.5) * bs,
    vox.origin[1] + (by + 0.5) * bs,
    vox.origin[2] + (bz + 0.5) * bs,
  ];
}

/** world size of one brick edge (BRICK_DIM cells). */
export function brickWorldSize(vox: CrownVoxelization): number {
  return BRICK_DIM * vox.cellSize;
}

// ---------------------------------------------------------------------------
// REGISTRY WIRING (§5.2/§5.3) — the offline-voxelizer ↔ GeometryRegistry path:
// reserve bricks BEFORE build() (addLate freezes caps), then append + register a
// voxel:7 sibling head AFTER build(). Stage 1 exercises this whole path so the
// addLate.bricks precondition and appendBricks are proven before Stage 2's raster.
// ---------------------------------------------------------------------------

export interface PreparedVoxelCrown {
  vox: CrownVoxelization;
  /** number of OCCUPIED bricks (what gets uploaded; the reservation count). */
  brickCount: number;
}

/**
 * Voxelize a crown and report the OCCUPIED brick count to reserve. Call BEFORE
 * `reg.build()` and pass `prep.brickCount` into `reg.addLate({ bricks })` (summing
 * across all crowns) — the registry freezes the brick buffer size at build (§5.3).
 */
export function prepareVoxelCrown(
  src: ExplicitSource,
  albedo: { r: number; g: number; b: number },
  voxelGridDim: number = DEFAULT_VOXEL_GRID_DIM,
): PreparedVoxelCrown {
  const vox = voxelizeCrown(src, albedo, voxelGridDim);
  return { vox, brickCount: vox.occupied.length };
}

/**
 * Append a prepared crown's OCCUPIED bricks to the registry post-build and register a
 * `voxel:7` sibling mesh head over the SAME instances (§5.2 / Stage-2 A1). Returns the
 * head + the appended brick range. The head carries NO triangle geometry — its clusters
 * point at BRICKS (word6=brickBase, word7-lowbyte=brickCount) and its matClass=voxel
 * keeps them out of the triangle raster (§4.1). Re-clusterizing the full leaf crown
 * OVERFLOWED the late caps (§A1), so the head is authored DIRECTLY via registerVoxelHead
 * (0 verts/tris, only `blocks.length` clusters — within the `?voxreg` reservation).
 *
 * The occupied bricks (appended in grid order) are split into ≤MAX_BRICKS_PER_CLUSTER
 * contiguous BLOCKS (the §5.3 per-coarse-cluster fit), one cluster each, with a per-block
 * brick-AABB bound (the kVoxBin AABB projection, §6.2). `matParam` should be the same
 * packed leaf tint as the leaf head (§7.2.6). `leafSource` is unused (kept for signature
 * stability — the head no longer carries a placeholder tri from it).
 */
export function appendVoxelCrown(
  reg: GeometryRegistry,
  prep: PreparedVoxelCrown,
  _leafSource: ExplicitSource,
  opts: { matParam: number; swayPad?: number; maxDist: number; nearDist?: number; label?: string },
): { head: MeshHandle; brickBase: number; brickCount: number; clusters: number } {
  const { vox, brickCount } = prep;
  // append the occupied bricks into gpu.voxelBricks (uploads via pushRange), in grid
  // (occupied[]) order so the cluster BLOCKS below address contiguous brick sub-ranges.
  const brickBase = reg.appendBricks(brickCount, (bricks, base) => {
    for (let i = 0; i < brickCount; i++) {
      const bi = vox.occupied[i] as number;
      const brick = vox.bricks[bi];
      if (brick) writeBrick(bricks, base + i, brick);
    }
  });
  // split into ≤MAX_BRICKS_PER_CLUSTER blocks (§5.3) + per-block AABB (over the occupied
  // bricks' local-space centers; brick half-extent = BRICK_DIM·cellSize·0.5).
  const halfBrick = BRICK_DIM * vox.cellSize * 0.5;
  const blocks: {
    brickBase: number;
    brickCount: number;
    aabb: { min: [number, number, number]; max: [number, number, number] };
  }[] = [];
  for (let start = 0; start < brickCount; start += MAX_BRICKS_PER_CLUSTER) {
    const count = Math.min(MAX_BRICKS_PER_CLUSTER, brickCount - start);
    let mnX = Infinity, mnY = Infinity, mnZ = Infinity;
    let mxX = -Infinity, mxY = -Infinity, mxZ = -Infinity;
    for (let i = 0; i < count; i++) {
      const c = brickCenterLocal(vox, vox.occupied[start + i] as number);
      mnX = Math.min(mnX, c[0] - halfBrick); mxX = Math.max(mxX, c[0] + halfBrick);
      mnY = Math.min(mnY, c[1] - halfBrick); mxY = Math.max(mxY, c[1] + halfBrick);
      mnZ = Math.min(mnZ, c[2] - halfBrick); mxZ = Math.max(mxZ, c[2] + halfBrick);
    }
    blocks.push({
      brickBase: brickBase + start,
      brickCount: count,
      aabb: { min: [mnX, mnY, mnZ], max: [mxX, mxY, mxZ] },
    });
  }
  const head = reg.registerVoxelHead(opts.matParam, blocks, {
    swayPad: opts.swayPad ?? 3.8,
    maxDist: opts.maxDist,
    label: opts.label ?? 'voxel',
  });
  // voxel-foliage (spec §3 / Stage 3a): the NEAR side of the mesh→voxel handoff — the
  // voxel head seeds ONLY beyond nearDist (= transitionDist) so it renders the mid/far
  // band; the leaf sibling's maxDist=transitionDist owns nearer. 0/undefined = voxel
  // everywhere (the ?forcevox debug route, with the leaf head suppressed).
  if (opts.nearDist && opts.nearDist > 0) reg.setNearDistance(head, opts.nearDist);
  return { head, brickBase, brickCount, clusters: blocks.length };
}
