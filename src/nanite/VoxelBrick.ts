/**
 * VoxelBrick — the brick encoding for the 2-tier voxel-foliage subsystem
 * (docs/perf-runs/nanite-voxel-foliage-spec.md §4.3 / §6.4).
 *
 * THE AUTHORITATIVE BRICK LAYOUT. The offline voxelizer (Stage 1, §5.4) WRITES
 * bricks with the CPU helpers here; the runtime voxel raster (`kVoxBin`/`kRasterVox`,
 * Stage 2 §6) and the resolve voxel branch (§7.2) DECODE them on the GPU via
 * `elemU(gpu.voxelBricks, brickWord(bi, BRICK_*))` + the TSL helpers below. Both
 * sides MUST agree on this single layout — change it HERE only.
 *
 *   ──────────────────────────────────────────────────────────────────────────
 *   BRICK = a 4×4×4 = 64-cell block. BRICK_WORDS = 9 u32 (36 B).
 *   ──────────────────────────────────────────────────────────────────────────
 *   word 0  occupancy LO  — cells 0..31  (1 bit/cell; bit c = cell occupied)
 *   word 1  occupancy HI  — cells 32..63 (no u64 in WGSL r184 ⇒ 2×u32, §4.3)
 *   word 2  mean-normal   — oct snorm2x16 (the BRICK-mean normal, §5.4 / §6.4)
 *   word 3  normal spread — f32 bits in [0,1]: the normal-variance / SGGX scalar
 *                           (0 = flat/coherent, →1 = isotropic). Mean-normal path
 *                           IGNORES it at shade time; carried so the SGGX fallback
 *                           (§5.4 / §7.2.4) needs NO layout change (sized-in winner,
 *                           §4.3 "size the frozen reservation to the winner").
 *   word 4  albedo        — RGBA8 packed: RGB = density-weighted brick-mean color
 *                           (§5.4.3), A = brick coverage/density in [0,1] (§5.4.3).
 *   word 5  posX          — brick LOCAL-space center X (f32 bits)
 *   word 6  posY          — brick LOCAL-space center Y (f32 bits)
 *   word 7  posZ          — brick LOCAL-space center Z (f32 bits)
 *   word 8  half-extent   — brick LOCAL half-extent (f32 bits, = BRICK_DIM·cellSize·0.5)
 *                           ──────────────────────────────────────────────────────
 *                           THE OVERSIZED-SQUARE FIX: the raster (NaniteVoxelRaster.ts)
 *                           used to project the ≤128-brick BLOCK AABB (cluster word0-3)
 *                           as ONE footprint, so ~18 overlapping block-slabs painted
 *                           squares bigger than the whole tree. With a per-BRICK center
 *                           + extent here, the raster iterates the block's bricks and
 *                           paints each brick's SMALL footprint at its real grid cell ⇒
 *                           the voxel crown becomes tree-shaped (§6.2/§6.4).
 *   ──────────────────────────────────────────────────────────────────────────
 *
 * COARSE one-sample-per-brick is the HARD default (§6.4, correction #6). So:
 *  - occupancy (words 0/1) is BUILD/raster-only (drives the AABB-slab footprint +
 *    the per-cell fractional coverage that fed the mean); the RESOLVE never re-tests
 *    it (§4.4) — the resolve reads only the mean-normal + spread (words 2/3) and
 *    derives albedo via the leaf matParam tint (§4.4 / §7.2.6), so words 2..3 are the
 *    resolve-visible slice. Color (word 4) is a build/raster-side fallback if a future
 *    per-voxel-color path is taken; today it stays raster-only.
 *  - the election id is BRICK-granular (~7 bits), so per-cell ids never enter the
 *    32-bit election (§4.5/§6.4) and the §4.5 13-bit-cell fork is dissolved.
 *
 * The id-bit budget (§4.5): a voxel payload carries bit31=1 (the voxel marker — the
 * only bit free at BOTH the 128- and 256-tri caps) + (instId, brickIdx) in the low
 * ≤30 bits. That split is owned by the raster/resolve (Stage 2); the constants here
 * just pin the brick-record layout the voxelizer fills.
 */

import { Fn, uint } from 'three/tsl';
import type { NU, NV3 } from '../gpu/TSLTypes';
import { BRICK_WORDS, octDecode, octDecodeTsl, octEncode } from './GeometryRegistry';

export { BRICK_WORDS };

/** edge length of a brick cell-grid (4×4×4). */
export const BRICK_DIM = 4;
/** cells per brick = BRICK_DIM³. */
export const BRICK_CELLS = BRICK_DIM * BRICK_DIM * BRICK_DIM; // 64

/** word offsets inside a brick record (added to bi*BRICK_WORDS). */
export const BRICK_OCC_LO = 0;
export const BRICK_OCC_HI = 1;
export const BRICK_NORMAL = 2;
export const BRICK_SPREAD = 3;
export const BRICK_ALBEDO = 4;
/** per-brick LOCAL-space center (f32 bits) — the oversized-square fix (§6.2): the raster
 *  projects EACH brick's own AABB, not the whole ≤128-brick block. */
export const BRICK_POS_X = 5;
export const BRICK_POS_Y = 6;
export const BRICK_POS_Z = 7;
/** per-brick LOCAL half-extent (f32 bits) = BRICK_DIM·cellSize·0.5. */
export const BRICK_HALF = 8;

/** §4.1: a VOXEL cluster's word7 low-byte holds brickCount (reuses the triCount
 *  byte). ≤128 so it fits the u8 field AND keeps the brick-granular id ≤7 bits. */
export const MAX_BRICKS_PER_CLUSTER = 128;

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);
function f32Bits(v: number): number {
  f32Scratch[0] = v;
  return u32Scratch[0] as number;
}
function bitsF32(u: number): number {
  u32Scratch[0] = u >>> 0;
  return f32Scratch[0] as number;
}

/** linear cell index from (x,y,z) in [0,4) — x fastest, then y, then z. */
export function brickCellIndex(x: number, y: number, z: number): number {
  return x + y * BRICK_DIM + z * BRICK_DIM * BRICK_DIM;
}

/** pack one RGBA8 word (each component a 0..1 float → 8-bit). */
export function packRGBA8(r: number, g: number, b: number, a: number): number {
  const q = (v: number): number => Math.max(0, Math.min(255, Math.round(v * 255)));
  return (q(r) | (q(g) << 8) | (q(b) << 16) | (q(a) << 24)) >>> 0;
}

// ---------------------------------------------------------------------------
// CPU writer/reader — the offline voxelizer fills a flat Uint32Array of
// BRICK_WORDS-stride records; readBrick is the validation mirror of the GPU decode.
// ---------------------------------------------------------------------------

/** CPU-side brick value (what the voxelizer fits per brick). */
export interface BrickCPU {
  /** 64-bit occupancy split lo/hi (cells 0..31 / 32..63). */
  occLo: number;
  occHi: number;
  /** brick-mean unit normal (the bent normal, §5.4). */
  normal: [number, number, number];
  /** normal-variance / SGGX spread scalar in [0,1] (RAW pre-bend spread, §5.4). */
  spread: number;
  /** density-weighted brick-mean albedo, components in [0,1]. */
  albedo: [number, number, number];
  /** brick coverage/density in [0,1] (Σcov/total). */
  density: number;
  /** brick LOCAL-space center (crown-local) — the per-brick footprint origin (§6.2). */
  center: [number, number, number];
  /** brick LOCAL half-extent (= BRICK_DIM·cellSize·0.5) — the per-brick AABB radius. */
  half: number;
}

/** write one brick record into `bricks` at brick index `bi` (stride BRICK_WORDS). */
export function writeBrick(bricks: Uint32Array, bi: number, v: BrickCPU): void {
  const b = bi * BRICK_WORDS;
  bricks[b + BRICK_OCC_LO] = v.occLo >>> 0;
  bricks[b + BRICK_OCC_HI] = v.occHi >>> 0;
  bricks[b + BRICK_NORMAL] = octEncode(v.normal[0], v.normal[1], v.normal[2]);
  bricks[b + BRICK_SPREAD] = f32Bits(Math.max(0, Math.min(1, v.spread)));
  bricks[b + BRICK_ALBEDO] = packRGBA8(v.albedo[0], v.albedo[1], v.albedo[2], v.density);
  bricks[b + BRICK_POS_X] = f32Bits(v.center[0]);
  bricks[b + BRICK_POS_Y] = f32Bits(v.center[1]);
  bricks[b + BRICK_POS_Z] = f32Bits(v.center[2]);
  bricks[b + BRICK_HALF] = f32Bits(v.half);
}

/** read one brick record back (probe / validation mirror of the GPU decode). */
export function readBrick(bricks: Uint32Array, bi: number): BrickCPU {
  const b = bi * BRICK_WORDS;
  const alb = bricks[b + BRICK_ALBEDO] as number;
  return {
    occLo: bricks[b + BRICK_OCC_LO] as number,
    occHi: bricks[b + BRICK_OCC_HI] as number,
    normal: octDecode(bricks[b + BRICK_NORMAL] as number),
    spread: bitsF32(bricks[b + BRICK_SPREAD] as number),
    albedo: [(alb & 0xff) / 255, ((alb >>> 8) & 0xff) / 255, ((alb >>> 16) & 0xff) / 255],
    density: ((alb >>> 24) & 0xff) / 255,
    center: [
      bitsF32(bricks[b + BRICK_POS_X] as number),
      bitsF32(bricks[b + BRICK_POS_Y] as number),
      bitsF32(bricks[b + BRICK_POS_Z] as number),
    ],
    half: bitsF32(bricks[b + BRICK_HALF] as number),
  };
}

/** test cell occupancy on a CPU brick record (occLo/occHi as written by writeBrick). */
export function brickCellOccupied(occLo: number, occHi: number, cell: number): boolean {
  return cell < 32
    ? ((occLo >>> cell) & 1) === 1
    : ((occHi >>> (cell - 32)) & 1) === 1;
}

// ---------------------------------------------------------------------------
// GPU (TSL r184) helpers — the voxel raster (Stage 2) / resolve voxel branch read
// bricks with these against the gpu.voxelBricks StorageBufferNode<'uint'>.
//   word  = brickWord(bi, BRICK_NORMAL)   // u32 word index into voxelBricks
//   nrm   = brickNormalTsl(elemU(gpu.voxelBricks, word))   // mean normal as vec3
// ---------------------------------------------------------------------------

/** word index of a brick's `w`-th word in the flat voxelBricks buffer. */
export const brickWord = /*#__PURE__*/ Fn(([bi, w]: [NU, NU]): NU =>
  bi.mul(uint(BRICK_WORDS)).add(w),
);

/** GPU: decode an oct-packed brick mean-normal word into a unit vec3 (mirror of
 *  the CPU octDecode in readBrick). `packed` = the raw BRICK_NORMAL word. */
export const brickNormalTsl = /*#__PURE__*/ Fn(([packed]: [NU]): NV3 => octDecodeTsl(packed));
