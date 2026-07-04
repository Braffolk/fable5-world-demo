/**
 * VoxelBrick — the brick encoding for the 2-tier voxel-foliage subsystem
 * (docs/perf-runs/nanite-voxel-foliage-spec.md §4.3 / §6.4).
 *
 * THE AUTHORITATIVE BRICK LAYOUT. The offline voxelizer (Stage 1, §5.4) WRITES
 * bricks with the CPU helpers here; the runtime voxel raster (`kVoxScatter`,
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
import { octDecodeTsl } from './GeometryRegistry';
import { BRICK_WORDS } from './VoxelBrickCore';

// CPU codec + constants live in VoxelBrickCore.ts (THREE-FREE so the offline
// voxelizer can run in module Workers — cold-boot crown prep, 2026-07-04).
// Re-exported here so main-thread importers keep a single import site.
export * from './VoxelBrickCore';

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
