/**
 * VoxelBrickCore — the CPU side of the brick codec (layout doc: VoxelBrick.ts),
 * THREE-FREE so the offline voxelizer (VoxelizeCrown.ts) can run inside module
 * Workers (cold-boot crown prep, 2026-07-04). VoxelBrick.ts re-exports all of
 * this next to its TSL/GPU helpers — main-thread code keeps importing from
 * VoxelBrick; worker-reachable code imports from HERE (the FarTilesSplat
 * discipline: no three/tsl, no GeometryRegistry in the import chain).
 *
 * BRICK_WORDS / octEncode / octDecode moved here from GeometryRegistry.ts
 * (which re-exports them unchanged) — they are pure math the codec needs.
 */

/** u32 words per packed brick record in gpu.voxelBricks (see VoxelBrick.ts layout). */
export const BRICK_WORDS = 9;

/** edge length of a brick cell-grid (4×4×4). */
export const BRICK_DIM = 4;

/** word offsets inside a brick record (added to bi*BRICK_WORDS). */
export const BRICK_OCC_LO = 0;
export const BRICK_OCC_HI = 1;
export const BRICK_NORMAL = 2;
const BRICK_SPREAD = 3;
export const BRICK_ALBEDO = 4;
/** per-brick LOCAL-space center (f32 bits) — the oversized-square fix (§6.2): the raster
 *  projects EACH brick's own AABB, not the whole ≤128-brick block. */
export const BRICK_POS_X = 5;
const BRICK_POS_Y = 6;
const BRICK_POS_Z = 7;
/** per-brick LOCAL half-extent (f32 bits) = BRICK_DIM·cellSize·0.5. */
export const BRICK_HALF = 8;

/** §4.1: a VOXEL cluster's word7 low-byte holds brickCount (reuses the triCount
 *  byte). ≤128 so it fits the u8 field AND keeps the brick-granular id ≤7 bits. */
export const MAX_BRICKS_PER_CLUSTER = 128;

function snorm16(v: number): number {
  const c = Math.max(-1, Math.min(1, v));
  const r = Math.round(c * 32767);
  return (r < 0 ? r + 65536 : r) & 0xffff;
}

/** octahedral-encode a (not necessarily unit) normal into snorm2x16 */
export function octEncode(nx: number, ny: number, nz: number): number {
  const a = Math.abs(nx) + Math.abs(ny) + Math.abs(nz);
  let ox = 0;
  let oy = 0;
  if (a > 1e-20) {
    ox = nx / a;
    oy = ny / a;
    if (nz < 0) {
      const tx = (1 - Math.abs(oy)) * (ox >= 0 ? 1 : -1);
      const ty = (1 - Math.abs(ox)) * (oy >= 0 ? 1 : -1);
      ox = tx;
      oy = ty;
    }
  }
  return (snorm16(ox) | (snorm16(oy) << 16)) >>> 0;
}

/** mirror of the GPU decode: unpack2x16snorm + oct → unit vector */
export function octDecode(packed: number): [number, number, number] {
  const sx = ((packed & 0xffff) << 16) >> 16;
  const sy = ((packed >>> 16) << 16) >> 16;
  let fx = Math.max(sx / 32767, -1);
  let fy = Math.max(sy / 32767, -1);
  const nz = 1 - Math.abs(fx) - Math.abs(fy);
  const t = Math.max(-nz, 0);
  fx += fx >= 0 ? -t : t;
  fy += fy >= 0 ? -t : t;
  const l = Math.hypot(fx, fy, nz) || 1;
  return [fx / l, fy / l, nz / l];
}

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
function packRGBA8(r: number, g: number, b: number, a: number): number {
  const q = (v: number): number => Math.max(0, Math.min(255, Math.round(v * 255)));
  return (q(r) | (q(g) << 8) | (q(b) << 16) | (q(a) << 24)) >>> 0;
}

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

/** lean read of just a packed brick's LOCAL center + half-extent (words[bi*BRICK_WORDS+…]),
 *  avoiding the full readBrick decode — the hot AABB path in the append codec. */
export function brickCenterHalf(
  words: Uint32Array,
  bi: number,
): [number, number, number, number] {
  const b = bi * BRICK_WORDS;
  return [
    bitsF32(words[b + BRICK_POS_X] as number),
    bitsF32(words[b + BRICK_POS_Y] as number),
    bitsF32(words[b + BRICK_POS_Z] as number),
    bitsF32(words[b + BRICK_HALF] as number),
  ];
}

