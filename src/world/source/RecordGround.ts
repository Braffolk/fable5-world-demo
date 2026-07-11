/**
 * RecordGround — the ONE derive-if-absent rule for record grounding (SPEC §2, S2;
 * extracted PURE for S8 so the StreamBrain worker's fartile bakes ground trees
 * BIT-IDENTICALLY to the main-thread instance paths — ChunkContent imports the
 * same function, so there is exactly one derivation in the codebase):
 *   y    = bilinear height sample at (x,z) on the texel-centered lattice
 *   yaw  = pcg2d hash(cx,cz, xq,zq) · τ   (xq/zq = u16-quantized local coords,
 *          the wire lattice — stable across refetches)
 *   lean = slope-normal.xz · 0.18 + (hash − ½)·0.12  (Scatter.ts:469-470 formula)
 */

const TAU = 6.2831853;

export type GroundDeriver = (xLocal: number, zLocal: number) => { h: number; yaw: number; leanX: number; leanZ: number };

/** grounding + cosmetic-orientation deriver for one chunk's decoded height window:
 *  samples bilinearly on the texel-centered lattice, slope via central differences. */
export function makeGroundDeriver(heights: Float32Array, res: number, footprint: number, cx: number, cz: number): GroundDeriver {
  const texel = footprint / (res - 1);
  const sample = (gx: number, gz: number): number => {
    const cgx = Math.min(Math.max(gx, 0), res - 1.001);
    const cgz = Math.min(Math.max(gz, 0), res - 1.001);
    const x0 = Math.floor(cgx);
    const z0 = Math.floor(cgz);
    const fx = cgx - x0;
    const fz = cgz - z0;
    const at = (xx: number, zz: number): number => heights[Math.min(zz, res - 1) * res + Math.min(xx, res - 1)] as number;
    return (
      at(x0, z0) * (1 - fx) * (1 - fz) +
      at(x0 + 1, z0) * fx * (1 - fz) +
      at(x0, z0 + 1) * (1 - fx) * fz +
      at(x0 + 1, z0 + 1) * fx * fz
    );
  };
  return (xLocal, zLocal) => {
    const gx = xLocal / texel - 0.5; // texel (i,j) centers at (i+0.5)·texel
    const gz = zLocal / texel - 0.5;
    const h = sample(gx, gz);
    // slope normal: central differences, normalized (-dh/dx, 1, -dh/dz)
    const dhdx = (sample(gx + 1, gz) - sample(gx - 1, gz)) / (2 * texel);
    const dhdz = (sample(gx, gz + 1) - sample(gx, gz - 1)) / (2 * texel);
    const inv = 1 / Math.hypot(dhdx, 1, dhdz);
    const xq = Math.min(65535, Math.max(0, Math.round((xLocal / footprint) * 65535)));
    const zq = Math.min(65535, Math.max(0, Math.round((zLocal / footprint) * 65535)));
    const [h1, h2] = pcg2d(((cx & 0x7fff) << 16) ^ xq, ((cz & 0x7fff) << 16) ^ zq);
    return {
      h,
      yaw: h1 * TAU,
      leanX: -dhdx * inv * 0.18 + (h2 - 0.5) * 0.12,
      leanZ: -dhdz * inv * 0.18 + (pcg2d(xq ^ 0x5b1e, zq ^ 0x2c9d)[0] - 0.5) * 0.12,
    };
  };
}

/** CPU pcg2d, same mix as gpu/passes/Scatter.ts pcg2d — returns two [0,1) floats. */
export function pcg2d(px: number, pz: number): [number, number] {
  const M = 1664525;
  let a = Math.imul(px >>> 0, M) + 1013904223;
  let b = Math.imul(pz >>> 0, M) + 1013904223;
  a = (a + Math.imul(b, M)) >>> 0;
  b = (b + Math.imul(a, M)) >>> 0;
  a = (a ^ (a >>> 16)) >>> 0;
  b = (b ^ (b >>> 16)) >>> 0;
  a = (a + Math.imul(b, M)) >>> 0;
  b = (b + Math.imul(a, M)) >>> 0;
  a = (a ^ (a >>> 16)) >>> 0;
  b = (b ^ (b >>> 16)) >>> 0;
  return [(a & 0xffffff) / 16777216, (b & 0xffffff) / 16777216];
}
