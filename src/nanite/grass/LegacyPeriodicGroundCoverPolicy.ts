/** @deprecated Remove with the legacy Sannikov/GCAR periodic renderer after the
 * boundary-transfer renderer becomes production-green. */
export const GRASS_GRID = 3072;
export const GRASS_CELL_METRES = 0.105;
export const GRASS_SALT = 0x51a55e & 0x7fffffff;
export const GRASS_TIER_KEEP_SALT = GRASS_SALT ^ 0x37a1;
export const GRASS_ELECTION_FLAGS = 0xc0000000;
export const GRASS_FAR_BASE = 0x28000000;
export const GRASS_NEAR_EPSILON = 1e-4;

export function grassNumberParam(key: string, fallback: number, low: number, high: number): number {
  const value = Number(new URLSearchParams(window.location.search).get(key) ?? String(fallback));
  return Number.isFinite(value) && value >= low && value <= high ? value : fallback;
}

export const GRASS_BAKE_RESOLUTION = Math.round(grassNumberParam('grassbakres', 64, 16, 256));
export const GRASS_BAKE_ANGLES = Math.round(grassNumberParam('grassbakang', 8, 4, 64));
export const GRASS_BAKE_SHIFT = grassNumberParam('grassshiftk', 0, 0, 1);
export const GRASS_BAKE_THICKNESS = grassNumberParam('grassthickk', 0, 0, 2);
export const GRASS_RAY_SWAY = grassNumberParam('grasssway', 1, 0, 5);
export const GRASS_RAY_END = grassNumberParam('grassrayend', 155, 20, 300);
export const GRASS_DENSITY_TIERS = [1, 0.55, 0.3, 0.12] as const;
export const GRASS_PROFILE_BLOCK_BASE = [0, 1, 2, 3, 4, 5] as const;
export const GRASS_RAY_ATLAS_MAX_BYTES = 64 * 1024 * 1024;

export const GUIDE_SUB = 8;
export const GUIDE_PITCH = GRASS_CELL_METRES * GUIDE_SUB;
export const GUIDE_RESOLUTION = Math.max(
  384,
  Math.ceil(((GRASS_RAY_END + 4) * 2) / GUIDE_PITCH),
);
export const GUIDE_TEXEL_COUNT = GUIDE_RESOLUTION * GUIDE_RESOLUTION;

/** Functional fallback parameters: height, rag, deformation, atlas block. */
export const COVER_PARAMETER_TABLE = [
  [0.5, 0.24, 1, GRASS_PROFILE_BLOCK_BASE[0]],
  [0.085, 0.08, 0, GRASS_PROFILE_BLOCK_BASE[1]],
  [0.68, 0.22, 0.72, GRASS_PROFILE_BLOCK_BASE[2]],
  [0.045, 0.06, 0, GRASS_PROFILE_BLOCK_BASE[3]],
  [0.22, 0.16, 0.22, GRASS_PROFILE_BLOCK_BASE[4]],
  [0.3, 0.12, 0.12, GRASS_PROFILE_BLOCK_BASE[5]],
] as const;

export interface LegacyPeriodicBladeBakeParam {
  c: number;
  s: number;
  ox: number;
  oz: number;
  hk: number;
  lean: number;
}

function bladeTable(blades: number, segments: number): LegacyPeriodicBladeBakeParam[] {
  let state = 1234567 + blades * 77 + segments * 13;
  const random = (): number => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
  return Array.from({ length: blades }, () => {
    const yaw = random() * Math.PI * 2;
    return {
      c: Math.cos(yaw),
      s: Math.sin(yaw),
      ox: (random() - 0.5) * 0.16,
      oz: (random() - 0.5) * 0.16,
      hk: 0.62 + random() * 0.65,
      lean: (random() - 0.5) * 0.42,
    };
  });
}

export const GRASS_BLADE_TABLE = bladeTable(5, 4);
