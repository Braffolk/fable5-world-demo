/** Pure packed-height geomorph policy shared by CPU sampling and lean tests. */

export interface MorphBand {
  innerM: number;
  outerM: number;
  availabilityM: number;
}

export const MICRO_MORPH_BANDS: Readonly<Record<-2 | -1, MorphBand>> = {
  [-2]: { innerM: 32, outerM: 40, availabilityM: 2 },
  [-1]: { innerM: 128, outerM: 160, availabilityM: 8 },
};

export function smootherStep01(value: number): number {
  const t = Math.min(1, Math.max(0, value));
  return t * t * t * (t * (t * 6 - 15) + 10);
}

export function cameraMorphWeight(lod: -2 | -1, x: number, z: number, centerX: number, centerZ: number): number {
  const band = MICRO_MORPH_BANDS[lod];
  const radius = Math.max(Math.abs(x - centerX), Math.abs(z - centerZ));
  return 1 - smootherStep01((radius - band.innerM) / (band.outerM - band.innerM));
}

export function availabilityMorphWeight(
  lod: -2 | -1,
  x: number,
  z: number,
  originX: number,
  originZ: number,
  texel: number,
  res: number,
  coverageMinX = Number.NEGATIVE_INFINITY,
  coverageMinZ = Number.NEGATIVE_INFINITY,
  coverageMaxX = Number.POSITIVE_INFINITY,
  coverageMaxZ = Number.POSITIVE_INFINITY,
): number {
  const gx = (x - originX) / texel;
  const gz = (z - originZ) / texel;
  const windowEdgeM = Math.min(gx - 1, gz - 1, res - 2 - gx, res - 2 - gz) * texel;
  const coverageEdgeM = Math.min(
    x - coverageMinX - texel,
    z - coverageMinZ - texel,
    coverageMaxX - texel - x,
    coverageMaxZ - texel - z,
  );
  const edgeM = Math.min(windowEdgeM, coverageEdgeM);
  return smootherStep01(edgeM / MICRO_MORPH_BANDS[lod].availabilityM);
}
