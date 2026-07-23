import type { DataArrayTexture, DataTexture } from 'three';

export interface PeriodicProfileSlice {
  direction: readonly [number, number, number];
  depthMin: number;
  depthMax: number;
  copyRange?: readonly [number, number, number, number];
}

export interface PeriodicProfileData {
  version: 2 | 3 | 4;
  profileId: number;
  storedTileWidth: number;
  storedTileHeight: number;
  interiorTileWidth: number;
  interiorTileHeight: number;
  atlasColumns: number;
  atlasRows: number;
  gutter: number;
  topH: number;
  tileOriginX: number;
  tileOriginZ: number;
  tileSizeX: number;
  tileSizeZ: number;
  slices: readonly PeriodicProfileSlice[];
  /** Atlas-order little-endian rgba16unorm payload. */
  texels: Uint16Array;
  /** v3-only geometric-face/support/authored-colour rgba16unorm payload. */
  correspondenceTexels?: Uint16Array;
  /** Metres represented by correspondence B==1; zero for v2. */
  supportScale: number;
  ownerTexels?: Uint32Array;
  vertexRecords?: Uint32Array;
  triangleRecords?: Uint32Array;
  sourceBounds?: readonly [number, number, number, number, number, number];
}

export interface PeriodicProfileLattice {
  azimuthCount: number;
  elevationCount: number;
  order: 'azimuth-major' | 'elevation-major';
  /** Positive downward elevations in radians, low to high. */
  elevations: readonly number[];
}

export interface LoadedPeriodicProfile extends Omit<
  PeriodicProfileData,
  'texels' | 'correspondenceTexels' | 'ownerTexels' | 'vertexRecords' | 'triangleRecords'
> {
  lattice: PeriodicProfileLattice;
  /** All production profiles share one array texture; null is the legacy single-profile loader. */
  texture: DataTexture | DataArrayTexture;
  correspondenceTexture: DataTexture | null;
  /** Premultiplied authored first-hit RGB with coverage in A. Standalone
   * acceptance profiles only; production arrays can add the same carrier
   * without changing the ray contract. */
  colorTexture: DataTexture | null;
  textureLayer: number | null;
}

export interface LoadedPeriodicProfileArray {
  texture: DataArrayTexture;
  profiles: readonly LoadedPeriodicProfile[];
  sourceSha256: readonly string[];
}
