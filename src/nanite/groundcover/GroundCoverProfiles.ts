import {
  ClampToEdgeWrapping,
  DataArrayTexture,
  DataUtils,
  DataTexture,
  HalfFloatType,
  LinearFilter,
  NearestFilter,
  RedIntegerFormat,
  RGBAIntegerFormat,
  RGBAFormat,
  UnsignedByteType,
  UnsignedIntType,
} from 'three';
import { GroundCoverId } from './GroundCoverTypes';

/** Exact native-profile ids. These are deliberately independent from the
 * six functional GroundCoverId classes used for wind and broad materials. */
export const GroundCoverProfileId = {
  AgrostisCapillaris: 0,
  AvenellaFlexuosa: 1,
  CalamagrostisCanescens: 2,
  CarexCespitosa: 3,
  EriophorumVaginatum: 4,
  SphagnumCapillifolium: 5,
  PleuroziumSchreberi: 6,
  CladoniaRangiferina: 7,
  OxalisAcetosella: 8,
  MaianthemumBifolium: 9,
  VacciniumMyrtillus: 10,
  CallunaVulgaris: 11,
} as const;

/** Canonical GCAR layer order. The array loader verifies this exact closure. */
export const GROUND_COVER_PROFILE_IDS = [
  GroundCoverProfileId.AgrostisCapillaris,
  GroundCoverProfileId.AvenellaFlexuosa,
  GroundCoverProfileId.CalamagrostisCanescens,
  GroundCoverProfileId.CarexCespitosa,
  GroundCoverProfileId.EriophorumVaginatum,
  GroundCoverProfileId.SphagnumCapillifolium,
  GroundCoverProfileId.PleuroziumSchreberi,
  GroundCoverProfileId.CladoniaRangiferina,
  GroundCoverProfileId.OxalisAcetosella,
  GroundCoverProfileId.MaianthemumBifolium,
  GroundCoverProfileId.VacciniumMyrtillus,
  GroundCoverProfileId.CallunaVulgaris,
] as const;

export const GROUND_COVER_PROFILE_COUNT = GROUND_COVER_PROFILE_IDS.length;

export const GROUND_COVER_PROFILE_ARRAY_URL = new URL(
  '../../assets/groundcover/estonia-native-groundcover.gcar',
  import.meta.url,
);

/** High-resolution single-species acceptance carrier. Production keeps the
 * compact canonical array; `?grassprofile=2` substitutes this standalone layer
 * so source fidelity can be perfected without forcing every provisional
 * species to the same 256² allocation. */
export const CALAMAGROSTIS_ACCEPTANCE_PROFILE_URL = new URL(
  '../../assets/groundcover/calamagrostis-canescens.gcrp',
  import.meta.url,
);

/** Exact profile -> functional response. Array order is the stable profile id. */
export const GROUND_COVER_PROFILE_FUNCTIONAL_IDS = [
  GroundCoverId.Grass,
  GroundCoverId.Grass,
  GroundCoverId.Grass,
  GroundCoverId.Sedge,
  GroundCoverId.Sedge,
  GroundCoverId.Moss,
  GroundCoverId.Moss,
  GroundCoverId.Lichen,
  GroundCoverId.Forb,
  GroundCoverId.Forb,
  GroundCoverId.DwarfShrub,
  GroundCoverId.DwarfShrub,
] as const;

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
  ownerTexture: DataTexture | null;
  vertexTexture: DataTexture | null;
  triangleTexture: DataTexture | null;
  vertexTextureWidth: number;
  triangleTextureWidth: number;
  vertexCount: number;
  triangleCount: number;
  textureLayer: number | null;
}

export interface LoadedPeriodicProfileArray {
  texture: DataArrayTexture;
  profiles: readonly LoadedPeriodicProfile[];
  sourceSha256: readonly string[];
}

const HEADER_BYTES = 80;
const SLICE_BYTES = 64;
const V2_TEXEL_BYTES = 8;
const V3_TEXEL_BYTES = 16;
const UNIT_EPS = 2e-4;
const ANGLE_EPS = 2e-4;
const ARRAY_HEADER_BYTES = 96;
const ARRAY_DIRECTION_BYTES = 16;
const ARRAY_PROFILE_BYTES = 64;
const ARRAY_DEPTH_BYTES = 8;
const ARRAY_TEXEL_BYTES = 8;
const ARRAY_SLICE_COUNT = 64;
const ARRAY_AZIMUTH_COUNT = 16;
const ARRAY_ELEVATION_COUNT = 4;
const ARRAY_ELEVATIONS = [15, 35, 55, 75].map((degrees) => degrees * Math.PI / 180);

function expectedArrayDirection(slice: number): readonly [number, number, number] {
  const azimuth = Math.floor(slice / ARRAY_ELEVATION_COUNT) * Math.PI * 2 / ARRAY_AZIMUTH_COUNT;
  const elevation = ARRAY_ELEVATIONS[slice % ARRAY_ELEVATION_COUNT]!;
  return [
    Math.cos(elevation) * Math.cos(azimuth),
    -Math.sin(elevation),
    Math.cos(elevation) * Math.sin(azimuth),
  ];
}

function finite(view: DataView, offset: number, label: string): number {
  const value = view.getFloat32(offset, true);
  if (!Number.isFinite(value)) throw new Error(`GCRP/v2 ${label} is not finite`);
  return value;
}

function positiveInt(value: number, label: string): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`GCRP/v2 ${label} must be a positive integer`);
  }
  return value;
}

/** Strict runtime parser for the periodic top-plane GCRP/v2 contract. */
export function parsePeriodicProfile(bytes: Uint8Array): PeriodicProfileData {
  if (
    bytes.byteLength < HEADER_BYTES
    || bytes[0] !== 0x47
    || bytes[1] !== 0x43
    || bytes[2] !== 0x52
    || bytes[3] !== 0x50
  ) {
    throw new Error('ground-cover profile is not a GCRP container');
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint32(4, true);
  if (version !== 2 && version !== 3 && version !== 4) {
    throw new Error(`ground-cover runtime requires GCRP/v2, v3, or v4, got v${version}`);
  }
  const profileId = view.getUint32(8, true);
  if (profileId >= GROUND_COVER_PROFILE_COUNT) {
    throw new Error(`GCRP/v2 profile id ${profileId} is outside the native palette`);
  }
  const storedTileWidth = positiveInt(view.getUint32(12, true), 'stored tile width');
  const storedTileHeight = positiveInt(view.getUint32(16, true), 'stored tile height');
  const atlasColumns = positiveInt(view.getUint32(20, true), 'atlas columns');
  const atlasRows = positiveInt(view.getUint32(24, true), 'atlas rows');
  const sliceCount = positiveInt(view.getUint32(28, true), 'slice count');
  if (sliceCount > atlasColumns * atlasRows) throw new Error('GCRP/v2 slices do not fit the atlas');
  const texelBytes = view.getUint32(32, true);
  if (
    (version === 2 && texelBytes !== V2_TEXEL_BYTES)
    || (version === 3 && texelBytes !== V3_TEXEL_BYTES)
    || (version === 4 && texelBytes !== V2_TEXEL_BYTES)
  ) throw new Error(`GCRP/v${version} texel size is not canonical`);
  const payloadOffset = view.getUint32(36, true);
  if (view.getUint32(40, true) !== 1) throw new Error('GCRP/v2 must use periodic top-plane XZ projection');
  const interiorTileWidth = positiveInt(view.getUint32(44, true), 'interior tile width');
  const interiorTileHeight = positiveInt(view.getUint32(48, true), 'interior tile height');
  const gutter = positiveInt(view.getUint32(52, true), 'gutter');
  if (
    storedTileWidth !== interiorTileWidth + gutter * 2
    || storedTileHeight !== interiorTileHeight + gutter * 2
  ) {
    throw new Error('GCRP/v2 stored tile dimensions do not match its wrapped gutters');
  }
  const headerBytes = version === 4 ? positiveInt(view.getUint32(76, true), 'header bytes') : HEADER_BYTES;
  if (version === 4 && headerBytes !== 128) throw new Error('GCRP/v4 header size is not canonical');
  const expectedPayloadOffset = headerBytes + sliceCount * SLICE_BYTES;
  if (payloadOffset !== expectedPayloadOffset) throw new Error('GCRP/v2 payload offset is not canonical');
  const atlasWidth = storedTileWidth * atlasColumns;
  const atlasHeight = storedTileHeight * atlasRows;
  const payloadBytes = atlasWidth * atlasHeight * texelBytes;
  const ownerOffset = version === 4 ? view.getUint32(80, true) : 0;
  const vertexOffset = version === 4 ? view.getUint32(84, true) : 0;
  const triangleOffset = version === 4 ? view.getUint32(88, true) : 0;
  const vertexCount = version === 4 ? positiveInt(view.getUint32(92, true), 'vertex count') : 0;
  const triangleCount = version === 4 ? positiveInt(view.getUint32(96, true), 'triangle count') : 0;
  const expectedEnd = version === 4
    ? triangleOffset + triangleCount * 16
    : payloadOffset + payloadBytes;
  if (
    (version === 4 && (
      ownerOffset !== payloadOffset + payloadBytes
      || vertexOffset !== ownerOffset + atlasWidth * atlasHeight * 4
      || triangleOffset !== vertexOffset + vertexCount * 16
    ))
    || expectedEnd !== bytes.byteLength
  ) {
    throw new Error('GCRP/v2 payload length does not match the declared atlas');
  }
  const topH = finite(view, 56, 'top height');
  const tileOriginX = finite(view, 60, 'tile origin X');
  const tileOriginZ = finite(view, 64, 'tile origin Z');
  const tileSizeX = finite(view, 68, 'tile size X');
  const tileSizeZ = finite(view, 72, 'tile size Z');
  const supportScale = version === 3 ? finite(view, 76, 'support scale') : 0;
  const sourceBounds = version === 4
    ? [
        finite(view, 100, 'source minimum X'),
        finite(view, 104, 'source minimum Y'),
        finite(view, 108, 'source minimum Z'),
        finite(view, 112, 'source maximum X'),
        finite(view, 116, 'source maximum Y'),
        finite(view, 120, 'source maximum Z'),
      ] as const
    : undefined;
  if (!(topH > 0 && tileSizeX > 0 && tileSizeZ > 0)) {
    throw new Error('GCRP/v2 tile height and dimensions must be positive');
  }
  if (version === 3 && !(supportScale > 0)) {
    throw new Error('GCRP/v3 support scale must be positive');
  }
  const slices: PeriodicProfileSlice[] = [];
  for (let i = 0; i < sliceCount; i++) {
    const base = headerBytes + i * SLICE_BYTES;
    const x = finite(view, base, `slice ${i} direction X`);
    const y = finite(view, base + 4, `slice ${i} direction Y`);
    const z = finite(view, base + 8, `slice ${i} direction Z`);
    const length = Math.hypot(x, y, z);
    if (Math.abs(length - 1) > UNIT_EPS || !(y < -1e-4)) {
      throw new Error(`GCRP/v2 slice ${i} direction must be unit length and downward`);
    }
    const depthMin = finite(view, base + 12, `slice ${i} depth minimum`);
    const depthMax = finite(view, base + 16, `slice ${i} depth maximum`);
    if (!(depthMin >= 0 && depthMax > depthMin)) {
      throw new Error(`GCRP/v2 slice ${i} has invalid depth bounds`);
    }
    slices.push({
      direction: [x, y, z],
      depthMin,
      depthMax,
      ...(version === 4 ? {
        copyRange: [
          view.getInt32(base + 20, true),
          view.getInt32(base + 24, true),
          view.getInt32(base + 28, true),
          view.getInt32(base + 32, true),
        ] as const,
      } : {}),
    });
  }
  // Copy to an aligned native-endian array. GCRP itself is explicitly LE; all
  // WebGPU targets supported by the browser are LE, while the copy also detaches
  // texture lifetime from the fetch ArrayBuffer.
  const texelCount = atlasWidth * atlasHeight;
  const texels = new Uint16Array(texelCount * 4);
  const correspondenceTexels = version === 3 ? new Uint16Array(texelCount * 4) : undefined;
  for (let texel = 0; texel < texelCount; texel++) {
    const source = payloadOffset + texel * texelBytes;
    const target = texel * 4;
    for (let component = 0; component < 4; component++) {
      texels[target + component] = view.getUint16(source + component * 2, true);
      if (correspondenceTexels) {
        correspondenceTexels[target + component] = view.getUint16(source + 8 + component * 2, true);
      }
    }
  }
  const ownerTexels = version === 4 ? new Uint32Array(texelCount) : undefined;
  const vertexRecords = version === 4 ? new Uint32Array(vertexCount * 4) : undefined;
  const triangleRecords = version === 4 ? new Uint32Array(triangleCount * 4) : undefined;
  if (version === 4 && ownerTexels && vertexRecords && triangleRecords) {
    for (let texel = 0; texel < texelCount; texel++) {
      ownerTexels[texel] = view.getUint32(ownerOffset + texel * 4, true);
    }
    for (let word = 0; word < vertexRecords.length; word++) {
      vertexRecords[word] = view.getUint32(vertexOffset + word * 4, true);
    }
    for (let word = 0; word < triangleRecords.length; word++) {
      triangleRecords[word] = view.getUint32(triangleOffset + word * 4, true);
    }
  }
  return {
    version,
    profileId,
    storedTileWidth,
    storedTileHeight,
    interiorTileWidth,
    interiorTileHeight,
    atlasColumns,
    atlasRows,
    gutter,
    topH,
    tileOriginX,
    tileOriginZ,
    tileSizeX,
    tileSizeZ,
    slices,
    texels,
    correspondenceTexels,
    supportScale,
    ownerTexels,
    vertexRecords,
    triangleRecords,
    sourceBounds,
  };
}

/** Require one of the baker's documented regular azimuth/elevation lattices. */
export function derivePeriodicProfileLattice(
  profile: Pick<PeriodicProfileData, 'slices'>,
): PeriodicProfileLattice {
  const angle = (slice: PeriodicProfileSlice): { azimuth: number; elevation: number } => ({
    azimuth: ((Math.atan2(slice.direction[2], slice.direction[0]) / (Math.PI * 2)) + 1) % 1,
    elevation: Math.asin(-slice.direction[1]),
  });
  const first = angle(profile.slices[0]!);
  const second = angle(profile.slices[1]!);
  const order = Math.abs(second.elevation - first.elevation) > ANGLE_EPS
    ? 'azimuth-major' as const
    : 'elevation-major' as const;
  let elevationCount: number;
  let azimuthCount: number;
  if (order === 'azimuth-major') {
    elevationCount = 0;
    while (
      elevationCount < profile.slices.length
      && Math.abs(angle(profile.slices[elevationCount]!).azimuth - first.azimuth) < ANGLE_EPS
    ) elevationCount++;
    if (elevationCount < 2 || profile.slices.length % elevationCount !== 0) {
      throw new Error('GCRP/v2 direction slices are not an azimuth-major regular lattice');
    }
    azimuthCount = profile.slices.length / elevationCount;
  } else {
    azimuthCount = 0;
    while (
      azimuthCount < profile.slices.length
      && Math.abs(angle(profile.slices[azimuthCount]!).elevation - first.elevation) < ANGLE_EPS
    ) azimuthCount++;
    if (azimuthCount < 4 || profile.slices.length % azimuthCount !== 0) {
      throw new Error('GCRP/v2 direction slices are not an elevation-major regular lattice');
    }
    elevationCount = profile.slices.length / azimuthCount;
  }
  if (azimuthCount < 4 || elevationCount < 2) {
    throw new Error('GCRP/v2 direction lattice is too small');
  }
  const sliceAt = (azimuth: number, elevation: number): PeriodicProfileSlice =>
    profile.slices[
      order === 'azimuth-major'
        ? azimuth * elevationCount + elevation
        : elevation * azimuthCount + azimuth
    ]!;
  const elevations: number[] = [];
  for (let elevation = 0; elevation < elevationCount; elevation++) {
    const expectedElevation = angle(sliceAt(0, elevation)).elevation;
    if (elevation > 0 && !(expectedElevation > elevations[elevation - 1]! + ANGLE_EPS)) {
      throw new Error('GCRP/v2 elevations must be strictly increasing');
    }
    elevations.push(expectedElevation);
    for (let azimuth = 0; azimuth < azimuthCount; azimuth++) {
      const actual = angle(sliceAt(azimuth, elevation));
      const actualElevation = actual.elevation;
      if (Math.abs(actualElevation - expectedElevation) > ANGLE_EPS) {
        throw new Error('GCRP/v2 elevation row is inconsistent');
      }
      const expectedAzimuth = azimuth / azimuthCount;
      const wrappedError = Math.abs((((actual.azimuth - expectedAzimuth) + 0.5) % 1) - 0.5);
      if (wrappedError > ANGLE_EPS) throw new Error('GCRP/v2 azimuth row is not regular');
    }
  }
  return { azimuthCount, elevationCount, order, elevations };
}

export function makePeriodicProfileTexture(profile: PeriodicProfileData): DataTexture {
  // GCRP stores rgba16unorm. WebGPU exposes that format only with the optional
  // texture-formats-tier1 feature; on devices without it three correctly falls
  // UnsignedShortType back to rgba16uint, which cannot feed a filterable float
  // texture binding. Convert once at load time to universally filterable
  // rgba16float while retaining the same 8-byte/texel GPU footprint.
  const halfTexels = new Uint16Array(profile.texels.length);
  const atlasWidth = profile.storedTileWidth * profile.atlasColumns;
  const atlasHeight = profile.storedTileHeight * profile.atlasRows;
  const maxProjectedTiles = Math.max(...profile.slices.map((entry) =>
    entry.depthMax * Math.hypot(entry.direction[0], entry.direction[2]) / profile.tileSizeX));
  for (let texel = 0; texel < atlasWidth * atlasHeight; texel++) {
    const x = texel % atlasWidth;
    const y = Math.floor(texel / atlasWidth);
    const slice = Math.floor(y / profile.storedTileHeight) * profile.atlasColumns
      + Math.floor(x / profile.storedTileWidth);
    const record = texel * 4;
    const direction = profile.slices[slice]!.direction;
    const horizontal = Math.hypot(direction[0], direction[2]);
    const coverage = profile.texels[record + 3]! / 65535;
    const depth01 = profile.texels[record]! / 65535;
    const storedT = profile.slices[slice]!.depthMin
      + depth01 * (profile.slices[slice]!.depthMax - profile.slices[slice]!.depthMin);
    const projectedTiles = coverage > 0.5
      ? storedT * horizontal / profile.tileSizeX
      : maxProjectedTiles;
    // Sannikov's filterable carrier is inverse in-plane path, not a normalized
    // per-view 3D depth. All elevation slices therefore share one metric before
    // hardware filtering; runtime performs |OB|=|OA|/cos(alpha) once.
    halfTexels[record] = DataUtils.toHalfFloat(1 / (1 + projectedTiles));
    halfTexels[record + 1] = DataUtils.toHalfFloat(profile.texels[record + 1]! / 65535);
    halfTexels[record + 2] = DataUtils.toHalfFloat(profile.texels[record + 2]! / 65535);
    halfTexels[record + 3] = DataUtils.toHalfFloat(coverage);
  }
  const texture = new DataTexture(
    halfTexels,
    atlasWidth,
    atlasHeight,
    RGBAFormat,
    HalfFloatType,
  );
  texture.minFilter = LinearFilter;
  texture.magFilter = LinearFilter;
  texture.wrapS = ClampToEdgeWrapping;
  texture.wrapT = ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  texture.name = `groundCoverPeriodicProfile${profile.profileId}`;
  return texture;
}

export function makePeriodicCorrespondenceTexture(profile: PeriodicProfileData): DataTexture | null {
  if (!profile.correspondenceTexels) return null;
  const halfTexels = new Uint16Array(profile.correspondenceTexels.length);
  for (let i = 0; i < profile.correspondenceTexels.length; i++) {
    halfTexels[i] = DataUtils.toHalfFloat(profile.correspondenceTexels[i]! / 65535);
  }
  const texture = new DataTexture(
    halfTexels,
    profile.storedTileWidth * profile.atlasColumns,
    profile.storedTileHeight * profile.atlasRows,
    RGBAFormat,
    HalfFloatType,
  );
  texture.minFilter = LinearFilter;
  texture.magFilter = LinearFilter;
  texture.wrapS = ClampToEdgeWrapping;
  texture.wrapT = ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  texture.name = `groundCoverPeriodicCorrespondence${profile.profileId}`;
  return texture;
}

/** Build the filterable authored first-hit colour carrier already implied by a
 * GCRP/v4 owner atlas. This is a load-time transcode from the accepted bake,
 * not runtime geometry: the render path receives one ordinary RGBA8 texture.
 * RGB is premultiplied by binary coverage so hardware bilinear filtering cannot
 * bleed a miss colour into silhouettes. */
export function makePeriodicProfileColorTexture(
  profile: PeriodicProfileData,
): DataTexture | null {
  const owners = profile.ownerTexels;
  const vertices = profile.vertexRecords;
  const triangles = profile.triangleRecords;
  const bounds = profile.sourceBounds;
  if (!owners || !vertices || !triangles || !bounds) return null;

  const atlasWidth = profile.storedTileWidth * profile.atlasColumns;
  const atlasHeight = profile.storedTileHeight * profile.atlasRows;
  const colorTexels = new Uint8Array(atlasWidth * atlasHeight * 4);
  const spanX = Math.max(1e-9, bounds[3] - bounds[0]);
  const spanY = Math.max(1e-9, bounds[4] - bounds[1]);
  const spanZ = Math.max(1e-9, bounds[5] - bounds[2]);
  const triangleCount = triangles.length / 4;
  const unit16 = 1 / 65535;
  const mod = (value: number, modulus: number): number => ((value % modulus) + modulus) % modulus;

  for (let sliceIndex = 0; sliceIndex < profile.slices.length; sliceIndex++) {
    const slice = profile.slices[sliceIndex]!;
    const dx = slice.direction[0];
    const dy = slice.direction[1];
    const dz = slice.direction[2];
    const tileX = (sliceIndex % profile.atlasColumns) * profile.storedTileWidth;
    const tileY = Math.floor(sliceIndex / profile.atlasColumns) * profile.storedTileHeight;
    for (let localY = 0; localY < profile.storedTileHeight; localY++) {
      const sourceY = mod(localY - profile.gutter, profile.interiorTileHeight);
      const originZ = profile.tileOriginZ
        + (1 - (sourceY + 0.5) / profile.interiorTileHeight) * profile.tileSizeZ;
      const atlasY = tileY + localY;
      for (let localX = 0; localX < profile.storedTileWidth; localX++) {
        const atlasX = tileX + localX;
        const texel = atlasY * atlasWidth + atlasX;
        const owner = owners[texel]!;
        if (owner === 0xffff_ffff || profile.texels[texel * 4 + 3] === 0) continue;
        const triangleId = owner & 0x3f_ffff;
        if (triangleId >= triangleCount) {
          throw new Error(`GCRP/v4 owner triangle ${triangleId} is outside ${triangleCount}`);
        }

        const sourceX = mod(localX - profile.gutter, profile.interiorTileWidth);
        const originX = profile.tileOriginX
          + (sourceX + 0.5) / profile.interiorTileWidth * profile.tileSizeX;
        const copyX = ((owner >>> 22) & 0x1f) - 16;
        const copyZ = ((owner >>> 27) & 0x1f) - 16;
        const copyOffsetX = copyX * profile.tileSizeX;
        const copyOffsetZ = copyZ * profile.tileSizeZ;
        const triangleBase = triangleId * 4;
        const ia = triangles[triangleBase]!;
        const ib = triangles[triangleBase + 1]!;
        const ic = triangles[triangleBase + 2]!;
        const a0 = vertices[ia * 4]!;
        const a1 = vertices[ia * 4 + 1]!;
        const a2 = vertices[ia * 4 + 2]!;
        const b0 = vertices[ib * 4]!;
        const b1 = vertices[ib * 4 + 1]!;
        const b2 = vertices[ib * 4 + 2]!;
        const c0 = vertices[ic * 4]!;
        const c1 = vertices[ic * 4 + 1]!;
        const c2 = vertices[ic * 4 + 2]!;

        // Decode source vertices relative to this ray's top-plane origin. The
        // v4 owner already names the exact periodic copy and primitive.
        const ax = (a0 & 0xffff) * unit16 * spanX + bounds[0] + copyOffsetX - originX;
        const ay = (a0 >>> 16) * unit16 * spanY + bounds[1] - profile.topH;
        const az = (a1 & 0xffff) * unit16 * spanZ + bounds[2] + copyOffsetZ - originZ;
        const bx = (b0 & 0xffff) * unit16 * spanX + bounds[0] + copyOffsetX - originX;
        const by = (b0 >>> 16) * unit16 * spanY + bounds[1] - profile.topH;
        const bz = (b1 & 0xffff) * unit16 * spanZ + bounds[2] + copyOffsetZ - originZ;
        const cx = (c0 & 0xffff) * unit16 * spanX + bounds[0] + copyOffsetX - originX;
        const cy = (c0 >>> 16) * unit16 * spanY + bounds[1] - profile.topH;
        const cz = (c1 & 0xffff) * unit16 * spanZ + bounds[2] + copyOffsetZ - originZ;
        const e1x = bx - ax;
        const e1y = by - ay;
        const e1z = bz - az;
        const e2x = cx - ax;
        const e2y = cy - ay;
        const e2z = cz - az;
        const pvx = dy * e2z - dz * e2y;
        const pvy = dz * e2x - dx * e2z;
        const pvz = dx * e2y - dy * e2x;
        const determinant = e1x * pvx + e1y * pvy + e1z * pvz;
        let wa = 1 / 3;
        let wb = 1 / 3;
        let wc = 1 / 3;
        if (Math.abs(determinant) > 1e-12) {
          const tx = -ax;
          const ty = -ay;
          const tz = -az;
          const inverse = 1 / determinant;
          wb = (tx * pvx + ty * pvy + tz * pvz) * inverse;
          const qx = ty * e1z - tz * e1y;
          const qy = tz * e1x - tx * e1z;
          const qz = tx * e1y - ty * e1x;
          wc = (dx * qx + dy * qy + dz * qz) * inverse;
          wa = 1 - wb - wc;
          // Position quantization can move a silhouette ray a few microns
          // outside its owning triangle. Preserve the owner's colour while
          // forbidding extrapolated RGB.
          wa = Math.max(0, wa);
          wb = Math.max(0, wb);
          wc = Math.max(0, wc);
          const sum = wa + wb + wc;
          if (sum > 1e-12) {
            wa /= sum;
            wb /= sum;
            wc /= sum;
          } else {
            wa = wb = wc = 1 / 3;
          }
        }
        const ar = (a1 >>> 16) * unit16;
        const ag = (a2 & 0xffff) * unit16;
        const ab = (a2 >>> 16) * unit16;
        const br = (b1 >>> 16) * unit16;
        const bg = (b2 & 0xffff) * unit16;
        const bb = (b2 >>> 16) * unit16;
        const cr = (c1 >>> 16) * unit16;
        const cg = (c2 & 0xffff) * unit16;
        const cb = (c2 >>> 16) * unit16;
        const target = texel * 4;
        colorTexels[target] = Math.round(Math.min(1, wa * ar + wb * br + wc * cr) * 255);
        colorTexels[target + 1] = Math.round(Math.min(1, wa * ag + wb * bg + wc * cg) * 255);
        colorTexels[target + 2] = Math.round(Math.min(1, wa * ab + wb * bb + wc * cb) * 255);
        colorTexels[target + 3] = 255;
      }
    }
  }

  const texture = new DataTexture(colorTexels, atlasWidth, atlasHeight, RGBAFormat, UnsignedByteType);
  texture.minFilter = LinearFilter;
  texture.magFilter = LinearFilter;
  texture.wrapS = ClampToEdgeWrapping;
  texture.wrapT = ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  texture.name = `groundCoverPeriodicColor${profile.profileId}`;
  return texture;
}

function makeOwnedProfileTextures(profile: PeriodicProfileData, enabled: boolean): {
  ownerTexture: DataTexture | null;
  vertexTexture: DataTexture | null;
  triangleTexture: DataTexture | null;
  vertexTextureWidth: number;
  triangleTextureWidth: number;
  vertexCount: number;
  triangleCount: number;
} {
  if (!enabled || !profile.ownerTexels || !profile.vertexRecords || !profile.triangleRecords) {
    return {
      ownerTexture: null,
      vertexTexture: null,
      triangleTexture: null,
      vertexTextureWidth: 0,
      triangleTextureWidth: 0,
      vertexCount: 0,
      triangleCount: 0,
    };
  }
  const ownerTexture = new DataTexture(
    profile.ownerTexels,
    profile.storedTileWidth * profile.atlasColumns,
    profile.storedTileHeight * profile.atlasRows,
    RedIntegerFormat,
    UnsignedIntType,
  );
  ownerTexture.minFilter = NearestFilter;
  ownerTexture.magFilter = NearestFilter;
  ownerTexture.generateMipmaps = false;
  ownerTexture.needsUpdate = true;
  ownerTexture.name = `groundCoverPeriodicOwner${profile.profileId}`;
  const makeTable = (records: Uint32Array, label: string): {
    texture: DataTexture;
    width: number;
    count: number;
  } => {
    const count = records.length / 4;
    const width = Math.min(2048, count);
    const height = Math.ceil(count / width);
    const padded = new Uint32Array(width * height * 4);
    padded.set(records);
    const texture = new DataTexture(padded, width, height, RGBAIntegerFormat, UnsignedIntType);
    texture.minFilter = NearestFilter;
    texture.magFilter = NearestFilter;
    texture.generateMipmaps = false;
    texture.needsUpdate = true;
    texture.name = label;
    return { texture, width, count };
  };
  const vertices = makeTable(
    profile.vertexRecords,
    `groundCoverPeriodicVertices${profile.profileId}`,
  );
  const triangles = makeTable(
    profile.triangleRecords,
    `groundCoverPeriodicTriangles${profile.profileId}`,
  );
  return {
    ownerTexture,
    vertexTexture: vertices.texture,
    triangleTexture: triangles.texture,
    vertexTextureWidth: vertices.width,
    triangleTextureWidth: triangles.width,
    vertexCount: vertices.count,
    triangleCount: triangles.count,
  };
}

export interface ParsedPeriodicProfileArray {
  layerWidth: number;
  layerHeight: number;
  profiles: readonly Omit<LoadedPeriodicProfile, 'texture' | 'textureLayer'>[];
  sourceSha256: readonly string[];
  /** Layer-major rgba16float bit patterns, directly uploadable without conversion. */
  halfTexels: Uint16Array;
}

function shaHex(bytes: Uint8Array): string {
  return Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
}

/** Strict browser-side parser for the one-binding GCAR/v1 production container. */
export function parsePeriodicProfileArray(
  bytes: Uint8Array,
  expectedProfileIds?: readonly number[],
): ParsedPeriodicProfileArray {
  if (
    bytes.byteLength < ARRAY_HEADER_BYTES
    || bytes[0] !== 0x47
    || bytes[1] !== 0x43
    || bytes[2] !== 0x41
    || bytes[3] !== 0x52
  ) throw new Error('ground-cover profile array is not a GCAR container');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(4, true) !== 1) throw new Error('ground-cover runtime requires GCAR/v1');
  if (view.getUint32(8, true) !== ARRAY_HEADER_BYTES) throw new Error('GCAR/v1 header size is not canonical');
  const profileCount = positiveInt(view.getUint32(12, true), 'profile count');
  if (profileCount > GROUND_COVER_PROFILE_COUNT) throw new Error('GCAR/v1 has too many native profiles');
  const layerWidth = positiveInt(view.getUint32(16, true), 'layer width');
  const layerHeight = positiveInt(view.getUint32(20, true), 'layer height');
  const storedTileWidth = positiveInt(view.getUint32(24, true), 'stored tile width');
  const storedTileHeight = positiveInt(view.getUint32(28, true), 'stored tile height');
  const interiorTileWidth = positiveInt(view.getUint32(32, true), 'interior tile width');
  const interiorTileHeight = positiveInt(view.getUint32(36, true), 'interior tile height');
  const atlasColumns = positiveInt(view.getUint32(40, true), 'atlas columns');
  const atlasRows = positiveInt(view.getUint32(44, true), 'atlas rows');
  const sliceCount = view.getUint32(48, true);
  const azimuthCount = view.getUint32(52, true);
  const elevationCount = view.getUint32(56, true);
  const gutter = positiveInt(view.getUint32(60, true), 'gutter');
  if (
    sliceCount !== ARRAY_SLICE_COUNT
    || azimuthCount !== ARRAY_AZIMUTH_COUNT
    || elevationCount !== ARRAY_ELEVATION_COUNT
    || atlasColumns !== 8
    || atlasRows !== 8
  ) throw new Error('GCAR/v1 must use the canonical 16x4 lattice and 8x8 atlas');
  if (
    storedTileWidth !== interiorTileWidth + gutter * 2
    || storedTileHeight !== interiorTileHeight + gutter * 2
    || layerWidth !== storedTileWidth * atlasColumns
    || layerHeight !== storedTileHeight * atlasRows
  ) throw new Error('GCAR/v1 layer dimensions are inconsistent');
  if (
    view.getUint32(64, true) !== ARRAY_TEXEL_BYTES
    || view.getUint32(68, true) !== 1
    || view.getUint32(72, true) !== 1
    || view.getUint32(76, true) !== 1
  ) throw new Error('GCAR/v1 format flags are unsupported');
  const directionOffset = view.getUint32(80, true);
  const profileOffset = view.getUint32(84, true);
  const depthOffset = view.getUint32(88, true);
  const payloadOffset = view.getUint32(92, true);
  const expectedProfileOffset = ARRAY_HEADER_BYTES + sliceCount * ARRAY_DIRECTION_BYTES;
  const expectedDepthOffset = expectedProfileOffset + profileCount * ARRAY_PROFILE_BYTES;
  const expectedPayloadOffset = expectedDepthOffset + profileCount * sliceCount * ARRAY_DEPTH_BYTES;
  if (
    directionOffset !== ARRAY_HEADER_BYTES
    || profileOffset !== expectedProfileOffset
    || depthOffset !== expectedDepthOffset
    || payloadOffset !== expectedPayloadOffset
  ) throw new Error('GCAR/v1 table offsets are not canonical');
  const layerBytes = layerWidth * layerHeight * ARRAY_TEXEL_BYTES;
  if (payloadOffset + profileCount * layerBytes !== bytes.byteLength) {
    throw new Error('GCAR/v1 payload length is inconsistent');
  }
  const directions: Array<readonly [number, number, number]> = [];
  for (let slice = 0; slice < sliceCount; slice++) {
    const offset = directionOffset + slice * ARRAY_DIRECTION_BYTES;
    const x = finite(view, offset, `array direction ${slice} X`);
    const y = finite(view, offset + 4, `array direction ${slice} Y`);
    const z = finite(view, offset + 8, `array direction ${slice} Z`);
    const expected = expectedArrayDirection(slice);
    if (
      Math.abs(Math.hypot(x, y, z) - 1) > UNIT_EPS
      || !(y < -1e-4)
      || Math.abs(x - expected[0]) > ANGLE_EPS
      || Math.abs(y - expected[1]) > ANGLE_EPS
      || Math.abs(z - expected[2]) > ANGLE_EPS
    ) {
      throw new Error(`GCAR/v1 direction ${slice} is outside the canonical lattice`);
    }
    directions.push([x, y, z]);
  }
  const profiles: Array<Omit<LoadedPeriodicProfile, 'texture' | 'textureLayer'>> = [];
  const sourceSha256: string[] = [];
  const seen = new Set<number>();
  for (let layer = 0; layer < profileCount; layer++) {
    const offset = profileOffset + layer * ARRAY_PROFILE_BYTES;
    const profileId = view.getUint32(offset, true);
    if (profileId >= GROUND_COVER_PROFILE_COUNT || seen.has(profileId)) {
      throw new Error(`GCAR/v1 profile id ${profileId} is invalid or duplicated`);
    }
    seen.add(profileId);
    if (view.getUint32(offset + 4, true) !== layer) throw new Error('GCAR/v1 layer index is not canonical');
    const topH = finite(view, offset + 8, `profile ${profileId} top height`);
    const tileOriginX = finite(view, offset + 12, `profile ${profileId} tile origin X`);
    const tileOriginZ = finite(view, offset + 16, `profile ${profileId} tile origin Z`);
    const tileSizeX = finite(view, offset + 20, `profile ${profileId} tile size X`);
    const tileSizeZ = finite(view, offset + 24, `profile ${profileId} tile size Z`);
    if (!(topH > 0 && tileSizeX > 0 && tileSizeZ > 0)) {
      throw new Error(`GCAR/v1 profile ${profileId} dimensions must be positive`);
    }
    if (view.getUint32(offset + 28, true) !== 0) throw new Error('GCAR/v1 profile flags are unsupported');
    const slices: PeriodicProfileSlice[] = [];
    for (let slice = 0; slice < sliceCount; slice++) {
      const boundsOffset = depthOffset + (layer * sliceCount + slice) * ARRAY_DEPTH_BYTES;
      const depthMin = finite(view, boundsOffset, `profile ${profileId} slice ${slice} depth minimum`);
      const depthMax = finite(view, boundsOffset + 4, `profile ${profileId} slice ${slice} depth maximum`);
      if (!(depthMin >= 0 && depthMax > depthMin)) {
        throw new Error(`GCAR/v1 profile ${profileId} slice ${slice} has invalid depth bounds`);
      }
      slices.push({ direction: directions[slice]!, depthMin, depthMax });
    }
    const lattice = derivePeriodicProfileLattice({ slices });
    profiles.push({
      version: 2,
      profileId,
      storedTileWidth,
      storedTileHeight,
      interiorTileWidth,
      interiorTileHeight,
      atlasColumns,
      atlasRows,
      gutter,
      topH,
      tileOriginX,
      tileOriginZ,
      tileSizeX,
      tileSizeZ,
      slices,
      lattice,
      supportScale: 0,
      correspondenceTexture: null,
      colorTexture: null,
      ownerTexture: null,
      vertexTexture: null,
      triangleTexture: null,
      vertexTextureWidth: 0,
      triangleTextureWidth: 0,
      vertexCount: 0,
      triangleCount: 0,
    });
    sourceSha256.push(shaHex(bytes.subarray(offset + 32, offset + 64)));
  }
  if (expectedProfileIds) {
    const actual = profiles.map((profile) => profile.profileId);
    if (
      actual.length !== expectedProfileIds.length
      || expectedProfileIds.some((id, index) => actual[index] !== id)
    ) throw new Error(`GCAR/v1 profile ids [${actual.join(',')}] do not match expected [${expectedProfileIds.join(',')}]`);
  }
  const payloadByteOffset = bytes.byteOffset + payloadOffset;
  if ((payloadByteOffset & 1) !== 0) throw new Error('GCAR/v1 payload is not uint16-aligned');
  const halfTexels = new Uint16Array(bytes.buffer, payloadByteOffset, profileCount * layerBytes / 2);
  return { layerWidth, layerHeight, profiles, sourceSha256, halfTexels };
}

export function makePeriodicProfileArrayTexture(profileArray: ParsedPeriodicProfileArray): DataArrayTexture {
  const texture = new DataArrayTexture(
    profileArray.halfTexels,
    profileArray.layerWidth,
    profileArray.layerHeight,
    profileArray.profiles.length,
  );
  texture.format = RGBAFormat;
  texture.type = HalfFloatType;
  texture.minFilter = LinearFilter;
  texture.magFilter = LinearFilter;
  texture.wrapS = ClampToEdgeWrapping;
  texture.wrapT = ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  texture.name = 'groundCoverPeriodicProfileArray';
  return texture;
}

export async function loadPeriodicProfileArray(
  url: URL | string,
  expectedProfileIds: readonly number[],
): Promise<LoadedPeriodicProfileArray> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`ground-cover profile-array fetch failed (${response.status} ${response.statusText})`);
  const parsed = parsePeriodicProfileArray(new Uint8Array(await response.arrayBuffer()), expectedProfileIds);
  const texture = makePeriodicProfileArrayTexture(parsed);
  return {
    texture,
    profiles: parsed.profiles.map((profile, textureLayer) => ({ ...profile, texture, textureLayer })),
    sourceSha256: parsed.sourceSha256,
  };
}

export async function loadPeriodicProfile(
  url: URL | string,
  expectedProfileId: number,
  options: { authoredColor?: boolean; ownedTables?: boolean } = {},
): Promise<LoadedPeriodicProfile> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`ground-cover profile fetch failed (${response.status} ${response.statusText})`);
  const profile = parsePeriodicProfile(new Uint8Array(await response.arrayBuffer()));
  if (profile.profileId !== expectedProfileId) {
    throw new Error(`ground-cover profile id ${profile.profileId} != expected ${expectedProfileId}`);
  }
  const texture = makePeriodicProfileTexture(profile);
  const correspondenceTexture = makePeriodicCorrespondenceTexture(profile);
  const colorTexture = options.authoredColor ? makePeriodicProfileColorTexture(profile) : null;
  const owned = makeOwnedProfileTextures(profile, options.ownedTables ?? false);
  const {
    texels: _texels,
    correspondenceTexels: _correspondenceTexels,
    ownerTexels: _ownerTexels,
    vertexRecords: _vertexRecords,
    triangleRecords: _triangleRecords,
    ...metadata
  } = profile;
  return {
    ...metadata,
    lattice: derivePeriodicProfileLattice(profile),
    texture,
    correspondenceTexture,
    colorTexture,
    ...owned,
    textureLayer: null,
  };
}
