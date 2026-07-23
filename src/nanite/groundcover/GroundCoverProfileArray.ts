import {
  ClampToEdgeWrapping,
  DataArrayTexture,
  HalfFloatType,
  LinearFilter,
  RGBAFormat,
} from 'three';
import { GROUND_COVER_PROFILE_COUNT } from './GroundCoverProfileIds';
import type {
  LoadedPeriodicProfile,
  LoadedPeriodicProfileArray,
  PeriodicProfileSlice,
} from './GroundCoverProfileTypes';

export const GROUND_COVER_PROFILE_ARRAY_URL = new URL(
  '../../assets/groundcover/estonia-native-groundcover.gcar',
  import.meta.url,
);

const ARRAY_HEADER_BYTES = 96;
const ARRAY_DIRECTION_BYTES = 16;
const ARRAY_PROFILE_BYTES = 64;
const ARRAY_DEPTH_BYTES = 8;
const ARRAY_TEXEL_BYTES = 8;
const ARRAY_SLICE_COUNT = 64;
const ARRAY_AZIMUTH_COUNT = 16;
const ARRAY_ELEVATION_COUNT = 4;
const ARRAY_ELEVATIONS = [15, 35, 55, 75].map((degrees) => degrees * Math.PI / 180);
const UNIT_EPS = 2e-4;
const ANGLE_EPS = 2e-4;

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
    const lattice = {
      azimuthCount: ARRAY_AZIMUTH_COUNT,
      elevationCount: ARRAY_ELEVATION_COUNT,
      order: 'azimuth-major' as const,
      elevations: ARRAY_ELEVATIONS,
    };
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
