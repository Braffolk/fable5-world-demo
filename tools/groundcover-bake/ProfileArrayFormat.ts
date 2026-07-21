import { createHash } from 'node:crypto';
import { DataUtils } from 'three';

/**
 * GCAR/v1 is the production, multi-profile companion to the single-profile
 * GCRP/v2 bake artifact. Every layer uses the same 16x4 direction lattice and
 * guarded 8x8 slice atlas, so all profile payloads can be uploaded as one
 * filterable texture_2d_array. Profile selection is the integer array layer;
 * filtering therefore remains strictly inside one botanical profile.
 */
export const PROFILE_ARRAY_MAGIC = 'GCAR';
export const PROFILE_ARRAY_VERSION = 1;
export const PROFILE_ARRAY_HEADER_BYTES = 96;
export const PROFILE_ARRAY_DIRECTION_BYTES = 16;
export const PROFILE_ARRAY_PROFILE_BYTES = 64;
export const PROFILE_ARRAY_DEPTH_BYTES = 8;
export const PROFILE_ARRAY_TEXEL_BYTES = 8;
export const PROFILE_ARRAY_TEXEL_FORMAT_RGBA16FLOAT = 1;
export const PROFILE_ARRAY_PROJECTION_PERIODIC_TOP_XZ = 1;
export const PROFILE_ARRAY_LATTICE_AZIMUTH_MAJOR = 1;
export const PROFILE_ARRAY_AZIMUTH_COUNT = 16;
export const PROFILE_ARRAY_ELEVATIONS_DEG = [15, 35, 55, 75] as const;
export const PROFILE_ARRAY_ELEVATION_COUNT = PROFILE_ARRAY_ELEVATIONS_DEG.length;
export const PROFILE_ARRAY_SLICE_COUNT =
  PROFILE_ARRAY_AZIMUTH_COUNT * PROFILE_ARRAY_ELEVATION_COUNT;
export const PROFILE_ARRAY_ATLAS_COLUMNS = 8;
export const PROFILE_ARRAY_ATLAS_ROWS = 8;

const GCRP_HEADER_BYTES = 80;
const GCRP_SLICE_BYTES = 64;
const GCRP_VERSION = 2;
const GCRP_TEXEL_BYTES = 8;
const DIRECTION_EPS = 2e-4;

export interface ProfileArrayDirection {
  x: number;
  y: number;
  z: number;
}

export interface ProfileArrayProfile {
  profileId: number;
  layerIndex: number;
  topH: number;
  tileOriginX: number;
  tileOriginZ: number;
  tileSizeX: number;
  tileSizeZ: number;
  sourceSha256: string;
}

export interface ProfileArrayHeader {
  version: 1;
  profileCount: number;
  layerWidth: number;
  layerHeight: number;
  storedTileWidth: number;
  storedTileHeight: number;
  interiorTileWidth: number;
  interiorTileHeight: number;
  atlasColumns: number;
  atlasRows: number;
  sliceCount: number;
  azimuthCount: number;
  elevationCount: number;
  gutter: number;
  texelBytes: number;
  texelFormat: number;
  projectionMode: number;
  latticeOrder: number;
  directionOffset: number;
  profileOffset: number;
  depthOffset: number;
  payloadOffset: number;
}

export interface ParsedPeriodicProfileArray {
  header: ProfileArrayHeader;
  directions: readonly ProfileArrayDirection[];
  profiles: readonly ProfileArrayProfile[];
  /** Profile-major, then azimuth-major/elevation-minor [min,max] pairs. */
  depthBounds: Float32Array;
  /** Layer-major rgba16float bit patterns, ready for DataArrayTexture upload. */
  halfTexels: Uint16Array;
}

export interface PackedPeriodicProfileArray extends ParsedPeriodicProfileArray {
  bytes: Uint8Array;
  sha256: string;
}

export interface PackPeriodicProfileArrayOptions {
  /** Fail closed unless the sorted input profile ids exactly match this list. */
  expectedProfileIds?: readonly number[];
}

interface SourceProfile {
  bytes: Uint8Array;
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
  directions: ProfileArrayDirection[];
  depthBounds: Float32Array;
  payloadOffset: number;
  sourceSha256: string;
}

function finite(view: DataView, offset: number, label: string): number {
  const value = view.getFloat32(offset, true);
  if (!Number.isFinite(value)) throw new Error(`${label} is not finite`);
  return value;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${label} must be a positive integer`);
  return value;
}

function hash(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function hexBytes(hex: string): Uint8Array {
  if (!/^[0-9a-f]{64}$/.test(hex)) throw new Error('SHA-256 must be 64 lowercase hexadecimal characters');
  const bytes = new Uint8Array(32);
  for (let i = 0; i < bytes.length; i++) bytes[i] = Number.parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  return bytes;
}

function bytesHex(bytes: Uint8Array): string {
  return Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
}

function expectedDirection(sliceIndex: number): ProfileArrayDirection {
  const azimuthIndex = Math.floor(sliceIndex / PROFILE_ARRAY_ELEVATION_COUNT);
  const elevationIndex = sliceIndex % PROFILE_ARRAY_ELEVATION_COUNT;
  const azimuth = azimuthIndex * Math.PI * 2 / PROFILE_ARRAY_AZIMUTH_COUNT;
  const elevation = PROFILE_ARRAY_ELEVATIONS_DEG[elevationIndex]! * Math.PI / 180;
  return {
    x: Math.cos(elevation) * Math.cos(azimuth),
    y: -Math.sin(elevation),
    z: Math.cos(elevation) * Math.sin(azimuth),
  };
}

function parseGcrpV2(bytes: Uint8Array): SourceProfile {
  if (
    bytes.byteLength < GCRP_HEADER_BYTES
    || bytes[0] !== 0x47
    || bytes[1] !== 0x43
    || bytes[2] !== 0x52
    || bytes[3] !== 0x50
  ) throw new Error('profile-array source is not a GCRP container');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(4, true) !== GCRP_VERSION) throw new Error('profile-array source must be GCRP/v2');
  const profileId = view.getUint32(8, true);
  const storedTileWidth = positiveInteger(view.getUint32(12, true), 'stored tile width');
  const storedTileHeight = positiveInteger(view.getUint32(16, true), 'stored tile height');
  const atlasColumns = positiveInteger(view.getUint32(20, true), 'atlas columns');
  const atlasRows = positiveInteger(view.getUint32(24, true), 'atlas rows');
  const sliceCount = positiveInteger(view.getUint32(28, true), 'slice count');
  if (sliceCount !== PROFILE_ARRAY_SLICE_COUNT) {
    throw new Error(`profile-array source must have ${PROFILE_ARRAY_SLICE_COUNT} slices`);
  }
  if (
    atlasColumns !== PROFILE_ARRAY_ATLAS_COLUMNS
    || atlasRows !== PROFILE_ARRAY_ATLAS_ROWS
  ) throw new Error('profile-array source must use the canonical 8x8 slice atlas');
  if (view.getUint32(32, true) !== GCRP_TEXEL_BYTES) throw new Error('profile-array source must use rgba16unorm');
  const payloadOffset = view.getUint32(36, true);
  if (view.getUint32(40, true) !== PROFILE_ARRAY_PROJECTION_PERIODIC_TOP_XZ) {
    throw new Error('profile-array source must use periodic top-plane XZ projection');
  }
  const interiorTileWidth = positiveInteger(view.getUint32(44, true), 'interior tile width');
  const interiorTileHeight = positiveInteger(view.getUint32(48, true), 'interior tile height');
  const gutter = positiveInteger(view.getUint32(52, true), 'gutter');
  if (
    storedTileWidth !== interiorTileWidth + gutter * 2
    || storedTileHeight !== interiorTileHeight + gutter * 2
  ) throw new Error('profile-array source has inconsistent guarded tile dimensions');
  const canonicalPayloadOffset = GCRP_HEADER_BYTES + sliceCount * GCRP_SLICE_BYTES;
  if (payloadOffset !== canonicalPayloadOffset) throw new Error('profile-array source payload offset is not canonical');
  const layerWidth = storedTileWidth * atlasColumns;
  const layerHeight = storedTileHeight * atlasRows;
  const payloadBytes = layerWidth * layerHeight * GCRP_TEXEL_BYTES;
  if (payloadOffset + payloadBytes !== bytes.byteLength) {
    throw new Error('profile-array source payload length is inconsistent');
  }
  const topH = finite(view, 56, 'topH');
  const tileOriginX = finite(view, 60, 'tile origin X');
  const tileOriginZ = finite(view, 64, 'tile origin Z');
  const tileSizeX = finite(view, 68, 'tile size X');
  const tileSizeZ = finite(view, 72, 'tile size Z');
  if (!(topH > 0 && tileSizeX > 0 && tileSizeZ > 0)) {
    throw new Error('profile-array source tile dimensions must be positive');
  }
  const directions: ProfileArrayDirection[] = [];
  const depthBounds = new Float32Array(sliceCount * 2);
  for (let slice = 0; slice < sliceCount; slice++) {
    const offset = GCRP_HEADER_BYTES + slice * GCRP_SLICE_BYTES;
    const direction = {
      x: finite(view, offset, `slice ${slice} direction X`),
      y: finite(view, offset + 4, `slice ${slice} direction Y`),
      z: finite(view, offset + 8, `slice ${slice} direction Z`),
    };
    const expected = expectedDirection(slice);
    if (
      Math.abs(direction.x - expected.x) > DIRECTION_EPS
      || Math.abs(direction.y - expected.y) > DIRECTION_EPS
      || Math.abs(direction.z - expected.z) > DIRECTION_EPS
    ) throw new Error(`profile-array source slice ${slice} is outside the canonical 16x4 lattice`);
    const depthMin = finite(view, offset + 12, `slice ${slice} depth minimum`);
    const depthMax = finite(view, offset + 16, `slice ${slice} depth maximum`);
    if (!(depthMin >= 0 && depthMax > depthMin)) {
      throw new Error(`profile-array source slice ${slice} has invalid depth bounds`);
    }
    directions.push(direction);
    depthBounds[slice * 2] = depthMin;
    depthBounds[slice * 2 + 1] = depthMax;
  }
  return {
    bytes,
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
    directions,
    depthBounds,
    payloadOffset,
    sourceSha256: hash(bytes),
  };
}

function sameNumber(left: number, right: number): boolean {
  return Object.is(Math.fround(left), Math.fround(right));
}

/**
 * Deterministically packs independently validated GCRP/v2 artifacts into one
 * layer-major rgba16float GCAR/v1 payload. Inputs are sorted by native profile
 * id; source order therefore cannot change the artifact hash or runtime layer.
 */
export function packPeriodicProfileArray(
  inputs: readonly Uint8Array[],
  options: PackPeriodicProfileArrayOptions = {},
): PackedPeriodicProfileArray {
  if (inputs.length === 0) throw new Error('profile array requires at least one GCRP/v2 source');
  const sources = inputs.map(parseGcrpV2).sort((left, right) => left.profileId - right.profileId);
  for (let i = 1; i < sources.length; i++) {
    if (sources[i - 1]!.profileId === sources[i]!.profileId) {
      throw new Error(`duplicate profile id ${sources[i]!.profileId}`);
    }
  }
  if (options.expectedProfileIds) {
    const expected = [...options.expectedProfileIds].sort((left, right) => left - right);
    const actual = sources.map((source) => source.profileId);
    if (expected.length !== actual.length || expected.some((id, index) => id !== actual[index])) {
      throw new Error(`profile id set [${actual.join(',')}] does not match expected [${expected.join(',')}]`);
    }
  }
  const first = sources[0]!;
  for (const source of sources.slice(1)) {
    if (
      source.storedTileWidth !== first.storedTileWidth
      || source.storedTileHeight !== first.storedTileHeight
      || source.interiorTileWidth !== first.interiorTileWidth
      || source.interiorTileHeight !== first.interiorTileHeight
      || source.atlasColumns !== first.atlasColumns
      || source.atlasRows !== first.atlasRows
      || source.gutter !== first.gutter
    ) throw new Error(`profile ${source.profileId} is not texture-array compatible`);
    for (let slice = 0; slice < PROFILE_ARRAY_SLICE_COUNT; slice++) {
      const left = first.directions[slice]!;
      const right = source.directions[slice]!;
      if (
        !sameNumber(left.x, right.x)
        || !sameNumber(left.y, right.y)
        || !sameNumber(left.z, right.z)
      ) throw new Error(`profile ${source.profileId} slice ${slice} direction differs from layer 0`);
    }
  }

  const directionOffset = PROFILE_ARRAY_HEADER_BYTES;
  const profileOffset = directionOffset + PROFILE_ARRAY_SLICE_COUNT * PROFILE_ARRAY_DIRECTION_BYTES;
  const depthOffset = profileOffset + sources.length * PROFILE_ARRAY_PROFILE_BYTES;
  const payloadOffset = depthOffset
    + sources.length * PROFILE_ARRAY_SLICE_COUNT * PROFILE_ARRAY_DEPTH_BYTES;
  const layerWidth = first.storedTileWidth * first.atlasColumns;
  const layerHeight = first.storedTileHeight * first.atlasRows;
  const layerBytes = layerWidth * layerHeight * PROFILE_ARRAY_TEXEL_BYTES;
  const bytes = new Uint8Array(payloadOffset + sources.length * layerBytes);
  const view = new DataView(bytes.buffer);
  bytes.set(Array.from(PROFILE_ARRAY_MAGIC, (character) => character.charCodeAt(0)), 0);
  view.setUint32(4, PROFILE_ARRAY_VERSION, true);
  view.setUint32(8, PROFILE_ARRAY_HEADER_BYTES, true);
  view.setUint32(12, sources.length, true);
  view.setUint32(16, layerWidth, true);
  view.setUint32(20, layerHeight, true);
  view.setUint32(24, first.storedTileWidth, true);
  view.setUint32(28, first.storedTileHeight, true);
  view.setUint32(32, first.interiorTileWidth, true);
  view.setUint32(36, first.interiorTileHeight, true);
  view.setUint32(40, first.atlasColumns, true);
  view.setUint32(44, first.atlasRows, true);
  view.setUint32(48, PROFILE_ARRAY_SLICE_COUNT, true);
  view.setUint32(52, PROFILE_ARRAY_AZIMUTH_COUNT, true);
  view.setUint32(56, PROFILE_ARRAY_ELEVATION_COUNT, true);
  view.setUint32(60, first.gutter, true);
  view.setUint32(64, PROFILE_ARRAY_TEXEL_BYTES, true);
  view.setUint32(68, PROFILE_ARRAY_TEXEL_FORMAT_RGBA16FLOAT, true);
  view.setUint32(72, PROFILE_ARRAY_PROJECTION_PERIODIC_TOP_XZ, true);
  view.setUint32(76, PROFILE_ARRAY_LATTICE_AZIMUTH_MAJOR, true);
  view.setUint32(80, directionOffset, true);
  view.setUint32(84, profileOffset, true);
  view.setUint32(88, depthOffset, true);
  view.setUint32(92, payloadOffset, true);

  first.directions.forEach((direction, slice) => {
    const offset = directionOffset + slice * PROFILE_ARRAY_DIRECTION_BYTES;
    view.setFloat32(offset, direction.x, true);
    view.setFloat32(offset + 4, direction.y, true);
    view.setFloat32(offset + 8, direction.z, true);
  });
  sources.forEach((source, layerIndex) => {
    const offset = profileOffset + layerIndex * PROFILE_ARRAY_PROFILE_BYTES;
    view.setUint32(offset, source.profileId, true);
    view.setUint32(offset + 4, layerIndex, true);
    view.setFloat32(offset + 8, source.topH, true);
    view.setFloat32(offset + 12, source.tileOriginX, true);
    view.setFloat32(offset + 16, source.tileOriginZ, true);
    view.setFloat32(offset + 20, source.tileSizeX, true);
    view.setFloat32(offset + 24, source.tileSizeZ, true);
    view.setUint32(offset + 28, 0, true);
    bytes.set(hexBytes(source.sourceSha256), offset + 32);
    for (let bound = 0; bound < source.depthBounds.length; bound++) {
      view.setFloat32(
        depthOffset + (layerIndex * source.depthBounds.length + bound) * 4,
        source.depthBounds[bound]!,
        true,
      );
    }
    const sourceView = new DataView(source.bytes.buffer, source.bytes.byteOffset, source.bytes.byteLength);
    const texelValues = layerBytes / 2;
    const layerOffset = payloadOffset + layerIndex * layerBytes;
    for (let value = 0; value < texelValues; value++) {
      const unorm = sourceView.getUint16(source.payloadOffset + value * 2, true) / 65535;
      view.setUint16(layerOffset + value * 2, DataUtils.toHalfFloat(unorm), true);
    }
  });

  const parsed = parsePeriodicProfileArray(bytes);
  return { ...parsed, bytes, sha256: hash(bytes) };
}

/** Strict, canonical GCAR/v1 parser. */
export function parsePeriodicProfileArray(bytes: Uint8Array): ParsedPeriodicProfileArray {
  if (
    bytes.byteLength < PROFILE_ARRAY_HEADER_BYTES
    || String.fromCharCode(...bytes.subarray(0, 4)) !== PROFILE_ARRAY_MAGIC
  ) throw new Error('not a GCAR profile array');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint32(4, true);
  if (version !== PROFILE_ARRAY_VERSION) throw new Error(`expected GCAR/v1, got v${version}`);
  if (view.getUint32(8, true) !== PROFILE_ARRAY_HEADER_BYTES) throw new Error('GCAR/v1 header size is not canonical');
  const profileCount = positiveInteger(view.getUint32(12, true), 'profile count');
  const layerWidth = positiveInteger(view.getUint32(16, true), 'layer width');
  const layerHeight = positiveInteger(view.getUint32(20, true), 'layer height');
  const storedTileWidth = positiveInteger(view.getUint32(24, true), 'stored tile width');
  const storedTileHeight = positiveInteger(view.getUint32(28, true), 'stored tile height');
  const interiorTileWidth = positiveInteger(view.getUint32(32, true), 'interior tile width');
  const interiorTileHeight = positiveInteger(view.getUint32(36, true), 'interior tile height');
  const atlasColumns = positiveInteger(view.getUint32(40, true), 'atlas columns');
  const atlasRows = positiveInteger(view.getUint32(44, true), 'atlas rows');
  const sliceCount = view.getUint32(48, true);
  const azimuthCount = view.getUint32(52, true);
  const elevationCount = view.getUint32(56, true);
  const gutter = positiveInteger(view.getUint32(60, true), 'gutter');
  const texelBytes = view.getUint32(64, true);
  const texelFormat = view.getUint32(68, true);
  const projectionMode = view.getUint32(72, true);
  const latticeOrder = view.getUint32(76, true);
  const directionOffset = view.getUint32(80, true);
  const profileOffset = view.getUint32(84, true);
  const depthOffset = view.getUint32(88, true);
  const payloadOffset = view.getUint32(92, true);
  if (
    sliceCount !== PROFILE_ARRAY_SLICE_COUNT
    || azimuthCount !== PROFILE_ARRAY_AZIMUTH_COUNT
    || elevationCount !== PROFILE_ARRAY_ELEVATION_COUNT
    || atlasColumns !== PROFILE_ARRAY_ATLAS_COLUMNS
    || atlasRows !== PROFILE_ARRAY_ATLAS_ROWS
  ) throw new Error('GCAR/v1 does not use the canonical 16x4 lattice and 8x8 atlas');
  if (
    storedTileWidth !== interiorTileWidth + gutter * 2
    || storedTileHeight !== interiorTileHeight + gutter * 2
    || layerWidth !== storedTileWidth * atlasColumns
    || layerHeight !== storedTileHeight * atlasRows
  ) throw new Error('GCAR/v1 layer dimensions are inconsistent');
  if (
    texelBytes !== PROFILE_ARRAY_TEXEL_BYTES
    || texelFormat !== PROFILE_ARRAY_TEXEL_FORMAT_RGBA16FLOAT
    || projectionMode !== PROFILE_ARRAY_PROJECTION_PERIODIC_TOP_XZ
    || latticeOrder !== PROFILE_ARRAY_LATTICE_AZIMUTH_MAJOR
  ) throw new Error('GCAR/v1 format flags are unsupported');
  const expectedProfileOffset = PROFILE_ARRAY_HEADER_BYTES
    + sliceCount * PROFILE_ARRAY_DIRECTION_BYTES;
  const expectedDepthOffset = expectedProfileOffset + profileCount * PROFILE_ARRAY_PROFILE_BYTES;
  const expectedPayloadOffset = expectedDepthOffset
    + profileCount * sliceCount * PROFILE_ARRAY_DEPTH_BYTES;
  if (
    directionOffset !== PROFILE_ARRAY_HEADER_BYTES
    || profileOffset !== expectedProfileOffset
    || depthOffset !== expectedDepthOffset
    || payloadOffset !== expectedPayloadOffset
  ) throw new Error('GCAR/v1 table offsets are not canonical');
  const layerBytes = layerWidth * layerHeight * texelBytes;
  if (payloadOffset + profileCount * layerBytes !== bytes.byteLength) {
    throw new Error('GCAR/v1 payload length is inconsistent');
  }

  const directions: ProfileArrayDirection[] = [];
  for (let slice = 0; slice < sliceCount; slice++) {
    const offset = directionOffset + slice * PROFILE_ARRAY_DIRECTION_BYTES;
    const direction = {
      x: finite(view, offset, `direction ${slice} X`),
      y: finite(view, offset + 4, `direction ${slice} Y`),
      z: finite(view, offset + 8, `direction ${slice} Z`),
    };
    const expected = expectedDirection(slice);
    if (
      Math.abs(direction.x - expected.x) > DIRECTION_EPS
      || Math.abs(direction.y - expected.y) > DIRECTION_EPS
      || Math.abs(direction.z - expected.z) > DIRECTION_EPS
    ) throw new Error(`GCAR/v1 direction ${slice} is outside the canonical lattice`);
    directions.push(direction);
  }
  const profiles: ProfileArrayProfile[] = [];
  const seenIds = new Set<number>();
  for (let layerIndex = 0; layerIndex < profileCount; layerIndex++) {
    const offset = profileOffset + layerIndex * PROFILE_ARRAY_PROFILE_BYTES;
    const profileId = view.getUint32(offset, true);
    if (seenIds.has(profileId)) throw new Error(`GCAR/v1 duplicates profile id ${profileId}`);
    seenIds.add(profileId);
    if (view.getUint32(offset + 4, true) !== layerIndex) throw new Error('GCAR/v1 layer index is not canonical');
    const topH = finite(view, offset + 8, `profile ${profileId} topH`);
    const tileOriginX = finite(view, offset + 12, `profile ${profileId} tile origin X`);
    const tileOriginZ = finite(view, offset + 16, `profile ${profileId} tile origin Z`);
    const tileSizeX = finite(view, offset + 20, `profile ${profileId} tile size X`);
    const tileSizeZ = finite(view, offset + 24, `profile ${profileId} tile size Z`);
    if (!(topH > 0 && tileSizeX > 0 && tileSizeZ > 0)) {
      throw new Error(`GCAR/v1 profile ${profileId} dimensions must be positive`);
    }
    if (view.getUint32(offset + 28, true) !== 0) throw new Error('GCAR/v1 profile flags are unsupported');
    profiles.push({
      profileId,
      layerIndex,
      topH,
      tileOriginX,
      tileOriginZ,
      tileSizeX,
      tileSizeZ,
      sourceSha256: bytesHex(bytes.subarray(offset + 32, offset + 64)),
    });
  }
  const depthBounds = new Float32Array(profileCount * sliceCount * 2);
  for (let bound = 0; bound < depthBounds.length; bound++) {
    const value = finite(view, depthOffset + bound * 4, `depth bound ${bound}`);
    depthBounds[bound] = value;
  }
  for (let profile = 0; profile < profileCount; profile++) {
    for (let slice = 0; slice < sliceCount; slice++) {
      const offset = (profile * sliceCount + slice) * 2;
      if (!(depthBounds[offset]! >= 0 && depthBounds[offset + 1]! > depthBounds[offset]!)) {
        throw new Error(`GCAR/v1 profile ${profile} slice ${slice} has invalid depth bounds`);
      }
    }
  }
  const halfTexels = new Uint16Array(profileCount * layerBytes / 2);
  for (let value = 0; value < halfTexels.length; value++) {
    halfTexels[value] = view.getUint16(payloadOffset + value * 2, true);
  }
  return {
    header: {
      version: 1,
      profileCount,
      layerWidth,
      layerHeight,
      storedTileWidth,
      storedTileHeight,
      interiorTileWidth,
      interiorTileHeight,
      atlasColumns,
      atlasRows,
      sliceCount,
      azimuthCount,
      elevationCount,
      gutter,
      texelBytes,
      texelFormat,
      projectionMode,
      latticeOrder,
      directionOffset,
      profileOffset,
      depthOffset,
      payloadOffset,
    },
    directions,
    profiles,
    depthBounds,
    halfTexels,
  };
}
