import {
  ClampToEdgeWrapping,
  DataArrayTexture,
  DataTexture,
  HalfFloatType,
  LinearFilter,
  NearestFilter,
  RGBAFormat,
  RGBAIntegerFormat,
  UnsignedIntType,
} from 'three';

export const CALAMAGROSTIS_GBC2_URL = new URL(
  '../../assets/groundcover/calamagrostis-canescens.gbc2',
  import.meta.url,
);

const HEADER_BYTES = 256;
const VERSION = 1;
const DESCRIPTOR_BYTES_PER_TEXEL = 16;
const PAYLOAD_BYTES_PER_TRIANGLE = 32;
const PAYLOAD_TEXTURE_WIDTH = 2048;
const K6_LAYERS = 6;

export interface LoadedGbc2Profile {
  profileId: number;
  descriptorTexture: DataTexture;
  descriptorWidth: number;
  descriptorHeight: number;
  payloadTexture: DataTexture;
  payloadTextureWidth: number;
  triangleCount: number;
  k6Texture: DataArrayTexture | null;
  k6Size: number;
  k6MipCount: number;
  head: Float32Array;
  tileOriginX: number;
  tileOriginZ: number;
  tileSizeX: number;
  tileSizeZ: number;
  sourceBounds: readonly [number, number, number, number, number, number];
  sourceSha256: string;
  residentBytes: number;
}

function positiveInt(view: DataView, offset: number, label: string): number {
  const value = view.getUint32(offset, true);
  if (value <= 0) throw new Error(`GBC2/v1 ${label} must be positive`);
  return value;
}

function finite(view: DataView, offset: number, label: string): number {
  const value = view.getFloat32(offset, true);
  if (!Number.isFinite(value)) throw new Error(`GBC2/v1 ${label} must be finite`);
  return value;
}

function section(
  bytes: Uint8Array,
  offset: number,
  length: number,
  label: string,
  alignment: number,
): Uint8Array {
  if (
    offset < HEADER_BYTES
    || length < 0
    || offset % alignment !== 0
    || offset + length > bytes.byteLength
  ) throw new Error(`GBC2/v1 ${label} section is invalid`);
  return bytes.subarray(offset, offset + length);
}

function hex(bytes: Uint8Array): string {
  return Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
}

/** Strict parser for the shader-ready GBC2/v1 container. Heavy compilation is
 * cook-side; this function only validates sections and exposes upload views. */
export function parseGbc2Profile(bytes: Uint8Array): LoadedGbc2Profile {
  if (
    bytes.byteLength < HEADER_BYTES
    || bytes[0] !== 0x47
    || bytes[1] !== 0x42
    || bytes[2] !== 0x43
    || bytes[3] !== 0x32
  ) throw new Error('ground-cover codec is not a GBC2 container');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(4, true) !== VERSION) throw new Error('ground-cover runtime requires GBC2/v1');
  if (view.getUint32(8, true) !== HEADER_BYTES) throw new Error('GBC2/v1 header size is not canonical');

  const profileId = view.getUint32(12, true);
  const descriptorWidth = positiveInt(view, 16, 'descriptor width');
  const descriptorHeight = positiveInt(view, 20, 'descriptor height');
  if (view.getUint32(24, true) !== DESCRIPTOR_BYTES_PER_TEXEL) {
    throw new Error('GBC2/v1 descriptor must be RGBA32Uint');
  }
  const descriptorOffset = view.getUint32(28, true);
  const descriptorBytes = view.getUint32(32, true);
  if (descriptorBytes !== descriptorWidth * descriptorHeight * DESCRIPTOR_BYTES_PER_TEXEL) {
    throw new Error('GBC2/v1 descriptor byte count is inconsistent');
  }
  const payloadOffset = view.getUint32(36, true);
  const payloadBytes = view.getUint32(40, true);
  const triangleCount = positiveInt(view, 44, 'triangle count');
  if (
    view.getUint32(48, true) !== PAYLOAD_BYTES_PER_TRIANGLE
    || payloadBytes !== triangleCount * PAYLOAD_BYTES_PER_TRIANGLE
  ) throw new Error('GBC2/v1 triangle payload is not the canonical 32-byte format');

  const descriptorSection = section(bytes, descriptorOffset, descriptorBytes, 'descriptor', 256);
  const payloadSection = section(bytes, payloadOffset, payloadBytes, 'payload', 256);
  const descriptorWords = new Uint32Array(
    descriptorSection.buffer,
    descriptorSection.byteOffset,
    descriptorSection.byteLength / 4,
  );
  const descriptorTexture = new DataTexture(
    descriptorWords,
    descriptorWidth,
    descriptorHeight,
    RGBAIntegerFormat,
    UnsignedIntType,
  );
  descriptorTexture.minFilter = NearestFilter;
  descriptorTexture.magFilter = NearestFilter;
  descriptorTexture.wrapS = ClampToEdgeWrapping;
  descriptorTexture.wrapT = ClampToEdgeWrapping;
  descriptorTexture.generateMipmaps = false;
  descriptorTexture.needsUpdate = true;
  descriptorTexture.name = `groundCoverGbc2Descriptor${profileId}`;

  const payloadTexels = triangleCount * 2;
  const payloadHeight = Math.ceil(payloadTexels / PAYLOAD_TEXTURE_WIDTH);
  const payloadWords = new Uint32Array(PAYLOAD_TEXTURE_WIDTH * payloadHeight * 4);
  payloadWords.set(new Uint32Array(
    payloadSection.buffer,
    payloadSection.byteOffset,
    payloadSection.byteLength / 4,
  ));
  const payloadTexture = new DataTexture(
    payloadWords,
    PAYLOAD_TEXTURE_WIDTH,
    payloadHeight,
    RGBAIntegerFormat,
    UnsignedIntType,
  );
  payloadTexture.minFilter = NearestFilter;
  payloadTexture.magFilter = NearestFilter;
  payloadTexture.wrapS = ClampToEdgeWrapping;
  payloadTexture.wrapT = ClampToEdgeWrapping;
  payloadTexture.generateMipmaps = false;
  payloadTexture.needsUpdate = true;
  payloadTexture.name = `groundCoverGbc2Payload${profileId}`;

  const k6Present = view.getUint32(52, true) !== 0;
  const k6Offset = view.getUint32(56, true);
  const k6Bytes = view.getUint32(60, true);
  const k6Size = k6Present ? positiveInt(view, 64, 'K6 size') : 0;
  const k6Layers = view.getUint32(68, true);
  const k6MipCount = view.getUint32(72, true);
  let k6Texture: DataArrayTexture | null = null;
  if (k6Present) {
    if (k6Layers !== K6_LAYERS || k6MipCount <= 0) throw new Error('GBC2/v1 K6 layout is invalid');
    const k6Section = section(bytes, k6Offset, k6Bytes, 'K6', 256);
    let byteOffset = 0;
    const levels: Array<{ data: Uint16Array; width: number; height: number; depth: number }> = [];
    let size = k6Size;
    for (let mip = 0; mip < k6MipCount; mip++) {
      const levelBytes = size * size * K6_LAYERS * 4 * 2;
      if (byteOffset + levelBytes > k6Section.byteLength) throw new Error('GBC2/v1 K6 mip bytes overflow');
      levels.push({
        data: new Uint16Array(
          k6Section.buffer,
          k6Section.byteOffset + byteOffset,
          levelBytes / 2,
        ),
        width: size,
        height: size,
        depth: K6_LAYERS,
      });
      byteOffset += levelBytes;
      size = Math.max(1, Math.floor(size / 2));
    }
    if (byteOffset !== k6Section.byteLength) throw new Error('GBC2/v1 K6 byte count is inconsistent');
    const base = levels[0]!;
    k6Texture = new DataArrayTexture(base.data, base.width, base.height, base.depth);
    k6Texture.format = RGBAFormat;
    k6Texture.type = HalfFloatType;
    k6Texture.minFilter = LinearFilter;
    k6Texture.magFilter = LinearFilter;
    k6Texture.wrapS = ClampToEdgeWrapping;
    k6Texture.wrapT = ClampToEdgeWrapping;
    k6Texture.generateMipmaps = false;
    k6Texture.mipmaps = levels;
    k6Texture.needsUpdate = true;
    k6Texture.name = `groundCoverGbc2K6${profileId}`;
  } else if (k6Bytes !== 0 || k6Layers !== 0 || k6MipCount !== 0) {
    throw new Error('GBC2/v1 absent K6 must have zero layout fields');
  }

  const headOffset = view.getUint32(76, true);
  const headBytes = view.getUint32(80, true);
  const headSection = headBytes > 0 ? section(bytes, headOffset, headBytes, 'head', 256) : null;
  if ((headBytes & 3) !== 0) throw new Error('GBC2/v1 head must contain float32 words');
  const head = headSection
    ? new Float32Array(headSection.buffer, headSection.byteOffset, headSection.byteLength / 4)
    : new Float32Array(0);
  if (k6Present !== (head.length > 0)) throw new Error('GBC2/v1 K6 and fixed head must be present together');

  const tileOriginX = finite(view, 84, 'tile origin X');
  const tileOriginZ = finite(view, 88, 'tile origin Z');
  const tileSizeX = finite(view, 92, 'tile size X');
  const tileSizeZ = finite(view, 96, 'tile size Z');
  const sourceBounds = [
    finite(view, 100, 'source minimum X'),
    finite(view, 104, 'source minimum Y'),
    finite(view, 108, 'source minimum Z'),
    finite(view, 112, 'source maximum X'),
    finite(view, 116, 'source maximum Y'),
    finite(view, 120, 'source maximum Z'),
  ] as const;
  if (!(tileSizeX > 0 && tileSizeZ > 0)) throw new Error('GBC2/v1 tile size must be positive');

  return {
    profileId,
    descriptorTexture,
    descriptorWidth,
    descriptorHeight,
    payloadTexture,
    payloadTextureWidth: PAYLOAD_TEXTURE_WIDTH,
    triangleCount,
    k6Texture,
    k6Size,
    k6MipCount,
    head,
    tileOriginX,
    tileOriginZ,
    tileSizeX,
    tileSizeZ,
    sourceBounds,
    sourceSha256: hex(bytes.subarray(128, 160)),
    residentBytes: descriptorBytes + payloadWords.byteLength + k6Bytes + headBytes,
  };
}

export async function loadGbc2Profile(
  url: URL | string,
  expectedProfileId: number,
): Promise<LoadedGbc2Profile | null> {
  // Inspect the fixed header first.  Asset servers which support byte ranges
  // avoid transferring an incomplete 100+ MiB cook; development servers that
  // answer 200 simply give us the complete body once and it is reused below.
  const headerResponse = await fetch(url, { headers: { Range: `bytes=0-${HEADER_BYTES - 1}` } });
  if (!headerResponse.ok) {
    throw new Error(`ground-cover GBC2 header fetch failed (${headerResponse.status} ${headerResponse.statusText})`);
  }
  const firstBytes = new Uint8Array(await headerResponse.arrayBuffer());
  if (firstBytes.byteLength < HEADER_BYTES) throw new Error('ground-cover GBC2 header is truncated');
  // A descriptor/payload-only checkpoint is deliberately not runtime-ready.
  // Do not allocate/upload its ~131 MiB while the active visual path still has
  // to fall back to GCRP; once the fitted terminal is present the same strict
  // parser below exposes the complete immutable resource set.
  const header = new DataView(firstBytes.buffer, firstBytes.byteOffset, HEADER_BYTES);
  if (header.getUint32(52, true) === 0) return null;
  const bytes = headerResponse.status === 206
    ? await (async (): Promise<Uint8Array> => {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`ground-cover GBC2 fetch failed (${response.status} ${response.statusText})`);
        }
        return new Uint8Array(await response.arrayBuffer());
      })()
    : firstBytes;
  const profile = parseGbc2Profile(bytes);
  if (profile.profileId !== expectedProfileId) {
    throw new Error(`GBC2/v1 profile id ${profile.profileId} != expected ${expectedProfileId}`);
  }
  return profile;
}
