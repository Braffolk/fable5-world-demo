import {
  ClampToEdgeWrapping,
  DataTexture,
  FloatType,
  HalfFloatType,
  NearestFilter,
  RedIntegerFormat,
  RGBAFormat,
  RGBAIntegerFormat,
  UnsignedIntType,
} from 'three';

/**
 * Strict loader for the honest GBR4/v4 boundary-transfer reference format.
 *
 * v4 is intentionally not interchangeable with the experimental v2 cap
 * tensor.  Each address record is tagged INVALID, certified MISS, certified
 * REGULAR, or physically-filtered MIXED.  A MIXED record never supplies a
 * categorical surface.  The format also makes physical footprint level an
 * explicit address component rather than silently reusing one four-metre
 * filter at every standoff.
 */

export const CALAMAGROSTIS_GBR4_V4_REFERENCE_URL = new URL(
  '../../assets/groundcover/calamagrostis-canescens.reference-v4.gbr4',
  import.meta.url,
);

export const enum Gbr4V4RecordMode {
  INVALID = 0,
  CERTIFIED_MISS = 1,
  CERTIFIED_REGULAR = 2,
  FILTERED_MIXED = 3,
}

const HEADER_BYTES = 512;
const VERSION = 4;
const SECTION_ALIGNMENT = 256;
const FACE_COUNT = 6;
const CONTINUOUS_DIMENSIONS = 4;
const ADDRESS_MODE_BOUNDARY_LAMBERT = 4;
const LAMBERT_HEMISPHERE_DISK = 4;
const RECORD_CLASS_ENUM = 1;
const FACE_TIE_LOWEST_ID = 1;
const FACE_ENTRY_BYTES = 16;
const LEVEL_ENTRY_BYTES = 64;
const MIXED_TEXELS_PER_RECORD = 4;
const REGULAR_TEXELS_PER_RECORD = 5;
const REQUIRED_FLAGS = 0b1_1111;

interface ByteSection {
  readonly offset: number;
  readonly length: number;
  readonly label: string;
}

export interface Gbr4V4Face {
  readonly faceId: number;
  readonly axis: 0 | 1 | 2;
  readonly side: -1 | 1;
}

export interface Gbr4V4FootprintLevel {
  readonly standoffMetres: number;
  readonly footprintDiameterMetres: number;
  readonly pixelAngleRadians: number;
  readonly addressFirst: number;
  readonly addressCount: number;
  readonly mixedFirst: number;
  readonly mixedCount: number;
}

export interface LoadedGbr4V4Profile {
  readonly profileId: number;
  readonly addressTexture: DataTexture;
  readonly mixedPayloadTexture: DataTexture;
  readonly regularPayloadTexture: DataTexture;
  readonly correctionTexture: DataTexture;
  readonly addressTextureWidth: number;
  readonly addressTextureHeight: number;
  readonly mixedAtlasWidth: number;
  readonly mixedAtlasHeight: number;
  readonly regularAtlasWidth: number;
  readonly regularAtlasHeight: number;
  readonly correctionTextureWidth: number;
  readonly correctionTextureHeight: number;
  readonly faces: readonly Gbr4V4Face[];
  readonly levels: readonly Gbr4V4FootprintLevel[];
  readonly boundaryCells: number;
  readonly directionCells: number;
  readonly addressRecordCount: number;
  readonly mixedRecordCount: number;
  readonly regularRecordCount: number;
  readonly correctionCount: number;
  readonly correctionSeed: number;
  readonly fineBoundaryCells: number;
  readonly fineDirectionCells: number;
  readonly recordModeCounts: Readonly<Record<Gbr4V4RecordMode, number>>;
  readonly tileOriginX: number;
  readonly tileOriginZ: number;
  readonly tileSizeX: number;
  readonly tileSizeZ: number;
  readonly carrierBounds: readonly [number, number, number, number, number, number];
  readonly horizonMetres: number;
  readonly quadratureSide: number;
  readonly sourceSha256: string;
  readonly recipeSha256: string;
  readonly publicationStatus: 'REFERENCE_RED' | 'VISUAL_PRODUCTION_GREEN';
  /** Fail-closed: visible runtime binding is forbidden unless this is true. */
  readonly runtimeBindAllowed: boolean;
  /** Exact bytes allocated by the three upload views, including atlas slack. */
  readonly gpuResidentBytes: number;
  readonly containerBytes: number;
  /** Fixed terminal schedule: correction, base tag, four selected payload reads. */
  readonly maximumTerminalReads: 6;
}

function checkedProduct(label: string, ...factors: readonly number[]): number {
  let value = 1;
  for (const factor of factors) {
    if (!Number.isSafeInteger(factor) || factor < 0) throw new Error(`GBR4/v4 ${label} has an invalid factor`);
    value *= factor;
    if (!Number.isSafeInteger(value)) throw new Error(`GBR4/v4 ${label} overflows`);
  }
  return value;
}

function positiveInt(view: DataView, offset: number, label: string): number {
  const value = view.getUint32(offset, true);
  if (!(value > 0)) throw new Error(`GBR4/v4 ${label} must be positive`);
  return value;
}

function finite(view: DataView, offset: number, label: string): number {
  const value = view.getFloat32(offset, true);
  if (!Number.isFinite(value)) throw new Error(`GBR4/v4 ${label} must be finite`);
  return value;
}

function positiveFinite(view: DataView, offset: number, label: string): number {
  const value = finite(view, offset, label);
  if (!(value > 0)) throw new Error(`GBR4/v4 ${label} must be positive`);
  return value;
}

function section(bytes: Uint8Array, offset: number, length: number, label: string): ByteSection {
  const end = offset + length;
  if (
    offset < HEADER_BYTES
    || length <= 0
    || offset % SECTION_ALIGNMENT !== 0
    || !Number.isSafeInteger(end)
    || end > bytes.byteLength
  ) throw new Error(`GBR4/v4 ${label} section is invalid`);
  return Object.freeze({ offset, length, label });
}

function ensureDisjoint(sections: readonly ByteSection[]): void {
  const ordered = [...sections].sort((a, b) => a.offset - b.offset);
  let end = HEADER_BYTES;
  for (const item of ordered) {
    if (item.offset < end) throw new Error(`GBR4/v4 ${item.label} overlaps an earlier section`);
    end = item.offset + item.length;
  }
}

function requireZero(bytes: Uint8Array, begin: number, end: number, label: string): void {
  for (let index = begin; index < end; index++) {
    if (bytes[index] !== 0) throw new Error(`GBR4/v4 ${label} must be zero`);
  }
}

function hex(bytes: Uint8Array): string {
  return Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
}

function correctionHash(low: number, high: number, seed: number): number {
  let value = Math.imul((low ^ seed) >>> 0, 0x9e37_79b1);
  value = (value + Math.imul(high >>> 0, 0x85eb_ca6b)) >>> 0;
  value ^= value >>> 16;
  return value >>> 0;
}

function configureAddressTexture(data: Uint32Array, width: number, height: number, profileId: number): DataTexture {
  const texture = new DataTexture(data, width, height, RedIntegerFormat, UnsignedIntType);
  texture.minFilter = NearestFilter;
  texture.magFilter = NearestFilter;
  texture.wrapS = ClampToEdgeWrapping;
  texture.wrapT = ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  texture.name = `groundCoverGbr4V4Address${profileId}`;
  return texture;
}

function configureCorrectionTexture(data: Uint32Array, width: number, height: number, profileId: number): DataTexture {
  const texture = new DataTexture(data, width, height, RGBAIntegerFormat, UnsignedIntType);
  texture.minFilter = NearestFilter;
  texture.magFilter = NearestFilter;
  texture.wrapS = ClampToEdgeWrapping;
  texture.wrapT = ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  texture.name = `groundCoverGbr4V4Correction${profileId}`;
  return texture;
}

function configurePayloadTexture(
  data: Uint16Array | Float32Array,
  width: number,
  height: number,
  profileId: number,
  kind: 'Mixed' | 'Regular',
): DataTexture {
  const texture = new DataTexture(
    data,
    width,
    height,
    RGBAFormat,
    kind === 'Mixed' ? HalfFloatType : FloatType,
  );
  texture.minFilter = NearestFilter;
  texture.magFilter = NearestFilter;
  texture.wrapS = ClampToEdgeWrapping;
  texture.wrapT = ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  texture.name = `groundCoverGbr4V4${kind}Payload${profileId}`;
  return texture;
}

export function parseGbr4V4Profile(bytes: Uint8Array): LoadedGbr4V4Profile {
  if (
    bytes.byteLength < HEADER_BYTES
    || bytes[0] !== 0x47 || bytes[1] !== 0x42 || bytes[2] !== 0x52 || bytes[3] !== 0x34
  ) throw new Error('ground-cover codec is not a GBR4 container');
  if (bytes.byteLength % SECTION_ALIGNMENT !== 0) throw new Error('GBR4/v4 container is not 256-byte aligned');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(4, true) !== VERSION) throw new Error('ground-cover v4 loader requires GBR4/v4');
  if (view.getUint32(8, true) !== HEADER_BYTES) throw new Error('GBR4/v4 header size is not canonical');
  const profileId = view.getUint32(12, true);
  if (view.getUint32(16, true) !== FACE_COUNT) throw new Error('GBR4/v4 requires six boundary faces');
  if (view.getUint32(20, true) !== ADDRESS_MODE_BOUNDARY_LAMBERT) {
    throw new Error('GBR4/v4 address is not boundary-face UV plus Lambert direction disk');
  }
  if (view.getUint32(24, true) !== CONTINUOUS_DIMENSIONS) throw new Error('GBR4/v4 address is not four-dimensional');
  const levelCount = positiveInt(view, 28, 'footprint level count');
  const boundaryCells = positiveInt(view, 32, 'boundary cell count');
  const directionCells = positiveInt(view, 36, 'direction cell count');
  if (view.getUint32(40, true) !== 1) throw new Error('GBR4/v4 address records must be packed R32Uint');
  if (view.getUint32(44, true) !== MIXED_TEXELS_PER_RECORD) throw new Error('GBR4/v4 mixed payload stride is not canonical');
  if (view.getUint32(48, true) !== REGULAR_TEXELS_PER_RECORD) throw new Error('GBR4/v4 regular payload stride is not canonical');

  const faceTable = section(bytes, view.getUint32(52, true), view.getUint32(56, true), 'face table');
  const levelTable = section(bytes, view.getUint32(60, true), view.getUint32(64, true), 'level table');
  const address = section(bytes, view.getUint32(68, true), view.getUint32(72, true), 'address');
  const mixed = section(bytes, view.getUint32(76, true), view.getUint32(80, true), 'mixed payload');
  const regular = section(bytes, view.getUint32(84, true), view.getUint32(88, true), 'regular payload');
  const correction = section(bytes, view.getUint32(280, true), view.getUint32(284, true), 'correction');
  ensureDisjoint([faceTable, levelTable, address, mixed, regular, correction]);
  if (faceTable.length !== FACE_COUNT * FACE_ENTRY_BYTES) throw new Error('GBR4/v4 face table byte count is inconsistent');
  if (levelTable.length !== levelCount * LEVEL_ENTRY_BYTES) throw new Error('GBR4/v4 level table byte count is inconsistent');

  const mixedRecordCount = view.getUint32(92, true);
  const regularRecordCount = view.getUint32(96, true);
  const addressRecordCount = positiveInt(view, 100, 'address record count');
  const addressTextureWidth = positiveInt(view, 104, 'address texture width');
  const addressTextureHeight = positiveInt(view, 108, 'address texture height');
  const mixedAtlasWidth = positiveInt(view, 112, 'mixed atlas width');
  const mixedAtlasHeight = positiveInt(view, 116, 'mixed atlas height');
  const regularAtlasWidth = positiveInt(view, 120, 'regular atlas width');
  const regularAtlasHeight = positiveInt(view, 124, 'regular atlas height');
  const correctionCount = view.getUint32(288, true);
  const correctionTextureWidth = positiveInt(view, 292, 'correction texture width');
  const correctionTextureHeight = positiveInt(view, 296, 'correction texture height');
  const fineBoundaryCells = positiveInt(view, 300, 'fine boundary cell count');
  const fineDirectionCells = positiveInt(view, 304, 'fine direction cell count');
  const correctionSeed = view.getUint32(308, true);
  const publicationStatusCode = view.getUint32(312, true);
  if (publicationStatusCode > 1) throw new Error('GBR4/v4 publication status is invalid');
  if (view.getUint32(264, true) !== REQUIRED_FLAGS) throw new Error('GBR4/v4 flags are not canonical');
  if (view.getUint32(268, true) !== LAMBERT_HEMISPHERE_DISK) throw new Error('GBR4/v4 direction chart is not Lambert');
  if (view.getUint32(272, true) !== FACE_TIE_LOWEST_ID) throw new Error('GBR4/v4 face tie rule is not canonical');
  if (view.getUint32(276, true) !== RECORD_CLASS_ENUM) throw new Error('GBR4/v4 record class enum is not canonical');

  const expectedAddressRecords = checkedProduct(
    'address lattice', levelCount, FACE_COUNT, boundaryCells, boundaryCells, directionCells, directionCells,
  );
  if (addressRecordCount !== expectedAddressRecords) throw new Error('GBR4/v4 address lattice size is inconsistent');
  const addressTexels = checkedProduct('address texture texels', addressTextureWidth, addressTextureHeight);
  if (addressTexels < addressRecordCount || address.length !== addressTexels * 4) {
    throw new Error('GBR4/v4 address texture dimensions are inconsistent');
  }
  const mixedTexels = checkedProduct('mixed atlas texels', mixedAtlasWidth, mixedAtlasHeight);
  const regularTexels = checkedProduct('regular atlas texels', regularAtlasWidth, regularAtlasHeight);
  const correctionTexels = checkedProduct('correction texture texels', correctionTextureWidth, correctionTextureHeight);
  if (mixedTexels < Math.max(1, mixedRecordCount) * MIXED_TEXELS_PER_RECORD || mixed.length !== mixedTexels * 8) {
    throw new Error('GBR4/v4 mixed atlas dimensions are inconsistent');
  }
  if (regularTexels < Math.max(1, regularRecordCount) * REGULAR_TEXELS_PER_RECORD || regular.length !== regularTexels * 16) {
    throw new Error('GBR4/v4 regular atlas dimensions are inconsistent');
  }
  if (
    correctionTexels < Math.max(1, correctionCount)
    || correction.length !== correctionTexels * 16
    || (correctionTexels & (correctionTexels - 1)) !== 0
  ) throw new Error('GBR4/v4 correction texture is not a power-of-two RGBA32Uint table');

  const faceView = new DataView(bytes.buffer, bytes.byteOffset + faceTable.offset, faceTable.length);
  const faces: Gbr4V4Face[] = [];
  for (let face = 0; face < FACE_COUNT; face++) {
    const offset = face * FACE_ENTRY_BYTES;
    const faceId = faceView.getUint8(offset);
    const axis = faceView.getUint8(offset + 1);
    const side = faceView.getInt8(offset + 2);
    if (faceId !== face || axis !== Math.floor(face / 2) || side !== (face % 2 === 0 ? -1 : 1)) {
      throw new Error(`GBR4/v4 face ${face} is not canonical`);
    }
    requireZero(bytes, faceTable.offset + offset + 3, faceTable.offset + offset + FACE_ENTRY_BYTES, `face ${face} reserved bytes`);
    faces.push(Object.freeze({ faceId, axis: axis as 0 | 1 | 2, side: side as -1 | 1 }));
  }

  const levelView = new DataView(bytes.buffer, bytes.byteOffset + levelTable.offset, levelTable.length);
  const recordsPerLevel = addressRecordCount / levelCount;
  const levels: Gbr4V4FootprintLevel[] = [];
  for (let level = 0; level < levelCount; level++) {
    const offset = level * LEVEL_ENTRY_BYTES;
    const standoffMetres = positiveFinite(levelView, offset, `level ${level} standoff`);
    const footprintDiameterMetres = positiveFinite(levelView, offset + 4, `level ${level} footprint diameter`);
    const pixelAngleRadians = positiveFinite(levelView, offset + 8, `level ${level} pixel angle`);
    const addressFirst = levelView.getUint32(offset + 12, true);
    const addressCount = levelView.getUint32(offset + 16, true);
    const mixedFirst = levelView.getUint32(offset + 20, true);
    const mixedCount = levelView.getUint32(offset + 24, true);
    if (addressFirst !== level * recordsPerLevel || addressCount !== recordsPerLevel) {
      throw new Error(`GBR4/v4 level ${level} address range is inconsistent`);
    }
    if (mixedFirst + mixedCount > mixedRecordCount) throw new Error(`GBR4/v4 level ${level} mixed range is invalid`);
    requireZero(bytes, levelTable.offset + offset + 28, levelTable.offset + offset + LEVEL_ENTRY_BYTES, `level ${level} reserved bytes`);
    levels.push(Object.freeze({
      standoffMetres,
      footprintDiameterMetres,
      pixelAngleRadians,
      addressFirst,
      addressCount,
      mixedFirst,
      mixedCount,
    }));
  }

  const addressWords = new Uint32Array(bytes.buffer, bytes.byteOffset + address.offset, addressTexels);
  const modeCounts: Record<Gbr4V4RecordMode, number> = {
    [Gbr4V4RecordMode.INVALID]: 0,
    [Gbr4V4RecordMode.CERTIFIED_MISS]: 0,
    [Gbr4V4RecordMode.CERTIFIED_REGULAR]: 0,
    [Gbr4V4RecordMode.FILTERED_MIXED]: 0,
  };
  for (let record = 0; record < addressRecordCount; record++) {
    const word = addressWords[record]!;
    const mode = (word & 3) as Gbr4V4RecordMode;
    const payload = word >>> 2;
    modeCounts[mode]++;
    if (mode === Gbr4V4RecordMode.CERTIFIED_REGULAR && payload >= regularRecordCount) {
      throw new Error('GBR4/v4 regular address references an absent payload');
    }
    if (mode === Gbr4V4RecordMode.FILTERED_MIXED && payload >= mixedRecordCount) {
      throw new Error('GBR4/v4 mixed address references an absent payload');
    }
    if ((mode === Gbr4V4RecordMode.INVALID || mode === Gbr4V4RecordMode.CERTIFIED_MISS) && payload !== 0) {
      throw new Error('GBR4/v4 empty address carries a payload index');
    }
  }
  for (let record = addressRecordCount; record < addressWords.length; record++) {
    if (addressWords[record] !== 0) throw new Error('GBR4/v4 address atlas padding must be zero');
  }
  const correctionWords = new Uint32Array(
    bytes.buffer,
    bytes.byteOffset + correction.offset,
    correction.length / 4,
  );
  let populatedCorrections = 0;
  for (let slot = 0; slot < correctionTexels; slot++) {
    const payloadPlusOne = correctionWords[slot * 4 + 2]!;
    if (payloadPlusOne === 0) {
      if (correctionWords[slot * 4] !== 0 || correctionWords[slot * 4 + 1] !== 0 || correctionWords[slot * 4 + 3] !== 0) {
        throw new Error('GBR4/v4 empty correction slot is not zero');
      }
      continue;
    }
    const keyLow = correctionWords[slot * 4]!;
    const keyHigh = correctionWords[slot * 4 + 1]!;
    if (
      payloadPlusOne - 1 >= regularRecordCount
      || correctionWords[slot * 4 + 3] !== correctionSeed
      || (correctionHash(keyLow, keyHigh, correctionSeed) & (correctionTexels - 1)) !== slot
    ) {
      throw new Error('GBR4/v4 correction references an absent regular payload or wrong seed');
    }
    populatedCorrections++;
  }
  if (populatedCorrections !== correctionCount) throw new Error('GBR4/v4 correction count is inconsistent');

  const tileOriginX = finite(view, 128, 'tile origin X');
  const tileOriginZ = finite(view, 132, 'tile origin Z');
  const tileSizeX = positiveFinite(view, 136, 'tile size X');
  const tileSizeZ = positiveFinite(view, 140, 'tile size Z');
  const carrierBounds = [
    finite(view, 144, 'carrier minimum X'), finite(view, 148, 'carrier minimum Y'), finite(view, 152, 'carrier minimum Z'),
    finite(view, 156, 'carrier maximum X'), finite(view, 160, 'carrier maximum Y'), finite(view, 164, 'carrier maximum Z'),
  ] as const;
  if (!(carrierBounds[0] < carrierBounds[3] && carrierBounds[1] < carrierBounds[4] && carrierBounds[2] < carrierBounds[5])) {
    throw new Error('GBR4/v4 carrier bounds are empty or inverted');
  }
  const sourceHash = bytes.subarray(168, 200);
  const recipeHash = bytes.subarray(200, 232);
  if (sourceHash.every((value) => value === 0) || recipeHash.every((value) => value === 0)) {
    throw new Error('GBR4/v4 provenance hashes must be populated');
  }
  const horizonMetres = positiveFinite(view, 232, 'horizon');
  const quadratureSide = positiveInt(view, 236, 'quadrature side');
  requireZero(bytes, 240, 264, 'header reserved bytes 240..263');
  requireZero(bytes, 316, HEADER_BYTES, 'header reserved bytes 316..511');

  const mixedWords = new Uint16Array(bytes.buffer, bytes.byteOffset + mixed.offset, mixed.length / 2);
  const regularWords = new Float32Array(bytes.buffer, bytes.byteOffset + regular.offset, regular.length / 4);
  for (let record = 0; record < regularRecordCount; record++) {
    const base = record * REGULAR_TEXELS_PER_RECORD * 4;
    for (let component = 0; component < REGULAR_TEXELS_PER_RECORD * 4; component++) {
      if (!Number.isFinite(regularWords[base + component]!)) {
        throw new Error(`GBR4/v4 regular payload ${record} is not finite`);
      }
    }
    const normalLength = Math.hypot(regularWords[base]!, regularWords[base + 1]!, regularWords[base + 2]!);
    if (
      Math.abs(normalLength - 1) > 1e-4
      || !(regularWords[base + 12]! > 0)
      || !(regularWords[base + 13]! > 0)
      || !(regularWords[base + 14]! > 0)
      || regularWords[base + 15]! < 0
    ) throw new Error(`GBR4/v4 regular payload ${record} lacks positive source firstness margins`);
    if (publicationStatusCode === 1 && !(regularWords[base + 15]! > 0)) {
      throw new Error(`GBR4/v4 GREEN regular payload ${record} lacks a positive terrain/motion firstness margin`);
    }
  }
  const addressTexture = configureAddressTexture(addressWords, addressTextureWidth, addressTextureHeight, profileId);
  const mixedPayloadTexture = configurePayloadTexture(mixedWords, mixedAtlasWidth, mixedAtlasHeight, profileId, 'Mixed');
  const regularPayloadTexture = configurePayloadTexture(regularWords, regularAtlasWidth, regularAtlasHeight, profileId, 'Regular');
  const correctionTexture = configureCorrectionTexture(
    correctionWords,
    correctionTextureWidth,
    correctionTextureHeight,
    profileId,
  );

  return Object.freeze({
    profileId,
    addressTexture,
    mixedPayloadTexture,
    regularPayloadTexture,
    correctionTexture,
    addressTextureWidth,
    addressTextureHeight,
    mixedAtlasWidth,
    mixedAtlasHeight,
    regularAtlasWidth,
    regularAtlasHeight,
    correctionTextureWidth,
    correctionTextureHeight,
    faces: Object.freeze(faces),
    levels: Object.freeze(levels),
    boundaryCells,
    directionCells,
    addressRecordCount,
    mixedRecordCount,
    regularRecordCount,
    correctionCount,
    correctionSeed,
    fineBoundaryCells,
    fineDirectionCells,
    recordModeCounts: Object.freeze(modeCounts),
    tileOriginX,
    tileOriginZ,
    tileSizeX,
    tileSizeZ,
    carrierBounds: Object.freeze(carrierBounds),
    horizonMetres,
    quadratureSide,
    sourceSha256: hex(sourceHash),
    recipeSha256: hex(recipeHash),
    publicationStatus: publicationStatusCode === 1 ? 'VISUAL_PRODUCTION_GREEN' : 'REFERENCE_RED',
    runtimeBindAllowed: publicationStatusCode === 1,
    gpuResidentBytes:
      addressWords.byteLength
      + mixedWords.byteLength
      + regularWords.byteLength
      + correctionWords.byteLength,
    containerBytes: bytes.byteLength,
    maximumTerminalReads: 6,
  });
}

export async function loadGbr4V4Profile(
  url: URL | string,
  expectedProfileId: number,
): Promise<LoadedGbr4V4Profile> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`ground-cover GBR4/v4 fetch failed (${response.status} ${response.statusText})`);
  const profile = parseGbr4V4Profile(new Uint8Array(await response.arrayBuffer()));
  if (profile.profileId !== expectedProfileId) {
    throw new Error(`GBR4/v4 profile id ${profile.profileId} != expected ${expectedProfileId}`);
  }
  return profile;
}
