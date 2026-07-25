import {
  ClampToEdgeWrapping,
  DataTexture,
  HalfFloatType,
  LinearFilter,
  NearestFilter,
  RedIntegerFormat,
  RGBAFormat,
  RGBAIntegerFormat,
  UnsignedIntType,
} from 'three';

export const CALAMAGROSTIS_GBR4_URL = new URL(
  '../../assets/groundcover/calamagrostis-canescens.gbr4',
  import.meta.url,
);

const HEADER_BYTES = 512;
const VERSION = 2;
const SECTION_ALIGNMENT = 256;
const CHART_COUNT = 2;
const ADDRESS_MODE_PERIODIC_CAP = 2;
const CONTINUOUS_DIMENSIONS = 4;
const DESCRIPTOR_FORMAT_U16 = 1;
const SCALE_COUNT = 1;
const SCALE_ENTRY_BYTES = 64;
const CHART_ENTRY_BYTES = 16;
const BYTES_PER_ATLAS_TEXEL = 8;
const MAX_DESCRIPTOR_ID = 65535;
const REQUIRED_FLAGS = 0b1111;
const FORWARD_HORIZON_METRES = 155;
const SLOPE_CHART_RATIONAL_COMPACT = 2;
const CAP_CONVENTION = 2;
const DESCRIPTOR_TEXTURE_MAX_WIDTH = 2048;

export interface Gbr4Chart {
  readonly chartId: number;
  readonly inwardYSign: -1 | 1;
  readonly descriptorBase: number;
  readonly blockCount: number;
}

export interface Gbr4Scale {
  readonly standoffMetres: number;
  readonly pixelAngleRadians: number;
  readonly descriptorFirst: number;
  readonly descriptorCount: number;
  readonly codebookFirst: number;
  readonly codebookCount: number;
  readonly atlasWidth: number;
  readonly atlasHeight: number;
}

export interface LoadedGbr4Profile {
  readonly profileId: number;
  readonly descriptorTexture: DataTexture;
  readonly descriptorTextureWidth: number;
  readonly descriptorTextureHeight: number;
  readonly frontColorTexture: DataTexture;
  readonly backColorTexture: DataTexture;
  readonly categoricalTexture: DataTexture;
  readonly charts: readonly Gbr4Chart[];
  readonly scale: Gbr4Scale;
  readonly cellResolution: number;
  readonly blockCellEdge: number;
  readonly blockSampleEdge: number;
  readonly blocksPerAxis: number;
  readonly atlasWidth: number;
  readonly atlasHeight: number;
  readonly codewordTileWidth: number;
  readonly codewordTileHeight: number;
  readonly codebookCount: number;
  readonly flags: number;
  readonly tileOriginX: number;
  readonly tileOriginZ: number;
  readonly tileSizeX: number;
  readonly tileSizeZ: number;
  readonly carrierBounds: readonly [number, number, number, number, number, number];
  readonly horizonMetres: number;
  readonly pixelAngleRadians: number;
  readonly standoffMetres: number;
  readonly quadratureSide: number;
  readonly sourceSha256: string;
  readonly recipeSha256: string;
  readonly residentBytes: number;
  readonly containerBytes: number;
}

interface ByteSection {
  readonly offset: number;
  readonly length: number;
  readonly label: string;
}

function positiveInt(view: DataView, offset: number, label: string): number {
  const value = view.getUint32(offset, true);
  if (value <= 0) throw new Error(`GBR4/v2 ${label} must be positive`);
  return value;
}

function finite(view: DataView, offset: number, label: string): number {
  const value = view.getFloat32(offset, true);
  if (!Number.isFinite(value)) throw new Error(`GBR4/v2 ${label} must be finite`);
  return value;
}

function positiveFinite(view: DataView, offset: number, label: string): number {
  const value = finite(view, offset, label);
  if (!(value > 0)) throw new Error(`GBR4/v2 ${label} must be positive`);
  return value;
}

function checkedProduct(label: string, ...factors: readonly number[]): number {
  let product = 1;
  for (const factor of factors) {
    if (!Number.isSafeInteger(factor) || factor < 0) {
      throw new Error(`GBR4/v2 ${label} has an invalid factor`);
    }
    product *= factor;
    if (!Number.isSafeInteger(product)) throw new Error(`GBR4/v2 ${label} overflows`);
  }
  return product;
}

function power4(value: number, label: string): number {
  return checkedProduct(label, value, value, value, value);
}

function byteSection(
  bytes: Uint8Array,
  offset: number,
  length: number,
  label: string,
): ByteSection {
  const end = offset + length;
  if (
    offset < HEADER_BYTES
    || length <= 0
    || offset % SECTION_ALIGNMENT !== 0
    || !Number.isSafeInteger(end)
    || end > bytes.byteLength
  ) throw new Error(`GBR4/v2 ${label} section is invalid`);
  return Object.freeze({ offset, length, label });
}

function sectionBytes(bytes: Uint8Array, section: ByteSection): Uint8Array {
  return bytes.subarray(section.offset, section.offset + section.length);
}

function ensureDisjoint(sections: readonly ByteSection[]): void {
  const ordered = [...sections].sort((left, right) => left.offset - right.offset);
  let previousEnd = HEADER_BYTES;
  for (const current of ordered) {
    if (current.offset < previousEnd) {
      throw new Error(`GBR4/v2 ${current.label} section overlaps an earlier section`);
    }
    previousEnd = current.offset + current.length;
  }
}

function requireZero(bytes: Uint8Array, begin: number, end: number, label: string): void {
  for (let index = begin; index < end; index++) {
    if (bytes[index] !== 0) throw new Error(`GBR4/v2 ${label} must be zero`);
  }
}

function hex(bytes: Uint8Array): string {
  return Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
}

function configureDescriptorTexture(data: Uint32Array, width: number, height: number, profileId: number): DataTexture {
  // Three's WebGPU upload table currently exposes integer sampled textures only
  // as 32-bit formats. The container remains compact u16; expansion is an upload
  // compatibility detail and does not change descriptor values or shader reads.
  const texture = new DataTexture(data, width, height, RedIntegerFormat, UnsignedIntType);
  texture.minFilter = NearestFilter;
  texture.magFilter = NearestFilter;
  texture.wrapS = ClampToEdgeWrapping;
  texture.wrapT = ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  texture.name = `groundCoverGbr4Descriptor${profileId}`;
  return texture;
}

function configureColorTexture(
  data: Uint16Array,
  width: number,
  height: number,
  profileId: number,
  stratum: 'Front' | 'Back',
): DataTexture {
  const texture = new DataTexture(data, width, height, RGBAFormat, HalfFloatType);
  texture.minFilter = LinearFilter;
  texture.magFilter = LinearFilter;
  texture.wrapS = ClampToEdgeWrapping;
  texture.wrapT = ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  texture.name = `groundCoverGbr4${stratum}Color${profileId}`;
  return texture;
}

function configureCategoricalTexture(
  data: Uint32Array,
  width: number,
  height: number,
  profileId: number,
): DataTexture {
  const texture = new DataTexture(data, width, height, RGBAIntegerFormat, UnsignedIntType);
  texture.minFilter = NearestFilter;
  texture.magFilter = NearestFilter;
  texture.wrapS = ClampToEdgeWrapping;
  texture.wrapT = ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  texture.name = `groundCoverGbr4Categorical${profileId}`;
  return texture;
}

/** Strict parser and upload-view constructor for the active periodic-cap
 * GBR4/v2 container. Heavy ray integration and brick construction remain
 * cook-side; the returned resource set is the fixed ten-read runtime input. */
export function parseGbr4Profile(bytes: Uint8Array): LoadedGbr4Profile {
  if (
    bytes.byteLength < HEADER_BYTES
    || bytes[0] !== 0x47
    || bytes[1] !== 0x42
    || bytes[2] !== 0x52
    || bytes[3] !== 0x34
  ) throw new Error('ground-cover codec is not a GBR4 container');
  if (bytes.byteLength % SECTION_ALIGNMENT !== 0) {
    throw new Error('GBR4/v2 container size is not 256-byte aligned');
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(4, true) !== VERSION) throw new Error('ground-cover runtime requires GBR4/v2');
  if (view.getUint32(8, true) !== HEADER_BYTES) throw new Error('GBR4/v2 header size is not canonical');
  const profileId = view.getUint32(12, true);
  if (view.getUint32(16, true) !== CHART_COUNT) throw new Error('GBR4/v2 requires two cap charts');
  if (view.getUint32(20, true) !== ADDRESS_MODE_PERIODIC_CAP) {
    throw new Error('GBR4/v2 address mode is not periodic cap phase+slope');
  }
  if (view.getUint32(24, true) !== CONTINUOUS_DIMENSIONS) {
    throw new Error('GBR4/v2 address must have four continuous dimensions');
  }

  const cellResolution = positiveInt(view, 28, 'cell resolution');
  const blockCellEdge = positiveInt(view, 32, 'block cell edge');
  const blockSampleEdge = positiveInt(view, 36, 'block sample edge');
  const blocksPerAxis = positiveInt(view, 40, 'blocks per axis');
  if (blockSampleEdge !== blockCellEdge + 1) throw new Error('GBR4/v2 block sample edge must equal B+1');
  if (cellResolution % blockCellEdge !== 0 || blocksPerAxis !== cellResolution / blockCellEdge) {
    throw new Error('GBR4/v2 block grid does not exactly partition the cell grid');
  }
  if (view.getUint32(44, true) !== SCALE_COUNT) throw new Error('GBR4/v2 currently requires one scale');
  if (view.getUint32(48, true) !== DESCRIPTOR_FORMAT_U16) {
    throw new Error('GBR4/v2 descriptor format must be u16');
  }

  const descriptor = byteSection(bytes, view.getUint32(52, true), view.getUint32(56, true), 'descriptor');
  const chartTable = byteSection(bytes, view.getUint32(60, true), view.getUint32(64, true), 'chart table');
  const scaleTable = byteSection(bytes, view.getUint32(68, true), view.getUint32(72, true), 'scale table');
  const frontColor = byteSection(bytes, view.getUint32(76, true), view.getUint32(80, true), 'front color');
  const backColor = byteSection(bytes, view.getUint32(84, true), view.getUint32(88, true), 'back color');
  const categorical = byteSection(bytes, view.getUint32(92, true), view.getUint32(96, true), 'categorical');
  ensureDisjoint([descriptor, chartTable, scaleTable, frontColor, backColor, categorical]);

  const atlasWidth = positiveInt(view, 100, 'atlas width');
  const atlasHeight = positiveInt(view, 104, 'atlas height');
  const codewordTileWidth = positiveInt(view, 108, 'codeword tile width');
  const codewordTileHeight = positiveInt(view, 112, 'codeword tile height');
  const codebookCount = positiveInt(view, 116, 'codebook count');
  if (codebookCount > MAX_DESCRIPTOR_ID + 1) throw new Error('GBR4/v2 codebook exceeds u16 address space');
  if (view.getUint32(120, true) !== MAX_DESCRIPTOR_ID) {
    throw new Error('GBR4/v2 maximum descriptor id is not canonical u16');
  }
  const flags = view.getUint32(124, true);
  if (flags !== REQUIRED_FLAGS) throw new Error('GBR4/v2 feature flags are not the canonical periodic-cap set');
  const expectedTileEdge = checkedProduct('codeword tile edge', blockSampleEdge, blockSampleEdge);
  if (codewordTileWidth !== expectedTileEdge || codewordTileHeight !== expectedTileEdge) {
    throw new Error('GBR4/v2 codeword tile dimensions do not encode a complete 4D brick');
  }
  if (atlasWidth % codewordTileWidth !== 0 || atlasHeight % codewordTileHeight !== 0) {
    throw new Error('GBR4/v2 atlas dimensions are not whole codeword tiles');
  }
  const atlasCapacity = checkedProduct(
    'atlas tile capacity',
    atlasWidth / codewordTileWidth,
    atlasHeight / codewordTileHeight,
  );
  if (atlasCapacity < codebookCount) throw new Error('GBR4/v2 atlas cannot contain the declared codebook');
  const expectedAtlasBytes = checkedProduct('atlas byte count', atlasWidth, atlasHeight, BYTES_PER_ATLAS_TEXEL);
  for (const atlas of [frontColor, backColor, categorical]) {
    if (atlas.length !== expectedAtlasBytes) throw new Error(`GBR4/v2 ${atlas.label} byte count is inconsistent`);
  }

  const blocksPerChart = power4(blocksPerAxis, 'blocks per chart');
  const descriptorCount = checkedProduct('descriptor count', CHART_COUNT, blocksPerChart);
  if (descriptor.length !== checkedProduct('descriptor byte count', descriptorCount, 2)) {
    throw new Error('GBR4/v2 descriptor byte count is inconsistent');
  }
  if (chartTable.length !== CHART_COUNT * CHART_ENTRY_BYTES) {
    throw new Error('GBR4/v2 chart-table byte count is inconsistent');
  }
  if (scaleTable.length !== SCALE_ENTRY_BYTES) throw new Error('GBR4/v2 scale-table byte count is inconsistent');

  const chartView = new DataView(bytes.buffer, bytes.byteOffset + chartTable.offset, chartTable.length);
  const charts: Gbr4Chart[] = [];
  for (let index = 0; index < CHART_COUNT; index++) {
    const offset = index * CHART_ENTRY_BYTES;
    const chartId = chartView.getUint8(offset);
    const inwardYSign = chartView.getInt8(offset + 1);
    const descriptorBase = chartView.getUint32(offset + 4, true);
    const blockCount = chartView.getUint32(offset + 8, true);
    if (
      chartId !== index
      || inwardYSign !== (index === 0 ? -1 : 1)
      || chartView.getUint16(offset + 2, true) !== 0
      || descriptorBase !== index * blocksPerChart
      || blockCount !== blocksPerChart
      || chartView.getUint32(offset + 12, true) !== 0
    ) throw new Error(`GBR4/v2 chart ${index} entry is not canonical`);
    charts.push(Object.freeze({ chartId, inwardYSign, descriptorBase, blockCount }));
  }

  const descriptorSection = sectionBytes(bytes, descriptor);
  const descriptorWords = new Uint16Array(
    descriptorSection.buffer,
    descriptorSection.byteOffset,
    descriptorCount,
  );
  for (const descriptorId of descriptorWords) {
    if (descriptorId >= codebookCount) throw new Error('GBR4/v2 descriptor references an absent codeword');
  }

  const scaleView = new DataView(bytes.buffer, bytes.byteOffset + scaleTable.offset, scaleTable.length);
  const scaleStandoff = positiveFinite(scaleView, 0, 'scale standoff');
  const scalePixelAngle = positiveFinite(scaleView, 4, 'scale pixel angle');
  const scaleDescriptorFirst = scaleView.getUint32(8, true);
  const scaleDescriptorCount = scaleView.getUint32(12, true);
  const scaleCodebookFirst = scaleView.getUint32(16, true);
  const scaleCodebookCount = scaleView.getUint32(20, true);
  const scaleAtlasWidth = scaleView.getUint32(24, true);
  const scaleAtlasHeight = scaleView.getUint32(28, true);
  if (
    scaleDescriptorFirst !== 0
    || scaleDescriptorCount !== descriptorCount
    || scaleCodebookFirst !== 0
    || scaleCodebookCount !== codebookCount
    || scaleAtlasWidth !== atlasWidth
    || scaleAtlasHeight !== atlasHeight
  ) throw new Error('GBR4/v2 scale-table ranges do not match the container');
  requireZero(sectionBytes(bytes, scaleTable), 32, SCALE_ENTRY_BYTES, 'scale-table reserved bytes');

  const tileOriginX = finite(view, 128, 'tile origin X');
  const tileOriginZ = finite(view, 132, 'tile origin Z');
  const tileSizeX = positiveFinite(view, 136, 'tile size X');
  const tileSizeZ = positiveFinite(view, 140, 'tile size Z');
  const carrierBounds = [
    finite(view, 144, 'carrier minimum X'),
    finite(view, 148, 'carrier minimum Y'),
    finite(view, 152, 'carrier minimum Z'),
    finite(view, 156, 'carrier maximum X'),
    finite(view, 160, 'carrier maximum Y'),
    finite(view, 164, 'carrier maximum Z'),
  ] as const;
  if (
    !(carrierBounds[0] < carrierBounds[3])
    || !(carrierBounds[1] < carrierBounds[4])
    || !(carrierBounds[2] < carrierBounds[5])
  ) throw new Error('GBR4/v2 carrier bounds are empty or inverted');
  if (
    carrierBounds[0] !== tileOriginX
    || carrierBounds[2] !== tileOriginZ
    || carrierBounds[3] !== Math.fround(tileOriginX + tileSizeX)
    || carrierBounds[5] !== Math.fround(tileOriginZ + tileSizeZ)
  ) throw new Error('GBR4/v2 carrier XZ bounds do not match the periodic tile');

  const sourceHashBytes = bytes.subarray(168, 200);
  const recipeHashBytes = bytes.subarray(200, 232);
  if (sourceHashBytes.every((value) => value === 0) || recipeHashBytes.every((value) => value === 0)) {
    throw new Error('GBR4/v2 provenance hashes must be populated');
  }
  const pixelAngleRadians = positiveFinite(view, 232, 'pixel angle');
  const standoffMetres = positiveFinite(view, 236, 'camera standoff');
  const quadratureSide = positiveInt(view, 240, 'quadrature side');
  const horizonMetres = positiveFinite(view, 244, 'forward horizon');
  if (horizonMetres !== FORWARD_HORIZON_METRES) throw new Error('GBR4/v2 forward horizon must be 155 metres');
  if (view.getUint32(248, true) !== SLOPE_CHART_RATIONAL_COMPACT) {
    throw new Error('GBR4/v2 slope chart is not rational compact slopes');
  }
  if (view.getUint32(252, true) !== CAP_CONVENTION) throw new Error('GBR4/v2 cap convention is not canonical');
  if (pixelAngleRadians !== scalePixelAngle || standoffMetres !== scaleStandoff) {
    throw new Error('GBR4/v2 header and scale sampling parameters disagree');
  }
  requireZero(bytes, 256, HEADER_BYTES, 'reserved header bytes');

  const descriptorTextureWidth = Math.min(DESCRIPTOR_TEXTURE_MAX_WIDTH, descriptorCount);
  const descriptorTextureHeight = Math.ceil(descriptorCount / descriptorTextureWidth);
  const descriptorTextureWords = new Uint32Array(
    checkedProduct('descriptor texture texels', descriptorTextureWidth, descriptorTextureHeight),
  );
  descriptorTextureWords.set(descriptorWords);
  const frontColorWords = new Uint16Array(
    bytes.buffer,
    bytes.byteOffset + frontColor.offset,
    expectedAtlasBytes / 2,
  );
  const backColorWords = new Uint16Array(
    bytes.buffer,
    bytes.byteOffset + backColor.offset,
    expectedAtlasBytes / 2,
  );
  const categoricalContainerWords = new Uint16Array(
    bytes.buffer,
    bytes.byteOffset + categorical.offset,
    expectedAtlasBytes / 2,
  );
  const categoricalWords = new Uint32Array(categoricalContainerWords);

  const descriptorTexture = configureDescriptorTexture(
    descriptorTextureWords,
    descriptorTextureWidth,
    descriptorTextureHeight,
    profileId,
  );
  const frontColorTexture = configureColorTexture(frontColorWords, atlasWidth, atlasHeight, profileId, 'Front');
  const backColorTexture = configureColorTexture(backColorWords, atlasWidth, atlasHeight, profileId, 'Back');
  const categoricalTexture = configureCategoricalTexture(categoricalWords, atlasWidth, atlasHeight, profileId);
  const scale: Gbr4Scale = Object.freeze({
    standoffMetres: scaleStandoff,
    pixelAngleRadians: scalePixelAngle,
    descriptorFirst: scaleDescriptorFirst,
    descriptorCount: scaleDescriptorCount,
    codebookFirst: scaleCodebookFirst,
    codebookCount: scaleCodebookCount,
    atlasWidth: scaleAtlasWidth,
    atlasHeight: scaleAtlasHeight,
  });

  return Object.freeze({
    profileId,
    descriptorTexture,
    descriptorTextureWidth,
    descriptorTextureHeight,
    frontColorTexture,
    backColorTexture,
    categoricalTexture,
    charts: Object.freeze(charts),
    scale,
    cellResolution,
    blockCellEdge,
    blockSampleEdge,
    blocksPerAxis,
    atlasWidth,
    atlasHeight,
    codewordTileWidth,
    codewordTileHeight,
    codebookCount,
    flags,
    tileOriginX,
    tileOriginZ,
    tileSizeX,
    tileSizeZ,
    carrierBounds: Object.freeze(carrierBounds),
    horizonMetres,
    pixelAngleRadians,
    standoffMetres,
    quadratureSide,
    sourceSha256: hex(sourceHashBytes),
    recipeSha256: hex(recipeHashBytes),
    // Upload accounting, not compact-container accounting: descriptors expand
    // u16 -> R32Uint and categorical RGBA16Uint expands -> RGBA32Uint because
    // Three's WebGPU sampled-integer texture table exposes those 32-bit forms.
    residentBytes:
      descriptorTextureWords.byteLength
      + frontColorWords.byteLength
      + backColorWords.byteLength
      + categoricalWords.byteLength,
    containerBytes: bytes.byteLength,
  });
}

/** Fetch and strictly parse an exclusive GBR4/v2 profile. A server supporting
 * ranges avoids downloading an incompatible container before the fixed header
 * has been inspected; a 200 response is reused as the complete asset. */
export async function loadGbr4Profile(
  url: URL | string,
  expectedProfileId: number,
): Promise<LoadedGbr4Profile> {
  const headerResponse = await fetch(url, { headers: { Range: `bytes=0-${HEADER_BYTES - 1}` } });
  if (!headerResponse.ok) {
    throw new Error(`ground-cover GBR4 header fetch failed (${headerResponse.status} ${headerResponse.statusText})`);
  }
  const firstBytes = new Uint8Array(await headerResponse.arrayBuffer());
  if (firstBytes.byteLength < HEADER_BYTES) throw new Error('ground-cover GBR4 header is truncated');
  const bytes = headerResponse.status === 206
    ? await (async (): Promise<Uint8Array> => {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`ground-cover GBR4 fetch failed (${response.status} ${response.statusText})`);
        }
        return new Uint8Array(await response.arrayBuffer());
      })()
    : firstBytes;
  const profile = parseGbr4Profile(bytes);
  if (profile.profileId !== expectedProfileId) {
    throw new Error(`GBR4/v2 profile id ${profile.profileId} != expected ${expectedProfileId}`);
  }
  return profile;
}
