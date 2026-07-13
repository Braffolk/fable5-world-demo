/** LAC2 signed-LOD wire primitives. LAC1 remains byte-for-byte unchanged. */
import { crc32, parseLac1Header } from './Lac1';
import type { LacContainer, ChunkKey, ChunkRef } from './WorldSource';
import type { Lac1Header } from './Lac1';

export const LAC2_HEADER_SIZE = 56;
export const LAC2_INDEX_RECORD_SIZE = 21;
const KEY_BIAS = 1 << 20;
const KEY_AXIS = 2 ** 21;
const KEY_LOD = 2 ** 42;

function validateKey(lod: number, cx: number, cz: number): void {
  if (!Number.isInteger(lod) || lod < -8 || lod > 55) throw new Error(`LAC2: lod ${lod} outside -8..55`);
  if (!Number.isInteger(cx) || !Number.isInteger(cz) || Math.abs(cx) >= KEY_BIAS || Math.abs(cz) >= KEY_BIAS) {
    throw new Error(`LAC2: coordinate (${cx},${cz}) outside signed key range`);
  }
}

export function parseLac2Header(bytes: Uint8Array): Lac1Header {
  if (bytes.length < LAC2_HEADER_SIZE) throw new Error(`LAC2: ${bytes.length} B < header size`);
  if (bytes[0] !== 0x4c || bytes[1] !== 0x41 || bytes[2] !== 0x43 || bytes[3] !== 0x32) {
    throw new Error('LAC2: bad magic');
  }
  const v = new DataView(bytes.buffer, bytes.byteOffset, LAC2_HEADER_SIZE);
  const header: Lac1Header = {
    layer: v.getUint8(4),
    lod: v.getInt8(5),
    enc: v.getUint8(6),
    flags: v.getUint8(7),
    cx: v.getInt32(8, true),
    cz: v.getInt32(12, true),
    res: v.getUint16(16, true),
    count: v.getUint32(20, true),
    originE: v.getFloat64(24, true),
    originN: v.getFloat64(32, true),
    qoffset: v.getFloat32(40, true),
    qscale: v.getFloat32(44, true),
    payloadLen: v.getUint32(48, true),
    payloadCrc: v.getUint32(52, true),
  };
  validateKey(header.lod, header.cx, header.cz);
  if (header.layer > 9) throw new Error(`LAC2: unknown layer id ${header.layer}`);
  return header;
}

export function serializeLac2Header(header: Lac1Header): Uint8Array {
  validateKey(header.lod, header.cx, header.cz);
  const bytes = new Uint8Array(LAC2_HEADER_SIZE);
  const v = new DataView(bytes.buffer);
  bytes.set([0x4c, 0x41, 0x43, 0x32]);
  v.setUint8(4, header.layer);
  v.setInt8(5, header.lod);
  v.setUint8(6, header.enc);
  v.setUint8(7, header.flags);
  v.setInt32(8, header.cx, true);
  v.setInt32(12, header.cz, true);
  v.setUint16(16, header.res, true);
  v.setUint32(20, header.count, true);
  v.setFloat64(24, header.originE, true);
  v.setFloat64(32, header.originN, true);
  v.setFloat32(40, header.qoffset, true);
  v.setFloat32(44, header.qscale, true);
  v.setUint32(48, header.payloadLen, true);
  v.setUint32(52, header.payloadCrc, true);
  return bytes;
}

export function assembleLac2Chunk(
  header: Omit<Lac1Header, 'payloadLen' | 'payloadCrc'>,
  compressed: Uint8Array,
): Uint8Array {
  const encoded = serializeLac2Header({
    ...header,
    payloadLen: compressed.length,
    payloadCrc: crc32(compressed),
  });
  const out = new Uint8Array(encoded.length + compressed.length);
  out.set(encoded);
  out.set(compressed, encoded.length);
  return out;
}

export function parseLacHeader(
  bytes: Uint8Array,
  allowedContainers: readonly LacContainer[],
): { container: LacContainer; header: Lac1Header } {
  if (bytes.length < 4) throw new Error('LAC: truncated magic');
  const container = bytes[3] === 0x31 ? 'LAC1' : bytes[3] === 0x32 ? 'LAC2' : null;
  if (bytes[0] !== 0x4c || bytes[1] !== 0x41 || bytes[2] !== 0x43 || container === null) {
    throw new Error('LAC: unknown magic');
  }
  if (!allowedContainers.includes(container)) throw new Error(`LAC: ${container} is not allowed by manifest`);
  return {
    container,
    header: container === 'LAC1' ? parseLac1Header(bytes) : parseLac2Header(bytes),
  };
}

export function parseLayerIndexV2(bytes: Uint8Array): ChunkRef[] {
  if (bytes.length % LAC2_INDEX_RECORD_SIZE !== 0) {
    throw new Error(`LAC2 index: ${bytes.length} B not a multiple of ${LAC2_INDEX_RECORD_SIZE}`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.length);
  const refs: ChunkRef[] = [];
  let previous: readonly [number, number, number] | null = null;
  for (let off = 0; off < bytes.length; off += LAC2_INDEX_RECORD_SIZE) {
    const ref: ChunkRef = {
      lod: view.getInt8(off),
      cx: view.getInt32(off + 1, true),
      cz: view.getInt32(off + 5, true),
      size: view.getUint32(off + 9, true),
      hash64: view.getBigUint64(off + 13, true),
    };
    validateKey(ref.lod, ref.cx, ref.cz);
    const key = [ref.lod, ref.cx, ref.cz] as const;
    if (previous && compareKey(previous, key) >= 0) throw new Error('LAC2 index: keys are duplicate or unsorted');
    previous = key;
    refs.push(ref);
  }
  return refs;
}

function compareKey(a: readonly number[], b: readonly number[]): number {
  return a[0] !== b[0] ? (a[0] as number) - (b[0] as number)
    : a[1] !== b[1] ? (a[1] as number) - (b[1] as number)
      : (a[2] as number) - (b[2] as number);
}

export function packChunkKeyV2(lod: number, cx: number, cz: number): number {
  validateKey(lod, cx, cz);
  return (lod + 8) * KEY_LOD + (cx + KEY_BIAS) * KEY_AXIS + (cz + KEY_BIAS);
}

export function unpackChunkKeyV2(packed: number): ChunkKey {
  if (!Number.isSafeInteger(packed) || packed < 0) throw new Error(`LAC2: invalid packed key ${packed}`);
  const biasedLod = Math.floor(packed / KEY_LOD);
  const rest = packed - biasedLod * KEY_LOD;
  const cx = Math.floor(rest / KEY_AXIS) - KEY_BIAS;
  const cz = rest % KEY_AXIS - KEY_BIAS;
  const lod = biasedLod - 8;
  validateKey(lod, cx, cz);
  return { lod, cx, cz };
}
