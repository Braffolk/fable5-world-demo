/**
 * LAC1 wire codec — pure transforms, node-testable, zero three.js/DOM imports.
 * Ground truth: asset-gen/src/assetgen/cook/chunkio.py (56-byte LE header),
 * cook/encode.py (enc1 quant16+2D-delta, enc2 u8 planes, enc3 record SoA) and
 * assetgen/manifest.py (21-byte index record `<BiiIQ`, content-addressed URLs).
 *
 * The (de)compression boundary is OUTSIDE this file by design: every function here
 * takes/returns the raw (inflated) payload bytes. Callers pair it with node:zlib
 * (tools/probe-lac1.ts) or DecompressionStream('deflate') (Lac1Decode.worker.ts) —
 * the payload is raw-zlib, self-contained, CRC'd over the COMPRESSED bytes.
 * All typed-array views assume little-endian hosts (every target platform is).
 */
import type { ChunkPayload, ChunkRef, RecordDtype } from './WorldSource';

// --- header (chunkio.py FMT = "<4sBBBBiiHxxIddffII", 56 bytes) -------------------------

export const LAC1_HEADER_SIZE = 56;

export const LAC1_LAYER_IDS = {
  height: 0,
  biome: 1,
  water: 2,
  trees: 3,
  soil: 4,
  understory: 5,
  debris: 6,
  boulders: 7,
  canopy: 8,
  watercover: 9,
  geology: 10,
} as const;

export interface Lac1Header {
  layer: number;
  lod: number;
  enc: number; // 1 = quant16+2D-delta, 2 = u8 planes, 3 = record SoA
  flags: number;
  cx: number;
  cz: number;
  res: number; // texels per side incl. apron (rasters) or 0 (record layers)
  count: number; // record count (record layers) or 0
  originE: number; // float64 absolute L-EST97 QA fields — position chunks from
  originN: number; // (cx, cz) + the manifest anchor, NEVER from these
  qoffset: number;
  qscale: number; // enc1 dequant step; enc3 x/z dequant step (footprint/65535)
  payloadLen: number;
  payloadCrc: number; // crc32 of the COMPRESSED payload
}

export function parseLac1Header(bytes: Uint8Array): Lac1Header {
  if (bytes.length < LAC1_HEADER_SIZE) throw new Error(`LAC1: ${bytes.length} B < header size`);
  if (bytes[0] !== 0x4c || bytes[1] !== 0x41 || bytes[2] !== 0x43 || bytes[3] !== 0x31) {
    throw new Error('LAC1: bad magic');
  }
  const v = new DataView(bytes.buffer, bytes.byteOffset, LAC1_HEADER_SIZE);
  return {
    layer: v.getUint8(4),
    lod: v.getUint8(5),
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
}

export function serializeLac1Header(h: Lac1Header): Uint8Array {
  const bytes = new Uint8Array(LAC1_HEADER_SIZE);
  const v = new DataView(bytes.buffer);
  bytes[0] = 0x4c;
  bytes[1] = 0x41;
  bytes[2] = 0x43;
  bytes[3] = 0x31;
  v.setUint8(4, h.layer);
  v.setUint8(5, h.lod);
  v.setUint8(6, h.enc);
  v.setUint8(7, h.flags);
  v.setInt32(8, h.cx, true);
  v.setInt32(12, h.cz, true);
  v.setUint16(16, h.res, true);
  v.setUint32(20, h.count, true);
  v.setFloat64(24, h.originE, true);
  v.setFloat64(32, h.originN, true);
  v.setFloat32(40, h.qoffset, true);
  v.setFloat32(44, h.qscale, true);
  v.setUint32(48, h.payloadLen, true);
  v.setUint32(52, h.payloadCrc, true);
  return bytes;
}

/** Header + compressed payload → one complete chunk file (fills payloadLen/Crc). */
export function assembleChunk(h: Omit<Lac1Header, 'payloadLen' | 'payloadCrc'>, compressed: Uint8Array): Uint8Array {
  const header = serializeLac1Header({ ...h, payloadLen: compressed.length, payloadCrc: crc32(compressed) });
  const out = new Uint8Array(LAC1_HEADER_SIZE + compressed.length);
  out.set(header, 0);
  out.set(compressed, LAC1_HEADER_SIZE);
  return out;
}

// --- crc32 (zlib polynomial, matches python zlib.crc32) --------------------------------

const CRC_TABLE = ((): Uint32Array => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

export function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = (CRC_TABLE[(c ^ (bytes[i] as number)) & 0xff] as number) ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// --- layer index (manifest.py INDEX_RECORD = "<BiiIQ", 21 bytes) ------------------------

export const LAC1_INDEX_RECORD_SIZE = 21;

export function parseLayerIndex(bytes: Uint8Array): ChunkRef[] {
  if (bytes.length % LAC1_INDEX_RECORD_SIZE !== 0) {
    throw new Error(`LAC1 index: ${bytes.length} B not a multiple of ${LAC1_INDEX_RECORD_SIZE}`);
  }
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.length);
  const refs: ChunkRef[] = [];
  for (let off = 0; off < bytes.length; off += LAC1_INDEX_RECORD_SIZE) {
    refs.push({
      lod: v.getUint8(off),
      cx: v.getInt32(off + 1, true),
      cz: v.getInt32(off + 5, true),
      size: v.getUint32(off + 9, true),
      hash64: v.getBigUint64(off + 13, true),
    });
  }
  return refs;
}

/** hash64 = int.from_bytes(sha256[:8], "big") ⇒ the filename's 8-hex-char tag is
 *  sha256[:4].hex() = the TOP 32 bits of hash64. */
export function chunkHash8(hash64: bigint): string {
  return ((hash64 >> 32n) & 0xffffffffn).toString(16).padStart(8, '0');
}

/** Path of a chunk inside the release tree: c/<layer>/<lod>/<cx>_<cz>.<hash8>.bin */
export function chunkPath(layer: string, ref: ChunkRef): string {
  return `c/${layer}/${ref.lod}/${ref.cx}_${ref.cz}.${chunkHash8(ref.hash64)}.bin`;
}

/** (lod, cx, cz) → one Map-friendly number. lod ∈ [0, 63], |cx|,|cz| < 2^20 (Estonia
 *  at LOD0 is < 400 chunks across); max value ≈ 2^48 < 2^53, exact in a JS number. */
export function packChunkKey(lod: number, cx: number, cz: number): number {
  return lod * 2 ** 42 + (cx + (1 << 20)) * 2 ** 21 + (cz + (1 << 20));
}

// --- enc 1: quantized u16 raster with wrapping 2D delta (encode.py) ---------------------

/** In-place inverse of delta2d: column 0 cumsums down, then each row cumsums right
 *  (all mod 2^16). Matches encode.py undelta2d. */
export function undelta2d(d: Uint16Array, res: number): Uint16Array {
  for (let j = 1; j < res; j++) d[j * res] = ((d[j * res] as number) + (d[(j - 1) * res] as number)) & 0xffff;
  for (let j = 0; j < res; j++) {
    const row = j * res;
    let acc = d[row] as number;
    for (let i = 1; i < res; i++) {
      acc = (acc + (d[row + i] as number)) & 0xffff;
      d[row + i] = acc;
    }
  }
  return d;
}

/** Wrapping u16 delta: each texel predicts from its left neighbor; the first column
 *  predicts from the texel above; (0,0) stays raw. Matches encode.py delta2d. */
export function delta2d(q: Uint16Array, res: number): Uint16Array {
  const d = new Uint16Array(q.length);
  for (let j = 0; j < res; j++) {
    const row = j * res;
    d[row] = j === 0 ? (q[0] as number) : ((q[row] as number) - (q[row - res] as number)) & 0xffff;
    for (let i = 1; i < res; i++) d[row + i] = ((q[row + i] as number) - (q[row + i - 1] as number)) & 0xffff;
  }
  return d;
}

/** f32 → u16 quantization. qoffset = floor(min(finite) - 1); NaN → qoffset (q = 0);
 *  ±inf clip. Values land in [qoffset, qoffset + 65535·qscale]. Matches encode_quant16. */
export function quant16(values: Float32Array, qscale: number): { q: Uint16Array; qoffset: number } {
  let min = Infinity;
  for (let i = 0; i < values.length; i++) {
    const h = values[i] as number;
    if (Number.isFinite(h) && h < min) min = h;
  }
  const qoffset = Number.isFinite(min) ? Math.floor(min - 1.0) : 0.0;
  const q = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const h = values[i] as number;
    const s = Math.round(((Number.isNaN(h) ? qoffset : h) - qoffset) / qscale);
    q[i] = s < 0 ? 0 : s > 65535 ? 65535 : s;
  }
  return { q, qoffset };
}

/** Raw (inflated) enc1 payload → dequantized f32 raster. CONSUMES raw (un-deltas the
 *  u16 view in place). dryZeroToNaN: the water layer reserves q = 0 for DRY texels
 *  (manifest semantics) — decode those to NaN so consumers can't mistake them for a
 *  real surface at qoffset. */
export function decodeQuant16(raw: Uint8Array, res: number, qoffset: number, qscale: number, dryZeroToNaN = false): Float32Array {
  const n = res * res;
  if (raw.length !== n * 2) throw new Error(`enc1: ${raw.length} B != ${res}²·2`);
  const q = undelta2d(u16View(raw, n), res);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const s = q[i] as number;
    out[i] = dryZeroToNaN && s === 0 ? NaN : s * qscale + qoffset;
  }
  return out;
}

/** f32 raster → raw (uncompressed) enc1 payload bytes + the chosen qoffset. */
export function encodeQuant16(values: Float32Array, res: number, qscale: number): { raw: Uint8Array; qoffset: number } {
  if (values.length !== res * res) throw new Error(`enc1: ${values.length} values != ${res}²`);
  const { q, qoffset } = quant16(values, qscale);
  const d = delta2d(q, res);
  return { raw: new Uint8Array(d.buffer, 0, d.length * 2), qoffset };
}

// --- enc 2: stacked u8 planes ------------------------------------------------------------

/** Raw payload → per-plane views (share the raw buffer; plane count derived from size). */
export function decodePlanes(raw: Uint8Array, res: number): Uint8Array[] {
  const per = res * res;
  const n = raw.length / per;
  if (!Number.isInteger(n) || n < 1) throw new Error(`enc2: ${raw.length} B not a multiple of ${res}²`);
  const planes: Uint8Array[] = [];
  for (let i = 0; i < n; i++) planes.push(raw.subarray(i * per, (i + 1) * per));
  return planes;
}

export function encodePlanes(planes: Uint8Array[], res: number): Uint8Array {
  const per = res * res;
  const out = new Uint8Array(per * planes.length);
  for (let i = 0; i < planes.length; i++) {
    const p = planes[i] as Uint8Array;
    if (p.length !== per) throw new Error(`enc2: plane ${i} is ${p.length} B != ${res}²`);
    out.set(p, i * per);
  }
  return out;
}

// --- enc 3: record SoA --------------------------------------------------------------------

export type RecordColumn = Uint8Array | Uint16Array | Uint32Array | Float32Array;

const DTYPE_BYTES: Record<RecordDtype, number> = { u8: 1, u16: 2, u32: 4, f32: 4 };

function columnView(raw: Uint8Array, off: number, count: number, dtype: RecordDtype): RecordColumn {
  // typed views need dtype alignment; the SoA concat rarely provides it → aligned copy.
  const bytes = raw.slice(off, off + count * DTYPE_BYTES[dtype]);
  switch (dtype) {
    case 'u8':
      return bytes;
    case 'u16':
      return new Uint16Array(bytes.buffer, 0, count);
    case 'u32':
      return new Uint32Array(bytes.buffer, 0, count);
    case 'f32':
      return new Float32Array(bytes.buffer, 0, count);
  }
}

export function decodeRecordColumns(raw: Uint8Array, count: number, dtypes: readonly RecordDtype[]): RecordColumn[] {
  const out: RecordColumn[] = [];
  let off = 0;
  for (const dt of dtypes) {
    out.push(columnView(raw, off, count, dt));
    off += count * DTYPE_BYTES[dt];
  }
  if (off !== raw.length) throw new Error(`enc3: ${raw.length} B != ${off} (count ${count} × ${dtypes.join(',')})`);
  return out;
}

export function encodeRecordColumns(cols: readonly RecordColumn[]): Uint8Array {
  const n = cols.length > 0 ? (cols[0] as RecordColumn).length : 0;
  let total = 0;
  for (const c of cols) {
    if (c.length !== n) throw new Error(`enc3: column length ${c.length} != ${n}`);
    total += c.byteLength;
  }
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of cols) {
    out.set(new Uint8Array(c.buffer, c.byteOffset, c.byteLength), off);
    off += c.byteLength;
  }
  return out;
}

// --- payload assembly: raw bytes → decoded ChunkPayload (SPEC §2) --------------------------

/** trees.scale u8 = crownHeight/refHeight·64 → f32 multiplier vs species ref height. */
export const TREE_SCALE_Q = 1 / 64;
/** boulders.size u8 = height_m·40 → meters = size/40 (0.2..6.4 m). Raw 40/80 usually
 *  means the cook's unmeasured default (1 m single / 2 m pile) — ETAK korgus is absent
 *  for ~97% of surveyed features. */
export const BOULDER_SIZE_STEP_M = 1 / 40;

/** Per-layer wire schema from the manifest (enc2 plane count, enc3 column list). */
export interface Lac1LayerSchema {
  planes?: number;
  columns?: readonly (readonly [string, RecordDtype])[];
}

/** Decode one inflated payload into the spec §2 decoded-f32 ChunkPayload. Record layers
 *  map manifest columns onto the fixed cols shape: x/z u16 dequantize by header qscale
 *  (= footprint/65535 → chunk-local meters); trees scale = u8/64; boulders kind→species,
 *  size→scale (meters = size/40). */
export function decodeChunkPayload(h: Lac1Header, raw: Uint8Array, schema: Lac1LayerSchema = {}): ChunkPayload {
  switch (h.enc) {
    case 1:
      return {
        kind: 'height',
        res: h.res,
        heights: decodeQuant16(raw, h.res, h.qoffset, h.qscale, h.layer === LAC1_LAYER_IDS.water),
      };
    case 2: {
      const planes = decodePlanes(raw, h.res);
      if (schema.planes !== undefined && planes.length !== schema.planes) {
        throw new Error(`enc2: ${planes.length} planes != manifest's ${schema.planes}`);
      }
      return { kind: 'planes', res: h.res, planes };
    }
    case 3: {
      const columns = schema.columns;
      if (!columns) throw new Error('enc3: record layer needs manifest columns');
      const arrays = decodeRecordColumns(
        raw,
        h.count,
        columns.map(([, dt]) => dt),
      );
      const byName = new Map<string, RecordColumn>(columns.map(([name], i) => [name, arrays[i] as RecordColumn]));
      const col = (name: string): RecordColumn => {
        const c = byName.get(name);
        if (!c) throw new Error(`enc3: missing column '${name}' (have ${columns.map(([n]) => n).join(',')})`);
        return c;
      };
      const dequantXZ = (c: RecordColumn): Float32Array => {
        const out = new Float32Array(h.count);
        for (let i = 0; i < h.count; i++) out[i] = (c[i] as number) * h.qscale;
        return out;
      };
      const scaleSrc = byName.has('scale') ? col('scale') : col('size');
      const scaleStep = byName.has('scale') ? TREE_SCALE_Q : BOULDER_SIZE_STEP_M;
      const scale = new Float32Array(h.count);
      for (let i = 0; i < h.count; i++) scale[i] = (scaleSrc[i] as number) * scaleStep;
      return {
        kind: 'records',
        count: h.count,
        cols: {
          x: dequantXZ(col('x')),
          z: dequantXZ(col('z')),
          species: (byName.has('species') ? col('species') : col('kind')) as Uint8Array,
          scale,
          variant: col('variant') as Uint8Array,
        },
      };
    }
    default:
      throw new Error(`LAC1: unknown enc ${h.enc}`);
  }
}

// --- helpers -------------------------------------------------------------------------------

function u16View(bytes: Uint8Array, count: number): Uint16Array {
  if (bytes.byteOffset % 2 === 0) return new Uint16Array(bytes.buffer, bytes.byteOffset, count);
  const copy = bytes.slice();
  return new Uint16Array(copy.buffer, 0, count);
}
