/** Container-neutral validation and decode used inside Lac1Decode.worker. */
import { crc32, decodeChunkPayload, LAC1_HEADER_SIZE, type Lac1LayerSchema } from './Lac1';
import { parseLacHeader } from './Lac2';
import type { ChunkPayload, LacContainer, WireCodec } from './WorldSource';

export interface LacDecodeSpec {
  layerId: number;
  lod: number;
  cx: number;
  cz: number;
  expectedEnc: number;
  allowedContainers: readonly LacContainer[];
  codec: WireCodec;
  expectedFileSize: number;
  expectedRes: number;
  expectedOriginE: number;
  expectedOriginN: number;
  schema: Lac1LayerSchema;
}

async function inflateExact(bytes: Uint8Array, expectedBytes: number): Promise<Uint8Array> {
  const MAX_INFLATED_BYTES = 256 * 1024 * 1024;
  if (!Number.isSafeInteger(expectedBytes) || expectedBytes < 0 || expectedBytes > MAX_INFLATED_BYTES) {
    throw new Error(`invalid inflated payload size ${expectedBytes}`);
  }
  const stream = new Blob([Uint8Array.from(bytes)]).stream().pipeThrough(new DecompressionStream('deflate'));
  const reader = stream.getReader();
  const out = new Uint8Array(expectedBytes);
  let offset = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (offset + value.length > expectedBytes) {
      await reader.cancel('inflated payload exceeds declared schema');
      throw new Error(`inflated payload exceeds ${expectedBytes} B`);
    }
    out.set(value, offset);
    offset += value.length;
  }
  if (offset !== expectedBytes) throw new Error(`inflated payload ${offset} B != expected ${expectedBytes} B`);
  return out;
}

function expectedInflatedBytes(
  header: ReturnType<typeof parseLacHeader>['header'],
  schema: Lac1LayerSchema,
): number {
  if (header.enc === 1) return header.res * header.res * 2;
  if (header.enc === 2) {
    if (!Number.isInteger(schema.planes) || (schema.planes as number) < 1) throw new Error('enc2 schema lacks planes');
    return header.res * header.res * (schema.planes as number);
  }
  if (header.enc === 3) {
    if (!schema.columns?.length) throw new Error('enc3 schema lacks columns');
    const widths: Record<string, number> = { u8: 1, u16: 2, u32: 4, f32: 4 };
    let rowBytes = 0;
    for (const [, dtype] of schema.columns) {
      const width = widths[dtype];
      if (width === undefined) throw new Error(`unknown record dtype ${dtype}`);
      rowBytes += width;
    }
    return header.count * rowBytes;
  }
  throw new Error(`unsupported encoding ${header.enc}`);
}

export async function decodeLacBytes(blob: Uint8Array, spec: LacDecodeSpec): Promise<ChunkPayload> {
  if (blob.length !== spec.expectedFileSize) throw new Error(`file size ${blob.length} B != index ${spec.expectedFileSize} B`);
  if (spec.codec !== 'deflate') throw new Error(`unsupported chunk codec ${spec.codec}`);
  const { header } = parseLacHeader(blob, spec.allowedContainers);
  if (
    header.layer !== spec.layerId
    || header.lod !== spec.lod
    || header.cx !== spec.cx
    || header.cz !== spec.cz
    || header.enc !== spec.expectedEnc
  ) {
    throw new Error(
      `header (${header.layer},${header.lod},${header.cx},${header.cz},enc${header.enc}) `
      + `!= expected (${spec.layerId},${spec.lod},${spec.cx},${spec.cz},enc${spec.expectedEnc})`,
    );
  }
  if ((header.enc === 1 || header.enc === 2) && (header.res === 0 || header.count !== 0)) {
    throw new Error('raster header shape is invalid');
  }
  if (header.enc === 3 && header.res !== 0) throw new Error('record header shape is invalid');
  if (header.res !== spec.expectedRes) throw new Error(`header res ${header.res} != expected ${spec.expectedRes}`);
  if (header.originE !== spec.expectedOriginE || header.originN !== spec.expectedOriginN) {
    throw new Error(
      `header origin (${header.originE},${header.originN}) != expected (${spec.expectedOriginE},${spec.expectedOriginN})`,
    );
  }
  if (blob.length !== LAC1_HEADER_SIZE + header.payloadLen) {
    throw new Error(`file length ${blob.length} B != 56+${header.payloadLen}`);
  }
  const compressed = blob.subarray(LAC1_HEADER_SIZE, LAC1_HEADER_SIZE + header.payloadLen);
  if (crc32(compressed) !== header.payloadCrc) throw new Error('payload crc mismatch');
  const raw = await inflateExact(compressed, expectedInflatedBytes(header, spec.schema));
  return decodeChunkPayload(header, raw, spec.schema);
}
