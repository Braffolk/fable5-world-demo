import assert from 'node:assert/strict';
import test from 'node:test';
import { deflateSync } from 'node:zlib';

import {
  assembleChunk,
  encodePlanes,
  encodeQuant16,
  LAC1_LAYER_IDS,
  type Lac1Header,
} from './Lac1';
import { decodeLacBytes, type LacDecodeSpec } from './LacDecode';
import { assembleLac2Chunk } from './Lac2';

const originE = 368640;
const originN = 6635520;

function compressed(raw: Uint8Array): Uint8Array<ArrayBuffer> {
  return Uint8Array.from(deflateSync(raw));
}

function baseHeader(overrides: Partial<Omit<Lac1Header, 'payloadLen' | 'payloadCrc'>> = {}) {
  return {
    layer: LAC1_LAYER_IDS.height,
    lod: -2,
    enc: 1,
    flags: 0,
    cx: 0,
    cz: 0,
    res: 3,
    count: 0,
    originE,
    originN,
    qoffset: 0,
    qscale: 0.002,
    ...overrides,
  };
}

function spec(file: Uint8Array, overrides: Partial<LacDecodeSpec> = {}): LacDecodeSpec {
  return {
    layerId: LAC1_LAYER_IDS.height,
    lod: -2,
    cx: 0,
    cz: 0,
    expectedEnc: 1,
    allowedContainers: ['LAC1', 'LAC2'],
    codec: 'deflate',
    expectedFileSize: file.length,
    expectedRes: 3,
    expectedOriginE: originE,
    expectedOriginN: originN,
    schema: {},
    ...overrides,
  };
}

test('production decoder handles new LAC2 height and inherited LAC1 planes', async () => {
  const values = Float32Array.from([10, 10.01, 10.02, 10.03, 10.04, 10.05, 10.06, 10.07, 10.08]);
  const encoded = encodeQuant16(values, 3, 0.002);
  const fine = assembleLac2Chunk(baseHeader({ qoffset: encoded.qoffset }), compressed(encoded.raw));
  const finePayload = await decodeLacBytes(fine, spec(fine));
  assert.equal(finePayload.kind, 'height');
  if (finePayload.kind === 'height') {
    assert.equal(finePayload.res, 3);
    for (let i = 0; i < values.length; i++) assert.ok(Math.abs((finePayload.heights[i] as number) - (values[i] as number)) <= 0.0011);
  }

  const planes = [Uint8Array.from({ length: 9 }, (_, i) => i), Uint8Array.from({ length: 9 }, (_, i) => 100 + i)];
  const inherited = assembleChunk(
    baseHeader({ layer: LAC1_LAYER_IDS.biome, lod: 0, enc: 2, qscale: 0 }),
    compressed(encodePlanes(planes, 3)),
  );
  const inheritedPayload = await decodeLacBytes(inherited, spec(inherited, {
    layerId: LAC1_LAYER_IDS.biome,
    lod: 0,
    expectedEnc: 2,
    schema: { planes: 2 },
  }));
  assert.equal(inheritedPayload.kind, 'planes');
  if (inheritedPayload.kind === 'planes') {
    assert.deepEqual(inheritedPayload.planes[0], planes[0]);
    assert.deepEqual(inheritedPayload.planes[1], planes[1]);
  }
});

test('production decoder rejects disallowed magic, bad geometry, and trailing bytes', async () => {
  const encoded = encodeQuant16(Float32Array.from({ length: 9 }, (_, i) => i), 3, 0.002);
  const file = assembleLac2Chunk(baseHeader({ qoffset: encoded.qoffset }), compressed(encoded.raw));
  await assert.rejects(() => decodeLacBytes(file, spec(file, { allowedContainers: ['LAC1'] })), /not allowed/);
  await assert.rejects(() => decodeLacBytes(file, spec(file, { expectedRes: 2049 })), /header res/);
  await assert.rejects(() => decodeLacBytes(file, spec(file, { expectedOriginE: originE + 1 })), /header origin/);

  const trailing = new Uint8Array(file.length + 1);
  trailing.set(file);
  await assert.rejects(() => decodeLacBytes(trailing, spec(trailing)), /file length/);
});
