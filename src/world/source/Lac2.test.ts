import assert from 'node:assert/strict';
import test from 'node:test';

import { packChunkKey, parseLac1Header } from './Lac1';
import {
  LAC2_HEADER_SIZE,
  packChunkKeyV2,
  parseLac2Header,
  parseLacHeader,
  parseLayerIndexV2,
  serializeLac2Header,
  unpackChunkKeyV2,
} from './Lac2';

function header(lod: number, cx = 0, cz = 0): Uint8Array {
  const bytes = new Uint8Array(LAC2_HEADER_SIZE);
  bytes.set([0x4c, 0x41, 0x43, 0x32]);
  const v = new DataView(bytes.buffer);
  v.setUint8(4, 0);
  v.setInt8(5, lod);
  v.setUint8(6, 1);
  v.setInt32(8, cx, true);
  v.setInt32(12, cz, true);
  v.setUint16(16, 2049, true);
  v.setFloat64(24, 368640, true);
  v.setFloat64(32, 6635520, true);
  v.setFloat32(44, 0.002, true);
  return bytes;
}

function indexRecord(lod: number, cx: number, cz: number): Uint8Array {
  const bytes = new Uint8Array(21);
  const v = new DataView(bytes.buffer);
  v.setInt8(0, lod);
  v.setInt32(1, cx, true);
  v.setInt32(5, cz, true);
  v.setUint32(9, 123, true);
  v.setBigUint64(13, 0x0102030405060708n, true);
  return bytes;
}

test('LAC2 parses signed physical LOD without changing the 56-byte layout', () => {
  const bytes = header(-2, -7, 11);
  const parsed = parseLac2Header(bytes);
  assert.equal(parsed.lod, -2);
  assert.equal(parsed.cx, -7);
  assert.equal(parsed.cz, 11);
  assert.equal(parsed.res, 2049);
  assert.deepEqual(serializeLac2Header(parsed), bytes);
  assert.deepEqual(parseLacHeader(bytes, ['LAC1', 'LAC2']), { container: 'LAC2', header: parsed });
  assert.throws(() => parseLacHeader(bytes, ['LAC1']), /not allowed/);
});

test('LAC1 parser and packed-key behavior remain unchanged', () => {
  const bytes = header(0, -2, 3);
  bytes[3] = 0x31;
  assert.equal(parseLac1Header(bytes).lod, 0);
  assert.equal(packChunkKey(4, -2, 3), 4 * 2 ** 42 + (2 ** 20 - 2) * 2 ** 21 + (2 ** 20 + 3));
  assert.deepEqual(parseLacHeader(bytes, ['LAC1']), { container: 'LAC1', header: parseLac1Header(bytes) });
});

test('format-2 key packing round-trips all supported height LODs and coordinate extrema', () => {
  const seen = new Set<number>();
  for (let lod = -2; lod <= 4; lod++) {
    for (const cx of [-(2 ** 20) + 1, -1, 0, 1, 2 ** 20 - 1]) {
      for (const cz of [-(2 ** 20) + 1, -1, 0, 1, 2 ** 20 - 1]) {
        const packed = packChunkKeyV2(lod, cx, cz);
        assert.deepEqual(unpackChunkKeyV2(packed), { lod, cx, cz });
        assert.equal(seen.has(packed), false);
        seen.add(packed);
      }
    }
  }
});

test('LAC2 rejects invalid key bounds and LAC1 magic', () => {
  assert.throws(() => packChunkKeyV2(-9, 0, 0), /outside/);
  assert.throws(() => packChunkKeyV2(0, 2 ** 20, 0), /outside/);
  const bytes = header(-2);
  bytes[3] = 0x31;
  assert.throws(() => parseLac2Header(bytes), /bad magic/);
});

test('signed index parsing requires strict sorted unique keys', () => {
  const ordered = new Uint8Array(42);
  ordered.set(indexRecord(-2, -1, 0), 0);
  ordered.set(indexRecord(-1, 0, 0), 21);
  assert.deepEqual(parseLayerIndexV2(ordered).map((r) => [r.lod, r.cx, r.cz]), [[-2, -1, 0], [-1, 0, 0]]);

  const duplicate = new Uint8Array(42);
  duplicate.set(indexRecord(-2, 0, 0), 0);
  duplicate.set(indexRecord(-2, 0, 0), 21);
  assert.throws(() => parseLayerIndexV2(duplicate), /duplicate or unsorted/);
});
