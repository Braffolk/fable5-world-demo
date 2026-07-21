import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { DataUtils } from 'three';
import {
  PROFILE_ARRAY_ELEVATIONS_DEG,
  PROFILE_ARRAY_SLICE_COUNT,
  packPeriodicProfileArray,
  parsePeriodicProfileArray,
} from './ProfileArrayFormat';

const HEADER_BYTES = 80;
const SLICE_BYTES = 64;

function source(profileId: number, options: { stored?: number; directionNudge?: number } = {}): Uint8Array {
  const stored = options.stored ?? 3;
  const payloadOffset = HEADER_BYTES + PROFILE_ARRAY_SLICE_COUNT * SLICE_BYTES;
  const layerWidth = stored * 8;
  const layerHeight = stored * 8;
  const bytes = new Uint8Array(payloadOffset + layerWidth * layerHeight * 8);
  const view = new DataView(bytes.buffer);
  bytes.set([0x47, 0x43, 0x52, 0x50]);
  view.setUint32(4, 2, true);
  view.setUint32(8, profileId, true);
  view.setUint32(12, stored, true);
  view.setUint32(16, stored, true);
  view.setUint32(20, 8, true);
  view.setUint32(24, 8, true);
  view.setUint32(28, PROFILE_ARRAY_SLICE_COUNT, true);
  view.setUint32(32, 8, true);
  view.setUint32(36, payloadOffset, true);
  view.setUint32(40, 1, true);
  view.setUint32(44, stored - 2, true);
  view.setUint32(48, stored - 2, true);
  view.setUint32(52, 1, true);
  view.setFloat32(56, 0.2 + profileId * 0.01, true);
  view.setFloat32(60, -0.1, true);
  view.setFloat32(64, -0.1, true);
  view.setFloat32(68, 0.2, true);
  view.setFloat32(72, 0.2, true);
  for (let slice = 0; slice < PROFILE_ARRAY_SLICE_COUNT; slice++) {
    const azimuthIndex = Math.floor(slice / PROFILE_ARRAY_ELEVATIONS_DEG.length);
    const elevationIndex = slice % PROFILE_ARRAY_ELEVATIONS_DEG.length;
    const azimuth = azimuthIndex * Math.PI * 2 / 16;
    const elevation = PROFILE_ARRAY_ELEVATIONS_DEG[elevationIndex]! * Math.PI / 180;
    const offset = HEADER_BYTES + slice * SLICE_BYTES;
    view.setFloat32(offset, Math.cos(elevation) * Math.cos(azimuth) + (slice === 0 ? options.directionNudge ?? 0 : 0), true);
    view.setFloat32(offset + 4, -Math.sin(elevation), true);
    view.setFloat32(offset + 8, Math.cos(elevation) * Math.sin(azimuth), true);
    view.setFloat32(offset + 12, profileId * 0.001, true);
    view.setFloat32(offset + 16, 0.5 + slice * 0.001 + profileId * 0.001, true);
  }
  for (let value = 0; value < layerWidth * layerHeight * 4; value++) {
    view.setUint16(payloadOffset + value * 2, (profileId * 4093 + value * 17) & 0xffff, true);
  }
  return bytes;
}

test('packs twelve shuffled GCRP/v2 profiles into a deterministic layer-major array', () => {
  const ordered = Array.from({ length: 12 }, (_value, profileId) => source(profileId));
  const shuffled = [ordered[7]!, ordered[2]!, ordered[11]!, ordered[0]!, ordered[5]!, ordered[9]!, ordered[1]!, ordered[8]!, ordered[4]!, ordered[10]!, ordered[3]!, ordered[6]!];
  const expectedProfileIds = Array.from({ length: 12 }, (_value, index) => index);
  const first = packPeriodicProfileArray(shuffled, { expectedProfileIds });
  const second = packPeriodicProfileArray(ordered, { expectedProfileIds });
  assert.equal(first.sha256, second.sha256);
  assert.deepEqual(first.bytes, second.bytes);
  assert.equal(first.header.profileCount, 12);
  assert.equal(first.header.layerWidth, 24);
  assert.equal(first.header.layerHeight, 24);
  assert.deepEqual(first.profiles.map((profile) => profile.profileId), expectedProfileIds);
  assert.deepEqual(first.profiles.map((profile) => profile.layerIndex), expectedProfileIds);
  ordered.forEach((bytes, profileId) => {
    assert.equal(
      first.profiles[profileId]!.sourceSha256,
      createHash('sha256').update(bytes).digest('hex'),
    );
  });
});

test('converts source rgba16unorm payloads to upload-ready rgba16float bits per layer', () => {
  const packed = packPeriodicProfileArray([source(3), source(9)]);
  const parsed = parsePeriodicProfileArray(packed.bytes);
  const valuesPerLayer = parsed.header.layerWidth * parsed.header.layerHeight * 4;
  const expectedFirst = DataUtils.toHalfFloat((3 * 4093 & 0xffff) / 65535);
  const expectedSecond = DataUtils.toHalfFloat((9 * 4093 & 0xffff) / 65535);
  assert.equal(parsed.halfTexels[0], expectedFirst);
  assert.equal(parsed.halfTexels[valuesPerLayer], expectedSecond);
  assert.ok(Math.abs(DataUtils.fromHalfFloat(parsed.halfTexels[0]!) - (3 * 4093 & 0xffff) / 65535) < 5e-4);
});

test('preserves per-profile, per-direction depth bounds', () => {
  const parsed = parsePeriodicProfileArray(packPeriodicProfileArray([source(2), source(6)]).bytes);
  const boundsPerProfile = PROFILE_ARRAY_SLICE_COUNT * 2;
  assert.ok(Math.abs(parsed.depthBounds[0]! - 0.002) < 1e-7);
  assert.ok(Math.abs(parsed.depthBounds[1]! - 0.502) < 1e-7);
  assert.ok(Math.abs(parsed.depthBounds[boundsPerProfile]! - 0.006) < 1e-7);
  assert.ok(Math.abs(parsed.depthBounds[boundsPerProfile + 1]! - 0.506) < 1e-7);
});

test('fails closed on missing, duplicate, incompatible, or noncanonical profile inputs', () => {
  assert.throws(
    () => packPeriodicProfileArray([source(0), source(2)], { expectedProfileIds: [0, 1, 2] }),
    /does not match expected/,
  );
  assert.throws(() => packPeriodicProfileArray([source(1), source(1)]), /duplicate profile id/);
  assert.throws(() => packPeriodicProfileArray([source(0), source(1, { stored: 4 })]), /not texture-array compatible/);
  assert.throws(() => packPeriodicProfileArray([source(0, { directionNudge: 0.01 })]), /canonical 16x4 lattice/);
});

test('rejects noncanonical GCAR offsets and truncated payloads', () => {
  const packed = packPeriodicProfileArray([source(5)]);
  const offsetCorrupt = packed.bytes.slice();
  new DataView(offsetCorrupt.buffer).setUint32(92, packed.header.payloadOffset + 8, true);
  assert.throws(() => parsePeriodicProfileArray(offsetCorrupt), /table offsets/);
  assert.throws(() => parsePeriodicProfileArray(packed.bytes.subarray(0, packed.bytes.length - 8)), /payload length/);
});
