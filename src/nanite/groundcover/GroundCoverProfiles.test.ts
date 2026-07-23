import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import {
  derivePeriodicProfileLattice,
  makePeriodicProfileColorTexture,
  makePeriodicProfileTexture,
  parsePeriodicProfile,
} from './GroundCoverPeriodicProfile';
import {
  makePeriodicProfileArrayTexture,
  parsePeriodicProfileArray,
} from './GroundCoverProfileArray';
import type { PeriodicProfileData } from './GroundCoverProfileTypes';
import {
  DataArrayTexture,
  DataUtils,
  HalfFloatType,
  NearestFilter,
  UnsignedByteType,
} from 'three';
import { packPeriodicProfileArray } from '../../../tools/groundcover-bake/ProfileArrayFormat';

function fixture(options: { profileId?: number; corruptPayload?: boolean } = {}): Uint8Array {
  const azimuthCount = 4;
  const elevations = [20, 55];
  const sliceCount = azimuthCount * elevations.length;
  const stored = 6;
  const headerBytes = 80;
  const payloadOffset = headerBytes + sliceCount * 64;
  const bytes = new Uint8Array(payloadOffset + stored * stored * 2 * 4 * 8);
  const view = new DataView(bytes.buffer);
  bytes.set([0x47, 0x43, 0x52, 0x50]);
  view.setUint32(4, 2, true);
  view.setUint32(8, options.profileId ?? 5, true);
  view.setUint32(12, stored, true);
  view.setUint32(16, stored, true);
  view.setUint32(20, 4, true);
  view.setUint32(24, 2, true);
  view.setUint32(28, sliceCount, true);
  view.setUint32(32, 8, true);
  view.setUint32(36, payloadOffset, true);
  view.setUint32(40, 1, true);
  view.setUint32(44, 4, true);
  view.setUint32(48, 4, true);
  view.setUint32(52, 1, true);
  view.setFloat32(56, 0.4, true);
  view.setFloat32(68, 1, true);
  view.setFloat32(72, 1, true);
  let slice = 0;
  for (const elevationDeg of elevations) {
    const elevation = elevationDeg * Math.PI / 180;
    for (let azimuth = 0; azimuth < azimuthCount; azimuth++, slice++) {
      const angle = azimuth * Math.PI * 2 / azimuthCount;
      const base = headerBytes + slice * 64;
      view.setFloat32(base, Math.cos(elevation) * Math.cos(angle), true);
      view.setFloat32(base + 4, -Math.sin(elevation), true);
      view.setFloat32(base + 8, Math.cos(elevation) * Math.sin(angle), true);
      view.setFloat32(base + 12, 0, true);
      view.setFloat32(base + 16, 0.8 + slice * 0.01, true);
    }
  }
  if (options.corruptPayload) view.setUint32(36, payloadOffset + 8, true);
  return bytes;
}

test('parses the canonical periodic profile and derives its regular lattice', () => {
  const profile = parsePeriodicProfile(fixture());
  assert.equal(profile.profileId, 5);
  assert.equal(profile.texels.length, 6 * 4 * 6 * 2 * 4);
  const lattice = derivePeriodicProfileLattice(profile);
  assert.equal(lattice.azimuthCount, 4);
  assert.equal(lattice.elevationCount, 2);
  assert.ok(Math.abs(lattice.elevations[0]! - 20 * Math.PI / 180) < 1e-6);
  assert.ok(Math.abs(lattice.elevations[1]! - 55 * Math.PI / 180) < 1e-6);
});

test('rejects non-canonical payload offsets and ids outside the native palette', () => {
  assert.throws(() => parsePeriodicProfile(fixture({ corruptPayload: true })), /payload offset/);
  assert.throws(() => parsePeriodicProfile(fixture({ profileId: 12 })), /outside the native palette/);
});

test('uploads the standalone full-depth carrier as filterable half-float values', () => {
  const profile = parsePeriodicProfile(fixture());
  profile.texels[0] = 32768;
  profile.texels[1] = 65535;
  profile.texels[3] = 65535;
  const texture = makePeriodicProfileTexture(profile);
  assert.equal(texture.type, HalfFloatType);
  assert.equal(texture.minFilter, NearestFilter);
  assert.equal(texture.magFilter, NearestFilter);
  const uploaded = texture.image.data as Uint16Array;
  assert.ok(Math.abs(DataUtils.fromHalfFloat(uploaded[0]!) - 32768 / 65535) < 1e-3);
  assert.equal(DataUtils.fromHalfFloat(uploaded[1]!), 1);
});

test('derives premultiplied authored first-hit colour from a v4 owner triangle', () => {
  const packVertex = (
    x: number,
    y: number,
    z: number,
    r: number,
    g: number,
    b: number,
  ): readonly [number, number, number, number] => {
    const q = (value: number): number => Math.round(value * 65535) & 0xffff;
    return [
      (q(x) | (q(y) << 16)) >>> 0,
      (q(z) | (q(r) << 16)) >>> 0,
      (q(g) | (q(b) << 16)) >>> 0,
      0,
    ];
  };
  const stored = 4;
  const texels = new Uint16Array(stored * stored * 4);
  const owners = new Uint32Array(stored * stored);
  owners.fill(0xffff_ffff);
  // Interior source (x=0,y=1) maps to ray origin (0.25, topH, 0.25).
  const hitTexel = 2 * stored + 1;
  texels[hitTexel * 4 + 3] = 65535;
  owners[hitTexel] = ((16 * 2 ** 22) + (16 * 2 ** 27)) >>> 0;
  const profile: PeriodicProfileData = {
    version: 4,
    profileId: 2,
    storedTileWidth: stored,
    storedTileHeight: stored,
    interiorTileWidth: 2,
    interiorTileHeight: 2,
    atlasColumns: 1,
    atlasRows: 1,
    gutter: 1,
    topH: 1,
    tileOriginX: 0,
    tileOriginZ: 0,
    tileSizeX: 1,
    tileSizeZ: 1,
    slices: [{ direction: [0, -1, 0], depthMin: 0, depthMax: 2 }],
    texels,
    supportScale: 0,
    ownerTexels: owners,
    vertexRecords: new Uint32Array([
      ...packVertex(0, 0, 0, 1, 0, 0),
      ...packVertex(1, 0, 0, 0, 1, 0),
      ...packVertex(0, 0, 1, 0, 0, 1),
    ]),
    triangleRecords: new Uint32Array([0, 1, 2, 0]),
    sourceBounds: [0, 0, 0, 1, 1, 1],
  };
  const texture = makePeriodicProfileColorTexture(profile);
  assert.ok(texture);
  assert.equal(texture.type, UnsignedByteType);
  const colors = texture.image.data as Uint8Array;
  assert.deepEqual(Array.from(colors.subarray(hitTexel * 4, hitTexel * 4 + 4)), [128, 64, 64, 255]);
  assert.deepEqual(Array.from(colors.subarray(0, 4)), [0, 0, 0, 0]);
});

test('parses the real Sphagnum profile through one zero-copy array layer', () => {
  const source = new Uint8Array(readFileSync(new URL(
    '../../assets/groundcover/sphagnum-capillifolium.gcrp',
    import.meta.url,
  )));
  const packed = packPeriodicProfileArray([source], { expectedProfileIds: [5] });
  const parsed = parsePeriodicProfileArray(packed.bytes, [5]);
  assert.equal(parsed.layerWidth, 528);
  assert.equal(parsed.layerHeight, 528);
  assert.equal(parsed.profiles[0]!.profileId, 5);
  assert.equal(parsed.sourceSha256[0], 'cb61dd42c6265067f0be9a320d763da928b1e65a60ae3ebb359f49b3137a9959');
  assert.equal(parsed.halfTexels.buffer, packed.bytes.buffer);
  const standalone = parsePeriodicProfile(source);
  assert.equal(parsed.halfTexels.length, standalone.texels.length);
  for (let value = 0; value < standalone.texels.length; value++) {
    assert.equal(
      parsed.halfTexels[value],
      DataUtils.toHalfFloat(standalone.texels[value]! / 65535),
      `GCAR half-float payload mismatch at ${value}`,
    );
  }
  const texture = makePeriodicProfileArrayTexture(parsed);
  assert.ok(texture instanceof DataArrayTexture);
  assert.equal(texture.type, HalfFloatType);
  assert.equal(texture.image.depth, 1);
  assert.equal(texture.image.data, parsed.halfTexels);
  assert.throws(() => parsePeriodicProfileArray(packed.bytes, [0, 5]), /do not match expected/);
  const noncanonical = packed.bytes.slice();
  const noncanonicalView = new DataView(noncanonical.buffer);
  const alternateElevations = [10, 30, 50, 70].map((degrees) => degrees * Math.PI / 180);
  for (let slice = 0; slice < 64; slice++) {
    const azimuth = Math.floor(slice / 4) * Math.PI * 2 / 16;
    const elevation = alternateElevations[slice % 4]!;
    const offset = 96 + slice * 16;
    noncanonicalView.setFloat32(offset, Math.cos(elevation) * Math.cos(azimuth), true);
    noncanonicalView.setFloat32(offset + 4, -Math.sin(elevation), true);
    noncanonicalView.setFloat32(offset + 8, Math.cos(elevation) * Math.sin(azimuth), true);
  }
  assert.throws(() => parsePeriodicProfileArray(noncanonical, [5]), /canonical lattice/);
});
