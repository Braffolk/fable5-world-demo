import assert from 'node:assert/strict';
import test from 'node:test';
import { makeDirection } from './ProfileFormat';
import {
  diagnosePeriodicGpuMismatches,
  enumeratePeriodicRasterPixelHits,
  enumeratePeriodicSelectedHits,
  makeCushionCarpetFixture,
  makeGuardedAtlas,
  makePeriodicSlice,
  packPeriodicProfile,
  parsePeriodicHeader,
  periodicAddress,
  type PeriodicBakeResult,
} from './PeriodicProfile';

test('periodic address is invariant under integer tile translations', () => {
  const { tile } = makeCushionCarpetFixture();
  for (const point of [{ x: -2.41, z: 3.77 }, { x: 0.17, z: 0.83 }, { x: 1.0001, z: -0.002 }]) {
    const base = periodicAddress(point.x, point.z, tile);
    for (let ix = -4; ix <= 4; ix++) {
      for (let iz = -4; iz <= 4; iz++) {
        const moved = periodicAddress(point.x + ix * tile.sizeX, point.z + iz * tile.sizeZ, tile);
        assert.ok(Math.abs(base.u - moved.u) < 2e-15);
        assert.ok(Math.abs(base.v - moved.v) < 2e-15);
      }
    }
  }
});

test('derived copy range conservatively includes every intersecting repeated projection', () => {
  const { mesh, tile } = makeCushionCarpetFixture();
  const slice = makePeriodicSlice(mesh, tile, makeDirection(31, 10));
  const projectedVertices: { x: number; z: number }[] = [];
  for (let i = 0; i < mesh.positions.length; i += 3) {
    const x = mesh.positions[i] as number;
    const y = mesh.positions[i + 1] as number;
    const z = mesh.positions[i + 2] as number;
    const t = (y - tile.topH) / slice.direction.y;
    projectedVertices.push({ x: x - slice.direction.x * t, z: z - slice.direction.z * t });
  }
  const minX = Math.min(...projectedVertices.map((point) => point.x));
  const maxX = Math.max(...projectedVertices.map((point) => point.x));
  const minZ = Math.min(...projectedVertices.map((point) => point.z));
  const maxZ = Math.max(...projectedVertices.map((point) => point.z));
  const selected = new Set(slice.copies.map((copy) => `${copy.ix},${copy.iz}`));
  for (let iz = -8; iz <= 8; iz++) {
    for (let ix = -8; ix <= 8; ix++) {
      const intersects =
        maxX + ix * tile.sizeX >= tile.originX &&
        minX + ix * tile.sizeX <= tile.originX + tile.sizeX &&
        maxZ + iz * tile.sizeZ >= tile.originZ &&
        minZ + iz * tile.sizeZ <= tile.originZ + tile.sizeZ;
      if (intersects) assert.ok(selected.has(`${ix},${iz}`), `missing copy ${ix},${iz}`);
    }
  }
  assert.ok(slice.copies.length > 9, 'grazing direction must prove this is not a hard-coded 3x3 range');
});

function fakeResult(): PeriodicBakeResult {
  const { mesh, tile } = makeCushionCarpetFixture();
  const slice = makePeriodicSlice(mesh, tile, makeDirection(0, 55));
  return {
    tileWidth: 2,
    tileHeight: 2,
    atlasColumns: 1,
    atlasRows: 1,
    tile,
    slices: [slice],
    pixels: [
      0.1, 0.5, 0.5, 1, 0.2, 0.5, 0.5, 1,
      0.3, 0.5, 0.5, 1, -1, 0.5, 0.5, 0,
    ],
    adapter: 'test',
  };
}

test('one-pixel gutters wrap the opposite canonical edge exactly', () => {
  const result = fakeResult();
  const guarded = makeGuardedAtlas(result, 1);
  assert.equal(guarded.storedTileWidth, 4);
  assert.equal(guarded.storedTileHeight, 4);
  const depth = (x: number, y: number): number => guarded.pixels[(y * 4 + x) * 4] as number;
  assert.equal(depth(0, 0), -1);
  assert.equal(depth(1, 1), 0.1);
  assert.equal(depth(2, 1), 0.2);
  assert.equal(depth(3, 1), 0.1);
  assert.equal(depth(1, 3), 0.1);
  assert.equal(depth(3, 3), 0.1);
});

test('GCRP v2 periodic header round-trips shared tile and guard metadata', () => {
  const result = fakeResult();
  const first = packPeriodicProfile(result, 41, 1);
  const second = packPeriodicProfile(result, 41, 1);
  assert.equal(first.sha256, second.sha256);
  assert.deepEqual(first.bytes, second.bytes);
  const header = parsePeriodicHeader(first.bytes);
  assert.equal(header.version, 2);
  assert.equal(header.profileId, 41);
  assert.equal(header.mode, 1);
  assert.equal(header.interiorTileWidth, 2);
  assert.equal(header.storedTileWidth, 4);
  assert.equal(header.gutter, 1);
  assert.ok(Math.abs(header.tile.topH - result.tile.topH) < 1e-6);
  assert.equal(header.payloadOffset, first.payloadOffset);
});

test('selected-hit diagnostic preserves nearest and second triangle/copy identity', () => {
  const mesh = {
    positions: [
      -1, 0.3, -1, 3, 0.3, -1, -1, 0.3, 3,
      -1, 0.1, -1, 3, 0.1, -1, -1, 0.1, 3,
    ],
    normals: [
      0, 1, 0, 0, 1, 0, 0, 1, 0,
      0, -1, 0, 0, -1, 0, 0, -1, 0,
    ],
    indices: [0, 1, 2, 3, 4, 5],
  };
  const tile = { originX: 0, originZ: 0, sizeX: 1, sizeZ: 1, topH: 0.5 };
  const direction = { x: 0, y: -1, z: 0 };
  const slice = {
    direction,
    depthMin: 0,
    depthMax: 1,
    copies: [{ ix: 0, iz: 0 }],
    copyRange: { minX: 0, maxX: 0, minZ: 0, maxZ: 0 },
    conservativeProjectedBounds: { minX: -1, maxX: 3, minZ: -1, maxZ: 3 },
  };
  const [record] = enumeratePeriodicSelectedHits(mesh, tile, [slice], 8, 8, 8);
  assert.ok(record?.nearest);
  assert.ok(record.second);
  assert.equal(record.nearest.triangleId, 0);
  assert.deepEqual(record.nearest.copy, { ix: 0, iz: 0 });
  assert.equal(record.second.triangleId, 1);
  assert.ok(Math.abs(record.nearest.t - 0.2) < 1e-12);
  assert.ok(Math.abs(record.second.t - 0.4) < 1e-12);
  assert.ok(Math.abs((record.depthSeparation as number) - 0.2) < 1e-12);
  assert.ok((record.normalDot as number) > 0.999999);
});

test('GPU mismatch diagnostic binds a second-surface value to its submitted triangle', () => {
  const mesh = {
    positions: [
      -1, 0.3, -1, 3, 0.3, -1, -1, 0.3, 3,
      -1, 0.1, -1, 3, 0.1, -1, -1, 0.1, 3,
    ],
    normals: [
      0, 1, 0, 0, 1, 0, 0, 1, 0,
      0, -1, 0, 0, -1, 0, 0, -1, 0,
    ],
    indices: [0, 1, 2, 3, 4, 5],
  };
  const tile = { originX: 0, originZ: 0, sizeX: 1, sizeZ: 1, topH: 0.5 };
  const slice = {
    direction: { x: 0, y: -1, z: 0 },
    depthMin: 0,
    depthMax: 1,
    copies: [{ ix: 0, iz: 0 }],
    copyRange: { minX: 0, maxX: 0, minZ: 0, maxZ: 0 },
    conservativeProjectedBounds: { minX: -1, maxX: 3, minZ: -1, maxZ: 3 },
  };
  const pixels = new Array<number>(8 * 8 * 4).fill(0);
  for (let texel = 0; texel < 8 * 8; texel++) pixels[texel * 4] = -1;
  const selected = (3 * 8 + 3) * 4;
  pixels[selected] = 0.4;
  pixels[selected + 1] = 0.5;
  pixels[selected + 2] = 1;
  pixels[selected + 3] = 1;
  const result = {
    tileWidth: 8,
    tileHeight: 8,
    atlasColumns: 1,
    atlasRows: 1,
    tile,
    slices: [slice],
    pixels,
    adapter: 'test',
  };
  const [mismatch] = diagnosePeriodicGpuMismatches(result, mesh, 8);
  assert.ok(mismatch);
  assert.equal(mismatch.sliceIndex, 0);
  assert.equal(mismatch.pixelX, 3);
  assert.equal(mismatch.cpu.nearest?.triangleId, 0);
  assert.equal(mismatch.alternativeComparisons[0]?.triangleId, 1);
  assert.ok((mismatch.alternativeComparisons[0]?.depthError ?? 1) < 1e-12);
  assert.equal(mismatch.nearestSubmission?.drawIndex, 0);
  assert.equal(mismatch.nearestSubmission?.projectedVertices.length, 3);
  assert.equal(mismatch.adjacentGpuTexels.length, 4);
});

test('8-bit raster ownership captures the cushion edge that ideal ray ownership misses', () => {
  const fixture = makeCushionCarpetFixture();
  const slice = makePeriodicSlice(fixture.mesh, fixture.tile, makeDirection(0, 20));
  const analytic = enumeratePeriodicSelectedHits(fixture.mesh, fixture.tile, [slice], 96, 96, 14)
    .find((record) => record.pixelX === 3 && record.pixelY === 45);
  const raster = enumeratePeriodicRasterPixelHits(fixture.mesh, fixture.tile, slice, 3, 45, 96, 96, 8)[0];
  assert.equal(analytic?.nearest?.triangleId, 3551);
  assert.equal(raster?.triangleId, 3032);
  assert.ok(Math.abs((raster?.t ?? 0) - 0.48297979832216587) < 1e-12);
  assert.ok((raster?.edgeMargin ?? -1) > 0);
});
