import assert from 'node:assert/strict';
import test from 'node:test';
import {
  makeDirection,
  makeEllipsoidFixture,
  makeProjectionSlice,
  packProfile,
  projectionFrame,
  validateIndexedMesh,
  type BakeResult,
} from './ProfileFormat';

test('ellipsoid fixture is a valid arbitrary indexed triangle mesh', () => {
  const mesh = makeEllipsoidFixture({
    center: { x: 0, y: 0.5, z: 0 },
    radii: { x: 0.3, y: 0.5, z: 0.2 },
    sectors: 16,
    stacks: 8,
  });
  validateIndexedMesh(mesh);
  assert.equal(mesh.positions.length / 3, 17 * 9);
  assert.equal(mesh.indices.length / 3, 16 * 2 * 7);
});

test('projection frame is orthonormal and padded bounds contain every vertex', () => {
  const mesh = makeEllipsoidFixture({
    center: { x: 0.1, y: 0.4, z: -0.05 },
    radii: { x: 0.3, y: 0.4, z: 0.2 },
    sectors: 16,
    stacks: 8,
  });
  const slice = makeProjectionSlice(mesh, makeDirection(37, 28));
  const { direction: d, axisU: u, axisV: v } = projectionFrame(slice.direction);
  const dot = (a: typeof d, b: typeof d): number => a.x * b.x + a.y * b.y + a.z * b.z;
  assert.ok(Math.abs(dot(u, v)) < 1e-12);
  assert.ok(Math.abs(dot(u, d)) < 1e-12);
  assert.ok(Math.abs(dot(v, d)) < 1e-12);
  for (let i = 0; i < mesh.positions.length; i += 3) {
    const p = { x: mesh.positions[i] as number, y: mesh.positions[i + 1] as number, z: mesh.positions[i + 2] as number };
    const pu = dot(p, u);
    const pv = dot(p, v);
    const pd = dot(p, d);
    assert.ok(pu > slice.bounds.uMin && pu < slice.bounds.uMax);
    assert.ok(pv > slice.bounds.vMin && pv < slice.bounds.vMax);
    assert.ok(pd > slice.bounds.depthMin && pd < slice.bounds.depthMax);
  }
});

test('profile packing is deterministic and preserves the miss sentinel', () => {
  const slice = {
    ...projectionFrame(makeDirection(0, 45)),
    bounds: { uMin: -1, uMax: 1, vMin: -1, vMax: 1, depthMin: -1, depthMax: 1 },
  };
  const result: BakeResult = {
    tileWidth: 2,
    tileHeight: 1,
    atlasColumns: 1,
    atlasRows: 1,
    slices: [slice],
    pixels: [0.25, 0.5, 0.5, 1, -1, 0.5, 0.5, 0],
    adapter: 'test',
  };
  const first = packProfile(result, 7);
  const second = packProfile(result, 7);
  assert.equal(first.sha256, second.sha256);
  assert.deepEqual(first.bytes, second.bytes);
  const view = new DataView(first.bytes.buffer, first.bytes.byteOffset, first.bytes.byteLength);
  assert.equal(String.fromCharCode(...first.bytes.subarray(0, 4)), 'GCRP');
  assert.equal(view.getUint32(8, true), 7);
  assert.equal(view.getUint16(first.payloadOffset + 6, true), 65535);
  assert.equal(view.getUint16(first.payloadOffset + 8, true), 65535);
  assert.equal(view.getUint16(first.payloadOffset + 14, true), 0);
});

test('invalid triangle indices fail before GPU submission', () => {
  assert.throws(
    () => validateIndexedMesh({ positions: [0, 0, 0], normals: [0, 1, 0], indices: [0, 1, 0] }),
    /outside/,
  );
});
