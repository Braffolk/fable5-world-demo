import assert from 'node:assert/strict';
import test from 'node:test';
import {
  epipolarTopAddress,
  sameOwnerKey,
  unpackOwner,
} from './analyze-chart-correspondence';

test('epipolar top-plane reprojection preserves the exact 3D point', () => {
  const point = [2.25, 0.47, -1.75] as const;
  const topH = 1.176;
  const direction = [0.42, -0.8, 0.428485705712571] as const;
  const address = epipolarTopAddress(point, direction, topH);
  const depth = (point[1] - topH) / direction[1];
  assert.ok(Math.abs(address[0] + depth * direction[0] - point[0]) < 1e-12);
  assert.ok(Math.abs(topH + depth * direction[1] - point[1]) < 1e-12);
  assert.ok(Math.abs(address[1] + depth * direction[2] - point[2]) < 1e-12);
});

test('owner unpacking adds address tiles to the signed relative copy exactly', () => {
  const triangle = 0x2a_bcde;
  const relativeX = -7;
  const relativeZ = 11;
  const token = (
    triangle
    | ((relativeX + 16) << 22)
    | ((relativeZ + 16) << 27)
  ) >>> 0;
  const key = unpackOwner(token, 13, -9);
  assert.deepEqual(key, { triangleId: triangle, copyX: 6, copyZ: 2 });
  assert.ok(sameOwnerKey(key!, { triangleId: triangle, copyX: 6, copyZ: 2 }));
  assert.equal(unpackOwner(0xffff_ffff, 13, -9), null);
});

