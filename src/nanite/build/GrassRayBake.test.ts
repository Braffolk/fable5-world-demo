import assert from 'node:assert/strict';
import test from 'node:test';

import {
  bakeGrassRayTile,
  decodeOctNormal,
  encodeOctNormal,
  packGroundCoverRayAtlas,
  recoverPeriodicRootAnchor,
  type GrassRayBakeOpts,
} from './GrassRayBake';

test('canonical periodic root is invariant across hit points, world signs, and layer transforms', () => {
  const angle = Math.PI * 1.618033988749895;
  const scale = 1.071773462536293;
  const phaseX = 0.371;
  const phaseZ = 0.619;
  const cs = Math.cos(angle);
  const sn = Math.sin(angle);
  const rootU = 2;
  const rootV = 6;
  for (const p of [
    { x: 370384.125, z: -225570.375 },
    { x: -370384.125, z: 225570.375 },
  ]) {
    const rootQx = p.x * scale * cs - p.z * scale * sn + phaseX;
    const rootQz = p.x * scale * sn + p.z * scale * cs + phaseZ;
    const canonicalQx = Math.floor(rootQx) + (rootU + 0.5) / 8;
    const canonicalQz = Math.floor(rootQz) + (rootV + 0.5) / 8;
    for (const dTile of [0.07, 0.61, 1.31]) {
      const dirX = 0.8;
      const dirZ = -0.6;
      const qx = canonicalQx - dirX * dTile;
      const qz = canonicalQz - dirZ * dTile;
      const root = recoverPeriodicRootAnchor(
        qx, qz, dirX, dirZ, dTile, rootV * 8 + rootU, 8,
        angle, scale, phaseX, phaseZ,
      );
      const expectedX = ((canonicalQx - phaseX) * cs + (canonicalQz - phaseZ) * sn) / scale;
      const expectedZ = (-(canonicalQx - phaseX) * sn + (canonicalQz - phaseZ) * cs) / scale;
      assert.ok(Math.abs(root.pTileX - expectedX) < 1e-9);
      assert.ok(Math.abs(root.pTileZ - expectedZ) < 1e-9);
      assert.equal(root.rootU, rootU);
      assert.equal(root.rootV, rootV);
    }
  }
});

test('octahedral normal payload round-trips across the scalar-azimuth seam', () => {
  for (const normal of [
    [0.99995, 0.01, 0.0001],
    [0.99995, 0.01, -0.0001],
    [-0.99995, 0.01, 0.0001],
    [-0.99995, 0.01, -0.0001],
    [0, 1, 0],
  ] as const) {
    const encoded = encodeOctNormal(normal[0], normal[1], normal[2]);
    const decoded = decodeOctNormal(...encoded);
    const length = Math.hypot(normal[0], normal[1], normal[2]);
    const dot = decoded[0] * normal[0] / length
      + decoded[1] * normal[1] / length
      + decoded[2] * normal[2] / length;
    assert.ok(dot > 0.999999, `normal round-trip dot ${dot}`);
  }
  const a = encodeOctNormal(1, 0.02, 0.0001);
  const b = encodeOctNormal(1, 0.02, -0.0001);
  assert.ok(Math.hypot(a[0] - b[0], a[1] - b[1]) < 0.001);
});

const blade = { c: 1, s: 0, ox: 0, oz: 0, hk: 1, lean: 0.1 };

function opts(section: 'rectangle' | 'ellipse' = 'rectangle'): GrassRayBakeOpts {
  return {
    label: 'test',
    res: 8,
    angles: 4,
    blades: [blade],
    sub: 2,
    cellM: 0.105,
    shiftK: 0,
    thickK: 0,
    halfW: 0.018,
    halfT: 0.006,
    fibers: 1,
    section,
    spread: 1,
    tiers: [1, 0.3],
    keepSalt: 0x51a55e ^ 0x37a1,
    arcK: 0,
  };
}

test('ray profile bake is deterministic and density tiers remain nested', () => {
  const a = bakeGrassRayTile(opts());
  const b = bakeGrassRayTile(opts());
  assert.deepEqual(a, b);

  const missR = Math.round(255 / (1 + a.dMaxTile));
  const hits = (data: Uint8Array): number => {
    let n = 0;
    for (let i = 0; i < data.length; i += 4) if ((data[i] as number) > missR) n++;
    return n;
  };
  assert.ok(hits(a.data[0] as Uint8Array) > 0);
  assert.ok(hits(a.data[1] as Uint8Array) <= hits(a.data[0] as Uint8Array));
});

test('ellipse profile is analytic, distinct, and rejects unsupported thickening', () => {
  const rectangle = bakeGrassRayTile(opts('rectangle'));
  const ellipse = bakeGrassRayTile(opts('ellipse'));
  assert.notDeepEqual(ellipse.data[0], rectangle.data[0]);
  assert.ok((ellipse.data[0] as Uint8Array).some((v, i) => i % 4 === 2 && v >= 126 && v <= 129));

  const bad = opts('ellipse');
  bad.thickK = 0.1;
  assert.throws(() => bakeGrassRayTile(bad), /ellipse sections require thickK=0/);
});

test('ellipsoid-cap bake produces rounded upward normals for a bounded elevation slice', () => {
  const capOpts = opts('ellipse');
  capOpts.shape = 'ellipsoid-cap';
  capOpts.dropPerTile = 1;
  const cap = bakeGrassRayTile(capOpts);
  const extruded = bakeGrassRayTile(opts('ellipse'));
  assert.notDeepEqual(cap.data[0], extruded.data[0]);
  assert.ok((cap.data[0] as Uint8Array).some(
    (v, i) => i % 4 === 2 && v > 140 && v < 250,
  ));

  const missingSlope = opts('ellipse');
  missingSlope.shape = 'ellipsoid-cap';
  assert.throws(() => bakeGrassRayTile(missingSlope), /ellipsoid-cap requires/);
});

test('profile atlas copies real slices exactly and guards every angular seam', () => {
  const p0 = bakeGrassRayTile(opts('rectangle'));
  const p1 = bakeGrassRayTile(opts('ellipse'));
  const atlas = packGroundCoverRayAtlas([p0, p1]);
  const sliceBytes = p0.res * p0.res * 4;
  assert.equal(atlas.angleStride, p0.angles + 2);
  assert.equal(atlas.depth, 2 * (p0.angles + 2));

  for (let pi = 0; pi < 2; pi++) {
    const src = (pi === 0 ? p0 : p1).data[0] as Uint8Array;
    const dest = atlas.data[0] as Uint8Array;
    const base = pi * atlas.angleStride;
    const slice = (a: Uint8Array, i: number): Uint8Array =>
      a.slice(i * sliceBytes, (i + 1) * sliceBytes);
    assert.deepEqual(slice(dest, base), slice(src, p0.angles - 1));
    for (let ai = 0; ai < p0.angles; ai++) {
      assert.deepEqual(slice(dest, base + 1 + ai), slice(src, ai));
    }
    assert.deepEqual(slice(dest, base + p0.angles + 1), slice(src, 0));
  }
});

test('profile atlas rejects incompatible layouts', () => {
  const a = bakeGrassRayTile(opts());
  const bOpts = opts();
  bOpts.angles = 8;
  const b = bakeGrassRayTile(bOpts);
  assert.throws(() => packGroundCoverRayAtlas([a, b]), /layouts must match/);
});
