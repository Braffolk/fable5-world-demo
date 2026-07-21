import assert from 'node:assert/strict';
import test from 'node:test';
import {
  TriangleBvh,
  decodeOwnedProfileGeometry,
  enumeratePeriodicRootHits,
  runExteriorClosureCensus,
  type CensusRay,
  type DecodedOwnedProfileGeometry,
} from './ExteriorClosureCensus';

function verticalPeriodicSheet(topH: number): DecodedOwnedProfileGeometry {
  const maximumY = topH - 0.1;
  return {
    version: 4,
    profileId: 2,
    topH,
    tileOriginX: 0,
    tileOriginZ: 0,
    tileSizeX: 1,
    tileSizeZ: 1,
    bounds: { min: [0.5, 0, 0], max: [0.5, maximumY, 1] },
    positions: new Float64Array([
      0.5, 0, 0,
      0.5, maximumY, 0,
      0.5, 0, 1,
      0.5, maximumY, 1,
    ]),
    triangles: new Uint32Array([0, 1, 2, 2, 1, 3]),
    vertexCount: 4,
    triangleCount: 2,
  };
}

function ray(elevationDegrees: number, phaseX = 0.25): CensusRay {
  const elevation = elevationDegrees * Math.PI / 180;
  return {
    phaseX,
    phaseZ: 0.25,
    azimuthDegrees: 0,
    elevationDegrees,
    direction: [Math.cos(elevation), -Math.sin(elevation), 0],
  };
}

function minimalV4Fixture(): Uint8Array {
  const stored = 3;
  const atlasTexels = stored * stored;
  const headerBytes = 128;
  const sliceBytes = 64;
  const geometryOffset = headerBytes + sliceBytes;
  const ownerOffset = geometryOffset + atlasTexels * 8;
  const vertexOffset = ownerOffset + atlasTexels * 4;
  const vertexCount = 3;
  const triangleOffset = vertexOffset + vertexCount * 16;
  const triangleCount = 1;
  const bytes = new Uint8Array(triangleOffset + triangleCount * 16);
  const view = new DataView(bytes.buffer);
  bytes.set([0x47, 0x43, 0x52, 0x50]);
  view.setUint32(4, 4, true);
  view.setUint32(8, 2, true);
  view.setUint32(12, stored, true);
  view.setUint32(16, stored, true);
  view.setUint32(20, 1, true);
  view.setUint32(24, 1, true);
  view.setUint32(28, 1, true);
  view.setUint32(32, 8, true);
  view.setUint32(36, geometryOffset, true);
  view.setUint32(40, 1, true);
  view.setUint32(44, 1, true);
  view.setUint32(48, 1, true);
  view.setUint32(52, 1, true);
  view.setFloat32(56, 2, true);
  view.setFloat32(60, 0, true);
  view.setFloat32(64, 0, true);
  view.setFloat32(68, 1, true);
  view.setFloat32(72, 1, true);
  view.setUint32(76, headerBytes, true);
  view.setUint32(80, ownerOffset, true);
  view.setUint32(84, vertexOffset, true);
  view.setUint32(88, triangleOffset, true);
  view.setUint32(92, vertexCount, true);
  view.setUint32(96, triangleCount, true);
  view.setFloat32(100, 0, true);
  view.setFloat32(104, 0, true);
  view.setFloat32(108, 0, true);
  view.setFloat32(112, 1, true);
  view.setFloat32(116, 1, true);
  view.setFloat32(120, 1, true);
  view.setFloat32(headerBytes, 0, true);
  view.setFloat32(headerBytes + 4, -1, true);
  view.setFloat32(headerBytes + 8, 0, true);
  view.setFloat32(headerBytes + 12, 0, true);
  view.setFloat32(headerBytes + 16, 2, true);
  const vertices = [
    [0, 1, 0],
    [1, 1, 0],
    [0, 1, 1],
  ] as const;
  for (let vertex = 0; vertex < vertices.length; vertex++) {
    const offset = vertexOffset + vertex * 16;
    view.setUint16(offset, Math.round(vertices[vertex]![0] * 65535), true);
    view.setUint16(offset + 2, Math.round(vertices[vertex]![1] * 65535), true);
    view.setUint16(offset + 4, Math.round(vertices[vertex]![2] * 65535), true);
  }
  view.setUint32(triangleOffset, 0, true);
  view.setUint32(triangleOffset + 4, 1, true);
  view.setUint32(triangleOffset + 8, 2, true);
  return bytes;
}

test('decodes only the canonical v4 quantized geometry tables', () => {
  const geometry = decodeOwnedProfileGeometry(minimalV4Fixture());
  assert.equal(geometry.vertexCount, 3);
  assert.equal(geometry.triangleCount, 1);
  assert.deepEqual(Array.from(geometry.triangles), [0, 1, 2]);
  assert.deepEqual(Array.from(geometry.positions), [0, 1, 0, 1, 1, 0, 0, 1, 1]);
});

test('BVH returns the nearest analytic triangle, not input order', () => {
  const geometry: DecodedOwnedProfileGeometry = {
    version: 4,
    profileId: 2,
    topH: 2,
    tileOriginX: 0,
    tileOriginZ: 0,
    tileSizeX: 1,
    tileSizeZ: 1,
    bounds: { min: [0, 0.5, 0], max: [1, 1.5, 1] },
    positions: new Float64Array([
      0, 0.5, 0, 1, 0.5, 0, 0, 0.5, 1,
      0, 1.5, 0, 1, 1.5, 0, 0, 1.5, 1,
    ]),
    triangles: new Uint32Array([0, 1, 2, 3, 4, 5]),
    vertexCount: 6,
    triangleCount: 2,
  };
  const hit = TriangleBvh.build(geometry, 1).intersectNearest([0.2, 2, 0.2], [0, -1, 0], 2);
  assert.deepEqual(hit, { triangleId: 1, t: 0.5 });
});

test('enumerates one nearest owner per periodic copy root', () => {
  const geometry = verticalPeriodicSheet(3);
  const hits = enumeratePeriodicRootHits(geometry, TriangleBvh.build(geometry, 1), ray(45));
  assert.deepEqual(hits.map((hit) => hit.copyX), [0, 1, 2]);
  assert.ok(hits.every((hit) => hit.copyZ === 0));
  assert.ok(hits[0]!.t < hits[1]!.t && hits[1]!.t < hits[2]!.t);
});

test('census preserves copies outside the v4 signed five-bit owner range', () => {
  const geometry = verticalPeriodicSheet(40);
  const bvh = TriangleBvh.build(geometry, 1);
  const hits = enumeratePeriodicRootHits(geometry, bvh, ray(45));
  assert.equal(hits.length, 40);
  assert.equal(hits.at(-1)!.copyX, 39);
  const report = runExteriorClosureCensus(geometry, bvh, {
    phaseGrid: 1,
    phaseOffsets: [0.25],
    azimuthCount: 1,
    elevationsDegrees: [45],
    worstRayLimit: 1,
  });
  assert.equal(report.samples.histogram['40'], 1);
  assert.deepEqual(report.samples.byElevation['45'], {
    rayCount: 1,
    rootHitReferences: 40,
    maximum: 40,
    histogram: { '40': 1 },
  });
  assert.equal(report.samples.maximum, 40);
  assert.equal(report.samples.copyRange?.maximumAbsolute, 39);
  assert.equal(report.samples.worstRays[0]!.owners.at(-1)!.copyX, 39);
});
