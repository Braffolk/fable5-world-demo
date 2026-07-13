import assert from 'node:assert/strict';
import test from 'node:test';

import type { ChunkKey, WorldManifest } from '../../world/source/WorldSource';
import {
  coverageBoxM,
  coverageExtentLattice,
  latticeWorld,
  layerGeom,
  normalizedStride,
  planLayer,
} from './PlaneFill';

function manifest(format: 1 | 2): WorldManifest {
  const byLod = new Map<number, ChunkKey[]>();
  if (format === 1) {
    byLod.set(0, [{ lod: 0, cx: 0, cz: 0 }]);
  } else {
    byLod.set(-2, Array.from({ length: 16 }, (_, i) => ({ lod: -2, cx: i % 4, cz: Math.floor(i / 4) })));
    byLod.set(-1, [{ lod: -1, cx: 0, cz: 0 }]);
    byLod.set(0, Array.from({ length: 64 }, (_, i) => ({ lod: 0, cx: i % 8, cz: Math.floor(i / 8) })));
    byLod.set(1, Array.from({ length: 4 }, (_, i) => ({ lod: 1, cx: i % 2, cz: Math.floor(i / 2) })));
    byLod.set(2, [{ lod: 2, cx: 0, cz: 0 }]);
    byLod.set(3, [{ lod: 3, cx: 0, cz: 0 }]);
    byLod.set(4, [{ lod: 4, cx: 0, cz: 0 }]);
  }
  return {
    format,
    containers: format === 1 ? ['LAC1'] : ['LAC1', 'LAC2'],
    codec: 'deflate',
    grid: { anchorE: 0, anchorN: 0, chunkMeters: 2048, chunkRes: 2048, lodStep: 4, originX: 0, originZ: 0 },
    layers: {
      height: format === 1
        ? { enc: 1, lods: [0], chunkCount: 1, texelMeters: 1 }
        : {
            enc: 1, lods: [-2, -1, 0, 1, 2, 3, 4], chunkCount: 88, texelMeters: 1,
            baseTexelMeters: 1, finestLod: -2, authorityLod: 0, synthesis: 'microtopography-v1',
          },
    },
    dictionaries: { species: new Map(), understory: new Map(), debris: new Map() },
    coverage: () => null,
    chunks: (_layer, lod) => byLod.get(lod) ?? [],
  };
}

test('format-2 physical levels use exact own-lattice sample centers', () => {
  const geo = layerGeom(manifest(2), 'height');
  assert.equal(geo.mode, 'physical-level');
  assert.equal(geo.texel0, 0.0625);
  const expected = new Map([[-2, 0.03125], [-1, 0.125], [0, 0.5], [1, 2], [4, 128]]);
  for (const [lod, world] of expected) assert.equal(latticeWorld(geo, lod, 0, 'x'), world);
  assert.deepEqual([-2, -1, 0, 1].map((lod) => normalizedStride(geo, lod)), [1, 4, 16, 64]);
});

test('LOD -2 128m chunk boundary shares the exact neighbor sample', () => {
  const geo = layerGeom(manifest(2), 'height');
  const westApron = latticeWorld(geo, -2, 2048, 'x');
  const eastFirst = latticeWorld(geo, -2, 1 * 2048, 'x');
  assert.equal(westApron, 128.03125);
  assert.equal(eastFirst, westApron);
});

test('format-1 legacy placement remains on its historical offset convention', () => {
  const geo = layerGeom(manifest(1), 'height');
  assert.equal(geo.mode, 'legacy-offset');
  assert.equal(latticeWorld(geo, 0, 0, 'x'), 0.5);
  assert.equal(latticeWorld(geo, 1, 0, 'x'), 2.5);
});

test('fine hero coverage does not shrink authority coverage or render domain', () => {
  const m = manifest(2);
  assert.deepEqual(coverageBoxM(m), { minX: 0, minZ: 0, maxX: 16384, maxZ: 16384 });
  const domain = coverageExtentLattice(m);
  assert.ok(domain.latMax - domain.latMin + 1 > 512 / 0.0625);
});

test('format-2 fine windows are near-only while LOD0 keeps the established window size', () => {
  const plans = planLayer(manifest(2), 'height', 2048, 4096);
  assert.equal(plans.find((plan) => plan.lod === -2)?.res, 1536);
  assert.equal(plans.find((plan) => plan.lod === -1)?.res, 1536);
  assert.equal(plans.find((plan) => plan.lod === 0)?.res, 2048);
});
