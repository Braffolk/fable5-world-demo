import assert from 'node:assert/strict';
import test from 'node:test';

import {
  availabilityMorphWeight,
  cameraMorphWeight,
  MICRO_MORPH_BANDS,
} from './TerrainMorph';

test('packed micro morph is flat at both radial endpoints', () => {
  for (const lod of [-2, -1] as const) {
    const band = MICRO_MORPH_BANDS[lod];
    assert.equal(cameraMorphWeight(lod, band.innerM, 0, 0, 0), 1);
    assert.equal(cameraMorphWeight(lod, band.outerM, 0, 0, 0), 0);
    const eps = 1e-4;
    assert.ok(1 - cameraMorphWeight(lod, band.innerM + eps, 0, 0, 0) < 1e-10);
    assert.ok(cameraMorphWeight(lod, band.outerM - eps, 0, 0, 0) < 1e-10);
  }
});

test('morph radii fit inside the worst snapped packed windows', () => {
  const res = 1536;
  for (const [lod, texel, snapM] of [[-2, 0.0625, 12], [-1, 0.25, 48]] as const) {
    const halfWindowM = (res * texel) / 2;
    const required = MICRO_MORPH_BANDS[lod].outerM
      + MICRO_MORPH_BANDS[lod].availabilityM
      + snapM / 2;
    assert.ok(required <= halfWindowM, `${lod}: ${required}m exceeds ${halfWindowM}m`);
  }
});

test('availability taper hands ownership to the parent continuously', () => {
  const lod = -2;
  const texel = 0.0625;
  const res = 1536;
  const origin = 1000;
  const boundary = origin + texel;
  assert.equal(availabilityMorphWeight(lod, boundary, 1020, origin, 1000, texel, res), 0);
  assert.equal(
    availabilityMorphWeight(
      lod,
      boundary + MICRO_MORPH_BANDS[lod].availabilityM,
      1020,
      origin,
      1000,
      texel,
      res,
    ),
    1,
  );
  assert.equal(availabilityMorphWeight(lod, boundary - texel, 1020, origin, 1000, texel, res), 0);
});

test('published coverage wins over a plane window that extends into absent data', () => {
  const lod = -1;
  const texel = 0.25;
  const origin = 900;
  const coverageMaxX = 1100;
  assert.equal(availabilityMorphWeight(
    lod,
    coverageMaxX - texel,
    0,
    origin,
    -100,
    texel,
    1536,
    800,
    -1000,
    coverageMaxX,
    1000,
  ), 0);
  assert.equal(availabilityMorphWeight(
    lod,
    coverageMaxX - texel - MICRO_MORPH_BANDS[lod].availabilityM,
    0,
    origin,
    -100,
    texel,
    1536,
    800,
    -1000,
    coverageMaxX,
    1000,
  ), 1);
  assert.equal(availabilityMorphWeight(
    lod, coverageMaxX + 1, 0, origin, -100, texel, 1536, 800, -1000, coverageMaxX, 1000,
  ), 0);
});
