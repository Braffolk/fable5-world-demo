import assert from 'node:assert/strict';
import test from 'node:test';
import { computeAtlasSizing, selectSlopeTriangle } from './RaySurrogateEvaluator';

const elevations = [15, 35, 55, 75].map((value) => value * Math.PI / 180);

function direction(azimuthDegrees: number, elevationDegrees: number): readonly [number, number, number] {
  const azimuth = azimuthDegrees * Math.PI / 180;
  const elevation = elevationDegrees * Math.PI / 180;
  return [
    Math.cos(elevation) * Math.cos(azimuth),
    -Math.sin(elevation),
    Math.cos(elevation) * Math.sin(azimuth),
  ];
}

test('slope triangle barycentrics reconstruct a live interior slope', () => {
  const ray = direction(11.25, 25);
  const selection = selectSlopeTriangle(16, elevations, ray);
  const reconstructedX = selection.vertices.reduce(
    (sum, vertex, index) => sum + vertex.x * selection.weights[index]!,
    0,
  );
  const reconstructedZ = selection.vertices.reduce(
    (sum, vertex, index) => sum + vertex.z * selection.weights[index]!,
    0,
  );
  assert.ok(Math.abs(reconstructedX - ray[0] / -ray[1]) < 1e-12);
  assert.ok(Math.abs(reconstructedZ - ray[2] / -ray[1]) < 1e-12);
  assert.equal(selection.extrapolated, false);
});

test('slope triangle labels the shipped 5-to-15 degree gap as extrapolation', () => {
  const ray = direction(11.25, 5);
  const selection = selectSlopeTriangle(16, elevations, ray);
  assert.equal(selection.extrapolated, true);
  assert.ok(selection.weights.some((weight) => weight < 0));
});

test('fresh 5-degree row and vertical singleton close the evaluator slope domain', () => {
  const augmented = [5, 15, 35, 55, 75, 90].map((value) => value * Math.PI / 180);
  assert.equal(selectSlopeTriangle(16, augmented, direction(11.25, 10)).extrapolated, false);
  assert.equal(selectSlopeTriangle(16, augmented, direction(11.25, 82.5)).extrapolated, false);
});

test('byte-matched denser azimuth layouts stay within one percent of the shipped atlas', () => {
  const shipped = computeAtlasSizing(16, 256, 4, 0);
  const azimuth32 = computeAtlasSizing(32, 161);
  const azimuth64 = computeAtlasSizing(64, 113);

  assert.equal(shipped.storedAtlasRecordCount, 4_260_096);
  assert.equal(shipped.storedAtlasBytes, 34_080_768);
  assert.equal(azimuth32.storedAtlasRecordCount, 4_277_609);
  assert.equal(azimuth32.storedAtlasBytes, 34_220_872);
  assert.equal(azimuth64.storedAtlasRecordCount, 4_245_225);
  assert.equal(azimuth64.storedAtlasBytes, 33_961_800);
  assert.ok(Math.abs(azimuth32.storedAtlasBytes / shipped.storedAtlasBytes - 1) < 0.01);
  assert.ok(Math.abs(azimuth64.storedAtlasBytes / shipped.storedAtlasBytes - 1) < 0.01);
});
