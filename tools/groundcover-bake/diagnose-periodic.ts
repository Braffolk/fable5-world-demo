import { makeDirection } from './ProfileFormat';
import {
  closestOwnershipCandidates,
  enumeratePeriodicPixelHit,
  enumeratePeriodicRasterPixelHits,
  enumeratePeriodicSelectedHits,
  makeCushionCarpetFixture,
  makePeriodicSlice,
  searchProjectedTriangleCandidates,
} from './PeriodicProfile';

const TARGET_DEPTH = 0.13544333259743552;
const TARGET_NORMAL_DOT = 0.7398475352224524;

const fixture = makeCushionCarpetFixture();
const directions = [
  makeDirection(0, 20), makeDirection(90, 20), makeDirection(180, 20), makeDirection(270, 20),
  makeDirection(0, 55), makeDirection(90, 55), makeDirection(180, 55), makeDirection(270, 55),
];
const slices = directions.map((direction) => makePeriodicSlice(fixture.mesh, fixture.tile, direction));
const records = enumeratePeriodicSelectedHits(fixture.mesh, fixture.tile, slices, 96, 96, 14);
const strictRecords = records.filter((record) => (record.nearest?.edgeMargin ?? -1) >= 0.03);
const allStrictCandidates = closestOwnershipCandidates(strictRecords, TARGET_DEPTH, TARGET_NORMAL_DOT, 100_000);
const closestJoint = allStrictCandidates.slice(0, 12);
const closestDepth = [...allStrictCandidates].sort((a, b) => a.depthError - b.depthError).slice(0, 5);
const closestNormal = [...allStrictCandidates].sort((a, b) => a.normalDotError - b.normalDotError).slice(0, 5);
const depthTight = allStrictCandidates.filter((candidate) => candidate.depthError <= 1e-4);
const normalTight = allStrictCandidates.filter((candidate) => candidate.normalDotError <= 1e-3);

const f32 = (value: number): number => Math.fround(value);
const submittedMesh = {
  positions: fixture.mesh.positions.map(f32),
  normals: fixture.mesh.normals.map(f32),
  indices: [...fixture.mesh.indices],
};
const submittedTile = {
  originX: f32(fixture.tile.originX),
  originZ: f32(fixture.tile.originZ),
  sizeX: f32(fixture.tile.sizeX),
  sizeZ: f32(fixture.tile.sizeZ),
  topH: f32(fixture.tile.topH),
};
const submittedSlices = directions.map((direction) => {
  const slice = makePeriodicSlice(submittedMesh, submittedTile, direction);
  return {
    ...slice,
    direction: { x: f32(slice.direction.x), y: f32(slice.direction.y), z: f32(slice.direction.z) },
    depthMin: f32(slice.depthMin),
    depthMax: f32(slice.depthMax),
  };
});
const capturedPixelF32 = enumeratePeriodicPixelHit(
  submittedMesh,
  submittedTile,
  submittedSlices[0]!,
  0,
  3,
  45,
  96,
  96,
);
const capturedProjectedCandidates = searchProjectedTriangleCandidates(
  submittedMesh,
  submittedTile,
  submittedSlices[0]!,
  3,
  45,
  96,
  96,
  0.48297967318105767,
  { x: 0.11128679172272392, y: 0.5511646441752851, z: 0.8269418268531316 },
  12,
);
const capturedRasterF32 = enumeratePeriodicRasterPixelHits(
  submittedMesh,
  submittedTile,
  submittedSlices[0]!,
  3,
  45,
  96,
  96,
  8,
).slice(0, 4);

console.log(JSON.stringify({
  target: { depthSeparation: TARGET_DEPTH, normalDot: TARGET_NORMAL_DOT },
  selectedRecords: records.length,
  strictInteriorRecords: strictRecords.length,
  recordsWithTwoHits: records.filter((record) => record.second !== null).length,
  strictAlternativePairs: allStrictCandidates.length,
  matchCounts: {
    depthErrorAtMost1eMinus4: depthTight.length,
    bothDepth1eMinus4AndNormalDot1eMinus3: depthTight.filter((candidate) => candidate.normalDotError <= 1e-3).length,
    normalDotErrorAtMost1eMinus3: normalTight.length,
  },
  closestJoint,
  closestDepth,
  closestNormal,
  capturedPixelSubmittedF32: capturedPixelF32,
  capturedProjectedCandidateSummary: capturedProjectedCandidates.map((candidate) => ({
    triangleId: candidate.triangleId,
    copy: candidate.copy,
    barycentric: candidate.barycentric,
    minBarycentric: candidate.minBarycentric,
    t: candidate.t,
    depthError: candidate.depthError,
    normal: candidate.normal,
    normalDot: candidate.normalDot,
  })),
  capturedRasterF32,
}, null, 2));
