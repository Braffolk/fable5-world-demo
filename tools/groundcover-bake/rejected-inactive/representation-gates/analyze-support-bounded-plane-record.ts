/**
 * Pure-offline gate for a support-bounded canonical plane record.
 *
 * The proposed runtime shape is statically fixed: four angular records at one
 * nearest spatial phase, each containing hit height, one geometric plane
 * normal, and a conservative in-triangle support radius.  The exact live ray
 * intersects each plane and a branchless minimum elects the nearest supported
 * candidate.  A separately costed 2x2 phase footprint evaluates sixteen
 * records; it is not part of the primary proposal.  This file never edits or
 * exercises the runtime shader.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import sharp from 'sharp';
import {
  TriangleBvh,
  decodeOwnedProfileGeometry,
  type CensusVec3,
  type DecodedOwnedProfileGeometry,
} from '../../ExteriorClosureCensus';

const SOURCE = resolve(
  'data/work/groundcover-gpu-bake-estonian-graminoids-v4',
  '2-calamagrostis-canescens',
  '2ed57f59d86e8376',
  'calamagrostis-canescens-periodic-profile.gcrp',
);
const EXPECTED_SOURCE_SHA256 = '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const HEADER_BYTES = 128;
const SLICE_BYTES = 64;
const GEOMETRY_TEXEL_BYTES = 8;
const OWNER_TEXEL_BYTES = 4;
const MISS_OWNER = 0xffff_ffff;
const SUPPORT_SCALE_METRES = 0.064;
const SUPPORT_GUARD_METRES = 2e-5;
const HIT_POSITION_EPSILON_METRES = 5e-5;
const PLANE_DISTANCE_EPSILON_METRES = 5e-5;
const PLANE_POLE_EPSILON = 1e-8;
const QA_GRID = 32;
const SAMPLE_GRID = 28;
const MOTION_SAMPLES = 192;
const BOUNDS_EPSILON = 1e-12;

type Category = 'structure' | 'panicle';

interface Slice {
  direction: CensusVec3;
  azimuthRadians: number;
  elevationRadians: number;
}

interface Field {
  bytes: Uint8Array;
  view: DataView;
  geometry: DecodedOwnedProfileGeometry;
  storedWidth: number;
  storedHeight: number;
  interiorWidth: number;
  interiorHeight: number;
  atlasColumns: number;
  atlasRows: number;
  gutter: number;
  geometryOffset: number;
  ownerOffset: number;
  vertexOffset: number;
  slices: readonly Slice[];
  azimuthCount: number;
  elevationCount: number;
  elevationsRadians: readonly number[];
  triangleColors: Float32Array;
}

interface RawRecord {
  hit: boolean;
  triangleId: number;
  copyX: number;
  copyZ: number;
  origin: CensusVec3;
  direction: CensusVec3;
  sliceIndex: number;
  atlasTexel: number;
}

interface TriangleChart {
  triangleId: number;
  copyX: number;
  copyZ: number;
  a: CensusVec3;
  b: CensusVec3;
  c: CensusVec3;
  normal: CensusVec3;
  canonicalPoint: CensusVec3;
  safeRadius: number;
  category: Category;
  atlasTexel: number;
  sliceIndex: number;
  canonicalDirection: CensusVec3;
  canonicalOrigin: CensusVec3;
}

interface PackedChart extends TriangleChart {
  packedWords: readonly [number, number];
  reconstructedNormal: CensusVec3;
  reconstructedPoint: CensusVec3;
  reconstructedRadius: number;
}

interface Candidate {
  t: number;
  point: CensusVec3;
  chart: TriangleChart | PackedChart;
  supportAccepted: boolean;
  triangleAccepted: boolean;
  planeDistance: number;
  displacement: number;
}

interface TruthHit {
  triangleId: number;
  copyX: number;
  copyZ: number;
  t: number;
  point: CensusVec3;
  category: Category;
}

interface CaseDefinition {
  name: string;
  azimuthDegrees: number;
  elevationDegrees: number;
  kind: 'grid' | 'motion' | 'canonical';
}

interface MutableMetrics {
  rays: number;
  truthHits: number;
  truthMisses: number;
  predictedHits: number;
  predictedMisses: number;
  truePositive: number;
  trueNegative: number;
  falsePositive: number;
  falseNegative: number;
  exactOwner: number;
  wrongOwner: number;
  truthPanicle: number;
  truthStructure: number;
  exactPanicle: number;
  exactStructure: number;
  categoryPanicle: number;
  categoryStructure: number;
  candidateRecords: number;
  canonicalMiss: number;
  invalidCanonicalTriangle: number;
  zeroSupport: number;
  planePole: number;
  outOfBand: number;
  supportRejected: number;
  triangleRejected: number;
  supportOnlyLeaks: number;
  positionErrors: number[];
  depthErrors: number[];
  planeDistances: number[];
  acceptedSupports: number[];
  canonicalSupports: number[];
  candidateDisplacements: number[];
  displacementToSupportRatios: number[];
  truthMask: Uint8Array;
  predictionMask: Uint8Array;
  exactMask: Uint8Array;
  wrongMask: Uint8Array;
}

interface Evaluation {
  truth: TruthHit | null;
  prediction: Candidate | null;
  supportOnlyPrediction: Candidate | null;
  records: number;
  canonicalMiss: number;
  invalidCanonicalTriangle: number;
  zeroSupport: number;
  planePole: number;
  outOfBand: number;
  supportRejected: number;
  triangleRejected: number;
  supportOnlyLeaks: number;
  canonicalSupports: number[];
  candidateDisplacements: number[];
  displacementToSupportRatios: number[];
}

interface Quantiles {
  count: number;
  p50: number | null;
  p95: number | null;
  p99: number | null;
  maximum: number | null;
}

function sha256(value: Uint8Array | string): string {
  return createHash('sha256').update(value).digest('hex');
}

function positiveModulo(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus;
}

function wrappedUnit(value: number): number {
  return value - Math.floor(value);
}

function dot(a: CensusVec3, b: CensusVec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function subtract(a: CensusVec3, b: CensusVec3): CensusVec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function addScaled(a: CensusVec3, b: CensusVec3, scale: number): CensusVec3 {
  return [a[0] + b[0] * scale, a[1] + b[1] * scale, a[2] + b[2] * scale];
}

function cross(a: CensusVec3, b: CensusVec3): CensusVec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function length3(value: CensusVec3): number {
  return Math.hypot(value[0], value[1], value[2]);
}

function normalize(value: CensusVec3): CensusVec3 {
  const inverse = 1 / Math.max(1e-30, length3(value));
  return [value[0] * inverse, value[1] * inverse, value[2] * inverse];
}

function direction(azimuthDegrees: number, elevationDegrees: number): CensusVec3 {
  const azimuth = azimuthDegrees * Math.PI / 180;
  const elevation = elevationDegrees * Math.PI / 180;
  const horizontal = Math.cos(elevation);
  return [horizontal * Math.cos(azimuth), -Math.sin(elevation), horizontal * Math.sin(azimuth)];
}

function decodeField(bytes: Uint8Array): Field {
  const geometry = decodeOwnedProfileGeometry(bytes);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const storedWidth = view.getUint32(12, true);
  const storedHeight = view.getUint32(16, true);
  const atlasColumns = view.getUint32(20, true);
  const atlasRows = view.getUint32(24, true);
  const sliceCount = view.getUint32(28, true);
  const geometryOffset = view.getUint32(36, true);
  const interiorWidth = view.getUint32(44, true);
  const interiorHeight = view.getUint32(48, true);
  const gutter = view.getUint32(52, true);
  const ownerOffset = view.getUint32(80, true);
  const vertexOffset = view.getUint32(84, true);
  if (geometryOffset !== HEADER_BYTES + sliceCount * SLICE_BYTES) throw new Error('noncanonical GCRP geometry offset');
  const atlasTexels = storedWidth * storedHeight * atlasColumns * atlasRows;
  if (ownerOffset !== geometryOffset + atlasTexels * GEOMETRY_TEXEL_BYTES) throw new Error('noncanonical GCRP owner offset');
  const slices: Slice[] = [];
  for (let index = 0; index < sliceCount; index++) {
    const offset = HEADER_BYTES + index * SLICE_BYTES;
    const ray: CensusVec3 = [
      view.getFloat32(offset, true),
      view.getFloat32(offset + 4, true),
      view.getFloat32(offset + 8, true),
    ];
    slices.push({
      direction: ray,
      azimuthRadians: positiveModulo(Math.atan2(ray[2], ray[0]), Math.PI * 2),
      elevationRadians: Math.asin(-ray[1]),
    });
  }
  let elevationCount = 1;
  while (
    elevationCount < slices.length
    && Math.abs(slices[elevationCount]!.azimuthRadians - slices[0]!.azimuthRadians) < 2e-4
  ) elevationCount++;
  if (elevationCount < 2 || slices.length % elevationCount !== 0) throw new Error('GCRP direction lattice is not azimuth-major');
  const azimuthCount = slices.length / elevationCount;
  const elevationsRadians = slices.slice(0, elevationCount).map((slice) => slice.elevationRadians);
  const bounds = geometry.bounds;
  const spanX = bounds.max[0] - bounds.min[0];
  const spanY = bounds.max[1] - bounds.min[1];
  const spanZ = bounds.max[2] - bounds.min[2];
  const vertexColors = new Float32Array(geometry.vertexCount * 3);
  for (let vertex = 0; vertex < geometry.vertexCount; vertex++) {
    const offset = vertexOffset + vertex * 16;
    // Validate the independently decoded position while visiting packed colour.
    const x = bounds.min[0] + view.getUint16(offset, true) / 65535 * spanX;
    const y = bounds.min[1] + view.getUint16(offset + 2, true) / 65535 * spanY;
    const z = bounds.min[2] + view.getUint16(offset + 4, true) / 65535 * spanZ;
    const at = vertex * 3;
    if (Math.max(
      Math.abs(x - geometry.positions[at]!),
      Math.abs(y - geometry.positions[at + 1]!),
      Math.abs(z - geometry.positions[at + 2]!),
    ) > 1e-12) throw new Error('packed GCRP vertex decode disagreement');
    vertexColors[at] = view.getUint16(offset + 6, true) / 65535;
    vertexColors[at + 1] = view.getUint16(offset + 8, true) / 65535;
    vertexColors[at + 2] = view.getUint16(offset + 10, true) / 65535;
  }
  const triangleColors = new Float32Array(geometry.triangleCount * 3);
  for (let triangle = 0; triangle < geometry.triangleCount; triangle++) {
    for (let corner = 0; corner < 3; corner++) {
      const vertex = geometry.triangles[triangle * 3 + corner]! * 3;
      for (let channel = 0; channel < 3; channel++) {
        triangleColors[triangle * 3 + channel] += vertexColors[vertex + channel]! / 3;
      }
    }
  }
  return {
    bytes,
    view,
    geometry,
    storedWidth,
    storedHeight,
    interiorWidth,
    interiorHeight,
    atlasColumns,
    atlasRows,
    gutter,
    geometryOffset,
    ownerOffset,
    vertexOffset,
    slices,
    azimuthCount,
    elevationCount,
    elevationsRadians,
    triangleColors,
  };
}

function category(field: Field, triangleId: number): Category {
  return field.triangleColors[triangleId * 3]! > field.triangleColors[triangleId * 3 + 1]!
    ? 'panicle'
    : 'structure';
}

function sliceIndex(field: Field, azimuthIndex: number, elevationIndex: number): number {
  return positiveModulo(azimuthIndex, field.azimuthCount) * field.elevationCount + elevationIndex;
}

function phaseCoordinates(field: Field, phaseX: number, phaseZ: number): readonly [number, number] {
  const u = wrappedUnit((phaseX - field.geometry.tileOriginX) / field.geometry.tileSizeX);
  const v = 1 - wrappedUnit((phaseZ - field.geometry.tileOriginZ) / field.geometry.tileSizeZ);
  return [u * field.interiorWidth - 0.5, v * field.interiorHeight - 0.5];
}

function atlasTexel(field: Field, targetSlice: number, x: number, y: number): number {
  const column = targetSlice % field.atlasColumns;
  const row = Math.floor(targetSlice / field.atlasColumns);
  const atlasWidth = field.storedWidth * field.atlasColumns;
  const atlasX = column * field.storedWidth + field.gutter + positiveModulo(x, field.interiorWidth);
  const atlasY = row * field.storedHeight + field.gutter + positiveModulo(y, field.interiorHeight);
  return atlasY * atlasWidth + atlasX;
}

function recordAt(
  field: Field,
  targetSlice: number,
  x: number,
  y: number,
  livePhaseX: number,
  livePhaseZ: number,
): RawRecord {
  const wrappedX = positiveModulo(x, field.interiorWidth);
  const wrappedY = positiveModulo(y, field.interiorHeight);
  const baseOriginX = field.geometry.tileOriginX + (wrappedX + 0.5) / field.interiorWidth * field.geometry.tileSizeX;
  const baseOriginZ = field.geometry.tileOriginZ + (1 - (wrappedY + 0.5) / field.interiorHeight) * field.geometry.tileSizeZ;
  const shiftX = Math.round((livePhaseX - baseOriginX) / field.geometry.tileSizeX);
  const shiftZ = Math.round((livePhaseZ - baseOriginZ) / field.geometry.tileSizeZ);
  const texel = atlasTexel(field, targetSlice, wrappedX, wrappedY);
  const owner = field.view.getUint32(field.ownerOffset + texel * OWNER_TEXEL_BYTES, true);
  return {
    hit: owner !== MISS_OWNER,
    triangleId: owner === MISS_OWNER ? -1 : owner & 0x3f_ffff,
    copyX: owner === MISS_OWNER ? 0 : ((owner >>> 22) & 0x1f) - 16 + shiftX,
    copyZ: owner === MISS_OWNER ? 0 : ((owner >>> 27) & 0x1f) - 16 + shiftZ,
    origin: [baseOriginX + shiftX * field.geometry.tileSizeX, field.geometry.topH, baseOriginZ + shiftZ * field.geometry.tileSizeZ],
    direction: field.slices[targetSlice]!.direction,
    sliceIndex: targetSlice,
    atlasTexel: texel,
  };
}

function triangleVertices(field: Field, triangleId: number, copyX: number, copyZ: number): readonly [CensusVec3, CensusVec3, CensusVec3] {
  const vertices: CensusVec3[] = [];
  for (let corner = 0; corner < 3; corner++) {
    const vertex = field.geometry.triangles[triangleId * 3 + corner]! * 3;
    vertices.push([
      field.geometry.positions[vertex]! + copyX * field.geometry.tileSizeX,
      field.geometry.positions[vertex + 1]!,
      field.geometry.positions[vertex + 2]! + copyZ * field.geometry.tileSizeZ,
    ]);
  }
  return vertices as unknown as readonly [CensusVec3, CensusVec3, CensusVec3];
}

function barycentric(point: CensusVec3, a: CensusVec3, b: CensusVec3, c: CensusVec3): readonly [number, number, number] | null {
  const e1 = subtract(b, a);
  const e2 = subtract(c, a);
  const q = subtract(point, a);
  const d00 = dot(e1, e1);
  const d01 = dot(e1, e2);
  const d11 = dot(e2, e2);
  const d20 = dot(q, e1);
  const d21 = dot(q, e2);
  const denominator = d00 * d11 - d01 * d01;
  if (!(Math.abs(denominator) > 1e-24)) return null;
  const wb = (d11 * d20 - d01 * d21) / denominator;
  const wc = (d00 * d21 - d01 * d20) / denominator;
  return [1 - wb - wc, wb, wc];
}

function chartFromRecord(field: Field, record: RawRecord): TriangleChart | null {
  if (!record.hit || record.triangleId < 0 || record.triangleId >= field.geometry.triangleCount) return null;
  const [a, b, c] = triangleVertices(field, record.triangleId, record.copyX, record.copyZ);
  const e1 = subtract(b, a);
  const e2 = subtract(c, a);
  const crossProduct = cross(e1, e2);
  const twiceArea = length3(crossProduct);
  if (!(twiceArea > 1e-20)) return null;
  const normal = normalize(crossProduct);
  const denominator = dot(normal, record.direction);
  if (Math.abs(denominator) <= PLANE_POLE_EPSILON) return null;
  const t = dot(normal, subtract(a, record.origin)) / denominator;
  if (!(t >= 0)) return null;
  const canonicalPoint = addScaled(record.origin, record.direction, t);
  const weights = barycentric(canonicalPoint, a, b, c);
  if (!weights || weights.some((weight) => weight < -1e-8)) return null;
  const altitudeA = twiceArea / Math.max(1e-30, length3(subtract(c, b)));
  const altitudeB = twiceArea / Math.max(1e-30, length3(subtract(c, a)));
  const altitudeC = twiceArea / Math.max(1e-30, length3(subtract(b, a)));
  const inTriangleRadius = Math.min(
    weights[0] * altitudeA,
    weights[1] * altitudeB,
    weights[2] * altitudeC,
  );
  const safeRadius = Math.max(0, inTriangleRadius - SUPPORT_GUARD_METRES);
  return {
    triangleId: record.triangleId,
    copyX: record.copyX,
    copyZ: record.copyZ,
    a,
    b,
    c,
    normal,
    canonicalPoint,
    safeRadius,
    category: category(field, record.triangleId),
    atlasTexel: record.atlasTexel,
    sliceIndex: record.sliceIndex,
    canonicalDirection: record.direction,
    canonicalOrigin: record.origin,
  };
}

function encodeOct(normal: CensusVec3): readonly [number, number] {
  const l1 = Math.max(1e-30, Math.abs(normal[0]) + Math.abs(normal[1]) + Math.abs(normal[2]));
  let x = normal[0] / l1;
  let y = normal[1] / l1;
  if (normal[2] < 0) {
    const oldX = x;
    x = (1 - Math.abs(y)) * (oldX >= 0 ? 1 : -1);
    y = (1 - Math.abs(oldX)) * (y >= 0 ? 1 : -1);
  }
  return [
    Math.max(0, Math.min(65535, Math.round((x * 0.5 + 0.5) * 65535))),
    Math.max(0, Math.min(65535, Math.round((y * 0.5 + 0.5) * 65535))),
  ];
}

function decodeOct(x16: number, y16: number): CensusVec3 {
  let x = x16 / 65535 * 2 - 1;
  let y = y16 / 65535 * 2 - 1;
  let z = 1 - Math.abs(x) - Math.abs(y);
  if (z < 0) {
    const oldX = x;
    x = (1 - Math.abs(y)) * (oldX >= 0 ? 1 : -1);
    y = (1 - Math.abs(oldX)) * (y >= 0 ? 1 : -1);
  }
  return normalize([x, y, z]);
}

function packChart(field: Field, chart: TriangleChart): PackedChart {
  const [octX, octY] = encodeOct(chart.normal);
  const yMin = field.geometry.bounds.min[1];
  const ySpan = field.geometry.bounds.max[1] - yMin;
  const hitY = Math.max(0, Math.min(65534, Math.round((chart.canonicalPoint[1] - yMin) / ySpan * 65534)));
  const support = Math.max(0, Math.min(65535, Math.floor(chart.safeRadius / SUPPORT_SCALE_METRES * 65535)));
  const reconstructedY = yMin + hitY / 65534 * ySpan;
  const reconstructedT = (field.geometry.topH - reconstructedY) / -chart.canonicalDirection[1];
  return {
    ...chart,
    packedWords: [((octX << 16) | hitY) >>> 0, ((support << 16) | octY) >>> 0],
    reconstructedNormal: decodeOct(octX, octY),
    reconstructedPoint: addScaled(chart.canonicalOrigin, chart.canonicalDirection, reconstructedT),
    reconstructedRadius: support / 65535 * SUPPORT_SCALE_METRES,
  };
}

function triangleContains(chart: TriangleChart, point: CensusVec3): { inside: boolean; planeDistance: number } {
  const planeDistance = Math.abs(dot(chart.normal, subtract(point, chart.a)));
  const weights = barycentric(point, chart.a, chart.b, chart.c);
  return {
    inside: planeDistance <= PLANE_DISTANCE_EPSILON_METRES
      && !!weights
      && weights.every((weight) => weight >= -1e-7),
    planeDistance,
  };
}

function intersectChart(
  field: Field,
  chart: TriangleChart | PackedChart,
  origin: CensusVec3,
  liveDirection: CensusVec3,
  packed: boolean,
): Candidate | null {
  const normal = packed ? (chart as PackedChart).reconstructedNormal : chart.normal;
  const pointOnPlane = packed ? (chart as PackedChart).reconstructedPoint : chart.canonicalPoint;
  const radius = packed ? (chart as PackedChart).reconstructedRadius : chart.safeRadius;
  const denominator = dot(normal, liveDirection);
  if (Math.abs(denominator) <= PLANE_POLE_EPSILON) return null;
  const t = dot(normal, subtract(pointOnPlane, origin)) / denominator;
  if (!(t >= 0)) return null;
  const point = addScaled(origin, liveDirection, t);
  if (point[1] < field.geometry.bounds.min[1] - BOUNDS_EPSILON || point[1] > field.geometry.topH + BOUNDS_EPSILON) return null;
  const displacement = length3(subtract(point, pointOnPlane));
  const supportAccepted = displacement <= radius + 1e-12;
  const triangle = triangleContains(chart, point);
  return {
    t,
    point,
    chart,
    supportAccepted,
    triangleAccepted: triangle.inside,
    planeDistance: triangle.planeDistance,
    displacement,
  };
}

function directionBracket(field: Field, liveDirection: CensusVec3): readonly [number, number, number, number] {
  const azimuthTurns = positiveModulo(Math.atan2(liveDirection[2], liveDirection[0]), Math.PI * 2)
    / (Math.PI * 2) * field.azimuthCount;
  const azimuth0 = Math.floor(azimuthTurns);
  const azimuth1 = (azimuth0 + 1) % field.azimuthCount;
  const elevation = Math.asin(-liveDirection[1]);
  let elevation0 = field.elevationCount - 2;
  if (elevation <= field.elevationsRadians[0]!) elevation0 = 0;
  else if (elevation >= field.elevationsRadians.at(-1)!) elevation0 = field.elevationCount - 2;
  else {
    for (let index = 0; index < field.elevationCount - 1; index++) {
      if (elevation >= field.elevationsRadians[index]! && elevation <= field.elevationsRadians[index + 1]!) {
        elevation0 = index;
        break;
      }
    }
  }
  return [azimuth0, azimuth1, elevation0, elevation0 + 1];
}

function periodicFirstHit(
  field: Field,
  bvh: TriangleBvh,
  phaseX: number,
  phaseZ: number,
  ray: CensusVec3,
): TruthHit | null {
  const verticalDrop = field.geometry.topH - field.geometry.bounds.min[1];
  const maximumT = verticalDrop / -ray[1];
  const endX = phaseX + ray[0] * maximumT;
  const endZ = phaseZ + ray[2] * maximumT;
  const minCopyX = Math.ceil((Math.min(phaseX, endX) - field.geometry.bounds.max[0]) / field.geometry.tileSizeX - BOUNDS_EPSILON);
  const maxCopyX = Math.floor((Math.max(phaseX, endX) - field.geometry.bounds.min[0]) / field.geometry.tileSizeX + BOUNDS_EPSILON);
  const minCopyZ = Math.ceil((Math.min(phaseZ, endZ) - field.geometry.bounds.max[2]) / field.geometry.tileSizeZ - BOUNDS_EPSILON);
  const maxCopyZ = Math.floor((Math.max(phaseZ, endZ) - field.geometry.bounds.min[2]) / field.geometry.tileSizeZ + BOUNDS_EPSILON);
  let best: Omit<TruthHit, 'point' | 'category'> | null = null;
  for (let copyZ = minCopyZ; copyZ <= maxCopyZ; copyZ++) {
    for (let copyX = minCopyX; copyX <= maxCopyX; copyX++) {
      const localOrigin: CensusVec3 = [
        phaseX - copyX * field.geometry.tileSizeX,
        field.geometry.topH,
        phaseZ - copyZ * field.geometry.tileSizeZ,
      ];
      const hit = bvh.intersectNearest(localOrigin, ray, best?.t ?? maximumT);
      if (
        hit
        && (!best
          || hit.t < best.t - BOUNDS_EPSILON
          || (Math.abs(hit.t - best.t) <= BOUNDS_EPSILON
            && (hit.triangleId < best.triangleId
              || (hit.triangleId === best.triangleId
                && (copyX < best.copyX || (copyX === best.copyX && copyZ < best.copyZ))))))
      ) best = { ...hit, copyX, copyZ };
    }
  }
  if (!best) return null;
  const point = addScaled([phaseX, field.geometry.topH, phaseZ], ray, best.t);
  return { ...best, point, category: category(field, best.triangleId) };
}

function evaluateRay(
  field: Field,
  bvh: TriangleBvh,
  phaseX: number,
  phaseZ: number,
  liveDirection: CensusVec3,
  footprint2x2: boolean,
  packed: boolean,
): Evaluation {
  const truthKey = `${phaseX}|${phaseZ}|${liveDirection[0]}|${liveDirection[1]}|${liveDirection[2]}`;
  let truth = truthCache.get(truthKey);
  if (truth === undefined) {
    truth = periodicFirstHit(field, bvh, phaseX, phaseZ, liveDirection);
    truthCache.set(truthKey, truth);
  }
  const [azimuth0, azimuth1, elevation0, elevation1] = directionBracket(field, liveDirection);
  const phase = phaseCoordinates(field, phaseX, phaseZ);
  const baseX = footprint2x2 ? Math.floor(phase[0]) : Math.round(phase[0]);
  const baseY = footprint2x2 ? Math.floor(phase[1]) : Math.round(phase[1]);
  const offsets = footprint2x2
    ? [[0, 0], [1, 0], [0, 1], [1, 1]] as const
    : [[0, 0]] as const;
  const candidates: Candidate[] = [];
  const supportOnlyCandidates: Candidate[] = [];
  let records = 0;
  let canonicalMiss = 0;
  let invalidCanonicalTriangle = 0;
  let zeroSupport = 0;
  let planePole = 0;
  let outOfBand = 0;
  let supportRejected = 0;
  let triangleRejected = 0;
  let supportOnlyLeaks = 0;
  const canonicalSupports: number[] = [];
  const candidateDisplacements: number[] = [];
  const displacementToSupportRatios: number[] = [];
  for (const azimuth of [azimuth0, azimuth1]) {
    for (const elevation of [elevation0, elevation1]) {
      const targetSlice = sliceIndex(field, azimuth, elevation);
      for (const offset of offsets) {
        records++;
        const record = recordAt(field, targetSlice, baseX + offset[0], baseY + offset[1], phaseX, phaseZ);
        if (!record.hit) {
          canonicalMiss++;
          continue;
        }
        const idealChart = chartFromRecord(field, record);
        if (!idealChart) {
          invalidCanonicalTriangle++;
          continue;
        }
        canonicalSupports.push(idealChart.safeRadius);
        if (!(idealChart.safeRadius > 0)) {
          zeroSupport++;
          continue;
        }
        const chart = packed ? packChart(field, idealChart) : idealChart;
        const denominator = dot(packed ? (chart as PackedChart).reconstructedNormal : chart.normal, liveDirection);
        if (Math.abs(denominator) <= PLANE_POLE_EPSILON) {
          planePole++;
          continue;
        }
        const candidate = intersectChart(field, chart, [phaseX, field.geometry.topH, phaseZ], liveDirection, packed);
        if (!candidate) {
          outOfBand++;
          continue;
        }
        candidateDisplacements.push(candidate.displacement);
        const candidateRadius = packed
          ? (chart as PackedChart).reconstructedRadius
          : chart.safeRadius;
        displacementToSupportRatios.push(candidate.displacement / Math.max(1e-12, candidateRadius));
        if (!candidate.supportAccepted) {
          supportRejected++;
          continue;
        }
        supportOnlyCandidates.push(candidate);
        if (!candidate.triangleAccepted) {
          triangleRejected++;
          supportOnlyLeaks++;
          continue;
        }
        candidates.push(candidate);
      }
    }
  }
  candidates.sort((left, right) => left.t - right.t);
  supportOnlyCandidates.sort((left, right) => left.t - right.t);
  return {
    truth,
    prediction: candidates[0] ?? null,
    supportOnlyPrediction: supportOnlyCandidates[0] ?? null,
    records,
    canonicalMiss,
    invalidCanonicalTriangle,
    zeroSupport,
    planePole,
    outOfBand,
    supportRejected,
    triangleRejected,
    supportOnlyLeaks,
    canonicalSupports,
    candidateDisplacements,
    displacementToSupportRatios,
  };
}

function mutableMetrics(sampleCount: number): MutableMetrics {
  return {
    rays: 0, truthHits: 0, truthMisses: 0, predictedHits: 0, predictedMisses: 0,
    truePositive: 0, trueNegative: 0, falsePositive: 0, falseNegative: 0,
    exactOwner: 0, wrongOwner: 0,
    truthPanicle: 0, truthStructure: 0, exactPanicle: 0, exactStructure: 0,
    categoryPanicle: 0, categoryStructure: 0,
    candidateRecords: 0, canonicalMiss: 0, invalidCanonicalTriangle: 0, zeroSupport: 0,
    planePole: 0, outOfBand: 0, supportRejected: 0, triangleRejected: 0, supportOnlyLeaks: 0,
    positionErrors: [], depthErrors: [], planeDistances: [], acceptedSupports: [], canonicalSupports: [],
    candidateDisplacements: [], displacementToSupportRatios: [],
    truthMask: new Uint8Array(sampleCount), predictionMask: new Uint8Array(sampleCount),
    exactMask: new Uint8Array(sampleCount), wrongMask: new Uint8Array(sampleCount),
  };
}

function accumulate(metrics: MutableMetrics, result: Evaluation, index: number, liveDirection: CensusVec3): void {
  metrics.rays++;
  metrics.candidateRecords += result.records;
  metrics.canonicalMiss += result.canonicalMiss;
  metrics.invalidCanonicalTriangle += result.invalidCanonicalTriangle;
  metrics.zeroSupport += result.zeroSupport;
  metrics.planePole += result.planePole;
  metrics.outOfBand += result.outOfBand;
  metrics.supportRejected += result.supportRejected;
  metrics.triangleRejected += result.triangleRejected;
  metrics.supportOnlyLeaks += result.supportOnlyLeaks;
  metrics.canonicalSupports.push(...result.canonicalSupports);
  metrics.candidateDisplacements.push(...result.candidateDisplacements);
  metrics.displacementToSupportRatios.push(...result.displacementToSupportRatios);
  const truth = result.truth;
  const prediction = result.prediction;
  if (truth) {
    metrics.truthHits++;
    metrics.truthMask[index] = 1;
    if (truth.category === 'panicle') metrics.truthPanicle++;
    else metrics.truthStructure++;
  } else metrics.truthMisses++;
  if (prediction) {
    metrics.predictedHits++;
    metrics.predictionMask[index] = 1;
    metrics.planeDistances.push(prediction.planeDistance);
    metrics.acceptedSupports.push((prediction.chart as PackedChart).reconstructedRadius ?? prediction.chart.safeRadius);
  } else metrics.predictedMisses++;
  if (!truth && !prediction) metrics.trueNegative++;
  else if (!truth && prediction) metrics.falsePositive++;
  else if (truth && !prediction) metrics.falseNegative++;
  else if (truth && prediction) {
    metrics.truePositive++;
    const positionError = length3(subtract(prediction.point, truth.point));
    const depthError = Math.abs(-liveDirection[1] * (prediction.t - truth.t));
    metrics.positionErrors.push(positionError);
    metrics.depthErrors.push(depthError);
    const ownerMatches = prediction.chart.triangleId === truth.triangleId
      && prediction.chart.copyX === truth.copyX
      && prediction.chart.copyZ === truth.copyZ
      && positionError <= HIT_POSITION_EPSILON_METRES;
    if (ownerMatches) {
      metrics.exactOwner++;
      metrics.exactMask[index] = 1;
      if (truth.category === 'panicle') metrics.exactPanicle++;
      else metrics.exactStructure++;
    } else {
      metrics.wrongOwner++;
      metrics.wrongMask[index] = 1;
    }
    if (prediction.chart.category === truth.category) {
      if (truth.category === 'panicle') metrics.categoryPanicle++;
      else metrics.categoryStructure++;
    }
  }
}

function quantiles(values: readonly number[]): Quantiles {
  if (!values.length) return { count: 0, p50: null, p95: null, p99: null, maximum: null };
  const sorted = [...values].sort((a, b) => a - b);
  const at = (fraction: number): number => sorted[Math.min(sorted.length - 1, Math.ceil(fraction * sorted.length) - 1)]!;
  return { count: values.length, p50: at(0.5), p95: at(0.95), p99: at(0.99), maximum: sorted.at(-1)! };
}

function connectedComponents(mask: Uint8Array, width: number, height: number): {
  count: number;
  maximumPixels: number;
  maximumWidthTexels: number;
  maximumHeightTexels: number;
  maximumDiagonalTexels: number;
} {
  const visited = new Uint8Array(mask.length);
  let count = 0;
  let maximumPixels = 0;
  let maximumWidthTexels = 0;
  let maximumHeightTexels = 0;
  let maximumDiagonalTexels = 0;
  for (let seed = 0; seed < mask.length; seed++) {
    if (!mask[seed] || visited[seed]) continue;
    count++;
    const queue = [seed];
    visited[seed] = 1;
    let cursor = 0;
    let pixels = 0;
    let minX = width;
    let maxX = 0;
    let minY = height;
    let maxY = 0;
    while (cursor < queue.length) {
      const index = queue[cursor++]!;
      const x = index % width;
      const y = Math.floor(index / width);
      pixels++;
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      for (const [nx, ny] of [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]] as const) {
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        const neighbour = ny * width + nx;
        if (mask[neighbour] && !visited[neighbour]) {
          visited[neighbour] = 1;
          queue.push(neighbour);
        }
      }
    }
    const componentWidth = maxX - minX + 1;
    const componentHeight = maxY - minY + 1;
    maximumPixels = Math.max(maximumPixels, pixels);
    maximumWidthTexels = Math.max(maximumWidthTexels, componentWidth);
    maximumHeightTexels = Math.max(maximumHeightTexels, componentHeight);
    maximumDiagonalTexels = Math.max(maximumDiagonalTexels, Math.hypot(componentWidth, componentHeight));
  }
  return { count, maximumPixels, maximumWidthTexels, maximumHeightTexels, maximumDiagonalTexels };
}

function summarize(metrics: MutableMetrics, grid: number | null, tileSize: number): Record<string, unknown> {
  const hitAgreement = (metrics.truePositive + metrics.trueNegative) / Math.max(1, metrics.rays);
  const exactRecall = metrics.exactOwner / Math.max(1, metrics.truthHits);
  const falsePositiveMask = metrics.predictionMask.map((predicted, index) => predicted && !metrics.truthMask[index] ? 1 : 0);
  const falseNegativeMask = metrics.truthMask.map((truth, index) => truth && !metrics.predictionMask[index] ? 1 : 0);
  const falsePositiveComponents = grid ? connectedComponents(falsePositiveMask, grid, grid) : null;
  const falseNegativeComponents = grid ? connectedComponents(falseNegativeMask, grid, grid) : null;
  const metresPerTexel = grid ? tileSize / grid : null;
  return {
    rays: metrics.rays,
    truthHits: metrics.truthHits,
    truthMisses: metrics.truthMisses,
    predictedHits: metrics.predictedHits,
    predictedMisses: metrics.predictedMisses,
    truePositive: metrics.truePositive,
    trueNegative: metrics.trueNegative,
    falsePositive: metrics.falsePositive,
    falseNegative: metrics.falseNegative,
    hitMissAgreement: hitAgreement,
    hitPrecision: metrics.predictedHits ? metrics.truePositive / metrics.predictedHits : 1,
    hitRecall: metrics.truthHits ? metrics.truePositive / metrics.truthHits : 1,
    exactFirstOwnerRecall: exactRecall,
    wrongOwnerHits: metrics.wrongOwner,
    wrongOwnerFractionPredicted: metrics.wrongOwner / Math.max(1, metrics.predictedHits),
    category: {
      truthPanicle: metrics.truthPanicle,
      truthStructure: metrics.truthStructure,
      exactPanicleRecall: metrics.exactPanicle / Math.max(1, metrics.truthPanicle),
      exactStructureRecall: metrics.exactStructure / Math.max(1, metrics.truthStructure),
      panicleCategoryRecall: metrics.categoryPanicle / Math.max(1, metrics.truthPanicle),
      structureCategoryRecall: metrics.categoryStructure / Math.max(1, metrics.truthStructure),
    },
    errorsMetres: {
      position: quantiles(metrics.positionErrors),
      verticalDepth: quantiles(metrics.depthErrors),
      sourcePlaneDistance: quantiles(metrics.planeDistances),
    },
    supportMetres: {
      canonical: quantiles(metrics.canonicalSupports),
      accepted: quantiles(metrics.acceptedSupports),
      candidateDisplacement: quantiles(metrics.candidateDisplacements),
      displacementToSupportRatio: quantiles(metrics.displacementToSupportRatios),
    },
    rejection: {
      records: metrics.candidateRecords,
      canonicalMiss: metrics.canonicalMiss,
      invalidCanonicalTriangle: metrics.invalidCanonicalTriangle,
      zeroSupport: metrics.zeroSupport,
      planePole: metrics.planePole,
      outOfBand: metrics.outOfBand,
      supportRejected: metrics.supportRejected,
      triangleRejected: metrics.triangleRejected,
      supportOnlyLeaks: metrics.supportOnlyLeaks,
    },
    connectedSpatialErrors: grid ? {
      grid,
      metresPerTexel,
      falsePositive: {
        ...falsePositiveComponents,
        maximumWidthMetres: falsePositiveComponents!.maximumWidthTexels * metresPerTexel!,
        maximumDiagonalMetres: falsePositiveComponents!.maximumDiagonalTexels * metresPerTexel!,
      },
      falseNegative: {
        ...falseNegativeComponents,
        maximumWidthMetres: falseNegativeComponents!.maximumWidthTexels * metresPerTexel!,
        maximumDiagonalMetres: falseNegativeComponents!.maximumDiagonalTexels * metresPerTexel!,
      },
    } : null,
    premultipliedCoverageInterpretation: {
      noRelocatedGeometry: metrics.falsePositive === 0 && metrics.wrongOwner === 0,
      transparentOmissionFractionOfTruth: metrics.falseNegative / Math.max(1, metrics.truthHits),
      statement: metrics.falsePositive === 0 && metrics.wrongOwner === 0
        ? 'all residual errors are transparent omissions representable by zero premultiplied coverage, but recall still determines whether the holes are visually acceptable'
        : 'premultiplied coverage cannot repair surviving false-positive or wrong-owner relocated geometry',
    },
  };
}

function samplePhases(field: Field, grid: number, offset: number): Array<readonly [number, number]> {
  const phases: Array<readonly [number, number]> = [];
  for (let y = 0; y < grid; y++) {
    for (let x = 0; x < grid; x++) {
      phases.push([
        field.geometry.tileOriginX + wrappedUnit((x + offset) / grid) * field.geometry.tileSizeX,
        field.geometry.tileOriginZ + wrappedUnit((y + offset * 0.6180339887498948) / grid) * field.geometry.tileSizeZ,
      ]);
    }
  }
  return phases;
}

function motionPhases(field: Field): Array<readonly [number, number]> {
  const phases: Array<readonly [number, number]> = [];
  for (let index = 0; index < MOTION_SAMPLES; index++) {
    const t = index / MOTION_SAMPLES;
    phases.push([
      field.geometry.tileOriginX + wrappedUnit(0.071 + t * 0.83) * field.geometry.tileSizeX,
      field.geometry.tileOriginZ + wrappedUnit(0.193 + t * 0.317) * field.geometry.tileSizeZ,
    ]);
  }
  return phases;
}

function canonicalRecordCentrePhases(field: Field, grid: number): Array<readonly [number, number]> {
  const phases: Array<readonly [number, number]> = [];
  for (let y = 0; y < grid; y++) {
    for (let x = 0; x < grid; x++) {
      const texelX = Math.floor((x + 0.5) / grid * field.interiorWidth);
      const texelY = Math.floor((y + 0.5) / grid * field.interiorHeight);
      phases.push([
        field.geometry.tileOriginX + (texelX + 0.5) / field.interiorWidth * field.geometry.tileSizeX,
        field.geometry.tileOriginZ + (1 - (texelY + 0.5) / field.interiorHeight) * field.geometry.tileSizeZ,
      ]);
    }
  }
  return phases;
}

function motionSummary(results: readonly Evaluation[]): Record<string, unknown> {
  let truthHitTransitions = 0;
  let predictedHitTransitions = 0;
  let transitionAgreement = 0;
  let comparableTransitions = 0;
  let spuriousOwnerChanges = 0;
  const stepErrors: number[] = [];
  for (let index = 1; index < results.length; index++) {
    const previous = results[index - 1]!;
    const current = results[index]!;
    const truthTransition = !!previous.truth !== !!current.truth;
    const predictionTransition = !!previous.prediction !== !!current.prediction;
    if (truthTransition) truthHitTransitions++;
    if (predictionTransition) predictedHitTransitions++;
    if (truthTransition === predictionTransition) transitionAgreement++;
    if (previous.truth && current.truth && previous.prediction && current.prediction) {
      comparableTransitions++;
      const truthStep = subtract(current.truth.point, previous.truth.point);
      const predictedStep = subtract(current.prediction.point, previous.prediction.point);
      stepErrors.push(length3(subtract(predictedStep, truthStep)));
      const truthSameOwner = previous.truth.triangleId === current.truth.triangleId;
      const predictedChangedOwner = previous.prediction.chart.triangleId !== current.prediction.chart.triangleId;
      if (truthSameOwner && predictedChangedOwner) spuriousOwnerChanges++;
    }
  }
  return {
    frames: results.length,
    truthHitTransitions,
    predictedHitTransitions,
    hitTransitionAgreement: transitionAgreement / Math.max(1, results.length - 1),
    comparableTransitions,
    spuriousOwnerChanges,
    worldStepErrorMetres: quantiles(stepErrors),
  };
}

function qaPixels(metrics: MutableMetrics, grid: number): Uint8Array {
  const panelWidth = grid;
  const pixels = new Uint8Array(panelWidth * 4 * grid * 4);
  const paint = (panel: number, index: number, color: readonly [number, number, number]): void => {
    const x = index % grid;
    const y = Math.floor(index / grid);
    const at = (y * panelWidth * 4 + panel * panelWidth + x) * 4;
    pixels[at] = color[0]; pixels[at + 1] = color[1]; pixels[at + 2] = color[2]; pixels[at + 3] = 255;
  };
  for (let index = 0; index < metrics.rays; index++) {
    paint(0, index, metrics.truthMask[index] ? [215, 190, 205] : [35, 35, 35]);
    paint(1, index, metrics.predictionMask[index] ? [80, 175, 70] : [35, 35, 35]);
    paint(2, index, metrics.exactMask[index] ? [70, 190, 80]
      : metrics.wrongMask[index] ? [230, 45, 45]
        : metrics.truthMask[index] ? [230, 150, 35]
          : metrics.predictionMask[index] ? [220, 35, 210] : [35, 35, 35]);
    paint(3, index, metrics.truthMask[index] && !metrics.predictionMask[index] ? [240, 150, 25]
      : metrics.predictionMask[index] && !metrics.truthMask[index] ? [230, 25, 210]
        : [35, 35, 35]);
  }
  return pixels;
}

const cases: readonly CaseDefinition[] = [
  { name: 'grazing-5-half-azimuth', azimuthDegrees: 11.25, elevationDegrees: 5, kind: 'grid' },
  { name: 'half-bin-25', azimuthDegrees: 11.25, elevationDegrees: 25, kind: 'grid' },
  { name: 'oblique-45-half-bin', azimuthDegrees: 11.25, elevationDegrees: 45, kind: 'grid' },
  { name: 'high-65-half-bin', azimuthDegrees: 11.25, elevationDegrees: 65, kind: 'grid' },
  { name: 'top-82_5-half-azimuth', azimuthDegrees: 11.25, elevationDegrees: 82.5, kind: 'grid' },
  { name: 'motion-oblique-35-half-azimuth', azimuthDegrees: 11.25, elevationDegrees: 35, kind: 'motion' },
  { name: 'canonical-record-centres-35', azimuthDegrees: 0, elevationDegrees: 35, kind: 'canonical' },
];

const truthCache = new Map<string, TruthHit | null>();

const sourceBytes = readFileSync(SOURCE);
const sourceSha256 = sha256(sourceBytes);
if (sourceSha256 !== EXPECTED_SOURCE_SHA256) throw new Error(`unexpected source SHA-256 ${sourceSha256}`);
const field = decodeField(sourceBytes);
console.error(`[support-plane] building BVH for ${field.geometry.triangleCount.toLocaleString()} GCRP/v4 triangles`);
const bvh = TriangleBvh.build(field.geometry, 8);

const policies = [
  { name: 'four-angular-nearest-phase-ideal', footprint2x2: false, packed: false },
  { name: 'four-angular-nearest-phase-packed8', footprint2x2: false, packed: true },
  { name: 'sixteen-angular-spatial2x2-packed8', footprint2x2: true, packed: true },
] as const;
const allReports: Record<string, unknown> = {};
const qa: Array<{ name: string; pixels: Uint8Array; width: number; height: number; interpretation: string }> = [];

for (const policy of policies) {
  const byCase: Record<string, unknown> = {};
  const aggregate = mutableMetrics(cases.filter((entry) => entry.kind === 'grid').length * SAMPLE_GRID * SAMPLE_GRID);
  for (const definition of cases) {
    const liveDirection = direction(definition.azimuthDegrees, definition.elevationDegrees);
    const phases = definition.kind === 'grid'
      ? samplePhases(field, SAMPLE_GRID, 0.3819660112501051)
      : definition.kind === 'motion'
        ? motionPhases(field)
        : canonicalRecordCentrePhases(field, SAMPLE_GRID);
    const metrics = mutableMetrics(phases.length);
    const results: Evaluation[] = [];
    for (let index = 0; index < phases.length; index++) {
      const phase = phases[index]!;
      const result = evaluateRay(
        field,
        bvh,
        phase[0],
        phase[1],
        liveDirection,
        policy.footprint2x2,
        policy.packed,
      );
      results.push(result);
      accumulate(metrics, result, index, liveDirection);
      if (definition.kind === 'grid') accumulate(aggregate, result, aggregate.rays, liveDirection);
    }
    byCase[definition.name] = {
      definition,
      metrics: summarize(metrics, definition.kind === 'grid' ? SAMPLE_GRID : null, field.geometry.tileSizeX),
      ...(definition.kind === 'motion' ? { motion: motionSummary(results) } : {}),
    };
    console.error(`[support-plane] ${policy.name} ${definition.name}: exact ${metrics.exactOwner}/${metrics.truthHits}, fp ${metrics.falsePositive}, wrong ${metrics.wrongOwner}`);
  }
  allReports[policy.name] = {
    fixedRecordReads: policy.footprint2x2 ? 16 : 4,
    aggregateGridCases: summarize(aggregate, null, field.geometry.tileSizeX),
    byCase,
  };
}

// Dedicated regular QA grids preserve connected fan/hole topology.
for (const definition of [cases[0]!, cases[2]!, cases[4]!]) {
  const liveDirection = direction(definition.azimuthDegrees, definition.elevationDegrees);
  const phases = samplePhases(field, QA_GRID, 0.5);
  const metrics = mutableMetrics(phases.length);
  for (let index = 0; index < phases.length; index++) {
    const phase = phases[index]!;
    const result = evaluateRay(field, bvh, phase[0], phase[1], liveDirection, false, true);
    accumulate(metrics, result, index, liveDirection);
  }
  qa.push({
    name: definition.name,
    pixels: qaPixels(metrics, QA_GRID),
    width: QA_GRID * 4,
    height: QA_GRID,
    interpretation: 'left to right: exact hit mask, supported packed-plane hit mask, exact/wrong/missing event classification, false-negative orange versus false-positive magenta',
  });
}

const primary = allReports['four-angular-nearest-phase-packed8'] as {
  aggregateGridCases: {
    exactFirstOwnerRecall: number;
    falsePositive: number;
    wrongOwnerHits: number;
    premultipliedCoverageInterpretation: { transparentOmissionFractionOfTruth: number };
  };
};
const primaryMetrics = primary.aggregateGridCases;
const packedControl = (allReports['four-angular-nearest-phase-packed8'] as {
  byCase: Record<string, { metrics: { exactFirstOwnerRecall: number } }>;
}).byCase['canonical-record-centres-35']!.metrics;
const idealControl = (allReports['four-angular-nearest-phase-ideal'] as {
  byCase: Record<string, { metrics: { exactFirstOwnerRecall: number } }>;
}).byCase['canonical-record-centres-35']!.metrics;
const canonicalControlPass = packedControl.exactFirstOwnerRecall >= 0.85
  && Math.abs(packedControl.exactFirstOwnerRecall - idealControl.exactFirstOwnerRecall) <= 0.01;
const geometryPass = primaryMetrics.exactFirstOwnerRecall >= 0.95
  && primaryMetrics.falsePositive === 0
  && primaryMetrics.wrongOwnerHits === 0;
const coveragePass = primaryMetrics.premultipliedCoverageInterpretation.transparentOmissionFractionOfTruth <= 0.05;
const costPass = true;
const decision = geometryPass && coveragePass && costPass && canonicalControlPass
  ? 'PROVISIONAL_PASS'
  : 'REJECT_AND_PARK';

const recipe = {
  schema: 'laas-groundcover-support-bounded-canonical-plane/v1',
  sourceSha256,
  cases,
  sampleGrid: SAMPLE_GRID,
  qaGrid: QA_GRID,
  motionSamples: MOTION_SAMPLES,
  supportScaleMetres: SUPPORT_SCALE_METRES,
  supportGuardMetres: SUPPORT_GUARD_METRES,
  hitPositionEpsilonMetres: HIT_POSITION_EPSILON_METRES,
  planeDistanceEpsilonMetres: PLANE_DISTANCE_EPSILON_METRES,
  implementationSha256: sha256(readFileSync(import.meta.filename)),
};
const recipeSha256 = sha256(JSON.stringify(recipe));
const outputRoot = resolve(
  'data/work/groundcover-support-bounded-plane-record',
  sourceSha256.slice(0, 16),
  recipeSha256.slice(0, 16),
);
const qaRoot = resolve(outputRoot, 'qa');
mkdirSync(qaRoot, { recursive: true });
const qaIndex: Array<Record<string, unknown>> = [];
for (let index = 0; index < qa.length; index++) {
  const diagnostic = qa[index]!;
  const file = `${String(index + 1).padStart(3, '0')}-${diagnostic.name}-truth-supported-error.png`;
  const path = resolve(qaRoot, file);
  await sharp(diagnostic.pixels, { raw: { width: diagnostic.width, height: diagnostic.height, channels: 4 } })
    .resize(diagnostic.width * 6, diagnostic.height * 6, { kernel: 'nearest' })
    .png()
    .toFile(path);
  qaIndex.push({
    file,
    sha256: sha256(readFileSync(path)),
    width: diagnostic.width * 6,
    height: diagnostic.height * 6,
    interpretation: diagnostic.interpretation,
  });
}

const report = {
  schema: recipe.schema,
  decision,
  source: {
    file: SOURCE,
    sha256: sourceSha256,
    bytes: sourceBytes.byteLength,
    vertices: field.geometry.vertexCount,
    triangles: field.geometry.triangleCount,
    tileMetres: [field.geometry.tileSizeX, field.geometry.tileSizeZ],
    bounds: field.geometry.bounds,
    lattice: {
      azimuthCount: field.azimuthCount,
      elevationsDegrees: field.elevationsRadians.map((value) => value * 180 / Math.PI),
    },
  },
  recipe,
  mathematicalRecord: {
    canonicalPoint: 'P_i = O(q_i) + d_i * dot(n,A-O(q_i))/dot(n,d_i)',
    exactLiveIntersection: 't = dot(n,P_i-O(q))/dot(n,d_live)',
    safeRadius: 'max(0,min_j distance(P_i, triangle edge_j)-20 micrometres)',
    acceptance: 't positive, in height band, ||P-P_i|| <= safeRadius, and exact offline source-triangle containment',
    selection: 'minimum positive accepted t by fixed compare/select network; complete records are never blended',
    caveat: 'the shader-shaped support test is conservative for the ideal plane; exact triangle containment is retained as an optimistic offline oracle for packed-plane quantisation and is separately counted as supportOnlyLeaks',
  },
  packingAndCost: {
    primaryRecord: {
      bytes: 8,
      word0: 'bits 0..15 hitY UNORM over source Y bounds (65535 reserved miss); bits 16..31 geometric oct-normal X UNORM16',
      word1: 'bits 0..15 geometric oct-normal Y UNORM16; bits 16..31 support radius floor-quantised over [0,64 mm]',
      ownerAndColour: 'winner record atlas coordinate selects one separate premultiplied RGBA8 authored-colour read; no owner interpolation',
    },
    primary: {
      planeReads: 4,
      planeBytes: 32,
      winnerColourReads: 1,
      winnerColourBytes: 4,
      totalBytes: 36,
      scalarShape: 'four unrolled oct decodes, four ray-plane divisions, four squared support tests, and a fixed four-way minimum; approximately 120-150 scalar FMA-equivalent operations plus four divisions',
      controlFlow: 'no loop, march, traversal, candidate table, barrier, pass, dispatch, or binding beyond the plane atlas and existing colour atlas',
      registerPressure: 'four candidate t/valid pairs can be reduced immediately; no four complete colour/normal records remain live',
      locality: 'four angular atlas records at one spatial phase, followed by one colour record at the elected coordinate',
    },
    spatial2x2Variant: {
      planeReads: 16,
      planeBytes: 128,
      winnerColourReads: 1,
      totalBytes: 132,
      scalarShape: 'sixteen unrolled intersections and a fixed reduction; separately measured and outside the requested primary traffic ceiling',
    },
    comparison: 'primary traffic is below four depth + four normal + four colour samples; ALU adds fixed plane divisions/support tests but removes filtered cross-owner blending',
  },
  policies: allReports,
  gates: {
    requiredExactFirstOwnerRecall: 0.95,
    requiredTransparentOmissionFractionMaximum: 0.05,
    requireZeroFalsePositive: true,
    requireZeroWrongOwner: true,
    geometryPass,
    coveragePass,
    costPass,
    canonicalControlPass,
    canonicalControl: {
      idealExactFirstOwnerRecall: idealControl.exactFirstOwnerRecall,
      packedExactFirstOwnerRecall: packedControl.exactFirstOwnerRecall,
      requirement: 'packed record-centre control >= 0.85 and within one percentage point of the f64 ideal',
    },
  },
  interpretation: [
    'Finite support can remove the long false plane fans only by declining to draw when the canonical primitive does not own the live line.',
    'A false negative is safe premultiplied transparency, not relocated geometry. It is nevertheless a visible hole; coverage omission above the five-percent gate is not accepted as grass.',
    'Any support-only leak proves packed plane/support quantisation needs a stronger analytic guard before a runtime path could be considered. The optimistic exact-triangle oracle cannot itself be implemented from the eight-byte record.',
    'The sixteen-read spatial variant is evidence only. It cannot rescue a failed primary while remaining within the accepted traffic shape.',
  ],
  provenance: [
    'Sannikov, Precomputed Raycasting for Efficient Grass and Fur Rendering: fixed precomputed directional ray records and low-cost live reconstruction objective.',
    'Lin and Shum (2004), A Geometric Analysis of Light Field Rendering: depth/geometry-assisted neighbouring-ray reconstruction and disocclusion boundary.',
    'LAAS-original: in-triangle radius certificate, packed eight-byte canonical plane record, fixed four-way winner, connected false-fan/transparent-hole gate on the accepted Calamagrostis GCRP/v4 asset.',
  ],
};
const reportPath = resolve(outputRoot, 'report.json');
writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
writeFileSync(resolve(qaRoot, 'index.json'), `${JSON.stringify({
  schema: 'laas-groundcover-support-bounded-plane-qa-index/v1',
  sourceSha256,
  recipeSha256,
  report: { file: '../report.json', sha256: sha256(readFileSync(reportPath)) },
  images: qaIndex,
}, null, 2)}\n`);
console.error(`[support-plane] ${decision}; wrote ${outputRoot}`);
console.log(outputRoot);
