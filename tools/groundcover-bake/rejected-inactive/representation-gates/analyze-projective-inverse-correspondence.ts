/**
 * Offline gate for a fixed-cost inverse-projective first-event reconstruction.
 *
 * One canonical complete-event lookup supplies vertical depth and the geometric
 * face normal.  Their exact local Jacobian gives a Sherman-Morrison inverse for
 * the canonical phase which could lie on a held-out live ray.  A second
 * categorical lookup is accepted only when its complete event maps back onto
 * that live ray.  There is no runtime implementation in this file.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import sharp from 'sharp';
import {
  TriangleBvh,
  type CensusVec3,
  type DecodedOwnedProfileGeometry,
} from '../../ExteriorClosureCensus';

const SOURCE_ROOT = resolve(
  'data/work/groundcover-bandlimited-botanical-source',
  '968e74302585c74af46f59fadb6c9ff541e4afb1bd3adc42602bf29c5e2ed87f',
  'transport',
);
const EXPECTED_SOURCE_SHA256 = '8cd69c2a6c61043c2861cad3a5ec06fee6dd4346a6722fd01fcc43468071ce5d';
const AZIMUTH_COUNT = 16;
const SOURCE_ELEVATIONS_DEGREES = [5, 15, 35, 55, 75] as const;
const TARGET_AZIMUTHS_DEGREES = [11.25, 68.25] as const;
const TARGET_ELEVATIONS_DEGREES = [0.1, 1, 5, 10, 25, 45, 65, 82.5] as const;
const CANDIDATE_COUNTS = [1, 2, 4, 8] as const;
const SAMPLES_PER_CLASS = 384;
const GRID = 24;
const TOP_MARGIN_METRES = 1e-5;
const PERIOD_METRES = 0.5199999809265137;
const MAXIMUM_RESIDENT_BYTES = 51_121_152;
const STORED_PHASE = 258;
const COMPLETE_RECORD_BYTES = 8;
const STRICT_RESIDUAL_METRES = 2e-5;
const RELAXED_RESIDUAL_METRES = [0.00025, 0.001, 0.004] as const;
const EVENT_EPSILON_METRES = 2e-5;
const DENOMINATOR_EPSILON = 1e-10;

interface SourceMesh {
  geometry: DecodedOwnedProfileGeometry;
  normals: Float32Array;
  colors: Float32Array;
  structureTriangles: Uint32Array;
  plumeTriangles: Uint32Array;
  sourceSha256: string;
}

interface CompleteEvent {
  hit: boolean;
  triangleId: number;
  t: number;
  h: number;
  point: CensusVec3;
  geometricNormal: CensusVec3;
  shadingNormal: CensusVec3;
  color: CensusVec3;
  plume: boolean;
  copyX: number;
  copyZ: number;
}

interface DirectionRecord {
  direction: CensusVec3;
  slope: readonly [number, number];
  azimuthDegrees: number;
  elevationDegrees: number;
}

interface Candidate {
  sourceDirection: DirectionRecord;
  initial: CompleteEvent;
  second: CompleteEvent;
  predictedPhase: readonly [number, number];
  residualMetres: number;
  reconstructedH: number;
  pointErrorMetres: number;
  sameTruthEvent: boolean;
}

interface Counts {
  rays: number;
  truthHits: number;
  truthMisses: number;
  predictedHits: number;
  truePositive: number;
  trueNegative: number;
  falsePositive: number;
  falseNegative: number;
  exactEvents: number;
  exactPanicleEvents: number;
  exactStructureEvents: number;
  truthPanicleEvents: number;
  truthStructureEvents: number;
  oracleRecoverable: number;
  oraclePanicleRecoverable: number;
  oracleStructureRecoverable: number;
  targetVisible: number;
  targetPanicleVisible: number;
  targetStructureVisible: number;
  poleRejects: number;
  firstMissRejects: number;
  secondMissRejects: number;
  residualRejects: number;
  wrongSelfConsistentEvents: number;
  positionErrors: number[];
  residuals: number[];
}

interface EvaluatedRay {
  truth: CompleteEvent;
  attempts: Array<{
    one: Candidate | null;
    two: Candidate | null;
    targetVisible: boolean;
    oneInvalid: 'pole' | 'first_miss' | 'second_miss' | null;
    twoInvalid: 'pole' | 'first_miss' | 'second_miss' | 'third_miss' | null;
  }>;
}

function sha256(value: Uint8Array | string): string {
  return createHash('sha256').update(value).digest('hex');
}

function bounds(values: Float32Array): { min: CensusVec3; max: CensusVec3 } {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let index = 0; index < values.length; index += 3) {
    for (let axis = 0; axis < 3; axis++) {
      min[axis] = Math.min(min[axis]!, values[index + axis]!);
      max[axis] = Math.max(max[axis]!, values[index + axis]!);
    }
  }
  return { min: min as unknown as CensusVec3, max: max as unknown as CensusVec3 };
}

function readFloat32(name: string): Float32Array {
  const bytes = readFileSync(resolve(SOURCE_ROOT, name));
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).slice();
}

function readUint32(name: string): Uint32Array {
  const bytes = readFileSync(resolve(SOURCE_ROOT, name));
  return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).slice();
}

function loadSource(): SourceMesh {
  const sourceIndex = JSON.parse(readFileSync(resolve(SOURCE_ROOT, '../qa/index.json'), 'utf8')) as {
    sourceMeshSha256?: string;
    sourceMesh?: { vertices?: number; triangles?: number; structureTriangles?: number; plumeTriangles?: number };
  };
  if (sourceIndex.sourceMeshSha256 !== EXPECTED_SOURCE_SHA256) {
    throw new Error(`isolated source index identity changed: ${sourceIndex.sourceMeshSha256}`);
  }
  const positions32 = readFloat32('positions.f32');
  const normals = readFloat32('normals.f32');
  const colors = readFloat32('colors.f32');
  const triangles = readUint32('indices.u32');
  const structureIndices = readUint32('structure-indices.u32');
  const plumeIndices = readUint32('plume-indices.u32');
  if (positions32.length !== normals.length || positions32.length !== colors.length) {
    throw new Error('isolated source vertex attributes have different lengths');
  }
  const positions = new Float64Array(positions32);
  if (
    sourceIndex.sourceMesh?.vertices !== positions32.length / 3
    || sourceIndex.sourceMesh.triangles !== triangles.length / 3
    || sourceIndex.sourceMesh.structureTriangles !== structureIndices.length / 3
    || sourceIndex.sourceMesh.plumeTriangles !== plumeIndices.length / 3
  ) throw new Error('isolated source transport no longer matches its immutable machine index');
  const sourceBounds = bounds(positions32);
  const geometry: DecodedOwnedProfileGeometry = {
    version: 4,
    profileId: 2,
    topH: sourceBounds.max[1] + TOP_MARGIN_METRES,
    tileOriginX: 0,
    tileOriginZ: 0,
    tileSizeX: PERIOD_METRES,
    tileSizeZ: PERIOD_METRES,
    bounds: sourceBounds,
    positions,
    triangles,
    vertexCount: positions.length / 3,
    triangleCount: triangles.length / 3,
  };
  const classify = (indexStream: Uint32Array): Uint32Array => {
    const byKey = new Map<string, number[]>();
    for (let triangleId = 0; triangleId < geometry.triangleCount; triangleId++) {
      const base = triangleId * 3;
      const key = [triangles[base]!, triangles[base + 1]!, triangles[base + 2]!].join(':');
      const list = byKey.get(key) ?? [];
      list.push(triangleId);
      byKey.set(key, list);
    }
    const result = new Uint32Array(indexStream.length / 3);
    for (let at = 0; at < indexStream.length; at += 3) {
      const key = [indexStream[at]!, indexStream[at + 1]!, indexStream[at + 2]!].join(':');
      const list = byKey.get(key);
      if (!list?.length) throw new Error(`partition triangle ${key} is absent from source`);
      result[at / 3] = list.shift()!;
    }
    return result;
  };
  return {
    geometry,
    normals,
    colors,
    structureTriangles: classify(structureIndices),
    plumeTriangles: classify(plumeIndices),
    sourceSha256: EXPECTED_SOURCE_SHA256,
  };
}

function direction(azimuthDegrees: number, elevationDegrees: number): CensusVec3 {
  const a = azimuthDegrees * Math.PI / 180;
  const e = elevationDegrees * Math.PI / 180;
  const horizontal = Math.cos(e);
  return [horizontal * Math.cos(a), -Math.sin(e), horizontal * Math.sin(a)];
}

function directionRecord(azimuthDegrees: number, elevationDegrees: number): DirectionRecord {
  const ray = direction(azimuthDegrees, elevationDegrees);
  const vertical = -ray[1];
  return {
    direction: ray,
    slope: [ray[0] / vertical, ray[2] / vertical],
    azimuthDegrees,
    elevationDegrees,
  };
}

function dot(a: CensusVec3, b: CensusVec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function length3(a: CensusVec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}

function normalized(value: CensusVec3): CensusVec3 {
  const inverse = 1 / Math.max(1e-30, length3(value));
  return [value[0] * inverse, value[1] * inverse, value[2] * inverse];
}

function triangleAttributes(
  source: SourceMesh,
  triangleId: number,
  point: CensusVec3,
  viewDirection: CensusVec3,
  copyX: number,
  copyZ: number,
): Pick<CompleteEvent, 'geometricNormal' | 'shadingNormal' | 'color' | 'plume'> {
  const { geometry } = source;
  const base = triangleId * 3;
  const ids = [geometry.triangles[base]!, geometry.triangles[base + 1]!, geometry.triangles[base + 2]!] as const;
  const vertices = ids.map((id) => {
    const at = id * 3;
    return [geometry.positions[at]!, geometry.positions[at + 1]!, geometry.positions[at + 2]!] as CensusVec3;
  });
  const e1: CensusVec3 = [
    vertices[1][0] - vertices[0][0],
    vertices[1][1] - vertices[0][1],
    vertices[1][2] - vertices[0][2],
  ];
  const e2: CensusVec3 = [
    vertices[2][0] - vertices[0][0],
    vertices[2][1] - vertices[0][1],
    vertices[2][2] - vertices[0][2],
  ];
  let geometricNormal = normalized([
    e1[1] * e2[2] - e1[2] * e2[1],
    e1[2] * e2[0] - e1[0] * e2[2],
    e1[0] * e2[1] - e1[1] * e2[0],
  ]);
  if (dot(geometricNormal, viewDirection) > 0) {
    geometricNormal = [-geometricNormal[0], -geometricNormal[1], -geometricNormal[2]];
  }
  const d00 = dot(e1, e1);
  const d01 = dot(e1, e2);
  const d11 = dot(e2, e2);
  const v2: CensusVec3 = [
    point[0] - copyX * source.geometry.tileSizeX - vertices[0][0],
    point[1] - vertices[0][1],
    point[2] - copyZ * source.geometry.tileSizeZ - vertices[0][2],
  ];
  const d20 = dot(v2, e1);
  const d21 = dot(v2, e2);
  const denominator = d00 * d11 - d01 * d01;
  const w1 = denominator === 0 ? 0 : (d11 * d20 - d01 * d21) / denominator;
  const w2 = denominator === 0 ? 0 : (d00 * d21 - d01 * d20) / denominator;
  const weights = [1 - w1 - w2, w1, w2] as const;
  const interpolate = (attribute: Float32Array): CensusVec3 => {
    const result = [0, 0, 0];
    for (let corner = 0; corner < 3; corner++) {
      const at = ids[corner]! * 3;
      for (let axis = 0; axis < 3; axis++) result[axis] += attribute[at + axis]! * weights[corner]!;
    }
    return result as unknown as CensusVec3;
  };
  let shadingNormal = normalized(interpolate(source.normals));
  if (dot(shadingNormal, viewDirection) > 0) {
    shadingNormal = [-shadingNormal[0], -shadingNormal[1], -shadingNormal[2]];
  }
  const color = interpolate(source.colors);
  return {
    geometricNormal,
    shadingNormal,
    color,
    plume: color[0] > color[1],
  };
}

function trace(
  source: SourceMesh,
  bvh: TriangleBvh,
  phase: readonly [number, number],
  ray: CensusVec3,
): CompleteEvent {
  const origin: CensusVec3 = [phase[0], source.geometry.topH, phase[1]];
  const maximumT = (source.geometry.topH - source.geometry.bounds.min[1] + TOP_MARGIN_METRES) / -ray[1];
  let best: { triangleId: number; t: number; copyX: number; copyZ: number } | null = null;
  let bestT = maximumT;
  let cellX = Math.floor(origin[0] / source.geometry.tileSizeX);
  let cellZ = Math.floor(origin[2] / source.geometry.tileSizeZ);
  const stepX = Math.sign(ray[0]);
  const stepZ = Math.sign(ray[2]);
  const firstBoundaryX = (cellX + (stepX > 0 ? 1 : 0)) * source.geometry.tileSizeX;
  const firstBoundaryZ = (cellZ + (stepZ > 0 ? 1 : 0)) * source.geometry.tileSizeZ;
  let nextX = stepX === 0 ? Infinity : Math.max(0, (firstBoundaryX - origin[0]) / ray[0]);
  let nextZ = stepZ === 0 ? Infinity : Math.max(0, (firstBoundaryZ - origin[2]) / ray[2]);
  const deltaX = stepX === 0 ? Infinity : source.geometry.tileSizeX / Math.abs(ray[0]);
  const deltaZ = stepZ === 0 ? Infinity : source.geometry.tileSizeZ / Math.abs(ray[2]);
  const maximumCells = Math.ceil(
    Math.abs(ray[0]) * maximumT / source.geometry.tileSizeX
      + Math.abs(ray[2]) * maximumT / source.geometry.tileSizeZ,
  ) + 4;
  for (let cell = 0; cell < maximumCells; cell++) {
    const localOrigin: CensusVec3 = [
      origin[0] - cellX * source.geometry.tileSizeX,
      origin[1],
      origin[2] - cellZ * source.geometry.tileSizeZ,
    ];
    const local = bvh.intersectNearest(localOrigin, ray, bestT);
    if (local && local.t < bestT) {
      bestT = local.t;
      best = { ...local, copyX: cellX, copyZ: cellZ };
    }
    const next = Math.min(nextX, nextZ);
    if (best && bestT <= next + 1e-10) break;
    if (next > maximumT + 1e-10 || next === Infinity) break;
    if (Math.abs(nextX - next) <= 1e-10) { cellX += stepX; nextX += deltaX; }
    if (Math.abs(nextZ - next) <= 1e-10) { cellZ += stepZ; nextZ += deltaZ; }
  }
  if (!best) {
    return {
      hit: false,
      triangleId: -1,
      t: maximumT,
      h: source.geometry.topH - source.geometry.bounds.min[1],
      point: [origin[0] + ray[0] * maximumT, source.geometry.bounds.min[1], origin[2] + ray[2] * maximumT],
      geometricNormal: [0, 1, 0],
      shadingNormal: [0, 1, 0],
      color: [0, 0, 0],
      plume: false,
      copyX: 0,
      copyZ: 0,
    };
  }
  const point: CensusVec3 = [
    origin[0] + ray[0] * best.t,
    origin[1] + ray[1] * best.t,
    origin[2] + ray[2] * best.t,
  ];
  return {
    hit: true,
    triangleId: best.triangleId,
    t: best.t,
    h: -ray[1] * best.t,
    point,
    copyX: best.copyX,
    copyZ: best.copyZ,
    ...triangleAttributes(source, best.triangleId, point, ray, best.copyX, best.copyZ),
  };
}

function phaseForPoint(source: SourceMesh, point: CensusVec3, live: DirectionRecord): readonly [number, number] {
  const h = source.geometry.topH - point[1];
  const wrap = (value: number): number => value - Math.floor(value / PERIOD_METRES) * PERIOD_METRES;
  return [wrap(point[0] - live.slope[0] * h), wrap(point[2] - live.slope[1] * h)];
}

function projectiveCandidate(
  source: SourceMesh,
  bvh: TriangleBvh,
  phase: readonly [number, number],
  live: DirectionRecord,
  canonical: DirectionRecord,
  truth: CompleteEvent,
): EvaluatedRay['attempts'][number] {
  const deltaSlope = [live.slope[0] - canonical.slope[0], live.slope[1] - canonical.slope[1]] as const;
  const exactTargetPhase = [
    phase[0] + deltaSlope[0] * truth.h,
    phase[1] + deltaSlope[1] * truth.h,
  ] as const;
  const exactTargetSource = truth.hit ? trace(source, bvh, exactTargetPhase, canonical.direction) : null;
  const targetVisible = !!exactTargetSource?.hit && Math.hypot(
    exactTargetSource.point[0] - truth.point[0],
    exactTargetSource.point[1] - truth.point[1],
    exactTargetSource.point[2] - truth.point[2],
  ) <= EVENT_EPSILON_METRES;
  const rejected = (
    oneInvalid: EvaluatedRay['attempts'][number]['oneInvalid'],
    twoInvalid: EvaluatedRay['attempts'][number]['twoInvalid'] = oneInvalid,
    one: Candidate | null = null,
  ): EvaluatedRay['attempts'][number] => ({ one, two: null, targetVisible, oneInvalid, twoInvalid });
  const initial = trace(source, bvh, phase, canonical.direction);
  if (!initial.hit) return rejected('first_miss');
  const n = initial.geometricNormal;
  const canonicalDenominator = n[1] - n[0] * canonical.slope[0] - n[2] * canonical.slope[1];
  const liveDenominator = n[1] - n[0] * live.slope[0] - n[2] * live.slope[1];
  if (Math.abs(canonicalDenominator) <= DENOMINATOR_EPSILON || Math.abs(liveDenominator) <= DENOMINATOR_EPSILON) {
    return rejected('pole');
  }
  // g = grad D_s(q) = n_h / (n_y - n_h.s), and
  // (I - DeltaS g^T)^-1 DeltaS h0 = DeltaS h0/(1-g.DeltaS).
  const hPlane = initial.h * canonicalDenominator / liveDenominator;
  if (!(hPlane > 0) || hPlane > source.geometry.topH - source.geometry.bounds.min[1] + 1e-6) {
    return rejected('pole');
  }
  const predictedPhase = [
    phase[0] + deltaSlope[0] * hPlane,
    phase[1] + deltaSlope[1] * hPlane,
  ] as const;
  const second = trace(source, bvh, predictedPhase, canonical.direction);
  if (!second.hit) return rejected('second_miss');
  const expectedPhase = [
    phase[0] + deltaSlope[0] * second.h,
    phase[1] + deltaSlope[1] * second.h,
  ] as const;
  const residualMetres = Math.hypot(predictedPhase[0] - expectedPhase[0], predictedPhase[1] - expectedPhase[1]);
  const livePoint: CensusVec3 = [
    phase[0] + live.slope[0] * second.h,
    source.geometry.topH - second.h,
    phase[1] + live.slope[1] * second.h,
  ];
  const pointErrorMetres = Math.hypot(
    livePoint[0] - truth.point[0],
    livePoint[1] - truth.point[1],
    livePoint[2] - truth.point[2],
  );
  const one: Candidate = {
      sourceDirection: canonical,
      initial,
      second,
      predictedPhase,
      residualMetres,
      reconstructedH: second.h,
      pointErrorMetres,
      sameTruthEvent: pointErrorMetres <= EVENT_EPSILON_METRES,
  };

  // One permitted normal-updated step. The second complete event defines a
  // new exact local plane at x1. Intersect that plane with the live ray, then
  // issue one final categorical lookup and apply the identical residual gate.
  const n2 = second.geometricNormal;
  const liveDenominator2 = n2[1] - n2[0] * live.slope[0] - n2[2] * live.slope[1];
  if (Math.abs(liveDenominator2) <= DENOMINATOR_EPSILON) return rejected(null, 'pole', one);
  const hPlane2 = (
    n2[0] * (phase[0] - predictedPhase[0])
      + n2[2] * (phase[1] - predictedPhase[1])
      + (n2[1] - n2[0] * canonical.slope[0] - n2[2] * canonical.slope[1]) * second.h
  ) / liveDenominator2;
  if (!(hPlane2 > 0) || hPlane2 > source.geometry.topH - source.geometry.bounds.min[1] + 1e-6) {
    return rejected(null, 'pole', one);
  }
  const predictedPhase2 = [
    phase[0] + deltaSlope[0] * hPlane2,
    phase[1] + deltaSlope[1] * hPlane2,
  ] as const;
  const third = trace(source, bvh, predictedPhase2, canonical.direction);
  if (!third.hit) return rejected(null, 'third_miss', one);
  const expectedPhase2 = [
    phase[0] + deltaSlope[0] * third.h,
    phase[1] + deltaSlope[1] * third.h,
  ] as const;
  const residualMetres2 = Math.hypot(
    predictedPhase2[0] - expectedPhase2[0],
    predictedPhase2[1] - expectedPhase2[1],
  );
  const livePoint2: CensusVec3 = [
    phase[0] + live.slope[0] * third.h,
    source.geometry.topH - third.h,
    phase[1] + live.slope[1] * third.h,
  ];
  const pointErrorMetres2 = Math.hypot(
    livePoint2[0] - truth.point[0],
    livePoint2[1] - truth.point[1],
    livePoint2[2] - truth.point[2],
  );
  return {
    one,
    targetVisible,
    oneInvalid: null,
    twoInvalid: null,
    two: {
      sourceDirection: canonical,
      initial,
      second: third,
      predictedPhase: predictedPhase2,
      residualMetres: residualMetres2,
      reconstructedH: third.h,
      pointErrorMetres: pointErrorMetres2,
      sameTruthEvent: pointErrorMetres2 <= EVENT_EPSILON_METRES,
    },
  };
}

function sourceDirections(): DirectionRecord[] {
  const result: DirectionRecord[] = [];
  for (let azimuth = 0; azimuth < AZIMUTH_COUNT; azimuth++) {
    for (const elevation of SOURCE_ELEVATIONS_DEGREES) {
      result.push(directionRecord(azimuth * 360 / AZIMUTH_COUNT, elevation));
    }
  }
  result.push(directionRecord(0, 90));
  return result;
}

function nearestDirections(live: DirectionRecord, canonical: readonly DirectionRecord[], count: number): DirectionRecord[] {
  return [...canonical].sort((a, b) => dot(b.direction, live.direction) - dot(a.direction, live.direction)).slice(0, count);
}

function quantile(values: readonly number[], q: number): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * q))]!;
}

function emptyCounts(): Counts {
  return {
    rays: 0, truthHits: 0, truthMisses: 0, predictedHits: 0,
    truePositive: 0, trueNegative: 0, falsePositive: 0, falseNegative: 0,
    exactEvents: 0, exactPanicleEvents: 0, exactStructureEvents: 0,
    truthPanicleEvents: 0, truthStructureEvents: 0,
    oracleRecoverable: 0, oraclePanicleRecoverable: 0, oracleStructureRecoverable: 0,
    targetVisible: 0, targetPanicleVisible: 0, targetStructureVisible: 0,
    poleRejects: 0, firstMissRejects: 0, secondMissRejects: 0, residualRejects: 0,
    wrongSelfConsistentEvents: 0, positionErrors: [], residuals: [],
  };
}

function summarize(counts: Counts): Record<string, unknown> {
  const precision = counts.predictedHits ? counts.truePositive / counts.predictedHits : 1;
  const recall = counts.truthHits ? counts.truePositive / counts.truthHits : 1;
  return {
    rays: counts.rays,
    truthHits: counts.truthHits,
    truthMisses: counts.truthMisses,
    predictedHits: counts.predictedHits,
    hitMissAgreement: (counts.truePositive + counts.trueNegative) / Math.max(1, counts.rays),
    hitPrecision: precision,
    hitRecall: recall,
    exactEventRecall: counts.exactEvents / Math.max(1, counts.truthHits),
    panicleExactEventRecall: counts.exactPanicleEvents / Math.max(1, counts.truthPanicleEvents),
    structureExactEventRecall: counts.exactStructureEvents / Math.max(1, counts.truthStructureEvents),
    oracleEventRecoverability: counts.oracleRecoverable / Math.max(1, counts.truthHits),
    oraclePanicleRecoverability: counts.oraclePanicleRecoverable / Math.max(1, counts.truthPanicleEvents),
    oracleStructureRecoverability: counts.oracleStructureRecoverable / Math.max(1, counts.truthStructureEvents),
    targetEventVisibility: counts.targetVisible / Math.max(1, counts.truthHits),
    targetPanicleVisibility: counts.targetPanicleVisible / Math.max(1, counts.truthPanicleEvents),
    targetStructureVisibility: counts.targetStructureVisible / Math.max(1, counts.truthStructureEvents),
    falsePositive: counts.falsePositive,
    falseNegative: counts.falseNegative,
    wrongSelfConsistentEvents: counts.wrongSelfConsistentEvents,
    invalid: {
      pole: counts.poleRejects,
      firstMiss: counts.firstMissRejects,
      secondMiss: counts.secondMissRejects,
      residual: counts.residualRejects,
    },
    acceptedPositionErrorMetres: {
      p50: quantile(counts.positionErrors, 0.5),
      p95: quantile(counts.positionErrors, 0.95),
      maximum: quantile(counts.positionErrors, 1),
    },
    acceptedResidualMetres: {
      p50: quantile(counts.residuals, 0.5),
      p95: quantile(counts.residuals, 0.95),
      maximum: quantile(counts.residuals, 1),
    },
  };
}

function deterministicPoint(source: SourceMesh, triangleId: number, sequence: number): CensusVec3 {
  const triangle = triangleId * 3;
  const hash = (value: number): number => {
    let x = value >>> 0;
    x ^= x >>> 16; x = Math.imul(x, 0x7feb352d); x ^= x >>> 15; x = Math.imul(x, 0x846ca68b); x ^= x >>> 16;
    return (x >>> 0) / 0x1_0000_0000;
  };
  const r1 = Math.sqrt(hash(sequence * 2 + 0x51a7));
  const r2 = hash(sequence * 2 + 0xc0de);
  const weights = [1 - r1, r1 * (1 - r2), r1 * r2] as const;
  const point = [0, 0, 0];
  for (let corner = 0; corner < 3; corner++) {
    const vertex = source.geometry.triangles[triangle + corner]! * 3;
    for (let axis = 0; axis < 3; axis++) point[axis] += source.geometry.positions[vertex + axis]! * weights[corner]!;
  }
  return point as unknown as CensusVec3;
}

function samplePhases(source: SourceMesh, live: DirectionRecord): Array<{ phase: readonly [number, number]; image: readonly [number, number] }> {
  const result: Array<{ phase: readonly [number, number]; image: readonly [number, number] }> = [];
  const addClass = (triangles: Uint32Array, salt: number): void => {
    for (let sample = 0; sample < SAMPLES_PER_CLASS; sample++) {
      const triangle = triangles[(sample * 2654435761 + salt) % triangles.length]!;
      const point = deterministicPoint(source, triangle, sample + salt);
      result.push({ phase: phaseForPoint(source, point, live), image: [-1, -1] });
    }
  };
  addClass(source.structureTriangles, 17);
  addClass(source.plumeTriangles, 104729);
  for (let y = 0; y < GRID; y++) {
    for (let x = 0; x < GRID; x++) {
      result.push({
        phase: [
          (x + 0.5) / GRID * source.geometry.tileSizeX,
          (y + 0.5) / GRID * source.geometry.tileSizeZ,
        ],
        image: [x, y],
      });
    }
  }
  return result;
}

function evaluatePolicy(
  rays: readonly EvaluatedRay[],
  candidateCount: number,
  residualThreshold: number,
  steps: 1 | 2,
): Counts {
  const counts = emptyCounts();
  for (const ray of rays) {
    counts.rays++;
    if (ray.truth.hit) {
      counts.truthHits++;
      if (ray.truth.plume) counts.truthPanicleEvents++; else counts.truthStructureEvents++;
    } else counts.truthMisses++;
    const attempts = ray.attempts.slice(0, candidateCount);
    for (const attempt of attempts) {
      const invalid = steps === 1 ? attempt.oneInvalid : attempt.twoInvalid;
      if (invalid === 'pole') counts.poleRejects++;
      else if (invalid === 'first_miss') counts.firstMissRejects++;
      else if (invalid === 'second_miss' || invalid === 'third_miss') counts.secondMissRejects++;
    }
    const candidates = attempts.flatMap((attempt) => {
      const candidate = steps === 1 ? attempt.one : attempt.two;
      return candidate ? [candidate] : [];
    });
    const targetVisible = attempts.some((attempt) => attempt.targetVisible);
    if (ray.truth.hit && targetVisible) {
      counts.targetVisible++;
      if (ray.truth.plume) counts.targetPanicleVisible++; else counts.targetStructureVisible++;
    }
    const oracle = candidates.some((candidate) => candidate.sameTruthEvent);
    if (ray.truth.hit && oracle) {
      counts.oracleRecoverable++;
      if (ray.truth.plume) counts.oraclePanicleRecoverable++; else counts.oracleStructureRecoverable++;
    }
    const accepted = candidates
      .filter((candidate) => candidate.residualMetres <= residualThreshold)
      .sort((a, b) => a.reconstructedH - b.reconstructedH);
    counts.residualRejects += candidates.length - accepted.length;
    const predicted = accepted[0] ?? null;
    if (predicted) {
      counts.predictedHits++;
      counts.residuals.push(predicted.residualMetres);
      counts.positionErrors.push(predicted.pointErrorMetres);
      if (ray.truth.hit) {
        counts.truePositive++;
        if (predicted.sameTruthEvent) {
          counts.exactEvents++;
          if (ray.truth.plume) counts.exactPanicleEvents++; else counts.exactStructureEvents++;
        } else counts.wrongSelfConsistentEvents++;
      } else counts.falsePositive++;
    } else if (ray.truth.hit) counts.falseNegative++;
    else counts.trueNegative++;
  }
  return counts;
}

const source = loadSource();
console.error(`[projective-inverse] building BVH for ${source.geometry.triangleCount.toLocaleString()} isolated-source triangles`);
const bvh = TriangleBvh.build(source.geometry, 8);
const canonical = sourceDirections();
const maximumCandidates = Math.max(...CANDIDATE_COUNTS);
const raysByAngle = new Map<string, EvaluatedRay[]>();
const qaPixels = new Uint8Array(GRID * GRID * 4);
qaPixels.fill(255);
let qaCaptured = false;

for (const elevation of TARGET_ELEVATIONS_DEGREES) {
  for (const azimuth of TARGET_AZIMUTHS_DEGREES) {
    const live = directionRecord(azimuth, elevation);
    const nearest = nearestDirections(live, canonical, maximumCandidates);
    const rays: EvaluatedRay[] = [];
    const phases = samplePhases(source, live);
    for (const sample of phases) {
      const truth = trace(source, bvh, sample.phase, live.direction);
      const attempts: EvaluatedRay['attempts'] = [];
      for (const sourceDirection of nearest) {
        attempts.push(projectiveCandidate(source, bvh, sample.phase, live, sourceDirection, truth));
      }
      rays.push({ truth, attempts });
      if (!qaCaptured && elevation === 10 && azimuth === TARGET_AZIMUTHS_DEGREES[0] && sample.image[0] >= 0) {
        const [x, y] = sample.image;
        const candidates = attempts.flatMap((attempt) => attempt.one ? [attempt.one] : []);
        const exact = candidates.some((candidate) => candidate.sameTruthEvent);
        const accepted = candidates
          .filter((candidate) => candidate.residualMetres <= STRICT_RESIDUAL_METRES)
          .sort((a, b) => a.reconstructedH - b.reconstructedH)[0];
        const at = (y * GRID + x) * 4;
        if (!truth.hit) {
          qaPixels[at] = 65; qaPixels[at + 1] = 65; qaPixels[at + 2] = 65;
        } else if (accepted?.sameTruthEvent) {
          qaPixels[at] = truth.plume ? 230 : 50;
          qaPixels[at + 1] = truth.plume ? 205 : 180;
          qaPixels[at + 2] = truth.plume ? 195 : 70;
        } else if (exact) {
          qaPixels[at] = 230; qaPixels[at + 1] = 145; qaPixels[at + 2] = 40;
        } else {
          qaPixels[at] = 210; qaPixels[at + 1] = 35; qaPixels[at + 2] = 45;
        }
      }
    }
    raysByAngle.set(`${azimuth}|${elevation}`, rays);
    if (elevation === 10 && azimuth === TARGET_AZIMUTHS_DEGREES[0]) qaCaptured = true;
    console.error(`[projective-inverse] completed azimuth ${azimuth} elevation ${elevation}`);
  }
}

const allRays = [...raysByAngle.values()].flat();
const controlDirection = directionRecord(0, 15);
const controlRays: EvaluatedRay[] = samplePhases(source, controlDirection).map((sample) => {
  const truth = trace(source, bvh, sample.phase, controlDirection.direction);
  return {
    truth,
    attempts: [projectiveCandidate(source, bvh, sample.phase, controlDirection, controlDirection, truth)],
  };
});
const makePolicies = (steps: 1 | 2, candidateCounts: readonly number[]) => Object.fromEntries(candidateCounts.map((candidateCount) => {
  const thresholds = [STRICT_RESIDUAL_METRES, ...RELAXED_RESIDUAL_METRES];
  return [String(candidateCount), Object.fromEntries(thresholds.map((threshold) => [
    String(threshold),
    {
      aggregate: summarize(evaluatePolicy(allRays, candidateCount, threshold, steps)),
      byElevation: Object.fromEntries(TARGET_ELEVATIONS_DEGREES.map((elevation) => [
        String(elevation),
        summarize(evaluatePolicy(
          [...raysByAngle.entries()].filter(([key]) => key.endsWith(`|${elevation}`)).flatMap(([, value]) => value),
          candidateCount,
          threshold,
          steps,
        )),
      ])),
    },
  ]))];
}));
const policies = makePolicies(1, CANDIDATE_COUNTS);
const secondStepPolicies = makePolicies(2, [1, 2, 4]);

const eightStrict = evaluatePolicy(allRays, 8, STRICT_RESIDUAL_METRES, 1);
const fourTwoStepStrict = evaluatePolicy(allRays, 4, STRICT_RESIDUAL_METRES, 2);
const exteriorPass = (value: Counts): boolean => value.exactEvents / Math.max(1, value.truthHits) >= 0.95
  && value.exactPanicleEvents / Math.max(1, value.truthPanicleEvents) >= 0.95
  && value.exactStructureEvents / Math.max(1, value.truthStructureEvents) >= 0.95;
const exactExteriorPass = exteriorPass(eightStrict) || exteriorPass(fourTwoStepStrict);
// A top-plane phase has no finite exact-horizontal limit. The previously proved
// first-cell split resolves only the local prefix; the suffix still requires a
// separate boundary pointed-line field. This candidate supplies neither that
// field nor the missing origin-height coordinate.
const horizontalInsidePass = false;
const decision = exactExteriorPass && horizontalInsidePass ? 'PROVISIONAL_PASS' : 'REJECT_AND_PARK';

const directionCount = canonical.length;
const storedRecords = directionCount * STORED_PHASE * STORED_PHASE;
const recipe = {
  schema: 'laas-groundcover-projective-inverse-correspondence/v2',
  sourceSha256: source.sourceSha256,
  sourceDirections: { azimuthCount: AZIMUTH_COUNT, elevationsDegrees: SOURCE_ELEVATIONS_DEGREES, verticalSingleton: true },
  targetAzimuthsDegrees: TARGET_AZIMUTHS_DEGREES,
  targetElevationsDegrees: TARGET_ELEVATIONS_DEGREES,
  candidateCounts: CANDIDATE_COUNTS,
  samplesPerClass: SAMPLES_PER_CLASS,
  grid: GRID,
  thresholdsMetres: [STRICT_RESIDUAL_METRES, ...RELAXED_RESIDUAL_METRES],
  periodicTileMetres: PERIOD_METRES,
  implementationSha256: sha256(readFileSync(import.meta.filename)),
};
const recipeSha256 = sha256(JSON.stringify(recipe));
const outputRoot = resolve('data/work/groundcover-projective-inverse-correspondence', source.sourceSha256.slice(0, 16), recipeSha256.slice(0, 16));
const qaRoot = resolve(outputRoot, 'qa');
mkdirSync(qaRoot, { recursive: true });
const qaFile = resolve(qaRoot, '001-heldout-10deg-k8-strict.png');
await sharp(qaPixels, { raw: { width: GRID, height: GRID, channels: 4 } })
  .resize(GRID * 16, GRID * 16, { kernel: 'nearest' })
  .png()
  .toFile(qaFile);

const report = {
  schema: recipe.schema,
  decision,
  source: {
    root: SOURCE_ROOT,
    sha256: source.sourceSha256,
    vertices: source.geometry.vertexCount,
    triangles: source.geometry.triangleCount,
    structureTriangles: source.structureTriangles.length,
    plumeTriangles: source.plumeTriangles.length,
    bounds: source.geometry.bounds,
  },
  method: {
    fixedPoint: 'x = q + (s_live - s_i) D_i(x)',
    jacobian: 'grad D_i = n_xz / (n_y - n_xz dot s_i)',
    inverse: 'x1 = q + DeltaS*h0/(1 - gradD dot DeltaS), the rank-one Sherman-Morrison inverse',
    secondLookup: 'one complete categorical first-event lookup at x1',
    permittedSecondUpdate: 'K4 only: update the local plane from the second record, issue a third complete categorical lookup, and apply the same residual identity; 12 total loads',
    validation: 'accept only when ||x1 - q - DeltaS*h1|| <= epsilon; attributes remain coupled to that one second event',
    selection: 'minimum positive h1 among residual-valid fixed candidates',
    generosity: 'continuous f64 source field, exact geometric face normal, no phase quantisation, no normal/depth packing error',
    signCheck: 'P=x+s_i*h=q+s_live*h, therefore x=q+(s_live-s_i)*h; this is the sign used by both the target-visibility oracle and each inverse update',
  },
  domain: {
    exterior: recipe,
    exactHorizontalAndInside: {
      result: 'FAIL_BY_DOMAIN',
      reason: 'top-plane slope and phase diverge at d_y=0; a camera-inside successor additionally depends on pointed-line origin height. The exact first-cell prefix does not supply the boundary suffix field.',
    },
  },
  storageAndCost: {
    directionCount,
    storedPhase: STORED_PHASE,
    storedRecords,
    completeRecordBytes: COMPLETE_RECORD_BYTES,
    residentBytes: storedRecords * COMPLETE_RECORD_BYTES,
    maximumResidentBytes: MAXIMUM_RESIDENT_BYTES,
    withinResidentCap: storedRecords * COMPLETE_RECORD_BYTES <= MAXIMUM_RESIDENT_BYTES,
    eightCandidateRuntimeShape: '16 categorical record loads, unrolled; no loop/march/traversal. Each candidate needs one division plus rank-one phase ALU and residual. This is a mathematical ceiling, not an approved shader cost.',
    payloadBoundary: '8 bytes can hold depth16 + geometric-oct16 + shading-oct16 + material/coverage16. Continuous premultiplied plume colour or a larger community mark would require a different packing or lookup.',
  },
  policies,
  secondNormalUpdatedPolicies: secondStepPolicies,
  canonicalDirectionSanityControl: {
    oneStep: summarize(evaluatePolicy(controlRays, 1, STRICT_RESIDUAL_METRES, 1)),
    twoStep: summarize(evaluatePolicy(controlRays, 1, STRICT_RESIDUAL_METRES, 2)),
    requirement: 'same live/source direction must reproduce every true hit exactly and preserve every miss',
  },
  gates: {
    requiredExteriorExactEventRecall: 0.95,
    exactExteriorPass,
    k8OneStepPass: exteriorPass(eightStrict),
    k4TwoStepPass: exteriorPass(fourTwoStepStrict),
    exactHorizontalAndInsideRequired: true,
    horizontalInsidePass,
  },
  interpretation: [
    'A strict residual-valid event lies on the live ray; this avoids stretching an unrelated infinite plane.',
    'Target-event visibility asks only whether the exact truth event is first-visible in any candidate canonical direction, using the exact target phase. Newton-basin recoverability separately asks whether the closed-form iterations can find it from q.',
    'Oracle event recoverability in each policy is the Newton-basin upper bound after one or two inverse updates, before the minimum-depth selector; an implementable selector cannot exceed it.',
    'Failure may arise from disocclusion, a first lookup miss, Newton basin failure, or the missing pointed-line coordinate. Looser residuals trade misses for wrong self-consistent events and are not correctness.',
    'The method is exact within one retained planar first-event chart. It is not an all-angle visibility representation for the accepted finite branching source.',
  ],
  provenance: [
    'Sherman and Morrison (1950), Adjustment of an Inverse Matrix Corresponding to a Change in One Element of a Given Matrix.',
    'Lin and Shum (2004), A Geometric Analysis of Light Field Rendering: geometry-assisted neighbouring-ray reconstruction and disocclusion limits.',
    'LAAS-original: complete-event residual identity, fixed-candidate actual-source gate, pointed-line composition boundary, and byte/read accounting.',
  ],
};
const reportPath = resolve(outputRoot, 'report.json');
writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
const index = {
  schema: 'laas-groundcover-projective-inverse-correspondence-qa-index/v1',
  sourceSha256: source.sourceSha256,
  recipeSha256,
  report: { file: 'report.json', sha256: sha256(readFileSync(reportPath)) },
  images: [{
    file: '001-heldout-10deg-k8-strict.png',
    sha256: sha256(readFileSync(qaFile)),
    width: GRID * 16,
    height: GRID * 16,
    interpretation: 'grey target miss; green/pale strict exact structure/plume event; orange oracle-recoverable but selector miss; red unrecoverable or wrong event',
  }],
};
writeFileSync(resolve(qaRoot, 'index.json'), `${JSON.stringify(index, null, 2)}\n`);
console.error(`[projective-inverse] ${decision}; wrote ${outputRoot}`);
console.log(outputRoot);
