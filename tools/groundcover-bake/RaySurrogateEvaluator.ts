/**
 * Pure-offline evaluator for fixed-read reconstructions of a periodic GCRP/v4
 * first-hit field. The mesh BVH is analytic truth only; none of this code is a
 * runtime candidate path and it deliberately does not enumerate owner closures.
 */

import {
  TriangleBvh,
  decodeOwnedProfileGeometry,
  type CensusVec3,
  type DecodedOwnedProfileGeometry,
} from './ExteriorClosureCensus';

export type SurrogateName =
  | 'projected_path_relift'
  | 'nearest_complete_tau'
  | 'nearest_complete_vertical_drop'
  | 'phase_reprojected_vertical_drop'
  | 'phase_reprojected_affine_fixed_point'
  | 'one_record_face_plane'
  | 'reciprocal_depth_slope_triangle';

interface RayFieldSlice {
  direction: CensusVec3;
  elevationRadians: number;
  azimuthRadians: number;
  depthMin: number;
  depthMax: number;
}

interface RawRecord {
  hit: boolean;
  t: number;
  owner: number;
  triangleId: number | null;
  copyX: number;
  copyZ: number;
  direction: CensusVec3;
  sliceIndex: number;
  texelX: number;
  texelY: number;
  originX: number;
  originZ: number;
}

interface SpatialSample {
  coverage: number;
  inverseProjectedPath: number;
  verticalDrop: number;
}

interface DecodedPeriodicRayField {
  bytes: Uint8Array;
  view: DataView;
  geometry: DecodedOwnedProfileGeometry;
  storedTileWidth: number;
  storedTileHeight: number;
  interiorTileWidth: number;
  interiorTileHeight: number;
  atlasColumns: number;
  atlasRows: number;
  gutter: number;
  geometryOffset: number;
  ownerOffset: number;
  slices: readonly RayFieldSlice[];
  azimuthCount: number;
  elevationCount: number;
  elevationsRadians: readonly number[];
  maximumProjectedTiles: number;
  missInverseProjectedPath: number;
  evaluationElevationsRadians: readonly number[];
  evaluationMaximumProjectedTiles: number;
  freshRecordCache: Map<string, RawRecord>;
}

export interface RaySurrogateEvaluationOptions {
  phaseGrid: number;
  phaseOffset: number;
  azimuthCount: number;
  azimuthOffset: number;
  elevationsDegrees: readonly number[];
  planePoleThreshold?: number;
  coverageThreshold?: number;
  /** Independent small grid for the byte-matched N=32/N=64 ring probes. */
  denserPhaseGrid?: number;
}

export interface ErrorQuantiles {
  count: number;
  signedMean: number | null;
  absolute: {
    p50: number | null;
    p90: number | null;
    p95: number | null;
    p99: number | null;
    maximum: number | null;
  };
}

export interface MethodMetrics {
  rays: number;
  truthHits: number;
  truthMisses: number;
  predictedHits: number;
  predictedMisses: number;
  truePositive: number;
  trueNegative: number;
  falsePositive: number;
  falseNegative: number;
  hitMissAgreement: number;
  hitPrecision: number;
  hitRecall: number;
  verticalDropErrorMetres: ErrorQuantiles;
  worldPositionErrorMetres: ErrorQuantiles;
  invalid: Record<string, number>;
  diagnostic: Record<string, number>;
}

export interface RaySurrogateEvaluationReport {
  domain: {
    phaseGrid: number;
    phaseOffset: number;
    azimuthCount: number;
    azimuthOffset: number;
    elevationsDegrees: readonly number[];
    rayCount: number;
    canonicalAzimuthCount: number;
    canonicalElevationsDegrees: readonly number[];
    evaluationElevationsDegrees: readonly number[];
    generatedCanonicalRowsDegrees: readonly number[];
    generatedCanonicalRecordCount: number;
    verticalDropHorizon: number;
    planePoleThreshold: number;
    coverageThreshold: number;
  };
  accelerator: typeof TriangleBvh.prototype.metrics;
  traceMilliseconds: number;
  methods: Record<SurrogateName, {
    aggregate: MethodMetrics;
    byElevation: Record<string, MethodMetrics>;
  }>;
  denserAzimuthRings: Record<string, {
    azimuthCount: number;
    phaseResolution: number;
    directionRecordCount: number;
    storedAtlasRecordCount: number;
    storedAtlasBytes: number;
    generatedSampleRecords: number;
    sampleRayCount: number;
    traceMilliseconds: number;
    aggregate: MethodMetrics;
    byElevation: Record<string, MethodMetrics>;
  }>;
  atlasSizing: {
    bytesPerRecord: number;
    gutter: number;
    currentShipped: AtlasSizing;
    correctedGrid: readonly AtlasSizing[];
    byteMatchedConfigurations: readonly AtlasSizing[];
  };
  runtimeCostModel: Record<SurrogateName, {
    filteredTextureReads: number;
    nearestTextureReads: number;
    effectiveSourceTexels: number;
    majorScalarOps: readonly string[];
    payloadAssumption: string;
  }>;
  interpretation: readonly string[];
}

export interface AtlasSizing {
  azimuthCount: number;
  phaseResolution: number;
  ringCount: number;
  verticalSingletonCount: number;
  directionRecordCount: number;
  interiorAtlasRecordCount: number;
  storedAtlasRecordCount: number;
  storedAtlasBytes: number;
}

interface Prediction {
  hit: boolean;
  t?: number;
  invalid?: string;
  diagnostic?: readonly string[];
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
  verticalSigned: number[];
  positionSigned: number[];
  invalid: Map<string, number>;
  diagnostic: Map<string, number>;
}

interface SlopeVertex {
  azimuthIndex: number;
  elevationIndex: number;
  x: number;
  z: number;
}

interface DirectionAddress {
  azimuthIndex: number;
  elevationIndex: number;
  direction: CensusVec3;
}

interface ExactCategoricalLattice {
  azimuthCount: number;
  phaseResolution: number;
  cache: Map<string, RawRecord>;
}

export interface SlopeTriangleSelection {
  vertices: readonly [SlopeVertex, SlopeVertex, SlopeVertex];
  weights: readonly [number, number, number];
  extrapolated: boolean;
}

const HEADER_BYTES = 128;
const SLICE_BYTES = 64;
const TEXEL_BYTES = 8;
const OWNER_BYTES = 4;
const MISS_OWNER = 0xffff_ffff;
const BOUNDS_EPSILON = 1e-12;

function positiveInteger(value: number, label: string): number {
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${label} must be a positive integer`);
  return value;
}

function finiteFloat(view: DataView, offset: number, label: string): number {
  const value = view.getFloat32(offset, true);
  if (!Number.isFinite(value)) throw new Error(`${label} is not finite`);
  return value;
}

function positiveModulo(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus;
}

function wrappedUnit(value: number): number {
  return value - Math.floor(value);
}

function unitDirection(azimuthRadians: number, elevationRadians: number): CensusVec3 {
  const horizontal = Math.cos(elevationRadians);
  return [
    horizontal * Math.cos(azimuthRadians),
    -Math.sin(elevationRadians),
    horizontal * Math.sin(azimuthRadians),
  ];
}

function decodeRayField(bytes: Uint8Array): DecodedPeriodicRayField {
  const geometry = decodeOwnedProfileGeometry(bytes);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const storedTileWidth = positiveInteger(view.getUint32(12, true), 'stored tile width');
  const storedTileHeight = positiveInteger(view.getUint32(16, true), 'stored tile height');
  const atlasColumns = positiveInteger(view.getUint32(20, true), 'atlas columns');
  const atlasRows = positiveInteger(view.getUint32(24, true), 'atlas rows');
  const sliceCount = positiveInteger(view.getUint32(28, true), 'slice count');
  const geometryOffset = view.getUint32(36, true);
  const interiorTileWidth = positiveInteger(view.getUint32(44, true), 'interior tile width');
  const interiorTileHeight = positiveInteger(view.getUint32(48, true), 'interior tile height');
  const gutter = positiveInteger(view.getUint32(52, true), 'gutter');
  const ownerOffset = view.getUint32(80, true);
  if (geometryOffset !== HEADER_BYTES + sliceCount * SLICE_BYTES) {
    throw new Error('GCRP/v4 geometry offset is not canonical');
  }
  const atlasTexels = storedTileWidth * storedTileHeight * atlasColumns * atlasRows;
  if (ownerOffset !== geometryOffset + atlasTexels * TEXEL_BYTES) {
    throw new Error('GCRP/v4 owner offset is not canonical');
  }
  const slices: RayFieldSlice[] = [];
  for (let sliceIndex = 0; sliceIndex < sliceCount; sliceIndex++) {
    const offset = HEADER_BYTES + sliceIndex * SLICE_BYTES;
    const direction: CensusVec3 = [
      finiteFloat(view, offset, `slice ${sliceIndex} direction X`),
      finiteFloat(view, offset + 4, `slice ${sliceIndex} direction Y`),
      finiteFloat(view, offset + 8, `slice ${sliceIndex} direction Z`),
    ];
    slices.push({
      direction,
      azimuthRadians: positiveModulo(Math.atan2(direction[2], direction[0]), Math.PI * 2),
      elevationRadians: Math.asin(-direction[1]),
      depthMin: finiteFloat(view, offset + 12, `slice ${sliceIndex} depth minimum`),
      depthMax: finiteFloat(view, offset + 16, `slice ${sliceIndex} depth maximum`),
    });
  }
  let elevationCount = 1;
  while (
    elevationCount < slices.length
    && Math.abs(slices[elevationCount]!.azimuthRadians - slices[0]!.azimuthRadians) < 2e-4
  ) elevationCount++;
  if (elevationCount < 2 || sliceCount % elevationCount !== 0) {
    throw new Error('evaluator requires the azimuth-major regular GCRP/v4 lattice');
  }
  const azimuthCount = sliceCount / elevationCount;
  const elevationsRadians = slices.slice(0, elevationCount).map((slice) => slice.elevationRadians);
  for (let azimuthIndex = 0; azimuthIndex < azimuthCount; azimuthIndex++) {
    for (let elevationIndex = 0; elevationIndex < elevationCount; elevationIndex++) {
      const slice = slices[azimuthIndex * elevationCount + elevationIndex]!;
      const expectedAzimuth = azimuthIndex * Math.PI * 2 / azimuthCount;
      const azimuthError = Math.abs(
        positiveModulo(slice.azimuthRadians - expectedAzimuth + Math.PI, Math.PI * 2) - Math.PI,
      );
      if (
        azimuthError > 2e-4
        || Math.abs(slice.elevationRadians - elevationsRadians[elevationIndex]!) > 2e-4
      ) throw new Error('GCRP/v4 direction lattice is not regular');
    }
  }
  const maximumProjectedTiles = Math.max(...slices.map((slice) =>
    slice.depthMax * Math.hypot(slice.direction[0], slice.direction[2]) / geometry.tileSizeX));
  const evaluationElevationsRadians = [
    5 * Math.PI / 180,
    ...elevationsRadians,
    Math.PI / 2,
  ];
  const evaluationMaximumProjectedTiles = Math.max(
    maximumProjectedTiles,
    (geometry.topH - geometry.bounds.min[1])
      / Math.tan(evaluationElevationsRadians[0]!) / geometry.tileSizeX,
  );
  return {
    bytes,
    view,
    geometry,
    storedTileWidth,
    storedTileHeight,
    interiorTileWidth,
    interiorTileHeight,
    atlasColumns,
    atlasRows,
    gutter,
    geometryOffset,
    ownerOffset,
    slices,
    azimuthCount,
    elevationCount,
    elevationsRadians,
    maximumProjectedTiles,
    missInverseProjectedPath: 1 / (1 + evaluationMaximumProjectedTiles),
    evaluationElevationsRadians,
    evaluationMaximumProjectedTiles,
    freshRecordCache: new Map(),
  };
}

function sliceIndex(field: DecodedPeriodicRayField, azimuthIndex: number, elevationIndex: number): number {
  return positiveModulo(azimuthIndex, field.azimuthCount) * field.elevationCount + elevationIndex;
}

function atlasTexelIndex(
  field: DecodedPeriodicRayField,
  targetSlice: number,
  interiorX: number,
  interiorY: number,
): number {
  const column = targetSlice % field.atlasColumns;
  const row = Math.floor(targetSlice / field.atlasColumns);
  const atlasWidth = field.storedTileWidth * field.atlasColumns;
  const x = column * field.storedTileWidth + field.gutter
    + positiveModulo(interiorX, field.interiorTileWidth);
  const y = row * field.storedTileHeight + field.gutter
    + positiveModulo(interiorY, field.interiorTileHeight);
  return y * atlasWidth + x;
}

function rawRecordAt(
  field: DecodedPeriodicRayField,
  targetSlice: number,
  interiorX: number,
  interiorY: number,
): RawRecord {
  const texel = atlasTexelIndex(field, targetSlice, interiorX, interiorY);
  const payload = field.geometryOffset + texel * TEXEL_BYTES;
  const depthU16 = field.view.getUint16(payload, true);
  const coverageU16 = field.view.getUint16(payload + 6, true);
  const slice = field.slices[targetSlice]!;
  const hit = coverageU16 > 32767 && depthU16 < 65535;
  const depth01 = depthU16 / 65535;
  const t = slice.depthMin + depth01 * (slice.depthMax - slice.depthMin);
  const owner = field.view.getUint32(field.ownerOffset + texel * OWNER_BYTES, true);
  return {
    hit,
    t,
    owner,
    triangleId: owner === MISS_OWNER ? null : owner & 0x3f_ffff,
    copyX: owner === MISS_OWNER ? 0 : ((owner >>> 22) & 0x1f) - 16,
    copyZ: owner === MISS_OWNER ? 0 : ((owner >>> 27) & 0x1f) - 16,
    direction: slice.direction,
    sliceIndex: targetSlice,
    texelX: positiveModulo(interiorX, field.interiorTileWidth),
    texelY: positiveModulo(interiorY, field.interiorTileHeight),
    originX: field.geometry.tileOriginX
      + (positiveModulo(interiorX, field.interiorTileWidth) + 0.5)
        / field.interiorTileWidth * field.geometry.tileSizeX,
    originZ: field.geometry.tileOriginZ
      + (1 - (positiveModulo(interiorY, field.interiorTileHeight) + 0.5)
        / field.interiorTileHeight) * field.geometry.tileSizeZ,
  };
}

function evaluationDirection(
  field: DecodedPeriodicRayField,
  azimuthIndex: number,
  evaluationElevationIndex: number,
): CensusVec3 {
  const elevation = field.evaluationElevationsRadians[evaluationElevationIndex]!;
  if (Math.abs(elevation - Math.PI / 2) <= 1e-12) return [0, -1, 0];
  return unitDirection(
    positiveModulo(azimuthIndex, field.azimuthCount) * Math.PI * 2 / field.azimuthCount,
    elevation,
  );
}

function evaluationRecordAt(
  field: DecodedPeriodicRayField,
  bvh: TriangleBvh,
  azimuthIndex: number,
  evaluationElevationIndex: number,
  interiorX: number,
  interiorY: number,
): RawRecord {
  const storedElevationIndex = evaluationElevationIndex - 1;
  if (storedElevationIndex >= 0 && storedElevationIndex < field.elevationCount) {
    return rawRecordAt(
      field,
      sliceIndex(field, azimuthIndex, storedElevationIndex),
      interiorX,
      interiorY,
    );
  }
  const x = positiveModulo(interiorX, field.interiorTileWidth);
  const y = positiveModulo(interiorY, field.interiorTileHeight);
  const isVertical = evaluationElevationIndex === field.evaluationElevationsRadians.length - 1;
  const canonicalAzimuth = isVertical ? 0 : positiveModulo(azimuthIndex, field.azimuthCount);
  const key = `${evaluationElevationIndex}:${canonicalAzimuth}:${x}:${y}`;
  const cached = field.freshRecordCache.get(key);
  if (cached) return cached;
  const originX = field.geometry.tileOriginX
    + (x + 0.5) / field.interiorTileWidth * field.geometry.tileSizeX;
  const originZ = field.geometry.tileOriginZ
    + (1 - (y + 0.5) / field.interiorTileHeight) * field.geometry.tileSizeZ;
  const direction = evaluationDirection(field, canonicalAzimuth, evaluationElevationIndex);
  const exact = periodicFirstHit(field.geometry, bvh, originX, originZ, direction);
  const record: RawRecord = {
    hit: exact !== null,
    t: exact?.t ?? (field.geometry.topH - field.geometry.bounds.min[1]) / -direction[1],
    owner: MISS_OWNER,
    triangleId: exact?.triangleId ?? null,
    copyX: exact?.copyX ?? 0,
    copyZ: exact?.copyZ ?? 0,
    direction,
    sliceIndex: -1,
    texelX: x,
    texelY: y,
    originX,
    originZ,
  };
  field.freshRecordCache.set(key, record);
  return record;
}

function phaseCoordinates(field: DecodedPeriodicRayField, phaseX: number, phaseZ: number): {
  x: number;
  y: number;
} {
  const u = wrappedUnit((phaseX - field.geometry.tileOriginX) / field.geometry.tileSizeX);
  const v = 1 - wrappedUnit((phaseZ - field.geometry.tileOriginZ) / field.geometry.tileSizeZ);
  return {
    x: u * field.interiorTileWidth - 0.5,
    y: v * field.interiorTileHeight - 0.5,
  };
}

function nearestEvaluationDirection(
  field: DecodedPeriodicRayField,
  direction: CensusVec3,
): DirectionAddress {
  let nearestAzimuth = 0;
  let nearestElevation = 0;
  let nearestDot = Number.NEGATIVE_INFINITY;
  for (let elevationIndex = 0; elevationIndex < field.evaluationElevationsRadians.length; elevationIndex++) {
    const azimuthSamples = elevationIndex === field.evaluationElevationsRadians.length - 1
      ? 1
      : field.azimuthCount;
    for (let azimuthIndex = 0; azimuthIndex < azimuthSamples; azimuthIndex++) {
      const canonical = evaluationDirection(field, azimuthIndex, elevationIndex);
      const dot = canonical[0] * direction[0] + canonical[1] * direction[1] + canonical[2] * direction[2];
      if (dot > nearestDot) {
        nearestDot = dot;
        nearestAzimuth = azimuthIndex;
        nearestElevation = elevationIndex;
      }
    }
  }
  return {
    azimuthIndex: nearestAzimuth,
    elevationIndex: nearestElevation,
    direction: evaluationDirection(field, nearestAzimuth, nearestElevation),
  };
}

function evaluationNearestPhaseRecord(
  field: DecodedPeriodicRayField,
  bvh: TriangleBvh,
  address: Pick<DirectionAddress, 'azimuthIndex' | 'elevationIndex'>,
  phaseX: number,
  phaseZ: number,
): RawRecord {
  const phase = phaseCoordinates(field, phaseX, phaseZ);
  return evaluationRecordAt(
    field,
    bvh,
    address.azimuthIndex,
    address.elevationIndex,
    Math.round(phase.x),
    Math.round(phase.y),
  );
}

function nearestRecord(
  field: DecodedPeriodicRayField,
  bvh: TriangleBvh,
  phaseX: number,
  phaseZ: number,
  direction: CensusVec3,
): RawRecord {
  return evaluationNearestPhaseRecord(
    field,
    bvh,
    nearestEvaluationDirection(field, direction),
    phaseX,
    phaseZ,
  );
}

function spatialSample(
  field: DecodedPeriodicRayField,
  bvh: TriangleBvh,
  azimuthIndex: number,
  evaluationElevationIndex: number,
  phaseX: number,
  phaseZ: number,
): SpatialSample {
  const phase = phaseCoordinates(field, phaseX, phaseZ);
  const x0 = Math.floor(phase.x);
  const y0 = Math.floor(phase.y);
  const fx = phase.x - x0;
  const fy = phase.y - y0;
  const weights = [
    (1 - fx) * (1 - fy),
    fx * (1 - fy),
    (1 - fx) * fy,
    fx * fy,
  ] as const;
  const coordinates = [
    [x0, y0],
    [x0 + 1, y0],
    [x0, y0 + 1],
    [x0 + 1, y0 + 1],
  ] as const;
  const direction = evaluationDirection(field, azimuthIndex, evaluationElevationIndex);
  const horizontal = Math.hypot(direction[0], direction[2]);
  let coverage = 0;
  let inverseCarrier = 0;
  let verticalDropCarrier = 0;
  for (let corner = 0; corner < 4; corner++) {
    const record = evaluationRecordAt(
      field,
      bvh,
      azimuthIndex,
      evaluationElevationIndex,
      coordinates[corner]![0],
      coordinates[corner]![1],
    );
    const weight = weights[corner]!;
    const hitCoverage = record.hit ? 1 : 0;
    const projectedTiles = record.hit
      ? record.t * horizontal / field.geometry.tileSizeX
      : field.evaluationMaximumProjectedTiles;
    inverseCarrier += weight / (1 + projectedTiles);
    coverage += weight * hitCoverage;
    if (record.hit) verticalDropCarrier += weight * (-direction[1] * record.t);
  }
  const safeCoverage = Math.max(coverage, 1 / 65535);
  return {
    coverage,
    inverseProjectedPath: Math.max(
      1 / 65535,
      Math.min(
        1,
        (inverseCarrier - (1 - coverage) * field.missInverseProjectedPath) / safeCoverage,
      ),
    ),
    verticalDrop: verticalDropCarrier / safeCoverage,
  };
}

function directionBracket(field: DecodedPeriodicRayField, direction: CensusVec3): {
  azimuth0: number;
  azimuth1: number;
  azimuthMix: number;
  elevation0: number;
  elevation1: number;
  elevationMix: number;
} {
  const azimuthTurns = positiveModulo(Math.atan2(direction[2], direction[0]), Math.PI * 2)
    / (Math.PI * 2) * field.azimuthCount;
  const azimuth0 = Math.floor(azimuthTurns);
  const elevation = Math.asin(-direction[1]);
  let elevation0 = field.evaluationElevationsRadians.length - 2;
  for (let index = 0; index < field.evaluationElevationsRadians.length - 1; index++) {
    if (elevation < field.evaluationElevationsRadians[index + 1]!) {
      elevation0 = index;
      break;
    }
  }
  const elevation1 = elevation0 + 1;
  return {
    azimuth0,
    azimuth1: (azimuth0 + 1) % field.azimuthCount,
    azimuthMix: azimuthTurns - azimuth0,
    elevation0,
    elevation1,
    elevationMix: Math.max(0, Math.min(1,
      (elevation - field.evaluationElevationsRadians[elevation0]!)
      / (field.evaluationElevationsRadians[elevation1]! - field.evaluationElevationsRadians[elevation0]!),
    )),
  };
}

function predictProjectedPath(
  field: DecodedPeriodicRayField,
  bvh: TriangleBvh,
  phaseX: number,
  phaseZ: number,
  direction: CensusVec3,
  coverageThreshold: number,
): Prediction {
  const bracket = directionBracket(field, direction);
  const s00 = spatialSample(field, bvh, bracket.azimuth0, bracket.elevation0, phaseX, phaseZ);
  const s10 = spatialSample(field, bvh, bracket.azimuth1, bracket.elevation0, phaseX, phaseZ);
  const s01 = spatialSample(field, bvh, bracket.azimuth0, bracket.elevation1, phaseX, phaseZ);
  const s11 = spatialSample(field, bvh, bracket.azimuth1, bracket.elevation1, phaseX, phaseZ);
  const oneAzimuth = 1 - bracket.azimuthMix;
  const oneElevation = 1 - bracket.elevationMix;
  const weights = [
    oneAzimuth * oneElevation * s00.coverage,
    bracket.azimuthMix * oneElevation * s10.coverage,
    oneAzimuth * bracket.elevationMix * s01.coverage,
    bracket.azimuthMix * bracket.elevationMix * s11.coverage,
  ] as const;
  const samples = [s00, s10, s01, s11] as const;
  const totalWeight = weights.reduce((sum, value) => sum + value, 0);
  if (!(totalWeight > coverageThreshold)) return { hit: false };
  let inverseProjectedPath = 0;
  for (let index = 0; index < samples.length; index++) {
    inverseProjectedPath += samples[index]!.inverseProjectedPath * weights[index]!;
  }
  inverseProjectedPath /= totalWeight;
  const projectedTiles = 1 / Math.max(1 / 65535, inverseProjectedPath) - 1;
  const liveHorizontal = Math.hypot(direction[0], direction[2]);
  if (!(liveHorizontal > 1e-8)) return { hit: false, invalid: 'vertical_projected_path' };
  if (!(projectedTiles < field.evaluationMaximumProjectedTiles * 0.97)) return { hit: false };
  return { hit: true, t: projectedTiles * field.geometry.tileSizeX / liveHorizontal };
}

function predictNearestTau(record: RawRecord): Prediction {
  return record.hit ? { hit: true, t: record.t } : { hit: false };
}

function predictNearestVerticalDrop(record: RawRecord, liveDirection: CensusVec3): Prediction {
  if (!record.hit) return { hit: false };
  const verticalDrop = -record.direction[1] * record.t;
  return { hit: true, t: verticalDrop / -liveDirection[1] };
}

function categoricalOwnerKey(record: RawRecord): string | null {
  return record.triangleId === null
    ? null
    : `${record.triangleId}:${record.copyX}:${record.copyZ}`;
}

/** Two fixed categorical reads of one canonical slope. Read zero supplies the
 * predictor h0=D_i(q). Read one applies the exact same-hit phase identity
 * q_i=q+(s-s_i)h0 and supplies h1=D_i(q_i). E1 uses h1 directly. E2 treats the
 * two samples of F(h)=D_i(q+(s-s_i)h) as affine and solves h=F(h):
 * h*=h0^2/(2h0-h1). No filtered depth or owner mixture participates. */
function predictPhaseReprojectedVerticalDrop(
  field: DecodedPeriodicRayField,
  bvh: TriangleBvh,
  phaseX: number,
  phaseZ: number,
  liveDirection: CensusVec3,
  verticalDropHorizon: number,
): {
  iteration: Prediction;
  affine: Prediction;
} {
  const address = nearestEvaluationDirection(field, liveDirection);
  const predictor = evaluationNearestPhaseRecord(field, bvh, address, phaseX, phaseZ);
  if (!predictor.hit) return {
    iteration: { hit: false, diagnostic: ['phase_predictor_miss'] },
    affine: { hit: false, diagnostic: ['phase_predictor_miss'] },
  };
  const h0 = -address.direction[1] * predictor.t;
  const liveSlopeX = liveDirection[0] / -liveDirection[1];
  const liveSlopeZ = liveDirection[2] / -liveDirection[1];
  const canonicalSlopeX = address.direction[0] / -address.direction[1];
  const canonicalSlopeZ = address.direction[2] / -address.direction[1];
  const reprojectedPhaseX = phaseX + (liveSlopeX - canonicalSlopeX) * h0;
  const reprojectedPhaseZ = phaseZ + (liveSlopeZ - canonicalSlopeZ) * h0;
  const corrected = evaluationNearestPhaseRecord(
    field,
    bvh,
    address,
    reprojectedPhaseX,
    reprojectedPhaseZ,
  );
  const diagnostics: string[] = [];
  if (categoricalOwnerKey(predictor) !== categoricalOwnerKey(corrected)) {
    diagnostics.push('phase_reprojection_changed_owner');
  }
  if (!corrected.hit) {
    return {
      iteration: { hit: false, diagnostic: [...diagnostics, 'phase_reprojected_miss'] },
      affine: { hit: false, diagnostic: [...diagnostics, 'phase_reprojected_miss'] },
    };
  }
  const h1 = -address.direction[1] * corrected.t;
  const iteration: Prediction = h1 >= 0 && h1 <= verticalDropHorizon
    ? { hit: true, t: h1 / -liveDirection[1], diagnostic: diagnostics }
    : { hit: false, invalid: 'phase_reprojected_out_of_band', diagnostic: diagnostics };
  const denominator = 2 * h0 - h1;
  let affine: Prediction;
  if (Math.abs(denominator) <= 1e-8) {
    affine = { hit: false, invalid: 'phase_affine_pole', diagnostic: diagnostics };
  } else {
    const hAffine = h0 * h0 / denominator;
    affine = hAffine >= 0 && hAffine <= verticalDropHorizon
      ? { hit: true, t: hAffine / -liveDirection[1], diagnostic: diagnostics }
      : { hit: false, invalid: 'phase_affine_out_of_band', diagnostic: diagnostics };
  }
  return { iteration, affine };
}

function triangleNormal(geometry: DecodedOwnedProfileGeometry, triangleId: number): CensusVec3 | null {
  if (triangleId < 0 || triangleId >= geometry.triangleCount) return null;
  const triangle = triangleId * 3;
  const ia = geometry.triangles[triangle]! * 3;
  const ib = geometry.triangles[triangle + 1]! * 3;
  const ic = geometry.triangles[triangle + 2]! * 3;
  const e1x = geometry.positions[ib]! - geometry.positions[ia]!;
  const e1y = geometry.positions[ib + 1]! - geometry.positions[ia + 1]!;
  const e1z = geometry.positions[ib + 2]! - geometry.positions[ia + 2]!;
  const e2x = geometry.positions[ic]! - geometry.positions[ia]!;
  const e2y = geometry.positions[ic + 1]! - geometry.positions[ia + 1]!;
  const e2z = geometry.positions[ic + 2]! - geometry.positions[ia + 2]!;
  const nx = e1y * e2z - e1z * e2y;
  const ny = e1z * e2x - e1x * e2z;
  const nz = e1x * e2y - e1y * e2x;
  const length = Math.hypot(nx, ny, nz);
  return length > 1e-20 ? [nx / length, ny / length, nz / length] : null;
}

function predictedPointOutsideOwnerTriangle(
  field: DecodedPeriodicRayField,
  record: RawRecord,
  point: CensusVec3,
): boolean {
  const triangleId = record.triangleId;
  if (triangleId === null || triangleId >= field.geometry.triangleCount) return true;
  const copyX = record.copyX;
  const copyZ = record.copyZ;
  const triangle = triangleId * 3;
  const ia = field.geometry.triangles[triangle]! * 3;
  const ib = field.geometry.triangles[triangle + 1]! * 3;
  const ic = field.geometry.triangles[triangle + 2]! * 3;
  const ax = field.geometry.positions[ia]! + copyX * field.geometry.tileSizeX;
  const ay = field.geometry.positions[ia + 1]!;
  const az = field.geometry.positions[ia + 2]! + copyZ * field.geometry.tileSizeZ;
  const e1: CensusVec3 = [
    field.geometry.positions[ib]! + copyX * field.geometry.tileSizeX - ax,
    field.geometry.positions[ib + 1]! - ay,
    field.geometry.positions[ib + 2]! + copyZ * field.geometry.tileSizeZ - az,
  ];
  const e2: CensusVec3 = [
    field.geometry.positions[ic]! + copyX * field.geometry.tileSizeX - ax,
    field.geometry.positions[ic + 1]! - ay,
    field.geometry.positions[ic + 2]! + copyZ * field.geometry.tileSizeZ - az,
  ];
  const q: CensusVec3 = [point[0] - ax, point[1] - ay, point[2] - az];
  const d00 = e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2];
  const d01 = e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2];
  const d11 = e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2];
  const d20 = q[0] * e1[0] + q[1] * e1[1] + q[2] * e1[2];
  const d21 = q[0] * e2[0] + q[1] * e2[1] + q[2] * e2[2];
  const denominator = d00 * d11 - d01 * d01;
  if (!(Math.abs(denominator) > 1e-24)) return true;
  const u = (d11 * d20 - d01 * d21) / denominator;
  const v = (d00 * d21 - d01 * d20) / denominator;
  const w = 1 - u - v;
  return u < -2e-4 || v < -2e-4 || w < -2e-4;
}

function predictFacePlane(
  field: DecodedPeriodicRayField,
  record: RawRecord,
  phaseX: number,
  phaseZ: number,
  direction: CensusVec3,
  verticalDropHorizon: number,
  planePoleThreshold: number,
): Prediction {
  if (!record.hit) return { hit: false };
  if (record.triangleId === null) return { hit: false, invalid: 'missing_owner' };
  const triangleId = record.triangleId;
  const normal = triangleNormal(field.geometry, triangleId);
  if (!normal) return { hit: false, invalid: 'invalid_owner_triangle' };
  const canonical = record.direction;
  const storedHit: CensusVec3 = [
    record.originX + canonical[0] * record.t,
    field.geometry.topH + canonical[1] * record.t,
    record.originZ + canonical[2] * record.t,
  ];
  const denominator = normal[0] * direction[0] + normal[1] * direction[1] + normal[2] * direction[2];
  if (Math.abs(denominator) <= planePoleThreshold) {
    return { hit: false, invalid: 'plane_pole' };
  }
  const numerator = normal[0] * (storedHit[0] - phaseX)
    + normal[1] * (storedHit[1] - field.geometry.topH)
    + normal[2] * (storedHit[2] - phaseZ);
  const t = numerator / denominator;
  const verticalDrop = -direction[1] * t;
  if (!(t >= 0) || !(verticalDrop >= 0 && verticalDrop <= verticalDropHorizon)) {
    return { hit: false, invalid: 'plane_out_of_band' };
  }
  const point: CensusVec3 = [
    phaseX + direction[0] * t,
    field.geometry.topH + direction[1] * t,
    phaseZ + direction[2] * t,
  ];
  return {
    hit: true,
    t,
    diagnostic: predictedPointOutsideOwnerTriangle(field, record, point)
      ? ['outside_source_triangle']
      : [],
  };
}

function barycentric2(
  p: readonly [number, number],
  a: readonly [number, number],
  b: readonly [number, number],
  c: readonly [number, number],
): readonly [number, number, number] | null {
  const denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
  if (!(Math.abs(denominator) > 1e-14)) return null;
  const wa = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / denominator;
  const wb = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / denominator;
  return [wa, wb, 1 - wa - wb];
}

function extrapolationPenalty(weights: readonly [number, number, number]): number {
  return weights.reduce((sum, value) => sum + Math.max(0, -value), 0);
}

/** Select one triangle in the canonical Cartesian slope lattice. The two edge
 * radial bands are deliberately extrapolated; the caller reports those rays. */
export function selectSlopeTriangle(
  azimuthCount: number,
  elevationsRadians: readonly number[],
  direction: CensusVec3,
): SlopeTriangleSelection {
  if (azimuthCount < 3 || elevationsRadians.length < 2) throw new Error('slope lattice is too small');
  const slopeX = direction[0] / -direction[1];
  const slopeZ = direction[2] / -direction[1];
  const radius = Math.hypot(slopeX, slopeZ);
  const radii = elevationsRadians.map((elevation) => 1 / Math.tan(elevation));
  let outerElevation = radii.length - 2;
  let extrapolated = false;
  if (radius >= radii[0]!) {
    outerElevation = 0;
    extrapolated = radius > radii[0]! + 1e-12;
  } else if (radius <= radii.at(-1)!) {
    outerElevation = radii.length - 2;
    extrapolated = radius < radii.at(-1)! - 1e-12;
  } else {
    for (let index = 0; index < radii.length - 1; index++) {
      if (radius <= radii[index]! && radius >= radii[index + 1]!) {
        outerElevation = index;
        break;
      }
    }
  }
  const azimuthTurns = positiveModulo(Math.atan2(slopeZ, slopeX), Math.PI * 2)
    / (Math.PI * 2) * azimuthCount;
  const azimuth0 = Math.floor(azimuthTurns);
  const azimuth1 = (azimuth0 + 1) % azimuthCount;
  const makeVertex = (azimuthIndex: number, elevationIndex: number): SlopeVertex => {
    const azimuth = azimuthIndex * Math.PI * 2 / azimuthCount;
    const r = radii[elevationIndex]!;
    return { azimuthIndex, elevationIndex, x: r * Math.cos(azimuth), z: r * Math.sin(azimuth) };
  };
  const o0 = makeVertex(azimuth0, outerElevation);
  const o1 = makeVertex(azimuth1, outerElevation);
  const i0 = makeVertex(azimuth0, outerElevation + 1);
  const i1 = makeVertex(azimuth1, outerElevation + 1);
  const candidates = [
    [o0, o1, i0],
    [o1, i1, i0],
  ] as const;
  let best: SlopeTriangleSelection | null = null;
  let bestPenalty = Number.POSITIVE_INFINITY;
  for (const vertices of candidates) {
    const weights = barycentric2(
      [slopeX, slopeZ],
      [vertices[0].x, vertices[0].z],
      [vertices[1].x, vertices[1].z],
      [vertices[2].x, vertices[2].z],
    );
    if (!weights) continue;
    const penalty = extrapolationPenalty(weights);
    if (penalty < bestPenalty) {
      bestPenalty = penalty;
      best = { vertices, weights, extrapolated };
    }
  }
  if (!best) throw new Error('slope lattice triangle is degenerate');
  return best;
}

function predictReciprocalDepth(
  field: DecodedPeriodicRayField,
  bvh: TriangleBvh,
  phaseX: number,
  phaseZ: number,
  direction: CensusVec3,
  verticalDropHorizon: number,
  coverageThreshold: number,
): Prediction {
  const selection = selectSlopeTriangle(
    field.azimuthCount,
    field.evaluationElevationsRadians,
    direction,
  );
  let inverseDropCarrier = 0;
  let coverage = 0;
  for (let index = 0; index < 3; index++) {
    const vertex = selection.vertices[index]!;
    const sample = spatialSample(
      field,
      bvh,
      vertex.azimuthIndex,
      vertex.elevationIndex,
      phaseX,
      phaseZ,
    );
    const coveredWeight = selection.weights[index]! * sample.coverage;
    if (coveredWeight > 0 && !(sample.verticalDrop > 1e-8)) {
      return {
        hit: false,
        invalid: 'reciprocal_zero_depth',
        diagnostic: selection.extrapolated ? ['slope_extrapolation'] : [],
      };
    }
    if (coveredWeight > 0) inverseDropCarrier += coveredWeight / sample.verticalDrop;
    coverage += coveredWeight;
  }
  if (!(coverage > coverageThreshold)) {
    return {
      hit: false,
      diagnostic: selection.extrapolated ? ['slope_extrapolation'] : [],
    };
  }
  const inverseDrop = inverseDropCarrier / coverage;
  if (!(inverseDrop > 1e-8) || !Number.isFinite(inverseDrop)) {
    return {
      hit: false,
      invalid: 'reciprocal_pole',
      diagnostic: selection.extrapolated ? ['slope_extrapolation'] : [],
    };
  }
  const verticalDrop = 1 / inverseDrop;
  if (!(verticalDrop >= 0 && verticalDrop <= verticalDropHorizon)) {
    return {
      hit: false,
      invalid: 'reciprocal_out_of_band',
      diagnostic: selection.extrapolated ? ['slope_extrapolation'] : [],
    };
  }
  return {
    hit: true,
    t: verticalDrop / -direction[1],
    diagnostic: selection.extrapolated ? ['slope_extrapolation'] : [],
  };
}

function periodicFirstHit(
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  phaseX: number,
  phaseZ: number,
  direction: CensusVec3,
): { triangleId: number; copyX: number; copyZ: number; t: number } | null {
  const verticalDrop = geometry.topH - geometry.bounds.min[1];
  const maximumT = verticalDrop / -direction[1];
  const endX = phaseX + direction[0] * maximumT;
  const endZ = phaseZ + direction[2] * maximumT;
  const rayMinX = Math.min(phaseX, endX);
  const rayMaxX = Math.max(phaseX, endX);
  const rayMinZ = Math.min(phaseZ, endZ);
  const rayMaxZ = Math.max(phaseZ, endZ);
  const minCopyX = Math.ceil((rayMinX - geometry.bounds.max[0]) / geometry.tileSizeX - BOUNDS_EPSILON);
  const maxCopyX = Math.floor((rayMaxX - geometry.bounds.min[0]) / geometry.tileSizeX + BOUNDS_EPSILON);
  const minCopyZ = Math.ceil((rayMinZ - geometry.bounds.max[2]) / geometry.tileSizeZ - BOUNDS_EPSILON);
  const maxCopyZ = Math.floor((rayMaxZ - geometry.bounds.min[2]) / geometry.tileSizeZ + BOUNDS_EPSILON);
  let best: { triangleId: number; copyX: number; copyZ: number; t: number } | null = null;
  for (let copyZ = minCopyZ; copyZ <= maxCopyZ; copyZ++) {
    for (let copyX = minCopyX; copyX <= maxCopyX; copyX++) {
      const origin: CensusVec3 = [
        phaseX - copyX * geometry.tileSizeX,
        geometry.topH,
        phaseZ - copyZ * geometry.tileSizeZ,
      ];
      const hit = bvh.intersectNearest(origin, direction, best?.t ?? maximumT);
      if (
        hit
        && (!best
          || hit.t < best.t
          || (Math.abs(hit.t - best.t) <= BOUNDS_EPSILON
            && (copyX < best.copyX || (copyX === best.copyX && copyZ < best.copyZ))))
      ) best = { ...hit, copyX, copyZ };
    }
  }
  return best;
}

function exactLatticeDirection(
  field: DecodedPeriodicRayField,
  lattice: ExactCategoricalLattice,
  azimuthIndex: number,
  elevationIndex: number,
): CensusVec3 {
  const elevation = field.evaluationElevationsRadians[elevationIndex]!;
  if (Math.abs(elevation - Math.PI / 2) <= 1e-12) return [0, -1, 0];
  return unitDirection(
    positiveModulo(azimuthIndex, lattice.azimuthCount) * Math.PI * 2 / lattice.azimuthCount,
    elevation,
  );
}

function nearestExactLatticeDirection(
  field: DecodedPeriodicRayField,
  lattice: ExactCategoricalLattice,
  direction: CensusVec3,
): DirectionAddress {
  let best: DirectionAddress | null = null;
  let bestDot = Number.NEGATIVE_INFINITY;
  for (let elevationIndex = 0; elevationIndex < field.evaluationElevationsRadians.length; elevationIndex++) {
    const azimuthSamples = elevationIndex === field.evaluationElevationsRadians.length - 1
      ? 1
      : lattice.azimuthCount;
    for (let azimuthIndex = 0; azimuthIndex < azimuthSamples; azimuthIndex++) {
      const canonical = exactLatticeDirection(field, lattice, azimuthIndex, elevationIndex);
      const dot = canonical[0] * direction[0] + canonical[1] * direction[1] + canonical[2] * direction[2];
      if (dot > bestDot) {
        bestDot = dot;
        best = { azimuthIndex, elevationIndex, direction: canonical };
      }
    }
  }
  if (!best) throw new Error('exact categorical lattice has no direction');
  return best;
}

function exactLatticeNearestRecord(
  field: DecodedPeriodicRayField,
  bvh: TriangleBvh,
  lattice: ExactCategoricalLattice,
  phaseX: number,
  phaseZ: number,
  direction: CensusVec3,
): RawRecord {
  const address = nearestExactLatticeDirection(field, lattice, direction);
  const u = wrappedUnit((phaseX - field.geometry.tileOriginX) / field.geometry.tileSizeX);
  const v = 1 - wrappedUnit((phaseZ - field.geometry.tileOriginZ) / field.geometry.tileSizeZ);
  const x = positiveModulo(Math.floor(u * lattice.phaseResolution), lattice.phaseResolution);
  const y = positiveModulo(Math.floor(v * lattice.phaseResolution), lattice.phaseResolution);
  const isVertical = address.elevationIndex === field.evaluationElevationsRadians.length - 1;
  const canonicalAzimuth = isVertical ? 0 : address.azimuthIndex;
  const key = `${address.elevationIndex}:${canonicalAzimuth}:${x}:${y}`;
  const cached = lattice.cache.get(key);
  if (cached) return cached;
  const originX = field.geometry.tileOriginX
    + (x + 0.5) / lattice.phaseResolution * field.geometry.tileSizeX;
  const originZ = field.geometry.tileOriginZ
    + (1 - (y + 0.5) / lattice.phaseResolution) * field.geometry.tileSizeZ;
  const exact = periodicFirstHit(field.geometry, bvh, originX, originZ, address.direction);
  const record: RawRecord = {
    hit: exact !== null,
    t: exact?.t ?? (field.geometry.topH - field.geometry.bounds.min[1]) / -address.direction[1],
    owner: MISS_OWNER,
    triangleId: exact?.triangleId ?? null,
    copyX: exact?.copyX ?? 0,
    copyZ: exact?.copyZ ?? 0,
    direction: address.direction,
    sliceIndex: -1,
    texelX: x,
    texelY: y,
    originX,
    originZ,
  };
  lattice.cache.set(key, record);
  return record;
}

export function computeAtlasSizing(
  azimuthCount: number,
  phaseResolution: number,
  ringCount = 5,
  verticalSingletonCount = 1,
  gutter = 1,
  bytesPerRecord = 8,
): AtlasSizing {
  positiveInteger(azimuthCount, 'atlas azimuth count');
  positiveInteger(phaseResolution, 'atlas phase resolution');
  positiveInteger(ringCount, 'atlas ring count');
  if (!Number.isInteger(verticalSingletonCount) || verticalSingletonCount < 0) {
    throw new Error('atlas vertical singleton count must be a non-negative integer');
  }
  const directionRecordCount = azimuthCount * ringCount + verticalSingletonCount;
  const interiorAtlasRecordCount = phaseResolution * phaseResolution * directionRecordCount;
  const storedSide = phaseResolution + gutter * 2;
  const storedAtlasRecordCount = storedSide * storedSide * directionRecordCount;
  return {
    azimuthCount,
    phaseResolution,
    ringCount,
    verticalSingletonCount,
    directionRecordCount,
    interiorAtlasRecordCount,
    storedAtlasRecordCount,
    storedAtlasBytes: storedAtlasRecordCount * bytesPerRecord,
  };
}

function mutableMetrics(): MutableMetrics {
  return {
    rays: 0,
    truthHits: 0,
    truthMisses: 0,
    predictedHits: 0,
    predictedMisses: 0,
    truePositive: 0,
    trueNegative: 0,
    falsePositive: 0,
    falseNegative: 0,
    verticalSigned: [],
    positionSigned: [],
    invalid: new Map(),
    diagnostic: new Map(),
  };
}

function increment(map: Map<string, number>, key: string): void {
  map.set(key, (map.get(key) ?? 0) + 1);
}

function accumulate(
  metrics: MutableMetrics,
  truthT: number | null,
  prediction: Prediction,
  direction: CensusVec3,
): void {
  metrics.rays++;
  if (truthT === null) metrics.truthMisses++;
  else metrics.truthHits++;
  if (prediction.hit) metrics.predictedHits++;
  else metrics.predictedMisses++;
  if (prediction.invalid) increment(metrics.invalid, prediction.invalid);
  for (const diagnostic of prediction.diagnostic ?? []) increment(metrics.diagnostic, diagnostic);
  if (truthT === null && !prediction.hit) {
    metrics.trueNegative++;
  } else if (truthT === null && prediction.hit) {
    metrics.falsePositive++;
  } else if (truthT !== null && !prediction.hit) {
    metrics.falseNegative++;
  } else if (truthT !== null && prediction.t !== undefined) {
    metrics.truePositive++;
    const signedPositionError = prediction.t - truthT;
    metrics.positionSigned.push(signedPositionError);
    metrics.verticalSigned.push(-direction[1] * signedPositionError);
  }
}

function quantile(sorted: readonly number[], fraction: number): number | null {
  if (sorted.length === 0) return null;
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * fraction) - 1));
  return sorted[index]!;
}

function summarizeErrors(signed: readonly number[]): ErrorQuantiles {
  const absolute = signed.map(Math.abs).sort((a, b) => a - b);
  return {
    count: signed.length,
    signedMean: signed.length > 0 ? signed.reduce((sum, value) => sum + value, 0) / signed.length : null,
    absolute: {
      p50: quantile(absolute, 0.5),
      p90: quantile(absolute, 0.9),
      p95: quantile(absolute, 0.95),
      p99: quantile(absolute, 0.99),
      maximum: absolute.at(-1) ?? null,
    },
  };
}

function objectFromMap(map: ReadonlyMap<string, number>): Record<string, number> {
  return Object.fromEntries([...map.entries()].sort(([left], [right]) => left.localeCompare(right)));
}

function summarize(metrics: MutableMetrics): MethodMetrics {
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
    hitMissAgreement: metrics.rays > 0
      ? (metrics.truePositive + metrics.trueNegative) / metrics.rays
      : 0,
    hitPrecision: metrics.truePositive + metrics.falsePositive > 0
      ? metrics.truePositive / (metrics.truePositive + metrics.falsePositive)
      : 0,
    hitRecall: metrics.truePositive + metrics.falseNegative > 0
      ? metrics.truePositive / (metrics.truePositive + metrics.falseNegative)
      : 0,
    verticalDropErrorMetres: summarizeErrors(metrics.verticalSigned),
    worldPositionErrorMetres: summarizeErrors(metrics.positionSigned),
    invalid: objectFromMap(metrics.invalid),
    diagnostic: objectFromMap(metrics.diagnostic),
  };
}

function runDenserAzimuthProbe(
  field: DecodedPeriodicRayField,
  bvh: TriangleBvh,
  lattice: ExactCategoricalLattice,
  phaseGrid: number,
  phaseOffset: number,
  elevationsDegrees: readonly number[],
): RaySurrogateEvaluationReport['denserAzimuthRings'][string] {
  const aggregate = mutableMetrics();
  const byElevation = new Map<number, MutableMetrics>();
  const started = performance.now();
  let sampleRayCount = 0;
  for (const elevationDegrees of elevationsDegrees) {
    const elevationRadians = elevationDegrees * Math.PI / 180;
    for (let azimuthIndex = 0; azimuthIndex < lattice.azimuthCount; azimuthIndex++) {
      // Worst within-bin angular point for this exact lattice, not a common set
      // which would accidentally land on N=32 directions while testing N=64.
      const azimuthRadians = (azimuthIndex + 0.5) / lattice.azimuthCount * Math.PI * 2;
      const direction = unitDirection(azimuthRadians, elevationRadians);
      for (let phaseZ = 0; phaseZ < phaseGrid; phaseZ++) {
        for (let phaseX = 0; phaseX < phaseGrid; phaseX++) {
          const originX = field.geometry.tileOriginX
            + (phaseX + phaseOffset) / phaseGrid * field.geometry.tileSizeX;
          const originZ = field.geometry.tileOriginZ
            + (phaseZ + phaseOffset) / phaseGrid * field.geometry.tileSizeZ;
          const truth = periodicFirstHit(field.geometry, bvh, originX, originZ, direction);
          const record = exactLatticeNearestRecord(field, bvh, lattice, originX, originZ, direction);
          const prediction = predictNearestVerticalDrop(record, direction);
          let elevationMetrics = byElevation.get(elevationDegrees);
          if (!elevationMetrics) {
            elevationMetrics = mutableMetrics();
            byElevation.set(elevationDegrees, elevationMetrics);
          }
          accumulate(aggregate, truth?.t ?? null, prediction, direction);
          accumulate(elevationMetrics, truth?.t ?? null, prediction, direction);
          sampleRayCount++;
        }
      }
    }
  }
  const sizing = computeAtlasSizing(lattice.azimuthCount, lattice.phaseResolution);
  return {
    azimuthCount: lattice.azimuthCount,
    phaseResolution: lattice.phaseResolution,
    directionRecordCount: sizing.directionRecordCount,
    storedAtlasRecordCount: sizing.storedAtlasRecordCount,
    storedAtlasBytes: sizing.storedAtlasBytes,
    generatedSampleRecords: lattice.cache.size,
    sampleRayCount,
    traceMilliseconds: performance.now() - started,
    aggregate: summarize(aggregate),
    byElevation: Object.fromEntries(
      [...byElevation.entries()]
        .sort(([left], [right]) => left - right)
        .map(([elevation, metrics]) => [String(elevation), summarize(metrics)]),
    ),
  };
}

function validateOptions(options: RaySurrogateEvaluationOptions): void {
  positiveInteger(options.phaseGrid, 'phase grid');
  positiveInteger(options.azimuthCount, 'azimuth count');
  if (options.denserPhaseGrid !== undefined) {
    positiveInteger(options.denserPhaseGrid, 'denser phase grid');
  }
  if (!(options.phaseOffset >= 0 && options.phaseOffset < 1)) throw new Error('phase offset must be in [0,1)');
  if (!(options.azimuthOffset >= 0 && options.azimuthOffset < 1)) throw new Error('azimuth offset must be in [0,1)');
  if (
    options.elevationsDegrees.length === 0
    || options.elevationsDegrees.some((value) => !(value >= 5 && value < 90))
  ) throw new Error('elevations must be in [5,90) degrees');
}

export function runRaySurrogateEvaluation(
  sourceBytes: Uint8Array,
  options: RaySurrogateEvaluationOptions,
): RaySurrogateEvaluationReport {
  validateOptions(options);
  const field = decodeRayField(sourceBytes);
  const bvh = TriangleBvh.build(field.geometry, 8);
  const planePoleThreshold = options.planePoleThreshold ?? 1e-4;
  const coverageThreshold = options.coverageThreshold ?? 0.02;
  const verticalDropHorizon = field.geometry.topH - field.geometry.bounds.min[1];
  const names: readonly SurrogateName[] = [
    'projected_path_relift',
    'nearest_complete_tau',
    'nearest_complete_vertical_drop',
    'phase_reprojected_vertical_drop',
    'phase_reprojected_affine_fixed_point',
    'one_record_face_plane',
    'reciprocal_depth_slope_triangle',
  ];
  const aggregate = Object.fromEntries(names.map((name) => [name, mutableMetrics()])) as Record<
    SurrogateName,
    MutableMetrics
  >;
  const byElevation = Object.fromEntries(names.map((name) => [name, new Map<number, MutableMetrics>()])) as Record<
    SurrogateName,
    Map<number, MutableMetrics>
  >;
  const started = performance.now();
  let rayCount = 0;
  for (const elevationDegrees of options.elevationsDegrees) {
    const elevationRadians = elevationDegrees * Math.PI / 180;
    for (let azimuthIndex = 0; azimuthIndex < options.azimuthCount; azimuthIndex++) {
      const azimuthRadians = (azimuthIndex + options.azimuthOffset)
        / options.azimuthCount * Math.PI * 2;
      const direction = unitDirection(azimuthRadians, elevationRadians);
      for (let phaseZ = 0; phaseZ < options.phaseGrid; phaseZ++) {
        for (let phaseX = 0; phaseX < options.phaseGrid; phaseX++) {
          const originX = field.geometry.tileOriginX
            + (phaseX + options.phaseOffset) / options.phaseGrid * field.geometry.tileSizeX;
          const originZ = field.geometry.tileOriginZ
            + (phaseZ + options.phaseOffset) / options.phaseGrid * field.geometry.tileSizeZ;
          const truth = periodicFirstHit(field.geometry, bvh, originX, originZ, direction);
          const nearest = nearestRecord(field, bvh, originX, originZ, direction);
          const phaseReprojected = predictPhaseReprojectedVerticalDrop(
            field,
            bvh,
            originX,
            originZ,
            direction,
            verticalDropHorizon,
          );
          const predictions: Record<SurrogateName, Prediction> = {
            projected_path_relift: predictProjectedPath(
              field,
              bvh,
              originX,
              originZ,
              direction,
              coverageThreshold,
            ),
            nearest_complete_tau: predictNearestTau(nearest),
            nearest_complete_vertical_drop: predictNearestVerticalDrop(nearest, direction),
            phase_reprojected_vertical_drop: phaseReprojected.iteration,
            phase_reprojected_affine_fixed_point: phaseReprojected.affine,
            one_record_face_plane: predictFacePlane(
              field,
              nearest,
              originX,
              originZ,
              direction,
              verticalDropHorizon,
              planePoleThreshold,
            ),
            reciprocal_depth_slope_triangle: predictReciprocalDepth(
              field,
              bvh,
              originX,
              originZ,
              direction,
              verticalDropHorizon,
              coverageThreshold,
            ),
          };
          for (const name of names) {
            let elevationMetrics = byElevation[name].get(elevationDegrees);
            if (!elevationMetrics) {
              elevationMetrics = mutableMetrics();
              byElevation[name].set(elevationDegrees, elevationMetrics);
            }
            accumulate(aggregate[name], truth?.t ?? null, predictions[name], direction);
            accumulate(elevationMetrics, truth?.t ?? null, predictions[name], direction);
          }
          rayCount++;
        }
      }
    }
  }
  const methods = Object.fromEntries(names.map((name) => [name, {
    aggregate: summarize(aggregate[name]),
    byElevation: Object.fromEntries(
      [...byElevation[name].entries()]
        .sort(([left], [right]) => left - right)
        .map(([elevation, metrics]) => [String(elevation), summarize(metrics)]),
    ),
  }])) as RaySurrogateEvaluationReport['methods'];
  const primaryTraceMilliseconds = performance.now() - started;
  const denserPhaseGrid = options.denserPhaseGrid ?? 4;
  const denserConfigs = [
    { azimuthCount: 32, phaseResolution: 161 },
    { azimuthCount: 64, phaseResolution: 113 },
  ] as const;
  const denserAzimuthRings: RaySurrogateEvaluationReport['denserAzimuthRings'] = {};
  for (const config of denserConfigs) {
    const lattice: ExactCategoricalLattice = {
      ...config,
      cache: new Map(),
    };
    denserAzimuthRings[`azimuth${config.azimuthCount}_phase${config.phaseResolution}`] =
      runDenserAzimuthProbe(
        field,
        bvh,
        lattice,
        denserPhaseGrid,
        options.phaseOffset,
        options.elevationsDegrees,
      );
  }
  const correctedGrid = [16, 32, 64].flatMap((azimuthCount) =>
    [64, 128, 256].map((phaseResolution) => computeAtlasSizing(azimuthCount, phaseResolution)));
  return {
    domain: {
      phaseGrid: options.phaseGrid,
      phaseOffset: options.phaseOffset,
      azimuthCount: options.azimuthCount,
      azimuthOffset: options.azimuthOffset,
      elevationsDegrees: [...options.elevationsDegrees],
      rayCount,
      canonicalAzimuthCount: field.azimuthCount,
      canonicalElevationsDegrees: field.elevationsRadians.map((value) => value * 180 / Math.PI),
      evaluationElevationsDegrees: field.evaluationElevationsRadians.map((value) => value * 180 / Math.PI),
      generatedCanonicalRowsDegrees: [5, 90],
      generatedCanonicalRecordCount: field.freshRecordCache.size,
      verticalDropHorizon,
      planePoleThreshold,
      coverageThreshold,
    },
    accelerator: bvh.metrics,
    traceMilliseconds: primaryTraceMilliseconds,
    methods,
    denserAzimuthRings,
    atlasSizing: {
      bytesPerRecord: 8,
      gutter: 1,
      currentShipped: computeAtlasSizing(16, 256, 4, 0),
      correctedGrid,
      byteMatchedConfigurations: [
        computeAtlasSizing(32, 161),
        computeAtlasSizing(64, 113),
      ],
    },
    runtimeCostModel: {
      projected_path_relift: {
        filteredTextureReads: 4,
        nearestTextureReads: 0,
        effectiveSourceTexels: 16,
        majorScalarOps: [
          'two direction angles',
          'four coverage-unmix operations',
          'four weighted accumulations',
          'one reciprocal-path inversion',
          'one live-horizontal relift division',
        ],
        payloadAssumption: 'existing filterable inverse-projected-path plus coverage carrier',
      },
      nearest_complete_tau: {
        filteredTextureReads: 0,
        nearestTextureReads: 1,
        effectiveSourceTexels: 1,
        majorScalarOps: ['nearest direction/phase address', 'one ray point multiply-add'],
        payloadAssumption: 'full 3D tau and binary hit are retained in one nearest record',
      },
      nearest_complete_vertical_drop: {
        filteredTextureReads: 0,
        nearestTextureReads: 1,
        effectiveSourceTexels: 1,
        majorScalarOps: ['nearest direction/phase address', 'one live vertical-speed division'],
        payloadAssumption: 'one nearest record stores vertical drop h and binary hit directly',
      },
      phase_reprojected_vertical_drop: {
        filteredTextureReads: 0,
        nearestTextureReads: 2,
        effectiveSourceTexels: 2,
        majorScalarOps: [
          'one predictor vertical drop',
          'one slope-delta times h phase shift',
          'one live vertical-speed division',
        ],
        payloadAssumption: 'two categorical nearest records from one canonical vertical-drop slice',
      },
      phase_reprojected_affine_fixed_point: {
        filteredTextureReads: 0,
        nearestTextureReads: 2,
        effectiveSourceTexels: 2,
        majorScalarOps: [
          'the same predictor and reprojected reads as E1',
          'h0 squared divided by 2h0 minus h1',
          'one live vertical-speed division',
        ],
        payloadAssumption: 'two categorical nearest records from one canonical vertical-drop slice',
      },
      one_record_face_plane: {
        filteredTextureReads: 0,
        nearestTextureReads: 1,
        effectiveSourceTexels: 1,
        majorScalarOps: ['two vec3 dot products', 'one division', 'one vertical-band test'],
        payloadAssumption: 'full tau and geometric source-face normal are packed in one record; the evaluator derives the normal from the v4 owner table offline',
      },
      reciprocal_depth_slope_triangle: {
        filteredTextureReads: 3,
        nearestTextureReads: 0,
        effectiveSourceTexels: 12,
        majorScalarOps: [
          'Cartesian slope-triangle address and barycentrics',
          'three weighted reciprocal-depth accumulations',
          'one final reciprocal',
          'one live vertical-speed division',
        ],
        payloadAssumption: 'each read stores premultiplied reciprocal vertical drop plus coverage; deriving it from ordinary depth instead costs three additional reciprocals',
      },
    },
    interpretation: [
      'analytic truth is the nearest decoded-u16 mesh hit across all periodic copies, not a canonical owner texel',
      'all evaluated rays start on the profile top plane; camera-inside successor semantics are outside this experiment',
      'nearest-complete and face-plane use one nearest phase texel and one nearest canonical direction without blending',
      'nearest-complete tau is retained as a negative control; nearest-complete vertical drop is the one-read h-chart surrogate',
      'phase-reprojected E1 applies q_i=q+(s-s_i)h0 once; E2 uses the same two reads and solves the affine fixed-point model h=h0+(h1-h0)h/h0',
      'face-plane is the infinite plane through the stored canonical hit with the baked winning source face normal; triangle containment is diagnostic only',
      'reciprocal-depth uses three spatially filtered canonical direction reads and affine barycentrics in Cartesian slope space',
      'the evaluator augments the shipped 15/35/55/75-degree field with exact-mesh 5-degree texels and a vertical singleton at every touched canonical phase texel',
      'the generated rows are sparse evaluation bakes on the same 256-square phase lattice, cached once and shared by every method',
      'the slope triangulation therefore covers 5-to-90 degrees without angular extrapolation; its outer 5-degree boundary remains the declared grazing limit',
      'error quantiles include true-positive rays only; hit/miss agreement separately exposes false positives and false negatives',
      'world-position error equals absolute t error because every compared ray direction is unit length',
      'the runtime cost table counts shader sampling instructions; effective source texels show the bilinear hardware footprint',
      'the independent N=32 and N=64 probes use their own worst half-bin live azimuths and exact sampled canonical records at byte-matched phase resolutions',
      'atlas byte counts include one wrapped gutter texel on every side and an eight-byte categorical record',
      'the evaluator contains offline loops and BVH traversal only; it proposes no runtime march, loop, or candidate enumeration',
    ],
  };
}
