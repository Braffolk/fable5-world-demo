/**
 * Offline Candidate-C gate for exact cross-direction chart correspondence.
 *
 * Records are never compared at equal texel addresses.  A covered source
 * record is intersected with its exact packed GCRP/v4 triangle, that point is
 * reprojected to the target direction's top plane, and the target's nearest
 * record is fetched there.  Periodic copy identity is then globalised using
 * the target address's tile index before exact key comparison.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  derivePeriodicProfileLattice,
  parsePeriodicProfile,
} from '../../src/nanite/groundcover/GroundCoverPeriodicProfile';
import type {
  PeriodicProfileData,
  PeriodicProfileLattice,
} from '../../src/nanite/groundcover/GroundCoverProfileTypes';

const EXPECTED_SOURCE_SHA256 =
  '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const ANALYSIS_VERSION = 'candidate-c-epipolar-global-owner-v1';
const MISS = 0xffff_ffff;

/** Frozen before inspecting any correspondence result; mirrors spec section 24.4. */
export const GREEN_THRESHOLDS = Object.freeze({
  primary45ConditionalAgreement: 0.60,
  aggregateConditionalAgreement: 0.50,
  outerBoundaryConditionalAgreement: 0.35,
  maximumForwardReverseGap: 0.10,
});

type Vec3 = readonly [number, number, number];

export interface ExactOwnerKey {
  readonly triangleId: number;
  readonly copyX: number;
  readonly copyZ: number;
}

interface PairSpec {
  readonly id: string;
  readonly family: 'elevation-aligned' | 'elevation-diagonal' | 'azimuth-adjacent';
  readonly boundaryDegrees: number | null;
  readonly sourceRow: number;
  readonly targetRow: number;
  readonly sourceAzimuth: number;
  readonly targetAzimuth: number;
  readonly direction: 'low-to-high' | 'high-to-low' | 'clockwise' | 'counterclockwise';
  readonly diagonal: -1 | 0 | 1;
  readonly sourceSlice: number;
  readonly targetSlice: number;
}

interface MutablePairCounts {
  sourceCovered: number;
  sourceGeometryValid: number;
  sourceGeometryInvalid: number;
  targetCovered: number;
  exactKey: number;
  sameTriangleWrongCopy: number;
  differentTriangle: number;
  targetMiss: number;
  bottomGutterSamples: number;
}

interface PairResult extends PairSpec, MutablePairCounts {
  readonly coverageRelay: number;
  readonly conditionalAgreement: number;
  readonly sharedKeyMass: number;
  readonly invalidSourceRate: number;
  readonly optimisticInvalidExactConditionalUpperBound: number;
  readonly optimisticInvalidExactMassUpperBound: number;
}

interface ExactSourceEvent {
  readonly point: Vec3;
  readonly key: ExactOwnerKey;
  readonly storedDepthError: number;
}

interface TargetRecord {
  readonly key: ExactOwnerKey | null;
  readonly nearestAddressError: number;
  readonly bottomGutter: boolean;
}

class Histogram {
  private readonly bins: Uint32Array;
  private count = 0;
  private sum = 0;
  private maximum = 0;

  constructor(private readonly limit: number, binCount = 4096) {
    this.bins = new Uint32Array(binCount);
  }

  add(value: number): void {
    if (!Number.isFinite(value) || value < 0) return;
    const index = Math.min(
      this.bins.length - 1,
      Math.floor(value / this.limit * this.bins.length),
    );
    this.bins[index]++;
    this.count++;
    this.sum += value;
    this.maximum = Math.max(this.maximum, value);
  }

  quantile(fraction: number): number {
    if (this.count === 0) return 0;
    const target = Math.ceil(this.count * fraction);
    let cumulative = 0;
    for (let index = 0; index < this.bins.length; index++) {
      cumulative += this.bins[index]!;
      if (cumulative >= target) return (index + 0.5) / this.bins.length * this.limit;
    }
    return this.maximum;
  }

  report(): Record<string, number> {
    return {
      count: this.count,
      mean: this.count === 0 ? 0 : this.sum / this.count,
      p50: this.quantile(0.50),
      p95: this.quantile(0.95),
      p99: this.quantile(0.99),
      maximum: this.maximum,
      histogramLimit: this.limit,
    };
  }
}

const fraction = (numerator: number, denominator: number): number =>
  denominator === 0 ? 0 : numerator / denominator;

const positiveModulo = (value: number, modulus: number): number =>
  ((value % modulus) + modulus) % modulus;

const dot = (left: Vec3, right: Vec3): number =>
  left[0] * right[0] + left[1] * right[1] + left[2] * right[2];

const subtract = (left: Vec3, right: Vec3): Vec3 => [
  left[0] - right[0], left[1] - right[1], left[2] - right[2],
];

const cross = (left: Vec3, right: Vec3): Vec3 => [
  left[1] * right[2] - left[2] * right[1],
  left[2] * right[0] - left[0] * right[2],
  left[0] * right[1] - left[1] * right[0],
];

const lo16 = (word: number): number => word & 0xffff;
const hi16 = (word: number): number => word >>> 16;
const unit16 = (word: number): number => word / 65535;

export const unpackOwner = (
  token: number,
  addressTileX = 0,
  addressTileZ = 0,
): ExactOwnerKey | null => token === MISS ? null : ({
  triangleId: token & 0x3f_ffff,
  copyX: addressTileX + ((token >>> 22) & 0x1f) - 16,
  copyZ: addressTileZ + ((token >>> 27) & 0x1f) - 16,
});

export const sameOwnerKey = (left: ExactOwnerKey, right: ExactOwnerKey): boolean =>
  left.triangleId === right.triangleId
  && left.copyX === right.copyX
  && left.copyZ === right.copyZ;

/** Exact top-plane address of point X for downward direction c. */
export const epipolarTopAddress = (
  point: Vec3,
  direction: Vec3,
  topH: number,
): readonly [number, number] => {
  if (!(direction[1] < 0)) throw new Error('epipolar target direction must point downward');
  const depth = (point[1] - topH) / direction[1];
  return [
    point[0] - depth * direction[0],
    point[2] - depth * direction[2],
  ];
};

function sliceIndex(
  lattice: PeriodicProfileLattice,
  azimuth: number,
  elevation: number,
): number {
  return lattice.order === 'azimuth-major'
    ? positiveModulo(azimuth, lattice.azimuthCount) * lattice.elevationCount + elevation
    : elevation * lattice.azimuthCount + positiveModulo(azimuth, lattice.azimuthCount);
}

function atlasTexel(
  profile: PeriodicProfileData,
  slice: number,
  storedX: number,
  storedY: number,
): number {
  const column = slice % profile.atlasColumns;
  const row = Math.floor(slice / profile.atlasColumns);
  const atlasWidth = profile.storedTileWidth * profile.atlasColumns;
  return (
    (row * profile.storedTileHeight + storedY) * atlasWidth
    + column * profile.storedTileWidth + storedX
  );
}

function interiorOwner(
  profile: PeriodicProfileData,
  slice: number,
  x: number,
  y: number,
): number {
  return profile.ownerTexels![atlasTexel(
    profile,
    slice,
    profile.gutter + x,
    profile.gutter + y,
  )]!;
}

function decodeVertex(profile: PeriodicProfileData, vertexId: number, key: ExactOwnerKey): Vec3 {
  const bounds = profile.sourceBounds!;
  const base = vertexId * 4;
  const xy = profile.vertexRecords![base]!;
  const zr = profile.vertexRecords![base + 1]!;
  return [
    bounds[0] + unit16(lo16(xy)) * (bounds[3] - bounds[0]) + key.copyX * profile.tileSizeX,
    bounds[1] + unit16(hi16(xy)) * (bounds[4] - bounds[1]),
    bounds[2] + unit16(lo16(zr)) * (bounds[5] - bounds[2]) + key.copyZ * profile.tileSizeZ,
  ];
}

function sourceEvent(
  profile: PeriodicProfileData,
  slice: number,
  x: number,
  y: number,
  token: number,
): ExactSourceEvent | null {
  const key = unpackOwner(token);
  if (!key) return null;
  const triangleBase = key.triangleId * 4;
  const a = decodeVertex(profile, profile.triangleRecords![triangleBase]!, key);
  const b = decodeVertex(profile, profile.triangleRecords![triangleBase + 1]!, key);
  const c = decodeVertex(profile, profile.triangleRecords![triangleBase + 2]!, key);
  const e1 = subtract(b, a);
  const e2 = subtract(c, a);
  const normal = cross(e1, e2);
  const direction = profile.slices[slice]!.direction;
  const origin: Vec3 = [
    profile.tileOriginX + (x + 0.5) / profile.interiorTileWidth * profile.tileSizeX,
    profile.topH,
    profile.tileOriginZ + (1 - (y + 0.5) / profile.interiorTileHeight) * profile.tileSizeZ,
  ];
  const denominator = dot(normal, direction);
  if (!(Math.abs(denominator) > 1e-14)) return null;
  const t = dot(normal, subtract(a, origin)) / denominator;
  if (!(t >= 0 && Number.isFinite(t))) return null;
  const point: Vec3 = [
    origin[0] + direction[0] * t,
    origin[1] + direction[1] * t,
    origin[2] + direction[2] * t,
  ];
  // A loose edge tolerance admits the fixed rasterizer's top-left boundary
  // convention while rejecting a corrupt owner/triangle association.
  const q = subtract(point, a);
  const d00 = dot(e1, e1);
  const d01 = dot(e1, e2);
  const d11 = dot(e2, e2);
  const d20 = dot(q, e1);
  const d21 = dot(q, e2);
  const barycentricDenominator = d00 * d11 - d01 * d01;
  if (!(Math.abs(barycentricDenominator) > 1e-24)) return null;
  const v = (d11 * d20 - d01 * d21) / barycentricDenominator;
  const w = (d00 * d21 - d01 * d20) / barycentricDenominator;
  const u = 1 - v - w;
  if (Math.min(u, v, w) < -2e-3 || Math.max(u, v, w) > 1.002) return null;

  const texel = atlasTexel(profile, slice, profile.gutter + x, profile.gutter + y);
  const depth01 = profile.texels[texel * 4]! / 65535;
  const sliceData = profile.slices[slice]!;
  const storedT = sliceData.depthMin + depth01 * (sliceData.depthMax - sliceData.depthMin);
  return { point, key, storedDepthError: Math.abs(t - storedT) };
}

/** Runtime-equivalent NEAREST fetch plus exact address-tile copy normalisation. */
function targetRecord(
  profile: PeriodicProfileData,
  slice: number,
  addressX: number,
  addressZ: number,
): TargetRecord {
  const normalisedX = (addressX - profile.tileOriginX) / profile.tileSizeX;
  const normalisedZ = (addressZ - profile.tileOriginZ) / profile.tileSizeZ;
  const tileX = Math.floor(normalisedX);
  const tileZ = Math.floor(normalisedZ);
  const phaseX = normalisedX - tileX;
  const phaseZ = normalisedZ - tileZ;
  const storedX = Math.floor(profile.gutter + phaseX * profile.interiorTileWidth);
  // The shader uses 1-fract(z), so an exact low-Z tile boundary addresses the
  // bottom wrapped gutter.  Keep that behavior rather than silently wrapping.
  const storedY = Math.floor(profile.gutter + (1 - phaseZ) * profile.interiorTileHeight);
  if (
    storedX < 0 || storedX >= profile.storedTileWidth
    || storedY < 0 || storedY >= profile.storedTileHeight
  ) throw new Error('nearest target address escaped the guarded tile');
  const texel = atlasTexel(profile, slice, storedX, storedY);
  const token = profile.ownerTexels![texel]!;
  const sourceX = positiveModulo(storedX - profile.gutter, profile.interiorTileWidth);
  const sourceY = positiveModulo(storedY - profile.gutter, profile.interiorTileHeight);
  const nearestX = profile.tileOriginX
    + (tileX + (sourceX + 0.5) / profile.interiorTileWidth) * profile.tileSizeX;
  const nearestZ = profile.tileOriginZ
    + (tileZ + 1 - (sourceY + 0.5) / profile.interiorTileHeight) * profile.tileSizeZ;
  return {
    key: unpackOwner(token, tileX, tileZ),
    nearestAddressError: Math.hypot(nearestX - addressX, nearestZ - addressZ),
    bottomGutter: storedY === profile.gutter + profile.interiorTileHeight,
  };
}

function emptyCounts(): MutablePairCounts {
  return {
    sourceCovered: 0,
    sourceGeometryValid: 0,
    sourceGeometryInvalid: 0,
    targetCovered: 0,
    exactKey: 0,
    sameTriangleWrongCopy: 0,
    differentTriangle: 0,
    targetMiss: 0,
    bottomGutterSamples: 0,
  };
}

function addCounts(target: MutablePairCounts, source: MutablePairCounts): void {
  for (const key of Object.keys(target) as (keyof MutablePairCounts)[]) target[key] += source[key];
}

function finishPair(spec: PairSpec, counts: MutablePairCounts): PairResult {
  return {
    ...spec,
    ...counts,
    coverageRelay: fraction(counts.targetCovered, counts.sourceGeometryValid),
    conditionalAgreement: fraction(counts.exactKey, counts.targetCovered),
    sharedKeyMass: fraction(counts.exactKey, counts.sourceGeometryValid),
    invalidSourceRate: fraction(counts.sourceGeometryInvalid, counts.sourceCovered),
    // Robustness bound only: assume every source event rejected by the exact
    // packed-triangle check would have found a covered target with the exact
    // same key.  If this still misses GREEN, raster-edge exclusions cannot be
    // the reason for parking the candidate.
    optimisticInvalidExactConditionalUpperBound: fraction(
      counts.exactKey + counts.sourceGeometryInvalid,
      counts.targetCovered + counts.sourceGeometryInvalid,
    ),
    optimisticInvalidExactMassUpperBound: fraction(
      counts.exactKey + counts.sourceGeometryInvalid,
      counts.sourceCovered,
    ),
  };
}

function pairSpecs(lattice: PeriodicProfileLattice): PairSpec[] {
  const result: PairSpec[] = [];
  const elevations = lattice.elevations.map((value) => value * 180 / Math.PI);
  for (let lowRow = 0; lowRow < lattice.elevationCount - 1; lowRow++) {
    const highRow = lowRow + 1;
    const boundary = Math.round((elevations[lowRow]! + elevations[highRow]!) * 0.5);
    for (let azimuth = 0; azimuth < lattice.azimuthCount; azimuth++) {
      const add = (
        family: PairSpec['family'],
        sourceRow: number,
        targetRow: number,
        sourceAzimuth: number,
        targetAzimuth: number,
        direction: PairSpec['direction'],
        diagonal: PairSpec['diagonal'],
      ): void => {
        result.push({
          id: `${family}-${boundary}-${direction}-az${sourceAzimuth}-to-az${positiveModulo(targetAzimuth, lattice.azimuthCount)}`,
          family,
          boundaryDegrees: boundary,
          sourceRow,
          targetRow,
          sourceAzimuth,
          targetAzimuth: positiveModulo(targetAzimuth, lattice.azimuthCount),
          direction,
          diagonal,
          sourceSlice: sliceIndex(lattice, sourceAzimuth, sourceRow),
          targetSlice: sliceIndex(lattice, targetAzimuth, targetRow),
        });
      };
      add('elevation-aligned', lowRow, highRow, azimuth, azimuth, 'low-to-high', 0);
      add('elevation-aligned', highRow, lowRow, azimuth, azimuth, 'high-to-low', 0);
      add('elevation-diagonal', lowRow, highRow, azimuth, azimuth + 1, 'low-to-high', 1);
      add('elevation-diagonal', highRow, lowRow, azimuth + 1, azimuth, 'high-to-low', -1);
      add('elevation-diagonal', lowRow, highRow, azimuth, azimuth - 1, 'low-to-high', -1);
      add('elevation-diagonal', highRow, lowRow, azimuth - 1, azimuth, 'high-to-low', 1);
    }
  }
  for (let row = 0; row < lattice.elevationCount; row++) {
    const elevation = Math.round(elevations[row]!);
    for (let azimuth = 0; azimuth < lattice.azimuthCount; azimuth++) {
      const next = (azimuth + 1) % lattice.azimuthCount;
      result.push({
        id: `azimuth-adjacent-${elevation}-clockwise-az${azimuth}-to-az${next}`,
        family: 'azimuth-adjacent', boundaryDegrees: null,
        sourceRow: row, targetRow: row, sourceAzimuth: azimuth, targetAzimuth: next,
        direction: 'clockwise', diagonal: 1,
        sourceSlice: sliceIndex(lattice, azimuth, row),
        targetSlice: sliceIndex(lattice, next, row),
      });
      result.push({
        id: `azimuth-adjacent-${elevation}-counterclockwise-az${next}-to-az${azimuth}`,
        family: 'azimuth-adjacent', boundaryDegrees: null,
        sourceRow: row, targetRow: row, sourceAzimuth: next, targetAzimuth: azimuth,
        direction: 'counterclockwise', diagonal: -1,
        sourceSlice: sliceIndex(lattice, next, row),
        targetSlice: sliceIndex(lattice, azimuth, row),
      });
    }
  }
  return result;
}

function aggregatePairs(results: readonly PairResult[]): ReturnType<typeof finishPair> {
  const counts = emptyCounts();
  for (const result of results) addCounts(counts, result);
  return finishPair({
    id: 'aggregate', family: 'elevation-aligned', boundaryDegrees: null,
    sourceRow: -1, targetRow: -1, sourceAzimuth: -1, targetAzimuth: -1,
    direction: 'low-to-high', diagonal: 0, sourceSlice: -1, targetSlice: -1,
  }, counts);
}

export function runChartCorrespondenceAnalysis(sourcePath: string): Record<string, unknown> {
  const sourceBytes = readFileSync(sourcePath);
  const sourceSha256 = createHash('sha256').update(sourceBytes).digest('hex');
  if (sourceSha256 !== EXPECTED_SOURCE_SHA256) {
    throw new Error(`Candidate-C gate expected ${EXPECTED_SOURCE_SHA256}, got ${sourceSha256}`);
  }
  const profile = parsePeriodicProfile(sourceBytes);
  if (
    profile.version !== 4
    || !profile.ownerTexels
    || !profile.vertexRecords
    || !profile.triangleRecords
    || !profile.sourceBounds
  ) throw new Error('Candidate-C gate requires the complete GCRP/v4 owned profile');
  const lattice = derivePeriodicProfileLattice(profile);
  if (lattice.elevationCount !== 4 || lattice.azimuthCount !== 16) {
    throw new Error('Candidate-C gate expected the accepted 16x4 lattice');
  }

  const specs = pairSpecs(lattice);
  const counts = specs.map(() => emptyCounts());
  const bySourceSlice = new Map<number, number[]>();
  specs.forEach((spec, index) => {
    const list = bySourceSlice.get(spec.sourceSlice) ?? [];
    list.push(index);
    bySourceSlice.set(spec.sourceSlice, list);
  });
  const storedDepthError = new Histogram(0.008);
  const nearestAddressError = new Histogram(0.008);
  let sourceEventsComputed = 0;

  for (const [sourceSlice, specIndices] of bySourceSlice) {
    for (let y = 0; y < profile.interiorTileHeight; y++) {
      for (let x = 0; x < profile.interiorTileWidth; x++) {
        const token = interiorOwner(profile, sourceSlice, x, y);
        if (token === MISS) continue;
        for (const specIndex of specIndices) counts[specIndex]!.sourceCovered++;
        const event = sourceEvent(profile, sourceSlice, x, y, token);
        if (!event) {
          for (const specIndex of specIndices) counts[specIndex]!.sourceGeometryInvalid++;
          continue;
        }
        sourceEventsComputed++;
        storedDepthError.add(event.storedDepthError);
        for (const specIndex of specIndices) {
          const spec = specs[specIndex]!;
          const metric = counts[specIndex]!;
          metric.sourceGeometryValid++;
          const targetDirection = profile.slices[spec.targetSlice]!.direction;
          const targetAddress = epipolarTopAddress(event.point, targetDirection, profile.topH);
          const target = targetRecord(
            profile, spec.targetSlice, targetAddress[0], targetAddress[1],
          );
          nearestAddressError.add(target.nearestAddressError);
          if (target.bottomGutter) metric.bottomGutterSamples++;
          if (!target.key) {
            metric.targetMiss++;
            continue;
          }
          metric.targetCovered++;
          if (sameOwnerKey(event.key, target.key)) metric.exactKey++;
          else if (event.key.triangleId === target.key.triangleId) metric.sameTriangleWrongCopy++;
          else metric.differentTriangle++;
        }
      }
    }
  }

  const detailed = specs.map((spec, index) => finishPair(spec, counts[index]!));
  const aligned = detailed.filter((value) => value.family === 'elevation-aligned');
  const diagonal = detailed.filter((value) => value.family === 'elevation-diagonal');
  const azimuth = detailed.filter((value) => value.family === 'azimuth-adjacent');
  const boundaries = [25, 45, 65].map((boundary) => {
    const selected = aligned.filter((value) => value.boundaryDegrees === boundary);
    const forward = aggregatePairs(selected.filter((value) => value.direction === 'low-to-high'));
    const reverse = aggregatePairs(selected.filter((value) => value.direction === 'high-to-low'));
    const combined = aggregatePairs(selected);
    return {
      boundaryDegrees: boundary,
      forward,
      reverse,
      combined,
      forwardReverseGap: Math.abs(forward.conditionalAgreement - reverse.conditionalAgreement),
      perAzimuth: selected,
    };
  });
  const aggregate = aggregatePairs(aligned);
  const boundary = (degrees: number) => boundaries.find((value) => value.boundaryDegrees === degrees)!;
  const gateChecks = {
    primary45: boundary(45).combined.conditionalAgreement
      >= GREEN_THRESHOLDS.primary45ConditionalAgreement,
    aggregate: aggregate.conditionalAgreement
      >= GREEN_THRESHOLDS.aggregateConditionalAgreement,
    outer25: boundary(25).combined.conditionalAgreement
      >= GREEN_THRESHOLDS.outerBoundaryConditionalAgreement,
    outer65: boundary(65).combined.conditionalAgreement
      >= GREEN_THRESHOLDS.outerBoundaryConditionalAgreement,
    forwardReverse25: boundary(25).forwardReverseGap
      <= GREEN_THRESHOLDS.maximumForwardReverseGap,
    forwardReverse45: boundary(45).forwardReverseGap
      <= GREEN_THRESHOLDS.maximumForwardReverseGap,
    forwardReverse65: boundary(65).forwardReverseGap
      <= GREEN_THRESHOLDS.maximumForwardReverseGap,
  };
  const green = Object.values(gateChecks).every(Boolean);

  return {
    analysisVersion: ANALYSIS_VERSION,
    source: {
      path: relative(resolve('.'), sourcePath),
      sha256: sourceSha256,
      bytes: sourceBytes.byteLength,
      profileVersion: profile.version,
      dimensions: [profile.interiorTileWidth, profile.interiorTileHeight],
      storedDimensions: [profile.storedTileWidth, profile.storedTileHeight],
      tile: [profile.tileOriginX, profile.tileOriginZ, profile.tileSizeX, profile.tileSizeZ],
      topH: profile.topH,
      triangleCount: profile.triangleRecords.length / 4,
    },
    method: {
      pairing: 'exact packed-triangle hit X_A -> epipolar top-plane reprojection -> target NEAREST fetch',
      targetAddressEquation: 'p_X(c_B)=X_A.xz-((X_A.y-H)/c_B.y)*c_B.xz',
      exactKey: '(triangle id, floor((address-origin)/tile size)+stored relative copy X/Z)',
      forbiddenControl: 'equal source/target texel address is never used',
      denominator: {
        coverageRelay: 'target covered / source covered with valid exact source geometry',
        conditionalAgreement: 'exact global key / epipolar target covered',
        sharedKeyMass: 'exact global key / source covered with valid exact source geometry',
      },
    },
    frozenThresholds: GREEN_THRESHOLDS,
    lattice: {
      order: lattice.order,
      azimuthCount: lattice.azimuthCount,
      elevationsDegrees: lattice.elevations.map((value) => value * 180 / Math.PI),
    },
    validation: {
      sourceEventsComputed,
      storedDepthErrorMetres: storedDepthError.report(),
      nearestTargetAddressErrorMetres: nearestAddressError.report(),
    },
    decision: {
      green,
      verdict: green ? 'GREEN' : 'RED',
      checks: gateChecks,
      primaryBoundaryDegrees: 45,
      note: 'The 45-degree threshold is independently stricter; no post-result threshold tuning is permitted.',
    },
    elevationAligned: {
      aggregate,
      boundaries,
    },
    elevationDiagonal: {
      aggregate: aggregatePairs(diagonal),
      byBoundary: [25, 45, 65].map((degrees) => ({
        boundaryDegrees: degrees,
        combined: aggregatePairs(diagonal.filter((value) => value.boundaryDegrees === degrees)),
      })),
      detailed: diagonal,
    },
    azimuthAdjacent: {
      aggregate: aggregatePairs(azimuth),
      byElevation: lattice.elevations.map((radians, row) => ({
        elevationDegrees: radians * 180 / Math.PI,
        combined: aggregatePairs(azimuth.filter((value) => value.sourceRow === row)),
      })),
      detailed: azimuth,
    },
    boundedClaim: green
      ? 'Exact chart consensus has enough measured support to justify a runtime Candidate-C gate; it remains bounded to records carrying identical exact global triangle/copy keys.'
      : 'Exact triangle/copy consensus cannot remove enough of the measured elevation-switch mass under the frozen gate; do not implement Candidate C from this carrier.',
  };
}

async function main(): Promise<void> {
  const sourcePath = resolve(process.argv[2] ?? 'src/assets/groundcover/calamagrostis-canescens.gcrp');
  const report = runChartCorrespondenceAnalysis(sourcePath);
  const analyzerPath = fileURLToPath(import.meta.url);
  const analyzerSha256 = createHash('sha256').update(readFileSync(analyzerPath)).digest('hex');
  const recipe = {
    analysisVersion: ANALYSIS_VERSION,
    analyzerSha256,
    sourceSha256: (report.source as { sha256: string }).sha256,
    frozenThresholds: GREEN_THRESHOLDS,
  };
  const recipeSha256 = createHash('sha256').update(JSON.stringify(recipe)).digest('hex');
  const output = resolve(
    'data/work/groundcover-chart-correspondence',
    EXPECTED_SOURCE_SHA256.slice(0, 16),
    recipeSha256.slice(0, 16),
    'report.json',
  );
  mkdirSync(dirname(output), { recursive: true });
  const fullReport = { recipe: { ...recipe, recipeSha256 }, ...report };
  const reportText = `${JSON.stringify(fullReport, null, 2)}\n`;
  writeFileSync(output, reportText);
  const reportSha256 = createHash('sha256').update(reportText).digest('hex');
  const aligned = report.elevationAligned as {
    aggregate: PairResult;
    boundaries: readonly {
      boundaryDegrees: number;
      forward: PairResult;
      reverse: PairResult;
      combined: PairResult;
      forwardReverseGap: number;
    }[];
  };
  const percentage = (value: number): string => `${(value * 100).toFixed(2)}%`;
  const summaryLines = [
    '# Candidate C exact chart-correspondence gate',
    '',
    `- Verdict: **${(report.decision as { verdict: string }).verdict}**`,
    `- Source SHA-256: \`${EXPECTED_SOURCE_SHA256}\``,
    `- Analyzer SHA-256: \`${analyzerSha256}\``,
    `- Recipe SHA-256: \`${recipeSha256}\``,
    `- Report SHA-256: \`${reportSha256}\``,
    '- Pairing: packed-triangle source point, exact epipolar reprojection, target NEAREST fetch, exact global triangle/copy key.',
    '- Equal-address pairing was never used.',
    '',
    '| Boundary | Relay | Conditional exact | Shared-key mass | Optimistic invalid-exact upper bound | Fwd/rev gap |',
    '|---:|---:|---:|---:|---:|---:|',
    ...aligned.boundaries.map((value) => (
      `| ${value.boundaryDegrees} deg | ${percentage(value.combined.coverageRelay)} | ${percentage(value.combined.conditionalAgreement)} | ${percentage(value.combined.sharedKeyMass)} | ${percentage(value.combined.optimisticInvalidExactConditionalUpperBound)} | ${percentage(value.forwardReverseGap)} |`
    )),
    `| aggregate | ${percentage(aligned.aggregate.coverageRelay)} | ${percentage(aligned.aggregate.conditionalAgreement)} | ${percentage(aligned.aggregate.sharedKeyMass)} | ${percentage(aligned.aggregate.optimisticInvalidExactConditionalUpperBound)} | - |`,
    '',
    'Frozen GREEN thresholds: 45 deg >=60%, aggregate >=50%, outer boundaries >=35%, every forward/reverse gap <=10 percentage points.',
    '',
    (report.boundedClaim as string),
    '',
  ];
  const summary = resolve(dirname(output), 'SUMMARY.md');
  writeFileSync(summary, `${summaryLines.join('\n')}\n`);
  const indexPath = resolve('data/work/groundcover-chart-correspondence/index.json');
  mkdirSync(dirname(indexPath), { recursive: true });
  writeFileSync(indexPath, `${JSON.stringify({
    current: relative(resolve('.'), output),
    sourceSha256: EXPECTED_SOURCE_SHA256,
    recipeSha256,
    analyzerSha256,
    reportSha256,
    verdict: (report.decision as { verdict: string }).verdict,
  }, null, 2)}\n`);
  console.log(JSON.stringify({
    output: relative(resolve('.'), output),
    recipeSha256,
    analyzerSha256,
    decision: report.decision,
    elevationAligned: report.elevationAligned,
  }, null, 2));
}

if (resolve(process.argv[1] ?? '') === fileURLToPath(import.meta.url)) {
  await main();
}
