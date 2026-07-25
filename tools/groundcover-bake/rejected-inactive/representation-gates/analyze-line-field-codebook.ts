/**
 * Offline categorical codebook audit for a GCRP/v4 line field.
 *
 * Every cluster is conditioned on the exact packed owner token. Therefore the
 * experiment can never blend unrelated owners, convert hit to miss, or move an
 * owner discontinuity. Only depth and oct-normal are represented by a centroid
 * within one owner and one nonlinear quantisation cell.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import {
  derivePeriodicProfileLattice,
  parsePeriodicProfile,
} from '../../src/nanite/groundcover/GroundCoverPeriodicProfile';

const EXPECTED_CALAMAGROSTIS_SHA256 =
  '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const MISS = 0xffff_ffff;
const PHASE_BLOCKS = 16;

interface QuantizerConfig {
  name: string;
  depthBits: number;
  normalBits: number;
}

interface ScalarAggregate {
  count: number;
  sumAbs: number;
  sumSq: number;
  maximum: number;
}

interface PairAggregate {
  depth: ScalarAggregate;
  normalAngleDeg: ScalarAggregate;
}

interface ClusterAccumulator {
  count: number;
  sumDepthWeight: number;
  sumWeightedDepth: number;
  sumOctX: number;
  sumOctY: number;
  depth: number;
  octX: number;
  octY: number;
  id: number;
}

const CONFIGS: readonly QuantizerConfig[] = [
  { name: 'owner-centroid', depthBits: 0, normalBits: 0 },
  { name: 'd8-n4', depthBits: 8, normalBits: 4 },
  { name: 'd10-n6', depthBits: 10, normalBits: 6 },
  { name: 'd12-n8', depthBits: 12, normalBits: 8 },
  { name: 'd14-n10', depthBits: 14, normalBits: 10 },
  { name: 'd16-n12', depthBits: 16, normalBits: 12 },
  { name: 'exact-d16-n16', depthBits: 16, normalBits: 16 },
];

function parseArg(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function emptyScalar(): ScalarAggregate {
  return { count: 0, sumAbs: 0, sumSq: 0, maximum: 0 };
}

function emptyPair(): PairAggregate {
  return { depth: emptyScalar(), normalAngleDeg: emptyScalar() };
}

function addScalar(aggregate: ScalarAggregate, value: number): void {
  aggregate.count++;
  aggregate.sumAbs += Math.abs(value);
  aggregate.sumSq += value * value;
  aggregate.maximum = Math.max(aggregate.maximum, Math.abs(value));
}

function addPair(aggregate: PairAggregate, depth: number, normalAngleDeg: number): void {
  addScalar(aggregate.depth, depth);
  addScalar(aggregate.normalAngleDeg, normalAngleDeg);
}

function finishScalar(aggregate: ScalarAggregate): Record<string, number> {
  return {
    count: aggregate.count,
    meanAbs: aggregate.count > 0 ? aggregate.sumAbs / aggregate.count : 0,
    rms: aggregate.count > 0 ? Math.sqrt(aggregate.sumSq / aggregate.count) : 0,
    maximum: aggregate.maximum,
  };
}

function finishPair(aggregate: PairAggregate): Record<string, unknown> {
  return {
    depthMetres: finishScalar(aggregate.depth),
    normalAngleDegrees: finishScalar(aggregate.normalAngleDeg),
  };
}

function quantile(sorted: Float32Array, fraction: number): number {
  if (sorted.length === 0) return 0;
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * fraction))]!;
}

function lowerBound(sorted: Float32Array, value: number): number {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (sorted[mid]! <= value) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function distribution(
  values: Float32Array,
  thresholds: readonly number[],
): Record<string, unknown> {
  values.sort();
  return {
    p50: quantile(values, 0.5),
    p90: quantile(values, 0.9),
    p95: quantile(values, 0.95),
    p99: quantile(values, 0.99),
    maximum: values.length > 0 ? values[values.length - 1] : 0,
    thresholdAgreement: Object.fromEntries(thresholds.map((threshold) => [
      String(threshold),
      values.length > 0 ? lowerBound(values, threshold) / values.length : 1,
    ])),
  };
}

function decodeOctU16(xU16: number, yU16: number): readonly [number, number, number] {
  let x = xU16 / 65535 * 2 - 1;
  let y = yU16 / 65535 * 2 - 1;
  let z = 1 - Math.abs(x) - Math.abs(y);
  if (z < 0) {
    const oldX = x;
    x = (1 - Math.abs(y)) * (oldX < 0 ? -1 : 1);
    y = (1 - Math.abs(oldX)) * (y < 0 ? -1 : 1);
  }
  const invLength = 1 / Math.hypot(x, y, z);
  return [x * invLength, y * invLength, z * invLength];
}

function normalAngleDegrees(
  sourceX: number,
  sourceY: number,
  reconstructedX: number,
  reconstructedY: number,
): number {
  if (sourceX === reconstructedX && sourceY === reconstructedY) return 0;
  const a = decodeOctU16(sourceX, sourceY);
  const b = decodeOctU16(reconstructedX, reconstructedY);
  const dot = Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]));
  return Math.acos(dot) * 180 / Math.PI;
}

function radixSortByOwner(indices: Uint32Array, owners: Uint32Array): Uint32Array {
  const scratch = new Uint32Array(indices.length);
  const counts = new Uint32Array(65536);
  const offsets = new Uint32Array(65536);
  const pass = (source: Uint32Array, target: Uint32Array, shift: number): void => {
    counts.fill(0);
    for (let i = 0; i < source.length; i++) {
      counts[(owners[source[i]!]! >>> shift) & 0xffff]++;
    }
    let offset = 0;
    for (let key = 0; key < counts.length; key++) {
      offsets[key] = offset;
      offset += counts[key]!;
    }
    for (let i = 0; i < source.length; i++) {
      const record = source[i]!;
      const key = (owners[record]! >>> shift) & 0xffff;
      target[offsets[key]!] = record;
      offsets[key]++;
    }
  };
  pass(indices, scratch, 0);
  pass(scratch, indices, 16);
  return indices;
}

function quantizedKey(
  depth: number,
  octX: number,
  octY: number,
  depthBits: number,
  normalBits: number,
): number {
  const qDepth = depthBits === 0 ? 0 : depth >>> (16 - depthBits);
  const qOctX = normalBits === 0 ? 0 : octX >>> (16 - normalBits);
  const qOctY = normalBits === 0 ? 0 : octY >>> (16 - normalBits);
  const depthBase = 2 ** depthBits;
  const normalBase = 2 ** normalBits;
  return qDepth + depthBase * (qOctX + normalBase * qOctY);
}

function reuseBucket(count: number): string {
  if (count === 1) return '1';
  if (count === 2) return '2';
  if (count <= 4) return '3-4';
  if (count <= 8) return '5-8';
  if (count <= 16) return '9-16';
  if (count <= 32) return '17-32';
  if (count <= 64) return '33-64';
  if (count <= 256) return '65-256';
  return '257+';
}

function ownerFrequencyBucket(count: number): string {
  if (count === 1) return '1';
  if (count <= 4) return '2-4';
  if (count <= 16) return '5-16';
  if (count <= 64) return '17-64';
  if (count <= 256) return '65-256';
  return '257+';
}

const source = resolve(parseArg('--source') ?? 'src/assets/groundcover/calamagrostis-canescens.gcrp');
const output = resolve(
  parseArg('--output')
    ?? `data/work/groundcover-line-field-codebook/${EXPECTED_CALAMAGROSTIS_SHA256}/metrics.json`,
);
const sourceBytes = readFileSync(source);
const sourceSha256 = createHash('sha256').update(sourceBytes).digest('hex');
if (sourceSha256 !== EXPECTED_CALAMAGROSTIS_SHA256) {
  throw new Error(
    `codebook audit is bound to Calamagrostis ${EXPECTED_CALAMAGROSTIS_SHA256}, got ${sourceSha256}`,
  );
}

const profile = parsePeriodicProfile(sourceBytes);
if (profile.version !== 4 || !profile.ownerTexels) {
  throw new Error('line-field codebook audit requires GCRP/v4 owner texels');
}
const lattice = derivePeriodicProfileLattice(profile);
const owners = profile.ownerTexels;
const texels = profile.texels;
const texelCount = owners.length;
const atlasWidth = profile.storedTileWidth * profile.atlasColumns;
const atlasHeight = profile.storedTileHeight * profile.atlasRows;
const sliceCount = profile.slices.length;

const sliceAtTexel = new Uint8Array(texelCount);
const interiorAtTexel = new Uint8Array(texelCount);
const phaseBlockAtTexel = new Uint8Array(texelCount);
const spatialBoundaryAtTexel = new Uint8Array(texelCount);
const angularBoundaryAtTexel = new Uint8Array(texelCount);
const elevationAtSlice = new Uint8Array(sliceCount);
const azimuthAtSlice = new Uint8Array(sliceCount);
const depthSpanAtSlice = new Float64Array(sliceCount);

const sliceIndex = (azimuth: number, elevation: number): number =>
  lattice.order === 'azimuth-major'
    ? ((azimuth % lattice.azimuthCount) + lattice.azimuthCount) % lattice.azimuthCount
      * lattice.elevationCount + elevation
    : elevation * lattice.azimuthCount
      + ((azimuth % lattice.azimuthCount) + lattice.azimuthCount) % lattice.azimuthCount;

for (let azimuth = 0; azimuth < lattice.azimuthCount; azimuth++) {
  for (let elevation = 0; elevation < lattice.elevationCount; elevation++) {
    const slice = sliceIndex(azimuth, elevation);
    azimuthAtSlice[slice] = azimuth;
    elevationAtSlice[slice] = elevation;
    const bounds = profile.slices[slice]!;
    depthSpanAtSlice[slice] = bounds.depthMax - bounds.depthMin;
  }
}

const texelIndex = (slice: number, phaseX: number, phaseZ: number): number => {
  const wrappedX = ((phaseX % profile.interiorTileWidth) + profile.interiorTileWidth)
    % profile.interiorTileWidth;
  const wrappedZ = ((phaseZ % profile.interiorTileHeight) + profile.interiorTileHeight)
    % profile.interiorTileHeight;
  const tileX = slice % profile.atlasColumns;
  const tileY = Math.floor(slice / profile.atlasColumns);
  const x = tileX * profile.storedTileWidth + profile.gutter + wrappedX;
  const y = tileY * profile.storedTileHeight + profile.gutter + wrappedZ;
  return y * atlasWidth + x;
};

for (let y = 0; y < atlasHeight; y++) {
  const tileY = Math.floor(y / profile.storedTileHeight);
  const localY = y % profile.storedTileHeight;
  for (let x = 0; x < atlasWidth; x++) {
    const tileX = Math.floor(x / profile.storedTileWidth);
    const localX = x % profile.storedTileWidth;
    const index = y * atlasWidth + x;
    const slice = tileY * profile.atlasColumns + tileX;
    sliceAtTexel[index] = slice;
    const phaseX = localX - profile.gutter;
    const phaseZ = localY - profile.gutter;
    const interior = phaseX >= 0
      && phaseX < profile.interiorTileWidth
      && phaseZ >= 0
      && phaseZ < profile.interiorTileHeight;
    if (interior) {
      interiorAtTexel[index] = 1;
      const blockX = Math.min(
        PHASE_BLOCKS - 1,
        Math.floor(phaseX * PHASE_BLOCKS / profile.interiorTileWidth),
      );
      const blockZ = Math.min(
        PHASE_BLOCKS - 1,
        Math.floor(phaseZ * PHASE_BLOCKS / profile.interiorTileHeight),
      );
      phaseBlockAtTexel[index] = blockZ * PHASE_BLOCKS + blockX;
    }
  }
}

for (let slice = 0; slice < sliceCount; slice++) {
  const azimuth = azimuthAtSlice[slice]!;
  const elevation = elevationAtSlice[slice]!;
  for (let phaseZ = 0; phaseZ < profile.interiorTileHeight; phaseZ++) {
    for (let phaseX = 0; phaseX < profile.interiorTileWidth; phaseX++) {
      const index = texelIndex(slice, phaseX, phaseZ);
      const owner = owners[index]!;
      const spatialBoundary = owners[texelIndex(slice, phaseX - 1, phaseZ)] !== owner
        || owners[texelIndex(slice, phaseX + 1, phaseZ)] !== owner
        || owners[texelIndex(slice, phaseX, phaseZ - 1)] !== owner
        || owners[texelIndex(slice, phaseX, phaseZ + 1)] !== owner;
      spatialBoundaryAtTexel[index] = spatialBoundary ? 1 : 0;
      let angularBoundary = false;
      if (lattice.azimuthCount > 1) {
        angularBoundary = owners[texelIndex(sliceIndex(azimuth - 1, elevation), phaseX, phaseZ)] !== owner
          || owners[texelIndex(sliceIndex(azimuth + 1, elevation), phaseX, phaseZ)] !== owner;
      }
      if (elevation > 0) {
        angularBoundary ||= owners[texelIndex(sliceIndex(azimuth, elevation - 1), phaseX, phaseZ)] !== owner;
      }
      if (elevation + 1 < lattice.elevationCount) {
        angularBoundary ||= owners[texelIndex(sliceIndex(azimuth, elevation + 1), phaseX, phaseZ)] !== owner;
      }
      angularBoundaryAtTexel[index] = angularBoundary ? 1 : 0;
    }
  }
}

let hitCount = 0;
let interiorHitCount = 0;
let ownerCoverageMismatch = 0;
let partialCoverageCount = 0;
for (let index = 0; index < texelCount; index++) {
  const coverage = texels[index * 4 + 3]!;
  const hitByCoverage = coverage !== 0;
  const hitByOwner = owners[index] !== MISS;
  if (hitByCoverage !== hitByOwner) ownerCoverageMismatch++;
  if (coverage !== 0 && coverage !== 65535) partialCoverageCount++;
  if (hitByCoverage && hitByOwner) {
    hitCount++;
    if (interiorAtTexel[index]) interiorHitCount++;
  }
}
if (ownerCoverageMismatch !== 0 || partialCoverageCount !== 0) {
  throw new Error(
    `source categorical contract failed: owner/coverage mismatch=${ownerCoverageMismatch}, partial=${partialCoverageCount}`,
  );
}

const sortedHitIndices = new Uint32Array(hitCount);
for (let index = 0, target = 0; index < texelCount; index++) {
  if (owners[index] !== MISS) sortedHitIndices[target++] = index;
}
radixSortByOwner(sortedHitIndices, owners);

let uniqueOwners = 0;
let singletonOwners = 0;
let maximumOwnerFrequency = 0;
for (let begin = 0; begin < sortedHitIndices.length;) {
  const owner = owners[sortedHitIndices[begin]!]!;
  let end = begin + 1;
  while (end < sortedHitIndices.length && owners[sortedHitIndices[end]!] === owner) end++;
  const frequency = end - begin;
  uniqueOwners++;
  if (frequency === 1) singletonOwners++;
  maximumOwnerFrequency = Math.max(maximumOwnerFrequency, frequency);
  begin = end;
}

const reports: Record<string, unknown>[] = [];

for (const config of CONFIGS) {
  const assignment = new Uint32Array(texelCount);
  const interiorDepthErrors = new Float32Array(interiorHitCount);
  const interiorNormalErrors = new Float32Array(interiorHitCount);
  const allAggregate = emptyPair();
  const interiorAggregate = emptyPair();
  const spatialBoundaryAggregate = emptyPair();
  const spatialInteriorAggregate = emptyPair();
  const angularBoundaryAggregate = emptyPair();
  const angularInteriorAggregate = emptyPair();
  const perElevation = Array.from({ length: lattice.elevationCount }, emptyPair);
  const perSlice = Array.from({ length: sliceCount }, emptyPair);
  const phaseHeatmap = Array.from({ length: PHASE_BLOCKS * PHASE_BLOCKS }, emptyPair);
  const frequencyAggregates = new Map<string, PairAggregate>();
  const reuseHistogram = new Map<string, number>();
  let codewordCount = 1; // one exact miss codeword
  let interiorErrorIndex = 0;

  for (let begin = 0; begin < sortedHitIndices.length;) {
    const owner = owners[sortedHitIndices[begin]!]!;
    let end = begin + 1;
    while (end < sortedHitIndices.length && owners[sortedHitIndices[end]!] === owner) end++;
    const ownerFrequency = end - begin;
    const clusters = new Map<number, ClusterAccumulator>();
    for (let cursor = begin; cursor < end; cursor++) {
      const record = sortedHitIndices[cursor]!;
      const source = record * 4;
      const depth = texels[source]!;
      const octX = texels[source + 1]!;
      const octY = texels[source + 2]!;
      const key = quantizedKey(depth, octX, octY, config.depthBits, config.normalBits);
      let cluster = clusters.get(key);
      if (!cluster) {
        cluster = {
          count: 0,
          sumDepthWeight: 0,
          sumWeightedDepth: 0,
          sumOctX: 0,
          sumOctY: 0,
          depth: 0,
          octX: 0,
          octY: 0,
          id: 0,
        };
        clusters.set(key, cluster);
      }
      const span = depthSpanAtSlice[sliceAtTexel[record]!]!;
      const depthWeight = span * span;
      cluster.count++;
      cluster.sumDepthWeight += depthWeight;
      cluster.sumWeightedDepth += depth * depthWeight;
      cluster.sumOctX += octX;
      cluster.sumOctY += octY;
    }
    for (const cluster of clusters.values()) {
      cluster.depth = Math.max(0, Math.min(
        65534,
        Math.round(cluster.sumWeightedDepth / cluster.sumDepthWeight),
      ));
      cluster.octX = Math.max(0, Math.min(65535, Math.round(cluster.sumOctX / cluster.count)));
      cluster.octY = Math.max(0, Math.min(65535, Math.round(cluster.sumOctY / cluster.count)));
      cluster.id = codewordCount++;
      const bucket = reuseBucket(cluster.count);
      reuseHistogram.set(bucket, (reuseHistogram.get(bucket) ?? 0) + 1);
    }
    const frequencyBucket = ownerFrequencyBucket(ownerFrequency);
    let frequencyAggregate = frequencyAggregates.get(frequencyBucket);
    if (!frequencyAggregate) {
      frequencyAggregate = emptyPair();
      frequencyAggregates.set(frequencyBucket, frequencyAggregate);
    }
    for (let cursor = begin; cursor < end; cursor++) {
      const record = sortedHitIndices[cursor]!;
      const source = record * 4;
      const depth = texels[source]!;
      const octX = texels[source + 1]!;
      const octY = texels[source + 2]!;
      const key = quantizedKey(depth, octX, octY, config.depthBits, config.normalBits);
      const cluster = clusters.get(key)!;
      assignment[record] = cluster.id;
      const slice = sliceAtTexel[record]!;
      const depthError = Math.abs(depth - cluster.depth) / 65535 * depthSpanAtSlice[slice]!;
      const angleError = normalAngleDegrees(octX, octY, cluster.octX, cluster.octY);
      addPair(allAggregate, depthError, angleError);
      addPair(frequencyAggregate, depthError, angleError);
      if (interiorAtTexel[record]) {
        interiorDepthErrors[interiorErrorIndex] = depthError;
        interiorNormalErrors[interiorErrorIndex] = angleError;
        interiorErrorIndex++;
        addPair(interiorAggregate, depthError, angleError);
        addPair(perElevation[elevationAtSlice[slice]!]!, depthError, angleError);
        addPair(perSlice[slice]!, depthError, angleError);
        addPair(phaseHeatmap[phaseBlockAtTexel[record]!]!, depthError, angleError);
        addPair(
          spatialBoundaryAtTexel[record] ? spatialBoundaryAggregate : spatialInteriorAggregate,
          depthError,
          angleError,
        );
        addPair(
          angularBoundaryAtTexel[record] ? angularBoundaryAggregate : angularInteriorAggregate,
          depthError,
          angleError,
        );
      }
    }
    begin = end;
  }

  if (interiorErrorIndex !== interiorHitCount) {
    throw new Error(`interior error count mismatch for ${config.name}`);
  }

  let spatialEdges = 0;
  let spatialSameCodeword = 0;
  let angularEdges = 0;
  let angularSameCodeword = 0;
  for (let slice = 0; slice < sliceCount; slice++) {
    const azimuth = azimuthAtSlice[slice]!;
    const elevation = elevationAtSlice[slice]!;
    for (let phaseZ = 0; phaseZ < profile.interiorTileHeight; phaseZ++) {
      for (let phaseX = 0; phaseX < profile.interiorTileWidth; phaseX++) {
        const record = texelIndex(slice, phaseX, phaseZ);
        const right = texelIndex(slice, phaseX + 1, phaseZ);
        const down = texelIndex(slice, phaseX, phaseZ + 1);
        spatialEdges += 2;
        if (assignment[record] === assignment[right]) spatialSameCodeword++;
        if (assignment[record] === assignment[down]) spatialSameCodeword++;
        if (lattice.azimuthCount > 1) {
          const nextAzimuth = texelIndex(sliceIndex(azimuth + 1, elevation), phaseX, phaseZ);
          angularEdges++;
          if (assignment[record] === assignment[nextAzimuth]) angularSameCodeword++;
        }
        if (elevation + 1 < lattice.elevationCount) {
          const nextElevation = texelIndex(sliceIndex(azimuth, elevation + 1), phaseX, phaseZ);
          angularEdges++;
          if (assignment[record] === assignment[nextElevation]) angularSameCodeword++;
        }
      }
    }
  }

  const minimumIndexBits = Math.max(1, Math.ceil(Math.log2(codewordCount)));
  const idealPackedIndexBytes = Math.ceil(texelCount * minimumIndexBits / 8);
  const baselineCompleteBytes = texelCount * 12;
  const r32IndexBytes = texelCount * 4;
  const compactCodewordBytes = codewordCount * 10;
  const bufferCodewordBytes = codewordCount * 12;
  const textureCodewordBytes = codewordCount * 16;
  const directionRows = lattice.elevations.map((elevation, index) => ({
    elevationDegrees: elevation * 180 / Math.PI,
    ...finishPair(perElevation[index]!),
  }));
  const worstSlices = perSlice
    .map((aggregate, slice) => ({
      slice,
      azimuth: azimuthAtSlice[slice],
      elevation: elevationAtSlice[slice],
      depthRms: aggregate.depth.count > 0
        ? Math.sqrt(aggregate.depth.sumSq / aggregate.depth.count)
        : 0,
      depthMaximum: aggregate.depth.maximum,
      normalAngleRms: aggregate.normalAngleDeg.count > 0
        ? Math.sqrt(aggregate.normalAngleDeg.sumSq / aggregate.normalAngleDeg.count)
        : 0,
    }))
    .sort((a, b) => b.depthRms - a.depthRms)
    .slice(0, 8);

  reports.push({
    config,
    categoricalAgreement: {
      hitMiss: 1,
      ownerToken: 1,
      unrelatedOwnerAverages: 0,
    },
    codebook: {
      entriesIncludingMiss: codewordCount,
      hitEntries: codewordCount - 1,
      entriesPerExactOwner: (codewordCount - 1) / uniqueOwners,
      minimumIndexBits,
      reuseHistogram: Object.fromEntries(reuseHistogram),
      spatialSameCodewordEdgeFraction: spatialEdges > 0 ? spatialSameCodeword / spatialEdges : 1,
      angularSameCodewordEdgeFraction: angularEdges > 0 ? angularSameCodeword / angularEdges : 1,
    },
    storage: {
      baselineCompleteBytes,
      idealPacked: {
        indexBytes: idealPackedIndexBytes,
        codewordBytes10: compactCodewordBytes,
        totalBytes: idealPackedIndexBytes + compactCodewordBytes,
        ratioToBaseline: (idealPackedIndexBytes + compactCodewordBytes) / baselineCompleteBytes,
        caveat: 'bit-packed indices crossing a 32-bit word can require two index reads',
      },
      fixedTwoReadBuffer: {
        indexBytesR32Uint: r32IndexBytes,
        codewordBytes12: bufferCodewordBytes,
        totalBytes: r32IndexBytes + bufferCodewordBytes,
        ratioToBaseline: (r32IndexBytes + bufferCodewordBytes) / baselineCompleteBytes,
        reads: ['one r32uint index', 'one dependent 12-byte codeword buffer record'],
      },
      fixedTwoReadTexture: {
        indexBytesR32Uint: r32IndexBytes,
        codewordBytes16: textureCodewordBytes,
        totalBytes: r32IndexBytes + textureCodewordBytes,
        ratioToBaseline: (r32IndexBytes + textureCodewordBytes) / baselineCompleteBytes,
        reads: ['one r32uint index', 'one dependent rgba32uint codeword texel'],
      },
    },
    errors: {
      allStoredTexels: finishPair(allAggregate),
      interiorTexels: {
        ...finishPair(interiorAggregate),
        depthDistributionMetres: distribution(
          interiorDepthErrors,
          [0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01],
        ),
        normalAngleDistributionDegrees: distribution(
          interiorNormalErrors,
          [0.1, 0.25, 0.5, 1, 2, 5],
        ),
      },
      topologyClasses: {
        spatialOwnerBoundary: finishPair(spatialBoundaryAggregate),
        spatialOwnerInterior: finishPair(spatialInteriorAggregate),
        angularOwnerBoundary: finishPair(angularBoundaryAggregate),
        angularOwnerInterior: finishPair(angularInteriorAggregate),
      },
      byOwnerFrequency: Object.fromEntries(
        [...frequencyAggregates.entries()].map(([bucket, aggregate]) => [bucket, finishPair(aggregate)]),
      ),
      byElevation: directionRows,
      worstSlices,
      phaseHeatmap16x16: phaseHeatmap.map((aggregate) => ({
        count: aggregate.depth.count,
        depthMeanAbs: aggregate.depth.count > 0
          ? aggregate.depth.sumAbs / aggregate.depth.count
          : 0,
        depthMaximum: aggregate.depth.maximum,
        normalAngleMeanAbs: aggregate.normalAngleDeg.count > 0
          ? aggregate.normalAngleDeg.sumAbs / aggregate.normalAngleDeg.count
          : 0,
      })),
    },
  });
}

const report = {
  schema: 'laas-groundcover-line-field-codebook-audit/v1',
  source: {
    path: source,
    bytes: sourceBytes.byteLength,
    sha256: sourceSha256,
    requiredSha256: EXPECTED_CALAMAGROSTIS_SHA256,
  },
  field: {
    version: profile.version,
    storedTile: [profile.storedTileWidth, profile.storedTileHeight],
    interiorTile: [profile.interiorTileWidth, profile.interiorTileHeight],
    atlas: [atlasWidth, atlasHeight],
    slices: sliceCount,
    azimuths: lattice.azimuthCount,
    elevationsDegrees: lattice.elevations.map((value) => value * 180 / Math.PI),
    texels: texelCount,
    hits: hitCount,
    misses: texelCount - hitCount,
    interiorHits: interiorHitCount,
    uniqueExactOwnerTokens: uniqueOwners,
    singletonOwnerTokens: singletonOwners,
    maximumOwnerFrequency,
    ownerCoverageMismatch,
    partialCoverageCount,
  },
  invariant: {
    clusteringKey: 'exact packed owner token + quantised depth/oct-normal cell',
    hitMissPreserved: true,
    ownerTokenPreserved: true,
    unrelatedOwnersNeverShareCodeword: true,
    runtimeSearch: false,
    runtimeCandidates: false,
  },
  reports,
};

mkdirSync(dirname(output), { recursive: true });
writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ output, sourceSha256, texelCount, hitCount, uniqueOwners, reports: reports.map((entry) => ({
  config: (entry.config as QuantizerConfig).name,
  entries: (entry.codebook as { entriesIncludingMiss: number }).entriesIncludingMiss,
  twoReadBufferBytes: ((entry.storage as Record<string, { totalBytes: number }>).fixedTwoReadBuffer).totalBytes,
  depthP95: (((entry.errors as Record<string, unknown>).interiorTexels as Record<string, Record<string, number>>).depthDistributionMetres).p95,
  normalP95: (((entry.errors as Record<string, unknown>).interiorTexels as Record<string, Record<string, number>>).normalAngleDistributionDegrees).p95,
})) }, null, 2));
