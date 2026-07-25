/**
 * Offline visual/event-measure codec audit for the accepted Calamagrostis
 * GCRP/v4 asset. This deliberately does not preserve or reconstruct triangle
 * owners. A block records a band-limited first-hit event measure:
 *
 *   C, C E[u], C E[u^2], C E[rgb], C E[n]
 *
 * where u is the slice-normalised ray parameter. A consumer may reduce this
 * distribution to one representative event at E[u], but the carried standard
 * deviation is the unresolved line-depth uncertainty, not geometry thickness
 * and not an exact source-surface claim.
 *
 * No runtime shader or asset is produced here. The executable emits metrics and
 * numbered visual QA only.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';
import { DataUtils } from 'three';
import {
  derivePeriodicProfileLattice,
  makePeriodicProfileColorTexture,
  parsePeriodicProfile,
} from '../../src/nanite/groundcover/GroundCoverPeriodicProfile';
import type { PeriodicProfileData } from '../../src/nanite/groundcover/GroundCoverProfileTypes';

const EXPECTED_SOURCE_SHA256 =
  '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const COMPONENTS = 9;
const C = 0;
const M1 = 1;
const M2 = 2;
const CR = 3;
const CG = 4;
const CB = 5;
const NX = 6;
const NY = 7;
const NZ = 8;
const PANEL = 256;
const LABEL = 28;
const PHASE_BINS = 16;
const MISS_OWNER = 0xffff_ffff;

const RECIPE = {
  schema: 'laas-groundcover-event-measure-codec/v1',
  sourceSha256: EXPECTED_SOURCE_SHA256,
  measure: ['coverage', 'coverage*E[u]', 'coverage*E[u^2]', 'coverage*E[rgb]', 'coverage*E[n]'],
  blockFilters: [
    'index-2 ray lattice B=[2v,w], det(v,w)=+/-1',
    '2x1 nearest-axis ray-aligned',
    '2x1 nearest-axis cross-ray',
    '2x2',
    '4x4',
  ],
  directLayout: {
    bytesPerBlock: 16,
    reads: ['rgba16float moments', 'rgba8unorm premultiplied colour', 'rgba8snorm normal moment'],
  },
  conditionalLayout: {
    bytesPerBlock: 12,
    warning: 'nonlinear conditional attributes are filtered directly and therefore do not preserve the event measure',
  },
  vq: [
    { name: 'vq-low', depthBits: 5, sigmaBits: 2, colorBits: 2, normalBits: 2, coherenceBits: 2 },
    { name: 'vq-balanced', depthBits: 6, sigmaBits: 3, colorBits: 3, normalBits: 2, coherenceBits: 3 },
    { name: 'vq-mid', depthBits: 6, sigmaBits: 3, colorBits: 3, normalBits: 3, coherenceBits: 3 },
    { name: 'vq-high', depthBits: 8, sigmaBits: 4, colorBits: 4, normalBits: 4, coherenceBits: 4 },
  ],
  qaSlices: [
    { number: 1, azimuth: 0, elevation: 0, label: 'grazing +X, 15 degrees' },
    { number: 2, azimuth: 4, elevation: 0, label: 'grazing +Z, 15 degrees' },
    { number: 3, azimuth: 2, elevation: 1, label: 'diagonal oblique, 35 degrees' },
    { number: 4, azimuth: 0, elevation: 3, label: 'near-top, 75 degrees' },
  ],
} as const;

interface BlockField {
  blockSizeX: number;
  blockSizeY: number;
  width: number;
  height: number;
  sliceSwap: Uint8Array;
  latticeVectors?: Int8Array;
  label: string;
  values: Float32Array;
}

interface CodecField {
  name: string;
  blockSizeX: number;
  blockSizeY: number;
  width: number;
  height: number;
  sliceSwap: Uint8Array;
  latticeVectors?: Int8Array;
  values: Float32Array;
  interpolation: 'bilinear-measure' | 'bilinear-conditional' | 'nearest-index';
  storage: Record<string, unknown>;
  codebookEntries?: number;
}

interface ScalarAggregate {
  count: number;
  sumAbs: number;
  sumSq: number;
  maximum: number;
}

interface CodecAggregate {
  samples: number;
  coverage: ScalarAggregate;
  premulColor: ScalarAggregate;
  premulDepth: ScalarAggregate;
  premulSecondMoment: ScalarAggregate;
  normalMoment: ScalarAggregate;
  rayDepth: ScalarAggregate;
  height: ScalarAggregate;
  normalAngle: ScalarAggregate;
  orientationCoherence: ScalarAggregate;
  uncertaintyRay: ScalarAggregate;
  falseLowCoverage: number;
  truthHits: number;
  intersection: number;
  union: number;
}

interface VqConfig {
  name: string;
  depthBits: number;
  sigmaBits: number;
  colorBits: number;
  normalBits: number;
  coherenceBits: number;
}

interface VqAccumulator {
  count: number;
  sums: Float64Array;
}

function sha256(bytes: Uint8Array | string): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) =>
    `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(',')}}`;
}

function clamp(value: number, minimum = 0, maximum = 1): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function mod(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus;
}

function emptyScalar(): ScalarAggregate {
  return { count: 0, sumAbs: 0, sumSq: 0, maximum: 0 };
}

function emptyCodecAggregate(): CodecAggregate {
  return {
    samples: 0,
    coverage: emptyScalar(),
    premulColor: emptyScalar(),
    premulDepth: emptyScalar(),
    premulSecondMoment: emptyScalar(),
    normalMoment: emptyScalar(),
    rayDepth: emptyScalar(),
    height: emptyScalar(),
    normalAngle: emptyScalar(),
    orientationCoherence: emptyScalar(),
    uncertaintyRay: emptyScalar(),
    falseLowCoverage: 0,
    truthHits: 0,
    intersection: 0,
    union: 0,
  };
}

function addScalar(target: ScalarAggregate, error: number): void {
  const magnitude = Math.abs(error);
  target.count++;
  target.sumAbs += magnitude;
  target.sumSq += error * error;
  target.maximum = Math.max(target.maximum, magnitude);
}

function finishScalar(target: ScalarAggregate): Record<string, number> {
  return {
    count: target.count,
    meanAbs: target.count > 0 ? target.sumAbs / target.count : 0,
    rms: target.count > 0 ? Math.sqrt(target.sumSq / target.count) : 0,
    maximum: target.maximum,
  };
}

function quantiles(values: number[]): Record<string, number> {
  values.sort((a, b) => a - b);
  const at = (fraction: number): number => values.length === 0
    ? 0
    : values[Math.floor((values.length - 1) * fraction)]!;
  return {
    p01: at(0.01),
    p05: at(0.05),
    p50: at(0.5),
    p90: at(0.9),
    p95: at(0.95),
    p99: at(0.99),
    maximum: values.length > 0 ? values[values.length - 1]! : 0,
  };
}

function fractionsAtOrBelow(values: readonly number[], thresholds: readonly number[]): Record<string, number> {
  return Object.fromEntries(thresholds.map((threshold) => [
    String(threshold),
    values.length > 0 ? values.reduce((count, value) => count + (value <= threshold ? 1 : 0), 0) / values.length : 1,
  ]));
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
  const inverseLength = 1 / Math.max(1e-12, Math.hypot(x, y, z));
  return [x * inverseLength, y * inverseLength, z * inverseLength];
}

function encodeOctUnit(x: number, y: number, z: number): readonly [number, number] {
  const inverseL1 = 1 / Math.max(1e-12, Math.abs(x) + Math.abs(y) + Math.abs(z));
  let ox = x * inverseL1;
  let oy = y * inverseL1;
  if (z < 0) {
    const oldX = ox;
    ox = (1 - Math.abs(oy)) * (oldX < 0 ? -1 : 1);
    oy = (1 - Math.abs(oldX)) * (oy < 0 ? -1 : 1);
  }
  return [ox * 0.5 + 0.5, oy * 0.5 + 0.5];
}

function decodeOctUnit(ux: number, uy: number): readonly [number, number, number] {
  return decodeOctU16(Math.round(clamp(ux) * 65535), Math.round(clamp(uy) * 65535));
}

function normalAngleDegrees(
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
): number {
  const aLength = Math.hypot(ax, ay, az);
  const bLength = Math.hypot(bx, by, bz);
  if (aLength < 1e-8 || bLength < 1e-8) return 0;
  const dot = clamp((ax * bx + ay * by + az * bz) / (aLength * bLength), -1, 1);
  return Math.acos(dot) * 180 / Math.PI;
}

function half(value: number): number {
  return DataUtils.fromHalfFloat(DataUtils.toHalfFloat(value));
}

function unorm8(value: number): number {
  return Math.round(clamp(value) * 255) / 255;
}

function snorm8(value: number): number {
  return Math.round(clamp(value, -1, 1) * 127) / 127;
}

function quantizeUnit(value: number, bits: number): number {
  const maximum = 2 ** bits - 1;
  return Math.round(clamp(value) * maximum);
}

function packKey(fields: readonly { value: number; bits: number }[]): bigint {
  let key = 0n;
  for (const field of fields) {
    key = (key << BigInt(field.bits)) | BigInt(field.value);
  }
  return key;
}

function profileTexelIndex(
  profile: PeriodicProfileData,
  slice: number,
  x: number,
  y: number,
): number {
  const atlasWidth = profile.storedTileWidth * profile.atlasColumns;
  const tileX = slice % profile.atlasColumns;
  const tileY = Math.floor(slice / profile.atlasColumns);
  const storedX = tileX * profile.storedTileWidth + profile.gutter + mod(x, profile.interiorTileWidth);
  const storedY = tileY * profile.storedTileHeight + profile.gutter + mod(y, profile.interiorTileHeight);
  return storedY * atlasWidth + storedX;
}

function buildBlockField(
  profile: PeriodicProfileData,
  colorTexels: Uint8Array,
  blockSizeX: number,
  blockSizeY: number,
  orientation: 'fixed' | 'ray-aligned' | 'cross-ray' = 'fixed',
): BlockField {
  if (
    profile.interiorTileWidth % blockSizeX !== 0
    || profile.interiorTileHeight % blockSizeY !== 0
  ) {
    throw new Error(`block size ${blockSizeX}x${blockSizeY} does not divide the profile interior`);
  }
  const width = profile.interiorTileWidth / blockSizeX;
  const height = profile.interiorTileHeight / blockSizeY;
  const sliceSwap = new Uint8Array(profile.slices.length);
  const values = new Float32Array(profile.slices.length * width * height * COMPONENTS);
  const inverseArea = 1 / (blockSizeX * blockSizeY);
  for (let slice = 0; slice < profile.slices.length; slice++) {
    const direction = profile.slices[slice]!.direction;
    const zDominant = Math.abs(direction[2]) > Math.abs(direction[0]);
    const swap = orientation === 'ray-aligned'
      ? zDominant
      : orientation === 'cross-ray'
        ? !zDominant
        : false;
    sliceSwap[slice] = swap ? 1 : 0;
    for (let blockY = 0; blockY < height; blockY++) {
      for (let blockX = 0; blockX < width; blockX++) {
        const output = ((slice * height + blockY) * width + blockX) * COMPONENTS;
        for (let dy = 0; dy < blockSizeY; dy++) {
          for (let dx = 0; dx < blockSizeX; dx++) {
            const localX = blockX * blockSizeX + dx;
            const localY = blockY * blockSizeY + dy;
            const texel = profileTexelIndex(
              profile,
              slice,
              swap ? localY : localX,
              swap ? localX : localY,
            );
            const record = texel * 4;
            const coverage = profile.texels[record + 3]! / 65535;
            if (coverage <= 0) continue;
            const depth = profile.texels[record]! / 65535;
            const normal = decodeOctU16(profile.texels[record + 1]!, profile.texels[record + 2]!);
            values[output + C] += coverage * inverseArea;
            values[output + M1] += coverage * depth * inverseArea;
            values[output + M2] += coverage * depth * depth * inverseArea;
            values[output + CR] += colorTexels[record]! / 255 * inverseArea;
            values[output + CG] += colorTexels[record + 1]! / 255 * inverseArea;
            values[output + CB] += colorTexels[record + 2]! / 255 * inverseArea;
            values[output + NX] += coverage * normal[0] * inverseArea;
            values[output + NY] += coverage * normal[1] * inverseArea;
            values[output + NZ] += coverage * normal[2] * inverseArea;
          }
        }
      }
    }
  }
  return {
    blockSizeX,
    blockSizeY,
    width,
    height,
    sliceSwap,
    label: orientation === 'fixed' ? `${blockSizeX}x${blockSizeY}` : `${blockSizeX}x${blockSizeY}-${orientation}`,
    values,
  };
}

function primitiveRayVector(direction: readonly [number, number, number]): readonly [number, number] {
  let vx = Math.round(direction[0] * 2);
  let vy = Math.round(-direction[2] * 2);
  const gcd = (a: number, b: number): number => {
    let x = Math.abs(a);
    let y = Math.abs(b);
    while (y !== 0) [x, y] = [y, x % y];
    return Math.max(1, x);
  };
  const divisor = gcd(vx, vy);
  vx /= divisor;
  vy /= divisor;
  if (vx === 0 && vy === 0) return [1, 0];
  return [vx, vy];
}

function bezoutCrossVector(vx: number, vy: number): readonly [number, number] {
  let best: readonly [number, number] | null = null;
  let bestScore = Infinity;
  for (let wy = -4; wy <= 4; wy++) {
    for (let wx = -4; wx <= 4; wx++) {
      const determinant = vx * wy - vy * wx;
      if (Math.abs(determinant) !== 1) continue;
      const dot = vx * wx + vy * wy;
      const score = dot * dot + 0.01 * (wx * wx + wy * wy);
      if (score < bestScore) {
        bestScore = score;
        best = [wx, wy];
      }
    }
  }
  if (!best) throw new Error(`no Bezout cross vector for (${vx}, ${vy})`);
  return best;
}

/** Half-density periodic lattice whose coarse axis follows each slice's
 * projected ray. B=[2v,w] has determinant +/-2, so its 128x256 torus is a
 * bijective index-two sublattice of the 256x256 source torus. */
function buildRayLatticeBlockField(
  profile: PeriodicProfileData,
  colorTexels: Uint8Array,
): BlockField {
  if (profile.interiorTileWidth !== 256 || profile.interiorTileHeight !== 256) {
    throw new Error('ray-lattice audit is bound to the accepted 256x256 interior');
  }
  const width = 128;
  const height = 256;
  const values = new Float32Array(profile.slices.length * width * height * COMPONENTS);
  const latticeVectors = new Int8Array(profile.slices.length * 4);
  const sliceSwap = new Uint8Array(profile.slices.length);
  for (let slice = 0; slice < profile.slices.length; slice++) {
    const [vx, vy] = primitiveRayVector(profile.slices[slice]!.direction);
    const [wx, wy] = bezoutCrossVector(vx, vy);
    const determinant = vx * wy - vy * wx;
    if (Math.abs(determinant) !== 1) throw new Error('ray-lattice basis is not index two');
    latticeVectors.set([vx, vy, wx, wy], slice * 4);
    for (let blockY = 0; blockY < height; blockY++) {
      for (let blockX = 0; blockX < width; blockX++) {
        const output = ((slice * height + blockY) * width + blockX) * COMPONENTS;
        const baseX = mod(2 * blockX * vx + blockY * wx, 256);
        const baseY = mod(2 * blockX * vy + blockY * wy, 256);
        for (let event = 0; event < 2; event++) {
          const x = mod(baseX + event * vx, 256);
          const y = mod(baseY + event * vy, 256);
          const texel = profileTexelIndex(profile, slice, x, y);
          const record = texel * 4;
          const coverage = profile.texels[record + 3]! / 65535;
          if (coverage <= 0) continue;
          const depth = profile.texels[record]! / 65535;
          const normal = decodeOctU16(profile.texels[record + 1]!, profile.texels[record + 2]!);
          values[output + C] += coverage * 0.5;
          values[output + M1] += coverage * depth * 0.5;
          values[output + M2] += coverage * depth * depth * 0.5;
          values[output + CR] += colorTexels[record]! / 255 * 0.5;
          values[output + CG] += colorTexels[record + 1]! / 255 * 0.5;
          values[output + CB] += colorTexels[record + 2]! / 255 * 0.5;
          values[output + NX] += coverage * normal[0] * 0.5;
          values[output + NY] += coverage * normal[1] * 0.5;
          values[output + NZ] += coverage * normal[2] * 0.5;
        }
      }
    }
  }
  return {
    blockSizeX: 2,
    blockSizeY: 1,
    width,
    height,
    sliceSwap,
    latticeVectors,
    label: 'index2-ray-lattice',
    values,
  };
}

function sanitizeMeasure(values: Float32Array, offset: number): void {
  const coverage = clamp(values[offset + C]!);
  values[offset + C] = coverage;
  values[offset + M1] = clamp(values[offset + M1]!, 0, coverage);
  values[offset + M2] = clamp(values[offset + M2]!, 0, coverage);
  values[offset + CR] = clamp(values[offset + CR]!, 0, coverage);
  values[offset + CG] = clamp(values[offset + CG]!, 0, coverage);
  values[offset + CB] = clamp(values[offset + CB]!, 0, coverage);
  const nx = values[offset + NX]!;
  const ny = values[offset + NY]!;
  const nz = values[offset + NZ]!;
  const length = Math.hypot(nx, ny, nz);
  if (length > coverage && length > 1e-12) {
    const scale = coverage / length;
    values[offset + NX] = nx * scale;
    values[offset + NY] = ny * scale;
    values[offset + NZ] = nz * scale;
  }
}

function directMeasureCodec(source: BlockField): CodecField {
  const values = new Float32Array(source.values.length);
  for (let offset = 0; offset < values.length; offset += COMPONENTS) {
    values[offset + C] = half(source.values[offset + C]!);
    values[offset + M1] = half(source.values[offset + M1]!);
    values[offset + M2] = half(source.values[offset + M2]!);
    values[offset + CR] = unorm8(source.values[offset + CR]!);
    values[offset + CG] = unorm8(source.values[offset + CG]!);
    values[offset + CB] = unorm8(source.values[offset + CB]!);
    values[offset + NX] = snorm8(source.values[offset + NX]!);
    values[offset + NY] = snorm8(source.values[offset + NY]!);
    values[offset + NZ] = snorm8(source.values[offset + NZ]!);
    sanitizeMeasure(values, offset);
  }
  return {
    name: `measure-${source.label}`,
    blockSizeX: source.blockSizeX,
    blockSizeY: source.blockSizeY,
    width: source.width,
    height: source.height,
    sliceSwap: source.sliceSwap,
    ...(source.latticeVectors ? { latticeVectors: source.latticeVectors } : {}),
    values,
    interpolation: 'bilinear-measure',
    storage: directStorage(source.width, source.height, 16, 3),
  };
}

function conditionalCodec(source: BlockField): CodecField {
  const values = new Float32Array(source.values.length);
  for (let offset = 0; offset < values.length; offset += COMPONENTS) {
    const coverage = source.values[offset + C]!;
    values[offset + C] = unorm8(coverage);
    if (coverage <= 1e-8) continue;
    const mean = clamp(source.values[offset + M1]! / coverage);
    const second = clamp(source.values[offset + M2]! / coverage);
    const sigma = Math.sqrt(Math.max(0, second - mean * mean));
    const nx = source.values[offset + NX]! / coverage;
    const ny = source.values[offset + NY]! / coverage;
    const nz = source.values[offset + NZ]! / coverage;
    const coherence = clamp(Math.hypot(nx, ny, nz));
    const normalLength = Math.max(1e-12, coherence);
    const oct = encodeOctUnit(nx / normalLength, ny / normalLength, nz / normalLength);
    const decodedNormal = decodeOctUnit(unorm8(oct[0]), unorm8(oct[1]));
    const decodedCoherence = unorm8(coherence);
    values[offset + M1] = half(mean);
    values[offset + M2] = half(sigma);
    values[offset + CR] = unorm8(source.values[offset + CR]! / coverage);
    values[offset + CG] = unorm8(source.values[offset + CG]! / coverage);
    values[offset + CB] = unorm8(source.values[offset + CB]! / coverage);
    values[offset + NX] = decodedNormal[0] * decodedCoherence;
    values[offset + NY] = decodedNormal[1] * decodedCoherence;
    values[offset + NZ] = decodedNormal[2] * decodedCoherence;
  }
  return {
    name: `conditional-${source.label}`,
    blockSizeX: source.blockSizeX,
    blockSizeY: source.blockSizeY,
    width: source.width,
    height: source.height,
    sliceSwap: source.sliceSwap,
    ...(source.latticeVectors ? { latticeVectors: source.latticeVectors } : {}),
    values,
    interpolation: 'bilinear-conditional',
    storage: directStorage(source.width, source.height, 12, 3),
  };
}

function vqKey(source: Float32Array, offset: number, blockArea: number, config: VqConfig): bigint {
  const coverage = source[offset + C]!;
  const hitLevels = blockArea;
  const qCoverage = Math.round(coverage * hitLevels);
  if (coverage <= 1e-8) {
    return packKey([
      { value: qCoverage, bits: Math.ceil(Math.log2(hitLevels + 1)) },
      { value: 0, bits: config.depthBits + config.sigmaBits
        + config.colorBits * 3 + config.normalBits * 2 + config.coherenceBits },
    ]);
  }
  const mean = clamp(source[offset + M1]! / coverage);
  const second = clamp(source[offset + M2]! / coverage);
  const sigma = Math.sqrt(Math.max(0, second - mean * mean));
  const colorR = clamp(source[offset + CR]! / coverage);
  const colorG = clamp(source[offset + CG]! / coverage);
  const colorB = clamp(source[offset + CB]! / coverage);
  const nx = source[offset + NX]! / coverage;
  const ny = source[offset + NY]! / coverage;
  const nz = source[offset + NZ]! / coverage;
  const coherence = clamp(Math.hypot(nx, ny, nz));
  const inverseLength = 1 / Math.max(1e-12, coherence);
  const oct = encodeOctUnit(nx * inverseLength, ny * inverseLength, nz * inverseLength);
  return packKey([
    { value: qCoverage, bits: Math.ceil(Math.log2(hitLevels + 1)) },
    { value: quantizeUnit(mean, config.depthBits), bits: config.depthBits },
    // Square-root companding gives the low-variance end more cells. This is
    // only an offline partition key; decoded codewords remain linear moments.
    { value: quantizeUnit(Math.sqrt(sigma), config.sigmaBits), bits: config.sigmaBits },
    { value: quantizeUnit(Math.sqrt(colorR), config.colorBits), bits: config.colorBits },
    { value: quantizeUnit(Math.sqrt(colorG), config.colorBits), bits: config.colorBits },
    { value: quantizeUnit(Math.sqrt(colorB), config.colorBits), bits: config.colorBits },
    { value: quantizeUnit(oct[0], config.normalBits), bits: config.normalBits },
    { value: quantizeUnit(oct[1], config.normalBits), bits: config.normalBits },
    { value: quantizeUnit(coherence, config.coherenceBits), bits: config.coherenceBits },
  ]);
}

function vqCodec(source: BlockField, config: VqConfig): CodecField {
  const records = source.values.length / COMPONENTS;
  const assignments = new Uint32Array(records);
  const map = new Map<bigint, number>();
  const accumulators: VqAccumulator[] = [];
  for (let record = 0; record < records; record++) {
    const offset = record * COMPONENTS;
    const key = vqKey(
      source.values,
      offset,
      source.blockSizeX * source.blockSizeY,
      config,
    );
    let codeword = map.get(key);
    if (codeword === undefined) {
      codeword = accumulators.length;
      map.set(key, codeword);
      accumulators.push({ count: 0, sums: new Float64Array(COMPONENTS) });
    }
    assignments[record] = codeword;
    const accumulator = accumulators[codeword]!;
    accumulator.count++;
    for (let component = 0; component < COMPONENTS; component++) {
      accumulator.sums[component] += source.values[offset + component]!;
    }
  }
  const codebook = new Float32Array(accumulators.length * COMPONENTS);
  for (let codeword = 0; codeword < accumulators.length; codeword++) {
    const accumulator = accumulators[codeword]!;
    const offset = codeword * COMPONENTS;
    for (let component = 0; component < COMPONENTS; component++) {
      codebook[offset + component] = accumulator.sums[component]! / accumulator.count;
    }
    // Codewords use the same concrete 16-byte moment packing as the direct
    // path. The index is categorical and therefore nearest-only.
    codebook[offset + C] = half(codebook[offset + C]!);
    codebook[offset + M1] = half(codebook[offset + M1]!);
    codebook[offset + M2] = half(codebook[offset + M2]!);
    codebook[offset + CR] = unorm8(codebook[offset + CR]!);
    codebook[offset + CG] = unorm8(codebook[offset + CG]!);
    codebook[offset + CB] = unorm8(codebook[offset + CB]!);
    codebook[offset + NX] = snorm8(codebook[offset + NX]!);
    codebook[offset + NY] = snorm8(codebook[offset + NY]!);
    codebook[offset + NZ] = snorm8(codebook[offset + NZ]!);
    sanitizeMeasure(codebook, offset);
  }
  const values = new Float32Array(source.values.length);
  for (let record = 0; record < records; record++) {
    const target = record * COMPONENTS;
    const sourceOffset = assignments[record]! * COMPONENTS;
    values.set(codebook.subarray(sourceOffset, sourceOffset + COMPONENTS), target);
  }
  const indexBytes = accumulators.length <= 65535 ? 2 : 4;
  const storedSamples = storedSamplesForDimensions(source.width, source.height, 64);
  return {
    name: `${config.name}-${source.label}`,
    blockSizeX: source.blockSizeX,
    blockSizeY: source.blockSizeY,
    width: source.width,
    height: source.height,
    sliceSwap: source.sliceSwap,
    ...(source.latticeVectors ? { latticeVectors: source.latticeVectors } : {}),
    values,
    interpolation: 'nearest-index',
    codebookEntries: accumulators.length,
    storage: {
      current64Directions: {
        indexBytes: storedSamples * indexBytes,
        codebookBytes: accumulators.length * 16,
        totalBytes: storedSamples * indexBytes + accumulators.length * 16,
      },
      denser81DirectionsProjection: {
        indexBytes: storedSamplesForDimensions(source.width, source.height, 81) * indexBytes,
        // A fixed-size learned codebook could retain this bound. This observed
        // grid codebook is not guaranteed to stay fixed when new rows are baked.
        codebookBytesAssumingRetrainedFixedCardinality: accumulators.length * 16,
        totalBytesAssumingRetrainedFixedCardinality:
          storedSamplesForDimensions(source.width, source.height, 81) * indexBytes + accumulators.length * 16,
        uncertainty: 'new direction rows can introduce new occupied scalar-grid cells',
      },
      bytesPerIndex: indexBytes,
      bytesPerCodeword: 16,
      fixedReads: 2,
      filtering: 'nearest index only; bilinear index filtering is undefined and four-corner decode would require eight reads',
    },
  };
}

function storedSamplesForDimensions(width: number, height: number, directions: number): number {
  return (width + 2) * (height + 2) * directions;
}

function directStorage(
  width: number,
  height: number,
  bytesPerBlock: number,
  reads: number,
): Record<string, unknown> {
  const current = storedSamplesForDimensions(width, height, 64) * bytesPerBlock;
  const denser = storedSamplesForDimensions(width, height, 81) * bytesPerBlock;
  const baseline = 258 * 258 * 64 * 12;
  return {
    current64DirectionsBytes: current,
    denser81DirectionsBytes: denser,
    denser81RatioToCurrentGeometryPlusColorBaseline: denser / baseline,
    currentGeometryPlusColorBaselineBytes: baseline,
    bytesPerBlock,
    fixedFilterableReads: reads,
  };
}

function sampleField(
  field: CodecField,
  slice: number,
  x: number,
  y: number,
  output: Float32Array,
): void {
  let gx: number;
  let gy: number;
  if (field.latticeVectors) {
    const vector = slice * 4;
    const vx = field.latticeVectors[vector]!;
    const vy = field.latticeVectors[vector + 1]!;
    const wx = field.latticeVectors[vector + 2]!;
    const wy = field.latticeVectors[vector + 3]!;
    const determinant = 2 * (vx * wy - vy * wx);
    const px = x - vx * 0.5;
    const py = y - vy * 0.5;
    gx = (px * wy - wx * py) / determinant;
    gy = (-2 * vy * px + 2 * vx * py) / determinant;
  } else {
    const swap = field.sliceSwap[slice] !== 0;
    const localX = swap ? y : x;
    const localY = swap ? x : y;
    gx = (localX + 0.5) / field.blockSizeX - 0.5;
    gy = (localY + 0.5) / field.blockSizeY - 0.5;
  }
  if (field.interpolation === 'nearest-index') {
    const bx = mod(Math.round(gx), field.width);
    const by = mod(Math.round(gy), field.height);
    const source = ((slice * field.height + by) * field.width + bx) * COMPONENTS;
    output.set(field.values.subarray(source, source + COMPONENTS));
    return;
  }
  const x0 = Math.floor(gx);
  const y0 = Math.floor(gy);
  const fx = gx - x0;
  const fy = gy - y0;
  output.fill(0);
  for (let dy = 0; dy <= 1; dy++) {
    for (let dx = 0; dx <= 1; dx++) {
      const bx = mod(x0 + dx, field.width);
      const by = mod(y0 + dy, field.height);
      const weight = (dx === 0 ? 1 - fx : fx) * (dy === 0 ? 1 - fy : fy);
      const source = ((slice * field.height + by) * field.width + bx) * COMPONENTS;
      for (let component = 0; component < COMPONENTS; component++) {
        output[component] += weight * field.values[source + component]!;
      }
    }
  }
  if (field.interpolation === 'bilinear-conditional') {
    const coverage = clamp(output[C]!);
    const mean = clamp(output[M1]!);
    const sigma = clamp(output[M2]!);
    const second = clamp(mean * mean + sigma * sigma);
    output[C] = coverage;
    output[M1] = coverage * mean;
    output[M2] = coverage * second;
    output[CR] = coverage * clamp(output[CR]!);
    output[CG] = coverage * clamp(output[CG]!);
    output[CB] = coverage * clamp(output[CB]!);
    output[NX] *= coverage;
    output[NY] *= coverage;
    output[NZ] *= coverage;
  }
  sanitizeMeasure(output, 0);
}

function truthMeasure(
  profile: PeriodicProfileData,
  colorTexels: Uint8Array,
  slice: number,
  x: number,
  y: number,
  output: Float32Array,
): void {
  output.fill(0);
  const texel = profileTexelIndex(profile, slice, x, y);
  const source = texel * 4;
  const coverage = profile.texels[source + 3]! / 65535;
  if (coverage <= 0) return;
  const depth = profile.texels[source]! / 65535;
  const normal = decodeOctU16(profile.texels[source + 1]!, profile.texels[source + 2]!);
  output[C] = coverage;
  output[M1] = coverage * depth;
  output[M2] = coverage * depth * depth;
  output[CR] = colorTexels[source]! / 255;
  output[CG] = colorTexels[source + 1]! / 255;
  output[CB] = colorTexels[source + 2]! / 255;
  output[NX] = coverage * normal[0];
  output[NY] = coverage * normal[1];
  output[NZ] = coverage * normal[2];
}

function evaluateCodec(
  profile: PeriodicProfileData,
  lattice: ReturnType<typeof derivePeriodicProfileLattice>,
  colorTexels: Uint8Array,
  field: CodecField,
): Record<string, unknown> {
  const aggregate = emptyCodecAggregate();
  const perElevation = Array.from({ length: lattice.elevationCount }, emptyCodecAggregate);
  const perSlice = Array.from({ length: profile.slices.length }, emptyCodecAggregate);
  const phase = Array.from({ length: PHASE_BINS * PHASE_BINS }, emptyCodecAggregate);
  const rayErrors: number[] = [];
  const heightErrors: number[] = [];
  const normalErrors: number[] = [];
  const coherences: number[] = [];
  const uncertainties: number[] = [];
  const truth = new Float32Array(COMPONENTS);
  const decoded = new Float32Array(COMPONENTS);
  for (let slice = 0; slice < profile.slices.length; slice++) {
    const sliceInfo = profile.slices[slice]!;
    const depthSpan = sliceInfo.depthMax - sliceInfo.depthMin;
    const elevation = lattice.order === 'azimuth-major'
      ? slice % lattice.elevationCount
      : Math.floor(slice / lattice.azimuthCount);
    for (let y = 0; y < profile.interiorTileHeight; y++) {
      for (let x = 0; x < profile.interiorTileWidth; x++) {
        truthMeasure(profile, colorTexels, slice, x, y, truth);
        sampleField(field, slice, x, y, decoded);
        const blockX = Math.min(PHASE_BINS - 1, Math.floor(x * PHASE_BINS / profile.interiorTileWidth));
        const blockY = Math.min(PHASE_BINS - 1, Math.floor(y * PHASE_BINS / profile.interiorTileHeight));
        const targets = [aggregate, perElevation[elevation]!, perSlice[slice]!, phase[blockY * PHASE_BINS + blockX]!];
        for (const target of targets) {
          target.samples++;
          addScalar(target.coverage, decoded[C]! - truth[C]!);
          addScalar(target.premulDepth, decoded[M1]! - truth[M1]!);
          addScalar(target.premulSecondMoment, decoded[M2]! - truth[M2]!);
          for (const component of [CR, CG, CB]) {
            addScalar(target.premulColor, decoded[component]! - truth[component]!);
          }
          for (const component of [NX, NY, NZ]) {
            addScalar(target.normalMoment, decoded[component]! - truth[component]!);
          }
          const truthClass = truth[C]! >= 0.5;
          const decodedClass = decoded[C]! >= 0.5;
          if (truthClass || decodedClass) target.union++;
          if (truthClass && decodedClass) target.intersection++;
          if (truthClass) target.truthHits++;
        }
        if (truth[C]! > 0.5) {
          const decodedCoverage = decoded[C]!;
          if (decodedCoverage <= 0.02) {
            for (const target of targets) target.falseLowCoverage++;
          } else {
            const truthMean = truth[M1]! / truth[C]!;
            const decodedMean = decoded[M1]! / decodedCoverage;
            const rayError = Math.abs(decodedMean - truthMean) * depthSpan;
            const heightError = rayError * Math.abs(sliceInfo.direction[1]);
            const second = decoded[M2]! / decodedCoverage;
            const sigma = Math.sqrt(Math.max(0, second - decodedMean * decodedMean)) * depthSpan;
            const angle = normalAngleDegrees(
              truth[NX]!, truth[NY]!, truth[NZ]!,
              decoded[NX]!, decoded[NY]!, decoded[NZ]!,
            );
            const coherence = clamp(Math.hypot(decoded[NX]!, decoded[NY]!, decoded[NZ]!) / decodedCoverage);
            rayErrors.push(rayError);
            heightErrors.push(heightError);
            normalErrors.push(angle);
            coherences.push(coherence);
            uncertainties.push(sigma);
            for (const target of targets) {
              addScalar(target.rayDepth, rayError);
              addScalar(target.height, heightError);
              addScalar(target.normalAngle, angle);
              addScalar(target.orientationCoherence, coherence);
              addScalar(target.uncertaintyRay, sigma);
            }
          }
        }
      }
    }
  }
  const finish = (value: CodecAggregate): Record<string, unknown> => ({
    samples: value.samples,
    coverage: finishScalar(value.coverage),
    premultipliedColorChannels: finishScalar(value.premulColor),
    premultipliedDepth01: finishScalar(value.premulDepth),
    premultipliedSecondMoment01: finishScalar(value.premulSecondMoment),
    normalMomentComponents: finishScalar(value.normalMoment),
    representativeRayDepthMetres: finishScalar(value.rayDepth),
    representativeHeightMetres: finishScalar(value.height),
    normalAngleDegrees: finishScalar(value.normalAngle),
    orientationMomentCoherence: finishScalar(value.orientationCoherence),
    unresolvedRayDepthSigmaMetres: finishScalar(value.uncertaintyRay),
    falseCoverageBelow002OnTruthHit: value.falseLowCoverage,
    threshold05IntersectionOverUnion: value.union > 0 ? value.intersection / value.union : 1,
  });
  const worstSlices = perSlice.map((value, slice) => ({
    slice,
    azimuth: lattice.order === 'azimuth-major'
      ? Math.floor(slice / lattice.elevationCount)
      : slice % lattice.azimuthCount,
    elevation: lattice.order === 'azimuth-major'
      ? slice % lattice.elevationCount
      : Math.floor(slice / lattice.azimuthCount),
    coverageRms: value.coverage.count > 0 ? Math.sqrt(value.coverage.sumSq / value.coverage.count) : 0,
    premultipliedColorRms: value.premulColor.count > 0
      ? Math.sqrt(value.premulColor.sumSq / value.premulColor.count)
      : 0,
    representativeRayDepthRms: value.rayDepth.count > 0
      ? Math.sqrt(value.rayDepth.sumSq / value.rayDepth.count)
      : 0,
  })).sort((a, b) =>
    (b.coverageRms + b.premultipliedColorRms) - (a.coverageRms + a.premultipliedColorRms)).slice(0, 12);
  return {
    name: field.name,
    interpolation: field.interpolation,
    storage: field.storage,
    ...(field.codebookEntries === undefined ? {} : { codebookEntries: field.codebookEntries }),
    metrics: finish(aggregate),
    distributionsOnTruthHitsWithDecodedCoverageAbove002: {
      representativeRayDepthErrorMetres: quantiles(rayErrors),
      representativeHeightErrorMetres: quantiles(heightErrors),
      normalAngleErrorDegrees: quantiles(normalErrors),
      orientationMomentCoherence: quantiles(coherences),
      unresolvedRayDepthSigmaMetres: quantiles(uncertainties),
      representativeRayDepthErrorFractionsAtOrBelowMetres: fractionsAtOrBelow(
        rayErrors,
        [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1],
      ),
      unresolvedRayDepthSigmaFractionsAtOrBelowMetres: fractionsAtOrBelow(
        uncertainties,
        [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1],
      ),
    },
    byElevation: perElevation.map((value, elevation) => ({
      elevationDegrees: lattice.elevations[elevation]! * 180 / Math.PI,
      ...finish(value),
    })),
    worstSlices,
    phaseHeatmap16x16: phase.map((value, cell) => ({
      x: cell % PHASE_BINS,
      y: Math.floor(cell / PHASE_BINS),
      coverageMeanAbs: value.coverage.count > 0 ? value.coverage.sumAbs / value.coverage.count : 0,
      premultipliedColorRms: value.premulColor.count > 0
        ? Math.sqrt(value.premulColor.sumSq / value.premulColor.count)
        : 0,
      representativeRayDepthRms: value.rayDepth.count > 0
        ? Math.sqrt(value.rayDepth.sumSq / value.rayDepth.count)
        : 0,
      unresolvedRayDepthSigmaMean: value.uncertaintyRay.count > 0
        ? value.uncertaintyRay.sumAbs / value.uncertaintyRay.count
        : 0,
    })),
  };
}

function linearToSrgb(value: number): number {
  const clamped = clamp(value);
  return clamped <= 0.0031308
    ? clamped * 12.92
    : 1.055 * clamped ** (1 / 2.4) - 0.055;
}

function heat(value: number): readonly [number, number, number] {
  const t = clamp(value);
  return [
    clamp(1.5 * t),
    clamp(1.5 - Math.abs(2 * t - 1.2)),
    clamp(1.2 - 1.8 * t),
  ];
}

function renderSlicePanels(
  profile: PeriodicProfileData,
  lattice: ReturnType<typeof derivePeriodicProfileLattice>,
  colorTexels: Uint8Array,
  slice: number,
  fields: readonly CodecField[],
): { panels: Uint8Array[]; labels: string[] } {
  const truth = new Float32Array(COMPONENTS);
  const sample = new Float32Array(COMPONENTS);
  const sliceInfo = profile.slices[slice]!;
  const depthSpan = sliceInfo.depthMax - sliceInfo.depthMin;
  const renderColor = (field: CodecField | null): Uint8Array => {
    const image = new Uint8Array(PANEL * PANEL * 3);
    for (let y = 0; y < PANEL; y++) {
      for (let x = 0; x < PANEL; x++) {
        if (field) sampleField(field, slice, x, y, sample);
        else truthMeasure(profile, colorTexels, slice, x, y, sample);
        const coverage = clamp(sample[C]!);
        const background = [0.29, 0.265, 0.22] as const;
        const output = (y * PANEL + x) * 3;
        for (let channel = 0; channel < 3; channel++) {
          const premul = sample[CR + channel]!;
          const linear = premul + (1 - coverage) * background[channel]!;
          image[output + channel] = Math.round(linearToSrgb(linear) * 255);
        }
      }
    }
    return image;
  };
  const renderTruthHeight = (): Uint8Array => {
    const image = new Uint8Array(PANEL * PANEL * 3);
    const bounds = profile.sourceBounds!;
    const spanY = bounds[4] - bounds[1];
    for (let y = 0; y < PANEL; y++) {
      for (let x = 0; x < PANEL; x++) {
        truthMeasure(profile, colorTexels, slice, x, y, truth);
        const output = (y * PANEL + x) * 3;
        if (truth[C]! <= 0) continue;
        const mean = truth[M1]! / truth[C]!;
        const t = sliceInfo.depthMin + mean * depthSpan;
        const height = profile.topH + sliceInfo.direction[1] * t;
        const value = clamp((height - bounds[1]) / spanY);
        const color = heat(value);
        image[output] = Math.round(color[0] * 255);
        image[output + 1] = Math.round(color[1] * 255);
        image[output + 2] = Math.round(color[2] * 255);
      }
    }
    return image;
  };
  const renderUncertainty = (field: CodecField): Uint8Array => {
    const sigmas: number[] = [];
    const sigmaAt = new Float32Array(PANEL * PANEL);
    for (let y = 0; y < PANEL; y++) {
      for (let x = 0; x < PANEL; x++) {
        sampleField(field, slice, x, y, sample);
        if (sample[C]! <= 0.02) continue;
        const mean = sample[M1]! / sample[C]!;
        const second = sample[M2]! / sample[C]!;
        const sigma = Math.sqrt(Math.max(0, second - mean * mean)) * depthSpan;
        sigmaAt[y * PANEL + x] = sigma;
        sigmas.push(sigma);
      }
    }
    sigmas.sort((a, b) => a - b);
    const scale = Math.max(0.001, sigmas[Math.floor(Math.max(0, sigmas.length - 1) * 0.99)] ?? 0.001);
    const image = new Uint8Array(PANEL * PANEL * 3);
    for (let index = 0; index < sigmaAt.length; index++) {
      const color = heat(sigmaAt[index]! / scale);
      image[index * 3] = Math.round(color[0] * 255);
      image[index * 3 + 1] = Math.round(color[1] * 255);
      image[index * 3 + 2] = Math.round(color[2] * 255);
    }
    return image;
  };
  const renderDepthError = (field: CodecField): Uint8Array => {
    const errors: number[] = [];
    const errorAt = new Float32Array(PANEL * PANEL);
    for (let y = 0; y < PANEL; y++) {
      for (let x = 0; x < PANEL; x++) {
        truthMeasure(profile, colorTexels, slice, x, y, truth);
        sampleField(field, slice, x, y, sample);
        if (truth[C]! <= 0.5 || sample[C]! <= 0.02) continue;
        const error = Math.abs(sample[M1]! / sample[C]! - truth[M1]! / truth[C]!) * depthSpan;
        errorAt[y * PANEL + x] = error;
        errors.push(error);
      }
    }
    errors.sort((a, b) => a - b);
    const scale = Math.max(0.001, errors[Math.floor(Math.max(0, errors.length - 1) * 0.99)] ?? 0.001);
    const image = new Uint8Array(PANEL * PANEL * 3);
    for (let index = 0; index < errorAt.length; index++) {
      const color = heat(errorAt[index]! / scale);
      image[index * 3] = Math.round(color[0] * 255);
      image[index * 3 + 1] = Math.round(color[1] * 255);
      image[index * 3 + 2] = Math.round(color[2] * 255);
    }
    return image;
  };
  const labels = [
    'truth authored colour',
    'index-2 ray lattice colour',
    'square 2x2 colour',
    'VQ16 nearest colour',
    'truth hit height',
    'ray lattice unresolved sigma',
    'square 2x2 ray-depth error',
    'VQ16 nearest ray-depth error',
  ];
  const panels = [
    renderColor(null),
    ...fields.map(renderColor),
    renderTruthHeight(),
    renderUncertainty(fields[0]!),
    ...fields.slice(1).map(renderDepthError),
  ];
  if (panels.length !== labels.length || panels.length % 2 !== 0) {
    throw new Error('QA panel layout must contain two equal rows');
  }
  void lattice;
  return { panels, labels };
}

function xmlEscape(value: string): string {
  return value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}

async function writeContactSheet(
  path: string,
  title: string,
  panels: readonly Uint8Array[],
  labels: readonly string[],
): Promise<{ width: number; height: number; sha256: string }> {
  const columns = panels.length / 2;
  const width = columns * PANEL;
  const titleHeight = 34;
  const height = titleHeight + 2 * (LABEL + PANEL);
  const raw = new Uint8Array(width * height * 3);
  raw.fill(24);
  for (let panel = 0; panel < panels.length; panel++) {
    const column = panel % columns;
    const row = Math.floor(panel / columns);
    const top = titleHeight + row * (LABEL + PANEL) + LABEL;
    const left = column * PANEL;
    const source = panels[panel]!;
    for (let y = 0; y < PANEL; y++) {
      const target = ((top + y) * width + left) * 3;
      raw.set(source.subarray(y * PANEL * 3, (y + 1) * PANEL * 3), target);
    }
  }
  const text = [
    `<text x="12" y="23" fill="#ffffff" font-family="sans-serif" font-size="17">${xmlEscape(title)}</text>`,
    ...labels.map((label, panel) => {
      const column = panel % columns;
      const row = Math.floor(panel / columns);
      const x = column * PANEL + 8;
      const y = titleHeight + row * (LABEL + PANEL) + 19;
      return `<text x="${x}" y="${y}" fill="#ffffff" font-family="sans-serif" font-size="13">${xmlEscape(label)}</text>`;
    }),
  ].join('');
  const svg = Buffer.from(`<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">${text}</svg>`);
  await sharp(raw, { raw: { width, height, channels: 3 } })
    .composite([{ input: svg, top: 0, left: 0 }])
    .png({ compressionLevel: 9, adaptiveFiltering: false })
    .toFile(path);
  const bytes = readFileSync(path);
  return { width, height, sha256: sha256(bytes) };
}

async function main(): Promise<void> {
  const sourcePath = resolve(process.argv[2] ?? 'src/assets/groundcover/calamagrostis-canescens.gcrp');
  const sourceBytes = readFileSync(sourcePath);
  const sourceHash = sha256(sourceBytes);
  if (sourceHash !== EXPECTED_SOURCE_SHA256) {
    throw new Error(`event-measure audit expected ${EXPECTED_SOURCE_SHA256}, got ${sourceHash}`);
  }
  const profile = parsePeriodicProfile(sourceBytes);
  if (profile.version !== 4 || !profile.ownerTexels || !profile.vertexRecords || !profile.triangleRecords) {
    throw new Error('event-measure audit requires the complete GCRP/v4 source mesh and owner field');
  }
  let ownershipMismatch = 0;
  for (let texel = 0; texel < profile.ownerTexels.length; texel++) {
    const hit = profile.texels[texel * 4 + 3]! !== 0;
    if (hit !== (profile.ownerTexels[texel] !== MISS_OWNER)) ownershipMismatch++;
  }
  if (ownershipMismatch !== 0) throw new Error(`source owner/coverage mismatch: ${ownershipMismatch}`);
  const lattice = derivePeriodicProfileLattice(profile);
  const colorTexture = makePeriodicProfileColorTexture(profile);
  if (!colorTexture) throw new Error('accepted GCRP/v4 did not produce authored colour');
  const colorTexels = colorTexture.image.data as Uint8Array;
  const scriptBytes = readFileSync(fileURLToPath(import.meta.url));
  const scriptHash = sha256(scriptBytes);
  const recipeHash = sha256(canonicalJson({ recipe: RECIPE, scriptHash }));
  const artifactHash = sha256(`${sourceHash}:${recipeHash}`);
  const root = resolve(`data/work/groundcover-event-measure-codec/${artifactHash}`);
  const qa = resolve(root, 'qa');
  mkdirSync(qa, { recursive: true });

  console.log('building anisotropic, 2x2, and 4x4 event measures');
  const blocksLattice = buildRayLatticeBlockField(profile, colorTexels);
  const blocksAlong = buildBlockField(profile, colorTexels, 2, 1, 'ray-aligned');
  const blocksCross = buildBlockField(profile, colorTexels, 2, 1, 'cross-ray');
  const blocks2 = buildBlockField(profile, colorTexels, 2, 2);
  const blocks4 = buildBlockField(profile, colorTexels, 4, 4);
  const directLattice = directMeasureCodec(blocksLattice);
  const directAlong = directMeasureCodec(blocksAlong);
  const directCross = directMeasureCodec(blocksCross);
  const direct2 = directMeasureCodec(blocks2);
  const direct4 = directMeasureCodec(blocks4);
  const conditional2 = conditionalCodec(blocks2);
  const vqFields = RECIPE.vq.map((config) => {
    console.log(`building ${config.name}`);
    return vqCodec(blocksLattice, config);
  });
  const admissibleVq = vqFields.filter((field) => (field.codebookEntries ?? Infinity) <= 65535);
  const strongestVq = admissibleVq.at(-1) ?? vqFields[0]!;
  const codecs = [directLattice, directAlong, directCross, direct2, direct4, conditional2, ...vqFields];
  const reports: Record<string, unknown>[] = [];
  for (const codec of codecs) {
    console.log(`evaluating ${codec.name}`);
    reports.push(evaluateCodec(profile, lattice, colorTexels, codec));
  }

  const qaFields = [directLattice, direct2, strongestVq];
  const imageIndex: Record<string, unknown>[] = [];
  for (const selection of RECIPE.qaSlices) {
    const slice = lattice.order === 'azimuth-major'
      ? selection.azimuth * lattice.elevationCount + selection.elevation
      : selection.elevation * lattice.azimuthCount + selection.azimuth;
    const rendered = renderSlicePanels(profile, lattice, colorTexels, slice, qaFields);
    const filename = `${String(selection.number).padStart(3, '0')}-${selection.label
      .replaceAll(/[^a-zA-Z0-9]+/g, '-').replaceAll(/^-|-$/g, '').toLowerCase()}.png`;
    const path = resolve(qa, filename);
    const metadata = await writeContactSheet(path, selection.label, rendered.panels, rendered.labels);
    imageIndex.push({
      number: selection.number,
      file: `qa/${filename}`,
      slice,
      azimuth: selection.azimuth,
      elevation: selection.elevation,
      elevationDegrees: lattice.elevations[selection.elevation]! * 180 / Math.PI,
      interpretation: 'top row compares premultiplied authored-colour composites; bottom row exposes true hit height, direct-measure unresolved depth sigma, and representative-depth errors',
      ...metadata,
    });
  }

  const metrics = {
    schema: RECIPE.schema,
    source: {
      path: sourcePath,
      bytes: sourceBytes.byteLength,
      sha256: sourceHash,
    },
    recipe: RECIPE,
    recipeHash,
    scriptHash,
    artifactHash,
    field: {
      interior: [profile.interiorTileWidth, profile.interiorTileHeight],
      stored: [profile.storedTileWidth, profile.storedTileHeight],
      directions: profile.slices.length,
      azimuths: lattice.azimuthCount,
      elevationsDegrees: lattice.elevations.map((value) => value * 180 / Math.PI),
      tileMetres: [profile.tileSizeX, profile.tileSizeZ],
      topH: profile.topH,
      sourceBounds: profile.sourceBounds,
      ownershipMismatch,
    },
    contract: {
      eventMeasure: 'C, C E[u], C E[u^2], C E[RGB], C E[n]',
      representativeSingleEvent: 'uHat=(C E[u])/C only when C>0; this is a moment-matched visual representative, not an exact owner hit',
      unresolvedUncertainty: 'sigmaT=(sliceDepthSpan)*sqrt(E[u^2]-E[u]^2)',
      positionRecovery: 'for one slice, mean point equals mean ray origin plus direction*(depthMin+uHat*depthSpan); no independent xyz payload is needed',
      unrelatedEvents: 'may contribute only through the linear event moments; no centroid is interpreted as an exact source owner',
    },
    fundingGate: {
      currentGeometryPlusColorBytes: 258 * 258 * 64 * 12,
      correctedDirectionCount: 81,
      maximumPerDirectionRatioToKeepCurrentResidentBytes: 64 / 81,
      note: 'storage headroom can fund additional baked directions but does not itself reconstruct the omitted angular dimension',
    },
    selectedVqForQa: strongestVq.name,
    reports,
  };
  const metricsPath = resolve(root, 'metrics.json');
  writeFileSync(metricsPath, `${JSON.stringify(metrics, null, 2)}\n`);
  const metricsHash = sha256(readFileSync(metricsPath));
  const index = {
    schema: 'laas-groundcover-event-measure-codec-index/v1',
    artifactHash,
    sourceSha256: sourceHash,
    recipeHash,
    scriptHash,
    metrics: {
      file: 'metrics.json',
      sha256: metricsHash,
    },
    images: imageIndex,
  };
  const indexPath = resolve(root, 'index.json');
  writeFileSync(indexPath, `${JSON.stringify(index, null, 2)}\n`);
  colorTexture.dispose();
  console.log(JSON.stringify({ root, index: indexPath, metrics: metricsPath, artifactHash, reports: reports.map((report) => {
    const value = report as Record<string, unknown>;
    const metric = value.metrics as Record<string, Record<string, number>>;
    const distribution = value.distributionsOnTruthHitsWithDecodedCoverageAbove002 as Record<string, Record<string, number>>;
    return {
      name: value.name,
      entries: value.codebookEntries,
      coverageRms: metric.coverage.rms,
      colorRms: metric.premultipliedColorChannels.rms,
      rayDepthP95: distribution.representativeRayDepthErrorMetres.p95,
      uncertaintyP95: distribution.unresolvedRayDepthSigmaMetres.p95,
    };
  }) }, null, 2));
}

await main();
