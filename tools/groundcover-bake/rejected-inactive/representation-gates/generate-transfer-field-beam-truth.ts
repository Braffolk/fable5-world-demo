/**
 * Offline actual-mesh beam truth for the line-invariant transfer-field gate.
 * Each output is an explicitly traced 4D phase/direction footprint containing
 * coverage and three equal-mass conditional first-interaction strata.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  TriangleBvh,
  decodeOwnedProfileGeometry,
  type CensusVec3,
  type DecodedOwnedProfileGeometry,
} from '../../ExteriorClosureCensus';
import { periodicNearestSuccessor, type OriginAwareTruthHit } from '../../OriginAwareRayTruth';

const EXPECTED_SOURCE_SHA256 =
  '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const TRAIN_ELEVATIONS = [0, 0.1, 1, 5, 15, 35, 75, 90] as const;
const VALIDATION_ELEVATIONS = [0.05, 0.5, 2, 10, 25, 55, 82.5] as const;
const TRAIN_HEIGHTS = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] as const;
const VALIDATION_HEIGHTS = [0.01, 0.055, 0.1, 0.175, 0.25, 0.375, 0.5, 0.625, 0.75, 0.825, 0.9, 0.945, 0.99] as const;
const STRATA = 3;
const OUTPUT_FIELDS = 6 + 1 + STRATA * 8;
const GOLDEN_RATIO = 0.6180339887498949;

type Sense = 'horizontal' | 'downward' | 'upward';
interface Args { [key: string]: string | boolean }
interface DirectionClass { elevationDegrees: number; sense: Sense }
interface SurfaceEvent {
  t: number;
  color: readonly [number, number, number];
  normal: readonly [number, number, number];
}

function parseArgs(argv: readonly string[]): Args {
  const result: Args = {};
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index]!;
    if (!argument.startsWith('--')) continue;
    const value = argv[index + 1];
    if (value !== undefined && !value.startsWith('--')) {
      result[argument.slice(2)] = value;
      index++;
    } else result[argument.slice(2)] = true;
  }
  return result;
}

function numberArg(value: string | boolean | undefined, fallback: number, label: string): number {
  const parsed = Number(typeof value === 'string' ? value : fallback);
  if (!Number.isFinite(parsed)) throw new Error(`${label} must be finite`);
  return parsed;
}

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function hashUnit(seed: number, index: number): number {
  let value = (index ^ seed) >>> 0;
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d);
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b);
  return ((value ^ (value >>> 16)) >>> 0) / 0x1_0000_0000;
}

function wrap01(value: number): number {
  return value - Math.floor(value);
}

function directionClasses(elevations: readonly number[]): DirectionClass[] {
  const result: DirectionClass[] = [];
  for (const elevationDegrees of elevations) {
    if (elevationDegrees === 0) result.push({ elevationDegrees, sense: 'horizontal' });
    else {
      result.push({ elevationDegrees, sense: 'downward' });
      result.push({ elevationDegrees, sense: 'upward' });
    }
  }
  return result;
}

function direction(definition: DirectionClass, azimuth: number): CensusVec3 {
  const elevation = definition.elevationDegrees * Math.PI / 180;
  const horizontal = definition.elevationDegrees === 90 ? 0 : Math.cos(elevation);
  const y = definition.sense === 'downward' ? -Math.sin(elevation)
    : definition.sense === 'upward' ? Math.sin(elevation) : 0;
  return [horizontal * Math.cos(azimuth), y, horizontal * Math.sin(azimuth)];
}

function cross(a: CensusVec3, b: CensusVec3): CensusVec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function normalize(value: CensusVec3): CensusVec3 {
  const length = Math.hypot(...value);
  return [value[0] / length, value[1] / length, value[2] / length];
}

function tangentFrame(d: CensusVec3): [CensusVec3, CensusVec3] {
  const helper: CensusVec3 = Math.abs(d[1]) < 0.9 ? [0, 1, 0] : [0, 0, 1];
  const u = normalize(cross(helper, d));
  return [u, normalize(cross(d, u))];
}

function decodeOct(x: number, y: number): [number, number, number] {
  let nx = x * 2 - 1;
  let ny = y * 2 - 1;
  let nz = 1 - Math.abs(nx) - Math.abs(ny);
  if (nz < 0) {
    const oldX = nx;
    nx = (1 - Math.abs(ny)) * (oldX >= 0 ? 1 : -1);
    ny = (1 - Math.abs(oldX)) * (ny >= 0 ? 1 : -1);
  }
  return normalize([nx, ny, nz]) as [number, number, number];
}

function vertexAttributes(source: Uint8Array, vertex: number): { color: CensusVec3; normal: CensusVec3 } {
  const view = new DataView(source.buffer, source.byteOffset, source.byteLength);
  const vertexOffset = view.getUint32(84, true);
  const record = vertexOffset + vertex * 16;
  return {
    color: [
      view.getUint16(record + 6, true) / 65535,
      view.getUint16(record + 8, true) / 65535,
      view.getUint16(record + 10, true) / 65535,
    ],
    normal: decodeOct(
      view.getUint16(record + 12, true) / 65535,
      view.getUint16(record + 14, true) / 65535,
    ),
  };
}

function eventAttributes(
  source: Uint8Array,
  geometry: DecodedOwnedProfileGeometry,
  origin: CensusVec3,
  d: CensusVec3,
  hit: OriginAwareTruthHit,
): SurfaceEvent {
  const point: CensusVec3 = [
    origin[0] + d[0] * hit.t - hit.copyX * geometry.tileSizeX,
    origin[1] + d[1] * hit.t,
    origin[2] + d[2] * hit.t - hit.copyZ * geometry.tileSizeZ,
  ];
  const triangle = hit.triangleId * 3;
  const vertexIndices = [
    geometry.triangles[triangle]!,
    geometry.triangles[triangle + 1]!,
    geometry.triangles[triangle + 2]!,
  ] as const;
  const positions = vertexIndices.map((vertex) => {
    const offset = vertex * 3;
    return [geometry.positions[offset]!, geometry.positions[offset + 1]!, geometry.positions[offset + 2]!] as CensusVec3;
  });
  const v0: CensusVec3 = [positions[1]![0] - positions[0]![0], positions[1]![1] - positions[0]![1], positions[1]![2] - positions[0]![2]];
  const v1: CensusVec3 = [positions[2]![0] - positions[0]![0], positions[2]![1] - positions[0]![1], positions[2]![2] - positions[0]![2]];
  const v2: CensusVec3 = [point[0] - positions[0]![0], point[1] - positions[0]![1], point[2] - positions[0]![2]];
  const d00 = v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2];
  const d01 = v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
  const d11 = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
  const d20 = v2[0] * v0[0] + v2[1] * v0[1] + v2[2] * v0[2];
  const d21 = v2[0] * v1[0] + v2[1] * v1[1] + v2[2] * v1[2];
  const denominator = Math.max(1e-30, d00 * d11 - d01 * d01);
  let wb = (d11 * d20 - d01 * d21) / denominator;
  let wc = (d00 * d21 - d01 * d20) / denominator;
  let wa = 1 - wb - wc;
  wa = Math.max(0, wa);
  wb = Math.max(0, wb);
  wc = Math.max(0, wc);
  const sum = Math.max(1e-30, wa + wb + wc);
  wa /= sum;
  wb /= sum;
  wc /= sum;
  const weights = [wa, wb, wc] as const;
  const attributes = vertexIndices.map((vertex) => vertexAttributes(source, vertex));
  const color: [number, number, number] = [0, 0, 0];
  const normal: [number, number, number] = [0, 0, 0];
  for (let corner = 0; corner < 3; corner++) {
    for (let component = 0; component < 3; component++) {
      color[component] += attributes[corner]!.color[component]! * weights[corner]!;
      normal[component] += attributes[corner]!.normal[component]! * weights[corner]!;
    }
  }
  return { t: hit.t, color, normal: normalize(normal) };
}

function microRay(
  phaseX: number,
  height: number,
  phaseZ: number,
  d: CensusVec3,
  microIndex: number,
  beamSamples: number,
  phaseRadiusNormalized: number,
  angularRadiusRadians: number,
  rotation: number,
): { phaseX: number; height: number; phaseZ: number; direction: CensusVec3 } {
  if (microIndex === 0) return { phaseX, height, phaseZ, direction: d };
  const sequence = microIndex + rotation;
  const phaseAngle = Math.PI * 2 * wrap01(sequence * GOLDEN_RATIO);
  const phaseRadius = phaseRadiusNormalized * Math.sqrt((microIndex - 0.5) / beamSamples);
  const angularAngle = Math.PI * 2 * wrap01(sequence * 0.7548776662466927);
  const angularRadius = angularRadiusRadians * Math.sqrt(wrap01(sequence * 0.5698402909980532));
  const [u, v] = tangentFrame(d);
  return {
    phaseX: wrap01(phaseX + Math.cos(phaseAngle) * phaseRadius),
    height,
    phaseZ: wrap01(phaseZ + Math.sin(phaseAngle) * phaseRadius),
    direction: normalize([
      d[0] + angularRadius * (Math.cos(angularAngle) * u[0] + Math.sin(angularAngle) * v[0]),
      d[1] + angularRadius * (Math.cos(angularAngle) * u[1] + Math.sin(angularAngle) * v[1]),
      d[2] + angularRadius * (Math.cos(angularAngle) * u[2] + Math.sin(angularAngle) * v[2]),
    ]),
  };
}

function encodeBeam(target: Float32Array, offset: number, events: SurfaceEvent[], beamSamples: number): void {
  target[offset + 6] = events.length / beamSamples;
  if (events.length === 0) return;
  events.sort((a, b) => a.t - b.t);
  for (let stratum = 0; stratum < STRATA; stratum++) {
    const members = events.filter((_, index) => Math.min(STRATA - 1, Math.floor(index * STRATA / events.length)) === stratum);
    const selected = members.length > 0
      ? members
      : [events[Math.min(events.length - 1, Math.floor((stratum + 0.5) / STRATA * events.length))]!];
    const mean = selected.reduce((sum, event) => sum + event.t, 0) / selected.length;
    const variance = selected.reduce((sum, event) => sum + (event.t - mean) ** 2, 0) / selected.length;
    const color: [number, number, number] = [0, 0, 0];
    const normal: [number, number, number] = [0, 0, 0];
    for (const event of selected) for (let component = 0; component < 3; component++) {
      color[component] += event.color[component]! / selected.length;
      normal[component] += event.normal[component]! / selected.length;
    }
    const normalizedNormal = normalize(normal);
    const base = offset + 7 + stratum * 8;
    target[base] = mean;
    target[base + 1] = variance;
    target.set(color, base + 2);
    target.set(normalizedNormal, base + 5);
  }
}

function expectedCenters(
  elevations: readonly number[],
  heights: readonly number[],
  azimuthBins: number,
  centersPerStratum: number,
): number {
  return directionClasses(elevations).reduce(
    (sum, entry) => sum + heights.length * (entry.elevationDegrees === 90 ? 1 : azimuthBins) * centersPerStratum,
    0,
  );
}

const cli = parseArgs(process.argv.slice(2));
const sourcePath = resolve(typeof cli.source === 'string' ? cli.source : 'src/assets/groundcover/calamagrostis-canescens.gcrp');
const sourceBytes = readFileSync(sourcePath);
const sourceHash = sha256(sourceBytes);
if (sourceHash !== EXPECTED_SOURCE_SHA256) throw new Error(`unexpected source hash ${sourceHash}`);
const horizon = numberArg(cli.horizon, 155, 'horizon');
const originEpsilon = numberArg(cli['origin-epsilon'], 1e-9, 'origin epsilon');
const beamSamples = numberArg(cli['beam-samples'], 16, 'beam samples');
const phaseResolution = numberArg(cli['phase-resolution'], 192, 'phase resolution');
const angularRadiusDegrees = numberArg(cli['angular-radius-degrees'], 0.05, 'angular radius degrees');
const coherencePairs = numberArg(cli['coherence-pairs'], 512, 'coherence pairs');
const coherenceShiftMetres = numberArg(cli['coherence-shift-metres'], 0.02, 'coherence shift metres');
const trainAzimuthBins = numberArg(cli['train-azimuth-bins'], 8, 'train azimuth bins');
const validationAzimuthBins = numberArg(cli['validation-azimuth-bins'], 16, 'validation azimuth bins');
const trainCentersPerStratum = numberArg(cli['train-per-stratum'], 32, 'train centers per stratum');
const validationCentersPerStratum = numberArg(cli['validation-per-stratum'], 4, 'validation centers per stratum');
const qaGrid = numberArg(cli['qa-grid'], 32, 'QA grid');
for (const [label, value] of Object.entries({ beamSamples, phaseResolution, trainAzimuthBins, validationAzimuthBins, trainCentersPerStratum, validationCentersPerStratum, qaGrid, coherencePairs })) {
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${label} must be a positive integer`);
}
if (beamSamples < STRATA) throw new Error(`beam samples must be at least ${STRATA}`);
if (!(coherenceShiftMetres > 0)) throw new Error('coherence shift must be positive');
const phaseRadiusNormalized = 0.5 / phaseResolution;
const angularRadiusRadians = angularRadiusDegrees * Math.PI / 180;
const toolPath = fileURLToPath(import.meta.url);
const configuration = {
  schema: 'laas-groundcover-transfer-beam-truth/v1',
  horizon,
  originEpsilon,
  trainElevationsDegrees: TRAIN_ELEVATIONS,
  validationElevationsDegrees: VALIDATION_ELEVATIONS,
  trainHeights: TRAIN_HEIGHTS,
  validationHeights: VALIDATION_HEIGHTS,
  trainAzimuthBins,
  validationAzimuthBins,
  trainCentersPerStratum,
  validationCentersPerStratum,
  beam: {
    samples: beamSamples,
    includesCentralRay: true,
    phaseKernel: `deterministic disk, radius half of one ${phaseResolution}-sample tile texel`,
    phaseRadiusNormalized,
    angularKernel: 'deterministic tangent-disk',
    angularRadiusDegrees,
  },
  strata: { count: STRATA, partition: 'equal conditional hit mass, ordered front to back' },
  coherence: {
    pairs: coherencePairs,
    shiftMetres: coherenceShiftMetres,
    contract: 'second physical origin is shifted forward on the same held-out central oriented line; paired records are consecutive',
  },
  qaGrid,
  toolSha256: sha256(readFileSync(toolPath)),
};
const recipeHash = sha256(Buffer.from(JSON.stringify(configuration)));
const outputDirectory = resolve(typeof cli.output === 'string'
  ? cli.output
  : `data/work/groundcover-transfer-field/${sourceHash}/${recipeHash}/truth`);
mkdirSync(outputDirectory, { recursive: true });
console.error(`[transfer-beam] decoding ${(sourceBytes.byteLength / 1024 / 1024).toFixed(1)} MiB actual mesh`);
const geometry = decodeOwnedProfileGeometry(sourceBytes);
const bvh = TriangleBvh.build(geometry, 8);

let maximumExteriorEntryInvarianceError = 0;
for (let sample = 0; sample < 1024; sample++) {
  const azimuth = hashUnit(0xab735a19, sample * 3) * Math.PI * 2;
  const elevation = (5 + hashUnit(0x51ed270b, sample * 3 + 1) * 84) * Math.PI / 180;
  const d: CensusVec3 = [Math.cos(elevation) * Math.cos(azimuth), -Math.sin(elevation), Math.cos(elevation) * Math.sin(azimuth)];
  const camera: CensusVec3 = [
    geometry.tileOriginX + hashUnit(0x2c1b3c6d, sample * 3 + 2) * geometry.tileSizeX,
    geometry.topH + 0.25,
    geometry.tileOriginZ + hashUnit(0x297a2d39, sample * 3 + 3) * geometry.tileSizeZ,
  ];
  const lambda = -2 + hashUnit(0x7f4a7c15, sample * 3 + 4) * 4;
  const datum: CensusVec3 = [camera[0] + lambda * d[0], camera[1] + lambda * d[1], camera[2] + lambda * d[2]];
  const entry = (origin: CensusVec3): CensusVec3 => {
    const t = (geometry.topH - origin[1]) / d[1];
    return [origin[0] + t * d[0], geometry.topH, origin[2] + t * d[2]];
  };
  const first = entry(camera);
  const second = entry(datum);
  maximumExteriorEntryInvarianceError = Math.max(
    maximumExteriorEntryInvarianceError,
    Math.hypot(first[0] - second[0], first[1] - second[1], first[2] - second[2]),
  );
}

function traceBeam(
  phaseX: number,
  height: number,
  phaseZ: number,
  d: CensusVec3,
  rotation: number,
): SurfaceEvent[] {
  const events: SurfaceEvent[] = [];
  for (let micro = 0; micro < beamSamples; micro++) {
    const ray = microRay(phaseX, height, phaseZ, d, micro, beamSamples, phaseRadiusNormalized, angularRadiusRadians, rotation);
    const origin: CensusVec3 = [
      geometry.tileOriginX + ray.phaseX * geometry.tileSizeX,
      geometry.bounds.min[1] + ray.height * (geometry.bounds.max[1] - geometry.bounds.min[1]),
      geometry.tileOriginZ + ray.phaseZ * geometry.tileSizeZ,
    ];
    const hit = periodicNearestSuccessor(geometry, bvh, origin, ray.direction, horizon, originEpsilon);
    if (hit) events.push(eventAttributes(sourceBytes, geometry, origin, ray.direction, hit));
  }
  return events;
}

function generateSplit(
  name: 'train' | 'validation',
  elevations: readonly number[],
  heights: readonly number[],
  azimuthBins: number,
  centersPerStratum: number,
  seed: number,
): { path: string; records: number; microRays: number; sha256: string; coverageMean: number } {
  const count = expectedCenters(elevations, heights, azimuthBins, centersPerStratum);
  const records = new Float32Array(count * OUTPUT_FIELDS);
  let record = 0;
  let coverageSum = 0;
  const classes = directionClasses(elevations);
  console.error(`[transfer-beam] ${name}: ${count.toLocaleString()} beams / ${(count * beamSamples).toLocaleString()} micro-rays`);
  for (let classIndex = 0; classIndex < classes.length; classIndex++) {
    const definition = classes[classIndex]!;
    const bins = definition.elevationDegrees === 90 ? 1 : azimuthBins;
    for (let heightIndex = 0; heightIndex < heights.length; heightIndex++) {
      for (let azimuthBin = 0; azimuthBin < bins; azimuthBin++) {
        for (let repetition = 0; repetition < centersPerStratum; repetition++) {
          const key = record + classIndex * 0x100000 + heightIndex * 0x10000 + azimuthBin * 0x100;
          const phaseX = hashUnit(seed ^ 0x9e3779b9, key * 5 + 0);
          const phaseZ = hashUnit(seed ^ 0x85ebca6b, key * 5 + 1);
          const azimuthJitter = hashUnit(seed ^ 0xc2b2ae35, key * 5 + 2);
          const azimuth = definition.elevationDegrees === 90 ? 0 : (azimuthBin + azimuthJitter) / bins * Math.PI * 2;
          const height = name === 'validation' || repetition % 4 === 0
            ? heights[heightIndex]!
            : 0.001 + hashUnit(seed ^ 0x27d4eb2f, key * 5 + 3) * 0.998;
          const d = direction(definition, azimuth);
          const offset = record * OUTPUT_FIELDS;
          records.set([phaseX, height, phaseZ, ...d], offset);
          const events = traceBeam(phaseX, height, phaseZ, d, hashUnit(seed ^ 0x165667b1, key * 5 + 4));
          encodeBeam(records, offset, events, beamSamples);
          coverageSum += events.length / beamSamples;
          record++;
          if (record % 2_000 === 0) console.error(`[transfer-beam] ${name}: ${record.toLocaleString()}/${count.toLocaleString()}`);
        }
      }
    }
  }
  if (record !== count) throw new Error(`${name} record count mismatch ${record}/${count}`);
  const bytes = Buffer.from(records.buffer, records.byteOffset, records.byteLength);
  const path = resolve(outputDirectory, `${name}.f32`);
  writeFileSync(path, bytes);
  return { path, records: count, microRays: count * beamSamples, sha256: sha256(bytes), coverageMean: coverageSum / count };
}

const train = generateSplit(
  'train', TRAIN_ELEVATIONS, TRAIN_HEIGHTS, trainAzimuthBins, trainCentersPerStratum, 0x73a7f00d,
);
const validation = generateSplit(
  'validation', VALIDATION_ELEVATIONS, VALIDATION_HEIGHTS,
  validationAzimuthBins, validationCentersPerStratum, 0xdadc0ffe,
);

const coherence = new Float32Array(coherencePairs * 2 * OUTPUT_FIELDS);
let coherencePair = 0;
let coherenceAttempt = 0;
const coherenceClasses = directionClasses(VALIDATION_ELEVATIONS);
const heightSpan = geometry.bounds.max[1] - geometry.bounds.min[1];
console.error(`[transfer-beam] coherence: ${coherencePairs.toLocaleString()} held-out origin-shift pairs`);
while (coherencePair < coherencePairs) {
  const key = coherenceAttempt++;
  if (coherenceAttempt > coherencePairs * 100) throw new Error('could not construct enough in-slab coherence pairs');
  const definition = coherenceClasses[key % coherenceClasses.length]!;
  const phaseX = hashUnit(0x91e10da5, key * 5 + 0);
  const phaseZ = hashUnit(0x7f4a7c15, key * 5 + 1);
  const height = 0.05 + hashUnit(0x94d049bb, key * 5 + 2) * 0.90;
  const azimuth = hashUnit(0x6c8e9cf5, key * 5 + 3) * Math.PI * 2;
  const d = direction(definition, azimuth);
  const shiftedHeight = height + coherenceShiftMetres * d[1] / heightSpan;
  if (!(shiftedHeight > 0.001 && shiftedHeight < 0.999)) continue;
  const shiftedX = wrap01(phaseX + coherenceShiftMetres * d[0] / geometry.tileSizeX);
  const shiftedZ = wrap01(phaseZ + coherenceShiftMetres * d[2] / geometry.tileSizeZ);
  const rotation = hashUnit(0x4cf5ad43, key * 5 + 4);
  const firstEvents = traceBeam(phaseX, height, phaseZ, d, rotation);
  const secondEvents = traceBeam(shiftedX, shiftedHeight, shiftedZ, d, rotation);
  // Require a meaningful right-censoring comparison rather than two empty
  // beams. Crossing events are retained: the successor is then allowed to
  // advance, and the evaluator reports that separately.
  if (firstEvents.length === 0 || secondEvents.length === 0) continue;
  const firstOffset = coherencePair * 2 * OUTPUT_FIELDS;
  const secondOffset = firstOffset + OUTPUT_FIELDS;
  coherence.set([phaseX, height, phaseZ, ...d], firstOffset);
  coherence.set([shiftedX, shiftedHeight, shiftedZ, ...d], secondOffset);
  encodeBeam(coherence, firstOffset, firstEvents, beamSamples);
  encodeBeam(coherence, secondOffset, secondEvents, beamSamples);
  coherencePair++;
}
const coherenceBytes = Buffer.from(coherence.buffer, coherence.byteOffset, coherence.byteLength);
const coherencePath = resolve(outputDirectory, 'coherence.f32');
writeFileSync(coherencePath, coherenceBytes);

const qaDefinitions = [
  { id: '001-horizontal-y075-az000', elevationDegrees: 0, sense: 'horizontal' as const, height: 0.75, azimuthDegrees: 0 },
  { id: '002-grazing-down-y090-az045', elevationDegrees: 0.1, sense: 'downward' as const, height: 0.9, azimuthDegrees: 45 },
  { id: '003-grazing-up-y010-az045', elevationDegrees: 0.1, sense: 'upward' as const, height: 0.1, azimuthDegrees: 45 },
  { id: '004-five-down-y050-az090', elevationDegrees: 5, sense: 'downward' as const, height: 0.5, azimuthDegrees: 90 },
  { id: '005-thirtyfive-down-y050-az045', elevationDegrees: 35, sense: 'downward' as const, height: 0.5, azimuthDegrees: 45 },
  { id: '006-thirtyfive-up-y050-az045', elevationDegrees: 35, sense: 'upward' as const, height: 0.5, azimuthDegrees: 45 },
  { id: '007-vertical-down-y090', elevationDegrees: 90, sense: 'downward' as const, height: 0.9, azimuthDegrees: 0 },
  { id: '008-vertical-up-y010', elevationDegrees: 90, sense: 'upward' as const, height: 0.1, azimuthDegrees: 0 },
];
const qa = new Float32Array(qaDefinitions.length * qaGrid * qaGrid * OUTPUT_FIELDS);
let qaRecord = 0;
console.error(`[transfer-beam] QA: ${(qaDefinitions.length * qaGrid * qaGrid).toLocaleString()} beams`);
for (const definition of qaDefinitions) {
  const d = direction(definition, definition.azimuthDegrees * Math.PI / 180);
  for (let zIndex = 0; zIndex < qaGrid; zIndex++) for (let xIndex = 0; xIndex < qaGrid; xIndex++) {
    const phaseX = (xIndex + 0.5) / qaGrid;
    const phaseZ = (zIndex + 0.5) / qaGrid;
    const offset = qaRecord * OUTPUT_FIELDS;
    qa.set([phaseX, definition.height, phaseZ, ...d], offset);
    encodeBeam(qa, offset, traceBeam(phaseX, definition.height, phaseZ, d, qaRecord * GOLDEN_RATIO), beamSamples);
    qaRecord++;
    if (qaRecord % 1_000 === 0) console.error(`[transfer-beam] QA: ${qaRecord.toLocaleString()}/${(qa.length / OUTPUT_FIELDS).toLocaleString()}`);
  }
}
const qaBytes = Buffer.from(qa.buffer, qa.byteOffset, qa.byteLength);
const qaPath = resolve(outputDirectory, 'qa.f32');
writeFileSync(qaPath, qaBytes);
const manifest = {
  schema: 'laas-groundcover-transfer-field-beam-truth/v1',
  source: {
    file: sourcePath,
    bytes: sourceBytes.byteLength,
    sha256: sourceHash,
    profileId: geometry.profileId,
    tileSize: [geometry.tileSizeX, geometry.tileSizeZ],
    bounds: geometry.bounds,
  },
  command: [process.execPath, ...process.execArgv, ...process.argv.slice(1)],
  configuration,
  recipeSha256: recipeHash,
  record: {
    type: 'little-endian float32',
    floats: OUTPUT_FIELDS,
    fields: [
      'phaseX', 'heightFraction', 'phaseZ', 'directionX', 'directionY', 'directionZ', 'coverage',
      ...Array.from({ length: STRATA }, (_, index) => [
        `stratum${index}DepthMeanMetres`, `stratum${index}DepthVarianceMetres2`,
        `stratum${index}LinearColorR`, `stratum${index}LinearColorG`, `stratum${index}LinearColorB`,
        `stratum${index}NormalX`, `stratum${index}NormalY`, `stratum${index}NormalZ`,
      ]).flat(),
    ],
  },
  splitContract: {
    train: 'seed 0x73a7f00d; stratified named elevation/height/azimuth; exact height anchors plus deterministic open-Y samples',
    validation: 'disjoint seed 0xdadc0ffe; twice the azimuth strata, held-out midpoint heights, and only inter-anchor elevation rows absent from training',
    qa: 'regular phase grids for eight named full-sphere origin/direction slices',
  },
  files: {
    train: { ...train, path: relative(outputDirectory, train.path) },
    validation: { ...validation, path: relative(outputDirectory, validation.path) },
    qa: {
      path: relative(outputDirectory, qaPath), records: qa.length / OUTPUT_FIELDS,
      microRays: qa.length / OUTPUT_FIELDS * beamSamples, bytes: qaBytes.byteLength,
      sha256: sha256(qaBytes), grid: qaGrid, slices: qaDefinitions,
    },
    coherence: {
      path: relative(outputDirectory, coherencePath), records: coherence.length / OUTPUT_FIELDS,
      pairs: coherencePairs, microRays: coherencePairs * 2 * beamSamples,
      shiftMetres: coherenceShiftMetres, bytes: coherenceBytes.byteLength,
      sha256: sha256(coherenceBytes),
    },
  },
  truthContract: [
    'every micro-ray is the nearest analytic decoded GCRP/v4 mesh event over periodic XZ copies in the declared forward slab/horizon',
    'coverage and three equal-conditional-mass depth/colour/normal strata are constructed from an explicit deterministic phase/direction disk before codec training',
    'stratum means are representative transfer events with recorded within-stratum depth variance; they are not exact center-ray triangle depths',
    'coherence records are held-out forward origin shifts on one central oriented line and test stochastic right-censoring independently of aggregate images',
    'all traversal and micro-ray work is offline truth generation; the proposed runtime consumes one fixed decoded record',
  ],
  invariants: {
    exteriorTopEntryDatumShiftTrials: 1024,
    maximumExteriorTopEntryErrorMetresF64: maximumExteriorEntryInvarianceError,
  },
};
const manifestPath = resolve(outputDirectory, 'manifest.json');
writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
console.error(`[transfer-beam] wrote ${manifestPath}`);
process.stdout.write(`${manifestPath}\n`);
