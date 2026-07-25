/**
 * Exact actual-source truth for the frozen rank-eight endpoint-conditioned GDM
 * gate. Runtime is not involved. Every record is a finite, half-open pointed
 * segment through the exact isolated Calamagrostis source.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { makeCalamagrostisCanescensBandlimitedSource } from '../../EstonianGraminoids';
import { TriangleBvh, type CensusVec3, type DecodedOwnedProfileGeometry } from '../../ExteriorClosureCensus';
import { periodicNearestSuccessor, type OriginAwareTruthHit } from '../../OriginAwareRayTruth';
import type { IndexedMesh } from '../../ProfileFormat';

const FIELDS = 16;
const HORIZON = 155;
const ENDPOINT_ALPHA = 10;
const PHASE_RESOLUTION = 212;
const MICRO_SAMPLES = 4;
const TRAIN_RECORDS = 60_000;
const VALIDATION_RECORDS_PER_AXIS = 4_096;
const GENERAL_VALIDATION_RECORDS = 8_192;
const SEQUENCE_COUNT = 1_024;
const SEQUENCE_STEPS = 8;
const CENSOR_PAIRS = 2_048;
const ORACLE_W_ROWS = 256;
const ORACLE_E_COLUMNS = 256;

interface Args { [key: string]: string | boolean }
interface EventAttributes { color: CensusVec3; normal: CensusVec3 }
interface Sample { u: number; v: number; y: number; phi: number; elevation: number; endpointQ: number }
interface Interval { min: number; max: number }

const HOLDS: Record<'u' | 'v' | 'y' | 'phi' | 'elevation' | 'endpointQ', Interval> = {
  u: { min: 0.43, max: 0.49 },
  v: { min: 0.69, max: 0.75 },
  y: { min: 0.46, max: 0.56 },
  phi: { min: 0.30, max: 0.36 },
  elevation: { min: 0.05, max: 0.13 },
  endpointQ: { min: 0.52, max: 0.62 },
};

function args(argv: readonly string[]): Args {
  const result: Args = {};
  for (let index = 0; index < argv.length; index++) {
    const key = argv[index]!;
    if (!key.startsWith('--')) continue;
    const value = argv[index + 1];
    if (value !== undefined && !value.startsWith('--')) { result[key.slice(2)] = value; index++; }
    else result[key.slice(2)] = true;
  }
  return result;
}

function sha(value: Uint8Array | string): string { return createHash('sha256').update(value).digest('hex'); }
function wrap01(value: number): number { return value - Math.floor(value); }
function hashUnit(seed: number, index: number): number {
  let value = (seed ^ index) >>> 0;
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d);
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b);
  return ((value ^ (value >>> 16)) >>> 0) / 0x1_0000_0000;
}
function normalized(value: CensusVec3): CensusVec3 {
  const length = Math.hypot(...value);
  return length > 1e-20 ? [value[0] / length, value[1] / length, value[2] / length] : [0, 1, 0];
}
function inInterval(value: number, interval: Interval): boolean { return value >= interval.min && value < interval.max; }
function endpointFromQ(q: number): number { return Math.log1p(ENDPOINT_ALPHA * HORIZON) === 0 ? 0 : Math.expm1(q * Math.log1p(ENDPOINT_ALPHA * HORIZON)) / ENDPOINT_ALPHA; }
function qFromEndpoint(length: number): number { return Math.log1p(ENDPOINT_ALPHA * Math.max(0, length)) / Math.log1p(ENDPOINT_ALPHA * HORIZON); }

function meshBounds(mesh: IndexedMesh): { min: CensusVec3; max: CensusVec3 } {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let vertex = 0; vertex < mesh.positions.length; vertex += 3) for (let axis = 0; axis < 3; axis++) {
    min[axis] = Math.min(min[axis], mesh.positions[vertex + axis]!);
    max[axis] = Math.max(max[axis], mesh.positions[vertex + axis]!);
  }
  return { min, max };
}

function geometryFrom(mesh: IndexedMesh, tile: { originX: number; originZ: number; sizeX: number; sizeZ: number; topH: number }): DecodedOwnedProfileGeometry {
  const bounds = meshBounds(mesh);
  return {
    version: 4, profileId: 2, topH: tile.topH,
    tileOriginX: tile.originX, tileOriginZ: tile.originZ,
    tileSizeX: tile.sizeX, tileSizeZ: tile.sizeZ,
    bounds, positions: Float64Array.from(mesh.positions), triangles: Uint32Array.from(mesh.indices),
    vertexCount: mesh.positions.length / 3, triangleCount: mesh.indices.length / 3,
  };
}

function plumeTriangles(mesh: IndexedMesh): Uint8Array {
  if (!mesh.colors) throw new Error('source requires authored colours');
  const result = new Uint8Array(mesh.indices.length / 3);
  for (let triangle = 0; triangle < result.length; triangle++) {
    let plume = 0;
    for (let corner = 0; corner < 3; corner++) {
      const vertex = mesh.indices[triangle * 3 + corner]!;
      plume += mesh.colors[vertex * 3]! > mesh.colors[vertex * 3 + 1]! ? 1 : 0;
    }
    result[triangle] = plume >= 2 ? 1 : 0;
  }
  return result;
}

function attributes(mesh: IndexedMesh, geometry: DecodedOwnedProfileGeometry, origin: CensusVec3, direction: CensusVec3, hit: OriginAwareTruthHit): EventAttributes {
  const point: CensusVec3 = [
    origin[0] + direction[0] * hit.t - hit.copyX * geometry.tileSizeX,
    origin[1] + direction[1] * hit.t,
    origin[2] + direction[2] * hit.t - hit.copyZ * geometry.tileSizeZ,
  ];
  const vertices = [0, 1, 2].map((corner) => mesh.indices[hit.triangleId * 3 + corner]!);
  const positions = vertices.map((vertex) => [mesh.positions[vertex * 3]!, mesh.positions[vertex * 3 + 1]!, mesh.positions[vertex * 3 + 2]!] as CensusVec3);
  const e0: CensusVec3 = [positions[1]![0] - positions[0]![0], positions[1]![1] - positions[0]![1], positions[1]![2] - positions[0]![2]];
  const e1: CensusVec3 = [positions[2]![0] - positions[0]![0], positions[2]![1] - positions[0]![1], positions[2]![2] - positions[0]![2]];
  const p: CensusVec3 = [point[0] - positions[0]![0], point[1] - positions[0]![1], point[2] - positions[0]![2]];
  const d00 = e0[0] ** 2 + e0[1] ** 2 + e0[2] ** 2;
  const d01 = e0[0] * e1[0] + e0[1] * e1[1] + e0[2] * e1[2];
  const d11 = e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2;
  const d20 = p[0] * e0[0] + p[1] * e0[1] + p[2] * e0[2];
  const d21 = p[0] * e1[0] + p[1] * e1[1] + p[2] * e1[2];
  const inverse = 1 / Math.max(1e-30, d00 * d11 - d01 * d01);
  const b = Math.max(0, (d11 * d20 - d01 * d21) * inverse);
  const c = Math.max(0, (d00 * d21 - d01 * d20) * inverse);
  const a = Math.max(0, 1 - b - c);
  const sum = Math.max(1e-30, a + b + c);
  const w = [a / sum, b / sum, c / sum];
  const color: [number, number, number] = [0, 0, 0];
  const normal: [number, number, number] = [0, 0, 0];
  for (let corner = 0; corner < 3; corner++) for (let axis = 0; axis < 3; axis++) {
    const vertex = vertices[corner]!;
    color[axis] += (mesh.colors?.[vertex * 3 + axis] ?? 1) * w[corner]!;
    normal[axis] += mesh.normals[vertex * 3 + axis]! * w[corner]!;
  }
  return { color, normal: normalized(normal) };
}

const cli = args(process.argv.slice(2));
const fixture = makeCalamagrostisCanescensBandlimitedSource();
const geometry = geometryFrom(fixture.mesh, fixture.tile);
const isPlume = plumeTriangles(fixture.mesh);
const sourceSha256 = sha(JSON.stringify(fixture.mesh));
const configuration = {
  schema: 'laas-endpoint-conditioned-gdm-truth/v1', sourceSha256,
  fields: ['u', 'v', 'originHeight', 'azimuth01', 'signedElevation', 'endpointQ', 'A', 'm', 'C.r', 'C.g', 'C.b', 'N.x', 'N.y', 'N.z', 'structureCoverage', 'plumeCoverage'],
  endpointMap: { q: 'log1p(10*L)/log1p(10*155)', alpha: ENDPOINT_ALPHA, horizonMetres: HORIZON },
  halfOpenSegment: '[0,L)', phaseResolution: PHASE_RESOLUTION, microSamples: MICRO_SAMPLES,
  microFootprint: 'four deterministic phase-disk samples within one 212-sample phase texel; direction, physical origin height, and finite endpoint are shared; opaque microscopic hits become premultiplied fractional footprint moments',
  normalSemantics: 'each interpolated source normal is face-forwarded so dot(n,-d)>=0 before premultiplied accumulation',
  counts: { train: TRAIN_RECORDS, validationPerHeldAxis: VALIDATION_RECORDS_PER_AXIS, generalValidation: GENERAL_VALIDATION_RECORDS, sequenceCount: SEQUENCE_COUNT, sequenceSteps: SEQUENCE_STEPS, censorPairs: CENSOR_PAIRS, oracleWRows: ORACLE_W_ROWS, oracleEColumns: ORACLE_E_COLUMNS },
  heldOutIntervals: HOLDS,
  source: { generator: fixture.generator, vertices: geometry.vertexCount, triangles: geometry.triangleCount, bounds: geometry.bounds, tile: fixture.tile },
  toolSha256: sha(readFileSync(fileURLToPath(import.meta.url))),
};
const recipeSha256 = sha(JSON.stringify(configuration));
const outputRoot = resolve(typeof cli.output === 'string' ? cli.output : `data/work/groundcover-endpoint-gdm/${sourceSha256}/${recipeSha256}/truth`);
mkdirSync(outputRoot, { recursive: true });
console.error(`[endpoint-gdm-truth] BVH ${geometry.triangleCount.toLocaleString()} triangles`);
const bvh = TriangleBvh.build(geometry, 8);
const heightSpan = geometry.bounds.max[1] - geometry.bounds.min[1];
const microOffsets = [[-0.28, -0.11], [0.21, -0.32], [-0.09, 0.30], [0.34, 0.17]] as const;

function direction(sample: Sample): CensusVec3 {
  const theta = sample.elevation * Math.PI * 0.5;
  const phi = sample.phi * Math.PI * 2;
  const horizontal = Math.abs(sample.elevation) === 1 ? 0 : Math.cos(theta);
  return [horizontal * Math.cos(phi), Math.sin(theta), horizontal * Math.sin(phi)];
}

function slabLength(y: number, d: CensusVec3): number {
  const physicalY = geometry.bounds.min[1] + y * heightSpan;
  if (d[1] > 1e-12) return Math.max(0, (geometry.bounds.max[1] - physicalY) / d[1]);
  if (d[1] < -1e-12) return Math.max(0, (geometry.bounds.min[1] - physicalY) / d[1]);
  return HORIZON;
}

function trace(sampleInput: Sample): Float32Array {
  const d = direction(sampleInput);
  const length = Math.max(1e-5, Math.min(endpointFromQ(sampleInput.endpointQ), slabLength(sampleInput.y, d), HORIZON));
  const sample = { ...sampleInput, endpointQ: qFromEndpoint(length) };
  let hits = 0;
  let moment = 0;
  let structure = 0;
  let plume = 0;
  const color: [number, number, number] = [0, 0, 0];
  const normal: [number, number, number] = [0, 0, 0];
  for (let micro = 0; micro < MICRO_SAMPLES; micro++) {
    const jitter = microOffsets[micro]!;
    const u = wrap01(sample.u + jitter[0] / PHASE_RESOLUTION);
    const v = wrap01(sample.v + jitter[1] / PHASE_RESOLUTION);
    const origin: CensusVec3 = [geometry.tileOriginX + u * geometry.tileSizeX, geometry.bounds.min[1] + sample.y * heightSpan, geometry.tileOriginZ + v * geometry.tileSizeZ];
    const candidate = periodicNearestSuccessor(geometry, bvh, origin, d, length, 1e-9);
    const hit = candidate && candidate.t < length ? candidate : null;
    if (!hit) continue;
    const attr = attributes(fixture.mesh, geometry, origin, d, hit);
    const facing = attr.normal[0] * -d[0] + attr.normal[1] * -d[1] + attr.normal[2] * -d[2];
    const eventNormal: CensusVec3 = facing >= 0 ? attr.normal : [-attr.normal[0], -attr.normal[1], -attr.normal[2]];
    hits++;
    moment += hit.t;
    if (isPlume[hit.triangleId]) plume++; else structure++;
    for (let axis = 0; axis < 3; axis++) {
      color[axis] += attr.color[axis]!;
      normal[axis] += eventNormal[axis]!;
    }
  }
  const A = hits / MICRO_SAMPLES;
  const M = moment / MICRO_SAMPLES;
  const output = new Float32Array(FIELDS);
  output.set([sample.u, sample.v, sample.y, sample.phi, sample.elevation, sample.endpointQ, A, M / length], 0);
  for (let axis = 0; axis < 3; axis++) {
    output[8 + axis] = color[axis]! / MICRO_SAMPLES;
    output[11 + axis] = normal[axis]! / MICRO_SAMPLES;
  }
  output[14] = structure / MICRO_SAMPLES;
  output[15] = plume / MICRO_SAMPLES;
  return output;
}

function randomSample(seed: number, index: number): Sample {
  return {
    u: hashUnit(seed ^ 0x1234, index * 7), v: hashUnit(seed ^ 0x2345, index * 7 + 1),
    y: 0.002 + hashUnit(seed ^ 0x3456, index * 7 + 2) * 0.996,
    phi: hashUnit(seed ^ 0x4567, index * 7 + 3),
    elevation: hashUnit(seed ^ 0x5678, index * 7 + 4) * 2 - 1,
    endpointQ: 0.01 + hashUnit(seed ^ 0x6789, index * 7 + 5) * 0.99,
  };
}

function excludedFromTrain(sample: Sample): boolean {
  return (Object.keys(HOLDS) as Array<keyof typeof HOLDS>).some((key) => inInterval(sample[key], HOLDS[key]));
}

function writeRecords(name: string, samples: readonly Sample[]): { path: string; records: number; sha256: string; bytes: number } {
  const records = new Float32Array(samples.length * FIELDS);
  for (let index = 0; index < samples.length; index++) {
    records.set(trace(samples[index]!), index * FIELDS);
    if ((index + 1) % 10_000 === 0) console.error(`[endpoint-gdm-truth] ${name} ${index + 1}/${samples.length}`);
  }
  const bytes = Buffer.from(records.buffer, records.byteOffset, records.byteLength);
  const path = resolve(outputRoot, `${name}.f32`);
  writeFileSync(path, bytes);
  return { path: `${name}.f32`, records: samples.length, sha256: sha(bytes), bytes: bytes.byteLength };
}

const trainSamples: Sample[] = [];
for (let attempt = 0; trainSamples.length < TRAIN_RECORDS; attempt++) {
  const sample = randomSample(0x7a41d13, attempt);
  if (!excludedFromTrain(sample)) trainSamples.push(sample);
}
const files: Record<string, { path: string; records: number; sha256: string; bytes: number }> = {};
files.train = writeRecords('train', trainSamples);

for (const [axisIndex, axis] of (Object.keys(HOLDS) as Array<keyof typeof HOLDS>).entries()) {
  const samples: Sample[] = [];
  for (let index = 0; index < VALIDATION_RECORDS_PER_AXIS; index++) {
    const sample = randomSample(0x17e1d + axisIndex * 0x1000, index);
    const interval = HOLDS[axis];
    sample[axis] = interval.min + hashUnit(0x9f43 + axisIndex, index) * (interval.max - interval.min);
    samples.push(sample);
  }
  files[`holdout-${axis}`] = writeRecords(`holdout-${axis}`, samples);
}

const general: Sample[] = [];
for (let attempt = 0; general.length < GENERAL_VALIDATION_RECORDS; attempt++) {
  const sample = randomSample(0xc0ffee, attempt);
  if (!excludedFromTrain(sample)) general.push(sample);
}
files.validation = writeRecords('validation', general);

// Direct matrix oracle: every fixed W=(u,v,phi) row is crossed with every
// fixed E=(originY,signedElevation,L) column. Independent best rank-8 SVDs for
// A and M are an optimistic lower bound on the shared-factor/head candidate.
const oracleW = Array.from({ length: ORACLE_W_ROWS }, (_unused, row) => {
  const sample = randomSample(0x08ac1e, row);
  return { u: sample.u, v: sample.v, phi: sample.phi };
});
const oracleE = Array.from({ length: ORACLE_E_COLUMNS }, (_unused, column) => {
  const sample = randomSample(0xe11d51, column);
  const elevation = column % 32 === 0 ? 0 : column % 32 === 1 ? 1 : column % 32 === 2 ? -1 : sample.elevation;
  return { y: sample.y, elevation, endpointQ: sample.endpointQ };
});
const oracle: Sample[] = [];
for (const w of oracleW) for (const e of oracleE) oracle.push({ ...w, ...e });
files.oracle = writeRecords('oracle', oracle);

const horizontal: Sample[] = [];
const vertical: Sample[] = [];
for (let index = 0; index < 4_096; index++) {
  const base = randomSample(0x501a, index);
  horizontal.push({ ...base, elevation: 0 });
}
for (let pair = 0; pair < 2_048; pair++) {
  const base = randomSample(0x901a, pair);
  const elevation = pair & 1 ? 1 : -1;
  vertical.push(
    { ...base, elevation, phi: 0 },
    { ...base, elevation, phi: hashUnit(0x7e471ca1, pair) },
  );
}
files.horizontal = writeRecords('horizontal', horizontal);
files.vertical = writeRecords('vertical', vertical);

const sequence: Sample[] = [];
for (let chain = 0; chain < SEQUENCE_COUNT; chain++) {
  const base = randomSample(0x5e9a, chain);
  for (let step = 0; step < SEQUENCE_STEPS; step++) sequence.push({ ...base, endpointQ: 0.01 + step / (SEQUENCE_STEPS - 1) * 0.99 });
}
files.monotonic = writeRecords('monotonic', sequence);

const censor: Sample[] = [];
const censorShifts = new Float32Array(CENSOR_PAIRS);
for (let pair = 0; pair < CENSOR_PAIRS; pair++) {
  let base = randomSample(0xc3150, pair);
  base = { ...base, y: 0.1 + base.y * 0.8, endpointQ: 0.55 + base.endpointQ * 0.4 };
  const d = direction(base);
  const L = Math.min(endpointFromQ(base.endpointQ), slabLength(base.y, d), HORIZON);
  const shift = Math.min(0.08, Math.max(0.005, L * 0.15));
  const shiftedY = base.y + d[1] * shift / heightSpan;
  if (!(shiftedY > 0.001 && shiftedY < 0.999 && L > shift + 1e-4)) { pair--; continue; }
  censorShifts[pair] = shift;
  censor.push(base, {
    u: wrap01(base.u + d[0] * shift / geometry.tileSizeX),
    v: wrap01(base.v + d[2] * shift / geometry.tileSizeZ),
    y: shiftedY, phi: base.phi, elevation: base.elevation,
    endpointQ: qFromEndpoint(L - shift),
  });
}
files.censor = writeRecords('censor', censor);
const shiftBytes = Buffer.from(censorShifts.buffer);
writeFileSync(resolve(outputRoot, 'censor-shifts.f32'), shiftBytes);

const manifest = {
  ...configuration, recipeSha256, outputRoot,
  files,
  censorShifts: { path: 'censor-shifts.f32', records: CENSOR_PAIRS, sha256: sha(shiftBytes), bytes: shiftBytes.byteLength },
  oracle: { WRows: ORACLE_W_ROWS, EColumns: ORACLE_E_COLUMNS, ordering: 'row-major W row then E column; independent A/M rank-8 SVD is an optimistic lower bound' },
  contract: [
    'outputs are linear-premultiplied A,m,C,N over the complete structure+plume union',
    'training excludes every declared contiguous coordinate interval',
    'exact horizontal, vertical, endpoint-monotonic and same-line shifted-origin sequences are separate evaluation files',
    'vertical records are consecutive phi-invariance pairs with identical physical rays',
    'no runtime shader or packed factor texture is produced by truth generation',
  ],
};
writeFileSync(resolve(outputRoot, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
console.log(resolve(outputRoot, 'manifest.json'));
