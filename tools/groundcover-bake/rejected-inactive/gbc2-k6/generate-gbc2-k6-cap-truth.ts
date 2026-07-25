/**
 * Honest shared-origin exterior truth for one production GBC2-K6 cap level.
 *
 * Every record is one real pinhole pixel. Its sixteen subrays share one
 * camera origin, intersect the top carrier independently, and raycast the
 * infinitely repeated accepted Calamagrostis mesh. This is an offline cook;
 * no traversal or quadrature belongs in the runtime representation.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';
import {
  TriangleBvh,
  decodeOwnedProfileGeometry,
  type CensusVec3,
  type DecodedOwnedProfileGeometry,
} from '../../ExteriorClosureCensus';
import { periodicNearestSuccessor, type OriginAwareTruthHit } from '../../OriginAwareRayTruth';

const WORKSPACE = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const SOURCE = resolve(WORKSPACE, 'src/assets/groundcover/calamagrostis-canescens.gcrp');
const EXPECTED_SOURCE_SHA = '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const TRAIN_RECORDS = 4096;
const VALIDATION_RECORDS = 1024;
const RECORD_FLOATS = 22;
const QUADRATURE_SIDE = 4;
const PRODUCTION_PIXEL_ANGLE = 60 * Math.PI / 180 / 1920;
const STANDOFF_METRES = 4;
const MINIMUM_ELEVATION_DEGREES = 5;
const DEPTH_HORIZON_METRES = 155;
const CARRIER_GUARD_METRES = 0.025;
const TAU = Math.PI * 2;

interface SurfaceSample {
  depth: number;
  color: CensusVec3;
  normal: CensusVec3;
}

function sha256(value: Uint8Array | string): string {
  return createHash('sha256').update(value).digest('hex');
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(',')}}`;
}

function normalized(value: CensusVec3): CensusVec3 {
  const length = Math.hypot(value[0], value[1], value[2]);
  if (!(length > 0)) throw new Error('cannot normalize a zero vector');
  return [value[0] / length, value[1] / length, value[2] / length];
}

function cross(a: CensusVec3, b: CensusVec3): CensusVec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function addScaled(a: CensusVec3, b: CensusVec3, scale: number): CensusVec3 {
  return [a[0] + b[0] * scale, a[1] + b[1] * scale, a[2] + b[2] * scale];
}

function subtract(a: CensusVec3, b: CensusVec3): CensusVec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function decodeOct(x01: number, y01: number): CensusVec3 {
  let x = x01 * 2 - 1;
  let y = y01 * 2 - 1;
  let z = 1 - Math.abs(x) - Math.abs(y);
  if (z < 0) {
    const oldX = x;
    x = (1 - Math.abs(y)) * (oldX >= 0 ? 1 : -1);
    y = (1 - Math.abs(oldX)) * (y >= 0 ? 1 : -1);
  }
  return normalized([x, y, z]);
}

function attributes(
  bytes: Uint8Array,
  geometry: DecodedOwnedProfileGeometry,
  origin: CensusVec3,
  direction: CensusVec3,
  hit: OriginAwareTruthHit,
): SurfaceSample {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const vertexOffset = view.getUint32(84, true);
  const triangle = hit.triangleId * 3;
  const vertices = [
    geometry.triangles[triangle]!,
    geometry.triangles[triangle + 1]!,
    geometry.triangles[triangle + 2]!,
  ] as const;
  const point: CensusVec3 = [
    origin[0] + direction[0] * hit.t - hit.copyX * geometry.tileSizeX,
    origin[1] + direction[1] * hit.t,
    origin[2] + direction[2] * hit.t - hit.copyZ * geometry.tileSizeZ,
  ];
  const position = (vertex: number): CensusVec3 => {
    const offset = vertex * 3;
    return [geometry.positions[offset]!, geometry.positions[offset + 1]!, geometry.positions[offset + 2]!];
  };
  const a = position(vertices[0]);
  const b = position(vertices[1]);
  const c = position(vertices[2]);
  const e0 = subtract(b, a);
  const e1 = subtract(c, a);
  const q = subtract(point, a);
  const d00 = e0[0] ** 2 + e0[1] ** 2 + e0[2] ** 2;
  const d01 = e0[0] * e1[0] + e0[1] * e1[1] + e0[2] * e1[2];
  const d11 = e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2;
  const d20 = q[0] * e0[0] + q[1] * e0[1] + q[2] * e0[2];
  const d21 = q[0] * e1[0] + q[1] * e1[1] + q[2] * e1[2];
  const denominator = d00 * d11 - d01 * d01;
  const baryB = denominator === 0 ? 0 : (d11 * d20 - d01 * d21) / denominator;
  const baryC = denominator === 0 ? 0 : (d00 * d21 - d01 * d20) / denominator;
  const weights = [1 - baryB - baryC, baryB, baryC];
  const color: [number, number, number] = [0, 0, 0];
  const normal: [number, number, number] = [0, 0, 0];
  for (let corner = 0; corner < 3; corner++) {
    const offset = vertexOffset + vertices[corner]! * 16;
    const vertexNormal = decodeOct(
      view.getUint16(offset + 12, true) / 65535,
      view.getUint16(offset + 14, true) / 65535,
    );
    color[0] += weights[corner]! * view.getUint16(offset + 6, true) / 65535;
    color[1] += weights[corner]! * view.getUint16(offset + 8, true) / 65535;
    color[2] += weights[corner]! * view.getUint16(offset + 10, true) / 65535;
    normal[0] += weights[corner]! * vertexNormal[0];
    normal[1] += weights[corner]! * vertexNormal[1];
    normal[2] += weights[corner]! * vertexNormal[2];
  }
  return { depth: hit.t, color, normal: normalized(normal) };
}

function radicalInverse(indexInput: number, base: number): number {
  let index = indexInput;
  let denominator = 1;
  let value = 0;
  while (index > 0) {
    denominator *= base;
    value += (index % base) / denominator;
    index = Math.floor(index / base);
  }
  return value;
}

function coordinates(record: number): readonly [number, number, number, number] {
  const index = record + 1;
  const minimumElevation = MINIMUM_ELEVATION_DEGREES / 90;
  return [
    radicalInverse(index, 2),
    radicalInverse(index, 3),
    radicalInverse(index, 5),
    minimumElevation + (1 - minimumElevation) * radicalInverse(index, 7),
  ];
}

function clusterByDepth(samples: readonly SurfaceSample[]): readonly [SurfaceSample[], SurfaceSample[]] {
  if (samples.length < 2) return [[...samples], []];
  let centre0 = Math.min(...samples.map((sample) => sample.depth));
  let centre1 = Math.max(...samples.map((sample) => sample.depth));
  let group0: SurfaceSample[] = [];
  let group1: SurfaceSample[] = [];
  for (let iteration = 0; iteration < 8; iteration++) {
    group0 = [];
    group1 = [];
    for (const sample of samples) {
      (Math.abs(sample.depth - centre0) <= Math.abs(sample.depth - centre1) ? group0 : group1).push(sample);
    }
    if (group0.length > 0) centre0 = group0.reduce((sum, sample) => sum + sample.depth, 0) / group0.length;
    if (group1.length > 0) centre1 = group1.reduce((sum, sample) => sum + sample.depth, 0) / group1.length;
  }
  return centre0 <= centre1 ? [group0, group1] : [group1, group0];
}

function encodeStratum(samples: readonly SurfaceSample[], quadratureCount: number): number[] {
  const output = new Array<number>(9).fill(0);
  for (const sample of samples) {
    const weight = 1 / quadratureCount;
    const depth = Math.min(1, sample.depth / DEPTH_HORIZON_METRES);
    output[0]! += weight;
    output[1]! += weight * depth;
    output[2]! += weight * depth * depth;
    output[3]! += weight * sample.color[0];
    output[4]! += weight * sample.color[1];
    output[5]! += weight * sample.color[2];
    output[6]! += weight * sample.normal[0];
    output[7]! += weight * sample.normal[1];
    output[8]! += weight * sample.normal[2];
  }
  return output;
}

function integrateRecord(
  bytes: Uint8Array,
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  coordinate: readonly [number, number, number, number],
  top: number,
): Float32Array {
  const [u, v, azimuth01, elevation01] = coordinate;
  const azimuth = azimuth01 * TAU;
  const elevation = elevation01 * Math.PI * 0.5;
  const horizontal = Math.cos(elevation);
  const centralDirection: CensusVec3 = [
    horizontal * Math.cos(azimuth),
    -Math.sin(elevation),
    horizontal * Math.sin(azimuth),
  ];
  const centralQ: CensusVec3 = [
    geometry.tileOriginX + u * geometry.tileSizeX,
    top,
    geometry.tileOriginZ + v * geometry.tileSizeZ,
  ];
  const cameraOrigin = addScaled(centralQ, centralDirection, -STANDOFF_METRES);
  if (!(cameraOrigin[1] > top)) throw new Error('cap truth camera origin is not exterior');
  const right = Math.abs(centralDirection[1]) > 0.999999
    ? normalized([Math.cos(azimuth), 0, Math.sin(azimuth)])
    : normalized(cross(centralDirection, [0, 1, 0]));
  const up = normalized(cross(right, centralDirection));
  const tangent = Math.tan(PRODUCTION_PIXEL_ANGLE * 0.5);
  const samples: SurfaceSample[] = [];
  for (let sy = 0; sy < QUADRATURE_SIDE; sy++) for (let sx = 0; sx < QUADRATURE_SIDE; sx++) {
    const px = (2 * (sx + 0.5) / QUADRATURE_SIDE - 1) * tangent;
    const py = (2 * (sy + 0.5) / QUADRATURE_SIDE - 1) * tangent;
    const direction = normalized([
      centralDirection[0] + right[0] * px + up[0] * py,
      centralDirection[1] + right[1] * px + up[1] * py,
      centralDirection[2] + right[2] * px + up[2] * py,
    ]);
    if (!(direction[1] < 0)) continue;
    const entryT = (top - cameraOrigin[1]) / direction[1];
    if (!(entryT > 0)) continue;
    const entry = addScaled(cameraOrigin, direction, entryT);
    const hit = periodicNearestSuccessor(
      geometry,
      bvh,
      entry,
      direction,
      DEPTH_HORIZON_METRES,
      0,
    );
    if (hit) samples.push(attributes(bytes, geometry, entry, direction, hit));
  }
  const groups = clusterByDepth(samples);
  return Float32Array.from([
    u, v, azimuth01, elevation01,
    ...encodeStratum(groups[0], QUADRATURE_SIDE ** 2),
    ...encodeStratum(groups[1], QUADRATURE_SIDE ** 2),
  ]);
}

function quantiles(valuesInput: readonly number[]): Record<string, number> {
  if (valuesInput.length === 0) return { p50: 0, p95: 0, p99: 0, maximum: 0 };
  const values = [...valuesInput].sort((a, b) => a - b);
  const at = (p: number): number => values[Math.floor((values.length - 1) * p)]!;
  return { p50: at(0.5), p95: at(0.95), p99: at(0.99), maximum: values.at(-1)! };
}

function writeRecords(path: string, records: readonly Float32Array[]): void {
  const values = new Float32Array(records.length * RECORD_FLOATS);
  for (let index = 0; index < records.length; index++) values.set(records[index]!, index * RECORD_FLOATS);
  writeFileSync(path, Buffer.from(values.buffer, values.byteOffset, values.byteLength));
}

async function main(): Promise<void> {
  const started = performance.now();
  const bytes = readFileSync(SOURCE);
  const sourceSha = sha256(bytes);
  if (sourceSha !== EXPECTED_SOURCE_SHA) throw new Error(`accepted source changed: ${sourceSha}`);
  const geometry = decodeOwnedProfileGeometry(bytes);
  const top = geometry.bounds.max[1] + CARRIER_GUARD_METRES;
  console.error(`[gbc2-k6-truth] building BVH for ${geometry.triangleCount.toLocaleString()} triangles`);
  const bvh = TriangleBvh.build(geometry);
  const all: Float32Array[] = [];
  const count = TRAIN_RECORDS + VALIDATION_RECORDS;
  console.error(`[gbc2-k6-truth] tracing ${count.toLocaleString()} shared-origin pixels x ${QUADRATURE_SIDE ** 2} subrays`);
  for (let record = 0; record < count; record++) {
    all.push(integrateRecord(bytes, geometry, bvh, coordinates(record), top));
    if ((record + 1) % 256 === 0 || record + 1 === count) {
      console.error(`[gbc2-k6-truth] ${record + 1}/${count}`);
    }
  }
  const recipe = {
    schema: 'laas-gbc2-k6-shared-origin-cap-truth-recipe/v1',
    implementationSha256: sha256(readFileSync(fileURLToPath(import.meta.url))),
    sourceSha256: sourceSha,
    records: { train: TRAIN_RECORDS, validation: VALIDATION_RECORDS, floats: RECORD_FLOATS },
    domain: {
      chart: 'top cap', coordinates: ['phaseU', 'phaseV', 'azimuth01', 'elevation01'],
      minimumElevationDegrees: MINIMUM_ELEVATION_DEGREES,
      maximumElevationDegrees: 90,
      cameraStandoffMetres: STANDOFF_METRES,
      carrierTopMetres: top,
      exteriorCameraRequired: true,
    },
    pixel: {
      angleRadians: PRODUCTION_PIXEL_ANGLE,
      derivation: '60 degree horizontal FOV / 1920 pixels',
      quadrature: `${QUADRATURE_SIDE}x${QUADRATURE_SIDE}`,
      sharedPinholeOrigin: true,
      independentlyIntersectedCarrierPointPerSubray: true,
    },
    output: {
      depthHorizonMetres: DEPTH_HORIZON_METRES,
      floats: '4 address + 2 strata x [alpha, premul normalized depth m1/m2, premul RGB, premul normal XYZ]',
      strata: 'deterministic two-means clustering of first-hit subray depths',
    },
  };
  const recipeSha = sha256(canonicalJson(recipe));
  const outputRoot = resolve(WORKSPACE, 'data/work/groundcover-gbc2-k6-cap-truth', sourceSha.slice(0, 16), recipeSha.slice(0, 16));
  const qaRoot = resolve(outputRoot, 'qa');
  mkdirSync(qaRoot, { recursive: true });
  const trainPath = resolve(outputRoot, 'train.f32');
  const validationPath = resolve(outputRoot, 'validation.f32');
  writeRecords(trainPath, all.slice(0, TRAIN_RECORDS));
  writeRecords(validationPath, all.slice(TRAIN_RECORDS));
  const coverages = all.map((record) => record[4]! + record[13]!);
  const depths = all.flatMap((record) => {
    const result: number[] = [];
    if (record[4]! > 0) result.push(DEPTH_HORIZON_METRES * record[5]! / record[4]!);
    if (record[13]! > 0) result.push(DEPTH_HORIZON_METRES * record[14]! / record[13]!);
    return result;
  });
  const swatchWidth = 128;
  const swatchHeight = 40;
  const raw = Buffer.alloc(swatchWidth * swatchHeight * 3);
  for (let pixel = 0; pixel < swatchWidth * swatchHeight; pixel++) {
    const record = all[pixel % all.length]!;
    const alpha = record[4]! + record[13]!;
    for (let channel = 0; channel < 3; channel++) {
      const premul = record[7 + channel]! + record[16 + channel]!;
      raw[pixel * 3 + channel] = Math.round(Math.max(0, Math.min(1, premul + (1 - alpha) * 0.12)) * 255);
    }
  }
  const qaPath = resolve(qaRoot, '01-held-domain-truth-swatches.png');
  await sharp(raw, { raw: { width: swatchWidth, height: swatchHeight, channels: 3 } })
    .resize(1024, 320, { kernel: 'nearest' }).png().toFile(qaPath);
  const manifest = {
    schema: 'laas-gbc2-k6-shared-origin-cap-truth/v1',
    created: new Date().toISOString(),
    recipe,
    source: {
      path: relative(WORKSPACE, SOURCE), sha256: sourceSha,
      vertices: geometry.vertexCount, triangles: geometry.triangleCount,
      bounds: geometry.bounds, tileSize: [geometry.tileSizeX, geometry.tileSizeZ],
    },
    accelerator: bvh.metrics,
    files: {
      train: { path: relative(WORKSPACE, trainPath), sha256: sha256(readFileSync(trainPath)), records: TRAIN_RECORDS },
      validation: { path: relative(WORKSPACE, validationPath), sha256: sha256(readFileSync(validationPath)), records: VALIDATION_RECORDS },
      qa: { path: relative(WORKSPACE, qaPath), sha256: sha256(readFileSync(qaPath)) },
    },
    observed: { totalCoverage: quantiles(coverages), conditionalDepthMetres: quantiles(depths) },
    exclusions: [
      'This bounded terminal covers the top-cap chart from 5 through 90 degrees elevation only.',
      'The exact-horizontal/side chart and additional standoff scale levels are not represented by this truth set.',
      'A green fit validates only this production pixel level and terminal, not the complete exterior asset.',
    ],
    elapsedMilliseconds: performance.now() - started,
  };
  const manifestPath = resolve(outputRoot, 'manifest.json');
  writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(JSON.stringify({
    outputRoot: relative(WORKSPACE, outputRoot),
    manifest: relative(WORKSPACE, manifestPath),
    observed: manifest.observed,
    elapsedMilliseconds: manifest.elapsedMilliseconds,
  }, null, 2));
}

await main();
