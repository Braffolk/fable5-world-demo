/**
 * Enrich the immutable origin-aware successor truth with authored colour and
 * normal at each known analytic hit.  The expensive periodic first-hit query
 * is not repeated: a very short BVH window around the recorded hit recovers
 * the owning decoded-u16 triangle and barycentrics.  Offline analysis only.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  TriangleBvh,
  decodeOwnedProfileGeometry,
  type CensusVec3,
  type DecodedOwnedProfileGeometry,
  type TriangleRayHit,
} from '../../ExteriorClosureCensus';

const INPUT_FIELDS = 8;
const OUTPUT_FIELDS = 14;
const EXPECTED_SOURCE_SHA256 =
  '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const COPY_EPSILON = 1e-9;

interface Args { [key: string]: string | boolean }

interface VertexAttributes {
  color: readonly [number, number, number];
  normal: readonly [number, number, number];
}

interface RecoveredHit extends TriangleRayHit {
  copyX: number;
  copyZ: number;
  recoveredT: number;
  error: number;
}

function args(argv: readonly string[]): Args {
  const result: Args = {};
  for (let index = 0; index < argv.length; index++) {
    const item = argv[index]!;
    if (!item.startsWith('--')) continue;
    const next = argv[index + 1];
    if (next !== undefined && !next.startsWith('--')) {
      result[item.slice(2)] = next;
      index++;
    } else result[item.slice(2)] = true;
  }
  return result;
}

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function sha256File(path: string): string {
  return sha256(readFileSync(path));
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
  const length = Math.hypot(nx, ny, nz);
  return [nx / length, ny / length, nz / length];
}

function vertexAttributes(bytes: Uint8Array, vertex: number): VertexAttributes {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
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

function interpolateAttributes(
  sourceBytes: Uint8Array,
  geometry: DecodedOwnedProfileGeometry,
  hit: RecoveredHit,
): VertexAttributes {
  const triangle = hit.triangleId * 3;
  const a = vertexAttributes(sourceBytes, geometry.triangles[triangle]!);
  const b = vertexAttributes(sourceBytes, geometry.triangles[triangle + 1]!);
  const c = vertexAttributes(sourceBytes, geometry.triangles[triangle + 2]!);
  const weights = [1 - hit.u - hit.v, hit.u, hit.v] as const;
  const vertices = [a, b, c] as const;
  const color: [number, number, number] = [0, 0, 0];
  const normal: [number, number, number] = [0, 0, 0];
  for (let corner = 0; corner < 3; corner++) {
    const weight = Math.max(0, weights[corner]!);
    for (let component = 0; component < 3; component++) {
      color[component] += vertices[corner]!.color[component]! * weight;
      normal[component] += vertices[corner]!.normal[component]! * weight;
    }
  }
  const weightSum = Math.max(1e-20, weights.reduce((sum, value) => sum + Math.max(0, value), 0));
  for (let component = 0; component < 3; component++) color[component] /= weightSum;
  const normalLength = Math.hypot(...normal);
  if (normalLength > 1e-20) for (let component = 0; component < 3; component++) normal[component] /= normalLength;
  else normal[1] = 1;
  return { color, normal };
}

function recoverHit(
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  origin: CensusVec3,
  direction: CensusVec3,
  knownT: number,
): RecoveredHit | null {
  // The recorded distance is float32.  The window therefore scales with T and
  // remains much wider than its ulp while still being sub-millimetric at 155 m.
  const radius = Math.max(5e-5, Math.abs(knownT) * 2e-6);
  const startT = Math.max(0, knownT - radius);
  const length = knownT + radius - startT;
  const start: CensusVec3 = [
    origin[0] + direction[0] * startT,
    origin[1] + direction[1] * startT,
    origin[2] + direction[2] * startT,
  ];
  const end: CensusVec3 = [
    start[0] + direction[0] * length,
    start[1] + direction[1] * length,
    start[2] + direction[2] * length,
  ];
  const minimumX = Math.min(start[0], end[0]);
  const maximumX = Math.max(start[0], end[0]);
  const minimumZ = Math.min(start[2], end[2]);
  const maximumZ = Math.max(start[2], end[2]);
  const minCopyX = Math.ceil((minimumX - geometry.bounds.max[0]) / geometry.tileSizeX - COPY_EPSILON);
  const maxCopyX = Math.floor((maximumX - geometry.bounds.min[0]) / geometry.tileSizeX + COPY_EPSILON);
  const minCopyZ = Math.ceil((minimumZ - geometry.bounds.max[2]) / geometry.tileSizeZ - COPY_EPSILON);
  const maxCopyZ = Math.floor((maximumZ - geometry.bounds.min[2]) / geometry.tileSizeZ + COPY_EPSILON);
  let best: RecoveredHit | null = null;
  for (let copyZ = minCopyZ; copyZ <= maxCopyZ; copyZ++) {
    for (let copyX = minCopyX; copyX <= maxCopyX; copyX++) {
      const localStart: CensusVec3 = [
        start[0] - copyX * geometry.tileSizeX,
        start[1],
        start[2] - copyZ * geometry.tileSizeZ,
      ];
      bvh.intersectAll(localStart, direction, 0, length, (hit) => {
        const recoveredT = startT + hit.t;
        const error = Math.abs(recoveredT - knownT);
        if (
          best === null
          || error < best.error
          || (error === best.error && (copyX < best.copyX
            || (copyX === best.copyX && (copyZ < best.copyZ
              || (copyZ === best.copyZ && hit.triangleId < best.triangleId)))))
        ) best = { ...hit, copyX, copyZ, recoveredT, error };
      });
    }
  }
  return best;
}

const cli = args(process.argv.slice(2));
if (typeof cli.manifest !== 'string') throw new Error('--manifest is required');
const inputManifestPath = resolve(cli.manifest);
const inputManifestBytes = readFileSync(inputManifestPath);
const inputManifest = JSON.parse(inputManifestBytes.toString('utf8'));
const sourcePath = resolve(inputManifest.source.file);
const sourceBytes = readFileSync(sourcePath);
const sourceHash = sha256(sourceBytes);
if (sourceHash !== EXPECTED_SOURCE_SHA256 || sourceHash !== inputManifest.source.sha256) {
  throw new Error(`unexpected source hash ${sourceHash}`);
}
const toolPath = fileURLToPath(import.meta.url);
const configuration = {
  schema: 'laas-groundcover-transfer-field-enrichment/v1',
  inputManifestSha256: sha256(inputManifestBytes),
  sourceSha256: sourceHash,
  recoveryWindow: 'max(5e-5 m, abs(recordedT)*2e-6) around float32 distance',
  attributes: 'barycentric decoded-u16 vertex linear-sRGB colour and oct16 authored normal',
  toolSha256: sha256File(toolPath),
};
const recipeHash = sha256(Buffer.from(JSON.stringify(configuration)));
const outputDirectory = resolve(typeof cli.output === 'string'
  ? cli.output
  : `data/work/groundcover-transfer-field/${sourceHash}/${recipeHash}/truth`);
mkdirSync(outputDirectory, { recursive: true });

console.error(`[transfer-truth] decoding ${(sourceBytes.byteLength / 1024 / 1024).toFixed(1)} MiB actual mesh`);
const geometry = decodeOwnedProfileGeometry(sourceBytes);
const bvh = TriangleBvh.build(geometry, 8);
const files: Record<string, unknown> = {};

for (const label of ['train', 'validation', 'qa'] as const) {
  const input = inputManifest.files[label];
  const inputPath = resolve(dirname(inputManifestPath), input.path);
  const inputBytes = readFileSync(inputPath);
  if (sha256(inputBytes) !== input.sha256) throw new Error(`${label} input hash mismatch`);
  const sourceRecords = new Float32Array(
    inputBytes.buffer,
    inputBytes.byteOffset,
    inputBytes.byteLength / 4,
  );
  if (sourceRecords.length !== input.records * INPUT_FIELDS) throw new Error(`${label} input length mismatch`);
  const target = new Float32Array(input.records * OUTPUT_FIELDS);
  let recovered = 0;
  let recoveryMisses = 0;
  let maximumRecoveryError = 0;
  console.error(`[transfer-truth] ${label}: ${input.records.toLocaleString()} records`);
  for (let record = 0; record < input.records; record++) {
    const sourceOffset = record * INPUT_FIELDS;
    const targetOffset = record * OUTPUT_FIELDS;
    for (let field = 0; field < INPUT_FIELDS; field++) target[targetOffset + field] = sourceRecords[sourceOffset + field]!;
    if (sourceRecords[sourceOffset + 6]! >= 0.5) {
      const x = sourceRecords[sourceOffset]!;
      const y = sourceRecords[sourceOffset + 1]!;
      const z = sourceRecords[sourceOffset + 2]!;
      const origin: CensusVec3 = [
        geometry.tileOriginX + x * geometry.tileSizeX,
        geometry.bounds.min[1] + y * (geometry.bounds.max[1] - geometry.bounds.min[1]),
        geometry.tileOriginZ + z * geometry.tileSizeZ,
      ];
      const direction: CensusVec3 = [
        sourceRecords[sourceOffset + 3]!,
        sourceRecords[sourceOffset + 4]!,
        sourceRecords[sourceOffset + 5]!,
      ];
      const knownT = sourceRecords[sourceOffset + 7]!;
      const hit = recoverHit(geometry, bvh, origin, direction, knownT);
      if (hit === null) recoveryMisses++;
      else {
        recovered++;
        maximumRecoveryError = Math.max(maximumRecoveryError, hit.error);
        const attributes = interpolateAttributes(sourceBytes, geometry, hit);
        target.set(attributes.color, targetOffset + 8);
        target.set(attributes.normal, targetOffset + 11);
      }
    }
    if ((record + 1) % 50_000 === 0) {
      console.error(`[transfer-truth] ${label}: ${(record + 1).toLocaleString()}/${input.records.toLocaleString()}`);
    }
  }
  // The immutable generator traced from f64 query coordinates but stored the
  // replay coordinates as f32.  A tiny number of silhouette hits can therefore
  // disappear when the f32 ray is replayed.  Preserve their exact hit/distance
  // supervision, leave the zero normal as an explicit invalid-attribute flag,
  // and reject the enrichment if this is more than a 0.01% tail.
  const maximumAllowedRecoveryMisses = Math.max(32, Math.ceil(input.hits * 1e-4));
  if (recoveryMisses > maximumAllowedRecoveryMisses) {
    throw new Error(`${label} failed to recover ${recoveryMisses} known hits (allowed ${maximumAllowedRecoveryMisses})`);
  }
  const targetBytes = Buffer.from(target.buffer, target.byteOffset, target.byteLength);
  const targetPath = resolve(outputDirectory, `${label}.f32`);
  writeFileSync(targetPath, targetBytes);
  files[label] = {
    path: relative(outputDirectory, targetPath),
    records: input.records,
    hits: input.hits,
    recovered,
    recoveryMisses,
    attributeValidFractionOfHits: recovered / input.hits,
    maximumRecoveryErrorMetres: maximumRecoveryError,
    bytes: targetBytes.byteLength,
    sha256: sha256(targetBytes),
    ...(label === 'qa' ? { grid: input.grid, slices: input.slices } : {}),
  };
}

const manifest = {
  schema: 'laas-groundcover-transfer-field-actual-mesh-truth/v1',
  source: {
    ...inputManifest.source,
    file: sourcePath,
    sha256: sourceHash,
  },
  inputTruth: {
    manifest: inputManifestPath,
    manifestSha256: sha256(inputManifestBytes),
    contract: inputManifest.truthContract,
    splitContract: inputManifest.splitContract,
    configuration: inputManifest.configuration,
  },
  configuration,
  recipeSha256: recipeHash,
  command: [process.execPath, ...process.execArgv, ...process.argv.slice(1)],
  record: {
    type: 'little-endian float32',
    floats: OUTPUT_FIELDS,
    fields: [
      'phaseX', 'heightFraction', 'phaseZ',
      'directionX', 'directionY', 'directionZ',
      'hit', 'distanceMetres',
      'linearColorR', 'linearColorG', 'linearColorB',
      'normalX', 'normalY', 'normalZ',
    ],
  },
  files,
  truthContract: [
    ...inputManifest.truthContract,
    'authored colour and normal come from the exact recovered decoded-u16 hit triangle and barycentrics',
    'recovery is a sub-millimetric offline BVH window around the immutable analytic successor distance, not a new approximate first-hit query',
  ],
};
const manifestPath = resolve(outputDirectory, 'manifest.json');
writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
console.error(`[transfer-truth] wrote ${manifestPath}`);
process.stdout.write(`${manifestPath}\n`);
