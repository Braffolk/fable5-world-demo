import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { TriangleBvh, decodeOwnedProfileGeometry, type CensusVec3 } from '../../ExteriorClosureCensus';
import { periodicNearestSuccessor } from '../../OriginAwareRayTruth';

interface Args { [key: string]: string | boolean }
interface DirectionClass {
  elevationDegrees: number;
  sense: 'horizontal' | 'downward' | 'upward';
}

const ELEVATIONS = [0, 0.1, 1, 5, 15, 35, 75, 90] as const;
const TRAIN_HEIGHTS = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] as const;
const VALIDATION_HEIGHTS = [0.01, 0.055, 0.1, 0.175, 0.25, 0.375, 0.5, 0.625, 0.75, 0.825, 0.9, 0.945, 0.99] as const;
const FLOATS_PER_RECORD = 8;

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

function hashUnit(seed: number, index: number): number {
  let value = (index ^ seed) >>> 0;
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d);
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b);
  return ((value ^ (value >>> 16)) >>> 0) / 0x1_0000_0000;
}

function directionClasses(): DirectionClass[] {
  const result: DirectionClass[] = [];
  for (const elevationDegrees of ELEVATIONS) {
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

function expectedRecords(
  heights: readonly number[],
  azimuthBins: number,
  samplesPerStratum: number,
): number {
  return directionClasses().reduce(
    (sum, entry) => sum + heights.length * (entry.elevationDegrees === 90 ? 1 : azimuthBins) * samplesPerStratum,
    0,
  );
}

const args = parseArgs(process.argv.slice(2));
const source = resolve(typeof args.source === 'string' ? args.source : 'src/assets/groundcover/calamagrostis-canescens.gcrp');
const sourceBytes = readFileSync(source);
const sourceSha256 = createHash('sha256').update(sourceBytes).digest('hex');
const horizon = numberArg(args.horizon, 155, 'horizon');
const originEpsilon = numberArg(args['origin-epsilon'], 1e-9, 'origin epsilon');
const trainAzimuthBins = numberArg(args['train-azimuth-bins'], 8, 'train azimuth bins');
const validationAzimuthBins = numberArg(args['validation-azimuth-bins'], 16, 'validation azimuth bins');
const trainSamplesPerStratum = numberArg(args['train-per-stratum'], 64, 'train samples per stratum');
const validationSamplesPerStratum = numberArg(args['validation-per-stratum'], 8, 'validation samples per stratum');
const qaGrid = numberArg(args['qa-grid'], 64, 'QA grid');
for (const [label, value] of [
  ['train azimuth bins', trainAzimuthBins],
  ['validation azimuth bins', validationAzimuthBins],
  ['train samples per stratum', trainSamplesPerStratum],
  ['validation samples per stratum', validationSamplesPerStratum],
  ['QA grid', qaGrid],
] as const) if (!Number.isInteger(value) || value <= 0) throw new Error(`${label} must be a positive integer`);

const configuration = {
  horizon,
  originEpsilon,
  elevationsDegrees: ELEVATIONS,
  trainHeights: TRAIN_HEIGHTS,
  validationHeights: VALIDATION_HEIGHTS,
  trainAzimuthBins,
  validationAzimuthBins,
  trainSamplesPerStratum,
  validationSamplesPerStratum,
  qaGrid,
  heightSampling: 'every fourth training sample is an exact named anchor; remaining training samples cover the full open Y interval',
};
const configurationSha256 = createHash('sha256').update(JSON.stringify(configuration)).digest('hex');
const outputDirectory = resolve(typeof args.output === 'string'
  ? args.output
  : `data/work/groundcover-origin-aware-codec/${sourceSha256.slice(0, 16)}/${configurationSha256.slice(0, 16)}/truth`);
mkdirSync(outputDirectory, { recursive: true });

console.error(`[origin-truth] decoding ${(sourceBytes.byteLength / 1024 / 1024).toFixed(1)} MiB actual mesh`);
const geometry = decodeOwnedProfileGeometry(sourceBytes);
const bvh = TriangleBvh.build(geometry, 8);

function generateSplit(
  name: 'train' | 'validation',
  heights: readonly number[],
  azimuthBins: number,
  samplesPerStratum: number,
  seed: number,
): { path: string; records: number; hits: number; sha256: string } {
  const count = expectedRecords(heights, azimuthBins, samplesPerStratum);
  const records = new Float32Array(count * FLOATS_PER_RECORD);
  let cursor = 0;
  let hits = 0;
  let sampleIndex = 0;
  const classes = directionClasses();
  console.error(`[origin-truth] ${name}: ${count.toLocaleString()} actual-mesh rays`);
  for (let classIndex = 0; classIndex < classes.length; classIndex++) {
    const definition = classes[classIndex]!;
    const bins = definition.elevationDegrees === 90 ? 1 : azimuthBins;
    for (let heightIndex = 0; heightIndex < heights.length; heightIndex++) {
      const heightAnchor = heights[heightIndex]!;
      for (let azimuthBin = 0; azimuthBin < bins; azimuthBin++) {
        for (let repetition = 0; repetition < samplesPerStratum; repetition++) {
          const key = sampleIndex + classIndex * 0x100000 + heightIndex * 0x10000 + azimuthBin * 0x100;
          const x = hashUnit(seed ^ 0x9e3779b9, key * 3 + 0);
          const z = hashUnit(seed ^ 0x85ebca6b, key * 3 + 1);
          const azimuthJitter = hashUnit(seed ^ 0xc2b2ae35, key * 3 + 2);
          const azimuth = definition.elevationDegrees === 90
            ? 0
            : (azimuthBin + azimuthJitter) / bins * Math.PI * 2;
          // Retain exact named heights every fourth sample. The remainder
          // cover all Y so held-out midpoint heights never fall in an
          // accidentally unsupervised slab of a dense feature texture.
          const y = name === 'validation' || repetition % 4 === 0
            ? heightAnchor
            : 0.001 + hashUnit(seed ^ 0x27d4eb2f, key) * 0.998;
          const rayDirection = direction(definition, azimuth);
          const origin: CensusVec3 = [
            geometry.tileOriginX + x * geometry.tileSizeX,
            geometry.bounds.min[1] + y * (geometry.bounds.max[1] - geometry.bounds.min[1]),
            geometry.tileOriginZ + z * geometry.tileSizeZ,
          ];
          const hit = periodicNearestSuccessor(geometry, bvh, origin, rayDirection, horizon, originEpsilon);
          records[cursor++] = x;
          records[cursor++] = y;
          records[cursor++] = z;
          records[cursor++] = rayDirection[0];
          records[cursor++] = rayDirection[1];
          records[cursor++] = rayDirection[2];
          records[cursor++] = hit ? 1 : 0;
          records[cursor++] = hit?.t ?? horizon;
          if (hit) hits++;
          sampleIndex++;
          if (sampleIndex % 50_000 === 0) console.error(`[origin-truth] ${name}: ${sampleIndex.toLocaleString()}/${count.toLocaleString()}`);
        }
      }
    }
  }
  if (cursor !== records.length) throw new Error(`${name} record count mismatch ${cursor}/${records.length}`);
  const bytes = Buffer.from(records.buffer, records.byteOffset, records.byteLength);
  const path = resolve(outputDirectory, `${name}.f32`);
  writeFileSync(path, bytes);
  return { path, records: count, hits, sha256: createHash('sha256').update(bytes).digest('hex') };
}

const train = generateSplit('train', TRAIN_HEIGHTS, trainAzimuthBins, trainSamplesPerStratum, 0x51a7f00d);
const validation = generateSplit(
  'validation',
  VALIDATION_HEIGHTS,
  validationAzimuthBins,
  validationSamplesPerStratum,
  0xbadc0ffe,
);

const qaDefinitions = [
  { id: '001-horizontal-y075-az000', elevationDegrees: 0, sense: 'horizontal' as const, height: 0.75, azimuthDegrees: 0 },
  { id: '002-grazing-down-y090-az045', elevationDegrees: 0.1, sense: 'downward' as const, height: 0.9, azimuthDegrees: 45 },
  { id: '003-five-down-y050-az090', elevationDegrees: 5, sense: 'downward' as const, height: 0.5, azimuthDegrees: 90 },
  { id: '004-thirtyfive-down-y050-az045', elevationDegrees: 35, sense: 'downward' as const, height: 0.5, azimuthDegrees: 45 },
  { id: '005-vertical-down-y090', elevationDegrees: 90, sense: 'downward' as const, height: 0.9, azimuthDegrees: 0 },
];
const qa = new Float32Array(qaDefinitions.length * qaGrid * qaGrid * FLOATS_PER_RECORD);
let qaCursor = 0;
let qaHits = 0;
console.error(`[origin-truth] QA: ${(qaDefinitions.length * qaGrid * qaGrid).toLocaleString()} actual-mesh rays`);
for (const definition of qaDefinitions) {
  const rayDirection = direction(definition, definition.azimuthDegrees * Math.PI / 180);
  for (let zIndex = 0; zIndex < qaGrid; zIndex++) {
    const z = (zIndex + 0.5) / qaGrid;
    for (let xIndex = 0; xIndex < qaGrid; xIndex++) {
      const x = (xIndex + 0.5) / qaGrid;
      const origin: CensusVec3 = [
        geometry.tileOriginX + x * geometry.tileSizeX,
        geometry.bounds.min[1] + definition.height * (geometry.bounds.max[1] - geometry.bounds.min[1]),
        geometry.tileOriginZ + z * geometry.tileSizeZ,
      ];
      const hit = periodicNearestSuccessor(geometry, bvh, origin, rayDirection, horizon, originEpsilon);
      qa[qaCursor++] = x;
      qa[qaCursor++] = definition.height;
      qa[qaCursor++] = z;
      qa[qaCursor++] = rayDirection[0];
      qa[qaCursor++] = rayDirection[1];
      qa[qaCursor++] = rayDirection[2];
      qa[qaCursor++] = hit ? 1 : 0;
      qa[qaCursor++] = hit?.t ?? horizon;
      if (hit) qaHits++;
    }
  }
}
const qaBytes = Buffer.from(qa.buffer, qa.byteOffset, qa.byteLength);
const qaPath = resolve(outputDirectory, 'qa.f32');
writeFileSync(qaPath, qaBytes);
const manifest = {
  schema: 'laas-origin-aware-actual-mesh-ray-truth/v1',
  source: {
    file: source,
    bytes: sourceBytes.byteLength,
    sha256: sourceSha256,
    profileId: geometry.profileId,
    tileSize: [geometry.tileSizeX, geometry.tileSizeZ],
    bounds: geometry.bounds,
  },
  command: [process.execPath, ...process.execArgv, ...process.argv.slice(1)],
  configuration,
  configurationSha256,
  record: {
    type: 'little-endian float32',
    floats: FLOATS_PER_RECORD,
    fields: ['phaseX', 'heightFraction', 'phaseZ', 'directionX', 'directionY', 'directionZ', 'hit', 'distanceMetres'],
  },
  splitContract: {
    train: 'seed 0x51a7f00d; stratified named elevation/height/azimuth bins; exact height anchors plus deterministic local height jitter',
    validation: 'disjoint seed 0xbadc0ffe; twice the azimuth strata and named plus held-out midpoint heights',
    qa: 'regular held-out phase grids for five declared origin/direction slices',
  },
  files: {
    train: { ...train, path: 'train.f32', hitFraction: train.hits / train.records },
    validation: { ...validation, path: 'validation.f32', hitFraction: validation.hits / validation.records },
    qa: {
      path: 'qa.f32',
      records: qa.length / FLOATS_PER_RECORD,
      hits: qaHits,
      hitFraction: qaHits / (qa.length / FLOATS_PER_RECORD),
      sha256: createHash('sha256').update(qaBytes).digest('hex'),
      grid: qaGrid,
      slices: qaDefinitions,
    },
  },
  truthContract: [
    'nearest analytic decoded GCRP/v4 mesh surface event in the forward interval (originEpsilon,horizon]',
    'periodic XZ copies are enumerated in tile order and exact BVH first-hit tested; there is no baked-atlas supervision',
    'the finite horizon is explicit; a miss means no surface event before that horizon or before exiting the finite Y envelope',
    'this is offline truth generation only and implies no runtime traversal, loop, march, or candidate set',
  ],
};
const manifestPath = resolve(outputDirectory, 'manifest.json');
writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
console.error(`[origin-truth] wrote ${manifestPath}`);
process.stdout.write(`${manifestPath}\n`);
