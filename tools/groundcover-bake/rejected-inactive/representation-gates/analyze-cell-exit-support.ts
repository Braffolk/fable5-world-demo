import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

interface Args { [key: string]: string | boolean }
interface TruthManifest {
  source: {
    sha256: string;
    tileSize: [number, number];
    bounds: { min: [number, number, number]; max: [number, number, number] };
  };
  record: { floats: number; fields: string[] };
  files: Record<string, { path: string; records: number; hits: number; sha256: string }>;
}

interface Count {
  rays: number;
  hits: number;
  localHits: number;
  misses: number;
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

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function ratio(numerator: number, denominator: number): number {
  return denominator > 0 ? numerator / denominator : 0;
}

const args = parseArgs(process.argv.slice(2));
const manifestPath = resolve(typeof args.manifest === 'string'
  ? args.manifest
  : 'data/work/groundcover-origin-aware-codec/2ed57f59d86e8376/5c69afaee4f55926/truth/manifest.json');
const split = typeof args.split === 'string' ? args.split : 'validation';
const toleranceMetres = Number(typeof args.tolerance === 'string' ? args.tolerance : 1e-5);
if (!(toleranceMetres >= 0) || !Number.isFinite(toleranceMetres)) {
  throw new Error('tolerance must be finite and nonnegative');
}

const manifestBytes = readFileSync(manifestPath);
const manifest = JSON.parse(manifestBytes.toString('utf8')) as TruthManifest;
const splitRecord = manifest.files[split];
if (!splitRecord) throw new Error(`manifest has no split ${split}`);
if (manifest.record.floats !== 8 || manifest.record.fields.join(',')
  !== 'phaseX,heightFraction,phaseZ,directionX,directionY,directionZ,hit,distanceMetres') {
  throw new Error('unexpected truth record schema');
}
const truthPath = resolve(dirname(manifestPath), splitRecord.path);
const truthBytes = readFileSync(truthPath);
const truthSha256 = sha256(truthBytes);
if (truthSha256 !== splitRecord.sha256) throw new Error('truth split SHA-256 mismatch');
if (truthBytes.byteLength !== splitRecord.records * manifest.record.floats * 4) {
  throw new Error('truth split byte length mismatch');
}

const truth = new Float32Array(
  truthBytes.buffer,
  truthBytes.byteOffset,
  truthBytes.byteLength / Float32Array.BYTES_PER_ELEMENT,
);
const tileX = manifest.source.tileSize[0];
const tileZ = manifest.source.tileSize[1];
const height = manifest.source.bounds.max[1] - manifest.source.bounds.min[1];
const byElevation = new Map<number, Count>();
const all: Count = { rays: 0, hits: 0, localHits: 0, misses: 0 };

for (let offset = 0; offset < truth.length; offset += manifest.record.floats) {
  const phaseX = truth[offset]!;
  const heightFraction = truth[offset + 1]!;
  const phaseZ = truth[offset + 2]!;
  const dx = truth[offset + 3]!;
  const dy = truth[offset + 4]!;
  const dz = truth[offset + 5]!;
  const hit = truth[offset + 6]! > 0.5;
  const distance = truth[offset + 7]!;
  const exitX = dx > 0 ? (1 - phaseX) * tileX / dx
    : dx < 0 ? -phaseX * tileX / dx : Number.POSITIVE_INFINITY;
  const exitY = dy > 0 ? (1 - heightFraction) * height / dy
    : dy < 0 ? -heightFraction * height / dy : Number.POSITIVE_INFINITY;
  const exitZ = dz > 0 ? (1 - phaseZ) * tileZ / dz
    : dz < 0 ? -phaseZ * tileZ / dz : Number.POSITIVE_INFINITY;
  const exit = Math.min(exitX, exitY, exitZ);
  const localHit = hit && distance <= exit + toleranceMetres;
  const elevation = Math.round(Math.asin(Math.min(1, Math.abs(dy))) * 180 / Math.PI * 10) / 10;
  const row = byElevation.get(elevation) ?? { rays: 0, hits: 0, localHits: 0, misses: 0 };
  for (const target of [row, all]) {
    target.rays++;
    if (hit) target.hits++;
    else target.misses++;
    if (localHit) target.localHits++;
  }
  byElevation.set(elevation, row);
}

if (all.rays !== splitRecord.records || all.hits !== splitRecord.hits) {
  throw new Error('truth count mismatch');
}

const configuration = {
  sourceSha256: manifest.source.sha256,
  manifestSha256: sha256(manifestBytes),
  truthSha256,
  split,
  toleranceMetres,
  classifier: 'hit distance <= first canonical X/Y/Z cell-face exit + tolerance',
};
const recipeSha256 = createHash('sha256').update(JSON.stringify(configuration)).digest('hex');
const rows = [...byElevation].sort(([a], [b]) => a - b).map(([elevationDegrees, count]) => ({
  elevationDegrees,
  ...count,
  localFractionAllRays: ratio(count.localHits, count.rays),
  localFractionHits: ratio(count.localHits, count.hits),
}));
const report = {
  schema: 'laas-groundcover-cell-exit-support/v1',
  configuration,
  recipeSha256,
  source: {
    tileSize: manifest.source.tileSize,
    bounds: manifest.source.bounds,
  },
  rows,
  all: {
    ...all,
    localFractionAllRays: ratio(all.localHits, all.rays),
    localFractionHits: ratio(all.localHits, all.hits),
  },
  decision: 'reject-primary-codec',
  reason: 'the local field resolves too little of the low-oblique workload; the unsolved long-range boundary field remains dominant',
};

const outputPath = resolve(typeof args.output === 'string'
  ? args.output
  : `data/work/groundcover-cell-exit-support/${manifest.source.sha256.slice(0, 16)}/${recipeSha256.slice(0, 16)}/report.json`);
mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(outputPath);

