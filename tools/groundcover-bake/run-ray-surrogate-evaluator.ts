import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { runRaySurrogateEvaluation } from './RaySurrogateEvaluator';

interface Args {
  [key: string]: string | boolean;
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
    } else {
      result[argument.slice(2)] = true;
    }
  }
  return result;
}

function stringArg(value: string | boolean | undefined, fallback: string): string {
  return typeof value === 'string' ? value : fallback;
}

function numberArg(value: string | boolean | undefined, fallback: number, label: string): number {
  const parsed = Number(typeof value === 'string' ? value : fallback);
  if (!Number.isFinite(parsed)) throw new Error(`${label} is not finite`);
  return parsed;
}

function numberList(value: string | boolean | undefined, fallback: readonly number[], label: string): number[] {
  if (typeof value !== 'string') return [...fallback];
  const parsed = value.split(',').map(Number);
  if (parsed.length === 0 || parsed.some((entry) => !Number.isFinite(entry))) {
    throw new Error(`${label} must be a comma-separated finite number list`);
  }
  return parsed;
}

const args = parseArgs(process.argv.slice(2));
const source = resolve(stringArg(args.source, 'src/assets/groundcover/calamagrostis-canescens.gcrp'));
const sourceBytes = readFileSync(source);
const sourceSha256 = createHash('sha256').update(sourceBytes).digest('hex');
const options = {
  phaseGrid: numberArg(args['phase-grid'], 8, 'phase grid'),
  phaseOffset: numberArg(args['phase-offset'], 0.3819660112501051, 'phase offset'),
  azimuthCount: numberArg(args['azimuth-count'], 16, 'azimuth count'),
  azimuthOffset: numberArg(args['azimuth-offset'], 0.5, 'azimuth offset'),
  elevationsDegrees: numberList(
    args.elevations,
    [5, 7.5, 10, 12.5, 15, 20, 25, 35, 45, 55, 65, 75, 82.5],
    'elevations',
  ),
  planePoleThreshold: numberArg(args['plane-pole'], 1e-4, 'plane pole threshold'),
  coverageThreshold: numberArg(args.coverage, 0.02, 'coverage threshold'),
  denserPhaseGrid: numberArg(args['denser-phase-grid'], 4, 'denser phase grid'),
};
console.error(`[ray-surrogate] decoding ${sourceBytes.byteLength.toLocaleString()} bytes from ${source}`);
console.error(
  `[ray-surrogate] tracing ${options.phaseGrid}x${options.phaseGrid} phases, `
  + `${options.azimuthCount} azimuths, ${options.elevationsDegrees.length} elevations`,
);
const evaluation = runRaySurrogateEvaluation(sourceBytes, options);
const report = {
  schema: 'laas-groundcover-ray-surrogate-evaluation/v2',
  source: {
    file: source,
    bytes: sourceBytes.byteLength,
    sha256: sourceSha256,
  },
  ...evaluation,
};
const serialized = `${JSON.stringify(report, null, 2)}\n`;
const output = typeof args.output === 'string' ? resolve(args.output) : null;
if (output) {
  writeFileSync(output, serialized);
  console.error(`[ray-surrogate] wrote ${output}`);
}
process.stdout.write(serialized);
