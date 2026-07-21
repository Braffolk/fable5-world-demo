import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  TriangleBvh,
  decodeOwnedProfileGeometry,
  runExteriorClosureCensus,
} from './ExteriorClosureCensus';

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
console.error(`[closure-census] decoding ${sourceBytes.byteLength} bytes from ${source}`);
const geometry = decodeOwnedProfileGeometry(sourceBytes);
const leafSize = numberArg(args['leaf-size'], 8, 'leaf size');
console.error(`[closure-census] building BVH for ${geometry.triangleCount.toLocaleString()} triangles`);
const bvh = TriangleBvh.build(geometry, leafSize);
console.error(`[closure-census] BVH ${bvh.metrics.nodeCount.toLocaleString()} nodes in ${bvh.metrics.buildMilliseconds.toFixed(1)} ms`);
const options = {
  phaseGrid: numberArg(args['phase-grid'], 4, 'phase grid'),
  phaseOffsets: numberList(args['phase-offsets'], [0.5], 'phase offsets'),
  azimuthCount: numberArg(args['azimuth-count'], 16, 'azimuth count'),
  elevationsDegrees: numberList(args.elevations, [5, 15, 35, 55, 75, 90], 'elevations'),
  worstRayLimit: numberArg(args['worst-rays'], 8, 'worst-ray limit'),
};
console.error(`[closure-census] tracing deterministic ${options.phaseGrid}x${options.phaseGrid} phase grid`);
const census = runExteriorClosureCensus(geometry, bvh, options);
const report = {
  schema: 'laas-groundcover-exterior-closure-stage1-census/v1',
  source: {
    file: source,
    bytes: sourceBytes.byteLength,
    sha256: sourceSha256,
    profileId: geometry.profileId,
    vertices: geometry.vertexCount,
    triangles: geometry.triangleCount,
  },
  ...census,
};
const serialized = `${JSON.stringify(report, null, 2)}\n`;
const output = typeof args.output === 'string' ? resolve(args.output) : null;
if (output) {
  writeFileSync(output, serialized);
  console.error(`[closure-census] wrote ${output}`);
}
process.stdout.write(serialized);
