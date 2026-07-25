/** Exact-BVH positive-measure pages used by the Candidate-O offline gate. */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { TriangleBvh, decodeOwnedProfileGeometry } from '../../ExteriorClosureCensus';
import { renderPointPage, sourceAttributes, toroidalBoxFilter, type DirectionSpec } from './gate-candidate-k-heldout-fe';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');

function args(): Record<string, string> {
  const result: Record<string, string> = {};
  for (let i = 2; i < process.argv.length; i += 2) result[process.argv[i]!.replace(/^--/, '')] = process.argv[i + 1]!;
  return result;
}

const options = args();
const output = resolve(options.output!);
const config = JSON.parse(readFileSync(resolve(options.config!), 'utf8')) as {
  source: string;
  resolution: number;
  referenceY: number;
  radii: number[];
  entries: { key: string; elevationDegrees: number; azimuthDegrees: number; phaseOffsetX: number; phaseOffsetZ: number }[];
};
const bytes = readFileSync(resolve(ROOT, config.source));
const geometry = decodeOwnedProfileGeometry(bytes);
const bvh = TriangleBvh.build(geometry);
const attributes = sourceAttributes(bytes);
mkdirSync(output, { recursive: true });
for (let index = 0; index < config.entries.length; index++) {
  const entry = config.entries[index]!;
  const spec: DirectionSpec = { key: entry.key, elevationDegrees: entry.elevationDegrees, azimuthDegrees: entry.azimuthDegrees, kind: 'heldout' };
  const point = renderPointPage(geometry, bvh, attributes, config.resolution, config.referenceY, spec, entry.phaseOffsetX, entry.phaseOffsetZ);
  for (const radius of config.radii) {
    const filtered = toroidalBoxFilter(point, config.resolution, radius);
    writeFileSync(resolve(output, `${entry.key}-r${radius}.f32`), Buffer.from(filtered.rgba.buffer));
  }
  console.log(`[candidate-o-truth] ${index + 1}/${config.entries.length} ${entry.key}`);
}
