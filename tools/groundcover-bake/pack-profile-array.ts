import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { packPeriodicProfileArray } from './ProfileArrayFormat';

function usage(): never {
  throw new Error(
    'usage: npx tsx tools/groundcover-bake/pack-profile-array.ts --out <array.gcar> --expected <id,id,...> <profile.gcrp>...',
  );
}

const args = process.argv.slice(2);
const outIndex = args.indexOf('--out');
const expectedIndex = args.indexOf('--expected');
if (outIndex < 0 || !args[outIndex + 1] || expectedIndex < 0 || !args[expectedIndex + 1]) usage();
const output = resolve(args[outIndex + 1]!);
const expectedProfileIds = args[expectedIndex + 1]!.split(',').map((value) => Number(value));
if (expectedProfileIds.some((id) => !Number.isInteger(id) || id < 0)) usage();
const inputs = args.filter((_value, index) =>
  index !== outIndex
  && index !== outIndex + 1
  && index !== expectedIndex
  && index !== expectedIndex + 1,
);
if (inputs.length === 0) usage();
const sources = inputs.map((path) => new Uint8Array(readFileSync(resolve(path))));
const packed = packPeriodicProfileArray(sources, { expectedProfileIds });
mkdirSync(dirname(output), { recursive: true });
writeFileSync(output, packed.bytes);
// eslint-disable-next-line no-console
console.log(JSON.stringify({
  output,
  sha256: packed.sha256,
  profileIds: packed.profiles.map((profile) => profile.profileId),
  sourceSha256: packed.profiles.map((profile) => profile.sourceSha256),
  dimensions: [packed.header.layerWidth, packed.header.layerHeight, packed.header.profileCount],
  bytes: packed.bytes.byteLength,
}, null, 2));
