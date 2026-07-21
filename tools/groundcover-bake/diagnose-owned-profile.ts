import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parsePeriodicProfile } from '../../src/nanite/groundcover/GroundCoverProfiles';

const source = resolve(process.argv[2] ?? 'src/assets/groundcover/calamagrostis-canescens.gcrp');
const profile = parsePeriodicProfile(readFileSync(source));
if (
  profile.version !== 4
  || !profile.ownerTexels
  || !profile.vertexRecords
  || !profile.triangleRecords
  || !profile.sourceBounds
) throw new Error('diagnostic requires GCRP/v4');

const lo16 = (word: number): number => word & 0xffff;
const hi16 = (word: number): number => word >>> 16;
const unit16 = (word: number): number => word / 65535;
const bounds = profile.sourceBounds;
const vertex = (id: number, copyX: number, copyZ: number): readonly [number, number, number] => {
  const base = id * 4;
  const x = profile.vertexRecords![base]!;
  const y = profile.vertexRecords![base + 1]!;
  return [
    bounds[0] + unit16(lo16(x)) * (bounds[3] - bounds[0]) + copyX,
    bounds[1] + unit16(hi16(x)) * (bounds[4] - bounds[1]),
    bounds[2] + unit16(lo16(y)) * (bounds[5] - bounds[2]) + copyZ,
  ];
};

const sub = (a: readonly number[], b: readonly number[]): [number, number, number] =>
  [a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!];
const cross = (a: readonly number[], b: readonly number[]): [number, number, number] => [
  a[1]! * b[2]! - a[2]! * b[1]!,
  a[2]! * b[0]! - a[0]! * b[2]!,
  a[0]! * b[1]! - a[1]! * b[0]!,
];
const dot = (a: readonly number[], b: readonly number[]): number =>
  a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;

let checked = 0;
let failed = 0;
let maxDepthError = 0;
const atlasWidth = profile.storedTileWidth * profile.atlasColumns;
for (let sliceIndex = 0; sliceIndex < profile.slices.length; sliceIndex += 7) {
  const slice = profile.slices[sliceIndex]!;
  const tileColumn = sliceIndex % profile.atlasColumns;
  const tileRow = Math.floor(sliceIndex / profile.atlasColumns);
  for (let py = 5; py < profile.interiorTileHeight; py += 19) {
    for (let px = 3; px < profile.interiorTileWidth; px += 17) {
      const atlasX = tileColumn * profile.storedTileWidth + profile.gutter + px;
      const atlasY = tileRow * profile.storedTileHeight + profile.gutter + py;
      const texel = atlasY * atlasWidth + atlasX;
      const owner = profile.ownerTexels[texel]!;
      if (owner === 0xffff_ffff) continue;
      const triangleId = owner & 0x3f_ffff;
      const copyX = (((owner >>> 22) & 0x1f) - 16) * profile.tileSizeX;
      const copyZ = (((owner >>> 27) & 0x1f) - 16) * profile.tileSizeZ;
      const triangleBase = triangleId * 4;
      const a = vertex(profile.triangleRecords[triangleBase]!, copyX, copyZ);
      const b = vertex(profile.triangleRecords[triangleBase + 1]!, copyX, copyZ);
      const c = vertex(profile.triangleRecords[triangleBase + 2]!, copyX, copyZ);
      const origin: [number, number, number] = [
        profile.tileOriginX + (px + 0.5) / profile.interiorTileWidth * profile.tileSizeX,
        profile.topH,
        profile.tileOriginZ + (1 - (py + 0.5) / profile.interiorTileHeight) * profile.tileSizeZ,
      ];
      const direction = slice.direction;
      const e1 = sub(b, a);
      const e2 = sub(c, a);
      const pvec = cross(direction, e2);
      const determinant = dot(e1, pvec);
      const inverse = 1 / determinant;
      const originToA = sub(origin, a);
      const u = dot(originToA, pvec) * inverse;
      const qvec = cross(originToA, e1);
      const v = dot(direction, qvec) * inverse;
      const t = dot(e2, qvec) * inverse;
      const w = 1 - u - v;
      const storedDepth = profile.texels[texel * 4]! / 65535;
      const expectedT = slice.depthMin + storedDepth * (slice.depthMax - slice.depthMin);
      const depthError = Math.abs(t - expectedT);
      checked++;
      maxDepthError = Math.max(maxDepthError, depthError);
      if (
        Math.abs(determinant) <= 1e-8
        || u < -2e-4
        || v < -2e-4
        || w < -2e-4
        || t < 0
        || depthError > 5e-4
      ) failed++;
    }
  }
}

console.log(JSON.stringify({ source, checked, failed, maxDepthError }, null, 2));
if (checked === 0 || failed !== 0) process.exitCode = 1;
