/**
 * Offline Candidate-H 16-entry palette gate for the accepted Calamagrostis
 * first-hit field. This file does not participate in runtime or shader code.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';
import {
  ESTONIAN_GRAMINOID_PROFILE_IDS,
  makeEstonianGraminoidFixture,
  type GraminoidPrimitiveRecipe,
} from '../../EstonianGraminoids';

const REPO_ROOT = resolve(import.meta.dirname, '../../../..');
const SOURCE_RELATIVE = 'data/work/groundcover-gpu-bake-estonian-graminoids-v4/2-calamagrostis-canescens/2ed57f59d86e8376/calamagrostis-canescens-periodic-profile.gcrp';
const SOURCE_MESH_SHA256 = '37b0cf1d33f632b5cde3dd60410ad987eec2e977661d0bc804dd6f2bee7054c0';
const TRANSPORT_INDICES_RELATIVE = `data/work/groundcover-bake-transport/${SOURCE_MESH_SHA256}/indices.u32`;
const MISS_OWNER = 0xffff_ffff;
const MISS_COLOR = 0xffff_ffff;
const PALETTE_SIZE = 16;
const ERROR_THRESHOLD = 5;

const GATES = Object.freeze({
  overallP95: 2.5,
  overallP99: 5.0,
  plumeP95: 3.0,
  structureP95: 3.0,
  maxConnectedBadFractionPerSlice: 0.01,
});

type Semantic = 0 | 1; // 0 structure, 1 plume
type Lab = readonly [number, number, number];

interface Slice {
  direction: readonly [number, number, number];
  depthMin: number;
  depthMax: number;
}

interface BoundProfile {
  bytes: Buffer;
  view: DataView;
  storedWidth: number;
  storedHeight: number;
  interiorWidth: number;
  interiorHeight: number;
  columns: number;
  rows: number;
  sliceCount: number;
  gutter: number;
  atlasWidth: number;
  atlasHeight: number;
  payloadOffset: number;
  ownerOffset: number;
  vertexOffset: number;
  triangleOffset: number;
  vertexCount: number;
  triangleCount: number;
  topH: number;
  originX: number;
  originZ: number;
  sizeX: number;
  sizeZ: number;
  bounds: readonly [number, number, number, number, number, number];
  slices: readonly Slice[];
}

interface ColorBin {
  key: number;
  rgb: readonly [number, number, number];
  lab: Lab;
  total: number;
  structure: number;
  plume: number;
}

function sha256(value: Uint8Array | string): string {
  return createHash('sha256').update(value).digest('hex');
}

function parseProfile(path: string): BoundProfile {
  const bytes = readFileSync(path);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.toString('ascii', 0, 4) !== 'GCRP' || view.getUint32(4, true) !== 4) {
    throw new Error('Candidate-H palette gate requires GCRP/v4');
  }
  const storedWidth = view.getUint32(12, true);
  const storedHeight = view.getUint32(16, true);
  const columns = view.getUint32(20, true);
  const rows = view.getUint32(24, true);
  const sliceCount = view.getUint32(28, true);
  const payloadOffset = view.getUint32(36, true);
  const interiorWidth = view.getUint32(44, true);
  const interiorHeight = view.getUint32(48, true);
  const gutter = view.getUint32(52, true);
  const ownerOffset = view.getUint32(80, true);
  const vertexOffset = view.getUint32(84, true);
  const triangleOffset = view.getUint32(88, true);
  const vertexCount = view.getUint32(92, true);
  const triangleCount = view.getUint32(96, true);
  if (view.getUint32(32, true) !== 8 || view.getUint32(76, true) !== 128) {
    throw new Error('accepted GCRP/v4 has a non-canonical layout');
  }
  if (storedWidth !== interiorWidth + 2 * gutter || storedHeight !== interiorHeight + 2 * gutter) {
    throw new Error('accepted GCRP/v4 gutter geometry is inconsistent');
  }
  const atlasWidth = storedWidth * columns;
  const atlasHeight = storedHeight * rows;
  if (ownerOffset !== payloadOffset + atlasWidth * atlasHeight * 8) {
    throw new Error('accepted GCRP/v4 owner section is not canonical');
  }
  const slices: Slice[] = [];
  for (let index = 0; index < sliceCount; index++) {
    const base = 128 + index * 64;
    slices.push({
      direction: [
        view.getFloat32(base, true),
        view.getFloat32(base + 4, true),
        view.getFloat32(base + 8, true),
      ],
      depthMin: view.getFloat32(base + 12, true),
      depthMax: view.getFloat32(base + 16, true),
    });
  }
  return {
    bytes,
    view,
    storedWidth,
    storedHeight,
    interiorWidth,
    interiorHeight,
    columns,
    rows,
    sliceCount,
    gutter,
    atlasWidth,
    atlasHeight,
    payloadOffset,
    ownerOffset,
    vertexOffset,
    triangleOffset,
    vertexCount,
    triangleCount,
    topH: view.getFloat32(56, true),
    originX: view.getFloat32(60, true),
    originZ: view.getFloat32(64, true),
    sizeX: view.getFloat32(68, true),
    sizeZ: view.getFloat32(72, true),
    bounds: [100, 104, 108, 112, 116, 120].map((offset) => view.getFloat32(offset, true)) as unknown as BoundProfile['bounds'],
    slices,
  };
}

function isPlume(recipe: GraminoidPrimitiveRecipe): boolean {
  if (recipe.kind === 'hair-filament' || recipe.kind === 'cotton-bristle') return true;
  if (recipe.kind === 'lanceolate-surface') {
    return recipe.role === 'glume' || recipe.role === 'lemma' || recipe.role === 'anther';
  }
  if (recipe.kind === 'tube' || recipe.kind === 'rhizome') {
    return recipe.role === 'panicle-axis'
      || recipe.role === 'spikelet-axis'
      || recipe.role === 'anther-filament';
  }
  return false;
}

function bindTriangleSemantics(profile: BoundProfile): Uint8Array {
  const fixture = makeEstonianGraminoidFixture(ESTONIAN_GRAMINOID_PROFILE_IDS.CALAMAGROSTIS_CANESCENS);
  if (fixture.mesh.indices.length / 3 !== profile.triangleCount) {
    throw new Error(`fixture/GCRP triangle mismatch: ${fixture.mesh.indices.length / 3} != ${profile.triangleCount}`);
  }
  if (fixture.mesh.positions.length / 3 !== profile.vertexCount) {
    throw new Error(`fixture/GCRP vertex mismatch: ${fixture.mesh.positions.length / 3} != ${profile.vertexCount}`);
  }
  const transportBytes = readFileSync(resolve(REPO_ROOT, TRANSPORT_INDICES_RELATIVE));
  const transport = new Uint32Array(
    transportBytes.buffer,
    transportBytes.byteOffset,
    transportBytes.byteLength / 4,
  );
  if (transport.length !== fixture.mesh.indices.length) throw new Error('accepted transport index count changed');
  for (let index = 0; index < transport.length; index++) {
    const expected = fixture.mesh.indices[index]!;
    if (transport[index] !== expected) throw new Error(`fixture/accepted transport topology differs at index ${index}`);
    const triangleRecord = profile.triangleOffset + index * 4 + Math.floor(index / 3) * 4;
    if (profile.view.getUint32(triangleRecord, true) !== expected) {
      throw new Error(`GCRP/accepted transport topology differs at index ${index}`);
    }
  }
  const semantic = new Uint8Array(profile.triangleCount);
  let cursor = 0;
  for (const recipe of fixture.primitiveRecipes) {
    if (recipe.sourceTriangleStart !== cursor) {
      throw new Error(`primitive semantic ranges leave a gap at triangle ${cursor}`);
    }
    const value: Semantic = isPlume(recipe) ? 1 : 0;
    semantic.fill(value, recipe.sourceTriangleStart, recipe.sourceTriangleStart + recipe.sourceTriangleCount);
    cursor += recipe.sourceTriangleCount;
  }
  if (cursor !== profile.triangleCount) throw new Error('primitive semantic ranges do not cover the source');
  // Release the very large authoring mesh before decoding the atlas. The
  // semantic sidecar above is the only fixture data retained by this gate.
  fixture.mesh.positions.length = 0;
  fixture.mesh.normals.length = 0;
  fixture.mesh.colors!.length = 0;
  fixture.mesh.indices.length = 0;
  return semantic;
}

function linearRgbToLab(rgb: readonly [number, number, number]): Lab {
  const [r, g, b] = rgb;
  const x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
  const y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
  const z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
  const delta = 6 / 29;
  const f = (value: number): number => value > delta ** 3
    ? Math.cbrt(value)
    : value / (3 * delta * delta) + 4 / 29;
  const fx = f(x / 0.95047);
  const fy = f(y);
  const fz = f(z / 1.08883);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

function deltaE00(a: Lab, b: Lab): number {
  const [l1, a1, b1] = a;
  const [l2, a2, b2] = b;
  const c1 = Math.hypot(a1, b1);
  const c2 = Math.hypot(a2, b2);
  const cBar = (c1 + c2) * 0.5;
  const c7 = cBar ** 7;
  const g = 0.5 * (1 - Math.sqrt(c7 / (c7 + 25 ** 7)));
  const ap1 = (1 + g) * a1;
  const ap2 = (1 + g) * a2;
  const cp1 = Math.hypot(ap1, b1);
  const cp2 = Math.hypot(ap2, b2);
  const hp = (x: number, y: number): number => {
    if (x === 0 && y === 0) return 0;
    const angle = Math.atan2(y, x) * 180 / Math.PI;
    return angle < 0 ? angle + 360 : angle;
  };
  const hp1 = hp(ap1, b1);
  const hp2 = hp(ap2, b2);
  const dlp = l2 - l1;
  const dcp = cp2 - cp1;
  let dhp = 0;
  if (cp1 * cp2 !== 0) {
    dhp = hp2 - hp1;
    if (dhp > 180) dhp -= 360;
    else if (dhp < -180) dhp += 360;
  }
  const dh = 2 * Math.sqrt(cp1 * cp2) * Math.sin((dhp * Math.PI / 180) * 0.5);
  const lBar = (l1 + l2) * 0.5;
  const cpBar = (cp1 + cp2) * 0.5;
  let hpBar = hp1 + hp2;
  if (cp1 * cp2 === 0) hpBar = hp1 + hp2;
  else if (Math.abs(hp1 - hp2) <= 180) hpBar *= 0.5;
  else if (hp1 + hp2 < 360) hpBar = (hp1 + hp2 + 360) * 0.5;
  else hpBar = (hp1 + hp2 - 360) * 0.5;
  const t = 1
    - 0.17 * Math.cos((hpBar - 30) * Math.PI / 180)
    + 0.24 * Math.cos(2 * hpBar * Math.PI / 180)
    + 0.32 * Math.cos((3 * hpBar + 6) * Math.PI / 180)
    - 0.20 * Math.cos((4 * hpBar - 63) * Math.PI / 180);
  const dTheta = 30 * Math.exp(-(((hpBar - 275) / 25) ** 2));
  const rc = 2 * Math.sqrt(cpBar ** 7 / (cpBar ** 7 + 25 ** 7));
  const sl = 1 + 0.015 * ((lBar - 50) ** 2) / Math.sqrt(20 + ((lBar - 50) ** 2));
  const sc = 1 + 0.045 * cpBar;
  const sh = 1 + 0.015 * cpBar * t;
  const rt = -Math.sin(2 * dTheta * Math.PI / 180) * rc;
  const x = dlp / sl;
  const y = dcp / sc;
  const z = dh / sh;
  return Math.sqrt(Math.max(0, x * x + y * y + z * z + rt * y * z));
}

function unpackRgb(key: number): readonly [number, number, number] {
  return [((key >>> 16) & 255) / 255, ((key >>> 8) & 255) / 255, (key & 255) / 255];
}

function decodeColorKey(
  profile: BoundProfile,
  sliceIndex: number,
  px: number,
  py: number,
  texel: number,
  owner: number,
): number {
  const triangleId = owner & 0x3f_ffff;
  const copyX = (((owner >>> 22) & 0x1f) - 16) * profile.sizeX;
  const copyZ = (((owner >>> 27) & 0x1f) - 16) * profile.sizeZ;
  const slice = profile.slices[sliceIndex]!;
  const payload = profile.payloadOffset + texel * 8;
  const depth01 = profile.view.getUint16(payload, true) / 65535;
  const t = slice.depthMin + depth01 * (slice.depthMax - slice.depthMin);
  const ox = profile.originX + (px + 0.5) / profile.interiorWidth * profile.sizeX;
  const oz = profile.originZ + (1 - (py + 0.5) / profile.interiorHeight) * profile.sizeZ;
  const hx = ox + slice.direction[0] * t;
  const hy = profile.topH + slice.direction[1] * t;
  const hz = oz + slice.direction[2] * t;
  const tri = profile.triangleOffset + triangleId * 16;
  const vertexIndices = [
    profile.view.getUint32(tri, true),
    profile.view.getUint32(tri + 4, true),
    profile.view.getUint32(tri + 8, true),
  ];
  const position = new Array<number>(9);
  const color = new Array<number>(9);
  const [minX, minY, minZ, maxX, maxY, maxZ] = profile.bounds;
  for (let corner = 0; corner < 3; corner++) {
    const base = profile.vertexOffset + vertexIndices[corner]! * 16;
    position[corner * 3] = minX + profile.view.getUint16(base, true) / 65535 * (maxX - minX) + copyX;
    position[corner * 3 + 1] = minY + profile.view.getUint16(base + 2, true) / 65535 * (maxY - minY);
    position[corner * 3 + 2] = minZ + profile.view.getUint16(base + 4, true) / 65535 * (maxZ - minZ) + copyZ;
    color[corner * 3] = profile.view.getUint16(base + 6, true) / 65535;
    color[corner * 3 + 1] = profile.view.getUint16(base + 8, true) / 65535;
    color[corner * 3 + 2] = profile.view.getUint16(base + 10, true) / 65535;
  }
  const v0x = position[3]! - position[0]!;
  const v0y = position[4]! - position[1]!;
  const v0z = position[5]! - position[2]!;
  const v1x = position[6]! - position[0]!;
  const v1y = position[7]! - position[1]!;
  const v1z = position[8]! - position[2]!;
  const v2x = hx - position[0]!;
  const v2y = hy - position[1]!;
  const v2z = hz - position[2]!;
  const d00 = v0x * v0x + v0y * v0y + v0z * v0z;
  const d01 = v0x * v1x + v0y * v1y + v0z * v1z;
  const d11 = v1x * v1x + v1y * v1y + v1z * v1z;
  const d20 = v2x * v0x + v2y * v0y + v2z * v0z;
  const d21 = v2x * v1x + v2y * v1y + v2z * v1z;
  const denominator = d00 * d11 - d01 * d01;
  let wb = 1 / 3;
  let wc = 1 / 3;
  if (Math.abs(denominator) > 1e-24) {
    wb = (d11 * d20 - d01 * d21) / denominator;
    wc = (d00 * d21 - d01 * d20) / denominator;
  }
  let wa = 1 - wb - wc;
  wa = Math.max(0, wa);
  wb = Math.max(0, wb);
  wc = Math.max(0, wc);
  const sum = wa + wb + wc;
  if (sum > 1e-12) {
    wa /= sum;
    wb /= sum;
    wc /= sum;
  } else {
    wa = wb = wc = 1 / 3;
  }
  // The accepted runtime authored-colour carrier is linear RGB8. Quantising
  // here measures the exact population the proposed 16-entry palette replaces.
  const channel = (index: number): number => Math.max(0, Math.min(255, Math.round(255 * (
    wa * color[index]! + wb * color[index + 3]! + wc * color[index + 6]!
  ))));
  return (channel(0) << 16) | (channel(1) << 8) | channel(2);
}

function buildPopulation(profile: BoundProfile, semanticByTriangle: Uint8Array): {
  bins: ColorBin[];
  colorAt: Uint32Array;
  semanticAt: Uint8Array;
  coveredAt: Uint8Array;
  covered: number;
} {
  const compactWidth = profile.interiorWidth * profile.columns;
  const compactHeight = profile.interiorHeight * profile.rows;
  const compactCount = compactWidth * compactHeight;
  const colorAt = new Uint32Array(compactCount);
  colorAt.fill(MISS_COLOR);
  const semanticAt = new Uint8Array(compactCount);
  const coveredAt = new Uint8Array(compactCount);
  const histogram = new Map<number, { total: number; structure: number; plume: number }>();
  let covered = 0;
  for (let sliceIndex = 0; sliceIndex < profile.sliceCount; sliceIndex++) {
    const tileColumn = sliceIndex % profile.columns;
    const tileRow = Math.floor(sliceIndex / profile.columns);
    for (let py = 0; py < profile.interiorHeight; py++) {
      const atlasY = tileRow * profile.storedHeight + profile.gutter + py;
      const compactY = tileRow * profile.interiorHeight + py;
      for (let px = 0; px < profile.interiorWidth; px++) {
        const atlasX = tileColumn * profile.storedWidth + profile.gutter + px;
        const texel = atlasY * profile.atlasWidth + atlasX;
        const owner = profile.view.getUint32(profile.ownerOffset + texel * 4, true);
        if (owner === MISS_OWNER || profile.view.getUint16(profile.payloadOffset + texel * 8 + 6, true) === 0) continue;
        const triangleId = owner & 0x3f_ffff;
        const semantic = semanticByTriangle[triangleId] as Semantic;
        const key = decodeColorKey(profile, sliceIndex, px, py, texel, owner);
        const compactX = tileColumn * profile.interiorWidth + px;
        const compact = compactY * compactWidth + compactX;
        colorAt[compact] = key;
        semanticAt[compact] = semantic;
        coveredAt[compact] = 1;
        const prior = histogram.get(key) ?? { total: 0, structure: 0, plume: 0 };
        prior.total++;
        if (semantic === 1) prior.plume++;
        else prior.structure++;
        histogram.set(key, prior);
        covered++;
      }
    }
  }
  const bins = [...histogram.entries()]
    .sort(([left], [right]) => left - right)
    .map(([key, weights]) => {
      const rgb = unpackRgb(key);
      return { key, rgb, lab: linearRgbToLab(rgb), ...weights };
    });
  return { bins, colorAt, semanticAt, coveredAt, covered };
}

function fitPalette(bins: readonly ColorBin[]): { medoids: number[]; nearest: Uint16Array; errors: Float32Array; objective: number } {
  if (bins.length < PALETTE_SIZE) throw new Error(`only ${bins.length} occupied colours cannot fill palette16`);
  const count = bins.length;
  if (count > 5000) throw new Error(`exact deterministic PAM has ${count} colours; explicit gate ceiling is 5000`);
  const distances = new Float32Array(count * count);
  for (let left = 0; left < count; left++) {
    for (let right = left + 1; right < count; right++) {
      const value = deltaE00(bins[left]!.lab, bins[right]!.lab);
      distances[left * count + right] = value;
      distances[right * count + left] = value;
    }
  }
  const medoids: number[] = [];
  const nearestDistance = new Float64Array(count);
  nearestDistance.fill(Number.POSITIVE_INFINITY);
  for (let slot = 0; slot < PALETTE_SIZE; slot++) {
    let best = -1;
    let bestObjective = Number.POSITIVE_INFINITY;
    for (let candidate = 0; candidate < count; candidate++) {
      if (medoids.includes(candidate)) continue;
      let objective = 0;
      for (let sample = 0; sample < count; sample++) {
        objective += bins[sample]!.total * Math.min(nearestDistance[sample]!, distances[sample * count + candidate]!);
      }
      if (objective < bestObjective - 1e-9 || (
        Math.abs(objective - bestObjective) <= 1e-9
        && (best < 0 || bins[candidate]!.key < bins[best]!.key)
      )) {
        best = candidate;
        bestObjective = objective;
      }
    }
    medoids.push(best);
    for (let sample = 0; sample < count; sample++) {
      nearestDistance[sample] = Math.min(nearestDistance[sample]!, distances[sample * count + best]!);
    }
  }
  let improved = true;
  while (improved) {
    improved = false;
    let currentObjective = 0;
    for (let sample = 0; sample < count; sample++) currentObjective += bins[sample]!.total * nearestDistance[sample]!;
    let bestObjective = currentObjective;
    let bestSlot = -1;
    let bestCandidate = -1;
    const medoidSet = new Set(medoids);
    for (let slot = 0; slot < medoids.length; slot++) {
      for (let candidate = 0; candidate < count; candidate++) {
        if (medoidSet.has(candidate)) continue;
        let objective = 0;
        for (let sample = 0; sample < count; sample++) {
          let distance = distances[sample * count + candidate]!;
          for (let other = 0; other < medoids.length; other++) {
            if (other !== slot) distance = Math.min(distance, distances[sample * count + medoids[other]!]!);
          }
          objective += bins[sample]!.total * distance;
        }
        if (objective < bestObjective - 1e-7 || (
          Math.abs(objective - bestObjective) <= 1e-7
          && bestCandidate >= 0
          && bins[candidate]!.key < bins[bestCandidate]!.key
        )) {
          bestObjective = objective;
          bestSlot = slot;
          bestCandidate = candidate;
        }
      }
    }
    if (bestSlot >= 0) {
      medoids[bestSlot] = bestCandidate;
      for (let sample = 0; sample < count; sample++) {
        let distance = Number.POSITIVE_INFINITY;
        for (const medoid of medoids) distance = Math.min(distance, distances[sample * count + medoid]!);
        nearestDistance[sample] = distance;
      }
      improved = true;
    }
  }
  medoids.sort((left, right) => bins[left]!.key - bins[right]!.key);
  const nearest = new Uint16Array(count);
  const errors = new Float32Array(count);
  let objective = 0;
  for (let sample = 0; sample < count; sample++) {
    let bestSlot = 0;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (let slot = 0; slot < medoids.length; slot++) {
      const distance = distances[sample * count + medoids[slot]!]!;
      if (distance < bestDistance - 1e-7 || (Math.abs(distance - bestDistance) <= 1e-7 && slot < bestSlot)) {
        bestDistance = distance;
        bestSlot = slot;
      }
    }
    nearest[sample] = bestSlot;
    errors[sample] = bestDistance;
    objective += bins[sample]!.total * bestDistance;
  }
  return { medoids, nearest, errors, objective };
}

function weightedPercentile(
  bins: readonly ColorBin[],
  errors: Float32Array,
  percentile: number,
  weight: (bin: ColorBin) => number,
): number {
  const ordered = bins.map((bin, index) => ({ error: errors[index]!, weight: weight(bin) }))
    .filter((entry) => entry.weight > 0)
    .sort((left, right) => left.error - right.error);
  const total = ordered.reduce((sum, entry) => sum + entry.weight, 0);
  const target = total * percentile;
  let cumulative = 0;
  for (const entry of ordered) {
    cumulative += entry.weight;
    if (cumulative >= target) return entry.error;
  }
  return ordered.at(-1)?.error ?? Number.NaN;
}

function evaluateConnectedSupport(
  profile: BoundProfile,
  colorAt: Uint32Array,
  coveredAt: Uint8Array,
  errorByKey: Map<number, number>,
): { maxFraction: number; slices: Array<{ slice: number; covered: number; largestBad: number; fraction: number }>; largestMask: Uint8Array } {
  const compactWidth = profile.interiorWidth * profile.columns;
  const visited = new Uint8Array(profile.interiorWidth * profile.interiorHeight);
  const queue = new Uint32Array(visited.length);
  const largestMask = new Uint8Array(colorAt.length);
  const slices: Array<{ slice: number; covered: number; largestBad: number; fraction: number }> = [];
  let maxFraction = 0;
  for (let slice = 0; slice < profile.sliceCount; slice++) {
    visited.fill(0);
    const tileColumn = slice % profile.columns;
    const tileRow = Math.floor(slice / profile.columns);
    let covered = 0;
    let largest: number[] = [];
    for (let py = 0; py < profile.interiorHeight; py++) {
      for (let px = 0; px < profile.interiorWidth; px++) {
        const compact = (tileRow * profile.interiorHeight + py) * compactWidth
          + tileColumn * profile.interiorWidth + px;
        if (coveredAt[compact]) covered++;
        const local = py * profile.interiorWidth + px;
        if (visited[local] || !coveredAt[compact] || (errorByKey.get(colorAt[compact]!) ?? 0) <= ERROR_THRESHOLD) continue;
        let read = 0;
        let write = 0;
        queue[write++] = local;
        visited[local] = 1;
        const component: number[] = [];
        while (read < write) {
          const current = queue[read++]!;
          component.push(current);
          const x = current % profile.interiorWidth;
          const y = Math.floor(current / profile.interiorWidth);
          const neighbors = [
            y * profile.interiorWidth + (x + 1) % profile.interiorWidth,
            y * profile.interiorWidth + (x + profile.interiorWidth - 1) % profile.interiorWidth,
            ((y + 1) % profile.interiorHeight) * profile.interiorWidth + x,
            ((y + profile.interiorHeight - 1) % profile.interiorHeight) * profile.interiorWidth + x,
          ];
          for (const neighbor of neighbors) {
            if (visited[neighbor]) continue;
            const nx = neighbor % profile.interiorWidth;
            const ny = Math.floor(neighbor / profile.interiorWidth);
            const nextCompact = (tileRow * profile.interiorHeight + ny) * compactWidth
              + tileColumn * profile.interiorWidth + nx;
            if (!coveredAt[nextCompact] || (errorByKey.get(colorAt[nextCompact]!) ?? 0) <= ERROR_THRESHOLD) continue;
            visited[neighbor] = 1;
            queue[write++] = neighbor;
          }
        }
        if (component.length > largest.length) largest = component;
      }
    }
    const fraction = covered > 0 ? largest.length / covered : 0;
    maxFraction = Math.max(maxFraction, fraction);
    for (const local of largest) {
      const px = local % profile.interiorWidth;
      const py = Math.floor(local / profile.interiorWidth);
      const compact = (tileRow * profile.interiorHeight + py) * compactWidth
        + tileColumn * profile.interiorWidth + px;
      largestMask[compact] = 1;
    }
    slices.push({ slice, covered, largestBad: largest.length, fraction });
  }
  return { maxFraction, slices, largestMask };
}

function linearToSrgbByte(value: number): number {
  const srgb = value <= 0.0031308 ? 12.92 * value : 1.055 * value ** (1 / 2.4) - 0.055;
  return Math.max(0, Math.min(255, Math.round(srgb * 255)));
}

function displayRgb(key: number): readonly [number, number, number] {
  const rgb = unpackRgb(key);
  return [linearToSrgbByte(rgb[0]), linearToSrgbByte(rgb[1]), linearToSrgbByte(rgb[2])];
}

async function writeQa(
  qaRoot: string,
  profile: BoundProfile,
  population: ReturnType<typeof buildPopulation>,
  bins: readonly ColorBin[],
  fit: ReturnType<typeof fitPalette>,
  connected: ReturnType<typeof evaluateConnectedSupport>,
): Promise<Array<{ number: number; file: string; width: number; height: number; sha256: string; interpretation: string }>> {
  const width = profile.interiorWidth * profile.columns;
  const height = profile.interiorHeight * profile.rows;
  const binByKey = new Map(bins.map((bin, index) => [bin.key, index]));
  const paletteKeys = fit.medoids.map((index) => bins[index]!.key);
  const original = new Uint8Array(width * height * 3);
  const quantized = new Uint8Array(width * height * 3);
  const error = new Uint8Array(width * height * 3);
  const support = new Uint8Array(width * height * 3);
  for (let pixel = 0; pixel < population.colorAt.length; pixel++) {
    if (!population.coveredAt[pixel]) {
      const x = pixel % width;
      const y = Math.floor(pixel / width);
      const checker = ((x >>> 4) ^ (y >>> 4)) & 1 ? 38 : 58;
      original.fill(checker, pixel * 3, pixel * 3 + 3);
      quantized.fill(checker, pixel * 3, pixel * 3 + 3);
      error.fill(24, pixel * 3, pixel * 3 + 3);
      support.fill(24, pixel * 3, pixel * 3 + 3);
      continue;
    }
    const bin = binByKey.get(population.colorAt[pixel]!)!;
    const originalRgb = displayRgb(bins[bin]!.key);
    const paletteRgb = displayRgb(paletteKeys[fit.nearest[bin]!]!);
    original.set(originalRgb, pixel * 3);
    quantized.set(paletteRgb, pixel * 3);
    const e = fit.errors[bin]!;
    const t = Math.min(1, e / 10);
    error.set([
      Math.round(255 * t),
      Math.round(255 * Math.max(0, 1 - Math.abs(t - 0.45) / 0.45)),
      Math.round(255 * (1 - t)),
    ], pixel * 3);
    if (e > ERROR_THRESHOLD) {
      support.set(connected.largestMask[pixel] ? [255, 30, 20] : [255, 170, 20], pixel * 3);
    } else {
      support.set(population.semanticAt[pixel] === 1 ? [105, 55, 125] : [36, 92, 44], pixel * 3);
    }
  }
  const labelHeight = 36;
  const swatchHeight = 72;
  const sheetWidth = width * 2;
  const sheetHeight = labelHeight + swatchHeight + height;
  const sheet = new Uint8Array(sheetWidth * sheetHeight * 3);
  sheet.fill(22);
  for (let y = 0; y < height; y++) {
    const targetY = labelHeight + swatchHeight + y;
    sheet.set(original.subarray(y * width * 3, (y + 1) * width * 3), targetY * sheetWidth * 3);
    sheet.set(quantized.subarray(y * width * 3, (y + 1) * width * 3), (targetY * sheetWidth + width) * 3);
  }
  for (let slot = 0; slot < PALETTE_SIZE; slot++) {
    const left = Math.floor(slot * sheetWidth / PALETTE_SIZE);
    const right = Math.floor((slot + 1) * sheetWidth / PALETTE_SIZE);
    const rgb = displayRgb(paletteKeys[slot]!);
    for (let y = labelHeight; y < labelHeight + swatchHeight; y++) {
      for (let x = left; x < right; x++) sheet.set(rgb, (y * sheetWidth + x) * 3);
    }
  }
  const title = Buffer.from(`<svg width="${sheetWidth}" height="${sheetHeight}" xmlns="http://www.w3.org/2000/svg">
    <text x="12" y="25" fill="white" font-family="sans-serif" font-size="19">Original linear-RGB8 first-hit field</text>
    <text x="${width + 12}" y="25" fill="white" font-family="sans-serif" font-size="19">Joint deterministic palette16</text>
  </svg>`);
  const paths = [
    {
      number: 1,
      file: '001-original-vs-quantized-palette-mosaic.png',
      width: sheetWidth,
      height: sheetHeight,
      raw: sheet,
      overlay: title,
      interpretation: 'All 64 accepted first-hit slices without gutters: original authored linear-RGB8 carrier versus its joint palette16 reconstruction; palette swatches run left-to-right.',
    },
    {
      number: 2,
      file: '002-deltae00-error-diagnostics.png',
      width,
      height,
      raw: error,
      overlay: null,
      interpretation: 'CIEDE2000 error over all covered first hits; blue is zero, yellow is around the gate region, red is Delta-E-00 >= 10.',
    },
    {
      number: 3,
      file: '003-connected-error-support.png',
      width,
      height,
      raw: support,
      overlay: null,
      interpretation: 'Green structure and violet plume are within Delta-E-00 <= 5; orange is error >5; red is each slice largest toroidal four-connected bad component.',
    },
  ];
  const artifacts: Array<{ number: number; file: string; width: number; height: number; sha256: string; interpretation: string }> = [];
  for (const artifact of paths) {
    const output = resolve(qaRoot, artifact.file);
    let pipeline = sharp(artifact.raw, { raw: { width: artifact.width, height: artifact.height, channels: 3 } });
    if (artifact.overlay) pipeline = pipeline.composite([{ input: artifact.overlay }]);
    await pipeline.png({ compressionLevel: 9, adaptiveFiltering: false }).toFile(output);
    artifacts.push({
      number: artifact.number,
      file: artifact.file,
      width: artifact.width,
      height: artifact.height,
      sha256: sha256(readFileSync(output)),
      interpretation: artifact.interpretation,
    });
  }
  return artifacts;
}

async function main(): Promise<void> {
  const sourcePath = resolve(REPO_ROOT, process.argv[2] ?? SOURCE_RELATIVE);
  const profile = parseProfile(sourcePath);
  const sourceSha256 = sha256(profile.bytes);
  if (sourceSha256 !== '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c') {
    throw new Error(`palette gate is bound to accepted source 2ed57f59..., got ${sourceSha256}`);
  }
  const semanticByTriangle = bindTriangleSemantics(profile);
  const population = buildPopulation(profile, semanticByTriangle);
  const fit = fitPalette(population.bins);
  const errorByKey = new Map(population.bins.map((bin, index) => [bin.key, fit.errors[index]!]));
  const connected = evaluateConnectedSupport(profile, population.colorAt, population.coveredAt, errorByKey);
  const metrics = {
    occupiedLinearRgb8Colors: population.bins.length,
    coveredFirstHits: population.covered,
    objectiveWeightedMeanDeltaE00: fit.objective / population.covered,
    overall: {
      p95: weightedPercentile(population.bins, fit.errors, 0.95, (bin) => bin.total),
      p99: weightedPercentile(population.bins, fit.errors, 0.99, (bin) => bin.total),
    },
    plume: {
      covered: population.bins.reduce((sum, bin) => sum + bin.plume, 0),
      p95: weightedPercentile(population.bins, fit.errors, 0.95, (bin) => bin.plume),
    },
    structure: {
      covered: population.bins.reduce((sum, bin) => sum + bin.structure, 0),
      p95: weightedPercentile(population.bins, fit.errors, 0.95, (bin) => bin.structure),
    },
    connectedErrorSupport: {
      thresholdDeltaE00: ERROR_THRESHOLD,
      maxFractionPerSlice: connected.maxFraction,
      slices: connected.slices,
    },
  };
  const checks = {
    overallP95: { value: metrics.overall.p95, threshold: GATES.overallP95, pass: metrics.overall.p95 <= GATES.overallP95 },
    overallP99: { value: metrics.overall.p99, threshold: GATES.overallP99, pass: metrics.overall.p99 <= GATES.overallP99 },
    plumeP95: { value: metrics.plume.p95, threshold: GATES.plumeP95, pass: metrics.plume.p95 <= GATES.plumeP95 },
    structureP95: { value: metrics.structure.p95, threshold: GATES.structureP95, pass: metrics.structure.p95 <= GATES.structureP95 },
    connectedSupport: { value: connected.maxFraction, threshold: GATES.maxConnectedBadFractionPerSlice, pass: connected.maxFraction < GATES.maxConnectedBadFractionPerSlice },
  };
  const pass = Object.values(checks).every((check) => check.pass);
  const implementationSha256 = sha256(readFileSync(fileURLToPath(import.meta.url)));
  const recipe = {
    schema: 'laas-groundcover-candidate-h-palette16-gate/v1',
    sourceSha256,
    sourceMeshSha256: SOURCE_MESH_SHA256,
    implementationSha256,
    paletteSize: PALETTE_SIZE,
    fit: 'joint deterministic BUILD plus exhaustive improving PAM swaps; CIEDE2000 objective; RGB-key tie break',
    semantics: 'production primitiveRecipes triangle ranges; reproductive roles -> plume; all remaining roles -> structure; no colour threshold',
    coverage: 'binary accepted GCRP first-hit coverage across all 64 interior 256x256 slices; wrapped gutters excluded',
    connectivity: 'per-slice toroidal four-neighbour; max largest-bad-component / covered-slice-area',
    gates: GATES,
  };
  const recipeSha256 = sha256(JSON.stringify(recipe));
  const root = resolve(REPO_ROOT, 'data/work/groundcover-candidate-h-palette16', sourceSha256, recipeSha256);
  const qaRoot = resolve(root, 'qa');
  mkdirSync(qaRoot, { recursive: true });
  const artifacts = await writeQa(qaRoot, profile, population, population.bins, fit, connected);
  const palette = fit.medoids.map((index, slot) => ({
    slot,
    linearRgb8Hex: `#${population.bins[index]!.key.toString(16).padStart(6, '0')}`,
    linearRgb: population.bins[index]!.rgb,
    labD65: population.bins[index]!.lab,
  }));
  const index = {
    schema: recipe.schema,
    generatedAt: new Date().toISOString(),
    verdict: pass ? 'GREEN' : 'RED',
    source: { file: SOURCE_RELATIVE, sha256: sourceSha256, sourceMeshSha256: SOURCE_MESH_SHA256 },
    recipe: { ...recipe, recipeSha256 },
    palette,
    metrics,
    checks,
    artifacts,
  };
  writeFileSync(resolve(qaRoot, 'index.json'), `${JSON.stringify(index, null, 2)}\n`);
  console.log(JSON.stringify({ root, verdict: index.verdict, palette, metrics, checks, artifacts }, null, 2));
  if (!pass) process.exitCode = 2;
}

await main();
