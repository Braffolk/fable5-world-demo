/**
 * Offline gate for the guarded common-slab + boundary-transfer construction.
 *
 * All loops, rectangle enumeration, triangle BVH traversal, and sampling in
 * this file are cook/gate work.  The prospective runtime path is a fixed
 * carrier lookup followed by one direct 4D transfer lookup/correction.  This
 * file deliberately does not import or modify runtime or shader code.
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
const EXPECTED_SOURCE_SHA256 =
  '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const TUFT_CENTRES = [
  [0.052, 0.092], [0.226, 0.055], [0.431, 0.124],
  [0.133, 0.318], [0.335, 0.291], [0.469, 0.438],
] as const;
const GUARDS = [0.01, 0.025, 0.05] as const;
const ACTIVE_GUARD = 0.025;
const HORIZON = 64;
const MASK_RESOLUTION = 512;
const QA_RESOLUTION = 256;
const VIEW_SAMPLE_COUNT = 80;
const FACTORIZATION_RAYS = 192;
const MACROBLOCKS_CAP = 16;
const MACROBLOCKS_SIDE = 8;
const MACRO_EDGE = 4;
const PIXEL_ANGLE = 60 * Math.PI / 180 / 1920;
const MEMORY_CAP = 250 * 1024 * 1024;
const TAU = Math.PI * 2;
const EPSILON = 1e-9;

const CODEC_TIERS = [
  { key: 'q128-a32-e16', q: 128, azimuth: 32, elevation: 16 },
  { key: 'q256-a64-e32', q: 256, azimuth: 64, elevation: 32 },
  { key: 'q512-a128-e64', q: 512, azimuth: 128, elevation: 64 },
] as const;

const VIEW_FAMILIES = [
  { key: 'near-35deg', elevationDegrees: 35 },
  { key: 'oblique-10deg', elevationDegrees: 10 },
  { key: 'grazing-1deg', elevationDegrees: 1 },
] as const;

interface Rect {
  id: number;
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
  minY: number;
  maxY: number;
}

interface Interval {
  start: number;
  end: number;
  rectKeys: string[];
}

interface CarrierEntry {
  t: number;
  q: CensusVec3;
  intervalEndT: number;
  rectKeys: string[];
  face: 'cap' | 'side';
}

interface BoundaryEdge {
  x: number;
  z: number;
  outwardX: number;
  outwardZ: number;
}

interface HitRecord {
  key: string;
  hit: OriginAwareTruthHit | null;
  color: CensusVec3;
}

interface MacroStats {
  blocks: number;
  activeCells: number;
  correctionCells: number;
  correctionFraction: number;
  meanLocalOwners: number;
  patternUniqueFraction: number;
  patternBytesPerBlockObserved: number;
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

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

function mod(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus;
}

function normalized(value: CensusVec3): CensusVec3 {
  const length = Math.hypot(value[0], value[1], value[2]);
  if (!(length > 0)) throw new Error('cannot normalize zero vector');
  return [value[0] / length, value[1] / length, value[2] / length];
}

function quantiles(valuesInput: readonly number[]): Record<string, number | null> {
  if (valuesInput.length === 0) return { p50: null, p90: null, p95: null, p99: null, maximum: null };
  const values = [...valuesInput].sort((a, b) => a - b);
  const at = (p: number): number => values[Math.floor((values.length - 1) * p)]!;
  return { p50: at(0.5), p90: at(0.9), p95: at(0.95), p99: at(0.99), maximum: values.at(-1)! };
}

function makeRng(seed = 0x6b6f_7862): () => number {
  let state = seed >>> 0;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x1_0000_0000;
  };
}

function periodicDelta(value: number, centre: number, period: number): number {
  const raw = value - centre;
  return raw - Math.round(raw / period) * period;
}

function deriveShootBoxes(geometry: DecodedOwnedProfileGeometry): Rect[] {
  const boxes = TUFT_CENTRES.map((_, id) => ({
    id,
    minX: Number.POSITIVE_INFINITY,
    maxX: Number.NEGATIVE_INFINITY,
    minZ: Number.POSITIVE_INFINITY,
    maxZ: Number.NEGATIVE_INFINITY,
    minY: Number.POSITIVE_INFINITY,
    maxY: Number.NEGATIVE_INFINITY,
  }));
  for (let triangleId = 0; triangleId < geometry.triangleCount; triangleId++) {
    const triangle = triangleId * 3;
    let cx = 0;
    let cz = 0;
    for (let corner = 0; corner < 3; corner++) {
      const vertex = geometry.triangles[triangle + corner]! * 3;
      cx += geometry.positions[vertex]!;
      cz += geometry.positions[vertex + 2]!;
    }
    cx /= 3;
    cz /= 3;
    let nearest = 0;
    let nearestD2 = Number.POSITIVE_INFINITY;
    for (let tuft = 0; tuft < TUFT_CENTRES.length; tuft++) {
      const [tx, tz] = TUFT_CENTRES[tuft]!;
      const dx = periodicDelta(cx, tx, geometry.tileSizeX);
      const dz = periodicDelta(cz, tz, geometry.tileSizeZ);
      const d2 = dx * dx + dz * dz;
      if (d2 < nearestD2) {
        nearestD2 = d2;
        nearest = tuft;
      }
    }
    const box = boxes[nearest]!;
    const [tx, tz] = TUFT_CENTRES[nearest]!;
    for (let corner = 0; corner < 3; corner++) {
      const vertex = geometry.triangles[triangle + corner]! * 3;
      const x = tx + periodicDelta(geometry.positions[vertex]!, tx, geometry.tileSizeX);
      const y = geometry.positions[vertex + 1]!;
      const z = tz + periodicDelta(geometry.positions[vertex + 2]!, tz, geometry.tileSizeZ);
      box.minX = Math.min(box.minX, x);
      box.maxX = Math.max(box.maxX, x);
      box.minY = Math.min(box.minY, y);
      box.maxY = Math.max(box.maxY, y);
      box.minZ = Math.min(box.minZ, z);
      box.maxZ = Math.max(box.maxZ, z);
    }
  }
  for (const box of boxes) {
    if (!Number.isFinite(box.minX)) throw new Error(`empty derived shoot box ${box.id}`);
  }
  return boxes;
}

function guarded(boxes: readonly Rect[], guard: number): Rect[] {
  return boxes.map((box) => ({
    ...box,
    minX: box.minX - guard,
    maxX: box.maxX + guard,
    minZ: box.minZ - guard,
    maxZ: box.maxZ + guard,
    minY: box.minY - guard,
    maxY: box.maxY + guard,
  }));
}

function shiftedContains(rect: Rect, x: number, z: number, geometry: DecodedOwnedProfileGeometry): boolean {
  const centreX = (rect.minX + rect.maxX) * 0.5;
  const centreZ = (rect.minZ + rect.maxZ) * 0.5;
  const localX = centreX + periodicDelta(x, centreX, geometry.tileSizeX);
  const localZ = centreZ + periodicDelta(z, centreZ, geometry.tileSizeZ);
  return localX >= rect.minX - EPSILON && localX <= rect.maxX + EPSILON
    && localZ >= rect.minZ - EPSILON && localZ <= rect.maxZ + EPSILON;
}

function insideP(rects: readonly Rect[], x: number, z: number, geometry: DecodedOwnedProfileGeometry): boolean {
  return rects.some((rect) => shiftedContains(rect, x, z, geometry));
}

function containingRectIds(rects: readonly Rect[], x: number, z: number, geometry: DecodedOwnedProfileGeometry): Set<number> {
  return new Set(rects.filter((rect) => shiftedContains(rect, x, z, geometry)).map((rect) => rect.id));
}

function maskMetrics(rects: readonly Rect[], geometry: DecodedOwnedProfileGeometry, resolution: number): {
  mask: Uint8Array;
  areaFraction: number;
  perimeterMetres: number;
  components: number;
} {
  const mask = new Uint8Array(resolution * resolution);
  let occupied = 0;
  for (let z = 0; z < resolution; z++) {
    for (let x = 0; x < resolution; x++) {
      const wx = geometry.tileOriginX + (x + 0.5) / resolution * geometry.tileSizeX;
      const wz = geometry.tileOriginZ + (z + 0.5) / resolution * geometry.tileSizeZ;
      const value = insideP(rects, wx, wz, geometry) ? 1 : 0;
      mask[z * resolution + x] = value;
      occupied += value;
    }
  }
  let horizontalEdges = 0;
  let verticalEdges = 0;
  for (let z = 0; z < resolution; z++) {
    for (let x = 0; x < resolution; x++) {
      const value = mask[z * resolution + x]!;
      if (value !== mask[z * resolution + mod(x + 1, resolution)]!) verticalEdges++;
      if (value !== mask[mod(z + 1, resolution) * resolution + x]!) horizontalEdges++;
    }
  }
  const seen = new Uint8Array(mask.length);
  let components = 0;
  const queue = new Int32Array(mask.length);
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || seen[start]) continue;
    components++;
    let head = 0;
    let tail = 0;
    queue[tail++] = start;
    seen[start] = 1;
    while (head < tail) {
      const index = queue[head++]!;
      const x = index % resolution;
      const z = Math.floor(index / resolution);
      const neighbours = [
        z * resolution + mod(x - 1, resolution),
        z * resolution + mod(x + 1, resolution),
        mod(z - 1, resolution) * resolution + x,
        mod(z + 1, resolution) * resolution + x,
      ];
      for (const neighbour of neighbours) {
        if (mask[neighbour] && !seen[neighbour]) {
          seen[neighbour] = 1;
          queue[tail++] = neighbour;
        }
      }
    }
  }
  return {
    mask,
    areaFraction: occupied / mask.length,
    perimeterMetres: verticalEdges * geometry.tileSizeZ / resolution
      + horizontalEdges * geometry.tileSizeX / resolution,
    components,
  };
}

function tightUnionVolume(boxes: readonly Rect[], geometry: DecodedOwnedProfileGeometry, resolution: number): number {
  let accumulatedHeight = 0;
  for (let z = 0; z < resolution; z++) {
    for (let x = 0; x < resolution; x++) {
      const wx = geometry.tileOriginX + (x + 0.5) / resolution * geometry.tileSizeX;
      const wz = geometry.tileOriginZ + (z + 0.5) / resolution * geometry.tileSizeZ;
      const intervals = boxes
        .filter((box) => shiftedContains(box, wx, wz, geometry))
        .map((box) => [box.minY, box.maxY] as [number, number])
        .sort((a, b) => a[0] - b[0]);
      let end = Number.NEGATIVE_INFINITY;
      for (const interval of intervals) {
        accumulatedHeight += Math.max(0, interval[1] - Math.max(interval[0], end));
        end = Math.max(end, interval[1]);
      }
    }
  }
  return accumulatedHeight / (resolution * resolution) * geometry.tileSizeX * geometry.tileSizeZ;
}

function rayRectInterval(
  qx: number,
  qz: number,
  wx: number,
  wz: number,
  rect: Rect,
  maximumL: number,
): [number, number] | null {
  let near = 0;
  let far = maximumL;
  const slab = (origin: number, direction: number, minimum: number, maximum: number): boolean => {
    if (Math.abs(direction) <= 1e-15) return origin >= minimum - EPSILON && origin <= maximum + EPSILON;
    let a = (minimum - origin) / direction;
    let b = (maximum - origin) / direction;
    if (a > b) [a, b] = [b, a];
    near = Math.max(near, a);
    far = Math.min(far, b);
    return near <= far + EPSILON;
  };
  return slab(qx, wx, rect.minX, rect.maxX) && slab(qz, wz, rect.minZ, rect.maxZ)
    ? [Math.max(0, near), Math.min(maximumL, far)]
    : null;
}

function unionRayIntervals(
  qx: number,
  qz: number,
  wx: number,
  wz: number,
  maximumL: number,
  rects: readonly Rect[],
  geometry: DecodedOwnedProfileGeometry,
): Interval[] {
  const endX = qx + wx * maximumL;
  const endZ = qz + wz * maximumL;
  const pathMinX = Math.min(qx, endX);
  const pathMaxX = Math.max(qx, endX);
  const pathMinZ = Math.min(qz, endZ);
  const pathMaxZ = Math.max(qz, endZ);
  const raw: Interval[] = [];
  for (const rect of rects) {
    const minCopyX = Math.ceil((pathMinX - rect.maxX) / geometry.tileSizeX - EPSILON);
    const maxCopyX = Math.floor((pathMaxX - rect.minX) / geometry.tileSizeX + EPSILON);
    const minCopyZ = Math.ceil((pathMinZ - rect.maxZ) / geometry.tileSizeZ - EPSILON);
    const maxCopyZ = Math.floor((pathMaxZ - rect.minZ) / geometry.tileSizeZ + EPSILON);
    for (let copyZ = minCopyZ; copyZ <= maxCopyZ; copyZ++) {
      for (let copyX = minCopyX; copyX <= maxCopyX; copyX++) {
        const shifted: Rect = {
          ...rect,
          minX: rect.minX + copyX * geometry.tileSizeX,
          maxX: rect.maxX + copyX * geometry.tileSizeX,
          minZ: rect.minZ + copyZ * geometry.tileSizeZ,
          maxZ: rect.maxZ + copyZ * geometry.tileSizeZ,
        };
        const hit = rayRectInterval(qx, qz, wx, wz, shifted, maximumL);
        if (hit) raw.push({ start: hit[0], end: hit[1], rectKeys: [`${rect.id}:${copyX}:${copyZ}`] });
      }
    }
  }
  raw.sort((a, b) => a.start - b.start || a.end - b.end);
  const merged: Interval[] = [];
  for (const interval of raw) {
    const prior = merged.at(-1);
    if (!prior || interval.start > prior.end + EPSILON) {
      merged.push({ ...interval, rectKeys: [...interval.rectKeys] });
    } else {
      prior.end = Math.max(prior.end, interval.end);
      prior.rectKeys.push(...interval.rectKeys);
    }
  }
  return merged;
}

function slabInterval(originY: number, directionY: number, hMin: number, hMax: number, horizon: number): [number, number] | null {
  if (Math.abs(directionY) <= 1e-15) return originY >= hMin && originY <= hMax ? [0, horizon] : null;
  let a = (hMin - originY) / directionY;
  let b = (hMax - originY) / directionY;
  if (a > b) [a, b] = [b, a];
  const near = Math.max(0, a);
  const far = Math.min(horizon, b);
  return near <= far + EPSILON ? [near, far] : null;
}

function carrierEntry(
  origin: CensusVec3,
  direction: CensusVec3,
  hMin: number,
  hMax: number,
  rects: readonly Rect[],
  geometry: DecodedOwnedProfileGeometry,
  horizon: number,
): CarrierEntry | null {
  const slab = slabInterval(origin[1], direction[1], hMin, hMax, horizon);
  if (!slab) return null;
  const [tA, tB] = slab;
  const qAx = origin[0] + direction[0] * tA;
  const qAz = origin[2] + direction[2] * tA;
  const horizontal = Math.hypot(direction[0], direction[2]);
  if (horizontal <= 1e-15) {
    if (!insideP(rects, qAx, qAz, geometry)) return null;
    return { t: tA, q: [qAx, origin[1] + direction[1] * tA, qAz], intervalEndT: tB, rectKeys: [], face: 'cap' };
  }
  const intervals = unionRayIntervals(
    qAx,
    qAz,
    direction[0] / horizontal,
    direction[2] / horizontal,
    (tB - tA) * horizontal,
    rects,
    geometry,
  );
  if (intervals.length === 0) return null;
  const first = intervals[0]!;
  const t = tA + first.start / horizontal;
  return {
    t,
    q: [origin[0] + direction[0] * t, origin[1] + direction[1] * t, origin[2] + direction[2] * t],
    intervalEndT: tA + first.end / horizontal,
    rectKeys: [...new Set(first.rectKeys)],
    face: first.start <= EPSILON && tA > EPSILON ? 'cap' : 'side',
  };
}

function bruteCarrierEntry(
  origin: CensusVec3,
  direction: CensusVec3,
  hMin: number,
  hMax: number,
  rects: readonly Rect[],
  geometry: DecodedOwnedProfileGeometry,
  horizon: number,
): number | null {
  const end: CensusVec3 = [origin[0] + direction[0] * horizon, origin[1] + direction[1] * horizon, origin[2] + direction[2] * horizon];
  const minX = Math.min(origin[0], end[0]);
  const maxX = Math.max(origin[0], end[0]);
  const minZ = Math.min(origin[2], end[2]);
  const maxZ = Math.max(origin[2], end[2]);
  let best = Number.POSITIVE_INFINITY;
  for (const rect of rects) {
    const minCopyX = Math.ceil((minX - rect.maxX) / geometry.tileSizeX - EPSILON);
    const maxCopyX = Math.floor((maxX - rect.minX) / geometry.tileSizeX + EPSILON);
    const minCopyZ = Math.ceil((minZ - rect.maxZ) / geometry.tileSizeZ - EPSILON);
    const maxCopyZ = Math.floor((maxZ - rect.minZ) / geometry.tileSizeZ + EPSILON);
    for (let copyZ = minCopyZ; copyZ <= maxCopyZ; copyZ++) {
      for (let copyX = minCopyX; copyX <= maxCopyX; copyX++) {
        let near = 0;
        let far = horizon;
        const axes = [
          [origin[0], direction[0], rect.minX + copyX * geometry.tileSizeX, rect.maxX + copyX * geometry.tileSizeX],
          [origin[1], direction[1], hMin, hMax],
          [origin[2], direction[2], rect.minZ + copyZ * geometry.tileSizeZ, rect.maxZ + copyZ * geometry.tileSizeZ],
        ] as const;
        let valid = true;
        for (const [o, d, low, high] of axes) {
          if (Math.abs(d) <= 1e-15) {
            if (o < low - EPSILON || o > high + EPSILON) valid = false;
          } else {
            let a = (low - o) / d;
            let b = (high - o) / d;
            if (a > b) [a, b] = [b, a];
            near = Math.max(near, a);
            far = Math.min(far, b);
            if (near > far + EPSILON) valid = false;
          }
          if (!valid) break;
        }
        if (valid) best = Math.min(best, near);
      }
    }
  }
  return Number.isFinite(best) ? best : null;
}

function ownerKey(hit: OriginAwareTruthHit | null): string {
  return hit ? `${hit.triangleId}:${hit.copyX}:${hit.copyZ}` : 'MISS';
}

function vertexColor(bytes: Uint8Array, geometry: DecodedOwnedProfileGeometry, triangleId: number): CensusVec3 {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const vertexOffset = view.getUint32(84, true);
  const triangle = triangleId * 3;
  const color: [number, number, number] = [0, 0, 0];
  for (let corner = 0; corner < 3; corner++) {
    const vertex = geometry.triangles[triangle + corner]!;
    const offset = vertexOffset + vertex * 16;
    color[0] += view.getUint16(offset + 6, true) / 65535 / 3;
    color[1] += view.getUint16(offset + 8, true) / 65535 / 3;
    color[2] += view.getUint16(offset + 10, true) / 65535 / 3;
  }
  return color;
}

function trace(
  bytes: Uint8Array,
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  origin: CensusVec3,
  direction: CensusVec3,
  cache?: Map<string, HitRecord>,
): HitRecord {
  const cacheKey = `${origin.map((v) => v.toFixed(7)).join(',')}|${direction.map((v) => v.toFixed(8)).join(',')}`;
  const cached = cache?.get(cacheKey);
  if (cached) return cached;
  const hit = periodicNearestSuccessor(geometry, bvh, origin, direction, HORIZON, 0);
  const record = {
    key: ownerKey(hit),
    hit,
    color: hit ? vertexColor(bytes, geometry, hit.triangleId) : [0, 0, 0] as CensusVec3,
  };
  cache?.set(cacheKey, record);
  return record;
}

function directionDown(azimuth: number, elevation: number): CensusVec3 {
  const horizontal = Math.cos(elevation);
  return [horizontal * Math.cos(azimuth), -Math.sin(elevation), horizontal * Math.sin(azimuth)];
}

function directionSide(azimuth: number, elevation: number): CensusVec3 {
  const horizontal = Math.cos(elevation);
  return [horizontal * Math.cos(azimuth), Math.sin(elevation), horizontal * Math.sin(azimuth)];
}

function perturbDirection(direction: CensusVec3, yaw: number, pitch: number): CensusVec3 {
  const upSeed: CensusVec3 = Math.abs(direction[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const right = normalized([
    direction[1] * upSeed[2] - direction[2] * upSeed[1],
    direction[2] * upSeed[0] - direction[0] * upSeed[2],
    direction[0] * upSeed[1] - direction[1] * upSeed[0],
  ]);
  const up = normalized([
    right[1] * direction[2] - right[2] * direction[1],
    right[2] * direction[0] - right[0] * direction[2],
    right[0] * direction[1] - right[1] * direction[0],
  ]);
  return normalized([
    direction[0] + right[0] * yaw + up[0] * pitch,
    direction[1] + right[1] * yaw + up[1] * pitch,
    direction[2] + right[2] * yaw + up[2] * pitch,
  ]);
}

function randomTopBoundary(
  rng: () => number,
  rects: readonly Rect[],
  geometry: DecodedOwnedProfileGeometry,
  hMax: number,
): CensusVec3 {
  for (let attempt = 0; attempt < 10_000; attempt++) {
    const x = geometry.tileOriginX + rng() * geometry.tileSizeX;
    const z = geometry.tileOriginZ + rng() * geometry.tileSizeZ;
    if (insideP(rects, x, z, geometry)) return [x, hMax, z];
  }
  throw new Error('failed to sample active top boundary');
}

function boundaryEdges(mask: Uint8Array, geometry: DecodedOwnedProfileGeometry, resolution: number): BoundaryEdge[] {
  const result: BoundaryEdge[] = [];
  const dx = geometry.tileSizeX / resolution;
  const dz = geometry.tileSizeZ / resolution;
  for (let z = 0; z < resolution; z++) {
    for (let x = 0; x < resolution; x++) {
      if (!mask[z * resolution + x]) continue;
      const x0 = geometry.tileOriginX + x * dx;
      const z0 = geometry.tileOriginZ + z * dz;
      if (!mask[z * resolution + mod(x - 1, resolution)]!) result.push({ x: x0, z: z0 + dz * 0.5, outwardX: -1, outwardZ: 0 });
      if (!mask[z * resolution + mod(x + 1, resolution)]!) result.push({ x: x0 + dx, z: z0 + dz * 0.5, outwardX: 1, outwardZ: 0 });
      if (!mask[mod(z - 1, resolution) * resolution + x]!) result.push({ x: x0 + dx * 0.5, z: z0, outwardX: 0, outwardZ: -1 });
      if (!mask[mod(z + 1, resolution) * resolution + x]!) result.push({ x: x0 + dx * 0.5, z: z0 + dz, outwardX: 0, outwardZ: 1 });
    }
  }
  return result;
}

function factorizationGate(
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  rects: readonly Rect[],
  hMin: number,
  hMax: number,
): Record<string, unknown> {
  const rng = makeRng(0xfac7_0a11);
  let entryAgreement = 0;
  let transferAgreement = 0;
  let comparedHits = 0;
  let firstComponentMisses = 0;
  let vertical = 0;
  let horizontal = 0;
  const entryErrors: number[] = [];
  const transferErrors: number[] = [];
  const families = ['above', 'below', 'gap-horizontal'] as const;
  for (let index = 0; index < FACTORIZATION_RAYS; index++) {
    const family = families[index % families.length]!;
    let origin: CensusVec3;
    let direction: CensusVec3;
    if (family === 'above') {
      origin = [
        geometry.tileOriginX + rng() * geometry.tileSizeX,
        hMax + 0.02 + rng() * 1.8,
        geometry.tileOriginZ + rng() * geometry.tileSizeZ,
      ];
      if (index % 24 === 0) {
        direction = [0, -1, 0];
        vertical++;
      } else {
        const elevation = (1 + rng() * 88) * Math.PI / 180;
        direction = directionDown(rng() * TAU, elevation);
      }
    } else if (family === 'below') {
      origin = [
        geometry.tileOriginX + rng() * geometry.tileSizeX,
        hMin - 0.02 - rng() * 0.4,
        geometry.tileOriginZ + rng() * geometry.tileSizeZ,
      ];
      const elevation = (5 + rng() * 84) * Math.PI / 180;
      const down = directionDown(rng() * TAU, elevation);
      direction = [down[0], -down[1], down[2]];
    } else {
      let x = geometry.tileOriginX + rng() * geometry.tileSizeX;
      let z = geometry.tileOriginZ + rng() * geometry.tileSizeZ;
      for (let attempt = 0; attempt < 200 && insideP(rects, x, z, geometry); attempt++) {
        x = geometry.tileOriginX + rng() * geometry.tileSizeX;
        z = geometry.tileOriginZ + rng() * geometry.tileSizeZ;
      }
      origin = insideP(rects, x, z, geometry)
        ? [x, hMax + 0.02, z]
        : [x, hMin + (hMax - hMin) * rng(), z];
      direction = directionDown(rng() * TAU, 0);
      horizontal++;
    }
    const entry = carrierEntry(origin, direction, hMin, hMax, rects, geometry, HORIZON);
    const brute = bruteCarrierEntry(origin, direction, hMin, hMax, rects, geometry, HORIZON);
    if ((entry === null) === (brute === null)) {
      if (entry && brute !== null) {
        const error = Math.abs(entry.t - brute);
        entryErrors.push(error);
        if (error <= 1e-7) entryAgreement++;
      } else entryAgreement++;
    }
    const direct = periodicNearestSuccessor(geometry, bvh, origin, direction, HORIZON, 0);
    if (!entry) {
      if (!direct) transferAgreement++;
      continue;
    }
    const remaining = HORIZON - entry.t;
    const transferred = periodicNearestSuccessor(geometry, bvh, entry.q, direction, remaining, 0);
    const directKey = ownerKey(direct);
    const transferredKey = ownerKey(transferred);
    if (directKey === transferredKey) {
      if (direct && transferred) {
        const error = Math.abs(direct.t - (entry.t + transferred.t));
        transferErrors.push(error);
        comparedHits++;
        if (error <= 1e-7) transferAgreement++;
        if (direct.t > entry.intervalEndT + 1e-6) firstComponentMisses++;
      } else transferAgreement++;
    }
  }
  return {
    rays: FACTORIZATION_RAYS,
    vertical,
    horizontal,
    entryAgreement,
    transferAgreement,
    comparedHits,
    firstComponentMisses,
    entryErrorMetres: quantiles(entryErrors),
    composedHitErrorMetres: quantiles(transferErrors),
    pass: entryAgreement === FACTORIZATION_RAYS && transferAgreement === FACTORIZATION_RAYS,
    semantics: 'direct periodic triangle truth compared with exact 3D periodic rectangle-union entry plus whole-forward boundary transfer',
  };
}

function sampleViewRegularity(
  bytes: Uint8Array,
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  rects: readonly Rect[],
  hMax: number,
): Record<string, unknown> {
  const rng = makeRng(0x51ab_1e55);
  const cache = new Map<string, HitRecord>();
  const result: Record<string, unknown> = {};
  for (const view of VIEW_FAMILIES) {
    const byTier = Object.fromEntries(CODEC_TIERS.map((tier) => [tier.key, {
      mixed: 0,
      samples: 0,
      rgbErrors: [] as number[],
      coverageErrors: [] as number[],
    }])) as Record<string, { mixed: number; samples: number; rgbErrors: number[]; coverageErrors: number[] }>;
    let partiallyCoveredPixels = 0;
    const footprintColorRanges: number[] = [];
    const changes: Record<string, number> = { '1mm': 0, '2.5mm': 0, '4.5mm': 0 };
    let translationSamples = 0;
    for (let sample = 0; sample < VIEW_SAMPLE_COUNT; sample++) {
      const q = randomTopBoundary(rng, rects, geometry, hMax);
      const azimuth = rng() * TAU;
      const elevation = (view.elevationDegrees + (rng() - 0.5) * Math.min(1, view.elevationDegrees * 0.2)) * Math.PI / 180;
      const d = directionDown(azimuth, elevation);
      const centre = trace(bytes, geometry, bvh, q, d, cache);
      for (const tier of CODEC_TIERS) {
        const qx = geometry.tileSizeX / tier.q * 0.5;
        const qz = geometry.tileSizeZ / tier.q * 0.5;
        const da = TAU / tier.azimuth * 0.5;
        const de = (Math.PI * 0.5) / tier.elevation * 0.5;
        const records = [
          centre,
          trace(bytes, geometry, bvh, [q[0] - qx, q[1], q[2] - qz], directionDown(azimuth - da, clamp(elevation - de, 0, Math.PI / 2)), cache),
          trace(bytes, geometry, bvh, [q[0] + qx, q[1], q[2] + qz], directionDown(azimuth + da, clamp(elevation + de, 0, Math.PI / 2)), cache),
          trace(bytes, geometry, bvh, [q[0] - qx, q[1], q[2] + qz], directionDown(azimuth - da, clamp(elevation + de, 0, Math.PI / 2)), cache),
          trace(bytes, geometry, bvh, [q[0] + qx, q[1], q[2] - qz], directionDown(azimuth + da, clamp(elevation - de, 0, Math.PI / 2)), cache),
        ];
        const state = byTier[tier.key]!;
        state.samples++;
        if (new Set(records.map((record) => record.key)).size > 1) state.mixed++;
        const corners = records.slice(1);
        const predictedCoverage = corners.filter((record) => record.hit).length / corners.length;
        state.coverageErrors.push(Math.abs((centre.hit ? 1 : 0) - predictedCoverage));
        let rgbError = 0;
        for (let channel = 0; channel < 3; channel++) {
          const predicted = corners.reduce((sum, record) => sum + record.color[channel]!, 0) / corners.length;
          rgbError = Math.max(rgbError, Math.abs(centre.color[channel]! - predicted));
        }
        state.rgbErrors.push(rgbError);
      }
      const footprint = [
        centre,
        trace(bytes, geometry, bvh, q, perturbDirection(d, PIXEL_ANGLE * 0.5, 0), cache),
        trace(bytes, geometry, bvh, q, perturbDirection(d, -PIXEL_ANGLE * 0.5, 0), cache),
        trace(bytes, geometry, bvh, q, perturbDirection(d, 0, PIXEL_ANGLE * 0.5), cache),
        trace(bytes, geometry, bvh, q, perturbDirection(d, 0, -PIXEL_ANGLE * 0.5), cache),
      ];
      const coverage = footprint.filter((record) => record.hit).length / footprint.length;
      if (coverage > 0 && coverage < 1) partiallyCoveredPixels++;
      const mean: [number, number, number] = [0, 0, 0];
      for (const record of footprint) for (let channel = 0; channel < 3; channel++) mean[channel] += record.color[channel]! / footprint.length;
      let range = 0;
      for (const record of footprint) for (let channel = 0; channel < 3; channel++) range = Math.max(range, Math.abs(record.color[channel]! - mean[channel]!));
      footprintColorRanges.push(range);
      if (sample < 32) {
        translationSamples++;
        const right: CensusVec3 = [-Math.sin(azimuth), 0, Math.cos(azimuth)];
        for (const [label, delta] of [['1mm', 0.001], ['2.5mm', 0.0025], ['4.5mm', 0.0045]] as const) {
          const shifted: CensusVec3 = [q[0] + right[0] * delta, q[1], q[2] + right[2] * delta];
          if (trace(bytes, geometry, bvh, shifted, d, cache).key !== centre.key) changes[label]++;
        }
      }
    }
    result[view.key] = {
      samples: VIEW_SAMPLE_COUNT,
      atlasCellCategoricalMixedFraction: Object.fromEntries(Object.entries(byTier).map(([key, state]) => [key, state.mixed / state.samples])),
      heldOutCentreFromFourCellCorners: Object.fromEntries(Object.entries(byTier).map(([key, state]) => [key, {
        premultipliedRgbMaxChannelError: quantiles(state.rgbErrors),
        coverageAbsoluteError: quantiles(state.coverageErrors),
      }])),
      physicalPixelFootprint: {
        angularDiameterRadians: PIXEL_ANGLE,
        partialCoverageFraction: partiallyCoveredPixels / VIEW_SAMPLE_COUNT,
        maxChannelDeviationFromFiveTapMean: quantiles(footprintColorRanges),
      },
      categoricalChangeUnderCameraTranslation: Object.fromEntries(Object.entries(changes).map(([key, count]) => [key, count / translationSamples])),
    };
  }
  return result;
}

function canonicalPattern(keys: readonly string[]): string {
  const labels = new Map<string, number>();
  let next = 0;
  return keys.map((key) => {
    let label = labels.get(key);
    if (label === undefined) {
      label = next++;
      labels.set(key, label);
    }
    return label.toString(36);
  }).join('.');
}

function macroStats(patterns: readonly string[][]): MacroStats {
  let activeCells = 0;
  let correctionCells = 0;
  let localOwners = 0;
  let patternBytes = 0;
  const canonical = new Set<string>();
  for (const pattern of patterns) {
    const active = pattern.filter((key) => key !== 'OUT');
    activeCells += active.length;
    const histogram = new Map<string, number>();
    for (const key of active) histogram.set(key, (histogram.get(key) ?? 0) + 1);
    const modal = [...histogram.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? 'OUT';
    correctionCells += active.filter((key) => key !== modal).length;
    const owners = new Set(active);
    localOwners += owners.size;
    patternBytes += 256 + owners.size * 8 + 8;
    canonical.add(canonicalPattern(pattern));
  }
  return {
    blocks: patterns.length,
    activeCells,
    correctionCells,
    correctionFraction: correctionCells / Math.max(1, activeCells),
    meanLocalOwners: localOwners / Math.max(1, patterns.length),
    patternUniqueFraction: canonical.size / Math.max(1, patterns.length),
    patternBytesPerBlockObserved: patternBytes / Math.max(1, patterns.length),
  };
}

function sampleMacroblocks(
  bytes: Uint8Array,
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  rects: readonly Rect[],
  hMin: number,
  hMax: number,
  edges: readonly BoundaryEdge[],
): { cap: MacroStats; side: MacroStats } {
  const tier = CODEC_TIERS[1];
  const rng = makeRng(0xc0de_b10c);
  const cache = new Map<string, HitRecord>();
  const capPatterns: string[][] = [];
  const sidePatterns: string[][] = [];
  for (let block = 0; block < MACROBLOCKS_CAP; block++) {
    const qx0 = Math.floor(rng() * (tier.q / MACRO_EDGE)) * MACRO_EDGE;
    const qz0 = Math.floor(rng() * (tier.q / MACRO_EDGE)) * MACRO_EDGE;
    const a0 = Math.floor(rng() * (tier.azimuth / MACRO_EDGE)) * MACRO_EDGE;
    const e0 = Math.floor(rng() * (tier.elevation / MACRO_EDGE)) * MACRO_EDGE;
    const keys: string[] = [];
    for (let qe = 0; qe < MACRO_EDGE; qe++) for (let qf = 0; qf < MACRO_EDGE; qf++) {
      const x = geometry.tileOriginX + (qx0 + qe + 0.5) / tier.q * geometry.tileSizeX;
      const z = geometry.tileOriginZ + (qz0 + qf + 0.5) / tier.q * geometry.tileSizeZ;
      for (let ae = 0; ae < MACRO_EDGE; ae++) for (let ee = 0; ee < MACRO_EDGE; ee++) {
        if (!insideP(rects, x, z, geometry)) {
          keys.push('OUT');
          continue;
        }
        const azimuth = (a0 + ae + 0.5) / tier.azimuth * TAU;
        const elevation = (e0 + ee + 0.5) / tier.elevation * Math.PI / 2;
        keys.push(trace(bytes, geometry, bvh, [x, hMax, z], directionDown(azimuth, elevation), cache).key);
      }
    }
    capPatterns.push(keys);
  }
  if (edges.length > 0) {
    for (let block = 0; block < MACROBLOCKS_SIDE; block++) {
      const edge0 = Math.floor(rng() * Math.max(1, edges.length - MACRO_EDGE));
      const y0 = Math.floor(rng() * (tier.q / MACRO_EDGE)) * MACRO_EDGE;
      const a0 = Math.floor(rng() * (tier.azimuth / MACRO_EDGE)) * MACRO_EDGE;
      const e0 = Math.floor(rng() * (tier.elevation / MACRO_EDGE)) * MACRO_EDGE;
      const keys: string[] = [];
      for (let qe = 0; qe < MACRO_EDGE; qe++) for (let qf = 0; qf < MACRO_EDGE; qf++) {
        const edge = edges[(edge0 + qe) % edges.length]!;
        const y = hMin + (y0 + qf + 0.5) / tier.q * (hMax - hMin);
        for (let ae = 0; ae < MACRO_EDGE; ae++) for (let ee = 0; ee < MACRO_EDGE; ee++) {
          const azimuth = (a0 + ae + 0.5) / tier.azimuth * TAU;
          const elevation = ((e0 + ee + 0.5) / tier.elevation - 0.5) * Math.PI;
          const d = directionSide(azimuth, elevation);
          if (d[0] * edge.outwardX + d[2] * edge.outwardZ >= 0) {
            keys.push('OUT');
            continue;
          }
          keys.push(trace(bytes, geometry, bvh, [edge.x, y, edge.z], d, cache).key);
        }
      }
      sidePatterns.push(keys);
    }
  }
  return { cap: macroStats(capPatterns), side: macroStats(sidePatterns) };
}

function resourceEstimates(
  geometry: DecodedOwnedProfileGeometry,
  areaFraction: number,
  perimeter: number,
  slabHeight: number,
  macros: { cap: MacroStats; side: MacroStats },
  regularity: Record<string, unknown>,
): Record<string, unknown> {
  const tileArea = geometry.tileSizeX * geometry.tileSizeZ;
  const boundaryChartEquivalents = 2 * areaFraction + perimeter * slabHeight / tileArea;
  const mixedByTier = new Map<string, number[]>();
  const rgbP95ByTier = new Map<string, number[]>();
  const coverageP95ByTier = new Map<string, number[]>();
  for (const view of Object.values(regularity) as Array<Record<string, unknown>>) {
    const mixed = view.atlasCellCategoricalMixedFraction as Record<string, number>;
    for (const [key, fraction] of Object.entries(mixed)) {
      const values = mixedByTier.get(key) ?? [];
      values.push(fraction);
      mixedByTier.set(key, values);
    }
    const heldOut = view.heldOutCentreFromFourCellCorners as Record<string, {
      premultipliedRgbMaxChannelError: Record<string, number | null>;
      coverageAbsoluteError: Record<string, number | null>;
    }>;
    for (const [key, errors] of Object.entries(heldOut)) {
      const rgb = rgbP95ByTier.get(key) ?? [];
      const coverage = coverageP95ByTier.get(key) ?? [];
      rgb.push(errors.premultipliedRgbMaxChannelError.p95 ?? Number.POSITIVE_INFINITY);
      coverage.push(errors.coverageAbsoluteError.p95 ?? Number.POSITIVE_INFINITY);
      rgbP95ByTier.set(key, rgb);
      coverageP95ByTier.set(key, coverage);
    }
  }
  const correctionFraction = (
    macros.cap.correctionFraction * Math.max(0, 2 * areaFraction)
    + macros.side.correctionFraction * Math.max(0, perimeter * slabHeight / tileArea)
  ) / Math.max(EPSILON, boundaryChartEquivalents);
  const patternUniqueFraction = Math.max(macros.cap.patternUniqueFraction, macros.side.patternUniqueFraction || 0);
  const patternBytesPerBlock = Math.max(macros.cap.patternBytesPerBlockObserved, macros.side.patternBytesPerBlockObserved || 0);
  const tiers: Record<string, unknown> = {};
  for (const tier of CODEC_TIERS) {
    const directionCells = tier.azimuth * tier.elevation;
    const fineCells = boundaryChartEquivalents * tier.q * tier.q * directionCells;
    const blocks = fineCells / (MACRO_EDGE ** 4);
    const carrierBase = tier.q * tier.q * tier.azimuth * 4;
    const carrierWithMips = carrierBase * 4 / 3;
    const blockTable = blocks * 8 * 4 / 3;
    const observedMixed = Math.max(...(mixedByTier.get(tier.key) ?? [0]));
    const recordFraction = Math.min(1, Math.max(correctionFraction, observedMixed));
    const sparseRecords = fineCells * recordFraction * 16;
    const sparseBytes = carrierWithMips + blockTable + sparseRecords;
    const projectedUniquePatterns = Math.min(blocks, blocks * patternUniqueFraction);
    // A categorical centre-label pattern cannot represent a final mixed cell.
    // Charge a deliberately optimistic 12-byte filtered RGBA/depth/moment
    // payload for every observed mixed cell before considering the codebook.
    const mixedAppearancePayload = fineCells * observedMixed * 12;
    const codebookBytes = carrierWithMips + blockTable
      + blocks * Math.max(8, patternBytesPerBlock - 256)
      + projectedUniquePatterns * 256
      + mixedAppearancePayload;
    const heldOutRgbP95WorstView = Math.max(...(rgbP95ByTier.get(tier.key) ?? [Number.POSITIVE_INFINITY]));
    const heldOutCoverageP95WorstView = Math.max(...(coverageP95ByTier.get(tier.key) ?? [Number.POSITIVE_INFINITY]));
    const sampledQualityPass = heldOutRgbP95WorstView <= 0.15 && heldOutCoverageP95WorstView <= 0.25;
    tiers[tier.key] = {
      boundaryChartEquivalents,
      fineCells: Math.round(fineCells),
      macroblocks: Math.round(blocks),
      carrierR32BytesWithMips: Math.round(carrierWithMips),
      blockTableBytesWithMips: Math.round(blockTable),
      observedMixedFractionWorstView: observedMixed,
      observedModalCorrectionFraction: correctionFraction,
      heldOutRgbP95WorstView,
      heldOutCoverageP95WorstView,
      sampledQualityPass,
      sparsePerfectHashBytes: Math.round(sparseBytes),
      sparsePerfectHashMiB: sparseBytes / 1048576,
      sparseFits250MiB: sparseBytes <= MEMORY_CAP,
      macroblockCodebookBytes: Math.round(codebookBytes),
      macroblockCodebookMiB: codebookBytes / 1048576,
      macroblockCodebookFits250MiB: codebookBytes <= MEMORY_CAP,
      mixedAppearancePayloadBytes: Math.round(mixedAppearancePayload),
      estimatedExteriorReadsPerLane: { sparse: 5, codebook: 4 },
      estimatedTwoTier1LayerReadsIncludingControl: { sparse: 11, codebook: 9 },
    };
  }
  return {
    boundaryChartEquivalents,
    oneAtlasSharedByTwoAffineTier1Layers: true,
    macroSampling: macros,
    tiers,
    caveat: 'sampled modal/codebook estimates are deterministic projections, not a universal compression proof; a selected codec must be fully cooked and scored next',
  };
}

function rgb8(value: number): number {
  return Math.round(clamp(value, 0, 1) * 255);
}

async function writeFootprintQa(
  output: string,
  geometry: DecodedOwnedProfileGeometry,
  boxes: readonly Rect[],
  guardMasks: readonly { guard: number; metrics: ReturnType<typeof maskMetrics>; rects: readonly Rect[] }[],
): Promise<void> {
  const panel = QA_RESOLUTION;
  const image = Buffer.alloc(panel * guardMasks.length * panel * 3, 18);
  for (let p = 0; p < guardMasks.length; p++) {
    const item = guardMasks[p]!;
    for (let y = 0; y < panel; y++) {
      for (let x = 0; x < panel; x++) {
        const wx = geometry.tileOriginX + (x + 0.5) / panel * geometry.tileSizeX;
        const wz = geometry.tileOriginZ + (y + 0.5) / panel * geometry.tileSizeZ;
        const ids = containingRectIds(item.rects, wx, wz, geometry);
        const index = (y * panel * guardMasks.length + p * panel + x) * 3;
        if (ids.size > 0) {
          const first = [...ids][0]!;
          const hue = [[61, 112, 54], [90, 135, 60], [121, 154, 74], [55, 125, 104], [100, 94, 142], [154, 104, 115]][first]!;
          image[index] = hue[0]; image[index + 1] = hue[1]; image[index + 2] = hue[2];
          if (ids.size > 1) { image[index] = 210; image[index + 1] = 143; image[index + 2] = 55; }
        }
        const border = x < 2 || y < 2 || x >= panel - 2 || y >= panel - 2;
        if (border) { image[index] = 220; image[index + 1] = 220; image[index + 2] = 220; }
      }
    }
    for (const box of boxes) {
      const cx = Math.floor(mod((TUFT_CENTRES[box.id]![0] - geometry.tileOriginX) / geometry.tileSizeX, 1) * panel);
      const cz = Math.floor(mod((TUFT_CENTRES[box.id]![1] - geometry.tileOriginZ) / geometry.tileSizeZ, 1) * panel);
      for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
        if (dx * dx + dy * dy > 9) continue;
        const index = ((mod(cz + dy, panel)) * panel * guardMasks.length + p * panel + mod(cx + dx, panel)) * 3;
        image[index] = 255; image[index + 1] = 255; image[index + 2] = 255;
      }
    }
  }
  await sharp(image, { raw: { width: panel * guardMasks.length, height: panel, channels: 3 } }).png().toFile(output);
}

async function writeOwnerHeatmap(
  output: string,
  bytes: Uint8Array,
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  rects: readonly Rect[],
  hMax: number,
): Promise<void> {
  const resolution = 48;
  const views = [directionDown(73 * Math.PI / 180, 10 * Math.PI / 180), directionDown(211 * Math.PI / 180, Math.PI / 180)];
  const image = Buffer.alloc(resolution * views.length * resolution * 3, 14);
  const cache = new Map<string, HitRecord>();
  for (let view = 0; view < views.length; view++) {
    for (let z = 0; z < resolution; z++) for (let x = 0; x < resolution; x++) {
      const wx = geometry.tileOriginX + (x + 0.5) / resolution * geometry.tileSizeX;
      const wz = geometry.tileOriginZ + (z + 0.5) / resolution * geometry.tileSizeZ;
      const index = (z * resolution * views.length + view * resolution + x) * 3;
      if (!insideP(rects, wx, wz, geometry)) continue;
      const record = trace(bytes, geometry, bvh, [wx, hMax, wz], views[view]!, cache);
      if (!record.hit) { image[index] = 38; image[index + 1] = 32; image[index + 2] = 25; continue; }
      image[index] = rgb8(record.color[0]);
      image[index + 1] = rgb8(record.color[1]);
      image[index + 2] = rgb8(record.color[2]);
    }
  }
  await sharp(image, { raw: { width: resolution * views.length, height: resolution, channels: 3 } })
    .resize({ width: resolution * views.length * 5, height: resolution * 5, kernel: 'nearest' })
    .png().toFile(output);
}

function escapeXml(value: string): string {
  return value.replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' })[character]!);
}

async function writeSummaryQa(output: string, report: Record<string, unknown>): Promise<void> {
  const footprint = report.carrier as Record<string, unknown>;
  const resources = report.resources as Record<string, unknown>;
  const tiers = resources.tiers as Record<string, Record<string, unknown>>;
  const rows = [
    `source: ${EXPECTED_SOURCE_SHA256.slice(0, 16)} / arbitrary-soup production Calamagrostis`,
    `active guard: ${(ACTIVE_GUARD * 100).toFixed(1)} cm; fade footprint: ${((footprint.areaFraction as number) * 100).toFixed(1)}% of tile`,
    `common slab: ${(footprint.slabHeight as number).toFixed(3)} m; boundary chart equivalents: ${(resources.boundaryChartEquivalents as number).toFixed(2)}`,
    ...Object.entries(tiers).map(([key, tier]) => `${key}: sparse ${(tier.sparsePerfectHashMiB as number).toFixed(0)} MiB / codebook ${(tier.macroblockCodebookMiB as number).toFixed(0)} MiB`),
    `factorization: ${(report.factorization as Record<string, unknown>).pass ? 'PASS' : 'FAIL'}; verdict: ${(report.verdict as Record<string, unknown>).status}`,
  ];
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="520">
    <rect width="1200" height="520" fill="#111713"/>
    <text x="48" y="65" fill="#e8eee8" font-family="monospace" font-size="30">03 — BOX-BOUNDARY RESOURCE GATE</text>
    ${rows.map((row, index) => `<text x="48" y="${125 + index * 55}" fill="${index === rows.length - 1 ? '#f0c86a' : '#c2d4c0'}" font-family="monospace" font-size="22">${escapeXml(row)}</text>`).join('')}
  </svg>`;
  await sharp(Buffer.from(svg)).png().toFile(output);
}

async function main(): Promise<void> {
  const started = performance.now();
  const sourceBytes = readFileSync(SOURCE);
  const sourceSha256 = sha256(sourceBytes);
  if (sourceSha256 !== EXPECTED_SOURCE_SHA256) throw new Error(`accepted Calamagrostis source changed: ${sourceSha256}`);
  console.error('[box-boundary] decoding accepted production Calamagrostis');
  const geometry = decodeOwnedProfileGeometry(sourceBytes);
  const boxes = deriveShootBoxes(geometry);
  const guardMasks = GUARDS.map((guard) => {
    const rects = guarded(boxes, guard);
    return { guard, rects, metrics: maskMetrics(rects, geometry, MASK_RESOLUTION) };
  });
  const active = guardMasks.find((item) => item.guard === ACTIVE_GUARD)!;
  const hMin = geometry.bounds.min[1] - ACTIVE_GUARD;
  const hMax = geometry.bounds.max[1] + ACTIVE_GUARD;
  const slabHeight = hMax - hMin;
  const tightVolume = tightUnionVolume(boxes, geometry, 256);
  const carrierVolume = active.metrics.areaFraction * geometry.tileSizeX * geometry.tileSizeZ * slabHeight;
  const edges = boundaryEdges(active.metrics.mask, geometry, MASK_RESOLUTION);
  console.error(`[box-boundary] building BVH for ${geometry.triangleCount.toLocaleString()} triangles`);
  const bvh = TriangleBvh.build(geometry);
  console.error(`[box-boundary] BVH ready in ${bvh.metrics.buildMilliseconds.toFixed(0)} ms; factorization gate`);
  const factorization = factorizationGate(geometry, bvh, active.rects, hMin, hMax);
  console.error('[box-boundary] view/filter regularity gate');
  const regularity = sampleViewRegularity(sourceBytes, geometry, bvh, active.rects, hMax);
  console.error('[box-boundary] sampled macroblock codec census');
  const macros = sampleMacroblocks(sourceBytes, geometry, bvh, active.rects, hMin, hMax, edges);
  const resources = resourceEstimates(
    geometry,
    active.metrics.areaFraction,
    active.metrics.perimeterMetres,
    slabHeight,
    macros,
    regularity,
  );
  const tierResults = Object.values(resources.tiers as Record<string, Record<string, unknown>>);
  const fits = tierResults.some((tier) => tier.sampledQualityPass
    && (tier.sparseFits250MiB || tier.macroblockCodebookFits250MiB));
  const factorPass = Boolean((factorization as Record<string, unknown>).pass);
  const finiteEcologicalPatchSideChartTested = edges.length > 0;
  const recipe = {
    model: 'guarded-common-slab-boundary-transfer-v1',
    sourceSha256,
    guardsMetres: GUARDS,
    activeGuardMetres: ACTIVE_GUARD,
    horizonMetres: HORIZON,
    maskResolution: MASK_RESOLUTION,
    viewSampleCount: VIEW_SAMPLE_COUNT,
    factorizationRays: FACTORIZATION_RAYS,
    macroblocks: { cap: MACROBLOCKS_CAP, side: MACROBLOCKS_SIDE, edge: MACRO_EDGE },
    codecTiers: CODEC_TIERS,
    physicalPixelAngleRadians: PIXEL_ANGLE,
    implementationSha256: sha256(readFileSync(fileURLToPath(import.meta.url))),
  };
  const recipeSha256 = sha256(canonicalJson(recipe));
  const outputRoot = resolve(WORKSPACE, 'data/work/groundcover-box-boundary-transfer', sourceSha256.slice(0, 16), recipeSha256.slice(0, 16));
  const qaRoot = resolve(outputRoot, 'qa');
  mkdirSync(qaRoot, { recursive: true });
  const report: Record<string, unknown> = {
    schema: 'laas-groundcover-box-boundary-transfer-gate/v1',
    created: new Date().toISOString(),
    recipe,
    source: {
      path: relative(WORKSPACE, SOURCE),
      sha256: sourceSha256,
      vertices: geometry.vertexCount,
      triangles: geometry.triangleCount,
      tile: [geometry.tileSizeX, geometry.tileSizeZ],
      bounds: geometry.bounds,
    },
    carrier: {
      derivation: 'each source triangle assigned to its nearest authored tuft centre in periodic XZ; the assigned triangles define six containing AABBs; XZ guards are unioned and all boxes share one guarded vertical interval',
      shootBoxes: boxes,
      guardSweep: guardMasks.map((item) => ({ guardMetres: item.guard, ...item.metrics, mask: undefined })),
      activeGuardMetres: ACTIVE_GUARD,
      areaFraction: active.metrics.areaFraction,
      perimeterMetres: active.metrics.perimeterMetres,
      overlapConnectedComponents: active.metrics.components,
      slab: [hMin, hMax],
      slabHeight,
      tightPerShootBoxUnionVolumeApproxM3: tightVolume,
      commonSlabCarrierVolumeApproxM3: carrierVolume,
      fadeVolumeExpansionRatio: carrierVolume / tightVolume,
      fadeSemantics: 'the whole carrier lane fades while the camera is inside P x I; this gate does not promise per-original-AABB selective fade',
    },
    factorization,
    physicallyFilteredSampling: regularity,
    resources,
    performanceContract: {
      runtimeLoops: 0,
      runtimeMarches: 0,
      runtimeCandidateLists: 0,
      runtimeGeometry: 0,
      projectedReads: '4-5 per affine lane; 9-11 for two Tier-1 affine layers including control',
      atlasBytesSharedAcrossTier1Transforms: true,
    },
    verdict: {
      status: factorPass && fits && finiteEcologicalPatchSideChartTested
        ? 'GREEN_FOR_FULL_CODEC_COOK'
        : 'RED_INCOMPLETE_OR_OVER_BUDGET',
      periodicInteriorFactorizationPass: factorPass,
      finiteEcologicalPatchSideChartTested,
      sampledCodecUnder250MiB: fits,
      meaning: factorPass && fits && finiteEcologicalPatchSideChartTested
        ? 'the exact state reduction passed and at least one sampled fixed codec projects under the memory cap; the next gate is a complete cook and held-out reconstruction score'
        : 'do not implement runtime; the periodic interior identity may pass while the finite ecological-patch side entry remains untested, or no sampled fixed codec satisfies both quality and resident bytes',
    },
    elapsedMilliseconds: performance.now() - started,
  };
  const qaFiles = [
    resolve(qaRoot, '01-carrier-footprint-guard-sweep.png'),
    resolve(qaRoot, '02-boundary-owner-fields-oblique-grazing.png'),
    resolve(qaRoot, '03-resource-verdict.png'),
  ];
  await writeFootprintQa(qaFiles[0]!, geometry, boxes, guardMasks);
  await writeOwnerHeatmap(qaFiles[1]!, sourceBytes, geometry, bvh, active.rects, hMax);
  await writeSummaryQa(qaFiles[2]!, report);
  const reportPath = resolve(outputRoot, 'report.json');
  const reportSerialized = `${JSON.stringify(report, null, 2)}\n`;
  writeFileSync(reportPath, reportSerialized);
  const index = {
    schema: 'laas-groundcover-box-boundary-transfer-qa-index/v1',
    sourceSha256,
    recipeSha256,
    report: { path: relative(WORKSPACE, reportPath), sha256: sha256(reportSerialized) },
    images: qaFiles.map((path, index) => ({
      number: index + 1,
      path: relative(WORKSPACE, path),
      sha256: sha256(readFileSync(path)),
      interpretation: [
        'periodic top-down carrier footprints for 1 cm, 2.5 cm, and 5 cm guards; white dots are authored tuft centres and amber is overlap',
        'authored hit colours at top-boundary samples for 10-degree oblique and 1-degree grazing transfer queries; pixels are deliberately nearest-scaled to expose categorical frequency',
        'machine-derived carrier, memory, read, exact-factorization, and go/no-go summary',
      ][index],
    })),
  };
  writeFileSync(resolve(qaRoot, 'index.json'), `${JSON.stringify(index, null, 2)}\n`);
  console.error(`[box-boundary] wrote ${relative(WORKSPACE, outputRoot)}`);
  console.log(JSON.stringify({ outputRoot: relative(WORKSPACE, outputRoot), verdict: report.verdict, factorization, resources }, null, 2));
}

await main();
