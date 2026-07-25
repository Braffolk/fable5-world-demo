/**
 * Corrected offline gate for physically filtered box-boundary transfer.
 *
 * Unlike the earlier categorical-cell census, every pixel here has one
 * explicit exterior pinhole origin.  All quadrature subrays share that origin,
 * obtain their own exact carrier boundary point, and trace the accepted mesh.
 * The tested codec stores a finest point field and performs one fixed two-tap
 * covariance-family filter live.  All traversal and loops are offline only.
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
const EXPECTED_SOURCE_SHA = '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const GUARD = 0.025;
const PATCH_TILES = 8;
const IMAGE_WIDTH = 6;
const IMAGE_HEIGHT = 4;
const PRODUCTION_PIXEL_ANGLE = 60 * Math.PI / 180 / 1920;
const QUADRATURE_LEVELS = [4, 8, 16] as const;
const CONVERGENCE = { rgb: 0.025, coverage: 0.025, depthMetres: 0.015, normalMoment: 0.06 };
const GRID = { q: 128, azimuth: 64, elevation: 32 } as const;
const FILTER_AXES = ['q0', 'q1', 'azimuth', 'elevation'] as const;
const FILTER_RADII_CELLS = [0, 1, 2, 4] as const;
const VQ_SIZE = 64;
const VQ_ITERATIONS = 24;
const RECORD_BYTES = 16;
const INDEX_BYTES = 1;
const CARRIER_TEXEL_BYTES = 4;
const MEMORY_CAP = 250 * 1024 * 1024;
const TAU = Math.PI * 2;
const KNOWN_INVALIDITIES = [
  'side point-field decode does not reproduce finite-patch truth phase and exact exit horizon',
  'chart-specific covariance wrapping is not implemented',
  'filter-family selection uses oracle subray addresses instead of the analytic live central-ray Jacobian',
  'the charged 16-byte centroid is not quantized and decoded from its physical format',
  'fit and score address sets overlap and translation metrics do not isolate excess decoded change',
  'total camera-depth moments omit the entry/transfer cross term required for compositing depth',
] as const;

type Chart = 'cap' | 'side';
type Vec = number[];

interface Camera {
  key: string;
  chart: Chart;
  origin: CensusVec3;
  forward: CensusVec3;
  right: CensusVec3;
  up: CensusVec3;
  horizontalFov: number;
  verticalFov: number;
  translationMetres: number;
}

interface CarrierRay {
  q: CensusVec3;
  direction: CensusVec3;
  entryT: number;
  transferHorizon: number;
  chart: Chart;
}

interface Surface {
  hit: boolean;
  premul: CensusVec3;
  coverage: number;
  transferDepth: number;
  normal: CensusVec3;
}

interface AddressFloat {
  chart: Chart;
  coordinates: readonly [number, number, number, number];
}

interface AddressCell {
  chart: Chart;
  indices: readonly [number, number, number, number];
  key: string;
}

interface Aggregate {
  vector: Vec;
  samples: number;
  addresses: AddressFloat[];
}

interface PixelTruth {
  camera: Camera;
  x: number;
  y: number;
  aggregate: Aggregate;
  priorAggregate: Aggregate;
  converged: boolean;
  centralAddress: AddressFloat | null;
  filterAxis: number;
  filterRadius: number;
  filterCells: AddressCell[];
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
  if (!(length > 0)) throw new Error('zero vector');
  return [value[0] / length, value[1] / length, value[2] / length];
}

function cross(a: CensusVec3, b: CensusVec3): CensusVec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function subtract(a: CensusVec3, b: CensusVec3): CensusVec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function addScaled(a: CensusVec3, b: CensusVec3, scale: number): CensusVec3 {
  return [a[0] + b[0] * scale, a[1] + b[1] * scale, a[2] + b[2] * scale];
}

function quantiles(valuesInput: readonly number[]): Record<string, number | null> {
  if (valuesInput.length === 0) return { p50: null, p90: null, p95: null, p99: null, maximum: null };
  const values = [...valuesInput].sort((a, b) => a - b);
  const at = (p: number): number => values[Math.floor((values.length - 1) * p)]!;
  return { p50: at(0.5), p90: at(0.9), p95: at(0.95), p99: at(0.99), maximum: values.at(-1)! };
}

function decodeOct(x01: number, y01: number): CensusVec3 {
  let x = x01 * 2 - 1;
  let y = y01 * 2 - 1;
  let z = 1 - Math.abs(x) - Math.abs(y);
  if (z < 0) {
    const oldX = x;
    x = (1 - Math.abs(y)) * (oldX >= 0 ? 1 : -1);
    y = (1 - Math.abs(oldX)) * (y >= 0 ? 1 : -1);
  }
  return normalized([x, y, z]);
}

function attributes(
  bytes: Uint8Array,
  geometry: DecodedOwnedProfileGeometry,
  origin: CensusVec3,
  direction: CensusVec3,
  hit: OriginAwareTruthHit,
): Surface {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const vertexOffset = view.getUint32(84, true);
  const triangle = hit.triangleId * 3;
  const vertices = [
    geometry.triangles[triangle]!,
    geometry.triangles[triangle + 1]!,
    geometry.triangles[triangle + 2]!,
  ] as const;
  const point: CensusVec3 = [
    origin[0] + direction[0] * hit.t - hit.copyX * geometry.tileSizeX,
    origin[1] + direction[1] * hit.t,
    origin[2] + direction[2] * hit.t - hit.copyZ * geometry.tileSizeZ,
  ];
  const position = (vertex: number): CensusVec3 => {
    const o = vertex * 3;
    return [geometry.positions[o]!, geometry.positions[o + 1]!, geometry.positions[o + 2]!];
  };
  const a = position(vertices[0]);
  const b = position(vertices[1]);
  const c = position(vertices[2]);
  const e0 = subtract(b, a);
  const e1 = subtract(c, a);
  const q = subtract(point, a);
  const d00 = e0[0] ** 2 + e0[1] ** 2 + e0[2] ** 2;
  const d01 = e0[0] * e1[0] + e0[1] * e1[1] + e0[2] * e1[2];
  const d11 = e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2;
  const d20 = q[0] * e0[0] + q[1] * e0[1] + q[2] * e0[2];
  const d21 = q[0] * e1[0] + q[1] * e1[1] + q[2] * e1[2];
  const denominator = d00 * d11 - d01 * d01;
  const u = denominator === 0 ? 0 : (d11 * d20 - d01 * d21) / denominator;
  const v = denominator === 0 ? 0 : (d00 * d21 - d01 * d20) / denominator;
  const weights = [1 - u - v, u, v];
  const color: [number, number, number] = [0, 0, 0];
  const normal: [number, number, number] = [0, 0, 0];
  for (let corner = 0; corner < 3; corner++) {
    const o = vertexOffset + vertices[corner]! * 16;
    const n = decodeOct(view.getUint16(o + 12, true) / 65535, view.getUint16(o + 14, true) / 65535);
    color[0] += weights[corner]! * view.getUint16(o + 6, true) / 65535;
    color[1] += weights[corner]! * view.getUint16(o + 8, true) / 65535;
    color[2] += weights[corner]! * view.getUint16(o + 10, true) / 65535;
    normal[0] += weights[corner]! * n[0];
    normal[1] += weights[corner]! * n[1];
    normal[2] += weights[corner]! * n[2];
  }
  return { hit: true, premul: color, coverage: 1, transferDepth: hit.t, normal: normalized(normal) };
}

function lookCamera(
  key: string,
  chart: Chart,
  origin: CensusVec3,
  target: CensusVec3,
  horizontalFovDegrees: number,
  verticalFovDegrees: number,
  translationMetres = 0,
): Camera {
  const forward = normalized(subtract(target, origin));
  const right = normalized(cross(forward, Math.abs(forward[1]) > 0.99 ? [0, 0, -1] : [0, 1, 0]));
  const up = normalized(cross(right, forward));
  return {
    key,
    chart,
    origin: addScaled(origin, right, translationMetres),
    forward,
    right,
    up,
    horizontalFov: horizontalFovDegrees * Math.PI / 180,
    verticalFov: verticalFovDegrees * Math.PI / 180,
    translationMetres,
  };
}

function makeCameras(geometry: DecodedOwnedProfileGeometry, hMin: number, hMax: number): Camera[] {
  const cx = geometry.tileOriginX + geometry.tileSizeX * 0.38196601125;
  const cz = geometry.tileOriginZ + geometry.tileSizeZ * 0.61803398875;
  const patch = geometry.tileSizeX * PATCH_TILES;
  const midY = (hMin + hMax) * 0.5;
  const base = [
    lookCamera('top', 'cap', [cx, hMax + 2, cz], [cx, hMax, cz], 10, 7),
    lookCamera('oblique-10deg', 'cap', [cx - 2.82, hMax + 0.5, cz - 0.3], [cx, hMax, cz], 6, 4),
    lookCamera('grazing-1deg', 'cap', [cx - 8.58, hMax + 0.15, cz], [cx, hMax, cz], 1.5, 1),
    lookCamera('side-horizontal', 'side', [-0.5, midY, patch * 0.5], [0, midY, patch * 0.5], 10, 7),
    lookCamera('side-oblique', 'side', [-0.5, hMax + 0.3, patch * 0.5], [2, midY, patch * 0.5], 8, 6),
  ];
  const translated: Camera[] = [];
  for (const camera of base) {
    translated.push(camera);
    for (const delta of [0.001, 0.0045]) {
      translated.push({ ...camera, key: `${camera.key}-shift-${delta * 1000}mm`, origin: addScaled(camera.origin, camera.right, delta), translationMetres: delta });
    }
  }
  return translated;
}

function pixelRay(camera: Camera, x: number, y: number, sx: number, sy: number, width: number, height: number): CensusVec3 {
  // IMAGE_WIDTH x IMAGE_HEIGHT are stratified sample locations across the
  // authored FOV, not the sensor resolution.  Select one production-sized
  // pixel in each stratum of a virtual sensor, then keep quadrature inside it.
  const virtualWidth = Math.max(1, Math.round(camera.horizontalFov / PRODUCTION_PIXEL_ANGLE));
  const virtualHeight = Math.max(1, Math.round(camera.verticalFov / PRODUCTION_PIXEL_ANGLE));
  const virtualX = Math.min(virtualWidth - 1, Math.floor((x + 0.5) / width * virtualWidth));
  const virtualY = Math.min(virtualHeight - 1, Math.floor((y + 0.5) / height * virtualHeight));
  const px = (((virtualX + sx) / virtualWidth) * 2 - 1) * Math.tan(camera.horizontalFov * 0.5);
  const py = (1 - ((virtualY + sy) / virtualHeight) * 2) * Math.tan(camera.verticalFov * 0.5);
  return normalized([
    camera.forward[0] + camera.right[0] * px + camera.up[0] * py,
    camera.forward[1] + camera.right[1] * px + camera.up[1] * py,
    camera.forward[2] + camera.right[2] * px + camera.up[2] * py,
  ]);
}

function rayBoxFar(origin: CensusVec3, direction: CensusVec3, minimum: CensusVec3, maximum: CensusVec3): number | null {
  let near = 0;
  let far = 64;
  for (let axis = 0; axis < 3; axis++) {
    if (Math.abs(direction[axis]!) <= 1e-15) {
      if (origin[axis]! < minimum[axis]! || origin[axis]! > maximum[axis]!) return null;
      continue;
    }
    let a = (minimum[axis]! - origin[axis]!) / direction[axis]!;
    let b = (maximum[axis]! - origin[axis]!) / direction[axis]!;
    if (a > b) [a, b] = [b, a];
    near = Math.max(near, a);
    far = Math.min(far, b);
    if (near > far) return null;
  }
  return far;
}

function carrierRay(
  camera: Camera,
  direction: CensusVec3,
  geometry: DecodedOwnedProfileGeometry,
  hMin: number,
  hMax: number,
): CarrierRay | null {
  if (camera.chart === 'cap') {
    if (!(direction[1] < -1e-12)) return null;
    const t = (hMax - camera.origin[1]) / direction[1];
    if (!(t > 0)) return null;
    return {
      q: addScaled(camera.origin, direction, t), direction, entryT: t, transferHorizon: 64 - t, chart: 'cap',
    };
  }
  if (!(direction[0] > 1e-12)) return null;
  const t = -camera.origin[0] / direction[0];
  if (!(t > 0)) return null;
  const q = addScaled(camera.origin, direction, t);
  const patchX = geometry.tileSizeX * PATCH_TILES;
  const patchZ = geometry.tileSizeZ * PATCH_TILES;
  if (q[1] < hMin || q[1] > hMax || q[2] < 0 || q[2] > patchZ) return null;
  const far = rayBoxFar(q, direction, [0, hMin, 0], [patchX, hMax, patchZ]);
  return far && far > 0 ? { q, direction, entryT: t, transferHorizon: far, chart: 'side' } : null;
}

function addressOf(ray: CarrierRay, geometry: DecodedOwnedProfileGeometry, hMin: number, hMax: number): AddressFloat {
  if (ray.chart === 'cap') {
    const azimuth = mod(Math.atan2(ray.direction[2], ray.direction[0]), TAU);
    const elevation = Math.asin(clamp(-ray.direction[1], 0, 1));
    return {
      chart: 'cap',
      coordinates: [
        mod((ray.q[0] - geometry.tileOriginX) / geometry.tileSizeX, 1),
        mod((ray.q[2] - geometry.tileOriginZ) / geometry.tileSizeZ, 1),
        azimuth / TAU,
        elevation / (Math.PI * 0.5),
      ],
    };
  }
  const azimuth = clamp(Math.atan2(ray.direction[2], ray.direction[0]) / Math.PI + 0.5, 0, 1 - Number.EPSILON);
  const elevation = clamp(Math.asin(clamp(ray.direction[1], -1, 1)) / Math.PI + 0.5, 0, 1 - Number.EPSILON);
  return {
    chart: 'side',
    coordinates: [
      mod((ray.q[2] - geometry.tileOriginZ) / geometry.tileSizeZ, 1),
      clamp((ray.q[1] - hMin) / (hMax - hMin), 0, 1 - Number.EPSILON),
      azimuth,
      elevation,
    ],
  };
}

function traceSubray(
  bytes: Uint8Array,
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  ray: CarrierRay | null,
): Surface {
  if (!ray || !(ray.transferHorizon > 0)) return { hit: false, premul: [0, 0, 0], coverage: 0, transferDepth: 0, normal: [0, 0, 0] };
  const hit = periodicNearestSuccessor(geometry, bvh, ray.q, ray.direction, ray.transferHorizon, 0);
  return hit ? attributes(bytes, geometry, ray.q, ray.direction, hit)
    : { hit: false, premul: [0, 0, 0], coverage: 0, transferDepth: 0, normal: [0, 0, 0] };
}

function integrate(
  bytes: Uint8Array,
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  camera: Camera,
  x: number,
  y: number,
  level: number,
  hMin: number,
  hMax: number,
): Aggregate {
  const vector = new Array<number>(10).fill(0);
  const addresses: AddressFloat[] = [];
  const count = level * level;
  for (let sy = 0; sy < level; sy++) for (let sx = 0; sx < level; sx++) {
    const direction = pixelRay(camera, x, y, (sx + 0.5) / level, (sy + 0.5) / level, IMAGE_WIDTH, IMAGE_HEIGHT);
    const ray = carrierRay(camera, direction, geometry, hMin, hMax);
    if (ray) addresses.push(addressOf(ray, geometry, hMin, hMax));
    const surface = traceSubray(bytes, geometry, bvh, ray);
    vector[0] += surface.premul[0] / count;
    vector[1] += surface.premul[1] / count;
    vector[2] += surface.premul[2] / count;
    vector[3] += surface.coverage / count;
    if (surface.hit) {
      vector[4] += surface.transferDepth / count;
      vector[5] += surface.transferDepth * surface.transferDepth / count;
      vector[6] += surface.normal[0] / count;
      vector[7] += surface.normal[1] / count;
      vector[8] += surface.normal[2] / count;
    }
    vector[9] += ray ? ray.entryT / count : 0;
  }
  return { vector, samples: count, addresses };
}

function aggregateDifference(a: Aggregate, b: Aggregate): { rgb: number; coverage: number; depth: number; normal: number } {
  return {
    rgb: Math.max(Math.abs(a.vector[0]! - b.vector[0]!), Math.abs(a.vector[1]! - b.vector[1]!), Math.abs(a.vector[2]! - b.vector[2]!)),
    coverage: Math.abs(a.vector[3]! - b.vector[3]!),
    depth: Math.abs(a.vector[4]! - b.vector[4]!),
    normal: Math.hypot(a.vector[6]! - b.vector[6]!, a.vector[7]! - b.vector[7]!, a.vector[8]! - b.vector[8]!),
  };
}

function isConverged(difference: ReturnType<typeof aggregateDifference>): boolean {
  return difference.rgb <= CONVERGENCE.rgb
    && difference.coverage <= CONVERGENCE.coverage
    && difference.depth <= CONVERGENCE.depthMetres
    && difference.normal <= CONVERGENCE.normalMoment;
}

function cellOf(address: AddressFloat): AddressCell {
  const dimensions = [GRID.q, GRID.q, GRID.azimuth, GRID.elevation] as const;
  const indices = address.coordinates.map((coordinate, axis) => {
    const wrapped = axis === 0 || axis === 2 ? mod(coordinate, 1) : clamp(coordinate, 0, 1 - Number.EPSILON);
    return Math.min(dimensions[axis]! - 1, Math.floor(wrapped * dimensions[axis]!));
  }) as [number, number, number, number];
  return { chart: address.chart, indices, key: `${address.chart}:${indices.join(':')}` };
}

function offsetCell(cell: AddressCell, axis: number, offset: number): AddressCell {
  const dimensions = [GRID.q, GRID.q, GRID.azimuth, GRID.elevation] as const;
  const indices = [...cell.indices] as [number, number, number, number];
  indices[axis] = axis === 0 || axis === 2
    ? mod(indices[axis]! + offset, dimensions[axis]!)
    : Math.round(clamp(indices[axis]! + offset, 0, dimensions[axis]! - 1));
  return { chart: cell.chart, indices, key: `${cell.chart}:${indices.join(':')}` };
}

function covarianceFamily(addresses: readonly AddressFloat[], centre: AddressFloat): { axis: number; radius: number; covariance: number[][] } {
  const dimensions = [GRID.q, GRID.q, GRID.azimuth, GRID.elevation];
  const covariance = Array.from({ length: 4 }, () => new Array<number>(4).fill(0));
  for (const address of addresses) {
    if (address.chart !== centre.chart) continue;
    const delta = address.coordinates.map((value, axis) => {
      let d = value - centre.coordinates[axis]!;
      if (axis === 0 || axis === 2) d -= Math.round(d);
      return d * dimensions[axis]!;
    });
    for (let row = 0; row < 4; row++) for (let column = 0; column < 4; column++) {
      covariance[row]![column] += delta[row]! * delta[column]! / Math.max(1, addresses.length);
    }
  }
  let vector = [1, 1, 1, 1].map((v) => v * 0.5);
  for (let iteration = 0; iteration < 12; iteration++) {
    const next = covariance.map((row) => row.reduce((sum, value, column) => sum + value * vector[column]!, 0));
    const length = Math.hypot(...next);
    vector = length > 0 ? next.map((v) => v / length) : [1, 0, 0, 0];
  }
  let axis = 0;
  for (let i = 1; i < 4; i++) if (Math.abs(vector[i]!) > Math.abs(vector[axis]!)) axis = i;
  const sigma = Math.sqrt(Math.max(0, covariance[axis]![axis]!));
  let radius: number = FILTER_RADII_CELLS[0];
  for (const candidate of FILTER_RADII_CELLS) if (Math.abs(candidate - sigma) < Math.abs(radius - sigma)) radius = candidate;
  return { axis, radius, covariance };
}

function pointAtCell(
  bytes: Uint8Array,
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  cell: AddressCell,
  hMin: number,
  hMax: number,
): Vec {
  const c = cell.indices.map((index, axis) => (index + 0.5) / [GRID.q, GRID.q, GRID.azimuth, GRID.elevation][axis]!) as [number, number, number, number];
  let ray: CarrierRay;
  if (cell.chart === 'cap') {
    const azimuth = c[2] * TAU;
    const elevation = c[3] * Math.PI * 0.5;
    ray = {
      chart: 'cap',
      q: [geometry.tileOriginX + c[0] * geometry.tileSizeX, hMax, geometry.tileOriginZ + c[1] * geometry.tileSizeZ],
      direction: [Math.cos(elevation) * Math.cos(azimuth), -Math.sin(elevation), Math.cos(elevation) * Math.sin(azimuth)],
      entryT: 0,
      transferHorizon: 64,
    };
  } else {
    const azimuth = (c[2] - 0.5) * Math.PI;
    const elevation = (c[3] - 0.5) * Math.PI;
    const horizontal = Math.cos(elevation);
    ray = {
      chart: 'side',
      q: [0, hMin + c[1] * (hMax - hMin), geometry.tileOriginZ + c[0] * geometry.tileSizeZ],
      direction: [horizontal * Math.cos(azimuth), Math.sin(elevation), horizontal * Math.sin(azimuth)],
      entryT: 0,
      transferHorizon: geometry.tileSizeX * PATCH_TILES,
    };
  }
  const surface = traceSubray(bytes, geometry, bvh, ray);
  return surface.hit
    ? [surface.premul[0], surface.premul[1], surface.premul[2], 1, surface.transferDepth, surface.transferDepth ** 2, surface.normal[0], surface.normal[1], surface.normal[2], 0]
    : new Array<number>(10).fill(0);
}

function vectorDistance(a: Vec, b: Vec): number {
  const weights = [5, 5, 5, 7, 0.3, 0.05, 1, 1, 1, 0];
  return a.reduce((sum, value, index) => sum + weights[index]! * (value - b[index]!) ** 2, 0);
}

function fitVq(vectors: readonly Vec[]): { codebook: Vec[]; assignments: number[] } {
  const k = Math.min(VQ_SIZE, vectors.length);
  const codebook: Vec[] = [];
  codebook.push([...vectors[0]!]);
  while (codebook.length < k) {
    let farthest = vectors[0]!;
    let farthestDistance = -1;
    for (const vector of vectors) {
      const distance = Math.min(...codebook.map((centroid) => vectorDistance(vector, centroid)));
      if (distance > farthestDistance) { farthestDistance = distance; farthest = vector; }
    }
    codebook.push([...farthest]);
  }
  const assignments = new Array<number>(vectors.length).fill(0);
  for (let iteration = 0; iteration < VQ_ITERATIONS; iteration++) {
    const sums = Array.from({ length: k }, () => new Array<number>(10).fill(0));
    const counts = new Array<number>(k).fill(0);
    for (let i = 0; i < vectors.length; i++) {
      let best = 0;
      let bestDistance = Number.POSITIVE_INFINITY;
      for (let c = 0; c < k; c++) {
        const distance = vectorDistance(vectors[i]!, codebook[c]!);
        if (distance < bestDistance) { bestDistance = distance; best = c; }
      }
      assignments[i] = best;
      counts[best]++;
      for (let d = 0; d < 10; d++) sums[best]![d] += vectors[i]![d]!;
    }
    for (let c = 0; c < k; c++) if (counts[c]! > 0) {
      for (let d = 0; d < 10; d++) codebook[c]![d] = sums[c]![d]! / counts[c]!;
    }
  }
  return { codebook, assignments };
}

function packingLowerBound(vectors: readonly Vec[]): number {
  const selected: Vec[] = [];
  const separated = (a: Vec, b: Vec): boolean => {
    const rgb = Math.max(Math.abs(a[0]! - b[0]!), Math.abs(a[1]! - b[1]!), Math.abs(a[2]! - b[2]!));
    const coverage = Math.abs(a[3]! - b[3]!);
    return rgb > 0.30 || coverage > 0.5;
  };
  for (const vector of vectors) if (selected.every((prior) => separated(vector, prior))) selected.push(vector);
  return selected.length;
}

function percentileMetric(truth: readonly PixelTruth[], decoded: readonly Vec[]): Record<string, unknown> {
  const rgb: number[] = [];
  const coverage: number[] = [];
  const depth: number[] = [];
  const normal: number[] = [];
  let intersection = 0;
  let union = 0;
  for (let i = 0; i < truth.length; i++) {
    const a = truth[i]!.aggregate.vector;
    const b = decoded[i]!;
    rgb.push(Math.max(Math.abs(a[0]! - b[0]!), Math.abs(a[1]! - b[1]!), Math.abs(a[2]! - b[2]!)));
    coverage.push(Math.abs(a[3]! - b[3]!));
    if (a[3]! > 0.5 && b[3]! > 0.5) depth.push(Math.abs(a[4]! / a[3]! - b[4]! / b[3]!));
    normal.push(Math.hypot(a[6]! - b[6]!, a[7]! - b[7]!, a[8]! - b[8]!));
    const ah = a[3]! >= 0.5;
    const bh = b[3]! >= 0.5;
    if (ah && bh) intersection++;
    if (ah || bh) union++;
  }
  return {
    pixels: truth.length,
    premultipliedRgbMaxChannel: quantiles(rgb),
    coverageAbsolute: quantiles(coverage),
    conditionalTransferDepthMetres: quantiles(depth),
    normalMomentL2: quantiles(normal),
    silhouetteIoU: intersection / Math.max(1, union),
  };
}

function renderRows(pixels: readonly PixelTruth[], decoded: readonly Vec[], cameraKeys: readonly string[]): Buffer {
  const scale = 18;
  const width = IMAGE_WIDTH * scale * 2;
  const height = IMAGE_HEIGHT * scale * cameraKeys.length;
  const image = Buffer.alloc(width * height * 3, 18);
  const draw = (row: number, x: number, y: number, vector: Vec, panel: number): void => {
    const background = [0.115, 0.09, 0.065];
    const color = [0, 1, 2].map((channel) => vector[channel]! + background[channel]! * (1 - clamp(vector[3]!, 0, 1)));
    for (let py = 0; py < scale; py++) for (let px = 0; px < scale; px++) {
      const ix = panel * IMAGE_WIDTH * scale + x * scale + px;
      const iy = row * IMAGE_HEIGHT * scale + y * scale + py;
      const index = (iy * width + ix) * 3;
      image[index] = Math.round(clamp(color[0]!, 0, 1) * 255);
      image[index + 1] = Math.round(clamp(color[1]!, 0, 1) * 255);
      image[index + 2] = Math.round(clamp(color[2]!, 0, 1) * 255);
    }
  };
  for (let row = 0; row < cameraKeys.length; row++) {
    const key = cameraKeys[row]!;
    const members = pixels.filter((pixel) => pixel.camera.key === key);
    for (const pixel of members) {
      const index = pixels.indexOf(pixel);
      draw(row, pixel.x, pixel.y, pixel.aggregate.vector, 0);
      draw(row, pixel.x, pixel.y, decoded[index]!, 1);
    }
  }
  return image;
}

async function main(): Promise<void> {
  const started = performance.now();
  const bytes = readFileSync(SOURCE);
  const sourceSha = sha256(bytes);
  if (sourceSha !== EXPECTED_SOURCE_SHA) throw new Error(`source changed: ${sourceSha}`);
  const geometry = decodeOwnedProfileGeometry(bytes);
  const hMin = geometry.bounds.min[1] - GUARD;
  const hMax = geometry.bounds.max[1] + GUARD;
  console.error(`[filtered-codec] BVH for ${geometry.triangleCount.toLocaleString()} triangles`);
  const bvh = TriangleBvh.build(geometry);
  const cameras = makeCameras(geometry, hMin, hMax);
  const pixels: PixelTruth[] = [];
  const convergenceDifferences: ReturnType<typeof aggregateDifference>[] = [];
  let escalated = 0;
  let failedConvergence = 0;
  console.error(`[filtered-codec] integrating ${cameras.length} explicit camera configurations`);
  for (const camera of cameras) {
    const translated = camera.translationMetres > 0;
    const coordinates: Array<[number, number]> = translated
      ? [[1, 1], [3, 1], [5, 1], [1, 3], [3, 3], [5, 3]]
      : Array.from({ length: IMAGE_WIDTH * IMAGE_HEIGHT }, (_, index) => [index % IMAGE_WIDTH, Math.floor(index / IMAGE_WIDTH)] as [number, number]);
    for (const [x, y] of coordinates) {
      const q4 = integrate(bytes, geometry, bvh, camera, x, y, QUADRATURE_LEVELS[0], hMin, hMax);
      const q8 = integrate(bytes, geometry, bvh, camera, x, y, QUADRATURE_LEVELS[1], hMin, hMax);
      let aggregate = q8;
      let prior = q4;
      let difference = aggregateDifference(q4, q8);
      let converged = isConverged(difference);
      if (!converged) {
        escalated++;
        const q16 = integrate(bytes, geometry, bvh, camera, x, y, QUADRATURE_LEVELS[2], hMin, hMax);
        prior = q8;
        aggregate = q16;
        difference = aggregateDifference(q8, q16);
        converged = isConverged(difference);
      }
      if (!converged) failedConvergence++;
      convergenceDifferences.push(difference);
      const centralDirection = pixelRay(camera, x, y, 0.5, 0.5, IMAGE_WIDTH, IMAGE_HEIGHT);
      const centralRay = carrierRay(camera, centralDirection, geometry, hMin, hMax);
      const centralAddress = centralRay ? addressOf(centralRay, geometry, hMin, hMax) : null;
      const family = centralAddress ? covarianceFamily(aggregate.addresses, centralAddress) : { axis: 0, radius: 0, covariance: [] };
      const centralCell = centralAddress ? cellOf(centralAddress) : null;
      const filterCells = centralCell
        ? [offsetCell(centralCell, family.axis, -family.radius), offsetCell(centralCell, family.axis, family.radius)]
        : [];
      pixels.push({ camera, x, y, aggregate, priorAggregate: prior, converged, centralAddress, filterAxis: family.axis, filterRadius: family.radius, filterCells });
    }
  }

  const uniqueCells = new Map<string, AddressCell>();
  for (const pixel of pixels) for (const cell of pixel.filterCells) uniqueCells.set(cell.key, cell);
  console.error(`[filtered-codec] baking ${uniqueCells.size} actually addressed finest-field cells`);
  const cellEntries = [...uniqueCells.values()];
  const pointVectors = cellEntries.map((cell) => pointAtCell(bytes, geometry, bvh, cell, hMin, hMax));
  const fitted = fitVq(pointVectors);
  const decodedByKey = new Map<string, Vec>();
  for (let i = 0; i < cellEntries.length; i++) decodedByKey.set(cellEntries[i]!.key, fitted.codebook[fitted.assignments[i]!]!);
  const decodedPixels = pixels.map((pixel) => {
    if (pixel.filterCells.length === 0) return new Array<number>(10).fill(0);
    const result = new Array<number>(10).fill(0);
    for (const cell of pixel.filterCells) {
      const vector = decodedByKey.get(cell.key)!;
      for (let dimension = 0; dimension < result.length; dimension++) result[dimension] += vector[dimension]! / pixel.filterCells.length;
    }
    return result;
  });

  const basePixels = pixels.filter((pixel) => pixel.camera.translationMetres === 0);
  const baseIndices = basePixels.map((pixel) => pixels.indexOf(pixel));
  const baseDecoded = baseIndices.map((index) => decodedPixels[index]!);
  const allMetrics = percentileMetric(pixels, decodedPixels);
  const baseMetrics = percentileMetric(basePixels, baseDecoded);
  const byCamera = Object.fromEntries(cameras.map((camera) => {
    const members = pixels.filter((pixel) => pixel.camera.key === camera.key);
    const indices = members.map((pixel) => pixels.indexOf(pixel));
    return [camera.key, percentileMetric(members, indices.map((index) => decodedPixels[index]!))];
  }));
  const capCells = GRID.q * GRID.q * GRID.azimuth * GRID.elevation;
  const sideCells = capCells;
  const totalCells = capCells + sideCells;
  const indexBytes = totalCells * INDEX_BYTES;
  const codebookBytes = VQ_SIZE * RECORD_BYTES;
  const carrierBytes = GRID.q * GRID.q * GRID.azimuth * CARRIER_TEXEL_BYTES;
  const exactProjectedBytes = indexBytes + codebookBytes + carrierBytes;
  const rgbP95 = (baseMetrics.premultipliedRgbMaxChannel as Record<string, number>).p95;
  const coverageP95 = (baseMetrics.coverageAbsolute as Record<string, number>).p95;
  const iou = baseMetrics.silhouetteIoU as number;
  const convergencePass = failedConvergence / pixels.length <= 0.05;
  const sampledQualityPass = rgbP95 <= 0.15 && coverageP95 <= 0.15 && iou >= 0.97;
  const cameraKeys = cameras.filter((camera) => camera.translationMetres === 0).map((camera) => camera.key);
  const recipe = {
    model: 'box-boundary-shared-origin-filtered-codec-v1', sourceSha, guardMetres: GUARD,
    patchTiles: PATCH_TILES, image: [IMAGE_WIDTH, IMAGE_HEIGHT], quadrature: QUADRATURE_LEVELS,
    productionPixelAngleRadians: PRODUCTION_PIXEL_ANGLE,
    convergence: CONVERGENCE, grid: GRID, filterAxes: FILTER_AXES,
    filterRadiiCells: FILTER_RADII_CELLS, vqSize: VQ_SIZE, vqIterations: VQ_ITERATIONS,
    implementationSha256: sha256(readFileSync(fileURLToPath(import.meta.url))),
  };
  const recipeSha = sha256(canonicalJson(recipe));
  const outputRoot = resolve(WORKSPACE, 'data/work/groundcover-box-boundary-filtered-codec', sourceSha.slice(0, 16), recipeSha.slice(0, 16));
  const qaRoot = resolve(outputRoot, 'qa');
  mkdirSync(qaRoot, { recursive: true });
  const report = {
    schema: 'laas-groundcover-box-boundary-filtered-codec-gate/v1',
    created: new Date().toISOString(), recipe,
    source: { path: relative(WORKSPACE, SOURCE), sha256: sourceSha, vertices: geometry.vertexCount, triangles: geometry.triangleCount },
    target: {
      definition: 'adaptive shared-pinhole-origin subray integration; every subray computes its own exact cap or finite-patch side carrier q',
      outputs: 'premultiplied RGB, coverage, first two transfer-depth moments, and coverage-weighted barycentric normal moment',
      exteriorOnly: true,
    },
    quadrature: {
      pixels: pixels.length, escalatedTo16: escalated, failedConvergence,
      failedFraction: failedConvergence / pixels.length,
      finalDifference: {
        rgb: quantiles(convergenceDifferences.map((d) => d.rgb)),
        coverage: quantiles(convergenceDifferences.map((d) => d.coverage)),
        depthMetres: quantiles(convergenceDifferences.map((d) => d.depth)),
        normalMoment: quantiles(convergenceDifferences.map((d) => d.normal)),
      },
    },
    filterFamily: {
      construction: 'rank-2 subray address covariance, principal eigenvector quantized to one of four address axes; sigma quantized to radius 0/1/2/4; two symmetric finest-field taps',
      axes: FILTER_AXES, radiiCells: FILTER_RADII_CELLS,
      histogram: Object.fromEntries(FILTER_AXES.flatMap((axis, a) => FILTER_RADII_CELLS.map((radius) => [
        `${axis}:r${radius}`, pixels.filter((pixel) => pixel.filterAxis === a && pixel.filterRadius === radius).length,
      ]))),
      storageLevelsCharged: 'no prefiltered levels are hidden: every family reads the one charged finest table at two direct offsets',
    },
    codec: {
      actualFit: { addressedCells: cellEntries.length, codebookEntries: fitted.codebook.length, iterations: VQ_ITERATIONS },
      sampledPackingLowerBoundCodewordsAtTwoTimesRgbCoverageTolerance: packingLowerBound(pointVectors),
      fullAddressSpace: { capCells, canonicalD4SideCells: sideCells, totalCells },
      exactBytes: { indices: indexBytes, codebook: codebookBytes, carrierR32: carrierBytes, total: exactProjectedBytes, totalMiB: exactProjectedBytes / 1048576 },
      projectedReads: {
        perLane: 5,
        twoTier1LayersPlusControl: 11,
        explanation: 'one carrier read + two VQ-index reads + two dependent codebook reads per lane; radius/family selection is fixed ALU',
      },
      limitation: 'only cells addressed by the frozen explicit camera set were baked and fitted. Exact index bytes cover the full address grid, but full-domain codebook error is unmeasured; this is a sampled rate-distortion result, not a whole-domain codec certificate.',
    },
    metrics: { baseExteriorCameras: baseMetrics, allIncludingMillimetreTranslations: allMetrics, byCamera },
    validity: {
      status: 'INVALID_UNDECIDED',
      invalidatingFindings: KNOWN_INVALIDITIES,
      retainedFindings: [
        'production-size shared-origin pinhole pixels are now parameterized correctly',
        'periodic point-ray factorization is tested independently by analyze-box-boundary-transfer.ts',
      ],
    },
    verdict: {
      status: 'INVALID_UNDECIDED',
      sampledQualityPass, convergencePass, projectedMemoryPass: exactProjectedBytes <= MEMORY_CAP,
      fullDomainDecision: 'UNDECIDED',
      implementationAuthorized: false,
      reason: 'known setup defects invalidate codec quality and full-domain feasibility conclusions; see validity.invalidatingFindings',
    },
    elapsedMilliseconds: performance.now() - started,
  };
  const reportPath = resolve(outputRoot, 'report.json');
  const reportText = `${JSON.stringify(report, null, 2)}\n`;
  writeFileSync(reportPath, reportText);
  const qaFiles = [
    resolve(qaRoot, '01-filtered-truth-vs-decoded.png'),
    resolve(qaRoot, '02-rate-distortion-verdict.png'),
  ];
  const image = renderRows(pixels, decodedPixels, cameraKeys);
  await sharp(image, { raw: { width: IMAGE_WIDTH * 18 * 2, height: IMAGE_HEIGHT * 18 * cameraKeys.length, channels: 3 } }).png().toFile(qaFiles[0]!);
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="480"><rect width="1200" height="480" fill="#101713"/><text x="48" y="62" fill="#edf2ed" font-family="monospace" font-size="28">02 — PRODUCTION-PIXEL CODEC VALIDITY GATE</text><text x="48" y="125" fill="#c6d8c5" font-family="monospace" font-size="22">shared-origin pixels: ${pixels.length}; 16x escalations: ${escalated}; unconverged: ${failedConvergence}</text><text x="48" y="180" fill="#c6d8c5" font-family="monospace" font-size="22">diagnostic VQ fit: ${cellEntries.length} addressed cells → ${fitted.codebook.length} codewords</text><text x="48" y="235" fill="#c6d8c5" font-family="monospace" font-size="22">hypothetical full-grid bytes: ${(exactProjectedBytes / 1048576).toFixed(1)} MiB; projected reads: 11</text><text x="48" y="290" fill="#c6d8c5" font-family="monospace" font-size="22">metrics retained for diagnosis only; six setup invalidities remain</text><text x="48" y="365" fill="#f0c86a" font-family="monospace" font-size="26">INVALID / FULL DOMAIN UNDECIDED</text></svg>`;
  await sharp(Buffer.from(svg)).png().toFile(qaFiles[1]!);
  const index = {
    schema: 'laas-groundcover-box-boundary-filtered-codec-qa-index/v1', sourceSha256: sourceSha, recipeSha256: recipeSha,
    report: { path: relative(WORKSPACE, reportPath), sha256: sha256(reportText) },
    images: qaFiles.map((path, index) => ({ number: index + 1, path: relative(WORKSPACE, path), sha256: sha256(readFileSync(path)) })),
  };
  writeFileSync(resolve(qaRoot, 'index.json'), `${JSON.stringify(index, null, 2)}\n`);
  console.error(`[filtered-codec] wrote ${relative(WORKSPACE, outputRoot)}`);
  console.log(JSON.stringify({ outputRoot: relative(WORKSPACE, outputRoot), verdict: report.verdict, metrics: baseMetrics, quadrature: report.quadrature, codec: report.codec }, null, 2));
}

await main();
