import { createHash } from 'node:crypto';
import {
  PROFILE_TEXEL_BYTES,
  decodeOct,
  validateIndexedMesh,
  type IndexedMesh,
  type PackedProfile,
  type Vec3,
} from './ProfileFormat';

export const PERIODIC_PROFILE_VERSION = 2;

export interface PeriodicTile {
  originX: number;
  originZ: number;
  sizeX: number;
  sizeZ: number;
  topH: number;
}

export interface PeriodicCopy {
  ix: number;
  iz: number;
}

export interface PeriodicSlice {
  direction: Vec3;
  depthMin: number;
  depthMax: number;
  copies: PeriodicCopy[];
  copyRange: { minX: number; maxX: number; minZ: number; maxZ: number };
  conservativeProjectedBounds: { minX: number; maxX: number; minZ: number; maxZ: number };
}

export interface PeriodicBakeResult {
  tileWidth: number;
  tileHeight: number;
  atlasColumns: number;
  atlasRows: number;
  tile: PeriodicTile;
  slices: PeriodicSlice[];
  /** Unguarded canonical tiles, atlas-order RGBA32F. */
  pixels: number[];
  adapter: string;
}

export interface PackedPeriodicProfile extends PackedProfile {
  gutter: number;
  storedTileWidth: number;
  storedTileHeight: number;
  guardedPixels: number[];
}

export interface CushionSpec {
  centerX: number;
  centerZ: number;
  radiusX: number;
  radiusZ: number;
  height: number;
  yaw: number;
  /** Validation of double-sided normal handling; geometry itself is unchanged. */
  invertNormals?: boolean;
}

export interface CushionFixture {
  mesh: IndexedMesh;
  tile: PeriodicTile;
  cushions: CushionSpec[];
}

export interface PeriodicValidationStats {
  selected: number;
  skippedTriangleBoundary: number;
  rasterOwnerDiffersFromAnalytic: number;
  comparedHits: number;
  comparedMisses: number;
  hitMismatch: number;
  maxDepthTError: number;
  rmsDepthTError: number;
  minNormalDot: number;
  meanNormalDot: number;
}

export interface PeriodicHeader {
  version: number;
  profileId: number;
  storedTileWidth: number;
  storedTileHeight: number;
  atlasColumns: number;
  atlasRows: number;
  sliceCount: number;
  texelBytes: number;
  payloadOffset: number;
  mode: number;
  interiorTileWidth: number;
  interiorTileHeight: number;
  gutter: number;
  tile: PeriodicTile;
}

function normalized(v: Vec3): Vec3 {
  const length = Math.hypot(v.x, v.y, v.z);
  if (!(length > 0)) throw new Error('cannot normalize a zero vector');
  return { x: v.x / length, y: v.y / length, z: v.z / length };
}

function dot(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function mod(value: number, divisor: number): number {
  return ((value % divisor) + divisor) % divisor;
}

export function meshBounds(mesh: IndexedMesh): { min: Vec3; max: Vec3 } {
  validateIndexedMesh(mesh);
  const min = { x: Number.POSITIVE_INFINITY, y: Number.POSITIVE_INFINITY, z: Number.POSITIVE_INFINITY };
  const max = { x: Number.NEGATIVE_INFINITY, y: Number.NEGATIVE_INFINITY, z: Number.NEGATIVE_INFINITY };
  for (let i = 0; i < mesh.positions.length; i += 3) {
    const x = mesh.positions[i] as number;
    const y = mesh.positions[i + 1] as number;
    const z = mesh.positions[i + 2] as number;
    min.x = Math.min(min.x, x);
    min.y = Math.min(min.y, y);
    min.z = Math.min(min.z, z);
    max.x = Math.max(max.x, x);
    max.y = Math.max(max.y, y);
    max.z = Math.max(max.z, z);
  }
  return { min, max };
}

/**
 * Conservatively derives every periodic source copy whose top-plane projection can
 * touch the canonical tile. The bound uses mesh XZ bounds plus both extrema of the
 * height-dependent ray slope; it does not assume a fixed 3x3/5x5 neighborhood.
 */
export function makePeriodicSlice(mesh: IndexedMesh, tile: PeriodicTile, directionInput: Vec3): PeriodicSlice {
  if (!(tile.sizeX > 0 && tile.sizeZ > 0)) throw new Error('periodic tile dimensions must be positive');
  const direction = normalized(directionInput);
  if (!(direction.y < -1e-4)) throw new Error('periodic top-plane projection requires a downward ray');
  const bounds = meshBounds(mesh);
  if (!(tile.topH > bounds.max.y)) throw new Error('topH must be strictly above the mesh');
  const shift = (component: number, y: number): number => -component * ((y - tile.topH) / direction.y);
  const xShift0 = shift(direction.x, bounds.min.y);
  const xShift1 = shift(direction.x, bounds.max.y);
  const zShift0 = shift(direction.z, bounds.min.y);
  const zShift1 = shift(direction.z, bounds.max.y);
  const projected = {
    minX: bounds.min.x + Math.min(xShift0, xShift1),
    maxX: bounds.max.x + Math.max(xShift0, xShift1),
    minZ: bounds.min.z + Math.min(zShift0, zShift1),
    maxZ: bounds.max.z + Math.max(zShift0, zShift1),
  };
  const eps = 1e-8;
  const minX = Math.ceil((tile.originX - projected.maxX) / tile.sizeX - eps);
  const maxX = Math.floor((tile.originX + tile.sizeX - projected.minX) / tile.sizeX + eps);
  const minZ = Math.ceil((tile.originZ - projected.maxZ) / tile.sizeZ - eps);
  const maxZ = Math.floor((tile.originZ + tile.sizeZ - projected.minZ) / tile.sizeZ + eps);
  const copies: PeriodicCopy[] = [];
  for (let iz = minZ; iz <= maxZ; iz++) {
    for (let ix = minX; ix <= maxX; ix++) copies.push({ ix, iz });
  }
  const farthestT = (bounds.min.y - tile.topH) / direction.y;
  return {
    direction,
    depthMin: 0,
    depthMax: farthestT * 1.001 + 1e-5,
    copies,
    copyRange: { minX, maxX, minZ, maxZ },
    conservativeProjectedBounds: projected,
  };
}

function appendCushion(mesh: IndexedMesh, spec: CushionSpec, sectors: number, rings: number): void {
  const base = mesh.positions.length / 3;
  const cy = Math.cos(spec.yaw);
  const sy = Math.sin(spec.yaw);
  const addVertex = (lx: number, ly: number, lz: number): void => {
    const x = lx * cy - lz * sy;
    const z = lx * sy + lz * cy;
    mesh.positions.push(spec.centerX + x, ly, spec.centerZ + z);
    const nx0 = lx / (spec.radiusX * spec.radiusX);
    const ny0 = ly / (spec.height * spec.height);
    const nz0 = lz / (spec.radiusZ * spec.radiusZ);
    const nx = nx0 * cy - nz0 * sy;
    const nz = nx0 * sy + nz0 * cy;
    const n = normalized({ x: nx, y: ny0, z: nz });
    const sign = spec.invertNormals === true ? -1 : 1;
    mesh.normals.push(n.x * sign, n.y * sign, n.z * sign);
  };
  addVertex(0, spec.height, 0);
  for (let ring = 1; ring <= rings; ring++) {
    const phi = (ring / rings) * (Math.PI / 2);
    const radial = Math.sin(phi);
    const y = spec.height * Math.cos(phi);
    for (let sector = 0; sector < sectors; sector++) {
      const theta = (sector / sectors) * Math.PI * 2;
      addVertex(spec.radiusX * radial * Math.cos(theta), y, spec.radiusZ * radial * Math.sin(theta));
    }
  }
  for (let sector = 0; sector < sectors; sector++) {
    const next = (sector + 1) % sectors;
    mesh.indices.push(base, base + 1 + sector, base + 1 + next);
  }
  for (let ring = 0; ring < rings - 1; ring++) {
    const a0 = base + 1 + ring * sectors;
    const b0 = a0 + sectors;
    for (let sector = 0; sector < sectors; sector++) {
      const next = (sector + 1) % sectors;
      mesh.indices.push(a0 + sector, b0 + sector, a0 + next);
      mesh.indices.push(a0 + next, b0 + sector, b0 + next);
    }
  }
}

/** Connected/overlapping non-grid carpet used only to validate periodic geometry. */
export function makeCushionCarpetFixture(): CushionFixture {
  const tile: PeriodicTile = { originX: 0, originZ: 0, sizeX: 1, sizeZ: 1, topH: 0.38 };
  const cushions: CushionSpec[] = [
    { centerX: 0.04, centerZ: 0.10, radiusX: 0.23, radiusZ: 0.18, height: 0.24, yaw: 0.19 },
    { centerX: 0.29, centerZ: 0.18, radiusX: 0.22, radiusZ: 0.17, height: 0.29, yaw: -0.41 },
    { centerX: 0.57, centerZ: 0.09, radiusX: 0.24, radiusZ: 0.19, height: 0.21, yaw: 0.73 },
    { centerX: 0.87, centerZ: 0.16, radiusX: 0.25, radiusZ: 0.18, height: 0.27, yaw: -0.16 },
    { centerX: 0.13, centerZ: 0.43, radiusX: 0.24, radiusZ: 0.21, height: 0.19, yaw: 0.92 },
    { centerX: 0.43, centerZ: 0.38, radiusX: 0.27, radiusZ: 0.20, height: 0.31, yaw: 0.27 },
    { centerX: 0.73, centerZ: 0.42, radiusX: 0.23, radiusZ: 0.22, height: 0.23, yaw: -0.64, invertNormals: true },
    { centerX: 1.01, centerZ: 0.50, radiusX: 0.22, radiusZ: 0.19, height: 0.28, yaw: 0.48 },
    { centerX: 0.01, centerZ: 0.75, radiusX: 0.24, radiusZ: 0.20, height: 0.25, yaw: -0.76 },
    { centerX: 0.31, centerZ: 0.70, radiusX: 0.25, radiusZ: 0.23, height: 0.22, yaw: 0.11 },
    { centerX: 0.61, centerZ: 0.78, radiusX: 0.24, radiusZ: 0.19, height: 0.30, yaw: 0.58 },
    { centerX: 0.91, centerZ: 0.84, radiusX: 0.27, radiusZ: 0.22, height: 0.20, yaw: -0.33 },
  ];
  const mesh: IndexedMesh = { positions: [], normals: [], indices: [] };
  for (const cushion of cushions) appendCushion(mesh, cushion, 24, 12);
  validateIndexedMesh(mesh);
  return { mesh, tile, cushions };
}

export function periodicAddress(x: number, z: number, tile: PeriodicTile): { u: number; v: number } {
  return {
    u: mod((x - tile.originX) / tile.sizeX, 1),
    v: mod((z - tile.originZ) / tile.sizeZ, 1),
  };
}

/** Expand every canonical tile by wrapped gutters so bilinear taps never enter another slice. */
export function makeGuardedAtlas(result: PeriodicBakeResult, gutter: number): {
  pixels: number[];
  storedTileWidth: number;
  storedTileHeight: number;
} {
  if (!Number.isInteger(gutter) || gutter < 1) throw new Error('periodic atlas requires at least one integer gutter texel');
  const storedTileWidth = result.tileWidth + gutter * 2;
  const storedTileHeight = result.tileHeight + gutter * 2;
  const sourceWidth = result.tileWidth * result.atlasColumns;
  const storedWidth = storedTileWidth * result.atlasColumns;
  const storedHeight = storedTileHeight * result.atlasRows;
  if (result.pixels.length !== sourceWidth * result.tileHeight * result.atlasRows * 4) {
    throw new Error('periodic pixel payload has the wrong size');
  }
  const pixels = new Array<number>(storedWidth * storedHeight * 4).fill(0);
  result.slices.forEach((_slice, sliceIndex) => {
    const srcTileX = (sliceIndex % result.atlasColumns) * result.tileWidth;
    const srcTileY = Math.floor(sliceIndex / result.atlasColumns) * result.tileHeight;
    const dstTileX = (sliceIndex % result.atlasColumns) * storedTileWidth;
    const dstTileY = Math.floor(sliceIndex / result.atlasColumns) * storedTileHeight;
    for (let y = 0; y < storedTileHeight; y++) {
      const sourceY = mod(y - gutter, result.tileHeight);
      for (let x = 0; x < storedTileWidth; x++) {
        const sourceX = mod(x - gutter, result.tileWidth);
        const src = ((srcTileY + sourceY) * sourceWidth + srcTileX + sourceX) * 4;
        const dst = ((dstTileY + y) * storedWidth + dstTileX + x) * 4;
        pixels[dst] = result.pixels[src] as number;
        pixels[dst + 1] = result.pixels[src + 1] as number;
        pixels[dst + 2] = result.pixels[src + 2] as number;
        pixels[dst + 3] = result.pixels[src + 3] as number;
      }
    }
  });
  return { pixels, storedTileWidth, storedTileHeight };
}

function clampU16(value: number): number {
  return Math.max(0, Math.min(65535, Math.round(value * 65535)));
}

/**
 * GCRP/v2 periodic-top-XZ container. v2 is required because v1's per-direction
 * orthogonal UV bounds are not a runtime-compatible shared world tile.
 */
export function packPeriodicProfile(result: PeriodicBakeResult, profileId: number, gutter = 1): PackedPeriodicProfile {
  const guarded = makeGuardedAtlas(result, gutter);
  const atlasWidth = guarded.storedTileWidth * result.atlasColumns;
  const atlasHeight = guarded.storedTileHeight * result.atlasRows;
  const headerBytes = 80;
  const sliceBytes = 64;
  const payloadOffset = headerBytes + sliceBytes * result.slices.length;
  const bytes = new Uint8Array(payloadOffset + atlasWidth * atlasHeight * PROFILE_TEXEL_BYTES);
  const view = new DataView(bytes.buffer);
  bytes.set([0x47, 0x43, 0x52, 0x50], 0);
  view.setUint32(4, PERIODIC_PROFILE_VERSION, true);
  view.setUint32(8, profileId >>> 0, true);
  view.setUint32(12, guarded.storedTileWidth, true);
  view.setUint32(16, guarded.storedTileHeight, true);
  view.setUint32(20, result.atlasColumns, true);
  view.setUint32(24, result.atlasRows, true);
  view.setUint32(28, result.slices.length, true);
  view.setUint32(32, PROFILE_TEXEL_BYTES, true);
  view.setUint32(36, payloadOffset, true);
  view.setUint32(40, 1, true); // projection mode: periodic top-plane XZ
  view.setUint32(44, result.tileWidth, true);
  view.setUint32(48, result.tileHeight, true);
  view.setUint32(52, gutter, true);
  view.setFloat32(56, result.tile.topH, true);
  view.setFloat32(60, result.tile.originX, true);
  view.setFloat32(64, result.tile.originZ, true);
  view.setFloat32(68, result.tile.sizeX, true);
  view.setFloat32(72, result.tile.sizeZ, true);
  result.slices.forEach((slice, index) => {
    const base = headerBytes + index * sliceBytes;
    view.setFloat32(base, slice.direction.x, true);
    view.setFloat32(base + 4, slice.direction.y, true);
    view.setFloat32(base + 8, slice.direction.z, true);
    view.setFloat32(base + 12, slice.depthMin, true);
    view.setFloat32(base + 16, slice.depthMax, true);
    view.setInt32(base + 20, slice.copyRange.minX, true);
    view.setInt32(base + 24, slice.copyRange.maxX, true);
    view.setInt32(base + 28, slice.copyRange.minZ, true);
    view.setInt32(base + 32, slice.copyRange.maxZ, true);
  });
  for (let texel = 0; texel < atlasWidth * atlasHeight; texel++) {
    const src = texel * 4;
    const dst = payloadOffset + texel * PROFILE_TEXEL_BYTES;
    const hit = (guarded.pixels[src + 3] as number) > 0.5;
    view.setUint16(dst, hit ? Math.min(65534, clampU16(guarded.pixels[src] as number)) : 65535, true);
    view.setUint16(dst + 2, hit ? clampU16(guarded.pixels[src + 1] as number) : 32768, true);
    view.setUint16(dst + 4, hit ? clampU16(guarded.pixels[src + 2] as number) : 32768, true);
    view.setUint16(dst + 6, hit ? 65535 : 0, true);
  }
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  return {
    bytes,
    sha256,
    payloadOffset,
    gutter,
    storedTileWidth: guarded.storedTileWidth,
    storedTileHeight: guarded.storedTileHeight,
    guardedPixels: guarded.pixels,
  };
}

export function parsePeriodicHeader(bytes: Uint8Array): PeriodicHeader {
  if (bytes.byteLength < 80 || String.fromCharCode(...bytes.subarray(0, 4)) !== 'GCRP') {
    throw new Error('not a GCRP profile');
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint32(4, true);
  if (version !== PERIODIC_PROFILE_VERSION) throw new Error(`expected periodic GCRP/v2, got v${version}`);
  return {
    version,
    profileId: view.getUint32(8, true),
    storedTileWidth: view.getUint32(12, true),
    storedTileHeight: view.getUint32(16, true),
    atlasColumns: view.getUint32(20, true),
    atlasRows: view.getUint32(24, true),
    sliceCount: view.getUint32(28, true),
    texelBytes: view.getUint32(32, true),
    payloadOffset: view.getUint32(36, true),
    mode: view.getUint32(40, true),
    interiorTileWidth: view.getUint32(44, true),
    interiorTileHeight: view.getUint32(48, true),
    gutter: view.getUint32(52, true),
    tile: {
      topH: view.getFloat32(56, true),
      originX: view.getFloat32(60, true),
      originZ: view.getFloat32(64, true),
      sizeX: view.getFloat32(68, true),
      sizeZ: view.getFloat32(72, true),
    },
  };
}

export interface PeriodicTriangleHit {
  t: number;
  normal: Vec3;
  edgeMargin: number;
  triangleId: number;
  copy: PeriodicCopy;
}

interface TriangleIntersection {
  t: number;
  normal: Vec3;
  edgeMargin: number;
}

export interface PeriodicSelectedHitDiagnostic {
  sliceIndex: number;
  pixelX: number;
  pixelY: number;
  origin: Vec3;
  nearest: PeriodicTriangleHit | null;
  second: PeriodicTriangleHit | null;
  alternatives: PeriodicTriangleHit[];
  depthSeparation: number | null;
  normalDot: number | null;
}

export interface PeriodicOwnershipCandidate {
  sliceIndex: number;
  pixelX: number;
  pixelY: number;
  origin: Vec3;
  alternativeRank: number;
  nearest: PeriodicTriangleHit;
  alternative: PeriodicTriangleHit;
  depthSeparation: number;
  normalDot: number;
  depthError: number;
  normalDotError: number;
}

export interface PeriodicProjectedVertex {
  vertexIndex: number;
  world: Vec3;
  rayOriginXZ: { x: number; z: number };
  t: number;
  normalizedDepth: number;
  clip: { x: number; y: number; z: number; w: number };
}

export interface PeriodicProjectedCandidate {
  triangleId: number;
  copy: PeriodicCopy;
  barycentric: [number, number, number];
  minBarycentric: number;
  t: number;
  normal: Vec3;
  depthError: number;
  normalDot: number;
  projectedVertices: PeriodicProjectedVertex[];
}

export interface PeriodicRasterHit extends PeriodicTriangleHit {
  normalizedDepth: number;
  rasterBarycentric: [number, number, number];
}

export interface PeriodicGpuTexelDiagnostic {
  pixelX: number;
  pixelY: number;
  raw: [number, number, number, number];
  hit: boolean;
  t: number | null;
  normal: Vec3 | null;
}

export interface PeriodicGpuMismatchDiagnostic {
  sliceIndex: number;
  pixelX: number;
  pixelY: number;
  origin: Vec3;
  depthError: number;
  normalDot: number;
  gpu: PeriodicGpuTexelDiagnostic;
  adjacentGpuTexels: PeriodicGpuTexelDiagnostic[];
  cpu: PeriodicSelectedHitDiagnostic;
  alternativeComparisons: Array<{
    rank: number;
    triangleId: number;
    copy: PeriodicCopy;
    depthError: number;
    normalDot: number;
  }>;
  nearestSubmission: {
    drawIndex: number;
    copyIndexWithinSlice: number;
    copyUniform: { offsetX: number; offsetZ: number };
    projectedVertices: PeriodicProjectedVertex[];
  } | null;
}

function rayTriangle(
  origin: Vec3,
  direction: Vec3,
  a: Vec3,
  b: Vec3,
  c: Vec3,
  na: Vec3,
  nb: Vec3,
  nc: Vec3,
): TriangleIntersection | null {
  const e1 = { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z };
  const e2 = { x: c.x - a.x, y: c.y - a.y, z: c.z - a.z };
  const p = {
    x: direction.y * e2.z - direction.z * e2.y,
    y: direction.z * e2.x - direction.x * e2.z,
    z: direction.x * e2.y - direction.y * e2.x,
  };
  const determinant = dot(e1, p);
  if (Math.abs(determinant) < 1e-10) return null;
  const inv = 1 / determinant;
  const s = { x: origin.x - a.x, y: origin.y - a.y, z: origin.z - a.z };
  const u = dot(s, p) * inv;
  if (u < 0 || u > 1) return null;
  const q = {
    x: s.y * e1.z - s.z * e1.y,
    y: s.z * e1.x - s.x * e1.z,
    z: s.x * e1.y - s.y * e1.x,
  };
  const v = dot(direction, q) * inv;
  if (v < 0 || u + v > 1) return null;
  const t = dot(e2, q) * inv;
  if (t < 0) return null;
  const w = 1 - u - v;
  let normal = normalized({
    x: na.x * w + nb.x * u + nc.x * v,
    y: na.y * w + nb.y * u + nc.y * v,
    z: na.z * w + nb.z * u + nc.z * v,
  });
  if (dot(normal, direction) > 0) normal = { x: -normal.x, y: -normal.y, z: -normal.z };
  return { t, normal, edgeMargin: Math.min(u, v, w) };
}

function periodicHits(mesh: IndexedMesh, tile: PeriodicTile, slice: PeriodicSlice, origin: Vec3): PeriodicTriangleHit[] {
  const hits: PeriodicTriangleHit[] = [];
  for (const copy of slice.copies) {
    const ox = copy.ix * tile.sizeX;
    const oz = copy.iz * tile.sizeZ;
    for (let tri = 0; tri < mesh.indices.length; tri += 3) {
      const ia = mesh.indices[tri] as number;
      const ib = mesh.indices[tri + 1] as number;
      const ic = mesh.indices[tri + 2] as number;
      const point = (index: number): Vec3 => ({
        x: (mesh.positions[index * 3] as number) + ox,
        y: mesh.positions[index * 3 + 1] as number,
        z: (mesh.positions[index * 3 + 2] as number) + oz,
      });
      const normal = (index: number): Vec3 => ({
        x: mesh.normals[index * 3] as number,
        y: mesh.normals[index * 3 + 1] as number,
        z: mesh.normals[index * 3 + 2] as number,
      });
      const hit = rayTriangle(origin, slice.direction, point(ia), point(ib), point(ic), normal(ia), normal(ib), normal(ic));
      if (hit) hits.push({ ...hit, triangleId: tri / 3, copy });
    }
  }
  hits.sort((a, b) => a.t - b.t || a.triangleId - b.triangleId || a.copy.ix - b.copy.ix || a.copy.iz - b.copy.iz);
  return hits;
}

function nearestPeriodicHit(mesh: IndexedMesh, tile: PeriodicTile, slice: PeriodicSlice, origin: Vec3): PeriodicTriangleHit | null {
  return periodicHits(mesh, tile, slice, origin)[0] ?? null;
}

/**
 * Identity-bearing CPU diagnostics at the exact pixel sequence used by the parked
 * WebGPU validation. `second` is the literal second ray/triangle intersection,
 * including a shared-edge duplicate if the selected center lies exactly on one.
 */
export function enumeratePeriodicSelectedHits(
  mesh: IndexedMesh,
  tile: PeriodicTile,
  slices: PeriodicSlice[],
  tileWidth: number,
  tileHeight: number,
  sampleStep = 14,
): PeriodicSelectedHitDiagnostic[] {
  const records: PeriodicSelectedHitDiagnostic[] = [];
  slices.forEach((slice, sliceIndex) => {
    for (let pixelY = 3; pixelY < tileHeight - 3; pixelY += sampleStep) {
      for (let pixelX = 3; pixelX < tileWidth - 3; pixelX += sampleStep) {
        records.push(enumeratePeriodicPixelHit(mesh, tile, slice, sliceIndex, pixelX, pixelY, tileWidth, tileHeight));
      }
    }
  });
  return records;
}

export function enumeratePeriodicPixelHit(
  mesh: IndexedMesh,
  tile: PeriodicTile,
  slice: PeriodicSlice,
  sliceIndex: number,
  pixelX: number,
  pixelY: number,
  tileWidth: number,
  tileHeight: number,
): PeriodicSelectedHitDiagnostic {
  const origin = {
    x: tile.originX + ((pixelX + 0.5) / tileWidth) * tile.sizeX,
    y: tile.topH,
    z: tile.originZ + (1 - (pixelY + 0.5) / tileHeight) * tile.sizeZ,
  };
  const hits = periodicHits(mesh, tile, slice, origin);
  const nearest = hits[0] ?? null;
  const second = hits[1] ?? null;
  return {
    sliceIndex,
    pixelX,
    pixelY,
    origin,
    nearest,
    second,
    alternatives: hits.slice(1),
    depthSeparation: nearest && second ? second.t - nearest.t : null,
    normalDot: nearest && second ? dot(nearest.normal, second.normal) : null,
  };
}

export function projectPeriodicHitTriangle(
  mesh: IndexedMesh,
  tile: PeriodicTile,
  slice: PeriodicSlice,
  hit: PeriodicTriangleHit,
): PeriodicProjectedVertex[] {
  const triBase = hit.triangleId * 3;
  return [0, 1, 2].map((corner) => {
    const vertexIndex = mesh.indices[triBase + corner] as number;
    const world = {
      x: (mesh.positions[vertexIndex * 3] as number) + hit.copy.ix * tile.sizeX,
      y: mesh.positions[vertexIndex * 3 + 1] as number,
      z: (mesh.positions[vertexIndex * 3 + 2] as number) + hit.copy.iz * tile.sizeZ,
    };
    const t = (world.y - tile.topH) / slice.direction.y;
    const rayOriginXZ = {
      x: world.x - slice.direction.x * t,
      z: world.z - slice.direction.z * t,
    };
    const u = (rayOriginXZ.x - tile.originX) / tile.sizeX;
    const v = (rayOriginXZ.z - tile.originZ) / tile.sizeZ;
    const normalizedDepth = (t - slice.depthMin) / (slice.depthMax - slice.depthMin);
    return {
      vertexIndex,
      world,
      rayOriginXZ,
      t,
      normalizedDepth,
      clip: { x: u * 2 - 1, y: v * 2 - 1, z: normalizedDepth, w: 1 },
    };
  });
}

/** Rank every submitted primitive by how closely its screen-space interpolation matches a GPU texel. */
export function searchProjectedTriangleCandidates(
  mesh: IndexedMesh,
  tile: PeriodicTile,
  slice: PeriodicSlice,
  pixelX: number,
  pixelY: number,
  tileWidth: number,
  tileHeight: number,
  targetT: number,
  targetNormal: Vec3,
  limit = 12,
): PeriodicProjectedCandidate[] {
  const targetU = (pixelX + 0.5) / tileWidth;
  const targetV = 1 - (pixelY + 0.5) / tileHeight;
  const candidates: PeriodicProjectedCandidate[] = [];
  for (const copy of slice.copies) {
    for (let triangleId = 0; triangleId < mesh.indices.length / 3; triangleId++) {
      const hitIdentity: PeriodicTriangleHit = {
        t: 0,
        normal: { x: 0, y: 1, z: 0 },
        edgeMargin: 0,
        triangleId,
        copy,
      };
      const projectedVertices = projectPeriodicHitTriangle(mesh, tile, slice, hitIdentity);
      const uv = projectedVertices.map((vertex) => ({
        u: vertex.clip.x * 0.5 + 0.5,
        v: vertex.clip.y * 0.5 + 0.5,
      }));
      const a = uv[0]!;
      const b = uv[1]!;
      const c = uv[2]!;
      const denominator = (b.v - c.v) * (a.u - c.u) + (c.u - b.u) * (a.v - c.v);
      if (Math.abs(denominator) < 1e-14) continue;
      const wa = ((b.v - c.v) * (targetU - c.u) + (c.u - b.u) * (targetV - c.v)) / denominator;
      const wb = ((c.v - a.v) * (targetU - c.u) + (a.u - c.u) * (targetV - c.v)) / denominator;
      const wc = 1 - wa - wb;
      const weights = [wa, wb, wc] as const;
      const t = projectedVertices.reduce((sum, vertex, index) => sum + vertex.t * weights[index]!, 0);
      const vertexNormal = (corner: number): Vec3 => {
        const vertexIndex = projectedVertices[corner]!.vertexIndex;
        return {
          x: mesh.normals[vertexIndex * 3] as number,
          y: mesh.normals[vertexIndex * 3 + 1] as number,
          z: mesh.normals[vertexIndex * 3 + 2] as number,
        };
      };
      const normals = [vertexNormal(0), vertexNormal(1), vertexNormal(2)];
      let normal = normalized({
        x: normals.reduce((sum, value, index) => sum + value.x * weights[index]!, 0),
        y: normals.reduce((sum, value, index) => sum + value.y * weights[index]!, 0),
        z: normals.reduce((sum, value, index) => sum + value.z * weights[index]!, 0),
      });
      if (dot(normal, slice.direction) > 0) normal = { x: -normal.x, y: -normal.y, z: -normal.z };
      candidates.push({
        triangleId,
        copy,
        barycentric: [wa, wb, wc],
        minBarycentric: Math.min(wa, wb, wc),
        t,
        normal,
        depthError: Math.abs(t - targetT),
        normalDot: dot(normal, targetNormal),
        projectedVertices,
      });
    }
  }
  return candidates
    .sort((left, right) =>
      (left.depthError + Math.abs(1 - left.normalDot) * 0.1 + Math.max(0, -left.minBarycentric)) -
      (right.depthError + Math.abs(1 - right.normalDot) * 0.1 + Math.max(0, -right.minBarycentric)))
    .slice(0, limit);
}

/**
 * CPU transcription of fixed-function triangle ownership. WebGPU rasterization
 * has implementation-defined subpixel precision; Apple Metal exposes the usual
 * 8-bit fractional window grid. Coverage uses snapped window vertices while the
 * interpolants remain the submitted f32 values.
 */
export function enumeratePeriodicRasterPixelHits(
  mesh: IndexedMesh,
  tile: PeriodicTile,
  slice: PeriodicSlice,
  pixelX: number,
  pixelY: number,
  tileWidth: number,
  tileHeight: number,
  subpixelBits = 8,
): PeriodicRasterHit[] {
  const f32 = (value: number): number => Math.fround(value);
  const snapScale = 2 ** subpixelBits;
  const snap = (value: number): number => Math.round(value * snapScale) / snapScale;
  const target = { x: pixelX + 0.5, y: pixelY + 0.5 };
  const direction = { x: f32(slice.direction.x), y: f32(slice.direction.y), z: f32(slice.direction.z) };
  const topH = f32(tile.topH);
  const originX = f32(tile.originX);
  const originZ = f32(tile.originZ);
  const sizeX = f32(tile.sizeX);
  const sizeZ = f32(tile.sizeZ);
  const depthMin = f32(slice.depthMin);
  const depthMax = f32(slice.depthMax);
  const depthSpan = f32(depthMax - depthMin);
  const hits: PeriodicRasterHit[] = [];
  for (const copy of slice.copies) {
    const copyX = f32(copy.ix * sizeX);
    const copyZ = f32(copy.iz * sizeZ);
    for (let triangleId = 0; triangleId < mesh.indices.length / 3; triangleId++) {
      const vertices = [0, 1, 2].map((corner) => {
        const vertexIndex = mesh.indices[triangleId * 3 + corner] as number;
        const x = f32(f32(mesh.positions[vertexIndex * 3] as number) + copyX);
        const y = f32(mesh.positions[vertexIndex * 3 + 1] as number);
        const z = f32(f32(mesh.positions[vertexIndex * 3 + 2] as number) + copyZ);
        const t = f32(f32(y - topH) / direction.y);
        const rayX = f32(x - f32(direction.x * t));
        const rayZ = f32(z - f32(direction.z * t));
        const u = f32(f32(rayX - originX) / sizeX);
        const v = f32(f32(rayZ - originZ) / sizeZ);
        const normalizedDepth = f32(f32(t - depthMin) / depthSpan);
        return {
          vertexIndex,
          screen: { x: snap(f32(u * tileWidth)), y: snap(f32((1 - v) * tileHeight)) },
          normalizedDepth,
          normal: {
            x: f32(mesh.normals[vertexIndex * 3] as number),
            y: f32(mesh.normals[vertexIndex * 3 + 1] as number),
            z: f32(mesh.normals[vertexIndex * 3 + 2] as number),
          },
        };
      });
      const a = vertices[0]!;
      const b = vertices[1]!;
      const c = vertices[2]!;
      const denominator = (b.screen.y - c.screen.y) * (a.screen.x - c.screen.x) +
        (c.screen.x - b.screen.x) * (a.screen.y - c.screen.y);
      if (Math.abs(denominator) < 1e-14) continue;
      const wa = ((b.screen.y - c.screen.y) * (target.x - c.screen.x) +
        (c.screen.x - b.screen.x) * (target.y - c.screen.y)) / denominator;
      const wb = ((c.screen.y - a.screen.y) * (target.x - c.screen.x) +
        (a.screen.x - c.screen.x) * (target.y - c.screen.y)) / denominator;
      const wc = 1 - wa - wb;
      const barycentric = [wa, wb, wc] as [number, number, number];
      const edgeMargin = Math.min(...barycentric);
      if (edgeMargin < -1e-12) continue;
      const normalizedDepth = vertices.reduce((sum, vertex, index) =>
        sum + vertex.normalizedDepth * barycentric[index]!, 0);
      if (normalizedDepth < 0 || normalizedDepth > 1) continue;
      let normal = normalized({
        x: vertices.reduce((sum, vertex, index) => sum + vertex.normal.x * barycentric[index]!, 0),
        y: vertices.reduce((sum, vertex, index) => sum + vertex.normal.y * barycentric[index]!, 0),
        z: vertices.reduce((sum, vertex, index) => sum + vertex.normal.z * barycentric[index]!, 0),
      });
      if (dot(normal, direction) > 0) normal = { x: -normal.x, y: -normal.y, z: -normal.z };
      hits.push({
        triangleId,
        copy,
        t: depthMin + normalizedDepth * depthSpan,
        normal,
        edgeMargin,
        normalizedDepth,
        rasterBarycentric: barycentric,
      });
    }
  }
  return hits.sort((left, right) => left.normalizedDepth - right.normalizedDepth || left.triangleId - right.triangleId);
}

function gpuTexel(result: PeriodicBakeResult, sliceIndex: number, pixelX: number, pixelY: number): PeriodicGpuTexelDiagnostic {
  const atlasWidth = result.tileWidth * result.atlasColumns;
  const tileX = (sliceIndex % result.atlasColumns) * result.tileWidth;
  const tileY = Math.floor(sliceIndex / result.atlasColumns) * result.tileHeight;
  const index = ((tileY + pixelY) * atlasWidth + tileX + pixelX) * 4;
  const raw: [number, number, number, number] = [
    result.pixels[index] as number,
    result.pixels[index + 1] as number,
    result.pixels[index + 2] as number,
    result.pixels[index + 3] as number,
  ];
  const hit = raw[3] > 0.5;
  const slice = result.slices[sliceIndex] as PeriodicSlice;
  return {
    pixelX,
    pixelY,
    raw,
    hit,
    t: hit ? slice.depthMin + raw[0] * (slice.depthMax - slice.depthMin) : null,
    normal: hit ? decodeOct(raw[1], raw[2]) : null,
  };
}

/** Full identity capture for every strict selected pixel that fails the original acceptance. */
export function diagnosePeriodicGpuMismatches(result: PeriodicBakeResult, mesh: IndexedMesh, sampleStep = 14): PeriodicGpuMismatchDiagnostic[] {
  const records = enumeratePeriodicSelectedHits(mesh, result.tile, result.slices, result.tileWidth, result.tileHeight, sampleStep);
  const mismatches: PeriodicGpuMismatchDiagnostic[] = [];
  for (const cpu of records) {
    if (!cpu.nearest || cpu.nearest.edgeMargin < 0.03) continue;
    const gpu = gpuTexel(result, cpu.sliceIndex, cpu.pixelX, cpu.pixelY);
    const slice = result.slices[cpu.sliceIndex] as PeriodicSlice;
    const raster = enumeratePeriodicRasterPixelHits(
      mesh,
      result.tile,
      slice,
      cpu.pixelX,
      cpu.pixelY,
      result.tileWidth,
      result.tileHeight,
    )[0] ?? null;
    const expected = raster ?? cpu.nearest;
    const depthError = gpu.t === null ? Number.POSITIVE_INFINITY : Math.abs(gpu.t - expected.t);
    const normalDot = gpu.normal === null ? -1 : dot(expected.normal, gpu.normal);
    if (gpu.hit && depthError <= 2e-5 && normalDot >= 0.999) continue;
    const copyIndexWithinSlice = slice.copies.findIndex((copy) =>
      copy.ix === expected.copy.ix && copy.iz === expected.copy.iz);
    const priorDraws = result.slices.slice(0, cpu.sliceIndex).reduce((sum, prior) => sum + prior.copies.length, 0);
    const offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]] as const;
    mismatches.push({
      sliceIndex: cpu.sliceIndex,
      pixelX: cpu.pixelX,
      pixelY: cpu.pixelY,
      origin: cpu.origin,
      depthError,
      normalDot,
      gpu,
      adjacentGpuTexels: offsets.map(([dx, dy]) => gpuTexel(result, cpu.sliceIndex, cpu.pixelX + dx, cpu.pixelY + dy)),
      cpu,
      alternativeComparisons: cpu.alternatives.map((alternative, index) => ({
        rank: index + 2,
        triangleId: alternative.triangleId,
        copy: alternative.copy,
        depthError: gpu.t === null ? Number.POSITIVE_INFINITY : Math.abs(gpu.t - alternative.t),
        normalDot: gpu.normal === null ? -1 : dot(alternative.normal, gpu.normal),
      })),
      nearestSubmission: copyIndexWithinSlice < 0 ? null : {
        drawIndex: priorDraws + copyIndexWithinSlice,
        copyIndexWithinSlice,
        copyUniform: {
          offsetX: expected.copy.ix * result.tile.sizeX,
          offsetZ: expected.copy.iz * result.tile.sizeZ,
        },
        projectedVertices: projectPeriodicHitTriangle(mesh, result.tile, slice, expected),
      },
    });
  }
  return mismatches.sort((a, b) => b.depthError - a.depthError || a.normalDot - b.normalDot);
}

export function closestOwnershipCandidates(
  records: PeriodicSelectedHitDiagnostic[],
  targetDepthSeparation: number,
  targetNormalDot: number,
  limit = 8,
): PeriodicOwnershipCandidate[] {
  return records
    .flatMap((record): PeriodicOwnershipCandidate[] => {
      if (!record.nearest) return [];
      return record.alternatives.map((alternative, index) => {
        const depthSeparation = alternative.t - record.nearest!.t;
        const normalDot = dot(record.nearest!.normal, alternative.normal);
        return {
          sliceIndex: record.sliceIndex,
          pixelX: record.pixelX,
          pixelY: record.pixelY,
          origin: record.origin,
          alternativeRank: index + 2,
          nearest: record.nearest as PeriodicTriangleHit,
          alternative,
          depthSeparation,
          normalDot,
          depthError: Math.abs(depthSeparation - targetDepthSeparation),
          normalDotError: Math.abs(normalDot - targetNormalDot),
        };
      });
    })
    .sort((a, b) =>
      (a.depthError / Math.max(targetDepthSeparation, 1e-9) + a.normalDotError) -
      (b.depthError / Math.max(targetDepthSeparation, 1e-9) + b.normalDotError))
    .slice(0, limit);
}

/** Selected-pixel CPU ray/triangle oracle for the periodic GPU result. */
export function validatePeriodicBake(result: PeriodicBakeResult, mesh: IndexedMesh, sampleStep = 12): PeriodicValidationStats {
  const atlasWidth = result.tileWidth * result.atlasColumns;
  let selected = 0;
  let skippedTriangleBoundary = 0;
  let rasterOwnerDiffersFromAnalytic = 0;
  let comparedHits = 0;
  let comparedMisses = 0;
  let hitMismatch = 0;
  let depthSq = 0;
  let maxDepthTError = 0;
  let normalSum = 0;
  let minNormalDot = 1;
  result.slices.forEach((slice, sliceIndex) => {
    const tileX = (sliceIndex % result.atlasColumns) * result.tileWidth;
    const tileY = Math.floor(sliceIndex / result.atlasColumns) * result.tileHeight;
    for (let py = 3; py < result.tileHeight - 3; py += sampleStep) {
      for (let px = 3; px < result.tileWidth - 3; px += sampleStep) {
        selected++;
        const origin = {
          x: result.tile.originX + ((px + 0.5) / result.tileWidth) * result.tile.sizeX,
          y: result.tile.topH,
          z: result.tile.originZ + (1 - (py + 0.5) / result.tileHeight) * result.tile.sizeZ,
        };
        const cpu = nearestPeriodicHit(mesh, result.tile, slice, origin);
        const index = ((tileY + py) * atlasWidth + tileX + px) * 4;
        const gpuHit = (result.pixels[index + 3] as number) > 0.5;
        if (!cpu) {
          if (gpuHit) hitMismatch++;
          else comparedMisses++;
          continue;
        }
        // Only validate strict triangle interiors. The fixture deliberately overlaps
        // independent cushions, so points close to a projected triangle boundary can
        // change owner under the rasterizer's specified top-left fill convention.
        if (cpu.edgeMargin < 0.03) {
          skippedTriangleBoundary++;
          continue;
        }
        const raster = enumeratePeriodicRasterPixelHits(
          mesh,
          result.tile,
          slice,
          px,
          py,
          result.tileWidth,
          result.tileHeight,
        )[0] ?? null;
        const expected = raster ?? cpu;
        if (raster && (raster.triangleId !== cpu.triangleId || raster.copy.ix !== cpu.copy.ix || raster.copy.iz !== cpu.copy.iz)) {
          rasterOwnerDiffersFromAnalytic++;
        }
        if (!gpuHit) {
          hitMismatch++;
          continue;
        }
        const gpuT = slice.depthMin + (result.pixels[index] as number) * (slice.depthMax - slice.depthMin);
        const depthError = Math.abs(gpuT - expected.t);
        const gpuNormal = decodeOct(result.pixels[index + 1] as number, result.pixels[index + 2] as number);
        const nDot = dot(expected.normal, gpuNormal);
        depthSq += depthError * depthError;
        maxDepthTError = Math.max(maxDepthTError, depthError);
        normalSum += nDot;
        minNormalDot = Math.min(minNormalDot, nDot);
        comparedHits++;
      }
    }
  });
  return {
    selected,
    skippedTriangleBoundary,
    rasterOwnerDiffersFromAnalytic,
    comparedHits,
    comparedMisses,
    hitMismatch,
    maxDepthTError,
    rmsDepthTError: comparedHits > 0 ? Math.sqrt(depthSq / comparedHits) : Number.POSITIVE_INFINITY,
    minNormalDot: comparedHits > 0 ? minNormalDot : -1,
    meanNormalDot: comparedHits > 0 ? normalSum / comparedHits : -1,
  };
}
