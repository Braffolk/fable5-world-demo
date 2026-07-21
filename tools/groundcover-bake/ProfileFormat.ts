import { createHash } from 'node:crypto';

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface IndexedMesh {
  positions: number[];
  normals: number[];
  /** Optional linear-sRGB authoring color per vertex (rgb triples in 0..1).
   *  Direct mesh QA and material-aware bakers consume this exact attribute. */
  colors?: number[];
  indices: number[];
}

export interface EllipsoidFixture {
  center: Vec3;
  radii: Vec3;
  sectors: number;
  stacks: number;
}

export interface ProjectionSlice {
  direction: Vec3;
  axisU: Vec3;
  axisV: Vec3;
  bounds: {
    uMin: number;
    uMax: number;
    vMin: number;
    vMax: number;
    depthMin: number;
    depthMax: number;
  };
}

export interface BakeResult {
  tileWidth: number;
  tileHeight: number;
  atlasColumns: number;
  atlasRows: number;
  slices: ProjectionSlice[];
  /** Atlas-order RGBA32F: normalized first depth, oct X/Y in [0,1], hit mask. */
  pixels: number[];
  adapter: string;
}

export interface PackedProfile {
  bytes: Uint8Array;
  sha256: string;
  payloadOffset: number;
}

const MAGIC = 'GCRP';
export const PROFILE_VERSION = 1;
export const PROFILE_TEXEL_BYTES = 8;

function dot(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

function length(v: Vec3): number {
  return Math.hypot(v.x, v.y, v.z);
}

function normalized(v: Vec3): Vec3 {
  const l = length(v);
  if (!(l > 0)) throw new Error('cannot normalize a zero vector');
  return { x: v.x / l, y: v.y / l, z: v.z / l };
}

export function makeDirection(azimuthDeg: number, elevationDeg: number): Vec3 {
  const az = (azimuthDeg * Math.PI) / 180;
  const el = (elevationDeg * Math.PI) / 180;
  return normalized({
    x: Math.cos(el) * Math.cos(az),
    y: -Math.sin(el),
    z: Math.cos(el) * Math.sin(az),
  });
}

/** Stable orthonormal ray-origin frame. Rays advance from depthMin along direction. */
export function projectionFrame(directionInput: Vec3): Pick<ProjectionSlice, 'direction' | 'axisU' | 'axisV'> {
  const direction = normalized(directionInput);
  const helper = Math.abs(direction.y) < 0.92 ? { x: 0, y: 1, z: 0 } : { x: 0, y: 0, z: 1 };
  const axisU = normalized(cross(helper, direction));
  const axisV = normalized(cross(direction, axisU));
  return { direction, axisU, axisV };
}

export function validateIndexedMesh(mesh: IndexedMesh): void {
  if (mesh.positions.length === 0 || mesh.positions.length % 3 !== 0) {
    throw new Error('positions must contain xyz triples');
  }
  if (mesh.normals.length !== mesh.positions.length) {
    throw new Error('normals must match positions');
  }
  if (mesh.colors !== undefined && mesh.colors.length !== mesh.positions.length) {
    throw new Error('colors must match positions');
  }
  if (mesh.indices.length === 0 || mesh.indices.length % 3 !== 0) {
    throw new Error('indices must contain triangle triples');
  }
  const vertexCount = mesh.positions.length / 3;
  for (const i of mesh.indices) {
    if (!Number.isInteger(i) || i < 0 || i >= vertexCount) {
      throw new Error(`index ${i} is outside 0..${vertexCount - 1}`);
    }
  }
  for (const value of [...mesh.positions, ...mesh.normals, ...(mesh.colors ?? [])]) {
    if (!Number.isFinite(value)) throw new Error('mesh attributes must be finite');
  }
  if (mesh.colors?.some((value) => value < 0 || value > 1)) {
    throw new Error('mesh colors must be normalized to 0..1');
  }
}

export function makeEllipsoidFixture(config: EllipsoidFixture): IndexedMesh {
  const { center: c, radii: r, sectors, stacks } = config;
  if (!(r.x > 0 && r.y > 0 && r.z > 0)) throw new Error('ellipsoid radii must be positive');
  if (sectors < 8 || stacks < 4) throw new Error('ellipsoid tessellation is too small');
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];
  for (let iy = 0; iy <= stacks; iy++) {
    const phi = (iy / stacks) * Math.PI;
    const sp = Math.sin(phi);
    const cp = Math.cos(phi);
    for (let ix = 0; ix <= sectors; ix++) {
      const theta = (ix / sectors) * Math.PI * 2;
      const ct = Math.cos(theta);
      const st = Math.sin(theta);
      const lx = sp * ct;
      const ly = cp;
      const lz = sp * st;
      positions.push(c.x + r.x * lx, c.y + r.y * ly, c.z + r.z * lz);
      const n = normalized({ x: lx / r.x, y: ly / r.y, z: lz / r.z });
      normals.push(n.x, n.y, n.z);
    }
  }
  const stride = sectors + 1;
  for (let iy = 0; iy < stacks; iy++) {
    for (let ix = 0; ix < sectors; ix++) {
      const a = iy * stride + ix;
      const b = a + stride;
      if (iy !== 0) indices.push(a, b, a + 1);
      if (iy !== stacks - 1) indices.push(a + 1, b, b + 1);
    }
  }
  const mesh = { positions, normals, indices };
  validateIndexedMesh(mesh);
  return mesh;
}

export function makeProjectionSlice(mesh: IndexedMesh, direction: Vec3, paddingFraction = 0.04): ProjectionSlice {
  validateIndexedMesh(mesh);
  const frame = projectionFrame(direction);
  let uMin = Number.POSITIVE_INFINITY;
  let uMax = Number.NEGATIVE_INFINITY;
  let vMin = Number.POSITIVE_INFINITY;
  let vMax = Number.NEGATIVE_INFINITY;
  let depthMin = Number.POSITIVE_INFINITY;
  let depthMax = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < mesh.positions.length; i += 3) {
    const p = {
      x: mesh.positions[i] as number,
      y: mesh.positions[i + 1] as number,
      z: mesh.positions[i + 2] as number,
    };
    const u = dot(p, frame.axisU);
    const v = dot(p, frame.axisV);
    const d = dot(p, frame.direction);
    uMin = Math.min(uMin, u);
    uMax = Math.max(uMax, u);
    vMin = Math.min(vMin, v);
    vMax = Math.max(vMax, v);
    depthMin = Math.min(depthMin, d);
    depthMax = Math.max(depthMax, d);
  }
  const pad = (lo: number, hi: number): [number, number] => {
    const amount = Math.max((hi - lo) * paddingFraction, 1e-5);
    return [lo - amount, hi + amount];
  };
  [uMin, uMax] = pad(uMin, uMax);
  [vMin, vMax] = pad(vMin, vMax);
  [depthMin, depthMax] = pad(depthMin, depthMax);
  return { ...frame, bounds: { uMin, uMax, vMin, vMax, depthMin, depthMax } };
}

function clampU16(v: number): number {
  return Math.max(0, Math.min(65535, Math.round(v * 65535)));
}

/**
 * Little-endian v1 container. Header is 64 bytes; each slice has 16 float32s
 * (direction, U, V, six projection bounds, one reserved). Payload is atlas-order
 * RGBA16: normalized first depth, oct-normal X/Y, hit mask.
 */
export function packProfile(result: BakeResult, profileId: number): PackedProfile {
  const atlasWidth = result.tileWidth * result.atlasColumns;
  const atlasHeight = result.tileHeight * result.atlasRows;
  if (result.slices.length > result.atlasColumns * result.atlasRows) throw new Error('atlas does not fit slices');
  if (result.pixels.length !== atlasWidth * atlasHeight * 4) throw new Error('pixel payload has the wrong size');
  const headerBytes = 64;
  const sliceBytes = 64;
  const payloadOffset = headerBytes + result.slices.length * sliceBytes;
  const bytes = new Uint8Array(payloadOffset + atlasWidth * atlasHeight * PROFILE_TEXEL_BYTES);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < MAGIC.length; i++) bytes[i] = MAGIC.charCodeAt(i);
  view.setUint32(4, PROFILE_VERSION, true);
  view.setUint32(8, profileId >>> 0, true);
  view.setUint32(12, result.tileWidth, true);
  view.setUint32(16, result.tileHeight, true);
  view.setUint32(20, result.atlasColumns, true);
  view.setUint32(24, result.atlasRows, true);
  view.setUint32(28, result.slices.length, true);
  view.setUint32(32, PROFILE_TEXEL_BYTES, true);
  view.setUint32(36, payloadOffset, true);
  result.slices.forEach((slice, i) => {
    const b = slice.bounds;
    const values = [
      slice.direction.x, slice.direction.y, slice.direction.z,
      slice.axisU.x, slice.axisU.y, slice.axisU.z,
      slice.axisV.x, slice.axisV.y, slice.axisV.z,
      b.uMin, b.uMax, b.vMin, b.vMax, b.depthMin, b.depthMax, 0,
    ];
    values.forEach((value, j) => view.setFloat32(headerBytes + i * sliceBytes + j * 4, value, true));
  });
  for (let texel = 0; texel < atlasWidth * atlasHeight; texel++) {
    const src = texel * 4;
    const dst = payloadOffset + texel * PROFILE_TEXEL_BYTES;
    const hit = (result.pixels[src + 3] as number) > 0.5;
    view.setUint16(dst, hit ? Math.min(65534, clampU16(result.pixels[src] as number)) : 65535, true);
    view.setUint16(dst + 2, hit ? clampU16(result.pixels[src + 1] as number) : 32768, true);
    view.setUint16(dst + 4, hit ? clampU16(result.pixels[src + 2] as number) : 32768, true);
    view.setUint16(dst + 6, hit ? 65535 : 0, true);
  }
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  return { bytes, sha256, payloadOffset };
}

export function decodeOct(x01: number, y01: number): Vec3 {
  const x = x01 * 2 - 1;
  const y = y01 * 2 - 1;
  let n = { x, y, z: 1 - Math.abs(x) - Math.abs(y) };
  if (n.z < 0) {
    const oldX = n.x;
    n.x = (1 - Math.abs(n.y)) * (oldX < 0 ? -1 : 1);
    n.y = (1 - Math.abs(oldX)) * (n.y < 0 ? -1 : 1);
  }
  return normalized(n);
}

export interface ValidationStats {
  compared: number;
  missingInterior: number;
  falseInterior: number;
  rmsDepth: number;
  maxDepth: number;
  meanNormalDot: number;
  minNormalDot: number;
}

/** Compare raster first hits against the exact implicit ellipsoid, away from its silhouette. */
export function validateEllipsoidBake(result: BakeResult, fixture: EllipsoidFixture): ValidationStats {
  const atlasWidth = result.tileWidth * result.atlasColumns;
  let compared = 0;
  let missingInterior = 0;
  let falseInterior = 0;
  let depthSq = 0;
  let maxDepth = 0;
  let normalDotSum = 0;
  let minNormalDot = 1;
  result.slices.forEach((slice, sliceIndex) => {
    const tileX = (sliceIndex % result.atlasColumns) * result.tileWidth;
    const tileY = Math.floor(sliceIndex / result.atlasColumns) * result.tileHeight;
    const b = slice.bounds;
    const depthSpan = b.depthMax - b.depthMin;
    for (let py = 0; py < result.tileHeight; py++) {
      for (let px = 0; px < result.tileWidth; px++) {
        const u = b.uMin + ((px + 0.5) / result.tileWidth) * (b.uMax - b.uMin);
        const v = b.vMax - ((py + 0.5) / result.tileHeight) * (b.vMax - b.vMin);
        const origin = {
          x: slice.axisU.x * u + slice.axisV.x * v + slice.direction.x * b.depthMin,
          y: slice.axisU.y * u + slice.axisV.y * v + slice.direction.y * b.depthMin,
          z: slice.axisU.z * u + slice.axisV.z * v + slice.direction.z * b.depthMin,
        };
        const oc = {
          x: origin.x - fixture.center.x,
          y: origin.y - fixture.center.y,
          z: origin.z - fixture.center.z,
        };
        const invR2 = {
          x: 1 / (fixture.radii.x * fixture.radii.x),
          y: 1 / (fixture.radii.y * fixture.radii.y),
          z: 1 / (fixture.radii.z * fixture.radii.z),
        };
        const d = slice.direction;
        const qa = d.x * d.x * invR2.x + d.y * d.y * invR2.y + d.z * d.z * invR2.z;
        const qb = 2 * (oc.x * d.x * invR2.x + oc.y * d.y * invR2.y + oc.z * d.z * invR2.z);
        const qc = oc.x * oc.x * invR2.x + oc.y * oc.y * invR2.y + oc.z * oc.z * invR2.z - 1;
        const disc = qb * qb - 4 * qa * qc;
        const atlasIndex = ((tileY + py) * atlasWidth + tileX + px) * 4;
        const gpuHit = (result.pixels[atlasIndex + 3] as number) > 0.5;
        if (disc < 0) {
          if (gpuHit) falseInterior++;
          continue;
        }
        const t = (-qb - Math.sqrt(disc)) / (2 * qa);
        if (t < 0) continue;
        const hit = { x: origin.x + d.x * t, y: origin.y + d.y * t, z: origin.z + d.z * t };
        const exactNormal = normalized({
          x: (hit.x - fixture.center.x) * invR2.x,
          y: (hit.y - fixture.center.y) * invR2.y,
          z: (hit.z - fixture.center.z) * invR2.z,
        });
        const incidence = Math.abs(dot(exactNormal, d));
        if (incidence < 0.25) continue;
        if (!gpuHit) {
          missingInterior++;
          continue;
        }
        const expectedDepth = t / depthSpan;
        const depthError = Math.abs((result.pixels[atlasIndex] as number) - expectedDepth);
        const gpuNormal = decodeOct(result.pixels[atlasIndex + 1] as number, result.pixels[atlasIndex + 2] as number);
        const nDot = dot(exactNormal, gpuNormal);
        depthSq += depthError * depthError;
        maxDepth = Math.max(maxDepth, depthError);
        normalDotSum += nDot;
        minNormalDot = Math.min(minNormalDot, nDot);
        compared++;
      }
    }
  });
  return {
    compared,
    missingInterior,
    falseInterior,
    rmsDepth: compared > 0 ? Math.sqrt(depthSq / compared) : Number.POSITIVE_INFINITY,
    maxDepth,
    meanNormalDot: compared > 0 ? normalDotSum / compared : -1,
    minNormalDot: compared > 0 ? minNormalDot : -1,
  };
}
