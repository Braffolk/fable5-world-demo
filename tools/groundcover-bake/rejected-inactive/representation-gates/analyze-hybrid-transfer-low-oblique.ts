/**
 * One bounded actual-source gate for the rejected rank-four/four-read hybrid
 * transfer candidate. It traces a persistent supersampled 18-degree camera path through
 * the exact isolated Calamagrostis source, writes well-conditioned complete
 * (A,m,C,N) truth and
 * source-event labels, audits whether that path constrains the declared factor
 * texels, and records the mathematical/runtime-interface reasons no fit is
 * allowed.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import sharp from 'sharp';
import { makeCalamagrostisCanescensBandlimitedSource } from '../../EstonianGraminoids';
import {
  TriangleBvh,
  type CensusVec3,
  type DecodedOwnedProfileGeometry,
} from '../../ExteriorClosureCensus';
import { periodicNearestSuccessor, type OriginAwareTruthHit } from '../../OriginAwareRayTruth';
import type { IndexedMesh } from '../../ProfileFormat';

interface Args { [key: string]: string | boolean }
interface EventAttributes { color: CensusVec3; normal: CensusVec3 }
interface ExitRecord { face: number; u: number; v: number; distance: number }
interface AddressAudit {
  trainRecords: number;
  validationRecords: number;
  activeFactorScalars: number;
  trainOutputEquations: number;
  equationToActiveParameterRatio: number;
  validationAllCornersSeenFraction: number;
  validationCornerCoverage: Record<string, number>;
  uniqueJointTrainSignatures: number;
  uniqueJointValidationSignatures: number;
  decision: 'fit-identifiable' | 'underdetermined';
  reasons: string[];
}

const FIELDS = 27;
const MICRO_RECORD_BYTES = 32;
const HORIZON = 155;
const LOCAL_X = 208;
const LOCAL_Z = 208;
const LOCAL_Y = 96;
const BOUNDARY_U = 256;
const BOUNDARY_V = 256;
const CUBE = 64;
const HEAD_REJECTION_REASONS = [
  'the proposed affine head h(b,r)=p(b)+c*r has identically zero mixed cross-difference between boundary state b and remaining distance r',
  'exact finite-horizon suffix transfer T(b,r)=exp(-kappa(b)*r) has nonzero mixed cross-difference whenever two boundary states have different kappa, so the affine head cannot represent it',
  'the current live grass interface resolves opaque depth/body plus normal/tip and writes final alpha=1; fractional A cannot survive without a new compositing or stochastic-coverage contract',
] as const;

function parseArgs(argv: readonly string[]): Args {
  const result: Args = {};
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index]!;
    if (!arg.startsWith('--')) continue;
    const value = argv[index + 1];
    if (value !== undefined && !value.startsWith('--')) {
      result[arg.slice(2)] = value;
      index++;
    } else result[arg.slice(2)] = true;
  }
  return result;
}

function numberArg(value: string | boolean | undefined, fallback: number, label: string): number {
  const parsed = Number(typeof value === 'string' ? value : fallback);
  if (!Number.isFinite(parsed)) throw new Error(`${label} must be finite`);
  return parsed;
}

function sha(value: Uint8Array | string): string {
  return createHash('sha256').update(value).digest('hex');
}

function normalized(value: CensusVec3): CensusVec3 {
  const length = Math.hypot(...value);
  return [value[0] / length, value[1] / length, value[2] / length];
}

function cross(a: CensusVec3, b: CensusVec3): CensusVec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function add(a: CensusVec3, b: CensusVec3): CensusVec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function scaled(value: CensusVec3, amount: number): CensusVec3 {
  return [value[0] * amount, value[1] * amount, value[2] * amount];
}

function wrap01(value: number): number {
  return value - Math.floor(value);
}

function meshBounds(mesh: IndexedMesh): { min: CensusVec3; max: CensusVec3 } {
  const min = [Infinity, Infinity, Infinity] as unknown as [number, number, number];
  const max = [-Infinity, -Infinity, -Infinity] as unknown as [number, number, number];
  for (let vertex = 0; vertex < mesh.positions.length; vertex += 3) {
    for (let axis = 0; axis < 3; axis++) {
      min[axis] = Math.min(min[axis], mesh.positions[vertex + axis]!);
      max[axis] = Math.max(max[axis], mesh.positions[vertex + axis]!);
    }
  }
  return { min, max };
}

function geometryFrom(mesh: IndexedMesh, tile: { originX: number; originZ: number; sizeX: number; sizeZ: number; topH: number }): DecodedOwnedProfileGeometry {
  const bounds = meshBounds(mesh);
  return {
    version: 4,
    profileId: 2,
    topH: tile.topH,
    tileOriginX: tile.originX,
    tileOriginZ: tile.originZ,
    tileSizeX: tile.sizeX,
    tileSizeZ: tile.sizeZ,
    bounds,
    positions: Float64Array.from(mesh.positions),
    triangles: Uint32Array.from(mesh.indices),
    vertexCount: mesh.positions.length / 3,
    triangleCount: mesh.indices.length / 3,
  };
}

function plumeTriangles(mesh: IndexedMesh): Uint8Array {
  if (!mesh.colors) throw new Error('hybrid source requires authored colors');
  const result = new Uint8Array(mesh.indices.length / 3);
  for (let triangle = 0; triangle < result.length; triangle++) {
    let reproductive = 0;
    for (let corner = 0; corner < 3; corner++) {
      const vertex = mesh.indices[triangle * 3 + corner]!;
      reproductive += mesh.colors[vertex * 3]! > mesh.colors[vertex * 3 + 1]! ? 1 : 0;
    }
    result[triangle] = reproductive >= 2 ? 1 : 0;
  }
  return result;
}

function interpolateAttributes(
  mesh: IndexedMesh,
  geometry: DecodedOwnedProfileGeometry,
  origin: CensusVec3,
  direction: CensusVec3,
  hit: OriginAwareTruthHit,
): EventAttributes {
  const point: CensusVec3 = [
    origin[0] + direction[0] * hit.t - hit.copyX * geometry.tileSizeX,
    origin[1] + direction[1] * hit.t,
    origin[2] + direction[2] * hit.t - hit.copyZ * geometry.tileSizeZ,
  ];
  const vertices = [0, 1, 2].map((corner) => mesh.indices[hit.triangleId * 3 + corner]!);
  const p = vertices.map((vertex) => [
    mesh.positions[vertex * 3]!, mesh.positions[vertex * 3 + 1]!, mesh.positions[vertex * 3 + 2]!,
  ] as CensusVec3);
  const e0: CensusVec3 = [p[1]![0] - p[0]![0], p[1]![1] - p[0]![1], p[1]![2] - p[0]![2]];
  const e1: CensusVec3 = [p[2]![0] - p[0]![0], p[2]![1] - p[0]![1], p[2]![2] - p[0]![2]];
  const q: CensusVec3 = [point[0] - p[0]![0], point[1] - p[0]![1], point[2] - p[0]![2]];
  const d00 = e0[0] ** 2 + e0[1] ** 2 + e0[2] ** 2;
  const d01 = e0[0] * e1[0] + e0[1] * e1[1] + e0[2] * e1[2];
  const d11 = e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2;
  const d20 = q[0] * e0[0] + q[1] * e0[1] + q[2] * e0[2];
  const d21 = q[0] * e1[0] + q[1] * e1[1] + q[2] * e1[2];
  const inverse = 1 / Math.max(1e-30, d00 * d11 - d01 * d01);
  const b = Math.max(0, (d11 * d20 - d01 * d21) * inverse);
  const c = Math.max(0, (d00 * d21 - d01 * d20) * inverse);
  const a = Math.max(0, 1 - b - c);
  const sum = Math.max(1e-30, a + b + c);
  const weights = [a / sum, b / sum, c / sum];
  const color = [0, 0, 0] as unknown as [number, number, number];
  const normal = [0, 0, 0] as unknown as [number, number, number];
  for (let corner = 0; corner < 3; corner++) {
    const vertex = vertices[corner]!;
    for (let axis = 0; axis < 3; axis++) {
      color[axis] += (mesh.colors?.[vertex * 3 + axis] ?? 1) * weights[corner]!;
      normal[axis] += mesh.normals[vertex * 3 + axis]! * weights[corner]!;
    }
  }
  return { color, normal: normalized(normal) };
}

function cellExit(
  origin: CensusVec3,
  direction: CensusVec3,
  geometry: DecodedOwnedProfileGeometry,
): ExitRecord {
  const phaseX = wrap01((origin[0] - geometry.tileOriginX) / geometry.tileSizeX);
  const phaseZ = wrap01((origin[2] - geometry.tileOriginZ) / geometry.tileSizeZ);
  const x = geometry.tileOriginX + phaseX * geometry.tileSizeX;
  const z = geometry.tileOriginZ + phaseZ * geometry.tileSizeZ;
  const candidates: Array<{ t: number; face: number }> = [];
  if (direction[0] > 1e-12) candidates.push({ t: (geometry.tileOriginX + geometry.tileSizeX - x) / direction[0], face: 1 });
  if (direction[0] < -1e-12) candidates.push({ t: (geometry.tileOriginX - x) / direction[0], face: 0 });
  if (direction[2] > 1e-12) candidates.push({ t: (geometry.tileOriginZ + geometry.tileSizeZ - z) / direction[2], face: 3 });
  if (direction[2] < -1e-12) candidates.push({ t: (geometry.tileOriginZ - z) / direction[2], face: 2 });
  if (direction[1] > 1e-12) candidates.push({ t: (geometry.bounds.max[1] - origin[1]) / direction[1], face: 5 });
  if (direction[1] < -1e-12) candidates.push({ t: (geometry.bounds.min[1] - origin[1]) / direction[1], face: 4 });
  const selected = candidates.filter((entry) => entry.t > 1e-10).sort((left, right) => left.t - right.t)[0];
  if (!selected) throw new Error('camera-path ray has no positive cell/slab exit');
  const point = add(origin, scaled(direction, selected.t));
  const height = (point[1] - geometry.bounds.min[1]) / (geometry.bounds.max[1] - geometry.bounds.min[1]);
  if (selected.face <= 1) return { face: selected.face, u: wrap01((point[2] - geometry.tileOriginZ) / geometry.tileSizeZ), v: height, distance: selected.t };
  if (selected.face <= 3) return { face: selected.face, u: wrap01((point[0] - geometry.tileOriginX) / geometry.tileSizeX), v: height, distance: selected.t };
  return { face: selected.face, u: phaseX, v: phaseZ, distance: selected.t };
}

function cubeAddress(direction: CensusVec3): { face: number; u: number; v: number } {
  const ax = Math.abs(direction[0]);
  const ay = Math.abs(direction[1]);
  const az = Math.abs(direction[2]);
  let face: number;
  let u: number;
  let v: number;
  if (ax >= ay && ax >= az) {
    face = direction[0] >= 0 ? 0 : 1;
    u = (direction[0] >= 0 ? -direction[2] : direction[2]) / ax;
    v = -direction[1] / ax;
  } else if (ay >= az) {
    face = direction[1] >= 0 ? 2 : 3;
    u = direction[0] / ay;
    v = (direction[1] >= 0 ? direction[2] : -direction[2]) / ay;
  } else {
    face = direction[2] >= 0 ? 4 : 5;
    u = (direction[2] >= 0 ? direction[0] : -direction[0]) / az;
    v = -direction[1] / az;
  }
  return { face, u: u * 0.5 + 0.5, v: v * 0.5 + 0.5 };
}

function axisCorners(value: number, size: number, wrap: boolean): Array<{ index: number; weight: number }> {
  const coordinate = value * size - 0.5;
  const lower = Math.floor(coordinate);
  const fraction = coordinate - lower;
  const fix = (index: number): number => wrap
    ? ((index % size) + size) % size
    : Math.max(0, Math.min(size - 1, index));
  return [{ index: fix(lower), weight: 1 - fraction }, { index: fix(lower + 1), weight: fraction }];
}

function localSpatialCorners(record: Float32Array, offset: number): string[] {
  const xs = axisCorners(record[offset + 3]!, LOCAL_X, true);
  const ys = axisCorners(record[offset + 4]!, LOCAL_Y, false);
  const zs = axisCorners(record[offset + 5]!, LOCAL_Z, true);
  return xs.flatMap((x) => ys.flatMap((y) => zs.map((z) => `ls:${x.index}:${y.index}:${z.index}`)));
}

function angularCorners(direction: CensusVec3, prefix: string, condition = -1): string[] {
  const address = cubeAddress(direction);
  const us = axisCorners(address.u, CUBE, false);
  const vs = axisCorners(address.v, CUBE, false);
  return us.flatMap((u) => vs.map((v) => `${prefix}:${condition}:${address.face}:${u.index}:${v.index}`));
}

function boundaryCorners(record: Float32Array, offset: number): string[] {
  const face = Math.round(record[offset + 9]!);
  if (face >= 4) return [];
  const us = axisCorners(record[offset + 10]!, BOUNDARY_U, false);
  const vs = axisCorners(record[offset + 11]!, BOUNDARY_V, false);
  return us.flatMap((u) => vs.map((v) => `bs:${face}:${u.index}:${v.index}`));
}

function auditAddresses(records: Float32Array, frames: number): AddressAudit {
  const validationFrames = new Set<number>();
  for (let frame = 2; frame < frames - 1; frame += 4) validationFrames.add(frame);
  const train = { ls: new Set<string>(), la: new Set<string>(), bs: new Set<string>(), ba: new Set<string>(), joint: new Set<string>() };
  const validation = { ls: new Set<string>(), la: new Set<string>(), bs: new Set<string>(), ba: new Set<string>(), joint: new Set<string>() };
  let trainRecords = 0;
  let validationRecords = 0;
  const validationCornerHits = { ls: 0, la: 0, bs: 0, ba: 0 };
  const validationCornerTotals = { ls: 0, la: 0, bs: 0, ba: 0 };
  let validationAllSeen = 0;
  for (let record = 0; record < records.length / FIELDS; record++) {
    const offset = record * FIELDS;
    const frame = Math.round(records[offset]!);
    const direction: CensusVec3 = [records[offset + 6]!, records[offset + 7]!, records[offset + 8]!];
    const face = Math.round(records[offset + 9]!);
    const corners = {
      ls: localSpatialCorners(records, offset),
      la: angularCorners(direction, 'la'),
      bs: boundaryCorners(records, offset),
      ba: face < 4 ? angularCorners(direction, 'ba', face) : [],
    };
    const target = validationFrames.has(frame) ? validation : train;
    for (const key of Object.keys(corners) as Array<keyof typeof corners>) corners[key].forEach((value) => target[key].add(value));
    target.joint.add(`${corners.ls[0]}|${corners.la[0]}|${corners.bs[0] ?? 'none'}|${corners.ba[0] ?? 'none'}`);
    if (validationFrames.has(frame)) validationRecords++;
    else trainRecords++;
  }
  for (let record = 0; record < records.length / FIELDS; record++) {
    const offset = record * FIELDS;
    const frame = Math.round(records[offset]!);
    if (!validationFrames.has(frame)) continue;
    const direction: CensusVec3 = [records[offset + 6]!, records[offset + 7]!, records[offset + 8]!];
    const face = Math.round(records[offset + 9]!);
    const corners = {
      ls: localSpatialCorners(records, offset),
      la: angularCorners(direction, 'la'),
      bs: boundaryCorners(records, offset),
      ba: face < 4 ? angularCorners(direction, 'ba', face) : [],
    };
    let allSeen = true;
    for (const key of Object.keys(corners) as Array<keyof typeof corners>) for (const value of corners[key]) {
      validationCornerTotals[key]++;
      if (train[key].has(value)) validationCornerHits[key]++;
      else allSeen = false;
    }
    if (allSeen) validationAllSeen++;
  }
  const activeFactorScalars = 4 * (train.ls.size + train.la.size + train.bs.size + train.ba.size) + 72;
  const trainOutputEquations = trainRecords * 8;
  const ratio = trainOutputEquations / activeFactorScalars;
  const allSeenFraction = validationRecords > 0 ? validationAllSeen / validationRecords : 0;
  const reasons: string[] = [];
  if (ratio < 1) reasons.push(`only ${ratio.toFixed(3)} training output equations per active factor/head scalar`);
  if (allSeenFraction < 0.95) reasons.push(`only ${(allSeenFraction * 100).toFixed(2)}% of held-out path records address factor corners all observed by training frames`);
  return {
    trainRecords,
    validationRecords,
    activeFactorScalars,
    trainOutputEquations,
    equationToActiveParameterRatio: ratio,
    validationAllCornersSeenFraction: allSeenFraction,
    validationCornerCoverage: Object.fromEntries((Object.keys(validationCornerHits) as Array<keyof typeof validationCornerHits>).map((key) => [
      key,
      validationCornerTotals[key] > 0 ? validationCornerHits[key] / validationCornerTotals[key] : 1,
    ])),
    uniqueJointTrainSignatures: train.joint.size,
    uniqueJointValidationSignatures: validation.joint.size,
    decision: reasons.length === 0 ? 'fit-identifiable' : 'underdetermined',
    reasons,
  };
}

function heat(value: number): [number, number, number, number] {
  const t = Math.max(0, Math.min(1, value));
  return [Math.round(255 * Math.min(1, t * 2)), Math.round(255 * (1 - Math.abs(t * 2 - 1))), Math.round(255 * Math.max(0, 1 - t * 2)), 255];
}

async function truthPanel(
  records: Float32Array,
  frames: number,
  grid: number,
  qaRoot: string,
  mode: 'color' | 'coverage' | 'depth' | 'variance',
  number: number,
  maximumLength: number,
): Promise<{ file: string; sha256: string; width: number; height: number; interpretation: string }> {
  const selectedFrames = [0, Math.floor(frames / 2), frames - 1];
  const width = grid * selectedFrames.length;
  const pixels = new Uint8Array(width * grid * 4);
  for (let panel = 0; panel < selectedFrames.length; panel++) {
    const frame = selectedFrames[panel]!;
    for (let y = 0; y < grid; y++) for (let x = 0; x < grid; x++) {
      const offset = ((frame * grid * grid) + y * grid + x) * FIELDS;
      const alpha = records[offset + 15]!;
      let rgba: [number, number, number, number];
      if (mode === 'color') rgba = [
        Math.round(255 * Math.max(0, Math.min(1, records[offset + 17]! / Math.max(alpha, 1e-6)))),
        Math.round(255 * Math.max(0, Math.min(1, records[offset + 18]! / Math.max(alpha, 1e-6)))),
        Math.round(255 * Math.max(0, Math.min(1, records[offset + 19]! / Math.max(alpha, 1e-6)))),
        Math.round(255 * alpha),
      ];
      else if (mode === 'coverage') rgba = [
        Math.round(255 * records[offset + 26]!),
        Math.round(255 * records[offset + 25]!),
        Math.round(180 * records[offset + 26]!),
        255,
      ];
      else if (mode === 'depth') rgba = heat(alpha > 0 ? records[offset + 23]! / maximumLength : 1);
      else rgba = heat(Math.sqrt(Math.max(0, records[offset + 24]!)) / 0.25);
      pixels.set(rgba, (y * width + panel * grid + x) * 4);
    }
  }
  const file = `${String(number).padStart(3, '0')}-truth-${mode}-frames-0-mid-last.png`;
  const path = resolve(qaRoot, file);
  await sharp(pixels, { raw: { width, height: grid, channels: 4 } })
    .resize(width * 6, grid * 6, { kernel: 'nearest' })
    .png({ compressionLevel: 9 })
    .toFile(path);
  return {
    file,
    sha256: sha(readFileSync(path)),
    width: width * 6,
    height: grid * 6,
    interpretation: `${mode} truth at the first, middle, and last persistent 18-degree camera-path frames`,
  };
}

const cli = parseArgs(process.argv.slice(2));
const grid = numberArg(cli.grid, 64, 'grid');
const frames = numberArg(cli.frames, 17, 'frames');
const microSamples = numberArg(cli.micro, 8, 'micro samples');
const pathMetres = numberArg(cli.path, 0.03, 'path metres');
const fovDegrees = numberArg(cli.fov, 18, 'FOV degrees');
for (const [label, value] of Object.entries({ grid, frames, microSamples })) {
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${label} must be a positive integer`);
}
if (frames < 5 || (frames & 1) === 0) throw new Error('frames must be an odd integer >= 5');

const fixture = makeCalamagrostisCanescensBandlimitedSource();
const geometry = geometryFrom(fixture.mesh, fixture.tile);
const isPlume = plumeTriangles(fixture.mesh);
const sourceSha256 = sha(JSON.stringify(fixture.mesh));
const configuration = {
  schema: 'laas-hybrid-transfer-low-oblique-truth/v2',
  sourceSha256,
  grid,
  frames,
  microSamples,
  pathMetres,
  centralElevationDegrees: 18,
  centralAzimuthDegrees: 45,
  fovDegrees,
  horizonMetres: HORIZON,
  transferAlgebra: {
    record: '(A,m,C,N), where A is opacity, M is the premultiplied first moment in metres, and m=M/L',
    composition: 'A12=A1+(1-A1)*A2; M12=M1+(1-A1)*(M2+L1*A2)',
    segmentOwnership: '[0,L); an event at exactly L belongs only to the following segment',
    terminalOwnership: 'a Y exit terminates the finite botanical slab and is marked explicitly; it never addresses a periodic side-boundary factor',
  },
  rejectedCandidateFactorAllocation: {
    localSpatial: [LOCAL_X, LOCAL_Z, LOCAL_Y, 4],
    boundarySpatial: [4, BOUNDARY_U, BOUNDARY_V, 4],
    localSphere: [6, CUBE, CUBE, 4],
    boundarySphere: [4, 6, CUBE, CUBE, 4],
    reads: 4,
    rank: 4,
    residentBytesIncludingMips: 42_080_048,
  },
  toolSha256: sha(readFileSync(import.meta.filename)),
};
const recipeSha256 = sha(JSON.stringify(configuration));
const outputRoot = resolve(
  typeof cli.output === 'string'
    ? cli.output
    : `data/work/groundcover-hybrid-transfer/${sourceSha256}/${recipeSha256}`,
);
const truthRoot = resolve(outputRoot, 'truth');
const qaRoot = resolve(outputRoot, 'qa');
mkdirSync(truthRoot, { recursive: true });
mkdirSync(qaRoot, { recursive: true });

console.error(`[hybrid-transfer] building BVH for ${geometry.triangleCount.toLocaleString()} exact source triangles`);
const bvh = TriangleBvh.build(geometry, 8);
const recordCount = frames * grid * grid;
const records = new Float32Array(recordCount * FIELDS);
const microLabels = Buffer.alloc(recordCount * microSamples * MICRO_RECORD_BYTES);
const elevation = 18 * Math.PI / 180;
const centerDirection: CensusVec3 = normalized([Math.cos(elevation) * Math.SQRT1_2, -Math.sin(elevation), Math.cos(elevation) * Math.SQRT1_2]);
const right = normalized(cross(centerDirection, [0, 1, 0]));
const up = normalized(cross(right, centerDirection));
const target: CensusVec3 = [fixture.tile.sizeX * 0.5, geometry.bounds.min[1] + (geometry.bounds.max[1] - geometry.bounds.min[1]) * 0.56, fixture.tile.sizeZ * 0.5];
const cameraDistance = 2.35;
const referenceCamera = add(target, scaled(centerDirection, -cameraDistance));
const tangent = Math.tan(fovDegrees * Math.PI / 360);
const microOffsets = Array.from({ length: microSamples }, (_value, index) => ({
  x: ((index * 0.7548776662466927) % 1) - 0.5,
  y: ((index * 0.5698402909980532 + 0.25) % 1) - 0.5,
}));
let maximumLength = 0;
let macro = 0;
for (let frame = 0; frame < frames; frame++) {
  const pathOffset = ((frame / (frames - 1)) - 0.5) * pathMetres;
  const camera = add(referenceCamera, scaled(right, pathOffset));
  for (let pixelY = 0; pixelY < grid; pixelY++) for (let pixelX = 0; pixelX < grid; pixelX++) {
    const ndcX = ((pixelX + 0.5) / grid * 2 - 1) * tangent;
    const ndcY = (1 - (pixelY + 0.5) / grid * 2) * tangent;
    const d = normalized(add(centerDirection, add(scaled(right, ndcX), scaled(up, ndcY))));
    const entryT = (geometry.topH - camera[1]) / d[1];
    const origin = add(camera, scaled(d, entryT));
    const length = Math.min(HORIZON, Math.max(0, (geometry.bounds.min[1] - origin[1]) / d[1]));
    maximumLength = Math.max(maximumLength, length);
    const exit = cellExit(origin, d, geometry);
    const offset = macro * FIELDS;
    records.set([
      frame, pixelX, pixelY,
      wrap01((origin[0] - geometry.tileOriginX) / geometry.tileSizeX),
      (origin[1] - geometry.bounds.min[1]) / (geometry.bounds.max[1] - geometry.bounds.min[1]),
      wrap01((origin[2] - geometry.tileOriginZ) / geometry.tileSizeZ),
      ...d,
      exit.face, exit.u, exit.v, exit.distance, exit.face >= 4 ? 1 : 0, length,
    ], offset);
    let hitCount = 0;
    let firstMoment = 0;
    const color = [0, 0, 0];
    const normal = [0, 0, 0];
    const hitDistances: number[] = [];
    let structureHits = 0;
    let plumeHits = 0;
    for (let micro = 0; micro < microSamples; micro++) {
      const jitter = microOffsets[micro]!;
      const microD = normalized(add(centerDirection, add(
        scaled(right, ndcX + jitter.x * tangent * 2 / grid),
        scaled(up, ndcY + jitter.y * tangent * 2 / grid),
      )));
      const microEntry = (geometry.topH - camera[1]) / microD[1];
      const microOrigin = add(camera, scaled(microD, microEntry));
      const candidateHit = periodicNearestSuccessor(geometry, bvh, microOrigin, microD, length, 1e-8);
      // Segment ownership is exactly half-open. The successor helper permits
      // the horizon endpoint, so exclude it here; the following segment owns it.
      const hit = candidateHit && candidateHit.t < length ? candidateHit : null;
      const labelOffset = (macro * microSamples + micro) * MICRO_RECORD_BYTES;
      // Pixel/micro identities persist across camera frames. The frame field
      // distinguishes repeated observations without changing the ray label.
      microLabels.writeUInt32LE(pixelY * grid + pixelX, labelOffset);
      microLabels.writeUInt16LE(micro, labelOffset + 4);
      microLabels.writeUInt16LE(frame, labelOffset + 6);
      microLabels.writeUInt32LE(hit?.triangleId ?? 0xffff_ffff, labelOffset + 8);
      microLabels.writeInt16LE(hit?.copyX ?? 0, labelOffset + 12);
      microLabels.writeInt16LE(hit?.copyZ ?? 0, labelOffset + 14);
      microLabels.writeUInt8(hit ? (isPlume[hit.triangleId] ? 2 : 1) : 0, labelOffset + 16);
      microLabels.writeFloatLE(hit?.t ?? length, labelOffset + 20);
      if (!hit) {
        continue;
      }
      const attributes = interpolateAttributes(fixture.mesh, geometry, microOrigin, microD, hit);
      hitCount++;
      firstMoment += hit.t;
      hitDistances.push(hit.t);
      if (isPlume[hit.triangleId]) plumeHits++;
      else structureHits++;
      for (let axis = 0; axis < 3; axis++) {
        color[axis]! += attributes.color[axis]!;
        normal[axis]! += attributes.normal[axis]!;
      }
    }
    const A = hitCount / microSamples;
    const M = firstMoment / microSamples;
    const m = length > 0 ? M / length : 0;
    const C = color.map((value) => value / microSamples);
    const N = normal.map((value) => value / microSamples);
    const mu = A > 0 ? M / A : 0;
    const variance = hitDistances.length > 0
      ? hitDistances.reduce((sum, value) => sum + (value - mu) ** 2, 0) / hitDistances.length
      : 0;
    records.set([
      A, m, ...C, ...N, mu, variance,
      structureHits / microSamples,
      plumeHits / microSamples,
    ], offset + 15);
    macro++;
  }
  console.error(`[hybrid-transfer] frame ${frame + 1}/${frames}`);
}

const truthBytes = Buffer.from(records.buffer, records.byteOffset, records.byteLength);
const truthPath = resolve(truthRoot, 'camera-path.f32');
const labelsPath = resolve(truthRoot, 'persistent-micro-ray-labels.bin');
writeFileSync(truthPath, truthBytes);
writeFileSync(labelsPath, microLabels);
const addressAudit = auditAddresses(records, frames);
const qa = await Promise.all([
  truthPanel(records, frames, grid, qaRoot, 'color', 1, maximumLength),
  truthPanel(records, frames, grid, qaRoot, 'coverage', 2, maximumLength),
  truthPanel(records, frames, grid, qaRoot, 'depth', 3, maximumLength),
  truthPanel(records, frames, grid, qaRoot, 'variance', 4, maximumLength),
]);
const supportMessage = addressAudit.decision === 'underdetermined'
  ? `Data support also fails: ${addressAudit.reasons.join('; ')}`
  : 'Sampled factor addresses are data-supported, but that cannot repair the rejected head algebra.';
const message = `FIT STOPPED: ${HEAD_REJECTION_REASONS[1]}`;
const svg = Buffer.from(`<svg width="1800" height="420" xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="#21191d"/><text x="40" y="80" fill="#ffb4b4" font-family="Arial" font-size="42" font-weight="700">RANK-FOUR AFFINE HEAD REJECTED BEFORE FIT</text><text x="40" y="145" fill="#f6e7e7" font-family="Arial" font-size="23">Affine mixed cross-difference: 0. Exact finite-horizon suffix transfer: nonzero when boundary extinction differs.</text><text x="40" y="205" fill="#f6e7e7" font-family="Arial" font-size="23">The live opaque resolve also cannot preserve fractional coverage A.</text><text x="40" y="275" fill="#d7c9cf" font-family="Arial" font-size="21">${supportMessage.replaceAll('&', '&amp;').replaceAll('<', '&lt;')}</text><text x="40" y="345" fill="#d7c9cf" font-family="Arial" font-size="22">No extra rank, read, layer, loop, stochastic coverage, or memorising reconstruction was introduced.</text></svg>`);
const stopPath = resolve(qaRoot, '005-reconstruction-stopped-head-rejected.png');
await sharp(svg).png().toFile(stopPath);
qa.push({ file: '005-reconstruction-stopped-head-rejected.png', sha256: sha(readFileSync(stopPath)), width: 1800, height: 420, interpretation: `${message}; ${supportMessage}` });
const manifest = {
  ...configuration,
  recipeSha256,
  source: {
    generator: fixture.generator,
    vertices: geometry.vertexCount,
    triangles: geometry.triangleCount,
    structureTriangles: Array.from(isPlume).filter((value) => value === 0).length,
    plumeTriangles: Array.from(isPlume).filter((value) => value === 1).length,
    bounds: geometry.bounds,
    tile: fixture.tile,
  },
  cameraPath: {
    referenceCamera,
    target,
    fixedCentralDirection: centerDirection,
    right,
    pathMetres,
    frames,
    grid,
    fovDegrees,
    persistentMicroPattern: microOffsets,
    label: '[persistentPixelId, microIndex] is stable across every frame; persistentPixelId=pixelY*grid+pixelX',
  },
  truthRecord: {
    type: 'little-endian float32',
    floats: FIELDS,
    fields: [
      'frame', 'pixelX', 'pixelY', 'phaseX', 'heightFraction', 'phaseZ',
      'directionX', 'directionY', 'directionZ', 'exitFace', 'exitU', 'exitV',
      'firstExitMetres', 'yTerminalMask', 'segmentLengthMetres', 'A', 'm', 'C.r', 'C.g', 'C.b',
      'N.x', 'N.y', 'N.z', 'conditionalMeanMetres', 'conditionalVarianceMetres2',
      'structureCoverage', 'plumeCoverage',
    ],
    semantics: 'each persistent micro-ray records the nearest complete structure+plume event in the half-open segment [0,L) before the N-ray footprint average; misses contribute A=0,M=0; hits contribute A=1,M=t,C=c,N=n; m=M/L avoids subtracting near-equal horizon-scale FP16 values',
  },
  persistentMicroLabelRecord: {
    bytes: MICRO_RECORD_BYTES,
    fields: 'u32 persistentPixelId, u16 microIndex, u16 frame, u32 triangleId (0xffffffff miss), i16 copyX, i16 copyZ, u8 category (0 miss/1 structure/2 plume), 3 pad bytes, f32 t, 8 pad bytes',
  },
  accelerator: bvh.metrics,
  addressAudit,
  mathematicalGate: {
    decision: 'head-rejected-before-fit',
    reasons: HEAD_REJECTION_REASONS,
    proofWitness: {
      affineMixedCrossDifference: 'h(b1,r1)-h(b1,r0)-h(b0,r1)+h(b0,r0)=0',
      exactSuffixMixedCrossDifference: 'exp(-kappa(b1)*r1)-exp(-kappa(b1)*r0)-exp(-kappa(b0)*r1)+exp(-kappa(b0)*r0), generally nonzero',
    },
  },
  decision: 'stop before fitting: the proposed affine fixed-cost head is mathematically insufficient and its fractional opacity output is incompatible with the current opaque live interface',
  files: {
    truth: { file: 'truth/camera-path.f32', bytes: truthBytes.byteLength, sha256: sha(truthBytes), records: recordCount },
    labels: { file: 'truth/persistent-micro-ray-labels.bin', bytes: microLabels.byteLength, sha256: sha(microLabels), records: recordCount * microSamples },
    qa,
  },
};
const manifestPath = resolve(outputRoot, 'report.json');
writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
console.log('[hybrid-transfer] HEAD-REJECTED-BEFORE-FIT');
console.log(`  report: ${manifestPath}`);
console.log(`  QA: ${qaRoot}`);
