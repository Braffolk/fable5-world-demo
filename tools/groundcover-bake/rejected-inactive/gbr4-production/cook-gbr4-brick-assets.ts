/**
 * Historical finite-soup GBR4/v1 cook.
 *
 * This must never publish the active periodic Calamagrostis asset: v1 stops
 * at one physical tile-box exit and loses later periodic hits. Use
 * cook-gbr4-periodic-cap-assets.ts for the active v2 container.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
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
const HEADER_BYTES = 512;
const ALIGNMENT = 256;
const FACE_COUNT = 6;
const CELL_RESOLUTION = 6;
const BLOCK_CELL_EDGE = 3;
const BLOCK_SAMPLE_EDGE = BLOCK_CELL_EDGE + 1;
const BLOCKS_PER_AXIS = CELL_RESOLUTION / BLOCK_CELL_EDGE;
const SAMPLES_PER_BRICK = BLOCK_SAMPLE_EDGE ** 4;
const QUADRATURE_SIDE = 2;
const STANDOFF_METRES = 4;
const PIXEL_ANGLE = 60 * Math.PI / 180 / 1920;
const CARRIER_GUARD = 0.025;
const TEXEL_BYTES = 8;
const FLAG_PREMULTIPLIED = 1 << 0;
const FLAG_TWO_STRATA = 1 << 1;
const FLAG_CATEGORICAL = 1 << 2;
const FLAG_PERIODIC_XZ = 1 << 3;

interface FacePair { entry: number; exit: number; }
interface RayBoxHit { entryT: number; exitT: number; entryFace: number; exitFace: number; }
interface SurfaceSample { depth01: number; color: CensusVec3; normal: CensusVec3; }
interface FilteredNode { color0: readonly [number, number, number, number]; color1: readonly [number, number, number, number]; categorical: readonly [number, number, number, number]; }

export function sha256(value: Uint8Array | string): string {
  return createHash('sha256').update(value).digest('hex');
}

export function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(',')}}`;
}

export function align(value: number): number { return Math.ceil(value / ALIGNMENT) * ALIGNMENT; }
export function clamp(value: number, low: number, high: number): number { return Math.max(low, Math.min(high, value)); }

export function normalized(value: CensusVec3): CensusVec3 {
  const length = Math.hypot(value[0], value[1], value[2]);
  if (!(length > 0)) throw new Error('zero vector');
  return [value[0] / length, value[1] / length, value[2] / length];
}

export function cross(a: CensusVec3, b: CensusVec3): CensusVec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function addScaled(a: CensusVec3, b: CensusVec3, scale: number): CensusVec3 {
  return [a[0] + b[0] * scale, a[1] + b[1] * scale, a[2] + b[2] * scale];
}

function subtract(a: CensusVec3, b: CensusVec3): CensusVec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function decodeOct(x01: number, y01: number): CensusVec3 {
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

export function encodeOct(normal: CensusVec3): number {
  const denominator = Math.abs(normal[0]) + Math.abs(normal[1]) + Math.abs(normal[2]);
  let x = normal[0] / Math.max(1e-30, denominator);
  let y = normal[1] / Math.max(1e-30, denominator);
  if (normal[2] < 0) {
    const oldX = x;
    x = (1 - Math.abs(y)) * (oldX >= 0 ? 1 : -1);
    y = (1 - Math.abs(oldX)) * (y >= 0 ? 1 : -1);
  }
  const qx = Math.round(clamp(x * 0.5 + 0.5, 0, 1) * 255);
  const qy = Math.round(clamp(y * 0.5 + 0.5, 0, 1) * 255);
  return qx | (qy << 8);
}

export function float16(value: number): number {
  if (Number.isNaN(value)) return 0x7e00;
  if (value === Number.POSITIVE_INFINITY) return 0x7c00;
  if (value === Number.NEGATIVE_INFINITY) return 0xfc00;
  const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0;
  const absolute = Math.abs(value);
  if (absolute === 0) return sign;
  if (absolute >= 65504) return sign | 0x7bff;
  if (absolute < 2 ** -14) return sign | Math.round(absolute / 2 ** -24);
  const exponent = Math.floor(Math.log2(absolute));
  let mantissa = Math.round((absolute / 2 ** exponent - 1) * 1024);
  let adjustedExponent = exponent;
  if (mantissa === 1024) { mantissa = 0; adjustedExponent++; }
  return sign | ((adjustedExponent + 15) << 10) | mantissa;
}

export function attributes(
  bytes: Uint8Array,
  geometry: DecodedOwnedProfileGeometry,
  origin: CensusVec3,
  direction: CensusVec3,
  hit: OriginAwareTruthHit,
): { color: CensusVec3; normal: CensusVec3 } {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const vertexOffset = view.getUint32(84, true);
  const triangle = hit.triangleId * 3;
  const vertices = [geometry.triangles[triangle]!, geometry.triangles[triangle + 1]!, geometry.triangles[triangle + 2]!] as const;
  const point: CensusVec3 = [
    origin[0] + direction[0] * hit.t - hit.copyX * geometry.tileSizeX,
    origin[1] + direction[1] * hit.t,
    origin[2] + direction[2] * hit.t - hit.copyZ * geometry.tileSizeZ,
  ];
  const position = (vertex: number): CensusVec3 => {
    const offset = vertex * 3;
    return [geometry.positions[offset]!, geometry.positions[offset + 1]!, geometry.positions[offset + 2]!];
  };
  const a = position(vertices[0]);
  const e0 = subtract(position(vertices[1]), a);
  const e1 = subtract(position(vertices[2]), a);
  const q = subtract(point, a);
  const d00 = e0[0] ** 2 + e0[1] ** 2 + e0[2] ** 2;
  const d01 = e0[0] * e1[0] + e0[1] * e1[1] + e0[2] * e1[2];
  const d11 = e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2;
  const d20 = q[0] * e0[0] + q[1] * e0[1] + q[2] * e0[2];
  const d21 = q[0] * e1[0] + q[1] * e1[1] + q[2] * e1[2];
  const denominator = d00 * d11 - d01 * d01;
  const baryB = denominator === 0 ? 0 : (d11 * d20 - d01 * d21) / denominator;
  const baryC = denominator === 0 ? 0 : (d00 * d21 - d01 * d20) / denominator;
  const weights = [1 - baryB - baryC, baryB, baryC];
  const color: [number, number, number] = [0, 0, 0];
  const normal: [number, number, number] = [0, 0, 0];
  for (let corner = 0; corner < 3; corner++) {
    const offset = vertexOffset + vertices[corner]! * 16;
    const n = decodeOct(view.getUint16(offset + 12, true) / 65535, view.getUint16(offset + 14, true) / 65535);
    for (let channel = 0; channel < 3; channel++) {
      color[channel] += weights[corner]! * view.getUint16(offset + 6 + channel * 2, true) / 65535;
      normal[channel] += weights[corner]! * n[channel]!;
    }
  }
  return { color, normal: normalized(normal) };
}

function facePoint(face: number, u: number, v: number, minimum: CensusVec3, maximum: CensusVec3): CensusVec3 {
  const x = minimum[0] + (maximum[0] - minimum[0]) * u;
  const y = minimum[1] + (maximum[1] - minimum[1]) * u;
  const z = minimum[2] + (maximum[2] - minimum[2]) * v;
  switch (face) {
    case 0: return [minimum[0], y, z];
    case 1: return [maximum[0], y, z];
    case 2: return [x, minimum[1], z];
    case 3: return [x, maximum[1], z];
    case 4: return [x, minimum[1] + (maximum[1] - minimum[1]) * v, minimum[2]];
    case 5: return [x, minimum[1] + (maximum[1] - minimum[1]) * v, maximum[2]];
    default: throw new Error(`invalid face ${face}`);
  }
}

function rayBox(origin: CensusVec3, direction: CensusVec3, minimum: CensusVec3, maximum: CensusVec3): RayBoxHit | null {
  let near = Number.NEGATIVE_INFINITY;
  let far = Number.POSITIVE_INFINITY;
  let nearFace = -1;
  let farFace = -1;
  const minFaces = [0, 2, 4];
  const maxFaces = [1, 3, 5];
  for (let axis = 0; axis < 3; axis++) {
    const d = direction[axis]!;
    if (Math.abs(d) <= 1e-15) {
      if (origin[axis]! < minimum[axis]! || origin[axis]! > maximum[axis]!) return null;
      continue;
    }
    let a = (minimum[axis]! - origin[axis]!) / d;
    let b = (maximum[axis]! - origin[axis]!) / d;
    let faceA = minFaces[axis]!;
    let faceB = maxFaces[axis]!;
    if (a > b) { [a, b] = [b, a]; [faceA, faceB] = [faceB, faceA]; }
    if (a > near + 1e-12 || (Math.abs(a - near) <= 1e-12 && faceA < nearFace)) { near = a; nearFace = faceA; }
    if (b < far - 1e-12 || (Math.abs(b - far) <= 1e-12 && faceB < farFace)) { far = b; farFace = faceB; }
    if (near > far + 1e-12) return null;
  }
  return far > Math.max(0, near) && nearFace >= 0 && farFace >= 0
    ? { entryT: Math.max(0, near), exitT: far, entryFace: nearFace, exitFace: farFace }
    : null;
}

function cluster(samples: readonly SurfaceSample[]): readonly [SurfaceSample[], SurfaceSample[]] {
  if (samples.length < 2) return [[...samples], []];
  let c0 = Math.min(...samples.map((sample) => sample.depth01));
  let c1 = Math.max(...samples.map((sample) => sample.depth01));
  let g0: SurfaceSample[] = [];
  let g1: SurfaceSample[] = [];
  for (let iteration = 0; iteration < 8; iteration++) {
    g0 = [];
    g1 = [];
    for (const sample of samples) (Math.abs(sample.depth01 - c0) <= Math.abs(sample.depth01 - c1) ? g0 : g1).push(sample);
    if (g0.length) c0 = g0.reduce((sum, sample) => sum + sample.depth01, 0) / g0.length;
    if (g1.length) c1 = g1.reduce((sum, sample) => sum + sample.depth01, 0) / g1.length;
  }
  return c0 <= c1 ? [g0, g1] : [g1, g0];
}

function filtered(group: readonly SurfaceSample[], total: number): { color: readonly [number, number, number, number]; depth: number; oct: number } {
  if (!group.length) return { color: [0, 0, 0, 0], depth: 0, oct: encodeOct([0, 1, 0]) };
  const alpha = group.length / total;
  const rgb: [number, number, number] = [0, 0, 0];
  const meanDepth = group.reduce((sum, sample) => sum + sample.depth01, 0) / group.length;
  let representative = group[0]!;
  for (const sample of group) if (Math.abs(sample.depth01 - meanDepth) < Math.abs(representative.depth01 - meanDepth)) representative = sample;
  for (const sample of group) for (let channel = 0; channel < 3; channel++) rgb[channel] += sample.color[channel]! / total;
  return { color: [rgb[0], rgb[1], rgb[2], alpha], depth: Math.round(clamp(representative.depth01, 0, 1) * 65535), oct: encodeOct(representative.normal) };
}

function integrateNode(
  bytes: Uint8Array,
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  entry: CensusVec3,
  exit: CensusVec3,
  minimum: CensusVec3,
  maximum: CensusVec3,
): FilteredNode {
  const delta = subtract(exit, entry);
  const segmentLength = Math.hypot(...delta);
  if (!(segmentLength > 1e-8)) return { color0: [0, 0, 0, 0], color1: [0, 0, 0, 0], categorical: [0, encodeOct([0, 1, 0]), 0, encodeOct([0, 1, 0])] };
  const central = normalized(delta);
  const origin = addScaled(entry, central, -STANDOFF_METRES);
  const right = Math.abs(central[1]) > 0.999999 ? normalized([1, 0, 0]) : normalized(cross(central, [0, 1, 0]));
  const up = normalized(cross(right, central));
  const tangent = Math.tan(PIXEL_ANGLE * 0.5);
  const samples: SurfaceSample[] = [];
  for (let sy = 0; sy < QUADRATURE_SIDE; sy++) for (let sx = 0; sx < QUADRATURE_SIDE; sx++) {
    const px = (2 * (sx + 0.5) / QUADRATURE_SIDE - 1) * tangent;
    const py = (2 * (sy + 0.5) / QUADRATURE_SIDE - 1) * tangent;
    const direction = normalized([
      central[0] + right[0] * px + up[0] * py,
      central[1] + right[1] * px + up[1] * py,
      central[2] + right[2] * px + up[2] * py,
    ]);
    const box = rayBox(origin, direction, minimum, maximum);
    if (!box || box.entryT < 0 || !(box.exitT > box.entryT)) continue;
    const q = addScaled(origin, direction, box.entryT);
    const horizon = box.exitT - box.entryT;
    const hit = periodicNearestSuccessor(geometry, bvh, q, direction, horizon, 0);
    if (!hit) continue;
    const decoded = attributes(bytes, geometry, q, direction, hit);
    samples.push({ depth01: clamp(hit.t / horizon, 0, 1), ...decoded });
  }
  const groups = cluster(samples);
  const front = filtered(groups[0], QUADRATURE_SIDE ** 2);
  const back = filtered(groups[1], QUADRATURE_SIDE ** 2);
  return { color0: front.color, color1: back.color, categorical: [front.depth, front.oct, back.depth, back.oct] };
}

function nodeLinear(pair: number, i0: number, i1: number, i2: number, i3: number): number {
  const edge = CELL_RESOLUTION + 1;
  return ((((pair * edge + i3) * edge + i2) * edge + i1) * edge + i0);
}

async function main(): Promise<void> {
  const started = performance.now();
  const bytes = readFileSync(SOURCE);
  const sourceSha = sha256(bytes);
  if (sourceSha !== EXPECTED_SOURCE_SHA) throw new Error(`accepted source changed: ${sourceSha}`);
  const geometry = decodeOwnedProfileGeometry(bytes);
  const minimum: CensusVec3 = [geometry.tileOriginX, geometry.bounds.min[1] - CARRIER_GUARD, geometry.tileOriginZ];
  const maximum: CensusVec3 = [geometry.tileOriginX + geometry.tileSizeX, geometry.bounds.max[1] + CARRIER_GUARD, geometry.tileOriginZ + geometry.tileSizeZ];
  const pairs: FacePair[] = [];
  for (let entry = 0; entry < FACE_COUNT; entry++) for (let exit = 0; exit < FACE_COUNT; exit++) if (entry !== exit) pairs.push({ entry, exit });
  const blocksPerPair = BLOCKS_PER_AXIS ** 4;
  const codebookCount = pairs.length * blocksPerPair;
  const tileEdge = BLOCK_SAMPLE_EDGE ** 2;
  const atlasColumns = Math.ceil(Math.sqrt(codebookCount));
  const atlasRows = Math.ceil(codebookCount / atlasColumns);
  const atlasWidth = atlasColumns * tileEdge;
  const atlasHeight = atlasRows * tileEdge;
  const atlasTexels = atlasWidth * atlasHeight;
  const nodeCount = pairs.length * (CELL_RESOLUTION + 1) ** 4;
  const nodeColor0 = new Float32Array(nodeCount * 4);
  const nodeColor1 = new Float32Array(nodeCount * 4);
  const nodeCategorical = new Uint16Array(nodeCount * 4);
  console.error(`[gbr4] BVH for ${geometry.triangleCount.toLocaleString()} triangles`);
  const bvh = TriangleBvh.build(geometry);
  console.error(`[gbr4] cooking ${nodeCount.toLocaleString()} shared 4D nodes x ${QUADRATURE_SIDE ** 2} subrays`);
  let cooked = 0;
  for (let pair = 0; pair < pairs.length; pair++) {
    const facePair = pairs[pair]!;
    for (let i3 = 0; i3 <= CELL_RESOLUTION; i3++) for (let i2 = 0; i2 <= CELL_RESOLUTION; i2++) {
      for (let i1 = 0; i1 <= CELL_RESOLUTION; i1++) for (let i0 = 0; i0 <= CELL_RESOLUTION; i0++) {
        const entry = facePoint(facePair.entry, i0 / CELL_RESOLUTION, i1 / CELL_RESOLUTION, minimum, maximum);
        const exit = facePoint(facePair.exit, i2 / CELL_RESOLUTION, i3 / CELL_RESOLUTION, minimum, maximum);
        const filteredNode = integrateNode(bytes, geometry, bvh, entry, exit, minimum, maximum);
        const node = nodeLinear(pair, i0, i1, i2, i3);
        nodeColor0.set(filteredNode.color0, node * 4);
        nodeColor1.set(filteredNode.color1, node * 4);
        nodeCategorical.set(filteredNode.categorical, node * 4);
        cooked++;
      }
    }
    console.error(`[gbr4] pair ${pair + 1}/${pairs.length}; nodes ${cooked.toLocaleString()}/${nodeCount.toLocaleString()}`);
  }

  const descriptorCount = codebookCount;
  const descriptorBytes = descriptorCount * 2;
  const pairTableBytes = pairs.length * 16;
  const scaleTableBytes = 64;
  const atlasBytes = atlasTexels * TEXEL_BYTES;
  const descriptorOffset = HEADER_BYTES;
  const pairTableOffset = align(descriptorOffset + descriptorBytes);
  const scaleTableOffset = align(pairTableOffset + pairTableBytes);
  const color0Offset = align(scaleTableOffset + scaleTableBytes);
  const color1Offset = align(color0Offset + atlasBytes);
  const categoricalOffset = align(color1Offset + atlasBytes);
  const totalBytes = align(categoricalOffset + atlasBytes);
  const recipe = {
    schema: 'laas-gbr4-minimal-real-cook-recipe/v1',
    implementationSha256: sha256(readFileSync(fileURLToPath(import.meta.url))),
    sourceSha256: sourceSha,
    address: 'ordered entry-face UV plus exit-face UV',
    cellResolution: CELL_RESOLUTION, blockCellEdge: BLOCK_CELL_EDGE,
    quadratureSide: QUADRATURE_SIDE, standoffMetres: STANDOFF_METRES,
    pixelAngleRadians: PIXEL_ANGLE,
    codebook: 'lossless one codeword per block; VQ intentionally not claimed',
  };
  const recipeSha = sha256(canonicalJson(recipe));
  const artifact = Buffer.alloc(totalBytes);
  const view = new DataView(artifact.buffer, artifact.byteOffset, artifact.byteLength);
  artifact.write('GBR4', 0, 'ascii');
  view.setUint32(4, 1, true);
  view.setUint32(8, HEADER_BYTES, true);
  view.setUint32(12, geometry.profileId, true);
  view.setUint32(16, FACE_COUNT, true);
  view.setUint32(20, pairs.length, true);
  view.setUint32(24, 4, true);
  view.setUint32(28, CELL_RESOLUTION, true);
  view.setUint32(32, BLOCK_CELL_EDGE, true);
  view.setUint32(36, BLOCK_SAMPLE_EDGE, true);
  view.setUint32(40, BLOCKS_PER_AXIS, true);
  view.setUint32(44, 1, true);
  view.setUint32(48, 1, true);
  for (const [offset, value] of [
    [52, descriptorOffset], [56, descriptorBytes], [60, pairTableOffset], [64, pairTableBytes],
    [68, scaleTableOffset], [72, scaleTableBytes], [76, color0Offset], [80, atlasBytes],
    [84, color1Offset], [88, atlasBytes], [92, categoricalOffset], [96, atlasBytes],
    [100, atlasWidth], [104, atlasHeight], [108, tileEdge], [112, tileEdge],
    [116, codebookCount], [120, 65535],
    [124, FLAG_PREMULTIPLIED | FLAG_TWO_STRATA | FLAG_CATEGORICAL | FLAG_PERIODIC_XZ],
  ] as const) view.setUint32(offset, value, true);
  view.setFloat32(128, geometry.tileOriginX, true);
  view.setFloat32(132, geometry.tileOriginZ, true);
  view.setFloat32(136, geometry.tileSizeX, true);
  view.setFloat32(140, geometry.tileSizeZ, true);
  for (let axis = 0; axis < 3; axis++) view.setFloat32(144 + axis * 4, minimum[axis]!, true);
  for (let axis = 0; axis < 3; axis++) view.setFloat32(156 + axis * 4, maximum[axis]!, true);
  artifact.set(Buffer.from(sourceSha, 'hex'), 168);
  artifact.set(Buffer.from(recipeSha, 'hex'), 200);
  view.setFloat32(232, PIXEL_ANGLE, true);
  view.setFloat32(236, STANDOFF_METRES, true);
  view.setUint32(240, QUADRATURE_SIDE, true);
  view.setUint32(244, 1, true);
  view.setUint32(248, 1, true);
  view.setUint32(252, 1, true);

  const descriptors = new Uint16Array(artifact.buffer, artifact.byteOffset + descriptorOffset, descriptorCount);
  const pairTable = new DataView(artifact.buffer, artifact.byteOffset + pairTableOffset, pairTableBytes);
  for (let pair = 0; pair < pairs.length; pair++) {
    const base = pair * blocksPerPair;
    const entry = pair * 16;
    pairTable.setUint8(entry, pairs[pair]!.entry);
    pairTable.setUint8(entry + 1, pairs[pair]!.exit);
    pairTable.setUint32(entry + 4, base, true);
    pairTable.setUint32(entry + 8, blocksPerPair, true);
    for (let local = 0; local < blocksPerPair; local++) descriptors[base + local] = base + local;
  }
  const scaleTable = new DataView(artifact.buffer, artifact.byteOffset + scaleTableOffset, scaleTableBytes);
  scaleTable.setFloat32(0, STANDOFF_METRES, true);
  scaleTable.setFloat32(4, PIXEL_ANGLE, true);
  scaleTable.setUint32(8, 0, true);
  scaleTable.setUint32(12, descriptorCount, true);
  scaleTable.setUint32(16, 0, true);
  scaleTable.setUint32(20, codebookCount, true);
  scaleTable.setUint32(24, atlasWidth, true);
  scaleTable.setUint32(28, atlasHeight, true);
  const color0 = new Uint16Array(artifact.buffer, artifact.byteOffset + color0Offset, atlasTexels * 4);
  const color1 = new Uint16Array(artifact.buffer, artifact.byteOffset + color1Offset, atlasTexels * 4);
  const categorical = new Uint16Array(artifact.buffer, artifact.byteOffset + categoricalOffset, atlasTexels * 4);
  for (let pair = 0; pair < pairs.length; pair++) {
    for (let b3 = 0; b3 < BLOCKS_PER_AXIS; b3++) for (let b2 = 0; b2 < BLOCKS_PER_AXIS; b2++) {
      for (let b1 = 0; b1 < BLOCKS_PER_AXIS; b1++) for (let b0 = 0; b0 < BLOCKS_PER_AXIS; b0++) {
        const localBlock = (((b3 * BLOCKS_PER_AXIS + b2) * BLOCKS_PER_AXIS + b1) * BLOCKS_PER_AXIS + b0);
        const codeword = pair * blocksPerPair + localBlock;
        const tileX = (codeword % atlasColumns) * tileEdge;
        const tileY = Math.floor(codeword / atlasColumns) * tileEdge;
        for (let l3 = 0; l3 < BLOCK_SAMPLE_EDGE; l3++) for (let l2 = 0; l2 < BLOCK_SAMPLE_EDGE; l2++) {
          for (let l1 = 0; l1 < BLOCK_SAMPLE_EDGE; l1++) for (let l0 = 0; l0 < BLOCK_SAMPLE_EDGE; l0++) {
            const i0 = b0 * BLOCK_CELL_EDGE + l0;
            const i1 = b1 * BLOCK_CELL_EDGE + l1;
            const i2 = b2 * BLOCK_CELL_EDGE + l2;
            const i3 = b3 * BLOCK_CELL_EDGE + l3;
            const node = nodeLinear(pair, i0, i1, i2, i3);
            const x = tileX + l0 + BLOCK_SAMPLE_EDGE * l2;
            const y = tileY + l1 + BLOCK_SAMPLE_EDGE * l3;
            const texel = (y * atlasWidth + x) * 4;
            for (let channel = 0; channel < 4; channel++) {
              color0[texel + channel] = float16(nodeColor0[node * 4 + channel]!);
              color1[texel + channel] = float16(nodeColor1[node * 4 + channel]!);
              categorical[texel + channel] = nodeCategorical[node * 4 + channel]!;
            }
          }
        }
      }
    }
  }

  // Decode-time structural validation: exact offsets, ids and atlas capacity.
  if (artifact.toString('ascii', 0, 4) !== 'GBR4' || view.getUint32(4, true) !== 1) throw new Error('GBR4 self-validation failed');
  if (descriptors.at(-1) !== codebookCount - 1 || atlasWidth * atlasHeight < codebookCount * tileEdge * tileEdge) throw new Error('GBR4 descriptor/atlas self-validation failed');
  const outputRoot = resolve(WORKSPACE, 'data/work/groundcover-gbr4-brick-assets', sourceSha.slice(0, 16), recipeSha.slice(0, 16));
  const qaRoot = resolve(outputRoot, 'qa');
  mkdirSync(qaRoot, { recursive: true });
  const assetPath = resolve(outputRoot, 'calamagrostis-canescens.gbr4');
  const stablePath = resolve(WORKSPACE, 'src/assets/groundcover/calamagrostis-canescens.finite-v1.gbr4');
  writeFileSync(assetPath, artifact);
  writeFileSync(stablePath, artifact);
  const qaWidth = 160;
  const qaHeight = 90;
  const qaRaw = Buffer.alloc(qaWidth * qaHeight * 3);
  for (let pixel = 0; pixel < qaWidth * qaHeight; pixel++) {
    const node = pixel % nodeCount;
    const alpha = nodeColor0[node * 4 + 3]! + nodeColor1[node * 4 + 3]!;
    for (let channel = 0; channel < 3; channel++) {
      const premul = nodeColor0[node * 4 + channel]! + nodeColor1[node * 4 + channel]!;
      qaRaw[pixel * 3 + channel] = Math.round(clamp(premul + (1 - alpha) * 0.1, 0, 1) * 255);
    }
  }
  const qaPath = resolve(qaRoot, '01-real-mesh-filtered-node-swatches.png');
  await sharp(qaRaw, { raw: { width: qaWidth, height: qaHeight, channels: 3 } }).resize(1280, 720, { kernel: 'nearest' }).png().toFile(qaPath);
  const report = {
    schema: 'laas-gbr4-minimal-real-asset/v1', created: new Date().toISOString(), recipe,
    source: { path: relative(WORKSPACE, SOURCE), sha256: sourceSha, vertices: geometry.vertexCount, triangles: geometry.triangleCount },
    container: { path: relative(WORKSPACE, assetPath), stablePath: relative(WORKSPACE, stablePath), sha256: sha256(artifact), bytes: artifact.byteLength, MiB: artifact.byteLength / 1048576 },
    address: { faces: FACE_COUNT, orderedPairs: pairs.length, cellResolution: CELL_RESOLUTION, blockCellEdge: BLOCK_CELL_EDGE, blocksPerPair, descriptorCount },
    codebook: { mode: 'lossless one codeword per block', count: codebookCount, samplesPerBrick: SAMPLES_PER_BRICK, tile: [tileEdge, tileEdge], atlas: [atlasWidth, atlasHeight] },
    sections: {
      header: { offset: 0, bytes: HEADER_BYTES }, descriptor: { offset: descriptorOffset, bytes: descriptorBytes, format: 'u16' },
      pairTable: { offset: pairTableOffset, bytes: pairTableBytes }, scaleTable: { offset: scaleTableOffset, bytes: scaleTableBytes },
      frontColor: { offset: color0Offset, bytes: atlasBytes, format: 'RGBA16F' }, backColor: { offset: color1Offset, bytes: atlasBytes, format: 'RGBA16F' },
      categorical: { offset: categoricalOffset, bytes: atlasBytes, format: 'RGBA16Uint' },
    },
    runtimeBudget: { logicalReads: 10, descriptorReads: 1, filteredColorReads: 8, categoricalReads: 1, estimatedScalarAluExcludingCommonRayBox: '<150', loops: 0, marches: 0, candidates: 0, runtimeGeometry: false },
    validity: {
      status: 'SUPERSEDED_FOR_PERIODIC_USE', runtimeLoadable: false, visualProductionReady: false,
      proven: ['real accepted mesh truth', 'exterior shared-origin filtering', 'deterministic binary layout', 'lossless block-border samples', 'fixed O(1) reconstruction schedule'],
      missing: ['periodic later-tile visibility; use GBR4/v2', 'production spatial resolution/VQ fit', 'complete footprint scale cascade', 'held-out visual/depth/normal/translation gate', 'runtime implementation'],
    },
    qa: { path: relative(WORKSPACE, qaPath), sha256: sha256(readFileSync(qaPath)) },
    elapsedMilliseconds: performance.now() - started,
  };
  const reportPath = resolve(outputRoot, 'report.json');
  writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify({ outputRoot: relative(WORKSPACE, outputRoot), report: relative(WORKSPACE, reportPath), container: report.container, validity: report.validity, runtimeBudget: report.runtimeBudget }, null, 2));
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await main();
