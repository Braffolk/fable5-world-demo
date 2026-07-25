/**
 * Offline-only gate for a one-read 32-bit categorical exterior light field.
 *
 * The source BVH is truth/bake machinery only. The proposed live operation is
 * one fixed two-plane line address, one nearest u32 read, and one height relift;
 * it contains no live traversal, loop, march, candidate set, or owner table.
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
const DEFAULT_SOURCE = resolve(WORKSPACE, 'src/assets/groundcover/calamagrostis-canescens.gcrp');
const EXPECTED_SOURCE_SHA256 =
  '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const RESIDENT_CAP_BYTES = 51_121_152;
const RECORD_BYTES = 4;
const GUTTER = 1;
const MINIMUM_ELEVATION_DEGREES = 5;
const MAXIMUM_SLOPE = 1 / Math.tan(MINIMUM_ELEVATION_DEGREES * Math.PI / 180);
const REFERENCE_PLANE_SEPARATION_METRES = 1;
const TAU = Math.PI * 2;

const PALETTE = [
  { rgb: [0.10, 0.27, 0.055] as const, family: 0, name: 'foliage-ribbon' },
  { rgb: [0.37, 0.51, 0.13] as const, family: 0, name: 'foliage-ribbon' },
  { rgb: [0.43, 0.53, 0.20] as const, family: 1, name: 'culm-tube' },
  { rgb: [0.31, 0.24, 0.075] as const, family: 2, name: 'rhizome-tube' },
  { rgb: [0.48, 0.31, 0.24] as const, family: 3, name: 'panicle-axis-tube' },
  { rgb: [0.57, 0.35, 0.39] as const, family: 4, name: 'spikelet-surface' },
  { rgb: [0.88, 0.80, 0.70] as const, family: 5, name: 'callus-hair-or-filament' },
  { rgb: [0.42, 0.10, 0.34] as const, family: 6, name: 'anther-surface' },
] as const;
const PANICLE_FAMILIES = new Set([3, 4, 5, 6]);

interface Allocation {
  key: string;
  label: string;
  phaseResolution: number;
  ringCounts: readonly number[];
}

const ALLOCATIONS: readonly Allocation[] = [
  {
    key: 'phase-rich-p255-d193',
    label: 'phase-rich P255 / D193',
    phaseResolution: 255,
    ringCounts: [3, 9, 15, 21, 27, 33, 39, 45],
  },
  {
    key: 'angle-rich-p220-d257',
    label: 'angle-rich P220 / D257',
    phaseResolution: 220,
    ringCounts: [3, 9, 16, 22, 28, 35, 41, 48, 54],
  },
] as const;

interface CameraSweepSpec {
  key: string;
  elevationDegrees: number;
  azimuthDegrees: number;
  horizontalFovDegrees: number;
  verticalFovDegrees: number;
}

const CAMERA_SWEEPS: readonly CameraSweepSpec[] = [
  { key: 'outside-domain-2.5deg', elevationDegrees: 2.5, azimuthDegrees: 11, horizontalFovDegrees: 3, verticalFovDegrees: 0.4 },
  { key: 'grazing-5.5deg', elevationDegrees: 5.5, azimuthDegrees: 73, horizontalFovDegrees: 3, verticalFovDegrees: 0.8 },
  { key: 'low-oblique-10deg', elevationDegrees: 10, azimuthDegrees: 137, horizontalFovDegrees: 4, verticalFovDegrees: 1.5 },
  { key: 'oblique-25deg', elevationDegrees: 25, azimuthDegrees: 211, horizontalFovDegrees: 5, verticalFovDegrees: 3 },
  { key: 'high-55deg', elevationDegrees: 55, azimuthDegrees: 293, horizontalFovDegrees: 6, verticalFovDegrees: 5 },
  { key: 'near-top-80deg', elevationDegrees: 80, azimuthDegrees: 337, horizontalFovDegrees: 6, verticalFovDegrees: 5 },
] as const;
const CAMERA_WIDTH = 24;
const CAMERA_HEIGHT = 16;
const QA_SCALE = 8;
const CAMERA_FRAME_OFFSETS_METRES = [0, 0.001, 0.0025, 0.0045] as const;
const ERROR_THRESHOLDS_METRES = [0.05, 0.25, 1] as const;

interface Args { [key: string]: string | boolean }

interface SourceAttributes {
  view: DataView;
  vertexOffset: number;
}

interface SurfaceSample {
  triangleId: number;
  copyX: number;
  copyZ: number;
  t: number;
  heightDrop: number;
  point: CensusVec3;
  normal: CensusVec3;
  color: CensusVec3;
  colorCode: number;
  family: number;
}

interface LineAddress {
  qTop: CensusVec3;
  qLower: CensusVec3;
  slopeX: number;
  slopeZ: number;
}

interface DirectionAddress {
  id: number;
  ring: number;
  sector: number;
  slopeX: number;
  slopeZ: number;
  direction: CensusVec3;
  outsideDomain: boolean;
}

interface AtlasAddress {
  key: string;
  phaseX: number;
  phaseZ: number;
  direction: DirectionAddress;
  canonicalTop: CensusVec3;
}

interface DecodedRecord {
  packed: number;
  hit: boolean;
  heightDrop: number;
  normal: CensusVec3;
  color: CensusVec3;
  colorCode: number;
  family: number;
  source?: SurfaceSample;
}

interface CameraRay {
  sweep: string;
  frame: number;
  x: number;
  y: number;
  origin: CensusVec3;
  direction: CensusVec3;
  line: LineAddress;
  truth: SurfaceSample | null;
}

interface PixelResult {
  ray: CameraRay;
  address: AtlasAddress;
  record: DecodedRecord;
  predictedPoint: CensusVec3 | null;
  positionError: number | null;
  normalDot: number | null;
  colorCodeExact: boolean;
  familyExact: boolean;
  panicleClassRetained: boolean;
}

interface MutableMetrics {
  rays: number;
  truthHits: number;
  predictedHits: number;
  truePositive: number;
  trueNegative: number;
  falsePositive: number;
  falseNegative: number;
  outsideDomain: number;
  positionErrors: number[];
  heightErrors: number[];
  normalDots: number[];
  slopeErrors: number[];
  phaseErrors: number[];
  exactColorCodes: number;
  exactFamilies: number;
  panicleTruthHits: number;
  panicleClassRetained: number;
  panicleExactFamily: number;
  panicleExactColorCode: number;
  fanWidthsByThreshold: Map<number, number[]>;
  fanPixelsByThreshold: Map<number, number[]>;
}

function parseArgs(argv: readonly string[]): Args {
  const result: Args = {};
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index]!;
    if (!argument.startsWith('--')) throw new Error(`unexpected positional argument ${argument}`);
    const value = argv[index + 1];
    if (value !== undefined && !value.startsWith('--')) {
      result[argument.slice(2)] = value;
      index++;
    } else result[argument.slice(2)] = true;
  }
  return result;
}

function sha256(bytes: Uint8Array | string): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) =>
    `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(',')}}`;
}

function wrapped(value: number): number {
  return value - Math.floor(value);
}

function positiveModulo(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus;
}

function normalized(value: CensusVec3): CensusVec3 {
  const length = Math.hypot(value[0], value[1], value[2]);
  if (!(length > 0)) throw new Error('cannot normalize zero vector');
  return [value[0] / length, value[1] / length, value[2] / length];
}

function dot(left: CensusVec3, right: CensusVec3): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function distance(left: CensusVec3, right: CensusVec3): number {
  return Math.hypot(left[0] - right[0], left[1] - right[1], left[2] - right[2]);
}

function quantiles(valuesInput: readonly number[]): Record<string, number | null> {
  if (valuesInput.length === 0) return { p50: null, p90: null, p95: null, p99: null, maximum: null };
  const values = [...valuesInput].sort((left, right) => left - right);
  const at = (fraction: number): number => values[Math.floor((values.length - 1) * fraction)]!;
  return { p50: at(0.5), p90: at(0.9), p95: at(0.95), p99: at(0.99), maximum: values.at(-1)! };
}

function sourceAttributes(bytes: Uint8Array): SourceAttributes {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return { view, vertexOffset: view.getUint32(84, true) };
}

function decodeOct01(x01: number, y01: number): CensusVec3 {
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

function vertexColor(attributes: SourceAttributes, vertex: number): CensusVec3 {
  const offset = attributes.vertexOffset + vertex * 16;
  return [
    attributes.view.getUint16(offset + 6, true) / 65535,
    attributes.view.getUint16(offset + 8, true) / 65535,
    attributes.view.getUint16(offset + 10, true) / 65535,
  ];
}

function vertexNormal(attributes: SourceAttributes, vertex: number): CensusVec3 {
  const offset = attributes.vertexOffset + vertex * 16;
  return decodeOct01(
    attributes.view.getUint16(offset + 12, true) / 65535,
    attributes.view.getUint16(offset + 14, true) / 65535,
  );
}

function colorCode(color: CensusVec3): number {
  const r = Math.max(0, Math.min(7, Math.round(color[0] * 7)));
  const g = Math.max(0, Math.min(15, Math.round(color[1] * 15)));
  const b = Math.max(0, Math.min(7, Math.round(color[2] * 7)));
  return (r << 7) | (g << 3) | b;
}

function decodeColorCode(code: number): CensusVec3 {
  return [((code >>> 7) & 7) / 7, ((code >>> 3) & 15) / 15, (code & 7) / 7];
}

function nearestFamily(color: CensusVec3): number {
  let bestFamily = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const entry of PALETTE) {
    const d = (color[0] - entry.rgb[0]) ** 2
      + (color[1] - entry.rgb[1]) ** 2
      + (color[2] - entry.rgb[2]) ** 2;
    if (d < bestDistance) {
      bestDistance = d;
      bestFamily = entry.family;
    }
  }
  return bestFamily;
}

function surfaceSample(
  geometry: DecodedOwnedProfileGeometry,
  attributes: SourceAttributes,
  origin: CensusVec3,
  direction: CensusVec3,
  hit: OriginAwareTruthHit,
): SurfaceSample {
  const point: CensusVec3 = [
    origin[0] + direction[0] * hit.t,
    origin[1] + direction[1] * hit.t,
    origin[2] + direction[2] * hit.t,
  ];
  const triangle = hit.triangleId * 3;
  const vertices = [
    geometry.triangles[triangle]!,
    geometry.triangles[triangle + 1]!,
    geometry.triangles[triangle + 2]!,
  ] as const;
  const localPoint: CensusVec3 = [
    point[0] - hit.copyX * geometry.tileSizeX,
    point[1],
    point[2] - hit.copyZ * geometry.tileSizeZ,
  ];
  const position = (vertex: number): CensusVec3 => {
    const offset = vertex * 3;
    return [geometry.positions[offset]!, geometry.positions[offset + 1]!, geometry.positions[offset + 2]!];
  };
  const a = position(vertices[0]);
  const b = position(vertices[1]);
  const c = position(vertices[2]);
  const e0: CensusVec3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const e1: CensusVec3 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const q: CensusVec3 = [localPoint[0] - a[0], localPoint[1] - a[1], localPoint[2] - a[2]];
  const d00 = dot(e0, e0);
  const d01 = dot(e0, e1);
  const d11 = dot(e1, e1);
  const d20 = dot(q, e0);
  const d21 = dot(q, e1);
  const denominator = d00 * d11 - d01 * d01;
  const u = denominator !== 0 ? (d11 * d20 - d01 * d21) / denominator : 0;
  const v = denominator !== 0 ? (d00 * d21 - d01 * d20) / denominator : 0;
  const weights = [1 - u - v, u, v] as const;
  const colors = vertices.map((vertex) => vertexColor(attributes, vertex));
  const normals = vertices.map((vertex) => vertexNormal(attributes, vertex));
  const color: CensusVec3 = [0, 1, 2].map((channel) =>
    weights[0] * colors[0]![channel]! + weights[1] * colors[1]![channel]! + weights[2] * colors[2]![channel]!,
  ) as unknown as CensusVec3;
  let normal = normalized([0, 1, 2].map((channel) =>
    weights[0] * normals[0]![channel]! + weights[1] * normals[1]![channel]! + weights[2] * normals[2]![channel]!,
  ) as unknown as CensusVec3);
  if (dot(normal, direction) > 0) normal = [-normal[0], -normal[1], -normal[2]];
  const code = colorCode(color);
  return {
    triangleId: hit.triangleId,
    copyX: hit.copyX,
    copyZ: hit.copyZ,
    t: hit.t,
    heightDrop: geometry.topH - point[1],
    point,
    normal,
    color,
    colorCode: code,
    family: nearestFamily(color),
  };
}

function traceSurface(
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  attributes: SourceAttributes,
  origin: CensusVec3,
  direction: CensusVec3,
): SurfaceSample | null {
  if (!(direction[1] < 0)) return null;
  const horizon = (origin[1] - geometry.bounds.min[1] + 1e-8) / -direction[1];
  const hit = periodicNearestSuccessor(geometry, bvh, origin, direction, horizon, 0);
  return hit ? surfaceSample(geometry, attributes, origin, direction, hit) : null;
}

function lineAddress(origin: CensusVec3, direction: CensusVec3, topH: number): LineAddress {
  if (!(Math.abs(direction[1]) > 1e-12)) throw new Error('horizontal line has no finite two-Y-plane address');
  const intersect = (height: number): CensusVec3 => {
    const t = (height - origin[1]) / direction[1];
    return [origin[0] + direction[0] * t, height, origin[2] + direction[2] * t];
  };
  const qTop = intersect(topH);
  const qLower = intersect(topH - REFERENCE_PLANE_SEPARATION_METRES);
  return {
    qTop,
    qLower,
    slopeX: (qLower[0] - qTop[0]) / REFERENCE_PLANE_SEPARATION_METRES,
    slopeZ: (qLower[2] - qTop[2]) / REFERENCE_PLANE_SEPARATION_METRES,
  };
}

function directionCount(allocation: Allocation): number {
  return 1 + allocation.ringCounts.reduce((sum, value) => sum + value, 0);
}

function selectDirection(allocation: Allocation, slopeX: number, slopeZ: number): DirectionAddress {
  const radius = Math.hypot(slopeX, slopeZ);
  const radialStep = MAXIMUM_SLOPE / allocation.ringCounts.length;
  const outsideDomain = radius > MAXIMUM_SLOPE;
  if (radius < radialStep * 0.5) {
    return { id: 0, ring: -1, sector: 0, slopeX: 0, slopeZ: 0, direction: [0, -1, 0], outsideDomain };
  }
  const ring = Math.max(0, Math.min(allocation.ringCounts.length - 1, Math.floor(radius / radialStep)));
  const sectors = allocation.ringCounts[ring]!;
  const offsetTurns = (ring & 1) === 0 ? 0 : 0.5 / sectors;
  const thetaTurns = positiveModulo(Math.atan2(slopeZ, slopeX) / TAU, 1);
  const sector = positiveModulo(Math.round((thetaTurns - offsetTurns) * sectors), sectors);
  const angle = (sector / sectors + offsetTurns) * TAU;
  const canonicalRadius = (ring + 0.5) * radialStep;
  const canonicalSlopeX = canonicalRadius * Math.cos(angle);
  const canonicalSlopeZ = canonicalRadius * Math.sin(angle);
  let id = 1;
  for (let index = 0; index < ring; index++) id += allocation.ringCounts[index]!;
  id += sector;
  return {
    id,
    ring,
    sector,
    slopeX: canonicalSlopeX,
    slopeZ: canonicalSlopeZ,
    direction: normalized([canonicalSlopeX, -1, canonicalSlopeZ]),
    outsideDomain,
  };
}

function atlasAddress(
  geometry: DecodedOwnedProfileGeometry,
  allocation: Allocation,
  line: LineAddress,
): AtlasAddress {
  const u = wrapped((line.qTop[0] - geometry.tileOriginX) / geometry.tileSizeX);
  const v = wrapped((line.qTop[2] - geometry.tileOriginZ) / geometry.tileSizeZ);
  const phaseX = Math.floor(u * allocation.phaseResolution);
  const phaseZ = Math.floor(v * allocation.phaseResolution);
  const direction = selectDirection(allocation, line.slopeX, line.slopeZ);
  const canonicalTop: CensusVec3 = [
    geometry.tileOriginX + (phaseX + 0.5) / allocation.phaseResolution * geometry.tileSizeX,
    geometry.topH,
    geometry.tileOriginZ + (phaseZ + 0.5) / allocation.phaseResolution * geometry.tileSizeZ,
  ];
  return {
    key: `${direction.id}:${phaseX}:${phaseZ}`,
    phaseX,
    phaseZ,
    direction,
    canonicalTop,
  };
}

function encodeOct5(normal: CensusVec3): readonly [number, number] {
  const l1 = Math.max(1e-12, Math.abs(normal[0]) + Math.abs(normal[1]) + Math.abs(normal[2]));
  let x = normal[0] / l1;
  let y = normal[1] / l1;
  if (normal[2] < 0) {
    const oldX = x;
    x = (1 - Math.abs(y)) * (oldX >= 0 ? 1 : -1);
    y = (1 - Math.abs(oldX)) * (y >= 0 ? 1 : -1);
  }
  return [
    Math.max(0, Math.min(31, Math.round((x * 0.5 + 0.5) * 31))),
    Math.max(0, Math.min(31, Math.round((y * 0.5 + 0.5) * 31))),
  ];
}

function packRecord(sample: SurfaceSample | null, verticalSpan: number): number {
  if (!sample) return 0;
  const height = Math.max(0, Math.min(2047, Math.round(sample.heightDrop / verticalSpan * 2047)));
  const [octX, octY] = encodeOct5(sample.normal);
  return (1 | (height << 1) | (octX << 12) | (octY << 17) | (sample.colorCode << 22)) >>> 0;
}

function decodeRecord(packed: number, verticalSpan: number, source?: SurfaceSample): DecodedRecord {
  const hit = (packed & 1) !== 0;
  if (!hit) return {
    packed, hit: false, heightDrop: 0, normal: [0, 1, 0], color: [0, 0, 0], colorCode: 0, family: -1,
  };
  const heightDrop = ((packed >>> 1) & 2047) / 2047 * verticalSpan;
  const normal = decodeOct01(((packed >>> 12) & 31) / 31, ((packed >>> 17) & 31) / 31);
  const code = (packed >>> 22) & 1023;
  const color = decodeColorCode(code);
  return { packed, hit, heightDrop, normal, color, colorCode: code, family: nearestFamily(color), source };
}

function emptyMetrics(): MutableMetrics {
  return {
    rays: 0,
    truthHits: 0,
    predictedHits: 0,
    truePositive: 0,
    trueNegative: 0,
    falsePositive: 0,
    falseNegative: 0,
    outsideDomain: 0,
    positionErrors: [],
    heightErrors: [],
    normalDots: [],
    slopeErrors: [],
    phaseErrors: [],
    exactColorCodes: 0,
    exactFamilies: 0,
    panicleTruthHits: 0,
    panicleClassRetained: 0,
    panicleExactFamily: 0,
    panicleExactColorCode: 0,
    fanWidthsByThreshold: new Map(ERROR_THRESHOLDS_METRES.map((threshold) => [threshold, []])),
    fanPixelsByThreshold: new Map(ERROR_THRESHOLDS_METRES.map((threshold) => [threshold, []])),
  };
}

function addPixel(metrics: MutableMetrics, pixel: PixelResult, geometry: DecodedOwnedProfileGeometry): void {
  metrics.rays++;
  if (pixel.ray.truth) metrics.truthHits++;
  if (pixel.record.hit) metrics.predictedHits++;
  if (pixel.address.direction.outsideDomain) metrics.outsideDomain++;
  metrics.slopeErrors.push(Math.hypot(
    pixel.ray.line.slopeX - pixel.address.direction.slopeX,
    pixel.ray.line.slopeZ - pixel.address.direction.slopeZ,
  ));
  const wrappedPhaseDistance = (value: number, target: number, size: number): number => {
    const direct = positiveModulo(Math.abs(value - target), size);
    return Math.min(direct, size - direct);
  };
  metrics.phaseErrors.push(Math.hypot(
    wrappedPhaseDistance(pixel.ray.line.qTop[0], pixel.address.canonicalTop[0], geometry.tileSizeX),
    wrappedPhaseDistance(pixel.ray.line.qTop[2], pixel.address.canonicalTop[2], geometry.tileSizeZ),
  ));
  if (pixel.ray.truth && pixel.record.hit) {
    metrics.truePositive++;
    metrics.positionErrors.push(pixel.positionError!);
    metrics.heightErrors.push(Math.abs(pixel.record.heightDrop - pixel.ray.truth.heightDrop));
    metrics.normalDots.push(pixel.normalDot!);
    if (pixel.colorCodeExact) metrics.exactColorCodes++;
    if (pixel.familyExact) metrics.exactFamilies++;
    if (PANICLE_FAMILIES.has(pixel.ray.truth.family)) {
      metrics.panicleTruthHits++;
      if (pixel.panicleClassRetained) metrics.panicleClassRetained++;
      if (pixel.familyExact) metrics.panicleExactFamily++;
      if (pixel.colorCodeExact) metrics.panicleExactColorCode++;
    }
  } else if (!pixel.ray.truth && !pixel.record.hit) metrics.trueNegative++;
  else if (!pixel.ray.truth) metrics.falsePositive++;
  else {
    metrics.falseNegative++;
    if (PANICLE_FAMILIES.has(pixel.ray.truth.family)) metrics.panicleTruthHits++;
  }
}

function mergeMetrics(target: MutableMetrics, source: MutableMetrics): void {
  for (const key of [
    'rays', 'truthHits', 'predictedHits', 'truePositive', 'trueNegative', 'falsePositive', 'falseNegative',
    'outsideDomain', 'exactColorCodes', 'exactFamilies', 'panicleTruthHits', 'panicleClassRetained',
    'panicleExactFamily', 'panicleExactColorCode',
  ] as const) target[key] += source[key];
  target.positionErrors.push(...source.positionErrors);
  target.heightErrors.push(...source.heightErrors);
  target.normalDots.push(...source.normalDots);
  target.slopeErrors.push(...source.slopeErrors);
  target.phaseErrors.push(...source.phaseErrors);
  for (const threshold of ERROR_THRESHOLDS_METRES) {
    target.fanWidthsByThreshold.get(threshold)!.push(...source.fanWidthsByThreshold.get(threshold)!);
    target.fanPixelsByThreshold.get(threshold)!.push(...source.fanPixelsByThreshold.get(threshold)!);
  }
}

function summarize(metrics: MutableMetrics): Record<string, unknown> {
  const truePositive = Math.max(1, metrics.truePositive);
  const panicle = Math.max(1, metrics.panicleTruthHits);
  return {
    rays: metrics.rays,
    truthHits: metrics.truthHits,
    predictedHits: metrics.predictedHits,
    truePositive: metrics.truePositive,
    trueNegative: metrics.trueNegative,
    falsePositive: metrics.falsePositive,
    falseNegative: metrics.falseNegative,
    hitMissAgreement: metrics.rays > 0 ? (metrics.truePositive + metrics.trueNegative) / metrics.rays : 1,
    hitPrecision: metrics.predictedHits > 0 ? metrics.truePositive / metrics.predictedHits : 1,
    hitRecall: metrics.truthHits > 0 ? metrics.truePositive / metrics.truthHits : 1,
    outsideDirectionDomainFraction: metrics.rays > 0 ? metrics.outsideDomain / metrics.rays : 0,
    positionErrorMetres: quantiles(metrics.positionErrors),
    heightErrorMetres: quantiles(metrics.heightErrors),
    oneMinusNormalDot: quantiles(metrics.normalDots.map((value) => 1 - value)),
    slopeCoordinateError: quantiles(metrics.slopeErrors),
    phaseCoordinateErrorMetres: quantiles(metrics.phaseErrors),
    exactRgb343ClassFraction: metrics.exactColorCodes / truePositive,
    exactSemanticFamilyFraction: metrics.exactFamilies / truePositive,
    panicle: {
      truthHits: metrics.panicleTruthHits,
      panicleVsVegetativeRetained: metrics.panicleClassRetained / panicle,
      exactSemanticFamily: metrics.panicleExactFamily / panicle,
      exactRgb343Class: metrics.panicleExactColorCode / panicle,
    },
    connectedAngularBinErrorFans: Object.fromEntries(ERROR_THRESHOLDS_METRES.map((threshold) => [
      `${threshold}m`,
      {
        components: metrics.fanWidthsByThreshold.get(threshold)!.length,
        widthMetres: quantiles(metrics.fanWidthsByThreshold.get(threshold)!),
        pixels: quantiles(metrics.fanPixelsByThreshold.get(threshold)!),
      },
    ])),
  };
}

function cameraDirection(spec: CameraSweepSpec, pixelX: number, pixelY: number): CensusVec3 {
  const azimuth = spec.azimuthDegrees * Math.PI / 180;
  const elevation = spec.elevationDegrees * Math.PI / 180;
  const forward: CensusVec3 = [Math.cos(elevation) * Math.cos(azimuth), -Math.sin(elevation), Math.cos(elevation) * Math.sin(azimuth)];
  const right: CensusVec3 = [-Math.sin(azimuth), 0, Math.cos(azimuth)];
  const up: CensusVec3 = [Math.sin(elevation) * Math.cos(azimuth), Math.cos(elevation), Math.sin(elevation) * Math.sin(azimuth)];
  const x = ((pixelX + 0.5) / CAMERA_WIDTH * 2 - 1) * Math.tan(spec.horizontalFovDegrees * Math.PI / 360);
  const y = (1 - (pixelY + 0.5) / CAMERA_HEIGHT * 2) * Math.tan(spec.verticalFovDegrees * Math.PI / 360);
  return normalized([
    forward[0] + right[0] * x + up[0] * y,
    forward[1] + right[1] * x + up[1] * y,
    forward[2] + right[2] * x + up[2] * y,
  ]);
}

function generateCameraRays(
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  attributes: SourceAttributes,
): Map<string, CameraRay[][]> {
  const result = new Map<string, CameraRay[][]>();
  for (const spec of CAMERA_SWEEPS) {
    const azimuth = spec.azimuthDegrees * Math.PI / 180;
    const perpendicular: CensusVec3 = [-Math.sin(azimuth), 0, Math.cos(azimuth)];
    const frames: CameraRay[][] = [];
    for (let frame = 0; frame < CAMERA_FRAME_OFFSETS_METRES.length; frame++) {
      const shift = CAMERA_FRAME_OFFSETS_METRES[frame]!;
      const origin: CensusVec3 = [
        geometry.tileOriginX + geometry.tileSizeX * 0.381966 + perpendicular[0] * shift,
        geometry.topH + 0.4,
        geometry.tileOriginZ + geometry.tileSizeZ * 0.618034 + perpendicular[2] * shift,
      ];
      const pixels: CameraRay[] = [];
      for (let y = 0; y < CAMERA_HEIGHT; y++) {
        for (let x = 0; x < CAMERA_WIDTH; x++) {
          const direction = cameraDirection(spec, x, y);
          const line = lineAddress(origin, direction, geometry.topH);
          const truth = traceSurface(geometry, bvh, attributes, line.qTop, direction);
          pixels.push({ sweep: spec.key, frame, x, y, origin, direction, line, truth });
        }
      }
      frames.push(pixels);
      console.error(`[packed-light-field] truth ${spec.key} frame ${frame + 1}/${CAMERA_FRAME_OFFSETS_METRES.length}`);
    }
    result.set(spec.key, frames);
  }
  return result;
}

function evaluatePixel(
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  attributes: SourceAttributes,
  allocation: Allocation,
  cache: Map<string, DecodedRecord>,
  ray: CameraRay,
): PixelResult {
  const address = atlasAddress(geometry, allocation, ray.line);
  let record = cache.get(address.key);
  if (!record) {
    const source = traceSurface(geometry, bvh, attributes, address.canonicalTop, address.direction.direction);
    const packed = packRecord(source, geometry.topH - geometry.bounds.min[1]);
    record = decodeRecord(packed, geometry.topH - geometry.bounds.min[1], source ?? undefined);
    cache.set(address.key, record);
  }
  const predictedT = record.hit ? record.heightDrop / -ray.direction[1] : null;
  const predictedPoint: CensusVec3 | null = predictedT === null ? null : [
    ray.line.qTop[0] + ray.direction[0] * predictedT,
    geometry.topH + ray.direction[1] * predictedT,
    ray.line.qTop[2] + ray.direction[2] * predictedT,
  ];
  const positionError = ray.truth && predictedPoint ? distance(ray.truth.point, predictedPoint) : null;
  const normalDot = ray.truth && record.hit ? dot(ray.truth.normal, record.normal) : null;
  const colorCodeExact = ray.truth !== null && record.hit && ray.truth.colorCode === record.colorCode;
  const familyExact = ray.truth !== null && record.hit && ray.truth.family === record.family;
  return {
    ray,
    address,
    record,
    predictedPoint,
    positionError,
    normalDot,
    colorCodeExact,
    familyExact,
    panicleClassRetained: ray.truth !== null && PANICLE_FAMILIES.has(ray.truth.family) && PANICLE_FAMILIES.has(record.family),
  };
}

function addFanComponents(metrics: MutableMetrics, pixels: readonly PixelResult[]): void {
  for (const threshold of ERROR_THRESHOLDS_METRES) {
    const visited = new Uint8Array(pixels.length);
    for (let seed = 0; seed < pixels.length; seed++) {
      if (visited[seed]) continue;
      const first = pixels[seed]!;
      if (!(first.positionError !== null && first.positionError >= threshold && first.predictedPoint)) continue;
      const directionId = first.address.direction.id;
      const stack = [seed];
      visited[seed] = 1;
      const points: CensusVec3[] = [];
      while (stack.length > 0) {
        const index = stack.pop()!;
        const pixel = pixels[index]!;
        points.push(pixel.predictedPoint!);
        const x = index % CAMERA_WIDTH;
        const y = Math.floor(index / CAMERA_WIDTH);
        for (const neighbor of [
          x > 0 ? index - 1 : -1,
          x + 1 < CAMERA_WIDTH ? index + 1 : -1,
          y > 0 ? index - CAMERA_WIDTH : -1,
          y + 1 < CAMERA_HEIGHT ? index + CAMERA_WIDTH : -1,
        ]) {
          if (neighbor < 0 || visited[neighbor]) continue;
          const candidate = pixels[neighbor]!;
          if (
            candidate.address.direction.id === directionId
            && candidate.positionError !== null
            && candidate.positionError >= threshold
            && candidate.predictedPoint
          ) {
            visited[neighbor] = 1;
            stack.push(neighbor);
          }
        }
      }
      if (points.length < 2) continue;
      const minimum = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
      const maximum = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
      for (const point of points) {
        for (let axis = 0; axis < 3; axis++) {
          minimum[axis] = Math.min(minimum[axis]!, point[axis]!);
          maximum[axis] = Math.max(maximum[axis]!, point[axis]!);
        }
      }
      metrics.fanWidthsByThreshold.get(threshold)!.push(Math.hypot(
        maximum[0]! - minimum[0]!, maximum[1]! - minimum[1]!, maximum[2]! - minimum[2]!,
      ));
      metrics.fanPixelsByThreshold.get(threshold)!.push(points.length);
    }
  }
}

function rgb8(color: CensusVec3 | null): readonly [number, number, number] {
  if (!color) return [28, 24, 20];
  return color.map((value) => Math.max(0, Math.min(255, Math.round(Math.pow(value, 1 / 2.2) * 255)))) as unknown as readonly [number, number, number];
}

function diagnosticRow(pixels: readonly PixelResult[]): Uint8Array {
  const width = CAMERA_WIDTH * 3;
  const output = new Uint8Array(width * CAMERA_HEIGHT * 3);
  for (const pixel of pixels) {
    const truth = rgb8(pixel.ray.truth?.color ?? null);
    const predicted = rgb8(pixel.record.hit ? pixel.record.color : null);
    let error: readonly [number, number, number];
    if (pixel.ray.truth === null && !pixel.record.hit) error = [20, 32, 20];
    else if (pixel.positionError === null) error = [255, 0, 255];
    else {
      const value = pixel.positionError;
      error = value < 0.05 ? [20, 160, 40]
        : value < 0.25 ? [230, 210, 20]
          : value < 1 ? [240, 100, 10] : [220, 15, 15];
    }
    for (const [panel, color] of [[0, truth], [1, predicted], [2, error]] as const) {
      const x = panel * CAMERA_WIDTH + pixel.ray.x;
      const offset = (pixel.ray.y * width + x) * 3;
      output[offset] = color[0];
      output[offset + 1] = color[1];
      output[offset + 2] = color[2];
    }
  }
  return output;
}

async function writeContactSheet(path: string, rows: readonly Uint8Array[]): Promise<void> {
  const rowWidth = CAMERA_WIDTH * 3;
  const rowHeight = CAMERA_HEIGHT;
  const output = new Uint8Array(rowWidth * rowHeight * rows.length * 3);
  rows.forEach((row, rowIndex) => output.set(row, rowIndex * rowWidth * rowHeight * 3));
  await sharp(output, { raw: { width: rowWidth, height: rowHeight * rows.length, channels: 3 } })
    .resize(rowWidth * QA_SCALE, rowHeight * rows.length * QA_SCALE, { kernel: 'nearest' })
    .png().toFile(path);
}

function lineDatumAudit(
  geometry: DecodedOwnedProfileGeometry,
  allocation: Allocation,
  rays: readonly CameraRay[],
): Record<string, unknown> {
  let addressMismatches = 0;
  let maximumPlanePointDrift = 0;
  let tested = 0;
  for (let index = 0; index < rays.length; index += Math.max(1, Math.floor(rays.length / 1024))) {
    const ray = rays[index]!;
    const base = atlasAddress(geometry, allocation, ray.line);
    for (const shift of [-0.37, 0.19, 1.31]) {
      const shifted: CensusVec3 = [
        ray.origin[0] + ray.direction[0] * shift,
        ray.origin[1] + ray.direction[1] * shift,
        ray.origin[2] + ray.direction[2] * shift,
      ];
      const shiftedLine = lineAddress(shifted, ray.direction, geometry.topH);
      const shiftedAddress = atlasAddress(geometry, allocation, shiftedLine);
      maximumPlanePointDrift = Math.max(
        maximumPlanePointDrift,
        distance(ray.line.qTop, shiftedLine.qTop),
        distance(ray.line.qLower, shiftedLine.qLower),
      );
      if (base.key !== shiftedAddress.key) addressMismatches++;
      tested++;
    }
  }
  return {
    testedDatumShifts: tested,
    addressMismatches,
    maximumFixedPlaneIntersectionDriftMetres: maximumPlanePointDrift,
    invariant: addressMismatches === 0,
  };
}

const args = parseArgs(process.argv.slice(2));
const sourcePath = resolve(WORKSPACE, typeof args.source === 'string' ? args.source : DEFAULT_SOURCE);
const sourceBytes = readFileSync(sourcePath);
const sourceSha256 = sha256(sourceBytes);
if (sourceSha256 !== EXPECTED_SOURCE_SHA256) {
  throw new Error(`unexpected source SHA-256 ${sourceSha256}`);
}
const scriptPath = fileURLToPath(import.meta.url);
const scriptSha256 = sha256(readFileSync(scriptPath));
const configuration = {
  schema: 'laas-packed-categorical-light-field-gate/v1',
  sourceSha256,
  residentCapBytes: RESIDENT_CAP_BYTES,
  record: {
    bytes: RECORD_BYTES,
    bits: { hitCoverage: 1, verticalDrop: 11, octNormalX: 5, octNormalY: 5, rgb343MaterialClass: 10 },
    spatialFilter: 'nearest categorical',
    angularFilter: 'nearest fixed radial-band/two-plane-slope category',
  },
  fixedPlanes: { upper: 'profile topH', separationMetres: REFERENCE_PLANE_SEPARATION_METRES },
  minimumExteriorElevationDegrees: MINIMUM_ELEVATION_DEGREES,
  allocations: ALLOCATIONS,
  camera: {
    width: CAMERA_WIDTH,
    height: CAMERA_HEIGHT,
    frameOffsetsMetres: CAMERA_FRAME_OFFSETS_METRES,
    sweeps: CAMERA_SWEEPS,
  },
  errorThresholdsMetres: ERROR_THRESHOLDS_METRES,
  scriptSha256,
};
const recipeSha256 = sha256(canonicalJson(configuration));
const outputRoot = resolve(
  WORKSPACE,
  typeof args.output === 'string'
    ? args.output
    : `data/work/groundcover-packed-light-field/${sourceSha256.slice(0, 16)}/${recipeSha256.slice(0, 16)}`,
);
mkdirSync(resolve(outputRoot, 'qa'), { recursive: true });

const geometry = decodeOwnedProfileGeometry(sourceBytes);
const attributes = sourceAttributes(sourceBytes);
const bvh = TriangleBvh.build(geometry, 8);
console.error(`[packed-light-field] BVH ${bvh.metrics.buildMilliseconds.toFixed(0)} ms`);
const traceStarted = performance.now();
const cameraRays = generateCameraRays(geometry, bvh, attributes);
const allRays = [...cameraRays.values()].flat(2);
const allocationReports: Record<string, unknown> = {};
const qaFiles: Array<{ file: string; width: number; height: number; interpretation: string }> = [];

for (const allocation of ALLOCATIONS) {
  const cache = new Map<string, DecodedRecord>();
  const aggregate = emptyMetrics();
  const bySweep: Record<string, unknown> = {};
  const diagnosticRows: Uint8Array[] = [];
  let unforcedSemanticChanges = 0;
  let stableTruthComparisons = 0;
  for (const spec of CAMERA_SWEEPS) {
    const frames = cameraRays.get(spec.key)!;
    const framePixels: PixelResult[][] = [];
    const sweepMetrics = emptyMetrics();
    let worstRow: Uint8Array | null = null;
    let worstMean = -1;
    for (const rays of frames) {
      const pixels = rays.map((ray) => evaluatePixel(geometry, bvh, attributes, allocation, cache, ray));
      const frameMetrics = emptyMetrics();
      for (const pixel of pixels) addPixel(frameMetrics, pixel, geometry);
      addFanComponents(frameMetrics, pixels);
      mergeMetrics(sweepMetrics, frameMetrics);
      framePixels.push(pixels);
      const mean = frameMetrics.positionErrors.reduce((sum, value) => sum + value, 0)
        / Math.max(1, frameMetrics.positionErrors.length);
      if (mean > worstMean) {
        worstMean = mean;
        worstRow = diagnosticRow(pixels);
      }
    }
    for (let frame = 1; frame < framePixels.length; frame++) {
      for (let pixel = 0; pixel < framePixels[frame]!.length; pixel++) {
        const prior = framePixels[frame - 1]![pixel]!;
        const current = framePixels[frame]![pixel]!;
        if (prior.ray.truth?.family === current.ray.truth?.family && prior.ray.truth !== null && current.ray.truth !== null) {
          stableTruthComparisons++;
          if (prior.record.family !== current.record.family) unforcedSemanticChanges++;
        }
      }
    }
    diagnosticRows.push(worstRow!);
    mergeMetrics(aggregate, sweepMetrics);
    bySweep[spec.key] = summarize(sweepMetrics);
    console.error(`[packed-light-field] ${allocation.key} ${spec.key} cache=${cache.size}`);
  }
  const qaFile = `qa/${String(qaFiles.length + 1).padStart(3, '0')}-${allocation.key}-truth-prediction-error.png`;
  await writeContactSheet(resolve(outputRoot, qaFile), diagnosticRows);
  qaFiles.push({
    file: qaFile,
    width: CAMERA_WIDTH * 3 * QA_SCALE,
    height: CAMERA_HEIGHT * CAMERA_SWEEPS.length * QA_SCALE,
    interpretation: 'rows follow configured camera sweeps; panels are truth authored colour, decoded RGB343 prediction, and error class (<5cm green, <25cm yellow, <1m orange, >=1m red, hit/miss mismatch magenta)',
  });
  const directions = directionCount(allocation);
  const storedSide = allocation.phaseResolution + GUTTER * 2;
  const storedRecords = storedSide * storedSide * directions;
  const storedBytes = storedRecords * RECORD_BYTES;
  allocationReports[allocation.key] = {
    label: allocation.label,
    allocation: {
      phaseResolution: allocation.phaseResolution,
      phaseTexelMetres: [geometry.tileSizeX / allocation.phaseResolution, geometry.tileSizeZ / allocation.phaseResolution],
      directionCount: directions,
      rings: allocation.ringCounts.length,
      ringCounts: allocation.ringCounts,
      maximumSlope: MAXIMUM_SLOPE,
      storedSide,
      storedRecords,
      storedBytes,
      capRemainingBytes: RESIDENT_CAP_BYTES - storedBytes,
      fixedTextureReads: 1,
      liveAddress: 'two fixed-Y-plane intersections -> periodic top phase + radial-band slope ring/sector -> one nearest u32',
    },
    aggregate: summarize(aggregate),
    bySweep,
    temporalSweep: {
      stableTruthSemanticComparisons: stableTruthComparisons,
      unforcedPredictedSemanticChanges: unforcedSemanticChanges,
      fraction: unforcedSemanticChanges / Math.max(1, stableTruthComparisons),
    },
    lineDatumInvariance: lineDatumAudit(geometry, allocation, allRays),
    generatedCanonicalRecords: cache.size,
    qaFile,
  };
}

const report = {
  schema: configuration.schema,
  source: {
    path: relative(WORKSPACE, sourcePath),
    sha256: sourceSha256,
    bytes: sourceBytes.byteLength,
    triangles: geometry.triangleCount,
    tileSizeMetres: [geometry.tileSizeX, geometry.tileSizeZ],
    topH: geometry.topH,
    minimumY: geometry.bounds.min[1],
  },
  reproducibility: { configuration, recipeSha256 },
  accelerator: bvh.metrics,
  traceMilliseconds: performance.now() - traceStarted,
  allocationReports,
  exactRuntimeCostContract: {
    textureReads: 1,
    sourceTexels: 1,
    payloadBytes: 4,
    filtering: 'nearest categorical; no arithmetic mixture of depth, normal, colour, coverage, or owner',
    fixedAddressAlu: [
      'intersect the live line with two fixed botanical Y planes',
      'wrap one top-plane XZ phase',
      'one slope radius and atan2, one radial-band floor/clamp, one fixed 8/9-entry ring-count select, one sector round',
      'decode one u32 and divide vertical drop by live vertical speed',
    ],
    forbiddenAndAbsent: ['runtime loop', 'march', 'mesh traversal', 'candidate list', 'owner table read', 'species query', 'extra pass', 'distance-dependent work'],
  },
  cameraInsideBoundary: {
    solved: false,
    reason: 'the atlas stores the exterior first event from the fixed top plane; it does not store the pointed-line origin phase or an ordered successor field',
    exactHorizontal: 'a line parallel to both fixed Y planes has no finite top-entry address and belongs to a separate side-entry/pointed-line problem',
    implication: 'an exterior result cannot be presented as the complete arbitrary-camera ground-cover solution',
  },
  multiSpeciesBoundary: {
    recordCapacity: 'the 10-bit colour/material class can name 1024 categorical appearance classes, but it does not itself encode species/population/root eligibility',
    validComposition: 'offline-union overlapping species and moss into one categorical community winner field before packing',
    invalidComposition: 'one live atlas query per species or arithmetic blending of unrelated winners',
  },
  decisionRule: {
    plausibleOnlyIf: 'connected error fans are visually sub-centimetric/isolated, position tails do not form decimetre/metre sheets, and panicle class remains stable across distance and camera motion',
    noEscape: 'failure does not authorize more live samples, candidates, loops, marches, per-species queries, or a raised resident cap',
  },
};

const metricsPath = resolve(outputRoot, 'metrics.json');
const serialized = `${JSON.stringify(report, null, 2)}\n`;
writeFileSync(metricsPath, serialized);
const files = [
  { file: 'metrics.json', bytes: Buffer.byteLength(serialized), sha256: sha256(serialized) },
  ...qaFiles.map((file) => {
    const bytes = readFileSync(resolve(outputRoot, file.file));
    return { ...file, bytes: bytes.byteLength, sha256: sha256(bytes) };
  }),
];
const index = {
  schema: 'laas-packed-categorical-light-field-gate-index/v1',
  sourceSha256,
  recipeSha256,
  script: { file: relative(WORKSPACE, scriptPath), sha256: scriptSha256 },
  files,
};
const indexSerialized = `${JSON.stringify(index, null, 2)}\n`;
writeFileSync(resolve(outputRoot, 'index.json'), indexSerialized);
console.error(`[packed-light-field] wrote ${metricsPath}`);
console.error(`[packed-light-field] metrics sha256 ${sha256(serialized)}`);
process.stdout.write(`${outputRoot}\n`);
