/**
 * Offline-only gate for a literal exterior ground-cover radiance field.
 *
 * The predictor maps the exact live oriented ray to its intersection with the
 * fixed botanical top plane and its direction. Four angular lattice nodes are
 * sampled; every angular sample is itself the ordinary bilinear interpolation
 * of premultiplied authored RGB and binary coverage on the periodic phase
 * plane. Geometry, depth, owner, and normal are never interpolated into colour.
 *
 * All BVH traversal and loops in this file are offline truth/bake work. The
 * prospective live colour path is four spatially-filtered texture reads and a
 * fixed bilinear combination. This file does not edit or exercise a shader.
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
const CHARGED_BYTES_PER_CELL = 12;
const GUTTER = 1;
const AZIMUTH_COUNT = 16;
const TAU = Math.PI * 2;
const DEG = Math.PI / 180;
const CAMERA_WIDTH = 32;
const CAMERA_HEIGHT = 24;
const QA_SCALE = 7;
const BACKGROUND: CensusVec3 = [0.115, 0.09, 0.065];
const COVERAGE_THRESHOLD = 0.5;
const CAMERA_PATH_OFFSETS_METRES = [0, 0.001, 0.0025, 0.0045] as const;
const CAMERA_PATH_PIXEL_STRIDE = 4;

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
  elevationsDegrees: readonly number[];
  collapsedVerticalPole?: boolean;
}

const ALLOCATIONS: readonly Allocation[] = [
  {
    key: 'current-p256-a16-e4',
    label: 'current P256 / A16 / E4',
    phaseResolution: 256,
    elevationsDegrees: [15, 35, 55, 75],
  },
  {
    key: 'low-angle-p224-a16-e5',
    label: 'low-angle P224 / A16 / E5',
    phaseResolution: 224,
    elevationsDegrees: [5, 15, 35, 60, 85],
  },
  {
    key: 'balanced-p208-a16-e6',
    label: 'balanced P208 / A16 / E6',
    phaseResolution: 208,
    elevationsDegrees: [1, 5, 15, 35, 60, 85],
  },
  {
    key: 'balanced-p207-a16-e6-plus-pole',
    label: 'balanced P207 / A16 / E6 + shared vertical pole',
    phaseResolution: 207,
    elevationsDegrees: [1, 5, 15, 35, 60, 85, 90],
    collapsedVerticalPole: true,
  },
] as const;

interface CameraSpec {
  key: string;
  elevationDegrees: number;
  azimuthDegrees: number;
  heightAboveTopMetres: number;
  horizontalFovDegrees: number;
  verticalFovDegrees: number;
}

const CAMERA_SWEEPS: readonly CameraSpec[] = [
  { key: 'top-down-90deg', elevationDegrees: 90, azimuthDegrees: 17, heightAboveTopMetres: 4, horizontalFovDegrees: 18, verticalFovDegrees: 13.5 },
  { key: 'oblique-18deg', elevationDegrees: 18, azimuthDegrees: 73, heightAboveTopMetres: 0.5, horizontalFovDegrees: 8, verticalFovDegrees: 5 },
  { key: 'low-oblique-10deg', elevationDegrees: 10, azimuthDegrees: 137, heightAboveTopMetres: 0.4, horizontalFovDegrees: 5, verticalFovDegrees: 3 },
  { key: 'grazing-5deg', elevationDegrees: 5, azimuthDegrees: 211, heightAboveTopMetres: 0.3, horizontalFovDegrees: 3, verticalFovDegrees: 1.8 },
  { key: 'near-horizontal-1deg', elevationDegrees: 1, azimuthDegrees: 293, heightAboveTopMetres: 0.15, horizontalFovDegrees: 1.2, verticalFovDegrees: 0.6 },
] as const;

interface Args { [key: string]: string | boolean }

interface SourceAttributes {
  view: DataView;
  vertexOffset: number;
}

interface SurfaceSample {
  triangleId: number;
  copyX: number;
  copyZ: number;
  point: CensusVec3;
  heightDrop: number;
  color: CensusVec3;
  family: number;
}

interface CameraRay {
  spec: CameraSpec;
  x: number;
  y: number;
  origin: CensusVec3;
  direction: CensusVec3;
  qTop: CensusVec3;
  truth: SurfaceSample | null;
}

interface CanonicalSample {
  hit: boolean;
  premul: CensusVec3;
  coverage: number;
  depthMoment: number;
  heightDrop: number | null;
  family: number;
}

interface WeightedEvent {
  sample: CanonicalSample;
  weight: number;
}

interface FilteredTap {
  premul: CensusVec3;
  coverage: number;
  depthMoment: number;
  weight: number;
  events: WeightedEvent[];
}

interface Prediction {
  premul: CensusVec3;
  coverage: number;
  unassociatedColor: CensusVec3;
  family: number;
  partialCoverage: boolean;
  depthDrops: Record<DepthMethod, number | null>;
}

type DepthMethod = 'conditionalMean' | 'dominantAngular' | 'dominantCategorical' | 'oracleStrongestHit';

interface PixelResult {
  ray: CameraRay;
  prediction: Prediction;
  compositedTruth: CensusVec3;
  compositedPrediction: CensusVec3;
  compositedError: number;
  premulError: number;
  coverageError: number;
}

interface MutableMetrics {
  rays: number;
  truthHits: number;
  predictedBinaryHits: number;
  intersection: number;
  union: number;
  truePositive: number;
  trueNegative: number;
  falsePositive: number;
  falseNegative: number;
  partialCoverage: number;
  premulErrors: number[];
  compositedErrors: number[];
  coverageErrors: number[];
  ghostAlpha: number[];
  lostAlpha: number[];
  panicleTruthHits: number;
  panicleRetained: number;
  depth: Record<DepthMethod, { predictedHits: number; agreements: number; truePositive: number; positionErrors: number[] }>;
  connectedErrorComponents: number[];
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

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

function positiveModulo(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus;
}

function wrapped(value: number): number {
  return value - Math.floor(value);
}

function normalized(value: CensusVec3): CensusVec3 {
  const length = Math.hypot(value[0], value[1], value[2]);
  if (!(length > 0)) throw new Error('cannot normalize a zero vector');
  return [value[0] / length, value[1] / length, value[2] / length];
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

function vertexColor(attributes: SourceAttributes, vertex: number): CensusVec3 {
  const offset = attributes.vertexOffset + vertex * 16;
  return [
    attributes.view.getUint16(offset + 6, true) / 65535,
    attributes.view.getUint16(offset + 8, true) / 65535,
    attributes.view.getUint16(offset + 10, true) / 65535,
  ];
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
  const d00 = e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2];
  const d01 = e0[0] * e1[0] + e0[1] * e1[1] + e0[2] * e1[2];
  const d11 = e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2];
  const d20 = q[0] * e0[0] + q[1] * e0[1] + q[2] * e0[2];
  const d21 = q[0] * e1[0] + q[1] * e1[1] + q[2] * e1[2];
  const denominator = d00 * d11 - d01 * d01;
  const u = denominator !== 0 ? (d11 * d20 - d01 * d21) / denominator : 0;
  const v = denominator !== 0 ? (d00 * d21 - d01 * d20) / denominator : 0;
  const weights = [1 - u - v, u, v] as const;
  const colors = vertices.map((vertex) => vertexColor(attributes, vertex));
  const color: CensusVec3 = [0, 1, 2].map((channel) =>
    clamp(
      weights[0] * colors[0]![channel]! + weights[1] * colors[1]![channel]! + weights[2] * colors[2]![channel]!,
      0,
      1,
    ),
  ) as unknown as CensusVec3;
  return {
    triangleId: hit.triangleId,
    copyX: hit.copyX,
    copyZ: hit.copyZ,
    point,
    heightDrop: geometry.topH - point[1],
    color,
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

function unitDirection(azimuthIndex: number, elevationDegrees: number): CensusVec3 {
  if (elevationDegrees >= 90 - 1e-12) return [0, -1, 0];
  const azimuth = azimuthIndex * TAU / AZIMUTH_COUNT;
  const elevation = elevationDegrees * DEG;
  const horizontal = Math.cos(elevation);
  return [horizontal * Math.cos(azimuth), -Math.sin(elevation), horizontal * Math.sin(azimuth)];
}

function cameraDirection(spec: CameraSpec, pixelX: number, pixelY: number): CensusVec3 {
  const azimuth = spec.azimuthDegrees * DEG;
  const elevation = spec.elevationDegrees * DEG;
  const forward: CensusVec3 = [Math.cos(elevation) * Math.cos(azimuth), -Math.sin(elevation), Math.cos(elevation) * Math.sin(azimuth)];
  const right: CensusVec3 = [-Math.sin(azimuth), 0, Math.cos(azimuth)];
  const up: CensusVec3 = [Math.sin(elevation) * Math.cos(azimuth), Math.cos(elevation), Math.sin(elevation) * Math.sin(azimuth)];
  const x = ((pixelX + 0.5) / CAMERA_WIDTH * 2 - 1) * Math.tan(spec.horizontalFovDegrees * DEG * 0.5);
  const y = (1 - (pixelY + 0.5) / CAMERA_HEIGHT * 2) * Math.tan(spec.verticalFovDegrees * DEG * 0.5);
  return normalized([
    forward[0] + right[0] * x + up[0] * y,
    forward[1] + right[1] * x + up[1] * y,
    forward[2] + right[2] * x + up[2] * y,
  ]);
}

function makeCameraRays(
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  attributes: SourceAttributes,
  spec: CameraSpec,
  lateralShiftMetres = 0,
): CameraRay[] {
  const azimuth = spec.azimuthDegrees * DEG;
  const right: CensusVec3 = [-Math.sin(azimuth), 0, Math.cos(azimuth)];
  const origin: CensusVec3 = [
    geometry.tileOriginX + geometry.tileSizeX * 0.38196601125 + right[0] * lateralShiftMetres,
    geometry.topH + spec.heightAboveTopMetres,
    geometry.tileOriginZ + geometry.tileSizeZ * 0.61803398875 + right[2] * lateralShiftMetres,
  ];
  const rays: CameraRay[] = [];
  for (let y = 0; y < CAMERA_HEIGHT; y++) {
    for (let x = 0; x < CAMERA_WIDTH; x++) {
      const direction = cameraDirection(spec, x, y);
      if (!(direction[1] < -1e-8)) throw new Error(`${spec.key} generated a non-descending ray`);
      const tTop = (geometry.topH - origin[1]) / direction[1];
      const qTop: CensusVec3 = [
        origin[0] + direction[0] * tTop,
        geometry.topH,
        origin[2] + direction[2] * tTop,
      ];
      rays.push({ spec, x, y, origin, direction, qTop, truth: traceSurface(geometry, bvh, attributes, qTop, direction) });
    }
  }
  return rays;
}

function canonicalSample(
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  attributes: SourceAttributes,
  allocation: Allocation,
  cache: Map<string, CanonicalSample>,
  azimuthIndexInput: number,
  elevationRow: number,
  phaseXInput: number,
  phaseZInput: number,
): CanonicalSample {
  const azimuthIndex = positiveModulo(azimuthIndexInput, AZIMUTH_COUNT);
  const storedAzimuthIndex = allocation.collapsedVerticalPole
    && allocation.elevationsDegrees[elevationRow]! >= 90 - 1e-12
    ? 0
    : azimuthIndex;
  const phaseX = positiveModulo(phaseXInput, allocation.phaseResolution);
  const phaseZ = positiveModulo(phaseZInput, allocation.phaseResolution);
  const key = `${elevationRow}:${storedAzimuthIndex}:${phaseX}:${phaseZ}`;
  const found = cache.get(key);
  if (found) return found;
  const origin: CensusVec3 = [
    geometry.tileOriginX + (phaseX + 0.5) / allocation.phaseResolution * geometry.tileSizeX,
    geometry.topH,
    geometry.tileOriginZ + (phaseZ + 0.5) / allocation.phaseResolution * geometry.tileSizeZ,
  ];
  const direction = unitDirection(storedAzimuthIndex, allocation.elevationsDegrees[elevationRow]!);
  const surface = traceSurface(geometry, bvh, attributes, origin, direction);
  const sample: CanonicalSample = surface ? {
    hit: true,
    premul: surface.color,
    coverage: 1,
    depthMoment: surface.heightDrop,
    heightDrop: surface.heightDrop,
    family: surface.family,
  } : {
    hit: false,
    premul: [0, 0, 0],
    coverage: 0,
    depthMoment: 0,
    heightDrop: null,
    family: -1,
  };
  cache.set(key, sample);
  return sample;
}

function spatialTap(
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  attributes: SourceAttributes,
  allocation: Allocation,
  cache: Map<string, CanonicalSample>,
  qTop: CensusVec3,
  azimuthIndex: number,
  elevationRow: number,
  angularWeight: number,
): FilteredTap {
  const u = wrapped((qTop[0] - geometry.tileOriginX) / geometry.tileSizeX);
  const v = wrapped((qTop[2] - geometry.tileOriginZ) / geometry.tileSizeZ);
  const texelX = u * allocation.phaseResolution - 0.5;
  const texelZ = v * allocation.phaseResolution - 0.5;
  const x0 = Math.floor(texelX);
  const z0 = Math.floor(texelZ);
  const fx = texelX - x0;
  const fz = texelZ - z0;
  const corners = [
    { x: x0, z: z0, weight: (1 - fx) * (1 - fz) },
    { x: x0 + 1, z: z0, weight: fx * (1 - fz) },
    { x: x0, z: z0 + 1, weight: (1 - fx) * fz },
    { x: x0 + 1, z: z0 + 1, weight: fx * fz },
  ] as const;
  const premul: [number, number, number] = [0, 0, 0];
  let coverage = 0;
  let depthMoment = 0;
  const events: WeightedEvent[] = [];
  for (const corner of corners) {
    const sample = canonicalSample(
      geometry, bvh, attributes, allocation, cache,
      azimuthIndex, elevationRow, corner.x, corner.z,
    );
    coverage += corner.weight * sample.coverage;
    depthMoment += corner.weight * sample.depthMoment;
    for (let channel = 0; channel < 3; channel++) premul[channel] += corner.weight * sample.premul[channel]!;
    events.push({ sample, weight: angularWeight * corner.weight });
  }
  return { premul, coverage, depthMoment, weight: angularWeight, events };
}

function elevationBracket(allocation: Allocation, elevationDegrees: number): readonly [number, number, number] {
  const rows = allocation.elevationsDegrees;
  if (elevationDegrees <= rows[0]!) return [0, 0, 0];
  if (elevationDegrees >= rows.at(-1)!) return [rows.length - 1, rows.length - 1, 0];
  for (let row = 0; row + 1 < rows.length; row++) {
    if (elevationDegrees <= rows[row + 1]!) {
      const mix = (elevationDegrees - rows[row]!) / (rows[row + 1]! - rows[row]!);
      return [row, row + 1, mix];
    }
  }
  throw new Error('unreachable elevation bracket');
}

function predict(
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  attributes: SourceAttributes,
  allocation: Allocation,
  cache: Map<string, CanonicalSample>,
  ray: CameraRay,
): Prediction {
  const azimuth = positiveModulo(Math.atan2(ray.direction[2], ray.direction[0]), TAU);
  const azimuthCoordinate = azimuth / TAU * AZIMUTH_COUNT;
  const azimuth0 = Math.floor(azimuthCoordinate);
  const azimuthMix = azimuthCoordinate - azimuth0;
  const elevationDegrees = Math.asin(clamp(-ray.direction[1], -1, 1)) / DEG;
  const [elevation0, elevation1, elevationMix] = elevationBracket(allocation, elevationDegrees);
  const angular = [
    { azimuth: azimuth0, elevation: elevation0, weight: (1 - azimuthMix) * (1 - elevationMix) },
    { azimuth: azimuth0 + 1, elevation: elevation0, weight: azimuthMix * (1 - elevationMix) },
    { azimuth: azimuth0, elevation: elevation1, weight: (1 - azimuthMix) * elevationMix },
    { azimuth: azimuth0 + 1, elevation: elevation1, weight: azimuthMix * elevationMix },
  ] as const;
  const taps = angular.map((node) => spatialTap(
    geometry, bvh, attributes, allocation, cache, ray.qTop,
    node.azimuth, node.elevation, node.weight,
  ));
  const premul: [number, number, number] = [0, 0, 0];
  let coverage = 0;
  let depthMoment = 0;
  for (const tap of taps) {
    coverage += tap.weight * tap.coverage;
    depthMoment += tap.weight * tap.depthMoment;
    for (let channel = 0; channel < 3; channel++) premul[channel] += tap.weight * tap.premul[channel]!;
  }
  coverage = clamp(coverage, 0, 1);
  const unassociatedColor: CensusVec3 = coverage > 1e-8
    ? [premul[0] / coverage, premul[1] / coverage, premul[2] / coverage]
    : [0, 0, 0];
  const dominantAngular = taps.reduce((best, tap) =>
    tap.weight * tap.coverage > best.weight * best.coverage ? tap : best,
  );
  const events = taps.flatMap((tap) => tap.events);
  const dominantCategorical = events.reduce((best, event) => event.weight > best.weight ? event : best);
  const hitEvents = events.filter((event) => event.sample.hit);
  const oracleStrongestHit = hitEvents.length > 0
    ? hitEvents.reduce((best, event) => event.weight > best.weight ? event : best)
    : null;
  return {
    premul,
    coverage,
    unassociatedColor,
    family: coverage >= COVERAGE_THRESHOLD ? nearestFamily(unassociatedColor) : -1,
    partialCoverage: coverage > 0.02 && coverage < 0.98,
    depthDrops: {
      conditionalMean: coverage >= COVERAGE_THRESHOLD ? depthMoment / Math.max(coverage, 1e-8) : null,
      dominantAngular: dominantAngular.coverage >= COVERAGE_THRESHOLD
        ? dominantAngular.depthMoment / Math.max(dominantAngular.coverage, 1e-8)
        : null,
      dominantCategorical: dominantCategorical.sample.heightDrop,
      oracleStrongestHit: oracleStrongestHit?.sample.heightDrop ?? null,
    },
  };
}

function composite(premul: CensusVec3, coverage: number): CensusVec3 {
  return [
    premul[0] + BACKGROUND[0] * (1 - coverage),
    premul[1] + BACKGROUND[1] * (1 - coverage),
    premul[2] + BACKGROUND[2] * (1 - coverage),
  ];
}

function maximumChannelError(left: CensusVec3, right: CensusVec3): number {
  return Math.max(Math.abs(left[0] - right[0]), Math.abs(left[1] - right[1]), Math.abs(left[2] - right[2]));
}

function evaluatePixel(
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  attributes: SourceAttributes,
  allocation: Allocation,
  cache: Map<string, CanonicalSample>,
  ray: CameraRay,
): PixelResult {
  const prediction = predict(geometry, bvh, attributes, allocation, cache, ray);
  const truthPremul: CensusVec3 = ray.truth?.color ?? [0, 0, 0];
  const truthCoverage = ray.truth ? 1 : 0;
  const compositedTruth = composite(truthPremul, truthCoverage);
  const compositedPrediction = composite(prediction.premul, prediction.coverage);
  return {
    ray,
    prediction,
    compositedTruth,
    compositedPrediction,
    compositedError: maximumChannelError(compositedTruth, compositedPrediction),
    premulError: maximumChannelError(truthPremul, prediction.premul),
    coverageError: Math.abs(truthCoverage - prediction.coverage),
  };
}

function emptyMetrics(): MutableMetrics {
  const makeDepth = (): { predictedHits: number; agreements: number; truePositive: number; positionErrors: number[] } => ({
    predictedHits: 0, agreements: 0, truePositive: 0, positionErrors: [],
  });
  return {
    rays: 0,
    truthHits: 0,
    predictedBinaryHits: 0,
    intersection: 0,
    union: 0,
    truePositive: 0,
    trueNegative: 0,
    falsePositive: 0,
    falseNegative: 0,
    partialCoverage: 0,
    premulErrors: [],
    compositedErrors: [],
    coverageErrors: [],
    ghostAlpha: [],
    lostAlpha: [],
    panicleTruthHits: 0,
    panicleRetained: 0,
    depth: {
      conditionalMean: makeDepth(),
      dominantAngular: makeDepth(),
      dominantCategorical: makeDepth(),
      oracleStrongestHit: makeDepth(),
    },
    connectedErrorComponents: [],
  };
}

function addPixel(metrics: MutableMetrics, pixel: PixelResult): void {
  const truthHit = pixel.ray.truth !== null;
  const predictedHit = pixel.prediction.coverage >= COVERAGE_THRESHOLD;
  metrics.rays++;
  if (truthHit) metrics.truthHits++;
  if (predictedHit) metrics.predictedBinaryHits++;
  if (truthHit && predictedHit) {
    metrics.intersection++;
    metrics.truePositive++;
  } else if (!truthHit && !predictedHit) metrics.trueNegative++;
  else if (truthHit) metrics.falseNegative++;
  else metrics.falsePositive++;
  if (truthHit || predictedHit) metrics.union++;
  if (pixel.prediction.partialCoverage) metrics.partialCoverage++;
  metrics.premulErrors.push(pixel.premulError);
  metrics.compositedErrors.push(pixel.compositedError);
  metrics.coverageErrors.push(pixel.coverageError);
  if (!truthHit) metrics.ghostAlpha.push(pixel.prediction.coverage);
  else metrics.lostAlpha.push(1 - pixel.prediction.coverage);
  if (pixel.ray.truth && PANICLE_FAMILIES.has(pixel.ray.truth.family)) {
    metrics.panicleTruthHits++;
    if (PANICLE_FAMILIES.has(pixel.prediction.family)) metrics.panicleRetained++;
  }
  for (const method of Object.keys(pixel.prediction.depthDrops) as DepthMethod[]) {
    const drop = pixel.prediction.depthDrops[method];
    const depth = metrics.depth[method];
    if (drop !== null) depth.predictedHits++;
    if ((drop !== null) === truthHit) depth.agreements++;
    if (drop !== null && pixel.ray.truth) {
      depth.truePositive++;
      depth.positionErrors.push(Math.abs(drop - pixel.ray.truth.heightDrop) / -pixel.ray.direction[1]);
    }
  }
}

function addConnectedErrorComponents(metrics: MutableMetrics, pixels: readonly PixelResult[]): void {
  const visited = new Uint8Array(pixels.length);
  for (let seed = 0; seed < pixels.length; seed++) {
    if (visited[seed] || pixels[seed]!.compositedError < 0.2) continue;
    const stack = [seed];
    visited[seed] = 1;
    let count = 0;
    while (stack.length > 0) {
      const index = stack.pop()!;
      count++;
      const x = index % CAMERA_WIDTH;
      const y = Math.floor(index / CAMERA_WIDTH);
      for (const neighbor of [
        x > 0 ? index - 1 : -1,
        x + 1 < CAMERA_WIDTH ? index + 1 : -1,
        y > 0 ? index - CAMERA_WIDTH : -1,
        y + 1 < CAMERA_HEIGHT ? index + CAMERA_WIDTH : -1,
      ]) {
        if (neighbor < 0 || visited[neighbor] || pixels[neighbor]!.compositedError < 0.2) continue;
        visited[neighbor] = 1;
        stack.push(neighbor);
      }
    }
    metrics.connectedErrorComponents.push(count);
  }
}

function summarize(metrics: MutableMetrics): Record<string, unknown> {
  const panicle = Math.max(1, metrics.panicleTruthHits);
  return {
    rays: metrics.rays,
    truthHits: metrics.truthHits,
    predictedBinaryHits: metrics.predictedBinaryHits,
    silhouette: {
      threshold: COVERAGE_THRESHOLD,
      intersectionOverUnion: metrics.intersection / Math.max(1, metrics.union),
      agreement: (metrics.truePositive + metrics.trueNegative) / Math.max(1, metrics.rays),
      precision: metrics.truePositive / Math.max(1, metrics.truePositive + metrics.falsePositive),
      recall: metrics.truePositive / Math.max(1, metrics.truePositive + metrics.falseNegative),
    },
    partialCoverageFraction: metrics.partialCoverage / Math.max(1, metrics.rays),
    premultipliedRgbMaxChannelError: quantiles(metrics.premulErrors),
    compositedRgbMaxChannelError: quantiles(metrics.compositedErrors),
    coverageAbsoluteError: quantiles(metrics.coverageErrors),
    ghostAlphaOnTruthMisses: quantiles(metrics.ghostAlpha),
    lostAlphaOnTruthHits: quantiles(metrics.lostAlpha),
    panicle: {
      truthHits: metrics.panicleTruthHits,
      retainedAsPanicleFraction: metrics.panicleRetained / panicle,
    },
    connectedScreenErrorComponentsAtPoint2: {
      count: metrics.connectedErrorComponents.length,
      pixels: quantiles(metrics.connectedErrorComponents),
      maximumFrameFraction: (Math.max(0, ...metrics.connectedErrorComponents)) / (CAMERA_WIDTH * CAMERA_HEIGHT),
    },
    depthElection: Object.fromEntries((Object.keys(metrics.depth) as DepthMethod[]).map((method) => {
      const depth = metrics.depth[method];
      return [method, {
        hitMissAgreement: depth.agreements / Math.max(1, metrics.rays),
        predictedHits: depth.predictedHits,
        truePositive: depth.truePositive,
        sameLiveRayPositionErrorMetres: quantiles(depth.positionErrors),
      }];
    })),
  };
}

function mergeMetrics(target: MutableMetrics, source: MutableMetrics): void {
  for (const key of [
    'rays', 'truthHits', 'predictedBinaryHits', 'intersection', 'union', 'truePositive', 'trueNegative',
    'falsePositive', 'falseNegative', 'partialCoverage', 'panicleTruthHits', 'panicleRetained',
  ] as const) target[key] += source[key];
  target.premulErrors.push(...source.premulErrors);
  target.compositedErrors.push(...source.compositedErrors);
  target.coverageErrors.push(...source.coverageErrors);
  target.ghostAlpha.push(...source.ghostAlpha);
  target.lostAlpha.push(...source.lostAlpha);
  target.connectedErrorComponents.push(...source.connectedErrorComponents);
  for (const method of Object.keys(target.depth) as DepthMethod[]) {
    target.depth[method].predictedHits += source.depth[method].predictedHits;
    target.depth[method].agreements += source.depth[method].agreements;
    target.depth[method].truePositive += source.depth[method].truePositive;
    target.depth[method].positionErrors.push(...source.depth[method].positionErrors);
  }
}

function temporalSummary(frames: readonly PixelResult[][]): Record<string, unknown> {
  const truthRgbDelta: number[] = [];
  const predictedRgbDelta: number[] = [];
  const excessRgbDelta: number[] = [];
  const coverageDelta: number[] = [];
  let pairs = 0;
  let stableTruthVisibilityPairs = 0;
  let unforcedPredictedVisibilityChanges = 0;
  let stableTruthFamilyPairs = 0;
  let unforcedPredictedFamilyChanges = 0;
  for (let frame = 1; frame < frames.length; frame++) {
    const priorFrame = frames[frame - 1]!;
    const currentFrame = frames[frame]!;
    if (priorFrame.length !== currentFrame.length) throw new Error('camera path frames differ in size');
    for (let pixel = 0; pixel < currentFrame.length; pixel++) {
      const prior = priorFrame[pixel]!;
      const current = currentFrame[pixel]!;
      const truthDelta = maximumChannelError(prior.compositedTruth, current.compositedTruth);
      const predictedDelta = maximumChannelError(prior.compositedPrediction, current.compositedPrediction);
      truthRgbDelta.push(truthDelta);
      predictedRgbDelta.push(predictedDelta);
      excessRgbDelta.push(Math.max(0, predictedDelta - truthDelta));
      coverageDelta.push(Math.abs(prior.prediction.coverage - current.prediction.coverage));
      pairs++;
      const priorTruthHit = prior.ray.truth !== null;
      const currentTruthHit = current.ray.truth !== null;
      if (priorTruthHit === currentTruthHit) {
        stableTruthVisibilityPairs++;
        const priorPredictedHit = prior.prediction.coverage >= COVERAGE_THRESHOLD;
        const currentPredictedHit = current.prediction.coverage >= COVERAGE_THRESHOLD;
        if (priorPredictedHit !== currentPredictedHit) unforcedPredictedVisibilityChanges++;
      }
      if (prior.ray.truth?.family === current.ray.truth?.family && prior.ray.truth && current.ray.truth) {
        stableTruthFamilyPairs++;
        if (prior.prediction.family !== current.prediction.family) unforcedPredictedFamilyChanges++;
      }
    }
  }
  return {
    pairs,
    truthCompositedRgbDelta: quantiles(truthRgbDelta),
    predictedCompositedRgbDelta: quantiles(predictedRgbDelta),
    predictedExcessOverTruthRgbDelta: quantiles(excessRgbDelta),
    predictionCoverageDelta: quantiles(coverageDelta),
    stableTruthVisibilityPairs,
    unforcedPredictedVisibilityChanges,
    unforcedPredictedVisibilityChangeFraction:
      unforcedPredictedVisibilityChanges / Math.max(1, stableTruthVisibilityPairs),
    stableTruthFamilyPairs,
    unforcedPredictedFamilyChanges,
    unforcedPredictedFamilyChangeFraction:
      unforcedPredictedFamilyChanges / Math.max(1, stableTruthFamilyPairs),
  };
}

function rgb8(color: CensusVec3): readonly [number, number, number] {
  return color.map((value) => clamp(Math.round(Math.pow(clamp(value, 0, 1), 1 / 2.2) * 255), 0, 255)) as unknown as readonly [number, number, number];
}

function diagnosticRow(pixels: readonly PixelResult[]): Uint8Array {
  const panelCount = 4;
  const width = CAMERA_WIDTH * panelCount;
  const output = new Uint8Array(width * CAMERA_HEIGHT * 3);
  for (const pixel of pixels) {
    const truth = rgb8(pixel.compositedTruth);
    const predicted = rgb8(pixel.compositedPrediction);
    const errorValue = pixel.compositedError;
    const error: readonly [number, number, number] = errorValue < 0.05 ? [15, 70, 25]
      : errorValue < 0.12 ? [195, 190, 30]
        : errorValue < 0.25 ? [235, 105, 10] : [225, 15, 45];
    const alphaValue = Math.round(clamp(pixel.prediction.coverage, 0, 1) * 255);
    const alpha = [alphaValue, alphaValue, alphaValue] as const;
    for (const [panel, color] of [[0, truth], [1, predicted], [2, error], [3, alpha]] as const) {
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
  const rowWidth = CAMERA_WIDTH * 4;
  const rowHeight = CAMERA_HEIGHT;
  const output = new Uint8Array(rowWidth * rowHeight * rows.length * 3);
  rows.forEach((row, rowIndex) => output.set(row, rowIndex * rowWidth * rowHeight * 3));
  await sharp(output, { raw: { width: rowWidth, height: rowHeight * rows.length, channels: 3 } })
    .resize(rowWidth * QA_SCALE, rowHeight * rows.length * QA_SCALE, { kernel: 'nearest' })
    .png().toFile(path);
}

const args = parseArgs(process.argv.slice(2));
const sourcePath = resolve(WORKSPACE, typeof args.source === 'string' ? args.source : DEFAULT_SOURCE);
const sourceBytes = readFileSync(sourcePath);
const sourceSha256 = sha256(sourceBytes);
if (sourceSha256 !== EXPECTED_SOURCE_SHA256) throw new Error(`unexpected source SHA-256 ${sourceSha256}`);
const scriptPath = fileURLToPath(import.meta.url);
const scriptSha256 = sha256(readFileSync(scriptPath));
const configuration = {
  schema: 'laas-direct-premultiplied-radiance-field-gate/v1',
  sourceSha256,
  residentCapBytes: RESIDENT_CAP_BYTES,
  chargedBytesPerCell: CHARGED_BYTES_PER_CELL,
  gutter: GUTTER,
  azimuthCount: AZIMUTH_COUNT,
  allocations: ALLOCATIONS,
  camera: { width: CAMERA_WIDTH, height: CAMERA_HEIGHT, sweeps: CAMERA_SWEEPS },
  cameraPath: { offsetsMetres: CAMERA_PATH_OFFSETS_METRES, pixelStride: CAMERA_PATH_PIXEL_STRIDE },
  sampling: {
    lineAddress: 'exact live ray intersection with fixed profile top plane plus exact direction',
    spatial: 'periodic hardware-equivalent bilinear interpolation of premultiplied authored RGB and binary coverage',
    angular: 'bilinear interpolation across two azimuth and two elevation nodes',
    colourReads: 4,
    forbiddenColourChannels: ['depth', 'owner', 'normal'],
  },
  depthMethodsAreOfflineDiagnosticsOnly: [
    'conditional mean vertical drop',
    'dominant angular filtered tap',
    'dominant categorical phase/direction corner',
    'oracle strongest hit among the fixed 16 offline point contributors',
  ],
  scriptSha256,
};
const recipeSha256 = sha256(canonicalJson(configuration));
const outputRoot = resolve(
  WORKSPACE,
  typeof args.output === 'string'
    ? args.output
    : `data/work/groundcover-direct-radiance-field/${sourceSha256.slice(0, 16)}/${recipeSha256.slice(0, 16)}`,
);
mkdirSync(resolve(outputRoot, 'qa'), { recursive: true });

const geometry = decodeOwnedProfileGeometry(sourceBytes);
const attributes = sourceAttributes(sourceBytes);
const bvh = TriangleBvh.build(geometry, 8);
console.error(`[direct-radiance] BVH ${bvh.metrics.buildMilliseconds.toFixed(0)} ms`);
const started = performance.now();
const cameraRays = new Map<string, CameraRay[]>();
const cameraPathRays = new Map<string, CameraRay[][]>();
for (const spec of CAMERA_SWEEPS) {
  const base = makeCameraRays(geometry, bvh, attributes, spec);
  cameraRays.set(spec.key, base);
  const frames = CAMERA_PATH_OFFSETS_METRES.map((offset, frame) => {
    const rays = frame === 0 ? base : makeCameraRays(geometry, bvh, attributes, spec, offset);
    return rays.filter((_, index) => index % CAMERA_PATH_PIXEL_STRIDE === 0);
  });
  cameraPathRays.set(spec.key, frames);
  console.error(`[direct-radiance] truth ${spec.key}`);
}

const allocationReports: Record<string, unknown> = {};
const qaFiles: Array<{ file: string; width: number; height: number; interpretation: string }> = [];
for (const allocation of ALLOCATIONS) {
  const storedSide = allocation.phaseResolution + GUTTER * 2;
  const directions = allocation.collapsedVerticalPole
    ? AZIMUTH_COUNT * (allocation.elevationsDegrees.length - 1) + 1
    : AZIMUTH_COUNT * allocation.elevationsDegrees.length;
  const storedCells = storedSide * storedSide * directions;
  const storedBytes = storedCells * CHARGED_BYTES_PER_CELL;
  if (storedBytes > RESIDENT_CAP_BYTES) throw new Error(`${allocation.key} exceeds resident cap`);
  const cache = new Map<string, CanonicalSample>();
  const aggregate = emptyMetrics();
  const bySweep: Record<string, unknown> = {};
  const cameraPathBySweep: Record<string, unknown> = {};
  const rows: Uint8Array[] = [];
  for (const spec of CAMERA_SWEEPS) {
    const pixels = cameraRays.get(spec.key)!.map((ray) =>
      evaluatePixel(geometry, bvh, attributes, allocation, cache, ray));
    const metrics = emptyMetrics();
    for (const pixel of pixels) addPixel(metrics, pixel);
    addConnectedErrorComponents(metrics, pixels);
    mergeMetrics(aggregate, metrics);
    bySweep[spec.key] = summarize(metrics);
    rows.push(diagnosticRow(pixels));
    console.error(`[direct-radiance] ${allocation.key} ${spec.key} cache=${cache.size}`);
    const pathFrames = cameraPathRays.get(spec.key)!.map((frame) => frame.map((ray) =>
      evaluatePixel(geometry, bvh, attributes, allocation, cache, ray)));
    cameraPathBySweep[spec.key] = temporalSummary(pathFrames);
  }
  const qaFile = `qa/${String(qaFiles.length + 1).padStart(3, '0')}-${allocation.key}.png`;
  await writeContactSheet(resolve(outputRoot, qaFile), rows);
  qaFiles.push({
    file: qaFile,
    width: CAMERA_WIDTH * 4 * QA_SCALE,
    height: CAMERA_HEIGHT * CAMERA_SWEEPS.length * QA_SCALE,
    interpretation: 'rows: top-down 90, oblique 18, low-oblique 10, grazing 5, near-horizontal 1 degrees; panels: exact authored-colour composite, four-read premultiplied-radiance prediction, max-channel error class, predicted coverage',
  });
  allocationReports[allocation.key] = {
    label: allocation.label,
    allocation: {
      phaseResolution: allocation.phaseResolution,
      phaseTexelMetres: [geometry.tileSizeX / allocation.phaseResolution, geometry.tileSizeZ / allocation.phaseResolution],
      azimuthCount: AZIMUTH_COUNT,
      elevationCount: allocation.elevationsDegrees.length,
      elevationsDegrees: allocation.elevationsDegrees,
      directionCount: directions,
      collapsedVerticalPole: allocation.collapsedVerticalPole ?? false,
      storedSide,
      storedCells,
      chargedBytesPerCell: CHARGED_BYTES_PER_CELL,
      storedBytes,
      capRemainingBytes: RESIDENT_CAP_BYTES - storedBytes,
      fixedSpatiallyFilteredColourReads: 4,
    },
    aggregate: summarize(aggregate),
    bySweep,
    cameraPathBySweep,
    generatedCanonicalPointSamples: cache.size,
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
  traceMilliseconds: performance.now() - started,
  allocationReports,
  exactProspectiveColourCost: {
    textureReads: 4,
    readMeaning: 'four spatially bilinear premultiplied RGBA samples at the surrounding two azimuth by two elevation nodes',
    approximateScalarLerpFma: 28,
    addressWork: 'one top-plane intersection; atan2/asin direction coordinates; fixed azimuth/elevation bracket; four atlas addresses',
    bindings: 'one filterable colour/coverage atlas; the 12-byte charge conservatively retains the full current acceptance cell budget',
    absent: ['runtime loop', 'march', 'candidate traversal', 'owner read', 'normal read', 'depth-to-colour relift', 'extra pass', 'species multiplier'],
  },
  interpretationBoundary: {
    colour: 'the predictor is a literal sampled exterior radiance function; it cannot geometrically stretch a decoded event because it never reconstructs or relocates one, but coarse angular interpolation may ghost, blur, or select the wrong appearance',
    depth: 'all reported depth choices are separated diagnostics, not part of the colour predictor and not accepted runtime solutions',
    cameraInside: 'unsolved: a top-entry exterior ray field has no pointed-line origin/successor coordinate',
    exactHorizontal: 'undefined: the top-plane intersection is at infinity when direction.y is exactly zero; the finite gate ends at 1 degree',
    overlappingCover: 'a viable atlas would bake one offline-unioned marked community; one runtime field per species remains invalid',
  },
};

const metricsSerialized = `${JSON.stringify(report, null, 2)}\n`;
writeFileSync(resolve(outputRoot, 'metrics.json'), metricsSerialized);
const files = [
  { file: 'metrics.json', bytes: Buffer.byteLength(metricsSerialized), sha256: sha256(metricsSerialized), interpretation: 'complete numeric report' },
  ...qaFiles.map((entry) => {
    const bytes = readFileSync(resolve(outputRoot, entry.file));
    return { ...entry, bytes: bytes.byteLength, sha256: sha256(bytes) };
  }),
];
const index = {
  schema: 'laas-direct-premultiplied-radiance-field-gate-index/v1',
  sourceSha256,
  recipeSha256,
  script: { file: relative(WORKSPACE, scriptPath), sha256: scriptSha256 },
  files,
};
const indexSerialized = `${JSON.stringify(index, null, 2)}\n`;
writeFileSync(resolve(outputRoot, 'index.json'), indexSerialized);
console.error(`[direct-radiance] wrote ${outputRoot}`);
console.error(`[direct-radiance] metrics sha256 ${sha256(metricsSerialized)}`);
process.stdout.write(`${outputRoot}\n`);
