/**
 * Candidate-K held-out angular finite-element oracle.
 *
 * This is offline truth/bake work only. It raycasts the real infinitely
 * repeated GCRP triangle soup, filters positive premultiplied RGBA measures,
 * and separates unlimited-precision angular FE error from RGBA4444 packing
 * error. It deliberately contains no runtime or shader path.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
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
const DEFAULT_SOURCE = resolve(WORKSPACE, 'src/assets/groundcover/calamagrostis-canescens.gcrp');
const THRESHOLD_DOC = resolve(
  WORKSPACE,
  'docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-CANDIDATE-K-HELDOUT-THRESHOLDS-FROZEN.md',
);
const VERSION = 'candidate-k-heldout-fe-v1';
const TAU = Math.PI * 2;
const DEG = Math.PI / 180;
const AZIMUTH_COUNT = 16;
const RINGS_DEGREES = [0.25, 2, 5, 10, 18, 30, 55, 75] as const;
const POLE_DEGREES = 90;
const HELDOUT_ELEVATIONS = [0.05, 0.1, 0.2, 0.25, 0.5, 1, 3.5, 7.5, 14, 24, 45, 65, 82, 89] as const;
const HELDOUT_SECTORS = [0, 3, 7, 11] as const;
/** Accepted source-kernel radii on the canonical 256-square page. */
const SOURCE_FILTER_RADII = [4, 16] as const;
const MIDDLE_REFERENCE_Y = 0.49255;
const LIMITS = {
  coverageP95: 0.08,
  coverageP99: 0.20,
  rgbP95: 0.06,
  rgbP99: 0.15,
  connected: 0.01,
} as const;

interface Args { [key: string]: string | boolean }
export interface SourceAttributes { view: DataView; vertexOffset: number }
export interface FieldPage { rgba: Float32Array }
export interface DirectionSpec {
  key: string;
  elevationDegrees: number;
  azimuthDegrees: number;
  kind: 'node' | 'heldout';
}
interface QuantileSummary {
  count: number;
  p50: number;
  p95: number;
  p99: number;
  maximum: number;
}
interface PageScore {
  green: boolean;
  coverage: QuantileSummary;
  premulRgb: QuantileSummary;
  largestConnectedFraction: number;
  checks: Record<string, boolean>;
}
interface PageEvaluation {
  spec: DirectionSpec;
  radius: number;
  unlimited: PageScore;
  rgba4444: PageScore;
  packingIncrement: {
    coverage: QuantileSummary;
    premulRgb: QuantileSummary;
  };
}
interface QaCandidate {
  score: number;
  title: string;
  truth: Float32Array;
  unlimited: Float32Array;
  packed: Float32Array;
  coverageError: Float32Array;
  rgbError: Float32Array;
}

function parseArgs(argv: readonly string[]): Args {
  const result: Args = {};
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index]!;
    if (!argument.startsWith('--')) throw new Error(`unexpected positional argument ${argument}`);
    const next = argv[index + 1];
    if (next !== undefined && !next.startsWith('--')) {
      result[argument.slice(2)] = next;
      index++;
    } else result[argument.slice(2)] = true;
  }
  return result;
}

export function sha256(value: Uint8Array | string): string {
  return createHash('sha256').update(value).digest('hex');
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  const object = value as Record<string, unknown>;
  return `{${Object.keys(object).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(object[key])}`).join(',')}}`;
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

function positiveModulo(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus;
}

export function sourceAttributes(bytes: Uint8Array): SourceAttributes {
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

function hitColor(
  geometry: DecodedOwnedProfileGeometry,
  attributes: SourceAttributes,
  origin: CensusVec3,
  direction: CensusVec3,
  hit: OriginAwareTruthHit,
): CensusVec3 {
  const point: CensusVec3 = [
    origin[0] + direction[0] * hit.t - hit.copyX * geometry.tileSizeX,
    origin[1] + direction[1] * hit.t,
    origin[2] + direction[2] * hit.t - hit.copyZ * geometry.tileSizeZ,
  ];
  const tri = hit.triangleId * 3;
  const ids = [geometry.triangles[tri]!, geometry.triangles[tri + 1]!, geometry.triangles[tri + 2]!] as const;
  const position = (id: number): CensusVec3 => {
    const base = id * 3;
    return [geometry.positions[base]!, geometry.positions[base + 1]!, geometry.positions[base + 2]!];
  };
  const a = position(ids[0]);
  const b = position(ids[1]);
  const c = position(ids[2]);
  const e0: CensusVec3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const e1: CensusVec3 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const q: CensusVec3 = [point[0] - a[0], point[1] - a[1], point[2] - a[2]];
  const d00 = e0[0] ** 2 + e0[1] ** 2 + e0[2] ** 2;
  const d01 = e0[0] * e1[0] + e0[1] * e1[1] + e0[2] * e1[2];
  const d11 = e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2;
  const d20 = q[0] * e0[0] + q[1] * e0[1] + q[2] * e0[2];
  const d21 = q[0] * e1[0] + q[1] * e1[1] + q[2] * e1[2];
  const denominator = d00 * d11 - d01 * d01;
  const u = denominator !== 0 ? (d11 * d20 - d01 * d21) / denominator : 0;
  const v = denominator !== 0 ? (d00 * d21 - d01 * d20) / denominator : 0;
  const weights = [1 - u - v, u, v] as const;
  const colors = ids.map((id) => vertexColor(attributes, id));
  return [0, 1, 2].map((channel) => clamp(
    weights[0] * colors[0]![channel]!
      + weights[1] * colors[1]![channel]!
      + weights[2] * colors[2]![channel]!,
    0,
    1,
  )) as CensusVec3;
}

function direction(elevationDegrees: number, azimuthDegrees: number): CensusVec3 {
  if (elevationDegrees >= 90 - 1e-12) return [0, -1, 0];
  const elevation = elevationDegrees * DEG;
  const azimuth = azimuthDegrees * DEG;
  const horizontal = Math.cos(elevation);
  return [horizontal * Math.cos(azimuth), -Math.sin(elevation), horizontal * Math.sin(azimuth)];
}

export function renderPointPage(
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  attributes: SourceAttributes,
  resolution: number,
  referenceY: number,
  spec: DirectionSpec,
  phaseOffsetX = 0,
  phaseOffsetZ = 0,
): FieldPage {
  const d = direction(spec.elevationDegrees, spec.azimuthDegrees);
  const rgba = new Float32Array(resolution * resolution * 4);
  const topParameter = (geometry.topH - referenceY) / d[1];
  for (let z = 0; z < resolution; z++) {
    const qz = geometry.tileOriginZ + (z + 0.5 + phaseOffsetZ) / resolution * geometry.tileSizeZ;
    for (let x = 0; x < resolution; x++) {
      const qx = geometry.tileOriginX + (x + 0.5 + phaseOffsetX) / resolution * geometry.tileSizeX;
      const origin: CensusVec3 = [qx + topParameter * d[0], geometry.topH, qz + topParameter * d[2]];
      const horizon = (origin[1] - geometry.bounds.min[1] + 1e-8) / -d[1];
      const hit = periodicNearestSuccessor(geometry, bvh, origin, d, horizon, 0);
      if (!hit) continue;
      const color = hitColor(geometry, attributes, origin, d, hit);
      const offset = (z * resolution + x) * 4;
      rgba[offset] = color[0];
      rgba[offset + 1] = color[1];
      rgba[offset + 2] = color[2];
      rgba[offset + 3] = 1;
    }
  }
  return { rgba };
}

export function toroidalBoxFilter(page: FieldPage, resolution: number, radius: number): FieldPage {
  const width = radius * 2 + 1;
  const horizontal = new Float64Array(page.rgba.length);
  const output = new Float32Array(page.rgba.length);
  for (let z = 0; z < resolution; z++) {
    for (let channel = 0; channel < 4; channel++) {
      let sum = 0;
      for (let dx = -radius; dx <= radius; dx++) {
        const x = positiveModulo(dx, resolution);
        sum += page.rgba[(z * resolution + x) * 4 + channel]!;
      }
      for (let x = 0; x < resolution; x++) {
        horizontal[(z * resolution + x) * 4 + channel] = sum;
        const remove = positiveModulo(x - radius, resolution);
        const add = positiveModulo(x + radius + 1, resolution);
        sum += page.rgba[(z * resolution + add) * 4 + channel]!
          - page.rgba[(z * resolution + remove) * 4 + channel]!;
      }
    }
  }
  const inverseArea = 1 / (width * width);
  for (let x = 0; x < resolution; x++) {
    for (let channel = 0; channel < 4; channel++) {
      let sum = 0;
      for (let dz = -radius; dz <= radius; dz++) {
        const z = positiveModulo(dz, resolution);
        sum += horizontal[(z * resolution + x) * 4 + channel]!;
      }
      for (let z = 0; z < resolution; z++) {
        output[(z * resolution + x) * 4 + channel] = sum * inverseArea;
        const remove = positiveModulo(z - radius, resolution);
        const add = positiveModulo(z + radius + 1, resolution);
        sum += horizontal[(add * resolution + x) * 4 + channel]!
          - horizontal[(remove * resolution + x) * 4 + channel]!;
      }
    }
  }
  return { rgba: output };
}

function nodeKey(row: number, azimuth: number): string {
  return `${row}:${positiveModulo(azimuth, AZIMUTH_COUNT)}`;
}

function poleKey(): string { return `${RINGS_DEGREES.length}:0`; }

function elevationBracket(elevationDegrees: number): readonly [number, number, number] {
  if (elevationDegrees <= RINGS_DEGREES[0]) return [0, 0, 0];
  if (elevationDegrees >= POLE_DEGREES) return [RINGS_DEGREES.length, RINGS_DEGREES.length, 0];
  for (let row = 0; row < RINGS_DEGREES.length; row++) {
    const lower = RINGS_DEGREES[row]!;
    const upper = row + 1 < RINGS_DEGREES.length ? RINGS_DEGREES[row + 1]! : POLE_DEGREES;
    if (elevationDegrees <= upper) return [row, row + 1, (elevationDegrees - lower) / (upper - lower)];
  }
  throw new Error('unreachable elevation bracket');
}

function quantize4(value: number): number { return Math.round(clamp(value, 0, 1) * 15) / 15; }

function predictPage(
  nodes: ReadonlyMap<string, FieldPage>,
  resolution: number,
  spec: DirectionSpec,
  packed: boolean,
): FieldPage {
  const azimuthCoordinate = positiveModulo(spec.azimuthDegrees, 360) / 360 * AZIMUTH_COUNT;
  const azimuth0 = Math.floor(azimuthCoordinate);
  const a = azimuthCoordinate - azimuth0;
  const [row0, row1, t] = elevationBracket(spec.elevationDegrees);
  const get = (row: number, azimuth: number): Float32Array => {
    const page = nodes.get(row === RINGS_DEGREES.length ? poleKey() : nodeKey(row, azimuth));
    if (!page) throw new Error(`missing angular node row=${row} azimuth=${azimuth}`);
    return page.rgba;
  };
  const pages = [get(row0, azimuth0), get(row0, azimuth0 + 1), get(row1, azimuth0), get(row1, azimuth0 + 1)];
  const weights = [(1 - a) * (1 - t), a * (1 - t), (1 - a) * t, a * t] as const;
  const rgba = new Float32Array(resolution * resolution * 4);
  for (let index = 0; index < rgba.length; index++) {
    let value = 0;
    for (let corner = 0; corner < 4; corner++) {
      const sample = pages[corner]![index]!;
      value += weights[corner]! * (packed ? quantize4(sample) : sample);
    }
    rgba[index] = value;
  }
  return { rgba };
}

function quantiles(values: Float32Array): QuantileSummary {
  const sorted = Array.from(values).sort((left, right) => left - right);
  const at = (fraction: number): number => sorted[Math.floor((sorted.length - 1) * fraction)]!;
  return { count: sorted.length, p50: at(0.5), p95: at(0.95), p99: at(0.99), maximum: sorted.at(-1)! };
}

function largestPeriodicComponent(mask: Uint8Array, resolution: number): number {
  const visited = new Uint8Array(mask.length);
  const stack = new Int32Array(mask.length);
  let maximum = 0;
  for (let seed = 0; seed < mask.length; seed++) {
    if (!mask[seed] || visited[seed]) continue;
    let head = 0;
    let tail = 0;
    stack[tail++] = seed;
    visited[seed] = 1;
    let count = 0;
    while (head < tail) {
      const index = stack[head++]!;
      count++;
      const x = index % resolution;
      const z = Math.floor(index / resolution);
      const neighbors = [
        z * resolution + positiveModulo(x - 1, resolution),
        z * resolution + positiveModulo(x + 1, resolution),
        positiveModulo(z - 1, resolution) * resolution + x,
        positiveModulo(z + 1, resolution) * resolution + x,
      ];
      for (const neighbor of neighbors) {
        if (!mask[neighbor] || visited[neighbor]) continue;
        visited[neighbor] = 1;
        stack[tail++] = neighbor;
      }
    }
    maximum = Math.max(maximum, count);
  }
  return maximum;
}

export function scorePage(truth: FieldPage, prediction: FieldPage, resolution: number): {
  score: PageScore;
  coverageError: Float32Array;
  rgbError: Float32Array;
} {
  const pixels = resolution * resolution;
  const coverageError = new Float32Array(pixels);
  const rgbError = new Float32Array(pixels);
  const exceedance = new Uint8Array(pixels);
  for (let pixel = 0; pixel < pixels; pixel++) {
    const offset = pixel * 4;
    coverageError[pixel] = Math.abs(prediction.rgba[offset + 3]! - truth.rgba[offset + 3]!);
    rgbError[pixel] = Math.max(
      Math.abs(prediction.rgba[offset]! - truth.rgba[offset]!),
      Math.abs(prediction.rgba[offset + 1]! - truth.rgba[offset + 1]!),
      Math.abs(prediction.rgba[offset + 2]! - truth.rgba[offset + 2]!),
    );
    exceedance[pixel] = coverageError[pixel]! > LIMITS.coverageP99 || rgbError[pixel]! > LIMITS.rgbP99 ? 1 : 0;
  }
  const coverage = quantiles(coverageError);
  const premulRgb = quantiles(rgbError);
  const largestConnectedFraction = largestPeriodicComponent(exceedance, resolution) / pixels;
  const checks = {
    coverageP95: coverage.p95 <= LIMITS.coverageP95,
    coverageP99: coverage.p99 <= LIMITS.coverageP99,
    rgbP95: premulRgb.p95 <= LIMITS.rgbP95,
    rgbP99: premulRgb.p99 <= LIMITS.rgbP99,
    connected: largestConnectedFraction < LIMITS.connected,
  };
  return {
    score: { green: Object.values(checks).every(Boolean), coverage, premulRgb, largestConnectedFraction, checks },
    coverageError,
    rgbError,
  };
}

function packingIncrement(unlimited: FieldPage, packed: FieldPage, resolution: number): {
  coverage: QuantileSummary;
  premulRgb: QuantileSummary;
} {
  const pixels = resolution * resolution;
  const coverage = new Float32Array(pixels);
  const rgb = new Float32Array(pixels);
  for (let pixel = 0; pixel < pixels; pixel++) {
    const offset = pixel * 4;
    coverage[pixel] = Math.abs(packed.rgba[offset + 3]! - unlimited.rgba[offset + 3]!);
    rgb[pixel] = Math.max(
      Math.abs(packed.rgba[offset]! - unlimited.rgba[offset]!),
      Math.abs(packed.rgba[offset + 1]! - unlimited.rgba[offset + 1]!),
      Math.abs(packed.rgba[offset + 2]! - unlimited.rgba[offset + 2]!),
    );
  }
  return { coverage: quantiles(coverage), premulRgb: quantiles(rgb) };
}

function updateWorst(list: QaCandidate[], candidate: QaCandidate, maximum = 4): void {
  list.push(candidate);
  list.sort((left, right) => right.score - left.score);
  list.length = Math.min(list.length, maximum);
}

function heat(value: number, limit: number): readonly [number, number, number] {
  const t = clamp(value / limit, 0, 1);
  return [Math.round(255 * t), Math.round(255 * Math.min(1, 2 * t) * (1 - 0.6 * t)), Math.round(40 * (1 - t))];
}

function appearancePixel(page: Float32Array, pixel: number): readonly [number, number, number] {
  const offset = pixel * 4;
  const a = page[offset + 3]!;
  return [0, 1, 2].map((channel) => Math.round(255 * clamp(page[offset + channel]! + 0.12 * (1 - a), 0, 1))) as unknown as readonly [number, number, number];
}

async function writeQa(path: string, candidate: QaCandidate, resolution: number): Promise<void> {
  const panels = 5;
  const pixels = Buffer.alloc(resolution * panels * resolution * 3);
  for (let y = 0; y < resolution; y++) {
    for (let x = 0; x < resolution; x++) {
      const pixel = y * resolution + x;
      const colors = [
        appearancePixel(candidate.truth, pixel),
        appearancePixel(candidate.unlimited, pixel),
        appearancePixel(candidate.packed, pixel),
        heat(candidate.coverageError[pixel]!, LIMITS.coverageP99),
        heat(candidate.rgbError[pixel]!, LIMITS.rgbP99),
      ];
      for (let panel = 0; panel < panels; panel++) {
        const target = (y * resolution * panels + panel * resolution + x) * 3;
        pixels[target] = colors[panel]![0];
        pixels[target + 1] = colors[panel]![1];
        pixels[target + 2] = colors[panel]![2];
      }
    }
  }
  await sharp(pixels, { raw: { width: resolution * panels, height: resolution, channels: 3 } })
    .resize({ width: resolution * panels * 3, height: resolution * 3, kernel: 'nearest' })
    .png()
    .toFile(path);
}

function heldoutDirections(): DirectionSpec[] {
  const result: DirectionSpec[] = [];
  for (const elevationDegrees of HELDOUT_ELEVATIONS) {
    for (const sector of HELDOUT_SECTORS) {
      for (const local of [0, 0.5] as const) {
        const azimuthDegrees = (sector + local) * 360 / AZIMUTH_COUNT;
        result.push({
          key: `heldout-e${elevationDegrees}-a${azimuthDegrees}`,
          elevationDegrees,
          azimuthDegrees,
          kind: 'heldout',
        });
      }
    }
  }
  return result;
}

function nodeDirections(): DirectionSpec[] {
  const result: DirectionSpec[] = [];
  for (let row = 0; row < RINGS_DEGREES.length; row++) {
    for (let azimuth = 0; azimuth < AZIMUTH_COUNT; azimuth++) {
      result.push({
        key: nodeKey(row, azimuth),
        elevationDegrees: RINGS_DEGREES[row]!,
        azimuthDegrees: azimuth * 360 / AZIMUTH_COUNT,
        kind: 'node',
      });
    }
  }
  result.push({ key: poleKey(), elevationDegrees: POLE_DEGREES, azimuthDegrees: 0, kind: 'node' });
  return result;
}

async function evaluateReference(
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  attributes: SourceAttributes,
  resolution: number,
  referenceKey: string,
  referenceY: number,
  maximumHeldout: number | null,
  qaRoot: string,
): Promise<Record<string, unknown>> {
  const nodesByRadius = new Map<number, Map<string, FieldPage>>();
  const effectiveRadius = (sourceRadius: number): number => {
    const radius = sourceRadius * resolution / 256;
    if (!Number.isInteger(radius)) {
      throw new Error(`resolution ${resolution} requires fractional box integration for sigma=${sourceRadius}`);
    }
    return radius;
  };
  for (const radius of SOURCE_FILTER_RADII) nodesByRadius.set(radius, new Map());
  const nodes = nodeDirections();
  for (let index = 0; index < nodes.length; index++) {
    const spec = nodes[index]!;
    console.log(`[candidate-k] ${referenceKey} node ${index + 1}/${nodes.length} ${spec.key}`);
    const point = renderPointPage(geometry, bvh, attributes, resolution, referenceY, spec);
    for (const radius of SOURCE_FILTER_RADII) {
      nodesByRadius.get(radius)!.set(spec.key, toroidalBoxFilter(point, resolution, effectiveRadius(radius)));
    }
  }

  const heldout = heldoutDirections().slice(0, maximumHeldout ?? Number.POSITIVE_INFINITY);
  const evaluations: PageEvaluation[] = [];
  const worst: QaCandidate[] = [];
  for (let index = 0; index < heldout.length; index++) {
    const spec = heldout[index]!;
    console.log(`[candidate-k] ${referenceKey} heldout ${index + 1}/${heldout.length} ${spec.key}`);
    const point = renderPointPage(geometry, bvh, attributes, resolution, referenceY, spec);
    for (const radius of SOURCE_FILTER_RADII) {
      const truth = toroidalBoxFilter(point, resolution, effectiveRadius(radius));
      const unlimited = predictPage(nodesByRadius.get(radius)!, resolution, spec, false);
      const packed = predictPage(nodesByRadius.get(radius)!, resolution, spec, true);
      const unlimitedScore = scorePage(truth, unlimited, resolution);
      const packedScore = scorePage(truth, packed, resolution);
      evaluations.push({
        spec,
        radius,
        unlimited: unlimitedScore.score,
        rgba4444: packedScore.score,
        packingIncrement: packingIncrement(unlimited, packed, resolution),
      });
      const normalized = Math.max(
        packedScore.score.coverage.p95 / LIMITS.coverageP95,
        packedScore.score.premulRgb.p95 / LIMITS.rgbP95,
        packedScore.score.largestConnectedFraction / LIMITS.connected,
      );
      updateWorst(worst, {
        score: normalized,
        title: `${referenceKey} ${spec.key} sigma=${radius}`,
        truth: truth.rgba,
        unlimited: unlimited.rgba,
        packed: packed.rgba,
        coverageError: packedScore.coverageError,
        rgbError: packedScore.rgbError,
      });
    }
  }

  const nodePacking: Record<string, unknown>[] = [];
  for (const spec of nodes) {
    for (const radius of SOURCE_FILTER_RADII) {
      const truth = nodesByRadius.get(radius)!.get(spec.key)!;
      const packed = new Float32Array(truth.rgba.length);
      for (let index = 0; index < packed.length; index++) packed[index] = quantize4(truth.rgba[index]!);
      nodePacking.push({ spec, radius, rgba4444: scorePage(truth, { rgba: packed }, resolution).score });
    }
  }

  mkdirSync(qaRoot, { recursive: true });
  for (let index = 0; index < worst.length; index++) {
    await writeQa(resolve(qaRoot, `${String(index + 1).padStart(3, '0')}-${referenceKey}-worst.png`), worst[index]!, resolution);
  }
  const everyUnlimited = evaluations.every((entry) => entry.unlimited.green);
  const everyPacked = evaluations.every((entry) => entry.rgba4444.green);
  const everyNodePacked = nodePacking.every((entry) => (entry.rgba4444 as PageScore).green);
  return {
    referenceKey,
    referenceY,
    heldoutCount: heldout.length,
    verdict: everyUnlimited && everyPacked && everyNodePacked ? 'GREEN' : 'RED',
    everyUnlimitedHeldoutGreen: everyUnlimited,
    everyRgba4444HeldoutGreen: everyPacked,
    everyStoredNodeRgba4444Green: everyNodePacked,
    heldout: evaluations,
    storedNodePacking: nodePacking,
  };
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const sourcePath = resolve(WORKSPACE, typeof args.source === 'string' ? args.source : DEFAULT_SOURCE);
  const sourceBytes = readFileSync(sourcePath);
  const sourceHash = sha256(sourceBytes);
  const resolution = Number(typeof args.resolution === 'string' ? args.resolution : 128);
  if (!Number.isInteger(resolution) || resolution < 4) throw new Error('--resolution must be an integer >= 4');
  const maximumHeldout = typeof args['max-heldout'] === 'string' ? Number(args['max-heldout']) : null;
  const requestedReferences = String(typeof args.references === 'string' ? args.references : 'middle,top').split(',');
  const references = requestedReferences.map((key) => {
    if (key === 'middle') return { key, y: MIDDLE_REFERENCE_Y };
    if (key === 'top') return { key, y: Number.NaN };
    throw new Error(`unknown reference ${key}`);
  });
  const geometry = decodeOwnedProfileGeometry(sourceBytes);
  const bvh = TriangleBvh.build(geometry);
  const attributes = sourceAttributes(sourceBytes);
  const recipe = {
    version: VERSION,
    sourceSha256: sourceHash,
    scriptSha256: sha256(readFileSync(fileURLToPath(import.meta.url))),
    thresholdDocSha256: sha256(readFileSync(THRESHOLD_DOC)),
    resolution,
    maximumHeldout,
    references: references.map((entry) => entry.key),
    middleReferenceY: MIDDLE_REFERENCE_Y,
    ringsDegrees: RINGS_DEGREES,
    heldoutElevations: HELDOUT_ELEVATIONS,
    heldoutSectors: HELDOUT_SECTORS,
    sourceFilterRadiiAt256: SOURCE_FILTER_RADII,
    effectiveFilterRadii: SOURCE_FILTER_RADII.map((radius) => radius * resolution / 256),
    limits: LIMITS,
  };
  const recipeHash = sha256(canonicalJson(recipe));
  const output = resolve(WORKSPACE, 'data/work/groundcover-candidate-k-heldout-fe', sourceHash.slice(0, 16), recipeHash.slice(0, 16));
  mkdirSync(output, { recursive: true });
  const results: Record<string, unknown> = {};
  for (const reference of references) {
    const y = reference.key === 'top' ? geometry.topH : reference.y;
    results[reference.key] = await evaluateReference(
      geometry,
      bvh,
      attributes,
      resolution,
      reference.key,
      y,
      maximumHeldout,
      resolve(output, 'qa', reference.key),
    );
  }
  const verdict = Object.values(results).every((entry) => (entry as { verdict: string }).verdict === 'GREEN') ? 'GREEN' : 'RED';
  const report = {
    schema: VERSION,
    verdict,
    source: {
      path: sourcePath,
      sha256: sourceHash,
      topH: geometry.topH,
      bounds: geometry.bounds,
      triangleCount: geometry.triangleCount,
      vertexCount: geometry.vertexCount,
    },
    bvh: bvh.metrics,
    recipe,
    results,
  };
  writeFileSync(resolve(output, 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
  writeFileSync(resolve(output, 'SUMMARY.md'), [
    '# Candidate K held-out FE result',
    '',
    `Verdict: **${verdict}**`,
    '',
    `Source SHA-256: \`${sourceHash}\``,
    `Recipe SHA-256: \`${recipeHash}\``,
    `Phase resolution: \`${resolution}\``,
    '',
    ...Object.entries(results).map(([key, value]) => {
      const result = value as Record<string, unknown>;
      return `- ${key}: ${result.verdict}; unlimited=${result.everyUnlimitedHeldoutGreen}; RGBA4444=${result.everyRgba4444HeldoutGreen}; stored nodes=${result.everyStoredNodeRgba4444Green}`;
    }),
    '',
    'See `report.json` for every binding direction and `qa/` for the worst cases.',
  ].join('\n'));
  console.log(JSON.stringify({ output, verdict, sourceHash, recipeHash, results: Object.fromEntries(
    Object.entries(results).map(([key, value]) => [key, {
      verdict: (value as Record<string, unknown>).verdict,
      unlimited: (value as Record<string, unknown>).everyUnlimitedHeldoutGreen,
      rgba4444: (value as Record<string, unknown>).everyRgba4444HeldoutGreen,
    }]),
  ) }, null, 2));
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await main();
