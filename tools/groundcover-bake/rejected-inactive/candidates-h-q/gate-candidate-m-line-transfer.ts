/**
 * Candidate-M decisive line-conditioned analytic-transfer gate.
 *
 * Offline only.  Fits the frozen four-mode positive model to exact-BVH
 * positive-measure pages and evaluates directions/phases excluded from fit.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import sharp from 'sharp';
import { TriangleBvh, decodeOwnedProfileGeometry } from '../../ExteriorClosureCensus';
import {
  renderPointPage,
  scorePage,
  sourceAttributes,
  toroidalBoxFilter,
  sha256,
  type DirectionSpec,
  type FieldPage,
} from './gate-candidate-k-heldout-fe';

const WORKSPACE = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const SOURCE = resolve(WORKSPACE, 'src/assets/groundcover/calamagrostis-canescens.gcrp');
const SPEC = resolve(WORKSPACE, 'docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-CANDIDATE-M-LINE-CONDITIONED-ANALYTIC-TRANSFER.md');
const VERSION = 'candidate-m-line-transfer-gate-v1';
const N = 64;
const SOURCE_RADII = [4, 16] as const;
const EFFECTIVE_RADII = [1, 4] as const;
const TRAIN_ELEVATIONS = [0.25, 2, 5, 10, 18, 30, 55, 75, 90] as const;
const HELDOUT_ELEVATIONS = [0.1, 1, 3.5, 7.5, 14, 24, 45, 65, 82, 89] as const;
const AZ_STEP = 360 / 64;
const AZ_STARTS = [0, 90, 180, 270] as const;
const PHASE_TESTS = [
  { key: 'node', x: 0, z: 0 },
  { key: 'cell-centre', x: 0.5, z: 0.5 },
  { key: 'translation-4.5mm', x: 4.5e-3 / 0.52 * N, z: 0 },
] as const;
const LIMITS = { coverageP95: 0.08, coverageP99: 0.20, rgbP95: 0.06, rgbP99: 0.15, connected: 0.01 } as const;

type Rates = { b: readonly number[]; gamma: readonly number[]; key: string };
type CoeffField = Float32Array; // pixel-major, [B0..B3,A0..A3]
type FitPair = { fine: CoeffField; coarseIndependent: CoeffField; coarseContracted: CoeffField };

const RATE_CANDIDATES: readonly Rates[] = [
  { key: 'r0', b: [0, 0.5, 2, 8], gamma: [0, 0.25, 1, 4] },
  { key: 'r1', b: [0, 0.25, 1, 4], gamma: [0.05, 0.3, 1.5, 6] },
  { key: 'r2', b: [0, 1, 4, 16], gamma: [0, 0.5, 2, 8] },
  { key: 'r3', b: [0.25, 1, 4, 16], gamma: [0.05, 0.25, 1, 4] },
  { key: 'r4', b: [0, 0.5, 4, 16], gamma: [0, 0.5, 1.5, 6] },
  { key: 'r5', b: [0, 0.25, 2, 12], gamma: [0.05, 0.5, 2, 8] },
] as const;

function canonical(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  const object = value as Record<string, unknown>;
  return `{${Object.keys(object).sort().map((key) => `${JSON.stringify(key)}:${canonical(object[key])}`).join(',')}}`;
}

function clamp(value: number, low: number, high: number): number { return Math.max(low, Math.min(high, value)); }
function mod(value: number, n: number): number { return ((value % n) + n) % n; }
function directionSpec(elevationDegrees: number, azimuthDegrees: number, key = ''): DirectionSpec {
  return { key: key || `e${elevationDegrees}-a${azimuthDegrees}`, elevationDegrees, azimuthDegrees: mod(azimuthDegrees, 360), kind: 'heldout' };
}

function eIntegral(z: number, r: number): number {
  const zr = z * r;
  if (Math.abs(zr) < 1e-4) return r * (1 - zr / 2 + zr * zr / 6 - zr * zr * zr / 24);
  return -Math.expm1(-zr) / z;
}

function basis(rates: Rates, elevationDegrees: number, height: number): Float64Array {
  const result = new Float64Array(8);
  if (elevationDegrees >= 90 - 1e-9) {
    for (let k = 0; k < 4; k++) {
      const gamma = rates.gamma[k]!;
      result[k] = gamma > 0 ? -Math.expm1(-gamma * height) / gamma : height;
      result[4 + k] = 0;
    }
    return result;
  }
  const e = elevationDegrees * Math.PI / 180;
  const sp = Math.cos(e);
  const lambda = Math.tan(e);
  const r = height / lambda;
  for (let k = 0; k < 4; k++) {
    result[k] = eIntegral(rates.gamma[k]! * lambda, r) / sp;
    result[4 + k] = eIntegral(rates.b[k]! + rates.gamma[k]! * lambda, r);
  }
  return result;
}

function opticalTarget(page: FieldPage, pixel: number): readonly [number, number, number, number] {
  const o = pixel * 4;
  const a = clamp(page.rgba[o + 3]!, 0, 0.999);
  const tau = -Math.log1p(-a);
  if (a <= 1e-8 || tau <= 1e-8) return [0, 0, 0, 0];
  return [tau, tau * page.rgba[o]! / a, tau * page.rgba[o + 1]! / a, tau * page.rgba[o + 2]! / a];
}

function solveDense(matrix: Float64Array, rhs: Float64Array, size: number): Float64Array {
  const a = new Float64Array(matrix);
  const b = new Float64Array(rhs);
  for (let column = 0; column < size; column++) {
    let pivot = column;
    for (let row = column + 1; row < size; row++) if (Math.abs(a[row * size + column]!) > Math.abs(a[pivot * size + column]!)) pivot = row;
    if (pivot !== column) {
      for (let j = column; j < size; j++) [a[column * size + j], a[pivot * size + j]] = [a[pivot * size + j]!, a[column * size + j]!];
      [b[column], b[pivot]] = [b[pivot]!, b[column]!];
    }
    const denominator = a[column * size + column]!;
    if (Math.abs(denominator) < 1e-14) continue;
    for (let row = column + 1; row < size; row++) {
      const factor = a[row * size + column]! / denominator;
      for (let j = column; j < size; j++) a[row * size + j] -= factor * a[column * size + j]!;
      b[row] -= factor * b[column]!;
    }
  }
  const x = new Float64Array(size);
  for (let row = size - 1; row >= 0; row--) {
    let value = b[row]!;
    for (let j = row + 1; j < size; j++) value -= a[row * size + j]! * x[j]!;
    x[row] = Math.abs(a[row * size + row]!) > 1e-14 ? value / a[row * size + row]! : 0;
  }
  return x;
}

function nnls(design: readonly Float64Array[], target: readonly number[], first: number, fixed: Float64Array): Float64Array {
  const count = 8 - first;
  const reducedTarget = target.map((value, e) => {
    let residual = value;
    for (let j = 0; j < first; j++) residual -= design[e]![j]! * fixed[j]!;
    return residual;
  });
  const g = new Float64Array(count * count);
  const rhs = new Float64Array(count);
  for (let e = 0; e < design.length; e++) for (let i = 0; i < count; i++) {
    const ai = design[e]![first + i]!;
    rhs[i] += ai * reducedTarget[e]!;
    for (let j = 0; j < count; j++) g[i * count + j] += ai * design[e]![first + j]!;
  }
  const x = new Float64Array(count);
  const passive = new Uint8Array(count);
  for (let outer = 0; outer < count * 3; outer++) {
    let best = -1;
    let bestGradient = 1e-10;
    for (let i = 0; i < count; i++) if (!passive[i]) {
      let gradient = rhs[i]!;
      for (let j = 0; j < count; j++) gradient -= g[i * count + j]! * x[j]!;
      if (gradient > bestGradient) { bestGradient = gradient; best = i; }
    }
    if (best < 0) break;
    passive[best] = 1;
    for (let inner = 0; inner < count * 2; inner++) {
      const indices = [...passive].map((value, index) => value ? index : -1).filter((index) => index >= 0);
      const n = indices.length;
      const pg = new Float64Array(n * n);
      const pr = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        pr[i] = rhs[indices[i]!]!;
        for (let j = 0; j < n; j++) pg[i * n + j] = g[indices[i]! * count + indices[j]!]! + (i === j ? 1e-12 : 0);
      }
      const solution = solveDense(pg, pr, n);
      if (solution.every((value) => value > 0)) {
        x.fill(0);
        for (let i = 0; i < n; i++) x[indices[i]!] = solution[i]!;
        break;
      }
      let alpha = 1;
      for (let i = 0; i < n; i++) if (solution[i]! <= 0) alpha = Math.min(alpha, x[indices[i]!]! / Math.max(x[indices[i]!]! - solution[i]!, 1e-30));
      for (let i = 0; i < count; i++) x[i] += alpha * ((passive[i] ? solution[indices.indexOf(i)]! : 0) - x[i]!);
      for (let i = 0; i < count; i++) if (passive[i] && x[i]! <= 1e-10) { x[i] = 0; passive[i] = 0; }
    }
  }
  const result = new Float64Array(fixed);
  for (let i = 0; i < count; i++) result[first + i] = x[i]!;
  return result;
}

function fitNonnegative(
  pages: readonly FieldPage[],
  design: readonly Float64Array[],
  fixedB?: CoeffField,
  stride = 1,
): { coefficients: CoeffField; squaredError: number; samples: number } {
  const pixels = N * N;
  const coefficients = new Float32Array(pixels * 8);
  let squaredError = 0;
  let samples = 0;
  for (let pixel = 0; pixel < pixels; pixel += stride) {
    const target = pages.map((page) => opticalTarget(page, pixel)[0]);
    const fixed = new Float64Array(8);
    if (fixedB) for (let k = 0; k < 4; k++) fixed[k] = fixedB[pixel * 8 + k]!;
    const first = fixedB ? 4 : 0;
    const x = nnls(design, target, first, fixed);
    for (let j = 0; j < 8; j++) coefficients[pixel * 8 + j] = x[j]!;
    for (let e = 0; e < design.length; e++) {
      let prediction = 0;
      for (let j = 0; j < 8; j++) prediction += design[e]![j]! * x[j]!;
      squaredError += (prediction - target[e]!) ** 2;
      samples++;
    }
  }
  return { coefficients, squaredError, samples };
}

function averageBase(fields: readonly CoeffField[]): CoeffField {
  const result = new Float32Array(N * N * 8);
  for (let pixel = 0; pixel < N * N; pixel++) for (let k = 0; k < 4; k++) {
    let sum = 0;
    for (const field of fields) sum += field[pixel * 8 + k]!;
    result[pixel * 8 + k] = sum / fields.length;
  }
  return result;
}

function phaseMeans(field: CoeffField): Float64Array {
  const means = new Float64Array(8);
  for (let pixel = 0; pixel < N * N; pixel++) for (let k = 0; k < 8; k++) means[k] += field[pixel * 8 + k]!;
  for (let k = 0; k < 8; k++) means[k] /= N * N;
  return means;
}

function contractFineToCoarse(fine: CoeffField, coarse: CoeffField): { field: CoeffField; weights: number[] } {
  const mean = phaseMeans(fine);
  const weights = new Array<number>(8).fill(0);
  for (let k = 0; k < 8; k++) {
    let numerator = 0;
    let denominator = 0;
    for (let p = 0; p < N * N; p++) {
      const d = fine[p * 8 + k]! - mean[k]!;
      numerator += d * (coarse[p * 8 + k]! - mean[k]!);
      denominator += d * d;
    }
    weights[k] = clamp(numerator / Math.max(denominator, 1e-18), 0, 1);
  }
  const field = new Float32Array(fine.length);
  for (let p = 0; p < N * N; p++) for (let k = 0; k < 8; k++) field[p * 8 + k] = mean[k]! + weights[k]! * (fine[p * 8 + k]! - mean[k]!);
  return { field, weights };
}

function sampleCoefficient(field: CoeffField, pixel: number, component: number, ox: number, oz: number): number {
  const x = pixel % N;
  const z = Math.floor(pixel / N);
  const qx = x + ox;
  const qz = z + oz;
  const x0 = Math.floor(qx);
  const z0 = Math.floor(qz);
  const fx = qx - x0;
  const fz = qz - z0;
  const at = (ix: number, iz: number): number => field[(mod(iz, N) * N + mod(ix, N)) * 8 + component]!;
  return (1 - fz) * ((1 - fx) * at(x0, z0) + fx * at(x0 + 1, z0))
    + fz * ((1 - fx) * at(x0, z0 + 1) + fx * at(x0 + 1, z0 + 1));
}

function solveColours(
  training: readonly { pages: readonly FieldPage[]; field: CoeffField }[],
  design: readonly Float64Array[],
): number[][] {
  const g = new Float64Array(16);
  const rhs = [new Float64Array(4), new Float64Array(4), new Float64Array(4)];
  for (const item of training) for (let p = 0; p < N * N; p += 2) for (let e = 0; e < design.length; e++) {
    const w = new Float64Array(4);
    for (let k = 0; k < 4; k++) w[k] = design[e]![k]! * item.field[p * 8 + k]! + design[e]![4 + k]! * item.field[p * 8 + 4 + k]!;
    const target = opticalTarget(item.pages[e]!, p);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) g[i * 4 + j] += w[i]! * w[j]!;
      for (let c = 0; c < 3; c++) rhs[c]![i] += w[i]! * target[1 + c]!;
    }
  }
  const colours = Array.from({ length: 4 }, () => [0, 0, 0]);
  for (let c = 0; c < 3; c++) {
    const x = new Float64Array(4);
    for (let sweep = 0; sweep < 80; sweep++) for (let i = 0; i < 4; i++) {
      let other = 0;
      for (let j = 0; j < 4; j++) if (i !== j) other += g[i * 4 + j]! * x[j]!;
      x[i] = clamp((rhs[c]![i]! - other) / Math.max(g[i * 4 + i]!, 1e-18), 0, 1);
    }
    for (let i = 0; i < 4; i++) colours[i]![c] = x[i]!;
  }
  return colours;
}

function quantizers(fields: readonly CoeffField[]): { scale: number[]; encodeDecode(value: number, k: number): number } {
  const maxima = new Array<number>(8).fill(0);
  for (const field of fields) for (let i = 0; i < field.length; i++) maxima[i % 8] = Math.max(maxima[i % 8]!, field[i]!);
  const scale = maxima.map((maximum) => Math.max(maximum / 255, 1e-8));
  return {
    scale,
    encodeDecode(value: number, k: number): number {
      const s = scale[k]!;
      const maximum = maxima[k]!;
      const denominator = Math.log1p(maximum / s);
      if (!(denominator > 0)) return 0;
      const code = Math.round(255 * Math.log1p(Math.max(0, value) / s) / denominator);
      return s * Math.expm1(clamp(code, 0, 255) / 255 * denominator);
    },
  };
}

function predict(
  low: CoeffField,
  high: CoeffField,
  azMix: number,
  elevation: number,
  phaseX: number,
  phaseZ: number,
  rates: Rates,
  height: number,
  colours: readonly number[][],
  quantizer: ReturnType<typeof quantizers> | null,
): FieldPage {
  const d = basis(rates, elevation, height);
  const rgba = new Float32Array(N * N * 4);
  for (let p = 0; p < N * N; p++) {
    let tau = 0;
    const source = [0, 0, 0];
    for (let k = 0; k < 4; k++) {
      let b = (1 - azMix) * sampleCoefficient(low, p, k, phaseX, phaseZ) + azMix * sampleCoefficient(high, p, k, phaseX, phaseZ);
      let a = (1 - azMix) * sampleCoefficient(low, p, 4 + k, phaseX, phaseZ) + azMix * sampleCoefficient(high, p, 4 + k, phaseX, phaseZ);
      if (quantizer) {
        b = quantizer.encodeDecode(b, k);
        a = quantizer.encodeDecode(a, 4 + k);
      }
      const modeTau = b * d[k]! + a * d[4 + k]!;
      tau += modeTau;
      for (let c = 0; c < 3; c++) source[c] += modeTau * colours[k]![c]!;
    }
    const alpha = -Math.expm1(-Math.max(0, tau));
    const o = p * 4;
    rgba[o + 3] = alpha;
    if (tau > 1e-8) for (let c = 0; c < 3; c++) rgba[o + c] = alpha * source[c]! / tau;
  }
  return { rgba };
}

async function writeQa(path: string, truth: FieldPage, prediction: FieldPage): Promise<void> {
  const panels = 3;
  const pixels = Buffer.alloc(N * panels * N * 3);
  for (let p = 0; p < N * N; p++) {
    const x = p % N;
    const y = Math.floor(p / N);
    const values = [truth.rgba, prediction.rgba];
    for (let panel = 0; panel < 2; panel++) for (let c = 0; c < 3; c++) {
      const target = (y * N * panels + panel * N + x) * 3 + c;
      pixels[target] = Math.round(255 * clamp(values[panel]![p * 4 + c]! + 0.12 * (1 - values[panel]![p * 4 + 3]!), 0, 1));
    }
    const error = Math.max(...[0, 1, 2, 3].map((c) => Math.abs(truth.rgba[p * 4 + c]! - prediction.rgba[p * 4 + c]!)));
    const target = (y * N * panels + 2 * N + x) * 3;
    pixels[target] = Math.round(255 * clamp(error / 0.2, 0, 1));
    pixels[target + 1] = Math.round(180 * clamp(error / 0.2, 0, 1));
  }
  await sharp(pixels, { raw: { width: N * panels, height: N, channels: 3 } }).resize({ width: N * panels * 4, height: N * 4, kernel: 'nearest' }).png().toFile(path);
}

async function main(): Promise<void> {
  const sourceBytes = readFileSync(SOURCE);
  const geometry = decodeOwnedProfileGeometry(sourceBytes);
  const bvh = TriangleBvh.build(geometry);
  const attributes = sourceAttributes(sourceBytes);
  const height = geometry.topH - geometry.bounds.min[1];
  const sourceHash = sha256(sourceBytes);
  const trainAzimuths = AZ_STARTS.flatMap((start) => [start, start + AZ_STEP]);
  const truth = new Map<string, readonly [FieldPage, FieldPage]>();
  const truthKey = (az: number, e: number, ox = 0, oz = 0) => `${mod(az, 360).toFixed(6)}:${e.toFixed(6)}:${ox.toFixed(6)}:${oz.toFixed(6)}`;
  const render = (az: number, e: number, ox = 0, oz = 0): readonly [FieldPage, FieldPage] => {
    const key = truthKey(az, e, ox, oz);
    const found = truth.get(key);
    if (found) return found;
    const point = renderPointPage(geometry, bvh, attributes, N, geometry.topH, directionSpec(e, az), ox, oz);
    const pages = EFFECTIVE_RADII.map((radius) => toroidalBoxFilter(point, N, radius)) as unknown as readonly [FieldPage, FieldPage];
    truth.set(key, pages);
    return pages;
  };

  console.log(`[candidate-m] exact-BVH training pages: ${trainAzimuths.length * TRAIN_ELEVATIONS.length}`);
  for (const az of trainAzimuths) for (const e of TRAIN_ELEVATIONS) render(az, e);

  const sampledAz = trainAzimuths[0]!;
  const sampledPages = TRAIN_ELEVATIONS.map((e) => render(sampledAz, e)[0]);
  const tuning = RATE_CANDIDATES.map((rates) => {
    const design = TRAIN_ELEVATIONS.map((e) => basis(rates, e, height));
    const fit = fitNonnegative(sampledPages, design, undefined, 16);
    return { key: rates.key, mse: fit.squaredError / fit.samples };
  }).sort((a, b) => a.mse - b.mse);
  const rates = RATE_CANDIDATES.find((candidate) => candidate.key === tuning[0]!.key)!;
  const design = TRAIN_ELEVATIONS.map((e) => basis(rates, e, height));
  console.log(`[candidate-m] selected ${rates.key} training MSE=${tuning[0]!.mse}`);

  const independent = new Map<string, readonly [CoeffField, CoeffField]>();
  for (const az of trainAzimuths) {
    const fitted = [0, 1].map((scale) => fitNonnegative(TRAIN_ELEVATIONS.map((e) => render(az, e)[scale]!), design).coefficients) as unknown as readonly [CoeffField, CoeffField];
    independent.set(az.toFixed(6), fitted);
  }
  for (let scale = 0; scale < 2; scale++) {
    const base = averageBase(trainAzimuths.map((az) => independent.get(az.toFixed(6))![scale]!));
    for (const az of trainAzimuths) {
      const pair = [...independent.get(az.toFixed(6))!] as [CoeffField, CoeffField];
      pair[scale] = fitNonnegative(TRAIN_ELEVATIONS.map((e) => render(az, e)[scale]!), design, base).coefficients;
      independent.set(az.toFixed(6), pair);
    }
  }

  const fits = new Map<string, FitPair>();
  for (const az of trainAzimuths) {
    const [fine, coarseIndependent] = independent.get(az.toFixed(6))!;
    const contracted = contractFineToCoarse(fine, coarseIndependent);
    fits.set(az.toFixed(6), { fine, coarseIndependent, coarseContracted: contracted.field });
  }
  const colourTraining: { pages: readonly FieldPage[]; field: CoeffField }[] = [];
  for (const az of trainAzimuths) colourTraining.push({ pages: TRAIN_ELEVATIONS.map((e) => render(az, e)[0]), field: fits.get(az.toFixed(6))!.fine });
  const colours = solveColours(colourTraining, design);
  const allFine = [...fits.values()].map((fit) => fit.fine);
  const quantizer = quantizers(allFine);

  const recipe = {
    version: VERSION,
    sourceSha256: sourceHash,
    scriptSha256: sha256(readFileSync(fileURLToPath(import.meta.url))),
    specSha256: sha256(readFileSync(SPEC)),
    resolution: N,
    sourceFilterRadiiAt256: SOURCE_RADII,
    trainElevations: TRAIN_ELEVATIONS,
    heldoutElevations: HELDOUT_ELEVATIONS,
    trainAzimuths,
    heldoutAzimuths: AZ_STARTS.map((start) => start + AZ_STEP / 2),
    phaseTests: PHASE_TESTS,
    rates,
    rateTuning: tuning,
    limits: LIMITS,
    model: 'frozen four modes; B shared across trained azimuths; sigma16 analytic contraction from sigma4; global colours; UNORM8 log decode',
  };
  const recipeHash = createHash('sha256').update(canonical(recipe)).digest('hex');
  const output = resolve(WORKSPACE, 'data/work/groundcover-candidate-m-line-transfer', sourceHash.slice(0, 16), recipeHash.slice(0, 16));
  const qaRoot = resolve(output, 'qa');
  mkdirSync(qaRoot, { recursive: true });

  const records: Record<string, unknown>[] = [];
  const worst: { normalized: number; key: string; truth: FieldPage; prediction: FieldPage }[] = [];
  const evaluate = (az0: number, az1: number, azMix: number, e: number, phase: typeof PHASE_TESTS[number], scale: 0 | 1): void => {
    const actualAz = az0 + azMix * (az1 - az0);
    const actualTruth = render(actualAz, e, phase.x, phase.z)[scale];
    const lowFit = fits.get(mod(az0, 360).toFixed(6))!;
    const highFit = fits.get(mod(az1, 360).toFixed(6))!;
    const low = scale === 0 ? lowFit.fine : lowFit.coarseContracted;
    const high = scale === 0 ? highFit.fine : highFit.coarseContracted;
    const unlimited = predict(low, high, azMix, e, phase.x, phase.z, rates, height, colours, null);
    const packed = predict(low, high, azMix, e, phase.x, phase.z, rates, height, colours, quantizer);
    const u = scorePage(actualTruth, unlimited, N).score;
    const p = scorePage(actualTruth, packed, N).score;
    const normalized = Math.max(p.coverage.p95 / LIMITS.coverageP95, p.premulRgb.p95 / LIMITS.rgbP95, p.largestConnectedFraction / LIMITS.connected);
    const key = `a${actualAz.toFixed(4)}-e${e}-${phase.key}-sigma${SOURCE_RADII[scale]}`;
    records.push({ key, azimuthDegrees: actualAz, elevationDegrees: e, phase: phase.key, sourceRadius: SOURCE_RADII[scale], unlimited: u, quantized: p });
    worst.push({ normalized, key, truth: actualTruth, prediction: packed });
  };

  for (const start of AZ_STARTS) for (const e of HELDOUT_ELEVATIONS) for (const scale of [0, 1] as const) {
    evaluate(start, start + AZ_STEP, 0, e, PHASE_TESTS[0], scale);
    evaluate(start, start + AZ_STEP, 0.5, e, PHASE_TESTS[0], scale);
  }
  for (const start of AZ_STARTS) for (const phase of PHASE_TESTS.slice(1)) for (const scale of [0, 1] as const) {
    evaluate(start, start + AZ_STEP, 0.5, 14, phase, scale);
  }

  worst.sort((a, b) => b.normalized - a.normalized);
  for (let i = 0; i < Math.min(5, worst.length); i++) await writeQa(resolve(qaRoot, `${String(i + 1).padStart(3, '0')}-${worst[i]!.key}.png`), worst[i]!.truth, worst[i]!.prediction);
  const allUnlimitedGreen = records.every((record) => (record.unlimited as { green: boolean }).green);
  const allQuantizedGreen = records.every((record) => (record.quantized as { green: boolean }).green);
  const oldBoundaryJump = 0; // one shared linear field; no row/LOD/Voronoi selector exists
  const verdict = allUnlimitedGreen && allQuantizedGreen && oldBoundaryJump <= 1e-5 ? 'GREEN' : 'RED';
  const summary = {
    schema: VERSION,
    verdict,
    scope: 'periodic-interior fidelity subgate only; never an overall GREEN',
    source: { path: SOURCE, sha256: sourceHash, triangles: geometry.triangleCount, topH: geometry.topH, bounds: geometry.bounds },
    recipe,
    fit: {
      colours,
      quantizerScale: quantizer.scale,
      contractionWeights: Object.fromEntries([...fits.entries()].map(([key, value]) => [key, contractFineToCoarse(value.fine, value.coarseIndependent).weights])),
    },
    checks: { allUnlimitedGreen, allQuantizedGreen, oldDirectionBoundaryJump: oldBoundaryJump },
    evaluations: records,
  };
  writeFileSync(resolve(output, 'report.json'), `${JSON.stringify(summary, null, 2)}\n`);
  writeFileSync(resolve(output, 'SUMMARY.md'), [
    '# Candidate M line-conditioned analytic-transfer gate', '',
    `Verdict: **${verdict}**`, '',
    'Scope: periodic-interior fidelity subgate only; never an overall Candidate-M GREEN.', '',
    `Source: \`${sourceHash}\``,
    `Recipe: \`${recipeHash}\``,
    `Unquantized all-green: \`${allUnlimitedGreen}\``,
    `Quantized all-green: \`${allQuantizedGreen}\``,
    `Old categorical boundary jump: \`${oldBoundaryJump}\``, '',
    `Worst normalized case: \`${worst[0]?.key}\` at \`${worst[0]?.normalized.toFixed(3)}x\` the binding limits.`,
    '', 'QA panels are truth | quantized Candidate M | max-channel error.',
  ].join('\n'));
  console.log(JSON.stringify({ output, verdict, allUnlimitedGreen, allQuantizedGreen, worst: worst.slice(0, 5).map((entry) => ({ key: entry.key, normalized: entry.normalized })) }, null, 2));
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await main();
