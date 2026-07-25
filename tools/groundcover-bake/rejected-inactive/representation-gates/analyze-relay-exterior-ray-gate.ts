/**
 * One offline exterior first-visible gate for the F=4 lateral relay core plus
 * a direction-analytic soft residual with K_total=4 after expanding the two
 * Tier-1 affine layers (K=2 in each layer).
 *
 * Reference and candidate are integrated over the identical perspective pixel
 * ray bundle.  Distance therefore removes only detail that the reference also
 * loses; it never grants a waiver to coherent screen-space artifacts.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import sharp from 'sharp';
import {
  TriangleBvh,
  type DecodedOwnedProfileGeometry,
} from '../../ExteriorClosureCensus';
import {
  ESTONIAN_GRAMINOID_PROFILE_IDS,
  makeEstonianGraminoidFixture,
  type GraminoidPrimitiveRecipe,
} from '../../EstonianGraminoids';
import { periodicNearestSuccessor, type OriginAwareTruthHit } from '../../OriginAwareRayTruth';
import type { IndexedMesh } from '../../ProfileFormat';

type Vec2 = readonly [number, number];
type Vec3 = readonly [number, number, number];
type Payload = 'foliage' | 'reproductive-purple' | 'reproductive-cream' | 'structure';
interface Complex { re: number; im: number; }
interface ArcEdge { a: Vec2; b: Vec2; color: Vec3; normal: Vec3; }
interface ArcNetwork {
  payload: Exclude<Payload, 'structure'>;
  candidate: { beta: Vec2; hMin: number; hMax: number };
  edges: ArcEdge[];
}
interface RuledNetworkEntry { field: number; kind?: 'ruled'; network: ArcNetwork; }
interface SheetCell { ix: number; iz: number; color: Vec3; normal: Vec3; payload: Exclude<Payload, 'foliage' | 'structure'>; }
interface SheetNetworkEntry { field: number; kind: 'sheet'; sheet: { height: number; resolution: number; cells: SheetCell[] }; }
type NetworkEntry = RuledNetworkEntry | SheetNetworkEntry;
interface SurfaceHit { t: number; point: Vec3; color: Vec3; payload: Payload; triangleId: number; }
interface CoreHit extends SurfaceHit { payload: Exclude<Payload, 'structure'>; }
interface Mode { mx: number; my: number; mz: number; k: Vec3; c: Complex; d: readonly [Complex, Complex, Complex]; }
interface ResidualModel {
  hMin: number; hMax: number; a0: number; b0: Vec3; modes: Mode[];
  opacityModeScale: number; colorModeScales: Vec3; kappaMinimum: number;
}
interface CameraSpec { key: string; elevationDegrees: number; azimuthDegrees: number; distanceMetres: number; }
interface LayerTransform { key: string; angle: number; scale: number; phaseX: number; phaseZ: number; }
interface FilteredPixel {
  truthAlpha: number; truthPremul: Vec3; truthPayload: Record<Payload, number>;
  candidateAlpha: number; candidatePremul: Vec3;
  coreHitFraction: number; coreMatchedFraction: number; coreFalseForegroundFraction: number;
  coreFalseCoverageFraction: number; coreEarlyDepthExcessFraction: number;
  firstHitErrors: number[];
}

const ROOT = resolve(import.meta.dirname, '../../../..');
const NETWORK_PATH = resolve(ROOT, process.env.LAAS_GROUNDCOVER_NETWORK_PATH
  ?? 'data/work/groundcover-relay-arc-network/37b0cf1d33f632b5/6219c2153f9523c7/9917249acc582ac1/network.json');
const OUTPUT_ROOT = resolve(ROOT, process.env.LAAS_GROUNDCOVER_GATE_OUTPUT
  ?? 'data/work/groundcover-relay-exterior-ray-gate');
const EXPECTED_MESH_SHA = '37b0cf1d33f632b5cde3dd60410ad987eec2e977661d0bc804dd6f2bee7054c0';
const NETWORK_VARIANT = process.env.LAAS_GROUNDCOVER_NETWORK_VARIANT ?? '45';
const RESIDUAL_ENABLED = process.env.LAAS_GROUNDCOVER_RESIDUAL !== '0';
const WIDTH = 24;
const HEIGHT = 18;
const SUBPIXEL_SIDE = 4;
const PIXEL_ANGLE = 55 * Math.PI / 180 / 2160;
const DISTANCES = [1, 2, 4, 8, 16, 32] as const;
const VIEW_DIRECTIONS = [
  { key: 'vertical-90-az0', elevationDegrees: 90, azimuthDegrees: 0 },
  { key: 'oblique-18-az0', elevationDegrees: 18, azimuthDegrees: 0 },
  { key: 'oblique-10-az270', elevationDegrees: 10, azimuthDegrees: 270 },
  { key: 'low-oblique-5-az90', elevationDegrees: 5, azimuthDegrees: 90 },
  { key: 'near-horizontal-1-az180', elevationDegrees: 1, azimuthDegrees: 180 },
] as const;
const FIT_PHASE_SIDE = 6;
const FIT_SUBPIXEL_SIDE = 4;
const MODE_LIMIT_PER_LAYER = 2;
const MODE_LIMIT_TOTAL = MODE_LIMIT_PER_LAYER * 2;
const COMPOSITING_DEPTH_TOLERANCE_METRES = 0.05;
const TRANSLATIONS_METRES = [0.001, 0.0025, 0.0045] as const;
const LAYER_TRANSFORMS: readonly LayerTransform[] = [
  { key: 'layer0', angle: 0, scale: 1, phaseX: 0, phaseZ: 0 },
  { key: 'layer1', angle: Math.PI * 1.618033988749895, scale: 1.071773462536293, phaseX: 0.371, phaseZ: 0.619 },
];
const BACKGROUND: Vec3 = [0.115, 0.09, 0.065];
const QA_SCALE = 10;
const TAU = Math.PI * 2;
const EPSILON = 1e-10;
const PALETTE = [
  { rgb: [0.10, 0.27, 0.055] as Vec3, payload: 'foliage' as Payload },
  { rgb: [0.37, 0.51, 0.13] as Vec3, payload: 'foliage' as Payload },
  { rgb: [0.43, 0.53, 0.20] as Vec3, payload: 'foliage' as Payload },
  { rgb: [0.31, 0.24, 0.075] as Vec3, payload: 'structure' as Payload },
  { rgb: [0.48, 0.31, 0.24] as Vec3, payload: 'reproductive-purple' as Payload },
  { rgb: [0.57, 0.35, 0.39] as Vec3, payload: 'reproductive-purple' as Payload },
  { rgb: [0.88, 0.80, 0.70] as Vec3, payload: 'reproductive-cream' as Payload },
  { rgb: [0.42, 0.10, 0.34] as Vec3, payload: 'reproductive-purple' as Payload },
] as const;

function sha256(value: Uint8Array | string): string { return createHash('sha256').update(value).digest('hex'); }
function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(',')}}`;
}
function meshHash(mesh: IndexedMesh): string { return sha256(JSON.stringify(mesh)); }
function clamp(value: number, low: number, high: number): number { return Math.max(low, Math.min(high, value)); }
function add(a: Vec3, b: Vec3): Vec3 { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function scale(a: Vec3, s: number): Vec3 { return [a[0] * s, a[1] * s, a[2] * s]; }
function normalized(value: Vec3): Vec3 { const length = Math.hypot(...value); return [value[0] / length, value[1] / length, value[2] / length]; }
function transformRay(origin: Vec3, direction: Vec3, layer: LayerTransform): { origin: Vec3; direction: Vec3 } {
  const c = Math.cos(layer.angle); const s = Math.sin(layer.angle);
  return {
    origin: [layer.scale * (c * origin[0] - s * origin[2]) + layer.phaseX, origin[1], layer.scale * (s * origin[0] + c * origin[2]) + layer.phaseZ],
    direction: [layer.scale * (c * direction[0] - s * direction[2]), direction[1], layer.scale * (s * direction[0] + c * direction[2])],
  };
}
function cadd(a: Complex, b: Complex): Complex { return { re: a.re + b.re, im: a.im + b.im }; }
function cscale(a: Complex, s: number): Complex { return { re: a.re * s, im: a.im * s }; }
function cmul(a: Complex, b: Complex): Complex { return { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re }; }
function cabs(a: Complex): number { return Math.hypot(a.re, a.im); }
function cis(angle: number): Complex { return { re: Math.cos(angle), im: Math.sin(angle) }; }
function quantiles(valuesInput: readonly number[]): Record<string, number | null> {
  if (valuesInput.length === 0) return { p50: null, p90: null, p95: null, p99: null, maximum: null };
  const values = [...valuesInput].sort((a, b) => a - b);
  const at = (fraction: number): number => values[Math.floor((values.length - 1) * fraction)]!;
  return { p50: at(0.5), p90: at(0.9), p95: at(0.95), p99: at(0.99), maximum: values.at(-1)! };
}

function bounds(mesh: IndexedMesh): { min: Vec3; max: Vec3 } {
  const min = [Infinity, Infinity, Infinity]; const max = [-Infinity, -Infinity, -Infinity];
  for (let index = 0; index < mesh.positions.length; index += 3) for (let axis = 0; axis < 3; axis++) {
    min[axis] = Math.min(min[axis]!, mesh.positions[index + axis]!);
    max[axis] = Math.max(max[axis]!, mesh.positions[index + axis]!);
  }
  return { min: min as unknown as Vec3, max: max as unknown as Vec3 };
}
function decodedGeometry(mesh: IndexedMesh, tile: { originX: number; originZ: number; sizeX: number; sizeZ: number; topH: number }, profileId: number): DecodedOwnedProfileGeometry {
  const meshBounds = bounds(mesh);
  return {
    version: 4, profileId, topH: tile.topH,
    tileOriginX: tile.originX, tileOriginZ: tile.originZ, tileSizeX: tile.sizeX, tileSizeZ: tile.sizeZ,
    bounds: meshBounds,
    positions: Float64Array.from(mesh.positions), triangles: Uint32Array.from(mesh.indices),
    vertexCount: mesh.positions.length / 3, triangleCount: mesh.indices.length / 3,
  };
}

function recognitionClass(recipe: GraminoidPrimitiveRecipe): 'foliage' | 'reproductive' | 'structure' {
  if (recipe.kind === 'blade' || (recipe.kind === 'tube' && ['culm', 'basal-sheath', 'upper-sheath'].includes(recipe.role))) return 'foliage';
  if (recipe.kind === 'lanceolate-surface' || recipe.kind === 'hair-filament'
    || (recipe.kind === 'tube' && ['panicle-axis', 'spikelet-axis', 'anther-filament'].includes(recipe.role))) return 'reproductive';
  return 'structure';
}
function meshVertexColor(mesh: IndexedMesh, vertex: number): Vec3 {
  if (mesh.colors === undefined) throw new Error('exterior ray gate requires vertex colours');
  const offset = vertex * 3; return [mesh.colors[offset]!, mesh.colors[offset + 1]!, mesh.colors[offset + 2]!];
}
function meshTriangleColor(mesh: IndexedMesh, triangleId: number): Vec3 {
  const offset = triangleId * 3;
  const colors = [meshVertexColor(mesh, mesh.indices[offset]!), meshVertexColor(mesh, mesh.indices[offset + 1]!), meshVertexColor(mesh, mesh.indices[offset + 2]!)];
  return [0, 1, 2].map((channel) => colors.reduce((sum, color) => sum + color[channel]!, 0) / 3) as unknown as Vec3;
}
function nearestPayload(color: Vec3): Payload {
  let best = Infinity; let result: Payload = 'foliage';
  for (const entry of PALETTE) {
    const distance = (color[0] - entry.rgb[0]) ** 2 + (color[1] - entry.rgb[1]) ** 2 + (color[2] - entry.rgb[2]) ** 2;
    if (distance < best) { best = distance; result = entry.payload; }
  }
  return result;
}
function sourcePayloads(mesh: IndexedMesh, recipes: readonly GraminoidPrimitiveRecipe[]): Payload[] {
  const result = Array.from({ length: mesh.indices.length / 3 }, (_, triangleId) => nearestPayload(meshTriangleColor(mesh, triangleId)));
  for (const recipe of recipes) {
    const recognition = recognitionClass(recipe);
    for (let triangleId = recipe.sourceTriangleStart; triangleId < recipe.sourceTriangleStart + recipe.sourceTriangleCount; triangleId++) {
      result[triangleId] = recognition === 'foliage' ? 'foliage'
        : recognition === 'reproductive' ? nearestPayload(meshTriangleColor(mesh, triangleId)) : 'structure';
    }
  }
  return result;
}

function unwrapEdge(a: Vec2, bInput: Vec2, tile: number): readonly [Vec2, Vec2] {
  const b = [bInput[0], bInput[1]];
  for (let axis = 0; axis < 2; axis++) {
    while (b[axis]! - a[axis]! > tile * 0.5) b[axis] -= tile;
    while (b[axis]! - a[axis]! < -tile * 0.5) b[axis] += tile;
  }
  return [a, b as unknown as Vec2];
}
function compileCore(networkEntries: readonly NetworkEntry[], tile: number, topH: number): {
  geometry: DecodedOwnedProfileGeometry; colors: Vec3[]; payloads: Array<Exclude<Payload, 'structure'>>;
} {
  const positions: number[] = []; const triangles: number[] = []; const colors: Vec3[] = []; const payloads: Array<Exclude<Payload, 'structure'>> = [];
  for (const entry of networkEntries) {
    if ('sheet' in entry) {
      const cellSize = tile / entry.sheet.resolution;
      for (const cell of entry.sheet.cells) {
        const x0 = cell.ix * cellSize; const x1 = x0 + cellSize;
        const z0 = cell.iz * cellSize; const z1 = z0 + cellSize;
        const base = positions.length / 3;
        positions.push(
          x0, entry.sheet.height, z0,
          x1, entry.sheet.height, z0,
          x1, entry.sheet.height, z1,
          x0, entry.sheet.height, z1,
        );
        triangles.push(base, base + 2, base + 1, base, base + 3, base + 2);
        colors.push(cell.color, cell.color); payloads.push(cell.payload, cell.payload);
      }
    } else {
      const { beta, hMin, hMax } = entry.network.candidate;
      for (const edge of entry.network.edges) {
        const [a, b] = unwrapEdge(edge.a, edge.b, tile);
        const base = positions.length / 3;
        positions.push(
          a[0] + beta[0] * hMin, hMin, a[1] + beta[1] * hMin,
          b[0] + beta[0] * hMin, hMin, b[1] + beta[1] * hMin,
          b[0] + beta[0] * hMax, hMax, b[1] + beta[1] * hMax,
          a[0] + beta[0] * hMax, hMax, a[1] + beta[1] * hMax,
        );
        triangles.push(base, base + 1, base + 2, base, base + 2, base + 3);
        colors.push(edge.color, edge.color); payloads.push(entry.network.payload, entry.network.payload);
      }
    }
  }
  if (triangles.length === 0) throw new Error('compiled core is empty');
  const mesh: IndexedMesh = { positions, indices: triangles, colors: positions.map(() => 0), normals: positions.map(() => 0) };
  const geometry = decodedGeometry(mesh, { originX: 0, originZ: 0, sizeX: tile, sizeZ: tile, topH }, 2);
  return { geometry, colors, payloads };
}

function barycentricColor(mesh: IndexedMesh, geometry: DecodedOwnedProfileGeometry, hit: OriginAwareTruthHit, origin: Vec3, direction: Vec3): Vec3 {
  const point: Vec3 = add(origin, scale(direction, hit.t));
  const local: Vec3 = [point[0] - hit.copyX * geometry.tileSizeX, point[1], point[2] - hit.copyZ * geometry.tileSizeZ];
  const triangle = hit.triangleId * 3;
  const ids = [geometry.triangles[triangle]!, geometry.triangles[triangle + 1]!, geometry.triangles[triangle + 2]!] as const;
  const p = ids.map((id) => {
    const offset = id * 3; return [geometry.positions[offset]!, geometry.positions[offset + 1]!, geometry.positions[offset + 2]!] as Vec3;
  });
  const e0: Vec3 = [p[1]![0] - p[0]![0], p[1]![1] - p[0]![1], p[1]![2] - p[0]![2]];
  const e1: Vec3 = [p[2]![0] - p[0]![0], p[2]![1] - p[0]![1], p[2]![2] - p[0]![2]];
  const q: Vec3 = [local[0] - p[0]![0], local[1] - p[0]![1], local[2] - p[0]![2]];
  const d00 = e0[0] ** 2 + e0[1] ** 2 + e0[2] ** 2; const d01 = e0[0] * e1[0] + e0[1] * e1[1] + e0[2] * e1[2]; const d11 = e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2;
  const d20 = q[0] * e0[0] + q[1] * e0[1] + q[2] * e0[2]; const d21 = q[0] * e1[0] + q[1] * e1[1] + q[2] * e1[2];
  const denominator = d00 * d11 - d01 * d01; const u = denominator ? (d11 * d20 - d01 * d21) / denominator : 0; const v = denominator ? (d00 * d21 - d01 * d20) / denominator : 0;
  const weights = [1 - u - v, u, v]; const colors = ids.map((id) => meshVertexColor(mesh, id));
  return [0, 1, 2].map((channel) => clamp(weights.reduce((sum, weight, index) => sum + weight * colors[index]![channel]!, 0), 0, 1)) as unknown as Vec3;
}
function traceSource(mesh: IndexedMesh, geometry: DecodedOwnedProfileGeometry, bvh: TriangleBvh, payloads: readonly Payload[], origin: Vec3, direction: Vec3, finiteCutoff = Number.POSITIVE_INFINITY): SurfaceHit | null {
  const horizon = Number.isFinite(finiteCutoff) ? finiteCutoff
    : direction[1] < -1e-14 ? (origin[1] - geometry.bounds.min[1] + 1e-8) / -direction[1]
      : direction[1] > 1e-14 ? (geometry.bounds.max[1] - origin[1] + 1e-8) / direction[1]
        : 0;
  if (!(horizon > 0)) return null;
  const hit = periodicNearestSuccessor(geometry, bvh, origin, direction, horizon, 0);
  return hit ? { t: hit.t, point: add(origin, scale(direction, hit.t)), color: barycentricColor(mesh, geometry, hit, origin, direction), payload: payloads[hit.triangleId]!, triangleId: hit.triangleId } : null;
}
function traceCore(geometry: DecodedOwnedProfileGeometry, bvh: TriangleBvh, colors: readonly Vec3[], payloads: ReadonlyArray<Exclude<Payload, 'structure'>>, origin: Vec3, direction: Vec3, finiteCutoff = Number.POSITIVE_INFINITY): CoreHit | null {
  const horizon = Number.isFinite(finiteCutoff) ? finiteCutoff
    : direction[1] < -1e-14 ? (origin[1] - geometry.bounds.min[1] + 1e-8) / -direction[1]
      : direction[1] > 1e-14 ? (geometry.bounds.max[1] - origin[1] + 1e-8) / direction[1]
        : 0;
  if (!(horizon > 0)) return null;
  const hit = periodicNearestSuccessor(geometry, bvh, origin, direction, horizon, 0);
  return hit ? { t: hit.t, point: add(origin, scale(direction, hit.t)), color: colors[hit.triangleId]!, payload: payloads[hit.triangleId]!, triangleId: hit.triangleId } : null;
}

interface FitRay { origin: Vec3; direction: Vec3; cutoff: number; coreWithin: CoreHit | null; }
interface FitRow { rays: FitRay[]; referenceAlpha: number; referencePremul: Vec3; unavoidableEarlyCoreAlpha: number; basis?: Float64Array[]; }

function solveLeastSquares(design: readonly Float64Array[], target: readonly number[]): Float64Array {
  const columns = design[0]!.length; const matrix = Array.from({ length: columns }, () => new Float64Array(columns + 1));
  for (let row = 0; row < design.length; row++) for (let a = 0; a < columns; a++) {
    matrix[a]![columns] += design[row]![a]! * target[row]!;
    for (let b = 0; b < columns; b++) matrix[a]![b] += design[row]![a]! * design[row]![b]!;
  }
  const diagonal = Math.max(...matrix.map((row, index) => row[index]!));
  for (let index = 0; index < columns; index++) matrix[index]![index] += Math.max(1e-12, diagonal * 1e-9);
  for (let column = 0; column < columns; column++) {
    let pivot = column;
    for (let row = column + 1; row < columns; row++) if (Math.abs(matrix[row]![column]!) > Math.abs(matrix[pivot]![column]!)) pivot = row;
    [matrix[column], matrix[pivot]] = [matrix[pivot]!, matrix[column]!];
    const divisor = matrix[column]![column]!;
    if (Math.abs(divisor) < 1e-20) continue;
    for (let item = column; item <= columns; item++) matrix[column]![item] /= divisor;
    for (let row = 0; row < columns; row++) {
      if (row === column) continue; const factor = matrix[row]![column]!;
      for (let item = column; item <= columns; item++) matrix[row]![item] -= factor * matrix[column]![item]!;
    }
  }
  return Float64Array.from(matrix.map((row) => row[columns]!));
}

function fitResidual(
  mesh: IndexedMesh, geometry: DecodedOwnedProfileGeometry, bvh: TriangleBvh, payloads: readonly Payload[],
  coreGeometry: DecodedOwnedProfileGeometry, coreBvh: TriangleBvh, coreColors: readonly Vec3[], corePayloads: ReadonlyArray<Exclude<Payload, 'structure'>>,
): { model: ResidualModel; report: Record<string, unknown> } {
  const hMin = geometry.bounds.min[1]; const hMax = geometry.topH; const shell: ResidualModel = { hMin, hMax, a0: 0, b0: [0, 0, 0], modes: [], opacityModeScale: 1, colorModeScales: [1, 1, 1], kappaMinimum: 0 };
  const sampleCount = FIT_SUBPIXEL_SIDE ** 2; const rows: FitRow[] = []; let opaqueRows = 0; let maximumEarlyCoreLowerBound = 0;
  const centerRay = (spec: CameraSpec, phaseX: number, phaseZ: number, sx: number, sy: number): { origin: Vec3; direction: Vec3 } => {
    const basis = cameraBasis(spec); const origin: Vec3 = [phaseX - basis.forward[0] * spec.distanceMetres, hMax - basis.forward[1] * spec.distanceMetres, phaseZ - basis.forward[2] * spec.distanceMetres];
    const x = (sx - 0.5) * 2 * Math.tan(PIXEL_ANGLE * 0.5); const y = (0.5 - sy) * 2 * Math.tan(PIXEL_ANGLE * 0.5);
    return { origin, direction: normalized(add(basis.forward, add(scale(basis.right, x), scale(basis.up, y)))) };
  };
  for (const distanceMetres of DISTANCES) for (const view of VIEW_DIRECTIONS) {
    const spec: CameraSpec = { ...view, key: `fit-${view.key}-d${distanceMetres}`, distanceMetres };
    for (let pz = 0; pz < FIT_PHASE_SIDE; pz++) for (let px = 0; px < FIT_PHASE_SIDE; px++) {
      const phaseX = geometry.tileOriginX + (px + 0.5) / FIT_PHASE_SIDE * geometry.tileSizeX;
      const phaseZ = geometry.tileOriginZ + (pz + 0.5) / FIT_PHASE_SIDE * geometry.tileSizeZ;
      const traced: Array<{ origin: Vec3; direction: Vec3; source: SurfaceHit | null; core: CoreHit | null; exit: number }> = [];
      const events: number[] = [];
      for (let sy = 0; sy < FIT_SUBPIXEL_SIDE; sy++) for (let sx = 0; sx < FIT_SUBPIXEL_SIDE; sx++) {
        const ray = centerRay(spec, phaseX, phaseZ, (sx + 0.5) / FIT_SUBPIXEL_SIDE, (sy + 0.5) / FIT_SUBPIXEL_SIDE);
        const exit = (hMin - ray.origin[1]) / ray.direction[1]; const source = traceSource(mesh, geometry, bvh, payloads, ray.origin, ray.direction); const coreHit = traceCore(coreGeometry, coreBvh, coreColors, corePayloads, ray.origin, ray.direction);
        traced.push({ ...ray, source, core: coreHit, exit });
        if (source) events.push(source.t); if (coreHit) events.push(coreHit.t);
      }
      events.sort((a, b) => a - b);
      const cutoffs = new Set<number>([Math.max(...traced.map((ray) => ray.exit))]);
      for (const fraction of [0, 0.25, 0.5, 0.75, 1]) if (events.length) {
        const event = events[Math.floor((events.length - 1) * fraction)]!; cutoffs.add(Math.max(1e-8, event - 1e-5)); cutoffs.add(event + 1e-5);
      }
      for (const cutoff of [...cutoffs].sort((a, b) => a - b)) {
        let sourceBefore = 0; let coreBefore = 0; const premul = [0, 0, 0]; const fitRays: FitRay[] = [];
        for (const ray of traced) {
          if (ray.source && ray.source.t <= cutoff) { sourceBefore++; for (let channel = 0; channel < 3; channel++) premul[channel] += ray.source.color[channel]! / sampleCount; }
          const coreWithin = ray.core && ray.core.t <= cutoff ? ray.core : null; if (coreWithin) coreBefore++;
          fitRays.push({ origin: ray.origin, direction: ray.direction, cutoff: Math.min(cutoff, ray.exit, coreWithin?.t ?? Infinity), coreWithin });
        }
        const referenceAlpha = sourceBefore / sampleCount; const unavoidableEarlyCoreAlpha = Math.max(0, (coreBefore - sourceBefore) / sampleCount);
        maximumEarlyCoreLowerBound = Math.max(maximumEarlyCoreLowerBound, unavoidableEarlyCoreAlpha); if (referenceAlpha === 1) opaqueRows++;
        rows.push({ rays: fitRays, referenceAlpha, referencePremul: premul as unknown as Vec3, unavoidableEarlyCoreAlpha });
      }
    }
  }

  const integrate = (ray: FitRay, k: Vec3): Complex => {
    const interval = residualInterval(shell, ray.origin, ray.direction, ray.cutoff);
    return interval ? weightedModeIntegral(shell, ray.origin, ray.direction, interval[0], interval[1], k) : { re: 0, im: 0 };
  };
  const candidates: Array<{ mx: number; my: number; mz: number; k: Vec3; score: number }> = [];
  for (let my = 0; my <= 3; my++) for (let mz = -3; mz <= 3; mz++) for (let mx = -3; mx <= 3; mx++) {
    if (my === 0 && (mx < 0 || (mx === 0 && mz <= 0))) continue;
    const k: Vec3 = [TAU * mx / geometry.tileSizeX, Math.PI * my / (hMax - hMin), TAU * mz / geometry.tileSizeZ];
    const columns = rows.map((row) => {
      const value = row.rays.reduce((sum, ray) => cadd(sum, integrate(ray, k)), { re: 0, im: 0 });
      return [2 * value.re / sampleCount, -2 * value.im / sampleCount] as const;
    });
    let score = 0;
    for (const target of [rows.map((row) => row.referenceAlpha), ...[0, 1, 2].map((channel) => rows.map((row) => row.referencePremul[channel]!))]) {
      const mean = target.reduce((sum, value) => sum + value, 0) / target.length; const energy = target.reduce((sum, value) => sum + (value - mean) ** 2, 0) + 1e-12;
      for (let component = 0; component < 2; component++) {
        const norm = columns.reduce((sum, value) => sum + value[component]! ** 2, 0) + 1e-12;
        const correlation = columns.reduce((sum, value, index) => sum + value[component]! * (target[index]! - mean), 0);
        score += correlation * correlation / (norm * energy);
      }
    }
    candidates.push({ mx, my, mz, k, score });
  }
  candidates.sort((a, b) => b.score - a.score || a.my - b.my || a.mx - b.mx || a.mz - b.mz);
  const selected = candidates.slice(0, MODE_LIMIT_PER_LAYER);
  for (const row of rows) row.basis = row.rays.map((ray) => {
    const basis = new Float64Array(1 + 2 * MODE_LIMIT_PER_LAYER); basis[0] = integrate(ray, [0, 0, 0]).re;
    for (let mode = 0; mode < selected.length; mode++) { const value = integrate(ray, selected[mode]!.k); basis[1 + 2 * mode] = 2 * value.re; basis[2 + 2 * mode] = -2 * value.im; }
    return basis;
  });
  const rowDesign = rows.map((row) => {
    const design = new Float64Array(1 + 2 * MODE_LIMIT_PER_LAYER); for (const basis of row.basis!) for (let column = 0; column < design.length; column++) design[column] += basis[column]! / sampleCount; return design;
  });
  const opacityProxyTarget = rows.map((row) => row.referenceAlpha >= 1 ? 8 : -Math.log1p(-row.referenceAlpha));
  const initial = solveLeastSquares(rowDesign, opacityProxyTarget); initial[0] = Math.max(1e-5, initial[0]!);
  const parameters = Float64Array.from(initial); const adamM = new Float64Array(parameters.length); const adamV = new Float64Array(parameters.length);
  const projectPositive = (): void => {
    const amplitude = 2 * selected.reduce((sum, _, index) => sum + Math.hypot(parameters[1 + 2 * index]!, parameters[2 + 2 * index]!), 0);
    parameters[0] = Math.max(parameters[0]!, amplitude + 1e-5);
  };
  projectPositive();
  for (let step = 1; step <= 320; step++) {
    const gradient = new Float64Array(parameters.length);
    for (const row of rows) {
      let predicted = 0;
      const derivatives = new Float64Array(parameters.length);
      for (let sample = 0; sample < row.rays.length; sample++) {
        const ray = row.rays[sample]!;
        if (ray.coreWithin) { predicted += 1 / sampleCount; continue; }
        const basis = row.basis![sample]!; let tau = 0; for (let column = 0; column < parameters.length; column++) tau += basis[column]! * parameters[column]!;
        tau = Math.max(0, tau); const transmission = Math.exp(-tau); predicted += (1 - transmission) / sampleCount;
        for (let column = 0; column < parameters.length; column++) derivatives[column] += transmission * basis[column]! / sampleCount;
      }
      const error = predicted - row.referenceAlpha; for (let column = 0; column < parameters.length; column++) gradient[column] += 2 * error * derivatives[column] / rows.length;
    }
    const learningRate = 0.035;
    for (let column = 0; column < parameters.length; column++) {
      adamM[column] = 0.9 * adamM[column]! + 0.1 * gradient[column]!; adamV[column] = 0.999 * adamV[column]! + 0.001 * gradient[column]! ** 2;
      parameters[column] -= learningRate * (adamM[column]! / (1 - 0.9 ** step)) / (Math.sqrt(adamV[column]! / (1 - 0.999 ** step)) + 1e-8);
    }
    projectPositive();
  }
  const rawC = selected.map((_, index) => ({ re: parameters[1 + 2 * index]!, im: parameters[2 + 2 * index]! }));
  const a0 = parameters[0]!; const rawAmplitude = 2 * rawC.reduce((sum, value) => sum + cabs(value), 0); const opacityModeScale = rawAmplitude > 0 ? Math.min(1, 0.98 * a0 / rawAmplitude) : 1;
  const c = rawC.map((value) => cscale(value, opacityModeScale)); const kappaMinimum = a0 - 2 * c.reduce((sum, value) => sum + cabs(value), 0);

  const fittedColourParameters: Float64Array[] = [];
  for (let channel = 0; channel < 3; channel++) {
    const design: Float64Array[] = []; const target: number[] = [];
    for (const row of rows) {
      const coefficients = new Float64Array(parameters.length); let coreOffset = 0;
      for (let sample = 0; sample < row.rays.length; sample++) {
        const basis = row.basis![sample]!; let tau = a0 * basis[0]!;
        for (let mode = 0; mode < c.length; mode++) tau += basis[1 + 2 * mode]! * c[mode]!.re + basis[2 + 2 * mode]! * c[mode]!.im;
        const transmission = Math.exp(-Math.max(0, tau)); const factor = tau > 1e-10 ? (1 - transmission) / tau : 1;
        for (let column = 0; column < coefficients.length; column++) coefficients[column] += factor * basis[column]! / sampleCount;
        if (row.rays[sample]!.coreWithin) coreOffset += transmission * row.rays[sample]!.coreWithin!.color[channel]! / sampleCount;
      }
      design.push(coefficients); target.push(row.referencePremul[channel]! - coreOffset);
    }
    fittedColourParameters.push(solveLeastSquares(design, target));
  }
  const b0: number[] = []; const colorModeScales: number[] = []; const dByChannel: Complex[][] = [[], [], []];
  for (let channel = 0; channel < 3; channel++) {
    const fitted = fittedColourParameters[channel]!; const mu = clamp(fitted[0]! / a0, 0, 1); b0[channel] = mu * a0;
    const residual = c.map((value, mode) => cadd({ re: fitted[1 + 2 * mode]!, im: fitted[2 + 2 * mode]! }, cscale(value, -mu)));
    const residualAmplitude = 2 * residual.reduce((sum, value) => sum + cabs(value), 0); const safeScale = residualAmplitude > 0 ? Math.min(1, Math.min(mu, 1 - mu) * kappaMinimum / residualAmplitude) : 1;
    colorModeScales[channel] = safeScale; dByChannel[channel] = c.map((value, mode) => cadd(cscale(value, mu), cscale(residual[mode]!, safeScale)));
  }
  const modes: Mode[] = selected.map((mode, index) => ({ mx: mode.mx, my: mode.my, mz: mode.mz, k: mode.k, c: c[index]!, d: [dByChannel[0]![index]!, dByChannel[1]![index]!, dByChannel[2]![index]!] }));
  const model: ResidualModel = { hMin, hMax, a0, b0: b0 as unknown as Vec3, modes, opacityModeScale, colorModeScales: colorModeScales as unknown as Vec3, kappaMinimum };
  const alphaErrors: number[] = []; const rgbErrors: number[] = []; let tauMaximum = 0;
  for (const row of rows) {
    let predictedAlpha = 0; const predictedPremul = [0, 0, 0];
    for (const ray of row.rays) {
      const soft = residualIntegral(model, ray.origin, ray.direction, ray.cutoff); tauMaximum = Math.max(tauMaximum, soft.tau); const transmission = Math.exp(-soft.tau); const alpha = 1 - transmission;
      predictedAlpha += (ray.coreWithin ? 1 : alpha) / sampleCount;
      for (let channel = 0; channel < 3; channel++) predictedPremul[channel] += ((soft.tau > 1e-12 ? soft.J[channel]! / soft.tau * alpha : 0) + (ray.coreWithin ? transmission * ray.coreWithin.color[channel]! : 0)) / sampleCount;
    }
    alphaErrors.push(Math.abs(predictedAlpha - row.referenceAlpha)); rgbErrors.push(Math.max(...row.referencePremul.map((value, channel) => Math.abs(value - predictedPremul[channel]!))));
  }
  return { model, report: {
    fitDomain: { phaseSide: FIT_PHASE_SIDE, distancesMetres: DISTANCES, views: VIEW_DIRECTIONS, physicallyReachablePrefixCutoffs: 'immediately before/after the 0/25/50/75/100% source/core event order statistics plus full cover exit (fit subset, not an all-event certificate)', rows: rows.length, subpixelSamples: sampleCount },
    modeSelection: 'joint opacity/RGB correlation over vertical-capable reciprocal candidates, followed by direct final-hybrid opacity optimization; heuristic K2 per layer, not claimed globally optimal over the joint K2+K2 frequency allocation',
    modes: modes.map((mode) => ({ mx: mode.mx, my: mode.my, mz: mode.mz, opacityMagnitude: cabs(mode.c), colourMagnitudes: mode.d.map(cabs) })),
    a0, b0: model.b0, opacityModeScale, colorModeScales: model.colorModeScales, kappaMinimum,
    positivity: { kappaNonnegativeByAmplitudeBound: kappaMinimum >= -1e-12, colourDensityBound: 'j=mu*kappa+bounded residual, 0<=j<=kappa by global amplitude bound' },
    noCoverageClamp: true, exactOpaqueReferenceRows: opaqueRows, integratedTauMaximum: tauMaximum,
    eventSubsetCoverageExcessLowerBound: maximumEarlyCoreLowerBound,
    fittedPrefixAlphaError: quantiles(alphaErrors), fittedPrefixRgbError: quantiles(rgbErrors),
  } };
}

function sincDerivatives(z: number): readonly [number, number, number] {
  const absolute = Math.abs(z);
  if (absolute < 1e-3) {
    const z2 = z * z; const z4 = z2 * z2; const z6 = z4 * z2;
    return [1 - z2 / 6 + z4 / 120 - z6 / 5040, -z / 3 + z * z2 / 30 - z * z4 / 840 + z * z6 / 45360, -1 / 3 + z2 / 10 - z4 / 168 + z6 / 6480];
  }
  return [Math.sin(z) / z, (z * Math.cos(z) - Math.sin(z)) / (z * z), (-z * z * Math.sin(z) - 2 * z * Math.cos(z) + 2 * Math.sin(z)) / (z * z * z)];
}
function weightedModeIntegral(model: ResidualModel, origin: Vec3, direction: Vec3, a: number, b: number, k: Vec3): Complex {
  const length = b - a; const half = length * 0.5; const middle = (a + b) * 0.5;
  const yMiddle = origin[1] + direction[1] * middle; const u = (yMiddle - model.hMin) / (model.hMax - model.hMin); const nu = direction[1] / (model.hMax - model.hMin);
  const p0 = 4 * (u - u * u); const p1 = 4 * nu * (1 - 2 * u); const p2 = -4 * nu * nu;
  const lambda = k[0] * direction[0] + k[1] * direction[1] + k[2] * direction[2]; const z = lambda * half; const [s0, s1, s2] = sincDerivatives(z);
  const m0: Complex = { re: length * s0, im: 0 }; const m1: Complex = { re: 0, im: -2 * half * half * s1 }; const m2: Complex = { re: -2 * half ** 3 * s2, im: 0 };
  const moment = cadd(cadd(cscale(m0, p0), cscale(m1, p1)), cscale(m2, p2));
  const point = add(origin, scale(direction, middle)); return cmul(cis(k[0] * point[0] + k[1] * point[1] + k[2] * point[2]), moment);
}
function residualIntegral(model: ResidualModel, origin: Vec3, direction: Vec3, cutoff: number): { tau: number; J: Vec3 } {
  const interval = residualInterval(model, origin, direction, cutoff);
  if (!interval) return { tau: 0, J: [0, 0, 0] };
  const [entry, exit] = interval;
  const constantIntegral = weightedModeIntegral(model, origin, direction, entry, exit, [0, 0, 0]).re;
  let tau = model.a0 * constantIntegral; const J = model.b0.map((value) => value * constantIntegral);
  for (const mode of model.modes) {
    const integral = weightedModeIntegral(model, origin, direction, entry, exit, mode.k);
    tau += 2 * cmul(mode.c, integral).re;
    for (let channel = 0; channel < 3; channel++) J[channel] += 2 * cmul(mode.d[channel]!, integral).re;
  }
  return { tau: Math.max(0, tau), J: J.map((value) => Math.max(0, value)) as unknown as Vec3 };
}

/** Exact intersection of the forward ray segment [0, cutoff] with the
 * residual's closed vertical slab.  Horizontal rays are deliberately valid:
 * if their height lies in the slab, their interval is the finite scene
 * cutoff, not an invented near-horizontal epsilon. */
function residualInterval(model: Pick<ResidualModel, 'hMin' | 'hMax'>, origin: Vec3, direction: Vec3, cutoff: number): readonly [number, number] | null {
  if (!(cutoff > 0)) return null;
  if (Math.abs(direction[1]) <= 1e-14) {
    if (origin[1] < model.hMin - EPSILON || origin[1] > model.hMax + EPSILON || !Number.isFinite(cutoff)) return null;
    return [0, cutoff];
  }
  const atMin = (model.hMin - origin[1]) / direction[1];
  const atMax = (model.hMax - origin[1]) / direction[1];
  const entry = Math.max(0, Math.min(atMin, atMax));
  const exit = Math.min(cutoff, Math.max(atMin, atMax));
  return exit > entry + EPSILON ? [entry, exit] : null;
}

function cameraBasis(spec: CameraSpec): { forward: Vec3; right: Vec3; up: Vec3; origin: Vec3 } {
  const azimuth = spec.azimuthDegrees * Math.PI / 180; const elevation = spec.elevationDegrees * Math.PI / 180;
  const forward: Vec3 = elevation >= Math.PI / 2 - 1e-12 ? [0, -1, 0] : [Math.cos(elevation) * Math.cos(azimuth), -Math.sin(elevation), Math.cos(elevation) * Math.sin(azimuth)];
  const right: Vec3 = [-Math.sin(azimuth), 0, Math.cos(azimuth)]; const up: Vec3 = [Math.sin(elevation) * Math.cos(azimuth), Math.cos(elevation), Math.sin(elevation) * Math.sin(azimuth)];
  const target: Vec3 = [0.38196601125, 0, 0.61803398875];
  return { forward, right, up, origin: [target[0] - forward[0] * spec.distanceMetres, target[1] - forward[1] * spec.distanceMetres, target[2] - forward[2] * spec.distanceMetres] };
}
function pixelRay(spec: CameraSpec, pixelX: number, pixelY: number, subX: number, subY: number, topH: number, tile: number): { origin: Vec3; direction: Vec3 } {
  const basis = cameraBasis(spec); const targetOffset: Vec3 = [tile * 0.38196601125 - 0.38196601125, topH, tile * 0.61803398875 - 0.61803398875];
  const origin = add(basis.origin, targetOffset);
  const x = (pixelX + subX - WIDTH * 0.5) * 2 * Math.tan(PIXEL_ANGLE * 0.5); const y = (HEIGHT * 0.5 - pixelY - subY) * 2 * Math.tan(PIXEL_ANGLE * 0.5);
  return { origin, direction: normalized(add(basis.forward, add(scale(basis.right, x), scale(basis.up, y)))) };
}

function phaseBundleRays(spec: CameraSpec, phaseX: number, phaseZ: number, topH: number, subpixelSide: number, cameraTranslationMetres = 0): Array<{ origin: Vec3; direction: Vec3 }> {
  const basis = cameraBasis(spec);
  const target: Vec3 = [phaseX, topH, phaseZ];
  const origin = add(add(target, scale(basis.forward, -spec.distanceMetres)), scale(basis.right, cameraTranslationMetres));
  const rays: Array<{ origin: Vec3; direction: Vec3 }> = [];
  for (let sy = 0; sy < subpixelSide; sy++) for (let sx = 0; sx < subpixelSide; sx++) {
    const x = ((sx + 0.5) / subpixelSide - 0.5) * 2 * Math.tan(PIXEL_ANGLE * 0.5);
    const y = (0.5 - (sy + 0.5) / subpixelSide) * 2 * Math.tan(PIXEL_ANGLE * 0.5);
    rays.push({ origin, direction: normalized(add(basis.forward, add(scale(basis.right, x), scale(basis.up, y)))) });
  }
  return rays;
}

function evaluateRayBundle(
  rays: readonly { origin: Vec3; direction: Vec3; cutoff?: number }[],
  sourceMesh: IndexedMesh, sourceGeometry: DecodedOwnedProfileGeometry, sourceBvh: TriangleBvh, sourcePayload: readonly Payload[],
  coreGeometry: DecodedOwnedProfileGeometry, coreBvh: TriangleBvh, coreColors: readonly Vec3[], corePayloads: ReadonlyArray<Exclude<Payload, 'structure'>>,
  residual: ResidualModel, layers: readonly LayerTransform[],
): FilteredPixel {
  const sampleCount = rays.length; let truthAlpha = 0; const truthPremul = [0, 0, 0]; const truthPayload: Record<Payload, number> = { foliage: 0, 'reproductive-purple': 0, 'reproductive-cream': 0, structure: 0 };
  let candidateAlpha = 0; const candidatePremul = [0, 0, 0]; let coreHitFraction = 0; let coreMatchedFraction = 0; let coreFalseForegroundFraction = 0;
  let coreFalseCoverageFraction = 0; let coreEarlyDepthExcessFraction = 0; const firstHitErrors: number[] = [];
  for (const ray of rays) {
    const finiteCutoff = ray.cutoff ?? Number.POSITIVE_INFINITY;
    const sourceHits = layers.map((layer) => {
      const local = transformRay(ray.origin, ray.direction, layer);
      return traceSource(sourceMesh, sourceGeometry, sourceBvh, sourcePayload, local.origin, local.direction, finiteCutoff);
    });
    const coreHits = layers.map((layer) => {
      const local = transformRay(ray.origin, ray.direction, layer);
      return traceCore(coreGeometry, coreBvh, coreColors, corePayloads, local.origin, local.direction, finiteCutoff);
    });
    const truth = sourceHits.filter((hit): hit is SurfaceHit => hit !== null).sort((a, b) => a.t - b.t)[0] ?? null;
    const core = coreHits.filter((hit): hit is CoreHit => hit !== null).sort((a, b) => a.t - b.t)[0] ?? null;
    if (truth) {
      truthAlpha += 1 / sampleCount; truthPayload[truth.payload] += 1 / sampleCount;
      for (let channel = 0; channel < 3; channel++) truthPremul[channel] += truth.color[channel]! / sampleCount;
    }
    if (core) {
      coreHitFraction += 1 / sampleCount;
      if (truth) {
        const error = Math.abs(core.t - truth.t); firstHitErrors.push(error);
        if (error <= 0.01 && core.payload === truth.payload) coreMatchedFraction += 1 / sampleCount;
        if (core.t < truth.t - 0.01) coreFalseForegroundFraction += 1 / sampleCount;
        if (core.t < truth.t - COMPOSITING_DEPTH_TOLERANCE_METRES) coreEarlyDepthExcessFraction += 1 / sampleCount;
      } else { coreFalseForegroundFraction += 1 / sampleCount; coreFalseCoverageFraction += 1 / sampleCount; }
    }
    const cutoff = Math.min(finiteCutoff, core?.t ?? Number.POSITIVE_INFINITY);
    let totalTau = 0; const totalJ = [0, 0, 0];
    for (const layer of layers) {
      const local = transformRay(ray.origin, ray.direction, layer); const soft = residualIntegral(residual, local.origin, local.direction, cutoff);
      totalTau += soft.tau; for (let channel = 0; channel < 3; channel++) totalJ[channel] += soft.J[channel]!;
    }
    const softAlpha = 1 - Math.exp(-totalTau);
    const softColor = totalTau > 1e-12 ? totalJ.map((value) => value / totalTau) : [0, 0, 0];
    const alpha = core ? 1 : softAlpha; candidateAlpha += alpha / sampleCount;
    for (let channel = 0; channel < 3; channel++) {
      const value = softColor[channel]! * softAlpha + (core ? Math.exp(-totalTau) * core.color[channel]! : 0);
      candidatePremul[channel] += value / sampleCount;
    }
  }
  return { truthAlpha, truthPremul: truthPremul as unknown as Vec3, truthPayload, candidateAlpha, candidatePremul: candidatePremul as unknown as Vec3, coreHitFraction, coreMatchedFraction, coreFalseForegroundFraction, coreFalseCoverageFraction, coreEarlyDepthExcessFraction, firstHitErrors };
}

function evaluatePixel(
  spec: CameraSpec, x: number, y: number,
  sourceMesh: IndexedMesh, sourceGeometry: DecodedOwnedProfileGeometry, sourceBvh: TriangleBvh, sourcePayload: readonly Payload[],
  coreGeometry: DecodedOwnedProfileGeometry, coreBvh: TriangleBvh, coreColors: readonly Vec3[], corePayloads: ReadonlyArray<Exclude<Payload, 'structure'>>,
  residual: ResidualModel, layers: readonly LayerTransform[] = [LAYER_TRANSFORMS[0]!],
): FilteredPixel {
  const rays: Array<{ origin: Vec3; direction: Vec3 }> = [];
  for (let sy = 0; sy < SUBPIXEL_SIDE; sy++) for (let sx = 0; sx < SUBPIXEL_SIDE; sx++) rays.push(pixelRay(spec, x, y, (sx + 0.5) / SUBPIXEL_SIDE, (sy + 0.5) / SUBPIXEL_SIDE, sourceGeometry.topH, sourceGeometry.tileSizeX));
  return evaluateRayBundle(rays, sourceMesh, sourceGeometry, sourceBvh, sourcePayload, coreGeometry, coreBvh, coreColors, corePayloads, residual, layers);
}

function connectedWrongRegions(pixels: readonly FilteredPixel[], gridWidth = WIDTH, gridHeight = HEIGHT): number[] {
  const wrong = pixels.map((pixel) => (pixel.truthAlpha >= 0.5) !== (pixel.candidateAlpha >= 0.5)); const visited = new Uint8Array(wrong.length); const sizes: number[] = [];
  for (let start = 0; start < wrong.length; start++) {
    if (!wrong[start] || visited[start]) continue;
    let size = 0; const queue = [start]; visited[start] = 1;
    while (queue.length) {
      const current = queue.pop()!; size++; const x = current % gridWidth; const y = Math.floor(current / gridWidth);
      for (const [nx, ny] of [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]) {
        if (nx < 0 || nx >= gridWidth || ny < 0 || ny >= gridHeight) continue;
        const next = ny * gridWidth + nx; if (wrong[next] && !visited[next]) { visited[next] = 1; queue.push(next); }
      }
    }
    sizes.push(size);
  }
  return sizes;
}
function summarizePixels(spec: CameraSpec, pixels: readonly FilteredPixel[], gridWidth = WIDTH, gridHeight = HEIGHT): Record<string, unknown> {
  let intersection = 0; let union = 0; const rgbErrors: number[] = []; const alphaErrors: number[] = []; const hitErrors = pixels.flatMap((pixel) => pixel.firstHitErrors);
  const byPayload: Record<string, { referenceMass: number; candidateAlphaOnReference: number; rgbErrors: number[] }> = {};
  for (const payload of ['foliage', 'reproductive-purple', 'reproductive-cream', 'structure'] as Payload[]) byPayload[payload] = { referenceMass: 0, candidateAlphaOnReference: 0, rgbErrors: [] };
  for (const pixel of pixels) {
    const truth = pixel.truthAlpha >= 0.5; const candidate = pixel.candidateAlpha >= 0.5; intersection += Number(truth && candidate); union += Number(truth || candidate);
    alphaErrors.push(Math.abs(pixel.truthAlpha - pixel.candidateAlpha)); rgbErrors.push(Math.max(...pixel.truthPremul.map((value, channel) => Math.abs(value - pixel.candidatePremul[channel]!))));
    for (const payload of Object.keys(byPayload) as Payload[]) if (pixel.truthPayload[payload] > 0) {
      const entry = byPayload[payload]!; entry.referenceMass += pixel.truthPayload[payload]; entry.candidateAlphaOnReference += pixel.candidateAlpha * pixel.truthPayload[payload]; entry.rgbErrors.push(Math.max(...pixel.truthPremul.map((value, channel) => Math.abs(value - pixel.candidatePremul[channel]!))));
    }
  }
  const regions = connectedWrongRegions(pixels, gridWidth, gridHeight);
  return {
    spec,
    referenceCoverageMean: pixels.reduce((sum, pixel) => sum + pixel.truthAlpha, 0) / pixels.length,
    candidateCoverageMean: pixels.reduce((sum, pixel) => sum + pixel.candidateAlpha, 0) / pixels.length,
    silhouetteIoU: union ? intersection / union : 1,
    alphaError: quantiles(alphaErrors), rgbMaxChannelError: quantiles(rgbErrors),
    connectedWrongRegions: { count: regions.length, maximumPixels: Math.max(0, ...regions), maximumFraction: Math.max(0, ...regions) / pixels.length },
    core: {
      hitFraction: pixels.reduce((sum, pixel) => sum + pixel.coreHitFraction, 0) / pixels.length,
      matchedFirstVisibleFraction: pixels.reduce((sum, pixel) => sum + pixel.coreMatchedFraction, 0) / pixels.length,
      falseForegroundFraction: pixels.reduce((sum, pixel) => sum + pixel.coreFalseForegroundFraction, 0) / pixels.length,
      sourceMissCoreHitFalseCoverageFraction: pixels.reduce((sum, pixel) => sum + pixel.coreFalseCoverageFraction, 0) / pixels.length,
      earlyDepthExcessBeyondToleranceFraction: pixels.reduce((sum, pixel) => sum + pixel.coreEarlyDepthExcessFraction, 0) / pixels.length,
      compositingDepthToleranceMetres: COMPOSITING_DEPTH_TOLERANCE_METRES,
      firstHitDistanceErrorMetres: quantiles(hitErrors),
    },
    byPayload: Object.fromEntries(Object.entries(byPayload).map(([payload, value]) => [payload, {
      referenceMass: value.referenceMass,
      candidateAlphaOnReferenceMean: value.referenceMass ? value.candidateAlphaOnReference / value.referenceMass : null,
      rgbMaxChannelError: quantiles(value.rgbErrors),
    }])),
    filtering: {
      method: `${SUBPIXEL_SIDE}x${SUBPIXEL_SIDE} identical perspective pinhole ray-bundle quadrature per physical pixel`,
      nominalTransversePixelMetres: spec.distanceMetres * PIXEL_ANGLE,
      topPlaneGrazingStretchMetres: spec.elevationDegrees > 0 ? spec.distanceMetres * PIXEL_ANGLE / Math.sin(spec.elevationDegrees * Math.PI / 180) : null,
    },
  };
}

function composite(premul: Vec3, alpha: number): Vec3 { return premul.map((value, channel) => clamp(value + BACKGROUND[channel]! * (1 - alpha), 0, 1)) as unknown as Vec3; }
function byte(value: number): number { return Math.round(clamp(value, 0, 1) * 255); }
async function writeQa(path: string, pixels: readonly FilteredPixel[]): Promise<void> {
  const panels: Uint8Array[] = [new Uint8Array(WIDTH * HEIGHT * 4), new Uint8Array(WIDTH * HEIGHT * 4), new Uint8Array(WIDTH * HEIGHT * 4), new Uint8Array(WIDTH * HEIGHT * 4)];
  for (let index = 0; index < pixels.length; index++) {
    const pixel = pixels[index]!; const truth = composite(pixel.truthPremul, pixel.truthAlpha); const candidate = composite(pixel.candidatePremul, pixel.candidateAlpha);
    const error = Math.max(...truth.map((value, channel) => Math.abs(value - candidate[channel]!))); const totalPayload = Object.values(pixel.truthPayload).reduce((sum, value) => sum + value, 0);
    const payload: Vec3 = totalPayload > 0 ? [
      (pixel.truthPayload.foliage * 0.15 + pixel.truthPayload['reproductive-purple'] * 0.65 + pixel.truthPayload['reproductive-cream']) / totalPayload,
      (pixel.truthPayload.foliage * 0.55 + pixel.truthPayload['reproductive-purple'] * 0.20 + pixel.truthPayload['reproductive-cream'] * 0.90) / totalPayload,
      (pixel.truthPayload.foliage * 0.10 + pixel.truthPayload['reproductive-purple'] * 0.70 + pixel.truthPayload['reproductive-cream'] * 0.75) / totalPayload,
    ] : BACKGROUND;
    const colors: Vec3[] = [truth, candidate, [Math.min(1, error * 4), Math.min(1, error), 0], payload];
    for (let panel = 0; panel < panels.length; panel++) {
      const offset = index * 4; panels[panel]![offset] = byte(colors[panel]![0]); panels[panel]![offset + 1] = byte(colors[panel]![1]); panels[panel]![offset + 2] = byte(colors[panel]![2]); panels[panel]![offset + 3] = 255;
    }
  }
  const joined = await sharp({ create: { width: WIDTH * 4, height: HEIGHT, channels: 4, background: '#000' } })
    .composite(panels.map((input, panel) => ({ input: Buffer.from(input), raw: { width: WIDTH, height: HEIGHT, channels: 4 as const }, left: panel * WIDTH, top: 0 }))).png().toBuffer();
  await sharp(joined).resize(WIDTH * 4 * QA_SCALE, HEIGHT * QA_SCALE, { kernel: 'nearest' }).png().toFile(path);
}

const fixture = makeEstonianGraminoidFixture(ESTONIAN_GRAMINOID_PROFILE_IDS.CALAMAGROSTIS_CANESCENS);
const sourceMeshSha256 = meshHash(fixture.mesh); if (sourceMeshSha256 !== EXPECTED_MESH_SHA) throw new Error(`source changed: ${sourceMeshSha256}`);
const networkBytes = readFileSync(NETWORK_PATH); const networkSha256 = sha256(networkBytes); const network = JSON.parse(networkBytes.toString()) as any;
const variant = network.variants[NETWORK_VARIANT] as NetworkEntry[] | undefined;
if (!variant) throw new Error(`network variant not found: ${NETWORK_VARIANT}`);
const entries = variant.filter((entry) => 'sheet' in entry ? entry.sheet.cells.length > 0 : entry.network.edges.length > 0);
const fieldCount = new Set(entries.map((entry) => entry.field)).size;
const sourceGeometry = decodedGeometry(fixture.mesh, fixture.tile, fixture.profileId); const sourcePayload = sourcePayloads(fixture.mesh, fixture.primitiveRecipes);
console.error('[relay-exterior] building source BVH'); const sourceBvh = TriangleBvh.build(sourceGeometry, 8);
const core = compileCore(entries, fixture.tile.sizeX, fixture.tile.topH); const coreBvh = TriangleBvh.build(core.geometry, 8);
console.error(`[relay-exterior] source triangles=${sourceGeometry.triangleCount}, core triangles=${core.geometry.triangleCount}`);
const residual = RESIDUAL_ENABLED
  ? (console.error('[relay-exterior] fitting K=2 per-layer residual (K_total=4 for Tier-1 union)'), fitResidual(
    fixture.mesh, sourceGeometry, sourceBvh, sourcePayload,
    core.geometry, coreBvh, core.colors, core.payloads,
  ))
  : {
    model: {
      hMin: sourceGeometry.bounds.min[1], hMax: sourceGeometry.topH,
      a0: 0, b0: [0, 0, 0] as Vec3, modes: [],
      opacityModeScale: 0, colorModeScales: [0, 0, 0] as Vec3, kappaMinimum: 0,
    },
    report: { enabled: false, reason: 'ideal inverse-authored opaque F4/F5 gate; no general smooth residual' },
  };

const specs: CameraSpec[] = DISTANCES.flatMap((distanceMetres) => VIEW_DIRECTIONS.map((view) => ({ ...view, key: `${view.key}-d${distanceMetres}`, distanceMetres })));
const configuration = {
  schema: RESIDUAL_ENABLED ? 'laas-groundcover-relay-exterior-ray-gate/v1' : 'laas-groundcover-inverse-authored-ideal-field-gate/v1',
  sourceMeshSha256, networkSha256, networkVariant: NETWORK_VARIANT,
  fixedResourceShape: { fieldCount, populationCount: LAYER_TRANSFORMS.length, controlReads: 1, totalTextureReads: 2 * fieldCount + 1 },
  camera: { width: WIDTH, height: HEIGHT, pixelAngleRadians: PIXEL_ANGLE, subpixelSide: SUBPIXEL_SIDE, distancesMetres: DISTANCES, views: VIEW_DIRECTIONS },
  residual: {
    enabled: RESIDUAL_ENABLED,
    fitPhaseSide: FIT_PHASE_SIDE, fitSubpixelSide: FIT_SUBPIXEL_SIDE,
    modeLimitPerLayer: MODE_LIMIT_PER_LAYER, totalLiveModeLimitAcrossTwoLayers: MODE_LIMIT_TOTAL,
    target: 'complete final core-plus-residual prefix transfer at physically reachable event-adjacent cutoffs',
    window: '4u(1-u)', integration: 'closed-form centered sinc and first two derivatives',
    coverageClamp: false,
  },
  tier1Layers: LAYER_TRANSFORMS,
  translationsMetres: TRANSLATIONS_METRES,
  compositingDepthToleranceMetres: COMPOSITING_DEPTH_TOLERANCE_METRES,
  distanceRule: 'source and candidate use the identical perspective subpixel ray bundle; no scalar distance waiver or isotropic footprint is used',
};
const toolSha256 = sha256(readFileSync(import.meta.filename)); const recipeSha256 = sha256(canonicalJson(configuration));
const output = resolve(OUTPUT_ROOT, sourceMeshSha256.slice(0, 16), recipeSha256.slice(0, 16), toolSha256.slice(0, 16)); mkdirSync(resolve(output, 'qa'), { recursive: true });
const reports: Record<string, unknown> = {}; const frames = new Map<string, FilteredPixel[]>();
for (const spec of specs) {
  console.error(`[relay-exterior] ${spec.key}`); const pixels: FilteredPixel[] = [];
  for (let y = 0; y < HEIGHT; y++) for (let x = 0; x < WIDTH; x++) pixels.push(evaluatePixel(spec, x, y, fixture.mesh, sourceGeometry, sourceBvh, sourcePayload, core.geometry, coreBvh, core.colors, core.payloads, residual.model, LAYER_TRANSFORMS));
  frames.set(spec.key, pixels); reports[spec.key] = summarizePixels(spec, pixels);
}

const evaluatePhaseGrid = (spec: CameraSpec, subpixelSide: number, cameraTranslationMetres: number, layers: readonly LayerTransform[]): FilteredPixel[] => {
  const pixels: FilteredPixel[] = [];
  for (let pz = 0; pz < FIT_PHASE_SIDE; pz++) for (let px = 0; px < FIT_PHASE_SIDE; px++) {
    const phaseX = sourceGeometry.tileOriginX + (px + 0.5) / FIT_PHASE_SIDE * sourceGeometry.tileSizeX;
    const phaseZ = sourceGeometry.tileOriginZ + (pz + 0.5) / FIT_PHASE_SIDE * sourceGeometry.tileSizeZ;
    pixels.push(evaluateRayBundle(
      phaseBundleRays(spec, phaseX, phaseZ, sourceGeometry.topH, subpixelSide, cameraTranslationMetres),
      fixture.mesh, sourceGeometry, sourceBvh, sourcePayload,
      core.geometry, coreBvh, core.colors, core.payloads, residual.model, layers,
    ));
  }
  return pixels;
};

const translationSummary = (baseline: readonly FilteredPixel[], moved: readonly FilteredPixel[]): Record<string, unknown> => {
  let stableReferencePixels = 0; let unforcedCandidateClassChanges = 0; const alphaDeltaErrors: number[] = []; const rgbDeltaErrors: number[] = [];
  for (let index = 0; index < baseline.length; index++) {
    const a = baseline[index]!; const b = moved[index]!;
    const referenceA = a.truthAlpha >= 0.5; const referenceB = b.truthAlpha >= 0.5;
    const candidateA = a.candidateAlpha >= 0.5; const candidateB = b.candidateAlpha >= 0.5;
    if (referenceA === referenceB) { stableReferencePixels++; if (candidateA !== candidateB) unforcedCandidateClassChanges++; }
    alphaDeltaErrors.push(Math.abs((b.candidateAlpha - a.candidateAlpha) - (b.truthAlpha - a.truthAlpha)));
    rgbDeltaErrors.push(Math.max(...a.truthPremul.map((_, channel) => Math.abs(
      (b.candidatePremul[channel]! - a.candidatePremul[channel]!) - (b.truthPremul[channel]! - a.truthPremul[channel]!),
    ))));
  }
  return {
    stableReferencePixels, unforcedCandidateClassChanges,
    unforcedCandidateClassChangeFraction: stableReferencePixels ? unforcedCandidateClassChanges / stableReferencePixels : 0,
    filteredAlphaDeltaError: quantiles(alphaDeltaErrors), filteredRgbDeltaError: quantiles(rgbDeltaErrors),
  };
};

console.error('[relay-exterior] phase-stratified two-layer production gate and millimetric translations');
const productionPhaseReports: Record<string, unknown> = {}; const productionPhaseFrames = new Map<string, FilteredPixel[]>();
for (const spec of specs) {
  const baseline = evaluatePhaseGrid(spec, SUBPIXEL_SIDE, 0, LAYER_TRANSFORMS); productionPhaseFrames.set(spec.key, baseline);
  const translations: Record<string, unknown> = {};
  for (const shift of TRANSLATIONS_METRES) translations[String(shift)] = translationSummary(baseline, evaluatePhaseGrid(spec, SUBPIXEL_SIDE, shift, LAYER_TRANSFORMS));
  productionPhaseReports[spec.key] = { summary: summarizePixels(spec, baseline, FIT_PHASE_SIDE, FIT_PHASE_SIDE), translations };
}

console.error('[relay-exterior] 4x4 to 8x8 common-footprint convergence');
const quadratureConvergence: Record<string, unknown> = {};
for (const spec of specs) {
  const q4 = productionPhaseFrames.get(spec.key)!; const q8 = evaluatePhaseGrid(spec, 8, 0, LAYER_TRANSFORMS);
  const sourceAlpha: number[] = []; const sourceRgb: number[] = []; const candidateAlpha: number[] = []; const candidateRgb: number[] = [];
  for (let index = 0; index < q4.length; index++) {
    const a = q4[index]!; const b = q8[index]!;
    sourceAlpha.push(Math.abs(a.truthAlpha - b.truthAlpha)); candidateAlpha.push(Math.abs(a.candidateAlpha - b.candidateAlpha));
    sourceRgb.push(Math.max(...a.truthPremul.map((value, channel) => Math.abs(value - b.truthPremul[channel]!))));
    candidateRgb.push(Math.max(...a.candidatePremul.map((value, channel) => Math.abs(value - b.candidatePremul[channel]!))));
  }
  quadratureConvergence[spec.key] = {
    sourceAlpha: quantiles(sourceAlpha), sourceRgb: quantiles(sourceRgb), candidateAlpha: quantiles(candidateAlpha), candidateRgb: quantiles(candidateRgb),
    fourByFourCoreFalseCoverageMean: q4.reduce((sum, pixel) => sum + pixel.coreFalseCoverageFraction, 0) / q4.length,
    eightByEightCoreFalseCoverageMean: q8.reduce((sum, pixel) => sum + pixel.coreFalseCoverageFraction, 0) / q8.length,
    fourByFourCoreEarlyDepthExcessMean: q4.reduce((sum, pixel) => sum + pixel.coreEarlyDepthExcessFraction, 0) / q4.length,
    eightByEightCoreEarlyDepthExcessMean: q8.reduce((sum, pixel) => sum + pixel.coreEarlyDepthExcessFraction, 0) / q8.length,
  };
}

console.error('[relay-exterior] exact-horizontal exterior air-gap battery');
const horizontalDirections: readonly Vec3[] = [[1, 0, 0], [0, 0, 1], [-1, 0, 0], [0, 0, -1]];
let horizontalOrigin: Vec3 = [sourceGeometry.tileSizeX * 0.5, (sourceGeometry.bounds.min[1] + sourceGeometry.topH) * 0.5, sourceGeometry.tileSizeZ * 0.5]; let bestAirReach = -1;
for (let iy = 0; iy < 8; iy++) for (let iz = 0; iz < 8; iz++) for (let ix = 0; ix < 8; ix++) {
  const origin: Vec3 = [sourceGeometry.tileOriginX + (ix + 0.5) / 8 * sourceGeometry.tileSizeX, sourceGeometry.bounds.min[1] + (iy + 0.5) / 8 * (sourceGeometry.topH - sourceGeometry.bounds.min[1]), sourceGeometry.tileOriginZ + (iz + 0.5) / 8 * sourceGeometry.tileSizeZ];
  let reach = 4;
  for (const direction of horizontalDirections) for (const layer of LAYER_TRANSFORMS) {
    const local = transformRay(origin, direction, layer);
    reach = Math.min(reach, traceSource(fixture.mesh, sourceGeometry, sourceBvh, sourcePayload, local.origin, local.direction, 4)?.t ?? 4, traceCore(core.geometry, coreBvh, core.colors, core.payloads, local.origin, local.direction, 4)?.t ?? 4);
  }
  if (reach > bestAirReach) { bestAirReach = reach; horizontalOrigin = origin; }
}
const exactHorizontal: Record<string, unknown> = { origin: horizontalOrigin, certifiedMinimumSourceOrCoreReachMetresAcrossCardinalDirections: bestAirReach, finiteSceneCutoffMetres: 4, directions: {} };
for (let index = 0; index < horizontalDirections.length; index++) {
  const direction = horizontalDirections[index]!;
  const pixel = evaluateRayBundle([{ origin: horizontalOrigin, direction, cutoff: 4 }], fixture.mesh, sourceGeometry, sourceBvh, sourcePayload, core.geometry, coreBvh, core.colors, core.payloads, residual.model, LAYER_TRANSFORMS);
  (exactHorizontal.directions as Record<string, unknown>)[`cardinal-${index}`] = {
    direction, referenceAlpha: pixel.truthAlpha, candidateAlpha: pixel.candidateAlpha,
    referencePremul: pixel.truthPremul, candidatePremul: pixel.candidatePremul,
    coreFalseCoverageFraction: pixel.coreFalseCoverageFraction, coreEarlyDepthExcessFraction: pixel.coreEarlyDepthExcessFraction,
    firstHitDistanceErrorMetres: pixel.firstHitErrors[0] ?? null,
  };
}

const prefixCounterexampleAt = (spec: CameraSpec, phaseX: number, phaseZ: number, subpixelSide: number): { coverageExcess: number; falseCoverage: number; earlyDepthExcess: number; depthErrors: number[] } => {
  const rays = phaseBundleRays(spec, phaseX, phaseZ, sourceGeometry.topH, subpixelSide);
  const sourceTimes: Array<number | null> = []; const coreTimes: Array<number | null> = []; const events: number[] = []; const depthErrors: number[] = [];
  let falseCoverage = 0; let earlyDepthExcess = 0;
  for (const ray of rays) {
    const sourceHits = LAYER_TRANSFORMS.map((layer) => { const local = transformRay(ray.origin, ray.direction, layer); return traceSource(fixture.mesh, sourceGeometry, sourceBvh, sourcePayload, local.origin, local.direction); }).filter((hit): hit is SurfaceHit => hit !== null).sort((a, b) => a.t - b.t);
    const coreHits = LAYER_TRANSFORMS.map((layer) => { const local = transformRay(ray.origin, ray.direction, layer); return traceCore(core.geometry, coreBvh, core.colors, core.payloads, local.origin, local.direction); }).filter((hit): hit is CoreHit => hit !== null).sort((a, b) => a.t - b.t);
    const sourceTime = sourceHits[0]?.t ?? null; const coreTime = coreHits[0]?.t ?? null; sourceTimes.push(sourceTime); coreTimes.push(coreTime);
    if (sourceTime !== null) events.push(sourceTime); if (coreTime !== null) events.push(coreTime);
    if (coreTime !== null && sourceTime === null) falseCoverage++;
    if (coreTime !== null && sourceTime !== null) { depthErrors.push(Math.abs(coreTime - sourceTime)); if (coreTime < sourceTime - COMPOSITING_DEPTH_TOLERANCE_METRES) earlyDepthExcess++; }
  }
  let coverageExcess = 0;
  for (const event of events) for (const cutoff of [Math.max(0, event - 1e-7), event + 1e-7]) {
    const sourceBefore = sourceTimes.filter((value) => value !== null && value <= cutoff).length;
    const coreBefore = coreTimes.filter((value) => value !== null && value <= cutoff).length;
    coverageExcess = Math.max(coverageExcess, (coreBefore - sourceBefore) / rays.length);
  }
  return { coverageExcess, falseCoverage: falseCoverage / rays.length, earlyDepthExcess: earlyDepthExcess / rays.length, depthErrors };
};

console.error('[relay-exterior] all-event core/source prefix counterexample battery');
const prefixCounterexamples: Record<string, unknown> = {};
const prefixSpecs = specs.filter((spec) => [1, 32].includes(spec.distanceMetres));
for (const spec of prefixSpecs) {
  const byQuadrature: Record<string, unknown> = {};
  for (const subpixelSide of [4, 8]) {
    const coverageExcess: number[] = []; const falseCoverage: number[] = []; const earlyDepthExcess: number[] = []; const depthErrors: number[] = [];
    for (let pz = 0; pz < FIT_PHASE_SIDE; pz++) for (let px = 0; px < FIT_PHASE_SIDE; px++) {
      const phaseX = sourceGeometry.tileOriginX + (px + 0.5) / FIT_PHASE_SIDE * sourceGeometry.tileSizeX;
      const phaseZ = sourceGeometry.tileOriginZ + (pz + 0.5) / FIT_PHASE_SIDE * sourceGeometry.tileSizeZ;
      const result = prefixCounterexampleAt(spec, phaseX, phaseZ, subpixelSide);
      coverageExcess.push(result.coverageExcess); falseCoverage.push(result.falseCoverage); earlyDepthExcess.push(result.earlyDepthExcess); depthErrors.push(...result.depthErrors);
    }
    byQuadrature[`${subpixelSide}x${subpixelSide}`] = {
      maximumPositiveCoreMinusSourceCdf: Math.max(...coverageExcess), positiveCoreMinusSourceCdf: quantiles(coverageExcess),
      pairedSourceMissCoreHit: quantiles(falseCoverage), pairedCoreEarlyBeyondDepthTolerance: quantiles(earlyDepthExcess),
      pairedCoreSourceDepthErrorMetres: quantiles(depthErrors),
      note: 'all first-hit source/core event cutoffs in each sampled bundle; CDF excess is aggregate and paired false coverage/depth are reported separately',
    };
  }
  prefixCounterexamples[spec.key] = byQuadrature;
}
const qaSelection = [
  'oblique-18-az0-d1', 'oblique-18-az0-d32', 'low-oblique-5-az90-d1', 'low-oblique-5-az90-d32',
];
const qa: Array<{ file: string; sha256: string; width: number; height: number; interpretation: string }> = [];
for (let index = 0; index < qaSelection.length; index++) {
  const key = qaSelection[index]!; const file = `qa/${String(index + 1).padStart(3, '0')}-${key}.png`; await writeQa(resolve(output, file), frames.get(key)!);
  qa.push({ file, sha256: sha256(readFileSync(resolve(output, file))), width: WIDTH * 4 * QA_SCALE, height: HEIGHT * QA_SCALE, interpretation: `left-to-right: exact two-layer source filtered through the perspective pixel bundle; two-layer compiled ${NETWORK_VARIANT} ${RESIDUAL_ENABLED ? 'core plus K_total=4 analytic residual' : 'opaque ideal field with no smooth residual'} through the identical bundle; 4x RGB error heat; exact-source payload diagnostic (green foliage, purple reproductive, cream reproductive)` });
}
const report = {
  ...configuration,
  source: { species: fixture.species, tile: fixture.tile, triangleCount: sourceGeometry.triangleCount, bvh: sourceBvh.metrics },
  compiledCore: { triangleCount: core.geometry.triangleCount, arcEdges: core.geometry.triangleCount / 2, payloadTriangles: Object.fromEntries(['foliage', 'reproductive-purple', 'reproductive-cream'].map((payload) => [payload, core.payloads.filter((value) => value === payload).length])) },
  residualFit: residual.report,
  twoLayerContiguousWindowDiagnostics: reports,
  bindingTwoLayerPhaseStratifiedGate: productionPhaseReports,
  quadratureConvergence,
  exactHorizontal,
  prefixCounterexamples,
  qa,
  acceptanceReminder: 'Microscopic reference detail may disappear only where the identical filtered reference loses it. Coherent wedges, gaps, colour loss, fixed-screen bands, and temporal instability remain failures.',
};
const metricsPath = resolve(output, 'metrics.json'); writeFileSync(metricsPath, `${JSON.stringify(report, null, 2)}\n`);
const index = { schema: 'laas-groundcover-relay-exterior-ray-gate-index/v1', tool: 'tools/groundcover-bake/analyze-relay-exterior-ray-gate.ts', toolSha256, sourceMeshSha256, recipeSha256, networkSha256, metrics: 'metrics.json', metricsSha256: sha256(readFileSync(metricsPath)), qa };
writeFileSync(resolve(output, 'index.json'), `${JSON.stringify(index, null, 2)}\n`);
console.log(JSON.stringify({ output, ...index }, null, 2));
