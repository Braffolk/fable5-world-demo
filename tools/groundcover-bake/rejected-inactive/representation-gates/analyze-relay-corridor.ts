/**
 * Optimistic finite-slice positional pre-gate for Section 3.1.1 relay fields.
 *
 * This is intentionally a superset, not a compiler: it intersects epsilon-
 * dilated source-surface slices at finitely many heights after one analytic
 * shear. It ignores normal cones, payload compatibility, owner changes, and
 * all unsampled critical heights. Every certifiable continuous relay boundary
 * is contained by this positional superset. A sparse result is decisive; a
 * generous result only authorizes the stricter event-aware gate.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  ESTONIAN_GRAMINOID_PROFILE_IDS,
  makeEstonianGraminoidFixture,
  type EstonianGraminoidFixture,
  type GraminoidPrimitiveRecipe,
} from '../../EstonianGraminoids';
import type { IndexedMesh } from '../../ProfileFormat';

type Vec2 = readonly [number, number];

const REPO_ROOT = resolve(import.meta.dirname, '../../../..');
const OUTPUT_ROOT = resolve(REPO_ROOT, 'data/work/groundcover-relay-corridor-gate');
const EXPECTED_MESH_SHA = '37b0cf1d33f632b5cde3dd60410ad987eec2e977661d0bc804dd6f2bee7054c0';
const GRID = 256;
const HEIGHT_SLICES = 64;
const TOLERANCES = [0.0025, 0.005, 0.01] as const;
const BAND_SLICE_COUNTS = [2, 3, 5, 9, 17] as const;
const EPSILON = 1e-12;

function sha256(value: Uint8Array | string): string {
  return createHash('sha256').update(value).digest('hex');
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(',')}}`;
}

function meshHash(mesh: IndexedMesh): string { return sha256(JSON.stringify(mesh)); }

function triangleArea(mesh: IndexedMesh, triangle: number): number {
  const i = triangle * 3;
  const ai = mesh.indices[i]! * 3; const bi = mesh.indices[i + 1]! * 3; const ci = mesh.indices[i + 2]! * 3;
  const ax = mesh.positions[ai]!; const ay = mesh.positions[ai + 1]!; const az = mesh.positions[ai + 2]!;
  const ux = mesh.positions[bi]! - ax; const uy = mesh.positions[bi + 1]! - ay; const uz = mesh.positions[bi + 2]! - az;
  const vx = mesh.positions[ci]! - ax; const vy = mesh.positions[ci + 1]! - ay; const vz = mesh.positions[ci + 2]! - az;
  return 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
}

function lateralTriangles(recipe: GraminoidPrimitiveRecipe): number[] {
  if (recipe.disposition !== 'crisp') return [];
  if (recipe.kind === 'blade' || recipe.kind === 'lanceolate-surface') {
    const segments = recipe.kind === 'blade' ? recipe.centerline.length - 2 : recipe.centerline.length - 1;
    return Array.from({ length: segments * 2 }, (_, index) => recipe.sourceTriangleStart + 1 + index);
  }
  if (recipe.kind === 'tube' || recipe.kind === 'rhizome') {
    const start = recipe.sourceTriangleStart + (recipe.anchorVertex !== null ? recipe.sides : 0);
    return Array.from({ length: (recipe.centerline.length - 1) * recipe.sides * 2 }, (_, index) => start + index);
  }
  return [];
}

function recognitionClass(recipe: GraminoidPrimitiveRecipe): 'foliage' | 'reproductive' | 'structure' {
  if (recipe.kind === 'blade' || (recipe.kind === 'tube' && ['culm', 'basal-sheath', 'upper-sheath'].includes(recipe.role))) return 'foliage';
  if (recipe.kind === 'lanceolate-surface' || (recipe.kind === 'tube' && ['panicle-axis', 'spikelet-axis', 'anther-filament'].includes(recipe.role))) return 'reproductive';
  return 'structure';
}

function sourceBounds(fixture: EstonianGraminoidFixture, triangles: readonly number[]): { minY: number; maxY: number } {
  let minY = Infinity; let maxY = -Infinity;
  for (const triangle of triangles) for (let corner = 0; corner < 3; corner++) {
    const y = fixture.mesh.positions[fixture.mesh.indices[triangle * 3 + corner]! * 3 + 1]!;
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  return { minY, maxY };
}

function wrap(value: number): number { return ((value % GRID) + GRID) % GRID; }

function drawSegment(mask: Uint8Array, a: Vec2, b: Vec2, tileSize: number): void {
  let ax = a[0] / tileSize * GRID; let az = a[1] / tileSize * GRID;
  let bx = b[0] / tileSize * GRID; let bz = b[1] / tileSize * GRID;
  if (bx - ax > GRID * 0.5) bx -= GRID;
  if (bx - ax < -GRID * 0.5) bx += GRID;
  if (bz - az > GRID * 0.5) bz -= GRID;
  if (bz - az < -GRID * 0.5) bz += GRID;
  const steps = Math.max(1, Math.ceil(Math.max(Math.abs(bx - ax), Math.abs(bz - az)) * 1.5));
  for (let step = 0; step <= steps; step++) {
    const t = step / steps;
    const x = wrap(Math.round(ax + (bx - ax) * t));
    const z = wrap(Math.round(az + (bz - az) * t));
    mask[z * GRID + x] = 1;
  }
}

function intersectTriangleAtHeight(mesh: IndexedMesh, triangle: number, h: number): Vec2[] {
  const points: Vec2[] = [];
  const vertices = Array.from({ length: 3 }, (_, corner) => {
    const offset = mesh.indices[triangle * 3 + corner]! * 3;
    return [mesh.positions[offset]!, mesh.positions[offset + 1]!, mesh.positions[offset + 2]!] as const;
  });
  for (let edge = 0; edge < 3; edge++) {
    const a = vertices[edge]!; const b = vertices[(edge + 1) % 3]!;
    const da = a[1] - h; const db = b[1] - h;
    if (Math.abs(da) < 1e-10) points.push([a[0], a[2]]);
    if (da * db < 0) {
      const t = (h - a[1]) / (b[1] - a[1]);
      points.push([a[0] + (b[0] - a[0]) * t, a[2] + (b[2] - a[2]) * t]);
    }
  }
  const unique: Vec2[] = [];
  for (const point of points) if (!unique.some((other) => Math.hypot(point[0] - other[0], point[1] - other[1]) < 1e-9)) unique.push(point);
  return unique;
}

function rasterSourceSlices(
  fixture: EstonianGraminoidFixture, triangles: readonly number[], heights: readonly number[],
): Uint8Array[] {
  const masks = heights.map(() => new Uint8Array(GRID * GRID));
  const step = heights[1]! - heights[0]!;
  for (const triangle of triangles) {
    let triangleMin = Infinity; let triangleMax = -Infinity;
    for (let corner = 0; corner < 3; corner++) {
      const y = fixture.mesh.positions[fixture.mesh.indices[triangle * 3 + corner]! * 3 + 1]!;
      triangleMin = Math.min(triangleMin, y); triangleMax = Math.max(triangleMax, y);
    }
    const first = Math.max(0, Math.ceil((triangleMin - heights[0]!) / step - 1e-10));
    const last = Math.min(heights.length - 1, Math.floor((triangleMax - heights[0]!) / step + 1e-10));
    for (let slice = first; slice <= last; slice++) {
      const points = intersectTriangleAtHeight(fixture.mesh, triangle, heights[slice]!);
      if (points.length >= 2) drawSegment(masks[slice]!, points[0]!, points[1]!, fixture.tile.sizeX);
    }
  }
  return masks;
}

function dilationOffsets(tolerance: number, pixel: number): Vec2[] {
  // Cell-centre rasterization plus an extra half-diagonal makes this a
  // positional superset rather than an accidental erosion of the source.
  const radius = tolerance + Math.SQRT2 * pixel;
  const cells = Math.ceil(radius / pixel);
  const offsets: Vec2[] = [];
  for (let z = -cells; z <= cells; z++) for (let x = -cells; x <= cells; x++) {
    if (Math.hypot(x, z) * pixel <= radius + EPSILON) offsets.push([x, z]);
  }
  return offsets;
}

function dilate(mask: Uint8Array, offsets: readonly Vec2[]): Uint8Array {
  const result = new Uint8Array(mask.length);
  for (let index = 0; index < mask.length; index++) {
    if (!mask[index]) continue;
    const x = index % GRID; const z = Math.floor(index / GRID);
    for (const offset of offsets) result[wrap(z + offset[1]) * GRID + wrap(x + offset[0])] = 1;
  }
  return result;
}

function toShiftedBits(mask: Uint8Array, shiftX: number, shiftZ: number): Uint32Array {
  const bits = new Uint32Array(Math.ceil(mask.length / 32));
  for (let index = 0; index < mask.length; index++) {
    if (!mask[index]) continue;
    const x = wrap(index % GRID + shiftX); const z = wrap(Math.floor(index / GRID) + shiftZ);
    const target = z * GRID + x;
    bits[target >>> 5] |= (1 << (target & 31)) >>> 0;
  }
  return bits;
}

function popcount32(value: number): number {
  value -= (value >>> 1) & 0x55555555;
  value = (value & 0x33333333) + ((value >>> 2) & 0x33333333);
  return (((value + (value >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

function corridorBits(slices: readonly Uint32Array[], first: number, count: number): Uint32Array {
  const result = new Uint32Array(slices[first]!);
  for (let slice = first + 1; slice < first + count; slice++) {
    const current = slices[slice]!;
    for (let word = 0; word < result.length; word++) result[word] &= current[word]!;
  }
  return result;
}

function corridorMetrics(bits: Uint32Array, pixel: number, erosionCells = 1): {
  corridorAreaMetres2: number;
  regularClosedBoundaryLengthMetres: number;
  rawGridBoundaryLengthMetres: number;
  erodedInteriorAreaMetres2: number;
  minimumFeatureRadiusMetres: number;
} {
  let occupied = 0;
  for (const word of bits) occupied += popcount32(word);
  const at = (x: number, z: number): boolean => {
    const index = wrap(z) * GRID + wrap(x);
    return (bits[index >>> 5]! & ((1 << (index & 31)) >>> 0)) !== 0;
  };
  const eroded = new Uint8Array(GRID * GRID);
  let erodedCount = 0;
  for (let z = 0; z < GRID; z++) for (let x = 0; x < GRID; x++) {
    let survives = true;
    for (let dz = -erosionCells; dz <= erosionCells && survives; dz++) for (let dx = -erosionCells; dx <= erosionCells; dx++) {
      if (dx * dx + dz * dz <= erosionCells * erosionCells && !at(x + dx, z + dz)) { survives = false; break; }
    }
    if (survives) {
      eroded[z * GRID + x] = 1; erodedCount++;
    }
  }
  let boundaryEdges = 0;
  for (let z = 0; z < GRID; z++) for (let x = 0; x < GRID; x++) {
    if (!eroded[z * GRID + x]) continue;
    boundaryEdges += !eroded[z * GRID + wrap(x - 1)] ? 1 : 0;
    boundaryEdges += !eroded[z * GRID + wrap(x + 1)] ? 1 : 0;
    boundaryEdges += !eroded[wrap(z - 1) * GRID + x] ? 1 : 0;
    boundaryEdges += !eroded[wrap(z + 1) * GRID + x] ? 1 : 0;
  }
  const corridorArea = occupied * pixel * pixel;
  const rawBoundary = boundaryEdges * pixel;
  return {
    corridorAreaMetres2: corridorArea,
    regularClosedBoundaryLengthMetres: Math.min(rawBoundary, corridorArea / Math.max(EPSILON, erosionCells * pixel)),
    rawGridBoundaryLengthMetres: rawBoundary,
    erodedInteriorAreaMetres2: erodedCount * pixel * pixel,
    minimumFeatureRadiusMetres: erosionCells * pixel,
  };
}

function betaCandidates(fixture: EstonianGraminoidFixture, count = 24): Vec2[] {
  const samples: Array<{ beta: Vec2; weight: number }> = [];
  for (const recipe of fixture.primitiveRecipes) {
    if (recipe.disposition !== 'crisp' || !('centerline' in recipe)) continue;
    for (let segment = 0; segment < recipe.centerline.length - 1; segment++) {
      const a = recipe.centerline[segment]!; const b = recipe.centerline[segment + 1]!;
      const dy = b.y - a.y;
      if (Math.abs(dy) < 1e-5) continue;
      const beta: Vec2 = [(b.x - a.x) / dy, (b.z - a.z) / dy];
      if (Math.hypot(...beta) > 4) continue;
      let width = 0.001;
      if ('halfWidths' in recipe) width = recipe.halfWidths[segment]! + recipe.halfWidths[segment + 1]!;
      else if ('radii' in recipe) width = recipe.radii[segment]! + recipe.radii[segment + 1]!;
      samples.push({ beta, weight: Math.max(1e-9, Math.abs(dy) * width) });
    }
  }
  if (samples.length === 0) return [[0, 0]];
  const centres: Vec2[] = [[0, 0]];
  while (centres.length < count) {
    let winner = samples[0]!; let winnerScore = -Infinity;
    for (const sample of samples) {
      const distance2 = Math.min(...centres.map((centre) =>
        (sample.beta[0] - centre[0]) ** 2 + (sample.beta[1] - centre[1]) ** 2));
      const score = sample.weight * distance2;
      if (score > winnerScore) { winner = sample; winnerScore = score; }
    }
    centres.push(winner.beta);
  }
  for (let iteration = 0; iteration < 16; iteration++) {
    const sums = centres.map(() => ({ x: 0, z: 0, weight: 0 }));
    for (const sample of samples) {
      let winner = 0; let distance = Infinity;
      centres.forEach((centre, index) => {
        const current = (sample.beta[0] - centre[0]) ** 2 + (sample.beta[1] - centre[1]) ** 2;
        if (current < distance) { distance = current; winner = index; }
      });
      sums[winner]!.x += sample.beta[0] * sample.weight;
      sums[winner]!.z += sample.beta[1] * sample.weight;
      sums[winner]!.weight += sample.weight;
    }
    sums.forEach((sum, index) => {
      if (sum.weight > 0) centres[index] = [sum.x / sum.weight, sum.z / sum.weight];
    });
  }
  return centres;
}

const fixture = makeEstonianGraminoidFixture(ESTONIAN_GRAMINOID_PROFILE_IDS.CALAMAGROSTIS_CANESCENS);
const sourceMeshSha = meshHash(fixture.mesh);
if (sourceMeshSha !== EXPECTED_MESH_SHA) throw new Error(`accepted Calamagrostis source changed: ${sourceMeshSha}`);
const triangleGroups: Record<string, number[]> = { foliage: [], reproductive: [], structure: [] };
for (const recipe of fixture.primitiveRecipes) triangleGroups[recognitionClass(recipe)]!.push(...lateralTriangles(recipe));
const allTriangles = Object.values(triangleGroups).flat();
const bounds = sourceBounds(fixture, allTriangles);
const heights = Array.from({ length: HEIGHT_SLICES }, (_, index) =>
  bounds.minY + (bounds.maxY - bounds.minY) * index / (HEIGHT_SLICES - 1));
const heightStep = heights[1]! - heights[0]!;
const pixel = fixture.tile.sizeX / GRID;
const betas = betaCandidates(fixture);

const recipe = {
  schema: 'laas-groundcover-relay-corridor-optimistic-pre-gate/v1',
  methodRevision: 'finite-slice-position-payload-class-split-closed-arc-superset-v3',
  grid: GRID,
  heightSlices: HEIGHT_SLICES,
  tolerancesMetres: TOLERANCES,
  bandSliceCounts: BAND_SLICE_COUNTS,
  betaCandidates: betas,
  betaCandidateMethod: '24 deterministic weighted k-means centres from actual crisp recipe pullback slopes dx/dy,dz/dy; weights are vertical span times local width/radius; |beta|>4 and nearly horizontal dy are excluded',
  fieldCount: 4,
  periodicTileMetres: fixture.tile.sizeX,
  source: 'accepted production Calamagrostis crisp lateral recipe bodies; caps, hubs, and plume hairs excluded',
  payloadSplit: ['foliage', 'reproductive', 'structure'],
  finiteSliceCaveat: 'intersection uses only selected sampled heights; normal cones, exact categorical payload within each coarse recognition class, owner identity, and between-slice critical events are ignored. It remains an optimistic positional superset.',
  arcPackingProxy: 'a closed-set relay may be a finite 1D arc with included endpoints; optimistic atlas-resolvable arc length is bounded by corridorArea/(2 grid cells), avoiding raster-perimeter divergence',
  distanceRule: 'this structural gate makes no scalar-width distance rejection. Any later exterior-ray gate must use the full projection Jacobian and identical anisotropic pixel footprints; coherent fixed-screen artifacts remain failures.',
};
const recipeSha = sha256(canonicalJson(recipe));
const toolSha = sha256(readFileSync(import.meta.filename));
const output = resolve(OUTPUT_ROOT, sourceMeshSha.slice(0, 16), recipeSha.slice(0, 16), toolSha.slice(0, 16));
mkdirSync(output, { recursive: true });

function selectHeightDisjoint(candidates: Array<Record<string, number | Vec2>>, count: number): Array<Record<string, number | Vec2>> {
  const selected: Array<Record<string, number | Vec2>> = [];
  for (const candidate of candidates) {
    const hMin = Number(candidate.hMin); const hMax = Number(candidate.hMax);
    if (selected.some((other) => Math.min(hMax, Number(other.hMax)) - Math.max(hMin, Number(other.hMin)) > EPSILON)) continue;
    selected.push(candidate);
    if (selected.length === count) break;
  }
  return selected;
}

function analyzeGroup(name: string, triangles: readonly number[]): Record<string, unknown> {
  const totalLateralArea = triangles.reduce((sum, triangle) => sum + triangleArea(fixture.mesh, triangle), 0);
  console.error(`[relay-corridor] rasterizing ${name} (${triangles.length} triangles)`);
  const thinSlices = rasterSourceSlices(fixture, triangles, heights);
  const toleranceReports: Record<string, unknown> = {};
  for (const tolerance of TOLERANCES) {
    const offsets = dilationOffsets(tolerance, pixel);
    const dilated = thinSlices.map((mask) => dilate(mask, offsets));
    const shiftedByBeta = betas.map((beta) => dilated.map((mask, slice) => toShiftedBits(
      mask,
      Math.round(-beta[0] * heights[slice]! / pixel),
      Math.round(-beta[1] * heights[slice]! / pixel),
    )));
    const candidates: Array<Record<string, number | Vec2>> = [];
    for (let betaIndex = 0; betaIndex < betas.length; betaIndex++) for (const count of BAND_SLICE_COUNTS) {
      const stride = Math.max(1, Math.floor((count - 1) / 2));
      for (let first = 0; first + count <= HEIGHT_SLICES; first += stride) {
        const bits = corridorBits(shiftedByBeta[betaIndex]!, first, count);
        const metrics1 = corridorMetrics(bits, pixel, 1);
        const metrics2 = corridorMetrics(bits, pixel, 2);
        const metrics4 = corridorMetrics(bits, pixel, 4);
        const bandHeight = heights[first + count - 1]! - heights[first]!;
        const optimisticArcPackingLength = metrics1.corridorAreaMetres2 / (2 * pixel);
        candidates.push({
          beta: betas[betaIndex]!, firstSlice: first, sliceCount: count,
          hMin: heights[first]!, hMax: heights[first + count - 1]!, bandHeightMetres: bandHeight,
          corridorAreaMetres2: metrics1.corridorAreaMetres2,
          erodedInteriorAreaRadius1Metres2: metrics1.erodedInteriorAreaMetres2,
          boundaryLengthRadius1Metres: metrics1.regularClosedBoundaryLengthMetres,
          boundaryLengthRadius2Metres: metrics2.regularClosedBoundaryLengthMetres,
          boundaryLengthRadius4Metres: metrics4.regularClosedBoundaryLengthMetres,
          optimisticArcPackingLengthMetres: optimisticArcPackingLength,
          optimisticLateralSupportMetres2: optimisticArcPackingLength * bandHeight,
        });
      }
    }
    candidates.sort((a, b) => Number(b.optimisticLateralSupportMetres2) - Number(a.optimisticLateralSupportMetres2));
    const duplicateTopFour = candidates.slice(0, 4);
    const heightDisjointTopFour = selectHeightDisjoint(candidates, 4);
    const summarize = (selected: Array<Record<string, number | Vec2>>) => ({
      fields: selected,
      summedSupportMetres2: selected.reduce((sum, item) => sum + Number(item.optimisticLateralSupportMetres2), 0),
      supportFraction: Math.min(1, selected.reduce((sum, item) => sum + Number(item.optimisticLateralSupportMetres2), 0) / Math.max(EPSILON, totalLateralArea)),
      heightCoverageFraction: selected.reduce((sum, item) => sum + Number(item.bandHeightMetres), 0) / Math.max(EPSILON, bounds.maxY - bounds.minY),
    });
    toleranceReports[String(tolerance)] = {
      candidateCount: candidates.length,
      duplicatePermittingTopFour: summarize(duplicateTopFour),
      heightDisjointTopFour: summarize(heightDisjointTopFour),
      sourceLateralAreaMetres2: totalLateralArea,
      interpretation: 'duplicate-permitting is optimistic non-refutation only; height-disjoint is a deduplicated diagnostic, not a certified optimum. Arc-packing support is an area/minimum-separation upper proxy and still ignores normal and continuous-height constraints.',
      topThirtyTwo: candidates.slice(0, 32),
    };
  }
  return { sourceLateralTriangleCount: triangles.length, sourceLateralAreaMetres2: totalLateralArea, tolerances: toleranceReports };
}

const groups: Record<string, unknown> = { all: analyzeGroup('all', allTriangles) };
for (const [name, triangles] of Object.entries(triangleGroups)) groups[name] = analyzeGroup(name, triangles);

const report = {
  ...recipe,
  status: 'OPTIMISTIC_POSITIONAL_SUPERSET_ONLY',
  sourceMeshSha256: sourceMeshSha,
  sourceLateralTriangleCount: allTriangles.length,
  sourceLateralAreaMetres2: allTriangles.reduce((sum, triangle) => sum + triangleArea(fixture.mesh, triangle), 0),
  sourceHeightBoundsMetres: bounds,
  heightStepMetres: heightStep,
  pixelMetres: pixel,
  decisionRule: 'Only negligible persistent positional arc support is decisively RED. A generous result requires continuous-height, owner/payload, oriented-normal, finite-atlas, and exterior-ray certification. Closed loops are not required: finite arcs with included endpoints are closed sets.',
  groups,
};
const metricsPath = resolve(output, 'metrics.json');
writeFileSync(metricsPath, `${JSON.stringify(report, null, 2)}\n`);
const index = {
  schema: 'laas-groundcover-relay-corridor-optimistic-pre-gate-index/v1',
  tool: 'tools/groundcover-bake/analyze-relay-corridor.ts',
  toolSha256: toolSha,
  sourceMeshSha256: sourceMeshSha,
  recipeSha256: recipeSha,
  metrics: 'metrics.json',
  metricsSha256: sha256(readFileSync(metricsPath)),
};
writeFileSync(resolve(output, 'index.json'), `${JSON.stringify(index, null, 2)}\n`);
console.log(JSON.stringify({ output, ...index }, null, 2));
