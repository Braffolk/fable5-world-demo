/** Constructive conservative relay certificate for four selected class bands. */
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  ESTONIAN_GRAMINOID_PROFILE_IDS,
  makeEstonianGraminoidFixture,
  type GraminoidPrimitiveRecipe,
} from '../../EstonianGraminoids';
import type { IndexedMesh } from '../../ProfileFormat';

type Vec2 = readonly [number, number];
type Vec3 = readonly [number, number, number];
interface Candidate { beta: Vec2; hMin: number; hMax: number; }

const REPO_ROOT = resolve(import.meta.dirname, '../../../..');
const INPUT = resolve(REPO_ROOT, 'data/work/groundcover-relay-corridor-gate/37b0cf1d33f632b5/68750bca57259e7c/0234934c9a4f87a6/metrics.json');
const OUTPUT_ROOT = resolve(REPO_ROOT, 'data/work/groundcover-relay-constructive-gate');
const EXPECTED_MESH_SHA = '37b0cf1d33f632b5cde3dd60410ad987eec2e977661d0bc804dd6f2bee7054c0';
const GRID = 256;
const NORMAL_BINS = 16;
const HEIGHT_BINS = 64;
const NORMAL_ANGLES = [15, 30, 45] as const;
const MINIMUM_ARC_EDGES = [2, 4, 8] as const;
const TOLERANCES = [0.0025, 0.005, 0.01] as const;
const EPSILON = 1e-12;

function sha256(value: Uint8Array | string): string { return createHash('sha256').update(value).digest('hex'); }
function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(',')}}`;
}
function meshHash(mesh: IndexedMesh): string { return sha256(JSON.stringify(mesh)); }
function wrap(value: number): number { return ((value % GRID) + GRID) % GRID; }

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

interface PolyVertex { qx: number; qz: number; h: number; }
function clipPolygon(poly: PolyVertex[], axis: 'qx' | 'qz', bound: number, greater: boolean): PolyVertex[] {
  const result: PolyVertex[] = [];
  for (let index = 0; index < poly.length; index++) {
    const a = poly[index]!; const b = poly[(index + 1) % poly.length]!;
    const ain = greater ? a[axis] >= bound : a[axis] <= bound;
    const bin = greater ? b[axis] >= bound : b[axis] <= bound;
    if (ain) result.push(a);
    if (ain !== bin) {
      const t = (bound - a[axis]) / (b[axis] - a[axis]);
      result.push({ qx: a.qx + (b.qx - a.qx) * t, qz: a.qz + (b.qz - a.qz) * t, h: a.h + (b.h - a.h) * t });
    }
  }
  return result;
}

function clipToCell(poly: PolyVertex[], cx: number, cz: number, half: number): PolyVertex[] {
  let current = clipPolygon(poly, 'qx', cx - half, true);
  if (current.length) current = clipPolygon(current, 'qx', cx + half, false);
  if (current.length) current = clipPolygon(current, 'qz', cz - half, true);
  if (current.length) current = clipPolygon(current, 'qz', cz + half, false);
  return current;
}

function triangle(mesh: IndexedMesh, triangleIndex: number, beta: Vec2, tile: number): { poly: PolyVertex[]; normal: Vec3; minH: number; maxH: number } {
  const world = Array.from({ length: 3 }, (_, corner) => {
    const offset = mesh.indices[triangleIndex * 3 + corner]! * 3;
    return [mesh.positions[offset]!, mesh.positions[offset + 1]!, mesh.positions[offset + 2]!] as Vec3;
  });
  const poly = world.map(([x, y, z]) => ({ qx: x - beta[0] * y, qz: z - beta[1] * y, h: y }));
  for (let index = 1; index < 3; index++) {
    while (poly[index]!.qx - poly[0]!.qx > tile * 0.5) poly[index]!.qx -= tile;
    while (poly[index]!.qx - poly[0]!.qx < -tile * 0.5) poly[index]!.qx += tile;
    while (poly[index]!.qz - poly[0]!.qz > tile * 0.5) poly[index]!.qz -= tile;
    while (poly[index]!.qz - poly[0]!.qz < -tile * 0.5) poly[index]!.qz += tile;
  }
  const u: Vec3 = [world[1]![0] - world[0]![0], world[1]![1] - world[0]![1], world[1]![2] - world[0]![2]];
  const v: Vec3 = [world[2]![0] - world[0]![0], world[2]![1] - world[0]![1], world[2]![2] - world[0]![2]];
  const raw: Vec3 = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  const magnitude = Math.hypot(...raw);
  const normal: Vec3 = magnitude > EPSILON ? [raw[0] / magnitude, raw[1] / magnitude, raw[2] / magnitude] : [0, 1, 0];
  return { poly, normal, minH: Math.min(...poly.map((p) => p.h)), maxH: Math.max(...poly.map((p) => p.h)) };
}

function templateNormal(beta: Vec2, bin: number): Vec3 {
  const angle = 2 * Math.PI * bin / NORMAL_BINS;
  const nx = Math.cos(angle); const nz = Math.sin(angle);
  const raw: Vec3 = [nx, -(nx * beta[0] + nz * beta[1]), nz];
  const magnitude = Math.hypot(...raw);
  return [raw[0] / magnitude, raw[1] / magnitude, raw[2] / magnitude];
}

function angularErrorDegrees(a: Vec3, b: Vec3): number {
  return Math.acos(Math.max(-1, Math.min(1, Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])))) * 180 / Math.PI;
}

function setBinRange(target: Uint32Array, offset: number, first: number, last: number, words: number): void {
  for (let bin = first; bin <= last; bin++) target[offset + (bin >>> 5)] |= (1 << (bin & 31)) >>> 0;
  // `words` is included to keep the packed layout explicit at this call site.
  void words;
}

function erode(mask: Uint8Array, radius: number): Uint8Array {
  const result = new Uint8Array(mask.length);
  for (let z = 0; z < GRID; z++) for (let x = 0; x < GRID; x++) {
    let keep = true;
    for (let dz = -radius; dz <= radius && keep; dz++) for (let dx = -radius; dx <= radius; dx++) {
      if (dx * dx + dz * dz <= radius * radius && !mask[wrap(z + dz) * GRID + wrap(x + dx)]) { keep = false; break; }
    }
    result[z * GRID + x] = keep ? 1 : 0;
  }
  return result;
}

function arcBins(a: number, b: number): number[] {
  let clockwise = (b - a + NORMAL_BINS) % NORMAL_BINS;
  if (clockwise > NORMAL_BINS / 2) { [a, b] = [b, a]; clockwise = NORMAL_BINS - clockwise; }
  return Array.from({ length: clockwise + 1 }, (_, index) => (a + index) % NORMAL_BINS);
}

// Retained research primitives used by the next erosion/arc ablation.
void erode;
void arcBins;

function certifiedArcs(safeByNormal: Uint8Array[], minimumEdges: number, pixel: number, bandHeight: number): Record<string, number> {
  const parent = new Int32Array(GRID * GRID); parent.fill(-1);
  const find = (input: number): number => {
    let node = input;
    while (parent[node] !== node) node = parent[node]!;
    let cursor = input;
    while (parent[cursor] !== cursor) { const next = parent[cursor]!; parent[cursor] = node; cursor = next; }
    return node;
  };
  const unite = (a: number, b: number): void => {
    if (parent[a] < 0) parent[a] = a;
    if (parent[b] < 0) parent[b] = b;
    const ra = find(a); const rb = find(b); if (ra !== rb) parent[rb] = ra;
  };
  const edges: Array<readonly [number, number, number]> = [];
  // Tangent directions 0,45,90,135 degrees. Their transported q-normal is
  // +90 degrees; unoriented source-normal comparison also admits the reverse.
  const directions = [
    { dx: 1, dz: 0, normalBin: 4, length: 1 },
    { dx: 1, dz: 1, normalBin: 6, length: Math.SQRT2 },
    { dx: 0, dz: 1, normalBin: 8, length: 1 },
    { dx: -1, dz: 1, normalBin: 10, length: Math.SQRT2 },
  ];
  for (let z = 0; z < GRID; z++) for (let x = 0; x < GRID; x++) {
    const cell = z * GRID + x;
    for (const direction of directions) {
      const neighbour = wrap(z + direction.dz) * GRID + wrap(x + direction.dx);
      if (!safeByNormal[direction.normalBin]![cell] || !safeByNormal[direction.normalBin]![neighbour]) continue;
      unite(cell, neighbour); edges.push([cell, neighbour, direction.length]);
    }
  }
  const components = new Map<number, { edges: number; length: number; nodes: Set<number> }>();
  for (const [a, b, edgeLength] of edges) {
    const root = find(a); const component = components.get(root) ?? { edges: 0, length: 0, nodes: new Set<number>() };
    component.edges++; component.length += edgeLength * pixel; component.nodes.add(a); component.nodes.add(b);
    components.set(root, component);
  }
  const accepted = [...components.values()].filter((component) => component.edges >= minimumEdges);
  const length = accepted.reduce((sum, component) => sum + component.length, 0);
  return {
    persistentArcComponents: accepted.length,
    rejectedShortComponents: components.size - accepted.length,
    certifiedArcNodeCount: accepted.reduce((sum, component) => sum + component.nodes.size, 0),
    certifiedArcLengthMetres: length,
    certifiedLateralSupportMetres2: length * bandHeight,
  };
}

function certifyCandidate(
  mesh: IndexedMesh, triangles: readonly number[], candidate: Candidate,
  tolerance: number, tile: number,
): Record<string, unknown> {
  const pixel = tile / GRID;
  const cellRadius = Math.SQRT2 * pixel * 0.5;
  const sourceReach = tolerance - cellRadius;
  if (sourceReach <= 0) return { candidate, status: 'NO_CELL_CERTIFICATE', reason: 'epsilon <= cell half-diagonal' };
  const clipHalfWidth = sourceReach / Math.SQRT2;
  const words = Math.ceil(HEIGHT_BINS / 32);
  const coverages = NORMAL_ANGLES.map(() => new Uint32Array(GRID * GRID * NORMAL_BINS * words));
  const binHeight = (candidate.hMax - candidate.hMin) / HEIGHT_BINS;
  let overlappingTriangles = 0; let clippedTriangleCells = 0; let shortIntervals = 0; let qualifiedNormalWrites = 0;
  for (const triangleIndex of triangles) {
    const source = triangle(mesh, triangleIndex, candidate.beta, tile);
    if (source.maxH < candidate.hMin - EPSILON || source.minH > candidate.hMax + EPSILON) continue;
    overlappingTriangles++;
    const minQx = Math.min(...source.poly.map((p) => p.qx)) - clipHalfWidth;
    const maxQx = Math.max(...source.poly.map((p) => p.qx)) + clipHalfWidth;
    const minQz = Math.min(...source.poly.map((p) => p.qz)) - clipHalfWidth;
    const maxQz = Math.max(...source.poly.map((p) => p.qz)) + clipHalfWidth;
    const errors = Array.from({ length: NORMAL_BINS }, (_, bin) => angularErrorDegrees(source.normal, templateNormal(candidate.beta, bin)));
    for (let iz = Math.floor(minQz / pixel); iz <= Math.floor(maxQz / pixel); iz++) for (let ix = Math.floor(minQx / pixel); ix <= Math.floor(maxQx / pixel); ix++) {
      const clipped = clipToCell(source.poly, (ix + 0.5) * pixel, (iz + 0.5) * pixel, clipHalfWidth);
      if (clipped.length === 0) continue;
      clippedTriangleCells++;
      const low = Math.max(candidate.hMin, Math.min(...clipped.map((p) => p.h)));
      const high = Math.min(candidate.hMax, Math.max(...clipped.map((p) => p.h)));
      const firstFullBin = Math.ceil((low - candidate.hMin) / binHeight - 1e-10);
      const lastFullBin = Math.floor((high - candidate.hMin) / binHeight + 1e-10) - 1;
      if (lastFullBin < firstFullBin) { shortIntervals++; continue; }
      const cell = wrap(iz) * GRID + wrap(ix);
      for (let angleIndex = 0; angleIndex < NORMAL_ANGLES.length; angleIndex++) for (let normalBin = 0; normalBin < NORMAL_BINS; normalBin++) {
        if (errors[normalBin]! > NORMAL_ANGLES[angleIndex]!) continue;
        const offset = (cell * NORMAL_BINS + normalBin) * words;
        setBinRange(coverages[angleIndex]!, offset, Math.max(0, firstFullBin), Math.min(HEIGHT_BINS - 1, lastFullBin), words);
        qualifiedNormalWrites++;
      }
    }
  }
  const fullWords = Array.from({ length: words }, (_, word) => word === words - 1 && HEIGHT_BINS % 32
    ? (2 ** (HEIGHT_BINS % 32) - 1) >>> 0 : 0xffffffff);
  const pareto: Record<string, unknown> = {};
  for (let angleIndex = 0; angleIndex < NORMAL_ANGLES.length; angleIndex++) {
    const safe = Array.from({ length: NORMAL_BINS }, () => new Uint8Array(GRID * GRID));
    let safeCellNormalPairs = 0;
    for (let cell = 0; cell < GRID * GRID; cell++) for (let normalBin = 0; normalBin < NORMAL_BINS; normalBin++) {
      const offset = (cell * NORMAL_BINS + normalBin) * words;
      let full = true;
      for (let word = 0; word < words; word++) if (coverages[angleIndex]![offset + word] !== fullWords[word]) { full = false; break; }
      if (full) { safe[normalBin]![cell] = 1; safeCellNormalPairs++; }
    }
    pareto[String(NORMAL_ANGLES[angleIndex])] = {
      safeCellNormalPairs,
      minimumArcEdges: Object.fromEntries(MINIMUM_ARC_EDGES.map((minimumEdges) => [String(minimumEdges), certifiedArcs(
        safe, minimumEdges, pixel, candidate.hMax - candidate.hMin,
      )])),
    };
  }
  return {
    candidate, status: 'CONSTRUCTIVE_CONSERVATIVE_CLOSED_ARC_GRID_CERTIFICATE',
    cellHalfDiagonalMetres: cellRadius, inscribedSourceReachMetres: sourceReach,
    verticalBinHeightMetres: binHeight,
    counters: { overlappingTriangles, clippedTriangleCells, shortIntervals, qualifiedNormalWrites },
    pareto,
    caveat: 'continuous height is certified per vertical bin only when one clipped source-triangle interval covers that entire bin; relay may change triangles between bins. Full 3D normals use an unoriented comparison. Finite included arc endpoints and graph crossings are allowed but become unsafe finite-atlas cells. First-visible source correspondence remains a later ray gate.',
  };
}

const fixture = makeEstonianGraminoidFixture(ESTONIAN_GRAMINOID_PROFILE_IDS.CALAMAGROSTIS_CANESCENS);
const sourceMeshSha = meshHash(fixture.mesh);
if (sourceMeshSha !== EXPECTED_MESH_SHA) throw new Error(`source changed: ${sourceMeshSha}`);
const inputBytes = readFileSync(INPUT); const inputSha = sha256(inputBytes);
const input = JSON.parse(inputBytes.toString()) as any;
const triangleGroups: Record<string, number[]> = { foliage: [], reproductive: [], structure: [] };
for (const primitive of fixture.primitiveRecipes) triangleGroups[recognitionClass(primitive)]!.push(...lateralTriangles(primitive));

const recipe = {
  schema: 'laas-groundcover-relay-constructive-closed-arc-certificate/v2',
  inputMetricsSha256: inputSha,
  grid: GRID, heightBins: HEIGHT_BINS, normalBins: NORMAL_BINS,
  normalAnglesDegrees: NORMAL_ANGLES, minimumArcEdges: MINIMUM_ARC_EDGES,
  allocation: 'two strongest height-disjoint foliage fields plus two strongest height-disjoint reproductive fields for each epsilon',
  cellCertificate: 'clip source triangle in sheared-q against an inscribed cell-centre square of radius (epsilon-cellHalfDiagonal)/sqrt(2); mark only vertical bins wholly covered by one clipped interval',
  arcCertificate: 'retain 8-neighbour grid arcs only where both endpoint cells have continuous-height coverage for the transported full-3D normal bin; finite endpoints are included, so each retained graph is a closed subset and needs no artificial loop closure',
};
const toolSha = sha256(readFileSync(import.meta.filename)); const recipeSha = sha256(canonicalJson(recipe));
const output = resolve(OUTPUT_ROOT, sourceMeshSha.slice(0, 16), recipeSha.slice(0, 16), toolSha.slice(0, 16));
mkdirSync(output, { recursive: true });
const results: Record<string, unknown> = {};
for (const tolerance of TOLERANCES) {
  const fields = ['foliage', 'reproductive'].flatMap((group) =>
    input.groups[group].tolerances[String(tolerance)].heightDisjointTopFour.fields.slice(0, 2)
      .map((candidate: Candidate) => ({ group, candidate })));
  console.error(`[relay-certificate] epsilon=${tolerance}, fields=${fields.length}`);
  results[String(tolerance)] = fields.map(({ group, candidate }, index) => ({
    field: index, payloadClass: group,
    result: certifyCandidate(fixture.mesh, triangleGroups[group]!, candidate, tolerance, fixture.tile.sizeX),
  }));
}
const report = {
  ...recipe, status: 'CONSTRUCTIVE_CONSERVATIVE_SELECTED_F4_CLOSED_ARC_GATE', sourceMeshSha256: sourceMeshSha,
  pixelMetres: fixture.tile.sizeX / GRID, results,
  decisionRule: 'Any nonzero atlas-resolvable persistent arc support is a constructive survivor and advances to the exact exterior-ray truth harness. Zero support explains whether loss occurred at continuous-height/normal coverage or minimum-arc extraction.',
  distanceRule: 'No scalar nominal-width distance waiver is applied here. Later truth scoring uses the full projection Jacobian and identical anisotropic pixel footprints; coherent seams, gaps, colour loss, and shimmer remain failures.',
};
const metricsPath = resolve(output, 'metrics.json'); writeFileSync(metricsPath, `${JSON.stringify(report, null, 2)}\n`);
const index = { schema: 'laas-groundcover-relay-constructive-certificate-index/v1', tool: 'tools/groundcover-bake/certify-relay-candidates.ts', toolSha256: toolSha, sourceMeshSha256: sourceMeshSha, recipeSha256: recipeSha, metrics: 'metrics.json', metricsSha256: sha256(readFileSync(metricsPath)) };
writeFileSync(resolve(output, 'index.json'), `${JSON.stringify(index, null, 2)}\n`);
console.log(JSON.stringify({ output, ...index }, null, 2));
