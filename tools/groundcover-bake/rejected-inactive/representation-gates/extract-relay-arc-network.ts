/**
 * Offline-only extraction of a non-redundant arc network from the conservative
 * relay-cell certificate.
 *
 * This deliberately does not count every compatible edge in a safe grid
 * region.  Each phase cell selects one transported orientation, components are
 * straight degree-two paths, and emitted sheet area is charged against the
 * exact, height-clipped area of the source triangles that supplied the
 * continuous-height certificate.  Foliage, purple/brown reproductive content,
 * and cream reproductive content are categorical arc payloads.
 */

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
type Payload = 'foliage' | 'reproductive-purple' | 'reproductive-cream';

interface Candidate { beta: Vec2; hMin: number; hMax: number; }
interface PolyVertex { qx: number; qz: number; h: number; }
interface WorldVertex { x: number; y: number; z: number; }
interface Orientation { dx: number; dz: number; normalBin: number; length: number; }
interface EdgeDemand { triangleId: number; area: number; }
interface NetworkEdge {
  aCell: number;
  bCell: number;
  a: Vec2;
  b: Vec2;
  sourceDemand: EdgeDemand[];
  color: Vec3;
  normal: Vec3;
  lengthMetres: number;
  sheetAreaMetres2: number;
}

const REPO_ROOT = resolve(import.meta.dirname, '../../../..');
const INPUT = resolve(REPO_ROOT, 'data/work/groundcover-relay-corridor-gate/37b0cf1d33f632b5/68750bca57259e7c/0234934c9a4f87a6/metrics.json');
const OUTPUT_ROOT = resolve(REPO_ROOT, 'data/work/groundcover-relay-arc-network');
const EXPECTED_MESH_SHA = '37b0cf1d33f632b5cde3dd60410ad987eec2e977661d0bc804dd6f2bee7054c0';
const GRID = 256;
const HEIGHT_BINS = 64;
const NORMAL_BINS = 16;
const TOLERANCE = 0.01;
const NORMAL_ANGLES = [30, 45] as const;
const MINIMUM_ARC_EDGES = 8;
const EPSILON = 1e-12;
const ORIENTATIONS: readonly Orientation[] = [
  { dx: 1, dz: 0, normalBin: 4, length: 1 },
  { dx: 1, dz: 1, normalBin: 6, length: Math.SQRT2 },
  { dx: 0, dz: 1, normalBin: 8, length: 1 },
  { dx: -1, dz: 1, normalBin: 10, length: Math.SQRT2 },
];
const PALETTE = [
  { rgb: [0.10, 0.27, 0.055] as Vec3, payload: 'foliage' as Payload },
  { rgb: [0.37, 0.51, 0.13] as Vec3, payload: 'foliage' as Payload },
  { rgb: [0.43, 0.53, 0.20] as Vec3, payload: 'foliage' as Payload },
  { rgb: [0.48, 0.31, 0.24] as Vec3, payload: 'reproductive-purple' as Payload },
  { rgb: [0.57, 0.35, 0.39] as Vec3, payload: 'reproductive-purple' as Payload },
  { rgb: [0.42, 0.10, 0.34] as Vec3, payload: 'reproductive-purple' as Payload },
  { rgb: [0.88, 0.80, 0.70] as Vec3, payload: 'reproductive-cream' as Payload },
] as const;

function sha256(value: Uint8Array | string): string { return createHash('sha256').update(value).digest('hex'); }
function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(',')}}`;
}
function meshHash(mesh: IndexedMesh): string { return sha256(JSON.stringify(mesh)); }
function wrap(value: number): number { return ((value % GRID) + GRID) % GRID; }
function cellX(cell: number): number { return cell % GRID; }
function cellZ(cell: number): number { return Math.floor(cell / GRID); }

function recognitionClass(recipe: GraminoidPrimitiveRecipe): 'foliage' | 'reproductive' | 'structure' {
  if (recipe.kind === 'blade' || (recipe.kind === 'tube' && ['culm', 'basal-sheath', 'upper-sheath'].includes(recipe.role))) return 'foliage';
  if (recipe.kind === 'lanceolate-surface' || (recipe.kind === 'tube' && ['panicle-axis', 'spikelet-axis', 'anther-filament'].includes(recipe.role))) return 'reproductive';
  return 'structure';
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

function vertex(mesh: IndexedMesh, index: number): Vec3 {
  const offset = index * 3;
  return [mesh.positions[offset]!, mesh.positions[offset + 1]!, mesh.positions[offset + 2]!];
}
function vertexColor(mesh: IndexedMesh, index: number): Vec3 {
  const colors = mesh.colors;
  if (!colors) return [1, 1, 1];
  const offset = index * 3;
  return [colors[offset]!, colors[offset + 1]!, colors[offset + 2]!];
}
function triangleVertices(mesh: IndexedMesh, triangleIndex: number): readonly [Vec3, Vec3, Vec3] {
  const offset = triangleIndex * 3;
  return [vertex(mesh, mesh.indices[offset]!), vertex(mesh, mesh.indices[offset + 1]!), vertex(mesh, mesh.indices[offset + 2]!)];
}
function triangleColor(mesh: IndexedMesh, triangleIndex: number): Vec3 {
  const offset = triangleIndex * 3;
  const colors = [
    vertexColor(mesh, mesh.indices[offset]!),
    vertexColor(mesh, mesh.indices[offset + 1]!),
    vertexColor(mesh, mesh.indices[offset + 2]!),
  ];
  return [0, 1, 2].map((channel) => colors.reduce((sum, color) => sum + color[channel]!, 0) / 3) as unknown as Vec3;
}
function nearestPayload(color: Vec3): Payload {
  let winner = PALETTE[0]!;
  let best = Number.POSITIVE_INFINITY;
  for (const entry of PALETTE) {
    const distance = (color[0] - entry.rgb[0]) ** 2 + (color[1] - entry.rgb[1]) ** 2 + (color[2] - entry.rgb[2]) ** 2;
    if (distance < best) { best = distance; winner = entry; }
  }
  return winner.payload;
}

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
function clipWorldY(poly: WorldVertex[], bound: number, greater: boolean): WorldVertex[] {
  const result: WorldVertex[] = [];
  for (let index = 0; index < poly.length; index++) {
    const a = poly[index]!; const b = poly[(index + 1) % poly.length]!;
    const ain = greater ? a.y >= bound : a.y <= bound;
    const bin = greater ? b.y >= bound : b.y <= bound;
    if (ain) result.push(a);
    if (ain !== bin) {
      const t = (bound - a.y) / (b.y - a.y);
      result.push({ x: a.x + (b.x - a.x) * t, y: bound, z: a.z + (b.z - a.z) * t });
    }
  }
  return result;
}
function triangleArea(a: WorldVertex, b: WorldVertex, c: WorldVertex): number {
  const ux = b.x - a.x; const uy = b.y - a.y; const uz = b.z - a.z;
  const vx = c.x - a.x; const vy = c.y - a.y; const vz = c.z - a.z;
  return 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
}
function clippedTriangleArea(mesh: IndexedMesh, triangleIndex: number, hMin: number, hMax: number): number {
  let poly = triangleVertices(mesh, triangleIndex).map(([x, y, z]) => ({ x, y, z }));
  poly = clipWorldY(poly, hMin, true);
  if (poly.length) poly = clipWorldY(poly, hMax, false);
  if (poly.length < 3) return 0;
  let area = 0;
  for (let index = 1; index + 1 < poly.length; index++) area += triangleArea(poly[0]!, poly[index]!, poly[index + 1]!);
  return area;
}

interface SourceTriangle {
  triangleId: number;
  poly: PolyVertex[];
  normal: Vec3;
  minH: number;
  maxH: number;
  color: Vec3;
  clippedArea: number;
}
function sourceTriangle(mesh: IndexedMesh, triangleIndex: number, beta: Vec2, tile: number, candidate: Candidate): SourceTriangle {
  const world = triangleVertices(mesh, triangleIndex);
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
  return {
    triangleId: triangleIndex,
    poly,
    normal: magnitude > EPSILON ? [raw[0] / magnitude, raw[1] / magnitude, raw[2] / magnitude] : [0, 1, 0],
    minH: Math.min(...poly.map((point) => point.h)),
    maxH: Math.max(...poly.map((point) => point.h)),
    color: triangleColor(mesh, triangleIndex),
    clippedArea: clippedTriangleArea(mesh, triangleIndex, candidate.hMin, candidate.hMax),
  };
}

function templateNormal(beta: Vec2, normalBin: number): Vec3 {
  const angle = 2 * Math.PI * normalBin / NORMAL_BINS;
  const nx = Math.cos(angle); const nz = Math.sin(angle);
  const raw: Vec3 = [nx, -(nx * beta[0] + nz * beta[1]), nz];
  const magnitude = Math.hypot(...raw);
  return [raw[0] / magnitude, raw[1] / magnitude, raw[2] / magnitude];
}
function angularErrorDegrees(a: Vec3, b: Vec3): number {
  return Math.acos(Math.max(-1, Math.min(1, Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])))) * 180 / Math.PI;
}
function setBinRange(target: Uint32Array, offset: number, first: number, last: number): void {
  for (let bin = first; bin <= last; bin++) target[offset + (bin >>> 5)] |= (1 << (bin & 31)) >>> 0;
}
function fullCoverage(target: Uint32Array, offset: number, words: number): boolean {
  for (let word = 0; word < words; word++) {
    const expected = word === words - 1 && HEIGHT_BINS % 32 ? (2 ** (HEIGHT_BINS % 32) - 1) >>> 0 : 0xffffffff;
    if (target[offset + word] !== expected) return false;
  }
  return true;
}
function qAtCell(cell: number, pixel: number): Vec2 { return [(cellX(cell) + 0.5) * pixel, (cellZ(cell) + 0.5) * pixel]; }
function quantiles(valuesInput: readonly number[]): Record<string, number | null> {
  if (valuesInput.length === 0) return { p50: null, p95: null, maximum: null };
  const values = [...valuesInput].sort((a, b) => a - b);
  return {
    p50: values[Math.floor((values.length - 1) * 0.5)]!,
    p95: values[Math.floor((values.length - 1) * 0.95)]!,
    maximum: values.at(-1)!,
  };
}

function certifyNetwork(
  mesh: IndexedMesh,
  triangleIds: readonly number[],
  candidate: Candidate,
  tile: number,
  payload: Payload,
  normalAngle: number,
): Record<string, unknown> & { edges: NetworkEdge[] } {
  const pixel = tile / GRID;
  const cellRadius = Math.SQRT2 * pixel * 0.5;
  const sourceReach = TOLERANCE - cellRadius;
  if (!(sourceReach > 0)) throw new Error('epsilon cannot certify a phase cell');
  const clipHalfWidth = sourceReach / Math.SQRT2;
  const words = Math.ceil(HEIGHT_BINS / 32);
  const coverage = new Uint32Array(GRID * GRID * ORIENTATIONS.length * words);
  const scoreSum = new Float64Array(GRID * GRID * ORIENTATIONS.length);
  const scoreCount = new Uint32Array(GRID * GRID * ORIENTATIONS.length);
  const binHeight = (candidate.hMax - candidate.hMin) / HEIGHT_BINS;
  const sources = triangleIds.map((triangleId) => sourceTriangle(mesh, triangleId, candidate.beta, tile, candidate))
    .filter((source) => source.maxH >= candidate.hMin - EPSILON && source.minH <= candidate.hMax + EPSILON && source.clippedArea > 0);

  const forEachClippedCell = (source: SourceTriangle, callback: (cell: number, first: number, last: number) => void): void => {
    const minQx = Math.min(...source.poly.map((point) => point.qx)) - clipHalfWidth;
    const maxQx = Math.max(...source.poly.map((point) => point.qx)) + clipHalfWidth;
    const minQz = Math.min(...source.poly.map((point) => point.qz)) - clipHalfWidth;
    const maxQz = Math.max(...source.poly.map((point) => point.qz)) + clipHalfWidth;
    for (let iz = Math.floor(minQz / pixel); iz <= Math.floor(maxQz / pixel); iz++) for (let ix = Math.floor(minQx / pixel); ix <= Math.floor(maxQx / pixel); ix++) {
      const clipped = clipToCell(source.poly, (ix + 0.5) * pixel, (iz + 0.5) * pixel, clipHalfWidth);
      if (clipped.length === 0) continue;
      const low = Math.max(candidate.hMin, Math.min(...clipped.map((point) => point.h)));
      const high = Math.min(candidate.hMax, Math.max(...clipped.map((point) => point.h)));
      const first = Math.max(0, Math.ceil((low - candidate.hMin) / binHeight - 1e-10));
      const last = Math.min(HEIGHT_BINS - 1, Math.floor((high - candidate.hMin) / binHeight + 1e-10) - 1);
      if (last >= first) callback(wrap(iz) * GRID + wrap(ix), first, last);
    }
  };

  for (const source of sources) {
    const errors = ORIENTATIONS.map((orientation) => angularErrorDegrees(source.normal, templateNormal(candidate.beta, orientation.normalBin)));
    forEachClippedCell(source, (cell, first, last) => {
      for (let orientation = 0; orientation < ORIENTATIONS.length; orientation++) {
        if (errors[orientation]! > normalAngle) continue;
        const pair = cell * ORIENTATIONS.length + orientation;
        setBinRange(coverage, pair * words, first, last);
        const bins = last - first + 1;
        scoreSum[pair] += errors[orientation]! * bins;
        scoreCount[pair] += bins;
      }
    });
  }

  const safe = new Uint8Array(GRID * GRID * ORIENTATIONS.length);
  let safeCellOrientationPairs = 0;
  for (let cell = 0; cell < GRID * GRID; cell++) {
    for (let orientation = 0; orientation < ORIENTATIONS.length; orientation++) {
      const pair = cell * ORIENTATIONS.length + orientation;
      if (!fullCoverage(coverage, pair * words, words)) continue;
      safe[pair] = 1; safeCellOrientationPairs++;
    }
  }

  interface RawEdge { a: number; b: number; orientation: number; lengthMetres: number; score: number; }
  const compatibleEdges: RawEdge[] = [];
  for (let cell = 0; cell < GRID * GRID; cell++) for (let orientation = 0; orientation < ORIENTATIONS.length; orientation++) {
    const pair = cell * ORIENTATIONS.length + orientation;
    if (!safe[pair]) continue;
    const direction = ORIENTATIONS[orientation]!;
    const neighbour = wrap(cellZ(cell) + direction.dz) * GRID + wrap(cellX(cell) + direction.dx);
    const neighbourPair = neighbour * ORIENTATIONS.length + orientation;
    if (!safe[neighbourPair]) continue;
    const aScore = scoreCount[pair] ? scoreSum[pair] / scoreCount[pair] : normalAngle;
    const bScore = scoreCount[neighbourPair] ? scoreSum[neighbourPair] / scoreCount[neighbourPair] : normalAngle;
    compatibleEdges.push({ a: cell, b: neighbour, orientation, lengthMetres: direction.length * pixel, score: (aScore + bScore) * 0.5 });
  }
  // Jointly assemble the straight, orientation-certified edges into an
  // acyclic degree-two path cover.  This allows a botanical arc to bend at a
  // cell while still forbidding a branching fill or crossing at that cell.
  compatibleEdges.sort((left, right) => left.score - right.score || left.a - right.a || left.b - right.b || left.orientation - right.orientation);
  const parent = new Int32Array(GRID * GRID); parent.fill(-1);
  const degree = new Uint8Array(GRID * GRID);
  const incidentOrientation = new Int8Array(GRID * GRID); incidentOrientation.fill(-1);
  const smoothTurn = (cell: number, orientation: number): boolean => {
    if (degree[cell] === 0) return true;
    const previous = ORIENTATIONS[incidentOrientation[cell]!]!.normalBin;
    const next = ORIENTATIONS[orientation]!.normalBin;
    const raw = Math.abs(previous - next) % (NORMAL_BINS / 2);
    return Math.min(raw, NORMAL_BINS / 2 - raw) <= 2; // at most 45 degrees, unoriented
  };
  const find = (input: number): number => {
    let node = input;
    while (parent[node] !== node) node = parent[node]!;
    let cursor = input;
    while (parent[cursor] !== cursor) { const next = parent[cursor]!; parent[cursor] = node; cursor = next; }
    return node;
  };
  const rawEdges: RawEdge[] = [];
  for (const edge of compatibleEdges) {
    if (degree[edge.a] >= 2 || degree[edge.b] >= 2) continue;
    if (!smoothTurn(edge.a, edge.orientation) || !smoothTurn(edge.b, edge.orientation)) continue;
    if (parent[edge.a] < 0) parent[edge.a] = edge.a;
    if (parent[edge.b] < 0) parent[edge.b] = edge.b;
    const aRoot = find(edge.a); const bRoot = find(edge.b);
    if (aRoot === bRoot) continue;
    parent[bRoot] = aRoot;
    if (degree[edge.a] === 0) incidentOrientation[edge.a] = edge.orientation;
    if (degree[edge.b] === 0) incidentOrientation[edge.b] = edge.orientation;
    degree[edge.a]++; degree[edge.b]++; rawEdges.push(edge);
  }
  const adjacency = new Map<number, number[]>();
  for (let edge = 0; edge < rawEdges.length; edge++) {
    const item = rawEdges[edge]!;
    const aa = adjacency.get(item.a) ?? []; aa.push(edge); adjacency.set(item.a, aa);
    const bb = adjacency.get(item.b) ?? []; bb.push(edge); adjacency.set(item.b, bb);
  }

  const visited = new Uint8Array(rawEdges.length);
  const components: number[][] = [];
  const walk = (startEdge: number, startCell: number): number[] => {
    const result: number[] = [];
    let edgeIndex = startEdge; let current = startCell;
    while (!visited[edgeIndex]) {
      visited[edgeIndex] = 1; result.push(edgeIndex);
      const edge = rawEdges[edgeIndex]!;
      current = edge.a === current ? edge.b : edge.a;
      const next = (adjacency.get(current) ?? []).find((candidateEdge) => !visited[candidateEdge]);
      if (next === undefined) break;
      edgeIndex = next;
    }
    return result;
  };
  for (const [cell, edges] of adjacency) {
    if (edges.length !== 1 || visited[edges[0]!]) continue;
    components.push(walk(edges[0]!, cell));
  }
  for (let edge = 0; edge < rawEdges.length; edge++) if (!visited[edge]) components.push(walk(edge, rawEdges[edge]!.a));
  components.sort((a, b) => {
    const aScore = a.reduce((sum, edge) => sum + rawEdges[edge]!.score, 0) / a.length;
    const bScore = b.reduce((sum, edge) => sum + rawEdges[edge]!.score, 0) / b.length;
    return aScore - bScore || b.length - a.length || a[0]! - b[0]!;
  });

  const requiredPairIndex = new Int32Array(GRID * GRID * ORIENTATIONS.length); requiredPairIndex.fill(-1);
  let requiredPairs = 0;
  for (const component of components) {
    if (component.length < MINIMUM_ARC_EDGES) continue;
    for (const edgeIndex of component) {
      const edge = rawEdges[edgeIndex]!;
      for (const cell of [edge.a, edge.b]) {
        const pair = cell * ORIENTATIONS.length + edge.orientation;
        if (requiredPairIndex[pair] < 0) requiredPairIndex[pair] = requiredPairs++;
      }
    }
  }
  const owner = new Int32Array(requiredPairs * HEIGHT_BINS); owner.fill(-1);
  const ownerError = new Float32Array(owner.length); ownerError.fill(Number.POSITIVE_INFINITY);
  for (const source of sources) {
    const errors = ORIENTATIONS.map((orientation) => angularErrorDegrees(source.normal, templateNormal(candidate.beta, orientation.normalBin)));
    forEachClippedCell(source, (cell, first, last) => {
      for (let orientation = 0; orientation < ORIENTATIONS.length; orientation++) {
        const compactPair = requiredPairIndex[cell * ORIENTATIONS.length + orientation]!;
        if (compactPair < 0 || errors[orientation]! > normalAngle) continue;
        for (let bin = first; bin <= last; bin++) {
          const address = compactPair * HEIGHT_BINS + bin;
          if (errors[orientation]! < ownerError[address] || (errors[orientation] === ownerError[address] && source.triangleId < owner[address]!)) {
            ownerError[address] = errors[orientation]!;
            owner[address] = source.triangleId;
          }
        }
      }
    });
  }

  const capacities = new Map(sources.map((source) => [source.triangleId, source.clippedArea]));
  const used = new Map<number, number>();
  const usedSourceTriangles = new Set<number>();
  const sourceColors = new Map(sources.map((source) => [source.triangleId, source.color]));
  const accepted: NetworkEdge[] = [];
  const rejectedCapacity: number[] = [];
  const edgeDemands = (edge: RawEdge): EdgeDemand[] | null => {
    const demand = new Map<number, number>();
    const orientation = ORIENTATIONS[edge.orientation]!;
    const angle = 2 * Math.PI * orientation.normalBin / NORMAL_BINS;
    const areaJacobian = Math.hypot(1, candidate.beta[0] * Math.cos(angle) + candidate.beta[1] * Math.sin(angle));
    const sliceArea = edge.lengthMetres * (candidate.hMax - candidate.hMin) * areaJacobian / HEIGHT_BINS;
    const aPair = requiredPairIndex[edge.a * ORIENTATIONS.length + edge.orientation]!;
    const bPair = requiredPairIndex[edge.b * ORIENTATIONS.length + edge.orientation]!;
    if (aPair < 0 || bPair < 0) return null;
    for (let bin = 0; bin < HEIGHT_BINS; bin++) {
      const a = owner[aPair * HEIGHT_BINS + bin]!;
      const b = owner[bPair * HEIGHT_BINS + bin]!;
      if (a < 0 || b < 0) return null;
      if (a === b) demand.set(a, (demand.get(a) ?? 0) + sliceArea);
      else {
        demand.set(a, (demand.get(a) ?? 0) + sliceArea * 0.5);
        demand.set(b, (demand.get(b) ?? 0) + sliceArea * 0.5);
      }
    }
    return [...demand].map(([triangleId, area]) => ({ triangleId, area }));
  };
  let emittedArea = 0;
  const payloadBandSourceArea = sources.reduce((sum, source) => sum + source.clippedArea, 0);
  for (const component of components) {
    if (component.length < MINIMUM_ARC_EDGES) continue;
    const path = component.map((edgeIndex) => ({ edge: rawEdges[edgeIndex]!, demands: edgeDemands(rawEdges[edgeIndex]!) }));
    if (path.some((item) => item.demands === null)) { rejectedCapacity.push(component[0]!); continue; }
    const pathArea = path.reduce((sum, item) => sum + item.demands!.reduce((area, demand) => area + demand.area, 0), 0);
    if (emittedArea + pathArea > payloadBandSourceArea + 1e-14) { rejectedCapacity.push(component[0]!); continue; }
    emittedArea += pathArea;
    for (const item of path) {
      const demands = item.demands!;
      for (const demand of demands) {
        usedSourceTriangles.add(demand.triangleId);
        used.set(demand.triangleId, (used.get(demand.triangleId) ?? 0) + demand.area);
      }
      const total = demands.reduce((sum, demand) => sum + demand.area, 0);
      const color = demands.reduce<Vec3>((sum, demand) => {
        const source = sourceColors.get(demand.triangleId)!;
        return [sum[0] + source[0] * demand.area / total, sum[1] + source[1] * demand.area / total, sum[2] + source[2] * demand.area / total];
      }, [0, 0, 0]);
      const orientation = ORIENTATIONS[item.edge.orientation]!;
      accepted.push({
        aCell: item.edge.a, bCell: item.edge.b,
        a: qAtCell(item.edge.a, pixel), b: qAtCell(item.edge.b, pixel),
        sourceDemand: demands, color,
        normal: templateNormal(candidate.beta, orientation.normalBin),
        lengthMetres: item.edge.lengthMetres,
        sheetAreaMetres2: demands.reduce((sum, demand) => sum + demand.area, 0),
      });
    }
  }

  const usedEntries = [...used].filter(([, area]) => area > 1e-14);
  const emittedAreaFromEdges = accepted.reduce((sum, edge) => sum + edge.sheetAreaMetres2, 0);
  if (Math.abs(emittedAreaFromEdges - emittedArea) > 1e-10) throw new Error('ruled-sheet area accounting drifted');
  const deduplicatedSourceArea = usedEntries.reduce((sum, [triangleId]) => sum + (capacities.get(triangleId) ?? 0), 0);
  const chargedSourceArea = usedEntries.reduce((sum, [, area]) => sum + area, 0);
  const colorErrors: number[] = [];
  for (const edge of accepted) for (const demand of edge.sourceDemand) {
    const source = sourceColors.get(demand.triangleId)!;
    const error = Math.max(Math.abs(source[0] - edge.color[0]), Math.abs(source[1] - edge.color[1]), Math.abs(source[2] - edge.color[2]));
    colorErrors.push(error);
  }
  return {
    payload, candidate, normalAngleDegrees: normalAngle,
    cellHalfDiagonalMetres: cellRadius,
    sourceTriangleCount: sources.length,
    sourceAreaInBandMetres2: payloadBandSourceArea,
    safeCellOrientationPairs,
    selectedCells: new Set(rawEdges.flatMap((edge) => [edge.a, edge.b])).size,
    rawCompatibleEdges: compatibleEdges.length,
    jointDegreeTwoPathCoverEdges: rawEdges.length,
    rawComponents: components.length,
    rejectedPathsByPayloadBandSourceAreaCapacity: rejectedCapacity.length,
    acceptedEdges: accepted.length,
    acceptedArcComponentsMinimumEdges: MINIMUM_ARC_EDGES,
    emittedSheetAreaMetres2: emittedAreaFromEdges,
    payloadBandDensityBudgetMetres2: payloadBandSourceArea,
    chargedDeduplicatedSourceAreaMetres2: chargedSourceArea,
    availableDeduplicatedCorrespondingSourceAreaMetres2: deduplicatedSourceArea,
    uniqueCorrespondingSourceTriangles: usedEntries.length,
    fixedArcColorFitMaxChannel: quantiles(colorErrors),
    crossingPriority: 'field index, then payload order foliage < reproductive-purple < reproductive-cream, then emitted edge index; geometry depth wins except exact ties',
    edges: accepted,
  };
}

const fixture = makeEstonianGraminoidFixture(ESTONIAN_GRAMINOID_PROFILE_IDS.CALAMAGROSTIS_CANESCENS);
const sourceMeshSha256 = meshHash(fixture.mesh);
if (sourceMeshSha256 !== EXPECTED_MESH_SHA) throw new Error(`source changed: ${sourceMeshSha256}`);
const inputBytes = readFileSync(INPUT); const inputSha256 = sha256(inputBytes); const input = JSON.parse(inputBytes.toString()) as any;
const triangleGroups: Record<Payload, number[]> = { foliage: [], 'reproductive-purple': [], 'reproductive-cream': [] };
for (const primitive of fixture.primitiveRecipes) {
  const recognition = recognitionClass(primitive);
  if (recognition === 'structure') continue;
  for (const triangleId of lateralTriangles(primitive)) {
    const payload = recognition === 'foliage' ? 'foliage' : nearestPayload(triangleColor(fixture.mesh, triangleId));
    triangleGroups[payload].push(triangleId);
  }
}

const fields: Array<{ candidateSlot: number; sourceGroup: 'foliage' | 'reproductive'; candidate: Candidate; payloads: Payload[] }> = [];
for (const [candidateSlot, candidate] of input.groups.foliage.tolerances[String(TOLERANCE)].heightDisjointTopFour.fields.entries()) {
  fields.push({ candidateSlot, sourceGroup: 'foliage', candidate, payloads: ['foliage'] });
}
for (const [candidateSlot, candidate] of input.groups.reproductive.tolerances[String(TOLERANCE)].heightDisjointTopFour.fields.entries()) {
  fields.push({ candidateSlot, sourceGroup: 'reproductive', candidate, payloads: ['reproductive-purple', 'reproductive-cream'] });
}

const recipe = {
  schema: 'laas-groundcover-relay-nonredundant-arc-network/v1',
  sourceMeshSha256, inputSha256, toleranceMetres: TOLERANCE,
  grid: GRID, heightBins: HEIGHT_BINS, normalBins: NORMAL_BINS,
  normalAnglesDegrees: NORMAL_ANGLES, minimumArcEdges: MINIMUM_ARC_EDGES,
  allocation: 'strict-score F=4 diagnosis: certify all four height-disjoint candidates per source group, then select the two greatest emitted certified path areas per group; reproductive arcs are marked purple/brown or cream without adding fields',
  pathRule: 'straight orientation-certified edges are jointly assembled by minimum normal error into an acyclic degree-two path cover; piecewise turns are at most 45 degrees; branches/crossings at a cell and short fragments are discarded',
  areaRule: 'path-level source correspondence is deduplicated for reporting; total emitted ruled-sheet area uses sqrt(1+(beta dot n)^2) and may not exceed the exact height-clipped source area pool of its categorical payload and field band',
  colorRule: 'one area-weighted fixed RGB record per arc edge; foliage, purple/brown reproductive, and cream reproductive supports are certified separately',
};
const toolSha256 = sha256(readFileSync(import.meta.filename)); const recipeSha256 = sha256(canonicalJson(recipe));
const output = resolve(OUTPUT_ROOT, sourceMeshSha256.slice(0, 16), recipeSha256.slice(0, 16), toolSha256.slice(0, 16));
mkdirSync(output, { recursive: true });
const variants: Record<string, unknown> = {};
const evaluatedVariants: Record<string, unknown> = {};
for (const normalAngle of NORMAL_ANGLES) {
  const reports: any[] = [];
  for (const field of fields) for (const payload of field.payloads) {
    console.error(`[relay-network] angle=${normalAngle} group=${field.sourceGroup} slot=${field.candidateSlot} payload=${payload}`);
    reports.push({
      candidateSlot: field.candidateSlot, sourceGroup: field.sourceGroup,
      network: certifyNetwork(fixture.mesh, triangleGroups[payload], field.candidate, fixture.tile.sizeX, payload, normalAngle),
    });
  }
  const selectedSlots: Record<'foliage' | 'reproductive', number[]> = { foliage: [], reproductive: [] };
  for (const sourceGroup of ['foliage', 'reproductive'] as const) {
    const scores = [...new Set(reports.filter((entry) => entry.sourceGroup === sourceGroup).map((entry) => entry.candidateSlot))]
      .map((candidateSlot) => ({
        candidateSlot,
        emittedArea: reports.filter((entry) => entry.sourceGroup === sourceGroup && entry.candidateSlot === candidateSlot)
          .reduce((sum, entry) => sum + Number(entry.network.emittedSheetAreaMetres2), 0),
      }))
      .sort((left, right) => right.emittedArea - left.emittedArea || left.candidateSlot - right.candidateSlot);
    selectedSlots[sourceGroup] = scores.slice(0, 2).map((entry) => entry.candidateSlot);
  }
  const selected = reports.filter((entry) => selectedSlots[entry.sourceGroup as 'foliage' | 'reproductive'].includes(entry.candidateSlot))
    .map((entry) => ({
      ...entry,
      field: (entry.sourceGroup === 'foliage' ? 0 : 2) + selectedSlots[entry.sourceGroup as 'foliage' | 'reproductive'].indexOf(entry.candidateSlot),
    }))
    .sort((left, right) => left.field - right.field || String(left.network.payload).localeCompare(String(right.network.payload)));
  evaluatedVariants[String(normalAngle)] = reports;
  variants[String(normalAngle)] = selected;
}
const report = {
  ...recipe,
  source: { species: fixture.species, triangleCount: fixture.mesh.indices.length / 3, tile: fixture.tile },
  payloadSourceTriangles: Object.fromEntries(Object.entries(triangleGroups).map(([payload, triangles]) => [payload, triangles.length])),
  variants, evaluatedVariants,
  interpretation: [
    'This is an emitted arc network, not the superseded all-compatible-edge support proxy.',
    'Emitted sheet area and deduplicated corresponding source area are separate; neither is named retained first-visible area.',
    'The next exterior-ray gate measures first-visible coverage directly through identical perspective pixel ray bundles.',
  ],
};
const metricsPath = resolve(output, 'network.json'); writeFileSync(metricsPath, `${JSON.stringify(report, null, 2)}\n`);
const index = { schema: 'laas-groundcover-relay-nonredundant-arc-network-index/v1', tool: 'tools/groundcover-bake/extract-relay-arc-network.ts', toolSha256, sourceMeshSha256, recipeSha256, network: 'network.json', networkSha256: sha256(readFileSync(metricsPath)) };
writeFileSync(resolve(output, 'index.json'), `${JSON.stringify(index, null, 2)}\n`);
console.log(JSON.stringify({ output, ...index }, null, 2));
