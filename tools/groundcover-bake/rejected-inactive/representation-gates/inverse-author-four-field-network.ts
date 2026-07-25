/**
 * First ideal-field inverse-authoring seed for the accepted Calamagrostis.
 *
 * Each field takes one real semantic cross-section of the source and sweeps
 * that arbitrarily detailed periodic arc set through a deliberately broader
 * global height interval.  This is not a source-proximity certificate.  It is
 * a cheap, deterministic seed for the transfer-space optimization described
 * in GRASS-INVERSE-AUTHORED-FOUR-FIELD-MATH.md.
 *
 * The emitted quads are consumed only by the offline exact-ray oracle.  They
 * are not a runtime mesh proposal.
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

interface FieldRecipe {
  field: number;
  sourceClass: 'foliage' | 'reproductive';
  beta: Vec2;
  hMin: number;
  hMax: number;
  sampleH: number;
  purpose: string;
  selection?: 'slice' | 'primitive-area';
  midpointMin?: number;
  midpointMax?: number;
}

interface SheetRecipe {
  field: number;
  hMin: number;
  hMax: number;
  height: number;
  purpose: string;
}

interface Edge {
  a: Vec2;
  b: Vec2;
  color: Vec3;
  normal: Vec3;
}

interface SheetCell {
  ix: number;
  iz: number;
  color: Vec3;
  normal: Vec3;
  payload: Exclude<Payload, 'foliage'>;
}

const ROOT = resolve(import.meta.dirname, '../../../..');
const OUTPUT_ROOT = resolve(ROOT, 'data/work/groundcover-inverse-authored-four-field-network');
const EXPECTED_MESH_SHA = '37b0cf1d33f632b5cde3dd60410ad987eec2e977661d0bc804dd6f2bee7054c0';
const MINIMUM_SEGMENT_METRES = 1e-5;
const SHEET_COVERAGE_RESOLUTION = 128;
const SHEET_DITHER_SIDE = 4;

const PALETTE = [
  { rgb: [0.10, 0.27, 0.055] as Vec3, payload: 'foliage' as Payload },
  { rgb: [0.37, 0.51, 0.13] as Vec3, payload: 'foliage' as Payload },
  { rgb: [0.43, 0.53, 0.20] as Vec3, payload: 'foliage' as Payload },
  { rgb: [0.48, 0.31, 0.24] as Vec3, payload: 'reproductive-purple' as Payload },
  { rgb: [0.57, 0.35, 0.39] as Vec3, payload: 'reproductive-purple' as Payload },
  { rgb: [0.42, 0.10, 0.34] as Vec3, payload: 'reproductive-purple' as Payload },
  { rgb: [0.88, 0.80, 0.70] as Vec3, payload: 'reproductive-cream' as Payload },
] as const;

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

function vertex(mesh: IndexedMesh, index: number): Vec3 {
  const offset = index * 3;
  return [mesh.positions[offset]!, mesh.positions[offset + 1]!, mesh.positions[offset + 2]!];
}

function vertexColor(mesh: IndexedMesh, index: number): Vec3 {
  if (mesh.colors === undefined) throw new Error('inverse authoring requires vertex colours');
  const offset = index * 3;
  return [mesh.colors[offset]!, mesh.colors[offset + 1]!, mesh.colors[offset + 2]!];
}

function triangleVertices(mesh: IndexedMesh, triangle: number): readonly [Vec3, Vec3, Vec3] {
  const offset = triangle * 3;
  return [
    vertex(mesh, mesh.indices[offset]!),
    vertex(mesh, mesh.indices[offset + 1]!),
    vertex(mesh, mesh.indices[offset + 2]!),
  ];
}

function triangleColor(mesh: IndexedMesh, triangle: number): Vec3 {
  const offset = triangle * 3;
  const colors = [
    vertexColor(mesh, mesh.indices[offset]!),
    vertexColor(mesh, mesh.indices[offset + 1]!),
    vertexColor(mesh, mesh.indices[offset + 2]!),
  ];
  return [0, 1, 2].map((channel) =>
    colors.reduce((sum, color) => sum + color[channel]!, 0) / 3) as unknown as Vec3;
}

function triangleNormal(mesh: IndexedMesh, triangle: number): Vec3 {
  const [a, b, c] = triangleVertices(mesh, triangle);
  const u: Vec3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const v: Vec3 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const raw: Vec3 = [
    u[1] * v[2] - u[2] * v[1],
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0],
  ];
  const length = Math.hypot(...raw);
  return length > 1e-14 ? [raw[0] / length, raw[1] / length, raw[2] / length] : [0, 1, 0];
}

function nearestPayload(color: Vec3): Payload {
  let best = Number.POSITIVE_INFINITY;
  let payload: Payload = 'foliage';
  for (const entry of PALETTE) {
    const distance = (color[0] - entry.rgb[0]) ** 2
      + (color[1] - entry.rgb[1]) ** 2
      + (color[2] - entry.rgb[2]) ** 2;
    if (distance < best) { best = distance; payload = entry.payload; }
  }
  return payload;
}

function recognitionClass(recipe: GraminoidPrimitiveRecipe): 'foliage' | 'reproductive' | 'structure' {
  if (recipe.kind === 'blade'
    || (recipe.kind === 'tube' && ['culm', 'basal-sheath', 'upper-sheath'].includes(recipe.role))) return 'foliage';
  if (recipe.kind === 'lanceolate-surface' || recipe.kind === 'hair-filament'
    || (recipe.kind === 'tube' && ['panicle-axis', 'spikelet-axis', 'anther-filament'].includes(recipe.role))) return 'reproductive';
  return 'structure';
}

function sourceClasses(mesh: IndexedMesh, recipes: readonly GraminoidPrimitiveRecipe[]): Array<'foliage' | 'reproductive' | 'structure'> {
  const result: Array<'foliage' | 'reproductive' | 'structure'> = Array.from(
    { length: mesh.indices.length / 3 },
    () => 'structure',
  );
  for (const recipe of recipes) {
    const sourceClass = recognitionClass(recipe);
    for (let triangle = recipe.sourceTriangleStart;
      triangle < recipe.sourceTriangleStart + recipe.sourceTriangleCount;
      triangle++) result[triangle] = sourceClass;
  }
  return result;
}

interface PrimitiveBounds { hMin: number; hMax: number; }

function primitiveBounds(mesh: IndexedMesh, recipes: readonly GraminoidPrimitiveRecipe[]): PrimitiveBounds[] {
  return recipes.map((recipe) => {
    let hMin = Number.POSITIVE_INFINITY;
    let hMax = Number.NEGATIVE_INFINITY;
    for (let triangle = recipe.sourceTriangleStart;
      triangle < recipe.sourceTriangleStart + recipe.sourceTriangleCount;
      triangle++) {
      for (const point of triangleVertices(mesh, triangle)) {
        hMin = Math.min(hMin, point[1]);
        hMax = Math.max(hMax, point[1]);
      }
    }
    return { hMin, hMax };
  });
}

function stableUnit(field: number, primitiveId: number): number {
  let value = Math.imul((primitiveId + 1) ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(field + 11, 0xc2b2ae35);
  value ^= value >>> 16;
  value = Math.imul(value, 0x7feb352d);
  value ^= value >>> 15;
  return (value >>> 0) / 0x1_0000_0000;
}

function crossSection(mesh: IndexedMesh, triangle: number, h: number): readonly [Vec3, Vec3] | null {
  const vertices = triangleVertices(mesh, triangle);
  const points: Vec3[] = [];
  for (let edge = 0; edge < 3; edge++) {
    const a = vertices[edge]!;
    const b = vertices[(edge + 1) % 3]!;
    const da = a[1] - h;
    const db = b[1] - h;
    if (Math.abs(da) < 1e-12) points.push(a);
    if (da * db < 0) {
      const t = (h - a[1]) / (b[1] - a[1]);
      points.push([
        a[0] + (b[0] - a[0]) * t,
        h,
        a[2] + (b[2] - a[2]) * t,
      ]);
    }
  }
  const unique: Vec3[] = [];
  for (const point of points) {
    if (!unique.some((other) => Math.hypot(point[0] - other[0], point[2] - other[2]) < 1e-10)) unique.push(point);
  }
  if (unique.length < 2) return null;
  let winner: readonly [Vec3, Vec3] = [unique[0]!, unique[1]!];
  let winnerLength = 0;
  for (let a = 0; a < unique.length; a++) for (let b = a + 1; b < unique.length; b++) {
    const length = Math.hypot(unique[b]![0] - unique[a]![0], unique[b]![2] - unique[a]![2]);
    if (length > winnerLength) { winnerLength = length; winner = [unique[a]!, unique[b]!]; }
  }
  return winnerLength >= MINIMUM_SEGMENT_METRES ? winner : null;
}

function wrappedPhase(point: Vec3, beta: Vec2, tile: number): Vec2 {
  const wrap = (value: number): number => ((value % tile) + tile) % tile;
  return [wrap(point[0] - beta[0] * point[1]), wrap(point[2] - beta[1] * point[1])];
}

const fixture = makeEstonianGraminoidFixture(ESTONIAN_GRAMINOID_PROFILE_IDS.CALAMAGROSTIS_CANESCENS);
const sourceMeshSha256 = meshHash(fixture.mesh);
if (sourceMeshSha256 !== EXPECTED_MESH_SHA) throw new Error(`accepted Calamagrostis source changed: ${sourceMeshSha256}`);

const top = fixture.tile.topH;
const F4_RECIPES: readonly FieldRecipe[] = [
  {
    field: 0, sourceClass: 'foliage', beta: [0.2789063544616453, 0.1268529214878116],
    hMin: 0.025, hMax: Math.min(0.48, top), sampleH: 0.22,
    purpose: 'lower green blades and basal foliage',
  },
  {
    field: 1, sourceClass: 'foliage', beta: [-0.30414402714653005, 0.2984817847895657],
    hMin: 0.24, hMax: Math.min(0.88, top), sampleH: 0.56,
    purpose: 'tall culms and upper leaves',
  },
  {
    field: 2, sourceClass: 'reproductive', beta: [-0.04950770429847599, 0.19862877603080348],
    hMin: Math.min(0.72, top), hMax: top, sampleH: Math.min(0.88, top - 1e-5),
    purpose: 'lower purple/brown panicle scaffold and glumes',
  },
  {
    field: 3, sourceClass: 'reproductive', beta: [0.2789063544616453, 0.1268529214878116],
    hMin: Math.min(0.84, top), hMax: top, sampleH: Math.min(1.02, top - 1e-5),
    purpose: 'upper pale/purple micro-ribbon head population',
  },
];

const F5_RECIPES: readonly FieldRecipe[] = [
  {
    field: 0, sourceClass: 'foliage', beta: [0.29837129004682617, -0.1662328461093142],
    hMin: 0.025, hMax: Math.min(0.34, top), sampleH: 0.17,
    purpose: 'lower green blades and basal foliage',
  },
  {
    field: 1, sourceClass: 'foliage', beta: [-0.30414402714653005, 0.2984817847895657],
    hMin: 0.20, hMax: Math.min(0.62, top), sampleH: 0.40,
    purpose: 'middle foliage and leaning leaves',
  },
  {
    field: 2, sourceClass: 'foliage', beta: [0.11705731523646419, 0.4447701471929796],
    hMin: 0.43, hMax: Math.min(0.92, top), sampleH: 0.68,
    purpose: 'tall culms and upper leaves',
  },
  {
    field: 3, sourceClass: 'reproductive', beta: [-0.04950770429847599, 0.19862877603080348],
    hMin: Math.min(0.72, top), hMax: top, sampleH: Math.min(0.88, top - 1e-5),
    purpose: 'lower purple/brown panicle scaffold and glumes',
  },
  {
    field: 4, sourceClass: 'reproductive', beta: [0.2789063544616453, 0.1268529214878116],
    hMin: Math.min(0.90, top), hMax: top, sampleH: Math.min(1.04, top - 1e-5),
    purpose: 'upper pale/purple micro-ribbon head population',
  },
];

const COMPLEMENTARY_RULED_RECIPES: readonly FieldRecipe[] = [
  {
    field: 0, sourceClass: 'foliage', beta: [0.2789063544616453, 0.1268529214878116],
    hMin: 0.025, hMax: Math.min(0.52, top), sampleH: 0.24,
    purpose: 'lower/basal foliage ruled field',
  },
  {
    field: 1, sourceClass: 'foliage', beta: [-0.30414402714653005, 0.2984817847895657],
    hMin: 0.24, hMax: Math.min(0.92, top), sampleH: 0.58,
    purpose: 'tall/upper foliage ruled field',
  },
  {
    field: 2, sourceClass: 'reproductive', beta: [-0.04950770429847599, 0.19862877603080348],
    hMin: Math.min(0.72, top), hMax: Math.min(0.95, top), sampleH: 0.84,
    midpointMin: 0.72, midpointMax: 0.95, selection: 'primitive-area',
    purpose: 'area-preserving lower reproductive side/scaffold field',
  },
  {
    field: 3, sourceClass: 'reproductive', beta: [0.2789063544616453, 0.1268529214878116],
    hMin: Math.min(0.90, top), hMax: top, sampleH: Math.min(1.03, top - 1e-5),
    midpointMin: 0.95, midpointMax: top + 1e-6, selection: 'primitive-area',
    purpose: 'area-preserving upper reproductive side/scaffold field',
  },
];

const COMPLEMENTARY_F5_SHEETS: readonly SheetRecipe[] = [
  {
    field: 4, hMin: Math.min(0.74, top), hMax: top,
    height: Math.min(0.96, top - 1e-5),
    purpose: 'single reproductive top/oblique mask cost ablation',
  },
];

const COMPLEMENTARY_F6_SHEETS: readonly SheetRecipe[] = [
  {
    field: 4, hMin: Math.min(0.74, top), hMax: Math.min(0.95, top),
    height: Math.min(0.855, top - 1e-5),
    purpose: 'lower/middle reproductive top/oblique mask',
  },
  {
    field: 5, hMin: Math.min(0.95, top), hMax: top,
    height: Math.min(1.055, top - 1e-5),
    purpose: 'middle/upper reproductive top/oblique mask',
  },
];

const classes = sourceClasses(fixture.mesh, fixture.primitiveRecipes);
const boundsByPrimitive = primitiveBounds(fixture.mesh, fixture.primitiveRecipes);
const primitiveByTriangle = new Int32Array(fixture.mesh.indices.length / 3).fill(-1);
for (const recipe of fixture.primitiveRecipes) for (let triangle = recipe.sourceTriangleStart;
  triangle < recipe.sourceTriangleStart + recipe.sourceTriangleCount;
  triangle++) primitiveByTriangle[triangle] = recipe.primitiveId;
function buildVariant(recipes: readonly FieldRecipe[]): {
  entries: Array<{ field: number; network: { payload: Payload; candidate: { beta: Vec2; hMin: number; hMax: number }; edges: Edge[] } }>;
  reportFields: Array<Record<string, unknown>>;
} {
  const entries: Array<{ field: number; network: { payload: Payload; candidate: { beta: Vec2; hMin: number; hMax: number }; edges: Edge[] } }> = [];
  const reportFields: Array<Record<string, unknown>> = [];
  for (const recipe of recipes) {
    const byPayload: Record<Payload, Edge[]> = {
      foliage: [],
      'reproductive-purple': [],
      'reproductive-cream': [],
    };
    if (recipe.selection === 'primitive-area') {
      const bandWidth = recipe.hMax - recipe.hMin;
      for (const primitive of fixture.primitiveRecipes) {
        if (recognitionClass(primitive) !== recipe.sourceClass) continue;
        const primitiveBounds = boundsByPrimitive[primitive.primitiveId]!;
        const midpoint = (primitiveBounds.hMin + primitiveBounds.hMax) * 0.5;
        if (midpoint < (recipe.midpointMin ?? recipe.hMin)
          || midpoint >= (recipe.midpointMax ?? recipe.hMax)) continue;
        const keep = Math.min(1, Math.max(0, (primitiveBounds.hMax - primitiveBounds.hMin) / bandWidth));
        if (stableUnit(recipe.field, primitive.primitiveId) >= keep) continue;
        const sampleH = Math.max(primitiveBounds.hMin + 1e-8, Math.min(primitiveBounds.hMax - 1e-8, midpoint));
        for (let triangle = primitive.sourceTriangleStart;
          triangle < primitive.sourceTriangleStart + primitive.sourceTriangleCount;
          triangle++) {
          const segment = crossSection(fixture.mesh, triangle, sampleH);
          if (!segment) continue;
          const color = triangleColor(fixture.mesh, triangle);
          const payload = nearestPayload(color);
          if (payload === 'foliage') continue;
          byPayload[payload].push({
            a: wrappedPhase(segment[0], recipe.beta, fixture.tile.sizeX),
            b: wrappedPhase(segment[1], recipe.beta, fixture.tile.sizeX),
            color,
            normal: triangleNormal(fixture.mesh, triangle),
          });
        }
      }
    } else {
      for (let triangle = 0; triangle < fixture.mesh.indices.length / 3; triangle++) {
        if (classes[triangle] !== recipe.sourceClass) continue;
        const segment = crossSection(fixture.mesh, triangle, recipe.sampleH);
        if (!segment) continue;
        const color = triangleColor(fixture.mesh, triangle);
        const payload = recipe.sourceClass === 'foliage' ? 'foliage' : nearestPayload(color);
        if (payload === 'foliage' && recipe.sourceClass === 'reproductive') continue;
        byPayload[payload].push({
          a: wrappedPhase(segment[0], recipe.beta, fixture.tile.sizeX),
          b: wrappedPhase(segment[1], recipe.beta, fixture.tile.sizeX),
          color,
          normal: triangleNormal(fixture.mesh, triangle),
        });
      }
    }
    for (const payload of Object.keys(byPayload) as Payload[]) {
      if (byPayload[payload].length === 0) continue;
      entries.push({
        field: recipe.field,
        network: {
          payload,
          candidate: { beta: recipe.beta, hMin: recipe.hMin, hMax: recipe.hMax },
          edges: byPayload[payload],
        },
      });
    }
    reportFields.push({
      ...recipe,
      edgeCounts: Object.fromEntries((Object.keys(byPayload) as Payload[]).map((payload) => [payload, byPayload[payload].length])),
    });
  }
  return { entries, reportFields };
}

type Polygon = Vec2[];

function clipPolygonAxis(polygon: Polygon, axis: 0 | 1, bound: number, keepGreater: boolean): Polygon {
  const result: Polygon = [];
  for (let index = 0; index < polygon.length; index++) {
    const a = polygon[index]!;
    const b = polygon[(index + 1) % polygon.length]!;
    const aInside = keepGreater ? a[axis] >= bound : a[axis] <= bound;
    const bInside = keepGreater ? b[axis] >= bound : b[axis] <= bound;
    if (aInside) result.push(a);
    if (aInside !== bInside) {
      const t = (bound - a[axis]) / (b[axis] - a[axis]);
      result.push(axis === 0
        ? [bound, a[1] + (b[1] - a[1]) * t]
        : [a[0] + (b[0] - a[0]) * t, bound]);
    }
  }
  return result;
}

function polygonArea(polygon: readonly Vec2[]): number {
  let twice = 0;
  for (let index = 0; index < polygon.length; index++) {
    const a = polygon[index]!;
    const b = polygon[(index + 1) % polygon.length]!;
    twice += a[0] * b[1] - a[1] * b[0];
  }
  return Math.abs(twice) * 0.5;
}

function buildSheet(recipe: SheetRecipe): {
  entry: { field: number; sheet: { height: number; resolution: number; cells: SheetCell[] } };
  report: Record<string, unknown>;
} {
  const resolution = SHEET_COVERAGE_RESOLUTION;
  const tile = fixture.tile.sizeX;
  const cellSize = tile / resolution;
  const cellArea = cellSize * cellSize;
  const alpha = new Float64Array(resolution * resolution);
  const frontY = new Float64Array(resolution * resolution).fill(Number.NEGATIVE_INFINITY);
  const colors: Vec3[] = Array.from({ length: resolution * resolution }, () => [0, 0, 0] as Vec3);
  const normals: Vec3[] = Array.from({ length: resolution * resolution }, () => [0, 1, 0] as Vec3);
  const payloads: Array<Exclude<Payload, 'foliage'>> = Array.from({ length: resolution * resolution }, () => 'reproductive-purple');
  let projectedTriangles = 0;
  for (let triangle = 0; triangle < fixture.mesh.indices.length / 3; triangle++) {
    if (classes[triangle] !== 'reproductive') continue;
    const primitiveId = primitiveByTriangle[triangle]!;
    if (primitiveId < 0) continue;
    const points = triangleVertices(fixture.mesh, triangle);
    const y = (points[0][1] + points[1][1] + points[2][1]) / 3;
    if (y < recipe.hMin || y >= recipe.hMax) continue;
    const base: Vec2 = [points[0][0], points[0][2]];
    const polygon: Polygon = points.map((point) => {
      let x = point[0];
      let z = point[2];
      while (x - base[0] > tile * 0.5) x -= tile;
      while (x - base[0] < -tile * 0.5) x += tile;
      while (z - base[1] > tile * 0.5) z -= tile;
      while (z - base[1] < -tile * 0.5) z += tile;
      return [x, z] as Vec2;
    });
    if (polygonArea(polygon) <= 1e-14) continue;
    projectedTriangles++;
    const color = triangleColor(fixture.mesh, triangle);
    const payload = nearestPayload(color);
    if (payload === 'foliage') continue;
    const normal = triangleNormal(fixture.mesh, triangle);
    const minX = Math.min(...polygon.map((point) => point[0]));
    const maxX = Math.max(...polygon.map((point) => point[0]));
    const minZ = Math.min(...polygon.map((point) => point[1]));
    const maxZ = Math.max(...polygon.map((point) => point[1]));
    for (let copyZ = Math.floor(-maxZ / tile); copyZ <= Math.floor((tile - minZ) / tile); copyZ++) {
      for (let copyX = Math.floor(-maxX / tile); copyX <= Math.floor((tile - minX) / tile); copyX++) {
        const shifted = polygon.map((point) => [point[0] + copyX * tile, point[1] + copyZ * tile] as Vec2);
        const sx0 = Math.max(0, Math.floor(Math.min(...shifted.map((point) => point[0])) / cellSize));
        const sx1 = Math.min(resolution - 1, Math.floor(Math.max(...shifted.map((point) => point[0])) / cellSize));
        const sz0 = Math.max(0, Math.floor(Math.min(...shifted.map((point) => point[1])) / cellSize));
        const sz1 = Math.min(resolution - 1, Math.floor(Math.max(...shifted.map((point) => point[1])) / cellSize));
        for (let iz = sz0; iz <= sz1; iz++) for (let ix = sx0; ix <= sx1; ix++) {
          let clipped = clipPolygonAxis(shifted, 0, ix * cellSize, true);
          clipped = clipPolygonAxis(clipped, 0, (ix + 1) * cellSize, false);
          clipped = clipPolygonAxis(clipped, 1, iz * cellSize, true);
          clipped = clipPolygonAxis(clipped, 1, (iz + 1) * cellSize, false);
          const coverage = Math.min(1, polygonArea(clipped) / cellArea);
          if (!(coverage > 1e-8)) continue;
          const cell = iz * resolution + ix;
          alpha[cell] = 1 - (1 - alpha[cell]!) * (1 - coverage);
          if (y > frontY[cell]!) {
            frontY[cell] = y;
            colors[cell] = color;
            normals[cell] = normal;
            payloads[cell] = payload;
          }
        }
      }
    }
  }
  const bayer = [
    0, 8, 2, 10,
    12, 4, 14, 6,
    3, 11, 1, 9,
    15, 7, 13, 5,
  ];
  const cells: SheetCell[] = [];
  let alphaSum = 0;
  for (let iz = 0; iz < resolution; iz++) for (let ix = 0; ix < resolution; ix++) {
    const cell = iz * resolution + ix;
    const coverage = alpha[cell]!;
    alphaSum += coverage;
    const count = Math.round(coverage * SHEET_DITHER_SIDE * SHEET_DITHER_SIDE);
    for (let subZ = 0; subZ < SHEET_DITHER_SIDE; subZ++) for (let subX = 0; subX < SHEET_DITHER_SIDE; subX++) {
      if (bayer[subZ * SHEET_DITHER_SIDE + subX]! >= count) continue;
      cells.push({
        ix: ix * SHEET_DITHER_SIDE + subX,
        iz: iz * SHEET_DITHER_SIDE + subZ,
        color: colors[cell]!,
        normal: normals[cell]!,
        payload: payloads[cell]!,
      });
    }
  }
  return {
    entry: {
      field: recipe.field,
      sheet: { height: recipe.height, resolution: resolution * SHEET_DITHER_SIDE, cells },
    },
    report: {
      ...recipe,
      coverageResolution: resolution,
      ditherResolution: resolution * SHEET_DITHER_SIDE,
      projectedTriangles,
      meanCoverage: alphaSum / (resolution * resolution),
      occupiedDitherCells: cells.length,
      payloadCells: Object.fromEntries(['reproductive-purple', 'reproductive-cream'].map((payload) => [payload, cells.filter((cell) => cell.payload === payload).length])),
    },
  };
}

function buildComplementaryVariant(sheets: readonly SheetRecipe[]): {
  entries: Array<unknown>;
  reportFields: Array<Record<string, unknown>>;
} {
  const ruled = buildVariant(COMPLEMENTARY_RULED_RECIPES);
  const bakedSheets = sheets.map(buildSheet);
  return {
    entries: [...ruled.entries.map((entry) => ({ ...entry, kind: 'ruled' })), ...bakedSheets.map((sheet) => ({ ...sheet.entry, kind: 'sheet' }))],
    reportFields: [...ruled.reportFields, ...bakedSheets.map((sheet) => sheet.report)],
  };
}

const f4 = buildVariant(F4_RECIPES);
const f5 = buildVariant(F5_RECIPES);
const complementaryF5 = buildComplementaryVariant(COMPLEMENTARY_F5_SHEETS);
const complementaryF6 = buildComplementaryVariant(COMPLEMENTARY_F6_SHEETS);

const configuration = {
  schema: 'laas-groundcover-inverse-authored-four-field-network/v1',
  sourceMeshSha256,
  source: fixture.species,
  variants: {
    F4: F4_RECIPES,
    F5: F5_RECIPES,
    COMPLEMENTARY_F5: { ruled: COMPLEMENTARY_RULED_RECIPES, sheets: COMPLEMENTARY_F5_SHEETS },
    COMPLEMENTARY_F6: { ruled: COMPLEMENTARY_RULED_RECIPES, sheets: COMPLEMENTARY_F6_SHEETS },
  },
  method: 'one exact semantic source cross-section per field, swept through a broader global interval; transfer-space seed, not pointwise source certificate',
  runtimeMeaning: 'offline continuous-field oracle input only; emitted quads are not runtime geometry',
};
const recipeSha256 = sha256(canonicalJson(configuration));
const toolSha256 = sha256(readFileSync(import.meta.filename));
const output = resolve(OUTPUT_ROOT, sourceMeshSha256.slice(0, 16), recipeSha256.slice(0, 16), toolSha256.slice(0, 16));
mkdirSync(output, { recursive: true });
const network = {
  ...configuration,
  variants: {
    F4: f4.entries,
    F5: f5.entries,
    COMPLEMENTARY_F5: complementaryF5.entries,
    COMPLEMENTARY_F6: complementaryF6.entries,
  },
  fields: {
    F4: f4.reportFields,
    F5: f5.reportFields,
    COMPLEMENTARY_F5: complementaryF5.reportFields,
    COMPLEMENTARY_F6: complementaryF6.reportFields,
  },
};
const networkPath = resolve(output, 'network.json');
writeFileSync(networkPath, `${JSON.stringify(network, null, 2)}\n`);
const index = {
  schema: 'laas-groundcover-inverse-authored-four-field-network-index/v1',
  tool: 'tools/groundcover-bake/inverse-author-four-field-network.ts',
  toolSha256,
  sourceMeshSha256,
  recipeSha256,
  network: 'network.json',
  networkSha256: sha256(readFileSync(networkPath)),
};
writeFileSync(resolve(output, 'index.json'), `${JSON.stringify(index, null, 2)}\n`);
console.log(JSON.stringify({ output, ...index, fields: network.fields }, null, 2));
