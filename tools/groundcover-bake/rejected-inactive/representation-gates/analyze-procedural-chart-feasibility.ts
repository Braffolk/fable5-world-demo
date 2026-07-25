/**
 * Offline-only feasibility gate for generator-derived botanical surface charts.
 *
 * The deterministic Calamagrostis fixture is grouped into semantic,
 * edge-connected charts without using triangle IDs as the representation.
 * Actual periodic-mesh rays then measure angular winner-chart support and
 * same-line camera-inside chart succession. No runtime asset or shader is
 * emitted and no runtime traversal/candidate list/loop/march is proposed.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  ESTONIAN_GRAMINOID_PROFILE_IDS,
  makeEstonianGraminoidFixture,
} from '../../EstonianGraminoids';
import {
  deduplicateLineEvents,
  tracePeriodicLine,
} from '../../DeepLineEventCensus';
import {
  TriangleBvh,
  decodeOwnedProfileGeometry,
  type CensusVec3,
  type DecodedOwnedProfileGeometry,
} from '../../ExteriorClosureCensus';
import { periodicNearestSuccessor } from '../../OriginAwareRayTruth';

const EXPECTED_SOURCE_SHA256 =
  '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const EXPECTED_GENERATOR_MESH_SHA256 =
  '37b0cf1d33f632b5cde3dd60410ad987eec2e977661d0bc804dd6f2bee7054c0';
const MAXIMUM_RESIDENT_BYTES = 51_121_152;
const STORED_TILE = 258;
const PROPOSED_DIRECTION_COUNT = 81;
const ANGULAR_PHASE_GRID = 8;
const LINE_PHASE_GRID = 2;
const PHASE_OFFSET = 0.3819660112501051;
const AZIMUTH_COUNT = 16;
const LINE_AZIMUTH_COUNT = 8;
const CANONICAL_ELEVATIONS = [5, 15, 35, 55, 75] as const;
const LIVE_ELEVATION_BRACKETS = [
  { live: 10, low: 5, high: 15 },
  { live: 25, low: 15, high: 35 },
  { live: 45, low: 35, high: 55 },
  { live: 65, low: 55, high: 75 },
  { live: 82.5, low: 75, high: 90 },
] as const;
const LINE_ELEVATIONS = [0.1, 1, 5, 15] as const;
const HORIZONTAL_HEIGHTS = [0.25, 0.5, 0.75, 0.9] as const;
const HORIZONTAL_HORIZON_METRES = 155;
const ORIGIN_EPSILON_METRES = 1e-9;
const EVENT_MERGE_EPSILON_METRES = 1e-8;
const BOUNDARY_MARGIN_METRES = 1e-6;
const TRIANGLE_BITS = 22n;
const TRIANGLE_MASK = (1n << TRIANGLE_BITS) - 1n;

const FAMILY_NAMES = [
  'foliage-ribbon',
  'culm-tube',
  'rhizome-tube',
  'panicle-axis-tube',
  'spikelet-surface',
  'callus-hair-or-filament',
  'anther-surface',
] as const;
type FamilyName = typeof FAMILY_NAMES[number];

const PALETTE = [
  { rgb: [0.10, 0.27, 0.055], family: 0 },
  { rgb: [0.37, 0.51, 0.13], family: 0 },
  { rgb: [0.43, 0.53, 0.20], family: 1 },
  { rgb: [0.31, 0.24, 0.075], family: 2 },
  { rgb: [0.48, 0.31, 0.24], family: 3 },
  { rgb: [0.57, 0.35, 0.39], family: 4 },
  { rgb: [0.88, 0.80, 0.70], family: 5 },
  { rgb: [0.42, 0.10, 0.34], family: 6 },
] as const;

interface Args { [key: string]: string | boolean }

interface HitToken {
  hit: boolean;
  triangle: number;
  chart: number;
  family: number;
}

interface AngularAccumulator {
  samples: number;
  truthHits: number;
  cornerHitUnion: number;
  hitMissAgreementNearest: number;
  truthTriangleInCorners: number;
  truthChartInCorners: number;
  truthFamilyInCorners: number;
  nearestTriangleExact: number;
  nearestChartExact: number;
  nearestFamilyExact: number;
  chartPresentButTriangleAbsent: number;
  unanimousNonMissCornerChart: number;
  cornerChartCardinalityHistogram: Map<number, number>;
}

interface LineAccumulator {
  lines: number;
  emptyLines: number;
  events: number[];
  distinctCharts: number[];
  chartRuns: number[];
  distinctFamilies: number[];
  repeatedEventsWithinChart: number;
  totalEvents: number;
}

class UnionFind {
  readonly parent: Int32Array;
  readonly rank: Uint8Array;

  constructor(count: number) {
    this.parent = new Int32Array(count);
    this.rank = new Uint8Array(count);
    for (let index = 0; index < count; index++) this.parent[index] = index;
  }

  find(value: number): number {
    let root = value;
    while (this.parent[root] !== root) root = this.parent[root]!;
    while (this.parent[value] !== value) {
      const next = this.parent[value]!;
      this.parent[value] = root;
      value = next;
    }
    return root;
  }

  union(left: number, right: number): void {
    let a = this.find(left);
    let b = this.find(right);
    if (a === b) return;
    if (this.rank[a]! < this.rank[b]!) [a, b] = [b, a];
    this.parent[b] = a;
    if (this.rank[a] === this.rank[b]) this.rank[a]++;
  }
}

function parseArgs(argv: readonly string[]): Args {
  const result: Args = {};
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index]!;
    if (!argument.startsWith('--')) continue;
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

function quantiles(valuesInput: readonly number[]): Record<string, number | null> {
  if (valuesInput.length === 0) {
    return { p50: null, p90: null, p95: null, p99: null, maximum: null };
  }
  const values = [...valuesInput].sort((a, b) => a - b);
  const at = (fraction: number): number => values[Math.floor((values.length - 1) * fraction)]!;
  return { p50: at(0.5), p90: at(0.9), p95: at(0.95), p99: at(0.99), maximum: values.at(-1)! };
}

function emptyAngular(): AngularAccumulator {
  return {
    samples: 0,
    truthHits: 0,
    cornerHitUnion: 0,
    hitMissAgreementNearest: 0,
    truthTriangleInCorners: 0,
    truthChartInCorners: 0,
    truthFamilyInCorners: 0,
    nearestTriangleExact: 0,
    nearestChartExact: 0,
    nearestFamilyExact: 0,
    chartPresentButTriangleAbsent: 0,
    unanimousNonMissCornerChart: 0,
    cornerChartCardinalityHistogram: new Map(),
  };
}

function finishAngular(value: AngularAccumulator): Record<string, unknown> {
  const truth = Math.max(1, value.truthHits);
  const histogram = Object.fromEntries([...value.cornerChartCardinalityHistogram.entries()]
    .sort((left, right) => left[0] - right[0])
    .map(([key, count]) => [String(key), count]));
  return {
    samples: value.samples,
    truthHits: value.truthHits,
    nearestHitMissAgreement: value.samples > 0 ? value.hitMissAgreementNearest / value.samples : 1,
    truthHitCoveredByAnyCorner: value.truthHits > 0 ? value.cornerHitUnion / value.truthHits : 1,
    truthTrianglePresentAmongFourCorners: value.truthTriangleInCorners / truth,
    truthChartPresentAmongFourCorners: value.truthChartInCorners / truth,
    truthSemanticFamilyPresentAmongFourCorners: value.truthFamilyInCorners / truth,
    nearestTriangleExact: value.nearestTriangleExact / truth,
    nearestChartExact: value.nearestChartExact / truth,
    nearestSemanticFamilyExact: value.nearestFamilyExact / truth,
    chartPresentButLocalTriangleAbsent: value.chartPresentButTriangleAbsent / truth,
    unanimousNonMissCornerChart: value.samples > 0 ? value.unanimousNonMissCornerChart / value.samples : 1,
    distinctNonMissCornerChartsHistogram: histogram,
  };
}

function emptyLine(): LineAccumulator {
  return {
    lines: 0,
    emptyLines: 0,
    events: [],
    distinctCharts: [],
    chartRuns: [],
    distinctFamilies: [],
    repeatedEventsWithinChart: 0,
    totalEvents: 0,
  };
}

function finishLine(value: LineAccumulator): Record<string, unknown> {
  return {
    lines: value.lines,
    emptyLines: value.emptyLines,
    eventsPerLine: quantiles(value.events),
    distinctChartsPerLine: quantiles(value.distinctCharts),
    chartRunsPerLine: quantiles(value.chartRuns),
    distinctSemanticFamiliesPerLine: quantiles(value.distinctFamilies),
    eventToDistinctChartRatio: value.totalEvents > 0
      ? value.distinctCharts.reduce((sum, count) => sum + count, 0) / value.totalEvents : 1,
    repeatedEventFractionWithinSameChart: value.totalEvents > 0
      ? value.repeatedEventsWithinChart / value.totalEvents : 0,
  };
}

function nearestPaletteFamily(r: number, g: number, b: number): number {
  // appendBlade is the only primitive with a colour ramp. Recognise its exact
  // generator segment before nearest-palette classification, otherwise the
  // middle of the leaf-root -> leaf-tip ramp can be spuriously labelled culm
  // and split one analytic ribbon into multiple inferred charts.
  const leafRoot = PALETTE[0]!.rgb;
  const leafTip = PALETTE[1]!.rgb;
  const vx = leafTip[0] - leafRoot[0];
  const vy = leafTip[1] - leafRoot[1];
  const vz = leafTip[2] - leafRoot[2];
  const px = r - leafRoot[0];
  const py = g - leafRoot[1];
  const pz = b - leafRoot[2];
  const t = (px * vx + py * vy + pz * vz) / (vx * vx + vy * vy + vz * vz);
  const rx = px - t * vx;
  const ry = py - t * vy;
  const rz = pz - t * vz;
  if (t >= -1e-9 && t <= 1 + 1e-9 && rx * rx + ry * ry + rz * rz <= 1e-18) return 0;
  let best = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < PALETTE.length; index++) {
    const color = PALETTE[index]!.rgb;
    const distance = (r - color[0]) ** 2 + (g - color[1]) ** 2 + (b - color[2]) ** 2;
    if (distance < bestDistance) {
      bestDistance = distance;
      best = PALETTE[index]!.family;
    }
  }
  return best;
}

function triangleFamilies(
  positions: readonly number[],
  colors: readonly number[],
  indices: readonly number[],
): Uint8Array {
  const vertexCount = positions.length / 3;
  const vertexFamily = new Uint8Array(vertexCount);
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const offset = vertex * 3;
    vertexFamily[vertex] = nearestPaletteFamily(
      colors[offset]!, colors[offset + 1]!, colors[offset + 2]!,
    );
  }
  const triangleCount = indices.length / 3;
  const result = new Uint8Array(triangleCount);
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const offset = triangle * 3;
    const a = vertexFamily[indices[offset]!]!;
    const b = vertexFamily[indices[offset + 1]!]!;
    const c = vertexFamily[indices[offset + 2]!]!;
    if (a === b || a === c) result[triangle] = a;
    else if (b === c) result[triangle] = b;
    else {
      let r = 0;
      let g = 0;
      let blue = 0;
      for (let corner = 0; corner < 3; corner++) {
        const vertex = indices[offset + corner]! * 3;
        r += colors[vertex]! / 3;
        g += colors[vertex + 1]! / 3;
        blue += colors[vertex + 2]! / 3;
      }
      result[triangle] = nearestPaletteFamily(r, g, blue);
    }
  }
  return result;
}

function encodedEdge(aInput: number, bInput: number, triangle: number): bigint {
  const a = Math.min(aInput, bInput);
  const b = Math.max(aInput, bInput);
  return (BigInt(a) << 43n) | (BigInt(b) << TRIANGLE_BITS) | BigInt(triangle);
}

interface ChartPartition {
  triangleChart: Uint32Array;
  triangleFamily: Uint8Array;
  chartCount: number;
  chartTriangleCounts: number[];
  chartFamilies: number[];
  chartMinimumTriangle: number[];
  chartMaximumTriangle: number[];
}

function partitionCharts(
  positions: readonly number[],
  colors: readonly number[],
  indices: readonly number[],
): ChartPartition {
  const triangleCount = indices.length / 3;
  const families = triangleFamilies(positions, colors, indices);
  const edges = new BigUint64Array(triangleCount * 3);
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const offset = triangle * 3;
    const a = indices[offset]!;
    const b = indices[offset + 1]!;
    const c = indices[offset + 2]!;
    edges[offset] = encodedEdge(a, b, triangle);
    edges[offset + 1] = encodedEdge(b, c, triangle);
    edges[offset + 2] = encodedEdge(c, a, triangle);
  }
  console.error(`[chart-gate] sorting ${edges.length.toLocaleString()} indexed edges`);
  edges.sort();
  const union = new UnionFind(triangleCount);
  for (let first = 0; first < edges.length;) {
    const pair = edges[first]! >> TRIANGLE_BITS;
    let end = first + 1;
    while (end < edges.length && edges[end]! >> TRIANGLE_BITS === pair) end++;
    for (let index = first + 1; index < end; index++) {
      const triangle = Number(edges[index]! & TRIANGLE_MASK);
      for (let prior = first; prior < index; prior++) {
        const other = Number(edges[prior]! & TRIANGLE_MASK);
        if (families[triangle] === families[other]) union.union(triangle, other);
      }
    }
    first = end;
  }
  const rootChart = new Int32Array(triangleCount);
  rootChart.fill(-1);
  const triangleChart = new Uint32Array(triangleCount);
  const chartTriangleCounts: number[] = [];
  const chartFamilies: number[] = [];
  const chartMinimumTriangle: number[] = [];
  const chartMaximumTriangle: number[] = [];
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const root = union.find(triangle);
    let chart = rootChart[root]!;
    if (chart < 0) {
      chart = chartTriangleCounts.length;
      rootChart[root] = chart;
      chartTriangleCounts.push(0);
      chartFamilies.push(families[triangle]!);
      chartMinimumTriangle.push(triangle);
      chartMaximumTriangle.push(triangle);
    }
    triangleChart[triangle] = chart;
    chartTriangleCounts[chart]++;
    chartMinimumTriangle[chart] = Math.min(chartMinimumTriangle[chart]!, triangle);
    chartMaximumTriangle[chart] = Math.max(chartMaximumTriangle[chart]!, triangle);
  }
  return {
    triangleChart,
    triangleFamily: families,
    chartCount: chartTriangleCounts.length,
    chartTriangleCounts,
    chartFamilies,
    chartMinimumTriangle,
    chartMaximumTriangle,
  };
}

function direction(azimuthDegrees: number, elevationDegrees: number): CensusVec3 {
  const azimuth = azimuthDegrees * Math.PI / 180;
  const elevation = elevationDegrees * Math.PI / 180;
  return [
    Math.cos(elevation) * Math.cos(azimuth),
    -Math.sin(elevation),
    Math.cos(elevation) * Math.sin(azimuth),
  ];
}

function tokenForRay(
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  partition: ChartPartition,
  origin: CensusVec3,
  rayDirection: CensusVec3,
): HitToken {
  const vertical = Math.abs(rayDirection[1]);
  const horizon = vertical > 1e-12
    ? (origin[1] - geometry.bounds.min[1] + BOUNDARY_MARGIN_METRES) / vertical
    : HORIZONTAL_HORIZON_METRES;
  const hit = periodicNearestSuccessor(
    geometry,
    bvh,
    origin,
    rayDirection,
    horizon,
    ORIGIN_EPSILON_METRES,
  );
  if (!hit) return { hit: false, triangle: -1, chart: -1, family: -1 };
  return {
    hit: true,
    triangle: hit.triangleId,
    chart: partition.triangleChart[hit.triangleId]!,
    family: partition.triangleFamily[hit.triangleId]!,
  };
}

function addAngular(
  accumulator: AngularAccumulator,
  truth: HitToken,
  corners: readonly HitToken[],
): void {
  accumulator.samples++;
  const nearest = corners[0]!;
  if (nearest.hit === truth.hit) accumulator.hitMissAgreementNearest++;
  if (!truth.hit) return;
  accumulator.truthHits++;
  const hits = corners.filter((corner) => corner.hit);
  if (hits.length > 0) accumulator.cornerHitUnion++;
  if (hits.some((corner) => corner.triangle === truth.triangle)) accumulator.truthTriangleInCorners++;
  if (hits.some((corner) => corner.chart === truth.chart)) accumulator.truthChartInCorners++;
  if (hits.some((corner) => corner.family === truth.family)) accumulator.truthFamilyInCorners++;
  if (nearest.hit && nearest.triangle === truth.triangle) accumulator.nearestTriangleExact++;
  if (nearest.hit && nearest.chart === truth.chart) accumulator.nearestChartExact++;
  if (nearest.hit && nearest.family === truth.family) accumulator.nearestFamilyExact++;
  if (
    hits.some((corner) => corner.chart === truth.chart)
    && !hits.some((corner) => corner.triangle === truth.triangle)
  ) accumulator.chartPresentButTriangleAbsent++;
  const uniqueCharts = new Set(hits.map((corner) => corner.chart));
  accumulator.cornerChartCardinalityHistogram.set(
    uniqueCharts.size,
    (accumulator.cornerChartCardinalityHistogram.get(uniqueCharts.size) ?? 0) + 1,
  );
  if (hits.length > 0 && uniqueCharts.size === 1) accumulator.unanimousNonMissCornerChart++;
}

function canonicalIndex(azimuthIndex: number, elevation: number): number {
  if (elevation === 90) return AZIMUTH_COUNT * CANONICAL_ELEVATIONS.length;
  const elevationIndex = CANONICAL_ELEVATIONS.indexOf(elevation as typeof CANONICAL_ELEVATIONS[number]);
  if (elevationIndex < 0) throw new Error(`not a canonical elevation ${elevation}`);
  return azimuthIndex * CANONICAL_ELEVATIONS.length + elevationIndex;
}

function lineDirection(azimuthIndex: number, elevation: number, upward: boolean): CensusVec3 {
  const down = direction(azimuthIndex / LINE_AZIMUTH_COUNT * 360, elevation);
  return upward ? [down[0], -down[1], down[2]] : down;
}

function addLine(
  accumulator: LineAccumulator,
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  partition: ChartPartition,
  origin: CensusVec3,
  rayDirection: CensusVec3,
  maximumT: number,
): void {
  const trace = tracePeriodicLine(
    geometry,
    bvh,
    origin,
    rayDirection,
    ORIGIN_EPSILON_METRES,
    maximumT,
  );
  const events = deduplicateLineEvents(trace.rawHits, EVENT_MERGE_EPSILON_METRES);
  const charts = events.map((event) => partition.triangleChart[event.firstTriangleId]!);
  const families = events.map((event) => partition.triangleFamily[event.firstTriangleId]!);
  const distinctCharts = new Set(charts).size;
  let runs = 0;
  for (let index = 0; index < charts.length; index++) {
    if (index === 0 || charts[index] !== charts[index - 1]) runs++;
  }
  accumulator.lines++;
  if (events.length === 0) accumulator.emptyLines++;
  accumulator.events.push(events.length);
  accumulator.distinctCharts.push(distinctCharts);
  accumulator.chartRuns.push(runs);
  accumulator.distinctFamilies.push(new Set(families).size);
  accumulator.totalEvents += events.length;
  accumulator.repeatedEventsWithinChart += events.length - distinctCharts;
}

const args = parseArgs(process.argv.slice(2));
const source = resolve(typeof args.source === 'string'
  ? args.source : 'src/assets/groundcover/calamagrostis-canescens.gcrp');
const sourceBytes = readFileSync(source);
const sourceSha256 = sha256(sourceBytes);
if (sourceSha256 !== EXPECTED_SOURCE_SHA256) {
  throw new Error(`procedural chart gate expected ${EXPECTED_SOURCE_SHA256}, got ${sourceSha256}`);
}
const generatorPath = resolve('tools/groundcover-bake/EstonianGraminoids.ts');
const generatorSha256 = sha256(readFileSync(generatorPath));
const sourceLedgerPath = resolve('tools/groundcover-bake/ESTONIAN-GRAMINOIDS-README.md');
const sourceLedgerSha256 = sha256(readFileSync(sourceLedgerPath));
const qaIndexPath = resolve(
  'data/work/groundcover-gpu-bake-estonian-graminoids-v4/2-calamagrostis-canescens/',
  '2ed57f59d86e8376/qa/index.json',
);
const qaIndexSha256 = sha256(readFileSync(qaIndexPath));
const scriptPath = fileURLToPath(import.meta.url);
const scriptSha256 = sha256(readFileSync(scriptPath));
const recipe = {
  schema: 'laas-groundcover-procedural-chart-feasibility/v1',
  sourceSha256,
  generatorSha256,
  sourceLedgerSha256,
  qaIndexSha256,
  scriptSha256,
  chartPartition: 'semantic palette family plus shared-indexed-edge connected component',
  proposedCanonicalDirections: {
    azimuths: AZIMUTH_COUNT,
    elevations: CANONICAL_ELEVATIONS,
    vertical: 90,
    count: PROPOSED_DIRECTION_COUNT,
  },
  liveMidDirections: LIVE_ELEVATION_BRACKETS,
  angularPhaseGrid: ANGULAR_PHASE_GRID,
  linePhaseGrid: LINE_PHASE_GRID,
  sameLineElevations: LINE_ELEVATIONS,
  horizontalHeights: HORIZONTAL_HEIGHTS,
  horizontalHorizonMetres: HORIZONTAL_HORIZON_METRES,
  maximumResidentBytes: MAXIMUM_RESIDENT_BYTES,
};
const recipeSha256 = sha256(canonicalJson(recipe));
const root = resolve(typeof args.output === 'string' ? args.output : resolve(
  'data/work/groundcover-procedural-chart-feasibility',
  sourceSha256.slice(0, 16),
  recipeSha256.slice(0, 16),
));
mkdirSync(root, { recursive: true });

console.error('[chart-gate] regenerating deterministic Calamagrostis fixture');
const fixture = makeEstonianGraminoidFixture(ESTONIAN_GRAMINOID_PROFILE_IDS.CALAMAGROSTIS_CANESCENS);
if (!fixture.mesh.colors) throw new Error('Calamagrostis fixture has no authored colours');
const geometry = decodeOwnedProfileGeometry(sourceBytes);
if (
  fixture.mesh.positions.length / 3 !== geometry.vertexCount
  || fixture.mesh.indices.length / 3 !== geometry.triangleCount
) throw new Error('deterministic generator no longer matches the accepted GCRP counts');
let indexMismatches = 0;
for (let index = 0; index < fixture.mesh.indices.length; index++) {
  if (fixture.mesh.indices[index] !== geometry.triangles[index]) indexMismatches++;
}
if (indexMismatches !== 0) throw new Error(`generator/GCRP index mismatch count ${indexMismatches}`);

console.error('[chart-gate] partitioning semantic edge-connected charts');
const partition = partitionCharts(
  fixture.mesh.positions,
  fixture.mesh.colors,
  fixture.mesh.indices,
);
const familyReport: Record<FamilyName, { charts: number; triangles: number; chartTriangles: number[] }> =
  Object.fromEntries(FAMILY_NAMES.map((name) => [name, { charts: 0, triangles: 0, chartTriangles: [] }])) as never;
let contiguousCharts = 0;
for (let chart = 0; chart < partition.chartCount; chart++) {
  const family = FAMILY_NAMES[partition.chartFamilies[chart]!]!;
  const target = familyReport[family];
  target.charts++;
  target.triangles += partition.chartTriangleCounts[chart]!;
  target.chartTriangles.push(partition.chartTriangleCounts[chart]!);
  if (
    partition.chartMaximumTriangle[chart]! - partition.chartMinimumTriangle[chart]! + 1
    === partition.chartTriangleCounts[chart]
  ) contiguousCharts++;
}
const familySummary = Object.fromEntries(FAMILY_NAMES.map((name) => {
  const value = familyReport[name];
  return [name, {
    charts: value.charts,
    triangles: value.triangles,
    trianglesPerChart: quantiles(value.chartTriangles),
  }];
}));

const rawView = new DataView(sourceBytes.buffer, sourceBytes.byteOffset, sourceBytes.byteLength);
const triangleOffset = rawView.getUint32(88, true);
let nonzeroReservedTriangleWords = 0;
for (let triangle = 0; triangle < geometry.triangleCount; triangle++) {
  if (rawView.getUint32(triangleOffset + triangle * 16 + 12, true) !== 0) nonzeroReservedTriangleWords++;
}

console.error('[chart-gate] building actual-mesh BVH');
const bvh = TriangleBvh.build(geometry, 8);
const angularAll = emptyAngular();
const angularByElevation = new Map<number, AngularAccumulator>();
for (const bracket of LIVE_ELEVATION_BRACKETS) angularByElevation.set(bracket.live, emptyAngular());
let angularPhases = 0;
for (let phaseZIndex = 0; phaseZIndex < ANGULAR_PHASE_GRID; phaseZIndex++) {
  const phaseZ = geometry.tileOriginZ
    + (phaseZIndex + PHASE_OFFSET) / ANGULAR_PHASE_GRID * geometry.tileSizeZ;
  for (let phaseXIndex = 0; phaseXIndex < ANGULAR_PHASE_GRID; phaseXIndex++) {
    const phaseX = geometry.tileOriginX
      + (phaseXIndex + PHASE_OFFSET) / ANGULAR_PHASE_GRID * geometry.tileSizeX;
    const origin: CensusVec3 = [phaseX, geometry.topH, phaseZ];
    const canonical: HitToken[] = Array.from({ length: PROPOSED_DIRECTION_COUNT });
    for (let azimuth = 0; azimuth < AZIMUTH_COUNT; azimuth++) {
      for (const elevation of CANONICAL_ELEVATIONS) {
        canonical[canonicalIndex(azimuth, elevation)] = tokenForRay(
          geometry,
          bvh,
          partition,
          origin,
          direction(azimuth / AZIMUTH_COUNT * 360, elevation),
        );
      }
    }
    canonical[canonicalIndex(0, 90)] = tokenForRay(
      geometry, bvh, partition, origin, direction(0, 90),
    );
    for (let azimuth = 0; azimuth < AZIMUTH_COUNT; azimuth++) {
      const nextAzimuth = (azimuth + 1) % AZIMUTH_COUNT;
      const liveAzimuth = (azimuth + 0.5) / AZIMUTH_COUNT * 360;
      for (const bracket of LIVE_ELEVATION_BRACKETS) {
        const truth = tokenForRay(
          geometry, bvh, partition, origin, direction(liveAzimuth, bracket.live),
        );
        const corners = bracket.high === 90
          ? [
            canonical[canonicalIndex(azimuth, bracket.low)]!,
            canonical[canonicalIndex(nextAzimuth, bracket.low)]!,
            canonical[canonicalIndex(0, 90)]!,
            canonical[canonicalIndex(0, 90)]!,
          ]
          : [
            canonical[canonicalIndex(azimuth, bracket.low)]!,
            canonical[canonicalIndex(nextAzimuth, bracket.low)]!,
            canonical[canonicalIndex(azimuth, bracket.high)]!,
            canonical[canonicalIndex(nextAzimuth, bracket.high)]!,
          ];
        addAngular(angularAll, truth, corners);
        addAngular(angularByElevation.get(bracket.live)!, truth, corners);
      }
    }
    angularPhases++;
    if (angularPhases % 8 === 0) console.error(`[chart-gate] ${angularPhases}/${ANGULAR_PHASE_GRID ** 2} angular phases`);
  }
}

console.error('[chart-gate] tracing same-line chart succession');
const lineAll = emptyLine();
const lineByDirection = new Map<string, LineAccumulator>();
const lineAccumulator = (label: string): LineAccumulator => {
  let result = lineByDirection.get(label);
  if (!result) {
    result = emptyLine();
    lineByDirection.set(label, result);
  }
  return result;
};
for (let phaseZIndex = 0; phaseZIndex < LINE_PHASE_GRID; phaseZIndex++) {
  const phaseZ = geometry.tileOriginZ
    + (phaseZIndex + PHASE_OFFSET) / LINE_PHASE_GRID * geometry.tileSizeZ;
  for (let phaseXIndex = 0; phaseXIndex < LINE_PHASE_GRID; phaseXIndex++) {
    const phaseX = geometry.tileOriginX
      + (phaseXIndex + PHASE_OFFSET) / LINE_PHASE_GRID * geometry.tileSizeX;
    for (let azimuth = 0; azimuth < LINE_AZIMUTH_COUNT; azimuth++) {
      for (const heightFraction of HORIZONTAL_HEIGHTS) {
        const y = geometry.bounds.min[1]
          + heightFraction * (geometry.bounds.max[1] - geometry.bounds.min[1]);
        const rayDirection = lineDirection(azimuth, 0, false);
        addLine(
          lineAll, geometry, bvh, partition,
          [phaseX, y, phaseZ], rayDirection, HORIZONTAL_HORIZON_METRES,
        );
        addLine(
          lineAccumulator('0:horizontal'), geometry, bvh, partition,
          [phaseX, y, phaseZ], rayDirection, HORIZONTAL_HORIZON_METRES,
        );
      }
      for (const elevation of LINE_ELEVATIONS) {
        for (const upward of [false, true]) {
          const rayDirection = lineDirection(azimuth, elevation, upward);
          const vertical = Math.abs(rayDirection[1]);
          const originY = upward
            ? geometry.bounds.min[1] - BOUNDARY_MARGIN_METRES
            : geometry.bounds.max[1] + BOUNDARY_MARGIN_METRES;
          const maximumT = (
            geometry.bounds.max[1] - geometry.bounds.min[1] + BOUNDARY_MARGIN_METRES * 2
          ) / vertical;
          const label = `${elevation}:${upward ? 'upward' : 'downward'}`;
          addLine(
            lineAll, geometry, bvh, partition,
            [phaseX, originY, phaseZ], rayDirection, maximumT,
          );
          addLine(
            lineAccumulator(label), geometry, bvh, partition,
            [phaseX, originY, phaseZ], rayDirection, maximumT,
          );
        }
      }
    }
  }
}

const tokenBytes = 8;
const tokenAtlasBytes = STORED_TILE * STORED_TILE * PROPOSED_DIRECTION_COUNT * tokenBytes;
const chartTableBudget = MAXIMUM_RESIDENT_BYTES - tokenAtlasBytes;
const chartBits = Math.ceil(Math.log2(partition.chartCount + 1));
const report = {
  schema: 'laas-groundcover-procedural-chart-feasibility/v1',
  source: {
    asset: relative(process.cwd(), source),
    assetSha256: sourceSha256,
    generator: relative(process.cwd(), generatorPath),
    generatorSha256,
    expectedGeneratorMeshSha256: EXPECTED_GENERATOR_MESH_SHA256,
    sourceLedger: relative(process.cwd(), sourceLedgerPath),
    sourceLedgerSha256,
    acceptedQaIndex: relative(process.cwd(), qaIndexPath),
    acceptedQaIndexSha256: qaIndexSha256,
    generatorIdentity: fixture.generator,
    structure: fixture.structure,
    generatorToPackedIndexMismatches: indexMismatches,
  },
  reproducibility: { recipeSha256, recipe },
  chartPartition: {
    definition: recipe.chartPartition,
    charts: partition.chartCount,
    chartIdBits: chartBits,
    triangles: geometry.triangleCount,
    trianglesPerChart: quantiles(partition.chartTriangleCounts),
    contiguousTriangleRangeCharts: contiguousCharts,
    contiguousTriangleRangeFraction: contiguousCharts / partition.chartCount,
    bySemanticFamily: familySummary,
    interpretation: [
      'shared-edge connectivity separates primitive calls which meet only at an anchor vertex',
      'leaf-root and leaf-tip colours are one foliage family so a tapered blade remains one chart',
      'the partition uses botanical generator provenance and topology; chart IDs are not source triangle IDs',
    ],
  },
  currentFormatMetadataAudit: {
    nonzeroReservedFourthTriangleWords: nonzeroReservedTriangleWords,
    triangleChartIdPresent: false,
    vertexLocalUvPresent: false,
    analyticRecipeTablePresent: false,
    populationOrSpeciesMarkPerChartPresent: false,
    exactMissingRecookMetadata: [
      'stable chart instance ID and semantic primitive kind per emitted triangle',
      'local chart coordinates or local panel coordinates per vertex',
      'blade root/angle/height/width/lean/curl/phase and segment convention',
      'tube control points/radii/side count and endpoint weld semantics',
      'lanceolate and hair base/frame/length/width/roll/keel parameters',
      'parent chart, species/community/population mark, and periodic-copy semantics',
    ],
    boundedRecookPath: [
      'instrument appendBlade/appendTube/appendLanceolateSurface/appendHairFilament with an offline ChartRecipe recorder',
      'write chart ID into the currently zero fourth u32 of each GCRP triangle record',
      'append a versioned chart-recipe table and optional quantized local UV/panel data; preserve the original triangles for bake truth only',
      'offline-union all overlapping species/moss charts into one marked community field before runtime; never add one live field per species',
    ],
  },
  angularDisocclusion: {
    domain: {
      phaseGrid: ANGULAR_PHASE_GRID,
      canonicalDirections: PROPOSED_DIRECTION_COUNT,
      liveMidDirectionsPerPhase: AZIMUTH_COUNT * LIVE_ELEVATION_BRACKETS.length,
      canonicalElevations: [...CANONICAL_ELEVATIONS, 90],
      liveElevations: LIVE_ELEVATION_BRACKETS.map((entry) => entry.live),
    },
    overall: finishAngular(angularAll),
    byLiveElevation: Object.fromEntries([...angularByElevation.entries()]
      .map(([elevation, value]) => [String(elevation), finishAngular(value)])),
    meaning: [
      'truthChartPresentAmongFourCorners is an optimistic upper bound for a four-corner chart selector',
      'truthTrianglePresentAmongFourCorners is the local-panel stability available to a one-patch exact test',
      'a truth chart absent from all corners cannot be recovered by chart-local coordinates or an intra-cell separator over those tokens',
    ],
  },
  sameLineCameraPath: {
    overall: finishLine(lineAll),
    byDirection: Object.fromEntries([...lineByDirection.entries()]
      .sort((left, right) => left[0].localeCompare(right[0]))
      .map(([label, value]) => [label, finishLine(value)])),
    meaning: [
      'moving the camera origin on one oriented line changes the required successor chart',
      'distinct chart count is the categorical origin dependence which a single exterior chart token cannot encode',
      'offline chart grouping does not remove the fifth origin coordinate or the exact-horizontal finite-horizon complexity',
    ],
  },
  cost: {
    residentCapBytes: MAXIMUM_RESIDENT_BYTES,
    proposedDirectionCount: PROPOSED_DIRECTION_COUNT,
    token: {
      bytes: tokenBytes,
      fields: `${chartBits}-bit chart ID plus two quantized local coordinates/valid state`,
      atlasBytes: tokenAtlasBytes,
      fixedReads: 1,
    },
    bytesRemainingForEveryChartRecipe: chartTableBudget,
    bytesRemainingPerChart: chartTableBudget / partition.chartCount,
    optimistic16ByteDescriptorTableBytes: partition.chartCount * 16,
    optimistic16ByteDescriptorTableFits: tokenAtlasBytes + partition.chartCount * 16 <= MAXIMUM_RESIDENT_BYTES,
    minimum32ByteDescriptorTableBytes: partition.chartCount * 32,
    minimum32ByteDescriptorTableFits: tokenAtlasBytes + partition.chartCount * 32 <= MAXIMUM_RESIDENT_BYTES,
    fixedReadLowerBoundIfDescriptorFits: 2,
    exactEvaluationObstacle: [
      'a chart is generally a piecewise ribbon/tube with multiple local panels, not one plane or closed-form primitive',
      'blade centre/width use sin and noninteger powers; exact ray intersection has no fixed affine/projective inverse',
      'tube charts have variable polyline sections and side counts; selecting the intersected section is itself a candidate problem unless the baked local panel remains valid',
    ],
  },
  offlineMultiSpeciesUnion: {
    supportedInPrinciple: true,
    mark: '(community, species, population/root, chart, local coordinate)',
    contract: [
      'compose overlapping grass, forb, moss, and lichen charts offline into one periodic marked winner/event field',
      'retain categorical mark with depth/normal/colour; never numerically blend unrelated winners',
      'periodic copy is implicit in line phase where possible; finite population/root eligibility must be baked into the mark',
      'union increases angular disocclusions and same-line successors, so it cannot repair a single-species chart gate failure',
    ],
  },
  decisionRule: {
    acceptOnlyIf: [
      'exact live winner chart is stably recoverable without a runtime candidate set',
      'camera-inside successor chart is recoverable from the same fixed record',
      'the selected chart admits one fixed closed-form ray evaluation from stored local coordinates',
      'complete token atlas plus exact chart table stays within the resident cap and one/few fixed reads',
    ],
    noRuntimeEscape: 'failure does not authorize mesh traversal, a candidate list, a march, a loop, another pass, or distance-dependent work',
  },
};

const metricsPath = resolve(root, 'metrics.json');
const serialized = `${JSON.stringify(report, null, 2)}\n`;
writeFileSync(metricsPath, serialized);
const metricsSha256 = sha256(serialized);
const index = {
  schema: 'laas-groundcover-procedural-chart-feasibility-index/v1',
  sourceSha256,
  generatorSha256,
  recipeSha256,
  script: { file: relative(process.cwd(), scriptPath), sha256: scriptSha256 },
  files: [{ file: 'metrics.json', bytes: Buffer.byteLength(serialized), sha256: metricsSha256 }],
};
writeFileSync(resolve(root, 'index.json'), `${JSON.stringify(index, null, 2)}\n`);
console.error(`[chart-gate] wrote ${metricsPath}`);
console.error(`[chart-gate] metrics sha256 ${metricsSha256}`);
process.stdout.write(`${root}\n`);
