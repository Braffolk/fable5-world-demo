/**
 * Offline-only Class-E structural feasibility lower bound.
 *
 * This analyzer deliberately stops before compiling a Class-E community. It
 * extracts only those crisp, edge-connected components whose axes can be
 * inferred from accepted source geometry, then brackets the axis catalogue
 * size with:
 *
 *   - a valid pairwise-incompatible packing lower bound; and
 *   - a constructive surface-area-weighted greedy catalogue upper bound.
 *
 * The complete tolerance is optimistically assigned to axis quantization.
 * Curvature, cross-section, caps, endpoint anchoring, ownership, support,
 * interval, attribute-layout, plume, and minification costs remain explicit
 * unknowns. Consequently this can prove an axis/read budget impossible, but
 * it cannot prove the full compiler feasible.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  ESTONIAN_GRAMINOID_PROFILE_IDS,
  makeCalamagrostisCanescensBandlimitedSource,
  makeEstonianGraminoidFixture,
  type GraminoidPrimitiveRecipe,
} from './EstonianGraminoids';
import {
  makeSphagnumCapillifoliumFixture,
  type SphagnumPrimitiveRecipe,
} from './SphagnumCapillifolium';

type Vec3 = readonly [number, number, number];

interface MeshSelection {
  positions: ArrayLike<number>;
  indices: ArrayLike<number>;
  label: string;
  semantic: string;
}

interface ComponentMeasurement {
  id: number;
  triangles: number;
  vertices: number;
  surfaceAreaMetres2: number;
  centroid: Vec3;
  pcaAxis: Vec3;
  chordAxis: Vec3;
  chordLengthMetres: number;
  pcaEigenvalues: Vec3;
  axialElongation: number;
  maximumVertexDistanceToChordMetres: number;
  axisEligible: boolean;
}

interface AxisItem {
  componentId: number;
  axis: Vec3;
  lengthMetres: number;
  weightMetres2: number;
  start?: Vec3;
  end?: Vec3;
  compatibilityKey?: string;
}

interface SplitAxisItem extends AxisItem {
  start: Vec3;
  end: Vec3;
  compatibilityKey: string;
  sourceComponentId: number;
  splitPart: number;
  splitParts: number;
}

interface RecipeSegmentSummary {
  items: AxisItem[];
  primitiveCount: number;
  segmentCount: number;
  zeroLengthSegments: number;
  sourceTriangles: number;
  sourceSurfaceAreaMetres2: number;
  bySemantic: Record<string, {
    primitives: number;
    segments: number;
    sourceTriangles: number;
    sourceSurfaceAreaMetres2: number;
    segmentLengthsMetres: number[];
  }>;
}

interface CoverResult {
  frozenAtomization: string;
  toleranceMetres: number;
  componentCount: number;
  directionConstrainingComponentCount: number;
  surfaceAreaMetres2: number;
  conditionalFrozenAtomPackingLowerBound: number;
  packingComponentIds: number[];
  conditionalFrozenAtomGreedyUpperBound: number;
  greedyCatalogueComponentIds: number[];
  hardFieldLowerBoundWithSegmentationStillFree: number;
  hardTwoLayerReadLowerBoundWithSegmentationStillFree: number;
  conditionalFrozenAtomTwoLayerReadLowerBound: number;
  conditionalFrozenAtomTwoLayerReadUpperBound: number;
  conditionalNineReadVerdict: 'RED_IF_ATOMIZATION_FROZEN' | 'NOT_REFUTED_IF_ATOMIZATION_FROZEN';
}

const REPO_ROOT = resolve(import.meta.dirname, '../..');
const BANDLIMITED_ROOT = resolve(
  REPO_ROOT,
  'data/work/groundcover-bandlimited-botanical-source',
  '968e74302585c74af46f59fadb6c9ff541e4afb1bd3adc42602bf29c5e2ed87f',
);
const SPHAGNUM_PACKED = resolve(REPO_ROOT, 'src/assets/groundcover/sphagnum-capillifolium.gcrp');
const PRODUCTION_CALAMAGROSTIS_PACKED = resolve(REPO_ROOT, 'src/assets/groundcover/calamagrostis-canescens.gcrp');
const EXPECTED_BANDLIMITED_INDEX_SHA256 =
  '68f8b5af43567e4bbd5f9b135b172864f5b224f023ef488b749be1985266353e';
const EXPECTED_SPHAGNUM_PACKED_SHA256 =
  'cb61dd42c6265067f0be9a320d763da928b1e65a60ae3ebb359f49b3137a9959';
const EXPECTED_PRODUCTION_CALAMAGROSTIS_PACKED_SHA256 =
  '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const EXPECTED_PRODUCTION_CALAMAGROSTIS_MESH_SHA256 =
  '37b0cf1d33f632b5cde3dd60410ad987eec2e977661d0bc804dd6f2bee7054c0';
const TOLERANCES_METRES = [0.0005, 0.001, 0.0025, 0.005, 0.01] as const;
const STRUCTURAL_SENSITIVITY_TOLERANCES_METRES = [
  ...TOLERANCES_METRES,
  0.02,
  0.05,
  0.1,
  0.2,
] as const;
const A_MIN_GRID = [0.01, 0.025, 0.05, 0.1, 0.2] as const;
const AXIAL_ELONGATION_MINIMUM = 2;
const EXACT_GREEDY_LIMIT = 5_000;
const LARGE_COVER_PROBE_COUNT = 8;
const TWO_GLOBAL_LAYERS = 2;
const OPTIMISTIC_NON_FIELD_READS = 1;
const PROVISIONAL_READ_CEILING = 9;
const EPSILON = 1e-12;

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

function float32File(path: string): Float32Array {
  const bytes = readFileSync(path);
  if (bytes.byteLength % 4 !== 0) throw new Error(`${path} is not a Float32 array`);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

function uint32File(path: string): Uint32Array {
  const bytes = readFileSync(path);
  if (bytes.byteLength % 4 !== 0) throw new Error(`${path} is not a Uint32 array`);
  return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(a: Vec3, factor: number): Vec3 {
  return [a[0] * factor, a[1] * factor, a[2] * factor];
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function length(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}

function normalize(a: Vec3): Vec3 {
  const magnitude = length(a);
  return magnitude > EPSILON ? scale(a, 1 / magnitude) : [0, 1, 0];
}

function canonicalAxis(axisInput: Vec3): Vec3 {
  let axis = normalize(axisInput);
  if (
    axis[1] < -EPSILON
    || (Math.abs(axis[1]) <= EPSILON && axis[0] < -EPSILON)
    || (Math.abs(axis[1]) <= EPSILON && Math.abs(axis[0]) <= EPSILON && axis[2] < 0)
  ) axis = scale(axis, -1);
  return axis;
}

function unorientedAngle(a: Vec3, b: Vec3): number {
  return Math.acos(Math.min(1, Math.max(0, Math.abs(dot(a, b)))));
}

function vertex(positions: ArrayLike<number>, index: number): Vec3 {
  const offset = index * 3;
  return [positions[offset]!, positions[offset + 1]!, positions[offset + 2]!];
}

function triangleArea(positions: ArrayLike<number>, a: number, b: number, c: number): number {
  return length(cross(sub(vertex(positions, b), vertex(positions, a)), sub(vertex(positions, c), vertex(positions, a)))) * 0.5;
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

function edgeKey(aInput: number, bInput: number): bigint {
  const a = Math.min(aInput, bInput);
  const b = Math.max(aInput, bInput);
  return (BigInt(a) << 32n) | BigInt(b);
}

function edgeConnectedTriangleComponents(indices: ArrayLike<number>): number[][] {
  if (indices.length % 3 !== 0) throw new Error('selected indices are not triangles');
  const triangleCount = indices.length / 3;
  const union = new UnionFind(triangleCount);
  const firstTriangleByEdge = new Map<bigint, number>();
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const offset = triangle * 3;
    const a = indices[offset]!;
    const b = indices[offset + 1]!;
    const c = indices[offset + 2]!;
    for (const key of [edgeKey(a, b), edgeKey(b, c), edgeKey(c, a)]) {
      const first = firstTriangleByEdge.get(key);
      if (first === undefined) firstTriangleByEdge.set(key, triangle);
      else union.union(first, triangle);
    }
  }
  const byRoot = new Map<number, number[]>();
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const root = union.find(triangle);
    let target = byRoot.get(root);
    if (!target) {
      target = [];
      byRoot.set(root, target);
    }
    target.push(triangle);
  }
  return [...byRoot.values()].sort((left, right) => left[0]! - right[0]!);
}

function multiplySymmetric(matrix: readonly number[], vector: Vec3): Vec3 {
  return [
    matrix[0]! * vector[0] + matrix[1]! * vector[1] + matrix[2]! * vector[2],
    matrix[1]! * vector[0] + matrix[3]! * vector[1] + matrix[4]! * vector[2],
    matrix[2]! * vector[0] + matrix[4]! * vector[1] + matrix[5]! * vector[2],
  ];
}

function principalEigenvector(matrix: readonly number[]): Vec3 {
  const diagonal = [matrix[0]!, matrix[3]!, matrix[5]!];
  const largest = diagonal.indexOf(Math.max(...diagonal));
  let vector: Vec3 = largest === 0 ? [1, 0, 0] : largest === 1 ? [0, 1, 0] : [0, 0, 1];
  for (let iteration = 0; iteration < 48; iteration++) {
    const next = multiplySymmetric(matrix, vector);
    if (length(next) <= EPSILON) break;
    vector = normalize(next);
  }
  return canonicalAxis(vector);
}

function symmetricEigenvalues(matrix: readonly number[]): Vec3 {
  // Stable closed-form eigenvalues for a real symmetric 3x3 matrix.
  const a00 = matrix[0]!;
  const a01 = matrix[1]!;
  const a02 = matrix[2]!;
  const a11 = matrix[3]!;
  const a12 = matrix[4]!;
  const a22 = matrix[5]!;
  const p1 = a01 * a01 + a02 * a02 + a12 * a12;
  if (p1 <= EPSILON * EPSILON) {
    return [...[a00, a11, a22].sort((a, b) => b - a)] as [number, number, number];
  }
  const q = (a00 + a11 + a22) / 3;
  const p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2 * p1;
  const p = Math.sqrt(p2 / 6);
  const b00 = (a00 - q) / p;
  const b01 = a01 / p;
  const b02 = a02 / p;
  const b11 = (a11 - q) / p;
  const b12 = a12 / p;
  const b22 = (a22 - q) / p;
  const determinant =
    b00 * (b11 * b22 - b12 * b12)
    - b01 * (b01 * b22 - b12 * b02)
    + b02 * (b01 * b12 - b11 * b02);
  const phi = Math.acos(Math.min(1, Math.max(-1, determinant / 2))) / 3;
  const eigen0 = q + 2 * p * Math.cos(phi);
  const eigen2 = q + 2 * p * Math.cos(phi + 2 * Math.PI / 3);
  const eigen1 = 3 * q - eigen0 - eigen2;
  return [...[eigen0, eigen1, eigen2].sort((a, b) => b - a)] as [number, number, number];
}

function measureComponent(
  id: number,
  triangleIds: readonly number[],
  selection: MeshSelection,
): ComponentMeasurement {
  const unique = new Set<number>();
  let surfaceAreaMetres2 = 0;
  for (const triangle of triangleIds) {
    const offset = triangle * 3;
    const a = selection.indices[offset]!;
    const b = selection.indices[offset + 1]!;
    const c = selection.indices[offset + 2]!;
    unique.add(a);
    unique.add(b);
    unique.add(c);
    surfaceAreaMetres2 += triangleArea(selection.positions, a, b, c);
  }
  const vertices = [...unique].sort((a, b) => a - b);
  let centroid: Vec3 = [0, 0, 0];
  for (const index of vertices) centroid = add(centroid, vertex(selection.positions, index));
  centroid = scale(centroid, 1 / Math.max(1, vertices.length));
  const covariance = [0, 0, 0, 0, 0, 0];
  for (const index of vertices) {
    const delta = sub(vertex(selection.positions, index), centroid);
    covariance[0]! += delta[0] * delta[0];
    covariance[1]! += delta[0] * delta[1];
    covariance[2]! += delta[0] * delta[2];
    covariance[3]! += delta[1] * delta[1];
    covariance[4]! += delta[1] * delta[2];
    covariance[5]! += delta[2] * delta[2];
  }
  for (let index = 0; index < covariance.length; index++) covariance[index]! /= Math.max(1, vertices.length);
  const pcaAxis = principalEigenvector(covariance);
  const pcaEigenvalues = symmetricEigenvalues(covariance);
  let minimumProjection = Number.POSITIVE_INFINITY;
  let maximumProjection = Number.NEGATIVE_INFINITY;
  for (const index of vertices) {
    const projection = dot(sub(vertex(selection.positions, index), centroid), pcaAxis);
    minimumProjection = Math.min(minimumProjection, projection);
    maximumProjection = Math.max(maximumProjection, projection);
  }
  const pcaSpan = Math.max(0, maximumProjection - minimumProjection);
  const endpointBand = Math.max(pcaSpan * 0.1, 1e-8);
  let lowMean: Vec3 = [0, 0, 0];
  let highMean: Vec3 = [0, 0, 0];
  let lowCount = 0;
  let highCount = 0;
  for (const index of vertices) {
    const point = vertex(selection.positions, index);
    const projection = dot(sub(point, centroid), pcaAxis);
    if (projection <= minimumProjection + endpointBand) {
      lowMean = add(lowMean, point);
      lowCount++;
    }
    if (projection >= maximumProjection - endpointBand) {
      highMean = add(highMean, point);
      highCount++;
    }
  }
  lowMean = scale(lowMean, 1 / Math.max(1, lowCount));
  highMean = scale(highMean, 1 / Math.max(1, highCount));
  let chordAxis = canonicalAxis(sub(highMean, lowMean));
  if (length(sub(highMean, lowMean)) <= EPSILON) chordAxis = pcaAxis;
  let chordMinimum = Number.POSITIVE_INFINITY;
  let chordMaximum = Number.NEGATIVE_INFINITY;
  let maximumVertexDistanceToChordMetres = 0;
  for (const index of vertices) {
    const delta = sub(vertex(selection.positions, index), centroid);
    const projection = dot(delta, chordAxis);
    chordMinimum = Math.min(chordMinimum, projection);
    chordMaximum = Math.max(chordMaximum, projection);
    maximumVertexDistanceToChordMetres = Math.max(
      maximumVertexDistanceToChordMetres,
      length(sub(delta, scale(chordAxis, projection))),
    );
  }
  const chordLengthMetres = Math.max(0, chordMaximum - chordMinimum);
  const axialElongation = pcaEigenvalues[0]!
    / Math.max(EPSILON * EPSILON, pcaEigenvalues[1]!);
  return {
    id,
    triangles: triangleIds.length,
    vertices: vertices.length,
    surfaceAreaMetres2,
    centroid,
    pcaAxis,
    chordAxis,
    chordLengthMetres,
    pcaEigenvalues,
    axialElongation,
    maximumVertexDistanceToChordMetres,
    axisEligible: chordLengthMetres > 1e-7 && axialElongation >= AXIAL_ELONGATION_MINIMUM,
  };
}

function measureSelection(selection: MeshSelection): ComponentMeasurement[] {
  return edgeConnectedTriangleComponents(selection.indices)
    .map((triangles, index) => measureComponent(index, triangles, selection));
}

function quantiles(valuesInput: readonly number[]): Record<string, number | null> {
  if (valuesInput.length === 0) {
    return { p50: null, p90: null, p95: null, p99: null, maximum: null };
  }
  const values = [...valuesInput].sort((a, b) => a - b);
  const at = (fraction: number): number => values[Math.floor((values.length - 1) * fraction)]!;
  return { p50: at(0.5), p90: at(0.9), p95: at(0.95), p99: at(0.99), maximum: values.at(-1)! };
}

function angularRadius(item: AxisItem, toleranceMetres: number): number {
  return Math.asin(Math.min(1, 2 * toleranceMetres / Math.max(EPSILON, item.lengthMetres)));
}

function covers(candidate: Vec3, item: AxisItem, toleranceMetres: number): boolean {
  return item.lengthMetres * 0.5 * Math.sin(unorientedAngle(candidate, item.axis))
    <= toleranceMetres + 1e-12;
}

function greedyPacking(items: readonly AxisItem[], toleranceMetres: number, order: readonly number[]): number[] {
  const radii = items.map((item) => angularRadius(item, toleranceMetres));
  const selected: number[] = [];
  for (const index of order) {
    if (selected.every((other) =>
      unorientedAngle(items[index]!.axis, items[other]!.axis) > radii[index]! + radii[other]! + 1e-12
    )) selected.push(index);
  }
  return selected;
}

function packingLowerBound(items: readonly AxisItem[], toleranceMetres: number): number[] {
  const indices = items.map((_, index) => index);
  const radii = items.map((item) => angularRadius(item, toleranceMetres));
  const orders = [
    [...indices].sort((a, b) => radii[a]! - radii[b]! || items[b]!.weightMetres2 - items[a]!.weightMetres2 || a - b),
    [...indices].sort((a, b) => items[b]!.lengthMetres - items[a]!.lengthMetres || a - b),
    [...indices].sort((a, b) => items[b]!.weightMetres2 - items[a]!.weightMetres2 || a - b),
    [...indices],
  ];
  let best: number[] = [];
  for (const order of orders) {
    const candidate = greedyPacking(items, toleranceMetres, order);
    if (candidate.length > best.length) best = candidate;
  }
  return best;
}

function weightedGreedyCover(items: readonly AxisItem[], toleranceMetres: number): number[] {
  if (items.length > EXACT_GREEDY_LIMIT) {
    return weightedRestrictiveFrontierCover(items, toleranceMetres);
  }
  const count = items.length;
  const coveredByCandidate: number[][] = Array.from({ length: count }, () => []);
  const candidatesByItem: number[][] = Array.from({ length: count }, () => []);
  const scores = new Float64Array(count);
  for (let candidate = 0; candidate < count; candidate++) {
    for (let item = 0; item < count; item++) {
      if (!covers(items[candidate]!.axis, items[item]!, toleranceMetres)) continue;
      coveredByCandidate[candidate]!.push(item);
      candidatesByItem[item]!.push(candidate);
      scores[candidate]! += items[item]!.weightMetres2;
    }
  }
  const covered = new Uint8Array(count);
  const selected: number[] = [];
  let remaining = count;
  while (remaining > 0) {
    let best = -1;
    let bestScore = -1;
    for (let candidate = 0; candidate < count; candidate++) {
      const score = scores[candidate]!;
      if (score > bestScore + EPSILON || (Math.abs(score - bestScore) <= EPSILON && candidate < best)) {
        best = candidate;
        bestScore = score;
      }
    }
    if (best < 0 || bestScore <= 0) throw new Error('axis cover has uncovered components but no covering candidate');
    selected.push(best);
    scores[best] = -1;
    for (const item of coveredByCandidate[best]!) {
      if (covered[item]) continue;
      covered[item] = 1;
      remaining--;
      const weight = items[item]!.weightMetres2;
      for (const candidate of candidatesByItem[item]!) {
        if (scores[candidate]! >= 0) scores[candidate]! -= weight;
      }
    }
  }
  return selected;
}

/**
 * Scalable constructive cover for a large authored segment set. The first
 * few uncovered items with the smallest angular radii form a deterministic
 * candidate frontier; the candidate covering the greatest remaining source
 * area wins. It is a valid cover and an upper bound, not exact set cover.
 */
function weightedRestrictiveFrontierCover(items: readonly AxisItem[], toleranceMetres: number): number[] {
  const order = items.map((_, index) => index).sort((a, b) =>
    angularRadius(items[a]!, toleranceMetres) - angularRadius(items[b]!, toleranceMetres)
    || items[b]!.weightMetres2 - items[a]!.weightMetres2
    || a - b
  );
  const uncovered = new Uint8Array(items.length);
  uncovered.fill(1);
  let remaining = items.length;
  const selected: number[] = [];
  while (remaining > 0) {
    const frontier: number[] = [];
    for (const index of order) {
      if (!uncovered[index]) continue;
      frontier.push(index);
      if (frontier.length >= LARGE_COVER_PROBE_COUNT) break;
    }
    let best = frontier[0]!;
    let bestScore = -1;
    for (const candidate of frontier) {
      let score = 0;
      for (let item = 0; item < items.length; item++) {
        if (uncovered[item] && covers(items[candidate]!.axis, items[item]!, toleranceMetres)) {
          score += items[item]!.weightMetres2;
        }
      }
      if (score > bestScore + EPSILON || (Math.abs(score - bestScore) <= EPSILON && candidate < best)) {
        best = candidate;
        bestScore = score;
      }
    }
    selected.push(best);
    for (let item = 0; item < items.length; item++) {
      if (uncovered[item] && covers(items[best]!.axis, items[item]!, toleranceMetres)) {
        uncovered[item] = 0;
        remaining--;
      }
    }
  }
  return selected;
}

function coverCurve(items: readonly AxisItem[], frozenAtomization: string): CoverResult[] {
  const area = items.reduce((sum, item) => sum + item.weightMetres2, 0);
  return TOLERANCES_METRES.map((toleranceMetres) => {
    // ell/2 <= epsilon makes the displayed axis residual <= epsilon for every
    // catalogue axis. Such atoms impose no directional constraint and can be
    // omitted from the covering solve without changing K.
    const constrained = items.filter((item) => item.lengthMetres > 2 * toleranceMetres + 1e-12);
    const packing = packingLowerBound(constrained, toleranceMetres);
    const catalogue = weightedGreedyCover(constrained, toleranceMetres);
    const lowerFields = Math.max(items.length > 0 ? 1 : 0, packing.length);
    const upperFields = Math.max(items.length > 0 ? 1 : 0, catalogue.length);
    const conditionalFrozenAtomTwoLayerReadLowerBound =
      TWO_GLOBAL_LAYERS * lowerFields + OPTIMISTIC_NON_FIELD_READS;
    const conditionalFrozenAtomTwoLayerReadUpperBound =
      TWO_GLOBAL_LAYERS * upperFields + OPTIMISTIC_NON_FIELD_READS;
    return {
      frozenAtomization,
      toleranceMetres,
      componentCount: items.length,
      directionConstrainingComponentCount: constrained.length,
      surfaceAreaMetres2: area,
      conditionalFrozenAtomPackingLowerBound: packing.length,
      packingComponentIds: packing.map((index) => constrained[index]!.componentId),
      conditionalFrozenAtomGreedyUpperBound: catalogue.length,
      greedyCatalogueComponentIds: catalogue.map((index) => constrained[index]!.componentId),
      hardFieldLowerBoundWithSegmentationStillFree: items.length > 0 ? 1 : 0,
      hardTwoLayerReadLowerBoundWithSegmentationStillFree:
        TWO_GLOBAL_LAYERS * (items.length > 0 ? 1 : 0) + OPTIMISTIC_NON_FIELD_READS,
      conditionalFrozenAtomTwoLayerReadLowerBound,
      conditionalFrozenAtomTwoLayerReadUpperBound,
      conditionalNineReadVerdict: conditionalFrozenAtomTwoLayerReadLowerBound > PROVISIONAL_READ_CEILING
        ? 'RED_IF_ATOMIZATION_FROZEN'
        : 'NOT_REFUTED_IF_ATOMIZATION_FROZEN',
    };
  });
}

function nearestAllowedAxis(axisInput: Vec3, aMin: number): Vec3 {
  const axis = canonicalAxis(axisInput);
  if (axis[1] >= aMin) return axis;
  const horizontal = Math.hypot(axis[0], axis[2]);
  if (horizontal <= EPSILON) return [0, 1, 0];
  const horizontalTarget = Math.sqrt(Math.max(0, 1 - aMin * aMin));
  return [axis[0] / horizontal * horizontalTarget, aMin, axis[2] / horizontal * horizontalTarget];
}

function splitForAxisConditioning(
  items: readonly AxisItem[],
  toleranceMetres: number,
  aMin: number,
): { items: SplitAxisItem[]; splitSourceSegments: number; maximumParts: number } {
  const result: SplitAxisItem[] = [];
  let splitSourceSegments = 0;
  let maximumParts = 1;
  for (const item of items) {
    if (!item.start || !item.end || !item.compatibilityKey) {
      throw new Error('interval grouping requires authored segment endpoints and a compatibility key');
    }
    const nearest = nearestAllowedAxis(item.axis, aMin);
    const wholeResidual = item.lengthMetres * 0.5 * Math.sin(unorientedAngle(item.axis, nearest));
    const parts = Math.max(1, Math.ceil(wholeResidual / toleranceMetres - 1e-12));
    if (parts > 1) splitSourceSegments++;
    maximumParts = Math.max(maximumParts, parts);
    for (let part = 0; part < parts; part++) {
      const t0 = part / parts;
      const t1 = (part + 1) / parts;
      const start = add(item.start, scale(sub(item.end, item.start), t0));
      const end = add(item.start, scale(sub(item.end, item.start), t1));
      result.push({
        componentId: result.length,
        sourceComponentId: item.componentId,
        splitPart: part,
        splitParts: parts,
        axis: item.axis,
        lengthMetres: item.lengthMetres / parts,
        weightMetres2: item.weightMetres2 / parts,
        start,
        end,
        compatibilityKey: item.compatibilityKey,
      });
    }
  }
  return { items: result, splitSourceSegments, maximumParts };
}

function conditionedCatalogue(
  items: readonly SplitAxisItem[],
  toleranceMetres: number,
  aMin: number,
): { axes: Vec3[]; seedComponentIds: number[] } {
  if (items.length === 0) return { axes: [], seedComponentIds: [] };
  const candidates = items.map((item) => nearestAllowedAxis(item.axis, aMin));
  const constrainedOrder = items.map((_, index) => index)
    .filter((index) => items[index]!.lengthMetres > 2 * toleranceMetres + 1e-12)
    .sort((a, b) =>
      angularRadius(items[a]!, toleranceMetres) - angularRadius(items[b]!, toleranceMetres)
      || items[b]!.weightMetres2 - items[a]!.weightMetres2
      || a - b
    );
  if (constrainedOrder.length === 0) {
    return { axes: [candidates[0]!], seedComponentIds: [items[0]!.sourceComponentId] };
  }
  const uncovered = new Uint8Array(items.length);
  uncovered.fill(1);
  let remaining = items.length;
  const axes: Vec3[] = [];
  const seedComponentIds: number[] = [];
  while (remaining > 0) {
    const frontier: number[] = [];
    for (const index of constrainedOrder) {
      if (!uncovered[index]) continue;
      frontier.push(index);
      if (frontier.length >= LARGE_COVER_PROBE_COUNT) break;
    }
    if (frontier.length === 0) {
      // Only direction-unconstrained atoms remain; every existing axis covers
      // them. If this is the first iteration, one conditioned axis is enough.
      if (axes.length === 0) {
        axes.push(candidates[0]!);
        seedComponentIds.push(items[0]!.sourceComponentId);
      }
      break;
    }
    let best = frontier[0]!;
    let bestScore = -1;
    for (const candidate of frontier) {
      let score = 0;
      const axis = candidates[candidate]!;
      for (let item = 0; item < items.length; item++) {
        if (uncovered[item] && covers(axis, items[item]!, toleranceMetres)) {
          score += items[item]!.weightMetres2;
        }
      }
      if (score > bestScore + EPSILON || (Math.abs(score - bestScore) <= EPSILON && candidate < best)) {
        best = candidate;
        bestScore = score;
      }
    }
    const axis = candidates[best]!;
    axes.push(axis);
    seedComponentIds.push(items[best]!.sourceComponentId);
    let coveredNow = 0;
    for (let item = 0; item < items.length; item++) {
      if (uncovered[item] && covers(axis, items[item]!, toleranceMetres)) {
        uncovered[item] = 0;
        remaining--;
        coveredNow++;
      }
    }
    if (coveredNow === 0) throw new Error('conditioned catalogue failed to cover its own seed');
  }
  return { axes, seedComponentIds };
}

interface AxialRectangle {
  start: number;
  end: number;
  radius: number;
  axisIndex: number;
  compatibilityKey: string;
}

function oneDimensionalPiercingCount(
  rectangles: readonly AxialRectangle[],
  coordinate: 'start' | 'end',
): number {
  const intervals = rectangles.map((rectangle) => ({
    low: rectangle[coordinate] - rectangle.radius,
    high: rectangle[coordinate] + rectangle.radius,
  })).sort((left, right) => left.high - right.high || left.low - right.low);
  let count = 0;
  let point = Number.NEGATIVE_INFINITY;
  for (const interval of intervals) {
    if (point >= interval.low - 1e-12 && point <= interval.high + 1e-12) continue;
    point = interval.high;
    count++;
  }
  return count;
}

function fixedCentreRectangleGroups(
  rectangles: readonly AxialRectangle[],
  toleranceMetres: number,
): number {
  if (rectangles.length === 0) return 0;
  const cellSize = Math.max(toleranceMetres, 1e-12);
  const centres: Array<{ start: number; end: number }> = [];
  const buckets = new Map<string, number[]>();
  for (const rectangle of rectangles) {
    const startLow = Math.floor((rectangle.start - rectangle.radius) / cellSize);
    const startHigh = Math.floor((rectangle.start + rectangle.radius) / cellSize);
    const endLow = Math.floor((rectangle.end - rectangle.radius) / cellSize);
    const endHigh = Math.floor((rectangle.end + rectangle.radius) / cellSize);
    let matched = false;
    for (let startCell = startLow; startCell <= startHigh && !matched; startCell++) {
      for (let endCell = endLow; endCell <= endHigh && !matched; endCell++) {
        for (const group of buckets.get(`${startCell}:${endCell}`) ?? []) {
          const centre = centres[group]!;
          if (
            Math.abs(centre.start - rectangle.start) <= rectangle.radius + 1e-12
            && Math.abs(centre.end - rectangle.end) <= rectangle.radius + 1e-12
          ) {
            matched = true;
            break;
          }
        }
      }
    }
    if (matched) continue;
    const group = centres.length;
    centres.push({ start: rectangle.start, end: rectangle.end });
    const key = `${Math.floor(rectangle.start / cellSize)}:${Math.floor(rectangle.end / cellSize)}`;
    let target = buckets.get(key);
    if (!target) {
      target = [];
      buckets.set(key, target);
    }
    target.push(group);
  }
  return centres.length;
}

function intervalCompatiblePareto(
  items: readonly AxisItem[],
  tolerancesMetres: readonly number[] = TOLERANCES_METRES,
): Array<Record<string, unknown>> {
  const results: Array<Record<string, unknown>> = [];
  for (const toleranceMetres of tolerancesMetres) {
    for (const aMin of A_MIN_GRID) {
      const split = splitForAxisConditioning(items, toleranceMetres, aMin);
      const catalogue = conditionedCatalogue(split.items, toleranceMetres, aMin);
      const rectanglesByGroup = new Map<string, AxialRectangle[]>();
      const axisResiduals: number[] = [];
      const capBudgets: number[] = [];
      for (const item of split.items) {
        let winner = -1;
        let winnerResidual = Number.POSITIVE_INFINITY;
        for (let axisIndex = 0; axisIndex < catalogue.axes.length; axisIndex++) {
          const residual = item.lengthMetres * 0.5
            * Math.sin(unorientedAngle(item.axis, catalogue.axes[axisIndex]!));
          if (residual < winnerResidual) {
            winner = axisIndex;
            winnerResidual = residual;
          }
        }
        if (winner < 0 || winnerResidual > toleranceMetres + 1e-10) {
          throw new Error(`conditioned catalogue left segment ${item.componentId} uncovered`);
        }
        const axis = catalogue.axes[winner]!;
        const midpoint = scale(add(item.start, item.end), 0.5);
        const axialHalfLength = item.lengthMetres * 0.5 * Math.abs(dot(item.axis, axis));
        const axialCentre = midpoint[1] / axis[1];
        const capBudget = Math.sqrt(Math.max(0, toleranceMetres * toleranceMetres - winnerResidual * winnerResidual));
        const rectangle: AxialRectangle = {
          start: axialCentre - axialHalfLength,
          end: axialCentre + axialHalfLength,
          radius: capBudget,
          axisIndex: winner,
          compatibilityKey: item.compatibilityKey,
        };
        const key = `${winner}:${item.compatibilityKey}`;
        let target = rectanglesByGroup.get(key);
        if (!target) {
          target = [];
          rectanglesByGroup.set(key, target);
        }
        target.push(rectangle);
        axisResiduals.push(winnerResidual);
        capBudgets.push(capBudget);
      }
      let fixedCatalogueIntervalFieldLowerBound = 0;
      let constructiveFixedCentreIntervalFields = 0;
      const byAxis: Array<Record<string, unknown>> = [];
      for (const [key, rectangles] of rectanglesByGroup) {
        const startLower = oneDimensionalPiercingCount(rectangles, 'start');
        const endLower = oneDimensionalPiercingCount(rectangles, 'end');
        const lower = Math.max(startLower, endLower);
        const upper = fixedCentreRectangleGroups(rectangles, toleranceMetres);
        fixedCatalogueIntervalFieldLowerBound += lower;
        constructiveFixedCentreIntervalFields += upper;
        byAxis.push({ key, atoms: rectangles.length, startPiercingLowerBound: startLower, endPiercingLowerBound: endLower, fieldLowerBound: lower, constructiveFields: upper });
      }
      const lowerReads = TWO_GLOBAL_LAYERS * fixedCatalogueIntervalFieldLowerBound + OPTIMISTIC_NON_FIELD_READS;
      const upperReads = TWO_GLOBAL_LAYERS * constructiveFixedCentreIntervalFields + OPTIMISTIC_NON_FIELD_READS;
      results.push({
        toleranceMetres,
        aMin,
        sourceSegments: items.length,
        conditionedSegments: split.items.length,
        sourceSegmentsSplitForAmin: split.splitSourceSegments,
        maximumSplitParts: split.maximumParts,
        catalogueAxes: catalogue.axes.length,
        catalogueAxisVectors: catalogue.axes,
        catalogueSeedSourceSegments: catalogue.seedComponentIds,
        axisResidualMetres: quantiles(axisResiduals),
        remainingPerCapBudgetMetres: quantiles(capBudgets),
        fixedCatalogueIntervalFieldLowerBound,
        constructiveFixedCentreIntervalFields,
        optimisticTwoLayerReadLowerBound: lowerReads,
        constructiveTwoLayerReads: upperReads,
        nineReadVerdictForThisCatalogue: lowerReads > PROVISIONAL_READ_CEILING
          ? 'RED_FIXED_CATALOGUE_INTERVAL_LOWER_BOUND'
          : 'NOT_REFUTED_FOR_THIS_CATALOGUE',
        byAxis,
      });
    }
  }
  return results;
}

function scalarPiercingCount(values: readonly number[], radius: number): number {
  const intervals = values.map((value) => ({ low: value - radius, high: value + radius }))
    .sort((left, right) => left.high - right.high || left.low - right.low);
  let point = Number.NEGATIVE_INFINITY;
  let count = 0;
  for (const interval of intervals) {
    if (point >= interval.low - 1e-12 && point <= interval.high + 1e-12) continue;
    point = interval.high;
    count++;
  }
  return count;
}

function mandatoryPrimitiveCapLowerBound(
  recipes: readonly GraminoidPrimitiveRecipe[],
  tolerancesMetres: readonly number[] = TOLERANCES_METRES,
): Array<Record<string, unknown>> {
  const endpoints: number[] = [];
  const primitiveKinds: Record<string, number> = {};
  for (const recipe of recipes) {
    if (recipe.disposition !== 'crisp' || !('centerline' in recipe) || recipe.centerline.length < 2) continue;
    endpoints.push(recipe.centerline[0]!.y, recipe.centerline.at(-1)!.y);
    const semantic = recipeSemantic(recipe);
    primitiveKinds[semantic] = (primitiveKinds[semantic] ?? 0) + 1;
  }
  return tolerancesMetres.map((toleranceMetres) => {
    const requiredDistinctHorizontalCapPlanes = scalarPiercingCount(endpoints, toleranceMetres);
    // Every M x I field supplies at most two horizontal world cap planes on a
    // flat ground chart. Allow either source endpoint to use either plane;
    // ceil(P/2) is therefore deliberately more optimistic than preserving
    // root/tip orientation or endpoint pairing.
    const opaqueFieldLowerBound = Math.ceil(requiredDistinctHorizontalCapPlanes / 2);
    const optimisticTwoLayerReadLowerBound =
      TWO_GLOBAL_LAYERS * opaqueFieldLowerBound + OPTIMISTIC_NON_FIELD_READS;
    return {
      toleranceMetres,
      primitives: endpoints.length / 2,
      primitiveKinds,
      endpoints: endpoints.length,
      requiredDistinctHorizontalCapPlanes,
      opaqueFieldLowerBound,
      optimisticTwoLayerReadLowerBound,
      verdict: optimisticTwoLayerReadLowerBound > PROVISIONAL_READ_CEILING
        ? 'RED_GLOBAL_CAP_PLANE_LOWER_BOUND'
        : 'NOT_REFUTED_BY_CAP_PLANES',
    };
  });
}

function forcedNonhorizontalReport(items: readonly AxisItem[]): Array<Record<string, unknown>> {
  const totalArea = items.reduce((sum, item) => sum + item.weightMetres2, 0);
  const totalLength = items.reduce((sum, item) => sum + item.lengthMetres, 0);
  return A_MIN_GRID.map((aMin) => {
    const residuals = items.map((item) => Math.abs(item.axis[1]) >= aMin
      ? 0
      : item.lengthMetres * 0.5 * Math.sin(unorientedAngle(item.axis, nearestAllowedAxis(item.axis, aMin))));
    return {
      aMin,
      affectedComponents: items.filter((item) => Math.abs(item.axis[1]) < aMin).length,
      affectedSurfaceAreaFraction: totalArea > 0
        ? items.reduce((sum, item) => sum + (Math.abs(item.axis[1]) < aMin ? item.weightMetres2 : 0), 0) / totalArea
        : 0,
      affectedLengthFraction: totalLength > 0
        ? items.reduce((sum, item) => sum + (Math.abs(item.axis[1]) < aMin ? item.lengthMetres : 0), 0) / totalLength
        : 0,
      forcedAxisResidualMetres: quantiles(residuals),
      fractionWithinTolerance: Object.fromEntries(TOLERANCES_METRES.map((tolerance) => [
        String(tolerance),
        residuals.length > 0 ? residuals.filter((residual) => residual <= tolerance).length / residuals.length : 1,
      ])),
    };
  });
}

function selectionArea(selection: MeshSelection): number {
  let area = 0;
  for (let index = 0; index < selection.indices.length; index += 3) {
    area += triangleArea(
      selection.positions,
      selection.indices[index]!,
      selection.indices[index + 1]!,
      selection.indices[index + 2]!,
    );
  }
  return area;
}

function bounds(positions: ArrayLike<number>, indices: ArrayLike<number>): { minimum: Vec3; maximum: Vec3 } {
  let minimum: Vec3 = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  let maximum: Vec3 = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
  const unique = new Set<number>();
  for (let index = 0; index < indices.length; index++) unique.add(indices[index]!);
  for (const index of unique) {
    const point = vertex(positions, index);
    minimum = [Math.min(minimum[0], point[0]), Math.min(minimum[1], point[1]), Math.min(minimum[2], point[2])];
    maximum = [Math.max(maximum[0], point[0]), Math.max(maximum[1], point[1]), Math.max(maximum[2], point[2])];
  }
  return { minimum, maximum };
}

function summarizeComponents(components: readonly ComponentMeasurement[]): Record<string, unknown> {
  const eligible = components.filter((component) => component.axisEligible);
  const ambiguous = components.filter((component) => !component.axisEligible);
  const items: AxisItem[] = eligible.map((component) => ({
    componentId: component.id,
    axis: component.chordAxis,
    lengthMetres: component.chordLengthMetres,
    weightMetres2: component.surfaceAreaMetres2,
  }));
  const totalArea = components.reduce((sum, component) => sum + component.surfaceAreaMetres2, 0);
  const eligibleArea = eligible.reduce((sum, component) => sum + component.surfaceAreaMetres2, 0);
  return {
    edgeConnectedComponents: components.length,
    axisEligibleComponents: eligible.length,
    axisAmbiguousComponents: ambiguous.length,
    totalSurfaceAreaMetres2: totalArea,
    axisEligibleSurfaceAreaFraction: totalArea > 0 ? eligibleArea / totalArea : 1,
    componentTriangles: quantiles(components.map((component) => component.triangles)),
    componentChordLengthMetres: quantiles(components.map((component) => component.chordLengthMetres)),
    componentAxialElongation: quantiles(components.map((component) => component.axialElongation)),
    vertexDistanceToChordMetres: quantiles(components.map((component) => component.maximumVertexDistanceToChordMetres)),
    axisCurve: coverCurve(items, 'one chord per semantic edge-connected component'),
    forcedNonhorizontal: forcedNonhorizontalReport(items),
    components,
  };
}

function recipeSemantic(recipe: GraminoidPrimitiveRecipe): string {
  if ('role' in recipe) return `${recipe.kind}:${recipe.role}`;
  return recipe.kind;
}

function authoredPoint(point: { x: number; y: number; z: number }): Vec3 {
  return [point.x, point.y, point.z];
}

function recipeSegments(
  recipes: readonly GraminoidPrimitiveRecipe[],
  positions: readonly number[],
  indices: readonly number[],
  disposition: 'crisp' | 'plume' = 'crisp',
): RecipeSegmentSummary {
  const triangleAreas = new Float64Array(indices.length / 3);
  for (let triangle = 0; triangle < triangleAreas.length; triangle++) {
    const offset = triangle * 3;
    triangleAreas[triangle] = triangleArea(
      positions,
      indices[offset]!,
      indices[offset + 1]!,
      indices[offset + 2]!,
    );
  }
  const items: AxisItem[] = [];
  let primitiveCount = 0;
  let zeroLengthSegments = 0;
  let sourceTriangles = 0;
  let sourceSurfaceAreaMetres2 = 0;
  const bySemantic: RecipeSegmentSummary['bySemantic'] = {};
  for (const recipe of recipes) {
    if (recipe.disposition !== disposition || !('centerline' in recipe)) continue;
    primitiveCount++;
    const semantic = recipeSemantic(recipe);
    const target = bySemantic[semantic] ??= {
      primitives: 0,
      segments: 0,
      sourceTriangles: 0,
      sourceSurfaceAreaMetres2: 0,
      segmentLengthsMetres: [],
    };
    target.primitives++;
    target.sourceTriangles += recipe.sourceTriangleCount;
    sourceTriangles += recipe.sourceTriangleCount;
    let primitiveArea = 0;
    for (
      let triangle = recipe.sourceTriangleStart;
      triangle < recipe.sourceTriangleStart + recipe.sourceTriangleCount;
      triangle++
    ) primitiveArea += triangleAreas[triangle]!;
    target.sourceSurfaceAreaMetres2 += primitiveArea;
    sourceSurfaceAreaMetres2 += primitiveArea;
    const segments = [] as Array<{ axis: Vec3; lengthMetres: number; start: Vec3; end: Vec3 }>;
    let primitiveLength = 0;
    for (let segment = 0; segment < recipe.centerline.length - 1; segment++) {
      const start = authoredPoint(recipe.centerline[segment]!);
      const end = authoredPoint(recipe.centerline[segment + 1]!);
      const delta = sub(end, start);
      const lengthMetres = length(delta);
      if (lengthMetres <= EPSILON) {
        zeroLengthSegments++;
        continue;
      }
      segments.push({ axis: canonicalAxis(delta), lengthMetres, start, end });
      primitiveLength += lengthMetres;
      target.segmentLengthsMetres.push(lengthMetres);
    }
    target.segments += segments.length;
    for (const segment of segments) {
      items.push({
        componentId: items.length,
        axis: segment.axis,
        lengthMetres: segment.lengthMetres,
        weightMetres2: primitiveArea * segment.lengthMetres / Math.max(EPSILON, primitiveLength),
        start: segment.start,
        end: segment.end,
        compatibilityKey: 'one-packed-opaque-owner-and-attribute-layout',
      });
    }
  }
  return {
    items,
    primitiveCount,
    segmentCount: items.length,
    zeroLengthSegments,
    sourceTriangles,
    sourceSurfaceAreaMetres2,
    bySemantic,
  };
}

function sphagnumRecipeSegments(
  recipes: readonly SphagnumPrimitiveRecipe[],
  positions: readonly number[],
  indices: readonly number[],
  disposition: 'crisp' | 'medium-candidate',
): RecipeSegmentSummary {
  const triangleAreas = new Float64Array(indices.length / 3);
  for (let triangle = 0; triangle < triangleAreas.length; triangle++) {
    const offset = triangle * 3;
    triangleAreas[triangle] = triangleArea(
      positions,
      indices[offset]!,
      indices[offset + 1]!,
      indices[offset + 2]!,
    );
  }
  const items: AxisItem[] = [];
  let primitiveCount = 0;
  let zeroLengthSegments = 0;
  let sourceTriangles = 0;
  let sourceSurfaceAreaMetres2 = 0;
  const bySemantic: RecipeSegmentSummary['bySemantic'] = {};
  for (const recipe of recipes) {
    if (recipe.disposition !== disposition || !('centerline' in recipe)) continue;
    primitiveCount++;
    const semantic = `${recipe.kind}:${recipe.classEAxisCandidacy}`;
    const target = bySemantic[semantic] ??= {
      primitives: 0,
      segments: 0,
      sourceTriangles: 0,
      sourceSurfaceAreaMetres2: 0,
      segmentLengthsMetres: [],
    };
    target.primitives++;
    target.sourceTriangles += recipe.sourceTriangleCount;
    sourceTriangles += recipe.sourceTriangleCount;
    let primitiveArea = 0;
    for (
      let triangle = recipe.sourceTriangleStart;
      triangle < recipe.sourceTriangleStart + recipe.sourceTriangleCount;
      triangle++
    ) primitiveArea += triangleAreas[triangle]!;
    target.sourceSurfaceAreaMetres2 += primitiveArea;
    sourceSurfaceAreaMetres2 += primitiveArea;
    const segments: Array<{ axis: Vec3; lengthMetres: number; start: Vec3; end: Vec3 }> = [];
    let primitiveLength = 0;
    for (let segment = 0; segment < recipe.centerline.length - 1; segment++) {
      const start = authoredPoint(recipe.centerline[segment]!);
      const end = authoredPoint(recipe.centerline[segment + 1]!);
      const delta = sub(end, start);
      const lengthMetres = length(delta);
      if (lengthMetres <= EPSILON) {
        zeroLengthSegments++;
        continue;
      }
      segments.push({ axis: canonicalAxis(delta), lengthMetres, start, end });
      primitiveLength += lengthMetres;
      target.segmentLengthsMetres.push(lengthMetres);
    }
    target.segments += segments.length;
    for (const segment of segments) {
      items.push({
        componentId: items.length,
        axis: segment.axis,
        lengthMetres: segment.lengthMetres,
        weightMetres2: primitiveArea * segment.lengthMetres / Math.max(EPSILON, primitiveLength),
        start: segment.start,
        end: segment.end,
        compatibilityKey: 'one-packed-opaque-owner-and-attribute-layout',
      });
    }
  }
  return {
    items,
    primitiveCount,
    segmentCount: items.length,
    zeroLengthSegments,
    sourceTriangles,
    sourceSurfaceAreaMetres2,
    bySemantic,
  };
}

function summarizeRecipeSegments(
  summary: RecipeSegmentSummary,
  frozenAtomization = 'generator-authored centerline intervals',
): Record<string, unknown> {
  return {
    crispPrimitivesWithCenterlines: summary.primitiveCount,
    authoredChordSegments: summary.segmentCount,
    zeroLengthSegments: summary.zeroLengthSegments,
    sourceTriangles: summary.sourceTriangles,
    sourceSurfaceAreaMetres2: summary.sourceSurfaceAreaMetres2,
    segmentLengthMetres: quantiles(summary.items.map((item) => item.lengthMetres)),
    bySemantic: Object.fromEntries(Object.entries(summary.bySemantic).map(([semantic, value]) => [semantic, {
      primitives: value.primitives,
      segments: value.segments,
      sourceTriangles: value.sourceTriangles,
      sourceSurfaceAreaMetres2: value.sourceSurfaceAreaMetres2,
      segmentLengthMetres: quantiles(value.segmentLengthsMetres),
    }])),
    axisCurve: coverCurve(summary.items, frozenAtomization),
    forcedNonhorizontal: forcedNonhorizontalReport(summary.items),
  };
}

function summarizeNonOpaqueSegments(summary: RecipeSegmentSummary): Record<string, unknown> {
  return {
    primitivesWithCenterlines: summary.primitiveCount,
    authoredSegments: summary.segmentCount,
    zeroLengthSegments: summary.zeroLengthSegments,
    sourceTriangles: summary.sourceTriangles,
    sourceSurfaceAreaMetres2: summary.sourceSurfaceAreaMetres2,
    segmentLengthMetres: quantiles(summary.items.map((item) => item.lengthMetres)),
    bySemantic: Object.fromEntries(Object.entries(summary.bySemantic).map(([semantic, value]) => [semantic, {
      primitives: value.primitives,
      segments: value.segments,
      sourceTriangles: value.sourceTriangles,
      sourceSurfaceAreaMetres2: value.sourceSurfaceAreaMetres2,
      segmentLengthMetres: quantiles(value.segmentLengthsMetres),
    }])),
  };
}

function validateFixtureAgainstTransport(
  positions: Float32Array,
  fullIndices: Uint32Array,
  fixturePositions: readonly number[],
  fixtureIndices: readonly number[],
): { positionMismatches: number; indexMismatches: number } {
  if (positions.length !== fixturePositions.length || fullIndices.length !== fixtureIndices.length) {
    throw new Error('primitive-recipe fixture and accepted isolated transport have different dimensions');
  }
  let positionMismatches = 0;
  for (let index = 0; index < positions.length; index++) {
    if (positions[index] !== Math.fround(fixturePositions[index]!)) positionMismatches++;
  }
  let indexMismatches = 0;
  for (let index = 0; index < fullIndices.length; index++) {
    if (fullIndices[index] !== fixtureIndices[index]) indexMismatches++;
  }
  if (positionMismatches !== 0 || indexMismatches !== 0) {
    throw new Error(`primitive-recipe fixture does not match accepted transport: ${positionMismatches} positions, ${indexMismatches} indices`);
  }
  return { positionMismatches, indexMismatches };
}

function partitionConfusion(
  recipes: readonly GraminoidPrimitiveRecipe[],
  fullIndices: Uint32Array,
  acceptedStructureIndices: Uint32Array,
  acceptedPlumeIndices: Uint32Array,
): Record<string, unknown> {
  const triangleCount = fullIndices.length / 3;
  const recipeDisposition = new Int8Array(triangleCount);
  recipeDisposition.fill(-1);
  let overlappingRecipeTriangles = 0;
  for (const recipe of recipes) {
    for (
      let triangle = recipe.sourceTriangleStart;
      triangle < recipe.sourceTriangleStart + recipe.sourceTriangleCount;
      triangle++
    ) {
      if (recipeDisposition[triangle] !== -1) overlappingRecipeTriangles++;
      recipeDisposition[triangle] = recipe.disposition === 'crisp' ? 0 : 1;
    }
  }
  const acceptedDisposition = new Int8Array(triangleCount);
  acceptedDisposition.fill(-1);
  const consume = (selection: Uint32Array, disposition: number): void => {
    let selectedTriangle = 0;
    for (let triangle = 0; triangle < triangleCount && selectedTriangle < selection.length / 3; triangle++) {
      const full = triangle * 3;
      const selected = selectedTriangle * 3;
      if (
        fullIndices[full] === selection[selected]
        && fullIndices[full + 1] === selection[selected + 1]
        && fullIndices[full + 2] === selection[selected + 2]
      ) {
        acceptedDisposition[triangle] = disposition;
        selectedTriangle++;
      }
    }
    if (selectedTriangle !== selection.length / 3) {
      throw new Error('accepted structure/plume triangle stream is not an ordered partition of the full transport');
    }
  };
  consume(acceptedStructureIndices, 0);
  consume(acceptedPlumeIndices, 1);
  let uncoveredRecipeTriangles = 0;
  let uncoveredAcceptedTriangles = 0;
  const confusion = { crispCrisp: 0, crispOldPlume: 0, plumeOldCrisp: 0, plumePlume: 0 };
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const recipe = recipeDisposition[triangle]!;
    const accepted = acceptedDisposition[triangle]!;
    if (recipe < 0) uncoveredRecipeTriangles++;
    if (accepted < 0) uncoveredAcceptedTriangles++;
    if (recipe === 0 && accepted === 0) confusion.crispCrisp++;
    else if (recipe === 0 && accepted === 1) confusion.crispOldPlume++;
    else if (recipe === 1 && accepted === 0) confusion.plumeOldCrisp++;
    else if (recipe === 1 && accepted === 1) confusion.plumePlume++;
  }
  return {
    ...confusion,
    overlappingRecipeTriangles,
    uncoveredRecipeTriangles,
    uncoveredAcceptedTriangles,
    interpretation: 'recipe disposition follows Class-E semantic crisp/plume intent; old transport partition is the accepted colour-majority structure/plume artifact and is reported rather than silently treated as equivalent',
  };
}

const scriptPath = fileURLToPath(import.meta.url);
const bandlimitedIndexPath = resolve(BANDLIMITED_ROOT, 'qa/index.json');
const bandlimitedIndexBytes = readFileSync(bandlimitedIndexPath);
const bandlimitedIndexSha256 = sha256(bandlimitedIndexBytes);
if (bandlimitedIndexSha256 !== EXPECTED_BANDLIMITED_INDEX_SHA256) {
  throw new Error(`accepted band-limited index hash changed: ${bandlimitedIndexSha256}`);
}
const sphagnumPackedBytes = readFileSync(SPHAGNUM_PACKED);
const sphagnumPackedSha256 = sha256(sphagnumPackedBytes);
if (sphagnumPackedSha256 !== EXPECTED_SPHAGNUM_PACKED_SHA256) {
  throw new Error(`accepted Sphagnum packed hash changed: ${sphagnumPackedSha256}`);
}
const productionCalamagrostisPackedBytes = readFileSync(PRODUCTION_CALAMAGROSTIS_PACKED);
const productionCalamagrostisPackedSha256 = sha256(productionCalamagrostisPackedBytes);
if (productionCalamagrostisPackedSha256 !== EXPECTED_PRODUCTION_CALAMAGROSTIS_PACKED_SHA256) {
  throw new Error(`accepted production Calamagrostis packed hash changed: ${productionCalamagrostisPackedSha256}`);
}

const transportRoot = resolve(BANDLIMITED_ROOT, 'transport');
const tallPositionsPath = resolve(transportRoot, 'positions.f32');
const tallFullIndicesPath = resolve(transportRoot, 'indices.u32');
const tallStructureIndicesPath = resolve(transportRoot, 'structure-indices.u32');
const tallPlumeIndicesPath = resolve(transportRoot, 'plume-indices.u32');
const tallPositions = float32File(tallPositionsPath);
const tallFullIndices = uint32File(tallFullIndicesPath);
const tallStructureIndices = uint32File(tallStructureIndicesPath);
const tallPlumeIndices = uint32File(tallPlumeIndicesPath);
const tallStructure: MeshSelection = {
  positions: tallPositions,
  indices: tallStructureIndices,
  label: 'accepted isolated Calamagrostis crisp structure',
  semantic: 'accepted G>=R opaque structure partition; edge-connected components only',
};
const tallPlume: MeshSelection = {
  positions: tallPositions,
  indices: tallPlumeIndices,
  label: 'accepted isolated Calamagrostis plume',
  semantic: 'accepted R>G reproductive partition; excluded from opaque-axis sizing',
};

console.error('[class-e-structural] measuring isolated Calamagrostis crisp components');
const tallComponents = measureSelection(tallStructure);
console.error('[class-e-structural] regenerating Calamagrostis primitive recipes');
const tallFixture = makeCalamagrostisCanescensBandlimitedSource();
const tallFixtureBinding = validateFixtureAgainstTransport(
  tallPositions,
  tallFullIndices,
  tallFixture.mesh.positions,
  tallFixture.mesh.indices,
);
const tallPartitionConfusion = partitionConfusion(
  tallFixture.primitiveRecipes,
  tallFullIndices,
  tallStructureIndices,
  tallPlumeIndices,
);
const tallRecipeSegments = recipeSegments(
  tallFixture.primitiveRecipes,
  tallFixture.mesh.positions,
  tallFixture.mesh.indices,
);
const tallMandatorySkeletonSegments = recipeSegments(
  tallFixture.primitiveRecipes.filter((recipe) =>
    recipe.kind === 'blade'
    || (recipe.kind === 'tube' && recipe.role === 'culm')
  ),
  tallFixture.mesh.positions,
  tallFixture.mesh.indices,
);
const tallSemanticPlumeSegments = recipeSegments(
  tallFixture.primitiveRecipes,
  tallFixture.mesh.positions,
  tallFixture.mesh.indices,
  'plume',
);
console.error('[class-e-structural] regenerating accepted production Calamagrostis recipes');
const productionTallFixture = makeEstonianGraminoidFixture(
  ESTONIAN_GRAMINOID_PROFILE_IDS.CALAMAGROSTIS_CANESCENS,
);
const productionTallMeshSha256 = sha256(JSON.stringify(productionTallFixture.mesh));
if (productionTallMeshSha256 !== EXPECTED_PRODUCTION_CALAMAGROSTIS_MESH_SHA256) {
  throw new Error(`accepted production Calamagrostis source mesh changed: ${productionTallMeshSha256}`);
}
const productionTallCrispSegments = recipeSegments(
  productionTallFixture.primitiveRecipes,
  productionTallFixture.mesh.positions,
  productionTallFixture.mesh.indices,
);
const productionTallPlumeSegments = recipeSegments(
  productionTallFixture.primitiveRecipes,
  productionTallFixture.mesh.positions,
  productionTallFixture.mesh.indices,
  'plume',
);

console.error('[class-e-structural] regenerating accepted dense Sphagnum fixture');
const sphagnum = makeSphagnumCapillifoliumFixture();
const sphagnumPositions = sphagnum.mesh.positions;
const sphagnumCarpetIndices = sphagnum.mesh.indices.slice(0, sphagnum.carpetTriangleCount * 3);
const sphagnumCapitulaIndices = sphagnum.mesh.indices.slice(sphagnum.carpetTriangleCount * 3);
const sphagnumCarpet: MeshSelection = {
  positions: sphagnumPositions,
  indices: sphagnumCarpetIndices,
  label: 'dense Sphagnum connected carpet',
  semantic: 'generator-explicit carpet; reported separately from axial capitulum structures',
};
const sphagnumCapitula: MeshSelection = {
  positions: sphagnumPositions,
  indices: sphagnumCapitulaIndices,
  label: 'dense Sphagnum capitula',
  semantic: 'generator-explicit non-carpet geometry; no accepted crisp/fuzz partition exists',
};
console.error('[class-e-structural] measuring Sphagnum capitulum components');
const sphagnumComponents = measureSelection(sphagnumCapitula);
const sphagnumCrispSegments = sphagnumRecipeSegments(
  sphagnum.primitiveRecipes,
  sphagnum.mesh.positions,
  sphagnum.mesh.indices,
  'crisp',
);
const sphagnumMediumSegments = sphagnumRecipeSegments(
  sphagnum.primitiveRecipes,
  sphagnum.mesh.positions,
  sphagnum.mesh.indices,
  'medium-candidate',
);

const sourceSet = {
  bandlimitedIndex: {
    file: relative(REPO_ROOT, bandlimitedIndexPath),
    sha256: bandlimitedIndexSha256,
  },
  tallPositions: { file: relative(REPO_ROOT, tallPositionsPath), sha256: sha256(readFileSync(tallPositionsPath)) },
  tallFullIndices: { file: relative(REPO_ROOT, tallFullIndicesPath), sha256: sha256(readFileSync(tallFullIndicesPath)) },
  tallStructureIndices: { file: relative(REPO_ROOT, tallStructureIndicesPath), sha256: sha256(readFileSync(tallStructureIndicesPath)) },
  tallPlumeIndices: { file: relative(REPO_ROOT, tallPlumeIndicesPath), sha256: sha256(readFileSync(tallPlumeIndicesPath)) },
  tallPrimitiveRecipeGenerator: {
    file: 'tools/groundcover-bake/EstonianGraminoids.ts',
    sha256: sha256(readFileSync(resolve(REPO_ROOT, 'tools/groundcover-bake/EstonianGraminoids.ts'))),
    recipe: tallFixture.generator,
  },
  sphagnumPacked: { file: relative(REPO_ROOT, SPHAGNUM_PACKED), sha256: sphagnumPackedSha256 },
  productionCalamagrostisPacked: {
    file: relative(REPO_ROOT, PRODUCTION_CALAMAGROSTIS_PACKED),
    sha256: productionCalamagrostisPackedSha256,
    sourceMeshSha256: productionTallMeshSha256,
  },
  sphagnumGenerator: {
    file: 'tools/groundcover-bake/SphagnumCapillifolium.ts',
    sha256: sha256(readFileSync(resolve(REPO_ROOT, 'tools/groundcover-bake/SphagnumCapillifolium.ts'))),
    recipe: sphagnum.generator,
  },
};
const sourceSetSha256 = sha256(canonicalJson(sourceSet));
const scriptSha256 = sha256(readFileSync(scriptPath));
const recipe = {
  schema: 'laas-class-e-structural-feasibility/v1',
  sourceSetSha256,
  scriptSha256,
  tolerancesMetres: TOLERANCES_METRES,
  mandatorySkeletonSensitivityTolerancesMetres: STRUCTURAL_SENSITIVITY_TOLERANCES_METRES,
  aMinGrid: A_MIN_GRID,
  axialElongationMinimum: AXIAL_ELONGATION_MINIMUM,
  componentRule: 'generator-authored primitive centreline intervals for the binding candidate; shared-edge PCA components are diagnostic only',
  axisRule: 'endpoint-band chord initialized by vertex PCA; axes are unoriented',
  axisResidual: '(chordLength/2)*sin(unorientedAngle)',
  packingRule: 'largest of four deterministic greedy pairwise-incompatible packings over a named frozen atomization; conditional lower bound because further segmentation shortens ell',
  coverRule: 'surface-area-weighted greedy catalogue over the named frozen atoms; constructive upper bound for that model, not a global joint axis/interval optimum',
  cost: { layers: TWO_GLOBAL_LAYERS, optimisticNonFieldReads: OPTIMISTIC_NON_FIELD_READS, ceiling: PROVISIONAL_READ_CEILING },
};
const recipeSha256 = sha256(canonicalJson(recipe));
const outputRoot = resolve(
  REPO_ROOT,
  'data/work/class-e-structural-feasibility',
  sourceSetSha256.slice(0, 16),
  recipeSha256.slice(0, 16),
);
mkdirSync(outputRoot, { recursive: true });

const report = {
  schema: recipe.schema,
  status: 'RED_STRUCTURAL_FIELD_BUDGET',
  purpose: 'CPU/offline Gate-B structural axis feasibility; no compiler, reconstruction, runtime, shader, or render-pipeline claim',
  reproducibility: {
    sourceSetSha256,
    recipeSha256,
    script: relative(REPO_ROOT, scriptPath),
    scriptSha256,
    sources: sourceSet,
    recipe,
  },
  mathematics: {
    residual: 'e_axis=(ell/2)*sin(angle(a,a_hat)); the full tolerance is assigned to this term, so curvature/shape/tip/endpoint residuals are optimistically zero',
    conditionalLowerBound: 'if one chord per edge component is frozen, pairwise axes i,j with angle(a_i,a_j)>asin(2eps/ell_i)+asin(2eps/ell_j) cannot share one catalogue axis',
    segmentationCaveat: 'the compiler may split authored centreline intervals into shorter chords; because e_axis scales with ell, axis packing alone is not a lower bound for the freely segmented compiler',
    hardLowerBound: 'segmentation is not free under G_f=A_f(M_f x I_f): every field supplies only one cap pair. The mandatory primitive endpoint cap-plane piercing bound is catalogue/assignment independent and survives arbitrary subdivision. The much larger start/end rectangle piercing number is a valid lower bound for each explicitly emitted catalogue and assignment, but is not claimed globally optimal over a different representation or every possible joint catalogue.',
    upperBound: 'each selected observed axis covers every assigned whole component under e_axis<=eps; weighted greedy is constructive for that conditional model but not globally minimal',
    twoLayerReads: '2*opaqueAxisFields+1; the +1 assumes all interval/control/plume/material costs fit one sample and minification is zero',
  },
  acceptedProductionCalamagrostis: {
    source: {
      packedSha256: productionCalamagrostisPackedSha256,
      sourceMeshSha256: productionTallMeshSha256,
      generator: productionTallFixture.generator,
      vertices: productionTallFixture.mesh.positions.length / 3,
      triangles: productionTallFixture.mesh.indices.length / 3,
      primitiveRecipes: productionTallFixture.primitiveRecipes.length,
      tufts: productionTallFixture.structure.tufts,
    },
    crispRecipeSegmentCandidate: {
      ...summarizeRecipeSegments(productionTallCrispSegments),
      interpretation: [
        'This is the accepted six-shoot periodic community and is the binding tall-community axis Pareto.',
        'Every atom is an authored straight centerline interval; its triangle range and semantic disposition come from the deterministic source recorder.',
        'The curve is still conditional on freezing those intervals: further segmentation can reduce axis error but must pay endpoint/cap/interval compatibility elsewhere.',
      ],
    },
    mandatoryBladeAndCulmGlobalCapPlaneLowerBound: mandatoryPrimitiveCapLowerBound(
      productionTallFixture.primitiveRecipes.filter((recipe) =>
        recipe.kind === 'blade'
        || (recipe.kind === 'tube' && recipe.role === 'culm')
      ),
      STRUCTURAL_SENSITIVITY_TOLERANCES_METRES,
    ),
    semanticPlume: {
      ...summarizeNonOpaqueSegments(productionTallPlumeSegments),
      opaqueAxisSizing: 'excluded',
      analyticBasisFit: 'not performed; mode/stratum count and maximal partial-ray integral residual remain Gate-B blockers',
    },
    optimisticAssumptions: [
      'The complete tolerance is assigned to the axis term; curvature/shape/tip/endpoint residuals are zero.',
      'All atoms assigned one catalogue axis may share one field regardless of cap interval, support, owner, coincident tie, or attribute layout.',
      'The +1 non-field read contains all interval/control/plume/material work and minification costs zero reads.',
    ],
  },
  tallCalamagrostis: {
    source: {
      label: tallStructure.label,
      semantic: tallStructure.semantic,
      vertices: tallPositions.length / 3,
      structureTriangles: tallStructureIndices.length / 3,
      plumeTriangles: tallPlumeIndices.length / 3,
      primitiveRecipes: tallFixture.primitiveRecipes.length,
      recipeFixtureTransportBinding: tallFixtureBinding,
      semanticPartitionComparison: tallPartitionConfusion,
    },
    recipeSegmentCandidate: {
      ...summarizeRecipeSegments(tallRecipeSegments),
      interpretation: [
        'These are the generator-authored straight centerline intervals already used to build the accepted mesh; unlike whole-component PCA, they are a coherent candidate segmentation.',
        'The curve remains conditional because the compiler may further segment them, and because one catalogue axis does not prove compatible cap/interval/support/owner/attribute grouping.',
      ],
    },
    mandatoryBladeAndCulmIntervalPareto: {
      subset: 'all 17 crisp authored blades plus the crisp culm of the accepted isolated production shoot; reproductive structure omitted, so every field/read count is optimistic for the complete plant',
      sourceSegments: tallMandatorySkeletonSegments.segmentCount,
      globalCapPlaneLowerBound: mandatoryPrimitiveCapLowerBound(
        tallFixture.primitiveRecipes.filter((recipe) =>
          recipe.kind === 'blade'
          || (recipe.kind === 'tube' && recipe.role === 'culm')
        ),
        STRUCTURAL_SENSITIVITY_TOLERANCES_METRES,
      ),
      intervalCompatibleCatalogueResults: intervalCompatiblePareto(
        tallMandatorySkeletonSegments.items,
        STRUCTURAL_SENSITIVITY_TOLERANCES_METRES,
      ),
      proofBoundary: [
        'In G_f=A_f(M_f x I_f) on flat ground, each field has at most two horizontal world cap planes. Every retained crisp primitive endpoint must lie within epsilon of one such plane. The one-dimensional endpoint-interval piercing number divided by two is therefore a catalogue-, assignment-, and subdivision-independent field lower bound for this mandatory subset.',
        'The cap-plane bound lets either endpoint use either cap and ignores root/tip pairing, axis, ownership, support, mask conflicts, and every reproductive primitive; it is intentionally optimistic.',
        'For each emitted catalogue, every source segment is split only when the frozen a_min makes its whole-chord axis residual exceed epsilon.',
        'For a chosen catalogue axis b, h_center=y_mid/b_y and half length=(ell/2)|a dot b|. The remaining cap displacement is sqrt(epsilon^2-e_axis^2).',
        'Within each fixed (axis, optimistic shared owner/attribute layout), the maximum of the exact one-dimensional start-interval and end-interval piercing numbers is a valid lower bound on common (h_min,h_max) fields.',
        'The bound is conditional on the emitted catalogue/assignment; it is not asserted as the global optimum over every joint catalogue, subdivision, and interval grouping.',
      ],
    },
    acceptedColourPartitionEdgeComponentDiagnostic: {
      ...summarizeComponents(tallComponents),
      interpretation: 'PCA/chord measurements of the accepted G>=R colour partition only; the curve is conditional on one chord per component and never drives the hard verdict.',
    },
    semanticPlume: {
      ...summarizeNonOpaqueSegments(tallSemanticPlumeSegments),
      opaqueAxisSizing: 'excluded',
      analyticBasisFit: 'not performed; mode/stratum count and maximal partial-ray integral residual remain Gate-B blockers',
    },
    oldColourMajorityPlumeDiagnostic: {
      semantic: tallPlume.semantic,
      triangles: tallPlumeIndices.length / 3,
      surfaceAreaMetres2: selectionArea(tallPlume),
      boundsMetres: bounds(tallPositions, tallPlumeIndices),
      note: 'not used as the Class-E plume boundary: it contains 244,654 semantically crisp glume/lemma/anther/axis triangles',
    },
    optimisticAssumptions: [
      'Generator crisp/plume semantics are used for the candidate segmentation; disagreement with the older colour-majority transport partition is measured explicitly.',
      'Each authored centerline interval is a candidate chord, but further compiler segmentation remains legal.',
      'The full tolerance is assigned to axis error while curvature/shape/tip/endpoint residuals are optimistically zero.',
      'Axis-compatible components may share one field regardless of axial interval, support, owner, root, cap, or attribute-layout compatibility.',
    ],
  },
  denseSphagnum: {
    source: {
      label: 'accepted deterministic dense Sphagnum fixture',
      generator: sphagnum.generator,
      packedSha256: sphagnumPackedSha256,
      vertices: sphagnum.mesh.positions.length / 3,
      triangles: sphagnum.mesh.indices.length / 3,
      carpetTriangles: sphagnum.carpetTriangleCount,
      capitulaTriangles: sphagnumCapitulaIndices.length / 3,
      capitula: sphagnum.capitula.length,
      primitiveRecipes: sphagnum.primitiveRecipes.length,
    },
    carpet: {
      semantic: sphagnumCarpet.semantic,
      triangles: sphagnumCarpetIndices.length / 3,
      surfaceAreaMetres2: selectionArea(sphagnumCarpet),
      boundsMetres: bounds(sphagnumPositions, sphagnumCarpetIndices),
      hardOpaqueFieldLowerBound: 1,
      note: 'one field is only an optimistic counting lower bound; no proof here turns the connected height carpet into a compatible affine mask/interval field',
    },
    crispRecipeSegmentCandidate: {
      ...summarizeRecipeSegments(sphagnumCrispSegments),
      interpretation: 'generator-authored vertical stem/core intervals only; the connected carpet remains a separate unresolved support field',
    },
    mediumCandidates: {
      ...summarizeNonOpaqueSegments(sphagnumMediumSegments),
      opaqueAxisSizing: 'excluded by authoring sidecar',
      semantic: 'primary/fork branches are explicitly medium-candidate and rejected as near-horizontal curved Class-E axes',
      analyticBasisFit: 'not performed; exclusion from opaque K is conditional on passing the maximal partial-ray plume-medium gate',
    },
    edgeComponentDiagnostic: {
      ...summarizeComponents(sphagnumComponents),
      interpretation: 'conditional all-non-carpet-crisp counterfactual only; not used for the authored opaque-axis Pareto',
    },
    routing: {
      status: 'AUTHORED_BUT_NOT_CERTIFIED',
      reason: 'the fixture now explicitly routes near-horizontal curved branches to medium-candidate, but no finite analytic basis or partial-ray error certificate exists yet',
      hardCommunityFieldLowerBound: 1,
    },
    optimisticAssumptions: [
      'The connected carpet costs only one opaque field.',
      'Only generator-authored vertical stem/core recipes size opaque axes; branches are excluded conditionally on a future medium certificate.',
      'Generator recipes supply geometry semantics but not ownership-compatible affine field grouping.',
    ],
  },
  unresolvedCompilerSemantics: [
    'final compiler choice of authored versus further centreline segmentation, with every added cap/interval charged',
    'curvature, cross-section, tip, cap, and endpoint/root residual budgets',
    'frozen a_min; the report gives a parameter sweep instead',
    'axial interval, support, ownership, coincident-tie, and attribute-layout compatibility splits',
    'pointwise-versus-root community support and bounded interval oracle',
    'visibility-cell and tangent-safe classification',
    'Sphagnum crisp-versus-subpixel partition after D_min and theta_pix are frozen',
    'analytic plume basis, ordered coloured segments, and maximal partial-ray integral residual',
    'finite first-passage sampling, resident bytes, and correlated minification cost',
  ],
  decisionBoundary: {
    currentHardVerdict: 'RED under the proposed M x I field/read model at any plausible millimetric fidelity. The full accepted production community blade+culm endpoints alone require at least 14 fields / 29 reads at 10 mm, 9 / 19 at 20 mm, and 5 / 11 even at 50 mm. These catalogue-independent bounds omit the entire reproductive head.',
    isolatedTenMillimetreResult: 'For only 17 blades plus one culm, the catalogue-independent cap-plane bound is 7 fields / 15 reads. Under the emitted four-axis catalogue, exact remaining-error cap intervals require at least 83 distinct (axis,hMin,hMax) fields / 167 reads; the constructive grouping uses 122 / 245.',
    sensitivity: 'For that isolated mandatory subset, the fixed-catalogue interval lower bounds are 19 fields / 39 reads at 20 mm and 8 / 17 at 50 mm. Only at 100 mm does the lower bound reach 4 / 9, while the constructive grouping still needs 10 / 21; at 200 mm it still constructs 6 / 13. A slight threshold relaxation cannot rescue this representation.',
    productionAxisContext: 'Even before interval grouping, the accepted six-shoot community frozen at authored centreline intervals has a 10 mm pairwise-incompatible packing of 6 axes, or 13 optimistic reads.',
    mossBoundary: 'Sphagnum vertical stem/core axes fit one direction; its 1,228 near-horizontal branches are only medium candidates and cannot be credited until the analytic partial-ray error gate passes.',
    globalOptimumBoundary: 'The cap-plane result is a global lower bound for the current M x I representation and mandatory endpoint fidelity. The 83/19/8/4 interval counts are lower bounds for the explicitly emitted catalogues and assignments, not a theorem over every alternate joint catalogue. No claim is made against a new representation whose field can encode q-dependent finite endpoints at one fixed read.',
    requiredToResume: 'change the representation so finite per-primitive endpoints do not consume one common axial interval per field, relax the read ceiling, or explicitly accept botanical-scale 100 mm compilation errors; tuning the sampled first-passage texture cannot remove the structural cap-plane bound',
    neverProvesGreen: 'an axis census cannot establish field compatibility, compilation fidelity, plume fidelity, sampled reconstruction, minification, memory, or runtime feasibility',
  },
};

const metrics = `${JSON.stringify(report, null, 2)}\n`;
writeFileSync(resolve(outputRoot, 'metrics.json'), metrics);
const index = {
  schema: 'laas-class-e-structural-feasibility-index/v1',
  status: report.status,
  sourceSetSha256,
  recipeSha256,
  files: [{
    file: 'metrics.json',
    bytes: Buffer.byteLength(metrics),
    sha256: sha256(metrics),
    interpretation: 'Gate-B structural axis feasibility lower bound; not a compiler or reconstruction acceptance',
  }],
};
writeFileSync(resolve(outputRoot, 'index.json'), `${JSON.stringify(index, null, 2)}\n`);
console.error(`[class-e-structural] wrote ${outputRoot}`);
process.stdout.write(`${outputRoot}\n`);
