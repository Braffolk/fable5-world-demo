/**
 * Renderer-independent feasibility gate for the cap-free lateral-core plus
 * analytic-residual representation.
 *
 * This is intentionally an optimistic compiler bound.  A retained atom is
 * credited with its exact source triangles even though a production compiler
 * must still construct a source-contained common-cross-section mask.  The
 * residual fit is an independently fitted one-dimensional height marginal,
 * which is strictly easier than one shared three-dimensional spectral field.
 * Failure is therefore useful evidence; success is only permission to run the
 * later full exterior-ray visual gate.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  ESTONIAN_GRAMINOID_PROFILE_IDS,
  makeCalamagrostisCanescensBandlimitedSource,
  makeEstonianGraminoidFixture,
  type EstonianGraminoidFixture,
  type GraminoidPrimitiveRecipe,
} from '../../EstonianGraminoids';
import {
  makeSphagnumCapillifoliumFixture,
  type SphagnumCapillifoliumFixture,
} from '../../SphagnumCapillifolium';
import type { IndexedMesh, Vec3 as ObjectVec3 } from '../../ProfileFormat';

type Vec3 = readonly [number, number, number];
type Rgb = readonly [number, number, number];

interface Atom {
  id: number;
  primitiveId: number;
  segment: number;
  semantic: string;
  recognitionClass: 'foliage' | 'structure' | 'reproductive' | 'moss';
  axis: Vec3;
  capacityMetres: number;
  shapeErrorMetres: number;
  areaMetres2: number;
  longitudinalExtentMetres: number;
  midpoint: Vec3;
  color: Rgb;
  sourceTriangleStart: number;
  sourceTriangleCount: number;
  /** Alternative ruled coordinates for the same source patch. They are
   * elected categorically; source area is never counted more than once. */
  alternatives: readonly Sweep[];
}

interface Sweep {
  label: 'width' | 'centerline' | 'tube-axis';
  axis: Vec3;
  capacityMetres: number;
  shapeErrorMetres: number;
  hMin: number;
  hMax: number;
  slopeX: number;
  slopeZ: number;
  periodicCompatible: boolean;
}

interface Field {
  axis: Vec3;
  intervalMetres: number;
  hMin: number;
  hMax: number;
}

interface AtomAssignment {
  fraction: number;
  field: number;
  transverseErrorMetres: number;
}

interface HistogramFit {
  modes: number;
  selectedFrequencies: number[];
  normalizedDensityL1: number;
  partialMassSupError: number;
  partialRgbMaxChannelSupError: number;
  optimisticPremultipliedRgbP95: number;
}

function solveNormalEquations(basis: readonly Float64Array[], target: Float64Array): number[] {
  const count = basis.length;
  const matrix = Array.from({ length: count }, () => new Float64Array(count + 1));
  for (let row = 0; row < count; row++) {
    for (let column = 0; column < count; column++) {
      let value = 0;
      for (let sample = 0; sample < target.length; sample++) value += basis[row]![sample]! * basis[column]![sample]!;
      matrix[row]![column] = value;
    }
    let rhs = 0;
    for (let sample = 0; sample < target.length; sample++) rhs += basis[row]![sample]! * target[sample]!;
    matrix[row]![count] = rhs;
  }
  for (let pivot = 0; pivot < count; pivot++) {
    let winner = pivot;
    for (let row = pivot + 1; row < count; row++) if (Math.abs(matrix[row]![pivot]!) > Math.abs(matrix[winner]![pivot]!)) winner = row;
    [matrix[pivot], matrix[winner]] = [matrix[winner]!, matrix[pivot]!];
    const diagonal = matrix[pivot]![pivot]!;
    if (Math.abs(diagonal) < 1e-18) continue;
    for (let column = pivot; column <= count; column++) matrix[pivot]![column] /= diagonal;
    for (let row = 0; row < count; row++) {
      if (row === pivot) continue;
      const factor = matrix[row]![pivot]!;
      for (let column = pivot; column <= count; column++) matrix[row]![column] -= factor * matrix[pivot]![column]!;
    }
  }
  return matrix.map((row, index) => Math.abs(row[index]!) < 1e-18 ? 0 : row[count]!);
}

const REPO_ROOT = resolve(import.meta.dirname, '../../../..');
const OUTPUT_ROOT = resolve(REPO_ROOT, 'data/work/groundcover-lateral-core-residual-gate');
const EXPECTED_PRODUCTION_CAL_MESH = '37b0cf1d33f632b5cde3dd60410ad987eec2e977661d0bc804dd6f2bee7054c0';
const EXPECTED_SPHAGNUM_MESH = 'ec8612e197dcc9a801c6146059cf8f85b7549ef33ac189252e91c8efbf01caa7';
const TOLERANCES = [0.0025, 0.005, 0.01] as const;
const FIELD_COUNTS = [2, 3, 4, 8] as const;
const TWO_LAYER_MAX_FIELDS = 4;
const ONE_LAYER_MAX_FIELDS = 8;
const AXIS_SAMPLES = 36;
const INTERVAL_SAMPLES = 14;
const HEIGHT_BINS = 256;
const PIXEL_ANGLE_RADIANS = (55 * Math.PI / 180) / 2160;
const DISTANCES_METRES = [1, 2, 4, 8, 16, 32] as const;
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

function objectVec(value: ObjectVec3): Vec3 { return [value.x, value.y, value.z]; }
function add(a: Vec3, b: Vec3): Vec3 { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function sub(a: Vec3, b: Vec3): Vec3 { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function scale(a: Vec3, s: number): Vec3 { return [a[0] * s, a[1] * s, a[2] * s]; }
function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function length(a: Vec3): number { return Math.hypot(a[0], a[1], a[2]); }
function normalize(a: Vec3): Vec3 {
  const magnitude = length(a);
  return magnitude > EPSILON ? scale(a, 1 / magnitude) : [0, 1, 0];
}
function canonicalAxis(input: Vec3): Vec3 {
  let axis = normalize(input);
  if (axis[1] < 0 || (Math.abs(axis[1]) < EPSILON && (axis[0] < 0 || (Math.abs(axis[0]) < EPSILON && axis[2] < 0)))) {
    axis = scale(axis, -1);
  }
  return axis;
}
function midpoint(a: Vec3, b: Vec3): Vec3 { return scale(add(a, b), 0.5); }
function clamp01(value: number): number { return Math.max(0, Math.min(1, value)); }

function vertex(mesh: IndexedMesh, index: number): Vec3 {
  const offset = index * 3;
  return [mesh.positions[offset]!, mesh.positions[offset + 1]!, mesh.positions[offset + 2]!];
}

function vertexColor(mesh: IndexedMesh, index: number): Rgb {
  const offset = index * 3;
  return mesh.colors
    ? [mesh.colors[offset]!, mesh.colors[offset + 1]!, mesh.colors[offset + 2]!]
    : [0.32, 0.48, 0.18];
}

function triangleArea(mesh: IndexedMesh, triangle: number): number {
  const offset = triangle * 3;
  const a = vertex(mesh, mesh.indices[offset]!);
  const b = vertex(mesh, mesh.indices[offset + 1]!);
  const c = vertex(mesh, mesh.indices[offset + 2]!);
  return length(cross(sub(b, a), sub(c, a))) * 0.5;
}

function triangleColor(mesh: IndexedMesh, triangle: number): Rgb {
  const offset = triangle * 3;
  const a = vertexColor(mesh, mesh.indices[offset]!);
  const b = vertexColor(mesh, mesh.indices[offset + 1]!);
  const c = vertexColor(mesh, mesh.indices[offset + 2]!);
  return [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3];
}

function triangleRangeStats(mesh: IndexedMesh, start: number, count: number): { area: number; color: Rgb } {
  let area = 0;
  const color = [0, 0, 0];
  for (let triangle = start; triangle < start + count; triangle++) {
    const weight = triangleArea(mesh, triangle);
    const rgb = triangleColor(mesh, triangle);
    area += weight;
    color[0] += rgb[0] * weight;
    color[1] += rgb[1] * weight;
    color[2] += rgb[2] * weight;
  }
  return { area, color: area > 0 ? [color[0] / area, color[1] / area, color[2] / area] : [0, 0, 0] };
}

function meshHash(mesh: IndexedMesh): string { return sha256(JSON.stringify(mesh)); }

function meshBinaryHash(mesh: IndexedMesh): string {
  const digest = createHash('sha256');
  for (const values of [
    Float64Array.from(mesh.positions),
    Float64Array.from(mesh.normals),
    Uint32Array.from(mesh.indices),
  ]) digest.update(new Uint8Array(values.buffer));
  return digest.digest('hex');
}

function fibonacciAxes(count: number): Vec3[] {
  const axes: Vec3[] = [];
  const phi = (1 + Math.sqrt(5)) * 0.5;
  for (let index = 0; index < count; index++) {
    const y = 1 - 2 * ((index + 0.5) / count);
    const radius = Math.sqrt(Math.max(0, 1 - y * y));
    const angle = 2 * Math.PI * index / phi;
    axes.push(canonicalAxis([radius * Math.cos(angle), y, radius * Math.sin(angle)]));
  }
  axes.push([0, 1, 0], [1, 0, 0], [0, 0, 1], normalize([1, 0, 1]), normalize([1, 0, -1]));
  return axes;
}

function geometricGrid(minimum: number, maximum: number, count: number): number[] {
  const low = Math.log(Math.max(1e-6, minimum));
  const high = Math.log(Math.max(minimum * 1.0001, maximum));
  return Array.from({ length: count }, (_, index) => Math.exp(low + (high - low) * index / (count - 1)));
}

function recipeClass(recipe: GraminoidPrimitiveRecipe): Atom['recognitionClass'] {
  if (recipe.kind === 'blade' || (recipe.kind === 'tube' && (recipe.role === 'culm' || recipe.role === 'basal-sheath' || recipe.role === 'upper-sheath'))) return 'foliage';
  if (recipe.kind === 'lanceolate-surface' || (recipe.kind === 'tube' && ['panicle-axis', 'spikelet-axis', 'anther-filament'].includes(recipe.role))) return 'reproductive';
  return 'structure';
}

function appendAtom(
  atoms: Atom[], mesh: IndexedMesh, primitiveId: number, segment: number, semantic: string,
  recognitionClass: Atom['recognitionClass'], axis: Vec3, capacityMetres: number,
  shapeErrorMetres: number, longitudinalExtentMetres: number, mid: Vec3,
  sourceTriangleStart: number, sourceTriangleCount: number,
  alternatives?: readonly Sweep[],
): void {
  if (capacityMetres <= EPSILON || sourceTriangleCount <= 0) return;
  const stats = triangleRangeStats(mesh, sourceTriangleStart, sourceTriangleCount);
  if (stats.area <= EPSILON) return;
  atoms.push({
    id: atoms.length, primitiveId, segment, semantic, recognitionClass,
    axis: canonicalAxis(axis), capacityMetres, shapeErrorMetres,
    areaMetres2: stats.area, longitudinalExtentMetres, midpoint: mid, color: stats.color,
    sourceTriangleStart, sourceTriangleCount,
    alternatives: alternatives ?? [],
  });
}

function verticalChartSweep(
  label: Sweep['label'], start: Vec3, end: Vec3, shapeErrorMetres: number,
): Sweep {
  const delta = sub(end, start);
  const dy = delta[1];
  const periodicCompatible = Math.abs(dy) > 1e-8;
  return {
    label,
    axis: canonicalAxis(delta),
    capacityMetres: Math.abs(dy),
    shapeErrorMetres,
    hMin: Math.min(start[1], end[1]),
    hMax: Math.max(start[1], end[1]),
    slopeX: periodicCompatible ? delta[0] / dy : Infinity,
    slopeZ: periodicCompatible ? delta[2] / dy : Infinity,
    periodicCompatible,
  };
}

function ribbonAtoms(
  atoms: Atom[], mesh: IndexedMesh, recipe: Extract<GraminoidPrimitiveRecipe, { kind: 'blade' | 'lanceolate-surface' }>,
  allowCenterlineSweep: boolean,
): void {
  const bodySegments = recipe.kind === 'blade' ? recipe.centerline.length - 2 : recipe.centerline.length - 1;
  const centerOffset = recipe.kind === 'blade' ? 1 : 0;
  for (let segment = 0; segment < bodySegments; segment++) {
    const c0 = objectVec(recipe.centerline[segment + centerOffset]!);
    const c1 = objectVec(recipe.centerline[segment + centerOffset + 1]!);
    const longitudinal = length(sub(c1, c0));
    const w0 = recipe.halfWidths[segment + centerOffset]!;
    const w1 = recipe.halfWidths[segment + centerOffset + 1]!;
    const capacity = 2 * Math.min(w0, w1);
    const triangleStart = recipe.sourceTriangleStart + 1 + segment * 2;
    const aLeft = vertex(mesh, mesh.indices[triangleStart * 3]!);
    const aRight = vertex(mesh, mesh.indices[triangleStart * 3 + 2]!);
    const bLeft = vertex(mesh, mesh.indices[(triangleStart + 1) * 3 + 1]!);
    const bRight = vertex(mesh, mesh.indices[(triangleStart + 1) * 3 + 2]!);
    const widthAxisA = canonicalAxis(sub(aLeft, aRight));
    const widthAxisB = canonicalAxis(sub(bLeft, bRight));
    const widthFrameDrift = 0.5 * capacity * Math.min(length(sub(widthAxisA, widthAxisB)), length(add(widthAxisA, widthAxisB)));
    const alternatives: Sweep[] = [{
      label: 'width', axis: widthAxisA, capacityMetres: capacity,
      shapeErrorMetres: widthFrameDrift,
      hMin: Math.min(aLeft[1], aRight[1]), hMax: Math.max(aLeft[1], aRight[1]),
      slopeX: Infinity, slopeZ: Infinity, periodicCompatible: false,
    }];
    if (allowCenterlineSweep) alternatives.push(verticalChartSweep('centerline', c0, c1, widthFrameDrift));
    appendAtom(
      atoms, mesh, recipe.primitiveId, segment,
      recipe.kind === 'blade' ? 'blade-face' : `lanceolate:${recipe.role}`,
      recipeClass(recipe), widthAxisA, capacity,
      widthFrameDrift, longitudinal, midpoint(c0, c1), triangleStart, 2, alternatives,
    );
  }
}

function tubeAtoms(
  atoms: Atom[], mesh: IndexedMesh, recipe: Extract<GraminoidPrimitiveRecipe, { kind: 'tube' | 'rhizome' }>,
): void {
  const bodyStart = recipe.sourceTriangleStart + (recipe.anchorVertex !== null ? recipe.sides : 0);
  for (let segment = 0; segment < recipe.centerline.length - 1; segment++) {
    const c0 = objectVec(recipe.centerline[segment]!);
    const c1 = objectVec(recipe.centerline[segment + 1]!);
    const axis = sub(c1, c0);
    const capacity = length(axis);
    appendAtom(
      atoms, mesh, recipe.primitiveId, segment, `${recipe.kind}:${recipe.role}`,
      recipeClass(recipe), axis, capacity,
      Math.abs(recipe.radii[segment]! - recipe.radii[segment + 1]!) * 0.5,
      Math.PI * (recipe.radii[segment]! + recipe.radii[segment + 1]!), midpoint(c0, c1),
      bodyStart + segment * recipe.sides * 2, recipe.sides * 2,
      [verticalChartSweep(
        'tube-axis', c0, c1,
        Math.abs(recipe.radii[segment]! - recipe.radii[segment + 1]!) * 0.5,
      )],
    );
  }
}

function calamagrostisAtoms(fixture: EstonianGraminoidFixture, allowCenterlineSweep = true): Atom[] {
  const atoms: Atom[] = [];
  for (const recipe of fixture.primitiveRecipes) {
    if (recipe.disposition !== 'crisp') continue;
    if (recipe.kind === 'blade' || recipe.kind === 'lanceolate-surface') ribbonAtoms(atoms, fixture.mesh, recipe, allowCenterlineSweep);
    else if (recipe.kind === 'tube' || recipe.kind === 'rhizome') tubeAtoms(atoms, fixture.mesh, recipe);
  }
  return atoms;
}

function sphagnumAtoms(fixture: SphagnumCapillifoliumFixture): Atom[] {
  const atoms: Atom[] = [];
  for (const recipe of fixture.primitiveRecipes) {
    if (!('centerline' in recipe)) continue;
    const segments = recipe.centerline.length - 1;
    const sides = recipe.kind === 'stem' ? 6 : recipe.kind === 'core' ? 10 : 4;
    for (let segment = 0; segment < segments; segment++) {
      const triangleCount = Math.min(
        sides * 2,
        recipe.sourceTriangleStart + recipe.sourceTriangleCount
          - (recipe.sourceTriangleStart + segment * sides * 2),
      );
      // A short final fan is an axial cap/tip and is residual by contract.
      if (triangleCount < sides * 2) continue;
      const c0 = objectVec(recipe.centerline[segment]!);
      const c1 = objectVec(recipe.centerline[segment + 1]!);
      const axis = sub(c1, c0);
      const capacity = length(axis);
      let shapeError = 0;
      let circumference = 0;
      if ('radii' in recipe) {
        shapeError = Math.abs(recipe.radii[segment]! - recipe.radii[segment + 1]!) * 0.5;
        circumference = Math.PI * (recipe.radii[segment]! + recipe.radii[segment + 1]!);
      } else {
        shapeError = Math.max(
          Math.abs(recipe.halfWidths[segment]! - recipe.halfWidths[segment + 1]!),
          Math.abs(recipe.halfThicknesses[segment]! - recipe.halfThicknesses[segment + 1]!),
        ) * 0.5;
        circumference = 2 * (recipe.halfWidths[segment]! + recipe.halfWidths[segment + 1]!
          + recipe.halfThicknesses[segment]! + recipe.halfThicknesses[segment + 1]!);
      }
      appendAtom(
        atoms, fixture.mesh, recipe.primitiveId, segment, `sphagnum:${recipe.kind}`, 'moss',
        axis, capacity, shapeError, circumference, midpoint(c0, c1),
        recipe.sourceTriangleStart + segment * sides * 2,
        triangleCount,
        [verticalChartSweep('tube-axis', c0, c1, shapeError)],
      );
    }
  }
  return atoms;
}

function sweepFieldError(sweep: Sweep, field: Field): number {
  if (!sweep.periodicCompatible || field.intervalMetres <= EPSILON) return Infinity;
  if (field.hMin < sweep.hMin - EPSILON || field.hMax > sweep.hMax + EPSILON) return Infinity;
  const fieldSlopeX = field.axis[0] / field.axis[1];
  const fieldSlopeZ = field.axis[2] / field.axis[1];
  const fraction = clamp01(field.intervalMetres / Math.max(EPSILON, sweep.hMax - sweep.hMin));
  return sweep.shapeErrorMetres * fraction + 0.5 * field.intervalMetres * Math.hypot(
    sweep.slopeX - fieldSlopeX,
    sweep.slopeZ - fieldSlopeZ,
  );
}

function mergeIntervals(intervals: ReadonlyArray<readonly [number, number]>): Array<[number, number]> {
  const ordered = [...intervals].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const merged: Array<[number, number]> = [];
  for (const interval of ordered) {
    const last = merged.at(-1);
    if (!last || interval[0] > last[1] + EPSILON) merged.push([interval[0], interval[1]]);
    else last[1] = Math.max(last[1], interval[1]);
  }
  return merged;
}

function intersectionLength(interval: readonly [number, number], union: ReadonlyArray<readonly [number, number]>): number {
  let result = 0;
  for (const member of union) result += Math.max(0, Math.min(interval[1], member[1]) - Math.max(interval[0], member[0]));
  return result;
}

function scoreAtomFields(atom: Atom, fields: readonly Field[], tolerance: number): AtomAssignment {
  let best: AtomAssignment = { fraction: 0, field: -1, transverseErrorMetres: Infinity };
  for (const sweep of atom.alternatives) {
    if (!sweep.periodicCompatible) continue;
    const accepted: Array<readonly [number, number]> = [];
    let maximumError = 0;
    fields.forEach((field) => {
      const error = sweepFieldError(sweep, field);
      if (error <= tolerance) {
        accepted.push([field.hMin, field.hMax]);
        maximumError = Math.max(maximumError, error);
      }
    });
    const union = mergeIntervals(accepted);
    const covered = union.reduce((sum, interval) => sum + interval[1] - interval[0], 0);
    const fraction = clamp01(covered / Math.max(EPSILON, sweep.hMax - sweep.hMin));
    // Alternative factorizations describe the same source patch and are
    // categorical. Distinct global bands inside one factorization are a union.
    if (fraction > best.fraction) best = { fraction, field: -1, transverseErrorMetres: maximumError };
  }
  return best;
}

function scoreFields(atoms: readonly Atom[], fields: readonly Field[], tolerance: number): { score: number; assignments: AtomAssignment[] } {
  const assignments: AtomAssignment[] = [];
  let score = 0;
  for (const atom of atoms) {
    const best = scoreAtomFields(atom, fields, tolerance);
    assignments.push(best);
    score += atom.areaMetres2 * best.fraction;
  }
  return { score, assignments };
}

interface PiecewiseEvent {
  startValue: number;
  startSlope: number;
  slopeDelta: number;
  endValue: number;
  endSlope: number;
}

function eventAt(events: Map<number, PiecewiseEvent>, position: number): PiecewiseEvent {
  const current = events.get(position);
  if (current) return current;
  const created = { startValue: 0, startSlope: 0, slopeDelta: 0, endValue: 0, endSlope: 0 };
  events.set(position, created);
  return created;
}

/** Exact marginal placement over the deterministic search atoms. For the
 * current sources every atom has exactly one periodic-compatible chart; this
 * is asserted so a future alternative chart cannot silently invalidate the
 * piecewise-union sweep. */
function bestFieldPlacement(
  atoms: readonly Atom[], existing: readonly Field[], axis: Vec3,
  intervalMetres: number, tolerance: number,
): { hMin: number; gain: number } | null {
  const events = new Map<number, PiecewiseEvent>();
  const probeAtZero: Field = { axis, intervalMetres, hMin: 0, hMax: intervalMetres };
  for (const atom of atoms) {
    const sweeps = atom.alternatives.filter((sweep) => sweep.periodicCompatible);
    if (sweeps.length !== 1) throw new Error(`atom ${atom.id} has ${sweeps.length} periodic charts; piecewise gate requires exactly one`);
    const sweep = sweeps[0]!;
    const span = sweep.hMax - sweep.hMin;
    if (intervalMetres > span + EPSILON) continue;
    // Direction/shape error is independent of the global band's y placement.
    const directionError = sweep.shapeErrorMetres * (intervalMetres / span)
      + 0.5 * intervalMetres * Math.hypot(
        sweep.slopeX - probeAtZero.axis[0] / probeAtZero.axis[1],
        sweep.slopeZ - probeAtZero.axis[2] / probeAtZero.axis[1],
      );
    if (directionError > tolerance) continue;
    const currentUnion = mergeIntervals(existing.flatMap((field) =>
      sweepFieldError(sweep, field) <= tolerance ? [[field.hMin, field.hMax] as const] : []));
    const firstStart = sweep.hMin;
    const lastStart = sweep.hMax - intervalMetres;
    const breakpoints = [firstStart, lastStart];
    for (const interval of currentUnion) {
      for (const point of [interval[0], interval[1], interval[0] - intervalMetres, interval[1] - intervalMetres]) {
        if (point > firstStart + EPSILON && point < lastStart - EPSILON) breakpoints.push(point);
      }
    }
    const points = [...new Set(breakpoints)].sort((a, b) => a - b);
    const values = points.map((start) => atom.areaMetres2 * (
      intervalMetres - intersectionLength([start, start + intervalMetres], currentUnion)
    ) / span);
    if (points.length === 1) {
      const event = eventAt(events, points[0]!);
      event.startValue += values[0]!;
      event.endValue += values[0]!;
      continue;
    }
    const slopes = points.slice(0, -1).map((point, index) =>
      (values[index + 1]! - values[index]!) / (points[index + 1]! - point));
    const startEvent = eventAt(events, points[0]!);
    startEvent.startValue += values[0]!;
    startEvent.startSlope += slopes[0]!;
    for (let index = 1; index < points.length - 1; index++) {
      eventAt(events, points[index]!).slopeDelta += slopes[index]! - slopes[index - 1]!;
    }
    const endEvent = eventAt(events, points.at(-1)!);
    endEvent.endValue += values.at(-1)!;
    endEvent.endSlope += slopes.at(-1)!;
  }
  if (events.size === 0) return null;
  let value = 0; let slope = 0; let previous = [...events.keys()].sort((a, b) => a - b)[0]!;
  let bestGain = -Infinity; let bestStart = previous;
  for (const position of [...events.keys()].sort((a, b) => a - b)) {
    value += slope * (position - previous);
    const event = events.get(position)!;
    value += event.startValue;
    slope += event.startSlope + event.slopeDelta;
    if (value > bestGain) { bestGain = value; bestStart = position; }
    value -= event.endValue;
    slope -= event.endSlope;
    previous = position;
  }
  return { hMin: bestStart, gain: Math.max(0, bestGain) };
}

function selectFields(atoms: readonly Atom[], tolerance: number, maximumFields: number): { fields: Field[]; scores: number[] } {
  const admissible = atoms.filter((atom) => atom.alternatives.some((sweep) => sweep.periodicCompatible));
  if (admissible.length === 0) return { fields: [], scores: [] };
  const baseAxes = fibonacciAxes(AXIS_SAMPLES).filter((axis) => Math.abs(axis[1]) >= 0.18);
  const classAxes = new Map<string, Array<{ area: number; axis: Vec3 }>>();
  for (const atom of admissible) for (const sweep of atom.alternatives) {
    if (!sweep.periodicCompatible) continue;
    const list = classAxes.get(atom.recognitionClass) ?? [];
    list.push({ area: atom.areaMetres2, axis: sweep.axis });
    classAxes.set(atom.recognitionClass, list);
  }
  const axes = [...baseAxes];
  for (const list of classAxes.values()) {
    list.sort((a, b) => b.area - a.area);
    for (const item of list.slice(0, 12)) axes.push(item.axis);
  }
  const uniqueAxes = [...new Map(axes.map((axis) => {
    const slopeX = axis[0] / axis[1]; const slopeZ = axis[2] / axis[1];
    return [`${slopeX.toFixed(4)}/${slopeZ.toFixed(4)}`, axis] as const;
  })).values()];
  const spans = admissible.flatMap((atom) => atom.alternatives
    .filter((sweep) => sweep.periodicCompatible)
    .map((sweep) => sweep.hMax - sweep.hMin)
    .filter((span) => span > 1e-5));
  let minimumSpan = Infinity; let maximumSpan = 0; let globalMinH = Infinity; let globalMaxH = -Infinity;
  for (const span of spans) { minimumSpan = Math.min(minimumSpan, span); maximumSpan = Math.max(maximumSpan, span); }
  for (const atom of admissible) for (const sweep of atom.alternatives) if (sweep.periodicCompatible) {
    globalMinH = Math.min(globalMinH, sweep.hMin); globalMaxH = Math.max(globalMaxH, sweep.hMax);
  }
  const lengths = geometricGrid(minimumSpan, maximumSpan, INTERVAL_SAMPLES);
  // Class-stratified deterministic search subset. Exact scores and final
  // assignments below always use every atom.
  const searchAtoms: Atom[] = [];
  for (const recognitionClass of ['foliage', 'structure', 'reproductive', 'moss'] as const) {
    const group = admissible.filter((atom) => atom.recognitionClass === recognitionClass);
    const stride = Math.max(1, Math.ceil(group.length / 750));
    for (let index = 0; index < group.length; index += stride) searchAtoms.push(group[index]!);
  }
  const fields: Field[] = [];
  const scores: number[] = [];
  let exactCurrentScore = 0;
  for (let iteration = 0; iteration < maximumFields; iteration++) {
    const ranked: Array<{ field: Field; score: number }> = [];
    for (const axis of uniqueAxes) for (const intervalMetres of lengths) {
      if (intervalMetres > globalMaxH - globalMinH) continue;
      const placement = bestFieldPlacement(searchAtoms, fields, axis, intervalMetres, tolerance);
      if (!placement || placement.gain <= EPSILON) continue;
      const hMin = placement.hMin;
      ranked.push({
        field: { axis, intervalMetres, hMin, hMax: hMin + intervalMetres },
        score: placement.gain,
      });
    }
    ranked.sort((a, b) => b.score - a.score || a.field.hMin - b.field.hMin);
    let winner: Field | null = null;
    let winnerScore = exactCurrentScore;
    for (const probe of ranked.slice(0, 24)) {
      const field = probe.field;
      const score = scoreFields(atoms, [...fields, field], tolerance).score;
      if (score > winnerScore + 1e-15) { winner = field; winnerScore = score; }
    }
    if (!winner) break;
    fields.push(winner);
    exactCurrentScore = winnerScore;
    scores.push(winnerScore);
  }
  return { fields, scores };
}

function quantile(valuesInput: readonly number[], p: number): number {
  if (valuesInput.length === 0) return 0;
  const values = [...valuesInput].sort((a, b) => a - b);
  const index = (values.length - 1) * p;
  const low = Math.floor(index); const high = Math.ceil(index); const t = index - low;
  return values[low]! * (1 - t) + values[high]! * t;
}

function fitHeightMarginal(
  atoms: readonly Atom[], assignments: readonly AtomAssignment[], maximumModes: number,
  minY: number, maxY: number,
): HistogramFit[] {
  const truth = Array.from({ length: 4 }, () => new Float64Array(HEIGHT_BINS));
  atoms.forEach((atom, index) => {
    const omitted = atom.areaMetres2 * (1 - assignments[index]!.fraction);
    if (omitted <= EPSILON) return;
    const bin = Math.max(0, Math.min(HEIGHT_BINS - 1, Math.floor((atom.midpoint[1] - minY) / Math.max(EPSILON, maxY - minY) * HEIGHT_BINS)));
    truth[0]![bin] += omitted;
    for (let channel = 0; channel < 3; channel++) truth[channel + 1]![bin] += omitted * atom.color[channel]!;
  });
  const window = new Float64Array(HEIGHT_BINS);
  for (let bin = 0; bin < HEIGHT_BINS; bin++) {
    const u = (bin + 0.5) / HEIGHT_BINS;
    window[bin] = 4 * u * (1 - u);
  }
  const selected: number[] = [];
  const totalMass = truth[0]!.reduce((sum, value) => sum + value, 0);
  const fits: HistogramFit[] = [];
  for (let modes = 0; modes <= maximumModes; modes++) {
    if (modes > 0) {
      const basisPrior = [window, ...selected.flatMap((frequency) => {
        const cosine = new Float64Array(HEIGHT_BINS); const sine = new Float64Array(HEIGHT_BINS);
        for (let bin = 0; bin < HEIGHT_BINS; bin++) {
          const angle = 2 * Math.PI * frequency * (bin + 0.5) / HEIGHT_BINS;
          cosine[bin] = 2 * window[bin]! * Math.cos(angle);
          sine[bin] = 2 * window[bin]! * Math.sin(angle);
        }
        return [cosine, sine];
      })];
      const residuals = truth.map((target) => {
        const coefficients = solveNormalEquations(basisPrior, target);
        return Float64Array.from(target, (value, bin) => value - basisPrior.reduce((sum, basis, index) => sum + basis[bin]! * coefficients[index]!, 0));
      });
      let winner = 1; let winnerEnergy = -Infinity;
      for (let frequency = 1; frequency < HEIGHT_BINS / 2; frequency++) {
        if (selected.includes(frequency)) continue;
        let energy = 0;
        for (const residual of residuals) {
          let cosine = 0; let sine = 0;
          for (let bin = 0; bin < HEIGHT_BINS; bin++) {
            const angle = 2 * Math.PI * frequency * (bin + 0.5) / HEIGHT_BINS;
            cosine += residual[bin]! * window[bin]! * Math.cos(angle);
            sine += residual[bin]! * window[bin]! * Math.sin(angle);
          }
          energy += cosine * cosine + sine * sine;
        }
        if (energy > winnerEnergy) { winner = frequency; winnerEnergy = energy; }
      }
      selected.push(winner);
    }
    const basis = [window, ...selected.flatMap((frequency) => {
      const cosine = new Float64Array(HEIGHT_BINS); const sine = new Float64Array(HEIGHT_BINS);
      for (let bin = 0; bin < HEIGHT_BINS; bin++) {
        const angle = 2 * Math.PI * frequency * (bin + 0.5) / HEIGHT_BINS;
        cosine[bin] = 2 * window[bin]! * Math.cos(angle);
        sine[bin] = 2 * window[bin]! * Math.sin(angle);
      }
      return [cosine, sine];
    })];
    const predicted = Array.from({ length: 4 }, () => new Float64Array(HEIGHT_BINS));
    for (let channel = 0; channel < 4; channel++) {
      const coefficients = solveNormalEquations(basis, truth[channel]!);
      for (let bin = 0; bin < HEIGHT_BINS; bin++) {
        const value = basis.reduce((sum, item, index) => sum + item[bin]! * coefficients[index]!, 0);
        // Optimistic pointwise projection: production must certify these
        // inequalities from coefficients, without a live clamp.
        predicted[channel]![bin] = Math.max(0, value);
      }
    }
    const predictedMass = predicted[0]!.reduce((sum, value) => sum + value, 0);
    const massScale = predictedMass > 0 ? totalMass / predictedMass : 0;
    for (const channel of predicted) for (let bin = 0; bin < HEIGHT_BINS; bin++) channel[bin] *= massScale;
    for (let bin = 0; bin < HEIGHT_BINS; bin++) for (let channel = 1; channel < 4; channel++) {
      predicted[channel]![bin] = Math.min(predicted[channel]![bin]!, predicted[0]![bin]!);
    }
    let truthPrefix = 0; let predictedPrefix = 0; let partialMass = 0; let partialRgb = 0; let l1 = 0;
    const rgbErrors: number[] = [];
    const rgbTotals = truth.slice(1).map((channel) => channel.reduce((sum, value) => sum + value, 0));
    for (let bin = 0; bin < HEIGHT_BINS; bin++) {
      truthPrefix += truth[0]![bin]!; predictedPrefix += predicted[0]![bin]!;
      partialMass = Math.max(partialMass, Math.abs(truthPrefix - predictedPrefix) / Math.max(EPSILON, totalMass));
      l1 += Math.abs(truth[0]![bin]! - predicted[0]![bin]!);
      let binError = 0;
      for (let channel = 1; channel < 4; channel++) {
        let truthChannelPrefix = 0; let predictedChannelPrefix = 0;
        for (let cursor = 0; cursor <= bin; cursor++) {
          truthChannelPrefix += truth[channel]![cursor]!;
          predictedChannelPrefix += predicted[channel]![cursor]!;
        }
        const error = Math.abs(truthChannelPrefix - predictedChannelPrefix) / Math.max(EPSILON, rgbTotals[channel - 1]!);
        partialRgb = Math.max(partialRgb, error);
        binError = Math.max(binError, Math.abs(truth[channel]![bin]! - predicted[channel]![bin]!) / Math.max(EPSILON, truth[channel]![bin]!, totalMass / HEIGHT_BINS));
      }
      rgbErrors.push(binError);
    }
    fits.push({
      modes, selectedFrequencies: [...selected],
      normalizedDensityL1: l1 / Math.max(EPSILON, 2 * totalMass),
      partialMassSupError: partialMass,
      partialRgbMaxChannelSupError: partialRgb,
      optimisticPremultipliedRgbP95: quantile(rgbErrors, 0.95),
    });
  }
  return fits;
}

function reportCandidate(
  atoms: readonly Atom[], fields: readonly Field[], fieldCount: number, tolerance: number,
  bounds: { minY: number; maxY: number },
): Record<string, unknown> {
  const activeFields = fields.slice(0, fieldCount);
  const scored = scoreFields(atoms, activeFields, tolerance);
  const totalArea = atoms.reduce((sum, atom) => sum + atom.areaMetres2, 0);
  const retained = scored.score;
  const byClass: Record<string, { total: number; retained: number }> = {};
  atoms.forEach((atom, index) => {
    const target = byClass[atom.recognitionClass] ??= { total: 0, retained: 0 };
    target.total += atom.areaMetres2;
    target.retained += atom.areaMetres2 * scored.assignments[index]!.fraction;
  });
  const primitiveFractions = new Map<number, { total: number; retained: number }>();
  atoms.forEach((atom, index) => {
    const target = primitiveFractions.get(atom.primitiveId) ?? { total: 0, retained: 0 };
    target.total += atom.areaMetres2;
    target.retained += atom.areaMetres2 * scored.assignments[index]!.fraction;
    primitiveFractions.set(atom.primitiveId, target);
  });
  const completelyOmitted = [...primitiveFractions.values()].filter((item) => item.retained <= EPSILON).length;
  const partiallyOmitted = [...primitiveFractions.values()].filter((item) => item.retained > EPSILON && item.retained < item.total - EPSILON).length;
  const distanceAware = DISTANCES_METRES.map((distance) => {
    const pixel = distance * PIXEL_ANGLE_RADIANS;
    let visibleOmittedArea = 0; let connectedVisibleAtoms = 0; let maximumSpanPixels = 0; let maximumWidthPixels = 0;
    atoms.forEach((atom, index) => {
      const omittedArea = atom.areaMetres2 * (1 - scored.assignments[index]!.fraction);
      if (omittedArea <= EPSILON) return;
      const span = Math.max(atom.longitudinalExtentMetres, atom.capacityMetres);
      const areaPixels = omittedArea / (pixel * pixel);
      const spanPixels = span / pixel;
      const widthPixels = omittedArea / Math.max(EPSILON, span) / pixel;
      maximumSpanPixels = Math.max(maximumSpanPixels, spanPixels);
      maximumWidthPixels = Math.max(maximumWidthPixels, widthPixels);
      if (areaPixels >= 0.25 && spanPixels >= 1 && widthPixels >= 0.25) {
        visibleOmittedArea += omittedArea;
        connectedVisibleAtoms++;
      }
    });
    return {
      distanceMetres: distance,
      worldPixelFootprintMetres: pixel,
      conservativeFaceOnVisibleOmittedAreaFraction: visibleOmittedArea / Math.max(EPSILON, totalArea),
      connectedVisibleAtomCount: connectedVisibleAtoms,
      maximumOmissionSpanPixels: maximumSpanPixels,
      maximumProjectedFeatureWidthPixels: maximumWidthPixels,
      filterRule: 'conservative face-on pre-gate: projected area >=0.25 pixel^2, span >=1 pixel, and feature width >=0.25 pixel; not an exact anisotropic pixel reconstruction/filter',
    };
  });
  return {
    fieldCount,
    runtime: fieldCount <= TWO_LAYER_MAX_FIELDS
      ? { layers: 2, reads: fieldCount * 2 + 1 }
      : { layers: 1, reads: fieldCount + 1, explicitTrade: 'loses second quasiperiodic geometric population' },
    fields: activeFields,
    retainedSourceLateralAreaFraction: retained / Math.max(EPSILON, totalArea),
    omittedSourceLateralAreaFraction: 1 - retained / Math.max(EPSILON, totalArea),
    byRecognitionClass: Object.fromEntries(Object.entries(byClass).map(([key, value]) => [key, {
      totalAreaMetres2: value.total,
      retainedAreaFraction: value.retained / Math.max(EPSILON, value.total),
    }])),
    connectedOmissions: {
      sourcePrimitiveCount: primitiveFractions.size,
      completelyOmittedPrimitives: completelyOmitted,
      partiallyOmittedPrimitives: partiallyOmitted,
      note: 'primitive connectivity is exact; visible-atom count is a conservative disconnected-patch upper count, not image connected components',
    },
    distanceAware,
    optimisticResidualHeightMarginal: fitHeightMarginal(atoms, scored.assignments, 4, bounds.minY, bounds.maxY),
  };
}

function boundsY(mesh: IndexedMesh): { minY: number; maxY: number } {
  let minY = Infinity; let maxY = -Infinity;
  for (let index = 1; index < mesh.positions.length; index += 3) {
    minY = Math.min(minY, mesh.positions[index]!); maxY = Math.max(maxY, mesh.positions[index]!);
  }
  return { minY, maxY };
}

function recognitionBalancedAtoms(atoms: readonly Atom[]): Atom[] {
  const totals = new Map<Atom['recognitionClass'], number>();
  for (const atom of atoms) totals.set(atom.recognitionClass, (totals.get(atom.recognitionClass) ?? 0) + atom.areaMetres2);
  return atoms.map((atom) => ({
    ...atom,
    areaMetres2: atom.areaMetres2 / Math.max(EPSILON, totals.get(atom.recognitionClass) ?? 0),
  }));
}

function independentAtomUpperBound(
  atoms: readonly Atom[], fieldCount: number, tolerance: number,
): number {
  let retained = 0;
  let total = 0;
  for (const atom of atoms) {
    total += atom.areaMetres2;
    let best = 0;
    for (const sweep of atom.alternatives) {
      if (!sweep.periodicCompatible) continue;
      // Optimistic relaxation: every atom receives its own perfectly matched
      // beta and independently placed disjoint bands. Only its recorded frame
      // remainder remains. This contains every globally shared-field solution.
      const perField = sweep.shapeErrorMetres <= EPSILON
        ? 1
        : Math.min(1, tolerance / sweep.shapeErrorMetres);
      best = Math.max(best, Math.min(1, fieldCount * perField));
    }
    retained += atom.areaMetres2 * best;
  }
  return retained / Math.max(EPSILON, total);
}

function retainedByClass(
  atoms: readonly Atom[], fields: readonly Field[], tolerance: number,
): Record<string, number> {
  const totals = new Map<string, number>();
  const retained = new Map<string, number>();
  const scored = scoreFields(atoms, fields, tolerance);
  atoms.forEach((atom, index) => {
    totals.set(atom.recognitionClass, (totals.get(atom.recognitionClass) ?? 0) + atom.areaMetres2);
    retained.set(atom.recognitionClass, (retained.get(atom.recognitionClass) ?? 0)
      + atom.areaMetres2 * scored.assignments[index]!.fraction);
  });
  return Object.fromEntries([...totals].map(([key, total]) => [key, (retained.get(key) ?? 0) / Math.max(EPSILON, total)]));
}

function allocationVectors(total: number, dimensions: number, prefix: number[] = []): number[][] {
  if (dimensions === 1) return [[...prefix, total]];
  return Array.from({ length: total + 1 }, (_, count) => allocationVectors(total - count, dimensions - 1, [...prefix, count])).flat();
}

function objectiveCatalogues(atoms: readonly Atom[], tolerance: number): Record<string, unknown> {
  const classes = [...new Set(atoms.map((atom) => atom.recognitionClass))];
  const selections = new Map<string, { atoms: Atom[]; fields: Field[] }>();
  for (const recognitionClass of classes) {
    const objectiveAtoms = atoms.filter((atom) => atom.recognitionClass === recognitionClass);
    selections.set(recognitionClass, {
      atoms: objectiveAtoms,
      fields: selectFields(objectiveAtoms, tolerance, ONE_LAYER_MAX_FIELDS).fields,
    });
  }
  const totalAreaFields = selectFields(atoms, tolerance, ONE_LAYER_MAX_FIELDS).fields;
  const catalogues: Record<string, unknown> = {};
  for (const [name, selection] of selections) {
    const total = selection.atoms.reduce((sum, atom) => sum + atom.areaMetres2, 0);
    catalogues[`${name}-only`] = Object.fromEntries([4, 8].map((fieldCount) => {
      const fields = selection.fields.slice(0, fieldCount);
      return [String(fieldCount), {
        status: 'heuristic achieved lower bound, not a certified optimum',
        retainedObjectiveAreaFraction: scoreFields(selection.atoms, fields, tolerance).score / Math.max(EPSILON, total),
        independentPerAtomOptimisticUpperBound: independentAtomUpperBound(selection.atoms, fieldCount, tolerance),
        fields,
      }];
    }));
  }
  const totalArea = atoms.reduce((sum, atom) => sum + atom.areaMetres2, 0);
  catalogues['total-area'] = Object.fromEntries([4, 8].map((fieldCount) => {
    const fields = totalAreaFields.slice(0, fieldCount);
    return [String(fieldCount), {
      status: 'heuristic achieved lower bound, not a certified optimum',
      retainedSourceLateralAreaFraction: scoreFields(atoms, fields, tolerance).score / Math.max(EPSILON, totalArea),
      retainedByRecognitionClass: retainedByClass(atoms, fields, tolerance),
      independentPerAtomOptimisticUpperBound: independentAtomUpperBound(atoms, fieldCount, tolerance),
      fields,
    }];
  }));
  const balancedAtoms = recognitionBalancedAtoms(atoms);
  catalogues['mixed-class-prefix-allocation'] = Object.fromEntries([4, 8].map((fieldCount) => {
    let winner: { allocation: number[]; fields: Field[]; score: number } | null = null;
    for (const allocation of allocationVectors(fieldCount, classes.length)) {
      const fields = classes.flatMap((name, index) => selections.get(name)!.fields.slice(0, allocation[index]!));
      const score = scoreFields(balancedAtoms, fields, tolerance).score;
      if (!winner || score > winner.score) winner = { allocation, fields, score };
    }
    const fields = winner?.fields ?? [];
    return [String(fieldCount), {
      status: 'small mixed-prefix Pareto over class-specific heuristic catalogues; not a certified optimum',
      allocation: Object.fromEntries(classes.map((name, index) => [name, winner?.allocation[index] ?? 0])),
      retainedSourceLateralAreaFraction: scoreFields(atoms, fields, tolerance).score / Math.max(EPSILON, totalArea),
      retainedByRecognitionClass: retainedByClass(atoms, fields, tolerance),
      fields,
    }];
  }));
  return catalogues;
}

function summarizeSource(
  name: string, mesh: IndexedMesh, atoms: Atom[], expectedHash: string,
  sourceHasher: (mesh: IndexedMesh) => string = meshHash,
): Record<string, unknown> {
  const hash = sourceHasher(mesh);
  if (hash !== expectedHash) throw new Error(`${name} source mesh changed: ${hash}`);
  const totalMeshArea = Array.from({ length: mesh.indices.length / 3 }, (_, triangle) => triangleArea(mesh, triangle))
    .reduce((sum, value) => sum + value, 0);
  const candidateArea = atoms.reduce((sum, atom) => sum + atom.areaMetres2, 0);
  const candidates: Record<string, unknown> = {};
  for (const tolerance of TOLERANCES) {
    const selection = selectFields(recognitionBalancedAtoms(atoms), tolerance, ONE_LAYER_MAX_FIELDS);
    candidates[String(tolerance)] = {
      toleranceMetres: tolerance,
      selectionObjective: 'equal total weight for foliage, crisp structure, reproductive landmarks, and moss when present; source area within each class',
      greedyBalancedScores: selection.scores,
      pareto: FIELD_COUNTS.map((fieldCount) => reportCandidate(atoms, selection.fields, fieldCount, tolerance, boundsY(mesh))),
    };
  }
  return {
    name, sourceMeshSha256: hash,
    vertices: mesh.positions.length / 3, triangles: mesh.indices.length / 3,
    totalMeshAreaMetres2: totalMeshArea,
    lateralCandidateAtoms: atoms.length,
    lateralCandidateAreaMetres2: candidateArea,
    lateralCandidateAreaFractionOfWholeMesh: candidateArea / totalMeshArea,
    excludedBeforeFieldSelection: 'hubs, all axial/root/terminal caps, plume hairs, support sheets, and degenerate zero-capacity tip patches',
    candidates,
  };
}

const productionCal = makeEstonianGraminoidFixture(ESTONIAN_GRAMINOID_PROFILE_IDS.CALAMAGROSTIS_CANESCENS);
const isolatedCal = makeCalamagrostisCanescensBandlimitedSource();
const sphagnum = makeSphagnumCapillifoliumFixture();
console.error('[lateral-core] extracting exact source-body atoms');
const productionAtoms = calamagrostisAtoms(productionCal);
const productionNoRibbonCenterlineAtoms = calamagrostisAtoms(productionCal, false);
const isolatedAtoms = calamagrostisAtoms(isolatedCal);
const mossAtoms = sphagnumAtoms(sphagnum);
console.error(`[lateral-core] atoms production=${productionAtoms.length} isolated=${isolatedAtoms.length} moss=${mossAtoms.length}`);

const recipe = {
  schema: 'laas-groundcover-lateral-core-residual-cpu-gate/v1',
  methodRevision: 'global-periodic-y-bands-union-v2-with-objective-catalogue-audit',
  toleranceMetres: TOLERANCES,
  fieldCounts: FIELD_COUNTS,
  axes: AXIS_SAMPLES,
  intervalSamples: INTERVAL_SAMPLES,
  heightBins: HEIGHT_BINS,
  residualModes: [0, 1, 2, 3, 4],
  residualWindow: 'w(u)=4u(1-u), one joint RGB/extinction field, K is total live count after both layers',
  fieldSelectionObjective: 'recognition-class-balanced greedy, exact-atom refinement of top sixteen stratified candidates',
  periodicFieldChart: 'A_beta(q,h)=(q_x+beta_x*h,h,q_z+beta_z*h), with h=y and q=(x-beta_x*y,z-beta_z*y); every field owns one global interval I_f=[hMin,hMax]',
  horizontalSweepPolicy: 'ineligible: exact horizontal lattice periodicity forces h to be the world-y covector, so width-axis ribbon sweeps cannot be runtime field coordinates',
  ribbonSweepCoordinates: ['centerline'],
  ribbonSweepPremise: 'a ribbon segment is credited only where one globally placed I_f is contained in its endpoint y-range; retained fraction is the union length of qualifying global bands divided by |delta-y|, and endpoint side-frame drift is charged',
  globalIntervalPlacement: 'exact piecewise-linear marginal union optimization over deterministic class-stratified sample atoms, followed by exact all-atom scoring of the best twenty-four candidates',
  distanceMetric: 'conservative face-on projected-feature pre-gate, not exact anisotropic pixel filtering',
  pixelAngleRadians: PIXEL_ANGLE_RADIANS,
  distancesMetres: DISTANCES_METRES,
};
const recipeSha = sha256(canonicalJson(recipe));
const toolSha = sha256(await import('node:fs').then(({ readFileSync }) => readFileSync(import.meta.filename)));
const sourceSha = sha256(canonicalJson({
  productionCal: meshHash(productionCal.mesh),
  isolatedCal: meshHash(isolatedCal.mesh),
  sphagnum: meshHash(sphagnum.mesh),
}));
const output = resolve(OUTPUT_ROOT, sourceSha.slice(0, 16), recipeSha.slice(0, 16), toolSha.slice(0, 16));
mkdirSync(output, { recursive: true });

const report: Record<string, any> = {
  ...recipe,
  status: 'STRUCTURAL_AND_OPTIMISTIC_RESIDUAL_BOUND_ONLY',
  methodBoundary: {
    core: 'retained patches are credited as exact source triangles after common-axis/common-interval compatibility; this overstates a realizable cap-free mask core',
    residual: 'optimistic one-joint-field 1D height marginal using the required zero-boundary quadratic window and K_live=0..4; the shared 3D live-ray field and arbitrary partial opaque cutoffs remain untested',
    residualRevision: 'one joint quadratic-window RGB/extinction field; K=0..4 is the total live mode count, with no species or stratum multiplier. The fit remains more permissive than production because it projects positivity pointwise.',
    coreQuery: 'selection is compatible with the revised first oriented boundary crossing from either outside or inside occupancy; no cap triangle is credited',
    distance: 'world-space omissions are screened by a conservative face-on 55deg/2160px projected-feature pre-gate using width, span, and area. It avoids handwaving away coherent distant omissions but is not an exact anisotropic pixel filter.',
    crossSection: 'ribbon endpoint side-frame drift is charged. Tube and lanceolate slice-frame variation not represented by the recorded endpoints is optimistically ignored, so this structural pre-gate cannot prove GREEN; a RED result remains decisive.',
  },
  sources: {
    productionCalamagrostis: summarizeSource('production Calamagrostis six-shoot community', productionCal.mesh, productionAtoms, EXPECTED_PRODUCTION_CAL_MESH),
    productionCalamagrostisNoRibbonCenterlineAblation: summarizeSource(
      'production Calamagrostis with all periodic ribbon-centerline charts disabled',
      productionCal.mesh,
      productionNoRibbonCenterlineAtoms,
      EXPECTED_PRODUCTION_CAL_MESH,
    ),
    isolatedCalamagrostis: {},
    sphagnum: summarizeSource('dense Sphagnum community', sphagnum.mesh, mossAtoms, EXPECTED_SPHAGNUM_MESH, meshBinaryHash),
  },
};

console.error('[lateral-core] auditing class-specific and mixed field allocations');
report.optimizationAudit = {
  status: 'all reported class/total/mixed solutions are achieved heuristic lower bounds; independent-atom values are valid but deliberately loose optimistic upper bounds',
  productionCalamagrostis: Object.fromEntries(TOLERANCES.map((tolerance) => [
    String(tolerance),
    objectiveCatalogues(productionAtoms, tolerance),
  ])),
};

// Replace the accidental production structural duplicate in the isolated
// report with a genuine isolated calculation while keeping its source hash.
report.sources.isolatedCalamagrostis = {
  name: 'isolated exact production shoot',
  sourceMeshSha256: meshHash(isolatedCal.mesh),
  vertices: isolatedCal.mesh.positions.length / 3,
  triangles: isolatedCal.mesh.indices.length / 3,
  lateralCandidateAtoms: isolatedAtoms.length,
  analyses: Object.fromEntries(TOLERANCES.map((tolerance) => {
    const selection = selectFields(recognitionBalancedAtoms(isolatedAtoms), tolerance, ONE_LAYER_MAX_FIELDS);
    return [String(tolerance), {
      toleranceMetres: tolerance,
      pareto: FIELD_COUNTS.map((fieldCount) => reportCandidate(isolatedAtoms, selection.fields, fieldCount, tolerance, boundsY(isolatedCal.mesh))),
    }];
  })),
};

const metricsPath = resolve(output, 'metrics.json');
writeFileSync(metricsPath, `${JSON.stringify(report, null, 2)}\n`);
const index = {
  schema: 'laas-groundcover-lateral-core-residual-cpu-gate-index/v1',
  tool: 'tools/groundcover-bake/analyze-lateral-core-analytic-residual.ts',
  toolSha256: toolSha,
  sourceSetSha256: sourceSha,
  recipeSha256: recipeSha,
  metrics: 'metrics.json',
  metricsSha256: sha256(await import('node:fs').then(({ readFileSync }) => readFileSync(metricsPath))),
};
writeFileSync(resolve(output, 'index.json'), `${JSON.stringify(index, null, 2)}\n`);
console.log(JSON.stringify({ output, ...index }, null, 2));
