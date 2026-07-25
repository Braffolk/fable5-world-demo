/**
 * Offline-only actual-asset gate for a tiny coupled, band-limited event-sheet
 * record. This does not write a runtime asset and does not edit a shader.
 *
 * The candidate is deliberately optimistic. Nine persistent pinhole labels are
 * traced exactly, their conditional hit measure is split into three monotone
 * depth intervals, and each interval stores an actual authored medoid event.
 * Nothing averages positions, colours, normals, materials, or botanical marks.
 * The gate asks whether those three sheets remain a globally coherent coupling
 * under transported origins and a structured camera/pixel lattice.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';
import {
  TriangleBvh,
  decodeOwnedProfileGeometry,
  type CensusVec3,
} from '../../ExteriorClosureCensus';
import { periodicNearestSuccessor } from '../../OriginAwareRayTruth';

interface Args { [key: string]: string | boolean }

interface Event {
  id: string;
  triangleId: number;
  copyX: number;
  copyZ: number;
  t: number;
  world: CensusVec3;
  classIndex: number;
  panicle: boolean;
}

interface Sheet {
  event: Event;
  mass: number;
  maximumDepthResidual: number;
}

interface Record3 {
  hitCount: number;
  coverage: number;
  exactPanicleMass: number;
  representedPanicleMass: number;
  conditionalWasserstein1Metres: number;
  sheets: readonly [Sheet | null, Sheet | null, Sheet | null];
}

interface CameraGrid {
  label: string;
  elevationDegrees: number;
  footprintLevel: number;
  records: Record3[][];
}

const EXPECTED_SOURCE_SHA256 =
  '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const SOURCE = 'src/assets/groundcover/calamagrostis-canescens.gcrp';
const HORIZON_METRES = 155;
const ORIGIN_EPSILON_METRES = 1e-9;
const MICRO_SAMPLE_GRID = [-2 / 3, 0, 2 / 3] as const;
const MICRO_SAMPLE_COUNT = 9;
const SHEET_COUNT = 3;
const FOV_DEGREES = 55;
const DEVICE_PIXEL_HEIGHT = 2160;
const PIXEL_ANGLE_RADIANS = FOV_DEGREES / DEVICE_PIXEL_HEIGHT * Math.PI / 180;
const FOOTPRINT_LEVELS = [1, 2, 4, 8] as const;
const CAMERA_GRID_SIZE = 17;
const CAMERA_FRAME_STEP_METRES = 0.005;
const SAME_LINE_STEP_METRES = 0.01;
const SAME_LINE_STEP_COUNT = 17;
const MAXIMUM_RESIDENT_BYTES = 51_121_152;
const ATLAS_DIRECTIONS = 71;
const STORED_TILE_WIDTH = 258;
const RECORD_BYTES = 8;
const PALETTE_ENTRY_LIMIT = 64;
const TARGETED_PANICLE_TRIANGLES = 16;
const PERMUTATIONS = [
  [0, 1, 2], [0, 2, 1], [1, 0, 2],
  [1, 2, 0], [2, 0, 1], [2, 1, 0],
] as const;

const PALETTE = [
  { name: 'leaf-root', rgb: [0.10, 0.27, 0.055], panicle: false },
  { name: 'leaf-tip', rgb: [0.37, 0.51, 0.13], panicle: false },
  { name: 'culm', rgb: [0.43, 0.53, 0.20], panicle: false },
  { name: 'rhizome', rgb: [0.31, 0.24, 0.075], panicle: false },
  { name: 'panicle-axis', rgb: [0.48, 0.31, 0.24], panicle: true },
  { name: 'spikelet', rgb: [0.57, 0.35, 0.39], panicle: true },
  { name: 'callus-hair', rgb: [0.88, 0.80, 0.70], panicle: true },
  { name: 'anther', rgb: [0.42, 0.10, 0.34], panicle: true },
] as const;

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

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) =>
    `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(',')}}`;
}

function sha256(value: Uint8Array | string): string {
  return createHash('sha256').update(value).digest('hex');
}

function add(a: CensusVec3, b: CensusVec3, scale = 1): CensusVec3 {
  return [a[0] + b[0] * scale, a[1] + b[1] * scale, a[2] + b[2] * scale];
}

function normalize(value: CensusVec3): CensusVec3 {
  const inverse = 1 / Math.hypot(value[0], value[1], value[2]);
  return [value[0] * inverse, value[1] * inverse, value[2] * inverse];
}

function direction(elevationDegrees: number, azimuthRadians = 0): CensusVec3 {
  const elevation = elevationDegrees * Math.PI / 180;
  const horizontal = Math.cos(elevation);
  return normalize([
    horizontal * Math.cos(azimuthRadians),
    -Math.sin(elevation),
    horizontal * Math.sin(azimuthRadians),
  ]);
}

/** Analytic azimuth/elevation differential frame; no helper-axis switch. */
function sensorFrame(elevationDegrees: number, azimuthRadians = 0): {
  direction: CensusVec3;
  horizontal: CensusVec3;
  vertical: CensusVec3;
} {
  const elevation = elevationDegrees * Math.PI / 180;
  const d = direction(elevationDegrees, azimuthRadians);
  const horizontal: CensusVec3 = [-Math.sin(azimuthRadians), 0, Math.cos(azimuthRadians)];
  const vertical: CensusVec3 = [
    -Math.sin(elevation) * Math.cos(azimuthRadians),
    -Math.cos(elevation),
    -Math.sin(elevation) * Math.sin(azimuthRadians),
  ];
  return { direction: d, horizontal, vertical };
}

function microDirections(
  elevationDegrees: number,
  azimuthRadians: number,
  footprintLevel: number,
  pixelOffset = 0,
): CensusVec3[] {
  const frame = sensorFrame(elevationDegrees, azimuthRadians);
  const central = normalize(add(frame.direction, frame.horizontal, pixelOffset * PIXEL_ANGLE_RADIANS));
  // Rebuild the local frame around the small horizontal pixel offset while
  // retaining the same persistent 3x3 sensor labels.
  const h = frame.horizontal;
  const v = normalize([
    h[1] * central[2] - h[2] * central[1],
    h[2] * central[0] - h[0] * central[2],
    h[0] * central[1] - h[1] * central[0],
  ]);
  const radius = 0.5 * PIXEL_ANGLE_RADIANS * FOOTPRINT_LEVELS[footprintLevel]!;
  const result: CensusVec3[] = [];
  for (const y of MICRO_SAMPLE_GRID) {
    for (const x of MICRO_SAMPLE_GRID) {
      result.push(normalize(add(add(central, h, x * radius), v, y * radius)));
    }
  }
  return result;
}

function microDirectionsAround(
  centralInput: CensusVec3,
  footprintLevel: number,
): CensusVec3[] {
  const central = normalize(centralInput);
  const helper: CensusVec3 = Math.abs(central[1]) < 0.8 ? [0, 1, 0] : [1, 0, 0];
  const horizontal = normalize([
    helper[1] * central[2] - helper[2] * central[1],
    helper[2] * central[0] - helper[0] * central[2],
    helper[0] * central[1] - helper[1] * central[0],
  ]);
  const vertical = normalize([
    central[1] * horizontal[2] - central[2] * horizontal[1],
    central[2] * horizontal[0] - central[0] * horizontal[2],
    central[0] * horizontal[1] - central[1] * horizontal[0],
  ]);
  const radius = 0.5 * PIXEL_ANGLE_RADIANS * FOOTPRINT_LEVELS[footprintLevel]!;
  const result: CensusVec3[] = [];
  for (const y of MICRO_SAMPLE_GRID) for (const x of MICRO_SAMPLE_GRID) {
    result.push(normalize(add(add(central, horizontal, x * radius), vertical, y * radius)));
  }
  return result;
}

function quantiles(valuesInput: readonly number[]): Record<string, number | null> {
  if (valuesInput.length === 0) {
    return { p50: null, p90: null, p95: null, p99: null, maximum: null };
  }
  const values = [...valuesInput].sort((a, b) => a - b);
  const at = (fraction: number): number => values[Math.floor((values.length - 1) * fraction)]!;
  return {
    p50: at(0.5), p90: at(0.9), p95: at(0.95), p99: at(0.99), maximum: values.at(-1)!,
  };
}

function atlasMipSizes(): number[] {
  const result: number[] = [];
  let size = STORED_TILE_WIDTH;
  while (true) {
    result.push(size);
    if (size === 1) return result;
    size = Math.ceil(size / 2);
  }
}

const args = parseArgs(process.argv.slice(2));
const sourcePath = resolve(typeof args.source === 'string' ? args.source : SOURCE);
const sourceBytes = readFileSync(sourcePath);
const sourceSha256 = sha256(sourceBytes);
if (sourceSha256 !== EXPECTED_SOURCE_SHA256) {
  throw new Error(`unexpected source SHA-256 ${sourceSha256}`);
}
const geometry = decodeOwnedProfileGeometry(sourceBytes);
const bvh = TriangleBvh.build(geometry);
const triangleClass = new Int8Array(geometry.triangleCount).fill(-1);
const sourceView = new DataView(sourceBytes.buffer, sourceBytes.byteOffset, sourceBytes.byteLength);
const vertexOffset = sourceView.getUint32(84, true);
const triangleOffset = sourceView.getUint32(88, true);

function classForTriangle(triangleId: number): number {
  const cached = triangleClass[triangleId]!;
  if (cached >= 0) return cached;
  const triangle = triangleOffset + triangleId * 16;
  const rgb = [0, 0, 0];
  for (let corner = 0; corner < 3; corner++) {
    const vertex = sourceView.getUint32(triangle + corner * 4, true);
    const record = vertexOffset + vertex * 16;
    rgb[0] += sourceView.getUint16(record + 6, true) / (65535 * 3);
    rgb[1] += sourceView.getUint16(record + 8, true) / (65535 * 3);
    rgb[2] += sourceView.getUint16(record + 10, true) / (65535 * 3);
  }
  let best = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < PALETTE.length; index++) {
    const reference = PALETTE[index]!.rgb;
    const distance = (rgb[0]! - reference[0]) ** 2
      + (rgb[1]! - reference[1]) ** 2
      + (rgb[2]! - reference[2]) ** 2;
    if (distance < bestDistance) {
      bestDistance = distance;
      best = index;
    }
  }
  triangleClass[triangleId] = best;
  return best;
}

let tracedRays = 0;
let tracedHits = 0;
function trace(origin: CensusVec3, rayDirection: CensusVec3): Event | null {
  tracedRays++;
  const hit = periodicNearestSuccessor(
    geometry, bvh, origin, rayDirection, HORIZON_METRES, ORIGIN_EPSILON_METRES,
  );
  if (!hit) return null;
  tracedHits++;
  const classIndex = classForTriangle(hit.triangleId);
  return {
    id: `${hit.copyX}:${hit.copyZ}:${hit.triangleId}`,
    triangleId: hit.triangleId,
    copyX: hit.copyX,
    copyZ: hit.copyZ,
    t: hit.t,
    world: add(origin, rayDirection, hit.t),
    classIndex,
    panicle: PALETTE[classIndex]!.panicle,
  };
}

function reduce(eventsInput: readonly (Event | null)[]): Record3 {
  const events = eventsInput.filter((event): event is Event => event !== null)
    .sort((left, right) => left.t - right.t || left.id.localeCompare(right.id));
  const sheets: [Sheet | null, Sheet | null, Sheet | null] = [null, null, null];
  let wasserstein = 0;
  let representedPanicleMass = 0;
  for (let sheet = 0; sheet < SHEET_COUNT; sheet++) {
    const start = Math.floor(sheet * events.length / SHEET_COUNT);
    const end = Math.floor((sheet + 1) * events.length / SHEET_COUNT);
    if (end <= start) continue;
    const members = events.slice(start, end);
    const medoid = members[Math.floor((members.length - 1) / 2)]!;
    const mass = members.length / Math.max(1, events.length);
    let maximumDepthResidual = 0;
    for (const member of members) {
      const residual = Math.abs(member.t - medoid.t);
      wasserstein += residual / events.length;
      maximumDepthResidual = Math.max(maximumDepthResidual, residual);
    }
    representedPanicleMass += mass * Number(medoid.panicle);
    sheets[sheet] = { event: medoid, mass, maximumDepthResidual };
  }
  const panicleHits = events.filter((event) => event.panicle).length;
  return {
    hitCount: events.length,
    coverage: events.length / MICRO_SAMPLE_COUNT,
    exactPanicleMass: events.length > 0 ? panicleHits / events.length : 0,
    representedPanicleMass,
    conditionalWasserstein1Metres: wasserstein,
    sheets,
  };
}

function traceBeam(origin: CensusVec3, directions: readonly CensusVec3[]): Record3 {
  return reduce(directions.map((rayDirection) => trace(origin, rayDirection)));
}

function eventCost(left: Event | null, right: Event | null): number {
  if (!left && !right) return 0;
  if (!left || !right) return 5;
  const spatial = Math.hypot(
    left.world[0] - right.world[0],
    left.world[1] - right.world[1],
    left.world[2] - right.world[2],
  );
  return spatial + Number(left.panicle !== right.panicle) * 0.25
    + Number(left.classIndex !== right.classIndex) * 0.05;
}

function bestPermutation(left: Record3, right: Record3): {
  permutation: readonly [number, number, number];
  identityCost: number;
  optimalCost: number;
} {
  const a = left.sheets.map((sheet) => sheet?.event ?? null);
  const b = right.sheets.map((sheet) => sheet?.event ?? null);
  const cost = (permutation: readonly number[]): number =>
    permutation.reduce((sum, target, source) => sum + eventCost(a[source]!, b[target]!), 0);
  let best: readonly [number, number, number] = PERMUTATIONS[0];
  let optimalCost = cost(best);
  for (const permutation of PERMUTATIONS.slice(1)) {
    const candidate = cost(permutation);
    if (candidate < optimalCost) {
      optimalCost = candidate;
      best = permutation;
    }
  }
  return { permutation: best, identityCost: cost(PERMUTATIONS[0]), optimalCost };
}

function hasThreeSheets(record: Record3): boolean {
  return record.sheets.every((sheet) => sheet !== null);
}

function inverse(permutation: readonly number[]): [number, number, number] {
  const result: [number, number, number] = [0, 0, 0];
  for (let source = 0; source < 3; source++) result[permutation[source]!] = source;
  return result;
}

function compose(
  first: readonly number[],
  second: readonly number[],
): [number, number, number] {
  return [second[first[0]!]!, second[first[1]!]!, second[first[2]!]!];
}

const w1Errors: number[] = [];
const panicleMassErrors: number[] = [];
const maximumClusterDepthResiduals: number[] = [];
let intrinsicHitEvents = 0;
let intrinsicPanicleHitEvents = 0;
let intrinsicRecordsWithPanicle = 0;
function collectIntrinsic(record: Record3): void {
  if (record.hitCount === 0) return;
  intrinsicHitEvents += record.hitCount;
  intrinsicPanicleHitEvents += Math.round(record.exactPanicleMass * record.hitCount);
  if (record.exactPanicleMass > 0) intrinsicRecordsWithPanicle++;
  w1Errors.push(record.conditionalWasserstein1Metres);
  panicleMassErrors.push(Math.abs(record.exactPanicleMass - record.representedPanicleMass));
  for (const sheet of record.sheets) {
    if (sheet) maximumClusterDepthResiduals.push(sheet.maximumDepthResidual);
  }
}

// 1. Exact same-micro-line transport. Every 3x3 member is created once and
// subsequent origins advance along that member's own direction.
let censorComparisons = 0;
let censorIdentityMatches = 0;
let censorClassMatches = 0;
let censorPanicleMatches = 0;
const censorWorldJumps: number[] = [];
const censorDefinitions = [0.1, 1, 5, 15, 35, 75];
for (let phase = 0; phase < 4; phase++) {
  const origin: CensusVec3 = [
    geometry.tileOriginX + geometry.tileSizeX * ((phase * 0.2795084971874737 + 0.173) % 1),
    geometry.bounds.min[1] + (geometry.bounds.max[1] - geometry.bounds.min[1]) * (0.25 + 0.16 * phase),
    geometry.tileOriginZ + geometry.tileSizeZ * ((phase * 0.3819660112501051 + 0.317) % 1),
  ];
  for (const elevation of censorDefinitions) {
    const rays = microDirections(elevation, phase * Math.PI / 4, 0);
    let previous: Record3 | null = null;
    for (let step = 0; step < SAME_LINE_STEP_COUNT; step++) {
      const distance = step * SAME_LINE_STEP_METRES;
      const events = rays.map((rayDirection) => trace(add(origin, rayDirection, distance), rayDirection));
      const current = reduce(events);
      collectIntrinsic(current);
      if (previous) {
        for (let sheet = 0; sheet < SHEET_COUNT; sheet++) {
          const before = previous.sheets[sheet]?.event ?? null;
          const after = current.sheets[sheet]?.event ?? null;
          if (!before || before.t <= SAME_LINE_STEP_METRES + ORIGIN_EPSILON_METRES) continue;
          censorComparisons++;
          if (after?.id === before.id) censorIdentityMatches++;
          if (after?.classIndex === before.classIndex) censorClassMatches++;
          if (after?.panicle === before.panicle) censorPanicleMatches++;
          if (after) censorWorldJumps.push(Math.hypot(
            after.world[0] - before.world[0],
            after.world[1] - before.world[1],
            after.world[2] - before.world[2],
          ));
        }
      }
      previous = current;
    }
  }
}

// 2. Structured pinhole camera/pixel grids. Columns are adjacent real sensor
// pixels; rows are 5 mm lateral camera translations with persistent labels.
const grids: CameraGrid[] = [];
const cameraCases = [
  { phase: 0, elevation: 5, level: 0 },
  { phase: 1, elevation: 5, level: 0 },
  { phase: 0, elevation: 35, level: 0 },
  { phase: 1, elevation: 35, level: 0 },
  { phase: 0, elevation: 5, level: 1 },
  { phase: 0, elevation: 5, level: 2 },
  { phase: 0, elevation: 5, level: 3 },
] as const;
for (const test of cameraCases) {
  const azimuth = test.phase * Math.PI / 3;
  const frame = sensorFrame(test.elevation, azimuth);
  const baseOrigin: CensusVec3 = [
    geometry.tileOriginX + geometry.tileSizeX * (test.phase === 0 ? 0.271 : 0.683),
    geometry.bounds.min[1] + (geometry.bounds.max[1] - geometry.bounds.min[1]) * 0.55,
    geometry.tileOriginZ + geometry.tileSizeZ * (test.phase === 0 ? 0.417 : 0.193),
  ];
  const records: Record3[][] = [];
  for (let row = 0; row < CAMERA_GRID_SIZE; row++) {
    const rowOffset = row - (CAMERA_GRID_SIZE - 1) / 2;
    const camera = add(baseOrigin, frame.horizontal, rowOffset * CAMERA_FRAME_STEP_METRES);
    const recordRow: Record3[] = [];
    for (let column = 0; column < CAMERA_GRID_SIZE; column++) {
      const pixelOffset = column - (CAMERA_GRID_SIZE - 1) / 2;
      const record = traceBeam(
        camera,
        microDirections(test.elevation, azimuth, test.level, pixelOffset),
      );
      collectIntrinsic(record);
      recordRow.push(record);
    }
    records.push(recordRow);
  }
  grids.push({
    label: `phase-${test.phase}-elevation-${test.elevation}-footprint-${test.level}`,
    elevationDegrees: test.elevation,
    footprintLevel: test.level,
    records,
  });
}

// 3. Non-vacuous panicle preservation probes. Select actual high-panicle
// triangles, start 2 cm outside their geometric normal, and trace all four
// footprint levels back into the authored surface. These probes do not make
// the global camera-path result easier; they only prevent a zero-panicle path
// from falsely passing the mark-retention metric.
const panicleTargets: Array<{
  triangleId: number;
  centroid: CensusVec3;
  normal: CensusVec3;
}> = [];
const botanicalMiddleY = 0.5 * (geometry.bounds.min[1] + geometry.bounds.max[1]);
for (let triangleId = 0;
  triangleId < geometry.triangleCount && panicleTargets.length < TARGETED_PANICLE_TRIANGLES;
  triangleId++) {
  if (!PALETTE[classForTriangle(triangleId)]!.panicle) continue;
  const triangle = triangleId * 3;
  const aIndex = geometry.triangles[triangle]! * 3;
  const bIndex = geometry.triangles[triangle + 1]! * 3;
  const cIndex = geometry.triangles[triangle + 2]! * 3;
  const a: CensusVec3 = [geometry.positions[aIndex]!, geometry.positions[aIndex + 1]!, geometry.positions[aIndex + 2]!];
  const b: CensusVec3 = [geometry.positions[bIndex]!, geometry.positions[bIndex + 1]!, geometry.positions[bIndex + 2]!];
  const c: CensusVec3 = [geometry.positions[cIndex]!, geometry.positions[cIndex + 1]!, geometry.positions[cIndex + 2]!];
  const centroid: CensusVec3 = [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3];
  if (centroid[1] < botanicalMiddleY) continue;
  const ab: CensusVec3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const ac: CensusVec3 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const cross: CensusVec3 = [
    ab[1] * ac[2] - ab[2] * ac[1],
    ab[2] * ac[0] - ab[0] * ac[2],
    ab[0] * ac[1] - ab[1] * ac[0],
  ];
  if (Math.hypot(cross[0], cross[1], cross[2]) <= 1e-10) continue;
  if (panicleTargets.some((target) => Math.hypot(
    target.centroid[0] - centroid[0], target.centroid[1] - centroid[1], target.centroid[2] - centroid[2],
  ) < 0.004)) continue;
  panicleTargets.push({ triangleId, centroid, normal: normalize(cross) });
}
let targetedPanicleRecords = 0;
let targetedPanicleHitEvents = 0;
let targetedPanicleClassHits = 0;
for (const target of panicleTargets) {
  const origin = add(target.centroid, target.normal, 0.02);
  const central: CensusVec3 = [-target.normal[0], -target.normal[1], -target.normal[2]];
  for (let level = 0; level < FOOTPRINT_LEVELS.length; level++) {
    const record = traceBeam(origin, microDirectionsAround(central, level));
    collectIntrinsic(record);
    targetedPanicleRecords++;
    targetedPanicleHitEvents += record.hitCount;
    targetedPanicleClassHits += Math.round(record.exactPanicleMass * record.hitCount);
  }
}

let edgeCount = 0;
let edgeIdentityOptimal = 0;
const edgeExcessCosts: number[] = [];
let plaquettes = 0;
let nontrivialHolonomy = 0;
const gridMetrics: Record<string, unknown>[] = [];
for (const grid of grids) {
  let localEdges = 0;
  let localIdentity = 0;
  const localExcess: number[] = [];
  let localPlaquettes = 0;
  let localHolonomy = 0;
  for (let row = 0; row < CAMERA_GRID_SIZE; row++) {
    for (let column = 0; column < CAMERA_GRID_SIZE; column++) {
      const record = grid.records[row]![column]!;
      for (const neighbour of [
        column + 1 < CAMERA_GRID_SIZE ? grid.records[row]![column + 1]! : null,
        row + 1 < CAMERA_GRID_SIZE ? grid.records[row + 1]![column]! : null,
      ]) {
        // Missing sheets have a permutation symmetry of their own and can
        // manufacture false holonomy. The hard transport metric therefore
        // uses only records where all three authored-event sheets exist.
        if (!neighbour || !hasThreeSheets(record) || !hasThreeSheets(neighbour)) continue;
        const match = bestPermutation(record, neighbour);
        const identity = match.permutation.every((target, source) => target === source);
        edgeCount++;
        localEdges++;
        if (identity) {
          edgeIdentityOptimal++;
          localIdentity++;
        }
        const excess = Math.max(0, match.identityCost - match.optimalCost);
        edgeExcessCosts.push(excess);
        localExcess.push(excess);
      }
      if (row + 1 >= CAMERA_GRID_SIZE || column + 1 >= CAMERA_GRID_SIZE) continue;
      const a = record;
      const b = grid.records[row]![column + 1]!;
      const c = grid.records[row + 1]![column]!;
      const d = grid.records[row + 1]![column + 1]!;
      if (![a, b, c, d].every(hasThreeSheets)) continue;
      const ab = bestPermutation(a, b).permutation;
      const bd = bestPermutation(b, d).permutation;
      const cd = bestPermutation(c, d).permutation;
      const ac = bestPermutation(a, c).permutation;
      const loop = compose(compose(compose(ab, bd), inverse(cd)), inverse(ac));
      const nontrivial = loop.some((target, source) => target !== source);
      plaquettes++;
      localPlaquettes++;
      if (nontrivial) {
        nontrivialHolonomy++;
        localHolonomy++;
      }
    }
  }
  gridMetrics.push({
    label: grid.label,
    edges: localEdges,
    monotoneRankIsOptimalFraction: localIdentity / localEdges,
    monotoneRankExcessTransportCostMetres: quantiles(localExcess),
    plaquettes: localPlaquettes,
    nontrivialHolonomyFraction: localHolonomy / localPlaquettes,
  });
}

const mipSizes = atlasMipSizes();
const atlasRecordCount = mipSizes.reduce((sum, size) => sum + size * size, 0) * ATLAS_DIRECTIONS;
const atlasBytes = atlasRecordCount * RECORD_BYTES;
const metadataBudget = MAXIMUM_RESIDENT_BYTES - atlasBytes;
const configuration = {
  sourceSha256,
  candidate: {
    recordBits: {
      conditionalHitCount: 4,
      sheets: 3,
      eachSheet: { lineDepth: 14, jointAuthoredAppearanceNormalMarkPalette: 6 },
      total: 64,
    },
    coupling: 'persistent xi; conditional monotone depth transport into three actual-event medoids; no attribute averaging',
    angularCarrier: '71 sphere directions; one enclosing direction triangle; corresponding sheet records remain coupled by rank',
    liveReconstruction: 'three fixed record reads; selected-sheet three-point plane reconstruction; categorical appearance from one selected actual atom',
    runtimeCostCeiling: { reads: 3, fmaEquivalent: 192 },
    runtimeForbidden: ['loop', 'march', 'traversal', 'candidate list', 'shell', 'per-species query'],
  },
  footprint: {
    sampleLabels: 'persistent 3x3 sensor grid',
    pixelAngleRadians: PIXEL_ANGLE_RADIANS,
    levels: FOOTPRINT_LEVELS,
    bakeRule: 'every level is retraced from exact micro-lines; coarse levels are not arithmetic mips',
  },
  atlas: {
    directions: ATLAS_DIRECTIONS,
    mipSizes,
    recordBytes: RECORD_BYTES,
    recordCount: atlasRecordCount,
    residentBytes: atlasBytes,
    residentLimitBytes: MAXIMUM_RESIDENT_BYTES,
    metadataAndPaletteBudgetBytes: metadataBudget,
    paletteEntryLimit: PALETTE_ENTRY_LIMIT,
  },
  gate: {
    sameLine: { phases: 4, elevationsDegrees: censorDefinitions, steps: SAME_LINE_STEP_COUNT, stepMetres: SAME_LINE_STEP_METRES },
    cameraGrid: { size: CAMERA_GRID_SIZE, frameStepMetres: CAMERA_FRAME_STEP_METRES, cases: cameraCases },
    targetedPanicle: { triangles: TARGETED_PANICLE_TRIANGLES, footprintLevels: FOOTPRINT_LEVELS, normalOffsetMetres: 0.02 },
    hardThresholds: {
      sameLineUncrossedEventIdentity: 1,
      nontrivialPlaquetteHolonomyFraction: 0,
      panicleMassAbsoluteErrorP95Maximum: 1 / MICRO_SAMPLE_COUNT,
      conditionalDepthWassersteinP95MaximumMetres: 0.02,
      monotoneRankExcessTransportP95MaximumMetres: 0.02,
    },
    transportMetricDomain: 'only edges/plaquettes whose every record contains all three sheets; missing-sheet permutation symmetry is excluded',
  },
};
const configurationSha256 = sha256(canonicalJson(configuration));
const scriptBytes = readFileSync(fileURLToPath(import.meta.url));
const toolSha256 = sha256(scriptBytes);
const outputRoot = resolve(
  typeof args.output === 'string'
    ? args.output
    : `data/work/groundcover-coupled-quantile-sheet-gate/${sourceSha256.slice(0, 16)}/${configurationSha256.slice(0, 16)}`,
);
const qaRoot = resolve(outputRoot, 'qa');
mkdirSync(qaRoot, { recursive: true });

const metrics = {
  schema: 'laas-groundcover-coupled-quantile-sheet-gate/v1',
  source: {
    path: sourcePath,
    sha256: sourceSha256,
    triangleCount: geometry.triangleCount,
    tileSizeMetres: [geometry.tileSizeX, geometry.tileSizeZ],
    botanicalBounds: geometry.bounds,
  },
  configurationSha256,
  toolSha256,
  accelerator: bvh.metrics,
  trace: { rays: tracedRays, hits: tracedHits, hitFraction: tracedHits / tracedRays },
  intrinsicReduction: {
    records: w1Errors.length,
    exactHitEvents: intrinsicHitEvents,
    exactPanicleHitEvents: intrinsicPanicleHitEvents,
    recordsContainingPanicle: intrinsicRecordsWithPanicle,
    conditionalDepthWasserstein1Metres: quantiles(w1Errors),
    maximumWithinSheetDepthResidualMetres: quantiles(maximumClusterDepthResiduals),
    panicleMassAbsoluteError: quantiles(panicleMassErrors),
  },
  transportedSameLine: {
    uncrossedSheetComparisons: censorComparisons,
    exactEventIdentityFraction: censorIdentityMatches / Math.max(1, censorComparisons),
    botanicalClassIdentityFraction: censorClassMatches / Math.max(1, censorComparisons),
    panicleIdentityFraction: censorPanicleMatches / Math.max(1, censorComparisons),
    decodedWorldEventJumpMetres: quantiles(censorWorldJumps),
    interpretation: 'all micro-lines are transported exactly; changing a medoid before its stored event is crossed violates the required right-censoring coupling',
  },
  pinholeCameraPath: {
    edges: edgeCount,
    monotoneRankIsMinimumCostFraction: edgeIdentityOptimal / edgeCount,
    monotoneRankExcessTransportCostMetres: quantiles(edgeExcessCosts),
    plaquettes,
    nontrivialHolonomy: nontrivialHolonomy,
    nontrivialHolonomyFraction: nontrivialHolonomy / plaquettes,
    grids: gridMetrics,
    interpretation: 'nonidentity loop transport means no single global three-sheet labelling exists on this sampled camera/pixel patch; interpolation must tear, swap, or stretch somewhere',
  },
  targetedPanicle: {
    sourceTriangleIds: panicleTargets.map((target) => target.triangleId),
    records: targetedPanicleRecords,
    hitEvents: targetedPanicleHitEvents,
    panicleHitEvents: targetedPanicleClassHits,
    panicleFraction: targetedPanicleClassHits / Math.max(1, targetedPanicleHitEvents),
  },
  cost: configuration.atlas,
};

const hard = configuration.gate.hardThresholds;
const w1p95 = quantiles(w1Errors).p95 ?? Number.POSITIVE_INFINITY;
const panicleP95 = quantiles(panicleMassErrors).p95 ?? Number.POSITIVE_INFINITY;
const excessP95 = quantiles(edgeExcessCosts).p95 ?? Number.POSITIVE_INFINITY;
const passes = {
  residentMemory: atlasBytes <= MAXIMUM_RESIDENT_BYTES,
  fixedRuntimeWork: true,
  sameLineRightCensor: metrics.transportedSameLine.exactEventIdentityFraction === hard.sameLineUncrossedEventIdentity,
  integrableGlobalSheets: metrics.pinholeCameraPath.nontrivialHolonomyFraction === hard.nontrivialPlaquetteHolonomyFraction,
  panicleMass: panicleP95 <= hard.panicleMassAbsoluteErrorP95Maximum,
  conditionalDepth: w1p95 <= hard.conditionalDepthWassersteinP95MaximumMetres,
  monotoneTransport: excessP95 <= hard.monotoneRankExcessTransportP95MaximumMetres,
};
const accepted = Object.values(passes).every(Boolean);
const report = {
  ...metrics,
  passes,
  accepted,
  decision: accepted
    ? 'PROVISIONAL PASS: intrinsic coupling/cost gate passed; a full continuum codec and overlap-community bake are still required before runtime work.'
    : 'REJECT AND PARK: the optimistic three-sheet coupled record fails actual-asset identity/transport before quantisation, codec fitting, multi-species overlap, or shader integration.',
  limitations: [
    'This is one bounded actual-asset feasibility attempt, not a continuum certificate.',
    'Medoid events retain exact authored source marks. The 64-entry joint appearance/normal/mark palette is budgeted but not fitted; any palette error can only worsen quality.',
    'The camera-path oracle constructs the pointed-query record directly. A production atlas must additionally encode arbitrary inside-origin successor phase; failure here is therefore a lower bound.',
    'The source contains one Calamagrostis population. Offline union with other grass, forbs, litter, and moss increases event ambiguity and cannot repair a failure.',
  ],
};
const reportPath = resolve(outputRoot, 'metrics.json');
writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);

// Compact structured QA: each grid contributes three sheet-class panels and a
// plaquette-holonomy panel. Nearest scaling is intentional; every square is one
// exact camera-frame/pixel query, not a beauty render.
const scale = 8;
const panelWidth = CAMERA_GRID_SIZE * scale;
const panelHeight = CAMERA_GRID_SIZE * scale;
const imageWidth = panelWidth * 4;
const imageHeight = panelHeight * grids.length;
const rgba = new Uint8Array(imageWidth * imageHeight * 4);
function putPixel(x: number, y: number, rgb: readonly number[]): void {
  const index = (y * imageWidth + x) * 4;
  rgba[index] = Math.round(Math.max(0, Math.min(1, rgb[0]!)) * 255);
  rgba[index + 1] = Math.round(Math.max(0, Math.min(1, rgb[1]!)) * 255);
  rgba[index + 2] = Math.round(Math.max(0, Math.min(1, rgb[2]!)) * 255);
  rgba[index + 3] = 255;
}
for (let gridIndex = 0; gridIndex < grids.length; gridIndex++) {
  const grid = grids[gridIndex]!;
  for (let row = 0; row < CAMERA_GRID_SIZE; row++) {
    for (let column = 0; column < CAMERA_GRID_SIZE; column++) {
      const record = grid.records[row]![column]!;
      for (let panel = 0; panel < 3; panel++) {
        const event = record.sheets[panel]?.event ?? null;
        const rgb = event ? PALETTE[event.classIndex]!.rgb : [0.02, 0.02, 0.02];
        for (let py = 0; py < scale; py++) for (let px = 0; px < scale; px++) {
          putPixel(panel * panelWidth + column * scale + px, gridIndex * panelHeight + row * scale + py, rgb);
        }
      }
      let holonomy = false;
      if (row + 1 < CAMERA_GRID_SIZE && column + 1 < CAMERA_GRID_SIZE) {
        const a = record;
        const b = grid.records[row]![column + 1]!;
        const c = grid.records[row + 1]![column]!;
        const d = grid.records[row + 1]![column + 1]!;
        if ([a, b, c, d].every(hasThreeSheets)) {
          const loop = compose(
            compose(compose(bestPermutation(a, b).permutation, bestPermutation(b, d).permutation), inverse(bestPermutation(c, d).permutation)),
            inverse(bestPermutation(a, c).permutation),
          );
          holonomy = loop.some((target, source) => target !== source);
        }
      }
      const rgb = holonomy ? [0.95, 0.08, 0.04] : [0.06, 0.35, 0.10];
      for (let py = 0; py < scale; py++) for (let px = 0; px < scale; px++) {
        putPixel(3 * panelWidth + column * scale + px, gridIndex * panelHeight + row * scale + py, rgb);
      }
    }
  }
}
const qaPath = resolve(qaRoot, '000-coupled-camera-path-sheets-and-holonomy.png');
await sharp(rgba, { raw: { width: imageWidth, height: imageHeight, channels: 4 } })
  .png({ compressionLevel: 9 })
  .toFile(qaPath);

const index = {
  schema: 'laas-groundcover-coupled-quantile-sheet-gate-index/v1',
  sourceSha256,
  configurationSha256,
  toolSha256,
  files: {
    metrics: { path: 'metrics.json', sha256: sha256(readFileSync(reportPath)) },
    qa: { path: 'qa/000-coupled-camera-path-sheets-and-holonomy.png', sha256: sha256(readFileSync(qaPath)) },
  },
  qaInterpretation: {
    rows: grids.map((grid) => grid.label),
    panels: ['front conditional quantile medoid class', 'middle medoid class', 'back medoid class', 'red = nontrivial local sheet-transport holonomy'],
  },
};
writeFileSync(resolve(outputRoot, 'index.json'), `${JSON.stringify(index, null, 2)}\n`);
console.error(`[coupled-quantile-sheet] ${report.decision}`);
process.stdout.write(`${reportPath}\n`);
