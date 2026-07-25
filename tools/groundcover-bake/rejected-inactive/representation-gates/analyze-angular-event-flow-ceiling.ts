/**
 * Actual-source oracle ceiling for angular event correspondence.
 *
 * The test gives a hypothetical novel-view codec the exact target surface
 * point for free, projects that point onto neighbouring oriented lines, and
 * asks whether it is still their first visible event.  Failure is therefore a
 * hard disocclusion failure, not a flow-estimation or interpolation failure.
 * This is offline analysis only; it emits no runtime asset.
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
  type DecodedOwnedProfileGeometry,
} from '../../ExteriorClosureCensus';

const SOURCE = resolve('src/assets/groundcover/calamagrostis-canescens.gcrp');
const EXPECTED_SOURCE_SHA256 = '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const SPACINGS = [22.5, 11.25, 5, 2, 1, 0.5] as const;
const ELEVATIONS = [0.1, 1, 5, 15, 35, 75, 90] as const;
const AZIMUTHS = [0, 57] as const;
const PHASES = [[0.173, 0.291], [0.683, 0.817]] as const;
const GRID = 5;
const GRID_ANGLE_DEGREES = 0.06;
const HORIZONTAL_HORIZON = 155;
const CAMERA_HEIGHT_FRACTIONS = [0.1, 0.35, 0.65, 0.9] as const;
const EPSILON = 1e-7;
const EVENT_EPSILON = 2e-5;
const MAXIMUM_RESIDENT_BYTES = 51_121_152;
const STORED_PHASE = 258;
const COMPLETE_RECORD_BYTES = 8;
const FLOW_RECORD_BYTES = 4;

const PALETTE = [
  { rgb: [0.10, 0.27, 0.055], panicle: false },
  { rgb: [0.37, 0.51, 0.13], panicle: false },
  { rgb: [0.43, 0.53, 0.20], panicle: false },
  { rgb: [0.31, 0.24, 0.075], panicle: false },
  { rgb: [0.48, 0.31, 0.24], panicle: true },
  { rgb: [0.57, 0.35, 0.39], panicle: true },
  { rgb: [0.88, 0.80, 0.70], panicle: true },
  { rgb: [0.42, 0.10, 0.34], panicle: true },
] as const;

interface PeriodicHit {
  triangleId: number;
  copyX: number;
  copyZ: number;
  t: number;
}

interface Accumulator {
  targets: number;
  panicleTargets: number;
  vegetativeTargets: number;
  any2Primitive: number;
  any2Event: number;
  any4Primitive: number;
  any4Event: number;
  panicleAny4Event: number;
  vegetativeAny4Event: number;
  maximumHoleWidthPixels: number;
  maximumHoleComponentPixels: number;
  grids: number;
}

function sha256(bytes: Uint8Array | string): string {
  return createHash('sha256').update(bytes).digest('hex');
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

function normalizedDirection(baseAzimuth: number, baseElevation: number, gx: number, gy: number): CensusVec3 {
  return direction(
    baseAzimuth + gx * GRID_ANGLE_DEGREES,
    Math.max(0.0001, Math.min(90, baseElevation + gy * GRID_ANGLE_DEGREES)),
  );
}

/** Ordered periodic DDA with early exit once a found hit precedes the next
 * lattice boundary. This traverses source triangles only in the offline gate. */
function firstPeriodicHit(
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  origin: CensusVec3,
  ray: CensusVec3,
  requestedMaximumT: number,
): PeriodicHit | null {
  let maximumT = requestedMaximumT;
  if (Math.abs(ray[1]) > 1e-30) {
    const exitY = ray[1] < 0 ? geometry.bounds.min[1] : geometry.bounds.max[1];
    maximumT = Math.min(maximumT, Math.max(0, (exitY - origin[1]) / ray[1] + 1e-9));
  } else if (origin[1] < geometry.bounds.min[1] || origin[1] > geometry.bounds.max[1]) return null;
  if (!(maximumT > EPSILON)) return null;

  const visited = new Set<string>();
  let best: PeriodicHit | null = null;
  let bestT = maximumT;
  let cellX = Math.floor((origin[0] - geometry.tileOriginX) / geometry.tileSizeX);
  let cellZ = Math.floor((origin[2] - geometry.tileOriginZ) / geometry.tileSizeZ);
  const stepX = Math.sign(ray[0]);
  const stepZ = Math.sign(ray[2]);
  const boundaryX = geometry.tileOriginX + (cellX + (stepX > 0 ? 1 : 0)) * geometry.tileSizeX;
  const boundaryZ = geometry.tileOriginZ + (cellZ + (stepZ > 0 ? 1 : 0)) * geometry.tileSizeZ;
  let nextX = stepX === 0 ? Infinity : Math.max(0, (boundaryX - origin[0]) / ray[0]);
  let nextZ = stepZ === 0 ? Infinity : Math.max(0, (boundaryZ - origin[2]) / ray[2]);
  const deltaX = stepX === 0 ? Infinity : geometry.tileSizeX / Math.abs(ray[0]);
  const deltaZ = stepZ === 0 ? Infinity : geometry.tileSizeZ / Math.abs(ray[2]);
  const maximumCells = Math.ceil(
    Math.abs(ray[0]) * maximumT / geometry.tileSizeX
    + Math.abs(ray[2]) * maximumT / geometry.tileSizeZ,
  ) + 4;

  for (let cell = 0; cell < maximumCells; cell++) {
    const cellMinX = geometry.tileOriginX + cellX * geometry.tileSizeX;
    const cellMaxX = cellMinX + geometry.tileSizeX;
    const cellMinZ = geometry.tileOriginZ + cellZ * geometry.tileSizeZ;
    const cellMaxZ = cellMinZ + geometry.tileSizeZ;
    const minCopyX = Math.ceil((cellMinX - geometry.bounds.max[0]) / geometry.tileSizeX - 1e-11);
    const maxCopyX = Math.floor((cellMaxX - geometry.bounds.min[0]) / geometry.tileSizeX + 1e-11);
    const minCopyZ = Math.ceil((cellMinZ - geometry.bounds.max[2]) / geometry.tileSizeZ - 1e-11);
    const maxCopyZ = Math.floor((cellMaxZ - geometry.bounds.min[2]) / geometry.tileSizeZ + 1e-11);
    for (let copyZ = minCopyZ; copyZ <= maxCopyZ; copyZ++) {
      for (let copyX = minCopyX; copyX <= maxCopyX; copyX++) {
        const key = `${copyX}:${copyZ}`;
        if (visited.has(key)) continue;
        visited.add(key);
        const localOrigin: CensusVec3 = [
          origin[0] - copyX * geometry.tileSizeX,
          origin[1],
          origin[2] - copyZ * geometry.tileSizeZ,
        ];
        const hit = bvh.intersectNearest(localOrigin, ray, bestT);
        if (hit && hit.t > EPSILON && hit.t < bestT) {
          bestT = hit.t;
          best = { ...hit, copyX, copyZ };
        }
      }
    }
    const next = Math.min(nextX, nextZ);
    if (best && bestT <= next + 1e-10) break;
    if (next > maximumT + 1e-10 || next === Infinity) break;
    if (Math.abs(nextX - next) <= 1e-10) { cellX += stepX; nextX += deltaX; }
    if (Math.abs(nextZ - next) <= 1e-10) { cellZ += stepZ; nextZ += deltaZ; }
  }
  return best;
}

function isPanicle(bytes: Uint8Array, triangleId: number): boolean {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const vertexOffset = view.getUint32(84, true);
  const triangleOffset = view.getUint32(88, true);
  const triangle = triangleOffset + triangleId * 16;
  const rgb = [0, 0, 0];
  for (let corner = 0; corner < 3; corner++) {
    const vertex = view.getUint32(triangle + corner * 4, true);
    const record = vertexOffset + vertex * 16;
    rgb[0] += view.getUint16(record + 6, true) / (65535 * 3);
    rgb[1] += view.getUint16(record + 8, true) / (65535 * 3);
    rgb[2] += view.getUint16(record + 10, true) / (65535 * 3);
  }
  let best = 0;
  let distance = Infinity;
  for (let index = 0; index < PALETTE.length; index++) {
    const p = PALETTE[index]!.rgb;
    const d = (rgb[0]! - p[0]) ** 2 + (rgb[1]! - p[1]) ** 2 + (rgb[2]! - p[2]) ** 2;
    if (d < distance) { distance = d; best = index; }
  }
  return PALETTE[best]!.panicle;
}

function sourceVisibility(
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  point: CensusVec3,
  target: PeriodicHit,
  sourceDirection: CensusVec3,
): { primitive: boolean; event: boolean } {
  let origin: CensusVec3;
  let targetT: number;
  if (sourceDirection[1] < -1e-10) {
    const sourceY = geometry.topH + 1e-5;
    targetT = (sourceY - point[1]) / -sourceDirection[1];
    origin = [
      point[0] - sourceDirection[0] * targetT,
      sourceY,
      point[2] - sourceDirection[2] * targetT,
    ];
  } else {
    targetT = HORIZONTAL_HORIZON;
    origin = [
      point[0] - sourceDirection[0] * targetT,
      point[1] - sourceDirection[1] * targetT,
      point[2] - sourceDirection[2] * targetT,
    ];
  }
  const source = firstPeriodicHit(geometry, bvh, origin, sourceDirection, targetT + EVENT_EPSILON);
  if (!source) return { primitive: false, event: false };
  const event = Math.abs(source.t - targetT) <= EVENT_EPSILON;
  return {
    primitive: event && source.triangleId === target.triangleId
      && source.copyX === target.copyX && source.copyZ === target.copyZ,
    event,
  };
}

function neighbours(azimuth: number, elevation: number, spacing: number): {
  two: CensusVec3[];
  four: CensusVec3[];
} {
  const h = spacing / 2;
  const two = [direction(azimuth - h, elevation), direction(azimuth + h, elevation)];
  const low = Math.max(0, elevation - h);
  const high = Math.min(90, elevation + h);
  return {
    two,
    four: [
      direction(azimuth - h, low), direction(azimuth + h, low),
      direction(azimuth - h, high), direction(azimuth + h, high),
    ],
  };
}

function holeMetrics(mask: readonly boolean[]): { width: number; component: number } {
  let width = 0;
  for (let y = 0; y < GRID; y++) {
    let run = 0;
    for (let x = 0; x < GRID; x++) {
      if (!mask[y * GRID + x]) { run++; width = Math.max(width, run); } else run = 0;
    }
  }
  const seen = new Uint8Array(mask.length);
  let component = 0;
  for (let start = 0; start < mask.length; start++) {
    if (mask[start] || seen[start]) continue;
    const queue = [start]; seen[start] = 1; let count = 0;
    while (queue.length) {
      const at = queue.pop()!; count++;
      const x = at % GRID; const y = Math.floor(at / GRID);
      for (const next of [x > 0 ? at - 1 : -1, x + 1 < GRID ? at + 1 : -1, y > 0 ? at - GRID : -1, y + 1 < GRID ? at + GRID : -1]) {
        if (next >= 0 && !mask[next] && !seen[next]) { seen[next] = 1; queue.push(next); }
      }
    }
    component = Math.max(component, count);
  }
  return { width, component };
}

function emptyAccumulator(): Accumulator {
  return {
    targets: 0, panicleTargets: 0, vegetativeTargets: 0,
    any2Primitive: 0, any2Event: 0, any4Primitive: 0, any4Event: 0,
    panicleAny4Event: 0, vegetativeAny4Event: 0,
    maximumHoleWidthPixels: 0, maximumHoleComponentPixels: 0, grids: 0,
  };
}

const bytes = readFileSync(SOURCE);
const sourceSha256 = sha256(bytes);
if (sourceSha256 !== EXPECTED_SOURCE_SHA256) throw new Error(`unexpected source ${sourceSha256}`);
const geometry = decodeOwnedProfileGeometry(bytes);
console.error(`[angular-flow] building BVH for ${geometry.triangleCount.toLocaleString()} triangles`);
const bvh = TriangleBvh.build(geometry, 8);
const bySpacing = new Map<number, Accumulator>(SPACINGS.map((spacing) => [spacing, emptyAccumulator()]));
const bySpacingElevation = new Map<string, Accumulator>(
  SPACINGS.flatMap((spacing) => ELEVATIONS.map((elevation) => [
    `${spacing}|${elevation}`,
    emptyAccumulator(),
  ] as const)),
);
const images = new Map<number, number[]>();
for (const spacing of SPACINGS) images.set(spacing, []);

let targetRays = 0;
for (const elevation of ELEVATIONS) {
  for (const azimuth of AZIMUTHS) {
    for (const phase of PHASES) {
      const camera: CensusVec3 = [
        geometry.tileOriginX + phase[0] * geometry.tileSizeX,
        geometry.topH + 1e-5,
        geometry.tileOriginZ + phase[1] * geometry.tileSizeZ,
      ];
      const masks = new Map<number, boolean[]>(SPACINGS.map((spacing) => [spacing, []]));
      for (let gy = -(GRID >> 1); gy <= (GRID >> 1); gy++) {
        for (let gx = -(GRID >> 1); gx <= (GRID >> 1); gx++) {
          const targetDirection = normalizedDirection(azimuth, elevation, gx, gy);
          const maximumT = (camera[1] - geometry.bounds.min[1]) / -targetDirection[1];
          const target = firstPeriodicHit(geometry, bvh, camera, targetDirection, maximumT);
          targetRays++;
          for (const spacing of SPACINGS) {
            const acc = bySpacing.get(spacing)!;
            const elevationAcc = bySpacingElevation.get(`${spacing}|${elevation}`)!;
            if (!target) { masks.get(spacing)!.push(true); images.get(spacing)!.push(80); continue; }
            acc.targets++; elevationAcc.targets++;
            const panicle = isPanicle(bytes, target.triangleId);
            if (panicle) { acc.panicleTargets++; elevationAcc.panicleTargets++; }
            else { acc.vegetativeTargets++; elevationAcc.vegetativeTargets++; }
            const point: CensusVec3 = [
              camera[0] + targetDirection[0] * target.t,
              camera[1] + targetDirection[1] * target.t,
              camera[2] + targetDirection[2] * target.t,
            ];
            const angular = neighbours(
              azimuth + gx * GRID_ANGLE_DEGREES,
              Math.max(0, Math.min(90, elevation + gy * GRID_ANGLE_DEGREES)),
              spacing,
            );
            const two = angular.two.map((ray) => sourceVisibility(geometry, bvh, point, target, ray));
            const four = angular.four.map((ray) => sourceVisibility(geometry, bvh, point, target, ray));
            const any2Primitive = two.some((value) => value.primitive);
            const any2Event = two.some((value) => value.event);
            const any4Primitive = four.some((value) => value.primitive);
            const any4Event = four.some((value) => value.event);
            for (const targetAcc of [acc, elevationAcc]) {
              if (any2Primitive) targetAcc.any2Primitive++;
              if (any2Event) targetAcc.any2Event++;
              if (any4Primitive) targetAcc.any4Primitive++;
              if (any4Event) targetAcc.any4Event++;
              if (panicle && any4Event) targetAcc.panicleAny4Event++;
              if (!panicle && any4Event) targetAcc.vegetativeAny4Event++;
            }
            masks.get(spacing)!.push(any4Event);
            images.get(spacing)!.push(any4Event ? (panicle ? 220 : 150) : 0);
          }
        }
      }
      for (const spacing of SPACINGS) {
        const holes = holeMetrics(masks.get(spacing)!);
        const acc = bySpacing.get(spacing)!;
        acc.grids++;
        acc.maximumHoleWidthPixels = Math.max(acc.maximumHoleWidthPixels, holes.width);
        acc.maximumHoleComponentPixels = Math.max(acc.maximumHoleComponentPixels, holes.component);
        const elevationAcc = bySpacingElevation.get(`${spacing}|${elevation}`)!;
        elevationAcc.grids++;
        elevationAcc.maximumHoleWidthPixels = Math.max(elevationAcc.maximumHoleWidthPixels, holes.width);
        elevationAcc.maximumHoleComponentPixels = Math.max(elevationAcc.maximumHoleComponentPixels, holes.component);
      }
    }
  }
  console.error(`[angular-flow] completed elevation ${elevation} degrees`);
}

// Exact-horizontal camera-inside pointed-line successors and a one-centimetre
// forward camera shift on the same line.
const insideBySpacing = new Map<number, { targets: number; event4: number; stableAfter1cm: number }>(
  SPACINGS.map((spacing) => [spacing, { targets: 0, event4: 0, stableAfter1cm: 0 }]),
);
for (const azimuth of AZIMUTHS) {
  const targetDirection = direction(azimuth, 0);
  for (const height of CAMERA_HEIGHT_FRACTIONS) {
    for (const phase of PHASES) {
      const origin: CensusVec3 = [
        geometry.tileOriginX + phase[0] * geometry.tileSizeX,
        geometry.bounds.min[1] + height * (geometry.bounds.max[1] - geometry.bounds.min[1]),
        geometry.tileOriginZ + phase[1] * geometry.tileSizeZ,
      ];
      const target = firstPeriodicHit(geometry, bvh, origin, targetDirection, HORIZONTAL_HORIZON);
      if (!target) continue;
      const point: CensusVec3 = [origin[0] + targetDirection[0] * target.t, origin[1], origin[2] + targetDirection[2] * target.t];
      const shiftedOrigin: CensusVec3 = [origin[0] + targetDirection[0] * 0.01, origin[1], origin[2] + targetDirection[2] * 0.01];
      const shifted = firstPeriodicHit(geometry, bvh, shiftedOrigin, targetDirection, HORIZONTAL_HORIZON);
      const shiftedSame = !!shifted && shifted.triangleId === target.triangleId
        && shifted.copyX === target.copyX && shifted.copyZ === target.copyZ;
      for (const spacing of SPACINGS) {
        const acc = insideBySpacing.get(spacing)!; acc.targets++;
        if (shiftedSame) acc.stableAfter1cm++;
        const angular = neighbours(azimuth, 0, spacing);
        if (angular.four.some((ray) => sourceVisibility(geometry, bvh, point, target, ray).event)) acc.event4++;
      }
    }
  }
}

const results = SPACINGS.map((spacing) => {
  const value = bySpacing.get(spacing)!;
  const inside = insideBySpacing.get(spacing)!;
  const directions = Math.ceil(2 * Math.PI / (spacing * Math.PI / 180) ** 2);
  const birthFraction = value.targets > 0 ? 1 - value.any4Event / value.targets : 1;
  const rawBytes = directions * STORED_PHASE ** 2 * COMPLETE_RECORD_BYTES;
  const optimisticDeltaBytes = STORED_PHASE ** 2 * COMPLETE_RECORD_BYTES
    + directions * STORED_PHASE ** 2 * (FLOW_RECORD_BYTES + birthFraction * COMPLETE_RECORD_BYTES);
  return {
    spacingDegrees: spacing,
    sampledHemisphereDirections: directions,
    exterior: {
      targets: value.targets,
      anyTwoPrimitiveVisibility: value.any2Primitive / value.targets,
      anyTwoExactPointEventVisibility: value.any2Event / value.targets,
      anyFourPrimitiveVisibility: value.any4Primitive / value.targets,
      anyFourExactPointEventVisibility: value.any4Event / value.targets,
      panicleAnyFourEventVisibility: value.panicleTargets ? value.panicleAny4Event / value.panicleTargets : 1,
      vegetativeAnyFourEventVisibility: value.vegetativeTargets ? value.vegetativeAny4Event / value.vegetativeTargets : 1,
      maximumDisocclusionHoleWidthPixels: value.maximumHoleWidthPixels,
      maximumConnectedDisocclusionHolePixels: value.maximumHoleComponentPixels,
      byElevationDegrees: Object.fromEntries(ELEVATIONS.map((elevation) => {
        const sample = bySpacingElevation.get(`${spacing}|${elevation}`)!;
        return [String(elevation), {
          targets: sample.targets,
          anyFourExactPointEventVisibility: sample.targets ? sample.any4Event / sample.targets : null,
          panicleAnyFourEventVisibility: sample.panicleTargets ? sample.panicleAny4Event / sample.panicleTargets : null,
          vegetativeAnyFourEventVisibility: sample.vegetativeTargets ? sample.vegetativeAny4Event / sample.vegetativeTargets : null,
          maximumDisocclusionHoleWidthPixels: sample.maximumHoleWidthPixels,
          maximumConnectedDisocclusionHolePixels: sample.maximumHoleComponentPixels,
        }];
      })),
    },
    exactHorizontalCameraInside: {
      ...inside,
      anyFourEventVisibility: inside.targets ? inside.event4 / inside.targets : 0,
      sameLineSuccessorStableAfterOneCentimetre: inside.targets ? inside.stableAfter1cm / inside.targets : 0,
    },
    storage: {
      rawCompleteAtlasBytes: rawBytes,
      rawWithinCap: rawBytes <= MAXIMUM_RESIDENT_BYTES,
      optimisticFlowPlusResidualBirthBytes: Math.ceil(optimisticDeltaBytes),
      optimisticDeltaWithinCap: optimisticDeltaBytes <= MAXIMUM_RESIDENT_BYTES,
      fixedReads: '2 when covered (flow + complete source), 3 on residual birth; <=4',
      lowerBoundWarning: 'flow is granted a packed 2D inverse coordinate in 4 bytes; residual births store one 8-byte complete record; no topology/index metadata is charged',
    },
  };
});

const firstGate = results.find((entry) =>
  entry.exterior.anyFourExactPointEventVisibility >= 0.95
  && entry.exterior.panicleAnyFourEventVisibility >= 0.95
  && entry.exactHorizontalCameraInside.anyFourEventVisibility >= 0.95);
const decision = firstGate && firstGate.storage.optimisticDeltaWithinCap
  ? 'PROVISIONAL_PASS_REQUIRES_CODEC'
  : 'REJECT_AND_PARK';
const scriptPath = fileURLToPath(import.meta.url);
const report = {
  schema: 'laas-groundcover-angular-event-flow-oracle-ceiling/v1',
  decision,
  source: { file: SOURCE, bytes: bytes.byteLength, sha256: sourceSha256, vertices: geometry.vertexCount, triangles: geometry.triangleCount },
  method: {
    target: 'analytic f64 first hit on the decoded actual periodic Calamagrostis mesh',
    oracle: 'exact target world point, primitive, periodic copy and target direction are free; neighbouring source lines are forced through that exact point',
    identity: 'primitive is strict triangle+periodic-copy equality; event/chart is the same world surface point within 20 micrometres even when an edge tie changes triangle ID',
    neighbours: 'two azimuth-bracketing lines; four azimuth/elevation cell-corner lines',
    pinhole: `${GRID}x${GRID} structured camera ray grids, ${GRID_ANGLE_DEGREES} degree pixel spacing, two azimuths and two camera phases`,
    exclusions: 'none: exact horizontal camera-inside successors are a hard separate gate',
  },
  recipe: { spacingsDegrees: SPACINGS, elevationsDegrees: [0, ...ELEVATIONS], azimuthsDegrees: AZIMUTHS, phases: PHASES, targetRays, horizontalHorizonMetres: HORIZONTAL_HORIZON, maximumResidentBytes: MAXIMUM_RESIDENT_BYTES },
  ancestry: [
    'Lin and Shum, A Geometric Analysis of Light Field Rendering, IJCV 58(2), 2004: geometry-assisted neighbouring-ray reconstruction and occlusion limits',
    'Chai, Tong, Chan, and Shum, Plenoptic Sampling, SIGGRAPH 2000: spatial/angular sampling tradeoff',
    'LAAS-original: exact authored-event oracle visibility ceiling, periodic camera-inside successor gate, and complete-record residual-birth accounting on the accepted GCRP/v4 source',
  ],
  results,
  firstSpacingMeetingAllVisibilityGates: firstGate?.spacingDegrees ?? null,
  rejection: decision === 'REJECT_AND_PARK' ? [
    'an oracle correspondence cannot reconstruct an event which is disoccluded in every neighbouring source view',
    'a spacing is not admissible unless exterior, panicle, and exact-horizontal camera-inside visibility each reach 95%',
    'even an admissible spacing must fit the 51,121,152-byte cap with complete marks and residual births',
  ] : [],
  hashes: { scriptSha256: sha256(readFileSync(scriptPath)) },
};

const recipeHash = sha256(JSON.stringify(report.recipe));
const root = resolve('data/work/groundcover-angular-event-flow', sourceSha256.slice(0, 16), recipeHash.slice(0, 16));
mkdirSync(`${root}/qa`, { recursive: true });
const metricsPath = `${root}/metrics.json`;
writeFileSync(metricsPath, `${JSON.stringify(report, null, 2)}\n`);
const qaImages: Array<{ file: string; sha256: string; width: number; height: number; interpretation: string }> = [];
for (const spacing of SPACINGS) {
  const pixels = images.get(spacing)!;
  const width = GRID * PHASES.length * AZIMUTHS.length;
  const height = ELEVATIONS.length * GRID;
  const rgba = Buffer.alloc(width * height * 4);
  for (let i = 0; i < rgba.length / 4; i++) {
    const value = pixels[i] ?? 0;
    rgba[i * 4] = value === 0 ? 210 : value;
    rgba[i * 4 + 1] = value === 0 ? 45 : value;
    rgba[i * 4 + 2] = value === 0 ? 45 : value === 220 ? 190 : 75;
    rgba[i * 4 + 3] = 255;
  }
  const qaFile = `${String(spacing).replace('.', '_')}-degree-event-visibility.png`;
  await sharp(rgba, { raw: { width, height, channels: 4 } })
    .resize(width * 10, height * 10, { kernel: 'nearest' })
    .png()
    .toFile(`${root}/qa/${qaFile}`);
  qaImages.push({
    file: qaFile,
    sha256: sha256(readFileSync(`${root}/qa/${qaFile}`)),
    width: width * 10,
    height: height * 10,
    interpretation: `${spacing}-degree any-four oracle event visibility over all structured target grids`,
  });
}
const index = {
  schema: 'laas-groundcover-angular-event-flow-qa-index/v1',
  sourceSha256,
  recipeHash,
  metrics: { file: 'metrics.json', sha256: sha256(readFileSync(metricsPath)) },
  images: qaImages,
  interpretation: 'green=vegetative exact event visible in at least one of four neighbours; pale=panicle visible; red=oracle disocclusion hole; grey=target ray missed the accepted mesh',
};
writeFileSync(`${root}/qa/index.json`, `${JSON.stringify(index, null, 2)}\n`);
console.error(`[angular-flow] ${decision}; wrote ${root}`);
console.log(root);
