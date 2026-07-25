/**
 * Offline fixed-four-view gate for the existing GCRP/v4 Calamagrostis field.
 *
 * This deliberately tests the strongest cheap interpretation of the stored
 * data: each of the four angular neighbours contributes one complete nearest
 * event, recentered at the botanical middle plane.  Its canonical hit height
 * is placed on the exact live ray.  The source mesh/BVH is truth only; none of
 * the traversal in this file is proposed for runtime.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import {
  TriangleBvh,
  decodeOwnedProfileGeometry,
  type CensusVec3,
  type DecodedOwnedProfileGeometry,
} from '../../ExteriorClosureCensus';
import { periodicNearestSuccessor, type OriginAwareTruthHit } from '../../OriginAwareRayTruth';

const HEADER_BYTES = 128;
const SLICE_BYTES = 64;
const TEXEL_BYTES = 8;
const OWNER_BYTES = 4;
const MISS_OWNER = 0xffff_ffff;
const SOURCE = resolve('src/assets/groundcover/calamagrostis-canescens.gcrp');
const EXPECTED_SOURCE_SHA256 = '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const PHASE_GRID = 8;
const PHASE_OFFSET_X = 0.3819660112501051;
const PHASE_OFFSET_Z = 0.6180339887498949;
const EXTERIOR_ELEVATIONS = [5, 18, 45, 82.5] as const;
const EXTERIOR_AZIMUTH_CELLS = [0, 4, 8, 12] as const;
const MOTION_ELEVATIONS = [5, 18, 45] as const;
const MOTION_STEPS = 129;
const MOTION_STEP_METRES = 0.002;
const INSIDE_LIMIT = 192;
const EPSILON = 1e-8;

interface Slice {
  direction: CensusVec3;
  depthMin: number;
  depthMax: number;
}

interface Field {
  bytes: Uint8Array;
  view: DataView;
  geometry: DecodedOwnedProfileGeometry;
  storedTileWidth: number;
  storedTileHeight: number;
  interiorTileWidth: number;
  interiorTileHeight: number;
  atlasColumns: number;
  gutter: number;
  geometryOffset: number;
  ownerOffset: number;
  slices: readonly Slice[];
  azimuthCount: number;
  elevationCount: number;
  elevations: readonly number[];
  referenceY: number;
}

interface RecordEvent {
  hit: boolean;
  weight: number;
  hitY: number;
  liveT: number;
  triangleId: number | null;
  copyX: number;
  copyZ: number;
  onAttachedTriangle: boolean;
}

type MethodName = 'height_blend' | 'front_complete' | 'weighted_median_complete'
  | 'maximum_weight_complete' | 'oracle_best_of_four';

interface Prediction {
  hit: boolean;
  t?: number;
  event?: RecordEvent;
  fabricatedDepth?: boolean;
  mixedOwners?: boolean;
  mixedHitMiss?: boolean;
}

interface MutableMetrics {
  rays: number;
  truthHits: number;
  predictedHits: number;
  truePositive: number;
  falsePositive: number;
  falseNegative: number;
  depthErrors: number[];
  exactOwners: number;
  attachedSurface: number;
  fabricatedDepth: number;
  mixedOwners: number;
  mixedHitMiss: number;
}

function positiveModulo(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus;
}

function fract(value: number): number {
  return value - Math.floor(value);
}

function quantile(values: readonly number[], q: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(q * sorted.length) - 1))]!;
}

function direction(azimuth: number, elevation: number): CensusVec3 {
  const horizontal = Math.cos(elevation);
  return [horizontal * Math.cos(azimuth), -Math.sin(elevation), horizontal * Math.sin(azimuth)];
}

function decodeField(bytes: Uint8Array): Field {
  const geometry = decodeOwnedProfileGeometry(bytes);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const storedTileWidth = view.getUint32(12, true);
  const storedTileHeight = view.getUint32(16, true);
  const atlasColumns = view.getUint32(20, true);
  const sliceCount = view.getUint32(28, true);
  const geometryOffset = view.getUint32(36, true);
  const interiorTileWidth = view.getUint32(44, true);
  const interiorTileHeight = view.getUint32(48, true);
  const gutter = view.getUint32(52, true);
  const ownerOffset = view.getUint32(80, true);
  if (geometryOffset !== HEADER_BYTES + sliceCount * SLICE_BYTES) throw new Error('noncanonical GCRP/v4 header');
  const atlasTexels = storedTileWidth * storedTileHeight * atlasColumns * view.getUint32(24, true);
  if (ownerOffset !== geometryOffset + atlasTexels * TEXEL_BYTES) throw new Error('noncanonical GCRP/v4 owner table');
  const slices: Slice[] = [];
  for (let index = 0; index < sliceCount; index++) {
    const offset = HEADER_BYTES + index * SLICE_BYTES;
    slices.push({
      direction: [
        view.getFloat32(offset, true),
        view.getFloat32(offset + 4, true),
        view.getFloat32(offset + 8, true),
      ],
      depthMin: view.getFloat32(offset + 12, true),
      depthMax: view.getFloat32(offset + 16, true),
    });
  }
  let elevationCount = 1;
  const firstAzimuth = Math.atan2(slices[0]!.direction[2], slices[0]!.direction[0]);
  while (
    elevationCount < slices.length
    && Math.abs(Math.atan2(
      slices[elevationCount]!.direction[2],
      slices[elevationCount]!.direction[0],
    ) - firstAzimuth) < 2e-4
  ) elevationCount++;
  const azimuthCount = sliceCount / elevationCount;
  if (!Number.isInteger(azimuthCount) || elevationCount < 2) throw new Error('irregular direction lattice');
  return {
    bytes,
    view,
    geometry,
    storedTileWidth,
    storedTileHeight,
    interiorTileWidth,
    interiorTileHeight,
    atlasColumns,
    gutter,
    geometryOffset,
    ownerOffset,
    slices,
    azimuthCount,
    elevationCount,
    elevations: slices.slice(0, elevationCount).map((slice) => Math.asin(-slice.direction[1])),
    referenceY: (geometry.bounds.min[1] + geometry.bounds.max[1]) * 0.5,
  };
}

function atlasRecord(
  field: Field,
  azimuthIndex: number,
  elevationIndex: number,
  phaseX: number,
  phaseZ: number,
): Omit<RecordEvent, 'weight' | 'liveT' | 'onAttachedTriangle'> & { sourceX: number; sourceZ: number } {
  const sliceIndex = positiveModulo(azimuthIndex, field.azimuthCount) * field.elevationCount + elevationIndex;
  const slice = field.slices[sliceIndex]!;
  const u = fract((phaseX - field.geometry.tileOriginX) / field.geometry.tileSizeX);
  const v = 1 - fract((phaseZ - field.geometry.tileOriginZ) / field.geometry.tileSizeZ);
  const tx = positiveModulo(Math.round(u * field.interiorTileWidth - 0.5), field.interiorTileWidth);
  const ty = positiveModulo(Math.round(v * field.interiorTileHeight - 0.5), field.interiorTileHeight);
  const column = sliceIndex % field.atlasColumns;
  const row = Math.floor(sliceIndex / field.atlasColumns);
  const atlasWidth = field.storedTileWidth * field.atlasColumns;
  const ax = column * field.storedTileWidth + field.gutter + tx;
  const ay = row * field.storedTileHeight + field.gutter + ty;
  const texel = ay * atlasWidth + ax;
  const payload = field.geometryOffset + texel * TEXEL_BYTES;
  const depthU16 = field.view.getUint16(payload, true);
  const coverageU16 = field.view.getUint16(payload + 6, true);
  const owner = field.view.getUint32(field.ownerOffset + texel * OWNER_BYTES, true);
  const hit = coverageU16 > 32767 && depthU16 < 65535 && owner !== MISS_OWNER;
  const t = slice.depthMin + depthU16 / 65535 * (slice.depthMax - slice.depthMin);
  return {
    hit,
    hitY: field.geometry.topH + slice.direction[1] * t,
    triangleId: hit ? owner & 0x3f_ffff : null,
    copyX: hit ? ((owner >>> 22) & 0x1f) - 16 : 0,
    copyZ: hit ? ((owner >>> 27) & 0x1f) - 16 : 0,
    sourceX: field.geometry.tileOriginX + (tx + 0.5) / field.interiorTileWidth * field.geometry.tileSizeX,
    sourceZ: field.geometry.tileOriginZ
      + (1 - (ty + 0.5) / field.interiorTileHeight) * field.geometry.tileSizeZ,
  };
}

function pointOnTriangle(
  geometry: DecodedOwnedProfileGeometry,
  triangleId: number,
  copyX: number,
  copyZ: number,
  point: CensusVec3,
): boolean {
  if (triangleId < 0 || triangleId >= geometry.triangleCount) return false;
  const tri = triangleId * 3;
  const ia = geometry.triangles[tri]! * 3;
  const ib = geometry.triangles[tri + 1]! * 3;
  const ic = geometry.triangles[tri + 2]! * 3;
  const offsetX = copyX * geometry.tileSizeX;
  const offsetZ = copyZ * geometry.tileSizeZ;
  const a: CensusVec3 = [geometry.positions[ia]! + offsetX, geometry.positions[ia + 1]!, geometry.positions[ia + 2]! + offsetZ];
  const b: CensusVec3 = [geometry.positions[ib]! + offsetX, geometry.positions[ib + 1]!, geometry.positions[ib + 2]! + offsetZ];
  const c: CensusVec3 = [geometry.positions[ic]! + offsetX, geometry.positions[ic + 1]!, geometry.positions[ic + 2]! + offsetZ];
  const e0: CensusVec3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const e1: CensusVec3 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const q: CensusVec3 = [point[0] - a[0], point[1] - a[1], point[2] - a[2]];
  const nx = e0[1] * e1[2] - e0[2] * e1[1];
  const ny = e0[2] * e1[0] - e0[0] * e1[2];
  const nz = e0[0] * e1[1] - e0[1] * e1[0];
  const n2 = nx * nx + ny * ny + nz * nz;
  if (!(n2 > 1e-24)) return false;
  const planeDistance = Math.abs(nx * q[0] + ny * q[1] + nz * q[2]) / Math.sqrt(n2);
  if (planeDistance > 2e-3) return false;
  const d00 = e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2];
  const d01 = e0[0] * e1[0] + e0[1] * e1[1] + e0[2] * e1[2];
  const d11 = e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2];
  const d20 = q[0] * e0[0] + q[1] * e0[1] + q[2] * e0[2];
  const d21 = q[0] * e1[0] + q[1] * e1[1] + q[2] * e1[2];
  const denominator = d00 * d11 - d01 * d01;
  if (!(Math.abs(denominator) > 1e-24)) return false;
  const u = (d11 * d20 - d01 * d21) / denominator;
  const v = (d00 * d21 - d01 * d20) / denominator;
  return u >= -2e-3 && v >= -2e-3 && u + v <= 1 + 2e-3;
}

function directionBracket(field: Field, live: CensusVec3): {
  azimuth0: number; azimuth1: number; elevation0: number; elevation1: number; weights: readonly number[];
} {
  const azimuthTurns = positiveModulo(Math.atan2(live[2], live[0]), Math.PI * 2)
    / (Math.PI * 2) * field.azimuthCount;
  const azimuth0 = Math.floor(azimuthTurns);
  const azimuthMix = azimuthTurns - azimuth0;
  const elevation = Math.asin(-live[1]);
  let elevation0 = field.elevationCount - 2;
  for (let index = 0; index < field.elevationCount - 1; index++) {
    if (elevation < field.elevations[index + 1]!) {
      elevation0 = index;
      break;
    }
  }
  const elevation1 = elevation0 + 1;
  const elevationMix = Math.max(0, Math.min(1,
    (elevation - field.elevations[elevation0]!)
      / (field.elevations[elevation1]! - field.elevations[elevation0]!),
  ));
  return {
    azimuth0,
    azimuth1: (azimuth0 + 1) % field.azimuthCount,
    elevation0,
    elevation1,
    weights: [
      (1 - azimuthMix) * (1 - elevationMix),
      azimuthMix * (1 - elevationMix),
      (1 - azimuthMix) * elevationMix,
      azimuthMix * elevationMix,
    ],
  };
}

function fourEvents(field: Field, top: CensusVec3, live: CensusVec3): RecordEvent[] {
  if (!(live[1] < -1e-10)) return [];
  const bracket = directionBracket(field, live);
  const addresses = [
    [bracket.azimuth0, bracket.elevation0],
    [bracket.azimuth1, bracket.elevation0],
    [bracket.azimuth0, bracket.elevation1],
    [bracket.azimuth1, bracket.elevation1],
  ] as const;
  const fromTopToReference = (field.referenceY - field.geometry.topH) / live[1];
  const reference: CensusVec3 = [
    top[0] + live[0] * fromTopToReference,
    field.referenceY,
    top[2] + live[2] * fromTopToReference,
  ];
  return addresses.map(([azimuthIndex, elevationIndex], index) => {
    const canonical = field.slices[azimuthIndex * field.elevationCount + elevationIndex]!.direction;
    const canonicalToTop = (field.referenceY - field.geometry.topH) / canonical[1];
    const canonicalTopX = reference[0] - canonical[0] * canonicalToTop;
    const canonicalTopZ = reference[2] - canonical[2] * canonicalToTop;
    const record = atlasRecord(field, azimuthIndex, elevationIndex, canonicalTopX, canonicalTopZ);
    const tileShiftX = Math.round((canonicalTopX - record.sourceX) / field.geometry.tileSizeX);
    const tileShiftZ = Math.round((canonicalTopZ - record.sourceZ) / field.geometry.tileSizeZ);
    const liveT = record.hit ? (record.hitY - field.geometry.topH) / live[1] : Number.POSITIVE_INFINITY;
    const copyX = record.copyX + tileShiftX;
    const copyZ = record.copyZ + tileShiftZ;
    const point: CensusVec3 = [top[0] + live[0] * liveT, record.hitY, top[2] + live[2] * liveT];
    return {
      hit: record.hit,
      weight: bracket.weights[index]!,
      hitY: record.hitY,
      liveT,
      triangleId: record.triangleId,
      copyX,
      copyZ,
      onAttachedTriangle: record.hit && record.triangleId !== null
        ? pointOnTriangle(field.geometry, record.triangleId, copyX, copyZ, point)
        : false,
    };
  });
}

function ownerKey(event: RecordEvent): string | null {
  return event.triangleId === null ? null : `${event.triangleId}:${event.copyX}:${event.copyZ}`;
}

function predict(method: MethodName, events: readonly RecordEvent[], truth: OriginAwareTruthHit | null): Prediction {
  if (events.length === 0) return { hit: false };
  if (method === 'maximum_weight_complete') {
    const event = events.reduce((best, candidate) => candidate.weight > best.weight ? candidate : best);
    return event.hit && event.liveT > EPSILON ? { hit: true, t: event.liveT, event } : { hit: false };
  }
  if (method === 'oracle_best_of_four') {
    if (!truth) return events.some((event) => !event.hit)
      ? { hit: false }
      : { hit: true, t: events[0]!.liveT, event: events[0] };
    const hits = events.filter((event) => event.hit && event.liveT > EPSILON);
    if (hits.length === 0) return { hit: false };
    const owner = `${truth.triangleId}:${truth.copyX}:${truth.copyZ}`;
    const exact = hits.filter((event) => ownerKey(event) === owner);
    const pool = exact.length > 0 ? exact : hits;
    const event = pool.reduce((best, candidate) =>
      Math.abs(candidate.liveT - truth.t) < Math.abs(best.liveT - truth.t) ? candidate : best);
    return { hit: true, t: event.liveT, event };
  }
  const hits = events.filter((event) => event.hit && event.weight > 0 && event.liveT > EPSILON);
  const hitWeight = hits.reduce((sum, event) => sum + event.weight, 0);
  if (!(hitWeight > 0.02)) return { hit: false };
  if (method === 'front_complete') {
    const event = hits.reduce((best, candidate) => candidate.liveT < best.liveT ? candidate : best);
    return { hit: true, t: event.liveT, event };
  }
  if (method === 'weighted_median_complete') {
    const sorted = [...hits].sort((a, b) => a.liveT - b.liveT);
    let cumulative = 0;
    let event = sorted.at(-1)!;
    for (const candidate of sorted) {
      cumulative += candidate.weight;
      if (cumulative >= hitWeight * 0.5) {
        event = candidate;
        break;
      }
    }
    return { hit: true, t: event.liveT, event };
  }
  const t = hits.reduce((sum, event) => sum + event.weight * event.liveT, 0) / hitWeight;
  const distinctOwners = new Set(hits.map(ownerKey).filter((key): key is string => key !== null));
  return {
    hit: true,
    t,
    fabricatedDepth: hits.every((event) => Math.abs(event.liveT - t) > 1e-4),
    mixedOwners: distinctOwners.size > 1,
    mixedHitMiss: events.some((event) => event.weight > 0 && event.hit)
      && events.some((event) => event.weight > 0 && !event.hit),
  };
}

function emptyMetrics(): MutableMetrics {
  return {
    rays: 0, truthHits: 0, predictedHits: 0, truePositive: 0, falsePositive: 0,
    falseNegative: 0, depthErrors: [], exactOwners: 0, attachedSurface: 0,
    fabricatedDepth: 0, mixedOwners: 0, mixedHitMiss: 0,
  };
}

function accumulate(metrics: MutableMetrics, truth: OriginAwareTruthHit | null, prediction: Prediction): void {
  metrics.rays++;
  if (truth) metrics.truthHits++;
  if (prediction.hit) metrics.predictedHits++;
  if (truth && prediction.hit && prediction.t !== undefined) {
    metrics.truePositive++;
    metrics.depthErrors.push(Math.abs(prediction.t - truth.t));
    if (prediction.event && ownerKey(prediction.event) === `${truth.triangleId}:${truth.copyX}:${truth.copyZ}`) {
      metrics.exactOwners++;
    }
  } else if (!truth && prediction.hit) metrics.falsePositive++;
  else if (truth && !prediction.hit) metrics.falseNegative++;
  if (prediction.event?.onAttachedTriangle) metrics.attachedSurface++;
  if (prediction.fabricatedDepth) metrics.fabricatedDepth++;
  if (prediction.mixedOwners) metrics.mixedOwners++;
  if (prediction.mixedHitMiss) metrics.mixedHitMiss++;
}

function finalize(metrics: MutableMetrics): Record<string, number | null> {
  const union = metrics.truePositive + metrics.falsePositive + metrics.falseNegative;
  return {
    rays: metrics.rays,
    truthHits: metrics.truthHits,
    predictedHits: metrics.predictedHits,
    silhouetteIoU: metrics.truePositive / Math.max(1, union),
    precision: metrics.truePositive / Math.max(1, metrics.predictedHits),
    recall: metrics.truePositive / Math.max(1, metrics.truthHits),
    depthP50Metres: quantile(metrics.depthErrors, 0.5),
    depthP95Metres: quantile(metrics.depthErrors, 0.95),
    depthMaximumMetres: quantile(metrics.depthErrors, 1),
    exactOwnerAmongTruePositive: metrics.exactOwners / Math.max(1, metrics.truePositive),
    attachedTriangleAmongPredictedHits: metrics.attachedSurface / Math.max(1, metrics.predictedHits),
    fabricatedDepthAmongPredictedHits: metrics.fabricatedDepth / Math.max(1, metrics.predictedHits),
    mixedOwnersAmongPredictedHits: metrics.mixedOwners / Math.max(1, metrics.predictedHits),
    mixedHitMissAmongPredictedHits: metrics.mixedHitMiss / Math.max(1, metrics.predictedHits),
  };
}

function exteriorTruth(field: Field, bvh: TriangleBvh, top: CensusVec3, live: CensusVec3): OriginAwareTruthHit | null {
  const horizon = (field.geometry.topH - field.geometry.bounds.min[1]) / -live[1] + 1e-6;
  return periodicNearestSuccessor(field.geometry, bvh, top, live, horizon, EPSILON);
}

const sourceBytes = readFileSync(SOURCE);
const sourceSha256 = createHash('sha256').update(sourceBytes).digest('hex');
if (sourceSha256 !== EXPECTED_SOURCE_SHA256) throw new Error(`unexpected Calamagrostis source ${sourceSha256}`);
const field = decodeField(sourceBytes);
const bvh = TriangleBvh.build(field.geometry);
const methods: readonly MethodName[] = [
  'height_blend',
  'front_complete',
  'weighted_median_complete',
  'maximum_weight_complete',
  'oracle_best_of_four',
];

// Control: before judging any cross-direction reconstruction, prove that this
// evaluator decodes a canonical stored record, its periodic owner copy, and
// its quantized depth in the same coordinate system as the exact source mesh.
const canonicalControl = emptyMetrics();
for (const azimuthIndex of EXTERIOR_AZIMUTH_CELLS) {
  for (let elevationIndex = 0; elevationIndex < field.elevationCount; elevationIndex++) {
    const canonical = field.slices[azimuthIndex * field.elevationCount + elevationIndex]!.direction;
    for (let sampleZ = 0; sampleZ < PHASE_GRID; sampleZ++) {
      for (let sampleX = 0; sampleX < PHASE_GRID; sampleX++) {
        const texelX = Math.floor((sampleX + 0.5) / PHASE_GRID * field.interiorTileWidth);
        const texelY = Math.floor((sampleZ + 0.5) / PHASE_GRID * field.interiorTileHeight);
        const top: CensusVec3 = [
          field.geometry.tileOriginX + (texelX + 0.5) / field.interiorTileWidth * field.geometry.tileSizeX,
          field.geometry.topH,
          field.geometry.tileOriginZ
            + (1 - (texelY + 0.5) / field.interiorTileHeight) * field.geometry.tileSizeZ,
        ];
        const raw = atlasRecord(field, azimuthIndex, elevationIndex, top[0], top[2]);
        const liveT = raw.hit ? (raw.hitY - field.geometry.topH) / canonical[1] : Number.POSITIVE_INFINITY;
        const point: CensusVec3 = [
          top[0] + canonical[0] * liveT,
          raw.hitY,
          top[2] + canonical[2] * liveT,
        ];
        const event: RecordEvent = {
          hit: raw.hit,
          weight: 1,
          hitY: raw.hitY,
          liveT,
          triangleId: raw.triangleId,
          copyX: raw.copyX,
          copyZ: raw.copyZ,
          onAttachedTriangle: raw.hit && raw.triangleId !== null
            ? pointOnTriangle(field.geometry, raw.triangleId, raw.copyX, raw.copyZ, point)
            : false,
        };
        accumulate(
          canonicalControl,
          exteriorTruth(field, bvh, top, canonical),
          raw.hit ? { hit: true, t: liveT, event } : { hit: false },
        );
      }
    }
  }
}

const aggregate = Object.fromEntries(methods.map((name) => [name, emptyMetrics()])) as Record<MethodName, MutableMetrics>;
const byElevation: Record<string, Record<MethodName, MutableMetrics>> = {};
let truthOwnerAvailable = 0;
let exteriorTruthHits = 0;

for (const elevationDegrees of EXTERIOR_ELEVATIONS) {
  const elevationMetrics = Object.fromEntries(methods.map((name) => [name, emptyMetrics()])) as Record<MethodName, MutableMetrics>;
  byElevation[String(elevationDegrees)] = elevationMetrics;
  const elevation = elevationDegrees * Math.PI / 180;
  for (const azimuthCell of EXTERIOR_AZIMUTH_CELLS) {
    const azimuth = (azimuthCell + 0.5) * Math.PI * 2 / field.azimuthCount;
    const live = direction(azimuth, elevation);
    for (let iz = 0; iz < PHASE_GRID; iz++) {
      for (let ix = 0; ix < PHASE_GRID; ix++) {
        const top: CensusVec3 = [
          field.geometry.tileOriginX + fract((ix + PHASE_OFFSET_X) / PHASE_GRID) * field.geometry.tileSizeX,
          field.geometry.topH,
          field.geometry.tileOriginZ + fract((iz + PHASE_OFFSET_Z) / PHASE_GRID) * field.geometry.tileSizeZ,
        ];
        const truth = exteriorTruth(field, bvh, top, live);
        const events = fourEvents(field, top, live);
        if (truth) {
          exteriorTruthHits++;
          const key = `${truth.triangleId}:${truth.copyX}:${truth.copyZ}`;
          if (events.some((event) => ownerKey(event) === key)) truthOwnerAvailable++;
        }
        for (const method of methods) {
          const prediction = predict(method, events, truth);
          accumulate(aggregate[method], truth, prediction);
          accumulate(elevationMetrics[method], truth, prediction);
        }
      }
    }
  }
}

const motion: Record<string, unknown> = {};
for (const elevationDegrees of MOTION_ELEVATIONS) {
  const live = direction(Math.PI / field.azimuthCount, elevationDegrees * Math.PI / 180);
  const deltaErrors = Object.fromEntries(
    methods.map((name) => [name, [] as number[]]),
  ) as unknown as Record<MethodName, number[]>;
  const falseToggles: Record<MethodName, number> = Object.fromEntries(methods.map((name) => [name, 0])) as Record<MethodName, number>;
  let previousTruth: OriginAwareTruthHit | null = null;
  const previousPrediction: Partial<Record<MethodName, Prediction>> = {};
  for (let step = 0; step < MOTION_STEPS; step++) {
    const top: CensusVec3 = [
      field.geometry.tileOriginX + 0.127 + step * MOTION_STEP_METRES,
      field.geometry.topH,
      field.geometry.tileOriginZ + 0.311,
    ];
    const truth = exteriorTruth(field, bvh, top, live);
    const events = fourEvents(field, top, live);
    for (const method of methods) {
      const current = predict(method, events, truth);
      const previous = previousPrediction[method];
      if (previous) {
        const truthToggle = Boolean(previousTruth) !== Boolean(truth);
        const predictionToggle = previous.hit !== current.hit;
        if (predictionToggle !== truthToggle) falseToggles[method]++;
        if (
          previousTruth && truth && previous.hit && current.hit
          && previous.t !== undefined && current.t !== undefined
        ) {
          deltaErrors[method].push(Math.abs(
            (current.t - previous.t) - (truth.t - previousTruth.t),
          ));
        }
      }
      previousPrediction[method] = current;
    }
    previousTruth = truth;
  }
  motion[String(elevationDegrees)] = Object.fromEntries(methods.map((method) => [method, {
    twoMillimetreDeltaErrorP95Metres: quantile(deltaErrors[method], 0.95),
    twoMillimetreDeltaErrorMaximumMetres: quantile(deltaErrors[method], 1),
    falseCoverageToggleSteps: falseToggles[method],
  }]));
}

const insideMetrics = Object.fromEntries(methods.map((name) => [name, emptyMetrics()])) as Record<MethodName, MutableMetrics>;
let insideSeeds = 0;
let successorTruthHits = 0;
let exhaustedFirstEventWithSuccessor = 0;
outer:
for (const elevationDegrees of [18, 45] as const) {
  const live = direction(Math.PI / field.azimuthCount, elevationDegrees * Math.PI / 180);
  for (let iz = 0; iz < PHASE_GRID; iz++) {
    for (let ix = 0; ix < PHASE_GRID; ix++) {
      const top: CensusVec3 = [
        field.geometry.tileOriginX + fract((ix + PHASE_OFFSET_X) / PHASE_GRID) * field.geometry.tileSizeX,
        field.geometry.topH,
        field.geometry.tileOriginZ + fract((iz + PHASE_OFFSET_Z) / PHASE_GRID) * field.geometry.tileSizeZ,
      ];
      const first = exteriorTruth(field, bvh, top, live);
      if (!first) continue;
      const advance = first.t + 0.002;
      const inside: CensusVec3 = [top[0] + live[0] * advance, top[1] + live[1] * advance, top[2] + live[2] * advance];
      if (inside[1] <= field.geometry.bounds.min[1] + 0.005) continue;
      const successor = periodicNearestSuccessor(field.geometry, bvh, inside, live, 8, EPSILON);
      insideSeeds++;
      if (successor) successorTruthHits++;
      const topEvents = fourEvents(field, top, live).map((event) => ({
        ...event,
        liveT: event.liveT - advance,
      }));
      if (successor && !topEvents.some((event) => event.hit && event.liveT > EPSILON)) {
        exhaustedFirstEventWithSuccessor++;
      }
      for (const method of methods) accumulate(insideMetrics[method], successor, predict(method, topEvents, successor));
      if (insideSeeds >= INSIDE_LIMIT) break outer;
    }
  }
}

const horizontalMetrics = emptyMetrics();
const horizontal = direction(Math.PI / field.azimuthCount, 0);
for (let iz = 0; iz < PHASE_GRID; iz++) {
  for (let ix = 0; ix < PHASE_GRID; ix++) {
    const origin: CensusVec3 = [
      field.geometry.tileOriginX + fract((ix + PHASE_OFFSET_X) / PHASE_GRID) * field.geometry.tileSizeX,
      field.referenceY,
      field.geometry.tileOriginZ + fract((iz + PHASE_OFFSET_Z) / PHASE_GRID) * field.geometry.tileSizeZ,
    ];
    const truth = periodicNearestSuccessor(field.geometry, bvh, origin, horizontal, 8, EPSILON);
    // A top-entry height record has no finite horizontal top phase. Every
    // prospective four-view-height decoder is therefore undefined here.
    accumulate(horizontalMetrics, truth, { hit: false });
  }
}

const report = {
  schema: 'laas-groundcover-four-view-height-gate/v1',
  source: {
    file: SOURCE,
    bytes: sourceBytes.byteLength,
    sha256: sourceSha256,
    vertices: field.geometry.vertexCount,
    triangles: field.geometry.triangleCount,
    tileMetres: [field.geometry.tileSizeX, field.geometry.tileSizeZ],
    yBand: [field.geometry.bounds.min[1], field.geometry.bounds.max[1]],
    referenceY: field.referenceY,
  },
  frozenProspectiveRuntime: {
    recordReads: 4,
    bytesPerRecord: 8,
    requestedRecordBytesPerQuery: 32,
    winnerColorReads: 1,
    loops: 0,
    marches: 0,
    geometryCandidates: 0,
    extraPasses: 0,
    extraBindings: 0,
    selectorUpperBound: 'one fixed four-input sorting network (five compare/select stages) plus one live vertical-speed division',
    note: 'The oracle selector is not implementable; it exists only as an upper bound on these four stored events.',
  },
  exterior: {
    canonicalDecodeControl: finalize(canonicalControl),
    phaseGrid: PHASE_GRID,
    azimuthHalfBins: EXTERIOR_AZIMUTH_CELLS,
    elevationsDegrees: EXTERIOR_ELEVATIONS,
    exactTruthOwnerPresentInFour: truthOwnerAvailable / Math.max(1, exteriorTruthHits),
    methods: Object.fromEntries(methods.map((method) => [method, finalize(aggregate[method])])),
    byElevation: Object.fromEntries(Object.entries(byElevation).map(([elevation, values]) => [
      elevation,
      Object.fromEntries(methods.map((method) => [method, finalize(values[method])])),
    ])),
  },
  motion: {
    stepMetres: MOTION_STEP_METRES,
    steps: MOTION_STEPS,
    results: motion,
  },
  nearInside: {
    construction: 'origin moved 2 mm beyond an exact exterior first hit on the same oriented line',
    seeds: insideSeeds,
    successorTruthHits,
    exhaustedAllFourForwardFirstEventsWhileSuccessorExists: exhaustedFirstEventWithSuccessor,
    methods: Object.fromEntries(methods.map((method) => [method, finalize(insideMetrics[method])])),
  },
  exactHorizontalInside: {
    horizonMetres: 8,
    metrics: finalize(horizontalMetrics),
    reason: 'dy=0 has no finite top-plane phase or t=(H-yh)/(-dy); a four-view top-entry height record cannot encode pointed-line successor phase.',
  },
  acceptance: {
    silhouetteIoUMinimum: 0.95,
    depthP95MaximumMetres: 0.02,
    exactOwnerMinimum: 0.95,
    attachedTriangleMinimum: 0.95,
    motionDeltaP95MaximumMetres: 0.02,
    insideSuccessorRecallMinimum: 0.95,
    ownerBlendAllowed: false,
  },
  conclusion: 'reject: no four-view stored-height reconstruction is an implementation-ready geometric event field',
};

const configHash = createHash('sha256').update(JSON.stringify({
  sourceSha256,
  phaseGrid: PHASE_GRID,
  exteriorElevations: EXTERIOR_ELEVATIONS,
  azimuthCells: EXTERIOR_AZIMUTH_CELLS,
  motionSteps: MOTION_STEPS,
  motionStepMetres: MOTION_STEP_METRES,
  insideLimit: INSIDE_LIMIT,
})).digest('hex').slice(0, 16);
const output = resolve(
  'data/work/groundcover-four-view-height-gate',
  sourceSha256.slice(0, 16),
  configHash,
  'report.json',
);
mkdirSync(dirname(output), { recursive: true });
writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
process.stdout.write(`${output}\n`);
