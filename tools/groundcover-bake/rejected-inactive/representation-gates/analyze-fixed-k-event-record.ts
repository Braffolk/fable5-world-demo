/**
 * Offline-only feasibility gate for a tiny fixed-K line-space event record.
 *
 * Each sampled oriented line is intersected analytically with every periodic
 * copy of the accepted Calamagrostis mesh. The exact ordered surface-event
 * list is then reduced to K=2/4/8 atoms. No runtime asset or shader is emitted.
 * Runtime loops, marches, traversals, and candidate lists are not proposed.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  deduplicateLineEvents,
  tracePeriodicLine,
  type DeduplicatedLineEvent,
} from '../../DeepLineEventCensus';
import {
  TriangleBvh,
  decodeOwnedProfileGeometry,
  type CensusVec3,
  type DecodedOwnedProfileGeometry,
} from '../../ExteriorClosureCensus';

const EXPECTED_SOURCE_SHA256 =
  '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const K_VALUES = [2, 4, 8] as const;
const PHASE_GRID = 4;
const PHASE_OFFSET = 0.3819660112501051;
const AZIMUTH_COUNT = 8;
const ELEVATIONS_DEGREES = [0, 0.1, 1, 5, 15, 35, 75, 90] as const;
const CAMERA_HEIGHT_FRACTIONS = [
  0.01, 0.055, 0.1, 0.175, 0.25, 0.375, 0.5,
  0.625, 0.75, 0.825, 0.9, 0.945, 0.99,
] as const;
const METRIC_QUERY_COUNT = 32;
const HORIZONTAL_HORIZON_METRES = 155;
const ORIGIN_EPSILON_METRES = 1e-9;
const EVENT_MERGE_EPSILON_METRES = 1e-8;
const BOUNDARY_MARGIN_METRES = 1e-6;
const MAXIMUM_RESIDENT_BYTES = 51_121_152;
const MISS = -1;

type Sense = 'horizontal' | 'downward' | 'upward';
type QueryFamily = 'anchor_first' | 'camera_height' | 'metric_uniform';
type Method = 'endpoint_cdf' | 'oracle_family_upper_bound';

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

interface Args { [key: string]: string | boolean }

interface ClassifiedEvent extends DeduplicatedLineEvent {
  classIndex: number;
  y: number;
}

interface DirectionDefinition {
  elevationDegrees: number;
  sense: Sense;
  azimuthDegrees: number;
  direction: CensusVec3;
}

interface LineDefinition {
  label: string;
  origin: CensusVec3;
  direction: CensusVec3;
  maximumT: number;
  cameraOrigins: number[];
  metricOrigins: number[];
}

interface MetricAccumulator {
  queries: number;
  truthHits: number;
  predictedHits: number;
  hitAgreement: number;
  exactEvent: number;
  classAgreement: number;
  panicleTruthHits: number;
  paniclePredictedPanicle: number;
  panicleExactEvent: number;
  stemTruthHits: number;
  stemPredictedStem: number;
  stemExactEvent: number;
  rayErrors: number[];
  heightErrors: number[];
}

interface TopologyAccumulator {
  lines: number;
  emptyLines: number;
  totalEvents: number;
  panicleEvents: number;
  stemEvents: number;
  linesExactlyRepresentable: number;
  retainedEvents: number;
  retainedPanicleEvents: number;
  retainedStemEvents: number;
  maximumEventsOnLine: number;
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

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) =>
    `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(',')}}`;
}

function sha256(bytes: Uint8Array | string): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function emptyMetrics(): MetricAccumulator {
  return {
    queries: 0,
    truthHits: 0,
    predictedHits: 0,
    hitAgreement: 0,
    exactEvent: 0,
    classAgreement: 0,
    panicleTruthHits: 0,
    paniclePredictedPanicle: 0,
    panicleExactEvent: 0,
    stemTruthHits: 0,
    stemPredictedStem: 0,
    stemExactEvent: 0,
    rayErrors: [],
    heightErrors: [],
  };
}

function emptyTopology(): TopologyAccumulator {
  return {
    lines: 0,
    emptyLines: 0,
    totalEvents: 0,
    panicleEvents: 0,
    stemEvents: 0,
    linesExactlyRepresentable: 0,
    retainedEvents: 0,
    retainedPanicleEvents: 0,
    retainedStemEvents: 0,
    maximumEventsOnLine: 0,
  };
}

function quantiles(values: number[]): Record<string, number | null> {
  if (values.length === 0) {
    return { p50: null, p90: null, p95: null, p99: null, maximum: null };
  }
  values.sort((a, b) => a - b);
  const at = (fraction: number): number => values[Math.floor((values.length - 1) * fraction)]!;
  return {
    p50: at(0.5),
    p90: at(0.9),
    p95: at(0.95),
    p99: at(0.99),
    maximum: values.at(-1)!,
  };
}

function finishMetrics(metrics: MetricAccumulator): Record<string, unknown> {
  const bothHits = metrics.rayErrors.length;
  return {
    queries: metrics.queries,
    truthHits: metrics.truthHits,
    predictedHits: metrics.predictedHits,
    hitMissAgreement: metrics.queries > 0 ? metrics.hitAgreement / metrics.queries : 1,
    hitRecall: metrics.truthHits > 0 ? bothHits / metrics.truthHits : 1,
    exactEventAgreementAmongTruthHits:
      metrics.truthHits > 0 ? metrics.exactEvent / metrics.truthHits : 1,
    botanicalClassAgreementAmongJointHits:
      bothHits > 0 ? metrics.classAgreement / bothHits : 1,
    panicle: {
      truthHits: metrics.panicleTruthHits,
      classRecall: metrics.panicleTruthHits > 0
        ? metrics.paniclePredictedPanicle / metrics.panicleTruthHits : 1,
      exactEventRecall: metrics.panicleTruthHits > 0
        ? metrics.panicleExactEvent / metrics.panicleTruthHits : 1,
    },
    stemAndLeaf: {
      truthHits: metrics.stemTruthHits,
      classRecall: metrics.stemTruthHits > 0
        ? metrics.stemPredictedStem / metrics.stemTruthHits : 1,
      exactEventRecall: metrics.stemTruthHits > 0
        ? metrics.stemExactEvent / metrics.stemTruthHits : 1,
    },
    successorRayDistanceAbsoluteErrorMetres: quantiles(metrics.rayErrors),
    successorWorldHeightAbsoluteErrorMetres: quantiles(metrics.heightErrors),
  };
}

function finishTopology(topology: TopologyAccumulator): Record<string, unknown> {
  return {
    lines: topology.lines,
    emptyLines: topology.emptyLines,
    totalEvents: topology.totalEvents,
    panicleEvents: topology.panicleEvents,
    stemAndLeafEvents: topology.stemEvents,
    maximumEventsOnOneSampledLine: topology.maximumEventsOnLine,
    linesExactlyRepresentable: topology.linesExactlyRepresentable,
    lineExactFraction: topology.lines > 0 ? topology.linesExactlyRepresentable / topology.lines : 1,
    retainedEvents: topology.retainedEvents,
    eventUniformOracleExactUpperBound: topology.totalEvents > 0
      ? topology.retainedEvents / topology.totalEvents : 1,
    endpointQuantilePanicleRetention: topology.panicleEvents > 0
      ? topology.retainedPanicleEvents / topology.panicleEvents : 1,
    endpointQuantileStemAndLeafRetention: topology.stemEvents > 0
      ? topology.retainedStemEvents / topology.stemEvents : 1,
  };
}

function lowerBoundEvent(events: readonly ClassifiedEvent[], originT: number): number {
  let low = 0;
  let high = events.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (events[middle]!.t <= originT + ORIGIN_EPSILON_METRES) low = middle + 1;
    else high = middle;
  }
  return low < events.length ? low : MISS;
}

function endpointQuantileIndices(eventCount: number, k: number): number[] {
  if (eventCount <= k) return Array.from({ length: eventCount }, (_, index) => index);
  return Array.from({ length: k }, (_, index) =>
    Math.round(index * (eventCount - 1) / (k - 1)));
}

function oracleIndices(
  events: readonly ClassifiedEvent[],
  origins: readonly number[],
  k: number,
): number[] {
  if (events.length <= k) return Array.from({ length: events.length }, (_, index) => index);
  const demand = new Uint32Array(events.length);
  // Any production record must retain the exterior first hit too. Give it one
  // forced slot before optimizing for the query family under test.
  demand[0] = 0xffff_ffff;
  for (const origin of origins) {
    const event = lowerBoundEvent(events, origin);
    if (event !== MISS && event !== 0) demand[event]++;
  }
  return Array.from({ length: events.length }, (_, index) => index)
    .sort((left, right) => demand[right]! - demand[left]! || left - right)
    .slice(0, k)
    .sort((left, right) => left - right);
}

function predictedIndex(
  events: readonly ClassifiedEvent[],
  retainedIndices: readonly number[],
  originT: number,
): number {
  for (const index of retainedIndices) {
    if (events[index]!.t > originT + ORIGIN_EPSILON_METRES) return index;
  }
  return MISS;
}

function addQuery(
  metrics: MetricAccumulator,
  events: readonly ClassifiedEvent[],
  retainedIndices: readonly number[],
  originT: number,
  verticalMagnitude: number,
): void {
  const truthIndex = lowerBoundEvent(events, originT);
  const predictionIndex = predictedIndex(events, retainedIndices, originT);
  const truthHit = truthIndex !== MISS;
  const predictionHit = predictionIndex !== MISS;
  metrics.queries++;
  if (truthHit) metrics.truthHits++;
  if (predictionHit) metrics.predictedHits++;
  if (truthHit === predictionHit) metrics.hitAgreement++;
  if (!truthHit) return;
  const truth = events[truthIndex]!;
  const truthPanicle = PALETTE[truth.classIndex]!.panicle;
  if (truthPanicle) metrics.panicleTruthHits++;
  else metrics.stemTruthHits++;
  if (!predictionHit) return;
  const prediction = events[predictionIndex]!;
  const predictionPanicle = PALETTE[prediction.classIndex]!.panicle;
  if (truthIndex === predictionIndex) {
    metrics.exactEvent++;
    if (truthPanicle) metrics.panicleExactEvent++;
    else metrics.stemExactEvent++;
  }
  if (truth.classIndex === prediction.classIndex) metrics.classAgreement++;
  if (truthPanicle && predictionPanicle) metrics.paniclePredictedPanicle++;
  if (!truthPanicle && !predictionPanicle) metrics.stemPredictedStem++;
  const rayError = Math.abs(prediction.t - truth.t);
  metrics.rayErrors.push(rayError);
  metrics.heightErrors.push(rayError * verticalMagnitude);
}

function paletteClassForTriangle(
  bytes: Uint8Array,
  triangleId: number,
): number {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const vertexOffset = view.getUint32(84, true);
  const triangleOffset = view.getUint32(88, true);
  const triangle = triangleOffset + triangleId * 16;
  const color = [0, 0, 0];
  for (let corner = 0; corner < 3; corner++) {
    const vertex = view.getUint32(triangle + corner * 4, true);
    const record = vertexOffset + vertex * 16;
    color[0] += view.getUint16(record + 6, true) / (65535 * 3);
    color[1] += view.getUint16(record + 8, true) / (65535 * 3);
    color[2] += view.getUint16(record + 10, true) / (65535 * 3);
  }
  let best = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < PALETTE.length; index++) {
    const reference = PALETTE[index]!.rgb;
    const distance = (color[0]! - reference[0]) ** 2
      + (color[1]! - reference[1]) ** 2
      + (color[2]! - reference[2]) ** 2;
    if (distance < bestDistance) {
      bestDistance = distance;
      best = index;
    }
  }
  return best;
}

function directions(): DirectionDefinition[] {
  const result: DirectionDefinition[] = [];
  for (const elevationDegrees of ELEVATIONS_DEGREES) {
    const senses: Sense[] = elevationDegrees === 0
      ? ['horizontal']
      : ['downward', 'upward'];
    const vertical = elevationDegrees === 90;
    const azimuths = vertical ? 1 : AZIMUTH_COUNT;
    for (const sense of senses) {
      for (let azimuth = 0; azimuth < azimuths; azimuth++) {
        const azimuthRadians = azimuth / azimuths * Math.PI * 2;
        const elevationRadians = elevationDegrees * Math.PI / 180;
        const horizontal = vertical ? 0 : Math.cos(elevationRadians);
        const y = sense === 'downward' ? -Math.sin(elevationRadians)
          : sense === 'upward' ? Math.sin(elevationRadians) : 0;
        result.push({
          elevationDegrees,
          sense,
          azimuthDegrees: vertical ? 0 : azimuth / azimuths * 360,
          direction: [
            horizontal * Math.cos(azimuthRadians),
            y,
            horizontal * Math.sin(azimuthRadians),
          ],
        });
      }
    }
  }
  return result;
}

function lineDefinition(
  geometry: DecodedOwnedProfileGeometry,
  definition: DirectionDefinition,
  phaseX: number,
  phaseZ: number,
  heightFraction: number | null,
): LineDefinition {
  const spanY = geometry.bounds.max[1] - geometry.bounds.min[1];
  if (definition.sense === 'horizontal') {
    const y = geometry.bounds.min[1] + heightFraction! * spanY;
    return {
      label: `${definition.elevationDegrees}:${definition.sense}`,
      origin: [phaseX, y, phaseZ],
      direction: definition.direction,
      maximumT: HORIZONTAL_HORIZON_METRES,
      cameraOrigins: [],
      metricOrigins: Array.from({ length: METRIC_QUERY_COUNT }, (_, index) =>
        (index + 0.5) / METRIC_QUERY_COUNT * HORIZONTAL_HORIZON_METRES),
    };
  }
  const vertical = Math.abs(definition.direction[1]);
  const maximumT = (spanY + BOUNDARY_MARGIN_METRES * 2) / vertical;
  const downward = definition.sense === 'downward';
  const originY = downward
    ? geometry.bounds.max[1] + BOUNDARY_MARGIN_METRES
    : geometry.bounds.min[1] - BOUNDARY_MARGIN_METRES;
  const cameraOrigins = CAMERA_HEIGHT_FRACTIONS.map((fraction) => {
    const y = geometry.bounds.min[1] + fraction * spanY;
    return downward ? (originY - y) / vertical : (y - originY) / vertical;
  });
  return {
    label: `${definition.elevationDegrees}:${definition.sense}`,
    origin: [phaseX, originY, phaseZ],
    direction: definition.direction,
    maximumT,
    cameraOrigins,
    metricOrigins: Array.from({ length: METRIC_QUERY_COUNT }, (_, index) =>
      (index + 0.5) / METRIC_QUERY_COUNT * maximumT),
  };
}

function storageTable(): Record<string, unknown> {
  const dense64 = 258 * 258 * 64;
  const dense81 = 258 * 258 * 81;
  const indexTwo64 = 130 * 258 * 64;
  const indexTwo81 = 130 * 258 * 81;
  return Object.fromEntries(K_VALUES.map((k) => {
    const completeBytesPerLine = k * 8;
    const geometryOnlyBytesPerLine = k * 2;
    return [String(k), {
      concreteCompleteAtom: {
        bytesPerAtom: 8,
        fields: 'depth u16 + oct-normal 2xu8 + RGB8 + 7-bit botanical/species class and valid bit',
        bytesPerLineRecord: completeBytesPerLine,
        fixedLogicalReads: Math.ceil(k / 2),
        decode: `${k} fixed compare/select lanes, statically unrolled; no loop or traversal`,
      },
      denseSameResolution: {
        current64DirectionsBytes: dense64 * completeBytesPerLine,
        denser81DirectionsBytes: dense81 * completeBytesPerLine,
        current64WithinBudget: dense64 * completeBytesPerLine <= MAXIMUM_RESIDENT_BYTES,
        denser81WithinBudget: dense81 * completeBytesPerLine <= MAXIMUM_RESIDENT_BYTES,
      },
      optimisticIndexTwoSpatialLattice: {
        warning: 'halves line-phase samples before any measured spatial/angular interpolation loss',
        current64DirectionsBytes: indexTwo64 * completeBytesPerLine,
        denser81DirectionsBytes: indexTwo81 * completeBytesPerLine,
        current64WithinBudget: indexTwo64 * completeBytesPerLine <= MAXIMUM_RESIDENT_BYTES,
        denser81WithinBudget: indexTwo81 * completeBytesPerLine <= MAXIMUM_RESIDENT_BYTES,
      },
      geometryOnlyInformationLowerBound: {
        warning: 'depth atoms only; cannot preserve panicle/stem colour, normal, or species identity',
        bytesPerLineRecord: geometryOnlyBytesPerLine,
        current64DirectionsBytes: dense64 * geometryOnlyBytesPerLine,
        denser81DirectionsBytes: dense81 * geometryOnlyBytesPerLine,
        current64WithinBudget: dense64 * geometryOnlyBytesPerLine <= MAXIMUM_RESIDENT_BYTES,
        denser81WithinBudget: dense81 * geometryOnlyBytesPerLine <= MAXIMUM_RESIDENT_BYTES,
      },
    }];
  }));
}

const args = parseArgs(process.argv.slice(2));
const source = resolve(typeof args.source === 'string'
  ? args.source : 'src/assets/groundcover/calamagrostis-canescens.gcrp');
const sourceBytes = readFileSync(source);
const sourceSha256 = sha256(sourceBytes);
if (sourceSha256 !== EXPECTED_SOURCE_SHA256) {
  throw new Error(`fixed-K gate expected ${EXPECTED_SOURCE_SHA256}, got ${sourceSha256}`);
}
const scriptPath = fileURLToPath(import.meta.url);
const scriptSha256 = sha256(readFileSync(scriptPath));
const recipe = {
  schema: 'laas-groundcover-fixed-k-event-gate/v1',
  sourceSha256,
  scriptSha256,
  kValues: K_VALUES,
  phaseGrid: PHASE_GRID,
  phaseOffset: PHASE_OFFSET,
  azimuthCount: AZIMUTH_COUNT,
  elevationsDegrees: ELEVATIONS_DEGREES,
  cameraHeightFractions: CAMERA_HEIGHT_FRACTIONS,
  metricQueryCount: METRIC_QUERY_COUNT,
  horizontalHorizonMetres: HORIZONTAL_HORIZON_METRES,
  originEpsilonMetres: ORIGIN_EPSILON_METRES,
  eventMergeEpsilonMetres: EVENT_MERGE_EPSILON_METRES,
  maximumResidentBytes: MAXIMUM_RESIDENT_BYTES,
  endpointCdf: 'retain exact first and last events; uniformly spaced inclusive event-rank quantiles in between',
  oracle: 'per line and query family, force the exterior first event and retain the K most demanded exact successors',
};
const recipeSha256 = sha256(canonicalJson(recipe));
const defaultRoot = resolve(
  'data/work/groundcover-fixed-k-event-gate',
  sourceSha256.slice(0, 16),
  recipeSha256.slice(0, 16),
);
const root = resolve(typeof args.output === 'string' ? args.output : defaultRoot);
mkdirSync(root, { recursive: true });

console.error(`[fixed-k-event] decoding ${(sourceBytes.byteLength / 1024 / 1024).toFixed(1)} MiB actual mesh`);
const geometry = decodeOwnedProfileGeometry(sourceBytes);
console.error(`[fixed-k-event] building BVH for ${geometry.triangleCount.toLocaleString()} triangles`);
const bvh = TriangleBvh.build(geometry, 8);
const triangleClasses = new Int8Array(geometry.triangleCount);
triangleClasses.fill(-1);
const classify = (triangleId: number): number => {
  let classIndex = triangleClasses[triangleId]!;
  if (classIndex < 0) {
    classIndex = paletteClassForTriangle(sourceBytes, triangleId);
    triangleClasses[triangleId] = classIndex;
  }
  return classIndex;
};

const metricMap = new Map<string, MetricAccumulator>();
const topologyMap = new Map<string, TopologyAccumulator>();
const getMetrics = (
  method: Method,
  k: number,
  family: QueryFamily,
  group: string,
): MetricAccumulator => {
  const key = `${method}|${k}|${family}|${group}`;
  let metrics = metricMap.get(key);
  if (!metrics) {
    metrics = emptyMetrics();
    metricMap.set(key, metrics);
  }
  return metrics;
};
const getTopology = (k: number, group: string): TopologyAccumulator => {
  const key = `${k}|${group}`;
  let topology = topologyMap.get(key);
  if (!topology) {
    topology = emptyTopology();
    topologyMap.set(key, topology);
  }
  return topology;
};

let lines = 0;
let totalEvents = 0;
const started = performance.now();
for (const definition of directions()) {
  const horizontalHeights = definition.sense === 'horizontal'
    ? CAMERA_HEIGHT_FRACTIONS : [null];
  for (let phaseZIndex = 0; phaseZIndex < PHASE_GRID; phaseZIndex++) {
    const phaseZ = geometry.tileOriginZ
      + (phaseZIndex + PHASE_OFFSET) / PHASE_GRID * geometry.tileSizeZ;
    for (let phaseXIndex = 0; phaseXIndex < PHASE_GRID; phaseXIndex++) {
      const phaseX = geometry.tileOriginX
        + (phaseXIndex + PHASE_OFFSET) / PHASE_GRID * geometry.tileSizeX;
      for (const heightFraction of horizontalHeights) {
        const line = lineDefinition(geometry, definition, phaseX, phaseZ, heightFraction);
        const trace = tracePeriodicLine(
          geometry,
          bvh,
          line.origin,
          line.direction,
          ORIGIN_EPSILON_METRES,
          line.maximumT,
        );
        const events: ClassifiedEvent[] = deduplicateLineEvents(
          trace.rawHits,
          EVENT_MERGE_EPSILON_METRES,
        ).map((event) => ({
          ...event,
          classIndex: classify(event.firstTriangleId),
          y: line.origin[1] + line.direction[1] * event.t,
        }));
        lines++;
        totalEvents += events.length;
        const families: Record<QueryFamily, number[]> = {
          anchor_first: [0],
          camera_height: line.cameraOrigins,
          metric_uniform: line.metricOrigins,
        };
        for (const k of K_VALUES) {
          const endpoint = endpointQuantileIndices(events.length, k);
          for (const group of ['all', line.label]) {
            const topology = getTopology(k, group);
            topology.lines++;
            if (events.length === 0) topology.emptyLines++;
            topology.totalEvents += events.length;
            topology.maximumEventsOnLine = Math.max(topology.maximumEventsOnLine, events.length);
            if (events.length <= k) topology.linesExactlyRepresentable++;
            topology.retainedEvents += Math.min(k, events.length);
            for (const event of events) {
              if (PALETTE[event.classIndex]!.panicle) topology.panicleEvents++;
              else topology.stemEvents++;
            }
            for (const index of endpoint) {
              if (PALETTE[events[index]!.classIndex]!.panicle) topology.retainedPanicleEvents++;
              else topology.retainedStemEvents++;
            }
          }
          for (const [family, origins] of Object.entries(families) as [QueryFamily, number[]][]) {
            if (origins.length === 0) continue;
            const oracle = oracleIndices(events, origins, k);
            for (const originT of origins) {
              for (const group of ['all', line.label]) {
                addQuery(
                  getMetrics('endpoint_cdf', k, family, group),
                  events,
                  endpoint,
                  originT,
                  Math.abs(line.direction[1]),
                );
                addQuery(
                  getMetrics('oracle_family_upper_bound', k, family, group),
                  events,
                  oracle,
                  originT,
                  Math.abs(line.direction[1]),
                );
              }
            }
          }
        }
        if (lines % 128 === 0) {
          console.error(`[fixed-k-event] ${lines.toLocaleString()} lines, ${totalEvents.toLocaleString()} events`);
        }
      }
    }
  }
}

const evaluation: Record<string, unknown> = {};
for (const method of ['endpoint_cdf', 'oracle_family_upper_bound'] as const) {
  const byK: Record<string, unknown> = {};
  for (const k of K_VALUES) {
    const byFamily: Record<string, unknown> = {};
    for (const family of ['anchor_first', 'camera_height', 'metric_uniform'] as const) {
      const overall = metricMap.get(`${method}|${k}|${family}|all`);
      if (!overall) continue;
      const byDirection: Record<string, unknown> = {};
      for (const definition of directions()) {
        const label = `${definition.elevationDegrees}:${definition.sense}`;
        if (byDirection[label]) continue;
        const metrics = metricMap.get(`${method}|${k}|${family}|${label}`);
        if (metrics) byDirection[label] = finishMetrics(metrics);
      }
      byFamily[family] = { overall: finishMetrics(overall), byDirection };
    }
    byK[String(k)] = byFamily;
  }
  evaluation[method] = byK;
}

const topology: Record<string, unknown> = {};
for (const k of K_VALUES) {
  const all = topologyMap.get(`${k}|all`)!;
  const byDirection: Record<string, unknown> = {};
  for (const definition of directions()) {
    const label = `${definition.elevationDegrees}:${definition.sense}`;
    if (byDirection[label]) continue;
    const value = topologyMap.get(`${k}|${label}`);
    if (value) byDirection[label] = finishTopology(value);
  }
  topology[String(k)] = { overall: finishTopology(all), byDirection };
}

const report = {
  schema: 'laas-groundcover-fixed-k-event-gate/v1',
  source: {
    file: relative(process.cwd(), source),
    bytes: sourceBytes.byteLength,
    sha256: sourceSha256,
    profileId: geometry.profileId,
    vertexCount: geometry.vertexCount,
    triangleCount: geometry.triangleCount,
    bounds: geometry.bounds,
    tileSize: [geometry.tileSizeX, geometry.tileSizeZ],
  },
  reproducibility: {
    command: [process.execPath, ...process.execArgv, ...process.argv.slice(1)],
    recipeSha256,
    recipe,
  },
  domain: {
    sampledLines: lines,
    exactSurfaceEvents: totalEvents,
    phaseGrid: PHASE_GRID,
    azimuthCount: AZIMUTH_COUNT,
    elevationsDegrees: ELEVATIONS_DEGREES,
    signedNonhorizontalDirections: true,
    exactHorizontalIncluded: true,
    cameraInsideIncluded: true,
    horizontalHorizonMetres: HORIZONTAL_HORIZON_METRES,
    elapsedMilliseconds: performance.now() - started,
  },
  representation: {
    endpointCdf: recipe.endpointCdf,
    oracleUpperBound: recipe.oracle,
    eventIdentity: `analytic periodic mesh intersections merged within ${EVENT_MERGE_EPSILON_METRES} m on one unit ray`,
    query: 'first retained event strictly after live line coordinate; K fixed compares/selects can be statically unrolled',
    favorableBias: [
      'atoms are exact f64 actual-mesh events: no depth quantization error',
      'each line is exact: no phase pooling, spatial filtering, angular interpolation, or line-address error',
      'the oracle may choose a different ideal subset for each query family',
      'therefore failure is a lower bound on practical codec error',
    ],
  },
  storage: {
    maximumResidentBytes: MAXIMUM_RESIDENT_BYTES,
    maximumLogicalReads: 4,
    byK: storageTable(),
  },
  topology,
  evaluation,
  interpretation: [
    'Coverage agreement alone is insufficient: endpoint quantiles include the last event and can report a hit while skipping the visible first surface.',
    'Exact-event agreement, ray-distance error, world-height error, and botanical-class recall expose panicle/stem substitutions.',
    'The event-uniform oracle upper bound is min(K,N)/N per line even before storage, filtering, direction, colour, normal, or multi-species overlap costs.',
    'A rejected K does not authorise runtime traversal or a larger candidate list; it rejects this representation under the fixed-cost contract.',
  ],
};
const metricsPath = resolve(root, 'metrics.json');
const serialized = `${JSON.stringify(report, null, 2)}\n`;
writeFileSync(metricsPath, serialized);
const metricsSha256 = sha256(serialized);
const index = {
  schema: 'laas-groundcover-fixed-k-event-gate-index/v1',
  sourceSha256,
  script: { file: relative(process.cwd(), scriptPath), sha256: scriptSha256 },
  recipeSha256,
  files: [{ file: 'metrics.json', bytes: Buffer.byteLength(serialized), sha256: metricsSha256 }],
};
const indexPath = resolve(root, 'index.json');
writeFileSync(indexPath, `${JSON.stringify(index, null, 2)}\n`);
console.error(`[fixed-k-event] wrote ${metricsPath}`);
console.error(`[fixed-k-event] metrics sha256 ${metricsSha256}`);
process.stdout.write(`${root}\n`);
