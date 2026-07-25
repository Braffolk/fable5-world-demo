/**
 * Offline-only sampled lower-bound probe for a fixed-indirection adaptive
 * pointed-line visibility atlas.
 *
 * This does not emit runtime data. It traces the accepted periodic mesh on a
 * dense 4D lattice inside stratified coarse pointed-line cells, counts exact
 * successor owners, and asks whether uniform per-page refinement plus a fixed
 * shallow binary separator program can fit the resident byte ceiling.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  TriangleBvh,
  decodeOwnedProfileGeometry,
  type CensusVec3,
} from '../../ExteriorClosureCensus';
import { periodicNearestSuccessor } from '../../OriginAwareRayTruth';

interface Args { [key: string]: string | boolean }

interface DirectionClass {
  elevationDegrees: number;
  sense: 'horizontal' | 'downward' | 'upward';
}

interface Sample {
  local: number[];
  owner: string;
  hit: boolean;
  distance: number;
}

interface CellStats {
  sampleCount: number;
  subcellCount: number;
  ownerHistogram: Record<string, number>;
  maximumOwners: number;
  maximumInformationDecisions: number;
  mixedHitMissSubcells: number;
  maximumObservedDepthSpanMetres: number;
  subcells: Array<{
    owners: number;
    hits: number;
    misses: number;
    minimumHitDistance: number | null;
    maximumHitDistance: number | null;
  }>;
}

const EXPECTED_SOURCE_SHA256 =
  '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const DIRECTION_CLASSES: DirectionClass[] = [
  { elevationDegrees: 0, sense: 'horizontal' },
  ...[0.1, 1, 5, 15, 35, 75, 90].flatMap((elevationDegrees) => [
    { elevationDegrees, sense: 'downward' as const },
    { elevationDegrees, sense: 'upward' as const },
  ]),
];
const SOURCE = 'src/assets/groundcover/calamagrostis-canescens.gcrp';
const HORIZON_METRES = 155;
const ORIGIN_EPSILON_METRES = 1e-9;
const PHASE_HEIGHT_COARSE_BINS = 4;
const AZIMUTH_COARSE_BINS = 8;
const COARSE_REPLICATES_PER_DIRECTION_CLASS = 2;
const SAMPLES_PER_AXIS = 9;
const MAXIMUM_MEASURED_REFINEMENT = 2;
const SEPARATOR_DEPTHS = [2, 4, 6] as const;
const COMPLETE_EVENT_BYTES = 12;
const SEPARATOR_BYTES = 16;
const PAGE_DESCRIPTOR_BYTES = 8;
const MAXIMUM_RESIDENT_BYTES = 51_121_152;
const VERTICAL_EPSILON = 1e-12;
const FOV_DEGREES = 55;
const DEVICE_PIXEL_HEIGHT = 2160;
const REFERENCE_RANGE_METRES = 155;

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

function numberArg(value: string | boolean | undefined, fallback: number, label: string): number {
  const parsed = Number(typeof value === 'string' ? value : fallback);
  if (!Number.isFinite(parsed)) throw new Error(`${label} must be finite`);
  return parsed;
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

function hash32(seed: number): number {
  let value = seed >>> 0;
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d);
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b);
  return (value ^ (value >>> 16)) >>> 0;
}

function direction(definition: DirectionClass, azimuth: number): CensusVec3 {
  const elevation = definition.elevationDegrees * Math.PI / 180;
  const horizontal = definition.elevationDegrees === 90 ? 0 : Math.cos(elevation);
  const y = definition.sense === 'downward' ? -Math.sin(elevation)
    : definition.sense === 'upward' ? Math.sin(elevation) : 0;
  return [horizontal * Math.cos(azimuth), y, horizontal * Math.sin(azimuth)];
}

function informationDecisions(ownerCount: number): number {
  return ownerCount <= 1 ? 0 : Math.ceil(Math.log2(ownerCount));
}

function analyzeSamples(samples: readonly Sample[], dimensions: number, refinement: number): CellStats {
  const subdivisions = 1 << refinement;
  const subcellCount = subdivisions ** dimensions;
  const owners = Array.from({ length: subcellCount }, () => new Set<string>());
  const hits = new Uint32Array(subcellCount);
  const misses = new Uint32Array(subcellCount);
  const minimum = new Float64Array(subcellCount).fill(Number.POSITIVE_INFINITY);
  const maximum = new Float64Array(subcellCount).fill(Number.NEGATIVE_INFINITY);

  for (const sample of samples) {
    let index = 0;
    let stride = 1;
    for (let axis = 0; axis < dimensions; axis++) {
      const coordinate = Math.min(subdivisions - 1, Math.floor(sample.local[axis]! * subdivisions));
      index += coordinate * stride;
      stride *= subdivisions;
    }
    owners[index]!.add(sample.owner);
    if (sample.hit) {
      hits[index]++;
      minimum[index] = Math.min(minimum[index]!, sample.distance);
      maximum[index] = Math.max(maximum[index]!, sample.distance);
    } else misses[index]++;
  }

  const histogram = new Map<number, number>();
  let maximumOwners = 0;
  let maximumInformationDecisions = 0;
  let mixedHitMissSubcells = 0;
  let maximumObservedDepthSpanMetres = 0;
  const records: CellStats['subcells'] = [];
  for (let index = 0; index < subcellCount; index++) {
    const ownerCount = owners[index]!.size;
    histogram.set(ownerCount, (histogram.get(ownerCount) ?? 0) + 1);
    maximumOwners = Math.max(maximumOwners, ownerCount);
    maximumInformationDecisions = Math.max(maximumInformationDecisions, informationDecisions(ownerCount));
    if (hits[index]! > 0 && misses[index]! > 0) mixedHitMissSubcells++;
    const depthSpan = hits[index]! > 0 ? maximum[index]! - minimum[index]! : 0;
    maximumObservedDepthSpanMetres = Math.max(maximumObservedDepthSpanMetres, depthSpan);
    records.push({
      owners: ownerCount,
      hits: hits[index]!,
      misses: misses[index]!,
      minimumHitDistance: hits[index]! > 0 ? minimum[index]! : null,
      maximumHitDistance: hits[index]! > 0 ? maximum[index]! : null,
    });
  }
  return {
    sampleCount: samples.length,
    subcellCount,
    ownerHistogram: Object.fromEntries([...histogram.entries()].sort((a, b) => a[0] - b[0]).map(([key, value]) => [String(key), value])),
    maximumOwners,
    maximumInformationDecisions,
    mixedHitMissSubcells,
    maximumObservedDepthSpanMetres,
    subcells: records,
  };
}

function bytesForMeasuredPage(stats: CellStats): { optimistic: number; explicit: number } {
  let optimistic = PAGE_DESCRIPTOR_BYTES;
  let explicit = PAGE_DESCRIPTOR_BYTES;
  for (const subcell of stats.subcells) {
    const owners = Math.max(1, subcell.owners);
    // Optimistic form assumes one uint owner token and one uint separator token
    // per binary distinction, with all chart payload globally free.
    optimistic += owners * 4 + Math.max(0, owners - 1) * 4;
    // Explicit form attaches one complete event/chart payload to every leaf and
    // one compact half/packed separator covector to every binary distinction.
    explicit += owners * COMPLETE_EVENT_BYTES + Math.max(0, owners - 1) * SEPARATOR_BYTES;
  }
  return { optimistic, explicit };
}

function nextLevelOneOwnerLowerBound(dimensions: number): number {
  return PAGE_DESCRIPTOR_BYTES + (2 ** (dimensions * (MAXIMUM_MEASURED_REFINEMENT + 1))) * COMPLETE_EVENT_BYTES;
}

const args = parseArgs(process.argv.slice(2));
const source = resolve(typeof args.source === 'string' ? args.source : SOURCE);
const sourceBytes = readFileSync(source);
const sourceSha256 = sha256(sourceBytes);
if (sourceSha256 !== EXPECTED_SOURCE_SHA256) {
  throw new Error(`unexpected source SHA-256 ${sourceSha256}`);
}
const horizon = numberArg(args.horizon, HORIZON_METRES, 'horizon');
const samplesPerAxis = numberArg(args['samples-per-axis'], SAMPLES_PER_AXIS, 'samples per axis');
const replicates = numberArg(
  args.replicates,
  COARSE_REPLICATES_PER_DIRECTION_CLASS,
  'replicates',
);
if (!Number.isInteger(samplesPerAxis) || samplesPerAxis < 3) {
  throw new Error('samples per axis must be an integer >= 3');
}
if (!Number.isInteger(replicates) || replicates < 1) {
  throw new Error('replicates must be a positive integer');
}

const configuration = {
  horizon,
  originEpsilonMetres: ORIGIN_EPSILON_METRES,
  phaseHeightCoarseBins: PHASE_HEIGHT_COARSE_BINS,
  azimuthCoarseBins: AZIMUTH_COARSE_BINS,
  replicatesPerDirectionClass: replicates,
  samplesPerAxis,
  maximumMeasuredRefinement: MAXIMUM_MEASURED_REFINEMENT,
  directionClasses: DIRECTION_CLASSES,
  separatorDepths: SEPARATOR_DEPTHS,
  completeEventBytes: COMPLETE_EVENT_BYTES,
  separatorBytes: SEPARATOR_BYTES,
  pageDescriptorBytes: PAGE_DESCRIPTOR_BYTES,
  maximumResidentBytes: MAXIMUM_RESIDENT_BYTES,
  coordinateDomain: 'one fixed elevation/sense stratum; periodic phase X/Z, inside-origin height, and azimuth vary continuously inside each sampled coarse cell',
};
const configurationSha256 = sha256(canonicalJson(configuration));
const scriptPath = fileURLToPath(import.meta.url);
const scriptSha256 = sha256(readFileSync(scriptPath));
const outputRoot = resolve(typeof args.output === 'string'
  ? args.output
  : `data/work/groundcover-adaptive-line-cell-probe/${sourceSha256.slice(0, 16)}/${configurationSha256.slice(0, 16)}`);
mkdirSync(outputRoot, { recursive: true });

console.error(`[adaptive-line-cell] decoding ${(sourceBytes.byteLength / 1024 / 1024).toFixed(1)} MiB mesh`);
const geometry = decodeOwnedProfileGeometry(sourceBytes);
const bvh = TriangleBvh.build(geometry, 8);
const sampledCells: Array<{
  directionClass: DirectionClass;
  directionClassIndex: number;
  replicate: number;
  coarseIndex: { x: number; y: number; z: number; azimuth: number };
  dimensions: number;
  weightInFullCoarseDomain: number;
  stats: CellStats[];
}> = [];

let tracedRays = 0;
let tracedHits = 0;
for (let directionClassIndex = 0; directionClassIndex < DIRECTION_CLASSES.length; directionClassIndex++) {
  const definition = DIRECTION_CLASSES[directionClassIndex]!;
  const vertical = Math.abs(definition.elevationDegrees - 90) <= VERTICAL_EPSILON;
  const dimensions = vertical ? 3 : 4;
  const azimuthBins = vertical ? 1 : AZIMUTH_COARSE_BINS;
  const totalCoarseCellsInClass = PHASE_HEIGHT_COARSE_BINS ** 3 * azimuthBins;
  for (let replicate = 0; replicate < replicates; replicate++) {
    const seed = hash32((directionClassIndex + 1) * 0x9e3779b9 ^ (replicate + 1) * 0x85ebca6b);
    const coarseX = hash32(seed ^ 0x243f6a88) % PHASE_HEIGHT_COARSE_BINS;
    const coarseY = hash32(seed ^ 0x13198a2e) % PHASE_HEIGHT_COARSE_BINS;
    const coarseZ = hash32(seed ^ 0xa4093822) % PHASE_HEIGHT_COARSE_BINS;
    const coarseAzimuth = vertical ? 0 : hash32(seed ^ 0x299f31d0) % azimuthBins;
    const samples: Sample[] = [];
    const azimuthSampleCount = vertical ? 1 : samplesPerAxis;
    for (let azimuthIndex = 0; azimuthIndex < azimuthSampleCount; azimuthIndex++) {
      const localAzimuth = vertical ? 0.5 : (azimuthIndex + 0.5) / samplesPerAxis;
      const azimuthFraction = (coarseAzimuth + localAzimuth) / azimuthBins;
      const rayDirection = direction(definition, azimuthFraction * Math.PI * 2);
      for (let zIndex = 0; zIndex < samplesPerAxis; zIndex++) {
        const localZ = (zIndex + 0.5) / samplesPerAxis;
        const z = (coarseZ + localZ) / PHASE_HEIGHT_COARSE_BINS;
        for (let yIndex = 0; yIndex < samplesPerAxis; yIndex++) {
          const localY = (yIndex + 0.5) / samplesPerAxis;
          const y = (coarseY + localY) / PHASE_HEIGHT_COARSE_BINS;
          for (let xIndex = 0; xIndex < samplesPerAxis; xIndex++) {
            const localX = (xIndex + 0.5) / samplesPerAxis;
            const x = (coarseX + localX) / PHASE_HEIGHT_COARSE_BINS;
            const origin: CensusVec3 = [
              geometry.tileOriginX + x * geometry.tileSizeX,
              geometry.bounds.min[1] + y * (geometry.bounds.max[1] - geometry.bounds.min[1]),
              geometry.tileOriginZ + z * geometry.tileSizeZ,
            ];
            const hit = periodicNearestSuccessor(
              geometry,
              bvh,
              origin,
              rayDirection,
              horizon,
              ORIGIN_EPSILON_METRES,
            );
            samples.push({
              local: vertical ? [localX, localY, localZ] : [localX, localY, localZ, localAzimuth],
              owner: hit ? `${hit.triangleId}:${hit.copyX}:${hit.copyZ}` : 'miss',
              hit: hit !== null,
              distance: hit?.t ?? horizon,
            });
            tracedRays++;
            if (hit) tracedHits++;
          }
        }
      }
    }
    const stats = Array.from(
      { length: MAXIMUM_MEASURED_REFINEMENT + 1 },
      (_, refinement) => analyzeSamples(samples, dimensions, refinement),
    );
    sampledCells.push({
      directionClass: definition,
      directionClassIndex,
      replicate,
      coarseIndex: { x: coarseX, y: coarseY, z: coarseZ, azimuth: coarseAzimuth },
      dimensions,
      weightInFullCoarseDomain: totalCoarseCellsInClass / replicates,
      stats,
    });
    console.error(
      `[adaptive-line-cell] ${sampledCells.length}/${DIRECTION_CLASSES.length * replicates} ${definition.elevationDegrees}/${definition.sense}: ${samples.length.toLocaleString()} rays, level-2 max owners ${stats[2]!.maximumOwners}`,
    );
  }
}

const projection = SEPARATOR_DEPTHS.map((separatorDepth) => {
  let projectedOptimisticBytes = 0;
  let projectedExplicitBytes = 0;
  let projectedNextLevelLowerBoundBytes = 0;
  let sampledPassingCells = 0;
  let sampledFailingCells = 0;
  let sampledUnresolvedSubcellsAtLevel2 = 0;
  let sampledUnresolvedMixedHitMissAtLevel2 = 0;
  let maximumUnresolvedObservedDepthSpanMetres = 0;
  let maximumRequiredDecisions = 0;
  const selectedLevels: Record<string, number> = {};
  for (const cell of sampledCells) {
    maximumRequiredDecisions = Math.max(
      maximumRequiredDecisions,
      ...cell.stats.map((stats) => stats.maximumInformationDecisions),
    );
    const selected = cell.stats.findIndex((stats) => stats.maximumInformationDecisions <= separatorDepth);
    if (selected >= 0) {
      sampledPassingCells++;
      selectedLevels[String(selected)] = (selectedLevels[String(selected)] ?? 0) + 1;
      const bytes = bytesForMeasuredPage(cell.stats[selected]!);
      projectedOptimisticBytes += bytes.optimistic * cell.weightInFullCoarseDomain;
      projectedExplicitBytes += bytes.explicit * cell.weightInFullCoarseDomain;
      projectedNextLevelLowerBoundBytes += bytes.explicit * cell.weightInFullCoarseDomain;
    } else {
      sampledFailingCells++;
      selectedLevels[`>${MAXIMUM_MEASURED_REFINEMENT}`] =
        (selectedLevels[`>${MAXIMUM_MEASURED_REFINEMENT}`] ?? 0) + 1;
      const level2 = cell.stats[MAXIMUM_MEASURED_REFINEMENT]!;
      for (const subcell of level2.subcells) {
        if (informationDecisions(subcell.owners) <= separatorDepth) continue;
        sampledUnresolvedSubcellsAtLevel2++;
        if (subcell.hits > 0 && subcell.misses > 0) sampledUnresolvedMixedHitMissAtLevel2++;
        if (subcell.minimumHitDistance !== null && subcell.maximumHitDistance !== null) {
          maximumUnresolvedObservedDepthSpanMetres = Math.max(
            maximumUnresolvedObservedDepthSpanMetres,
            subcell.maximumHitDistance - subcell.minimumHitDistance,
          );
        }
      }
      const lower = nextLevelOneOwnerLowerBound(cell.dimensions);
      projectedOptimisticBytes += lower * cell.weightInFullCoarseDomain;
      projectedExplicitBytes += lower * cell.weightInFullCoarseDomain;
      projectedNextLevelLowerBoundBytes += lower * cell.weightInFullCoarseDomain;
    }
  }
  const azimuthWidthAtNextLevelDegrees = 360
    / AZIMUTH_COARSE_BINS
    / (1 << (MAXIMUM_MEASURED_REFINEMENT + 1));
  const angularPixelWidth = 2 * Math.atan(Math.tan(FOV_DEGREES * Math.PI / 360) / DEVICE_PIXEL_HEIGHT);
  const nextLevelAngularWidthPixels = azimuthWidthAtNextLevelDegrees * Math.PI / 180 / angularPixelWidth;
  const nextLevelWorldWidthAtReferenceRange = 2 * REFERENCE_RANGE_METRES
    * Math.tan(azimuthWidthAtNextLevelDegrees * Math.PI / 360);
  return {
    separatorDepth,
    fixedSeparatorLeaves: 1 << separatorDepth,
    fixedRuntimeIndirections: 2,
    selectedMeasuredRefinementHistogram: selectedLevels,
    sampledPassingCells,
    sampledFailingCells,
    sampledUnresolvedSubcellsAtLevel2,
    sampledUnresolvedMixedHitMissAtLevel2,
    maximumRequiredDecisionsObservedAtMeasuredLevels: maximumRequiredDecisions,
    maximumUnresolvedObservedDepthSpanMetres,
    conservativeUnresolvedDepthErrorMetres: sampledFailingCells > 0 ? horizon : 0,
    conservativeUnresolvedHitMissOrRadianceError: sampledFailingCells > 0
      ? 'unbounded/categorical until the complete owner arrangement is certified'
      : 'none on sampled cells only; continuum certification remains absent',
    nextUniformRefinementAngularWidthDegrees: azimuthWidthAtNextLevelDegrees,
    nextUniformRefinementAngularWidthDevicePixels: nextLevelAngularWidthPixels,
    nextUniformRefinementWorldWidthAt155Metres: nextLevelWorldWidthAtReferenceRange,
    projectedOptimisticBytes: Math.ceil(projectedOptimisticBytes),
    projectedExplicitBytes: Math.ceil(projectedExplicitBytes),
    projectedNextLevelOneOwnerLowerBoundBytes: Math.ceil(projectedNextLevelLowerBoundBytes),
    residentCapBytes: MAXIMUM_RESIDENT_BYTES,
    passesResidentCapEvenUnderNextLevelOneOwnerLowerBound:
      projectedNextLevelLowerBoundBytes <= MAXIMUM_RESIDENT_BYTES,
  };
});

const report = {
  schema: 'laas-groundcover-adaptive-pointed-line-cell-probe/v1',
  status: projection.every((entry) => entry.passesResidentCapEvenUnderNextLevelOneOwnerLowerBound)
    ? 'sampled-byte-gate-not-rejected-continuum-uncertified'
    : 'rejected-by-sampled-next-level-byte-lower-bound',
  source: {
    file: source,
    bytes: sourceBytes.byteLength,
    sha256: sourceSha256,
    profileId: geometry.profileId,
    tileSize: [geometry.tileSizeX, geometry.tileSizeZ],
    bounds: geometry.bounds,
  },
  tool: { file: scriptPath, sha256: scriptSha256 },
  command: [process.execPath, ...process.execArgv, ...process.argv.slice(1)],
  configuration,
  configurationSha256,
  sampleDomain: {
    coarseCellsSampled: sampledCells.length,
    totalCoarseCellsProjected: DIRECTION_CLASSES.reduce((sum, definition) =>
      sum + PHASE_HEIGHT_COARSE_BINS ** 3
        * (definition.elevationDegrees === 90 ? 1 : AZIMUTH_COARSE_BINS), 0),
    rays: tracedRays,
    hits: tracedHits,
    hitFraction: tracedHits / tracedRays,
    scope: 'actual periodic mesh successors for full named horizontal/up/down angle strata and inside-origin X/Y/Z; two deterministic coarse pages per stratum; dense tensor samples within every selected page',
    limitation: 'a sampled information/memory lower bound, not a continuum visibility-cell certificate; unseen owners can only increase required separators and storage',
  },
  fixedLayoutUnderTest: {
    indirections: 2,
    reads: 'one coarse page descriptor plus one direct refined record/page read; separator program is fixed predicated ALU, not runtime traversal',
    measuredRefinements: [0, 1, 2],
    nextLevelProjection: 3,
    refinementBranching: '2^D uniform subcells per page level, D=4 for nonvertical angle strata and D=3 at the vertical pole',
    runtimeExcluded: ['loop', 'march', 'tree traversal', 'candidate list', 'per-copy geometry', 'per-species query'],
  },
  projection,
  sampledCells,
  decision: [
    'A pass would only justify a denser continuum/certification bake; it would not itself authorize a shader.',
    'A failure is decisive because page bytes assume just one 12-byte owner record per next-level subcell and zero separator bytes, while sampled cells already require deeper refinement.',
    'Any unresolved cell retains the formal 155 m depth bound and categorical hit/miss/radiance uncertainty; a mean or median event is not accepted as geometry.',
    'The source mesh is already one marked population. Multi-species and moss must be unioned offline before the identical probe; runtime cost remains one community query.',
  ],
};
const reportPath = resolve(outputRoot, 'report.json');
writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
const index = {
  schema: 'laas-groundcover-adaptive-pointed-line-cell-probe-index/v1',
  sourceSha256,
  configurationSha256,
  toolSha256: scriptSha256,
  report: { file: 'report.json', sha256: sha256(readFileSync(reportPath)) },
};
writeFileSync(resolve(outputRoot, 'index.json'), `${JSON.stringify(index, null, 2)}\n`);
console.error(`[adaptive-line-cell] wrote ${reportPath}`);
process.stdout.write(`${reportPath}\n`);
