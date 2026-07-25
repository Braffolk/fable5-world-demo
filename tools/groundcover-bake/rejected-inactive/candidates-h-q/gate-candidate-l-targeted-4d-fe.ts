/**
 * Candidate-L targeted product-simplex rate gate.
 *
 * Offline only: exact periodic GCRP first hits -> canonical positive measures
 * -> unlimited-precision interpolation comparisons.  No runtime/shader code.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';
import { TriangleBvh, decodeOwnedProfileGeometry } from '../../ExteriorClosureCensus';
import {
  renderPointPage,
  scorePage,
  sha256,
  sourceAttributes,
  type DirectionSpec,
  type FieldPage,
} from './gate-candidate-k-heldout-fe';

const WORKSPACE = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const SOURCE = resolve(WORKSPACE, 'src/assets/groundcover/calamagrostis-canescens.gcrp');
const THRESHOLDS = resolve(
  WORKSPACE,
  'docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-CANDIDATE-L-TARGETED-RATE-THRESHOLDS-FROZEN.md',
);
const VERSION = 'candidate-l-targeted-4d-fe-v1';
const RESOLUTION = 128;
const SPATIAL_PITCH = 2;
const REFERENCE_Y = 0.49255;
const RADII = [4, 16] as const;
const LIMITS = {
  coverageP95: 0.08,
  coverageP99: 0.20,
  rgbP95: 0.06,
  rgbP99: 0.15,
  connected: 0.01,
  continuity: 1e-5,
  materialRatio: 0.75,
  simplexToTensorRatio: 1.25,
} as const;

interface DirectionCell {
  key: string;
  elevationLow: number;
  elevationHigh: number;
  azimuthLow: number;
  azimuthHigh: number;
}

const CELLS: readonly DirectionCell[] = [
  { key: 'grazing-worst', elevationLow: 0.25, elevationHigh: 2, azimuthLow: 67.5, azimuthHigh: 90 },
  { key: 'standing-worst', elevationLow: 10, elevationHigh: 18, azimuthLow: 247.5, azimuthHigh: 270 },
] as const;

type Bary3 = readonly [number, number, number];
type Vec2 = readonly [number, number];
type QuadPages = readonly [FieldPage, FieldPage, FieldPage, FieldPage];
type TriPages = readonly [FieldPage, FieldPage, FieldPage];

const DIRECTION_INTERIORS: readonly { key: string; bary: Bary3; boundary: boolean }[] = [
  { key: 'centroid', bary: [1 / 3, 1 / 3, 1 / 3], boundary: false },
  { key: 'quarter-v0', bary: [0.5, 0.25, 0.25], boundary: false },
  { key: 'quarter-v1', bary: [0.25, 0.5, 0.25], boundary: false },
  { key: 'quarter-v2', bary: [0.25, 0.25, 0.5], boundary: false },
] as const;

function directionSites(triangle: number): readonly { key: string; bary: Bary3; boundary: boolean }[] {
  return [
    ...DIRECTION_INTERIORS,
    triangle === 0
      ? { key: 'shared-diagonal', bary: [0.5, 0, 0.5] as Bary3, boundary: true }
      : { key: 'shared-diagonal', bary: [0.5, 0.5, 0] as Bary3, boundary: true },
  ];
}

interface ScoreRecord {
  cell: string;
  directionTriangle: number;
  site: string;
  boundary: boolean;
  elevationDegrees: number;
  azimuthDegrees: number;
  radius: number;
  predictor: 'kSameQ' | 'lSimplex5' | 'tensor9';
  score: ReturnType<typeof scorePage>['score'];
  normalizedP95: number;
}

function canonical(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  const object = value as Record<string, unknown>;
  return `{${Object.keys(object).sort().map((key) => `${JSON.stringify(key)}:${canonical(object[key])}`).join(',')}}`;
}

function positiveModulo(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus;
}

function slope(elevationDegrees: number, azimuthDegrees: number): Vec2 {
  const elevation = elevationDegrees * Math.PI / 180;
  const azimuth = azimuthDegrees * Math.PI / 180;
  const magnitude = 1 / Math.tan(elevation);
  return [magnitude * Math.cos(azimuth), magnitude * Math.sin(azimuth)];
}

function fromSlope(value: Vec2): { elevationDegrees: number; azimuthDegrees: number } {
  const magnitude = Math.hypot(value[0], value[1]);
  return {
    elevationDegrees: Math.atan2(1, magnitude) * 180 / Math.PI,
    azimuthDegrees: positiveModulo(Math.atan2(value[1], value[0]) * 180 / Math.PI, 360),
  };
}

function mixSlope(vertices: readonly [Vec2, Vec2, Vec2], bary: Bary3): Vec2 {
  return [
    bary[0] * vertices[0][0] + bary[1] * vertices[1][0] + bary[2] * vertices[2][0],
    bary[0] * vertices[0][1] + bary[1] * vertices[1][1] + bary[2] * vertices[2][1],
  ];
}

function quadSpecs(cell: DirectionCell): readonly [DirectionSpec, DirectionSpec, DirectionSpec, DirectionSpec] {
  return [
    { key: `${cell.key}-00`, elevationDegrees: cell.elevationLow, azimuthDegrees: cell.azimuthLow, kind: 'node' },
    { key: `${cell.key}-10`, elevationDegrees: cell.elevationLow, azimuthDegrees: cell.azimuthHigh, kind: 'node' },
    { key: `${cell.key}-11`, elevationDegrees: cell.elevationHigh, azimuthDegrees: cell.azimuthHigh, kind: 'node' },
    { key: `${cell.key}-01`, elevationDegrees: cell.elevationHigh, azimuthDegrees: cell.azimuthLow, kind: 'node' },
  ];
}

function directionTriangle<T>(quad: readonly [T, T, T, T], triangle: number): readonly [T, T, T] {
  return triangle === 0 ? [quad[0], quad[1], quad[2]] : [quad[0], quad[2], quad[3]];
}

function pageValue(page: FieldPage, x: number, z: number, channel: number): number {
  const px = positiveModulo(x, RESOLUTION);
  const pz = positiveModulo(z, RESOLUTION);
  return page.rgba[(pz * RESOLUTION + px) * 4 + channel]!;
}

/** Exact physical box width, with fractional overlap on boundary cells. */
function physicalBoxFilter(page: FieldPage, canonicalRadius: number): FieldPage {
  const supportWidthCells = (2 * canonicalRadius + 1) * RESOLUTION / 256;
  const half = supportWidthCells / 2;
  const extent = Math.ceil(half + 0.5);
  const offsets: { offset: number; weight: number }[] = [];
  for (let offset = -extent; offset <= extent; offset++) {
    const overlap = Math.max(0, Math.min(half, offset + 0.5) - Math.max(-half, offset - 0.5));
    if (overlap > 0) offsets.push({ offset, weight: overlap });
  }
  const weightSum = offsets.reduce((sum, entry) => sum + entry.weight, 0);
  if (Math.abs(weightSum - supportWidthCells) > 1e-12) throw new Error('physical box weight mismatch');
  const horizontal = new Float64Array(page.rgba.length);
  const output = new Float32Array(page.rgba.length);
  for (let z = 0; z < RESOLUTION; z++) {
    for (let x = 0; x < RESOLUTION; x++) {
      for (let channel = 0; channel < 4; channel++) {
        let sum = 0;
        for (const entry of offsets) sum += entry.weight * pageValue(page, x + entry.offset, z, channel);
        horizontal[(z * RESOLUTION + x) * 4 + channel] = sum / weightSum;
      }
    }
  }
  for (let z = 0; z < RESOLUTION; z++) {
    for (let x = 0; x < RESOLUTION; x++) {
      for (let channel = 0; channel < 4; channel++) {
        let sum = 0;
        for (const entry of offsets) {
          const pz = positiveModulo(z + entry.offset, RESOLUTION);
          sum += entry.weight * horizontal[(pz * RESOLUTION + x) * 4 + channel]!;
        }
        output[(z * RESOLUTION + x) * 4 + channel] = sum / weightSum;
      }
    }
  }
  return { rgba: output };
}

function spatialTriangle(x: number, z: number): {
  vertices: readonly [[number, number], [number, number], [number, number]];
  bary: Bary3;
} {
  const baseX = Math.floor(x / SPATIAL_PITCH) * SPATIAL_PITCH;
  const baseZ = Math.floor(z / SPATIAL_PITCH) * SPATIAL_PITCH;
  const u = (x - baseX) / SPATIAL_PITCH;
  const v = (z - baseZ) / SPATIAL_PITCH;
  if (v <= u) {
    return {
      vertices: [[baseX, baseZ], [baseX + SPATIAL_PITCH, baseZ], [baseX + SPATIAL_PITCH, baseZ + SPATIAL_PITCH]],
      bary: [1 - u, u - v, v],
    };
  }
  return {
    vertices: [[baseX, baseZ], [baseX + SPATIAL_PITCH, baseZ + SPATIAL_PITCH], [baseX, baseZ + SPATIAL_PITCH]],
    bary: [1 - v, u, v - u],
  };
}

/** Standard staircase simplex in Delta2 x Delta2. */
function productSimplexPairs(spatial: Bary3, directional: Bary3): {
  pairs: readonly [number, number][];
  weights: readonly number[];
} {
  const ordered = [
    { value: spatial[1] + spatial[2], factor: 0 as const },
    { value: spatial[2], factor: 0 as const },
    { value: directional[1] + directional[2], factor: 1 as const },
    { value: directional[2], factor: 1 as const },
  ].sort((left, right) => right.value - left.value || left.factor - right.factor);
  const values = ordered.map((entry) => entry.value);
  const weights = [
    1 - values[0]!,
    values[0]! - values[1]!,
    values[1]! - values[2]!,
    values[2]! - values[3]!,
    values[3]!,
  ];
  const pairs: [number, number][] = [[0, 0]];
  let q = 0;
  let d = 0;
  for (const entry of ordered) {
    if (entry.factor === 0) q++;
    else d++;
    pairs.push([q, d]);
  }
  return { pairs, weights };
}

function lSimplexChannel(
  pages: TriPages,
  vertices: readonly [[number, number], [number, number], [number, number]],
  spatialBary: Bary3,
  directionBary: Bary3,
  channel: number,
): number {
  const simplex = productSimplexPairs(spatialBary, directionBary);
  let value = 0;
  for (let index = 0; index < 5; index++) {
    const [q, d] = simplex.pairs[index]!;
    const vertex = vertices[q]!;
    value += simplex.weights[index]! * pageValue(pages[d]!, vertex[0], vertex[1], channel);
  }
  return value;
}

function tensorChannel(
  pages: TriPages,
  vertices: readonly [[number, number], [number, number], [number, number]],
  spatialBary: Bary3,
  directionBary: Bary3,
  channel: number,
): number {
  let value = 0;
  for (let q = 0; q < 3; q++) {
    const vertex = vertices[q]!;
    for (let d = 0; d < 3; d++) {
      value += spatialBary[q]! * directionBary[d]! * pageValue(pages[d]!, vertex[0], vertex[1], channel);
    }
  }
  return value;
}

function kSameQChannel(
  pages: QuadPages,
  x: number,
  z: number,
  cell: DirectionCell,
  live: { elevationDegrees: number; azimuthDegrees: number },
  channel: number,
): number {
  const a = (live.azimuthDegrees - cell.azimuthLow) / (cell.azimuthHigh - cell.azimuthLow);
  const t = (live.elevationDegrees - cell.elevationLow) / (cell.elevationHigh - cell.elevationLow);
  const weights = [(1 - a) * (1 - t), a * (1 - t), a * t, (1 - a) * t] as const;
  let value = 0;
  for (let corner = 0; corner < 4; corner++) value += weights[corner]! * pageValue(pages[corner]!, x, z, channel);
  return value;
}

function predictPages(
  quad: QuadPages,
  triangle: number,
  directionBary: Bary3,
  cell: DirectionCell,
  live: { elevationDegrees: number; azimuthDegrees: number },
): { kSameQ: FieldPage; lSimplex5: FieldPage; tensor9: FieldPage } {
  const tri = directionTriangle(quad, triangle);
  const outputs = {
    kSameQ: new Float32Array(RESOLUTION * RESOLUTION * 4),
    lSimplex5: new Float32Array(RESOLUTION * RESOLUTION * 4),
    tensor9: new Float32Array(RESOLUTION * RESOLUTION * 4),
  };
  for (let z = 0; z < RESOLUTION; z++) {
    for (let x = 0; x < RESOLUTION; x++) {
      const spatial = spatialTriangle(x, z);
      const offset = (z * RESOLUTION + x) * 4;
      for (let channel = 0; channel < 4; channel++) {
        outputs.kSameQ[offset + channel] = kSameQChannel(quad, x, z, cell, live, channel);
        outputs.lSimplex5[offset + channel] = lSimplexChannel(
          tri,
          spatial.vertices,
          spatial.bary,
          directionBary,
          channel,
        );
        outputs.tensor9[offset + channel] = tensorChannel(
          tri,
          spatial.vertices,
          spatial.bary,
          directionBary,
          channel,
        );
      }
    }
  }
  return {
    kSameQ: { rgba: outputs.kSameQ },
    lSimplex5: { rgba: outputs.lSimplex5 },
    tensor9: { rgba: outputs.tensor9 },
  };
}

function normalizedP95(score: ReturnType<typeof scorePage>['score']): number {
  return Math.max(score.coverage.p95 / LIMITS.coverageP95, score.premulRgb.p95 / LIMITS.rgbP95);
}

function appearance(page: FieldPage, pixel: number): readonly [number, number, number] {
  const offset = pixel * 4;
  const a = page.rgba[offset + 3]!;
  return [0, 1, 2].map((channel) => Math.round(255 * Math.max(0, Math.min(1, page.rgba[offset + channel]! + 0.12 * (1 - a))))) as unknown as readonly [number, number, number];
}

async function writeQa(
  path: string,
  truth: FieldPage,
  predictions: { kSameQ: FieldPage; lSimplex5: FieldPage; tensor9: FieldPage },
): Promise<void> {
  const panels = [truth, predictions.kSameQ, predictions.lSimplex5, predictions.tensor9] as const;
  const buffer = Buffer.alloc(RESOLUTION * panels.length * RESOLUTION * 3);
  for (let y = 0; y < RESOLUTION; y++) {
    for (let x = 0; x < RESOLUTION; x++) {
      const pixel = y * RESOLUTION + x;
      for (let panel = 0; panel < panels.length; panel++) {
        const color = appearance(panels[panel]!, pixel);
        const target = (y * RESOLUTION * panels.length + panel * RESOLUTION + x) * 3;
        buffer[target] = color[0];
        buffer[target + 1] = color[1];
        buffer[target + 2] = color[2];
      }
    }
  }
  await sharp(buffer, { raw: { width: RESOLUTION * panels.length, height: RESOLUTION, channels: 3 } })
    .png()
    .toFile(path);
}

function continuityProbe(pages: TriPages, radius: number): number {
  let maximum = 0;
  const vertices = [[0, 0], [SPATIAL_PITCH, 0], [SPATIAL_PITCH, SPATIAL_PITCH]] as const;
  const epsilon = 1e-8;
  const spatial: Bary3 = [0.2, 0.3, 0.5];
  const minus: Bary3 = [0.2, 0.3 + epsilon, 0.5 - epsilon];
  const plus: Bary3 = [0.2, 0.3 - epsilon, 0.5 + epsilon];
  for (let channel = 0; channel < 4; channel++) {
    const left = lSimplexChannel(pages, vertices, spatial, minus, channel);
    const right = lSimplexChannel(pages, vertices, spatial, plus, channel);
    maximum = Math.max(maximum, Math.abs(left - right));
  }
  if (!Number.isFinite(maximum)) throw new Error(`non-finite continuity result radius=${radius}`);
  return maximum;
}

async function main(): Promise<void> {
  const sourceBytes = readFileSync(SOURCE);
  const geometry = decodeOwnedProfileGeometry(sourceBytes);
  const attributes = sourceAttributes(sourceBytes);
  const bvh = TriangleBvh.build(geometry);
  const pageCache = new Map<string, Map<number, FieldPage>>();

  const page = (spec: DirectionSpec): Map<number, FieldPage> => {
    const found = pageCache.get(spec.key);
    if (found) return found;
    console.log(`[candidate-l] raycast ${pageCache.size + 1} ${spec.key}`);
    const point = renderPointPage(geometry, bvh, attributes, RESOLUTION, REFERENCE_Y, spec);
    const result = new Map<number, FieldPage>();
    for (const radius of RADII) result.set(radius, physicalBoxFilter(point, radius));
    pageCache.set(spec.key, result);
    return result;
  };

  const scores: ScoreRecord[] = [];
  const qaCandidates: {
    normalized: number;
    key: string;
    truth: FieldPage;
    predictions: { kSameQ: FieldPage; lSimplex5: FieldPage; tensor9: FieldPage };
  }[] = [];
  const continuity: Record<string, number> = {};

  for (const cell of CELLS) {
    const specs = quadSpecs(cell);
    const slopes = specs.map((spec) => slope(spec.elevationDegrees, spec.azimuthDegrees)) as unknown as readonly [Vec2, Vec2, Vec2, Vec2];
    for (let triangle = 0; triangle < 2; triangle++) {
      const triSpecs = directionTriangle(specs, triangle);
      const triSlopes = directionTriangle(slopes, triangle);
      for (const radius of RADII) {
        const triPages = triSpecs.map((spec) => page(spec).get(radius)!) as unknown as TriPages;
        continuity[`${cell.key}-tri${triangle}-r${radius}`] = continuityProbe(triPages, radius);
      }
      for (const site of directionSites(triangle)) {
        const live = fromSlope(mixSlope(triSlopes, site.bary));
        const liveSpec: DirectionSpec = {
          key: `${cell.key}-tri${triangle}-${site.key}-${live.elevationDegrees.toFixed(9)}-${live.azimuthDegrees.toFixed(9)}`,
          elevationDegrees: live.elevationDegrees,
          azimuthDegrees: live.azimuthDegrees,
          kind: 'heldout',
        };
        for (const radius of RADII) {
          const quad = specs.map((spec) => page(spec).get(radius)!) as unknown as QuadPages;
          const truth = page(liveSpec).get(radius)!;
          const predictions = predictPages(quad, triangle, site.bary, cell, live);
          for (const predictor of ['kSameQ', 'lSimplex5', 'tensor9'] as const) {
            const score = scorePage(truth, predictions[predictor], RESOLUTION).score;
            scores.push({
              cell: cell.key,
              directionTriangle: triangle,
              site: site.key,
              boundary: site.boundary,
              elevationDegrees: live.elevationDegrees,
              azimuthDegrees: live.azimuthDegrees,
              radius,
              predictor,
              score,
              normalizedP95: normalizedP95(score),
            });
          }
          const lScore = scores.at(-2)!;
          qaCandidates.push({
            normalized: Math.max(lScore.normalizedP95, lScore.score.largestConnectedFraction / LIMITS.connected),
            key: `${cell.key}-tri${triangle}-${site.key}-r${radius}`,
            truth,
            predictions,
          });
        }
      }
    }
  }

  const byPredictor = Object.fromEntries((['kSameQ', 'lSimplex5', 'tensor9'] as const).map((predictor) => {
    const values = scores.filter((entry) => entry.predictor === predictor);
    return [predictor, {
      cases: values.length,
      everyGreen: values.every((entry) => entry.score.green),
      greenCases: values.filter((entry) => entry.score.green).length,
      worstNormalizedP95: Math.max(...values.map((entry) => entry.normalizedP95)),
      worstConnected: Math.max(...values.map((entry) => entry.score.largestConnectedFraction)),
      worstCoverageP95: Math.max(...values.map((entry) => entry.score.coverage.p95)),
      worstRgbP95: Math.max(...values.map((entry) => entry.score.premulRgb.p95)),
    }];
  })) as Record<'kSameQ' | 'lSimplex5' | 'tensor9', {
    cases: number;
    everyGreen: boolean;
    greenCases: number;
    worstNormalizedP95: number;
    worstConnected: number;
    worstCoverageP95: number;
    worstRgbP95: number;
  }>;
  const continuityMaximum = Math.max(...Object.values(continuity));
  const materialRatio = byPredictor.lSimplex5.worstNormalizedP95 / byPredictor.kSameQ.worstNormalizedP95;
  const simplexToTensorRatio = byPredictor.lSimplex5.worstNormalizedP95 / byPredictor.tensor9.worstNormalizedP95;
  const checks = {
    absolute: byPredictor.lSimplex5.everyGreen,
    materialImprovement: materialRatio <= LIMITS.materialRatio,
    nearTensorCeiling: byPredictor.lSimplex5.worstNormalizedP95 <= 1
      && byPredictor.tensor9.worstNormalizedP95 <= 1
      ? true
      : simplexToTensorRatio <= LIMITS.simplexToTensorRatio,
    continuity: continuityMaximum <= LIMITS.continuity,
  };
  const verdict = Object.values(checks).every(Boolean) ? 'GREEN_FOR_EXPANSION' : 'RED';

  const recipe = {
    version: VERSION,
    sourceSha256: sha256(sourceBytes),
    scriptSha256: sha256(readFileSync(fileURLToPath(import.meta.url))),
    thresholdSha256: sha256(readFileSync(THRESHOLDS)),
    resolution: RESOLUTION,
    spatialPitch: SPATIAL_PITCH,
    spatialVerticesPerAxis: RESOLUTION / SPATIAL_PITCH,
    referenceY: REFERENCE_Y,
    radii: RADII,
    cells: CELLS,
    directionInteriors: DIRECTION_INTERIORS,
    sharedDiagonalSiteByTriangle: [
      { triangle: 0, bary: [0.5, 0, 0.5] },
      { triangle: 1, bary: [0.5, 0.5, 0] },
    ],
    limits: LIMITS,
  };
  const recipeHash = createHash('sha256').update(canonical(recipe)).digest('hex');
  const output = resolve(
    WORKSPACE,
    'data/work/groundcover-candidate-l-targeted-fe',
    recipe.sourceSha256.slice(0, 16),
    recipeHash.slice(0, 16),
  );
  const qaRoot = resolve(output, 'qa');
  mkdirSync(qaRoot, { recursive: true });
  qaCandidates.sort((left, right) => right.normalized - left.normalized);
  const qaIndex: Record<string, unknown>[] = [];
  for (let index = 0; index < Math.min(4, qaCandidates.length); index++) {
    const candidate = qaCandidates[index]!;
    const file = `${String(index + 1).padStart(3, '0')}-${candidate.key}.png`;
    const path = resolve(qaRoot, file);
    await writeQa(path, candidate.truth, candidate.predictions);
    qaIndex.push({ file, sha256: sha256(readFileSync(path)), interpretation: 'truth | K-same-q | L-simplex5 | tensor9', key: candidate.key });
  }
  writeFileSync(resolve(qaRoot, 'index.json'), `${JSON.stringify({ schema: `${VERSION}-qa`, images: qaIndex }, null, 2)}\n`);
  const report = {
    schema: VERSION,
    verdict,
    checks,
    recipe,
    bvh: bvh.metrics,
    byPredictor,
    ratios: { materialRatio, simplexToTensorRatio },
    continuity: { maximum: continuityMaximum, cases: continuity },
    scores,
  };
  writeFileSync(resolve(output, 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
  writeFileSync(resolve(output, 'SUMMARY.md'), [
    '# Candidate L targeted 4D FE gate',
    '',
    `Verdict: **${verdict}**`,
    '',
    `- K same-q worst normalized p95: ${byPredictor.kSameQ.worstNormalizedP95}`,
    `- L simplex5 worst normalized p95: ${byPredictor.lSimplex5.worstNormalizedP95}`,
    `- tensor9 worst normalized p95: ${byPredictor.tensor9.worstNormalizedP95}`,
    `- L/K ratio: ${materialRatio}`,
    `- L/tensor ratio: ${simplexToTensorRatio}`,
    `- continuity maximum: ${continuityMaximum}`,
    '',
    'See report.json for every representative case and qa/ for the worst images.',
  ].join('\n'));
  console.log(JSON.stringify({ output, verdict, checks, byPredictor, ratios: { materialRatio, simplexToTensorRatio }, continuityMaximum }, null, 2));
}

await main();
