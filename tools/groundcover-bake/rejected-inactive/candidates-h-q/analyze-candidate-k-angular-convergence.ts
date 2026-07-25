/** Offline Candidate-K premise audit: angular FE convergence and reference-plane ablation. */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';
import { TriangleBvh, decodeOwnedProfileGeometry, type DecodedOwnedProfileGeometry } from '../../ExteriorClosureCensus';
import {
  renderPointPage,
  scorePage,
  sha256,
  sourceAttributes,
  toroidalBoxFilter,
  type DirectionSpec,
  type FieldPage,
  type SourceAttributes,
} from './gate-candidate-k-heldout-fe';

const WORKSPACE = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const SOURCE = resolve(WORKSPACE, 'src/assets/groundcover/calamagrostis-canescens.gcrp');
const SCREEN_REPORT = resolve(
  WORKSPACE,
  'data/work/groundcover-candidate-k-heldout-fe/e3e0a4175b151b89/60a8e06b5f69e390/report.json',
);
const THRESHOLD_DOC = resolve(
  WORKSPACE,
  'docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-CANDIDATE-K-HELDOUT-THRESHOLDS-FROZEN.md',
);
const VERSION = 'candidate-k-angular-convergence-v1';
const RESOLUTION = 64;
const SOURCE_SIGMAS = [4, 16] as const;
const LEVELS = [0, 1, 2, 3] as const;
const AZIMUTH_CELL_WIDTH = 22.5;
const MIDDLE_REFERENCE_Y = 0.49255;
const LIMITS = { coverageP95: 0.08, rgbP95: 0.06 } as const;
const ELEVATION_FAMILIES = [
  { key: 'horizon-low', low: 0.25, high: 2, screenElevation: 1 },
  { key: 'low-2-5', low: 2, high: 5, screenElevation: 3.5 },
  { key: 'low-5-10', low: 5, high: 10, screenElevation: 7.5 },
  { key: 'standing-10-18', low: 10, high: 18, screenElevation: 14 },
  { key: 'standing-18-30', low: 18, high: 30, screenElevation: 24 },
  { key: 'mid-30-55', low: 30, high: 55, screenElevation: 45 },
  { key: 'high-55-75', low: 55, high: 75, screenElevation: 65 },
  { key: 'pole-cap', low: 75, high: 90, screenElevation: 82 },
] as const;

interface ScreenEntry {
  spec: { elevationDegrees: number; azimuthDegrees: number };
  radius: number;
  unlimited: { coverage: { p95: number }; premulRgb: { p95: number } };
}
interface Representative {
  family: typeof ELEVATION_FAMILIES[number];
  rank: 'worst' | 'median';
  azimuthCenter: number;
  screenScore: number;
}
interface ConvergenceResult {
  family: string;
  rank: string;
  axis: 'azimuth' | 'elevation';
  scale: number;
  level: number;
  widthDegrees: number;
  midpoint: { elevationDegrees: number; azimuthDegrees: number };
  endpoints: readonly [{ elevationDegrees: number; azimuthDegrees: number }, { elevationDegrees: number; azimuthDegrees: number }];
  coverageP95: number;
  rgbP95: number;
  normalizedP95: number;
  greenP95: boolean;
  connected: number;
}

function canonical(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  const object = value as Record<string, unknown>;
  return `{${Object.keys(object).sort().map((key) => `${JSON.stringify(key)}:${canonical(object[key])}`).join(',')}}`;
}

function meanPages(left: FieldPage, right: FieldPage): FieldPage {
  const rgba = new Float32Array(left.rgba.length);
  for (let index = 0; index < rgba.length; index++) rgba[index] = 0.5 * (left.rgba[index]! + right.rgba[index]!);
  return { rgba };
}

function key(reference: string, elevation: number, azimuth: number): string {
  return `${reference}:${elevation.toFixed(9)}:${(((azimuth % 360) + 360) % 360).toFixed(9)}`;
}

function selectRepresentatives(screen: { results: { middle: { heldout: ScreenEntry[] } } }): Representative[] {
  const entries = screen.results.middle.heldout;
  const result: Representative[] = [];
  for (const family of ELEVATION_FAMILIES) {
    const candidates = entries.filter((entry) =>
      entry.radius === 4
      && entry.spec.elevationDegrees === family.screenElevation
      && Math.abs((entry.spec.azimuthDegrees / AZIMUTH_CELL_WIDTH) % 1 - 0.5) < 1e-8,
    ).map((entry) => ({
      azimuthCenter: entry.spec.azimuthDegrees,
      score: Math.max(entry.unlimited.coverage.p95 / LIMITS.coverageP95, entry.unlimited.premulRgb.p95 / LIMITS.rgbP95),
    })).sort((left, right) => left.score - right.score);
    if (candidates.length < 2) throw new Error(`not enough midpoint candidates for ${family.key}`);
    const median = candidates[Math.floor((candidates.length - 1) / 2)]!;
    const worst = candidates.at(-1)!;
    result.push({ family, rank: 'median', azimuthCenter: median.azimuthCenter, screenScore: median.score });
    result.push({ family, rank: 'worst', azimuthCenter: worst.azimuthCenter, screenScore: worst.score });
  }
  return result;
}

function empiricalRequirement(results: readonly ConvergenceResult[], axis: 'azimuth' | 'elevation', scale: number): Record<string, unknown> {
  const levels = LEVELS.map((level) => {
    const values = results.filter((entry) => entry.axis === axis && entry.scale === scale && entry.level === level);
    return {
      level,
      factor: 2 ** level,
      worstCoverageP95: Math.max(...values.map((entry) => entry.coverageP95)),
      worstRgbP95: Math.max(...values.map((entry) => entry.rgbP95)),
      worstNormalizedP95: Math.max(...values.map((entry) => entry.normalizedP95)),
      everyRepresentativeGreen: values.every((entry) => entry.greenP95),
    };
  });
  const firstGreen = levels.find((entry) => entry.everyRepresentativeGreen)?.level ?? null;
  const last = levels.at(-1)!;
  const prior = levels.at(-2)!;
  const empiricalSlope = prior.worstNormalizedP95 > last.worstNormalizedP95 && last.worstNormalizedP95 > 0
    ? Math.log2(prior.worstNormalizedP95 / last.worstNormalizedP95)
    : null;
  const extraHalvings = firstGreen === null && empiricalSlope !== null && empiricalSlope > 0
    ? Math.max(0, Math.ceil(Math.log2(last.worstNormalizedP95) / empiricalSlope))
    : null;
  return {
    axis,
    scale,
    levels,
    firstEmpiricalGreenLevel: firstGreen,
    extrapolatedAdditionalHalvingsAfterLevel3: extraHalvings,
    extrapolatedRequiredLevel: firstGreen ?? (extraHalvings === null ? null : 3 + extraHalvings),
    extrapolationSlopeOrder: empiricalSlope,
  };
}

async function chart(path: string, title: string, requirements: readonly Record<string, unknown>[]): Promise<void> {
  const width = 960;
  const height = 520;
  const colors = ['#ff5f56', '#2bc4ff', '#ffd866', '#7bd88f'];
  const polylines: string[] = [];
  let color = 0;
  for (const requirement of requirements) {
    const levels = requirement.levels as { level: number; worstNormalizedP95: number }[];
    const points = levels.map((entry) => {
      const x = 100 + entry.level * 250;
      const y = 450 - Math.min(8, entry.worstNormalizedP95) / 8 * 380;
      return `${x},${y}`;
    }).join(' ');
    polylines.push(`<polyline fill="none" stroke="${colors[color++ % colors.length]}" stroke-width="5" points="${points}"/>`);
  }
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">
  <rect width="100%" height="100%" fill="#11151b"/><text x="40" y="40" fill="white" font-size="22">${title}</text>
  <line x1="100" y1="402.5" x2="850" y2="402.5" stroke="#ffffff" stroke-dasharray="8 8"/>
  <text x="45" y="410" fill="#ddd">1x</text>${polylines.join('')}
  <text x="90" y="490" fill="#ddd">current</text><text x="335" y="490" fill="#ddd">1/2</text>
  <text x="585" y="490" fill="#ddd">1/4</text><text x="835" y="490" fill="#ddd">1/8</text></svg>`;
  await sharp(Buffer.from(svg)).png().toFile(path);
}

async function main(): Promise<void> {
  const sourceBytes = readFileSync(SOURCE);
  const geometry = decodeOwnedProfileGeometry(sourceBytes);
  const attributes = sourceAttributes(sourceBytes);
  const bvh = TriangleBvh.build(geometry);
  const screen = JSON.parse(readFileSync(SCREEN_REPORT, 'utf8')) as { results: { middle: { heldout: ScreenEntry[] } } };
  const representatives = selectRepresentatives(screen);
  const cache = new Map<string, Map<number, FieldPage>>();
  const page = (reference: string, referenceY: number, elevationDegrees: number, azimuthDegrees: number): Map<number, FieldPage> => {
    const cacheKey = key(reference, elevationDegrees, azimuthDegrees);
    const found = cache.get(cacheKey);
    if (found) return found;
    console.log(`[candidate-k-convergence] ${cache.size + 1} ${cacheKey}`);
    const spec: DirectionSpec = { key: cacheKey, elevationDegrees, azimuthDegrees, kind: 'heldout' };
    const point = renderPointPage(geometry, bvh, attributes, RESOLUTION, referenceY, spec);
    const filtered = new Map<number, FieldPage>();
    for (const sigma of SOURCE_SIGMAS) filtered.set(sigma, toroidalBoxFilter(point, RESOLUTION, sigma * RESOLUTION / 256));
    cache.set(cacheKey, filtered);
    return filtered;
  };

  const results: ConvergenceResult[] = [];
  for (const representative of representatives) {
    const elevationCenter = 0.5 * (representative.family.low + representative.family.high);
    for (const axis of ['azimuth', 'elevation'] as const) {
      const baseWidth = axis === 'azimuth'
        ? AZIMUTH_CELL_WIDTH
        : representative.family.high - representative.family.low;
      for (const level of LEVELS) {
        const width = baseWidth / 2 ** level;
        const endpoints = axis === 'azimuth'
          ? [
            { elevationDegrees: elevationCenter, azimuthDegrees: representative.azimuthCenter - width / 2 },
            { elevationDegrees: elevationCenter, azimuthDegrees: representative.azimuthCenter + width / 2 },
          ] as const
          : [
            { elevationDegrees: elevationCenter - width / 2, azimuthDegrees: representative.azimuthCenter },
            { elevationDegrees: elevationCenter + width / 2, azimuthDegrees: representative.azimuthCenter },
          ] as const;
        const midpoint = { elevationDegrees: elevationCenter, azimuthDegrees: representative.azimuthCenter };
        const truth = page('middle', MIDDLE_REFERENCE_Y, midpoint.elevationDegrees, midpoint.azimuthDegrees);
        const left = page('middle', MIDDLE_REFERENCE_Y, endpoints[0].elevationDegrees, endpoints[0].azimuthDegrees);
        const right = page('middle', MIDDLE_REFERENCE_Y, endpoints[1].elevationDegrees, endpoints[1].azimuthDegrees);
        for (const scale of SOURCE_SIGMAS) {
          const score = scorePage(truth.get(scale)!, meanPages(left.get(scale)!, right.get(scale)!), RESOLUTION).score;
          results.push({
            family: representative.family.key,
            rank: representative.rank,
            axis,
            scale,
            level,
            widthDegrees: width,
            midpoint,
            endpoints,
            coverageP95: score.coverage.p95,
            rgbP95: score.premulRgb.p95,
            normalizedP95: Math.max(score.coverage.p95 / LIMITS.coverageP95, score.premulRgb.p95 / LIMITS.rgbP95),
            greenP95: score.coverage.p95 <= LIMITS.coverageP95 && score.premulRgb.p95 <= LIMITS.rgbP95,
            connected: score.largestConnectedFraction,
          });
        }
      }
    }
  }

  const ablationRepresentatives = representatives.filter((entry) =>
    entry.rank === 'worst' && ['horizon-low', 'standing-18-30', 'mid-30-55', 'pole-cap'].includes(entry.family.key),
  );
  const referenceAblation: Record<string, unknown>[] = [];
  for (const representative of ablationRepresentatives) {
    const elevationCenter = 0.5 * (representative.family.low + representative.family.high);
    for (const axis of ['azimuth', 'elevation'] as const) {
      const width = axis === 'azimuth' ? AZIMUTH_CELL_WIDTH : representative.family.high - representative.family.low;
      const endpoints = axis === 'azimuth'
        ? [
          { elevationDegrees: elevationCenter, azimuthDegrees: representative.azimuthCenter - width / 2 },
          { elevationDegrees: elevationCenter, azimuthDegrees: representative.azimuthCenter + width / 2 },
        ] as const
        : [
          { elevationDegrees: elevationCenter - width / 2, azimuthDegrees: representative.azimuthCenter },
          { elevationDegrees: elevationCenter + width / 2, azimuthDegrees: representative.azimuthCenter },
        ] as const;
      for (const reference of [{ key: 'middle', y: MIDDLE_REFERENCE_Y }, { key: 'top', y: geometry.topH }]) {
        const truth = page(reference.key, reference.y, elevationCenter, representative.azimuthCenter);
        const left = page(reference.key, reference.y, endpoints[0].elevationDegrees, endpoints[0].azimuthDegrees);
        const right = page(reference.key, reference.y, endpoints[1].elevationDegrees, endpoints[1].azimuthDegrees);
        for (const scale of SOURCE_SIGMAS) {
          const score = scorePage(truth.get(scale)!, meanPages(left.get(scale)!, right.get(scale)!), RESOLUTION).score;
          referenceAblation.push({
            family: representative.family.key,
            axis,
            scale,
            reference: reference.key,
            coverageP95: score.coverage.p95,
            rgbP95: score.premulRgb.p95,
            normalizedP95: Math.max(score.coverage.p95 / LIMITS.coverageP95, score.premulRgb.p95 / LIMITS.rgbP95),
          });
        }
      }
    }
  }

  const requirements = (['azimuth', 'elevation'] as const).flatMap((axis) =>
    SOURCE_SIGMAS.map((scale) => empiricalRequirement(results, axis, scale)),
  );
  const requiredAzimuthLevel = Math.max(...requirements.filter((entry) => entry.axis === 'azimuth').map((entry) =>
    Number(entry.extrapolatedRequiredLevel ?? Number.POSITIVE_INFINITY)));
  const requiredElevationLevel = Math.max(...requirements.filter((entry) => entry.axis === 'elevation').map((entry) =>
    Number(entry.extrapolatedRequiredLevel ?? Number.POSITIVE_INFINITY)));
  const estimatedPages = Number.isFinite(requiredAzimuthLevel) && Number.isFinite(requiredElevationLevel)
    ? 128 * 2 ** requiredAzimuthLevel * 2 ** requiredElevationLevel
    : null;
  const estimatedScaleMiB = estimatedPages === null ? null : estimatedPages * 128 * 128 * 16 / 1048576;
  const recipe = {
    version: VERSION,
    sourceSha256: sha256(sourceBytes),
    scriptSha256: sha256(readFileSync(fileURLToPath(import.meta.url))),
    thresholdDocSha256: sha256(readFileSync(THRESHOLD_DOC)),
    screenReportSha256: sha256(readFileSync(SCREEN_REPORT)),
    resolution: RESOLUTION,
    sourceSigmas: SOURCE_SIGMAS,
    levels: LEVELS,
    families: ELEVATION_FAMILIES,
    representatives,
    limits: LIMITS,
  };
  const recipeHash = createHash('sha256').update(canonical(recipe)).digest('hex');
  const output = resolve(WORKSPACE, 'data/work/groundcover-candidate-k-angular-convergence', recipe.sourceSha256.slice(0, 16), recipeHash.slice(0, 16));
  mkdirSync(resolve(output, 'qa'), { recursive: true });
  for (const scale of SOURCE_SIGMAS) {
    await chart(
      resolve(output, 'qa', `convergence-sigma${scale}.png`),
      `Candidate K normalized p95 convergence, sigma=${scale}`,
      requirements.filter((entry) => entry.scale === scale),
    );
  }
  const report = {
    schema: VERSION,
    verdict: requirements.every((entry) => entry.firstEmpiricalGreenLevel !== null) ? 'EMPIRICALLY_BOUNDED' : 'NOT_GREEN_BY_1_OVER_8',
    recipe,
    results,
    requirements,
    referenceAblation,
    combinedEstimate: {
      requiredAzimuthLevel,
      requiredElevationLevel,
      estimatedPages,
      estimatedScaleMiB,
      candidateKScaleBudgetMiB: 32,
    },
  };
  writeFileSync(resolve(output, 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify({ output, verdict: report.verdict, requirements, combinedEstimate: report.combinedEstimate }, null, 2));
}

await main();
