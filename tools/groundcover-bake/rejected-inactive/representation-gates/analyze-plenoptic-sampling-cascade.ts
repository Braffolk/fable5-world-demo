/**
 * Offline-only actual-source gate for a fixed plenoptic sampling cascade.
 *
 * The candidate live path is one analytically selected level and one complete
 * record read. All loops, ray cones, BVH traversal, and record construction in
 * this file are bake/evaluation work and are not proposed runtime operations.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';
import {
  TriangleBvh,
  decodeOwnedProfileGeometry,
  type CensusVec3,
  type DecodedOwnedProfileGeometry,
} from '../../ExteriorClosureCensus';
import { periodicNearestSuccessor, type OriginAwareTruthHit } from '../../OriginAwareRayTruth';

const WORKSPACE = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const SOURCE = resolve(WORKSPACE, 'src/assets/groundcover/calamagrostis-canescens.gcrp');
const SOURCE_SHA = '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const CAP = 51_121_152;
const RECORD_BYTES = 8;
const PIXEL_ANGLE = 55 * Math.PI / 180 / 2160;
const WIDTH = 12;
const HEIGHT = 8;
const FRAMES = [0, 0.001, 0.0025, 0.0045] as const;
const ELEVATIONS = [0.1, 1, 5, 15, 35, 75, 90] as const;
const TAU = Math.PI * 2;
const MICRO = [
  [-1, -1], [0, -1], [1, -1],
  [-1, 0], [0, 0], [1, 0],
  [-1, 1], [0, 1], [1, 1],
] as const;
const LEVELS = [
  { phase: 255, angularSide: 4 },
  { phase: 160, angularSide: 7 },
  { phase: 96, angularSide: 11 },
  { phase: 56, angularSide: 17 },
  { phase: 32, angularSide: 26 },
  { phase: 18, angularSide: 40 },
] as const;
const PANICLE_FAMILIES = new Set([3, 4, 5, 6]);
const PALETTE = [
  [0.10, 0.27, 0.055], [0.37, 0.51, 0.13], [0.43, 0.53, 0.20], [0.31, 0.24, 0.075],
  [0.48, 0.31, 0.24], [0.57, 0.35, 0.39], [0.88, 0.80, 0.70], [0.42, 0.10, 0.34],
] as const;

interface Attributes { view: DataView; vertexOffset: number }
interface Sample { t: number; point: CensusVec3; height: number; color: CensusVec3; family: number }
interface ConeRecord {
  coverage: number;
  premul: CensusVec3;
  panicleMass: number;
  representative: Sample | null;
}
interface DirectionCell { direction: CensusVec3; azimuth: number; y: number; ix: number; iy: number }
interface LevelAddress {
  level: number; phaseX: number; phaseZ: number; direction: DirectionCell; key: string;
  canonicalTop: CensusVec3; footprint: number; angularRadius: number;
}
interface Pixel {
  elevation: number; frame: number; x: number; y: number; origin: CensusVec3; direction: CensusVec3;
  distanceToTop: number; footprint: number; truth: ConeRecord; predicted: ConeRecord; address: LevelAddress;
  positionError: number | null; coverageError: number; radianceError: number; panicleError: number;
}

function sha(bytes: Uint8Array | string): string { return createHash('sha256').update(bytes).digest('hex'); }
function canonical(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonical(record[key])}`).join(',')}}`;
}
function norm(v: CensusVec3): CensusVec3 {
  const n = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / n, v[1] / n, v[2] / n];
}
function dist(a: CensusVec3, b: CensusVec3): number { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
function wrap01(v: number): number { return v - Math.floor(v); }
function q(values: readonly number[]): Record<string, number | null> {
  if (!values.length) return { p50: null, p95: null, p99: null, maximum: null };
  const s = [...values].sort((a, b) => a - b);
  const at = (f: number): number => s[Math.floor((s.length - 1) * f)]!;
  return { p50: at(.5), p95: at(.95), p99: at(.99), maximum: s.at(-1)! };
}
function attributes(bytes: Uint8Array): Attributes {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return { view, vertexOffset: view.getUint32(84, true) };
}
function vertexColor(a: Attributes, vertex: number): CensusVec3 {
  const o = a.vertexOffset + vertex * 16;
  return [a.view.getUint16(o + 6, true) / 65535, a.view.getUint16(o + 8, true) / 65535, a.view.getUint16(o + 10, true) / 65535];
}
function family(color: CensusVec3): number {
  let winner = 0;
  let best = Infinity;
  for (let i = 0; i < PALETTE.length; i++) {
    const p = PALETTE[i]!;
    const d = (color[0] - p[0]) ** 2 + (color[1] - p[1]) ** 2 + (color[2] - p[2]) ** 2;
    if (d < best) { best = d; winner = i; }
  }
  return winner;
}
function surface(
  g: DecodedOwnedProfileGeometry, a: Attributes, origin: CensusVec3, direction: CensusVec3, hit: OriginAwareTruthHit,
): Sample {
  const point: CensusVec3 = [origin[0] + direction[0] * hit.t, origin[1] + direction[1] * hit.t, origin[2] + direction[2] * hit.t];
  const tri = hit.triangleId * 3;
  const ids = [g.triangles[tri]!, g.triangles[tri + 1]!, g.triangles[tri + 2]!] as const;
  const color = ids.map((id) => vertexColor(a, id)).reduce<CensusVec3>((sum, c) =>
    [sum[0] + c[0] / 3, sum[1] + c[1] / 3, sum[2] + c[2] / 3], [0, 0, 0]);
  return { t: hit.t, point, height: point[1], color, family: family(color) };
}
function trace(g: DecodedOwnedProfileGeometry, bvh: TriangleBvh, a: Attributes, origin: CensusVec3, d: CensusVec3): Sample | null {
  if (d[1] >= -1e-12) return null;
  const horizon = (origin[1] - g.bounds.min[1] + 1e-8) / -d[1];
  const hit = periodicNearestSuccessor(g, bvh, origin, d, horizon, 0);
  return hit ? surface(g, a, origin, d, hit) : null;
}
function basis(d: CensusVec3): readonly [CensusVec3, CensusVec3] {
  const right = norm(Math.abs(d[1]) < .999 ? [-d[2], 0, d[0]] : [1, 0, 0]);
  const up = norm([right[1] * d[2] - right[2] * d[1], right[2] * d[0] - right[0] * d[2], right[0] * d[1] - right[1] * d[0]]);
  return [right, up];
}
function perturbed(d: CensusVec3, x: number, y: number, radius: number): CensusVec3 {
  const [r, u] = basis(d);
  return norm([d[0] + (r[0] * x + u[0] * y) * radius, d[1] + (r[1] * x + u[1] * y) * radius, d[2] + (r[2] * x + u[2] * y) * radius]);
}
function cone(
  g: DecodedOwnedProfileGeometry, bvh: TriangleBvh, a: Attributes, top: CensusVec3, d: CensusVec3,
  phaseRadius: number, angularRadius: number,
): ConeRecord {
  const [r] = basis(d);
  const hits: Sample[] = [];
  let premul: CensusVec3 = [0, 0, 0];
  let panicles = 0;
  for (const [mx, my] of MICRO) {
    const dd = perturbed(d, mx, my, angularRadius);
    const origin: CensusVec3 = [top[0] + r[0] * mx * phaseRadius, g.topH, top[2] + r[2] * mx * phaseRadius];
    const hit = trace(g, bvh, a, origin, dd);
    if (!hit) continue;
    hits.push(hit);
    premul = [premul[0] + hit.color[0] / MICRO.length, premul[1] + hit.color[1] / MICRO.length, premul[2] + hit.color[2] / MICRO.length];
    if (PANICLE_FAMILIES.has(hit.family)) panicles++;
  }
  hits.sort((x, y) => x.height - y.height);
  return {
    coverage: hits.length / MICRO.length,
    premul,
    panicleMass: panicles / MICRO.length,
    representative: hits.length ? hits[Math.floor((hits.length - 1) / 2)]! : null,
  };
}

function topIntersection(origin: CensusVec3, d: CensusVec3, topH: number): CensusVec3 {
  const t = (topH - origin[1]) / d[1];
  return [origin[0] + d[0] * t, topH, origin[2] + d[2] * t];
}
function selectLevel(g: DecodedOwnedProfileGeometry, footprint: number): number {
  let best = 0;
  let score = Infinity;
  for (let i = 0; i < LEVELS.length; i++) {
    const pitch = Math.max(g.tileSizeX, g.tileSizeZ) / LEVELS[i]!.phase;
    const s = Math.abs(Math.log(Math.max(1e-12, footprint) / pitch));
    if (s < score) { score = s; best = i; }
  }
  return best;
}
function quantDirection(d: CensusVec3, side: number): DirectionCell {
  let az = Math.atan2(d[2], d[0]);
  if (az < 0) az += TAU;
  const y = Math.max(0, Math.min(1, -d[1]));
  const ix = Math.min(side - 1, Math.floor(az / TAU * side));
  const iy = Math.min(side - 1, Math.floor(y * side));
  const qa = (ix + .5) / side * TAU;
  const qy = (iy + .5) / side;
  const radial = Math.sqrt(Math.max(0, 1 - qy * qy));
  return { direction: [Math.cos(qa) * radial, -qy, Math.sin(qa) * radial], azimuth: qa, y: qy, ix, iy };
}
function address(g: DecodedOwnedProfileGeometry, origin: CensusVec3, d: CensusVec3, footprint: number): LevelAddress {
  const level = selectLevel(g, footprint);
  const spec = LEVELS[level]!;
  const top = topIntersection(origin, d, g.topH);
  const px = Math.floor(wrap01((top[0] - g.tileOriginX) / g.tileSizeX) * spec.phase);
  const pz = Math.floor(wrap01((top[2] - g.tileOriginZ) / g.tileSizeZ) * spec.phase);
  const direction = quantDirection(d, spec.angularSide);
  const canonicalTop: CensusVec3 = [
    g.tileOriginX + (px + .5) / spec.phase * g.tileSizeX,
    g.topH,
    g.tileOriginZ + (pz + .5) / spec.phase * g.tileSizeZ,
  ];
  const phasePitch = Math.max(g.tileSizeX, g.tileSizeZ) / spec.phase;
  const angularRadius = Math.max(Math.PI / spec.angularSide, .5 / spec.angularSide);
  return { level, phaseX: px, phaseZ: pz, direction, key: `${level}:${px}:${pz}:${direction.ix}:${direction.iy}`, canonicalTop, footprint: phasePitch, angularRadius };
}
function cameraDirection(elevation: number, x: number, y: number): CensusVec3 {
  const e = elevation * Math.PI / 180;
  const az = 0.73;
  const forward: CensusVec3 = [Math.cos(e) * Math.cos(az), -Math.sin(e), Math.cos(e) * Math.sin(az)];
  const right: CensusVec3 = [-Math.sin(az), 0, Math.cos(az)];
  const up: CensusVec3 = [Math.sin(e) * Math.cos(az), Math.cos(e), Math.sin(e) * Math.sin(az)];
  const pitch = Math.min(PIXEL_ANGLE, e / Math.max(8, HEIGHT) * .75);
  const sx = x - (WIDTH - 1) / 2;
  const sy = (HEIGHT - 1) / 2 - y;
  return norm([forward[0] + right[0] * sx * pitch + up[0] * sy * pitch, forward[1] + right[1] * sx * pitch + up[1] * sy * pitch, forward[2] + right[2] * sx * pitch + up[2] * sy * pitch]);
}

function connectedFans(pixels: readonly Pixel[], threshold: number): number[] {
  const seen = new Uint8Array(pixels.length);
  const widths: number[] = [];
  for (let seed = 0; seed < pixels.length; seed++) {
    if (seen[seed] || !(pixels[seed]!.positionError !== null && pixels[seed]!.positionError! >= threshold)) continue;
    const stack = [seed]; seen[seed] = 1; const points: CensusVec3[] = [];
    while (stack.length) {
      const i = stack.pop()!; const p = pixels[i]!;
      if (p.predicted.representative) points.push(p.predicted.representative.point);
      const x = i % WIDTH; const y = Math.floor(i / WIDTH);
      for (const n of [x ? i - 1 : -1, x + 1 < WIDTH ? i + 1 : -1, y ? i - WIDTH : -1, y + 1 < HEIGHT ? i + WIDTH : -1]) {
        if (n < 0 || seen[n]) continue;
        const c = pixels[n]!;
        if (c.positionError !== null && c.positionError >= threshold) { seen[n] = 1; stack.push(n); }
      }
    }
    if (points.length > 1) {
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (const p of points) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k]!, p[k]!); hi[k] = Math.max(hi[k]!, p[k]!); }
      widths.push(Math.hypot(hi[0]! - lo[0]!, hi[1]! - lo[1]!, hi[2]! - lo[2]!));
    }
  }
  return widths;
}
function rgb(c: CensusVec3): readonly [number, number, number] {
  return c.map((v) => Math.round(Math.max(0, Math.min(1, Math.pow(v, 1 / 2.2))) * 255)) as unknown as readonly [number, number, number];
}
async function qa(path: string, groups: readonly Pixel[][]): Promise<void> {
  const panelW = WIDTH * 3, panelH = HEIGHT * groups.length;
  const out = new Uint8Array(panelW * panelH * 3);
  groups.forEach((pixels, row) => pixels.forEach((p) => {
    const truth = rgb(p.truth.premul), pred = rgb(p.predicted.premul);
    const err: readonly [number, number, number] = p.positionError === null ? [255, 0, 255]
      : p.positionError < .05 ? [20, 170, 30] : p.positionError < .25 ? [230, 210, 20] : p.positionError < 1 ? [240, 90, 10] : [220, 10, 15];
    for (const [panel, color] of [[0, truth], [1, pred], [2, err]] as const) {
      const xx = panel * WIDTH + p.x, yy = row * HEIGHT + p.y, o = (yy * panelW + xx) * 3;
      out[o] = color[0]; out[o + 1] = color[1]; out[o + 2] = color[2];
    }
  }));
  await sharp(out, { raw: { width: panelW, height: panelH, channels: 3 } }).resize(panelW * 10, panelH * 10, { kernel: 'nearest' }).png().toFile(path);
}

const bytes = readFileSync(SOURCE);
if (sha(bytes) !== SOURCE_SHA) throw new Error('immutable source hash mismatch');
const g = decodeOwnedProfileGeometry(bytes), a = attributes(bytes), bvh = TriangleBvh.build(g, 8);
const records = LEVELS.map((l) => (l.phase + 2) ** 2 * l.angularSide ** 2);
const residentBytes = records.reduce((s, n) => s + n, 0) * RECORD_BYTES;
if (residentBytes > CAP) throw new Error(`cascade exceeds resident cap: ${residentBytes}`);
const configuration = { schema: 'laas-plenoptic-sampling-cascade-gate/v1', sourceSha: SOURCE_SHA, recordBytes: RECORD_BYTES, levels: LEVELS, residentBytes, elevations: ELEVATIONS, frames: FRAMES, pixelAngleRadians: PIXEL_ANGLE, cameraGrid: [WIDTH, HEIGHT], microRays: MICRO.length };
const recipe = sha(canonical(configuration));
const root = resolve(WORKSPACE, `data/work/groundcover-plenoptic-cascade/${SOURCE_SHA.slice(0, 16)}/${recipe.slice(0, 16)}`);
mkdirSync(resolve(root, 'qa'), { recursive: true });
const cache = new Map<string, ConeRecord>();
const pixels: Pixel[] = [];
const byElevationFrames = new Map<number, Pixel[][]>();
for (const elevation of ELEVATIONS) {
  const frames: Pixel[][] = [];
  for (let frame = 0; frame < FRAMES.length; frame++) {
    const framePixels: Pixel[] = [];
    for (let y = 0; y < HEIGHT; y++) for (let x = 0; x < WIDTH; x++) {
      const d = cameraDirection(elevation, x, y);
      const origin: CensusVec3 = [g.tileOriginX + g.tileSizeX * .314159 + FRAMES[frame]!, g.topH + .4, g.tileOriginZ + g.tileSizeZ * .271828];
      const distanceToTop = .4 / -d[1];
      const footprint = 2 * distanceToTop * Math.tan(PIXEL_ANGLE / 2);
      const top = topIntersection(origin, d, g.topH);
      const truth = cone(g, bvh, a, top, d, 0, PIXEL_ANGLE / 2);
      const ad = address(g, origin, d, footprint);
      let predicted = cache.get(ad.key);
      if (!predicted) {
        predicted = cone(g, bvh, a, ad.canonicalTop, ad.direction.direction, ad.footprint / 2, ad.angularRadius / 2);
        cache.set(ad.key, predicted);
      }
      const tr = truth.representative, pr = predicted.representative;
      const predictedPoint = pr ? [top[0] + d[0] * ((g.topH - pr.height) / -d[1]), pr.height, top[2] + d[2] * ((g.topH - pr.height) / -d[1])] as CensusVec3 : null;
      const positionError = tr && predictedPoint ? dist(tr.point, predictedPoint) : tr || pr ? Infinity : 0;
      const radianceError = Math.hypot(truth.premul[0] - predicted.premul[0], truth.premul[1] - predicted.premul[1], truth.premul[2] - predicted.premul[2]);
      const p: Pixel = { elevation, frame, x, y, origin, direction: d, distanceToTop, footprint, truth, predicted, address: ad, positionError, coverageError: Math.abs(truth.coverage - predicted.coverage), radianceError, panicleError: Math.abs(truth.panicleMass - predicted.panicleMass) };
      pixels.push(p); framePixels.push(p);
    }
    frames.push(framePixels);
    console.error(`[plenoptic-cascade] ${elevation}deg frame ${frame + 1}/${FRAMES.length}, cache ${cache.size}`);
  }
  byElevationFrames.set(elevation, frames);
}
const finitePosition = pixels.map((p) => p.positionError).filter((v): v is number => v !== null && Number.isFinite(v));
const temporalExcess: number[] = [];
let stableTruthPairs = 0, unforcedClassChanges = 0;
for (const frames of byElevationFrames.values()) for (let f = 1; f < frames.length; f++) for (let i = 0; i < frames[f]!.length; i++) {
  const prev = frames[f - 1]![i]!, next = frames[f]![i]!;
  const truthDelta = Math.hypot(next.truth.premul[0] - prev.truth.premul[0], next.truth.premul[1] - prev.truth.premul[1], next.truth.premul[2] - prev.truth.premul[2]);
  const predDelta = Math.hypot(next.predicted.premul[0] - prev.predicted.premul[0], next.predicted.premul[1] - prev.predicted.premul[1], next.predicted.premul[2] - prev.predicted.premul[2]);
  temporalExcess.push(Math.max(0, predDelta - truthDelta));
  const tf0 = prev.truth.representative?.family ?? -1, tf1 = next.truth.representative?.family ?? -1;
  if (tf0 === tf1) { stableTruthPairs++; if ((prev.predicted.representative?.family ?? -1) !== (next.predicted.representative?.family ?? -1)) unforcedClassChanges++; }
}
const elevationReport: Record<string, unknown> = {};
for (const elevation of ELEVATIONS) {
  const subset = pixels.filter((p) => p.elevation === elevation);
  const levelHistogram: Record<string, number> = {};
  subset.forEach((p) => { levelHistogram[String(p.address.level)] = (levelHistogram[String(p.address.level)] ?? 0) + 1; });
  elevationReport[String(elevation)] = {
    distanceToTopMetres: q(subset.map((p) => p.distanceToTop)), footprintMetres: q(subset.map((p) => p.footprint)), levelHistogram,
    positionErrorMetres: q(subset.map((p) => p.positionError).filter((v): v is number => v !== null && Number.isFinite(v))),
    disagreementFraction: subset.filter((p) => !Number.isFinite(p.positionError ?? 0)).length / subset.length,
    coverageError: q(subset.map((p) => p.coverageError)), radianceError: q(subset.map((p) => p.radianceError)), panicleMassError: q(subset.map((p) => p.panicleError)),
    connectedWrongViewWidthsAt025m: q(byElevationFrames.get(elevation)!.flatMap((frame) => connectedFans(frame, .25))),
  };
}
const angularBounds = LEVELS.map((l, i) => {
  const maxAngularCell = Math.hypot(Math.PI / l.angularSide, .5 / l.angularSide);
  const representativeDistance = pixels.filter((p) => p.address.level === i).map((p) => p.distanceToTop);
  return { level: i, phasePitchMetres: Math.max(g.tileSizeX, g.tileSizeZ) / l.phase, directions: l.angularSide ** 2, conservativeAngularRadiusRadians: maxAngularCell / 2, distance: q(representativeDistance), rDeltaThetaMetres: q(representativeDistance.map((r) => r * maxAngularCell / 2)) };
});
const report = {
  configuration, sourceBytes: bytes.byteLength, bvh: bvh.metrics, cachedOracleRecords: cache.size,
  memory: { recordsByLevel: records, residentBytes, capBytes: CAP, bytesRemaining: CAP - residentBytes },
  lineSpace: {
    coordinates: 'fixed top-Y-plane periodic phase plus analytic hemispherical direction cell; origin invariant for every non-horizontal oriented line',
    errorBound: 'transverse <= phasePitch/2 + range*angularCellRadius; ray-cone filtering changes the signal bandwidth but not this address displacement',
    angularBounds,
    exactHorizontal: 'FAIL: direction is representable but top-Y-plane phase is at infinity; a second projective chart or pointed-line successor coordinate is required',
  },
  aggregate: {
    rays: pixels.length, positionErrorMetres: q(finitePosition), hitMissDisagreementFraction: pixels.filter((p) => !Number.isFinite(p.positionError ?? 0)).length / pixels.length,
    coverageError: q(pixels.map((p) => p.coverageError)), radianceError: q(pixels.map((p) => p.radianceError)), panicleMassError: q(pixels.map((p) => p.panicleError)),
    temporalRadianceExcess: q(temporalExcess), stableTruthPairs, unforcedRepresentativeFamilyChanges: unforcedClassChanges, unforcedRepresentativeFamilyChangeFraction: unforcedClassChanges / Math.max(1, stableTruthPairs),
  },
  byElevation: elevationReport,
  contractAssessment: {
    nearCrispGeometry: 'tested by representative-event world error at steep/near views; pass requires p95 <= 0.02 m',
    cameraInsideSuccessor: 'FAIL by representation: exterior cone record has no pointed-line origin/successor phase',
    exactHorizontal: 'FAIL by fixed two-Y-plane chart singularity',
    terrainDepth: 'a selected medoid height can write exact live-ray depth, but the selected wrong event produces the measured range-amplified error; radiance-only records collapse to a textured carrier plane',
    multiSpeciesMoss: 'only one offline-unioned marked community can occupy the one read; independent species queries are forbidden. Overlap adds successor/disocclusion entropy and is not solved by this single-species failure.',
  },
  decision: finitePosition.length && (q(finitePosition).p95 as number) <= .02 && unforcedClassChanges === 0 ? 'provisional-pass' : 'reject-full-carrier',
};
const qaPath = resolve(root, 'qa/001-truth-prediction-event-error.png');
await qa(qaPath, ELEVATIONS.map((e) => byElevationFrames.get(e)![0]!));
writeFileSync(resolve(root, 'metrics.json'), `${JSON.stringify(report, null, 2)}\n`);
writeFileSync(resolve(root, 'index.json'), `${JSON.stringify({ schema: 'laas-groundcover-qa-index/v1', sourceSha256: SOURCE_SHA, recipeSha256: recipe, files: [{ file: 'qa/001-truth-prediction-event-error.png', sha256: sha(readFileSync(qaPath)), width: WIDTH * 3 * 10, height: HEIGHT * ELEVATIONS.length * 10, interpretation: 'rows 0.1/1/5/15/35/75/90 degrees; truth prefiltered radiance, cascade prefiltered radiance, representative-event error' }, { file: 'metrics.json', sha256: sha(readFileSync(resolve(root, 'metrics.json'))) }] }, null, 2)}\n`);
console.log(JSON.stringify({ root, recipe, residentBytes, decision: report.decision, aggregate: report.aggregate }, null, 2));
