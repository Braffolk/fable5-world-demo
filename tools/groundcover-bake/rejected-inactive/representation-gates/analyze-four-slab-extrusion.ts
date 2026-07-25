/**
 * Offline-only, one-attempt feasibility gate for a union of four affine
 * images of clipped parallel mask extrusions. It never writes a runtime asset
 * and deliberately uses exact source-triangle truth plus an optimistic,
 * conservative high-resolution mask projection.
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

const SOURCE = 'src/assets/groundcover/calamagrostis-canescens.gcrp';
const EXPECTED_SOURCE_SHA256 = '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const MASK_RESOLUTION = 512;
const RENDER_WIDTH = 128;
const RENDER_HEIGHT = 160;
const ELEVATIONS = [0.1, 1, 5, 15, 35, 75, 90] as const;
const BAND_FRACTIONS = [0, 0.45, 0.70, 0.86, 1] as const;
const DIRECTIONS = 24;
const STORED_TILE = 220 + 2;
const RECORD_BYTES = 8;
const MIP_FACTOR = 4 / 3;
const PALETTE = [
  { rgb: [0.10, 0.27, 0.055], family: 0, panicle: false },
  { rgb: [0.37, 0.51, 0.13], family: 0, panicle: false },
  { rgb: [0.43, 0.53, 0.20], family: 1, panicle: false },
  { rgb: [0.31, 0.24, 0.075], family: 2, panicle: false },
  { rgb: [0.48, 0.31, 0.24], family: 3, panicle: true },
  { rgb: [0.57, 0.35, 0.39], family: 4, panicle: true },
  { rgb: [0.88, 0.80, 0.70], family: 5, panicle: true },
  { rgb: [0.42, 0.10, 0.34], family: 6, panicle: true },
] as const;
const FAMILY_RGB = [
  [48, 112, 32], [112, 137, 51], [79, 61, 19], [122, 79, 61],
  [145, 89, 99], [224, 204, 179], [107, 26, 87],
] as const;

interface V3 { x: number; y: number; z: number }
interface Band {
  index: number;
  y0: number;
  y1: number;
  yc: number;
  leanX: number;
  leanZ: number;
  u0: number;
  u1: number;
  v0: number;
  v1: number;
  masks: Uint8Array;
  occupied: number;
}
interface Hit { t: number; familyMask: number; cap: boolean; band: number }

function hash(bytes: Uint8Array | string): string {
  return createHash('sha256').update(bytes).digest('hex');
}
function dot(a: CensusVec3, b: CensusVec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function add(a: CensusVec3, b: CensusVec3, s = 1): CensusVec3 {
  return [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s];
}
function familyForTriangle(
  triangleId: number,
  bytes: Uint8Array,
  vertexOffset: number,
  triangleOffset: number,
): number {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const rgb = [0, 0, 0];
  for (let corner = 0; corner < 3; corner++) {
    const vertex = view.getUint32(triangleOffset + triangleId * 16 + corner * 4, true);
    const source = vertexOffset + vertex * 16;
    rgb[0] += view.getUint16(source + 6, true) / (65535 * 3);
    rgb[1] += view.getUint16(source + 8, true) / (65535 * 3);
    rgb[2] += view.getUint16(source + 10, true) / (65535 * 3);
  }
  let best = 0;
  let distance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < PALETTE.length; index++) {
    const p = PALETTE[index]!.rgb;
    const d = (rgb[0]! - p[0]) ** 2 + (rgb[1]! - p[1]) ** 2 + (rgb[2]! - p[2]) ** 2;
    if (d < distance) { distance = d; best = PALETTE[index]!.family; }
  }
  return best;
}

function clipY(input: readonly V3[], boundary: number, keepAbove: boolean): V3[] {
  const output: V3[] = [];
  for (let index = 0; index < input.length; index++) {
    const a = input[index]!;
    const b = input[(index + 1) % input.length]!;
    const ain = keepAbove ? a.y >= boundary : a.y <= boundary;
    const bin = keepAbove ? b.y >= boundary : b.y <= boundary;
    if (ain) output.push(a);
    if (ain !== bin) {
      const t = (boundary - a.y) / (b.y - a.y);
      output.push({ x: a.x + (b.x - a.x) * t, y: boundary, z: a.z + (b.z - a.z) * t });
    }
  }
  return output;
}

function mark(band: Band, x: number, y: number, family: number): void {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  if (ix < 0 || iy < 0 || ix >= MASK_RESOLUTION || iy >= MASK_RESOLUTION) return;
  const offset = iy * MASK_RESOLUTION + ix;
  const before = band.masks[offset]!;
  band.masks[offset] = before | (1 << family);
  if (before === 0) band.occupied++;
}

function rasterPolygon(band: Band, polygon: readonly V3[], family: number): void {
  if (polygon.length < 3) return;
  const points = polygon.map((p) => {
    const u = p.x - band.leanX * (p.y - band.yc);
    const v = p.z - band.leanZ * (p.y - band.yc);
    return {
      x: (u - band.u0) / (band.u1 - band.u0) * MASK_RESOLUTION,
      y: (v - band.v0) / (band.v1 - band.v0) * MASK_RESOLUTION,
    };
  });
  let minX = Number.POSITIVE_INFINITY; let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY; let maxY = Number.NEGATIVE_INFINITY;
  for (const p of points) {
    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
  }
  // Conservative supercover of every projected edge; this preserves thin hairs.
  for (let i = 0; i < points.length; i++) {
    const a = points[i]!; const b = points[(i + 1) % points.length]!;
    const steps = Math.max(1, Math.ceil(Math.max(Math.abs(b.x - a.x), Math.abs(b.y - a.y)) * 2));
    for (let s = 0; s <= steps; s++) mark(band, a.x + (b.x - a.x) * s / steps, a.y + (b.y - a.y) * s / steps, family);
  }
  const x0 = Math.max(0, Math.floor(minX)); const x1 = Math.min(MASK_RESOLUTION - 1, Math.floor(maxX));
  const y0 = Math.max(0, Math.floor(minY)); const y1 = Math.min(MASK_RESOLUTION - 1, Math.floor(maxY));
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const px = x + 0.5; const py = y + 0.5;
    let inside = false;
    for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
      const a = points[i]!; const b = points[j]!;
      if ((a.y > py) !== (b.y > py) && px < (b.x - a.x) * (py - a.y) / (b.y - a.y) + a.x) inside = !inside;
    }
    if (inside) mark(band, px, py, family);
  }
}

function firstFamily(mask: number): number {
  for (let i = 0; i < 7; i++) if ((mask & (1 << i)) !== 0) return i;
  return 0;
}
function panicleMask(mask: number): boolean {
  return (mask & 0b1111000) !== 0;
}

function traceBand(band: Band, origin: CensusVec3, direction: CensusVec3, maximumT: number): Hit | null {
  let enter = 0; let exit = maximumT;
  if (Math.abs(direction[1]) < 1e-14) {
    if (origin[1] < band.y0 || origin[1] > band.y1) return null;
  } else {
    let a = (band.y0 - origin[1]) / direction[1];
    let b = (band.y1 - origin[1]) / direction[1];
    if (a > b) [a, b] = [b, a];
    enter = Math.max(enter, a); exit = Math.min(exit, b);
  }
  if (!(exit >= enter)) return null;
  const ou = origin[0] - band.leanX * (origin[1] - band.yc);
  const ov = origin[2] - band.leanZ * (origin[1] - band.yc);
  const du = direction[0] - band.leanX * direction[1];
  const dv = direction[2] - band.leanZ * direction[1];
  const cellU = (band.u1 - band.u0) / MASK_RESOLUTION;
  const cellV = (band.v1 - band.v0) / MASK_RESOLUTION;
  const projectedSpeed = Math.hypot(du / cellU, dv / cellV);
  const at = (t: number): [number, number] => [
    (ou + du * t - band.u0) / cellU,
    (ov + dv * t - band.v0) / cellV,
  ];
  if (projectedSpeed < 1e-12) {
    const p = at(enter + 1e-10);
    const ix = Math.floor(p[0]); const iy = Math.floor(p[1]);
    if (ix < 0 || iy < 0 || ix >= MASK_RESOLUTION || iy >= MASK_RESOLUTION) return null;
    const familyMask = band.masks[iy * MASK_RESOLUTION + ix]!;
    return familyMask === 0 ? null : { t: enter, familyMask, cap: true, band: band.index };
  }
  const maxSteps = Math.ceil((exit - enter) * projectedSpeed) + 4;
  const dt = 0.45 / projectedSpeed;
  for (let step = 0; step <= maxSteps; step++) {
    const t = Math.min(exit, enter + step * dt);
    const p = at(t + 1e-10);
    const ix = Math.floor(p[0]); const iy = Math.floor(p[1]);
    if (ix >= 0 && iy >= 0 && ix < MASK_RESOLUTION && iy < MASK_RESOLUTION) {
      const familyMask = band.masks[iy * MASK_RESOLUTION + ix]!;
      if (familyMask !== 0) return { t, familyMask, cap: step === 0, band: band.index };
    }
    if (t >= exit) break;
  }
  return null;
}

function render(
  elevation: number,
  bounds: { min: CensusVec3; max: CensusVec3 },
  bvh: TriangleBvh,
  bands: readonly Band[],
  triangleFamilies: Uint8Array,
): { image: Uint8Array; metrics: Record<string, number> } {
  const angle = elevation * Math.PI / 180;
  const d: CensusVec3 = [Math.cos(angle), -Math.sin(angle), 0];
  const u: CensusVec3 = [0, 0, 1];
  const v: CensusVec3 = [Math.sin(angle), Math.cos(angle), 0];
  const corners: CensusVec3[] = [];
  for (const x of [bounds.min[0], bounds.max[0]]) for (const y of [bounds.min[1], bounds.max[1]]) for (const z of [bounds.min[2], bounds.max[2]]) corners.push([x, y, z]);
  const range = (axis: CensusVec3): [number, number] => {
    const values = corners.map((p) => dot(p, axis));
    return [Math.min(...values), Math.max(...values)];
  };
  const [umin0, umax0] = range(u); const [vmin0, vmax0] = range(v); const [smin, smax] = range(d);
  const upad = (umax0 - umin0) * 0.04; const vpad = (vmax0 - vmin0) * 0.04;
  const umin = umin0 - upad; const umax = umax0 + upad;
  const vmin = vmin0 - vpad; const vmax = vmax0 + vpad;
  const margin = Math.max(0.01, (smax - smin) * 0.02);
  const maximumT = smax - smin + margin * 2;
  const image = new Uint8Array(RENDER_WIDTH * 2 * RENDER_HEIGHT * 3);
  let truthHits = 0; let candidateHits = 0; let intersection = 0; let union = 0;
  let truthPanicle = 0; let retainedPanicle = 0; let optimisticOwner = 0; let capHits = 0;
  for (let py = 0; py < RENDER_HEIGHT; py++) for (let px = 0; px < RENDER_WIDTH; px++) {
    const qu = umin + (px + 0.5) / RENDER_WIDTH * (umax - umin);
    const qv = vmax - (py + 0.5) / RENDER_HEIGHT * (vmax - vmin);
    let origin = add(add([0, 0, 0], u, qu), v, qv);
    origin = add(origin, d, smin - margin);
    const truth = bvh.intersectNearest(origin, d, maximumT);
    let candidate: Hit | null = null;
    for (const band of bands) {
      const hit = traceBand(band, origin, d, maximumT);
      if (hit && (!candidate || hit.t < candidate.t)) candidate = hit;
    }
    const truthFamily = truth ? triangleFamilies[truth.triangleId]! : -1;
    if (truth) truthHits++;
    if (candidate) candidateHits++;
    if (truth && candidate) intersection++;
    if (truth || candidate) union++;
    if (truthFamily >= 3) {
      truthPanicle++;
      if (candidate && panicleMask(candidate.familyMask)) retainedPanicle++;
    }
    if (truth && candidate && (candidate.familyMask & (1 << truthFamily)) !== 0) optimisticOwner++;
    if (candidate?.cap) capHits++;
    const truthRgb = truth ? FAMILY_RGB[truthFamily]! : [18, 18, 18];
    const candidateRgb = candidate ? FAMILY_RGB[firstFamily(candidate.familyMask)]! : [18, 18, 18];
    let offset = (py * RENDER_WIDTH * 2 + px) * 3;
    image.set(truthRgb, offset);
    offset = (py * RENDER_WIDTH * 2 + RENDER_WIDTH + px) * 3;
    image.set(candidateRgb, offset);
  }
  return { image, metrics: {
    truthHits, candidateHits,
    silhouetteIoU: intersection / Math.max(1, union),
    overfillFractionOfCandidate: Math.max(0, candidateHits - intersection) / Math.max(1, candidateHits),
    truthLossFraction: Math.max(0, truthHits - intersection) / Math.max(1, truthHits),
    optimisticOwnerRecall: optimisticOwner / Math.max(1, intersection),
    panicleRetention: retainedPanicle / Math.max(1, truthPanicle),
    artificialCapFraction: capHits / Math.max(1, candidateHits),
  } };
}

const bytes = readFileSync(resolve(SOURCE));
if (hash(bytes) !== EXPECTED_SOURCE_SHA256) throw new Error('unexpected accepted Calamagrostis source');
const geometry = decodeOwnedProfileGeometry(bytes);
const bvh = TriangleBvh.build(geometry, 8);
const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
const vertexOffset = view.getUint32(84, true); const triangleOffset = view.getUint32(88, true);
const triangleFamilies = new Uint8Array(geometry.triangleCount);
console.error('[four-slab] classifying source triangles and fitting affine lean');
const height = geometry.bounds.max[1] - geometry.bounds.min[1];
const bands: Band[] = [];
for (let i = 0; i < 4; i++) {
  const y0 = geometry.bounds.min[1] + height * BAND_FRACTIONS[i]!;
  const y1 = geometry.bounds.min[1] + height * BAND_FRACTIONS[i + 1]!;
  bands.push({ index: i, y0, y1, yc: (y0 + y1) / 2, leanX: 0, leanZ: 0, u0: 0, u1: 0, v0: 0, v1: 0, masks: new Uint8Array(MASK_RESOLUTION ** 2), occupied: 0 });
}
for (let triangle = 0; triangle < geometry.triangleCount; triangle++) triangleFamilies[triangle] = familyForTriangle(triangle, bytes, vertexOffset, triangleOffset);
// Least-squares global lean in each band. This is the strongest one-axis affine
// fit available without splitting the band into per-plant carriers.
for (const band of bands) {
  let syy = 0; let sxy = 0; let szy = 0; let count = 0;
  for (let vertex = 0; vertex < geometry.vertexCount; vertex++) {
    const p = vertex * 3; const y = geometry.positions[p + 1]!;
    if (y < band.y0 || y > band.y1) continue;
    const dy = y - band.yc;
    syy += dy * dy; sxy += dy * (geometry.positions[p]! - (geometry.bounds.min[0] + geometry.bounds.max[0]) / 2);
    szy += dy * (geometry.positions[p + 2]! - (geometry.bounds.min[2] + geometry.bounds.max[2]) / 2); count++;
  }
  // A global regression can mistake the generator's strongly height-varying
  // triangle density for a common botanical lean (the six tufts do not share
  // one). Restrict the common affine generator to a generous 14-degree cone;
  // larger slopes would already contradict the authored culm displacements.
  const clampLean = (value: number): number => Math.max(-0.25, Math.min(0.25, value));
  band.leanX = clampLean(syy > 0 ? sxy / syy : 0);
  band.leanZ = clampLean(syy > 0 ? szy / syy : 0);
  const corners: V3[] = [];
  for (const x of [geometry.bounds.min[0], geometry.bounds.max[0]]) for (const y of [band.y0, band.y1]) for (const z of [geometry.bounds.min[2], geometry.bounds.max[2]]) corners.push({ x, y, z });
  const us = corners.map((p) => p.x - band.leanX * (p.y - band.yc));
  const vs = corners.map((p) => p.z - band.leanZ * (p.y - band.yc));
  band.u0 = Math.min(...us); band.u1 = Math.max(...us); band.v0 = Math.min(...vs); band.v1 = Math.max(...vs);
  console.error(`[four-slab] band ${band.index} y=${band.y0.toFixed(3)}..${band.y1.toFixed(3)} lean=${band.leanX.toFixed(4)},${band.leanZ.toFixed(4)} vertices=${count}`);
}
console.error('[four-slab] projecting conservative high-resolution masks');
for (let triangle = 0; triangle < geometry.triangleCount; triangle++) {
  const base = triangle * 3;
  const polygon: V3[] = [];
  for (let corner = 0; corner < 3; corner++) {
    const p = geometry.triangles[base + corner]! * 3;
    polygon.push({ x: geometry.positions[p]!, y: geometry.positions[p + 1]!, z: geometry.positions[p + 2]! });
  }
  const family = triangleFamilies[triangle]!;
  const minY = Math.min(...polygon.map((p) => p.y)); const maxY = Math.max(...polygon.map((p) => p.y));
  for (const band of bands) {
    if (maxY < band.y0 || minY > band.y1) continue;
    const clipped = clipY(clipY(polygon, band.y0, true), band.y1, false);
    rasterPolygon(band, clipped, family);
  }
}

const recipe = JSON.stringify({ source: EXPECTED_SOURCE_SHA256, mask: MASK_RESOLUTION, render: [RENDER_WIDTH, RENDER_HEIGHT], elevations: ELEVATIONS, fractions: BAND_FRACTIONS, directions: DIRECTIONS, storedTile: STORED_TILE, recordBytes: RECORD_BYTES });
const output = resolve('data/work/groundcover-four-slab-extrusion', EXPECTED_SOURCE_SHA256.slice(0, 16), hash(recipe).slice(0, 16));
mkdirSync(output, { recursive: true });
const views: Record<string, Record<string, number>> = {};
let number = 1;
for (const elevation of ELEVATIONS) {
  console.error(`[four-slab] rendering ${elevation} degrees`);
  const rendered = render(elevation, geometry.bounds, bvh, bands, triangleFamilies);
  const name = `${String(number++).padStart(3, '0')}-${String(elevation).replace('.', '_')}deg-truth-left-four-slab-right.png`;
  await sharp(rendered.image, { raw: { width: RENDER_WIDTH * 2, height: RENDER_HEIGHT, channels: 3 } }).png().toFile(resolve(output, name));
  views[String(elevation)] = rendered.metrics;
}
const residentBytes = Math.ceil(4 * STORED_TILE * STORED_TILE * DIRECTIONS * RECORD_BYTES * MIP_FACTOR);
const report = {
  schema: 'laas-groundcover-four-affine-slab-feasibility/v1',
  verdict: 'see hard gates',
  source: { path: SOURCE, sha256: EXPECTED_SOURCE_SHA256, vertices: geometry.vertexCount, triangles: geometry.triangleCount },
  recipe: JSON.parse(recipe),
  bands: bands.map((b) => ({ index: b.index, y: [b.y0, b.y1], lean: [b.leanX, b.leanZ], occupiedMaskTexels: b.occupied, maskCoverage: b.occupied / MASK_RESOLUTION ** 2 })),
  views,
  cost: { residentBytes, residentCapBytes: 51_121_152, completeRecordReads: 4, recordBytes: RECORD_BYTES, directionSlices: DIRECTIONS, atlasInterior: 220, wrappedStoredTile: STORED_TILE, mipFactor: MIP_FACTOR, bindings: 1, loopsMarchesOrCandidateListsAtRuntime: 0, speciesMultiplier: 0 },
  gates: { minimumSilhouetteIoU: 0.95, maximumOverfill: 0.03, maximumLoss: 0.03, minimumOptimisticOwnerRecall: 0.95, minimumPanicleRetention: 0.95, maximumArtificialCapFraction: 0.01 },
  interpretation: [
    'Each mask is the conservative 512^2 projection of every accepted source triangle clipped to the band after the best one-axis affine lean fit.',
    'This is optimistic: family ownership passes when the truth family is merely one of any families projected into the winning cell.',
    'The right image is the exact union of the four pixel-mask extrusions with closed slab caps; runtime would replace the offline mask traversal by one complete precomputed record read per band.',
    'The upper two bands isolate the panicle-bearing height regime. Any loss/overfill there directly tests fluffy-head retention versus slab filling.',
  ],
};
writeFileSync(resolve(output, 'report.json'), JSON.stringify(report, null, 2));
const files = readFileSync(fileURLToPath(import.meta.url));
writeFileSync(resolve(output, 'index.json'), JSON.stringify({ tool: 'tools/groundcover-bake/analyze-four-slab-extrusion.ts', toolSha256: hash(files), recipeSha256: hash(recipe), report: 'report.json' }, null, 2));
console.log(JSON.stringify({ output, ...report }, null, 2));
