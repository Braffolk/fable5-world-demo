/**
 * One bounded offline image gate for a camera-coherent pinhole conjugacy of
 * fixed canonical botanical surface/light fields. Canonical directions and
 * angular weights are selected once per camera, never per pixel.
 *
 * This emits no runtime asset and changes no shader.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import sharp from 'sharp';
import { makeCalamagrostisCanescensBandlimitedSource } from '../../EstonianGraminoids';
import {
  TriangleBvh,
  type CensusVec3,
  type DecodedOwnedProfileGeometry,
} from '../../ExteriorClosureCensus';
import { periodicNearestSuccessor, type OriginAwareTruthHit } from '../../OriginAwareRayTruth';
import type { IndexedMesh } from '../../ProfileFormat';

const SOURCE_SHA256 = '8cd69c2a6c61043c2861cad3a5ec06fee6dd4346a6722fd01fcc43468071ce5d';
const SOURCE_ELEVATIONS = [5, 15, 35, 55, 75] as const;
const AZIMUTH_COUNT = 16;
const STATIC_PIXELS = 192;
const PATH_PIXELS = 80;
const PATH_FRAMES = 7;
const PATH_METRES = 0.012;
const HORIZON = 155;
const RECORD_BYTES = 8;
const STORED_PHASE = 258;
const MAXIMUM_RESIDENT_BYTES = 51_121_152;
const EPSILON = 1e-8;
const MIDPLANE_REGULARITY_EPSILON = 1e-7;
const ACCEPTANCE = {
  minimumCoverageIoU: 0.90,
  minimumPanicleRecall: 0.90,
  maximumGhostAlphaFraction: 0.05,
  maximumDepthP95Metres: 0.02,
  maximumConnectedGhostFraction: 0.01,
  maximumMotionDeltaError: 0.03,
} as const;

interface EventAttributes { color: CensusVec3; normal: CensusVec3; plume: boolean }
interface CanonicalDirection { direction: CensusVec3; azimuthDegrees: number; elevationDegrees: number; weight: number }
interface CameraCase {
  id: string;
  label: string;
  camera: CensusVec3;
  axis: CensusVec3;
  right: CensusVec3;
  up: CensusVec3;
  fovDegrees: number;
  pixels: number;
  domain: 'exterior' | 'inside-horizontal';
}
interface CanonicalEvent extends EventAttributes {
  hit: boolean;
  sigma: number;
  point: CensusVec3;
  triangleId: number;
}
interface Sample {
  alpha: number;
  colorPremul: CensusVec3;
  depthMoment: number;
  conditionalDepth: number;
  panicleAlpha: number;
  valid: boolean;
  regularityRejects: number;
  behindRejects: number;
}
interface ImageResult {
  camera: CameraCase;
  canonical: CanonicalDirection[];
  truth: Uint8Array;
  reconstruction: Uint8Array;
  error: Uint8Array;
  truthSamples: Sample[];
  reconstructionSamples: Sample[];
  metrics: Record<string, unknown>;
}

function sha(value: Uint8Array | string): string {
  return createHash('sha256').update(value).digest('hex');
}

function add(a: CensusVec3, b: CensusVec3): CensusVec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function sub(a: CensusVec3, b: CensusVec3): CensusVec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scaled(value: CensusVec3, amount: number): CensusVec3 {
  return [value[0] * amount, value[1] * amount, value[2] * amount];
}

function dot(a: CensusVec3, b: CensusVec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: CensusVec3, b: CensusVec3): CensusVec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalized(value: CensusVec3): CensusVec3 {
  const inverse = 1 / Math.max(1e-30, Math.hypot(value[0], value[1], value[2]));
  return [value[0] * inverse, value[1] * inverse, value[2] * inverse];
}

function direction(azimuthDegrees: number, elevationDegrees: number): CensusVec3 {
  const a = azimuthDegrees * Math.PI / 180;
  const e = elevationDegrees * Math.PI / 180;
  return [Math.cos(e) * Math.cos(a), -Math.sin(e), Math.cos(e) * Math.sin(a)];
}

function axisBasis(axis: CensusVec3): { right: CensusVec3; up: CensusVec3 } {
  const reference: CensusVec3 = Math.abs(axis[1]) > 0.98 ? [0, 0, -1] : [0, 1, 0];
  const right = normalized(cross(axis, reference));
  return { right, up: normalized(cross(right, axis)) };
}

function meshBounds(mesh: IndexedMesh): { min: CensusVec3; max: CensusVec3 } {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let at = 0; at < mesh.positions.length; at += 3) {
    for (let axis = 0; axis < 3; axis++) {
      min[axis] = Math.min(min[axis]!, mesh.positions[at + axis]!);
      max[axis] = Math.max(max[axis]!, mesh.positions[at + axis]!);
    }
  }
  return { min: min as unknown as CensusVec3, max: max as unknown as CensusVec3 };
}

function geometryFrom(mesh: IndexedMesh, tile: { originX: number; originZ: number; sizeX: number; sizeZ: number; topH: number }): DecodedOwnedProfileGeometry {
  const bounds = meshBounds(mesh);
  return {
    version: 4,
    profileId: 2,
    topH: tile.topH,
    tileOriginX: tile.originX,
    tileOriginZ: tile.originZ,
    tileSizeX: tile.sizeX,
    tileSizeZ: tile.sizeZ,
    bounds,
    positions: Float64Array.from(mesh.positions),
    triangles: Uint32Array.from(mesh.indices),
    vertexCount: mesh.positions.length / 3,
    triangleCount: mesh.indices.length / 3,
  };
}

function trianglePlume(mesh: IndexedMesh): Uint8Array {
  if (!mesh.colors) throw new Error('pinhole light field gate requires authored RGB');
  const result = new Uint8Array(mesh.indices.length / 3);
  for (let triangle = 0; triangle < result.length; triangle++) {
    let reproductive = 0;
    for (let corner = 0; corner < 3; corner++) {
      const vertex = mesh.indices[triangle * 3 + corner]!;
      if (mesh.colors[vertex * 3]! > mesh.colors[vertex * 3 + 1]!) reproductive++;
    }
    result[triangle] = reproductive >= 2 ? 1 : 0;
  }
  return result;
}

function attributes(
  mesh: IndexedMesh,
  geometry: DecodedOwnedProfileGeometry,
  origin: CensusVec3,
  ray: CensusVec3,
  hit: OriginAwareTruthHit,
  plume: Uint8Array,
): EventAttributes {
  const point: CensusVec3 = [
    origin[0] + ray[0] * hit.t - hit.copyX * geometry.tileSizeX,
    origin[1] + ray[1] * hit.t,
    origin[2] + ray[2] * hit.t - hit.copyZ * geometry.tileSizeZ,
  ];
  const ids = [0, 1, 2].map((corner) => mesh.indices[hit.triangleId * 3 + corner]!);
  const p = ids.map((vertex) => [
    mesh.positions[vertex * 3]!, mesh.positions[vertex * 3 + 1]!, mesh.positions[vertex * 3 + 2]!,
  ] as CensusVec3);
  const e0 = sub(p[1]!, p[0]!);
  const e1 = sub(p[2]!, p[0]!);
  const q = sub(point, p[0]!);
  const d00 = dot(e0, e0); const d01 = dot(e0, e1); const d11 = dot(e1, e1);
  const d20 = dot(q, e0); const d21 = dot(q, e1);
  const inverse = 1 / Math.max(1e-30, d00 * d11 - d01 * d01);
  const b = (d11 * d20 - d01 * d21) * inverse;
  const c = (d00 * d21 - d01 * d20) * inverse;
  const weights = [1 - b - c, b, c];
  const color = [0, 0, 0];
  const normal = [0, 0, 0];
  for (let corner = 0; corner < 3; corner++) {
    const vertex = ids[corner]!;
    for (let component = 0; component < 3; component++) {
      color[component] += (mesh.colors?.[vertex * 3 + component] ?? 1) * weights[corner]!;
      normal[component] += mesh.normals[vertex * 3 + component]! * weights[corner]!;
    }
  }
  return {
    color: color as unknown as CensusVec3,
    normal: normalized(normal as unknown as CensusVec3),
    plume: plume[hit.triangleId] === 1,
  };
}

function emptySample(valid = true): Sample {
  return {
    alpha: 0,
    colorPremul: [0, 0, 0],
    depthMoment: 0,
    conditionalDepth: 0,
    panicleAlpha: 0,
    valid,
    regularityRejects: 0,
    behindRejects: 0,
  };
}

function rayForPixel(camera: CameraCase, x: number, y: number): CensusVec3 {
  const tangent = Math.tan(camera.fovDegrees * Math.PI / 360);
  const aspectX = ((x + 0.5) / camera.pixels * 2 - 1) * tangent;
  const aspectY = (1 - (y + 0.5) / camera.pixels * 2) * tangent;
  return normalized(add(camera.axis, add(scaled(camera.right, aspectX), scaled(camera.up, aspectY))));
}

function trueSample(
  mesh: IndexedMesh,
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  plume: Uint8Array,
  camera: CensusVec3,
  ray: CensusVec3,
): Sample {
  let origin = camera;
  let horizon = HORIZON;
  if (camera[1] > geometry.bounds.max[1] + EPSILON) {
    if (ray[1] >= 0) return emptySample();
    const entry = (geometry.bounds.max[1] - camera[1]) / ray[1];
    if (!(entry >= 0)) return emptySample();
    origin = add(camera, scaled(ray, entry));
    horizon = Math.min(HORIZON - entry, (geometry.bounds.min[1] - origin[1]) / ray[1]);
    if (!(horizon > EPSILON)) return emptySample();
    const hit = periodicNearestSuccessor(geometry, bvh, origin, ray, horizon, EPSILON);
    if (!hit) return emptySample();
    const event = attributes(mesh, geometry, origin, ray, hit, plume);
    const depth = entry + hit.t;
    return {
      alpha: 1,
      colorPremul: event.color,
      depthMoment: depth,
      conditionalDepth: depth,
      panicleAlpha: event.plume ? 1 : 0,
      valid: true,
      regularityRejects: 0,
      behindRejects: 0,
    };
  }
  if (camera[1] < geometry.bounds.min[1] - EPSILON) return emptySample();
  const hit = periodicNearestSuccessor(geometry, bvh, origin, ray, horizon, EPSILON);
  if (!hit) return emptySample();
  const event = attributes(mesh, geometry, origin, ray, hit, plume);
  return {
    alpha: 1,
    colorPremul: event.color,
    depthMoment: hit.t,
    conditionalDepth: hit.t,
    panicleAlpha: event.plume ? 1 : 0,
    valid: true,
    regularityRejects: 0,
    behindRejects: 0,
  };
}

function canonicalEvent(
  mesh: IndexedMesh,
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  plume: Uint8Array,
  midY: number,
  q: CensusVec3,
  canonical: CensusVec3,
): CanonicalEvent {
  if (!(canonical[1] < -EPSILON)) {
    return { hit: false, sigma: 0, point: q, triangleId: -1, color: [0, 0, 0], normal: [0, 1, 0], plume: false };
  }
  const toMid = (geometry.bounds.max[1] - midY) / -canonical[1];
  const origin = sub(q, scaled(canonical, toMid));
  const horizon = (geometry.bounds.max[1] - geometry.bounds.min[1]) / -canonical[1];
  const hit = periodicNearestSuccessor(geometry, bvh, origin, canonical, horizon, EPSILON);
  if (!hit) {
    return { hit: false, sigma: 0, point: q, triangleId: -1, color: [0, 0, 0], normal: [0, 1, 0], plume: false };
  }
  const event = attributes(mesh, geometry, origin, canonical, hit, plume);
  const sigma = hit.t - toMid;
  return {
    hit: true,
    sigma,
    point: add(q, scaled(canonical, sigma)),
    triangleId: hit.triangleId,
    ...event,
  };
}

function canonicalDirections(axis: CensusVec3): CanonicalDirection[] {
  const elevation = Math.asin(Math.max(-1, Math.min(1, -axis[1]))) * 180 / Math.PI;
  if (elevation >= 82.5) return [{ direction: [0, -1, 0], azimuthDegrees: 0, elevationDegrees: 90, weight: 1 }];
  const rawAzimuth = Math.atan2(axis[2], axis[0]) * 180 / Math.PI;
  const azimuth = ((rawAzimuth % 360) + 360) % 360;
  const azimuthCoordinate = azimuth / (360 / AZIMUTH_COUNT);
  const az0 = Math.floor(azimuthCoordinate) % AZIMUTH_COUNT;
  const az1 = (az0 + 1) % AZIMUTH_COUNT;
  const azMix = azimuthCoordinate - Math.floor(azimuthCoordinate);
  let low: number = SOURCE_ELEVATIONS[0];
  let high: number = SOURCE_ELEVATIONS[0];
  for (let index = 0; index < SOURCE_ELEVATIONS.length - 1; index++) {
    if (elevation >= SOURCE_ELEVATIONS[index]! && elevation <= SOURCE_ELEVATIONS[index + 1]!) {
      low = SOURCE_ELEVATIONS[index]!;
      high = SOURCE_ELEVATIONS[index + 1]!;
      break;
    }
    if (elevation > SOURCE_ELEVATIONS[SOURCE_ELEVATIONS.length - 1]!) {
      low = SOURCE_ELEVATIONS[SOURCE_ELEVATIONS.length - 1]!;
      high = low;
    }
  }
  const elevationMix = high === low ? 0 : (Math.max(low, Math.min(high, elevation)) - low) / (high - low);
  const rows = high === low ? [{ elevation: low, weight: 1 }] : [
    { elevation: low, weight: 1 - elevationMix },
    { elevation: high, weight: elevationMix },
  ];
  const columns = az0 === az1 || azMix === 0 ? [{ azimuth: az0, weight: 1 }] : [
    { azimuth: az0, weight: 1 - azMix },
    { azimuth: az1, weight: azMix },
  ];
  return rows.flatMap((row) => columns.map((column) => ({
    direction: direction(column.azimuth * 360 / AZIMUTH_COUNT, row.elevation),
    azimuthDegrees: column.azimuth * 360 / AZIMUTH_COUNT,
    elevationDegrees: row.elevation,
    weight: row.weight * column.weight,
  }))).filter((entry) => entry.weight > 1e-12);
}

function reconstructedSample(
  mesh: IndexedMesh,
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  plume: Uint8Array,
  midY: number,
  camera: CensusVec3,
  ray: CensusVec3,
  canonical: readonly CanonicalDirection[],
): Sample {
  const denominator = midY - camera[1];
  if (Math.abs(denominator) <= EPSILON || Math.abs(ray[1]) <= EPSILON) return emptySample(false);
  const carrierT = denominator / ray[1];
  const q = add(camera, scaled(ray, carrierT));
  let alpha = 0;
  const color = [0, 0, 0];
  let moment = 0;
  let panicleAlpha = 0;
  let regularityRejects = 0;
  let behindRejects = 0;
  for (const source of canonical) {
    const event = canonicalEvent(mesh, geometry, bvh, plume, midY, q, source.direction);
    if (!event.hit) continue;
    const a = source.direction[1] * event.sigma / denominator;
    const regularity = 1 + a;
    if (Math.abs(regularity) <= MIDPLANE_REGULARITY_EPSILON) { regularityRejects++; continue; }
    const liveT = carrierT * regularity;
    if (!(liveT > EPSILON) || liveT > HORIZON) { behindRejects++; continue; }
    alpha += source.weight;
    moment += source.weight * liveT;
    if (event.plume) panicleAlpha += source.weight;
    for (let component = 0; component < 3; component++) {
      color[component] += source.weight * event.color[component]!;
    }
  }
  return {
    alpha,
    colorPremul: color as unknown as CensusVec3,
    depthMoment: moment,
    conditionalDepth: alpha > 1e-12 ? moment / alpha : 0,
    panicleAlpha,
    valid: true,
    regularityRejects,
    behindRejects,
  };
}

function rgba(sample: Sample): readonly [number, number, number, number] {
  const inverseAlpha = sample.alpha > 1e-12 ? 1 / sample.alpha : 0;
  return [
    Math.round(255 * Math.max(0, Math.min(1, sample.colorPremul[0] * inverseAlpha))),
    Math.round(255 * Math.max(0, Math.min(1, sample.colorPremul[1] * inverseAlpha))),
    Math.round(255 * Math.max(0, Math.min(1, sample.colorPremul[2] * inverseAlpha))),
    Math.round(255 * Math.max(0, Math.min(1, sample.alpha))),
  ];
}

function connected(mask: Uint8Array, width: number, height: number): { maximum: number; maximumRun: number; maximumAspect: number } {
  const seen = new Uint8Array(mask.length);
  let maximum = 0;
  let maximumRun = 0;
  let maximumAspect = 0;
  for (let y = 0; y < height; y++) {
    let run = 0;
    for (let x = 0; x < width; x++) {
      if (mask[y * width + x]) { run++; maximumRun = Math.max(maximumRun, run); } else run = 0;
    }
  }
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || seen[start]) continue;
    const queue = [start]; seen[start] = 1;
    let count = 0; let minX = width; let minY = height; let maxX = 0; let maxY = 0;
    while (queue.length) {
      const at = queue.pop()!; count++;
      const x = at % width; const y = Math.floor(at / width);
      minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      for (const next of [x > 0 ? at - 1 : -1, x + 1 < width ? at + 1 : -1, y > 0 ? at - width : -1, y + 1 < height ? at + width : -1]) {
        if (next >= 0 && mask[next] && !seen[next]) { seen[next] = 1; queue.push(next); }
      }
    }
    const spanX = maxX - minX + 1; const spanY = maxY - minY + 1;
    maximum = Math.max(maximum, count);
    maximumAspect = Math.max(maximumAspect, Math.max(spanX / spanY, spanY / spanX));
  }
  return { maximum, maximumRun, maximumAspect };
}

function quantile(values: readonly number[], q: number): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * q))]!;
}

function measure(truth: readonly Sample[], reconstruction: readonly Sample[], width: number, height: number): Record<string, unknown> {
  let intersection = 0; let union = 0; let ghostAlpha = 0; let reconstructionAlpha = 0;
  let truthPanicle = 0; let reconstructionOnTruthPanicle = 0; let regularityRejects = 0; let behindRejects = 0;
  const depthErrors: number[] = [];
  const ghostMask = new Uint8Array(width * height);
  const holeMask = new Uint8Array(width * height);
  for (let pixel = 0; pixel < truth.length; pixel++) {
    const t = truth[pixel]!; const r = reconstruction[pixel]!;
    intersection += Math.min(t.alpha, r.alpha);
    union += Math.max(t.alpha, r.alpha);
    reconstructionAlpha += r.alpha;
    if (t.alpha <= 0.001) ghostAlpha += r.alpha;
    if (t.panicleAlpha > 0.5) {
      truthPanicle++;
      reconstructionOnTruthPanicle += r.panicleAlpha;
    }
    if (t.alpha > 0.5 && r.alpha > 0.05) depthErrors.push(Math.abs(t.conditionalDepth - r.conditionalDepth));
    ghostMask[pixel] = t.alpha <= 0.001 && r.alpha >= 0.1 ? 1 : 0;
    holeMask[pixel] = t.alpha > 0.5 && r.alpha < 0.1 ? 1 : 0;
    regularityRejects += r.regularityRejects;
    behindRejects += r.behindRejects;
  }
  const ghost = connected(ghostMask, width, height);
  const holes = connected(holeMask, width, height);
  return {
    coverageIoU: union > 0 ? intersection / union : 1,
    ghostAlphaFraction: reconstructionAlpha > 0 ? ghostAlpha / reconstructionAlpha : 0,
    panicleRecall: truthPanicle > 0 ? reconstructionOnTruthPanicle / truthPanicle : 1,
    depthAbsoluteErrorMetres: { p50: quantile(depthErrors, 0.5), p95: quantile(depthErrors, 0.95), maximum: quantile(depthErrors, 1) },
    connectedGhost: { ...ghost, fraction: ghost.maximum / (width * height) },
    connectedHoles: { ...holes, fraction: holes.maximum / (width * height) },
    regularityRejects,
    behindCameraOrHorizonRejects: behindRejects,
  };
}

function renderCase(
  mesh: IndexedMesh,
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  plume: Uint8Array,
  midY: number,
  camera: CameraCase,
): ImageResult {
  const canonical = canonicalDirections(camera.axis);
  const truth = new Uint8Array(camera.pixels * camera.pixels * 4);
  const reconstruction = new Uint8Array(truth.length);
  const error = new Uint8Array(truth.length);
  const truthSamples: Sample[] = [];
  const reconstructionSamples: Sample[] = [];
  for (let y = 0; y < camera.pixels; y++) for (let x = 0; x < camera.pixels; x++) {
    const ray = rayForPixel(camera, x, y);
    const t = trueSample(mesh, geometry, bvh, plume, camera.camera, ray);
    const r = camera.domain === 'exterior'
      ? reconstructedSample(mesh, geometry, bvh, plume, midY, camera.camera, ray, canonical)
      : emptySample(false);
    truthSamples.push(t); reconstructionSamples.push(r);
    const index = (y * camera.pixels + x) * 4;
    truth.set(rgba(t), index); reconstruction.set(rgba(r), index);
    error[index] = Math.round(255 * Math.min(1, Math.abs(t.alpha - r.alpha)));
    error[index + 1] = Math.round(255 * Math.min(1, Math.abs(t.colorPremul[1] - r.colorPremul[1])));
    error[index + 2] = Math.round(255 * Math.min(1, Math.abs(t.colorPremul[2] - r.colorPremul[2])));
    error[index + 3] = 255;
  }
  return {
    camera,
    canonical,
    truth,
    reconstruction,
    error,
    truthSamples,
    reconstructionSamples,
    metrics: measure(truthSamples, reconstructionSamples, camera.pixels, camera.pixels),
  };
}

async function labelledPanel(result: ImageResult, path: string): Promise<void> {
  const pixels = result.camera.pixels;
  const labelHeight = 34;
  const images = await Promise.all([result.truth, result.reconstruction, result.error].map((data) =>
    sharp(data, { raw: { width: pixels, height: pixels, channels: 4 } }).png().toBuffer()));
  const labels = ['TRUTH', 'FRAME-GLOBAL PINHOLE FIELD', 'ABS ERROR'];
  const svg = Buffer.from(`<svg width="${pixels * 3}" height="${labelHeight}" xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="#201d20"/>${labels.map((label, index) => `<text x="${index * pixels + 8}" y="24" fill="#f4edf3" font-family="Arial" font-size="16">${label}</text>`).join('')}</svg>`);
  await sharp({ create: { width: pixels * 3, height: pixels + labelHeight, channels: 4, background: '#201d20' } })
    .composite([
      { input: svg, left: 0, top: 0 },
      ...images.map((input, index) => ({ input, left: index * pixels, top: labelHeight })),
    ])
    .png()
    .toFile(path);
}

function cameraAt(
  id: string,
  label: string,
  target: CensusVec3,
  azimuth: number,
  elevation: number,
  distance: number,
  fovDegrees: number,
  pixels: number,
): CameraCase {
  const axis = direction(azimuth, elevation);
  const basis = axisBasis(axis);
  return { id, label, camera: sub(target, scaled(axis, distance)), axis, ...basis, fovDegrees, pixels, domain: 'exterior' };
}

const fixture = makeCalamagrostisCanescensBandlimitedSource();
const sourceHash = sha(JSON.stringify(fixture.mesh));
if (sourceHash !== SOURCE_SHA256) throw new Error(`isolated Calamagrostis source changed: ${sourceHash}`);
const geometry = geometryFrom(fixture.mesh, fixture.tile);
const plume = trianglePlume(fixture.mesh);
const midY = (geometry.bounds.min[1] + geometry.bounds.max[1]) * 0.5;
const target: CensusVec3 = [
  fixture.tile.originX + fixture.tile.sizeX * 0.5,
  geometry.bounds.min[1] + (geometry.bounds.max[1] - geometry.bounds.min[1]) * 0.56,
  fixture.tile.originZ + fixture.tile.sizeZ * 0.5,
];
const cases: CameraCase[] = [
  cameraAt('narrow-18', '18 degree path reference', target, 45, 18, 2.35, 18, STATIC_PIXELS),
  cameraAt('wide-oblique', 'wide-FOV 18 degree oblique', target, 45, 18, 2.35, 75, STATIC_PIXELS),
  cameraAt('grazing-2', '2 degree grazing exterior', target, 45, 2, 22, 42, STATIC_PIXELS),
  cameraAt('top-82_5', '82.5 degree top-down', target, 45, 82.5, 2.35, 50, STATIC_PIXELS),
];
const insideAxis = direction(45, 0);
const insideBasis = axisBasis(insideAxis);
cases.push({
  id: 'inside-horizontal',
  label: 'exact-horizontal camera-inside hard domain',
  camera: [target[0], midY, target[2]],
  axis: insideAxis,
  ...insideBasis,
  fovDegrees: 30,
  pixels: STATIC_PIXELS,
  domain: 'inside-horizontal',
});

console.error(`[pinhole-field] building BVH for ${geometry.triangleCount.toLocaleString()} exact source triangles`);
const bvh = TriangleBvh.build(geometry, 8);
const recipe = {
  schema: 'laas-frame-global-pinhole-light-field-gate/v1',
  sourceSha256: sourceHash,
  midPlaneY: midY,
  canonical: { azimuthCount: AZIMUTH_COUNT, elevationsDegrees: SOURCE_ELEVATIONS, verticalSingleton: true, selection: 'once per camera optical axis' },
  cases: cases.map(({ id, label, camera, axis, fovDegrees, pixels, domain }) => ({ id, label, camera, axis, fovDegrees, pixels, domain })),
  path: { frames: PATH_FRAMES, metres: PATH_METRES, pixels: PATH_PIXELS, persistentAxisAndWeights: true },
  acceptance: ACCEPTANCE,
  implementationSha256: sha(readFileSync(import.meta.filename)),
};
const recipeHash = sha(JSON.stringify(recipe));
const outputRoot = resolve('data/work/groundcover-frame-global-pinhole-field', sourceHash.slice(0, 16), recipeHash.slice(0, 16));
const qaRoot = resolve(outputRoot, 'qa');
mkdirSync(qaRoot, { recursive: true });
const qa: Array<{ file: string; sha256: string; width: number; height: number; interpretation: string }> = [];
const results: ImageResult[] = [];
for (let index = 0; index < cases.length; index++) {
  const camera = cases[index]!;
  console.error(`[pinhole-field] rendering ${camera.id}`);
  const result = renderCase(fixture.mesh, geometry, bvh, plume, midY, camera);
  results.push(result);
  const file = `${String(index + 1).padStart(3, '0')}-${camera.id}-truth-reconstruction-error.png`;
  const path = resolve(qaRoot, file);
  await labelledPanel(result, path);
  qa.push({
    file,
    sha256: sha(readFileSync(path)),
    width: camera.pixels * 3,
    height: camera.pixels + 34,
    interpretation: `${camera.label}: exact periodic source truth, coherent frame-global pinhole conjugacy, absolute premultiplied error`,
  });
}

const reference = cameraAt('path', '18 degree millimetric path', target, 45, 18, 2.35, 18, PATH_PIXELS);
const pathFrames: ImageResult[] = [];
for (let frame = 0; frame < PATH_FRAMES; frame++) {
  const offset = (frame / (PATH_FRAMES - 1) - 0.5) * PATH_METRES;
  const camera = { ...reference, id: `path-${frame}`, camera: add(reference.camera, scaled(reference.right, offset)) };
  console.error(`[pinhole-field] rendering path ${frame + 1}/${PATH_FRAMES}`);
  pathFrames.push(renderCase(fixture.mesh, geometry, bvh, plume, midY, camera));
}
let motionError = 0;
let motionTerms = 0;
for (let frame = 1; frame < pathFrames.length; frame++) {
  const previous = pathFrames[frame - 1]!; const current = pathFrames[frame]!;
  for (let pixel = 0; pixel < previous.truthSamples.length; pixel++) {
    for (let component = 0; component < 3; component++) {
      const truthDelta = current.truthSamples[pixel]!.colorPremul[component]! - previous.truthSamples[pixel]!.colorPremul[component]!;
      const reconstructionDelta = current.reconstructionSamples[pixel]!.colorPremul[component]! - previous.reconstructionSamples[pixel]!.colorPremul[component]!;
      motionError += Math.abs(reconstructionDelta - truthDelta);
      motionTerms++;
    }
  }
}
const motionDeltaError = motionError / Math.max(1, motionTerms);
const selectedPathFrames = [pathFrames[0]!, pathFrames[Math.floor(PATH_FRAMES / 2)]!, pathFrames[PATH_FRAMES - 1]!];
for (const [kind, key] of [['truth', 'truth'], ['reconstruction', 'reconstruction']] as const) {
  const images = await Promise.all(selectedPathFrames.map((frame) => sharp(frame[key], {
    raw: { width: PATH_PIXELS, height: PATH_PIXELS, channels: 4 },
  }).png().toBuffer()));
  const file = `${kind === 'truth' ? '006' : '007'}-18deg-path-${kind}-first-mid-last.png`;
  const path = resolve(qaRoot, file);
  await sharp({ create: { width: PATH_PIXELS * 3, height: PATH_PIXELS, channels: 4, background: '#00000000' } })
    .composite(images.map((input, index) => ({ input, left: index * PATH_PIXELS, top: 0 })))
    .png().toFile(path);
  qa.push({ file, sha256: sha(readFileSync(path)), width: PATH_PIXELS * 3, height: PATH_PIXELS, interpretation: `persistent-pixel ${kind} at first, middle and last 18-degree millimetric camera frames` });
}

const exterior = results.filter((result) => result.camera.domain === 'exterior');
const numeric = exterior.map((result) => result.metrics as {
  coverageIoU: number; ghostAlphaFraction: number; panicleRecall: number;
  depthAbsoluteErrorMetres: { p95: number | null }; connectedGhost: { fraction: number };
});
const exteriorPass = numeric.every((metrics) =>
  metrics.coverageIoU >= ACCEPTANCE.minimumCoverageIoU
  && metrics.panicleRecall >= ACCEPTANCE.minimumPanicleRecall
  && metrics.ghostAlphaFraction <= ACCEPTANCE.maximumGhostAlphaFraction
  && (metrics.depthAbsoluteErrorMetres.p95 ?? Infinity) <= ACCEPTANCE.maximumDepthP95Metres
  && metrics.connectedGhost.fraction <= ACCEPTANCE.maximumConnectedGhostFraction);
const horizontalInsidePass = false;
const motionPass = motionDeltaError <= ACCEPTANCE.maximumMotionDeltaError;
const decision = exteriorPass && horizontalInsidePass && motionPass ? 'PROVISIONAL_PASS' : 'REJECT_AND_PARK';
const report = {
  schema: recipe.schema,
  decision,
  source: { sha256: sourceHash, vertices: geometry.vertexCount, triangles: geometry.triangleCount, structureTriangles: plume.length - plume.reduce((a, b) => a + b, 0), plumeTriangles: plume.reduce((a, b) => a + b, 0), tile: fixture.tile, bounds: geometry.bounds },
  method: {
    carrier: `one fixed plane y=${midY}`,
    canonicalGraph: 'P_c(Q)=Q+c*sigma_c(Q), exterior first event of the actual periodic source',
    pinholeMap: 'F_C(Q)=Q+(Q-C)*(n.c)*sigma(Q)/(k-n.C)',
    selection: '2-4 canonical directions and weights selected once from the camera optical axis; identical for every pixel',
    combination: 'only premultiplied radiance, coverage, and first depth moment use frame-global weights; owners are never averaged or elected per pixel',
    normalBoundary: 'normal images are deliberately not shown; an implementation displaying normals must use D_u F=(1+a)E+(Q-C)*gamma*(grad sigma)^T and its cross-product/inverse-transpose normal, never the per-ray rank-one normal',
  },
  cost: {
    residentBytes: AZIMUTH_COUNT * SOURCE_ELEVATIONS.length * STORED_PHASE ** 2 * RECORD_BYTES + STORED_PHASE ** 2 * RECORD_BYTES,
    maximumResidentBytes: MAXIMUM_RESIDENT_BYTES,
    filteredCompleteFieldSamplesPerPixel: 4,
    estimatedFmaEquivalentCeilingIncludingExactStitchedNormal: 224,
    shape: 'one texture-array binding; frame-global direction indices/weights; sequential four-record accumulation; no loop in compiled specialization, march, candidate election, shell, pass, barrier, or species query',
  },
  cases: Object.fromEntries(results.map((result) => [result.camera.id, {
    camera: result.camera,
    canonical: result.canonical,
    metrics: result.metrics,
  }])),
  path: { frames: PATH_FRAMES, lateralStepMetres: PATH_METRES / (PATH_FRAMES - 1), meanPremultipliedDeltaError: motionDeltaError, pass: motionPass },
  gates: { exteriorPass, horizontalInsidePass, motionPass, acceptance: ACCEPTANCE },
  exactHorizontalAndInside: {
    result: 'FAIL_BY_DOMAIN',
    reason: 'for camera C on the botanical mid-plane, k-n.C=0; exact-horizontal rays also have no unique intersection Q with that plane. The exterior surface graph does not encode the pointed-line successor after C.',
  },
  interpretation: [
    'Frame-global selection removes camera-front per-pixel angular Voronoi cones by construction.',
    'It does not make independent canonical first-event sheets agree. Premultiplied blending can therefore ghost or erase silhouettes while remaining spatially coherent.',
    'The image gate compares the coherently deformed canonical light/surface field to unchanged actual-source first-hit truth; it does not penalize the method for failing an unrelated per-pixel owner blend.',
    'A failure of exact horizontal/inside is a representation-domain failure and is not hidden with a grazing clamp or five-degree substitution.',
  ],
  provenance: [
    'LAAS GRASS-CHEAP-PROJECTIVE-RAY-MATH.md section 20.2: fixed-carrier height-preserving pinhole conjugacy and stitched Jacobian.',
    'Lin and Shum (2004), A Geometric Analysis of Light Field Rendering: geometry-assisted light-field reconstruction and disocclusion boundary.',
    'LAAS-original: actual-source periodic image gate, frame-global angular contract, connected ghost/stretch metrics, panicle retention, millimetric path test, and fixed cost accounting.',
  ],
  qa,
};
const reportPath = resolve(outputRoot, 'report.json');
writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
const index = {
  schema: 'laas-frame-global-pinhole-light-field-qa-index/v1',
  sourceSha256: sourceHash,
  recipeHash,
  report: { file: 'report.json', sha256: sha(readFileSync(reportPath)) },
  images: qa,
};
writeFileSync(resolve(qaRoot, 'index.json'), `${JSON.stringify(index, null, 2)}\n`);
console.error(`[pinhole-field] ${decision}; wrote ${outputRoot}`);
console.log(outputRoot);
