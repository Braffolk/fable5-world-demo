/**
 * Offline falsification gate for the shell-frame field proposal.
 *
 * Runtime work represented here is fixed: three direction nodes and either
 * two, three, or four texture reads per node. All triangle traversal, lattice
 * construction, shell fitting, and metric loops are cook/gate work only.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';
import {
  TriangleBvh,
  decodeOwnedProfileGeometry,
  type CensusVec3,
  type DecodedOwnedProfileGeometry,
} from '../../ExteriorClosureCensus';
import {
  ESTONIAN_GRAMINOID_PROFILE_IDS,
  makeEstonianGraminoidFixture,
} from '../../EstonianGraminoids';
import { periodicNearestSuccessor, type OriginAwareTruthHit } from '../../OriginAwareRayTruth';
import { makeSphagnumCapillifoliumFixture } from '../../SphagnumCapillifolium';

const WORKSPACE = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const TALL_SOURCE = resolve(WORKSPACE, 'src/assets/groundcover/calamagrostis-canescens.gcrp');
const LOW_PACKED_SOURCE = resolve(WORKSPACE, 'src/assets/groundcover/agrostis-capillaris.gcrp');
const DENSE_PACKED_SOURCE = resolve(WORKSPACE, 'src/assets/groundcover/sphagnum-capillifolium.gcrp');
const TALL_SHA256 = '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const TAU = Math.PI * 2;
const DEG = Math.PI / 180;
const CAMERA_WIDTH = 32;
const CAMERA_HEIGHT = 24;
const QA_SCALE = 6;
const FRAME_RESOLUTION = 192;
const SHELL_RESOLUTION = 32;
const SHELL_SOURCE_RESOLUTION = 192;
const BACKGROUND: CensusVec3 = [0.115, 0.09, 0.065];
const CAMERA_SHIFTS = [0, 0.001, 0.0025, 0.0045] as const;
const K_PIXELS = 2.5;
const REFERENCE_THETA_PIXEL = 60 * DEG / 1920;
const INNER_RING_END = 2;
const INNER_SPACING = 0.18;
const MAX_SLOPE = 1 / Math.tan(1 * DEG);
const MAX_LAW_RINGS = 4096;
const RECORD_BYTES = 10;
const SHELL_BYTES = 2;
const EXP024_DIRECTIONS = 97;
const EXP024_BYTES = 50_844_684;
const COVERAGE_THRESHOLD = 0.5;

const CAMERA_SWEEPS = [
  { key: 'top-down-90deg', elevation: 90, azimuth: 17, height: 1.6, hFov: 18, vFov: 13.5 },
  { key: 'oblique-18deg', elevation: 18, azimuth: 73, height: 1.2, hFov: 8, vFov: 5 },
  { key: 'low-oblique-10deg', elevation: 10, azimuth: 137, height: 0.8, hFov: 5, vFov: 3 },
  { key: 'grazing-5deg', elevation: 5, azimuth: 211, height: 0.55, hFov: 3, vFov: 1.8 },
  { key: 'near-horizontal-1deg', elevation: 1, azimuth: 293, height: 0.4, hFov: 1.2, vFov: 0.6 },
] as const;

const PALETTES = {
  low: [
    [0.12, 0.29, 0.055], [0.34, 0.52, 0.12], [0.42, 0.55, 0.18],
    [0.30, 0.25, 0.08], [0.48, 0.28, 0.20], [0.58, 0.32, 0.30],
    [0.86, 0.78, 0.66], [0.39, 0.12, 0.30],
  ],
  tall: [
    [0.10, 0.27, 0.055], [0.37, 0.51, 0.13], [0.43, 0.53, 0.20],
    [0.31, 0.24, 0.075], [0.48, 0.31, 0.24], [0.57, 0.35, 0.39],
    [0.88, 0.80, 0.70], [0.42, 0.10, 0.34],
  ],
} as const;
const FAMILY_FOR_PALETTE_ENTRY = [0, 0, 1, 2, 3, 4, 5, 6] as const;

type Variant = 'noRealign6' | 'winner7' | 'strict9' | 'second12';

interface Community {
  key: 'low' | 'tall';
  label: string;
  geometry: DecodedOwnedProfileGeometry;
  colors: Float64Array;
  bvh: TriangleBvh;
  meanHeight: number;
  minimumEyeHeight: number;
  sources: Record<string, unknown>;
}

interface Surface {
  point: CensusVec3;
  color: CensusVec3;
  normal: CensusVec3;
  family: number;
  triangleId: number;
  copyX: number;
  copyZ: number;
}

interface ShellMap {
  values: Float64Array;
  rawResidualCrisp: number[];
  rawResidualPlume: number[];
  rawLocalSpread: number[];
  residualCrisp: number[];
  residualPlume: number[];
  hitFraction: number;
  smoothingPasses: number;
  constantFallback: boolean;
  maximumGradient: number;
  conditionProduct: number;
  conditioned: boolean;
}

interface DirectionNode {
  ring: number;
  angularIndex: number;
  slopeX: number;
  slopeZ: number;
  direction: CensusVec3;
  /** Maximum |s_live-s_i| over the triangles supported by this node.
   * The frame is parameterized by its mean-plane XZ crossing, so this is
   * the exact coordinate-equivalent of tan(Delta) in the orthogonal frame. */
  cellRadius: number;
}

interface DirectionRing {
  slope: number;
  count: number;
  sigma: number;
  radialStep: number;
  conditionSlopeRadius?: number;
}

interface DirectionLattice {
  rings: DirectionRing[];
  totalDirections: number;
  truncated: boolean;
}

interface FrameRecord {
  hit: boolean;
  premul: CensusVec3;
  coverage: number;
  tau: number;
  residual: number;
  family: number;
  normal: CensusVec3;
  point: CensusVec3 | null;
}

interface FilteredRecord {
  premul: CensusVec3;
  coverage: number;
  categorical: FrameRecord | null;
}

interface CameraRay {
  x: number;
  y: number;
  origin: CensusVec3;
  direction: CensusVec3;
  truth: Surface | null;
}

interface NodePrediction {
  premul: CensusVec3;
  coverage: number;
  categorical: FrameRecord | null;
  node: DirectionNode;
  weight: number;
  delta: readonly [number, number];
}

interface Prediction {
  premul: CensusVec3;
  coverage: number;
  family: number;
  point: CensusVec3 | null;
  cellDelta: number;
  winnerSigma: number;
}

interface PixelResult {
  ray: CameraRay;
  prediction: Prediction;
  truthRgb: CensusVec3;
  predictedRgb: CensusVec3;
  rgbError: number;
  truthHit: boolean;
  predictedHit: boolean;
  geometryError: number | null;
}

interface Metrics {
  rays: number;
  intersection: number;
  union: number;
  rgbErrors: number[];
  geometryErrors: number[];
  crispGeometryErrors: number[];
  coverageErrors: number;
  wrongComponents: number[];
  edgeBounds: number[];
}

function sha256(value: Uint8Array | string): string {
  return createHash('sha256').update(value).digest('hex');
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(',')}}`;
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

function mod(value: number, divisor: number): number {
  return ((value % divisor) + divisor) % divisor;
}

function normalize(value: CensusVec3): CensusVec3 {
  const length = Math.hypot(value[0], value[1], value[2]);
  return [value[0] / length, value[1] / length, value[2] / length];
}

function quantiles(input: readonly number[]): Record<string, number | null> {
  if (input.length === 0) return { p50: null, p90: null, p95: null, p99: null, maximum: null };
  const sorted = [...input].sort((a, b) => a - b);
  const at = (f: number): number => sorted[Math.floor((sorted.length - 1) * f)]!;
  return { p50: at(0.5), p90: at(0.9), p95: at(0.95), p99: at(0.99), maximum: sorted.at(-1)! };
}

function q95(input: readonly number[], fallback: number): number {
  if (input.length === 0) return fallback;
  const sorted = [...input].sort((a, b) => a - b);
  return sorted[Math.floor((sorted.length - 1) * 0.95)]!;
}

function nearestFamily(color: CensusVec3, palette: typeof PALETTES.low | typeof PALETTES.tall): number {
  let best = 0;
  let distance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < palette.length; index++) {
    const entry = palette[index]!;
    const d = (entry[0] - color[0]) ** 2 + (entry[1] - color[1]) ** 2 + (entry[2] - color[2]) ** 2;
    if (d < distance) { distance = d; best = index; }
  }
  return FAMILY_FOR_PALETTE_ENTRY[best]!;
}

function geometryFromLowFixture(): { geometry: DecodedOwnedProfileGeometry; colors: Float64Array; fixtureHash: string } {
  const fixture = makeEstonianGraminoidFixture(ESTONIAN_GRAMINOID_PROFILE_IDS.AGROSTIS_CAPILLARIS);
  const positions = Float64Array.from(fixture.mesh.positions);
  const triangles = Uint32Array.from(fixture.mesh.indices);
  const colors = Float64Array.from(fixture.mesh.colors ?? []);
  if (colors.length !== positions.length) throw new Error('Agrostis fixture has no complete vertex colours');
  const minimum = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const maximum = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
  for (let vertex = 0; vertex < positions.length; vertex += 3) {
    for (let axis = 0; axis < 3; axis++) {
      minimum[axis] = Math.min(minimum[axis]!, positions[vertex + axis]!);
      maximum[axis] = Math.max(maximum[axis]!, positions[vertex + axis]!);
    }
  }
  const geometry: DecodedOwnedProfileGeometry = {
    version: 4,
    profileId: fixture.profileId,
    topH: fixture.tile.topH,
    tileOriginX: fixture.tile.originX,
    tileOriginZ: fixture.tile.originZ,
    tileSizeX: fixture.tile.sizeX,
    tileSizeZ: fixture.tile.sizeZ,
    bounds: { min: minimum as unknown as CensusVec3, max: maximum as unknown as CensusVec3 },
    positions,
    triangles,
    vertexCount: positions.length / 3,
    triangleCount: triangles.length / 3,
  };
  return { geometry, colors, fixtureHash: sha256(canonicalJson(fixture)) };
}

function geometryFromDenseFixture(): { geometry: DecodedOwnedProfileGeometry; colors: Float64Array; fixtureHash: string } {
  const fixture = makeSphagnumCapillifoliumFixture();
  const positions = Float64Array.from(fixture.mesh.positions);
  const triangles = Uint32Array.from(fixture.mesh.indices);
  const colors = new Float64Array(positions.length);
  for (let vertex = 0; vertex < positions.length / 3; vertex++) {
    colors[vertex * 3] = 0.19;
    colors[vertex * 3 + 1] = 0.37;
    colors[vertex * 3 + 2] = 0.10;
  }
  const minimum = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const maximum = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
  for (let vertex = 0; vertex < positions.length; vertex += 3) for (let axis = 0; axis < 3; axis++) {
    minimum[axis] = Math.min(minimum[axis]!, positions[vertex + axis]!);
    maximum[axis] = Math.max(maximum[axis]!, positions[vertex + axis]!);
  }
  return {
    geometry: {
      version: 4,
      profileId: 5,
      topH: fixture.tile.topH,
      tileOriginX: fixture.tile.originX,
      tileOriginZ: fixture.tile.originZ,
      tileSizeX: fixture.tile.sizeX,
      tileSizeZ: fixture.tile.sizeZ,
      bounds: { min: minimum as unknown as CensusVec3, max: maximum as unknown as CensusVec3 },
      positions,
      triangles,
      vertexCount: positions.length / 3,
      triangleCount: triangles.length / 3,
    },
    colors,
    fixtureHash: sha256(canonicalJson(fixture)),
  };
}

function tallVertexColors(bytes: Uint8Array, geometry: DecodedOwnedProfileGeometry): Float64Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const vertexOffset = view.getUint32(84, true);
  const colors = new Float64Array(geometry.vertexCount * 3);
  for (let vertex = 0; vertex < geometry.vertexCount; vertex++) {
    const source = vertexOffset + vertex * 16;
    const target = vertex * 3;
    colors[target] = view.getUint16(source + 6, true) / 65535;
    colors[target + 1] = view.getUint16(source + 8, true) / 65535;
    colors[target + 2] = view.getUint16(source + 10, true) / 65535;
  }
  return colors;
}

function makeCommunities(includeTall = true): Community[] {
  const low = geometryFromLowFixture();
  const lowPacked = readFileSync(LOW_PACKED_SOURCE);
  const build = (
    key: Community['key'], label: string, geometry: DecodedOwnedProfileGeometry,
    colors: Float64Array, minimumEyeHeight: number, sources: Record<string, unknown>,
  ): Community => ({
    key, label, geometry, colors, minimumEyeHeight, sources,
    meanHeight: (geometry.bounds.min[1] + geometry.bounds.max[1]) * 0.5,
    bvh: TriangleBvh.build(geometry, 8),
  });
  const communities: Community[] = [
    build('low', 'Agrostis capillaris sparse open authored stand', low.geometry, low.colors, 1.6, {
      packedPath: relative(WORKSPACE, LOW_PACKED_SOURCE), packedSha256: sha256(lowPacked),
      deterministicFixtureSha256: low.fixtureHash,
    }),
  ];
  if (includeTall) {
    const tallBytes = readFileSync(TALL_SOURCE);
    if (sha256(tallBytes) !== TALL_SHA256) throw new Error('Calamagrostis source hash changed');
    const tallGeometry = decodeOwnedProfileGeometry(tallBytes);
    communities.push(build('tall', 'Calamagrostis canescens accepted tall community', tallGeometry,
      tallVertexColors(tallBytes, tallGeometry), 0.4,
      { packedPath: relative(WORKSPACE, TALL_SOURCE), packedSha256: TALL_SHA256 }));
  }
  return communities;
}

function makeDenseControlCommunity(): Community {
  const fixture = geometryFromDenseFixture();
  const packed = readFileSync(DENSE_PACKED_SOURCE);
  return {
    key: 'low',
    label: 'Sphagnum capillifolium connected dense carpet control',
    geometry: fixture.geometry,
    colors: fixture.colors,
    bvh: TriangleBvh.build(fixture.geometry, 8),
    meanHeight: (fixture.geometry.bounds.min[1] + fixture.geometry.bounds.max[1]) * 0.5,
    minimumEyeHeight: 1.6,
    sources: {
      packedPath: relative(WORKSPACE, DENSE_PACKED_SOURCE),
      packedSha256: sha256(packed),
      deterministicFixtureSha256: fixture.fixtureHash,
      regime: 'connected carpet; near-total top-down interception required by Section 12',
    },
  };
}

function surfaceFromHit(
  community: Community, origin: CensusVec3, direction: CensusVec3, hit: OriginAwareTruthHit,
): Surface {
  const { geometry, colors } = community;
  const point: CensusVec3 = [
    origin[0] + hit.t * direction[0],
    origin[1] + hit.t * direction[1],
    origin[2] + hit.t * direction[2],
  ];
  const tri = hit.triangleId * 3;
  const ids = [geometry.triangles[tri]!, geometry.triangles[tri + 1]!, geometry.triangles[tri + 2]!] as const;
  const p = (id: number): CensusVec3 => {
    const offset = id * 3;
    return [geometry.positions[offset]!, geometry.positions[offset + 1]!, geometry.positions[offset + 2]!];
  };
  const a = p(ids[0]); const b = p(ids[1]); const c = p(ids[2]);
  const local: CensusVec3 = [
    point[0] - hit.copyX * geometry.tileSizeX,
    point[1],
    point[2] - hit.copyZ * geometry.tileSizeZ,
  ];
  const e0: CensusVec3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const e1: CensusVec3 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const q: CensusVec3 = [local[0] - a[0], local[1] - a[1], local[2] - a[2]];
  const d00 = e0[0] ** 2 + e0[1] ** 2 + e0[2] ** 2;
  const d01 = e0[0] * e1[0] + e0[1] * e1[1] + e0[2] * e1[2];
  const d11 = e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2;
  const d20 = q[0] * e0[0] + q[1] * e0[1] + q[2] * e0[2];
  const d21 = q[0] * e1[0] + q[1] * e1[1] + q[2] * e1[2];
  const denominator = d00 * d11 - d01 * d01;
  const u = denominator === 0 ? 0 : (d11 * d20 - d01 * d21) / denominator;
  const v = denominator === 0 ? 0 : (d00 * d21 - d01 * d20) / denominator;
  const weights = [1 - u - v, u, v] as const;
  const color: number[] = [0, 0, 0];
  for (let channel = 0; channel < 3; channel++) {
    color[channel] = clamp(ids.reduce((sum, id, corner) => sum + weights[corner]! * colors[id * 3 + channel]!, 0), 0, 1);
  }
  const normal = normalize([
    e0[1] * e1[2] - e0[2] * e1[1],
    e0[2] * e1[0] - e0[0] * e1[2],
    e0[0] * e1[1] - e0[1] * e1[0],
  ]);
  const rgb = color as unknown as CensusVec3;
  return {
    point, color: rgb, normal, family: nearestFamily(rgb, PALETTES[community.key]),
    triangleId: hit.triangleId, copyX: hit.copyX, copyZ: hit.copyZ,
  };
}

function trace(
  community: Community, origin: CensusVec3, direction: CensusVec3,
): Surface | null {
  if (!(direction[1] < -1e-12)) return null;
  const horizon = (origin[1] - community.geometry.bounds.min[1] + 1e-8) / -direction[1];
  const hit = periodicNearestSuccessor(community.geometry, community.bvh, origin, direction, horizon, 0);
  return hit ? surfaceFromHit(community, origin, direction, hit) : null;
}

function nodeFromSlope(ring: number, angularIndex: number, slope: number, count: number, cellRadius: number): DirectionNode {
  const angle = count === 1 ? 0 : angularIndex * TAU / count;
  const slopeX = slope * Math.cos(angle);
  const slopeZ = slope * Math.sin(angle);
  return {
    ring, angularIndex: mod(angularIndex, count), slopeX, slopeZ, cellRadius,
    direction: normalize([slopeX, -1, slopeZ]),
  };
}

function phasePoint(community: Community, phaseX: number, phaseZ: number): CensusVec3 {
  return [
    community.geometry.tileOriginX + phaseX * community.geometry.tileSizeX,
    community.meanHeight,
    community.geometry.tileOriginZ + phaseZ * community.geometry.tileSizeZ,
  ];
}

function traceFramePhase(community: Community, node: DirectionNode, phaseX: number, phaseZ: number): Surface | null {
  const base = phasePoint(community, phaseX, phaseZ);
  const toTop = (community.geometry.topH - community.meanHeight) / node.direction[1];
  const origin: CensusVec3 = [
    base[0] + toTop * node.direction[0], community.geometry.topH,
    base[2] + toTop * node.direction[2],
  ];
  return trace(community, origin, node.direction);
}

function bilinearScalar(values: Float64Array, resolution: number, phaseX: number, phaseZ: number): { value: number; gx: number; gz: number } {
  const x = mod(phaseX, 1) * resolution - 0.5;
  const z = mod(phaseZ, 1) * resolution - 0.5;
  const x0 = Math.floor(x); const z0 = Math.floor(z);
  const fx = x - x0; const fz = z - z0;
  const at = (ix: number, iz: number): number => values[mod(iz, resolution) * resolution + mod(ix, resolution)]!;
  const a = at(x0, z0); const b = at(x0 + 1, z0);
  const c = at(x0, z0 + 1); const d = at(x0 + 1, z0 + 1);
  return {
    value: (a * (1 - fx) + b * fx) * (1 - fz) + (c * (1 - fx) + d * fx) * fz,
    gx: ((b - a) * (1 - fz) + (d - c) * fz) * resolution,
    gz: ((c - a) * (1 - fx) + (d - b) * fx) * resolution,
  };
}

function smoothPeriodic(values: Float64Array, resolution: number): Float64Array {
  const result = new Float64Array(values.length);
  const weights = [1, 2, 1] as const;
  for (let z = 0; z < resolution; z++) for (let x = 0; x < resolution; x++) {
    let sum = 0; let weight = 0;
    for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) {
      const w = weights[dx + 1]! * weights[dz + 1]!;
      sum += values[mod(z + dz, resolution) * resolution + mod(x + dx, resolution)]! * w;
      weight += w;
    }
    result[z * resolution + x] = sum / weight;
  }
  return result;
}

function shellMaximumGradient(community: Community, values: Float64Array): number {
  let maximum = 0;
  for (let z = 0; z < SHELL_RESOLUTION; z++) for (let x = 0; x < SHELL_RESOLUTION; x++) {
    const left = values[z * SHELL_RESOLUTION + mod(x - 1, SHELL_RESOLUTION)]!;
    const right = values[z * SHELL_RESOLUTION + mod(x + 1, SHELL_RESOLUTION)]!;
    const down = values[mod(z - 1, SHELL_RESOLUTION) * SHELL_RESOLUTION + x]!;
    const up = values[mod(z + 1, SHELL_RESOLUTION) * SHELL_RESOLUTION + x]!;
    const gx = (right - left) * SHELL_RESOLUTION / (2 * community.geometry.tileSizeX);
    const gz = (up - down) * SHELL_RESOLUTION / (2 * community.geometry.tileSizeZ);
    maximum = Math.max(maximum, Math.hypot(gx, gz));
  }
  return maximum;
}

function fitShell(community: Community, node: DirectionNode): ShellMap {
  const sums = new Float64Array(SHELL_RESOLUTION * SHELL_RESOLUTION);
  const squares = new Float64Array(SHELL_RESOLUTION * SHELL_RESOLUTION);
  const counts = new Uint32Array(sums.length);
  const samples: Array<{ px: number; pz: number; tau: number; family: number }> = [];
  for (let z = 0; z < SHELL_SOURCE_RESOLUTION; z++) for (let x = 0; x < SHELL_SOURCE_RESOLUTION; x++) {
    const px = (x + 0.5) / SHELL_SOURCE_RESOLUTION;
    const pz = (z + 0.5) / SHELL_SOURCE_RESOLUTION;
    const surface = traceFramePhase(community, node, px, pz);
    if (!surface) continue;
    const base = phasePoint(community, px, pz);
    const tau = base[1] - surface.point[1];
    const bx = Math.floor(px * SHELL_RESOLUTION);
    const bz = Math.floor(pz * SHELL_RESOLUTION);
    const index = bz * SHELL_RESOLUTION + bx;
    sums[index] += tau; squares[index] += tau * tau; counts[index]++;
    samples.push({ px, pz, tau, family: surface.family });
  }
  const values = new Float64Array(sums.length);
  let globalMean = 0;
  for (const sample of samples) globalMean += sample.tau;
  globalMean /= Math.max(1, samples.length);
  for (let index = 0; index < values.length; index++) values[index] = counts[index] ? sums[index]! / counts[index]! : Number.NaN;
  for (let pass = 0; pass < SHELL_RESOLUTION && values.some(Number.isNaN); pass++) {
    const prior = values.slice();
    for (let z = 0; z < SHELL_RESOLUTION; z++) for (let x = 0; x < SHELL_RESOLUTION; x++) {
      const index = z * SHELL_RESOLUTION + x;
      if (!Number.isNaN(prior[index]!)) continue;
      let sum = 0; let count = 0;
      for (const [dx, dz] of [[-1, 0], [1, 0], [0, -1], [0, 1]] as const) {
        const value = prior[mod(z + dz, SHELL_RESOLUTION) * SHELL_RESOLUTION + mod(x + dx, SHELL_RESOLUTION)]!;
        if (!Number.isNaN(value)) { sum += value; count++; }
      }
      if (count) values[index] = sum / count;
    }
  }
  for (let index = 0; index < values.length; index++) if (Number.isNaN(values[index]!)) values[index] = globalMean;
  const rawResidualCrisp: number[] = [];
  const rawResidualPlume: number[] = [];
  const rawLocalSpread: number[] = [];
  for (let index = 0; index < values.length; index++) if (counts[index]! > 1) {
    const mean = sums[index]! / counts[index]!;
    rawLocalSpread.push(Math.sqrt(Math.max(0, squares[index]! / counts[index]! - mean * mean)));
  }
  for (const sample of samples) {
    const residual = Math.abs(sample.tau - bilinearScalar(values, SHELL_RESOLUTION, sample.px, sample.pz).value);
    (sample.family <= 3 ? rawResidualCrisp : rawResidualPlume).push(residual);
  }
  let fitted: Float64Array = values;
  let smoothingPasses = 0;
  let constantFallback = false;
  let maximumGradient = shellMaximumGradient(community, fitted);
  while (maximumGradient * node.cellRadius > 0.5 && smoothingPasses < 4096) {
    fitted = smoothPeriodic(fitted, SHELL_RESOLUTION);
    smoothingPasses++;
    maximumGradient = shellMaximumGradient(community, fitted);
  }
  if (maximumGradient * node.cellRadius > 0.5) {
    fitted = new Float64Array(values.length);
    fitted.fill(globalMean);
    maximumGradient = 0;
    constantFallback = true;
  }
  const residualCrisp: number[] = [];
  const residualPlume: number[] = [];
  for (const sample of samples) {
    const residual = Math.abs(sample.tau - bilinearScalar(fitted, SHELL_RESOLUTION, sample.px, sample.pz).value);
    (sample.family <= 3 ? residualCrisp : residualPlume).push(residual);
  }
  return {
    values: fitted, rawResidualCrisp, rawResidualPlume, rawLocalSpread,
    residualCrisp, residualPlume,
    hitFraction: samples.length / (SHELL_SOURCE_RESOLUTION ** 2),
    smoothingPasses, constantFallback, maximumGradient,
    conditionProduct: maximumGradient * node.cellRadius,
    conditioned: maximumGradient * node.cellRadius <= 0.5,
  };
}

function pilotSigmaCurve(community: Community): { slopes: number[]; crisp: number[]; plume: number[]; reports: unknown[] } {
  const elevations = [90, 75, 60, 45, 30, 18, 10, 5, 2, 1];
  const slopes: number[] = []; const crisp: number[] = []; const plume: number[] = []; const reports: unknown[] = [];
  for (const elevation of elevations) {
    const slope = elevation === 90 ? 0 : 1 / Math.tan(elevation * DEG);
    const crispValues: number[] = []; const plumeValues: number[] = []; const shells: ShellMap[] = [];
    const azimuths = elevation === 90 ? 1 : 4;
    for (let az = 0; az < azimuths; az++) {
      const node = nodeFromSlope(0, az, slope, azimuths, INNER_SPACING);
      const shell = fitShell(community, node);
      shells.push(shell); crispValues.push(...shell.residualCrisp); plumeValues.push(...shell.residualPlume);
    }
    slopes.push(slope);
    crisp.push(Math.max(0.001, q95(crispValues, 0.001)));
    plume.push(q95(plumeValues, 0));
    reports.push({ elevationDegrees: elevation, slope, crisp: quantiles(crispValues), plume: quantiles(plumeValues), shells: shells.map((shell) => ({
      rawCrisp: quantiles(shell.rawResidualCrisp), rawPlume: quantiles(shell.rawResidualPlume),
      rawLocalStandardDeviation: quantiles(shell.rawLocalSpread),
      hitFraction: shell.hitFraction, smoothingPasses: shell.smoothingPasses,
      constantFallback: shell.constantFallback,
      maximumGradient: shell.maximumGradient, conditionProduct: shell.conditionProduct, conditioned: shell.conditioned,
    })) });
    console.error(`[shell-frame] ${community.key} pilot ${elevation}deg sigma=${crisp.at(-1)!.toFixed(4)}m`);
  }
  return { slopes, crisp, plume, reports };
}

function interpolatedSigma(curve: { slopes: number[]; crisp: number[] }, slope: number): number {
  if (slope <= curve.slopes[0]!) return curve.crisp[0]!;
  if (slope >= curve.slopes.at(-1)!) return curve.crisp.at(-1)!;
  for (let index = 0; index + 1 < curve.slopes.length; index++) {
    const a = curve.slopes[index]!; const b = curve.slopes[index + 1]!;
    if (slope <= b) {
      const f = (slope - a) / (b - a);
      return curve.crisp[index]! * (1 - f) + curve.crisp[index + 1]! * f;
    }
  }
  return curve.crisp.at(-1)!;
}

function makeLattice(community: Community, curve: { slopes: number[]; crisp: number[] }): DirectionLattice {
  const rings: DirectionRing[] = [{ slope: 0, count: 1, sigma: curve.crisp[0]!, radialStep: INNER_SPACING }];
  const denominator = K_PIXELS * community.minimumEyeHeight * REFERENCE_THETA_PIXEL;
  let slope = INNER_SPACING;
  let truncated = false;
  while (slope <= MAX_SLOPE + 1e-9) {
    const sigma = interpolatedSigma(curve, slope);
    const inner = slope <= INNER_RING_END;
    const radialStep = inner ? INNER_SPACING : clamp(denominator * slope / sigma, 0.01, 8);
    const tangentSpacing = inner ? INNER_SPACING : denominator * slope / sigma;
    const count = Math.max(6, Math.ceil(TAU * slope / Math.max(1e-6, tangentSpacing)));
    rings.push({ slope, count, sigma, radialStep });
    slope += radialStep;
    if (rings.length >= MAX_LAW_RINGS) { truncated = true; break; }
  }
  if (!truncated && rings.at(-1)!.slope < MAX_SLOPE) {
    const sigma = interpolatedSigma(curve, MAX_SLOPE);
    rings.push({ slope: MAX_SLOPE, count: Math.max(6, Math.ceil(TAU * sigma / denominator)), sigma, radialStep: 0 });
  }
  return { rings, totalDirections: rings.reduce((sum, ring) => sum + ring.count, 0), truncated };
}

/** Exact 97-node allocation retained from Experiment 024: six 16-azimuth
 * rings plus one shared vertical point. conditionSlopeRadius is the maximum
 * slope-chart distance to an adjacent triangle vertex, not a shell texel
 * size or a pooled angular statistic. */
function fixed97Lattice(): DirectionLattice {
  const elevations = [90, 85, 60, 35, 15, 5, 1] as const;
  const rings: DirectionRing[] = elevations.map((elevation, ring) => ({
    slope: elevation === 90 ? 0 : 1 / Math.tan(elevation * DEG),
    count: ring === 0 ? 1 : 16,
    sigma: 0,
    radialStep: 0,
  }));
  for (let ringIndex = 0; ringIndex < rings.length; ringIndex++) {
    const ring = rings[ringIndex]!;
    const node = nodeFromSlope(ringIndex, 0, ring.slope, ring.count, 0);
    let maximumSlopeDelta = 0;
    const include = (candidate: DirectionNode): void => {
      maximumSlopeDelta = Math.max(maximumSlopeDelta, Math.hypot(
        candidate.slopeX - node.slopeX,
        candidate.slopeZ - node.slopeZ,
      ));
    };
    if (ring.count > 1) {
      include(nodeFromSlope(ringIndex, 1, ring.slope, ring.count, 0));
    }
    for (const adjacentRingIndex of [ringIndex - 1, ringIndex + 1]) {
      const adjacent = rings[adjacentRingIndex];
      if (!adjacent) continue;
      for (const offset of adjacent.count === 1 ? [0] : [-1, 0, 1]) {
        include(nodeFromSlope(adjacentRingIndex, offset, adjacent.slope, adjacent.count, 0));
      }
    }
    ring.conditionSlopeRadius = maximumSlopeDelta;
  }
  return { rings, totalDirections: 97, truncated: false };
}

function barycentric(
  px: number, pz: number, a: DirectionNode, b: DirectionNode, c: DirectionNode,
): readonly [number, number, number] | null {
  const denominator = (b.slopeZ - c.slopeZ) * (a.slopeX - c.slopeX)
    + (c.slopeX - b.slopeX) * (a.slopeZ - c.slopeZ);
  if (Math.abs(denominator) < 1e-14) return null;
  const w0 = ((b.slopeZ - c.slopeZ) * (px - c.slopeX) + (c.slopeX - b.slopeX) * (pz - c.slopeZ)) / denominator;
  const w1 = ((c.slopeZ - a.slopeZ) * (px - c.slopeX) + (a.slopeX - c.slopeX) * (pz - c.slopeZ)) / denominator;
  const w2 = 1 - w0 - w1;
  return w0 >= -1e-8 && w1 >= -1e-8 && w2 >= -1e-8 ? [w0, w1, w2] : null;
}

function latticeTriangle(lattice: DirectionLattice, direction: CensusVec3): Array<{ node: DirectionNode; weight: number }> {
  let px = direction[0] / -direction[1]; let pz = direction[2] / -direction[1];
  let slope = Math.hypot(px, pz);
  // Section 12 deliberately reuses Experiment 024's finite 1-degree row.
  // Its sub-1-degree camera footprint belongs to the separately charged
  // fringe.  Clamp only node selection to that boundary; addressing still
  // uses the exact live slope below, so this cannot hide reconstruction
  // error behind a changed ray.
  const outerRing = lattice.rings.at(-1)!;
  const outerSlope = outerRing.slope;
  if (slope > outerSlope * Math.cos(Math.PI / outerRing.count)) {
    const angle = mod(Math.atan2(pz, px), TAU);
    const sectorAngle = TAU / outerRing.count;
    const edgeNormalAngle = (Math.floor(angle / sectorAngle) + 0.5) * sectorAngle;
    const outerPolygonRadius = outerSlope * Math.cos(sectorAngle * 0.5)
      / Math.cos(angle - edgeNormalAngle);
    const scale = Math.min(1, outerPolygonRadius * (1 - 1e-12) / slope);
    px *= scale; pz *= scale; slope = outerSlope;
  }
  const latticeNode = (ringIndex: number, angularIndex: number): DirectionNode => {
    const ring = lattice.rings[ringIndex]!;
    return nodeFromSlope(
      ringIndex,
      angularIndex,
      ring.slope,
      ring.count,
      ring.conditionSlopeRadius ?? 0,
    );
  };
  const regularCount = lattice.rings[1]?.count;
  const isRegular = regularCount !== undefined
    && lattice.rings[0]?.count === 1
    && lattice.rings.slice(1).every((ring) => ring.count === regularCount);
  if (isRegular) {
    for (let ringIndex = 0; ringIndex + 1 < lattice.rings.length; ringIndex++) {
      const inner = lattice.rings[ringIndex]!;
      const outer = lattice.rings[ringIndex + 1]!;
      if (slope > outer.slope + 1e-8) continue;
      for (let sector = 0; sector < regularCount; sector++) {
        const next = sector + 1;
        const triangles = inner.count === 1
          ? [[latticeNode(ringIndex, 0), latticeNode(ringIndex + 1, sector), latticeNode(ringIndex + 1, next)]]
          : [
            [latticeNode(ringIndex, sector), latticeNode(ringIndex + 1, sector), latticeNode(ringIndex + 1, next)],
            [latticeNode(ringIndex, sector), latticeNode(ringIndex + 1, next), latticeNode(ringIndex, next)],
          ];
        for (const triangle of triangles) {
          const weights = barycentric(px, pz, triangle[0]!, triangle[1]!, triangle[2]!);
          if (weights) return triangle.map((node, index) => ({ node, weight: Math.max(0, weights[index]!) }));
        }
      }
    }
    throw new Error(`regular direction lattice did not contain slope (${px}, ${pz})`);
  }
  let upper = lattice.rings.findIndex((ring) => ring.slope * Math.cos(Math.PI / Math.max(3, ring.count)) >= slope);
  if (upper < 0) upper = lattice.rings.length - 1;
  if (upper === 0) upper = 1;
  const lower = upper - 1;
  const ring0 = lattice.rings[lower]!; const ring1 = lattice.rings[upper]!;
  const fallbackCellRadius = Math.max(ring1.slope - ring0.slope, TAU * ring1.slope / ring1.count, TAU * ring0.slope / ring0.count);
  const angle = mod(Math.atan2(pz, px), TAU);
  const candidates0: DirectionNode[] = [];
  const candidates1: DirectionNode[] = [];
  if (ring0.count === 1) candidates0.push(nodeFromSlope(lower, 0, ring0.slope, 1, ring0.conditionSlopeRadius ?? fallbackCellRadius));
  else {
    const center = Math.floor(angle / TAU * ring0.count);
    for (let offset = -2; offset <= 2; offset++) candidates0.push(nodeFromSlope(lower, center + offset, ring0.slope, ring0.count, ring0.conditionSlopeRadius ?? fallbackCellRadius));
  }
  const center1 = Math.floor(angle / TAU * ring1.count);
  for (let offset = -2; offset <= 2; offset++) candidates1.push(nodeFromSlope(upper, center1 + offset, ring1.slope, ring1.count, ring1.conditionSlopeRadius ?? fallbackCellRadius));
  const triples: DirectionNode[][] = [];
  for (let a = 0; a < candidates0.length; a++) for (let b = a + 1; b < candidates0.length; b++) {
    for (const outer of candidates1) triples.push([candidates0[a]!, candidates0[b]!, outer]);
  }
  for (const inner of candidates0) for (let a = 0; a < candidates1.length; a++) for (let b = a + 1; b < candidates1.length; b++) {
    triples.push([inner, candidates1[a]!, candidates1[b]!]);
  }
  for (const triple of triples) {
    const weights = barycentric(px, pz, triple[0]!, triple[1]!, triple[2]!);
    if (weights) return triple.map((node, index) => ({ node, weight: weights[index]! }));
  }
  throw new Error(`direction lattice did not contain slope (${px}, ${pz})`);
}

function nodeKey(node: DirectionNode): string { return `${node.ring}:${node.angularIndex}`; }

function framePointSample(
  community: Community, node: DirectionNode, shell: ShellMap, cache: Map<string, FrameRecord>,
  texelXInput: number, texelZInput: number,
): FrameRecord {
  const texelX = mod(texelXInput, FRAME_RESOLUTION); const texelZ = mod(texelZInput, FRAME_RESOLUTION);
  const key = `${nodeKey(node)}:${texelX}:${texelZ}`;
  const found = cache.get(key); if (found) return found;
  const phaseX = (texelX + 0.5) / FRAME_RESOLUTION;
  const phaseZ = (texelZ + 0.5) / FRAME_RESOLUTION;
  const surface = traceFramePhase(community, node, phaseX, phaseZ);
  if (!surface) {
    const miss: FrameRecord = { hit: false, premul: [0, 0, 0], coverage: 0, tau: 0, residual: 0, family: -1, normal: [0, 1, 0], point: null };
    cache.set(key, miss); return miss;
  }
  const base = phasePoint(community, phaseX, phaseZ);
  const tau = base[1] - surface.point[1];
  const residual = tau - bilinearScalar(shell.values, SHELL_RESOLUTION, phaseX, phaseZ).value;
  const record: FrameRecord = { hit: true, premul: surface.color, coverage: 1, tau, residual, family: surface.family, normal: surface.normal, point: surface.point };
  cache.set(key, record); return record;
}

function filteredRecord(
  community: Community, node: DirectionNode, shell: ShellMap, cache: Map<string, FrameRecord>,
  worldX: number, worldZ: number,
): FilteredRecord {
  const phaseX = (worldX - community.geometry.tileOriginX) / community.geometry.tileSizeX;
  const phaseZ = (worldZ - community.geometry.tileOriginZ) / community.geometry.tileSizeZ;
  const x = phaseX * FRAME_RESOLUTION - 0.5; const z = phaseZ * FRAME_RESOLUTION - 0.5;
  const x0 = Math.floor(x); const z0 = Math.floor(z); const fx = x - x0; const fz = z - z0;
  const corners = [
    [x0, z0, (1 - fx) * (1 - fz)], [x0 + 1, z0, fx * (1 - fz)],
    [x0, z0 + 1, (1 - fx) * fz], [x0 + 1, z0 + 1, fx * fz],
  ] as const;
  const premul = [0, 0, 0]; let coverage = 0; let categorical: FrameRecord | null = null; let categoricalWeight = -1;
  for (const [cx, cz, weight] of corners) {
    const record = framePointSample(community, node, shell, cache, cx, cz);
    coverage += weight * record.coverage;
    for (let channel = 0; channel < 3; channel++) premul[channel] += weight * record.premul[channel]!;
    if (record.hit && weight > categoricalWeight) {
      const wrappedX = mod(cx, FRAME_RESOLUTION); const wrappedZ = mod(cz, FRAME_RESOLUTION);
      const copyX = Math.floor((cx - wrappedX) / FRAME_RESOLUTION);
      const copyZ = Math.floor((cz - wrappedZ) / FRAME_RESOLUTION);
      categorical = record.point ? {
        ...record,
        point: [
          record.point[0] + copyX * community.geometry.tileSizeX,
          record.point[1],
          record.point[2] + copyZ * community.geometry.tileSizeZ,
        ],
      } : record;
      categoricalWeight = weight;
    }
  }
  return { premul: premul as unknown as CensusVec3, coverage, categorical };
}

function nodePrediction(
  community: Community, ray: CameraRay, node: DirectionNode, weight: number, variant: Variant,
  shellCache: Map<string, ShellMap>, recordCache: Map<string, FrameRecord>,
): NodePrediction {
  const key = nodeKey(node);
  let shell = shellCache.get(key);
  if (!shell) { shell = fitShell(community, node); shellCache.set(key, shell); }
  const tMean = (community.meanHeight - ray.origin[1]) / ray.direction[1];
  const qx = ray.origin[0] + tMean * ray.direction[0]; const qz = ray.origin[2] + tMean * ray.direction[2];
  const liveSlope: readonly [number, number] = [ray.direction[0] / -ray.direction[1], ray.direction[2] / -ray.direction[1]];
  const delta: readonly [number, number] = [liveSlope[0] - node.slopeX, liveSlope[1] - node.slopeZ];
  const phaseQx = (qx - community.geometry.tileOriginX) / community.geometry.tileSizeX;
  const phaseQz = (qz - community.geometry.tileOriginZ) / community.geometry.tileSizeZ;
  const shell0 = bilinearScalar(shell.values, SHELL_RESOLUTION, phaseQx, phaseQz);
  const gradientWorld: readonly [number, number] = [shell0.gx / community.geometry.tileSizeX, shell0.gz / community.geometry.tileSizeZ];
  const denominator = 1 - gradientWorld[0] * delta[0] - gradientWorld[1] * delta[1];
  let tau = shell0.value / (Math.abs(denominator) < 1e-6 ? Math.sign(denominator || 1) * 1e-6 : denominator);
  let worldX = qx + tau * delta[0]; let worldZ = qz + tau * delta[1];
  let record = filteredRecord(community, node, shell, recordCache, worldX, worldZ);
  const corrections = variant === 'noRealign6' ? 0 : variant === 'second12' ? 2 : 1;
  for (let correction = 0; correction < corrections; correction++) {
    if (!record.categorical) break;
    // Binding Section 3.3 correction: u*=u(lambda0+r(u(lambda0))). The shell
    // denominator belongs to the smooth solve only; it is not applied again
    // to the stored residual step.
    tau += record.categorical.residual;
    worldX = qx + tau * delta[0]; worldZ = qz + tau * delta[1];
    record = filteredRecord(community, node, shell, recordCache, worldX, worldZ);
  }
  return { ...record, node, weight, delta };
}

function predict(
  community: Community, ray: CameraRay, lattice: DirectionLattice, variant: Variant,
  shellCache: Map<string, ShellMap>, recordCache: Map<string, FrameRecord>,
): Prediction {
  const angular = latticeTriangle(lattice, ray.direction);
  const winnerIndex = angular.reduce((best, item, index) => item.weight > angular[best]!.weight ? index : best, 0);
  const nodes = angular.map(({ node, weight }, index) => nodePrediction(
    community,
    ray,
    node,
    weight,
    variant === 'winner7' && index !== winnerIndex ? 'noRealign6' : variant,
    shellCache,
    recordCache,
  ));
  const premul = [0, 0, 0]; let coverage = 0;
  for (const item of nodes) {
    coverage += item.weight * item.coverage;
    for (let channel = 0; channel < 3; channel++) premul[channel] += item.weight * item.premul[channel]!;
  }
  const winner = nodes.reduce((best, item) => item.weight > best.weight ? item : best);
  const point = winner.categorical?.point ?? null;
  const px = ray.direction[0] / -ray.direction[1]; const pz = ray.direction[2] / -ray.direction[1];
  const cellDelta = Math.max(...nodes.map((item) => Math.hypot(px - item.node.slopeX, pz - item.node.slopeZ)));
  const shell = shellCache.get(nodeKey(winner.node))!;
  return {
    premul: premul as unknown as CensusVec3, coverage: clamp(coverage, 0, 1),
    family: winner.categorical?.family ?? -1, point, cellDelta,
    winnerSigma: q95(shell.residualCrisp, 0),
  };
}

function cameraDirection(spec: typeof CAMERA_SWEEPS[number], x: number, y: number): CensusVec3 {
  const azimuth = spec.azimuth * DEG; const elevation = spec.elevation * DEG;
  const forward: CensusVec3 = [Math.cos(elevation) * Math.cos(azimuth), -Math.sin(elevation), Math.cos(elevation) * Math.sin(azimuth)];
  const right: CensusVec3 = [-Math.sin(azimuth), 0, Math.cos(azimuth)];
  const up: CensusVec3 = [Math.sin(elevation) * Math.cos(azimuth), Math.cos(elevation), Math.sin(elevation) * Math.sin(azimuth)];
  const sx = ((x + 0.5) / CAMERA_WIDTH * 2 - 1) * Math.tan(spec.hFov * DEG * 0.5);
  const sy = (1 - (y + 0.5) / CAMERA_HEIGHT * 2) * Math.tan(spec.vFov * DEG * 0.5);
  return normalize([forward[0] + right[0] * sx + up[0] * sy, forward[1] + up[1] * sy, forward[2] + right[2] * sx + up[2] * sy]);
}

function cameraRays(community: Community, spec: typeof CAMERA_SWEEPS[number], lateralShift = 0): CameraRay[] {
  const azimuth = spec.azimuth * DEG; const right: CensusVec3 = [-Math.sin(azimuth), 0, Math.cos(azimuth)];
  const height = community.key === 'low' ? Math.max(spec.height, 1.6) : spec.height;
  const origin: CensusVec3 = [
    community.geometry.tileOriginX + community.geometry.tileSizeX * 0.38196601125 + right[0] * lateralShift,
    community.geometry.topH + height,
    community.geometry.tileOriginZ + community.geometry.tileSizeZ * 0.61803398875 + right[2] * lateralShift,
  ];
  const rays: CameraRay[] = [];
  for (let y = 0; y < CAMERA_HEIGHT; y++) for (let x = 0; x < CAMERA_WIDTH; x++) {
    const direction = cameraDirection(spec, x, y);
    rays.push({ x, y, origin, direction, truth: trace(community, origin, direction) });
  }
  return rays;
}

function composite(premul: CensusVec3, alpha: number): CensusVec3 {
  return [premul[0] + BACKGROUND[0] * (1 - alpha), premul[1] + BACKGROUND[1] * (1 - alpha), premul[2] + BACKGROUND[2] * (1 - alpha)];
}

function evaluate(ray: CameraRay, prediction: Prediction): PixelResult {
  const truthHit = ray.truth !== null; const predictedHit = prediction.coverage >= COVERAGE_THRESHOLD;
  const truthRgb = truthHit ? ray.truth!.color : BACKGROUND;
  const predictedRgb = composite(prediction.premul, prediction.coverage);
  const rgbError = Math.max(...truthRgb.map((value, channel) => Math.abs(value - predictedRgb[channel]!)));
  let geometryError: number | null = null;
  if (ray.truth && prediction.point) {
    const dx = prediction.point[0] - ray.origin[0]; const dy = prediction.point[1] - ray.origin[1]; const dz = prediction.point[2] - ray.origin[2];
    const projectedT = dx * ray.direction[0] + dy * ray.direction[1] + dz * ray.direction[2];
    const truthDx = ray.truth.point[0] - ray.origin[0]; const truthDy = ray.truth.point[1] - ray.origin[1]; const truthDz = ray.truth.point[2] - ray.origin[2];
    const truthT = truthDx * ray.direction[0] + truthDy * ray.direction[1] + truthDz * ray.direction[2];
    geometryError = Math.abs(projectedT - truthT);
  }
  return { ray, prediction, truthRgb, predictedRgb, rgbError, truthHit, predictedHit, geometryError };
}

function emptyMetrics(): Metrics {
  return { rays: 0, intersection: 0, union: 0, rgbErrors: [], geometryErrors: [], crispGeometryErrors: [], coverageErrors: 0, wrongComponents: [], edgeBounds: [] };
}

function addMetrics(metrics: Metrics, pixels: readonly PixelResult[], include?: readonly boolean[]): void {
  for (let pixelIndex = 0; pixelIndex < pixels.length; pixelIndex++) {
    if (include && !include[pixelIndex]) continue;
    const pixel = pixels[pixelIndex]!;
    metrics.rays++;
    if (pixel.truthHit && pixel.predictedHit) metrics.intersection++;
    if (pixel.truthHit || pixel.predictedHit) metrics.union++;
    metrics.rgbErrors.push(pixel.rgbError);
    if (pixel.truthHit !== pixel.predictedHit) metrics.coverageErrors++;
    if (pixel.geometryError !== null) {
      metrics.geometryErrors.push(pixel.geometryError);
      if (pixel.ray.truth!.family <= 3) metrics.crispGeometryErrors.push(pixel.geometryError);
    }
    metrics.edgeBounds.push(2 * pixel.prediction.cellDelta * pixel.prediction.winnerSigma);
  }
  const wrong = pixels.map((pixel, index) => (!include || include[index]!)
    && (pixel.truthHit !== pixel.predictedHit || pixel.rgbError > 0.15));
  const visited = new Uint8Array(wrong.length);
  for (let start = 0; start < wrong.length; start++) {
    if (!wrong[start] || visited[start]) continue;
    let size = 0; const queue = [start]; visited[start] = 1;
    for (let cursor = 0; cursor < queue.length; cursor++) {
      const index = queue[cursor]!; size++;
      const x = index % CAMERA_WIDTH; const y = Math.floor(index / CAMERA_WIDTH);
      for (const candidate of [x > 0 ? index - 1 : -1, x + 1 < CAMERA_WIDTH ? index + 1 : -1, y > 0 ? index - CAMERA_WIDTH : -1, y + 1 < CAMERA_HEIGHT ? index + CAMERA_WIDTH : -1]) {
        if (candidate >= 0 && wrong[candidate] && !visited[candidate]) { visited[candidate] = 1; queue.push(candidate); }
      }
    }
    metrics.wrongComponents.push(size / pixels.length);
  }
}

function screenBands(
  pixels: readonly PixelResult[], radiusPixels = 2,
): { silhouetteCrest: boolean[]; inBandContent: boolean[] } {
  const silhouetteCrest = new Array<boolean>(pixels.length).fill(false);
  const inBandContent = new Array<boolean>(pixels.length).fill(false);
  for (let x = 0; x < CAMERA_WIDTH; x++) {
    let crest = CAMERA_HEIGHT;
    let floor = -1;
    for (let y = 0; y < CAMERA_HEIGHT; y++) {
      if (!pixels[y * CAMERA_WIDTH + x]!.truthHit) continue;
      crest = Math.min(crest, y);
      floor = Math.max(floor, y);
    }
    if (floor < 0) continue;
    const crestStart = Math.max(0, crest - radiusPixels);
    const crestEnd = Math.min(CAMERA_HEIGHT - 1, crest + radiusPixels);
    for (let y = crestStart; y <= crestEnd; y++) silhouetteCrest[y * CAMERA_WIDTH + x] = true;
    for (let y = crestEnd + 1; y <= floor; y++) inBandContent[y * CAMERA_WIDTH + x] = true;
  }
  return { silhouetteCrest, inBandContent };
}

function metricsReport(metrics: Metrics): Record<string, unknown> {
  return {
    rays: metrics.rays,
    silhouetteIoU: metrics.intersection / Math.max(1, metrics.union),
    compositedRgbMaxChannel: quantiles(metrics.rgbErrors),
    coverageErrorFraction: metrics.coverageErrors / Math.max(1, metrics.rays),
    winnerGeometry: quantiles(metrics.geometryErrors),
    crispWinnerGeometry: quantiles(metrics.crispGeometryErrors),
    connectedWrongViewRegionFraction: quantiles(metrics.wrongComponents),
    largestConnectedWrongViewRegionFraction: Math.max(0, ...metrics.wrongComponents),
    predictedEdgeDoublingBoundMetres: quantiles(metrics.edgeBounds),
  };
}

function temporalSlice(
  frames: readonly PixelResult[][],
  includePair?: (before: PixelResult, after: PixelResult, frame: number, pixel: number) => boolean,
): Record<string, unknown> {
  let stable = 0; let changed = 0;
  for (let frame = 1; frame < frames.length; frame++) for (let pixel = 0; pixel < frames[frame]!.length; pixel++) {
    const before = frames[frame - 1]![pixel]!; const after = frames[frame]![pixel]!;
    if (includePair && !includePair(before, after, frame, pixel)) continue;
    if (before.ray.truth && after.ray.truth && before.ray.truth.family === after.ray.truth.family) {
      stable++;
      if (before.prediction.family !== after.prediction.family) changed++;
    }
  }
  return { stableTruthClassPairs: stable, unforcedClassChanges: changed, unforcedClassChangeFraction: changed / Math.max(1, stable) };
}

function temporalReport(frames: readonly PixelResult[][]): Record<string, unknown> {
  const bands = frames.map((frame) => screenBands(frame));
  return {
    all: temporalSlice(frames),
    silhouetteBand: temporalSlice(frames, (_before, _after, frame, pixel) =>
      bands[frame - 1]!.silhouetteCrest[pixel]! || bands[frame]!.silhouetteCrest[pixel]!),
    inBandContent: temporalSlice(frames, (before, after, frame, pixel) =>
      before.truthHit && after.truthHit
      && bands[frame - 1]!.inBandContent[pixel]! && bands[frame]!.inBandContent[pixel]!),
  };
}

function rgb8(color: CensusVec3): readonly [number, number, number] {
  return color.map((value) => clamp(Math.round(Math.pow(clamp(value, 0, 1), 1 / 2.2) * 255), 0, 255)) as unknown as readonly [number, number, number];
}

function diagnosticRow(byVariant: Record<Variant, PixelResult[]>): Uint8Array {
  const panels = 5; const width = CAMERA_WIDTH * panels;
  const output = new Uint8Array(width * CAMERA_HEIGHT * 3);
  for (let index = 0; index < byVariant.strict9.length; index++) {
    const aligned = byVariant.strict9[index]!;
    const colors = [
      rgb8(aligned.truthRgb), rgb8(byVariant.noRealign6[index]!.predictedRgb),
      rgb8(byVariant.winner7[index]!.predictedRgb), rgb8(aligned.predictedRgb),
      aligned.rgbError < 0.05 ? [15, 70, 25] : aligned.rgbError < 0.15 ? [195, 190, 30] : [225, 15, 45],
    ] as const;
    for (let panel = 0; panel < panels; panel++) {
      const offset = (aligned.ray.y * width + panel * CAMERA_WIDTH + aligned.ray.x) * 3;
      output[offset] = colors[panel]![0]; output[offset + 1] = colors[panel]![1]; output[offset + 2] = colors[panel]![2];
    }
  }
  return output;
}

async function writeQa(path: string, rows: readonly Uint8Array[]): Promise<void> {
  const width = CAMERA_WIDTH * 5; const height = CAMERA_HEIGHT * rows.length;
  const output = new Uint8Array(width * height * 3);
  rows.forEach((row, index) => output.set(row, index * width * CAMERA_HEIGHT * 3));
  await sharp(output, { raw: { width, height, channels: 3 } }).resize(width * QA_SCALE, height * QA_SCALE, { kernel: 'nearest' }).png().toFile(path);
}

const residualAuditOnly = process.argv.includes('--residual-audit-low');
const denseControlAuditOnly = process.argv.includes('--dense-control-audit');
const sparseReconstructionOnly = process.argv.includes('--sparse-reconstruction');
const scriptPath = fileURLToPath(import.meta.url);
const configuration = {
  schema: 'laas-shell-frame-field-gate/v1',
  phase: 'offline-only',
  frameResolution: FRAME_RESOLUTION,
  shellResolution: SHELL_RESOLUTION,
  shellSourceResolution: SHELL_SOURCE_RESOLUTION,
  camera: { width: CAMERA_WIDTH, height: CAMERA_HEIGHT, sweeps: CAMERA_SWEEPS, shifts: CAMERA_SHIFTS },
  latticeLaw: { kPixels: K_PIXELS, referenceThetaPixel: REFERENCE_THETA_PIXEL, innerRingEnd: INNER_RING_END, innerSpacing: INNER_SPACING, maximumSlope: MAX_SLOPE },
  variants: { noRealign6: 6, winner7: 7, strict9: 9, second12: 12 },
  constraints: { noRuntimeLoop: true, noMarch: true, noCandidateList: true, categoricalGeometry: true, coverOnly: true },
  executionMode: denseControlAuditOnly
    ? 'dense-connected-control-audit'
    : sparseReconstructionOnly
      ? 'fixed97-sparse-reconstruction-gate'
      : residualAuditOnly
        ? 'legacy-sparse-residual-premise-audit'
        : 'complete-two-community-gate',
};
const recipeSha256 = sha256(canonicalJson({ ...configuration, scriptSha256: sha256(readFileSync(scriptPath)) }));
const sourceSetSha256 = sha256(canonicalJson({
  tall: TALL_SHA256,
  lowPacked: sha256(readFileSync(LOW_PACKED_SOURCE)),
  densePacked: sha256(readFileSync(DENSE_PACKED_SOURCE)),
}));
const outputRoot = resolve(WORKSPACE, `data/work/groundcover-shell-frame-field/${sourceSetSha256.slice(0, 16)}/${recipeSha256.slice(0, 16)}`);
mkdirSync(resolve(outputRoot, 'qa'), { recursive: true });

if (denseControlAuditOnly) {
  const community = makeDenseControlCommunity();
  const lattice = fixed97Lattice();
  const started = performance.now();
  const perView: Record<string, unknown> = {};
  for (let ringIndex = 0; ringIndex < lattice.rings.length; ringIndex++) {
    const ring = lattice.rings[ringIndex]!;
    const azimuthIndices = ring.count === 1 ? [0] : [0, 4, 8, 12];
    for (const angularIndex of azimuthIndices) {
      const node = nodeFromSlope(ringIndex, angularIndex, ring.slope, ring.count, ring.conditionSlopeRadius!);
      const shell = fitShell(community, node);
      const elevation = Math.atan2(1, ring.slope) / DEG;
      const key = `e${elevation.toFixed(0)}-a${angularIndex * 360 / ring.count}`;
      perView[key] = {
        elevationDegrees: elevation,
        azimuthDegrees: angularIndex * 360 / ring.count,
        directionCellSlopeRadius: ring.conditionSlopeRadius,
        hitFraction: shell.hitFraction,
        rawCrisp: quantiles(shell.rawResidualCrisp),
        rawPlume: quantiles(shell.rawResidualPlume),
        rawLocalStandardDeviation: quantiles(shell.rawLocalSpread),
        conditionedCrisp: quantiles(shell.residualCrisp),
        conditionedPlume: quantiles(shell.residualPlume),
        smoothingPasses: shell.smoothingPasses,
        constantFallback: shell.constantFallback,
        maximumGradient: shell.maximumGradient,
        conditionProduct: shell.conditionProduct,
        conditioned: shell.conditioned,
      };
      console.error(`[shell-frame] dense ${key} hit=${shell.hitFraction.toFixed(4)} raw-p95=${q95(shell.rawResidualCrisp, 0).toFixed(4)}m conditioned-p95=${q95(shell.residualCrisp, 0).toFixed(4)}m`);
    }
  }
  const topDown = perView['e90-a0'] as Record<string, any>;
  const audit = {
    schema: 'laas-shell-frame-field-dense-control-audit/v1',
    result: topDown.hitFraction >= 0.95 ? 'MEASURED' : 'INVALID_CONTROL',
    purpose: 'Section 12 gate 1: measure the crest-forming dense regime without pooling independent per-view shells.',
    configuration,
    source: { label: community.label, ...community.sources, triangles: community.geometry.triangleCount, vertices: community.geometry.vertexCount,
      tileSize: [community.geometry.tileSizeX, community.geometry.tileSizeZ], topH: community.geometry.topH },
    fixedDirectionLattice: { directionCount: lattice.totalDirections, rings: lattice.rings },
    topDownInterceptionRequirement: { threshold: 0.95, measured: topDown.hitFraction, passes: topDown.hitFraction >= 0.95 },
    perView,
    elapsedMilliseconds: performance.now() - started,
    interpretation: [
      'Every entry is one view shell; residuals are never pooled across azimuths.',
      'Raw residuals use the unsmoothed 32x32 cover-hit local mean. Conditioning uses tan of that direction node\'s spherical adjacent-triangle radius from the fixed 97-node lattice.',
      'The connected authored carpet makes this a genuine near-total top-down dense control; it is not the prior 8.45%-hit open Agrostis stand.',
    ],
  };
  const serialized = `${JSON.stringify(audit, null, 2)}\n`;
  writeFileSync(resolve(outputRoot, 'metrics.json'), serialized);
  writeFileSync(resolve(outputRoot, 'index.json'), `${JSON.stringify({
    schema: 'laas-shell-frame-field-gate-index/v1', result: audit.result, sourceSetSha256, recipeSha256,
    files: [{ file: 'metrics.json', sha256: sha256(serialized), bytes: Buffer.byteLength(serialized), interpretation: 'Section 12 dense-control per-view shell audit' }],
  }, null, 2)}\n`);
  console.error(`[shell-frame] dense control ${audit.result} wrote ${outputRoot}`);
  process.stdout.write(`${outputRoot}\n`);
  process.exit(0);
}

if (residualAuditOnly) {
  const community = makeCommunities(false)[0]!;
  console.error(`[shell-frame] ${community.key} BVH ${community.bvh.metrics.buildMilliseconds.toFixed(0)} ms`);
  const started = performance.now();
  const pilot = pilotSigmaCurve(community);
  const lattice = makeLattice(community, pilot);
  const audit = {
    schema: 'laas-shell-frame-field-residual-premise-audit/v1',
    result: 'RED',
    reason: 'The expected-easy community does not have the claimed millimetric or grazing-decreasing crisp shell residual, so the specified lattice law exceeds the Exp-024 direction budget before reconstruction is evaluated.',
    configuration,
    source: { label: community.label, ...community.sources, triangles: community.geometry.triangleCount, vertices: community.geometry.vertexCount,
      tileSize: [community.geometry.tileSizeX, community.geometry.tileSizeZ], topH: community.geometry.topH },
    parameterization: {
      frameChart: 'periodic XZ phase at the chart mean-height plane; affine-equivalent to the orthogonal frame chart for every finite descending direction',
      depth: 'vertical lambda in L_i(u,lambda)=B(u)+lambda(s_i.x,-1,s_i.z)',
      epipolarRelation: 'u=q+lambda(s_live-s_i)',
      note: 'Using unit-ray distance is algebraically equivalent only if its slope coefficient is inversely rescaled; vertical lambda is the coordinate used by the Section 5.2 Delta-s times sigma law.',
    },
    pilotResiduals: pilot.reports,
    pooledConditionedCrispP95Metres: pilot.crisp,
    pooledConditionedPlumeP95Metres: pilot.plume,
    emittedLattice: { ringCount: lattice.rings.length, directionCount: lattice.totalDirections, truncated: lattice.truncated, rings: lattice.rings,
      baselineDirectionCount: EXP024_DIRECTIONS, directionBudgetPass: lattice.totalDirections <= EXP024_DIRECTIONS },
    elapsedMilliseconds: performance.now() - started,
    interpretation: [
      'The shell is the specified cover-hit local mean at 32x32, measured from a 192x192 first-hit frame; misses are excluded and periodically extrapolated.',
      'Raw unsmoothed residuals and per-shell-cell standard deviations are reported separately from conditioning-smoothed residuals, preventing the conditioning pass from being blamed for the premise result.',
      'Crisp includes foliage, culm, rhizome, and panicle-axis families; plume/fuzz is reported separately and does not size the lattice.',
      'No shader/runtime path was touched. No radiance or geometry threshold is claimed because the prerequisite direction/memory law is already RED.',
    ],
  };
  const serialized = `${JSON.stringify(audit, null, 2)}\n`;
  writeFileSync(resolve(outputRoot, 'metrics.json'), serialized);
  const index = { schema: 'laas-shell-frame-field-gate-index/v1', result: 'RED', sourceSetSha256, recipeSha256,
    files: [{ file: 'metrics.json', sha256: sha256(serialized), bytes: Buffer.byteLength(serialized), interpretation: 'dense low-community residual premise audit' }] };
  writeFileSync(resolve(outputRoot, 'index.json'), `${JSON.stringify(index, null, 2)}\n`);
  console.error(`[shell-frame] RED residual premise wrote ${outputRoot}`);
  process.stdout.write(`${outputRoot}\n`);
  process.exit(0);
}

const reports: Record<string, unknown> = {};
const qaEntries: Array<Record<string, unknown>> = [];
for (const community of makeCommunities(!sparseReconstructionOnly)) {
  console.error(`[shell-frame] ${community.key} BVH ${community.bvh.metrics.buildMilliseconds.toFixed(0)} ms`);
  const started = performance.now();
  const pilot = sparseReconstructionOnly ? null : pilotSigmaCurve(community);
  const lattice = sparseReconstructionOnly ? fixed97Lattice() : makeLattice(community, pilot!);
  console.error(`[shell-frame] ${community.key} lattice ${lattice.rings.length} rings / ${lattice.totalDirections} directions`);
  const shellCache = new Map<string, ShellMap>();
  const recordCache = new Map<string, FrameRecord>();
  const aggregate = { noRealign6: emptyMetrics(), winner7: emptyMetrics(), strict9: emptyMetrics(), second12: emptyMetrics() };
  const aggregateSilhouette = { noRealign6: emptyMetrics(), winner7: emptyMetrics(), strict9: emptyMetrics(), second12: emptyMetrics() };
  const aggregateInBand = { noRealign6: emptyMetrics(), winner7: emptyMetrics(), strict9: emptyMetrics(), second12: emptyMetrics() };
  const bySweep: Record<string, unknown> = {};
  const temporal: Record<string, unknown> = {};
  const rows: Uint8Array[] = [];
  for (const spec of CAMERA_SWEEPS) {
    const raysByShift = CAMERA_SHIFTS.map((shift) => cameraRays(community, spec, shift));
    const results = {} as Record<Variant, PixelResult[]>;
    const temporalFrames = {} as Record<Variant, PixelResult[][]>;
    for (const variant of ['noRealign6', 'winner7', 'strict9', 'second12'] as const) {
      temporalFrames[variant] = raysByShift.map((frame) => frame.map((ray) =>
        evaluate(ray, predict(community, ray, lattice, variant, shellCache, recordCache))));
      results[variant] = temporalFrames[variant][0]!;
      addMetrics(aggregate[variant], results[variant]);
    }
    rows.push(diagnosticRow(results));
    const bands = screenBands(results.strict9);
    const silhouetteMask = bands.silhouetteCrest;
    const inBandMask = bands.inBandContent;
    for (const variant of ['noRealign6', 'winner7', 'strict9', 'second12'] as const) {
      addMetrics(aggregateSilhouette[variant], results[variant], silhouetteMask);
      addMetrics(aggregateInBand[variant], results[variant], inBandMask);
    }
    bySweep[spec.key] = Object.fromEntries((['noRealign6', 'winner7', 'strict9', 'second12'] as const).map((variant) => {
      const metrics = emptyMetrics(); addMetrics(metrics, results[variant]);
      const silhouette = emptyMetrics(); addMetrics(silhouette, results[variant], silhouetteMask);
      const inBand = emptyMetrics(); addMetrics(inBand, results[variant], inBandMask);
      return [variant, {
        all: metricsReport(metrics),
        silhouetteBand: { ...metricsReport(silhouette), pixelFraction: silhouette.rays / results[variant].length },
        inBandContent: { ...metricsReport(inBand), pixelFraction: inBand.rays / results[variant].length },
      }];
    }));
    for (const variant of ['noRealign6', 'winner7', 'strict9', 'second12'] as const) {
      (temporal[spec.key] ??= {} as Record<string, unknown>);
      (temporal[spec.key] as Record<string, unknown>)[variant] = temporalReport(temporalFrames[variant]);
    }
    console.error(`[shell-frame] ${community.key} ${spec.key} shells=${shellCache.size} records=${recordCache.size}`);
  }
  const qaFile = `qa/${community.key === 'low' ? '001' : '002'}-${community.key}.png`;
  await writeQa(resolve(outputRoot, qaFile), rows);
  const qaBytes = readFileSync(resolve(outputRoot, qaFile));
  qaEntries.push({ file: qaFile, sha256: sha256(qaBytes), bytes: qaBytes.byteLength, width: CAMERA_WIDTH * 5 * QA_SCALE, height: CAMERA_HEIGHT * CAMERA_SWEEPS.length * QA_SCALE,
    interpretation: 'rows 90/18/10/5/1 degrees; panels truth, 6-read no-realignment ablation, 7-read winner-only alignment, strict 9-read alignment, strict error class' });
  const rawFrameBytes = lattice.rings.reduce((sum, ring) => {
    const elevation = Math.atan2(1, ring.slope || 0) / DEG;
    const foreshortening = ring.slope === 0 ? 1 : clamp(Math.sin(elevation * DEG) + 0.15, 0.2, 1);
    return sum + ring.count * Math.ceil(FRAME_RESOLUTION * FRAME_RESOLUTION * foreshortening) * RECORD_BYTES;
  }, 0);
  const shellBytes = lattice.totalDirections * SHELL_RESOLUTION * SHELL_RESOLUTION * SHELL_BYTES;
  const summary = Object.fromEntries((['noRealign6', 'winner7', 'strict9', 'second12'] as const).map((variant) => [variant, metricsReport(aggregate[variant])]));
  const stratified = {
    silhouetteBand: Object.fromEntries((['noRealign6', 'winner7', 'strict9', 'second12'] as const)
      .map((variant) => [variant, metricsReport(aggregateSilhouette[variant])])),
    inBandContent: Object.fromEntries((['noRealign6', 'winner7', 'strict9', 'second12'] as const)
      .map((variant) => [variant, metricsReport(aggregateInBand[variant])])),
  };
  const aligned = summary.strict9 as Record<string, any>;
  const allTemporal = Object.values(temporal).map((entry) => (entry as Record<string, any>).strict9.all.unforcedClassChangeFraction as number);
  const denseBudgetGreen = community.key !== 'low' || (lattice.totalDirections <= EXP024_DIRECTIONS && rawFrameBytes + shellBytes <= EXP024_BYTES);
  const thresholds = {
    silhouetteIoU: aligned.silhouetteIoU >= 0.97,
    rgbP95: aligned.compositedRgbMaxChannel.p95 <= 0.15,
    wrongRegion: aligned.largestConnectedWrongViewRegionFraction < 0.01,
    classChange: Math.max(...allTemporal) < 0.05,
    crispGeometryP95: aligned.crispWinnerGeometry.p95 !== null && aligned.crispWinnerGeometry.p95 <= 0.05,
    denseBudget: denseBudgetGreen,
  };
  reports[community.key] = {
    label: community.label, sources: community.sources,
    geometry: { triangles: community.geometry.triangleCount, vertices: community.geometry.vertexCount, tileSize: [community.geometry.tileSizeX, community.geometry.tileSizeZ], topH: community.geometry.topH, meanHeight: community.meanHeight },
    accelerator: community.bvh.metrics, pilotResiduals: pilot?.reports ?? null,
    lattice: { rings: lattice.rings, ringCount: lattice.rings.length, directionCount: lattice.totalDirections, truncated: lattice.truncated, finiteMinimumElevationDegrees: 1,
      chargedMarginRing: true, fringeFrameChargedSeparately: true, rawFrameBytes, shellBytes, totalRawBytes: rawFrameBytes + shellBytes,
      estimatedBlockCompressedBytes: Math.ceil((rawFrameBytes + shellBytes) / 3.5), baselineDirections: EXP024_DIRECTIONS, baselineBytes: EXP024_BYTES },
    variants: summary, stratified, bySweep, temporal, thresholds,
    stratification: {
      silhouetteBand: 'per-column outer cover crest, expanded by two screen pixels above and below',
      inBandContent: 'screen pixels below the crest band through the last cover hit in that column; includes transparent gaps',
    },
    green: Object.values(thresholds).every(Boolean),
    generated: { shells: shellCache.size, pointRecords: recordCache.size, elapsedMilliseconds: performance.now() - started },
    qaFile,
  };
}

const overallGreen = Object.values(reports).every((report) => (report as Record<string, unknown>).green === true);
const report = {
  schema: configuration.schema,
  result: overallGreen ? 'GREEN' : 'RED',
  attempt: 1,
  configuration,
  reproducibility: { sourceSetSha256, recipeSha256, script: relative(WORKSPACE, scriptPath), scriptSha256: sha256(readFileSync(scriptPath)) },
  communities: reports,
  costVerdict: 'strict model is 9 fixed coherent reads; 7-read winner-only alignment is the designated low-end variant; 6-read no-realignment and 12-read second-step paths are ablations',
  interpretationBoundary: {
    runtime: 'No runtime or shader path was edited or executed.',
    loops: 'All loops and BVH traversals are offline bake/truth work. Prospective runtime variants have fixed 3-node costs.',
    categorical: 'Depth, normal, species/material mark, and winner point are selected categorically; only premultiplied radiance is filtered.',
    fringe: 'The 1-degree finite ring is scored directly. The quasi-periodic fringe is charged but cannot improve the finite gate.',
  },
};
const metrics = `${JSON.stringify(report, null, 2)}\n`;
writeFileSync(resolve(outputRoot, 'metrics.json'), metrics);
const index = {
  schema: 'laas-shell-frame-field-gate-index/v1', result: report.result, sourceSetSha256, recipeSha256,
  files: [{
    file: 'metrics.json', sha256: sha256(metrics), bytes: Buffer.byteLength(metrics),
    interpretation: sparseReconstructionOnly
      ? 'Section-12 fixed-97 sparse reconstruction gate report'
      : 'complete two-community gate report',
  }, ...qaEntries],
};
writeFileSync(resolve(outputRoot, 'index.json'), `${JSON.stringify(index, null, 2)}\n`);
console.error(`[shell-frame] ${report.result} wrote ${outputRoot}`);
process.stdout.write(`${outputRoot}\n`);
