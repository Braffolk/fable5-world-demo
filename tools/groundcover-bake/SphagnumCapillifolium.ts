import { validateIndexedMesh, type IndexedMesh, type Vec3 } from './ProfileFormat';
import type { PeriodicTile } from './PeriodicProfile';

export const SPHAGNUM_CAPILLIFOLIUM_PROFILE_ID = 5;
export const SPHAGNUM_CAPILLIFOLIUM_SPECIES = 'Sphagnum capillifolium';

export interface SphagnumCapitulumRecipe {
  centerX: number;
  centerZ: number;
  carpetY: number;
  crownY: number;
  radius: number;
  primaryBranches: number;
  forkedBranches: number;
  palettePhase: number;
}

export interface SphagnumCapillifoliumFixture {
  mesh: IndexedMesh;
  tile: PeriodicTile;
  carpetVertexCount: number;
  carpetTriangleCount: number;
  capitula: SphagnumCapitulumRecipe[];
  generator: string;
}

const TILE_SIZE = 0.24;
const CARPET_CELLS = 32;
const CAPITULUM_COUNT = 88;

const HUMMOCKS = [
  { x: 0.018, z: 0.032, rx: 0.070, rz: 0.050, yaw: 0.23, h: 0.010 },
  { x: 0.091, z: 0.019, rx: 0.052, rz: 0.078, yaw: -0.51, h: 0.015 },
  { x: 0.177, z: 0.044, rx: 0.067, rz: 0.044, yaw: 0.82, h: 0.012 },
  { x: 0.221, z: 0.112, rx: 0.058, rz: 0.080, yaw: -0.17, h: 0.014 },
  { x: 0.145, z: 0.127, rx: 0.080, rz: 0.052, yaw: 0.44, h: 0.011 },
  { x: 0.061, z: 0.113, rx: 0.061, rz: 0.057, yaw: -0.89, h: 0.016 },
  { x: 0.027, z: 0.203, rx: 0.075, rz: 0.047, yaw: 0.31, h: 0.013 },
  { x: 0.113, z: 0.211, rx: 0.054, rz: 0.072, yaw: 0.73, h: 0.010 },
  { x: 0.207, z: 0.205, rx: 0.068, rz: 0.057, yaw: -0.38, h: 0.015 },
] as const;

function torusDelta(value: number, center: number): number {
  let delta = value - center;
  delta -= Math.round(delta / TILE_SIZE) * TILE_SIZE;
  return delta;
}

function carpetHeight(x: number, z: number): number {
  let height = 0.0045;
  for (const hummock of HUMMOCKS) {
    const dx0 = torusDelta(x, hummock.x);
    const dz0 = torusDelta(z, hummock.z);
    const cy = Math.cos(hummock.yaw);
    const sy = Math.sin(hummock.yaw);
    const dx = dx0 * cy - dz0 * sy;
    const dz = dx0 * sy + dz0 * cy;
    const q = Math.hypot(dx / hummock.rx, dz / hummock.rz);
    if (q < 1) {
      const compact = 1 - q * q;
      height += hummock.h * compact * compact * (0.82 + 0.18 * Math.cos(q * Math.PI * 2.5));
    }
  }
  // Low branching ridges connect hummocks without turning the carpet into a cap union.
  const ridgeA = Math.sin((x / TILE_SIZE) * Math.PI * 6.0 + Math.sin((z / TILE_SIZE) * Math.PI * 4.0));
  const ridgeB = Math.sin((z / TILE_SIZE) * Math.PI * 8.0 - Math.sin((x / TILE_SIZE) * Math.PI * 4.0));
  height += 0.0011 * Math.max(0, ridgeA * 0.65 + ridgeB * 0.35);
  return height;
}

function normalized(value: Vec3): Vec3 {
  const length = Math.hypot(value.x, value.y, value.z);
  return { x: value.x / length, y: value.y / length, z: value.z / length };
}

function carpetNormal(x: number, z: number): Vec3 {
  const epsilon = 0.0006;
  const dx = (carpetHeight(x + epsilon, z) - carpetHeight(x - epsilon, z)) / (epsilon * 2);
  const dz = (carpetHeight(x, z + epsilon) - carpetHeight(x, z - epsilon)) / (epsilon * 2);
  return normalized({ x: -dx, y: 1, z: -dz });
}

function appendCarpet(mesh: IndexedMesh): { vertices: number; triangles: number } {
  const stride = CARPET_CELLS + 1;
  for (let iz = 0; iz <= CARPET_CELLS; iz++) {
    const z = (iz / CARPET_CELLS) * TILE_SIZE;
    for (let ix = 0; ix <= CARPET_CELLS; ix++) {
      const x = (ix / CARPET_CELLS) * TILE_SIZE;
      const y = carpetHeight(x, z);
      const normal = carpetNormal(x, z);
      mesh.positions.push(x, y, z);
      mesh.normals.push(normal.x, normal.y, normal.z);
    }
  }
  for (let iz = 0; iz < CARPET_CELLS; iz++) {
    for (let ix = 0; ix < CARPET_CELLS; ix++) {
      const a = iz * stride + ix;
      const b = a + 1;
      const c = a + stride;
      const d = c + 1;
      const flip = ((ix * 17 + iz * 29) & 1) === 0;
      if (flip) mesh.indices.push(a, c, b, b, c, d);
      else mesh.indices.push(a, c, d, a, d, b);
    }
  }
  return { vertices: stride * stride, triangles: CARPET_CELLS * CARPET_CELLS * 2 };
}

function xorshift(seedInput: number): () => number {
  let state = seedInput >>> 0;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x1_0000_0000;
  };
}

function torusDistanceSq(a: { x: number; z: number }, b: { x: number; z: number }): number {
  const dx = torusDelta(a.x, b.x);
  const dz = torusDelta(a.z, b.z);
  return dx * dx + dz * dz;
}

function densityWeight(x: number, z: number): number {
  const h = carpetHeight(x, z);
  return 0.72 + Math.min(0.55, Math.max(0, (h - 0.005) * 28));
}

/** Deterministic toroidal best-candidate placement: irregular and periodic, with no lattice. */
function placeCapitula(): Array<{ x: number; z: number; phase: number }> {
  const random = xorshift(0x53ca_911f);
  const points: Array<{ x: number; z: number; phase: number }> = [];
  for (let index = 0; index < CAPITULUM_COUNT; index++) {
    let best = { x: random() * TILE_SIZE, z: random() * TILE_SIZE, phase: random() };
    let bestScore = -1;
    const trials = index < 8 ? 24 : 48;
    for (let trial = 0; trial < trials; trial++) {
      const candidate = { x: random() * TILE_SIZE, z: random() * TILE_SIZE, phase: random() };
      const nearest = points.length === 0
        ? TILE_SIZE * TILE_SIZE
        : Math.min(...points.map((point) => torusDistanceSq(candidate, point)));
      const score = nearest * densityWeight(candidate.x, candidate.z) * (0.97 + candidate.phase * 0.06);
      if (score > bestScore) {
        best = candidate;
        bestScore = score;
      }
    }
    points.push(best);
  }
  return points;
}

function appendRing(
  mesh: IndexedMesh,
  center: Vec3,
  radius: number,
  yOffset: number,
  sides: number,
  normalTilt: number,
): number[] {
  const indices: number[] = [];
  for (let side = 0; side < sides; side++) {
    const angle = (side / sides) * Math.PI * 2;
    const cs = Math.cos(angle);
    const sn = Math.sin(angle);
    indices.push(mesh.positions.length / 3);
    mesh.positions.push(center.x + cs * radius, center.y + yOffset, center.z + sn * radius);
    const normal = normalized({ x: cs, y: normalTilt, z: sn });
    mesh.normals.push(normal.x, normal.y, normal.z);
  }
  return indices;
}

function connectRings(mesh: IndexedMesh, lower: number[], upper: number[]): void {
  const sides = lower.length;
  for (let side = 0; side < sides; side++) {
    const next = (side + 1) % sides;
    mesh.indices.push(lower[side]!, upper[side]!, lower[next]!);
    mesh.indices.push(lower[next]!, upper[side]!, upper[next]!);
  }
}

function appendStemAndCore(mesh: IndexedMesh, centerX: number, centerZ: number, baseY: number, crownY: number, phase: number): void {
  const sides = 6;
  const stemCenter = { x: centerX, y: baseY, z: centerZ };
  const lower = appendRing(mesh, stemCenter, 0.00145, 0, sides, 0.16);
  const middle = appendRing(mesh, stemCenter, 0.00125, (crownY - baseY) * 0.56, sides, 0.12);
  const upper = appendRing(mesh, stemCenter, 0.0028, crownY - baseY, sides, 0.42);
  connectRings(mesh, lower, middle);
  connectRings(mesh, middle, upper);

  const core = { x: centerX, y: crownY, z: centerZ };
  const coreLower = appendRing(mesh, core, 0.0042 + phase * 0.0006, 0, 10, 0.62);
  const coreUpper = appendRing(mesh, core, 0.0025 + phase * 0.0004, 0.0030, 10, 1.15);
  connectRings(mesh, coreLower, coreUpper);
  const top = mesh.positions.length / 3;
  mesh.positions.push(centerX, crownY + 0.0043, centerZ);
  mesh.normals.push(0, 1, 0);
  for (let side = 0; side < coreUpper.length; side++) {
    mesh.indices.push(coreUpper[side]!, top, coreUpper[(side + 1) % coreUpper.length]!);
  }
}

/** Flattened convex tube following a slightly arched radial branch. */
function appendBranch(
  mesh: IndexedMesh,
  centerX: number,
  centerZ: number,
  crownY: number,
  angle: number,
  startRadius: number,
  length: number,
  width: number,
  rise: number,
  phase: number,
): void {
  const segments = 3;
  const crossSides = 4;
  const dx = Math.cos(angle);
  const dz = Math.sin(angle);
  const lx = -dz;
  const lz = dx;
  const rings: number[][] = [];
  for (let segment = 0; segment <= segments; segment++) {
    const t = segment / segments;
    const radial = startRadius + length * t;
    const arch = rise * Math.sin(t * Math.PI) - length * 0.10 * t;
    const cx = centerX + dx * radial;
    const cz = centerZ + dz * radial;
    const cy = crownY + 0.0012 + arch;
    const halfWidth = width * (1 - t * 0.72);
    const halfThick = width * (0.34 - t * 0.18);
    const ring: number[] = [];
    const cross = [
      { x: lx * halfWidth, y: 0, z: lz * halfWidth, n: { x: lx, y: 0.35, z: lz } },
      { x: 0, y: halfThick, z: 0, n: { x: 0, y: 1, z: 0 } },
      { x: -lx * halfWidth, y: 0, z: -lz * halfWidth, n: { x: -lx, y: 0.35, z: -lz } },
      { x: 0, y: -halfThick, z: 0, n: { x: -dx * 0.2, y: -0.9, z: -dz * 0.2 } },
    ];
    for (const value of cross) {
      ring.push(mesh.positions.length / 3);
      mesh.positions.push(cx + value.x, cy + value.y, cz + value.z);
      const normal = normalized(value.n);
      mesh.normals.push(normal.x, normal.y, normal.z);
    }
    rings.push(ring);
  }
  for (let segment = 0; segment < segments; segment++) connectRings(mesh, rings[segment]!, rings[segment + 1]!);
  const tip = mesh.positions.length / 3;
  const endRadial = startRadius + length + width * (0.12 + phase * 0.08);
  mesh.positions.push(
    centerX + dx * endRadial,
    crownY + 0.0012 - length * 0.1,
    centerZ + dz * endRadial,
  );
  const tipNormal = normalized({ x: dx * 0.45, y: 0.72, z: dz * 0.45 });
  mesh.normals.push(tipNormal.x, tipNormal.y, tipNormal.z);
  const last = rings[rings.length - 1]!;
  for (let side = 0; side < crossSides; side++) mesh.indices.push(last[side]!, tip, last[(side + 1) % crossSides]!);
}

function appendCapitulum(mesh: IndexedMesh, point: { x: number; z: number; phase: number }, index: number): SphagnumCapitulumRecipe {
  const localRandom = xorshift((0x9e37_79b9 ^ Math.imul(index + 1, 0x85eb_ca6b)) >>> 0);
  const carpetY = carpetHeight(point.x, point.z);
  const hummockGain = Math.min(1, Math.max(0, (carpetY - 0.004) / 0.016));
  const crownY = carpetY + 0.011 + localRandom() * 0.007 + hummockGain * 0.003;
  const radius = 0.0080 + localRandom() * 0.0032;
  const primaryBranches = 9 + Math.floor(localRandom() * 4);
  const forkedBranches = Math.floor(primaryBranches / 3);
  const rotation = localRandom() * Math.PI * 2;
  appendStemAndCore(mesh, point.x, point.z, carpetY - 0.0003, crownY, point.phase);
  for (let branch = 0; branch < primaryBranches; branch++) {
    const angle = rotation + (branch / primaryBranches) * Math.PI * 2 + (localRandom() - 0.5) * 0.13;
    const length = radius * (0.78 + localRandom() * 0.27);
    const width = 0.00145 + localRandom() * 0.00065;
    appendBranch(mesh, point.x, point.z, crownY, angle, 0.0018, length, width, 0.0020 + localRandom() * 0.0018, point.phase);
    if (branch % 3 === 1) {
      const forkSign = ((branch + index) & 1) === 0 ? 1 : -1;
      appendBranch(
        mesh,
        point.x + Math.cos(angle) * length * 0.48,
        point.z + Math.sin(angle) * length * 0.48,
        crownY - length * 0.025,
        angle + forkSign * (0.30 + localRandom() * 0.16),
        0,
        length * (0.42 + localRandom() * 0.12),
        width * 0.72,
        0.0012,
        point.phase,
      );
    }
  }
  return {
    centerX: point.x,
    centerZ: point.z,
    carpetY,
    crownY,
    radius,
    primaryBranches,
    forkedBranches,
    palettePhase: point.phase,
  };
}

export function makeSphagnumCapillifoliumFixture(): SphagnumCapillifoliumFixture {
  const mesh: IndexedMesh = { positions: [], normals: [], indices: [] };
  const carpet = appendCarpet(mesh);
  const capitula = placeCapitula().map((point, index) => appendCapitulum(mesh, point, index));
  validateIndexedMesh(mesh);
  let maxY = -Infinity;
  for (let index = 1; index < mesh.positions.length; index += 3) {
    maxY = Math.max(maxY, mesh.positions[index] as number);
  }
  return {
    mesh,
    tile: { originX: 0, originZ: 0, sizeX: TILE_SIZE, sizeZ: TILE_SIZE, topH: maxY + 0.012 },
    carpetVertexCount: carpet.vertices,
    carpetTriangleCount: carpet.triangles,
    capitula,
    generator: 'original-procedural-sphagnum-capillifolium-vegetative-v1',
  };
}
