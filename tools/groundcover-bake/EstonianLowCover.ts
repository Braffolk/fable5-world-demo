import { validateIndexedMesh, type IndexedMesh, type Vec3 } from './ProfileFormat';
import type { PeriodicTile } from './PeriodicProfile';

export const ESTONIAN_LOW_COVER_PROFILE_IDS = [6, 7, 8, 9, 10, 11] as const;
export type EstonianLowCoverProfileId = (typeof ESTONIAN_LOW_COVER_PROFILE_IDS)[number];

export interface LowCoverMorphology {
  growthForm: 'pleurocarp-mat' | 'fruticose-lichen' | 'trifoliate-herb' | 'two-leaved-herb' | 'dwarf-shrub' | 'scale-leaved-shrub';
  shootCount: number;
  primaryAxisCount: number;
  branchCount: number;
  forkCount: number;
  leafCount: number;
  leafletCount: number;
  flowerCount: number;
  rhizomeSegments: number;
  serratedLeafCount: number;
  fourRankScaleNodes: number;
  twoLeafShootCount: number;
  edgeCrossingAxes: number;
}

export interface EstonianLowCoverFixture {
  profileId: EstonianLowCoverProfileId;
  species: string;
  generator: string;
  mesh: IndexedMesh;
  tile: PeriodicTile;
  morphology: LowCoverMorphology;
  claimBoundary: string;
}

interface Vec2 {
  x: number;
  z: number;
}

const TAU = Math.PI * 2;

function add(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function mul(a: Vec3, scale: number): Vec3 {
  return { x: a.x * scale, y: a.y * scale, z: a.z * scale };
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

function normalized(v: Vec3): Vec3 {
  const length = Math.hypot(v.x, v.y, v.z);
  if (!(length > 1e-12)) throw new Error('cannot normalize a zero vector');
  return { x: v.x / length, y: v.y / length, z: v.z / length };
}

function point(x: number, y: number, z: number): Vec3 {
  return { x, y, z };
}

function emptyMorphology(growthForm: LowCoverMorphology['growthForm']): LowCoverMorphology {
  return {
    growthForm,
    shootCount: 0,
    primaryAxisCount: 0,
    branchCount: 0,
    forkCount: 0,
    leafCount: 0,
    leafletCount: 0,
    flowerCount: 0,
    rhizomeSegments: 0,
    serratedLeafCount: 0,
    fourRankScaleNodes: 0,
    twoLeafShootCount: 0,
    edgeCrossingAxes: 0,
  };
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

function appendTube(mesh: IndexedMesh, centers: Vec3[], radii: number[], sides = 6, cap = true): void {
  if (centers.length < 2 || radii.length !== centers.length || sides < 3) throw new Error('invalid tube recipe');
  const rings: number[][] = [];
  for (let index = 0; index < centers.length; index++) {
    const tangent = normalized(index === 0
      ? sub(centers[1]!, centers[0]!)
      : index === centers.length - 1
        ? sub(centers[index]!, centers[index - 1]!)
        : sub(centers[index + 1]!, centers[index - 1]!));
    const helper = Math.abs(tangent.y) < 0.91 ? point(0, 1, 0) : point(1, 0, 0);
    const axisA = normalized(cross(tangent, helper));
    const axisB = normalized(cross(tangent, axisA));
    const ring: number[] = [];
    for (let side = 0; side < sides; side++) {
      const angle = (side / sides) * TAU;
      const normal = normalized(add(mul(axisA, Math.cos(angle)), mul(axisB, Math.sin(angle))));
      ring.push(mesh.positions.length / 3);
      const position = add(centers[index]!, mul(normal, radii[index]!));
      mesh.positions.push(position.x, position.y, position.z);
      mesh.normals.push(normal.x, normal.y, normal.z);
    }
    rings.push(ring);
  }
  for (let segment = 0; segment < rings.length - 1; segment++) {
    const current = rings[segment]!;
    const next = rings[segment + 1]!;
    for (let side = 0; side < sides; side++) {
      const after = (side + 1) % sides;
      mesh.indices.push(current[side]!, next[side]!, current[after]!);
      mesh.indices.push(current[after]!, next[side]!, next[after]!);
    }
  }
  if (!cap) return;
  const start = mesh.positions.length / 3;
  const startNormal = normalized(sub(centers[0]!, centers[1]!));
  mesh.positions.push(centers[0]!.x, centers[0]!.y, centers[0]!.z);
  mesh.normals.push(startNormal.x, startNormal.y, startNormal.z);
  const end = mesh.positions.length / 3;
  const last = centers.length - 1;
  const endNormal = normalized(sub(centers[last]!, centers[last - 1]!));
  mesh.positions.push(centers[last]!.x, centers[last]!.y, centers[last]!.z);
  mesh.normals.push(endNormal.x, endNormal.y, endNormal.z);
  for (let side = 0; side < sides; side++) {
    const after = (side + 1) % sides;
    mesh.indices.push(start, rings[0]![after]!, rings[0]![side]!);
    mesh.indices.push(end, rings[last]![side]!, rings[last]![after]!);
  }
}

function appendSurfaceFan(mesh: IndexedMesh, center: Vec3, rim: Vec3[], normalInput: Vec3): void {
  if (rim.length < 3) throw new Error('surface fan needs at least three rim vertices');
  const normal = normalized(normalInput);
  const centerIndex = mesh.positions.length / 3;
  mesh.positions.push(center.x, center.y, center.z);
  mesh.normals.push(normal.x, normal.y, normal.z);
  const rimStart = mesh.positions.length / 3;
  for (const position of rim) {
    mesh.positions.push(position.x, position.y, position.z);
    mesh.normals.push(normal.x, normal.y, normal.z);
  }
  for (let index = 0; index < rim.length; index++) {
    mesh.indices.push(centerIndex, rimStart + index, rimStart + ((index + 1) % rim.length));
  }
}

function leafBasis(yaw: number, tilt = 0): { forward: Vec3; right: Vec3; normal: Vec3 } {
  const forward = normalized(point(Math.cos(yaw) * Math.cos(tilt), Math.sin(tilt), Math.sin(yaw) * Math.cos(tilt)));
  const right = normalized(point(-Math.sin(yaw), 0, Math.cos(yaw)));
  return { forward, right, normal: normalized(cross(right, forward)) };
}

function appendOvalLeaf(
  mesh: IndexedMesh,
  base: Vec3,
  yaw: number,
  length: number,
  width: number,
  tilt: number,
  serrations = 0,
): void {
  const basis = leafBasis(yaw, tilt);
  const center = add(base, add(mul(basis.forward, length * 0.48), point(0, length * 0.025, 0)));
  const rim: Vec3[] = [];
  const samples = serrations > 0 ? Math.max(12, serrations * 2) : 12;
  for (let index = 0; index < samples; index++) {
    const angle = (index / samples) * TAU;
    const longitudinal = (Math.sin(angle) * 0.5 + 0.5) * length;
    const taper = Math.max(0, Math.sin((longitudinal / length) * Math.PI));
    const tooth = serrations > 0 && (index & 1) === 0 ? 1.13 : 1;
    const lateral = Math.cos(angle) * width * 0.5 * taper * tooth;
    const camber = Math.sin((longitudinal / length) * Math.PI) * length * 0.045;
    rim.push(add(base, add(mul(basis.forward, longitudinal), add(mul(basis.right, lateral), mul(basis.normal, camber)))));
  }
  appendSurfaceFan(mesh, center, rim, basis.normal);
}

/** Obcordate leaflet: narrow attachment, broad distal lobes, and a deep apical notch. */
function appendHeartLeaflet(mesh: IndexedMesh, base: Vec3, yaw: number, length: number, width: number, tilt: number): void {
  const basis = leafBasis(yaw, tilt);
  const local = (forward: number, right: number, camber: number): Vec3 => add(
    base,
    add(mul(basis.forward, forward), add(mul(basis.right, right), mul(basis.normal, camber))),
  );
  const rim = [
    local(0, 0, 0),
    local(length * 0.28, -width * 0.34, length * 0.018),
    local(length * 0.69, -width * 0.55, length * 0.040),
    local(length, -width * 0.30, length * 0.025),
    local(length * 0.78, 0, length * 0.012),
    local(length, width * 0.30, length * 0.025),
    local(length * 0.69, width * 0.55, length * 0.040),
    local(length * 0.28, width * 0.34, length * 0.018),
  ];
  appendSurfaceFan(mesh, local(length * 0.55, 0, length * 0.055), rim, basis.normal);
}

function appendCordateLeaf(mesh: IndexedMesh, base: Vec3, yaw: number, length: number, width: number, tilt: number): void {
  const basis = leafBasis(yaw, tilt);
  const local = (forward: number, right: number, camber: number): Vec3 => add(
    base,
    add(mul(basis.forward, forward), add(mul(basis.right, right), mul(basis.normal, camber))),
  );
  const rim = [
    local(length * 0.08, 0, 0),
    local(0, -width * 0.24, 0),
    local(length * 0.20, -width * 0.54, length * 0.025),
    local(length * 0.58, -width * 0.43, length * 0.040),
    local(length, 0, 0),
    local(length * 0.58, width * 0.43, length * 0.040),
    local(length * 0.20, width * 0.54, length * 0.025),
    local(0, width * 0.24, 0),
  ];
  appendSurfaceFan(mesh, local(length * 0.47, 0, length * 0.050), rim, basis.normal);
}

function appendScaleLeaf(mesh: IndexedMesh, base: Vec3, yaw: number, length: number, width: number, rise: number): void {
  const basis = leafBasis(yaw, 0.34);
  const rim = [
    base,
    add(base, add(mul(basis.forward, length * 0.48), mul(basis.right, -width * 0.5))),
    add(base, add(mul(basis.forward, length), point(0, rise, 0))),
    add(base, add(mul(basis.forward, length * 0.48), mul(basis.right, width * 0.5))),
  ];
  appendSurfaceFan(mesh, add(base, add(mul(basis.forward, length * 0.52), point(0, rise * 0.4, 0))), rim, basis.normal);
}

function appendTinyFlower(mesh: IndexedMesh, center: Vec3, yaw: number, radius: number, petals: number): void {
  const basis = leafBasis(yaw, 0.15);
  const rim: Vec3[] = [];
  for (let petal = 0; petal < petals; petal++) {
    const angle = (petal / petals) * TAU;
    const direction = add(mul(basis.forward, Math.cos(angle)), mul(basis.right, Math.sin(angle)));
    rim.push(add(center, mul(direction, radius)));
    rim.push(add(center, mul(direction, radius * 0.35)));
  }
  appendSurfaceFan(mesh, add(center, mul(basis.normal, radius * 0.14)), rim, basis.normal);
}

function finalizeFixture(
  profileId: EstonianLowCoverProfileId,
  species: string,
  generator: string,
  tileSize: number,
  mesh: IndexedMesh,
  morphology: LowCoverMorphology,
  claimBoundary: string,
): EstonianLowCoverFixture {
  validateIndexedMesh(mesh);
  let maxY = Number.NEGATIVE_INFINITY;
  for (let index = 1; index < mesh.positions.length; index += 3) maxY = Math.max(maxY, mesh.positions[index] as number);
  return {
    profileId,
    species,
    generator,
    mesh,
    tile: { originX: 0, originZ: 0, sizeX: tileSize, sizeZ: tileSize, topH: maxY + Math.max(0.012, maxY * 0.08) },
    morphology,
    claimBoundary,
  };
}

function toroidalAnchors(count: number, tileSize: number, seed: number): Vec2[] {
  const random = xorshift(seed);
  const points: Vec2[] = [];
  const distanceSq = (a: Vec2, b: Vec2): number => {
    let dx = a.x - b.x;
    let dz = a.z - b.z;
    dx -= Math.round(dx / tileSize) * tileSize;
    dz -= Math.round(dz / tileSize) * tileSize;
    return dx * dx + dz * dz;
  };
  for (let index = 0; index < count; index++) {
    let best = { x: random() * tileSize, z: random() * tileSize };
    let score = -1;
    for (let attempt = 0; attempt < 40; attempt++) {
      const candidate = { x: random() * tileSize, z: random() * tileSize };
      const nearest = points.length === 0 ? tileSize * tileSize : Math.min(...points.map((other) => distanceSq(candidate, other)));
      if (nearest > score) {
        best = candidate;
        score = nearest;
      }
    }
    points.push(best);
  }
  return points;
}

export function makePleuroziumSchreberiFixture(): EstonianLowCoverFixture {
  const tileSize = 0.18;
  const mesh: IndexedMesh = { positions: [], normals: [], indices: [] };
  const morphology = emptyMorphology('pleurocarp-mat');
  const random = xorshift(0x6a31_29f1);
  const anchors = toroidalAnchors(24, tileSize, 0x3106_5b9d);
  anchors.forEach((anchor, shoot) => {
    const yaw = random() * TAU;
    const length = 0.040 + random() * 0.035;
    const baseY = 0.003 + random() * 0.005;
    const centers = Array.from({ length: 5 }, (_value, index) => {
      const t = index / 4;
      const bend = Math.sin(t * Math.PI) * (random() - 0.5) * 0.012;
      return point(
        anchor.x + Math.cos(yaw) * length * t - Math.sin(yaw) * bend,
        baseY + t * (0.007 + random() * 0.007),
        anchor.z + Math.sin(yaw) * length * t + Math.cos(yaw) * bend,
      );
    });
    appendTube(mesh, centers, [0.0010, 0.0010, 0.0009, 0.00075, 0.00045], 5);
    morphology.primaryAxisCount++;
    morphology.shootCount++;
    if (centers.some((value) => value.x < 0 || value.x > tileSize || value.z < 0 || value.z > tileSize)) morphology.edgeCrossingAxes++;
    for (let node = 1; node <= 4; node++) {
      const t = node / 5;
      const stem = centers[Math.min(3, node - 1)]!;
      const side = ((node + shoot) & 1) === 0 ? 1 : -1;
      const branchYaw = yaw + side * (1.18 + random() * 0.23);
      const branchLength = length * (0.26 + random() * 0.17);
      const branch = [
        stem,
        point(stem.x + Math.cos(branchYaw) * branchLength * 0.52, stem.y + 0.0018, stem.z + Math.sin(branchYaw) * branchLength * 0.52),
        point(stem.x + Math.cos(branchYaw) * branchLength, stem.y + 0.0005, stem.z + Math.sin(branchYaw) * branchLength),
      ];
      appendTube(mesh, branch, [0.00070, 0.00055, 0.00030], 4);
      morphology.branchCount++;
      for (let leaf = 0; leaf < 4; leaf++) {
        const lt = (leaf + 0.45) / 4;
        const leafBase = point(
          stem.x + Math.cos(branchYaw) * branchLength * lt,
          stem.y + 0.0015 + Math.sin(lt * Math.PI) * 0.001,
          stem.z + Math.sin(branchYaw) * branchLength * lt,
        );
        appendOvalLeaf(mesh, leafBase, branchYaw + (((leaf + node) & 1) === 0 ? 1 : -1) * 1.18, 0.0021, 0.00145, 0.38);
        morphology.leafCount++;
      }
      const mainBase = point(
        anchor.x + Math.cos(yaw) * length * t,
        baseY + t * 0.010,
        anchor.z + Math.sin(yaw) * length * t,
      );
      appendOvalLeaf(mesh, mainBase, yaw + (((node + shoot) & 1) === 0 ? 1 : -1) * 1.32, 0.0025, 0.0017, 0.44);
      morphology.leafCount++;
    }
  });
  return finalizeFixture(
    6,
    'Pleurozium schreberi',
    'original-procedural-pleurozium-schreberi-pinnate-mat-v1',
    tileSize,
    mesh,
    morphology,
    'Vegetative pleurocarp shoot, simple pinnation, and concave leaf silhouette; no cell anatomy, sporophytes, calibrated color, or abundance claim.',
  );
}

function appendCladoniaFork(
  mesh: IndexedMesh,
  base: Vec3,
  yaw: number,
  height: number,
  radius: number,
  depth: number,
  random: () => number,
  morphology: LowCoverMorphology,
): void {
  const lean = 0.18 + random() * 0.16;
  const tip = point(base.x + Math.cos(yaw) * height * lean, base.y + height, base.z + Math.sin(yaw) * height * lean);
  const mid = point((base.x + tip.x) * 0.5, base.y + height * 0.54, (base.z + tip.z) * 0.5);
  appendTube(mesh, [base, mid, tip], [radius, radius * 0.82, radius * 0.64], 6);
  morphology.branchCount++;
  if (depth <= 0) return;
  morphology.forkCount++;
  for (const sign of [-1, 1]) {
    appendCladoniaFork(
      mesh,
      tip,
      yaw + sign * (0.48 + random() * 0.27),
      height * (0.53 + random() * 0.11),
      radius * 0.67,
      depth - 1,
      random,
      morphology,
    );
  }
}

export function makeCladoniaRangiferinaFixture(): EstonianLowCoverFixture {
  const tileSize = 0.14;
  const mesh: IndexedMesh = { positions: [], normals: [], indices: [] };
  const morphology = emptyMorphology('fruticose-lichen');
  const random = xorshift(0xc1ad_071a);
  const anchors = toroidalAnchors(30, tileSize, 0x071a_6f3d);
  anchors.forEach((anchor, index) => {
    const base = point(anchor.x, 0.0015 + random() * 0.002, anchor.z);
    const height = 0.035 + random() * 0.017;
    appendCladoniaFork(mesh, base, random() * TAU, height, 0.00058 + random() * 0.00028, index % 4 === 0 ? 3 : 2, random, morphology);
    morphology.primaryAxisCount++;
    morphology.shootCount++;
  });
  return finalizeFixture(
    7,
    'Cladonia rangiferina',
    'original-procedural-cladonia-rangiferina-dichotomous-podetia-v1',
    tileSize,
    mesh,
    morphology,
    'Dense vegetative fruticose podetia and repeated dichotomous/trichotomous-looking crown forks; no microscopic wall texture, chemistry, apothecia frequency, or calibrated color claim.',
  );
}

export function makeOxalisAcetosellaFixture(): EstonianLowCoverFixture {
  const tileSize = 0.32;
  const mesh: IndexedMesh = { positions: [], normals: [], indices: [] };
  const morphology = emptyMorphology('trifoliate-herb');
  const random = xorshift(0x08a1_15ac);
  const anchors = toroidalAnchors(18, tileSize, 0x51af_7083);
  const rhizome = [point(-0.018, 0.0012, tileSize * 0.46), point(tileSize * 0.34, 0.0015, tileSize * 0.49), point(tileSize * 0.68, 0.0011, tileSize * 0.44), point(tileSize + 0.020, 0.0014, tileSize * 0.47)];
  appendTube(mesh, rhizome, [0.0015, 0.0014, 0.0013, 0.0012], 5);
  morphology.rhizomeSegments += rhizome.length - 1;
  morphology.edgeCrossingAxes++;
  anchors.forEach((anchor, index) => {
    const height = 0.052 + random() * 0.063;
    const yaw = random() * TAU;
    const base = point(anchor.x, 0.001, anchor.z);
    const head = point(anchor.x + Math.cos(yaw) * height * 0.08, height, anchor.z + Math.sin(yaw) * height * 0.08);
    appendTube(mesh, [base, point((base.x + head.x) * 0.5, height * 0.48, (base.z + head.z) * 0.5), head], [0.00115, 0.00088, 0.00055], 6);
    morphology.primaryAxisCount++;
    morphology.shootCount++;
    for (let leaflet = 0; leaflet < 3; leaflet++) {
      const angle = yaw + (leaflet / 3) * TAU + (random() - 0.5) * 0.11;
      const joint = point(head.x + Math.cos(angle) * 0.006, head.y + 0.001, head.z + Math.sin(angle) * 0.006);
      appendTube(mesh, [head, joint], [0.00052, 0.00030], 4);
      appendHeartLeaflet(mesh, joint, angle, 0.015 + random() * 0.008, 0.020 + random() * 0.009, 0.03 + random() * 0.18);
      morphology.leafletCount++;
      morphology.leafCount++;
    }
    if (index % 6 === 0) {
      const flowerTop = point(base.x - Math.sin(yaw) * 0.010, height * 1.05, base.z + Math.cos(yaw) * 0.010);
      appendTube(mesh, [base, flowerTop], [0.00072, 0.00038], 5);
      appendTinyFlower(mesh, flowerTop, yaw, 0.0072, 5);
      morphology.flowerCount++;
    }
  });
  return finalizeFixture(
    8,
    'Oxalis acetosella',
    'original-procedural-oxalis-acetosella-trifoliate-rhizome-v1',
    tileSize,
    mesh,
    morphology,
    'Rhizomatous leaf-bearing form with long petioles, exactly three obcordate leaflets, and sparse five-part flowers; no pubescence, venation, nastic animation, or phenology claim.',
  );
}

export function makeMaianthemumBifoliumFixture(): EstonianLowCoverFixture {
  const tileSize = 0.48;
  const mesh: IndexedMesh = { positions: [], normals: [], indices: [] };
  const morphology = emptyMorphology('two-leaved-herb');
  const random = xorshift(0xb1f0_11a7);
  const anchors = toroidalAnchors(12, tileSize, 0x20f1_0ae5);
  const rhizome = [point(tileSize * 0.20, 0.001, -0.025), point(tileSize * 0.24, 0.0014, tileSize * 0.36), point(tileSize * 0.18, 0.0011, tileSize * 0.73), point(tileSize * 0.22, 0.0013, tileSize + 0.026)];
  appendTube(mesh, rhizome, [0.0013, 0.00125, 0.00115, 0.0010], 5);
  morphology.rhizomeSegments += rhizome.length - 1;
  morphology.edgeCrossingAxes++;
  anchors.forEach((anchor, index) => {
    const yaw = random() * TAU;
    const height = 0.105 + random() * 0.095;
    const base = point(anchor.x, 0.001, anchor.z);
    const top = point(anchor.x + Math.cos(yaw) * height * 0.025, height, anchor.z + Math.sin(yaw) * height * 0.025);
    appendTube(mesh, [base, point(anchor.x, height * 0.48, anchor.z), top], [0.0015, 0.00115, 0.00075], 6);
    morphology.primaryAxisCount++;
    morphology.shootCount++;
    const flowering = index % 3 !== 0;
    const leafCount = flowering ? 2 : 1;
    for (let leaf = 0; leaf < leafCount; leaf++) {
      const y = height * (0.53 + leaf * 0.18);
      const angle = yaw + leaf * (Math.PI * 0.84) + (random() - 0.5) * 0.15;
      const stemPoint = point(anchor.x + Math.cos(yaw) * y * 0.025, y, anchor.z + Math.sin(yaw) * y * 0.025);
      const petioleEnd = point(stemPoint.x + Math.cos(angle) * 0.012, y + 0.003, stemPoint.z + Math.sin(angle) * 0.012);
      appendTube(mesh, [stemPoint, petioleEnd], [0.00072, 0.00043], 4);
      appendCordateLeaf(mesh, petioleEnd, angle, 0.045 + random() * 0.027, 0.034 + random() * 0.020, -0.08 + random() * 0.22);
      morphology.leafCount++;
    }
    if (flowering) {
      morphology.twoLeafShootCount++;
      const racemeTop = point(top.x, top.y + 0.028 + random() * 0.015, top.z);
      appendTube(mesh, [top, racemeTop], [0.00065, 0.00035], 5);
      const flowers = 10 + (index % 6);
      for (let flower = 0; flower < flowers; flower++) {
        const t = (flower + 0.5) / flowers;
        const angle = yaw + flower * 2.399963;
        const stalkBase = point(top.x, top.y + (racemeTop.y - top.y) * t, top.z);
        const flowerCenter = point(stalkBase.x + Math.cos(angle) * 0.006, stalkBase.y, stalkBase.z + Math.sin(angle) * 0.006);
        appendTube(mesh, [stalkBase, flowerCenter], [0.00028, 0.00018], 3);
        appendTinyFlower(mesh, flowerCenter, angle, 0.0023, 4);
        morphology.flowerCount++;
      }
    }
  });
  return finalizeFixture(
    9,
    'Maianthemum bifolium',
    'original-procedural-maianthemum-bifolium-two-leaf-raceme-v1',
    tileSize,
    mesh,
    morphology,
    'Colony-forming rhizome, one-leaf sterile shoots, two cordate-leaf flowering shoots, and four-part terminal racemes; no vein/pubescence anatomy, berry phase, or phenology claim.',
  );
}

export function makeVacciniumMyrtillusFixture(): EstonianLowCoverFixture {
  const tileSize = 0.52;
  const mesh: IndexedMesh = { positions: [], normals: [], indices: [] };
  const morphology = emptyMorphology('dwarf-shrub');
  const random = xorshift(0x0acc_1a10);
  const anchors = toroidalAnchors(9, tileSize, 0x10ba_771e);
  anchors.forEach((anchor, shoot) => {
    const yaw = random() * TAU;
    const height = 0.18 + random() * 0.13;
    const base = point(anchor.x, 0.001, anchor.z);
    const trunk = [
      base,
      point(anchor.x + Math.cos(yaw) * 0.010, height * 0.34, anchor.z + Math.sin(yaw) * 0.010),
      point(anchor.x - Math.sin(yaw) * 0.012, height * 0.68, anchor.z + Math.cos(yaw) * 0.012),
      point(anchor.x + Math.cos(yaw) * 0.018, height, anchor.z + Math.sin(yaw) * 0.018),
    ];
    appendTube(mesh, trunk, [0.0025, 0.0020, 0.0015, 0.0008], 3);
    morphology.primaryAxisCount++;
    morphology.shootCount++;
    for (let branch = 0; branch < 4; branch++) {
      const stemT = 0.30 + branch * 0.16;
      const stem = point(
        anchor.x + Math.cos(yaw + branch) * 0.007,
        height * stemT,
        anchor.z + Math.sin(yaw + branch) * 0.007,
      );
      const branchYaw = yaw + (branch % 2 === 0 ? 1 : -1) * (0.72 + branch * 0.17);
      const branchLength = 0.055 + random() * 0.040;
      const tip = point(stem.x + Math.cos(branchYaw) * branchLength, stem.y + 0.035 + random() * 0.025, stem.z + Math.sin(branchYaw) * branchLength);
      appendTube(mesh, [
        stem,
        point(
          stem.x + (tip.x - stem.x) * 0.52,
          stem.y + (tip.y - stem.y) * 0.52,
          stem.z + (tip.z - stem.z) * 0.52,
        ),
        tip,
      ], [0.0013, 0.0009, 0.00045], 3);
      morphology.branchCount++;
      for (let leaf = 0; leaf < 7; leaf++) {
        const t = (leaf + 0.7) / 8;
        const leafBase = point(stem.x + (tip.x - stem.x) * t, stem.y + (tip.y - stem.y) * t, stem.z + (tip.z - stem.z) * t);
        const side = ((leaf + branch + shoot) & 1) === 0 ? 1 : -1;
        appendOvalLeaf(mesh, leafBase, branchYaw + side * 1.23, 0.020 + random() * 0.008, 0.010 + random() * 0.005, 0.12 + random() * 0.25, 5);
        morphology.leafCount++;
        morphology.serratedLeafCount++;
      }
    }
  });
  return finalizeFixture(
    10,
    'Vaccinium myrtillus',
    'original-procedural-vaccinium-myrtillus-angular-serrulate-shrub-v1',
    tileSize,
    mesh,
    morphology,
    'Vegetative rhizomatous dwarf-shrub architecture with explicitly three-angled green-shoot geometry and alternate serrulate ovate leaves; no flowers, berries, age calibration, or color claim.',
  );
}

export function makeCallunaVulgarisFixture(): EstonianLowCoverFixture {
  const tileSize = 0.38;
  const mesh: IndexedMesh = { positions: [], normals: [], indices: [] };
  const morphology = emptyMorphology('scale-leaved-shrub');
  const random = xorshift(0xca11_0a11);
  const anchors = toroidalAnchors(18, tileSize, 0x71ea_7e12);
  anchors.forEach((anchor, shoot) => {
    const yaw = random() * TAU;
    const height = 0.14 + random() * 0.12;
    const base = point(anchor.x, 0.001, anchor.z);
    const mainTop = point(anchor.x + Math.cos(yaw) * 0.025, height, anchor.z + Math.sin(yaw) * 0.025);
    appendTube(mesh, [base, point(anchor.x, height * 0.48, anchor.z), mainTop], [0.0022, 0.0015, 0.00065], 5);
    morphology.primaryAxisCount++;
    morphology.shootCount++;
    for (let branch = 0; branch < 8; branch++) {
      const level = 0.18 + branch * 0.092;
      const branchYaw = yaw + branch * 2.399963 + (random() - 0.5) * 0.22;
      const stem = point(anchor.x + Math.cos(yaw) * 0.025 * level, height * level, anchor.z + Math.sin(yaw) * 0.025 * level);
      const length = 0.045 + random() * 0.038;
      const tip = point(stem.x + Math.cos(branchYaw) * length * 0.55, stem.y + length, stem.z + Math.sin(branchYaw) * length * 0.55);
      appendTube(mesh, [stem, tip], [0.0010, 0.00038], 4);
      morphology.branchCount++;
      const nodes = 24;
      for (let node = 0; node < nodes; node++) {
        const t = (node + 0.5) / nodes;
        const leafBase = point(stem.x + (tip.x - stem.x) * t, stem.y + (tip.y - stem.y) * t, stem.z + (tip.z - stem.z) * t);
        const phase = ((node + shoot) & 1) * Math.PI * 0.25;
        for (let rank = 0; rank < 4; rank++) {
          appendScaleLeaf(mesh, leafBase, branchYaw + phase + rank * Math.PI * 0.5, 0.0031, 0.00065, 0.00072);
          morphology.leafCount++;
        }
        morphology.fourRankScaleNodes++;
      }
      for (let fork = 0; fork < 2; fork++) {
        const forkT = 0.46 + fork * 0.24;
        const forkBase = point(
          stem.x + (tip.x - stem.x) * forkT,
          stem.y + (tip.y - stem.y) * forkT,
          stem.z + (tip.z - stem.z) * forkT,
        );
        const forkYaw = branchYaw + (fork === 0 ? -1 : 1) * (0.62 + random() * 0.18);
        const forkLength = length * (0.30 + random() * 0.10);
        const forkTip = point(
          forkBase.x + Math.cos(forkYaw) * forkLength * 0.62,
          forkBase.y + forkLength * 0.56,
          forkBase.z + Math.sin(forkYaw) * forkLength * 0.62,
        );
        appendTube(mesh, [forkBase, forkTip], [0.00052, 0.00024], 4);
        morphology.branchCount++;
        morphology.forkCount++;
        const forkNodes = 8;
        for (let node = 0; node < forkNodes; node++) {
          const t = (node + 0.5) / forkNodes;
          const leafBase = point(
            forkBase.x + (forkTip.x - forkBase.x) * t,
            forkBase.y + (forkTip.y - forkBase.y) * t,
            forkBase.z + (forkTip.z - forkBase.z) * t,
          );
          const phase = ((node + fork + shoot) & 1) * Math.PI * 0.25;
          for (let rank = 0; rank < 4; rank++) {
            appendScaleLeaf(mesh, leafBase, forkYaw + phase + rank * Math.PI * 0.5, 0.0031, 0.00065, 0.00072);
            morphology.leafCount++;
          }
          morphology.fourRankScaleNodes++;
        }
      }
    }
  });
  return finalizeFixture(
    11,
    'Calluna vulgaris',
    'original-procedural-calluna-vulgaris-four-ranked-imbricate-shrub-v1',
    tileSize,
    mesh,
    morphology,
    'Vegetative richly branched dwarf-shrub form with explicit second-order branchlets and dense four-ranked imbricate scale leaves; no flowers, fruit, age calibration, environmental response, or color claim.',
  );
}

const FACTORIES: Record<EstonianLowCoverProfileId, () => EstonianLowCoverFixture> = {
  6: makePleuroziumSchreberiFixture,
  7: makeCladoniaRangiferinaFixture,
  8: makeOxalisAcetosellaFixture,
  9: makeMaianthemumBifoliumFixture,
  10: makeVacciniumMyrtillusFixture,
  11: makeCallunaVulgarisFixture,
};

export function makeEstonianLowCoverFixture(profileId: EstonianLowCoverProfileId): EstonianLowCoverFixture {
  return FACTORIES[profileId]();
}

export function makeAllEstonianLowCoverFixtures(): EstonianLowCoverFixture[] {
  return ESTONIAN_LOW_COVER_PROFILE_IDS.map((profileId) => makeEstonianLowCoverFixture(profileId));
}
