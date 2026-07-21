import { validateIndexedMesh, type IndexedMesh, type Vec3 } from './ProfileFormat';
import type { PeriodicTile } from './PeriodicProfile';

export const ESTONIAN_GRAMINOID_PROFILE_IDS = {
  AGROSTIS_CAPILLARIS: 0,
  AVENELLA_FLEXUOSA: 1,
  CALAMAGROSTIS_CANESCENS: 2,
  CAREX_CESPITOSA: 3,
  ERIOPHORUM_VAGINATUM: 4,
} as const;

export type EstonianGraminoidProfileId = typeof ESTONIAN_GRAMINOID_PROFILE_IDS[keyof typeof ESTONIAN_GRAMINOID_PROFILE_IDS];

export interface GraminoidStructureAudit {
  tufts: number;
  leaves: number;
  culms: number;
  rhizomes: number;
  panicleBranches: number;
  spikelets: number;
  maleSpikes: number;
  femaleSpikes: number;
  cottonHeads: number;
  cottonBristles: number;
  culmCrossSectionSides: number;
  growthForm: string;
  leafForm: string;
  inflorescenceForm: string;
}

export interface EstonianGraminoidFixture {
  profileId: EstonianGraminoidProfileId;
  species: string;
  family: 'Poaceae' | 'Cyperaceae';
  mesh: IndexedMesh;
  tile: PeriodicTile;
  tuftCenters: Array<{ x: number; z: number }>;
  structure: GraminoidStructureAudit;
  generator: string;
  sourceKeys: string[];
  claimBoundary: string;
}

interface TubeResult {
  rings: number[][];
  endCenter: number;
}

interface BuildContext {
  mesh: IndexedMesh;
  random: () => number;
  structure: GraminoidStructureAudit;
  palette: GraminoidPalette;
}

type Rgb = readonly [number, number, number];

interface GraminoidPalette {
  leafRoot: Rgb;
  leafTip: Rgb;
  culm: Rgb;
  rhizome: Rgb;
  panicleAxis: Rgb;
  spikelet: Rgb;
  callusHair: Rgb;
  anther: Rgb;
  cotton: Rgb;
}

const PALETTES: Record<EstonianGraminoidProfileId, GraminoidPalette> = {
  0: {
    leafRoot: [0.12, 0.29, 0.055], leafTip: [0.34, 0.52, 0.12],
    culm: [0.42, 0.55, 0.18], rhizome: [0.30, 0.25, 0.08],
    panicleAxis: [0.48, 0.28, 0.20], spikelet: [0.58, 0.32, 0.30],
    callusHair: [0.86, 0.78, 0.66], anther: [0.39, 0.12, 0.30], cotton: [0.9, 0.88, 0.75],
  },
  1: {
    leafRoot: [0.10, 0.22, 0.045], leafTip: [0.31, 0.43, 0.10],
    culm: [0.49, 0.50, 0.20], rhizome: [0.28, 0.22, 0.07],
    panicleAxis: [0.46, 0.29, 0.17], spikelet: [0.52, 0.25, 0.22],
    callusHair: [0.85, 0.78, 0.65], anther: [0.37, 0.11, 0.28], cotton: [0.9, 0.88, 0.75],
  },
  2: {
    leafRoot: [0.10, 0.27, 0.055], leafTip: [0.37, 0.51, 0.13],
    culm: [0.43, 0.53, 0.20], rhizome: [0.31, 0.24, 0.075],
    panicleAxis: [0.48, 0.31, 0.24], spikelet: [0.57, 0.35, 0.39],
    callusHair: [0.88, 0.80, 0.70], anther: [0.42, 0.10, 0.34], cotton: [0.9, 0.88, 0.75],
  },
  3: {
    leafRoot: [0.18, 0.36, 0.035], leafTip: [0.54, 0.65, 0.10],
    culm: [0.40, 0.51, 0.12], rhizome: [0.31, 0.20, 0.08],
    panicleAxis: [0.30, 0.20, 0.10], spikelet: [0.29, 0.14, 0.11],
    callusHair: [0.76, 0.69, 0.55], anther: [0.30, 0.08, 0.24], cotton: [0.9, 0.88, 0.75],
  },
  4: {
    leafRoot: [0.08, 0.24, 0.045], leafTip: [0.31, 0.45, 0.10],
    culm: [0.38, 0.49, 0.17], rhizome: [0.28, 0.20, 0.07],
    panicleAxis: [0.34, 0.22, 0.12], spikelet: [0.31, 0.20, 0.11],
    callusHair: [0.89, 0.86, 0.75], anther: [0.34, 0.09, 0.27], cotton: [0.94, 0.92, 0.82],
  },
};

const TAU = Math.PI * 2;

function normalized(value: Vec3): Vec3 {
  const length = Math.hypot(value.x, value.y, value.z);
  if (!(length > 1e-12)) return { x: 0, y: 1, z: 0 };
  return { x: value.x / length, y: value.y / length, z: value.z / length };
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

function add(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function scaled(value: Vec3, amount: number): Vec3 {
  return { x: value.x * amount, y: value.y * amount, z: value.z * amount };
}

function mix(a: number, b: number, t: number): number {
  return a + (b - a) * t;
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

function vertex(mesh: IndexedMesh, point: Vec3, normal: Vec3, color: Rgb): number {
  const index = mesh.positions.length / 3;
  mesh.positions.push(point.x, point.y, point.z);
  const n = normalized(normal);
  mesh.normals.push(n.x, n.y, n.z);
  mesh.colors?.push(...color);
  return index;
}

function appendHub(context: BuildContext, center: Vec3): number {
  return vertex(context.mesh, center, { x: 0, y: 1, z: 0 }, context.palette.rhizome);
}

function curvePoint(
  root: Vec3,
  angle: number,
  height: number,
  lean: number,
  curl: number,
  phase: number,
  t: number,
): Vec3 {
  const radial = lean * (t ** 1.35) + curl * Math.sin(Math.PI * t) * (0.35 + 0.65 * t);
  const sway = curl * 0.42 * Math.sin(phase + t * Math.PI * 1.6) * t;
  return {
    x: root.x + Math.cos(angle) * radial - Math.sin(angle) * sway,
    y: root.y + height * t,
    z: root.z + Math.sin(angle) * radial + Math.cos(angle) * sway,
  };
}

/** A segmented, tapered blade surface. Every blade shares its tuft hub index. */
function appendBlade(
  context: BuildContext,
  hub: number,
  root: Vec3,
  angle: number,
  height: number,
  width: number,
  lean: number,
  curl: number,
  phase: number,
  segments = 7,
): void {
  const { mesh } = context;
  const sides: Array<[number, number]> = [];
  for (let segment = 1; segment <= segments; segment++) {
    const t = segment / segments;
    const prior = curvePoint(root, angle, height, lean, curl, phase, Math.max(0, t - 1 / segments));
    const next = curvePoint(root, angle, height, lean, curl, phase, Math.min(1, t + 1 / segments));
    const tangent = normalized(sub(next, prior));
    let side = normalized(cross({ x: 0, y: 1, z: 0 }, tangent));
    if (Math.hypot(side.x, side.z) < 0.1) side = { x: Math.cos(angle + Math.PI / 2), y: 0, z: Math.sin(angle + Math.PI / 2) };
    const center = curvePoint(root, angle, height, lean, curl, phase, t);
    const halfWidth = width * 0.5 * ((1 - t) ** 0.62) + width * 0.035;
    const normal = normalized(cross(side, tangent));
    const color = context.palette.leafRoot.map((value, channel) =>
      mix(value, context.palette.leafTip[channel]!, t),
    ) as unknown as Rgb;
    sides.push([
      vertex(mesh, add(center, scaled(side, halfWidth)), normal, color),
      vertex(mesh, add(center, scaled(side, -halfWidth)), normal, color),
    ]);
  }
  const first = sides[0]!;
  mesh.indices.push(hub, first[0], first[1]);
  for (let segment = 0; segment < sides.length - 1; segment++) {
    const a = sides[segment]!;
    const b = sides[segment + 1]!;
    mesh.indices.push(a[0], b[0], a[1], a[1], b[0], b[1]);
  }
  context.structure.leaves++;
}

function frameForTangent(tangent: Vec3): { side: Vec3; binormal: Vec3 } {
  const helper = Math.abs(tangent.y) < 0.9 ? { x: 0, y: 1, z: 0 } : { x: 1, y: 0, z: 0 };
  const side = normalized(cross(helper, tangent));
  return { side, binormal: normalized(cross(tangent, side)) };
}

/** Indexed tube with a fan to an existing parent index, used for culms and axes. */
function appendTube(
  mesh: IndexedMesh,
  points: Vec3[],
  radii: number[],
  sides: number,
  anchor?: number,
  endAnchor?: number,
  color: Rgb = [0.3, 0.5, 0.12],
): TubeResult {
  if (points.length < 2 || radii.length !== points.length || sides < 3) throw new Error('invalid tube recipe');
  const rings: number[][] = [];
  for (let pointIndex = 0; pointIndex < points.length; pointIndex++) {
    const point = points[pointIndex]!;
    const prior = points[Math.max(0, pointIndex - 1)]!;
    const next = points[Math.min(points.length - 1, pointIndex + 1)]!;
    const tangent = normalized(sub(next, prior));
    const frame = frameForTangent(tangent);
    const ring: number[] = [];
    for (let sideIndex = 0; sideIndex < sides; sideIndex++) {
      const angle = (sideIndex / sides) * TAU;
      const radial = add(scaled(frame.side, Math.cos(angle)), scaled(frame.binormal, Math.sin(angle)));
      ring.push(vertex(mesh, add(point, scaled(radial, radii[pointIndex]!)), radial, color));
    }
    rings.push(ring);
  }
  if (anchor !== undefined) {
    const ring = rings[0]!;
    for (let side = 0; side < sides; side++) mesh.indices.push(anchor, ring[side]!, ring[(side + 1) % sides]!);
  }
  for (let ringIndex = 0; ringIndex < rings.length - 1; ringIndex++) {
    const a = rings[ringIndex]!;
    const b = rings[ringIndex + 1]!;
    for (let side = 0; side < sides; side++) {
      const next = (side + 1) % sides;
      mesh.indices.push(a[side]!, b[side]!, a[next]!, a[next]!, b[side]!, b[next]!);
    }
  }
  const endPoint = points[points.length - 1]!;
  const tangent = normalized(sub(endPoint, points[points.length - 2]!));
  // A supplied endpoint welds the tube's cap fan directly to existing indexed
  // geometry. This is real topological connectivity, not a duplicate-position
  // vertex joined by a zero-area triangle.
  const endCenter = endAnchor ?? vertex(mesh, endPoint, tangent, color);
  const last = rings[rings.length - 1]!;
  for (let side = 0; side < sides; side++) mesh.indices.push(last[side]!, endCenter, last[(side + 1) % sides]!);
  return { rings, endCenter };
}

function appendRhizome(context: BuildContext, startHub: number, endHub: number, start: Vec3, end: Vec3, phase: number): void {
  const perpendicular = normalized({ x: -(end.z - start.z), y: 0, z: end.x - start.x });
  const middle = add(scaled(add(start, end), 0.5), scaled(perpendicular, Math.sin(phase) * 0.018));
  middle.y = 0.004 + Math.cos(phase) * 0.001;
  appendTube(
    context.mesh,
    [{ ...start, y: 0.004 }, middle, { ...end, y: 0.004 }],
    [0.0022, 0.0018, 0.0022],
    5,
    startHub,
    endHub,
    context.palette.rhizome,
  );
  context.structure.rhizomes++;
}

function appendSpikelet(
  context: BuildContext,
  anchor: number,
  center: Vec3,
  directionInput: Vec3,
  length: number,
  width: number,
  sides = 4,
): number {
  const direction = normalized(directionInput);
  const start = add(center, scaled(direction, -length * 0.45));
  const middle = add(center, scaled(direction, length * 0.05));
  const end = add(center, scaled(direction, length * 0.55));
  const tube = appendTube(
    context.mesh,
    [start, middle, end],
    [width * 0.18, width, width * 0.06],
    sides,
    anchor,
    undefined,
    context.palette.spikelet,
  );
  context.structure.spikelets++;
  return tube.endCenter;
}

function rotatedFrame(directionInput: Vec3, roll: number): { side: Vec3; binormal: Vec3 } {
  const direction = normalized(directionInput);
  const frame = frameForTangent(direction);
  return {
    side: add(scaled(frame.side, Math.cos(roll)), scaled(frame.binormal, Math.sin(roll))),
    binormal: add(scaled(frame.binormal, Math.cos(roll)), scaled(frame.side, -Math.sin(roll))),
  };
}

/**
 * A pointed, gently keeled glume/lemma surface. These surfaces are deliberately
 * explicit rather than represented by a generic tube: the alternating pale
 * edges and purple-brown faces are what make a Calamagrostis panicle read as a
 * plume at source-mesh inspection scale.
 */
function appendLanceolateSurface(
  context: BuildContext,
  anchor: number,
  base: Vec3,
  directionInput: Vec3,
  length: number,
  halfWidth: number,
  roll: number,
  color: Rgb,
  keel = 0.00035,
): void {
  const direction = normalized(directionInput);
  const frame = rotatedFrame(direction, roll);
  const sections: Array<[number, number]> = [];
  const sectionCount = 6;
  for (let section = 0; section <= sectionCount; section++) {
    const t = section / sectionCount;
    const widthEnvelope = section === 0
      ? 0.10
      : section === sectionCount
        ? 0.018
        : Math.sin(Math.PI * t) ** 0.72;
    const center = add(
      add(base, scaled(direction, length * t)),
      scaled(frame.binormal, keel * Math.sin(Math.PI * t)),
    );
    const width = halfWidth * widthEnvelope;
    const normal = normalized(add(frame.binormal, scaled(frame.side, 0.12 * Math.cos(Math.PI * t))));
    sections.push([
      vertex(context.mesh, add(center, scaled(frame.side, width)), normal, color),
      vertex(context.mesh, add(center, scaled(frame.side, -width)), normal, color),
    ]);
  }
  context.mesh.indices.push(anchor, sections[0]![0], sections[0]![1]);
  for (let section = 0; section < sectionCount; section++) {
    const a = sections[section]!;
    const b = sections[section + 1]!;
    context.mesh.indices.push(a[0], b[0], a[1], a[1], b[0], b[1]);
  }
}

/** A tapered two-segment ribbon used for the millimetric callus hairs. */
function appendHairFilament(
  context: BuildContext,
  anchor: number,
  root: Vec3,
  directionInput: Vec3,
  length: number,
  width: number,
  color: Rgb,
  roll: number,
): void {
  const direction = normalized(directionInput);
  const frame = rotatedFrame(direction, roll);
  const middle = add(
    add(root, scaled(direction, length * 0.56)),
    scaled(frame.binormal, length * 0.075),
  );
  const tip = add(
    add(root, scaled(direction, length)),
    scaled(frame.binormal, length * 0.025),
  );
  const a = vertex(context.mesh, add(root, scaled(frame.side, width)), frame.binormal, color);
  const b = vertex(context.mesh, add(root, scaled(frame.side, -width)), frame.binormal, color);
  const c = vertex(context.mesh, add(middle, scaled(frame.side, width * 0.58)), frame.binormal, color);
  const d = vertex(context.mesh, add(middle, scaled(frame.side, -width * 0.58)), frame.binormal, color);
  const e = vertex(context.mesh, tip, frame.binormal, color);
  context.mesh.indices.push(anchor, a, b, a, c, b, b, c, d, c, e, d);
}

/**
 * Flowering Calamagrostis canescens spikelet: two 4.5--6 mm glumes enclosing a
 * shorter lemma, a halo of lemma-length callus hairs, and three small pendant
 * purple anthers. This is source geometry for the offline ray bake, not runtime
 * geometry.
 */
function appendCalamagrostisSpikelet(
  context: BuildContext,
  anchor: number,
  base: Vec3,
  directionInput: Vec3,
  length: number,
  roll: number,
): void {
  const direction = normalized(directionInput);
  const frame = rotatedFrame(direction, roll);
  const axisEnd = add(base, scaled(direction, length));
  const axis = appendTube(
    context.mesh,
    [base, add(base, scaled(direction, length * 0.52)), axisEnd],
    [0.00027, 0.00022, 0.00010],
    4,
    anchor,
    undefined,
    context.palette.panicleAxis,
  );

  // The paired glumes are offset about the compressed spikelet axis; a shorter
  // crossed lemma closes the centre and prevents a hollow papery silhouette.
  appendLanceolateSurface(
    context,
    anchor,
    add(base, scaled(frame.binormal, 0.00018)),
    direction,
    length,
    length * 0.205,
    roll,
    context.palette.spikelet,
    length * 0.055,
  );
  appendLanceolateSurface(
    context,
    anchor,
    add(base, scaled(frame.binormal, -0.00018)),
    normalized(add(direction, scaled(frame.side, 0.035))),
    length * mix(0.94, 1.02, context.random()),
    length * 0.195,
    roll + Math.PI,
    context.palette.spikelet,
    length * 0.050,
  );
  appendLanceolateSurface(
    context,
    anchor,
    add(base, scaled(direction, length * 0.09)),
    normalized(add(direction, scaled(frame.binormal, 0.045))),
    length * 0.68,
    length * 0.135,
    roll + Math.PI * 0.5,
    context.palette.callusHair,
    length * 0.032,
  );

  // Kew records the callus hairs as about 1.2x the lemma. Their collective
  // cream fringe is a major part of the fluffy, light-catching appearance.
  const hairCount = 18;
  for (let hair = 0; hair < hairCount; hair++) {
    const azimuth = roll + (hair / hairCount) * TAU + mix(-0.18, 0.18, context.random());
    const radial = add(
      scaled(frame.side, Math.cos(azimuth - roll)),
      scaled(frame.binormal, Math.sin(azimuth - roll)),
    );
    const root = add(
      add(base, scaled(direction, length * mix(0.05, 0.23, context.random()))),
      scaled(radial, length * mix(0.025, 0.075, context.random())),
    );
    const hairDirection = normalized(add(
      scaled(direction, mix(0.42, 0.72, context.random())),
      scaled(radial, mix(0.68, 1.08, context.random())),
    ));
    appendHairFilament(
      context,
      anchor,
      root,
      hairDirection,
      length * mix(0.57, 0.84, context.random()),
      mix(0.00012, 0.00022, context.random()),
      context.palette.callusHair,
      azimuth,
    );
  }

  for (let anther = 0; anther < 3; anther++) {
    const azimuth = roll + (anther / 3) * TAU + 0.35;
    const radial = add(
      scaled(frame.side, Math.cos(azimuth - roll)),
      scaled(frame.binormal, Math.sin(azimuth - roll)),
    );
    const filamentRoot = add(base, scaled(direction, length * mix(0.32, 0.48, context.random())));
    const antherBase = add(
      add(filamentRoot, scaled(radial, length * 0.20)),
      { x: 0, y: -length * 0.10, z: 0 },
    );
    appendTube(
      context.mesh,
      [filamentRoot, antherBase],
      [0.00010, 0.00008],
      3,
      axis.rings[1]![anther % axis.rings[1]!.length],
      undefined,
      context.palette.callusHair,
    );
    appendLanceolateSurface(
      context,
      axis.endCenter,
      antherBase,
      normalized(add(scaled(direction, 0.18), { x: 0, y: -1, z: 0 })),
      0.0015,
      0.00042,
      azimuth,
      context.palette.anther,
      0.00012,
    );
  }
  context.structure.spikelets++;
}

function appendOpenPanicle(
  context: BuildContext,
  anchor: number,
  base: Vec3,
  height: number,
  radius: number,
  levels: number,
  branchesPerLevel: number,
  phase: number,
  spikeletLength: number,
  spikeletWidth: number,
  branchRadius: number,
  nod: number,
): void {
  const rachisPoints = Array.from({ length: levels + 2 }, (_value, index) => {
    const t = index / (levels + 1);
    return {
      x: base.x + nod * t * t,
      y: base.y + height * t,
      z: base.z + nod * 0.35 * Math.sin(t * Math.PI),
    };
  });
  const rachis = appendTube(
    context.mesh,
    rachisPoints,
    rachisPoints.map((_point, index) => mix(branchRadius * 1.5, branchRadius * 0.75, index / (rachisPoints.length - 1))),
    5,
    anchor,
    undefined,
    context.palette.panicleAxis,
  );
  for (let level = 0; level < levels; level++) {
    const t = (level + 0.35) / levels;
    const origin = {
      x: base.x + nod * t * t,
      y: base.y + height * t,
      z: base.z + nod * 0.35 * Math.sin(t * Math.PI),
    };
    const levelRadius = radius * (1 - t * 0.66);
    for (let branch = 0; branch < branchesPerLevel; branch++) {
      const angle = phase + level * 1.29 + (branch / branchesPerLevel) * TAU;
      const branchLength = levelRadius * (0.78 + context.random() * 0.28);
      const rise = height * (0.10 + context.random() * 0.05);
      const end = {
        x: origin.x + Math.cos(angle) * branchLength,
        y: origin.y + rise,
        z: origin.z + Math.sin(angle) * branchLength,
      };
      const mid = {
        x: mix(origin.x, end.x, 0.54) - Math.sin(angle) * branchLength * 0.08,
        y: mix(origin.y, end.y, 0.54),
        z: mix(origin.z, end.z, 0.54) + Math.cos(angle) * branchLength * 0.08,
      };
      const nearestRachisRing = rachis.rings[Math.round(t * (rachis.rings.length - 1))]!;
      const branchTube = appendTube(
        context.mesh,
        [origin, mid, end],
        [branchRadius, branchRadius * 0.74, branchRadius * 0.42],
        4,
        nearestRachisRing[0],
        undefined,
        context.palette.panicleAxis,
      );
      context.structure.panicleBranches++;
      appendSpikelet(
        context,
        branchTube.endCenter,
        add(end, { x: 0, y: spikeletLength * 0.3, z: 0 }),
        normalized({ x: Math.cos(angle) * 0.25, y: 1, z: Math.sin(angle) * 0.25 }),
        spikeletLength,
        spikeletWidth,
      );
    }
  }
}

function pointOnPolyline(points: Vec3[], tInput: number): { point: Vec3; segment: number; tangent: Vec3 } {
  const t = Math.max(0, Math.min(0.999999, tInput));
  const scaledT = t * (points.length - 1);
  const segment = Math.min(points.length - 2, Math.floor(scaledT));
  const localT = scaledT - segment;
  const a = points[segment]!;
  const b = points[segment + 1]!;
  return {
    point: { x: mix(a.x, b.x, localT), y: mix(a.y, b.y, localT), z: mix(a.z, b.z, localT) },
    segment,
    tangent: normalized(sub(b, a)),
  };
}

/**
 * Recursively expands one compound-panicle branch. Depth two makes primary,
 * secondary, then tertiary axes; only the outermost axes carry the crowded
 * pedicelled spikelets. The authored recursion is intentionally high-detail and
 * is completely absent from the runtime shader.
 */
function appendCalamagrostisBranch(
  context: BuildContext,
  anchor: number,
  origin: Vec3,
  directionInput: Vec3,
  length: number,
  radius: number,
  depth: number,
  phase: number,
): void {
  const direction = normalized(directionInput);
  const frame = rotatedFrame(direction, phase);
  const bend = length * mix(-0.13, 0.13, context.random());
  const droop = length * mix(-0.035, 0.045, context.random());
  const points = [
    origin,
    add(add(origin, scaled(direction, length * 0.30)), scaled(frame.side, bend * 0.72)),
    add(add(add(origin, scaled(direction, length * 0.66)), scaled(frame.side, bend)), { x: 0, y: droop, z: 0 }),
    add(add(add(origin, scaled(direction, length)), scaled(frame.side, bend * 0.44)), { x: 0, y: droop * 1.65, z: 0 }),
  ];
  const branch = appendTube(
    context.mesh,
    points,
    [radius, radius * 0.76, radius * 0.50, radius * 0.24],
    4,
    anchor,
    undefined,
    context.palette.panicleAxis,
  );
  context.structure.panicleBranches++;

  if (depth > 0) {
    const childCount = depth === 2
      ? 4 + Math.floor(context.random() * 2)
      : 3 + Math.floor(context.random() * 2);
    for (let child = 0; child < childCount; child++) {
      const t = mix(0.22, 0.91, (child + mix(0.18, 0.82, context.random())) / childCount);
      const sampled = pointOnPolyline(points, t);
      const childPhase = phase + child * 2.399963229728653 + mix(-0.55, 0.55, context.random());
      const childFrame = rotatedFrame(sampled.tangent, childPhase);
      const outward = add(
        scaled(childFrame.side, mix(0.43, 0.67, context.random())),
        scaled(childFrame.binormal, mix(-0.27, 0.27, context.random())),
      );
      const childDirection = normalized(add(
        add(scaled(sampled.tangent, depth === 2 ? 0.72 : 0.62), outward),
        { x: 0, y: mix(0.42, 0.72, context.random()), z: 0 },
      ));
      const ring = branch.rings[Math.min(branch.rings.length - 1, sampled.segment + 1)]!;
      appendCalamagrostisBranch(
        context,
        ring[child % ring.length]!,
        sampled.point,
        childDirection,
        length * (depth === 2
          ? mix(0.31, 0.45, context.random())
          : mix(0.34, 0.50, context.random())),
        radius * 0.66,
        depth - 1,
        childPhase,
      );
    }
    return;
  }

  const spikeletCount = 4 + Math.floor(context.random() * 3);
  for (let spikelet = 0; spikelet < spikeletCount; spikelet++) {
    const t = mix(0.17, 0.98, (spikelet + mix(0.10, 0.90, context.random())) / spikeletCount);
    const sampled = pointOnPolyline(points, t);
    const roll = phase + spikelet * 2.399963229728653 + mix(-0.32, 0.32, context.random());
    const spikeletFrame = rotatedFrame(sampled.tangent, roll);
    const sideSign = (spikelet & 1) === 0 ? 1 : -1;
    const pedicelDirection = normalized(add(
      add(scaled(sampled.tangent, 0.34), scaled(spikeletFrame.side, sideSign * 0.84)),
      { x: 0, y: mix(0.18, 0.44, context.random()), z: 0 },
    ));
    const pedicelLength = mix(0.0007, 0.0030, context.random());
    const pedicelEnd = add(sampled.point, scaled(pedicelDirection, pedicelLength));
    const ring = branch.rings[Math.min(branch.rings.length - 1, sampled.segment + 1)]!;
    const pedicel = appendTube(
      context.mesh,
      [sampled.point, pedicelEnd],
      [0.00030, 0.00016],
      4,
      ring[spikelet % ring.length],
      undefined,
      context.palette.panicleAxis,
    );
    appendCalamagrostisSpikelet(
      context,
      pedicel.endCenter,
      pedicelEnd,
      normalized(add(
        scaled(sampled.tangent, mix(0.30, 0.52, context.random())),
        add(scaled(pedicelDirection, 0.74), { x: 0, y: mix(0.20, 0.48, context.random()), z: 0 }),
      )),
      mix(0.0045, 0.0060, context.random()),
      roll,
    );
  }
}

/**
 * Calamagrostis-specific open lanceolate panicle. Irregular golden-angle branch
 * insertion avoids artificial whorls, while recursive compound branches form a
 * dense, overlapping three-dimensional plume instead of horizontal ladders.
 */
function appendCalamagrostisPanicle(
  context: BuildContext,
  anchor: number,
  base: Vec3,
  height: number,
  radius: number,
  phase: number,
  nod: number,
): void {
  const rachisSections = 19;
  const rachisPoints = Array.from({ length: rachisSections + 1 }, (_value, index) => {
    const t = index / rachisSections;
    return {
      x: base.x + nod * t * t + Math.sin(phase + t * Math.PI * 1.4) * 0.0025 * t,
      y: base.y + height * t,
      z: base.z + nod * 0.42 * Math.sin(t * Math.PI) + Math.cos(phase + t * Math.PI * 1.1) * 0.0020 * t,
    };
  });
  const rachis = appendTube(
    context.mesh,
    rachisPoints,
    rachisPoints.map((_point, index) => mix(0.00130, 0.00034, index / rachisSections)),
    5,
    anchor,
    undefined,
    context.palette.panicleAxis,
  );

  const primaryCount = 22;
  for (let primary = 0; primary < primaryCount; primary++) {
    const t = mix(0.045, 0.965, (primary + mix(0.12, 0.88, context.random())) / primaryCount);
    const sampled = pointOnPolyline(rachisPoints, t);
    const angle = phase + primary * 2.399963229728653 + mix(-0.52, 0.52, context.random());
    const radial = { x: Math.cos(angle), y: 0, z: Math.sin(angle) };
    const middleFullness = Math.sin(Math.PI * Math.min(0.995, t * 0.94 + 0.04)) ** 0.68;
    const branchLength = radius
      * mix(0.78, 1.12, context.random())
      * (0.28 + middleFullness * 0.88)
      * mix(1.0, 0.64, t);
    const direction = normalized(add(
      scaled(radial, mix(0.55, 0.74, context.random())),
      { x: 0, y: mix(0.70, 1.08, context.random()) + t * 0.30, z: 0 },
    ));
    const ringIndex = Math.min(rachis.rings.length - 1, Math.max(0, Math.round(t * (rachis.rings.length - 1))));
    const ring = rachis.rings[ringIndex]!;
    appendCalamagrostisBranch(
      context,
      ring[primary % ring.length]!,
      sampled.point,
      direction,
      branchLength,
      mix(0.00062, 0.00082, 1 - t),
      t > 0.84 ? 1 : 2,
      angle,
    );
  }
}

function newStructure(overrides: Partial<GraminoidStructureAudit>): GraminoidStructureAudit {
  return {
    tufts: 0,
    leaves: 0,
    culms: 0,
    rhizomes: 0,
    panicleBranches: 0,
    spikelets: 0,
    maleSpikes: 0,
    femaleSpikes: 0,
    cottonHeads: 0,
    cottonBristles: 0,
    culmCrossSectionSides: 0,
    growthForm: '',
    leafForm: '',
    inflorescenceForm: '',
    ...overrides,
  };
}

function makeContext(
  seed: number,
  structure: GraminoidStructureAudit,
  profileId: EstonianGraminoidProfileId,
): BuildContext {
  return {
    mesh: { positions: [], normals: [], colors: [], indices: [] },
    random: xorshift(seed),
    structure,
    palette: PALETTES[profileId],
  };
}

function maxY(mesh: IndexedMesh): number {
  let value = Number.NEGATIVE_INFINITY;
  for (let index = 1; index < mesh.positions.length; index += 3) value = Math.max(value, mesh.positions[index] as number);
  return value;
}

function finalize(
  fixture: Omit<EstonianGraminoidFixture, 'tile'> & { tileSize: number },
): EstonianGraminoidFixture {
  validateIndexedMesh(fixture.mesh);
  const { tileSize, ...rest } = fixture;
  return {
    ...rest,
    tile: { originX: 0, originZ: 0, sizeX: tileSize, sizeZ: tileSize, topH: maxY(fixture.mesh) + 0.025 },
  };
}

function makeAgrostisCapillaris(): EstonianGraminoidFixture {
  const tileSize = 0.40;
  const tuftCenters = [
    { x: 0.038, z: 0.071 }, { x: 0.139, z: 0.038 }, { x: 0.271, z: 0.082 }, { x: 0.367, z: 0.151 },
    { x: 0.064, z: 0.259 }, { x: 0.187, z: 0.204 }, { x: 0.309, z: 0.293 }, { x: 0.151, z: 0.365 },
  ];
  const structure = newStructure({
    tufts: tuftCenters.length,
    culmCrossSectionSides: 6,
    growthForm: 'caespitose turf joined by elongated near-surface rhizomes and occasional stolon-like axes',
    leafForm: 'flat narrow ribbed blades with acuminate tips',
    inflorescenceForm: 'open oblong-ovate panicle with capillary whorled primary branches and solitary small spikelets',
  });
  const context = makeContext(0xa610_57c1, structure, ESTONIAN_GRAMINOID_PROFILE_IDS.AGROSTIS_CAPILLARIS);
  const hubs = tuftCenters.map((center) => appendHub(context, { ...center, y: 0.004 }));
  tuftCenters.forEach((center, tuft) => {
    const hub = hubs[tuft]!;
    for (let leaf = 0; leaf < 12; leaf++) {
      const angle = (leaf / 12) * TAU + context.random() * 0.31;
      appendBlade(context, hub, { ...center, y: 0.004 }, angle, mix(0.070, 0.145, context.random()), mix(0.0018, 0.0044, context.random()), mix(0.018, 0.061, context.random()), mix(0.004, 0.018, context.random()), context.random() * TAU, 6);
    }
    const culmHeight = mix(0.245, 0.335, context.random());
    const leanAngle = context.random() * TAU;
    const culm = appendTube(context.mesh, [
      { ...center, y: 0.005 },
      { x: center.x + Math.cos(leanAngle) * 0.006, y: culmHeight * 0.50, z: center.z + Math.sin(leanAngle) * 0.006 },
      { x: center.x + Math.cos(leanAngle) * 0.013, y: culmHeight, z: center.z + Math.sin(leanAngle) * 0.013 },
    ], [0.00155, 0.00130, 0.00102], 6, hub, undefined, context.palette.culm);
    structure.culms++;
    appendOpenPanicle(context, culm.endCenter, {
      x: center.x + Math.cos(leanAngle) * 0.013,
      y: culmHeight - 0.004,
      z: center.z + Math.sin(leanAngle) * 0.013,
    }, 0.075, 0.047, 4, 3, context.random() * TAU, 0.0030, 0.0012, 0.00062, 0.006 * Math.cos(leanAngle));
  });
  for (let index = 0; index < tuftCenters.length; index++) {
    const next = (index + 1) % tuftCenters.length;
    appendRhizome(context, hubs[index]!, hubs[next]!, { ...tuftCenters[index]!, y: 0.004 }, { ...tuftCenters[next]!, y: 0.004 }, index * 1.71);
  }
  return finalize({
    profileId: ESTONIAN_GRAMINOID_PROFILE_IDS.AGROSTIS_CAPILLARIS,
    species: 'Agrostis capillaris',
    family: 'Poaceae',
    mesh: context.mesh,
    tileSize,
    tuftCenters,
    structure,
    generator: 'original-procedural-agrostis-capillaris-v2-welded-rhizomes',
    sourceKeys: ['powo-agrostis-capillaris', 'eelurikkus-estonian-species-portal'],
    claimBoundary: 'Architectural flowering turf profile; not calibrated to one population, phenological date, material color, abundance, or wind state.',
  });
}

function makeAvenellaFlexuosa(): EstonianGraminoidFixture {
  const tileSize = 0.42;
  const tuftCenters = [
    { x: 0.049, z: 0.084 }, { x: 0.178, z: 0.047 }, { x: 0.337, z: 0.109 },
    { x: 0.105, z: 0.275 }, { x: 0.254, z: 0.239 }, { x: 0.371, z: 0.349 },
  ];
  const structure = newStructure({
    tufts: tuftCenters.length,
    culmCrossSectionSides: 5,
    growthForm: 'discrete dense caespitose tufts without a forced creeping-rhizome network',
    leafForm: 'mostly basal stiff filiform involute blades with pronounced flexuous paths',
    inflorescenceForm: 'very open ovate-effuse panicle with capillary branches and larger two-floret-scale spikelet bodies',
  });
  const context = makeContext(0xa7e1_f10a, structure, ESTONIAN_GRAMINOID_PROFILE_IDS.AVENELLA_FLEXUOSA);
  tuftCenters.forEach((center) => {
    const hub = appendHub(context, { ...center, y: 0.004 });
    for (let leaf = 0; leaf < 22; leaf++) {
      const angle = (leaf / 22) * TAU + context.random() * 0.42;
      appendBlade(context, hub, { ...center, y: 0.004 }, angle, mix(0.085, 0.190, context.random()), mix(0.0009, 0.00145, context.random()), mix(0.026, 0.082, context.random()), mix(0.014, 0.036, context.random()), context.random() * TAU, 8);
    }
    const culmCount = context.random() > 0.58 ? 2 : 1;
    for (let culmIndex = 0; culmIndex < culmCount; culmIndex++) {
      const culmHeight = mix(0.285, 0.390, context.random());
      const leanAngle = context.random() * TAU;
      const culm = appendTube(context.mesh, [
        { ...center, y: 0.005 },
        { x: center.x + Math.cos(leanAngle) * 0.008, y: culmHeight * 0.48, z: center.z + Math.sin(leanAngle) * 0.008 },
        { x: center.x + Math.cos(leanAngle) * 0.020, y: culmHeight, z: center.z + Math.sin(leanAngle) * 0.020 },
      ], [0.0011, 0.0009, 0.00068], 5, hub, undefined, context.palette.culm);
      structure.culms++;
      appendOpenPanicle(context, culm.endCenter, {
        x: center.x + Math.cos(leanAngle) * 0.020,
        y: culmHeight - 0.005,
        z: center.z + Math.sin(leanAngle) * 0.020,
      }, 0.092, 0.071, 4, 4, context.random() * TAU, 0.0052, 0.00175, 0.00046, 0.013 * Math.cos(leanAngle));
    }
  });
  return finalize({
    profileId: ESTONIAN_GRAMINOID_PROFILE_IDS.AVENELLA_FLEXUOSA,
    species: 'Avenella flexuosa',
    family: 'Poaceae',
    mesh: context.mesh,
    tileSize,
    tuftCenters,
    structure,
    generator: 'original-procedural-avenella-flexuosa-v1',
    sourceKeys: ['powo-avenella-flexuosa', 'eelurikkus-avenella-synonym-record'],
    claimBoundary: 'Architectural flowering tuft profile; filiform blades are widened only enough to survive the 64-texel geometric carrier and are not a micrometric specimen reconstruction.',
  });
}

function makeCalamagrostisCanescens(): EstonianGraminoidFixture {
  const tileSize = 0.52;
  const tuftCenters = [
    { x: 0.052, z: 0.092 }, { x: 0.226, z: 0.055 }, { x: 0.431, z: 0.124 },
    { x: 0.133, z: 0.318 }, { x: 0.335, z: 0.291 }, { x: 0.469, z: 0.438 },
  ];
  const structure = newStructure({
    tufts: tuftCenters.length,
    culmCrossSectionSides: 7,
    growthForm: 'loosely clumped perennial shoots joined by short rhizomes',
    leafForm: 'broader flat-to-convolute attenuate blades borne basally and along erect culms',
    inflorescenceForm: 'open lanceolate-oblong, gently nodding panicle with many short pedicelled spikelets',
  });
  const context = makeContext(0xca1a_6c35, structure, ESTONIAN_GRAMINOID_PROFILE_IDS.CALAMAGROSTIS_CANESCENS);
  const hubs = tuftCenters.map((center) => appendHub(context, { ...center, y: 0.004 }));
  tuftCenters.forEach((center, tuft) => {
    const hub = hubs[tuft]!;
    for (let leaf = 0; leaf < 15; leaf++) {
      const angle = (leaf / 15) * TAU + context.random() * 0.33;
      appendBlade(context, hub, { ...center, y: 0.004 }, angle, mix(0.260, 0.550, context.random()), mix(0.0030, 0.0060, context.random()), mix(0.055, 0.180, context.random()), mix(0.012, 0.048, context.random()), context.random() * TAU, 10);
    }
    const culmHeight = mix(0.680, 0.940, context.random());
    const leanAngle = context.random() * TAU;
    const culmPoints = [
      { ...center, y: 0.005 },
      { x: center.x + Math.cos(leanAngle) * 0.008, y: culmHeight * 0.42, z: center.z + Math.sin(leanAngle) * 0.008 },
      { x: center.x + Math.cos(leanAngle) * 0.017, y: culmHeight * 0.78, z: center.z + Math.sin(leanAngle) * 0.017 },
      { x: center.x + Math.cos(leanAngle) * 0.032, y: culmHeight, z: center.z + Math.sin(leanAngle) * 0.032 },
    ];
    const culm = appendTube(
      context.mesh,
      culmPoints,
      [0.0023, 0.0020, 0.00165, 0.00125],
      7,
      hub,
      undefined,
      context.palette.culm,
    );
    structure.culms++;
    // Two explicit cauline leaves: Calamagrostis foliage is not exclusively a
    // basal fountain. Their culm-ring anchors keep the authored mesh connected.
    for (const node of [1, 2]) {
      const nodePoint = culmPoints[node]!;
      const leafAngle = leanAngle + (node === 1 ? 1.85 : -1.35) + context.random() * 0.35;
      appendBlade(
        context,
        culm.rings[node]![0]!,
        nodePoint,
        leafAngle,
        node === 1 ? mix(0.32, 0.48, context.random()) : mix(0.22, 0.36, context.random()),
        mix(0.0035, 0.0060, context.random()),
        mix(0.12, 0.24, context.random()),
        mix(0.025, 0.065, context.random()),
        context.random() * TAU,
        9,
      );
    }
    appendCalamagrostisPanicle(context, culm.endCenter, {
      x: center.x + Math.cos(leanAngle) * 0.032,
      y: culmHeight - 0.006,
      z: center.z + Math.sin(leanAngle) * 0.032,
    }, mix(0.180, 0.230, context.random()), mix(0.045, 0.055, context.random()), context.random() * TAU, 0.040 * Math.cos(leanAngle));
  });
  for (let index = 0; index < tuftCenters.length - 1; index++) {
    appendRhizome(context, hubs[index]!, hubs[index + 1]!, { ...tuftCenters[index]!, y: 0.004 }, { ...tuftCenters[index + 1]!, y: 0.004 }, index * 1.43 + 0.4);
  }
  return finalize({
    profileId: ESTONIAN_GRAMINOID_PROFILE_IDS.CALAMAGROSTIS_CANESCENS,
    species: 'Calamagrostis canescens',
    family: 'Poaceae',
    mesh: context.mesh,
    tileSize,
    tuftCenters,
    structure,
    generator: 'original-procedural-calamagrostis-canescens-v5-ascending-recursive-hairy-panicle',
    sourceKeys: ['powo-calamagrostis-canescens', 'eelurikkus-estonian-species-portal'],
    claimBoundary: 'High-detail flowering architectural stand with explicit glumes, lemmas, callus hairs, and anthers; geometry is species-grounded but not a scan of one local genotype or phenological specimen.',
  });
}

function appendCarexSpike(
  context: BuildContext,
  anchor: number,
  base: Vec3,
  length: number,
  radius: number,
  female: boolean,
  phase: number,
): void {
  const axis = appendTube(
    context.mesh,
    [base, { ...base, y: base.y + length }],
    [radius * 0.28, radius * 0.22],
    5,
    anchor,
    undefined,
    context.palette.panicleAxis,
  );
  const scaleCount = female ? 9 : 11;
  for (let scale = 0; scale < scaleCount; scale++) {
    const t = (scale + 0.5) / scaleCount;
    const angle = phase + scale * 2.399963229728653;
    const center = {
      x: base.x + Math.cos(angle) * radius * 0.54,
      y: base.y + length * t,
      z: base.z + Math.sin(angle) * radius * 0.54,
    };
    appendSpikelet(context, axis.endCenter, center, normalized({ x: Math.cos(angle) * 0.32, y: 1, z: Math.sin(angle) * 0.32 }), female ? 0.0048 : 0.0039, female ? radius * 0.58 : radius * 0.40, 4);
  }
  if (female) context.structure.femaleSpikes++;
  else context.structure.maleSpikes++;
}

function makeCarexCespitosa(): EstonianGraminoidFixture {
  const tileSize = 0.48;
  const tuftCenters = [
    { x: 0.070, z: 0.092 }, { x: 0.325, z: 0.078 }, { x: 0.163, z: 0.333 }, { x: 0.413, z: 0.371 },
  ];
  const structure = newStructure({
    tufts: tuftCenters.length,
    culmCrossSectionSides: 3,
    growthForm: 'separate very dense tussocks with no creeping-rhizome connection',
    leafForm: 'bright narrow arched basal sedge blades emerging through broader basal sheath geometry',
    inflorescenceForm: 'one terminal male spike above two short stout lateral female spikes on triangular culms',
  });
  const context = makeContext(0xca2e_ce51, structure, ESTONIAN_GRAMINOID_PROFILE_IDS.CAREX_CESPITOSA);
  tuftCenters.forEach((center) => {
    const hub = appendHub(context, { ...center, y: 0.005 });
    appendTube(context.mesh, [{ ...center, y: 0.004 }, { ...center, y: 0.055 }], [0.017, 0.010], 10, hub, undefined, context.palette.rhizome);
    for (let leaf = 0; leaf < 30; leaf++) {
      const angle = (leaf / 30) * TAU + context.random() * 0.23;
      appendBlade(context, hub, { ...center, y: 0.006 }, angle, mix(0.155, 0.315, context.random()), mix(0.0026, 0.0048, context.random()), mix(0.035, 0.120, context.random()), mix(0.009, 0.030, context.random()), context.random() * TAU, 8);
    }
    for (let culmIndex = 0; culmIndex < 2; culmIndex++) {
      const angle = context.random() * TAU;
      const height = mix(0.305, 0.410, context.random());
      const tip = { x: center.x + Math.cos(angle) * 0.017, y: height, z: center.z + Math.sin(angle) * 0.017 };
      const culm = appendTube(context.mesh, [{ ...center, y: 0.007 }, { x: mix(center.x, tip.x, 0.55), y: height * 0.55, z: mix(center.z, tip.z, 0.55) }, tip], [0.0025, 0.0021, 0.0018], 3, hub, undefined, context.palette.culm);
      structure.culms++;
      appendCarexSpike(context, culm.endCenter, { ...tip, y: tip.y - 0.002 }, 0.028, 0.0042, false, context.random() * TAU);
      for (let lateral = 0; lateral < 2; lateral++) {
        const sideAngle = angle + (lateral === 0 ? 1.45 : -1.15);
        const origin = { ...tip, y: tip.y - 0.026 - lateral * 0.023 };
        const stalkEnd = {
          x: origin.x + Math.cos(sideAngle) * (0.010 + lateral * 0.004),
          y: origin.y - 0.004,
          z: origin.z + Math.sin(sideAngle) * (0.010 + lateral * 0.004),
        };
        const stalk = appendTube(context.mesh, [origin, stalkEnd], [0.0009, 0.00065], 4, culm.endCenter, undefined, context.palette.panicleAxis);
        appendCarexSpike(context, stalk.endCenter, stalkEnd, 0.022 + lateral * 0.004, 0.0062, true, context.random() * TAU);
      }
    }
  });
  return finalize({
    profileId: ESTONIAN_GRAMINOID_PROFILE_IDS.CAREX_CESPITOSA,
    species: 'Carex cespitosa',
    family: 'Cyperaceae',
    mesh: context.mesh,
    tileSize,
    tuftCenters,
    structure,
    generator: 'original-procedural-carex-cespitosa-v1',
    sourceKeys: ['powo-carex-cespitosa', 'james-2012-carex-cespitosa', 'eelurikkus-carex-cespitosa-occurrence'],
    claimBoundary: 'Architectural flowering tussock profile; diagnostic sheath color, utricle venation, stomatal surface, and wetland abundance are not encoded in the geometric carrier.',
  });
}

function appendCottonHead(context: BuildContext, anchor: number, base: Vec3, phase: number): void {
  const spike = appendTube(
    context.mesh,
    [base, { ...base, y: base.y + 0.023 }],
    [0.0052, 0.0038],
    8,
    anchor,
    undefined,
    context.palette.spikelet,
  );
  const bristles = 42;
  for (let bristle = 0; bristle < bristles; bristle++) {
    const u = (bristle + 0.5) / bristles;
    const angle = phase + bristle * 2.399963229728653;
    const axial = Math.cos(Math.PI * u);
    const radial = Math.sqrt(Math.max(0, 1 - axial * axial));
    const root = {
      x: base.x + Math.cos(angle) * radial * 0.0035,
      y: base.y + 0.0115 + axial * 0.009,
      z: base.z + Math.sin(angle) * radial * 0.0035,
    };
    const length = mix(0.014, 0.025, context.random());
    const outward = normalized({ x: Math.cos(angle) * (0.65 + radial), y: 0.25 + axial * 0.22, z: Math.sin(angle) * (0.65 + radial) });
    const middle = add(root, scaled(outward, length * 0.56));
    const end = add(root, scaled(outward, length));
    const side = normalized(cross(outward, { x: 0, y: 1, z: 0 }));
    const width = 0.00048;
    const a = vertex(context.mesh, add(root, scaled(side, width)), outward, context.palette.cotton);
    const b = vertex(context.mesh, add(root, scaled(side, -width)), outward, context.palette.cotton);
    const c = vertex(context.mesh, add(middle, scaled(side, width * 0.68)), outward, context.palette.cotton);
    const d = vertex(context.mesh, add(middle, scaled(side, -width * 0.68)), outward, context.palette.cotton);
    const tip = vertex(context.mesh, end, outward, context.palette.cotton);
    context.mesh.indices.push(spike.endCenter, a, b, a, c, b, b, c, d, c, tip, d);
    context.structure.cottonBristles++;
  }
  context.structure.cottonHeads++;
}

function makeEriophorumVaginatum(): EstonianGraminoidFixture {
  const tileSize = 0.46;
  const tuftCenters = [
    { x: 0.075, z: 0.086 }, { x: 0.324, z: 0.104 }, { x: 0.161, z: 0.340 }, { x: 0.398, z: 0.362 },
  ];
  const structure = newStructure({
    tufts: tuftCenters.length,
    culmCrossSectionSides: 3,
    growthForm: 'separate compact tussocks built from persistent basal sheaths and dense leaves',
    leafForm: 'mostly basal, short filiform leaves; flowering culms conspicuously overtop the foliage',
    inflorescenceForm: 'one erect terminal spike per culm, expanded in fruit into many explicit silky bristle ribbons',
  });
  const context = makeContext(0xe710_f4a9, structure, ESTONIAN_GRAMINOID_PROFILE_IDS.ERIOPHORUM_VAGINATUM);
  tuftCenters.forEach((center) => {
    const hub = appendHub(context, { ...center, y: 0.005 });
    appendTube(context.mesh, [{ ...center, y: 0.004 }, { ...center, y: 0.062 }], [0.020, 0.011], 11, hub, undefined, context.palette.rhizome);
    for (let leaf = 0; leaf < 34; leaf++) {
      const angle = (leaf / 34) * TAU + context.random() * 0.22;
      appendBlade(context, hub, { ...center, y: 0.006 }, angle, mix(0.105, 0.235, context.random()), mix(0.00115, 0.00175, context.random()), mix(0.020, 0.068, context.random()), mix(0.006, 0.019, context.random()), context.random() * TAU, 7);
    }
    for (let culmIndex = 0; culmIndex < 2; culmIndex++) {
      const angle = context.random() * TAU;
      const height = mix(0.305, 0.405, context.random());
      const tip = { x: center.x + Math.cos(angle) * 0.012, y: height, z: center.z + Math.sin(angle) * 0.012 };
      const culm = appendTube(context.mesh, [{ ...center, y: 0.008 }, { x: mix(center.x, tip.x, 0.52), y: height * 0.55, z: mix(center.z, tip.z, 0.52) }, tip], [0.0023, 0.0019, 0.00155], 3, hub, undefined, context.palette.culm);
      structure.culms++;
      // The uppermost bladeless sheath is deliberately inflated around the culm.
      const sheathBase = { x: mix(center.x, tip.x, 0.57), y: height * 0.54, z: mix(center.z, tip.z, 0.57) };
      appendTube(context.mesh, [sheathBase, { x: mix(center.x, tip.x, 0.72), y: height * 0.69, z: mix(center.z, tip.z, 0.72) }], [0.0042, 0.0031], 7, culm.rings[1]![0], undefined, context.palette.culm);
      appendCottonHead(context, culm.endCenter, { ...tip, y: tip.y - 0.001 }, context.random() * TAU);
    }
  });
  return finalize({
    profileId: ESTONIAN_GRAMINOID_PROFILE_IDS.ERIOPHORUM_VAGINATUM,
    species: 'Eriophorum vaginatum',
    family: 'Cyperaceae',
    mesh: context.mesh,
    tileSize,
    tuftCenters,
    structure,
    generator: 'original-procedural-eriophorum-vaginatum-fruiting-v1',
    sourceKeys: ['powo-eriophorum-vaginatum', 'canadian-arctic-flora-eriophorum-vaginatum', 'eelurikkus-eriophorum-vaginatum'],
    claimBoundary: 'Fruiting architectural tussock profile with one head per culm; bristle color, seasonal abundance, peat chemistry, and exact local subspecies are outside the geometric claim.',
  });
}

export function makeEstonianGraminoidFixture(profileId: EstonianGraminoidProfileId): EstonianGraminoidFixture {
  switch (profileId) {
    case ESTONIAN_GRAMINOID_PROFILE_IDS.AGROSTIS_CAPILLARIS: return makeAgrostisCapillaris();
    case ESTONIAN_GRAMINOID_PROFILE_IDS.AVENELLA_FLEXUOSA: return makeAvenellaFlexuosa();
    case ESTONIAN_GRAMINOID_PROFILE_IDS.CALAMAGROSTIS_CANESCENS: return makeCalamagrostisCanescens();
    case ESTONIAN_GRAMINOID_PROFILE_IDS.CAREX_CESPITOSA: return makeCarexCespitosa();
    case ESTONIAN_GRAMINOID_PROFILE_IDS.ERIOPHORUM_VAGINATUM: return makeEriophorumVaginatum();
    default: throw new Error(`unknown Estonian graminoid profile id ${profileId}`);
  }
}

export function makeAllEstonianGraminoidFixtures(): EstonianGraminoidFixture[] {
  return [0, 1, 2, 3, 4].map((profileId) => makeEstonianGraminoidFixture(profileId as EstonianGraminoidProfileId));
}
