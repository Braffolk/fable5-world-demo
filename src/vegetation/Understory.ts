/**
 * Understory: shrubs ×3 (incl. the reference's pink flowering shrub),
 * ferns (frond rosettes from a captured pinnate frond), flowers ×4.
 * Shrubs are multi-stem trees grown from bush-tuned species params and
 * merged; ferns/flowers are bespoke small builders on MeshGrower.
 */

import { Matrix4, Quaternion, Vector3 } from 'three';
import type { BufferGeometry } from 'three';
import type { Rng } from '../core/Seed';
import { buildTree } from './TreeBuilder';
import { MeshGrower } from './TubeMesh';
import type { LeafAnchor, SpeciesParams } from './VegTypes';
import { buildFoliageCards } from './FoliageCards';

// ---------------------------------------------------------------------------
// Shrub species (bush-tuned growth params; same grammar)
// ---------------------------------------------------------------------------

const bushLevels = (gnarl: number): SpeciesParams['levels'] => [
  {
    density: 0, whorl: 0, childStart: 0, childEnd: 0,
    angleBase: 0, angleTip: 0, lenRatio: 0, lenJitter: 0, radRatio: 0,
    segs: 5, wander: 0.18 * gnarl, gravitropism: 0.06, droop: 0, tipCurl: 0, taper: 0.9,
  },
  {
    density: 4.5, whorl: 0, childStart: 0.2, childEnd: 1.0,
    angleBase: 1.0, angleTip: 0.5, lenRatio: 0.62, lenJitter: 0.4, radRatio: 0.55,
    segs: 4, wander: 0.16 * gnarl, gravitropism: 0.1, droop: 0.2, tipCurl: 0.1, taper: 0.85,
  },
  {
    density: 7.0, whorl: 0, childStart: 0.2, childEnd: 1.0,
    angleBase: 0.85, angleTip: 0.5, lenRatio: 0.45, lenJitter: 0.4, radRatio: 0.55,
    segs: 2, wander: 0.2 * gnarl, gravitropism: 0.05, droop: 0.15, tipCurl: 0.05, taper: 0.85,
    planar: 0.5,
  },
];

export const BUSH_HAZEL: SpeciesParams = {
  id: 'bushHazel',
  label: 'Hazel shrub',
  kind: 'broadleaf',
  height: [1.9, 2.9],
  trunkRadiusK: 0.02,
  crown: 'dome',
  asym: 0.35,
  levels: bushLevels(1),
  foliage: {
    kind: 'leafCluster',
    anchorLevel: 2,
    spacing: 0.09,
    tStart: 0.15,
    scale: [0.08, 0.13],
    tilt: 0.9,
    clusterSize: [2, 3],
    normalBend: 0.6,
    planarLeaves: true,
    card: { mode: 'cross', sizeK: 2.3 },
    leaf: { len: 1.0, width: 0.6, shapePow: 1.2, fold: 0.3, curl: 0.2, needleCount: 0, brush: 0 },
  },
  flare: { amp: 0.2, height: 0.3, lobes: 3 },
  barkLayer: 2,
  barkRepeats: 2,
  foliageColor: { r: 0.055, g: 0.125, b: 0.03, hueVar: 0.24 },
  brokenTop: 0,
  stubChance: 0.02,
};

export const BUSH_PINKFLOWER: SpeciesParams = {
  id: 'bushPink',
  label: 'Pink flowering shrub',
  kind: 'broadleaf',
  height: [1.5, 2.4],
  trunkRadiusK: 0.018,
  crown: 'dome',
  asym: 0.3,
  levels: bushLevels(1.2),
  foliage: {
    kind: 'leafCluster',
    anchorLevel: 2,
    spacing: 0.055,
    tStart: 0.12,
    scale: [0.09, 0.14],
    tilt: 0.95,
    clusterSize: [2, 3],
    normalBend: 0.62,
    planarLeaves: true,
    card: { mode: 'cross', sizeK: 2.3 },
    leaf: { len: 1.0, width: 0.5, shapePow: 1.25, fold: 0.28, curl: 0.18, needleCount: 0, brush: 0 },
  },
  flare: { amp: 0.2, height: 0.3, lobes: 3 },
  barkLayer: 2,
  barkRepeats: 2,
  foliageColor: { r: 0.05, g: 0.115, b: 0.032, hueVar: 0.2 },
  blossom: { r: 0.58, g: 0.16, b: 0.24, frac: 0.56 },
  brokenTop: 0,
  stubChance: 0.02,
};

export const BUSH_JUNIPER: SpeciesParams = {
  id: 'bushJuniper',
  label: 'Juniper mound',
  kind: 'conifer',
  height: [0.9, 1.5],
  trunkRadiusK: 0.03,
  crown: 'dome',
  asym: 0.4,
  levels: [
    {
      density: 0, whorl: 0, childStart: 0, childEnd: 0,
      angleBase: 0, angleTip: 0, lenRatio: 0, lenJitter: 0, radRatio: 0,
      segs: 4, wander: 0.3, gravitropism: -0.12, droop: 0, tipCurl: 0.05, taper: 0.8,
    },
    {
      density: 7, whorl: 0, childStart: 0.05, childEnd: 1.0,
      angleBase: 1.5, angleTip: 0.7, lenRatio: 0.85, lenJitter: 0.4, radRatio: 0.6,
      segs: 4, wander: 0.22, gravitropism: 0.12, droop: 0.25, tipCurl: 0.18, taper: 0.85,
    },
    {
      density: 8, whorl: 0, childStart: 0.2, childEnd: 1.0,
      angleBase: 0.9, angleTip: 0.5, lenRatio: 0.4, lenJitter: 0.4, radRatio: 0.55,
      segs: 2, wander: 0.2, gravitropism: 0.08, droop: 0.1, tipCurl: 0.1, taper: 0.85,
      planar: 0.6,
    },
  ],
  foliage: {
    kind: 'needleSpray',
    anchorLevel: 2,
    spacing: 0.07,
    tStart: 0.1,
    scale: [0.12, 0.2],
    tilt: 0.55,
    clusterSize: [1, 1],
    normalBend: 0.6,
    planarLeaves: true,
    card: { mode: 'lying', sizeK: 2.5 },
    leaf: { len: 0.05, width: 0.012, shapePow: 1, fold: 0, curl: 0, needleCount: 26, brush: 0 },
  },
  flare: { amp: 0.25, height: 0.25, lobes: 3 },
  barkLayer: 4,
  barkRepeats: 2,
  foliageColor: { r: 0.05, g: 0.095, b: 0.055, hueVar: 0.18 },
  brokenTop: 0,
  stubChance: 0.05,
};

export const UNDERSTORY_SPECIES: readonly SpeciesParams[] = [
  BUSH_HAZEL,
  BUSH_PINKFLOWER,
  BUSH_JUNIPER,
];

/** multi-stem shrub: 3–5 leaning stems merged into one bark+foliage pair */
export function buildShrub(
  sp: SpeciesParams,
  rng: Rng,
): { bark: BufferGeometry; foliage: BufferGeometry | null; tris: number } {
  const stems = 3 + rng.int(3);
  const barkG = new MeshGrower();
  const folG = new MeshGrower();
  const m = new Matrix4();
  const q = new Quaternion();
  const p = new Vector3();
  let any = false;
  for (let i = 0; i < stems; i++) {
    const a = (i / stems) * Math.PI * 2 + rng.float();
    const lean = 0.12 + rng.float() * 0.22;
    const tree = buildTree(sp, rng.fork(`stem${i}`), {
      inst: {
        leanX: Math.cos(a) * lean,
        leanZ: Math.sin(a) * lean,
        age: 0.4 + rng.float() * 0.5,
      },
    });
    p.set(Math.cos(a) * 0.09, 0, Math.sin(a) * 0.09);
    q.identity();
    m.compose(p, q, new Vector3(1, 1, 1));
    appendGeometry(barkG, tree.bark, m);
    if (tree.foliage) {
      appendGeometry(folG, tree.foliage, m);
      any = true;
    }
  }
  const bark = barkG.build();
  const foliage = any ? folG.build() : null;
  return { bark, foliage, tris: barkG.triCount + folG.triCount };
}

/** append a built BufferGeometry into a grower (positions/normals/uv/vdata) */
function appendGeometry(g: MeshGrower, src: BufferGeometry, m: Matrix4): void {
  const pos = src.getAttribute('position');
  const nrm = src.getAttribute('normal');
  const uvA = src.getAttribute('uv');
  const dat = src.getAttribute('vdata');
  const idx = src.getIndex();
  const p = new Vector3();
  const n = new Vector3();
  const base = g.vertCount;
  for (let i = 0; i < pos.count; i++) {
    p.set(pos.getX(i), pos.getY(i), pos.getZ(i)).applyMatrix4(m);
    n.set(nrm.getX(i), nrm.getY(i), nrm.getZ(i)).transformDirection(m);
    g.vertex(
      p.x, p.y, p.z, n.x, n.y, n.z,
      uvA ? uvA.getX(i) : 0, uvA ? uvA.getY(i) : 0,
      dat ? dat.getX(i) : 0, dat ? dat.getY(i) : 0,
      dat ? dat.getZ(i) : 0, dat ? dat.getW(i) : 1,
    );
  }
  if (idx) {
    for (let i = 0; i < idx.count; i += 3) {
      g.tri(base + idx.getX(i), base + idx.getX(i + 1), base + idx.getX(i + 2));
    }
  }
}

// ---------------------------------------------------------------------------
// Ferns
// ---------------------------------------------------------------------------

/** capture species for the fern frond atlas (pinnate comb spray) */
export const FERN_CAPTURE: SpeciesParams = {
  ...BUSH_HAZEL,
  id: 'fern',
  label: 'Fern',
  foliage: {
    kind: 'needleSpray',
    anchorLevel: 2,
    spacing: 0.1,
    tStart: 0.1,
    scale: [0.3, 0.45],
    tilt: 0.6,
    clusterSize: [1, 1],
    normalBend: 0.55,
    planarLeaves: true,
    captureStyle: 'frond',
    card: { mode: 'cross', sizeK: 2.2 },
    leaf: { len: 0.1, width: 0.032, shapePow: 1, fold: 0, curl: 0, needleCount: 30, brush: 0 },
  },
  foliageColor: { r: 0.045, g: 0.14, b: 0.028, hueVar: 0.22 },
};

/** fern plant: rosette of 6–10 frond cards rising from a center */
export function buildFern(rng: Rng): BufferGeometry {
  const g = new MeshGrower();
  const fronds = 6 + rng.int(5);
  const anchors: LeafAnchor[] = [];
  const q = new Quaternion();
  const qt = new Quaternion();
  const Y = new Vector3(0, 1, 0);
  const X = new Vector3(1, 0, 0);
  for (let i = 0; i < fronds; i++) {
    const az = (i / fronds) * Math.PI * 2 + rng.float() * 0.6;
    const pitch = 0.75 + rng.float() * 0.4; // steep at the base, arches over
    q.setFromAxisAngle(Y, az);
    qt.setFromAxisAngle(X, -(Math.PI / 2 - pitch));
    q.multiply(qt);
    anchors.push({
      pos: new Vector3(Math.cos(az) * 0.03, 0.02, Math.sin(az) * 0.03),
      quat: q.clone(),
      scale: 0.2 + rng.float() * 0.14,
      hue: rng.float() * 2 - 1,
      age: rng.float() * 0.4,
    });
  }
  buildFoliageCards(g, anchors, { mode: 'lying', sizeK: 2.4, bend: 1.0 }, rng);
  return g.build();
}

// ---------------------------------------------------------------------------
// Flowers
// ---------------------------------------------------------------------------

export type FlowerKind = 'umbel' | 'bell' | 'daisy';

/**
 * Small flowering plant: thin stem + leaves + REAL petal geometry.
 * vdata.x: 0 = stem/leaf (green), 1 = petal, 0.5 = flower center.
 */
export function buildFlower(kind: FlowerKind, rng: Rng): BufferGeometry {
  const g = new MeshGrower();
  const H = kind === 'umbel' ? 0.55 + rng.float() * 0.3 : 0.28 + rng.float() * 0.2;
  const sway = (rng.float() - 0.5) * 0.25;
  // stem: 2-segment thin strip pair (cross)
  const top = new Vector3(sway * H, H, sway * H * 0.6);
  const mid = new Vector3(sway * H * 0.4, H * 0.55, 0);
  for (let pl = 0; pl < 2; pl++) {
    const w = 0.006;
    const ox = pl === 0 ? w : 0;
    const oz = pl === 0 ? 0 : w;
    const a0 = g.vertex(-ox, 0, -oz, 0, 0, 1, 0, 0, 0, 0, 0, 0.8);
    const a1 = g.vertex(ox, 0, oz, 0, 0, 1, 1, 0, 0, 0, 0, 0.8);
    const b0 = g.vertex(mid.x - ox, mid.y, mid.z - oz, 0, 0, 1, 0, 0.5, 0, 0, 0, 0.9);
    const b1 = g.vertex(mid.x + ox, mid.y, mid.z + oz, 0, 0, 1, 1, 0.5, 0, 0, 0, 0.9);
    const c0 = g.vertex(top.x - ox * 0.6, top.y, top.z - oz * 0.6, 0, 0, 1, 0, 1, 0, 0, 0, 1);
    const c1 = g.vertex(top.x + ox * 0.6, top.y, top.z + oz * 0.6, 0, 0, 1, 1, 1, 0, 0, 0, 1);
    g.quad(a0, a1, b1, b0);
    g.quad(b0, b1, c1, c0);
  }
  // 2-3 basal leaves: small bent quads
  const leaves = 2 + rng.int(2);
  for (let i = 0; i < leaves; i++) {
    const az = rng.float() * Math.PI * 2;
    const ll = 0.07 + rng.float() * 0.06;
    const lx = Math.cos(az);
    const lz = Math.sin(az);
    const y0 = 0.02 + rng.float() * H * 0.3;
    const a0 = g.vertex(lx * 0.01, y0, lz * 0.01, 0, 1, 0, 0, 0, 0, 0, 0, 0.85);
    const a1 = g.vertex(lx * 0.01 - lz * 0.012, y0 + 0.005, lz * 0.01 + lx * 0.012, 0, 1, 0, 1, 0, 0, 0, 0, 0.85);
    const b0 = g.vertex(lx * ll, y0 + ll * 0.5, lz * ll, 0, 1, 0, 0, 1, 0, 0, 0, 1);
    const b1 = g.vertex(lx * ll - lz * 0.01, y0 + ll * 0.5 + 0.005, lz * ll + lx * 0.01, 0, 1, 0, 1, 1, 0, 0, 0, 1);
    g.quad(a0, a1, b1, b0);
  }
  // head(s)
  const head = (cx: number, cy: number, cz: number, s: number): void => {
    if (kind === 'daisy') {
      const petals = 8 + rng.int(5);
      for (let i = 0; i < petals; i++) {
        const az = (i / petals) * Math.PI * 2;
        const dx = Math.cos(az);
        const dz = Math.sin(az);
        const pw = s * 0.3;
        const plen = s;
        const a0 = g.vertex(cx + dx * s * 0.18 - dz * pw * 0.5, cy, cz + dz * s * 0.18 + dx * pw * 0.5, 0, 1, 0.2, 0, 0, 1, 0, 0, 1);
        const a1 = g.vertex(cx + dx * s * 0.18 + dz * pw * 0.5, cy, cz + dz * s * 0.18 - dx * pw * 0.5, 0, 1, 0.2, 1, 0, 1, 0, 0, 1);
        const b0 = g.vertex(cx + dx * plen - dz * pw * 0.25, cy + s * 0.16, cz + dz * plen + dx * pw * 0.25, 0, 1, 0.2, 0.4, 1, 1, 0, 0, 1);
        const b1 = g.vertex(cx + dx * plen + dz * pw * 0.25, cy + s * 0.16, cz + dz * plen - dx * pw * 0.25, 0, 1, 0.2, 0.6, 1, 1, 0, 0, 1);
        g.quad(a0, a1, b1, b0);
      }
      // center disc: small fan
      const c = g.vertex(cx, cy + s * 0.08, cz, 0, 1, 0, 0.5, 0.5, 0.5, 0, 0, 1);
      const ringN = 6;
      const ring: number[] = [];
      for (let i = 0; i <= ringN; i++) {
        const az = (i / ringN) * Math.PI * 2;
        ring.push(
          g.vertex(cx + Math.cos(az) * s * 0.2, cy + s * 0.03, cz + Math.sin(az) * s * 0.2, 0, 1, 0, 0.5, 0.5, 0.5, 0, 0, 1),
        );
      }
      for (let i = 0; i < ringN; i++) g.tri(c, ring[i + 1] as number, ring[i] as number);
    } else if (kind === 'bell') {
      // drooping bell: cone of petals pointing down
      const petals = 5;
      for (let i = 0; i < petals; i++) {
        const az = (i / petals) * Math.PI * 2;
        const dx = Math.cos(az);
        const dz = Math.sin(az);
        const a0 = g.vertex(cx + dx * s * 0.12, cy, cz + dz * s * 0.12, dx, 0.3, dz, 0.4, 0, 1, 0, 0, 1);
        const a1 = g.vertex(cx + Math.cos(az + 1.25) * s * 0.12, cy, cz + Math.sin(az + 1.25) * s * 0.12, dx, 0.3, dz, 0.6, 0, 1, 0, 0, 1);
        const b0 = g.vertex(cx + dx * s * 0.3, cy - s * 0.5, cz + dz * s * 0.3, dx, 0, dz, 0.4, 1, 1, 0, 0, 1);
        const b1 = g.vertex(cx + Math.cos(az + 1.25) * s * 0.3, cy - s * 0.5, cz + Math.sin(az + 1.25) * s * 0.3, dx, 0, dz, 0.6, 1, 1, 0, 0, 1);
        g.quad(a0, a1, b1, b0);
      }
    } else {
      // umbel: cluster of tiny 4-petal florets on a dome
      const florets = 12 + rng.int(8);
      for (let i = 0; i < florets; i++) {
        const az = rng.float() * Math.PI * 2;
        const rr = Math.sqrt(rng.float()) * s;
        const fx = cx + Math.cos(az) * rr;
        const fz = cz + Math.sin(az) * rr;
        const fy = cy + (1 - (rr / s) * (rr / s)) * s * 0.35;
        const fs = s * 0.16;
        const a0 = g.vertex(fx - fs, fy, fz - fs, 0, 1, 0, 0, 0, 1, 0, 0, 1);
        const a1 = g.vertex(fx + fs, fy, fz - fs, 0, 1, 0, 1, 0, 1, 0, 0, 1);
        const b1 = g.vertex(fx + fs, fy + fs * 0.2, fz + fs, 0, 1, 0, 1, 1, 1, 0, 0, 1);
        const b0 = g.vertex(fx - fs, fy + fs * 0.2, fz + fs, 0, 1, 0, 0, 1, 1, 0, 0, 1);
        g.quad(a0, a1, b1, b0);
      }
    }
  };
  if (kind === 'bell') {
    // several bells hanging along the stem top
    const bells = 2 + rng.int(3);
    for (let i = 0; i < bells; i++) {
      const t = 0.6 + (i / bells) * 0.4;
      head(top.x * t + 0.02 * i, H * t, top.z * t, 0.05 + rng.float() * 0.02);
    }
  } else {
    head(top.x, H + 0.02, top.z, kind === 'umbel' ? 0.09 + rng.float() * 0.04 : 0.045 + rng.float() * 0.02);
  }
  return g.build();
}
