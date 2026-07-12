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
import type { SpeciesParams } from './VegTypes';

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

const BUSH_HAZEL: SpeciesParams = {
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
    leaf: { len: 1.0, width: 0.6, shapePow: 1.2, fold: 0.3, curl: 0.2, needleCount: 0, brush: 0 },
  },
  flare: { amp: 0.2, height: 0.3, lobes: 3 },
  barkLayer: 2,
  foliageColor: { r: 0.055, g: 0.125, b: 0.03, hueVar: 0.24 },
  brokenTop: 0,
  stubChance: 0.02,
};

const BUSH_PINKFLOWER: SpeciesParams = {
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
    leaf: { len: 1.0, width: 0.5, shapePow: 1.25, fold: 0.28, curl: 0.18, needleCount: 0, brush: 0 },
  },
  flare: { amp: 0.2, height: 0.3, lobes: 3 },
  barkLayer: 2,
  foliageColor: { r: 0.05, g: 0.115, b: 0.032, hueVar: 0.2 },
  blossom: { r: 0.58, g: 0.16, b: 0.24, frac: 0.56 },
  brokenTop: 0,
  stubChance: 0.02,
};

const BUSH_JUNIPER: SpeciesParams = {
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
    leaf: { len: 0.05, width: 0.012, shapePow: 1, fold: 0, curl: 0, needleCount: 26, brush: 0 },
  },
  flare: { amp: 0.25, height: 0.25, lobes: 3 },
  barkLayer: 4,
  foliageColor: { r: 0.05, g: 0.095, b: 0.055, hueVar: 0.18 },
  brokenTop: 0,
  stubChance: 0.05,
};

export const UNDERSTORY_SPECIES: readonly SpeciesParams[] = [
  BUSH_HAZEL,
  BUSH_PINKFLOWER,
  BUSH_JUNIPER,
];

/** per-stem real-leaf anchor budget for the shrub crown (foliageMode 'mesh').
 *  Understory is dense (~495 k instances) + short-range (≤170 m), so the crown is
 *  kept LOW-POLY: this strides buildTree's anchors down to ~10 leaf clusters /
 *  needle sprays per stem — a small leafy mass, not a hero canopy. Measured merged-
 *  shrub crown: ~1.4–2.4 k tris (leaf-cluster hazel/pink), ~4.6–6.6 k (needle-spray
 *  juniper); the aggregate DAG coarsens it with distance and it culls at clsMaxDist. */
const SHRUB_LEAF_ANCHORS = 10;

/**
 * multi-stem shrub: 3–5 leaning stems merged into ONE bark geometry + ONE leaf
 * crown geometry. The crown is the SAME real MESH foliage the tree hero ring builds
 * (buildTree foliageMode 'mesh' → leaf-cluster / needle-spray per anchor), strided to
 * SHRUB_LEAF_ANCHORS so it stays cheap. The bark stream is byte-identical to the
 * bark-only path (foliage forks its own RNG in buildTree, never touching the stem/
 * tube stream), so this adds leaves without moving any existing placement.
 */
export function buildShrub(
  sp: SpeciesParams,
  rng: Rng,
): { bark: BufferGeometry; crown: BufferGeometry | null; barkTris: number; crownTris: number } {
  const stems = 3 + rng.int(3);
  const barkG = new MeshGrower();
  const crownG = new MeshGrower();
  let crownTris = 0;
  const m = new Matrix4();
  const q = new Quaternion();
  const p = new Vector3();
  for (let i = 0; i < stems; i++) {
    const a = (i / stems) * Math.PI * 2 + rng.float();
    const lean = 0.12 + rng.float() * 0.22;
    const tree = buildTree(sp, rng.fork(`stem${i}`), {
      foliageMode: 'mesh',
      hero: { meshAnchorTarget: SHRUB_LEAF_ANCHORS },
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
    if (tree.foliageMesh) {
      appendGeometry(crownG, tree.foliageMesh, m);
      crownTris = crownG.triCount;
    }
  }
  const bark = barkG.build();
  const crown = crownTris > 0 ? crownG.build() : null;
  return { bark, crown, barkTris: barkG.triCount, crownTris };
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

/**
 * Understory fern — a fully 3-D BIPINNATE shuttlecock (the canonical fern
 * algorithm: a self-similar rachis → pinnae → pinnules hierarchy, cf. the
 * Barnsley IFS / an L-system / Infinigen's 2-pinnate composition; asset-gen
 * understory-communities.toml `ostrich_fern`/`lady_fern`). 6–8 fronds radiate
 * from one crown; each frond is a rachis arcing up-and-out, bearing paired
 * PINNAE, and each pinna is ITSELF a mini-frond bearing paired PINNULES (the
 * recursion = the lacy detail). Both levels follow a lanceolate envelope (widest
 * ~mid, tapering base+tip — Matteuccia/Athyrium; NCSU/RHS: twice-pinnate vase).
 *
 * VOLUME (NOT a flat billboard): the frond is NOT coplanar. A per-frond CUP lifts
 * both pinna rows up out of the base plane into a shallow trough (the two sides
 * face different ways), each pinnule adds its own out-of-plane tilt, and EVERY
 * leaflet's normal is the true cross-product of its edges — so light catches the
 * surface from many angles and no single plane dominates. Tall (~0.7–1.3 m pre
 * scatter-scale) so it stands above the grass floor.
 *
 * COLOUR (NOT a flat tint): the leaf material tints per-mesh but modulates by
 * per-vertex vdata.x (hue jitter × the high FERN hueVar → warm↔cool green) and
 * vdata.w (AO → luminance). Every pinnule draws an independent hue jitter + graded
 * AO (interior/base dark → tips bright); rachides carry a paler, warmer stem tone.
 *
 * vdata: x hue jitter · y sway flex · z phase · w AO. ~2–2.6 k tris/variant — a
 * detailed near-field plant (short range, DAG-coarsened; on par with the shrub
 * crowns already shipping).
 */
export function buildFern(rng: Rng): BufferGeometry {
  const g = new MeshGrower();
  const fronds = 6 + rng.int(3); // 6–8 → a full shuttlecock
  const hueBase = (rng.float() - 0.5) * 0.35;
  const sv = 0.9 + rng.float() * 0.4; // per-fern overall size (m, pre scatter scale)
  const up = new Vector3(0, 1, 0);
  // scratch
  const T = new Vector3();
  const S = new Vector3();
  const Fn = new Vector3();
  const base = new Vector3();
  const paxis = new Vector3();
  const pwid = new Vector3();
  const pfaceN = new Vector3();
  const lax = new Vector3();
  const lwid = new Vector3();
  const lnrm = new Vector3();
  const lbase = new Vector3();
  const ltip = new Vector3();
  const a0p = new Vector3();
  const a1p = new Vector3();
  const c0p = new Vector3();
  const c1p = new Vector3();

  // tapered leaflet quad in 3-D: base b, unit axis ax (length len), unit width w,
  // half-widths wb→we; true face normal ax×w; per-leaflet hue jitter + graded AO.
  const leaflet = (
    b: Vector3, ax: Vector3, w: Vector3, len: number, wb: number, we: number,
    flex: number, hue: number, hueJit: number, phase: number, aoB: number, aoT: number,
  ): void => {
    ltip.copy(ax).multiplyScalar(len).add(b);
    lnrm.copy(ax).cross(w).normalize();
    a0p.copy(w).multiplyScalar(-wb).add(b);
    a1p.copy(w).multiplyScalar(wb).add(b);
    c1p.copy(w).multiplyScalar(we).add(ltip);
    c0p.copy(w).multiplyScalar(-we).add(ltip);
    const j = hue + hueJit;
    const v0 = g.vertex(a0p.x, a0p.y, a0p.z, lnrm.x, lnrm.y, lnrm.z, 0, 0, j, flex, phase, aoB);
    const v1 = g.vertex(a1p.x, a1p.y, a1p.z, lnrm.x, lnrm.y, lnrm.z, 1, 0, j, flex, phase, aoB);
    const v2 = g.vertex(c1p.x, c1p.y, c1p.z, lnrm.x, lnrm.y, lnrm.z, 1, 1, j, flex, phase, aoT);
    const v3 = g.vertex(c0p.x, c0p.y, c0p.z, lnrm.x, lnrm.y, lnrm.z, 0, 1, j, flex, phase, aoT);
    g.quad(v0, v1, v2, v3);
  };
  // thin ribbon along an axis (rachis/sub-rachis stem), pale/warmer tone.
  const ribbon = (b: Vector3, ax: Vector3, w: Vector3, len: number, hw: number, hue: number, phase: number): void => {
    ltip.copy(ax).multiplyScalar(len).add(b);
    lnrm.copy(ax).cross(w).normalize();
    a0p.copy(w).multiplyScalar(-hw).add(b);
    a1p.copy(w).multiplyScalar(hw).add(b);
    c1p.copy(w).multiplyScalar(hw * 0.4).add(ltip);
    c0p.copy(w).multiplyScalar(-hw * 0.4).add(ltip);
    const v0 = g.vertex(a0p.x, a0p.y, a0p.z, lnrm.x, lnrm.y, lnrm.z, 0, 0, hue, 0.4, phase, 0.6);
    const v1 = g.vertex(a1p.x, a1p.y, a1p.z, lnrm.x, lnrm.y, lnrm.z, 1, 0, hue, 0.4, phase, 0.6);
    const v2 = g.vertex(c1p.x, c1p.y, c1p.z, lnrm.x, lnrm.y, lnrm.z, 1, 1, hue, 0.5, phase, 0.72);
    const v3 = g.vertex(c0p.x, c0p.y, c0p.z, lnrm.x, lnrm.y, lnrm.z, 0, 1, hue, 0.5, phase, 0.72);
    g.quad(v0, v1, v2, v3);
  };

  for (let f = 0; f < fronds; f++) {
    const az = (f / fronds) * Math.PI * 2 + rng.float() * 0.4;
    const outX = Math.cos(az);
    const outZ = Math.sin(az);
    const H = sv * (0.74 + rng.float() * 0.34); // frond height
    const reach = sv * (0.34 + rng.float() * 0.2); // outward splay of the arching tip
    const droop = 0.28 + rng.float() * 0.14; // tip arch-over
    const phase = rng.float() * Math.PI * 2;
    const frondHue = hueBase + (rng.float() - 0.5) * 0.3; // per-frond tonal offset
    const blade = sv * (0.17 + rng.float() * 0.05); // peak pinna length
    const cup = 0.5 + rng.float() * 0.22; // radians the pinna rows lift out of the base plane

    // ---- 3-D rachis: rises near-vertical, then ARCHES up-and-over outward ----
    const M = 12;
    const R: Vector3[] = [];
    for (let i = 0; i <= M; i++) {
      const t = i / M;
      const rad = reach * Math.pow(t, 1.25); // stays near-vertical at the base, splays at the tip
      const y = H * Math.sin(t * 1.8) - droop * H * Math.pow(t, 2.4); // peak ~0.85, drooping tip
      R.push(new Vector3(outX * rad, 0.03 + y, outZ * rad));
    }
    const frameAt = (i0: number): void => {
      // tangent, horizontal side S, frond face normal Fn (up-ish)
      T.copy(R[Math.min(M, i0 + 1)] as Vector3).sub(R[Math.max(0, i0 - 1)] as Vector3).normalize();
      S.copy(up).cross(T);
      if (S.lengthSq() < 1e-6) S.set(1, 0, 0);
      S.normalize();
      Fn.copy(T).cross(S).normalize();
    };

    // main rachis ribbon
    for (let i = 0; i < M; i++) {
      frameAt(i);
      const t = i / M;
      const w2 = 0.006 * sv * (1 - 0.5 * t);
      const seg = (R[i + 1] as Vector3).distanceTo(R[i] as Vector3);
      lax.copy(R[i + 1] as Vector3).sub(R[i] as Vector3).normalize();
      ribbon(R[i] as Vector3, lax, S, seg, w2, frondHue + 0.55, phase);
    }

    // ---- pinnae; each is itself pinnate ----
    const P = 7 + rng.int(3); // pinna pairs
    for (let j = 1; j <= P; j++) {
      const t = j / (P + 1);
      const env = Math.pow(Math.sin(Math.PI * Math.min(1, t)), 0.7);
      const lp = blade * (0.3 + 0.95 * env);
      if (lp < 0.02) continue;
      const fi = t * M;
      const i0 = Math.min(M - 1, Math.floor(fi));
      base.copy(R[i0] as Vector3).lerp(R[i0 + 1] as Vector3, fi - i0);
      frameAt(i0);
      const flexP = 0.3 + 0.6 * t;
      const cs = Math.cos(cup);
      const sn = Math.sin(cup);
      for (const s of [1, -1]) {
        // pinna axis: sideways (s) lifted toward the face by `cup` + angled to the tip
        paxis.copy(S).multiplyScalar(s * cs).addScaledVector(Fn, sn).addScaledVector(T, 0.42).normalize();
        // pinna width dir (perpendicular to the pinna axis, ~in the frond face)
        pwid.copy(paxis).cross(Fn);
        if (pwid.lengthSq() < 1e-5) pwid.copy(paxis).cross(up);
        pwid.normalize();
        pfaceN.copy(paxis).cross(pwid).normalize(); // pinna face normal
        ribbon(base, paxis, pwid, lp, 0.004 * sv, frondHue + 0.5, phase);
        // pinnules along the pinna, both sides, lanceolate + out-of-plane tilt.
        // Spaced (lp/0.04) + narrow (0.2·pl) so the blade reads LACY, not a solid leaf.
        const K = Math.max(3, Math.round(lp / 0.04));
        for (let k = 1; k <= K; k++) {
          const tp = k / (K + 0.5);
          lbase.copy(paxis).multiplyScalar(lp * tp).add(base);
          const penv = Math.pow(Math.sin(Math.PI * Math.min(1, tp)), 0.6);
          const pl = lp * (0.2 + 0.44 * penv);
          if (pl < 0.008) continue;
          const aoB = 0.34 + 0.28 * tp;
          const aoT = Math.min(1, 0.66 + 0.34 * tp);
          for (const ss of [1, -1]) {
            // pinnule axis: off the pinna (ss) toward its tip + a random out-of-plane tilt
            const tilt = (rng.float() - 0.5) * 0.8;
            lax.copy(pwid).multiplyScalar(ss * 0.86).addScaledVector(paxis, 0.5).addScaledVector(pfaceN, tilt).normalize();
            lwid.copy(lax).cross(pfaceN);
            if (lwid.lengthSq() < 1e-5) lwid.copy(lax).cross(up);
            lwid.normalize();
            const hueJit = rng.float() * 2 - 1; // per-pinnule tonal variation (−1..1)
            leaflet(lbase, lax, lwid, pl, 0.2 * pl, 0.02 * pl, flexP + 0.15, frondHue, hueJit, phase, aoB, aoT);
          }
        }
      }
    }
  }
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
