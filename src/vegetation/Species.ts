/**
 * Species presets — 6+ species per spec §2 (conifer ×2, broadleaf ×2,
 * karst-gnarled cliff tree, standing snag). Numbers are growth-grammar
 * parameters (Skeleton.ts); foliage geometry params feed LeafMesh.ts.
 *
 * Structure rule (user feedback): foliage NEVER sits on primaries — every
 * species ends in a fine twig/branchlet level (planar lattice for spruce
 * boughs / beech plates) and the needles/leaves attach THERE. The lushness
 * comes from thousands of small sprays on that lattice.
 */

import type { SpeciesParams } from './VegTypes';

export const SPRUCE: SpeciesParams = {
  id: 'spruce',
  label: 'Spruce (conifer)',
  kind: 'conifer',
  height: [19, 27],
  trunkRadiusK: 0.017,
  crown: 'cone',
  asym: 0.22,
  levels: [
    {
      density: 0, whorl: 0, childStart: 0, childEnd: 0,
      angleBase: 0, angleTip: 0, lenRatio: 0, lenJitter: 0, radRatio: 0,
      segs: 16, wander: 0.015, gravitropism: 0.05, droop: 0, tipCurl: 0, taper: 1.0,
    },
    {
      // primaries: near-horizontal spokes, slight sag, up-hooked tips
      density: 5.0, whorl: 4, childStart: 0.09, childEnd: 0.985,
      angleBase: 1.78, angleTip: 0.55, lenRatio: 0.19, lenJitter: 0.2, radRatio: 0.32,
      segs: 6, wander: 0.06, gravitropism: -0.03, droop: 0.3, tipCurl: 0.28, taper: 1.05,
    },
    {
      // branchlets: two-sided planar lattice filling the bough plane
      density: 5.5, whorl: 0, childStart: 0.12, childEnd: 0.98,
      angleBase: 1.05, angleTip: 0.8, lenRatio: 0.24, lenJitter: 0.35, radRatio: 0.4,
      segs: 3, wander: 0.08, gravitropism: -0.05, droop: 0.45, tipCurl: 0.12, taper: 0.9,
      planar: 1,
    },
  ],
  foliage: {
    kind: 'needleSpray',
    anchorLevel: 2,
    spacing: 0.16,
    tStart: 0.05,
    scale: [0.22, 0.35],
    tilt: 0.5,
    clusterSize: [1, 1],
    normalBend: 0.62,
    planarLeaves: true,
    leaf: { len: 0.1, width: 0.024, shapePow: 1, fold: 0, curl: 0, needleCount: 30, brush: 0 },
  },
  flare: { amp: 0.5, height: 1.0, lobes: 5 },
  barkLayer: 0,
  barkRepeats: 5,
  foliageColor: { r: 0.045, g: 0.10, b: 0.05, hueVar: 0.24 },
  brokenTop: 0,
  stubChance: 0.02,
};

export const PINE: SpeciesParams = {
  id: 'pine',
  label: 'Mountain pine (conifer)',
  kind: 'conifer',
  height: [12, 19],
  trunkRadiusK: 0.021,
  crown: 'dome',
  asym: 0.34,
  levels: [
    {
      density: 0, whorl: 0, childStart: 0, childEnd: 0,
      angleBase: 0, angleTip: 0, lenRatio: 0, lenJitter: 0, radRatio: 0,
      segs: 12, wander: 0.06, gravitropism: 0.03, droop: 0, tipCurl: 0, taper: 0.92,
    },
    {
      density: 1.8, whorl: 3, childStart: 0.42, childEnd: 0.97,
      angleBase: 1.5, angleTip: 0.55, lenRatio: 0.45, lenJitter: 0.32, radRatio: 0.4,
      segs: 8, wander: 0.14, gravitropism: 0.08, droop: 0.3, tipCurl: 0.32, taper: 0.85,
    },
    {
      density: 2.6, whorl: 0, childStart: 0.35, childEnd: 1.0,
      angleBase: 0.9, angleTip: 0.55, lenRatio: 0.32, lenJitter: 0.34, radRatio: 0.45,
      segs: 4, wander: 0.13, gravitropism: 0.06, droop: 0.16, tipCurl: 0.22, taper: 0.85,
    },
    {
      // twiglets rising at the ends — pine carries needles on these
      density: 4.2, whorl: 0, childStart: 0.4, childEnd: 1.0,
      angleBase: 0.8, angleTip: 0.5, lenRatio: 0.4, lenJitter: 0.4, radRatio: 0.5,
      segs: 2, wander: 0.15, gravitropism: 0.1, droop: 0.1, tipCurl: 0.15, taper: 0.8,
    },
  ],
  foliage: {
    kind: 'needleSpray',
    anchorLevel: 3,
    spacing: 0.11,
    tStart: 0.3,
    scale: [0.26, 0.42],
    tilt: 0.55,
    clusterSize: [1, 1],
    normalBend: 0.66,
    leaf: { len: 0.21, width: 0.018, shapePow: 1, fold: 0, curl: 0, needleCount: 88, brush: 1 },
  },
  flare: { amp: 0.42, height: 0.8, lobes: 4 },
  barkLayer: 1,
  barkRepeats: 4,
  foliageColor: { r: 0.04, g: 0.092, b: 0.048, hueVar: 0.22 },
  brokenTop: 0,
  stubChance: 0.04,
};

export const BEECH: SpeciesParams = {
  id: 'beech',
  label: 'Beech (broadleaf)',
  kind: 'broadleaf',
  height: [13, 20],
  trunkRadiusK: 0.024,
  crown: 'ellipsoid',
  asym: 0.3,
  levels: [
    {
      density: 0, whorl: 0, childStart: 0, childEnd: 0,
      angleBase: 0, angleTip: 0, lenRatio: 0, lenJitter: 0, radRatio: 0,
      segs: 9, wander: 0.05, gravitropism: 0.04, droop: 0, tipCurl: 0, taper: 1.25,
    },
    {
      density: 1.5, whorl: 0, childStart: 0.32, childEnd: 0.94,
      angleBase: 1.05, angleTip: 0.5, lenRatio: 0.56, lenJitter: 0.26, radRatio: 0.5,
      segs: 8, wander: 0.1, gravitropism: 0.085, droop: 0.22, tipCurl: 0.12, taper: 0.95,
    },
    {
      density: 2.3, whorl: 0, childStart: 0.25, childEnd: 0.97,
      angleBase: 0.92, angleTip: 0.55, lenRatio: 0.46, lenJitter: 0.3, radRatio: 0.52,
      segs: 5, wander: 0.13, gravitropism: 0.05, droop: 0.3, tipCurl: 0.08, taper: 0.9,
    },
    {
      // distichous twig plates — beech's layered horizontal foliage
      density: 8.0, whorl: 0, childStart: 0.15, childEnd: 1.0,
      angleBase: 0.9, angleTip: 0.6, lenRatio: 0.28, lenJitter: 0.35, radRatio: 0.55,
      segs: 3, wander: 0.1, gravitropism: -0.02, droop: 0.15, tipCurl: 0.04, taper: 0.85,
      planar: 1,
    },
  ],
  foliage: {
    kind: 'leafCluster',
    anchorLevel: 3,
    spacing: 0.13,
    tStart: 0.1,
    scale: [0.16, 0.24],
    tilt: 1.0,
    clusterSize: [2, 3],
    normalBend: 0.7,
    planarLeaves: true,
    leaf: { len: 1.0, width: 0.42, shapePow: 1.15, fold: 0.32, curl: 0.22, needleCount: 0, brush: 0 },
  },
  flare: { amp: 0.55, height: 1.2, lobes: 6 },
  barkLayer: 2,
  barkRepeats: 4,
  foliageColor: { r: 0.06, g: 0.145, b: 0.035, hueVar: 0.3 },
  brokenTop: 0,
  stubChance: 0.02,
};

export const BIRCH: SpeciesParams = {
  id: 'birch',
  label: 'Birch (broadleaf)',
  kind: 'broadleaf',
  height: [9, 15],
  trunkRadiusK: 0.015,
  crown: 'column',
  asym: 0.26,
  levels: [
    {
      density: 0, whorl: 0, childStart: 0, childEnd: 0,
      angleBase: 0, angleTip: 0, lenRatio: 0, lenJitter: 0, radRatio: 0,
      segs: 11, wander: 0.05, gravitropism: 0.045, droop: 0, tipCurl: 0, taper: 1.1,
    },
    {
      density: 2.2, whorl: 0, childStart: 0.3, childEnd: 0.96,
      angleBase: 0.95, angleTip: 0.45, lenRatio: 0.4, lenJitter: 0.3, radRatio: 0.42,
      segs: 7, wander: 0.11, gravitropism: 0.02, droop: 0.4, tipCurl: -0.04, taper: 0.95,
    },
    {
      density: 3.8, whorl: 0, childStart: 0.3, childEnd: 1.0,
      angleBase: 0.8, angleTip: 0.5, lenRatio: 0.42, lenJitter: 0.34, radRatio: 0.5,
      segs: 4, wander: 0.14, gravitropism: -0.1, droop: 0.5, tipCurl: -0.05, taper: 0.9,
    },
    {
      // weeping twig streamers
      density: 6.0, whorl: 0, childStart: 0.3, childEnd: 1.0,
      angleBase: 0.7, angleTip: 0.45, lenRatio: 0.35, lenJitter: 0.4, radRatio: 0.5,
      segs: 3, wander: 0.12, gravitropism: -0.3, droop: 0.7, tipCurl: -0.05, taper: 0.85,
      planar: 0.5,
    },
  ],
  foliage: {
    kind: 'leafCluster',
    anchorLevel: 3,
    spacing: 0.11,
    tStart: 0.15,
    scale: [0.1, 0.16],
    tilt: 0.9,
    clusterSize: [2, 3],
    normalBend: 0.66,
    planarLeaves: true,
    leaf: { len: 1.0, width: 0.55, shapePow: 1.4, fold: 0.22, curl: 0.3, needleCount: 0, brush: 0 },
  },
  flare: { amp: 0.32, height: 0.7, lobes: 4 },
  barkLayer: 3,
  barkRepeats: 3,
  foliageColor: { r: 0.075, g: 0.15, b: 0.03, hueVar: 0.34 },
  brokenTop: 0,
  stubChance: 0.03,
};

export const KARST_GNARL: SpeciesParams = {
  id: 'karst',
  label: 'Karst gnarl (cliff broadleaf)',
  kind: 'broadleaf',
  height: [3.5, 6.5],
  trunkRadiusK: 0.045,
  crown: 'irregular',
  asym: 0.5,
  levels: [
    {
      density: 0, whorl: 0, childStart: 0, childEnd: 0,
      angleBase: 0, angleTip: 0, lenRatio: 0, lenJitter: 0, radRatio: 0,
      segs: 9, wander: 0.34, gravitropism: -0.05, droop: 0, tipCurl: 0.1, taper: 0.8,
    },
    {
      density: 2.6, whorl: 0, childStart: 0.15, childEnd: 0.95,
      angleBase: 1.35, angleTip: 0.7, lenRatio: 0.62, lenJitter: 0.45, radRatio: 0.55,
      segs: 7, wander: 0.3, gravitropism: 0.06, droop: 0.35, tipCurl: 0.18, taper: 0.8,
    },
    {
      density: 3.8, whorl: 0, childStart: 0.2, childEnd: 1.0,
      angleBase: 1.0, angleTip: 0.6, lenRatio: 0.42, lenJitter: 0.4, radRatio: 0.55,
      segs: 4, wander: 0.3, gravitropism: 0.05, droop: 0.25, tipCurl: 0.1, taper: 0.85,
    },
    {
      // gnarled twiglets carrying layered leaf plates
      density: 5.0, whorl: 0, childStart: 0.25, childEnd: 1.0,
      angleBase: 0.85, angleTip: 0.55, lenRatio: 0.4, lenJitter: 0.45, radRatio: 0.5,
      segs: 2, wander: 0.25, gravitropism: 0.04, droop: 0.2, tipCurl: 0.1, taper: 0.85,
      planar: 0.4,
    },
  ],
  foliage: {
    kind: 'leafCluster',
    anchorLevel: 3,
    spacing: 0.055,
    tStart: 0.12,
    scale: [0.11, 0.16],
    tilt: 0.9,
    clusterSize: [2, 4],
    normalBend: 0.66,
    planarLeaves: true,
    leaf: { len: 1.0, width: 0.5, shapePow: 1.2, fold: 0.3, curl: 0.24, needleCount: 0, brush: 0 },
  },
  flare: { amp: 0.9, height: 0.7, lobes: 6 },
  barkLayer: 4,
  barkRepeats: 3,
  foliageColor: { r: 0.05, g: 0.12, b: 0.04, hueVar: 0.24 },
  brokenTop: 0,
  stubChance: 0.1,
};

export const SNAG: SpeciesParams = {
  id: 'snag',
  label: 'Snag (dead standing)',
  kind: 'snag',
  height: [8, 15],
  trunkRadiusK: 0.022,
  crown: 'cone',
  asym: 0.3,
  levels: [
    {
      density: 0, whorl: 0, childStart: 0, childEnd: 0,
      angleBase: 0, angleTip: 0, lenRatio: 0, lenJitter: 0, radRatio: 0,
      segs: 13, wander: 0.06, gravitropism: 0.04, droop: 0, tipCurl: 0, taper: 0.9,
    },
    {
      density: 2.4, whorl: 0, childStart: 0.2, childEnd: 0.97,
      angleBase: 1.6, angleTip: 0.85, lenRatio: 0.38, lenJitter: 0.45, radRatio: 0.32,
      segs: 6, wander: 0.14, gravitropism: -0.1, droop: 0.6, tipCurl: 0.05, taper: 0.75,
    },
    {
      density: 1.8, whorl: 0, childStart: 0.2, childEnd: 1.0,
      angleBase: 1.1, angleTip: 0.7, lenRatio: 0.3, lenJitter: 0.5, radRatio: 0.4,
      segs: 3, wander: 0.2, gravitropism: -0.08, droop: 0.4, tipCurl: 0, taper: 0.7,
    },
  ],
  foliage: null,
  flare: { amp: 0.6, height: 0.9, lobes: 5 },
  barkLayer: 5,
  barkRepeats: 4,
  foliageColor: { r: 0.1, g: 0.09, b: 0.07, hueVar: 0.1 },
  brokenTop: 0.62,
  stubChance: 0.28,
};

// ---- #112 distinct species (research-grounded forms) -----------------------
//
// LARCH — European/Siberian larch (Larix, Estonian "lehis", planted-conifer stands).
// Deciduous conifer: the crown reads NOTHING like spruce — a straight leader with
// SPARSE, near-level primaries whose branchlets hang PENDULOUS, carrying soft LIGHT
// yellow-green needle tufts (rosettes on short shoots) in an OPEN, airy cone. Taller
// and slenderer than spruce. (Larix decidua: 25–45 m, greyish-pink fissured bark,
// needles 2–4 cm in fascicles of 30–65, mid/upper branches down-swept with ascending
// ends, side branches pendulous — Woodland Trust / Wikipedia / Morton Arboretum.)
export const LARCH: SpeciesParams = {
  id: 'larch',
  label: 'Larch (deciduous conifer)',
  kind: 'conifer',
  height: [22, 32],
  trunkRadiusK: 0.013,
  crown: 'cone',
  asym: 0.2,
  levels: [
    {
      // straight persistent leader, gentle wander
      density: 0, whorl: 0, childStart: 0, childEnd: 0,
      angleBase: 0, angleTip: 0, lenRatio: 0, lenJitter: 0, radRatio: 0,
      segs: 18, wander: 0.012, gravitropism: 0.06, droop: 0, tipCurl: 0, taper: 1.0,
    },
    {
      // primaries: SPARSE (open crown), near-horizontal spokes, sag then ascending tips
      density: 3.2, whorl: 0, childStart: 0.12, childEnd: 0.985,
      angleBase: 1.62, angleTip: 0.7, lenRatio: 0.22, lenJitter: 0.28, radRatio: 0.3,
      segs: 6, wander: 0.09, gravitropism: -0.02, droop: 0.34, tipCurl: 0.22, taper: 1.0,
    },
    {
      // branchlets: PENDULOUS side twigs hanging off the boughs (the larch signature),
      // carrying the needle tufts; strong negative gravitropism + droop. #112 VRAM
      // right-size: density 4.0→3.2 — a gentle thin of the pendulous branchlets (kept
      // dense enough to READ as the drooping larch signature); the crown-DAG cut is
      // carried mostly by needleCount + spacing below so the silhouette holds.
      density: 3.2, whorl: 0, childStart: 0.1, childEnd: 1.0,
      angleBase: 1.15, angleTip: 0.9, lenRatio: 0.3, lenJitter: 0.4, radRatio: 0.42,
      segs: 3, wander: 0.12, gravitropism: -0.16, droop: 0.62, tipCurl: 0.04, taper: 0.85,
      planar: 0.4,
    },
  ],
  foliage: {
    kind: 'needleSpray',
    anchorLevel: 2,
    // #112 VRAM right-size: spacing 0.2→0.28 (fewer sprays along each twig) —
    // reinforces the OPEN airy crown while cutting the per-crown spray count.
    spacing: 0.28, // wider than spruce (0.16) → open, airy crown
    tStart: 0.05,
    scale: [0.16, 0.26], // small soft tufts
    tilt: 0.6,
    clusterSize: [1, 1],
    normalBend: 0.6,
    planarLeaves: false, // rosette-ish tufts around the twig, not a flat comb
    // #112 VRAM right-size: needleCount 26→13 (hero ×3 = 39, vs spruce's 90). A larch
    // short-shoot rosette is a SPARSE soft tuft, not a dense spray — halving the needles
    // reads MORE correct (airy) and is the single biggest crown-DAG lever (each spray is
    // 2·needle tris). The tuft SHAPE/scale is unchanged, so the silhouette holds.
    leaf: { len: 0.075, width: 0.02, shapePow: 1, fold: 0, curl: 0, needleCount: 13, brush: 0.7 },
  },
  flare: { amp: 0.42, height: 0.9, lobes: 5 },
  barkLayer: 0, // reuse spruce bark (grey-brown vertical fissures ≈ larch) — no new VRAM
  barkRepeats: 5,
  foliageColor: { r: 0.09, g: 0.175, b: 0.05, hueVar: 0.3 }, // light fresh yellow-green
  brokenTop: 0,
  stubChance: 0.03,
};

// OAK — pedunculate/English oak (Quercus robur, Estonian "tamm", native hemiboreal
// broadleaf). The classic BROAD, spreading, rounded crown: a short STOUT bole, then
// FEW HEAVY primaries at wide angles that run out into long low limbs, an unevenly
// domed rounded crown of deep dull-green leaf clusters. Reads distinct from beech
// (upright ellipsoid) and birch (slim column) by breadth + stoutness + heavy low
// limbs. (Quercus robur: 20–40 m and ~equally wide, massive lower branches, greyish-
// brown ridged/fissured bark — Woodland Trust / Wikipedia.)
export const OAK: SpeciesParams = {
  id: 'oak',
  label: 'Oak (broad broadleaf)',
  kind: 'broadleaf',
  height: [14, 22],
  trunkRadiusK: 0.032, // stout bole
  crown: 'round',
  asym: 0.34,
  levels: [
    {
      // short, stout, strongly-tapered bole
      density: 0, whorl: 0, childStart: 0, childEnd: 0,
      angleBase: 0, angleTip: 0, lenRatio: 0, lenJitter: 0, radRatio: 0,
      segs: 9, wander: 0.07, gravitropism: 0.03, droop: 0, tipCurl: 0, taper: 1.35,
    },
    {
      // primaries: FEW, HEAVY, wide-angled, long — spreading limbs from low on the bole
      density: 1.25, whorl: 0, childStart: 0.24, childEnd: 0.9,
      angleBase: 1.24, angleTip: 0.72, lenRatio: 0.62, lenJitter: 0.34, radRatio: 0.6,
      segs: 8, wander: 0.16, gravitropism: 0.02, droop: 0.2, tipCurl: 0.14, taper: 0.95,
    },
    {
      // secondaries: sinuous, spreading, filling the broad dome
      density: 2.2, whorl: 0, childStart: 0.2, childEnd: 0.98,
      angleBase: 1.0, angleTip: 0.6, lenRatio: 0.5, lenJitter: 0.36, radRatio: 0.55,
      segs: 5, wander: 0.2, gravitropism: 0.02, droop: 0.22, tipCurl: 0.08, taper: 0.9,
    },
    {
      // twig plates carrying the leaf clusters. #112 VRAM right-size: density 7.0→4.8 —
      // oak's measured BARK DAG was 2× beech's (dense twig tubes at barkK 0.7). Fewer
      // twig plates cut the level-3 bark tubes hard; the broad DOME is carried by the
      // 'round' crown envelope + the heavy primaries/secondaries, not the twig count.
      density: 4.8, whorl: 0, childStart: 0.15, childEnd: 1.0,
      angleBase: 0.95, angleTip: 0.62, lenRatio: 0.3, lenJitter: 0.38, radRatio: 0.55,
      segs: 3, wander: 0.16, gravitropism: 0, droop: 0.16, tipCurl: 0.05, taper: 0.85,
      planar: 0.3,
    },
  ],
  foliage: {
    kind: 'leafCluster',
    anchorLevel: 3,
    spacing: 0.15,
    tStart: 0.1,
    scale: [0.15, 0.22],
    tilt: 0.95,
    // #112 VRAM right-size: clusterSize [2,4]→[2,3] (avg leaves 3→2.5, matching beech).
    // The crown mesh caps at ~4000 anchors, so this per-anchor leaf count is the crown-
    // DAG lever; the broad rounded dome (envelope + big scale leaves) is unchanged.
    clusterSize: [2, 3],
    normalBend: 0.7,
    planarLeaves: true,
    leaf: { len: 1.0, width: 0.46, shapePow: 1.1, fold: 0.26, curl: 0.2, needleCount: 0, brush: 0 },
  },
  flare: { amp: 0.8, height: 1.3, lobes: 7 }, // buttressed stout base
  barkLayer: 4, // reuse karst deep-ridged bark (≈ oak rugged fissures) — no new VRAM
  barkRepeats: 4,
  foliageColor: { r: 0.05, g: 0.12, b: 0.035, hueVar: 0.26 }, // deep dull green
  brokenTop: 0,
  stubChance: 0.03,
};

// ---- species batch-1: three prevalent Estonian broadleaves (research-grounded) ----
//
// ASPEN — European aspen (Populus tremula, Estonian "haab"; 6.5% of forest area).
// A large pioneer to ~25 m on a STRAIGHT slender bole, carrying a broad, rounded/oval
// UPRIGHT crown of small ROUND leaves. Reads distinct from birch precisely because it
// does NOT weep — branches ASCEND (positive gravitropism, low droop), where birch's
// twigs stream down (droop 0.4–0.7, negative gravitropism). Smooth pale greenish-grey
// bark (reuse beech layer). The famous flutter is a leaf-petiole trait, evoked here by a
// HIGH foliage hueVar (shimmer/autumn-gold variance across the crown), not weeping twigs.
// (Woodland Trust / Wikipedia / EUFORGEN Populus tremula.)
export const ASPEN: SpeciesParams = {
  id: 'aspen',
  label: 'Aspen (broadleaf)',
  kind: 'broadleaf',
  height: [15, 24],
  trunkRadiusK: 0.016, // slender straight bole
  crown: 'ellipsoid',
  asym: 0.26,
  levels: [
    {
      // tall, straight, clean pale bole — minimal wander
      density: 0, whorl: 0, childStart: 0, childEnd: 0,
      angleBase: 0, angleTip: 0, lenRatio: 0, lenJitter: 0, radRatio: 0,
      segs: 11, wander: 0.04, gravitropism: 0.05, droop: 0, tipCurl: 0, taper: 1.1,
    },
    {
      // primaries: fewer + shorter than beech, ASCENDING (aspen does not weep)
      density: 1.3, whorl: 0, childStart: 0.3, childEnd: 0.95,
      angleBase: 1.0, angleTip: 0.5, lenRatio: 0.45, lenJitter: 0.28, radRatio: 0.46,
      segs: 7, wander: 0.1, gravitropism: 0.09, droop: 0.16, tipCurl: 0.14, taper: 0.95,
    },
    {
      // secondaries: sparse, ascending
      density: 1.7, whorl: 0, childStart: 0.25, childEnd: 0.97,
      angleBase: 0.9, angleTip: 0.55, lenRatio: 0.42, lenJitter: 0.32, radRatio: 0.5,
      segs: 4, wander: 0.12, gravitropism: 0.06, droop: 0.14, tipCurl: 0.1, taper: 0.9,
    },
    {
      // slender shiny twigs — airier than beech (density 5.5 vs 8) but FULL; low droop +
      // neutral gravitropism = NO weep (the birch contrast). A light, open aspen crown.
      density: 5.5, whorl: 0, childStart: 0.18, childEnd: 1.0,
      angleBase: 0.85, angleTip: 0.55, lenRatio: 0.3, lenJitter: 0.35, radRatio: 0.52,
      segs: 3, wander: 0.1, gravitropism: 0.0, droop: 0.1, tipCurl: 0.05, taper: 0.85,
      planar: 0.6,
    },
  ],
  foliage: {
    kind: 'leafCluster',
    anchorLevel: 3,
    spacing: 0.12, // beech-parity anchor density → a full aspen canopy
    tStart: 0.1,
    scale: [0.12, 0.18],
    tilt: 0.95,
    clusterSize: [2, 3], // beech/oak-parity lush clusters (airiness = twig density, not thinning)
    normalBend: 0.7,
    planarLeaves: true,
    // ROUND blade: width ≈ len, shapePow 1 (rounded, not pointed) — the aspen leaf.
    // width 0.76 is still clearly round (vs beech 0.42) but trims voxel-crown occupancy.
    leaf: { len: 1.0, width: 0.76, shapePow: 1.0, fold: 0.14, curl: 0.12, needleCount: 0, brush: 0 },
  },
  flare: { amp: 0.4, height: 0.9, lobes: 5 },
  barkLayer: 2, // beech smooth pale ≈ aspen greenish-grey — no new bark VRAM
  barkRepeats: 4,
  foliageColor: { r: 0.07, g: 0.152, b: 0.042, hueVar: 0.34 }, // fresh mid green, high hueVar = flutter shimmer
  brokenTop: 0,
  stubChance: 0.03,
};

// GREY ALDER — Alnus incana (Estonian "hall lepp"; 9.8% — the largest unmodeled species).
// A fast wet-ground / abandoned-farmland pioneer, SMALL-to-MEDIUM (15–20 m, here 10–18 m),
// often multi-stemmed/suckering ("scrappy"), with an OPEN, rounded-to-ovoid crown on a
// relatively short bole; ascending branches. Bark stays SMOOTH GREY even old (beech layer).
// Leaves ovate, dull, grey-green with a pale underside → a DULLER, greyer, lower-saturation
// crown than aspen; airier (open) silhouette. Reads as low scrubby wet forest, NOT cathedral
// beech. (Trees&Shrubs Online / Wikipedia / EUFORGEN Alnus incana.)
export const GREY_ALDER: SpeciesParams = {
  id: 'greyAlder',
  label: 'Grey alder (broadleaf)',
  kind: 'broadleaf',
  height: [10, 18],
  trunkRadiusK: 0.02,
  crown: 'ellipsoid',
  asym: 0.32, // open, slightly irregular multi-stem feel
  levels: [
    {
      // short bole with a touch of scrappy wander
      density: 0, whorl: 0, childStart: 0, childEnd: 0,
      angleBase: 0, angleTip: 0, lenRatio: 0, lenJitter: 0, radRatio: 0,
      segs: 8, wander: 0.07, gravitropism: 0.05, droop: 0, tipCurl: 0, taper: 1.1,
    },
    {
      // open ascending primaries — fewer/shorter, scrappier than aspen
      density: 1.2, whorl: 0, childStart: 0.24, childEnd: 0.95,
      angleBase: 1.05, angleTip: 0.55, lenRatio: 0.5, lenJitter: 0.34, radRatio: 0.48,
      segs: 6, wander: 0.14, gravitropism: 0.07, droop: 0.15, tipCurl: 0.1, taper: 0.9,
    },
    {
      density: 1.6, whorl: 0, childStart: 0.2, childEnd: 0.98,
      angleBase: 0.95, angleTip: 0.55, lenRatio: 0.44, lenJitter: 0.36, radRatio: 0.5,
      segs: 4, wander: 0.16, gravitropism: 0.05, droop: 0.14, tipCurl: 0.08, taper: 0.88,
    },
    {
      // airiest of the three broadleaves — the open scrappy wet-ground pioneer crown,
      // but still FULL (density 5.0, botanically airier than beech's 8, not thinned).
      density: 5.0, whorl: 0, childStart: 0.2, childEnd: 1.0,
      angleBase: 0.85, angleTip: 0.55, lenRatio: 0.3, lenJitter: 0.4, radRatio: 0.5,
      segs: 3, wander: 0.14, gravitropism: 0.02, droop: 0.12, tipCurl: 0.05, taper: 0.85,
      planar: 0.4,
    },
  ],
  foliage: {
    kind: 'leafCluster',
    anchorLevel: 3,
    spacing: 0.13, // beech-parity anchor density → a full (if scrappy) alder crown
    tStart: 0.1,
    scale: [0.12, 0.17],
    tilt: 0.9,
    clusterSize: [2, 3], // lush clusters; the open scrappy read comes from twig density + asym, not thinning
    normalBend: 0.66,
    planarLeaves: true,
    leaf: { len: 1.0, width: 0.54, shapePow: 1.1, fold: 0.2, curl: 0.15, needleCount: 0, brush: 0 },
  },
  flare: { amp: 0.4, height: 0.7, lobes: 4 }, // short suckering base
  barkLayer: 2, // smooth pale grey even in old age — beech layer, no new VRAM
  barkRepeats: 4,
  foliageColor: { r: 0.082, g: 0.12, b: 0.062, hueVar: 0.26 }, // dull grey-green, low saturation
  brokenTop: 0,
  stubChance: 0.04,
};

// BLACK ALDER — Alnus glutinosa (Estonian "sanglepp"; 4.2%, the wet-site / riparian alder).
// To 20–30 m. SIGNATURE: young trees hold an upright habit with a persistent axial stem —
// a broad conical crown / narrow broadleaf SPIRE (older trees arch, but the young spire is
// the read). It is the ONLY broadleaf modeled on a conifer-style 'cone' envelope, so it is
// instantly distinct — massed in wet hollows near water. Bark dark grey and fissured old
// (reuse spruce layer). Leaves obovate/racquet, blunt tip, dark leathery green.
// (Woodland Trust / Wikipedia / NCSU Alnus glutinosa.)
export const BLACK_ALDER: SpeciesParams = {
  id: 'blackAlder',
  label: 'Black alder (broadleaf)',
  kind: 'broadleaf',
  height: [16, 26],
  trunkRadiusK: 0.022,
  crown: 'cone', // the only broadleaf on a conic envelope — a narrow upright spire
  asym: 0.28,
  levels: [
    {
      // persistent straight central leader (spruce/larch-style bole → the spire)
      density: 0, whorl: 0, childStart: 0, childEnd: 0,
      angleBase: 0, angleTip: 0, lenRatio: 0, lenJitter: 0, radRatio: 0,
      segs: 16, wander: 0.02, gravitropism: 0.06, droop: 0, tipCurl: 0, taper: 1.05,
    },
    {
      // short ascending primaries — the cone envelope shortens them toward the leader
      // so the crown reads as a narrow cone, not a broad dome
      density: 2.0, whorl: 0, childStart: 0.12, childEnd: 0.97,
      angleBase: 1.35, angleTip: 0.6, lenRatio: 0.32, lenJitter: 0.3, radRatio: 0.34,
      segs: 6, wander: 0.1, gravitropism: 0.04, droop: 0.18, tipCurl: 0.15, taper: 0.92,
    },
    {
      density: 2.0, whorl: 0, childStart: 0.15, childEnd: 1.0,
      angleBase: 1.0, angleTip: 0.6, lenRatio: 0.36, lenJitter: 0.34, radRatio: 0.46,
      segs: 4, wander: 0.12, gravitropism: 0.03, droop: 0.16, tipCurl: 0.1, taper: 0.88,
    },
    {
      // narrow-spire twigs — FULL within the tight cone envelope (density 5.5); the cone
      // shape reads as a spire, the crown itself is leafy.
      density: 5.5, whorl: 0, childStart: 0.15, childEnd: 1.0,
      angleBase: 0.9, angleTip: 0.6, lenRatio: 0.3, lenJitter: 0.36, radRatio: 0.5,
      segs: 3, wander: 0.11, gravitropism: 0.02, droop: 0.12, tipCurl: 0.05, taper: 0.85,
      planar: 0.4,
    },
  ],
  foliage: {
    kind: 'leafCluster',
    anchorLevel: 3,
    spacing: 0.13, // beech-parity anchor density → a leafy spire (the cone shape, not sparse leaves)
    tStart: 0.08,
    scale: [0.13, 0.18],
    tilt: 0.9,
    clusterSize: [2, 3], // lush clusters; the narrow spire read comes from the cone envelope, not thinning
    normalBend: 0.66,
    planarLeaves: true,
    // obovate/racquet, blunt tip → rounded (shapePow 1), medium width
    leaf: { len: 1.0, width: 0.56, shapePow: 1.0, fold: 0.18, curl: 0.12, needleCount: 0, brush: 0 },
  },
  flare: { amp: 0.5, height: 0.9, lobes: 5 }, // basal adventitious prop-root flare
  barkLayer: 0, // spruce narrow vertical grey-brown fissures ≈ dark fissured alder — no new VRAM
  barkRepeats: 5,
  foliageColor: { r: 0.045, g: 0.105, b: 0.038, hueVar: 0.2 }, // dark leathery green
  brokenTop: 0,
  stubChance: 0.03,
};

// ---- species batch-2: five Estonian accent broadleaves (research-grounded) --------
//
// ASH — European ash (Fraxinus excelsior, Estonian "saar", code SA). A TALL tree
// (typically 18–28 m, to 43 m) with a distinctly OPEN, AIRY, light-passing crown —
// stout greenish-grey shoots carrying UPSWEPT primaries with hooked (up-curled) tips,
// and PINNATE compound leaves (7–13 narrow serrated leaflets) that read as feathery.
// The signature vs its lush neighbours is AIRINESS by twig DENSITY (5.0 — airier than
// beech's 8, below birch's 6), NOT thinned foliage: each twig is still fully clustered.
// Bark: smooth pale grey when young → finely VERTICALLY fissured old (reuse spruce
// grey-brown vertical-fissure layer). (Woodland Trust / Wikipedia Fraxinus excelsior.)
export const ASH: SpeciesParams = {
  id: 'ash',
  label: 'Ash (broadleaf)',
  kind: 'broadleaf',
  height: [15, 24], // tall for the batch, within the engine's compressed broadleaf band
  trunkRadiusK: 0.018, // straight, moderately slender bole
  crown: 'ellipsoid', // tall airy — narrowed via short primaries + low twig density
  asym: 0.24,
  levels: [
    {
      // tall straight bole
      density: 0, whorl: 0, childStart: 0, childEnd: 0,
      angleBase: 0, angleTip: 0, lenRatio: 0, lenJitter: 0, radRatio: 0,
      segs: 11, wander: 0.04, gravitropism: 0.05, droop: 0, tipCurl: 0, taper: 1.1,
    },
    {
      // primaries: UPSWEPT with hooked tips (the ash signature) — ascending, open
      density: 1.3, whorl: 0, childStart: 0.28, childEnd: 0.95,
      angleBase: 0.95, angleTip: 0.5, lenRatio: 0.5, lenJitter: 0.28, radRatio: 0.44,
      segs: 7, wander: 0.1, gravitropism: 0.09, droop: 0.14, tipCurl: 0.2, taper: 0.95,
    },
    {
      // secondaries: sparse, ascending, hook-tipped
      density: 1.6, whorl: 0, childStart: 0.22, childEnd: 0.97,
      angleBase: 0.9, angleTip: 0.55, lenRatio: 0.44, lenJitter: 0.32, radRatio: 0.48,
      segs: 4, wander: 0.12, gravitropism: 0.07, droop: 0.12, tipCurl: 0.14, taper: 0.9,
    },
    {
      // airiest twig level of the batch — an OPEN light-passing crown, but still FULL
      // (density 5.0, below birch's 6; airiness is twig SPACING not thinned clusters).
      density: 5.0, whorl: 0, childStart: 0.18, childEnd: 1.0,
      angleBase: 0.85, angleTip: 0.55, lenRatio: 0.3, lenJitter: 0.35, radRatio: 0.5,
      segs: 3, wander: 0.1, gravitropism: 0.03, droop: 0.1, tipCurl: 0.08, taper: 0.85,
      planar: 0.5,
    },
  ],
  foliage: {
    kind: 'leafCluster',
    anchorLevel: 3,
    spacing: 0.13, // beech-parity anchor density → a full (if airy) ash canopy
    tStart: 0.1,
    scale: [0.14, 0.2],
    tilt: 0.95,
    clusterSize: [2, 3], // lush clusters; the open read comes from twig density, not thinning
    normalBend: 0.7,
    planarLeaves: true,
    // narrow pointed leaflet blade (pinnate leaf read): shapePow 1.2 pointier than beech
    leaf: { len: 1.0, width: 0.4, shapePow: 1.2, fold: 0.2, curl: 0.15, needleCount: 0, brush: 0 },
  },
  flare: { amp: 0.45, height: 1.0, lobes: 5 },
  barkLayer: 0, // spruce grey-brown vertical fissures ≈ mature ash's fine vertical fissuring
  barkRepeats: 4,
  foliageColor: { r: 0.07, g: 0.155, b: 0.045, hueVar: 0.28 }, // fresh light green
  brokenTop: 0,
  stubChance: 0.03,
};

// MAPLE — Norway maple (Acer platanoides, Estonian "vaher", code VA). A medium-tall
// tree (16–26 m; wild to 30 m) with a DENSE, ROUNDED, SYMMETRICAL crown — the fuller,
// more regular dome the reference calls out. MANY regular ascending-then-spreading
// primaries + a DENSE twig level (7.0, near beech's 8) fill a tight symmetric dome of
// big palmate 5-lobe leaves. Reads distinct from OAK (also 'round') by LOW asym (0.18
// vs oak 0.34), denser regular branching, and a slenderer bole — a neat symmetric dome
// vs oak's gnarled heavy-low-limb spread. Bark grey-brown, shallowly grooved (reuse
// spruce grooved layer). (Wikipedia Acer platanoides.)
export const MAPLE: SpeciesParams = {
  id: 'maple',
  label: 'Maple (broadleaf)',
  kind: 'broadleaf',
  height: [15, 24], // within the engine's compressed broadleaf band (voxel-crown volume)
  trunkRadiusK: 0.026,
  crown: 'round', // broad rounded — a DENSE symmetric dome (distinct from oak via grammar)
  asym: 0.18, // symmetric (the diagnostic vs oak's asymmetric spread)
  levels: [
    {
      density: 0, whorl: 0, childStart: 0, childEnd: 0,
      angleBase: 0, angleTip: 0, lenRatio: 0, lenJitter: 0, radRatio: 0,
      segs: 9, wander: 0.05, gravitropism: 0.04, droop: 0, tipCurl: 0, taper: 1.2,
    },
    {
      // MANY regular ascending-then-spreading primaries (denser than oak's few heavy
      // limbs) → the fuller, more even dome
      density: 1.6, whorl: 0, childStart: 0.24, childEnd: 0.92,
      angleBase: 1.1, angleTip: 0.6, lenRatio: 0.55, lenJitter: 0.28, radRatio: 0.5,
      segs: 8, wander: 0.1, gravitropism: 0.05, droop: 0.2, tipCurl: 0.12, taper: 0.95,
    },
    {
      density: 2.4, whorl: 0, childStart: 0.2, childEnd: 0.98,
      angleBase: 0.95, angleTip: 0.58, lenRatio: 0.46, lenJitter: 0.3, radRatio: 0.52,
      segs: 5, wander: 0.12, gravitropism: 0.04, droop: 0.2, tipCurl: 0.08, taper: 0.9,
    },
    {
      // DENSE twig level (7.0, near beech's 8) → the fullest crown of the batch
      density: 7.0, whorl: 0, childStart: 0.15, childEnd: 1.0,
      angleBase: 0.9, angleTip: 0.6, lenRatio: 0.28, lenJitter: 0.35, radRatio: 0.55,
      segs: 3, wander: 0.1, gravitropism: 0.0, droop: 0.14, tipCurl: 0.05, taper: 0.85,
      planar: 0.35,
    },
  ],
  foliage: {
    kind: 'leafCluster',
    anchorLevel: 3,
    spacing: 0.12, // beech-parity anchor density → a dense maple dome
    tStart: 0.1,
    scale: [0.17, 0.24], // large palmate leaves
    tilt: 0.95,
    clusterSize: [2, 3],
    normalBend: 0.7,
    planarLeaves: true,
    // broad palmate blade: width ~0.62 (wide), shapePow 1.0 (rounded lobed outline)
    leaf: { len: 1.0, width: 0.62, shapePow: 1.0, fold: 0.15, curl: 0.12, needleCount: 0, brush: 0 },
  },
  flare: { amp: 0.55, height: 1.1, lobes: 6 },
  barkLayer: 0, // spruce grey-brown grooves ≈ Norway maple's shallowly grooved grey-brown bark
  barkRepeats: 4,
  foliageColor: { r: 0.055, g: 0.14, b: 0.04, hueVar: 0.26 }, // rich mid green
  brokenTop: 0,
  stubChance: 0.03,
};

// LIME — small-leaved lime (Tilia cordata, Estonian "pärn", code PN). A TALL tree
// (18–28 m; to 40 m) with a BROAD, DENSE, rounded/oval-to-domed crown — upright
// branching that "increases in density with age" (Wikipedia) over a clear bole, so it
// reads as a big dense shade dome. DENSE twigs (7.0) carry heart-shaped (cordate) leaves.
// Distinct from MAPLE by envelope ('dome' — a tall high dome that clears a bole with age
// vs maple's broad-round low-full crown) and greater height, and from ASH by density
// (dense vs airy). Bark smooth grey-brown young (reuse beech smooth-grey layer).
// (Wikipedia / Woodland Trust Tilia cordata.)
export const LIME: SpeciesParams = {
  id: 'lime',
  label: 'Lime (broadleaf)',
  kind: 'broadleaf',
  height: [16, 25], // the batch's tallest, just under blackAlder's 26 (compressed band)
  trunkRadiusK: 0.024,
  crown: 'dome', // broad domed — a tall dense dome over a clear bole (dome ontogeny)
  asym: 0.22,
  levels: [
    {
      density: 0, whorl: 0, childStart: 0, childEnd: 0,
      angleBase: 0, angleTip: 0, lenRatio: 0, lenJitter: 0, radRatio: 0,
      segs: 10, wander: 0.045, gravitropism: 0.05, droop: 0, tipCurl: 0, taper: 1.15,
    },
    {
      // upright ascending primaries — lime's dense upright branching
      density: 1.5, whorl: 0, childStart: 0.26, childEnd: 0.95,
      angleBase: 1.0, angleTip: 0.55, lenRatio: 0.52, lenJitter: 0.28, radRatio: 0.48,
      segs: 8, wander: 0.1, gravitropism: 0.07, droop: 0.18, tipCurl: 0.12, taper: 0.95,
    },
    {
      density: 2.3, whorl: 0, childStart: 0.2, childEnd: 0.98,
      angleBase: 0.92, angleTip: 0.56, lenRatio: 0.46, lenJitter: 0.3, radRatio: 0.5,
      segs: 5, wander: 0.12, gravitropism: 0.05, droop: 0.18, tipCurl: 0.08, taper: 0.9,
    },
    {
      // DENSE twig level (7.0) → a full lime dome
      density: 7.0, whorl: 0, childStart: 0.15, childEnd: 1.0,
      angleBase: 0.9, angleTip: 0.6, lenRatio: 0.28, lenJitter: 0.35, radRatio: 0.55,
      segs: 3, wander: 0.1, gravitropism: 0.02, droop: 0.12, tipCurl: 0.05, taper: 0.85,
      planar: 0.35,
    },
  ],
  foliage: {
    kind: 'leafCluster',
    anchorLevel: 3,
    spacing: 0.12, // beech-parity anchor density → a dense lime canopy
    tStart: 0.1,
    scale: [0.15, 0.22],
    tilt: 0.95,
    clusterSize: [2, 3],
    normalBend: 0.7,
    planarLeaves: true,
    // heart-shaped (cordate) blade: width ~0.66 (broad), shapePow 1.0 (rounded, pointed tip)
    leaf: { len: 1.0, width: 0.66, shapePow: 1.0, fold: 0.16, curl: 0.12, needleCount: 0, brush: 0 },
  },
  flare: { amp: 0.55, height: 1.1, lobes: 6 },
  barkLayer: 2, // beech smooth grey ≈ lime's smooth grey-brown young bark — no new VRAM
  barkRepeats: 4,
  foliageColor: { r: 0.08, g: 0.16, b: 0.045, hueVar: 0.28 }, // bright fresh lime-green
  brokenTop: 0,
  stubChance: 0.03,
};

// WILLOW — willow (Salix, Estonian "remmelgas", code RE; goat/white willow, wet edge).
// A riparian small-to-medium tree (10–18 m) of lake/stream margins with a BROAD,
// IRREGULAR, often-leaning crown and DROOPING/pendulous shoots — a GENTLE weep, quite
// unlike birch: birch is a slim WEEPING COLUMN (crown 'column', strong negative
// gravitropism −0.3, droop 0.7); willow is a BROAD 'irregular' crown with a mild droop
// (gravitropism −0.08, droop 0.52) over NARROW lanceolate SILVERY leaves (silky pale
// undersides). Bark grey-brown, deeply/diamond fissured (reuse spruce fissured layer).
// (Wikipedia Salix alba/caprea; Woodland Trust goat willow — wet woodland/lakes/streams.)
export const WILLOW: SpeciesParams = {
  id: 'willow',
  label: 'Willow (broadleaf)',
  kind: 'broadleaf',
  height: [10, 18],
  trunkRadiusK: 0.02,
  crown: 'irregular', // broad, asymmetric, leaning riparian crown
  asym: 0.42, // strongly irregular
  levels: [
    {
      // short sinuous/leaning bole
      density: 0, whorl: 0, childStart: 0, childEnd: 0,
      angleBase: 0, angleTip: 0, lenRatio: 0, lenJitter: 0, radRatio: 0,
      segs: 10, wander: 0.09, gravitropism: 0.03, droop: 0, tipCurl: 0, taper: 1.05,
    },
    {
      // spreading primaries, wide-angled, beginning to arch
      density: 1.4, whorl: 0, childStart: 0.18, childEnd: 0.95,
      angleBase: 1.2, angleTip: 0.65, lenRatio: 0.55, lenJitter: 0.4, radRatio: 0.5,
      segs: 7, wander: 0.18, gravitropism: 0.0, droop: 0.3, tipCurl: 0.05, taper: 0.9,
    },
    {
      density: 1.9, whorl: 0, childStart: 0.2, childEnd: 0.98,
      angleBase: 1.0, angleTip: 0.6, lenRatio: 0.46, lenJitter: 0.42, radRatio: 0.5,
      segs: 4, wander: 0.2, gravitropism: -0.04, droop: 0.4, tipCurl: -0.02, taper: 0.88,
    },
    {
      // DROOPING shoots — a GENTLE weep (mild negative gravitropism, NOT birch's −0.3),
      // FULL (density 5.5, birch-parity). The broad irregular crown + gentle droop +
      // narrow silvery leaves reads as willow, not the slim weeping birch column.
      density: 5.5, whorl: 0, childStart: 0.2, childEnd: 1.0,
      angleBase: 0.8, angleTip: 0.5, lenRatio: 0.38, lenJitter: 0.4, radRatio: 0.5,
      segs: 3, wander: 0.16, gravitropism: -0.08, droop: 0.52, tipCurl: -0.03, taper: 0.85,
      planar: 0.4,
    },
  ],
  foliage: {
    kind: 'leafCluster',
    anchorLevel: 3,
    spacing: 0.12, // beech/birch-parity anchor density → a full drooping willow crown
    tStart: 0.12,
    scale: [0.1, 0.16], // small narrow leaves, densely borne
    tilt: 0.9,
    clusterSize: [2, 3],
    normalBend: 0.66,
    planarLeaves: true,
    // NARROW lanceolate blade: width 0.28 (narrow), shapePow 1.4 (pointed) — the willow leaf
    leaf: { len: 1.0, width: 0.28, shapePow: 1.4, fold: 0.15, curl: 0.2, needleCount: 0, brush: 0 },
  },
  flare: { amp: 0.5, height: 0.8, lobes: 5 },
  barkLayer: 0, // spruce deep vertical fissures ≈ white willow's deeply fissured grey-brown bark
  barkRepeats: 4,
  foliageColor: { r: 0.09, g: 0.135, b: 0.08, hueVar: 0.24 }, // silvery grey-green (pale undersides)
  brokenTop: 0,
  stubChance: 0.03,
};

// ROWAN — rowan / mountain-ash (Sorbus aucuparia, Estonian "pihlakas", code PI). A SMALL
// tree (6–12 m; to 15 m) — the shortest of the batch by far, its stature alone reads
// distinct. Slender smooth SILVERY-GREY bole; an OPEN, fine-textured crown, narrow when
// young → broad ovoid with age, on UPSWEPT branches; PINNATE leaves (5–8 leaflet pairs +
// terminal) give a light feathery texture. Airier twig level (5.0) keeps it open but FULL.
// Bark smooth silvery-grey (reuse beech smooth-grey layer). (Woodland Trust Sorbus aucuparia.)
export const ROWAN: SpeciesParams = {
  id: 'rowan',
  label: 'Rowan (broadleaf)',
  kind: 'broadleaf',
  height: [6, 12], // small tree — the batch's shortest
  trunkRadiusK: 0.017, // slender bole
  crown: 'ellipsoid', // narrow-young → broad-ovoid-old (ellipsoid ontogeny)
  asym: 0.3,
  levels: [
    {
      density: 0, whorl: 0, childStart: 0, childEnd: 0,
      angleBase: 0, angleTip: 0, lenRatio: 0, lenJitter: 0, radRatio: 0,
      segs: 9, wander: 0.06, gravitropism: 0.05, droop: 0, tipCurl: 0, taper: 1.05,
    },
    {
      // UPSWEPT open primaries
      density: 1.3, whorl: 0, childStart: 0.22, childEnd: 0.95,
      angleBase: 0.9, angleTip: 0.5, lenRatio: 0.48, lenJitter: 0.3, radRatio: 0.44,
      segs: 6, wander: 0.12, gravitropism: 0.1, droop: 0.12, tipCurl: 0.18, taper: 0.92,
    },
    {
      density: 1.6, whorl: 0, childStart: 0.2, childEnd: 0.97,
      angleBase: 0.85, angleTip: 0.55, lenRatio: 0.42, lenJitter: 0.34, radRatio: 0.48,
      segs: 4, wander: 0.14, gravitropism: 0.07, droop: 0.12, tipCurl: 0.12, taper: 0.9,
    },
    {
      // open fine twig level — FULL (density 5.0) but airy, for the fine-textured read
      density: 5.0, whorl: 0, childStart: 0.2, childEnd: 1.0,
      angleBase: 0.8, angleTip: 0.55, lenRatio: 0.3, lenJitter: 0.38, radRatio: 0.5,
      segs: 3, wander: 0.12, gravitropism: 0.03, droop: 0.1, tipCurl: 0.06, taper: 0.85,
      planar: 0.45,
    },
  ],
  foliage: {
    kind: 'leafCluster',
    anchorLevel: 3,
    spacing: 0.12, // beech-parity anchor density → a full (if small/fine) rowan crown
    tStart: 0.1,
    scale: [0.12, 0.18], // small fine leaves
    tilt: 0.9,
    clusterSize: [2, 3],
    normalBend: 0.68,
    planarLeaves: true,
    // narrow-oval leaflet blade (pinnate leaf read): shapePow 1.15
    leaf: { len: 1.0, width: 0.42, shapePow: 1.15, fold: 0.2, curl: 0.15, needleCount: 0, brush: 0 },
  },
  flare: { amp: 0.35, height: 0.7, lobes: 4 },
  barkLayer: 2, // beech smooth pale grey ≈ rowan's smooth silvery-grey bark — no new VRAM
  barkRepeats: 3,
  foliageColor: { r: 0.06, g: 0.135, b: 0.04, hueVar: 0.3 }, // mid green
  brokenTop: 0,
  stubChance: 0.03,
};

export const TREE_SPECIES: readonly SpeciesParams[] = [
  SPRUCE,
  PINE,
  BEECH,
  BIRCH,
  KARST_GNARL,
  SNAG,
  LARCH, // cls 6 (VegClass.Larch)
  OAK, // cls 7 (VegClass.Oak)
  ASPEN, // cls 8 (VegClass.Aspen)
  GREY_ALDER, // cls 9 (VegClass.GreyAlder)
  BLACK_ALDER, // cls 10 (VegClass.BlackAlder)
  ASH, // cls 11 (VegClass.Ash)
  MAPLE, // cls 12 (VegClass.Maple)
  LIME, // cls 13 (VegClass.Lime)
  WILLOW, // cls 14 (VegClass.Willow)
  ROWAN, // cls 15 (VegClass.Rowan)
];
