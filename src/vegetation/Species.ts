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
    card: { mode: 'lying', sizeK: 2.6 },
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
    card: { mode: 'cross', sizeK: 2.2 },
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
    card: { mode: 'cross', sizeK: 2.3 },
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
    card: { mode: 'cross', sizeK: 2.3 },
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
    card: { mode: 'cross', sizeK: 2.2 },
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

export const TREE_SPECIES: readonly SpeciesParams[] = [
  SPRUCE,
  PINE,
  BEECH,
  BIRCH,
  KARST_GNARL,
  SNAG,
];
