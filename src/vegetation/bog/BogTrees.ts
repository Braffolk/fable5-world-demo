/**
 * Bog trees — stunted raba Scots pine + scrubby bog downy birch (TREE path).
 *
 * Two distinct BOG GROWTH FORMS as their own SpeciesParams, fed to `buildTree`
 * (Species.ts / TreeBuilder.ts precedent: KARST_GNARL is a short gnarled cliff
 * pine defined the same way alongside PINE). The engine's age-form/ontogeny axis
 * is a MATURITY knob (juvenile→veteran) and cannot express a dwarf BOG HABIT, so
 * — exactly like KARST_GNARL — the habit lives in a dedicated species.
 *
 * Real reference (grounded before modelling):
 *  - Raba/bog Scots pine (Pinus sylvestris f. litwinowii): peat + high water table
 *    + no nutrients → SLOW, STUNTED, GNARLED, twisted-branched, widely-scattered
 *    trees, often centuries old at 1.5–4 m. Thick-for-its-height twisted bole,
 *    bare lower trunk, a SPARSE, OPEN, flattened/candelabra crown you can see
 *    through — NOT the tall conical forest pine. Short blue-green needle tufts on
 *    stubby shoots. (Trees for Life bog-woodland; USFS FEIS Pinus sylvestris.)
 *  - Downy birch bog form (Betula pubescens): on wet peat/bog margins it is a
 *    SMALL, SCRUBBY, often MULTI-STEMMED tree 2–6 m — thin, sparse, drooping
 *    twigs, small leaves, whitish bark low down; a poor stunted version of the
 *    forest birch. (Wikipedia / EUFORGEN Betula pubescens; BSBI accounts.)
 *
 * NEEDLE-COVERAGE anti-clumping (user-flagged existing-pine defect): the shared
 * needle builder (LeafMesh.buildNeedleSpray) puts `needleCount` needles in ONE
 * spray per anchor; the stock PINE uses 88 (×3 hero = 264) in a tight radial
 * brush, so a few anchors become big DENSE balls with bare crown between them.
 * The fix here is purely PARAMETRIC (no shared-file edit): spread the SAME needle
 * budget over MANY small tufts — modest `needleCount` per spray (short tuft, not a
 * ball) + tight anchor `spacing` (overlapping tufts run continuous needle cover
 * along every twig) + a real level-3 twig layer so twigs reach across the whole
 * (sparse, open) crown. Sparse bog-pine crown = FEWER/opener branches, never
 * tight clumps with gaps. Verified in the QA preview.
 *
 * Self-contained: exports the two SpeciesParams + a materialed `buildPreview`.
 * Does NOT touch Species.ts / TREE_SPECIES / SpeciesMap / enum / VegLibrary — the
 * integrator wires those from the report.
 */

import { DoubleSide, Group, Mesh, MeshStandardMaterial, Object3D } from 'three';
import type { Rng } from '../../core/Seed';
import { buildTree } from '../TreeBuilder';
import type { SpeciesParams } from '../VegTypes';

/**
 * BOG_PINE — stunted gnarled raba Scots pine. Dwarf stature + thick twisted bole
 * + sparse OPEN flat/candelabra crown. Conifer needleSpray foliage (like PINE)
 * but on a KARST_GNARL-style gnarled skeleton, shrunk to 1.5–4 m.
 */
export const BOG_PINE: SpeciesParams = {
  id: 'bogPine',
  label: 'Bog pine (stunted Scots pine)',
  kind: 'conifer',
  // real stunted range; the cook applies a per-instance scale from ref_height
  // (recommend ~4 m — LOW — so the cooked `scale` byte reads as a dwarf).
  height: [2.2, 3.8],
  trunkRadiusK: 0.052, // thick-for-its-height twisted bole (vs forest PINE 0.021)
  crown: 'irregular', // idiosyncratic, bonsai-like, open — never a clean cone/dome
  asym: 0.38, // one-sided lean, but not so strong the crown collapses to a blob
  // NB: only 3 levels (trunk → gnarled candelabra arms → needle shoots). A 4th
  // level starves on a dwarf — each level roughly halves branch length, so a
  // 4-level chain drives the needle-bearing twigs to ~5 cm and near-zero anchors
  // (measured: an empty crown). Needles ride the longer LEVEL-2 shoots instead.
  levels: [
    {
      // trunk: short, THICK, strongly TWISTED/gnarled (high wander), bare low bole
      density: 0, whorl: 0, childStart: 0, childEnd: 0,
      angleBase: 0, angleTip: 0, lenRatio: 0, lenJitter: 0, radRatio: 0,
      segs: 8, wander: 0.24, gravitropism: 0.02, droop: 0, tipCurl: 0.08, taper: 0.8,
    },
    {
      // primaries: FEW gnarled wide near-horizontal candelabra arms starting HIGH
      // on the bole (bare lower trunk); up-hooked tips → the flat umbrella top.
      density: 2.9, whorl: 0, childStart: 0.4, childEnd: 0.98,
      angleBase: 1.5, angleTip: 0.8, lenRatio: 0.72, lenJitter: 0.5, radRatio: 0.5,
      segs: 6, wander: 0.32, gravitropism: 0.02, droop: 0.3, tipCurl: 0.28, taper: 0.82,
    },
    {
      // needle shoots — MANY short rising shoots along each arm, spread across the
      // whole crown; the needle tufts ride these (anchorLevel 2). Density + the
      // shoot LENGTH here are what SPREAD the tufts into continuous cover.
      density: 6.5, whorl: 0, childStart: 0.12, childEnd: 1.0,
      angleBase: 0.9, angleTip: 0.5, lenRatio: 0.56, lenJitter: 0.45, radRatio: 0.5,
      segs: 3, wander: 0.26, gravitropism: 0.1, droop: 0.12, tipCurl: 0.15, taper: 0.82,
    },
  ],
  foliage: {
    kind: 'needleSpray',
    anchorLevel: 2,
    // TIGHT spacing → many anchors per shoot; overlapping small tufts run
    // CONTINUOUS needle cover along each shoot (the anti-clump lever, vs PINE 0.11).
    spacing: 0.045,
    tStart: 0.12,
    // SHORT tufts (13–22 cm) — a bog-pine shoot cluster, not a big forest spray.
    scale: [0.13, 0.22],
    tilt: 0.55,
    clusterSize: [1, 1],
    normalBend: 0.66,
    // MODEST needle budget per spray (×3 hero = 60) so a tuft is small & readable,
    // NOT the stock 88 (×3 = 264) dense ball. Short bluish needles.
    leaf: { len: 0.095, width: 0.02, shapePow: 1, fold: 0, curl: 0, needleCount: 20, brush: 0.7 },
  },
  flare: { amp: 0.7, height: 0.5, lobes: 6 }, // gnarled buttressed base
  barkLayer: 1, // reuse PINE bark (reddish-scaly upper / dark furrowed lower) — no new VRAM
  foliageColor: { r: 0.035, g: 0.075, b: 0.058, hueVar: 0.2 }, // dark blue-green needles
  brokenTop: 0,
  stubChance: 0.14, // dead/broken stubs are common on bog pines
};

/**
 * BOG_BIRCH — scrubby multi-stem downy birch bog form. Small, thin, sparse, with
 * drooping twigs + small leaves. Multi-stem is achieved on the single-trunk tree
 * grammar by a FEW steep, long, co-dominant level-1 stems emerging from the LOW
 * bole (childStart 0.03, childEnd 0.35) — they ascend nearly parallel to the
 * trunk and read as a clump of stems, not lateral branches.
 */
export const BOG_BIRCH: SpeciesParams = {
  id: 'bogBirch',
  label: 'Bog birch (scrubby downy birch)',
  kind: 'broadleaf',
  height: [2.9, 4.0], // small stunted bog form (vs forest BIRCH 9–15 m). Height is
  // the DOMINANT tri driver: branch/twig/anchor counts scale super-linearly with
  // stem length, so a wide range (2.6–4.4) swung per-instance tris ~6.5×. This
  // tight band (2.9–4.0, all clearly SMALL bog birches) is the main tri-stability
  // lever — it caps the top (no lush overshoot) AND lifts the floor (no tiny bare
  // sapling), keeping the cross-seed tri spread modest.
  trunkRadiusK: 0.014, // slender birch stems
  crown: 'irregular', // scrubby, asymmetric, open
  asym: 0.32,
  levels: [
    {
      // low central stem: sinuous, a little leaning
      density: 0, whorl: 0, childStart: 0, childEnd: 0,
      angleBase: 0, angleTip: 0, lenRatio: 0, lenJitter: 0, radRatio: 0,
      segs: 9, wander: 0.14, gravitropism: 0.04, droop: 0, tipCurl: 0, taper: 1.0,
    },
    {
      // CO-DOMINANT STEMS: a clear clump of steep (near-vertical), long slender
      // stems from the LOW bole → the multi-stem bog-birch signature. Density
      // raised (2.2→3.1) so 2–3 stems reliably read even on unlucky seeds; tighter
      // lenJitter (0.32→0.2) keeps every stem near the same length so no lucky seed
      // grows one dominant forest-tree leader.
      density: 3.1, whorl: 0, childStart: 0.015, childEnd: 0.28,
      angleBase: 0.55, angleTip: 0.5, lenRatio: 0.72, lenJitter: 0.15, radRatio: 0.78,
      segs: 8, wander: 0.16, gravitropism: 0.05, droop: 0.14, tipCurl: 0.0, taper: 0.95,
    },
    {
      // branches off each stem: thin, spreading, beginning to droop. Raised
      // (2.2→3.0) so twig-bearing branches reach ACROSS the whole small crown —
      // the coverage lever that fills the silhouette instead of leaving bare gaps.
      density: 3.0, whorl: 0, childStart: 0.16, childEnd: 0.99,
      angleBase: 1.0, angleTip: 0.55, lenRatio: 0.5, lenJitter: 0.22, radRatio: 0.5,
      segs: 5, wander: 0.16, gravitropism: -0.05, droop: 0.32, tipCurl: -0.05, taper: 0.9,
    },
    {
      // slender drooping twigs carrying the small-leaf clusters (anchorLevel 3).
      // Raised (4.4→6.8) + earlier start (0.28→0.18) so leafy twigs spread down
      // each branch and fill the crown — a MODEST but clearly-leafy small bog
      // birch, not a bare stick. Still airy/open (small crown, thin twigs); tighter
      // lenJitter (0.4→0.3) keeps twig count — and thus tri count — stable per seed.
      density: 6.0, whorl: 0, childStart: 0.16, childEnd: 1.0,
      angleBase: 0.75, angleTip: 0.45, lenRatio: 0.44, lenJitter: 0.2, radRatio: 0.5,
      segs: 3, wander: 0.14, gravitropism: -0.12, droop: 0.55, tipCurl: -0.05, taper: 0.85,
      planar: 0.4,
    },
  ],
  foliage: {
    kind: 'leafCluster',
    anchorLevel: 3,
    // tighter anchor spacing (0.075→0.052): overlapping small-leaf nodes run
    // near-continuous leaf cover along every drooping twig (the anti-gap lever) —
    // still small blades on a small crown, so it reads leafy without going lush.
    spacing: 0.058,
    tStart: 0.12,
    scale: [0.085, 0.13], // SMALL leaves (bog-form) — actual blade ≈ 8–13 cm
    tilt: 0.9,
    clusterSize: [3, 3], // a fixed few small leaves per node → leafy twig, not bare
    // (fixed count, not a range, so per-instance tri count stays stable across seeds)
    normalBend: 0.66,
    planarLeaves: true,
    // small pointed toothed ovate birch blade
    leaf: { len: 1.0, width: 0.5, shapePow: 1.4, fold: 0.22, curl: 0.3, needleCount: 0, brush: 0 },
  },
  flare: { amp: 0.3, height: 0.5, lobes: 4 },
  barkLayer: 3, // reuse BIRCH bark (whitish with dark lenticels) — no new VRAM
  foliageColor: { r: 0.07, g: 0.14, b: 0.035, hueVar: 0.34 }, // fresh green, slightly dull
  brokenTop: 0,
  stubChance: 0.05,
};

// ---- preview -----------------------------------------------------------------

/** Studio materials mirroring how VegLibrary tints a tree pool (bark opaque +
 *  needle/leaf crown). The foliage mesh carries no vertex-colour attribute, so a
 *  solid tuned MeshStandardMaterial stands in for the runtime vertex-tint path. */
function pineMats(): { bark: MeshStandardMaterial; foliage: MeshStandardMaterial } {
  return {
    // reddish-brown scaly upper / dark furrowed lower Scots-pine bark (one tone)
    bark: new MeshStandardMaterial({ color: 0x6e4a30, roughness: 0.93, metalness: 0 }),
    // dark blue-green needle tufts, two-sided (single-quad needles)
    foliage: new MeshStandardMaterial({ color: 0x35513f, roughness: 0.62, metalness: 0, side: DoubleSide }),
  };
}
function birchMats(): { bark: MeshStandardMaterial; foliage: MeshStandardMaterial } {
  return {
    // whitish birch bark (dark lenticels not modelled at this scale)
    bark: new MeshStandardMaterial({ color: 0xcfc9bb, roughness: 0.82, metalness: 0 }),
    // fresh green small leaves, two-sided
    foliage: new MeshStandardMaterial({ color: 0x5f7d33, roughness: 0.58, metalness: 0, side: DoubleSide }),
  };
}

/** Build BOTH bog trees side by side, materialed, real scale (m), +Y up, base y=0.
 *  Each is grown via `buildTree` with the real needle/leaf crown (foliageMode
 *  'mesh', lod 0) — the exact hero path the cook uses, so the QA reflects runtime. */
export function buildPreview(rng: Rng): Object3D {
  const root = new Group();

  const pine = buildTree(BOG_PINE, rng.fork('bogPine'), { lod: 0, foliageMode: 'mesh' });
  const pm = pineMats();
  const pineGroup = new Group();
  pineGroup.add(new Mesh(pine.bark, pm.bark));
  if (pine.foliageMesh) pineGroup.add(new Mesh(pine.foliageMesh, pm.foliage));
  pineGroup.position.x = -1.6;
  root.add(pineGroup);

  const birch = buildTree(BOG_BIRCH, rng.fork('bogBirch'), { lod: 0, foliageMode: 'mesh' });
  const bm = birchMats();
  const birchGroup = new Group();
  birchGroup.add(new Mesh(birch.bark, bm.bark));
  if (birch.foliageMesh) birchGroup.add(new Mesh(birch.foliageMesh, bm.foliage));
  birchGroup.position.x = 1.6;
  root.add(birchGroup);

  return root;
}
