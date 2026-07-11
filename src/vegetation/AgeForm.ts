/**
 * AgeForm — the single source of truth for the tree AGE/SIZE model (#110).
 *
 * PROBLEM this replaces: a per-tree `scale` (A.w = crownHeight/refHeight, real
 * nDSM canopy height on Estonia; a power-biased jitter on the generated world)
 * was applied as a UNIFORM matrix multiply of the whole tree mesh
 * (NaniteCommon.instTransformPoint: `p.mul(A.w)`). A "big tree" was therefore a
 * photo-enlarged small tree — trunk, crown width, needle size all ×the same
 * factor — so proportions stayed CONSTANT across a ~2.4× size range and the
 * extremes read as insane (a bonsai-proportioned giant, a fat-trunked dwarf).
 *
 * FIX: real trees do NOT scale isometrically with age. A young tree is slender,
 * short, crowned near the ground, with a narrow cone and low height-variance; an
 * old tree is stouter, taller, self-pruned to a bare lower trunk, with a broader
 * / flatter / fuller crown and higher height-variance. So we keep A.w carrying
 * the REAL absolute size (never fake the height), but interpret `scale` as an
 * AGE proxy that selects among a modest set of age-appropriate variant FORMS
 * (differing in PROPORTION, baked by the growth grammar — see Skeleton.ts
 * ontogeny), NOT an unbounded uniform blow-up.
 *
 * The library bakes TREE_VARIANTS forms per species along a young→old maturity
 * ladder (VegLibrary.variantInstance ← ageForSlot); the placement pipeline picks
 * the slot from the per-tree scale (ChunkContent ← ageStageVariant). Both sides
 * MUST agree on what a slot means, so both live here. No count increase: the
 * 4 existing variant slots are repurposed as the age ladder.
 */

import { TREE_VARIANTS } from '../gpu/passes/Scatter';

/** number of age-stage forms per species = the existing variant-slot budget (4). */
export const TREE_AGE_STAGES = TREE_VARIANTS;

/**
 * Baked maturity (0 = juvenile .. 1 = veteran) for variant slot `v`. Slots are a
 * young→old ladder so that scale-driven selection maps a small tree to a young
 * FORM and a large tree to an old FORM. Endpoints are kept off 0/1 (real stands
 * have neither pure saplings nor pure snags among the canopy). Monotone in v.
 */
export function ageForSlot(v: number): number {
  const n = TREE_AGE_STAGES;
  if (n <= 1) return 0.6;
  // even spacing across [0.12 .. 0.92]
  return 0.12 + (0.8 * v) / (n - 1);
}

/** scale window mapped onto the age ladder. `scale` ~ height/refHeight, so 1.0 is
 *  a reference-mature tree; below LO reads as young, above HI as veteran. Centred
 *  on the generated-world scale median (~0.95) and sensible for Estonia's
 *  crownHeight/refHeight. Tunable if a stand reads too uniformly young/old. */
const AGE_SCALE_LO = 0.6;
const AGE_SCALE_HI = 1.3;
/** band-edge dither (±AGE_DITHER/2 of a slot) so an even-aged stand still shows two
 *  adjacent forms instead of one repeated DAG — variety without more geometry. */
const AGE_DITHER = 0.9;

/**
 * Per-tree age-stage variant slot from the real `scale` (age proxy) + a stable
 * per-tree dither. `dither01` ∈ [0,1) decorrelates band edges — pass the cooked
 * random variant byte's fraction so trees of equal size still split across two
 * adjacent forms (structural variety is retained; the size→age correlation is
 * kept loose, not rigid). Deterministic: identical inputs → identical slot, so
 * the count and fill passes agree.
 */
export function ageStageVariant(scale: number, dither01: number): number {
  const n = TREE_AGE_STAGES;
  const t = Math.max(0, Math.min(1, (scale - AGE_SCALE_LO) / (AGE_SCALE_HI - AGE_SCALE_LO)));
  const idx = Math.floor(t * n + (dither01 - 0.5) * AGE_DITHER);
  return Math.max(0, Math.min(n - 1, idx));
}

/** dither seed from a cooked/scattered random variant byte (0..TREE_VARIANTS-1). */
export function ditherOfVariant(variant: number): number {
  return (((variant | 0) % TREE_AGE_STAGES) + 0.5) / TREE_AGE_STAGES;
}
