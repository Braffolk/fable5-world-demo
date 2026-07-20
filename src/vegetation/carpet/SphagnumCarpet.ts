/**
 * SPHAGNUM — carpet #1 (the raised-bog peat-moss lawn). Meshes + reference live in
 * vegetation/bog/Sphagnum.ts (packed-capitula LAWN tile + hero hummock cushion,
 * rigid vdata, BogLodCtx prune-and-preserve); this file is only the CarpetSpec.
 *
 * ── scatter/scale numbers (the perf plan — derived, not vibes) ────────────────
 * The instance budget pins the grid: carpet instances ride the streamed instance
 * pool at 2 slots each (leaf + voxel head), and the carpet band stays resident to
 * maxDist 160 m ⇒ resident area ≈ π·(160 + cell reach)² ≈ 1.8·10⁵ m². A step-S
 * lattice costs ≈ 2·1.8·10⁵/S² slots: the authored 0.45 m tile scattered at its
 * own footprint would be ~1.8 M slots (>3× the whole pool) — impossible. step
 * 1.6 m ⇒ ~70 k patches ≈ 140 k slots (≤ ~26 blocks) worst-case in open
 * full-cover bog, where the tree band is at its emptiest (moss and dense forest
 * are disjoint by habitat, so the two pool peaks never stack).
 * scale [2.8, 3.4] closes the coverage: 0.45 m × ~3.1 ≈ 1.4 m mean footprint on
 * the 1.6 m lattice (±1-step jitter) ⇒ ~75 % ground cover with irregular peat
 * showing through (real lawns have hollows; the tint floor colors them). The
 * tile CANNOT go bigger: the bog's cooked hummock-hollow micro-relief has
 * ±30-50 cm swings at 1-2 m wavelength, and a rigid tile bridges anything wider
 * than that (boot-verified: ×4 tiles read as floating rafts) — lean-shear is a
 * no-op on a flat tile (y≈0 verts), so tile size + sink are the ONLY conformance
 * levers. sink 0.08 seats the rim under the litter line so bridged edges tuck
 * instead of hover. Tri cost is scale-free (tris/m² = patch tris/step² ≈
 * 1.4 k/m², ~0.6 M in-frustum tris at 0-30 m before the BogLod rungs cut in; the
 * crown-DAG error cut is screen-space × A.w — screen-correct at any scale).
 * KNOWN TRADE (flagged for the visual gate): ×3 capitula read as chunky ~10-15 cm
 * moss bumps up close; true 2-5 cm capitulum grain at full coverage needs a
 * natively coarser-authored ~1.5 m tile (mesh follow-up), not a denser lattice.
 * voxGridDim 32: local L0 cell 14 mm ⇒ brick 56 mm ⇒ ~22 cm world bricks at ×4 —
 * carpet-relief scale, ~60-100 occupied bricks/variant (≲ a few KB each; the mid
 * band splats ≈ tree-voxel-lane volume in the worst bog pose, again where trees
 * are absent). coverThreshold 0.12: the density plane is already suitability-cut,
 * so any meaningfully mossy texel carpets — holes come from real zero-density
 * ground, not acceptance noise.
 */

import { VegClass } from '../../gpu/passes/Scatter';
import {
  buildSphagnumHummock,
  buildSphagnumPatch,
  SPHAGNUM_FOLIAGE,
  SPHAGNUM_MAX_DIST,
  SPHAGNUM_VOX_NEAR,
} from '../bog/Sphagnum';
import type { CarpetSpec } from './CarpetTypes';

export const SPHAGNUM_CARPET: CarpetSpec = {
  id: 'sphagnum',
  patchClass: VegClass.SphagnumPatch,
  buildPatch: buildSphagnumPatch,
  tint: SPHAGNUM_FOLIAGE,
  voxNear: SPHAGNUM_VOX_NEAR,
  maxDist: SPHAGNUM_MAX_DIST,
  voxGridDim: 32,
  keywords: /sphagnum|bog_moss|peat_moss/,
  coverThreshold: 0.12,
  step: 1.6,
  scale: [2.8, 3.4],
  sink: 0.08,
  hero: {
    cls: VegClass.SphagnumHummock,
    build: buildSphagnumHummock,
    // ~0.02 cushions/m² in full moss (≈100 inside the 40 m hero ring) at natural
    // hummock size (0.46-0.7 m) — silhouette relief, not coverage.
    step: 6.0,
    perM2: 0.02,
    scale: [1.0, 1.5],
    sink: 0.03,
    maxDist: 40,
  },
};
