/**
 * BogLod — crown-LOD ladder for the six bog understory leaf pools (CottonGrass,
 * Heather, LabradorTea, BogRosemary, Cranberry, Cloudberry).
 *
 * WHY (in-world diagnosis 2026-07-20, issues 4/5): these pools carry mm-scale
 * elements (0.9-1.7 mm blades, 0.7 mm cotton fibres, 1-2 mm scale-leaves, wiry
 * stems) that rendered SINGLE-LOD at all distances. Beyond ~2-3 m every triangle
 * is sub-pixel; the SW raster has no MSAA, so each dark element lands as isolated
 * stochastic pixels — a world-anchored, TRAA-shimmering speckle cloud stretched
 * along the geometry ("3D dark rays"). Trees never show this because their crowns
 * ride a pre-pruned LOD ladder (TreeBuilder.foliageLadder → buildCrownLodDag).
 *
 * FIX = the SAME proven recipe (Cook §3.3 prune-and-preserve): each coarser rung
 * deterministically REGENERATES the plant with only a fraction λ of the whole
 * elements kept and the survivors WIDENED ×1/λ, so the covered silhouette area is
 * preserved — fewer, wider, still-visible elements instead of a dissolving
 * speckle cloud. Two levers per rung, mirroring the tree schedule's λ/leafRows
 * split (macro pruning stays gentle; most of the tri cut comes from INSIDE the
 * surviving elements):
 *   - `lambda`  whole-element keep fraction (blades / fibres / scale-leaves /
 *               florets / bells / leathery leaves). Survivor WIDTH ×1/λ — width
 *               only, never length/transform (the spike law). The few STRUCTURAL
 *               stems a plant cannot lose (culms under cotton heads, runners,
 *               petioles) are never pruned but widen by the same factor: their
 *               absolute silhouette area is negligible (mm × cm), and unwidened
 *               they would remain exactly the sub-pixel dissolve being fixed.
 *   - `detail`  intra-element tier 0/1/2 (native → reduced → coarse): per-builder
 *               segment/ring/teeth tables (blade segs, fibre segments, berry
 *               sphere resolution, palmate-leaf rings×teeth). Builders whose
 *               element count is too SMALL for stochastic pruning (Cloudberry:
 *               2-3 parasol leaves per shoot — λ could zero a plant) cut on this
 *               lever alone.
 *
 * DETERMINISM: a rung replays the EXACT rng stream of LOD0 — pruned elements are
 * still built, into the ctx's discard grower, so every survivor's draws stay in
 * lockstep and its geometry is LOD0's (width-scaled only). The keep decision is a
 * position-hash threshold (1 mm quantum, the TreeBuilder/BuildAggregateDag mixing
 * constants) ⇒ keep-sets are NESTED across rungs (coarser ⊆ finer), the same
 * crack-free-parenting property the tree ladder establishes. Rung 0 (λ=1,
 * detail 0, widthMul 1) is byte-identical to LOD0 (keeps everything, ×1.0 width).
 *
 * CONSUMED BY: VegLibrary's bog `pool.leaf.buildLadder` (the tree template) →
 * WorldRegistry ladderToMeshes → buildCrownLodDag. The ladder rides the existing
 * knobs (crownLodErrorK 0.4, crownLodOwnErrors placement). ZERO renderer change;
 * strictly fewer tris at range. The rung VALUES ride the 'world-veg' BootCache
 * key (WorldRegistry `bogLod` param) so any edit here invalidates stale DAGs.
 */

import type { Vector3 } from 'three';
import { mix32 } from '../../core/Seed';
import { MeshGrower } from '../TubeMesh';

/** one rung of the bog understory LOD ladder (finest → coarsest). */
export interface BogLodRung {
  /** whole-element keep fraction λ; survivors widen ×1/λ (area-preserving). */
  lambda: number;
  /** intra-element detail tier: 0 = native, 1 = reduced, 2 = coarse. */
  detail: 0 | 1 | 2;
}

/** The shared 4-rung schedule. Rung 0 is native (== LOD0). Coarser rungs land
 *  ≈ [1.0, ~0.5, ~0.3, ~0.15] of LOD0 tris (λ × the per-builder detail cut) —
 *  the same coarsest-rung target band as the tree ladders. At the default
 *  crownLodErrorK 0.4 bake the coarser rungs engage at ≈ 5.3 / 9.6 / 13.2 m
 *  (crownLodOwnErrors CROWN_ENGAGE_FRAC × transitionDist 60 × 0.4, unit scale) —
 *  right where the mm-scale detail goes sub-pixel and the speckle onset was
 *  observed (~2-3 m); the coarsest rung owns everything out to clsMaxDist
 *  (bog pools have no voxel sibling). */
export const BOG_LOD_LADDER: readonly BogLodRung[] = [
  { lambda: 1.0, detail: 0 },
  { lambda: 0.55, detail: 1 },
  { lambda: 0.32, detail: 1 },
  { lambda: 0.18, detail: 2 },
];

/** 1 mm quantum for the stable keep-key — matches TreeBuilder.CROWN_HASH_QUANT. */
const QUANT = 1000;

/**
 * Per-rung build context threaded through a bog builder. The NATIVE ctx (λ=1)
 * is the default parameter everywhere, so existing callers (pool LOD0 build,
 * previews) are untouched and rung 0 rebuilds bit-identically.
 */
export class BogLodCtx {
  /** survivor width multiplier = 1/λ (Cook §3.3 area preservation, width-only). */
  readonly widthMul: number;
  /** discard sink for pruned elements — they are still BUILT (identical rng
   *  draws → survivors stay in lockstep with LOD0), just never emitted. */
  private readonly discard = new MeshGrower();

  constructor(
    readonly lambda: number,
    readonly detail: 0 | 1 | 2,
    private readonly seed: number,
  ) {
    this.widthMul = lambda > 0 ? 1 / lambda : 1;
  }

  /** does the element anchored at `p` (disambiguated by a rung-stable `salt`,
   *  e.g. its emission index) survive this rung? Nested across rungs. */
  keep(p: Vector3, salt: number): boolean {
    if (this.lambda >= 1) return true;
    const h =
      (Math.imul(Math.round(p.x * QUANT), 73856093) ^
        Math.imul(Math.round(p.y * QUANT), 19349663) ^
        Math.imul(Math.round(p.z * QUANT), 83492791) ^
        Math.imul(salt | 0, 2654435761) ^
        Math.imul(this.seed | 0, 0x9e3779b9)) >>>
      0;
    return (mix32(h) >>> 0) / 4294967296 < this.lambda;
  }

  /** route an element: the real grower if it survives, the discard sink if not. */
  target(g: MeshGrower, p: Vector3, salt: number): MeshGrower {
    return this.keep(p, salt) ? g : this.discard;
  }
}

/** the identity ctx — keeps every element at native width/detail (LOD0). */
export const BOG_LOD_NATIVE = new BogLodCtx(1, 0, 0);
