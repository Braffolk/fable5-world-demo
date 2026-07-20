/**
 * The CARPET layer — continuous, terrain-conforming, NON-WIND ground-cover strata
 * (sphagnum lawns today; lichen mats / feather-moss floors later). A carpet is not
 * understory: understory is discrete wind-swayed plants with acceptance-thinning;
 * a carpet is gap-free where present and ends in an aggregate band, not a cull.
 *
 * Render triad (all existing machinery — a carpet is DATA, not a render path):
 *   near   0..voxNear       patch MESH instances (leaf matClass, 'rigid' channel —
 *                           the wind fetch skips channel 0 entirely) + sparse HERO
 *                           cushions for silhouette right around the camera
 *   mid    voxNear..maxDist the SAME patch mesh's VOXEL sibling on the brick raster
 *                           (prepareVoxelCrown → registerVoxelHead, per-spec grid)
 *   far    beyond maxDist   the terrain material's per-class ground tint — the
 *                           zero-instance representation (already cooked+shipped)
 *
 * Placement is DETERMINISTIC (understory density-plane ≥ coverThreshold ⇒ emit; a
 * carpet with acceptance-thinning holes is not a carpet) on its own instance band
 * (UnderstoryScatter.carpetPlan), so the understory plant scatter rides ON TOP
 * independently — bog plants grow IN the moss, they never compete for cells.
 *
 * Every class-specific behavior lives in this descriptor; the renderer gates
 * (WorldRegistry classPolicy / voxelFarClass, ScatterMap carpetCover) are generic
 * over CARPET_SPECS. Slot-in cost of a new cover: 1 builder file + 1 spec entry +
 * 1 VegClass id (+ its cooked palette token).
 */

import type { BufferGeometry } from 'three';
import type { Rng } from '../../core/Seed';
import { VegClass } from '../../gpu/passes/Scatter';
import type { BogLodCtx } from '../bog/BogLod';
import { SPHAGNUM_CARPET } from './SphagnumCarpet';

/** a prune-and-preserve LOD-ready mesh builder (the bog kit contract): default
 *  lod = native LOD0; a BogLodCtx regenerates a coarser rung from the same seed. */
export type CarpetBuilder = (rng: Rng, lod?: BogLodCtx) => { geo: BufferGeometry; tris: number };

export interface CarpetSpec {
  id: string;
  /** the carpet tile class — near mesh head + mid voxel sibling on one instance. */
  patchClass: VegClass;
  buildPatch: CarpetBuilder;
  /** leaf-head tint (packLeafTint payload); vdata.x spreads it by hueVar. */
  tint: { r: number; g: number; b: number; hueVar: number };
  /** patch mesh → voxel sibling handoff distance (m). Per-class — carpets hug the
   *  ground, so they hand off far nearer than the tree crowns' global ?voxnear. */
  voxNear: number;
  /** voxel band end (m) = the class max draw distance (≤ the understory band). */
  maxDist: number;
  /** per-crown voxel grid dim. The tree default (256) assumes crown-scale extents;
   *  on a sub-metre patch it would build mm cells whose bricks stay sub-pixel
   *  across the whole mid band (brick-splat spam). Sized so the L0 brick
   *  (4 cells) lands at carpet-relief scale instead. */
  voxGridDim: number;
  /** understory palette tokens that light this carpet up (ScatterMap.carpetCover).
   *  The same tokens stay SKIP for the understory distribution — presence here
   *  adds the carpet, it never rebalances the plant mix. */
  keywords: RegExp;
  /** min understory density-plane value (0..1) for a patch cell to emit. */
  coverThreshold: number;
  /** carpet grid step (m) — the deterministic global patch lattice. */
  step: number;
  /** per-instance uniform scale range, hy-lerped (footprint ≈ mesh footprint ×
   *  scale; pick so step ≈ scaled footprint → gap-free abutment). */
  scale: readonly [number, number];
  /** bed sink (m) — seats the skirted patch rim under the litter line. */
  sink: number;
  /** optional sparse hero class (individual cushions/hummocks, mesh-only). */
  hero?: {
    cls: VegClass;
    build: CarpetBuilder;
    /** hero jitter-grid step (m) + acceptance density (heroes/m², density-scaled). */
    step: number;
    perM2: number;
    scale: readonly [number, number];
    sink: number;
    maxDist: number;
  };
}

export const CARPET_SPECS: readonly CarpetSpec[] = [SPHAGNUM_CARPET];

/** every carpet class (patch + hero) — the classPolicy routing set. */
export const CARPET_CLASSES: ReadonlySet<number> = new Set(
  CARPET_SPECS.flatMap((s) => (s.hero ? [s.patchClass, s.hero.cls] : [s.patchClass])),
);

const BY_PATCH_CLASS = new Map<number, CarpetSpec>(CARPET_SPECS.map((s) => [s.patchClass, s]));

/** the spec whose PATCH class this is (voxel-far routing) — heroes return undefined
 *  (mesh-only, no voxel sibling). */
export function carpetSpecOfPatch(cls: number): CarpetSpec | undefined {
  return BY_PATCH_CLASS.get(cls);
}

/** serializable spec digest for the 'world-veg' BootCache key — every field that
 *  shapes the cached crowns/DAGs (classes, bands, voxel grid, tint). Keywords and
 *  scatter params are runtime-only (they shape placement, not cached geometry). */
export const CARPET_CACHE_PARAMS = CARPET_SPECS.map((s) => ({
  id: s.id,
  patch: s.patchClass as number,
  hero: s.hero ? (s.hero.cls as number) : null,
  voxNear: s.voxNear,
  maxDist: s.maxDist,
  voxGridDim: s.voxGridDim,
  tint: s.tint,
}));
