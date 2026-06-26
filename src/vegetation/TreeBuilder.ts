/**
 * TreeBuilder — species params + seed → renderable geometry.
 * LOD0 (hero): full tube hierarchy + REAL foliage meshes (needle quads /
 * leaf strips) merged into two geometries (bark, foliage). LOD1/2 swap
 * foliage for captured cards and drop tube levels (Phase 4 capture rig).
 */

import { Vector3 } from 'three';
import type { BufferGeometry } from 'three';
import type { Rng } from '../core/Seed';
import { buildFoliageCards } from './FoliageCards';
import { buildLeafCluster, buildSprayAt } from './LeafMesh';
import { growSkeleton } from './Skeleton';
import { MeshGrower, tubesForSkeleton } from './TubeMesh';
import type { GrowthInstance, Skeleton, SpeciesParams } from './VegTypes';

export interface BuiltTree {
  bark: BufferGeometry;
  /** card foliage (atlas material) — null for snags or mesh-only mode */
  foliage: BufferGeometry | null;
  /** real leaf/needle geometry (vertex-color material) — hero/hybrid mode */
  foliageMesh: BufferGeometry | null;
  skeleton: Skeleton;
  stats: { tris: number; anchors: number; branches: number; height: number };
}

export interface HeroDiet {
  /** card-spray budget (anchors strided to this, survivors enlarged) */
  cardTarget?: number;
  /** real-leaf anchor budget (stride over anchors, full leaf density each) */
  meshAnchorTarget?: number;
  /** tube radial-segment multiplier (1 = gallery hero) */
  barkK?: number;
}

export function buildTree(
  sp: SpeciesParams,
  rng: Rng,
  opts?: {
    lod?: 0 | 1 | 2;
    inst?: Partial<GrowthInstance>;
    /** 'cards' (default) | 'mesh' (real leaves only) | 'hybrid' (hero: both) */
    foliageMode?: 'cards' | 'mesh' | 'hybrid';
    /** budgets the lod-0 hero down from gallery scale (~1.2M) to a ring cost */
    hero?: HeroDiet;
    /** false → legacy independent open-tube bark (G5 A/B ablation, ?nojunctions) */
    junctions?: boolean;
  },
): BuiltTree {
  const lod = opts?.lod ?? 0;
  const skel = growSkeleton(sp, rng, opts?.inst);

  // ---- bark/tubes ------------------------------------------------------------
  // Ring LODs stop the tube hierarchy BELOW the anchor level — the card
  // sprays visually own that level, so its tubes are pure waste (a forest
  // beech carried 98k card + 13k twig tris before this diet).
  const anchorLevel = sp.foliage?.anchorLevel ?? 2;
  const barkG = new MeshGrower();
  const lodK = lod === 0 ? (opts?.hero?.barkK ?? 1) : lod === 1 ? 0.6 : 0.32;
  const maxLevel =
    lod === 0 ? 99 : lod === 1 ? Math.max(1, anchorLevel - 1) : Math.max(1, anchorLevel - 2);
  tubesForSkeleton(barkG, skel, rng.fork('tubes'), {
    lodK,
    uRepeats: sp.barkRepeats,
    flare: { ...sp.flare, phase: rng.float() * Math.PI * 2 },
    maxLevel,
    branchStride: lod === 2 ? 2 : 1,
    junctions: opts?.junctions ?? true,
  });
  const barkTris = barkG.triCount;
  const bark = barkG.build();

  // ---- foliage ---------------------------------------------------------------
  let foliage: BufferGeometry | null = null;
  let foliageMesh: BufferGeometry | null = null;
  let folTris = 0;
  if (sp.foliage && skel.anchors.length > 0) {
    const fol = sp.foliage;
    const mode = opts?.foliageMode ?? 'cards';
    const crownC = new Vector3(0, skel.crownCenterY, 0);
    const crownR = Math.max(skel.crownRadius, (skel.height - skel.crownCenterY) * 0.9);
    if (mode === 'cards' || mode === 'hybrid') {
      // ring LODs thin anchors to a card budget and enlarge the survivors
      // (≈ sqrt(stride) keeps painted coverage), so high-anchor species
      // (beech: 24k anchors) cost the same as low-anchor ones
      // R1 keeps more, smaller cards (enlargement cap 1.9): the old 1100 ×
      // 3.1-size cards were meter-scale sheets that read as dark slabs at
      // grazing angles 30–100 m out (beech: 24k anchors → stride 23)
      const target =
        lod === 0 ? opts?.hero?.cardTarget ?? Infinity : lod === 1 ? 2600 : 300;
      const stride = Math.max(1, Math.ceil(skel.anchors.length / target));
      const anchors =
        stride > 1 ? skel.anchors.filter((_, i) => i % stride === 0) : skel.anchors;
      const sizeCap = lod === 1 ? 1.9 : 3.1; // R2 cards are distant px — keep coverage
      const card =
        stride > 1
          ? {
              ...fol.card,
              sizeK: fol.card.sizeK * Math.min(sizeCap, Math.sqrt(stride) * 0.9 + 0.12),
            }
          : fol.card;
      const folG = new MeshGrower();
      buildFoliageCards(folG, anchors, card, rng.fork('foliage'));
      folG.bendNormals(crownC, crownR, fol.normalBend);
      folG.crownAO(crownC, crownR, 0.55);
      folTris += folG.triCount;
      foliage = folG.build();
    }
    if ((mode === 'mesh' || mode === 'hybrid') && lod === 0) {
      const folG = new MeshGrower();
      const folRng = rng.fork('foliageMesh');
      // real needles need ~3x density to match the painted card sprays
      const heroLeaf =
        fol.kind === 'needleSpray'
          ? { ...fol.leaf, needleCount: Math.round(fol.leaf.needleCount * 3), len: fol.leaf.len * 1.15 }
          : fol.leaf;
      const meshTarget = opts?.hero?.meshAnchorTarget ?? Infinity;
      const mStride = Math.max(1, Math.ceil(skel.anchors.length / meshTarget));
      const meshAnchors =
        mStride > 1 ? skel.anchors.filter((_, i) => i % mStride === 0) : skel.anchors;
      for (const anchor of meshAnchors) {
        if (fol.kind === 'needleSpray') buildSprayAt(folG, anchor, heroLeaf, folRng);
        else buildLeafCluster(folG, anchor, fol.leaf, fol.clusterSize, folRng);
      }
      folG.bendNormals(crownC, crownR, fol.normalBend);
      folG.crownAO(crownC, crownR, 0.55);
      folTris += folG.triCount;
      foliageMesh = folG.build();
    }
  }

  return {
    bark,
    foliage,
    foliageMesh,
    skeleton: skel,
    stats: {
      tris: barkTris + folTris,
      anchors: skel.anchors.length,
      branches: skel.branches.length,
      height: skel.height,
    },
  };
}
