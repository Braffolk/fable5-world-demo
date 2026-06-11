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

export function buildTree(
  sp: SpeciesParams,
  rng: Rng,
  opts?: {
    lod?: 0 | 1 | 2;
    inst?: Partial<GrowthInstance>;
    /** 'cards' (default) | 'mesh' (real leaves only) | 'hybrid' (hero: both) */
    foliageMode?: 'cards' | 'mesh' | 'hybrid';
  },
): BuiltTree {
  const lod = opts?.lod ?? 0;
  const skel = growSkeleton(sp, rng, opts?.inst);

  // ---- bark/tubes ------------------------------------------------------------
  const barkG = new MeshGrower();
  const lodK = lod === 0 ? 1 : lod === 1 ? 0.7 : 0.45;
  const maxLevel = lod === 0 ? 99 : lod === 1 ? 2 : 1;
  tubesForSkeleton(barkG, skel, rng.fork('tubes'), {
    lodK,
    uRepeats: sp.barkRepeats,
    flare: { ...sp.flare, phase: rng.float() * Math.PI * 2 },
    maxLevel,
  });
  const barkTris = barkG.triCount;
  const bark = barkG.build();

  // ---- foliage ---------------------------------------------------------------
  let foliage: BufferGeometry | null = null;
  let foliageMesh: BufferGeometry | null = null;
  let folTris = 0;
  if (sp.foliage && skel.anchors.length > 0 && lod === 0) {
    const fol = sp.foliage;
    const mode = opts?.foliageMode ?? 'cards';
    const crownC = new Vector3(0, skel.crownCenterY, 0);
    const crownR = Math.max(skel.crownRadius, (skel.height - skel.crownCenterY) * 0.9);
    if (mode === 'cards' || mode === 'hybrid') {
      const folG = new MeshGrower();
      buildFoliageCards(folG, skel.anchors, fol.card, rng.fork('foliage'));
      folG.bendNormals(crownC, crownR, fol.normalBend);
      folG.crownAO(crownC, crownR, 0.55);
      folTris += folG.triCount;
      foliage = folG.build();
    }
    if (mode === 'mesh' || mode === 'hybrid') {
      const folG = new MeshGrower();
      const folRng = rng.fork('foliageMesh');
      // real needles need ~3x density to match the painted card sprays
      const heroLeaf =
        fol.kind === 'needleSpray'
          ? { ...fol.leaf, needleCount: Math.round(fol.leaf.needleCount * 3), len: fol.leaf.len * 1.15 }
          : fol.leaf;
      for (const anchor of skel.anchors) {
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
