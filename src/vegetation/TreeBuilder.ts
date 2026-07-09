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
import type { GrowthInstance, LeafAnchor, Skeleton, SpeciesParams } from './VegTypes';

/** One rung of the crown-LOD ladder param table (design §5.2). Rung 0 is native
 *  (λ=1, full detail) ⇒ byte-identical to LOD0; coarser rungs prune whole elements
 *  (λ) AND coarsen survivor internal detail (leafRows / needleMu / stemSegs), with
 *  survivors grown to preserve area. Defined here (not VegLibrary) so LeafMesh/
 *  TreeBuilder — the consumers — do not import upward. */
export interface CrownLodRung {
  /** element keep-fraction λ (whole leaves / whole sprays). Survivors grow width ×(1/λ). */
  lambda: number;
  /** BROADLEAF: blade rows (LOD0 = 4, clamped ≥2). CONIFER: unused. */
  leafRows: number;
  /** CONIFER: intra-spray needle keep-fraction μ. BROADLEAF: unused. */
  needleMu: number;
  /** CONIFER: stem strip render segments (LOD0 = 4). BROADLEAF: unused. */
  stemSegs: number;
}

/** One rung of the crown-LOD ladder (crown-LOD Phase 1). Each level regenerates
 *  the foliage mesh from a de-correlated random keep-subset of WHOLE anchors
 *  (Cook stochastic pruning); `lambda` is the kept fraction. Levels are NESTED:
 *  a coarser level's kept anchors are a subset of every finer level's. */
export interface CrownLodLevel {
  /** target fraction of anchors kept (1 = full density) */
  lambda: number;
  /** regenerated foliage geometry for the survivors of this level */
  geo: BufferGeometry;
  tris: number;
  keptAnchors: number;
}

export interface BuiltTree {
  bark: BufferGeometry;
  /** card foliage (atlas material) — null for snags or mesh-only mode */
  foliage: BufferGeometry | null;
  /** real leaf/needle geometry (vertex-color material) — hero/hybrid mode */
  foliageMesh: BufferGeometry | null;
  /** crown-LOD Phase 1 ladder (only when `opts.crownLodLevels` was supplied);
   *  each rung is a stochastically-pruned regeneration of `foliageMesh`. The
   *  λ=1 rung is bit-identical to `foliageMesh`. null = not requested. Inert
   *  until Phase 2 wires it to the DAG. */
  foliageLadder: CrownLodLevel[] | null;
  skeleton: Skeleton;
  stats: { tris: number; anchors: number; branches: number; height: number };
}

/** 1 mm anchor-centroid quantum for the stable keep-key — matches the aggregate
 *  DAG's per-island hash quantum (BuildAggregateDag.ts) so the two agree. */
const CROWN_HASH_QUANT = 1000;

/** de-correlated, seed-deterministic hash of an anchor's (quantized) position.
 *  The anchor keep-key AND each conifer spray's per-needle keep seed derive from
 *  this, so both are position-de-correlated and reproducible across boots. */
function anchorPosHash(px: number, py: number, pz: number, seed: number): number {
  const ix = Math.round(px * CROWN_HASH_QUANT);
  const iy = Math.round(py * CROWN_HASH_QUANT);
  const iz = Math.round(pz * CROWN_HASH_QUANT);
  return (
    (Math.imul(ix, 73856093) ^
      Math.imul(iy, 19349663) ^
      Math.imul(iz, 83492791) ^
      Math.imul(seed | 0, 2654435761)) >>>
    0
  );
}

/**
 * De-correlated random keep-mask over WHOLE anchors (BASE COOK, §5.3) — the
 * crown-LOD Phase-1 rendering-priority order. Key = hash(quantized anchor
 * centroid ⊕ seed), exactly the `hashOf` of BuildAggregateDag: de-correlated
 * from position/size/normal/colour and seed-deterministic (reproducible builds
 * + bootcache). We keep the LOW-KEY PREFIX and lower the cutoff per coarser
 * level, so keep-sets are NESTED (each level's kept set ⊇ the next-coarser
 * level's) — the prerequisite for crack-free DAG parenting and later temporal
 * fade. NO interior/silhouette bias in Phase 1 (that is an optional extension).
 *
 * Returns a boolean per anchor: true = survives at fraction `lambda`.
 */
export function crownLodKeepMask(
  anchors: readonly LeafAnchor[],
  lambda: number,
  seed: number,
): boolean[] {
  const N = anchors.length;
  const keep = new Array<boolean>(N).fill(false);
  if (N === 0) return keep;
  const frac = Math.max(0, Math.min(1, lambda));
  const keepCount = Math.max(0, Math.min(N, Math.round(frac * N)));
  if (keepCount >= N) return keep.fill(true);
  if (keepCount === 0) return keep;
  const hashOf = (i: number): number => {
    const p = (anchors[i] as LeafAnchor).pos;
    return anchorPosHash(p.x, p.y, p.z, seed);
  };
  // sort indices by ascending key (index tie-break → stable + nested), keep the
  // low-key prefix. Sorting the SAME order for every level guarantees nesting.
  const order = Array.from({ length: N }, (_, i) => i);
  order.sort((a, b) => {
    const ha = hashOf(a);
    const hb = hashOf(b);
    return ha !== hb ? ha - hb : a - b;
  });
  for (let i = 0; i < keepCount; i++) keep[order[i] as number] = true;
  return keep;
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
    /**
     * crown-LOD rung schedule (finest→coarsest; e.g. VegLibrary.crownLodScheduleFor).
     * When supplied AND the foliage MESH is built (mesh/hybrid, lod 0), emit a
     * `foliageLadder` — one regeneration per rung: prune whole anchors to λ,
     * coarsen + area-grow the survivors per the rung's kind-appropriate levers
     * (§3.3/§5). Rung 0 (λ=1, full detail) is byte-identical to the LOD0 mesh.
     * Omitted (default) ⇒ no ladder, zero extra work; the runtime path is unchanged.
     */
    crownLodLevels?: readonly CrownLodRung[];
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
  let foliageLadder: CrownLodLevel[] | null = null;
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
      // crown-LOD Phase 1: snapshot the foliage RNG stream at its START, BEFORE
      // the L0 loop below consumes it. Each ladder level replays this identical
      // stream (folRng is a single stream threaded across ALL anchors, so a
      // survivor's per-leaf jitter depends on the draws of every earlier anchor)
      // — cloning the initial state + advancing the stream for pruned anchors
      // too (see below) keeps a survivor's draws in lockstep, so its vertices
      // are BIT-IDENTICAL to LOD0 regardless of which siblings were pruned.
      const wantLadder = (opts?.crownLodLevels?.length ?? 0) > 0;
      const ladderRngInit = wantLadder ? folRng.clone() : null;
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

      // ---- crown-LOD ladder: prune whole anchors (λ) + coarsen & area-grow survivors -
      if (ladderRngInit && opts?.crownLodLevels) {
        // per-tree keep-key seed: stable + deterministic, derived from a clone so
        // it never disturbs the real RNG order (order-independence law).
        const keepSeed = ladderRngInit.clone().u32() >>> 0;
        foliageLadder = [];
        for (const rung of opts.crownLodLevels) {
          const lambda = rung.lambda;
          const keep = crownLodKeepMask(meshAnchors, lambda, keepSeed);
          const levelRng = ladderRngInit.clone(); // replay the identical stream
          const lg = new MeshGrower();
          // pruned anchors are still BUILT — into a throwaway grower — so they
          // consume the exact same RNG draws; only survivors reach `lg`. This is
          // the "same build path, just a subset of anchors" guarantee. NEVER
          // welds/collapses; a whole anchor (leaf-cluster / needle-spray) is
          // present-at-grown-size or absent — atomic.
          const discard = new MeshGrower();
          // area-preserving survivor growth (§3.3 Eq 3; s = 1/λ exact since λ ≫ r,
          // Eq 7 — one leaf/spray vs the whole crown gives r ≈ 5e-4). Realized as
          // WIDTH-only growth (paper Fig 2c/d), never length/transform ⇒ no spikes.
          const widthMul = lambda > 0 ? 1 / lambda : 1;
          let kept = 0;
          for (let i = 0; i < meshAnchors.length; i++) {
            const anchor = meshAnchors[i] as LeafAnchor;
            const survives = keep[i] as boolean;
            const target = survives ? lg : discard;
            if (survives) kept++;
            if (fol.kind === 'needleSpray') {
              // conifer: element width ×(1/λ) + intra-spray needle prune μ (survivors
              // width ×count/kept) + gentle stem coarsening. Per-spray needle seed is
              // position-hashed (stable across rungs ⇒ nested needle keep-sets).
              const needleSeed = anchorPosHash(anchor.pos.x, anchor.pos.y, anchor.pos.z, keepSeed ^ 0x9e3779b9);
              buildSprayAt(target, anchor, heroLeaf, levelRng, {
                widthMul,
                needleKeep: rung.needleMu < 1 ? { mu: rung.needleMu, seed: needleSeed } : undefined,
                stemSegs: rung.stemSegs,
              });
            } else {
              // broadleaf: leaf width ×(1/λ) + connected-strip row reduction.
              buildLeafCluster(target, anchor, fol.leaf, fol.clusterSize, levelRng, {
                widthMul,
                rows: rung.leafRows,
              });
            }
          }
          // survivor-only normal-bend + crown AO: position-deterministic with the
          // SAME crownC/crownR (recomputed on the grown geometry per rung).
          lg.bendNormals(crownC, crownR, fol.normalBend);
          lg.crownAO(crownC, crownR, 0.55);
          foliageLadder.push({
            lambda,
            geo: lg.build(),
            tris: lg.triCount,
            keptAnchors: kept,
          });
        }
      }
    }
  }

  return {
    bark,
    foliage,
    foliageMesh,
    foliageLadder,
    skeleton: skel,
    stats: {
      tris: barkTris + folTris,
      anchors: skel.anchors.length,
      branches: skel.branches.length,
      height: skel.height,
    },
  };
}
