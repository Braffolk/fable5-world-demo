/**
 * VegLibrary — boot-time geometry/material pools for the planted world.
 *
 * K=4 structural variants per species (decision D5): each variant grows its
 * own skeleton (own lean/bias/age GrowthInstance), and every LOD ring of a
 * variant derives from the SAME skeleton (seed.rng(label) is stateless per
 * label) — so a ring transition changes triangle cost, never the tree.
 *
 * Pools carry geometry + a material FACTORY (each indirect draw needs its own
 * material instance for its group-offset uniform); Forests wires instancing,
 * GI, and dither fades on top.
 */

import type { BufferGeometry, DataTexture } from "three";
import { yieldIfDue } from "../debug/BootTrace";
import type { MeshStandardNodeMaterial, Renderer } from "three/webgpu";
import type { WorldSeed } from "../core/Seed";
import {
  bakeBarkArray,
  bakeBarkTextures,
  BARK_TABLE,
  type BarkArrayTextures,
  type BarkTextures,
} from "../gpu/passes/BarkSynth";
import { TREE_VARIANTS, VegClass } from "../gpu/passes/Scatter";
import {
  barkTexturedMaterial,
  deadwoodMaterial,
  flowerMaterial,
  foliageCardMaterial,
  foliageMaterial,
  rockMaterial,
} from "../render/VegMaterials";
import { buildLog, buildStump, type DecayState } from "./Deadfall";
import { captureFoliageAtlas } from "./FoliageCards";
import { twigGeometry } from "./GroundCover";
import { buildRock } from "./RockBuilder";
import { TREE_SPECIES } from "./Species";
import { buildTree, type CrownLodLevel, type CrownLodRung, type HeroDiet } from "./TreeBuilder";
import {
  buildFern,
  buildFlower,
  buildShrub,
  FERN_CAPTURE,
  UNDERSTORY_SPECIES,
  type FlowerKind,
} from "./Understory";
import type { GrowthInstance, SpeciesParams } from "./VegTypes";

export interface PoolPart {
  geo: BufferGeometry;
  tris: number;
  make: () => MeshStandardNodeMaterial;
  castShadow: boolean;
}

export interface VegPool {
  cls: number;
  variant: number;
  /** BARK_TABLE layer the OPAQUE (parts[0]) bark/deadwood texture uses —
   *  threaded to the nanite resolve as the texture-array slice (matParam).
   *  Undefined for rock/leaf pools (no bark texture). */
  barkLayer?: number;
  /** hero ring (trees only): full bark + cards + real mesh leaves, ≤26 m */
  r0?: PoolPart[] | null;
  r1: PoolPart[] | null;
  r2: PoolPart[] | null;
  trisR1: number;
  trisR2: number;
  /** cull-sphere data (from geometry bounds, conservative over parts) */
  height: number;
  radius: number;
  /** N9-C0: hero-ring REAL leaf crown (≤26 m) + its per-species tint, for the
   *  MATERIAL_CLASS.leaf registration — a co-located mesh bound to the SAME
   *  instances as the bark trunk (trunk + crown render together, not LODs). The
   *  tint packs into the leaf head's matParam. Tree pools only (undefined else). */
  leaf?: {
    geo: BufferGeometry;
    tris: number;
    color: { r: number; g: number; b: number; hueVar: number };
    /** crown-LOD Phase 2: LAZY builder for the pre-pruned mesh ladder (finest→
     *  coarsest, λ per CROWN_LOD_SCHEDULE) fed to BuildCrownLodDag as the DAG's LOD
     *  levels. Deterministically REGENERATES the crown (same seed) and returns its
     *  `foliageLadder`; invoked ONLY on a crown-DAG cache MISS (the ~17 s ladder gen
     *  never runs on a warm boot). null when the species built no foliage mesh. */
    buildLadder?: () => CrownLodLevel[] | null;
  };
}

/**
 * Hero-ring tri budgets per species (spec floor: hero tree ≥100k tris for the
 * canopy species — karst gnarl and snags are small/leafless by nature).
 * Measured by tools/herotris.ts; mesh leaves carry the detail, bark radial
 * segs dieted where twig tube counts explode (beech: 24k anchors).
 */
/**
 * Crown-LOD ladder schedule (design §5.2) — one entry per rung, finest →
 * coarsest. Rung 0 is native (λ=1, full detail, growth 1) so it stays
 * byte-identical to LOD0; coarser rungs both PRUNE whole elements (λ) and coarsen
 * the SURVIVORS' internal detail, then GROW survivors to preserve area (§3.3).
 *
 * The reduction is split so the ELEMENT-pruning λ can stay GENTLE (little macro
 * sparsening — the thing the eye reads as "thinning") while the tri cut is carried
 * mostly by INTRA-element detail reduction, which does not empty the silhouette:
 *   - broadleaf: fewer blade `leafRows` at the same leaf envelope, survivor WIDTH
 *     ×(1/λ). (needleMu/stemSegs unused.)
 *   - conifer: intra-spray needle pruning `needleMu` (survivors width ×count/kept,
 *     spray placement untouched) + gentle stem `stemSegs`, survivor WIDTH ×(1/λ).
 *     Spruce/pine share this; conifer λ is gentler than broadleaf (spruce was the
 *     worst historical offender ⇒ least macro pruning). (leafRows unused.)
 *
 * Seed-deterministic; the rung VALUES ride the BootCache key via WorldRegistry's
 * `crownLod` param (CROWN_LOD_SCHEDULE below), and the meshing that consumes them
 * (LeafMesh.ts/TreeBuilder.ts) is in the SRC_HASH ⇒ any change auto-invalidates.
 */
/** measured per-rung fraction of LOD0 tris (node validation): 6 rungs, the ladder
 *  lands ≈ [1.0, 0.70, 0.46, 0.41, 0.24, 0.145]. tri math: a leaf = 4·rows+3 tris
 *  (LOD0 rows4 = 19); with rows clamped ≥2 the row lever bottoms at 11/19 = 0.579×
 *  by rung 2, so rungs 3-5 carry the cut with the ELEMENT-λ (whole 2-4-leaf clusters,
 *  survivor WIDTH ×1/λ preserving area). Rung 5 λ=0.25 ⇒ 0.25·0.579 = 0.145× ≤ the
 *  0.15 target that lets errorScale actually reach the ~3-5M mid reference band (the
 *  4-rung ladder's 0.41 coarsest SATURATED at 9.46M — MID-DECOMPOSITION 2026-07-09).
 *  Deep-rung fat-leaf px @ engagement (projK≈1143 px·m, world leaf w≈0.08 m): rung 5
 *  ×4.0 width = 0.32 m ⇒ ~6.6 px at 55 m (native ~1.7 px) — the aggressive regime;
 *  EYEBALL leaf chunkiness at 50 m. rows stays ≥2 (mid-leaf width peak always sampled).
 *  HANDOFF WIDTH-OVERSHOOT RAMP (2026-07-09): rungs 4/5 carry widthBoost 1.15/1.3 on top
 *  of 1/λ ⇒ TOTAL width ×2.74 (rung 4) / ×5.2 (rung 5); at the K=0.4 bake these deep rungs
 *  own ~17-60 m, so the boost walks the mesh density UP to the conservatively-voxelized
 *  sibling at the 60 m handoff (mesh at ~59 m was reading thinner than voxel at 61 m). */
const CROWN_LOD_BROADLEAF: readonly CrownLodRung[] = [
  { lambda: 1.0, leafRows: 4, needleMu: 1, stemSegs: 4, widthBoost: 1.0 },
  { lambda: 0.88, leafRows: 3, needleMu: 1, stemSegs: 4, widthBoost: 1.0 },
  { lambda: 0.8, leafRows: 2, needleMu: 1, stemSegs: 4, widthBoost: 1.0 },
  { lambda: 0.7, leafRows: 2, needleMu: 1, stemSegs: 4, widthBoost: 1.0 },
  { lambda: 0.42, leafRows: 2, needleMu: 1, stemSegs: 4, widthBoost: 1.15 },
  { lambda: 0.25, leafRows: 2, needleMu: 1, stemSegs: 4, widthBoost: 1.3 },
];

/** conifer ladder ≈ [1.0, 0.72, 0.51, 0.37, 0.25, 0.13] of LOD0 tris (6 rungs). tri
 *  math: a spray = 2·(stemSegs + round(μ·C)) tris (spruce C=90 needles, LOD0=188;
 *  pine C=264, LOD0=536), so frac ≈ λ·μ + a small stem term. The needle-μ lever
 *  carries most of the cut so the whole-spray λ stays gentle (spruce was the worst
 *  historical spike offender ⇒ least MACRO sparsening): rung 5 λ=0.55 keeps 55% of
 *  sprays. rung 5 (λ0.55,μ0.22,seg2) measures ≈ 0.13× (spruce) / 0.12× (pine) ≤ 0.15.
 *  Fat-needle px @ engagement (projK≈1143 px·m, native needle w=0.024 m): area-preserve
 *  width = ×(1/λ)·(count/kept) ≈ ×8.2 at rung 5 ⇒ 0.20 m ⇒ ~4.1 px at 55 m (native
 *  ~0.5 px, sub-pixel). Needles fill the same spray footprint (fewer, wider) — the
 *  Cook aggressive regime; EYEBALL spruce 40-60 m for blobbiness. LENGTH never scaled
 *  (elongation = the spike) — width-only.
 *  HANDOFF WIDTH-OVERSHOOT RAMP (2026-07-09): rungs 4/5 carry widthBoost 1.15/1.3 on top
 *  of (1/λ·count/kept) ⇒ TOTAL width ≈ ×4.7 (rung 4) / ×10.7 (rung 5); walks the deep-rung
 *  mesh needle density up to the voxel sibling at the 60 m handoff (K=0.4 bake ⇒ rung 5
 *  owns ~20-60 m). */
const CROWN_LOD_CONIFER: readonly CrownLodRung[] = [
  { lambda: 1.0, leafRows: 4, needleMu: 1.0, stemSegs: 4, widthBoost: 1.0 },
  { lambda: 0.92, leafRows: 4, needleMu: 0.78, stemSegs: 4, widthBoost: 1.0 },
  { lambda: 0.85, leafRows: 4, needleMu: 0.6, stemSegs: 3, widthBoost: 1.0 },
  { lambda: 0.8, leafRows: 4, needleMu: 0.46, stemSegs: 2, widthBoost: 1.0 },
  { lambda: 0.70, leafRows: 4, needleMu: 0.35, stemSegs: 2, widthBoost: 1.15 },
  { lambda: 0.55, leafRows: 4, needleMu: 0.22, stemSegs: 2, widthBoost: 1.3 },
];

/** the rung schedule for a species (conifer vs broadleaf lever set). */
function crownLodScheduleFor(sp: SpeciesParams): readonly CrownLodRung[] {
  return sp.kind === 'conifer' ? CROWN_LOD_CONIFER : CROWN_LOD_BROADLEAF;
}

/** cache-key witness: BOTH per-kind schedules, JSON-serialized into the crown-DAG
 *  BootCache key (WorldRegistry) so ANY rung edit invalidates stale crowns. The
 *  rung COUNT (both length 6) also sets the DAG ladder depth. */
export const CROWN_LOD_SCHEDULE = {
  broadleaf: CROWN_LOD_BROADLEAF,
  conifer: CROWN_LOD_CONIFER,
} as const;

export const HERO_DIETS: Record<string, HeroDiet> = {
  // cards stay UNTHINNED at hero range: thinning enlarges the survivors
  // (sqrt-coverage rule) and a 1.65×-size card 4 m away is a giant flat
  // sheet — full-count original-size cards + mesh leaves is the gallery look
  spruce: { meshAnchorTarget: 850, barkK: 0.8 },
  pine: { meshAnchorTarget: 350, barkK: 0.8 },
  beech: { meshAnchorTarget: 2200, barkK: 0.5 },
  birch: { meshAnchorTarget: 4000, barkK: 1 },
  karst: { meshAnchorTarget: 4000, barkK: 1.1 },
  snag: { barkK: 1.3 },
};

export interface VegLib {
  pools: VegPool[];
  /** per-class cull data, indexed by VegClass (length 20) */
  clsHeight: number[];
  clsRadius: number[];
  clsMaxDist: number[];
  atlases: Map<string, DataTexture>;
  barks: Map<number, BarkTextures>;
  /** bark texture-array (slice == BARK_TABLE layer) for the nanite resolve */
  barkArray: BarkArrayTextures;
}

const FLOWER_COLOR: Record<FlowerKind, { r: number; g: number; b: number }> = {
  umbel: { r: 0.75, g: 0.75, b: 0.7 },
  bell: { r: 0.28, g: 0.14, b: 0.5 },
  daisy: { r: 0.85, g: 0.72, b: 0.12 },
};

function bounds(geos: BufferGeometry[]): { height: number; radius: number } {
  let height = 0.5;
  let radius = 0.5;
  for (const g of geos) {
    g.computeBoundingBox();
    g.computeBoundingSphere();
    const bb = g.boundingBox;
    const bs = g.boundingSphere;
    if (bb) height = Math.max(height, bb.max.y);
    if (bs) radius = Math.max(radius, bs.center.length() + bs.radius);
  }
  return { height, radius };
}

function variantInstance(
  seed: WorldSeed,
  id: string,
  v: number,
): Partial<GrowthInstance> {
  const vr = seed.rng(`veginst/${id}/${v}`);
  return {
    leanX: (vr.float() - 0.5) * 0.14,
    leanZ: (vr.float() - 0.5) * 0.14,
    biasX: (vr.float() - 0.5) * 1.6,
    biasZ: (vr.float() - 0.5) * 1.6,
    age: 0.7 + vr.float() * 0.3,
  };
}

export async function buildVegLibrary(
  renderer: Renderer,
  seed: WorldSeed,
  progress: (p: number, msg: string) => void = () => {},
  /** N9-C0: per-crown real-leaf anchor budget for the nanite leaf head (`?naniteleafdensity=N`).
   *  The card-era hero leaned on all-anchor cards for fill, so the mesh crown was a sparse
   *  detail layer (~850); the nanite path has NO cards (D-N3), so it must carry the crown.
   *  Full density (all anchors) is ~827 MB / the D-N41 cluster wall — the aggregate (C2) is the
   *  real fix; this caps it to a stable hero density. Default 2500; higher = fuller + heavier. */
  opts?: {
    leafAnchorTarget?: number;
  },
): Promise<VegLib> {
  // N9-C0: per-crown real-leaf anchor budget for the nanite leaf head. Default 4000
  // — the density the user signed off on for spruce + pine; ?naniteleafdensity=N dials it.
  const leafAnchorTarget = opts?.leafAnchorTarget ?? 4000;
  // ---- shared captures -------------------------------------------------------
  progress(0, "veg: capturing foliage atlases");
  const atlases = new Map<string, DataTexture>();
  for (const sp of [...TREE_SPECIES, ...UNDERSTORY_SPECIES, FERN_CAPTURE]) {
    if (!sp.foliage || atlases.has(sp.id)) continue;
    atlases.set(
      sp.id,
      await captureFoliageAtlas(renderer, sp, seed.rng(`cards/${sp.id}`)),
    );
  }
  progress(0.2, "veg: baking bark textures");
  const barks = new Map<number, BarkTextures>();
  const layers = new Set<number>([
    ...TREE_SPECIES.map((s) => s.barkLayer),
    2,
    5,
  ]);
  for (const layer of layers) {
    barks.set(
      layer,
      await bakeBarkTextures(renderer, layer, seed.sub(`bark/${layer}`) % 977),
    );
  }
  const barkOf = (layer: number): BarkTextures => {
    const b = barks.get(layer);
    if (!b) throw new Error(`bark layer ${layer} not baked`);
    return b;
  };
  // nanite resolve sampled-array: every BARK_TABLE layer, same per-layer seedK
  // as the 2D bake above (so the array is visually identical to the old path).
  const barkArray = await bakeBarkArray(
    renderer,
    BARK_TABLE.map((_, layer) => seed.sub(`bark/${layer}`) % 977),
  );

  const pools: VegPool[] = [];
  const clsHeight = new Array<number>(24).fill(1);
  const clsRadius = new Array<number>(24).fill(1);
  const clsMaxDist = new Array<number>(24).fill(150);
  const trackCls = (cls: number, h: number, r: number): void => {
    clsHeight[cls] = Math.max(clsHeight[cls] ?? 1, h);
    clsRadius[cls] = Math.max(clsRadius[cls] ?? 1, r);
  };

  // ---- trees: 6 species × 4 variants × (R1 cards, R2 branch-cards) ----------
  progress(0.3, "veg: growing tree variant pools");
  const treeParts = (
    sp: SpeciesParams,
    t: ReturnType<typeof buildTree>,
  ): PoolPart[] => {
    const parts: PoolPart[] = [
      {
        geo: t.bark,
        tris: t.bark.index ? t.bark.index.count / 3 : 0,
        make: () => barkTexturedMaterial(barkOf(sp.barkLayer)),
        castShadow: true,
      },
    ];
    const atlas = atlases.get(sp.id);
    if (t.foliage && atlas) {
      parts.push({
        geo: t.foliage,
        tris: t.foliage.index ? t.foliage.index.count / 3 : 0,
        make: () => foliageCardMaterial(atlas, { color: sp.foliageColor }),
        castShadow: true,
      });
    }
    return parts;
  };

  // ?nojunctions — G5 A/B ablation: legacy independent open-tube bark (no welded
  // junctions). DEFAULT is the connected-junction rework (junctions on).
  const junctionsOn =
    (typeof window !== "undefined"
      ? new URLSearchParams(window.location.search).get("nojunctions")
      : null) === null;

  for (let ci = 0; ci < TREE_SPECIES.length; ci++) {
    const sp = TREE_SPECIES[ci] as SpeciesParams;
    for (let v = 0; v < TREE_VARIANTS; v++) {
      // cold-boot: hero buildTree calls are 100k-tri slabs — yield ~every 250 ms
      // so this loop interleaves with the GPU boot phases (overlap kick) and
      // never freezes the tab
      await yieldIfDue();
      const label = `veg/${sp.id}/${v}`;
      const inst = variantInstance(seed, sp.id, v);
      // hero ring: full tube hierarchy + thinned cards + REAL mesh leaves.
      // Cards stay in the hero so the R0↔R1 swap only adds leaf geometry —
      // the painted silhouette never changes (no pop).
      const t0 = buildTree(sp, seed.rng(label), {
        lod: 0,
        inst,
        junctions: junctionsOn,
        foliageMode: "hybrid",
        // N9-C0: the nanite leaf head renders foliageMesh as the WHOLE crown — there
        // are NO cards in the SW raster (alpha-test, D-N3). The card-era
        // meshAnchorTarget (a sparse detail layer ON TOP of all-anchor cards) read as
        // a near-bare tree through nanite, so build the real needle/leaf crown at
        // FULL anchor density to match the old card coverage. (cardTarget still shapes
        // the card foliage build; only the mesh anchors densify.)
        hero: {
          ...(HERO_DIETS[sp.id] ?? { cardTarget: 1500 }),
          meshAnchorTarget: leafAnchorTarget,
        },
        // crown-LOD Phase 2: the ladder is NOT built here — regenerating 4 pruned
        // rungs per crown adds ~17 s to EVERY boot (measured; pine +2.5 s/variant),
        // and it is consumed ONLY to build the crown DAG, which the BootCache stores.
        // So it is built LAZILY (pool.leaf.buildLadder, below) — invoked only on a
        // DAG cache MISS. Warm boots (DAG cached) pay zero, exactly as before Phase 2.
      });
      const t1 = buildTree(sp, seed.rng(label), { lod: 1, inst, junctions: junctionsOn });
      const t2 = buildTree(sp, seed.rng(label), { lod: 2, inst, junctions: junctionsOn });
      const r0 = treeParts(sp, t0);
      if (t0.foliageMesh) {
        r0.push({
          geo: t0.foliageMesh,
          tris: t0.foliageMesh.index ? t0.foliageMesh.index.count / 3 : 0,
          make: () => foliageMaterial({ color: sp.foliageColor }),
          // cards already cast equivalent crown coverage — mesh-leaf shadow
          // casting would double the caster load for no visible gain
          castShadow: false,
        });
      }
      const r1 = treeParts(sp, t1);
      const r2 = treeParts(sp, t2);
      const b = bounds(r1.map((p) => p.geo));
      trackCls(ci, b.height, b.radius);
      pools.push({
        cls: ci,
        variant: v,
        barkLayer: sp.barkLayer,
        r0,
        r1,
        r2,
        trisR1: t1.stats.tris,
        trisR2: t2.stats.tris,
        height: b.height,
        radius: b.radius,
        // N9-C0: the same real mesh-leaf crown pushed into r0 above, exposed for
        // the leaf MATERIAL_CLASS registration (the nanite leaf head repacks this
        // geometry separately).
        leaf: t0.foliageMesh
          ? {
              geo: t0.foliageMesh,
              tris: t0.foliageMesh.index ? t0.foliageMesh.index.count / 3 : 0,
              color: sp.foliageColor,
              // crown-LOD Phase 2: lazy ladder regen (DAG-cache-miss only). Rebuilds
              // this crown WITH the LOD schedule using the SAME seed/variant, so the
              // λ=1 rung matches this pool's LOD0 crown deterministically. `sp`, `inst`,
              // `label`, and the hero diet are captured from this pool's build above.
              buildLadder: ((spC, rngLabel, instC) => (): CrownLodLevel[] | null =>
                buildTree(spC, seed.rng(rngLabel), {
                  lod: 0,
                  inst: instC,
                  junctions: junctionsOn,
                  foliageMode: "hybrid",
                  hero: {
                    ...(HERO_DIETS[spC.id] ?? { cardTarget: 1500 }),
                    meshAnchorTarget: leafAnchorTarget,
                  },
                  crownLodLevels: crownLodScheduleFor(spC),
                }).foliageLadder)(sp, label, inst),
            }
          : undefined,
      });
    }
    clsMaxDist[ci] = 1e8; // trees continue as impostors
    progress(
      0.3 + 0.25 * ((ci + 1) / TREE_SPECIES.length),
      `veg: ${sp.id} pool`,
    );
  }

  // ---- understory: shrubs / fern / flowers (R1 only) -------------------------
  progress(0.76, "veg: understory pools");
  const underSpecies = [
    { cls: VegClass.BushHazel, sp: UNDERSTORY_SPECIES[0] as SpeciesParams },
    { cls: VegClass.BushPink, sp: UNDERSTORY_SPECIES[1] as SpeciesParams },
    { cls: VegClass.Juniper, sp: UNDERSTORY_SPECIES[2] as SpeciesParams },
  ];
  for (const { cls, sp } of underSpecies) {
    for (let v = 0; v < 4; v++) {
      await yieldIfDue();
      const rng = seed.rng(`veg/${sp.id}/${v}`);
      const shrub = buildShrub(sp, rng);
      const atlas = atlases.get(sp.id);
      const parts: PoolPart[] = [
        {
          geo: shrub.bark,
          tris: shrub.bark.index ? shrub.bark.index.count / 3 : 0,
          make: () => barkTexturedMaterial(barkOf(2)),
          castShadow: true,
        },
      ];
      if (shrub.foliage && atlas) {
        parts.push({
          geo: shrub.foliage,
          tris: shrub.foliage.index ? shrub.foliage.index.count / 3 : 0,
          make: () => foliageCardMaterial(atlas, { color: sp.foliageColor }),
          castShadow: true,
        });
      }
      const b = bounds(parts.map((p) => p.geo));
      trackCls(cls, b.height, b.radius);
      pools.push({
        cls,
        variant: v,
        barkLayer: 2, // shrub opaque part uses barkOf(2) above
        r1: parts,
        r2: null,
        trisR1: shrub.tris,
        trisR2: 0,
        height: b.height,
        radius: b.radius,
      });
    }
    clsMaxDist[cls] = 170;
  }
  // ferns
  const fernAtlas = atlases.get("fern");
  for (let v = 0; v < 4; v++) {
    await yieldIfDue();
    const geo = buildFern(seed.rng(`veg/fern/${v}`));
    const tris = geo.index ? geo.index.count / 3 : 0;
    const b = bounds([geo]);
    trackCls(VegClass.Fern, b.height, b.radius);
    pools.push({
      cls: VegClass.Fern,
      variant: v,
      r1: fernAtlas
        ? [
            {
              geo,
              tris,
              make: () =>
                foliageCardMaterial(fernAtlas, {
                  color: FERN_CAPTURE.foliageColor,
                }),
              castShadow: false,
            },
          ]
        : null,
      r2: null,
      trisR1: tris,
      trisR2: 0,
      height: b.height,
      radius: b.radius,
    });
  }
  clsMaxDist[VegClass.Fern] = 140;
  // flowers
  const flowerKinds: { cls: number; kind: FlowerKind }[] = [
    { cls: VegClass.FlowerUmbel, kind: "umbel" },
    { cls: VegClass.FlowerBell, kind: "bell" },
    { cls: VegClass.FlowerDaisy, kind: "daisy" },
  ];
  for (const { cls, kind } of flowerKinds) {
    for (let v = 0; v < 4; v++) {
      await yieldIfDue();
      const geo = buildFlower(kind, seed.rng(`veg/flower/${kind}/${v}`));
      const tris = geo.index ? geo.index.count / 3 : 0;
      const b = bounds([geo]);
      trackCls(cls, b.height, b.radius);
      pools.push({
        cls,
        variant: v,
        r1: [
          {
            geo,
            tris,
            make: () => flowerMaterial(FLOWER_COLOR[kind]),
            castShadow: false,
          },
        ],
        r2: null,
        trisR1: tris,
        trisR2: 0,
        height: b.height,
        radius: b.radius,
      });
    }
    clsMaxDist[cls] = 90;
  }

  // ---- extras: deadfall + boulders/slabs -------------------------------------
  progress(0.86, "veg: deadfall + boulder pools");
  const deadTex = barkOf(5);
  // weathered-wood darkening: the snag bark bake is pale gray and logs read
  // as glowing white slivers in noon sun without it
  const logDim = { r: 0.6, g: 0.52, b: 0.44 };
  const decayOf: DecayState[] = ["fresh", "mossy", "rotten", "mossy"];
  for (let v = 0; v < 4; v++) {
    await yieldIfDue();
    const log = buildLog(seed.rng(`veg/log/${v}`), decayOf[v] as DecayState);
    const b = bounds([log.geometry]);
    trackCls(VegClass.Log, b.height, b.radius);
    pools.push({
      cls: VegClass.Log,
      variant: v,
      barkLayer: 5, // deadwood: snag bark (barkOf(5))
      r1: [
        {
          geo: log.geometry,
          tris: log.tris,
          make: () => deadwoodMaterial(deadTex, logDim),
          castShadow: true,
        },
      ],
      r2: null,
      trisR1: log.tris,
      trisR2: 0,
      height: b.height,
      radius: b.radius,
    });
  }
  clsMaxDist[VegClass.Log] = 220;
  for (let v = 0; v < 4; v++) {
    await yieldIfDue();
    const stump = buildStump(seed.rng(`veg/stump/${v}`));
    const b = bounds([stump.geometry]);
    trackCls(VegClass.Stump, b.height, b.radius);
    pools.push({
      cls: VegClass.Stump,
      variant: v,
      barkLayer: 5, // deadwood: snag bark (barkOf(5))
      r1: [
        {
          geo: stump.geometry,
          tris: stump.tris,
          make: () => deadwoodMaterial(deadTex, logDim),
          castShadow: true,
        },
      ],
      r2: null,
      trisR1: stump.tris,
      trisR2: 0,
      height: b.height,
      radius: b.radius,
    });
  }
  clsMaxDist[VegClass.Stump] = 170;

  const rockPools: { cls: number; preset: "boulder" | "slab"; moss: number }[] =
    [
      { cls: VegClass.Boulder, preset: "boulder", moss: 0.3 },
      { cls: VegClass.Slab, preset: "slab", moss: 0.12 },
    ];
  // scatter keys boulder/slab variants by rock exposure: 0/1 = pale bedrock
  // blocks beside cliffs (matching them), 2/3 = dark mossy forest rocks
  const paleRock = { r: 0.34, g: 0.33, b: 0.3 };
  for (const { cls, preset, moss } of rockPools) {
    for (let v = 0; v < 4; v++) {
      await yieldIfDue();
      const tone = v < 2 ? paleRock : undefined;
      const vMoss = v < 2 ? 0.08 : moss;
      const hi = buildRock(preset, seed.rng(`veg/${preset}/${v}`), 4);
      const lo = buildRock(preset, seed.rng(`veg/${preset}/${v}`), 3);
      const b = bounds([hi.geometry]);
      trackCls(cls, b.height, b.radius);
      pools.push({
        cls,
        variant: v,
        r1: [
          {
            geo: hi.geometry,
            tris: hi.stats.tris,
            make: () => rockMaterial({ moss: vMoss, tone }),
            castShadow: true,
          },
        ],
        r2: [
          {
            geo: lo.geometry,
            tris: lo.stats.tris,
            make: () => rockMaterial({ moss: vMoss, tone }),
            castShadow: true,
          },
        ],
        trisR1: hi.stats.tris,
        trisR2: lo.stats.tris,
        height: b.height,
        radius: b.radius,
      });
    }
    clsMaxDist[cls] = 700;
  }

  // ---- size-stratified stones + fallen branches (no-bare-ground layer) ------
  progress(0.93, "veg: stone/branch pools");
  const stoneClasses: {
    cls: number;
    preset: "boulder" | "cobble";
    d1: number;
    d2: number | null;
    moss: number;
    maxDist: number;
  }[] = [
    {
      cls: VegClass.StoneL,
      preset: "boulder",
      d1: 3,
      d2: 2,
      moss: 0.22,
      maxDist: 900,
    },
    {
      cls: VegClass.StoneM,
      preset: "cobble",
      d1: 2,
      d2: 1,
      moss: 0.12,
      maxDist: 280,
    },
    {
      cls: VegClass.StoneS,
      preset: "cobble",
      d1: 1,
      d2: null,
      moss: 0.06,
      maxDist: 90,
    },
  ];
  for (const sc of stoneClasses) {
    for (let v = 0; v < 4; v++) {
      await yieldIfDue();
      // StoneL variants are context-keyed by the scatter kernel: 0/1 spawn
      // on dry scree (pale faceted talus matching the cliff that shed it),
      // 2/3 in streambeds (dark water-rounded, mossy) — scree stops reading
      // as smooth dark blobs
      const isTalus = sc.cls === VegClass.StoneL && v < 2;
      const preset =
        sc.cls === VegClass.StoneL
          ? isTalus
            ? "talus"
            : "boulder"
          : sc.preset;
      const moss =
        sc.cls === VegClass.StoneL ? (isTalus ? 0.06 : 0.3) : sc.moss;
      const tone = isTalus ? { r: 0.35, g: 0.34, b: 0.31 } : undefined;
      const hi = buildRock(preset, seed.rng(`veg/stone${sc.cls}/${v}`), sc.d1);
      const lo =
        sc.d2 !== null
          ? buildRock(preset, seed.rng(`veg/stone${sc.cls}/${v}`), sc.d2)
          : null;
      const b = bounds([hi.geometry]);
      trackCls(sc.cls, b.height, b.radius);
      pools.push({
        cls: sc.cls,
        variant: v,
        r1: [
          {
            geo: hi.geometry,
            tris: hi.stats.tris,
            make: () => rockMaterial({ moss, tone }),
            castShadow: sc.cls !== VegClass.StoneS,
          },
        ],
        r2: lo
          ? [
              {
                geo: lo.geometry,
                tris: lo.stats.tris,
                make: () => rockMaterial({ moss, tone }),
                castShadow: sc.cls === VegClass.StoneL,
              },
            ]
          : null,
        trisR1: hi.stats.tris,
        trisR2: lo ? lo.stats.tris : 0,
        height: b.height,
        radius: b.radius,
      });
    }
    clsMaxDist[sc.cls] = sc.maxDist;
  }
  // fallen branches: scaled twig tubes, deadwood-shaded. Dimmed hard: the
  // snag-bark albedo is pale gray and read as glowing white sticks at noon.
  const branchDim = { r: 0.5, g: 0.42, b: 0.34 };
  for (let v = 0; v < 4; v++) {
    await yieldIfDue();
    const geo = twigGeometry(seed.rng(`veg/branch/${v}`));
    geo.scale(6.5, 5, 6.5);
    const tris = geo.index ? geo.index.count / 3 : 0;
    const b = bounds([geo]);
    trackCls(VegClass.Branch, b.height, b.radius);
    pools.push({
      cls: VegClass.Branch,
      variant: v,
      barkLayer: 5, // deadwood: snag bark (barkOf(5))
      r1: [
        {
          geo,
          tris,
          make: () => deadwoodMaterial(deadTex, branchDim),
          castShadow: false,
        },
      ],
      // clone: a geometry holds ONE indirect slot — sharing it across draws
      // would overwrite the first draw's offset
      r2: [
        {
          geo: geo.clone(),
          tris,
          make: () => deadwoodMaterial(deadTex, branchDim),
          castShadow: false,
        },
      ],
      trisR1: tris,
      trisR2: tris,
      height: b.height,
      radius: b.radius,
    });
  }
  clsMaxDist[VegClass.Branch] = 230;

  progress(1, "veg: pools ready");
  return {
    pools,
    clsHeight,
    clsRadius,
    clsMaxDist,
    atlases,
    barks,
    barkArray,
  };
}
