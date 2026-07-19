/**
 * VegLibrary — boot-time geometry/material pools for the planted world.
 *
 * K=4 structural variants per species (decision D5): each variant grows its
 * own skeleton (own lean/bias/age GrowthInstance), and every LOD ring of a
 * variant derives from the SAME skeleton (seed.rng(label) is stateless per
 * label) — so a ring transition changes triangle cost, never the tree.
 *
 * Pools carry geometry + cull data; the nanite resolve shades every opaque part
 * (bark/deadwood via the shared BarkField array, rock/leaf procedurally), so
 * there are no per-pool material factories. Forests wires instancing/GI/fades.
 */

import { BufferAttribute, BufferGeometry, type DataArrayTexture } from "three";
import { BootTrace, yieldIfDue } from "../debug/BootTrace";
import type { WorldSeed } from "../core/Seed";
import { bakeBarkArray } from "./BarkTexture";
import { TREE_VARIANTS, VegClass } from "../gpu/passes/Scatter";
import { buildLog, buildStump, type DecayState } from "./Deadfall";
import { twigGeometry } from "./GroundCover";
import {
  ETAK_ERRATIC_CLASS,
  generateRock,
  ROCK_LIBRARY,
  type RockMesh,
  type RockVariantSpec,
} from "./RockGen";
import { BootCache } from "../nanite/world/BootCache";
import { DagWorkerPool } from "../nanite/build/DagWorkerClient";
import { TREE_SPECIES } from "./Species";
import { ageForSlot } from "./AgeForm";
import { buildTree, type CrownLodLevel, type CrownLodRung, type HeroDiet } from "./TreeBuilder";
import {
  buildFern,
  buildFlower,
  buildShrub,
  UNDERSTORY_SPECIES,
  type FlowerKind,
} from "./Understory";
// Estonia raised-bog understory (8 QA-approved bog meshes; ScatterMap-only, dormant
// in the generated world). Bog TREES (BogTrees.ts) are intentionally NOT imported —
// their VegClass can't be allocated without a design pass (tree block 0–15 is full).
import { buildCottonGrass, COTTONGRASS_BLADE_TINT, COTTONGRASS_COTTON_TINT, COTTONGRASS_CLS_MAXDIST } from "./bog/CottonGrass";
import { buildHeather, HEATHER_FOLIAGE, HEATHER_BLOSSOM, HEATHER_CLS_MAX_DIST } from "./bog/Heather";
import { buildLabradorTea, LABTEA_FOLIAGE, LABTEA_BLOSSOM, LABTEA_CLS_MAX_DIST } from "./bog/LabradorTea";
import { buildBogRosemary, BOGROSEMARY_FOLIAGE, BOGROSEMARY_BLOSSOM, BOGROSEMARY_CLS_MAX_DIST } from "./bog/BogRosemary";
import { buildCranberry, CRANBERRY_FOLIAGE, CRANBERRY_BERRY, CRANBERRY_CLS_MAX_DIST } from "./bog/Cranberry";
import { buildCloudberry, CLOUDBERRY_FOLIAGE, CLOUDBERRY_BERRY, CLOUDBERRY_CLS_MAX_DIST } from "./bog/Cloudberry";
import { mergeGeo } from "./bog/EricaceousKit";
import type { GrowthInstance, SpeciesParams } from "./VegTypes";

export interface PoolPart {
  geo: BufferGeometry;
  tris: number;
  castShadow: boolean;
}

export interface VegPool {
  cls: number;
  variant: number;
  /** BarkField layer the OPAQUE (parts[0]) bark/deadwood uses — threaded to the
   *  nanite resolve as the texture-array slice (matParam low byte). Undefined for
   *  rock/leaf pools (no bark). */
  barkLayer?: number;
  /** hero ring (trees only): full bark + real mesh leaves, ≤26 m */
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
    /** #113: optional BLOSSOM/berry tint for crowns that merge flowers/berries into
     *  the leaf geo (bog dwarf-shrubs, cotton heads, understory flowers). The leaf
     *  resolve mixes this in on the part-id channel (vdata.x → petal), so petals/
     *  berries read their real colour while leaves/stems keep `color`. Undefined for
     *  plain crowns (trees, ferns) → the resolve's mix is a no-op. */
    blossom?: { r: number; g: number; b: number };
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
  // per-species real-leaf anchor budget for the hero ring. (Trunk bark density is
  // now radius/relief-driven inside TubeMesh — no per-species radial-seg diet.)
  spruce: { meshAnchorTarget: 850 },
  pine: { meshAnchorTarget: 350 },
  beech: { meshAnchorTarget: 2200 },
  birch: { meshAnchorTarget: 4000 },
  karst: { meshAnchorTarget: 4000 },
  larch: { meshAnchorTarget: 550 }, // open airy needle crown → fewer anchors
  oak: { meshAnchorTarget: 2400 }, // broad leaf dome
  // remaining broadleaves build crowns at the shared global leafAnchorTarget (4000)
  // with beech/oak-parity clusterSize [2,3] — full lush canopies (airiness comes from
  // twig DENSITY, not fewer anchors), so they need no HeroDiet entry.
};

export interface VegLib {
  pools: VegPool[];
  /** per-class cull data, indexed by VegClass (covers ETAK_ERRATIC_CLASS 31) */
  clsHeight: number[];
  clsRadius: number[];
  clsMaxDist: number[];
  /** single 6-layer BarkField array (slice == barkLayer) — albedo tone + cavity
   *  + micro-grain normal for the nanite resolve. Macro relief is real geometry. */
  barkTex: DataArrayTexture;
}

/** per-kind leaf-class tint (packLeafTint) for the understory flower pools — the
 *  whole small plant reads as ONE muted herb-layer tint at understory range (the
 *  bloom SHAPE carries the read, not per-part colour; the leaf head has one tint).
 *  Grounded in the Estonia herb palette (goutweed/yarrow umbels, may-lily/hepatica
 *  bells, oxeye-daisy/buttercup composites; understory-communities.toml). */
const FLOWER_TINT: Record<FlowerKind, { r: number; g: number; b: number; hueVar: number }> = {
  umbel: { r: 0.82, g: 0.83, b: 0.75, hueVar: 0.22 }, // bright creamy-white florets (yarrow/goutweed)
  bell: { r: 0.42, g: 0.46, b: 0.74, hueVar: 0.4 }, // saturated periwinkle blue-lilac (hepatica/harebell)
  daisy: { r: 0.86, g: 0.80, b: 0.38, hueVar: 0.36 }, // vivid buttercup/oxeye warm-yellow radiate
};
/** #113: fresh herb-green for the flower STEM + basal leaves (vdata.x=0). FLOWER_TINT
 *  is now the BLOSSOM (petals, vdata.x=1) — the leaf resolve splits them so the bloom
 *  reads its real colour against green foliage instead of one muted whole-plant tint. */
const FLOWER_FOLIAGE = { r: 0.12, g: 0.24, b: 0.07, hueVar: 0.2 };
/** understory fern frond tint — fresh mid forest-green with a HIGH hueVar so the
 *  per-pinnule vdata.x jitter (buildFern) spreads warm↔cool across the frond (a
 *  many-toned green, not one flat colour); the leaf resolve mixes base×warm for
 *  jitter>0 and base×cool for <0, then scales by AO. */
const FERN_TINT = { r: 0.11, g: 0.24, b: 0.06, hueVar: 0.55 };

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

// ---- rock library (SPEC-ROCKS R1): SDF-composed RockGen meshes -------------

/** deterministic RockGen bake of the full §B library — BootCache 'rocks' store
 *  first (rev/params/RockGen-source keyed), else fanned across a DagWorkerPool
 *  ('rock' job kind; per-job sync fallback). Keyed `${class}/${variant}`. */
async function buildRockMeshes(seedNum: number): Promise<Map<string, RockMesh>> {
  const t0 = performance.now();
  type Rec = { key: string; mesh: RockMesh };
  const jobs: { key: string; gridRes: number; v: number; spec: RockVariantSpec }[] = [];
  for (const cls of ROCK_LIBRARY) {
    for (let v = 0; v < cls.variants.length; v++) {
      jobs.push({ key: `${cls.name}/${v}`, gridRes: cls.gridRes, v, spec: cls.variants[v] as RockVariantSpec });
    }
  }
  // the library table IS the per-tier res/count/recipe param set; RockGen.ts
  // itself is in the BootCache SRC_HASH (any generator edit invalidates).
  const cache = new BootCache({ params: { scene: "rocks", seed: seedNum, library: ROCK_LIBRARY } });
  const cached = await cache.get<Rec[]>("rocks");
  let recs: Rec[];
  if (cached && cached.length === jobs.length) {
    recs = cached;
  } else {
    const cores = (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 4;
    let pool: DagWorkerPool | null = null;
    try {
      pool = new DagWorkerPool(Math.max(1, Math.min(6, cores - 2)));
    } catch {
      pool = null; // no Worker (headless) — synchronous fallback below
    }
    recs = await Promise.all(
      jobs.map(async (j): Promise<Rec> => {
        let mesh: RockMesh | null = null;
        if (pool) {
          try {
            mesh = await pool.buildRock({
              archetype: j.spec.archetype,
              variant: j.v,
              seed: seedNum,
              gridRes: j.gridRes,
              mod: j.spec.mod,
              domainScale: j.spec.domainScale ?? 1,
            });
          } catch (e) {
            console.warn(`[rockgen] worker ${j.key} failed — sync fallback:`, e);
          }
        }
        if (!mesh) {
          await yieldIfDue();
          mesh = generateRock(j.spec.archetype, j.v, seedNum, j.gridRes, j.spec.mod, j.spec.domainScale ?? 1);
        }
        return { key: j.key, mesh };
      }),
    );
    pool?.dispose();
    // store AFTER first frame (BootCache put discipline — serialization is
    // main-thread work; the library is ~10 MB of typed arrays)
    void BootTrace.whenFinished().then(() => cache.put("rocks", recs));
  }
  let tris = 0;
  let verts = 0;
  for (const r of recs) {
    tris += r.mesh.stats.tris;
    verts += r.mesh.stats.verts;
  }
  console.log(
    `[rockgen] library: ${recs.length} variants, ${tris} tris / ${verts} verts ` +
      `(${cached ? "cache" : "built"}, ${(performance.now() - t0).toFixed(0)} ms)`,
  );
  return new Map(recs.map((r) => [r.key, r.mesh]));
}

/** RockGen typed arrays → BufferGeometry. vdata u8 → float 0..1 vec4 attribute
 *  (geometryToSource re-quantizes ×255 — an exact u8 round trip). */
function rockGeometry(mesh: RockMesh): BufferGeometry {
  const g = new BufferGeometry();
  g.setAttribute("position", new BufferAttribute(mesh.positions, 3));
  g.setAttribute("normal", new BufferAttribute(mesh.normals, 3));
  const vd = new Float32Array(mesh.vdata.length);
  for (let i = 0; i < vd.length; i++) vd[i] = (mesh.vdata[i] as number) / 255;
  g.setAttribute("vdata", new BufferAttribute(vd, 4));
  g.setIndex(new BufferAttribute(mesh.indices, 1));
  return g;
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
    // #110 age ladder: the K=4 variant slots are a young→old maturity gradient
    // (ageForSlot) so that scale-driven slot selection maps small trees to young
    // FORMS and large trees to old FORMS (Skeleton.ts ontogeny turns this into
    // real proportion differences — crown-base lift, broaden, trunk stoutening —
    // instead of a uniform blow-up). Small per-slot jitter keeps the draw count.
    age: Math.max(0, Math.min(1, ageForSlot(v) + (vr.float() - 0.5) * 0.08)),
  };
}

export async function buildVegLibrary(
  seed: WorldSeed,
  progress: (p: number, msg: string) => void = () => {},
  /** N9-C0: per-crown real-leaf anchor budget for the nanite leaf head (`?naniteleafdensity=N`).
   *  The nanite path has NO cards (D-N3), so the mesh crown must carry the whole canopy.
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
  // ONE 6-layer BarkField array (albedo tone + cavity + micro-grain normal),
  // CPU-baked deterministically. The MACRO furrows/ridges are real displaced
  // geometry in the trunk tubes (TubeMesh), not a texture — so this replaces the
  // old ~535 MB of 2048² GPU-baked bark maps with ≈ 8.4 MB.
  progress(0.2, "veg: baking bark field");
  const barkTex = await bakeBarkArray();

  const pools: VegPool[] = [];
  // sized to the highest VegClass + 1: ETAK_ERRATIC_CLASS = 31 plus the bog
  // understory block (CottonGrass..Cloudberry = 32..37), so 38.
  const clsHeight = new Array<number>(38).fill(1);
  const clsRadius = new Array<number>(38).fill(1);
  const clsMaxDist = new Array<number>(38).fill(150);
  const trackCls = (cls: number, h: number, r: number): void => {
    clsHeight[cls] = Math.max(clsHeight[cls] ?? 1, h);
    clsRadius[cls] = Math.max(clsRadius[cls] ?? 1, r);
  };

  // ---- trees: TREE_SPECIES × 4 variants × (R0 hero, R1, R2 LOD rings) --------
  progress(0.3, "veg: growing tree variant pools");
  const treeParts = (t: ReturnType<typeof buildTree>): PoolPart[] => [
    {
      geo: t.bark,
      tris: t.bark.index ? t.bark.index.count / 3 : 0,
      castShadow: true,
    },
  ];

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
      // hero ring: full tube hierarchy + REAL mesh leaves (the crown the nanite
      // leaf head renders as the WHOLE canopy — no cards in the SW raster, D-N3).
      const t0 = buildTree(sp, seed.rng(label), {
        lod: 0,
        inst,
        // #110: age-dependent growth form (crown-base lift, broaden, trunk
        // stoutening) driven by inst.age — the ladder slot's maturity.
        ageForm: true,
        junctions: junctionsOn,
        foliageMode: "mesh",
        // build the real needle/leaf crown at the global hero anchor density — the canopy fill.
        hero: {
          ...(HERO_DIETS[sp.id] ?? {}),
          meshAnchorTarget: leafAnchorTarget,
        },
        // crown-LOD Phase 2: the ladder is NOT built here — regenerating 4 pruned
        // rungs per crown adds ~17 s to EVERY boot (measured; pine +2.5 s/variant),
        // and it is consumed ONLY to build the crown DAG, which the BootCache stores.
        // So it is built LAZILY (pool.leaf.buildLadder, below) — invoked only on a
        // DAG cache MISS. Warm boots (DAG cached) pay zero, exactly as before Phase 2.
      });
      const t1 = buildTree(sp, seed.rng(label), { lod: 1, inst, ageForm: true, junctions: junctionsOn });
      const t2 = buildTree(sp, seed.rng(label), { lod: 2, inst, ageForm: true, junctions: junctionsOn });
      const r0 = treeParts(t0);
      if (t0.foliageMesh) {
        r0.push({
          geo: t0.foliageMesh,
          tris: t0.foliageMesh.index ? t0.foliageMesh.index.count / 3 : 0,
          // mesh-leaf shadow casting would ~double the caster load for little gain
          castShadow: false,
        });
      }
      const r1 = treeParts(t1);
      const r2 = treeParts(t2);
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
                  ageForm: true,
                  junctions: junctionsOn,
                  foliageMode: "mesh",
                  hero: {
                    ...(HERO_DIETS[spC.id] ?? {}),
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

  // ---- understory: shrubs (bark head + leaf crown), ferns / flowers (leaf-only)
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
      const parts: PoolPart[] = [
        {
          geo: shrub.bark,
          tris: shrub.barkTris,
          castShadow: true,
        },
      ];
      // bounds over bark + crown so the cull sphere covers the leaves.
      const b = bounds(shrub.crown ? [shrub.bark, shrub.crown] : [shrub.bark]);
      trackCls(cls, b.height, b.radius);
      pools.push({
        cls,
        variant: v,
        barkLayer: 2, // shrub opaque part uses bark field layer 2
        r1: parts,
        r2: null,
        trisR1: shrub.barkTris,
        trisR2: 0,
        height: b.height,
        radius: b.radius,
        // real MESH leaf crown — the co-located MATERIAL_CLASS.leaf head (the SAME
        // path the tree hero crown rides), so understory reads as leafy shrubs, not
        // bare stems. WorldRegistry scopes it to understory: capped at clsMaxDist
        // (170 m) with NO tree far-field voxel sibling (short-range dense cover).
        leaf: shrub.crown ? { geo: shrub.crown, tris: shrub.crownTris, color: sp.foliageColor } : undefined,
      });
    }
    clsMaxDist[cls] = 170;
  }
  // ferns: pure-foliage frond rosettes (VegClass.Fern). NO bark stem — the plant
  // IS the crown, so it is registered as a leaf-class PRIMARY head (WorldRegistry),
  // riding the SAME leaf-head path as the tree/shrub crown (two-sided, aggregate
  // DAG, per-species tint via packLeafTint) but bound DIRECTLY to instances. Their
  // instances were scattered all along (Scatter.ts underK / veg.under) and only
  // DROPPED for lack of a pool + non-null classPolicy — this restores them.
  for (let v = 0; v < 4; v++) {
    await yieldIfDue();
    const geo = buildFern(seed.rng(`veg/fern/${v}`));
    const tris = geo.index ? geo.index.count / 3 : 0;
    const b = bounds([geo]);
    trackCls(VegClass.Fern, b.height, b.radius);
    pools.push({
      cls: VegClass.Fern,
      variant: v,
      r1: null,
      r2: null,
      trisR1: 0,
      trisR2: 0,
      height: b.height,
      radius: b.radius,
      leaf: { geo, tris, color: FERN_TINT },
    });
  }
  clsMaxDist[VegClass.Fern] = 140;
  // flowers: thin stalk + real petal geometry (buildFlower), also leaf-class
  // primary. #113: the leaf resolve now splits foliage/blossom by vdata.x, so the
  // stem+basal leaves take a green (FLOWER_FOLIAGE) and the PETALS the bloom colour
  // (FLOWER_TINT) — the muted single-tint plant is gone.
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
        r1: null,
        r2: null,
        trisR1: 0,
        trisR2: 0,
        height: b.height,
        radius: b.radius,
        leaf: { geo, tris, color: FLOWER_FOLIAGE, blossom: FLOWER_TINT[kind] },
      });
    }
    clsMaxDist[cls] = 90;
  }

  // ---- Estonia raised-bog understory (ScatterMap-only; the generated world never
  //      emits classes 32–37, so its byte output is unchanged). Two shapes:
  //      dwarf SHRUBS (bark head + leaf crown, the BushPink path) and leaf-only
  //      FOLIAGE (the fern/flower path). The bog crowns merge flowers/berries with
  //      vdata.x, but the nanite leaf resolve is single-tint (no vdata.x blossom
  //      select — that's the shared #113 follow-up), so each pool takes ONE tint:
  //      the shrubs/berry-plants their FOLIAGE tint; cotton-grass its near-white
  //      COTTON tint (the white hare's-tail head is the plant's read). ----
  progress(0.82, "veg: bog understory pools");
  const bogShrubs = [
    { cls: VegClass.Heather, build: buildHeather, tint: HEATHER_FOLIAGE, blossom: HEATHER_BLOSSOM, maxDist: HEATHER_CLS_MAX_DIST },
    { cls: VegClass.LabradorTea, build: buildLabradorTea, tint: LABTEA_FOLIAGE, blossom: LABTEA_BLOSSOM, maxDist: LABTEA_CLS_MAX_DIST },
    { cls: VegClass.BogRosemary, build: buildBogRosemary, tint: BOGROSEMARY_FOLIAGE, blossom: BOGROSEMARY_BLOSSOM, maxDist: BOGROSEMARY_CLS_MAX_DIST },
  ];
  for (const { cls, build, tint, blossom, maxDist } of bogShrubs) {
    for (let v = 0; v < 4; v++) {
      await yieldIfDue();
      const shrub = build(seed.rng(`veg/bog/shrub/${cls}/${v}`));
      const b = bounds([shrub.bark, shrub.crown]);
      trackCls(cls, b.height, b.radius);
      pools.push({
        cls,
        variant: v,
        barkLayer: 2, // shrub opaque part uses bark field layer 2 (as BushPink)
        r1: [{ geo: shrub.bark, tris: shrub.barkTris, castShadow: true }],
        r2: null,
        trisR1: shrub.barkTris,
        trisR2: 0,
        height: b.height,
        radius: b.radius,
        leaf: { geo: shrub.crown, tris: shrub.crownTris, color: tint, blossom },
      });
    }
    clsMaxDist[cls] = maxDist;
  }
  // cotton-grass: leaf-only, merge the green blade tussock + the white cotton heads
  // into ONE leaf geometry. #113: the leaf resolve splits on vdata.x — blades/culm
  // (x<1) take the green blade tint, the cotton bristle heads (x=1) the near-white
  // blossom, so the hero white head reads against GREEN blades (not a pale plant).
  for (let v = 0; v < 4; v++) {
    await yieldIfDue();
    const cg = buildCottonGrass(seed.rng(`veg/bog/cottongrass/${v}`));
    const geo = mergeGeo([cg.blades, cg.cotton]);
    const tris = geo.index ? geo.index.count / 3 : 0;
    const b = bounds([geo]);
    trackCls(VegClass.CottonGrass, b.height, b.radius);
    pools.push({
      cls: VegClass.CottonGrass,
      variant: v,
      r1: null,
      r2: null,
      trisR1: 0,
      trisR2: 0,
      height: b.height,
      radius: b.radius,
      leaf: { geo, tris, color: COTTONGRASS_BLADE_TINT, blossom: COTTONGRASS_COTTON_TINT },
    });
  }
  clsMaxDist[VegClass.CottonGrass] = COTTONGRASS_CLS_MAXDIST;
  // cranberry / cloudberry: leaf-only, one merged geo. #113: berries (vdata.x=1) now
  // take the BERRY tint via the leaf resolve's split; leaves/stems keep FOLIAGE.
  const bogFoliage = [
    { cls: VegClass.Cranberry, build: buildCranberry, tint: CRANBERRY_FOLIAGE, blossom: CRANBERRY_BERRY, maxDist: CRANBERRY_CLS_MAX_DIST },
    { cls: VegClass.Cloudberry, build: buildCloudberry, tint: CLOUDBERRY_FOLIAGE, blossom: CLOUDBERRY_BERRY, maxDist: CLOUDBERRY_CLS_MAX_DIST },
  ];
  for (const { cls, build, tint, blossom, maxDist } of bogFoliage) {
    for (let v = 0; v < 4; v++) {
      await yieldIfDue();
      const { geo, tris } = build(seed.rng(`veg/bog/foliage/${cls}/${v}`));
      const b = bounds([geo]);
      trackCls(cls, b.height, b.radius);
      pools.push({
        cls,
        variant: v,
        r1: null,
        r2: null,
        trisR1: 0,
        trisR2: 0,
        height: b.height,
        radius: b.radius,
        leaf: { geo, tris, color: tint, blossom },
      });
    }
    clsMaxDist[cls] = maxDist;
  }

  // ---- extras: deadfall -------------------------------------------------------
  progress(0.86, "veg: deadfall pools");
  const decayOf: DecayState[] = ["fresh", "mossy", "rotten", "mossy"];
  for (let v = 0; v < 4; v++) {
    await yieldIfDue();
    const log = buildLog(seed.rng(`veg/log/${v}`), decayOf[v] as DecayState);
    const b = bounds([log.geometry]);
    trackCls(VegClass.Log, b.height, b.radius);
    pools.push({
      cls: VegClass.Log,
      variant: v,
      barkLayer: 5, // deadwood: snag bark (field layer 5)
      r1: [
        {
          geo: log.geometry,
          tris: log.tris,
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
      barkLayer: 5, // deadwood: snag bark (field layer 5)
      r1: [
        {
          geo: stump.geometry,
          tris: stump.tris,
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

  // ---- rocks: SDF-composed RockGen library (SPEC-ROCKS §B, R1) --------------
  // One LOD0 mesh per variant — continuous LOD rides the nanite QEM DAG
  // (dagClasses always carries 'rock'), so there is no discrete r2 ring.
  // Variant semantics preserved for the context-keyed scatter kernels:
  // v0/1 = pale/talus context, v2/3 = dark/mossy context (moss/tone below feed
  // only the OLD-path rockMaterial; the nanite resolve is R2's rockShadeV2).
  progress(0.9, "veg: rock library (RockGen)");
  const rockMeshes = await buildRockMeshes(seed.seed);
  const rockMaxDist: Record<number, number> = {
    [VegClass.Boulder]: 700,
    [VegClass.Slab]: 700,
    [VegClass.StoneL]: 900,
    [VegClass.StoneM]: 280,
    [VegClass.StoneS]: 90,
    // ETAK hero erratics (≥2.5 m): landmark-sized, visible as far as StoneL
    [ETAK_ERRATIC_CLASS]: 900,
  };
  for (const rockCls of ROCK_LIBRARY) {
    for (let v = 0; v < rockCls.variants.length; v++) {
      await yieldIfDue();
      const mesh = rockMeshes.get(`${rockCls.name}/${v}`);
      if (!mesh) throw new Error(`VegLibrary: rock mesh ${rockCls.name}/${v} missing from bake`);
      const geo = rockGeometry(mesh);
      const b = bounds([geo]);
      trackCls(rockCls.classId, b.height, b.radius);
      pools.push({
        cls: rockCls.classId,
        variant: v,
        r1: [
          {
            geo,
            tris: mesh.stats.tris,
            castShadow: rockCls.classId !== VegClass.StoneS,
          },
        ],
        r2: null,
        trisR1: mesh.stats.tris,
        trisR2: 0,
        height: b.height,
        radius: b.radius,
      });
    }
    clsMaxDist[rockCls.classId] = rockMaxDist[rockCls.classId] ?? 150;
  }

  // ---- fallen branches (no-bare-ground layer) --------------------------------
  progress(0.93, "veg: branch pools");
  // fallen branches: scaled twig tubes, deadwood-shaded (dim/moss in the resolve).
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
      barkLayer: 5, // deadwood: snag bark (layer 5)
      r1: [
        {
          geo,
          tris,
          castShadow: false,
        },
      ],
      // clone: a geometry holds ONE indirect slot — sharing it across draws
      // would overwrite the first draw's offset
      r2: [
        {
          geo: geo.clone(),
          tris,
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
    barkTex,
  };
}
