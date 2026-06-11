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

import type { BufferGeometry, DataTexture } from 'three';
import type { MeshStandardNodeMaterial, Renderer } from 'three/webgpu';
import type { WorldSeed } from '../core/Seed';
import { bakeBarkTextures, type BarkTextures } from '../gpu/passes/BarkSynth';
import { TREE_VARIANTS, VegClass } from '../gpu/passes/Scatter';
import {
  barkTexturedMaterial,
  deadwoodMaterial,
  flowerMaterial,
  foliageCardMaterial,
  rockMaterial,
} from '../render/VegMaterials';
import { buildLog, buildStump, type DecayState } from './Deadfall';
import { captureFoliageAtlas } from './FoliageCards';
import { twigGeometry } from './GroundCover';
import { captureImpostor, type ImpostorAtlas, type ImpostorPart } from './Impostors';
import { buildRock } from './RockBuilder';
import { TREE_SPECIES } from './Species';
import { buildTree } from './TreeBuilder';
import {
  buildFern,
  buildFlower,
  buildShrub,
  FERN_CAPTURE,
  UNDERSTORY_SPECIES,
  type FlowerKind,
} from './Understory';
import type { GrowthInstance, SpeciesParams } from './VegTypes';

export interface PoolPart {
  geo: BufferGeometry;
  tris: number;
  make: () => MeshStandardNodeMaterial;
  castShadow: boolean;
}

export interface VegPool {
  cls: number;
  variant: number;
  r1: PoolPart[] | null;
  r2: PoolPart[] | null;
  trisR1: number;
  trisR2: number;
  /** cull-sphere data (from geometry bounds, conservative over parts) */
  height: number;
  radius: number;
}

export interface VegLib {
  pools: VegPool[];
  /** tree species cls → octahedral impostor atlas (captured from variant 0) */
  impostors: Map<number, ImpostorAtlas>;
  /** per-class cull data, indexed by VegClass (length 20) */
  clsHeight: number[];
  clsRadius: number[];
  clsMaxDist: number[];
  atlases: Map<string, DataTexture>;
  barks: Map<number, BarkTextures>;
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

function variantInstance(seed: WorldSeed, id: string, v: number): Partial<GrowthInstance> {
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
): Promise<VegLib> {
  // ---- shared captures -------------------------------------------------------
  progress(0, 'veg: capturing foliage atlases');
  const atlases = new Map<string, DataTexture>();
  for (const sp of [...TREE_SPECIES, ...UNDERSTORY_SPECIES, FERN_CAPTURE]) {
    if (!sp.foliage || atlases.has(sp.id)) continue;
    atlases.set(sp.id, await captureFoliageAtlas(renderer, sp, seed.rng(`cards/${sp.id}`)));
  }
  progress(0.2, 'veg: baking bark textures');
  const barks = new Map<number, BarkTextures>();
  const layers = new Set<number>([...TREE_SPECIES.map((s) => s.barkLayer), 2, 5]);
  for (const layer of layers) {
    barks.set(layer, await bakeBarkTextures(renderer, layer, seed.sub(`bark/${layer}`) % 977));
  }
  const barkOf = (layer: number): BarkTextures => {
    const b = barks.get(layer);
    if (!b) throw new Error(`bark layer ${layer} not baked`);
    return b;
  };

  const pools: VegPool[] = [];
  const clsHeight = new Array<number>(24).fill(1);
  const clsRadius = new Array<number>(24).fill(1);
  const clsMaxDist = new Array<number>(24).fill(150);
  const trackCls = (cls: number, h: number, r: number): void => {
    clsHeight[cls] = Math.max(clsHeight[cls] ?? 1, h);
    clsRadius[cls] = Math.max(clsRadius[cls] ?? 1, r);
  };

  // ---- trees: 6 species × 4 variants × (R1 cards, R2 branch-cards) ----------
  progress(0.3, 'veg: growing tree variant pools');
  const treeParts = (sp: SpeciesParams, t: ReturnType<typeof buildTree>): PoolPart[] => {
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

  for (let ci = 0; ci < TREE_SPECIES.length; ci++) {
    const sp = TREE_SPECIES[ci] as SpeciesParams;
    for (let v = 0; v < TREE_VARIANTS; v++) {
      const label = `veg/${sp.id}/${v}`;
      const inst = variantInstance(seed, sp.id, v);
      const t1 = buildTree(sp, seed.rng(label), { lod: 1, inst });
      const t2 = buildTree(sp, seed.rng(label), { lod: 2, inst });
      const r1 = treeParts(sp, t1);
      const r2 = treeParts(sp, t2);
      const b = bounds(r1.map((p) => p.geo));
      trackCls(ci, b.height, b.radius);
      pools.push({
        cls: ci,
        variant: v,
        r1,
        r2,
        trisR1: t1.stats.tris,
        trisR2: t2.stats.tris,
        height: b.height,
        radius: b.radius,
      });
    }
    clsMaxDist[ci] = 1e8; // trees continue as impostors
    progress(0.3 + 0.25 * ((ci + 1) / TREE_SPECIES.length), `veg: ${sp.id} pool`);
  }

  // ---- tree impostors (variant 0 R1 geometry, relightable octahedral) -------
  progress(0.56, 'veg: capturing octahedral impostors');
  const impostors = new Map<number, ImpostorAtlas>();
  for (let ci = 0; ci < TREE_SPECIES.length; ci++) {
    const sp = TREE_SPECIES[ci] as SpeciesParams;
    const t = buildTree(sp, seed.rng(`veg/${sp.id}/0`), {
      lod: 1,
      inst: variantInstance(seed, sp.id, 0),
    });
    const parts: ImpostorPart[] = [
      { geometry: t.bark, kind: 'bark', barkTex: barkOf(sp.barkLayer) },
    ];
    const atlas = atlases.get(sp.id);
    if (t.foliage && atlas) parts.push({ geometry: t.foliage, kind: 'cards', atlas });
    const radius = Math.max(
      t.stats.height * 0.55,
      t.skeleton.crownRadius * 1.4,
      2,
    );
    impostors.set(
      ci,
      await captureImpostor(renderer, parts, { centerY: t.stats.height * 0.5, radius }),
    );
    progress(0.56 + 0.18 * ((ci + 1) / TREE_SPECIES.length), `veg: impostor ${sp.id}`);
  }

  // ---- understory: shrubs / fern / flowers (R1 only) -------------------------
  progress(0.76, 'veg: understory pools');
  const underSpecies = [
    { cls: VegClass.BushHazel, sp: UNDERSTORY_SPECIES[0] as SpeciesParams },
    { cls: VegClass.BushPink, sp: UNDERSTORY_SPECIES[1] as SpeciesParams },
    { cls: VegClass.Juniper, sp: UNDERSTORY_SPECIES[2] as SpeciesParams },
  ];
  for (const { cls, sp } of underSpecies) {
    for (let v = 0; v < 4; v++) {
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
  const fernAtlas = atlases.get('fern');
  for (let v = 0; v < 4; v++) {
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
                foliageCardMaterial(fernAtlas, { color: FERN_CAPTURE.foliageColor }),
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
    { cls: VegClass.FlowerUmbel, kind: 'umbel' },
    { cls: VegClass.FlowerBell, kind: 'bell' },
    { cls: VegClass.FlowerDaisy, kind: 'daisy' },
  ];
  for (const { cls, kind } of flowerKinds) {
    for (let v = 0; v < 4; v++) {
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
  progress(0.86, 'veg: deadfall + boulder pools');
  const deadTex = barkOf(5);
  const decayOf: DecayState[] = ['fresh', 'mossy', 'rotten', 'mossy'];
  for (let v = 0; v < 4; v++) {
    const log = buildLog(seed.rng(`veg/log/${v}`), decayOf[v] as DecayState);
    const b = bounds([log.geometry]);
    trackCls(VegClass.Log, b.height, b.radius);
    pools.push({
      cls: VegClass.Log,
      variant: v,
      r1: [
        {
          geo: log.geometry,
          tris: log.tris,
          make: () => deadwoodMaterial(deadTex),
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
    const stump = buildStump(seed.rng(`veg/stump/${v}`));
    const b = bounds([stump.geometry]);
    trackCls(VegClass.Stump, b.height, b.radius);
    pools.push({
      cls: VegClass.Stump,
      variant: v,
      r1: [
        {
          geo: stump.geometry,
          tris: stump.tris,
          make: () => deadwoodMaterial(deadTex),
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

  const rockPools: { cls: number; preset: 'boulder' | 'slab'; moss: number }[] = [
    { cls: VegClass.Boulder, preset: 'boulder', moss: 0.3 },
    { cls: VegClass.Slab, preset: 'slab', moss: 0.12 },
  ];
  for (const { cls, preset, moss } of rockPools) {
    for (let v = 0; v < 4; v++) {
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
            make: () => rockMaterial({ moss }),
            castShadow: true,
          },
        ],
        r2: [
          {
            geo: lo.geometry,
            tris: lo.stats.tris,
            make: () => rockMaterial({ moss }),
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
  progress(0.93, 'veg: stone/branch pools');
  const stoneClasses: {
    cls: number;
    preset: 'boulder' | 'cobble';
    d1: number;
    d2: number | null;
    moss: number;
    maxDist: number;
  }[] = [
    { cls: VegClass.StoneL, preset: 'boulder', d1: 3, d2: 2, moss: 0.22, maxDist: 900 },
    { cls: VegClass.StoneM, preset: 'cobble', d1: 2, d2: 1, moss: 0.12, maxDist: 280 },
    { cls: VegClass.StoneS, preset: 'cobble', d1: 1, d2: null, moss: 0.06, maxDist: 90 },
  ];
  for (const sc of stoneClasses) {
    for (let v = 0; v < 4; v++) {
      const hi = buildRock(sc.preset, seed.rng(`veg/stone${sc.cls}/${v}`), sc.d1);
      const lo =
        sc.d2 !== null
          ? buildRock(sc.preset, seed.rng(`veg/stone${sc.cls}/${v}`), sc.d2)
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
            make: () => rockMaterial({ moss: sc.moss }),
            castShadow: sc.cls !== VegClass.StoneS,
          },
        ],
        r2: lo
          ? [
              {
                geo: lo.geometry,
                tris: lo.stats.tris,
                make: () => rockMaterial({ moss: sc.moss }),
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
  // fallen branches: scaled twig tubes, deadwood-shaded
  for (let v = 0; v < 4; v++) {
    const geo = twigGeometry(seed.rng(`veg/branch/${v}`));
    geo.scale(6.5, 5, 6.5);
    const tris = geo.index ? geo.index.count / 3 : 0;
    const b = bounds([geo]);
    trackCls(VegClass.Branch, b.height, b.radius);
    pools.push({
      cls: VegClass.Branch,
      variant: v,
      r1: [
        {
          geo,
          tris,
          make: () => deadwoodMaterial(deadTex),
          castShadow: false,
        },
      ],
      // clone: a geometry holds ONE indirect slot — sharing it across draws
      // would overwrite the first draw's offset
      r2: [
        {
          geo: geo.clone(),
          tris,
          make: () => deadwoodMaterial(deadTex),
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

  progress(1, 'veg: pools ready');
  return { pools, impostors, clsHeight, clsRadius, clsMaxDist, atlases, barks };
}
