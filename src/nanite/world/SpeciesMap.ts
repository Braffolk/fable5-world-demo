/**
 * SpeciesMap — the Estonia manifest species dictionary → VegLibrary tree pools
 * (SPEC-STREAMING-WORLD §7, S7). The cooked release enumerates ~21 species ids
 * (code + leaf class + ref height); the library has 16 distinct tree pools (Spruce,
 * Pine, Beech, Birch, KarstGnarl, Snag, Larch, Oak, Aspen, GreyAlder, BlackAlder, Ash,
 * Maple, Lime, Willow, Rowan).
 * This resolver folds the many ids onto the pools by Estonian forestry code first,
 * then by leaf class, and routes anything it can't place to the LOUD checker
 * placeholder (F15) — logged ONCE at boot.
 *
 * Pure/node-testable (no GPU/DOM). idF = cls·8 + (variant & 3), the VegLibrary
 * pool identity (Scatter.ts TREE_VARIANTS = 4); the generated world never uses
 * this (its species column IS the VegClass — ChunkContent's identity default).
 */

import { VegClass, TREE_VARIANTS } from '../../gpu/passes/Scatter';
import type { SpeciesEntry, WorldDictionaries } from '../../world/source/WorldSource';

/** const-enum has no runtime reverse map — name the tree pools for the boot line. */
const CLASS_NAME: Record<number, string> = {
  [VegClass.Spruce]: 'Spruce',
  [VegClass.Pine]: 'Pine',
  [VegClass.Beech]: 'Beech',
  [VegClass.Birch]: 'Birch',
  [VegClass.KarstGnarl]: 'KarstGnarl',
  [VegClass.Snag]: 'Snag',
  [VegClass.Larch]: 'Larch',
  [VegClass.Oak]: 'Oak',
  [VegClass.Aspen]: 'Aspen',
  [VegClass.GreyAlder]: 'GreyAlder',
  [VegClass.BlackAlder]: 'BlackAlder',
  [VegClass.Ash]: 'Ash',
  [VegClass.Maple]: 'Maple',
  [VegClass.Lime]: 'Lime',
  [VegClass.Willow]: 'Willow',
  [VegClass.Rowan]: 'Rowan',
};

/** Estonian forestry codes → the specific library pool the pilot expects. Anything
 *  not named here falls through to the leaf-class buckets below. */
const CODE_TO_CLASS: Record<string, VegClass> = {
  MA: VegClass.Pine, // mänd — Scots pine (the pilot's dominant)
  KU: VegClass.Spruce, // kuusk — Norway spruce
  KS: VegClass.Birch, // kask — birch
  SK: VegClass.Birch, // sookask — downy/bog birch (was falling through to Beech via
  // the broadleaf leaf-class default). Downy birch IS a birch → the Birch pool. A
  // dedicated stunted BogBirch pool is deferred with the bog-tree task (block full).
  HB: VegClass.Aspen, // haab — European aspen (own upright-oval form; was folding to Birch/weeping)
  PP: VegClass.Aspen, // hybrid poplar — same genus (Populus), reads as aspen (was Beech-fallback)
  LV: VegClass.GreyAlder, // hall lepp — grey alder (own short open ovoid; was Beech-fallback, 9.8%)
  LM: VegClass.BlackAlder, // sanglepp — black alder (own conic spire; was Beech-fallback, 4.2%)
  LH: VegClass.Larch, // lehis — larch (#112: own deciduous-conifer form, was folding to Spruce)
  TA: VegClass.Oak, // tamm — pedunculate oak (#112: own broad crown, was folding to Beech)
  SA: VegClass.Ash, // saar — European ash (batch-2: own tall airy crown; was Beech-fallback)
  VA: VegClass.Maple, // vaher — Norway maple (batch-2: own dense symmetric dome; was Beech-fallback)
  PN: VegClass.Lime, // pärn — small-leaved lime (batch-2: own broad dense dome; was Beech-fallback)
  RE: VegClass.Willow, // remmelgas — willow (batch-2: own broad drooping riparian crown; was Beech-fallback)
  PI: VegClass.Rowan, // pihlakas — rowan (batch-2: own small slender form; was Beech-fallback)
};

/** one species entry → a tree VegClass, or null when nothing in the library fits
 *  (→ placeholder). Code wins; leaf class is the fallback so an unlisted broadleaf
 *  still gets a plausible crown rather than the checker. */
function classForSpecies(e: SpeciesEntry): VegClass | null {
  const code = (e.code ?? '').toUpperCase();
  if (code in CODE_TO_CLASS) return CODE_TO_CLASS[code] as VegClass;
  if (e.leaf === 'snag') return VegClass.Snag;
  if (e.leaf === 'conifer') return VegClass.Spruce;
  if (e.leaf === 'broadleaf') return VegClass.Beech;
  return null;
}

export interface SpeciesMap {
  /** (species id, variant) → library idF; unmapped ids → placeholderIdF. */
  idFOf(species: number, variant: number): number;
  /** one boot summary line (F15: "placeholders active: …"). */
  summary: string;
  /** species ids routed to the placeholder (empty = clean map). */
  unmapped: number[];
}

/**
 * Build the resolver from the manifest dictionary. `placeholderIdF` is the
 * checker-tree pool's idF (registered through the normal TreeBuilder→DAG path).
 * The per-id table is precomputed so idFOf is a plain array lookup on the hot path.
 */
export function buildSpeciesMap(dict: WorldDictionaries, placeholderIdF: number): SpeciesMap {
  const table = new Map<number, VegClass>();
  const unmapped: number[] = [];
  const perPool = new Map<VegClass, number>();
  for (const [id, entry] of dict.species) {
    const cls = classForSpecies(entry);
    if (cls === null) {
      unmapped.push(id);
      continue;
    }
    table.set(id, cls);
    perPool.set(cls, (perPool.get(cls) ?? 0) + 1);
  }
  const idFOf = (species: number, variant: number): number => {
    const cls = table.get(species);
    if (cls === undefined) return placeholderIdF;
    return cls * 8 + (variant & (TREE_VARIANTS - 1));
  };
  const poolBits = [...perPool.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([cls, n]) => `${CLASS_NAME[cls] ?? `c${cls}`}×${n}`)
    .join(', ');
  const summary =
    `[laas] SpeciesMap: ${dict.species.size} manifest species → ${perPool.size} pools (${poolBits})` +
    (unmapped.length > 0 ? `; ${unmapped.length} placeholder-routed [${unmapped.join(',')}]` : '; 0 placeholder');
  return { idFOf, summary, unmapped };
}
