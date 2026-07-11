/**
 * ScatterMap — the Estonia understory/debris guidance dictionaries → VegLibrary
 * ground pools (SPEC-STREAMING-WORLD §7, S9a; the sibling of SpeciesMap for trees).
 *
 * The cooked release ships two scatter-guidance layers, each a u8 plane pair
 * (categoryId, density) plus a dictionary of named communities/surfaces whose
 * `palette` lists "plant:weight" tokens (understoryMap, debrisMap). The library has
 * NO botanical species — it renders the SAME abstract ground pools on both sources:
 * shrubs (BushHazel/BushPink/Juniper), deadwood (Log/Stump/Branch) and stones
 * (Boulder/Slab/StoneL/M/S). This resolver folds each community's palette onto that
 * pool set by keyword, so Estonia ground cover MATCHES the generated world's — which
 * itself renders only the pools that have a head: ferns/flowers/moss/grass/litter are
 * groundcover with no mesh (deferred on BOTH sources), so they fall to the SKIP
 * bucket and are covered by the grass lane + biome tint, exactly as generated.
 *
 * Pure/node-testable (no GPU/DOM). A community with an all-skip palette (moss/grass
 * only) resolves to the empty distribution — one boot summary line, never a throw
 * (§F placeholders never block).
 */

import { VegClass } from '../../gpu/passes/Scatter';
import type { CommunityEntry, WorldDictionaries } from '../../world/source/WorldSource';

/** SKIP marker — a palette token with no library mesh (moss/grass/fern/flower/litter);
 *  the grass raymarch lane + biome tint cover these, matching the generated world. */
const SKIP = -1;

/** understory palette token → the shrub pool that renders it (the only understory
 *  pools with a head). Ericaceous dwarf shrubs → BushPink; tall scrub → BushHazel;
 *  juniper → Juniper. Everything else (herbs/ferns/flowers/grass/moss) → SKIP. */
const UNDER_KEYWORDS: readonly (readonly [RegExp, VegClass])[] = [
  [/juniper/, VegClass.Juniper],
  [/heather|cowberry|lingonberry|labrador|cranberry|cloudberry|crowberry|bilberry|blueberry|whortle/, VegClass.BushPink],
  [/raspberry|hazel|willow|bramble|buckthorn|dogwood|spiraea|scrub|shrub|bush/, VegClass.BushHazel],
];

/** debris palette token → the deadwood/rock pool that renders it. Litter (needle
 *  duff, leaf litter, reed, tussock) → SKIP (ground texture, no mesh). */
const DEBRIS_KEYWORDS: readonly (readonly [RegExp, VegClass])[] = [
  [/boulder/, VegClass.Boulder],
  [/slab/, VegClass.Slab],
  [/stone_large|large_stone|rock_large/, VegClass.StoneL],
  [/stone_med|med_stone|cobble/, VegClass.StoneM],
  [/stone_small|small_stone|gravel|scree|pebble/, VegClass.StoneS],
  [/stump/, VegClass.Stump],
  [/fallen_log|mossy_log|driftwood|(^|_)log(_|$)/, VegClass.Log],
  [/branch|twig|dead_?wood|fallen_cone|(^|_)cone|acorn|stick/, VegClass.Branch],
];

const CLASS_NAME: Record<number, string> = {
  [VegClass.BushHazel]: 'BushHazel',
  [VegClass.BushPink]: 'BushPink',
  [VegClass.Juniper]: 'Juniper',
  [VegClass.Log]: 'Log',
  [VegClass.Stump]: 'Stump',
  [VegClass.Branch]: 'Branch',
  [VegClass.Boulder]: 'Boulder',
  [VegClass.Slab]: 'Slab',
  [VegClass.StoneL]: 'StoneL',
  [VegClass.StoneM]: 'StoneM',
  [VegClass.StoneS]: 'StoneS',
};

/** one guidance category → its renderable-pool distribution + lushness. classes may
 *  include SKIP (-1); `total` = Σweights INCLUDING skip, so the pick thins by the
 *  non-mesh fraction of the palette. `base` = the dict's base_density (plants/m²). */
export interface CategoryDist {
  classes: Int32Array; // one per palette token bucket (may be SKIP = -1)
  cumw: Float32Array; // running Σweight, last entry == total
  total: number;
  base: number;
}

/** the resolved guidance → pool maps for one manifest. */
export interface ScatterMap {
  understory(id: number): CategoryDist;
  debris(id: number): CategoryDist;
  /** max debris base_density — normalises debris acceptance so litter floors ≫ stony
   *  ground (understory does NOT scale by base; its density plane carries the lushness). */
  maxDebrisBase: number;
  summary: string;
}

const EMPTY: CategoryDist = { classes: new Int32Array(0), cumw: new Float32Array(0), total: 0, base: 0 };

/** parse a palette token ("bilberry:5" / "twig") → [name, weight] (weight default 1). */
function parseToken(tok: string): [string, number] {
  const c = tok.lastIndexOf(':');
  if (c < 0) return [tok.trim().toLowerCase(), 1];
  const w = Number(tok.slice(c + 1));
  return [tok.slice(0, c).trim().toLowerCase(), Number.isFinite(w) && w > 0 ? w : 1];
}

function classify(name: string, keywords: readonly (readonly [RegExp, VegClass])[]): number {
  for (const [re, cls] of keywords) if (re.test(name)) return cls;
  return SKIP;
}

/** fold one community's palette into a renderable-pool distribution. Empty/all-skip
 *  palettes yield a distribution whose renderable weight is 0 (→ nothing scattered). */
function distFor(entry: CommunityEntry, keywords: readonly (readonly [RegExp, VegClass])[]): CategoryDist {
  const buckets = new Map<number, number>();
  for (const tok of entry.palette) {
    const [name, w] = parseToken(tok);
    const cls = classify(name, keywords);
    buckets.set(cls, (buckets.get(cls) ?? 0) + w);
  }
  if (buckets.size === 0) return { ...EMPTY, base: entry.base_density };
  const classes = new Int32Array(buckets.size);
  const cumw = new Float32Array(buckets.size);
  let i = 0;
  let acc = 0;
  for (const [cls, w] of buckets) {
    classes[i] = cls;
    acc += w;
    cumw[i] = acc;
    i++;
  }
  return { classes, cumw, total: acc, base: entry.base_density };
}

/**
 * Build the resolver from the manifest dictionaries. The generated world never uses
 * this (its understory/extras/stones are explicit records — ChunkContent binds them
 * directly); Estonia's guidance planes flow through it in the UnderstoryScatter.
 */
export function buildScatterMap(dict: WorldDictionaries): ScatterMap {
  const under = new Map<number, CategoryDist>();
  const debris = new Map<number, CategoryDist>();
  let maxDebrisBase = 1;
  const emptyU: number[] = [];
  const emptyD: number[] = [];
  const poolHits = new Set<number>();
  for (const [id, e] of dict.understory) {
    const d = distFor(e, UNDER_KEYWORDS);
    under.set(id, d);
    if (renderableWeight(d) === 0) emptyU.push(id);
    for (const c of d.classes) if (c !== SKIP) poolHits.add(c);
  }
  for (const [id, e] of dict.debris) {
    const d = distFor(e, DEBRIS_KEYWORDS);
    debris.set(id, d);
    maxDebrisBase = Math.max(maxDebrisBase, d.base);
    if (renderableWeight(d) === 0) emptyD.push(id);
    for (const c of d.classes) if (c !== SKIP) poolHits.add(c);
  }
  const pools = [...poolHits].sort((a, b) => a - b).map((c) => CLASS_NAME[c] ?? `c${c}`).join(', ');
  const summary =
    `[laas] ScatterMap: ${dict.understory.size} understory + ${dict.debris.size} debris communities → ` +
    `pools {${pools}}` +
    (emptyU.length + emptyD.length > 0
      ? `; ${emptyU.length + emptyD.length} groundcover-only (no mesh, grass-lane covered) [u:${emptyU.join(',') || '-'} d:${emptyD.join(',') || '-'}]`
      : '; 0 groundcover-only');
  return {
    understory: (id) => under.get(id) ?? EMPTY,
    debris: (id) => debris.get(id) ?? EMPTY,
    maxDebrisBase,
    summary,
  };
}

/** Σ of renderable (non-skip) weights in a distribution. */
function renderableWeight(d: CategoryDist): number {
  let sum = 0;
  let prev = 0;
  for (let i = 0; i < d.classes.length; i++) {
    const w = (d.cumw[i] as number) - prev;
    prev = d.cumw[i] as number;
    if (d.classes[i] !== SKIP) sum += w;
  }
  return sum;
}

/** pick a VegClass from a distribution by a hash in [0,1); SKIP (-1) = no instance. */
export function pickClass(d: CategoryDist, r01: number): number {
  if (d.total <= 0) return SKIP;
  const r = r01 * d.total;
  for (let i = 0; i < d.classes.length; i++) {
    if (r < (d.cumw[i] as number)) return d.classes[i] as number;
  }
  return d.classes[d.classes.length - 1] as number;
}
