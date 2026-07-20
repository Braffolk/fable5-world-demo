/**
 * ScatterMap — the Estonia understory/debris guidance dictionaries → VegLibrary
 * ground pools (SPEC-STREAMING-WORLD §7, S9a; the sibling of SpeciesMap for trees).
 *
 * The cooked release ships two scatter-guidance layers, each a u8 plane pair
 * (categoryId, density) plus a dictionary of named communities/surfaces whose
 * `palette` lists "plant:weight" tokens (understoryMap, debrisMap). The library has
 * NO botanical species — it renders the SAME abstract ground pools on both sources:
 * shrubs (BushHazel/BushPink/Juniper), ferns (Fern), herb-layer flowers (Umbel/Bell/
 * Daisy), deadwood (Log/Stump/Branch) and stones (Boulder/Slab/StoneL/M/S). This
 * resolver folds each community's palette onto that pool set by keyword, so Estonia
 * ground cover MATCHES the generated world's. Only the true groundcover with no mesh
 * (lichen/grass/sedge/reed/litter) falls to the SKIP bucket — covered by the grass
 * lane + biome tint, exactly as generated.
 *
 * CARPET layer: palette tokens matching a CarpetSpec (sphagnum, …) STAY skip for the
 * understory distribution (they thin plant acceptance exactly as before — the plant
 * mix never rebalances) and ADDITIONALLY light up `carpetCover`, the per-community
 * cover-weight fractions the carpet band's deterministic patch grid reads
 * (UnderstoryScatter.carpetPlan). Moss carpets and the plants that grow IN them are
 * independent layers, never a per-cell either/or.
 *
 * Pure/node-testable (no GPU/DOM). A community with an all-skip palette (moss/grass
 * only) resolves to the empty distribution — one boot summary line, never a throw
 * (§F placeholders never block).
 */

import { VEG_CLASS_NAME, VegClass } from '../../gpu/passes/Scatter';
import { CARPET_SPECS, type CarpetSpec } from '../../vegetation/carpet/CarpetTypes';
import type { CommunityEntry, WorldDictionaries } from '../../world/source/WorldSource';

/** SKIP marker — a palette token with no library mesh (moss/lichen/grass/sedge/reed/
 *  litter); the grass raymarch lane + biome tint cover these, matching the generated
 *  world. Ferns + herb-layer flowers NOW have real meshes (#113), so they map to pools. */
const SKIP = -1;

/** understory palette token → the library pool that renders it. Ferns → Fern; the
 *  herb-layer flowers map to their nearest FORM archetype (umbel/bell/daisy compound);
 *  ericaceous dwarf shrubs → BushPink; tall scrub → BushHazel; juniper → Juniper. Moss/
 *  lichen/grass/sedge/reed/nettle/horsetail (no mesh) → SKIP. Grounded in the Estonia
 *  community palettes (asset-gen/config/understory-communities.toml). */
const UNDER_KEYWORDS: readonly (readonly [RegExp, VegClass])[] = [
  [/fern|bracken/, VegClass.Fern],
  [/goutweed|meadowsweet|yarrow|angelica|cow_parsley|hogweed|umbel/, VegClass.FlowerUmbel],
  [/hepatica|anemone|may_lily|lily_of|harebell|bellflower|campanula|wintergreen/, VegClass.FlowerBell],
  [/daisy|buttercup|knapweed|sorrel|oxalis|marigold|clover|dandelion|globeflower|hawkweed/, VegClass.FlowerDaisy],
  [/juniper/, VegClass.Juniper],
  // Estonia raised-bog palette (understory-communities.toml community 5 + heath/fen):
  // these have their own QA-approved bog meshes, so they route to dedicated pools —
  // ABOVE the generic BushPink line so they win the first-match. bog_rosemary has no
  // cooked token yet (dormant until the cook adds it), wired here so it's ready.
  [/cottongrass|cotton_grass|eriophorum/, VegClass.CottonGrass],
  [/heather|calluna/, VegClass.Heather],
  [/labrador_tea|labrador|ledum/, VegClass.LabradorTea],
  [/bog_rosemary|andromeda/, VegClass.BogRosemary],
  [/cranberry/, VegClass.Cranberry], // matches bog_cranberry
  [/cloudberry/, VegClass.Cloudberry],
  // remaining ericaceous berries with no dedicated mesh keep the generic dwarf-shrub pool
  [/cowberry|lingonberry|crowberry|bilberry|blueberry|whortle/, VegClass.BushPink],
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
  /** CARPET layer: per-CARPET_SPECS cover-weight fraction (0..1) of this understory
   *  community's palette — 0 everywhere for carpet-free communities. Presence gates
   *  the carpet band's deterministic patch grid; it never touches `understory`. */
  carpetCover(id: number): Float32Array;
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

const ZERO_COVER = new Float32Array(CARPET_SPECS.length);

/** per-CarpetSpec cover-weight fraction of a community palette (Σ matching token
 *  weights / Σ all weights); null when no token matches any carpet. */
function carpetCoverOf(entry: CommunityEntry): Float32Array | null {
  let total = 0;
  let any = false;
  const w = new Float32Array(CARPET_SPECS.length);
  for (const tok of entry.palette) {
    const [name, wt] = parseToken(tok);
    total += wt;
    for (let i = 0; i < CARPET_SPECS.length; i++) {
      if ((CARPET_SPECS[i] as CarpetSpec).keywords.test(name)) {
        w[i] = (w[i] as number) + wt;
        any = true;
      }
    }
  }
  if (!any || total <= 0) return null;
  for (let i = 0; i < w.length; i++) w[i] = (w[i] as number) / total;
  return w;
}

/**
 * Build the resolver from the manifest dictionaries. The generated world never uses
 * this (its understory/extras/stones are explicit records — ChunkContent binds them
 * directly); Estonia's guidance planes flow through it in the UnderstoryScatter.
 */
export function buildScatterMap(dict: WorldDictionaries): ScatterMap {
  const under = new Map<number, CategoryDist>();
  const debris = new Map<number, CategoryDist>();
  const carpets = new Map<number, Float32Array>();
  let maxDebrisBase = 1;
  const emptyU: number[] = [];
  const emptyD: number[] = [];
  const poolHits = new Set<number>();
  for (const [id, e] of dict.understory) {
    const d = distFor(e, UNDER_KEYWORDS);
    under.set(id, d);
    if (renderableWeight(d) === 0) emptyU.push(id);
    for (const c of d.classes) if (c !== SKIP) poolHits.add(c);
    const cov = carpetCoverOf(e);
    if (cov) carpets.set(id, cov);
  }
  for (const [id, e] of dict.debris) {
    const d = distFor(e, DEBRIS_KEYWORDS);
    debris.set(id, d);
    maxDebrisBase = Math.max(maxDebrisBase, d.base);
    if (renderableWeight(d) === 0) emptyD.push(id);
    for (const c of d.classes) if (c !== SKIP) poolHits.add(c);
  }
  const pools = [...poolHits].sort((a, b) => a - b).map((c) => VEG_CLASS_NAME[c] ?? `c${c}`).join(', ');
  const summary =
    `[laas] ScatterMap: ${dict.understory.size} understory + ${dict.debris.size} debris communities → ` +
    `pools {${pools}}` +
    (emptyU.length + emptyD.length > 0
      ? `; ${emptyU.length + emptyD.length} groundcover-only (no mesh, grass-lane covered) [u:${emptyU.join(',') || '-'} d:${emptyD.join(',') || '-'}]`
      : '; 0 groundcover-only') +
    `; ${carpets.size} carpet-bearing (${CARPET_SPECS.map((s) => s.id).join('/')})`;
  return {
    understory: (id) => under.get(id) ?? EMPTY,
    debris: (id) => debris.get(id) ?? EMPTY,
    carpetCover: (id) => carpets.get(id) ?? ZERO_COVER,
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
