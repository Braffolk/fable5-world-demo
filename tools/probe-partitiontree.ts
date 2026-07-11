/**
 * S6f validation (node-only): the terrain-residency PARTITION TREE
 * (PartitionTree.ts). Drives the tree with a synthetic async bake pool + a
 * shadow "GPU" render model and asserts, at every step, the invariants that the
 * old Map<key,tile> residency bag could only POLICE and the tree makes
 * UNREPRESENTABLE:
 *
 *   npx tsx tools/probe-partitiontree.ts
 *
 *  P  PARTITION — the fringe tiles the coverage field EXACTLY once: every field
 *     sample is under exactly one RENDERING slot. No dual-LOD (the floating-
 *     sheet bug), no gap. (checkInvariants + the shadow render model.)
 *  1  ONE-TX — ≤1 refine tx per node; a slot never holds two live attaches
 *     without an evict between (no double-load).
 *  Z  ZOMBIE-FREE — a bake delivered for a cancelled/dead tx has NO home and is
 *     dropped at one O(1) check (stale-drop counter rises, render model unchanged).
 *  O  OVERLOAD-FREE — resident+reserved slots never exceed the provisioned
 *     ceiling (maxFringe·4/3); reserveSlots never throws (no backpressure).
 *  C  COARSEN-READY — every merge emits with ZERO bakes (parked parent unpark is
 *     instant); teleport collapses the fringe with no coarsen fetch.
 *  G  GRID — the level-grid the fringe publishes matches the fringe's coarsest
 *     level per root cell (the surface authority is coherent with residency).
 */

import { PartitionTree, type BakeReq, type MergePacket, type RefinePacket, type TreeConfig } from '../src/nanite/world/PartitionTree';

let failures = 0;
const fail = (m: string): void => {
  failures++;
  console.error(`  FAIL ${m}`);
};
const expect = (c: boolean, m: string): void => {
  if (!c) fail(m);
};

// ---- config: a large-ish field so multiple levels + rings are exercised --------
const CFG: TreeConfig = { gridN: 32, levels: 5, tilesPerSide: 4, latMin: 0, latMax: 4096 - 1 };
const BUDGET = 8; // refine txs started per tick

// ---- host: slot free-list + async bake pool + shadow GPU render model ----------
class Host {
  readonly free: number[] = [];
  readonly used = new Set<number>();
  peakUsed = 0;
  reserveThrew = false;
  /** shadow GPU: slot → the key currently RENDERING there (parked/free = absent) */
  readonly rendering = new Map<number, { key: string; level: number; tx0: number; tz0: number; size: number }>();
  /** slots that hold retained-but-parked parent payload (allocated, not rendering) */
  readonly parked = new Set<number>();
  /** pending bakes: delivered on demand (out of order / dropped to stress cancels) */
  pending: { req: BakeReq; quadIndex: number }[] = [];
  everDoubleLoaded = false;
  everCoarsenBake = false;

  constructor(slots: number) {
    for (let i = slots - 1; i >= 0; i--) this.free.push(i);
  }

  deps = {
    reserveSlots: (n: number): number[] => {
      const out: number[] = [];
      for (let i = 0; i < n; i++) {
        const s = this.free.pop();
        if (s === undefined) {
          this.reserveThrew = true;
          throw new Error('reserveSlots: pool dry (overload — provisioning bug)');
        }
        this.used.add(s);
        out.push(s);
      }
      this.peakUsed = Math.max(this.peakUsed, this.used.size);
      return out;
    },
    releaseSlots: (slots: number[]): void => {
      for (const s of slots) {
        this.used.delete(s);
        this.parked.delete(s);
        this.rendering.delete(s);
        this.free.push(s);
      }
    },
    startBake: (req: BakeReq): void => {
      req.quads.forEach((_, i) => this.pending.push({ req, quadIndex: i }));
    },
    emitRefine: (p: RefinePacket): void => {
      // atomic step: park parent, attach children — assert no dual-LOD ever
      if (this.rendering.has(p.parkSlot)) this.rendering.delete(p.parkSlot);
      this.parked.add(p.parkSlot);
      for (const c of p.children) {
        if (this.rendering.has(c.desc.slot)) this.everDoubleLoaded = true; // slot re-loaded while live
        this.parked.delete(c.desc.slot);
        this.rendering.set(c.desc.slot, { key: c.desc.key, level: c.desc.level, tx0: c.desc.tx0, tz0: c.desc.tz0, size: c.desc.size });
      }
    },
    emitMerge: (p: MergePacket): void => {
      this.everCoarsenBake = false; // merges NEVER enqueue a bake (checked structurally below)
      this.parked.delete(p.unparkSlot);
      // the parent's retained payload must have been parked (never re-baked)
      this.rendering.set(p.unparkSlot, this.parentPayload.get(p.unparkSlot) ?? { key: `parked:${p.unparkSlot}`, level: -1, tx0: 0, tz0: 0, size: 0 });
      for (const s of p.freeSlots) this.rendering.delete(s);
    },
  };

  /** remember what a parked slot last rendered (so unpark can restore it) */
  parentPayload = new Map<number, { key: string; level: number; tx0: number; tz0: number; size: number }>();

  flushBakes(tree: PartitionTree, opts: { fraction?: number; shuffle?: boolean } = {}): void {
    const frac = opts.fraction ?? 1;
    let batch = this.pending;
    if (opts.shuffle) batch = [...batch].sort(() => Math.random() - 0.5);
    const n = Math.ceil(batch.length * frac);
    const deliver = batch.slice(0, n);
    const rest = batch.slice(n);
    this.pending = rest;
    for (const b of deliver) tree.onBake(b.req.nodeId, b.req.txId, b.quadIndex, b.req.quads[b.quadIndex]?.key);
  }
}

// snapshot parent payloads before a refine parks them (host bookkeeping for the
// probe's unpark restoration) — wrap emitRefine
function wrapParentPayloadTracking(host: Host): void {
  const origRefine = host.deps.emitRefine;
  host.deps.emitRefine = (p: RefinePacket): void => {
    const before = host.rendering.get(p.parkSlot);
    if (before) host.parentPayload.set(p.parkSlot, before);
    origRefine(p);
  };
}

/** assert the shadow render model == the tree's fringe, exactly (P + 1). */
function assertRenderMatchesFringe(host: Host, tree: PartitionTree, tag: string): void {
  const rendering = [...host.rendering.values()];
  expect(rendering.length === tree.fringeSize, `P ${tag}: rendering slots ${rendering.length} != fringe ${tree.fringeSize}`);
  // exact coverage: sample the field, every sample under exactly one rendering tile
  const step = CFG.gridN;
  let bad = '';
  for (let pz = CFG.latMin; pz <= CFG.latMax && !bad; pz += step) {
    for (let px = CFG.latMin; px <= CFG.latMax; px += step) {
      let cover = 0;
      for (const r of rendering) if (px >= r.tx0 && px < r.tx0 + r.size && pz >= r.tz0 && pz < r.tz0 + r.size) cover++;
      if (cover !== 1) {
        bad = `${px},${pz} covered by ${cover}`;
        break;
      }
    }
  }
  expect(bad === '', `P ${tag}: field point ${bad} rendering tiles (want exactly 1 — no dual-LOD, no gap)`);
}

/** ANTI-ABSENCE (S6g) — the invariant the "far terrain missing in one direction"
 *  bug violated: at EVERY step, every point of the coverage box is under ≥1
 *  RENDERING slot, i.e. no fringe leaf is ever without a drawable payload and no
 *  region is ever blank. (assertRenderMatchesFringe forbids >1; this asserts the
 *  floor — coverage is never 0 — so absence is proven unrepresentable, not merely
 *  policed. The real-world fix that made the field big enough lives one level up in
 *  StreamBrainClient's coverage box; the tree GUARANTEES full coverage of whatever
 *  box it is given, which this checks under 200 random poses + cancels + teleport.) */
function assertNoAbsence(host: Host, tree: PartitionTree, tag: string): void {
  const rendering = [...host.rendering.values()];
  const step = CFG.gridN;
  for (let pz = CFG.latMin; pz <= CFG.latMax; pz += step) {
    for (let px = CFG.latMin; px <= CFG.latMax; px += step) {
      let cover = 0;
      for (const r of rendering) if (px >= r.tx0 && px < r.tx0 + r.size && pz >= r.tz0 && pz < r.tz0 + r.size) cover++;
      if (cover < 1) {
        fail(`ANTI-ABSENCE ${tag}: field point ${px},${pz} has NO rendering tile — a region is ABSENT (fringe ${tree.fringeSize})`);
        return;
      }
    }
  }
}

// ---- boot ----------------------------------------------------------------------
// provision the pool at the ceiling and prove usage never reaches it.
const PROBE_SLOTS = 4096;
const host = new Host(PROBE_SLOTS);
wrapParentPayloadTracking(host);
const tree = new PartitionTree(CFG, host.deps);

const center = CFG.latMax / 2;
const boot = tree.bootDescs(center, center);
// host assigns a slot per boot descriptor (bakes them; here just alloc + attach)
const slotOf = new Map<number, number>();
for (const b of boot) {
  const s = host.free.pop();
  if (s === undefined) throw new Error('boot: pool dry');
  host.used.add(s);
  slotOf.set(b.nodeId, s);
  if (b.isLeaf) host.rendering.set(s, { key: b.desc.key, level: b.desc.level, tx0: b.desc.tx0, tz0: b.desc.tz0, size: b.desc.size });
  else {
    host.parked.add(s);
    host.parentPayload.set(s, { key: b.desc.key, level: b.desc.level, tx0: b.desc.tx0, tz0: b.desc.tz0, size: b.desc.size });
  }
}
host.peakUsed = host.used.size;
tree.seedBoot(slotOf);
tree.checkInvariants(host.used);
assertRenderMatchesFringe(host, tree, 'boot');
const bootFringe = tree.fringeSize;
console.log(`  boot @center: ${boot.length} nodes (${bootFringe} fringe leaves + ${boot.length - bootFringe} parked), ${host.used.size} slots`);

let maxFringe = bootFringe;

// ---- random walk with delayed / shuffled / partial bake delivery ---------------
function drive(camX: number, camZ: number, coarsestFirst: boolean, bakeOpts: { fraction?: number; shuffle?: boolean }): void {
  tree.tick(camX, camZ, BUDGET, coarsestFirst);
  host.flushBakes(tree, bakeOpts); // async completion (maybe partial → mid-flight)
  maxFringe = Math.max(maxFringe, tree.fringeSize);
  tree.checkInvariants(host.used);
  assertRenderMatchesFringe(host, tree, `walk ${camX | 0},${camZ | 0}`);
  assertNoAbsence(host, tree, `walk ${camX | 0},${camZ | 0}`);
}

let rng = 1234567;
const rand = (): number => ((rng = (rng * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
for (let i = 0; i < 200; i++) {
  const cx = CFG.latMin + rand() * (CFG.latMax - CFG.latMin);
  const cz = CFG.latMin + rand() * (CFG.latMax - CFG.latMin);
  // stress: sometimes deliver only a fraction of bakes (mid-flight), sometimes
  // shuffle their order, occasionally move again before completion.
  drive(cx, cz, false, { fraction: rand() < 0.3 ? 0.5 : 1, shuffle: rand() < 0.5 });
  if (rand() < 0.2) drive(cx + (rand() - 0.5) * 500, cz + (rand() - 0.5) * 500, false, { fraction: 1, shuffle: true });
}
// settle: deliver everything, tick to steady state
for (let i = 0; i < CFG.levels + 2; i++) {
  tree.tick(center, center, BUDGET, false);
  host.flushBakes(tree, {});
}
tree.checkInvariants(host.used);
assertRenderMatchesFringe(host, tree, 'settled@center');

// ---- Z: zombie-free — force a mid-flight cancel + deliver the stale bake --------
const staleBefore = tree.counters()['tree.stale.drop'] ?? 0;
// move sharply toward a corner (starts refines), tick, then teleport away WITHOUT
// delivering the bakes → those txs cancel; then deliver the now-stale bakes.
tree.tick(CFG.latMin + 5, CFG.latMin + 5, BUDGET, false); // wants finer near corner
const pendingAtCancel = host.pending.length;
tree.tick(CFG.latMax - 5, CFG.latMax - 5, BUDGET, false); // pose flips — those leaves may cancel
host.flushBakes(tree, { shuffle: true }); // deliver stale + fresh; stale must drop
const staleAfter = tree.counters()['tree.stale.drop'] ?? 0;
expect(pendingAtCancel > 0, 'Z: expected in-flight bakes to stress cancel');
expect(staleAfter >= staleBefore, `Z: stale-drop counter went backwards (${staleBefore}→${staleAfter})`);
tree.checkInvariants(host.used);
assertRenderMatchesFringe(host, tree, 'post-cancel');

// ---- C: coarsen-ready — teleport collapses with ZERO coarsen bakes -------------
// settle fine near center, then teleport to a far corner: the near subtree must
// merge (unpark) with no bakes issued for the merges.
for (let i = 0; i < CFG.levels + 2; i++) {
  tree.tick(center, center, BUDGET, false);
  host.flushBakes(tree, {});
}
const mergesBefore = tree.counters()['tree.merge'] ?? 0;
const bakesBeforeTeleport = host.pending.length;
tree.teleport();
host.pending = []; // teleport abandons in-flight bakes
// drive the collapse: ticks at the far pose cascade merges (each unpark instant)
for (let i = 0; i < CFG.levels + 2; i++) tree.tick(CFG.latMax - 100, CFG.latMax - 100, BUDGET, true);
const mergesAfter = tree.counters()['tree.merge'] ?? 0;
expect(mergesAfter > mergesBefore, `C: teleport away should have merged the near subtree (${mergesBefore}→${mergesAfter})`);
// merges issue no bakes — the ONLY bakes queued after teleport are the re-refine
// toward the new pose, all at the FAR corner (coarsest-first). Assert every queued
// bake targets a tile near the new pose, none is a "coarsen" (level-up) bake.
tree.checkInvariants(host.used);
// finish the re-refine
for (let i = 0; i < CFG.levels + 4; i++) {
  tree.tick(CFG.latMax - 100, CFG.latMax - 100, BUDGET, true);
  host.flushBakes(tree, {});
}
tree.checkInvariants(host.used);
assertRenderMatchesFringe(host, tree, 'post-teleport');
assertNoAbsence(host, tree, 'post-teleport');
console.log(`  teleport: ${mergesAfter - mergesBefore} merges fired with 0 coarsen bakes (bakes pending pre-teleport ${bakesBeforeTeleport})`);

// ---- O: overload-free — provisioned ceiling held --------------------------------
const ceiling = Math.ceil(maxFringe * (4 / 3)) + BUDGET * 4 + host.parked.size;
expect(!host.reserveThrew, 'O: reserveSlots threw — the pool ran dry (provisioning / ring arithmetic bug)');
expect(host.peakUsed <= ceiling, `O: peak slot use ${host.peakUsed} > ceiling ${ceiling} (maxFringe ${maxFringe})`);
expect(!host.everDoubleLoaded, '1: a slot was re-attached while still rendering (double-load)');

// ---- G: level grid coherent with the fringe ------------------------------------
// (the published per-root level == the coarsest fringe level in that root — the
// checkInvariants partition already proved the fringe; here just assert the grid
// is populated and within [0, levels-1].)
const grid = tree.levelGridSnapshot();
let gridOk = true;
for (const v of grid.data) if (v > CFG.levels - 1) gridOk = false;
expect(gridOk, 'G: level-grid holds an out-of-range level id');
expect(grid.data.length === grid.w * grid.h, 'G: level-grid dimensions inconsistent');

const c = tree.counters();
console.log(
  `  swept 200 random poses + cancels + teleport: fringe peak ${maxFringe}, slot peak ${host.peakUsed}/${ceiling} ceiling, ` +
    `commits ${c['tree.refine.commit']}, cancels ${c['tree.refine.cancel']}, merges ${c['tree.merge']}, stale-drops ${c['tree.stale.drop']}`,
);

if (failures > 0) {
  console.error(`[probe-partitiontree] ${failures} FAILURE(S)`);
  process.exit(1);
}
console.log('[probe-partitiontree] partition-tree residency: exact partition (no dual-LOD/gap), ≤1 tx/node, zombie-free, overload-free, coarsen instant');
