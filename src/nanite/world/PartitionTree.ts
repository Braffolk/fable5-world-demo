/**
 * PartitionTree (S6f) — the streamed-terrain RESIDENCY as a partition tree, the
 * replacement for the old Map<key,tile> residency bag. Its whole reason to exist
 * is that the illegal states the bag POLICED are here UNREPRESENTABLE BY TYPE:
 *
 *  - A region of the field is EITHER a `Leaf` (one renderable tile in one slot)
 *    XOR a `Split` (four child nodes + a PARKED parent payload). A region can
 *    therefore never hold two live LODs at once — dual-LOD rendering (the
 *    "floating coarse sheet over fine tiles" bug) is not expressible.
 *  - A refine TRANSACTION is a FIELD of the node (`Leaf.tx`), so at most one is
 *    in flight per node — double-loading is not expressible. A bake exists IFF a
 *    live tx owns it; an arriving payload for a dead tx has no home and is
 *    dropped at one O(1) check — zombie payloads are not expressible.
 *  - Slots are RESERVED from a provisioned free-list at tx creation. Running dry
 *    is a throw-loud assert (the ring arithmetic is wrong), never a backpressure
 *    loop — overload is not expressible.
 *  - The RESIDENT SET is an ancestor-closed subtree: every `Split` keeps its
 *    parent payload PARKED (clusterCount 0, slot retained), so coarsening is an
 *    instant O(1) unpark — there is no coarsen bake, no retreat queue, no
 *    "always ready" to hope for. Payload eviction happens at exactly one moment:
 *    a node leaves the resident subtree because its parent merged.
 *
 * The render set is the FRINGE (the leaves), MAINTAINED by O(1) deltas per
 * committed rewrite (remove 1 / add ≤4, or the reverse), never re-walked. The
 * pose tick iterates the fringe ONLY: each leaf decides split / hold / vote-merge
 * from ring radii; merge agreement needs no sibling scan (each Split holds a
 * `mergeVotes` counter its leaf children bump as their vote flips; at
 * votes==childCount the merge fires). A full-tree walk exists ONLY as a dev
 * invariant assert (`checkInvariants`).
 *
 * PURE + host-agnostic (no three/DOM/worker deps): the brain wires it to the
 * bake pool + packet mailbox; the node probe drives it with synchronous fakes
 * and asserts the invariants above under random walks, delayed bakes, mid-flight
 * cancels and teleports.
 */

/** the geometry a tree config needs (a subset of ClipmapConfig — same lattice
 *  conventions: level k stride = 1<<k texels, tile side = gridN<<k texels, keys
 *  `L{level}:{gx},{gz}` on that level's tile grid). */
export interface TreeConfig {
  /** tile resolution — cells per side (every level, every tile) */
  gridN: number;
  /** number of concentric levels; roots sit at level (levels-1) */
  levels: number;
  /** even ≥2 — the ring width that sets the split/merge radii (M) */
  tilesPerSide: number;
  /** finest-lattice sample bounds (inclusive) — the coverage box in texels */
  latMin: number;
  latMax: number;
}

/** a tile's texel footprint + its slot identity, shared by bake requests and
 *  committed packets (geometry is opaque to the tree — the host bakes it). */
export interface QuadDesc {
  slot: number;
  level: number;
  tx0: number;
  tz0: number;
  size: number;
  key: string;
}

export interface BakeReq {
  nodeId: number;
  txId: number;
  quads: QuadDesc[];
}

/** one coarse-cell fringe-level edit (root-cell granularity) — the surface
 *  authority every height consumer clamps to. */
export interface LevelGridEdit {
  cellX: number;
  cellZ: number;
  level: number;
}

/** REFINE = park parent slot, attach ≤4 baked children, update the level grid —
 *  applied by the host as ONE atomic drain step. */
export interface RefinePacket {
  parkSlot: number;
  children: { desc: QuadDesc; payload: unknown }[];
  levelGrid: LevelGridEdit[];
}

/** MERGE = unpark parent slot, evict the ≤4 children, update the level grid. */
export interface MergePacket {
  unparkSlot: number;
  freeSlots: number[];
  levelGrid: LevelGridEdit[];
}

export interface TreeDeps {
  /** pop `n` slots off the provisioned free-list; THROW-LOUD if dry (§5 — a dry
   *  pool means the ring arithmetic or the tree is buggy, never backpressure). */
  reserveSlots(n: number): number[];
  releaseSlots(slots: number[]): void;
  /** issue the child bakes for a refine tx; the host calls tree.onBake(...) per
   *  quad as each returns (or never, if it aborts the bake on cancel). */
  startBake(req: BakeReq): void;
  emitRefine(p: RefinePacket): void;
  emitMerge(p: MergePacket): void;
  /** does finer source DATA EXIST for this leaf's children (manifest coverage,
   *  residency-independent)? Gates wantFiner so the tree never nominates a split
   *  whose bake can only ABORT (no child data there) — the perpetual re-nominate/
   *  abort/refetch loop at a fine↔coarse data boundary. Full-coverage / generated
   *  worlds always return true (no residency change). Optional: absent ⇒ always
   *  refinable (legacy behaviour / the node probe's synchronous fakes). */
  canRefine?(level: number, tx0: number, tz0: number, size: number): boolean;
}

/** the in-flight refine transaction — a FIELD of a Leaf, so ≤1 per node. */
interface Refine {
  txId: number;
  /** child quads still awaiting a bake (4→0) */
  pending: number;
  /** the quads this tx reserved (one slot each) */
  quads: QuadDesc[];
  /** baked payloads keyed by quad index (undefined until onBake) */
  payloads: (unknown | undefined)[];
}

interface NodeCommon {
  id: number;
  level: number;
  tx0: number;
  tz0: number;
  /** tile side in texels = gridN << level */
  size: number;
  key: string;
  parent: Split | null;
  /** quadrant index within the parent (0..3), or -1 for a root */
  quadrant: number;
}

interface Leaf extends NodeCommon {
  kind: 'leaf';
  /** the slot this leaf renders from */
  slot: number;
  /** at most one in-flight refine (double-request unrepresentable) */
  tx: Refine | null;
  /** whether this leaf currently contributes +1 to parent.mergeVotes */
  votedMerge: boolean;
}

interface Split extends NodeCommon {
  kind: 'split';
  children: Node[];
  /** the retained parent payload's slot (parked: clusterCount 0) */
  parkedSlot: number;
  /** how many leaf children currently vote to merge this Split */
  mergeVotes: number;
}

type Node = Leaf | Split;

/** hysteresis on the merge radius (a leaf splits inside R, its parent merges
 *  only past R·H) so a leaf hovering on a ring boundary does not thrash. */
const MERGE_HYSTERESIS = 1.3;

export class PartitionTree {
  private readonly cfg: TreeConfig;
  private readonly deps: TreeDeps;
  private readonly roots: Node[] = [];
  /** the maintained render set — every leaf, and ONLY leaves (§2) */
  private readonly fringe = new Set<Leaf>();
  private nextNodeId = 1;
  private nextTxId = 1;

  /** level-grid: one u8 per root cell = the shallowest (finest) fringe level in
   *  that root's subtree (the surface authority; §7). Maintained by a per-root
   *  histogram of fringe-leaf levels so the min is an O(levels) read. */
  private readonly gridW: number;
  private readonly gridH: number;
  private readonly rootTexels: number;
  private readonly gridMinX: number;
  private readonly gridMinZ: number;
  /** per-cell level-count histograms [cell*levels + level] */
  private readonly gridHist: Int32Array;
  /** per-cell published level (what the last emitted edit set) */
  private readonly gridLevel: Uint8Array;

  // counters (surfaced to the brain HUD)
  private nRefineCommit = 0;
  private nRefineCancel = 0;
  private nMerge = 0;
  private nStaleDrop = 0;
  private nTeleports = 0;

  constructor(cfg: TreeConfig, deps: TreeDeps) {
    if (cfg.tilesPerSide % 2 !== 0 || cfg.tilesPerSide < 2) {
      throw new Error(`PartitionTree: tilesPerSide must be even ≥2, got ${cfg.tilesPerSide}`);
    }
    if (cfg.levels < 1) throw new Error(`PartitionTree: levels must be ≥1, got ${cfg.levels}`);
    this.cfg = cfg;
    this.deps = deps;
    this.rootTexels = cfg.gridN << (cfg.levels - 1);
    // root grid spans the coverage box, root-tile aligned
    const gx0 = Math.floor(cfg.latMin / this.rootTexels);
    const gz0 = Math.floor(cfg.latMin / this.rootTexels);
    const gx1 = Math.floor((cfg.latMax) / this.rootTexels);
    const gz1 = Math.floor((cfg.latMax) / this.rootTexels);
    this.gridMinX = gx0;
    this.gridMinZ = gz0;
    this.gridW = gx1 - gx0 + 1;
    this.gridH = gz1 - gz0 + 1;
    this.gridHist = new Int32Array(this.gridW * this.gridH * cfg.levels);
    this.gridLevel = new Uint8Array(this.gridW * this.gridH).fill(cfg.levels - 1);
  }

  // ---- geometry ------------------------------------------------------------------

  /** split/merge radius at level k (texels): (M/2)·(gridN<<k) — half a level-k
   *  ring block, the doubling-clipmap radius. */
  private ringR(k: number): number {
    return (this.cfg.tilesPerSide / 2) * (this.cfg.gridN << k);
  }

  /** does finer child DATA exist for this leaf (deps predicate; absent ⇒ true)? */
  private canRefine(n: NodeCommon): boolean {
    return this.deps.canRefine ? this.deps.canRefine(n.level, n.tx0, n.tz0, n.size) : true;
  }

  /** Chebyshev distance (texels) from (px,pz) to a tile footprint's nearest point. */
  private nearDist(n: NodeCommon, px: number, pz: number): number {
    const dx = px < n.tx0 ? n.tx0 - px : px > n.tx0 + n.size ? px - (n.tx0 + n.size) : 0;
    const dz = pz < n.tz0 ? n.tz0 - pz : pz > n.tz0 + n.size ? pz - (n.tz0 + n.size) : 0;
    return Math.max(dx, dz);
  }

  /** is this footprint at least partly on the coverage field? (rim tiles are kept
   *  and clamp-extended, like clipmapTiles.) */
  private onField(tx0: number, tz0: number, size: number): boolean {
    return tx0 < this.cfg.latMax + 1 && tz0 < this.cfg.latMax + 1 && tx0 + size > this.cfg.latMin && tz0 + size > this.cfg.latMin;
  }

  private quadKey(level: number, tx0: number, tz0: number): string {
    const T = this.cfg.gridN << level;
    return `L${level}:${Math.round(tx0 / T)},${Math.round(tz0 / T)}`;
  }

  /** the ≤4 on-field child footprints of a node (next-finer level). */
  private childDescs(n: NodeCommon): { level: number; tx0: number; tz0: number; size: number; key: string }[] {
    const cl = n.level - 1;
    const cs = this.cfg.gridN << cl;
    const out: { level: number; tx0: number; tz0: number; size: number; key: string }[] = [];
    for (let j = 0; j < 2; j++) {
      for (let i = 0; i < 2; i++) {
        const tx0 = n.tx0 + i * cs;
        const tz0 = n.tz0 + j * cs;
        if (!this.onField(tx0, tz0, cs)) continue;
        out.push({ level: cl, tx0, tz0, size: cs, key: this.quadKey(cl, tx0, tz0) });
      }
    }
    return out;
  }

  // ---- level grid ----------------------------------------------------------------

  private gridCellsOf(n: NodeCommon, fn: (cell: number) => void): void {
    // the root cells a footprint overlaps (footprints are root-aligned or nested,
    // so this is exact)
    const cx0 = Math.max(this.gridMinX, Math.floor(n.tx0 / this.rootTexels));
    const cz0 = Math.max(this.gridMinZ, Math.floor(n.tz0 / this.rootTexels));
    const cx1 = Math.min(this.gridMinX + this.gridW - 1, Math.floor((n.tx0 + n.size - 1) / this.rootTexels));
    const cz1 = Math.min(this.gridMinZ + this.gridH - 1, Math.floor((n.tz0 + n.size - 1) / this.rootTexels));
    for (let cz = cz0; cz <= cz1; cz++) {
      for (let cx = cx0; cx <= cx1; cx++) {
        fn((cz - this.gridMinZ) * this.gridW + (cx - this.gridMinX));
      }
    }
  }

  /** fringe leaf entered (+1) / left (−1) the render set — update the per-root
   *  histograms; return the cells whose published min level changed. */
  private gridDelta(leaf: Leaf, sign: 1 | -1, edits: LevelGridEdit[]): void {
    const L = this.cfg.levels;
    this.gridCellsOf(leaf, (cell) => {
      this.gridHist[cell * L + leaf.level] += sign;
      // recompute the published min (finest) level in this cell
      let min = L - 1;
      for (let k = 0; k < L; k++) {
        if ((this.gridHist[cell * L + k] as number) > 0) {
          min = k;
          break;
        }
      }
      if (this.gridLevel[cell] !== min) {
        this.gridLevel[cell] = min;
        const cx = (cell % this.gridW) + this.gridMinX;
        const cz = Math.floor(cell / this.gridW) + this.gridMinZ;
        edits.push({ cellX: cx, cellZ: cz, level: min });
      }
    });
  }

  // ---- fringe membership (O(1) deltas) -------------------------------------------

  private addLeaf(leaf: Leaf, edits: LevelGridEdit[]): void {
    this.fringe.add(leaf);
    this.gridDelta(leaf, 1, edits);
  }

  private removeLeaf(leaf: Leaf, edits: LevelGridEdit[]): void {
    this.fringe.delete(leaf);
    this.gridDelta(leaf, -1, edits);
    // withdraw its merge vote from the parent, if any
    if (leaf.votedMerge && leaf.parent) {
      leaf.parent.mergeVotes--;
      leaf.votedMerge = false;
    }
  }

  // ---- boot: seed the ancestor-closed subtree at a pose --------------------------

  /**
   * The resident subtree at a spawn pose: every fringe leaf of the desired cut
   * PLUS its ancestors (parked). Returned as a flat descriptor list the host
   * bakes (measuring pool caps) BEFORE the pool is reserved; then `seedBoot`
   * installs them. Roots first, so the host can attach coarse-first.
   */
  bootDescs(camX: number, camZ: number): { desc: QuadDesc; isLeaf: boolean; nodeId: number }[] {
    if (this.roots.length > 0) throw new Error('PartitionTree: bootDescs after seed');
    const out: { desc: QuadDesc; isLeaf: boolean; nodeId: number }[] = [];
    const rootT = this.rootTexels;
    const rootLevel = this.cfg.levels - 1;
    // fixed root tiling of the coverage box
    for (let gz = this.gridMinZ; gz < this.gridMinZ + this.gridH; gz++) {
      for (let gx = this.gridMinX; gx < this.gridMinX + this.gridW; gx++) {
        const tx0 = gx * rootT;
        const tz0 = gz * rootT;
        if (!this.onField(tx0, tz0, rootT)) continue;
        this.bootWalk(rootLevel, tx0, tz0, null, -1, camX, camZ, out);
      }
    }
    return out;
  }

  /** recursively realise the desired cut: a node is a leaf when the camera does
   *  not want it finer, else a parked Split whose children recurse. */
  private bootWalk(
    level: number,
    tx0: number,
    tz0: number,
    parent: Split | null,
    quadrant: number,
    camX: number,
    camZ: number,
    out: { desc: QuadDesc; isLeaf: boolean; nodeId: number }[],
  ): void {
    const size = this.cfg.gridN << level;
    const key = this.quadKey(level, tx0, tz0);
    const commonBase = { id: this.nextNodeId++, level, tx0, tz0, size, key, parent, quadrant };
    const wantFiner =
      level > 0
      && this.nearDist(commonBase, camX, camZ) < this.ringR(level - 1)
      && (this.deps.canRefine ? this.deps.canRefine(level, tx0, tz0, size) : true);
    if (!wantFiner) {
      const leaf: Leaf = { ...commonBase, kind: 'leaf', slot: -1, tx: null, votedMerge: false };
      this.pendingBootNodes.set(leaf.id, leaf);
      out.push({ desc: this.leafDesc(leaf), isLeaf: true, nodeId: leaf.id });
      if (parent) parent.children[quadrant >= 0 ? quadrant : parent.children.length] = leaf;
      else this.roots.push(leaf);
      return;
    }
    const split: Split = { ...commonBase, kind: 'split', children: [], parkedSlot: -1, mergeVotes: 0 };
    this.pendingBootNodes.set(split.id, split);
    out.push({ desc: this.splitDesc(split), isLeaf: false, nodeId: split.id });
    if (parent) parent.children[quadrant >= 0 ? quadrant : parent.children.length] = split;
    else this.roots.push(split);
    const cs = this.cfg.gridN << (level - 1);
    let q = 0;
    for (let j = 0; j < 2; j++) {
      for (let i = 0; i < 2; i++) {
        const cx = tx0 + i * cs;
        const cz = tz0 + j * cs;
        if (!this.onField(cx, cz, cs)) continue;
        this.bootWalk(level - 1, cx, cz, split, q++, camX, camZ, out);
      }
    }
  }

  private pendingBootNodes = new Map<number, Node>();

  private leafDesc(leaf: Leaf): QuadDesc {
    return { slot: -1, level: leaf.level, tx0: leaf.tx0, tz0: leaf.tz0, size: leaf.size, key: leaf.key };
  }
  private splitDesc(split: Split): QuadDesc {
    return { slot: -1, level: split.level, tx0: split.tx0, tz0: split.tz0, size: split.size, key: split.key };
  }

  /**
   * Install the boot descriptors into slots (the host assigned one slot per
   * descriptor when it attached the baked geometry). Leaves join the fringe;
   * Splits keep their slot as `parkedSlot`. Returns the level-grid the host must
   * upload once (the boot surface authority).
   */
  seedBoot(slotOf: Map<number, number>): LevelGridEdit[] {
    const edits: LevelGridEdit[] = [];
    for (const [id, node] of this.pendingBootNodes) {
      const slot = slotOf.get(id);
      if (slot === undefined) throw new Error(`PartitionTree.seedBoot: node ${id} got no slot`);
      this.nodeIndex.set(id, node);
      if (node.kind === 'leaf') {
        node.slot = slot;
        this.addLeaf(node, edits);
      } else {
        node.parkedSlot = slot;
      }
    }
    // fringe boots AT the desired cut ⇒ every leaf's merge vote is false (nothing
    // wants to coarsen at the cut); the first real tick computes true votes.
    this.pendingBootNodes.clear();
    return edits;
  }

  // ---- pose tick (iterates the FRINGE only) --------------------------------------

  /**
   * One residency tick at a camera texel pose. Iterates the fringe: each leaf
   * (re)decides its refine tx and its merge vote; committed merges cascade
   * bottom-up. `budget` caps new refine txs this tick (bake bandwidth);
   * `coarsestFirst` orders them for boot/teleport re-seed vs finest-first roaming.
   */
  tick(camX: number, camZ: number, budget: number, coarsestFirst: boolean): void {
    // snapshot: merges mutate the fringe, so iterate a stable copy
    const leaves = [...this.fringe];
    const splitCandidates: Leaf[] = [];
    let txLive = 0;
    let camLevel = -1;
    for (const leaf of leaves) {
      // observability: the fringe level rendering UNDER the camera + live txs —
      // "wants finer but hasn't" must be a visible state, never a quiet coarse patch
      if (leaf.tx) txLive++;
      if (camX >= leaf.tx0 && camX < leaf.tx0 + leaf.size && camZ >= leaf.tz0 && camZ < leaf.tz0 + leaf.size) camLevel = leaf.level;
      // refine intent — ring geometry AND finer child data actually existing
      // (else the split's bake can only abort + re-nominate forever; §data-gate)
      const wantFiner =
        leaf.level > 0
        && this.nearDist(leaf, camX, camZ) < this.ringR(leaf.level - 1)
        && this.canRefine(leaf);
      if (wantFiner) {
        if (!leaf.tx) splitCandidates.push(leaf);
      } else if (leaf.tx) {
        this.cancelRefine(leaf); // no longer wants to split — abort the bakes
      }
      // merge vote (updates parent.mergeVotes; a Split at full votes merges below)
      this.evalMergeVote(leaf, camX, camZ);
    }
    this.camLevel = camLevel;
    // fire merges (cascades upward as new leaves appear)
    this.processMerges(camX, camZ);
    // start refine txs under budget, ordered
    splitCandidates.sort((a, b) =>
      coarsestFirst ? b.level - a.level : a.level - b.level || this.nearDist(a, camX, camZ) - this.nearDist(b, camX, camZ),
    );
    let started = 0;
    for (const leaf of splitCandidates) {
      if (started >= budget) break;
      if (!this.fringe.has(leaf) || leaf.tx) continue; // a merge may have consumed it
      this.startRefine(leaf);
      started++;
    }
    this.txLiveCount = txLive + started;
    this.wantDeferred = splitCandidates.length - started;
  }

  /** observability (HUD counters): the fringe level under the camera (-1 = off
   *  coverage), refine txs in flight, and split-wanting leaves the budget deferred
   *  this tick. */
  private camLevel = -1;
  private txLiveCount = 0;
  private wantDeferred = 0;

  /** (re)evaluate whether a leaf votes to merge its parent, and reflect the
   *  delta into parent.mergeVotes. A leaf votes iff the PARENT footprint is past
   *  the merge radius (past R·H — hysteresis). */
  private evalMergeVote(leaf: Leaf, camX?: number, camZ?: number): void {
    const parent = leaf.parent;
    if (!parent) return;
    let vote = false;
    if (camX !== undefined && camZ !== undefined) {
      // parent wants to STAY split while dParent < R(parent.level-1); the child
      // votes to merge once dParent ≥ R·H (past the hysteresis band).
      vote = this.nearDist(parent, camX, camZ) >= this.ringR(parent.level - 1) * MERGE_HYSTERESIS;
    }
    if (vote !== leaf.votedMerge) {
      leaf.votedMerge = vote;
      parent.mergeVotes += vote ? 1 : -1;
      // O(1): a vote that completes the parent's tally enqueues exactly it — no
      // sibling scan, no tree walk (§2).
      if (vote && this.mergeReady(parent)) this.mergeQueue.push(parent);
    }
  }

  /** Splits ready to fire this tick (each pushed by the vote that completed it). */
  private mergeQueue: Split[] = [];

  private processMerges(camX: number, camZ: number): void {
    // drain the queue; a merge births a coarser leaf whose own vote may complete
    // the grandparent → it re-enqueues from evalMergeVote. Cascades bottom-up.
    while (this.mergeQueue.length > 0) {
      const split = this.mergeQueue.shift() as Split;
      if (!this.mergeReady(split)) continue; // a sibling re-split since it enqueued
      this.doMerge(split, camX, camZ);
    }
  }

  private mergeReady(split: Split): boolean {
    if (split.children.length === 0) return false;
    for (const c of split.children) if (c.kind !== 'leaf') return false;
    return split.mergeVotes >= split.children.length;
  }

  // ---- refine transaction (a field of the leaf) ----------------------------------

  private startRefine(leaf: Leaf): void {
    const descs = this.childDescs(leaf);
    if (descs.length === 0) return; // fully off-field (should not happen for a resident leaf)
    const slots = this.deps.reserveSlots(descs.length); // throws if dry (§5)
    const txId = this.nextTxId++;
    const quads: QuadDesc[] = descs.map((d, i) => ({ ...d, slot: slots[i] as number }));
    const tx: Refine = { txId, pending: quads.length, quads, payloads: new Array(quads.length).fill(undefined) };
    leaf.tx = tx;
    this.deps.startBake({ nodeId: leaf.id, txId, quads });
  }

  private cancelRefine(leaf: Leaf): void {
    const tx = leaf.tx;
    if (!tx) return;
    this.deps.releaseSlots(tx.quads.map((q) => q.slot));
    leaf.tx = null;
    this.nRefineCancel++;
  }

  /** abort a refine tx by (nodeId, txId) — a child bake failed / over-cap, so the
   *  whole refine cannot commit; release its slots and drop it. No-op if the tx is
   *  already gone (a race with a pose-tick cancel). */
  abortRefine(nodeId: number, txId: number): void {
    const node = this.nodeById(nodeId);
    if (!node || node.kind !== 'leaf' || !node.tx || node.tx.txId !== txId) return;
    this.cancelRefine(node);
  }

  /**
   * A baked child payload arrives, keyed to (nodeId, txId). Live tx → record it,
   * decrement pending, and commit at 0. Dead tx (the leaf cancelled or already
   * split via another path) → no home, dropped at this O(1) check (§3 — the
   * reserved slots were already returned by cancelRefine).
   */
  onBake(nodeId: number, txId: number, quadIndex: number, payload: unknown): void {
    const node = this.nodeById(nodeId);
    if (!node || node.kind !== 'leaf' || !node.tx || node.tx.txId !== txId) {
      this.nStaleDrop++;
      return;
    }
    const tx = node.tx;
    if (tx.payloads[quadIndex] !== undefined) return; // duplicate delivery — idempotent
    tx.payloads[quadIndex] = payload;
    tx.pending--;
    if (tx.pending === 0) this.commitRefine(node);
  }

  private commitRefine(leaf: Leaf): void {
    const tx = leaf.tx as Refine;
    const edits: LevelGridEdit[] = [];
    // Leaf → Split: the leaf's own slot is PARKED (retained as the parent payload,
    // instant coarsen); the reserved child slots become child leaves.
    const split: Split = {
      id: leaf.id,
      kind: 'split',
      level: leaf.level,
      tx0: leaf.tx0,
      tz0: leaf.tz0,
      size: leaf.size,
      key: leaf.key,
      parent: leaf.parent,
      quadrant: leaf.quadrant,
      children: [],
      parkedSlot: leaf.slot,
      mergeVotes: 0,
    };
    this.removeLeaf(leaf, edits); // leaves the fringe (also drops its own merge vote)
    if (leaf.parent) leaf.parent.children[leaf.quadrant] = split;
    else this.roots[this.roots.indexOf(leaf)] = split;
    this.nodeIndex.set(split.id, split);

    const children: { desc: QuadDesc; payload: unknown }[] = [];
    tx.quads.forEach((q, i) => {
      const child: Leaf = {
        id: this.nextNodeId++,
        kind: 'leaf',
        level: q.level,
        tx0: q.tx0,
        tz0: q.tz0,
        size: q.size,
        key: q.key,
        parent: split,
        quadrant: i,
        slot: q.slot,
        tx: null,
        votedMerge: false,
      };
      split.children.push(child);
      this.nodeIndex.set(child.id, child);
      this.addLeaf(child, edits);
      children.push({ desc: q, payload: tx.payloads[i] });
    });
    leaf.tx = null;
    this.nRefineCommit++;
    this.deps.emitRefine({ parkSlot: split.parkedSlot, children, levelGrid: edits });
  }

  private doMerge(split: Split, camX: number, camZ: number): Leaf {
    const edits: LevelGridEdit[] = [];
    const freeSlots: number[] = [];
    for (const c of split.children) {
      const cl = c as Leaf;
      this.removeLeaf(cl, edits);
      freeSlots.push(cl.slot);
      this.nodeIndex.delete(cl.id);
    }
    // Split → Leaf: unpark the retained parent payload (instant, always ready).
    const leaf: Leaf = {
      id: split.id,
      kind: 'leaf',
      level: split.level,
      tx0: split.tx0,
      tz0: split.tz0,
      size: split.size,
      key: split.key,
      parent: split.parent,
      quadrant: split.quadrant,
      slot: split.parkedSlot,
      tx: null,
      votedMerge: false,
    };
    if (split.parent) split.parent.children[split.quadrant] = leaf;
    else this.roots[this.roots.indexOf(split)] = leaf;
    this.nodeIndex.set(leaf.id, leaf);
    this.addLeaf(leaf, edits);
    this.evalMergeVote(leaf, camX, camZ); // the new coarser leaf may itself want to merge up
    this.deps.releaseSlots(freeSlots);
    this.nMerge++;
    this.deps.emitMerge({ unparkSlot: leaf.slot, freeSlots, levelGrid: edits });
    return leaf;
  }

  // ---- teleport ------------------------------------------------------------------

  /**
   * Teleport: cancel every in-flight refine (their bakes are abandoned; slots
   * returned). The next tick's votes then cascade the fringe back toward coarse
   * (always ready) and re-refine toward the new pose. O(fringe).
   */
  teleport(): void {
    this.nTeleports++;
    for (const leaf of this.fringe) if (leaf.tx) this.cancelRefine(leaf);
  }

  // ---- node index (O(1) (nodeId → node) for bake delivery) -----------------------

  private readonly nodeIndex = new Map<number, Node>();
  private nodeById(id: number): Node | undefined {
    return this.nodeIndex.get(id) ?? this.pendingBootNodes.get(id);
  }

  // ---- introspection / dev invariant assert (§2 — the ONLY full-tree walk) -------

  get fringeSize(): number {
    return this.fringe.size;
  }

  /** iterate every RESIDENT tile footprint — fringe leaves (rendering) AND parked
   *  Split parents (retained). The plane-scroll demote check reads this to avoid
   *  overwriting height texels a resident bake sampled. O(resident subtree). */
  forEachResident(cb: (tx0: number, tz0: number, size: number, level: number) => void): void {
    const walk = (n: Node): void => {
      if (n.kind === 'leaf') cb(n.tx0, n.tz0, n.size, n.level);
      else {
        cb(n.tx0, n.tz0, n.size, n.level); // parked parent still resident
        for (const c of n.children) walk(c);
      }
    };
    for (const r of this.roots) walk(r);
  }
  levelGridSnapshot(): { w: number; h: number; minX: number; minZ: number; data: Uint8Array } {
    return { w: this.gridW, h: this.gridH, minX: this.gridMinX, minZ: this.gridMinZ, data: this.gridLevel.slice() };
  }
  counters(): Record<string, number> {
    return {
      'tree.fringe': this.fringe.size,
      'tree.cam.level': this.camLevel,
      'tree.tx.live': this.txLiveCount,
      'tree.want.deferred': this.wantDeferred,
      'tree.refine.commit': this.nRefineCommit,
      'tree.refine.cancel': this.nRefineCancel,
      'tree.merge': this.nMerge,
      'tree.stale.drop': this.nStaleDrop,
      'tree.teleports': this.nTeleports,
    };
  }

  /**
   * Full-tree invariant assert — DEV ONLY (throws loud). Proves the type-level
   * guarantees hold at runtime:
   *  - the fringe partitions the coverage box EXACTLY once (no dual-LOD, no gap);
   *  - ≤1 tx per node, every reserved slot owned by exactly one live tx or node;
   *  - slot accounting exact (each slot used by ≤1 resident node / reservation);
   *  - every Split has a parked parent slot (coarsen always ready).
   */
  checkInvariants(usedSlots: Set<number>): void {
    // 1. partition: walk the tree, collect leaf footprints; assert they tile the
    //    field with no overlap and no gap (sample the coverage box).
    const seenSlots = new Set<number>();
    const leaves: Leaf[] = [];
    const walk = (n: Node): void => {
      if (n.kind === 'leaf') {
        leaves.push(n);
        if (n.slot < 0) throw new Error(`tree invariant: leaf ${n.key} has no slot`);
        if (seenSlots.has(n.slot)) throw new Error(`tree invariant: slot ${n.slot} double-used (leaf ${n.key})`);
        seenSlots.add(n.slot);
        if (n.tx) {
          for (const q of n.tx.quads) {
            if (seenSlots.has(q.slot)) throw new Error(`tree invariant: reserved slot ${q.slot} double-used`);
            seenSlots.add(q.slot);
          }
        }
      } else {
        if (n.parkedSlot < 0) throw new Error(`tree invariant: split ${n.key} has no parked slot (coarsen not ready)`);
        if (seenSlots.has(n.parkedSlot)) throw new Error(`tree invariant: parked slot ${n.parkedSlot} double-used`);
        seenSlots.add(n.parkedSlot);
        if (n.children.length === 0) throw new Error(`tree invariant: split ${n.key} has no children`);
        for (const c of n.children) {
          if (c.parent !== n) throw new Error(`tree invariant: child ${c.key} parent link broken`);
          walk(c);
        }
      }
    };
    for (const r of this.roots) walk(r);
    // fringe set == the walked leaves
    if (this.fringe.size !== leaves.length) {
      throw new Error(`tree invariant: fringe set ${this.fringe.size} != walked leaves ${leaves.length}`);
    }
    for (const l of leaves) if (!this.fringe.has(l)) throw new Error(`tree invariant: leaf ${l.key} not in fringe set`);
    // 2. the tree's slot set == the host's used-slot set (no leak, no phantom)
    for (const s of seenSlots) if (!usedSlots.has(s)) throw new Error(`tree invariant: slot ${s} live in tree but free on host`);
    for (const s of usedSlots) if (!seenSlots.has(s)) throw new Error(`tree invariant: slot ${s} used on host but not in tree`);
    // 3. exact partition of coverage: sample a fine grid; every sample lands in
    //    EXACTLY one leaf footprint.
    const step = Math.max(1, this.cfg.gridN >> 1);
    for (let pz = this.cfg.latMin; pz <= this.cfg.latMax; pz += step) {
      for (let px = this.cfg.latMin; px <= this.cfg.latMax; px += step) {
        let cover = 0;
        for (const l of leaves) {
          if (px >= l.tx0 && px < l.tx0 + l.size && pz >= l.tz0 && pz < l.tz0 + l.size) cover++;
        }
        if (cover !== 1) throw new Error(`tree invariant: point ${px},${pz} covered by ${cover} leaves (want exactly 1)`);
      }
    }
  }
}
