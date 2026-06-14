/**
 * N8-D2 Stage 2b-3 (D-N39): the per-frame terrain CLIPMAP streamer. Makes the
 * 1 m detail FOLLOW the camera instead of sitting field-centered (2b-2).
 *
 * Residency model (the Nanite "page" loop): the field is a geometry clipmap —
 * L concentric same-gridN rings at doubling stride, centered on the camera. The
 * desired resident set is `clipmapTiles(camTexel)`. Each frame `update()` diffs
 * desired-vs-resident against the fixed slot pool (GeometryRegistry, Stage 2a):
 *   - EVICT departed tiles immediately (cheap: clusterCount→0 + sphere off-world
 *     ⇒ kInstCull drops the instance ⇒ zero clusters enqueued) — bounds memory +
 *     the vista cull every frame, never blocked on a build.
 *   - LOAD arrived tiles asynchronously, a few per frame (build on the persistent
 *     DagWorker — cache hit instant / miss off-thread — then cap-check → alloc a
 *     slot → attach). The COARSER already-resident ring backstops a fine tile
 *     WHILE it streams in ⇒ graceful, never a hole (the no-fallback floor).
 *
 * Key invariants that keep it leak- and hitch-free:
 *   - build BEFORE alloc: a slot is taken only once a tile's geometry is in hand,
 *     so an in-flight build never holds (or leaks) a slot.
 *   - cap pre-check BEFORE alloc: an oversized region is SKIPPED (coarser ring
 *     backstops) instead of throwing inside attachHeightDagTile + leaking the slot.
 *   - single-flight: `update()` only kicks a build batch when idle; intervening
 *     frames just overwrite `pendingCam` (coalesce to the latest), and the next
 *     frame re-drives the drain — frame-paced, no microtask spin.
 *
 * Boot (no fallback): WorldRegistry calls buildBootSet() PRE-build (builds the
 * spawn-centered ring set + measures the pool caps) then attachBootSet() POST-
 * build (frame-1 terrain). The worker is PERSISTENT — the streamer owns it and it
 * outlives boot so update() can build new tiles as the camera roams.
 */
import { type DagCluster } from './BuildDag';
import { buildHeightDag } from './BuildHeightDag';
import { getCachedHeightDag, heightDagCacheKey, putCachedHeightDag } from './DagCache';
import { type DagBuilder, type HeightDagResult } from './DagWorkerClient';
import { clipmapTiles, clipmapMaxTiles, type ClipmapConfig, type ClipmapTile } from './TerrainClipmap';
import type { GeometryRegistry } from './GeometryRegistry';

/** at most this many tile arrivals per diff batch. They BAKE CONCURRENTLY (#32 —
 *  one per DagWorkerPool thread), so this bounds BOTH the parallel-build width and
 *  the main-thread attach cost; the remainder streams over subsequent frames
 *  (update() re-drives). Match it to the pool size so a full batch maps 1:1 to
 *  threads — a batch then costs ~one bake, not N serial bakes. */
const MAX_LOADS_PER_DIFF = 4;

/** everything buildTerrainTile needs that is constant for the field's lifetime. */
export interface TileBuildDeps {
  heights: Float32Array;
  res: number;
  cell: number;
  origin: number;
  gridN: number;
  /** numeric seed for the per-tile DAG cache, or null to always build (no cache). */
  seed: number | null;
  /** persistent off-thread builder — a single Worker or a DagWorkerPool (concurrent
   *  bakes) — or null to build synchronously on this thread. */
  worker: DagBuilder | null;
}

/** mutable cache-hit / fresh-build tally threaded through a build pass. */
export interface TileBuildStats {
  nCache: number;
  nBuilt: number;
}

/** deterministic per-tile cache suffix (key = seed + gridN + this). */
function tileSuffix(gridN: number, strideTexels: number, tx0: number, tz0: number): string {
  return `-clip-g${gridN}-s${strideTexels}-${tx0}x${tz0}`;
}

function clampTexel(t: number, res: number): number {
  return t < 0 ? 0 : t > res - 1 ? res - 1 : t;
}

/**
 * Build ONE terrain tile DAG at texel origin (tx0,tz0), `tileStride` texels/cell,
 * gridN cells — subsample the field (cache-aware), build the DAG (worker or sync),
 * and remap the tile-local grid coords to GLOBAL texel coords (so the GPU reads
 * the full-res heightTex). Off-field samples clamp to the field edge. Shared by
 * the uniform-tile path (WorldRegistry) and the clipmap streamer.
 */
export async function buildTerrainTile(
  deps: TileBuildDeps,
  tx0: number,
  tz0: number,
  tileStride: number,
  suffix: string,
  stats: TileBuildStats,
  onDeferred: (msg: string) => void,
): Promise<{ gridVerts: Uint32Array; built: HeightDagResult }> {
  const { heights, res, cell, origin, gridN, seed, worker } = deps;
  const vpa = gridN + 1;
  const cacheKey = seed != null ? heightDagCacheKey(seed, gridN, suffix) : null;
  let built: HeightDagResult | null = cacheKey ? await getCachedHeightDag(cacheKey) : null;
  if (built) {
    stats.nCache++;
  } else {
    const sub = new Float32Array(vpa * vpa);
    for (let gz = 0; gz <= gridN; gz++) {
      const texZ = clampTexel(tz0 + gz * tileStride, res);
      const trow = texZ * res;
      const srow = gz * vpa;
      for (let gx = 0; gx <= gridN; gx++) {
        const texX = clampTexel(tx0 + gx * tileStride, res);
        sub[srow + gx] = heights[trow + texX] as number;
      }
    }
    const hfArgs = {
      heights: sub,
      gridN,
      cellSize: cell * tileStride,
      originX: origin + tx0 * cell,
      originZ: origin + tz0 * cell,
    };
    if (worker) {
      try {
        built = await worker.buildHeight(hfArgs);
      } catch (e) {
        onDeferred(`terrain DAG tile ${suffix}: worker failed (${e instanceof Error ? e.message : String(e)}) → sync`);
        built = buildHeightDag(hfArgs, {});
      }
    } else {
      built = buildHeightDag(hfArgs, {});
    }
    if (cacheKey) void putCachedHeightDag(cacheKey, built); // fire-and-forget
    stats.nBuilt++;
  }
  // remap tile-local grid coords (packed u16 gx|gz<<16) to GLOBAL texel coords
  const gridVerts = new Uint32Array(built.gridVerts.length);
  for (let i = 0; i < built.gridVerts.length; i++) {
    const p = built.gridVerts[i] as number;
    const texX = clampTexel(tx0 + (p & 0xffff) * tileStride, res);
    const texZ = clampTexel(tz0 + ((p >>> 16) & 0xffff) * tileStride, res);
    gridVerts[i] = ((texX & 0xffff) | ((texZ & 0xffff) << 16)) >>> 0;
  }
  return { gridVerts, built };
}

export interface StreamerDeps extends TileBuildDeps {
  reg: GeometryRegistry;
}

interface BootTile {
  key: string;
  x0: number;
  z0: number;
  size: number;
  gridVerts: Uint32Array;
  indices: Uint32Array;
  clusters: DagCluster[];
}

/** a resident tile: its pool slot + texel footprint [x0,x0+size]×[z0,z0+size], used
 *  for footprint-distance-ordered eviction under pool pressure (a coarse tile whose
 *  footprint still covers the camera has distance 0 ⇒ is reclaimed LAST — it is the
 *  backstop). */
interface ResidentTile {
  slot: number;
  x0: number;
  z0: number;
  size: number;
}

/** distance² from point (px,pz) to the axis-aligned tile footprint (0 if inside). */
function footprintDist2(t: ResidentTile, px: number, pz: number): number {
  const dx = px < t.x0 ? t.x0 - px : px > t.x0 + t.size ? px - (t.x0 + t.size) : 0;
  const dz = pz < t.z0 ? t.z0 - pz : pz > t.z0 + t.size ? pz - (t.z0 + t.size) : 0;
  return dx * dx + dz * dz;
}

export class TerrainStreamer {
  private readonly cfg: ClipmapConfig;
  /** tileKey → resident tile (slot + center). Departed tiles linger here until a
   *  replacement covers them (lazy eviction) ⇒ size can transiently exceed maxTiles
   *  during a transition, bounded by the pool's slot count. */
  private readonly resident = new Map<string, ResidentTile>();
  /** tiles that overflowed their slot cap — held so runDiff does NOT rebuild them
   *  every frame (the coarser ring backstops them). Pruned when they leave desired
   *  so a later revisit re-attempts (e.g. after a config change). */
  private readonly skipped = new Set<string>();
  /** single-flight: a build batch is running; new update()s just set pendingCam. */
  private busy = false;
  /** latest requested camera (texel), coalesced while busy; drained by tick(). */
  private pendingCam: { tx: number; tz: number } | null = null;
  /** boot geometry held between buildBootSet (pre-build) and attachBootSet (post). */
  private bootBuilt: BootTile[] = [];
  private readonly bstats: TileBuildStats = { nCache: 0, nBuilt: 0 };
  private poolMaxV = 0;
  private poolMaxT = 0;
  private poolMaxC = 0;
  private bootCl = 0;
  private bootTris = 0;
  private bootMaxErr = 0;
  private bootOffGrid = 0;
  private capWarned = false;
  private fullWarned = false;
  private nLoaded = 0;
  private nEvicted = 0;
  private nSkipped = 0;

  constructor(
    private readonly deps: StreamerDeps,
    private readonly onDeferred: (msg: string) => void = () => {},
    M = 4,
  ) {
    const { res, gridN } = deps;
    // size levels so the COARSEST ring spans the field (always-resident backstop ⇒
    // no holes from any pose). Same formula WorldRegistry's static 2b-2 path used.
    const levels = Math.max(1, Math.ceil(Math.log2((2 * res) / (M * gridN))) + 1);
    this.cfg = { res, gridN, baseStride: 1, levels, tilesPerSide: M };
  }

  /** upper bound on the resident tile count (pool-sizing input). */
  get maxTiles(): number {
    return clipmapMaxTiles(this.cfg);
  }
  get levels(): number {
    return this.cfg.levels;
  }
  get clipDesc(): string {
    return `CLIPMAP ${this.cfg.levels}L M${this.cfg.tilesPerSide}`;
  }
  /** worst per-tile vert/tri/cluster counts seen at boot — the pool cap basis. */
  get poolMax(): { v: number; t: number; c: number } {
    return { v: this.poolMaxV, t: this.poolMaxT, c: this.poolMaxC };
  }
  get residentCount(): number {
    return this.resident.size;
  }
  bootSummary(): { tiles: number; tCl: number; tTris: number; maxErr: number; offGrid: number; nCache: number; nBuilt: number } {
    return {
      tiles: this.resident.size + this.bootBuilt.length,
      tCl: this.bootCl,
      tTris: this.bootTris,
      maxErr: this.bootMaxErr,
      offGrid: this.bootOffGrid,
      nCache: this.bstats.nCache,
      nBuilt: this.bstats.nBuilt,
    };
  }

  private measure(built: HeightDagResult, vCount: number): void {
    const tCount = built.indices.length / 3;
    if (vCount > this.poolMaxV) this.poolMaxV = vCount;
    if (tCount > this.poolMaxT) this.poolMaxT = tCount;
    if (built.clusters.length > this.poolMaxC) this.poolMaxC = built.clusters.length;
    this.bootCl += built.clusters.length;
    this.bootTris += tCount;
    if (built.stats.maxError > this.bootMaxErr) this.bootMaxErr = built.stats.maxError;
    this.bootOffGrid += built.stats.offGridVerts;
  }

  /**
   * PRE-build: build the spawn-centered ring set + measure the pool caps. Does
   * NOT attach (the registry isn't built yet). Returns the boot tile count.
   */
  async buildBootSet(camTexelX: number, camTexelZ: number): Promise<number> {
    const set = clipmapTiles(camTexelX, camTexelZ, this.cfg);
    // bake the WHOLE boot ring CONCURRENTLY across the pool (#32) — cold boot drops
    // from ~set×170 ms serial to ~ceil(set/threads)×170 ms. Collect in `set` order
    // (not completion order) so bootBuilt + the pool-cap measurement are deterministic
    // regardless of which bake lands first.
    const baked = await Promise.all(
      set.map((t) =>
        buildTerrainTile(
          this.deps,
          t.tx0,
          t.tz0,
          t.strideTexels,
          tileSuffix(this.cfg.gridN, t.strideTexels, t.tx0, t.tz0),
          this.bstats,
          this.onDeferred,
        ),
      ),
    );
    set.forEach((t, i) => {
      const r = baked[i] as { gridVerts: Uint32Array; built: HeightDagResult };
      this.bootBuilt.push({
        key: t.key,
        x0: t.tx0,
        z0: t.tz0,
        size: t.tileTexels,
        gridVerts: r.gridVerts,
        indices: r.built.indices,
        clusters: r.built.clusters,
      });
      this.measure(r.built, r.gridVerts.length);
    });
    return set.length;
  }

  /** POST-build: attach the boot set into pool slots (frame-1 terrain). */
  attachBootSet(): void {
    for (const b of this.bootBuilt) {
      const slot = this.deps.reg.allocTileSlot();
      if (slot < 0) {
        this.onDeferred('terrain stream: out of slots at boot (raise reserveTilePool slots)');
        break;
      }
      this.deps.reg.attachHeightDagTile(slot, { gridVerts: b.gridVerts, indices: b.indices, clusters: b.clusters });
      this.resident.set(b.key, { slot, x0: b.x0, z0: b.z0, size: b.size });
      this.nLoaded++;
    }
    this.bootBuilt = [];
  }

  /** per-frame hook: re-center the clipmap on the live camera (world XZ). */
  update(camWorldX: number, camWorldZ: number): void {
    const { origin, cell, res } = this.deps;
    const tx = clampTexel(Math.round((camWorldX - origin) / cell), res);
    const tz = clampTexel(Math.round((camWorldZ - origin) / cell), res);
    this.pendingCam = { tx, tz };
    if (!this.busy) void this.tick();
  }

  /** drain one build batch for the latest pendingCam (single-flight). */
  private async tick(): Promise<void> {
    this.busy = true;
    try {
      const cam = this.pendingCam;
      this.pendingCam = null;
      if (cam) await this.runDiff(cam.tx, cam.tz);
    } finally {
      this.busy = false;
    }
  }

  /**
   * Force the clipmap fully resident at a camera pose, ignoring the per-frame
   * load cap (boot-equivalent / headless probe). Loops runDiff until no tile
   * makes progress (all desired loaded or skipped). Production uses update().
   */
  async settleAt(camWorldX: number, camWorldZ: number): Promise<void> {
    const { origin, cell, res } = this.deps;
    const tx = clampTexel(Math.round((camWorldX - origin) / cell), res);
    const tz = clampTexel(Math.round((camWorldZ - origin) / cell), res);
    for (let i = 0; i < 64; i++) {
      const before = this.nLoaded + this.nSkipped;
      await this.runDiff(tx, tz);
      if (this.nLoaded + this.nSkipped === before) break; // no progress ⇒ settled
    }
  }

  /**
   * LAZY-evict + capped CONCURRENT load. Departed tiles are NOT dropped up front —
   * they keep RENDERING (their footprint stays covered) through the bake window until
   * either (a) the whole desired set is resident again ⇒ the new coverage is complete,
   * so cleanup drops the now-redundant stragglers, or (b) the pool runs dry and a slot
   * must be reclaimed ⇒ we evict the FARTHEST departed tile (the one least likely to
   * leave a visible hole: far away + most likely already covered by a coarser resident
   * ring + smallest on screen). This is the fix for "the detailed LOD vanishes before
   * the new one bakes in" — the old LOD survives until replaced.
   *
   * #32: the batch's tiles BAKE CONCURRENTLY (one per DagWorkerPool thread) — the
   * whole batch costs ~one bake instead of N serial bakes, shrinking the coarse→fine
   * pop window ~pool-size×. The attach (alloc slot → write → resident.set) then runs
   * synchronously over the baked results with NO await between alloc and attach, so
   * the slot pool stays consistent (single-flight already bars re-entry).
   */
  private async runDiff(tx: number, tz: number): Promise<void> {
    const desired = clipmapTiles(tx, tz, this.cfg);
    const want = new Set<string>();
    for (const t of desired) want.add(t.key);
    // forget skip marks for tiles that left desired (a revisit re-attempts them)
    for (const key of this.skipped) if (!want.has(key)) this.skipped.delete(key);
    // pick this batch's ARRIVALS (capped) — tiles desired but neither resident nor
    // skipped — then bake them ALL AT ONCE across the pool.
    const batch: ClipmapTile[] = [];
    for (const t of desired) {
      if (this.resident.has(t.key) || this.skipped.has(t.key)) continue;
      batch.push(t);
      if (batch.length >= MAX_LOADS_PER_DIFF) break;
    }
    if (batch.length > 0) {
      // BAKE concurrently — each buildTerrainTile lands on its own pool thread (cache
      // hits resolve instantly without touching a worker). Capture per-tile so one
      // failed bake doesn't reject the whole batch.
      const baked = await Promise.all(
        batch.map((t) =>
          buildTerrainTile(
            this.deps,
            t.tx0,
            t.tz0,
            t.strideTexels,
            tileSuffix(this.cfg.gridN, t.strideTexels, t.tx0, t.tz0),
            this.bstats,
            this.onDeferred,
          ).then(
            (r) => ({ t, r, err: null as unknown }),
            (err: unknown) => ({ t, r: null as { gridVerts: Uint32Array; built: HeightDagResult } | null, err }),
          ),
        ),
      );
      // ATTACH synchronously — reclaim a slot from the farthest DEPARTED tile only on
      // pool pressure (NOT an eager up-front evict that would hole). NO await between
      // allocTileSlot() and attach/resident.set ⇒ the slot pool can't be raced.
      const cap = this.deps.reg.tilePoolCap;
      for (const { t, r, err } of baked) {
        if (r === null) {
          this.onDeferred(`terrain stream tile ${t.key}: build failed (${err instanceof Error ? err.message : String(err)})`);
          continue;
        }
        const { gridVerts, built } = r;
        // cap pre-check BEFORE alloc — an oversized region is skipped (the coarser
        // ring backstops it), never thrown inside attach + leaking the slot.
        const tCount = built.indices.length / 3;
        if (gridVerts.length > cap.vertCap || tCount > cap.triCap || built.clusters.length > cap.clusterCap) {
          if (!this.capWarned) {
            this.capWarned = true;
            this.onDeferred(
              `terrain stream: tile ${t.key} over slot cap (v${gridVerts.length}/${cap.vertCap} ` +
                `t${tCount}/${cap.triCap} c${built.clusters.length}/${cap.clusterCap}) — SKIPPED, coarser ring backstops`,
            );
          }
          this.skipped.add(t.key); // don't rebuild it every frame
          this.nSkipped++;
          continue;
        }
        // build-then-alloc: take a slot only now that the geometry is in hand. On
        // pool pressure, reclaim one from the farthest departed tile (then re-alloc
        // it properly via the free stack — never reuse a just-evicted slot directly).
        let slot = this.deps.reg.allocTileSlot();
        if (slot < 0 && this.evictFarthestDeparted(tx, tz, want)) slot = this.deps.reg.allocTileSlot();
        if (slot < 0) {
          if (!this.fullWarned) {
            this.fullWarned = true;
            this.onDeferred('terrain stream: pool full of DESIRED tiles — raise reserveTilePool slots');
          }
          break;
        }
        this.deps.reg.attachHeightDagTile(slot, { gridVerts, indices: built.indices, clusters: built.clusters });
        this.resident.set(t.key, { slot, x0: t.tx0, z0: t.tz0, size: t.tileTexels });
        this.nLoaded++;
      }
    }
    // CLEANUP — once the whole desired set is resident the new coverage is complete,
    // so any lingering departed tile is now redundant ⇒ drop it (keeps memory lean
    // when the camera settles; during motion the stragglers stay as backstop).
    let allResident = true;
    for (const t of desired) {
      if (!this.resident.has(t.key)) {
        allResident = false;
        break;
      }
    }
    if (allResident) {
      for (const [key, rt] of [...this.resident]) {
        if (!want.has(key)) {
          this.deps.reg.evictHeightDagTile(rt.slot);
          this.resident.delete(key);
          this.nEvicted++;
        }
      }
    }
  }

  /**
   * Reclaim ONE slot by evicting the DEPARTED (not-currently-desired) resident tile
   * FARTHEST from the camera — least likely to leave a visible hole. Frees the slot
   * to the registry free-stack (the caller re-allocs it). Returns false (evicts
   * nothing) if every resident tile is currently desired — the pool is genuinely
   * too small, so the caller stops loading rather than drop a needed tile.
   */
  private evictFarthestDeparted(tx: number, tz: number, want: Set<string>): boolean {
    let worstKey: string | null = null;
    let worstSlot = -1;
    let worstD = -1;
    for (const [key, rt] of this.resident) {
      if (want.has(key)) continue; // never evict a tile we still want
      // FOOTPRINT distance — a coarse tile still covering the camera scores 0 and is
      // reclaimed last (it's the backstop); far small fine tiles go first.
      const d = footprintDist2(rt, tx, tz);
      if (d > worstD) {
        worstD = d;
        worstKey = key;
        worstSlot = rt.slot;
      }
    }
    if (worstKey == null) return false;
    this.deps.reg.evictHeightDagTile(worstSlot);
    this.resident.delete(worstKey);
    this.nEvicted++;
    return true;
  }

  /** HUD/probe snapshot. */
  counters(): Record<string, number> {
    return {
      'terrain.stream.resident': this.resident.size,
      'terrain.stream.loaded': this.nLoaded,
      'terrain.stream.evicted': this.nEvicted,
      'terrain.stream.skipped': this.nSkipped,
      'terrain.stream.cacheHit': this.bstats.nCache,
      'terrain.stream.built': this.bstats.nBuilt,
    };
  }

  /** the desired resident key set at a camera pose (probe/debug). */
  desiredKeysAt(camWorldX: number, camWorldZ: number): Set<string> {
    const { origin, cell, res } = this.deps;
    const tx = clampTexel(Math.round((camWorldX - origin) / cell), res);
    const tz = clampTexel(Math.round((camWorldZ - origin) / cell), res);
    const out = new Set<string>();
    for (const t of clipmapTiles(tx, tz, this.cfg)) out.add(t.key);
    return out;
  }

  /** the live resident key set (probe/debug). */
  residentKeys(): Set<string> {
    return new Set(this.resident.keys());
  }

  dispose(): void {
    this.deps.worker?.dispose();
  }
}
