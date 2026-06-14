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
import { type DagBuildWorker, type HeightDagResult } from './DagWorkerClient';
import { clipmapTiles, clipmapMaxTiles, type ClipmapConfig } from './TerrainClipmap';
import type { GeometryRegistry } from './GeometryRegistry';

/** at most this many tile LOADS per frame batch — bounds the main-thread attach
 *  cost; the remainder streams over subsequent frames (update() re-drives). */
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
  /** persistent off-thread builder, or null to build synchronously on this thread. */
  worker: DagBuildWorker | null;
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
  gridVerts: Uint32Array;
  indices: Uint32Array;
  clusters: DagCluster[];
}

export class TerrainStreamer {
  private readonly cfg: ClipmapConfig;
  /** tileKey → occupied pool slot. The live resident set; size ≤ maxTiles. */
  private readonly resident = new Map<string, number>();
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
    for (const t of set) {
      const { gridVerts, built } = await buildTerrainTile(
        this.deps,
        t.tx0,
        t.tz0,
        t.strideTexels,
        tileSuffix(this.cfg.gridN, t.strideTexels, t.tx0, t.tz0),
        this.bstats,
        this.onDeferred,
      );
      this.bootBuilt.push({ key: t.key, gridVerts, indices: built.indices, clusters: built.clusters });
      this.measure(built, gridVerts.length);
    }
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
      this.resident.set(b.key, slot);
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

  /** evict everything departed (sync) + load a capped batch of everything arrived. */
  private async runDiff(tx: number, tz: number): Promise<void> {
    const desired = clipmapTiles(tx, tz, this.cfg);
    const want = new Set<string>();
    for (const t of desired) want.add(t.key);
    // EVICT departed — synchronous, cheap, runs EVERY batch (bounds memory + vista)
    for (const [key, slot] of [...this.resident]) {
      if (!want.has(key)) {
        this.deps.reg.evictHeightDagTile(slot);
        this.resident.delete(key);
        this.nEvicted++;
      }
    }
    // forget skip marks for tiles that left desired (a revisit re-attempts them)
    for (const key of this.skipped) if (!want.has(key)) this.skipped.delete(key);
    // LOAD arrived — capped per batch; coarser resident ring backstops the rest
    const cap = this.deps.reg.tilePoolCap;
    let loaded = 0;
    for (const t of desired) {
      if (this.resident.has(t.key) || this.skipped.has(t.key)) continue;
      if (loaded >= MAX_LOADS_PER_DIFF) break;
      let gridVerts: Uint32Array;
      let built: HeightDagResult;
      try {
        const r = await buildTerrainTile(
          this.deps,
          t.tx0,
          t.tz0,
          t.strideTexels,
          tileSuffix(this.cfg.gridN, t.strideTexels, t.tx0, t.tz0),
          this.bstats,
          this.onDeferred,
        );
        gridVerts = r.gridVerts;
        built = r.built;
      } catch (e) {
        this.onDeferred(`terrain stream tile ${t.key}: build failed (${e instanceof Error ? e.message : String(e)})`);
        continue;
      }
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
      // build-then-alloc: take a slot only now that the geometry is in hand
      const slot = this.deps.reg.allocTileSlot();
      if (slot < 0) {
        if (!this.fullWarned) {
          this.fullWarned = true;
          this.onDeferred('terrain stream: pool full — deferring loads (raise reserveTilePool slots)');
        }
        break;
      }
      this.deps.reg.attachHeightDagTile(slot, { gridVerts, indices: built.indices, clusters: built.clusters });
      this.resident.set(t.key, slot);
      this.nLoaded++;
      loaded++;
    }
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
