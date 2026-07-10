/**
 * StreamBrainCore — the streaming brain's logic (SPEC-STREAMING-WORLD §4, S5),
 * host-agnostic: StreamBrain.worker wires it to postMessage + a nested
 * DagWorkerPool; node probes drive it directly with a fake source. It owns
 * EVERY streaming policy decision (law 5 — the main thread only executes):
 *
 *  - PLANE WINDOW residency: per-level camera windows over the source lattice;
 *    a level-k chunk is fetched iff it overlaps level k's window (the demand
 *    law — no separate rings). Pinned levels (window covers the layer — every
 *    generated level) never scroll; wrapping levels scroll toroidally in
 *    SCROLL_STEP texel snaps with demote-before-scroll enforcement: a scroll
 *    that would overwrite texels a resident tile was baked against is DEFERRED
 *    (counted) until the tile diff demotes it — packet order then carries the
 *    promote-after-fill half (fills precede the origin commit precede attaches).
 *  - TILE clipmap residency (the plan is IDENTICAL to the pre-S5 main-thread
 *    streamer — same clipmapTiles geometry, same lazy-evict/backstop rules —
 *    so the generated world stays steady-state-identical): bakes read the
 *    brain's retained height windows (decoded chunks, NEVER a main-thread
 *    cpuHeights mirror), coarsest-first at boot/teleport, finest-relevant-first
 *    roaming, velocity prefetch (DagCache warm) at queue-idle.
 *  - decoded-chunk LRU (96 MB, discardable-after-consume), fetch dedupe,
 *    teleport detection from pose discontinuity, HUD counters.
 *
 * Attach packets carry TILE-LOCAL grid coords + per-tile mesh-record origin
 * words (StreamOrigin-relative; F-3's rebase path). At the generated world's
 * origin (0,0) and 1 m cell these evaluate f32-BIT-IDENTICAL to the old
 * global-coord form (integer+0.5 magnitudes ≤ 2048 are exact), and they unlock
 * Estonia's large signed lattice where 13-bit global coords cannot reach.
 */

import { buildHeightGrid, type HeightDagOpts } from '../build/BuildHeightGrid';
import { getCachedHeightDag, heightDagCacheKey, packClusters, putCachedHeightDag } from '../build/DagCache';
import type { DagBuilder, HeightDagResult } from '../build/DagWorkerClient';
import { packChunkKey } from '../../world/source/Lac1';
import type { ChunkKey, ChunkPayload, LayerName } from '../../world/source/WorldSource';
import { clipmapMaxTiles, clipmapTiles, type ClipmapConfig, type ClipmapTile } from './TerrainClipmap';
import {
  BIOME_CHANNELS,
  CANOPY_CHANNELS,
  FIELDS_CHANNELS,
  WATER_DRY_SENTINEL,
  WATER_FAR_FACTOR,
  chunksInWindow,
  clampExtend,
  copyChunkF32,
  copyChunkU8,
  minReduce,
  wrapRects,
  type FilledBox,
  type PlanePlan,
  type RasterGeom,
} from './PlaneFill';
import type {
  BrainInitMsg,
  BrainLayerMeta,
  BrainToMain,
  PlaneKind,
  PoolInfoMsg,
  StreamPacket,
  TileGeometry,
} from './StreamProtocol';

/** decoded-chunk LRU budget (spec §2: nothing pinned — plane windows are the
 *  persistent store; entries are discardable after plane-fill + tile bakes). */
const LRU_BYTES = 96 * 2 ** 20;
/** arrivals baked per diff batch — bounds parallel-bake width AND per-frame
 *  main-thread attach cost (same value as the pre-S5 streamer). */
const MAX_LOADS_PER_DIFF = 4;
/** wrapping plane windows re-center in snaps of res/8 texels (bounded packet
 *  counts; hysteresis = half a snap). */
const SCROLL_DIV = 8;
/** prefetch horizon (s) along the velocity estimate at queue-idle. */
const PREFETCH_S = 2.5;
const PREFETCH_MAX = 2;

interface HeightWindow {
  plan: PlanePlan;
  /** PHYSICAL (toroidal) layout, res² */
  data: Float32Array;
  n0x: number;
  n0z: number;
  phaseX: number;
  phaseY: number;
}

interface ResidentTile {
  x0: number;
  z0: number;
  size: number;
  /** clipmap level (probe) + the height-window level the bake sampled */
  level: number;
  srcLevel: number;
}

function footprintDist2(t: ResidentTile, px: number, pz: number): number {
  const dx = px < t.x0 ? t.x0 - px : px > t.x0 + t.size ? px - (t.x0 + t.size) : 0;
  const dz = pz < t.z0 ? t.z0 - pz : pz > t.z0 + t.size ? pz - (t.z0 + t.size) : 0;
  return dx * dx + dz * dz;
}

export interface BrainDeps {
  fetch(layer: LayerName, key: ChunkKey): Promise<ChunkPayload | null>;
  emit(msg: BrainToMain, transfer?: Transferable[]): void;
  /** nested bake pool, or null → synchronous buildHeightGrid in this thread */
  bake: DagBuilder | null;
}

export class StreamBrainCore {
  private readonly deps: BrainDeps;
  private grid!: BrainInitMsg['grid'];
  private layers!: BrainInitMsg['layers'];
  private plan!: BrainInitMsg['plan'];
  private tilesCfg!: BrainInitMsg['tiles'];
  private cfg!: ClipmapConfig;
  /** packed-key existence sets + content hashes per layer+lod */
  private readonly keySets = new Map<string, Set<number>>();
  private readonly keyHashes = new Map<string, Map<number, bigint>>();

  // plane windows (height retained for tile bakes; water retained for far reduce)
  private hWin: HeightWindow[] = [];
  private wWin: HeightWindow | null = null;

  // decoded-chunk LRU
  private readonly lru = new Map<string, { payload: ChunkPayload; bytes: number }>();
  private lruBytes = 0;
  private readonly inFlight = new Map<string, Promise<ChunkPayload | null>>();

  // tile residency
  private readonly resident = new Map<string, ResidentTile>();
  private readonly pendingAttach = new Map<string, ResidentTile>();
  private readonly skipped = new Set<string>();
  private readonly stalled = new Set<string>();
  private readonly prefetched = new Set<string>();
  private lastWant = new Set<string>();
  private pool: PoolInfoMsg | null = null;
  private busy = false;
  private pendingPose: { x: number; z: number; vx: number; vz: number } | null = null;
  private havePose = false;
  private lastX = 0;
  private lastZ = 0;
  private bootPriority = false;

  // counters
  private nFetch = 0;
  private nFetchInFlight = 0;
  private nBakeInFlight = 0;
  private nCache = 0;
  private nBuilt = 0;
  private nLoaded = 0;
  private nEvicted = 0;
  private nSkipped = 0;
  private nNacks = 0;
  private nTeleports = 0;
  private nScrollsDeferred = 0;
  private nScrolls = 0;
  private nPrefetched = 0;
  private capWarned = false;
  private fullWarned = false;

  constructor(deps: BrainDeps) {
    this.deps = deps;
  }

  // ---- lifecycle ------------------------------------------------------------------

  init(msg: BrainInitMsg): void {
    this.grid = msg.grid;
    this.layers = msg.layers;
    this.plan = msg.plan;
    this.tilesCfg = msg.tiles;
    for (const [layer, meta] of Object.entries(msg.layers) as [LayerName, BrainLayerMeta][]) {
      for (const lodStr of Object.keys(meta.chunkKeys)) {
        const lod = Number(lodStr);
        const keys = meta.chunkKeys[lod] as Float64Array;
        const hashes = meta.chunkHashes[lod] as BigUint64Array;
        const set = new Set<number>();
        const hmap = new Map<number, bigint>();
        for (let i = 0; i < keys.length; i++) {
          set.add(keys[i] as number);
          hmap.set(keys[i] as number, hashes[i] as bigint);
        }
        this.keySets.set(`${layer}:${lod}`, set);
        this.keyHashes.set(`${layer}:${lod}`, hmap);
      }
    }
    const t = msg.tiles;
    const span = t.latMax - t.latMin + 1;
    const M = t.tilesPerSide;
    // coarsest ring spans the coverage (always-resident backstop) — the same
    // level formula the pre-S5 streamer used, so the generated plan is identical
    const want = Math.max(1, Math.ceil(Math.log2((2 * span) / (M * t.gridN))) + 1);
    // A tile vertex packs its LOCAL texel coord (gridN·stride) in a 13-bit field
    // (mesh word0 bits 0-12; skirt code 13-15). The coarsest level's stride is
    // 2^(levels-1), so gridN·2^(levels-1) must stay ≤ 0x1fff. Large streamed
    // worlds (Estonia's pilot span) would exceed this; cap the base clipmap here
    // — the coarser far country is served by fartiles (S8), not this pyramid.
    // The generated 4 km world computes ≤5 levels ⇒ this cap never binds it.
    const maxStride = Math.max(1, Math.floor(0x1fff / t.gridN));
    const maxLevels = Math.max(1, Math.floor(Math.log2(maxStride)) + 1);
    const levels = Math.min(want, maxLevels);
    if (levels < want) {
      this.deps.emit({ kind: 'log', level: 'warn', msg: `clipmap levels capped ${want}→${levels} (13-bit vert packing; far country = fartiles/S8)` });
    }
    this.cfg = { res: t.latMax + 1, gridN: t.gridN, baseStride: 1, levels, tilesPerSide: M, latMin: t.latMin };
  }

  get maxTiles(): number {
    return clipmapMaxTiles(this.cfg);
  }

  // ---- chunk fetch (LRU + dedupe + authoritative absence) ---------------------------

  private layerGeo(layer: LayerName): RasterGeom {
    const meta = this.layers[layer];
    const t0 = meta?.texelMeters;
    if (!meta || !t0) throw new Error(`StreamBrain: layer '${layer}' missing texelMeters`);
    const g = this.grid;
    return {
      texel0: t0,
      chunkRes: Math.round(g.chunkMeters / t0),
      originX: g.originX,
      originZ: g.originZ,
      lodStep: g.lodStep,
    };
  }

  private chunkExists(layer: LayerName, key: ChunkKey): boolean {
    return this.keySets.get(`${layer}:${key.lod}`)?.has(packChunkKey(key.lod, key.cx, key.cz)) ?? false;
  }

  private chunkHash(layer: LayerName, key: ChunkKey): bigint {
    return this.keyHashes.get(`${layer}:${key.lod}`)?.get(packChunkKey(key.lod, key.cx, key.cz)) ?? 0n;
  }

  private payloadBytes(p: ChunkPayload): number {
    if (p.kind === 'height') return p.heights.byteLength;
    if (p.kind === 'planes') return p.planes.reduce((s, u) => s + u.byteLength, 0);
    let s = 0;
    for (const col of Object.values(p.cols)) if (col) s += col.byteLength;
    return s;
  }

  private async fetchChunk(layer: LayerName, key: ChunkKey): Promise<ChunkPayload | null> {
    if (!this.chunkExists(layer, key)) return null; // authoritative absence — no request
    const id = `${layer}:${key.lod}:${key.cx}:${key.cz}`;
    const hit = this.lru.get(id);
    if (hit) {
      this.lru.delete(id);
      this.lru.set(id, hit); // touch
      return hit.payload;
    }
    const inflight = this.inFlight.get(id);
    if (inflight) return inflight;
    this.nFetchInFlight++;
    const p = this.deps
      .fetch(layer, key)
      .then((payload) => {
        if (payload) {
          const bytes = this.payloadBytes(payload);
          this.lru.set(id, { payload, bytes });
          this.lruBytes += bytes;
          while (this.lruBytes > LRU_BYTES && this.lru.size > 1) {
            const oldest = this.lru.keys().next().value as string;
            this.lruBytes -= (this.lru.get(oldest) as { bytes: number }).bytes;
            this.lru.delete(oldest);
          }
        }
        return payload;
      })
      .finally(() => {
        this.inFlight.delete(id);
        this.nFetchInFlight--;
        this.nFetch++;
      });
    this.inFlight.set(id, p);
    return p;
  }

  /** drop consumed decode payloads (post-boot: the windows retain everything). */
  private dropLru(): void {
    this.lru.clear();
    this.lruBytes = 0;
  }

  // ---- boot plane fills ---------------------------------------------------------------

  async fillPlanesBoot(): Promise<void> {
    const packets: StreamPacket[] = [];
    const transfers: Transferable[] = [];
    // height levels — retained brain-side (tile bakes read them)
    this.hWin = [];
    for (let i = 0; i < this.plan.height.length; i++) {
      const plan = this.plan.height[i] as PlanePlan;
      const data = await this.assembleF32('height', plan, plan.n0x, plan.n0z);
      this.hWin.push({ plan, data, n0x: plan.n0x, n0z: plan.n0z, phaseX: 0, phaseY: 0 });
      const copy = data.slice(); // window is retained — ship a copy
      packets.push({ kind: 'fill', plane: 'height', level: i, x: 0, y: 0, w: plan.res, h: plan.res, f32: copy });
      transfers.push(copy.buffer);
    }
    // biome / fields — assembled transiently, transferred outright
    for (const [plane, plans, channels] of [
      ['biome', this.plan.biome, BIOME_CHANNELS],
      ['fields', this.plan.fields, FIELDS_CHANNELS],
    ] as const) {
      // far forests: the canopy layer (heightM, cover; LODs 1-4) merges into the
      // biome plane's channels 2/3. A source without a canopy layer (the generated
      // world) leaves the overlay undefined ⇒ the fill stays bit-identical.
      const overlay =
        plane === 'biome' && this.layers.canopy
          ? { layer: 'canopy' as LayerName, channels: CANOPY_CHANNELS }
          : undefined;
      for (let i = 0; i < plans.length; i++) {
        const plan = plans[i] as PlanePlan;
        const data = await this.assembleU8(plane, plan, plan.n0x, plan.n0z, channels, overlay);
        packets.push({ kind: 'fill', plane, level: i, x: 0, y: 0, w: plan.res, h: plan.res, u8: data });
        transfers.push(data.buffer);
      }
    }
    // water (+ derived far level) — water window retained for scroll re-reduce
    if (this.plan.water) {
      const plan = this.plan.water;
      const data = await this.assembleF32('water', plan, plan.n0x, plan.n0z, WATER_DRY_SENTINEL);
      this.wWin = { plan, data, n0x: plan.n0x, n0z: plan.n0z, phaseX: 0, phaseY: 0 };
      const copy = data.slice();
      packets.push({ kind: 'fill', plane: 'water', level: 0, x: 0, y: 0, w: plan.res, h: plan.res, f32: copy });
      transfers.push(copy.buffer);
      if (this.plan.waterFar) {
        const far = minReduce(data, plan.res, WATER_FAR_FACTOR);
        const farRes = this.plan.waterFar.res;
        packets.push({ kind: 'fill', plane: 'waterFar', level: 0, x: 0, y: 0, w: farRes, h: farRes, f32: far });
        transfers.push(far.buffer);
      }
    }
    this.deps.emit({ kind: 'packets', packets }, transfers);
    this.dropLru(); // consumed — the retained windows are the persistent store
    this.deps.emit({ kind: 'planesReady', ramBytes: this.ramBytes() });
  }

  private ramBytes(): number {
    let b = this.lruBytes;
    for (const w of this.hWin) b += w.data.byteLength;
    if (this.wWin) b += this.wWin.data.byteLength;
    return b;
  }

  /** assemble one f32 window at (n0x,n0z) from every overlapping chunk (the
   *  demand law) — same copy/clamp math as the pre-S5 static fill. */
  private async assembleF32(
    layer: LayerName,
    plan: PlanePlan,
    n0x: number,
    n0z: number,
    mapNaN?: number,
  ): Promise<Float32Array> {
    const geo = this.layerGeo(layer);
    const out = new Float32Array(plan.res * plan.res);
    const place = { ...plan, n0x, n0z };
    let box: FilledBox | null = null;
    for (const key of chunksInWindow(geo, plan.lod, n0x, n0z, plan.res)) {
      const payload = await this.fetchChunk(layer, key);
      if (!payload || payload.kind !== 'height') continue;
      box = copyChunkF32(out, plan.res, plan.res, place, geo, key.cx, key.cz, payload.heights, payload.res, box, mapNaN);
    }
    if (!box) {
      this.deps.emit({ kind: 'log', level: 'warn', msg: `terrain field: layer '${layer}' lod ${plan.lod} — no chunks landed, plane is zero` });
      return out;
    }
    clampExtend(out, plan.res, plan.res, 1, box);
    return out;
  }

  private async assembleU8(
    layer: LayerName,
    plan: PlanePlan,
    n0x: number,
    n0z: number,
    channels: readonly (readonly [string, number])[],
    overlay?: { layer: LayerName; channels: readonly (readonly [string, number])[] },
  ): Promise<Uint8Array> {
    const geo = this.layerGeo(layer);
    const names = this.layers[layer]?.planes ?? [];
    const out = new Uint8Array(plan.res * plan.res * 4);
    const place = { ...plan, n0x, n0z };
    let box: FilledBox | null = null;
    for (const key of chunksInWindow(geo, plan.lod, n0x, n0z, plan.res)) {
      const payload = await this.fetchChunk(layer, key);
      if (!payload || payload.kind !== 'planes') continue;
      box = copyChunkU8(out, plan.res, plan.res, place, geo, key.cx, key.cz, payload.planes, payload.res, names, channels, box);
    }
    // second-layer overlay (far-forest canopy → channels 2/3) on the levels that
    // layer cooks. It shares this plane's texelMeters, so the same lattice window
    // indexes it; untouched channels stay 0 (cover 0 = no forest). The rim clamp
    // rides the base layer's box — canopy sits inside the biome coverage.
    const oMeta = overlay ? this.layers[overlay.layer] : undefined;
    if (overlay && oMeta?.lods.includes(plan.lod)) {
      const oNames = oMeta.planes ?? [];
      for (const key of chunksInWindow(geo, plan.lod, n0x, n0z, plan.res)) {
        const payload = await this.fetchChunk(overlay.layer, key);
        if (!payload || payload.kind !== 'planes') continue;
        copyChunkU8(out, plan.res, plan.res, place, geo, key.cx, key.cz, payload.planes, payload.res, oNames, overlay.channels, box);
      }
    }
    if (box) clampExtend(out, plan.res, plan.res, 4, box);
    return out;
  }

  // ---- pose / residency drive ----------------------------------------------------------

  pose(x: number, z: number, vx: number, vz: number): void {
    if (this.havePose) {
      const jump = Math.hypot(x - this.lastX, z - this.lastZ);
      const finestExtent = this.cfg.tilesPerSide * this.cfg.gridN * this.tilesCfg.cell;
      if (jump > finestExtent) this.teleport();
    }
    this.havePose = true;
    this.lastX = x;
    this.lastZ = z;
    this.pendingPose = { x, z, vx, vz };
    if (!this.busy) void this.tick();
  }

  /** clear in-flight marks + re-seed coarsest-first (A13). */
  private teleport(): void {
    this.nTeleports++;
    this.bootPriority = true;
    this.skipped.clear();
    this.stalled.clear();
    this.prefetched.clear();
  }

  poolInfo(msg: PoolInfoMsg): void {
    this.pool = msg;
  }

  attachAck(key: string, ok: boolean): void {
    const t = this.pendingAttach.get(key);
    if (!t) return;
    this.pendingAttach.delete(key);
    if (ok) {
      this.resident.set(key, t);
      this.nLoaded++;
      return;
    }
    // NACK — pool dry. Pick the victim (farthest DEPARTED resident tile: least
    // likely to hole; footprint distance 0 = the covering backstop, reclaimed
    // last), evict it, and re-emit the attach (rebaked — DagCache-warm).
    this.nNacks++;
    const camX = this.latX(this.lastX);
    const camZ = this.latZ(this.lastZ);
    let worstKey: string | null = null;
    let worstD = -1;
    for (const [k, rt] of this.resident) {
      if (this.lastWant.has(k)) continue; // never evict a tile we still want
      const d = footprintDist2(rt, camX, camZ);
      if (d > worstD) {
        worstD = d;
        worstKey = k;
      }
    }
    if (worstKey === null) {
      if (!this.fullWarned) {
        this.fullWarned = true;
        this.deps.emit({ kind: 'log', level: 'warn', msg: 'stream: tile pool full of DESIRED tiles — raise the pool slot count' });
      }
      this.stalled.add(key);
      return;
    }
    this.deps.emit({ kind: 'packets', packets: [{ kind: 'evict', key: worstKey }] });
    this.resident.delete(worstKey);
    this.nEvicted++;
    void this.reattach(key);
  }

  private async reattach(key: string): Promise<void> {
    // the tile's arrays were transferred with the first attach — rebake (warm)
    const t = this.tileByKey(key);
    if (!t) return;
    const baked = await this.bakeTile(t);
    if (!baked || baked === 'overCap') return;
    this.emitAttach(t, baked);
  }

  private tileByKey(key: string): ClipmapTile | null {
    const camX = this.latX(this.lastX);
    const camZ = this.latZ(this.lastZ);
    for (const t of clipmapTiles(camX, camZ, this.cfg)) if (t.key === key) return t;
    return null;
  }

  private latX(worldX: number): number {
    const t = this.tilesCfg;
    return Math.min(Math.max(Math.round((worldX - t.origin) / t.cell), t.latMin), t.latMax);
  }
  private latZ(worldZ: number): number {
    const t = this.tilesCfg;
    return Math.min(Math.max(Math.round((worldZ - t.origin) / t.cell), t.latMin), t.latMax);
  }

  private async tick(): Promise<void> {
    this.busy = true;
    try {
      // drain to the LATEST pose (coalesce, single-flight — frame-paced by the
      // 10 Hz pose feed)
      while (this.pendingPose) {
        const p = this.pendingPose;
        this.pendingPose = null;
        await this.scrollPlanes(p.x, p.z);
        if (this.pool) await this.runDiff(p.x, p.z, p.vx, p.vz);
      }
    } catch (e) {
      this.deps.emit({ kind: 'log', level: 'error', msg: `stream brain tick: ${e instanceof Error ? e.message : String(e)}` });
    } finally {
      this.busy = false;
      this.emitCounters();
    }
  }

  // ---- plane window scroll (wrapping levels only — Estonia; generated levels
  // are pinned by plan.wraps=false and never enter) ------------------------------------

  private async scrollPlanes(camX: number, camZ: number): Promise<void> {
    for (let i = 0; i < this.hWin.length; i++) {
      await this.scrollLevel('height', i, this.hWin[i] as HeightWindow, camX, camZ);
    }
    if (this.wWin) await this.scrollLevel('water', 0, this.wWin, camX, camZ);
    // biome/fields planes scroll with the SAME rule but hold no brain window —
    // regions assemble straight from LRU'd chunks. (Their consumers are filtered
    // rgba8 taps; sub-texel placement is uncritical.) They ride height's snap
    // cadence at their own resolution once a wrapping level exists (S6).
  }

  /** re-center one wrapping window on the camera (SCROLL_DIV snaps, coverage
   *  clamp), emitting fills for the exposed L-shape THEN the origin commit —
   *  packet order carries promote-after-fill. */
  private async scrollLevel(plane: PlaneKind, level: number, win: HeightWindow, camX: number, camZ: number): Promise<void> {
    const plan = win.plan;
    if (!plan.wraps) return;
    const geo = this.layerGeo(plane === 'water' ? 'water' : 'height');
    const step = Math.max(1, Math.floor(plan.res / SCROLL_DIV));
    const wantX = Math.round((camX - (plan.res / 2) * plan.texel - geo.originX) / plan.texel / step) * step;
    const wantZ = Math.round((camZ - (plan.res / 2) * plan.texel - geo.originZ) / plan.texel / step) * step;
    // clamp the window into the layer's chunk coverage (a window past the rim
    // would only clamp-extend — the pinned-parity clamp, applied universally)
    const n0x = Math.min(Math.max(wantX, plan.nMinX), Math.max(plan.nMinX, plan.nMaxX - plan.res + 1));
    const n0z = Math.min(Math.max(wantZ, plan.nMinZ), Math.max(plan.nMinZ, plan.nMaxZ - plan.res + 1));
    const dx = n0x - win.n0x;
    const dz = n0z - win.n0z;
    if (dx === 0 && dz === 0) return;
    if (Math.abs(dx) >= plan.res || Math.abs(dz) >= plan.res) {
      // teleport-scale jump — nothing survives; refill the whole window
      await this.refillWindow(plane, level, win, n0x, n0z, geo);
      return;
    }
    // demote-before-scroll (§3 transaction): a resident tile baked against THIS
    // height level must not have its source texels overwritten under it — defer
    // the scroll (counted; the tile diff demotes it within a few poses).
    if (plane === 'height' && this.tileOverlapsVacated(level, win, n0x, n0z)) {
      this.nScrollsDeferred++;
      return;
    }
    const packets: StreamPacket[] = [];
    const transfers: Transferable[] = [];
    // toroidal invariant: a RETAINED sample keeps its physical address, so the
    // phase advances WITH the window (phase' = phase + Δn0 mod res) — boot fills
    // land at phase 0 (physical = logical) and every scroll preserves them.
    const phaseX = (((win.phaseX + dx) % plan.res) + plan.res) % plan.res;
    const phaseY = (((win.phaseY + dz) % plan.res) + plan.res) % plan.res;
    const emitRegion = async (rx0: number, rz0: number, rw: number, rh: number): Promise<void> => {
      if (rw <= 0 || rh <= 0) return;
      const sub = await this.assembleRegionF32(plane, plan, rx0, rz0, rw, rh);
      for (const r of wrapRects(rx0 - n0x, rz0 - n0z, rw, rh, phaseX, phaseY, plan.res)) {
        // region-local origin of this physical rect (rects are logically
        // contiguous, so the mapping is a constant offset)
        const lz0 = ((((r.y - phaseY) % plan.res) + plan.res) % plan.res) + n0z - rz0;
        const lx0 = ((((r.x - phaseX) % plan.res) + plan.res) % plan.res) + n0x - rx0;
        const part = new Float32Array(r.w * r.h);
        for (let y = 0; y < r.h; y++) {
          const src = (lz0 + y) * rw + lx0;
          for (let x = 0; x < r.w; x++) part[y * r.w + x] = sub[src + x] as number;
          win.data.set(part.subarray(y * r.w, (y + 1) * r.w), (r.y + y) * plan.res + r.x);
        }
        packets.push({ kind: 'fill', plane, level, x: r.x, y: r.y, w: r.w, h: r.h, f32: part });
        transfers.push(part.buffer);
      }
    };
    // exposed L-shape (new window minus old): a column strip + the remaining row strip
    if (dx > 0) await emitRegion(win.n0x + plan.res, n0z, dx, plan.res);
    else if (dx < 0) await emitRegion(n0x, n0z, -dx, plan.res);
    const rowX0 = dx >= 0 ? n0x : n0x - dx;
    const rowW = plan.res - Math.abs(dx);
    if (dz > 0) await emitRegion(rowX0, win.n0z + plan.res, rowW, dz);
    else if (dz < 0) await emitRegion(rowX0, n0z, rowW, -dz);
    win.n0x = n0x;
    win.n0z = n0z;
    win.phaseX = phaseX;
    win.phaseY = phaseY;
    packets.push(this.originPacket(plane, level, plan, geo, win));
    this.pushFarWater(plane, win, packets, transfers);
    this.nScrolls++;
    this.deps.emit({ kind: 'packets', packets }, transfers);
  }

  /** whole-window refill (teleport): assemble at the new placement, emit one
   *  full fill + the origin commit. */
  private async refillWindow(plane: PlaneKind, level: number, win: HeightWindow, n0x: number, n0z: number, geo: RasterGeom): Promise<void> {
    const plan = win.plan;
    if (plane === 'height' && this.tileOverlapsVacated(level, win, n0x, n0z)) {
      this.nScrollsDeferred++;
      return;
    }
    const layer: LayerName = plane === 'water' ? 'water' : 'height';
    const data = await this.assembleF32(layer, { ...plan, wraps: plan.wraps }, n0x, n0z, plane === 'water' ? WATER_DRY_SENTINEL : undefined);
    win.data.set(data);
    win.n0x = n0x;
    win.n0z = n0z;
    win.phaseX = 0;
    win.phaseY = 0;
    const copy = data.slice();
    const packets: StreamPacket[] = [
      { kind: 'fill', plane, level, x: 0, y: 0, w: plan.res, h: plan.res, f32: copy },
      this.originPacket(plane, level, plan, geo, win),
    ];
    const transfers: Transferable[] = [copy.buffer];
    this.pushFarWater(plane, win, packets, transfers);
    this.nScrolls++;
    this.deps.emit({ kind: 'packets', packets }, transfers);
  }

  private originPacket(plane: PlaneKind, level: number, plan: PlanePlan, geo: RasterGeom, win: HeightWindow): StreamPacket {
    const off = plan.stride >> 1;
    return {
      kind: 'planeOrigin',
      plane,
      level,
      originX: geo.originX + (win.n0x * plan.stride + off + 0.5) * geo.texel0,
      originZ: geo.originZ + (win.n0z * plan.stride + off + 0.5) * geo.texel0,
      n0x: win.n0x,
      n0z: win.n0z,
      phaseX: win.phaseX,
      phaseY: win.phaseY,
    };
  }

  /** far water rides its parent's scroll: rebuild the logical view (scrolls are
   *  rare, res ≤ 2048) and re-reduce the whole far level. */
  private pushFarWater(plane: PlaneKind, win: HeightWindow, packets: StreamPacket[], transfers: Transferable[]): void {
    if (plane !== 'water' || !this.plan.waterFar) return;
    const res = win.plan.res;
    const logical = new Float32Array(res * res);
    for (let z = 0; z < res; z++) {
      const pz = (z + win.phaseY) % res;
      for (let x = 0; x < res; x++) logical[z * res + x] = win.data[pz * res + ((x + win.phaseX) % res)] as number;
    }
    const far = minReduce(logical, res, WATER_FAR_FACTOR);
    const farRes = this.plan.waterFar.res;
    packets.push({ kind: 'fill', plane: 'waterFar', level: 0, x: 0, y: 0, w: farRes, h: farRes, f32: far });
    transfers.push(far.buffer);
  }

  /** does any resident/pending tile baked against height level `level` overlap
   *  the world region the scroll to (n0x,n0z) vacates? */
  private tileOverlapsVacated(level: number, win: HeightWindow, n0x: number, n0z: number): boolean {
    const plan = win.plan;
    // conservative vacated box: the old window minus the new (both L-strips)
    const rects: { x0: number; z0: number; x1: number; z1: number }[] = [];
    const dx = n0x - win.n0x;
    const dz = n0z - win.n0z;
    if (dx > 0) rects.push({ x0: win.n0x, z0: win.n0z, x1: win.n0x + Math.min(dx, plan.res), z1: win.n0z + plan.res });
    else if (dx < 0) rects.push({ x0: win.n0x + plan.res + Math.max(dx, -plan.res), z0: win.n0z, x1: win.n0x + plan.res, z1: win.n0z + plan.res });
    if (dz > 0) rects.push({ x0: win.n0x, z0: win.n0z, x1: win.n0x + plan.res, z1: win.n0z + Math.min(dz, plan.res) });
    else if (dz < 0) rects.push({ x0: win.n0x, z0: win.n0z + plan.res + Math.max(dz, -plan.res), x1: win.n0x + plan.res, z1: win.n0z + plan.res });
    if (rects.length === 0) return false;
    const t = this.tilesCfg;
    // level-j sample n covers finest-lattice ≈ [n·stride, (n+1)·stride)
    const s = plan.stride;
    for (const set of [this.resident, this.pendingAttach]) {
      for (const rt of set.values()) {
        if (rt.srcLevel !== level) continue;
        const tj0x = Math.floor(Math.max(rt.x0, t.latMin) / s);
        const tj1x = Math.ceil(Math.min(rt.x0 + rt.size, t.latMax) / s);
        const tj0z = Math.floor(Math.max(rt.z0, t.latMin) / s);
        const tj1z = Math.ceil(Math.min(rt.z0 + rt.size, t.latMax) / s);
        for (const r of rects) {
          if (tj0x < r.x1 && tj1x > r.x0 && tj0z < r.z1 && tj1z > r.z0) return true;
        }
      }
    }
    return false;
  }

  private async assembleRegionF32(plane: PlaneKind, plan: PlanePlan, rx0: number, rz0: number, rw: number, rh: number): Promise<Float32Array> {
    const layer: LayerName = plane === 'water' ? 'water' : 'height';
    const geo = this.layerGeo(layer);
    const out = new Float32Array(rw * rh);
    const place = { ...plan, n0x: rx0, n0z: rz0 };
    let box: FilledBox | null = null;
    for (const key of chunksInWindow(geo, plan.lod, rx0, rz0, rw, rh)) {
      const payload = await this.fetchChunk(layer, key);
      if (!payload || payload.kind !== 'height') continue;
      box = copyChunkF32(out, rw, rh, place, geo, key.cx, key.cz, payload.heights, payload.res, box, plane === 'water' ? WATER_DRY_SENTINEL : undefined);
    }
    if (box) clampExtend(out, rw, rh, 1, box);
    return out;
  }

  // ---- tile clipmap ---------------------------------------------------------------------

  /** boot: bake the whole spawn-centered ring set (coarsest-first is irrelevant
   *  here — ALL of it returns in one bundle for pool-cap sizing) and hand it to
   *  main pre-registry-build. */
  async bootTiles(camX: number, camZ: number): Promise<void> {
    const set = clipmapTiles(this.latX(camX), this.latZ(camZ), this.cfg);
    let pmV = 0;
    let pmT = 0;
    let pmC = 0;
    const tiles: TileGeometry[] = [];
    const transfers: Transferable[] = [];
    const baked = await Promise.all(set.map((t) => this.bakeTile(t)));
    set.forEach((t, i) => {
      const b = baked[i];
      if (!b || b === 'overCap') return;
      pmV = Math.max(pmV, b.gridVerts.length);
      pmT = Math.max(pmT, b.indices.length / 3);
      pmC = Math.max(pmC, b.clusterCount);
      tiles.push(b);
      transfers.push(b.gridVerts.buffer, b.indices.buffer, b.clusterData.buffer);
      this.pendingAttach.set(t.key, { x0: t.tx0, z0: t.tz0, size: t.tileTexels, level: t.level, srcLevel: this.bakeSrcLevel(t) });
    });
    this.deps.emit(
      {
        kind: 'bootTilesDone',
        tiles,
        poolMax: { v: pmV, t: pmT, c: pmC },
        maxTiles: this.maxTiles,
        levels: this.cfg.levels,
        nCache: this.nCache,
        nBuilt: this.nBuilt,
      },
      transfers,
    );
  }

  private async runDiff(camWX: number, camWZ: number, vx: number, vz: number): Promise<void> {
    const pool = this.pool as PoolInfoMsg;
    const tx = this.latX(camWX);
    const tz = this.latZ(camWZ);
    const desired = clipmapTiles(tx, tz, this.cfg);
    const want = new Set<string>();
    for (const t of desired) want.add(t.key);
    this.lastWant = want;
    for (const key of this.skipped) if (!want.has(key)) this.skipped.delete(key);
    for (const key of this.stalled) if (!want.has(key)) this.stalled.delete(key);

    const batch: ClipmapTile[] = [];
    const candidates = this.bootPriority ? [...desired].sort((a, b) => b.level - a.level) : desired;
    for (const t of candidates) {
      if (this.resident.has(t.key) || this.pendingAttach.has(t.key) || this.skipped.has(t.key) || this.stalled.has(t.key)) continue;
      batch.push(t);
      if (batch.length >= MAX_LOADS_PER_DIFF) break;
    }
    if (batch.length > 0) {
      const baked = await Promise.all(
        batch.map((t) =>
          this.bakeTile(t).then(
            (r) => ({ t, r, err: null as unknown }),
            (err: unknown) => ({ t, r: null, err }),
          ),
        ),
      );
      for (const { t, r, err } of baked) {
        if (err) {
          this.deps.emit({ kind: 'log', level: 'warn', msg: `stream tile ${t.key}: bake failed (${err instanceof Error ? err.message : String(err)})` });
          continue;
        }
        if (!r) continue;
        if (r === 'overCap') {
          if (!this.capWarned) {
            this.capWarned = true;
            this.deps.emit({ kind: 'log', level: 'warn', msg: `stream: tile ${t.key} over slot cap (v/t/c ${pool.vertCap}/${pool.triCap}/${pool.clusterCap}) — SKIPPED, coarser ring backstops` });
          }
          this.skipped.add(t.key);
          this.nSkipped++;
          continue;
        }
        this.emitAttach(t, r);
      }
    } else if (this.bootPriority) {
      // teleport/boot re-seed complete once nothing is left to arrive
      let all = true;
      for (const t of desired) {
        if (!this.resident.has(t.key) && !this.skipped.has(t.key) && !this.stalled.has(t.key)) {
          all = false;
          break;
        }
      }
      if (all) this.bootPriority = false;
    }

    // CLEANUP — the whole desired set resident (acked) ⇒ departed stragglers are
    // redundant; drop them. During motion they stay as the backstop (lazy evict).
    let allResident = true;
    for (const t of desired) {
      if (!this.resident.has(t.key) && !this.skipped.has(t.key) && !this.stalled.has(t.key)) {
        allResident = false;
        break;
      }
    }
    if (allResident) {
      const evicts: StreamPacket[] = [];
      for (const [key] of this.resident) {
        if (!want.has(key)) {
          evicts.push({ kind: 'evict', key });
          this.resident.delete(key);
          this.nEvicted++;
        }
      }
      if (evicts.length > 0) this.deps.emit({ kind: 'packets', packets: evicts });
    }

    // velocity prefetch at queue-idle: bake AHEAD along the heading into the
    // DagCache (no residency mutation — pure warm-up)
    const speed = Math.hypot(vx, vz);
    if (batch.length === 0 && allResident && speed > 2) {
      const px = this.latX(camWX + vx * PREFETCH_S);
      const pz = this.latZ(camWZ + vz * PREFETCH_S);
      let n = 0;
      for (const t of clipmapTiles(px, pz, this.cfg)) {
        if (n >= PREFETCH_MAX) break;
        if (this.resident.has(t.key) || this.pendingAttach.has(t.key) || this.prefetched.has(t.key)) continue;
        this.prefetched.add(t.key);
        if (this.prefetched.size > 512) this.prefetched.clear();
        this.nPrefetched++;
        n++;
        void this.bakeTile(t); // cache write inside; result discarded
      }
    }
  }

  private emitAttach(t: ClipmapTile, g: TileGeometry): void {
    this.pendingAttach.set(t.key, { x0: t.tx0, z0: t.tz0, size: t.tileTexels, level: t.level, srcLevel: this.bakeSrcLevel(t) });
    this.deps.emit(
      { kind: 'packets', packets: [{ kind: 'attach', tile: g }] },
      [g.gridVerts.buffer, g.indices.buffer, g.clusterData.buffer],
    );
  }

  // ---- tile bake (brain-side, from the retained height windows) ---------------------------

  /** the finest height window that fully contains the tile's (lattice-clamped)
   *  sample range — the bake's source (generated: L0 covers the whole lattice
   *  ⇒ always 0). Falls back to the coarsest window (clamp backstop). */
  private bakeSrcLevel(t: ClipmapTile): number {
    const cfg = this.tilesCfg;
    const nx0 = Math.min(Math.max(t.tx0, cfg.latMin), cfg.latMax);
    const nx1 = Math.min(Math.max(t.tx0 + t.tileTexels, cfg.latMin), cfg.latMax);
    const nz0 = Math.min(Math.max(t.tz0, cfg.latMin), cfg.latMax);
    const nz1 = Math.min(Math.max(t.tz0 + t.tileTexels, cfg.latMin), cfg.latMax);
    for (let i = 0; i < this.hWin.length; i++) {
      const w = this.hWin[i] as HeightWindow;
      const plan = w.plan;
      const s = plan.stride;
      const off = s >> 1;
      // nearest level-j sample indices the clamped range maps to
      const j0x = Math.round((nx0 - off) / s);
      const j1x = Math.round((nx1 - off) / s);
      const j0z = Math.round((nz0 - off) / s);
      const j1z = Math.round((nz1 - off) / s);
      if (j0x >= w.n0x && j1x <= w.n0x + plan.res - 1 && j0z >= w.n0z && j1z <= w.n0z + plan.res - 1) return i;
    }
    return this.hWin.length - 1;
  }

  /** height at finest-lattice (nx,nz) read from window level j (nearest level-j
   *  sample; exact on the level whose lattice the coord sits on). */
  private heightAtLattice(j: number, nx: number, nz: number): number {
    const w = this.hWin[j] as HeightWindow;
    const plan = w.plan;
    const s = plan.stride;
    const off = s >> 1;
    let jx = s === 1 ? nx : Math.round((nx - off) / s);
    let jz = s === 1 ? nz : Math.round((nz - off) / s);
    jx = Math.min(Math.max(jx, w.n0x), w.n0x + plan.res - 1);
    jz = Math.min(Math.max(jz, w.n0z), w.n0z + plan.res - 1);
    const px = (jx - w.n0x + w.phaseX) % plan.res;
    const pz = (jz - w.n0z + w.phaseY) % plan.res;
    return w.data[pz * plan.res + px] as number;
  }

  private async bakeTile(t: ClipmapTile): Promise<TileGeometry | 'overCap' | null> {
    const cfg = this.tilesCfg;
    const gridN = this.cfg.gridN;
    const vpa = gridN + 1;
    const j = this.bakeSrcLevel(t);
    // content-addressed cache key: seed salt + grid + placement + the fold of
    // the covering chunks' content hashes (source-agnostic — generated hashes
    // are zero and the seed carries identity; Estonia recuts invalidate).
    let fold = 0n;
    const geo = this.layerGeo('height');
    const plan = (this.hWin[j] as HeightWindow).plan;
    const s = plan.stride;
    const jx0 = Math.floor(t.tx0 / s);
    const jz0 = Math.floor(t.tz0 / s);
    const jres = Math.ceil(t.tileTexels / s) + 1;
    for (const key of chunksInWindow(geo, plan.lod, jx0, jz0, jres)) fold ^= this.chunkHash('height', key);
    const skirtLevel = cfg.skirt ? t.level : -1;
    const opts: HeightDagOpts = skirtLevel >= 0 ? { skirtLevel } : {};
    const suffix = `-sb-s${t.strideTexels}-${t.tx0}x${t.tz0}-h${fold.toString(16)}${skirtLevel >= 0 ? `-sk${skirtLevel}` : ''}`;
    const cacheKey = heightDagCacheKey(cfg.seed >>> 0, gridN, suffix);
    let built: HeightDagResult | null = await getCachedHeightDag(cacheKey);
    if (built) {
      this.nCache++;
    } else {
      this.nBakeInFlight++;
      try {
        const sub = new Float32Array(vpa * vpa);
        for (let gz = 0; gz <= gridN; gz++) {
          const nz = Math.min(Math.max(t.tz0 + gz * t.strideTexels, cfg.latMin), cfg.latMax);
          const srow = gz * vpa;
          for (let gx = 0; gx <= gridN; gx++) {
            const nx = Math.min(Math.max(t.tx0 + gx * t.strideTexels, cfg.latMin), cfg.latMax);
            sub[srow + gx] = this.heightAtLattice(j, nx, nz);
          }
        }
        const hfArgs = {
          heights: sub,
          gridN,
          cellSize: cfg.cell * t.strideTexels,
          originX: cfg.origin + t.tx0 * cfg.cell,
          originZ: cfg.origin + t.tz0 * cfg.cell,
        };
        if (this.deps.bake) {
          try {
            built = await this.deps.bake.buildHeight({ ...hfArgs, opts });
          } catch (e) {
            this.deps.emit({ kind: 'log', level: 'warn', msg: `stream tile ${t.key}: bake worker failed (${e instanceof Error ? e.message : String(e)}) → inline` });
            built = buildHeightGrid(hfArgs, opts);
          }
        } else {
          built = buildHeightGrid(hfArgs, opts);
        }
        void putCachedHeightDag(cacheKey, built); // fire-and-forget
        this.nBuilt++;
      } finally {
        this.nBakeInFlight--;
      }
    }
    // cap pre-check (runtime only — boot MEASURES the caps)
    if (this.pool) {
      const tCount = built.indices.length / 3;
      if (built.gridVerts.length > this.pool.vertCap || tCount > this.pool.triCap || built.clusters.length > this.pool.clusterCap) {
        return 'overCap';
      }
    }
    // remap tile-local grid coords → LOCAL texel coords (clamped to the lattice);
    // word0 = gx(0-12) | skirt code(13-15) | gz(16-31), per-tile origin words map
    // them to world. Identical world positions to the old global form (see header).
    const gridVerts = new Uint32Array(built.gridVerts.length);
    for (let i = 0; i < built.gridVerts.length; i++) {
      const p = built.gridVerts[i] as number;
      const code = (p >>> 13) & 0x7;
      const lx = Math.min(Math.max(t.tx0 + (p & 0x1fff) * t.strideTexels, cfg.latMin), cfg.latMax) - t.tx0;
      const lz = Math.min(Math.max(t.tz0 + ((p >>> 16) & 0xffff) * t.strideTexels, cfg.latMin), cfg.latMax) - t.tz0;
      if (lx < 0 || lx > 0x1fff || lz < 0 || lz > 0xffff) throw new Error(`stream tile ${t.key}: local vert coord out of range (${lx},${lz})`);
      gridVerts[i] = ((lx & 0x1fff) | (code << 13) | ((lz & 0xffff) << 16)) >>> 0;
    }
    return {
      key: t.key,
      gridVerts,
      indices: built.indices.slice(),
      clusterData: packClusters(built.clusters),
      clusterCount: built.clusters.length,
      originX: cfg.origin + t.tx0 * cfg.cell,
      originZ: cfg.origin + t.tz0 * cfg.cell,
      cellSize: cfg.cell,
      x0: t.tx0,
      z0: t.tz0,
      size: t.tileTexels,
    };
  }

  // ---- node-probe surface (tools/probe-streambrain.ts) — test-only ------------------------

  countersForProbe(): { scrolls: number; scrollsDeferred: number } {
    return { scrolls: this.nScrolls, scrollsDeferred: this.nScrollsDeferred };
  }
  probeAddResident(key: string, t: ResidentTile): void {
    this.resident.set(key, t);
  }
  probeRemoveResident(key: string): void {
    this.resident.delete(key);
  }

  // ---- counters (F-9) ----------------------------------------------------------------------

  emitCounters(): void {
    const perLevel: Record<string, number> = {};
    for (const rt of this.resident.values()) {
      const k = `stream.res.L${rt.level}`;
      perLevel[k] = (perLevel[k] ?? 0) + 1;
    }
    this.deps.emit({
      kind: 'counters',
      counters: {
        'stream.tiles.resident': this.resident.size,
        'stream.tiles.pending': this.pendingAttach.size,
        'stream.tiles.loaded': this.nLoaded,
        'stream.tiles.evicted': this.nEvicted,
        'stream.tiles.skipped': this.nSkipped,
        'stream.tiles.nacks': this.nNacks,
        'stream.bake.cache': this.nCache,
        'stream.bake.built': this.nBuilt,
        'stream.bake.inflight': this.nBakeInFlight,
        'stream.fetch.total': this.nFetch,
        'stream.fetch.inflight': this.nFetchInFlight,
        'stream.lru.mb': Math.round(this.lruBytes / 2 ** 20),
        'stream.ram.mb': Math.round(this.ramBytes() / 2 ** 20),
        'stream.scrolls': this.nScrolls,
        'stream.scrolls.deferred': this.nScrollsDeferred,
        'stream.teleports': this.nTeleports,
        'stream.prefetched': this.nPrefetched,
        ...perLevel,
      },
    });
  }
}
