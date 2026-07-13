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
 *    SCROLL_STEP texel snaps, freely: consumers live-sample the pyramid through
 *    finest-containing level chains, so a scroll only re-levels the trailing
 *    band (packet order still carries promote-after-fill: fills precede the
 *    origin commit).
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
import { packChunkKeyV2 } from '../../world/source/Lac2';
import type { ChunkKey, ChunkPayload, LayerName } from '../../world/source/WorldSource';
import type { ClipmapConfig, ClipmapTile } from './TerrainClipmap';
import { PartitionTree, type BakeReq, type MergePacket, type QuadDesc, type RefinePacket, type TreeConfig } from './PartitionTree';
import { FartileBand } from './FartileBand';
import {
  BIOME_CHANNELS,
  CANOPY_CHANNELS,
  FIELDS_CHANNELS,
  WATER_DRY_SENTINEL,
  WATER_FAR_FACTOR,
  WATERCOVER_CHANNELS,
  chunksInWindow,
  clampExtend,
  copyChunkF32,
  copyChunkU8,
  maxReduce,
  meanReduceU8,
  latticeWorld,
  minReduce,
  SOIL_CHANNELS,
  wrapRects,
  type FilledBox,
  type PlanePlan,
  type RasterGeom,
} from './PlaneFill';
import type {
  BootTile,
  BrainInitMsg,
  BrainLayerMeta,
  BrainToMain,
  FtArmMsg,
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
/** S6f: the root tiling of the coverage box is held ≤ this many tiles per side
 *  by adding coarser rungs — the always-resident country shell stays a small
 *  constant (≤ ~144 tiles) for ANY world size. */
const ROOT_SIDE_TARGET = 12;

interface HeightWindow {
  plan: PlanePlan;
  /** PHYSICAL (toroidal) layout, res² */
  data: Float32Array;
  n0x: number;
  n0z: number;
  phaseX: number;
  phaseY: number;
}

/** #114 watercover window — same toroidal scroll as HeightWindow but an rgba8
 *  (4-byte) backing (α in channel 0), so the fills ride the u8 mailbox path. */
interface U8Window {
  plan: PlanePlan;
  /** PHYSICAL (toroidal) layout, res²·4 (interleaved rgba8) */
  data: Uint8Array;
  n0x: number;
  n0z: number;
  phaseX: number;
  phaseY: number;
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
  private packKey: (lod: number, cx: number, cz: number) => number = packChunkKey;
  private manifestFormat: BrainInitMsg['manifestFormat'] = 1;
  private cfg!: ClipmapConfig;
  /** packed-key existence sets + content hashes per layer+lod */
  private readonly keySets = new Map<string, Set<number>>();
  private readonly keyHashes = new Map<string, Map<number, bigint>>();

  // plane windows (height retained for tile bakes; water retained for far reduce)
  private hWin: HeightWindow[] = [];
  private wWin: HeightWindow | null = null;
  /** #114 watercover window (retained so scrolls emit only the exposed L-shape,
   *  toroidally — the same graceful promote-after-fill as the water window). */
  private wcWin: U8Window | null = null;
  /** #116 soil window — the u8 twin of wcWin (no far level); retained for the same
   *  toroidal scroll. Null on the generated world (no soil layer). */
  private sWin: U8Window | null = null;

  // decoded-chunk LRU
  private readonly lru = new Map<string, { payload: ChunkPayload; bytes: number }>();
  private lruBytes = 0;
  private readonly inFlight = new Map<string, Promise<ChunkPayload | null>>();

  // tile residency — a PARTITION TREE (S6f). The tree owns the fringe + the pool
  // free-list; double-loading / zombie payloads / dual-LOD are unrepresentable by
  // its node types, not policed (see PartitionTree.ts).
  private tree!: PartitionTree;
  /** the pool free-list the tree reserves from (brain owns it — the single
   *  residency authority). Seeded [bootSlots .. slots) once main reports the pool. */
  private tileFree: number[] = [];
  private bootSlotCount = 0;
  private pool: PoolInfoMsg | null = null;
  private busy = false;
  /** S8: runtime far-tile residency (armed post-registry-build via ftArm). Null until
   *  armed (and on a source with no tree layer / fartiles disabled). */
  private ftBand: FartileBand | null = null;
  private pendingPose: { x: number; z: number; vx: number; vz: number } | null = null;
  private havePose = false;
  private lastX = 0;
  private lastZ = 0;
  /** ticks remaining to prioritise coarsest-first refines (boot/teleport re-seed). */
  private bootTicks = 0;

  // counters
  private nFetch = 0;
  private nFetchInFlight = 0;
  private nBakeInFlight = 0;
  private nCache = 0;
  private nBuilt = 0;
  private nScrolls = 0;
  private overCapWarned = false;

  constructor(deps: BrainDeps) {
    this.deps = deps;
  }

  // ---- lifecycle ------------------------------------------------------------------

  init(msg: BrainInitMsg): void {
    this.grid = msg.grid;
    this.layers = msg.layers;
    this.plan = msg.plan;
    this.tilesCfg = msg.tiles;
    this.packKey = msg.manifestFormat === 2 ? packChunkKeyV2 : packChunkKey;
    this.manifestFormat = msg.manifestFormat;
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
    // enough rungs that one M-ring of coarsest tiles spans the coverage — the
    // same level formula the pre-S5 streamer used (generated plan identical)
    const want = Math.max(1, Math.ceil(Math.log2((2 * span) / (M * t.gridN))) + 1);
    // S6f: extend UPWARD with coarser rungs until the ROOT tiling of the
    // coverage box is a small CONSTANT — the country-resident coarse shell stays
    // O(1) tiles for any world size. Super-data rungs bake by DECIMATION of the
    // same coarsest height window (stride 2^k — no new fetches, demand law
    // intact), and their verts pack in coarser units (packUnit in bakeTile), so
    // the 13-bit vert field no longer caps the pyramid. The generated world's
    // tiny box already satisfies the target at `want` rungs ⇒ identical levels.
    const rootsPerSide = (L: number): number => Math.ceil(span / (t.gridN << (L - 1)));
    let levels = want;
    while (rootsPerSide(levels) > ROOT_SIDE_TARGET) levels++;
    this.cfg = { res: t.latMax + 1, gridN: t.gridN, baseStride: 1, levels, tilesPerSide: M, latMin: t.latMin };
    // the residency tree — roots tile the coverage box at the coarsest level, and
    // refine toward the camera. The tree calls back into this core for slot
    // reservation, bakes, and the mailbox refine/merge packets.
    const treeCfg: TreeConfig = { gridN: t.gridN, levels, tilesPerSide: M, latMin: t.latMin, latMax: t.latMax };
    this.tree = new PartitionTree(treeCfg, {
      reserveSlots: (n) => this.reserveSlots(n),
      releaseSlots: (slots) => this.releaseSlots(slots),
      startBake: (req) => this.startBake(req),
      emitRefine: (p) => this.emitRefine(p),
      emitMerge: (p) => this.emitMerge(p),
    });
  }

  // ---- tree deps (§3/§5: slot free-list + async bakes + mailbox transactions) ------

  private reserveSlots(n: number): number[] {
    const out: number[] = [];
    for (let i = 0; i < n; i++) {
      const s = this.tileFree.pop();
      if (s === undefined) {
        // §5: a dry pool is a provisioning/ring-arithmetic bug, NEVER backpressure.
        throw new Error(`stream: tile pool dry reserving ${n} slots — raise the pool ceiling (ring arithmetic bug)`);
      }
      out.push(s);
    }
    return out;
  }

  private releaseSlots(slots: number[]): void {
    for (const s of slots) this.tileFree.push(s);
  }

  // ---- chunk fetch (LRU + dedupe + authoritative absence) ---------------------------

  private layerGeo(layer: LayerName): RasterGeom {
    const meta = this.layers[layer];
    const t0 = meta?.texelMeters;
    if (!meta || !t0) throw new Error(`StreamBrain: layer '${layer}' missing texelMeters`);
    const g = this.grid;
    if (this.manifestFormat === 2 && layer === 'height') {
      const baseTexel = meta.baseTexelMeters;
      const finestLod = meta.finestLod;
      if (!baseTexel || finestLod === undefined) throw new Error('StreamBrain: format-2 height lacks physical geometry');
      return {
        mode: 'physical-level',
        texel0: baseTexel * g.lodStep ** finestLod,
        baseTexel,
        finestLod,
        chunkRes: g.chunkRes,
        originX: g.originX,
        originZ: g.originZ,
        lodStep: g.lodStep,
      };
    }
    return {
      mode: 'legacy-offset',
      texel0: t0,
      baseTexel: t0,
      finestLod: 0,
      chunkRes: Math.round(g.chunkMeters / t0),
      originX: g.originX,
      originZ: g.originZ,
      lodStep: g.lodStep,
    };
  }

  private chunkExists(layer: LayerName, key: ChunkKey): boolean {
    return this.keySets.get(`${layer}:${key.lod}`)?.has(this.packKey(key.lod, key.cx, key.cz)) ?? false;
  }

  private chunkHash(layer: LayerName, key: ChunkKey): bigint {
    return this.keyHashes.get(`${layer}:${key.lod}`)?.get(this.packKey(key.lod, key.cx, key.cz)) ?? 0n;
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
        // #115: with a coverage plane (Estonia) the far surface MAX-reduces to the WET
        // level (the α-gate hides off-shore water); without one (generated) it keeps the
        // MIN-reduce bed dive verbatim ⇒ bit-identical generated far water.
        const far = this.plan.waterCover
          ? maxReduce(data, plan.res, WATER_FAR_FACTOR)
          : minReduce(data, plan.res, WATER_FAR_FACTOR);
        const farRes = this.plan.waterFar.res;
        packets.push({ kind: 'fill', plane: 'waterFar', level: 0, x: 0, y: 0, w: farRes, h: farRes, f32: far });
        transfers.push(far.buffer);
      }
    }
    // #114 watercover α — window retained for the toroidal scroll (bootTiles' scroll
    // re-centers it on the spawn, exactly like the water window). Absent layer ⇒ skipped.
    if (this.plan.waterCover) {
      const plan = this.plan.waterCover;
      const data = await this.assembleU8('watercover', plan, plan.n0x, plan.n0z, WATERCOVER_CHANNELS);
      this.wcWin = { plan, data, n0x: plan.n0x, n0z: plan.n0z, phaseX: 0, phaseY: 0 };
      const copy = data.slice();
      packets.push({ kind: 'fill', plane: 'watercover', level: 0, x: 0, y: 0, w: plan.res, h: plan.res, u8: copy });
      transfers.push(copy.buffer);
      // #115 far coverage α — the ×8 mean-reduce of the watercover window (the far
      // mirror of waterFar's min-reduce). Data is logical (phase 0) at boot.
      if (this.plan.waterCoverFar) {
        const far = meanReduceU8(data, plan.res, WATER_FAR_FACTOR);
        const farRes = this.plan.waterCoverFar.res;
        packets.push({ kind: 'fill', plane: 'waterCoverFar', level: 0, x: 0, y: 0, w: farRes, h: farRes, u8: far });
        transfers.push(far.buffer);
      }
    }
    // #116 soil pedology — a u8 window (4 packed channels), retained for the toroidal
    // scroll exactly like watercover but WITHOUT a far level (soil is pilot-near only).
    // Absent layer (generated) ⇒ skipped ⇒ no soil plane ever fills.
    if (this.plan.soil) {
      const plan = this.plan.soil;
      const data = await this.assembleU8('soil', plan, plan.n0x, plan.n0z, SOIL_CHANNELS);
      this.sWin = { plan, data, n0x: plan.n0x, n0z: plan.n0z, phaseX: 0, phaseY: 0 };
      const copy = data.slice();
      packets.push({ kind: 'fill', plane: 'soil', level: 0, x: 0, y: 0, w: plan.res, h: plan.res, u8: copy });
      transfers.push(copy.buffer);
    }
    this.deps.emit({ kind: 'packets', packets }, transfers);
    this.dropLru(); // consumed — the retained windows are the persistent store
    this.deps.emit({ kind: 'planesReady', ramBytes: this.ramBytes() });
  }

  private ramBytes(): number {
    let b = this.lruBytes;
    for (const w of this.hWin) b += w.data.byteLength;
    if (this.wWin) b += this.wWin.data.byteLength;
    if (this.wcWin) b += this.wcWin.data.byteLength;
    if (this.sWin) b += this.sWin.data.byteLength;
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

  /** cancel every in-flight refine (bakes abandoned, slots returned) so the
   *  fringe can cascade back toward coarse (§4 — always ready) and re-refine
   *  toward the new pose coarsest-first. */
  private teleport(): void {
    this.tree.teleport();
    this.bootTicks = 2 * this.cfg.levels;
  }

  /** S8: arm the runtime far-tile band with the library crown pools + pool geometry
   *  (main sends ftArm post-registry-build). The band owns per-cell residency + the
   *  splat/emit/pyramid (this worker thread) + slot/granule free-lists; it emits
   *  ftAttach/ftEvict onto the SAME FIFO mailbox the tile transactions ride. */
  armFartiles(msg: FtArmMsg): void {
    const g = this.grid;
    this.ftBand = new FartileBand({
      fetch: (layer, key) => this.fetchChunk(layer, key),
      emit: (packet, transfer) => this.deps.emit({ kind: 'packets', packets: [packet] }, transfer),
      treesExist: (cx, cz) => this.chunkExists('trees', { lod: 0, cx, cz }),
      grid: { originX: g.originX, originZ: g.originZ, chunkMeters: g.chunkMeters },
      // macrotask yield so a long cell bake never starves incoming pose/teleport
      // messages (the far forest streams in progressively — the demand law).
      yield: () => new Promise((r) => setTimeout(r, 0)),
    });
    this.ftBand.arm(msg);
  }

  poolInfo(msg: PoolInfoMsg): void {
    this.pool = msg;
    // the brain owns the free-list: boot claimed [0..bootSlotCount); the rest is
    // the refinement pool. Provisioned so reserveSlots never runs dry (§5).
    this.tileFree = [];
    for (let s = msg.slots - 1; s >= this.bootSlotCount; s--) this.tileFree.push(s);
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
        if (this.pool) {
          // iterate the FRINGE ONLY (§2): the tree decides split/hold/merge per
          // leaf, fires instant merges, and starts refine txs under the bake
          // budget. Committed txs emit refine/merge mailbox packets asynchronously
          // as their bakes land.
          this.tree.tick(this.latX(p.x), this.latZ(p.z), MAX_LOADS_PER_DIFF, this.bootTicks > 0);
          if (this.bootTicks > 0) this.bootTicks--;
        }
        // S8: drive the far-tile ring residency off the SAME pose (world meters);
        // its bakes are fire-and-forget, emitting ftAttach/ftEvict as they land.
        this.ftBand?.pose(p.x, p.z);
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
    if (this.wcWin) await this.scrollU8('watercover', WATERCOVER_CHANNELS, this.wcWin, camX, camZ, (p, t) => this.pushFarCover(p, t));
    if (this.sWin) await this.scrollU8('soil', SOIL_CHANNELS, this.sWin, camX, camZ);
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
    // S6f: scrolls never wait on tiles. Tile verts LIVE-SAMPLE the plane pyramid
    // through the finest-CONTAINING level chain (the same chain grass reads), so a
    // scroll only re-levels the trailing band — every sampler adapts at once, no
    // demote transaction needed. (The old bag's demote-before-scroll deferral would
    // DEADLOCK against the tree: the partition always keeps mid-level tiles inside
    // the window, so a deferred scroll never unblocked — the window froze at spawn
    // and everything streamed after it baked from coarse fallbacks.)
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

  private originPacket(
    plane: PlaneKind,
    level: number,
    plan: PlanePlan,
    geo: RasterGeom,
    win: { n0x: number; n0z: number; phaseX: number; phaseY: number },
  ): StreamPacket {
    return {
      kind: 'planeOrigin',
      plane,
      level,
      originX: latticeWorld(geo, plan.lod, win.n0x, 'x'),
      originZ: latticeWorld(geo, plan.lod, win.n0z, 'z'),
      n0x: win.n0x,
      n0z: win.n0z,
      phaseX: win.phaseX,
      phaseY: win.phaseY,
    };
  }

  /** far water rides its parent's scroll: rebuild the logical view (scrolls are
   *  rare, res ≤ 2048) and re-reduce the whole far level. MAX-reduce with a coverage
   *  plane (Estonia — wet level under the α-gate), MIN-reduce without one (generated). */
  private pushFarWater(plane: PlaneKind, win: HeightWindow, packets: StreamPacket[], transfers: Transferable[]): void {
    if (plane !== 'water' || !this.plan.waterFar) return;
    const res = win.plan.res;
    const logical = new Float32Array(res * res);
    for (let z = 0; z < res; z++) {
      const pz = (z + win.phaseY) % res;
      for (let x = 0; x < res; x++) logical[z * res + x] = win.data[pz * res + ((x + win.phaseX) % res)] as number;
    }
    const far = this.plan.waterCover ? maxReduce(logical, res, WATER_FAR_FACTOR) : minReduce(logical, res, WATER_FAR_FACTOR);
    const farRes = this.plan.waterFar.res;
    packets.push({ kind: 'fill', plane: 'waterFar', level: 0, x: 0, y: 0, w: farRes, h: farRes, f32: far });
    transfers.push(far.buffer);
  }

  /** #115 far COVERAGE rides the watercover scroll (the u8 mirror of pushFarWater):
   *  rebuild the logical view + mean-reduce the whole far coverage level. */
  private pushFarCover(packets: StreamPacket[], transfers: Transferable[]): void {
    if (!this.plan.waterCoverFar || !this.wcWin) return;
    const win = this.wcWin;
    const res = win.plan.res;
    const logical = new Uint8Array(res * res * 4);
    for (let z = 0; z < res; z++) {
      const pz = (z + win.phaseY) % res;
      for (let x = 0; x < res; x++) logical[(z * res + x) * 4] = win.data[(pz * res + ((x + win.phaseX) % res)) * 4] as number;
    }
    const far = meanReduceU8(logical, res, WATER_FAR_FACTOR);
    const farRes = this.plan.waterCoverFar.res;
    packets.push({ kind: 'fill', plane: 'waterCoverFar', level: 0, x: 0, y: 0, w: farRes, h: farRes, u8: far });
    transfers.push(far.buffer);
  }

  /** a ClipmapTile view of a footprint — the bake/srcLevel helpers key off
   *  tx0/tz0/tileTexels/strideTexels/level only. */
  private synthTile(level: number, tx0: number, tz0: number, size: number): ClipmapTile {
    const strideTexels = 1 << level;
    return { level, strideTexels, tx0, tz0, tileTexels: size, key: `L${level}:${Math.round(tx0 / size)},${Math.round(tz0 / size)}` };
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

  // ---- u8 camera-window scroll (#114 watercover / #116 soil — the u8 twin of
  // scrollLevel: toroidal L-shape). watercover passes an afterScroll hook that rides its
  // far-coverage reduce onto the same packet batch; soil has no far level ⇒ no hook. ----

  private async scrollU8(
    kind: 'watercover' | 'soil',
    channels: readonly (readonly [string, number])[],
    win: U8Window,
    camX: number,
    camZ: number,
    afterScroll?: (packets: StreamPacket[], transfers: Transferable[]) => void,
  ): Promise<void> {
    const plan = win.plan;
    if (!plan.wraps) return;
    const geo = this.layerGeo(kind);
    const step = Math.max(1, Math.floor(plan.res / SCROLL_DIV));
    const wantX = Math.round((camX - (plan.res / 2) * plan.texel - geo.originX) / plan.texel / step) * step;
    const wantZ = Math.round((camZ - (plan.res / 2) * plan.texel - geo.originZ) / plan.texel / step) * step;
    const n0x = Math.min(Math.max(wantX, plan.nMinX), Math.max(plan.nMinX, plan.nMaxX - plan.res + 1));
    const n0z = Math.min(Math.max(wantZ, plan.nMinZ), Math.max(plan.nMinZ, plan.nMaxZ - plan.res + 1));
    const dx = n0x - win.n0x;
    const dz = n0z - win.n0z;
    if (dx === 0 && dz === 0) return;
    if (Math.abs(dx) >= plan.res || Math.abs(dz) >= plan.res) {
      await this.refillU8(kind, channels, win, n0x, n0z, geo, afterScroll);
      return;
    }
    const packets: StreamPacket[] = [];
    const transfers: Transferable[] = [];
    const phaseX = (((win.phaseX + dx) % plan.res) + plan.res) % plan.res;
    const phaseY = (((win.phaseY + dz) % plan.res) + plan.res) % plan.res;
    const emitRegion = async (rx0: number, rz0: number, rw: number, rh: number): Promise<void> => {
      if (rw <= 0 || rh <= 0) return;
      const sub = await this.assembleRegionU8(kind, channels, plan, rx0, rz0, rw, rh);
      for (const r of wrapRects(rx0 - n0x, rz0 - n0z, rw, rh, phaseX, phaseY, plan.res)) {
        const lz0 = ((((r.y - phaseY) % plan.res) + plan.res) % plan.res) + n0z - rz0;
        const lx0 = ((((r.x - phaseX) % plan.res) + plan.res) % plan.res) + n0x - rx0;
        const part = new Uint8Array(r.w * r.h * 4);
        for (let y = 0; y < r.h; y++) {
          const src = ((lz0 + y) * rw + lx0) * 4;
          part.set(sub.subarray(src, src + r.w * 4), y * r.w * 4);
          win.data.set(part.subarray(y * r.w * 4, (y + 1) * r.w * 4), ((r.y + y) * plan.res + r.x) * 4);
        }
        packets.push({ kind: 'fill', plane: kind, level: 0, x: r.x, y: r.y, w: r.w, h: r.h, u8: part });
        transfers.push(part.buffer);
      }
    };
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
    packets.push(this.originPacket(kind, 0, plan, geo, win));
    afterScroll?.(packets, transfers);
    this.nScrolls++;
    this.deps.emit({ kind: 'packets', packets }, transfers);
  }

  /** whole-window refill (teleport-scale jump): reassemble at the new placement. */
  private async refillU8(
    kind: 'watercover' | 'soil',
    channels: readonly (readonly [string, number])[],
    win: U8Window,
    n0x: number,
    n0z: number,
    geo: RasterGeom,
    afterScroll?: (packets: StreamPacket[], transfers: Transferable[]) => void,
  ): Promise<void> {
    const plan = win.plan;
    const data = await this.assembleU8(kind, plan, n0x, n0z, channels);
    win.data.set(data);
    win.n0x = n0x;
    win.n0z = n0z;
    win.phaseX = 0;
    win.phaseY = 0;
    const packets: StreamPacket[] = [
      { kind: 'fill', plane: kind, level: 0, x: 0, y: 0, w: plan.res, h: plan.res, u8: data },
      this.originPacket(kind, 0, plan, geo, win),
    ];
    const transfers: Transferable[] = [data.buffer];
    afterScroll?.(packets, transfers);
    this.nScrolls++;
    this.deps.emit({ kind: 'packets', packets }, transfers);
  }

  /** assemble one rgba8 sub-region from every overlapping chunk of a u8 layer
   *  (watercover α / soil channels) — the u8 twin of assembleRegionF32. */
  private async assembleRegionU8(
    layer: LayerName,
    channels: readonly (readonly [string, number])[],
    plan: PlanePlan,
    rx0: number,
    rz0: number,
    rw: number,
    rh: number,
  ): Promise<Uint8Array> {
    const geo = this.layerGeo(layer);
    const names = this.layers[layer]?.planes ?? [];
    const out = new Uint8Array(rw * rh * 4);
    const place = { ...plan, n0x: rx0, n0z: rz0 };
    let box: FilledBox | null = null;
    for (const key of chunksInWindow(geo, plan.lod, rx0, rz0, rw, rh)) {
      const payload = await this.fetchChunk(layer, key);
      if (!payload || payload.kind !== 'planes') continue;
      box = copyChunkU8(out, rw, rh, place, geo, key.cx, key.cz, payload.planes, payload.res, names, channels, box);
    }
    if (box) clampExtend(out, rw, rh, 4, box);
    return out;
  }

  // ---- tile residency: partition tree (§2-§6) -------------------------------------------

  /** boot: bake the whole resident SUBTREE at the spawn pose (fringe leaves +
   *  their parked ancestors — the ancestor-closed set, §4), assign each a slot,
   *  seed the tree, and hand main the bundle (pre-registry — it sizes pool caps +
   *  the provisioned slot ceiling). */
  async bootTiles(camX: number, camZ: number): Promise<void> {
    // center the wrapping windows ON the boot pose before any bake — the plan's
    // default placement is not the spawn, and a fine tile baked from a coarser
    // window is the flat-terrain-at-spawn bug (the audit below throws loud).
    await this.scrollPlanes(camX, camZ);
    const descs = this.tree.bootDescs(this.latX(camX), this.latZ(camZ));
    // boot readiness audit: the plan places the wrapping windows around the same
    // coverage center the boot cut refines toward, so every boot bake should have
    // its source resident — if not, say so LOUDLY (a silent coarse-content "fine"
    // tile is the roamed-pose bug at frame 1).
    let notReady = 0;
    const nrByLevel: Record<number, number> = {};
    const cxT = this.latX(camX);
    const czT = this.latZ(camZ);
    let nrMinDist = Infinity;
    for (const d of descs) {
      if (d.isLeaf && !this.bakeSourceReady(this.synthTile(d.desc.level, d.desc.tx0, d.desc.tz0, d.desc.size))) {
        notReady++;
        nrByLevel[d.desc.level] = (nrByLevel[d.desc.level] ?? 0) + 1;
        const dx = cxT < d.desc.tx0 ? d.desc.tx0 - cxT : cxT > d.desc.tx0 + d.desc.size ? cxT - (d.desc.tx0 + d.desc.size) : 0;
        const dz = czT < d.desc.tz0 ? d.desc.tz0 - czT : czT > d.desc.tz0 + d.desc.size ? czT - (d.desc.tz0 + d.desc.size) : 0;
        nrMinDist = Math.min(nrMinDist, Math.max(dx, dz));
      }
    }
    if (notReady > 0) {
      this.deps.emit({ kind: 'log', level: 'warn', msg: `stream boot: ${notReady} boot leaves bake from coarser-than-own windows (byLevel ${JSON.stringify(nrByLevel)}, nearest ${nrMinDist}m)` });
    }
    const baked = await Promise.all(descs.map((d) => this.bakeTile(this.synthTile(d.desc.level, d.desc.tx0, d.desc.tz0, d.desc.size))));
    let pmV = 0;
    let pmT = 0;
    let pmC = 0;
    const tiles: BootTile[] = [];
    const transfers: Transferable[] = [];
    const slotOf = new Map<number, number>();
    descs.forEach((d, i) => {
      const b = baked[i];
      if (!b || b === 'overCap') throw new Error(`stream boot: tile ${d.desc.key} failed to bake`);
      b.slot = i; // boot slots are assigned deterministically 0..bootSlotCount-1
      pmV = Math.max(pmV, b.gridVerts.length);
      pmT = Math.max(pmT, b.indices.length / 3);
      pmC = Math.max(pmC, b.clusterCount);
      tiles.push({ tile: b, isLeaf: d.isLeaf });
      transfers.push(b.gridVerts.buffer, b.indices.buffer, b.clusterData.buffer);
      slotOf.set(d.nodeId, i);
    });
    this.tree.seedBoot(slotOf);
    this.bootSlotCount = descs.length;
    // §5 provisioning — PURE RING ARITHMETIC, world-size-independent: fringe ≤
    // roots + per finer rung one hysteresis ring annulus ((M+2)² − (M/2)² leaves,
    // clamped by that rung's world tiling); parked ancestors = the quadtree's
    // internal nodes ≤ fringe/3 (the ×4/3); + in-flight tx reservations. A dry
    // pool is therefore a bug (throw-loud in reserveSlots), never backpressure.
    const M = this.cfg.tilesPerSide;
    const span = this.tilesCfg.latMax - this.tilesCfg.latMin + 1;
    let roots = 0;
    for (const d of descs) if (d.desc.level === this.cfg.levels - 1) roots++;
    let fringeBound = roots;
    for (let k = 0; k < this.cfg.levels - 1; k++) {
      const worldSide = Math.ceil(span / (this.cfg.gridN << k)) + 1;
      fringeBound += Math.min((M + 2) * (M + 2) - (M / 2) * (M / 2), worldSide * worldSide);
    }
    const slots = Math.ceil((fringeBound * 4) / 3) + 4 * MAX_LOADS_PER_DIFF;
    if (descs.length > slots) throw new Error(`stream boot: subtree ${descs.length} > provisioned ceiling ${slots} (ring arithmetic bug)`);
    this.deps.emit({
      kind: 'log',
      level: 'log',
      msg: `stream pool ceiling: ${slots} slots (${roots} roots + ${fringeBound - roots} ring leaves ×4/3 + inflight; ${this.cfg.levels} rungs)`,
    });
    this.deps.emit(
      {
        kind: 'bootTilesDone',
        tiles,
        poolMax: { v: pmV, t: pmT, c: pmC },
        slots,
        levels: this.cfg.levels,
        nCache: this.nCache,
        nBuilt: this.nBuilt,
      },
      transfers,
    );
  }

  /** tree dep: bake the ≤4 children of a refine tx (nested pool / inline), and
   *  deliver each to the tree keyed to (nodeId, txId) — a stale delivery (the tx
   *  was cancelled) has no home and is dropped by onBake at one O(1) check (§3). */
  private startBake(req: BakeReq): void {
    // ready gate (see bakeSourceReady): every child's source window must have
    // arrived, else abort — the pose tick re-nominates the leaf after the scroll.
    for (const q of req.quads) {
      if (!this.bakeSourceReady(this.synthTile(q.level, q.tx0, q.tz0, q.size))) {
        this.tree.abortRefine(req.nodeId, req.txId);
        return;
      }
    }
    req.quads.forEach((q: QuadDesc, i: number) => {
      void this.bakeTile(this.synthTile(q.level, q.tx0, q.tz0, q.size)).then(
        (r) => {
          if (!r || r === 'overCap') {
            if (r === 'overCap' && !this.overCapWarned) {
              this.overCapWarned = true;
              this.deps.emit({ kind: 'log', level: 'warn', msg: `stream: tile ${q.key} over slot cap — raise reserveTilePool caps (refine aborted)` });
            }
            this.tree.abortRefine(req.nodeId, req.txId);
            return;
          }
          r.slot = q.slot;
          this.tree.onBake(req.nodeId, req.txId, i, r);
        },
        (err: unknown) => {
          this.deps.emit({ kind: 'log', level: 'warn', msg: `stream tile ${q.key}: bake failed (${err instanceof Error ? err.message : String(err)})` });
          this.tree.abortRefine(req.nodeId, req.txId);
        },
      );
    });
  }

  /** tree dep: a committed refine — park the parent slot, attach the ≤4 baked
   *  children, update the level grid. ONE atomic drain step on main (§6). */
  private emitRefine(p: RefinePacket): void {
    const children = p.children.map((c) => c.payload as TileGeometry);
    const transfers: Transferable[] = [];
    for (const c of children) transfers.push(c.gridVerts.buffer, c.indices.buffer, c.clusterData.buffer);
    this.deps.emit({ kind: 'packets', packets: [{ kind: 'tileRefine', parkSlot: p.parkSlot, children, levelGrid: p.levelGrid }] }, transfers);
  }

  /** tree dep: a committed merge — unpark the retained parent slot, evict the ≤4
   *  child slots, update the level grid (no bake — coarsen is always ready, §4). */
  private emitMerge(p: MergePacket): void {
    this.deps.emit({ kind: 'packets', packets: [{ kind: 'tileMerge', unparkSlot: p.unparkSlot, freeSlots: p.freeSlots, levelGrid: p.levelGrid }] });
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
    if (this.manifestFormat === 2) {
      // Coarsest resident source no coarser than the tile. Coordinates below are
      // in normalized finest samples; each plane window is in its own lattice.
      for (let i = this.hWin.length - 1; i >= 0; i--) {
        const w = this.hWin[i] as HeightWindow;
        const plan = w.plan;
        const s = plan.stride;
        if (s > t.strideTexels) continue;
        const x0 = Math.floor((nx0 + 0.5) / s - 0.5);
        const x1 = Math.ceil((nx1 + 0.5) / s - 0.5);
        const z0 = Math.floor((nz0 + 0.5) / s - 0.5);
        const z1 = Math.ceil((nz1 + 0.5) / s - 0.5);
        if (
          x0 >= w.n0x && x1 <= w.n0x + plan.res - 1
          && z0 >= w.n0z && z1 <= w.n0z + plan.res - 1
          && x0 >= plan.nMinX && x1 <= plan.nMaxX
          && z0 >= plan.nMinZ && z1 <= plan.nMaxZ
        ) return i;
      }
      return this.hWin.length - 1;
    }
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
    if (this.manifestFormat === 2) {
      const gx = (nx + 0.5) / s - 0.5;
      const gz = (nz + 0.5) / s - 0.5;
      const x0 = Math.floor(gx);
      const z0 = Math.floor(gz);
      const fx = gx - x0;
      const fz = gz - z0;
      const sample = (x: number, z: number): number => {
        const sx = Math.min(Math.max(x, w.n0x), w.n0x + plan.res - 1);
        const sz = Math.min(Math.max(z, w.n0z), w.n0z + plan.res - 1);
        const px = (sx - w.n0x + w.phaseX) % plan.res;
        const pz = (sz - w.n0z + w.phaseY) % plan.res;
        return w.data[pz * plan.res + px] as number;
      };
      const a = sample(x0, z0) * (1 - fx) + sample(x0 + 1, z0) * fx;
      const b = sample(x0, z0 + 1) * (1 - fx) + sample(x0 + 1, z0 + 1) * fx;
      return a * (1 - fz) + b * fz;
    }
    const off = s >> 1;
    let jx = s === 1 ? nx : Math.round((nx - off) / s);
    let jz = s === 1 ? nz : Math.round((nz - off) / s);
    jx = Math.min(Math.max(jx, w.n0x), w.n0x + plan.res - 1);
    jz = Math.min(Math.max(jz, w.n0z), w.n0z + plan.res - 1);
    const px = (jx - w.n0x + w.phaseX) % plan.res;
    const pz = (jz - w.n0z + w.phaseY) % plan.res;
    return w.data[pz * plan.res + px] as number;
  }

  /** S6f READY GATE: a level-k tile's bake needs a source window at stride ≤ its
   *  own (else the "fine" tile would be a decimation of coarse data, committed
   *  and cached as if it were the real content — the roamed-pose flat-tile bug).
   *  Super-data rungs (stride ≥ the coarsest window's) are always ready. A not-
   *  ready refine aborts; the pose tick re-nominates the leaf once the window's
   *  scroll lands (the demand law's promote-after-fill, applied to tiles). */
  private bakeSourceReady(t: ClipmapTile): boolean {
    const j = this.bakeSrcLevel(t);
    const stride = (this.hWin[j] as HeightWindow).plan.stride;
    const coarsest = (this.hWin[this.hWin.length - 1] as HeightWindow).plan.stride;
    return stride <= t.strideTexels || t.strideTexels >= coarsest;
  }

  private async bakeTile(t: ClipmapTile): Promise<TileGeometry | 'overCap' | null> {
    const cfg = this.tilesCfg;
    const gridN = this.cfg.gridN;
    const vpa = gridN + 1;
    // pick the source window AND extract the sub-grid SYNCHRONOUSLY (no await
    // between them): a window scroll landing mid-bake must never let the
    // extraction clamp into a stale placement — that baked flat garbage under a
    // content-correct cache key (the poisoned-cache half of the roamed-pose bug).
    const j = this.bakeSrcLevel(t);
    const sub = new Float32Array(vpa * vpa);
    for (let gz = 0; gz <= gridN; gz++) {
      const nz = Math.min(Math.max(t.tz0 + gz * t.strideTexels, cfg.latMin), cfg.latMax);
      const srow = gz * vpa;
      for (let gx = 0; gx <= gridN; gx++) {
        const nx = Math.min(Math.max(t.tx0 + gx * t.strideTexels, cfg.latMin), cfg.latMax);
        sub[srow + gx] = this.heightAtLattice(j, nx, nz);
      }
    }
    // content-addressed cache key: seed salt + grid + placement + source level +
    // the fold of the covering chunks' content hashes (source-agnostic —
    // generated hashes are zero and the seed carries identity; Estonia recuts
    // invalidate). The fold identifies the tile's INTENDED source chunks, not the
    // window state at bake time, so a far tile baked from a CLAMPED (pre-S8c,
    // too-small) coarsest window cached its flat geometry under the SAME key its
    // now-real bake computes — the version tag retires those stale flat entries
    // (-sb2 → -sb3; the coarsest window now always spans the box ⇒ far tiles always
    // bake real, so the fold is henceforth an honest identity).
    let fold = 0n;
    const geo = this.layerGeo('height');
    const plan = (this.hWin[j] as HeightWindow).plan;
    const s = plan.stride;
    const jx0 = this.manifestFormat === 2 ? Math.floor((t.tx0 + 0.5) / s - 0.5) : Math.floor(t.tx0 / s);
    const jz0 = this.manifestFormat === 2 ? Math.floor((t.tz0 + 0.5) / s - 0.5) : Math.floor(t.tz0 / s);
    const jres = this.manifestFormat === 2
      ? Math.ceil((t.tx0 + t.tileTexels + 0.5) / s - 0.5) - jx0 + 1
      : Math.ceil(t.tileTexels / s) + 1;
    for (const key of chunksInWindow(geo, plan.lod, jx0, jz0, jres)) fold ^= this.chunkHash('height', key);
    const skirtLevel = cfg.skirt ? t.level : -1;
    const opts: HeightDagOpts = skirtLevel >= 0 ? { skirtLevel } : {};
    const suffix = `-sb3-s${t.strideTexels}-j${j}-${t.tx0}x${t.tz0}-h${fold.toString(16)}${skirtLevel >= 0 ? `-sk${skirtLevel}` : ''}`;
    const cacheKey = heightDagCacheKey(cfg.seed >>> 0, gridN, suffix);
    let built: HeightDagResult | null = await getCachedHeightDag(cacheKey);
    if (built) {
      this.nCache++;
    } else {
      this.nBakeInFlight++;
      try {
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
    // remap tile-local grid coords → LOCAL coords in `packUnit`-texel units
    // (clamped to the lattice); word0 = gx(0-12) | skirt code(13-15) | gz(16-31),
    // per-tile origin/cellSize words map them to world. packUnit=1 on every data
    // rung (identical world positions to the old global form); the S6f super
    // rungs pack coarser units so gridN·stride at ANY level fits the 13-bit
    // field — world pos is exact either way: origin + lx·(cell·packUnit).
    let packUnit = 1;
    while (gridN * t.strideTexels > 0x1fff * packUnit) packUnit *= 2;
    const gridVerts = new Uint32Array(built.gridVerts.length);
    for (let i = 0; i < built.gridVerts.length; i++) {
      const p = built.gridVerts[i] as number;
      const code = (p >>> 13) & 0x7;
      const lx = Math.round((Math.min(Math.max(t.tx0 + (p & 0x1fff) * t.strideTexels, cfg.latMin), cfg.latMax) - t.tx0) / packUnit);
      const lz = Math.round((Math.min(Math.max(t.tz0 + ((p >>> 16) & 0xffff) * t.strideTexels, cfg.latMin), cfg.latMax) - t.tz0) / packUnit);
      if (lx < 0 || lx > 0x1fff || lz < 0 || lz > 0xffff) throw new Error(`stream tile ${t.key}: local vert coord out of range (${lx},${lz})`);
      gridVerts[i] = ((lx & 0x1fff) | (code << 13) | ((lz & 0xffff) << 16)) >>> 0;
    }
    return {
      key: t.key,
      slot: -1, // assigned by the caller (boot: index; refine: the reserved slot)
      gridVerts,
      indices: built.indices.slice(),
      clusterData: packClusters(built.clusters),
      clusterCount: built.clusters.length,
      originX: cfg.origin + t.tx0 * cfg.cell,
      originZ: cfg.origin + t.tz0 * cfg.cell,
      cellSize: cfg.cell * packUnit,
      x0: t.tx0,
      z0: t.tz0,
      size: t.tileTexels,
    };
  }

  // ---- node-probe surface (tools/probe-streambrain.ts) — test-only ------------------------

  countersForProbe(): { scrolls: number } {
    return { scrolls: this.nScrolls };
  }
  residentCountForProbe(): number {
    let n = 0;
    this.tree.forEachResident(() => n++);
    return n;
  }

  // ---- counters (F-9) ----------------------------------------------------------------------

  emitCounters(): void {
    this.deps.emit({
      kind: 'counters',
      counters: {
        ...this.tree.counters(),
        'stream.tiles.freeslots': this.tileFree.length,
        'stream.bake.cache': this.nCache,
        'stream.bake.built': this.nBuilt,
        'stream.bake.inflight': this.nBakeInFlight,
        'stream.fetch.total': this.nFetch,
        'stream.fetch.inflight': this.nFetchInFlight,
        'stream.lru.mb': Math.round(this.lruBytes / 2 ** 20),
        'stream.ram.mb': Math.round(this.ramBytes() / 2 ** 20),
        'stream.scrolls': this.nScrolls,
        ...(this.ftBand ? this.ftBand.counters() : {}),
      },
    });
  }
}
