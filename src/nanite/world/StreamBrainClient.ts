/**
 * StreamBrainClient — the main-thread end of the streaming brain (SPEC §4,
 * F-8, S5). The main thread does ONLY what cannot leave it (law 5):
 *
 *  - services the brain's WorldSource fetch RPC (the GENERATED source wraps
 *    GPU boot passes — Heightfield.generate/runScatter need the renderer — so
 *    it stays here behind the postMessage proxy; RemoteWorldSource moves
 *    brain-side at S6 with NO behavior change beyond where the fetch runs);
 *  - drains the packet MAILBOX strictly FIFO under ONE token bucket (≤2 MB or
 *    ≤1.5 ms a frame across ALL upload classes — plane writeTexture fills,
 *    tile attaches, future instance/brick writes). FIFO IS the fill→origin→tile
 *    transaction order (F-8); the bucket always passes ≥1 packet a frame so
 *    the stream can never stall.
 *  - allocates tile slots (registry state) and feeds attach acks/NACKs back —
 *    every POLICY decision (victim pick, retry, priorities) stays in the brain.
 *
 * Boot is a different drain mode: plane fills and the boot tile bundle land
 * before the frame loop exists, so they apply unbudgeted and the DataTextures
 * flip live once (needsUpdate) — on the generated world that reproduces the
 * pre-S5 static fill byte-for-byte, which is the S5 parity gate.
 */

import type { Renderer } from 'three/webgpu';
import type { Texture } from 'three';
import { unpackClusters } from '../build/DagCache';
import type { ChunkKey, LayerName, WorldManifest, WorldSource } from '../../world/source/WorldSource';
import { packChunkKey } from '../../world/source/Lac1';
import { packChunkKeyV2 } from '../../world/source/Lac2';
import type { GeometryRegistry } from './GeometryRegistry';
import { planField, layerGeom, latticeWorld, coverageExtentLattice, type FieldPlan } from './PlaneFill';
import { TerrainField } from './TerrainField';
import type {
  BootTile,
  BrainInitMsg,
  BrainLayerMeta,
  BrainToMain,
  BootTilesDoneMsg,
  FtArmMsg,
  MainToBrain,
  PlaneKind,
  StreamPacket,
  TileGeometry,
} from './StreamProtocol';
import { packetBytes, payloadTransfers } from './StreamProtocol';
import type { InstanceBand } from './InstanceBand';

/** the ONE token bucket (F13): per-frame upload budget across all classes. */
const BUCKET_BYTES = 2 * 2 ** 20;
const BUCKET_MS = 1.5;
/** camera pose feed rate to the brain. */
const POSE_HZ_MS = 100;

/** three's WebGPUBackend surface for raw partial texture writes (precedent:
 *  ProfileBoot/GpuProfiler backend accessors). */
interface RawBackend {
  device?: GPUDevice;
  get(tex: Texture): { texture?: GPUTexture } | undefined;
}

export interface BrainTileOpts {
  gridN: number;
  tilesPerSide: number;
  skirt: boolean;
  seed: number;
}

export interface BootTilesInfo {
  count: number;
  poolMax: { v: number; t: number; c: number };
  /** provisioned pool slot count (resident subtree + refinement headroom, §5) */
  slots: number;
  levels: number;
  nCache: number;
  nBuilt: number;
}

export class StreamBrainClient {
  private readonly source: WorldSource;
  private readonly worker: Worker;
  private readonly plan: FieldPlan;
  private readonly tileOpts: BrainTileOpts;
  private field: TerrainField | null = null;
  private reg: GeometryRegistry | null = null;
  /** S7/S9a: the streamed instance bands (Estonia) — the tree/boulder band and the
   *  understory/debris band. Empty on the generated world (its instances are boot-
   *  bound). All driven off the same pose/drain cadence, sharing one instance pool. */
  private readonly bands: InstanceBand[] = [];
  /** the FIFO mailbox — never reordered (F-8) */
  private readonly mailbox: StreamPacket[] = [];
  private planesReadyResolve: ((ram: number) => void) | null = null;
  private bootTilesResolve: ((msg: BootTilesDoneMsg) => void) | null = null;
  private bootTiles: BootTile[] = [];
  private booting = true;
  private brainCounters: Record<string, number> = {};
  /** serial fetch service — the generated source's lazy GPU readbacks
   *  (biome/fields planes) must not race themselves */
  private fetchChain: Promise<void> = Promise.resolve();
  private lastPoseAt = 0;
  private lastPoseX = 0;
  private lastPoseZ = 0;
  private havePose = false;
  // token-bucket telemetry (F-9)
  private lastDrainMs = 0;
  private lastDrainBytes = 0;
  private lastDrainPackets = 0;

  constructor(source: WorldSource, manifest: WorldManifest, tileOpts: BrainTileOpts) {
    this.source = source;
    this.tileOpts = tileOpts;
    this.plan = planField(manifest);
    this.worker = new Worker(new URL('./StreamBrain.worker.ts', import.meta.url), {
      type: 'module',
      name: 'stream-brain',
    });
    this.worker.onmessage = (e: MessageEvent<BrainToMain>): void => this.onBrain(e.data);
    this.worker.onerror = (e: ErrorEvent): void => {
      // eslint-disable-next-line no-console
      console.error(`[laas] stream brain crashed: ${e.message}`);
    };
    this.worker.postMessage(this.buildInit(manifest));
  }

  get fieldPlan(): FieldPlan {
    return this.plan;
  }

  /** S7/S9a: arm a streamed instance band (post-registry-build; Estonia only). Called
   *  once per band (tree/boulder, understory/debris); all share the pose/drain cadence. */
  addInstanceBand(band: InstanceBand): void {
    this.bands.push(band);
  }

  /** S8: ship the far-tile library pools + pool geometry to the brain (post-
   *  registry-build; BOTH sources). The brain then owns per-cell fartile residency;
   *  its ftAttach/ftEvict packets drain through the SAME token bucket as tiles. */
  armFartiles(msg: FtArmMsg): void {
    const transfer: Transferable[] = [msg.speciesToClass.buffer, ...msg.pools.map((p) => p.bricks.buffer)];
    this.post(msg, transfer);
  }

  /** manifest snapshot → transferable brain init (chunk key/hash tables per
   *  layer+lod — existence lookups become brain-local; hashes salt tile DAG
   *  cache keys). */
  private buildInit(manifest: WorldManifest): BrainInitMsg {
    const packKey = manifest.format === 2 ? packChunkKeyV2 : packChunkKey;
    const layers: BrainInitMsg['layers'] = {};
    // S8: 'trees' joins the brain's layer set — the runtime fartile band (brain-side)
    // needs the tree records' existence keys + fetches them through the RPC. (boulders
    // stay main-side via the InstanceBand — fartiles are trees only.)
    // #116: 'soil' too — the brain fills/scrolls the soil plane, so it needs soil's chunk
    // existence keys (else fetchChunk returns null and the plane fills all-zero). No-op on
    // the generated world (no soil layer ⇒ `!meta` skip).
    for (const layer of ['height', 'biome', 'fields', 'water', 'watercover', 'canopy', 'trees', 'soil', 'geology'] as const) {
      const meta = manifest.layers[layer];
      if (!meta) continue;
      const chunkKeys: Record<number, Float64Array> = {};
      const chunkHashes: Record<number, BigUint64Array> = {};
      for (const lod of meta.lods) {
        const keys = manifest.chunks(layer, lod);
        const packed = new Float64Array(keys.length);
        const hashes = new BigUint64Array(keys.length);
        keys.forEach((k, i) => {
          packed[i] = packKey(k.lod, k.cx, k.cz);
          hashes[i] = manifest.coverage(layer, k)?.hash64 ?? 0n;
        });
        chunkKeys[lod] = packed;
        chunkHashes[lod] = hashes;
      }
      const entry: BrainLayerMeta = { lods: [...meta.lods], chunkKeys, chunkHashes };
      if (meta.texelMeters !== undefined) entry.texelMeters = meta.texelMeters;
      if (meta.baseTexelMeters !== undefined) entry.baseTexelMeters = meta.baseTexelMeters;
      if (meta.finestLod !== undefined) entry.finestLod = meta.finestLod;
      if (meta.authorityLod !== undefined) entry.authorityLod = meta.authorityLod;
      if (meta.planes !== undefined) entry.planes = meta.planes;
      layers[layer] = entry;
    }
    const geo = layerGeom(manifest, 'height');
    // COVERAGE BOX (S6g) — the domain the tile-residency PARTITION TREE tiles, hence
    // the ONLY region terrain can render in. Derived from `coverageExtentLattice`,
    // the SAME function planField sizes the coarsest "country floor" height level
    // against (S8c), so the tree domain and its coarse bakeable source cannot drift:
    // every fringe leaf the tree can create has a resident real coarse source ⇒ a
    // "far region with data but flat/absent terrain" is unrepresentable, not policed.
    const { latMin, latMax } = coverageExtentLattice(manifest);
    return {
      kind: 'init',
      manifestFormat: manifest.format,
      grid: manifest.grid,
      layers,
      plan: this.plan,
      tiles: {
        gridN: this.tileOpts.gridN,
        tilesPerSide: this.tileOpts.tilesPerSide,
        skirt: this.tileOpts.skirt,
        seed: this.tileOpts.seed,
        cell: geo.texel0,
        origin: latticeWorld(geo, geo.finestLod, 0, 'x'),
        latMin,
        latMax,
      },
    };
  }

  private post(msg: MainToBrain, transfer?: Transferable[]): void {
    this.worker.postMessage(msg, transfer ?? []);
  }

  private onBrain(msg: BrainToMain): void {
    switch (msg.kind) {
      case 'fetch':
        this.fetchChain = this.fetchChain.then(() => this.serviceFetch(msg.id, msg.layer, msg.key));
        break;
      case 'packets':
        this.mailbox.push(...msg.packets);
        break;
      case 'planesReady':
        this.planesReadyResolve?.(msg.ramBytes);
        this.planesReadyResolve = null;
        break;
      case 'bootTilesDone':
        this.bootTiles = msg.tiles;
        this.bootTilesResolve?.(msg);
        this.bootTilesResolve = null;
        break;
      case 'counters':
        this.brainCounters = msg.counters;
        break;
      case 'log':
        // eslint-disable-next-line no-console
        console[msg.level](`[laas][brain] ${msg.msg}`);
        break;
    }
  }

  private async serviceFetch(id: number, layer: LayerName, key: ChunkKey): Promise<void> {
    try {
      const payload = await this.source.fetch(layer, key);
      // records payloads alias the source's live bins — clone, never transfer
      const transfer = payload && payload.kind !== 'records' ? payloadTransfers(payload) : [];
      this.post({ kind: 'fetchRes', id, ok: true, payload }, transfer);
    } catch (e) {
      this.post({ kind: 'fetchRes', id, ok: false, payload: null, error: e instanceof Error ? e.message : String(e) });
    }
  }

  // ---- boot ------------------------------------------------------------------------

  /** allocate the TerrainField from the plan, then let the brain fill it (the
   *  streaming path at boot = full-window fills, drained unbudgeted). */
  async openField(): Promise<TerrainField> {
    if (this.field) throw new Error('StreamBrainClient: openField twice');
    this.field = TerrainField.fromPlan(this.plan);
    const ram = await new Promise<number>((resolve) => {
      this.planesReadyResolve = resolve;
    });
    this.drainAllNow(null);
    this.field.markAllDirty();
    // eslint-disable-next-line no-console
    console.log(`[laas] stream brain: planes filled (brain RAM ${(ram / 2 ** 20).toFixed(1)} MB retained — §6 ledger ≤128 MB)`);
    this.booting = false;
    return this.field;
  }

  /** plan + bake the spawn-centered boot ring set brain-side (pre-registry —
   *  the result sizes the pool caps). */
  bootTilesAt(camX: number, camZ: number): Promise<BootTilesInfo> {
    return new Promise<BootTilesInfo>((resolve) => {
      this.bootTilesResolve = (msg) => {
        resolve({
          count: msg.tiles.length,
          poolMax: msg.poolMax,
          slots: msg.slots,
          levels: msg.levels,
          nCache: msg.nCache,
          nBuilt: msg.nBuilt,
        });
      };
      this.post({ kind: 'bootTiles', camX, camZ });
    });
  }

  /** attach the boot resident subtree into its brain-assigned slots (post-
   *  registry-build, frame-1 terrain): fringe leaves render, parked ancestors are
   *  attached then PARKED (§4). Then arm runtime streaming with the pool geometry.  */
  attachBootTiles(reg: GeometryRegistry): void {
    this.reg = reg;
    for (const b of this.bootTiles) {
      this.attachTile(b.tile);
      if (!b.isLeaf) reg.parkTileSlot(b.tile.slot); // retained parent payload
    }
    this.bootTiles = [];
    const cap = reg.tilePoolCap;
    this.post({ kind: 'poolInfo', slots: reg.tilePoolSlotCount, vertCap: cap.vertCap, triCap: cap.triCap, clusterCap: cap.clusterCap });
  }

  // ---- per-frame -------------------------------------------------------------------

  /** feed the camera pose to the brain at ~10 Hz (velocity from the pose delta;
   *  teleports are detected brain-side from the discontinuity). */
  update(camX: number, camZ: number): void {
    for (const band of this.bands) band.update(camX, camZ); // own ~5 Hz throttle; residency diff is cheap
    const now = performance.now();
    if (this.havePose && now - this.lastPoseAt < POSE_HZ_MS) return;
    const dt = this.havePose ? (now - this.lastPoseAt) / 1000 : 0;
    const vx = dt > 0 ? (camX - this.lastPoseX) / dt : 0;
    const vz = dt > 0 ? (camZ - this.lastPoseZ) / dt : 0;
    this.lastPoseAt = now;
    this.lastPoseX = camX;
    this.lastPoseZ = camZ;
    this.havePose = true;
    this.post({ kind: 'pose', x: camX, z: camZ, vx, vz });
  }

  /** drain the mailbox strictly FIFO under the token bucket (≥1 packet a frame). */
  drain(renderer: Renderer): void {
    const t0 = performance.now();
    let bytes = 0;
    let n = 0;
    while (this.mailbox.length > 0) {
      if (n > 0 && (bytes >= BUCKET_BYTES || performance.now() - t0 >= BUCKET_MS)) break;
      const p = this.mailbox.shift() as StreamPacket;
      bytes += packetBytes(p);
      n++;
      this.applyPacket(p, renderer);
    }
    // S7/S9a: the instance bands share the SAME token bucket, drained AFTER the brain's
    // FIFO packets (tile/plane transaction ordering wins the budget first) and in arm
    // order (trees before understory). Block writes go through the live instArr mirror.
    for (const band of this.bands) {
      if (bytes >= BUCKET_BYTES || performance.now() - t0 >= BUCKET_MS) break;
      bytes += band.drainBudget(BUCKET_BYTES - bytes);
    }
    this.lastDrainMs = performance.now() - t0;
    this.lastDrainBytes = bytes;
    this.lastDrainPackets = n;
  }

  /** unbudgeted drain (boot phases — no frame to pace against). */
  private drainAllNow(renderer: Renderer | null): void {
    while (this.mailbox.length > 0) this.applyPacket(this.mailbox.shift() as StreamPacket, renderer);
  }

  private applyPacket(p: StreamPacket, renderer: Renderer | null): void {
    const field = this.field;
    if (!field) throw new Error('StreamBrainClient: packet before openField');
    switch (p.kind) {
      case 'fill': {
        const data = p.f32 ?? p.u8;
        if (!data) return;
        field.applyFill(p.plane, p.level, p.x, p.y, p.w, p.h, data);
        if (!this.booting && renderer) this.writeRegion(renderer, field, p.plane, p.level, p.x, p.y, p.w, p.h, data);
        break;
      }
      case 'planeOrigin':
        field.commitOrigin(p.plane, p.level, p.originX, p.originZ, p.phaseX, p.phaseY);
        break;
      // S6f tile TRANSACTIONS — each applies as ONE atomic drain step so a region
      // is never seen at two LODs. The brain owns the slots; main is a pure applier.
      case 'tileRefine': {
        const reg = this.reg;
        if (!reg) break;
        reg.parkTileSlot(p.parkSlot); // retain the coarse parent payload (§4)
        for (const child of p.children) this.attachTile(child); // ≤4 fine children
        field.applyLevelGrid(p.levelGrid); // surface authority (§7)
        break;
      }
      case 'tileMerge': {
        const reg = this.reg;
        if (!reg) break;
        reg.unparkTileSlot(p.unparkSlot); // restore the coarse parent (instant)
        for (const slot of p.freeSlots) reg.evictHeightDagTile(slot); // drop the fine children
        field.applyLevelGrid(p.levelGrid);
        break;
      }
      // S8 far-tile attach/evict — one drain step each (brain owns slot/granule
      // allocation; main is a pure applier). attachFartileSlot writes the brick words
      // writeBuffer-DIRECT post-mirror-release (needs the renderer).
      case 'ftAttach': {
        const reg = this.reg;
        if (!reg) break;
        reg.attachFartileSlot(p.slot, { vox: { levels: p.levels } }, p.center, p.granules, renderer);
        break;
      }
      case 'ftEvict': {
        const reg = this.reg;
        if (!reg) break;
        for (const slot of p.slots) reg.evictFartileSlot(slot);
        break;
      }
    }
  }

  /** attach one baked tile into its brain-assigned slot (per-tile origin words —
   *  F-3; the brain owns slot allocation, so this never fails on a full pool). */
  private attachTile(tile: TileGeometry): void {
    const reg = this.reg;
    if (!reg) return;
    reg.attachHeightDagTile(
      tile.slot,
      { gridVerts: tile.gridVerts, indices: tile.indices, clusters: unpackClusters(tile.clusterData, tile.clusterCount) },
      { originX: tile.originX, originZ: tile.originZ, cellSize: tile.cellSize },
    );
  }

  /** partial plane upload — writeTexture straight at the backing rect; if the
   *  texture isn't on the device yet (e.g. right after the ?profile swap) fall
   *  back to a full lazy re-upload from the coherent backing store. */
  private writeRegion(
    renderer: Renderer,
    field: TerrainField,
    plane: PlaneKind,
    level: number,
    x: number,
    y: number,
    w: number,
    h: number,
    data: Float32Array | Uint8Array,
  ): void {
    const lvl = field.levelFor(plane, level);
    const backend = (renderer as unknown as { backend: RawBackend }).backend;
    const entry = backend.get(lvl.tex);
    const gpuTex = entry?.texture;
    const device = backend.device;
    if (!gpuTex || !device) {
      lvl.tex.needsUpdate = true; // not created yet — full upload from backing
      return;
    }
    device.queue.writeTexture(
      { texture: gpuTex, origin: { x, y, z: 0 } },
      data as unknown as BufferSource,
      { offset: 0, bytesPerRow: w * 4, rowsPerImage: h },
      { width: w, height: h, depthOrArrayLayers: 1 },
    );
  }

  // ---- HUD counters (F-9) -------------------------------------------------------------

  counters(): Record<string, number> {
    return {
      ...this.brainCounters,
      ...Object.assign({}, ...this.bands.map((b) => b.counters())),
      'stream.mailbox.depth': this.mailbox.length,
      'stream.bucket.ms': Math.round(this.lastDrainMs * 100) / 100,
      'stream.bucket.kb': Math.round(this.lastDrainBytes / 1024),
      'stream.bucket.packets': this.lastDrainPackets,
    };
  }

  dispose(): void {
    this.worker.terminate();
  }
}
