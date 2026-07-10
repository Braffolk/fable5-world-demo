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
 *    tile attaches, future instance/brick writes). FIFO IS the demote→scroll→
 *    promote transaction (F-8); the bucket always passes ≥1 packet a frame so
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
import type { GeometryRegistry } from './GeometryRegistry';
import { planField, layerGeom, latticeWorld, chunkBox, type FieldPlan } from './PlaneFill';
import { TerrainField } from './TerrainField';
import type {
  BrainInitMsg,
  BrainLayerMeta,
  BrainToMain,
  BootTilesDoneMsg,
  MainToBrain,
  PlaneKind,
  StreamPacket,
  TileGeometry,
} from './StreamProtocol';
import { packetBytes, payloadTransfers } from './StreamProtocol';

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
  maxTiles: number;
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
  /** the FIFO mailbox — never reordered (F-8) */
  private readonly mailbox: StreamPacket[] = [];
  /** tile key → pool slot (registry bookkeeping, not policy) */
  private readonly slots = new Map<string, number>();
  private planesReadyResolve: ((ram: number) => void) | null = null;
  private bootTilesResolve: ((msg: BootTilesDoneMsg) => void) | null = null;
  private bootTiles: TileGeometry[] = [];
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

  /** manifest snapshot → transferable brain init (chunk key/hash tables per
   *  layer+lod — existence lookups become brain-local; hashes salt tile DAG
   *  cache keys). */
  private buildInit(manifest: WorldManifest): BrainInitMsg {
    const layers: BrainInitMsg['layers'] = {};
    for (const layer of ['height', 'biome', 'fields', 'water'] as const) {
      const meta = manifest.layers[layer];
      if (!meta) continue;
      const chunkKeys: Record<number, Float64Array> = {};
      const chunkHashes: Record<number, BigUint64Array> = {};
      for (const lod of meta.lods) {
        const keys = manifest.chunks(layer, lod);
        const packed = new Float64Array(keys.length);
        const hashes = new BigUint64Array(keys.length);
        keys.forEach((k, i) => {
          packed[i] = packChunkKey(k.lod, k.cx, k.cz);
          hashes[i] = manifest.coverage(layer, k)?.hash64 ?? 0n;
        });
        chunkKeys[lod] = packed;
        chunkHashes[lod] = hashes;
      }
      const entry: BrainLayerMeta = { lods: [...meta.lods], chunkKeys, chunkHashes };
      if (meta.texelMeters !== undefined) entry.texelMeters = meta.texelMeters;
      if (meta.planes !== undefined) entry.planes = meta.planes;
      layers[layer] = entry;
    }
    const geo = layerGeom(manifest, 'height');
    const box = chunkBox(manifest.chunks('height', Math.min(...(manifest.layers.height?.lods ?? [0]))));
    if (!box) throw new Error('StreamBrainClient: height layer has no chunks');
    return {
      kind: 'init',
      grid: manifest.grid,
      layers,
      plan: this.plan,
      tiles: {
        gridN: this.tileOpts.gridN,
        tilesPerSide: this.tileOpts.tilesPerSide,
        skirt: this.tileOpts.skirt,
        seed: this.tileOpts.seed,
        cell: geo.texel0,
        origin: latticeWorld(geo, 0, 0, 'x'),
        // the tile clipmap is a SQUARE lattice; a non-square coverage (Estonia's
        // Taevaskoja pilot spans chunk X 148-155 but Z 93-94) needs bounds that
        // cover BOTH axes' ranges, else latZ clamps into the X band and the tiles
        // land tens of km off the camera (frustum+size cull then drop them all).
        // The generated world's box is square (2×2) ⇒ min/max collapse to the
        // old X-only values, bit-identical.
        latMin: Math.min(box.minX, box.minZ) * geo.chunkRes,
        latMax: (Math.max(box.maxX, box.maxZ) + 1) * geo.chunkRes - 1,
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
          maxTiles: msg.maxTiles,
          levels: msg.levels,
          nCache: msg.nCache,
          nBuilt: msg.nBuilt,
        });
      };
      this.post({ kind: 'bootTiles', camX, camZ });
    });
  }

  /** attach the boot ring into pool slots (post-registry-build, frame-1
   *  terrain), ack each, then arm runtime streaming with the pool geometry. */
  attachBootTiles(reg: GeometryRegistry): void {
    this.reg = reg;
    for (const tile of this.bootTiles) {
      const ok = this.attachTile(tile);
      this.post({ kind: 'attachAck', key: tile.key, ok });
    }
    this.bootTiles = [];
    const cap = reg.tilePoolCap;
    this.post({ kind: 'poolInfo', slots: reg.tilePoolSlotCount, vertCap: cap.vertCap, triCap: cap.triCap, clusterCap: cap.clusterCap });
  }

  // ---- per-frame -------------------------------------------------------------------

  /** feed the camera pose to the brain at ~10 Hz (velocity from the pose delta;
   *  teleports are detected brain-side from the discontinuity). */
  update(camX: number, camZ: number): void {
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
    if (this.mailbox.length === 0) {
      this.lastDrainMs = 0;
      this.lastDrainBytes = 0;
      this.lastDrainPackets = 0;
      return;
    }
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
      case 'attach': {
        const ok = this.attachTile(p.tile);
        this.post({ kind: 'attachAck', key: p.tile.key, ok });
        break;
      }
      case 'evict': {
        const slot = this.slots.get(p.key);
        if (slot !== undefined && this.reg) {
          this.reg.evictHeightDagTile(slot);
          this.slots.delete(p.key);
        }
        break;
      }
    }
  }

  /** alloc + attach one tile (per-tile origin words — F-3); false = pool dry
   *  (the brain picks a victim and retries — policy stays brain-side). */
  private attachTile(tile: TileGeometry): boolean {
    const reg = this.reg;
    if (!reg) return false;
    const existing = this.slots.get(tile.key);
    const slot = existing ?? reg.allocTileSlot();
    if (slot < 0) return false;
    reg.attachHeightDagTile(
      slot,
      { gridVerts: tile.gridVerts, indices: tile.indices, clusters: unpackClusters(tile.clusterData, tile.clusterCount) },
      { originX: tile.originX, originZ: tile.originZ, cellSize: tile.cellSize },
    );
    this.slots.set(tile.key, slot);
    return true;
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
