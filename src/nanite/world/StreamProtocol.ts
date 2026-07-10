/**
 * StreamProtocol — the brain↔main message contract (SPEC-STREAMING-WORLD §4,
 * F-8, S5). Pure types, importable by the worker, the main-thread client and
 * node probes.
 *
 * Direction MAIN → BRAIN: init (manifest snapshot + the field plan main
 * computed — one placement truth), fetch results (the WorldSource RPC proxy —
 * the GENERATED source wraps GPU boot passes so it cannot leave the main
 * thread; Estonia's RemoteWorldSource moves brain-side at S6 through the same
 * WorldSource interface), pose at ~10 Hz, the tile boot kick, the pool
 * geometry, and attach acks/nacks (slot alloc is registry state = main-side;
 * the brain owns every policy decision).
 *
 * Direction BRAIN → MAIN: fetch requests, one boot-tiles bundle (pre-registry,
 * NOT budget-drained — it sizes the pool caps), and the PACKET MAILBOX. The
 * mailbox is drained strictly FIFO under ONE token bucket (≤2 MB or ≤1.5 ms a
 * frame across ALL upload classes) — that ordering IS the demote-before-scroll
 * / promote-after-fill transaction (F-8): the brain emits tile evicts BEFORE
 * the plane fills that overwrite their source region, and promoted tile
 * attaches AFTER the fills their verts read.
 */

import type { ChunkKey, ChunkPayload, LayerName, WorldGrid } from '../../world/source/WorldSource';
import type { FieldPlan } from './PlaneFill';

/** which GPU plane a fill/origin packet targets. */
export type PlaneKind = 'height' | 'biome' | 'fields' | 'water' | 'waterFar';

/** serialized layer meta the brain needs (subset of WorldLayerMeta + the chunk
 *  key enumeration per lod — coverage() point queries become set lookups). */
export interface BrainLayerMeta {
  lods: number[];
  texelMeters?: number;
  planes?: readonly string[];
  /** packed keys per lod (packChunkKey) — existence = authoritative presence */
  chunkKeys: Record<number, Float64Array>;
  /** content hash64 per chunk (same order as chunkKeys), folded into the tile
   *  DAG cache key — source-agnostic identity (generated: zeros; seed salts). */
  chunkHashes: Record<number, BigUint64Array>;
}

export interface BrainTileConfig {
  gridN: number;
  tilesPerSide: number;
  skirt: boolean;
  /** numeric world seed — salts the tile DAG cache key */
  seed: number;
  /** finest-lattice cell size (m) + world coord of lattice sample 0 */
  cell: number;
  origin: number;
  /** finest lattice sample bounds (inclusive) — generated: [0, res-1] */
  latMin: number;
  latMax: number;
}

export interface BrainInitMsg {
  kind: 'init';
  grid: WorldGrid;
  layers: Partial<Record<LayerName, BrainLayerMeta>>;
  plan: FieldPlan;
  tiles: BrainTileConfig;
}

export interface PoseMsg {
  kind: 'pose';
  x: number;
  z: number;
  /** m/s velocity estimate (prefetch heading) */
  vx: number;
  vz: number;
}

export interface FetchResMsg {
  kind: 'fetchRes';
  id: number;
  ok: boolean;
  payload: ChunkPayload | null;
  error?: string;
}

/** kick the tile boot plan at a pose (pre-registry-build). */
export interface BootTilesMsg {
  kind: 'bootTiles';
  camX: number;
  camZ: number;
}

/** the reserved pool geometry — arms the runtime tile streaming. */
export interface PoolInfoMsg {
  kind: 'poolInfo';
  slots: number;
  vertCap: number;
  triCap: number;
  clusterCap: number;
}

export interface AttachAckMsg {
  kind: 'attachAck';
  key: string;
  ok: boolean;
}

export type MainToBrain = BrainInitMsg | PoseMsg | FetchResMsg | BootTilesMsg | PoolInfoMsg | AttachAckMsg;

// ---- brain → main -------------------------------------------------------------------

export interface FetchReqMsg {
  kind: 'fetch';
  id: number;
  layer: LayerName;
  key: ChunkKey;
}

/** one baked tile's transferable geometry (clusters in the DagCache packed
 *  Float64 form — deserializeClusters on the main side). */
export interface TileGeometry {
  key: string;
  /** tile-local packed grid verts (gx | code<<13 | gz<<16) */
  gridVerts: Uint32Array;
  indices: Uint32Array;
  clusterData: Float64Array;
  clusterCount: number;
  /** per-tile mesh-record origin words (StreamOrigin-relative; F-3) */
  originX: number;
  originZ: number;
  cellSize: number;
  /** finest-lattice footprint (probe/victim policy bookkeeping) */
  x0: number;
  z0: number;
  size: number;
}

/** MAILBOX packets — drained strictly FIFO under the ONE token bucket. */
export type StreamPacket =
  | {
      kind: 'fill';
      plane: PlaneKind;
      level: number;
      x: number;
      y: number;
      w: number;
      h: number;
      /** r32f planes */
      f32?: Float32Array;
      /** interleaved rgba8 planes */
      u8?: Uint8Array;
    }
  | {
      kind: 'planeOrigin';
      plane: PlaneKind;
      level: number;
      originX: number;
      originZ: number;
      n0x: number;
      n0z: number;
      phaseX: number;
      phaseY: number;
    }
  | { kind: 'attach'; tile: TileGeometry }
  | { kind: 'evict'; key: string };

export interface PacketsMsg {
  kind: 'packets';
  packets: StreamPacket[];
}

/** boot planes fully assembled + emitted (main drains unbudgeted, then flips
 *  the DataTextures live). */
export interface PlanesReadyMsg {
  kind: 'planesReady';
  /** brain-side retained RAM (stated + logged — §6 ledger) */
  ramBytes: number;
}

export interface BootTilesDoneMsg {
  kind: 'bootTilesDone';
  tiles: TileGeometry[];
  poolMax: { v: number; t: number; c: number };
  maxTiles: number;
  levels: number;
  nCache: number;
  nBuilt: number;
}

export interface BrainCountersMsg {
  kind: 'counters';
  counters: Record<string, number>;
}

export interface BrainLogMsg {
  kind: 'log';
  level: 'log' | 'warn' | 'error';
  msg: string;
}

export type BrainToMain =
  | FetchReqMsg
  | PacketsMsg
  | PlanesReadyMsg
  | BootTilesDoneMsg
  | BrainCountersMsg
  | BrainLogMsg;

/** transfer list for a ChunkPayload crossing the worker boundary. */
export function payloadTransfers(p: ChunkPayload | null): Transferable[] {
  if (!p) return [];
  if (p.kind === 'height') return [p.heights.buffer];
  if (p.kind === 'planes') return p.planes.map((u) => u.buffer);
  const t: Transferable[] = [];
  for (const col of Object.values(p.cols)) if (col) t.push(col.buffer);
  return [...new Set(t)];
}

/** transfer list for a mailbox packet batch. */
export function packetTransfers(packets: StreamPacket[]): Transferable[] {
  const t: Transferable[] = [];
  for (const p of packets) {
    if (p.kind === 'fill') {
      if (p.f32) t.push(p.f32.buffer);
      if (p.u8) t.push(p.u8.buffer);
    } else if (p.kind === 'attach') {
      t.push(p.tile.gridVerts.buffer, p.tile.indices.buffer, p.tile.clusterData.buffer);
    }
  }
  return [...new Set(t)];
}

/** upload-cost estimate (bytes) of one packet — the token bucket's currency. */
export function packetBytes(p: StreamPacket): number {
  if (p.kind === 'fill') return p.f32 ? p.f32.byteLength : p.u8 ? p.u8.byteLength : 0;
  if (p.kind === 'attach') {
    const t = p.tile;
    // hf verts (1 w) + indices + cluster/DAG/link records (~22 w per cluster) + mesh record
    return t.gridVerts.byteLength + t.indices.byteLength + t.clusterCount * 22 * 4 + 72;
  }
  return 64; // origin commit / evict — uniform pokes + a mesh record
}
