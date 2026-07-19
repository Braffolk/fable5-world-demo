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
 * frame across ALL upload classes) — that ordering IS the promote-after-fill
 * transaction (F-8): a scroll's fills precede its origin commit, and a tile
 * transaction applies as one atomic drain step.
 */

import type { ChunkKey, ChunkPayload, LayerName, ManifestFormat, WorldGrid } from '../../world/source/WorldSource';
import type { FieldPlan } from './PlaneFill';
import type { LevelGridEdit } from './PartitionTree';
import type { PackedLevel } from '../build/CrownPack';

/** which GPU plane a fill/origin packet targets. */
export type PlaneKind = 'height' | 'biome' | 'fields' | 'water' | 'waterFar' | 'watercover' | 'waterCoverFar' | 'soil' | 'geology';

/** serialized layer meta the brain needs (subset of WorldLayerMeta + the chunk
 *  key enumeration per lod — coverage() point queries become set lookups). */
export interface BrainLayerMeta {
  lods: number[];
  texelMeters?: number;
  baseTexelMeters?: number;
  finestLod?: number;
  authorityLod?: number;
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
  manifestFormat: ManifestFormat;
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

/** S8: arm the runtime FARTILE band (post-registry-build). Carries the library
 *  knowledge the brain lacks — per-species crown brick pools (the picked coarse
 *  pyramid level, flat stride-11 = FarTilesSplat SPECIES_BRICK_STRIDE) keyed by
 *  idF, the species→VegClass table (idF = class·8 + variant&3), the ring-grade
 *  cell ladder, and the pool geometry the brain seeds its slot/granule free-lists
 *  from. Sent ONCE; the brain then owns per-cell residency + splat + emit + pack. */
export interface FtSpeciesPool {
  idF: number;
  /** occupied bricks of the picked crown pyramid level, flat stride-11 (crown-local). */
  bricks: Float32Array;
  crownMinY: number;
  barkR: number;
  barkG: number;
  barkB: number;
}
export interface FtArmMsg {
  kind: 'ftArm';
  pools: FtSpeciesPool[];
  /** species id → VegClass (idF = class·8 + variant&3); -1 = no crown (skip). */
  speciesToClass: Int32Array;
  /** ring-grade cell ladder: cellSizes[i] applies at ring distance < gradeRadii[i] (m). */
  cellSizes: number[];
  gradeRadii: number[];
  tileSize: number;
  /** far-tile residency unit side (m) — the graded bake cell (512). */
  cellMeters: number;
  /** the far-tile ring outer radius (m) — cells within this get baked. */
  horizon: number;
  /** near cutoff: the tile head renders only beyond nearDist (the aggDist handoff). */
  nearDist: number;
  /** crown reach margin (m) — trees within this of a cell's box still splat into it. */
  reachMargin: number;
  /** packed leaf tint (matParam) for the tile heads. */
  tint: number;
  /** pool geometry (from reg.fartilePoolInfo) — seeds the brain's free-lists. */
  slots: number;
  clusterCap: number;
  granules: number;
}

export type MainToBrain = BrainInitMsg | PoseMsg | FetchResMsg | BootTilesMsg | PoolInfoMsg | FtArmMsg;

// ---- brain → main -------------------------------------------------------------------

export interface FetchReqMsg {
  kind: 'fetch';
  id: number;
  layer: LayerName;
  key: ChunkKey;
}

/** one baked tile's transferable geometry (clusters in the DagCache packed
 *  Float64 form — deserializeClusters on the main side). The brain assigns the
 *  `slot` (it owns the pool free-list — the single residency authority, S6f);
 *  main is a pure applier. */
export interface TileGeometry {
  key: string;
  /** brain-assigned pool slot this tile loads into */
  slot: number;
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
  /** BRAIN-INTERNAL taint bookkeeping (format 2 only) — set by bakeTile when the
   *  bake read DEGRADED window content (a failed-fetch chunk's coarse underlay),
   *  consumed + STRIPPED brain-side before the geometry ships (never crosses the
   *  worker boundary). srcChunkKeys = the packed support chunk keys at srcWin. */
  srcDegraded?: boolean;
  srcWin?: number;
  srcChunkKeys?: number[];
}

/** MAILBOX packets — drained strictly FIFO under the ONE token bucket. The tile
 *  TRANSACTIONS (refine/merge) each apply as ONE atomic drain step (park+attach
 *  or unpark+evict complete before the next frame's cull), so a region is never
 *  seen at two LODs (S6f — the partition-tree residency rewrite). */
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
  /** REFINE: park the parent's slot (clusterCount→0, geometry retained), attach
   *  the ≤4 baked children into their brain-assigned slots, update the level grid. */
  | { kind: 'tileRefine'; parkSlot: number; children: TileGeometry[]; levelGrid: LevelGridEdit[] }
  /** MERGE: unpark the parent's retained slot (draw restored, instant — no bake),
   *  evict the ≤4 child slots, update the level grid. */
  | { kind: 'tileMerge'; unparkSlot: number; freeSlots: number[]; levelGrid: LevelGridEdit[] }
  /** REFRESH: re-attach honest geometry into an ALREADY-RESIDENT slot in place —
   *  no tree/level-grid change. Emitted after a degraded-content bake's source
   *  chunk HEALS (the fetch that failed at bake time finally landed): the flat
   *  simplified-lattice tile is replaced by the real full-lattice bake, closing
   *  the T-junction seam holes it opened against fine neighbors. `parked` re-parks
   *  a retained parent payload after the overwrite (unpark → attach → park). */
  | { kind: 'tileRefresh'; tile: TileGeometry; parked: boolean }
  /** S8 FARTILE ATTACH: one baked far-tile → its brain-assigned pool slot + granule
   *  ids (brickBase = poolBrickBase + gid·128). `levels` = the tile's packed voxel
   *  pyramid (CrownPack words). Applied as one drain step (reg.attachFartileSlot). */
  | { kind: 'ftAttach'; slot: number; granules: Uint32Array; center: [number, number, number]; levels: PackedLevel[] }
  /** S8 FARTILE EVICT: park these pool slots (draw dies; the brain returns the slots
   *  + their granules to ITS free-lists). */
  | { kind: 'ftEvict'; slots: number[] };

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

/** a boot node: its baked geometry (+ brain-assigned slot) and whether it is a
 *  fringe LEAF (rendered) or a parked ancestor (attached then parked — the
 *  ancestor-closed resident subtree, §4). */
export interface BootTile {
  tile: TileGeometry;
  isLeaf: boolean;
}

export interface BootTilesDoneMsg {
  kind: 'bootTilesDone';
  tiles: BootTile[];
  poolMax: { v: number; t: number; c: number };
  /** total pool slots to reserve (resident subtree + refinement headroom) */
  slots: number;
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
  // RemoteWorldSource's decodePlanes returns per-plane SUBARRAY views over ONE
  // raw buffer — dedupe so the transfer list never lists that buffer twice
  // (postMessage rejects duplicate ArrayBuffers). The generated source's planes
  // are distinct arrays, so this is a no-op there.
  if (p.kind === 'planes') return [...new Set(p.planes.map((u) => u.buffer))];
  const t: Transferable[] = [];
  for (const col of Object.values(p.cols)) if (col) t.push(col.buffer);
  return [...new Set(t)];
}

/** transferable buffers of one baked tile. */
function tileTransfers(t: TileGeometry): Transferable[] {
  return [t.gridVerts.buffer, t.indices.buffer, t.clusterData.buffer];
}

/** transfer list for a mailbox packet batch. */
export function packetTransfers(packets: StreamPacket[]): Transferable[] {
  const t: Transferable[] = [];
  for (const p of packets) {
    if (p.kind === 'fill') {
      if (p.f32) t.push(p.f32.buffer);
      if (p.u8) t.push(p.u8.buffer);
    } else if (p.kind === 'tileRefine') {
      for (const c of p.children) t.push(...tileTransfers(c));
    } else if (p.kind === 'tileRefresh') {
      t.push(...tileTransfers(p.tile));
    } else if (p.kind === 'ftAttach') {
      t.push(p.granules.buffer);
      for (const l of p.levels) t.push(l.words.buffer, l.occupied.buffer);
    }
  }
  return [...new Set(t)];
}

/** upload-cost estimate (bytes) of one baked tile. */
function tileBytes(t: TileGeometry): number {
  // hf verts (1 w) + indices + cluster/DAG/link records (~22 w per cluster) + mesh record
  return t.gridVerts.byteLength + t.indices.byteLength + t.clusterCount * 22 * 4 + 72;
}

/** upload-cost estimate (bytes) of one packet — the token bucket's currency. */
export function packetBytes(p: StreamPacket): number {
  if (p.kind === 'fill') return p.f32 ? p.f32.byteLength : p.u8 ? p.u8.byteLength : 0;
  if (p.kind === 'tileRefine') {
    let b = 72; // parent park + level-grid poke
    for (const c of p.children) b += tileBytes(c);
    return b;
  }
  if (p.kind === 'tileMerge') return 72 + p.freeSlots.length * 8; // unpark + evicts + grid poke
  if (p.kind === 'tileRefresh') return 72 + tileBytes(p.tile); // in-place re-attach
  if (p.kind === 'ftAttach') {
    let b = 72;
    for (const l of p.levels) b += l.words.byteLength; // the writeBuffer'd brick words dominate
    return b;
  }
  if (p.kind === 'ftEvict') return 72 + p.slots.length * 8;
  return 64; // origin commit — uniform poke
}
