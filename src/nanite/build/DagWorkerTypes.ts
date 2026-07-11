/**
 * Message protocol for the off-thread DAG builder (N8-D1d, D-N30). The build
 * chain (buildDag → clusterize, buildHeightGrid → buildDag) is three-free and
 * typed-arrays in/out, so it runs in a plain module Worker; the heavy CPU work
 * (terrain 4096² ≈ minutes) stays off the boot critical path. Kept in its own
 * file so the Worker and its main-thread client share one source of truth.
 */
import type { DagBuild, DagCluster, DagOpts } from './BuildDag';
import type { AggregateDagOpts } from './BuildAggregateDag';
import type { PackedPreparedCrown } from './CrownPack';
import type { HeightDagOpts, HeightDagStats } from './BuildHeightGrid';
import type { RockArchetype, RockMesh, RockMod } from '../../vegetation/RockGen';

/** build an adaptive terrain LOD DAG on a (gridN+1)² heightfield (gridN = 2^k) */
interface HeightDagReq {
  id: number;
  kind: 'height';
  /** (gridN+1)² row-major heights */
  heights: Float32Array;
  gridN: number;
  cellSize: number;
  originX: number;
  originZ: number;
  opts: HeightDagOpts;
}

/** QEM LOD DAG for an explicit mesh (BuildDag.buildDag) — a vegetation-head
 *  build that was a sync boot slab (cold-boot workerization, 2026-07-04).
 *  `clusterFill` mirrors the main thread's setClusterFill module knob into the
 *  worker's own Clusterize instance (worker builds must equal sync builds). */
export interface MeshDagReq {
  id: number;
  kind: 'mesh';
  /** interleaved DAG_VERT_STRIDE vertex pool (explicitToDagVerts output) */
  verts: Float32Array;
  vertStride: number;
  indices: Uint32Array;
  opts: DagOpts;
  clusterFill: number;
}

/** area-preserving aggregate DAG (BuildAggregateDag) for leaf crowns — the
 *  ~15 s sync boot slab. `aggErrorK` mirrors setAggLodErrorK (?leaflodk). */
export interface AggDagReq {
  id: number;
  kind: 'aggregate';
  verts: Float32Array;
  vertStride: number;
  indices: Uint32Array;
  opts: AggregateDagOpts;
  clusterFill: number;
  aggErrorK: number;
}

/** offline crown voxelization (VoxelizeCrown.prepareVoxelCrown — tri raster +
 *  MIP pyramid + block DAG), the ~36-48 s cold-boot slab. The result crosses
 *  back in the BootCache PACKED form (flat typed arrays, transferred zero-copy;
 *  unpackPreparedCrown on the caller side is the proven bit-exact path — the
 *  same one warm boots ride). cfg/occThreshold mirror the module knobs. */
export interface CrownReq {
  id: number;
  kind: 'crown';
  positions: Float32Array;
  normals: Float32Array;
  uvs?: Float32Array;
  vdata?: Uint32Array;
  indices: Uint32Array;
  color: { r: number; g: number; b: number; hueVar?: number };
  gridDim: number;
  voxlod: boolean;
  occThreshold: number;
  cfg: { levels: number; errorK: number; sparseK: number; shell: number; anchorL0: number };
}

/** seeded SDF-composed rock mesh bake (RockGen.generateRock — SPEC-ROCKS §G):
 *  deterministic params in, pure typed-array mesh out (transferred zero-copy).
 *  The SDF never leaves the generator; the result is ordinary mesh data. */
export interface RockReq {
  id: number;
  kind: 'rock';
  archetype: RockArchetype;
  variant: number;
  seed: number;
  gridRes: number;
  mod?: RockMod;
  domainScale: number;
}

export type DagReq = HeightDagReq | MeshDagReq | AggDagReq | CrownReq | RockReq;

/** the subset of HeightDagBuild the registry consumes (gridVerts in build grid
 *  coords 0..gridN — the caller remaps to texel coords); arrays are transferred */
export interface HeightDagOk {
  id: number;
  ok: true;
  kind: 'height';
  gridVerts: Uint32Array;
  indices: Uint32Array;
  clusters: DagCluster[];
  stats: HeightDagStats;
}

/** buildDag / buildAggregateDag result — DagBuild's typed arrays transfer
 *  zero-copy; clusters/groups/stats (plain-number structs) structured-clone. */
interface MeshDagOk {
  id: number;
  ok: true;
  kind: 'mesh' | 'aggregate';
  dag: DagBuild;
}

interface CrownOk {
  id: number;
  ok: true;
  kind: 'crown';
  pack: PackedPreparedCrown;
}

interface RockOk {
  id: number;
  ok: true;
  kind: 'rock';
  mesh: RockMesh;
}

interface DagErr {
  id: number;
  ok: false;
  error: string;
}

export type DagRes = HeightDagOk | MeshDagOk | CrownOk | RockOk | DagErr;
