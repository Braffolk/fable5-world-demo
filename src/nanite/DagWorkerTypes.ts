/**
 * Message protocol for the off-thread DAG builder (N8-D1d, D-N30). The build
 * chain (buildDag → clusterize, buildHeightDag → buildDag) is three-free and
 * typed-arrays in/out, so it runs in a plain module Worker; the heavy CPU work
 * (terrain 4096² ≈ minutes) stays off the boot critical path. Kept in its own
 * file so the Worker and its main-thread client share one source of truth.
 */
import type { DagCluster } from './BuildDag';
import type { HeightDagOpts, HeightDagStats } from './BuildHeightDag';

/** build an adaptive terrain LOD DAG on a (gridN+1)² heightfield (gridN = 2^k) */
export interface HeightDagReq {
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

export type DagReq = HeightDagReq;

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

export interface DagErr {
  id: number;
  ok: false;
  error: string;
}

export type DagRes = HeightDagOk | DagErr;
