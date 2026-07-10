/**
 * Main-thread client for the off-thread DAG builder (N8-D1d, D-N30; extended
 * 2026-07-04 with mesh/aggregate/crown jobs for the cold-boot arc). Wraps the
 * Worker in a promise-per-request queue so callers `await` a build that runs on
 * another thread. One worker handles a sequence of builds (terrain + the
 * vegetation pools); kept alive for background (Increment 2) attaches. Inputs are
 * COPIED (structured-cloned), not transferred, so a worker failure can fall back
 * to a synchronous build on the caller side without a detached input buffer.
 */
import type {
  AggDagReq,
  CrownReq,
  DagReq,
  DagRes,
  HeightDagOk,
  MeshDagReq,
  RockReq,
} from './DagWorkerTypes';
import type { HeightDagOpts } from './BuildHeightGrid';
import type { DagBuild } from './BuildDag';
import type { PackedPreparedCrown } from '../world/BootCache';
import type { RockMesh } from '../../vegetation/RockGen';

export interface HeightDagResult {
  gridVerts: HeightDagOk['gridVerts'];
  indices: HeightDagOk['indices'];
  clusters: HeightDagOk['clusters'];
  stats: HeightDagOk['stats'];
}

export interface HeightDagArgs {
  heights: Float32Array;
  gridN: number;
  cellSize: number;
  originX: number;
  originZ: number;
  opts?: HeightDagOpts;
}

/** buildDag job (workerized vegetation-head QEM DAG) — see MeshDagReq. */
export type MeshDagArgs = Omit<MeshDagReq, 'id' | 'kind'>;
/** buildAggregateDag job (workerized leaf-crown aggregate DAG) — see AggDagReq. */
export type AggDagArgs = Omit<AggDagReq, 'id' | 'kind'>;
/** prepareVoxelCrown job (workerized crown voxelization) — see CrownReq. */
export type CrownArgs = Omit<CrownReq, 'id' | 'kind'>;
/** generateRock job (workerized SDF rock mesh bake, SPEC-ROCKS §G) — see RockReq. */
export type RockArgs = Omit<RockReq, 'id' | 'kind'>;

/** an off-thread DAG builder — one Worker (DagBuildWorker) or a pool of them
 *  (DagWorkerPool). TileBuildDeps.worker is typed to this so either drops in. */
export interface DagBuilder {
  buildHeight(args: HeightDagArgs): Promise<HeightDagResult>;
  buildMesh(args: MeshDagArgs): Promise<DagBuild>;
  buildAggregate(args: AggDagArgs): Promise<DagBuild>;
  buildCrown(args: CrownArgs): Promise<PackedPreparedCrown>;
  buildRock(args: RockArgs): Promise<RockMesh>;
  dispose(): void;
}

export class DagBuildWorker implements DagBuilder {
  private readonly worker: Worker;
  private readonly pending = new Map<number, { resolve: (r: DagRes) => void; reject: (e: Error) => void }>();
  private nextId = 1;
  private dead = false;

  constructor() {
    this.worker = new Worker(new URL('./DagWorker.worker.ts', import.meta.url), {
      type: 'module',
      name: 'dag-build',
    });
    this.worker.onmessage = (e: MessageEvent<DagRes>): void => {
      const p = this.pending.get(e.data.id);
      if (!p) return;
      this.pending.delete(e.data.id);
      p.resolve(e.data);
    };
    this.worker.onerror = (e: ErrorEvent): void => {
      this.dead = true;
      const err = new Error(`DagWorker crashed: ${e.message}`);
      for (const p of this.pending.values()) p.reject(err);
      this.pending.clear();
    };
  }

  /** post one request; resolve with its (ok or error) response. NOTE: no
   *  transfer list — inputs are copied so a sync fallback keeps them. */
  private request(req: DagReq): Promise<DagRes> {
    if (this.dead) return Promise.reject(new Error('DagWorker is dead'));
    return new Promise<DagRes>((resolve, reject) => {
      this.pending.set(req.id, { resolve, reject });
      this.worker.postMessage(req);
    });
  }

  buildHeight(args: HeightDagArgs): Promise<HeightDagResult> {
    const req: DagReq = {
      id: this.nextId++,
      kind: 'height',
      heights: args.heights,
      gridN: args.gridN,
      cellSize: args.cellSize,
      originX: args.originX,
      originZ: args.originZ,
      opts: args.opts ?? {},
    };
    return this.request(req).then((r) => {
      if (!r.ok) throw new Error(r.error);
      if (r.kind !== 'height') throw new Error(`DagWorker: expected height, got ${r.kind}`);
      return { gridVerts: r.gridVerts, indices: r.indices, clusters: r.clusters, stats: r.stats };
    });
  }

  buildMesh(args: MeshDagArgs): Promise<DagBuild> {
    return this.request({ id: this.nextId++, kind: 'mesh', ...args }).then((r) => {
      if (!r.ok) throw new Error(r.error);
      if (r.kind !== 'mesh') throw new Error(`DagWorker: expected mesh, got ${r.kind}`);
      return r.dag;
    });
  }

  buildAggregate(args: AggDagArgs): Promise<DagBuild> {
    return this.request({ id: this.nextId++, kind: 'aggregate', ...args }).then((r) => {
      if (!r.ok) throw new Error(r.error);
      if (r.kind !== 'aggregate') throw new Error(`DagWorker: expected aggregate, got ${r.kind}`);
      return r.dag;
    });
  }

  buildCrown(args: CrownArgs): Promise<PackedPreparedCrown> {
    return this.request({ id: this.nextId++, kind: 'crown', ...args }).then((r) => {
      if (!r.ok) throw new Error(r.error);
      if (r.kind !== 'crown') throw new Error(`DagWorker: expected crown, got ${r.kind}`);
      return r.pack;
    });
  }

  buildRock(args: RockArgs): Promise<RockMesh> {
    return this.request({ id: this.nextId++, kind: 'rock', ...args }).then((r) => {
      if (!r.ok) throw new Error(r.error);
      if (r.kind !== 'rock') throw new Error(`DagWorker: expected rock, got ${r.kind}`);
      return r.mesh;
    });
  }

  dispose(): void {
    this.worker.terminate();
    this.dead = true;
    this.pending.clear();
  }
}

/**
 * A small pool of DagBuildWorkers for CONCURRENT builds (N8-D2 #32; cold-boot
 * fan-out 2026-07-04). A single persistent Worker builds serially on its one
 * thread, so a batch of K jobs stalls ~K× the per-job time. The pool dispatches
 * each job to the LEAST-LOADED worker, so up to `size` jobs run in parallel on
 * separate threads. Same DagBuilder shape as DagBuildWorker, so it drops into
 * TileBuildDeps.worker unchanged; callers fire several builds at once and they
 * land on distinct threads. Throws if NO worker can be constructed (headless
 * node has no Worker) so the caller can fall back to synchronous builds.
 */
export class DagWorkerPool implements DagBuilder {
  private workers: DagBuildWorker[] = [];
  /** in-flight build count per worker — least-loaded dispatch reads/decrements it. */
  private readonly inflight: number[];

  constructor(size: number) {
    const n = Math.max(1, Math.floor(size));
    try {
      for (let i = 0; i < n; i++) this.workers.push(new DagBuildWorker());
    } catch (e) {
      // partial construction (e.g. Worker unavailable mid-loop) — tear down the ones
      // that succeeded and rethrow so the caller falls back to sync builds.
      for (const w of this.workers) w.dispose();
      this.workers = [];
      throw e;
    }
    this.inflight = new Array(this.workers.length).fill(0);
  }

  get size(): number {
    return this.workers.length;
  }

  /** least-loaded dispatch: a fresh request goes to the most-idle thread rather
   *  than queueing behind a slow build on a round-robin victim. */
  private run<T>(f: (w: DagBuildWorker) => Promise<T>): Promise<T> {
    let pick = 0;
    for (let i = 1; i < this.inflight.length; i++) {
      if ((this.inflight[i] as number) < (this.inflight[pick] as number)) pick = i;
    }
    this.inflight[pick] = (this.inflight[pick] as number) + 1;
    const dec = (): void => {
      this.inflight[pick] = (this.inflight[pick] as number) - 1;
    };
    return f(this.workers[pick] as DagBuildWorker).then(
      (r) => {
        dec();
        return r;
      },
      (e) => {
        dec();
        throw e instanceof Error ? e : new Error(String(e));
      },
    );
  }

  buildHeight(args: HeightDagArgs): Promise<HeightDagResult> {
    return this.run((w) => w.buildHeight(args));
  }

  buildMesh(args: MeshDagArgs): Promise<DagBuild> {
    return this.run((w) => w.buildMesh(args));
  }

  buildAggregate(args: AggDagArgs): Promise<DagBuild> {
    return this.run((w) => w.buildAggregate(args));
  }

  buildCrown(args: CrownArgs): Promise<PackedPreparedCrown> {
    return this.run((w) => w.buildCrown(args));
  }

  buildRock(args: RockArgs): Promise<RockMesh> {
    return this.run((w) => w.buildRock(args));
  }

  dispose(): void {
    for (const w of this.workers) w.dispose();
    this.workers = [];
  }
}
