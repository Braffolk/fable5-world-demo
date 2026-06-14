/**
 * Main-thread client for the off-thread DAG builder (N8-D1d, D-N30). Wraps the
 * Worker in a promise-per-request queue so callers `await` a build that runs on
 * another thread. One worker handles a sequence of builds (terrain + later the
 * explicit pools); kept alive for background (Increment 2) attaches. Inputs are
 * COPIED (structured-cloned), not transferred, so a worker failure can fall back
 * to a synchronous build on the caller side without a detached input buffer.
 */
import type { DagReq, DagRes, HeightDagOk } from './DagWorkerTypes';
import type { HeightDagOpts } from './BuildHeightDag';

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

/** an off-thread height-DAG builder — one Worker (DagBuildWorker) or a pool of them
 *  (DagWorkerPool). TileBuildDeps.worker is typed to this so either drops in. */
export interface DagBuilder {
  buildHeight(args: HeightDagArgs): Promise<HeightDagResult>;
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

  buildHeight(args: HeightDagArgs): Promise<HeightDagResult> {
    if (this.dead) return Promise.reject(new Error('DagWorker is dead'));
    const id = this.nextId++;
    const req: DagReq = {
      id,
      kind: 'height',
      heights: args.heights,
      gridN: args.gridN,
      cellSize: args.cellSize,
      originX: args.originX,
      originZ: args.originZ,
      opts: args.opts ?? {},
    };
    return new Promise<HeightDagResult>((resolve, reject) => {
      this.pending.set(id, {
        resolve: (r: DagRes) => {
          if (!r.ok) {
            reject(new Error(r.error));
            return;
          }
          resolve({ gridVerts: r.gridVerts, indices: r.indices, clusters: r.clusters, stats: r.stats });
        },
        reject,
      });
      // NOTE: no transfer list — copy the input so a sync fallback keeps it.
      this.worker.postMessage(req);
    });
  }

  dispose(): void {
    this.worker.terminate();
    this.dead = true;
    this.pending.clear();
  }
}

/**
 * A small pool of DagBuildWorkers for CONCURRENT tile bakes (N8-D2 #32). A single
 * persistent Worker bakes serially on its one thread (~170 ms/tile cache-miss), so a
 * camera move needing K fresh tiles stalls ~K×170 ms of coarse→fine pop. The pool
 * dispatches each buildHeight() to the LEAST-LOADED worker, so up to `size` tiles
 * bake in parallel on separate threads — shrinking the window ~size×. Same
 * buildHeight/dispose shape as DagBuildWorker (both are DagBuilders), so it drops
 * into TileBuildDeps.worker unchanged; the streamer fires several builds at once and
 * they land on distinct threads. Throws if NO worker can be constructed (headless
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

  buildHeight(args: HeightDagArgs): Promise<HeightDagResult> {
    // least-loaded dispatch: a fresh request goes to the most-idle thread rather
    // than queueing behind a slow bake on a round-robin victim.
    let pick = 0;
    for (let i = 1; i < this.inflight.length; i++) {
      if ((this.inflight[i] as number) < (this.inflight[pick] as number)) pick = i;
    }
    this.inflight[pick] = (this.inflight[pick] as number) + 1;
    const dec = (): void => {
      this.inflight[pick] = (this.inflight[pick] as number) - 1;
    };
    return (this.workers[pick] as DagBuildWorker).buildHeight(args).then(
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

  dispose(): void {
    for (const w of this.workers) w.dispose();
    this.workers = [];
  }
}
