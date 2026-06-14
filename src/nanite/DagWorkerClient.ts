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

export class DagBuildWorker {
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

  buildHeight(args: {
    heights: Float32Array;
    gridN: number;
    cellSize: number;
    originX: number;
    originZ: number;
    opts?: HeightDagOpts;
  }): Promise<HeightDagResult> {
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
