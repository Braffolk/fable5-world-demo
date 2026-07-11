/**
 * StreamBrain.worker — the streaming brain's host (SPEC-STREAMING-WORLD §4,
 * law 5: streaming machinery lives on WORKERS). Wires StreamBrainCore to
 * postMessage and gives it a NESTED DagWorkerPool for concurrent tile bakes
 * (bake threads under the brain, never on the main thread; if nested workers
 * are unavailable the core falls back to inline buildHeightGrid — still
 * off-main). Chunk fetches are an RPC back to the main thread while the
 * WorldSource is the generated one (it wraps GPU boot passes); Estonia's
 * RemoteWorldSource constructs here natively at S6 through the same seam.
 */

import { DagWorkerPool, type DagBuilder } from '../build/DagWorkerClient';
import type { ChunkKey, ChunkPayload, LayerName } from '../../world/source/WorldSource';
import { StreamBrainCore } from './StreamBrainCore';
import type { BrainToMain, MainToBrain } from './StreamProtocol';

const post = (msg: BrainToMain, transfer?: Transferable[]): void => {
  (self as unknown as Worker).postMessage(msg, transfer ?? []);
};

// nested bake pool — sized modestly (the brain thread itself stays responsive
// for fetch/scroll work while bakes run beside it)
let bake: DagBuilder | null = null;
try {
  const cores = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4;
  bake = new DagWorkerPool(Math.max(2, Math.min(4, cores - 2)));
} catch {
  bake = null;
  post({ kind: 'log', level: 'warn', msg: 'stream brain: nested bake workers unavailable — baking inline in the brain' });
}

// fetch RPC (main services it against the live WorldSource)
let nextFetchId = 1;
const pendingFetch = new Map<number, { resolve: (p: ChunkPayload | null) => void; reject: (e: Error) => void }>();

const core = new StreamBrainCore({
  fetch(layer: LayerName, key: ChunkKey): Promise<ChunkPayload | null> {
    const id = nextFetchId++;
    return new Promise((resolve, reject) => {
      pendingFetch.set(id, { resolve, reject });
      post({ kind: 'fetch', id, layer, key });
    });
  },
  emit: post,
  bake,
});

self.onmessage = (e: MessageEvent<MainToBrain>): void => {
  const msg = e.data;
  try {
    switch (msg.kind) {
      case 'init':
        core.init(msg);
        void core.fillPlanesBoot().catch((err: unknown) => {
          post({ kind: 'log', level: 'error', msg: `stream brain planes boot: ${err instanceof Error ? err.message : String(err)}` });
        });
        break;
      case 'fetchRes': {
        const p = pendingFetch.get(msg.id);
        if (!p) break;
        pendingFetch.delete(msg.id);
        if (msg.ok) p.resolve(msg.payload);
        else p.reject(new Error(msg.error ?? 'fetch failed'));
        break;
      }
      case 'pose':
        core.pose(msg.x, msg.z, msg.vx, msg.vz);
        break;
      case 'bootTiles':
        void core.bootTiles(msg.camX, msg.camZ).catch((err: unknown) => {
          post({ kind: 'log', level: 'error', msg: `stream brain boot tiles: ${err instanceof Error ? err.message : String(err)}` });
        });
        break;
      case 'poolInfo':
        core.poolInfo(msg);
        break;
      case 'ftArm':
        core.armFartiles(msg);
        break;
    }
  } catch (err) {
    post({ kind: 'log', level: 'error', msg: `stream brain: ${err instanceof Error ? err.message : String(err)}` });
  }
};
