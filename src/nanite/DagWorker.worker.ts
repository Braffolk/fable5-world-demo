/**
 * Off-thread DAG builder (N8-D1d, D-N30). Runs buildHeightDag (and later
 * buildDag for explicit pools) in a module Worker so the minutes-long terrain
 * build never blocks the boot critical path / render loop. The whole build
 * chain is three-free + typed-arrays in/out, so this bundle carries no GPU/DOM
 * code. Output arrays are transferred back zero-copy; cluster records (small
 * plain-number structs) ride the structured clone.
 */
/// <reference lib="webworker" />
import { buildHeightDag } from './BuildHeightDag';
import type { DagReq, DagRes } from './DagWorkerTypes';

// DOM and webworker libs both define `self`; the cast pins the worker scope so
// postMessage takes a transfer list (the DOM overload would take targetOrigin).
const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = (e: MessageEvent<DagReq>): void => {
  const req = e.data;
  try {
    if (req.kind === 'height') {
      const b = buildHeightDag(
        {
          heights: req.heights,
          gridN: req.gridN,
          cellSize: req.cellSize,
          originX: req.originX,
          originZ: req.originZ,
        },
        req.opts,
      );
      const res: DagRes = {
        id: req.id,
        ok: true,
        kind: 'height',
        gridVerts: b.gridVerts,
        indices: b.indices,
        clusters: b.clusters,
        stats: b.stats,
      };
      ctx.postMessage(res, [b.gridVerts.buffer, b.indices.buffer]);
      return;
    }
    const bad: DagRes = { id: (req as { id: number }).id, ok: false, error: `unknown kind` };
    ctx.postMessage(bad);
  } catch (err) {
    const res: DagRes = { id: req.id, ok: false, error: err instanceof Error ? err.message : String(err) };
    ctx.postMessage(res);
  }
};
