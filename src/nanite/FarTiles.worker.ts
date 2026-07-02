/**
 * FarTiles splat worker (module Worker) — runs FarTilesSplat.splatTiles for an
 * assigned slice of tiles. THREE-FREE (imports only the pure splat core), same
 * discipline as DagWorker.worker.ts. Inputs are structured-cloned in (pools
 * stay usable on the main thread); each tile's 9 result arrays are TRANSFERRED
 * back zero-copy — ONE MESSAGE PER TILE (2026-07-03), so neither the worker nor
 * the main thread ever holds the whole slice's dense grids (the world-scene OOM;
 * the main thread pyramids + releases each tile as it lands).
 */
/// <reference lib="webworker" />
import { splatTiles, type SplatGridSpec, type SplatPoolFlat, type TileSplatJob, type TileSplatOut } from './FarTilesSplat';

export interface FtReq {
  id: number;
  grid: SplatGridSpec;
  pools: SplatPoolFlat[];
  jobs: TileSplatJob[];
}

export type FtRes =
  | { id: number; ok: true; done?: false; i: number; tile: TileSplatOut }
  | { id: number; ok: true; done: true }
  | { id: number; ok: false; error: string };

const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = (e: MessageEvent<FtReq>): void => {
  const req = e.data;
  try {
    for (let i = 0; i < req.jobs.length; i++) {
      const tiles = splatTiles(req.grid, req.pools, [req.jobs[i] as TileSplatJob]);
      const t = tiles[0] as TileSplatOut;
      const res: FtRes = { id: req.id, ok: true, i, tile: t };
      ctx.postMessage(res, [
        t.occLo.buffer,
        t.occHi.buffer,
        t.accW.buffer,
        t.accR.buffer,
        t.accG.buffer,
        t.accB.buffer,
        t.accNX.buffer,
        t.accNY.buffer,
        t.accNZ.buffer,
      ]);
    }
    ctx.postMessage({ id: req.id, ok: true, done: true } satisfies FtRes);
  } catch (err) {
    const res: FtRes = { id: req.id, ok: false, error: err instanceof Error ? err.message : String(err) };
    ctx.postMessage(res);
  }
};
