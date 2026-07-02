/**
 * FarTiles splat worker (module Worker) — runs FarTilesSplat.splatTiles for an
 * assigned slice of tiles. THREE-FREE (imports only the pure splat core), same
 * discipline as DagWorker.worker.ts. Inputs are structured-cloned in (pools
 * stay usable on the main thread); each tile's 9 result arrays are TRANSFERRED
 * back zero-copy.
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
  | { id: number; ok: true; tiles: TileSplatOut[] }
  | { id: number; ok: false; error: string };

const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = (e: MessageEvent<FtReq>): void => {
  const req = e.data;
  try {
    const tiles = splatTiles(req.grid, req.pools, req.jobs);
    const res: FtRes = { id: req.id, ok: true, tiles };
    const transfer: Transferable[] = [];
    for (const t of tiles) {
      transfer.push(
        t.occLo.buffer,
        t.occHi.buffer,
        t.accW.buffer,
        t.accR.buffer,
        t.accG.buffer,
        t.accB.buffer,
        t.accNX.buffer,
        t.accNY.buffer,
        t.accNZ.buffer,
      );
    }
    ctx.postMessage(res, transfer);
  } catch (err) {
    const res: FtRes = { id: req.id, ok: false, error: err instanceof Error ? err.message : String(err) };
    ctx.postMessage(res);
  }
};
