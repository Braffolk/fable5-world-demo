/**
 * FarTiles — cross-instance FAR-FIELD AGGREGATION (the UE5-HLOD-equivalent move,
 * 2026-07-02 wave 3, ?fartiles=1): beyond the aggregation distance, whole 64 m
 * tiles of trees render as ONE merged voxel head — the far cluster/brick count
 * collapses by orders of magnitude and the forest extends to the horizon.
 *
 * S8 (SPEC §4, the A3 named refactor): the pure pipeline — plan → splat →
 * dense-emit → pyramid → pack — lives in FarTilesCore.ts (THREE-FREE) +
 * FarTilesSplat.ts (the pure per-tile splat, worker-fanned), so it can run
 * inside the StreamBrain worker. THIS file keeps the MAIN-THREAD wrappers for
 * ForestScene's boot-time whole-map build:
 *  - buildFarTilesAsync — fans the splat across a FarTiles.worker pool, emits
 *    tile-by-tile (dense arrays released per message — the old all-at-the-end
 *    transfer OOM'd), deterministic job order, sync fallback;
 *  - buildFarTiles — the single-thread reference + worker-failure fallback;
 *  - appendFarTiles / appendPackedFarTiles — post-reg.build() registry appends
 *    (one voxel head + one identity instance per tile).
 * The scene=world path streams fartiles at runtime through the StreamBrain
 * instead (S8) and does not import this file.
 */

import type { GeometryRegistry } from './GeometryRegistry';
import { appendPackedCrown, appendVoxelCrown } from '../build/VoxelizeCrown';
import {
  emitTile,
  planFarTiles,
  type FarTileBuild,
  type FarTileOpts,
  type PackedFarTile,
} from './FarTilesCore';
import { splatTiles, type TileSplatJob } from './FarTilesSplat';

export { FT_TILE_SIZE, DEFAULT_FT_CELL, DEFAULT_AGG_DIST } from './FarTilesCore';
export type { FarTileBuild, FarTileOpts, FarTileSpecies } from './FarTilesCore';

/** Splat all instances into tile grids and build one voxel pyramid per tile (SYNC —
 *  the single-thread reference + worker-failure fallback). STREAMS one tile at a time
 *  (splat → pyramid → release the dense arrays) so peak memory is one tile's working
 *  set, not the whole map's (the world-scene OOM, 2026-07-03). `map` (optional)
 *  transforms each build AS IT EMITS — pass a packer to keep only the compact form
 *  in memory (the accumulated BrickCPU objects for a whole map are heap-fatal). */
function buildFarTiles<T = FarTileBuild>(
  opts: FarTileOpts,
  map?: (b: FarTileBuild, jobIndex: number) => T,
): T[] {
  const plan = planFarTiles(opts);
  if (!plan) return [];
  const out: T[] = [];
  for (let j = 0; j < plan.jobs.length; j++) {
    const results = splatTiles(plan.grid, plan.poolsFlat, [plan.jobs[j] as TileSplatJob]);
    const r = results[0];
    if (!r) continue;
    const t = emitTile(plan.grid, r);
    if (t) out.push(map ? map(t, j) : (t as unknown as T));
  }
  return out;
}

/** Worker-pool splat (wave 4): fans the per-tile splat across module Workers; the
 *  dense-emit + pyramid runs on the main thread AS EACH TILE ARRIVES (one message per
 *  tile since 2026-07-03 — the dense accumulator arrays are released tile-by-tile, so
 *  peak memory is the in-flight tiles, not the whole map; the old all-at-the-end
 *  transfer OOM'd the world scene). Deterministic: builds land in job order regardless
 *  of worker completion order. Falls back to buildFarTiles on any worker error. */
export async function buildFarTilesAsync<T = FarTileBuild>(
  opts: FarTileOpts,
  map?: (b: FarTileBuild, jobIndex: number) => T,
): Promise<T[]> {
  const plan = planFarTiles(opts);
  if (!plan) return [];
  const t0 = performance.now();
  const hw = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 4 : 4;
  const workerCount = Math.max(1, Math.min(8, hw - 2, plan.jobs.length));
  const builds = new Array<T | null>(plan.jobs.length).fill(null);
  let emitMs = 0;
  let emitted = 0;
  let runningBricks = 0;
  // chunk jobs round-robin-contiguous: worker w takes jobs [w·per, (w+1)·per)
  const per = Math.ceil(plan.jobs.length / workerCount);
  try {
    await Promise.all(
      Array.from({ length: workerCount }, (_, w) => {
        const jobs = plan.jobs.slice(w * per, (w + 1) * per);
        if (jobs.length === 0) return Promise.resolve();
        return new Promise<void>((resolve, reject) => {
          const worker = new Worker(new URL('./FarTiles.worker.ts', import.meta.url), {
            type: 'module',
            name: `fartiles-${w}`,
          });
          worker.onmessage = (e: MessageEvent<import('./FarTiles.worker').FtRes>): void => {
            const res = e.data;
            if (!res.ok) {
              worker.terminate();
              reject(new Error(res.error));
              return;
            }
            if (res.done) {
              worker.terminate();
              resolve();
              return;
            }
            // one tile — pyramid it now, release the dense arrays with this message;
            // `map` (e.g. the bootcache packer) immediately compacts the build so the
            // per-brick JS objects never accumulate across the whole map (heap-fatal).
            const tE0 = performance.now();
            const jobIndex = w * per + res.i;
            const built = emitTile(plan.grid, res.tile);
            builds[jobIndex] = built ? (map ? map(built, jobIndex) : (built as unknown as T)) : null;
            emitMs += performance.now() - tE0;
            emitted++;
            if (built) runningBricks += built.prep.brickCount;
            if (emitted % 256 === 0) {
              // eslint-disable-next-line no-console
              console.log(
                `[fartiles] progress ${emitted}/${plan.jobs.length} tiles, ${runningBricks} bricks, ` +
                  `${((performance.now() - t0) / 1000).toFixed(0)} s (emit ${(emitMs / 1000).toFixed(1)} s)`,
              );
            }
          };
          worker.onerror = (e: ErrorEvent): void => {
            worker.terminate();
            reject(new Error(`FarTiles worker crashed: ${e.message}`));
          };
          worker.postMessage({ id: w, grid: plan.grid, pools: plan.poolsFlat, jobs });
        });
      }),
    );
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn('[fartiles] worker splat failed — falling back to sync build:', err);
    return buildFarTiles(opts, map);
  }
  const out: T[] = [];
  for (const b of builds) {
    if (b) out.push(b);
  }
  // eslint-disable-next-line no-console
  console.log(
    `[fartiles] worker splat ${(performance.now() - t0).toFixed(0)} ms (${workerCount} workers, ` +
      `${plan.jobs.length} tiles, interleaved emit/pyramid ${emitMs.toFixed(0)} ms)`,
  );
  return out;
}

/** Append built tiles post-reg.build(): one voxel head + one identity instance each. */
export function appendFarTiles(
  reg: GeometryRegistry,
  tiles: FarTileBuild[],
  opts: { nearDist: number; matParam: number },
): number {
  let bricks = 0;
  for (const t of tiles) {
    const r = appendVoxelCrown(reg, t.prep, undefined as never, {
      matParam: opts.matParam,
      swayPad: 0,
      maxDist: 100000,
      nearDist: opts.nearDist,
      label: 'fartile',
    });
    bricks += r.brickCount;
    const a = new Float32Array([t.center[0], t.center[1], t.center[2], 1]);
    const b = new Float32Array([0, 0, 0, 0]);
    reg.bindInstances(r.head, { a, b });
  }
  return bricks;
}

/** Append PACKED tiles post-reg.build() — reads straight from each tile's 9×u32 words
 *  (appendPackedCrown), no BrickCPU materialization; `tick` yields between tiles
 *  (thousands of tiles = a long slab otherwise). Identical GPU output to appendFarTiles. */
export async function appendPackedFarTiles(
  reg: GeometryRegistry,
  tiles: PackedFarTile[],
  opts: { nearDist: number; matParam: number },
  tick?: () => Promise<void>,
): Promise<number> {
  let bricks = 0;
  for (const t of tiles) {
    if (tick) await tick();
    const r = appendPackedCrown(reg, t.prep, {
      matParam: opts.matParam,
      swayPad: 0,
      maxDist: 100000,
      nearDist: opts.nearDist,
      label: 'fartile',
    });
    bricks += r.brickCount;
    const a = new Float32Array([t.center[0], t.center[1], t.center[2], 1]);
    const b = new Float32Array([0, 0, 0, 0]);
    reg.bindInstances(r.head, { a, b });
  }
  return bricks;
}
