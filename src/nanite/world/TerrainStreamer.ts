/**
 * buildTerrainTile — the main-thread terrain-tile DAG bake over an in-RAM
 * height field (subsample → DAG build via worker/sync → remap to GLOBAL texel
 * coords). Since S5 this serves ONLY the explicit `?nanitedterrain=<gridN>`
 * uniform T×T tooling path (probe-dterrain and friends), which bakes from
 * hf.cpuHeights at boot and never streams.
 *
 * THE terrain streaming path — the camera-following clipmap over the tile
 * pool — lives in the StreamBrain worker (StreamBrainCore.bakeTile bakes from
 * the brain's decoded-chunk height windows and attaches per-tile origin
 * words); the old main-thread TerrainStreamer class it replaced is excised
 * (SPEC-STREAMING-WORLD §4/S5, worker law).
 */
import { buildHeightGrid, type HeightDagOpts } from '../build/BuildHeightGrid';
import { getCachedHeightDag, heightDagCacheKey, putCachedHeightDag } from '../build/DagCache';
import { type DagBuilder, type HeightDagResult } from '../build/DagWorkerClient';

/** everything buildTerrainTile needs that is constant for the field's lifetime. */
export interface TileBuildDeps {
  heights: Float32Array;
  res: number;
  cell: number;
  origin: number;
  gridN: number;
  /** numeric seed for the per-tile DAG cache, or null to always build (no cache). */
  seed: number | null;
  /** persistent off-thread builder — a single Worker or a DagWorkerPool (concurrent
   *  bakes) — or null to build synchronously on this thread. */
  worker: DagBuilder | null;
}

/** mutable cache-hit / fresh-build tally threaded through a build pass. */
export interface TileBuildStats {
  nCache: number;
  nBuilt: number;
}

function clampTexel(t: number, res: number): number {
  return t < 0 ? 0 : t > res - 1 ? res - 1 : t;
}

/**
 * Build ONE terrain tile DAG at texel origin (tx0,tz0), `tileStride` texels/cell,
 * gridN cells — subsample the field (cache-aware), build the DAG (worker or sync),
 * and remap the tile-local grid coords to GLOBAL texel coords (so the GPU reads
 * the full-res height lattice). Off-field samples clamp to the field edge.
 */
export async function buildTerrainTile(
  deps: TileBuildDeps,
  tx0: number,
  tz0: number,
  tileStride: number,
  suffix: string,
  stats: TileBuildStats,
  onDeferred: (msg: string) => void,
  skirtLevel: number,
): Promise<{ gridVerts: Uint32Array; built: HeightDagResult }> {
  const { heights, res, cell, origin, gridN, seed, worker } = deps;
  const vpa = gridN + 1;
  // N8-D2 Stage 2d: ≥0 appends a perimeter skirt at that clipmap level. Skirts change
  // the geometry ⇒ a distinct cache key (else a no-skirt build aliases a skirt one).
  const opts: HeightDagOpts = skirtLevel >= 0 ? { skirtLevel } : {};
  const cacheSuffix = skirtLevel >= 0 ? `${suffix}-sk${skirtLevel}` : suffix;
  const cacheKey = seed != null ? heightDagCacheKey(seed, gridN, cacheSuffix) : null;
  let built: HeightDagResult | null = cacheKey ? await getCachedHeightDag(cacheKey) : null;
  if (built) {
    stats.nCache++;
  } else {
    const sub = new Float32Array(vpa * vpa);
    for (let gz = 0; gz <= gridN; gz++) {
      const texZ = clampTexel(tz0 + gz * tileStride, res);
      const trow = texZ * res;
      const srow = gz * vpa;
      for (let gx = 0; gx <= gridN; gx++) {
        const texX = clampTexel(tx0 + gx * tileStride, res);
        sub[srow + gx] = heights[trow + texX] as number;
      }
    }
    const hfArgs = {
      heights: sub,
      gridN,
      cellSize: cell * tileStride,
      originX: origin + tx0 * cell,
      originZ: origin + tz0 * cell,
    };
    if (worker) {
      try {
        built = await worker.buildHeight({ ...hfArgs, opts });
      } catch (e) {
        onDeferred(`terrain DAG tile ${suffix}: worker failed (${e instanceof Error ? e.message : String(e)}) → sync`);
        built = buildHeightGrid(hfArgs, opts);
      }
    } else {
      built = buildHeightGrid(hfArgs, opts);
    }
    if (cacheKey) void putCachedHeightDag(cacheKey, built); // fire-and-forget
    stats.nBuilt++;
  }
  // remap tile-local grid coords to GLOBAL texel coords. word0 = gx(0-12) | skirt
  // code(13-15) | gz(16-31); the 3-bit code is PRESERVED (a surface vert's code is 0,
  // so this is identical to the old gx&0xffff for non-skirt verts).
  const gridVerts = new Uint32Array(built.gridVerts.length);
  for (let i = 0; i < built.gridVerts.length; i++) {
    const p = built.gridVerts[i] as number;
    const code = (p >>> 13) & 0x7;
    const texX = clampTexel(tx0 + (p & 0x1fff) * tileStride, res);
    const texZ = clampTexel(tz0 + ((p >>> 16) & 0xffff) * tileStride, res);
    gridVerts[i] = ((texX & 0x1fff) | (code << 13) | ((texZ & 0xffff) << 16)) >>> 0;
  }
  return { gridVerts, built };
}
