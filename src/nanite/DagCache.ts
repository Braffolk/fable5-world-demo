/**
 * Persistent build cache for the terrain LOD DAG (N8-D1d / D-N30). The full-res
 * 4096² build is minutes long but DETERMINISTIC (RNG-free, a pure function of the
 * heights + grid params), so it is built ONCE and stored in IndexedDB — every
 * subsequent boot loads it instantly and renders the DAG from the first frame
 * (NO window/default fallback, ever). The window path is only the separate
 * DAG-OFF mode, never a stand-in while a build runs.
 *
 * Key = seed + gridN + DAG_CACHE_VERSION. The heights are a deterministic
 * function of ?seed=N, so the seed identifies the field; bump DAG_CACHE_VERSION
 * whenever worldgen OR the DAG build algorithm changes (invalidates stale DAGs).
 * Cluster records are packed into one Float64Array (20 fields each — the subset
 * attachHeightDag consumes); gridVerts/indices ride as their own typed arrays.
 */
import type { DagCluster } from './BuildDag';
import type { HeightDagResult } from './DagWorkerClient';
import type { HeightDagStats } from './BuildHeightDag';

const DB_NAME = 'laas-dag-cache';
const STORE = 'heightdag';
const DB_VERSION = 1;

/** BUMP whenever worldgen or the DAG build changes → stale cached DAGs ignored. */
export const DAG_CACHE_VERSION = 1;

/** packed numeric fields per cluster (the subset attachHeightDag reads) */
const CF = 20;

interface CachedBlob {
  gridVerts: Uint32Array;
  indices: Uint32Array;
  clusterData: Float64Array;
  clusterCount: number;
  stats: HeightDagStats;
}

export function heightDagCacheKey(seed: number, gridN: number, suffix = ''): string {
  return `dag-terrain-v${DAG_CACHE_VERSION}-s${seed >>> 0}-g${gridN}${suffix}`;
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') {
      reject(new Error('indexedDB unavailable'));
      return;
    }
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (): void => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
    };
    req.onsuccess = (): void => resolve(req.result);
    req.onerror = (): void => reject(req.error ?? new Error('indexedDB open failed'));
  });
}

function serialize(r: HeightDagResult): CachedBlob {
  const n = r.clusters.length;
  const cd = new Float64Array(n * CF);
  for (let i = 0; i < n; i++) {
    const c = r.clusters[i] as DagCluster;
    const b = i * CF;
    cd[b] = c.sx;
    cd[b + 1] = c.sy;
    cd[b + 2] = c.sz;
    cd[b + 3] = c.sr;
    cd[b + 4] = c.cax;
    cd[b + 5] = c.cay;
    cd[b + 6] = c.caz;
    cd[b + 7] = c.ccos;
    cd[b + 8] = c.triStart;
    cd[b + 9] = c.triCount;
    cd[b + 10] = c.ownError;
    cd[b + 11] = c.oex;
    cd[b + 12] = c.oey;
    cd[b + 13] = c.oez;
    cd[b + 14] = c.oer;
    cd[b + 15] = c.parentError; // may be ±Infinity (root) — Float64Array preserves it
    cd[b + 16] = c.pex;
    cd[b + 17] = c.pey;
    cd[b + 18] = c.pez;
    cd[b + 19] = c.per;
  }
  return { gridVerts: r.gridVerts, indices: r.indices, clusterData: cd, clusterCount: n, stats: r.stats };
}

function deserialize(b: CachedBlob): HeightDagResult {
  const d = b.clusterData;
  const g = (k: number): number => d[k] as number;
  const clusters: DagCluster[] = new Array(b.clusterCount) as DagCluster[];
  for (let i = 0; i < b.clusterCount; i++) {
    const o = i * CF;
    clusters[i] = {
      level: 0,
      triStart: g(o + 8),
      triCount: g(o + 9),
      sx: g(o),
      sy: g(o + 1),
      sz: g(o + 2),
      sr: g(o + 3),
      cax: g(o + 4),
      cay: g(o + 5),
      caz: g(o + 6),
      ccos: g(o + 7),
      ownError: g(o + 10),
      oex: g(o + 11),
      oey: g(o + 12),
      oez: g(o + 13),
      oer: g(o + 14),
      parentError: g(o + 15),
      pex: g(o + 16),
      pey: g(o + 17),
      pez: g(o + 18),
      per: g(o + 19),
      groupAsInput: -1, // unused by attachHeightDag — recomputing the full DAG
      groupAsParent: -1, // linkage is not needed to render the cached cut
    };
  }
  return { gridVerts: b.gridVerts, indices: b.indices, clusters, stats: b.stats };
}

/** load a cached terrain DAG, or null on miss / any IndexedDB error (→ rebuild). */
export async function getCachedHeightDag(key: string): Promise<HeightDagResult | null> {
  let db: IDBDatabase;
  try {
    db = await openDb();
  } catch {
    return null;
  }
  try {
    const blob = await new Promise<CachedBlob | undefined>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readonly');
      const req = tx.objectStore(STORE).get(key);
      req.onsuccess = (): void => resolve(req.result as CachedBlob | undefined);
      req.onerror = (): void => reject(req.error ?? new Error('get failed'));
    });
    if (!blob || !blob.gridVerts || !blob.clusterData) return null;
    return deserialize(blob);
  } catch {
    return null;
  } finally {
    db.close();
  }
}

/** store a freshly built terrain DAG; best-effort (errors swallowed — the DAG is
 *  already attached, the cache only saves the NEXT boot's build). */
export async function putCachedHeightDag(key: string, r: HeightDagResult): Promise<void> {
  let db: IDBDatabase;
  try {
    db = await openDb();
  } catch {
    return;
  }
  try {
    const blob = serialize(r);
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).put(blob, key);
      tx.oncomplete = (): void => resolve();
      tx.onerror = (): void => reject(tx.error ?? new Error('put failed'));
    });
  } catch {
    /* best-effort */
  } finally {
    db.close();
  }
}
