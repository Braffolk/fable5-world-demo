/**
 * RemoteWorldSource — WorldSource over the cooked LAC1 release tree (S3/CDN layout,
 * asset-gen/src/assetgen/manifest.py): latest.json (cache:'no-cache') → immutable
 * m/<hash>/manifest.json → per-layer binary indexes into Map<packed(lod,cx,cz), ChunkRef>.
 * fetch() posts to a small pool of Lac1Decode workers (HTTP GET + inflate + CRC + decode
 * off the main thread; typed arrays transferred back). coverage()-absent chunks resolve
 * null without any request — index absence is authoritative (dry water, empty boulders,
 * outside-Estonia are one mechanism).
 */
import { chunkPath, LAC1_LAYER_IDS, packChunkKey, parseLayerIndex } from './Lac1';
import { packChunkKeyV2, parseLayerIndexV2 } from './Lac2';
import type { Lac1DecodeJob, Lac1DecodeReq, Lac1DecodeRes } from './Lac1Decode.worker';
import type {
  ChunkKey,
  ChunkPayload,
  ChunkRef,
  CommunityEntry,
  LayerName,
  LacContainer,
  ManifestFormat,
  SpeciesEntry,
  WorldDictionaries,
  WorldLayerMeta,
  WorldManifest,
  WorldSource,
  WireCodec,
} from './WorldSource';

/** braffolk.com serves the synced bucket at `laas-data/` (live-verified: CloudFront,
 *  CORS *); dev passes a localhost static server over asset-gen/data/out instead —
 *  the CDN caches latest.json longer than local recooks. */
const DEFAULT_BASE_URL = 'https://braffolk.com/laas-data';
const DECODE_WORKERS = 2;
/** Max concurrent chunk fetches in flight across ALL consumers (brain window
 *  scrolls + tile sources + the instance band). A cut swing or a StreamOrigin
 *  rebase re-wants a whole new chunk set at once; without a bound the burst
 *  stampedes the HTTP/1.1 per-host connection pool (~6) and fetches start failing
 *  with `Failed to fetch`. Queueing behind a small semaphore turns the burst into
 *  an orderly drain — the demand pool IS the queue, the pose tick's re-nomination
 *  IS the retry, so there is no retry storm. ≤ the browser's per-host limit. */
const MAX_INFLIGHT_FETCHES = 6;

/** The manifest.json fields this client consumes (manifest.py build_release). */
interface ManifestLayerJson {
  enc: number;
  lods: number[];
  count: number;
  bytes: number;
  index: string;
  texelMeters?: number;
  planes?: string[];
  columns?: [string, string][];
  baseTexelMeters?: number;
  finestLod?: number;
  authorityLod?: number;
  synthesis?: string;
}

interface ManifestJsonShared {
  format: ManifestFormat;
  codec: string;
  containers?: string[];
  speciesMap?: Record<string, SpeciesEntry>;
  understoryMap?: Record<string, CommunityEntry>;
  debrisMap?: Record<string, CommunityEntry>;
  layers: Record<string, ManifestLayerJson>;
}

interface ManifestJsonV1 extends ManifestJsonShared {
  format: 1;
  anchor: { e: number; n: number };
  chunkMeters: number;
  chunkRes: number;
  lodStep: number;
}

interface ManifestJsonV2 extends ManifestJsonShared {
  format: 2;
  containers: string[];
  grid: { anchorE: number; anchorN: number; chunkMeters: number; chunkRes: number; lodStep: number };
}

type ManifestJson = ManifestJsonV1 | ManifestJsonV2;

export class RemoteWorldSource implements WorldSource {
  private readonly baseUrl: string;
  private readonly indexes = new Map<LayerName, Map<number, ChunkRef>>();
  private layers: Partial<Record<LayerName, WorldLayerMeta>> = {};
  private pool: DecodePool | null = null;
  private format: ManifestFormat = 1;
  private containers: readonly LacContainer[] = ['LAC1'];
  private codec: WireCodec = 'deflate';
  private packKey: (lod: number, cx: number, cz: number) => number = packChunkKey;
  private manifestGrid: { anchorE: number; anchorN: number; chunkMeters: number; chunkRes: number; lodStep: number } | null = null;
  /** burst-shaping semaphore — bounds concurrent HTTP fetches (see MAX_INFLIGHT_FETCHES) */
  private readonly gate = new FetchGate(MAX_INFLIGHT_FETCHES);

  constructor(baseUrl: string = DEFAULT_BASE_URL) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
  }

  async open(progress?: (frac: number, msg: string) => void): Promise<WorldManifest> {
    this.close();
    const latest = (await getJson(`${this.baseUrl}/latest.json`, 'no-cache')) as { manifest: string };
    progress?.(0, `manifest ${latest.manifest}`);
    const m = (await getJson(`${this.baseUrl}/${latest.manifest}`, 'default')) as ManifestJson;
    if (m.format !== 1 && m.format !== 2) throw new Error(`RemoteWorldSource: unsupported manifest format ${(m as { format: number }).format}`);
    if (m.codec !== 'deflate') throw new Error(`RemoteWorldSource: unsupported codec ${m.codec}`);
    if (m.format === 1) {
      if (m.containers !== undefined && (m.containers.length !== 1 || m.containers[0] !== 'LAC1')) {
        throw new Error('RemoteWorldSource: format 1 may declare only LAC1');
      }
      this.format = 1;
      this.containers = ['LAC1'];
      this.packKey = packChunkKey;
    } else {
      if (m.containers.length !== 2 || m.containers[0] !== 'LAC1' || m.containers[1] !== 'LAC2') {
        throw new Error('RemoteWorldSource: format 2 must declare [LAC1,LAC2]');
      }
      this.format = 2;
      this.containers = ['LAC1', 'LAC2'];
      this.packKey = packChunkKeyV2;
    }
    this.codec = 'deflate';
    const grid = m.format === 1
      ? { anchorE: m.anchor.e, anchorN: m.anchor.n, chunkMeters: m.chunkMeters, chunkRes: m.chunkRes, lodStep: m.lodStep }
      : m.grid;
    if (![grid.anchorE, grid.anchorN].every(Number.isFinite)
      || ![grid.chunkMeters, grid.chunkRes, grid.lodStep].every(Number.isInteger)
      || grid.chunkMeters <= 0 || grid.chunkRes <= 0 || grid.lodStep <= 1) {
      throw new Error('RemoteWorldSource: invalid manifest grid');
    }
    if (m.format === 2 && (grid.chunkMeters !== 2048 || grid.chunkRes !== 2048 || grid.lodStep !== 4)) {
      throw new Error('RemoteWorldSource: format-2 micro v1 requires the frozen (2048,2048,4) grid');
    }
    this.manifestGrid = grid;
    const manifestDir = latest.manifest.slice(0, latest.manifest.lastIndexOf('/'));

    const entries = Object.entries(m.layers);
    for (let li = 0; li < entries.length; li++) {
      const [name, meta] = entries[li] as (typeof entries)[number];
      if (!(name in LAC1_LAYER_IDS)) continue; // future layers (e.g. canopy) until the codec knows them
      const layer = name as LayerName;
      if (
        !Number.isInteger(meta.enc)
        || !meta.lods.every(Number.isInteger)
        || meta.lods.some((lod, i) => i > 0 && lod <= (meta.lods[i - 1] as number))
      ) {
        throw new Error(`RemoteWorldSource: ${layer} has invalid encoding/LOD metadata`);
      }
      if (m.format === 2) {
        if (layer !== 'height' && meta.lods.some((lod) => lod < 0)) {
          throw new Error(`RemoteWorldSource: negative LOD is height-only (${layer})`);
        }
        if (layer === 'height' && (
          meta.baseTexelMeters !== 1
          || meta.finestLod !== Math.min(...meta.lods)
          || meta.authorityLod !== 0
          || meta.lods.some((lod) => lod < -2 || lod > 4)
        )) {
          throw new Error('RemoteWorldSource: invalid format-2 height geometry metadata');
        }
      }
      const bin = await getBytes(`${this.baseUrl}/${manifestDir}/${meta.index}`);
      const map = new Map<number, ChunkRef>();
      const refs = m.format === 1 ? parseLayerIndex(bin) : parseLayerIndexV2(bin);
      for (const ref of refs) {
        const packed = this.packKey(ref.lod, ref.cx, ref.cz);
        if (map.has(packed)) throw new Error(`RemoteWorldSource: duplicate ${layer} chunk key`);
        map.set(packed, ref);
      }
      if (map.size !== meta.count) throw new Error(`RemoteWorldSource: ${layer} index has ${map.size} != ${meta.count} chunks`);
      const indexedLods = [...new Set(refs.map((ref) => ref.lod))].sort((a, b) => a - b);
      if (indexedLods.length !== meta.lods.length || indexedLods.some((lod, i) => lod !== meta.lods[i])) {
        throw new Error(`RemoteWorldSource: ${layer} index LODs differ from manifest`);
      }
      this.indexes.set(layer, map);
      this.layers[layer] = {
        enc: meta.enc,
        lods: meta.lods,
        chunkCount: meta.count,
        // the base raster (height) omits texelMeters — it IS the grid's finest
        // lattice (chunkMeters/chunkRes); coarser rasters (biome/water/…) declare
        // their own (e.g. water = 2 m). Fill the implied default so every raster
        // layer carries an explicit texel for PlaneFill.layerGeom.
        texelMeters: meta.texelMeters ?? (meta.enc !== 3 ? grid.chunkMeters / grid.chunkRes : undefined),
        planes: meta.planes,
        columns: meta.columns as WorldLayerMeta['columns'],
        baseTexelMeters: meta.baseTexelMeters,
        finestLod: meta.finestLod,
        authorityLod: meta.authorityLod,
        synthesis: meta.synthesis,
      };
      progress?.((li + 1) / entries.length, `index ${layer}: ${map.size} chunks`);
    }

    this.pool = new DecodePool(DECODE_WORKERS);
    const indexes = this.indexes;
    const packKey = this.packKey;
    return {
      format: this.format,
      containers: this.containers,
      codec: this.codec,
      // Estonia game coords are anchored to the manifest anchor, so chunks tile from
      // game (0,0): originX/Z = 0.
      grid: { ...grid, originX: 0, originZ: 0 },
      layers: this.layers,
      dictionaries: parseDictionaries(m),
      coverage(layer: LayerName, key: ChunkKey): ChunkRef | null {
        return indexes.get(layer)?.get(packKey(key.lod, key.cx, key.cz)) ?? null;
      },
      chunks(layer: LayerName, lod: number): ChunkKey[] {
        const out: ChunkKey[] = [];
        for (const ref of indexes.get(layer)?.values() ?? []) {
          if (ref.lod === lod) out.push({ lod: ref.lod, cx: ref.cx, cz: ref.cz });
        }
        return out;
      },
    };
  }

  fetch(layer: LayerName, key: ChunkKey, signal?: AbortSignal): Promise<ChunkPayload | null> {
    const ref = this.indexes.get(layer)?.get(this.packKey(key.lod, key.cx, key.cz));
    if (!ref) return Promise.resolve(null); // authoritative absence — no request
    const pool = this.pool;
    const meta = this.layers[layer];
    const grid = this.manifestGrid;
    if (!pool || !meta || !grid) return Promise.reject(new Error('RemoteWorldSource: fetch before open()'));
    // burst-shaping: acquire an in-flight slot before dispatching (queued if the
    // pool is saturated); a failed/aborted fetch just releases its slot and returns
    // to the demand pool — the pose tick re-nominates it (no retry storm).
    return this.gate.run(() =>
      pool.decode(
        {
          url: `${this.baseUrl}/${chunkPath(layer, ref)}`,
          layerId: LAC1_LAYER_IDS[layer as keyof typeof LAC1_LAYER_IDS],
          lod: key.lod,
          cx: key.cx,
          cz: key.cz,
          expectedEnc: meta.enc,
          allowedContainers: this.containers,
          codec: this.codec,
          expectedFileSize: ref.size,
          expectedRes: meta.enc === 3
            ? 0
            : Math.round(grid.chunkMeters / (meta.texelMeters ?? 1)) + 1,
          expectedOriginE: grid.anchorE + key.cx * grid.chunkMeters * grid.lodStep ** key.lod,
          expectedOriginN: grid.anchorN - key.cz * grid.chunkMeters * grid.lodStep ** key.lod,
          schema: { planes: meta.planes?.length, columns: meta.columns },
        },
        signal,
      ),
    );
  }

  close(): void {
    this.pool?.close();
    this.pool = null;
    this.indexes.clear();
    this.layers = {};
    this.format = 1;
    this.containers = ['LAC1'];
    this.codec = 'deflate';
    this.packKey = packChunkKey;
    this.manifestGrid = null;
  }
}

// --- fetch burst gate ----------------------------------------------------------------------

/** A minimal async semaphore: `run` waits for a free slot, runs the task, and
 *  releases on settle (success OR failure). Bounds concurrent fetches so a cut
 *  swing / rebase burst queues instead of stampeding the connection pool. */
class FetchGate {
  private active = 0;
  private readonly waiters: (() => void)[] = [];
  constructor(private readonly max: number) {}
  async run<T>(task: () => Promise<T>): Promise<T> {
    if (this.active >= this.max) await new Promise<void>((r) => this.waiters.push(r));
    this.active++;
    try {
      return await task();
    } finally {
      this.active--;
      this.waiters.shift()?.();
    }
  }
}

// --- decode worker pool --------------------------------------------------------------------

interface PendingJob {
  resolve: (p: ChunkPayload) => void;
  reject: (e: Error) => void;
  cleanup: () => void;
}

class DecodePool {
  private readonly workers: Worker[] = [];
  private readonly pending = new Map<number, PendingJob>();
  private nextId = 1;
  private rr = 0;

  constructor(n: number) {
    for (let i = 0; i < n; i++) {
      const worker = new Worker(new URL('./Lac1Decode.worker.ts', import.meta.url), {
        type: 'module',
        name: `lac1-decode-${i}`,
      });
      worker.onmessage = (e: MessageEvent<Lac1DecodeRes>): void => {
        const job = this.pending.get(e.data.id);
        if (!job) return; // already aborted/settled
        this.pending.delete(e.data.id);
        job.cleanup();
        if (e.data.ok) job.resolve(e.data.payload);
        else if (e.data.aborted) job.reject(new DOMException('chunk fetch aborted', 'AbortError'));
        else job.reject(new Error(`Lac1Decode: ${e.data.error}`));
      };
      worker.onerror = (e: ErrorEvent): void => this.failAll(new Error(`Lac1Decode worker crashed: ${e.message}`));
      this.workers.push(worker);
    }
  }

  decode(job: Omit<Lac1DecodeJob, 'kind' | 'id'>, signal?: AbortSignal): Promise<ChunkPayload> {
    if (signal?.aborted) return Promise.reject(new DOMException('chunk fetch aborted', 'AbortError'));
    const id = this.nextId++;
    const worker = this.workers[this.rr++ % this.workers.length] as Worker;
    return new Promise<ChunkPayload>((resolve, reject) => {
      const onAbort = (): void => {
        const p = this.pending.get(id);
        if (!p) return;
        this.pending.delete(id);
        worker.postMessage({ kind: 'abort', id } satisfies Lac1DecodeReq);
        p.reject(new DOMException('chunk fetch aborted', 'AbortError'));
      };
      const cleanup = (): void => signal?.removeEventListener('abort', onAbort);
      signal?.addEventListener('abort', onAbort);
      this.pending.set(id, { resolve, reject, cleanup });
      worker.postMessage({ kind: 'decode', id, ...job } satisfies Lac1DecodeReq);
    });
  }

  close(): void {
    this.failAll(new Error('RemoteWorldSource closed'));
    for (const w of this.workers) w.terminate();
    this.workers.length = 0;
  }

  private failAll(err: Error): void {
    for (const job of this.pending.values()) {
      job.cleanup();
      job.reject(err);
    }
    this.pending.clear();
  }
}

// --- fetch + manifest helpers ----------------------------------------------------------------

async function getJson(url: string, cache: RequestCache): Promise<unknown> {
  const resp = await fetch(url, { cache });
  if (!resp.ok) throw new Error(`HTTP ${resp.status} ${url}`);
  return resp.json();
}

async function getBytes(url: string): Promise<Uint8Array> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} ${url}`);
  return new Uint8Array(await resp.arrayBuffer());
}

function parseDictionaries(m: ManifestJson): WorldDictionaries {
  const toMap = <T>(rec: Record<string, T> | undefined): Map<number, T> => {
    const map = new Map<number, T>();
    for (const [k, v] of Object.entries(rec ?? {})) map.set(Number(k), v);
    return map;
  };
  return { species: toMap(m.speciesMap), understory: toMap(m.understoryMap), debris: toMap(m.debrisMap) };
}
