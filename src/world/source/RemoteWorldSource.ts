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
import type { Lac1DecodeJob, Lac1DecodeReq, Lac1DecodeRes } from './Lac1Decode.worker';
import type {
  ChunkKey,
  ChunkPayload,
  ChunkRef,
  CommunityEntry,
  LayerName,
  SpeciesEntry,
  WorldDictionaries,
  WorldLayerMeta,
  WorldManifest,
  WorldSource,
} from './WorldSource';

/** braffolk.com serves the synced bucket at `laas-data/` (live-verified: CloudFront,
 *  CORS *); dev passes a localhost static server over asset-gen/data/out instead —
 *  the CDN caches latest.json longer than local recooks. */
const DEFAULT_BASE_URL = 'https://braffolk.com/laas-data';
const DECODE_WORKERS = 2;

/** The manifest.json fields this client consumes (manifest.py build_release). */
interface ManifestJson {
  format: number;
  anchor: { e: number; n: number };
  chunkMeters: number;
  chunkRes: number;
  lodStep: number;
  speciesMap?: Record<string, SpeciesEntry>;
  understoryMap?: Record<string, CommunityEntry>;
  debrisMap?: Record<string, CommunityEntry>;
  layers: Record<string, { enc: number; lods: number[]; count: number; bytes: number; index: string; texelMeters?: number; planes?: string[]; columns?: [string, string][] }>;
}

export class RemoteWorldSource implements WorldSource {
  private readonly baseUrl: string;
  private readonly indexes = new Map<LayerName, Map<number, ChunkRef>>();
  private layers: Partial<Record<LayerName, WorldLayerMeta>> = {};
  private pool: DecodePool | null = null;

  constructor(baseUrl: string = DEFAULT_BASE_URL) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
  }

  async open(progress?: (msg: string) => void): Promise<WorldManifest> {
    const latest = (await getJson(`${this.baseUrl}/latest.json`, 'no-cache')) as { manifest: string };
    progress?.(`manifest ${latest.manifest}`);
    const m = (await getJson(`${this.baseUrl}/${latest.manifest}`, 'default')) as ManifestJson;
    if (m.format !== 1) throw new Error(`RemoteWorldSource: unsupported manifest format ${m.format}`);
    const manifestDir = latest.manifest.slice(0, latest.manifest.lastIndexOf('/'));

    for (const [name, meta] of Object.entries(m.layers)) {
      if (!(name in LAC1_LAYER_IDS)) continue; // future layers (e.g. canopy) until the codec knows them
      const layer = name as LayerName;
      const bin = await getBytes(`${this.baseUrl}/${manifestDir}/${meta.index}`);
      const map = new Map<number, ChunkRef>();
      for (const ref of parseLayerIndex(bin)) map.set(packChunkKey(ref.lod, ref.cx, ref.cz), ref);
      if (map.size !== meta.count) throw new Error(`RemoteWorldSource: ${layer} index has ${map.size} != ${meta.count} chunks`);
      this.indexes.set(layer, map);
      this.layers[layer] = {
        enc: meta.enc,
        lods: meta.lods,
        chunkCount: meta.count,
        texelMeters: meta.texelMeters,
        planes: meta.planes,
        columns: meta.columns as WorldLayerMeta['columns'],
      };
      progress?.(`index ${layer}: ${map.size} chunks`);
    }

    this.pool = new DecodePool(DECODE_WORKERS);
    const indexes = this.indexes;
    return {
      grid: { anchorE: m.anchor.e, anchorN: m.anchor.n, chunkMeters: m.chunkMeters, chunkRes: m.chunkRes, lodStep: m.lodStep },
      layers: this.layers,
      dictionaries: parseDictionaries(m),
      coverage(layer: LayerName, key: ChunkKey): ChunkRef | null {
        return indexes.get(layer)?.get(packChunkKey(key.lod, key.cx, key.cz)) ?? null;
      },
    };
  }

  fetch(layer: LayerName, key: ChunkKey, signal?: AbortSignal): Promise<ChunkPayload | null> {
    const ref = this.indexes.get(layer)?.get(packChunkKey(key.lod, key.cx, key.cz));
    if (!ref) return Promise.resolve(null); // authoritative absence — no request
    const pool = this.pool;
    const meta = this.layers[layer];
    if (!pool || !meta) return Promise.reject(new Error('RemoteWorldSource: fetch before open()'));
    return pool.decode(
      {
        url: `${this.baseUrl}/${chunkPath(layer, ref)}`,
        layerId: LAC1_LAYER_IDS[layer as keyof typeof LAC1_LAYER_IDS],
        lod: key.lod,
        cx: key.cx,
        cz: key.cz,
        schema: { planes: meta.planes?.length, columns: meta.columns },
      },
      signal,
    );
  }

  close(): void {
    this.pool?.close();
    this.pool = null;
    this.indexes.clear();
    this.layers = {};
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
