/**
 * LAC1 decode worker (module Worker) — HTTP GET one chunk, verify header + CRC,
 * inflate (DecompressionStream 'deflate' = raw zlib, matching python zlib.compress),
 * decode to a spec-§2 ChunkPayload and TRANSFER the typed arrays back zero-copy.
 * THREE-FREE (imports only the pure Lac1 codec), same discipline as
 * DagWorker.worker.ts. Aborts: the main thread posts {kind:'abort'} and the worker
 * cancels its own fetch via a per-job AbortController.
 */
/// <reference lib="webworker" />
import { crc32, decodeChunkPayload, LAC1_HEADER_SIZE, parseLac1Header, type Lac1LayerSchema } from './Lac1';
import type { ChunkPayload } from './WorldSource';

export interface Lac1DecodeJob {
  kind: 'decode';
  id: number;
  url: string;
  /** expected identity — mismatches throw (stale index / wrong file). */
  layerId: number;
  lod: number;
  cx: number;
  cz: number;
  schema: Lac1LayerSchema;
}

export type Lac1DecodeReq = Lac1DecodeJob | { kind: 'abort'; id: number };

export type Lac1DecodeRes =
  | { id: number; ok: true; payload: ChunkPayload }
  | { id: number; ok: false; aborted: boolean; error: string };

const ctx = self as unknown as DedicatedWorkerGlobalScope;
const inflight = new Map<number, AbortController>();

async function inflate(bytes: Uint8Array<ArrayBuffer>): Promise<Uint8Array> {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function transferables(payload: ChunkPayload): ArrayBuffer[] {
  const buffers = new Set<ArrayBuffer>();
  if (payload.kind === 'height') buffers.add(payload.heights.buffer as ArrayBuffer);
  else if (payload.kind === 'planes') for (const p of payload.planes) buffers.add(p.buffer as ArrayBuffer);
  else for (const c of Object.values(payload.cols)) if (c) buffers.add(c.buffer as ArrayBuffer);
  return [...buffers];
}

async function decode(job: Lac1DecodeJob, ac: AbortController): Promise<ChunkPayload> {
  const resp = await fetch(job.url, { signal: ac.signal });
  if (!resp.ok) throw new Error(`HTTP ${resp.status} ${job.url}`);
  const blob = new Uint8Array(await resp.arrayBuffer());
  const h = parseLac1Header(blob);
  if (h.layer !== job.layerId || h.lod !== job.lod || h.cx !== job.cx || h.cz !== job.cz) {
    throw new Error(`header (${h.layer},${h.lod},${h.cx},${h.cz}) != expected (${job.layerId},${job.lod},${job.cx},${job.cz})`);
  }
  if (blob.length < LAC1_HEADER_SIZE + h.payloadLen) throw new Error(`truncated: ${blob.length} B < 56+${h.payloadLen}`);
  const compressed = blob.subarray(LAC1_HEADER_SIZE, LAC1_HEADER_SIZE + h.payloadLen);
  if (crc32(compressed) !== h.payloadCrc) throw new Error('payload crc mismatch');
  return decodeChunkPayload(h, await inflate(compressed), job.schema);
}

ctx.onmessage = (e: MessageEvent<Lac1DecodeReq>): void => {
  const req = e.data;
  if (req.kind === 'abort') {
    inflight.get(req.id)?.abort();
    return;
  }
  const ac = new AbortController();
  inflight.set(req.id, ac);
  void decode(req, ac)
    .then((payload) => {
      ctx.postMessage({ id: req.id, ok: true, payload } satisfies Lac1DecodeRes, transferables(payload));
    })
    .catch((err: unknown) => {
      const aborted = ac.signal.aborted;
      const res: Lac1DecodeRes = { id: req.id, ok: false, aborted, error: err instanceof Error ? err.message : String(err) };
      ctx.postMessage(res);
    })
    .finally(() => inflight.delete(req.id));
};
