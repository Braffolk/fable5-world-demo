/**
 * ChunkContent — the ONE placement consumer (SPEC-STREAMING-WORLD §2/§7, S2).
 * Record payloads from ANY WorldSource become the per-species instance streams
 * {a,b,fill} that the boot bindInstances bridge (WorldRegistry) consumes — the
 * subsystem never knows which source fed it (law 3).
 *
 * Word layout (Scatter.ts contract): A = (x, y, z, scale), B = (yaw, leanX, leanZ, idF).
 *
 * Exactness columns present (generated source) → used VERBATIM: instance words are
 * byte-identical to the direct scatter path (the S2 gate). Absent (Estonia) → the ONE
 * derive rule, shared with nothing else:
 *   y    = height sample at (x,z) − scale·0.12 sink (trees only — root flare seam)
 *   yaw  = pcg2d hash(cx,cz, xq,zq) · τ         (xq/zq = u16-quantized local coords,
 *          the wire lattice — stable across refetches)
 *   lean = slope-normal.xz · 0.18 + (hash − ½)·0.12   (Scatter.ts:469-470 formula)
 */
import type { ChunkKey, ChunkPayload, LayerName, WorldManifest, WorldSource } from '../../world/source/WorldSource';
import { makeGroundDeriver, type GroundDeriver } from '../../world/source/RecordGround';
import { ingestEtakBoulders, type EtakBoulderRecords } from '../../vegetation/EtakBoulders';

export interface InstanceStream {
  a: Float32Array;
  b: Float32Array;
  fill: number;
}

export interface ChunkContentStreams {
  perId: Map<number, InstanceStream>;
  total: number;
}

/** record layers a source may serve, in the canonical stream-build order. */
const RECORD_LAYER_ORDER: readonly LayerName[] = ['trees', 'understory', 'extras', 'stones', 'boulders'];


/**
 * Build the boot instance streams: every record chunk of every record layer the
 * manifest declares, chunks in row-major (cz,cx) order, in-chunk scatter order
 * preserved. idFOf maps (species, variant) → geometry-pool idF; the default is the
 * generated source's identity rule (species IS the VegClass, idF = cls·8 + variant).
 * Estonia's SpeciesMap resolver replaces it at S7.
 */
export async function buildChunkContentStreams(
  source: WorldSource,
  manifest: WorldManifest,
  idFOf: (species: number, variant: number) => number = (s, v) => s * 8 + v,
): Promise<ChunkContentStreams> {
  const jobs: { layer: LayerName; key: ChunkKey; payload: ChunkPayload & { kind: 'records' } }[] = [];
  for (const layer of RECORD_LAYER_ORDER) {
    if (!manifest.layers[layer]) continue;
    const keys = manifest.chunks(layer, 0).sort((p, q) => p.cz - q.cz || p.cx - q.cx);
    for (const key of keys) {
      const payload = await source.fetch(layer, key);
      if (!payload) continue;
      if (payload.kind !== 'records') throw new Error(`ChunkContent: ${layer} chunk is not records`);
      jobs.push({ layer, key, payload });
    }
  }

  // two passes: exact per-id sizes, then fill (same shape the registry pools expect)
  const counts = new Map<number, number>();
  for (const { payload } of jobs) {
    const { species, variant } = payload.cols;
    for (let i = 0; i < payload.count; i++) {
      const id = idFOf(species[i] as number, variant[i] as number);
      counts.set(id, (counts.get(id) ?? 0) + 1);
    }
  }
  const perId = new Map<number, InstanceStream>();
  for (const [id, n] of counts) perId.set(id, { a: new Float32Array(n * 4), b: new Float32Array(n * 4), fill: 0 });

  let total = 0;
  for (const { layer, key, payload } of jobs) {
    const { cols, count } = payload;
    const footprint = manifest.grid.chunkMeters * manifest.grid.lodStep ** key.lod;
    const minX = manifest.grid.originX + key.cx * footprint;
    const minZ = manifest.grid.originZ + key.cz * footprint;
    const derive = cols.y && cols.yaw && cols.leanX && cols.leanZ ? null : await makeDeriver(source, manifest, key);
    for (let i = 0; i < count; i++) {
      const id = idFOf(cols.species[i] as number, cols.variant[i] as number);
      const s = perId.get(id) as InstanceStream;
      const d = s.fill * 4;
      const x = cols.xw ? (cols.xw[i] as number) : minX + (cols.x[i] as number);
      const z = cols.zw ? (cols.zw[i] as number) : minZ + (cols.z[i] as number);
      const scale = cols.scale[i] as number;
      s.a[d] = x;
      s.a[d + 2] = z;
      s.a[d + 3] = scale;
      s.b[d + 3] = id;
      if (derive) {
        const dv = derive(cols.x[i] as number, cols.z[i] as number);
        s.a[d + 1] = cols.y ? (cols.y[i] as number) : dv.h - (layer === 'trees' ? scale * 0.12 : 0);
        s.b[d] = cols.yaw ? (cols.yaw[i] as number) : dv.yaw;
        s.b[d + 1] = cols.leanX ? (cols.leanX[i] as number) : dv.leanX;
        s.b[d + 2] = cols.leanZ ? (cols.leanZ[i] as number) : dv.leanZ;
      } else {
        s.a[d + 1] = (cols.y as Float32Array)[i] as number;
        s.b[d] = (cols.yaw as Float32Array)[i] as number;
        s.b[d + 1] = (cols.leanX as Float32Array)[i] as number;
        s.b[d + 2] = (cols.leanZ as Float32Array)[i] as number;
      }
      s.fill++;
      total++;
    }
  }
  return { perId, total };
}

// --- S7: single-chunk instance build (the streamed instance-band pool) -------------------

export interface ChunkInstanceOpts {
  /** (species, variant) → library idF for TREE records (SpeciesMap on Estonia). */
  idFOf: (species: number, variant: number) => number;
  /** nominal radius per rock class — presence enables ETAK boulder ingest. */
  boulderRadiusOf?: (cls: number) => number;
}

export interface ChunkInstances {
  a: Float32Array; // A-words (x,y,z,scale) — ABSOLUTE game space
  b: Float32Array; // B-words (yaw,leanX,leanZ,idF)
  count: number;
}

/**
 * Build ONE LOD0 chunk's flat instance list (trees + ETAK boulders) for the S7
 * streamed instance pool — the SAME placement rules as the boot streams, one chunk
 * at a time. Positions are ABSOLUTE game space (rewriteInstanceBlock stores them
 * StreamOrigin-relative). Trees derive y/yaw/lean when the source omits them
 * (Estonia); boulders ride EtakBoulders (§H) grounded on this chunk's height window.
 */
export async function buildChunkInstances(
  source: WorldSource,
  manifest: WorldManifest,
  key: ChunkKey,
  opts: ChunkInstanceOpts,
): Promise<ChunkInstances> {
  const footprint = manifest.grid.chunkMeters * manifest.grid.lodStep ** key.lod;
  const minX = manifest.grid.originX + key.cx * footprint;
  const minZ = manifest.grid.originZ + key.cz * footprint;
  const derive = await makeDeriver(source, manifest, key);
  const aArr: number[] = [];
  const bArr: number[] = [];

  const trees = manifest.layers.trees ? await source.fetch('trees', key) : null;
  if (trees && trees.kind === 'records') {
    const { cols, count } = trees;
    for (let i = 0; i < count; i++) {
      const scale = cols.scale[i] as number;
      const xLocal = cols.x[i] as number;
      const zLocal = cols.z[i] as number;
      const x = cols.xw ? (cols.xw[i] as number) : minX + xLocal;
      const z = cols.zw ? (cols.zw[i] as number) : minZ + zLocal;
      const id = opts.idFOf(cols.species[i] as number, cols.variant[i] as number);
      const dv = cols.y && cols.yaw && cols.leanX && cols.leanZ ? null : derive(xLocal, zLocal);
      aArr.push(x, cols.y ? (cols.y[i] as number) : (dv as { h: number }).h - scale * 0.12, z, scale);
      bArr.push(
        cols.yaw ? (cols.yaw[i] as number) : (dv as { yaw: number }).yaw,
        cols.leanX ? (cols.leanX[i] as number) : (dv as { leanX: number }).leanX,
        cols.leanZ ? (cols.leanZ[i] as number) : (dv as { leanZ: number }).leanZ,
        id,
      );
    }
  }

  if (opts.boulderRadiusOf && manifest.layers.boulders) {
    const bpay = await source.fetch('boulders', key);
    if (bpay && bpay.kind === 'records') {
      const recs: EtakBoulderRecords = {
        count: bpay.count,
        x: bpay.cols.x,
        z: bpay.cols.z,
        kind: bpay.cols.species,
        sizeM: bpay.cols.scale,
        variant: bpay.cols.variant,
      };
      const inst = ingestEtakBoulders(recs, {
        originX: minX,
        originZ: minZ,
        radiusOf: opts.boulderRadiusOf,
        heightAt: (wx, wz) => derive(wx - minX, wz - minZ).h,
      });
      for (let i = 0; i < inst.a.length; i++) aArr.push(inst.a[i] as number);
      for (let i = 0; i < inst.b.length; i++) bArr.push(inst.b[i] as number);
    }
  }

  return { a: Float32Array.from(aArr), b: Float32Array.from(bArr), count: aArr.length / 4 };
}

// --- derive-if-absent --------------------------------------------------------------------
// The ONE derivation rule lives in RecordGround.ts (pure — S8: the StreamBrain worker's
// fartile bakes import the SAME function, so streamed crowns ground bit-identically to
// the instance paths). This wrapper only fetches the chunk's height window.

async function makeDeriver(source: WorldSource, manifest: WorldManifest, key: ChunkKey): Promise<GroundDeriver> {
  const payload = await source.fetch('height', key);
  if (!payload || payload.kind !== 'height') {
    throw new Error(`ChunkContent: no height chunk (${key.lod},${key.cx},${key.cz}) to ground records on`);
  }
  const footprint = manifest.grid.chunkMeters * manifest.grid.lodStep ** key.lod;
  return makeGroundDeriver(payload.heights, payload.res, footprint, key.cx, key.cz);
}
