/**
 * LAC1 codec validation against the REAL cooked pilot release (asset-gen/data/out,
 * filesystem — no HTTP) + synthetic round-trips through the Lac1 encoder (S1 gate,
 * SPEC-STREAMING-WORLD §2/§9):
 *
 *   I  index integrity   — every layer index parses, record count matches the manifest,
 *                          every ref's content-addressed path exists with the indexed size
 *                          (validates the `<BiiIQ` parse AND the hash64→hash8 URL rule).
 *   H  header integrity  — magic/layer/enc/res/count/origin fields match the grid math;
 *                          re-serialized headers are byte-identical; payload CRC verifies.
 *   D  real decodes      — height (lod 0 + coarsest), biome/soil/understory/debris planes,
 *                          trees/boulders records, water: invariants + sane value ranges
 *                          (Taevaskoja ≈ 30–110 m EH2000).
 *   A  apron continuity  — a chunk's far row/col equals its east/south neighbor's first
 *                          row/col (grid.py: apron duplicates neighbor pixels).
 *   R  round-trip        — synthetic f32 height window / u8 planes / record SoA through
 *                          encode→decode: ε ≤ qscale/2 (height), bit-exact (planes/records);
 *                          delta2d/undelta2d wrap-exact on random u16.
 *
 *   npx tsx tools/probe-lac1.ts
 */
import { existsSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { deflateSync, inflateSync } from 'node:zlib';
import {
  assembleChunk,
  chunkPath,
  crc32,
  decodeChunkPayload,
  delta2d,
  encodePlanes,
  encodeQuant16,
  encodeRecordColumns,
  LAC1_HEADER_SIZE,
  LAC1_LAYER_IDS,
  packChunkKey,
  parseLac1Header,
  parseLayerIndex,
  serializeLac1Header,
  undelta2d,
  type Lac1Header,
  type Lac1LayerSchema,
} from '../src/world/source/Lac1';
import type { ChunkPayload, ChunkRef, LayerName, RecordDtype } from '../src/world/source/WorldSource';

const OUT = join(fileURLToPath(new URL('..', import.meta.url)), 'asset-gen/data/out');

let failures = 0;
const fail = (msg: string): void => {
  failures++;
  console.error(`  FAIL ${msg}`);
};
const assert = (cond: boolean, msg: string): void => {
  if (!cond) fail(msg);
};

interface LayerJson {
  enc: number;
  lods: number[];
  count: number;
  index: string;
  planes?: string[];
  columns?: [string, RecordDtype][];
}
interface ManifestJson {
  format: number;
  anchor: { e: number; n: number };
  chunkMeters: number;
  chunkRes: number;
  lodStep: number;
  layers: Record<string, LayerJson>;
}

const summary: string[][] = [];
const note = (layer: string, what: string, detail: string): void => {
  summary.push([layer, what, detail]);
};

// --- load release ---------------------------------------------------------------------------

const latest = JSON.parse(readFileSync(join(OUT, 'latest.json'), 'utf8')) as { manifest: string };
const manifest = JSON.parse(readFileSync(join(OUT, latest.manifest), 'utf8')) as ManifestJson;
const manifestDir = join(OUT, latest.manifest, '..');
assert(manifest.format === 1, `manifest format ${manifest.format} != 1`);
console.log(`[probe-lac1] release ${latest.manifest} — grid ${manifest.chunkMeters} m / ${manifest.chunkRes} px, lodStep ${manifest.lodStep}`);

// --- I: index integrity + content-addressed paths --------------------------------------------

const indexes = new Map<LayerName, Map<number, ChunkRef>>();
for (const [name, meta] of Object.entries(manifest.layers)) {
  const layer = name as LayerName;
  const refs = parseLayerIndex(new Uint8Array(readFileSync(join(manifestDir, meta.index))));
  assert(refs.length === meta.count, `I ${layer}: index has ${refs.length} != manifest count ${meta.count}`);
  const lods = new Set<number>();
  let missing = 0;
  let sizeMismatch = 0;
  for (const ref of refs) {
    lods.add(ref.lod);
    const p = join(OUT, chunkPath(layer, ref));
    if (!existsSync(p)) missing++;
    else if (statSync(p).size !== ref.size) sizeMismatch++;
  }
  assert(missing === 0, `I ${layer}: ${missing} indexed chunks missing on disk (hash8/path rule wrong?)`);
  assert(sizeMismatch === 0, `I ${layer}: ${sizeMismatch} chunks whose file size != indexed size`);
  assert(
    JSON.stringify([...lods].sort((a, b) => a - b)) === JSON.stringify(meta.lods),
    `I ${layer}: index lods {${[...lods]}} != manifest lods {${meta.lods}}`,
  );
  indexes.set(layer, new Map(refs.map((r) => [packChunkKey(r.lod, r.cx, r.cz), r])));
  note(layer, 'index', `${refs.length} chunks, lods [${meta.lods}], all paths+sizes verified`);
}

// --- H + D: read/decode a real chunk from disk ------------------------------------------------

function readChunk(layer: LayerName, ref: ChunkRef): { h: Lac1Header; payload: ChunkPayload } {
  const meta = manifest.layers[layer] as LayerJson;
  const blob = new Uint8Array(readFileSync(join(OUT, chunkPath(layer, ref))));
  const h = parseLac1Header(blob);

  // header invariants vs manifest grid math
  const layerId = LAC1_LAYER_IDS[layer as keyof typeof LAC1_LAYER_IDS];
  assert(h.layer === layerId && h.lod === ref.lod && h.cx === ref.cx && h.cz === ref.cz, `H ${layer}: identity mismatch`);
  assert(h.enc === meta.enc, `H ${layer}: enc ${h.enc} != manifest ${meta.enc}`);
  const footprint = manifest.chunkMeters * manifest.lodStep ** ref.lod;
  assert(h.originE === manifest.anchor.e + ref.cx * footprint, `H ${layer}: originE ${h.originE} off-grid`);
  assert(h.originN === manifest.anchor.n - ref.cz * footprint, `H ${layer}: originN ${h.originN} off-grid`);

  // byte-identical re-serialization + payload CRC
  const reser = serializeLac1Header(h);
  assert(Buffer.compare(Buffer.from(reser), Buffer.from(blob.subarray(0, LAC1_HEADER_SIZE))) === 0, `H ${layer}: re-serialized header differs`);
  const compressed = blob.subarray(LAC1_HEADER_SIZE, LAC1_HEADER_SIZE + h.payloadLen);
  assert(compressed.length === h.payloadLen, `H ${layer}: truncated payload`);
  assert(crc32(compressed) === h.payloadCrc, `H ${layer}: crc mismatch`);

  const schema: Lac1LayerSchema = { planes: meta.planes?.length, columns: meta.columns };
  return { h, payload: decodeChunkPayload(h, new Uint8Array(inflateSync(compressed)), schema) };
}

const pick = (layer: LayerName, lod: number): ChunkRef => {
  for (const ref of (indexes.get(layer) as Map<number, ChunkRef>).values()) if (ref.lod === lod) return ref;
  throw new Error(`no ${layer} lod${lod} chunk`);
};
/** the lod-k chunk spatially containing a lod-0 chunk (quadtree: coords divide by lodStep^k). */
const pickContaining = (layer: LayerName, lod: number, at: ChunkRef): ChunkRef => {
  const f = manifest.lodStep ** lod;
  const ref = (indexes.get(layer) as Map<number, ChunkRef>).get(packChunkKey(lod, Math.floor(at.cx / f), Math.floor(at.cz / f)));
  if (!ref) throw new Error(`no ${layer} lod${lod} chunk containing (${at.cx},${at.cz})`);
  return ref;
};
const stats = (a: Float32Array): { min: number; max: number; mean: number; nan: number } => {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let nan = 0;
  for (let i = 0; i < a.length; i++) {
    const v = a[i] as number;
    if (Number.isNaN(v)) nan++;
    else {
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
    }
  }
  return { min, max, mean: sum / (a.length - nan), nan };
};

// height — lod 0 + the coarsest shipped lod (the chunk CONTAINING the pilot area:
// off-pilot coarse chunks are legitimately flat-zero sea, e.g. lod4 (-1,0))
const pilot = pick('height', 0);
for (const lod of [0, Math.max(...(manifest.layers['height'] as LayerJson).lods)]) {
  const ref = pickContaining('height', lod, pilot);
  const { h, payload } = readChunk('height', ref);
  if (payload.kind !== 'height') throw new Error('height: wrong payload kind');
  assert(h.res === manifest.chunkRes + 1, `D height lod${lod}: res ${h.res} != chunkRes+1`);
  assert(payload.heights.length === h.res * h.res, `D height lod${lod}: length mismatch`);
  const s = stats(payload.heights);
  assert(s.nan === 0, `D height lod${lod}: ${s.nan} NaN texels`);
  // lower bound: country-wide coarse chunks include Baltic sea/nodata texels, which the
  // cooker encodes NaN → q=0 → qoffset = floor(min-1) — legitimately below sea level.
  assert(s.min >= h.qoffset - 1e-6 && s.min >= -20 && s.max <= 320, `D height lod${lod}: range [${s.min.toFixed(1)}, ${s.max.toFixed(1)}] outside Estonia EH2000 [max(-20, qoffset), 320]`);
  assert(s.max - s.min > 0.5, `D height lod${lod}: implausibly flat (${(s.max - s.min).toFixed(2)} m relief)`);
  note('height', `decode lod${lod} (${ref.cx},${ref.cz})`, `res ${h.res}, q ${h.qscale} m, [${s.min.toFixed(1)}, ${s.max.toFixed(1)}] m, mean ${s.mean.toFixed(1)} m`);
}

// A: apron continuity — far col/row == east/south neighbor's first col/row
{
  const heightIdx = indexes.get('height') as Map<number, ChunkRef>;
  let pairs = 0;
  let worst = 0;
  for (const ref of heightIdx.values()) {
    if (ref.lod !== 0 || pairs >= 2) continue;
    const east = heightIdx.get(packChunkKey(0, ref.cx + 1, ref.cz));
    const south = heightIdx.get(packChunkKey(0, ref.cx, ref.cz + 1));
    if (!east || !south) continue;
    const a = readChunk('height', ref).payload as { kind: 'height'; res: number; heights: Float32Array };
    const e = readChunk('height', east).payload as { kind: 'height'; res: number; heights: Float32Array };
    const so = readChunk('height', south).payload as { kind: 'height'; res: number; heights: Float32Array };
    const res = a.res;
    for (let j = 0; j < res; j++) {
      worst = Math.max(worst, Math.abs((a.heights[j * res + (res - 1)] as number) - (e.heights[j * res] as number)));
    }
    for (let i = 0; i < res; i++) {
      worst = Math.max(worst, Math.abs((a.heights[(res - 1) * res + i] as number) - (so.heights[i] as number)));
    }
    pairs++;
  }
  assert(pairs > 0, 'A: no lod0 chunk with both east+south neighbors found');
  assert(worst <= 1e-3, `A: apron deviates ${worst} m from neighbors (must duplicate exactly, ≤ fp noise)`);
  note('height', 'apron continuity', `${pairs} chunk(s) vs east+south neighbors, worst |Δ| = ${worst.toExponential(2)} m`);
}

// planes layers
for (const layer of ['biome', 'soil', 'understory', 'debris'] as const) {
  const meta = manifest.layers[layer] as LayerJson;
  const ref = pick(layer, 0);
  const { h, payload } = readChunk(layer, ref);
  if (payload.kind !== 'planes') throw new Error(`${layer}: wrong payload kind`);
  assert(payload.planes.length === (meta.planes as string[]).length, `D ${layer}: ${payload.planes.length} planes != manifest`);
  const nonzero = payload.planes.map((p) => {
    let n = 0;
    for (let i = 0; i < p.length; i++) if (p[i] !== 0) n++;
    return ((100 * n) / p.length).toFixed(0);
  });
  assert(nonzero.some((f) => Number(f) > 0), `D ${layer}: all planes entirely zero`);
  note(layer, `decode lod0 (${ref.cx},${ref.cz})`, `res ${h.res}, planes [${(meta.planes as string[]).join(',')}], nonzero ${nonzero.join('/')}%`);
}

// record layers
for (const layer of ['trees', 'boulders'] as const) {
  const ref = pick(layer, 0);
  const { h, payload } = readChunk(layer, ref);
  if (payload.kind !== 'records') throw new Error(`${layer}: wrong payload kind`);
  assert(payload.count === h.count && payload.cols.x.length === h.count, `D ${layer}: count mismatch`);
  const footprint = manifest.chunkMeters * manifest.lodStep ** ref.lod;
  assert(Math.abs(h.qscale - footprint / 65535) < 1e-6, `D ${layer}: header qscale ${h.qscale} != footprint/65535`);
  let bad = 0;
  for (let i = 0; i < payload.count; i++) {
    const x = payload.cols.x[i] as number;
    const z = payload.cols.z[i] as number;
    const sc = payload.cols.scale[i] as number;
    const sp = payload.cols.species[i] as number;
    if (x < 0 || x > footprint || z < 0 || z > footprint) bad++;
    else if (layer === 'trees' && (sp > 20 || sc <= 0 || sc > 4)) bad++;
    else if (layer === 'boulders' && (sp > 1 || sc <= 0 || sc > 102)) bad++;
  }
  assert(bad === 0, `D ${layer}: ${bad}/${payload.count} records out of range (x/z/species/scale)`);
  const sc = stats(payload.cols.scale);
  note(layer, `decode lod0 (${ref.cx},${ref.cz})`, `${payload.count} records, x/z ∈ [0, ${footprint}] m, scale [${sc.min.toFixed(2)}, ${sc.max.toFixed(2)}]`);
}

// water — dry texels decode to NaN, wet levels sit inside the height range
{
  const ref = pick('water', 0);
  const { h, payload } = readChunk('water', ref);
  if (payload.kind !== 'height') throw new Error('water: wrong payload kind');
  const s = stats(payload.heights);
  const wet = payload.heights.length - s.nan;
  assert(s.nan > 0, 'D water: no dry (NaN) texels in a pilot chunk — suspicious');
  if (wet > 0) assert(s.min >= -10 && s.max <= 320, `D water: wet levels [${s.min.toFixed(1)}, ${s.max.toFixed(1)}] outside sane EH2000`);
  note('water', `decode lod0 (${ref.cx},${ref.cz})`, `res ${h.res}, ${((100 * wet) / payload.heights.length).toFixed(1)}% wet, levels [${wet ? s.min.toFixed(1) : '—'}, ${wet ? s.max.toFixed(1) : '—'}] m`);
}

// --- R: synthetic round-trips through the encoder ---------------------------------------------

// deterministic PRNG (mulberry32) — reproducible failures
const rng = ((seed: number) => (): number => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
})(0x1ac1);

{
  // height: plausible Taevaskoja-like window (30–110 m) + one NaN texel, full container path
  const res = 257;
  const qscale = 0.01;
  const src = new Float32Array(res * res);
  for (let j = 0; j < res; j++)
    for (let i = 0; i < res; i++) {
      src[j * res + i] = 70 + 35 * Math.sin(i * 0.045) * Math.cos(j * 0.031) + 4 * Math.sin(i * 0.31 + j * 0.17);
    }
  src[res * 5 + 7] = NaN;
  const { raw, qoffset } = encodeQuant16(src, res, qscale);
  const file = assembleChunk(
    { layer: LAC1_LAYER_IDS.height, lod: 0, enc: 1, flags: 0, cx: 3, cz: 4, res, count: 0, originE: 0, originN: 0, qoffset, qscale },
    new Uint8Array(deflateSync(raw)),
  );
  const h = parseLac1Header(file);
  assert(crc32(file.subarray(LAC1_HEADER_SIZE)) === h.payloadCrc, 'R height: assembled crc invalid');
  const back = decodeChunkPayload(h, new Uint8Array(inflateSync(file.subarray(LAC1_HEADER_SIZE))), {}) as { kind: 'height'; heights: Float32Array };
  let worst = 0;
  for (let i = 0; i < src.length; i++) {
    const s = src[i] as number;
    if (Number.isNaN(s)) continue; // NaN encodes as qoffset by design (encode.py nan_to_num)
    worst = Math.max(worst, Math.abs(s - (back.heights[i] as number)));
  }
  const eps = qscale / 2 + 1e-4;
  assert(worst <= eps, `R height: ε ${worst.toExponential(3)} > qscale/2 (${eps})`);
  assert(back.heights[res * 5 + 7] === qoffset, 'R height: NaN texel should decode to qoffset');
  note('height', 'round-trip 257²', `qoffset ${qoffset}, ε_max ${worst.toExponential(2)} ≤ qscale/2 = ${qscale / 2}`);
}

{
  // delta2d/undelta2d wrap-exactness on adversarial random u16
  const res = 64;
  const q = new Uint16Array(res * res);
  for (let i = 0; i < q.length; i++) q[i] = (rng() * 65536) | 0;
  const rt = undelta2d(delta2d(q, res), res);
  let diff = 0;
  for (let i = 0; i < q.length; i++) if (rt[i] !== q[i]) diff++;
  assert(diff === 0, `R delta2d: ${diff} texels differ after round-trip`);
  note('—', 'delta2d wrap', `random u16 ${res}² exact (${diff} diffs)`);
}

{
  // planes: random u8 → encode → decode, bit-exact
  const res = 128;
  const planes = [0, 1].map(() => {
    const p = new Uint8Array(res * res);
    for (let i = 0; i < p.length; i++) p[i] = (rng() * 256) | 0;
    return p;
  });
  const file = assembleChunk(
    { layer: LAC1_LAYER_IDS.biome, lod: 0, enc: 2, flags: 0, cx: 0, cz: 0, res, count: 0, originE: 0, originN: 0, qoffset: 0, qscale: 0 },
    new Uint8Array(deflateSync(encodePlanes(planes, res))),
  );
  const h = parseLac1Header(file);
  const back = decodeChunkPayload(h, new Uint8Array(inflateSync(file.subarray(LAC1_HEADER_SIZE))), { planes: 2 }) as { kind: 'planes'; planes: Uint8Array[] };
  let diff = 0;
  for (let p = 0; p < 2; p++) for (let i = 0; i < res * res; i++) if ((back.planes[p] as Uint8Array)[i] !== (planes[p] as Uint8Array)[i]) diff++;
  assert(diff === 0, `R planes: ${diff} bytes differ`);
  note('biome', 'round-trip planes', `2 × ${res}² random u8 bit-exact`);
}

{
  // records: synthetic tree SoA → encode → decode; u16/u8 columns exact, x/z dequant exact.
  // qscale rides the header as f32 — expectations must quantize it the same way.
  const count = 1000;
  const qscale = Math.fround(2048 / 65535);
  const cols = {
    x: new Uint16Array(count),
    z: new Uint16Array(count),
    species: new Uint8Array(count),
    scale: new Uint8Array(count),
    variant: new Uint8Array(count),
  };
  for (let i = 0; i < count; i++) {
    cols.x[i] = (rng() * 65536) | 0;
    cols.z[i] = (rng() * 65536) | 0;
    cols.species[i] = (rng() * 21) | 0;
    cols.scale[i] = 16 + ((rng() * 112) | 0);
    cols.variant[i] = (rng() * 256) | 0;
  }
  const file = assembleChunk(
    { layer: LAC1_LAYER_IDS.trees, lod: 0, enc: 3, flags: 0, cx: 0, cz: 0, res: 0, count, originE: 0, originN: 0, qoffset: 0, qscale },
    new Uint8Array(deflateSync(encodeRecordColumns([cols.x, cols.z, cols.species, cols.scale, cols.variant]))),
  );
  const schema: Lac1LayerSchema = {
    columns: [
      ['x', 'u16'],
      ['z', 'u16'],
      ['species', 'u8'],
      ['scale', 'u8'],
      ['variant', 'u8'],
    ],
  };
  const h = parseLac1Header(file);
  const back = decodeChunkPayload(h, new Uint8Array(inflateSync(file.subarray(LAC1_HEADER_SIZE))), schema);
  if (back.kind !== 'records') throw new Error('R records: wrong kind');
  let diff = 0;
  for (let i = 0; i < count; i++) {
    if (back.cols.x[i] !== Math.fround((cols.x[i] as number) * qscale)) diff++;
    if (back.cols.z[i] !== Math.fround((cols.z[i] as number) * qscale)) diff++;
    if (back.cols.species[i] !== cols.species[i]) diff++;
    if (back.cols.scale[i] !== Math.fround((cols.scale[i] as number) / 64)) diff++;
    if (back.cols.variant[i] !== cols.variant[i]) diff++;
  }
  assert(diff === 0, `R records: ${diff} column values differ`);
  note('trees', 'round-trip records', `${count} random records exact (x/z/scale dequant, species/variant bytes)`);
}

// --- summary -----------------------------------------------------------------------------------

console.log('\n  layer      | check                    | result');
console.log('  -----------|--------------------------|-------------------------------------------------------------');
for (const [layer, what, detail] of summary) {
  console.log(`  ${(layer as string).padEnd(10)} | ${(what as string).padEnd(24)} | ${detail}`);
}

if (failures > 0) {
  console.error(`\n[probe-lac1] ${failures} FAILURES`);
  process.exit(1);
}
console.log('\n[probe-lac1] PASS — LAC1 header/index/enc1/enc2/enc3 verified against the real pilot release + encoder round-trips');
