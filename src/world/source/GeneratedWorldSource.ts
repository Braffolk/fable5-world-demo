/**
 * GeneratedWorldSource — WorldSource over today's procedural boot (SPEC-STREAMING-WORLD
 * §2, S2). open() runs Heightfield.generate + runScatter ONCE (BootCache behavior
 * untouched — the registry keys off per-id counts, which binning preserves exactly),
 * reads the scatter layers back, and bins everything into chunk-keyed views:
 *
 *  - height:  f32 windows of cpuHeights on the native texel-centered lattice
 *             ((chunkRes+1)² incl. apron; off-world texels clamp-extend; coarser
 *             lods subsample the native lattice at stride lodStep^k, offset-centered).
 *  - records: trees/understory/extras/stones re-binned per 2048 m chunk, scatter
 *             order preserved, WITH the exactness columns (y/yaw/leanX/leanZ AND
 *             xw/zw — the original absolute f32 words pass through byte-exact).
 *  - planes:  biome classification (classId/vegDensity/snow/rockExposure) from the
 *             boot biomeTex, read back lazily on first fetch (S3a's consumer; costs
 *             nothing until asked). Since S3a also: 'fields' surface planes
 *             (moisture/flowStrength/snow/rockExposure at sim res, hydrology flow
 *             buffers read back lazily) and 'water' (waterY f32 windows of the
 *             existing cpuWaterY mirror) — the TerrainField plane fills.
 *
 * The scene reads `.heightfield` for the still-live boot handles (waterY/flow
 * until S9, wind noise, registry terrain build) and `.scatter` for the HUD
 * counters/debug view; placement flows ONLY through fetch('records') →
 * ChunkContent; terrain planes flow ONLY through
 * fetch('height'|'biome'|'fields'|'water') → TerrainField; the S4 canopy window
 * flows through fetch('trees').
 */
import type { Renderer } from 'three/webgpu';
import { Fn, If, Return, instanceIndex, instancedArray, textureLoad, uint, uvec2 } from 'three/tsl';
import type { WorldSeed } from '../../core/Seed';
import { readBuffer } from '../../nanite/Tsl';
import type { NV4, NU } from '../../gpu/TSLTypes';
import { runScatter, type ScatterLayer, type ScatterResult } from '../../gpu/passes/Scatter';
import type { FloatBuffer } from '../../gpu/passes/HeightSynthesis';
import { Heightfield } from '../Heightfield';
import { WORLD_HALF, WORLD_SIZE } from '../WorldConst';
import { packChunkKey } from './Lac1';
import type { ChunkKey, ChunkPayload, ChunkRef, LayerName, WorldManifest, WorldSource } from './WorldSource';

/** chunk footprint at LOD0 (matches the Estonia grid — one shared convention). */
const CHUNK_METERS = 2048;
const LOD_STEP = 4;
/** height lods served: the world sits inside ONE lod1 chunk already, so further
 *  rungs would be pure clamp-padding (S3a's TerrainField decides if it wants them). */
const HEIGHT_LODS = [0, 1];
/** biome/fields lods: lod1 lets the TerrainField's 1024² u8 planes cover the whole
 *  4096 m world (lod0 = the native-texel near window). */
const BIOME_LODS = [0, 1];
const FIELDS_LODS = [0, 1];
/** waterY is already windowed at sim res; one lod covers the world at plane res */
const WATER_LODS = [0];
const BIOME_PLANES = ['classId', 'vegDensity', 'snow', 'rockExposure'] as const;
const FIELDS_PLANES = ['moisture', 'flowStrength', 'snow', 'rockExposure'] as const;

const RECORD_LAYERS = ['trees', 'understory', 'extras', 'stones'] as const;
type RecordLayer = (typeof RECORD_LAYERS)[number];

const RECORD_COLUMNS = [
  ['x', 'f32'],
  ['z', 'f32'],
  ['species', 'u8'],
  ['scale', 'f32'],
  ['variant', 'u8'],
] as const;

interface RecordBin {
  count: number;
  x: Float32Array; // chunk-local (lossy re-anchor — xw/zw carry the exact words)
  z: Float32Array;
  species: Uint8Array;
  scale: Float32Array;
  variant: Uint8Array;
  y: Float32Array;
  yaw: Float32Array;
  leanX: Float32Array;
  leanZ: Float32Array;
  xw: Float32Array;
  zw: Float32Array;
}

/** raw readback of one scatter layer (A/B interleaved vec4 quads). */
export interface RawScatterLayer {
  a: Float32Array;
  b: Float32Array;
  count: number;
}

export class GeneratedWorldSource implements WorldSource {
  private readonly renderer: Renderer;
  private readonly seed: WorldSeed;
  private hf: Heightfield | null = null;
  private scatterResult: ScatterResult | null = null;
  private readonly bins = new Map<RecordLayer, Map<number, RecordBin>>();
  private biomePlanes: Uint8Array[] | null = null;
  /** lazy sim-res surface-fields planes (FIELDS_PLANES order) — hydrology flow
   *  readback + biome snow/rock subsample, built on first fetch('fields') */
  private fieldsPlanes: Uint8Array[] | null = null;
  /** ?s2gate=1: the raw pre-binning readback, retained for the byte-identity probe. */
  rawLayers: Partial<Record<RecordLayer, RawScatterLayer>> | null = null;

  constructor(renderer: Renderer, seed: WorldSeed) {
    this.renderer = renderer;
    this.seed = seed;
  }

  /** the live boot heightfield — S3a ports its consumers onto height planes. */
  get heightfield(): Heightfield {
    if (!this.hf) throw new Error('GeneratedWorldSource: heightfield before open()');
    return this.hf;
  }

  /** the live GPU scatter result — HUD counters + the ?view=scatter debug. */
  get scatter(): ScatterResult {
    if (!this.scatterResult) throw new Error('GeneratedWorldSource: scatter before open()');
    return this.scatterResult;
  }

  async open(progress?: (frac: number, msg: string) => void): Promise<WorldManifest> {
    const hf = await Heightfield.generate(this.renderer, this.seed, (p, m) => progress?.(p * 0.86, m));
    this.hf = hf;
    progress?.(0.87, 'vegetation: scattering instances');
    this.scatterResult = await runScatter(this.renderer, hf, this.seed);
    progress?.(0.94, 'world source: chunk binning');
    const keepRaw = new URLSearchParams(window.location.search).get('s2gate') === '1';
    if (keepRaw) this.rawLayers = {};
    const layerOf: Record<RecordLayer, ScatterLayer> = {
      trees: this.scatterResult.trees,
      understory: this.scatterResult.understory,
      extras: this.scatterResult.extras,
      stones: this.scatterResult.stones,
    };
    for (const name of RECORD_LAYERS) {
      const raw = await readLayer(this.renderer, layerOf[name]);
      if (this.rawLayers) this.rawLayers[name] = raw;
      this.bins.set(name, binLayer(raw));
    }
    progress?.(1, 'world source: ready');

    const bins = this.bins;
    const texel = WORLD_SIZE / hf.res;
    const simTexel = WORLD_SIZE / hf.simRes;
    const chunkRes = Math.round(CHUNK_METERS / texel);
    const layers: WorldManifest['layers'] = {
      height: { enc: 1, lods: [...HEIGHT_LODS], chunkCount: 5, texelMeters: texel },
      biome: { enc: 2, lods: [...BIOME_LODS], chunkCount: 5, texelMeters: texel, planes: BIOME_PLANES },
      fields: { enc: 2, lods: [...FIELDS_LODS], chunkCount: 5, texelMeters: simTexel, planes: FIELDS_PLANES },
      water: { enc: 1, lods: [...WATER_LODS], chunkCount: 4, texelMeters: simTexel },
    };
    for (const name of RECORD_LAYERS) {
      layers[name] = { enc: 3, lods: [0], chunkCount: bins.get(name)?.size ?? 0, columns: RECORD_COLUMNS };
    }
    const rasterRef = (layer: LayerName, key: ChunkKey): ChunkRef | null => {
      const meta = layers[layer];
      if (!meta?.lods.includes(key.lod) || !meta.texelMeters) return null;
      const side = key.lod === 0 ? 2 : 1; // 2×2 lod0 chunks; ONE chunk per coarser lod
      if (key.cx < 0 || key.cz < 0 || key.cx >= side || key.cz >= side) return null;
      const resW = Math.round(CHUNK_METERS / meta.texelMeters) + 1;
      return { ...key, size: resW * resW * (meta.enc === 2 ? meta.planes?.length ?? 1 : 4), hash64: 0n };
    };
    return {
      format: 1,
      containers: ['LAC1'],
      codec: 'deflate',
      grid: {
        anchorE: 0,
        anchorN: 0,
        chunkMeters: CHUNK_METERS,
        chunkRes,
        lodStep: LOD_STEP,
        originX: -WORLD_HALF,
        originZ: -WORLD_HALF,
      },
      layers,
      dictionaries: { species: new Map(), understory: new Map(), debris: new Map() }, // veg classes ARE the pools (S7 SpeciesMap is Estonia's concern)
      coverage(layer: LayerName, key: ChunkKey): ChunkRef | null {
        const bin = bins.get(layer as RecordLayer)?.get(packChunkKey(key.lod, key.cx, key.cz));
        if (bin) return { ...key, size: bin.count * 44, hash64: 0n };
        if (bins.has(layer as RecordLayer)) return null; // empty record chunk = authoritative absence
        return rasterRef(layer, key);
      },
      chunks(layer: LayerName, lod: number): ChunkKey[] {
        const layerBins = bins.get(layer as RecordLayer);
        if (layerBins) {
          if (lod !== 0) return [];
          return [...layerBins.keys()]
            .map((k) => keyOfPacked(k))
            .sort((p, q) => p.cz - q.cz || p.cx - q.cx);
        }
        const out: ChunkKey[] = [];
        const side = lod === 0 ? 2 : 1;
        if ((layers[layer]?.lods ?? []).includes(lod)) {
          for (let cz = 0; cz < side; cz++) for (let cx = 0; cx < side; cx++) out.push({ lod, cx, cz });
        }
        return out;
      },
    };
  }

  async fetch(layer: LayerName, key: ChunkKey): Promise<ChunkPayload | null> {
    const hf = this.heightfield;
    const layerBins = this.bins.get(layer as RecordLayer);
    if (layerBins) {
      const bin = layerBins.get(packChunkKey(key.lod, key.cx, key.cz));
      if (!bin) return null;
      const { count, x, z, species, scale, variant, y, yaw, leanX, leanZ, xw, zw } = bin;
      return { kind: 'records', count, cols: { x, z, species, scale, variant, y, yaw, leanX, leanZ, xw, zw } };
    }
    if (layer === 'height') {
      if (!this.rasterKeyValid(key, HEIGHT_LODS)) return null;
      const src = hf.cpuHeights;
      if (!src) throw new Error('GeneratedWorldSource: cpuHeights missing');
      return { kind: 'height', ...this.rasterWindowF32(src, hf.res, key) };
    }
    if (layer === 'water') {
      if (!this.rasterKeyValid(key, WATER_LODS)) return null;
      const src = hf.cpuWaterY;
      if (!src) throw new Error('GeneratedWorldSource: cpuWaterY missing');
      return { kind: 'height', ...this.rasterWindowF32(src, hf.simRes, key) };
    }
    if (layer === 'biome') {
      if (!this.rasterKeyValid(key, BIOME_LODS)) return null;
      const planes = this.biomePlanes ?? (this.biomePlanes = await this.readBiomePlanes());
      return { kind: 'planes', ...this.planesWindow(planes, hf.res, key) };
    }
    if (layer === 'fields') {
      if (!this.rasterKeyValid(key, FIELDS_LODS)) return null;
      const planes = this.fieldsPlanes ?? (this.fieldsPlanes = await this.readFieldsPlanes());
      return { kind: 'planes', ...this.planesWindow(planes, hf.simRes, key) };
    }
    return null;
  }

  close(): void {
    this.bins.clear();
    this.biomePlanes = null;
    this.fieldsPlanes = null;
    this.rawLayers = null;
  }

  private rasterKeyValid(key: ChunkKey, lods: readonly number[]): boolean {
    const side = key.lod === 0 ? 2 : 1;
    return lods.includes(key.lod) && key.cx >= 0 && key.cz >= 0 && key.cx < side && key.cz < side;
  }

  /** (chunkRes+1)² clamp-extended window of a res×res f32 grid; coarse lods
   *  subsample the native lattice at stride lodStep^k with a centered offset
   *  (A11: the generated wire window is defined on the native texel-centered
   *  lattice). chunkRes follows the grid's texel (height res vs sim res). */
  private rasterWindowF32(src: Float32Array, res: number, key: ChunkKey): { res: number; heights: Float32Array } {
    const chunkRes = Math.round(CHUNK_METERS / (WORLD_SIZE / res));
    const stride = LOD_STEP ** key.lod;
    const off = stride >> 1;
    const resW = chunkRes + 1;
    const baseX = key.cx * chunkRes * stride;
    const baseZ = key.cz * chunkRes * stride;
    const out = new Float32Array(resW * resW);
    for (let j = 0; j < resW; j++) {
      const sz = Math.min(baseZ + j * stride + off, res - 1);
      const row = sz * res;
      for (let i = 0; i < resW; i++) {
        out[j * resW + i] = src[row + Math.min(baseX + i * stride + off, res - 1)] as number;
      }
    }
    return { res: resW, heights: out };
  }

  /** same window/subsample over a u8 plane stack */
  private planesWindow(planes: Uint8Array[], res: number, key: ChunkKey): { res: number; planes: Uint8Array[] } {
    const chunkRes = Math.round(CHUNK_METERS / (WORLD_SIZE / res));
    const stride = LOD_STEP ** key.lod;
    const off = stride >> 1;
    const resW = chunkRes + 1;
    const baseX = key.cx * chunkRes * stride;
    const baseZ = key.cz * chunkRes * stride;
    const out = planes.map((full) => {
      const o = new Uint8Array(resW * resW);
      for (let j = 0; j < resW; j++) {
        const sz = Math.min(baseZ + j * stride + off, res - 1);
        const row = sz * res;
        for (let i = 0; i < resW; i++) {
          o[j * resW + i] = full[row + Math.min(baseX + i * stride + off, res - 1)] as number;
        }
      }
      return o;
    });
    return { res: resW, planes: out };
  }

  /** one-time sim-res surface-fields build (FIELDS_PLANES order): moisture +
   *  flowStrength read back from the hydrology buffers, snow + rockExposure
   *  subsampled from the full-res biome planes (offset-centered stride). */
  private async readFieldsPlanes(): Promise<Uint8Array[]> {
    const hf = this.heightfield;
    const flow = hf.flow;
    if (!flow) throw new Error('GeneratedWorldSource: hydrology missing for fields planes');
    const res = hf.simRes;
    const n = res * res;
    const [moB, fsB] = await Promise.all([
      readBuffer(this.renderer, attrOf(flow.moisture), 0, n * 4),
      readBuffer(this.renderer, attrOf(flow.flowStrength), 0, n * 4),
    ]);
    const mo = new Float32Array(moB);
    const fs = new Float32Array(fsB);
    const biome = this.biomePlanes ?? (this.biomePlanes = await this.readBiomePlanes());
    const snowFull = biome[2] as Uint8Array;
    const rockFull = biome[3] as Uint8Array;
    const stride = Math.round(hf.res / res);
    const off = stride >> 1;
    const q8 = (v: number): number => Math.round(Math.min(1, Math.max(0, v)) * 255);
    const moisture = new Uint8Array(n);
    const flowStrength = new Uint8Array(n);
    const snow = new Uint8Array(n);
    const rock = new Uint8Array(n);
    for (let j = 0; j < res; j++) {
      const bRow = Math.min(j * stride + off, hf.res - 1) * hf.res;
      for (let i = 0; i < res; i++) {
        const k = j * res + i;
        moisture[k] = q8(mo[k] as number);
        flowStrength[k] = q8(fs[k] as number);
        const bi = bRow + Math.min(i * stride + off, hf.res - 1);
        snow[k] = snowFull[bi] as number;
        rock[k] = rockFull[bi] as number;
      }
    }
    return [moisture, flowStrength, snow, rock];
  }

  /** one-time biomeTex → CPU readback (lazy: only a biome consumer pays for it).
   *  rgba8 = (biomeId/8, snow, vegDensity, rockExposure); classId plane carries the
   *  DECODED small int so both sources serve ids, not unorm encodings. */
  private async readBiomePlanes(): Promise<Uint8Array[]> {
    const hf = this.heightfield;
    const tex = hf.biomeTex;
    if (!tex) throw new Error('GeneratedWorldSource: biomeTex missing');
    const res = hf.res;
    const packed = instancedArray(res * res, 'uint');
    const kernel = Fn(() => {
      const i = instanceIndex as unknown as NU;
      If(i.greaterThanEqual(uint(res * res)), () => {
        Return();
      });
      const t = textureLoad(tex, uvec2(i.mod(uint(res)), i.div(uint(res)))) as unknown as NV4;
      const q = (c: { mul: (n: number) => { round: () => { toUint: () => NU } } }): NU => c.mul(255).round().toUint();
      const word = q(t.x)
        .bitOr(q(t.y).shiftLeft(uint(8)))
        .bitOr(q(t.z).shiftLeft(uint(16)))
        .bitOr(q(t.w).shiftLeft(uint(24)));
      packed.element(i).assign(word);
    })().compute(res * res);
    kernel.setName('biomePlanesReadback');
    this.renderer.compute(kernel);
    const attr = (packed as unknown as { value: Parameters<typeof readBuffer>[1] }).value;
    const words = new Uint32Array(await readBuffer(this.renderer, attr, 0, res * res * 4));
    const classId = new Uint8Array(res * res);
    const vegDensity = new Uint8Array(res * res);
    const snow = new Uint8Array(res * res);
    const rockExposure = new Uint8Array(res * res);
    for (let i = 0; i < res * res; i++) {
      const w = words[i] as number;
      classId[i] = Math.round(((w & 0xff) * 8) / 255); // biomeId stored as id/8 unorm
      snow[i] = (w >>> 8) & 0xff;
      vegDensity[i] = (w >>> 16) & 0xff;
      rockExposure[i] = (w >>> 24) & 0xff;
    }
    return [classId, vegDensity, snow, rockExposure];
  }
}

// --- readback + binning ------------------------------------------------------------------

/** storage node → its backing attribute (scatter bufA/B and hydrology FloatBuffers
 *  share the instancedArray shape) */
function attrOf(node: ScatterLayer['bufA'] | FloatBuffer): Parameters<typeof readBuffer>[1] {
  return (node as unknown as { value: Parameters<typeof readBuffer>[1] }).value;
}

async function readLayer(renderer: Renderer, layer: ScatterLayer): Promise<RawScatterLayer> {
  const n = layer.count;
  if (n === 0) return { a: new Float32Array(0), b: new Float32Array(0), count: 0 };
  const [ab, bb] = await Promise.all([
    readBuffer(renderer, attrOf(layer.bufA), 0, n * 16),
    readBuffer(renderer, attrOf(layer.bufB), 0, n * 16),
  ]);
  return { a: new Float32Array(ab), b: new Float32Array(bb), count: n };
}

/** Partition one mixed-class layer into per-chunk record bins, scatter order preserved.
 *  A = (x, y, z, scale), B = (yaw, leanX, leanZ, idF); species = idF>>3, variant = idF&7. */
function binLayer(raw: RawScatterLayer): Map<number, RecordBin> {
  const chunkOf = (xw: number, zw: number): number => {
    const cx = Math.min(1, Math.max(0, Math.floor((xw + WORLD_HALF) / CHUNK_METERS)));
    const cz = Math.min(1, Math.max(0, Math.floor((zw + WORLD_HALF) / CHUNK_METERS)));
    return packChunkKey(0, cx, cz);
  };
  const counts = new Map<number, number>();
  for (let i = 0; i < raw.count; i++) {
    const k = chunkOf(raw.a[i * 4] as number, raw.a[i * 4 + 2] as number);
    counts.set(k, (counts.get(k) ?? 0) + 1);
  }
  const bins = new Map<number, RecordBin>();
  for (const [k, n] of counts) {
    bins.set(k, {
      count: 0,
      x: new Float32Array(n),
      z: new Float32Array(n),
      species: new Uint8Array(n),
      scale: new Float32Array(n),
      variant: new Uint8Array(n),
      y: new Float32Array(n),
      yaw: new Float32Array(n),
      leanX: new Float32Array(n),
      leanZ: new Float32Array(n),
      xw: new Float32Array(n),
      zw: new Float32Array(n),
    });
  }
  for (let i = 0; i < raw.count; i++) {
    const xw = raw.a[i * 4] as number;
    const zw = raw.a[i * 4 + 2] as number;
    const k = chunkOf(xw, zw);
    const bin = bins.get(k) as RecordBin;
    const d = bin.count++;
    const { cx, cz } = keyOfPacked(k);
    bin.x[d] = xw - (-WORLD_HALF + cx * CHUNK_METERS);
    bin.z[d] = zw - (-WORLD_HALF + cz * CHUNK_METERS);
    const idF = Math.round(raw.b[i * 4 + 3] as number);
    bin.species[d] = idF >> 3;
    bin.variant[d] = idF & 7;
    bin.scale[d] = raw.a[i * 4 + 3] as number;
    bin.y[d] = raw.a[i * 4 + 1] as number;
    bin.yaw[d] = raw.b[i * 4] as number;
    bin.leanX[d] = raw.b[i * 4 + 1] as number;
    bin.leanZ[d] = raw.b[i * 4 + 2] as number;
    bin.xw[d] = xw;
    bin.zw[d] = zw;
  }
  return bins;
}

/** inverse of packChunkKey (Lac1.ts): lod·2^42 + (cx+2^20)·2^21 + (cz+2^20). */
function keyOfPacked(packed: number): ChunkKey {
  const lod = Math.floor(packed / 2 ** 42);
  const rest = packed - lod * 2 ** 42;
  const cx = Math.floor(rest / 2 ** 21) - 2 ** 20;
  const cz = (rest % 2 ** 21) - 2 ** 20;
  return { lod, cx, cz };
}
