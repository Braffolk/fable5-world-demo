/**
 * WorldSource — the generic streamed-world data contract (SPEC-STREAMING-WORLD §2).
 * Pure types, no three.js/DOM/GPU deps: payloads are DECODED f32/u8 — quantization,
 * delta and deflate are LAC1 wire concerns that exist only inside RemoteWorldSource
 * (worker-side) and Lac1.ts. `coverage()` is authoritative absence: dry water,
 * empty boulders and outside-Estonia are one mechanism (null = no chunk, no request).
 */

export type LayerName =
  | 'height'
  | 'biome'
  | 'water'
  | 'trees'
  | 'soil'
  | 'understory'
  | 'debris'
  | 'boulders'
  | 'canopy'
  // generated-source record layers (F2: first-class streams; Estonia derives
  // equivalents from its understory/debris guidance planes at S9)
  | 'extras'
  | 'stones';

export interface ChunkKey {
  lod: number;
  cx: number;
  cz: number;
}

export type ChunkPayload =
  | { kind: 'height'; res: number; heights: Float32Array } // meters, absolute, incl. apron row/col (water: NaN = dry texel)
  | { kind: 'planes'; res: number; planes: Uint8Array[] }
  | {
      kind: 'records';
      count: number;
      cols: {
        x: Float32Array; // chunk-local meters
        z: Float32Array;
        species: Uint8Array;
        scale: Float32Array;
        variant: Uint8Array;
        // OPTIONAL exactness columns — provided when the source knows them
        // (GeneratedWorldSource passes scatter's exact values through; remote omits):
        y?: Float32Array;
        yaw?: Float32Array;
        leanX?: Float32Array;
        leanZ?: Float32Array;
        // exact game-absolute positions (the S2 byte-identity gate): re-anchoring
        // a f32 world coordinate to chunk-local f32 and back is NOT lossless (±half
        // an ulp at footprint magnitude), so the generated source passes the original
        // absolute f32 through. Consumers use xw/zw when present, else origin + x/z.
        xw?: Float32Array;
        zw?: Float32Array;
      };
    };

/** One indexed chunk: existence proof + fetch metadata (size for prioritisation,
 *  hash64 for the content-addressed URL / cache key). */
export interface ChunkRef {
  lod: number;
  cx: number;
  cz: number;
  size: number;
  hash64: bigint;
}

export type RecordDtype = 'u8' | 'u16' | 'u32' | 'f32';

/** Frozen grid contract (manifest `anchor`/`chunkMeters`/`chunkRes`/`lodStep`).
 *  gameX = E - anchorE; gameZ = anchorN - N; LOD k texel = lodStep^k m,
 *  chunk footprint = chunkMeters * lodStep^k m; rasters are (chunkRes/texelMeters + 1)²
 *  with the far row/col duplicating the east/south neighbor (apron). */
export interface WorldGrid {
  anchorE: number;
  anchorN: number;
  chunkMeters: number;
  chunkRes: number;
  lodStep: number;
  /** game-space position of chunk (0,0)'s min corner. Estonia: (0,0) — game coords
   *  are anchored so chunks tile from the anchor. Generated: (−WORLD_HALF, −WORLD_HALF)
   *  — the world is centered on the game origin, so tiling from the min corner keeps
   *  the 4096 m world = 2×2 LOD0 chunks + ONE LOD1 chunk. */
  originX: number;
  originZ: number;
}

export interface WorldLayerMeta {
  enc: number; // 1 = quant16+2D-delta, 2 = u8 planes, 3 = record SoA
  lods: number[];
  chunkCount: number;
  texelMeters?: number; // raster layers whose texel differs from the height grid
  planes?: readonly string[]; // enc 2
  columns?: readonly (readonly [string, RecordDtype])[]; // enc 3
}

export interface SpeciesEntry {
  code: string;
  english: string;
  latin: string;
  leaf: 'conifer' | 'broadleaf' | 'snag';
  ref_height_m: number;
}

export interface CommunityEntry {
  name: string;
  base_density: number;
  palette: string[];
  moisture?: string;
}

export interface WorldDictionaries {
  species: Map<number, SpeciesEntry>;
  understory: Map<number, CommunityEntry>;
  debris: Map<number, CommunityEntry>;
}

export interface WorldManifest {
  grid: WorldGrid;
  layers: Partial<Record<LayerName, WorldLayerMeta>>;
  dictionaries: WorldDictionaries;
  /** Authoritative absence: null = the chunk does not exist (no request needed). */
  coverage(layer: LayerName, key: ChunkKey): ChunkRef | null;
  /** Enumerate every existing chunk of a layer at one lod (boot fills iterate this;
   *  runtime residency uses ring arithmetic + coverage() point queries instead). */
  chunks(layer: LayerName, lod: number): ChunkKey[];
}

export interface WorldSource {
  open(progress?: (frac: number, msg: string) => void): Promise<WorldManifest>;
  fetch(layer: LayerName, key: ChunkKey, signal?: AbortSignal): Promise<ChunkPayload | null>;
  close(): void;
}
