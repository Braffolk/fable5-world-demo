/**
 * Boot-time derived-data cache (the "DDC"): IndexedDB store for the three expensive
 * deterministic forest-boot artifacts — per-species crown voxelizations (~34 s at 200k),
 * LOD DAG builds (~15 s) and the fartiles splat (~10 s) — cutting a warm boot from ~71 s
 * to the residual GPU/setup work. Correctness model:
 *
 *  - The cache KEY includes (a) a runtime FNV-1a hash of the builder SOURCES (vite `?raw`
 *    imports), so ANY edit to a builder auto-invalidates — a stale cache would silently
 *    poison perf A/Bs; (b) every boot param that feeds the builds (seed, counts, LOD
 *    config, anchor, caps); (c) a manual CACHE_REV for transitive-dep changes the source
 *    hash cannot see (e.g. DagCommon partition changes).
 *  - Values are stored via structured clone. CrownVoxelization brick grids are packed to
 *    occupied-only struct-of-arrays first (Float64 lanes ⇒ bit-exact reconstruction; the
 *    full JS-object grids are tens of millions of objects for fartiles — the same ~2 GB
 *    that showed up as post-boot garbage). The append paths only ever read occupied
 *    indices (verified appendVoxelCrown / appendVoxelCrownPyramid), so unpack leaves
 *    empty slots as array holes.
 *  - `?nocache=1` bypasses (build path untouched); `?cacheclear=1` purges then rebuilds.
 *  - GATE for any change here: byte-identical first-frame shot cold vs warm.
 */
import type { CrownVoxelization, PreparedVoxelCrown, VoxelLevel, VoxelBlock } from '../build/VoxelizeCrown';
import type { FarTileBuild } from './FarTiles';
import { BRICK_WORDS, readBrick, writeBrick } from '../voxel/VoxelBrickCore';
// builder sources — hashed into the cache key (auto-invalidation on edit)
import srcVoxelize from '../build/VoxelizeCrown.ts?raw';
import srcBuildDag from '../build/BuildDag.ts?raw';
import srcBuildAgg from '../build/BuildAggregateDag.ts?raw';
import srcBuildCrownLod from '../build/BuildCrownLodDag.ts?raw';
import srcFarTiles from './FarTiles.ts?raw';
import srcFarTilesSplat from './FarTilesSplat.ts?raw';
import srcVoxelBrick from '../voxel/VoxelBrick.ts?raw';
import srcVoxelBrickCore from '../voxel/VoxelBrickCore.ts?raw';
import srcClusterize from '../build/Clusterize.ts?raw';
// crown-LOD ladder GEN (keep-mask hash + per-rung λ/growth/needle-μ/rows dispatch)
// lives in TreeBuilder; the actual survivor MESHING (area-preserving width growth,
// blade-row + needle-μ + stem coarsening) lives in LeafMesh — a change to EITHER
// reshapes the ladder rungs and must invalidate the cached crown DAG.
import srcTreeBuilder from '../../vegetation/TreeBuilder.ts?raw';
import srcLeafMesh from '../../vegetation/LeafMesh.ts?raw';

/** bump on changes to builder TRANSITIVE deps not covered by the ?raw hash list
 *  (DagCommon, VegLibrary geometry gen, registry append semantics).
 *  rev 2 (2026-07-04): brick pack format f64-lanes → 9×u32 GPU words (heap cut).
 *  rev 3 (2026-07-05): crown aggregate DAG → LOD0-only (maxLevels 1) — the coarse
 *  aggregate mid-levels are dead post-crownlod0/castShadows:false; drop build+mem.
 *  rev 4 (2026-07-08): crown-LOD Phase 2 — crown DAG is now a MULTI-LEVEL ladder
 *  (BuildCrownLodDag, fed the pre-pruned CROWN_LOD_SCHEDULE rungs); the 'dags'
 *  payload for leaf crowns changed shape (LOD0-only → 4-rung anchor-chain).
 *  rev 5 (2026-07-09): crown-LOD area preservation + intra-element coarsening —
 *  ladder rungs now grow survivors (width ×1/λ) and coarsen them (blade rows /
 *  needle-μ / stem segs), so every coarse rung's GEOMETRY changed; schedule is now
 *  a per-kind rung table (+ LeafMesh.ts joins SRC_HASH).
 *  rev 6 (2026-07-09): crown-LOD ladder deepened 4→6 rungs (coarsest ~0.41→~0.13-0.15×
 *  of LOD0) — the crown DAG gains 2 anchor-chain levels (payload SHAPE change), and
 *  CROWN_ENGAGE_FRAC re-spaced to 5 fractions. Schedule count is in the params key too,
 *  but the DAG-shape change warrants the explicit rev bump.
 *  rev 7 (2026-07-09): crown default errorK 1.0→0.4 (user-approved bake) + per-rung
 *  HANDOFF WIDTH-OVERSHOOT ramp (CrownLodRung.widthBoost; rungs 4/5 ×1.15/1.3) — deep
 *  rung survivor GEOMETRY widens (walks mesh density up to the voxel at the 60 m handoff).
 *  Both ride the params key (crownLodErrorK + crownLod=CROWN_LOD_SCHEDULE) and the
 *  widthMul multiply is in srcTreeBuilder's SRC_HASH; the rev bump is belt-and-suspenders
 *  since the schedule table lives in VegLibrary (not in the ?raw SRC_HASH list).
 *  rev 8 (2026-07-09): crown VOXELIZER de-fattening — OCC_SOLID 0.02→0.15 + support-
 *  hysteresis erosion (VoxelizeCrown). Occupancy bits (crown silhouette) now differ, so any
 *  cached fat crown is stale. srcVoxelize's ?raw SRC_HASH already covers the edit; this rev is
 *  belt-and-suspenders + also invalidates the ?voxocc-DEFAULT crowns (default value changed,
 *  and voxOcc rides the params key so a non-default ?voxocc keyed separately regardless). */
const CACHE_REV = 8;

const DB_NAME = 'laas-bootcache';
const STORE = 'artifacts';

function fnv1a(s: string, h = 0x811c9dc5): number {
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

const SRC_HASH = [srcVoxelize, srcBuildDag, srcBuildAgg, srcBuildCrownLod, srcFarTiles, srcFarTilesSplat, srcVoxelBrick, srcVoxelBrickCore, srcClusterize, srcTreeBuilder, srcLeafMesh]
  .reduce((h, s) => fnv1a(s, h), 0x811c9dc5)
  .toString(16);

export interface BootCacheOpts {
  /** every build-affecting param — JSON-stringified into the key. */
  params: Record<string, unknown>;
}

export class BootCache {
  private readonly keyBase: string;
  readonly enabled: boolean;
  private readonly clearFirst: boolean;
  private db: IDBDatabase | null = null;

  constructor(opts: BootCacheOpts) {
    const q = new URLSearchParams(window.location.search);
    this.enabled = q.get('nocache') !== '1' && typeof indexedDB !== 'undefined';
    this.clearFirst = q.get('cacheclear') === '1';
    this.keyBase = `r${CACHE_REV}:${SRC_HASH}:${fnv1a(JSON.stringify(opts.params)).toString(16)}`;
  }

  private async open(): Promise<IDBDatabase | null> {
    if (!this.enabled) return null;
    if (this.db) return this.db;
    try {
      this.db = await new Promise<IDBDatabase>((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, 1);
        req.onupgradeneeded = () => {
          if (!req.result.objectStoreNames.contains(STORE)) req.result.createObjectStore(STORE);
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error ?? new Error('idb open failed'));
      });
      if (this.clearFirst) {
        await this.txn('readwrite', (s) => s.clear());
        // eslint-disable-next-line no-console
        console.log('[bootcache] cleared (?cacheclear=1)');
      }
      return this.db;
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn('[bootcache] disabled (open failed):', e);
      this.db = null;
      return null;
    }
  }

  private async txn<T>(mode: IDBTransactionMode, f: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
    const db = this.db;
    if (!db) throw new Error('bootcache: no db');
    return new Promise<T>((resolve, reject) => {
      const t = db.transaction(STORE, mode);
      const req = f(t.objectStore(STORE));
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error ?? new Error('idb txn failed'));
    });
  }

  /** get an artifact group; null on miss/bypass/error (caller falls back to building). */
  async get<T>(name: string): Promise<T | null> {
    const db = await this.open();
    if (!db) return null;
    const t0 = performance.now();
    try {
      const v = (await this.txn('readonly', (s) => s.get(`${this.keyBase}/${name}`))) as T | undefined;
      if (v !== undefined) {
        // eslint-disable-next-line no-console
        console.log(`[bootcache] HIT ${name} (read ${(performance.now() - t0).toFixed(0)} ms)`);
        return v;
      }
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn(`[bootcache] get ${name} failed:`, e);
    }
    // eslint-disable-next-line no-console
    console.log(`[bootcache] miss ${name}`);
    return null;
  }

  /** chunked list storage: one IDB record per item (Chrome's structured-clone serializer
   *  OOMs on a single multi-hundred-MB graph — seen on the 40-DagBuild array). */
  async getMany<T>(name: string): Promise<T[] | null> {
    const db = await this.open();
    if (!db) return null;
    const t0 = performance.now();
    try {
      const n = (await this.txn('readonly', (s) => s.get(`${this.keyBase}/${name}/count`))) as number | undefined;
      if (typeof n !== 'number') {
        // eslint-disable-next-line no-console
        console.log(`[bootcache] miss ${name}`);
        return null;
      }
      const out: T[] = new Array(n);
      for (let i = 0; i < n; i++) {
        const v = (await this.txn('readonly', (s) => s.get(`${this.keyBase}/${name}/${i}`))) as T | undefined;
        if (v === undefined) {
          // eslint-disable-next-line no-console
          console.log(`[bootcache] miss ${name} (chunk ${i}/${n} absent)`);
          return null;
        }
        out[i] = v;
      }
      // eslint-disable-next-line no-console
      console.log(`[bootcache] HIT ${name} (${n} chunks, read ${(performance.now() - t0).toFixed(0)} ms)`);
      return out;
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn(`[bootcache] getMany ${name} failed:`, e);
      return null;
    }
  }

  async putMany(name: string, items: unknown[]): Promise<void> {
    const db = await this.open();
    if (!db) return;
    const t0 = performance.now();
    try {
      for (let i = 0; i < items.length; i++) {
        await this.txn('readwrite', (s) => s.put(items[i], `${this.keyBase}/${name}/${i}`));
      }
      // count LAST — a partial write (abort/OOM mid-way) leaves no count record ⇒ clean miss.
      await this.txn('readwrite', (s) => s.put(items.length, `${this.keyBase}/${name}/count`));
      // eslint-disable-next-line no-console
      console.log(`[bootcache] stored ${name} (${items.length} chunks, ${(performance.now() - t0).toFixed(0)} ms)`);
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn(`[bootcache] putMany ${name} failed:`, e);
    }
  }

  /** store an artifact group (best-effort; a failed put only costs the next boot a rebuild). */
  async put(name: string, value: unknown): Promise<void> {
    const db = await this.open();
    if (!db) return;
    const t0 = performance.now();
    try {
      await this.txn('readwrite', (s) => s.put(value, `${this.keyBase}/${name}`));
      // eslint-disable-next-line no-console
      console.log(`[bootcache] stored ${name} (${(performance.now() - t0).toFixed(0)} ms)`);
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn(`[bootcache] put ${name} failed:`, e);
    }
  }
}

// ── CrownVoxelization compact pack ────────────────────────────────────────────
// BrickCPU → BRICK_WORDS (9) × u32 GPU words per occupied brick (the SAME record the
// append path copies straight into gpu.voxelBricks — writeBrick codec). Only OCCUPIED
// bricks are stored; unpack leaves empty grid slots as holes (append paths never read
// them). 36 B/brick vs the old 108 B (2×u32 + 12×f64) — cuts the fartile pack from
// ~866 MB to ~288 MB and lets appendPackedCrown skip per-brick BrickCPU objects.
// "Lossless enough": the words ARE what the GPU renders (byte-identical to a fresh
// build's writeBrick output); only build-time intermediates (fartile splat INPUT)
// see oct-normal / rgba8-albedo quantization — a cold-only, >280 m far-field effect.

interface PackedGrid {
  occupied: Uint32Array;
  words: Uint32Array; // BRICK_WORDS per brick — gpu.voxelBricks record, occupied[] order
  totalBricks: number;
}

export interface PackedLevel extends PackedGrid {
  level: number;
  brickGrid: { x: number; y: number; z: number };
  cellSize: number;
  geomError: number;
  blocks: VoxelBlock[]; // plain data (Sphere = plain xyzr) — structured clone handles it
}

interface PackedVox {
  grid: PackedGrid;
  brickGrid: CrownVoxelization['brickGrid'];
  cellGrid: CrownVoxelization['cellGrid'];
  origin: CrownVoxelization['origin'];
  cellSize: number;
  stats: CrownVoxelization['stats'];
  levels?: PackedLevel[];
}

export interface PackedPreparedCrown {
  vox: PackedVox;
  brickCount: number;
  clusterCount: number;
  dagLinkCount: number;
}

export interface PackedFarTile {
  center: [number, number, number];
  prep: PackedPreparedCrown;
}

type BrickList = CrownVoxelization['bricks'];

function packGrid(bricks: BrickList, occupied: number[]): PackedGrid {
  const n = occupied.length;
  const occ = new Uint32Array(occupied);
  const words = new Uint32Array(n * BRICK_WORDS);
  for (let i = 0; i < n; i++) {
    const b = bricks[occ[i] as number];
    if (!b) throw new Error('bootcache: occupied index out of range');
    writeBrick(words, i, b);
  }
  return { occupied: occ, words, totalBricks: bricks.length };
}

/** reconstruct a SPARSE BrickCPU grid from the packed words (holes for empty slots).
 *  Compat path — the world append reads `words` directly (appendPackedCrown); this is
 *  for ForestScene's BrickCPU appendVoxelCrown + probes. Values are word-precision
 *  (oct-normal / rgba8-albedo) = exactly what a fresh build's writeBrick emits. */
function unpackGrid(p: PackedGrid): { bricks: BrickList; occupied: number[] } {
  const bricks: BrickList = new Array(p.totalBricks);
  const occupied: number[] = new Array(p.occupied.length);
  for (let i = 0; i < p.occupied.length; i++) {
    const bi = p.occupied[i] as number;
    occupied[i] = bi;
    bricks[bi] = readBrick(p.words, i);
  }
  return { bricks, occupied };
}

function packVox(vox: CrownVoxelization): PackedVox {
  return {
    grid: packGrid(vox.bricks, vox.occupied),
    brickGrid: vox.brickGrid,
    cellGrid: vox.cellGrid,
    origin: vox.origin,
    cellSize: vox.cellSize,
    stats: vox.stats,
    levels: vox.levels?.map((l) => ({
      ...packGrid(l.bricks, l.occupied),
      level: l.level,
      brickGrid: l.brickGrid,
      cellSize: l.cellSize,
      geomError: l.geomError,
      blocks: l.blocks,
    })),
  };
}

function unpackVox(p: PackedVox): CrownVoxelization {
  const g = unpackGrid(p.grid);
  const levels: VoxelLevel[] | undefined = p.levels?.map((l) => {
    const lg = unpackGrid(l);
    return {
      level: l.level,
      bricks: lg.bricks,
      occupied: lg.occupied,
      brickGrid: l.brickGrid,
      cellSize: l.cellSize,
      geomError: l.geomError,
      blocks: l.blocks,
    };
  });
  const out: CrownVoxelization = {
    bricks: g.bricks,
    occupied: g.occupied,
    brickGrid: p.brickGrid,
    cellGrid: p.cellGrid,
    origin: p.origin,
    cellSize: p.cellSize,
    stats: p.stats,
  };
  if (levels) out.levels = levels;
  return out;
}

export function packPreparedCrown(prep: PreparedVoxelCrown): PackedPreparedCrown {
  return {
    vox: packVox(prep.vox),
    brickCount: prep.brickCount,
    clusterCount: prep.clusterCount,
    dagLinkCount: prep.dagLinkCount,
  };
}

export function unpackPreparedCrown(p: PackedPreparedCrown): PreparedVoxelCrown {
  return {
    vox: unpackVox(p.vox),
    brickCount: p.brickCount,
    clusterCount: p.clusterCount,
    dagLinkCount: p.dagLinkCount,
  };
}

export function packFarTiles(tiles: FarTileBuild[]): PackedFarTile[] {
  return tiles.map((t) => ({ center: t.center, prep: packPreparedCrown(t.prep) }));
}

export function unpackFarTiles(packed: PackedFarTile[]): FarTileBuild[] {
  return packed.map((t) => ({ center: t.center, prep: unpackPreparedCrown(t.prep) }));
}

