/**
 * WorldRegistry (N1-C4, docs/NANITE.md) — builds THE GeometryRegistry from the
 * real world's opaque pools behind ?nanite=1. No rendering changes yet (that
 * is N2/N3); this wires the content contract end to end and measures it.
 *
 * Opaque-part policy (PATH UNIFICATION AUDIT):
 *  - tree pools (cls 0–5): bark part of r0/r1/r2 as a discrete LOD chain
 *    (switch at R0_FAR=26 m, R1_FAR=150 m — Forests ring radii); foliage
 *    CARDS + hero mesh leaves DEFERRED to N9 (alpha/leaf path).
 *  - shrubs (8–10): bark part, single ring.
 *  - ferns/flowers (11–14): leafy card geometry — DEFERRED to N9 entirely.
 *  - logs/stumps (16–17), branches (23): deadwood, r1 (branch r2 is a clone
 *    that exists only for indirect-slot bookkeeping — one registration).
 *  - boulders/slabs (18–19), stones L/M (20–21): rock, r1→r2 LOD chain at
 *    EX_R1_FAR=120 m; StoneS (22): single ring.
 *  - terrain: ONE heightfield source over the full field (winQuads 7,
 *    partial edge windows), single identity instance.
 *  - GroundRing grass/debris: clipmap-instanced (not boot-static scatter) —
 *    stays on its bespoke path until N6/N10 per the audit table.
 *
 * Instances: scatter layers are GPU-resident MIXED-class buffers (idF =
 * cls·8+variant in B.w). C4 reads each layer back ONCE at boot (placements
 * are static) and partitions on CPU into per-mesh contiguous streams —
 * deterministic, order-preserving, and the registry uploads one packed blob.
 * Instances of deferred classes are counted and skipped.
 */

import type { Renderer, StorageBufferAttribute, StorageBufferNode } from 'three/webgpu';
import type { BufferGeometry } from 'three';
import type { ScatterLayer, ScatterResult } from '../gpu/passes/Scatter';
import { VegClass } from '../gpu/passes/Scatter';
import type { VegLib, PoolPart } from '../vegetation/VegLibrary';
import type { Heightfield } from '../world/Heightfield';
import { WORLD_SIZE } from '../world/WorldConst';
import { type DagBuild, type DagCluster, buildDag } from './BuildDag';
import { DagBuildWorker, type HeightDagResult } from './DagWorkerClient';
import { TerrainStreamer, buildTerrainTile, type TileBuildDeps, type TileBuildStats } from './TerrainStreamer';
import {
  type BuildReport,
  DAG_VERT_STRIDE,
  type ExplicitSource,
  GeometryRegistry,
  type MaterialClassId,
  type MeshHandle,
  type TransformChannel,
  explicitToDagVerts,
} from './GeometryRegistry';
import { readBuffer } from './Tsl';

/** Forests ring radii (Forests.ts) — discrete LOD switch distances until N8 */
const R0_FAR = 26;
const R1_FAR = 150;
const EX_R1_FAR = 120;
/** tree real-geometry envelope: R2_FAR 460 + BAND2 36 (Forests.ts) — beyond
 *  it the old path shows impostors, the hybrid's sanctioned far field (the
 *  N2 instance cull drops what impostors own; the N8 DAG retires this) */
const TREE_GEO_FAR = 496;
/** terrain window size: 7 quads → 98 tris, divides 4095 exactly (4096² field) */
const TERRAIN_WIN_QUADS = 7;

const TREE_MAX_CLS = 5;
const SHRUB_CLASSES: ReadonlySet<number> = new Set([
  VegClass.BushHazel,
  VegClass.BushPink,
  VegClass.Juniper,
]);
const DEADWOOD_CLASSES: ReadonlySet<number> = new Set([
  VegClass.Log,
  VegClass.Stump,
  VegClass.Branch,
]);
const ROCK_CLASSES: ReadonlySet<number> = new Set([
  VegClass.Boulder,
  VegClass.Slab,
  VegClass.StoneL,
  VegClass.StoneM,
  VegClass.StoneS,
]);

export interface WorldRegistryResult {
  registry: GeometryRegistry;
  report: BuildReport;
  /** ms: scatter readback, CPU partition, terrain minMax scan, registry build */
  readbackMs: number;
  partitionMs: number;
  terrainMs: number;
  buildMs: number;
  totalMs: number;
  /** instance count whose class is deferred (cards/leafy) */
  deferredInstances: number;
  /** per-part notes of geometry NOT migrated yet */
  deferred: string[];
  /** N8-D1: meshes given a continuous-LOD DAG + total sync build ms (0 = none) */
  dagMeshes: number;
  dagBuildMs: number;
  dagTris: number;
  /** N8-D2 Stage 2b-3 (D-N39): the live clipmap streamer in ?nanitedclip mode —
   *  TerrainScene drives streamer.update(camXZ) per frame. Absent otherwise. */
  terrainStreamer?: TerrainStreamer;
}

function attrOf(node: StorageBufferNode<'vec4'>): StorageBufferAttribute {
  return (node as unknown as { value: StorageBufferAttribute }).value;
}

/** BufferGeometry → packed ExplicitSource (vdata vec4 → 4×u8 word) */
export function geometryToSource(geo: BufferGeometry): ExplicitSource {
  const pos = geo.attributes.position;
  if (!pos || pos.itemSize !== 3) throw new Error('WorldRegistry: geometry lacks stride-3 positions');
  if (!geo.attributes.normal) geo.computeVertexNormals();
  const nrm = geo.attributes.normal;
  if (!nrm) throw new Error('WorldRegistry: normals missing after compute');
  const positions = pos.array as Float32Array;
  const normals = nrm.array as Float32Array;
  const vCount = pos.count;
  const uvAttr = geo.attributes.uv;
  let uvs: Float32Array | undefined;
  if (uvAttr && uvAttr.itemSize === 2) uvs = uvAttr.array as Float32Array;
  const datAttr = geo.attributes.vdata;
  let vdata: Uint32Array | undefined;
  if (datAttr && datAttr.itemSize === 4) {
    const d = datAttr.array as Float32Array;
    vdata = new Uint32Array(vCount);
    for (let v = 0; v < vCount; v++) {
      const q = (i: number): number =>
        Math.round(Math.max(0, Math.min(1, d[v * 4 + i] as number)) * 255);
      vdata[v] = (q(0) | (q(1) << 8) | (q(2) << 16) | (q(3) << 24)) >>> 0;
    }
  }
  const idx = geo.index;
  if (!idx) throw new Error('WorldRegistry: geometry not indexed');
  const indices =
    idx.array instanceof Uint32Array ? idx.array : new Uint32Array(idx.array as ArrayLike<number>);
  return { kind: 'mesh', positions, normals, uvs, vdata, indices };
}

/**
 * Material classes whose resolve port has LANDED (N4 chunk state) — the
 * default migration set for the full-frame mode (`?nanite=1`). Grows per
 * N4 chunk: terrain (C1) → rock (C2) → bark+deadwood (C3). Override with
 * `?naniteclasses=csv|all`; nanitedbg views default to all (pipeline
 * probes want the whole registry).
 */
export const PORTED_CLASSES: readonly MaterialClassId[] = ['terrain', 'rock', 'bark', 'deadwood'];

/**
 * Material class a pool's OPAQUE part (parts[0]) migrates as, or null while
 * it stays old-path. The old pipeline's camera-draw suppression (N4-C0,
 * D-N19) keys off the SAME predicate, so the migrated set and the
 * suppressed set can never drift apart.
 */
export function migratedMatClass(cls: number): MaterialClassId | null {
  return classPolicy(cls)?.matClass ?? null;
}

/** trunk-wind profile (Forests.ts windBind): 0 tree (k1,f1,h0=6), 1 snag
 *  (k.45,f.8,h6), 2 shrub (k1,f1.8,h.9). Packed into matParam high byte; the
 *  fetch reads it for the 'trunk' channel (rigid classes ignore it). */
function windProfile(cls: number): number {
  if (cls === 5) return 1; // snag species
  if (cls <= TREE_MAX_CLS) return 0; // canopy trees 0–4
  if (SHRUB_CLASSES.has(cls)) return 2; // understory shrubs
  return 0; // rigid (deadwood/rock) — unused
}

function classPolicy(
  cls: number,
): { matClass: MaterialClassId; channel: TransformChannel; lodDist: number; swayPad: number } | null {
  // swayPad = conservative world-space max wind displacement at strength 1
  // (F6; bounds derived from Wind.ts vegWindOffset term-by-term, e=g=1):
  // lean ≤ 2.46k + sway 0.8k + swayX 0.45k + branch 0.30k + flutter 0.17k,
  // prof capped 1.6 → trees (k=1) ≈ 3.8 m; snags k=0.45 → 1.7 m; understory
  // prof ≤ ~1.0 at its low knee/height → 2.4 m. Deadwood/stones are rigid.
  if (cls === 5) return { matClass: 'bark', channel: 'trunk', lodDist: R1_FAR, swayPad: 1.7 };
  if (cls <= TREE_MAX_CLS) return { matClass: 'bark', channel: 'trunk', lodDist: R1_FAR, swayPad: 3.8 };
  if (SHRUB_CLASSES.has(cls)) return { matClass: 'bark', channel: 'trunk', lodDist: 0, swayPad: 2.4 };
  if (DEADWOOD_CLASSES.has(cls)) return { matClass: 'deadwood', channel: 'rigid', lodDist: EX_R1_FAR, swayPad: 0 };
  if (ROCK_CLASSES.has(cls)) return { matClass: 'rock', channel: 'rigid', lodDist: EX_R1_FAR, swayPad: 0 };
  return null; // ferns/flowers — leafy, N9
}

async function readLayer(
  renderer: Renderer,
  layer: ScatterLayer,
): Promise<{ a: Float32Array; b: Float32Array; count: number }> {
  const n = layer.count;
  if (n === 0) return { a: new Float32Array(0), b: new Float32Array(0), count: 0 };
  const [ab, bb] = await Promise.all([
    readBuffer(renderer, attrOf(layer.bufA), 0, n * 16),
    readBuffer(renderer, attrOf(layer.bufB), 0, n * 16),
  ]);
  return { a: new Float32Array(ab), b: new Float32Array(bb), count: n };
}

export async function buildWorldRegistry(input: {
  renderer: Renderer;
  hf: Heightfield;
  scatter: ScatterResult;
  lib: VegLib;
  counters?: Record<string, number>;
  /** D-N19 incremental migration: only these material classes register +
   *  raster (their old camera draws get suppressed); omitted = all opaque */
  classes?: ReadonlySet<MaterialClassId>;
  /** N8-D1: explicit classes to give a continuous-LOD DAG (rock/bark/deadwood).
   *  Built SYNC here (D1d Workerizes per D-N30); terrain is never DAG'd in D1. */
  dag?: ReadonlySet<MaterialClassId>;
  /** N8-D2b: when > 0, register terrain as an adaptive LOD DAG built on a
   *  gridN² (power-of-two) SUBSAMPLE of the field instead of the discrete
   *  window grid. SYNC at boot (≈1 s @256², ≈5 s @512²) — full-res needs the
   *  D1d Worker. Texel coords are packed so the GPU still reads the full-res
   *  heightTex. 0 = the discrete window path (default). */
  dagTerrainGridN?: number;
  /** N8-D2 (D-N38): split the terrain DAG into T×T independent tiles, each gridN²
   *  over its texel sub-region. Tile perimeters auto-lock (mesh boundary) ⇒ seams
   *  are crack-free. 1 (default) = a single DAG. >1 is the path to streamed full-res. */
  dagTerrainTiles?: number;
  /** N8-D2 Stage 2b-1 (D-N39): route the terrain tiles through the streaming tile
   *  POOL (reserveTilePool/attachHeightDagTile) instead of per-tile registerHeightDag
   *  +attachHeightDag. GPU-render parity with the per-tile path; the foundation the
   *  per-frame clipmap streamer (2b-2/2b-3) builds on. Default false (per-tile path). */
  dagTerrainPool?: boolean;
  /** N8-D2 Stage 2b-2 (D-N39): render terrain as a geometry CLIPMAP — concentric
   *  same-gridN rings at doubling stride, centered on the field (boot-static; 2b-3
   *  re-centers on the live camera). True full-res at the center, coarse to the
   *  field edge, bounded resident set. Implies the pool. Default false. */
  dagTerrainClip?: boolean;
  /** N8-D1d: numeric world seed (WorldSeed.seed) → the terrain-DAG cache key.
   *  The heights are deterministic in the seed, so a cached DAG loads instantly
   *  (boot renders the DAG, no fallback). Omit to disable caching (always build). */
  seed?: number;
}): Promise<WorldRegistryResult> {
  const {
    renderer,
    hf,
    scatter,
    lib,
    counters,
    classes,
    dag,
    dagTerrainGridN,
    dagTerrainTiles,
    dagTerrainPool,
    dagTerrainClip,
    seed,
  } = input;
  const inSet = (c: MaterialClassId): boolean => !classes || classes.has(c);
  const t0 = performance.now();
  const deferred: string[] = [];

  // ---- scatter readback (placements are boot-static) ------------------------
  const layers = await Promise.all([
    readLayer(renderer, scatter.trees),
    readLayer(renderer, scatter.understory),
    readLayer(renderer, scatter.extras),
    readLayer(renderer, scatter.stones),
  ]);
  const tRead = performance.now();

  // ---- partition by idF = cls·8 + variant (order-preserving) ----------------
  const perId = new Map<number, { a: Float32Array; b: Float32Array; fill: number }>();
  const idCounts = new Map<number, number>();
  for (const layer of layers) {
    for (let i = 0; i < layer.count; i++) {
      const id = Math.round(layer.b[i * 4 + 3] as number);
      idCounts.set(id, (idCounts.get(id) ?? 0) + 1);
    }
  }
  for (const [id, n] of idCounts) {
    perId.set(id, { a: new Float32Array(n * 4), b: new Float32Array(n * 4), fill: 0 });
  }
  for (const layer of layers) {
    for (let i = 0; i < layer.count; i++) {
      const id = Math.round(layer.b[i * 4 + 3] as number);
      const s = perId.get(id);
      if (!s) continue;
      const d = s.fill * 4;
      s.a[d] = layer.a[i * 4] as number;
      s.a[d + 1] = layer.a[i * 4 + 1] as number;
      s.a[d + 2] = layer.a[i * 4 + 2] as number;
      s.a[d + 3] = layer.a[i * 4 + 3] as number;
      s.b[d] = layer.b[i * 4] as number;
      s.b[d + 1] = layer.b[i * 4 + 1] as number;
      s.b[d + 2] = layer.b[i * 4 + 2] as number;
      s.b[d + 3] = layer.b[i * 4 + 3] as number;
      s.fill++;
    }
  }
  const tPart = performance.now();

  // ---- register pools --------------------------------------------------------
  const reg = new GeometryRegistry();
  const heads = new Map<number, MeshHandle>(); // idF → chain head
  // N8-D1: heads whose class wants a DAG — built after registration, attached
  // after build(). The DAG comes off the head's FULL-detail source (rings[0]).
  const toDag: { handle: MeshHandle; source: ExplicitSource; label: string }[] = [];
  let deferredTris = 0;
  const notePart = (label: string, parts: PoolPart[] | null | undefined, from: number): void => {
    if (!parts) return;
    for (let p = from; p < parts.length; p++) {
      deferredTris += parts[p]?.tris ?? 0;
    }
    if (parts.length > from) deferred.push(`${label}: ${parts.length - from} card/leaf part(s)`);
  };

  for (const pool of lib.pools) {
    const policy = classPolicy(pool.cls);
    const idF = pool.cls * 8 + pool.variant;
    const label = `c${pool.cls}v${pool.variant}`;
    if (!policy) {
      notePart(label, pool.r1, 0);
      continue;
    }
    if (!inSet(policy.matClass)) {
      deferred.push(`${label}: class '${policy.matClass}' not in migration set`);
      continue;
    }
    // opaque part = parts[0] by construction (bark/rock/deadwood); the rest
    // are foliage cards / mesh leaves (deferred N9)
    const isTree = pool.cls <= TREE_MAX_CLS;
    const rings: { part: PoolPart; switchAt: number }[] = [];
    if (isTree && pool.r0?.[0]) rings.push({ part: pool.r0[0], switchAt: R0_FAR });
    if (pool.r1?.[0]) rings.push({ part: pool.r1[0], switchAt: isTree ? R1_FAR : policy.lodDist });
    // branch r2 is a geometry clone (indirect-slot bookkeeping) — skip it
    if (pool.r2?.[0] && pool.cls !== VegClass.Branch) {
      rings.push({ part: pool.r2[0], switchAt: 0 });
    }
    notePart(`${label}/r0`, pool.r0, 1);
    notePart(`${label}/r1`, pool.r1, 1);
    notePart(`${label}/r2`, pool.r2, 1);
    if (rings.length === 0) continue;
    const first = rings[0] as { part: PoolPart; switchAt: number };
    const headSource = geometryToSource(first.part.geo);
    const head = reg.registerMesh(headSource, policy.matClass, {
      transformChannel: policy.channel,
      castShadows: first.part.castShadow,
      label,
      swayPad: policy.swayPad,
      // matParam low byte = bark texture-array slice (rock ignores it); high
      // byte = trunk-wind profile (rigid channels ignore it). LODs inherit it.
      matParam: (pool.barkLayer ?? 0) | (windProfile(pool.cls) << 8),
    });
    for (let r = 1; r < rings.length; r++) {
      const prev = rings[r - 1] as { part: PoolPart; switchAt: number };
      const ring = rings[r] as { part: PoolPart; switchAt: number };
      reg.registerLod(head, geometryToSource(ring.part.geo), prev.switchAt);
    }
    // hybrid draw envelope = the old path's: trees hand over to impostors at
    // TREE_GEO_FAR; everything else uses its pool max distance (clsMaxDist;
    // trees' own entry is 1e8 "impostors continue" — not a geometry bound)
    const maxDist = isTree ? TREE_GEO_FAR : (lib.clsMaxDist[pool.cls] ?? 150);
    reg.setMaxDistance(head, maxDist);
    heads.set(idF, head);
    if (dag?.has(policy.matClass)) toDag.push({ handle: head, source: headSource, label });
  }

  // bind partitioned instances to chain heads. ?stress=N (synthetic, F3/F16
  // gate): bind N deterministic jittered copies of every stream — queue caps,
  // payload bits and overflow flags get exercised at ~N× instance counts.
  // Copies reuse the original y (hover/sink on slopes is fine for a stress
  // test — nothing reads it back).
  const stressParam = Number(new URLSearchParams(window.location.search).get('stress') ?? '1');
  const stress = Number.isFinite(stressParam) ? Math.max(1, Math.min(8, Math.floor(stressParam))) : 1;
  let deferredInstances = 0;
  for (const [id, s] of perId) {
    const head = heads.get(id);
    if (head === undefined) {
      deferredInstances += s.fill;
      continue;
    }
    reg.bindInstances(head, { a: s.a, b: s.b });
    for (let k = 1; k < stress; k++) {
      const a = new Float32Array(s.a);
      for (let i = 0; i < s.fill; i++) {
        a[i * 4] = (a[i * 4] as number) + (((k * 37 + i) % 13) - 6) * 0.61;
        a[i * 4 + 2] = (a[i * 4 + 2] as number) + (((k * 53 + i) % 11) - 5) * 0.73;
      }
      reg.bindInstances(head, { a, b: s.b });
    }
  }

  // ---- terrain: the REAL field as ONE heightfield source ---------------------
  const tTerr0 = performance.now();
  const dagTerrainTileAttaches: {
    handle: MeshHandle;
    gridVerts: Uint32Array;
    indices: Uint32Array;
    clusters: DagCluster[];
  }[] = [];
  // N8-D2 Stage 2b-1 (D-N39): when dagTerrainPool, tiles are collected here and
  // loaded into the streaming tile POOL post-build (instead of a per-tile mesh).
  const terrainPoolTiles: { gridVerts: Uint32Array; indices: Uint32Array; clusters: DagCluster[] }[] = [];
  // N8-D2 Stage 2b-3 (D-N39): the live clipmap streamer (?nanitedclip). Built +
  // pool-reserved here (pre-build), boot tiles attached post-build, returned for
  // TerrainScene to drive per frame. Owns the persistent DagWorker.
  let terrainStreamer: TerrainStreamer | null = null;
  if (inSet('terrain')) {
    const heights = hf.cpuHeights;
    if (!heights) throw new Error('WorldRegistry: hf.cpuHeights missing (boot order)');
    const res = hf.res;
    const quads = res - 1;
    const cell = WORLD_SIZE / res;
    const origin = cell / 2 - WORLD_SIZE / 2; // vertex (0,0) = texel-center 0
    if (dagTerrainGridN && dagTerrainGridN > 0) {
      // ---- terrain LOD DAG (D2b/D-N36), off-thread + cached (D1d/D-N37), TILED
      // for full-res scale (D-N38). The field splits into T×T independent tile
      // DAGs, each gridN² over its texel sub-region. A tile's outer perimeter is a
      // mesh boundary ⇒ buildDag auto-LOCKS those verts at full res ⇒ adjacent
      // tiles share identical edges ⇒ crack-free seams for free. T=1 = one DAG.
      const gridN = dagTerrainGridN;
      if ((gridN & (gridN - 1)) !== 0 || gridN > res) {
        throw new Error(`WorldRegistry: dagTerrainGridN must be a power of two ≤ ${res}, got ${gridN}`);
      }
      // clipmap implies the pool (its rings live in pool slots)
      const usePool = dagTerrainPool === true || dagTerrainClip === true;
      const T = dagTerrainTiles && dagTerrainTiles > 0 ? Math.floor(dagTerrainTiles) : 1;
      if (res % T !== 0) throw new Error(`WorldRegistry: dagTerrainTiles ${T} must divide res ${res}`);
      const tileTexels = res / T; // texels per tile side (far edge shared with neighbour)
      if (tileTexels % gridN !== 0 || tileTexels < gridN) {
        throw new Error(`WorldRegistry: tileTexels ${tileTexels} must be a positive multiple of gridN ${gridN}`);
      }
      const stride = tileTexels / gridN; // texels per tile-DAG cell
      const tHd0 = performance.now();
      let dagWorker: DagBuildWorker | null = null;
      try {
        dagWorker = new DagBuildWorker();
      } catch {
        dagWorker = null;
      }
      let poolMaxV = 0;
      let poolMaxT = 0;
      let poolMaxC = 0;
      const tileStats: TileBuildStats = { nCache: 0, nBuilt: 0 };
      const tileDeps: TileBuildDeps = { heights, res, cell, origin, gridN, seed: seed ?? null, worker: dagWorker };

      if (dagTerrainClip) {
        // ---- CLIPMAP STREAMER (D-N39 2b-2 boot + 2b-3 follow) ------------------
        // Build the spawn-centered ring set NOW (frame-1 terrain, no fallback),
        // size the pool for the WHOLE clipmap (any camera pose ⇒ ≤ maxTiles
        // resident) + headroom, attach post-build; the streamer then re-centers on
        // the live camera each frame (TerrainScene drives streamer.update). The
        // worker is PERSISTENT — handed to the streamer, NOT disposed at boot.
        const streamer = new TerrainStreamer(
          { reg, heights, res, cell, origin, gridN, seed: seed ?? null, worker: dagWorker },
          (m) => deferred.push(m),
        );
        const nBoot = await streamer.buildBootSet(res / 2, res / 2);
        const pm = streamer.poolMax;
        // caps = boot-worst × generous margin; a reload hits arbitrary regions and
        // an over-cap tile is SKIPPED (coarser ring backstops), never fatal.
        const vCap = Math.ceil(pm.v * 1.5) + 256;
        const tCap = Math.ceil(pm.t * 1.5) + 256;
        const cCap = Math.ceil(pm.c * 1.5) + 32;
        // headroom ABOVE clipmapMaxTiles so departed tiles can LINGER through the
        // async bake window (lazy eviction — the old LOD stays until its replacement
        // is resident; far stragglers are reclaimed first under pressure). ~1.5×.
        const slots = streamer.maxTiles + Math.ceil(streamer.maxTiles / 2);
        reg.reserveTilePool(
          'terrain',
          { originX: origin, originZ: origin, cellSize: cell },
          { slots, vertCap: vCap, triCap: tCap, clusterCap: cCap },
          { label: 'terrain' },
        );
        terrainStreamer = streamer;
        const s = streamer.bootSummary();
        deferred.push(
          `terrain DAG ${streamer.clipDesc}: ${nBoot} boot / ${streamer.maxTiles} max tiles, ` +
            `${s.tCl} cl, ${s.tTris | 0} tris, maxErr ${s.maxErr.toFixed(2)} m, offGrid ${s.offGrid}, ` +
            `${s.nCache} cached/${s.nBuilt} built, POOL ${slots}×(v${vCap}/t${tCap}/c${cCap}), ` +
            `${(performance.now() - tHd0).toFixed(0)} ms`,
        );
      } else {
        // ---- uniform T×T tiles: per-tile mesh, or the 2b-1 all-resident pool ----
        let tCl = 0;
        let tTris = 0;
        let maxErr = 0;
        let offGrid = 0;
        // collect a built tile (pool slot or per-tile mesh) + roll up stats
        const collectTile = (gridVerts: Uint32Array, built: HeightDagResult, label: string): void => {
          if (usePool) {
            terrainPoolTiles.push({ gridVerts, indices: built.indices, clusters: built.clusters });
          } else {
            const h = reg.registerHeightDag(
              'terrain',
              { originX: origin, originZ: origin, cellSize: cell },
              { label },
            );
            reg.bindInstances(h, { a: new Float32Array([0, 0, 0, 1]), b: new Float32Array([0, 0, 0, 0]) });
            reg.addLate({ verts: gridVerts.length, tris: built.indices.length / 3, clusters: built.clusters.length });
            dagTerrainTileAttaches.push({ handle: h, gridVerts, indices: built.indices, clusters: built.clusters });
          }
          if (gridVerts.length > poolMaxV) poolMaxV = gridVerts.length;
          if (built.indices.length / 3 > poolMaxT) poolMaxT = built.indices.length / 3;
          if (built.clusters.length > poolMaxC) poolMaxC = built.clusters.length;
          tCl += built.clusters.length;
          tTris += built.indices.length / 3;
          if (built.stats.maxError > maxErr) maxErr = built.stats.maxError;
          offGrid += built.stats.offGridVerts;
        };
        for (let tj = 0; tj < T; tj++) {
          for (let ti = 0; ti < T; ti++) {
            const { gridVerts, built } = await buildTerrainTile(
              tileDeps,
              ti * tileTexels,
              tj * tileTexels,
              stride,
              `-T${T}-${ti}x${tj}`,
              tileStats,
              (m) => deferred.push(m),
            );
            collectTile(gridVerts, built, T > 1 ? `terrain/t${ti}x${tj}` : 'terrain');
          }
        }
        if (usePool && terrainPoolTiles.length > 0) {
          // one shared reservation: S = tile count, each slot sized to the worst
          // tile this boot + margin.
          const vCap = Math.ceil(poolMaxV * 1.3) + 64;
          const tCap = Math.ceil(poolMaxT * 1.3) + 128;
          const cCap = Math.ceil(poolMaxC * 1.3) + 16;
          reg.reserveTilePool(
            'terrain',
            { originX: origin, originZ: origin, cellSize: cell },
            { slots: terrainPoolTiles.length, vertCap: vCap, triCap: tCap, clusterCap: cCap },
            { label: 'terrain' },
          );
        }
        deferred.push(
          `terrain DAG: ${T}×${T} tiles gridN ${gridN} → ${tCl} cl, ${tTris | 0} tris, ` +
            `maxErr ${maxErr.toFixed(2)} m, offGrid ${offGrid}, ${tileStats.nCache} cached/${tileStats.nBuilt} built, ` +
            `${usePool ? `POOL ${terrainPoolTiles.length}×(v${Math.ceil(poolMaxV * 1.3) + 64}/c${Math.ceil(poolMaxC * 1.3) + 16}) ` : ''}` +
            `${(performance.now() - tHd0).toFixed(0)} ms`,
        );
      }
      // clip mode hands the worker to the streamer (persistent); others dispose now.
      if (dagWorker && !dagTerrainClip) dagWorker.dispose();
    } else {
      const w = TERRAIN_WIN_QUADS;
      const windows = Math.ceil(quads / w);
      const minMax = new Float32Array(windows * windows * 2);
      for (let gz = 0; gz < windows; gz++) {
        const z0 = gz * w;
        const z1 = Math.min(z0 + w, quads);
        for (let gx = 0; gx < windows; gx++) {
          const x0 = gx * w;
          const x1 = Math.min(x0 + w, quads);
          let mn = Infinity;
          let mx = -Infinity;
          for (let z = z0; z <= z1; z++) {
            const row = z * res;
            for (let x = x0; x <= x1; x++) {
              const h = heights[row + x] as number;
              if (h < mn) mn = h;
              if (h > mx) mx = h;
            }
          }
          const i = (gz * windows + gx) * 2;
          minMax[i] = mn;
          minMax[i + 1] = mx;
        }
      }
      const hTerrain = reg.registerMesh(
        {
          kind: 'heightfield',
          quadsX: quads,
          quadsZ: quads,
          winQuads: w,
          cellSize: cell,
          originX: origin,
          originZ: origin,
          minMax,
        },
        'terrain',
        { label: 'terrain' },
      );
      reg.bindInstances(hTerrain, {
        a: new Float32Array([0, 0, 0, 1]),
        b: new Float32Array([0, 0, 0, 0]),
      });
    }
  } else {
    deferred.push("terrain: class 'terrain' not in migration set");
  }
  const tTerr1 = performance.now();
  deferred.push('GroundRing grass/debris: clipmap-instanced — N6/N10 per audit');
  deferred.push(`card/leaf tris deferred to N9: ${deferredTris}`);

  // ---- N8-D1: LOD DAGs for the selected explicit classes (SYNC — D1d moves
  // this to a background Worker per D-N30). Reserve the append budget before
  // build() freezes caps; a per-mesh build failure is logged + skipped (the
  // mesh keeps its discrete chain), never fatal.
  const tDag0 = performance.now();
  const dagBuilds: { handle: MeshHandle; dag: DagBuild }[] = [];
  let dagTris = 0;
  if (toDag.length > 0) {
    let lateV = 0;
    let lateT = 0;
    let lateC = 0;
    for (const item of toDag) {
      let built: DagBuild;
      try {
        built = buildDag(explicitToDagVerts(item.source), DAG_VERT_STRIDE, item.source.indices, {
          normalOffset: 3,
        });
      } catch (e) {
        deferred.push(`DAG ${item.label}: build failed (${e instanceof Error ? e.message : String(e)})`);
        continue;
      }
      lateV += built.verts.length / DAG_VERT_STRIDE;
      lateT += built.indices.length / 3;
      lateC += built.clusters.length;
      dagTris += built.stats.totalTris;
      dagBuilds.push({ handle: item.handle, dag: built });
    }
    reg.addLate({ verts: lateV, tris: lateT, clusters: lateC });
  }
  const tBuild0 = performance.now();
  const dagBuildMs = tBuild0 - tDag0;

  // ---- build ------------------------------------------------------------------
  const report = reg.build(renderer, counters);
  for (const b of dagBuilds) reg.attachDag(b.handle, b.dag);
  for (const t of dagTerrainTileAttaches) reg.attachHeightDag(t.handle, t);
  // N8-D2 Stage 2b-1: load the collected terrain tiles into pool slots (one per
  // tile here — all resident, render-parity with the per-tile path; the per-frame
  // clipmap streamer (2b-3) makes residency a camera-centered subset).
  for (const t of terrainPoolTiles) {
    const slot = reg.allocTileSlot();
    if (slot < 0) {
      deferred.push('terrain pool: out of slots (raise pool size)');
      break;
    }
    reg.attachHeightDagTile(slot, t);
  }
  // N8-D2 Stage 2b-3: attach the clipmap streamer's boot ring set into pool slots
  // (frame-1 terrain). The streamer then re-centers on the live camera per frame.
  if (terrainStreamer) terrainStreamer.attachBootSet();
  const t1 = performance.now();
  return {
    registry: reg,
    report,
    readbackMs: tRead - t0,
    partitionMs: tPart - tRead,
    terrainMs: tTerr1 - tTerr0,
    buildMs: t1 - tBuild0,
    totalMs: t1 - t0,
    deferredInstances,
    deferred,
    dagMeshes: dagBuilds.length,
    dagBuildMs,
    dagTris,
    ...(terrainStreamer ? { terrainStreamer } : {}),
  };
}
