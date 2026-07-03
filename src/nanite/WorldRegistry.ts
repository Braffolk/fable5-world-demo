/**
 * WorldRegistry (N1-C4, docs/NANITE-SPEC.md) — builds THE GeometryRegistry from the
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
import { Vector2 } from 'three';
import type { ScatterLayer, ScatterResult } from '../gpu/passes/Scatter';
import { VegClass } from '../gpu/passes/Scatter';
import type { VegLib, PoolPart } from '../vegetation/VegLibrary';
import type { Heightfield } from '../world/Heightfield';
import { WORLD_SIZE } from '../world/WorldConst';
import { type DagBuild, type DagCluster, buildDag } from './BuildDag';
import { buildAggregateDag, setAggLodErrorK } from './BuildAggregateDag';
import {
  BootCache,
  type PackedFarTile,
  packFarTiles,
  packPreparedCrown,
  type PackedPreparedCrown,
  unpackFarTiles,
  unpackPreparedCrown,
} from './BootCache';
import { appendFarTiles, buildFarTilesAsync, FT_TILE_SIZE, type FarTileSpecies } from './FarTiles';
import { setClusterFill } from './Clusterize';
import { DagBuildWorker, DagWorkerPool, type DagBuilder, type HeightDagResult } from './DagWorkerClient';
import { TerrainStreamer, buildTerrainTile, type TileBuildDeps, type TileBuildStats } from './TerrainStreamer';
import {
  type BuildReport,
  DAG_VERT_STRIDE,
  type ExplicitSource,
  GeometryRegistry,
  MAX_CLUSTER_TRIS,
  type MaterialClassId,
  type MeshHandle,
  type TransformChannel,
  explicitToDagVerts,
  setClusterTriCap,
} from './GeometryRegistry';
import { readBuffer } from './Tsl';
import {
  appendVoxelCrown,
  DEFAULT_VOXEL_GRID_DIM,
  type PreparedVoxelCrown,
  computeVoxlodAnchorL0,
  prepareVoxelCrown,
  setVoxlodConfig,
  setVoxOccThreshold,
  voxOccThreshold,
  voxlodLevels,
} from './VoxelizeCrown';
import { BRICK_WORDS, type BrickCPU } from './VoxelBrick';

/** Forests ring radii (Forests.ts) — discrete LOD switch distances until N8 */
const R0_FAR = 26;
const R1_FAR = 150;
const EX_R1_FAR = 120;
/** tree real-geometry envelope: R2_FAR 460 + BAND2 36 (Forests.ts) — beyond
 *  it the old path shows impostors, the hybrid's sanctioned far field (the
 *  N2 instance cull drops what impostors own; the N8 DAG retires this) */
const TREE_GEO_FAR = 496;
/** N9-C0: leaf-head cull-sphere wind padding (m). The crown does the FULL
 *  vegWindOffset (lean+sway+branch+flutter) and sits at the tree top where sway
 *  is maximal, so it rides the trunk's envelope — match the tree swayPad (3.8). */
const LEAF_SWAY_PAD = 3.8;
/** voxel-foliage (spec §3.2 / Stage 3a): the DEFAULT mesh→voxel handoff distance (m).
 *  The leaf head culls beyond it, the voxel head seeds beyond it → a clean hard switch.
 *  60 (2026-07-03, was 35 — the 2026-07-02 beautification pick, now SHARED forest+world):
 *  mesh only where it is genuinely near-LOD0; pushing the mesh band past ~60 m exposes
 *  the DAG simplifier's crown pathology ("massive leaves / spikes"), while fine voxels
 *  (grid 256, voxtaucap 4) read closer to real geometry from 60 m out. ?voxnear= tunes. */
export const DEFAULT_TRANSITION_DIST = 60;
/** terrain window size: 7 quads → 98 tris, divides 4095 exactly (4096² field) */
const TERRAIN_WIN_QUADS = 7;

/** N9-C0: pack a per-species foliage tint into the leaf head's matParam (mesh
 *  word 7) — linear RGB in the low 3 bytes + hueVar in the high byte (each 8-bit
 *  unorm; the resolve isL branch unpacks it). The leaf channel carries no bark
 *  layer / wind-profile byte, so all 32 bits are the tint. */
function packLeafTint(c: { r: number; g: number; b: number; hueVar: number }): number {
  const u8 = (x: number): number => Math.max(0, Math.min(255, Math.round(x * 255)));
  return (u8(c.r) | (u8(c.g) << 8) | (u8(c.b) << 16) | (u8(c.hueVar) << 24)) >>> 0;
}

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
  // N9-C2: two-sided meshes (leaf crowns) carry each triangle ONCE — the SW raster
  // re-winds a back-face to CCW in place (NaniteRaster.orientForRaster) instead of
  // backface-culling it, so the old reversed-winding duplicate is gone (D-N43 Stage
  // 0: half the leaf triangles/clusters). The resolve flips the leaf normal to face
  // the camera for lighting, independent of which side rasterized.
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
  /** N8-D2 Stage 2d: emit per-tile perimeter SKIRTS in clip mode (depth ∝ 2^level)
   *  to seal inter-level T-junction cracks that would otherwise show sky. Default
   *  true; `?nanitedskirt=0` turns them off for a same-pose A/B. */
  dagTerrainSkirt?: boolean;
  /** N9-C0/C2: also register each tree pool's REAL mesh-leaf crown as a
   *  MATERIAL_CLASS.leaf head, bound to the SAME instances as the bark trunk
   *  (a co-located mesh, not a LOD), with the 'leaf' flutter channel; the
   *  aggregate DAG extends it across the band and the voxel sibling owns
   *  beyond the handoff. DEFAULT ON from TerrainScene since 2026-07-03
   *  (`?naniteleaf=0` opts out). Tree pools only. */
  leaf?: boolean;
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
    dagTerrainSkirt,
    leaf: leafOn,
    seed,
  } = input;
  const inSet = (c: MaterialClassId): boolean => !classes || classes.has(c);
  const t0 = performance.now();
  const deferred: string[] = [];

  // A/B (?clustertris=256): set the cluster triangle cap BEFORE any registerMesh /
  // buildDag clusterizes — 256 halves the cluster COUNT (fewer per-cluster raster
  // workgroups, the dominant SW cost) at the cost of coarser per-cluster culling. The
  // raster/resolve shaders (built after this) read the same live cap. Default 128.
  const clusterParams = new URLSearchParams(window.location.search);
  setClusterTriCap(Number(clusterParams.get('clustertris')) || 256);
  // ?clusterfill: how full a cluster must be (fraction of the cap) before the clusterizer
  // FINALIZES on a dead adjacency frontier instead of joining the next (disconnected) leaf.
  // Default 0.75 (tight spheres); raise toward ~0.95 so chunky leaf-sprays pack fuller
  // (fewer clusters) — at the cost of looser per-cluster bounding spheres (watch visTris).
  setClusterFill(Number(clusterParams.get('clusterfill')) || 0.95);

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
  // N9-C0: idF → leaf-class head (the co-located crown on the same instances as
  // the bark trunk). Populated only when leafOn; bound in a separate pass below
  // (a mesh's instance streams must be consecutive in the cursor).
  const leafHeads = new Map<number, MeshHandle>();
  // N8-D1: heads whose class wants a DAG — built after registration, attached
  // after build(). The DAG comes off the head's FULL-detail source (rings[0]).
  const toDag: { handle: MeshHandle; source: ExplicitSource; label: string }[] = [];
  // N9-C2: leaf crowns get the AREA-PRESERVING aggregate DAG (BuildAggregateDag —
  // QEM degenerates on disconnected leaves) instead of QEM. Same DagBuild contract,
  // so it rides the identical attachDag + cut; extends the crown to TREE_GEO_FAR.
  const toAggregate: { handle: MeshHandle; source: ExplicitSource; label: string }[] = [];
  // voxel-foliage (spec §5.2/§5.3): ?voxreg=1 voxelizes each leaf crown OFFLINE and
  // collects the prepared bricks so the brick budget can be reserved BEFORE build()
  // (addLate freezes caps) and a voxel:7 sibling head appended AFTER. OFF by default —
  // the registry/raster wiring lands in Stage 2; this proves the reserve→append path.
  const qVox = new URLSearchParams(window.location.search);
  // ?forcevox=<idF> (or =1 / =all for every voxelized crown) — DEBUG override (spec §A1).
  // Forces the VOXEL path for the chosen crown(s) REGARDLESS of distance by voxelizing
  // them AND suppressing their LEAF mesh (leaf maxDist → ~0), with the voxel head's
  // nearDist=0 so it renders at ALL distances — the raster/resolve voxel path in isolation
  // WITHOUT the distance transition. Implies ?voxreg. Stage-3a makes the transition the
  // DEFAULT route (no flag): voxelize + register voxel heads + the mesh→voxel handoff.
  const forceVoxRaw = qVox.get('forcevox');
  const forceVoxAll = forceVoxRaw === '1' || forceVoxRaw === 'all';
  const forceVoxId = forceVoxRaw !== null && !forceVoxAll ? Number(forceVoxRaw) : null;
  const forceVoxOn = forceVoxRaw !== null;
  // ?voxreg=0 disables the automatic mesh→voxel transition in the WORLD scene. DEFAULT ON
  // since 2026-07-03 (the world-hookup arc): the world rides the SAME voxelised-foliage
  // stack the forest ships (voxel crowns beyond the handoff + far tiles), boot-cached.
  // Was opt-in (?voxreg=1) while the forest was the only calibrated path. ?forcevox implies.
  const voxReg = qVox.get('voxreg') !== '0' || forceVoxOn;
  const voxGridDim = Number(qVox.get('voxgrid') ?? DEFAULT_VOXEL_GRID_DIM) || DEFAULT_VOXEL_GRID_DIM;
  // ?voxnear= — the mesh→voxel handoff distance (m), TUNEABLE (spec §3.2.bis). Default
  // DEFAULT_TRANSITION_DIST (60 m — the shared beautification pick). The leaf head culls
  // beyond it; the voxel head seeds beyond it. ?forcevox overrides to 0 (voxel everywhere).
  const transitionDist = Number(qVox.get('voxnear') ?? DEFAULT_TRANSITION_DIST) || DEFAULT_TRANSITION_DIST;
  // ?fartiles=0 disables the cross-instance far-field tile aggregation (FarTiles.ts —
  // DEFAULT ON, 2026-07-03: same machinery as the forest; terrain-aware per-tile baseY).
  // ?aggdist= tile handoff distance; ?ftcell= tile cell size (m).
  const farTilesOn = qVox.get('fartiles') !== '0';
  // ⚠️ aggDist 280 here, NOT the forest's DEFAULT_AGG_DIST 140 (measured 2026-07-03,
  // wagg280 gallery): on real terrain an oblique view puts whole HILLSIDES in the
  // 94-280 m band — at 140 the fartile takeover painted them as a washed-out wall of
  // 2.9 m plates (user report). 280 keeps per-tree grid-256 voxels on everything you
  // actually look at; tiles own the background. Perf: oblique 47→58 fps (per-tree is
  // CHEAPER than the tile overlap there), high-aerial 109→107 (far collapse intact).
  // The forest keeps 140: flat ground only ever shows tile canopy TOPS, and its
  // F-vs-G sweep picked cell-size fidelity over takeover distance.
  const aggDist = Number(qVox.get('aggdist') ?? 280) || 280;
  // ⚠️ ftCell 0.75 here, NOT the forest's DEFAULT_FT_CELL 0.6 — the ONE knob the world
  // cannot share (measured 2026-07-03): the 4 km world splats ~3510 tiles vs the forest's
  // ~841, and 0.6 m cells produced ~12-13M far bricks ≈ 450 MB — over the 256 MB storage-
  // buffer cliff (and the BrickCPU JS heap OOM-crashed the tab at ~8M). 0.75 lands ~6-7M
  // (the forest-proven scale). Revisit if the brick payload slims (typed-array bricks /
  // buffer split); ?ftcell=0.6 works for A/B on machines with headroom.
  const ftCell = Number(qVox.get('ftcell') ?? 0.75) || 0.75;
  // ?leaflodk= — aggregate LEAF ladder error scale (BuildAggregateDag AGG_LOD_CFG,
  // shared default 0.4); ?voxocc= — occupancy coverage threshold (VoxelizeCrown).
  {
    const lk = qVox.get('leaflodk');
    if (lk !== null) setAggLodErrorK(Number(lk));
    const occRaw = qVox.get('voxocc');
    if (occRaw !== null) setVoxOccThreshold(Number(occRaw));
  }
  // ?voxlod (G1, DEFAULT ON): build the voxel MIP PYRAMID + a REAL multi-level DAG so the SAME
  // crown coarsens with distance through a BAND-ANCHORED octave ladder (far => bigger/fewer
  // voxels, near => finer), selected by the cull's screen-error cut (like UE5 Nanite Voxels).
  // ?voxlod=0 forces today's single-level degenerate always-cut DAG (the A/B baseline).
  const voxLod = qVox.get('voxlod') !== '0';
  // ANCHOR the ladder to the band (correction 1): ownError(L0) = transitionDist*tau/projK so the
  // FINEST level's cut lands at the handoff and each octave of distance descends one level (spans
  // [35,2000] m). projK mirrors the cull (cot(fovY/2)*renderHeight*0.5); no camera here, so the
  // app FOV (Engine.ts PerspectiveCamera = 55°) is used — the ladder SHAPE is projK-robust anyway.
  const APP_FOV_DEG = 55; // Engine.ts camera FOV
  const anchorH = input.renderer.getDrawingBufferSize(new Vector2()).y;
  {
    const anchorL0 = computeVoxlodAnchorL0(transitionDist, anchorH, APP_FOV_DEG);
    // ?voxlodk= (anchor multiplier) / ?voxlodlevels= / ?voxlodsparse= / ?voxlodshell= sweep the
    // ladder for A/B; unset = the band-anchored defaults (7 levels, K=1, sparse off, shell off).
    const kRaw = qVox.get('voxlodk');
    const lRaw = qVox.get('voxlodlevels');
    const spRaw = qVox.get('voxlodsparse');
    const shRaw = qVox.get('voxlodshell');
    setVoxlodConfig({
      anchorL0,
      errorK: kRaw !== null ? Number(kRaw) : undefined,
      levels: lRaw !== null ? Number(lRaw) : undefined,
      sparseK: spRaw !== null ? Number(spRaw) : undefined,
      shell: shRaw !== null ? Number(shRaw) : undefined,
    });
  }
  // ── boot cache (DDC — same store the forest path uses): crown voxelizations, LOD DAG
  // builds and the fartiles splat are deterministic in (sources × these params); key =
  // builder-source hash + RESOLVED values (a raw-null knob must never mask a code-default
  // change — the 2026-07-02 ftcell hazard). The scene marker keeps world/forest disjoint.
  const bootCache = new BootCache({
    params: {
      scene: 'world',
      seed: seed ?? 0,
      counts: [...idCounts.entries()].sort((x, y) => x[0] - y[0]),
      classes: classes ? [...classes].sort() : null,
      dagClasses: dag ? [...dag].sort() : null,
      leafOn: leafOn === true,
      voxReg,
      voxGridDim,
      voxLod,
      transitionDist,
      farTilesOn,
      aggDist,
      ftCell,
      voxOcc: voxOccThreshold(),
      anchorH,
      fov: APP_FOV_DEG,
      stress: qVox.get('stress'),
      leafDensity: qVox.get('naniteleafdensity'),
      knobs: ['voxlodk', 'voxlodlevels', 'voxlodsparse', 'voxlodshell', 'leaflodk', 'clustertris', 'clusterfill'].map(
        (k) => qVox.get(k),
      ),
    },
  });
  // crown voxelizations (idF-keyed — the pool walk below consumes them in place of
  // prepareVoxelCrown). Only fetched when the voxel path is live.
  const cachedCrownList =
    voxReg && leafOn ? await bootCache.get<{ idF: number; pack: PackedPreparedCrown }[]>('crowns') : null;
  const cachedCrowns = cachedCrownList ? new Map(cachedCrownList.map((c) => [c.idF, c.pack])) : null;
  const crownPacks: { idF: number; pack: PackedPreparedCrown }[] = [];
  /** idF → leaf head, for the ?forcevox leaf-suppression pass (filled in the loop). */
  const leafHeadForVox = new Map<number, MeshHandle>();
  const toVoxel: {
    idF: number;
    prep: PreparedVoxelCrown;
    source: ExplicitSource;
    matParam: number;
    label: string;
  }[] = [];
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

    // N9-C0/C2: the REAL leaf crown as a SEPARATE MATERIAL_CLASS.leaf mesh bound to
    // the SAME instances (trunk + crown render together — not LODs). The 'leaf'
    // channel carries the full vegWindOffset wind; the per-species tint packs into
    // matParam (this channel has no bark layer / wind profile). N9-C2: the
    // area-preserving aggregate DAG (built below) extends the crown across the FULL
    // trunk envelope (TREE_GEO_FAR), tapering hero-detail → coarse with distance.
    if (leafOn && pool.leaf) {
      // N9-C2: leaves are TWO-SIDED via the SW raster's back-face re-winding, not a
      // reversed-winding geometry duplicate — so the source carries each leaf tri
      // ONCE (half the leaf clusters feeding the per-instance flood; D-N43 Stage 0).
      const leafSource = geometryToSource(pool.leaf.geo);
      const leafHead = reg.registerMesh(leafSource, 'leaf', {
        transformChannel: 'leaf',
        castShadows: false,
        twoSided: true,
        label: `${label}/leaf`,
        swayPad: LEAF_SWAY_PAD,
        matParam: packLeafTint(pool.leaf.color),
        aggregate: true,
      });
      reg.setMaxDistance(leafHead, TREE_GEO_FAR);
      toAggregate.push({ handle: leafHead, source: leafSource, label: `${label}/leaf` });
      leafHeads.set(idF, leafHead);
      // voxel-foliage (§5.2): voxelize this crown OFFLINE now (cost-tolerant) so the
      // brick total is known before the addLate reservation freezes (§5.3). The voxel
      // sibling head + brick append happen post-build (a voxel cluster points at bricks,
      // not tris — that authoring is Stage 2; here we only reserve+upload the bricks).
      if (voxReg && (!forceVoxOn || forceVoxAll || forceVoxId === idF)) {
        const matParam = packLeafTint(pool.leaf.color);
        const cached = cachedCrowns?.get(idF);
        const prep = cached
          ? unpackPreparedCrown(cached)
          : prepareVoxelCrown(leafSource, pool.leaf.color, voxGridDim, voxLod);
        if (!cached) crownPacks.push({ idF, pack: packPreparedCrown(prep) });
        // a degenerate empty crown (0 bricks) would crash registerVoxelHead — skip it so
        // the leaf head keeps its full mesh envelope (set below to TREE_GEO_FAR, no handoff).
        if (prep.brickCount > 0) {
          toVoxel.push({ idF, prep, source: leafSource, matParam, label: `${label}/voxel` });
          leafHeadForVox.set(idF, leafHead);
        }
      }
    }
  }

  if (!cachedCrowns && crownPacks.length > 0) void bootCache.put('crowns', crownPacks);

  // Stage-3a: the mesh→voxel handoff, MESH side. For every voxelized crown, lower the LEAF
  // head's max draw distance to transitionDist so it renders ONLY nearer than the handoff;
  // its voxel sibling (nearDist=transitionDist, set at append) owns the mid/far band → a
  // clean hard switch (either mesh OR voxel at a distance — no double-render, no gap; the
  // cross-fade BAND is Stage 3b, here a hard line is acceptable, EXPECT some popping at the
  // boundary). ?forcevox is the DEBUG override: SUPPRESS the leaf entirely (maxDist→~0) so
  // the voxel head (nearDist=0) is the sole renderer at all distances — voxel path in
  // isolation. Pre-build, so a plain field set. (NaniteCull kSeedRoots reads both bounds.)
  if (leafHeadForVox.size > 0) {
    let handed = 0;
    let suppressed = 0;
    for (const [idF, leafHead] of leafHeadForVox) {
      const forced = forceVoxOn && (forceVoxAll || forceVoxId === idF);
      if (forced) {
        reg.setMaxDistance(leafHead, 0.001);
        suppressed++;
      } else {
        reg.setMaxDistance(leafHead, transitionDist);
        handed++;
      }
    }
    console.log(
      `[worldreg] voxel transition: ${handed} leaf head(s) handed off at ${transitionDist} m → voxel` +
        (suppressed > 0 ? `, ${suppressed} suppressed (?forcevox=${forceVoxRaw})` : ''),
    );
  }

  // bind partitioned instances to chain heads. ?stress=N (synthetic, F3/F16
  // gate): bind N deterministic jittered copies of every stream — queue caps,
  // payload bits and overflow flags get exercised at ~N× instance counts.
  // Copies reuse the original y (hover/sink on slopes is fine for a stress
  // test — nothing reads it back).
  const stressParam = Number(new URLSearchParams(window.location.search).get('stress') ?? '1');
  const stress = Number.isFinite(stressParam) ? Math.max(1, Math.min(8, Math.floor(stressParam))) : 1;
  let deferredInstances = 0;
  // bind a stream + its ?stress fan-out to one mesh head. A mesh's instance
  // streams must be CONSECUTIVE in the global cursor (GeometryRegistry.bindInstances
  // enforces it), so the leaf heads bind in their OWN pass below — never interleaved
  // with bark. The jitter is deterministic, so re-running it keeps each crown on its
  // trunk for both meshes.
  const bindStream = (
    head: MeshHandle,
    s: { a: Float32Array; b: Float32Array; fill: number },
  ): void => {
    reg.bindInstances(head, { a: s.a, b: s.b });
    for (let k = 1; k < stress; k++) {
      const a = new Float32Array(s.a);
      for (let i = 0; i < s.fill; i++) {
        a[i * 4] = (a[i * 4] as number) + (((k * 37 + i) % 13) - 6) * 0.61;
        a[i * 4 + 2] = (a[i * 4 + 2] as number) + (((k * 53 + i) % 11) - 5) * 0.73;
      }
      reg.bindInstances(head, { a, b: s.b });
    }
  };
  for (const [id, s] of perId) {
    const head = heads.get(id);
    if (head === undefined) {
      deferredInstances += s.fill;
      continue;
    }
    bindStream(head, s);
  }
  // N9-C0: the SAME instances → each tree pool's leaf head (separate pass per the
  // consecutive-streams rule; identical stress jitter keeps each crown on its trunk).
  if (leafOn) {
    for (const [id, s] of perId) {
      const leafHead = leafHeads.get(id);
      if (leafHead !== undefined) bindStream(leafHead, s);
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
      // clip mode bakes tiles CONTINUOUSLY as the camera roams → a POOL of workers so
      // a batch of arrivals bakes in parallel (#32, ~pool-size× shorter pop window);
      // the one-shot uniform path only needs a single worker. Headless node has no
      // Worker → construction throws → null → synchronous builds.
      const cores = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4;
      const poolSize = Math.max(2, Math.min(4, cores - 2));
      let dagWorker: DagBuilder | null = null;
      try {
        dagWorker = dagTerrainClip ? new DagWorkerPool(poolSize) : new DagBuildWorker();
      } catch {
        dagWorker = null;
      }
      const bakeThreads = dagWorker instanceof DagWorkerPool ? dagWorker.size : dagWorker ? 1 : 0;
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
          { reg, heights, res, cell, origin, gridN, seed: seed ?? null, worker: dagWorker, skirt: dagTerrainSkirt ?? true },
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
            `${s.nCache} cached/${s.nBuilt} built, POOL ${slots}×(v${vCap}/t${tCap}/c${cCap}), bake×${bakeThreads}, ` +
            `skirt ${dagTerrainSkirt ?? true ? 'on' : 'off'}, ${(performance.now() - tHd0).toFixed(0)} ms`,
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
            // 2e: terrain-DAG verts go in the stride-1 hf buffer (hfVerts), not `verts`.
            reg.addLate({ hfVerts: gridVerts.length, tris: built.indices.length / 3, clusters: built.clusters.length });
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
              -1, // uniform tiles are all one stride ⇒ crack-free perimeters, no skirt
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
  // boot cache: DagBuild[] in [toDag..., toAggregate...] order (handles re-derive from
  // THIS boot's registration, which always runs — only the expensive builds are skipped).
  // Usable only when the stored count matches BOTH job lists (a failed build desyncs the
  // order ⇒ length mismatch ⇒ clean rebuild).
  const cachedDags = await bootCache.getMany<DagBuild>('dags');
  const usableDags = cachedDags && cachedDags.length === toDag.length + toAggregate.length ? cachedDags : null;
  if (toDag.length > 0) {
    let lateV = 0;
    let lateT = 0;
    let lateC = 0;
    for (let di = 0; di < toDag.length; di++) {
      const item = toDag[di] as { handle: MeshHandle; source: ExplicitSource; label: string };
      let built: DagBuild;
      try {
        built = usableDags
          ? (usableDags[di] as DagBuild)
          : buildDag(explicitToDagVerts(item.source), DAG_VERT_STRIDE, item.source.indices, {
              normalOffset: 3,
              maxTris: MAX_CLUSTER_TRIS,
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
  // N9-C2: leaf-crown aggregate DAGs (area-preserving). Same DagBuild contract +
  // attach queue as the QEM meshes above, so they ride the identical cut. Built
  // synchronously here today; the boot cost is measured (the DAG-section flag) and
  // moves to the Worker/time-slice path if it threatens the D6 world-gen budget.
  const tAgg0 = performance.now();
  if (toAggregate.length > 0) {
    let aggV = 0;
    let aggT = 0;
    let aggC = 0;
    for (let ai = 0; ai < toAggregate.length; ai++) {
      const item = toAggregate[ai] as { handle: MeshHandle; source: ExplicitSource; label: string };
      let built: DagBuild;
      try {
        built = usableDags
          ? (usableDags[toDag.length + ai] as DagBuild)
          : buildAggregateDag(explicitToDagVerts(item.source), DAG_VERT_STRIDE, item.source.indices, {
              seed: seed ?? 0,
              maxTris: MAX_CLUSTER_TRIS,
            });
      } catch (e) {
        deferred.push(`AGG ${item.label}: build failed (${e instanceof Error ? e.message : String(e)})`);
        continue;
      }
      aggV += built.verts.length / DAG_VERT_STRIDE;
      aggT += built.indices.length / 3;
      aggC += built.clusters.length;
      dagTris += built.stats.totalTris;
      dagBuilds.push({ handle: item.handle, dag: built });
    }
    reg.addLate({ verts: aggV, tris: aggT, clusters: aggC });
  }
  const aggBuildMs = performance.now() - tAgg0;
  if (toAggregate.length > 0) {
    console.log(
      `[worldreg] leaf aggregate DAG: ${toAggregate.length} crowns in ${aggBuildMs.toFixed(0)} ms` +
        (usableDags ? ' (bootcache)' : ''),
    );
  }
  if (!usableDags && dagBuilds.length === toDag.length + toAggregate.length && dagBuilds.length > 0) {
    void bootCache.putMany('dags', dagBuilds.map((b) => b.dag));
  }
  // ---- GRASS S2 (31-grass-plan): ?grassreg=1 — a FULL-DENSITY patch field
  // around the origin. Patches are ~4×4 m merged blade meshes at the ring's
  // near-band density (~90 clumps/m², LUSHNESS LAW — no thinning), each with
  // an aggregate DAG (remove-whole-blades + grow-survivors = the ring's
  // thin×widen conservation, derived per level). The hier cull only seeds
  // meshes WITH DAG roots (kSeedRoots: rootCount 0 = skipped) — every grass
  // mesh MUST carry a DAG. Patch DAGs ride their OWN bootcache key
  // ('grassdags') so the shared 'dags' entry is never invalidated. Shipping
  // residency/density-law (biome/water/canopy gates) is S3.
  if (new URLSearchParams(window.location.search).get('grassreg') === '1') {
    const { grassPatchGeometry, GRASS_PATCH_SIZE, GRASS_PATCH_VARIANTS } = await import(
      '../vegetation/GrassPatch'
    );
    const tG0 = performance.now();
    const cachedGrass = await bootCache.getMany<DagBuild>('grassdags');
    const usableGrass = cachedGrass && cachedGrass.length === GRASS_PATCH_VARIANTS ? cachedGrass : null;
    const variantHandles: MeshHandle[] = [];
    const grassPacks: DagBuild[] = [];
    let grassTris = 0;
    for (let v = 0; v < GRASS_PATCH_VARIANTS; v++) {
      const src = geometryToSource(grassPatchGeometry(v));
      const h = reg.registerMesh(src, 'grass', {
        transformChannel: 'grass', // S1: GroundRing wind (tip² cantilever + shimmer)
        castShadows: false,
        twoSided: true,
        aggregate: true,
        // cull bound pad: wind tip deflection (≤0.6) + per-vertex terrain
        // conform delta within a 4 m patch (center-snap vs local bumps, ±~2 m)
        swayPad: 2.5,
        label: `grass/p${v}`,
      });
      reg.setMaxDistance(h, 265); // R3 far edge; the splat owns beyond (S4)
      variantHandles.push(h);
      try {
        const built = usableGrass
          ? (usableGrass[v] as DagBuild)
          : buildAggregateDag(explicitToDagVerts(src), DAG_VERT_STRIDE, src.indices, {
              seed: (seed ?? 0) + v,
              maxTris: MAX_CLUSTER_TRIS,
            });
        reg.addLate({
          verts: built.verts.length / DAG_VERT_STRIDE,
          tris: built.indices.length / 3,
          clusters: built.clusters.length,
        });
        dagBuilds.push({ handle: h, dag: built });
        grassPacks.push(built);
        grassTris += built.stats.totalTris;
      } catch (e) {
        deferred.push(`grass S2 DAG p${v} build failed: ${e instanceof Error ? e.message : String(e)}`);
      }
    }
    if (!usableGrass && grassPacks.length === GRASS_PATCH_VARIANTS) {
      void bootCache.putMany('grassdags', grassPacks);
    }
    // debug field: contiguous patch grid ±48 m (24×24 = 576 patches; the S3
    // residency replaces this with the toroidal biome-gated field).
    // ?grassat=x,z relocates it (ladder shots need FLAT ground — the origin
    // field straddles the ravine, where patch-center snap buries blades).
    const atRaw = (new URLSearchParams(window.location.search).get('grassat') ?? '0,0').split(',');
    const atX = Math.round((Number(atRaw[0]) || 0) / GRASS_PATCH_SIZE) * GRASS_PATCH_SIZE;
    const atZ = Math.round((Number(atRaw[1]) || 0) / GRASS_PATCH_SIZE) * GRASS_PATCH_SIZE;
    const HALF_PATCHES = 12;
    const NP = (HALF_PATCHES * 2) ** 2;
    let sd = 24680;
    const rnd = (): number => {
      sd = (sd * 1664525 + 1013904223) >>> 0;
      return sd / 4294967296;
    };
    const streams = new Map<number, { a: number[]; b: number[] }>();
    for (let pz = -HALF_PATCHES; pz < HALF_PATCHES; pz++) {
      for (let px = -HALF_PATCHES; px < HALF_PATCHES; px++) {
        const x = atX + px * GRASS_PATCH_SIZE;
        const z = atZ + pz * GRASS_PATCH_SIZE;
        const cx = x + GRASS_PATCH_SIZE / 2;
        const cz = z + GRASS_PATCH_SIZE / 2;
        const v = Math.floor(rnd() * GRASS_PATCH_VARIANTS) % GRASS_PATCH_VARIANTS;
        let st = streams.get(v);
        if (!st) {
          st = { a: [], b: [] };
          streams.set(v, st);
        }
        // patch-center terrain snap (S3 refines to per-vertex conform on slopes)
        st.a.push(x, hf.heightAtCpu(cx, cz), z, 1);
        st.b.push(0, 0, 0, st.a.length / 4);
      }
    }
    for (const [v, st] of streams) {
      const h = variantHandles[v];
      if (h !== undefined)
        reg.bindInstances(h, { a: new Float32Array(st.a), b: new Float32Array(st.b) });
    }
    // eslint-disable-next-line no-console
    console.log(
      `[worldreg] grass S2 field: ${NP} patches (${GRASS_PATCH_VARIANTS} variants @ ~${Math.round(
        grassTris / Math.max(1, GRASS_PATCH_VARIANTS) / 1000,
      )}k DAG tris), ${(performance.now() - tG0).toFixed(0)} ms${usableGrass ? ' (bootcache)' : ''}`,
    );
  }
  // voxel-foliage (§5.3 HARD precondition): reserve the brick budget BEFORE build()
  // freezes the caps. Total = Σ occupied bricks across the voxelized crowns. Also
  // reserve the voxel sibling heads (1 mesh each) + their instance streams (each
  // voxel head re-binds its leaf sibling's stream — same instances, a second mesh).
  if (toVoxel.length > 0) {
    let lateBricks = 0;
    let lateVoxInst = 0;
    let lateVoxClusters = 0;
    for (const v of toVoxel) {
      // voxlod=0: occupied bricks of the single grid. voxlod=1 (G4): Σ over all pyramid
      // levels (~1.0–1.36× the L0 count — the coarse tail is small). prep.brickCount carries
      // whichever it is, so the reservation auto-grows for the pyramid (no overflow/clipping).
      lateBricks += v.prep.brickCount;
      // each voxel head authors prep.clusterCount CLUSTERS, 0 verts / 0 tris — the bricks ARE
      // the payload. voxlod=0: ceil(occupied/128). voxlod=1: Σ over levels of ceil(occ_L/128).
      // dagLinks (roots+children) ride the auto-sized caps.clusters*2 headroom (no reservation).
      lateVoxClusters += v.prep.clusterCount;
      const s = perId.get(v.idF);
      // the voxel head re-binds ONE copy of the leaf stream (no ?stress fan-out)
      if (s) lateVoxInst += s.fill;
    }
    const lateVoxHeads = toVoxel.length;
    reg.addLate({
      bricks: lateBricks,
      meshes: lateVoxHeads,
      instances: lateVoxInst,
      clusters: lateVoxClusters,
    });
    console.log(
      `[worldreg] voxel-foliage: reserving ${lateBricks} bricks across ${toVoxel.length} crowns ` +
        `(grid ${voxGridDim}${voxLod ? `, voxlod ${voxlodLevels()}L pyramid` : ''}) = ${((lateBricks * BRICK_WORDS * 4) / (1024 * 1024)).toFixed(3)} MB, ` +
        `+${lateVoxClusters} voxel clusters, ` +
        `+${lateVoxInst} voxel instances, +${lateVoxHeads} voxel:7 heads`,
    );
  }
  // far-tile aggregation (FarTiles.ts — the forest path's move, world-wired 2026-07-03):
  // beyond aggDist whole 64 m tiles of trees render as ONE merged voxel head. Terrain-
  // aware: each tile grid floors at its members' min ground y (per-tile baseY); trunk
  // columns rise from each tree's own ground. Built pre-build for the exact reservation,
  // appended post-build. Splat rides the FarTiles worker pool; boot-cached.
  // PACKED end to end (2026-07-03): each tile is compacted to the bootcache form AS IT
  // EMITS and unpacked one-at-a-time at append — the whole-map BrickCPU object graph
  // (~250 B/brick × millions) OOM-crashed the tab twice before this.
  let ftPacked: PackedFarTile[] = [];
  if (farTilesOn && toVoxel.length > 0) {
    const tFt0 = performance.now();
    const ftPools: { a: Float32Array; b: Float32Array; species: FarTileSpecies }[] = [];
    for (const v of toVoxel) {
      const s = perId.get(v.idF);
      const levels = v.prep.vox.levels;
      if (!s || !levels || levels.length === 0) continue;
      // pick the crown pyramid level whose brick size best matches the tile cell size
      let pick = 0;
      let bestD = Infinity;
      for (let L = 0; L < levels.length; L++) {
        const bw = (levels[L] as { cellSize: number }).cellSize * 4;
        const d = Math.abs(bw - ftCell);
        if (d < bestD) {
          bestD = d;
          pick = L;
        }
      }
      const lvl = levels[pick] as { bricks: BrickCPU[]; occupied: number[] };
      const bricks = lvl.occupied.map((i) => lvl.bricks[i] as BrickCPU);
      let crownMinY = 2;
      for (const b of bricks) crownMinY = Math.min(crownMinY, b.center[1] - b.half);
      ftPools.push({
        a: s.a,
        b: s.b,
        species: { bricks, crownMinY: Math.max(0.5, crownMinY), bark: { r: 0.42, g: 0.33, b: 0.24 } },
      });
    }
    const cachedFt = await bootCache.get<PackedFarTile[]>('fartiles');
    if (cachedFt) {
      // filter(Boolean): a 2026-07-03 bug stored nulls (the fire-and-forget put's
      // structured clone ran AFTER the append loop released slots in place) — heal
      // any poisoned entry; the append no longer mutates the stored array.
      ftPacked = cachedFt.filter(Boolean);
    } else {
      ftPacked = await buildFarTilesAsync(
        { tileSize: FT_TILE_SIZE, cellSize: ftCell, pools: ftPools },
        (b) => packFarTiles([b])[0] as PackedFarTile,
      );
      void bootCache.put('fartiles', ftPacked);
    }
    let ftBricks = 0;
    let ftClusters = 0;
    for (const t of ftPacked) {
      ftBricks += t.prep.brickCount;
      ftClusters += t.prep.clusterCount;
    }
    reg.addLate({ bricks: ftBricks, meshes: ftPacked.length, instances: ftPacked.length, clusters: ftClusters });
    console.log(
      `[worldreg] fartiles: ${ftPacked.length} tiles, ${ftBricks} bricks (${((ftBricks * BRICK_WORDS * 4) / 1048576).toFixed(1)} MB), ` +
        `${ftClusters} clusters, aggDist ${aggDist} m, built in ${(performance.now() - tFt0).toFixed(0)} ms`,
    );
  }

  const tBuild0 = performance.now();
  const dagBuildMs = tBuild0 - tDag0;

  // ---- build ------------------------------------------------------------------
  const report = reg.build(renderer, counters);
  for (const b of dagBuilds) reg.attachDag(b.handle, b.dag);
  // voxel-foliage (§5.2): append the prepared crown bricks + register voxel:7 sibling
  // heads now that build() has frozen the caps. Bound to the SAME instances as the
  // leaf head (the per-mesh distance handoff is Stage 3 — here both heads draw full).
  if (toVoxel.length > 0) {
    const tVox0 = performance.now();
    let appended = 0;
    for (const v of toVoxel) {
      const r = appendVoxelCrown(reg, v.prep, v.source, {
        matParam: v.matParam,
        swayPad: LEAF_SWAY_PAD,
        // fartiles: the per-tree voxel crown ENDS at aggDist — the merged tile head owns
        // the far field beyond (ranges overlap by the tile radius, see FarTiles.ts).
        maxDist: ftPacked.length > 0 ? aggDist : TREE_GEO_FAR,
        // Stage-3a: the voxel head seeds only beyond transitionDist (the mesh→voxel
        // handoff); ?forcevox forces nearDist=0 (voxel everywhere, leaf suppressed below).
        nearDist: forceVoxOn && (forceVoxAll || forceVoxId === v.idF) ? 0 : transitionDist,
        label: v.label,
      });
      appended += r.brickCount;
      // bind the voxel head to the SAME instances as its leaf sibling
      const s = perId.get(v.idF);
      if (s) reg.bindInstances(r.head, { a: s.a, b: s.b });
      // fartiles: the per-tree BARK trunk ends at aggDist too — the tile splat carries
      // its own trunk columns beyond.
      if (ftPacked.length > 0) {
        const bark = heads.get(v.idF);
        if (bark !== undefined) reg.setMaxDistance(bark, aggDist);
      }
    }
    if (ftPacked.length > 0) {
      const ftTint = toVoxel[0]?.matParam ?? 0;
      const nTiles = ftPacked.length;
      let ftBricks = 0;
      // unpack + append ONE tile at a time (the whole-map object graph is heap-fatal;
      // peak = the packed array + one live tile). Do NOT release slots in place — the
      // fire-and-forget bootCache.put still references THIS array and its structured
      // clone runs after we get here (nulling slots stored a poisoned cache entry).
      for (let i = 0; i < nTiles; i++) {
        const tile = unpackFarTiles([ftPacked[i] as PackedFarTile]);
        ftBricks += appendFarTiles(reg, tile, { nearDist: Math.max(10, aggDist - 46), matParam: ftTint });
      }
      console.log(`[worldreg] fartiles: appended ${ftBricks} bricks across ${nTiles} tile heads`);
      ftPacked = []; // release
    }
    console.log(
      `[worldreg] voxel-foliage: appended ${appended} bricks (${reg.brickCount}/${reg.brickCapacity}) ` +
        `+ ${toVoxel.length} voxel:7 heads in ${(performance.now() - tVox0).toFixed(0)} ms`,
    );
    // Stage-2 A1: the voxel heads + their instance streams were registered/bound
    // POST-build (registerMesh/bindInstances stage clusterRecs + cpuStreams) — flush()
    // copies them into the mega-buffers + uploads the partial ranges so the cull sees
    // the voxel clusters (and their repointed brick word6/word7) this frame. Without
    // this the late heads never reach the GPU (build() only uploads pre-build entries).
    reg.flush(renderer, counters);
  }
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
