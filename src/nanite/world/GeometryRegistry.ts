/**
 * GeometryRegistry (N1-C3, docs/NANITE-SPEC.md) — the single entry point for solid
 * geometry per the content contract: registerMesh(ClusterSource, materialClass,
 * opts) + bindInstances(handle, stream), then build() packs everything into
 * PACKED mega-buffers sized against the 10-storage-bindings-per-stage ceiling:
 *
 *   vertex blob   6×u32/vert (24 B): pos 3×f32-bits, oct-normal snorm2x16,
 *                 uv half2x16, vdata (caller-packed aux word, e.g. wind params)
 *   index blob    1×u32/index (global vertex ids — u8 cluster-local triples are
 *                 a later memory pass)
 *   cluster recs  8×u32: sphere 4×f32-bits, cone oct-axis snorm2x16, cone-cos
 *                 f32-bits, triStart (heightfield: gx|gz<<16), triCount u8 |
 *                 flags u8 | meshId u16
 *   mesh table    12×u32: clusterStart/Count, instFirst/Count, lodNext,
 *                 lodDist f32-bits, channel|matClass|flags|winQuads bytes,
 *                 hf originX/originZ/cellSize f32-bits, quadsX|quadsZ u16s
 *                 (total quads — edge windows clamp), swayPad f32-bits
 *   instances     2×vec4/instance — A=(x,y,z,scale), B=(yaw,leanX,leanZ,idF)
 *                 verbatim (slot-hash variation law reads B.w)
 *   instanceMesh  1×u32/instance — owning mesh id (cull-side lookup)
 *
 * NOTE vs the NEXT-ACTIONS sketch: the cluster record's matClass byte became
 * meshId u16 — raster/resolve need cluster→mesh (heightfield params, channel,
 * matClass all live in the mesh record); triCount is a u8 field ⇒ the cluster tri cap
 * is at most 255 (setClusterTriCap caps the ?clustertris=256 path at 255, not 256 — a
 * 256-tri cluster would overflow triCount to 0 and raster nothing).
 *
 * Heightfield sources store NO vertices (F4): records carry grid-window
 * coordinates; kernels reconstruct positions from the resident heights buffer
 * (bound separately at raster/resolve — not owned here).
 *
 * Late registration (hero trees) is supported: construct with a `late` budget,
 * registerMesh/bindInstances after build(), then flush() — appended ranges
 * upload via addUpdateRange (partial writeBuffer, verified in
 * WebGPUAttributeUtils). Capacity overflow throws — never silent (F14).
 *
 * GPU instance sources (scatter layers) copy into the blob via one tiny
 * compute kernel per stream at build/flush; CPU arrays memcpy directly.
 */

import { Fn, If, float, instanceIndex, normalize, storage, uint, vec3, vec4 } from 'three/tsl';
import type { Renderer, StorageBufferNode } from 'three/webgpu';
import { StorageBufferAttribute } from 'three/webgpu';
import type { NF, NU, NV2, NV3, NV4 } from '../../gpu/TSLTypes';
import type { DagBuild, DagCluster } from '../build/BuildDag';
import { buildDagHierarchy, buildHeightGridHierarchy, maxChainDepth } from '../build/DagHierarchy';
import { BRICK_WORDS, brickCenterHalf, octDecode, octEncode } from '../voxel/VoxelBrickCore';
import { type BuiltClusters, type ClusterStats, clusterize } from '../build/Clusterize';
import {
  type BufOf,
  type V4W,
  bcF2U,
  bcU2F,
  dispatch,
  elemU,
  elemUW,
  readBuffer,
  returnIf,
  sF32Views,
  sU32Views,
  sVec4Views,
  toF,
  unpackHalfU,
  unpackSnormU,
} from '../Tsl';

export const VERT_WORDS = 6;
export const CLUSTER_WORDS = 8;
/** N8-HIC: words 0–15 as before (cluster range, LOD chain, sphere, …); words
 *  16–17 = the HIERARCHICAL-cull root seed range (rootBase, rootCount) into
 *  gpu.dagLinks — the coarsest clusters the BFS traversal starts from. */
export const MESH_WORDS = 18;
/** N8-D1/HIC: parallel per-cluster DAG record (f32, indexed by the SAME global
 *  clusterId as the 8-word cluster record). 0 ownErr + 1..4 ownSphere + 5
 *  parentErr + 6..9 parentSphere (the screen-error cut), then N8-HIC:
 *  10 childBase + 11 childCount (bitcast u32) — the cluster's children in
 *  gpu.dagLinks, for the top-down hierarchical traversal (only a group's owner
 *  parent carries them; 0 = leaf/non-owner). Sidecar keeps the cut kernel ≤10
 *  storage bindings (F9). */
const DAG_WORDS = 12;
/** N8-D1: vertex layout fed to buildDag for a registry mesh — pos@0..2,
 *  nrm@3..5, uv@6..7, vdata@8..11 (UNPACKED to 0..1 floats so QEM can
 *  interpolate them). attachDag re-packs this back into VERT_WORDS. */
export const DAG_VERT_STRIDE = 12;
/** N8-D1: a root cluster's parentError is +∞ — stored as this finite sentinel
 *  so the GPU projection (errToPx) yields a huge value (> any τ) without inf/NaN. */
export const DAG_ROOT_PARENT_ERR = 1e30;
/** cluster triangle cap. The SW raster runs ONE workgroup of MAX_CLUSTER_TRIS threads per
 *  cluster (1 thread/triangle), and the resolve payload packs the triangle index in
 *  CLUSTER_TRI_BITS low bits (itemIdx << BITS | localTri, MASK = cap−1). 128 is the Nanite
 *  standard; `?clustertris=256` (setClusterTriCap, A/B) HALVES the cluster COUNT — fewer
 *  per-cluster workgroup launches, the dominant SW-raster cost (D-N43 τ-sweep) — at the
 *  cost of COARSER per-cluster culling granularity (frustum/cone/LOD-cut). Mutable for the
 *  A/B; default 128. Read live by the raster/resolve/registry at build time. */
export let MAX_CLUSTER_TRIS = 128;
export let CLUSTER_TRI_BITS = 7; // log2(MAX_CLUSTER_TRIS) — payload localTri field width
export let CLUSTER_TRI_MASK = 127; // MAX_CLUSTER_TRIS − 1 — payload localTri mask
/** PERF-3 vertex-cache: a cluster whose vertex indices span ≤ this many entries is
 *  cooperatively transformed ONCE into a workgroup-shared vec3 array (3 f32 each →
 *  VCACHE_VERTS·12 B; 192 → 2.3 KB, well under the 16 KB workgroup limit). Bounds the
 *  shared array AND the per-cluster cache width. ≥ MAX_CLUSTER_TRIS so a 128-tri cluster
 *  with ~tight indexing always fits (avg unique 82, explicit 95% range ≤128). */
// LEVER C: raised 192/382 → 512 (= MAX_CLUSTER_VERTS in raster/Project.ts) so populateVCompact
// stores (vMin,count) for EVERY mesh cluster (real clusters ≤ ~494 unique < 512), CLOSING the
// tooWide (vcCount==0) fallback ⇒ Lever B's per-cluster mesh reservation vertCount = vcCount is
// EXACT for all mesh clusters. Verified-safe consumer: Project's stride loops
// ceil(512/MAX_CLUSTER_TRIS) still cover the unique range (3 strides @255, 4 @128). Bit-identical:
// a cluster that switches tooWide→covered writes the SAME deduped slots (Lever 1 identity).
export let VCACHE_VERTS = 512;

/**
 * A/B (`?clustertris`): set the cluster triangle cap to 128 or 256 + derive the payload
 * bits/mask and the vertex-cache width. MUST run at boot BEFORE buildWorldRegistry + the
 * nanite shaders build (all read these as live module bindings). Clamped to {128, 256} —
 * 256 is the WebGPU baseline `maxComputeInvocationsPerWorkgroup`, and the resolve payload
 * (`itemIdx<<BITS|localTri`, itemIdx 23 bits) has exactly 1 spare bit for an 8-bit index.
 */
export function setClusterTriCap(cap: number): void {
  const bits = cap >= 256 ? 8 : 7;
  CLUSTER_TRI_BITS = bits;
  CLUSTER_TRI_MASK = (1 << bits) - 1; // 127 or 255 — the PAYLOAD localTri field width
  // CRITICAL: triCount packs into an 8-bit cluster-record field (word 7, byte 0 — see the
  // `(dc.triCount & 0xff)` in attachDag). So a cluster can hold at most 255 triangles. The
  // 8-bit payload addresses 256, but a 256-tri cluster overflows the field (256 & 0xff = 0)
  // → the GPU reads triCount = 0 → `if (localTri < 0)` never fires → the cluster rasters
  // NOTHING → holes. So the high cap is 255, NOT 256 (identical cluster-count win).
  MAX_CLUSTER_TRIS = bits === 8 ? 255 : 128;
  VCACHE_VERTS = 512; // LEVER C: full MAX_CLUSTER_VERTS coverage (was 192 @128 / ~382 @255)
}
export const LOD_NONE = 0xffffffff;
/** N8-D2 Stage 2a: an evicted streaming-tile slot parks its mesh sphere here so
 *  kInstCull's frustum test always rejects it (belt + suspenders alongside the
 *  authoritative clusterCount=0, which makes lodSelectAndPush enqueue 0 chunks). */
const TILE_EVICTED_FAR = 1e9;

const MATERIAL_CLASS = {
  terrain: 0,
  rock: 1,
  bark: 2,
  deadwood: 3,
  leaf: 4,
  grass: 5,
  debris: 6,
  /** voxel-foliage sibling mesh (spec §4.1 / §0.2). The voxel decision rides the
   *  MESH record's matClass — there is NO free cluster-flag bit (§0.1). The resolve
   *  derives "is voxel" from gpu.meshes[meshId].matClass for free (§4.4). matClass is
   *  byte 1 of mesh word6 ((w6>>>8)&0xff), so the 8-bit field has room past debris(6). */
  voxel: 7,
} as const;
export type MaterialClassId = keyof typeof MATERIAL_CLASS;

export const TRANSFORM_CHANNEL = {
  rigid: 0,
  trunk: 1,
  leaf: 2,
  grass: 3,
  terrain: 4,
} as const;
export type TransformChannel = keyof typeof TRANSFORM_CHANNEL;

/** mesh-record flag bits (byte 2 of word 6) */
export const MESH_FLAG_HEIGHTFIELD = 1;
const MESH_FLAG_AGGREGATE = 2;
const MESH_FLAG_CAST_SHADOWS = 4;
/** N8-D1: mesh's clusterStart/Count point at its FULL DAG cluster range (all
 *  levels) and lodNext = NONE — the cull applies the per-cluster screen-error
 *  cut (project(own)≤τ AND project(parent)>τ) instead of the discrete LOD chain */
export const MESH_FLAG_HASDAG = 8;
/** N9-C2: mesh renders from BOTH faces. The SW raster re-winds a back-face to CCW
 *  in place instead of culling it (NaniteRaster.orientForRaster), so the geometry
 *  carries each triangle ONCE (no reversed-winding duplicate). Leaf crowns only. */
export const MESH_FLAG_TWO_SIDED = 16;
/** 2026-07-02 beautification: this voxel head is a cross-instance FAR TILE (FarTiles.ts,
 *  label 'fartile'). Consumers: the cull EXEMPTS it from the per-instance min-screen-size
 *  cull (a tile is the far representation — size-culling it deletes whole 64 m chunks past
 *  ~1.2 km, the user-visible aerial holes), and the resolve blends its brick normals toward
 *  up (?ftnrm) to kill the tile-pitch dark banding from splat-averaged mean normals. */
export const MESH_FLAG_FARTILE = 32;
/** cluster-record flag bits (byte 1 of word 7) */
export const CLUSTER_FLAG_HEIGHTFIELD = 1;
/** N8-D1: this cluster carries a DAG record at the same global index in gpu.dag */
export const CLUSTER_FLAG_DAG = 2;

export interface ExplicitSource {
  kind: 'mesh';
  /** xyz, stride 3 */
  positions: Float32Array;
  /** xyz, stride 3 (renormalized by the oct encoder) */
  normals: Float32Array;
  /** xy, stride 2; default 0 */
  uvs?: Float32Array;
  /** caller-packed per-vertex aux word (wind params etc.); default 0 */
  vdata?: Uint32Array;
  indices: Uint32Array;
}

export interface HeightfieldSource {
  kind: 'heightfield';
  /** total quads along x / z (vertex grid is quads+1; res−1 for a height texture) */
  quadsX: number;
  quadsZ: number;
  /** quads per window side; the LAST window per axis may be partial (edge clamp) */
  winQuads: number;
  /** world units per quad */
  cellSize: number;
  /** world position of vertex (0,0) */
  originX: number;
  originZ: number;
  /** per-WINDOW height (min,max) pairs, ceil(quadsX/w)·ceil(quadsZ/w) windows */
  minMax: Float32Array;
}

export type ClusterSource = ExplicitSource | HeightfieldSource;

export interface RegisterOpts {
  /** default 'rigid'; heightfield sources force 'terrain' */
  transformChannel?: TransformChannel;
  /** foliage-style DAG collapse at N8/N9 */
  aggregate?: boolean;
  /** default true */
  castShadows?: boolean;
  /** N9-C2: render from both faces — the SW raster re-winds back-faces instead of
   *  culling, so the source needs no reversed-winding duplicate (leaf crowns). */
  twoSided?: boolean;
  /** max wind sway amplitude in meters — cluster-bound padding at cull (F6) */
  swayPad?: number;
  /** explicit-mesh material parameter (e.g. bark texture-array slice). Stored
   *  in mesh-record word 7 — which holds hfOriginX for HEIGHTFIELD meshes only,
   *  so explicit meshes reuse it free (resolve reads it raw via meshWord(7)). */
  matParam?: number;
  /** stats-table label */
  label?: string;
}

export type MeshHandle = number;

/** CPU instance records — A/B vec4 pairs, counts must match */
interface InstanceStreamCPU {
  a: Float32Array;
  b: Float32Array;
}
/** GPU-resident instance records (scatter layers) — copied by kernel at build/flush */
interface InstanceStreamGPU {
  bufA: StorageBufferNode<'vec4'>;
  bufB: StorageBufferNode<'vec4'>;
  count: number;
  /** offset into the source buffers (default 0) */
  srcFirst?: number;
}
export type InstanceStream = InstanceStreamCPU | InstanceStreamGPU;

export interface LateBudget {
  verts: number;
  /** N8-D2 Stage 2e: terrain-DAG verts live in a SEPARATE stride-1 buffer (each
   *  vert is one packed texel coord — word0 only) so they don't waste 5/6 of a
   *  6-word `verts` record. Reserve them here (registerHeightDag/pool late attach). */
  hfVerts: number;
  tris: number;
  clusters: number;
  meshes: number;
  instances: number;
  /** voxel-foliage (spec §4.2/§5.3): number of BRICKS (BRICK_WORDS each) to reserve
   *  in gpu.voxelBricks. HARD precondition — addLate freezes caps at build(), and the
   *  voxelizer appends bricks post-build, so the per-palette brick total (§5.3) MUST
   *  be reserved here first. Sized to the BrickCPU layout (mean-normal default, §4.3).
   *  0 on worlds with no voxel foliage ⇒ a 1-word placeholder buffer (never bound). */
  bricks: number;
}

interface MeshReport {
  label: string;
  matClass: MaterialClassId;
  verts: number;
  tris: number;
  clusters: number;
  avgTris: number;
  fullFrac: number;
  buildMs: number;
}

export interface BuildReport {
  meshes: number;
  clusters: number;
  tris: number;
  verts: number;
  instances: number;
  bytes: {
    verts: number;
    /** N8-D2 Stage 2e: terrain-DAG stride-1 vertex buffer (separate from `verts`). */
    hfVerts: number;
    indices: number;
    clusters: number;
    meshTable: number;
    instances: number;
    /** voxel-foliage: gpu.voxelBricks bytes actually used (brickCount × BRICK_WORDS × 4). */
    bricks: number;
    total: number;
  };
  clusterizeMs: number;
  totalMs: number;
  perMesh: MeshReport[];
  table: string;
}

// ---------------------------------------------------------------------------
// CPU codecs (exported for the probe + GPU validation mirrors)
// ---------------------------------------------------------------------------

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);

function f32Bits(v: number): number {
  f32Scratch[0] = v;
  return u32Scratch[0] as number;
}
function bitsF32(u: number): number {
  u32Scratch[0] = u >>> 0;
  return f32Scratch[0] as number;
}

/** IEEE f32 → f16 bits, round-to-nearest-even (matches WGSL pack semantics) */
function f32ToF16(v: number): number {
  f32Scratch[0] = v;
  const x = u32Scratch[0] as number;
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let man = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (man ? 0x200 : 0);
  const e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    man |= 0x800000;
    const shift = 14 - e;
    let h = man >>> shift;
    const rem = man & ((1 << shift) - 1);
    const half = 1 << (shift - 1);
    if (rem > half || (rem === half && (h & 1))) h++;
    return sign | h;
  }
  let h = (e << 10) | (man >>> 13);
  const rem = man & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) h++;
  return sign | h;
}

function f16ToF32(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const man = h & 0x3ff;
  if (exp === 0) return sign * man * 2 ** -24;
  if (exp === 31) return man ? NaN : sign * Infinity;
  return sign * (1 + man / 1024) * 2 ** (exp - 15);
}

// snorm16/octEncode/octDecode moved to VoxelBrickCore.ts (three-free worker
// import chain, 2026-07-04) — imported below and re-exported above unchanged.

interface VertexCPU {
  pos: [number, number, number];
  nrm: [number, number, number];
  uv: [number, number];
  vdata: number;
}

export function decodeVertexCPU(verts: Uint32Array, vi: number): VertexCPU {
  const b = vi * VERT_WORDS;
  const uvw = verts[b + 4] as number;
  return {
    pos: [bitsF32(verts[b] as number), bitsF32(verts[b + 1] as number), bitsF32(verts[b + 2] as number)],
    nrm: octDecode(verts[b + 3] as number),
    uv: [f16ToF32(uvw & 0xffff), f16ToF32(uvw >>> 16)],
    vdata: verts[b + 5] as number,
  };
}

interface ClusterCPU {
  sphere: [number, number, number, number];
  coneAxis: [number, number, number];
  coneCos: number;
  triStart: number;
  triCount: number;
  flags: number;
  meshId: number;
}

export function decodeClusterCPU(recs: Uint32Array, ci: number): ClusterCPU {
  const b = ci * CLUSTER_WORDS;
  const w7 = recs[b + 7] as number;
  return {
    sphere: [
      bitsF32(recs[b] as number),
      bitsF32(recs[b + 1] as number),
      bitsF32(recs[b + 2] as number),
      bitsF32(recs[b + 3] as number),
    ],
    coneAxis: octDecode(recs[b + 4] as number),
    coneCos: bitsF32(recs[b + 5] as number),
    triStart: recs[b + 6] as number,
    triCount: w7 & 0xff,
    flags: (w7 >>> 8) & 0xff,
    meshId: w7 >>> 16,
  };
}

interface MeshCPU {
  clusterStart: number;
  clusterCount: number;
  instFirst: number;
  instCount: number;
  lodNext: number;
  lodDist: number;
  channel: number;
  matClass: number;
  flags: number;
  winQuads: number;
  hfOriginX: number;
  hfOriginZ: number;
  hfCellSize: number;
  quadsX: number;
  quadsZ: number;
  swayPad: number;
  /** mesh-local bounding sphere (heightfield: world-space) — instance cull */
  sphere: [number, number, number, number];
}

export function decodeMeshCPU(table: Uint32Array, mi: number): MeshCPU {
  const b = mi * MESH_WORDS;
  const w6 = table[b + 6] as number;
  const w10 = table[b + 10] as number;
  return {
    clusterStart: table[b] as number,
    clusterCount: table[b + 1] as number,
    instFirst: table[b + 2] as number,
    instCount: table[b + 3] as number,
    lodNext: table[b + 4] as number,
    lodDist: bitsF32(table[b + 5] as number),
    channel: w6 & 0xff,
    matClass: (w6 >>> 8) & 0xff,
    flags: (w6 >>> 16) & 0xff,
    winQuads: w6 >>> 24,
    hfOriginX: bitsF32(table[b + 7] as number),
    hfOriginZ: bitsF32(table[b + 8] as number),
    hfCellSize: bitsF32(table[b + 9] as number),
    quadsX: w10 & 0xffff,
    quadsZ: w10 >>> 16,
    swayPad: bitsF32(table[b + 11] as number),
    sphere: [
      bitsF32(table[b + 12] as number),
      bitsF32(table[b + 13] as number),
      bitsF32(table[b + 14] as number),
      bitsF32(table[b + 15] as number),
    ],
  };
}

/**
 * Conservative bounding sphere of a cluster-sphere set: center = extent-box
 * center, radius = max(dist(center, cᵢ) + rᵢ). Contains every cluster sphere
 * by construction (instance-level cull soundness needs containment, not
 * minimality).
 */
function meshSphereFromClusters(
  spheres: Float32Array,
  count: number,
): [number, number, number, number] {
  if (count === 0) return [0, 0, 0, 0];
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < count; i++) {
    const x = spheres[i * 4] as number;
    const y = spheres[i * 4 + 1] as number;
    const z = spheres[i * 4 + 2] as number;
    const r = spheres[i * 4 + 3] as number;
    minX = Math.min(minX, x - r);
    minY = Math.min(minY, y - r);
    minZ = Math.min(minZ, z - r);
    maxX = Math.max(maxX, x + r);
    maxY = Math.max(maxY, y + r);
    maxZ = Math.max(maxZ, z + r);
  }
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const cz = (minZ + maxZ) / 2;
  let rad = 0;
  for (let i = 0; i < count; i++) {
    const d =
      Math.hypot(
        (spheres[i * 4] as number) - cx,
        (spheres[i * 4 + 1] as number) - cy,
        (spheres[i * 4 + 2] as number) - cz,
      ) + (spheres[i * 4 + 3] as number);
    if (d > rad) rad = d;
  }
  return [cx, cy, cz, rad];
}

// ---------------------------------------------------------------------------
// TSL decode helpers (call inside an Fn() stack — they use .toVar())
// ---------------------------------------------------------------------------

export function octDecodeTsl(packed: NU): NV3 {
  const f = unpackSnormU(packed).toVar();
  const fx = f.x as NF;
  const fy = f.y as NF;
  const nz = float(1).sub(fx.abs()).sub(fy.abs()).toVar();
  const t = nz.negate().max(0).toVar();
  const nx = fx.add(fx.greaterThanEqual(0).select(t.negate(), t));
  const ny = fy.add(fy.greaterThanEqual(0).select(t.negate(), t));
  return normalize(vec3(nx, ny, nz)) as unknown as NV3;
}

export interface VertexNodes {
  pos: NV3;
  nrm: NV3;
  uv: NV2;
  vdata: NU;
}

export function readVertex(verts: StorageBufferNode<'uint'>, vi: NU): VertexNodes {
  const base = vi.mul(uint(VERT_WORDS)).toVar();
  const pos = vec3(
    bcU2F(elemU(verts, base)),
    bcU2F(elemU(verts, base.add(uint(1)))),
    bcU2F(elemU(verts, base.add(uint(2)))),
  ) as unknown as NV3;
  const nrm = octDecodeTsl(elemU(verts, base.add(uint(3))));
  const uv = unpackHalfU(elemU(verts, base.add(uint(4))));
  const vdata = elemU(verts, base.add(uint(5)));
  return { pos, nrm, uv, vdata };
}

export interface ClusterNodes {
  /** xyz center, w radius (mesh-local space) */
  sphere: NV4;
  coneAxis: NV3;
  /** -1 disables backface culling */
  coneCos: NF;
  /** global triangle index; heightfield: gx = lo16, gz = hi16 */
  triStart: NU;
  triCount: NU;
  flags: NU;
  meshId: NU;
}

export function readCluster(recs: StorageBufferNode<'uint'>, ci: NU): ClusterNodes {
  const base = ci.mul(uint(CLUSTER_WORDS)).toVar();
  const sphere = vec3(
    bcU2F(elemU(recs, base)),
    bcU2F(elemU(recs, base.add(uint(1)))),
    bcU2F(elemU(recs, base.add(uint(2)))),
  );
  const radius = bcU2F(elemU(recs, base.add(uint(3))));
  const coneAxis = octDecodeTsl(elemU(recs, base.add(uint(4))));
  const coneCos = bcU2F(elemU(recs, base.add(uint(5))));
  const triStart = elemU(recs, base.add(uint(6))).toVar();
  const w7 = elemU(recs, base.add(uint(7))).toVar();
  return {
    sphere: vec4(sphere, radius) as unknown as NV4,
    coneAxis,
    coneCos,
    triStart,
    triCount: w7.bitAnd(uint(0xff)),
    flags: w7.shiftRight(uint(8)).bitAnd(uint(0xff)),
    meshId: w7.shiftRight(uint(16)),
  };
}

interface DagRecordCPU {
  ownError: number;
  /** xyz center, w radius — the sphere ownError is measured against */
  ownSphere: [number, number, number, number];
  /** +∞ stored as DAG_ROOT_PARENT_ERR for roots */
  parentError: number;
  parentSphere: [number, number, number, number];
}

export function decodeDagCPU(dag: Float32Array, ci: number): DagRecordCPU {
  const b = ci * DAG_WORDS;
  return {
    ownError: dag[b] as number,
    ownSphere: [dag[b + 1] as number, dag[b + 2] as number, dag[b + 3] as number, dag[b + 4] as number],
    parentError: dag[b + 5] as number,
    parentSphere: [dag[b + 6] as number, dag[b + 7] as number, dag[b + 8] as number, dag[b + 9] as number],
  };
}

export interface DagNodes {
  ownError: NF;
  ownSphere: NV4;
  parentError: NF;
  parentSphere: NV4;
  /** N8-HIC: this cluster's children in gpu.dagLinks (global cluster ids) — the
   *  hierarchical-traversal descent. childCount > 0 only on a group's OWNER. */
  childBase: NU;
  childCount: NU;
}

export function readDag(dag: BufOf<NF>, ci: NU): DagNodes {
  const base = ci.mul(uint(DAG_WORDS)).toVar();
  return {
    ownError: dag.element(base),
    ownSphere: vec4(
      dag.element(base.add(uint(1))),
      dag.element(base.add(uint(2))),
      dag.element(base.add(uint(3))),
      dag.element(base.add(uint(4))),
    ) as unknown as NV4,
    parentError: dag.element(base.add(uint(5))),
    parentSphere: vec4(
      dag.element(base.add(uint(6))),
      dag.element(base.add(uint(7))),
      dag.element(base.add(uint(8))),
      dag.element(base.add(uint(9))),
    ) as unknown as NV4,
    childBase: bcF2U(dag.element(base.add(uint(10)))),
    childCount: bcF2U(dag.element(base.add(uint(11)))),
  };
}

/**
 * Interleave an ExplicitSource into the DAG_VERT_STRIDE layout buildDag expects
 * (pos@0, nrm@3, uv@6, vdata UNPACKED to 0..1 floats @8). The QEM build
 * interpolates uv/vdata linearly and renormalises the normal (normalOffset 3);
 * attachDag re-quantises the result back into the registry vertex format.
 */
export function explicitToDagVerts(src: ExplicitSource): Float32Array {
  const vCount = src.positions.length / 3;
  const out = new Float32Array(vCount * DAG_VERT_STRIDE);
  for (let v = 0; v < vCount; v++) {
    const o = v * DAG_VERT_STRIDE;
    out[o] = src.positions[v * 3] as number;
    out[o + 1] = src.positions[v * 3 + 1] as number;
    out[o + 2] = src.positions[v * 3 + 2] as number;
    out[o + 3] = src.normals[v * 3] as number;
    out[o + 4] = src.normals[v * 3 + 1] as number;
    out[o + 5] = src.normals[v * 3 + 2] as number;
    out[o + 6] = src.uvs ? (src.uvs[v * 2] as number) : 0;
    out[o + 7] = src.uvs ? (src.uvs[v * 2 + 1] as number) : 0;
    const d = src.vdata ? (src.vdata[v] as number) : 0;
    out[o + 8] = (d & 0xff) / 255;
    out[o + 9] = ((d >>> 8) & 0xff) / 255;
    out[o + 10] = ((d >>> 16) & 0xff) / 255;
    out[o + 11] = ((d >>> 24) & 0xff) / 255;
  }
  return out;
}

export interface MeshNodes {
  clusterStart: NU;
  clusterCount: NU;
  instFirst: NU;
  instCount: NU;
  /** LOD_NONE when last in chain */
  lodNext: NU;
  /** beyond this camera distance, use lodNext */
  lodDist: NF;
  channel: NU;
  matClass: NU;
  flags: NU;
  winQuads: NU;
  hfOriginX: NF;
  hfOriginZ: NF;
  hfCellSize: NF;
  /** total heightfield quads per axis (edge windows clamp against these) */
  quadsX: NU;
  quadsZ: NU;
  swayPad: NF;
  /** mesh-local bounding sphere (heightfield: world-space) — instance cull */
  sphere: NV4;
  /** N8-HIC: hierarchical-cull root seed range into gpu.dagLinks (0/0 = brute-force) */
  rootBase: NU;
  rootCount: NU;
}

export function readMesh(meshes: StorageBufferNode<'uint'>, mi: NU): MeshNodes {
  const base = mi.mul(uint(MESH_WORDS)).toVar();
  const w6 = elemU(meshes, base.add(uint(6))).toVar();
  const w10 = elemU(meshes, base.add(uint(10))).toVar();
  return {
    clusterStart: elemU(meshes, base),
    clusterCount: elemU(meshes, base.add(uint(1))),
    instFirst: elemU(meshes, base.add(uint(2))),
    instCount: elemU(meshes, base.add(uint(3))),
    lodNext: elemU(meshes, base.add(uint(4))),
    lodDist: bcU2F(elemU(meshes, base.add(uint(5)))),
    channel: w6.bitAnd(uint(0xff)),
    matClass: w6.shiftRight(uint(8)).bitAnd(uint(0xff)),
    flags: w6.shiftRight(uint(16)).bitAnd(uint(0xff)),
    winQuads: w6.shiftRight(uint(24)),
    hfOriginX: bcU2F(elemU(meshes, base.add(uint(7)))),
    hfOriginZ: bcU2F(elemU(meshes, base.add(uint(8)))),
    hfCellSize: bcU2F(elemU(meshes, base.add(uint(9)))),
    quadsX: w10.bitAnd(uint(0xffff)),
    quadsZ: w10.shiftRight(uint(16)),
    swayPad: bcU2F(elemU(meshes, base.add(uint(11)))),
    sphere: vec4(
      bcU2F(elemU(meshes, base.add(uint(12)))),
      bcU2F(elemU(meshes, base.add(uint(13)))),
      bcU2F(elemU(meshes, base.add(uint(14)))),
      bcU2F(elemU(meshes, base.add(uint(15)))),
    ) as unknown as NV4,
    rootBase: elemU(meshes, base.add(uint(16))),
    rootCount: elemU(meshes, base.add(uint(17))),
  };
}

// ---------------------------------------------------------------------------
// registry
// ---------------------------------------------------------------------------

interface MeshEntry {
  handle: number;
  label: string;
  matClass: MaterialClassId;
  channel: number;
  flags: number;
  winQuads: number;
  swayPad: number;
  /** explicit-mesh material param packed into word 7 (hfOriginX slot) */
  matParam: number;
  vertBase: number;
  vertCount: number;
  triBase: number;
  triCount: number;
  clusterBase: number;
  clusterCount: number;
  instFirst: number;
  instCount: number;
  lodNext: number;
  lodDist: number;
  /** voxel-foliage (spec §3 / Stage 3a): per-mesh NEAR draw envelope (m). The cull
   *  SEEDS this mesh's roots only when the instance distance ≥ nearDist (0 = unlimited
   *  near, the default). The mesh→voxel transition uses it: the VOXEL sibling head sets
   *  nearDist = transitionDist (renders only beyond the handoff) while the LEAF head's
   *  lodDist = transitionDist (renders only nearer) — a clean hard switch, no overlap.
   *  Stored in mesh word 8 (free on explicit/voxel meshes; only heightfields use word 8
   *  for hfOriginZ, and a heightfield is never a voxel head). */
  nearDist: number;
  /** N8-HIC: root seed range into gpu.dagLinks (the hierarchical-cull seeds);
   *  0/0 until attachDag wires the DAG. */
  rootBase: number;
  rootCount: number;
  /** mesh-local bounding sphere (heightfield: world-space) — instance cull */
  sphere: [number, number, number, number];
  hf?: {
    originX: number;
    originZ: number;
    cellSize: number;
    quadsX: number;
    quadsZ: number;
    windowsX: number;
    windowsZ: number;
  };
  /** staged until uploaded by build()/flush() */
  packedVerts?: Uint32Array;
  packedIdx?: Uint32Array;
  clusterRecs?: Uint32Array;
  stats?: ClusterStats;
  uploaded: boolean;
}

interface CpuInstances {
  meshId: number;
  first: number;
  a: Float32Array;
  b: Float32Array;
  copied: boolean;
}

interface GpuInstances {
  meshId: number;
  first: number;
  stream: InstanceStreamGPU;
  copied: boolean;
}

export interface RegistryGpu {
  verts: StorageBufferNode<'uint'>;
  /** N8-D2 Stage 2e: terrain-DAG verts, stride-1 (one packed texel coord per vert).
   *  Read in NaniteFetch's isHF&&isDAG branch as `hfVerts[vi]`; explicit meshes use
   *  the 6-word `verts`, the implicit window grid uses neither. */
  hfVerts: StorageBufferNode<'uint'>;
  indices: StorageBufferNode<'uint'>;
  clusters: StorageBufferNode<'uint'>;
  meshes: StorageBufferNode<'uint'>;
  /** vec4 records, 2 per instance: [i·2]=A (xyz,scale), [i·2+1]=B (yaw,leanX,leanZ,idF) */
  instances: BufOf<NV4>;
  instanceMesh: StorageBufferNode<'uint'>;
  /** N8-D1: parallel per-cluster DAG record (DAG_WORDS f32). Valid only where
   *  the cluster's CLUSTER_FLAG_DAG bit is set; zero elsewhere. */
  dag: BufOf<NF>;
  /** PERF-3: parallel per-cluster vertex-cache record (2 u32: vMin, count). count>0 ⇒
   *  the cluster's vertex indices occupy the tight range [vMin, vMin+count) (count ≤
   *  VCACHE_VERTS) so the SW raster cooperatively transforms them ONCE into workgroup
   *  shared memory; count=0 ⇒ per-thread fallback (terrain / window-grid / wide ranges). */
  vcompact: StorageBufferNode<'uint'>;
  /** N8-HIC: hierarchical-cull links — flat u32 of GLOBAL cluster ids. Per mesh:
   *  [roots…] (mesh.rootBase/rootCount) then owner clusters' [children…]
   *  (dag.childBase/childCount). The BFS traversal seeds from the roots and descends
   *  via the children, replacing the brute-force all-clusters dispatch. */
  dagLinks: StorageBufferNode<'uint'>;
  /** voxel-foliage (spec §4.2): the 11th RegistryGpu buffer. Flat u32, BRICK_WORDS
   *  per brick (VoxelBrick.ts codec). A VOXEL cluster (matClass=voxel(7)) points at
   *  its brick range via word6=brickBase and word7-lowbyte=brickCount (§4.1) — the
   *  cluster mega-buffer stride/layout is UNCHANGED (those reinterpret triStart/triCount
   *  ONLY on voxel meshes, never fed to the triangle raster). MUST NOT be bound on
   *  every stage — only the voxel raster permutation + a resolve slice (§4.2/§4.6);
   *  the DAG sidecar precedent (kept off the cut kernel to stay ≤10 bindings, §4.2).
   *  Empty worlds get a 1-word placeholder so the node always exists. */
  voxelBricks: StorageBufferNode<'uint'>;
}

export class GeometryRegistry {
  private readonly late: LateBudget;
  private readonly entries: MeshEntry[] = [];
  private readonly cpuStreams: CpuInstances[] = [];
  private readonly gpuStreams: GpuInstances[] = [];
  private vertCursor = 0;
  /** N8-D2 Stage 2e: monotonic cursor into the SEPARATE stride-1 terrain-DAG vertex
   *  buffer (hfVertsArr). Independent of vertCursor (the 6-word explicit buffer). */
  private hfVertCursor = 0;
  private triCursor = 0;
  private clusterCursor = 0;
  private instCursor = 0;
  /** voxel-foliage: monotonic cursor into voxelBricksArr (in BRICKS). The offline
   *  voxelizer appends bricks post-build; capped at this.late.bricks (frozen at build). */
  private brickCursor = 0;
  private clusterizeMs = 0;
  private built = false;

  // N8-D2 Stage 2a: streaming terrain tile POOL (D-N39). A fixed set of S slots,
  // each a constant-capacity byte block at poolBase+slot*cap, with a stable
  // heightfield mesh handle + identity instance. Streaming (re)loads a region
  // into a free slot (attachHeightDagTile — overwrites in place) and releases it
  // (evictHeightDagTile — zeroes the mesh draw, no tombstoning needed since the
  // cull is instance-driven). Bounded memory + cull regardless of field size;
  // O(1) alloc/free, no fragmentation (the Nanite "page" model).
  private tilePool: { slots: number; vertCap: number; triCap: number; clusterCap: number; dagLinksCap: number } | null = null;
  /** `vert` indexes the stride-1 hf vertex buffer (2e); tri/cluster/dagLinks index the shared buffers. */
  private tilePoolBase = { vert: 0, tri: 0, cluster: 0, dagLinks: 0 };
  private tilePoolHandles: number[] = [];
  private tileFreeSlots: number[] = [];
  /** handle resident in each slot, or -1 if free (parallel to the free-stack) */
  private tileSlotOccupant: Int32Array | null = null;
  /** S6f: PARKED slots — the retained parent payload of a Split node. Draw is
   *  killed (rootCount→0 ⇒ kSeedRoots skips it) but the slot + geometry stay
   *  resident, so coarsening is an instant unpark. Maps slot → its true rootCount
   *  (restored on unpark). Geometry/spheres stay truthful, so rebaseTilePoolOrigins
   *  needs no special case. */
  private tileSlotParkedRootCount = new Map<number, number>();

  /** S7 instance pool (SPEC-STREAMING-WORLD §5, A4): a FLAT reserved region of
   *  `capacity` instance slots pre-parked off-world at build() so the frozen cull
   *  dispatch (registry.instanceCount) already covers it. Streamed trees/boulders
   *  are written per-BLOCK (`rewriteInstanceBlock`) — each slot's meshId is set
   *  per-instance, so ONE block can host many species (unlike the per-mesh
   *  consecutive-stream bindInstances path). NULL on the generated world (its
   *  instances stay boot-bound; the pool is ADDITIONAL streamed-only capacity ⇒
   *  instanceCount is byte-identical there). A-words are StreamOrigin-relative
   *  (poolOrigin subtract on write; rebaseInstanceOrigins shifts the region). */
  private instPool: { first: number; blockSize: number; blocks: number; capacity: number } | null = null;
  private instBlockFree: number[] = [];
  /** parked-slot y (off any frustum — frustum-culled at NaniteCull kSeedRoots
   *  regardless of the parked meshId, so a freed/parked slot is ~free). */
  private static readonly INST_PARK_Y = -1e6;

  // backing arrays === attribute arrays (created at build; capacity-sized)
  private vertsArr!: Uint32Array;
  /** N8-D2 Stage 2e: stride-1 terrain-DAG vertex buffer (one packed texel coord/vert). */
  private hfVertsArr!: Uint32Array;
  private idxArr!: Uint32Array;
  private clusterArr!: Uint32Array;
  private meshArr!: Uint32Array;
  private instArr!: Float32Array;
  private instMeshArr!: Uint32Array;
  /** N8-D1: parallel per-cluster DAG records (DAG_WORDS f32 each) */
  private dagArr!: Float32Array;
  /** PERF-3: parallel per-cluster vertex-cache record (vMin, count) — see RegistryGpu.vcompact */
  private vcompactArr!: Uint32Array;
  /** N8-HIC: hierarchical-cull links (global cluster ids) — see RegistryGpu.dagLinks */
  private dagLinksArr!: Uint32Array;
  /** monotonic cursor into dagLinksArr; attachDag appends [roots, children] per DAG */
  private dagLinksCursor = 0;
  /** item 6: the deepest root->leaf anchor chain (in NODES) across every attached DAG -
   *  the MINIMUM number of GPU BFS passes (NaniteCull hierDepth) that still emits every
   *  leaf (fewer => HOLES). Folded up as DAGs attach; exposed via maxDagDepth. Starts 1
   *  (a flat / no-DAG world traverses in one pass). */
  private _maxDagDepth = 1;
  /** A14 (SPEC-STREAMING-WORLD §9b): the BFS pass count the cull FROZE at frame
   *  build. Once set, any attach that grows the depth PAST it would silently
   *  never emit its deep leaves (holes) — so growth past the freeze throws. */
  private frozenHierDepth: number | null = null;
  /** voxel-foliage: flat brick records (BRICK_WORDS u32 each) — see RegistryGpu.voxelBricks */
  private voxelBricksArr!: Uint32Array;

  private vertsAttr!: StorageBufferAttribute;
  private hfVertsAttr!: StorageBufferAttribute;
  private idxAttr!: StorageBufferAttribute;
  private clusterAttr!: StorageBufferAttribute;
  private meshAttr!: StorageBufferAttribute;
  private instAttr!: StorageBufferAttribute;
  private instMeshAttr!: StorageBufferAttribute;
  private dagAttr!: StorageBufferAttribute;
  private vcompactAttr!: StorageBufferAttribute;
  private dagLinksAttr!: StorageBufferAttribute;
  private voxelBricksAttr!: StorageBufferAttribute;
  /** D (memory arc): set once releaseImmutableMirrors() has dropped verts + brick mirrors. */
  private mirrorsReleased = false;

  private caps!: { verts: number; tris: number; clusters: number; meshes: number; instances: number };
  /** N8-D2 Stage 2e: capacity (in verts = words) of the stride-1 terrain-DAG buffer. */
  private hfCap = 0;
  /** voxel-foliage: capacity (in BRICKS) of voxelBricksArr — frozen at build() from
   *  this.late.bricks (§5.3). 0 ⇒ a 1-word placeholder buffer. */
  private brickCap = 0;
  private instRW!: BufOf<V4W>;
  private instMeshRW!: StorageBufferNode<'uint'>;

  /** read-only views for downstream kernels — valid after build() */
  gpu!: RegistryGpu;

  constructor(opts?: { late?: Partial<LateBudget> }) {
    this.late = {
      verts: 0,
      hfVerts: 0,
      tris: 0,
      clusters: 0,
      meshes: 0,
      instances: 0,
      bricks: 0,
      ...opts?.late,
    };
  }

  get meshCount(): number {
    return this.entries.length;
  }
  /** RP-1 (deep-review 16 tri-class specialization): matClass IDS with ≥1 registered
   *  mesh. The resolve strips ABSENT classes' shading subgraphs at build (register/
   *  occupancy pressure — the vox-pass 37.5ms-cliff mechanism). Reflects registration
   *  state WHEN CALLED: a mesh class registered AFTER the resolve builds would fall to
   *  the gray-slab default if its class was stripped (the resolve warns what it strips;
   *  ?resclasses=0 disables stripping). */
  get presentClasses(): ReadonlySet<number> {
    const s = new Set<number>();
    for (const e of this.entries) s.add(MATERIAL_CLASS[e.matClass]);
    return s;
  }
  get clusterCount(): number {
    return this.clusterCursor;
  }
  /** item 6: deepest root->leaf DAG anchor chain (in NODES) over all attached DAGs.
   *  The cull's BFS pass count (hierDepth) MUST be >= this to emit every leaf (under
   *  = holes). Stable by first frame (veg DAGs attach at build; terrain tiles share one
   *  uniform grid depth). >= 1 always. */
  get maxDagDepth(): number {
    return this._maxDagDepth;
  }
  /** A14: the frame build calls this with the hierDepth it bakes into the cull
   *  pipelines; later attaches assert against it (throw-loud, never silent holes). */
  freezeHierDepth(depth: number): void {
    if (this._maxDagDepth > depth) {
      throw new Error(`GeometryRegistry: freezing hierDepth ${depth} below the attached max DAG depth ${this._maxDagDepth}`);
    }
    this.frozenHierDepth = depth;
  }
  /** fold an attached DAG's chain depth into the global max — throws if a frozen
   *  cull pass count could no longer emit every leaf (A14) */
  private growDagDepth(depth: number): void {
    if (depth > this._maxDagDepth) this._maxDagDepth = depth;
    if (this.frozenHierDepth !== null && this._maxDagDepth > this.frozenHierDepth) {
      throw new Error(
        `GeometryRegistry: attached DAG depth ${this._maxDagDepth} exceeds the frozen cull hierDepth ` +
          `${this.frozenHierDepth} — its deep leaves would silently never render (rebuild the frame or raise the margin)`,
      );
    }
  }
  get triCount(): number {
    return this.triCursor;
  }
  get vertCount(): number {
    return this.vertCursor;
  }
  get instanceCount(): number {
    return this.instCursor;
  }
  /** voxel-foliage: bricks appended so far (≤ brickCap, frozen at build). */
  get brickCount(): number {
    return this.brickCursor;
  }
  /** voxel-foliage: reserved brick capacity (frozen at build from late.bricks). */
  get brickCapacity(): number {
    return this.brickCap;
  }
  get isBuilt(): boolean {
    return this.built;
  }

  /**
   * N8-D1: raise the late-registration budget before build(). attachDag() runs
   * post-build (the DAG is built off the boot clusterize path) but caps freeze
   * at build(), so the appended verts/tris/clusters must be reserved here first.
   * Throws once built. Missing fields default 0.
   */
  addLate(b: Partial<LateBudget>): void {
    if (this.built) throw new Error('GeometryRegistry: addLate after build()');
    this.late.verts += b.verts ?? 0;
    this.late.hfVerts += b.hfVerts ?? 0;
    this.late.tris += b.tris ?? 0;
    this.late.clusters += b.clusters ?? 0;
    this.late.meshes += b.meshes ?? 0;
    this.late.instances += b.instances ?? 0;
    this.late.bricks += b.bricks ?? 0;
  }

  /**
   * voxel-foliage (§4.2/§5.4): append `count` brick records post-build and upload
   * them. Returns the BASE brick index of the allocated contiguous range — the voxel
   * cluster record's word6 (brickBase) points here, word7-lowbyte = brickCount (§4.1).
   *
   * The voxelizer fills the freshly-allocated records via `fill(bricks, base)`, which
   * receives the WHOLE backing array + the base brick index (use the VoxelBrick.ts
   * `writeBrick(bricks, base + i, …)` codec). Throws on capacity overflow (caps froze
   * at build — raise late.bricks, F14: explicit never silent). build() must have run.
   */
  appendBricks(count: number, fill: (bricks: Uint32Array, base: number) => void): number {
    if (!this.built) throw new Error('GeometryRegistry: appendBricks before build()');
    // F-5: post-release, three's upload path for this buffer is dead BY DESIGN — the
    // only legal writes are the fartile pool's writeBuffer-direct path. A late append
    // here would silently write into a null mirror.
    if (this.mirrorsReleased) throw new Error('GeometryRegistry: appendBricks after releaseImmutableMirrors — the library brick region is frozen');
    if (count <= 0) return this.brickCursor;
    // S8: the fartile brick pool is a TAIL region of this buffer — library appends
    // grow upward and must never cross into it.
    const libCap = this.ftPool ? this.ftPoolBase.brick : this.brickCap;
    if (this.brickCursor + count > libCap) {
      throw new Error(
        `GeometryRegistry: voxel brick capacity exceeded (${this.brickCursor}+${count}/${libCap}` +
          `${this.ftPool ? ` library region; fartile pool tail holds ${this.brickCap - libCap}` : ''}) — raise the late.bricks budget`,
      );
    }
    const base = this.brickCursor;
    fill(this.voxelBricksArr, base);
    this.brickCursor += count;
    this.pushRange(this.voxelBricksAttr, base * BRICK_WORDS, count * BRICK_WORDS);
    return base;
  }

  /**
   * voxel-foliage (spec §4.1 / Stage-2 A1): register a VOXEL sibling head whose clusters
   * carry BRICK ranges (word6=brickBase, word7-lowbyte=brickCount ≤ MAX_BRICKS_PER_CLUSTER)
   * instead of triangles. NO geometry (0 verts / 0 tris) — the placeholder triangle path
   * the draft used (re-clusterizing the full leaf crown) OVERFLOWED the late caps (§A1);
   * authoring cluster records DIRECTLY costs only `clusterBlocks.length` clusters and
   * keeps the bricks as the sole payload. The cluster mega-buffer stride/layout is
   * UNCHANGED — word6/word7 are reinterpreted on voxel meshes (matClass=voxel keeps them
   * out of the triangle raster, §4.1). The brick band is split into ≤128-brick BLOCKS
   * (the §5.3 per-coarse-cluster fit), each one cluster with its own brick-AABB bound.
   *
   * The cull is HIER-ONLY (kSeedRoots SKIPS rootCount==0 meshes, NaniteCull.ts:469), so
   * each cluster gets a MINIMAL single-root DAG record (ownError=0 ⇒ pOwn=0 ≤ τ ⇒ always
   * CUT/emit, never descend; childCount=0; root sentinel parent keeps the sqrt(d²−r²)
   * denominator well-formed) and a dagLinks root, with rootBase/rootCount = block count.
   * Runs AFTER build() within the late budget (reserve clusters = Σ blocks, meshes = 1).
   */
  registerVoxelHead(
    matParam: number,
    blocks: {
      brickBase: number;
      brickCount: number;
      aabb: { min: [number, number, number]; max: [number, number, number] };
      /** voxlod (G1 voxlod=1): real multi-level DAG cut metadata. ABSENT on EVERY block =>
       *  the degenerate single-resolution path (voxlod=0): each block its own always-cut root,
       *  byte-identical to before. PRESENT => author a real per-level DAG (own/parent error +
       *  sphere, spatial-tree child links, roots=coarsest). MIXED is rejected. */
      dag?: {
        ownError: number;
        ownSphere: [number, number, number, number];
        parentError?: number;
        parentSphere?: [number, number, number, number];
        /** finer child cluster ids as GLOBAL block indices into THIS `blocks` list. */
        childClusterIdx: number[];
        isRoot: boolean;
        dagLevel: number;
      };
    }[],
    opts: { swayPad?: number; maxDist?: number; label?: string },
  ): MeshHandle {
    if (!this.built) throw new Error('GeometryRegistry: registerVoxelHead before build()');
    if (blocks.length === 0) throw new Error('GeometryRegistry: registerVoxelHead with no brick blocks');
    const handle = this.entries.length;
    if (handle >= 0xffff) throw new Error('GeometryRegistry: mesh id exceeds u16');
    if (handle >= this.caps.meshes) {
      throw new Error(`GeometryRegistry: mesh capacity ${this.caps.meshes} exceeded — raise late.meshes`);
    }
    const n = blocks.length;
    this.checkRoom(0, 0, n, 0); // n clusters, 0 verts/tris (bricks are the payload)
    const cBase = this.clusterCursor;
    const linkBase = this.dagLinksCursor;

    // voxlod (G1): does this head carry a REAL multi-level DAG? Either ALL blocks have `dag`
    // (voxlod=1) or NONE (voxlod=0) — a mix is a build bug. The dagLinks layout is the SAME
    // [roots…][children…] the mesh DAG (attachDag) uses, so it reuses the cull verbatim.
    const dagOn = (blocks[0] as { dag?: unknown }).dag !== undefined;
    let rootIdx: number[]; // GLOBAL-block indices that seed the traversal
    let childTotal = 0;    // total child links across all blocks
    if (dagOn) {
      rootIdx = [];
      for (let c = 0; c < n; c++) {
        const d = (blocks[c] as { dag?: { isRoot: boolean; childClusterIdx: number[] } }).dag;
        if (!d) throw new Error('GeometryRegistry: registerVoxelHead mixed dag/non-dag blocks');
        if (d.isRoot) rootIdx.push(c);
        childTotal += d.childClusterIdx.length;
      }
      if (rootIdx.length === 0) throw new Error('GeometryRegistry: voxel DAG has no roots');
    } else {
      // voxlod=0: every block is its own root, no children (the degenerate always-cut DAG).
      rootIdx = [];
      for (let c = 0; c < n; c++) rootIdx.push(c);
    }
    const rootCount = rootIdx.length;
    const linkTotal = rootCount + childTotal;
    if (linkBase + linkTotal > this.dagLinksArr.length) {
      throw new Error(`GeometryRegistry: dagLinks overflow authoring voxel head (${linkBase + linkTotal} > ${this.dagLinksArr.length})`);
    }
    const entry = this.newEntry(handle, 'voxel', {
      transformChannel: 'leaf',
      castShadows: false,
      twoSided: true,
      aggregate: true,
      matParam,
      swayPad: opts.swayPad ?? 0,
      label: opts.label ?? 'voxel',
    }, false, 0);
    entry.vertBase = this.vertCursor;
    entry.vertCount = 0;
    entry.triBase = this.triCursor;
    entry.triCount = 0;
    entry.clusterBase = cBase;
    entry.clusterCount = n;
    entry.rootBase = linkBase;
    entry.rootCount = rootCount;
    entry.flags |= MESH_FLAG_HASDAG; // hier mesh: kSeedRoots seeds it; resolve/dbg consistent
    if ((opts.label ?? '') === 'fartile') entry.flags |= MESH_FLAG_FARTILE;
    entry.lodNext = LOD_NONE;
    entry.lodDist = opts.maxDist ?? 0;

    const recs = new Uint32Array(n * CLUSTER_WORDS);
    const dArr = this.dagArr;
    const dl = this.dagLinksArr;
    // dagLinks layout (mirrors attachDag): [roots…][children…]. Roots first; the children
    // section is a flat concat of each block's child cluster ids, in block order. Each
    // block's childBase (DAG word10) points into the children section; childCount = word11.
    for (let i = 0; i < rootCount; i++) dl[linkBase + i] = cBase + (rootIdx[i] as number);
    const childSectionBase = linkBase + rootCount;
    let childCursor = childSectionBase;
    // pre-pass: per-block childBase/childCount within the children section (0 unless owner)
    const blockChildBase = new Int32Array(n);
    const blockChildCount = new Int32Array(n);
    if (dagOn) {
      for (let c = 0; c < n; c++) {
        const d = (blocks[c] as { dag: { childClusterIdx: number[] } }).dag;
        const kids = d.childClusterIdx;
        blockChildBase[c] = childCursor;
        blockChildCount[c] = kids.length;
        for (let k = 0; k < kids.length; k++) dl[childCursor + k] = cBase + (kids[k] as number);
        childCursor += kids.length;
      }
    }
    // union AABB → mesh sphere (the instance-cull seed bound)
    let uMinX = Infinity, uMinY = Infinity, uMinZ = Infinity;
    let uMaxX = -Infinity, uMaxY = -Infinity, uMaxZ = -Infinity;
    for (let c = 0; c < n; c++) {
      const blk = blocks[c] as {
        brickBase: number; brickCount: number;
        aabb: { min: [number, number, number]; max: [number, number, number] };
        dag?: { ownError: number; ownSphere: [number, number, number, number]; parentError?: number; parentSphere?: [number, number, number, number]; dagLevel: number };
      };
      if (blk.brickCount > MAX_CLUSTER_TRIS) {
        // word7 low byte is a u8 AND the brick-granular id is ≤7 bits (§4.5/§6.4).
        throw new Error(
          `GeometryRegistry: voxel cluster brickCount ${blk.brickCount} > ${MAX_CLUSTER_TRIS} — split into finer blocks (§5.3)`,
        );
      }
      const mn = blk.aabb.min, mx = blk.aabb.max;
      const cx = (mn[0] + mx[0]) * 0.5, cy = (mn[1] + mx[1]) * 0.5, cz = (mn[2] + mx[2]) * 0.5;
      const rr = Math.hypot((mx[0] - mn[0]) * 0.5, (mx[1] - mn[1]) * 0.5, (mx[2] - mn[2]) * 0.5);
      uMinX = Math.min(uMinX, mn[0]); uMinY = Math.min(uMinY, mn[1]); uMinZ = Math.min(uMinZ, mn[2]);
      uMaxX = Math.max(uMaxX, mx[0]); uMaxY = Math.max(uMaxY, mx[1]); uMaxZ = Math.max(uMaxZ, mx[2]);
      const b = c * CLUSTER_WORDS;
      recs[b] = f32Bits(cx);
      recs[b + 1] = f32Bits(cy);
      recs[b + 2] = f32Bits(cz);
      recs[b + 3] = f32Bits(rr);
      recs[b + 4] = octEncode(0, 1, 0);
      recs[b + 5] = f32Bits(-1); // cone disabled (bricks two-sided)
      recs[b + 6] = blk.brickBase >>> 0; // word6 = brickBase (was triStart)
      // word7: brickCount(0-7) | flags(8-9)=0 | LOD level(10-15) | handle(16-31). flags=0
      // (NOT CLUSTER_FLAG_DAG): the raster SKIPS voxel clusters by matClass + the cull reads
      // childBase/Count straight from the DAG record, so the flag is unneeded; bits 10-15
      // carry the PYRAMID level for ?nanitedbg=lod (0 on voxlod=0; 0=finest/L0 … higher=coarser
      // on voxlod=1). The raster is UNCHANGED (G3) — these bits are debug-only.
      const dagLevel = dagOn ? (blk.dag?.dagLevel ?? 0) & 0x3f : 0;
      recs[b + 7] = ((blk.brickCount & 0xff) | (dagLevel << 10) | (handle << 16)) >>> 0;

      const db = (cBase + c) * DAG_WORDS;
      if (dagOn) {
        // REAL per-level cut record: ownError = the level's geometric error (LOCAL metres,
        // matched to the mesh DAG scale, G2); ownSphere = the block's containing sphere; the
        // parent pair (the coarser level) drives the descend; childBase/Count wire the finer
        // level. Roots: parentError +∞ sentinel + parentSphere←own (keeps sqrt(d²−r²) sane).
        const d = blk.dag as { ownError: number; ownSphere: [number, number, number, number]; parentError?: number; parentSphere?: [number, number, number, number] };
        const root = d.parentError === undefined || !Number.isFinite(d.parentError);
        const os = d.ownSphere;
        dArr[db] = d.ownError;
        dArr[db + 1] = os[0]; dArr[db + 2] = os[1]; dArr[db + 3] = os[2]; dArr[db + 4] = os[3];
        if (root) {
          dArr[db + 5] = DAG_ROOT_PARENT_ERR;
          dArr[db + 6] = os[0]; dArr[db + 7] = os[1]; dArr[db + 8] = os[2]; dArr[db + 9] = os[3];
        } else {
          const ps = d.parentSphere as [number, number, number, number];
          dArr[db + 5] = d.parentError as number;
          dArr[db + 6] = ps[0]; dArr[db + 7] = ps[1]; dArr[db + 8] = ps[2]; dArr[db + 9] = ps[3];
        }
        dArr[db + 10] = bitsF32(blockChildBase[c] as number);
        dArr[db + 11] = bitsF32(blockChildCount[c] as number);
      } else {
        // voxlod=0: single-root DAG record per cluster (always-cut, no children) + a root.
        dArr[db] = 0; // ownError = 0 → pOwn=0 ≤ τ → always cut → emit
        dArr[db + 1] = cx; dArr[db + 2] = cy; dArr[db + 3] = cz; dArr[db + 4] = rr; // ownSphere
        dArr[db + 5] = DAG_ROOT_PARENT_ERR; // root sentinel parentError
        dArr[db + 6] = cx; dArr[db + 7] = cy; dArr[db + 8] = cz; dArr[db + 9] = rr; // parentSphere←own
        dArr[db + 10] = bitsF32(0); // childBase
        dArr[db + 11] = bitsF32(0); // childCount = 0 (leaf root)
      }
    }
    entry.sphere = [
      (uMinX + uMaxX) * 0.5,
      (uMinY + uMaxY) * 0.5,
      (uMinZ + uMaxZ) * 0.5,
      Math.hypot((uMaxX - uMinX) * 0.5, (uMaxY - uMinY) * 0.5, (uMaxZ - uMinZ) * 0.5),
    ];
    entry.clusterRecs = recs;
    this.entries.push(entry);
    this.clusterCursor += n;
    this.dagLinksCursor = linkBase + linkTotal;
    // voxlod: a deeper DAG needs more BFS ping-pong passes to EMIT every leaf (the hierDepth
    // floor). The pyramid is a SPATIAL TREE (roots=coarsest -> ... -> finest=L0), so the TRUE
    // longest root->leaf chain is the depth. Build the block-local hierarchy from each block's
    // child-cluster ids + roots and fold the real longest chain into the global max (under =>
    // holes; over => wasted pass).
    if (dagOn) {
      const lcStart = new Uint32Array(n);
      const lcCount = new Uint32Array(n);
      const lcIdx: number[] = [];
      const lcRoots: number[] = [];
      for (let c = 0; c < n; c++) {
        const d = (blocks[c] as { dag: { childClusterIdx: number[]; isRoot: boolean } }).dag;
        if (d.isRoot) lcRoots.push(c);
        lcStart[c] = lcIdx.length;
        lcCount[c] = d.childClusterIdx.length;
        for (const k of d.childClusterIdx) lcIdx.push(k);
      }
      const localHier = {
        childStart: lcStart,
        childCount: lcCount,
        childIndices: Uint32Array.from(lcIdx),
        rootIndices: Uint32Array.from(lcRoots),
      };
      this.growDagDepth(maxChainDepth(localHier));
    }

    // upload: copyEntry (clusters + mesh record) at flush; the DAG/dagLinks ranges are
    // written directly here (copyEntry does not touch them), so push them now.
    this.pushRange(this.dagAttr, cBase * DAG_WORDS, n * DAG_WORDS);
    this.pushRange(this.dagLinksAttr, linkBase, linkTotal);
    return handle;
  }

  meshEntry(h: MeshHandle): Readonly<MeshEntry> {
    const e = this.entries[h];
    if (!e) throw new Error(`GeometryRegistry: unknown mesh handle ${h}`);
    return e;
  }

  registerMesh(src: ClusterSource, matClass: MaterialClassId, opts: RegisterOpts = {}): MeshHandle {
    const handle = this.entries.length;
    if (handle >= 0xffff) throw new Error('GeometryRegistry: mesh id exceeds u16');
    if (this.built && handle >= this.caps.meshes) {
      throw new Error(
        `GeometryRegistry: mesh capacity ${this.caps.meshes} exceeded post-build — raise late.meshes`,
      );
    }
    const entry =
      src.kind === 'heightfield'
        ? this.packHeightfield(handle, src, matClass, opts)
        : this.packExplicit(handle, src, matClass, opts);
    this.entries.push(entry);
    return handle;
  }

  /**
   * Append a coarser discrete LOD to a mesh's chain (transitional until the
   * N8 DAG): beyond `switchDist`, culling uses the new entry's clusters. The
   * LOD entry shares the head's instances (instFirst/Count stay 0 on it).
   */
  registerLod(h: MeshHandle, src: ClusterSource, switchDist: number): MeshHandle {
    let tail = this.meshEntry(h) as MeshEntry;
    while (tail.lodNext !== LOD_NONE) tail = this.entries[tail.lodNext] as MeshEntry;
    const head = this.entries[h] as MeshEntry;
    const channelName = (Object.keys(TRANSFORM_CHANNEL) as TransformChannel[]).find(
      (key) => TRANSFORM_CHANNEL[key] === head.channel,
    );
    const lod = this.registerMesh(src, head.matClass, {
      transformChannel: channelName ?? 'rigid',
      castShadows: (head.flags & MESH_FLAG_CAST_SHADOWS) !== 0,
      swayPad: head.swayPad,
      matParam: head.matParam, // LOD bark shares the head's texture-array slice
      label: `${head.label}/lod`,
    });
    tail.lodNext = lod;
    tail.lodDist = switchDist;
    if (this.built && tail.uploaded) this.rewriteMeshRecord(tail);
    return lod;
  }

  /**
   * Set the chain's maximum draw distance: the TAIL keeps lodNext = NONE and
   * its lodDist becomes the cull-beyond envelope (0 = unlimited). During the
   * hybrid stage this mirrors the old path's ring/impostor envelope — the
   * far field belongs to impostors until the N8 DAG bottoms out.
   */
  setMaxDistance(h: MeshHandle, d: number): void {
    let tail = this.meshEntry(h) as MeshEntry;
    while (tail.lodNext !== LOD_NONE) tail = this.entries[tail.lodNext] as MeshEntry;
    tail.lodDist = d;
    if (this.built && tail.uploaded) this.rewriteMeshRecord(tail);
  }

  /**
   * voxel-foliage (spec §3 / Stage 3a): set the mesh's NEAR draw envelope (m) — the cull
   * seeds the mesh's roots ONLY when the instance distance ≥ d (0 = unlimited near). The
   * VOXEL sibling head sets this to transitionDist so it renders only BEYOND the handoff,
   * while the LEAF head's maxDist = transitionDist keeps it nearer — a clean hard switch
   * (no overlap, no gap). The head carries it; LODs are irrelevant (voxel heads are flat).
   * The mesh is the explicit/voxel kind (hf undefined), so word 8 is free for nearDist.
   */
  setNearDistance(h: MeshHandle, d: number): void {
    const e = this.meshEntry(h) as MeshEntry;
    if (e.hf) throw new Error('GeometryRegistry: setNearDistance on a heightfield mesh (word 8 = hfOriginZ)');
    e.nearDist = d;
    if (this.built && e.uploaded) this.rewriteMeshRecord(e);
  }

  bindInstances(h: MeshHandle, stream: InstanceStream): void {
    const e = this.meshEntry(h) as MeshEntry;
    const count = 'a' in stream ? stream.a.length / 4 : stream.count;
    if (!Number.isInteger(count)) throw new Error('GeometryRegistry: instance stream count not integral');
    if ('a' in stream && stream.b.length !== stream.a.length) {
      throw new Error('GeometryRegistry: instance A/B lengths differ');
    }
    if (e.instCount > 0 && e.instFirst + e.instCount !== this.instCursor) {
      throw new Error(
        `GeometryRegistry: instance streams for mesh ${e.label} must be bound consecutively`,
      );
    }
    this.checkRoom(0, 0, 0, count);
    if (e.instCount === 0) e.instFirst = this.instCursor;
    const first = this.instCursor;
    e.instCount += count;
    this.instCursor += count;
    if ('a' in stream) {
      this.cpuStreams.push({ meshId: h, first, a: stream.a, b: stream.b, copied: false });
    } else {
      this.gpuStreams.push({ meshId: h, first, stream, copied: false });
    }
    if (this.built && e.uploaded) this.rewriteMeshRecord(e);
  }

  /**
   * Pack everything registered so far into the mega-buffers. Call once after
   * boot registration; later registrations go through flush(). `renderer` is
   * required only when GPU instance streams are pending (copy kernels);
   * node-side probes may omit it.
   */
  build(renderer?: Renderer, counters?: Record<string, number>): BuildReport {
    if (this.built) throw new Error('GeometryRegistry: build() called twice — use flush()');
    const t0 = performance.now();
    this.caps = {
      verts: this.vertCursor + this.late.verts,
      tris: this.triCursor + this.late.tris,
      clusters: this.clusterCursor + this.late.clusters,
      meshes: this.entries.length + this.late.meshes,
      instances: this.instCursor + this.late.instances,
    };
    // N8-D2 Stage 2e: the stride-1 terrain-DAG buffer is sized independently — every
    // hf vert is 1 word (a packed texel coord), vs VERT_WORDS=6 for explicit verts.
    this.hfCap = this.hfVertCursor + this.late.hfVerts;
    // voxel-foliage (§4.2/§5.3): the brick buffer is sized in BRICKS (BRICK_WORDS each),
    // frozen here. The offline voxelizer appends bricks post-build (like attachDag) up to
    // this cap — under-reservation throws (appendBricks), over-reservation wastes memory.
    this.brickCap = this.brickCursor + this.late.bricks;
    this.vertsArr = new Uint32Array(Math.max(1, this.caps.verts * VERT_WORDS));
    this.hfVertsArr = new Uint32Array(Math.max(1, this.hfCap));
    this.idxArr = new Uint32Array(Math.max(1, this.caps.tris * 3));
    this.clusterArr = new Uint32Array(Math.max(1, this.caps.clusters * CLUSTER_WORDS));
    this.meshArr = new Uint32Array(Math.max(1, this.caps.meshes * MESH_WORDS));
    this.instArr = new Float32Array(Math.max(8, this.caps.instances * 8));
    this.instMeshArr = new Uint32Array(Math.max(1, this.caps.instances));
    const dagLen = Math.max(DAG_WORDS, this.caps.clusters * DAG_WORDS);
    this.dagArr = new Float32Array(dagLen);
    const vcompactLen = Math.max(2, this.caps.clusters * 2);
    this.vcompactArr = new Uint32Array(vcompactLen);
    // N8-HIC: hierarchical-cull links (global cluster ids) — per mesh: its roots,
    // then the owner clusters' children. Upper bound = 2× clusters (each cluster is
    // an input to ≤1 group ⇒ children ≤ clusters; roots ≤ clusters). Auto-sized, so
    // no caller reservation. Cursor appended alongside the DAG in attachDag.
    const dagLinksLen = Math.max(1, this.caps.clusters * 2);
    this.dagLinksArr = new Uint32Array(dagLinksLen);
    // voxel-foliage: BRICK_WORDS per brick; 1-word placeholder when no bricks reserved
    // (the node must always exist so the voxel-raster permutation can reference it).
    const voxelBricksLen = Math.max(1, this.brickCap * BRICK_WORDS);
    this.voxelBricksArr = new Uint32Array(voxelBricksLen);

    this.vertsAttr = new StorageBufferAttribute(this.vertsArr, 1);
    this.vertsAttr.name = 'nanGeoVerts';
    this.hfVertsAttr = new StorageBufferAttribute(this.hfVertsArr, 1);
    this.hfVertsAttr.name = 'nanGeoHfVerts';
    this.idxAttr = new StorageBufferAttribute(this.idxArr, 1);
    this.idxAttr.name = 'nanGeoIndices';
    this.clusterAttr = new StorageBufferAttribute(this.clusterArr, 1);
    this.clusterAttr.name = 'nanGeoClusters';
    this.meshAttr = new StorageBufferAttribute(this.meshArr, 1);
    this.meshAttr.name = 'nanGeoMeshes';
    this.instAttr = new StorageBufferAttribute(this.instArr, 4);
    this.instAttr.name = 'nanGeoInstances';
    this.instMeshAttr = new StorageBufferAttribute(this.instMeshArr, 1);
    this.instMeshAttr.name = 'nanGeoInstMesh';
    this.dagAttr = new StorageBufferAttribute(this.dagArr, 1);
    this.dagAttr.name = 'nanGeoDag';
    this.vcompactAttr = new StorageBufferAttribute(this.vcompactArr, 1);
    this.vcompactAttr.name = 'nanGeoVcompact';
    this.dagLinksAttr = new StorageBufferAttribute(this.dagLinksArr, 1);
    this.dagLinksAttr.name = 'nanGeoDagLinks';
    this.voxelBricksAttr = new StorageBufferAttribute(this.voxelBricksArr, 1);
    this.voxelBricksAttr.name = 'nanGeoVoxelBricks';

    const verts = sU32Views(this.vertsAttr, Math.max(1, this.caps.verts * VERT_WORDS));
    const hfVerts = sU32Views(this.hfVertsAttr, Math.max(1, this.hfCap));
    const idx = sU32Views(this.idxAttr, Math.max(1, this.caps.tris * 3));
    const clusters = sU32Views(this.clusterAttr, Math.max(1, this.caps.clusters * CLUSTER_WORDS));
    const meshes = sU32Views(this.meshAttr, Math.max(1, this.caps.meshes * MESH_WORDS));
    const inst = sVec4Views(this.instAttr, Math.max(2, this.caps.instances * 2));
    const instMesh = sU32Views(this.instMeshAttr, Math.max(1, this.caps.instances));
    const dag = sF32Views(this.dagAttr, dagLen);
    const vcompact = sU32Views(this.vcompactAttr, vcompactLen);
    const dagLinks = sU32Views(this.dagLinksAttr, dagLinksLen);
    const voxelBricks = sU32Views(this.voxelBricksAttr, voxelBricksLen);
    this.instRW = inst.rw;
    this.instMeshRW = instMesh.rw;
    this.gpu = {
      verts: verts.ro,
      hfVerts: hfVerts.ro,
      indices: idx.ro,
      clusters: clusters.ro,
      meshes: meshes.ro,
      instances: inst.ro,
      instanceMesh: instMesh.ro,
      dag: dag.ro,
      vcompact: vcompact.ro,
      dagLinks: dagLinks.ro,
      voxelBricks: voxelBricks.ro,
    };

    // N8-D2 Stage 2a: claim the tile pool as ONE fixed region just past the
    // pre-build entries (its bytes were reserved via addLate in reserveTilePool).
    // Cursors advance past it ONCE; per-slot writes address poolBase+slot*cap and
    // never bump — so later post-build attaches (explicit DAGs) land after the
    // pool, and slots are reused in place across the session.
    if (this.tilePool) {
      // 2e: the pool's verts live in the stride-1 hf buffer (tilePoolBase.vert indexes
      // hfVertsArr); tris/clusters stay in the shared buffers.
      this.tilePoolBase = {
        vert: this.hfVertCursor,
        tri: this.triCursor,
        cluster: this.clusterCursor,
        dagLinks: this.dagLinksCursor, // PERF-VB3: per-slot hierarchical-cull link region
      };
      this.hfVertCursor += this.tilePool.slots * this.tilePool.vertCap;
      this.triCursor += this.tilePool.slots * this.tilePool.triCap;
      this.clusterCursor += this.tilePool.slots * this.tilePool.clusterCap;
      this.dagLinksCursor += this.tilePool.slots * this.tilePool.dagLinksCap;
      this.tileFreeSlots = [];
      for (let s = this.tilePool.slots - 1; s >= 0; s--) this.tileFreeSlots.push(s);
      this.tileSlotOccupant = new Int32Array(this.tilePool.slots).fill(-1);
    }

    // S8: claim the FARTILE pool — cluster/dagLinks as one fixed region past the
    // pre-build entries (like the tile pool), bricks as the TAIL of the brick buffer
    // (library crowns append upward from brickCursor; appendBricks guards the floor).
    // Per-slot bases are assigned here once; attachFartileSlot rewrites in place.
    if (this.ftPool) {
      const p = this.ftPool;
      const totalCap = p.slots * p.clusterCap;
      this.ftPoolBase = {
        cluster: this.clusterCursor,
        dagLinks: this.dagLinksCursor, // linkTotal == clusterCount exactly (roots + one parent link per non-root)
        brick: this.brickCap - p.granules * 128,
      };
      this.clusterCursor += totalCap;
      this.dagLinksCursor += totalCap;
      if (this.ftPoolBase.brick < 0) throw new Error('GeometryRegistry: fartile granule pool exceeds the brick buffer');
      for (let s = 0; s < p.slots; s++) {
        const e = this.entries[this.ftPoolHandles[s] as number] as MeshEntry;
        e.clusterBase = this.ftSlotClusterBase(s);
        e.rootBase = this.ftPoolBase.dagLinks + (e.clusterBase - this.ftPoolBase.cluster);
      }
    }

    for (const e of this.entries) this.copyEntry(e);
    this.populateVCompact();
    for (const s of this.cpuStreams) this.copyCpuStream(s);
    this.parkInstancePoolRegion(); // S7: park the reserved streamed-instance pool off-world
    this.built = true;
    this.runGpuCopies(renderer);

    const report = this.report(performance.now() - t0);
    if (counters) this.updateCounters(counters);
    return report;
  }

  /**
   * Upload everything registered since build() (late hero meshes): partial
   * attribute updates via addUpdateRange + pending GPU instance copies.
   */
  flush(renderer?: Renderer, counters?: Record<string, number>): void {
    if (!this.built) throw new Error('GeometryRegistry: flush() before build()');
    for (const e of this.entries) {
      if (e.uploaded) continue;
      this.copyEntry(e);
      this.pushRange(this.vertsAttr, e.vertBase * VERT_WORDS, e.vertCount * VERT_WORDS);
      this.pushRange(this.idxAttr, e.triBase * 3, e.triCount * 3);
      this.pushRange(this.clusterAttr, e.clusterBase * CLUSTER_WORDS, e.clusterCount * CLUSTER_WORDS);
      this.pushRange(this.meshAttr, e.handle * MESH_WORDS, MESH_WORDS);
    }
    for (const s of this.cpuStreams) {
      if (s.copied) continue;
      this.copyCpuStream(s);
      this.pushRange(this.instAttr, s.first * 8, (s.a.length / 4) * 8);
      this.pushRange(this.instMeshAttr, s.first, s.a.length / 4);
    }
    this.runGpuCopies(renderer);
    if (counters) this.updateCounters(counters);
  }

  /**
   * D (memory arc, 2026-07-04): after boot + the first rendered frame has created and
   * uploaded the mega-buffers, drop the CPU backing arrays of the two IMMUTABLE ones —
   * the explicit-vert buffer (VERT_WORDS; hero/trunk/leaf geometry, static after boot) and
   * the voxel-brick buffer (appended once at boot, never mutated). three r184 re-reads
   * `attr.array` ONLY on a version bump (needsUpdate); neither ever bumps again, so the
   * created GPUBuffer stands and the hundreds of MB of mirrors free on the next GC.
   *
   * NOT released (they mutate post-boot): idx / clusters / dag / dagLinks / hfVerts (the
   * terrain tile-pool region streams into them) and inst / instMesh (instance streams).
   * PRECONDITION: ≥1 frame must have rendered so the buffers exist GPU-side — else three's
   * lazy createStorageAttribute would later read the nulled array. Idempotent. */
  releaseImmutableMirrors(): { vertsBytes: number; brickBytes: number } {
    if (!this.built || this.mirrorsReleased) return { vertsBytes: 0, brickBytes: 0 };
    this.mirrorsReleased = true;
    const vertsBytes = this.vertsArr ? this.vertsArr.byteLength : 0;
    const brickBytes = this.voxelBricksArr ? this.voxelBricksArr.byteLength : 0;
    // drop BOTH references (this.*Arr and attr.array) to the same buffer so GC can reclaim it.
    (this.vertsAttr as unknown as { array: Uint32Array | null }).array = null;
    (this.voxelBricksAttr as unknown as { array: Uint32Array | null }).array = null;
    this.vertsArr = null as unknown as Uint32Array;
    this.voxelBricksArr = null as unknown as Uint32Array;
    return { vertsBytes, brickBytes };
  }

  /**
   * N8-D1 — attach a built LOD DAG (BuildDag.ts) to a registered mesh. Appends
   * the DAG's full self-contained geometry (ALL levels, including a LOD0 copy —
   * D1 trades that duplication for zero index-rebase complexity) as a fresh
   * block in the mega-buffers, writes the parallel 10-float cut records, then
   * REPOINTS the mesh at the DAG cluster range (clusterStart/Count) and clears
   * its discrete LOD chain (lodNext = NONE, MESH_FLAG_HASDAG set). From here the
   * cull runs the per-cluster screen-error cut instead of the ring chain; the
   * original LOD0 clusters and any registerLod() entries go dead (unreferenced).
   *
   * Must run AFTER build() and within the `late` budget (the appended verts/
   * tris/clusters are counted — raise late.* if it throws). Uploads via partial
   * addUpdateRange like flush(); the next frame's kernels see the new data.
   */
  attachDag(handle: MeshHandle, dag: DagBuild): void {
    if (!this.built) throw new Error('GeometryRegistry: attachDag before build()');
    const entry = this.entries[handle];
    if (!entry) throw new Error(`GeometryRegistry: attachDag unknown handle ${handle}`);
    if ((entry.flags & MESH_FLAG_HASDAG) !== 0) {
      throw new Error(`GeometryRegistry: mesh ${entry.label} already has a DAG`);
    }
    if (dag.vertStride !== DAG_VERT_STRIDE) {
      throw new Error(`GeometryRegistry: DAG vertStride ${dag.vertStride} != ${DAG_VERT_STRIDE}`);
    }
    const vCount = dag.verts.length / DAG_VERT_STRIDE;
    const tCount = dag.indices.length / 3;
    const cCount = dag.clusters.length;
    if (!Number.isInteger(vCount) || !Number.isInteger(tCount)) {
      throw new Error('GeometryRegistry: DAG verts/indices not stride-aligned');
    }
    if (cCount === 0) throw new Error('GeometryRegistry: DAG has no clusters');
    this.checkRoom(vCount, tCount, cCount, 0);

    const vBase = this.vertCursor;
    const tBase = this.triCursor;
    const cBase = this.clusterCursor;

    // -- re-pack verts (DAG float layout → registry VERT_WORDS) ----------------
    const vArr = this.vertsArr;
    const sv = dag.verts;
    for (let v = 0; v < vCount; v++) {
      const s = v * DAG_VERT_STRIDE;
      const b = (vBase + v) * VERT_WORDS;
      vArr[b] = f32Bits(sv[s] as number);
      vArr[b + 1] = f32Bits(sv[s + 1] as number);
      vArr[b + 2] = f32Bits(sv[s + 2] as number);
      vArr[b + 3] = octEncode(sv[s + 3] as number, sv[s + 4] as number, sv[s + 5] as number);
      vArr[b + 4] = (f32ToF16(sv[s + 6] as number) | (f32ToF16(sv[s + 7] as number) << 16)) >>> 0;
      const q = (i: number): number => Math.round(Math.max(0, Math.min(1, sv[s + 8 + i] as number)) * 255);
      vArr[b + 5] = (q(0) | (q(1) << 8) | (q(2) << 16) | (q(3) << 24)) >>> 0;
    }

    // -- append indices (rebased onto the appended vertex block) ---------------
    const iArr = this.idxArr;
    const si = dag.indices;
    const ibase = tBase * 3;
    for (let i = 0; i < si.length; i++) iArr[ibase + i] = (si[i] as number) + vBase;

    // -- cluster records (8-word) + DAG cut records (10-float) -----------------
    const cArr = this.clusterArr;
    const dArr = this.dagArr;
    const dagSpheres = new Float32Array(cCount * 4);
    // N8-HIC: derive the hierarchical-cull links + append GLOBAL cluster ids to
    // dagLinks as [roots…][children…]. childBase per cluster (DAG words 10/11) +
    // rootBase/Count on the mesh seed the BFS traversal that replaces brute-force.
    const hier = buildDagHierarchy(dag);
    // item 6: fold this DAG's deepest anchor chain into the global max (sets the
    // minimum BFS pass count the cull needs to emit every leaf — under = holes).
    this.growDagDepth(maxChainDepth(hier));
    const linkBase = this.dagLinksCursor;
    const rootCount = hier.rootIndices.length;
    const childTotal = hier.childIndices.length;
    if (linkBase + rootCount + childTotal > this.dagLinksArr.length) {
      throw new Error(
        `GeometryRegistry: dagLinks overflow (${linkBase + rootCount + childTotal} > ${this.dagLinksArr.length})`,
      );
    }
    const dl = this.dagLinksArr;
    for (let i = 0; i < rootCount; i++) dl[linkBase + i] = cBase + (hier.rootIndices[i] as number);
    const childBlockBase = linkBase + rootCount;
    for (let j = 0; j < childTotal; j++) dl[childBlockBase + j] = cBase + (hier.childIndices[j] as number);
    this.dagLinksCursor = childBlockBase + childTotal;
    for (let c = 0; c < cCount; c++) {
      const dc = dag.clusters[c] as DagCluster;
      const cb = (cBase + c) * CLUSTER_WORDS;
      cArr[cb] = f32Bits(dc.sx);
      cArr[cb + 1] = f32Bits(dc.sy);
      cArr[cb + 2] = f32Bits(dc.sz);
      cArr[cb + 3] = f32Bits(dc.sr);
      cArr[cb + 4] = octEncode(dc.cax, dc.cay, dc.caz);
      cArr[cb + 5] = f32Bits(dc.ccos);
      cArr[cb + 6] = tBase + dc.triStart;
      if (dc.triCount > MAX_CLUSTER_TRIS) throw new Error('GeometryRegistry: DAG cluster exceeds tri cap');
      // word7: triCount(0-7) | flags(8-9) | LOD level(10-15, for ?nanitedbg=lod) | handle(16-31)
      cArr[cb + 7] =
        ((dc.triCount & 0xff) | (CLUSTER_FLAG_DAG << 8) | ((dc.level & 0x3f) << 10) | (entry.handle << 16)) >>> 0;

      const db = (cBase + c) * DAG_WORDS;
      const root = !Number.isFinite(dc.parentError);
      dArr[db] = dc.ownError;
      dArr[db + 1] = dc.oex;
      dArr[db + 2] = dc.oey;
      dArr[db + 3] = dc.oez;
      dArr[db + 4] = dc.oer;
      // root: parentError +∞ → finite sentinel; parentSphere ← ownSphere so the
      // sqrt(d²−r²) denominator stays well-formed (the err alone forces project > τ)
      dArr[db + 5] = root ? DAG_ROOT_PARENT_ERR : dc.parentError;
      dArr[db + 6] = root ? dc.oex : dc.pex;
      dArr[db + 7] = root ? dc.oey : dc.pey;
      dArr[db + 8] = root ? dc.oez : dc.pez;
      dArr[db + 9] = root ? dc.oer : dc.per;

      dagSpheres[c * 4] = dc.sx;
      dagSpheres[c * 4 + 1] = dc.sy;
      dagSpheres[c * 4 + 2] = dc.sz;
      dagSpheres[c * 4 + 3] = dc.sr;
    }

    // N8-HIC: child links into the DAG record (words 10/11, bitcast u32) — global
    // dagLinks offset of each cluster's children + count (non-zero only on a group
    // OWNER; 0 on non-owners + LOD0). Terrain/non-DAG meshes leave these 0 and use
    // the brute-force path (rootCount = 0 ⇒ not hierarchical).
    for (let c = 0; c < cCount; c++) {
      const db = (cBase + c) * DAG_WORDS;
      dArr[db + 10] = bitsF32(childBlockBase + (hier.childStart[c] as number));
      dArr[db + 11] = bitsF32(hier.childCount[c] as number);
    }

    // PERF-3 (stage 1): vcompact records for the appended DAG clusters. populateVCompact
    // ran ONCE over boot clusters only, so without this every post-build DAG cluster (all
    // veg trunk DAGs + the entire crown-LOD ladder) takes the per-corner projection
    // fallback — the majority of worst-pose visible clusters.
    // ⚠️ ANCHOR INVARIANT: the projection's unique-vert writer enumerates verts from
    // vcMin, but ALL readers (canonVertSlot) compute slot = vi − indices[triStart*3] —
    // the FIRST corner, not the min. Boot clusters get first-corner==min from
    // meshletizeDag; DAG clusters carry NO such guarantee. So anchor the record at the
    // reader's vBase: store (vBase, mx−vBase+1), and stay 0 (per-corner fallback) if any
    // vi < vBase (the reader's subtraction would wrap) or the anchored window > cache.
    let vcCached = 0;
    for (let c = 0; c < cCount; c++) {
      const dc = dag.clusters[c] as DagCluster;
      if (dc.triCount === 0) continue;
      const vBase0 = iArr[(tBase + dc.triStart) * 3] ?? 0;
      const { mn, count } = this.vcompactSpan(iArr, tBase + dc.triStart, dc.triCount);
      const width = mn + count - vBase0; // mx − vBase + 1
      if (mn >= vBase0 && count > 0 && width <= VCACHE_VERTS) {
        this.vcompactArr[(cBase + c) * 2] = vBase0;
        this.vcompactArr[(cBase + c) * 2 + 1] = width;
        vcCached++;
      }
    }
    console.log(`[laas][vcompact] attachDag ${entry.label}: cached ${vcCached}/${cCount} DAG clusters`);

    this.vertCursor += vCount;
    this.triCursor += tCount;
    this.clusterCursor += cCount;

    // -- repoint the mesh at its DAG range; retire the discrete LOD chain ------
    entry.clusterBase = cBase;
    entry.rootBase = linkBase;
    entry.rootCount = rootCount;
    entry.clusterCount = cCount;
    // The DAG is the COMPLETE continuous LOD, so it inherits the mesh's full
    // intended DRAW envelope: the max distance setMaxDistance configured on the
    // chain TAIL (trees TREE_GEO_FAR 496 m, rocks/deadwood clsMaxDist) — NOT the
    // head's short chain-SWITCH distance. Walk the about-to-be-retired chain to
    // recover it, THEN collapse to a single node. Leaving the head-switch distance
    // is the bug that made the envelope rule `lodNext==NONE && lodDist>0 &&
    // dist>lodDist` drop the WHOLE instance at ~26 m (trees) / ~120 m (rocks),
    // while shrubs (switch 0) never dropped. A truly UNLIMITED (0) envelope is the
    // N8 end state that retires the impostor far-field, but it needs the
    // min-screen-size cull first — the cut pins the root (parent sentinel never
    // cut), so without it an open vista floods (~3.7M clusters, ~90 ms even with
    // occlusion). (N8-D1 envelope fix; unbounded deferred to N8-D1e + min-screen.)
    let tail = entry;
    for (let g = 0; g < 8 && tail.lodNext !== LOD_NONE; g++) {
      const nxt = this.entries[tail.lodNext];
      if (!nxt) break;
      tail = nxt;
    }
    entry.lodDist = tail.lodDist;
    entry.lodNext = LOD_NONE;
    entry.flags |= MESH_FLAG_HASDAG;
    entry.sphere = meshSphereFromClusters(dagSpheres, cCount);
    this.writeMeshRecord(entry);

    // -- upload appended ranges (partial writeBuffer; no-op marker in node) ----
    this.pushRange(this.vertsAttr, vBase * VERT_WORDS, vCount * VERT_WORDS);
    this.pushRange(this.idxAttr, tBase * 3, tCount * 3);
    this.pushRange(this.clusterAttr, cBase * CLUSTER_WORDS, cCount * CLUSTER_WORDS);
    this.pushRange(this.vcompactAttr, cBase * 2, cCount * 2);
    this.pushRange(this.dagAttr, cBase * DAG_WORDS, cCount * DAG_WORDS);
    this.pushRange(this.dagLinksAttr, linkBase, rootCount + childTotal);
    this.pushRange(this.meshAttr, entry.handle * MESH_WORDS, MESH_WORDS);
  }

  /**
   * N8-D2b: reserve a heightfield mesh whose geometry is an adaptive terrain
   * LOD DAG (filled by attachHeightDag post-build) rather than the discrete
   * window grid. Allocates NO clusters up front — the DAG range is appended
   * late (reserve it with addLate) — but carries the hf origin/cell the GPU
   * decode needs (mesh-record words 7/8/9) and the HEIGHTFIELD flag.
   */
  registerHeightDag(
    matClass: MaterialClassId,
    hf: { originX: number; originZ: number; cellSize: number },
    opts: RegisterOpts = {},
  ): MeshHandle {
    const handle = this.entries.length;
    if (handle >= 0xffff) throw new Error('GeometryRegistry: mesh id exceeds u16');
    if (this.built && handle >= this.caps.meshes) {
      throw new Error('GeometryRegistry: mesh capacity exceeded post-build — raise late.meshes');
    }
    const entry = this.newEntry(handle, matClass, { ...opts, transformChannel: 'terrain' }, true, 0);
    entry.hf = {
      originX: hf.originX,
      originZ: hf.originZ,
      cellSize: hf.cellSize,
      quadsX: 0,
      quadsZ: 0,
      windowsX: 0,
      windowsZ: 0,
    };
    this.entries.push(entry);
    return handle;
  }

  /**
   * N8-D2b: attach a terrain LOD DAG (buildHeightGrid output) to a
   * registerHeightDag mesh. Mirrors attachDag — same cluster records + 10-float
   * DAG cut records + the SAME flat kClusterCull cut — but the vertex pool packs
   * TEXEL grid coords (word0 = gx | gz<<16, already clamped to [0,res-1]; words
   * 1-5 unused — height + normal come from the TerrainField planes on the GPU)
   * and the cluster carries CLUSTER_FLAG_HEIGHTFIELD|CLUSTER_FLAG_DAG so the
   * decode takes the indexed-heightfield path. (Wasting 5/6 vertex words is a
   * known memory cost; a stride-1 terrain vertex buffer is a later optimisation.)
   */
  attachHeightDag(
    handle: MeshHandle,
    build: { gridVerts: Uint32Array; indices: Uint32Array; clusters: DagCluster[] },
  ): void {
    if (!this.built) throw new Error('GeometryRegistry: attachHeightDag before build()');
    const entry = this.entries[handle];
    if (!entry) throw new Error(`GeometryRegistry: attachHeightDag unknown handle ${handle}`);
    if (!entry.hf) throw new Error(`GeometryRegistry: attachHeightDag mesh ${entry.label} is not a heightfield`);
    if ((entry.flags & MESH_FLAG_HASDAG) !== 0) {
      throw new Error(`GeometryRegistry: mesh ${entry.label} already has a DAG`);
    }
    const { gridVerts, indices, clusters } = build;
    const vCount = gridVerts.length;
    const tCount = indices.length / 3;
    const cCount = clusters.length;
    if (!Number.isInteger(tCount)) throw new Error('GeometryRegistry: height-DAG indices not tri-aligned');
    if (cCount === 0) throw new Error('GeometryRegistry: height-DAG has no clusters');
    this.checkRoom(0, tCount, cCount, 0);
    if (this.hfVertCursor + vCount > this.hfCap) {
      throw new Error(
        `GeometryRegistry: hf-vert capacity exceeded (${this.hfVertCursor}+${vCount}/${this.hfCap}) — raise late.hfVerts`,
      );
    }

    const vBase = this.hfVertCursor;
    const tBase = this.triCursor;
    const cBase = this.clusterCursor;

    // -- verts: stride-1 (2e) — one packed texel coord per vert, dedicated hf buffer
    const vArr = this.hfVertsArr;
    for (let v = 0; v < vCount; v++) vArr[vBase + v] = (gridVerts[v] as number) >>> 0;

    // -- indices (rebased onto the appended vertex block) ----------------------
    const iArr = this.idxArr;
    const ibase = tBase * 3;
    for (let i = 0; i < indices.length; i++) iArr[ibase + i] = (indices[i] as number) + vBase;

    // -- cluster records (8-word) + DAG cut records (10-float) -----------------
    const cArr = this.clusterArr;
    const dArr = this.dagArr;
    const hfFlags = (CLUSTER_FLAG_HEIGHTFIELD | CLUSTER_FLAG_DAG) & 0xff;
    const dagSpheres = new Float32Array(cCount * 4);
    for (let c = 0; c < cCount; c++) {
      const dc = clusters[c] as DagCluster;
      const cb = (cBase + c) * CLUSTER_WORDS;
      cArr[cb] = f32Bits(dc.sx);
      cArr[cb + 1] = f32Bits(dc.sy);
      cArr[cb + 2] = f32Bits(dc.sz);
      cArr[cb + 3] = f32Bits(dc.sr);
      cArr[cb + 4] = octEncode(dc.cax, dc.cay, dc.caz);
      cArr[cb + 5] = f32Bits(dc.ccos);
      cArr[cb + 6] = tBase + dc.triStart;
      if (dc.triCount > MAX_CLUSTER_TRIS) throw new Error('GeometryRegistry: height-DAG cluster exceeds tri cap');
      // word7: triCount(0-7) | flags(8-9) | LOD level(10-15, for ?nanitedbg=lod) | handle(16-31)
      cArr[cb + 7] =
        ((dc.triCount & 0xff) | (hfFlags << 8) | ((dc.level & 0x3f) << 10) | (entry.handle << 16)) >>> 0;

      const db = (cBase + c) * DAG_WORDS;
      const root = !Number.isFinite(dc.parentError);
      dArr[db] = dc.ownError;
      dArr[db + 1] = dc.oex;
      dArr[db + 2] = dc.oey;
      dArr[db + 3] = dc.oez;
      dArr[db + 4] = dc.oer;
      dArr[db + 5] = root ? DAG_ROOT_PARENT_ERR : dc.parentError;
      dArr[db + 6] = root ? dc.oex : dc.pex;
      dArr[db + 7] = root ? dc.oey : dc.pey;
      dArr[db + 8] = root ? dc.oez : dc.pez;
      dArr[db + 9] = root ? dc.oer : dc.per;

      dagSpheres[c * 4] = dc.sx;
      dagSpheres[c * 4 + 1] = dc.sy;
      dagSpheres[c * 4 + 2] = dc.sz;
      dagSpheres[c * 4 + 3] = dc.sr;
    }

    this.hfVertCursor += vCount;
    this.triCursor += tCount;
    this.clusterCursor += cCount;

    // terrain DAG = the COMPLETE continuous LOD over one identity instance, so
    // it inherits an UNLIMITED draw envelope (the whole field is eligible; the
    // cut + frustum bound it). Unlike scattered DAG instances (D-N35) there is
    // no multiplicity to flood — a single pinned root costs one cluster.
    entry.clusterBase = cBase;
    entry.clusterCount = cCount;
    entry.lodNext = LOD_NONE;
    entry.lodDist = 0;
    entry.flags |= MESH_FLAG_HASDAG;
    entry.sphere = meshSphereFromClusters(dagSpheres, cCount);
    this.writeMeshRecord(entry);

    this.pushRange(this.hfVertsAttr, vBase, vCount);
    this.pushRange(this.idxAttr, tBase * 3, tCount * 3);
    this.pushRange(this.clusterAttr, cBase * CLUSTER_WORDS, cCount * CLUSTER_WORDS);
    this.pushRange(this.dagAttr, cBase * DAG_WORDS, cCount * DAG_WORDS);
    this.pushRange(this.meshAttr, entry.handle * MESH_WORDS, MESH_WORDS);
  }

  // -- N8-D2 Stage 2a: streaming terrain tile pool (D-N39) ------------------------

  /**
   * Reserve a fixed pool of `slots` streaming terrain-tile slots — the memory
   * bound for full-res terrain. PRE-BUILD only. Creates one heightfield mesh
   * handle per slot (sharing the GLOBAL hf origin/cell — tiles store global texel
   * coords) + binds an identity instance to each, and reserves slots×cap of
   * vert/tri/cluster space (claimed as one fixed region at build()). A slot is
   * (re)loaded with attachHeightDagTile and released with evictHeightDagTile —
   * O(1), no fragmentation. Returns the slot→handle table (index i = slot i).
   */
  reserveTilePool(
    matClass: MaterialClassId,
    hf: { originX: number; originZ: number; cellSize: number },
    cap: { slots: number; vertCap: number; triCap: number; clusterCap: number },
    opts: RegisterOpts = {},
  ): MeshHandle[] {
    if (this.built) throw new Error('GeometryRegistry: reserveTilePool after build()');
    if (this.tilePool) throw new Error('GeometryRegistry: tile pool already reserved');
    const { slots, vertCap, triCap, clusterCap } = cap;
    if (slots <= 0 || vertCap <= 0 || triCap <= 0 || clusterCap <= 0) {
      throw new Error('GeometryRegistry: reserveTilePool caps must be positive');
    }
    // PERF-VB3: per-slot hierarchical-cull links = [roots][children] = totalClusters
    // (roots + every non-root once) ≤ clusterCap. The dagLinks BUFFER is auto-sized
    // 2× caps.clusters, which the pool's clusters (addLate below) already grow.
    this.tilePool = { slots, vertCap, triCap, clusterCap, dagLinksCap: clusterCap };
    // 2e: pool verts are reserved in the stride-1 hf buffer (hfVerts), not the 6-word `verts`.
    this.addLate({ hfVerts: slots * vertCap, tris: slots * triCap, clusters: slots * clusterCap });
    const handles: MeshHandle[] = [];
    for (let s = 0; s < slots; s++) {
      const h = this.registerHeightDag(matClass, hf, {
        ...opts,
        label: opts.label ? `${opts.label}/slot${s}` : `tilepool/slot${s}`,
      });
      this.bindInstances(h, { a: new Float32Array([0, 0, 0, 1]), b: new Float32Array([0, 0, 0, 0]) });
      handles.push(h);
    }
    this.tilePoolHandles = handles;
    return handles;
  }

  get tilePoolSlotCount(): number {
    return this.tilePool?.slots ?? 0;
  }
  get tileFreeSlotCount(): number {
    return this.tileFreeSlots.length;
  }
  /** the pool's per-slot capacity — the streamer pre-checks a built tile against
   *  this to SKIP an oversized region (coarser ring backstops) rather than have
   *  attachHeightDagTile throw + leak the alloc'd slot. */
  get tilePoolCap(): { vertCap: number; triCap: number; clusterCap: number } {
    const pool = this.tilePool;
    if (!pool) throw new Error('GeometryRegistry: no tile pool reserved');
    return { vertCap: pool.vertCap, triCap: pool.triCap, clusterCap: pool.clusterCap };
  }
  tileSlotHandle(slot: number): MeshHandle {
    const h = this.tilePoolHandles[slot];
    if (h == null) throw new Error(`GeometryRegistry: tile slot ${slot} has no handle`);
    return h;
  }

  // ---- S7 instance pool (§5, A4) --------------------------------------------------------

  /**
   * Reserve a FLAT instance-pool region of `blockSize·blocks` slots BEFORE build()
   * (streamed world only). Advances instCursor so registry.instanceCount — which
   * sizes the frozen cull dispatch (NaniteCull.ts:250, kSeedRoots :983) — covers the
   * whole pool; build() parks every slot off-world (≈free via frustum cull). Blocks
   * are the free-list granularity (8k/chunk-block by default); a streamed chunk claims
   * ≥1 block. NEVER call on the generated world (keeps its instance words identical).
   */
  reserveInstancePool(blockSize: number, blocks: number): void {
    if (this.built) throw new Error('GeometryRegistry: reserveInstancePool after build()');
    if (this.instPool) throw new Error('GeometryRegistry: instance pool already reserved');
    if (blockSize <= 0 || blocks <= 0) throw new Error('GeometryRegistry: reserveInstancePool needs positive blockSize/blocks');
    const capacity = blockSize * blocks;
    const first = this.instCursor;
    this.instCursor += capacity;
    this.instPool = { first, blockSize, blocks, capacity };
    this.instBlockFree = [];
    for (let b = blocks - 1; b >= 0; b--) this.instBlockFree.push(b);
  }

  /** park every pool slot off-world at build (called from build() once instArr
   *  exists). Zero-init already gives scale 0 / meshId 0; only y needs the sentinel. */
  private parkInstancePoolRegion(): void {
    const pool = this.instPool;
    if (!pool) return;
    for (let i = 0; i < pool.capacity; i++) this.instArr[(pool.first + i) * 8 + 1] = GeometryRegistry.INST_PARK_Y;
  }

  get instancePoolBlockCount(): number {
    return this.instPool?.blocks ?? 0;
  }
  get instancePoolFreeBlocks(): number {
    return this.instBlockFree.length;
  }
  get instancePoolCapacity(): number {
    return this.instPool?.capacity ?? 0;
  }
  get instancePoolBlockSize(): number {
    return this.instPool?.blockSize ?? 0;
  }

  /** pop a free instance block, or -1 if the pool is full (caller evicts first). */
  allocInstanceBlock(): number {
    if (!this.instPool) throw new Error('GeometryRegistry: no instance pool reserved');
    return this.instBlockFree.pop() ?? -1;
  }

  /** park a block's slots + return it to the free-list. */
  freeInstanceBlock(block: number): void {
    const pool = this.instPool;
    if (!pool) throw new Error('GeometryRegistry: no instance pool reserved');
    if (block < 0 || block >= pool.blocks) throw new Error(`GeometryRegistry: freeInstanceBlock ${block} out of range`);
    this.rewriteInstanceBlock(block, 0, new Float32Array(0), new Float32Array(0), new Uint32Array(0));
    this.instBlockFree.push(block);
  }

  /**
   * F-4: overwrite a pooled instance block with `count` live instances and PARK the
   * rest. `a` = A-words (x,y,z,scale) ABSOLUTE game-space; stored StreamOrigin-relative
   * (poolOrigin subtract — 0 on generated ⇒ no-op). `b` = B-words (yaw,leanX,leanZ,idF).
   * `meshIds` = per-slot chain-head handle (idF→head resolved main-side). Writes through
   * the live instArr/instMeshArr mirrors (NOT released post-boot) + one pushRange each
   * (the S7 §9b F-4 method — ~per-block, block ≤ blockSize).
   */
  rewriteInstanceBlock(block: number, count: number, a: Float32Array, b: Float32Array, meshIds: Uint32Array): void {
    const pool = this.instPool;
    if (!pool) throw new Error('GeometryRegistry: no instance pool reserved');
    if (block < 0 || block >= pool.blocks) throw new Error(`GeometryRegistry: rewriteInstanceBlock ${block} out of range`);
    if (count > pool.blockSize) throw new Error(`GeometryRegistry: block ${block} count ${count} > blockSize ${pool.blockSize}`);
    const base = pool.first + block * pool.blockSize;
    const oX = this.poolOriginX;
    const oZ = this.poolOriginZ;
    for (let i = 0; i < count; i++) {
      const d = (base + i) * 8;
      this.instArr[d] = (a[i * 4] as number) - oX;
      this.instArr[d + 1] = a[i * 4 + 1] as number;
      this.instArr[d + 2] = (a[i * 4 + 2] as number) - oZ;
      this.instArr[d + 3] = a[i * 4 + 3] as number;
      this.instArr[d + 4] = b[i * 4] as number;
      this.instArr[d + 5] = b[i * 4 + 1] as number;
      this.instArr[d + 6] = b[i * 4 + 2] as number;
      this.instArr[d + 7] = b[i * 4 + 3] as number;
      this.instMeshArr[base + i] = meshIds[i] as number;
    }
    for (let i = count; i < pool.blockSize; i++) {
      const d = (base + i) * 8;
      this.instArr[d] = 0;
      this.instArr[d + 1] = GeometryRegistry.INST_PARK_Y;
      this.instArr[d + 2] = 0;
      this.instArr[d + 3] = 0;
      this.instMeshArr[base + i] = 0;
    }
    this.pushRange(this.instAttr, base * 8, pool.blockSize * 8);
    this.pushRange(this.instMeshAttr, base, pool.blockSize);
  }

  /** Refresh only pooled-instance root Y values after the packed terrain morph
   *  centre changes. X/Z/scale, orientation, idF, mesh ownership, and parked slots
   *  remain untouched. `a` is the block's absolute A-word mirror. */
  rewriteInstanceBlockGround(block: number, first: number, count: number, a: Float32Array): void {
    const pool = this.instPool;
    if (!pool) throw new Error('GeometryRegistry: no instance pool reserved');
    if (block < 0 || block >= pool.blocks) throw new Error(`GeometryRegistry: rewriteInstanceBlockGround ${block} out of range`);
    if (first < 0 || count < 0 || first + count > pool.blockSize) {
      throw new Error(`GeometryRegistry: invalid ground range ${first}+${count}/${pool.blockSize}`);
    }
    const base = pool.first + block * pool.blockSize;
    for (let i = first; i < first + count; i++) this.instArr[(base + i) * 8 + 1] = a[i * 4 + 1] as number;
    this.pushRange(this.instAttr, (base + first) * 8, count * 8);
  }
  /** slot's fixed geometry base offsets (constant for the pool's lifetime;
   *  cluster base also = the mesh record's clusterStart after a load). Probe/debug. */
  tileSlotBase(slot: number): { vert: number; tri: number; cluster: number } {
    const pool = this.tilePool;
    if (!pool) throw new Error('GeometryRegistry: no tile pool reserved');
    if (slot < 0 || slot >= pool.slots) throw new Error(`GeometryRegistry: tile slot ${slot} out of range`);
    return {
      vert: this.tilePoolBase.vert + slot * pool.vertCap,
      tri: this.tilePoolBase.tri + slot * pool.triCap,
      cluster: this.tilePoolBase.cluster + slot * pool.clusterCap,
    };
  }

  /** pop a free tile slot, or -1 if the pool is full (caller evicts first). */
  allocTileSlot(): number {
    if (!this.tilePool) throw new Error('GeometryRegistry: no tile pool reserved');
    const slot = this.tileFreeSlots.pop();
    return slot ?? -1;
  }

  /** S6d: the live StreamOrigin the tile pool is stored relative to (accumulated
   *  rebase deltas). 0 on the generated world (never rebases) ⇒ attach is a no-op
   *  subtraction ⇒ byte-identical. */
  private poolOriginX = 0;
  private poolOriginZ = 0;

  /** S6d: monotone counter bumped on every tile attach/evict — the shadow-clip fit
   *  reads it to dirty (re-raster) its levels when terrain streams in/out while the
   *  camera is static (a texel-snap wouldn't fire, so newly-streamed casters would
   *  cast no shadow / evicted ones leave a stale shadow — the S6c latent finding).
   *  Never bumps on the generated world (no streaming) ⇒ shadow path byte-identical. */
  private _tileEpoch = 0;
  get tileEpoch(): number {
    return this._tileEpoch;
  }

  /**
   * StreamOrigin rebase (S5 / F-3): the tile-pool mesh records' hf origin words
   * are StreamOrigin-relative — shift every occupied slot by −Δ and rewrite
   * through the live rewriteMeshRecord path (hundreds × 72 B, trivial). Mesh
   * spheres shift with them (instance cull). Cluster/DAG sphere words stay as
   * attached — the brain re-attaches tiles after a rebase (it is a
   * teleport-scale event), and the kernel-side origin add lands with S6's
   * camera-relative pass; the generated world never rebases (origin (0,0)).
   */
  rebaseTilePoolOrigins(dx: number, dz: number): void {
    if (dx === 0 && dz === 0) return;
    // S6d: the accumulated StreamOrigin — tiles the brain streams in AFTER a rebase
    // arrive in ABSOLUTE coords, so attachHeightDagTile subtracts this to store them
    // in the same StreamOrigin-relative frame as the already-shifted resident tiles.
    this.poolOriginX += dx;
    this.poolOriginZ += dz;
    // f32 view over the cluster blob (sphere xyz-r are f32-bits in the u32 array)
    const clF = new Float32Array(this.clusterArr.buffer, this.clusterArr.byteOffset, this.clusterArr.length);
    for (const h of this.tilePoolHandles) {
      const e = this.entries[h];
      if (!e?.hf) continue;
      e.hf.originX -= dx;
      e.hf.originZ -= dz;
      e.sphere = [e.sphere[0] - dx, e.sphere[1], e.sphere[2] - dz, e.sphere[3]];
      if (this.built && e.uploaded) this.rewriteMeshRecord(e);
      // S6d: the per-cluster + DAG bounding spheres are baked ABSOLUTE world-space
      // (the cull reads c.sphere as world-space for HF — NaniteCull instWorldSphere),
      // so they must shift by −Δ too or the frustum/HZB/LOD tests reject every tile
      // cluster once cam.planes go anchor-relative (= "0 tris"). This is the
      // "kernel-side origin add [that] lands with S6's camera-relative pass". Cluster
      // record: sphere xyz @ words 0/1/2. DAG record (DAG_WORDS): ownSphere.x/z @ 1/3,
      // parentSphere.x/z @ 6/8 (siblings share the pair bit-exactly ⇒ a uniform shift
      // keeps them equal). Only tile-pool (streamed) meshes; the generated world never
      // rebases, so its cull inputs are byte-identical.
      if (this.built && e.uploaded) {
        const cB = e.clusterBase;
        const cN = e.clusterCount;
        for (let i = 0; i < cN; i++) {
          const cw = (cB + i) * CLUSTER_WORDS;
          clF[cw] = (clF[cw] as number) - dx;
          clF[cw + 2] = (clF[cw + 2] as number) - dz;
          const dw = (cB + i) * DAG_WORDS;
          this.dagArr[dw + 1] = (this.dagArr[dw + 1] as number) - dx;
          this.dagArr[dw + 3] = (this.dagArr[dw + 3] as number) - dz;
          this.dagArr[dw + 6] = (this.dagArr[dw + 6] as number) - dx;
          this.dagArr[dw + 8] = (this.dagArr[dw + 8] as number) - dz;
        }
        this.pushRange(this.clusterAttr, cB * CLUSTER_WORDS, cN * CLUSTER_WORDS);
        this.pushRange(this.dagAttr, cB * DAG_WORDS, cN * DAG_WORDS);
      }
    }
  }

  /**
   * StreamOrigin rebase (S6d): shift every CPU-stream instance's A-word
   * translation by −Δ so the pooled instance frame stays StreamOrigin-relative
   * alongside the terrain-tile origin words. The identity heads of the FarTiles
   * far-forest crowns live here (their absolute tile-centre A-word) — without
   * this shift they project off-frustum once the camera VP goes anchor-relative.
   * Instance cull bounds are instTransformPoint(A, localSphere), so shifting A
   * re-anchors both the transform and the cull in one write. Only CPU streams are
   * touched (GPU-scatter streams — generated veg — are written straight to the GPU
   * buffer and never rebase: the generated world holds StreamOrigin at (0,0)).
   *
   * EXCEPTION — heightfield tile instances are IDENTITY ANCHORS, not world-positioned:
   * a terrain tile is placed by its mesh-record origin words (rebaseTilePoolOrigins
   * shifts those) and its cluster/error spheres are stored world-space (the cull's
   * isHF branch reads them directly, skipping instTransformPoint). Their A MUST stay
   * (0,0,0,1). Shifting it makes the pOwn cut's ownC = instTransformPoint(A,ownSphere)
   * land −StreamOrigin away (~366 km on Estonia) ⇒ denO huge ⇒ pOwn≈0 ⇒ the whole tile
   * emits its ROOT and never descends (the 128 m single-cluster terrain bug). Skip them.
   */
  rebaseInstanceOrigins(dx: number, dz: number): void {
    if (dx === 0 && dz === 0 || !this.built) return;
    for (const s of this.cpuStreams) {
      if (this.entries[s.meshId]?.hf) continue; // identity anchor — never origin-shift (see above)
      const count = s.a.length / 4;
      for (let i = 0; i < count; i++) {
        const d = (s.first + i) * 8;
        this.instArr[d] = (this.instArr[d] as number) - dx;
        this.instArr[d + 2] = (this.instArr[d + 2] as number) - dz;
      }
      this.pushRange(this.instAttr, s.first * 8, count * 8);
    }
    // S7: the streamed-instance pool stores StreamOrigin-relative A-words too — shift
    // the WHOLE region (parked slots keep y = INST_PARK_Y ⇒ still frustum-culled, so
    // touching them is harmless and avoids per-block residency tracking here). Matches
    // rebaseTilePoolOrigins' poolOriginX/Z accumulation (called first by StreamOrigin).
    const pool = this.instPool;
    if (pool) {
      for (let i = 0; i < pool.capacity; i++) {
        const d = (pool.first + i) * 8;
        this.instArr[d] = (this.instArr[d] as number) - dx;
        this.instArr[d + 2] = (this.instArr[d + 2] as number) - dz;
      }
      this.pushRange(this.instAttr, pool.first * 8, pool.capacity * 8);
    }
  }

  /**
   * S6e FRAME-MIXING DETECTOR (diagnostic; cheap CPU scan). The cull invariant is
   * ONE coordinate frame for every input: cluster/DAG sphere centres, cam.camPos
   * and cam.planes must all be StreamOrigin-relative (= within a few km of the
   * anchor). This scans every RESIDENT tile-pool cluster and counts those whose
   * centre sits > `farM` metres from the anchor (ax,az) — a sphere still in the
   * absolute ~311 km frame while the camera went anchor-relative. Nonzero while
   * the anchor is non-zero ⇒ a rebase/attach bookkeeping bug (the S6e regression).
   * Returns totals + the worst offender for the console/HUD.
   */
  frameMixStats(
    camX: number,
    camY: number,
    camZ: number,
    ax: number,
    az: number,
    projK: number,
    tau: number,
    vp: number[],
    farM = 64000,
  ): {
    residentTiles: number;
    clustersScanned: number;
    farFromOrigin: number;
    maxCenterKm: number;
    roots: { total: number; emit: number; descend: number };
    nearest: {
      handle: number;
      cx: number;
      cz: number;
      centerKm: number;
      ownError: number;
      denO: number;
      pOwn: number;
      ndc: [number, number, number];
    } | null;
  } {
    const clF = new Float32Array(this.clusterArr.buffer, this.clusterArr.byteOffset, this.clusterArr.length);
    const cpX = camX - ax;
    const cpY = camY;
    const cpZ = camZ - az;
    let residentTiles = 0;
    let clustersScanned = 0;
    let farFromOrigin = 0;
    let maxCenter = 0;
    let rootTotal = 0;
    let rootEmit = 0;
    let rootDescend = 0;
    let nearest: {
      handle: number; cx: number; cz: number; centerKm: number;
      ownError: number; denO: number; pOwn: number; ndc: [number, number, number];
    } | null = null;
    let nearestDist = Infinity;
    // project (x,y,z) by the row-major .toArray() (column-major) mat4 → ndc
    const proj = (x: number, y: number, z: number): [number, number, number] => {
      const w = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
      const cx2 = vp[0] * x + vp[4] * y + vp[8] * z + vp[12];
      const cy2 = vp[1] * x + vp[5] * y + vp[9] * z + vp[13];
      const cz2 = vp[2] * x + vp[6] * y + vp[10] * z + vp[14];
      const iw = w !== 0 ? 1 / w : 0;
      return [cx2 * iw, cy2 * iw, cz2 * iw];
    };
    for (const h of this.tilePoolHandles) {
      const e = this.entries[h];
      if (!e?.hf || !e.uploaded || e.clusterCount === 0) continue;
      residentTiles++;
      for (let i = 0; i < e.clusterCount; i++) {
        const cw = (e.clusterBase + i) * CLUSTER_WORDS;
        const cx = clF[cw] as number;
        const cz = clF[cw + 2] as number;
        const center = Math.hypot(cx, cz); // distance from the RELATIVE origin
        clustersScanned++;
        if (center > farM) farFromOrigin++;
        if (center > maxCenter) maxCenter = center;
        // DAG record: ownError @ w0, ownSphere.xyz @ w1/2/3, ownR @ w4. Root ⇔ parentErr not finite.
        const db = (e.clusterBase + i) * DAG_WORDS;
        const parentErr = this.dagArr[db + 5] as number;
        if (parentErr !== DAG_ROOT_PARENT_ERR) continue; // roots only
        rootTotal++;
        const ownError = this.dagArr[db] as number;
        const oex = this.dagArr[db + 1] as number;
        const oey = this.dagArr[db + 2] as number;
        const oez = this.dagArr[db + 3] as number;
        const ownR = this.dagArr[db + 4] as number;
        const dvx = cpX - oex;
        const dvy = cpY - oey;
        const dvz = cpZ - oez;
        const denO = Math.max(1e-6, Math.sqrt(Math.max(1e-6, dvx * dvx + dvy * dvy + dvz * dvz - ownR * ownR)));
        const pOwn = (projK * 1 * ownError) / denO;
        if (pOwn <= tau) rootEmit++;
        else rootDescend++;
        const cdist = Math.hypot(cpX - oex, cpZ - oez);
        if (cdist < nearestDist) {
          nearestDist = cdist;
          nearest = {
            handle: h, cx, cz, centerKm: center / 1000,
            ownError, denO, pOwn, ndc: proj(oex, oey, oez),
          };
        }
      }
    }
    return {
      residentTiles, clustersScanned, farFromOrigin, maxCenterKm: maxCenter / 1000,
      roots: { total: rootTotal, emit: rootEmit, descend: rootDescend }, nearest,
    };
  }

  /**
   * (Re)load a terrain tile (buildHeightGrid output) into a slot's FIXED byte
   * range — overwrites the previous occupant in place; no cursor growth.
   * Mirrors attachHeightDag's pack but addresses poolBase+slot*cap and is
   * REUSABLE (no already-has-DAG guard). The slot's mesh record is repointed at
   * the new cluster range + sphere; partial uploads make the next frame's
   * kernels see it.
   *
   * S5: `hf` sets PER-TILE mesh-record origin words (gridVerts are then
   * TILE-LOCAL texel coords) — StreamOrigin-relative placement (F-3): rebase
   * rewrites these words via rebaseTilePoolOrigins, and Estonia's large signed
   * lattice fits the 13-bit coord field. Omitted ⇒ the slot keeps the shared
   * origin from reserveTilePool (the legacy uniform-tile path, global coords).
   */
  attachHeightDagTile(
    slot: number,
    build: { gridVerts: Uint32Array; indices: Uint32Array; clusters: DagCluster[] },
    hf?: { originX: number; originZ: number; cellSize: number },
  ): void {
    if (!this.built) throw new Error('GeometryRegistry: attachHeightDagTile before build()');
    const pool = this.tilePool;
    if (!pool) throw new Error('GeometryRegistry: no tile pool reserved');
    if (slot < 0 || slot >= pool.slots) throw new Error(`GeometryRegistry: tile slot ${slot} out of range`);
    const handle = this.tilePoolHandles[slot] as number;
    const entry = this.entries[handle] as MeshEntry;
    // S6d: convert the brain's ABSOLUTE tile coords into the StreamOrigin-relative
    // pool frame (poolOrigin=0 on generated ⇒ no-op). The origin word, every cluster/
    // DAG sphere, and the mesh sphere below all subtract it so a post-rebase attach
    // lands in the SAME frame as the resident tiles + the anchor-relative camera VP.
    const oX = this.poolOriginX;
    const oZ = this.poolOriginZ;
    if (hf) {
      const ehf = entry.hf;
      if (!ehf) throw new Error(`GeometryRegistry: tile slot ${slot} is not a heightfield mesh`);
      ehf.originX = hf.originX - oX;
      ehf.originZ = hf.originZ - oZ;
      ehf.cellSize = hf.cellSize;
    }
    const { gridVerts, indices, clusters } = build;
    const vCount = gridVerts.length;
    const tCount = indices.length / 3;
    const cCount = clusters.length;
    if (!Number.isInteger(tCount)) throw new Error('GeometryRegistry: tile indices not tri-aligned');
    if (cCount === 0) throw new Error('GeometryRegistry: tile has no clusters');
    if (vCount > pool.vertCap || tCount > pool.triCap || cCount > pool.clusterCap) {
      throw new Error(
        `GeometryRegistry: tile exceeds slot cap (verts ${vCount}/${pool.vertCap}, tris ${tCount}/` +
          `${pool.triCap}, clusters ${cCount}/${pool.clusterCap}) — raise reserveTilePool caps`,
      );
    }
    const vBase = this.tilePoolBase.vert + slot * pool.vertCap;
    const tBase = this.tilePoolBase.tri + slot * pool.triCap;
    const cBase = this.tilePoolBase.cluster + slot * pool.clusterCap;

    // verts: stride-1 (2e) — one packed GLOBAL texel coord per vert (height from tex)
    const vArr = this.hfVertsArr;
    for (let v = 0; v < vCount; v++) vArr[vBase + v] = (gridVerts[v] as number) >>> 0;
    // indices rebased onto the slot's vertex block
    const iArr = this.idxArr;
    const ibase = tBase * 3;
    for (let i = 0; i < indices.length; i++) iArr[ibase + i] = (indices[i] as number) + vBase;

    // cluster records (8-word) + DAG cut records (10-float)
    const cArr = this.clusterArr;
    const dArr = this.dagArr;
    const hfFlags = (CLUSTER_FLAG_HEIGHTFIELD | CLUSTER_FLAG_DAG) & 0xff;
    const dagSpheres = new Float32Array(cCount * 4);
    for (let c = 0; c < cCount; c++) {
      const dc = clusters[c] as DagCluster;
      const cb = (cBase + c) * CLUSTER_WORDS;
      cArr[cb] = f32Bits(dc.sx - oX); // S6d: WORLD-abs geo sphere → StreamOrigin-relative
      cArr[cb + 1] = f32Bits(dc.sy);
      cArr[cb + 2] = f32Bits(dc.sz - oZ);
      cArr[cb + 3] = f32Bits(dc.sr);
      cArr[cb + 4] = octEncode(dc.cax, dc.cay, dc.caz);
      cArr[cb + 5] = f32Bits(dc.ccos);
      cArr[cb + 6] = tBase + dc.triStart;
      if (dc.triCount > MAX_CLUSTER_TRIS) throw new Error('GeometryRegistry: tile cluster exceeds tri cap');
      // word7: triCount(0-7) | flags(8-9) | LOD level(10-15, for ?nanitedbg=lod) | handle(16-31)
      cArr[cb + 7] =
        ((dc.triCount & 0xff) | (hfFlags << 8) | ((dc.level & 0x3f) << 10) | (handle << 16)) >>> 0;

      const db = (cBase + c) * DAG_WORDS;
      const root = !Number.isFinite(dc.parentError);
      dArr[db] = dc.ownError;
      dArr[db + 1] = dc.oex - oX; // S6d: error-sphere centre → StreamOrigin-relative
      dArr[db + 2] = dc.oey;
      dArr[db + 3] = dc.oez - oZ;
      dArr[db + 4] = dc.oer;
      dArr[db + 5] = root ? DAG_ROOT_PARENT_ERR : dc.parentError;
      dArr[db + 6] = (root ? dc.oex : dc.pex) - oX;
      dArr[db + 7] = root ? dc.oey : dc.pey;
      dArr[db + 8] = (root ? dc.oez : dc.pez) - oZ;
      dArr[db + 9] = root ? dc.oer : dc.per;

      dagSpheres[c * 4] = dc.sx - oX; // S6d: → StreamOrigin-relative (mesh sphere below)
      dagSpheres[c * 4 + 1] = dc.sy;
      dagSpheres[c * 4 + 2] = dc.sz - oZ;
      dagSpheres[c * 4 + 3] = dc.sr;
    }

    // PERF-VB3: HIERARCHICAL-cull links for terrain. The regular grid is TILE-UNIFORM
    // (all clusters at a level share the error sphere), so the hierarchy is an
    // ANCHOR-CHAIN (buildHeightGridHierarchy), not a spatial tree: roots = coarsest
    // level; one anchor per level carries ALL of the next-finer level as children. Pack
    // into the slot's FIXED dagLinks region as [roots…][children…] (GLOBAL cluster ids);
    // each DAG record gets childBase/childCount (words 10/11), the mesh gets rootBase/
    // rootCount → kSeedRoots + the BFS traverse render terrain through the SAME hier cull
    // as vegetation (no brute path, no hybrid).
    const hier = buildHeightGridHierarchy(clusters);
    // item 6 + A14: the terrain anchor-chain depth also counts toward the global BFS
    // pass floor — tile attaches happen AFTER the frame froze hierDepth, so this is
    // the throw-loud site the silent _maxDagDepth growth used to hide behind.
    this.growDagDepth(maxChainDepth(hier));
    const dlBase = this.tilePoolBase.dagLinks + slot * pool.dagLinksCap;
    const rootCount = hier.rootIndices.length;
    const childTotal = hier.childIndices.length;
    if (rootCount + childTotal > pool.dagLinksCap) {
      throw new Error(`GeometryRegistry: tile dagLinks ${rootCount + childTotal} > cap ${pool.dagLinksCap}`);
    }
    const dl = this.dagLinksArr;
    for (let i = 0; i < rootCount; i++) dl[dlBase + i] = cBase + (hier.rootIndices[i] as number);
    const childBlockBase = dlBase + rootCount;
    for (let j = 0; j < childTotal; j++) dl[childBlockBase + j] = cBase + (hier.childIndices[j] as number);
    for (let c = 0; c < cCount; c++) {
      const db = (cBase + c) * DAG_WORDS;
      dArr[db + 10] = bitsF32(childBlockBase + (hier.childStart[c] as number));
      dArr[db + 11] = bitsF32(hier.childCount[c] as number);
    }

    entry.vertBase = vBase;
    entry.vertCount = vCount;
    entry.triBase = tBase;
    entry.triCount = tCount;
    entry.clusterBase = cBase;
    entry.clusterCount = cCount;
    entry.rootBase = dlBase;
    entry.rootCount = rootCount;
    entry.lodNext = LOD_NONE;
    entry.lodDist = 0;
    entry.flags |= MESH_FLAG_HASDAG;
    entry.sphere = meshSphereFromClusters(dagSpheres, cCount);
    this.writeMeshRecord(entry);

    this.pushRange(this.hfVertsAttr, vBase, vCount);
    this.pushRange(this.idxAttr, tBase * 3, tCount * 3);
    this.pushRange(this.clusterAttr, cBase * CLUSTER_WORDS, cCount * CLUSTER_WORDS);
    this.pushRange(this.dagAttr, cBase * DAG_WORDS, cCount * DAG_WORDS);
    this.pushRange(this.dagLinksAttr, dlBase, rootCount + childTotal);
    this.pushRange(this.meshAttr, handle * MESH_WORDS, MESH_WORDS);
    this._tileEpoch++; // S6d: terrain changed ⇒ shadow-clip levels re-raster (staggered)

    if (this.tileSlotOccupant) this.tileSlotOccupant[slot] = handle;
  }

  /**
   * Release a tile slot: zero its mesh draw (clusterCount=0 → lodSelectAndPush
   * enqueues ceil(0/64)=0 chunks → it vanishes from the cull, NO tombstoning) +
   * park its sphere off-world (frustum-reject in kInstCull) + return the slot to
   * the free-stack for reuse. Idempotent on an already-free slot. The slot's
   * stale buffer bytes stay until the next attachHeightDagTile overwrites them.
   */
  evictHeightDagTile(slot: number): void {
    if (!this.built) throw new Error('GeometryRegistry: evictHeightDagTile before build()');
    const pool = this.tilePool;
    if (!pool) throw new Error('GeometryRegistry: no tile pool reserved');
    if (slot < 0 || slot >= pool.slots) throw new Error(`GeometryRegistry: tile slot ${slot} out of range`);
    const occ = this.tileSlotOccupant;
    if (occ && occ[slot] === -1) return; // already free
    const handle = this.tilePoolHandles[slot] as number;
    const entry = this.entries[handle] as MeshEntry;
    entry.clusterCount = 0;
    entry.rootCount = 0; // PERF-VB3: kSeedRoots skips rootCount==0 (+ the parked sphere frustum-rejects)
    entry.flags &= ~MESH_FLAG_HASDAG;
    entry.sphere = [TILE_EVICTED_FAR, TILE_EVICTED_FAR, TILE_EVICTED_FAR, 0];
    this.writeMeshRecord(entry);
    this.pushRange(this.meshAttr, handle * MESH_WORDS, MESH_WORDS);
    if (occ) occ[slot] = -1;
    this.tileFreeSlots.push(slot);
    this.tileSlotParkedRootCount.delete(slot); // an evicted parked slot is no longer parked
    this._tileEpoch++; // S6d: terrain changed ⇒ shadow-clip levels re-raster (staggered)
  }

  /**
   * S6f PARK: retain a slot's geometry but kill its draw — set rootCount→0 so
   * kSeedRoots skips it (the same draw-kill evict uses, WITHOUT freeing the slot
   * or clearing clusterCount/sphere). This is the parked parent payload of a
   * Split node; `unparkTileSlot` restores the draw instantly (no bake). Cluster/
   * mesh spheres stay truthful, so a StreamOrigin rebase shifts a parked slot
   * exactly like a live one — no special case. Idempotent.
   */
  parkTileSlot(slot: number): void {
    if (!this.built) throw new Error('GeometryRegistry: parkTileSlot before build()');
    const pool = this.tilePool;
    if (!pool) throw new Error('GeometryRegistry: no tile pool reserved');
    if (slot < 0 || slot >= pool.slots) throw new Error(`GeometryRegistry: park tile slot ${slot} out of range`);
    if (this.tileSlotParkedRootCount.has(slot)) return; // already parked
    const entry = this.entries[this.tilePoolHandles[slot] as number] as MeshEntry;
    this.tileSlotParkedRootCount.set(slot, entry.rootCount);
    entry.rootCount = 0; // PERF-VB3: kSeedRoots skips rootCount==0 ⇒ emits nothing
    this.writeMeshRecord(entry);
    this.pushRange(this.meshAttr, (this.tilePoolHandles[slot] as number) * MESH_WORDS, MESH_WORDS);
    this._tileEpoch++;
  }

  /** S6f UNPARK: restore a parked slot's draw (the retained parent payload) — an
   *  O(1) rootCount rewrite, no bake. Idempotent on a non-parked slot. */
  unparkTileSlot(slot: number): void {
    if (!this.built) throw new Error('GeometryRegistry: unparkTileSlot before build()');
    const pool = this.tilePool;
    if (!pool) throw new Error('GeometryRegistry: no tile pool reserved');
    if (slot < 0 || slot >= pool.slots) throw new Error(`GeometryRegistry: unpark tile slot ${slot} out of range`);
    const saved = this.tileSlotParkedRootCount.get(slot);
    if (saved === undefined) return; // not parked
    const entry = this.entries[this.tilePoolHandles[slot] as number] as MeshEntry;
    entry.rootCount = saved;
    this.tileSlotParkedRootCount.delete(slot);
    this.writeMeshRecord(entry);
    this.pushRange(this.meshAttr, (this.tilePoolHandles[slot] as number) * MESH_WORDS, MESH_WORDS);
    this._tileEpoch++;
  }

  // ---- S8 FARTILE POOL (SPEC §5 A2/F-2) — streamed far-forest voxel heads --------------
  //
  // A fixed pool of parked 'voxel' mesh entries (one per 64 m tile slot, two calibrated
  // classes: FINE near-band tiles / COARSE far tiles) + a granule pool of 128-brick
  // blocks at the TAIL of the existing brick mega-buffer. Each voxel CLUSTER addresses
  // its own contiguous ≤128-brick range (word6 brickBase is absolute), so a tile's
  // bricks need NO cross-cluster contiguity: granules ≡ clusters, zero fragmentation.
  // The BRAIN owns all free-lists (single residency authority; throw-loud dry) — main
  // is a pure applier: attach writes the slot's cluster/DAG/dagLinks records through
  // the LIVE mirrors (never released) and the brick words writeBuffer-DIRECT (the
  // brick mirror IS released; F-5 guards appendBricks), then points the slot's
  // identity instance at the tile center (StreamOrigin-relative; rebase shifts it
  // through the normal non-hf cpuStream path).

  private ftPool: { slots: number; clusterCap: number; granules: number } | null = null;
  private ftPoolHandles: MeshHandle[] = [];
  private ftPoolInstFirst: number[] = [];
  private ftPoolBase = { cluster: 0, dagLinks: 0, brick: 0 };
  private ftSlotLive: Uint8Array | null = null;

  private ftSlotClusterBase(slot: number): number {
    return this.ftPoolBase.cluster + slot * (this.ftPool as NonNullable<typeof this.ftPool>).clusterCap;
  }

  /** the pool geometry the brain arms its free-lists from. */
  get fartilePoolInfo(): { slots: number; clusterCap: number; granules: number } {
    const p = this.ftPool;
    if (!p) throw new Error('GeometryRegistry: no fartile pool reserved');
    return { slots: p.slots, clusterCap: p.clusterCap, granules: p.granules };
  }

  /**
   * Reserve the fartile pool BEFORE build(): one parked voxel entry + one parked
   * identity instance per slot; cluster/dagLinks regions + the brick-tail granule
   * pool via addLate. Ceilings are logged by the caller (no-VRAM-hogs law).
   */
  reserveFartilePool(
    cap: { slots: number; clusterCap: number; granules: number },
    opts: { tint: number; nearDist: number },
  ): void {
    if (this.built) throw new Error('GeometryRegistry: reserveFartilePool after build()');
    if (this.ftPool) throw new Error('GeometryRegistry: fartile pool already reserved');
    const total = cap.slots;
    if (total <= 0 || cap.clusterCap <= 0 || cap.granules <= 0) {
      throw new Error('GeometryRegistry: reserveFartilePool caps must be positive');
    }
    this.ftPool = { ...cap };
    this.ftSlotLive = new Uint8Array(total);
    for (let s = 0; s < total; s++) {
      const handle = this.entries.length;
      if (handle >= 0xffff) throw new Error('GeometryRegistry: mesh id exceeds u16 (fartile pool too large)');
      const entry = this.newEntry(handle, 'voxel', {
        transformChannel: 'leaf',
        castShadows: false,
        twoSided: true,
        aggregate: true,
        matParam: opts.tint,
        swayPad: 0,
        label: `fartile/${s}`,
      }, false, 0);
      entry.flags |= MESH_FLAG_FARTILE;
      entry.nearDist = opts.nearDist;
      entry.lodDist = 100000; // maxDist — the tile head owns the far field outright
      entry.sphere = [TILE_EVICTED_FAR, TILE_EVICTED_FAR, TILE_EVICTED_FAR, 0];
      this.entries.push(entry);
      this.ftPoolHandles.push(handle);
      this.ftPoolInstFirst.push(this.instCursor);
      this.bindInstances(handle, {
        a: new Float32Array([0, GeometryRegistry.INST_PARK_Y, 0, 0]),
        b: new Float32Array([0, 0, 0, 0]),
      });
    }
    this.addLate({ clusters: cap.slots * cap.clusterCap, bricks: cap.granules * 128 });
    // fartile pyramids chain ≤ the crown voxlod depth (7) — grow pre-freeze so a
    // runtime attach can never out-deepen the frozen cull pass count (the attach
    // still folds its REAL chain depth, throw-loud, as the architectural guard).
    this.growDagDepth(7);
  }

  /**
   * Attach one baked fartile into a pool slot (post-build; runtime path). `packed`
   * = the tile's pyramid in CrownPack words form; `granules` = one granule id per
   * BLOCK (brain-reserved; brickBase = poolBrickBase + gid·128). Overwrites the
   * slot's cluster/DAG/dagLinks records + mesh record + identity instance in place.
   */
  attachFartileSlot(
    slot: number,
    packed: { vox: { levels?: { words: Uint32Array; occupied: Uint32Array; cellSize: number; blocks: { start: number; count: number; ownError: number; parentError: number; own: { x: number; y: number; z: number; r: number }; parent: { x: number; y: number; z: number; r: number } | null; childBlocks: number[] }[] }[] } },
    center: [number, number, number],
    granules: Uint32Array,
    renderer: Renderer | null,
  ): void {
    if (!this.built) throw new Error('GeometryRegistry: attachFartileSlot before build()');
    const pool = this.ftPool;
    if (!pool || !this.ftSlotLive) throw new Error('GeometryRegistry: no fartile pool reserved');
    if (slot < 0 || slot >= pool.slots) throw new Error(`GeometryRegistry: fartile slot ${slot} out of range`);
    const levels = packed.vox.levels;
    if (!levels || levels.length === 0) throw new Error('GeometryRegistry: fartile attach without pyramid levels');
    const handle = this.ftPoolHandles[slot] as number;
    const entry = this.entries[handle] as MeshEntry;
    const cBase = this.ftSlotClusterBase(slot);
    const dlBase = entry.rootBase; // slot's fixed dagLinks region (assigned at build)
    const cap = pool.clusterCap;

    // flatten blocks across levels → global block index; count clusters
    let n = 0;
    for (const l of levels) n += l.blocks.length;
    if (n === 0) throw new Error('GeometryRegistry: fartile attach with zero blocks');
    if (n > cap) throw new Error(`GeometryRegistry: fartile tile ${n} clusters > slot cap ${cap} (brain must pre-check)`);
    if (granules.length !== n) throw new Error(`GeometryRegistry: fartile granules ${granules.length} != blocks ${n}`);

    // brick words per block → granule bases (mirror pre-release, writeBuffer after)
    const brickWrite = this.ftBrickWriter(renderer);
    const globalOf: number[][] = levels.map(() => []);
    let g = 0;
    for (let L = 0; L < levels.length; L++) {
      const bl = (levels[L] as { blocks: unknown[] }).blocks;
      for (let bi = 0; bi < bl.length; bi++) globalOf[L]![bi] = g++;
    }
    const coarsest = levels.length - 1;
    const cArr = this.clusterArr;
    const dArr = this.dagArr;
    const dl = this.dagLinksArr;
    // roots = coarsest level's blocks; children flat-concat per block (same layout
    // as registerVoxelHead: [roots…][children…], childBase absolute into dagLinks)
    const rootIdx: number[] = [];
    for (let bi = 0; bi < (levels[coarsest] as { blocks: unknown[] }).blocks.length; bi++) {
      rootIdx.push(globalOf[coarsest]![bi] as number);
    }
    const rootCount = rootIdx.length;
    let linkTotal = rootCount;
    for (const l of levels) for (const blk of l.blocks) linkTotal += blk.childBlocks.length;
    if (linkTotal > cap) throw new Error(`GeometryRegistry: fartile dagLinks ${linkTotal} > slot cap ${cap}`);
    for (let i = 0; i < rootCount; i++) dl[dlBase + i] = cBase + (rootIdx[i] as number);
    let childCursor = dlBase + rootCount;

    let uMinX = Infinity, uMinY = Infinity, uMinZ = Infinity;
    let uMaxX = -Infinity, uMaxY = -Infinity, uMaxZ = -Infinity;
    // local hierarchy for the REAL chain-depth fold (throw-loud vs frozen hierDepth)
    const lcStart = new Uint32Array(n);
    const lcCount = new Uint32Array(n);
    const lcIdx: number[] = [];
    for (let L = 0; L < levels.length; L++) {
      const lvl = levels[L] as NonNullable<typeof levels>[number];
      for (let bi = 0; bi < lvl.blocks.length; bi++) {
        const blk = lvl.blocks[bi] as (typeof lvl.blocks)[number];
        const c = globalOf[L]![bi] as number;
        const gid = granules[c] as number;
        if (gid >= pool.granules) throw new Error(`GeometryRegistry: fartile granule ${gid} out of range`);
        const brickBase = this.ftPoolBase.brick + gid * 128;
        if (blk.count > 128) throw new Error(`GeometryRegistry: fartile block ${blk.count} bricks > granule 128`);
        brickWrite(lvl.words.subarray(blk.start * BRICK_WORDS, (blk.start + blk.count) * BRICK_WORDS), brickBase);
        // per-block AABB from the packed brick centers/halves (f32 — same as appendPackedCrown)
        let mnX = Infinity, mnY = Infinity, mnZ = Infinity, mxX = -Infinity, mxY = -Infinity, mxZ = -Infinity;
        for (let i = 0; i < blk.count; i++) {
          const [bcx, bcy, bcz, bh] = brickCenterHalf(lvl.words, blk.start + i);
          mnX = Math.min(mnX, bcx - bh); mxX = Math.max(mxX, bcx + bh);
          mnY = Math.min(mnY, bcy - bh); mxY = Math.max(mxY, bcy + bh);
          mnZ = Math.min(mnZ, bcz - bh); mxZ = Math.max(mxZ, bcz + bh);
        }
        uMinX = Math.min(uMinX, mnX); uMaxX = Math.max(uMaxX, mxX);
        uMinY = Math.min(uMinY, mnY); uMaxY = Math.max(uMaxY, mxY);
        uMinZ = Math.min(uMinZ, mnZ); uMaxZ = Math.max(uMaxZ, mxZ);
        const ccx = (mnX + mxX) * 0.5, ccy = (mnY + mxY) * 0.5, ccz = (mnZ + mxZ) * 0.5;
        const rr = Math.hypot((mxX - mnX) * 0.5, (mxY - mnY) * 0.5, (mxZ - mnZ) * 0.5);
        const cb = (cBase + c) * CLUSTER_WORDS;
        cArr[cb] = f32Bits(ccx);
        cArr[cb + 1] = f32Bits(ccy);
        cArr[cb + 2] = f32Bits(ccz);
        cArr[cb + 3] = f32Bits(rr);
        cArr[cb + 4] = octEncode(0, 1, 0);
        cArr[cb + 5] = f32Bits(-1); // cone disabled (bricks two-sided)
        cArr[cb + 6] = brickBase >>> 0; // word6 = brickBase (absolute)
        cArr[cb + 7] = ((blk.count & 0xff) | ((L & 0x3f) << 10) | (handle << 16)) >>> 0;
        // DAG cut record (same shape registerVoxelHead writes)
        const db = (cBase + c) * DAG_WORDS;
        const isRoot = L === coarsest;
        dArr[db] = blk.ownError;
        dArr[db + 1] = blk.own.x; dArr[db + 2] = blk.own.y; dArr[db + 3] = blk.own.z; dArr[db + 4] = blk.own.r;
        if (isRoot || !blk.parent) {
          dArr[db + 5] = DAG_ROOT_PARENT_ERR;
          dArr[db + 6] = blk.own.x; dArr[db + 7] = blk.own.y; dArr[db + 8] = blk.own.z; dArr[db + 9] = blk.own.r;
        } else {
          dArr[db + 5] = blk.parentError;
          dArr[db + 6] = blk.parent.x; dArr[db + 7] = blk.parent.y; dArr[db + 8] = blk.parent.z; dArr[db + 9] = blk.parent.r;
        }
        // children live one level FINER (L-1); ids resolve through globalOf
        lcStart[c] = lcIdx.length;
        lcCount[c] = blk.childBlocks.length;
        dArr[db + 10] = bitsF32(childCursor);
        dArr[db + 11] = bitsF32(blk.childBlocks.length);
        for (const cbi of blk.childBlocks) {
          const childGlobal = globalOf[L - 1]![cbi] as number;
          dl[childCursor++] = cBase + childGlobal;
          lcIdx.push(childGlobal);
        }
      }
    }
    this.growDagDepth(
      maxChainDepth({
        childStart: lcStart,
        childCount: lcCount,
        childIndices: Uint32Array.from(lcIdx),
        rootIndices: Uint32Array.from(rootIdx),
      }),
    );

    entry.clusterCount = n;
    entry.rootCount = rootCount;
    entry.flags |= MESH_FLAG_HASDAG;
    entry.sphere = [
      (uMinX + uMaxX) * 0.5,
      (uMinY + uMaxY) * 0.5,
      (uMinZ + uMaxZ) * 0.5,
      Math.hypot((uMaxX - uMinX) * 0.5, (uMaxY - uMinY) * 0.5, (uMaxZ - uMinZ) * 0.5),
    ];
    this.writeMeshRecord(entry);
    // identity instance at the tile center (StreamOrigin-relative pool frame)
    const d = (this.ftPoolInstFirst[slot] as number) * 8;
    this.instArr[d] = center[0] - this.poolOriginX;
    this.instArr[d + 1] = center[1];
    this.instArr[d + 2] = center[2] - this.poolOriginZ;
    this.instArr[d + 3] = 1;
    this.pushRange(this.clusterAttr, cBase * CLUSTER_WORDS, n * CLUSTER_WORDS);
    this.pushRange(this.dagAttr, cBase * DAG_WORDS, n * DAG_WORDS);
    this.pushRange(this.dagLinksAttr, dlBase, linkTotal);
    this.pushRange(this.meshAttr, handle * MESH_WORDS, MESH_WORDS);
    this.pushRange(this.instAttr, (this.ftPoolInstFirst[slot] as number) * 8, 8);
    this.ftSlotLive[slot] = 1;
  }

  /** brick words → pool region: live mirror + pushRange before release (early boot
   *  frames), queue.writeBuffer DIRECT after (the pool has no CPU mirror — F-5). */
  private ftBrickWriter(renderer: Renderer | null): (words: Uint32Array, brickBase: number) => void {
    if (!this.mirrorsReleased) {
      return (words, brickBase) => {
        this.voxelBricksArr.set(words, brickBase * BRICK_WORDS);
        this.pushRange(this.voxelBricksAttr, brickBase * BRICK_WORDS, words.length);
      };
    }
    if (!renderer) throw new Error('GeometryRegistry: fartile brick write after mirror release needs the renderer');
    const backend = (renderer as unknown as { backend: { device?: GPUDevice; get(a: unknown): { buffer?: GPUBuffer } | undefined } }).backend;
    const gpuBuf = backend.get(this.voxelBricksAttr)?.buffer;
    const device = backend.device;
    if (!gpuBuf || !device) throw new Error('GeometryRegistry: brick GPUBuffer missing post-release (backend accessor)');
    return (words, brickBase) => {
      device.queue.writeBuffer(gpuBuf, brickBase * BRICK_WORDS * 4, words.buffer, words.byteOffset, words.byteLength);
    };
  }

  /** Park a fartile slot (evict): draw dies (clusterCount/rootCount 0 + parked
   *  sphere + parked instance); records stay until the next attach overwrites.
   *  Idempotent. The brain returns the slot + its granules to ITS free-lists. */
  evictFartileSlot(slot: number): void {
    if (!this.built) throw new Error('GeometryRegistry: evictFartileSlot before build()');
    const pool = this.ftPool;
    if (!pool || !this.ftSlotLive) throw new Error('GeometryRegistry: no fartile pool reserved');
    if (slot < 0 || slot >= pool.slots) throw new Error(`GeometryRegistry: fartile slot ${slot} out of range`);
    if (this.ftSlotLive[slot] === 0) return;
    const handle = this.ftPoolHandles[slot] as number;
    const entry = this.entries[handle] as MeshEntry;
    entry.clusterCount = 0;
    entry.rootCount = 0;
    entry.flags &= ~MESH_FLAG_HASDAG;
    entry.sphere = [TILE_EVICTED_FAR, TILE_EVICTED_FAR, TILE_EVICTED_FAR, 0];
    this.writeMeshRecord(entry);
    const d = (this.ftPoolInstFirst[slot] as number) * 8;
    this.instArr[d] = 0;
    this.instArr[d + 1] = GeometryRegistry.INST_PARK_Y;
    this.instArr[d + 2] = 0;
    this.instArr[d + 3] = 0;
    this.pushRange(this.meshAttr, handle * MESH_WORDS, MESH_WORDS);
    this.pushRange(this.instAttr, (this.ftPoolInstFirst[slot] as number) * 8, 8);
    this.ftSlotLive[slot] = 0;
  }

  /** post-build backing arrays + attributes (probe/validation use only) */
  debug(): {
    arrays: {
      verts: Uint32Array;
      hfVerts: Uint32Array;
      indices: Uint32Array;
      clusters: Uint32Array;
      meshes: Uint32Array;
      instances: Float32Array;
      instanceMesh: Uint32Array;
      dag: Float32Array;
      dagLinks: Uint32Array;
    };
    attrs: {
      verts: StorageBufferAttribute;
      hfVerts: StorageBufferAttribute;
      indices: StorageBufferAttribute;
      clusters: StorageBufferAttribute;
      meshes: StorageBufferAttribute;
      instances: StorageBufferAttribute;
      instanceMesh: StorageBufferAttribute;
      dag: StorageBufferAttribute;
    };
  } {
    if (!this.built) throw new Error('GeometryRegistry: debug() before build()');
    return {
      arrays: {
        verts: this.vertsArr,
        hfVerts: this.hfVertsArr,
        indices: this.idxArr,
        clusters: this.clusterArr,
        meshes: this.meshArr,
        instances: this.instArr,
        instanceMesh: this.instMeshArr,
        dag: this.dagArr,
        dagLinks: this.dagLinksArr,
      },
      attrs: {
        verts: this.vertsAttr,
        hfVerts: this.hfVertsAttr,
        indices: this.idxAttr,
        clusters: this.clusterAttr,
        meshes: this.meshAttr,
        instances: this.instAttr,
        instanceMesh: this.instMeshAttr,
        dag: this.dagAttr,
      },
    };
  }

  /** memory actually used (not capacity), bytes per blob */
  bytes(): BuildReport['bytes'] {
    const verts = this.vertCursor * VERT_WORDS * 4;
    const hfVerts = this.hfVertCursor * 4; // stride-1 terrain-DAG buffer (2e)
    const indices = this.triCursor * 3 * 4;
    const clusters = this.clusterCursor * CLUSTER_WORDS * 4;
    const meshTable = this.entries.length * MESH_WORDS * 4;
    const instances = this.instCursor * (32 + 4);
    const bricks = this.brickCursor * BRICK_WORDS * 4;
    return {
      verts,
      hfVerts,
      indices,
      clusters,
      meshTable,
      instances,
      bricks,
      total: verts + hfVerts + indices + clusters + meshTable + instances + bricks,
    };
  }

  // -- internals ------------------------------------------------------------

  private packExplicit(
    handle: number,
    src: ExplicitSource,
    matClass: MaterialClassId,
    opts: RegisterOpts,
  ): MeshEntry {
    const vertCount = src.positions.length / 3;
    if (!Number.isInteger(vertCount)) throw new Error('GeometryRegistry: positions not stride-3');
    if (src.normals.length !== src.positions.length) {
      throw new Error('GeometryRegistry: normals length mismatch');
    }
    const built = clusterize(src.positions, 3, src.indices, MAX_CLUSTER_TRIS);
    this.clusterizeMs += built.stats.buildMs;
    this.checkRoom(vertCount, built.indices.length / 3, built.clusterCount, 0);

    const packed = new Uint32Array(vertCount * VERT_WORDS);
    for (let v = 0; v < vertCount; v++) {
      const b = v * VERT_WORDS;
      packed[b] = f32Bits(src.positions[v * 3] as number);
      packed[b + 1] = f32Bits(src.positions[v * 3 + 1] as number);
      packed[b + 2] = f32Bits(src.positions[v * 3 + 2] as number);
      packed[b + 3] = octEncode(
        src.normals[v * 3] as number,
        src.normals[v * 3 + 1] as number,
        src.normals[v * 3 + 2] as number,
      );
      const u = src.uvs ? (src.uvs[v * 2] as number) : 0;
      const w = src.uvs ? (src.uvs[v * 2 + 1] as number) : 0;
      packed[b + 4] = (f32ToF16(u) | (f32ToF16(w) << 16)) >>> 0;
      packed[b + 5] = src.vdata ? (src.vdata[v] as number) >>> 0 : 0;
    }

    const entry = this.newEntry(handle, matClass, opts, false, 0);
    entry.vertBase = this.vertCursor;
    entry.vertCount = vertCount;
    entry.triBase = this.triCursor;
    entry.triCount = built.indices.length / 3;
    entry.clusterBase = this.clusterCursor;
    entry.clusterCount = built.clusterCount;
    entry.stats = built.stats;
    this.vertCursor += vertCount;
    this.triCursor += entry.triCount;
    this.clusterCursor += built.clusterCount;

    // globalize indices
    const idx = new Uint32Array(built.indices.length);
    for (let i = 0; i < idx.length; i++) idx[i] = (built.indices[i] as number) + entry.vertBase;
    entry.packedVerts = packed;
    entry.packedIdx = idx;
    entry.clusterRecs = this.encodeClusterRecs(entry, built);
    entry.sphere = meshSphereFromClusters(built.sphere, built.clusterCount);
    return entry;
  }

  private packHeightfield(
    handle: number,
    src: HeightfieldSource,
    matClass: MaterialClassId,
    opts: RegisterOpts,
  ): MeshEntry {
    const w = src.winQuads;
    if (w * w * 2 > MAX_CLUSTER_TRIS) {
      throw new Error(`GeometryRegistry: heightfield window ${w}² → ${w * w * 2} tris > ${MAX_CLUSTER_TRIS}`);
    }
    if (src.quadsX > 0xffff || src.quadsZ > 0xffff) throw new Error('GeometryRegistry: heightfield quads exceed u16');
    const windowsX = Math.ceil(src.quadsX / w);
    const windowsZ = Math.ceil(src.quadsZ / w);
    if (src.minMax.length !== windowsX * windowsZ * 2) {
      throw new Error(
        `GeometryRegistry: heightfield minMax length ${src.minMax.length} != ${windowsX * windowsZ * 2}`,
      );
    }
    this.checkRoom(0, 0, windowsX * windowsZ, 0);
    const entry = this.newEntry(handle, matClass, { ...opts, transformChannel: 'terrain' }, true, w);
    entry.hf = {
      originX: src.originX,
      originZ: src.originZ,
      cellSize: src.cellSize,
      quadsX: src.quadsX,
      quadsZ: src.quadsZ,
      windowsX,
      windowsZ,
    };
    const n = windowsX * windowsZ;
    entry.clusterBase = this.clusterCursor;
    entry.clusterCount = n;
    this.clusterCursor += n;

    // mesh sphere (world-space — heightfield instances are identity): full
    // grid extent horizontally, global height range vertically
    let gMin = Number.POSITIVE_INFINITY;
    let gMax = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < n; i++) {
      gMin = Math.min(gMin, src.minMax[i * 2] as number);
      gMax = Math.max(gMax, src.minMax[i * 2 + 1] as number);
    }
    const exAll = src.quadsX * src.cellSize;
    const ezAll = src.quadsZ * src.cellSize;
    entry.sphere = [
      src.originX + exAll / 2,
      (gMin + gMax) / 2,
      src.originZ + ezAll / 2,
      Math.hypot(exAll / 2, (gMax - gMin) / 2, ezAll / 2),
    ];

    const recs = new Uint32Array(n * CLUSTER_WORDS);
    for (let gz = 0; gz < windowsZ; gz++) {
      for (let gx = 0; gx < windowsX; gx++) {
        // edge windows clamp to the remaining quads (partial clusters)
        const qx = Math.min(w, src.quadsX - gx * w);
        const qz = Math.min(w, src.quadsZ - gz * w);
        const i = gz * windowsX + gx;
        const mn = src.minMax[i * 2] as number;
        const mx = src.minMax[i * 2 + 1] as number;
        const ex = qx * src.cellSize;
        const ez = qz * src.cellSize;
        const b = i * CLUSTER_WORDS;
        recs[b] = f32Bits(src.originX + gx * w * src.cellSize + ex / 2);
        recs[b + 1] = f32Bits((mn + mx) / 2);
        recs[b + 2] = f32Bits(src.originZ + gz * w * src.cellSize + ez / 2);
        recs[b + 3] = f32Bits(Math.hypot(ex / 2, (mx - mn) / 2, ez / 2));
        recs[b + 4] = octEncode(0, 1, 0);
        recs[b + 5] = f32Bits(-1); // cone disabled — windows face anywhere
        recs[b + 6] = (gx | (gz << 16)) >>> 0;
        recs[b + 7] = ((qx * qz * 2) | (CLUSTER_FLAG_HEIGHTFIELD << 8) | (handle << 16)) >>> 0;
      }
    }
    entry.clusterRecs = recs;
    return entry;
  }

  private newEntry(
    handle: number,
    matClass: MaterialClassId,
    opts: RegisterOpts,
    heightfield: boolean,
    winQuads: number,
  ): MeshEntry {
    const channel = TRANSFORM_CHANNEL[opts.transformChannel ?? 'rigid'];
    let flags = 0;
    if (heightfield) flags |= MESH_FLAG_HEIGHTFIELD;
    if (opts.aggregate) flags |= MESH_FLAG_AGGREGATE;
    if (opts.castShadows !== false) flags |= MESH_FLAG_CAST_SHADOWS;
    if (opts.twoSided) flags |= MESH_FLAG_TWO_SIDED;
    return {
      handle,
      label: opts.label ?? `mesh${handle}`,
      matClass,
      channel,
      flags,
      winQuads,
      swayPad: opts.swayPad ?? 0,
      matParam: opts.matParam ?? 0,
      vertBase: 0,
      vertCount: 0,
      triBase: 0,
      triCount: 0,
      clusterBase: 0,
      clusterCount: 0,
      rootBase: 0,
      rootCount: 0,
      instFirst: 0,
      instCount: 0,
      lodNext: LOD_NONE,
      lodDist: 0,
      nearDist: 0,
      sphere: [0, 0, 0, 0],
      uploaded: false,
    };
  }

  /** PERF-3 (stage 1): per-cluster vertex-cache records for every BOOT cluster. A cluster
   *  whose global vertex indices span a tight range (≤ VCACHE_VERTS) gets [vMin, count]
   *  so the SW raster cooperatively transforms that range ONCE into workgroup shared
   *  memory (explicit meshes: avg unique 82, 95% range ≤128). Others stay 0 ⇒ per-thread
   *  fallback. Window-grid heightfield clusters (isHF && !isDAG) have NO index buffer
   *  (triStart packs gx|gz) ⇒ skipped. Streamed terrain tiles attach POST-build ⇒ stay 0
   *  (their DAG index ranges are wide anyway — 40% > 1024). */
  /** Global-index span of a cluster's triangles as [mn, count] (count = max−min+1,
   *  the unique-vertex window width). Caller writes vcompactArr only when
   *  0 < count ≤ VCACHE_VERTS; a wider span stays 0 = per-thread projection fallback.
   *  Shared by populateVCompact (boot clusters) and attachDag (post-build DAG clusters)
   *  so both cache the SAME window semantics. `globalTriStart` is the cluster's word-6
   *  global triStart; indices are already rebased onto the appended vertex block. */
  private vcompactSpan(
    idx: Uint32Array,
    globalTriStart: number,
    triCount: number,
  ): { mn: number; count: number } {
    let mn = 0xffffffff;
    let mx = 0;
    for (let t = 0; t < triCount; t++) {
      for (let v = 0; v < 3; v++) {
        const vi = idx[(globalTriStart + t) * 3 + v] ?? 0;
        if (vi < mn) mn = vi;
        if (vi > mx) mx = vi;
      }
    }
    return { mn, count: mx - mn + 1 };
  }

  private populateVCompact(): void {
    const idx = this.idxArr;
    const cl = this.clusterArr;
    let cached = 0;
    let eligible = 0;
    let tooWide = 0;
    let sumCount = 0;
    for (let c = 0; c < this.clusterCursor; c++) {
      const triStart = cl[c * CLUSTER_WORDS + 6] ?? 0;
      const w7 = cl[c * CLUSTER_WORDS + 7] ?? 0;
      const triCount = w7 & 0xff;
      const flags = (w7 >>> 8) & 0xff;
      const isHF = (flags & CLUSTER_FLAG_HEIGHTFIELD) !== 0;
      const isDAG = (flags & CLUSTER_FLAG_DAG) !== 0;
      if (triCount === 0 || (isHF && !isDAG)) continue;
      eligible++;
      const { mn, count } = this.vcompactSpan(idx, triStart, triCount);
      if (count > 0 && count <= VCACHE_VERTS) {
        this.vcompactArr[c * 2] = mn;
        this.vcompactArr[c * 2 + 1] = count;
        cached++;
        sumCount += count;
      } else if (count > VCACHE_VERTS) {
        tooWide++;
      }
    }
    const tot = this.clusterCursor;
    console.log(
      `[laas][vcompact] cached ${cached}/${tot} boot clusters (${((100 * cached) / Math.max(1, tot)).toFixed(1)}%) — ` +
        `eligible(non-window) ${eligible}, tooWide>${VCACHE_VERTS} ${tooWide}, avg count ${(sumCount / Math.max(1, cached)).toFixed(0)}. ` +
        `Window-grid + streamed-terrain stay 0=fallback (stage 1 = explicit only).`,
    );
  }

  private encodeClusterRecs(entry: MeshEntry, built: BuiltClusters): Uint32Array {
    const recs = new Uint32Array(built.clusterCount * CLUSTER_WORDS);
    for (let c = 0; c < built.clusterCount; c++) {
      const b = c * CLUSTER_WORDS;
      recs[b] = f32Bits(built.sphere[c * 4] as number);
      recs[b + 1] = f32Bits(built.sphere[c * 4 + 1] as number);
      recs[b + 2] = f32Bits(built.sphere[c * 4 + 2] as number);
      recs[b + 3] = f32Bits(built.sphere[c * 4 + 3] as number);
      recs[b + 4] = octEncode(
        built.cone[c * 4] as number,
        built.cone[c * 4 + 1] as number,
        built.cone[c * 4 + 2] as number,
      );
      recs[b + 5] = f32Bits(built.cone[c * 4 + 3] as number);
      recs[b + 6] = entry.triBase + (built.triStart[c] as number);
      const tc = built.triCount[c] as number;
      if (tc > MAX_CLUSTER_TRIS) throw new Error('GeometryRegistry: cluster exceeds tri cap');
      recs[b + 7] = (tc | (entry.handle << 16)) >>> 0;
    }
    return recs;
  }

  private copyEntry(e: MeshEntry): void {
    if (e.packedVerts) this.vertsArr.set(e.packedVerts, e.vertBase * VERT_WORDS);
    if (e.packedIdx) this.idxArr.set(e.packedIdx, e.triBase * 3);
    if (e.clusterRecs) this.clusterArr.set(e.clusterRecs, e.clusterBase * CLUSTER_WORDS);
    this.writeMeshRecord(e);
    e.packedVerts = undefined;
    e.packedIdx = undefined;
    e.clusterRecs = undefined;
    e.uploaded = true;
  }

  private writeMeshRecord(e: MeshEntry): void {
    const b = e.handle * MESH_WORDS;
    const m = this.meshArr;
    m[b] = e.clusterBase;
    m[b + 1] = e.clusterCount;
    m[b + 2] = e.instFirst;
    m[b + 3] = e.instCount;
    m[b + 4] = e.lodNext;
    m[b + 5] = f32Bits(e.lodDist);
    m[b + 6] = (e.channel | (MATERIAL_CLASS[e.matClass] << 8) | (e.flags << 16) | (e.winQuads << 24)) >>> 0;
    // word 7: hfOriginX (heightfield) | matParam raw-u32 (explicit, e.g. bark
    // texture-array slice — read raw by the resolve, never as a float)
    m[b + 7] = e.hf ? f32Bits(e.hf.originX) : e.matParam >>> 0;
    // word 8: hfOriginZ (heightfield) | nearDist f32 (explicit/voxel — the cull's
    // per-mesh NEAR draw envelope, Stage-3a mesh→voxel handoff). A heightfield is never
    // a voxel head, so the two readings never collide; the cull only reads nearDist on
    // hierarchical meshes (rootCount>0), where hf is undefined.
    m[b + 8] = e.hf ? f32Bits(e.hf.originZ) : f32Bits(e.nearDist);
    m[b + 9] = f32Bits(e.hf?.cellSize ?? 0);
    m[b + 10] = ((e.hf?.quadsX ?? 0) | ((e.hf?.quadsZ ?? 0) << 16)) >>> 0;
    m[b + 11] = f32Bits(e.swayPad);
    m[b + 12] = f32Bits(e.sphere[0]);
    m[b + 13] = f32Bits(e.sphere[1]);
    m[b + 14] = f32Bits(e.sphere[2]);
    m[b + 15] = f32Bits(e.sphere[3]);
    // N8-HIC: hierarchical-cull root seed range into gpu.dagLinks (0/0 ⇒ the mesh
    // is non-hierarchical: terrain DAG / discrete-LOD / window grid → brute-force path)
    m[b + 16] = e.rootBase >>> 0;
    m[b + 17] = e.rootCount >>> 0;
  }

  private rewriteMeshRecord(e: MeshEntry): void {
    this.writeMeshRecord(e);
    this.pushRange(this.meshAttr, e.handle * MESH_WORDS, MESH_WORDS);
  }

  private copyCpuStream(s: CpuInstances): void {
    const count = s.a.length / 4;
    for (let i = 0; i < count; i++) {
      const d = (s.first + i) * 8;
      this.instArr[d] = s.a[i * 4] as number;
      this.instArr[d + 1] = s.a[i * 4 + 1] as number;
      this.instArr[d + 2] = s.a[i * 4 + 2] as number;
      this.instArr[d + 3] = s.a[i * 4 + 3] as number;
      this.instArr[d + 4] = s.b[i * 4] as number;
      this.instArr[d + 5] = s.b[i * 4 + 1] as number;
      this.instArr[d + 6] = s.b[i * 4 + 2] as number;
      this.instArr[d + 7] = s.b[i * 4 + 3] as number;
      this.instMeshArr[s.first + i] = s.meshId;
    }
    s.copied = true;
  }

  private runGpuCopies(renderer?: Renderer): void {
    const pending = this.gpuStreams.filter((s) => !s.copied);
    if (pending.length === 0) return;
    if (!renderer) {
      throw new Error('GeometryRegistry: GPU instance streams pending — build/flush need the renderer');
    }
    for (const s of pending) {
      const { bufA, bufB, count, srcFirst = 0 } = s.stream;
      const dstFirst = s.first;
      const meshId = s.meshId;
      const instRW = this.instRW;
      const instMeshRW = this.instMeshRW;
      const kernel = Fn(() => {
        returnIf(instanceIndex.greaterThanEqual(uint(count)));
        const src = uint(srcFirst).add(instanceIndex).toVar();
        const dst = uint(dstFirst).add(instanceIndex).toVar();
        instRW.element(dst.mul(uint(2))).assign((bufA as unknown as BufOf<NV4>).element(src));
        instRW.element(dst.mul(uint(2)).add(uint(1))).assign((bufB as unknown as BufOf<NV4>).element(src));
        elemUW(instMeshRW, dst).assign(uint(meshId));
      })().compute(count, [64]);
      (kernel as { setName(n: string): unknown }).setName('nanGeoInstCopy');
      dispatch(renderer, kernel);
      s.copied = true;
    }
  }

  private pushRange(attr: StorageBufferAttribute, start: number, count: number): void {
    attr.addUpdateRange(start, count);
    attr.needsUpdate = true;
  }

  /** post-build capacity guard — checked BEFORE any cursor mutation (F14: explicit, never silent) */
  private checkRoom(addVerts: number, addTris: number, addClusters: number, addInst: number): void {
    if (!this.built) return;
    const c = this.caps;
    if (
      this.vertCursor + addVerts > c.verts ||
      this.triCursor + addTris > c.tris ||
      this.clusterCursor + addClusters > c.clusters ||
      this.instCursor + addInst > c.instances
    ) {
      throw new Error(
        `GeometryRegistry: late capacity exceeded (verts ${this.vertCursor}+${addVerts}/${c.verts}, ` +
          `tris ${this.triCursor}+${addTris}/${c.tris}, clusters ${this.clusterCursor}+${addClusters}/${c.clusters}, ` +
          `inst ${this.instCursor}+${addInst}/${c.instances}) — raise the late budget`,
      );
    }
  }

  private updateCounters(counters: Record<string, number>): void {
    counters['nanite.meshes'] = this.entries.length;
    counters['nanite.clusters'] = this.clusterCursor;
    counters['nanite.trisK'] = Math.round(this.triCursor / 1000);
    counters['nanite.inst'] = this.instCursor;
    counters['nanite.mb'] = Math.round(this.bytes().total / 1e6);
  }

  private report(totalMs: number): BuildReport {
    const perMesh: MeshReport[] = this.entries.map((e) => ({
      label: e.label,
      matClass: e.matClass,
      verts: e.vertCount,
      tris: e.triCount || e.clusterCount * e.winQuads * e.winQuads * 2,
      clusters: e.clusterCount,
      avgTris: e.stats?.avgTris ?? e.winQuads * e.winQuads * 2,
      fullFrac: e.stats?.fullFrac ?? 1,
      buildMs: e.stats?.buildMs ?? 0,
    }));
    const by = this.bytes();
    const rows = perMesh.map(
      (r) =>
        `${r.label.padEnd(18)} ${r.matClass.padEnd(8)} ${String(r.verts).padStart(8)} ` +
        `${String(r.tris).padStart(9)} ${String(r.clusters).padStart(7)} ${r.avgTris.toFixed(1).padStart(6)} ` +
        `${(r.fullFrac * 100).toFixed(0).padStart(4)}% ${r.buildMs.toFixed(1).padStart(7)}ms`,
    );
    const table =
      `${'mesh'.padEnd(18)} ${'class'.padEnd(8)} ${'verts'.padStart(8)} ${'tris'.padStart(9)} ` +
      `${'clstr'.padStart(7)} ${'avg'.padStart(6)} ${'full'.padStart(5)} ${'build'.padStart(9)}\n` +
      `${rows.join('\n')}\n` +
      `total: ${this.entries.length} meshes, ${this.clusterCursor} clusters, ${this.triCursor} tris, ` +
      `${this.vertCursor} verts, ${this.instCursor} instances; ` +
      `${(by.total / 1e6).toFixed(1)} MB (verts ${(by.verts / 1e6).toFixed(1)} + idx ${(by.indices / 1e6).toFixed(1)} + ` +
      `clstr ${(by.clusters / 1e6).toFixed(1)} + inst ${(by.instances / 1e6).toFixed(1)}); ` +
      `clusterize ${this.clusterizeMs.toFixed(0)} ms, build ${totalMs.toFixed(0)} ms`;
    return {
      meshes: this.entries.length,
      clusters: this.clusterCursor,
      tris: this.triCursor,
      verts: this.vertCursor,
      instances: this.instCursor,
      bytes: by,
      clusterizeMs: this.clusterizeMs,
      totalMs,
      perMesh,
      table,
    };
  }

  // -- GPU validation (browser only) -----------------------------------------

  /**
   * Decode-roundtrip test: a kernel reads the first K packed vertices (of the
   * first explicit mesh) and the first K cluster records through readVertex/
   * readCluster and writes the decoded values; compared against the CPU
   * mirrors bit-for-bit (within f32-vs-f64 normalize noise). Proves the TSL
   * decode path compiles and matches before N2/N3 consume it.
   */
  async validateOnGpu(renderer: Renderer, k = 64): Promise<{ pass: boolean; detail: string }> {
    if (!this.built) throw new Error('GeometryRegistry: validateOnGpu before build()');
    const probeMesh = this.entries.find((e) => e.vertCount > 0);
    const nv = Math.min(k, probeMesh?.vertCount ?? 0);
    const nc = Math.min(k, this.clusterCursor);
    const STRIDE = 24;
    const outArr = new Float32Array(Math.max(1, k * STRIDE));
    const outAttr = new StorageBufferAttribute(outArr, 1);
    outAttr.name = 'nanGeoProbeReadback';
    const out = storage(outAttr, 'float', outArr.length);
    const vertBase = probeMesh?.vertBase ?? 0;
    const gpu = this.gpu;
    const kernel = Fn(() => {
      const i = instanceIndex.toVar();
      returnIf(i.greaterThanEqual(uint(k)));
      const at = (o: number): NU => i.mul(uint(STRIDE)).add(uint(o));
      If(i.lessThan(uint(nv)), () => {
        const v = readVertex(gpu.verts, uint(vertBase).add(i));
        out.element(at(0)).assign(v.pos.x);
        out.element(at(1)).assign(v.pos.y);
        out.element(at(2)).assign(v.pos.z);
        out.element(at(3)).assign(v.nrm.x);
        out.element(at(4)).assign(v.nrm.y);
        out.element(at(5)).assign(v.nrm.z);
        out.element(at(6)).assign(v.uv.x);
        out.element(at(7)).assign(v.uv.y);
        out.element(at(8)).assign(toF(v.vdata.bitAnd(uint(0xffff))));
        out.element(at(9)).assign(toF(v.vdata.shiftRight(uint(16))));
      });
      If(i.lessThan(uint(nc)), () => {
        const c = readCluster(gpu.clusters, i);
        out.element(at(12)).assign(c.sphere.x);
        out.element(at(13)).assign(c.sphere.y);
        out.element(at(14)).assign(c.sphere.z);
        out.element(at(15)).assign(c.sphere.w);
        out.element(at(16)).assign(c.coneAxis.x);
        out.element(at(17)).assign(c.coneAxis.y);
        out.element(at(18)).assign(c.coneAxis.z);
        out.element(at(19)).assign(c.coneCos);
        out.element(at(20)).assign(toF(c.triStart.bitAnd(uint(0xffff))));
        out.element(at(21)).assign(toF(c.triStart.shiftRight(uint(16))));
        out.element(at(22)).assign(toF(c.triCount.add(c.flags.shiftLeft(uint(8)))));
        out.element(at(23)).assign(toF(c.meshId));
      });
    })().compute(k, [64]);
    (kernel as { setName(n: string): unknown }).setName('nanGeoValidate');
    dispatch(renderer, kernel);
    const ab = await readBuffer(renderer, outAttr, 0, outArr.length * 4);
    const got = new Float32Array(ab);

    let maxVert = 0;
    let maxCluster = 0;
    let exactFails = 0;
    for (let i = 0; i < nv; i++) {
      const cpu = decodeVertexCPU(this.vertsArr, vertBase + i);
      const o = i * STRIDE;
      const errs = [
        Math.abs((got[o] as number) - cpu.pos[0]),
        Math.abs((got[o + 1] as number) - cpu.pos[1]),
        Math.abs((got[o + 2] as number) - cpu.pos[2]),
        Math.abs((got[o + 3] as number) - cpu.nrm[0]),
        Math.abs((got[o + 4] as number) - cpu.nrm[1]),
        Math.abs((got[o + 5] as number) - cpu.nrm[2]),
        Math.abs((got[o + 6] as number) - cpu.uv[0]),
        Math.abs((got[o + 7] as number) - cpu.uv[1]),
      ];
      for (const e of errs) if (e > maxVert) maxVert = e;
      if ((got[o + 8] as number) !== (cpu.vdata & 0xffff) || (got[o + 9] as number) !== cpu.vdata >>> 16) {
        exactFails++;
      }
    }
    for (let i = 0; i < nc; i++) {
      const cpu = decodeClusterCPU(this.clusterArr, i);
      const o = i * STRIDE;
      const errs = [
        Math.abs((got[o + 12] as number) - cpu.sphere[0]),
        Math.abs((got[o + 13] as number) - cpu.sphere[1]),
        Math.abs((got[o + 14] as number) - cpu.sphere[2]),
        Math.abs((got[o + 15] as number) - cpu.sphere[3]),
        Math.abs((got[o + 16] as number) - cpu.coneAxis[0]),
        Math.abs((got[o + 17] as number) - cpu.coneAxis[1]),
        Math.abs((got[o + 18] as number) - cpu.coneAxis[2]),
        Math.abs((got[o + 19] as number) - cpu.coneCos),
      ];
      for (const e of errs) if (e > maxCluster) maxCluster = e;
      if (
        (got[o + 20] as number) !== (cpu.triStart & 0xffff) ||
        (got[o + 21] as number) !== cpu.triStart >>> 16 ||
        (got[o + 22] as number) !== cpu.triCount + (cpu.flags << 8) ||
        (got[o + 23] as number) !== cpu.meshId
      ) {
        exactFails++;
      }
    }
    const pass = maxVert < 1e-4 && maxCluster < 1e-4 && exactFails === 0;
    const detail =
      `verts ${nv} maxErr ${maxVert.toExponential(2)}, clusters ${nc} maxErr ` +
      `${maxCluster.toExponential(2)}, exactFails ${exactFails}`;
    return { pass, detail };
  }
}
