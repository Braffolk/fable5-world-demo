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
 * matClass all live in the mesh record); triCount fits u8 at the 128-tri cap.
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
import type { NF, NU, NV2, NV3, NV4 } from '../gpu/TSLTypes';
import type { DagBuild, DagCluster } from './BuildDag';
import { type BuiltClusters, type ClusterStats, clusterize } from './Clusterize';
import {
  type BufOf,
  type V4W,
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
} from './Tsl';

export const VERT_WORDS = 6;
export const CLUSTER_WORDS = 8;
export const MESH_WORDS = 16;
/** N8-D1: parallel per-cluster DAG record — f32 ownErr + ownSphere(4) +
 *  parentErr + parentSphere(4); indexed by the SAME global clusterId as the
 *  8-word cluster record (the 8-word rec is full, so the DAG cut metadata
 *  lives in a sidecar buffer — F9: the cut kernel stays ≤10 storage bindings). */
export const DAG_WORDS = 10;
/** N8-D1: vertex layout fed to buildDag for a registry mesh — pos@0..2,
 *  nrm@3..5, uv@6..7, vdata@8..11 (UNPACKED to 0..1 floats so QEM can
 *  interpolate them). attachDag re-packs this back into VERT_WORDS. */
export const DAG_VERT_STRIDE = 12;
/** N8-D1: a root cluster's parentError is +∞ — stored as this finite sentinel
 *  so the GPU projection (errToPx) yields a huge value (> any τ) without inf/NaN. */
export const DAG_ROOT_PARENT_ERR = 1e30;
export const MAX_CLUSTER_TRIS = 128;
export const LOD_NONE = 0xffffffff;
/** N8-D2 Stage 2a: an evicted streaming-tile slot parks its mesh sphere here so
 *  kInstCull's frustum test always rejects it (belt + suspenders alongside the
 *  authoritative clusterCount=0, which makes lodSelectAndPush enqueue 0 chunks). */
export const TILE_EVICTED_FAR = 1e9;

export const MATERIAL_CLASS = {
  terrain: 0,
  rock: 1,
  bark: 2,
  deadwood: 3,
  leaf: 4,
  grass: 5,
  debris: 6,
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
export const MESH_FLAG_AGGREGATE = 2;
export const MESH_FLAG_CAST_SHADOWS = 4;
/** N8-D1: mesh's clusterStart/Count point at its FULL DAG cluster range (all
 *  levels) and lodNext = NONE — the cull applies the per-cluster screen-error
 *  cut (project(own)≤τ AND project(parent)>τ) instead of the discrete LOD chain */
export const MESH_FLAG_HASDAG = 8;
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
export interface InstanceStreamCPU {
  a: Float32Array;
  b: Float32Array;
}
/** GPU-resident instance records (scatter layers) — copied by kernel at build/flush */
export interface InstanceStreamGPU {
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
}

export interface MeshReport {
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

export function f32Bits(v: number): number {
  f32Scratch[0] = v;
  return u32Scratch[0] as number;
}
export function bitsF32(u: number): number {
  u32Scratch[0] = u >>> 0;
  return f32Scratch[0] as number;
}

/** IEEE f32 → f16 bits, round-to-nearest-even (matches WGSL pack semantics) */
export function f32ToF16(v: number): number {
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

export function f16ToF32(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const man = h & 0x3ff;
  if (exp === 0) return sign * man * 2 ** -24;
  if (exp === 31) return man ? NaN : sign * Infinity;
  return sign * (1 + man / 1024) * 2 ** (exp - 15);
}

function snorm16(v: number): number {
  const c = Math.max(-1, Math.min(1, v));
  const r = Math.round(c * 32767);
  return (r < 0 ? r + 65536 : r) & 0xffff;
}

/** octahedral-encode a (not necessarily unit) normal into snorm2x16 */
export function octEncode(nx: number, ny: number, nz: number): number {
  const a = Math.abs(nx) + Math.abs(ny) + Math.abs(nz);
  let ox = 0;
  let oy = 0;
  if (a > 1e-20) {
    ox = nx / a;
    oy = ny / a;
    if (nz < 0) {
      const tx = (1 - Math.abs(oy)) * (ox >= 0 ? 1 : -1);
      const ty = (1 - Math.abs(ox)) * (oy >= 0 ? 1 : -1);
      ox = tx;
      oy = ty;
    }
  }
  return (snorm16(ox) | (snorm16(oy) << 16)) >>> 0;
}

/** mirror of the GPU decode: unpack2x16snorm + oct → unit vector */
export function octDecode(packed: number): [number, number, number] {
  const sx = ((packed & 0xffff) << 16) >> 16;
  const sy = ((packed >>> 16) << 16) >> 16;
  let fx = Math.max(sx / 32767, -1);
  let fy = Math.max(sy / 32767, -1);
  const nz = 1 - Math.abs(fx) - Math.abs(fy);
  const t = Math.max(-nz, 0);
  fx += fx >= 0 ? -t : t;
  fy += fy >= 0 ? -t : t;
  const l = Math.hypot(fx, fy, nz) || 1;
  return [fx / l, fy / l, nz / l];
}

export interface VertexCPU {
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

export interface ClusterCPU {
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

export interface MeshCPU {
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
export function meshSphereFromClusters(
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

export interface DagRecordCPU {
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
  private clusterizeMs = 0;
  private built = false;

  // N8-D2 Stage 2a: streaming terrain tile POOL (D-N39). A fixed set of S slots,
  // each a constant-capacity byte block at poolBase+slot*cap, with a stable
  // heightfield mesh handle + identity instance. Streaming (re)loads a region
  // into a free slot (attachHeightDagTile — overwrites in place) and releases it
  // (evictHeightDagTile — zeroes the mesh draw, no tombstoning needed since the
  // cull is instance-driven). Bounded memory + cull regardless of field size;
  // O(1) alloc/free, no fragmentation (the Nanite "page" model).
  private tilePool: { slots: number; vertCap: number; triCap: number; clusterCap: number } | null = null;
  /** `vert` indexes the stride-1 hf vertex buffer (2e); tri/cluster index the shared buffers. */
  private tilePoolBase = { vert: 0, tri: 0, cluster: 0 };
  private tilePoolHandles: number[] = [];
  private tileFreeSlots: number[] = [];
  /** handle resident in each slot, or -1 if free (parallel to the free-stack) */
  private tileSlotOccupant: Int32Array | null = null;

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

  private vertsAttr!: StorageBufferAttribute;
  private hfVertsAttr!: StorageBufferAttribute;
  private idxAttr!: StorageBufferAttribute;
  private clusterAttr!: StorageBufferAttribute;
  private meshAttr!: StorageBufferAttribute;
  private instAttr!: StorageBufferAttribute;
  private instMeshAttr!: StorageBufferAttribute;
  private dagAttr!: StorageBufferAttribute;

  private caps!: { verts: number; tris: number; clusters: number; meshes: number; instances: number };
  /** N8-D2 Stage 2e: capacity (in verts = words) of the stride-1 terrain-DAG buffer. */
  private hfCap = 0;
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
      ...opts?.late,
    };
  }

  get meshCount(): number {
    return this.entries.length;
  }
  get clusterCount(): number {
    return this.clusterCursor;
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
    this.vertsArr = new Uint32Array(Math.max(1, this.caps.verts * VERT_WORDS));
    this.hfVertsArr = new Uint32Array(Math.max(1, this.hfCap));
    this.idxArr = new Uint32Array(Math.max(1, this.caps.tris * 3));
    this.clusterArr = new Uint32Array(Math.max(1, this.caps.clusters * CLUSTER_WORDS));
    this.meshArr = new Uint32Array(Math.max(1, this.caps.meshes * MESH_WORDS));
    this.instArr = new Float32Array(Math.max(8, this.caps.instances * 8));
    this.instMeshArr = new Uint32Array(Math.max(1, this.caps.instances));
    const dagLen = Math.max(DAG_WORDS, this.caps.clusters * DAG_WORDS);
    this.dagArr = new Float32Array(dagLen);

    this.vertsAttr = new StorageBufferAttribute(this.vertsArr, 1);
    this.hfVertsAttr = new StorageBufferAttribute(this.hfVertsArr, 1);
    this.idxAttr = new StorageBufferAttribute(this.idxArr, 1);
    this.clusterAttr = new StorageBufferAttribute(this.clusterArr, 1);
    this.meshAttr = new StorageBufferAttribute(this.meshArr, 1);
    this.instAttr = new StorageBufferAttribute(this.instArr, 4);
    this.instMeshAttr = new StorageBufferAttribute(this.instMeshArr, 1);
    this.dagAttr = new StorageBufferAttribute(this.dagArr, 1);

    const verts = sU32Views(this.vertsAttr, Math.max(1, this.caps.verts * VERT_WORDS));
    const hfVerts = sU32Views(this.hfVertsAttr, Math.max(1, this.hfCap));
    const idx = sU32Views(this.idxAttr, Math.max(1, this.caps.tris * 3));
    const clusters = sU32Views(this.clusterAttr, Math.max(1, this.caps.clusters * CLUSTER_WORDS));
    const meshes = sU32Views(this.meshAttr, Math.max(1, this.caps.meshes * MESH_WORDS));
    const inst = sVec4Views(this.instAttr, Math.max(2, this.caps.instances * 2));
    const instMesh = sU32Views(this.instMeshAttr, Math.max(1, this.caps.instances));
    const dag = sF32Views(this.dagAttr, dagLen);
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
    };

    // N8-D2 Stage 2a: claim the tile pool as ONE fixed region just past the
    // pre-build entries (its bytes were reserved via addLate in reserveTilePool).
    // Cursors advance past it ONCE; per-slot writes address poolBase+slot*cap and
    // never bump — so later post-build attaches (explicit DAGs) land after the
    // pool, and slots are reused in place across the session.
    if (this.tilePool) {
      // 2e: the pool's verts live in the stride-1 hf buffer (tilePoolBase.vert indexes
      // hfVertsArr); tris/clusters stay in the shared buffers.
      this.tilePoolBase = { vert: this.hfVertCursor, tri: this.triCursor, cluster: this.clusterCursor };
      this.hfVertCursor += this.tilePool.slots * this.tilePool.vertCap;
      this.triCursor += this.tilePool.slots * this.tilePool.triCap;
      this.clusterCursor += this.tilePool.slots * this.tilePool.clusterCap;
      this.tileFreeSlots = [];
      for (let s = this.tilePool.slots - 1; s >= 0; s--) this.tileFreeSlots.push(s);
      this.tileSlotOccupant = new Int32Array(this.tilePool.slots).fill(-1);
    }

    for (const e of this.entries) this.copyEntry(e);
    for (const s of this.cpuStreams) this.copyCpuStream(s);
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
      cArr[cb + 7] = ((dc.triCount & 0xff) | (CLUSTER_FLAG_DAG << 8) | (entry.handle << 16)) >>> 0;

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

    this.vertCursor += vCount;
    this.triCursor += tCount;
    this.clusterCursor += cCount;

    // -- repoint the mesh at its DAG range; retire the discrete LOD chain ------
    entry.clusterBase = cBase;
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
    this.pushRange(this.dagAttr, cBase * DAG_WORDS, cCount * DAG_WORDS);
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
   * N8-D2b: attach a terrain LOD DAG (buildHeightDag output) to a
   * registerHeightDag mesh. Mirrors attachDag — same cluster records + 10-float
   * DAG cut records + the SAME flat kClusterCull cut — but the vertex pool packs
   * TEXEL grid coords (word0 = gx | gz<<16, already clamped to [0,res-1]; words
   * 1-5 unused — height comes from heightTex, normal from normalTex on the GPU)
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
      cArr[cb + 7] = ((dc.triCount & 0xff) | (hfFlags << 8) | (entry.handle << 16)) >>> 0;

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
    this.tilePool = { slots, vertCap, triCap, clusterCap };
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

  /**
   * (Re)load a terrain tile (buildHeightDag output, global-texel gridVerts) into
   * a slot's FIXED byte range — overwrites the previous occupant in place; no
   * cursor growth. Mirrors attachHeightDag's pack but addresses poolBase+slot*cap
   * and is REUSABLE (no already-has-DAG guard). The slot's mesh record is
   * repointed at the new cluster range + sphere; partial uploads make the next
   * frame's kernels see it.
   */
  attachHeightDagTile(
    slot: number,
    build: { gridVerts: Uint32Array; indices: Uint32Array; clusters: DagCluster[] },
  ): void {
    if (!this.built) throw new Error('GeometryRegistry: attachHeightDagTile before build()');
    const pool = this.tilePool;
    if (!pool) throw new Error('GeometryRegistry: no tile pool reserved');
    if (slot < 0 || slot >= pool.slots) throw new Error(`GeometryRegistry: tile slot ${slot} out of range`);
    const handle = this.tilePoolHandles[slot] as number;
    const entry = this.entries[handle] as MeshEntry;
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
      cArr[cb] = f32Bits(dc.sx);
      cArr[cb + 1] = f32Bits(dc.sy);
      cArr[cb + 2] = f32Bits(dc.sz);
      cArr[cb + 3] = f32Bits(dc.sr);
      cArr[cb + 4] = octEncode(dc.cax, dc.cay, dc.caz);
      cArr[cb + 5] = f32Bits(dc.ccos);
      cArr[cb + 6] = tBase + dc.triStart;
      if (dc.triCount > MAX_CLUSTER_TRIS) throw new Error('GeometryRegistry: tile cluster exceeds tri cap');
      cArr[cb + 7] = ((dc.triCount & 0xff) | (hfFlags << 8) | (handle << 16)) >>> 0;

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

    entry.vertBase = vBase;
    entry.vertCount = vCount;
    entry.triBase = tBase;
    entry.triCount = tCount;
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
    this.pushRange(this.meshAttr, handle * MESH_WORDS, MESH_WORDS);

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
    entry.flags &= ~MESH_FLAG_HASDAG;
    entry.sphere = [TILE_EVICTED_FAR, TILE_EVICTED_FAR, TILE_EVICTED_FAR, 0];
    this.writeMeshRecord(entry);
    this.pushRange(this.meshAttr, handle * MESH_WORDS, MESH_WORDS);
    if (occ) occ[slot] = -1;
    this.tileFreeSlots.push(slot);
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
    return {
      verts,
      hfVerts,
      indices,
      clusters,
      meshTable,
      instances,
      total: verts + hfVerts + indices + clusters + meshTable + instances,
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
      instFirst: 0,
      instCount: 0,
      lodNext: LOD_NONE,
      lodDist: 0,
      sphere: [0, 0, 0, 0],
      uploaded: false,
    };
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
    m[b + 8] = f32Bits(e.hf?.originZ ?? 0);
    m[b + 9] = f32Bits(e.hf?.cellSize ?? 0);
    m[b + 10] = ((e.hf?.quadsX ?? 0) | ((e.hf?.quadsZ ?? 0) << 16)) >>> 0;
    m[b + 11] = f32Bits(e.swayPad);
    m[b + 12] = f32Bits(e.sphere[0]);
    m[b + 13] = f32Bits(e.sphere[1]);
    m[b + 14] = f32Bits(e.sphere[2]);
    m[b + 15] = f32Bits(e.sphere[3]);
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
