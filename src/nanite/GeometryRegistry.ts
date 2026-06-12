/**
 * GeometryRegistry (N1-C3, docs/NANITE.md) — the single entry point for solid
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
 *                 hf originX/originZ/cellSize f32-bits, gridW|gridH u16s,
 *                 swayPad f32-bits
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
  sU32Views,
  sVec4Views,
  toF,
  unpackHalfU,
  unpackSnormU,
} from './Tsl';

export const VERT_WORDS = 6;
export const CLUSTER_WORDS = 8;
export const MESH_WORDS = 12;
export const MAX_CLUSTER_TRIS = 128;
export const LOD_NONE = 0xffffffff;

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
/** cluster-record flag bits (byte 1 of word 7) */
export const CLUSTER_FLAG_HEIGHTFIELD = 1;

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
  /** window counts along x / z */
  gridW: number;
  gridH: number;
  /** quads per window side (8 → 128 tris/window); tris must fit MAX_CLUSTER_TRIS */
  winQuads: number;
  /** world units per quad */
  cellSize: number;
  originX: number;
  originZ: number;
  /** per-window height (min,max) pairs, length 2·gridW·gridH — bounds spheres */
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
  gridW: number;
  gridH: number;
  swayPad: number;
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
    gridW: w10 & 0xffff,
    gridH: w10 >>> 16,
    swayPad: bitsF32(table[b + 11] as number),
  };
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
  gridW: NU;
  gridH: NU;
  swayPad: NF;
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
    gridW: w10.bitAnd(uint(0xffff)),
    gridH: w10.shiftRight(uint(16)),
    swayPad: bcU2F(elemU(meshes, base.add(uint(11)))),
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
  hf?: { originX: number; originZ: number; cellSize: number; gridW: number; gridH: number };
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
  indices: StorageBufferNode<'uint'>;
  clusters: StorageBufferNode<'uint'>;
  meshes: StorageBufferNode<'uint'>;
  /** vec4 records, 2 per instance: [i·2]=A (xyz,scale), [i·2+1]=B (yaw,leanX,leanZ,idF) */
  instances: BufOf<NV4>;
  instanceMesh: StorageBufferNode<'uint'>;
}

export class GeometryRegistry {
  private readonly late: LateBudget;
  private readonly entries: MeshEntry[] = [];
  private readonly cpuStreams: CpuInstances[] = [];
  private readonly gpuStreams: GpuInstances[] = [];
  private vertCursor = 0;
  private triCursor = 0;
  private clusterCursor = 0;
  private instCursor = 0;
  private clusterizeMs = 0;
  private built = false;

  // backing arrays === attribute arrays (created at build; capacity-sized)
  private vertsArr!: Uint32Array;
  private idxArr!: Uint32Array;
  private clusterArr!: Uint32Array;
  private meshArr!: Uint32Array;
  private instArr!: Float32Array;
  private instMeshArr!: Uint32Array;

  private vertsAttr!: StorageBufferAttribute;
  private idxAttr!: StorageBufferAttribute;
  private clusterAttr!: StorageBufferAttribute;
  private meshAttr!: StorageBufferAttribute;
  private instAttr!: StorageBufferAttribute;
  private instMeshAttr!: StorageBufferAttribute;

  private caps!: { verts: number; tris: number; clusters: number; meshes: number; instances: number };
  private instRW!: BufOf<V4W>;
  private instMeshRW!: StorageBufferNode<'uint'>;

  /** read-only views for downstream kernels — valid after build() */
  gpu!: RegistryGpu;

  constructor(opts?: { late?: Partial<LateBudget> }) {
    this.late = {
      verts: 0,
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
      label: `${head.label}/lod`,
    });
    tail.lodNext = lod;
    tail.lodDist = switchDist;
    if (this.built && tail.uploaded) this.rewriteMeshRecord(tail);
    return lod;
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
    this.vertsArr = new Uint32Array(Math.max(1, this.caps.verts * VERT_WORDS));
    this.idxArr = new Uint32Array(Math.max(1, this.caps.tris * 3));
    this.clusterArr = new Uint32Array(Math.max(1, this.caps.clusters * CLUSTER_WORDS));
    this.meshArr = new Uint32Array(Math.max(1, this.caps.meshes * MESH_WORDS));
    this.instArr = new Float32Array(Math.max(8, this.caps.instances * 8));
    this.instMeshArr = new Uint32Array(Math.max(1, this.caps.instances));

    this.vertsAttr = new StorageBufferAttribute(this.vertsArr, 1);
    this.idxAttr = new StorageBufferAttribute(this.idxArr, 1);
    this.clusterAttr = new StorageBufferAttribute(this.clusterArr, 1);
    this.meshAttr = new StorageBufferAttribute(this.meshArr, 1);
    this.instAttr = new StorageBufferAttribute(this.instArr, 4);
    this.instMeshAttr = new StorageBufferAttribute(this.instMeshArr, 1);

    const verts = sU32Views(this.vertsAttr, Math.max(1, this.caps.verts * VERT_WORDS));
    const idx = sU32Views(this.idxAttr, Math.max(1, this.caps.tris * 3));
    const clusters = sU32Views(this.clusterAttr, Math.max(1, this.caps.clusters * CLUSTER_WORDS));
    const meshes = sU32Views(this.meshAttr, Math.max(1, this.caps.meshes * MESH_WORDS));
    const inst = sVec4Views(this.instAttr, Math.max(2, this.caps.instances * 2));
    const instMesh = sU32Views(this.instMeshAttr, Math.max(1, this.caps.instances));
    this.instRW = inst.rw;
    this.instMeshRW = instMesh.rw;
    this.gpu = {
      verts: verts.ro,
      indices: idx.ro,
      clusters: clusters.ro,
      meshes: meshes.ro,
      instances: inst.ro,
      instanceMesh: instMesh.ro,
    };

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

  /** post-build backing arrays + attributes (probe/validation use only) */
  debug(): {
    arrays: {
      verts: Uint32Array;
      indices: Uint32Array;
      clusters: Uint32Array;
      meshes: Uint32Array;
      instances: Float32Array;
      instanceMesh: Uint32Array;
    };
    attrs: {
      verts: StorageBufferAttribute;
      indices: StorageBufferAttribute;
      clusters: StorageBufferAttribute;
      meshes: StorageBufferAttribute;
      instances: StorageBufferAttribute;
      instanceMesh: StorageBufferAttribute;
    };
  } {
    if (!this.built) throw new Error('GeometryRegistry: debug() before build()');
    return {
      arrays: {
        verts: this.vertsArr,
        indices: this.idxArr,
        clusters: this.clusterArr,
        meshes: this.meshArr,
        instances: this.instArr,
        instanceMesh: this.instMeshArr,
      },
      attrs: {
        verts: this.vertsAttr,
        indices: this.idxAttr,
        clusters: this.clusterAttr,
        meshes: this.meshAttr,
        instances: this.instAttr,
        instanceMesh: this.instMeshAttr,
      },
    };
  }

  /** memory actually used (not capacity), bytes per blob */
  bytes(): BuildReport['bytes'] {
    const verts = this.vertCursor * VERT_WORDS * 4;
    const indices = this.triCursor * 3 * 4;
    const clusters = this.clusterCursor * CLUSTER_WORDS * 4;
    const meshTable = this.entries.length * MESH_WORDS * 4;
    const instances = this.instCursor * (32 + 4);
    return { verts, indices, clusters, meshTable, instances, total: verts + indices + clusters + meshTable + instances };
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
    return entry;
  }

  private packHeightfield(
    handle: number,
    src: HeightfieldSource,
    matClass: MaterialClassId,
    opts: RegisterOpts,
  ): MeshEntry {
    const trisPerWin = src.winQuads * src.winQuads * 2;
    if (trisPerWin > MAX_CLUSTER_TRIS) {
      throw new Error(`GeometryRegistry: heightfield window ${src.winQuads}² → ${trisPerWin} tris > ${MAX_CLUSTER_TRIS}`);
    }
    if (src.gridW > 0xffff || src.gridH > 0xffff) throw new Error('GeometryRegistry: heightfield grid exceeds u16');
    if (src.minMax.length !== src.gridW * src.gridH * 2) {
      throw new Error('GeometryRegistry: heightfield minMax length mismatch');
    }
    this.checkRoom(0, 0, src.gridW * src.gridH, 0);
    const entry = this.newEntry(handle, matClass, { ...opts, transformChannel: 'terrain' }, true, src.winQuads);
    entry.hf = {
      originX: src.originX,
      originZ: src.originZ,
      cellSize: src.cellSize,
      gridW: src.gridW,
      gridH: src.gridH,
    };
    const n = src.gridW * src.gridH;
    entry.clusterBase = this.clusterCursor;
    entry.clusterCount = n;
    this.clusterCursor += n;

    const win = src.winQuads * src.cellSize;
    const recs = new Uint32Array(n * CLUSTER_WORDS);
    for (let gz = 0; gz < src.gridH; gz++) {
      for (let gx = 0; gx < src.gridW; gx++) {
        const i = gz * src.gridW + gx;
        const mn = src.minMax[i * 2] as number;
        const mx = src.minMax[i * 2 + 1] as number;
        const b = i * CLUSTER_WORDS;
        recs[b] = f32Bits(src.originX + (gx + 0.5) * win);
        recs[b + 1] = f32Bits((mn + mx) / 2);
        recs[b + 2] = f32Bits(src.originZ + (gz + 0.5) * win);
        recs[b + 3] = f32Bits(Math.hypot(win / 2, (mx - mn) / 2, win / 2));
        recs[b + 4] = octEncode(0, 1, 0);
        recs[b + 5] = f32Bits(-1); // cone disabled — windows face anywhere
        recs[b + 6] = (gx | (gz << 16)) >>> 0;
        recs[b + 7] = (trisPerWin | (CLUSTER_FLAG_HEIGHTFIELD << 8) | (handle << 16)) >>> 0;
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
    m[b + 7] = f32Bits(e.hf?.originX ?? 0);
    m[b + 8] = f32Bits(e.hf?.originZ ?? 0);
    m[b + 9] = f32Bits(e.hf?.cellSize ?? 0);
    m[b + 10] = ((e.hf?.gridW ?? 0) | ((e.hf?.gridH ?? 0) << 16)) >>> 0;
    m[b + 11] = f32Bits(e.swayPad);
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
