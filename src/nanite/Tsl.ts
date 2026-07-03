/**
 * Typed wrappers over the @types/three 0.184 TSL gaps the nanite code hits
 * (user note 2026-06-12: keep `as unknown` casts minimal — each gap gets ONE
 * cast HERE, call sites stay clean). Runtime behavior is identical to the
 * raw TSL calls; every workaround is documented against the typings gap.
 *
 * Gaps covered (NANITE-SPEC.md gotchas, N0):
 *  - storage() rejects 'uvec2'/'uvec4' type strings (runtime supports them)
 *  - uvec2() constructor rejects uint nodes
 *  - ranged Loop objects with custom names are untyped (runtime: LoopNode)
 *  - min/max reject uint nodes; float(uintNode) rejects — .toFloat() works
 *  - AtomicFunctionNode return lacks value-typed methods (THREE-NOTES)
 */

import {
  If,
  Loop,
  Return,
  atomicLoad,
  bitcast,
  int,
  localId,
  max,
  min,
  storage,
  texture,
  uint,
  packHalf2x16,
  uniform,
  uniformArray,
  unpackHalf2x16,
  unpackSnorm2x16,
  uvec2,
  workgroupId,
} from 'three/tsl';
import type { Matrix4, Texture, Vector3, Vector4 } from 'three';
import type { Renderer, StorageBufferNode } from 'three/webgpu';
import type { IndirectStorageBufferAttribute, StorageBufferAttribute } from 'three/webgpu';
import type { NB, NF, NI, NU, NV2, NV3, NV4 } from '../gpu/TSLTypes';

/** structural view of a uint vec2/vec4 storage element */
export interface UV2 {
  x: NU;
  y: NU;
}
export interface UV4 {
  x: NU;
  y: NU;
  z: NU;
  w: NU;
}

/** storage element accessor typed by element shape */
export interface BufOf<E> {
  element(i: NU | number): E;
}

type AtomicSlot = Parameters<typeof atomicLoad>[0];
export interface AtomicBuf {
  element(i: NU | number): AtomicSlot;
}

/** writable uvec2 element */
export interface UV2W extends UV2 {
  assign(v: UV2 | unknown): void;
}

/** uvec2 storage views ('uvec2' is valid WGSL; @types lacks the string) */
export function sUvec2(
  attr: StorageBufferAttribute,
  count: number,
): { rw: BufOf<UV2W>; ro: BufOf<UV2> } {
  const t = 'uvec2' as unknown as 'vec4';
  return {
    rw: storage(attr, t, count) as unknown as BufOf<UV2W>,
    ro: storage(attr, t, count).toReadOnly() as unknown as BufOf<UV2>,
  };
}

/** read-only uvec4 storage view */
export function sUvec4RO(attr: StorageBufferAttribute, count: number): BufOf<UV4> {
  return storage(attr, 'uvec4' as unknown as 'vec4', count).toReadOnly() as unknown as BufOf<UV4>;
}

/** uvec2 value from uint nodes (typings only accept numbers/float nodes) */
export function uv2(x: NU | number, y: NU | number): UV2 {
  return uvec2(x as unknown as number, y as unknown as number) as unknown as UV2;
}

/** uint min/max (typings restrict to float family; WGSL min/max are generic) */
export function minU(a: NU, b: NU): NU {
  return min(a as unknown as NF, b as unknown as NF) as unknown as NU;
}
export function maxU(a: NU, b: NU): NU {
  return max(a as unknown as NF, b as unknown as NF) as unknown as NU;
}

/** int min/max (same typings restriction as minU/maxU) */
export function minI(a: NI, b: NI): NI {
  return min(a as unknown as NF, b as unknown as NF) as unknown as NI;
}
export function maxI(a: NI, b: NI): NI {
  return max(a as unknown as NF, b as unknown as NF) as unknown as NI;
}

/** int constructor/conversion typed as NI (@types int() returns a bare node;
 *  WGSL i32(f32) SATURATES on overflow — callers may rely on that) */
export function toI(v: number | NF): NI {
  return int(v as unknown as number) as unknown as NI;
}

/** atomicLoad with a usable uint result (AtomicFunctionNode is method-less in @types) */
export function aLoadU(slot: AtomicSlot): NU {
  return atomicLoad(slot) as unknown as NU;
}

/** uint/int → float conversion (float(uintNode) is rejected; .toFloat() works) */
export function toF(v: NU | NI): NF {
  return (v as unknown as NF).toFloat();
}

/** ranged uint loop: for (i = start; i < end; i++) */
export function loopU(start: NU, end: NU, body: (i: NU) => void, step?: number): void {
  const params =
    step === undefined
      ? { start, end, type: 'uint', condition: '<' }
      : { start, end, type: 'uint', condition: '<', update: step };
  Loop(params as never, (lp: unknown) => {
    body((lp as { i: NU }).i);
  });
}

/** named ranged uint loop, EXCLUSIVE end — for nesting (unnamed loops both emit `i`,
 *  so an inner body's reference to the outer counter silently reads the inner one). */
export function loopUN(name: string, start: NU, end: NU, body: (i: NU) => void): void {
  Loop({ name, type: 'uint', start, end, condition: '<' } as never, (lp: unknown) => {
    body((lp as Record<string, NU>)[name] as NU);
  });
}

/**
 * named ranged int loop (inclusive end) — names keep nested WGSL loop vars
 * distinct (unnamed nested loops both emit `i`).
 */
export function loopI(
  name: string,
  start: NI,
  end: NI,
  body: (i: NI) => void,
): void {
  Loop(
    { name, type: 'int', start, end, condition: '<=' } as never,
    (lp: unknown) => {
      body((lp as Record<string, NI>)[name] as NI);
    },
  );
}

/** early-return guard: if (cond) return; (bare JS `return` builds nothing) */
export function returnIf(cond: NB): void {
  If(cond, () => {
    Return();
  });
}

/** f32 bits → uint (bitcast return is untyped in @types) */
export function bcF2U(f: NF): NU {
  return bitcast(f, 'uint') as unknown as NU;
}

/** uint bits → f32 */
export function bcU2F(u: NU): NF {
  return bitcast(u, 'float') as unknown as NF;
}

/** WGSL unpack2x16float (typings return a bare PackFloatNode) */
export function unpackHalfU(u: NU): NV2 {
  return unpackHalf2x16(u) as unknown as NV2;
}

/** WGSL pack2x16float */
export function packHalfU(v: NV2): NU {
  return packHalf2x16(v as unknown as Parameters<typeof packHalf2x16>[0]) as unknown as NU;
}

/** WGSL unpack2x16snorm */
export function unpackSnormU(u: NU): NV2 {
  return unpackSnorm2x16(u) as unknown as NV2;
}

/** read a u32 element from a (read-only) uint storage view */
export function elemU(buf: StorageBufferNode<'uint'>, i: NU | number): NU {
  return (buf as unknown as BufOf<NU>).element(i);
}

/** writable u32 element */
export interface UW {
  assign(v: NU | number | unknown): void;
}

/** writable u32 element of a read-write uint storage view */
export function elemUW(buf: StorageBufferNode<'uint'>, i: NU | number): UW {
  return (buf as unknown as BufOf<UW>).element(i);
}

/** writable vec4 element */
export interface V4W {
  assign(v: NV4 | unknown): void;
}

/** writable f32 element */
export interface FW {
  assign(v: NF | number): void;
}
/** readable + writable f32 element (kernels that reduce within ONE buffer
 *  must read and write through the SAME view — N0 same-scope law) */
export type FRW = NF & FW;

/** read-write + read-only f32 storage views */
export function sF32Views(
  attr: StorageBufferAttribute,
  count: number,
): { rw: BufOf<FRW>; ro: BufOf<NF> } {
  return {
    rw: storage(attr, 'float', count) as unknown as BufOf<FRW>,
    ro: storage(attr, 'float', count).toReadOnly() as unknown as BufOf<NF>,
  };
}

/** read-write + read-only vec4 storage views */
export function sVec4Views(
  attr: StorageBufferAttribute,
  count: number,
): { rw: BufOf<V4W>; ro: BufOf<NV4> } {
  return {
    rw: storage(attr, 'vec4', count) as unknown as BufOf<V4W>,
    ro: storage(attr, 'vec4', count).toReadOnly() as unknown as BufOf<NV4>,
  };
}

/** renderer.compute with the kernel-node cast (Fn().compute() returns an untyped node) */
export function dispatch(renderer: Renderer, kernel: unknown): void {
  renderer.compute(kernel as Parameters<Renderer['compute']>[0]);
}

/**
 * renderer.compute over an ARRAY of kernels → ONE command encoder, ONE compute
 * pass, ONE queue.submit (three's `finishCompute`), instead of one submit per
 * kernel. Safe for DEPENDENT chains (e.g. HZB mip levels, where level k reads
 * level k−1): within a compute pass WebGPU treats each dispatch as its own usage
 * scope and auto-synchronizes (UAV barrier) between them, so the dispatches are
 * serialized as-if run one-by-one — no manual barrier needed. Each kernel must
 * carry its own baked-in `.compute(count,[wg])` (the array shares no dispatchSize).
 */
export function dispatchBatch(renderer: Renderer, kernels: readonly unknown[]): void {
  renderer.compute(kernels as Parameters<Renderer['compute']>[0]);
}

/**
 * SUBMIT-COALESCE enabler — the ONE app-level mechanism that lets a BATCHED
 * (single-encoder, single-pass, single-submit) compute mix DIRECT and INDIRECT
 * dispatches while each indirect kernel KEEPS its own tight indirect dispatchSize.
 *
 * THE BLOCKER (three r184, verified): the array path forwards a SINGLE
 * `dispatchSize=null` to every node (Renderer.js:2773), so a batched indirect
 * kernel would fall back to its BAKED `.compute(count,[wg])` grid — for kTraverse/
 * kRaster* that is QRASTER_CAP(·MAX_CLUSTER_TRIS) threads = a ~1B-thread
 * over-dispatch regression. ComputeNode has no per-node indirect slot exposed to
 * the array path.
 *
 * THE FIX (NO node_modules edit): the backend's PER-NODE fallback at
 * WebGPUBackend.js:1433 is `dispatchSize = computeNode.dispatchSize ||
 * computeNode.count`, and lines 1439-1446 take the INDIRECT path whenever that
 * value `.isIndirectStorageBufferAttribute`. ComputeNode already stores a non-number
 * `count` arg into `.dispatchSize` (ComputeNode.js:295-302). So we simply ATTACH the
 * per-node indirect attribute onto `computeNode.dispatchSize`. In the batched call
 * the null outer arg makes the backend read THIS node's `.dispatchSize` (the attr,
 * truthy) and dispatch the tight indirect grid — per kernel, in ONE submit. Direct
 * kernels (dispatchSize null, count set) keep their baked `.compute(1,[1])` grid.
 * This touches only public ComputeNode fields + the documented `||` fallback; it is
 * isolated to src/ and breaks nothing if three changes the field (tsc/runtime would
 * surface it immediately).
 *
 * OVER-DISPATCH GUARD: the tagged kernel's baked grid is NEVER used while the attr
 * is attached (the `||` short-circuits on the truthy attr), so the 1B-thread fallback
 * cannot fire. Detaching (setIndirectDispatch(k, null)) restores the baked grid.
 */
interface IndirectTaggable {
  /** ComputeNode.dispatchSize — the slot the backend's `||` fallback reads first */
  dispatchSize: unknown;
}

/** Attach (or clear with null) a per-node INDIRECT dispatch attribute so a BATCHED
 *  run dispatches this kernel indirect at its tight size. Returns the kernel for
 *  chaining. Idempotent; safe to call once at build time (the attr is read fresh
 *  from the GPU buffer every dispatch). */
export function setIndirectDispatch(
  kernel: unknown,
  attr: IndirectStorageBufferAttribute | null,
): unknown {
  (kernel as IndirectTaggable).dispatchSize = attr;
  return kernel;
}

/** one batched compute element: either a plain (direct, baked-grid) kernel, or a
 *  kernel that has been tagged via setIndirectDispatch (indirect, per-node size). */
export type BatchKernel = unknown;

/**
 * dispatchBatch for a MIXED direct+indirect batch — ONE encoder, ONE compute pass,
 * ONE queue.submit, with each setIndirectDispatch-tagged kernel keeping its tight
 * indirect size (the enabler above). Identical submit semantics to dispatchBatch;
 * the only difference is callers may pass kernels carrying an indirect dispatchSize.
 * Inter-dispatch UAV auto-sync inside the pass preserves every producer→consumer
 * RAW/WAW exactly as if the kernels ran one-by-one (the array order IS the execution
 * order). Tag indirect kernels with setIndirectDispatch BEFORE adding them here.
 */
export function dispatchBatchMixed(renderer: Renderer, kernels: readonly BatchKernel[]): void {
  renderer.compute(kernels as Parameters<Renderer['compute']>[0]);
}

/** async storage-attribute readback (attr param is over-narrowed in @types) */
export function readBuffer(
  renderer: Renderer,
  attr: StorageBufferAttribute,
  byteOffset: number,
  byteLength: number,
): Promise<ArrayBuffer> {
  type Attr = Parameters<Renderer['getArrayBufferAsync']>[0];
  return renderer.getArrayBufferAsync(attr as unknown as Attr, null, byteOffset, byteLength);
}

/** local invocation index x (@types expose localId as a bare Node) */
export function localX(): NU {
  return (localId as unknown as { x: NU }).x;
}

/** linearized workgroup id for 2D-split indirect dispatches (row = 65535) */
export function wgLinear(rowSize: number): NU {
  const wid = workgroupId as unknown as { x: NU; y: NU };
  return wid.y.mul(uint(rowSize)).add(wid.x);
}

/** textureLoad .r at an integer texel coord (TextureNode.load is untyped) */
export function texLoadR(tex: Texture, x: NU, y: NU): NF {
  const t = texture(tex) as unknown as { load(c: unknown): { r: NF } };
  return t.load(uv2(x, y)).r;
}

/** mat4 uniform handle: TSL node with .mul(vec4) plus the CPU-side .value */
export type UniformMat4 = { mul(v: unknown): NV4; value: Matrix4 };
export function uniformMat4(m: Matrix4): UniformMat4 {
  return uniform(m) as unknown as UniformMat4;
}

/** vec3 uniform usable as an NV3 with CPU-side .value */
export type UniformV3 = NV3 & { value: Vector3 };
export function uniformV3(v: Vector3): UniformV3 {
  return uniform(v) as unknown as UniformV3;
}

/** float uniform usable as an NF with CPU-side .value */
export type UniformF = NF & { value: number };
export function uniformF(v: number): UniformF {
  return uniform(v) as unknown as UniformF;
}

/** uint uniform usable as an NU with CPU-side .value ('uint' rejected by @types) */
export type UniformU = NU & { value: number };
export function uniformU(v: number): UniformU {
  return uniform(v, 'uint' as unknown as 'float') as unknown as UniformU;
}

/** vec4 uniform array (frustum planes etc.) with CPU-side .array */
export type UniformArrV4 = { element(i: NU | NI | number): NV4; array: Vector4[] };
export function uniformArrV4(items: Vector4[]): UniformArrV4 {
  return uniformArray(items, 'vec4') as unknown as UniformArrV4;
}

/** renderer.compute with indirect dispatch args (param is untyped in @types) */
export function dispatchIndirect(
  renderer: Renderer,
  kernel: unknown,
  args: unknown,
): void {
  renderer.compute(
    kernel as Parameters<Renderer['compute']>[0],
    args as Parameters<Renderer['compute']>[1],
  );
}

/** read-write + read-only + atomic u32 views over one attribute */
export function sU32Views(
  attr: StorageBufferAttribute,
  count: number,
): { rw: StorageBufferNode<'uint'>; ro: StorageBufferNode<'uint'>; atomic: AtomicBuf } {
  // (cast: some TS servers resolve the storage() overload to 'struct')
  return {
    rw: storage(attr, 'uint', count) as unknown as StorageBufferNode<'uint'>,
    ro: storage(attr, 'uint', count).toReadOnly() as unknown as StorageBufferNode<'uint'>,
    atomic: storage(attr, 'uint', count).toAtomic() as unknown as AtomicBuf,
  };
}
