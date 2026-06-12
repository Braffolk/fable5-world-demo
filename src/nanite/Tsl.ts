/**
 * Typed wrappers over the @types/three 0.184 TSL gaps the nanite code hits
 * (user note 2026-06-12: keep `as unknown` casts minimal — each gap gets ONE
 * cast HERE, call sites stay clean). Runtime behavior is identical to the
 * raw TSL calls; every workaround is documented against the typings gap.
 *
 * Gaps covered (NANITE.md gotchas, N0):
 *  - storage() rejects 'uvec2'/'uvec4' type strings (runtime supports them)
 *  - uvec2() constructor rejects uint nodes
 *  - ranged Loop objects with custom names are untyped (runtime: LoopNode)
 *  - min/max reject uint nodes; float(uintNode) rejects — .toFloat() works
 *  - AtomicFunctionNode return lacks value-typed methods (THREE-NOTES)
 */

import { If, Loop, Return, atomicLoad, max, min, storage, uvec2 } from 'three/tsl';
import type { StorageBufferNode } from 'three/webgpu';
import type { StorageBufferAttribute } from 'three/webgpu';
import type { NB, NF, NI, NU } from '../gpu/TSLTypes';

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

/** atomicLoad with a usable uint result (AtomicFunctionNode is method-less in @types) */
export function aLoadU(slot: AtomicSlot): NU {
  return atomicLoad(slot) as unknown as NU;
}

/** uint/int → float conversion (float(uintNode) is rejected; .toFloat() works) */
export function toF(v: NU | NI): NF {
  return (v as unknown as NF).toFloat();
}

/** ranged uint loop: for (i = start; i < end; i++) */
export function loopU(start: NU, end: NU, body: (i: NU) => void): void {
  Loop({ start, end, type: 'uint', condition: '<' } as never, (lp: unknown) => {
    body((lp as { i: NU }).i);
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
