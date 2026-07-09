/**
 * VisBuffer.ts — the Option-C vis buffers, depth-key packing, the shipped
 * depth-keyed election, and the per-frame clear kernels.
 *
 *  • `makeVisBuffers` / `NaniteVisBuffers` — the three storage buffers (visDepthV /
 *    visPayloadV / visBV) created OUTSIDE the raster so the HZB can view depth
 *    without a builder cycle.
 *  • `depthKey16` / `depthKey24` — invert cz so nearer ⇒ larger key ⇒ wins atomicMax.
 *  • `elect` — the shipped (relect===1) depth-keyed atomicMax election into
 *    visPayloadV whose winner plain-stores the full id into visBV. This is the ONE
 *    election shared by the sub-pixel (Splat), mid-band (Mid) and HW consumers.
 *  • `buildVisClear` — kVisClear (full-screen) + kHwQueueClear (the counter-only tail
 *    reused by the scoped shadow clear). Both reset the queue + audit + scar counters.
 */

import {
  Fn,
  If,
  atomicMax,
  atomicStore,
  float,
  instanceIndex,
  uint,
} from 'three/tsl';
import { StorageBufferAttribute } from 'three/webgpu';
import type { NF, NU } from '../../gpu/TSLTypes';
import { markFragmentWritable } from '../../render/ThreePatches';
import { aLoadU, sU32Views } from '../Tsl';

type U32Views = ReturnType<typeof sU32Views>;

interface ComputeKernel {
  setName(name: string): unknown;
}

/** Option C vis buffers — created OUTSIDE the raster so the HZB (which the
 *  cull consumes) can view the depth buffer without a builder cycle */
export interface NaniteVisBuffers {
  depthAttr: StorageBufferAttribute;
  depthV: U32Views;
  payloadAttr: StorageBufferAttribute;
  payloadV: U32Views;
  /** PACKED-mode 2nd id buffer (idHi). visA(payloadV) holds (dk16<<16|idLo16), visB
   *  holds (dk16<<16|idHi16); both atomicMax'd so the nearest fragment wins BOTH and
   *  the depth+id can't desync (the race-free fix for the payload "branch-through-trunk"). */
  visBAttr: StorageBufferAttribute;
  visBV: U32Views;
}

export function makeVisBuffers(pixelCount: number): NaniteVisBuffers {
  const depthAttr = new StorageBufferAttribute(new Uint32Array(pixelCount), 1);
  depthAttr.name = 'nanVisDepth';
  markFragmentWritable(depthAttr);
  const payloadAttr = new StorageBufferAttribute(
    new Uint32Array(pixelCount),
    1,
  );
  payloadAttr.name = 'nanVisPayload';
  markFragmentWritable(payloadAttr);
  const visBAttr = new StorageBufferAttribute(new Uint32Array(pixelCount), 1);
  visBAttr.name = 'nanVisB';
  markFragmentWritable(visBAttr);
  return {
    depthAttr,
    depthV: sU32Views(depthAttr, pixelCount),
    payloadAttr,
    payloadV: sU32Views(payloadAttr, pixelCount),
    visBAttr,
    visBV: sU32Views(visBAttr, pixelCount),
  };
}

// depth → 16-bit key, INVERTED so nearer (smaller cz) ⇒ LARGER key ⇒ wins atomicMax.
export const depthKey16 = (cz: NF): NU =>
  uint(float(1).sub(cz).mul(65535).clamp(0, 65535)) as unknown as NU;

// PERF-VB4 world1: a 24-BIT key (8-bit id tiebreak below it). The full id is in the
// side buffer, so the election word only needs depth + a coarse tiebreak — 24 bits of
// depth (256× finer than the combined path's 16) makes the wp-reconstruction banding
// sub-pixel. The resolve/shadowHalf decode cz = 1 − (key>>8)/16777215; the HZB reads
// the top 16 bits (key>>16) which still decode as a valid coarse occluder.
export const depthKey24 = (cz: NF): NU =>
  uint(float(1).sub(cz).mul(16777215).clamp(0, 16777215)) as unknown as NU;

/** Bind the shipped (relect===1) depth-keyed election to a vis-buffer set. The
 *  relaxed atomicLoad guard → If(cand>prev) → atomicMax(payload) → winner-conditional
 *  atomicStore(visB) chain, emitted verbatim (byte-identical to every inline copy it
 *  replaces in the Splat/Mid/Hw consumers). */
export function makeElect(
  vis: NaniteVisBuffers,
): (px: NU, cand: NU, idStore: NU) => void {
  const visPayloadV = vis.payloadV;
  const visBV = vis.visBV;
  return (px: NU, cand: NU, idStore: NU): void => {
    const prevE = aLoadU(visPayloadV.atomic.element(px));
    If(cand.greaterThan(prevE), () => {
      const wonE = atomicMax(
        visPayloadV.atomic.element(px),
        cand,
      ) as unknown as NU;
      If(cand.greaterThan(wonE), () => {
        atomicStore(visBV.atomic.element(px), idStore);
      });
    });
  };
}

/** Parameters for the per-frame vis clear (kVisClear + the counter-only tail). */
export interface VisClearParams {
  pixelCount: number;
  vis: NaniteVisBuffers;
  hwQueueV: U32Views;
  auditV: U32Views;
  splatQueueV: U32Views | null;
  midQueueV: U32Views | null;
  /** singlePass/packed: clear payload/visB to 0 (atomicMax target) not the sentinel. */
  packedClear: boolean;
  /** skip the depthV sentinel clear on the single-pass path where nothing reads it. */
  skipDepthClear: boolean;
}

export function buildVisClear(p: VisClearParams): {
  kVisClear: unknown;
  kHwQueueClear: unknown;
} {
  const {
    pixelCount,
    vis,
    hwQueueV,
    auditV,
    splatQueueV,
    midQueueV,
    packedClear,
    skipDepthClear,
  } = p;
  const visDepthV = vis.depthV;
  const visPayloadV = vis.payloadV;
  const visBV = vis.visBV;

  // P9 (shadow strip clear): the tiny tail of kVisClear — ONLY the hw-queue +
  // audit counters. The strip-scoped shadow clear covers the depth texels itself
  // but MUST still reset this counter or the HW depth pass renders an
  // ever-growing stale triangle list (the P9 regression: +8ms moving).
  const kHwQueueClear = Fn(() => {
    If(instanceIndex.equal(uint(0)), () => {
      atomicStore(hwQueueV.atomic.element(0), uint(0));
      atomicStore(auditV.atomic.element(0), uint(0));
      atomicStore(auditV.atomic.element(1), uint(0));
    });
  })().compute(1, [1]);
  (kHwQueueClear as unknown as ComputeKernel).setName('nanHwQueueClear');

  const kVisClear = Fn(() => {
    If(instanceIndex.lessThan(uint(pixelCount)), () => {
      if (!skipDepthClear)
        atomicStore(visDepthV.atomic.element(instanceIndex), uint(0xffffffff));
      // packed/single-pass: payload(+visB) are atomicMax/side targets ⇒ clear to 0 (the
      // smallest, "no fragment"). legacy: payload keeps the 0xffffffff orphan sentinel.
      atomicStore(
        visPayloadV.atomic.element(instanceIndex),
        uint(packedClear ? 0 : 0xffffffff),
      );
      if (packedClear)
        atomicStore(visBV.atomic.element(instanceIndex), uint(0));
    });
    If(instanceIndex.equal(uint(0)), () => {
      atomicStore(hwQueueV.atomic.element(0), uint(0));
      atomicStore(auditV.atomic.element(0), uint(0));
      atomicStore(auditV.atomic.element(1), uint(0));
      // reset the sub-pixel + mid append counters (WAW-before the raster's atomicAdd).
      if (splatQueueV) atomicStore(splatQueueV.atomic.element(0), uint(0));
      if (midQueueV) atomicStore(midQueueV.atomic.element(0), uint(0));
    });
  })().compute(pixelCount, [256]);
  (kVisClear as unknown as ComputeKernel).setName('nanVisClear');

  return { kVisClear, kHwQueueClear };
}
