/**
 * Queues.ts — the three deferred-raster work queues and their indirect-args kernels.
 *
 *  • hwQueue — [0]=atomic count, then (payload, instId) pairs. The ?scar counters and
 *    the ?trihzb prev-frame HZB mirror are FOLDED into its tail (the WebGPU 10-storage-
 *    buffer ceiling forbids separate buffers on the already-full world1 stage).
 *  • splatQueue — the sub-pixel append list (px, cand, payload) consumed by Splat.
 *  • midQueue — the 2..swmax pre-projected corner records consumed by Mid.
 *  • kSplatArgs / kMidArgs — the indirect dispatch args (1-D for splat; 2-D split for
 *    mid, whose count can exceed the 65535-workgroup 1-D ceiling → the field blinks).
 */

import { Fn, uint } from 'three/tsl';
import {
  IndirectStorageBufferAttribute,
  StorageBufferAttribute,
} from 'three/webgpu';
import { DISPATCH_ROW } from '../NaniteCommon';
import { aLoadU, maxU, minU, sU32Views } from '../Tsl';

type U32Views = ReturnType<typeof sU32Views>;

interface ComputeKernel {
  setName(name: string): unknown;
}

// N9-C0: leaf needles are long, thin, near-camera tris — their bbox exceeds the
// SW raster's 16 px i32-safe limit, so a dense crown routes hundreds of thousands
// of them to the HW vertex-pulling path. The old 262k cap overflowed (clamp →
// dropped tris → black holes in foliage; measured 876k hwTris in a dense stand).
// Sized for that load; the queue only costs memory + the HW pass only pays for the
// tris actually present. The real reduction (halving the dup, shedding far crowns)
// is the two-sided raster + the aggregate DAG (N9-C2).
export const HW_CAP = 2_097_152;

// SPLAT/MID caps = 65535·64 (max 1-D indirect grid; ceil(N/64) ≤ 65535). ~50 MB /
// ~167 MB respectively. If the RAW append count exceeds a cap the extra fragments are
// DROPPED (overflow → holes) and WHICH subset lands is the non-deterministic atomic
// order ⇒ the field blinks; midQueue goes 2-D dispatch to cover past the 1-D ceiling.
export const SPLAT_CAP = 4_194_240;
export const MID_CAP = 4_194_240;
// Mid record = 10 u32 of the ALREADY-PROJECTED, ALREADY-WOUND corners (xi0,yi0,xi1,yi1,
// xi2,yi2 i32 bits; dz0,dz1,dz2 f32 bits; payload). Consumer recomputes edge-setup.
export const MID_STRIDE = 10;

/** Tail-fold pyramid level descriptor for the ?trihzb mirror (sizes the hwQueue tail). */
export interface TriHzbLevel {
  offset: number;
  w: number;
  h: number;
}

export interface Queues {
  // hwQueue (+ scar/trihzb tail folds)
  hwQueueAttr: StorageBufferAttribute;
  hwQueueV: U32Views;
  hwDrawAttr: IndirectStorageBufferAttribute;
  hwDrawBuf: U32Views['rw'];
  /** ?scar counters live at hwQueue[SCAR_BASE..+4). scarAttr aliases hwQueueAttr. */
  SCAR_BASE: number;
  scarEl: (i: number) => ReturnType<U32Views['atomic']['element']>;
  /** ?trihzb prev-frame HZB mirror at hwQueue[TRIHZB_BASE..+triTailN). */
  TRIHZB_BASE: number;
  triTailN: number;
  // splat queue (null when !splatElect)
  splatQueueAttr: StorageBufferAttribute | null;
  splatQueueV: U32Views | null;
  splatDrawAttr: IndirectStorageBufferAttribute | null;
  splatDrawBuf: U32Views['rw'] | null;
  // mid queue (null when !splatElect)
  midQueueAttr: StorageBufferAttribute | null;
  midQueueV: U32Views | null;
  midDrawAttr: IndirectStorageBufferAttribute | null;
  midDrawBuf: U32Views['rw'] | null;
  // indirect-args kernels (null when !splatElect)
  kSplatArgs: unknown | null;
  kMidArgs: unknown | null;
}

export function buildQueues(p: {
  /** = singlePass: the splat/mid queues + their args only exist on the world1 path. */
  splatElect: boolean;
  /** the sliced ?trihzb levels (or null) — sizes the hwQueue mirror tail. */
  triLvls: TriHzbLevel[] | null;
}): Queues {
  const { splatElect, triLvls } = p;

  // hwQueue: [0] = atomic count, then (payload, instId) pairs.
  // ⚠ SCAR FOLD (?scar=1): world1's compute stage is ALREADY at the WebGPU 10-storage-
  // buffer per-stage ceiling, so a SEPARATE scar buffer makes it 11 (validation error →
  // invalid pipeline → black frame, no counters). The scar counters (4 u32) are therefore
  // CARVED INTO THE TAIL of this already-bound hwQueue buffer at [SCAR_BASE..+4) — they
  // never overlap the queue's [0 .. 1+HW_CAP*2) range, so no new binding, stays at 10.
  const SCAR_BASE = 1 + HW_CAP * 2;
  // W2 ?trihzb TAIL FOLD (same binding-budget law as scar above): the per-tri
  // occlusion test needs the prev-frame HZB, but binding the pyramid buffer in
  // world1 is an 11th storage buffer (measured: validation error, dead pipeline).
  // So a tiny copy kernel (own 2-buffer pipeline, runs right after the pyramid
  // build) mirrors ONE fixed level into this buffer's tail at [TRIHZB_BASE..).
  // Tail is pre-filled with 1.0f bits (far) so frame-0 rejects nothing; clears
  // never touch it ([0] + scar slots only).
  const TRIHZB_BASE = SCAR_BASE + 4;
  const triTailN = (triLvls ?? []).reduce((a, l) => a + l.w * l.h, 0);
  const hwQueueInit = new Uint32Array(TRIHZB_BASE + triTailN);
  if (triTailN > 0) hwQueueInit.fill(0x3f800000, TRIHZB_BASE); // 1.0f = far
  const hwQueueAttr = new StorageBufferAttribute(hwQueueInit, 1);
  hwQueueAttr.name = 'nanHwQueue';
  const hwQueueV = sU32Views(hwQueueAttr, TRIHZB_BASE + triTailN);
  const hwDrawAttr = new IndirectStorageBufferAttribute(new Uint32Array(4), 4);
  hwDrawAttr.name = 'nanHwDraw';
  const hwDrawBuf = sU32Views(
    hwDrawAttr as unknown as StorageBufferAttribute,
    4,
  ).rw;

  // scar atomic element [i] lives at hwQueue slot SCAR_BASE+i. Returns the atomic ref
  // (what atomicStore/atomicAdd take), so call sites read like the old scarV.atomic.element.
  const scarEl = (i: number): ReturnType<U32Views['atomic']['element']> =>
    hwQueueV.atomic.element(uint(SCAR_BASE + i));

  // ─── Splat-election (task #76) — sub-pixel append list ───────────────────────────
  const splatQueueAttr = splatElect
    ? new StorageBufferAttribute(new Uint32Array(1 + SPLAT_CAP * 3), 1)
    : null;
  if (splatQueueAttr) splatQueueAttr.name = 'nanSplatQueue';
  const splatQueueV = splatQueueAttr
    ? sU32Views(splatQueueAttr, 1 + SPLAT_CAP * 3)
    : null;
  const splatDrawAttr = splatElect
    ? new IndirectStorageBufferAttribute(new Uint32Array(3), 3)
    : null;
  if (splatDrawAttr) splatDrawAttr.name = 'nanSplatDraw';
  const splatDrawBuf = splatDrawAttr
    ? sU32Views(splatDrawAttr as unknown as StorageBufferAttribute, 3).rw
    : null;

  // ─── Mid-band raster queue (task #76 de-über) ────────────────────────────────────
  const midQueueAttr = splatElect
    ? new StorageBufferAttribute(new Uint32Array(1 + MID_CAP * MID_STRIDE), 1)
    : null;
  if (midQueueAttr) midQueueAttr.name = 'nanMidQueue';
  const midQueueV = midQueueAttr
    ? sU32Views(midQueueAttr, 1 + MID_CAP * MID_STRIDE)
    : null;
  const midDrawAttr = splatElect
    ? new IndirectStorageBufferAttribute(new Uint32Array(3), 3)
    : null;
  if (midDrawAttr) midDrawAttr.name = 'nanMidDraw';
  const midDrawBuf = midDrawAttr
    ? sU32Views(midDrawAttr as unknown as StorageBufferAttribute, 3).rw
    : null;

  // kSplatArgs: ceil(min(count,CAP)/64) workgroups, compute dispatch-indirect [wgX,1,1].
  const kSplatArgs =
    splatElect && splatQueueV && splatDrawBuf
      ? (() => {
          const kn = Fn(() => {
            const n = minU(
              aLoadU(splatQueueV.atomic.element(0)),
              uint(SPLAT_CAP),
            );
            splatDrawBuf.element(0).assign(n.add(uint(63)).div(uint(64)));
            splatDrawBuf.element(1).assign(uint(1));
            splatDrawBuf.element(2).assign(uint(1));
          })().compute(1, [1]);
          (kn as unknown as ComputeKernel).setName('nanSplatArgs');
          return kn;
        })()
      : null;

  // kMidArgs: 2-D split (like the cull's split2D) — mid count can exceed 4.19M in dense
  // forests, and a 1-D grid caps at 65535 workgroups (=4.19M threads) — records beyond
  // that never dispatch, and WHICH ones do is the non-deterministic atomic append order
  // ⇒ the mid field blinks. Spread workgroups across x(≤65535)·y.
  const kMidArgs =
    splatElect && midQueueV && midDrawBuf
      ? (() => {
          const kn = Fn(() => {
            const n = minU(aLoadU(midQueueV.atomic.element(0)), uint(MID_CAP));
            const wg = n.add(uint(63)).div(uint(64));
            midDrawBuf.element(0).assign(minU(wg, uint(DISPATCH_ROW)));
            midDrawBuf
              .element(1)
              .assign(
                maxU(
                  wg.add(uint(DISPATCH_ROW - 1)).div(uint(DISPATCH_ROW)),
                  uint(1),
                ),
              );
            midDrawBuf.element(2).assign(uint(1));
          })().compute(1, [1]);
          (kn as unknown as ComputeKernel).setName('nanMidArgs');
          return kn;
        })()
      : null;

  return {
    hwQueueAttr,
    hwQueueV,
    hwDrawAttr,
    hwDrawBuf,
    SCAR_BASE,
    scarEl,
    TRIHZB_BASE,
    triTailN,
    splatQueueAttr,
    splatQueueV,
    splatDrawAttr,
    splatDrawBuf,
    midQueueAttr,
    midQueueV,
    midDrawAttr,
    midDrawBuf,
    kSplatArgs,
    kMidArgs,
  };
}
