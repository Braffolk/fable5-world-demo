/**
 * Queues.ts — the three deferred-raster work queues and their indirect-args kernels.
 *
 *  • hwQueue — [0]=atomic count, then (payload, instId) pairs.
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
// CENSUS 2026-07-09 POST crown-LOD ladder (docs/tasks/2026-07-09/census-post-ladder.json,
// 898-sample flythrough): peak hwTris = 262,186 at the dense-forest look-across
// (t=0.375, ?cam=-596.4,298.9,998.6,2.5760,-0.0060) — DOWN from the pre-ladder 480,178.
// KEPT at 1M: the binding constraint is NOT the flythrough peak but the historical 876k
// dense-stand load that overflowed the old 262k cap (black holes in foliage) — 1.5×262k
// ≈ 393k would sit BELOW that documented black-hole trigger. 1M = 1.2× the 876k load;
// reclaiming to 1.5× the fresh peak saves only ~4 MiB at real regression risk. Held.
export const HW_CAP = 1_048_576;

// SPLAT/MID overflow semantics: if the RAW append count exceeds a cap the extra
// fragments are DROPPED (overflow → holes) and WHICH subset lands is the
// non-deterministic atomic order ⇒ the field blinks; midQueue goes 2-D dispatch
// to cover past the 65535·64 1-D indirect-grid ceiling (SPLAT stays 1-D: ceil(cap/64) ≤ 65535).
// CENSUS 2026-07-09 POST crown-LOD ladder (docs/tasks/2026-07-09/census-post-ladder.json):
// peak splatFrags = 3,995 @ t=0.238 (?cam=157.7,319.2,1386.8,2.0355,-0.0591) — DOWN from
// the pre-ladder 27,668 (coarser far crowns emit far fewer sub-px tris). KEPT at 131,072
// (now 33× the fresh peak): splat overflow = visible holes and a slot is only 12 B, so
// the whole queue is ~1.5 MB — right-sizing to 1.5× would reclaim ~1.3 MB (noise) at real
// hole-risk on a moving cam. Fresh peak just confirms it is amply sized.
export const SPLAT_CAP = 131_072;
// MID goes 2-D dispatch + a 1-u32 record. The old 41.9M cap was sized to the pre-
// crown-LOD overdraw pathology (~32-35M mid tris in dense forest).
// CENSUS 2026-07-09 POST crown-LOD ladder (docs/tasks/2026-07-09/census-post-ladder.json,
// 898-sample flythrough): peak midTris = 4,774,720 at the dense-forest look-across
// (t=0.373, ?cam=-582.1,302.4,1006.1,2.5692,-0.0077) — DOWN from the pre-ladder
// 13,328,568 (−64%, the crown 4→6-rung ladder). 8Mi = 8,388,608 = 1.76× the fresh peak
// (with headroom for moving-cam undersampling; overflow drops a non-deterministic subset
// ⇒ the mid field blinks). (was 20_971_520; 4 B/record ⇒ 84 → 33.6 MB, −50.3 MB)
export const MID_CAP = 8_388_608;
// Mid record = 1 u32: the tri id (payload = itemIdx<<CLUSTER_TRI_BITS | localTri). The
// consumer (nanMidRaster) re-reads the tri's 3 already-projected corners from projVertBuf
// (they were projected once by nanProjectVerts and never freed) and re-derives the winding
// + edge-setup — so the fat 10-u32 corner record (xi0..yi2 + dz0..2 + payload) collapses to
// the id alone: ~10× smaller midQueue at zero projection cost + bit-identical render.
export const MID_STRIDE = 1;

export interface Queues {
  // hwQueue
  hwQueueAttr: StorageBufferAttribute;
  hwQueueV: U32Views;
  hwDrawAttr: IndirectStorageBufferAttribute;
  hwDrawBuf: U32Views['rw'];
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
}): Queues {
  const { splatElect } = p;

  // hwQueue: [0] = atomic count, then (payload, instId) pairs.
  const hwQueueLen = 1 + HW_CAP * 2;
  const hwQueueAttr = new StorageBufferAttribute(new Uint32Array(hwQueueLen), 1);
  hwQueueAttr.name = 'nanHwQueue';
  const hwQueueV = sU32Views(hwQueueAttr, hwQueueLen);
  const hwDrawAttr = new IndirectStorageBufferAttribute(new Uint32Array(4), 4);
  hwDrawAttr.name = 'nanHwDraw';
  const hwDrawBuf = sU32Views(
    hwDrawAttr as unknown as StorageBufferAttribute,
    4,
  ).rw;

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
            // Balanced grid: y=ceil(wg/65535), x=ceil(wg/y) ⇒ overshoot < y workgroups.
            // The old min(wg,65535)×ceil grid dispatched up to ~2× ghost threads once
            // wg > 65535 (mid > 4.19M tris — exactly the dense-forest worst pose), each
            // paying the full prologue + an atomic count-load before returning. The
            // consumer must index with the LIVE grid width (wgLinearDyn), not 65535.
            const gy = maxU(
              wg.add(uint(DISPATCH_ROW - 1)).div(uint(DISPATCH_ROW)),
              uint(1),
            );
            const gx = wg.add(gy).sub(uint(1)).div(gy);
            midDrawBuf.element(0).assign(gx);
            midDrawBuf.element(1).assign(gy);
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
