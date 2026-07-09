/**
 * Mid.ts — `nanMidRaster`, the world1 mid-band (2..swmax) consumer (task #76 de-über).
 *
 * world1 appends ONLY the 1-u32 tri id (payload = itemIdx<<CLUSTER_TRI_BITS | localTri) to
 * midQueue. This consumer re-reads that tri's 3 ALREADY-PROJECTED corners from projVertBuf
 * (nanProjectVerts wrote them this frame; they are never freed), reproduces world1's winding
 * (integer twice-area sign ⇒ the 1↔2 swap) + edge-setup (area2/bias/rw/rcpArea/bbox — cheap
 * per-tri setup, NOT rasterisation), and runs the ONE shared swScanline + the world1 election.
 * The corners it reconstructs are BIT-IDENTICAL to the fat 10-u32 record the old mid queue
 * carried (same projVertBuf slots via `canonVertSlot`; the in-queue tri already passed nearOK +
 * accept, so area2raw≠0 and the wound order = swap-to-positive-area) ⇒ same render. Collapsing
 * the record 10 u32 → 1 u32 is the ~10× midQueue shrink at zero re-projection cost.
 *
 * This is the ONLY place the mid band is rasterized — world1 no longer scanlines. INDIRECT,
 * 2-D dispatch (kMidArgs split), like nanSplatElect. The ?nomid disable-flag gates the APPEND
 * site inside world1's classify/route; this consumer draws whatever landed in the queue.
 */

import { Fn, float, uint } from 'three/tsl';
import { IndirectStorageBufferAttribute } from 'three/webgpu';
import type { NB, NF, NI, NU } from '../../gpu/TSLTypes';
import { CLUSTER_TRI_BITS, CLUSTER_TRI_MASK } from '../GeometryRegistry';
import {
  bcU2F,
  bcU2I,
  elemU,
  localX,
  maxI,
  minI,
  minU,
  returnIf,
  setIndirectDispatch,
  sU32Views,
  toF,
  toI,
  wgLinearDyn,
} from '../Tsl';
import { depthKey24 } from './VisBuffer';
import { canonVertSlot } from './Project';
import { CTX_STRIDE } from './ClusterCtx';
import { MID_CAP, MID_STRIDE } from './Queues';
import type { SwScanline } from './Scanline';

type U32Views = ReturnType<typeof sU32Views>;

interface ComputeKernel {
  setName(name: string): unknown;
}

/** Build `nanMidRaster` (null when the mid queue is absent). */
export function buildMid(p: {
  midQueueV: U32Views | null;
  midDrawAttr: IndirectStorageBufferAttribute | null;
  /** the projected-vert buffer nanProjectVerts filled + world1 read (READ here too). */
  projVertV: U32Views | null;
  /** the 35-word per-cluster ctx (READ for triStart+isHF, which canonVertSlot needs). */
  clusterCtxV: U32Views | null;
  /** gpu.indices — canonVertSlot reads it to dedup mesh corners (vi−vBase). */
  indices: Parameters<typeof canonVertSlot>[5];
  /** = MAX_CLUSTER_VERTS — the per-cluster unique-vert slot stride. */
  vertsPerCluster: number;
  swScanline: SwScanline;
  /** the shipped depth-keyed election bound to the vis buffers (VisBuffer.makeElect). */
  elect: (px: NU, cand: NU, idStore: NU) => void;
  width: number;
  height: number;
}): unknown | null {
  const {
    midQueueV,
    midDrawAttr,
    projVertV,
    clusterCtxV,
    indices,
    vertsPerCluster,
    swScanline,
    elect,
    width,
    height,
  } = p;
  if (!(midQueueV && midDrawAttr && projVertV && clusterCtxV)) return null;
  const kn = Fn(() => {
    // 2-D dispatch (nanMidArgs balanced split): linear record index uses the LIVE grid
    // width from the indirect args (wgLinearDyn), matching kMidArgs' x=ceil(wg/y) grid.
    // Queue reads are non-atomic: appends happened in world1 (a prior pass — RAW across a
    // pass boundary), this consumer never writes midQueue, so bind the read-only view only.
    const i = wgLinearDyn().mul(uint(64)).add(localX()).toVar();
    returnIf(
      i.greaterThanEqual(minU(elemU(midQueueV.ro, uint(0)), uint(MID_CAP))),
    );
    // 1-u32 record = the tri id (payload). Decode (itemIdx, localTri) — the SAME split the
    // resolve + world1 use — then read the 3 pre-projected corners from projVertBuf via
    // canonVertSlot (the identical deduped slot the classifier read before appending).
    // MID_STRIDE===1 ⇒ the record index is just `i`; emit it directly so the hot index
    // path carries no dead `*1u` (a general fallback keeps other strides correct).
    const recOff = (MID_STRIDE === 1 ? i : i.mul(uint(MID_STRIDE))) as unknown as NU;
    const pay = elemU(midQueueV.ro, uint(1).add(recOff)).toVar();
    const itemIdx = pay.shiftRight(uint(CLUSTER_TRI_BITS)).toVar();
    const localTri = pay.bitAnd(uint(CLUSTER_TRI_MASK)).toVar();
    const recCluster = itemIdx.mul(uint(vertsPerCluster)).toVar();
    // canonVertSlot needs the cluster's triStart + isHF (mesh vi−vBase vs terrain per-corner);
    // read from the SAME clusterCtx the projection + classifier used — CTX_U slots 0=isHF, 2=triStart.
    const cBase = itemIdx.mul(uint(CTX_STRIDE)).toVar();
    const triStart = elemU(clusterCtxV.ro, cBase.add(uint(2))).toVar();
    const isHF = elemU(clusterCtxV.ro, cBase).equal(uint(1));
    const rxi: NI[] = [];
    const ryi: NI[] = [];
    const rdz: NF[] = [];
    for (const v of [0, 1, 2] as const) {
      const rb = canonVertSlot(
        recCluster,
        triStart as unknown as NU,
        isHF as unknown as NB,
        localTri,
        v,
        indices,
      ).toVar();
      rxi[v] = bcU2I(elemU(projVertV.ro, rb)).toVar() as unknown as NI;
      ryi[v] = bcU2I(
        elemU(projVertV.ro, rb.add(uint(1))),
      ).toVar() as unknown as NI;
      rdz[v] = bcU2F(
        elemU(projVertV.ro, rb.add(uint(2))),
      ).toVar() as unknown as NF;
    }
    // reproduce world1's winding: the integer twice-area sign decides the 1↔2 swap so the
    // rasterised triangle has positive area (a two-sided back-face was flipped; a mid-queue
    // tri always passed accept ⇒ area2raw≠0 ⇒ the swap is unconditional on the sign).
    const area2raw = ryi[2]
      .sub(ryi[0])
      .mul(rxi[1].sub(rxi[0]))
      .sub(rxi[2].sub(rxi[0]).mul(ryi[1].sub(ryi[0])))
      .toVar();
    const flip = area2raw.lessThan(toI(0)).toVar();
    // A.1 — the re-wound twice-area is |area2raw| = flip ? −area2raw : area2raw (world1's
    // NEGATE form, NaniteRaster.ts:1418), NOT a fresh cross-product from the wound corners:
    // the SAME integer value, 2 fewer muls off the dependent chain the stall waits on.
    // A.4 — resolve area2/rcpArea HERE (before the winding selects) so area2raw dies
    // immediately and never coexists with the 6 wound-corner temps ⇒ smaller swap crest.
    const area2 = flip.select(area2raw.mul(toI(-1)), area2raw).toVar();
    const rcpArea = float(1).div(toF(area2 as unknown as NI)).toVar();
    // wound corners (positive twice-area). area2raw is already consumed ⇒ only the wound
    // set survives into the edge-setup below (SAME formulas as world1's inline edge-setup).
    const xi0 = rxi[0];
    const yi0 = ryi[0];
    const dz0 = rdz[0];
    const xi1 = flip.select(rxi[2], rxi[1]).toVar();
    const xi2 = flip.select(rxi[1], rxi[2]).toVar();
    const yi1 = flip.select(ryi[2], ryi[1]).toVar();
    const yi2 = flip.select(ryi[1], ryi[2]).toVar();
    const dz1 = flip.select(rdz[2], rdz[1]).toVar();
    const dz2 = flip.select(rdz[1], rdz[2]).toVar();
    const ex0 = yi1.sub(yi2);
    const ey0 = xi2.sub(xi1);
    const ex1 = yi2.sub(yi0);
    const ey1 = xi0.sub(xi2);
    const ex2 = yi0.sub(yi1);
    const ey2 = xi1.sub(xi0);
    const tlBias = (ex: NI, ey: NI): NI =>
      ex
        .lessThan(toI(0))
        .or(ex.equal(toI(0)).and(ey.greaterThan(toI(0))))
        .select(toI(0), toI(-1)) as unknown as NI;
    const bias0 = tlBias(ex0 as unknown as NI, ey0 as unknown as NI);
    const bias1 = tlBias(ex1 as unknown as NI, ey1 as unknown as NI);
    const bias2 = tlBias(ex2 as unknown as NI, ey2 as unknown as NI);
    const bbMinX = minI(xi0, minI(xi1, xi2)).shiftRight(uint(8));
    const bbMaxX = maxI(xi0, maxI(xi1, xi2)).shiftRight(uint(8));
    const bbMinY = minI(yi0, minI(yi1, yi2)).shiftRight(uint(8));
    const bbMaxY = maxI(yi0, maxI(yi1, yi2)).shiftRight(uint(8));
    const startX = maxI(toI(0), bbMinX as unknown as NI).toVar();
    const endX = minI(toI(width - 1), bbMaxX as unknown as NI).toVar();
    const startY = maxI(toI(0), bbMinY as unknown as NI).toVar();
    const endY = minI(toI(height - 1), bbMaxY as unknown as NI).toVar();
    const pcx = startX.mul(toI(256)).add(toI(128)).toVar();
    const pcy = startY.mul(toI(256)).add(toI(128)).toVar();
    const rw0 = pcy
      .sub(yi1)
      .mul(xi2.sub(xi1))
      .sub(pcx.sub(xi1).mul(yi2.sub(yi1)))
      .add(bias0)
      .toVar();
    const rw1 = pcy
      .sub(yi2)
      .mul(xi0.sub(xi2))
      .sub(pcx.sub(xi2).mul(yi0.sub(yi2)))
      .add(bias1)
      .toVar();
    const rw2 = pcy
      .sub(yi0)
      .mul(xi1.sub(xi0))
      .sub(pcx.sub(xi0).mul(yi1.sub(yi0)))
      .add(bias2)
      .toVar();
    const sx0 = ex0.mul(toI(256)).toVar();
    const sx1 = ex1.mul(toI(256)).toVar();
    const sx2 = ex2.mul(toI(256)).toVar();
    const sy0 = ey0.mul(toI(256)).toVar();
    const sy1 = ey1.mul(toI(256)).toVar();
    const sy2 = ey2.mul(toI(256)).toVar();
    const emit = (px: NU, cz: NF): void => {
      const cand = depthKey24(cz as unknown as NF)
        .shiftLeft(uint(8))
        .bitOr((pay as unknown as NU).bitAnd(uint(0xff)))
        .toVar();
      elect(px, cand as unknown as NU, pay as unknown as NU);
    };
    swScanline(
      rw0 as unknown as NI,
      rw1 as unknown as NI,
      rw2 as unknown as NI,
      sx0 as unknown as NI,
      sx1 as unknown as NI,
      sx2 as unknown as NI,
      sy0 as unknown as NI,
      sy1 as unknown as NI,
      sy2 as unknown as NI,
      bias0 as unknown as NI,
      bias1 as unknown as NI,
      bias2 as unknown as NI,
      dz0 as unknown as NF,
      dz1 as unknown as NF,
      dz2 as unknown as NF,
      rcpArea as unknown as NF,
      startX as unknown as NI,
      startY as unknown as NI,
      endX as unknown as NI,
      endY as unknown as NI,
      emit,
    );
  })().compute(MID_CAP, [64]); // grid overridden by setIndirectDispatch
  (kn as unknown as ComputeKernel).setName('nanMidRaster');
  setIndirectDispatch(kn, midDrawAttr);
  return kn;
}
