/**
 * Mid.ts — `nanMidRaster`, the world1 mid-band (2..swmax) consumer (task #76 de-über).
 *
 * world1 appended the pre-projected, pre-wound corners to midQueue; this recomputes the
 * edge-setup (area2/bias/rw/rcpArea/bbox — cheap per-tri setup, NOT rasterisation) and
 * runs the ONE shared swScanline + the world1 election. This is the ONLY place the mid
 * band is rasterized — world1 no longer scanlines, so no tri is rastered twice. INDIRECT,
 * 2-D dispatch (kMidArgs split), like nanSplatElect.
 *
 * The ?nomid disable-flag gates the APPEND site inside world1's classify/route (which is
 * left in NaniteRaster.ts this step); this consumer draws whatever landed in the queue.
 */

import { Fn, float, uint } from 'three/tsl';
import { IndirectStorageBufferAttribute } from 'three/webgpu';
import type { NF, NI, NU } from '../../gpu/TSLTypes';
import { DISPATCH_ROW } from '../NaniteCommon';
import {
  aLoadU,
  bcU2F,
  bcU2I,
  localX,
  maxI,
  minI,
  minU,
  returnIf,
  setIndirectDispatch,
  sU32Views,
  toF,
  toI,
  wgLinear,
} from '../Tsl';
import { depthKey24 } from './VisBuffer';
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
  swScanline: SwScanline;
  /** the shipped depth-keyed election bound to the vis buffers (VisBuffer.makeElect). */
  elect: (px: NU, cand: NU, idStore: NU) => void;
  width: number;
  height: number;
}): unknown | null {
  const { midQueueV, midDrawAttr, swScanline, elect, width, height } = p;
  if (!(midQueueV && midDrawAttr)) return null;
  const kn = Fn(() => {
    // 2-D dispatch (nanMidArgs split): linear record index = wgLinear·64 + localX.
    const i = wgLinear(DISPATCH_ROW).mul(uint(64)).add(localX()).toVar();
    returnIf(
      i.greaterThanEqual(
        minU(aLoadU(midQueueV.atomic.element(0)), uint(MID_CAP)),
      ),
    );
    const mb = uint(1).add(i.mul(uint(MID_STRIDE)));
    const xi0 = bcU2I(aLoadU(midQueueV.atomic.element(mb))).toVar();
    const yi0 = bcU2I(
      aLoadU(midQueueV.atomic.element(mb.add(uint(1)))),
    ).toVar();
    const xi1 = bcU2I(
      aLoadU(midQueueV.atomic.element(mb.add(uint(2)))),
    ).toVar();
    const yi1 = bcU2I(
      aLoadU(midQueueV.atomic.element(mb.add(uint(3)))),
    ).toVar();
    const xi2 = bcU2I(
      aLoadU(midQueueV.atomic.element(mb.add(uint(4)))),
    ).toVar();
    const yi2 = bcU2I(
      aLoadU(midQueueV.atomic.element(mb.add(uint(5)))),
    ).toVar();
    const dz0 = bcU2F(
      aLoadU(midQueueV.atomic.element(mb.add(uint(6)))),
    ).toVar();
    const dz1 = bcU2F(
      aLoadU(midQueueV.atomic.element(mb.add(uint(7)))),
    ).toVar();
    const dz2 = bcU2F(
      aLoadU(midQueueV.atomic.element(mb.add(uint(8)))),
    ).toVar();
    const pay = aLoadU(
      midQueueV.atomic.element(mb.add(uint(9))),
    ).toVar();
    // edge-setup from the wound corners (positive twice-area) — SAME formulas as
    // world1's inline edge-setup; cheap per-tri setup, not rasterisation.
    const area2 = yi2
      .sub(yi0)
      .mul(xi1.sub(xi0))
      .sub(xi2.sub(xi0).mul(yi1.sub(yi0)))
      .toVar();
    const rcpArea = float(1).div(toF(area2 as unknown as NI)).toVar();
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
