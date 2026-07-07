/**
 * Scanline.ts — THE single fixed-point coverage loop (`swScanline`).
 *
 * Extracted VERBATIM from NaniteRaster's world1 `oldScanline` (task #76 de-über)
 * so depth/combined (inline) AND `nanMidRaster` (the de-übered world1 mid-band
 * consumer) share ONE rasterizer — no duplicate rasterisation across shaders.
 *
 * Given the pre-computed edge-setup at the start pixel (rw), the per-pixel/per-row
 * steps (sx/sy), the top-left biases, the corner depths + rcpArea, and the bbox, it
 * walks every pixel and for each COVERED pixel with in-range depth calls emit(px, cz).
 * The per-mode write (depth atomicMin / combined packed / world1 election) lives
 * entirely in the caller's emit. The cheap per-tri edge-setup (bias/rw/rcpArea) is
 * derived by each caller — that's setup, not rasterisation, so its trivial recompute
 * in the consumer is fine.
 */

import { If, uint } from 'three/tsl';
import type { NF, NI, NU } from '../../gpu/TSLTypes';
import type { NaniteCam } from '../NaniteCommon';
import { loopI, toF, toI } from '../Tsl';

/** The shared fixed-point coverage loop signature. */
export type SwScanline = (
  rw0: NI, rw1: NI, rw2: NI,
  sx0: NI, sx1: NI, sx2: NI,
  sy0: NI, sy1: NI, sy2: NI,
  bias0: NI, bias1: NI, bias2: NI,
  dz0: NF, dz1: NF, dz2: NF,
  rcpArea: NF,
  startX: NI, startY: NI, endX: NI, endY: NI,
  emit: (px: NU, cz: NF) => void,
) => void;

/** Build the shared coverage loop bound to a camera (only reads `cam.uW`). */
export function makeScanline(cam: NaniteCam): SwScanline {
  return (
    rw0, rw1, rw2,
    sx0, sx1, sx2,
    sy0, sy1, sy2,
    bias0, bias1, bias2,
    dz0, dz1, dz2,
    rcpArea,
    startX, startY, endX, endY,
    emit,
  ): void => {
    const r0 = rw0.toVar();
    const r1 = rw1.toVar();
    const r2 = rw2.toVar();
    loopI('sy', startY as unknown as NI, endY as unknown as NI, (y) => {
      const cw0 = r0.toVar();
      const cw1 = r1.toVar();
      const cw2 = r2.toVar();
      const rowBase = uint(y).mul(uint(cam.uW)).toVar();
      loopI('sx', startX as unknown as NI, endX as unknown as NI, (x) => {
        If(
          cw0
            .greaterThanEqual(toI(0))
            .and(cw1.greaterThanEqual(toI(0)))
            .and(cw2.greaterThanEqual(toI(0))),
          () => {
            // UNBIASED weights (the top-left −1 biases belong to COVERAGE only; folding
            // them into the depth weights biases far slivers nearer — see the N4-C0 note).
            const uw0 = cw0.sub(bias0).toVar();
            const uw1 = cw1.sub(bias1).toVar();
            const uw2 = cw2.sub(bias2).toVar();
            const cz = toF(uw0 as unknown as NI)
              .mul(dz0)
              .add(toF(uw1 as unknown as NI).mul(dz1))
              .add(toF(uw2 as unknown as NI).mul(dz2))
              .mul(rcpArea)
              .toVar();
            If(cz.greaterThanEqual(0).and(cz.lessThanEqual(1)), () => {
              const px = rowBase.add(uint(x));
              emit(px, cz as unknown as NF);
            });
          },
        );
        cw0.addAssign(sx0);
        cw1.addAssign(sx1);
        cw2.addAssign(sx2);
      });
      r0.addAssign(sy0);
      r1.addAssign(sy1);
      r2.addAssign(sy2);
    });
  };
}
