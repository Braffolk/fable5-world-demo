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
 *
 * MID-ONLY register-crest cuts (task #76 nanMidRaster levers A.2/A.3) live behind
 * `MidScanOpts` build-time branches: the world1/depth/combined instance passes NO
 * opts and gets the DEFAULT loop below UNCHANGED (byte-identical values). Only
 * nanMidRaster's own instance (`makeScanline(cam, {…})`) takes the optimized paths.
 * The DEFAULT loop is thus provably unperturbed for world1.
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

/** MID-ONLY build-time crest cuts. Omitted (world1/depth/combined) ⇒ the default loop. */
export interface MidScanOpts {
  /** A.3 — hold the 3 top-left biases as one packed word instead of 3 registers.
   *  bias∈{0,−1} ⇒ `cw − bias == cw + ((biasP>>i)&1)`, so the unpack is bit-identical.
   *  −2 loop-carried registers, values unchanged. */
  packBias?: boolean;
  /** A.2 (?middz) — incremental depth: hold {z, dzdx, dzdy} and step `z += dzdx` per
   *  pixel / `+= dzdy` per row instead of the per-pixel barycentric recompute. cz is
   *  affine in the incrementally-stepped cw, so this is mathematically identical, but
   *  float accumulation can differ in the depthKey24 LSB (MEDIUM parity risk) ⇒ GATED,
   *  default OFF. When on, bias is not needed in the loop (folded into the setup z). */
  incDepth?: boolean;
}

/** Build the shared coverage loop bound to a camera (only reads `cam.uW`).
 *  Pass `opts` ONLY for nanMidRaster's own instance — world1/depth/combined omit it
 *  and get the byte-identical default loop. */
export function makeScanline(cam: NaniteCam, opts?: MidScanOpts): SwScanline {
  const packBias = opts?.packBias ?? false;
  const incDepth = opts?.incDepth ?? false;
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

    // ── MID-ONLY A.2 (?middz) incremental depth. Default builds never enter here.
    if (incDepth) {
      // dz/dx and dz/dy are constant (cz affine in cw); computed ONCE at setup.
      const dzdx = toF(sx0)
        .mul(dz0)
        .add(toF(sx1).mul(dz1))
        .add(toF(sx2).mul(dz2))
        .mul(rcpArea)
        .toVar();
      const dzdy = toF(sy0)
        .mul(dz0)
        .add(toF(sy1).mul(dz1))
        .add(toF(sy2).mul(dz2))
        .mul(rcpArea)
        .toVar();
      // z at the (startX,startY) pixel centre = unbiased barycentric of rw (cw==rw there).
      const rowZ = toF(rw0.sub(bias0) as unknown as NI)
        .mul(dz0)
        .add(toF(rw1.sub(bias1) as unknown as NI).mul(dz1))
        .add(toF(rw2.sub(bias2) as unknown as NI).mul(dz2))
        .mul(rcpArea)
        .toVar();
      loopI('sy', startY as unknown as NI, endY as unknown as NI, (y) => {
        const cw0 = r0.toVar();
        const cw1 = r1.toVar();
        const cw2 = r2.toVar();
        const z = rowZ.toVar();
        const rowBase = uint(y).mul(cam.uW).toVar();
        loopI('sx', startX as unknown as NI, endX as unknown as NI, (x) => {
          If(
            cw0
              .greaterThanEqual(toI(0))
              .and(cw1.greaterThanEqual(toI(0)))
              .and(cw2.greaterThanEqual(toI(0))),
            () => {
              If(z.greaterThanEqual(0).and(z.lessThanEqual(1)), () => {
                const px = rowBase.add(uint(x));
                emit(px, z as unknown as NF);
              });
            },
          );
          cw0.addAssign(sx0);
          cw1.addAssign(sx1);
          cw2.addAssign(sx2);
          z.addAssign(dzdx);
        });
        r0.addAssign(sy0);
        r1.addAssign(sy1);
        r2.addAssign(sy2);
        rowZ.addAssign(dzdy);
      });
      return;
    }

    // ── MID-ONLY A.3 packed bias (bit-identical). Default builds never enter here.
    if (packBias) {
      // bias∈{0,−1} ⇒ bit0 = bias&1 ∈{0,1}; pack all three into one loop-carried word.
      const biasP = bias0
        .bitAnd(toI(1))
        .bitOr(bias1.bitAnd(toI(1)).shiftLeft(uint(1)))
        .bitOr(bias2.bitAnd(toI(1)).shiftLeft(uint(2)))
        .toVar() as unknown as NI;
      loopI('sy', startY as unknown as NI, endY as unknown as NI, (y) => {
        const cw0 = r0.toVar();
        const cw1 = r1.toVar();
        const cw2 = r2.toVar();
        const rowBase = uint(y).mul(cam.uW).toVar();
        loopI('sx', startX as unknown as NI, endX as unknown as NI, (x) => {
          If(
            cw0
              .greaterThanEqual(toI(0))
              .and(cw1.greaterThanEqual(toI(0)))
              .and(cw2.greaterThanEqual(toI(0))),
            () => {
              // uw = cw − bias, unpacked as cw + ((biasP>>i)&1) — bit-identical.
              const uw0 = cw0.add(biasP.bitAnd(toI(1))).toVar();
              const uw1 = cw1
                .add(biasP.shiftRight(uint(1)).bitAnd(toI(1)))
                .toVar();
              const uw2 = cw2
                .add(biasP.shiftRight(uint(2)).bitAnd(toI(1)))
                .toVar();
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
      return;
    }

    // ── DEFAULT shared loop (world1 scar/coop + depth + combined) — VERBATIM from the
    //    task-#76 extraction. ONLY change vs. the original: rowBase reads the uint uW
    //    (`cam.uW` in place of `uint(cam.uW)`), which drops the per-row f32→u32 convert;
    //    bit-identical because uW.value is the integer framebuffer width.
    loopI('sy', startY as unknown as NI, endY as unknown as NI, (y) => {
      const cw0 = r0.toVar();
      const cw1 = r1.toVar();
      const cw2 = r2.toVar();
      const rowBase = uint(y).mul(cam.uW).toVar();
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
