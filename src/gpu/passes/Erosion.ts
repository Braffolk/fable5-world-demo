/**
 * Hydraulic (pipe-model, Mei et al. 2007) + thermal erosion on storage buffers.
 *
 * Grid: res² cells, texel size l = WORLD_SIZE/res (2 m default).
 * State: terrain h, water w, sediment s, outflow flux f (vec4 L,R,D,U),
 * velocity v (vec2), hardness (static), depo (deposition accumulator).
 *
 * Per iteration (each box = one dispatch; WebGPU orders dispatches):
 *   1. flux:    f' from hydraulic head differences
 *   2. water:   w' from flux divergence + rain − evaporation; velocity
 *   3. erode:   capacity C = Kc·sin(slope)·|v| → dissolve/deposit (vs hardness)
 *   4. advect:  s' = s sampled at x − v·dt (semi-Lagrangian, bilinear)
 *   5. thermal: talus relaxation (gather form, symmetric pair transfers)
 *
 * Buffer rotation (height is in hA at every iteration boundary):
 *   even iter: hydra(hA,wA,sA → hB,wB,sB), thermal(hB→hA)
 *   odd  iter: hydra(hA,wB,sB → hB,wA,sA), thermal(hB→hA)
 *
 * Borders: neighbor indices clamp; border cells drain water to zero.
 */

import type { ComputeNode, Renderer } from 'three/webgpu';
import {
  Fn,
  If,
  Return,
  clamp,
  float,
  instanceIndex,
  instancedArray,
  max,
  min,
  vec2,
  vec4,
} from 'three/tsl';
import type { NB, NF, NI } from '../TSLTypes';
import type { FloatBuffer } from './HeightSynthesis';
import { labelGroup } from './HeightSynthesis';

export interface ErosionResult {
  /** eroded height (res²) — alias of an internal buffer, do not write */
  eroded: FloatBuffer;
  /** water depth at end of simulation (moisture hint) */
  water: FloatBuffer;
  /** accumulated deposition (soil-depth hint) */
  sediment: FloatBuffer;
}

export interface ErosionOpts {
  res: number;
  texel: number;
  iters: number;
  onProgress?: (done: number, total: number) => void;
}

// tuning constants, calibrated for l≈2 m, dt 0.03.
// Conservative rates + per-iter caps: erosion should carve drainage detail
// into the synthesized macro forms, not re-landscape them.
const DT = 0.03;
const RAIN = 0.01;
const EVAP = 0.02;
const KC = 0.55; // transport capacity
const KS = 0.28; // dissolve rate
const KD = 0.5; // deposit rate
const G = 9.81;
const THERMAL_RATE = 0.14;
const MAX_VEL = 6;
const MAX_ERODE_PER_ITER = 0.06;
const MAX_DEPOSIT_PER_ITER = 0.1;

export async function runErosion(
  renderer: Renderer,
  heightIn: FloatBuffer,
  hardness: FloatBuffer,
  opts: ErosionOpts,
): Promise<ErosionResult> {
  const { res, texel, iters } = opts;
  const N = res * res;

  const hA = instancedArray(N, 'float');
  const hB = instancedArray(N, 'float');
  const wA = instancedArray(N, 'float');
  const wB = instancedArray(N, 'float');
  const sA = instancedArray(N, 'float');
  const sB = instancedArray(N, 'float');
  const sTmp = instancedArray(N, 'float');
  const flux = instancedArray(N, 'vec4');
  const vel = instancedArray(N, 'vec2');
  const depo = instancedArray(N, 'float');

  const guard = (body: () => void) =>
    Fn<void>(() => {
      If(instanceIndex.greaterThanEqual(N), () => {
        Return();
      });
      body();
    });

  const cellXY = (): { x: NI; y: NI; i: NI } => {
    const i = instanceIndex.toInt();
    return { x: i.mod(res), y: i.div(res), i };
  };
  /** clamped neighbor index */
  const at = (x: NI, y: NI, ox: number, oy: number): NI => {
    const cx = clamp(float(x).add(ox), 0, res - 1).toInt();
    const cy = clamp(float(y).add(oy), 0, res - 1).toInt();
    return cy.mul(res).add(cx);
  };
  const isBorder = (x: NI, y: NI): NB =>
    float(x)
      .lessThan(1)
      .or(float(x).greaterThan(res - 2))
      .or(float(y).lessThan(1))
      .or(float(y).greaterThan(res - 2));

  // --- init (split: ≤8 storage buffers per compute stage) ----------------------
  const initK1 = guard(() => {
    const { i } = cellXY();
    hA.element(i).assign(heightIn.element(i));
    hB.element(i).assign(heightIn.element(i));
    wA.element(i).assign(0);
    wB.element(i).assign(0);
  })().compute(N);
  initK1.setName('erosionInit1');
  const initK2 = guard(() => {
    const { i } = cellXY();
    sA.element(i).assign(0);
    sB.element(i).assign(0);
    flux.element(i).assign(vec4(0));
    vel.element(i).assign(vec2(0));
    depo.element(i).assign(0);
  })().compute(N);
  initK2.setName('erosionInit2');

  // --- hydraulic kernels, parameterized by buffer roles ------------------------
  interface Roles {
    wSrc: FloatBuffer;
    sSrc: FloatBuffer;
    wDst: FloatBuffer;
    sDst: FloatBuffer;
  }

  const makeHydra = (r: Roles): ComputeNode[] => {
    // 1. flux (reads hA + wSrc → writes flux)
    const fluxK = guard(() => {
      const { x, y, i } = cellXY();
      const head = hA.element(i).add(r.wSrc.element(i)).toVar();
      const headOf = (ox: number, oy: number): NF => {
        const j = at(x, y, ox, oy);
        return hA.element(j).add(r.wSrc.element(j));
      };
      const fOld = flux.element(i).toVar();
      const k = float((DT * G) / texel);
      const f = vec4(
        max(0, fOld.x.add(head.sub(headOf(-1, 0)).mul(k))),
        max(0, fOld.y.add(head.sub(headOf(1, 0)).mul(k))),
        max(0, fOld.z.add(head.sub(headOf(0, -1)).mul(k))),
        max(0, fOld.w.add(head.sub(headOf(0, 1)).mul(k))),
      ).toVar();
      const total = f.x.add(f.y).add(f.z).add(f.w).max(1e-6);
      const scale = min(1, r.wSrc.element(i).div(total.mul(DT)));
      flux.element(i).assign(f.mul(scale));
    })().compute(N);
    fluxK.setName('eroFlux');

    // 2. water + velocity (reads flux, wSrc → writes wDst, vel)
    const waterK = guard(() => {
      const { x, y, i } = cellXY();
      const f = flux.element(i).toVar();
      const fL = flux.element(at(x, y, -1, 0)).toVar();
      const fR = flux.element(at(x, y, 1, 0)).toVar();
      const fD = flux.element(at(x, y, 0, -1)).toVar();
      const fU = flux.element(at(x, y, 0, 1)).toVar();
      const inflow = fL.y.add(fR.x).add(fD.w).add(fU.z);
      const outflow = f.x.add(f.y).add(f.z).add(f.w);
      const w0 = r.wSrc.element(i).toVar();
      const w1 = max(0, w0.add(inflow.sub(outflow).mul(DT))).toVar();
      const w2 = w1.mul(1 - EVAP * DT).add(RAIN * DT).toVar();
      If(isBorder(x, y), () => {
        w2.assign(0);
      });
      r.wDst.element(i).assign(w2);
      const wAvg = w0.add(w1).mul(0.5).max(1e-4);
      const vx = fL.y.sub(f.x).add(f.y).sub(fR.x).mul(0.5).div(wAvg.mul(texel));
      const vy = fD.w.sub(f.z).add(f.w).sub(fU.z).mul(0.5).div(wAvg.mul(texel));
      const v = vec2(vx, vy).toVar();
      const speed = v.length().max(1e-5);
      vel.element(i).assign(v.mul(min(1, float(MAX_VEL).div(speed))));
    })().compute(N);
    waterK.setName('eroWater');

    // 3. erode/deposit (reads hA, sSrc, vel, wDst, hardness → writes hB, sTmp)
    const erodeK = guard(() => {
      const { x, y, i } = cellXY();
      const h0 = hA.element(i).toVar();
      const hL = hA.element(at(x, y, -1, 0));
      const hR = hA.element(at(x, y, 1, 0));
      const hD = hA.element(at(x, y, 0, -1));
      const hU = hA.element(at(x, y, 0, 1));
      const grad = vec2(hR.sub(hL), hU.sub(hD)).div(2 * texel);
      const slope = grad.length();
      const sinA = slope.div(slope.mul(slope).add(1).sqrt()).max(0.012);
      const speed = vel.element(i).length();
      const shallowFade = clamp(r.wDst.element(i).mul(4), 0, 1);
      const cap = float(KC).mul(sinA).mul(speed).mul(shallowFade);
      const s0 = r.sSrc.element(i).toVar();
      const hard = hardness.element(i);
      const erodedAmt = min(
        MAX_ERODE_PER_ITER,
        max(0, cap.sub(s0))
          .mul(KS * DT)
          .mul(float(1).sub(hard.mul(0.92))),
      );
      const depositedAmt = min(MAX_DEPOSIT_PER_ITER, max(0, s0.sub(cap)).mul(KD * DT));
      hB.element(i).assign(h0.add(depositedAmt).sub(erodedAmt));
      sTmp.element(i).assign(s0.add(erodedAmt).sub(depositedAmt));
      depo.element(i).assign(depo.element(i).add(depositedAmt));
    })().compute(N);
    erodeK.setName('eroErode');

    // 4. advect sediment (reads sTmp, vel → writes sDst)
    const advectK = guard(() => {
      const { x, y, i } = cellXY();
      const back = vec2(float(x), float(y)).sub(vel.element(i).mul(DT / texel));
      const bx = clamp(back.x, 0, res - 1);
      const by = clamp(back.y, 0, res - 1);
      const x0f = bx.floor();
      const y0f = by.floor();
      const x1f = min(x0f.add(1), res - 1);
      const y1f = min(y0f.add(1), res - 1);
      const fx = bx.sub(x0f);
      const fy = by.sub(y0f);
      const x0 = x0f.toInt();
      const y0 = y0f.toInt();
      const x1 = x1f.toInt();
      const y1 = y1f.toInt();
      const s00 = sTmp.element(y0.mul(res).add(x0));
      const s10 = sTmp.element(y0.mul(res).add(x1));
      const s01 = sTmp.element(y1.mul(res).add(x0));
      const s11 = sTmp.element(y1.mul(res).add(x1));
      const top = s00.mul(fx.oneMinus()).add(s10.mul(fx));
      const bot = s01.mul(fx.oneMinus()).add(s11.mul(fx));
      r.sDst.element(i).assign(top.mul(fy.oneMinus()).add(bot.mul(fy)));
    })().compute(N);
    advectK.setName('eroAdvect');

    return [fluxK, waterK, erodeK, advectK];
  };

  // 5. thermal talus relaxation hB → hA (gather form; symmetric pair terms)
  const thermalK = ((): ComputeNode => {
    const k = guard(() => {
      const { x, y, i } = cellXY();
      const h0 = hB.element(i).toVar();
      const hard0 = hardness.element(i).toVar();
      // hard rock holds near-cliff angles (tan up to ~3.1 ≈ 72°) and sheds
      // material far slower; soft soil relaxes to ~29°
      const talus0 = float(0.55).add(hard0.mul(hard0).mul(2.6));
      const rate0 = float(1).sub(hard0).pow(1.5).mul(THERMAL_RATE * DT * texel);
      let net: NF = float(0);
      const offs: [number, number][] = [
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
        [-1, -1],
        [1, 1],
        [-1, 1],
        [1, -1],
      ];
      for (const [ox, oy] of offs) {
        const dist = texel * Math.hypot(ox, oy);
        const j = at(x, y, ox, oy);
        const hn = hB.element(j);
        const hardN = hardness.element(j);
        const talusN = float(0.55).add(hardN.mul(hardN).mul(2.6));
        const rateN = float(1).sub(hardN).pow(1.5).mul(THERMAL_RATE * DT * texel);
        const out = max(0, h0.sub(hn).div(dist).sub(talus0)).mul(rate0);
        const inn = max(0, hn.sub(h0).div(dist).sub(talusN)).mul(rateN);
        net = net.add(inn).sub(out);
      }
      hA.element(i).assign(h0.add(clamp(net, -0.22, 0.22)));
    })().compute(N);
    k.setName('eroThermal');
    return k;
  })();

  const hydraEven = makeHydra({ wSrc: wA, sSrc: sA, wDst: wB, sDst: sB });
  const hydraOdd = makeHydra({ wSrc: wB, sSrc: sB, wDst: wA, sDst: sA });

  await renderer.computeAsync(labelGroup([initK1, initK2], 'erosionInit'));

  const BATCH = 8;
  let done = 0;
  while (done < iters) {
    const nodes: ComputeNode[] = [];
    const n = Math.min(BATCH, iters - done);
    for (let k = 0; k < n; k++) {
      nodes.push(...((done + k) % 2 === 0 ? hydraEven : hydraOdd), thermalK);
    }
    await renderer.computeAsync(labelGroup(nodes, 'erosionHydraulic'));
    done += n;
    opts.onProgress?.(done, iters);
  }

  // height ends in hA every iteration; water/sediment end in the last wDst/sDst
  const finalW = iters % 2 === 1 ? wB : wA;
  return { eroded: hA, water: finalW, sediment: depo };
}
