/**
 * Hydrology pass: depression fill → flow accumulation → river carve → lakes
 * → moisture field. Runs on the sim grid after erosion.
 *
 * 1. FILL: iterative priority-flood relaxation — W converges to the filled
 *    DEM (every cell drains to the border via an ε-sloped path). Lakes are
 *    where W − H > δ.
 * 2. ACCUMULATION: particle tracing — rain particles descend the filled DEM
 *    via steepest descent, atomicAdd into a u32 accumulation grid.
 * 3. RIVERS: cells with accumulation above a threshold form the river
 *    network; carve a channel into H proportional to log(accum) and record
 *    water surface + flow direction for rendering/Phase-6 streams.
 * 4. MOISTURE: separable blur of (water presence + erosion wetness),
 *    distance-faded — drives biome classification and vegetation density.
 */

import type { ComputeNode, Renderer, StorageBufferNode } from 'three/webgpu';
import {
  Break,
  Fn,
  If,
  Loop,
  Return,
  atomicAdd,
  atomicLoad,
  atomicStore,
  clamp,
  float,
  instanceIndex,
  instancedArray,
  max,
  min,
  smoothstep,
  uint,
  vec2,
} from 'three/tsl';
import { valleyFields, type MacroParams } from '../../world/MacroMap';
import { WORLD_SIZE } from '../../world/WorldConst';
import { bilerpFloatBuffer } from '../BufferSample';
import { hash12 } from '../noise/NoiseTSL';
import type { NB, NF, NI, NU } from '../TSLTypes';
import type { FloatBuffer } from './HeightSynthesis';
import { labelGroup } from './HeightSynthesis';

export type Vec2Buffer = StorageBufferNode<'vec2'>;

export interface FlowResult {
  /** filled water surface W (≥ H); lakes where W−H > δ */
  waterSurface: FloatBuffer;
  /** log-scaled flow accumulation 0..~1 */
  flowStrength: FloatBuffer;
  /** river water depth (m) at river cells, 0 elsewhere */
  riverDepth: FloatBuffer;
  /** flow direction × speed (|v| = log-flow strength 0..1; ZERO in lakes) */
  flowDir: Vec2Buffer;
  /** moisture 0..1 */
  moisture: FloatBuffer;
  /** renderable water surface: fill level W in lakes/ponds (FLAT per pond —
   *  bed+blurredDepth built 30 m water towers where deep pots abut high
   *  ground), carved bed + gated depth on rivers, −1e4 sentinel when dry */
  waterYRaw: FloatBuffer;
}

export interface FlowOpts {
  res: number;
  texel: number;
  seed: number;
  /** designed carving splines — enforced through erosion-deposited dams */
  mp: MacroParams;
  /** rock hardness 0..1 — post-carve talus relax respects it (protects towers) */
  hardness: FloatBuffer;
  fillIters?: number;
  particles?: number;
  onProgress?: (msg: string, frac: number) => void;
}

/** open water requires real depth — shallow filled bowls become marsh, not ponds */
const LAKE_DELTA = 2.2;
const MARSH_DELTA = 0.15;

export async function runFlowRivers(
  renderer: Renderer,
  height: FloatBuffer,
  erosionWater: FloatBuffer,
  opts: FlowOpts,
): Promise<FlowResult> {
  const { res, seed } = opts;
  const N = res * res;
  const fillIters = opts.fillIters ?? 700;
  const particles = opts.particles ?? 3_000_000;

  const wA = instancedArray(N, 'float');
  const wB = instancedArray(N, 'float');
  const accumU = instancedArray(N, 'uint').toAtomic();
  const flowStrength = instancedArray(N, 'float');
  const riverDepth = instancedArray(N, 'float');
  const waterYRaw = instancedArray(N, 'float');
  const flowDir = instancedArray(N, 'vec2');
  const moistA = instancedArray(N, 'float');
  const moistB = instancedArray(N, 'float');

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
  const at = (x: NI, y: NI, ox: number, oy: number): NI => {
    const cx = clamp(float(x).add(ox), 0, res - 1).toInt();
    const cy = clamp(float(y).add(oy), 0, res - 1).toInt();
    return cy.mul(res).add(cx);
  };
  const OFFS: [number, number][] = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
    [-1, -1],
    [1, 1],
    [-1, 1],
    [1, -1],
  ];

  // --- 1. depression fill (multigrid: relaxation propagates ~1 cell/iter,
  //        so converge coarse first, then refine) -----------------------------
  const initMisc = guard(() => {
    const { i } = cellXY();
    moistA.element(i).assign(0);
    atomicStore(accumU.element(i), uint(0));
    flowStrength.element(i).assign(0);
    riverDepth.element(i).assign(0);
    flowDir.element(i).assign(vec2(0));
  })().compute(N);
  initMisc.setName('flowInitMisc');

  // ENFORCE the designed channels BEFORE the fill: erosion deposits bars/dams
  // across the trench (real rivers keep their channels open by continuous
  // flow we don't simulate). The macro spline floor is authoritative.
  const enforceK = guard(() => {
    const { x, y, i } = cellXY();
    const wpos = vec2(float(x).add(0.5), float(y).add(0.5))
      .div(res)
      .sub(0.5)
      .mul(WORLD_SIZE);
    const vf = valleyFields(wpos, opts.mp);
    // fade enforcement across the lake exactly like the synthesis trench,
    // otherwise we'd cut the outlet sill and drain the lake
    const dLake = wpos.sub(vec2(opts.mp.lakeC[0], opts.mp.lakeC[1])).length();
    const tLake = smoothstep(opts.mp.lakeR, opts.mp.lakeR * 0.25, dLake);
    const trenchFade = smoothstep(0.5, 0.12, tLake);
    // V-profile: deepest at the centerline, rim allowance rises smoothly —
    // a hard select() at fixed distance cut razor-walled rectangular canyons.
    // Beyond the rim the ceiling exceeds local terrain → constraint inactive.
    const mainProf = smoothstep(34, 4, vf.valleyDist);
    const tribProf = smoothstep(14, 1.5, vf.tribDist);
    const enforced = min(
      vf.valleyFloor
        .sub(float(15.2).mul(trenchFade).mul(mainProf))
        .add(mainProf.oneMinus().mul(46))
        .add(max(vf.valleyDist.sub(30), 0).mul(3)),
      vf.tribFloor
        .add(0.4)
        .add(tribProf.oneMinus().mul(30))
        .add(max(vf.tribDist.sub(12), 0).mul(3)),
    );
    height.element(i).assign(min(height.element(i), enforced));
  })().compute(N);
  enforceK.setName('channelEnforce');
  await renderer.computeAsync(labelGroup([initMisc, enforceK], 'flowInitEnforce'));

  interface FillLevel {
    res: number;
    iters: number;
    h: FloatBuffer;
    wA: FloatBuffer;
    wB: FloatBuffer;
  }
  const levels: FillLevel[] = [];
  {
    // coarse levels are nearly free — converge hard there so only local
    // refinement remains at fine levels (relaxation moves ~1 cell/iter)
    const specs = [
      { res: res >> 3, iters: 3000 },
      { res: res >> 2, iters: 1300 },
      { res: res >> 1, iters: 700 },
      { res, iters: Math.max(700, fillIters) },
    ];
    for (const s of specs) {
      levels.push({
        res: s.res,
        iters: s.iters,
        h: s.res === res ? height : instancedArray(s.res * s.res, 'float'),
        wA: s.res === res ? wA : instancedArray(s.res * s.res, 'float'),
        wB: s.res === res ? wB : instancedArray(s.res * s.res, 'float'),
      });
    }
  }

  const lvlHelpers = (lres: number) => ({
    xy: () => {
      const i = instanceIndex.toInt();
      return { x: i.mod(lres), y: i.div(lres), i };
    },
    at: (x: NI, y: NI, ox: number, oy: number): NI => {
      const cx = clamp(float(x).add(ox), 0, lres - 1).toInt();
      const cy = clamp(float(y).add(oy), 0, lres - 1).toInt();
      return cy.mul(lres).add(cx);
    },
    border: (x: NI, y: NI): NB =>
      float(x)
        .lessThan(1)
        .or(float(x).greaterThan(lres - 2))
        .or(float(y).lessThan(1))
        .or(float(y).greaterThan(lres - 2)),
    guard: (body: () => void) =>
      Fn<void>(() => {
        If(instanceIndex.greaterThanEqual(lres * lres), () => {
          Return();
        });
        body();
      }),
  });

  // min-downsample height pyramid (min preserves drainage channels)
  for (let li = levels.length - 2; li >= 0; li--) {
    const fine = levels[li + 1] as FillLevel;
    const coarse = levels[li] as FillLevel;
    const H = lvlHelpers(coarse.res);
    const k = H.guard(() => {
      const { x, y, i } = H.xy();
      const fx = float(x).mul(2).toInt();
      const fy = float(y).mul(2).toInt();
      const fres = fine.res;
      const i00 = fy.mul(fres).add(fx);
      const i10 = fy.mul(fres).add(clamp(float(fx).add(1), 0, fres - 1).toInt());
      const i01 = clamp(float(fy).add(1), 0, fres - 1).toInt().mul(fres).add(fx);
      const i11 = clamp(float(fy).add(1), 0, fres - 1)
        .toInt()
        .mul(fres)
        .add(clamp(float(fx).add(1), 0, fres - 1).toInt());
      coarse.h
        .element(i)
        .assign(
          min(min(fine.h.element(i00), fine.h.element(i10)), min(fine.h.element(i01), fine.h.element(i11))),
        );
    })().compute(coarse.res * coarse.res);
    k.setName(`fillDown_${coarse.res}`);
    await renderer.computeAsync(k);
  }

  // relax each level, seeding W from the coarser solution
  for (let li = 0; li < levels.length; li++) {
    const lvl = levels[li] as FillLevel;
    const H = lvlHelpers(lvl.res);
    const coarser = li > 0 ? (levels[li - 1] as FillLevel) : null;

    const initW = H.guard(() => {
      const { x, y, i } = H.xy();
      const h = lvl.h.element(i).toVar();
      let start: NF;
      if (coarser) {
        const g = vec2(float(x).add(0.5), float(y).add(0.5))
          .div(lvl.res)
          .mul(coarser.res)
          .sub(0.5);
        start = max(h, bilerpFloatBuffer(coarser.wA, coarser.res, g));
      } else {
        start = h.add(4000);
      }
      const w0 = H.border(x, y).select(h, start);
      lvl.wA.element(i).assign(w0);
      lvl.wB.element(i).assign(w0);
    })().compute(lvl.res * lvl.res);
    initW.setName(`fillInit_${lvl.res}`);
    await renderer.computeAsync(initW);

    const mkStep = (src: FloatBuffer, dst: FloatBuffer): ComputeNode => {
      const k = H.guard(() => {
        const { x, y, i } = H.xy();
        const h = lvl.h.element(i).toVar();
        If(H.border(x, y), () => {
          dst.element(i).assign(h);
          Return();
        });
        let lowest: NF = float(1e9);
        for (const [ox, oy] of OFFS) {
          // small ε keeps flats draining; large ε visibly tilts lake surfaces
          const eps = 0.0045 * Math.hypot(ox, oy);
          lowest = min(lowest, src.element(H.at(x, y, ox, oy)).add(eps));
        }
        dst.element(i).assign(max(h, min(src.element(i), lowest)));
      })().compute(lvl.res * lvl.res);
      k.setName(`fillStep_${lvl.res}`);
      return k;
    };
    const stepAB = mkStep(lvl.wA, lvl.wB);
    const stepBA = mkStep(lvl.wB, lvl.wA);

    const BATCH = 32;
    for (let it = 0; it < lvl.iters; it += BATCH) {
      const nodes: ComputeNode[] = [];
      for (let k = 0; k < Math.min(BATCH, lvl.iters - it); k++) {
        nodes.push((it + k) % 2 === 0 ? stepAB : stepBA);
      }
      await renderer.computeAsync(labelGroup(nodes, `flowFillStep_${lvl.res}`));
      opts.onProgress?.(
        `hydrology: filling depressions (${lvl.res}²)`,
        (li + it / lvl.iters) / levels.length,
      );
    }
    // ensure result is in wA for the next level's seed
    if (lvl.iters % 2 === 1) {
      const copyK = H.guard(() => {
        const { i } = H.xy();
        lvl.wA.element(i).assign(lvl.wB.element(i));
      })().compute(lvl.res * lvl.res);
      copyK.setName(`fillDownCopy_${lvl.res}`);
      await renderer.computeAsync(copyK);
    }
  }
  const W = wA;

  // --- 2. flow accumulation by particle tracing -------------------------------
  const STEPS = 260;
  const traceK = Fn<void>(() => {
    If(instanceIndex.greaterThanEqual(particles), () => {
      Return();
    });
    const pid = instanceIndex.toFloat();
    // jittered-grid spawn (decorrelated, full coverage)
    const cells = float(N);
    const spawn = pid.mul(cells.div(particles)).floor().toVar();
    const jx = hash12(vec2(pid, seed % 1000)).toVar();
    const jy = hash12(vec2(pid.add(0.5), (seed >> 8) % 1000)).toVar();
    const px = spawn.mod(res).add(jx).toVar();
    const py = spawn.div(res).floor().add(jy).toVar();

    // continuous gradient descent on the filled DEM with directional inertia.
    // Discrete 8-neighbor steepest descent locked every path onto axis/45°
    // polylines — the carved rivers read as straight grid scars (user-flagged).
    const dirX = float(0).toVar();
    const dirY = float(0).toVar();
    Loop(STEPS, () => {
      const xi = clamp(px, 1, res - 2).toInt();
      const yi = clamp(py, 1, res - 2).toInt();
      const i = yi.mul(res).add(xi);
      atomicAdd(accumU.element(i), uint(1));
      // sediment settles where the water column is deep — stop in lakes
      If(W.element(i).sub(height.element(i)).greaterThan(LAKE_DELTA), () => {
        Break();
      });
      // central differences of bilinear W around the continuous position
      const gp = vec2(px, py);
      const e = 0.65;
      const gx = bilerpFloatBuffer(W, res, gp.add(vec2(e, 0))).sub(
        bilerpFloatBuffer(W, res, gp.sub(vec2(e, 0))),
      );
      const gy = bilerpFloatBuffer(W, res, gp.add(vec2(0, e))).sub(
        bilerpFloatBuffer(W, res, gp.sub(vec2(0, e))),
      );
      // flatness cutoff well above the fill's ε-tilt (~0.006/cell): on filled
      // flats the tilt is uniform, so surviving particles all walk the same
      // direction and print parallel straight lines across the marsh
      const gLen = vec2(gx, gy).length();
      If(gLen.lessThan(0.012), () => {
        Break();
      });
      // inertia keeps channels coherent through grid noise (gentle meanders)
      const nx = gx.div(gLen).negate();
      const ny = gy.div(gLen).negate();
      dirX.assign(dirX.mul(0.45).add(nx.mul(0.55)));
      dirY.assign(dirY.mul(0.45).add(ny.mul(0.55)));
      const dLen = vec2(dirX, dirY).length().max(1e-6);
      px.addAssign(dirX.div(dLen));
      py.addAssign(dirY.div(dLen));
      If(
        px.lessThan(1).or(px.greaterThan(res - 2)).or(py.lessThan(1)).or(py.greaterThan(res - 2)),
        () => {
          Break();
        },
      );
    });
  })().compute(particles);
  traceK.setName('flowTrace');
  opts.onProgress?.('hydrology: tracing flow', 0.55);
  await renderer.computeAsync(traceK);

  // shared separable triangle blur builder
  const makeBlur = (
    src: FloatBuffer,
    dst: FloatBuffer,
    dx: number,
    dy: number,
    R: number,
  ): ComputeNode => {
    const k = guard(() => {
      const { x, y, i } = cellXY();
      let sum: NF = float(0);
      let wsum = 0;
      for (let o = -R; o <= R; o++) {
        const wgt = 1 - Math.abs(o) / (R + 1);
        sum = sum.add(src.element(at(x, y, o * dx, o * dy)).mul(wgt));
        wsum += wgt;
      }
      dst.element(i).assign(sum.div(wsum));
    })().compute(N);
    k.setName('sepBlur');
    return k;
  };

  // --- 3a. flow strength from accumulation ------------------------------------
  // TWO thresholds with very different jobs (user: "40-60% of cliff sides
  // end up being rivers"):
  //  - RIVER_T (low) → flowStrength: drives CARVING, moisture, splat beds,
  //    boulder affinity. The dense drainage texture is good terrain.
  //  - WATER_T (≈15× stricter) → waterStrength: drives VISIBLE open water
  //    only. Small gullies stay dry cobbled scars; the main river, big
  //    tributaries and ravine runs keep their streams.
  const RIVER_T = particles / N + 14;
  // raised 220 → 320 with the stricter rSurf curve (user: "A TON of water
  // absolutely everywhere") — only genuine collectors render open water
  const WATER_T = particles / N + 320;
  const waterStrength = instancedArray(N, 'float');
  const strengthK = guard(() => {
    const { i } = cellXY();
    // @types/three models AtomicFunctionNode without value semantics; at
    // runtime atomicLoad yields a u32 expression — cast for the converter
    const acc = float(atomicLoad(accumU.element(i)) as unknown as NU).toVar();
    const t = clamp(acc.div(RIVER_T), 1e-5, 60);
    const s = clamp(t.log2().mul(0.18), 0, 1).mul(t.greaterThan(1).select(1, 0));
    flowStrength.element(i).assign(s);
    const tw = clamp(acc.div(WATER_T), 1e-5, 60);
    const sw = clamp(tw.log2().mul(0.21), 0, 1).mul(tw.greaterThan(1).select(1, 0));
    waterStrength.element(i).assign(sw);
  })().compute(N);
  strengthK.setName('flowStrength');

  // --- 3b. widen: blur the strength field (channels get real width — the
  //         raw particle lines are one cell wide and carve grid scars) --------
  opts.onProgress?.('hydrology: widening channels', 0.68);
  await renderer.computeAsync(
    labelGroup(
      [
        strengthK,
        makeBlur(flowStrength, moistB, 1, 0, 2),
        makeBlur(moistB, flowStrength, 0, 1, 2),
        makeBlur(waterStrength, moistB, 1, 0, 2),
        makeBlur(moistB, waterStrength, 0, 1, 2),
      ],
      'flowWiden',
    ),
  );

  // lake-depth field, blurred: post-erosion hummocks leave 2–6 m potholes
  // everywhere in the wetland — per-cell W−H painted them as dotted ponds.
  // Blur kills isolated pits; the real lake's interior depth is unaffected.
  const lakeDepthB = instancedArray(N, 'float');
  const lakeDepthK = guard(() => {
    const { i } = cellXY();
    lakeDepthB.element(i).assign(W.element(i).sub(height.element(i)));
  })().compute(N);
  lakeDepthK.setName('lakeDepth');
  await renderer.computeAsync(
    labelGroup(
      [
        lakeDepthK,
        makeBlur(lakeDepthB, moistB, 1, 0, 3),
        makeBlur(moistB, lakeDepthB, 0, 1, 3),
      ],
      'flowLakeDepth',
    ),
  );

  // --- 3c. carve from the blurred field, fade out inside lakes ----------------
  const carveK = guard(() => {
    const { x, y, i } = cellXY();
    const lakeD = lakeDepthB.element(i).toVar();
    const isLake = lakeD.greaterThan(LAKE_DELTA);
    // ×2.1 recovers the pre-blur peak so big rivers still reach full depth
    const sB = clamp(flowStrength.element(i).mul(2.1), 0, 1).toVar();
    flowStrength.element(i).assign(isLake.select(float(1), sB));
    // lakebeds keep their filled profile — carving there printed the particle
    // wander pattern into the basin floor (user-flagged artifact)
    const lakeFade = smoothstep(LAKE_DELTA * 0.7, 0.12, lakeD);
    const depth = sB.pow(1.35).mul(7.5).mul(lakeFade);
    const hNew = height.element(i).sub(depth).toVar();
    height.element(i).assign(hNew);
    const wl = W.element(at(x, y, -1, 0));
    const wr = W.element(at(x, y, 1, 0));
    const wd = W.element(at(x, y, 0, -1));
    const wu = W.element(at(x, y, 0, 1));
    const g = vec2(wl.sub(wr), wd.sub(wu));
    // open water only where the run is gentle: steep reaches are whitewater
    // chutes/falls, not standing sheets — they carve but render dry
    const slopeW = g.length().div(2 * opts.texel);
    const rdGate = smoothstep(0.5, 0.24, slopeW);
    const rdRiver = depth.mul(0.45).add(0.12).mul(rdGate);
    riverDepth.element(i).assign(
      isLake.select(lakeD, sB.greaterThan(0.02).select(rdRiver, float(0))),
    );
    // render surface: ponds sit at their FILL level W (flat, meets terrain
    // at the true shoreline — bed+blurredDepth towers over pot rims); rivers
    // at carved bed + a depth from the STRICT water threshold, minus the
    // widen-blur's 0.12 m apron floor. Carve-only gullies stay dry.
    // USER MANDATE (post water-shader): much stricter visible water — the
    // old curve (sat ×2.1, ^1.35, peak 3.4 m) flooded gorge floors wall-to-
    // wall. Slower saturation + sharper power keep water to the channel
    // CORE; peak ~1.5 m is wading depth, headwaters become trickles in a
    // cobbled bed. Lakes (fill level) and flowStrength consumers untouched.
    const wB = clamp(waterStrength.element(i).mul(1.5), 0, 1);
    const rSurf = wB.pow(2.2).mul(3.3).mul(lakeFade).mul(0.45).add(0.12)
      .mul(rdGate).sub(0.12).max(0);
    const riverWet = wB.greaterThan(0.05).and(rSurf.greaterThan(0.05));
    waterYRaw.element(i).assign(
      isLake.select(W.element(i), riverWet.select(hNew.add(rSurf), float(-1e4))),
    );
    // flow direction × speed: downhill gradient of W scaled by strength.
    // Lakes get ZERO — their filled W is flat so the raw gradient is noise,
    // and Phase-6 water reads |flowDir| as the ripple-advection speed
    // (still lakes vs streaming rivers).
    const spd = isLake.select(float(0), sB);
    flowDir.element(i).assign(g.div(g.length().max(1e-5)).mul(spd));
    // moisture source: lakes + marshes + rivers + residual erosion water
    const marsh = lakeD.greaterThan(MARSH_DELTA).select(float(0.8), float(0));
    const src = isLake
      .select(float(1), max(sB.mul(0.85), marsh))
      .add(clamp(erosionWater.element(i).mul(2), 0, 0.35));
    moistA.element(i).assign(clamp(src, 0, 1));
  })().compute(N);
  carveK.setName('riverCarve');
  opts.onProgress?.('hydrology: carving rivers', 0.72);
  await renderer.computeAsync(carveK);

  // --- 3d. talus relax: carved walls collapse to angle of repose --------------
  // The carve (and any residual erosion notching) leaves near-vertical cell
  // walls; real channels are flanked by talus. Hardness raises the stable
  // angle so karst towers and hard strata keep their cliffs.
  const hT = instancedArray(N, 'float');
  const hardness = opts.hardness;
  const texel = opts.texel;
  const mkRelax = (src: FloatBuffer, dst: FloatBuffer): ComputeNode => {
    const k = guard(() => {
      const { x, y, i } = cellXY();
      const hC = src.element(i).toVar();
      const hardC = hardness.element(i).toVar();
      const talusC = float(texel).mul(hardC.mul(hardC).mul(2.8).add(0.62));
      let delta: NF = float(0);
      for (const [ox, oy] of OFFS.slice(0, 4)) {
        const ni = at(x, y, ox, oy);
        const hN = src.element(ni);
        const hardN = hardness.element(ni);
        const talusN = float(texel).mul(hardN.mul(hardN).mul(2.8).add(0.62));
        const dOut = hC.sub(hN).sub(talusC).max(0); // we shed downhill
        const dIn = hN.sub(hC).sub(talusN).max(0); // neighbor sheds onto us
        delta = delta.add(dIn.sub(dOut));
      }
      dst.element(i).assign(hC.add(delta.mul(0.12)));
    })().compute(N);
    k.setName('talusRelax');
    return k;
  };
  const relaxAB = mkRelax(height, hT);
  const relaxBA = mkRelax(hT, height);
  opts.onProgress?.('hydrology: talus relax', 0.78);
  for (let it = 0; it < 13; it++) {
    await renderer.computeAsync(labelGroup([relaxAB, relaxBA], 'flowTalusRelax'));
  }

  // --- 4. moisture: separable blur --------------------------------------------
  opts.onProgress?.('hydrology: moisture field', 0.85);
  await renderer.computeAsync(
    labelGroup(
      [
        makeBlur(moistA, moistB, 1, 0, 10),
        makeBlur(moistB, moistA, 0, 1, 10),
        makeBlur(moistA, moistB, 1, 0, 10),
        makeBlur(moistB, moistA, 0, 1, 10),
      ],
      'flowMoisture',
    ),
  );

  return {
    waterSurface: W,
    flowStrength,
    riverDepth,
    flowDir,
    moisture: moistA,
    waterYRaw,
  };
}
