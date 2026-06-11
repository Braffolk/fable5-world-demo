/**
 * Heightfield — owner of all terrain GPU state. Orchestrates the generation
 * passes (synthesis → erosion → hydrology → classification) and exposes
 * buffers/textures + TSL sampling helpers to the rest of the engine.
 *
 * Layout: row-major res×res grids; texel (x,y) ↔ world
 * ((x+0.5)/res − 0.5)·WORLD_SIZE on both axes (x→world x, y→world z).
 */

import { FloatType, HalfFloatType, NearestFilter, RedFormat } from 'three';
import type { ComputeNode, Renderer } from 'three/webgpu';
import { StorageTexture } from 'three/webgpu';
import {
  Fn,
  If,
  Return,
  clamp,
  float,
  floor,
  fract,
  instanceIndex,
  instancedArray,
  mix,
  texture,
  textureStore,
  uvec2,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import type { LaasParams } from '../core/Params';
import type { WorldSeed } from '../core/Seed';
import { bilerpFloatBuffer, uvToGrid } from '../gpu/BufferSample';
import { bakeNoiseTextures } from '../gpu/passes/NoiseBake';
import type { NF, NI, NV2, NV3 } from '../gpu/TSLTypes';
import { runBiomeSnow } from '../gpu/passes/BiomeSnow';
import { runErosion } from '../gpu/passes/Erosion';
import { runFlowRivers, type FlowResult } from '../gpu/passes/FlowRivers';
import {
  runHeightSynthesis,
  type FloatBuffer,
  type SynthesisResult,
} from '../gpu/passes/HeightSynthesis';
import { makeMacroParams, type MacroParams } from './MacroMap';
import { WORLD_SIZE, qualityConfig, type QualityConfig } from './WorldConst';

export type ProgressFn = (p: number, msg: string) => void;

export class Heightfield {
  readonly cfg: QualityConfig;
  readonly mp: MacroParams;
  readonly res: number;

  /** final height (m), res×res storage buffer — single source of truth */
  readonly height: SynthesisResult['height'];
  readonly hardness: SynthesisResult['hardness'];
  /** pre-erosion copy kept for the ?scene=terrain split view */
  preErosion: FloatBuffer | null = null;
  /** erosion by-products at sim res (moisture/soil hints for later passes) */
  simWater: FloatBuffer | null = null;
  simSediment: FloatBuffer | null = null;
  simRes = 0;
  /** hydrology outputs at sim res */
  flow: FlowResult | null = null;
  /** renderable water surface (m) at sim res: carved bed + riverDepth at
   *  water cells; DRY cells hold simBed − 2 so bilinear shorelines cut
   *  below the banks (f32 buffer — f16 textures quantize ~1 m up here) */
  waterY: FloatBuffer | null = null;
  /** min-reduced waterY (simRes/8) for FAR clipmap levels: coarse vertices
   *  sampling the full field stretch one wet texel across a whole 48 m
   *  cell — "mountains half covered in water" from afar. The min makes
   *  distance conservative: narrow channels vanish, lakes survive. */
  waterYFar: FloatBuffer | null = null;
  waterFarRes = 0;
  /** rgba16f at sim res: moisture, flowStrength, riverDepth, waterSurface W */
  fieldsTex: StorageTexture | null = null;
  /** rgba8 at full res: biomeId/8, snow, vegDensity, rockExposure */
  biomeTex: StorageTexture | null = null;
  /** CPU height mirror for camera clamping / tools (filled by readback) */
  cpuHeights: Float32Array | null = null;
  /** CPU waterY mirror (sim res) — underwater camera guard */
  cpuWaterY: Float32Array | null = null;

  /** r32float height texture (nearest-sample / textureLoad only) */
  readonly heightTex: StorageTexture;
  /** rgba16f: xyz = world-space normal, w = slope (rise/run) */
  readonly normalTex: StorageTexture;
  /** baked tileable noise (see NoiseBake channel map) — materials sample these */
  noiseA: StorageTexture | null = null;
  noiseB: StorageTexture | null = null;

  private constructor(
    cfg: QualityConfig,
    mp: MacroParams,
    synth: SynthesisResult,
    heightTex: StorageTexture,
    normalTex: StorageTexture,
  ) {
    this.cfg = cfg;
    this.mp = mp;
    this.res = synth.res;
    this.height = synth.height;
    this.hardness = synth.hardness;
    this.heightTex = heightTex;
    this.normalTex = normalTex;
  }

  static async generate(
    renderer: Renderer,
    params: LaasParams,
    seed: WorldSeed,
    progress: ProgressFn,
  ): Promise<Heightfield> {
    const cfg = qualityConfig(params.preset);
    const mp = makeMacroParams(seed);

    progress(0.04, `terrain: synthesizing ${cfg.heightRes}² heightfield`);
    const synth = await runHeightSynthesis(renderer, cfg.heightRes, mp);

    const heightTex = new StorageTexture(cfg.heightRes, cfg.heightRes);
    heightTex.type = FloatType;
    heightTex.format = RedFormat;
    heightTex.magFilter = NearestFilter;
    heightTex.minFilter = NearestFilter;
    heightTex.generateMipmaps = false;

    const normalTex = new StorageTexture(cfg.heightRes, cfg.heightRes);
    normalTex.type = HalfFloatType;
    normalTex.generateMipmaps = false;

    const hf = new Heightfield(cfg, mp, synth, heightTex, normalTex);

    const noise = await bakeNoiseTextures(renderer);
    hf.noiseA = noise.texA;
    hf.noiseB = noise.texB;

    // --- erosion at sim res, then detail-preserving compose back to full res --
    progress(0.08, `terrain: synthesizing ${cfg.simRes}² erosion grid`);
    const synthSim = await runHeightSynthesis(renderer, cfg.simRes, mp);

    progress(0.1, `terrain: eroding (${cfg.erosionIters} iterations)`);
    const erosion = await runErosion(renderer, synthSim.height, synthSim.hardness, {
      res: cfg.simRes,
      texel: WORLD_SIZE / cfg.simRes,
      iters: cfg.erosionIters,
      onProgress: (d, t) => progress(0.1 + 0.45 * (d / t), `terrain: eroding ${d}/${t}`),
    });
    hf.simWater = erosion.water;
    hf.simSediment = erosion.sediment;
    hf.simRes = cfg.simRes;

    // hydrology BEFORE compose: river carve must reach the full-res field
    hf.flow = await runFlowRivers(renderer, erosion.eroded, erosion.water, {
      res: cfg.simRes,
      texel: WORLD_SIZE / cfg.simRes,
      seed: seed.sub('hydrology'),
      mp,
      hardness: synthSim.hardness,
      onProgress: (msg, frac) => progress(0.55 + frac * 0.12, msg),
    });

    // water render surface from the CARVED sim bed (runFlowRivers mutates
    // erosion.eroded in place: carve + talus relax)
    hf.waterY = await Heightfield.buildWaterY(
      renderer,
      erosion.eroded,
      hf.flow.waterYRaw,
      cfg.simRes,
    );
    hf.waterFarRes = Math.floor(cfg.simRes / 8);
    hf.waterYFar = await Heightfield.reduceWaterY(renderer, hf.waterY, cfg.simRes, 8);

    progress(0.7, 'terrain: composing eroded field');
    await hf.composeEroded(renderer, synthSim.height, erosion.eroded);

    progress(0.82, 'terrain: deriving maps');
    await hf.rebuildDerivedMaps(renderer);
    await hf.buildFieldsTex(renderer);

    progress(0.88, 'terrain: biome + snow classification');
    if (!hf.fieldsTex) throw new Error('fieldsTex missing before biome pass');
    hf.biomeTex = await runBiomeSnow(renderer, hf.height, {
      res: hf.res,
      mp,
      normalTex: hf.normalTex,
      fieldsTex: hf.fieldsTex,
    });

    progress(0.93, 'terrain: height readback for camera');
    const ab = await renderer.getArrayBufferAsync(hf.height.value);
    hf.cpuHeights = new Float32Array(ab);
    const wab = await renderer.getArrayBufferAsync(hf.waterY.value);
    hf.cpuWaterY = new Float32Array(wab);
    return hf;
  }

  /** CPU height lookup (bilinear) — camera clamping, bookmarks, tools */
  heightAtCpu(x: number, z: number): number {
    const hts = this.cpuHeights;
    if (!hts) return 0;
    const res = this.res;
    const gx = Math.min(Math.max(((x / WORLD_SIZE) + 0.5) * res - 0.5, 0), res - 1.001);
    const gz = Math.min(Math.max(((z / WORLD_SIZE) + 0.5) * res - 0.5, 0), res - 1.001);
    const x0 = Math.floor(gx);
    const z0 = Math.floor(gz);
    const fx = gx - x0;
    const fz = gz - z0;
    const i = (xx: number, zz: number): number => hts[Math.min(zz, res - 1) * res + Math.min(xx, res - 1)] ?? 0;
    const a = i(x0, z0) * (1 - fx) + i(x0 + 1, z0) * fx;
    const b = i(x0, z0 + 1) * (1 - fx) + i(x0 + 1, z0 + 1) * fx;
    return a * (1 - fz) + b * fz;
  }

  /** CPU waterY lookup (bilinear, sim res) — dry cells sit ~2 m below the
   *  bed, so `max(ground, waterYAtCpu + ε)` is a safe camera floor */
  waterYAtCpu(x: number, z: number): number {
    const wy = this.cpuWaterY;
    if (!wy) return -1e4;
    const res = this.simRes;
    const gx = Math.min(Math.max(((x / WORLD_SIZE) + 0.5) * res - 0.5, 0), res - 1.001);
    const gz = Math.min(Math.max(((z / WORLD_SIZE) + 0.5) * res - 0.5, 0), res - 1.001);
    const x0 = Math.floor(gx);
    const z0 = Math.floor(gz);
    const fx = gx - x0;
    const fz = gz - z0;
    const i = (xx: number, zz: number): number => wy[Math.min(zz, res - 1) * res + Math.min(xx, res - 1)] ?? -1e4;
    const a = i(x0, z0) * (1 - fx) + i(x0 + 1, z0) * fx;
    const b = i(x0, z0 + 1) * (1 - fx) + i(x0 + 1, z0 + 1) * fx;
    return a * (1 - fz) + b * fz;
  }

  private static async buildWaterY(
    renderer: Renderer,
    bed: FloatBuffer,
    waterYRaw: FloatBuffer,
    res: number,
  ): Promise<FloatBuffer> {
    const out = instancedArray(res * res, 'float');
    const wet = instancedArray(res * res, 'float');
    const kernel = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(res * res), () => {
        Return();
      });
      // Hydrology already decided where open water exists (waterYRaw: pond
      // fill level / river surface / −1e4 dry sentinel). Here: encode DRY
      // cells as 3×3 NEIGHBORHOOD-MIN bed − 2 — a raised bank texel at
      // bankBed−2 can still sit ABOVE the channel's water level, and the
      // bilinear then builds standing water walls up every bank (user-
      // reported "spikes"). With the min, wet→dry spans always cross under
      // the waterline.
      const x = i.mod(res).toInt();
      const y = i.div(res).toInt();
      const xm = clamp(float(x).sub(1), 0, res - 1).toInt();
      const xp = clamp(float(x).add(1), 0, res - 1).toInt();
      const ym = clamp(float(y).sub(1), 0, res - 1).toInt();
      const yp = clamp(float(y).add(1), 0, res - 1).toInt();
      const b = bed.element(i).toVar();
      const hl = bed.element(y.mul(res).add(xm)).toVar();
      const hr = bed.element(y.mul(res).add(xp)).toVar();
      const hd = bed.element(ym.mul(res).add(x)).toVar();
      const hu = bed.element(yp.mul(res).add(x)).toVar();
      const d00 = bed.element(ym.mul(res).add(xm));
      const d10 = bed.element(ym.mul(res).add(xp));
      const d01 = bed.element(yp.mul(res).add(xm));
      const d11 = bed.element(yp.mul(res).add(xp));
      const bMin = b
        .min(hl).min(hr).min(hd).min(hu)
        .min(d00).min(d10).min(d01).min(d11);
      const raw = waterYRaw.element(i);
      const isWet = raw.greaterThan(-1e3);
      wet.element(i).assign(isWet.select(float(1), float(0)));
      out.element(i).assign(isWet.select(raw, bMin.sub(2)));
    })().compute(res * res);
    kernel.setName('waterY');
    await renderer.computeAsync(kernel);

    // smooth WET cells toward their wet neighbors: steep cascade reaches
    // otherwise render as 2 m staircase shards — real chutes are slides.
    // Dry cells and lake flats are untouched (neighbors equal the mean).
    const tmp = instancedArray(res * res, 'float');
    const mkSmooth = (src: FloatBuffer, dst: FloatBuffer): ComputeNode => {
      const k = Fn(() => {
        const i = instanceIndex;
        If(i.greaterThanEqual(res * res), () => {
          Return();
        });
        const x = i.mod(res).toInt();
        const y = i.div(res).toInt();
        const xm = clamp(float(x).sub(1), 0, res - 1).toInt();
        const xp = clamp(float(x).add(1), 0, res - 1).toInt();
        const ym = clamp(float(y).sub(1), 0, res - 1).toInt();
        const yp = clamp(float(y).add(1), 0, res - 1).toInt();
        const c = src.element(i).toVar();
        const sum = c.toVar();
        const wsum = float(1).toVar();
        for (const [ox, oy] of [[xm, y], [xp, y], [x, ym], [x, yp]] as const) {
          const ni = (oy as NI).mul(res).add(ox as NI);
          const wn = wet.element(ni);
          sum.addAssign(src.element(ni).mul(wn));
          wsum.addAssign(wn);
        }
        const sm = sum.div(wsum);
        dst.element(i).assign(wet.element(i).greaterThan(0.5).select(sm, c));
      })().compute(res * res);
      k.setName('waterYSmooth');
      return k;
    };
    for (let it = 0; it < 2; it++) {
      await renderer.computeAsync([mkSmooth(out, tmp), mkSmooth(tmp, out)]);
    }

    // WET-TO-WET cliff cut: adjacent ponds can legitimately fill at levels
    // a meter+ apart; across their (sub-texel) divide the bilinear+smoothed
    // surface renders a steep dark water RAMP — a hovering slab from afar
    // (user-class artifact found at the twin lake). Water never ramps:
    // where the gradient BETWEEN WET CELLS exceeds ~0.35, sink the cell to
    // dry. Shorelines are untouched (their neighbor is dry, not wet).
    const texel = WORLD_SIZE / res;
    const cliffK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(res * res), () => {
        Return();
      });
      const x = i.mod(res).toInt();
      const y = i.div(res).toInt();
      const xm = clamp(float(x).sub(1), 0, res - 1).toInt();
      const xp = clamp(float(x).add(1), 0, res - 1).toInt();
      const ym = clamp(float(y).sub(1), 0, res - 1).toInt();
      const yp = clamp(float(y).add(1), 0, res - 1).toInt();
      const c = out.element(i).toVar();
      const dMax = float(0).toVar();
      for (const [ox, oy] of [[xm, y], [xp, y], [x, ym], [x, yp]] as const) {
        const ni = (oy as NI).mul(res).add(ox as NI);
        const wn = wet.element(ni);
        dMax.assign(dMax.max(c.sub(out.element(ni)).abs().mul(wn)));
      }
      const isWet = wet.element(i).greaterThan(0.5);
      const cliff = dMax.div(texel).greaterThan(0.35);
      tmp.element(i).assign(
        isWet.and(cliff).select(bed.element(i).sub(2), c),
      );
    })().compute(res * res);
    cliffK.setName('waterYCliffCut');
    const copyK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(res * res), () => {
        Return();
      });
      out.element(i).assign(tmp.element(i));
    })().compute(res * res);
    copyK.setName('waterYCopy');
    await renderer.computeAsync([cliffK, copyK]);
    return out;
  }

  private static async reduceWaterY(
    renderer: Renderer,
    src: FloatBuffer,
    res: number,
    factor: number,
  ): Promise<FloatBuffer> {
    const farRes = Math.floor(res / factor);
    const out = instancedArray(farRes * farRes, 'float');
    const kernel = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(farRes * farRes), () => {
        Return();
      });
      const bx = i.mod(farRes).mul(factor).toInt();
      const by = i.div(farRes).mul(factor).toInt();
      // Plain conservative MIN. Known limitation (diagnosed at the twin
      // lake, 2026-06-12): shore-overlapping blocks dip toward the dry
      // sentinel, so a low grazing view across a LARGE lake shows a thin
      // dark band at its far rim. Alternatives tried and rejected:
      // max-of-wet domes over river inlets; min-of-wet lenses where wide
      // inlet rivers meet the lake (two legitimate wet levels bridge
      // across 16 m far-texels). The real fix is a per-water-body far
      // field or the planar-lake pass (logged in STATUS) — at ≥384 m the
      // min's dip is the least-bad behavior and shore ramps fade out in
      // the material on the NEAR levels where they would be obvious.
      const mn = float(1e9).toVar();
      for (let oy = 0; oy < factor; oy++) {
        for (let ox = 0; ox < factor; ox++) {
          const idx = by.add(oy).mul(res).add(bx.add(ox));
          mn.assign(mn.min(src.element(idx)));
        }
      }
      out.element(i).assign(mn);
    })().compute(farRes * farRes);
    kernel.setName('waterYFar');
    await renderer.computeAsync(kernel);
    return out;
  }

  /** bilinear water-surface sample (vertex/fragment safe — buffer reads) */
  sampleWaterY(p: NV2): NF {
    const wy = this.waterY;
    if (!wy) throw new Error('waterY not built');
    const uv = clamp(this.uvFromWorld(p), 0, 1);
    return bilerpFloatBuffer(wy, this.simRes, uvToGrid(uv, this.simRes));
  }

  /** same, from the min-reduced far field (distant clipmap levels) */
  sampleWaterYFar(p: NV2): NF {
    const wy = this.waterYFar;
    if (!wy) throw new Error('waterYFar not built');
    const uv = clamp(this.uvFromWorld(p), 0, 1);
    return bilerpFloatBuffer(wy, this.waterFarRes, uvToGrid(uv, this.waterFarRes));
  }

  /** nearest-texel waterY (compute kernels: veg/debris water gating) */
  sampleWaterYNearest(p: NV2): NF {
    const wy = this.waterY;
    if (!wy) throw new Error('waterY not built');
    const res = this.simRes;
    const g = clamp(this.uvFromWorld(p), 0, 1).mul(res);
    const x = clamp(floor(g.x), 0, res - 1).toInt();
    const y = clamp(floor(g.y), 0, res - 1).toInt();
    return wy.element(y.mul(res).add(x));
  }

  /** pack sim-res hydrology fields into a filterable rgba16f texture */
  private async buildFieldsTex(renderer: Renderer): Promise<void> {
    const flow = this.flow;
    if (!flow) return;
    const res = this.simRes;
    const tex = new StorageTexture(res, res);
    tex.type = HalfFloatType;
    tex.generateMipmaps = false;
    const kernel = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(res * res), () => {
        Return();
      });
      const x = i.mod(res);
      const y = i.div(res);
      textureStore(
        tex,
        uvec2(x.toUint(), y.toUint()),
        vec4(
          flow.moisture.element(i),
          flow.flowStrength.element(i),
          flow.riverDepth.element(i),
          flow.waterSurface.element(i),
        ),
      ).toWriteOnly();
    })().compute(res * res);
    kernel.setName('fieldsTexPack');
    await renderer.computeAsync(kernel);
    this.fieldsTex = tex;
  }

  /**
   * height ← upsample(eroded_sim) + (height_full − upsample(preSim)).
   * Keeps full-res synthesis micro-detail riding on the eroded macro field.
   * Also snapshots the pre-erosion full-res height for the split view.
   */
  private async composeEroded(
    renderer: Renderer,
    preSim: FloatBuffer,
    erodedSim: FloatBuffer,
  ): Promise<void> {
    const res = this.res;
    const simRes = this.simRes;
    const pre = instancedArray(res * res, 'float');
    const kernel = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(res * res), () => {
        Return();
      });
      const x = i.mod(res);
      const y = i.div(res);
      const h = this.height.element(i).toVar();
      pre.element(i).assign(h);
      const uv = vec2(float(x).add(0.5), float(y).add(0.5)).div(res);
      const g = uvToGrid(uv, simRes);
      const macroEroded = bilerpFloatBuffer(erodedSim, simRes, g);
      const macroPre = bilerpFloatBuffer(preSim, simRes, g);
      this.height.element(i).assign(macroEroded.add(h.sub(macroPre)));
    })().compute(res * res);
    kernel.setName('erosionCompose');
    await renderer.computeAsync(kernel);
    this.preErosion = pre;
  }

  /** height buffer → height texture + central-difference normals/slope */
  async rebuildDerivedMaps(renderer: Renderer): Promise<void> {
    const res = this.res;
    const height = this.height;
    const texel = WORLD_SIZE / res;
    const kernel = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(res * res), () => {
        Return();
      });
      const x = i.mod(res).toInt();
      const y = i.div(res).toInt();
      const xm = clamp(float(x).sub(1), 0, res - 1).toInt();
      const xp = clamp(float(x).add(1), 0, res - 1).toInt();
      const ym = clamp(float(y).sub(1), 0, res - 1).toInt();
      const yp = clamp(float(y).add(1), 0, res - 1).toInt();
      const h = height.element(i).toVar();
      const hl = height.element(y.mul(res).add(xm)).toVar();
      const hr = height.element(y.mul(res).add(xp)).toVar();
      const hd = height.element(ym.mul(res).add(x)).toVar();
      const hu = height.element(yp.mul(res).add(x)).toVar();
      const n = vec3(hl.sub(hr), float(texel * 2), hd.sub(hu)).normalize();
      const slope = vec2(hl.sub(hr), hd.sub(hu)).length().div(texel * 2);
      textureStore(this.heightTex, uvec2(x.toUint(), y.toUint()), vec4(h, 0, 0, 1)).toWriteOnly();
      textureStore(
        this.normalTex,
        uvec2(x.toUint(), y.toUint()),
        vec4(n, slope),
      ).toWriteOnly();
    })().compute(res * res);
    kernel.setName('terrainDerivedMaps');
    await renderer.computeAsync(kernel);
  }

  /** world xz (m) → uv in [0,1]² over the height grid */
  uvFromWorld(p: NV2): NV2 {
    return p.div(WORLD_SIZE).add(0.5);
  }

  /**
   * Manual-bilinear height sample from the storage buffer (vertex-stage safe;
   * r32float textures are not filterable).
   */
  sampleHeight(p: NV2): NF {
    return this.sampleHeightFrom(this.height, p);
  }

  /** nearest-cell height read — for cost-insensitive paths (shadow casting) */
  sampleHeightNearest(p: NV2): NF {
    const res = this.res;
    const uv = this.uvFromWorld(p);
    const g = clamp(uv, 0, 1).mul(res);
    const x = clamp(floor(g.x), 0, res - 1).toInt();
    const y = clamp(floor(g.y), 0, res - 1).toInt();
    return this.height.element(y.mul(res).add(x));
  }

  /** same, from an arbitrary res×res float buffer (e.g. preErosion) */
  sampleHeightFrom(buf: FloatBuffer, p: NV2): NF {
    const res = this.res;
    const uv = this.uvFromWorld(p);
    const g = clamp(uv, 0, 1).mul(res).sub(0.5);
    const i0 = floor(g);
    const f = fract(g);
    const x0 = clamp(i0.x, 0, res - 1).toInt();
    const y0 = clamp(i0.y, 0, res - 1).toInt();
    const x1 = clamp(i0.x.add(1), 0, res - 1).toInt();
    const y1 = clamp(i0.y.add(1), 0, res - 1).toInt();
    const h00 = buf.element(y0.mul(res).add(x0));
    const h10 = buf.element(y0.mul(res).add(x1));
    const h01 = buf.element(y1.mul(res).add(x0));
    const h11 = buf.element(y1.mul(res).add(x1));
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
  }

  /** filtered normal+slope sample (fragment stage) */
  sampleNormalSlope(p: NV2): { normal: NV3; slope: NF } {
    const t = texture(this.normalTex, this.uvFromWorld(p));
    return { normal: t.xyz.normalize(), slope: t.w };
  }
}
