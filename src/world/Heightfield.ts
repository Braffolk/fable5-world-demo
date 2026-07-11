/**
 * Heightfield — owner of all terrain GPU state. Orchestrates the generation
 * passes (synthesis → erosion → hydrology → classification) and exposes
 * buffers/textures + TSL sampling helpers to the rest of the engine.
 *
 * Layout: row-major res×res grids; texel (x,y) ↔ world
 * ((x+0.5)/res − 0.5)·WORLD_SIZE on both axes (x→world x, y→world z).
 */

import { HalfFloatType } from 'three';
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
  textureStore,
  uvec2,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { activeTier } from '../core/Quality';
import type { WorldSeed } from '../core/Seed';
import { bilerpFloatBuffer, uvToGrid } from '../gpu/BufferSample';
import { bakeNoiseTextures } from '../gpu/passes/NoiseBake';
import type { NF, NI, NV2 } from '../gpu/TSLTypes';
import { runBiomeSnow } from '../gpu/passes/BiomeSnow';
import { runErosion } from '../gpu/passes/Erosion';
import { runFlowRivers, type FlowResult } from '../gpu/passes/FlowRivers';
import {
  labelGroup,
  runHeightSynthesis,
  type FloatBuffer,
  type SynthesisResult,
} from '../gpu/passes/HeightSynthesis';
import { makeMacroParams, type MacroParams } from './MacroMap';
import { WORLD_SIZE, qualityConfig, type QualityConfig } from './WorldConst';

export type ProgressFn = (p: number, msg: string) => void;

/** dry-water sentinel (m) for the streamed-world stub: far below any bed, so
 *  max(ground, waterAt) never lifts the camera and no water surface renders
 *  (matches PlaneFill.WATER_DRY_SENTINEL). */
const WATER_DRY_M = -1e4;

export class Heightfield {
  readonly cfg: QualityConfig;
  readonly mp: MacroParams;
  readonly res: number;

  /** final height (m), res×res storage buffer — single source of truth during
   *  GENERATION + the boot consumers inside source.open (scatter/classify);
   *  runtime terrain reads live on the TerrainField planes. GPU side released
   *  by releaseBootGpuSet() once boot bakes complete. */
  readonly height: SynthesisResult['height'];
  /** full-res synthesis hardness — generation-internal (erosion runs on the
   *  SIM-res copy); retained only so releaseBootGpuSet can free its GPU side */
  readonly hardness: SynthesisResult['hardness'];
  /** erosion by-products at sim res — boot-only; freed by releaseBootGpuSet */
  simWater: FloatBuffer | null = null;
  simSediment: FloatBuffer | null = null;
  simRes = 0;
  /** hydrology outputs at sim res */
  flow: FlowResult | null = null;
  /** renderable water surface (m) at sim res: carved bed + riverDepth at
   *  water cells; DRY cells hold simBed − 2 so bilinear shorelines cut
   *  below the banks (f32 buffer). BOOT-ONLY since the S9 water port: read
   *  back into cpuWaterY (which fills the generated source's TerrainField
   *  water plane), then freed by releaseBootGpuSet — the runtime water
   *  material samples that plane, not this buffer. */
  waterY: FloatBuffer | null = null;
  /** rgba16f at sim res: moisture, flowStrength, riverDepth, waterSurface W */
  fieldsTex: StorageTexture | null = null;
  /** rgba8 at full res: biomeId/8, snow, vegDensity, rockExposure */
  biomeTex: StorageTexture | null = null;
  /** CPU height mirror for camera clamping / tools (filled by readback) */
  cpuHeights: Float32Array | null = null;
  /** CPU waterY mirror (sim res) — the generated WorldSource bins it into the
   *  TerrainField water plane fill (fetch('water')); runtime water/camera reads
   *  live on that plane (field.waterAt / field.fieldWaterY). */
  cpuWaterY: Float32Array | null = null;
  private bootGpuReleased = false;

  /** rgba16f: xyz = world-space normal, w = slope (rise/run). BOOT-ONLY:
   *  consumed by the biome classification (BiomeSnow) + scatter placement
   *  inside source.open; every runtime consumer derives normals in-shader
   *  from the TerrainField height planes (S3b). Released post-boot. */
  readonly normalTex: StorageTexture;
  /** baked tileable noise (see NoiseBake channel map) — materials sample these */
  noiseA: StorageTexture | null = null;
  noiseB: StorageTexture | null = null;

  private constructor(
    cfg: QualityConfig,
    mp: MacroParams,
    synth: SynthesisResult,
    normalTex: StorageTexture,
  ) {
    this.cfg = cfg;
    this.mp = mp;
    this.res = synth.res;
    this.height = synth.height;
    this.hardness = synth.hardness;
    this.normalTex = normalTex;
  }

  static async generate(
    renderer: Renderer,
    seed: WorldSeed,
    progress: ProgressFn,
  ): Promise<Heightfield> {
    const cfg = qualityConfig(activeTier());
    const mp = makeMacroParams(seed);

    progress(0.04, `terrain: synthesizing ${cfg.heightRes}² heightfield`);
    const synth = await runHeightSynthesis(renderer, cfg.heightRes, mp);

    const normalTex = new StorageTexture(cfg.heightRes, cfg.heightRes);
    normalTex.name = 'hfNormalTex';
    normalTex.type = HalfFloatType;
    normalTex.generateMipmaps = false;

    const hf = new Heightfield(cfg, mp, synth, normalTex);

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

  /**
   * Streamed-world stub (SPEC-STREAMING-WORLD S6): a Heightfield with NO
   * procedural terrain — the height/biome/water DATA lives on the TerrainField
   * planes fed by the WorldSource. This exists ONLY to satisfy the boot handles
   * the subsystems still read off `hf` (they never learn which source feeds the
   * world — law 3): the REAL procedural noise bake (wind/froxels/resolve need
   * it, it is world-data-independent), plus tiny placeholder hydrology/water
   * buffers so water + caustics construct and run (they render nothing — the
   * water is the dry sentinel everywhere — until the S9 water port reads the
   * TerrainField water plane). No boot GPU set to release (returns 0).
   */
  static async forStreamedWorld(renderer: Renderer, seed: WorldSeed): Promise<Heightfield> {
    const cfg = qualityConfig(activeTier());
    const mp = makeMacroParams(seed);
    // the private ctor wants a SynthesisResult + normalTex; both are boot-only
    // (unused at runtime and never freed here) so a 1-texel placeholder suffices.
    const synth: SynthesisResult = {
      res: cfg.heightRes,
      height: instancedArray(1, 'float') as SynthesisResult['height'],
      hardness: instancedArray(1, 'float') as SynthesisResult['hardness'],
    };
    const normalTex = new StorageTexture(1, 1);
    normalTex.name = 'hfNormalTexStub';
    normalTex.type = HalfFloatType;
    normalTex.generateMipmaps = false;
    const hf = new Heightfield(cfg, mp, synth, normalTex);
    hf.bootGpuReleased = true; // nothing procedural to free — releaseBootGpuSet is a no-op

    // REAL noise (procedural, world-independent) — wind/froxels/resolve read it
    const noise = await bakeNoiseTextures(renderer);
    hf.noiseA = noise.texA;
    hf.noiseB = noise.texB;

    // placeholder hydrology at a small sim res — ZERO flow for the water
    // material's ripple/foam advection + the caustic drift. The water SURFACE
    // reads the streamed TerrainField water plane (S9 port), where Estonia's
    // real waterY lands — nothing here holds a water level.
    const simRes = 256;
    hf.simRes = simRes;
    const n = simRes * simRes;
    const zeroF = (): FloatBuffer => instancedArray(n, 'float') as FloatBuffer;
    const dryY = await Heightfield.fillConst(renderer, n, WATER_DRY_M);
    hf.flow = {
      waterSurface: zeroF(),
      flowStrength: zeroF(),
      riverDepth: zeroF(),
      flowDir: instancedArray(n, 'vec2') as FlowResult['flowDir'],
      moisture: zeroF(),
      waterYRaw: dryY,
    };
    // non-null so the registry's boot-order guard passes; its CONTENTS are never
    // read in clip mode (terrain streams through the brain, not cpuHeights).
    hf.cpuHeights = new Float32Array(1);
    return hf;
  }

  /** a `count`-element f32 storage buffer filled with a constant (stub water). */
  private static async fillConst(renderer: Renderer, count: number, val: number): Promise<FloatBuffer> {
    const out = instancedArray(count, 'float');
    const kernel = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(count), () => {
        Return();
      });
      out.element(i).assign(float(val));
    })().compute(count);
    kernel.setName('streamWaterDryFill');
    await renderer.computeAsync(kernel);
    return out as FloatBuffer;
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
      await renderer.computeAsync(
        labelGroup([mkSmooth(out, tmp), mkSmooth(tmp, out)], 'hfWaterYSmooth'),
      );
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
    await renderer.computeAsync(labelGroup([cliffK, copyK], 'hfWaterYCliff'));
    return out;
  }

  /** pack sim-res hydrology fields into a filterable rgba16f texture */
  private async buildFieldsTex(renderer: Renderer): Promise<void> {
    const flow = this.flow;
    if (!flow) return;
    const res = this.simRes;
    const tex = new StorageTexture(res, res);
    tex.name = 'hfFieldsTex';
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
   */
  private async composeEroded(
    renderer: Renderer,
    preSim: FloatBuffer,
    erodedSim: FloatBuffer,
  ): Promise<void> {
    const res = this.res;
    const simRes = this.simRes;
    const kernel = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(res * res), () => {
        Return();
      });
      const x = i.mod(res);
      const y = i.div(res);
      const h = this.height.element(i).toVar();
      const uv = vec2(float(x).add(0.5), float(y).add(0.5)).div(res);
      const g = uvToGrid(uv, simRes);
      const macroEroded = bilerpFloatBuffer(erodedSim, simRes, g);
      const macroPre = bilerpFloatBuffer(preSim, simRes, g);
      this.height.element(i).assign(macroEroded.add(h.sub(macroPre)));
    })().compute(res * res);
    kernel.setName('erosionCompose');
    await renderer.computeAsync(kernel);
  }

  /** height buffer → central-difference normals/slope (boot classification +
   *  scatter input; runtime normals derive in-shader from the height planes) */
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
      const hl = height.element(y.mul(res).add(xm)).toVar();
      const hr = height.element(y.mul(res).add(xp)).toVar();
      const hd = height.element(ym.mul(res).add(x)).toVar();
      const hu = height.element(yp.mul(res).add(x)).toVar();
      const n = vec3(hl.sub(hr), float(texel * 2), hd.sub(hu)).normalize();
      const slope = vec2(hl.sub(hr), hd.sub(hu)).length().div(texel * 2);
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
   * r32float textures are not filterable). BOOT-ONLY consumer surface (scatter
   * placement inside source.open) — the buffer is freed by releaseBootGpuSet.
   */
  sampleHeight(p: NV2): NF {
    const res = this.res;
    const uv = this.uvFromWorld(p);
    const g = clamp(uv, 0, 1).mul(res).sub(0.5);
    const i0 = floor(g);
    const f = fract(g);
    const x0 = clamp(i0.x, 0, res - 1).toInt();
    const y0 = clamp(i0.y, 0, res - 1).toInt();
    const x1 = clamp(i0.x.add(1), 0, res - 1).toInt();
    const y1 = clamp(i0.y.add(1), 0, res - 1).toInt();
    const h00 = this.height.element(y0.mul(res).add(x0));
    const h10 = this.height.element(y0.mul(res).add(x1));
    const h01 = this.height.element(y1.mul(res).add(x0));
    const h11 = this.height.element(y1.mul(res).add(x1));
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
  }

  /**
   * S3b/S4 finale: free the BOOT-ONLY GPU field set once every runtime consumer
   * reads the TerrainField planes — normalTex (classification/scatter input),
   * the full-res height + hardness buffers, the sim-res erosion scratch, and
   * since S4 biomeTex + fieldsTex (their last runtime readers — ProbeGI albedo,
   * Particles snow, Froxels moisture — sample the streamed biome/fields planes;
   * the remaining reads run inside source.open, before this call). Since the S9
   * water port, waterY joins the set — its last reader was the water material,
   * which now samples the TerrainField water plane (fed by cpuWaterY, already
   * read back). The flow field STAYS (ripple/foam advection + caustic drift read
   * it live). Call strictly AFTER boot bakes complete.
   * Safe under ?profile=1: these resources are not in the swap handoff and
   * nothing on the render device references them, so the loading-device copies
   * just die early. Idempotent; returns the MB freed for the boot ledger.
   */
  releaseBootGpuSet(renderer: Renderer): number {
    if (this.bootGpuReleased) return 0;
    this.bootGpuReleased = true;
    const backend = (
      renderer as unknown as {
        backend: {
          get(o: unknown): { buffer?: { destroy(): void } };
          destroyAttribute(a: unknown): void;
        };
      }
    ).backend;
    let bytes = 0;
    const killBuf = (node: { value: unknown } | null, byteLen: number): void => {
      if (!node) return;
      const attr = node.value;
      if (backend.get(attr).buffer) backend.destroyAttribute(attr);
      bytes += byteLen;
    };
    const r2 = this.res * this.res;
    const s2 = this.simRes * this.simRes;
    killBuf(this.height as unknown as { value: unknown }, r2 * 4);
    killBuf(this.hardness as unknown as { value: unknown }, r2 * 4);
    killBuf(this.simWater as unknown as { value: unknown } | null, s2 * 4);
    killBuf(this.simSediment as unknown as { value: unknown } | null, s2 * 4);
    killBuf(this.waterY as unknown as { value: unknown } | null, s2 * 4);
    this.simWater = null;
    this.simSediment = null;
    this.waterY = null;
    this.normalTex.dispose();
    bytes += r2 * 8; // rgba16f
    if (this.biomeTex) {
      this.biomeTex.dispose();
      this.biomeTex = null;
      bytes += r2 * 4; // rgba8
    }
    if (this.fieldsTex) {
      this.fieldsTex.dispose();
      this.fieldsTex = null;
      bytes += s2 * 8; // rgba16f
    }
    return bytes / 2 ** 20;
  }
}
