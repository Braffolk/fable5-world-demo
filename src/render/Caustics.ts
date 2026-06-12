/**
 * Procedural water caustics (Phase 6).
 *
 * A 512² tile is RE-BAKED EVERY FRAME by a compute kernel: an analytic
 * gravity-wave field (7 integer-lattice sine waves → exactly tileable,
 * real deep-water dispersion ω = √(g·k)) is pushed through the small-angle
 * refraction map, and caustic intensity is the inverse Jacobian determinant
 * of the surface→bed projection — the quantity that *defines* caustics
 * (det → 0 at projection folds = bright filaments). Closed-form Hessian,
 * zero fetches, ~0.05 ms; the filament network genuinely splits and merges
 * over time instead of scrolling a frozen pattern.
 *
 * Materials multiply `causticTint()` into their ALBEDO: the lighting
 * pipeline then scales it by sun/CSM-shadow/GI for free — a caustic inside
 * cliff shade dies with its direct light, which is the dominant physical
 * behavior (ambient caustics are below perception at our scale).
 *
 * Sampling model:
 *   depth   = waterY render field − fragment height (bilinear f32 buffer)
 *   pattern lives at the SURFACE ENTRY POINT: bed uv is back-projected
 *     along the refracted sun ray (depth-dependent parallax; the pattern
 *     slides when the sun moves)
 *   advected by the hydrology flow field with the same two-phase flowmap
 *     blend as the water ripples/foam (linear advection slides level sets
 *     into stripes — see the water gotchas in STATUS)
 *   deeper water → mip-biased blur (focusing degrades away from the focal
 *     band) + exponential contrast fade; gated by submersion and sun
 *     elevation.
 *
 * The context is a module singleton (same pattern as VegMaterials.sunU):
 * caustics are environmental state shared by terrain, rocks and debris —
 * factories self-apply via `applyCaustics()` and no-op when unset (gallery).
 */

import {
  HalfFloatType,
  LinearFilter,
  LinearMipmapLinearFilter,
  RepeatWrapping,
} from 'three';
import type { ComputeNode, MeshStandardNodeMaterial, Renderer } from 'three/webgpu';
import { StorageTexture } from 'three/webgpu';
import {
  Fn,
  If,
  Return,
  abs,
  clamp,
  exp,
  float,
  fract,
  instanceIndex,
  mix,
  positionWorld,
  refract,
  smoothstep,
  texture,
  textureStore,
  time,
  uniform,
  uvec2,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { bilerpVec2Buffer, uvToGrid } from '../gpu/BufferSample';
import { PERIOD_FBM } from '../gpu/passes/NoiseBake';
import type { NF, NV2, NV3, NV4 } from '../gpu/TSLTypes';
import type { Heightfield } from '../world/Heightfield';
import { WORLD_SIZE } from '../world/WorldConst';
import { FLOW_CYC } from './WaterMaterial';

/** world meters spanned by one caustic tile */
export const CAUSTIC_TILE = 11;
const RES = 512;

/**
 * Integer wave lattice (exact tileability), amplitudes on a k^-SPECTRUM_EXP
 * ripple spectrum, frequencies from the deep-water dispersion relation
 * slowed to 0.7× for readability. Wavelengths span 0.29–1.1 m — the band
 * that forms centimeter-to-decimeter caustic cells on stream beds.
 */
const WAVES = [
  { n: [9, 4], phi: 2.13 },
  { n: [-5, 11], phi: 5.71 },
  { n: [15, -5], phi: 0.97 },
  { n: [-13, -13], phi: 4.32 },
  { n: [22, 9], phi: 1.58 },
  { n: [-7, 26], phi: 3.05 },
  { n: [31, -20], phi: 5.02 },
  { n: [25, 17], phi: 1.21 },
  { n: [-18, 28], phi: 3.77 },
] as const;
const A0 = 0.0042; // m, amplitude of the longest wave
/**
 * Spectral slope of a·k² (the Hessian weight): 1.35 made fine cells crush
 * the large ones into uniform spaghetti; 1.9 ≈ flat focusing across the
 * band → mixed 10–40 cm cells like a real stream bed.
 */
const SPECTRUM_EXP = 1.9;
const SPEED = 0.7; // dispersion time scale

export class CausticsBake {
  readonly tex: StorageTexture;
  /**
   * Jacobian gain ≈ focusDepth·(1−1/n)·amplitude scale. The analytic sine
   * field underestimates real crest sharpness (Stokes waves fold earlier),
   * so this carries the missing nonlinearity; structure and dynamics stay
   * physical. ?caustk=N to tune live.
   */
  readonly focusK = uniform(2.0);
  private readonly kernel: ComputeNode;

  constructor() {
    const t = new StorageTexture(RES, RES);
    t.type = HalfFloatType;
    t.wrapS = RepeatWrapping;
    t.wrapT = RepeatWrapping;
    t.magFilter = LinearFilter;
    t.minFilter = LinearMipmapLinearFilter;
    t.generateMipmaps = true; // auto-rebuilt after each compute (depth blur via .bias)
    this.tex = t;

    const k0 = (2 * Math.PI * Math.hypot(WAVES[0].n[0], WAVES[0].n[1])) / CAUSTIC_TILE;
    this.kernel = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(RES * RES), () => {
        Return();
      });
      const x = i.mod(RES);
      const y = i.div(RES);
      const uv = vec2(float(x).add(0.5), float(y).add(0.5)).div(RES);

      // B = Σ aᵢ·Kᵢ⊗Kᵢ·sin θᵢ  (negated Hessian of the wave height);
      // J = I + focusK·B is the differential of the surface→bed map.
      const bxx = float(0).toVar();
      const bxy = float(0).toVar();
      const byy = float(0).toVar();
      for (const w of WAVES) {
        const kLen = (2 * Math.PI * Math.hypot(w.n[0], w.n[1])) / CAUSTIC_TILE;
        const a = A0 * (k0 / kLen) ** SPECTRUM_EXP;
        const omega = SPEED * Math.sqrt(9.81 * kLen);
        const c = a * (2 * Math.PI / CAUSTIC_TILE) ** 2; // a·|K|²/|n|²
        const theta = uv.x
          .mul(w.n[0])
          .add(uv.y.mul(w.n[1]))
          .mul(2 * Math.PI)
          .sub(time.mul(omega))
          .add(w.phi);
        const s = theta.sin().mul(c);
        bxx.addAssign(s.mul(w.n[0] * w.n[0]));
        bxy.addAssign(s.mul(w.n[0] * w.n[1]));
        byy.addAssign(s.mul(w.n[1] * w.n[1]));
      }
      const f = this.focusK as unknown as NF;
      const det = bxx
        .mul(f)
        .add(1)
        .mul(byy.mul(f).add(1))
        .sub(bxy.mul(f).pow(2));
      // intensity = 1/|det|, mean 1 by energy conservation → store the
      // excess, soft-saturated so filament cores stay HDR but bounded
      const inten = float(1).div(abs(det).max(0.06));
      const c = inten.sub(1).max(0);
      const cTone = c.div(c.mul(0.18).add(1));
      textureStore(this.tex, uvec2(x.toUint(), y.toUint()), vec4(cTone, 0, 0, 1)).toWriteOnly();
    })().compute(RES * RES);
    this.kernel.setName('caustics');
  }

  update(renderer: Renderer): void {
    renderer.compute(this.kernel);
  }
}

export interface CausticCtx {
  hf: Heightfield;
  bake: CausticsBake;
  /** unit direction TOWARD the sun (shared scene uniform) */
  sunDir: { value: { x: number; y: number; z: number } };
}

let ctx: CausticCtx | null = null;

export function setCausticContext(c: CausticCtx | null): void {
  ctx = c;
}
export function causticContext(): CausticCtx | null {
  return ctx;
}

/** water column above this fragment (m); negative above the waterline */
export function causticDepth(wp: NV3): NF {
  if (!ctx) throw new Error('caustic context not set');
  return ctx.hf.sampleWaterY(wp.xz).sub(wp.y);
}

/**
 * Caustic light factor at a submerged fragment (0 where dry/deep/night).
 * Pure expression — safe in material node graphs. Pass a precomputed
 * `causticDepth` when the caller also needs it (waterline wetness).
 */
export function causticTint(wp: NV3, depthIn?: NF): NF {
  if (!ctx) throw new Error('caustic context not set');
  const { hf, bake } = ctx;
  const flow = hf.flow;
  if (!flow) throw new Error('caustic context without hydrology');
  const depth = depthIn ?? causticDepth(wp);

  // bed point → surface entry point along the refracted sun ray
  const sunDir = vec3(ctx.sunDir as unknown as NV3);
  const rDir = refract(sunDir.negate().normalize(), vec3(0, 1, 0), 1 / 1.33) as unknown as NV3;
  const surf = wp.xz.sub(rDir.xz.mul(depth.max(0).div(rDir.y.negate().max(0.25))));

  // two-phase flowmap advection — same cycle as the water ripples/foam
  const g = uvToGrid(clamp(surf.div(WORLD_SIZE).add(0.5), 0, 1), hf.simRes);
  const flowV = bilerpVec2Buffer(flow.flowDir, hf.simRes, g);
  const vel = flowV.mul(1.9).add(vec2(0.045, 0.03));
  const ph1 = fract(time.mul(FLOW_CYC));
  const ph2 = fract(time.mul(FLOW_CYC).add(0.5));
  const w2 = abs(ph1.sub(0.5)).mul(2);
  const offA = vel.mul(ph1.div(FLOW_CYC));
  const offB = vel.mul(ph2.div(FLOW_CYC)).add(vec2(2.17, 5.31));

  // STATIC domain warp before the tile lookup: ±~0.9 m fbm-gradient
  // offsets over ~31 m features relabel space per tile instance, so the
  // repeat never lines up (user: "weird very repetitive pattern close to
  // the camera"). Time-invariant — the advection still flows through it.
  let surfW = surf;
  const nA = hf.noiseA;
  if (nA) {
    const wgrad = (
      texture(nA, surf.div(31 * PERIOD_FBM), 0) as unknown as NV4
    ).zw;
    surfW = surf.add(wgrad.clamp(-2, 2).mul(0.45)) as typeof surf;
  }

  // defocus with depth: push down the mip chain (texture is trilinear).
  // Gentle slope — clear water keeps cells crisp through ~1.5 m; ·2.0
  // washed the pattern to invisibility below knee depth.
  const lod = clamp(depth.mul(0.55), 0, 3);
  const tap = (off: NV2): NF =>
    (texture(bake.tex, surfW.sub(off).div(CAUSTIC_TILE)).bias(lod) as unknown as NV4).x;
  const pat = mix(tap(offA), tap(offB), w2);

  const submerged = smoothstep(0.025, 0.09, depth);
  // FOCAL RAMP (user: "horribly strong in shallow water"): refraction
  // needs travel distance to fold rays into filaments — a few cm of water
  // can't focus our 0.3–1.1 m surface waves. Contrast develops toward a
  // ~0.5 m focal band, then the existing deepFade defocuses it away.
  const focal = smoothstep(0.04, 0.5, depth);
  const deepFade = exp(depth.max(0).mul(-0.32));
  const sunUp = smoothstep(0.03, 0.16, sunDir.y);
  return pat.mul(submerged).mul(focal).mul(deepFade).mul(sunUp);
}

/** lit-graph triage variant: x = gated tint, y = gate product, z = raw pattern */
export function causticTintParts(wp: NV3, depthIn?: NF): NV3 {
  if (!ctx) throw new Error('caustic context not set');
  const depth = depthIn ?? causticDepth(wp);
  const sunDir = vec3(ctx.sunDir as unknown as NV3);
  const gates = smoothstep(0.025, 0.09, depth)
    .mul(smoothstep(0.04, 0.5, depth))
    .mul(exp(depth.max(0).mul(-0.32)))
    .mul(smoothstep(0.03, 0.16, sunDir.y));
  const tint = causticTint(wp, depth);
  return vec3(tint, gates, tint.div(gates.max(1e-4)));
}


/**
 * Wrap a material's albedo with the caustic factor. Call INSIDE the factory
 * (before instancing pins the vec4 shadow contract and before GI patches
 * wrap the color chain). No-op when no context is active (gallery scene).
 */
export function applyCaustics(mat: MeshStandardNodeMaterial, gain = 1.3): void {
  if (!ctx) return;
  const prev = mat.colorNode as unknown as NV3 | null;
  if (!prev) return;
  mat.colorNode = prev.mul(
    causticTint(positionWorld).mul(gain).add(1),
  ) as unknown as typeof mat.colorNode;
}
