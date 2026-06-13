/**
 * Hierarchical wind (Phase 6, reworked after user feedback) — one global
 * wind field sampled by all vegetation (spec GPU-systems #11).
 *
 * Field model: uniform direction + traveling gust fronts (two advected fbm
 * octaves: 85 m fronts + 17 m busy detail), sheltered under dense canopy.
 *
 * Fake-skeletal hierarchy (all vertex-stage, shadows share the nodes).
 * Strong wind makes everything deflect MORE, never oscillate faster:
 *   1. mean LEAN downwind ∝ strength² — trunk-bend cantilever profile
 *      (y/(y+h0))², streamlining the whole plant into the wind.
 *   2. slow SWAY rocking around that lean at a per-instance NATURAL
 *      frequency (0.15–0.45 Hz · freq / √scale — big trees swing slower).
 *      Gusts drive the AMPLITUDE only; the frequency of every oscillator
 *      here is constant in time. Never multiply `time` by a time-varying
 *      frequency: phase = t·f(t) slews by t·Δf, which grows with session
 *      time and explodes into chaotic fast jitter exactly where gust
 *      variance is highest (the cliff-top "bugged out" tree).
 *      A second axis at ×1.31 the rate draws Lissajous ellipses instead of
 *      a metronome line; per-instance rates kill the shared tempo.
 *   3. branch SECONDARY motion: flex-scaled deflection driven by the gust
 *      field sampled a few meters DOWNWIND = the front that hit the trunk
 *      ~½ s ago — branches visibly lag and catch up (skeletal feel).
 *   4. leaf/card micro-flutter: APERIODIC, from the advected fbm gradient
 *      channels (zero-mean, two independent axes per tap) — no sines, no
 *      beat pattern; decorrelated per vertex by the baked vdata.z phase.
 *      Fades out by ~120 m (sub-pixel at range, only feeds TRAA shimmer).
 *   5. grass: cantilever bend (tip², GroundRing) + fine shimmer, with the
 *      same lean² rule so strong wind flattens the sward.
 *
 * Context is a module singleton like sunU/caustics: set it before any
 * vegetation material builds; absent context (gallery) → no wind.
 */

import { Vector2 } from 'three';
import type { StorageTexture } from 'three/webgpu';
import { attribute, float, texture, time, vec2, vec3 } from 'three/tsl';
import type { NF, NV2, NV3, NV4 } from '../gpu/TSLTypes';
import { PERIOD_FBM } from '../gpu/passes/NoiseBake';
import { canopyAt } from '../gpu/passes/Scatter';
import { runiform } from '../gpu/RenderUniform';

/** global wind state (uniforms — live-tunable; ?wind=N sets strength) */
export const windU = {
  /** unit horizontal direction the wind BLOWS TOWARD */
  dir: runiform(new Vector2(0.78, 0.63).normalize()),
  /** 0 = still air, 1 = strong breeze (≈ Beaufort 6 visually) */
  strength: runiform(0.45),
};

/** gust front advection speed (m/s) — shared by the lag offset below */
const GUST_SPEED = 10.5;
/** branch response lag ≈ 0.5 s, expressed as a downwind sample offset */
const LAG_M = 5.5;

export interface WindCtx {
  noiseA: StorageTexture;
  canopyTex: StorageTexture | null;
}

let ctx: WindCtx | null = null;

export function setWindContext(c: WindCtx | null): void {
  ctx = c;
}
export function windContext(): WindCtx | null {
  return ctx;
}

/**
 * Traveling gust factor at a world position, 0..1 (vertex-stage safe:
 * explicit mip level). Two advected fbm octaves — fronts + busy detail.
 */
export function gustAt(xz: NV2): NF {
  if (!ctx) throw new Error('wind context not set');
  const d = vec2(windU.dir as unknown as NV2);
  const p1 = xz.sub(d.mul(time.mul(GUST_SPEED))).div(85 * PERIOD_FBM);
  const g1 = (texture(ctx.noiseA, p1, 0) as unknown as NV4).y;
  const p2 = xz.sub(d.mul(time.mul(7.2))).div(17 * PERIOD_FBM);
  const g2 = (texture(ctx.noiseA, p2, 0) as unknown as NV4).y;
  return g1.mul(0.6).add(g2.mul(0.4));
}

/** branch response lag in metres (≈0.5 s downwind) — exported for the nanite
 *  trunk channel, which ports this assembly into the shared vertex fetch */
export const WIND_LAG_M = LAG_M;

/** the 85 m front octave only, sampled `lagM` downwind = `lagM/speed` s ago */
export function gustLagAt(xz: NV2, lagM: number): NF {
  if (!ctx) throw new Error('wind context not set');
  const d = vec2(windU.dir as unknown as NV2);
  const p = xz
    .add(d.mul(lagM))
    .sub(d.mul(time.mul(GUST_SPEED)))
    .div(85 * PERIOD_FBM);
  return (texture(ctx.noiseA, p, 0) as unknown as NV4).y;
}

/** canopy shelter: interiors see ~40% of the open-field wind */
export function windExposure(xz: NV2): NF {
  if (!ctx) throw new Error('wind context not set');
  if (!ctx.canopyTex) return float(1);
  return float(1).sub(canopyAt(ctx.canopyTex, xz).mul(0.6));
}

export interface WindBind {
  /** overall response scale (1 = trees) */
  k: number;
  /** natural-frequency multiplier (understory rocks faster than trees) */
  freq: number;
  /** trunk-bend profile knee (m): ~6 for trees, ~0.9 for shrubs */
  h0: number;
}

export interface WindVertexArgs {
  /** world-space instance origin */
  origin: NV3;
  /** scaled local-space height of the vertex above the instance base (m) */
  localY: NF;
  /** per-instance uniform scale (A.w) — bigger plants swing slower */
  scale: NF;
  /** per-instance hash 0..1 — phase + natural-frequency jitter */
  instPhase: NF;
  /** main-camera distance (vegViewPos-based, NEVER TSL cameraPosition) */
  dist: NF;
  bind: WindBind;
}

/**
 * Per-vertex wind displacement for instanced vegetation. Reads the baked
 * vdata flex (y: 0 trunk base → 1 branch tips) / phase (z: along the
 * branch run) attributes. Pure expression chain — material node graphs
 * have no Fn stack, so no toVar/assign in here.
 */
export function vegWindOffset(a: WindVertexArgs): NV3 {
  if (!ctx) throw new Error('wind context not set');
  const vd = attribute('vdata', 'vec4') as unknown as NV4;
  const flex = vd.y;
  const d = vec2(windU.dir as unknown as NV2);
  const s = windU.strength as unknown as NF;
  const { k, freq, h0 } = a.bind;

  const e = windExposure(a.origin.xz);
  const g = gustAt(a.origin.xz);
  const gL = gustLagAt(a.origin.xz, LAG_M);

  // trunk-bend cantilever profile: 0 at the base, ~1 near a tall crown top
  // (asymptotic — no per-species height needed); tips streamline a little
  // extra via flex
  const yn = a.localY.div(a.localY.add(h0));
  const prof = yn.mul(yn).mul(1.7).add(flex.mul(0.3)).min(1.6);

  // everything stops by the impostor band — impostors are rigid, and a
  // displaced ring crossfading into a static one shimmers at the boundary
  const farAtten = float(1).sub(a.dist.sub(380).div(100).clamp(0, 1));
  const eks = e.mul(k).mul(farAtten);

  // 1) mean lean ∝ strength², modulated by the front field (slow)
  const lean = s.mul(s).mul(g.mul(0.9).add(0.5)).mul(eks).mul(1.1).mul(prof);

  // 2) sway at the per-instance natural frequency; gusts scale AMPLITUDE
  const fJit = a.instPhase.mul(7.31).fract();
  const natW = fJit
    .mul(0.3)
    .add(0.15)
    .mul(6.2832 * freq)
    .div(a.scale.max(0.25).sqrt());
  const ph = a.instPhase.mul(6.2832);
  const swayA = s.mul(g.mul(0.75).add(0.25)).mul(eks).mul(0.5).mul(prof);
  const sway = time.mul(natW).add(ph).sin().mul(swayA);
  const swayX = time.mul(natW.mul(1.31)).add(ph.mul(1.7)).sin().mul(swayA).mul(0.45);

  // 3) branch secondary motion: lagged front, flex-weighted, overshoots rest
  const brAtten = float(1).sub(a.dist.sub(160).div(140).clamp(0, 1));
  const branch = gL.sub(0.45).mul(flex).mul(s).mul(eks).mul(0.55).mul(brAtten);

  // 4) aperiodic micro-flutter: advected fbm GRADIENTS (zero-mean, two
  // independent axes in one tap), decorrelated per vertex by vdata.z.
  // Leaf flutter is a SHIMMER, not a shake: a few cm at the tips, features
  // ~6 m advected slowly (~0.75 Hz) — the first cut (±12 cm, 3–4 Hz
  // decorrelation) read as "leaves shaking wildly" (user)
  const flutAtten = float(1).sub(a.dist.sub(40).div(80).clamp(0, 1));
  const pF = a.origin.xz
    .add(vd.z.mul(vec2(37.1, 17.7)))
    .add(vec2(a.instPhase.mul(91), 0))
    .sub(d.mul(time.mul(4.5)))
    .div(6 * PERIOD_FBM);
  const fl = texture(ctx.noiseA, pF, 0) as unknown as NV4;
  const flutA = s.mul(g.mul(0.7).add(0.3)).mul(eks).mul(flex).mul(0.07).mul(flutAtten);
  const flutD = fl.z.clamp(-1.2, 1.2).mul(flutA);
  const flutP = fl.w.clamp(-1.2, 1.2).mul(flutA);

  const along = lean.add(sway).add(branch).add(flutD);
  const across = swayX.add(flutP);
  // cantilever arc: tips dip slightly as they deflect
  const dy = along.abs().add(across.abs()).mul(flex).mul(-0.2);
  return vec3(
    d.x.mul(along).sub(d.y.mul(across)),
    dy,
    d.y.mul(along).add(d.x.mul(across)),
  );
}
