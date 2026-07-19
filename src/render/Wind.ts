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
 * Context is a module singleton like sunU: set it before any
 * vegetation material builds; absent context (gallery) → no wind.
 */

import { Vector2 } from 'three';
import type { StorageTexture } from 'three/webgpu';
import { float, texture, time, vec2 } from 'three/tsl';
import type { NF, NV2, NV4 } from '../gpu/TSLTypes';
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

