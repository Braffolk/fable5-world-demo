/**
 * ?rscale=S — internal render scale (0.25..1, default 1 = today's frame).
 * The scene pass and every screen-space buffer that must match it
 * pixel-for-pixel (nanite vis/HZB/grass, half-res MRT, water snapshot,
 * τ anchor heights) render at floor(native × S); the TRAA fork upscales
 * temporally back to the native drawing buffer (TSR: jittered internal-res
 * input, history + resolve at NATIVE res), so bloom/grade/canvas stay
 * native. τ is device-px-relative (projK ∝ renderHeight), so the LOD cut
 * coarsens with the internal res automatically — intended, part of the win.
 *
 * internalSize() mirrors PassNode's effective-size math EXACTLY
 * (floor(logicalSize × pixelRatio × resolutionScale), r184 PassNode.setSize):
 * the nanite vis buffers are indexed by scene-pass fragCoord, so the two must
 * never disagree by a pixel — flooring the drawing buffer first and scaling
 * after CAN differ from PassNode's single-floor, so derive from logical size.
 */

import { Vector2 } from 'three';

const rawScale = Number(new URLSearchParams(window.location.search).get('rscale') ?? '1');
export const RSCALE: number =
  Number.isFinite(rawScale) && rawScale > 0 ? Math.min(1, Math.max(0.25, rawScale)) : 1;

interface SizedRenderer {
  getSize(target: Vector2): Vector2;
  getPixelRatio(): number;
}

const _logical = new Vector2();

/** internal render resolution (device px) — use INSTEAD of getDrawingBufferSize
 *  for every buffer that must match the scene pass exactly */
export function internalSize(renderer: SizedRenderer, target: Vector2): Vector2 {
  renderer.getSize(_logical);
  const pr = renderer.getPixelRatio();
  return target.set(
    Math.floor(_logical.x * pr * RSCALE),
    Math.floor(_logical.y * pr * RSCALE),
  );
}
