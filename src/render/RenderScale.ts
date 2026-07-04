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

/**
 * AUTO mode (user rule 2026-07-04): when ?rscale is NOT in the URL, cap the
 * internal render AREA at the 1920×1080 pixel budget (2.07 Mpx) — if the
 * native drawing buffer has more pixels, S = sqrt(1920·1080 / (W·H)), so every
 * aspect ratio keeps its own shape and gets the same pixel count as 1080p
 * (an ultrawide must not be letterbox-fit by its width). Explicit ?rscale
 * (including =1) always wins. Mirrors Engine's boot sizing exactly
 * (innerWidth/innerHeight × (?dpr ?? min(devicePixelRatio, 1.5))); boot-fixed
 * like the param — a post-boot window resize does not re-derive it (anchorH
 * sits in the bootcache key).
 */
function resolveScale(): number {
  const q = new URLSearchParams(window.location.search);
  const raw = q.get('rscale');
  if (raw !== null) {
    const v = Number(raw);
    return Number.isFinite(v) && v > 0 ? Math.min(1, Math.max(0.25, v)) : 1;
  }
  const dprQ = q.get('dpr');
  const dpr =
    dprQ !== null && Number(dprQ) > 0 ? Number(dprQ) : Math.min(window.devicePixelRatio, 1.5);
  const px = window.innerWidth * dpr * (window.innerHeight * dpr);
  if (px <= 1920 * 1080) return 1;
  return Math.min(1, Math.max(0.25, Math.sqrt((1920 * 1080) / px)));
}
export const RSCALE: number = resolveScale();

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
