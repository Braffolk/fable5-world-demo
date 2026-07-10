/**
 * StreamOrigin — the floating world origin for pooled GPU positions (SPEC §1
 * premise 3 + §5, S5). Estonia coords run 311-700 km from the L-EST97 anchor
 * (f32 ULP 3.1-6.3 cm → vertex/depth jitter), so every pooled GPU position is
 * stored relative to a LOD0-chunk-grid-snapped origin near the camera.
 *
 * The GENERATED world's coverage keeps the camera within ~3 km of (0,0) —
 * far under the 8 km threshold — so its origin is (0,0) FOREVER and every
 * stored word equals its absolute form bit-for-bit (the S5 identity gate).
 *
 * Rebase (rare, bounded): rewrites terrain-tile mesh-record origin words via
 * the live rewriteMeshRecord path (F-3; GeometryRegistry.rebaseTilePoolOrigins)
 * and identity-instance A-words (fartile heads today; the S7 instance pool
 * extends the same mirror-rewrite). Shadow-clip invalidation needs NO explicit
 * hook: the rebased camera/world shift moves every level's texel snap by ≫ a
 * full window, so NaniteShadowClip's own "teleport — nothing survives" full
 * path re-rasters each level under its existing stagger budget (:798-811).
 */

import type { GeometryRegistry } from './GeometryRegistry';

/** rebase when the camera strays this far from the origin (f32 ulp at 8 km
 *  ≈ 1 mm — comfortably sub-jitter; F-9 corrected 16 km ≈ 2 mm). */
const REBASE_DIST_M = 8192;

export class StreamOrigin {
  private ox = 0;
  private oz = 0;
  private readonly snapM: number;
  private nRebases = 0;

  /** snapM = the LOD0 chunk stride (grid.chunkMeters) — origins sit on the
   *  chunk lattice so chunk-local record positions re-anchor exactly. */
  constructor(snapM: number) {
    this.snapM = snapM;
  }

  get x(): number {
    return this.ox;
  }
  get z(): number {
    return this.oz;
  }
  get rebases(): number {
    return this.nRebases;
  }

  /** check the camera distance; on breach snap the origin to the LOD0 grid
   *  near the camera and rewrite everything origin-relative that exists today
   *  (terrain tile origins + identity A-words). Returns the delta or null. */
  maybeRebase(camX: number, camZ: number, reg: GeometryRegistry | null): { dx: number; dz: number } | null {
    const dx0 = camX - this.ox;
    const dz0 = camZ - this.oz;
    if (dx0 * dx0 + dz0 * dz0 < REBASE_DIST_M * REBASE_DIST_M) return null;
    const nx = Math.round(camX / this.snapM) * this.snapM;
    const nz = Math.round(camZ / this.snapM) * this.snapM;
    const dx = nx - this.ox;
    const dz = nz - this.oz;
    if (dx === 0 && dz === 0) return null;
    this.ox = nx;
    this.oz = nz;
    this.nRebases++;
    reg?.rebaseTilePoolOrigins(dx, dz);
    reg?.rebaseInstanceOrigins(dx, dz);
    // eslint-disable-next-line no-console
    console.log(`[laas] stream origin rebased to (${nx}, ${nz}) — Δ(${dx}, ${dz}); shadow clips full-refill via their teleport path`);
    return { dx, dz };
  }

  counters(): Record<string, number> {
    return { 'stream.origin.x': this.ox, 'stream.origin.z': this.oz, 'stream.origin.rebases': this.nRebases };
  }
}
