/**
 * VegInstance — turns a Phase-4 vegetation material into a GPU-driven
 * instanced draw: per-instance transform (pos/scale/yaw/lean-shear) is read
 * from the scatter buffers through a cull-compacted index list, LOD ring
 * transitions are dithered (IGN screen noise vs distance fade), and a small
 * per-instance tint breaks population uniformity beyond the per-vertex vdata
 * jitter (per-instance variation law).
 *
 * Normals skip the yaw rotation deliberately: trunk normals are quasi-radial
 * and card normals are crown-sphere-bent (also radial), so rotating positions
 * but not normals is visually lossless at ring distances. Lean shear ≤ ~7° —
 * likewise skipped for normals.
 */

import type { MeshStandardNodeMaterial } from 'three/webgpu';
import type { StorageBufferNode } from 'three/webgpu';
import {
  Discard,
  Fn,
  cameraPosition,
  float,
  instanceIndex,
  interleavedGradientNoise,
  mix,
  normalLocal,
  positionLocal,
  screenCoordinate,
  smoothstep,
  uint,
  uniform,
  varying,
  vec3,
  vec4,
} from 'three/tsl';
import type { NF, NU, NV3, NV4 } from '../gpu/TSLTypes';

export interface RingFade {
  /** dither IN as distance exceeds this (far ring of a boundary) */
  fadeInAt?: number;
  /** dither OUT as distance exceeds this (near ring of a boundary) */
  fadeOutAt?: number;
  band: number;
}

export interface InstanceBinding {
  bufA: StorageBufferNode<'vec4'>;
  bufB: StorageBufferNode<'vec4'>;
  compact: StorageBufferNode<'uint'>;
  /** offset of this draw's region in the compact list */
  groupBase: number;
  fade?: RingFade | null;
  /** per-instance tint strength (0 disables) */
  tint?: number;
}

/** cheap pcg-ish hash of the instance slot → 0..1 (pure expression) */
export function slotHash(slot: NU, salt: number): NF {
  const a = slot.add(uint(salt)).mul(uint(747796405)).add(uint(2891336453));
  const b = a.shiftRight(a.shiftRight(uint(28)).add(uint(4))).bitXor(a).mul(uint(277803737));
  const c = b.shiftRight(uint(22)).bitXor(b);
  return float(c.bitAnd(uint(0xffffff))).div(16777216);
}

export interface FetchedInstance {
  /** (x, y, z, scale) */
  A: NV4;
  /** (yaw, leanX, leanZ, idF) */
  B: NV4;
  slot: NU;
}

/** vertex-stage fetch of the instance record through the compact list */
export function fetchInstance(bind: InstanceBinding): FetchedInstance {
  const base = uniform(uint(bind.groupBase));
  const slot = bind.compact.element(
    instanceIndex.add(base as unknown as NU),
  ) as unknown as NU;
  return {
    A: bind.bufA.element(slot) as unknown as NV4,
    B: bind.bufB.element(slot) as unknown as NV4,
    slot,
  };
}

/** dithered LOD crossfade: discard by IGN screen noise vs distance fade */
export function applyDitherFade(
  mat: MeshStandardNodeMaterial,
  dist: NF,
  fade: RingFade,
): void {
  let fadeExpr: NF = float(1);
  if (fade.fadeInAt !== undefined) {
    fadeExpr = fadeExpr.mul(
      smoothstep(fade.fadeInAt - fade.band, fade.fadeInAt + fade.band, dist),
    );
  }
  if (fade.fadeOutAt !== undefined) {
    fadeExpr = fadeExpr.mul(
      float(1).sub(
        smoothstep(fade.fadeOutAt - fade.band, fade.fadeOutAt + fade.band, dist),
      ),
    );
  }
  const fadeV = varying(fadeExpr);
  const prev = mat.colorNode as unknown as NV3 | null;
  mat.colorNode = Fn(() => {
    Discard(fadeV.lessThanEqual(interleavedGradientNoise(screenCoordinate.xy)));
    return prev ?? vec3(1, 0, 1);
  })();
}

/** per-instance hue/value jitter on top of the per-vertex vdata jitter */
export function applyInstanceTint(
  mat: MeshStandardNodeMaterial,
  slot: NU,
  tintK: number,
): void {
  if (tintK <= 0) return;
  const h1 = varying(slotHash(slot, 17));
  const h2 = varying(slotHash(slot, 91));
  const warmCool = mix(
    vec3(1 + tintK, 1, 1 - tintK * 0.8),
    vec3(1 - tintK * 0.8, 1, 1 + tintK),
    h1,
  );
  const value = h2.mul(tintK * 1.6).add(1 - tintK * 0.8);
  const prev = mat.colorNode as unknown as NV3 | null;
  if (prev) mat.colorNode = prev.mul(warmCool).mul(value);
}

export interface InstancedHandles {
  /** world-space instance origin (vertex stage) */
  origin: NV3;
  slot: NU;
  dist: NF;
}

/**
 * Rewires `mat` for compacted-indirect instancing (transform + fade + tint).
 * Returns vertex-stage handles for callers building further on top.
 */
export function instanceVeg(
  mat: MeshStandardNodeMaterial,
  bind: InstanceBinding,
): InstancedHandles {
  const { A, B, slot } = fetchInstance(bind);

  const c = B.x.cos();
  const s = B.x.sin();
  const ls = positionLocal.mul(A.w);
  const rx = ls.x.mul(c).add(ls.z.mul(s));
  const rz = ls.z.mul(c).sub(ls.x.mul(s));
  // lean as shear: keeps the base planted, tips the crown
  const px = rx.add(B.y.mul(ls.y));
  const pz = rz.add(B.z.mul(ls.y));
  const wpos = vec3(px, ls.y, pz).add(A.xyz);
  // Normals MUST rotate with the instance (same mechanism as three's
  // InstanceNode: assign normalLocal before returning the position). With
  // unrotated normals a yawed trunk is lit from the wrong side — reads as
  // inverted faces ("seeing the far side of the trunk").
  mat.positionNode = Fn(() => {
    const n = vec3(
      normalLocal.x.mul(c).add(normalLocal.z.mul(s)),
      normalLocal.y,
      normalLocal.z.mul(c).sub(normalLocal.x.mul(s)),
    ).toVar();
    normalLocal.assign(n);
    return wpos;
  })();
  // shadow-map pass builds its own position pipeline — feed it the same
  // instance transform or casters render at the pool origin
  (mat as unknown as { castShadowPositionNode: unknown }).castShadowPositionNode = wpos;

  const dist = A.xyz.sub(cameraPosition).length();

  const f = bind.fade;
  if (f && (f.fadeInAt !== undefined || f.fadeOutAt !== undefined)) {
    applyDitherFade(mat, dist, f);
  }
  applyInstanceTint(mat, slot, bind.tint ?? 0.12);

  // Shadow-pass contract: three derives the caster's alpha from colorNode.a
  // and copies alphaTest over — a vec3 colorNode yields a bogus alpha below
  // the threshold and every shadow fragment silently discards. Pin alpha=1
  // and express alpha-tested cutouts through maskShadowNode instead.
  const rgb = mat.colorNode as unknown as NV3 | null;
  if (rgb) mat.colorNode = vec4(rgb, 1);
  const op = mat.opacityNode as unknown as NF | null;
  if (op) {
    (mat as unknown as { maskShadowNode: unknown }).maskShadowNode = op.greaterThan(
      Math.max(mat.alphaTest, 0.1),
    );
  }

  return { origin: A.xyz, slot, dist };
}
