/**
 * Shared N2+ pipeline pieces: per-frame camera uniforms and the instance
 * transform math (InstanceStream contract, F8: scale → yaw → lean shear →
 * translate; wind enters at the N3 transform stage — cull bounds absorb it
 * via swayPad instead).
 *
 * All TSL here is pure-expression or used inside the callers' Fn() stacks.
 */

import { Frustum, Matrix4, Vector3, Vector4 } from 'three';
import type { PerspectiveCamera } from 'three';
import { uint, vec3 } from 'three/tsl';
import type { NF, NU, NV3, NV4 } from '../gpu/TSLTypes';
import {
  toF,
  uniformArrV4,
  uniformF,
  uniformMat4,
  uniformV3,
  type UniformArrV4,
  type UniformF,
  type UniformMat4,
  type UniformV3,
} from './Tsl';

/** one work item per 64 clusters in the expansion queue */
export const CHUNK_CLUSTERS = 64;
/** chunk queue capacity (items; ~8 MB at uvec2) — F14: clamp + HUD flag */
export const QCHUNK_CAP = 1_048_576;
/** raster work queue capacity (one item per visible cluster; doubles as the
 *  visible-cluster list the resolve payload indexes — F3/F16: payload itemIdx
 *  has 25 bits of headroom). Pre-occlusion C1 measured 1.36M at the walk
 *  spawn (no impostor band: r2 runs to 4 km until N8) — 2M × 8 B = 16 MB,
 *  memory-bound per F14; C2 occlusion is expected to slash the live count. */
export const QRASTER_CAP = 2_097_152;
/** indirect-dispatch row size (maxComputeWorkgroupsPerDimension) */
export const DISPATCH_ROW = 65_535;
/** cone-test slack (radians, conservative on cos: sin(θ+Δ) ≤ sinθ + Δ) —
 *  absorbs lean shear (≤ ~0.15 rad normal tilt) + wind sway axis drift */
export const CONE_SLACK = 0.25;

/** per-frame camera state shared by cull/raster/resolve kernels */
export interface NaniteCam {
  /** projection · view (current frame) */
  vp: UniformMat4;
  /** inverse of vp — the resolve unprojects (ndc, storedZ) back to world */
  invVp: UniformMat4;
  camPos: UniformV3;
  planes: UniformArrV4;
  /** previous frame's VP + camera position — the occlusion-test pair
   *  (static world: prev matrices, current bounds). Frame 0 holds identity;
   *  the HZB's far-plane init makes the test pass-through regardless. */
  prevVp: UniformMat4;
  prevCamPos: UniformV3;
  /** cot(fovY/2) — screen-space projection factor (HZB level pick) */
  cotHalfFov: UniformF;
  uW: UniformF;
  uH: UniformF;
  width: number;
  height: number;
  update(camera: PerspectiveCamera): void;
}

export function makeNaniteCam(width: number, height: number): NaniteCam {
  const vp = uniformMat4(new Matrix4());
  const invVp = uniformMat4(new Matrix4());
  const camPos = uniformV3(new Vector3());
  const prevVp = uniformMat4(new Matrix4());
  const prevCamPos = uniformV3(new Vector3());
  const cotHalfFov = uniformF(1);
  const planes = uniformArrV4([
    new Vector4(),
    new Vector4(),
    new Vector4(),
    new Vector4(),
    new Vector4(),
    new Vector4(),
  ]);
  const uW = uniformF(width);
  const uH = uniformF(height);
  const projScreen = new Matrix4();
  const frustum = new Frustum();
  return {
    vp,
    invVp,
    camPos,
    prevVp,
    prevCamPos,
    cotHalfFov,
    planes,
    uW,
    uH,
    width,
    height,
    update(camera: PerspectiveCamera): void {
      prevVp.value.copy(vp.value);
      prevCamPos.value.copy(camPos.value);
      camera.updateMatrixWorld();
      projScreen.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
      vp.value.copy(projScreen);
      invVp.value.copy(projScreen).invert();
      camPos.value.copy(camera.position);
      cotHalfFov.value = 1 / Math.tan(((camera.fov * Math.PI) / 180) / 2);
      frustum.setFromProjectionMatrix(projScreen);
      for (let i = 0; i < 6; i++) {
        const p = frustum.planes[i];
        if (p) planes.array[i]?.set(p.normal.x, p.normal.y, p.normal.z, p.constant);
      }
    },
  };
}

/** instance yaw sin/cos pair, computed once per consumer */
export interface InstYaw {
  cy: NF;
  sy: NF;
}

export function instYaw(B: NV4): InstYaw {
  const yaw = B.x;
  return { cy: yaw.cos(), sy: yaw.sin() };
}

/**
 * Mesh-local point → world (contract transform, no wind): scale by A.w, yaw
 * about +Y, lean shear (B.yz · localY — base stays planted), translate A.xyz.
 */
export function instTransformPoint(A: NV4, B: NV4, yawSc: InstYaw, p: NV3): NV3 {
  const ls = p.mul(A.w);
  const rx = ls.x.mul(yawSc.cy).add(ls.z.mul(yawSc.sy));
  const rz = ls.z.mul(yawSc.cy).sub(ls.x.mul(yawSc.sy));
  const px = rx.add(B.y.mul(ls.y));
  const pz = rz.add(B.z.mul(ls.y));
  return vec3(px, ls.y, pz).add(A.xyz) as unknown as NV3;
}

/** yaw-rotate a direction (normals/cone axes — yaw is the only rotation) */
export function instRotateDir(yawSc: InstYaw, d: NV3): NV3 {
  return vec3(
    d.x.mul(yawSc.cy).add(d.z.mul(yawSc.sy)),
    d.y,
    d.z.mul(yawSc.cy).sub(d.x.mul(yawSc.sy)),
  ) as unknown as NV3;
}

/**
 * Conservative world radius for a local bounding sphere under the contract
 * transform + wind: r·scale·(1+|leanX|+|leanZ|) + swayPad. (Shear operator
 * norm ≤ 1+|l|; pads are world-space — wind displacement does not scale.)
 */
export function instSphereRadius(A: NV4, B: NV4, rLocal: NF, swayPad: NF): NF {
  const leanMag = B.y.abs().add(B.z.abs());
  return rLocal.mul(A.w).mul(leanMag.add(1)).add(swayPad) as unknown as NF;
}

/** stable hash → display color (cluster/debug tints) */
export function hashColor(id: NU): NV3 {
  const a = id.add(uint(0x9e3779b9)).mul(uint(747796405)).add(uint(289559509));
  const b = a.shiftRight(uint(16)).bitXor(a).mul(uint(277803737));
  const h = b.shiftRight(uint(16)).bitXor(b);
  return vec3(
    toF(h.bitAnd(uint(255))).div(255),
    toF(h.shiftRight(uint(8)).bitAnd(uint(255))).div(255),
    toF(h.shiftRight(uint(16)).bitAnd(uint(255))).div(255),
  )
    .mul(0.8)
    .add(0.2) as unknown as NV3;
}
