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

/** queue-cap URL override (?qrcap/?qvcap/?qfrontier/…): clamped [lo,hi] item count.
 *  Safe outside the browser main thread (workers/tools have no location → default).
 *  Every emit into these queues is atomicAdd + slot<cap bounds-check, so an
 *  undersized cap degrades to missing geometry for the frame, never OOB writes. */
export function queueCapParam(name: string, def: number, lo: number, hi: number): number {
  try {
    const search = (globalThis as { location?: { search?: string } }).location?.search ?? '';
    const raw = new URLSearchParams(search).get(name);
    if (raw != null) {
      const v = Math.round(Number(raw));
      if (Number.isFinite(v) && v > 0) return Math.min(hi, Math.max(lo, v));
    }
  } catch {
    /* no location (worker/node) — use the default */
  }
  return def;
}

/** the tight measured queue defaults apply to the WORLD scene only: probe scenes
 *  (probe-forest 200k trees etc.) historically flooded a 2M queue and other perf
 *  harnesses depend on the legacy headroom — shrinking them silently would clamp
 *  THEIR cuts. World demand was measured 2026-07-04 (@dpr2, eye/hill/oblique/
 *  aerial + 12 s moving leg + boot full shadow re-raster) via window.__qHW. */
const WORLD_SCENE = (() => {
  try {
    const search = (globalThis as { location?: { search?: string } }).location?.search ?? '';
    return new URLSearchParams(search).get('scene') === 'world';
  } catch {
    return false;
  }
})();

/** raster work queue capacity (one item per visible cluster; doubles as the
 *  visible-cluster list the resolve payload indexes — F3/F16: payload itemIdx
 *  has 25 bits of headroom). HISTORY: raised 2M→8M (2^23) for dense far-field
 *  views that flooded the old 2M cap PRE-fartiles/instMinPx; with the far-field
 *  cluster floor shipped, the MEASURED world worst case (2026-07-04, all poses +
 *  moving) is ~60 k items — the 8M sizing paid 67 MB GPU + a 67 MB permanent CPU
 *  mirror (three keeps the Uint32Array) per chain for a ~140× safety factor.
 *  WORLD default now 2^17 = 131,072 (~2.2× measured HW); non-world scenes keep
 *  8M. ?qrcap=N overrides either way (≤ the 8M ceiling: bit budget
 *  itemIdx<<CLUSTER_TRI_BITS|localTri ⇒ 23+7 = 30 of 32 bits at the default
 *  128-tri cap, and (CAP+1)×8 B must stay under the 128 MB per-binding limit).
 *  Every emit is slot<cap guarded; overflow = dropped clusters this frame,
 *  flagged via readCounts().overflow — never OOB writes. */
// World default raised 131k→1M for ?crownlod0 (default on): forcing leaf crowns to
// LOD0 in the mesh band floods the camera cut — measured qRaster 197k / frontier 213k
// at a dense static pose (vs 38k/27k without crownlod0), so the old 131k cap OVERFLOWED
// → dropped clusters = foliage holes. 1M = ~5× the observed worst for moving headroom;
// costs ~+42MB vs 131k but still ~340MB under the original 8M. ?qrcap overrides.
export const QRASTER_CAP = queueCapParam('qrcap', WORLD_SCENE ? 1_048_576 : 8_388_608, 4_096, 8_388_608);
/** voxel-foliage (spec §6.2): the VOXEL raster work queue capacity (one item per
 *  visible voxel BRICK-cluster fanned out of qRaster by matClass=voxel(7)). Far
 *  smaller than QRASTER_CAP — the coarse voxel band has far fewer cluster work-items
 *  than the triangle cut (§6.0). Measured world HW 2026-07-04: ~35 k (oblique) ⇒
 *  WORLD default 2^17 = 131,072 (~3.8×); non-world scenes keep 2M. Bit-budget
 *  (spec §6.2 verbatim assert): the voxel-raster payload (a brick work-item index
 *  into qVoxRaster) must stay < 1<<28 (BUCKET_SHIFT); the resolve additionally
 *  unpacks the item index from bits 0-20 (mask 0x1fffff) — so ?qvcap may never
 *  exceed 2^21 = 2M (the hard hi clamp). */
export const QVOX_CAP = queueCapParam('qvcap', WORLD_SCENE ? 131_072 : 2_097_152, 1_024, 2_097_152);

/** per-chain queue high-water diagnostics (cull-queue memory sizing): each cull
 *  chain registers a reader over a small GPU max-counter buffer (atomicMax'd by
 *  the chain's tiny args kernels from the RAW pre-clamp cursors, so the value is
 *  the true demand even when a cap clamps the queue). Probe access:
 *  for (c of window.__qHW.chains) await c.read(window.__qHW.renderer). */
export interface QueueHwChain {
  label: string;
  /** the caps the chain was built with (items) */
  caps: Record<string, number>;
  /** read the high-water counters (items, raw pre-clamp) */
  read(renderer: unknown): Promise<Record<string, number>>;
  /** zero the high-water counters (dispatches a tiny kernel) */
  reset(renderer: unknown): void;
}
interface QueueHwGlobal {
  chains: QueueHwChain[];
  renderer: unknown;
}
function queueHwGlobal(): QueueHwGlobal {
  const g = globalThis as { __qHW?: QueueHwGlobal };
  g.__qHW ??= { chains: [], renderer: null };
  return g.__qHW;
}
export function registerQueueHw(chain: QueueHwChain): void {
  queueHwGlobal().chains.push(chain);
}
/** stash the live renderer for the probe (called from per-frame cull entry points) */
export function noteQueueHwRenderer(renderer: unknown): void {
  queueHwGlobal().renderer = renderer;
}
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
