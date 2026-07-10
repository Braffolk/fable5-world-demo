/**
 * Rigid per-crown wind for VOXEL foliage — shared by the camera brick raster
 * (kVoxScatter, NaniteVoxelRaster) and the shadow voxel caster (kCaster,
 * NaniteShadowClip) so a crown and its wind-synced shadow move together.
 *
 * WHY RIGID: the mesh crown sways per-vertex via Wind.vegWindOffset (a
 * height-profiled bend + flutter). A voxel crown cannot deform per-leaf, and the
 * design (user-approved 2026-07-04) is a SINGLE logical transform of the whole
 * crown. We replicate vegWindOffset's two CROWN-COHERENT terms only — mean lean
 * (1) + primary sway (2) — and drop the per-leaf branch (3) / flutter (4) noise:
 * invisible on a coherent crown, and every dropped term is saved ALU. The offset
 * is applied per BRICK, scaled by the brick's local height ⇒ a piecewise-rigid
 * bend (base planted, top swings). Within a brick EVERY cell takes the SAME
 * offset, so the 4³ occupancy mask and the prev-frame brick sphere stay valid
 * (the friendly case for culling/occlusion).
 *
 * SYNC: the phase key is the IDENTICAL `posKey` the mesh uses (hash of instance
 * world XZ, NaniteFetch.ts:256) — so the voxel crown and the LOD0 mesh crown sway
 * in phase across the voxnear (~60 m) crossfade seam. `time`, gust field and
 * per-instance natural frequency all match the mesh leaf channel term-for-term.
 *
 * PERF (the reason for the split): both call sites are ONE-CLUSTER-PER-WORKGROUP,
 * so the 2 gust/exposure TEXTURE taps + the hash live in voxWindScalars() and the
 * caller hoists them ONCE per workgroup; voxWindOffset() is per-brick ALU only (no
 * texture). Only call these when wind is enabled — gustAt/windExposure read the
 * live wind context; the caller gates (wind off ⇒ skip, add nothing).
 */

import { float, time, uint, vec2, vec3 } from 'three/tsl';
import type { NF, NU, NV2, NV3, NV4 } from '../../gpu/TSLTypes';
import { WORLD_SIZE } from '../../world/WorldConst';
import { gustAt, windExposure, windU } from '../../render/Wind';
import { instRotateDir, type InstYaw } from '../NaniteCommon';

/** pcg-ish hash of a key → 0..1 — INLINE mirror of NaniteFetch/VegInstance
 *  `slotHash` (kept local so this module has no heavy import edge; the constants
 *  are frozen — they define the shipped per-tree wind phase). */
function slotHash(slot: NU, salt: number): NF {
  const a = slot.add(uint(salt)).mul(uint(747796405)).add(uint(2891336453));
  const b = a.shiftRight(a.shiftRight(uint(28)).add(uint(4))).bitXor(a).mul(uint(277803737));
  const c = b.shiftRight(uint(22)).bitXor(b);
  return float(c.bitAnd(uint(0xffffff))).div(16777216) as unknown as NF;
}

/** per-instance wind state (hoist ONCE per workgroup — holds the 2 texture taps) */
export interface VoxWindScalars {
  /** wind direction, world XZ */
  dirX: NF;
  dirY: NF;
  /** mean-lean amplitude (pre-height-profile) */
  leanBase: NF;
  /** primary-sway amplitude (pre-height-profile) */
  swayABase: NF;
  /** hoisted along-sway phase: sin(time·natW + ph) */
  swayS: NF;
  /** hoisted cross-sway phase: sin(time·natW·1.31 + ph·1.7) */
  swayXS: NF;
}

/**
 * PER-INSTANCE wind (2 gust/exposure texture taps + hash). Mirrors NaniteFetch's
 * leaf channel EXACTLY (same posKey, natW, gust field, farAtten) so the voxel
 * crown shares phase + amplitude with the mesh leaf crown at the crossfade seam.
 * `A` = instance row 0 (xyz = origin, w = scale). `camPos` = the MAIN camera
 * position (distance-based far attenuation must match the mesh, which keys on the
 * main camera even for shadow casters).
 */
export function voxWindScalars(A: NV4, camPos: NV3): VoxWindScalars {
  // per-tree phase key — BIT-IDENTICAL to NaniteFetch.ts:256 (hash of world XZ so
  // the trunk, the LOD0 leaf crown and this voxel crown all share one phase).
  const half = WORLD_SIZE * 0.5;
  const posKey = uint(A.x.add(half).mul(8))
    .mul(uint(73856093))
    .bitXor(uint(A.z.add(half).mul(8)).mul(uint(19349663)));
  const instPhase = slotHash(posKey as unknown as NU, 211);
  const fJit = instPhase.mul(7.31).fract();
  // leaf channel: freq = 1; natural angular frequency scaled by 1/sqrt(scale)
  const natW = fJit.mul(0.3).add(0.15).mul(6.2832).div((A.w as unknown as NF).max(0.25).sqrt());
  const ph = instPhase.mul(6.2832);

  const origin = A.xyz as unknown as NV3;
  const s = windU.strength as unknown as NF;
  const dist = origin.sub(camPos).length();
  const e = windExposure(origin.xz as unknown as NV2);
  const g = gustAt(origin.xz as unknown as NV2);
  // leaf k = 1; wind stops by the impostor band (fade 380 → 480 m), same as mesh
  const farAtten = float(1).sub(dist.sub(380).div(100).clamp(0, 1));
  const eks = e.mul(farAtten);
  const leanBase = s.mul(s).mul(g.mul(0.9).add(0.5)).mul(eks).mul(1.1);
  const swayABase = s.mul(g.mul(0.75).add(0.25)).mul(eks).mul(0.5);
  const swayS = time.mul(natW).add(ph).sin();
  const swayXS = time.mul(natW.mul(1.31)).add(ph.mul(1.7)).sin();
  const d = vec2(windU.dir as unknown as NV2);
  return {
    dirX: d.x as unknown as NF,
    dirY: d.y as unknown as NF,
    leanBase: leanBase as unknown as NF,
    swayABase: swayABase as unknown as NF,
    swayS: swayS as unknown as NF,
    swayXS: swayXS as unknown as NF,
  };
}

/**
 * PER-BRICK world-space wind offset (ALU only — no texture). `localY` = the
 * brick centre's LOCAL height (tree base = 0, same frame as the mesh vertices).
 * Height profile matches vegWindOffset's cantilever (yn²·1.7, h0 = 6, capped
 * 1.6) minus the per-vertex flex term (no per-brick flex) — base ≈ 0, top ≈ 1.
 * ADD the result to the brick's world centre (and to every occupancy-cell world
 * centre, using the SAME brick localY, to keep the brick rigid).
 */
export function voxWindOffset(w: VoxWindScalars, localY: NF): NV3 {
  const y = localY.max(0);
  const yn = y.div(y.add(float(6)));
  const prof = yn.mul(yn).mul(1.7).min(1.6);
  const along = w.leanBase.add(w.swayS.mul(w.swayABase)).mul(prof);
  const across = w.swayXS.mul(w.swayABase).mul(0.45).mul(prof);
  return vec3(
    w.dirX.mul(along).sub(w.dirY.mul(across)),
    float(0),
    w.dirY.mul(along).add(w.dirX.mul(across)),
  ) as unknown as NV3;
}

/**
 * PER-BRICK LOCAL-space wind offset for the CAMERA voxel raster (kVoxScatter). Add
 * this to the brick's LOCAL centre and EVERY derived quantity — the screen
 * footprint, the voxCell ray-march AABB bounds, the occupancy-mask cells — shifts
 * from one source, so the ray→local round-trip (via the unchanged instance A) still
 * lands on the swayed brick (no footprint/ray desync, the failure mode of a
 * world-only offset). Equals invYaw(worldOffset)/scale, chosen so
 * instTransformPoint(local + δ) shifts the world position by EXACTLY voxWindOffset()'s
 * world vector — the same displacement the shadow caster adds directly and the mesh
 * crown applies, so all three stay in phase. (A horizontal δ has no local-Y term, so
 * instTransformPoint's lean-shear leaves it untouched — the identity is exact.)
 */
export function voxWindLocalOffset(
  w: VoxWindScalars,
  localY: NF,
  yawSc: InstYaw,
  scale: NF,
): NV3 {
  const d = voxWindOffset(w, localY); // world XZ offset (y = 0)
  // inverse yaw = instRotateDir with the sine negated; then cancel the A.w scale that
  // instTransformPoint applies to local coords.
  const inv: InstYaw = { cy: yawSc.cy, sy: yawSc.sy.mul(-1) as unknown as NF };
  return instRotateDir(inv, d).div(scale) as unknown as NV3;
}

/**
 * Conservative WORLD-space sway pad for a voxel crown mesh's cull sphere (baked
 * into mesh word 11 / GeometryRegistry `swayPad`, added by instWorldSphere). Must
 * upper-bound |voxWindOffset| over the crown so a swaying brick never leaves the
 * frustum/HZB test. Peak ≈ (leanBase + swayABase)·prof_max with prof_max = 1.6;
 * with unit strength/gust/exposure that is ≈ (2.0 + 0.5)·1.6·scale ≈ 4·scale m.
 * Use a fixed metres pad ≥ that for the tallest expected crown — cheap and never
 * scaled per instance (wind displacement is world-space, not scale-multiplied).
 */
export const VOX_CROWN_SWAY_PAD_M = 4.0;
