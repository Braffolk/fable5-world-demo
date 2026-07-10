/**
 * Shared (instance, cluster) → triangle → world-vertex decode for every
 * consumer of the registry mega-buffers: the SW raster kernels, the HW
 * big/near-tri vertex pulling, and the material resolve (N4). One code path
 * means the raster and the resolve reconstruct bit-identical world positions
 * by construction — the same guarantee the N3 fixed-point core relies on
 * between its own depth and payload passes.
 *
 * Moved verbatim out of NaniteRaster at N4-C0 (no behavior change; the probe
 * battery is the witness).
 */

import type { Texture } from 'three';
import type { StorageTexture } from 'three/webgpu';
import { If, clamp, float, mix, smoothstep, texture, time, uint, vec2, vec3, wgslFn } from 'three/tsl';
import { TerrainField } from '../world/TerrainField';
import type { NB, NF, NU, NV2, NV3, NV4 } from '../../gpu/TSLTypes';
import { DISP } from '../../render/TerrainMaterial';
import { PERIOD_FBM, PERIOD_RID, PERIOD_VAL } from '../../gpu/passes/NoiseBake';
import { WORLD_SIZE } from '../../world/WorldConst';
import { gustAt, gustLagAt, windExposure, windU, WIND_LAG_M } from '../../render/Wind';
import { SKIRT_DEPTH_A, SKIRT_DEPTH_B } from '../build/BuildHeightGrid';
import {
  CLUSTER_FLAG_DAG,
  MESH_FLAG_TWO_SIDED,
  MESH_WORDS,
  TRANSFORM_CHANNEL,
  VERT_WORDS,
} from '../world/GeometryRegistry';
import type { RegistryGpu } from '../world/GeometryRegistry';
import { instTransformPoint, instYaw, type InstYaw } from '../NaniteCommon';
import type { UniformV3 } from '../Tsl';
import { bcU2F, elemU, maxU, minU, texLoadR, toF } from '../Tsl';

/** cheap pcg-ish hash of an instance slot → 0..1 (mirror of VegInstance.slotHash
 *  — the trunk wind needs the SAME per-instance phase the old path baked; the
 *  resolve reuses it for the per-instance tint, AUDIT-1a). */
export function slotHash(slot: NU, salt: number): NF {
  const a = slot.add(uint(salt)).mul(uint(747796405)).add(uint(2891336453));
  const b = a.shiftRight(a.shiftRight(uint(28)).add(uint(4))).bitXor(a).mul(uint(277803737));
  const c = b.shiftRight(uint(22)).bitXor(b);
  return float(c.bitAnd(uint(0xffffff))).div(16777216);
}

/** trunk-wind option for makeFetch (the gust FIELD comes from the Wind module
 *  context, set unconditionally at scene boot); camPos drives the far-fade dist */
export interface TrunkWindOpt {
  camPos: UniformV3;
}

/** per-instance wind scalars precomputed in makeCtx (the gust texture samples
 *  happen ONCE per instance here, not per rasterised corner). The per-vertex
 *  prof/flex scaling is applied in fetchWorldVert. Carries BOTH channels: the
 *  trunk fields stay 0 on the 'leaf' channel and `flutBase` (N9-C0 leaf flutter)
 *  stays 0 on the 'trunk' channel — a mesh is exactly one channel. */
interface TrunkWindFields {
  h0: NF;
  dirX: NF;
  dirY: NF;
  leanBase: NF;
  swayABase: NF;
  /** PRECOMPUTED sin(time·natW + ph) — cluster-invariant (natW/ph are per-instance,
   *  time per-frame), so the sine is hoisted out of the 384×/cluster per-vertex path
   *  into makeCtx (cached 1×). fetchWorldVert just scales it by the per-vertex amp. */
  swayPhase: NF;
  /** PRECOMPUTED sin(time·natW·1.31 + ph·1.7) — the cross-sway sine, same hoist. */
  swayXPhase: NF;
  /** per-instance phase — KEPT (the leaf flutter coord still needs it to decorrelate
   *  instances); natW is gone (folded into the two precomputed sines above). */
  ph: NF;
  branchBase: NF;
  /** N9-C0: leaf-flutter amplitude base (s·gust·exposure, ≤120 m faded); 0 on trunk */
  flutBase: NF;
}

/**
 * Terrain micro-displacement inputs ('terrain' transform channel, N4-C1):
 * the EXACT TerrainTiles vertex formula (world-space fields, distance-faded
 * 45→85 m) applied to heightfield vertices at fetch time, so the raster, the
 * HW passes and the resolve all see the displaced surface. (The full-frame
 * C1 gate vs ?nanite=0 is what verifies displacement.)
 *
 * S3b: two input shapes during the staged migration (?tfield A/B) — the FIELD
 * shape reads the TerrainField planes (slope from a height-plane CD instead of
 * normalTex.w; snow/rockExposure/flow from the fields plane; riverDepth derived
 * as waterY − ground). The legacy hf-texture shape is EXCISED at S3b end.
 */
export interface TerrainDispField {
  field: TerrainField;
  noiseA: StorageTexture;
  noiseB: StorageTexture;
  camPos: UniformV3;
}
export interface TerrainDispLegacy {
  normalTex: StorageTexture;
  biomeTex: StorageTexture;
  fieldsTex: StorageTexture;
  noiseA: StorageTexture;
  noiseB: StorageTexture;
  camPos: UniformV3;
}
export type TerrainDisp = TerrainDispField | TerrainDispLegacy;

/**
 * TerrainTiles micro-displacement at a world xz, verbatim (world-space fields;
 * amplitude gated by slope/rockExposure/snow, faded 45→85 m). Factored out of
 * hfWorld so NON-terrain surface followers (procedural grass roots — the ground
 * a blade stands on is the DISPLACED surface, not raw heightTex) evaluate the
 * EXACT same expression tree as the terrain vertices. Call inside an Fn stack.
 */
export function terrainDispAt(disp: TerrainDisp, wpos: NV2, groundH?: NF): NF {
  if ('field' in disp && !groundH) {
    // riverDepth is DERIVED (waterY − ground) on the field shape — every caller
    // already holds the ground height it displaces (vert fetch / grass root)
    throw new Error('terrainDispAt: field-shape disp needs the ground height');
  }
  const camD = wpos.sub(vec3(disp.camPos).xz).length();
  const dOut = float(0).toVar();
  If(camD.lessThan(float(DISP.fade1)), () => {
    let rockK: NF;
    let gravelK: NF;
    let snow: NF;
    if ('field' in disp) {
      // TerrainField shape (S3b): slope = height-plane CD (the retired
      // normalTex.w stencil); flow/snow/rockExposure = ONE filtered fields-plane
      // tap; riverDepth = waterY − ground (spec §3: not stored, derived).
      const fld = disp.field.fieldsAt(wpos);
      const slope = disp.field.fieldSlope(wpos);
      const riverDepth = disp.field.fieldWaterYNearest(wpos).sub(groundH as NF).max(0);
      rockK = smoothstep(DISP.slopeKnee0, DISP.slopeKnee1, slope).max(fld.w.mul(0.85)) as unknown as NF;
      gravelK = smoothstep(0.32, 0.7, fld.y)
        .max(smoothstep(0.02, 0.2, riverDepth))
        .mul(float(DISP.gravel)) as unknown as NF;
      snow = fld.z as unknown as NF;
    } else {
      const uvV = wpos.div(WORLD_SIZE).add(0.5) as unknown as NV2;
      const nsV = texture(disp.normalTex, uvV, 0) as unknown as NV4;
      const bioV = texture(disp.biomeTex, uvV, 0) as unknown as NV4;
      const fldV = texture(disp.fieldsTex, uvV, 0) as unknown as NV4;
      rockK = smoothstep(DISP.slopeKnee0, DISP.slopeKnee1, nsV.w).max(bioV.a.mul(0.85)) as unknown as NF;
      gravelK = smoothstep(0.32, 0.7, fldV.y)
        .max(smoothstep(0.02, 0.2, fldV.z))
        .mul(float(DISP.gravel)) as unknown as NF;
      snow = bioV.g as unknown as NF;
    }
    const dispAmp = (mix(float(DISP.base), float(DISP.rock), rockK) as unknown as NF)
      .max(gravelK)
      .mul(snow.mul(0.75).oneMinus())
      .mul(clamp(float(DISP.fade1).sub(camD).div(DISP.fade1 - DISP.fade0), 0, 1));
    const f1 = (texture(disp.noiseA, wpos.div(DISP.sF1 * PERIOD_FBM), 0) as unknown as NV4).y
      .mul(2)
      .sub(1);
    const f2 = (
      texture(
        disp.noiseA,
        wpos.div(DISP.sF2 * PERIOD_VAL).add(vec2(0.31, 0.77)),
        0,
      ) as unknown as NV4
    ).x
      .mul(2)
      .sub(1);
    const r1 = (texture(disp.noiseB, wpos.div(DISP.sRid * PERIOD_RID), 0) as unknown as NV4).z
      .mul(2)
      .sub(1);
    dOut.assign(
      f1
        .mul(DISP.wF1)
        .add(f2.mul(DISP.wF2))
        .add(r1.mul(rockK.mul(1 - DISP.ridBase).add(DISP.ridBase)).mul(DISP.wRid))
        .mul(dispAmp),
    );
  });
  return dOut as unknown as NF;
}

/** per-(instance, cluster) decode shared by the 3 corner fetches */
export interface VertCtx {
  isHF: NB;
  /** heightfield ADAPTIVE-DAG variant (CLUSTER_FLAG_DAG): explicit indexed
   *  topology, each vertex word0 = packed grid coord — vs the implicit window grid */
  isDAG: NB;
  A: NV4;
  B: NV4;
  yawSc: InstYaw;
  triStart: NU;
  triCount: NU;
  meshId: NU;
  /** transform channel (TRANSFORM_CHANNEL) — 'trunk' (1) gets wind */
  channel: NU;
  /** N9-C2: MESH_FLAG_TWO_SIDED — the raster re-winds back-faces instead of culling */
  twoSided: NB;
  /** precomputed per-instance trunk-wind scalars, or null (wind off / non-trunk) */
  wind: TrunkWindFields | null;
  /** heightfield: vertex-grid window base + partial width (quads) */
  gx: NU;
  gz: NU;
  qxw: NU;
  oX: NF;
  oZ: NF;
  cell: NF;
}

export interface NaniteFetch {
  makeCtx(instId: NU, ci: NU): VertCtx;
  /** world-space corner v of ctx's localTri (heightfield convention = spike:
   *  up-facing CCW, even tri (0,0)(0,1)(1,1), odd tri (0,0)(1,1)(1,0)) */
  fetchWorldVert(ctx: VertCtx, localTri: NU, v: 0 | 1 | 2): NV3;
  /** world-space position of a vertex BY its global index `vi` (the value
   *  `fetchWorldVert` reads from `gpu.indices`). Same math as `fetchWorldVert`,
   *  keyed by vi — the projection pre-pass (raster/Project) transforms each unique vi
   *  ONCE through this. Defined for the explicit and adaptive-DAG conventions;
   *  the window-grid heightfield has no index buffer (count=0 ⇒ never called). */
  fetchWorldVertByIndex(ctx: VertCtx, vi: NU): NV3;
  /** fetchWorldVert with a RUNTIME corner (0..2). Selects the index / grid offset
   *  by `corner` BEFORE fetching, so exactly ONE vertex is reconstructed (the HW
   *  vertex stage's single-fetch path). Same selected vertex by construction
   *  (identical per-corner math as the static-v arms of fetchWorldVert). */
  fetchWorldVertDyn(ctx: VertCtx, localTri: NU, corner: NU): NV3;
  /** mesh-record word 6: matClass u8 (bits 8–15) etc. */
  meshWord(meshId: NU, word: number): NU;
}

/**
 * ?fp16w (task #76): the per-vertex trunk/leaf wind offset in raw-WGSL f16. The offset is
 * a SMALL delta (≲ metres) added to the f32 world position, so half-precision is safe, and
 * the SAME wgslFn runs in the raster AND the resolve ⇒ they stay bit-consistent (no cracks).
 * f16 packs 2 values/register ⇒ halves this math's register pressure + ALU. TSL has no f16
 * node, so this is raw WGSL; the module-top `enable f16;` directive is plumbed by
 * gpu/EnableF16.ts (WebGPUBackend patch, active under ?fp16w). Faithful port of the f32 TSL
 * math below (yn/prof/swayA/sway/swayX/along/dy → the vec3 offset). Inputs are f32; the body
 * converts, computes in f16, returns vec3<f32>.
 */
let nanWindF16Fn: ReturnType<typeof wgslFn> | null = null;
/** lazily built (wgslFn parses its WGSL at construction) so a bad parse can only affect
 *  ?fp16w runs, never the shipped default fetch path this file also feeds. */
const nanWindF16 = (): ReturnType<typeof wgslFn> =>
  (nanWindF16Fn ??= wgslFn(`
  fn nanWindF16( h0: f32, dirX: f32, dirY: f32, leanBase: f32, swayABase: f32, swayPhase: f32, swayXPhase: f32, branchBase: f32, localY: f32, flex: f32 ) -> vec3<f32> {
    let ly = f16(localY);
    let fx = f16(flex);
    let yn = ly / (ly + f16(h0));
    let prof = min(yn * yn * 1.7h + fx * 0.3h, 1.6h);
    let swayA = f16(swayABase) * prof;
    let sway = f16(swayPhase) * swayA;
    let swayX = f16(swayXPhase) * swayA * 0.45h;
    let along = f16(leanBase) * prof + sway + f16(branchBase) * fx;
    let dy = (abs(along) + abs(swayX)) * fx * -0.2h;
    return vec3<f32>(
      f32(f16(dirX) * along - f16(dirY) * swayX),
      f32(dy),
      f32(f16(dirY) * along + f16(dirX) * swayX)
    );
  }
`));

export function makeFetch(
  gpu: RegistryGpu,
  /** terrain height source: the TerrainField plane pyramid (S3b default), or the
   *  legacy global heightTex (?tfield staging arm — EXCISED at S3b end) */
  heightSrc: Texture | TerrainField,
  disp?: TerrainDisp,
  wind?: TrunkWindOpt,
  /** N8-D2 Stage 2e: bind the stride-1 terrain-DAG vertex buffer (gpu.hfVerts) in
   *  the isHF&&isDAG branch. The RASTER needs it (terrain positions); the RESOLVE
   *  does NOT (terrain reconstructs its world pos from the depth buffer, rock/bark
   *  take the explicit-mesh else branch) — so the resolve passes false to keep one
   *  fewer storage buffer in its already buffer-heavy fragment stage. */
  bindHfVerts = true,
  /** PERF task #76 kernel-split: which fetch path THIS instance compiles.
   *  'both' (default) = the runtime If(isHF).Else UNION — byte-identical to the
   *    pre-split code (a closure emits the same nodes as the old inline body).
   *  'explicit' = leaf/trunk/rock ONLY — no heightfield arm ⇒ terrainDispAt (6 tex
   *    + fbm) is never compiled ⇒ the leaf kernel sheds those registers.
   *  'terrain' = heightfield ONLY — no explicit arm and NO wind precompute ⇒ the
   *    terrain kernel reserves zero instance-transform / wind registers.
   *  A specialized raster kernel picks one class so it never reserves the other's
   *  register set (the branch-union that inflated world1's occupancy floor). */
  variant: 'both' | 'explicit' | 'terrain' = 'both',
): NaniteFetch {
  // ?fp16w (task #76): route the per-vertex trunk/leaf wind offset through the f16 wgslFn
  // (nanWindF16) instead of the f32 TSL math — halves that math's registers + ALU. Off =
  // byte-identical f32 path.
  const fp16Wind =
    new URLSearchParams(window.location.search).get('fp16w') === '1';
  const makeCtx = (instId: NU, ci: NU): VertCtx => {
    const cBase = ci.mul(uint(8)).toVar();
    const triStart = elemU(gpu.clusters, cBase.add(uint(6))).toVar();
    const w7 = elemU(gpu.clusters, cBase.add(uint(7))).toVar();
    const triCount = w7.bitAnd(uint(0xff)).toVar();
    const flags = w7.shiftRight(uint(8)).bitAnd(uint(0xff)).toVar();
    const isHF = flags.bitAnd(uint(1)).notEqual(uint(0)).toVar();
    const isDAG = flags.bitAnd(uint(CLUSTER_FLAG_DAG)).notEqual(uint(0)).toVar();
    const meshId = w7.shiftRight(uint(16)).toVar();
    const A = gpu.instances.element(instId.mul(uint(2))).toVar() as unknown as NV4;
    const B = gpu.instances.element(instId.mul(uint(2)).add(uint(1))).toVar() as unknown as NV4;
    const mBase = meshId.mul(uint(MESH_WORDS)).toVar();
    const w6 = elemU(gpu.meshes, mBase.add(uint(6))).toVar();
    const winW = w6.shiftRight(uint(24)).toVar();
    const channel = w6.bitAnd(uint(0xff)).toVar();
    // N9-C2: two-sided bit (flags byte 2 of w6) — the raster re-winds back-faces
    // instead of culling for these meshes (leaf crowns). Free: w6 already loaded.
    const twoSided = w6.shiftRight(uint(16)).bitAnd(uint(MESH_FLAG_TWO_SIDED)).notEqual(uint(0)).toVar();
    const quadsX = elemU(gpu.meshes, mBase.add(uint(10))).bitAnd(uint(0xffff)).toVar();
    const gx = triStart.bitAnd(uint(0xffff)).mul(winW).toVar();
    const gz = triStart.shiftRight(uint(16)).mul(winW).toVar();
    const qxw = minU(winW, maxU(quadsX, gx).sub(gx)).toVar();
    const oX = bcU2F(elemU(gpu.meshes, mBase.add(uint(7)))).toVar();
    const oZ = bcU2F(elemU(gpu.meshes, mBase.add(uint(8)))).toVar();
    const cell = bcU2F(elemU(gpu.meshes, mBase.add(uint(9)))).toVar();

    // trunk-wind per-instance precompute (Wind.vegWindOffset, minus the leaf
    // flutter — trunks have low flex). The gust FIELD reads (e/g/gL = 4 texture
    // samples) happen ONCE here, gated on the trunk channel so terrain/rock pay
    // nothing; fetchWorldVert applies only the per-vertex prof/flex scaling.
    let windFields: TrunkWindFields | null = null;
    // 'terrain' variant carries NO wind (the heightfield channel has none) — skip the
    // whole precompute so the specialized terrain kernel reserves zero wind registers.
    if (wind && variant !== 'terrain') {
      const matParam = elemU(gpu.meshes, mBase.add(uint(7))).toVar();
      const profile = matParam.shiftRight(uint(8)).bitAnd(uint(0xff)).toVar();
      const d = vec2(windU.dir as unknown as NV2);
      const h0 = float(0).toVar();
      const leanBase = float(0).toVar();
      const swayABase = float(0).toVar();
      const natW = float(0).toVar();
      const ph = float(0).toVar();
      const branchBase = float(0).toVar();
      const flutBase = float(0).toVar();
      // N9-C0: SHARED per-tree wind key. The bark trunk and the leaf crown of ONE
      // tree are now SEPARATE meshes with DIFFERENT global instIds, so keying the
      // sway phase on instId desyncs them (crown swings out of phase with its
      // branches). Hash the world POSITION instead — identical for both, since the
      // leaf head binds the SAME instance A as the bark — so they sway in phase.
      // (Quantized ~0.125 m; +half keeps the f32→u32 cast positive. This also
      // restores the original's shared-slot behaviour: bark+foliage were one
      // instance there. The per-instance TINT still keys on instId, unchanged.)
      const half = WORLD_SIZE * 0.5;
      const posKey = uint(A.x.add(half).mul(8))
        .mul(uint(73856093))
        .bitXor(uint(A.z.add(half).mul(8)).mul(uint(19349663)))
        .toVar();
      If(channel.equal(uint(TRANSFORM_CHANNEL.trunk)), () => {
        const origin = A.xyz as unknown as NV3;
        const s = windU.strength as unknown as NF;
        const dist = origin.sub(vec3(wind.camPos)).length();
        const e = windExposure(origin.xz as unknown as NV2);
        const g = gustAt(origin.xz as unknown as NV2);
        const gL = gustLagAt(origin.xz as unknown as NV2, WIND_LAG_M);
        const isSnag = profile.equal(uint(1));
        const isShrub = profile.equal(uint(2));
        const k = isSnag.select(float(0.45), float(1));
        const freq = isSnag.select(float(0.8), isShrub.select(float(1.8), float(1)));
        h0.assign(isShrub.select(float(0.9), float(6)));
        const farAtten = float(1).sub(dist.sub(380).div(100).clamp(0, 1));
        const eks = e.mul(k).mul(farAtten).toVar();
        leanBase.assign(s.mul(s).mul(g.mul(0.9).add(0.5)).mul(eks).mul(1.1));
        swayABase.assign(s.mul(g.mul(0.75).add(0.25)).mul(eks).mul(0.5));
        const instPhase = slotHash(posKey, 211).toVar();
        const fJit = instPhase.mul(7.31).fract();
        natW.assign(fJit.mul(0.3).add(0.15).mul(6.2832 * 1).mul(freq).div(A.w.max(0.25).sqrt()));
        ph.assign(instPhase.mul(6.2832));
        const brAtten = float(1).sub(dist.sub(160).div(140).clamp(0, 1));
        branchBase.assign(gL.sub(0.45).mul(s).mul(eks).mul(0.55).mul(brAtten));
      });
      // N9-C0: leaf channel = the FULL Wind.vegWindOffset (terms 1–4), so the
      // crown SWAYS WITH the trunk (lean+sway+branch) AND flutters — NOT a
      // reinvented motion. Mirrors the trunk block with TREE-fixed params (k=1,
      // freq=1, h0=6: matParam carries the leaf TINT on this channel, not a wind
      // profile) and adds the flutter base (term 4). The gust/exposure FIELD reads
      // happen ONCE per instance here; explicitWorldByIndex does the per-vertex
      // prof/flex scaling + the shared leafFlutterAxes() noise tap.
      If(channel.equal(uint(TRANSFORM_CHANNEL.leaf)), () => {
        const origin = A.xyz as unknown as NV3;
        const s = windU.strength as unknown as NF;
        const dist = origin.sub(vec3(wind.camPos)).length();
        const e = windExposure(origin.xz as unknown as NV2);
        const g = gustAt(origin.xz as unknown as NV2);
        const gL = gustLagAt(origin.xz as unknown as NV2, WIND_LAG_M);
        h0.assign(float(6));
        const farAtten = float(1).sub(dist.sub(380).div(100).clamp(0, 1));
        const eks = e.mul(farAtten).toVar();
        leanBase.assign(s.mul(s).mul(g.mul(0.9).add(0.5)).mul(eks).mul(1.1));
        swayABase.assign(s.mul(g.mul(0.75).add(0.25)).mul(eks).mul(0.5));
        const instPhase = slotHash(posKey, 211).toVar();
        const fJit = instPhase.mul(7.31).fract();
        natW.assign(fJit.mul(0.3).add(0.15).mul(6.2832).div(A.w.max(0.25).sqrt()));
        ph.assign(instPhase.mul(6.2832));
        const brAtten = float(1).sub(dist.sub(160).div(140).clamp(0, 1));
        branchBase.assign(gL.sub(0.45).mul(s).mul(eks).mul(0.55).mul(brAtten));
        const flutAtten = float(1).sub(dist.sub(40).div(80).clamp(0, 1));
        flutBase.assign(s.mul(g.mul(0.7).add(0.3)).mul(eks).mul(0.07).mul(flutAtten));
      });
      // GRASS channel (S1, 31-grass-plan §6): the GroundRing wind model on the
      // SAME gust field — cantilever bend ∝ tip² (applied per vertex) + a fine
      // shimmer whose sine is fully per-instance (GroundRing keys it on the
      // instance world pos, so it hoists here). Field slots reused (NO new
      // TrunkWindFields members — the shF broadcast stays untouched):
      //   leanBase ← bend amplitude = amp·(s·0.55+0.6)·(bladeH·0.42), lean² rule
      //   flutBase ← shimmer amplitude = amp·0.05 × the leaf-style ~120 m fade
      //              (the shipped ring only zeroes shimmer in far mode — the fade
      //              is the new lane's TRAA-stability upgrade, §2 item 6)
      //   swayXPhase ← the hoisted shimmer sine
      // HOIST the per-vertex sines here: their args (natW, ph per-instance; time
      // per-frame) are cluster-invariant, so compute the 2 sines ONCE per cluster
      // instead of 384×/cluster in fetchWorldVert. Cached via wgcache like the rest.
      const swayPhase = time.mul(natW).add(ph).sin().toVar();
      const swayXPhase = time.mul(natW.mul(1.31)).add(ph.mul(1.7)).sin().toVar();
      windFields = {
        h0: h0 as unknown as NF,
        dirX: d.x as unknown as NF,
        dirY: d.y as unknown as NF,
        leanBase: leanBase as unknown as NF,
        swayABase: swayABase as unknown as NF,
        swayPhase: swayPhase as unknown as NF,
        swayXPhase: swayXPhase as unknown as NF,
        ph: ph as unknown as NF,
        branchBase: branchBase as unknown as NF,
        flutBase: flutBase as unknown as NF,
      };
    }
    return {
      isHF: isHF as unknown as NB,
      isDAG: isDAG as unknown as NB,
      A,
      B,
      yawSc: instYaw(B),
      triStart,
      triCount,
      meshId,
      channel,
      twoSided: twoSided as unknown as NB,
      wind: windFields,
      gx,
      gz,
      qxw,
      oX: oX as unknown as NF,
      oZ: oZ as unknown as NF,
      cell: cell as unknown as NF,
    };
  };

  // ---- world-reconstruction helpers (split out of fetchWorldVert at PERF-3 stage
  //      2a — behavior-preserving: fetchWorldVert recomposes the SAME math. The
  //      by-INDEX forms exist so the projection pre-pass (raster/Project) can
  //      transform each UNIQUE vert ONCE, keyed by its global index vi). ----------

  /** heightfield TAIL shared by both HF conventions: grid texel (sx,sz) + skirtDrop
   *  → world pos (height fetch + reconstruction + the verbatim micro-displacement).
   *  Only how (sx,sz) are obtained differs (adaptive-indexed vs implicit window grid). */
  const hfWorld = (ctx: VertCtx, sx: NU, sz: NU, skirtDrop: NF): NV3 => {
    const out = vec3(0).toVar();
    const wx = toF(sx).mul(ctx.cell).add(ctx.oX);
    const wz = toF(sz).mul(ctx.cell).add(ctx.oZ);
    // S3b: height from the TerrainField plane pyramid — terrain verts sit ON the
    // finest lattice, so the finest arm's exact-texel tap is bit-identical to the
    // legacy global heightTex read; coarser windows bilerp. One branch arm runs,
    // near-uniform per cluster (A15's level-select cost stays off the hot path).
    const h = heightSrc instanceof TerrainField
      ? heightSrc.fieldHeightHot(vec2(wx, wz) as unknown as NV2)
      : texLoadR(heightSrc, sx, sz);
    if (disp) {
      const dOut = terrainDispAt(disp, vec2(wx, wz) as unknown as NV2, h);
      out.assign(vec3(wx, h.add(dOut).sub(skirtDrop), wz));
    } else {
      out.assign(vec3(wx, h.sub(skirtDrop), wz));
    }
    return out as unknown as NV3;
  };

  /** adaptive terrain-DAG vertex BY INDEX (D2b): each vertex packs the texel coord —
   *  gx in bits 0-12, skirt code in 13-15, gz in 16-31. N8-D2 Stage 2e: terrain verts
   *  live in their OWN stride-1 buffer (gpu.hfVerts, one word/vert), so index it by vi
   *  directly. RASTER reads the stride-1 hf buffer; RESOLVE (bindHfVerts=false) never
   *  reaches this (terrain uses depth there) so it reads gpu.verts as harmless dead
   *  code, dropping the hfVerts binding from the resolve's fragment stage. */
  const dagWorldByIndex = (ctx: VertCtx, vi: NU): NV3 => {
    const packed = bindHfVerts ? elemU(gpu.hfVerts, vi) : elemU(gpu.verts, vi);
    const sx = packed.bitAnd(uint(0x1fff)).toVar();
    const sz = packed.shiftRight(uint(16)).toVar();
    // N8-D2 Stage 2d: a perimeter SKIRT vert carries a 3-bit depth-level code in
    // bits 13-15 of word0 (0 = surface vert); its world Y drops below the surface
    // to seal inter-level T-junction cracks.
    const skirtDrop = float(0).toVar();
    const code = packed.shiftRight(uint(13)).bitAnd(uint(0x7)).toVar();
    If(code.greaterThan(uint(0)), () => {
      // depth = SKIRT_DEPTH_A + SKIRT_DEPTH_B·level, level = code−1 (linear ⇒ hugs
      // the saturating inter-level crack; see BuildHeightGrid for the calibration)
      skirtDrop.assign(float(SKIRT_DEPTH_A).add(float(SKIRT_DEPTH_B).mul(toF(code.sub(uint(1))))));
    });
    return hfWorld(ctx, sx as unknown as NU, sz as unknown as NU, skirtDrop as unknown as NF);
  };

  /** per-vertex trunk/leaf wind offset (IDENTICAL math for both channels — deduped here).
   *  ?fp16w routes it through the f16 wgslFn (nanWindF16); off = the byte-identical f32 TSL
   *  math (node graph unchanged from the prior inline blocks). The offset is a small delta
   *  added to the f32 world pos ⇒ fp16 precision-safe; raster+resolve share this ⇒ bit-consistent. */
  const windOffset = (w: TrunkWindFields, localY: NF, flex: NF): NV3 => {
    if (fp16Wind) {
      return nanWindF16()({
        h0: w.h0,
        dirX: w.dirX,
        dirY: w.dirY,
        leanBase: w.leanBase,
        swayABase: w.swayABase,
        swayPhase: w.swayPhase,
        swayXPhase: w.swayXPhase,
        branchBase: w.branchBase,
        localY,
        flex,
      }) as unknown as NV3;
    }
    const yn = localY.div(localY.add(w.h0));
    const prof = yn.mul(yn).mul(1.7).add(flex.mul(0.3)).min(1.6);
    const swayA = w.swayABase.mul(prof);
    const sway = w.swayPhase.mul(swayA);
    const swayX = w.swayXPhase.mul(swayA).mul(0.45);
    const along = w.leanBase.mul(prof).add(sway).add(w.branchBase.mul(flex));
    const dy = along.abs().add(swayX.abs()).mul(flex).mul(-0.2);
    return vec3(
      w.dirX.mul(along).sub(w.dirY.mul(swayX)),
      dy,
      w.dirY.mul(along).add(w.dirX.mul(swayX)),
    ) as unknown as NV3;
  };

  /** explicit-mesh vertex BY INDEX: gpu.verts[vi·VERT_WORDS] → instance-transformed
   *  world pos (+ trunk wind). The SAME fetch runs in the raster (geometry) and the
   *  resolve (barycentric corners), so both reconstruct bit-identical windy positions. */
  const explicitWorldByIndex = (ctx: VertCtx, vi: NU): NV3 => {
    const out = vec3(0).toVar();
    const vb = vi.mul(uint(VERT_WORDS));
    const p = vec3(
      bcU2F(elemU(gpu.verts, vb)),
      bcU2F(elemU(gpu.verts, vb.add(uint(1)))),
      bcU2F(elemU(gpu.verts, vb.add(uint(2)))),
    );
    out.assign(instTransformPoint(ctx.A, ctx.B, ctx.yawSc, p as unknown as NV3));
    // trunk WIND (Wind.vegWindOffset assembly, flutter omitted): per-vertex
    // prof/flex scaling of the makeCtx-precomputed per-instance scalars.
    if (wind) {
      If(ctx.channel.equal(uint(TRANSFORM_CHANNEL.trunk)), () => {
        const w = ctx.wind as TrunkWindFields;
        const localY = (p as unknown as NV3).y.mul(ctx.A.w as unknown as NF);
        const vd = elemU(gpu.verts, vb.add(uint(5)));
        const flex = toF(vd.shiftRight(uint(8)).bitAnd(uint(0xff))).div(255);
        out.assign(out.add(windOffset(w, localY as unknown as NF, flex as unknown as NF)));
      });
      // N9-C0: leaf channel — the FULL Wind.vegWindOffset (terms 1–4) so the crown
      // SWAYS WITH the trunk (lean+sway+branch, mirroring the trunk block above)
      // PLUS the shared leaf flutter (term 4 via leafFlutterAxes — the SAME advected-
      // fbm shimmer the old foliage material uses, not a reinvention). vdata.y=flex,
      // vdata.z=phase: the same baked attributes the old path reads.
      If(ctx.channel.equal(uint(TRANSFORM_CHANNEL.leaf)), () => {
        // N9-C0: leaf uses the SAME lean+sway+branch as the trunk (flutter removed — the
        // per-vertex advected-fbm TEXTURE tap was ~96% of the transform cost). So the crown
        // SWAYS WITH the trunk via the identical windOffset() helper.
        const w = ctx.wind as TrunkWindFields;
        const localY = (p as unknown as NV3).y.mul(ctx.A.w as unknown as NF);
        const vd = elemU(gpu.verts, vb.add(uint(5)));
        const flex = toF(vd.shiftRight(uint(8)).bitAnd(uint(0xff))).div(255);
        out.assign(out.add(windOffset(w, localY as unknown as NF, flex as unknown as NF)));
      });
    }
    return out as unknown as NV3;
  };

  // ── composable fetch ARMS (PERF task #76 — kernel-split by cluster class) ──────────
  // Each arm is a self-contained block that writes a world pos into `out`; the three
  // fetch entry points compose {explicit} | {heightfield} | BOTH by `variant`. A
  // specialized kernel (makeFetch(..., 'explicit'|'terrain')) then compiles ONLY its
  // own arm and never reserves the other class's registers. When variant==='both' the
  // emitted node graph is IDENTICAL to the prior inline If(isHF).Else (calling a closure
  // emits the same nodes as inline code) ⇒ byte-identical default.
  type OutVar = { assign(v: NV3): unknown };

  /** explicit-mesh global vertex index for (localTri, corner∈{0,1,2}) */
  const explicitVi = (ctx: VertCtx, localTri: NU, corner: NU): NU =>
    elemU(gpu.indices, ctx.triStart.add(localTri).mul(uint(3)).add(corner));

  /** heightfield arm, COMPILE-TIME corner v — adaptive-DAG index else window grid */
  const hfArmStatic = (ctx: VertCtx, localTri: NU, v: 0 | 1 | 2, out: OutVar): void => {
    If(ctx.isDAG, () => {
      out.assign(dagWorldByIndex(ctx, explicitVi(ctx, localTri, uint(v))));
    }).Else(() => {
      // window-procedural: implicit regular grid within the cluster's window —
      // no index buffer (sx,sz derived from localTri).
      const quad = localTri.shiftRight(uint(1));
      const odd = localTri.bitAnd(uint(1)).equal(uint(1));
      const col = quad.mod(ctx.qxw);
      const row = quad.div(ctx.qxw);
      let dx: NU;
      let dz: NU;
      if (v === 0) {
        dx = uint(0) as unknown as NU;
        dz = uint(0) as unknown as NU;
      } else if (v === 1) {
        dx = odd.select(uint(1), uint(0));
        dz = uint(1) as unknown as NU;
      } else {
        dx = uint(1) as unknown as NU;
        dz = odd.select(uint(0), uint(1));
      }
      const sx = ctx.gx.add(col).add(dx);
      const sz = ctx.gz.add(row).add(dz);
      out.assign(hfWorld(ctx, sx as unknown as NU, sz as unknown as NU, float(0) as unknown as NF));
    });
  };

  /** heightfield arm, RUNTIME corner (0..2) — select arms EQUAL the static v values */
  const hfArmDyn = (ctx: VertCtx, localTri: NU, corner: NU, out: OutVar): void => {
    If(ctx.isDAG, () => {
      out.assign(dagWorldByIndex(ctx, explicitVi(ctx, localTri, corner)));
    }).Else(() => {
      const quad = localTri.shiftRight(uint(1));
      const odd = localTri.bitAnd(uint(1)).equal(uint(1));
      const col = quad.mod(ctx.qxw);
      const row = quad.div(ctx.qxw);
      const dx = corner
        .equal(uint(1))
        .select(odd.select(uint(1), uint(0)), corner.equal(uint(2)).select(uint(1), uint(0))) as unknown as NU;
      const dz = corner
        .equal(uint(1))
        .select(uint(1), corner.equal(uint(2)).select(odd.select(uint(0), uint(1)), uint(0))) as unknown as NU;
      const sx = ctx.gx.add(col).add(dx);
      const sz = ctx.gz.add(row).add(dz);
      out.assign(hfWorld(ctx, sx as unknown as NU, sz as unknown as NU, float(0) as unknown as NF));
    });
  };

  const fetchWorldVert = (ctx: VertCtx, localTri: NU, v: 0 | 1 | 2): NV3 => {
    const out = vec3(0).toVar();
    if (variant === 'explicit') {
      out.assign(explicitWorldByIndex(ctx, explicitVi(ctx, localTri, uint(v))));
    } else if (variant === 'terrain') {
      hfArmStatic(ctx, localTri, v, out);
    } else {
      If(ctx.isHF, () => {
        hfArmStatic(ctx, localTri, v, out);
      }).Else(() => {
        out.assign(explicitWorldByIndex(ctx, explicitVi(ctx, localTri, uint(v))));
      });
    }
    return out as unknown as NV3;
  };

  const fetchWorldVertDyn = (ctx: VertCtx, localTri: NU, corner: NU): NV3 => {
    const out = vec3(0).toVar();
    if (variant === 'explicit') {
      out.assign(explicitWorldByIndex(ctx, explicitVi(ctx, localTri, corner)));
    } else if (variant === 'terrain') {
      hfArmDyn(ctx, localTri, corner, out);
    } else {
      If(ctx.isHF, () => {
        hfArmDyn(ctx, localTri, corner, out);
      }).Else(() => {
        out.assign(explicitWorldByIndex(ctx, explicitVi(ctx, localTri, corner)));
      });
    }
    return out as unknown as NV3;
  };

  const fetchWorldVertByIndex = (ctx: VertCtx, vi: NU): NV3 => {
    const out = vec3(0).toVar();
    if (variant === 'explicit') {
      out.assign(explicitWorldByIndex(ctx, vi));
    } else if (variant === 'terrain') {
      // only the adaptive-DAG convention has explicit vertex indices; window-grid
      // clusters have vcompact count=0, so the projection pre-pass never calls this.
      out.assign(dagWorldByIndex(ctx, vi));
    } else {
      If(ctx.isHF, () => {
        out.assign(dagWorldByIndex(ctx, vi));
      }).Else(() => {
        out.assign(explicitWorldByIndex(ctx, vi));
      });
    }
    return out as unknown as NV3;
  };

  const meshWord = (meshId: NU, word: number): NU =>
    elemU(gpu.meshes, meshId.mul(uint(MESH_WORDS)).add(uint(word)));

  return { makeCtx, fetchWorldVert, fetchWorldVertDyn, fetchWorldVertByIndex, meshWord };
}
