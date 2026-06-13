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
import { If, clamp, float, mix, smoothstep, texture, time, uint, vec2, vec3 } from 'three/tsl';
import type { NB, NF, NU, NV2, NV3, NV4 } from '../gpu/TSLTypes';
import { DISP } from '../render/TerrainMaterial';
import { PERIOD_FBM, PERIOD_RID, PERIOD_VAL } from '../gpu/passes/NoiseBake';
import { WORLD_SIZE } from '../world/WorldConst';
import { gustAt, gustLagAt, windExposure, windU, WIND_LAG_M } from '../render/Wind';
import { MESH_WORDS, TRANSFORM_CHANNEL, VERT_WORDS } from './GeometryRegistry';
import type { RegistryGpu } from './GeometryRegistry';
import { instTransformPoint, instYaw, type InstYaw } from './NaniteCommon';
import type { UniformV3 } from './Tsl';
import { bcU2F, elemU, maxU, minU, texLoadR, toF } from './Tsl';

/** cheap pcg-ish hash of an instance slot → 0..1 (mirror of VegInstance.slotHash
 *  — the trunk wind needs the SAME per-instance phase the old path baked) */
function slotHash(slot: NU, salt: number): NF {
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

/** per-instance trunk-wind scalars precomputed in makeCtx (the 4 gust texture
 *  samples happen ONCE per instance here, not per rasterised corner). The
 *  per-vertex prof/flex scaling is applied in fetchWorldVert. */
interface TrunkWindFields {
  h0: NF;
  dirX: NF;
  dirY: NF;
  leanBase: NF;
  swayABase: NF;
  natW: NF;
  ph: NF;
  branchBase: NF;
}

/**
 * Terrain micro-displacement inputs ('terrain' transform channel, N4-C1):
 * the EXACT TerrainTiles vertex formula (world-space fields, distance-faded
 * 45→85 m) applied to heightfield vertices at fetch time, so the raster, the
 * HW passes and the resolve all see the displaced surface. The nanitedbg
 * views omit this (hwref's CPU build has no GPU noise textures — parity
 * stays an undisplaced-vs-undisplaced compare; the full-frame C1 gate vs
 * ?nanite=0 is what verifies displacement).
 */
export interface TerrainDisp {
  normalTex: StorageTexture;
  biomeTex: StorageTexture;
  fieldsTex: StorageTexture;
  noiseA: StorageTexture;
  noiseB: StorageTexture;
  camPos: UniformV3;
}

/** per-(instance, cluster) decode shared by the 3 corner fetches */
export interface VertCtx {
  isHF: NB;
  A: NV4;
  B: NV4;
  yawSc: InstYaw;
  triStart: NU;
  triCount: NU;
  meshId: NU;
  /** transform channel (TRANSFORM_CHANNEL) — 'trunk' (1) gets wind */
  channel: NU;
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
  /** mesh-record word 6: matClass u8 (bits 8–15) etc. */
  meshWord(meshId: NU, word: number): NU;
}

export function makeFetch(
  gpu: RegistryGpu,
  heightTex: Texture,
  disp?: TerrainDisp,
  wind?: TrunkWindOpt,
): NaniteFetch {
  const makeCtx = (instId: NU, ci: NU): VertCtx => {
    const cBase = ci.mul(uint(8)).toVar();
    const triStart = elemU(gpu.clusters, cBase.add(uint(6))).toVar();
    const w7 = elemU(gpu.clusters, cBase.add(uint(7))).toVar();
    const triCount = w7.bitAnd(uint(0xff)).toVar();
    const isHF = w7.shiftRight(uint(8)).bitAnd(uint(0xff)).bitAnd(uint(1)).notEqual(uint(0)).toVar();
    const meshId = w7.shiftRight(uint(16)).toVar();
    const A = gpu.instances.element(instId.mul(uint(2))).toVar() as unknown as NV4;
    const B = gpu.instances.element(instId.mul(uint(2)).add(uint(1))).toVar() as unknown as NV4;
    const mBase = meshId.mul(uint(MESH_WORDS)).toVar();
    const w6 = elemU(gpu.meshes, mBase.add(uint(6))).toVar();
    const winW = w6.shiftRight(uint(24)).toVar();
    const channel = w6.bitAnd(uint(0xff)).toVar();
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
    if (wind) {
      const matParam = elemU(gpu.meshes, mBase.add(uint(7))).toVar();
      const profile = matParam.shiftRight(uint(8)).bitAnd(uint(0xff)).toVar();
      const d = vec2(windU.dir as unknown as NV2);
      const h0 = float(0).toVar();
      const leanBase = float(0).toVar();
      const swayABase = float(0).toVar();
      const natW = float(0).toVar();
      const ph = float(0).toVar();
      const branchBase = float(0).toVar();
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
        const instPhase = slotHash(instId, 211).toVar();
        const fJit = instPhase.mul(7.31).fract();
        natW.assign(fJit.mul(0.3).add(0.15).mul(6.2832 * 1).mul(freq).div(A.w.max(0.25).sqrt()));
        ph.assign(instPhase.mul(6.2832));
        const brAtten = float(1).sub(dist.sub(160).div(140).clamp(0, 1));
        branchBase.assign(gL.sub(0.45).mul(s).mul(eks).mul(0.55).mul(brAtten));
      });
      windFields = {
        h0: h0 as unknown as NF,
        dirX: d.x as unknown as NF,
        dirY: d.y as unknown as NF,
        leanBase: leanBase as unknown as NF,
        swayABase: swayABase as unknown as NF,
        natW: natW as unknown as NF,
        ph: ph as unknown as NF,
        branchBase: branchBase as unknown as NF,
      };
    }
    return {
      isHF: isHF as unknown as NB,
      A,
      B,
      yawSc: instYaw(B),
      triStart,
      triCount,
      meshId,
      channel,
      wind: windFields,
      gx,
      gz,
      qxw,
      oX: oX as unknown as NF,
      oZ: oZ as unknown as NF,
      cell: cell as unknown as NF,
    };
  };

  const fetchWorldVert = (ctx: VertCtx, localTri: NU, v: 0 | 1 | 2): NV3 => {
    const out = vec3(0).toVar();
    If(ctx.isHF, () => {
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
      const h = texLoadR(heightTex, sx, sz);
      const wx = toF(sx).mul(ctx.cell).add(ctx.oX);
      const wz = toF(sz).mul(ctx.cell).add(ctx.oZ);
      if (disp) {
        // TerrainTiles micro-displacement, verbatim (world-space fields;
        // amplitude gated by slope/rockExposure/snow, faded 45→85 m)
        const wpos = vec2(wx, wz);
        const camD = wpos.sub(vec3(disp.camPos).xz).length();
        const dOut = float(0).toVar();
        If(camD.lessThan(float(DISP.fade1)), () => {
          const uvV = wpos.div(WORLD_SIZE).add(0.5) as unknown as NV2;
          const nsV = texture(disp.normalTex, uvV, 0) as unknown as NV4;
          const bioV = texture(disp.biomeTex, uvV, 0) as unknown as NV4;
          const fldV = texture(disp.fieldsTex, uvV, 0) as unknown as NV4;
          const rockK = smoothstep(DISP.slopeKnee0, DISP.slopeKnee1, nsV.w).max(
            bioV.a.mul(0.85),
          );
          const gravelK = smoothstep(0.32, 0.7, fldV.y)
            .max(smoothstep(0.02, 0.2, fldV.z))
            .mul(float(DISP.gravel));
          const dispAmp = (mix(float(DISP.base), float(DISP.rock), rockK) as unknown as NF)
            .max(gravelK)
            .mul(bioV.g.mul(0.75).oneMinus())
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
        out.assign(vec3(wx, h.add(dOut), wz));
      } else {
        out.assign(vec3(wx, h, wz));
      }
    }).Else(() => {
      const vi = elemU(gpu.indices, ctx.triStart.add(localTri).mul(uint(3)).add(uint(v)));
      const vb = vi.mul(uint(VERT_WORDS));
      const p = vec3(
        bcU2F(elemU(gpu.verts, vb)),
        bcU2F(elemU(gpu.verts, vb.add(uint(1)))),
        bcU2F(elemU(gpu.verts, vb.add(uint(2)))),
      );
      out.assign(instTransformPoint(ctx.A, ctx.B, ctx.yawSc, p as unknown as NV3));
      // trunk WIND (Wind.vegWindOffset assembly, flutter omitted): per-vertex
      // prof/flex scaling of the makeCtx-precomputed per-instance scalars. The
      // SAME shared fetch runs in the raster (geometry) and the resolve (the
      // barycentric corners), so both reconstruct bit-identical windy positions.
      if (wind) {
        If(ctx.channel.equal(uint(TRANSFORM_CHANNEL.trunk)), () => {
          const w = ctx.wind as TrunkWindFields;
          const localY = (p as unknown as NV3).y.mul(ctx.A.w as unknown as NF);
          const vd = elemU(gpu.verts, vb.add(uint(5)));
          const flex = toF(vd.shiftRight(uint(8)).bitAnd(uint(0xff))).div(255);
          const yn = localY.div(localY.add(w.h0));
          const prof = yn.mul(yn).mul(1.7).add(flex.mul(0.3)).min(1.6);
          const swayA = w.swayABase.mul(prof);
          const sway = time.mul(w.natW).add(w.ph).sin().mul(swayA);
          const swayX = time
            .mul(w.natW.mul(1.31))
            .add(w.ph.mul(1.7))
            .sin()
            .mul(swayA)
            .mul(0.45);
          const along = w.leanBase.mul(prof).add(sway).add(w.branchBase.mul(flex));
          const dy = along.abs().add(swayX.abs()).mul(flex).mul(-0.2);
          out.assign(
            out.add(
              vec3(
                w.dirX.mul(along).sub(w.dirY.mul(swayX)),
                dy,
                w.dirY.mul(along).add(w.dirX.mul(swayX)),
              ),
            ),
          );
        });
      }
    });
    return out as unknown as NV3;
  };

  const meshWord = (meshId: NU, word: number): NU =>
    elemU(gpu.meshes, meshId.mul(uint(MESH_WORDS)).add(uint(word)));

  return { makeCtx, fetchWorldVert, meshWord };
}
