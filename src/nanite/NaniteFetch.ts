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
import { If, clamp, float, mix, smoothstep, texture, uint, vec2, vec3 } from 'three/tsl';
import type { NB, NF, NU, NV2, NV3, NV4 } from '../gpu/TSLTypes';
import { DISP } from '../render/TerrainMaterial';
import { PERIOD_FBM, PERIOD_RID, PERIOD_VAL } from '../gpu/passes/NoiseBake';
import { WORLD_SIZE } from '../world/WorldConst';
import { MESH_WORDS, VERT_WORDS } from './GeometryRegistry';
import type { RegistryGpu } from './GeometryRegistry';
import { instTransformPoint, instYaw, type InstYaw } from './NaniteCommon';
import type { UniformV3 } from './Tsl';
import { bcU2F, elemU, maxU, minU, texLoadR, toF } from './Tsl';

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

export function makeFetch(gpu: RegistryGpu, heightTex: Texture, disp?: TerrainDisp): NaniteFetch {
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
    const winW = elemU(gpu.meshes, mBase.add(uint(6))).shiftRight(uint(24)).toVar();
    const quadsX = elemU(gpu.meshes, mBase.add(uint(10))).bitAnd(uint(0xffff)).toVar();
    const gx = triStart.bitAnd(uint(0xffff)).mul(winW).toVar();
    const gz = triStart.shiftRight(uint(16)).mul(winW).toVar();
    const qxw = minU(winW, maxU(quadsX, gx).sub(gx)).toVar();
    const oX = bcU2F(elemU(gpu.meshes, mBase.add(uint(7)))).toVar();
    const oZ = bcU2F(elemU(gpu.meshes, mBase.add(uint(8)))).toVar();
    const cell = bcU2F(elemU(gpu.meshes, mBase.add(uint(9)))).toVar();
    return {
      isHF: isHF as unknown as NB,
      A,
      B,
      yawSc: instYaw(B),
      triStart,
      triCount,
      meshId,
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
    });
    return out as unknown as NV3;
  };

  const meshWord = (meshId: NU, word: number): NU =>
    elemU(gpu.meshes, meshId.mul(uint(MESH_WORDS)).add(uint(word)));

  return { makeCtx, fetchWorldVert, meshWord };
}
