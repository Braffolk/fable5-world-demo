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
import { If, uint, vec3 } from 'three/tsl';
import type { NB, NF, NU, NV3, NV4 } from '../gpu/TSLTypes';
import { MESH_WORDS, VERT_WORDS } from './GeometryRegistry';
import type { RegistryGpu } from './GeometryRegistry';
import { instTransformPoint, instYaw, type InstYaw } from './NaniteCommon';
import { bcU2F, elemU, maxU, minU, texLoadR, toF } from './Tsl';

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

export function makeFetch(gpu: RegistryGpu, heightTex: Texture): NaniteFetch {
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
      out.assign(vec3(toF(sx).mul(ctx.cell).add(ctx.oX), h, toF(sz).mul(ctx.cell).add(ctx.oZ)));
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
