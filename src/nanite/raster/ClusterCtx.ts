/**
 * ClusterCtx.ts — the per-cluster makeCtx PRE-PASS (task 1, radical).
 *
 * The wgcache thread-0 makeCtx broadcast was 1 ACTIVE LANE / 127 masked doing the
 * metadata unpack + up-to-six gust/disp samples + wind setup + the clhw classify,
 * behind a workgroupBarrier — a divergence + barrier the raster occupancy cannot hide.
 * This moves it OUT to a ONE-THREAD-PER-CLUSTER pre-pass (`nanClusterCtxPrepass`) that
 * computes makeCtx once and writes the 35-word ctx (12 u32 + 23 f32-as-bits) to a global
 * buffer; the raster then reads it with cache-coherent loads (itemIdx is UNIFORM per
 * workgroup ⇒ one L1 line for all 128 lanes). BONUS: makeCtx leaving the raster SHEDS
 * gpu.clusters/instances/meshes from its bindings ⇒ 9 storage buffers, under the ceiling.
 * WORLD1 (singlePass) only. Buffer = QRASTER_CAP × 35 u32 (world QRASTER_CAP = 1M ⇒ 140 MB);
 * every visible cluster is rewritten each frame so the initial contents are irrelevant.
 */

import { Fn, instanceIndex, uint } from 'three/tsl';
import { StorageBufferAttribute } from 'three/webgpu';
import type { NB, NF, NU } from '../../gpu/TSLTypes';
import { MESH_WORDS } from '../GeometryRegistry';
import type { RegistryGpu } from '../GeometryRegistry';
import { QRASTER_CAP, type NaniteCam } from '../NaniteCommon';
import type { TrunkWindOpt, VertCtx } from '../NaniteFetch';
import { clusterHwClass } from '../NaniteHwClass';
import { bcF2U, elemU, returnIf, sU32Views } from '../Tsl';
import type { BufOf, UV2 } from '../Tsl';

interface ComputeKernel {
  setName(name: string): unknown;
}

// uint slots: isHF,isDAG,triStart,triCount,meshId,channel,gx,gz,qxw,twoSided,matClass,clhw
export const CTX_U = 12;
// float slots: A.xyzw,B.xyzw,oX,oZ,cell,wind[11..20],yawSc.cy,yawSc.sy
const CTX_F = 23;
export const CTX_STRIDE = CTX_U + CTX_F; // 35 u32 / cluster

type U32Views = ReturnType<typeof sU32Views>;

export interface ClusterCtxBundle {
  clusterCtxAttr: StorageBufferAttribute | null;
  clusterCtxV: U32Views | null;
  kClusterCtx: unknown | null;
}

export function buildClusterCtx(p: {
  /** = singlePass: only the world1 path allocates the buffer + runs the pre-pass. */
  ctxPrepass: boolean;
  gpu: RegistryGpu;
  cam: NaniteCam;
  qRasterRO: BufOf<UV2>;
  makeCtx: (instId: NU, ci: NU) => VertCtx;
  projK: NF;
  clhwMax: number;
  wind?: TrunkWindOpt;
  /** bool→uint select (shared with the world1 broadcast; kept identical). */
  b2u: (b: NB) => NU;
}): ClusterCtxBundle {
  const { ctxPrepass, gpu, cam, qRasterRO, makeCtx, projK, clhwMax, wind, b2u } = p;

  const clusterCtxAttr = ctxPrepass
    ? new StorageBufferAttribute(new Uint32Array(QRASTER_CAP * CTX_STRIDE), 1)
    : null;
  if (clusterCtxAttr) clusterCtxAttr.name = 'nanClusterCtx';
  const clusterCtxV = clusterCtxAttr
    ? sU32Views(clusterCtxAttr, QRASTER_CAP * CTX_STRIDE)
    : null;

  // ONE thread per visible cluster via a DIRECT fixed grid (instanceIndex) + early-out —
  // no indirect-arg-in-batch hazard, and launch of the ~15625 idle-tail workgroups is <
  // the makeCtx it replaces. Computes the UNIFIED makeCtx (variant 'both') and writes the
  // 35-word ctx the world1 raster reads.
  const kClusterCtx =
    ctxPrepass && clusterCtxV
      ? (() => {
          const kn = Fn(() => {
            const tid = instanceIndex;
            const count = qRasterRO.element(0).x;
            returnIf(tid.greaterThanEqual(count));
            const item = qRasterRO.element(tid.add(uint(1)));
            const instId = item.x.toVar();
            const ci = item.y.toVar();
            const c = makeCtx(instId, ci);
            const base = tid.mul(uint(CTX_STRIDE)).toVar();
            const wU = (i: number, v: NU): void =>
              void (
                clusterCtxV.rw.element(base.add(uint(i))) as unknown as {
                  assign(x: NU): unknown;
                }
              ).assign(v);
            const wF = (i: number, v: NF): void =>
              void (
                clusterCtxV.rw.element(
                  base.add(uint(CTX_U + i)),
                ) as unknown as { assign(x: NU): unknown }
              ).assign(bcF2U(v as unknown as NF));
            wU(0, b2u(c.isHF));
            wU(1, b2u(c.isDAG));
            wU(2, c.triStart);
            wU(3, c.triCount);
            wU(4, c.meshId);
            wU(5, c.channel);
            wU(6, c.gx);
            wU(7, c.gz);
            wU(8, c.qxw);
            wU(9, b2u(c.twoSided));
            wU(
              10,
              elemU(gpu.meshes, c.meshId.mul(uint(MESH_WORDS)).add(uint(6)))
                .shiftRight(uint(8))
                .bitAnd(uint(0xff)),
            );
            // slot 11 = SW/HW cluster classify (the permanent-default split); the world1
            // raster reads it for the uniform HW-cluster skip, and the projection pre-pass
            // reads it (with slot 0) for its terrain-HW skip — same helper as kHwPartition,
            // so routing and projection-skip stay consistent by construction.
            wU(
              11,
              b2u(clusterHwClass(gpu, cam, projK, instId, ci, clhwMax)),
            );
            wF(0, c.A.x as unknown as NF);
            wF(1, c.A.y as unknown as NF);
            wF(2, c.A.z as unknown as NF);
            wF(3, c.A.w as unknown as NF);
            wF(4, c.B.x as unknown as NF);
            wF(5, c.B.y as unknown as NF);
            wF(6, c.B.z as unknown as NF);
            wF(7, c.B.w as unknown as NF);
            wF(8, c.oX);
            wF(9, c.oZ);
            wF(10, c.cell);
            if (wind) {
              const w = c.wind as NonNullable<VertCtx['wind']>;
              wF(11, w.h0);
              wF(12, w.dirX);
              wF(13, w.dirY);
              wF(14, w.leanBase);
              wF(15, w.swayABase);
              wF(16, w.swayPhase);
              wF(17, w.ph);
              wF(18, w.branchBase);
              wF(19, w.flutBase);
              wF(20, w.swayXPhase);
            }
            wF(21, c.yawSc.cy);
            wF(22, c.yawSc.sy);
          })().compute(QRASTER_CAP, [64]);
          (kn as unknown as ComputeKernel).setName('nanClusterCtxPrepass');
          return kn;
        })()
      : null;

  return { clusterCtxAttr, clusterCtxV, kClusterCtx };
}
