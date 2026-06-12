/**
 * N2 culling chain (NANITE.md "Culling (N2)") — registry-fed two-level cull:
 *
 *   kClear      counters → 0
 *   kInstCull   one thread per instance: world-sphere frustum test (mesh
 *               sphere × instance transform + swayPad), discrete LOD select
 *               (lodNext/lodDist chain walk on camera distance), then push
 *               ceil(clusterCount/64) chunk items (bulk atomicAdd + serial
 *               writes — the spike's serial per-instance cluster loop, fixed)
 *   kChunkArgs  clamp count → qChunks[0], 2D-split indirect dispatch args
 *   kClusterCull one 64-thread workgroup per chunk item, one cluster per
 *               thread: frustum + cone backface (yaw-rotated axis, CONE_SLACK
 *               absorbs lean/wind) → push (instId, clusterId) raster items
 *   kRasterArgs clamp count → qRaster[0], 2D-split dispatch args (128/item)
 *
 * Heightfield meshes/clusters carry world-space spheres (identity instance)
 * and never cone-cull (cos = −1 at pack). Occlusion (prev-HZB) and the
 * two-phase reject loop land in N2-C2/C3 — the kernel seams are marked.
 *
 * Queue law (F14): explicit caps, clamp on overflow (HUD flag via
 * readCounts), counters-in-queue slot 0, 2D-split indirect dispatch.
 */

import { IndirectStorageBufferAttribute, StorageBufferAttribute } from 'three/webgpu';
import type { Renderer } from 'three/webgpu';
import { Fn, If, Loop, atomicAdd, atomicStore, dot, float, instanceIndex, uint, vec3 } from 'three/tsl';
import type { NB, NF, NU, NV3, NV4 } from '../gpu/TSLTypes';
import { LOD_NONE, MESH_FLAG_HEIGHTFIELD, MESH_WORDS, readCluster, readMesh } from './GeometryRegistry';
import type { RegistryGpu } from './GeometryRegistry';
import {
  CHUNK_CLUSTERS,
  CONE_SLACK,
  DISPATCH_ROW,
  QCHUNK_CAP,
  QRASTER_CAP,
  instRotateDir,
  instSphereRadius,
  instTransformPoint,
  instYaw,
  type NaniteCam,
} from './NaniteCommon';
import {
  aLoadU,
  bcU2F,
  dispatch,
  dispatchIndirect,
  elemU,
  localX,
  loopU,
  maxU,
  minU,
  readBuffer,
  returnIf,
  sU32Views,
  sUvec2,
  uv2,
  wgLinear,
} from './Tsl';
import type { BufOf, UV2 } from './Tsl';

interface ComputeKernel {
  setName(name: string): unknown;
}

export interface NaniteCullCounts {
  /** chunk items pushed (pre-clamp) */
  chunks: number;
  /** raster items = visible clusters (pre-clamp) */
  visClusters: number;
  /** non-null when a queue clamped this frame */
  overflow: string | null;
}

export interface NaniteCullChain {
  /** raster work queue: [0] = (count, 0), items (instId, clusterId) at 1.. */
  qRasterRO: BufOf<UV2>;
  qRasterAttr: StorageBufferAttribute;
  /** 2D-split dispatch args for one workgroup per raster item */
  rasterDispatchAttr: IndirectStorageBufferAttribute;
  /** run the phase-1 chain (clear → instance cull → cluster cull → args) */
  run(renderer: Renderer): void;
  readCounts(renderer: Renderer): Promise<NaniteCullCounts>;
}

export function buildNaniteCull(
  gpu: RegistryGpu,
  instanceCount: number,
  cam: NaniteCam,
  /** prev-HZB occlusion test (NaniteHzb.sphereOccluded); null = ?occl=0 */
  sphereOccluded: ((center: NV3, radius: NF) => NB) | null,
): NaniteCullChain {
  // ---- buffers ---------------------------------------------------------------
  // counters: [0] chunk pushes, [1] raster pushes, [2,3] reserved (C3 rejects)
  const countersAttr = new StorageBufferAttribute(new Uint32Array(4), 1);
  const counters = sU32Views(countersAttr, 4).atomic;

  // chunk queue: slot 0 = (clamped count, 0); item y packs clusterBase(26b) |
  // (count−1)(6b) — base < 67M covers 5×-density + DAG headroom
  const qChunksAttr = new StorageBufferAttribute(new Uint32Array((QCHUNK_CAP + 1) * 2), 2);
  const qChunksV = sUvec2(qChunksAttr, QCHUNK_CAP + 1);

  const qRasterAttr = new StorageBufferAttribute(new Uint32Array((QRASTER_CAP + 1) * 2), 2);
  const qRasterV = sUvec2(qRasterAttr, QRASTER_CAP + 1);

  const chunkDispatchAttr = new IndirectStorageBufferAttribute(new Uint32Array(3), 3);
  const chunkDispatch = sU32Views(chunkDispatchAttr as unknown as StorageBufferAttribute, 3).rw;
  const rasterDispatchAttr = new IndirectStorageBufferAttribute(new Uint32Array(3), 3);
  const rasterDispatch = sU32Views(rasterDispatchAttr as unknown as StorageBufferAttribute, 3).rw;

  // ---- kClear ------------------------------------------------------------------
  const kClear = Fn(() => {
    If(instanceIndex.lessThan(uint(4)), () => {
      atomicStore(counters.element(instanceIndex), uint(0));
    });
  })().compute(4, [4]);
  (kClear as unknown as ComputeKernel).setName('nanCullClear');

  // ---- frustum test helper (shared TSL) ------------------------------------------
  const frustumVisible = (center: NV3, radius: NF): NF => {
    const visible = float(1).toVar();
    Loop(6, ({ i: pi }) => {
      const plane = cam.planes.element(pi);
      const d = dot(plane.xyz, center).add(plane.w) as unknown as NF;
      If(d.lessThan(radius.negate()), () => {
        visible.assign(0);
      });
    });
    return visible as unknown as NF;
  };

  // ---- kInstCull -------------------------------------------------------------------
  const kInstCull = Fn(() => {
    returnIf(instanceIndex.greaterThanEqual(uint(instanceCount)));
    const A = gpu.instances.element(instanceIndex.mul(uint(2))).toVar() as unknown as NV4;
    const B = gpu.instances.element(instanceIndex.mul(uint(2)).add(uint(1))).toVar() as unknown as NV4;
    const headId = elemU(gpu.instanceMesh, instanceIndex).toVar();
    const head = readMesh(gpu.meshes, headId);

    // world bounding sphere of the whole instance (heightfield spheres are
    // world-space already — identity instance)
    const yawSc = instYaw(B);
    const isHF = head.flags.bitAnd(uint(MESH_FLAG_HEIGHTFIELD)).notEqual(uint(0)).toVar();
    const centerW = vec3(0).toVar();
    const radiusW = float(0).toVar();
    If(isHF, () => {
      centerW.assign(head.sphere.xyz);
      radiusW.assign(head.sphere.w);
    }).Else(() => {
      centerW.assign(instTransformPoint(A, B, yawSc, head.sphere.xyz as unknown as NV3));
      radiusW.assign(instSphereRadius(A, B, head.sphere.w, head.swayPad));
    });
    returnIf(frustumVisible(centerW as unknown as NV3, radiusW as unknown as NF).lessThan(0.5));
    if (sphereOccluded) {
      // C3 seam: record occlusion-only instance rejects for phase 2
      returnIf(sphereOccluded(centerW as unknown as NV3, radiusW as unknown as NF));
    }

    // discrete LOD select: walk the chain while the camera is beyond lodDist
    // (chains are ≤ 3 hops today; 4 iterations = headroom)
    const dist = cam.camPos.sub(A.xyz).length().toVar();
    const sel = headId.toVar();
    const selStart = head.clusterStart.toVar();
    const selCount = head.clusterCount.toVar();
    const nextId = head.lodNext.toVar();
    const nextDist = head.lodDist.toVar();
    Loop(4, () => {
      If(nextId.notEqual(uint(LOD_NONE)).and(dist.greaterThan(nextDist)), () => {
        sel.assign(nextId);
        const m = uint(nextId).mul(uint(MESH_WORDS)).toVar();
        selStart.assign(elemU(gpu.meshes, m));
        selCount.assign(elemU(gpu.meshes, m.add(uint(1))));
        nextId.assign(elemU(gpu.meshes, m.add(uint(4))));
        nextDist.assign(bcU2F(elemU(gpu.meshes, m.add(uint(5)))));
      });
    });
    // chain tail's lodDist = max draw envelope (0 = unlimited): beyond it the
    // old path shows impostors — the hybrid's far field, not ours until N8
    returnIf(
      nextId
        .equal(uint(LOD_NONE))
        .and(nextDist.greaterThan(0))
        .and(dist.greaterThan(nextDist)),
    );

    // push chunk items (bulk reservation, per-item cap guard)
    const nChunks = selCount.add(uint(CHUNK_CLUSTERS - 1)).div(uint(CHUNK_CLUSTERS)).toVar();
    const slotBase = (atomicAdd(counters.element(0), nChunks) as unknown as NU).toVar();
    loopU(uint(0), nChunks, (i) => {
      const slot = slotBase.add(i);
      If(slot.lessThan(uint(QCHUNK_CAP)), () => {
        const cBase = selStart.add(i.mul(uint(CHUNK_CLUSTERS)));
        const cnt = minU(uint(CHUNK_CLUSTERS), selStart.add(selCount).sub(cBase));
        qChunksV.rw
          .element(slot.add(uint(1)))
          .assign(uv2(instanceIndex, cBase.bitOr(cnt.sub(uint(1)).shiftLeft(uint(26)))));
      });
    });
  })().compute(instanceCount, [64]);
  (kInstCull as unknown as ComputeKernel).setName('nanInstCull');

  // ---- kChunkArgs ----------------------------------------------------------------
  const kChunkArgs = Fn(() => {
    const n = minU(aLoadU(counters.element(0)), uint(QCHUNK_CAP));
    qChunksV.rw.element(0).assign(uv2(n, 0));
    const rows = n.add(uint(DISPATCH_ROW - 1)).div(uint(DISPATCH_ROW));
    chunkDispatch.element(0).assign(minU(n, uint(DISPATCH_ROW)));
    chunkDispatch.element(1).assign(maxU(rows, uint(1)));
    chunkDispatch.element(2).assign(uint(1));
  })().compute(1, [1]);
  (kChunkArgs as unknown as ComputeKernel).setName('nanChunkArgs');

  // ---- kClusterCull -----------------------------------------------------------------
  const kClusterCull = Fn(() => {
    const itemIdx = wgLinear(DISPATCH_ROW).toVar();
    const itemCount = qChunksV.ro.element(0).x;
    returnIf(itemIdx.greaterThanEqual(itemCount));
    const item = qChunksV.ro.element(itemIdx.add(uint(1)));
    const instId = item.x.toVar();
    const cBase = item.y.bitAnd(uint(0x3ffffff));
    const cnt = item.y.shiftRight(uint(26)).add(uint(1));
    If(localX().lessThan(cnt), () => {
      const ci = cBase.add(localX()).toVar();
      const c = readCluster(gpu.clusters, ci);
      const A = gpu.instances.element(instId.mul(uint(2))).toVar() as unknown as NV4;
      const B = gpu.instances.element(instId.mul(uint(2)).add(uint(1))).toVar() as unknown as NV4;
      const yawSc = instYaw(B);
      const isHF = c.flags.bitAnd(uint(1)).notEqual(uint(0)).toVar();
      const centerW = vec3(0).toVar();
      const radiusW = float(0).toVar();
      If(isHF, () => {
        centerW.assign(c.sphere.xyz);
        radiusW.assign(c.sphere.w);
      }).Else(() => {
        const swayPad = bcU2F(elemU(gpu.meshes, c.meshId.mul(uint(MESH_WORDS)).add(uint(11))));
        centerW.assign(instTransformPoint(A, B, yawSc, c.sphere.xyz as unknown as NV3));
        radiusW.assign(instSphereRadius(A, B, c.sphere.w, swayPad));
      });

      const visible = frustumVisible(centerW as unknown as NV3, radiusW as unknown as NF).toVar();

      // cone backface (explicit meshes only; conservative slack for lean/wind)
      If(visible.greaterThan(0.5).and(c.coneCos.greaterThan(-0.99)).and(isHF.not()), () => {
        const sinTest = float(1)
          .sub(c.coneCos.mul(c.coneCos))
          .max(0)
          .sqrt()
          .add(CONE_SLACK)
          .toVar();
        If(sinTest.lessThan(1), () => {
          const axisW = instRotateDir(yawSc, c.coneAxis);
          const toC = (centerW as unknown as NV3).sub(cam.camPos).toVar();
          const d = toC.length();
          If(dot(toC as unknown as NV3, axisW).greaterThan(d.mul(sinTest).add(radiusW)), () => {
            visible.assign(0);
          });
        });
      });
      if (sphereOccluded) {
        // C3 seam: record occlusion-only cluster rejects for phase 2
        If(visible.greaterThan(0.5), () => {
          If(sphereOccluded(centerW as unknown as NV3, radiusW as unknown as NF), () => {
            visible.assign(0);
          });
        });
      }

      If(visible.greaterThan(0.5), () => {
        const slot = atomicAdd(counters.element(1), uint(1)) as unknown as NU;
        If(slot.lessThan(uint(QRASTER_CAP)), () => {
          qRasterV.rw.element(slot.add(uint(1))).assign(uv2(instId, ci));
        });
      });
    });
  })().compute(QCHUNK_CAP * CHUNK_CLUSTERS, [CHUNK_CLUSTERS]);
  (kClusterCull as unknown as ComputeKernel).setName('nanClusterCull');

  // ---- kRasterArgs ---------------------------------------------------------------
  const kRasterArgs = Fn(() => {
    const n = minU(aLoadU(counters.element(1)), uint(QRASTER_CAP));
    qRasterV.rw.element(0).assign(uv2(n, 0));
    const rows = n.add(uint(DISPATCH_ROW - 1)).div(uint(DISPATCH_ROW));
    rasterDispatch.element(0).assign(minU(n, uint(DISPATCH_ROW)));
    rasterDispatch.element(1).assign(maxU(rows, uint(1)));
    rasterDispatch.element(2).assign(uint(1));
  })().compute(1, [1]);
  (kRasterArgs as unknown as ComputeKernel).setName('nanRasterArgs');

  const run = (renderer: Renderer): void => {
    dispatch(renderer, kClear);
    dispatch(renderer, kInstCull);
    dispatch(renderer, kChunkArgs);
    dispatchIndirect(renderer, kClusterCull, chunkDispatchAttr);
    dispatch(renderer, kRasterArgs);
  };

  const readCounts = async (renderer: Renderer): Promise<NaniteCullCounts> => {
    const buf = await readBuffer(renderer, countersAttr, 0, 8);
    const u = new Uint32Array(buf);
    const chunks = u[0] ?? 0;
    const visClusters = u[1] ?? 0;
    let overflow: string | null = null;
    if (chunks > QCHUNK_CAP) overflow = `qChunks ${chunks} > ${QCHUNK_CAP}`;
    if (visClusters > QRASTER_CAP) {
      overflow = `${overflow ? `${overflow}; ` : ''}qRaster ${visClusters} > ${QRASTER_CAP}`;
    }
    return { chunks, visClusters, overflow };
  };

  return { qRasterRO: qRasterV.ro, qRasterAttr, rasterDispatchAttr, run, readCounts };
}
