/**
 * N2 culling chain (NANITE.md "Culling (N2)") — registry-fed two-level cull
 * with TWO-PHASE occlusion:
 *
 * PHASE 1 (prev-frame HZB, prev VP/camPos — static world: prev matrices,
 * current bounds):
 *   kClear      counters → 0
 *   kInstCull   one thread per instance: world-sphere frustum (mesh sphere ×
 *               instance transform + swayPad) → prev-HZB occlusion
 *               (occlusion-ONLY fail → rejInst, phase 2 re-tests) → discrete
 *               LOD chain walk + hybrid draw envelope (D-N14) → push
 *               ceil(clusterCount/64) chunk items (bulk atomicAdd)
 *   kChunkArgs  clamp → qChunks[0], 2D-split indirect args
 *   kClusterCull one 64-thread workgroup per chunk item: frustum + cone
 *               backface (yaw-rotated axis, CONE_SLACK) → prev-HZB occlusion
 *               (occlusion-only fail → rejClust) → push (instId, clusterId)
 *               raster items
 *   kRasterArgs clamp → qRaster[0] = (n1, 0), args for one wg per item
 *
 * PHASE 2 (after phase-1 raster + fresh HZB; CURRENT VP/camPos):
 *   kPhase2Args reset chunk counter; 2D-split args over both reject lists
 *   kInstCull2  re-test rejected instances vs fresh HZB → survivors LOD +
 *               push chunks (qChunks reused from slot 0)
 *   kChunkArgs + kClusterCull2  fresh-HZB cluster cull (frustum + cone too —
 *               instance-level rejects never reached cluster tests), APPEND
 *               to qRaster (phase-1 payload indices stay stable)
 *   kClusterCull2b  re-test rejected clusters (occlusion only) → append
 *   kRasterArgs2 qRaster[0] = (nTotal, n1); phase-2 args = appended range,
 *               full args = all items (the payload pass walks everything
 *               against final depth)
 *
 * Queue law (F14): explicit caps, clamp on overflow (HUD flag via
 * readCounts), counters-in-queue slot 0, 2D-split indirect dispatch.
 */

import { IndirectStorageBufferAttribute, StorageBufferAttribute } from 'three/webgpu';
import type { Renderer } from 'three/webgpu';
import {
  Fn,
  If,
  Loop,
  atomicAdd,
  atomicStore,
  dot,
  float,
  instanceIndex,
  uint,
  vec3,
} from 'three/tsl';
import type { NB, NF, NU, NV3, NV4 } from '../gpu/TSLTypes';
import {
  CLUSTER_FLAG_DAG,
  LOD_NONE,
  MESH_FLAG_HEIGHTFIELD,
  MESH_WORDS,
  readCluster,
  readDag,
  readMesh,
} from './GeometryRegistry';
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
  elemUW,
  localX,
  loopU,
  maxU,
  minU,
  readBuffer,
  returnIf,
  sU32Views,
  sUvec2,
  uniformF,
  uv2,
  wgLinear,
  type UniformF,
  type UniformMat4,
  type UniformV3,
} from './Tsl';
import type { BufOf, UV2 } from './Tsl';

/** occlusion-reject list caps (F14). Overflow is graceful: victims miss
 *  phase 2 THIS frame only (next frame's phase 1 re-tests everything) —
 *  measured at ?stress=5 bm3: 3.8M inst rejects, flag fired, image intact. */
const REJ_INST_CAP = 1_048_576;
const REJ_CLUST_CAP = 1_048_576;

interface ComputeKernel {
  setName(name: string): unknown;
}

export type SphereOccludedFn = (
  center: NV3,
  radius: NF,
  vp: UniformMat4,
  camPos: UniformV3,
) => NB;

export interface NaniteCullCounts {
  /** chunk items pushed in phase 1 (pre-clamp) */
  chunks: number;
  /** raster items = visible clusters across both phases (pre-clamp) */
  visClusters: number;
  /** phase-2 inputs: occlusion-rejected instances / clusters from phase 1 */
  rejInst: number;
  rejClust: number;
  /** clusters phase 2 brought back (appended past the phase-1 base) */
  p2Appends: number;
  /** N8-D1: emitted clusters carrying a DAG record (the rock screen-error cut) */
  dagClusters: number;
  /** non-null when a queue clamped this frame */
  overflow: string | null;
}

export interface NaniteCullChain {
  /** raster work queue: [0] = (count, phase2Base), items at 1.. */
  qRasterRO: BufOf<UV2>;
  qRasterAttr: StorageBufferAttribute;
  /** phase-1 items (one wg per item) */
  rasterDispatchAttr: IndirectStorageBufferAttribute;
  /** phase-2 appended items only */
  rasterDispatch2Attr: IndirectStorageBufferAttribute;
  /** all items (payload passes) */
  rasterDispatchFullAttr: IndirectStorageBufferAttribute;
  /** phase 1: clear → instance cull → cluster cull → raster args */
  runPhase1(renderer: Renderer): void;
  /** phase 2 (call after phase-1 raster + HZB build): re-test rejects */
  runPhase2(renderer: Renderer): void;
  /** write full-range args WITHOUT re-testing (?phase2=0 A/B + no-occl path) */
  syncFullArgs(renderer: Renderer): void;
  readCounts(renderer: Renderer): Promise<NaniteCullCounts>;
}

export function buildNaniteCull(
  gpu: RegistryGpu,
  instanceCount: number,
  cam: NaniteCam,
  /** prev/fresh-HZB occlusion test (NaniteHzb.sphereOccluded); null = ?occl=0 */
  sphereOccluded: SphereOccludedFn | null,
  /** N5: shadow casters disable cone backface — the cone axis is camera-
   *  relative (instRotateDir vs cam.camPos), but a cluster facing away from the
   *  CAMERA still casts toward the LIGHT, so cone-culling it punches shadow
   *  holes. Default true (camera path). */
  opts?: { coneCull?: boolean; tau?: UniformF },
): NaniteCullChain {
  const coneCull = opts?.coneCull !== false;
  // N8-D1 continuous-LOD cut threshold (screen-error px). The camera path wires
  // ?loderr into this uniform; shadow cascades take the default. projK =
  // (screenH/2)·cot(fovY/2) — for the ortho shadow cams (cotHalfFov stays 1,
  // uH = map size) it lands ~the camera's, so casters track the lit-surface LOD
  // (proper DAG-decoupled caster LOD is S4). Mirrors probe-dag.project exactly.
  const tau = opts?.tau ?? uniformF(1);
  const projK = cam.cotHalfFov.mul(cam.uH).mul(0.5) as unknown as NF;
  // ---- buffers ---------------------------------------------------------------
  // counters: [0] chunk pushes (phase 2 resets for re-expansion), [1] raster
  // pushes (appends across phases), [2] inst rejects, [3] cluster rejects,
  // [4] phase-1 chunk snapshot (HUD)
  const countersAttr = new StorageBufferAttribute(new Uint32Array(8), 1);
  const counters = sU32Views(countersAttr, 8).atomic;

  // chunk queue: slot 0 = (clamped count, 0); item y packs clusterBase(26b) |
  // (count−1)(6b)
  const qChunksAttr = new StorageBufferAttribute(new Uint32Array((QCHUNK_CAP + 1) * 2), 2);
  const qChunksV = sUvec2(qChunksAttr, QCHUNK_CAP + 1);

  const qRasterAttr = new StorageBufferAttribute(new Uint32Array((QRASTER_CAP + 1) * 2), 2);
  const qRasterV = sUvec2(qRasterAttr, QRASTER_CAP + 1);

  // occlusion-only rejects (phase-2 inputs)
  const rejInstAttr = new StorageBufferAttribute(new Uint32Array(REJ_INST_CAP), 1);
  const rejInstV = sU32Views(rejInstAttr, REJ_INST_CAP);
  const rejClustAttr = new StorageBufferAttribute(new Uint32Array(REJ_CLUST_CAP * 2), 2);
  const rejClustV = sUvec2(rejClustAttr, REJ_CLUST_CAP);

  const chunkDispatchAttr = new IndirectStorageBufferAttribute(new Uint32Array(3), 3);
  const chunkDispatch = sU32Views(chunkDispatchAttr as unknown as StorageBufferAttribute, 3).rw;
  const rasterDispatchAttr = new IndirectStorageBufferAttribute(new Uint32Array(3), 3);
  const rasterDispatch = sU32Views(rasterDispatchAttr as unknown as StorageBufferAttribute, 3).rw;
  const rasterDispatch2Attr = new IndirectStorageBufferAttribute(new Uint32Array(3), 3);
  const rasterDispatch2 = sU32Views(rasterDispatch2Attr as unknown as StorageBufferAttribute, 3).rw;
  const rasterDispatchFullAttr = new IndirectStorageBufferAttribute(new Uint32Array(3), 3);
  const rasterDispatchFull = sU32Views(
    rasterDispatchFullAttr as unknown as StorageBufferAttribute,
    3,
  ).rw;
  const rejInstDispatchAttr = new IndirectStorageBufferAttribute(new Uint32Array(3), 3);
  const rejInstDispatch = sU32Views(rejInstDispatchAttr as unknown as StorageBufferAttribute, 3).rw;
  const rejClustDispatchAttr = new IndirectStorageBufferAttribute(new Uint32Array(3), 3);
  const rejClustDispatch = sU32Views(
    rejClustDispatchAttr as unknown as StorageBufferAttribute,
    3,
  ).rw;

  const split2D = (
    args: ReturnType<typeof sU32Views>['rw'],
    n: NU,
  ): void => {
    const rows = n.add(uint(DISPATCH_ROW - 1)).div(uint(DISPATCH_ROW));
    elemUW(args, 0).assign(minU(n, uint(DISPATCH_ROW)));
    elemUW(args, 1).assign(maxU(rows, uint(1)));
    elemUW(args, 2).assign(uint(1));
  };

  // ---- kClear ------------------------------------------------------------------
  const kClear = Fn(() => {
    If(instanceIndex.lessThan(uint(8)), () => {
      atomicStore(counters.element(instanceIndex), uint(0));
    });
  })().compute(8, [8]);
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

  /** instance world sphere (heightfield = world-space already) */
  const instWorldSphere = (
    A: NV4,
    B: NV4,
    isHF: NB,
    localSphere: NV4,
    swayPad: NF,
  ): { center: NV3; radius: NF } => {
    const yawSc = instYaw(B);
    const centerW = vec3(0).toVar();
    const radiusW = float(0).toVar();
    If(isHF, () => {
      centerW.assign(localSphere.xyz);
      radiusW.assign(localSphere.w);
    }).Else(() => {
      centerW.assign(instTransformPoint(A, B, yawSc, localSphere.xyz as unknown as NV3));
      radiusW.assign(instSphereRadius(A, B, localSphere.w, swayPad));
    });
    return { center: centerW as unknown as NV3, radius: radiusW as unknown as NF };
  };

  /** LOD chain walk + envelope; pushes chunk items; shared by both phases */
  const lodSelectAndPush = (instId: NU, headId: NU, A: NV4): void => {
    const dist = cam.camPos.sub(A.xyz).length().toVar();
    const headBase = headId.mul(uint(MESH_WORDS)).toVar();
    const selStart = elemU(gpu.meshes, headBase).toVar();
    const selCount = elemU(gpu.meshes, headBase.add(uint(1))).toVar();
    const nextId = elemU(gpu.meshes, headBase.add(uint(4))).toVar();
    const nextDist = bcU2F(elemU(gpu.meshes, headBase.add(uint(5)))).toVar();
    Loop(4, () => {
      If(nextId.notEqual(uint(LOD_NONE)).and(dist.greaterThan(nextDist)), () => {
        const m = uint(nextId).mul(uint(MESH_WORDS)).toVar();
        selStart.assign(elemU(gpu.meshes, m));
        selCount.assign(elemU(gpu.meshes, m.add(uint(1))));
        nextId.assign(elemU(gpu.meshes, m.add(uint(4))));
        nextDist.assign(bcU2F(elemU(gpu.meshes, m.add(uint(5)))));
      });
    });
    // chain tail's lodDist = max draw envelope (0 = unlimited; D-N14)
    returnIf(
      nextId
        .equal(uint(LOD_NONE))
        .and(nextDist.greaterThan(0))
        .and(dist.greaterThan(nextDist)),
    );
    const nChunks = selCount.add(uint(CHUNK_CLUSTERS - 1)).div(uint(CHUNK_CLUSTERS)).toVar();
    const slotBase = (atomicAdd(counters.element(0), nChunks) as unknown as NU).toVar();
    loopU(uint(0), nChunks, (i) => {
      const slot = slotBase.add(i);
      If(slot.lessThan(uint(QCHUNK_CAP)), () => {
        const cBase = selStart.add(i.mul(uint(CHUNK_CLUSTERS)));
        const cnt = minU(uint(CHUNK_CLUSTERS), selStart.add(selCount).sub(cBase));
        qChunksV.rw
          .element(slot.add(uint(1)))
          .assign(uv2(instId, cBase.bitOr(cnt.sub(uint(1)).shiftLeft(uint(26)))));
      });
    });
  };

  // ---- kInstCull (phase 1) ------------------------------------------------------
  const kInstCull = Fn(() => {
    returnIf(instanceIndex.greaterThanEqual(uint(instanceCount)));
    const A = gpu.instances.element(instanceIndex.mul(uint(2))).toVar() as unknown as NV4;
    const B = gpu.instances
      .element(instanceIndex.mul(uint(2)).add(uint(1)))
      .toVar() as unknown as NV4;
    const headId = elemU(gpu.instanceMesh, instanceIndex).toVar();
    const head = readMesh(gpu.meshes, headId);
    const isHF = head.flags.bitAnd(uint(MESH_FLAG_HEIGHTFIELD)).notEqual(uint(0)).toVar();
    const s = instWorldSphere(A, B, isHF as unknown as NB, head.sphere, head.swayPad);
    returnIf(frustumVisible(s.center, s.radius).lessThan(0.5));
    if (sphereOccluded) {
      // occlusion-ONLY reject → record for phase 2, skip this phase
      If(sphereOccluded(s.center, s.radius, cam.prevVp, cam.prevCamPos), () => {
        const slot = atomicAdd(counters.element(2), uint(1)) as unknown as NU;
        If(slot.lessThan(uint(REJ_INST_CAP)), () => {
          elemUW(rejInstV.rw, slot).assign(instanceIndex);
        });
      }).Else(() => {
        lodSelectAndPush(instanceIndex, headId, A);
      });
    } else {
      lodSelectAndPush(instanceIndex, headId, A);
    }
  })().compute(instanceCount, [64]);
  (kInstCull as unknown as ComputeKernel).setName('nanInstCull');

  // ---- kChunkArgs ----------------------------------------------------------------
  const kChunkArgs = Fn(() => {
    const n = minU(aLoadU(counters.element(0)), uint(QCHUNK_CAP));
    qChunksV.rw.element(0).assign(uv2(n, 0));
    split2D(chunkDispatch, n);
  })().compute(1, [1]);
  (kChunkArgs as unknown as ComputeKernel).setName('nanChunkArgs');

  // ---- kClusterCull (phase selects occlusion inputs + reject recording) -----------
  const clusterCullKernel = (phase: 1 | 2): unknown => {
    const kn = Fn(() => {
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
        const B = gpu.instances
          .element(instId.mul(uint(2)).add(uint(1)))
          .toVar() as unknown as NV4;
        const yawSc = instYaw(B);
        const isHF = c.flags.bitAnd(uint(1)).notEqual(uint(0)).toVar();
        const swayPad = bcU2F(
          elemU(gpu.meshes, c.meshId.mul(uint(MESH_WORDS)).add(uint(11))),
        );
        const s = instWorldSphere(A, B, isHF as unknown as NB, c.sphere, swayPad);

        const visible = frustumVisible(s.center, s.radius).toVar();

        // N8-D1 continuous-LOD cut: a DAG cluster survives iff its OWN
        // simplification error projects ≤ τ px AND its PARENT's projects > τ px
        // (crack-free by the bit-exact sibling pairs). Gated on CLUSTER_FLAG_DAG
        // so terrain / not-yet-DAG'd pools keep their discrete chain. own/parent
        // spheres ride the instance transform like the geometric sphere (error
        // metric → no swayPad); roots carry the +∞ sentinel so parent>τ always.
        If(visible.greaterThan(0.5).and(c.flags.bitAnd(uint(CLUSTER_FLAG_DAG)).notEqual(uint(0))), () => {
          const rec = readDag(gpu.dag, ci);
          const ownC = instTransformPoint(A, B, yawSc, rec.ownSphere.xyz as unknown as NV3);
          const ownR = instSphereRadius(A, B, rec.ownSphere.w as unknown as NF, float(0));
          const parC = instTransformPoint(A, B, yawSc, rec.parentSphere.xyz as unknown as NV3);
          const parR = instSphereRadius(A, B, rec.parentSphere.w as unknown as NF, float(0));
          const dvo = cam.camPos.sub(ownC) as unknown as NV3;
          const dvp = cam.camPos.sub(parC) as unknown as NV3;
          const denO = dot(dvo, dvo).sub(ownR.mul(ownR)).max(float(1e-6)).sqrt() as unknown as NF;
          const denP = dot(dvp, dvp).sub(parR.mul(parR)).max(float(1e-6)).sqrt() as unknown as NF;
          const pOwn = projK.mul(rec.ownError).div(denO);
          const pPar = projK.mul(rec.parentError).div(denP);
          If(pOwn.greaterThan(tau).or(pPar.lessThanEqual(tau)), () => {
            visible.assign(0);
          });
        });

        // cone backface (explicit meshes only; conservative slack) — skipped
        // for shadow casters (D-N26: camera-relative cone wrong for light views)
        if (coneCull) {
          If(visible.greaterThan(0.5).and(c.coneCos.greaterThan(-0.99)).and(isHF.not()), () => {
            const sinTest = float(1)
              .sub(c.coneCos.mul(c.coneCos))
              .max(0)
              .sqrt()
              .add(CONE_SLACK)
              .toVar();
            If(sinTest.lessThan(1), () => {
              const axisW = instRotateDir(yawSc, c.coneAxis);
              const toC = s.center.sub(cam.camPos).toVar();
              const d = toC.length();
              If(dot(toC as unknown as NV3, axisW).greaterThan(d.mul(sinTest).add(s.radius)), () => {
                visible.assign(0);
              });
            });
          });
        }

        if (sphereOccluded) {
          const vp = phase === 1 ? cam.prevVp : cam.vp;
          const cp = phase === 1 ? cam.prevCamPos : cam.camPos;
          If(visible.greaterThan(0.5), () => {
            If(sphereOccluded(s.center, s.radius, vp, cp), () => {
              visible.assign(0);
              if (phase === 1) {
                // occlusion-ONLY reject → phase 2 re-tests vs the fresh HZB
                const slot = atomicAdd(counters.element(3), uint(1)) as unknown as NU;
                If(slot.lessThan(uint(REJ_CLUST_CAP)), () => {
                  rejClustV.rw.element(slot).assign(uv2(instId, ci));
                });
              }
            });
          });
        }

        If(visible.greaterThan(0.5), () => {
          const slot = atomicAdd(counters.element(1), uint(1)) as unknown as NU;
          If(slot.lessThan(uint(QRASTER_CAP)), () => {
            qRasterV.rw.element(slot.add(uint(1))).assign(uv2(instId, ci));
          });
          // N8-D1: count emitted DAG clusters separately (slot 5) — the rock-cut
          // signal for the HUD + the continuous-zoom gate; terrain/bark are
          // CLUSTER_FLAG_DAG=0 so they don't add, isolating the screen-error cut.
          If(c.flags.bitAnd(uint(CLUSTER_FLAG_DAG)).notEqual(uint(0)), () => {
            atomicAdd(counters.element(5), uint(1));
          });
        });
      });
    })().compute(QCHUNK_CAP * CHUNK_CLUSTERS, [CHUNK_CLUSTERS]);
    return kn;
  };
  const kClusterCull = clusterCullKernel(1);
  (kClusterCull as ComputeKernel).setName('nanClusterCull');
  const kClusterCull2 = sphereOccluded ? clusterCullKernel(2) : null;
  if (kClusterCull2) (kClusterCull2 as ComputeKernel).setName('nanClusterCull2');

  // ---- kRasterArgs (phase 1) -------------------------------------------------------
  const kRasterArgs = Fn(() => {
    const n = minU(aLoadU(counters.element(1)), uint(QRASTER_CAP));
    qRasterV.rw.element(0).assign(uv2(n, 0));
    split2D(rasterDispatch, n);
  })().compute(1, [1]);
  (kRasterArgs as unknown as ComputeKernel).setName('nanRasterArgs');

  // ---- phase 2 ----------------------------------------------------------------------
  // kPhase2Args: snapshot phase-1 raster count into qRaster[0].y (the phase-2
  // base), reset the chunk counter for re-expansion, args over reject lists
  const kPhase2Args = Fn(() => {
    const n1 = minU(aLoadU(counters.element(1)), uint(QRASTER_CAP));
    qRasterV.rw.element(0).assign(uv2(n1, n1));
    atomicStore(counters.element(4), aLoadU(counters.element(0)));
    atomicStore(counters.element(0), uint(0));
    // reject consumers run 64 threads/workgroup → ceil(n/64) workgroups
    const nI = minU(aLoadU(counters.element(2)), uint(REJ_INST_CAP));
    split2D(rejInstDispatch, nI.add(uint(63)).div(uint(64)));
    const nC = minU(aLoadU(counters.element(3)), uint(REJ_CLUST_CAP));
    split2D(rejClustDispatch, nC.add(uint(63)).div(uint(64)));
  })().compute(1, [1]);
  (kPhase2Args as unknown as ComputeKernel).setName('nanPhase2Args');

  // kInstCull2: one thread per rejected instance, fresh HZB + CURRENT pair
  const kInstCull2 = sphereOccluded
    ? Fn(() => {
        const i = wgLinear(DISPATCH_ROW).mul(uint(64)).add(localX()).toVar();
        returnIf(i.greaterThanEqual(minU(aLoadU(counters.element(2)), uint(REJ_INST_CAP))));
        const instId = elemU(rejInstV.ro, i).toVar();
        const A = gpu.instances.element(instId.mul(uint(2))).toVar() as unknown as NV4;
        const B = gpu.instances
          .element(instId.mul(uint(2)).add(uint(1)))
          .toVar() as unknown as NV4;
        const headId = elemU(gpu.instanceMesh, instId).toVar();
        const head = readMesh(gpu.meshes, headId);
        const isHF = head.flags.bitAnd(uint(MESH_FLAG_HEIGHTFIELD)).notEqual(uint(0)).toVar();
        const s = instWorldSphere(A, B, isHF as unknown as NB, head.sphere, head.swayPad);
        returnIf(sphereOccluded(s.center, s.radius, cam.vp, cam.camPos));
        lodSelectAndPush(instId, headId, A);
      })().compute(REJ_INST_CAP, [64])
    : null;
  if (kInstCull2) (kInstCull2 as unknown as ComputeKernel).setName('nanInstCull2');

  // kClusterCull2b: one thread per rejected cluster, occlusion re-test only
  // (frustum + cone already passed in phase 1)
  const kClusterCull2b = sphereOccluded
    ? Fn(() => {
        const i = wgLinear(DISPATCH_ROW).mul(uint(64)).add(localX()).toVar();
        returnIf(i.greaterThanEqual(minU(aLoadU(counters.element(3)), uint(REJ_CLUST_CAP))));
        const pair = rejClustV.ro.element(i);
        const instId = pair.x.toVar();
        const ci = pair.y.toVar();
        const c = readCluster(gpu.clusters, ci);
        const A = gpu.instances.element(instId.mul(uint(2))).toVar() as unknown as NV4;
        const B = gpu.instances
          .element(instId.mul(uint(2)).add(uint(1)))
          .toVar() as unknown as NV4;
        const isHF = c.flags.bitAnd(uint(1)).notEqual(uint(0)).toVar();
        const swayPad = bcU2F(
          elemU(gpu.meshes, c.meshId.mul(uint(MESH_WORDS)).add(uint(11))),
        );
        const s = instWorldSphere(A, B, isHF as unknown as NB, c.sphere, swayPad);
        returnIf(sphereOccluded(s.center, s.radius, cam.vp, cam.camPos));
        const slot = atomicAdd(counters.element(1), uint(1)) as unknown as NU;
        If(slot.lessThan(uint(QRASTER_CAP)), () => {
          qRasterV.rw.element(slot.add(uint(1))).assign(uv2(instId, ci));
        });
      })().compute(REJ_CLUST_CAP, [64])
    : null;
  if (kClusterCull2b) (kClusterCull2b as unknown as ComputeKernel).setName('nanClusterCull2b');

  // kRasterArgs2: total count + appended-range and full-range dispatch args.
  // The base read goes through the SAME rw view that writes slot 0 — mixing
  // the ro view into this dispatch is a same-scope usage violation (N0 law).
  const kRasterArgs2 = Fn(() => {
    const nT = minU(aLoadU(counters.element(1)), uint(QRASTER_CAP)).toVar();
    const base = qRasterV.rw.element(0).y.toVar();
    qRasterV.rw.element(0).assign(uv2(nT, base));
    split2D(rasterDispatch2, nT.sub(base));
    split2D(rasterDispatchFull, nT);
  })().compute(1, [1]);
  (kRasterArgs2 as unknown as ComputeKernel).setName('nanRasterArgs2');

  const runPhase1 = (renderer: Renderer): void => {
    dispatch(renderer, kClear);
    dispatch(renderer, kInstCull);
    dispatch(renderer, kChunkArgs);
    dispatchIndirect(renderer, kClusterCull, chunkDispatchAttr);
    dispatch(renderer, kRasterArgs);
  };

  const syncFullArgs = (renderer: Renderer): void => {
    dispatch(renderer, kRasterArgs2);
  };

  const runPhase2 = (renderer: Renderer): void => {
    if (!sphereOccluded || !kInstCull2 || !kClusterCull2 || !kClusterCull2b) {
      // no occlusion → nothing was rejected; keep full args in sync for the
      // payload pass (same contents as phase-1 args)
      syncFullArgs(renderer);
      return;
    }
    dispatch(renderer, kPhase2Args);
    dispatchIndirect(renderer, kInstCull2, rejInstDispatchAttr);
    dispatch(renderer, kChunkArgs);
    dispatchIndirect(renderer, kClusterCull2, chunkDispatchAttr);
    dispatchIndirect(renderer, kClusterCull2b, rejClustDispatchAttr);
    dispatch(renderer, kRasterArgs2);
  };

  const readCounts = async (renderer: Renderer): Promise<NaniteCullCounts> => {
    const [buf, head] = await Promise.all([
      readBuffer(renderer, countersAttr, 0, 32),
      readBuffer(renderer, qRasterAttr, 0, 8),
    ]);
    const u = new Uint32Array(buf);
    const q = new Uint32Array(head);
    // with occlusion, [0] holds the phase-2 re-expansion — phase 1 is in [4]
    const chunks = sphereOccluded ? (u[4] ?? 0) : (u[0] ?? 0);
    const visClusters = u[1] ?? 0;
    const rejInst = u[2] ?? 0;
    const rejClust = u[3] ?? 0;
    const dagClusters = u[5] ?? 0;
    const p2Appends = Math.max(0, (q[0] ?? 0) - (q[1] ?? 0));
    let overflow: string | null = null;
    const over = (label: string, n: number, cap: number): void => {
      if (n > cap) overflow = `${overflow ? `${overflow}; ` : ''}${label} ${n} > ${cap}`;
    };
    over('qChunks', chunks, QCHUNK_CAP);
    over('qRaster', visClusters, QRASTER_CAP);
    over('rejInst', rejInst, REJ_INST_CAP);
    over('rejClust', rejClust, REJ_CLUST_CAP);
    return { chunks, visClusters, rejInst, rejClust, dagClusters, p2Appends, overflow };
  };

  return {
    qRasterRO: qRasterV.ro,
    qRasterAttr,
    rasterDispatchAttr,
    rasterDispatch2Attr,
    rasterDispatchFullAttr,
    runPhase1,
    runPhase2,
    syncFullArgs,
    readCounts,
  };
}
