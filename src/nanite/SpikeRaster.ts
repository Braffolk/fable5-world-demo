/**
 * N0 spike GPU pipeline — cull → work queue → indirect SW raster → resolve,
 * with the HW big-triangle path writing the SAME vis-buffer from the fragment
 * stage (the unified-convention departure from the reference example, F10b —
 * this is also the fragment-storage-atomics verification the N0 gate asks for).
 *
 * Depth packing per docs/NANITE.md D-N5:
 *  - Option C (DEFAULT, ?packing=c): pass 1 atomicMin(f32-depth-bits) into
 *    visDepth; pass 2 re-walks the same work items and plain-stores the full
 *    32-bit payload where its depth bits equal the stored minimum.
 *  - Option A (?packing=a, COST COMPARISON ONLY): the example's dual-u32
 *    atomicMax (depth17|payloadLo15, depth15|payloadHi17) — known-inconsistent
 *    at depth-prefix ties (F2); kept purely to measure atomic-traffic cost.
 *
 * Payload = (workItemIdx << 7) | localTri — the work queue doubles as the
 * visible-cluster list, so the resolve recovers (instance, cluster) in one
 * indirection. Near-plane-crossing and oversized triangles route to the HW
 * queue (hardware clips); the SW path never drops geometry (F10c).
 *
 * Scanline core is the example's float-edge loop verbatim (top-left bias) —
 * fixed-point integer edges land at N3 per the plan; the spike measures cost.
 *
 * Spike limits (asserted): work items ≤ 49152 (single-dimension indirect
 * dispatch stays < 65535 workgroups), HW queue ≤ 262144 tris.
 */

import { Frustum, Matrix4, Mesh, Scene, Vector3, Vector4 } from 'three';
import { BufferGeometry, Float32BufferAttribute, RenderTarget, Sphere } from 'three';
import type { PerspectiveCamera } from 'three';
import {
  IndirectStorageBufferAttribute,
  NodeMaterial,
  StorageBufferAttribute,
  type Renderer,
} from 'three/webgpu';
import { markFragmentWritable } from '../render/ThreePatches';
import {
  Discard,
  Fn,
  If,
  Loop,
  atomicAdd,
  atomicLoad,
  atomicMax,
  atomicMin,
  atomicStore,
  bitcast,
  cross,
  dot,
  float,
  instanceIndex,
  int,
  max,
  min,
  normalize,
  positionGeometry,
  screenCoordinate,
  sqrt,
  storage,
  uint,
  uniform,
  uniformArray,
  uvec2,
  varyingProperty,
  vec2,
  vec3,
  vec4,
  vertexIndex,
} from 'three/tsl';
import type { NB, NF, NU, NV2, NV3, NV4 } from '../gpu/TSLTypes';
import type { SpikeContent } from './SpikeContent';
import { TERRAIN_CELL, TERRAIN_QUADS, TERRAIN_WIN } from './SpikeContent';

// single-dimension indirect dispatch ceiling; the spike content is sized to
// stay under it with frustum margin (proper 2D-split / hierarchical queues
// land at N2 — overflow here drops items SILENTLY, so the HUD shows
// spike.work vs the cap and the readback flags it)
const WORK_CAP = 65535;
const HW_CAP = 262144;
const MAX_RASTER_SIZE = 16;
const NEAR_EPS = 1e-4;

// Option A packing (example layout, comparison only)
const A_LO_BITS = 15; // bufA: depth17 | payloadLo15
const A_HI_BITS = 17; // bufB: depth15 | payloadHi17
const A_DEPTH_LO_MAX = 2 ** (32 - A_LO_BITS) - 1;
const A_DEPTH_HI_MAX = 2 ** (32 - A_HI_BITS) - 1;

interface ComputeKernel {
  setName(name: string): unknown;
}

export interface SpikeRasterHandles {
  /** fullscreen resolve mesh — already configured; add to the engine scene */
  resolveMesh: Mesh;
  /** run the whole per-frame chain (computes + HW passes); call from onUpdate */
  update(renderer: Renderer, camera: PerspectiveCamera): void;
  /** async GPU counter readback: visible work items + HW-queued triangles */
  readCounts(renderer: Renderer): Promise<{ work: number; hw: number }>;
}

export function buildSpikeRaster(
  content: SpikeContent,
  width: number,
  height: number,
  packing: 'a' | 'c',
  clusterTint: boolean,
): SpikeRasterHandles {
  const pixelCount = width * height;
  const isA = packing === 'a';

  // ---- uniforms --------------------------------------------------------------
  const uProj = uniform(new Matrix4());
  const uCamPos = uniform(new Vector3());
  const uW = uniform(width);
  const uH = uniform(height);
  const frustumPlanes = uniformArray(
    [new Vector4(), new Vector4(), new Vector4(), new Vector4(), new Vector4(), new Vector4()],
    'vec4',
  );

  // ---- storage buffers --------------------------------------------------------
  const c = content;
  const posBuf = storage(new StorageBufferAttribute(c.positions, 4), 'vec4', c.positions.length / 4).toReadOnly();
  const idxBuf = storage(new StorageBufferAttribute(c.indices, 1), 'uint', c.indices.length).toReadOnly();
  const sphereBuf = storage(new StorageBufferAttribute(c.clusterSphere, 4), 'vec4', c.clusterCount).toReadOnly();
  const metaBuf = storage(new StorageBufferAttribute(c.clusterMeta, 4), 'uvec4' as unknown as 'vec4', c.clusterCount).toReadOnly();
  const meshTableBuf = storage(new StorageBufferAttribute(c.meshTable, 2), 'uvec2' as unknown as 'vec4', c.meshTable.length / 2).toReadOnly();
  const instABuf = storage(new StorageBufferAttribute(c.instA, 4), 'vec4', c.instanceCount).toReadOnly();
  const instBBuf = storage(new StorageBufferAttribute(c.instB, 4), 'vec4', c.instanceCount).toReadOnly();
  const heightsBuf = storage(new StorageBufferAttribute(c.heights, 1), 'float', c.heights.length).toReadOnly();

  const workQueueAttr = new StorageBufferAttribute(new Uint32Array(WORK_CAP * 2), 2);
  const workQueue = storage(workQueueAttr, 'uvec2' as unknown as 'vec4', WORK_CAP);
  const workQueueRO = storage(workQueueAttr, 'uvec2' as unknown as 'vec4', WORK_CAP).toReadOnly();

  const countersAttr = new StorageBufferAttribute(new Uint32Array(4), 1);
  const counters = storage(countersAttr, 'uint', 4).toAtomic();
  const countersRO = storage(countersAttr, 'uint', 4).toReadOnly();

  const dispatchAttr = new IndirectStorageBufferAttribute(new Uint32Array(3), 3);
  const dispatchBuf = storage(dispatchAttr, 'uint', 3);

  // hwQueue: [0] = atomic count, then (payload, instId) pairs
  const hwQueueAttr = new StorageBufferAttribute(new Uint32Array(1 + HW_CAP * 2), 1);
  const hwQueue = storage(hwQueueAttr, 'uint', 1 + HW_CAP * 2).toAtomic();
  const hwQueueRO = storage(hwQueueAttr, 'uint', 1 + HW_CAP * 2).toReadOnly();

  const hwDrawAttr = new IndirectStorageBufferAttribute(new Uint32Array(4), 4);
  const hwDrawBuf = storage(hwDrawAttr, 'uint', 4);

  // vis buffers: depth (atomic u32) + payload (plain u32; atomic in A-mode);
  // the HW big-tri path writes both from FRAGMENT stage — needs the opt-in
  // three patch (ThreePatches.installFragmentStorageWrites)
  const visDepthAttr = new StorageBufferAttribute(new Uint32Array(pixelCount), 1);
  markFragmentWritable(visDepthAttr);
  const visDepthAtomic = storage(visDepthAttr, 'uint', pixelCount).toAtomic();
  const visDepthRO = storage(visDepthAttr, 'uint', pixelCount).toReadOnly();
  const visPayloadAttr = new StorageBufferAttribute(new Uint32Array(pixelCount), 1);
  markFragmentWritable(visPayloadAttr);
  const visPayload = storage(visPayloadAttr, 'uint', pixelCount);
  const visPayloadAtomic = storage(visPayloadAttr, 'uint', pixelCount).toAtomic();
  const visPayloadRO = storage(visPayloadAttr, 'uint', pixelCount).toReadOnly();

  // ---- shared TSL builders ------------------------------------------------------
  const HF_N = TERRAIN_QUADS + 1;
  const tileOX = c.tileOrigin.x;
  const tileOZ = c.tileOrigin.z;

  /** world-space vertex v of (instId, clusterId, localTri); pure expression per call site */
  const fetchWorldVert = (instId: NU, meta: NV4, localTri: NU, v: 0 | 1 | 2): NV3 => {
    const kind = (meta as unknown as { x: NU }).x;
    const A = instABuf.element(instId) as unknown as NV4;
    const B = instBBuf.element(instId) as unknown as NV4;
    const out = vec3(0).toVar();
    If((kind.equal(uint(1)) as unknown as NB), () => {
      const win = (meta as unknown as { w: NU }).w;
      const gx = win.bitAnd(uint(0xffff));
      const gz = win.shiftRight(uint(16));
      const quad = localTri.shiftRight(uint(1));
      const odd = localTri.bitAnd(uint(1)).equal(uint(1)) as unknown as NB;
      const qx = quad.mod(uint(TERRAIN_WIN));
      const qz = quad.div(uint(TERRAIN_WIN));
      // up-facing CCW — even tri (0,0)(0,1)(1,1), odd tri (0,0)(1,1)(1,0)
      let dx: NU;
      let dz: NU;
      if (v === 0) {
        dx = uint(0) as unknown as NU;
        dz = uint(0) as unknown as NU;
      } else if (v === 1) {
        dx = (odd as unknown as { select: (a: unknown, b: unknown) => NU }).select(uint(1), uint(0));
        dz = uint(1) as unknown as NU;
      } else {
        dx = uint(1) as unknown as NU;
        dz = (odd as unknown as { select: (a: unknown, b: unknown) => NU }).select(uint(0), uint(1));
      }
      const sx = gx.add(qx).add(dx);
      const sz = gz.add(qz).add(dz);
      const h = heightsBuf.element(sz.mul(uint(HF_N)).add(sx)) as unknown as NF;
      out.assign(
        vec3(
          (sx as unknown as NF).toFloat().mul(TERRAIN_CELL).add(tileOX),
          h,
          (sz as unknown as NF).toFloat().mul(TERRAIN_CELL).add(tileOZ),
        ),
      );
    }).Else(() => {
      const triStart = (meta as unknown as { y: NU }).y;
      const vi = idxBuf.element(triStart.add(localTri).mul(uint(3)).add(uint(v))) as unknown as NU;
      const p = (posBuf.element(vi) as unknown as NV4).xyz;
      const s = A.w;
      const yaw = B.x;
      const cy = yaw.cos();
      const sy = yaw.sin();
      const ls = p.mul(s);
      const rx = ls.x.mul(cy).add(ls.z.mul(sy));
      const rz = ls.z.mul(cy).sub(ls.x.mul(sy));
      const px = rx.add(B.y.mul(ls.y));
      const pz = rz.add(B.z.mul(ls.y));
      out.assign(vec3(px, ls.y, pz).add(A.xyz));
    });
    return out as unknown as NV3;
  };

  const edgeFn = (a: NV2, b: NV2, p: NV2): NF =>
    p.y.sub(a.y).mul(b.x.sub(a.x)).sub(p.x.sub(a.x).mul(b.y.sub(a.y))) as unknown as NF;

  const hashColor = (id: NU): NV3 => {
    const a = id.add(uint(0x9e3779b9)).mul(uint(747796405)).add(uint(289559509));
    const b = a.shiftRight(uint(16)).bitXor(a).mul(uint(277803737));
    const h = b.shiftRight(uint(16)).bitXor(b);
    return vec3(
      (h.bitAnd(uint(255)) as unknown as NF).toFloat().div(255),
      (h.shiftRight(uint(8)).bitAnd(uint(255)) as unknown as NF).toFloat().div(255),
      (h.shiftRight(uint(16)).bitAnd(uint(255)) as unknown as NF).toFloat().div(255),
    ).mul(0.8).add(0.2) as unknown as NV3;
  };

  // ---- kClear -------------------------------------------------------------------
  const kClear = Fn(() => {
    If((instanceIndex.lessThan(uint(pixelCount)) as unknown as NB), () => {
      atomicStore(visDepthAtomic.element(instanceIndex), isA ? uint(0) : uint(0xffffffff));
      atomicStore(visPayloadAtomic.element(instanceIndex), isA ? uint(0) : uint(0xffffffff));
    });
    If((instanceIndex.equal(uint(0)) as unknown as NB), () => {
      atomicStore(counters.element(0), uint(0));
      atomicStore(hwQueue.element(0), uint(0));
    });
  })().compute(pixelCount, [256]);
  (kClear as unknown as ComputeKernel).setName('spikeClear');

  // ---- kCull: per instance, loop its clusters, frustum test, append ---------------
  const kCull = Fn(() => {
    If((instanceIndex.lessThan(uint(c.instanceCount)) as unknown as NB), () => {
      const A = instABuf.element(instanceIndex) as unknown as NV4;
      const B = instBBuf.element(instanceIndex) as unknown as NV4;
      const meshId = uint(B.w) as unknown as NU;
      const range = meshTableBuf.element(meshId) as unknown as { x: NU; y: NU };
      const start = range.x;
      const count = range.y;
      Loop({ start: uint(0), end: count, type: 'uint', condition: '<' } as never, (lp: unknown) => {
        const clusterId = start.add((lp as { i: NU }).i);
        const sph = sphereBuf.element(clusterId) as unknown as NV4;
        const meta = metaBuf.element(clusterId) as unknown as NV4;
        const kind = (meta as unknown as { x: NU }).x;
        const centerW = vec3(0).toVar();
        const radiusW = float(0).toVar();
        If((kind.equal(uint(1)) as unknown as NB), () => {
          centerW.assign(sph.xyz);
          radiusW.assign(sph.w);
        }).Else(() => {
          const s = A.w;
          const yaw = B.x;
          const cy = yaw.cos();
          const sy = yaw.sin();
          const ls = sph.xyz.mul(s);
          centerW.assign(
            vec3(ls.x.mul(cy).add(ls.z.mul(sy)), ls.y, ls.z.mul(cy).sub(ls.x.mul(sy))).add(A.xyz),
          );
          radiusW.assign(sph.w.mul(s).mul(1.05));
        });
        const visible = float(1).toVar();
        Loop(6, ({ i: pi }) => {
          const plane = frustumPlanes.element(pi) as unknown as NV4;
          const d = dot(plane.xyz, centerW as unknown as NV3).add(plane.w) as unknown as NF;
          If((d.lessThan(radiusW.negate()) as unknown as NB), () => {
            visible.assign(0);
          });
        });
        If((visible.greaterThan(0.5) as unknown as NB), () => {
          const slot = atomicAdd(counters.element(0), uint(1)) as unknown as NU;
          If((slot.lessThan(uint(WORK_CAP)) as unknown as NB), () => {
            workQueue
              .element(slot)
              .assign(uvec2(instanceIndex as unknown as number, clusterId as unknown as number));
          });
        });
      });
    });
  })().compute(c.instanceCount, [64]);
  (kCull as unknown as ComputeKernel).setName('spikeCull');

  // ---- kArgs ---------------------------------------------------------------------
  const kArgs = Fn(() => {
    const n = countersRO.element(0) as unknown as NU;
    dispatchBuf.element(0).assign(min(n as unknown as NF, uint(WORK_CAP) as unknown as NF));
    dispatchBuf.element(1).assign(uint(1));
    dispatchBuf.element(2).assign(uint(1));
  })().compute(1, [1]);
  (kArgs as unknown as ComputeKernel).setName('spikeArgs');

  // ---- raster kernels --------------------------------------------------------------
  // shared front half: returns everything the inner loop needs, or marks skip
  const rasterKernel = (
    mode: 'depth' | 'payload' | 'optionA',
  ): ReturnType<ReturnType<typeof Fn>> => {
    const kn = Fn(() => {
      const itemIdx = instanceIndex.div(128) as unknown as NU;
      const localTri = instanceIndex.mod(128) as unknown as NU;
      const item = workQueueRO.element(itemIdx) as unknown as { x: NU; y: NU };
      const instId = item.x;
      const clusterId = item.y;
      const meta = metaBuf.element(clusterId) as unknown as NV4;
      const triCount = (meta as unknown as { z: NU }).z;

      If((localTri.lessThan(triCount) as unknown as NB), () => {
        const w0 = fetchWorldVert(instId, meta, localTri, 0);
        const w1 = fetchWorldVert(instId, meta, localTri, 1);
        const w2 = fetchWorldVert(instId, meta, localTri, 2);

        const p0 = (uProj as unknown as { mul: (v: unknown) => NV4 }).mul(vec4(w0, 1)).toVar();
        const p1 = (uProj as unknown as { mul: (v: unknown) => NV4 }).mul(vec4(w1, 1)).toVar();
        const p2 = (uProj as unknown as { mul: (v: unknown) => NV4 }).mul(vec4(w2, 1)).toVar();

        const payload = itemIdx.shiftLeft(uint(7)).bitOr(localTri).toVar();

        const nearOK = p0.w
          .greaterThan(NEAR_EPS)
          .and(p1.w.greaterThan(NEAR_EPS))
          .and(p2.w.greaterThan(NEAR_EPS)) as unknown as NB;

        If(nearOK.not() as unknown as NB, () => {
          // near-plane crossing → HW path clips it (never drop, F10c)
          if (mode !== 'payload') {
            const slot = atomicAdd(hwQueue.element(0), uint(1)) as unknown as NU;
            If((slot.lessThan(uint(HW_CAP)) as unknown as NB), () => {
              const base = slot.mul(uint(2)).add(uint(1));
              atomicStore(hwQueue.element(base), payload);
              atomicStore(hwQueue.element(base.add(uint(1))), instId);
            });
          }
        }).Else(() => {
          const ndc0 = p0.xyz.div(p0.w).toVar();
          const ndc1 = p1.xyz.div(p1.w).toVar();
          const ndc2 = p2.xyz.div(p2.w).toVar();

          const areaNdc = edgeFn(ndc0.xy as unknown as NV2, ndc1.xy as unknown as NV2, ndc2.xy as unknown as NV2);
          If((areaNdc.greaterThan(0) as unknown as NB), () => {
            const W = float(uW as unknown as NF);
            const H = float(uH as unknown as NF);
            const s0 = ndc0.xy.add(1).mul(0.5).mul(vec2(W, H)).toVar();
            const s1 = ndc1.xy.add(1).mul(0.5).mul(vec2(W, H)).toVar();
            const s2 = ndc2.xy.add(1).mul(0.5).mul(vec2(W, H)).toVar();

            const minX = max(float(0), min(s0.x, min(s1.x, s2.x)));
            const maxX = min(W.sub(1), max(s0.x, max(s1.x, s2.x)));
            const minY = max(float(0), min(s0.y, min(s1.y, s2.y)));
            const maxY = min(H.sub(1), max(s0.y, max(s1.y, s2.y)));

            const startX = int(minX.floor()).toVar();
            const endX = int(maxX.floor()).toVar();
            const startY = int(minY.floor()).toVar();
            const endY = int(maxY.floor()).toVar();

            const bbW = endX.sub(startX);
            const bbH = endY.sub(startY);
            const validBB = startX.lessThanEqual(endX).and(startY.lessThanEqual(endY)) as unknown as NB;
            const smallEnough = bbW
              .lessThanEqual(int(MAX_RASTER_SIZE))
              .and(bbH.lessThanEqual(int(MAX_RASTER_SIZE))) as unknown as NB;

            If((validBB as unknown as { and: (o: unknown) => NB }).and(smallEnough), () => {
              const area = edgeFn(s0 as unknown as NV2, s1 as unknown as NV2, s2 as unknown as NV2);

              const stepXw0 = s1.y.sub(s2.y);
              const stepYw0 = s2.x.sub(s1.x);
              const stepXw1 = s2.y.sub(s0.y);
              const stepYw1 = s0.x.sub(s2.x);
              const stepXw2 = s0.y.sub(s1.y);
              const stepYw2 = s1.x.sub(s0.x);

              // top-left rule (float-edge bias, example-verbatim; fixed point = N3)
              const tl0 = stepXw0.lessThan(0).or(stepXw0.equal(0).and(stepYw0.greaterThan(0)));
              const tl1 = stepXw1.lessThan(0).or(stepXw1.equal(0).and(stepYw1.greaterThan(0)));
              const tl2 = stepXw2.lessThan(0).or(stepXw2.equal(0).and(stepYw2.greaterThan(0)));
              const bias0 = (tl0 as unknown as { select: (a: unknown, b: unknown) => NF }).select(float(0), float(-1e-5));
              const bias1 = (tl1 as unknown as { select: (a: unknown, b: unknown) => NF }).select(float(0), float(-1e-5));
              const bias2 = (tl2 as unknown as { select: (a: unknown, b: unknown) => NF }).select(float(0), float(-1e-5));

              const pStart = vec2(float(startX).add(0.5), float(startY).add(0.5));
              const rw0 = edgeFn(s1 as unknown as NV2, s2 as unknown as NV2, pStart as unknown as NV2).add(bias0).toVar();
              const rw1 = edgeFn(s2 as unknown as NV2, s0 as unknown as NV2, pStart as unknown as NV2).add(bias1).toVar();
              const rw2 = edgeFn(s0 as unknown as NV2, s1 as unknown as NV2, pStart as unknown as NV2).add(bias2).toVar();

              const b0s = rw0.div(area);
              const b1s = rw1.div(area);
              const b2s = rw2.div(area);
              const rowZ = b0s.mul(ndc0.z).add(b1s.mul(ndc1.z)).add(b2s.mul(ndc2.z)).toVar();
              const stepXz = stepXw0.div(area).mul(ndc0.z).add(stepXw1.div(area).mul(ndc1.z)).add(stepXw2.div(area).mul(ndc2.z));
              const stepYz = stepYw0.div(area).mul(ndc0.z).add(stepYw1.div(area).mul(ndc1.z)).add(stepYw2.div(area).mul(ndc2.z));

              Loop(
                { name: 'sy', type: 'int', start: startY, end: endY, condition: '<=' } as never,
                (lpy: unknown) => {
                const y = (lpy as { sy: NU }).sy;
                const cw0 = rw0.toVar();
                const cw1 = rw1.toVar();
                const cw2 = rw2.toVar();
                const cz = rowZ.toVar();
                Loop(
                  { name: 'sx', type: 'int', start: startX, end: endX, condition: '<=' } as never,
                  (lpx: unknown) => {
                  const x = (lpx as { sx: NU }).sx;
                  If(
                    cw0
                      .greaterThanEqual(0)
                      .and(cw1.greaterThanEqual(0))
                      .and(cw2.greaterThanEqual(0))
                      .and(cz.greaterThanEqual(0))
                      .and(cz.lessThanEqual(1)) as unknown as NB,
                    () => {
                      const px = uint(y).mul(uint(uW as unknown as NF)).add(uint(x));
                      if (mode === 'depth') {
                        const bits = bitcast(cz as unknown as NF, 'uint') as unknown as NU;
                        const cur = atomicLoad(visDepthAtomic.element(px)) as unknown as NU;
                        If((bits.lessThan(cur) as unknown as NB), () => {
                          atomicMin(visDepthAtomic.element(px), bits);
                        });
                      } else if (mode === 'payload') {
                        const bits = bitcast(cz as unknown as NF, 'uint') as unknown as NU;
                        const cur = visDepthRO.element(px) as unknown as NU;
                        If((bits.equal(cur) as unknown as NB), () => {
                          visPayload.element(px).assign(payload);
                        });
                      } else {
                        // Option A: dual atomicMax, fourth-root encode
                        const zEnc = sqrt(sqrt(float(1).sub(cz)));
                        const dLo = uint(zEnc.mul(A_DEPTH_LO_MAX));
                        const dHi = uint(zEnc.mul(A_DEPTH_HI_MAX));
                        const packedA = dLo.shiftLeft(uint(A_LO_BITS)).bitOr(payload.bitAnd(uint((1 << A_LO_BITS) - 1)));
                        const packedB = dHi.shiftLeft(uint(A_HI_BITS)).bitOr(payload.shiftRight(uint(A_LO_BITS)));
                        const cur = (atomicLoad(visDepthAtomic.element(px)) as unknown as NU).shiftRight(uint(A_LO_BITS));
                        If((dLo.greaterThanEqual(cur) as unknown as NB), () => {
                          atomicMax(visDepthAtomic.element(px), packedA);
                          atomicMax(visPayloadAtomic.element(px), packedB);
                        });
                      }
                    },
                  );
                  cw0.addAssign(stepXw0);
                  cw1.addAssign(stepXw1);
                  cw2.addAssign(stepXw2);
                  cz.addAssign(stepXz);
                });
                rw0.addAssign(stepYw0);
                rw1.addAssign(stepYw1);
                rw2.addAssign(stepYw2);
                rowZ.addAssign(stepYz);
              });
            }).Else(() => {
              if (mode !== 'payload') {
                If(validBB, () => {
                  // big triangle → HW queue
                  const slot = atomicAdd(hwQueue.element(0), uint(1)) as unknown as NU;
                  If((slot.lessThan(uint(HW_CAP)) as unknown as NB), () => {
                    const base = slot.mul(uint(2)).add(uint(1));
                    atomicStore(hwQueue.element(base), payload);
                    atomicStore(hwQueue.element(base.add(uint(1))), instId);
                  });
                });
              }
            });
          });
        });
      });
    })().compute(WORK_CAP * 128, [128]);
    return kn;
  };

  const kRaster1 = rasterKernel(isA ? 'optionA' : 'depth');
  (kRaster1 as unknown as ComputeKernel).setName(isA ? 'spikeRasterA' : 'spikeRasterDepth');
  const kRaster2 = isA ? null : rasterKernel('payload');
  if (kRaster2) (kRaster2 as unknown as ComputeKernel).setName('spikeRasterPayload');

  // ---- kHwArgs ----------------------------------------------------------------------
  const kHwArgs = Fn(() => {
    const n = min(
      atomicLoad(hwQueue.element(0)) as unknown as NF,
      uint(HW_CAP) as unknown as NF,
    ) as unknown as NU;
    hwDrawBuf.element(0).assign(n.mul(uint(3)));
    hwDrawBuf.element(1).assign(uint(1));
    hwDrawBuf.element(2).assign(uint(0));
    hwDrawBuf.element(3).assign(uint(0));
  })().compute(1, [1]);
  (kHwArgs as unknown as ComputeKernel).setName('spikeHwArgs');

  // ---- HW big-triangle meshes (vertex pulling; fragment writes vis buffers) ----------
  const hwGeometry = new BufferGeometry();
  hwGeometry.setAttribute('position', new Float32BufferAttribute(new Float32Array(3), 3));
  hwGeometry.setIndirect(hwDrawAttr, 0);
  hwGeometry.boundingSphere = new Sphere(new Vector3(), Number.POSITIVE_INFINITY);

  const buildHwMaterial = (pass: 'depth' | 'payload' | 'optionA'): NodeMaterial => {
    const mat = new NodeMaterial();
    // exact split-payload varyings (interp of equal per-vertex values + round
    // in the fragment = exact integer transport without uint varyings)
    const vPayLo = varyingProperty('float', `hwPayLo_${pass}`) as unknown as NF;
    const vPayHi = varyingProperty('float', `hwPayHi_${pass}`) as unknown as NF;
    const vZ = varyingProperty('float', `hwZ_${pass}`) as unknown as NF;
    const vW = varyingProperty('float', `hwW_${pass}`) as unknown as NF;

    // If/toVar need an Fn stack — the whole vertex-pulling graph lives in one
    mat.vertexNode = Fn(() => {
      const triIndex = vertexIndex.div(3) as unknown as NU;
      const corner = vertexIndex.mod(3) as unknown as NU;
      const base = triIndex.mul(uint(2)).add(uint(1));
      const payload = (hwQueueRO.element(base) as unknown as NU).toVar();
      const instId = (hwQueueRO.element(base.add(uint(1))) as unknown as NU).toVar();
      const itemIdx = payload.shiftRight(uint(7));
      const localTri = payload.bitAnd(uint(127));
      const item = workQueueRO.element(itemIdx) as unknown as { x: NU; y: NU };
      const clusterId = item.y;
      const meta = metaBuf.element(clusterId) as unknown as NV4;

      const w0 = fetchWorldVert(instId, meta, localTri, 0);
      const w1 = fetchWorldVert(instId, meta, localTri, 1);
      const w2 = fetchWorldVert(instId, meta, localTri, 2);
      const world = (corner.equal(uint(1)) as unknown as { select: (a: unknown, b: unknown) => NV3 }).select(
        w1,
        (corner.equal(uint(2)) as unknown as { select: (a: unknown, b: unknown) => NV3 }).select(w2, w0),
      ) as unknown as NV3;
      const clip = ((uProj as unknown as { mul: (v: unknown) => NV4 }).mul(vec4(world, 1)) as NV4).toVar();

      (vPayLo as unknown as { assign: (v: unknown) => void }).assign(
        (payload.bitAnd(uint(0xffff)) as unknown as NF).toFloat(),
      );
      (vPayHi as unknown as { assign: (v: unknown) => void }).assign(
        (payload.shiftRight(uint(16)) as unknown as NF).toFloat(),
      );
      (vZ as unknown as { assign: (v: unknown) => void }).assign(clip.z);
      (vW as unknown as { assign: (v: unknown) => void }).assign(clip.w);
      return clip;
    })() as unknown as typeof mat.vertexNode;
    mat.fragmentNode = Fn(() => {
      const z = vZ.div(vW).toVar(); // screen-linear NDC depth (== SW kernel z)
      const pay = uint(vPayLo.round()).bitOr(uint(vPayHi.round()).shiftLeft(uint(16))).toVar();
      const fy = float(uH as unknown as NF).sub(screenCoordinate.y);
      const px = uint(fy).mul(uint(uW as unknown as NF)).add(uint(screenCoordinate.x));
      If((z.greaterThanEqual(0).and(z.lessThanEqual(1)) as unknown as NB), () => {
        if (pass === 'depth') {
          const bits = bitcast(z as unknown as NF, 'uint') as unknown as NU;
          atomicMin(visDepthAtomic.element(px), bits);
        } else if (pass === 'payload') {
          // cross-PIPELINE depth recompute diverges by a few ulp (the same
          // FMA-fusion non-invariance the depth-prepass hit, THREE-NOTES) —
          // the two HW materials are separate pipelines, so exact equality
          // drops fragments. Accept a ±64-ulp window (≈4e-6 NDC z — far
          // below any real surface separation); the SW kernels stay exact.
          // N3's fixed-point depth removes this class entirely.
          const bits = bitcast(z as unknown as NF, 'uint') as unknown as NU;
          const cur = visDepthRO.element(px) as unknown as NU;
          const du = max(bits as unknown as NF, cur as unknown as NF)
            .sub(min(bits as unknown as NF, cur as unknown as NF)) as unknown as NU;
          If((du.lessThanEqual(uint(64)) as unknown as NB), () => {
            visPayload.element(px).assign(pay);
          });
        } else {
          const zEnc = sqrt(sqrt(float(1).sub(z)));
          const dLo = uint(zEnc.mul(A_DEPTH_LO_MAX));
          const dHi = uint(zEnc.mul(A_DEPTH_HI_MAX));
          atomicMax(visDepthAtomic.element(px), dLo.shiftLeft(uint(A_LO_BITS)).bitOr(pay.bitAnd(uint((1 << A_LO_BITS) - 1))));
          atomicMax(visPayloadAtomic.element(px), dHi.shiftLeft(uint(A_HI_BITS)).bitOr(pay.shiftRight(uint(A_LO_BITS))));
        }
      });
      return vec4(0, 0, 0, 0);
    })() as unknown as typeof mat.fragmentNode;
    mat.depthTest = false;
    mat.depthWrite = false;
    mat.colorWrite = false;
    mat.fog = false;
    mat.lights = false;
    return mat;
  };

  const hwDepthMat = buildHwMaterial(isA ? 'optionA' : 'depth');
  const hwPayloadMat = isA ? null : buildHwMaterial('payload');

  const hwScene = new Scene();
  const hwMesh = new Mesh(hwGeometry, hwDepthMat);
  hwMesh.frustumCulled = false;
  hwScene.add(hwMesh);
  const hwRT = new RenderTarget(width, height, { depthBuffer: false });
  hwRT.texture.name = 'spikeHwPass';

  // ---- resolve material -----------------------------------------------------------------
  const resolveGeometry = new BufferGeometry();
  resolveGeometry.setAttribute(
    'position',
    new Float32BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3),
  );
  resolveGeometry.boundingSphere = new Sphere(new Vector3(), Number.POSITIVE_INFINITY);

  const resolveMat = new NodeMaterial();
  resolveMat.vertexNode = vec4(positionGeometry.xy, 0, 1) as unknown as typeof resolveMat.vertexNode;

  const fy = float(uH as unknown as NF).sub(screenCoordinate.y);
  const pixelIndex = uint(fy).mul(uint(uW as unknown as NF)).add(uint(screenCoordinate.x));
  const dRaw = visDepthRO.element(pixelIndex) as unknown as NU;
  const pRaw = visPayloadRO.element(pixelIndex) as unknown as NU;

  let payloadNode: NU;
  let emptyNode: NB;
  let depthOutNode: NF;
  if (isA) {
    emptyNode = dRaw.shiftRight(uint(A_LO_BITS)).equal(uint(0)) as unknown as NB;
    payloadNode = pRaw
      .bitAnd(uint((1 << A_HI_BITS) - 1))
      .shiftLeft(uint(A_LO_BITS))
      .bitOr(dRaw.bitAnd(uint((1 << A_LO_BITS) - 1))) as unknown as NU;
    const yN = (dRaw.shiftRight(uint(A_LO_BITS)) as unknown as NF).toFloat().div(A_DEPTH_LO_MAX);
    const y2 = yN.mul(yN);
    depthOutNode = float(1).sub(y2.mul(y2)) as unknown as NF;
  } else {
    emptyNode = dRaw.equal(uint(0xffffffff)) as unknown as NB;
    payloadNode = pRaw;
    depthOutNode = bitcast(dRaw, 'float') as unknown as NF;
  }

  resolveMat.fragmentNode = Fn(() => {
    If(emptyNode, () => {
      Discard();
    });
    // empty-depth pixels with a stale payload can't happen (cleared together),
    // but a payload miss (pass-2 tie loss) falls back to cluster 0 — visible
    // as a wrong-color pixel, which is exactly what we want to SEE if it occurs
    const itemIdx = payloadNode.shiftRight(uint(7));
    const localTri = payloadNode.bitAnd(uint(127));
    const item = workQueueRO.element(itemIdx) as unknown as { x: NU; y: NU };
    const instId = item.x;
    const clusterId = item.y;
    const meta = metaBuf.element(clusterId) as unknown as NV4;
    const kind = (meta as unknown as { x: NU }).x;

    const w0 = fetchWorldVert(instId, meta, localTri, 0);
    const w1 = fetchWorldVert(instId, meta, localTri, 1);
    const w2 = fetchWorldVert(instId, meta, localTri, 2);
    const faceN = normalize(cross(w1.sub(w0) as unknown as NV3, w2.sub(w0) as unknown as NV3)) as unknown as NV3;

    const L = normalize(vec3(0.55, 0.8, 0.25)) as unknown as NV3;
    const lambert = max(dot(faceN, L), 0).mul(0.85).add(0.18) as unknown as NF;
    const albedo = (kind.equal(uint(1)) as unknown as { select: (a: unknown, b: unknown) => NV3 }).select(
      vec3(0.30, 0.36, 0.22),
      vec3(0.42, 0.41, 0.40),
    );
    let col = albedo.mul(lambert) as unknown as NV3;
    if (clusterTint) {
      col = col.mul(0.65).add(hashColor(clusterId).mul(lambert).mul(0.35)) as unknown as NV3;
    }
    return vec4(col, 1);
  })() as unknown as typeof resolveMat.fragmentNode;
  resolveMat.depthNode = depthOutNode as unknown as typeof resolveMat.depthNode;
  resolveMat.depthTest = false;
  resolveMat.depthWrite = true;
  resolveMat.fog = false;
  resolveMat.lights = false;

  const resolveMesh = new Mesh(resolveGeometry, resolveMat);
  resolveMesh.frustumCulled = false;
  resolveMesh.renderOrder = 1;

  // ---- per-frame update -------------------------------------------------------------------
  const projScreen = new Matrix4();
  const frustum = new Frustum();

  const update = (renderer: Renderer, camera: PerspectiveCamera): void => {
    camera.updateMatrixWorld();
    projScreen.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    (uProj as unknown as { value: Matrix4 }).value.copy(projScreen);
    (uCamPos as unknown as { value: Vector3 }).value.copy(camera.position);
    frustum.setFromProjectionMatrix(projScreen);
    const arr = (frustumPlanes as unknown as { array: Vector4[] }).array;
    for (let i = 0; i < 6; i++) {
      const p = frustum.planes[i];
      if (p) arr[i]?.set(p.normal.x, p.normal.y, p.normal.z, p.constant);
    }

    type ComputeArg = Parameters<Renderer['compute']>[0];
    renderer.compute(kClear as unknown as ComputeArg);
    renderer.compute(kCull as unknown as ComputeArg);
    renderer.compute(kArgs as unknown as ComputeArg);
    renderer.compute(kRaster1 as unknown as ComputeArg, dispatchAttr as unknown as Parameters<Renderer['compute']>[1]);
    renderer.compute(kHwArgs as unknown as ComputeArg);

    // HW pass 1: hardware-rasterized big/near triangles write depth atomics
    const prevRT = renderer.getRenderTarget();
    renderer.setRenderTarget(hwRT);
    hwMesh.material = hwDepthMat;
    renderer.render(hwScene, camera);

    if (!isA && kRaster2 && hwPayloadMat) {
      // SW pass 2 sees final depth (incl. HW); then HW pass 2 stores payloads
      renderer.compute(kRaster2 as unknown as ComputeArg, dispatchAttr as unknown as Parameters<Renderer['compute']>[1]);
      hwMesh.material = hwPayloadMat;
      renderer.render(hwScene, camera);
    }
    renderer.setRenderTarget(prevRT);
  };

  const readCounts = async (renderer: Renderer): Promise<{ work: number; hw: number }> => {
    type Attr = Parameters<Renderer['getArrayBufferAsync']>[0];
    const [cBuf, hBuf] = await Promise.all([
      renderer.getArrayBufferAsync(countersAttr as unknown as Attr, null, 0, 4),
      renderer.getArrayBufferAsync(hwQueueAttr as unknown as Attr, null, 0, 4),
    ]);
    const work = new Uint32Array(cBuf)[0] ?? 0;
    if (work > WORK_CAP) {
      // eslint-disable-next-line no-console
      console.warn(`[spike] WORK QUEUE OVERFLOW: ${work} items > cap ${WORK_CAP} — geometry dropped`);
    }
    return { work, hw: new Uint32Array(hBuf)[0] ?? 0 };
  };

  return { resolveMesh, update, readCounts };
}
