/**
 * N2 vis-buffer raster on REGISTRY buffers — the spike pipeline (SpikeRaster)
 * ported onto GeometryRegistry mega-buffers + the world heightfield texture,
 * consuming the N2 cull chain's work queue. Option C only (D-N5): pass 1
 * atomicMin(f32-depth-bits), pass 2 equality payload store; near-crossing and
 * oversized triangles route to the HW queue whose fragment stage writes the
 * SAME buffers (depthWrite off — one resolve, one convention).
 *
 * Payload = (workItemIdx << 7) | localTri — qRaster doubles as the visible-
 * cluster list, so the resolve recovers (instance, cluster) in one
 * indirection (25-bit item headroom, F3).
 *
 * Float-edge scanline core stays example-verbatim (fixed-point integer edges
 * are N3); flat resolve = matClass palette or cluster-hash tint (the deferred
 * N1 checkpoint), face-normal lambert, real f32 depth out.
 */

import { Mesh, Scene, Vector3 } from 'three';
import { BufferGeometry, Float32BufferAttribute, RenderTarget, Sphere } from 'three';
import type { PerspectiveCamera, Texture } from 'three';
import {
  IndirectStorageBufferAttribute,
  NodeMaterial,
  StorageBufferAttribute,
  type Renderer,
} from 'three/webgpu';
import {
  Discard,
  Fn,
  If,
  atomicAdd,
  atomicMin,
  atomicStore,
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
  uint,
  varyingProperty,
  vec2,
  vec3,
  vec4,
  vertexIndex,
} from 'three/tsl';
import type { NB, NF, NI, NU, NV2, NV3, NV4 } from '../gpu/TSLTypes';
import { markFragmentWritable } from '../render/ThreePatches';
import { MESH_WORDS, VERT_WORDS } from './GeometryRegistry';
import type { RegistryGpu } from './GeometryRegistry';
import { DISPATCH_ROW, QRASTER_CAP, hashColor, instTransformPoint, instYaw, type NaniteCam } from './NaniteCommon';
import {
  aLoadU,
  bcF2U,
  bcU2F,
  dispatch,
  dispatchIndirect,
  elemU,
  localX,
  loopI,
  maxU,
  minU,
  readBuffer,
  returnIf,
  sU32Views,
  texLoadR,
  toF,
  wgLinear,
} from './Tsl';
import type { BufOf, UV2 } from './Tsl';

const HW_CAP = 262_144;
const MAX_RASTER_SIZE = 16;
const NEAR_EPS = 1e-4;

interface ComputeKernel {
  setName(name: string): unknown;
}

export interface NaniteRasterHandles {
  /** fullscreen resolve mesh in its own scene — render with the main camera */
  resolveScene: Scene;
  /** per-frame: vis clear → SW pass 1 → HW pass 1 → SW pass 2 → HW pass 2 */
  update(renderer: Renderer, camera: PerspectiveCamera): void;
  readHwCount(renderer: Renderer): Promise<number>;
}

/** Option C vis buffers — created OUTSIDE the raster so the HZB (which the
 *  cull consumes) can view the depth buffer without a builder cycle */
export interface NaniteVisBuffers {
  depthAttr: StorageBufferAttribute;
  depthV: ReturnType<typeof sU32Views>;
  payloadAttr: StorageBufferAttribute;
  payloadV: ReturnType<typeof sU32Views>;
}

export function makeVisBuffers(pixelCount: number): NaniteVisBuffers {
  const depthAttr = new StorageBufferAttribute(new Uint32Array(pixelCount), 1);
  markFragmentWritable(depthAttr);
  const payloadAttr = new StorageBufferAttribute(new Uint32Array(pixelCount), 1);
  markFragmentWritable(payloadAttr);
  return {
    depthAttr,
    depthV: sU32Views(depthAttr, pixelCount),
    payloadAttr,
    payloadV: sU32Views(payloadAttr, pixelCount),
  };
}

export function buildNaniteRaster(
  gpu: RegistryGpu,
  heightTex: Texture,
  cam: NaniteCam,
  cull: { qRasterRO: BufOf<UV2>; rasterDispatchAttr: IndirectStorageBufferAttribute },
  vis: NaniteVisBuffers,
  tint: 'flat' | 'cluster',
): NaniteRasterHandles {
  const { width, height } = cam;
  const pixelCount = width * height;
  const qRasterRO = cull.qRasterRO;
  const visDepthV = vis.depthV;
  const visPayloadV = vis.payloadV;

  // hwQueue: [0] = atomic count, then (payload, instId) pairs
  const hwQueueAttr = new StorageBufferAttribute(new Uint32Array(1 + HW_CAP * 2), 1);
  const hwQueueV = sU32Views(hwQueueAttr, 1 + HW_CAP * 2);
  const hwDrawAttr = new IndirectStorageBufferAttribute(new Uint32Array(4), 4);
  const hwDrawBuf = sU32Views(hwDrawAttr as unknown as StorageBufferAttribute, 4).rw;

  // ---- shared fetch helpers ------------------------------------------------------
  /** per-(instance, cluster) decode shared by the 3 corner fetches */
  interface VertCtx {
    isHF: NB;
    A: NV4;
    B: NV4;
    yawSc: ReturnType<typeof instYaw>;
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

  /** world-space corner v of ctx's localTri (heightfield convention = spike:
   *  up-facing CCW, even tri (0,0)(0,1)(1,1), odd tri (0,0)(1,1)(1,0)) */
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

  const edgeFn = (a: NV2, b: NV2, p: NV2): NF =>
    p.y.sub(a.y).mul(b.x.sub(a.x)).sub(p.x.sub(a.x).mul(b.y.sub(a.y))) as unknown as NF;

  // ---- kVisClear ------------------------------------------------------------------
  const kVisClear = Fn(() => {
    If(instanceIndex.lessThan(uint(pixelCount)), () => {
      atomicStore(visDepthV.atomic.element(instanceIndex), uint(0xffffffff));
      atomicStore(visPayloadV.atomic.element(instanceIndex), uint(0xffffffff));
    });
    If(instanceIndex.equal(uint(0)), () => {
      atomicStore(hwQueueV.atomic.element(0), uint(0));
    });
  })().compute(pixelCount, [256]);
  (kVisClear as unknown as ComputeKernel).setName('nanVisClear');

  // ---- SW raster kernels (Option C two-pass; spike-verbatim scanline core) ----------
  const rasterKernel = (mode: 'depth' | 'payload'): unknown => {
    const kn = Fn(() => {
      const itemIdx = wgLinear(DISPATCH_ROW).toVar();
      const localTri = localX().toVar();
      const itemCount = qRasterRO.element(0).x;
      returnIf(itemIdx.greaterThanEqual(itemCount));
      const item = qRasterRO.element(itemIdx.add(uint(1)));
      const instId = item.x.toVar();
      const ci = item.y.toVar();
      const ctx = makeCtx(instId, ci);

      If(localTri.lessThan(ctx.triCount), () => {
        const w0 = fetchWorldVert(ctx, localTri, 0);
        const w1 = fetchWorldVert(ctx, localTri, 1);
        const w2 = fetchWorldVert(ctx, localTri, 2);

        const p0 = cam.vp.mul(vec4(w0, 1)).toVar();
        const p1 = cam.vp.mul(vec4(w1, 1)).toVar();
        const p2 = cam.vp.mul(vec4(w2, 1)).toVar();

        const payload = itemIdx.shiftLeft(uint(7)).bitOr(localTri).toVar();

        const nearOK = p0.w
          .greaterThan(NEAR_EPS)
          .and(p1.w.greaterThan(NEAR_EPS))
          .and(p2.w.greaterThan(NEAR_EPS));

        If(nearOK.not(), () => {
          // near-plane crossing → HW path clips it (never drop, F10c)
          if (mode === 'depth') {
            const slot = atomicAdd(hwQueueV.atomic.element(0), uint(1)) as unknown as NU;
            If(slot.lessThan(uint(HW_CAP)), () => {
              const base = slot.mul(uint(2)).add(uint(1));
              atomicStore(hwQueueV.atomic.element(base), payload);
              atomicStore(hwQueueV.atomic.element(base.add(uint(1))), instId);
            });
          }
        }).Else(() => {
          const ndc0 = p0.xyz.div(p0.w).toVar();
          const ndc1 = p1.xyz.div(p1.w).toVar();
          const ndc2 = p2.xyz.div(p2.w).toVar();

          const areaNdc = edgeFn(
            ndc0.xy as unknown as NV2,
            ndc1.xy as unknown as NV2,
            ndc2.xy as unknown as NV2,
          );
          If(areaNdc.greaterThan(0), () => {
            const W = float(cam.uW);
            const H = float(cam.uH);
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
            const validBB = startX.lessThanEqual(endX).and(startY.lessThanEqual(endY));
            const smallEnough = bbW
              .lessThanEqual(int(MAX_RASTER_SIZE))
              .and(bbH.lessThanEqual(int(MAX_RASTER_SIZE)));

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
              const bias0 = tl0.select(float(0), float(-1e-5));
              const bias1 = tl1.select(float(0), float(-1e-5));
              const bias2 = tl2.select(float(0), float(-1e-5));

              const pStart = vec2(float(startX).add(0.5), float(startY).add(0.5));
              const rw0 = edgeFn(s1 as unknown as NV2, s2 as unknown as NV2, pStart as unknown as NV2)
                .add(bias0)
                .toVar();
              const rw1 = edgeFn(s2 as unknown as NV2, s0 as unknown as NV2, pStart as unknown as NV2)
                .add(bias1)
                .toVar();
              const rw2 = edgeFn(s0 as unknown as NV2, s1 as unknown as NV2, pStart as unknown as NV2)
                .add(bias2)
                .toVar();

              const b0s = rw0.div(area);
              const b1s = rw1.div(area);
              const b2s = rw2.div(area);
              const rowZ = b0s.mul(ndc0.z).add(b1s.mul(ndc1.z)).add(b2s.mul(ndc2.z)).toVar();
              const stepXz = stepXw0
                .div(area)
                .mul(ndc0.z)
                .add(stepXw1.div(area).mul(ndc1.z))
                .add(stepXw2.div(area).mul(ndc2.z));
              const stepYz = stepYw0
                .div(area)
                .mul(ndc0.z)
                .add(stepYw1.div(area).mul(ndc1.z))
                .add(stepYw2.div(area).mul(ndc2.z));

              loopI('sy', startY as unknown as NI, endY as unknown as NI, (y) => {
                const cw0 = rw0.toVar();
                const cw1 = rw1.toVar();
                const cw2 = rw2.toVar();
                const cz = rowZ.toVar();
                loopI('sx', startX as unknown as NI, endX as unknown as NI, (x) => {
                  If(
                    cw0
                      .greaterThanEqual(0)
                      .and(cw1.greaterThanEqual(0))
                      .and(cw2.greaterThanEqual(0))
                      .and(cz.greaterThanEqual(0))
                      .and(cz.lessThanEqual(1)),
                    () => {
                      const px = uint(y).mul(uint(cam.uW)).add(uint(x));
                      const bits = bcF2U(cz as unknown as NF);
                      if (mode === 'depth') {
                        const cur = aLoadU(visDepthV.atomic.element(px));
                        If(bits.lessThan(cur), () => {
                          atomicMin(visDepthV.atomic.element(px), bits);
                        });
                      } else {
                        const cur = elemU(visDepthV.ro, px);
                        If(bits.equal(cur), () => {
                          (visPayloadV.rw as unknown as { element(i: NU): { assign(v: NU): void } })
                            .element(px)
                            .assign(payload);
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
              if (mode === 'depth') {
                If(validBB, () => {
                  // big triangle → HW queue
                  const slot = atomicAdd(hwQueueV.atomic.element(0), uint(1)) as unknown as NU;
                  If(slot.lessThan(uint(HW_CAP)), () => {
                    const base = slot.mul(uint(2)).add(uint(1));
                    atomicStore(hwQueueV.atomic.element(base), payload);
                    atomicStore(hwQueueV.atomic.element(base.add(uint(1))), instId);
                  });
                });
              }
            });
          });
        });
      });
    })().compute(QRASTER_CAP * 128, [128]);
    return kn;
  };

  const kRasterDepth = rasterKernel('depth');
  (kRasterDepth as ComputeKernel).setName('nanRasterDepth');
  const kRasterPayload = rasterKernel('payload');
  (kRasterPayload as ComputeKernel).setName('nanRasterPayload');

  // ---- kHwArgs ----------------------------------------------------------------------
  const kHwArgs = Fn(() => {
    const n = minU(aLoadU(hwQueueV.atomic.element(0)), uint(HW_CAP));
    hwDrawBuf.element(0).assign(n.mul(uint(3)));
    hwDrawBuf.element(1).assign(uint(1));
    hwDrawBuf.element(2).assign(uint(0));
    hwDrawBuf.element(3).assign(uint(0));
  })().compute(1, [1]);
  (kHwArgs as unknown as ComputeKernel).setName('nanHwArgs');

  // ---- HW big/near-triangle passes (vertex pulling; fragment writes vis bufs) --------
  const hwGeometry = new BufferGeometry();
  hwGeometry.setAttribute('position', new Float32BufferAttribute(new Float32Array(3), 3));
  hwGeometry.setIndirect(hwDrawAttr, 0);
  hwGeometry.boundingSphere = new Sphere(new Vector3(), Number.POSITIVE_INFINITY);

  const buildHwMaterial = (pass: 'depth' | 'payload'): NodeMaterial => {
    const mat = new NodeMaterial();
    const vPayLo = varyingProperty('float', `nanPayLo_${pass}`) as unknown as NF;
    const vPayHi = varyingProperty('float', `nanPayHi_${pass}`) as unknown as NF;
    const vZ = varyingProperty('float', `nanZ_${pass}`) as unknown as NF;
    const vW = varyingProperty('float', `nanW_${pass}`) as unknown as NF;

    mat.vertexNode = Fn(() => {
      const triIndex = vertexIndex.div(3) as unknown as NU;
      const corner = vertexIndex.mod(3) as unknown as NU;
      const base = triIndex.mul(uint(2)).add(uint(1));
      const payload = elemU(hwQueueV.ro, base).toVar();
      const instId = elemU(hwQueueV.ro, base.add(uint(1))).toVar();
      const itemIdx = payload.shiftRight(uint(7));
      const localTri = payload.bitAnd(uint(127));
      const item = qRasterRO.element(itemIdx.add(uint(1)));
      const ci = item.y.toVar();
      const ctx = makeCtx(instId, ci);

      const w0 = fetchWorldVert(ctx, localTri, 0);
      const w1 = fetchWorldVert(ctx, localTri, 1);
      const w2 = fetchWorldVert(ctx, localTri, 2);
      const world = corner
        .equal(uint(1))
        .select(w1, corner.equal(uint(2)).select(w2, w0)) as unknown as NV3;
      const clip = cam.vp.mul(vec4(world, 1)).toVar();

      (vPayLo as unknown as { assign: (v: unknown) => void }).assign(toF(payload.bitAnd(uint(0xffff))));
      (vPayHi as unknown as { assign: (v: unknown) => void }).assign(toF(payload.shiftRight(uint(16))));
      (vZ as unknown as { assign: (v: unknown) => void }).assign(clip.z);
      (vW as unknown as { assign: (v: unknown) => void }).assign(clip.w);
      return clip;
    })() as unknown as typeof mat.vertexNode;

    mat.fragmentNode = Fn(() => {
      const z = vZ.div(vW).toVar();
      const pay = uint(vPayLo.round()).bitOr(uint(vPayHi.round()).shiftLeft(uint(16))).toVar();
      const fy = float(cam.uH).sub(screenCoordinate.y);
      const px = uint(fy).mul(uint(cam.uW)).add(uint(screenCoordinate.x));
      If(z.greaterThanEqual(0).and(z.lessThanEqual(1)), () => {
        const bits = bcF2U(z as unknown as NF);
        if (pass === 'depth') {
          atomicMin(visDepthV.atomic.element(px), bits);
        } else {
          // cross-pipeline FMA divergence: ±64-ulp equality window (N0 gotcha;
          // N3 fixed-point depth removes the class)
          const cur = elemU(visDepthV.ro, px);
          const du = maxU(bits, cur).sub(minU(bits, cur));
          If(du.lessThanEqual(uint(64)), () => {
            (visPayloadV.rw as unknown as { element(i: NU): { assign(v: NU): void } })
              .element(px)
              .assign(pay);
          });
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

  const hwDepthMat = buildHwMaterial('depth');
  const hwPayloadMat = buildHwMaterial('payload');
  const hwScene = new Scene();
  const hwMesh = new Mesh(hwGeometry, hwDepthMat);
  hwMesh.frustumCulled = false;
  hwScene.add(hwMesh);
  const hwRT = new RenderTarget(width, height, { depthBuffer: false });
  hwRT.texture.name = 'nanHwPass';

  // ---- flat resolve -----------------------------------------------------------------
  const resolveGeometry = new BufferGeometry();
  resolveGeometry.setAttribute(
    'position',
    new Float32BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3),
  );
  resolveGeometry.boundingSphere = new Sphere(new Vector3(), Number.POSITIVE_INFINITY);

  const resolveMat = new NodeMaterial();
  resolveMat.vertexNode = vec4(positionGeometry.xy, 0, 1) as unknown as typeof resolveMat.vertexNode;
  resolveMat.fragmentNode = Fn(() => {
    const fy = float(cam.uH).sub(screenCoordinate.y);
    const pixelIndex = uint(fy).mul(uint(cam.uW)).add(uint(screenCoordinate.x));
    const dRaw = elemU(visDepthV.ro, pixelIndex);
    const pRaw = elemU(visPayloadV.ro, pixelIndex);
    If(dRaw.equal(uint(0xffffffff)), () => {
      Discard();
    });
    const itemIdx = pRaw.shiftRight(uint(7));
    const localTri = pRaw.bitAnd(uint(127));
    const item = qRasterRO.element(itemIdx.add(uint(1)));
    const instId = item.x.toVar();
    const ci = item.y.toVar();
    const ctx = makeCtx(instId, ci);

    const w0 = fetchWorldVert(ctx, localTri, 0);
    const w1 = fetchWorldVert(ctx, localTri, 1);
    const w2 = fetchWorldVert(ctx, localTri, 2);
    const faceN = normalize(
      cross(w1.sub(w0) as unknown as NV3, w2.sub(w0) as unknown as NV3),
    ) as unknown as NV3;

    const L = normalize(vec3(0.55, 0.8, 0.25)) as unknown as NV3;
    const lambert = max(dot(faceN, L), 0).mul(0.85).add(0.18) as unknown as NF;

    // matClass palette (flat) — terrain/rock/bark/deadwood/leaf/grass/debris
    const matClass = elemU(gpu.meshes, ctx.meshId.mul(uint(MESH_WORDS)).add(uint(6)))
      .shiftRight(uint(8))
      .bitAnd(uint(0xff))
      .toVar();
    const isT = matClass.equal(uint(0));
    const isR = matClass.equal(uint(1));
    const isB = matClass.equal(uint(2));
    const isD = matClass.equal(uint(3));
    const albedo = isT
      .select(
        vec3(0.3, 0.36, 0.22),
        isR.select(
          vec3(0.42, 0.41, 0.4),
          isB.select(vec3(0.36, 0.27, 0.19), isD.select(vec3(0.33, 0.28, 0.22), vec3(0.35, 0.33, 0.3))),
        ),
      )
      .toVar();
    let col: NV3;
    if (tint === 'cluster') {
      col = hashColor(ci).mul(lambert) as unknown as NV3;
    } else {
      col = (albedo as unknown as NV3).mul(lambert) as unknown as NV3;
    }
    return vec4(col, 1);
  })() as unknown as typeof resolveMat.fragmentNode;
  resolveMat.depthNode = Fn(() => {
    const fy = float(cam.uH).sub(screenCoordinate.y);
    const pixelIndex = uint(fy).mul(uint(cam.uW)).add(uint(screenCoordinate.x));
    return bcU2F(elemU(visDepthV.ro, pixelIndex));
  })() as unknown as typeof resolveMat.depthNode;
  resolveMat.depthTest = false;
  resolveMat.depthWrite = true;
  resolveMat.fog = false;
  resolveMat.lights = false;

  const resolveMesh = new Mesh(resolveGeometry, resolveMat);
  resolveMesh.frustumCulled = false;
  const resolveScene = new Scene();
  resolveScene.add(resolveMesh);

  // ---- per-frame ------------------------------------------------------------------
  const update = (renderer: Renderer, camera: PerspectiveCamera): void => {
    dispatch(renderer, kVisClear);
    dispatchIndirect(renderer, kRasterDepth, cull.rasterDispatchAttr);
    dispatch(renderer, kHwArgs);

    const prevRT = renderer.getRenderTarget();
    renderer.setRenderTarget(hwRT);
    hwMesh.material = hwDepthMat;
    renderer.render(hwScene, camera);

    dispatchIndirect(renderer, kRasterPayload, cull.rasterDispatchAttr);
    hwMesh.material = hwPayloadMat;
    renderer.render(hwScene, camera);
    renderer.setRenderTarget(prevRT);
  };

  const readHwCount = async (renderer: Renderer): Promise<number> => {
    const buf = await readBuffer(renderer, hwQueueAttr, 0, 4);
    return new Uint32Array(buf)[0] ?? 0;
  };

  return { resolveScene, update, readHwCount };
}
