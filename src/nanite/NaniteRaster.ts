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
 * Scanline core (N3a): FIXED-POINT integer edge functions — verts snapped to
 * a 1/256-px grid (8 subpixel bits, HW convention), coverage + top-left rule
 * in exact i32 math (watertight; replaces the example's float −1e-5 bias).
 * Flat resolve = matClass palette or cluster-hash tint (the deferred N1
 * checkpoint), face-normal lambert, real f32 depth out.
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
  max,
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
import type { NB, NF, NI, NU, NV2, NV3 } from '../gpu/TSLTypes';
import { markFragmentWritable } from '../render/ThreePatches';
import { MESH_WORDS } from './GeometryRegistry';
import type { RegistryGpu } from './GeometryRegistry';
import { DISPATCH_ROW, QRASTER_CAP, hashColor, type NaniteCam } from './NaniteCommon';
import { makeFetch, type TerrainDisp, type TrunkWindOpt } from './NaniteFetch';
import {
  aLoadU,
  bcF2U,
  bcU2F,
  dispatch,
  dispatchIndirect,
  elemU,
  localX,
  loopI,
  maxI,
  minI,
  minU,
  readBuffer,
  returnIf,
  sU32Views,
  toF,
  toI,
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
  /** clear vis buffers + hw queue (every frame, before any pass) */
  clearVis(renderer: Renderer): void;
  /** SW depth over phase-1 items */
  depth1(renderer: Renderer): void;
  /** SW depth over phase-2 appended items (slot base; 0 workgroups when none) */
  depth2(renderer: Renderer): void;
  /** HW big/near-tri depth render (re-runs hwArgs; full-queue redraw is
   *  idempotent atomicMin) */
  hwDepth(renderer: Renderer, camera: PerspectiveCamera): void;
  /** SW payload over ALL items vs final depth + HW payload render */
  payload(renderer: Renderer, camera: PerspectiveCamera): void;
  readHwCount(renderer: Renderer): Promise<number>;
  /** count covered/orphan pixels (dispatch after payload; ?audit=1) */
  audit(renderer: Renderer): void;
  readAudit(renderer: Renderer): Promise<{ orphans: number; covered: number }>;
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
  cull: {
    qRasterRO: BufOf<UV2>;
    rasterDispatchAttr: IndirectStorageBufferAttribute;
    rasterDispatch2Attr: IndirectStorageBufferAttribute;
    rasterDispatchFullAttr: IndirectStorageBufferAttribute;
  },
  vis: NaniteVisBuffers,
  tint: 'flat' | 'cluster',
  /** false (?shade=0): pure matClass color, no lambert — the parity gate's
   *  shading-free mode (coverage/structure compare only) */
  shade = true,
  /** terrain micro-displacement (frame mode; dbg views omit — NaniteFetch) */
  disp?: TerrainDisp,
  /** trunk wind (frame mode) — MUST match the resolve's makeFetch so the
   *  rastered geometry and the resolve's barycentric corners agree */
  wind?: TrunkWindOpt,
): NaniteRasterHandles {
  const { width, height } = cam;
  const pixelCount = width * height;
  // PERF-3 raster ablation (?rdbg=1|2): BUILD-TIME gate — 0 / absent emits the
  // depth kernel UNCHANGED (production pristine, zero instrumentation). Used to
  // attribute nanRasterDepth's per-triangle setup (resolution scaling proved it
  // ~70% setup-bound) to its sub-stages — see the two sinks in the depth kernel.
  const rdbg = Number(new URLSearchParams(window.location.search).get('rdbg') ?? '0');
  const qRasterRO = cull.qRasterRO;
  const visDepthV = vis.depthV;
  const visPayloadV = vis.payloadV;

  // hwQueue: [0] = atomic count, then (payload, instId) pairs
  const hwQueueAttr = new StorageBufferAttribute(new Uint32Array(1 + HW_CAP * 2), 1);
  const hwQueueV = sU32Views(hwQueueAttr, 1 + HW_CAP * 2);
  const hwDrawAttr = new IndirectStorageBufferAttribute(new Uint32Array(4), 4);
  const hwDrawBuf = sU32Views(hwDrawAttr as unknown as StorageBufferAttribute, 4).rw;

  // consistency audit (?audit=1): [0] = orphans (depth written but payload
  // never matched it — ANY pass disagreement, SW or HW, shows up here),
  // [1] = covered pixels
  const auditAttr = new StorageBufferAttribute(new Uint32Array(4), 1);
  const auditV = sU32Views(auditAttr, 4);

  // ---- shared fetch helpers (NaniteFetch.ts — also the resolve's decode) ----------
  const { makeCtx, fetchWorldVert } = makeFetch(gpu, heightTex, disp, wind);

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
      atomicStore(auditV.atomic.element(0), uint(0));
      atomicStore(auditV.atomic.element(1), uint(0));
    });
  })().compute(pixelCount, [256]);
  (kVisClear as unknown as ComputeKernel).setName('nanVisClear');

  // ---- SW raster kernels (Option C two-pass; fixed-point integer scanline) ----------
  // `phase2` offsets the work-item index by qRaster[0].y (the phase-2 base
  // written by kRasterArgs2) so the appended range rasters without touching
  // phase-1 items; depth1/payload start at 0.
  const rasterKernel = (mode: 'depth' | 'payload', phase2 = false): unknown => {
    const kn = Fn(() => {
      const head = qRasterRO.element(0);
      const itemIdx = phase2
        ? head.y.add(wgLinear(DISPATCH_ROW)).toVar()
        : wgLinear(DISPATCH_ROW).toVar();
      const localTri = localX().toVar();
      const itemCount = head.x;
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

        if (mode === 'depth' && rdbg === 1) {
          // ?rdbg=1 — stop right after the 3 vertex fetch+transforms. The sink
          // atomicMin consumes ALL THREE clip positions BEFORE the early-out, so
          // the compiler cannot sink the fetchWorldVert/transform work past the
          // return (it would be a side-effecting use). returnIf's guard is a
          // runtime-true compare (uint ≥ 0) ⇒ always returns here, yet the
          // downstream stays REACHABLE (no unreachable-code error). Isolates:
          // work-item fetch + ctx + 3× fetchWorldVert + 3× vp transform.
          const sinkV = p0.z
            .div(p0.w.max(NEAR_EPS))
            .add(p1.z.div(p1.w.max(NEAR_EPS)))
            .add(p2.z.div(p2.w.max(NEAR_EPS)))
            .mul(1 / 3)
            .clamp(0, 1);
          const sinkPx = itemIdx.mul(uint(2654435761)).add(localTri).mod(uint(pixelCount));
          atomicMin(visDepthV.atomic.element(sinkPx), bcF2U(sinkV as unknown as NF));
          returnIf(itemCount.greaterThanEqual(uint(0)));
        }

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

            // FIXED-POINT snap (N3a): 1/256-px integer grid — 8 subpixel bits,
            // the D3D HW convention. All coverage below is exact i32 math:
            // watertight at shared edges and bit-identical between the depth
            // and payload kernels by construction. WGSL f32→i32 SATURATES, so
            // far off-screen verts read as a huge extent → HW route.
            const xi0 = toI(s0.x.mul(256).round()).toVar();
            const yi0 = toI(s0.y.mul(256).round()).toVar();
            const xi1 = toI(s1.x.mul(256).round()).toVar();
            const yi1 = toI(s1.y.mul(256).round()).toVar();
            const xi2 = toI(s2.x.mul(256).round()).toVar();
            const yi2 = toI(s2.y.mul(256).round()).toVar();

            // whole-pixel bbox (trunc-div ≠ floor only below 0, where start
            // clamps to 0 / validBB rejects — harmless)
            const bbMinX = minI(xi0, minI(xi1, xi2)).div(toI(256)).toVar();
            const bbMaxX = maxI(xi0, maxI(xi1, xi2)).div(toI(256)).toVar();
            const bbMinY = minI(yi0, minI(yi1, yi2)).div(toI(256)).toVar();
            const bbMaxY = maxI(yi0, maxI(yi1, yi2)).div(toI(256)).toVar();

            // SW only when the UNCLAMPED extent is small (the float core sized
            // the clamped box — a screen-spanning tri with a 16-px on-screen
            // sliver ran SW with unbounded edge terms; now it routes HW).
            // Bound: deltas ≤ ~17 px ⇒ every edge term < 2^26, i32-safe.
            const smallEnough = bbMaxX
              .sub(bbMinX)
              .lessThanEqual(toI(MAX_RASTER_SIZE))
              .and(bbMaxY.sub(bbMinY).lessThanEqual(toI(MAX_RASTER_SIZE)));

            const startX = maxI(toI(0), bbMinX).toVar();
            const endX = minI(toI(width - 1), bbMaxX).toVar();
            const startY = maxI(toI(0), bbMinY).toVar();
            const endY = minI(toI(height - 1), bbMaxY).toVar();
            const validBB = startX.lessThanEqual(endX).and(startY.lessThanEqual(endY));

            If((smallEnough as unknown as { and: (o: unknown) => NB }).and(validBB), () => {
              // integer twice-area; snapping can collapse/flip a sub-1/256-px
              // sliver → skip (also guards the reciprocal — the float core
              // divided blindly)
              const area2 = yi2
                .sub(yi0)
                .mul(xi1.sub(xi0))
                .sub(xi2.sub(xi0).mul(yi1.sub(yi0)))
                .toVar();
              If(area2.greaterThan(toI(0)), () => {
                // edge i is opposite vertex i; ex/ey = dE per +1 UNIT (1/256 px)
                const ex0 = yi1.sub(yi2).toVar();
                const ey0 = xi2.sub(xi1).toVar();
                const ex1 = yi2.sub(yi0).toVar();
                const ey1 = xi0.sub(xi2).toVar();
                const ex2 = yi0.sub(yi1).toVar();
                const ey2 = xi1.sub(xi0).toVar();

                // top-left rule, same orientation convention as the float core:
                // the boundary E == 0 is owned iff dE/dx < 0, or dE/dx == 0 and
                // dE/dy > 0; the -1 bias turns ≥0 into >0 on unowned edges —
                // exact and scale-free (the -1e-5 float bias competed with ulp
                // at edge-term scale ~1e6)
                const tlBias = (ex: NI, ey: NI): NI =>
                  ex
                    .lessThan(toI(0))
                    .or(ex.equal(toI(0)).and(ey.greaterThan(toI(0))))
                    .select(toI(0), toI(-1)) as unknown as NI;
                const bias0 = tlBias(ex0 as unknown as NI, ey0 as unknown as NI);
                const bias1 = tlBias(ex1 as unknown as NI, ey1 as unknown as NI);
                const bias2 = tlBias(ex2 as unknown as NI, ey2 as unknown as NI);

                // E at the center of (startX, startY), in units²
                const pcx = startX.mul(toI(256)).add(toI(128)).toVar();
                const pcy = startY.mul(toI(256)).add(toI(128)).toVar();
                const rw0 = pcy
                  .sub(yi1)
                  .mul(xi2.sub(xi1))
                  .sub(pcx.sub(xi1).mul(yi2.sub(yi1)))
                  .add(bias0)
                  .toVar();
                const rw1 = pcy
                  .sub(yi2)
                  .mul(xi0.sub(xi2))
                  .sub(pcx.sub(xi2).mul(yi0.sub(yi2)))
                  .add(bias1)
                  .toVar();
                const rw2 = pcy
                  .sub(yi0)
                  .mul(xi1.sub(xi0))
                  .sub(pcx.sub(xi0).mul(yi1.sub(yi0)))
                  .add(bias2)
                  .toVar();
                // per-PIXEL steps = ex/ey × 256 units
                const sx0 = ex0.mul(toI(256)).toVar();
                const sx1 = ex1.mul(toI(256)).toVar();
                const sx2 = ex2.mul(toI(256)).toVar();
                const sy0 = ey0.mul(toI(256)).toVar();
                const sy1 = ey1.mul(toI(256)).toVar();
                const sy2 = ey2.mul(toI(256)).toVar();
                const rcpArea = float(1).div(toF(area2 as unknown as NI)).toVar();

                if (mode === 'depth' && rdbg === 2) {
                  // ?rdbg=2 — stop right before the scanline loop. The sink folds
                  // the edge-setup vars (rcpArea, rw0..2, the per-pixel steps) so
                  // none are sunk past / DCE'd, forcing the full per-triangle
                  // setup to be timed. (rdbg2 − rdbg1) = near/backface + ndc +
                  // fixed-point snap + bbox + edge setup; (full − rdbg2) = the
                  // per-pixel loop (coverage + depth interp + atomicMin).
                  const sinkV = rcpArea
                    .add(toF(rw0 as unknown as NI))
                    .add(toF(rw1 as unknown as NI))
                    .add(toF(rw2 as unknown as NI))
                    .add(toF(sx0 as unknown as NI))
                    .add(toF(sy0 as unknown as NI))
                    .clamp(0, 1);
                  const sinkPx = uint(startY).mul(uint(cam.uW)).add(uint(startX));
                  atomicMin(visDepthV.atomic.element(sinkPx), bcF2U(sinkV as unknown as NF));
                  returnIf(itemCount.greaterThanEqual(uint(0)));
                }

                loopI('sy', startY as unknown as NI, endY as unknown as NI, (y) => {
                  const cw0 = rw0.toVar();
                  const cw1 = rw1.toVar();
                  const cw2 = rw2.toVar();
                  loopI('sx', startX as unknown as NI, endX as unknown as NI, (x) => {
                    If(
                      cw0
                        .greaterThanEqual(toI(0))
                        .and(cw1.greaterThanEqual(toI(0)))
                        .and(cw2.greaterThanEqual(toI(0))),
                      () => {
                        // depth from the UNBIASED integer weights: the top-left
                        // −1 biases belong to COVERAGE only. Folding them into
                        // the weights divides by area2 while the weights sum to
                        // area2−(1..3) — a RELATIVE error of ~bias/area2 that is
                        // ulp-level on big triangles but ~5e-4 on sub-pixel far
                        // slivers (area2 ~10³ units²) ⇒ far depth biased NEARER
                        // by hundreds of meters. Self-consistent across passes
                        // (audit/parity blind) — found at N4-C0 when WATER
                        // depth-tested against the buffer. Unbiased weights sum
                        // to area2 exactly (integer identity): cz is exact, and
                        // both passes still compute identical bits.
                        const uw0 = cw0.sub(bias0).toVar();
                        const uw1 = cw1.sub(bias1).toVar();
                        const uw2 = cw2.sub(bias2).toVar();
                        const cz = toF(uw0 as unknown as NI)
                          .mul(ndc0.z)
                          .add(toF(uw1 as unknown as NI).mul(ndc1.z))
                          .add(toF(uw2 as unknown as NI).mul(ndc2.z))
                          .mul(rcpArea)
                          .toVar();
                        If(cz.greaterThanEqual(0).and(cz.lessThanEqual(1)), () => {
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
                        });
                      },
                    );
                    cw0.addAssign(sx0);
                    cw1.addAssign(sx1);
                    cw2.addAssign(sx2);
                  });
                  rw0.addAssign(sy0);
                  rw1.addAssign(sy1);
                  rw2.addAssign(sy2);
                });
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
  const kRasterDepth2 = rasterKernel('depth', true);
  (kRasterDepth2 as ComputeKernel).setName('nanRasterDepth2');
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

  // ---- kAudit (?audit=1) — run AFTER the payload passes: a covered pixel
  // whose payload is still the clear sentinel means no pass-2 writer ever
  // reproduced the stored depth (raster pass inconsistency). Gate: 0.
  const kAudit = Fn(() => {
    returnIf(instanceIndex.greaterThanEqual(uint(pixelCount)));
    const d = elemU(visDepthV.ro, instanceIndex);
    If(d.notEqual(uint(0xffffffff)), () => {
      atomicAdd(auditV.atomic.element(1), uint(1));
      const p = elemU(visPayloadV.ro, instanceIndex);
      If(p.equal(uint(0xffffffff)), () => {
        atomicAdd(auditV.atomic.element(0), uint(1));
      });
    });
  })().compute(pixelCount, [256]);
  (kAudit as unknown as ComputeKernel).setName('nanAudit');

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
          // EXACT equality (N3a) — the N0 ±64-ulp cross-pipeline window is
          // retired: 0 orphans measured at exact equality on real HW load
          // (25k tris underfoot); the ?audit=1 oracle re-catches any future
          // driver/three divergence
          const cur = elemU(visDepthV.ro, px);
          If(bits.equal(cur), () => {
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
    // orphan (pass-2 never matched pass-1's depth): background, not a garbage
    // payload decode — black-pixel probes then see it as a hole
    If(pRaw.equal(uint(0xffffffff)), () => {
      Discard();
    });
    const itemIdx = pRaw.shiftRight(uint(7));
    const localTri = pRaw.bitAnd(uint(127));
    const item = qRasterRO.element(itemIdx.add(uint(1)));
    const instId = item.x.toVar();
    const ci = item.y.toVar();
    const ctx = makeCtx(instId, ci);

    let lambert: NF;
    if (shade) {
      const w0 = fetchWorldVert(ctx, localTri, 0);
      const w1 = fetchWorldVert(ctx, localTri, 1);
      const w2 = fetchWorldVert(ctx, localTri, 2);
      const faceN = normalize(
        cross(w1.sub(w0) as unknown as NV3, w2.sub(w0) as unknown as NV3),
      ) as unknown as NV3;
      const L = normalize(vec3(0.55, 0.8, 0.25)) as unknown as NV3;
      lambert = max(dot(faceN, L), 0).mul(0.85).add(0.18) as unknown as NF;
    } else {
      lambert = float(1) as unknown as NF;
    }

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

  // ---- per-frame passes --------------------------------------------------------------
  const clearVis = (renderer: Renderer): void => {
    dispatch(renderer, kVisClear);
  };
  const depth1 = (renderer: Renderer): void => {
    dispatchIndirect(renderer, kRasterDepth, cull.rasterDispatchAttr);
  };
  const depth2 = (renderer: Renderer): void => {
    dispatchIndirect(renderer, kRasterDepth2, cull.rasterDispatch2Attr);
  };
  const hwRender = (renderer: Renderer, camera: PerspectiveCamera, mat: NodeMaterial): void => {
    const prevRT = renderer.getRenderTarget();
    renderer.setRenderTarget(hwRT);
    hwMesh.material = mat;
    renderer.render(hwScene, camera);
    renderer.setRenderTarget(prevRT);
  };
  const hwDepth = (renderer: Renderer, camera: PerspectiveCamera): void => {
    dispatch(renderer, kHwArgs);
    hwRender(renderer, camera, hwDepthMat);
  };
  const payload = (renderer: Renderer, camera: PerspectiveCamera): void => {
    dispatchIndirect(renderer, kRasterPayload, cull.rasterDispatchFullAttr);
    hwRender(renderer, camera, hwPayloadMat);
  };

  const readHwCount = async (renderer: Renderer): Promise<number> => {
    const buf = await readBuffer(renderer, hwQueueAttr, 0, 4);
    return new Uint32Array(buf)[0] ?? 0;
  };
  const audit = (renderer: Renderer): void => {
    dispatch(renderer, kAudit);
  };
  const readAudit = async (renderer: Renderer): Promise<{ orphans: number; covered: number }> => {
    const buf = await readBuffer(renderer, auditAttr, 0, 8);
    const u = new Uint32Array(buf);
    return { orphans: u[0] ?? 0, covered: u[1] ?? 0 };
  };

  return { resolveScene, clearVis, depth1, depth2, hwDepth, payload, readHwCount, audit, readAudit };
}
