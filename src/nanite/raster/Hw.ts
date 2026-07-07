/**
 * Hw.ts — the HW big/near-triangle raster path (vertex pulling; the fragment stage
 * writes the SAME vis buffers as the SW path, depthWrite off — one resolve, one
 * convention).
 *
 *  • kHwArgs — builds the indirect draw args from the SW pass's hwQueue[0] append count.
 *  • buildHwMaterial — the SOUP (per-tri from hwQueue) OR the ?clhw INSTANCED per-cluster
 *    draw (verts pulled from qHwRaster). Both write depth (atomicMin) / combined (packed)
 *    / world1 (depth-keyed election), gated by `pass`.
 *  • hwRender / hwRenderCluster — the render-pass wrappers (own dead full-res rgba8 target,
 *    colorWrite off; ?hwrt=0 drops the redundant per-frame clear).
 *
 * NOTE: `clhw` is preserved exactly as computed by the caller (it defaults OFF — the cull
 * only supplies qHwRaster/hwClusterDraw under ?clhw=1). The instanced cluster scene is
 * therefore built only when the cull provided the queue, exactly as before.
 */

import { DoubleSide, Mesh, Scene, Vector3 } from 'three';
import {
  BufferGeometry,
  Float32BufferAttribute,
  RenderTarget,
  Sphere,
} from 'three';
import type { PerspectiveCamera } from 'three';
import {
  NodeMaterial,
  type Renderer,
  type StorageBufferNode,
} from 'three/webgpu';
import {
  Fn,
  If,
  atomicAdd,
  atomicMax,
  atomicMin,
  float,
  instanceIndex,
  screenCoordinate,
  uint,
  varyingProperty,
  vec4,
  vertexIndex,
} from 'three/tsl';
import type { NF, NU, NV3, NV4 } from '../../gpu/TSLTypes';
import { CLUSTER_TRI_BITS, CLUSTER_TRI_MASK } from '../GeometryRegistry';
import type { NaniteCam } from '../NaniteCommon';
import type { NaniteFetch, VertCtx } from '../NaniteFetch';
import {
  aLoadU,
  bcF2U,
  elemU,
  minU,
  sU32Views,
  toF,
} from '../Tsl';
import type { IndirectStorageBufferAttribute } from 'three/webgpu';
import type { NaniteVisBuffers } from './VisBuffer';
import { depthKey16, depthKey24 } from './VisBuffer';
import { HW_CAP } from './Queues';

type U32Views = ReturnType<typeof sU32Views>;

interface ComputeKernel {
  setName(name: string): unknown;
}

export interface HwPath {
  kHwArgs: unknown;
  hwDepthMat: NodeMaterial;
  hwCombinedMat: NodeMaterial;
  hwWorld1Mat: NodeMaterial;
  hwRT: RenderTarget;
  hwRender(renderer: Renderer, camera: PerspectiveCamera, mat: NodeMaterial): void;
  hwRenderCluster(renderer: Renderer, camera: PerspectiveCamera): void;
}

export function buildHw(p: {
  cam: NaniteCam;
  width: number;
  height: number;
  nfetch: NaniteFetch;
  /** M2l: HW vertex stage reconstructs ONE runtime-selected corner (?hw1fetch=1). */
  hw1fetch: boolean;
  qRasterRO: { element(i: NU | number): { x: NU; y: NU } };
  vis: NaniteVisBuffers;
  hwQueueV: U32Views;
  hwDrawAttr: IndirectStorageBufferAttribute;
  hwDrawBuf: U32Views['rw'];
  /** the shipped depth-keyed election bound to the vis buffers (VisBuffer.makeElect). */
  elect: (px: NU, cand: NU, idStore: NU) => void;
  scar: boolean;
  scarEl: (i: number) => ReturnType<U32Views['atomic']['element']>;
  /** ?hwrt=1 restores the per-frame full-res clear of the dead color target (A/B control). */
  hwrt: boolean;
  /** ?clhw: instanced per-cluster draw (built only when the cull supplied the queue). */
  clhw: boolean;
  qHwRasterRO: StorageBufferNode<'uint'> | null;
  hwClusterDrawAttr: IndirectStorageBufferAttribute | null;
}): HwPath {
  const {
    cam,
    width,
    height,
    nfetch,
    hw1fetch,
    qRasterRO,
    vis,
    hwQueueV,
    hwDrawAttr,
    hwDrawBuf,
    elect,
    scar,
    scarEl,
    hwrt,
    clhw,
    qHwRasterRO,
    hwClusterDrawAttr,
  } = p;
  const { makeCtx, fetchWorldVert, fetchWorldVertDyn } = nfetch;
  const visDepthV = vis.depthV;
  const visPayloadV = vis.payloadV;
  const visBV = vis.visBV;

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
  hwGeometry.setAttribute(
    'position',
    new Float32BufferAttribute(new Float32Array(3), 3),
  );
  hwGeometry.setIndirect(hwDrawAttr, 0);
  hwGeometry.boundingSphere = new Sphere(
    new Vector3(),
    Number.POSITIVE_INFINITY,
  );

  const buildHwMaterial = (
    pass: 'depth' | 'combined' | 'world1',
    // ?clhw INSTANCED per-cluster draw: one instance per HW cluster, verts pulled from
    // qHwRaster instead of the per-tri soup queue. Same FS election ⇒ resolve unchanged.
    instanced = false,
  ): NodeMaterial => {
    const sfx = instanced ? `${pass}_cl` : pass;
    const mat = new NodeMaterial();
    mat.name = `nanRaster_${sfx}`;
    const vPayLo = varyingProperty('float', `nanPayLo_${sfx}`) as unknown as NF;
    const vPayHi = varyingProperty('float', `nanPayHi_${sfx}`) as unknown as NF;
    const vZ = varyingProperty('float', `nanZ_${sfx}`) as unknown as NF;
    const vW = varyingProperty('float', `nanW_${sfx}`) as unknown as NF;

    mat.vertexNode = Fn(() => {
      const corner = vertexIndex.mod(3) as unknown as NU;
      const setVaryings = (payload: NU, clip: NV4): void => {
        (vPayLo as unknown as { assign: (v: unknown) => void }).assign(
          toF(payload.bitAnd(uint(0xffff))),
        );
        (vPayHi as unknown as { assign: (v: unknown) => void }).assign(
          toF(payload.shiftRight(uint(16))),
        );
        (vZ as unknown as { assign: (v: unknown) => void }).assign(clip.z);
        (vW as unknown as { assign: (v: unknown) => void }).assign(clip.w);
      };
      const fetchWorld = (ctx: VertCtx, localTri: NU): NV3 => {
        if (hw1fetch) return fetchWorldVertDyn(ctx, localTri, corner);
        const w0 = fetchWorldVert(ctx, localTri, 0);
        const w1 = fetchWorldVert(ctx, localTri, 1);
        const w2 = fetchWorldVert(ctx, localTri, 2);
        return corner
          .equal(uint(1))
          .select(w1, corner.equal(uint(2)).select(w2, w0)) as unknown as NV3;
      };
      if (instanced) {
        // ?clhw INSTANCED cluster draw: one instance per HW cluster. qHwRaster[instanceIndex]
        // = the cluster's qRaster INDEX (tid). localTri = vertexIndex/3 (partial-cluster tail
        // clips out). payload = tid<<bits|localTri EXACTLY like the soup ⇒ resolve unchanged.
        const localTri = (vertexIndex.div(3) as unknown as NU).toVar();
        const tid = elemU(
          qHwRasterRO as StorageBufferNode<'uint'>,
          instanceIndex,
        ).toVar();
        const item = qRasterRO.element(tid.add(uint(1)));
        const instId = item.x.toVar();
        const ci = item.y.toVar();
        const payload = tid.shiftLeft(uint(CLUSTER_TRI_BITS)).bitOr(localTri).toVar();
        const ctx = makeCtx(instId, ci);
        const world = fetchWorld(ctx, localTri);
        const clip = cam.vp.mul(vec4(world, 1)).toVar();
        // D (?clhw seam): snap clip.xy onto the SAME 1/256-px screen grid the SW
        // rasterizer quantizes to (s = (ndc+1)·0.5·(W,H) then ×256 round, ~:1175).
        // A shared SW↔HW cluster edge then lands on identical subpixel positions, so
        // no pixel near the seam is dropped by both paths. This is the exact inverse
        // of the SW screen transform with the quantize in the middle; the Y-up↔Y-down
        // framebuffer flip preserves the grid (H integer ⇒ H·256 integer). Per-component
        // scalar rounds match the SW path exactly.
        {
          const W = float(cam.uW);
          const H = float(cam.uH);
          const sx = clip.x.div(clip.w).add(1).mul(0.5).mul(W).mul(256).round().div(256);
          const sy = clip.y.div(clip.w).add(1).mul(0.5).mul(H).mul(256).round().div(256);
          clip.assign(
            vec4(
              sx.div(W).mul(2).sub(1).mul(clip.w),
              sy.div(H).mul(2).sub(1).mul(clip.w),
              clip.z,
              clip.w,
            ) as unknown as NV4,
          );
        }
        // partial-cluster tail: fixed MAX_CLUSTER_TRIS*3 verts over-cover a <128-tri cluster
        // ⇒ collapse those verts to a clipped point (z/w=2).
        clip.assign(
          (
            localTri.lessThan(ctx.triCount) as unknown as {
              select(a: NV4, b: NV4): NV4;
            }
          ).select(clip as unknown as NV4, vec4(0, 0, 2, 1) as unknown as NV4),
        );
        setVaryings(payload as unknown as NU, clip as unknown as NV4);
        return clip;
      }
      // SOUP (default, ?clhw off — byte-identical to the pre-clhw shader): per-tri from hwQueue.
      const triIndex = vertexIndex.div(3) as unknown as NU;
      const base = triIndex.mul(uint(2)).add(uint(1));
      const payload = elemU(hwQueueV.ro, base).toVar();
      const instId = elemU(hwQueueV.ro, base.add(uint(1))).toVar();
      const itemIdx = payload.shiftRight(uint(CLUSTER_TRI_BITS));
      const localTri = payload.bitAnd(uint(CLUSTER_TRI_MASK));
      const item = qRasterRO.element(itemIdx.add(uint(1)));
      const ci = item.y.toVar();
      const ctx = makeCtx(instId, ci);
      const world = fetchWorld(ctx, localTri as unknown as NU);
      const clip = cam.vp.mul(vec4(world, 1)).toVar();
      setVaryings(payload as unknown as NU, clip as unknown as NV4);
      return clip;
    })() as unknown as typeof mat.vertexNode;

    mat.fragmentNode = Fn(() => {
      const z = vZ.div(vW).toVar();
      const pay = uint(vPayLo.round())
        .bitOr(uint(vPayHi.round()).shiftLeft(uint(16)))
        .toVar();
      const fy = float(cam.uH).sub(screenCoordinate.y);
      const px = uint(fy).mul(uint(cam.uW)).add(uint(screenCoordinate.x));
      If(z.greaterThanEqual(0).and(z.lessThanEqual(1)), () => {
        const bits = bcF2U(z as unknown as NF);
        if (pass === 'depth') {
          atomicMin(visDepthV.atomic.element(px), bits);
        } else if (pass === 'combined') {
          // RACE-FREE packed (mirrors the SW combined path): id packed by depth-keyed
          // atomicMax into visA(payload)+visB, no separate depthV.
          const myDk = depthKey16(z as unknown as NF).toVar();
          const stored = aLoadU(visPayloadV.atomic.element(px));
          If(myDk.greaterThan(stored.shiftRight(uint(16))), () => {
            const dk = myDk.shiftLeft(uint(16));
            atomicMax(
              visPayloadV.atomic.element(px),
              dk.bitOr(pay.bitAnd(uint(0xffff))),
            );
            atomicMax(
              visBV.atomic.element(px),
              dk.bitOr(pay.shiftRight(uint(16)).bitAnd(uint(0xffff))),
            );
          });
        } else if (pass === 'world1') {
          // 0a SCAR (?scar=1): count this HW fragment into the whole-frame total [2] so
          // the band-share denominator includes the large/near tris the HW vertex-pull
          // path carries (terrain, trunks, near-plane-crossing leaf edges). The band
          // NUMERATOR [0] stays SW-only by design — mid/far foliage leaves rasterize on
          // the SW path; the HW path holds non-band near geometry. No band classify here.
          if (scar) atomicAdd(scarEl(2), uint(1));
          // PERF-VB4 single-pass WORLD (mirrors the SW world1 path): a depth24-keyed
          // atomicMax election whose WINNER write-stores the full id + its exact depth.
          const cand = depthKey24(z as unknown as NF)
            .shiftLeft(uint(8))
            .bitOr(pay.bitAnd(uint(0xff)))
            .toVar();
          // Mirror the SW world1 election's relaxed-load guard (VisBuffer.makeElect): the
          // atomicLoad guard skips the RMW for fragments behind the current front (the bulk
          // of HW overdraw). Bit-identical to the shipped inline election.
          elect(px, cand as unknown as NU, pay as unknown as NU);
        }
      });
      return vec4(0, 0, 0, 0);
    })() as unknown as typeof mat.fragmentNode;
    mat.depthTest = false;
    mat.depthWrite = false;
    mat.colorWrite = false;
    mat.fog = false;
    mat.lights = false;
    // N9-C2: don't HW back-face-cull. Two-sided leaf tris that cross the near plane
    // or oversize the i32 SW path land here with a single winding (the reversed dup
    // is gone), so culling would hole them. Safe for opaque solids: the nearer face
    // wins atomicMin and only the depth-match writes payload, so back-faces never win.
    mat.side = DoubleSide;
    return mat;
  };

  const hwDepthMat = buildHwMaterial('depth');
  const hwCombinedMat = buildHwMaterial('combined');
  const hwWorld1Mat = buildHwMaterial('world1');
  const hwScene = new Scene();
  const hwMesh = new Mesh(hwGeometry, hwDepthMat);
  hwMesh.frustumCulled = false;
  hwScene.add(hwMesh);

  // ?clhw: the INSTANCED per-cluster HW draw — one instance per big/near cluster, drawn
  // MAX_CLUSTER_TRIS*3 verts each (partial-cluster tail clips out). Own geometry + scene;
  // shares hwRT + the world1 election (buildHwMaterial('world1', instanced=true)). Rendered
  // in world1() right after the soup hwRender. Built only when the cull supplied the queue.
  const hwClusterScene = clhw ? new Scene() : null;
  if (clhw && hwClusterScene && hwClusterDrawAttr) {
    const g = new BufferGeometry();
    g.setAttribute('position', new Float32BufferAttribute(new Float32Array(3), 3));
    g.setIndirect(hwClusterDrawAttr, 0);
    g.boundingSphere = new Sphere(new Vector3(), Number.POSITIVE_INFINITY);
    const m = new Mesh(g, buildHwMaterial('world1', true));
    m.frustumCulled = false;
    hwClusterScene.add(m);
  }
  // The HW pass renders into this dead full-res rgba8 (colorWrite=false -> never read).
  // It stays full-res unconditionally: r184 derives the render-pass viewport from
  // RenderTarget.viewport (= texture size), so shrinking it would clip HW coverage and
  // starve the vis buffers. ?hwrt=0 instead drops only the per-frame CLEAR (see hwRender).
  const hwRT = new RenderTarget(width, height, { depthBuffer: false });
  hwRT.texture.name = 'nanHwPass';

  const hwRender = (
    renderer: Renderer,
    camera: PerspectiveCamera,
    mat: NodeMaterial,
  ): void => {
    const prevRT = renderer.getRenderTarget();
    renderer.setRenderTarget(hwRT);
    hwMesh.material = mat;
    // ?hwrt=0: skip the per-frame full-res CLEAR of the dead rgba8 color target. With
    // autoClear=false the backend uses loadOp=Load (Background.js:209-217 -> the
    // descriptor's loadOp becomes Load not Clear). The HW fragment has colorWrite=false,
    // so it never writes the target; nothing downstream reads it; the only effect is the
    // clear no longer runs. The viewport/coverage (full-res hwRT) and every vis-buffer
    // write are unchanged -> byte-identical. hwScene has no .background, so forceClear
    // stays false and autoClear=false is honored. RESTORED immediately after the render.
    const prevAutoClear = renderer.autoClear;
    if (!hwrt) renderer.autoClear = false;
    renderer.render(hwScene, camera);
    if (!hwrt) renderer.autoClear = prevAutoClear;
    renderer.setRenderTarget(prevRT);
  };
  // ?clhw: the instanced per-cluster HW draw into the same hwRT/vis buffers (never clears —
  // the election is atomicMax, order-independent; it just adds the big/near cluster winners).
  const hwRenderCluster = (renderer: Renderer, camera: PerspectiveCamera): void => {
    if (!hwClusterScene) return;
    const prevRT = renderer.getRenderTarget();
    renderer.setRenderTarget(hwRT);
    const prevAutoClear = renderer.autoClear;
    renderer.autoClear = false;
    renderer.render(hwClusterScene, camera);
    renderer.autoClear = prevAutoClear;
    renderer.setRenderTarget(prevRT);
  };

  return {
    kHwArgs,
    hwDepthMat,
    hwCombinedMat,
    hwWorld1Mat,
    hwRT,
    hwRender,
    hwRenderCluster,
  };
}
