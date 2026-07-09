/**
 * Hw.ts — the HW big/near-triangle raster path (vertex pulling; the fragment stage
 * writes the SAME vis buffers as the SW path, depthWrite off — one resolve, one
 * convention).
 *
 *  • kHwArgs — builds the indirect draw args from the SW pass's hwQueue[0] append count.
 *  • buildHwMaterial — the SOUP (per-tri from hwQueue) OR the INSTANCED per-cluster draw
 *    (verts pulled from qHwRaster). Both write depth (atomicMin) / combined (packed)
 *    / world1 (depth-keyed election), gated by `pass`.
 *  • hwRender / hwRenderCluster — the render-pass wrappers (own dead full-res rgba8 target,
 *    colorWrite off; the redundant per-frame clear is dropped).
 *
 * NOTE: the per-cluster SW/HW split is the permanent default — every camera/view/shadow cull
 * supplies qHwRaster/hwClusterDraw, so the instanced cluster scene is built whenever those
 * buffers are present. They are absent ONLY on the shadow-clipmap queue (depth-only), which
 * never runs the world1 HW-cluster draw.
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
  atomicMax,
  atomicMin,
  bool,
  float,
  instanceIndex,
  screenCoordinate,
  uint,
  varyingProperty,
  vec4,
  vertexIndex,
} from 'three/tsl';
import type { NB, NF, NU, NV3, NV4 } from '../../gpu/TSLTypes';
import { CLUSTER_TRI_BITS, CLUSTER_TRI_MASK } from '../GeometryRegistry';
import type { NaniteCam } from '../NaniteCommon';
import type { NaniteFetch, VertCtx } from '../NaniteFetch';
import {
  aLoadU,
  bcF2U,
  bcU2F,
  bcU2I,
  elemU,
  minU,
  sU32Views,
  toF,
} from '../Tsl';
import type { IndirectStorageBufferAttribute } from 'three/webgpu';
import type { NaniteVisBuffers } from './VisBuffer';
import { depthKey16, depthKey24 } from './VisBuffer';
import { HW_CAP } from './Queues';
import { CTX_STRIDE, CTX_U } from './ClusterCtx';
import { PROJ_CLUSTER_CAP, canonVertSlot } from './Project';

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
  qRasterRO: { element(i: NU | number): { x: NU; y: NU } };
  vis: NaniteVisBuffers;
  hwQueueV: U32Views;
  hwDrawAttr: IndirectStorageBufferAttribute;
  hwDrawBuf: U32Views['rw'];
  /** the shipped depth-keyed election bound to the vis buffers (VisBuffer.makeElect). */
  elect: (px: NU, cand: NU, idStore: NU) => void;
  /** instanced per-cluster draw buffers — supplied by every camera/view/shadow cull (the
   *  SW/HW split is the permanent default); null only on the shadow-clipmap queue. */
  qHwRasterRO: StorageBufferNode<'uint'> | null;
  hwClusterDrawAttr: IndirectStorageBufferAttribute | null;
  /** A2 (2026-07-09 HW vertex-prepass) — Option A + class split. Non-null ONLY on the world1
   *  (singlePass) camera path: the per-cluster ctx PRE-PASS buffer (ClusterCtx). When present
   *  the `_cl` world1 draw READS the decoded 35-word ctx (`clusterCtxV[tid·CTX_STRIDE]`) instead
   *  of re-running `makeCtx` per vertex (Option A), AND is CLASS-SPLIT into two draws — a mesh
   *  (explicit) draw + a terrain (isHF) draw — each with its own single-arm fetch variant. When
   *  null (shadow/view) the single 'both' makeCtx `_cl` material is kept (dormant — never rendered
   *  off the camera path). */
  clusterCtxV: U32Views | null;
  /** trunk/leaf wind present ⇒ the explicit `_cl` draw decodes the wind ctx slots (11–20). */
  hasWind: boolean;
  /** A2 class-split fetch variants: 'explicit' (mesh) and 'terrain' (isHF) — each compiles ONLY
   *  its own fetch arm, shedding the other class's branch-union registers. */
  nfetchExplicit: NaniteFetch;
  nfetchTerrain: NaniteFetch;
  /** A2 terrain-class draw args (isHF clusters, back-filled into qHwRaster from qCap-1). */
  hwClusterDrawTerrainAttr: IndirectStorageBufferAttribute | null;
  /** A2 qHwRaster capacity — the terrain draw reads qHwRaster[hwRasterCap-1-instanceIndex]. */
  hwRasterCap: number;
  /** HW vertex-prepass (2026-07-09): the projected-vert buffer nanProjectVerts filled (the
   *  SAME records the SW classifier + Mid read). Non-null ONLY on the world1 (singlePass)
   *  path — the `_clE` mesh draw reads its verts from here instead of re-fetching world +
   *  wind + projecting (see the vertex body). */
  projVertV: U32Views | null;
  /** gpu.indices — canonVertSlot reads it for the mesh dedup key (vi−vBase). */
  indices: Parameters<typeof canonVertSlot>[5];
  /** = MAX_CLUSTER_VERTS — the per-cluster unique-vert slot stride of projVertBuf. */
  vertsPerCluster: number;
  /** ?hwproj (default ON; `?hwproj=0` = the disable-only escape): `_clE` reads projVertBuf.
   *  OFF restores the previous compute-fetch `_clE` vertex — selected at BUILD time (two
   *  closures, the other body never compiled), not a runtime shader branch. */
  hwproj: boolean;
}): HwPath {
  const {
    cam,
    width,
    height,
    nfetch,
    qRasterRO,
    vis,
    hwQueueV,
    hwDrawAttr,
    hwDrawBuf,
    elect,
    qHwRasterRO,
    hwClusterDrawAttr,
    clusterCtxV,
    hasWind,
    nfetchExplicit,
    nfetchTerrain,
    hwClusterDrawTerrainAttr,
    hwRasterCap,
    projVertV,
    indices,
    vertsPerCluster,
    hwproj,
  } = p;
  // makeCtx used by the SOUP path (per-tri) + the legacy single 'both' `_cl` fallback; the
  // per-vertex fetch is resolved PER MATERIAL below (mFetch) so a class-split draw can bind its
  // single-arm variant. (A2: the `_cl` world1 flat path drops makeCtx entirely.)
  const { makeCtx } = nfetch;

  // A2 Option A: decode the 35-word per-cluster ctx the ClusterCtx PRE-PASS wrote to global
  // memory (keyed by tid = qHwRaster[instanceIndex]) into a VertCtx — the SAME layout + bitcasts
  // the SW classifier (NaniteRaster) and the projection pre-pass (Project) already read. The
  // `_cl` world1 vertex shader consumes this instead of re-running makeCtx per vertex, so the
  // per-vertex metadata loads + gust texture samples + wind precompute all vanish (they ran ONCE
  // per cluster in the pre-pass). `withWind` is false for the terrain draw (isHF has no wind) and
  // hasWind for the explicit draw ⇒ each draw reserves only its own wind registers. By
  // construction bit-identical to makeCtx (the pre-pass stored makeCtx's own outputs).
  const decodeCtx = (tid: NU, withWind: boolean): VertCtx => {
    const cv = clusterCtxV as U32Views;
    const base = tid.mul(uint(CTX_STRIDE)).toVar();
    const rU = (i: number): NU =>
      elemU(cv.ro, base.add(uint(i))).toVar() as unknown as NU;
    const rF = (i: number): NF =>
      bcU2F(elemU(cv.ro, base.add(uint(CTX_U + i)))).toVar() as unknown as NF;
    return {
      isHF: rU(0).equal(uint(1)),
      isDAG: rU(1).equal(uint(1)),
      A: vec4(rF(0), rF(1), rF(2), rF(3)) as unknown as NV4,
      B: vec4(rF(4), rF(5), rF(6), rF(7)) as unknown as NV4,
      yawSc: { cy: rF(21), sy: rF(22) },
      triStart: rU(2),
      triCount: rU(3),
      meshId: rU(4),
      channel: rU(5),
      twoSided: rU(9).equal(uint(1)),
      wind: withWind
        ? {
            h0: rF(11),
            dirX: rF(12),
            dirY: rF(13),
            leanBase: rF(14),
            swayABase: rF(15),
            swayPhase: rF(16),
            ph: rF(17),
            branchBase: rF(18),
            flutBase: rF(19),
            swayXPhase: rF(20),
          }
        : null,
      gx: rU(6),
      gz: rU(7),
      qxw: rU(8),
      oX: rF(8),
      oZ: rF(9),
      cell: rF(10),
    } as unknown as VertCtx;
  };
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
    // A2 (2026-07-09): the class-split instanced world1 draw. When present the vertex shader
    // READS the pre-decoded per-cluster ctx (Option A) via the given single-arm `fetch` variant
    // ('explicit'|'terrain'), and — for the terrain draw — reads its cluster index from the
    // BACK of qHwRaster (reversed = qHwRaster[hwRasterCap-1-instanceIndex]). Undefined ⇒ the
    // legacy single 'both' makeCtx `_cl` material (dormant off the camera path) or the soup.
    hwOpts?: { fetch: NaniteFetch; withWind: boolean; reversed: boolean },
  ): NodeMaterial => {
    const clSfx = hwOpts ? (hwOpts.reversed ? '_clT' : '_clE') : '_cl';
    const sfx = instanced ? `${pass}${clSfx}` : pass;
    // A2 Option A: when the ctx pre-pass exists AND this is a class-split draw, read the flat
    // decoded ctx instead of makeCtx. The per-material fetch is the single-arm variant (or the
    // module 'both' fetch for the legacy single draw / soup).
    const useFlat = !!hwOpts && clusterCtxV != null;
    // HW vertex-prepass: the MESH (`_clE`, forward-indexed) draw reads projVertBuf instead of
    // the compute-fetch path — BUILD-time selection (?hwproj=0 compiles the old body instead;
    // `_clT` terrain + the soup keep the full path unconditionally).
    const hwProjRead =
      hwproj && useFlat && !!hwOpts && !hwOpts.reversed && projVertV != null;
    const mFetch = hwOpts?.fetch ?? nfetch;
    const mMakeCtx = mFetch.makeCtx;
    const mFetchWorldVertDyn = mFetch.fetchWorldVertDyn;
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
      // HW vertex stage reconstructs ONE runtime-selected corner (fetchWorldVertDyn) rather
      // than fetching all 3 and selecting — bit-identical by NaniteFetch's contract ("same
      // selected vertex by construction"), and sheds 2× index-read + decode + transform + wind.
      const fetchWorld = (ctx: VertCtx, localTri: NU): NV3 =>
        mFetchWorldVertDyn(ctx, localTri, corner);
      if (hwProjRead) {
        // ── `_clE` (mesh class) projVertBuf READER (HW vertex-prepass, 2026-07-09) ──────────
        // The mesh HW cluster's verts were ALREADY projected this frame by nanProjectVerts
        // (Project.ts no longer skips mesh HW clusters; clusterHwClass's coverage gate
        // guarantees the per-unique-vert path filled every slot). This vertex stage therefore
        // does NO world fetch, NO wind, NO vp transform and NO snap/seam block — it reads the
        // 3-u32 record (xi, yi, dz) and rebuilds clip with w = 1. The verts are pre-quantized
        // on the SW 1/256-px grid — that IS the SW↔HW seam alignment (the old seam-D snap
        // reproduced exactly this grid).
        // NEAR_SENTINEL absence is GUARANTEED by construction — clusterHwClass admits a mesh
        // cluster to this draw only when its padded sphere's min view-z clears NEAR_MARGIN,
        // which implies projectVert's ok-test for every vert — so NO sentinel compare and NO
        // real-clip fallback is compiled (the compiled fallback is what kept the previous
        // Phase-2 attempt at the 80-register/80-byte-spill ceiling).
        // w = 1 makes HW depth/varying interpolation SCREEN-LINEAR — exactly the SW
        // scanline's cz metric (dz interpolated linearly in screen space) — INTENTIONAL: the
        // SW and HW paths now agree bit-for-bit on the depth metric, the leading fix for the
        // long-standing near-camera depth-flicker (SW screen-linear vs HW perspective-correct
        // mismatch). The FS depth key stays depthKey24(vZ/vW) = depthKey24(dz), identical to SW.
        const localTri = (vertexIndex.div(3) as unknown as NU).toVar();
        const tid = elemU(
          qHwRasterRO as StorageBufferNode<'uint'>,
          instanceIndex as unknown as NU,
        ).toVar();
        const payload = tid.shiftLeft(uint(CLUSTER_TRI_BITS)).bitOr(localTri).toVar();
        // slim ctx read — only triStart (slot 2) + triCount (slot 3); no full decodeCtx.
        const cv = clusterCtxV as U32Views;
        const cBase = tid.mul(uint(CTX_STRIDE)).toVar();
        const triStart = elemU(cv.ro, cBase.add(uint(2))).toVar();
        const triCount = elemU(cv.ro, cBase.add(uint(3))).toVar();
        const clip = vec4(0, 0, 2, 1).toVar();
        // tail-degenerate guard (localTri ≥ triCount ⇒ degenerate clip) + the projVertBuf
        // cluster-cap guard: a cluster past PROJ_CLUSTER_CAP has no projected verts (the
        // projection + SW classifier guard the same bound ⇒ it doesn't render anywhere —
        // surfaced overflow, not garbage pixels).
        If(
          localTri
            .lessThan(triCount)
            .and(tid.lessThan(uint(PROJ_CLUSTER_CAP))),
          () => {
            // THE shared slot function (Project.ts) — mesh key vi−vBase; isHF is
            // constant-false by the class partition (this draw receives mesh clusters only).
            const rb = canonVertSlot(
              tid.mul(uint(vertsPerCluster)) as unknown as NU,
              triStart as unknown as NU,
              bool(false) as unknown as NB,
              localTri,
              corner,
              indices,
            ).toVar();
            const pvv = projVertV as U32Views;
            const xi = toF(bcU2I(elemU(pvv.ro, rb)));
            const yi = toF(bcU2I(elemU(pvv.ro, rb.add(uint(1)))));
            const dz = bcU2F(elemU(pvv.ro, rb.add(uint(2))));
            // EXACT inverse of projectVert's screen mapping (Project.ts): forward is
            //   screen = (ndc.xy + 1)·0.5·(W,H);  xi/yi = round(screen·256)
            // (no half-pixel offset, same for x and y — the seam-D block inverted the same
            // way), so:  ndc.x = (xi/256)/W·2 − 1,  ndc.y = (yi/256)/H·2 − 1,  z = dz, w = 1.
            const W = float(cam.uW);
            const H = float(cam.uH);
            const ndcX = xi.div(256).div(W).mul(2).sub(1);
            const ndcY = yi.div(256).div(H).mul(2).sub(1);
            clip.assign(vec4(ndcX, ndcY, dz, 1) as unknown as NV4);
          },
        );
        setVaryings(payload as unknown as NU, clip as unknown as NV4);
        return clip;
      }
      if (instanced) {
        // ?clhw INSTANCED cluster draw: one instance per HW cluster. qHwRaster[instanceIndex]
        // = the cluster's qRaster INDEX (tid). localTri = vertexIndex/3 (partial-cluster tail
        // clips out). payload = tid<<bits|localTri EXACTLY like the soup ⇒ resolve unchanged.
        const localTri = (vertexIndex.div(3) as unknown as NU).toVar();
        // A2: the terrain draw reads its cluster index from the BACK of the shared qHwRaster
        // (partition back-fills isHF clusters from qCap-1); the mesh / legacy draw reads forward.
        const instIdx = hwOpts?.reversed
          ? (uint(hwRasterCap - 1).sub(instanceIndex) as unknown as NU)
          : (instanceIndex as unknown as NU);
        const tid = elemU(
          qHwRasterRO as StorageBufferNode<'uint'>,
          instIdx,
        ).toVar();
        const payload = tid.shiftLeft(uint(CLUSTER_TRI_BITS)).bitOr(localTri).toVar();
        // A2 Option A: read the pre-decoded 35-word ctx (no per-vertex makeCtx / gust samples /
        // wind precompute). The legacy single 'both' draw (useFlat false) keeps makeCtx — which
        // needs the qRaster item for (instId, ci); the flat path needs neither (ctx carries A/B),
        // so it also sheds the qRaster binding from this pipeline.
        const ctx = useFlat
          ? decodeCtx(tid, hwOpts!.withWind)
          : (() => {
              const item = qRasterRO.element(tid.add(uint(1)));
              const instId = item.x.toVar();
              const ci = item.y.toVar();
              return mMakeCtx(instId, ci);
            })();
        // partial-cluster tail: fixed MAX_CLUSTER_TRIS*3 verts over-cover a <triCount cluster.
        // Tail verts (localTri >= ctx.triCount) emit the degenerate clipped point (z/w=2)
        // DIRECTLY — the whole fetch + transform + seam-snap runs only under the If, never for
        // the tail. Divergence is warp-contiguous at the cluster tail. The degenerate sentinel
        // (vec4(0,0,2,1)) is exactly the value the old branchless collapse selected.
        const clip = vec4(0, 0, 2, 1).toVar();
        If(localTri.lessThan(ctx.triCount), () => {
          const world = fetchWorld(ctx, localTri);
          const c = cam.vp.mul(vec4(world, 1)).toVar();
          // D (?clhw seam): snap c.xy onto the SAME 1/256-px screen grid the SW
          // rasterizer quantizes to (s = (ndc+1)·0.5·(W,H) then ×256 round, ~:1175).
          // A shared SW↔HW cluster edge then lands on identical subpixel positions, so
          // no pixel near the seam is dropped by both paths. This is the exact inverse
          // of the SW screen transform with the quantize in the middle; the Y-up↔Y-down
          // framebuffer flip preserves the grid (H integer ⇒ H·256 integer). Per-component
          // scalar rounds match the SW path exactly.
          const W = float(cam.uW);
          const H = float(cam.uH);
          const sx = c.x.div(c.w).add(1).mul(0.5).mul(W).mul(256).round().div(256);
          const sy = c.y.div(c.w).add(1).mul(0.5).mul(H).mul(256).round().div(256);
          c.assign(
            vec4(
              sx.div(W).mul(2).sub(1).mul(c.w),
              sy.div(H).mul(2).sub(1).mul(c.w),
              c.z,
              c.w,
            ) as unknown as NV4,
          );
          clip.assign(c as unknown as NV4);
        });
        setVaryings(payload as unknown as NU, clip as unknown as NV4);
        return clip;
      }
      // SOUP path (non-instanced material): per-tri from hwQueue.
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

  // The INSTANCED per-cluster HW draw — one instance per big/near cluster, drawn
  // MAX_CLUSTER_TRIS*3 verts each (partial-cluster tail clips out). Own geometry + scene;
  // shares hwRT + the world1 election (buildHwMaterial('world1', instanced=true)). Rendered
  // in world1() right after the soup hwRender. Built whenever the cull supplied the queue
  // (absent only on the shadow-clipmap queue, which never runs the world1 HW-cluster draw).
  const hwClusterScene = hwClusterDrawAttr ? new Scene() : null;
  const clusterGeom = (attr: IndirectStorageBufferAttribute): BufferGeometry => {
    const g = new BufferGeometry();
    g.setAttribute('position', new Float32BufferAttribute(new Float32Array(3), 3));
    g.setIndirect(attr, 0);
    g.boundingSphere = new Sphere(new Vector3(), Number.POSITIVE_INFINITY);
    return g;
  };
  const addClusterMesh = (
    attr: IndirectStorageBufferAttribute,
    mat: NodeMaterial,
  ): void => {
    const m = new Mesh(clusterGeom(attr), mat);
    m.frustumCulled = false;
    (hwClusterScene as Scene).add(m);
  };
  if (hwClusterScene && hwClusterDrawAttr) {
    if (clusterCtxV != null) {
      // A2 (2026-07-09): CLASS-SPLIT `_cl` world1 draw — a MESH (explicit, forward-indexed)
      // draw + a TERRAIN (isHF, back-indexed) draw, each reading the flat pre-decoded ctx
      // (Option A) through its own single-arm fetch variant so it sheds the other class's
      // fetch-union registers. Both write the SAME vis election ⇒ resolve unchanged; the cull
      // partition guarantees each HW cluster lands in exactly one draw (disjoint sets).
      addClusterMesh(
        hwClusterDrawAttr,
        buildHwMaterial('world1', true, {
          fetch: nfetchExplicit,
          withWind: hasWind,
          reversed: false,
        }),
      );
      if (hwClusterDrawTerrainAttr) {
        addClusterMesh(
          hwClusterDrawTerrainAttr,
          buildHwMaterial('world1', true, {
            fetch: nfetchTerrain,
            withWind: false,
            reversed: true,
          }),
        );
      }
    } else {
      // Non-world1 (shadow/view) path: no ctx pre-pass ⇒ keep the single 'both' makeCtx `_cl`
      // material (dormant — hwRenderCluster is only invoked from the camera world1()).
      addClusterMesh(hwClusterDrawAttr, buildHwMaterial('world1', true));
    }
  }
  // The HW pass renders into this dead full-res rgba8 (colorWrite=false -> never read).
  // It stays full-res unconditionally: r184 derives the render-pass viewport from
  // RenderTarget.viewport (= texture size), so shrinking it would clip HW coverage and
  // starve the vis buffers. Only the per-frame CLEAR is dropped (see hwRender).
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
    // Skip the per-frame full-res CLEAR of the dead rgba8 color target. With
    // autoClear=false the backend uses loadOp=Load (Background.js:209-217 -> the
    // descriptor's loadOp becomes Load not Clear). The HW fragment has colorWrite=false,
    // so it never writes the target; nothing downstream reads it; the only effect is the
    // clear no longer runs. The viewport/coverage (full-res hwRT) and every vis-buffer
    // write are unchanged -> byte-identical. hwScene has no .background, so forceClear
    // stays false and autoClear=false is honored. RESTORED immediately after the render.
    const prevAutoClear = renderer.autoClear;
    renderer.autoClear = false;
    renderer.render(hwScene, camera);
    renderer.autoClear = prevAutoClear;
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
