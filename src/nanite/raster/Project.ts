/**
 * Project.ts — the per-VERTEX projection PRE-PASS (task #76 vis-buffer rewrite, Step 2).
 *
 * WHAT: `nanProjectVerts` runs the EXACT transform + wind + clip → NDC → 1/256-px snap
 * that `nanRasterWorld1` did inline per tri-corner, ONCE per corner, into `projVertBuf`.
 * `Classify` (the rewritten world1) then reads each tri's 3 pre-projected corners by slot
 * instead of projecting them — that deletion is the register win (world1's 64-reg
 * projection floor goes).
 *
 * BIT-IDENTITY (the #1 requirement): the pass decodes the SAME 35-word per-cluster ctx the
 * world1 raster reads (ClusterCtx layout), fetches the SAME world vertex via the SAME
 * `nfetch.fetchWorldVert(ctx, localTri, v)`, and runs `projectVert()` — the VERBATIM copy of
 * world1's inline projection (`cam.vp` clip → w-guard → NDC → +1·0.5·(W,H) → ×256 round →
 * i32). So the snapped `xi/yi/dz` written here are the byte-identical values world1 computed;
 * Classify reads them back and produces the identical winding / route / queue records.
 *
 * GRANULARITY — DEDUP (task #76): each cluster reserves MAX_CLUSTER_VERTS UNIQUE-vertex slots
 * (not 765 tri-CORNER slots), addressed by `canonVertSlot`:
 *   • mesh: `vi − vBase` (dense per-cluster key). This is universal now because BuildDag.
 *     meshletizeDag re-emits each cluster's verts CONTIGUOUSLY at build time (the leaf DAGs
 *     were scattered across a 1.4M-vertex pool — measured — which is what blocked this before);
 *     vBase = indices[triStart*3] (the meshletized first corner = the min index).
 *   • terrain (isHF): kept PER-CORNER (localTri*3+corner). Its blocks are ≤128 tris ⇒ ≤384
 *     corner slots, already < the stride, so no grid-index math is needed.
 * The pass is one workgroup / cluster, one thread / tri, each projecting its 3 corners into
 * their canonical slots (shared mesh verts collapse to one slot — idempotent plain stores, no
 * atomics: every writer of a slot writes the identical projected value).
 *
 * NEAR-PLANE: world1 routes a tri to HW if ANY corner has `pv.w ≤ NEAR_EPS`. A near-crossing
 * corner's fixed-point snaps to garbage that is DISCARDED on the HW path, so we don't need its
 * xi/yi — we only need the FLAG. We stash it as a NaN-bit SENTINEL in the corner's dz word;
 * Classify ANDs `dz != SENTINEL` across the 3 corners to reproduce world1's `nearOK` exactly
 * (a raw-u32 compare, no float NaN semantics). A valid (finite) dz can never equal the NaN
 * sentinel, so accepted tris carry their exact dz unchanged.
 */

import { Fn, float, uint, vec2, vec4 } from 'three/tsl';
import { StorageBufferAttribute } from 'three/webgpu';
import type { IndirectStorageBufferAttribute } from 'three/webgpu';
import type { NB, NF, NI, NU, NV3, NV4 } from '../../gpu/TSLTypes';
import { MAX_CLUSTER_TRIS } from '../GeometryRegistry';
import { DISPATCH_ROW, QRASTER_CAP, type NaniteCam } from '../NaniteCommon';
import type { NaniteFetch, VertCtx } from '../NaniteFetch';
import {
  bcF2U,
  bcI2U,
  bcU2F,
  elemU,
  localX,
  minU,
  returnIf,
  setIndirectDispatch,
  sU32Views,
  toI,
  wgLinear,
} from '../Tsl';
import type { BufOf, UV2 } from '../Tsl';
import { CTX_STRIDE, CTX_U } from './ClusterCtx';

type U32Views = ReturnType<typeof sU32Views>;

interface ComputeKernel {
  setName(name: string): unknown;
}

// DEDUPED per-cluster reservation (task #76). Each visible cluster reserves MAX_CLUSTER_VERTS
// UNIQUE-vertex slots (not the 765 tri-CORNER slots). Sized to cover the measured max unique
// verts/cluster across every cluster type in the render set:
//   • mesh (leaf/trunk/aggregate DAGs, meshletized ⇒ contiguous per cluster): ≤ 494 unique;
//   • terrain-DAG (isHF): kept per-CORNER (localTri*3+corner) — its blocks are ≤128 tris ⇒
//     ≤ 384 corner slots, already < the stride, so no grid-index math is needed.
// 512 covers both with margin (an overflow guard clamps + the runtime check reports any
// cluster that would exceed it). projVertBuf stride shrinks 765 → 512 (~⅓ smaller).
export const MAX_CLUSTER_VERTS = 512;
export const projVertsPerCluster = (): number => MAX_CLUSTER_VERTS;
// projected-vert record = xi(i32 bits) | yi(i32 bits) | dz(f32 bits, or NEAR_SENTINEL).
export const PROJ_VERT_STRIDE = 3;

/** THE shared canonical projVertBuf slot — element index of the xi word of the DEDUPED
 *  record for (cluster item, localTri, corner). ONE definition, used by ALL THREE stages
 *  that touch projVertBuf (the projection WRITE `nanProjectVerts`, the classifier READ
 *  world1, the mid consumer `nanMidRaster`) so they address the byte-identical slot.
 *  `recCluster` = itemIdx·MAX_CLUSTER_VERTS (caller hoists it once).
 *   • terrain (isHF): per-CORNER local = localTri*3 + corner (< stride; no dedup needed).
 *   • mesh: local = vi − vBase, where vi = indices[(triStart+localTri)*3+corner] and vBase
 *     = indices[triStart*3]. After MESHLETIZE (BuildDag.meshletizeDag) each cluster's verts
 *     are CONTIGUOUS and the first corner is the min index, so vBase is the cluster base and
 *     vi−vBase ∈ [0, uniqueCount) is a dense bijection — shared verts collapse to one slot.
 *  A minU clamp keeps a rogue cluster (unique > stride) in-bounds (contained corruption, not
 *  cross-cluster) — the ProjectVerts overflow counter reports if it ever fires. */
export const canonVertSlot = (
  recCluster: NU,
  triStart: NU,
  isHF: NB,
  localTri: NU,
  corner: number,
  indices: Parameters<typeof elemU>[0],
): NU => {
  const perCorner = localTri.mul(uint(3)).add(uint(corner));
  const vi = elemU(indices, triStart.add(localTri).mul(uint(3)).add(uint(corner)));
  const vBase = elemU(indices, triStart.mul(uint(3)));
  const meshLocal = vi.sub(vBase);
  const canonLocal = minU(
    (isHF as unknown as { select(a: NU, b: NU): NU }).select(
      perCorner as unknown as NU,
      meshLocal as unknown as NU,
    ),
    uint(MAX_CLUSTER_VERTS - 1),
  );
  return recCluster.add(canonLocal).mul(uint(PROJ_VERT_STRIDE)) as unknown as NU;
};
// near-crossing (pv.w ≤ NEAR_EPS) corner marker in the dz word — a signaling-NaN bit pattern
// no finite dz can equal; compared as a raw u32 (no float NaN semantics) ⇒ robust.
export const NEAR_SENTINEL = 0x7f800001;
// Cluster-indexed reservation: the buffer holds PROJ_CLUSTER_CAP cluster slots ×
// MAX_CLUSTER_VERTS unique-vert slots × 3 u32. Sized to cover the MEASURED dense-forest
// visible-cluster peak (nanite.visClusters ≈ 150K in the canonical world config) with
// headroom, while the single-buffer alloc stays well under the ~4.29 GB maxBufferSize /
// the OOM point. 192K × 512 × 3 u32 = 1.18 GB (was 128K × 765 = 1.20 GB, but 128K was
// BELOW the 150K peak ⇒ dropped clusters; this both fixes that AND dedups). A cluster past
// the cap gets no projected verts (ProjectVerts + Classify guard the same bound ⇒ it simply
// doesn't render — surfaced overflow, not corruption).
export const PROJ_CLUSTER_CAP = Math.min(QRASTER_CAP, 192 * 1024);

export interface ProjectBundle {
  projVertAttr: StorageBufferAttribute | null;
  projVertV: U32Views | null;
  kProjectVerts: unknown | null;
  /** = MAX_CLUSTER_VERTS, the per-cluster unique-vert slot stride (Classify + Mid reuse it). */
  vertsPerCluster: number;
}

/** Build `projVertBuf` + `nanProjectVerts`. Only the world1 (singlePass = ctxPrepass) path
 *  allocates + runs it; every other raster instance passes ctxPrepass=false and gets nulls. */
export function buildProject(p: {
  /** = singlePass: the projection pre-pass only exists on the world1 path. */
  ctxPrepass: boolean;
  cam: NaniteCam;
  qRasterRO: BufOf<UV2>;
  /** the 35-word per-cluster ctx the ClusterCtx pre-pass wrote (READ here). */
  clusterCtxV: U32Views | null;
  /** the SAME fetch world1 used (variant 'both', bound to disp+wind) ⇒ identical world pos. */
  nfetch: NaniteFetch;
  /** trunk/leaf wind present (⇒ decode the wind ctx slots, exactly like world1). */
  hasWind: boolean;
  /** world1's full-range indirect args (itemCount workgroups) — SAME as the raster. */
  rasterDispatchFullAttr: IndirectStorageBufferAttribute;
  /** gpu.indices — the per-corner vertex index buffer, read by canonVertSlot to DEDUP mesh
   *  clusters (vi−vBase). SAME buffer world1's fetch reads; ProjectVerts already binds it. */
  indices: Parameters<typeof elemU>[0];
}): ProjectBundle {
  const {
    ctxPrepass,
    cam,
    qRasterRO,
    clusterCtxV,
    nfetch,
    hasWind,
    rasterDispatchFullAttr,
    indices,
  } = p;

  const vertsPerCluster = projVertsPerCluster(); // MAX_CLUSTER_TRIS*3

  if (!(ctxPrepass && clusterCtxV)) {
    return {
      projVertAttr: null,
      projVertV: null,
      kProjectVerts: null,
      vertsPerCluster,
    };
  }

  const count = PROJ_CLUSTER_CAP * vertsPerCluster * PROJ_VERT_STRIDE;
  const projVertAttr = new StorageBufferAttribute(new Uint32Array(count), 1);
  projVertAttr.name = 'nanProjVert';
  const projVertV = sU32Views(projVertAttr, count);

  const { fetchWorldVert } = nfetch;
  const W = float(cam.uW);
  const H = float(cam.uH);
  const NEAR_EPS = 1e-4;

  // projectVert — VERBATIM the world1 inline projection (NaniteRaster streaming assembly):
  //   clip = vp·(wv,1) → ok = w>NEAR_EPS → ndc = clip.xyz/clip.w → dz = ndc.z →
  //   screen = (ndc.xy+1)·0.5·(W,H) → xi/yi = round(screen·256) as i32.
  // Called ONLY here now; kept a named closure so it can be reused verbatim later.
  const projectVert = (
    wv: NV3,
  ): { xi: NI; yi: NI; dz: NF; ok: NB } => {
    const pv = cam.vp.mul(vec4(wv, 1)).toVar() as unknown as NV4;
    const ok = pv.w.greaterThan(NEAR_EPS) as unknown as NB;
    const ndc = pv.xyz.div(pv.w).toVar();
    const dz = ndc.z.toVar() as unknown as NF;
    const sv = ndc.xy.add(1).mul(0.5).mul(vec2(W, H)).toVar();
    const xi = toI(sv.x.mul(256).round()).toVar() as unknown as NI;
    const yi = toI(sv.y.mul(256).round()).toVar() as unknown as NI;
    return { xi, yi, dz, ok };
  };

  const wU = (i: NU, v: NU): void =>
    void (
      projVertV.rw.element(i) as unknown as { assign(x: NU): unknown }
    ).assign(v);

  const kn = Fn(() => {
    // SAME (itemIdx, localTri) enumeration as world1 (wgLinear cluster, localX tri) so the
    // slot mapping is identical — dispatched over rasterDispatchFull (itemCount workgroups).
    const itemIdx = wgLinear(DISPATCH_ROW).toVar();
    const itemCount = qRasterRO.element(0).x;
    returnIf(itemIdx.greaterThanEqual(itemCount));
    // Cluster-cap guard: keep writes in-bounds. Clusters past the cap get no projected verts
    // (Classify guards the same slot ⇒ they simply do not render — an overflow parity hole,
    // surfaced; the cap must exceed the frame's visible-cluster count).
    returnIf(itemIdx.greaterThanEqual(uint(PROJ_CLUSTER_CAP)));
    const localTri = localX().toVar();

    // decode the per-cluster ctx from the ClusterCtx pre-pass buffer — SAME 35-word layout
    // the world1 raster reads (CTX_U u32 + CTX_F f32-as-bits).
    const base = itemIdx.mul(uint(CTX_STRIDE)).toVar();
    const rU = (i: number): NU =>
      elemU(clusterCtxV.ro, base.add(uint(i))).toVar() as unknown as NU;
    const rF = (i: number): NF =>
      bcU2F(elemU(clusterCtxV.ro, base.add(uint(CTX_U + i)))).toVar() as unknown as NF;

    // voxel(7) clusters do NO triangle work — world1 bails before projecting; match it so
    // their corner slots stay untouched (and Classify skips them too ⇒ never read).
    returnIf(rU(10).equal(uint(7)));
    // skip clusters the cull routed to the HW instanced draw (slot 11) — world1 does the
    // same before any vertex work (the SW/HW cluster split is the permanent default).
    returnIf(rU(11).equal(uint(1)));

    const triCount = rU(3);
    returnIf(localTri.greaterThanEqual(triCount));

    const ctx = {
      isHF: rU(0).equal(uint(1)),
      isDAG: rU(1).equal(uint(1)),
      A: vec4(rF(0), rF(1), rF(2), rF(3)) as unknown as NV4,
      B: vec4(rF(4), rF(5), rF(6), rF(7)) as unknown as NV4,
      yawSc: { cy: rF(21), sy: rF(22) },
      triStart: rU(2),
      triCount,
      meshId: rU(4),
      channel: rU(5),
      twoSided: rU(9).equal(uint(1)),
      wind: hasWind
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

    // one thread projects its tri's 3 corners into slots localTri*3 + {0,1,2}. Each slot has a
    // UNIQUE writer (itemIdx, localTri, v) ⇒ plain stores, no atomics. The record the raster
    // read this frame is exactly what THIS pass wrote this frame (same itemIdx/localTri/
    // triCount guards) ⇒ no clear needed, no stale reads.
    const recCluster = itemIdx.mul(uint(vertsPerCluster)).toVar();
    for (const v of [0, 1, 2] as const) {
      const wv = fetchWorldVert(ctx, localTri, v);
      const { xi, yi, dz, ok } = projectVert(wv as unknown as NV3);
      const rb = canonVertSlot(
        recCluster,
        ctx.triStart as unknown as NU,
        ctx.isHF as unknown as NB,
        localTri,
        v,
        indices,
      ).toVar();
      wU(rb, bcI2U(xi));
      wU(rb.add(uint(1)), bcI2U(yi));
      wU(
        rb.add(uint(2)),
        (ok as unknown as { select(a: NU, b: NU): NU }).select(
          bcF2U(dz),
          uint(NEAR_SENTINEL),
        ),
      );
    }
  })().compute(QRASTER_CAP * MAX_CLUSTER_TRIS, [MAX_CLUSTER_TRIS]);
  (kn as unknown as ComputeKernel).setName('nanProjectVerts');
  // tight indirect size (itemCount workgroups) — same override the world1 raster uses.
  setIndirectDispatch(kn, rasterDispatchFullAttr);

  return { projVertAttr, projVertV, kProjectVerts: kn, vertsPerCluster };
}
