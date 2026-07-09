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
 * DISPATCH GRANULARITY (task #76 Lever 1): one workgroup / cluster.
 *   • mesh, vcompact-covered (count>0): PER-UNIQUE-VERT — threads STRIDE over the cluster's
 *     dense [vBase, vBase+uniqueCount) range (vcompact[ci] = (vMin, count)) and project each
 *     unique vert ONCE via fetchWorldVertByIndex into slot recCluster + (vi−vBase). This is the
 *     mem-bound win: the per-corner dispatch re-fetched/re-transformed/re-wrote each SHARED vert
 *     ~3–6× into the SAME slot (idempotent — dedup was STORAGE-only); now once ⇒ ~½–⅙ the traffic.
 *   • mesh, NOT covered (tooWide / streamed-terrain / count=0) + terrain (isHF): one thread / tri,
 *     each projecting its 3 corners per-CORNER into their canonical slots (shared mesh verts
 *     collapse to one slot — idempotent plain stores, no atomics: every writer of a slot writes
 *     the identical projected value). The mesh>0 path writes byte-identical records to this.
 *
 * NEAR-PLANE: world1 routes a tri to HW if ANY corner has `pv.w ≤ NEAR_EPS`. A near-crossing
 * corner's fixed-point snaps to garbage that is DISCARDED on the HW path, so we don't need its
 * xi/yi — we only need the FLAG. We stash it as a NaN-bit SENTINEL in the corner's dz word;
 * Classify ANDs `dz != SENTINEL` across the 3 corners to reproduce world1's `nearOK` exactly
 * (a raw-u32 compare, no float NaN semantics). A valid (finite) dz can never equal the NaN
 * sentinel, so accepted tris carry their exact dz unchanged.
 */

import {
  Fn,
  If,
  float,
  uint,
  vec2,
  vec4,
  workgroupArray,
  workgroupBarrier,
} from 'three/tsl';
import { StorageBufferAttribute } from 'three/webgpu';
import type { IndirectStorageBufferAttribute } from 'three/webgpu';
import type { NB, NF, NI, NU, NV3, NV4 } from '../../gpu/TSLTypes';
import { MAX_CLUSTER_TRIS, VCACHE_VERTS } from '../GeometryRegistry';
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
import { HWPROJ } from '../NaniteHwClass';

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
const MAX_CLUSTER_VERTS = 512;
export const projVertsPerCluster = (): number => MAX_CLUSTER_VERTS;
// projected-vert record = xi(i32 bits) | yi(i32 bits) | dz(f32 bits, or NEAR_SENTINEL).
const PROJ_VERT_STRIDE = 3;

/** THE shared canonical projVertBuf slot — element index of the xi word of the DEDUPED
 *  record for (cluster item, localTri, corner). ONE definition, used by ALL THREE stages
 *  that touch projVertBuf (the projection WRITE `nanProjectVerts`, the classifier READ
 *  world1, the mid consumer `nanMidRaster`) so they address the byte-identical slot.
 *   • terrain (isHF): per-CORNER local = localTri*3 + corner (< reservation; no dedup needed).
 *   • mesh: local = vi − vBase, where vi = indices[(triStart+localTri)*3+corner] and vBase
 *     = indices[triStart*3]. After MESHLETIZE (BuildDag.meshletizeDag) each cluster's verts
 *     are CONTIGUOUS and the first corner is the min index, so vBase is the cluster base and
 *     vi−vBase ∈ [0, uniqueCount) is a dense bijection — shared verts collapse to one slot.
 *  `recCluster` = itemIdx·MAX_CLUSTER_VERTS (the flat per-cluster base; the caller hoists it
 *  once). The minU clamp uses the fixed MAX_CLUSTER_VERTS-1 moat so a rogue canonLocal can
 *  never spill into the next cluster's fixed region. `corner` may be a compile-time 0|1|2
 *  (the compute consumers unroll corners) or a runtime NU (the HW `_clE` vertex stage derives
 *  it from vertexIndex) — identical math either way. */
export const canonVertSlot = (
  recCluster: NU,
  triStart: NU,
  isHF: NB,
  localTri: NU,
  corner: number | NU,
  indices: Parameters<typeof elemU>[0],
): NU => {
  const cornerU = (typeof corner === 'number' ? uint(corner) : corner) as NU;
  const perCorner = localTri.mul(uint(3)).add(cornerU);
  const vi = elemU(indices, triStart.add(localTri).mul(uint(3)).add(cornerU));
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
// MAX_CLUSTER_VERTS unique-vert slots × 3 u32. Sized to the MEASURED dense-forest
// visible-cluster peak with headroom (a cluster past the cap gets no projected verts —
// ProjectVerts + Classify guard the same bound ⇒ it simply doesn't render; surfaced
// overflow, not corruption).
// CENSUS 2026-07-09 POST crown-LOD ladder (docs/tasks/2026-07-09/census-post-ladder.json,
// 898-sample flythrough): peak nanite.visClusters = 66,827 at the dense-forest look-across
// (t=0.373, ?cam=-582.1,302.4,1006.1,2.5692,-0.0077) — the stale "≈150K" that forced the
// old 192K cap was the PRE-ladder number; the crown 4→6-rung ladder more than halved the
// visible-cluster count. 128Ki = 131,072 = 1.96× the fresh peak.
// projVertBuf = cap × 512 × 3 u32 × 4 B = cap × 6144 B: 192Ki → 1.18 GB, 128Ki → 0.805 GB
// (−384 MiB / −402.7 MB, the single biggest memory item in the pipeline). Alloc stays far
// under the ~4.29 GB maxBufferSize. QRASTER_CAP (world = 1M) still caps visClusters upstream.
export const PROJ_CLUSTER_CAP = Math.min(QRASTER_CAP, 128 * 1024);

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
  /** gpu.vcompact — per-cluster (vMin, count) built in GeometryRegistry.populateVCompact
   *  (unconditional at upload; consumed here for the per-unique-vert dispatch).
   *  count>0 ⇒ the cluster's global vertex indices span the DENSE range [vMin, vMin+count)
   *  ⇒ this pass projects each UNIQUE vert ONCE (per-unique-vert dispatch, task #76 Lever 1)
   *  instead of ~3–6× per shared corner. count==0 (tooWide / window-grid / streamed-terrain)
   *  ⇒ fall back to the per-corner projection so no cluster is under-projected. */
  vcompact: Parameters<typeof elemU>[0];
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
    vcompact,
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

  // FLAT layout: PROJ_CLUSTER_CAP cluster slots × MAX_CLUSTER_VERTS unique-vert slots × 3 u32.
  // 128Ki × 512 × 3 u32 = 0.805 GB — each cluster owns a fixed itemIdx·512 region (no compaction).
  const count = PROJ_CLUSTER_CAP * vertsPerCluster * PROJ_VERT_STRIDE;
  const projVertAttr = new StorageBufferAttribute(new Uint32Array(count), 1);
  projVertAttr.name = 'nanProjVert';
  const projVertV = sU32Views(projVertAttr, count);

  const { fetchWorldVert, fetchWorldVertByIndex } = nfetch;
  const W = float(cam.uW);
  const H = float(cam.uH);
  const NEAR_EPS = 1e-4;

  // projectVert — the world1 inline projection (NaniteRaster streaming assembly):
  //   clip = vp·(wv,1) → ok gate → ndc = clip.xyz/clip.w → dz = ndc.z →
  //   screen = (ndc.xy+1)·0.5·(W,H) → xi/yi = round(screen·256) as i32.
  // ok = w>NEAR_EPS AND the fixed-point snap does NOT i32-SATURATE (FIX-FIRST audit
  // Finding A, 2026-07-09): a camera-plane-grazing corner (w>EPS but |ndc| huge) used
  // to pass ok with xi/yi clamped to ±2^31 — the SW classifier then routed the tri to
  // the hwQueue via the not-smallEnough escape (correct), but the Phase-2 `_cl` HOT
  // path has no oversize escape and rendered the bent corner. Folding saturation into
  // ok ⇒ the corner carries NEAR_SENTINEL ⇒ SW: nearOK=false → hwQueue soup (real
  // clip — the same destination smallEnough sent it, minus dead setup work); `_cl`:
  // the per-tri fallback (real clip). 1e8 < 2^31 leaves ~21× margin over any bbox
  // the SW path could accept while catching all saturating values.
  const projectVert = (
    wv: NV3,
  ): { xi: NI; yi: NI; dz: NF; ok: NB } => {
    const pv = cam.vp.mul(vec4(wv, 1)).toVar() as unknown as NV4;
    const ndc = pv.xyz.div(pv.w).toVar();
    const dz = ndc.z.toVar() as unknown as NF;
    const sv = ndc.xy.add(1).mul(0.5).mul(vec2(W, H)).toVar();
    const sx = sv.x.mul(256).toVar();
    const sy = sv.y.mul(256).toVar();
    const ok = pv.w
      .greaterThan(NEAR_EPS)
      .and(sx.abs().lessThan(1e8))
      .and(sy.abs().lessThan(1e8)) as unknown as NB;
    const xi = toI(sx.round()).toVar() as unknown as NI;
    const yi = toI(sy.round()).toVar() as unknown as NI;
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
    // ── UNIFORM early-outs — these run BEFORE the cooperative-load barrier below. itemIdx is
    // WORKGROUP-UNIFORM (derived from workgroupId only) and itemCount (address 0) / the cap
    // are broadcast/const ⇒ each condition is workgroup-uniform: every thread takes the same
    // branch, so all survivors reach the barrier together (never a partial-workgroup barrier).
    returnIf(itemIdx.greaterThanEqual(itemCount));
    // Cluster-cap guard: keep writes in-bounds. Clusters past the cap get no projected verts
    // (Classify guards the same slot ⇒ they simply do not render — an overflow parity hole,
    // surfaced; the cap must exceed the frame's visible-cluster count).
    returnIf(itemIdx.greaterThanEqual(uint(PROJ_CLUSTER_CAP)));
    // FLAT per-cluster base: this cluster owns the fixed region [itemIdx·MAX_CLUSTER_VERTS, +512)
    // (vertsPerCluster == MAX_CLUSTER_VERTS). The slot is recCluster + local — no reservation read.
    const recCluster = itemIdx.mul(uint(vertsPerCluster)).toVar();
    const localTri = localX().toVar();
    const base = itemIdx.mul(uint(CTX_STRIDE)).toVar();

    // ── Lever A (always-on): cooperative WORKGROUP-SHARED per-cluster ctx / vcompact ───────
    // This pass is dispatched ONE workgroup / cluster, MAX_CLUSTER_TRIS threads. The per-cluster
    // ctx (35 words), plus ci and vcompact's (vMin, uniqueCount), are WORKGROUP-UNIFORM — yet the
    // per-corner dispatch had ALL threads re-issue the IDENTICAL global loads (broadcast reads),
    // the measured buffer-READ-limiter(92%) / LLC(96%) wall (~94% of load-issues redundant). Load
    // them ONCE per workgroup into threadgroup memory, barrier, then decode from shared. Exact
    // idiom already shipping in the wgcache path (NaniteRaster ~866-961).
    // ≈148 B/workgroup on-chip (37 u32); zero VRAM; the values — hence the written xi/yi/dz
    // records — are BYTE-IDENTICAL (a plain u32 copy → same bitcasts → same projectVert()).
    const shCtx = workgroupArray('uint', CTX_STRIDE) as unknown as {
      element(i: NU): unknown;
    };
    // companion for the broadcast scalars the body re-reads: [0]=vMin, [1]=uniqueCount. (ci's
    // ONLY use is this vcompact lookup, so lane 0 derives it and stores the RESULTS — the ci
    // broadcast is fully subsumed; itemCount is NOT hoisted, its sole use is the uniform
    // pre-barrier early-out above.)
    const shScl = workgroupArray('uint', 2) as unknown as {
      element(i: NU): unknown;
    };
    const shSet = (arr: { element(i: NU): unknown }, i: NU, v: NU): void =>
      void (arr.element(i) as unknown as { assign(x: NU): unknown }).assign(v);
    // Cooperative fill in WORKGROUP-UNIFORM control flow — the per-lane If-stores CLOSE before
    // the barrier, so the barrier itself sits at uniform top-level scope. Lanes 0..CTX_STRIDE-1
    // each store one ctx word to a DISTINCT cell (no races); CTX_STRIDE(35) ≤ MAX_CLUSTER_TRIS
    // (128/255) ⇒ one stride, but the ceil-loop keeps it correct if that ever changes.
    for (let k = 0; k < Math.ceil(CTX_STRIDE / MAX_CLUSTER_TRIS); k++) {
      const w = (
        k === 0 ? localTri : localTri.add(uint(k * MAX_CLUSTER_TRIS))
      ).toVar();
      If(w.lessThan(uint(CTX_STRIDE)), () => {
        shSet(shCtx, w, elemU(clusterCtxV.ro, base.add(w)) as unknown as NU);
      });
    }
    // lane 0 loads the broadcast scalars. ci = qRaster[itemIdx+1].y — the SAME item world1
    // reads (item.y indexes the cluster array populateVCompact iterated, so vcompact[ci·2] =
    // (vMin, uniqueCount)). Store the derived (vMin, uniqueCount); nothing else needs ci.
    If(localTri.equal(uint(0)), () => {
      const ci0 = qRasterRO.element(itemIdx.add(uint(1))).y.toVar();
      shSet(shScl, uint(0), elemU(vcompact, ci0.mul(uint(2))) as unknown as NU);
      shSet(
        shScl,
        uint(1),
        elemU(vcompact, ci0.mul(uint(2)).add(uint(1))) as unknown as NU,
      );
    });
    // UNCONDITIONAL, at uniform top-level scope ⇒ every survivor reaches it together.
    workgroupBarrier();

    // decode the per-cluster ctx from SHARED memory (was clusterCtxV.ro global broadcast reads)
    // — SAME 35-word layout (CTX_U u32 + CTX_F f32-as-bits), same bits ⇒ byte-identical records.
    const rU = (i: number): NU =>
      (shCtx.element(uint(i)) as unknown as NU).toVar() as unknown as NU;
    const rF = (i: number): NF =>
      bcU2F(shCtx.element(uint(CTX_U + i)) as unknown as NU).toVar() as unknown as NF;

    // voxel(7) clusters do NO triangle work — world1 bails before projecting; match it so
    // their corner slots stay untouched (and Classify skips them too ⇒ never read). Reads
    // SHARED (post-barrier); the value is workgroup-uniform ⇒ all threads return together.
    returnIf(rU(10).equal(uint(7)));
    // skip TERRAIN (isHF) HW clusters only (slot 11 AND slot 0) — the `_clT` draw keeps the
    // full compute-fetch vertex path and never reads projVertBuf. MESH HW clusters DO flow
    // through (the `_clE` vertex now reads their pre-projected verts). The Phase-2 retry
    // conditions (the 2026-07-09 revert: projection ballooned 7.66→12.78% projecting
    // per-corner terrain HW clusters, and the compiled real-clip fallback kept `_cl`'s
    // 80-reg/80B-spill ceiling) are BOTH MET here:
    //   (a) near-crossers never reach the mesh HW path — clusterHwClass gates mesh
    //       HW-eligibility on view-z clear of the near plane (min sphere clip-w >
    //       NEAR_MARGIN), so `_clE` compiles NO sentinel/real-clip fallback;
    //   (b) terrain is EXCLUDED from projection (this skip) — only DEDUPED mesh clusters
    //       project, and clusterHwClass's coverage gate (vcompact count > 0) guarantees
    //       every projected mesh HW cluster takes the per-UNIQUE-VERT path below.
    // (Profile pair of the revert: profile-results-20260709-{194832,204125}.)
    // With the flagship OFF (HWPROJ, default — see NaniteHwClass) nobody reads projected
    // verts for HW clusters ⇒ skip ALL of them (the pre-flagship rule); narrowed to
    // terrain-only when the ?hwproj=1 opt-in compiles the `_clE` reader.
    if (HWPROJ) {
      returnIf(rU(11).equal(uint(1)).and(rU(0).equal(uint(1))));
    } else {
      returnIf(rU(11).equal(uint(1)));
    }

    const triCount = rU(3);
    // NB: the per-thread `localTri < triCount` gate is NOT a global early-out anymore — the
    // mesh per-unique-vert path (below) gives a thread with localTri ≥ triCount real vert work
    // (s = localTri stride). It is re-applied inside the terrain + mesh-fallback branches only.

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

    // per-CORNER writer — the VERBATIM original body (terrain + the mesh vcCount==0 fallback).
    // One thread owns one tri and projects its 3 corners into their canonVertSlot slots
    // (terrain: localTri*3+corner; mesh: vi−vBase). Each slot has a UNIQUE writer per frame ⇒
    // plain stores, no atomics; shared mesh verts collapse to one slot (idempotent — every
    // writer of a slot writes the identical projected value). No clear/stale-read: the record
    // the raster reads this frame is exactly what THIS pass wrote this frame.
    const emitPerCorner = (): void => {
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
    };

    // task #76 Lever 1 — per-UNIQUE-VERT dispatch for vcompact-covered mesh clusters.
    // nanProjectVerts is MEM-BOUND (vertex reads + projVertBuf writes); the per-corner path
    // re-fetched, re-transformed and re-wrote each SHARED vert ~3–6× into the SAME deduped slot
    // (idempotent — the dedup was STORAGE-only). Here each unique vert is projected ONCE ⇒
    // ~½–⅙ the transforms AND the mem traffic (the whole point).
    // BIT-IDENTITY: for a mesh cluster (isHF false, nfetch variant 'both'),
    // fetchWorldVertByIndex(ctx, vi) IS the exact explicitWorldByIndex(ctx, vi) that per-corner
    // fetchWorldVert(ctx, localTri, v) resolves to when vi = indices[(triStart+localTri)*3+v];
    // and vBase = vcMin == indices[triStart*3] (the meshletized first corner = the min index,
    // which canonVertSlot's mesh branch already relies on), so slot = recCluster + s
    // (s = vi−vBase) equals the per-corner canonVertSlot. Same value, same slot ⇒ byte-identical.
    If(ctx.isHF as unknown as NB, () => {
      // TERRAIN (isHF) — kept per-CORNER EXACTLY (≤384 corners, no universal dedup key, not the
      // whale). Per-thread tri gate (was the global returnIf; a barrier-free plain-store kernel
      // ⇒ a per-thread guard is safe — no deadlock). vcompact is mesh-only; terrain never reads it.
      If(localTri.lessThan(triCount), () => {
        emitPerCorner();
      });
    }).Else(() => {
      // vcompact[ci] = (vMin, uniqueCount) — broadcast-loaded ONCE into the shared companion
      // (shScl[0..1], Lever A above); read from shared here instead of re-issuing the global
      // load per thread. Same value ⇒ byte-identical records (terrain never reaches this Else).
      const vcMin = (shScl.element(uint(0)) as unknown as NU).toVar();
      const vcCount = (shScl.element(uint(1)) as unknown as NU).toVar();
      If(vcCount.greaterThan(uint(0)), () => {
        // MESH, vcompact-covered — project each unique vert ONCE. Threads STRIDE over the unique
        // range [0, uniqueCount): a thread with localTri ≥ triCount may still own vert s=localTri,
        // so this path must NOT take the localTri<triCount gate — it gates on s<uniqueCount.
        // uniqueCount ≤ VCACHE_VERTS ⇒ ceil(VCACHE_VERTS/WGSIZE) strides cover it (unrolled).
        for (let k = 0; k < Math.ceil(VCACHE_VERTS / MAX_CLUSTER_TRIS); k++) {
          const s = (
            k === 0 ? localTri : localTri.add(uint(k * MAX_CLUSTER_TRIS))
          ).toVar();
          If(s.lessThan(vcCount), () => {
            const wv = fetchWorldVertByIndex(ctx, vcMin.add(s));
            const { xi, yi, dz, ok } = projectVert(wv as unknown as NV3);
            // slot local = s = vi−vBase (== the per-corner canonVertSlot mesh local); base =
            // recCluster, clamp = MAX_CLUSTER_VERTS-1. For a covered mesh cluster s < vcCount ≤
            // MAX_CLUSTER_VERTS, so the minU is a no-op (bit-identical to the per-corner slot).
            const rb = recCluster
              .add(minU(s, uint(MAX_CLUSTER_VERTS - 1)))
              .mul(uint(PROJ_VERT_STRIDE))
              .toVar();
            wU(rb, bcI2U(xi));
            wU(rb.add(uint(1)), bcI2U(yi));
            wU(
              rb.add(uint(2)),
              (ok as unknown as { select(a: NU, b: NU): NU }).select(
                bcF2U(dz),
                uint(NEAR_SENTINEL),
              ),
            );
          });
        }
      }).Else(() => {
        // MESH, NOT vcompact-covered (tooWide > VCACHE_VERTS / streamed-terrain / any count=0)
        // — FALL BACK to the exact per-corner path so no vert is under-projected (holes).
        If(localTri.lessThan(triCount), () => {
          emitPerCorner();
        });
      });
    });
  })().compute(QRASTER_CAP * MAX_CLUSTER_TRIS, [MAX_CLUSTER_TRIS]);
  (kn as unknown as ComputeKernel).setName('nanProjectVerts');
  // tight indirect size (itemCount workgroups) — same override the world1 raster uses.
  setIndirectDispatch(kn, rasterDispatchFullAttr);

  return { projVertAttr, projVertV, kProjectVerts: kn, vertsPerCluster };
}
