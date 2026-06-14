/**
 * PERF-3 win #2 (D-N40 / LOG bb–bd) — the per-cluster cooperative VERTEX-TRANSFORM
 * cache, ISOLATED here so the core SW raster kernel (NaniteRaster) stays clean. One
 * workgroup == one cluster (128 threads); a 128-tri cluster references only ~82 UNIQUE
 * verts but fetchWorldVert runs per CORNER = 3×triCount ≈ 384× (~4.7× redundant). When
 * the cluster's vertex indices occupy a build-time-compacted contiguous range
 * (gpu.vcompact[ci] = (vMin, count), count>0 ≤ VCACHE_VERTS — explicit meshes;
 * terrain/window-grid stay 0 = fallback), the workgroup transforms each unique vert
 * ONCE into workgroup shared memory via fetchWorldVertByIndex, then each triangle reads
 * its corners from there.
 *
 * MEASURED (LOG bd) — a MARGINAL / CONDITIONAL optimization, DEFAULT OFF (?vcompact=1).
 * The win = R×C − (barrier + shared-mem overhead), and R is structurally only ~4.7
 * (vs makeCtx's 128 — win #1), so it needs a HIGH per-vertex cost C to clear the fixed
 * barrier: NEUTRAL on the terrain-heavy vista (far-terrain transform is a single
 * texLoadR, absorbed by the GPU texture cache), payload −1 quantum on the wind-trunk
 * forest (4 sin/vert), and −1 quantum WORSE with a window-grid extension (the barrier
 * taxes the cheap-terrain majority). The redundancy is INTRA-cluster (~82 verts ≈ 2 KB
 * fit L1 regardless of scene size) ⇒ SCALE-INVARIANT in object count — 5× more
 * instances of shared meshes does NOT change the per-cluster verdict. It becomes a
 * robust win only with COST-AWARE GATING (cull bins clusters by per-vertex cost; pay
 * the barrier only on the expensive ones) — UNBUILT. Kept off-by-default as the
 * foundation for that, and for heavier-WPO / animated geometry where C is high.
 *
 * gpu.vcompact is referenced HERE ONLY (the returned closures build into the raster
 * COMPUTE kernel) ⇒ bound to the SW raster alone; the resolve fragment stage (at the
 * 10-storage-buffer Metal ceiling) never sees it.
 */

import { If, uint, vec3, workgroupArray, workgroupBarrier } from 'three/tsl';
import type { NU, NV3 } from '../gpu/TSLTypes';
import { VCACHE_VERTS } from './GeometryRegistry';
import type { RegistryGpu } from './GeometryRegistry';
import type { NaniteFetch, VertCtx } from './NaniteFetch';
import { elemU } from './Tsl';

/** world position of (localTri, corner v) for the primed cluster — cached or direct. */
export type CornerFetch = (localTri: NU, v: 0 | 1 | 2) => NV3;

export interface VertexCache {
  /** Whether the cooperative cache is active (?vcompact=1). */
  readonly enabled: boolean;
  /** Call ONCE per cluster inside the raster kernel — after ctx is ready and all 128
   *  threads are live (past the UNIFORM itemIdx≥itemCount early-out), BEFORE the
   *  triangle loop. When enabled it emits the cooperative populate + an unconditional
   *  workgroup barrier and returns a cache-or-fallback corner fetcher; when disabled it
   *  emits nothing and returns the plain per-thread fetch. */
  prime(ctx: VertCtx, ci: NU, localTri: NU): CornerFetch;
}

/** Build the vertex cache for one raster instance. Reads its own ?vcompact gate so the
 *  flag never appears in the core kernel. `fetch` supplies both the by-index transform
 *  (cooperative populate) and the plain fetchWorldVert (fallback / disabled path). */
export function makeVertexCache(gpu: RegistryGpu, fetch: NaniteFetch): VertexCache {
  const { fetchWorldVert, fetchWorldVertByIndex } = fetch;
  const enabled = new URLSearchParams(window.location.search).get('vcompact') === '1';

  if (!enabled) {
    return { enabled: false, prime: (ctx) => (lt, v) => fetchWorldVert(ctx, lt, v) };
  }

  const prime = (ctx: VertCtx, ci: NU, localTri: NU): CornerFetch => {
    const shVerts = workgroupArray('vec3', VCACHE_VERTS) as unknown as {
      element(i: NU): unknown;
    };
    const vcMin = elemU(gpu.vcompact, ci.mul(uint(2))).toVar();
    const vcCount = elemU(gpu.vcompact, ci.mul(uint(2)).add(uint(1))).toVar();
    // strided cooperative populate — thread t writes slots t, t+128, … to DISTINCT
    // cells (no atomics). VCACHE_VERTS ≤ 2·128 ⇒ ≤2 strides, unrolled at build time.
    for (let s = 0; s < Math.ceil(VCACHE_VERTS / 128); s++) {
      const slot = (s === 0 ? localTri : localTri.add(uint(s * 128))).toVar();
      If(slot.lessThan(vcCount), () => {
        (shVerts.element(slot) as { assign(x: NV3): void }).assign(
          fetchWorldVertByIndex(ctx, vcMin.add(slot)),
        );
      });
    }
    // UNCONDITIONAL: vcCount is storage-derived ⇒ naga can't prove a gated barrier
    // uniform. count=0 clusters pay an empty barrier and fall back per corner below.
    workgroupBarrier();

    return (lt, v) => {
      const wv = vec3(0).toVar();
      If(vcCount.greaterThan(uint(0)), () => {
        // vi is derived from gpu.indices exactly as fetchWorldVert does ⇒ the cached
        // vert is bit-identical to the inline transform (NaniteFetch 2a split is
        // behavior-preserving). The vcCount>0 test is UNIFORM (shared ci) — no barrier.
        const vi = elemU(gpu.indices, ctx.triStart.add(lt).mul(uint(3)).add(uint(v)));
        wv.assign((shVerts.element(vi.sub(vcMin)) as { toVar(): NV3 }).toVar());
      }).Else(() => {
        wv.assign(fetchWorldVert(ctx, lt, v));
      });
      return wv as unknown as NV3;
    };
  };

  return { enabled: true, prime };
}
