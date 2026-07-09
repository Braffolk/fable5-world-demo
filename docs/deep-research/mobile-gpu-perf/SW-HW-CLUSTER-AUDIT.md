# SW/HW raster audit — are we doing cluster-level work per triangle?

**Question (user, 2026-07-05):** map every SW and HW raster path + the decision process that
routes between them, and find where we do *cluster-level* work *per triangle* — both **logical**
(a per-cluster calc accidentally recomputed per-tri) and **iterative** (iterating individual tris
where we could batch per-cluster). Then find a precision-safe way to make everything that should
be per-cluster, per-cluster.

**Verdict:** the two biggest per-tri-cluster-work sins were **already fixed** (`wgcache` broadcasts
`makeCtx` once/cluster; the vertex cache exists). What remains genuinely per-triangle-that-should-be-
per-cluster is the **SW/HW routing decision** and the **HW execution** — and, as a consequence,
**HW-bound clusters are fully processed by the SW kernel before being handed to a botched per-tri HW
draw.** The fix is to classify SW vs HW **per cluster at the cull cut** (which already computes
per-cluster projected size), which also *dissolves the i32 precision limit* by construction.

---

## 1. All raster paths

### Triangle rasterizer (the SW/HW split lives here) — `NaniteRaster.ts`
- **SW:** `rasterKernel(mode)` → `kRasterDepth` / `kRasterCombined` / `kRasterWorld1` (`:1528-1535`).
  One **workgroup == one cluster** (128 threads = 128 tris); dispatched indirect over `qRaster`
  (`compute(QRASTER_CAP * MAX_CLUSTER_TRIS)`, `:1523`). Writes the vis-buffer via `atomicMax`
  depth-keyed election.
- **HW:** `buildHwMaterial('depth'|'combined'|'world1')` (`:1642`) — a **non-indexed `3n`-vertex
  soup draw** (`hwDrawBuf[0] = n*3`, `:1544`). VS fetches per-vertex from `hwQueue` and transforms
  (`:1649-1679`); FS re-does the **same** `atomicMax` election into the vis-buffer (`:1681-1718`).
- **Shadow raster** reuses this machinery via the shared cut (`NaniteClipCull` → `qRaster` →
  `kRasterDepth`); same SW/HW split pattern (verify when touching).

### Separate rasterizers (NOT the tri SW/HW split — own paths)
- **Voxel splat** `kVoxScatter` (`NaniteVoxelRaster.ts:1579`) — per-brick, not per-tri. Own bin/
  count/prefix/scatter chain (`kVox*` in `NaniteCull.ts`). *Has its own per-brick-vs-per-cluster
  question — out of scope here, flagged.*
- **Grass** `kGrassRay` — per-pixel raymarch, not a triangle rasterizer.
- **FarTiles** aggregated splat — separate far representation.

## 2. All decision points (the "filtering")

| # | Decision | Granularity | Where | Verdict |
|---|---|---|---|---|
| 1 | LOD **cut** + frustum + cone + min-size ⇒ emit cluster | **per-CLUSTER** | `kTraverseAB/BA` BFS, `NaniteCull.ts` (`project(ownError) ≤ τ`) | ✅ correct — and already computes per-cluster projected size |
| 2 | **SW vs HW** (near-plane cross, bbox extent ≤ `swmax`) | **per-TRIANGLE** | `NaniteRaster.ts:943` (near), `:993-1004` (`smallEnough`) → push to `hwQueue` | ❌ **should be per-cluster** |
| 3 | swcoop small/large bin (`SW_SMALL_EXT`=4px) | per-triangle | `:1004+` (default `swcoop=0` off) | pixel-coverage strategy, ~ok |
| 4 | backface / winding (`orientForRaster`) | per-triangle | `:963` | ✅ genuinely per-tri (cone-cull is the per-cluster analog, done at #1) |

`swmax` is **clamped to [2,16]** (`:112`) — `?swmax>16` silently falls back to 16, so all
larger-value tests were no-ops. Empirically (clamp removed) the i32 coverage math holds clean to
~128px and breaks by 256 — matching `edge term = 2·(extent·256)²` hitting 2³¹ at ~128px.

## 3. Per-cluster-vs-per-tri inventory (the core question)

| Work | Nature | Current | Status |
|---|---|---|---|
| `makeCtx` (instance transform, wind, **trunk gust texture samples**, mesh decode) | per-cluster constant | **per-cluster** — `wgcache` computes once on thread 0, broadcasts via workgroup shared mem (`:705`, default ON) | ✅ **already fixed** (was 128×/cluster) |
| vertex world-transform | per-unique-vertex | **per-tri-CORNER, ~4.7× redundant** (~384 vs ~82 unique/cluster) | ⚠️ dedup exists (`NaniteVertexCache`) but **default OFF + bit-rotted**: binding `vcompact` is the 11th storage buffer > Metal's 10/stage limit → empty scene (`:56-63`). Also only a *marginal* win (R≈4.7, low per-vert cost). |
| **SW/HW routing decision** (bbox extent, near-plane) | per-cluster (LOD-coherent) | **per-triangle** | ❌ **the logical sin** |
| **HW execution** | per-cluster batchable | **per-tri soup** — 3n verts, no index sharing, no post-transform cache, **transformed twice** (SW-route + HW-draw) | ❌ **the iterative sin** |
| **HW-bound clusters** | should skip SW | **fully processed by SW kernel first** — 128 threads transform+bbox+push each, all wasted | ❌ consequence of #2 |
| bbox / edge setup / coverage / scanline | per-triangle | per-tri | ✅ genuinely per-tri (rasterization) |

**Answer:** yes — in exactly three linked places (SW/HW *decision*, HW *execution*, and HW clusters
*visiting the SW kernel*), plus a lesser one (vertex transform redundancy, blocked by the binding
budget). The expensive decode (`makeCtx`, incl. gust texture taps) is **already** per-cluster.

## 4. The precision-safe per-cluster split (the fix)

Move the SW/HW decision to the **cull cut** (`kTraverseAB`), which already runs per-cluster and
already computes projected size:

```
nearestDepth = dist(camPos, center) − radius
projTriSize  ≈ project(radius / sqrt(triCount), nearestDepth)   // worst-case (nearest) tri size
HW  if  projTriSize > SWMAX_PX  OR  nearestDepth < nearZ        // near-plane straddle → HW clips
SW  otherwise
```

- **Two cut queues** `qRasterSW` / `qRasterHW` (two atomic counters at emit) + own indirect args.
- **SW path:** dispatch over `qRasterSW` only. Clusters are pre-guaranteed all-small + non-near-
  crossing ⇒ **drop the per-tri `nearOK` + `smallEnough` + HW-push branches** → leaner hot loop
  (helps id178 occupancy) and **i32-safe by construction** (SWMAX_PX ≈ 64–100, safely < the ~128px
  cliff). **This is the "way around the precision issue"** — no per-tri size check, no per-tri
  overflow possible, because anything that could overflow was routed to HW as a whole cluster.
- **HW path:** a compute *gather* expands `qRasterHW` into a contiguous **index** buffer (global
  vertex ids + instId); **one `drawIndexedIndirect`**; VS transforms each vertex **once** (shared
  within a cluster → post-transform-cache reuse); FS reuses the **existing** election (`:1693-1718`).
  HW-bound clusters **never enter the SW kernel**. (WebGPU has no `multiDrawIndirect` → compaction +
  one draw, not per-cluster sub-draws.)

### Wins / non-wins (honest)
- **Wins:** proper HW path (single-transform, vertex-shared) for near/big geometry; HW clusters skip
  the SW kernel entirely (kills 128 wasted per-tri transforms+pushes each); leaner SW loop; routing
  moves from ~12–100M/frame to ~cluster-count; and it lets us **honestly re-measure the SW/HW
  crossover** (the old "SW≈HW parity" was measured on the botched per-tri HW path, doubly contaminated
  by the [2,16] clamp).
- **Non-win:** does **not** directly shrink the id178 micro-tri whale — dense mid/far crowns correctly
  stay SW. Separate lever (crown tri-count). But a fast HW path may pull near-crown/trunk cost off SW.

### Real risks
- **SW↔HW seam depth consistency** (SW fixed-point 1/256 vs HW float interp) at cluster boundaries →
  possible cracks/z-fight. Must be shotdiff-gated; mitigate with matched `depthKey` quantization
  (and possibly snap HW verts to the 1/256 grid in the VS).
- **Binding budget:** Metal 10 storage buffers/stage already bit `vcompact`. Two queues + gather + a
  proper HW draw must fit; may need to fold counters (e.g. `hwCount` into `hwQueue[0]`, as the vcache
  note suggests) or pack into spare cluster words.
- Conservative nearest-point classify over-routes depth-spanning clusters to HW (rare, fine).

## 5. Verification-first (before building)
1. Fix `hwref` (`?nanitedbg=hwref`) for a trustworthy proper-HW A/B reference.
2. Instrument the current HW-routed share (`?scar` already counts HW fragments, `:1706`) — size the win.
3. Raise the `swmax` clamp to find the real i32 cliff empirically and measure SW cost vs tri size.
