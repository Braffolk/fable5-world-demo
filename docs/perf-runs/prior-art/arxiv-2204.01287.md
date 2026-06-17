# Prior-art brief — arXiv:2204.01287

**Title:** *Software Rasterization of 2 Billion Points in Real Time*
**Authors:** Markus Schütz, Bernhard Kerbl, Michael Wimmer (TU Wien), 2022 (techreport / arXiv preprint, v1).
**Source code:** https://github.com/m-schuetz/compute_rasterizer (cloned & read; hot kernels in `modules/compute_loop_las2/render.cs`, `compute_loop_las/render.cs`, `compute_loop_las_hqs/{depth,color}.cs`).
**PDF:** `docs/perf-runs/prior-art/sources/papers/arxiv-2204.01287.pdf` (11 pp).

## What it is / why it is relevant

A GPU **compute-shader software rasterizer for point clouds** (OpenGL compute / GLSL, NVIDIA RTX 3090). It brute-forces up to ~144 billion points/s — 2 billion at 60fps — beating the bandwidth-limited prior SOTA (SKW21) by up to 3-5x. It is the **immediate point-cloud cousin of Nanite's SW raster** (the paper explicitly frames itself against Nanite [KSW21]).

**Critical caveat for our cost map:** this rasterizer draws **one point → exactly one pixel**. There is *no per-fragment coverage loop, no edge tests, no per-primitive overdraw* — each primitive touches a single pixel. So it has **no direct analog to OUR dominant 60% per-pixel coverage loop** (which arises from ~1px triangles with 3 edge tests + bary-z over 12-20x overdraw). The transferable value is concentrated in: (a) the **visibility/depth election** mechanism — and crucially how it depends on 64-bit atomics, which is exactly our portability question; (b) the **per-thread work-distribution / launch model** (one workgroup per batch, persistent threads, prefetch) — attacks our 40% transform+launch; (c) a **subgroup same-pixel atomic-collapse** trick in the HQS color pass that *does* map onto overdraw; (d) bandwidth/precision-on-demand. Be honest: this paper is more of a confirmation-and-adaptation source than a silver bullet for our 60% loop.

## Raster architecture

- **Work distribution (Sec 3.1-3.2):** points grouped into **batches of 10'240 consecutive points** (Morton-ordered for locality). **One compute workgroup per batch**, 128 threads, each thread loops over ~80 points (persistent threads). Batch-level frustum cull + LOD cull + coordinate-precision selection are computed once per workgroup and amortized over 10k points. (`render.cs:259-280`.) This is *exactly our one-workgroup-per-cluster model* — they validate it and add per-batch amortized setup.
- **Election (Sec 3.2, `render.cs:243-250`):** `depth32 = floatBitsToInt(pos.w)`; `newPoint(uint64) = (depth<<32) | pointIndex`; **relaxed plain load → compare → `atomicMin(framebuffer[pixelID], newPoint)`** only if it would win. Losers early-out after the relaxed load. **This is structurally identical to OUR election** (relaxed-load early-out, atomic only on a new front) — except they pack into a **64-bit** word and use `atomicMin`, whereas we pack 24b-depth|8b-tiebreak into a **32-bit** `atomicMax` then side-store the 25b id. **Their 64-bit depth|payload visbuffer is the exact trick that is NOT portable to WGSL.**
- **Bandwidth (Sec 3.3-3.4):** adaptive coordinate precision (load 4 / 8 / 12 bytes/point by projected batch size) + struct-of-arrays + vectorized 128-bit prefetch. They are **bandwidth-bound (~70% of kernel time in memory ops)**, NOT ALU/atomic-bound.

## Transferable techniques

### DIRECT ports

**T1 — Separate-pass 32-bit depth election (the no-64-bit-atomic answer).**
The HQS variant (`compute_loop_las_hqs/depth.cs:375-379`) does NOT pack a payload into the depth word at all: it runs a **depth-only pass** with a plain **32-bit `atomicMin(ssFbo_depth[pixelID], floatBitsToInt(pos.w))`**, then a *second* color pass reads back `ssFbo_depth` and keeps fragments within 1% of it (`color.cs:399-403`). This is the canonical way to sidestep 64-bit atomics. Our existing 32b-atomicMax+side-store already does this in a single pass, so this **CONFIRMS our packing is sound** and is not itself a new win — but it documents the standard fallback and the depth-band tolerance (the `*1.01` slack) we could exploit for cheap order-independent shading. Direct-port value: low (we already do the 32b trick); confirmation value: high.

**T2 — Subgroup same-pixel atomic collapse (`color.cs:408-441`).**
Before issuing an atomic, partition the subgroup by destination `pixelID` (`subgroupPartitionNV(pixelID)`), elect one leader per pixel-group (`subgroupPartitionedMinNV`), reduce all same-pixel contributions inside the subgroup (`subgroupPartitionedAddNV`), and have **only the leader issue ONE atomic** for the whole group. This collapses N same-pixel atomics → 1. **This maps onto OUR overdraw**: with 12-20x overdraw, many fragments in a subgroup land on the same pixel. WGSL has `subgroupBallot`/`subgroupAdd`/`subgroupMin` (and `subgroupShuffle`) — but **NOT** the NV *partitioned* ops. We would emulate partition-by-key via a subgroup loop (broadcast each distinct pixelID, ballot members, reduce). **Caveat:** our bottleneck is the *coverage test + relaxed load*, not atomic contention (already refuted as our bottleneck), and the atomicMax fires only on a new front — so this collapses an op we mostly *don't* execute. Likely **low real win for us**, and the partition-emulation cost may exceed it. Worth a measured spike only if a future change makes atomics hot.

**T3 — Per-thread vectorized prefetch + persistent loop (`render.cs:322-335`).**
Each thread issues a single 128-bit (`uvec4`) load that feeds **4 inner iterations**, prefetching iteration i+1's data before consuming iteration i's, hiding global-memory latency; reported **~30% gain** in their bandwidth-bound kernel. **Indirectly relevant:** our 40% transform+launch includes `3x fetchWorldVert` with ~5x redundant vertex re-fetch and ~48 wind-animation texture taps per triangle. Batch-prefetch the cluster's unique vertices once into workgroup shared memory / registers, then index them — eliminating the 5x re-fetch and amortizing wind taps per *vertex* not per *triangle-corner*. This is the strongest indirect lever from this paper for our 40%. (See T6.)

### INDIRECT (adapt-the-principle) ideas

**T4 — Batch-level amortized setup (Sec 3.2).** Compute frustum cull, LOD/precision selection, and (for us) **shared per-cluster constants** once at workgroup entry over a large batch so the cost amortizes across all triangles. We already do one-workgroup-per-cluster; the principle says: push *more* per-cluster invariant work to thread-0/shared-memory (e.g. the cluster transform, wind sampling basis, bbox) instead of recomputing per triangle. Attacks the 40% setup. **Note:** the paper's own frustum/LOD culling is a confirmed DEAD END for us — conservative culling is ~0% on holey foliage (a gap pins max-Z far) and sub-pixel LOD already emits at <=1px. Take the *amortization* principle, not their cull.

**T5 — Adaptive coordinate precision / bandwidth-on-demand (Sec 3.3, `render.cs:319-469`).** Load only as many coordinate bits as the projected size needs (4/8/12 B). For us: distant foliage clusters could fetch **quantized/compressed vertex positions** (and skip or downsample the 48 wind taps) when sub-pixel jitter is imperceptible — a bandwidth cut on the 40% transform side. Quality-preserving only below the sub-pixel threshold; above it, full precision. Indirect, moderate effort, plausible quality-preserving if gated by screen size.

**T6 — Re-fetch elimination via SoA + shared-memory staging (Sec 3.1 + 3.4).** Their SoA layout + coalesced batch loads exist so each datum is loaded once with perfect coalescence. OUR ~5x redundant `fetchWorldVert` per triangle is the inverse anti-pattern. Stage the cluster's unique transformed+wind-animated vertices into `var<workgroup>` once (one thread per vertex), `workgroupBarrier()`, then have triangle threads read shared memory. Converts ~3x per-triangle fetches + 48 taps into ~1x per-vertex. **Highest-confidence transfer to our 40% transform+launch.** Effort: medium (needs vertex/triangle index split within the workgroup).

## Honest portability summary

- The paper's headline mechanism (64-bit `atomicMin` depth|payload visbuffer, `render.cs:244-250`) is **NOT portable** to WGSL (no 64-bit atomics) — but we already solved this with the 32b-atomicMax + side-store; the paper **confirms** that family of solution and offers the separate-depth-pass fallback (T1).
- It is **bandwidth-bound with 1px primitives**; it has **no per-fragment coverage loop**, so it offers **no direct attack on our DOMINANT 60% coverage loop**. Its real gifts are to our **40% transform+launch** (T3, T6 prefetch + re-fetch elimination) and bandwidth (T5).
- Subgroup atomic-collapse (T2) is the only idea touching overdraw, but our atomics are already not the bottleneck, and WGSL lacks the NV *partitioned* subgroup ops (emulation required) — flag as low-priority.
- **Confirmed dead ends to flag:** their frustum/occlusion/LOD culling = our already-refuted conservative-cull dead end on holey foliage; their adaptive precision is a *quality-bounded sub-pixel* idea, consistent with our refuted "sub-pixel LOD over-render" only if pushed past the threshold (don't).
