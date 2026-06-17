# Prior-art brief: extra-0 — CuRast (Cuda-Based Software Rasterization for Billions of Triangles)

- **slug**: extra-0
- **kind**: paper (+ open-source CUDA reference implementation)
- **source**: Schütz, Lipp, Kristmann, Wimmer (TU Wien). *CuRast: Cuda-Based Software Rasterization for Billions of Triangles.* arXiv:2604.21749v2, 24 Apr 2026.
- **refs**: https://arxiv.org/abs/2604.21749 · code: https://github.com/m-schuetz/CuRast
- **fetched**: `curl` of `https://arxiv.org/pdf/2604.21749` → `docs/perf-runs/prior-art/sources/papers/extra-0.pdf` (16.6 MB, 11 pp, HTTP 200, OK). PDF page-render needs poppler (absent); used `pymupdf` (`pip3 install pymupdf`) to extract text — clean. Source `git clone --depth 1` of the repo → `/tmp/curast-src` succeeded; deep-read of the actual hot kernels: `src/kernels/triangles_visbuffer.cu` (895 L), `rasterization_helpers.cuh`, `HostDeviceInterface.h`.

---

## What it is

A pure-compute (CUDA) software rasterizer that brute-forces unstructured triangle soups — no BVH/LOD precompute — at 0.6–18.9 **billion** triangles, beating Vulkan HW by 2–5× (unique) / up to 12× (instanced) on dense, opaque, pixel-sized geometry (photogrammetry / Zorah). It is the most directly-relevant source in the pool: a 3-stage adaptive size-routed pipeline whose **Stage 1 is almost exactly our kernel** — one thread per triangle, bbox walk, barycentric DDA, atomic-election visbuffer (28b depth | 36b tri-id). Stage 1 is 99%+ of frame time on dense data (Table 3: Zorah 74.1 of 74.9 ms), i.e. they live in the same cost regime we do: the per-fragment coverage loop dominates everything.

**One blunt caveat up front:** CuRast targets *static* photogrammetry. It has NO occlusion culling, NO clustered/shared-vertex reuse (it re-fetches per triangle, exactly the redundancy we want to kill — they flag it as future work, §6), and uses **64-bit** `atomicMin`. So it validates our architecture and hands us two concrete quality-preserving wins, but its core election trick is the thing WGSL forbids and our team already replaced.

---

## Raster architecture

3 size-routed stages over a **single shared visibility framebuffer** (`uint64` per pixel = depth‖id), all `atomicMin` (smaller depth wins). Persistent kernels: a fixed grid of workgroups loops, pulling work via a global `atomicAdd` counter (not static one-block-per-cluster).

- **Stage 1 — small tris (bbox < 128 px)** `triangles_visbuffer.cu:315` `stage1_drawSmallTriangles`. Persistent workgroups of 256 threads; each block claims a batch of 256 tris via `atomicAdd(numProcessedBatches,1)` then advances through the mesh list to map batch→mesh (`:350-377`). One thread per triangle. Transform → NDC → screen, frustum + backface + **tiny-tri sample-miss** cull, then per-thread bbox scanline with incremental barycentric DDA and `atomicMin` election (`:271-310`). Tris that are large or cross the near plane are pushed to a global queue for Stage 2.
- **Stage 2 — medium tris (128…4096 px)** `:431`. 32-thread warp per queued triangle; thread 0 does the (expensive, Sutherland-Hodgman near-plane-clipped) setup and `shfl`-broadcasts it; the 32 lanes stride a 1-D `fragOffset += 32` loop over the bbox. Tris > 4096 px or near-plane are split along 64×64 tiles into a Stage-3 queue.
- **Stage 3 — huge tris** `:622`. One 64-thread block per 64×64 tile; rasterizes via **world-space ray–triangle intersection** (avoids near-plane clip in NDC). "Good enough", admittedly unoptimized.
- **Resolve** `resolve.cu`: 1 thread/pixel, decode id, **binary search** the per-mesh cumulative-triangle prefix-sum to recover mesh, re-fetch verts, shade in **world space** (so near-plane tris get a valid mip footprint via 4-ray plane intersection, Fig. 4).

The election word (`rasterization_helpers.cuh:35` `pack_pixel`): 28-bit depth (float, sign bit + 3 mantissa bits dropped) in the high bits, 36-bit *global cumulative* triangle id in the low bits, one `atomicMin`. **This is the 64-bit trick we cannot use.**

---

## Transferable techniques (ranked by expected value to us)

### 1. [DIRECT — top transfer] Tiny-triangle "bbox misses every pixel-center" cull, BEFORE any edge test
`triangles_visbuffer.cu:190-196`. After computing the screen bbox, a per-axis test: if the whole bbox falls strictly *between* two sample-center positions on x OR on y, the triangle covers no sample → discard immediately, before the coverage loop, before the inner barycentric work.
```c
float sample_x = floorf(min_x), sample_y = floorf(min_y);
if (min_x > sample_x+OFF && max_x < sample_x+1+OFF) return;   // misses all x-centers
if (min_y > sample_y+OFF && max_y < sample_y+1+OFF) return;   // misses all y-centers
```
**Mechanism / why it attacks our 60%:** on dense micropoly geometry a *large fraction* of transformed triangles land entirely between pixel centers and contribute zero coverage, yet still pay transform + edge-setup + a (possibly empty) loop entry. This is a 2-compare early-out using bbox values we already compute. **Measured: Table 4 — Zorah close-up 102.5 → 74.1 ms (−27%), overview 140 → 99 ms (−29%)**, zero effect on already-cheap scenes (Sponza unchanged). This is the "~27% before any edge test" claim in the mission, and it is REAL and quality-preserving (a sub-sample triangle correctly produces no fragment — exactly what a sample-coverage rasterizer must do).
- **Attacks**: per-triangle 40% (kills empty loop-entries) AND the 60% loop (fewer fragments enter).
- **No-64b-atomic adaptation**: none needed — pure coverage math, orthogonal to the election.
- **WGSL adaptation**: we snap to a 1/256 fixed-point grid with a top-left rule (`NaniteRaster.ts:16,547`) but we do NOT have this early miss-cull — confirmed by grep, our pipeline still enters the scanline for sub-sample tris. Port: after the integer bbox + before the edge-setup block, test whether `[minX,maxX]` contains an integer sample column and `[minY,maxY]` a sample row (in our 1/256 units, "no integer pixel-center in the span"). It is a strict-superset early-out of our shipped x-span (#5), evaluated one step earlier (per-triangle, not per-row) and cheaper. **HIGHEST-confidence single lever here**, with a published −27% on the same workload class. Honest risk: our cluster cut already emits ≤1 px tris, so our miss-fraction may be lower than Zorah's; treat the −27% as an upper bound and MEASURE.
- **effort**: low (a handful of WGSL lines in the per-triangle prologue; no data-structure change).

### 2. [DIRECT — confirms a DEAD END, do not re-explore] "Walk the bbox; do NOT restrict to in-triangle (Pineda) — cheap wasted work beats branchy precise work"
Paper §4 (`:348-354`) and §6 (`:1222`): "we perform naïve boundary constraints by processing all fragments inside the triangle's bounding box, despite well-known approaches that only traverse fragments within the triangle [Pineda88]. We initially tried to do so but found that the additional effort required to avoid wasted work is often **more expensive than unnecessary but cheap work**, especially for massive, dense data sets that produce pixel-sized triangles." Listed AGAIN in future work as "did not yet have success… may be a matter of crafting the right implementation… may be more advantageous for less dense geometry."
- **Relevance to us**: our team SHIPPED a scanline x-span (commit 19eb834, `NaniteRaster.ts:669-697`) that solves each edge for its crossing x and skips outside-bbox pixels. CuRast's authors tried the analogous "traverse only inside-triangle" optimization and found it a net LOSS on pixel-sized tris. This is a **caution flag**, not a refutation: the x-span we shipped is the *cheap* form (incremental edge solve, no per-pixel branch divergence), whereas the form they rejected is per-fragment exact-clip. The lesson to carry: for ~1px tris the win from skipping a fragment must cost less than the fragment itself — so do NOT chase a *more precise* per-pixel inside test (it'll regress, as it did for them); the only profitable direction is moving the cull EARLIER and CHEAPER (→ technique #1, which they DID find profitable).
- **Adaptation**: n/a (it's a "stop digging here" signal). Records that the Pineda / exact-edge-traversal direction is a measured non-win on our geometry class.
- **effort**: n/a.

### 3. [INDIRECT — adapt the principle] Persistent kernel + global `atomicAdd` work-stealing instead of static one-block-per-cluster
`:346-377`. A *bounded* set of resident workgroups loops `while(true)`, each grabbing the next 256-triangle batch via `atomicAdd(numProcessedBatches, 256)` and walking the mesh list to bind the batch to its mesh. This decouples #workgroups launched from #work items and self-balances when triangle counts per node vary wildly.
- **Mechanism / what it attacks**: our 40% transform+launch is partly "one workgroup per visible cluster, ~154k dispatched". A persistent pool sized to fill the GPU (a few hundred workgroups) that pulls clusters from a global counter removes per-cluster launch overhead and, more importantly, lets a workgroup that finished a cheap cluster immediately grab another instead of idling — relevant if cluster cost has any variance (e.g. near vs far, or partial-screen clusters), even though our clusters are uniformly well-filled (235/256).
- **No-64b-atomic adaptation**: the work-distribution mechanism is independent of the election; `atomicAdd` on a u32 counter is fully WGSL-portable.
- **WGSL adaptation**: WGSL/WebGPU has no `cooperative_groups::this_grid()` and no grid-wide `grid.sync()`. So a *single-dispatch* persistent grid spanning all clusters is NOT directly expressible (no portable global barrier). The portable form: dispatch a fixed `workgroupCount` (tuned to saturate the M-series GPU, e.g. a few hundred), each workgroup spins `loop { i = atomicAdd(&workCounter, 1u); if i >= clusterCount { break } process(cluster[i]) }`. This is a real change from `dispatchIndirect(... rasterDispatchFullAttr)` (currently one wg/cluster). Honest assessment: our clusters are uniform, so the load-balance upside is modest; the launch-overhead upside depends on whether ~154k workgroup launches are actually costing us (MEASURE against a persistent ~512-wg pool first). Medium-confidence; the bigger transform-side win is #4/#6.
- **effort**: medium (restructure dispatch + add atomic work counter; keep election untouched).

### 4. [DIRECT — strong transfer] Incremental barycentric DDA: zero per-pixel cross products
`:251-309`. Edge/barycentric coefficients (`ds_dx, ds_dy, dt_dx, dt_dy`, the `factor = 1/cross`) are computed ONCE per triangle outside the loop; inside the loop each pixel just does `s += ds_dx; t += dt_dx; pixelID++` (X) and a row reset + `s_row_start += ds_dy` (Y). Coverage is `s>=0 && t>=0 && (1-s-t)>=0`. Depth via precomputed inverse-z interpolated and one `__fdividef`.
- **Mechanism / what it attacks**: our DOMINANT 60% loop. The mission says each covered fragment does "3 integer edge tests + 1 barycentric-z". Our scanline already increments integer edge functions per pixel (`NaniteRaster.ts:591-601`, "ex/ey = dE per +1 UNIT") — so we ALREADY implement this principle in fixed point. The transferable refinements on top of what we have: (a) derive the 3 edge values from the *two* barycentrics `s,t` (and `v=1-s-t`) so you carry 2 incrementing accumulators not 3; (b) keep a running `pixelID` integer rather than recomputing `x + width*y` per fragment.
- **No-64b-atomic adaptation**: orthogonal to election.
- **WGSL adaptation**: we're integer not float, which is good (watertight). Check whether our inner loop maintains 3 independent edge accumulators vs 2 baryc — collapsing to 2 saves one add/compare per fragment across 36M fragments. Verify `pixelID` is incremented, not recomputed. Small, safe, exactly-same-image.
- **effort**: low-medium.

### 5. [DIRECT, ALREADY-SHIPPED — confirms our x-span] Row-bounded scanline span
Implicit in Stage 1's tight per-row loop; explicit in our commit 19eb834. CuRast itself does NOT row-clip (it relies on #1 + bbox), and §6 says deeper in-triangle traversal regressed (see #2). So the literature CONFIRMS our shipped x-span is at the favorable end (cheap incremental) and warns against pushing it toward exact per-fragment clipping.
- **Adaptation**: none — already shipped. Flag: do not extend it toward Pineda-exact traversal (#2).

### 6. [INDIRECT — adapt the principle] `__fdividef` / register-pressure → occupancy, and the "sample-offset vs vertex-offset" codegen cliff
Two micro-architectural findings with outsized effect:
- **(a)** Using `__fdividef(1, inv_depth)` instead of `1.0f/inv_depth` cut the Stage-1 kernel from **55 → 48 registers**, raising occupancy and overall speed (`:434-436`). The arithmetic result barely changes; the *register count* is the lever.
- **(b)** `:163-176`: applying the half-pixel sample offset by **subtracting 0.5 from the screen-space vertices** (once, per triangle) vs adding a `SAMPLE_OFFSET=0.5` **inside the per-pixel math** changes Zorah from **74 → 102 ms (~40%)** — same image, pure codegen/register effect of where the constant lives.
- **Mechanism**: on a per-fragment kernel run 36M× per frame, occupancy (how many warps hide memory latency) is set by register count; a single hoisted constant or a cheaper intrinsic that drops the kernel under an occupancy cliff is a giant win. This directly attacks both our 60% and 40%.
- **No-64b-atomic adaptation**: orthogonal.
- **WGSL adaptation**: WGSL gives us no register-count control and no `__fdividef`, BUT the *principle* transfers and is testable on Metal: (i) hoist EVERY constant out of the inner loop (sample offset, screen biases) to the per-triangle prologue — never recompute inside; (ii) minimize live values across the inner loop (fewer locals → fewer registers → more resident SIMD-groups on Apple GPUs, where occupancy is likewise register-bound); (iii) prefer the cheaper reciprocal path. Concretely: audit `NaniteRaster.ts` inner scanline for any value recomputed per-pixel that could be a per-triangle accumulator, and for the 0.5 sample bias being applied per-pixel rather than folded into the snapped vertices. This is the single most "free" class of win in the paper (same image, just code shape) and maps cleanly to "reduce the 40% transform+launch and the 60% loop" without touching the algorithm. MEASURE with the rdbg stage-split.
- **effort**: low (audit + hoist), but the payoff is empirically large and our 36M-fragment loop is exactly the regime where it bites.

### 7. [INDIRECT / future] Nanite-style shared-vertex reuse (the redundant-fetch fix CuRast itself lacks)
CuRast §6 (`:1213-1219`) explicitly contrasts itself with Nanite: "Instead of simply launching one thread per triangle and independently processing potentially shared vertices multiple times **as we do**, [Nanite] first process vertices in a cluster using one thread per vertex, store the results in **shared memory**, then process the triangles using one thread per triangle." CuRast names this as a known throughput gap it does not close.
- **Mechanism / what it attacks**: our 40% transform — the mission notes "~5× redundant vertex re-fetch, ~48 texture taps for wind animation" per triangle. A cluster has ~256 tris but far fewer unique verts; transforming each unique vert ONCE into workgroup shared memory, then having the per-triangle threads read transformed verts from shared memory, removes the 5× redundancy AND (critically for us) collapses the 48 wind-animation texture taps to per-unique-vertex instead of per-triangle-corner.
- **No-64b-atomic adaptation**: orthogonal to election.
- **WGSL adaptation**: HIGHLY portable and likely our biggest transform-side lever. WGSL has `var<workgroup>` shared memory + `workgroupBarrier()`. Restructure the per-cluster workgroup: phase A — lanes 0..numUniqueVerts each fetch+transform+wind-animate ONE vertex, write `vec3 posView` (and any wind-displaced position) to `var<workgroup>`; `workgroupBarrier()`; phase B — lanes 0..numTris each read its 3 corners from shared memory and run coverage. We even already have `NaniteVertexCache.ts` — check whether it does this in shared memory or just dedupes on the CPU/build side. This is the principled fix for the "5× re-fetch + 48 taps" cost the mission calls out, and it's exactly what CuRast says it's missing.
- **effort**: medium-high (workgroup restructure + a vertex-index→shared-slot map per cluster; needs the cluster's unique-vertex list, which a meshlet build typically already provides).

---

## The election: what breaks under no-64-bit-atomics, and the adaptation

CuRast's core is a 64-bit `atomicMin(framebuffer[px], (depth28<<36)|id36)`. **This is precisely the trick WGSL forbids** and is the central question the mission asks of every source. CuRast's own answer (its Related Work, §2.2) is the historical FreePipe workaround, and our team's shipped solution is strictly better than it — important to state plainly:

- **FreePipe two-buffer split** (paper §2.2 `:105-113`, confirmed via web): under 32-bit-only atomics, FreePipe did **two** `atomicMin`s — the SAME 20-bit depth key carrying DIFFERENT 12-bit halves of the color, into two separate buffers, then re-fused the 24-bit color in resolve. **This is NOT a safe payload-election under contention**: the winner of buffer-A's atomicMin and buffer-B's atomicMin can be DIFFERENT fragments when two fragments share the winning depth key (a depth tie, common at 20-bit quantization on dense overdraw) → a torn / Frankenstein payload. It "works" for color where a tie is visually tolerable; it would be WRONG for a triangle-id payload (you'd resolve a nonexistent triangle). **Do not adopt FreePipe's two-buffer split for our id payload.**
- **Our shipped election is the correct adaptation and beats both.** We pack a 24-bit inverted-depth key + 8-bit tiebreak into ONE 32-bit `atomicMax`; the unique winner then plain-stores the 25-bit id (`NaniteRaster.ts:200-207, 740-749`, split visA(payload)+visB). One atomic decides a single winner; the side-store is written only by that winner → no torn payload. This is superior to FreePipe (no tie hazard) and is the WGSL-native equivalent of CuRast's 64-bit single-atomic (same single-winner guarantee, 32 bits).
- **Net**: CuRast offers NO new election idea for us — it confirms that the single-atomic-decides-winner property is what matters, and that the historical 32-bit alternative (FreePipe) is the inferior path we already avoided. The transferable techniques above (#1, #4, #6, #7, #3) are all ELECTION-ORTHOGONAL — they cut coverage/transform/launch cost and apply on top of our existing 32-bit election unchanged. That is the good news: the paper's wins don't depend on the bit-width trick we can't have.

---

## Dead-end confirmations (flag, do not resurface)

- **Occlusion / HZB culling**: CuRast has NONE (static, no HZB) and still wins — consistent with our finding that conservative occlusion is ~0% on holey foliage. CuRast does not contradict our refutation.
- **Pineda / exact in-triangle traversal as a coverage optimization**: CuRast tried it and measured a LOSS on pixel-sized tris (§4, §6). Confirms our instinct to keep the scanline cheap and NOT push toward per-fragment exact clipping.
- **Sub-pixel LOD over-render**: n/a — CuRast has no LOD; not addressed.
- **SW-atomic contention**: CuRast relies on `atomicMin` exactly as we do and reports no contention problem at billions of tris (global atomics implicitly serialize, losers are cheap) — consistent with our refutation that atomic contention is NOT our bottleneck.

## Bottom line for our 20→60fps goal

The two highest-value, lowest-risk, exactly-same-image transfers are **#1 (tiny-tri sample-miss cull, published −27% on the same workload class, attacks both cost centers, a few WGSL lines)** and **#6 (constant-hoisting / occupancy / register-shape — empirically ~40% on Zorah from a single hoisted offset, free)**. The biggest structural transform-side lever is **#7 (shared-memory per-cluster vertex reuse)**, which is the documented fix for our "5× re-fetch + 48 wind taps" 40% cost and is exactly what CuRast admits it lacks. **#3 (persistent work-stealing)** is a maybe (our clusters are uniform; measure first). None of these depend on 64-bit atomics; our existing 32-bit `atomicMax` election is already the correct, contention-safe adaptation and is *better* than the FreePipe alternative CuRast cites.
