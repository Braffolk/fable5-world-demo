# Prior-art brief: LucidRaster (nadult/lucid)

- **Source**: repo + paper. Repo https://github.com/nadult/lucid (shallow-cloned to
  `docs/perf-runs/prior-art/sources/lucid`). Paper "LucidRaster: GPU Software Rasterizer for Exact
  Order-Independent Transparency", K. Jakubowski, arXiv:2405.13364
  (`docs/perf-runs/prior-art/sources/papers/lucid.pdf`, 15 pp).
- **Fetched**: clone OK (`--depth 1`); PDF `curl` from arxiv OK; text extracted with `pypdf` (the
  Read PDF path needs poppler which is absent, so used `pypdf` extraction instead).
- **What it is**: a Vulkan compute-shader **sort-middle** software rasterizer whose goal is *exact
  order-independent transparency* — it keeps and depth-sorts EVERY fragment in a tile, then blends
  front-to-back. License GPLv3.

## TL;DR relevance to our problem

Lucid solves the OPPOSITE of our visibility problem: it deliberately keeps all overdrawn fragments
and sorts them, whereas we want to elect ONE winner per pixel and discard the rest. So its core
"two-stage depth sort + sample accumulation" engine is **not** something we want to port — for
opaque foliage we want to *throw fragments away* as early as possible, not buffer + sort them.

BUT its front-end is a textbook **bin → block-row → block → half-block coarse-to-fine coverage
hierarchy** with a per-row constant-time scanline span generator and a 32-bit-packed
`depth|primitive-index` word — and THAT machinery is directly relevant to both our dominant 60%
per-pixel loop and our 40% transform/launch cost. The key levers are in the front-end, not the OIT
back-end.

Crucially: Lucid never does a per-fragment `atomicMax` depth election. Its 32-bit `depth|index`
packing (22b depth | 10b index) is used for an in-shared-memory **bitonic sort key**, not an
election word. So "how does it map to no-64-bit-atomics?" is mostly moot for the back-end (it has no
election), and for the front-end the packing trick is even friendlier than ours (no atomics at all
on the depth word).

## Architecture (3 stages, all compute)

1. **Quad setup** (`data/shaders/quad_setup.glsl`). Input is **quads, not triangles** (2 tris
   sharing data — saves setup bandwidth). One workgroup of 1024 threads per 1024-quad material
   chunk. Phase 1: cull (degenerate / backface-in-worldspace / frustum / "falls between sample
   centers"), classify small (≤4 bins) vs large, compact survivors in shared mem. Phase 2: compute &
   store per-tri rasterization data into **SoA storages grouped by access pattern** (depth eq;
   barycentric edge eqs; scanline edge eqs + Y-AABB; normals; and per-quad vertex attrs only if the
   material needs them). Paper §"Stage 1", code `processInputQuad` L136, `storeTri` L274.
2. **Binning** (`bin_counter.glsl`, `bin_categorizer.glsl`, `bin_dispatcher.glsl`). 32×32-pixel
   bins. 3 phases: (a) **persistent-thread** workgroups count per-bin primitive overlap (small quads
   counted cheaply by AABB; large tris counted by actually running the *bin-resolution* scanline);
   (b) one workgroup does prefix-sum offsets + **categorizes bins by triangle density** into
   empty/low(<1024 tris)/high(≥1024); (c) re-run the same persistent batches to *write* the per-bin
   primitive index lists, with a subgroup **work-balancing** scheme that splits a wide triangle's
   per-row bin-writes evenly across the subgroup (`dispatchLargeTriBalanced`, L123). Paper §"Stage 2".
3. **Bin rasterization** (`raster_low.glsl` / `raster_high.glsl`, shared in `shared/raster.glsl`).
   Two specialized kernels selected by the density category. Each is **persistent-threaded** (loop
   `loadNextBin` over a global bin work-queue, `raster.glsl` L87). Three sub-phases:
   - **P1 block-row gen** (`generateRowTris`, raster_low L39): for each tri, run the *pixel*-scanline
     to produce, per 32×8 block-row, a coverage record packed in <128 bits (per-row 5+5-bit
     x-intervals = 80b, 4b block-column mask, 24b tri index).
   - **P2 block/half-block extraction** (`generateBlocks`, L81): per 8×8 block, gather the tri-block-
     rows that touch it, compute one **block-centroid depth** per tri, pack `depth(22b)|index(10b)`
     into a 32-bit word, **bitonic-sort front-to-back** in shared memory (`sortBuffer`, L208), then
     emit per-half-block coverage bitmasks + fragment-count prefix sums.
   - **P3 shade & blend** (`shadeAndReduceSamples`/`reduceSample`, `shared/shading.glsl` L229):
     one thread per pixel of an 8×4 half-block; iterate samples *in sorted order*, run a fixed-size
     (3-deep) **depth-filter priority queue** to repair small mis-orderings, blend front-to-back,
     early-out when accumulated alpha→1 (`ALPHA_THRESHOLD`). Paper §"Stage 3".

Measured (paper, AMD 6700XT): SW is ~3.3× HW alpha-blend on average; raster stage dominates on
dense/foliage scenes (`docs/notes.wiki` perf tables: white_oak/hairball spend ~95% in `raster_*`).

---

## Transferable techniques

### DIRECT ports

**T1. Coarse-to-fine coverage hierarchy: bin (32×32) → block-row (32×8) → block (8×8) →
half-block (8×4), each producing a coverage BITMASK, never a per-pixel loop.**
`shared/raster.glsl` `rasterBinStep` L116, `rasterHalfBlockBits` L152; paper §"Triangle
rasterization", §"Stage 3 Phase 1/2". This is the single most relevant idea for our DOMINANT 60%
per-pixel loop. Lucid never iterates pixels inside a triangle: the scanline gives, per pixel-row, an
`[xmin,xmax]` interval in **constant time** from precomputed 3D edge functions; coverage becomes a
small set of integer-range → bitmask conversions (`blockRowsToBits` L272, `rasterHalfBlockBits`
L152). A covered fragment is materialized only as a set bit, and the only per-bit work in the inner
loop is `findLSB`/clear (`unpackSamples` L321). Our inner loop currently re-derives coverage with 3
integer edge tests + a barycentric-z + a relaxed atomic load *per candidate pixel*; Lucid computes a
covered-pixel **bitmask per 8-wide row** (or per 8×4 half-block) once and then only visits set bits.

> Adapt (no-64-atomic / WGSL): fully portable — no atomics involved in coverage at all. We already
> shipped an x-span scanline (commit 19eb834); the bigger win Lucid demonstrates is going one level
> further: emit a **per-row coverage bitmask** (e.g. a `u32` per 32-px row, or `u8` per 8-px block
> row) and iterate set bits, so the per-fragment cost drops to one `findLSB`+clear instead of
> recomputing edge tests. Our quality is preserved because coverage is identical; only the
> *enumeration* changes. The mask also lets us cheaply count fragments before doing any
> depth/election work.

**T2. Packed `depth|index` in a single 32-bit word (22b depth | 10b index), depth inverted so
sort/compare is monotonic.** `raster_low.glsl` L129 (`rasterBlockDepth` → `depth<<10 | tri_idx`),
paper §"Stage 3 Phase 2". Lucid uses it as a SORT key; we use the same idea as an **atomicMax
election key**. This *confirms* our exact packing strategy is the standard no-64-bit-atomic move and
shows the bit budget others find sufficient (22b depth here; we use 24b+8b tiebreak). Confirmation,
not a new lever — but it validates that 22–24 bits of depth is enough to disambiguate dense foliage.

> Adapt: we already do this. Lucid's choice of a **block/region-centroid depth** (one depth per
> triangle per 8×8 block, not per pixel — `rasterBlockDepth` evaluates the depth eq at the coverage
> centroid `cpos`) is the interesting variant: it computes depth ONCE per (tri,block) instead of per
> fragment. For us, electing per-pixel is mandatory (we need correct per-pixel nearest), so we can't
> use a single block depth as the election value — BUT we could use a cheap *conservative block depth*
> as a pre-pass reject (see T6).

**T3. SoA "storages grouped by access pattern", filled once in setup, including optional per-quad
vertex attributes only when the material needs them.** `quad_setup.glsl` `storeTri` L274 /
`storeQuad` L256, `shared/definitions.glsl` STORAGE_* offsets L132. Setup computes the depth eq, two
barycentric edge eqs, the scanline edge eqs, and the normal, and packs them into separate tightly
encoded buffers; the rasterizer/shader later reads exactly the lanes it needs. This directly attacks
our 40% transform+launch cost, where we re-fetch each vertex ~5× and re-run wind animation (~48
texture taps) per triangle.

> Adapt (WGSL): precompute per-visible-triangle a small struct {inverted-depth plane eq (3 floats),
> 2 edge eqs, packed normal} ONCE in a setup pass and store it in a buffer, so the raster kernel
> never re-fetches/re-animates vertices. This is the standard Nanite/Lucid split and is fully
> portable. The win is largest for us specifically because our per-triangle cost includes expensive
> wind-animation texture taps that are pure redundant recompute today.

**T4. Constant-time scanline span from precomputed 3D edge functions (Davidovič 3D rasterization),
shared between a coarse "bin-resolution" variant and a fine "pixel-resolution" variant.**
`shared/scanline.glsl` (`loadScanlineParamsRow` L13 / `loadScanlineParamsBin` L28),
`bin_dispatcher.glsl` `scanlineStep` L55, paper §"Triangle rasterization". The same precomputed edge
data drives a cheap bin-granular span (for binning/coarse reject) and a pixel-granular span (for
final coverage). 3D rasterization means **no near-plane clipping is needed** — only screen AABBs —
which simplifies setup.

> Adapt: portable. Two payoffs: (a) a coarse bin/tile-granular scanline lets us reject whole tiles
> for a cluster cheaply before the fine loop; (b) the "no clipping, just AABB via Blinn's screen-
> coverage" path (`quad_setup.glsl` `computeClippedAABB` L77) can simplify our setup. The 3D
> (homogeneous) edge formulation also gives perspective-correct depth/bary from the same eqs we
> already need, avoiding extra interpolation math.

### INDIRECT ports (adapt-the-principle)

**T5. Two density-classed kernels (low vs high) selected per tile, plus dynamic promotion.**
`bin_categorizer.glsl` `categorizeBins` L57 (empty / <1024 / ≥1024); `raster_low.glsl` overflow →
`s_raster_error` promotes the bin to the high-rasterizer queue (L230, L295). The cheap kernel uses
tight shared-mem limits (256 tris/block, fixed scratch) and is the common case; the rare dense tile
spills to a heavier kernel.

> Adapt: our foliage is *uniformly* dense/overdrawn so a simple low/high split may not map directly,
> BUT the principle "classify tiles by triangle/fragment count, then run a specialized inner loop"
> is exactly how to give the worst-5% (p0.05) tiles a different, bounded code path. Concretely: bin
> the screen, count per-tile triangles, and route hot tiles (the ones that blow our frame budget) to
> a kernel that uses the bitmask/coverage hierarchy (T1) while letting sparse tiles use the current
> path. The **promotion-via-error** mechanism (try cheap, fall back) is a clean way to keep the
> common path fast without a separate counting pass. Portable; no atomics issues.

**T6. Per-(tri,region) centroid depth computed once, used to sort/early-cull whole coverage blocks
before touching pixels.** `raster_low.glsl` L129–133; paper §"Phase 2". Lucid sorts tri-blocks
front-to-back by this single depth so it can early-out on alpha; the same single-depth-per-block can
serve as a **conservative occluder test** for opaque geometry.

> Adapt: this is the one idea that *might* dodge the refuted HZB-occlusion dead end. Our HZB occlusion
> is ~0% because holey crowns pin max-Z to the far plane. Lucid's angle is different: compute, per
> cluster-per-tile, the *nearest* representative depth, sort clusters front-to-back within a tile,
> rasterize near→far, and once a tile's per-pixel election word is "full/settled" you can skip
> clusters that are entirely behind the current per-pixel front. This is *front-to-back occlusion
> within a tile against the live election buffer*, not against a precomputed conservative HZB — so a
> gap does NOT pin anything to the far plane. Caveat: to stay quality-preserving it must test against
> the actual per-pixel election depths (read-back the packed word), not a quantile; that's a per-tile
> shared-memory min/coverage test, doable in WGSL with `atomicLoad` on the packed word (relaxed). Risk:
> the bookkeeping may cost more than it saves on ~1px tris — treat as a measure-first experiment, not
> a sure win. Flag: this is adjacent to the refuted "conservative occlusion" — the difference is
> *front-to-back ordering against the live buffer* vs *static conservative HZB*; only the former
> survives holey foliage.

**T7. Persistent-thread bin work-queue (spawn ~#cores workgroups, each loops `atomicAdd`-claims the
next bin) instead of one-dispatch-per-primitive-group.** `shared/raster.glsl` `loadNextBin` L87,
paper §"Stage 3". Compare to our "one workgroup per visible cluster" (~154k workgroups). Persistent
threads avoid launch overhead and give natural load balancing across uneven tiles.

> Adapt: directly attacks our 40% transform+launch. Instead of 154k cluster-workgroups, spawn a fixed
> pool that pulls clusters (or tiles) from a global queue via `atomicAdd` on a counter — fully
> portable in WGSL (32-bit `atomicAdd`). Worth noting our clusters are well-filled (235/256) so the
> *idle-lane* gain is small, but the *launch/scheduling* overhead reduction and better cache reuse
> (a workgroup that processes several adjacent tiles in a row reuses vertex/edge data) is the lever.
> Lucid's setup also processes work in **batches sized to amortize per-batch overhead** (small quads
> in big batches, large tris one-per-thread) — same principle: pick batch granularity by per-item
> cost.

**T8. Subgroup work-balancing for wide primitives (split a primitive's per-row output evenly across
the subgroup via shuffles).** `bin_dispatcher.glsl` `dispatchLargeTriBalanced` L123 + `sortBuffer`
subgroup-shuffle bitonic sort (`shared/raster.glsl` L203 `swap`/L208). Indirectly relevant: our tris
are tiny (~1px) so per-row balancing is moot, but the **subgroup-shuffle primitives** (ballot,
shuffle, inclusive-add) used throughout for compaction/prefix-sum/sort are the building blocks for
any binned approach we adopt.

> Adapt: WGSL `subgroup*` builtins exist (Chrome behind a flag / `enable subgroups;`). If we adopt
> tiled binning we'll need subgroup prefix-sum for per-tile fragment offsets exactly like Lucid's
> `subgroupInclusiveAddFast` (`bin_categorizer.glsl` L26). Portable but availability-gated; have a
> shared-memory scan fallback.

---

## Dead-end confirmations (no new ideas; flag only)

- **HZB conservative occlusion stays dead for opaque holey foliage.** Lucid never relies on it (it's
  a transparency rasterizer). The only occlusion-flavored idea here (T6) explicitly works
  *front-to-back against the live election buffer*, NOT against a static conservative max-Z, which is
  the precise reason a leaf-gap doesn't pin anything to the far plane. So Lucid neither resurrects nor
  refutes our HZB dead end — it sidesteps it.
- **Sub-pixel LOD over-render**: N/A here; Lucid renders all input geometry exactly.

## Things that are NOT portable / NOT wanted for us

- The entire **OIT back-end** (depth-sort all fragments + 3-deep depth-filter priority queue +
  front-to-back accumulate, `shared/shading.glsl` `reduceSample` L229) is the wrong tool for opaque
  visibility — it *retains* overdraw instead of eliminating it. For opaque foliage we want a winner-
  take-all election; keeping/sorting 12–20× fragments per pixel would be strictly worse. Do not port.
- **Bitonic sort of fragments in shared memory** (`sortBuffer` L208) is only needed because of OIT
  ordering; opaque election needs no sort.
- **64-bit atomics**: Lucid uses NONE — its `depth|index` 32-bit word is a sort key, not an atomic
  election. So there is no 64-bit-atomic dependency to work around here; if anything Lucid confirms
  our 32-bit packed-atomicMax approach is the right no-64-bit path.
