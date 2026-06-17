# UE5-Nanite gap report — LAAS WebGPU nanite renderer, frame 1286

**Operating point:** forest full-pipe, canvas 1101×1450, ~21 ms gpuWall, **raster-bound**.
`nanRasterWorld1` (cmd#378, INDIRECT 4594) = **16.7 ms = 75%** of the frame
(DIVERGENCE-REPORT.md:414). The remaining ~25% / ~5 ms is the cull BFS (cmd#26→#344, 18
ping-pong passes), the HW needle draw (cmd#397, `nanHwPass`), the HZB rebuild (cmd#408,
built AFTER raster for *next* frame), the resolve übershader (cmd#454, `NodeMaterial_69`,
shader 4749), and the bloom/post tail (submits 51-65).

**The stance holds.** UE5 renders denser geometry faster, so "everything is optimized, only
small wins remain" is false. But this round is **non-quality-loss only** — the big
geometry-volume levers UE5 wins on (cross-instance cluster merge / coarse-DAG far clusters,
per-cluster position quantization, offline HLOD proxies) are **build-pipeline changes that
either touch quality or are large projects** (Section 4). What is left that is genuinely
free this round is a set of **per-pixel-loop micro-architecture** wins inside the 16.7 ms
kernel plus two small tail cleanups. They are real, but they sum to **~3.7 ms (~17% of the
frame)**, not a 2× — and the honest conclusion is that *the structural win UE5 has and we
don't is the cluster-count / overdraw floor, which is out of scope for "same image, faster."*

> **HARD HONESTY:** the capture has **no GPU timestamps** (timeline.json has zero timing
> fields). Every per-win ms below is *derived* from the one measured anchor (16.7 ms / 75%,
> DIVERGENCE-REPORT.md:414) + the in-code PERF-3 ablation (NaniteRaster.ts:224-225: the
> per-pixel loop is "≈90% of world1") + the project's prior measured decompositions
> (NANITE-LOG.md). They are defensible estimates with stated discounts, not measurements.
> The #1 win must be confirmed by the falsifiable test in Section 5 before trusting the ms.

---

## 1. Executive summary — the verified non-quality-loss wins

| # | Win | real_win_ms | UE5 technique | one-line impl |
|---|-----|-------------|---------------|---------------|
| 1 | **Scanline x-span instead of full-bbox edge test** in the SW raster inner loop | **~2.5** | Nanite SW raster computes a per-row covered x-interval from edge functions, not a full-bbox test (Karis SIGGRAPH 2021: "between a linear DDA and looping over all pixels in the bounding box") | replace the inner full-width `loopI('sx')` with a per-row `[xL,xR]` computed from the 3 edge values + their per-pixel x-steps; keep the exact top-left test inside the span |
| 2 | **Incremental depth DDA** instead of per-pixel divide-by-area barycentric recompute | **~0.5** | Nanite computes dz/dx, dz/dy once per triangle in setup, increments depth per pixel (a depth DDA) | derive `dzdx`/`dzdy` once in setup from `sx_i`/`sy_i`·`ndc_i.z`·`rcpArea`; carry `cz` per column / `czRow` per row instead of the 3-term dot product |
| 3 | **Same-frame coarse Hi-Z micro-pass** to pre-cull residual occluded clusters before the heavy raster | **~0.5** | UE5 two-pass main/post: build a fresh HZB mid-frame and re-test clusters against it (candidstartup, confirmed) | split emitted clusters NEAR/FAR by `distC`; raster NEAR, build a cheap partial HZB, conservatively pre-cull the FAR slice before its raster dispatch |
| 4 | **Gate `buildTerrainShading` behind `If(isT)`** in the resolve | **~0.15** | Nanite deferred material pass evaluates each pixel's material exactly once, touches no occluded resource (Wihlidal GDC 2024) | wrap the terrain prelude (3 tex + ~10 ALU) in `If(isT, …)` exactly like the existing `If(isR)`/`If(isBD)`/`If(isL)` branches |
| 5 | **Cache `makeCtx` in the HW vertex pass** (recomputed 3× per triangle, uncached) | **~0.04** | Nanite transforms each unique vertex once into shared memory, then 1 thread/triangle reads the cache | precompute the per-HW-item ctx into `hwQueue` during the SW pass that already holds it |

**Realistic achievable sum: ~3.7 ms** (~17% of the 21 ms frame; ~22% of the 16.7 ms raster
since wins 1+2 = ~3.0 ms land inside it). Wins 1+2 are the load-bearing pair — both are pure
WGSL integer/float arithmetic inside the dominant kernel, no platform feature needed, and
together they remove the largest per-pixel ALU redundancy. Win 3 is the only one that
attacks *triangle volume* rather than per-pixel cost, but its magnitude is capped by the
max-depth-HZB-occludes-nothing-on-gaps problem (Section 2.3). Wins 4+5 are near-free
cleanups worth taking opportunistically.

**Why this is not "everything is optimized":** wins 1+2 prove the hottest kernel in the
renderer is doing ~half its inner-loop pixel work on guaranteed-empty pixels and
recomputing an affine quantity from scratch every covered pixel — both are exactly what
UE5's shipped rasterizer avoids. The renderer is *not* at the per-pixel-loop floor. It IS
close to the floor for *this geometry volume*; the volume itself is the bigger UE5 gap
(Section 3, Section 4) but cannot be closed without a build-pipeline change.

---

## 2. Per-win detail

### Win 1 — Scanline x-span instead of full-bbox edge test  ·  ~2.5 ms  ·  effort M  ·  quality: equivalent  ·  feasible: yes

**Our approach.** `NaniteRaster.ts:657-748` — `loopI('sy', startY..endY){ loopI('sx',
startX..endX){ If(cw0>=0 && cw1>=0 && cw2>=0, covered-body); cw0.addAssign(sx0);
cw1.addAssign(sx1); cw2.addAssign(sx2) } rw_i.addAssign(sy_i) }`. The capture is byte-for-byte
this loop: **4622.wgsl:1298-1353** (cmd#378 `compute_nanRasterWorld1`, INDIRECT 4594) — outer
`for(sy)` at 1298, inner `for(sx)` at 1304, 3-edge sign test at 1307, covered body 1309-1337,
3 incremental int adds 1343-1345, row adds 1349-1351. **Every pixel in the AABB rectangle
pays 3 i32 compares + 3 i32 adds + a branch, even the ones outside the triangle.** A triangle
covers ~half its tight AABB, so roughly half the visited pixels are guaranteed-empty.

**UE5 approach.** The SW rasterizer's inner loop is, per Karis SIGGRAPH 2021, "a lightweight
algorithm somewhere between a linear DDA and looping over all pixels in the bounding box and
checking if each is inside the triangle" — i.e. **explicitly not** the full bbox test. The
shipped form (UWA / candidstartup transcriptions of `NaniteRasterizer.ush`) iterates rows of
the screen-space AABB and per row computes the **x-interval of overlapped pixels** from the 3
edge functions.
Sources: <https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf> ;
<https://courses.grainger.illinois.edu/CS418/sp2023/text/nanite.html> (verbatim transcription of the Karis quote).

**Mechanism.** For row `y`, the 3 edge values `cw_i` (already maintained incrementally:
`rw_i.addAssign(sy_i)` per row, `NaniteRaster.ts:745-747`) are affine in `x` with slope `sx_i`
(the per-pixel x-step, already built at `NaniteRaster.ts:627-629` / WGSL 1290-1292). The
covered span is `cw_i + (x-startX)·sx_i >= 0` for all `i`. Each edge with `sx_i>0` gives a
lower bound, `sx_i<0` an upper bound. `xL = max(startX, lower bounds)`, `xR = min(endX, upper
bounds)`. Iterating only `[xL,xR]` skips the guaranteed-empty pixels with a handful of integer
ops per row, amortized over up to 16 columns (bbox ≤ `MAX_RASTER_SIZE=16`, `NaniteRaster.ts:90`).

**Impl sketch.** In `NaniteRaster.ts:657`, inside the row body, before the column loop:
compute per-edge bounds from `cw_i` (row-start value) and `sx_i`, fold into `xL`/`xR`, then
`loopI('sx', xL, xR, …)` running the **same** covered body. Keep the per-pixel
`cw0>=0 && cw1>=0 && cw2>=0` test inside the span so the top-left rule still decides edge
pixels exactly — the span is only a conservative fast-skip of pixels the test would reject
anyway.

**Quality.** Bit-identical. The exact per-pixel coverage test, fixed-point snap, depth math,
and the `atomicMax`/`atomicStore` election are all untouched; only guaranteed-empty pixels are
skipped. Not a coarsening.

**Feasibility.** Pure WGSL i32 arithmetic. No `atomic<u64>` (WGSL has none — confirmed:
statistics.json adapter features list has `subgroups` but no 64-bit atomics), no subgroup ops,
no persistent-thread forward progress. The kernel already uses only 32-bit
`atomicLoad`/`atomicMax`/`atomicStore`.

**Win size — derived, discounted from the raw 4.5 ms claim.** The raw proposal's 4.5 ms
assumed empties run on ~2× the covered pixels; at its own stated 50% coverage empties ==
covered (1×), and it ignored the new per-row span-setup cost. Re-derived: per-empty ≈ 6 ops
(3 cmp + 3 add), per-covered ≈ 25 ops + atomic traffic. At 50% coverage, eliminating empties
saves ≈ 3N/15.5N ≈ 19% of ~15 ms ≈ 2.9 ms, minus span-setup, and discounted because the empty-
pixel ALU has zero memory traffic and is partly hidden under the covered pixels' atomic
latency (the loop is likely atomic/latency-bound, not pure-ALU-bound). **Defensible ≈ 2.5 ms.**

**Caveat on the UE5 citation:** the most concrete WebGPU/Nanite reference originally cited
(Scthe/nanite-webgpu `rasterizeSwPass.wgsl`) actually uses the *same* full-bbox double loop we
do — so "the reference impls all do scanline" would be overstated. The technique is legitimate
(Karis deck + UWA transcription of the shipped `.ush`); the win stands on its own arithmetic
regardless of any one reference impl.

---

### Win 2 — Incremental depth DDA instead of per-pixel barycentric recompute  ·  ~0.5 ms  ·  effort S  ·  quality: equivalent (NOT bit-identical)  ·  feasible: yes

**Our approach.** `NaniteRaster.ts:679-687` / WGSL 4622.wgsl:1309-1312: per covered pixel,
`uw_i = cw_i - bias_i` (3 int subs) then `cz = (f32(uw0)·ndc0.z + f32(uw1)·ndc1.z +
f32(uw2)·ndc2.z) · rcpArea` (3 int→f32 casts + 3 mul + 2 add + 1 mul). A full barycentric
depth reconstruction **from scratch every covered pixel** inside the inner loop.

**UE5 approach.** Compute the depth gradient (dz/dx, dz/dy) once per triangle in setup and
increment depth per pixel — a depth DDA, consistent with Karis's "linear DDA" framing of the
inner loop. Source: <https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf>.

**Mechanism.** `uw_i = cw_i - bias_i` is affine in (x,y); `ndc_i.z` and `rcpArea` are per-
triangle constants. Therefore `cz` is affine in (x,y): `dzdx =
(sx0·ndc0.z+sx1·ndc1.z+sx2·ndc2.z)·rcpArea`, `dzdy` analogous, both per-triangle constants.
The DDA (`cz += dzdx` per column, `czRow += dzdy` per row) replaces ~9 float ops + 3 int→f32
casts per covered pixel with **1 add**.

**Impl sketch.** In setup near `NaniteRaster.ts:626-633` (where `sx0..sy2` and `rcpArea` are
built): add `dzdx`, `dzdy`, and `cz0` (depth at `(startX,startY)` via the existing unbiased-
weight formula, once). In the row loop carry `czRow.addAssign(dzdy)`; in the column loop carry
`cz` (init `czRow`, `cz.addAssign(dzdx)`). Replace the in-loop reconstruction (679-687) with
the carried `cz`. Keep the `cz in [0,1]` guard.

**Quality — equivalent, NOT bit-identical (the raw proposal overstated this).** world1 stores
only a 24-bit depth key: `depthKey24(cz)` then `<<8 | (payload&0xff)` (`NaniteRaster.ts:721`);
there is no exact `depthV` in this path (the 3rd atomic buffer is the documented 3× Metal cliff,
`NaniteRaster.ts:730-733`). A float DDA accumulating `cz += dzdx` over up to 16 columns is not
bit-identical to the fresh 3-term dot (f32 add is non-associative, and the dot mixes three
distinct f32 `ndc_i.z` constants so the in-progress sum is float). BUT the stored value is
already 24-bit quantized, so a few-ULP `cz` delta lands in the same bucket on the overwhelming
majority of pixels; the only deviation is a rare 1-bucket depth shift (1/16.7M of NDC depth,
sub-pixel) or an election tiebreak flip on near-equal-depth fragments — the exact "sparse
wrong-material speckle" the code comment at `NaniteRaster.ts:718-720` already accepts as
in-tolerance. **Below the existing quantization noise floor. The implementer must NOT claim
bit-identity — verify with `?audit=1` + a pixel diff to confirm the deviation stays at the
sparse-speckle level the code already tolerates.**

**Feasibility.** Pure f32/i32. No platform feature.

**Win size — derived, trimmed from 0.8 ms.** The reconstruction runs only on covered pixels
(~50-60% of iterated bbox pixels for ≤16px tris). The inner loop is ~90% of 16.7 ms
(NaniteRaster.ts:224-225), but a large share of per-covered-pixel cost is the contended global
`atomicLoad`/`atomicMax`/`atomicStore` election, which ALU savings can't recover. Optimistic
ALU ceiling 0.12·0.55·0.85·16.7 ≈ 0.9 ms; discounting for atomic/memory-bound stalls and the 1
retained add gives **~0.5 ms**.

---

### Win 3 — Same-frame coarse Hi-Z micro-pass to pre-cull residual occluded clusters  ·  ~0.5 ms  ·  effort L  ·  quality: none  ·  feasible: partial

**Our approach.** Single raster pass `nanRasterWorld1` (cmd#378) rasters every emitted
cluster's full 128-lane workgroup (`NaniteRaster.ts:764` `.compute(QRASTER_CAP*128,[128])`).
The HZB (`nanHzbL10`, cmd#408) is built **after** raster, only for next frame. Occlusion is
tested **once**, at cull-emit time, against the **previous** frame's HZB
(`NaniteCull.ts:507-513` `sphereOccluded(…, cam.prevVp, cam.prevCamPos)`). The per-pixel
election prefetch (`NaniteRaster.ts:725` `prevE=aLoadU(visPayloadV); If(cand>prevE,…)`) only
skips the atomic **write** — it can never dodge the per-triangle vertex fetch+transform+setup,
which is the part pre-cull recovers.

**UE5 approach.** Two-pass main/post: confirmed verbatim from candidstartup —
*"In the first pass instances and then clusters are tested against the HZB from the previous
frame. Visible clusters are rasterized. The HZB for the current frame is built based on what
was just rasterized. In the second pass, all the instances and clusters found to be occluded
based on the previous HZB are retested with the current HZB."*
Source: <https://www.thecandidstartup.org/2023/04/03/nanite-graphics-pipeline.html>. UE5 uses
this for disocclusion **add-back**, but the same fresh-HZB infrastructure is what enables
culling against just-rasterized near geometry.

**Mechanism / impl sketch.** Split the emitted `qRaster` set NEAR/FAR by `distC` (already
computed, `NaniteCull.ts:459-460`). Raster the NEAR slice first; run a cheap coarse depth
reduction over just the near-slice vis depth into a small (e.g. 1/8-res) same-frame HZB (reuse
`buildNaniteHzb`); then a lightweight per-cluster `sphereOccluded` (current `cam.vp`/`camPos`
vs the partial HZB) over the FAR slice to compact it before its raster dispatch.

**Quality.** None. The conservative `nearestZ > maxZ` test drops only clusters provably behind
same-frame near geometry — the same bit-exact guarantee as the existing prev-HZB cull
(`NaniteHzb.ts`). No coarsening, no dropped visible geometry.

**Feasibility — partial (honest).** No 64-bit atomics/subgroups/forward-progress needed;
reuses `buildNaniteHzb` + split indirect args + a compaction kernel, all 32-bit storage-buffer
compute. The 3× Metal cliff (adding a 3rd atomic buffer to the raster hot loop) does NOT apply
— this adds NO buffer to the raster kernel; the compaction/HZB are separate kernels. Cost is
added passes + the NEAR/FAR dispatch split + an extra mid-frame HZB build. This is the
ROADMAP's own listed-but-unbuilt "same-frame cluster Hi-Z (overdraw)" candidate.

**Win size — derived, cut hard from 1.6 ms to ~0.5 ms.** Two reasons the magnitude is capped:
(1) the HZB is a **max-depth (farthest) pyramid where empty = far-plane = 1.0 occludes
nothing** (`NaniteHzb.ts:4-8`: "each level the 2×2 max (farthest, classic depth)", "empty
0xffffffff → 1.0 far", "Initial fill 1.0 occludes nothing"). A partial HZB from only a sparse
NEAR slice leaves most footprint texels = far, so `maxZ = 1.0` and the conservative test culls
nothing there — and the project's own LOG measured the proper conservative `nearestZ>maxZ`
same-frame test culling ~0% (the loosest diagnostic only 15%). (2) The per-pixel fill of
occluded clusters is **already largely skipped** by the existing prefetch; only the per-cluster
setup (~59%) is pre-cull-recoverable. Re-derived at a defensible ~5% cull fraction:
16.7·0.05·0.59 ≈ 0.49 ms gross − ~0.4 ms (build+compaction) ≈ **~0.3-0.5 ms net**. UE5's
post-pass purpose is disocclusion add-back and is "a small fraction of the main pass" in
normal navigation — evidence *against* a large residual-removal win.

---

### Win 4 — Gate `buildTerrainShading` behind `If(isT)`  ·  ~0.15 ms  ·  effort S  ·  quality: none  ·  feasible: yes

**Our approach.** `NaniteResolve.ts:295` calls `buildTerrainShading(...)` **unconditionally**
at the top of the fragment; `matClass`/`isT` are known at 287-291. Rock (`If(isR)`, 330),
bark/deadwood (`If(isBD)`, 374), leaf (`If(isL)`, 521) are each already branch-gated — terrain
alone is not; its output is consumed only via `isT.select` at 567-574. The resolve fragment IS
shader 4749 (pipelines.json pipe 4751 = `renderPipeline_NodeMaterial_69` → fragment 4749;
cmd#454; 170666 B).

**UE5 approach.** Nanite shades deferred from the vis buffer with material classification:
"each material draw ignores pixels that don't have the desired material" and "no
textures/buffers are accessed that are later occluded" — a terrain shader never runs on a leaf
pixel. Sources: Wihlidal "Nanite GPU-Driven Materials" GDC 2024;
<https://www.thecandidstartup.org/2023/04/03/nanite-graphics-pipeline.html>.

**Impl sketch.** Wrap the terrain block in `If(isT, () => {…})` like the existing `isR`/`isBD`/
`isL` branches; hoist `terrainCol`/`terrainRough`/terrain worldNormal into `.toVar()` defaults
declared before the branch, assign inside. Move the `causticContext()` adjustment inside the
same `If(isT)` (caustics only apply to terrain). Keep the `isT.select` final composite
unchanged. One-branch change, no new buffers/bindings.

**Quality.** None — non-terrain pixels already discard the terrain output via `select`, so
gating it is bit-identical.

**Feasibility — yes, but the raw proposal's stated *reason* was wrong.** It claimed terrain
uses explicit-LOD so derivatives aren't needed; actually `TerrainMaterial.ts:117-137` uses
auto-mip `texture()` and shader 4749 lines 1383/1390 are auto-mip `textureSample`. It works
anyway because three.js emits `diagnostic(off, derivative_uniformity)` (4749 line 4) and the
shipped capture ALREADY runs 22 auto-mip `textureSample` calls inside `if(isT)` (non-uniform
control flow) successfully — so moving 3 more samples in is provably fine.

**Win size — derived, cut hard from 0.7 ms to ~0.15 ms.** The TSL compiler ALREADY lowered the
select-consumers into conditional blocks: the bulk of terrain shading (the 16-texture albedo
mix-tree + 6-texture worldNormal) is already gated inside `if(nodeVar954)` (= `isT`) at shader
4749 lines 3350-3441. The genuinely **unconditional** terrain work is only the prelude at
4749:1382-1395 — **3 texture ops + ~10 ALU** feeding vars consumed only inside the isT blocks.
So the hoistable waste is ~5-10× smaller than the raw "15-tex/90-op" claim. The resolve lives
in the ~5 ms non-raster tail; saving 3 tex + ~10 ALU on the ~65% non-terrain pixels ≈ resolve
(≤2 ms) · ~0.12 prelude-fraction · 0.65 ≈ **~0.15 ms**. Small, free, take it.

---

### Win 5 — Cache `makeCtx` in the HW vertex pass  ·  ~0.04 ms  ·  effort M  ·  quality: none  ·  feasible: partial

**Our approach.** `NaniteRaster.ts:817-841` (`buildHwMaterial` vertexNode): per vertex it
derives triIndex/corner, reads payload, then `const ctx = makeCtx(instId, ci)`
**unconditionally per vertex = 3× per triangle**, recomputing identical per-cluster ctx
including gated wind gust texture samples. The SW path hoists `makeCtx` via wgcache (thread-0
broadcast, `NaniteRaster.ts:336-381`); the HW path does not — confirmed by the project's own
`NANITE-LOG.md:764-765` ("the wgcache win does NOT help the HW pass").

**UE5 approach.** 1 thread/vertex transforms each unique vertex once into shared memory, then 1
thread/triangle reads the cache (cs418 nanite transcription of the Karis Deep Dive). Applies to
Nanite's **compute** SW rasterizer — which LAAS already mirrors via wgcache in its SW path —
NOT to a HW vertex-pulling fallback, so this doesn't elevate the HW-path win.

**Impl sketch.** HW vertex stages can't share workgroup memory across a draw, so precompute the
per-HW-item ctx into `hwQueue` during the SW pass (which holds the broadcast ctx when it
enqueues, `NaniteRaster.ts:753-758`): store the few ctx scalars alongside `(payload,instId)` so
the HW vertex shader reads them instead of recomputing `makeCtx`. Costs queue memory
(`HW_CAP=2,097,152`, so widening 2→~12 u32/entry ≈ +80 MB — non-trivial).

**Quality.** None — `makeCtx` is deterministic; the f32 round-trip is exact (the SW wgcache
already depends on this).

**Win size — derived, cut from 0.3 ms to ~0.04 ms.** The HW pass is **fragment-bound** (big
near-camera tris, heavy overdraw), not vertex-bound — `NANITE-LOG.md:817,824` measured the HW
pass at 2.62 ms dominated by fill. `makeCtx` redundancy here is only 3× (per-corner), not the
128× the SW path had. At the SW path's measured ~0.044 ns/makeCtx-invocation, 130k×3 = 390k HW
invocations ≈ 0.017 ms total; removing 2/3 ≈ 0.011 ms, ceiling ~0.05 ms even at higher per-
vertex cost. **In frame 1286 specifically the HW pass is a single small `drawIndirect` over a
light scene (statistics.json totalTriangles = 2001), so the realized win is well under
0.02 ms** — the ~0.04 figure is generous for a denser operating point. Lowest priority;
win 1 also shrinks the HW set this optimizes.

---

## 3. N8-HIC verdict (cross-instance cluster merge into far super-clusters)

**Does the no-quality-loss claim hold? NO — it is `quality: risk`, and that is why it is NOT
in the verified set.** N8-HIC merges the FAR bark/rock/deadwood tail across instances into
shared super-clusters. The renderer's per-instance forest floods the frontier (~335k visCl in
the testbed, LOG) because **each instance seeds its own roots and never merges with
neighbors** — this is the renderer's single biggest known geometry-volume lever (ROADMAP
N8-HIC / D-N43), and the root cause of the cluster-count / overdraw floor that wins 1+2 can
only paper over.

But merging cross-instance geometry **cannot preserve per-instance variation for free**: our
tint/wind is procedural per-instance (`slotHash`) and cannot survive a geometry merge without
**baking the variation into per-super-cluster vertex attributes** — i.e. a build-pipeline
change that bakes a specific approximation of each instance's appearance. That is a quality
trade (the merged proxy is an approximation of N distinct instances), and it requires a
cross-instance super-cluster *builder* that does not exist. The raw estimate was 1.6 ms at
`quality: risk, conf: low`.

**Ranking:** as a *raw upside* it is the largest single number on the board (~1.6 ms,
potentially more in denser views) and addresses the *real* UE5 advantage (Section 4). But it
**fails this round's non-quality-loss bar** and is a **build-pipeline project, not a kernel
edit**. It ranks **below all 5 verified wins for THIS round** (which require "same image,
faster") and **above all of them as the next-quarter structural bet** once a quality budget is
allowed. It is the honest answer to "what does UE5 do that we structurally don't": UE5's
coarse-DAG far clusters cover many trees with few tris (overdraw ~1 tri/px); ours stay above 1
tri/px even after banded-τ (LOG: flood was ~16 tris/px vs reference ~1).

---

## 4. UE5 gaps — what UE5 does that we do not do at all (structural, bigger projects)

These are NOT this round's wins (each is either a quality trade, a build-pipeline project, or
categorically unavailable in WebGPU), but they are the honest answer to "UE5 is faster on
denser scenes":

1. **Cross-instance cluster MERGE / coarse-DAG far clusters (N8-HIC).** UE5's whole distance
   advantage: a few large clusters cover many trees instead of thousands of tiny under-filled
   per-instance clusters. **Biggest upside; build-pipeline; quality-bake required** (Section 3).

2. **Per-cluster position QUANTIZATION to a local grid** (few bits/component, bitstream-decoded
   with `ldexp` + cluster center). We store full **24 B/vert f32** (id4617 = 894.7 MB) and
   re-transform full-precision verts per cluster. UE5's headline bandwidth/VRAM win (~6 B/vert)
   that also shrinks per-vertex raster fetch. Build-pipeline change with decode cost; a
   bandwidth/VRAM lever, not the primary frame-time lever for this per-pixel-loop-bound frame.

3. **Cluster-local u8/u16 triangle indices.** Our index blob is 1 full u32/index (id4616 =
   278.4 MB). UE5 packs 128-tri clusters at ~17 bits/triangle. Memory win, marginal frame-time;
   gated behind the vertex-cache restructure.

4. **Fused 64-bit `InterlockedMax` (depth<<32 | payload)** for a single visibility write.
   **Categorically unavailable** — WGSL has no 64-bit atomics (gpuweb #5071; confirmed: the
   frame-1286 adapter features list has `subgroups` but no 64-bit atomic feature). We are
   forced into the 32-bit election + side-buffer split (visPayloadV + visBV). The single
   largest structural per-pixel difference, and not recoverable.

5. **Persistent-thread / single-dispatch MPMC cull.** UE5 console path keeps all lanes
   saturated across variable hierarchy depth in one dispatch. **Infeasible in WebGPU** (no
   cross-workgroup forward-progress guarantee, gpuweb #2229) AND abandoned by UE5 on PC for the
   same reason. Our 18-pass ping-pong BFS is the correct PC-class match. NOT a gap worth
   closing.

6. **Deferred material classification + tile binning + count→reserve→scatter** (UE5.4 GPU-
   driven materials, per-material indirect dispatch, DEPTH_EQUALS early-Z). Real but low value
   for our <8 closed material set; the resolve is a single divergent übershader. Worth a look
   only if the resolve becomes divergence-bound after win 4. The same scatter machinery would
   also enable triangle-granular raster compaction.

7. **Offline HLOD proxy bake (World-Partition "Approximated Mesh") + Nanite Assemblies
   intra-asset part merging.** UE pre-bakes cross-instance proxies at cook time and swaps whole
   instance groups beyond a distance, and merges twigs-within-a-tree into the base mesh at far
   LODs. We have no offline cook step, no proxy-swap, and monolithic tree meshes with no part
   hierarchy. A build-time category we do not address at all.

---

## 5. Recommended order of attack

Biggest verified win first; wins 1+2 share the same kernel and same setup block, so land them
together.

1. **Win 1 — scanline x-span (~2.5 ms).** The single biggest non-quality-loss lever, inside the
   16.7 ms dominant kernel, pure WGSL integer math, bit-identical. Do this first.
2. **Win 2 — depth DDA (~0.5 ms).** Same kernel, same `NaniteRaster.ts:626-633` setup block,
   tiny diff. Land in the same PR as win 1 (verify the speckle stays in-tolerance).
3. **Win 4 — terrain `If(isT)` gate (~0.15 ms).** Trivial S-effort branch, bit-identical, free.
4. **Win 3 — same-frame coarse Hi-Z (~0.5 ms).** L-effort, partial-feasibility; magnitude is
   capped by the max-depth-HZB-on-gaps problem, so attempt only after 1/2/4 are banked and
   measure the realized cull fraction before committing the full split.
5. **Win 5 — HW makeCtx cache (~0.04 ms).** Lowest priority; win 1 shrinks the set it helps.

**Beyond this round:** N8-HIC cross-instance merge (Section 3) is the real structural bet —
schedule it as a build-pipeline project with an explicit quality budget for the variation bake.

### Cheapest falsifiable test for the #1 win (do this BEFORE implementing)

The 2.5 ms estimate rests on the inner loop being ALU/branch-bound on empty pixels rather than
fully atomic-latency-bound. Falsify it with the ablation harness that **already exists** in the
shipped code — **no new feature, no code edit to the algorithm:**

> Run the existing `?rdbg` world1 stop points (`NaniteRaster.ts:635-655`, documented at
> 220-229): `rdbg=2` stops right before the scanline loop (launch + makeCtx + per-triangle
> edge-setup, per-pixel loop excluded); `rdbg=3` is the full loop. The measured **(rdbg3 −
> rdbg2)** delta is exactly the per-pixel coverage/election loop cost — the slice win 1 cuts
> into. If that delta is ~15 ms (≈90% of 16.7, matching the in-code PERF-3 finding at
> NaniteRaster.ts:224-225), a ~half-the-pixels skip plausibly yields ~2.5 ms. **If the delta is
> small** (i.e. world1 is setup/atomic-bound, not per-pixel-ALU-bound), the win collapses and
> win 1 should be reprioritized. This reuses the renderer's own `nanRasterWorld1` GPU timestamp
> + the four world1 sinks — it is the single cheapest measurement and gates the whole round.
>
> *(Per the HARD CONSTRAINTS this synthesis agent does not run probes; this is the test the
> implementer runs first.)*

---

*Evidence base: capture frame 1286 (cmd#378 `nanRasterWorld1`/INDIRECT 4594, cmd#408
`nanHzbL10`, cmd#454 `NodeMaterial_69`/shader 4749; 4622.wgsl:1298-1353; statistics.json
adapter features — `subgroups` present, no 64-bit atomics; timeline.md). Code: NaniteRaster.ts
(:90 MAX_RASTER_SIZE, :224-225 PERF-3, :627-633 setup, :657-748 loop, :725-736 election,
:730-733 3-atomic cliff, :817-841 HW vertex), NaniteResolve.ts:295/326-330, NaniteCull.ts:459-460/507-513,
NaniteHzb.ts:4-8. Anchor: DIVERGENCE-REPORT.md:414 (16.7 ms = 75%). UE5: Karis SIGGRAPH 2021
(advances.realtimerendering.com — scanline-DDA framing, verified via CS418 transcription);
candidstartup 2023-04-03 (two-pass main/post HZB, verified verbatim); Wihlidal GDC 2024
(deferred material classification).*
