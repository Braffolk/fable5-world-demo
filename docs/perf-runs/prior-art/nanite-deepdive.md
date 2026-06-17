# Prior-art brief: *A Deep Dive into Nanite Virtualized Geometry* (Karis/Stubbe/Wihlidal, SIGGRAPH 2021 Advances in Real-Time Rendering)

**Source kind:** conference talk (video — not watchable). Mined via the **official slides+speaker-notes PDF**
(`docs/perf-runs/prior-art/sources/papers/nanite-deepdive-slides.pdf`, 16.8 MB, `pdftotext -layout`'d to
`…/papers/nanite-deepdive-slides.txt`, 4518 lines incl. the verbatim speaker notes which contain the real
detail) plus reputable writeups (elopezr "A Macro View of Nanite", thecandidstartup, trickybits Part 1,
Scthe's WebGPU reimplementation + its source). All citations below are `slides.txt:LINE`.

This is the *origin* document for our entire architecture: SW-raster-for-tiny-tris, the visibility-buffer
depth|payload atomic election, two-pass HZB occlusion, the persistent-thread work queue. So the value here is
less "new idea" and more **ground-truth on what Nanite actually does, where it is faster than us, and — most
importantly — the slides themselves diagnose foliage/aggregates as Nanite's WORST case**, with a cost-map that
matches ours almost line-for-line. That tells us which levers are real vs. which Nanite *also* never solved.

---

## Raster architecture (as described)

- **Visibility buffer** = one **64-bit** value per pixel: `[30b depth | 27b VisibleClusterIndex | 7b TriangleID]`
  (`slides.txt:2269-2271`). Written by a **global image `InterlockedMax`** (`:2267-2290`). Depth in the HIGH bits
  gives the depth test "for free"; payload in the low bits. Karis: payload must "pack in **34 bits or less**…
  Without that we wouldn't be able to do fast software rasterization" (`:2297-2299`). **This 64-bit fused
  max IS the technique we cannot port** — see adaptation notes.
- **SW vs HW per cluster.** Clusters whose triangles are **< 32 px long** are software-rasterized; bigger ones go
  to the HW rasterizer (`:2421`, `:2442`). HW path *also* atomic-writes the same UAV visbuffer (no depth/color
  target bound) so SW and HW results merge and can run async (`:2393-2416`). "The vast majority are SW
  rasterized" (`:2406-2407`).
- **Work distribution = one workgroup (threadgroup 128) per cluster** (`:2300-2301`). Phase 1: **1 thread per
  vertex** → transform position, store in groupshared (loops twice for ≤256 verts, `:2302-2305`). Phase 2:
  **1 thread per triangle** → fetch indices, fetch transformed verts from groupshared, compute edge eqns +
  depth gradient + screen bbox, rasterize (`:2306-2312`). Vertex work is shared across the cluster with **no
  post-transform cache** — "kind of similar in structure to mesh shaders" (`:2328-2329`).
- **Cluster culling** is a separate, earlier stage: **persistent-thread MPMC job queue**, single dispatch, threads
  pop a hierarchy node, test it, push passing children (`:1897-1969`). 10-60% faster (~25% typical) than naive
  per-level dependent dispatches (`:1940`). Relies on (undefined-by-spec) forward-progress scheduling.
- **Raster binning (from the writeups / UE5 source, not the 2021 slides):** before raster, a
  classify→reserve→scatter pipeline bins each visible cluster by *raster bin* (which raster path/shader permutation)
  so each rasterize dispatch processes a **homogeneous batch** and sets up indirect args. (Confirmed via
  elopezr + search: `RasterBinClassify/Reserve/Scatter`.)
- **Material eval is fully deferred** (`:2771-2811`): a later full-screen pass decodes the visbuffer
  (load → VisibleCluster→Instance/Cluster → transform → barycentric → lerp attrs) and runs the material shader.
  Materials are culled with a **Material-Depth buffer + `depth==` test** so each material's full-screen quad only
  touches its own pixels via HW HiZ (`:2812-2960`).

### Nanite's own cost map for the SW rasterizer (`slides.txt:2578-2631`)
This is the single most useful slide for us. Karis breaks overdraw cost by triangle size:
- **Small tris → "Vertex transform and triangle setup bound"**
- **Medium tris → "Pixel coverage test bound"**
- **Large tris → "Atomic bound"**

Our measured map (≈1px tris, 12-20× overdraw on holey foliage) is the *small+medium* regime: **40% transform+setup
+ 60% pixel-coverage**, with the atomic NOT the bottleneck — which is *exactly* what Karis predicts for our tri
size. The slides then explicitly call out our content as Nanite's worst case: overdraw is excessive from
"Aggregates" and surfaces "riddled with small holes… most aggregate geometry cases like leaves and grass.
Overdraw is one of many reasons Nanite doesn't perform as well with those." (`:2584-2617`). **Conclusion: there
is no secret per-fragment trick in Nanite that we are missing — Nanite eats the same overdraw we do.** The wins
must come from setup/launch reduction and content/structure, not from a smarter coverage inner loop.

---

## Most transferable techniques

### DIRECT ports (1-2 already shipped — flagged as CONFIRMATIONS)

**T1 — Scanline x-span (solve for the covered X interval) — CONFIRMS our shipped win.** (`slides.txt:2449-2575`)
Nanite's micropoly loop is our bbox loop; they then switch to a scanline variant that *solves for the X interval
that passes all 3 edges* instead of testing every pixel: `CrossX = C / Edge.y; x0 = ceil(max3(MinX)); x1 =
min3(MaxX)`, then iterate only `[x0,x1]` (`:2513-2538`). Chosen "if the X loop is >4 pixels for any triangle in
the wave" (`:2554`). **We already shipped this** (commit `19eb834`). The slides confirm it is correct and add
two tuning facts we should verify against our impl: (a) the divide makes it **not exact fixed-point**, so it can
differ at edges from the bbox path — Karis "haven't found any issues in practice" but for our no-quality-loss bar
we must keep SW/HW (or SW/bbox) edge rules matching to avoid 1px cracks (`:2551-2552`, `:2409-2411`); (b) the
**>4px gate** — if our scanline runs unconditionally we may be paying setup on tris too small to benefit. Effort:
low (tuning only). Quality-preserving: yes (re-verify the cell-fill/edge-rule). Indirect-vs-direct: direct.

**T2 — Depth-gradient incremental Z + edge-function incremental coverage — likely already have; confirm no
redundant ALU.** (`slides.txt:2344-2369`) The inner loop keeps `ZX += GradZ.x` per pixel and `CXn -= Edge.y`
per pixel / `CYn += Edge.x` per row — i.e. **all per-pixel work is one add per edge + one add for Z**, no
per-pixel barycentric divide. The Scthe WebGPU port does the same (`edgeC()` returns `A,B,C`; inner loop only
`CX += A`). Action: confirm our coverage loop is pure incremental adds and that the barycentric-Z is the only
multiply; "the inner loop does barely anything, a couple ALU instructions and the atomic max" (`:2569-2570`).
If true, the 60% is overdraw VOLUME, not per-pixel ALU — agreeing with our rdbg split. Effort: low (audit).

**T3 — `>32px-long` SW/HW split, per cluster — re-examine our threshold.** (`slides.txt:2419-2442`) Nanite SW-
rasterizes "much bigger than expected, far past micropoly" — anything <32px/edge, deciding **per cluster** by
projected area. We are pure-SW. For foliage at ≤1px this is moot, BUT: the slides note "there are situations where
drawing more triangles is actually faster" because large clusters → coarse cull → more pixel overdraw
(`:2626-2631`). Relevance to us is the inverse insight (T6). Effort: n/a for our tri-size; keep pure-SW. Direct.

### INDIRECT (adapt-the-principle) — these attack OUR two costs

**T4 — Raster binning by cluster batch to cut the 40% transform+launch + wave divergence. (HIGH-VALUE, NEW for us)**
Nanite does NOT naively dispatch one workgroup per cluster into one kernel; UE5 bins clusters (classify→reserve→
scatter) so a raster dispatch processes a homogeneous batch, and the persistent culling stage **batches work "to
avoid divergence"** (`slides.txt:2016`). The slides' future-work note is the key (`:2560-2577`): Karis explicitly
says **"there is a significant amount of divergence with a wave having backfacing triangles that early out, small
tris covering a few pixels, and larger tris… Distributing the work evenly could result in very large gains if the
overhead was low."** This is OUR 40%: one-workgroup-per-cluster means a wave mixes a culled tri (early-out),
a 1px tri, and (rarely) a bigger one — long-pole divergence + the ~5× redundant `fetchWorldVert` / 48 wind taps.
**Adapt for us:** (a) the Nanite vertex-share already kills our redundant fetch — Phase-1 transforms each cluster
vertex ONCE into groupshared (`:2302-2305`), then Phase-2 reads transformed verts; we re-fetch+re-transform per
triangle (~5×). Porting the **two-phase groupshared vertex transform (incl. wind animation done once per vertex,
not 3× per triangle)** directly attacks the 40%. (b) A **per-triangle compaction / work-queue** (persistent-thread
style, `:1903-1908`) that streams surviving triangles to evenly-loaded lanes — but Karis tested even-distribution
and found "**not a win**… no middle ground" between his scanline and HW raster (`:2573-2577`), so the *generic*
work-redistribution is dead; the *vertex-share* part is the live lever. Effort: medium (vertex prepass into
groupshared/workgroup buffer). Quality-preserving: yes (bit-identical transform, just deduplicated). Indirect.

**T5 — Material/visbuffer decode as a deferred pass gated per-pixel — CONFIRMS our terrain `If(isT)` gate.**
(`slides.txt:2812-2960`) Nanite never runs material work during raster; it builds a Material-Depth buffer and uses
`depth==` HW HiZ so each material pass only shades its own pixels, "preventing any pixel shader waves from spinning
up" (`:2947`). We shipped the analogous `If(isT)` gate on `buildTerrainShading` (`8b8a256`). The portable insight:
their no-wave-intrinsic fallback uses a **64×64 tile grid + 64-bit material mask** to skip whole tiles
(`:2953-2959`) — if our resolve has multiple shading branches, a coarse per-tile "which-branch-present" mask could
skip dead branches per tile. Effort: medium. Indirect. (Note: aliasing-prone, `:2959`.)

**T6 — Treat overdraw as a CLUSTER-GRANULARITY problem, because Nanite couldn't fix it per-pixel either. (FRAMING)**
(`slides.txt:2578-2653`) Karis is blunt: no per-triangle culling (= no EarlyZ), HZB "culls clusters not pixels"
at "resolution based on cluster screen size", and **on holey/aggregate content the HZB max-op makes small HZB
levels "almost useless"** — which is EXACTLY our refuted occlusion-culling dead end (a hole pins max-Z to far).
So the slides independently CONFIRM our two dead ends: (1) conservative HZB occlusion ≈ 0% on foliage; (2) per-
triangle occlusion not done "because… would need to be re-rasterized which nullifies any savings" + thread-per-tri
divergence (`:2635-2640`). **The only structural lever Nanite offers for our floor is finer cluster granularity /
fewer overlapping clusters** — i.e. the cluster-count/overdraw floor our memory already names (N8-HIC merge), and
Karis's "plane equations / streaming HiZ data… occluder fusion" wish (`:2642-2651`) which is itself unsolved future
work. Honest takeaway: **the 60% is a content/overdraw floor Nanite shares; the realistic 60fps path is reducing
overdrawn cluster volume (quality-budgeted), not a faster fragment loop.** Indirect / framing.

**T7 — Visibility-buffer imposters for tiny instances (8:8 depth:triID into the SAME visbuffer). (SPECULATIVE)**
(`slides.txt:2732-2767`) For sub-few-pixel instances Nanite injects a precomputed **12×12-direction octahedral
imposter atlas, 12×12 px each, storing only 8:8 Depth:TriangleID** (40.5 KB/mesh resident), ray-marched a few
steps, **drawn straight into the screen visbuffer** bypassing the cluster path (`:2733-2744`). For a *forest of
many small instances at distance*, replacing whole distant foliage instances with a visbuffer-imposter injection
removes their clusters from raster entirely → cuts both the 40% and the 60% for the far field. Karis flags a
noticeable pop on repeated neighbors (`:2762`) — so this is **quality-budgeted, not zero-loss**, but TAA + dither
(`:2735`) hides much of it. Adapt: an 8:8 (or 16:8) depth:id imposter writing our 32-bit election word directly,
gated to instances below a screen-px threshold. Effort: high (atlas bake + inject path). Quality: NOT zero-loss
(distance pop). Indirect / speculative.

---

## The 64-bit-atomic question (central, per the mission)

**Nanite's election is fundamentally 64-bit and is NOT portable to WGSL.** `InterlockedMax` on a 64-bit UAV
where `[30b depth | 34b payload]` makes one atomic do BOTH the depth test AND the winner's payload store
**atomically and race-free** (`slides.txt:2267-2299`). WGSL has no `atomic<u64>`. The adaptations seen:

- **Scthe WebGPU port (closest real-world precedent):** packs **16b depth | 16b payload (oct-normal) into one
  `atomicMax(&result[idx], value)`** with `depth = 1 - z` so max=nearest, buffer cleared to 0
  (`/tmp/scthe-nanite/src/passes/rasterizeSw/rasterizeSwPass.wgsl.ts`: `createPayload()` →
  `(depthU16<<16)|nPacked`, `storeResult()` → `atomicMax`). The author explicitly calls 16-bit depth "a hack…
  produces **tons** of artifacts like z-fighting or leaks" and stores the *material directly* (no triangle id),
  so it cannot reconstruct attributes. **Our design is strictly better than the only public WebGPU Nanite:** we
  keep **24b depth** (8× finer) and store the full **25-bit cluster/tri id in a SIDE buffer** by the election
  winner, so we get real visbuffer attributes. This brief therefore confirms our split-buffer approach is the
  correct no-64-bit adaptation, not a compromise to revisit.
- **Our race is benign and we should keep it:** because the winner plain-stores its id AFTER winning the 32-bit
  `atomicMax`, two fronts at identical packed-depth could race the side store — but they are at the same depth so
  either id is visually equivalent (our memory already notes this "accepted atomic race"). Nanite's 64-bit fuse
  avoids the race but at a cost we cannot pay; the 8-bit tiebreak in our low bits makes exact ties astronomically
  rare. No change needed.
- **What this rules out:** any technique whose performance depends on the fused 64-bit max (single-instruction
  depth+payload, lock-free with zero side store) is structurally unavailable — there is no faster WGSL primitive
  to chase here. Our 32-bit-election + relaxed-load-then-conditional-store is already the right shape, and our
  measurement that losers early-out after a relaxed load (not atomic-contention-bound) matches Karis's "inner loop
  does barely anything, a couple ALU + the atomic max" (`:2569-2570`).

---

## Bottom line for the 20→60fps push
1. **No new per-fragment coverage trick exists in Nanite** — Karis's own cost map + the explicit "leaves and grass
   are our worst case / HZB useless on holey surfaces" slides CONFIRM our 60% is a shared overdraw floor and CONFIRM
   our refuted dead ends (conservative occlusion, per-tri cull). Do not re-hunt the inner loop.
2. **The live, zero-loss lever is the 40% transform+setup:** port Nanite's **two-phase groupshared per-vertex
   transform** so each cluster vertex (and its 48 wind taps) is transformed ONCE, killing our ~5× redundant
   `fetchWorldVert` — the one place we are demonstrably doing more work than Nanite (T4a). Bit-identical.
3. **Confirmations of already-shipped work:** scanline x-span (T1), deferred per-pixel-gated material/terrain
   resolve (T5).
4. **Quality-budgeted bigger swings** (need a quality bar, not zero-loss): visbuffer imposters for distant
   instances (T7) and the cluster-count/overdraw-floor merge our memory already names (T6) — these match Nanite's
   own unsolved future work (streaming HiZ / occluder fusion, hierarchical instancing).
