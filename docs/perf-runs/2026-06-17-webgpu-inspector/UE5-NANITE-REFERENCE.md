# UE5-NANITE REFERENCE (perf-critical paths, sourced) — for the subsystem analysts

> Purpose: a precise, mechanism-level account of how UE5 Nanite actually works on the
> performance-critical paths, so analysts can compare it against OUR WebGPU software
> rasterizer (frame-1286 capture = ground truth) and find non-quality-loss wins.
> Every UE5 claim is sourced inline. Every contrast with our pipeline is grounded in the
> capture (cmd#/dims) and/or our code (file:line). **What is feasible vs infeasible in
> WebGPU is called out in §8 — read it before proposing any win that leans on UE5 mechanics.**

## TL;DR for analysts (the 6 things that matter for our frame)

1. **UE5 writes ONE 64-bit atomicMax per covered pixel** (`depth<<32 | payload`) — depth and
   the cluster/tri id move together, so there is exactly one visibility write and zero risk
   of a torn payload. **We cannot do this** (WGSL has no 64-bit atomics — §8). Our `world1`
   single-pass is the correct WebGPU substitute (a 24-bit-depth `atomicMax` election +
   side-buffer id store); it already pays the residual-speckle cost. Do not propose "just use
   a 64-bit atomic" — it is not in the platform.
2. **UE5's cluster cull is ONE persistent dispatch** draining an MPMC work queue. **Our cull
   is ~17 fixed ping-pong dispatches** (`nanTraverseAB`/`BA`, frame-1286 cmd#26→#344, one
   `submit` each). The ROADMAP already MEASURED this submit-overhead lever as MARGINAL for our
   frame — believe that; the frame is raster-bound, not dispatch-bound.
3. **UE5 picks SW vs HW per cluster by projected screen size** (small tris → SW, ~3× faster
   than HW for tiny tris). We split **per-triangle by a 16-px bbox** (`MAX_RASTER_SIZE`,
   `NaniteRaster.ts:90`). Same idea, finer granularity.
4. **UE5's SW raster is one thread per triangle, 128 threads per cluster**, scanline over a
   bbox with fixed-point/subpixel edges. **We match this** (8 subpixel bits, top-left rule,
   integer edges — `NaniteRaster.ts:13,539`). The per-pixel inner loop is the cost in both.
5. **UE5 shades DEFERRED from the visibility buffer**, binned by material into tiles, one
   dispatch per material over only its pixels. **We resolve in ONE fullscreen übershader pass
   with a `Switch(materialClass)`** (`NaniteResolve.ts`). For our small closed material set
   this is fine; UE5's binning matters when you have thousands of distinct materials, which we
   do not.
6. **UE5 quantizes vertex positions to a per-cluster local grid** (few bits/component, decoded
   in-shader). We recompute/transform verts per cluster instead. This is a **bandwidth/VRAM**
   lever (our capture shows 2208 MB of buffers, 850 MB textures — VRAM hygiene), **not a
   frame-time lever** for the raster-bound frame.

---

## 1. The SOFTWARE rasterizer

**Mechanism.** Nanite's SW rasterizer is a compute shader where **a single thread renders a
single triangle, and a thread-group of 128 threads renders the 128 triangles of one cluster
in parallel** ([trickybits](https://trickybitsblog.github.io/2024/04/20/nanite.html);
[candidstartup](https://www.thecandidstartup.org/2023/04/03/nanite-graphics-pipeline.html):
"All 128 triangles in a cluster... in parallel, one per execution unit"). The per-triangle
inner loop is described by Karis as **"somewhere between a linear DDA and looping over all
pixels in the bounding box and checking if each is inside the triangle"** — i.e. a
**bounding-box scanline** with incremental edge functions, deliberately lightweight because
the triangles are tiny (Karis, *A Deep Dive into Nanite Virtualized Geometry*, SIGGRAPH 2021;
summarized in [trickybits](https://trickybitsblog.github.io/2024/04/20/nanite.html)).

**Subpixel / fixed-point.** Edges are evaluated in fixed-point with subpixel precision and a
top-left fill rule so adjacent clusters are watertight (matches the HW rasterization
convention; D3D mandates 8 subpixel bits). This avoids the float-bias hacks that cause cracks.

**The 64-bit atomicMax visibility buffer (the key trick).** Both the SW inner loop and the HW
fragment shader write **one 64-bit value per pixel via a single atomicMax**, packed as
**depth in the high 32 bits, payload (cluster + triangle id) in the low 32 bits**
([trickybits](https://trickybitsblog.github.io/2024/04/20/nanite.html);
[jglrxavpok](https://jglrxavpok.github.io/2023/11/26/recreating-nanite-visibility-buffer.html):
"32 high bits: depth / 32 low bits: triangle ID"). The canonical bit layout is
**30 bits depth, 27 bits visible-cluster index, 7 bits triangle id**
([candidstartup](https://www.thecandidstartup.org/2023/04/03/nanite-graphics-pipeline.html))
— [elopezr "A Macro View of Nanite"](https://www.elopezr.com/a-macro-view-of-nanite/) reports
it as **R[31:7] ClusterID (25b), R[6:0] Triangle ID (7b), G = 32-bit depth**. Depth is inverted
so the nearest fragment produces the largest combined value and wins the `atomicMax`
([jglrxavpok](https://jglrxavpok.github.io/2023/11/26/recreating-nanite-visibility-buffer.html):
`depth = 0xFFFFFFFF - uint(ndc.z * 0xFFFFFFFF)`).

**Why this is fast / how it avoids overdraw:**
- **Depth + payload move together in ONE atomic.** There is no separate depth pass, no read-
  modify-write of a color target, no torn payload, and no need for the hardware ROP/early-Z
  machinery. The atomic *is* the depth test and the write, fused.
- **No fragment-shader overdraw.** The visibility buffer stores only an id; no material/shading
  work happens during rasterization. Occluded fragments cost one atomic compare and stop.
  All shading is deferred to §6, evaluated exactly once per visible pixel.
- **Per-pixel work is minimized** to: edge-function step, depth interpolate, one `atomicMax`.
- **128 tris/cluster fills a wavefront** (a "suspicious multiple of 32 and 64 chosen to fill
  wavefronts" — [candidstartup](https://www.thecandidstartup.org/2023/04/03/nanite-graphics-pipeline.html)).

**OUR mapping (capture + code).** Our SW raster is the same shape — one cluster per workgroup,
fixed-point integer edges, top-left rule, 8 subpixel bits (1/256-px grid),
`NaniteRaster.ts:13,539-589`. In the frame-1286 capture our combined raster is the single
compute pass **`nanRasterWorld1` (cmd#378, INDIRECT 4594)** — and the ROADMAP's MEASURED
verdict is this pass is ~16.7 ms ≈ 75% of the frame: **the frame is genuinely raster-bound,
dominated by the per-pixel inner loop, not the atomic** (ROADMAP "YOU ARE HERE"; LOG bz; and
PERF-3's per-triangle breakdown: atomic ≈ payload, *not* the cost). The single biggest
structural difference from UE5 is forced by §8: we cannot fuse depth+payload into one atomic.

## 2. The SW-vs-HW split

**Mechanism.** Nanite decides **per cluster, on the fly, from the cluster's projected screen
area**, whether to rasterize it in software or hand it to the hardware rasterizer; a cluster is
entirely one or the other ([trickybits](https://trickybitsblog.github.io/2024/04/20/nanite.html):
"decides on the fly based on the projected area of the cluster"). The SW path targets clusters
whose **triangles are less than ~32 pixels long**
([candidstartup](https://www.thecandidstartup.org/2023/04/03/nanite-graphics-pipeline.html)).

**The cost model behind the split.** For small/sub-pixel triangles the HW rasterizer is
inefficient because it shades in **2×2 quads**: a 1-pixel triangle still launches a full quad,
so **pixel-shader efficiency drops toward ~25%**, and "the hardware is quite terrible at
combining small triangles into quads"
([trickybits](https://trickybitsblog.github.io/2024/04/20/nanite.html)). Nanite's SW
rasterizer is **~3× faster on average than the HW rasterizer for small triangles**
([candidstartup](https://www.thecandidstartup.org/2023/04/03/nanite-graphics-pipeline.html);
trickybits). Conversely, **large triangles go to HW** because the fixed-function setup/edge/
attribute interpolation and quad efficiency win once a triangle covers many pixels — a SW
scanline over a huge bbox is O(area) ALU on a single thread and would serialize badly.

**OUR mapping (code).** We split at a **finer granularity — per triangle, not per cluster — by
bounding-box size**: a triangle whose *unclamped* screen bbox is ≤ `MAX_RASTER_SIZE = 16` px on
both axes rasters in SW; anything larger (or near-plane crossing) routes to the HW queue
(`NaniteRaster.ts:90,558-565`; near-crossing → HW at `NaniteRaster.ts:359` policy in
SPEC). In frame-1286 the HW remainder is the `nanHwPass` render pass
(**cmd#397, `drawIndirect(4625)`**, fed by `nanHwArgs` cmd#389). The 16-px threshold is tunable
and was chosen vs Nanite's ~32-px cluster threshold (SPEC line 361-363). **Note the differing
axis of the decision** — UE5's per-cluster decision means one branch per ~128 triangles and a
homogeneous SW/HW workload per cluster; our per-triangle decision is more precise but evaluates
the bbox test for every triangle. For our raster-bound frame the per-pixel loop dominates, so
the split granularity is not the lever.

## 3. Cluster CULLING — persistent threads / single dispatch + MPMC work queue

**The problem.** Hierarchy traversal in compute is awkward: the number of nodes to process is
dynamic, from zero to hundreds of thousands, and is unknown until you traverse
([trickybits](https://trickybitsblog.github.io/2024/04/20/nanite.html)). A naive
"one dispatch per BVH level" wastes the GPU on shallow/empty levels and round-trips to the CPU.

**Nanite's mechanism — `PersistentCull`.** The CPU spawns **just enough persistent worker
threads to fill the GPU**, and they share a **Multiple-Producer/Multiple-Consumer (MPMC) job
queue** seeded with the root node/cluster of each instance. A worker **atomically pops the next
item, runs frustum + LOD-error + occlusion tests, and — for a node — atomically pushes its
visible children back onto the queue**; for a leaf cluster it marks it visible. Workers loop
until the queue drains ([trickybits](https://trickybitsblog.github.io/2024/04/20/nanite.html):
"a worker thread atomically consumes the next item... sometimes produce new items which are
atomically added to the end... Once the queue is empty all the work is done";
[candidstartup](https://www.thecandidstartup.org/2023/04/03/nanite-graphics-pipeline.html)).
Crucially this is **a single kernel**: "PersistentCull is a single kernel that operates on an
atomic list of nodes and clusters, whereas NodeAndClusterCull is multiple compute kernels...
PersistentCull is the current default as it is more efficient"
([trickybits](https://trickybitsblog.github.io/2024/04/20/nanite.html)).

**Why occupancy stays high.** Because threads are persistent and self-feeding, **all lanes stay
busy across the entire variable-depth traversal in one dispatch** — no per-level barrier, no
empty trailing dispatches, no CPU round-trip, no indirect-args recompute between levels. The
queue load-balances irregular subtrees across the resident workers.

**OUR mapping (capture + code) — this is our biggest structural divergence in the cull.**
We use **fixed ping-pong dispatches**: an `Args` kernel computes indirect dispatch args, then a
`Traverse` kernel processes one frontier level reading frontier A / writing frontier B, then we
flip (B→A), repeated up to `HIER_MAX_DEPTH` times (`NaniteCull.ts:13-22,415-555`;
"Ping-pong frontiers (read A / write B)"). The frame-1286 timeline shows this verbatim:
**`nanArgsAB`→`nanTraverseAB`→`nanArgsBA`→`nanTraverseBA` repeating, each in its own `submit`,
cmd#26 through cmd#344** — roughly **17 traverse passes, ~34 submits, just for the cull**. Each
is its own indirect dispatch with empty trailing levels (the BFS converges around depth 9 but we
pad to the safe DAG-leaf bound; ROADMAP S3-perf note + `HIER_MAX_DEPTH` default 18).

**Perf reality for analysts:** the ROADMAP **already measured** the "collapse the ping-pong into
a persistent single dispatch to cut submit overhead" lever and found it **MARGINAL** for this
frame ("the submit-overhead lever is MARGINAL — measured, not assumed — don't refactor the cull
for it"; LOG bz, ROADMAP "YOU ARE HERE"). **Do not re-propose it as a frame-time win.** A true
persistent-MPMC cull is *also harder in WebGPU* (no forward-progress guarantee for spinning
workers across workgroups — §8), so it is both low-value and high-risk here. The cull's value is
in *cutting the visible-cluster count* (merge/aggregation, §N8-HIC), not in its dispatch shape.

## 4. TWO-PASS occlusion culling

**Mechanism.** Within one frame Nanite runs **two full passes — "main" and "post"** — differing
only in which HZB they cull against ([trickybits](https://trickybitsblog.github.io/2024/04/20/nanite.html)):

- **Pass 1 (main):** cull + raster everything that was **visible last frame**, tested against
  **last frame's HZB reprojected into this frame** ("project the current geometry into last
  frame's depth buffer by simply using last frame's transforms"). This draws the set that is
  almost certainly still visible.
- **Build HZB:** from the pass-1 depth, build a fresh Hierarchical-Z (a depth mip pyramid;
  "a Z buffer mip map uses min/max") — this HZB now contains all the geometry just drawn
  (Nanite + traditional).
- **Pass 2 (post):** re-test the clusters that were **rejected/indeterminate** in pass 1 against
  the *fresh* HZB, and raster the newly-disoccluded survivors. Build the final HZB for next
  frame.

**Why it beats single-phase.** Single-phase culling can only use a *stale* (previous-frame) HZB,
so it must conservatively keep anything not provably occluded last frame → re-draws geometry
that is actually occluded this frame (overdraw of the visibility atomic). The two-pass scheme
uses the **current frame's own depth** as the occluder set for the uncertain remainder, so it
removes essentially all false-negative overdraw while still being temporally robust to
disocclusion (newly-revealed geometry is caught in pass 2). Cost is kept low because **"in
normal camera navigation the second pass is a small fraction of the main pass"**
([candidstartup](https://www.thecandidstartup.org/2023/04/03/nanite-graphics-pipeline.html)) and
**"two passes are only performed if needed — if there are no indeterminate clusters after the
first pass then the second pass isn't needed"**
([trickybits](https://trickybitsblog.github.io/2024/04/20/nanite.html)). The HZB is built at
half resolution (level 0 at half-res, conservative max) which is cheap.

**OUR mapping (capture + code).** Our world raster was deliberately collapsed to **single-pass**
(`PERF-VB4`/D-N45, SHIPPED default; the 2-pass world path is DELETED — ROADMAP item 1,
SPEC line 340-348). In frame-1286 there is **one raster pass** (`nanRasterWorld1` cmd#378) and
the HZB is built **after** it (`nanHzbL10`, cmd#408, the 11→1 batched mip chain
6242→1566→…→1 workgroups). That HZB is used as **last-frame occlusion for next frame's cull**
(prev-HZB feeds the traverse occlusion test, `NaniteCull.ts:353` "HZB occlusion read"). So we
run **single-phase occlusion with a reprojected previous-frame HZB** — exactly the
"known example gap vs real Nanite" the SPEC flags (SPEC line 209: "single-phase occlusion
(prev-frame HZB only)"). **Analyst implication:** adding a true UE5 two-pass would remove
same-frame disocclusion overdraw, but for a static-ish forest the win is the "small fraction"
UE5 itself notes; weigh it against doubling the raster dispatch count. The ROADMAP's listed
candidate is "same-frame cluster Hi-Z" — that is the targeted, cheaper version of this idea.

## 5. Cluster DATA format — quantization & compression

**Mechanism.** A cluster is the only renderable unit and **always holds 128 triangles
(≈384 vertices)** ([trickybits](https://trickybitsblog.github.io/2024/04/20/nanite.html);
[candidstartup](https://www.thecandidstartup.org/2023/04/03/nanite-graphics-pipeline.html)).
Clusters are grouped into a **BVH of nodes** within each mesh (the hierarchy that §3 traverses).
The on-GPU geometry is **"heavily compressed and looks nothing like traditional index/vertex
buffers... bound as a block of data and entirely decoded by the shader"**
([trickybits](https://trickybitsblog.github.io/2024/04/20/nanite.html)).

**Position quantization (the bandwidth win).** Karis's scheme snaps vertices to a **uniform
grid**, and — critically — uses a **per-cluster local coordinate system / local origin** so each
vertex is a small offset from the cluster center and needs **few bits per component** (Karis,
SIGGRAPH 2021; see implementer write-ups
[Omniforce](https://daniilvinn.github.io/2024/05/04/omniforce-vertex-quantization.html):
"Local coordinate systems are applied to cluster meshes, allowing vertices to be located at
shorter distances from the original points... positions can be expressed accurately even with a
small number of bits"). A faithful implementation uses a **per-cluster (variable) bitrate =
the worst-case bits among that cluster's vertices**, with the **cluster center itself quantized
on the same global grid to avoid cracks between clusters**
([Omniforce](https://daniilvinn.github.io/2024/05/04/omniforce-vertex-quantization.html)). Decode
is a bitstream read + `ldexp`/POT divide + add center — "as fast as possible." Other vertex
attributes (normals/UVs) are likewise **quantized and bit-packed**, with a separate compressed
in-memory format used directly for rendering with near-instant decode
([candidstartup](https://www.thecandidstartup.org/2023/04/03/nanite-graphics-pipeline.html)).

**Streaming / bandwidth implications.** Clusters are packed by spatial locality + LOD level into
**128 KB pages** in a giant virtual byte buffer (`Nanite.StreamingManager.ClusterPageData`), the
BVH lives in `Nanite.StreamingManager.Hierarchy`, and both are dynamically streamed
([trickybits](https://trickybitsblog.github.io/2024/04/20/nanite.html); Karis SIGGRAPH 2021).
The payoff is **bandwidth and resident VRAM**: tiny compressed clusters mean far less memory
traffic per rasterized triangle and a small resident working set even for "billions" of source
triangles.

**OUR mapping (capture).** We do **not** quantize/compress cluster geometry this way: terrain
enters procedurally (heightfield windows reconstructed in the transform stage, SPEC line 121),
and explicit meshes carry full positions that we **recompute/transform per cluster** in the
kernel (SPEC line 374-379: "default = recompute"; `NaniteFetch.ts`). The frame-1286 capture
shows the consequence: **2208 MB across 274 buffers + 850 MB across 55 textures** (summary.md),
and the ROADMAP explicitly calls the ~1.5 GB dead full-world residency **"VRAM hygiene, not
frame-time"**. **Analyst implication:** position quantization is a real lever for **memory/VRAM
and streaming bandwidth**, and could reduce per-vertex fetch cost in the raster (the PERF-3
breakdown found `3× fetchWorldVert ≈ 39%` of the per-triangle cost), but it is a build-pipeline
change with decode cost, and the headline frame cost is the per-pixel loop, not vertex fetch.
Treat quantization as a bandwidth/VRAM win, not a primary frame-time win.

## 6. MATERIAL pass — visibility buffer → classification/binning → deferred eval

**Mechanism (UE5.4+, GPU-driven materials, Wihlidal GDC 2024).** Nanite never shades during
rasterization. After the visibility buffer is complete it does a **deferred material pass**:
- **Material classification / binning.** A pass dispatches **one thread per cluster** to count,
  per **raster bin** (a unique combination of material properties affecting rasterization), then
  **reserve** offsets and **scatter** meshlets into per-bin lists — a classic count→reserve→
  scatter ([Scthe notes on Wihlidal GDC 2024](https://www.sctheblog.com/blog/nanite-materials-notes/)).
  Shading uses the same count→reserve→scatter to sort **pixel locations** into **shading bins**
  (≈ unique materials).
- **The "material depth" trick.** The material id is written to **`SV_Depth`** so that a
  per-material full-screen pass can use **`DEPTH_EQUALS`** early-Z to reject every pixel not
  belonging to that material before its pixel shader runs — restricting each material's work to
  exactly its pixels without per-pixel branching
  ([Scthe](https://www.sctheblog.com/blog/nanite-materials-notes/)). Advanced path: **64×64
  pixel tiles with per-tile material lists** so whole tiles are skipped (this is the 20×12 = 240
  material-range texture for 64×64 tiles in [elopezr](https://www.elopezr.com/a-macro-view-of-nanite/)
  / [candidstartup](https://www.thecandidstartup.org/2023/04/03/nanite-graphics-pipeline.html)).
- **Deferred evaluation in compute.** UE5.4 moved shading to compute: **per material, an indirect
  dispatch with 1 thread == 1 pixel**, organized in **8×8 workgroups over 16×16-pixel blocks**,
  threads assigned by **Morton code** for locality, and **2×2 quads kept together** so gradients
  and Variable-Rate-Shading decisions (single-material 4×4 or 8×8 blocks shade once) work
  cooperatively ([Scthe](https://www.sctheblog.com/blog/nanite-materials-notes/)).

**Why deferred-binned beats forward per-cluster shading.** (a) Each pixel is shaded **exactly
once** (no overdraw — the visibility buffer already resolved the front surface). (b) **No
material-PSO thrash during raster** — material complexity is decoupled from rasterization. (c)
**Empty bins are nearly free** — indirect dispatch over a sorted pixel list means no context
rolls for absent materials. (d) **Coherent execution** — every lane in a workgroup runs the same
material, eliminating divergence; Morton ordering keeps texture cache hot
([Scthe](https://www.sctheblog.com/blog/nanite-materials-notes/)).

**OUR mapping (capture + code).** We are deferred (visibility-buffer) too, but **un-binned**: a
**single fullscreen-triangle resolve pass** runs an **übershader with `Switch(materialClass)`**
over a *small closed* material set (TERRAIN/ROCK/BARK/DEADWOOD/LEAF/GRASS/DEBRIS, kept < 16 —
SPEC line 136, 395; `NaniteResolve.ts:2,7,222`). In frame-1286 the resolve is the
`NodeMaterial_69` render pass (**cmd#454, `drawIndexed[5952]` + `draw[3]`** fullscreen).
**Analyst implication:** UE5's tile binning + material-depth is a win **when material divergence
is high** (thousands of distinct shaders → branch divergence + dead lanes in an übershader). Our
material count is tiny and closed, so the `Switch` übershader has low divergence and the binning
machinery would likely cost more than it saves. **This is probably NOT a frame-time win for us**
— and our resolve is not the bottleneck (raster is). De-prioritize material binning unless a
profile shows the resolve übershader is divergence-bound.

## 7. Workgroup OCCUPANCY — keeping lanes busy

**Mechanism.** Nanite's occupancy comes from **work granularity, not one-workgroup-per-cluster
with idle lanes**:
- **Raster:** 128 triangles per cluster = 128 threads = a full wavefront, **one triangle per
  lane** — chosen as a multiple of 32/64 to fill GPU wavefronts exactly
  ([candidstartup](https://www.thecandidstartup.org/2023/04/03/nanite-graphics-pipeline.html);
  [trickybits](https://trickybitsblog.github.io/2024/04/20/nanite.html)).
- **Cull:** the persistent MPMC queue (§3) **load-balances irregular hierarchy work across all
  resident workers** so no lane sits idle on a shallow subtree
  ([trickybits](https://trickybitsblog.github.io/2024/04/20/nanite.html)).
- **Shading:** count→reserve→scatter compaction + Morton-ordered, **material-coherent**
  workgroups (§6) → no divergence, no dead lanes from empty materials
  ([Scthe](https://www.sctheblog.com/blog/nanite-materials-notes/)).

The contrast is a *naive* one-workgroup-per-cluster scheme where lanes go idle whenever a
cluster has < workgroup-size of work, or where divergent triangle sizes within a workgroup
serialize — Nanite avoids both via uniform 128-tri clusters and binning.

**OUR mapping (code).** We *do* run one-workgroup-per-cluster, and we exploit it positively:
the **per-cluster `makeCtx` cache** computes the cluster transform once on thread 0 and broadcasts
it via `workgroupArray`/`workgroupBarrier` (PERF-3 WIN #1, `?wgcache` default on, **−11%** on the
SW raster; ROADMAP PERF-3, SPEC line 145). The known occupancy hazard for us is **triangle-size
divergence within a cluster's 128 lanes** and **per-pixel-loop length variance** (a lane covering
a 16×16-px triangle runs ~256 inner iterations while its neighbor covering a sub-pixel triangle
runs ~1) — that divergence is inherent to a SW scanline and is part of why our raster is
per-pixel-loop bound. The cooperative-vertex-transform cache (PERF-3 WIN #2) was built and
measured a **marginal/non-win** (kept off; ROADMAP PERF-3) — note that before re-proposing
shared-memory raster wins.

## 8. WebGPU / WGSL FEATURE GAPS vs UE5's D3D12/Vulkan (READ THIS — feasibility gate)

UE5 Nanite leans on D3D12/Vulkan features that **WebGPU/WGSL does not have**. Do **not** propose
wins that require these.

| UE5 relies on | WebGPU/WGSL status | Portable substitute / our reality |
|---|---|---|
| **64-bit atomics** (`InterlockedMax` on a 64-bit UAV for the fused depth+payload vis buffer) | **ABSENT.** WGSL atomics are 32-bit only; `atomicCompareExchangeWeak` is single-u32; only a gpuweb **proposal** exists ([issue #5071](https://github.com/gpuweb/gpuweb/issues/5071), explicitly motivated by Nanite vis-buffers) — and M1-class Metal may never get it (SPEC line 94-97). | Our `world1`: a **24-bit-depth `atomicMax` election** (`visPayloadV = dk24<<8 \| id8`) whose winner `atomicStore`s the full 25-bit id into a single side buffer `visBV` (`NaniteRaster.ts:243-253,302`; SPEC D-N45). CAS-pair 64-bit emulation is **STRUCK as unsound** (tears under contention, F1). Residual: <0.1% wrong-cluster speckle very close to objects — **accepted; the zero-speckle fix needs native 64-bit atomics** (SPEC line 346). **Do not propose recovering this.** |
| **Persistent threads with forward-progress** (spinning workers draining an MPMC queue across the whole GPU in one dispatch) | **No portable forward-progress guarantee** across workgroups; a spin-wait MPMC consumer can deadlock depending on scheduler. | Our fixed **ping-pong level dispatches** (`NaniteCull.ts`; frame-1286 cmd#26→#344). Already measured: collapsing them is a **MARGINAL** win (LOG bz). Keep the ping-pong. |
| **Portable wave/subgroup intrinsics** (ballot, prefix-sum, broadcast used in cull compaction and material scatter) | **PARTIAL.** `subgroups` ships (Chrome 134+, 2025-02) — basic subgroup ops available — but the surface is **narrower and less portable** than HLSL wave intrinsics, and not guaranteed across all our targets (SPEC line 89-90). | Use `workgroupArray` + `workgroupBarrier` for cooperative work (as in `?wgcache`). Subgroups are *available but optional* — don't make a win depend on a specific subgroup width. |
| **64-bit / 30-bit single-write depth precision in the vis buffer** | Tied to the 64-bit gap above. We pack depth into a **24-bit key**, not 30–32 bits. | 24-bit banded depth is sub-pixel-accurate for our ranges (16-bit was caught terracing on grazing terrain → 24-bit free fix; SPEC line 342). |
| **Atomic float add** (e.g. QEM quadric scatter in build, or accumulation) | **ABSENT** — no `atomicAdd` on f32. | CAS spin-loops (slow) or restructure. Relevant to build, not the frame. |
| **MRT depth-equal "material depth" + flexible early-Z PSOs** (§6) | Depth-equal test exists, but the GPU-driven count→reserve→scatter binning + per-material indirect-compute-dispatch machinery is heavy to port. | Our single fullscreen `Switch` übershader (`NaniteResolve.ts`) — fine for < 16 materials. |
| **>10 storage buffers in one shader stage** (UE5 binds large descriptor sets freely) | **CONSTRAINED.** We repeatedly hit a **~10 storage-buffer ceiling** per stage (SPEC line 353-358, 794; ROADMAP "avoids the resolve 10-buffer ceiling"). | Bind conditionally per path (e.g. `gpu.vcompact` in the raster only). A win that needs many extra storage buffers in the resolve/raster is likely infeasible. |
| Indirect dispatch / `maxComputeWorkgroupsPerDimension` | Present but **65535 per dimension** → 1D dispatches auto-split to 2D; pad-guard kernels (SPEC line 98-99). The capture shows 2D-split indirect dispatches throughout. | Fine, already handled. |

**Net feasibility guidance for analysts:** the single largest UE5 advantage on the
perf-critical path — the **fused 64-bit depth+payload atomic** — is **categorically unavailable**
to us and we already pay its best-available substitute's cost. The cull-shape advantage
(persistent MPMC) is both **infeasible-to-fully-port** and **measured-marginal**. So the real
non-quality-loss wins for OUR raster-bound frame are the ones UE5 also depends on but that ARE
expressible in WebGPU: **(a) cut the visible-CLUSTER count** (cross-instance merge/aggregation —
N8-HIC; UE5's whole advantage at distance is coarse-DAG clusters covering ~few tris/tree, not a
faster per-pixel loop), **(b) same-frame cluster Hi-Z to remove overdraw** (the cheap cousin of
§4's two-pass), and **(c) bandwidth/VRAM** via position quantization (§5) if memory pressure
matters. The per-pixel inner loop itself (the 16.7 ms) is already near its floor after PERF-3.

---

## Sources

- Brian Karis et al., **"A Deep Dive into Nanite Virtualized Geometry," SIGGRAPH 2021** (Advances in Real-Time Rendering course) — the primary source for the SW rasterizer, 128-tri clusters, quantization, persistent cull, and two-pass occlusion.
- [Nanite Deep Dive (trickybits)](https://trickybitsblog.github.io/2024/04/20/nanite.html) — SW raster (1 thread/tri, 128/cluster), 64-bit vis buffer, `PersistentCull` single kernel vs `NodeAndClusterCull`, two-pass main/post, SW/HW split by projected area, 128 KB page streaming, 128-tri clusters.
- [Recreating Nanite: Visibility buffer (jglrxavpok)](https://jglrxavpok.github.io/2023/11/26/recreating-nanite-visibility-buffer.html) — 64-bit packed format (32 hi depth / 32 lo id), inverted depth + `imageAtomicMax`.
- [From Navisworks to Nanite (thecandidstartup)](https://www.thecandidstartup.org/2023/04/03/nanite-graphics-pipeline.html) — vis-buffer 30b depth/27b cluster/7b tri, <32px SW threshold, 3× faster, 25% quad efficiency, 128 tris ≈384 verts, two-pass "small fraction," 64×64 material tiles / 20×12 range texture, quantized+bit-packed attributes.
- [A Macro View of Nanite (elopezr)](https://www.elopezr.com/a-macro-view-of-nanite/) — vis-buffer layout R[31:7] ClusterID 25b / R[6:0] tri 7b / 32-bit depth.
- Graham Wihlidal, **"Nanite GPU-Driven Materials," GDC 2024** ([GDC Vault](https://gdcvault.com/play/1034407/Nanite-GPU-Driven); [Epic blog](https://www.unrealengine.com/en-US/blog/take-a-deep-dive-into-nanite-gpu-driven-materials)) — material classification, raster/shading bins (count-reserve-scatter), material-depth `SV_Depth`/`DEPTH_EQUALS`, 64×64 tiles, compute shading 1 thread/pixel in 8×8 wg / 16×16 blocks, Morton, quad/VRS.
- [Notes on Wihlidal GDC 2024 (Scthe)](https://www.sctheblog.com/blog/nanite-materials-notes/) — detailed transcription of the material binning, material-depth, tile, and compute-shading dispatch structure.
- [Omniforce vertex quantization (Daniil Vinnik)](https://daniilvinn.github.io/2024/05/04/omniforce-vertex-quantization.html) — per-meshlet local-origin quantization, per-meshlet bitrate, quantized cluster center to avoid cracks, `ldexp` decode; explicitly inspired by Karis SIGGRAPH 2021.
- WebGPU gaps: [gpuweb 64-bit atomics proposal #5071](https://github.com/gpuweb/gpuweb/issues/5071); [subgroups in WebGPU/Chrome 134](https://developer.chrome.com/blog/new-in-webgpu-134). Confirming the WGSL 16-bit-depth fallback wall: [Scthe/nanite-webgpu](https://github.com/Scthe/nanite-webgpu) ("tons of artifacts like z-fighting or leaks").

### Grounding in our pipeline (capture cmd# + code file:line)
- SW raster = `nanRasterWorld1` (frame-1286 **cmd#378**, INDIRECT 4594); ~16.7 ms ≈ 75% of frame (raster-bound — ROADMAP, LOG bz, PERF-3).
- HW remainder = `nanHwArgs` (**cmd#389**) → `nanHwPass` render `drawIndirect(4625)` (**cmd#397**).
- HZB build after raster = `nanHzbL10` (**cmd#408**, 11→1 batched mips).
- Resolve übershader = `NodeMaterial_69` (**cmd#454**, `drawIndexed[5952]`+`draw[3]`).
- Ping-pong cull = `nanArgsAB`/`nanTraverseAB`/`nanArgsBA`/`nanTraverseBA`, **cmd#26→#344** (~17 traverse passes, one submit each).
- Code: SW core + fixed-point edges `NaniteRaster.ts:13,90,243-253,302,539-589`; ping-pong traverse `NaniteCull.ts:13-22,415-555`; resolve `Switch` übershader `NaniteResolve.ts:2,222`; vertex recompute `NaniteFetch.ts`; SPEC §"Vis-buffer + depth precision" (line 310-371) + WebGPU feature gaps (line 88-104) + D-N45 (line 340-348).
- VRAM context: 2208 MB / 274 buffers + 850 MB / 55 textures (summary.md) — quantization is a VRAM/bandwidth lever, not a frame-time lever.
