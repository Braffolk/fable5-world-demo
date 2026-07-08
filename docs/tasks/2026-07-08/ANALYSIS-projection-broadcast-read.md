# ANALYSIS — can the `nanProjectVerts` broadcast-read be eliminated *completely*, beyond Lever A?

**Task #76 continuation. Date 2026-07-08. THINK + RESEARCH only — no source edited, nothing committed.**
Scope: `src/nanite/raster/Project.ts` (`nanProjectVerts`), the per-cluster ctx broadcast-read.
Baseline to beat: **Lever A** = load the per-cluster ctx once into threadgroup (workgroup) shared memory + a
`workgroupBarrier`, then every thread decodes from shared. Hard constraint honoured throughout: **no refusion**
(the projection pass stays a separate compute shader; it is never merged back into Classify/world1).

Inputs read: `PROPOSAL-nanProjectVerts.md` §2 (the measured diagnosis), `raster/Project.ts` (the kernel),
`raster/ClusterCtx.ts` (the 35-word ctx), `NaniteFetch.ts` (what the transform actually consumes),
`NaniteRaster.ts` §wgcache (the OLD path — Lever A already shipped there once), `NaniteVertexCache.ts` (the
cooperative-shared primitive), `three@0.184.0` TSL/WGSL builder sources, and `Engine.ts` (device features).

---

## 0. TL;DR VERDICT (details + citations below)

**Complete elimination — a true one-fetch-all-lanes hardware broadcast with no barrier and no threadgroup-load
pressure — is NOT achievable in WebGPU/Dawn/Metal for this data.** The one mechanism that *is* a true
constant-cache hardware broadcast on Apple (the Metal `constant` address space, reached via a WebGPU **uniform**
buffer) is blocked by `maxUniformBufferBindingSize = 64 KiB` against a ~21 MB per-cluster working set, and the
GPU-driven **single indirect dispatch** cannot re-window a uniform buffer per workgroup. That path is dead.

**Lever A (threadgroup-shared cooperative load) is the practical floor — and it is a *good* floor.** It drops
the global ctx load-issues ~240x (~9 435 -> ~39 per cluster), which lands **below the irreducible
vertex-gather floor (~520 issues/cluster)**. Once the ctx broadcast is beneath the vertex-gather floor it stops
being the limiter, so *nothing else can materially beat A on this specific bottleneck* — the shader re-limits on
vertex gather / writes / LLC footprint (Lever B territory), not on ctx reads.

The other avenues rank as: **subgroups** = a viable barrier-free *alternative* to A, but it leaves **8x more**
global ctx loads than A (one per subgroup, not one per workgroup) and its only saving (no barrier / no
threadgroup-load unit) does not cash out because neither is the next limiter — so **not materially better than
A**. **Shrink-the-payload** (35 -> ~21 hot words) folds into A *for free* by simply not loading the words the
projection never reads, but is marginal because those words aren't the post-A bottleneck. **A broadcast hint on
a storage binding does not exist.** **Dispatch-reshape** collapses into A + payload-shrink (the broadcast is
intrinsic to "many verts share one cluster transform").

Recommendation: **ship Lever A, tightened to load only the ~21 words the projection reads and vectorised to
`vec4<u32>` loads (both free), and stop there on the ctx broadcast.** The next real lever is Lever B
(compacted `projVertBuf` -> LLC-footprint), not a better broadcast.

---

## 1. PREMISE-AUDIT (both firings, as required)

### 1a. First firing — after understanding the problem + surveying the feature space, before ranking

**Is the metric right?** Yes. The bottleneck is not bandwidth and not ALU: the counters are buffer-READ-limiter
92 % + LLC 96 % while buffer-READ-*util* is only 40 % and ALU-limiter 28 % (`PROPOSAL-nanProjectVerts.md` §2a).
That signature — *limiter pegged, byte-utilisation moderate* — is **load-instruction-issue-rate / LLC-request-rate
bound**, i.e. the count of load *instructions* to hot cache lines, not the bytes moved. The correct optimisation
metric is therefore **global load-issues per cluster**, and every avenue below is judged on how far it drives
that number *and on which HW unit the residual lands*, not on "bytes saved". Grounded.

**Is the framing right — is "broadcast read" even the real problem?** Yes, and it is quantified: ~94 % of the
~9 955 load-issues/cluster are workgroup-uniform re-reads (ctx 35 words x 255 threads ~ 8 925 = 88 %; `ci` ~
255; `vcompact` ~ 255), against only ~520 genuinely-scattered vertex-gather issues (`PROPOSAL` §2b table). So
the figure to attack is the ~9 435 broadcast issues, and the *floor* any fix converges to is the ~520
vertex-gather issues that are irreducible inside this split. That ~520 floor is the yardstick for "materially
below A".

**Did I survey the whole feature space, not just A's neighbourhood?** The design space for "N threads of a
cluster each need the cluster's transform" is exactly four points: (i) global re-read per thread — the current
bug; (ii) fetch once into **threadgroup memory**, share workgroup-wide — Lever A; (iii) fetch once per
**subgroup**, share via registers — avenue 2; (iv) **constant-cache** hardware broadcast — avenue 1. Points
(ii)/(iv) give ~1 fetch/word/*workgroup*; (iii) gives ~1 fetch/word/*subgroup* = 8x/workgroup on Apple; (i) is
255x/workgroup. Payload-shrink (avenue 3) is an orthogonal multiplier on the word count for *any* of (ii)-(iv).
Cleared to rank.

### 1b. Second firing — before concluding "A is the ceiling / complete elimination not possible"

**Go up a level: is the *setup* that generates this broadcast the real flaw?** Yes — and this is the honest
finding. The ctx lives in a **global `var<storage,read>` buffer** and is **re-read per thread** for exactly one
reason: `ClusterCtx.ts` was split out of world1 to shed registers (80 -> 56, memory §VIS-BUFFER REWRITE), so the
transform inputs no longer live in the raster's registers/threadgroup — they had to be parked in DRAM and
re-hydrated. **The true elimination is to keep the ctx off DRAM entirely (in registers / threadgroup, computed
in-pass) — which is precisely what the OLD `wgcache` path in `NaniteRaster.ts:867-980` did (thread-0 `makeCtx`
-> `workgroupArray` -> barrier -> all threads read shared), i.e. Lever A *fused into the raster*.** That fusion is
**exactly the refusion the task forbids by decree** (it re-raises world1's register count and kills its
occupancy). So the generating context IS the flaw, it IS changeable, and changing it (refusion) is off the table
by user law. Given that constraint fixed, the question legitimately drops to "what is the best broadcast *within
a separate pass*" — and the audit has cleared the setup as (deliberately) fixed. Surfaced for the record, not
actioned. Proceeding to rank against the specs, not against assumption.

**Re-check each "blocked" claim against the real spec before I write it down:** the only avenue I will mark
"blocked" is avenue 1 (uniform/constant-cache). I verified its blocker against the WebGPU spec limits table
(`maxUniformBufferBindingSize`), the WGSL->MSL address-space mapping, and the WebGPU bind-group/dynamic-offset
model (offsets are set per `setBindGroup`, i.e. per dispatch, not per workgroup) — §2.1 below. Not hand-waved.

---

## 2. AVENUE-BY-AVENUE

Notation: **global load-issues/cluster** is the bottleneck metric. Current ~ 9 955 (~9 435 broadcast + ~520
vertex gather). Workgroup = 255 threads; Apple subgroup size = 32 => 8 subgroups/workgroup. ctx = 35 words (of
which the projection actually *reads* ~21 — see avenue 3).

### 2.1 Avenue 1 — Uniform / `constant` address space (the true hardware broadcast) — **BLOCKED**

**Mechanism (real, and it IS the ideal).** On Apple GPUs the Metal `constant` address space is *the* broadcast
path: read-only memory "optimized for data being broadcast to every thread"; when a value is read uniformly, the
hardware can **preload it into constant registers** so the ALU reads it with *zero* per-lane memory issue.
Apple, WWDC16 §606 *Advanced Metal Shader Optimization*: *"Rather than loading through the constant address
space, what we can actually do is take your data and put it into special constant registers that are even faster
for the ALU to access"* — this is buffer preloading, and it is **specific to `constant`**; the `device` address
space (= WGSL `storage`) gets **no** such preload and is the space Apple explicitly tells you to use when
"different threads ... access the buffer using an index such as the vertex ID ... or the thread position in grid"
(Metal Shading Language Specification, *Address Spaces*; WWDC16 §606). WebGPU exposes this: a WGSL
`var<uniform>` is compiled by Tint/Dawn (and Naga) into the Metal **`constant`** address space (gpuweb issue
#2559, *Dynamic indexing in Uniform address space* — the whole issue exists *because* uniform -> constant). The
current ctx read is `var<storage,read>` -> Metal `device` -> N issues, no preload: exactly the 92 %/96 % counters.
So on paper, moving the ctx into a **uniform** buffer read uniformly by `itemIdx` (which *is* workgroup-uniform)
would give the true one-fetch-all-lanes broadcast the question asks for — better than A, better than subgroups.

**Why it is blocked — three independent, spec-cited walls, any one fatal:**

1. **Size.** WebGPU `maxUniformBufferBindingSize` default = **65 536 B (64 KiB)** (WebGPU spec, *Limits*
   table; corroborated by web3dsurvey's device census — 65536 is the near-universal tier). The per-cluster ctx
   buffer is `QRASTER_CAP(1 M) x 35 x 4 B = 140 MB` allocated, and the *live* working set is ~150 K visible
   clusters x 35 x 4 B ~ **21 MB** (`ClusterCtx.ts:60`, `PROPOSAL` §2c). 64 KiB holds **16 384 u32 = 468
   clusters' worth of ctx** — <0.32 % of a frame. The whole thing cannot be bound as a uniform buffer. Raising
   the limit via `requiredLimits` does not save it: Apple's `constant`-buffer argument size is itself bounded
   (historically 64 KB on the Metal constant path) and, even generously, nothing exposes a 21 MB constant
   binding — 21 MB is orders of magnitude past any constant-cache-eligible size, so even if bound it would spill
   to `device` and lose the broadcast. Dead on size alone.

2. **Per-workgroup windowing is impossible in one indirect dispatch.** The obvious rescue — bind a 64 KiB
   *window* of the ctx and slide it per cluster via a **dynamic offset** — cannot work here. WebGPU dynamic
   offsets are arguments to **`setBindGroup(index, group, dynamicOffsets)`**, applied **once per bind-group
   set = once per dispatch** on the CPU (WebGPU spec, *GPUBindingCommandsMixin.setBindGroup* /
   `hasDynamicOffset`). `nanProjectVerts` runs as a **single `dispatchWorkgroupsIndirect`** whose workgroup
   count is a *GPU-computed* `itemCount` (`Project.ts:383,386` `setIndirectDispatch(rasterDispatchFullAttr)`).
   The bind group is bound **once for all workgroups**; there is no GPU-side per-workgroup rebind, and the CPU
   does not know the cluster count (it's indirect). You cannot give workgroup *k* a different uniform window.

3. **The two-level / batched rescue destroys the model.** To make windows work you would replace the one
   indirect dispatch with ~`ceil(150 K / 468) ~ 320` *fixed* dispatches, each a CPU `setBindGroup` with its own
   dynamic offset + `dispatchWorkgroups`. That (a) throws away the GPU-driven single-dispatch design, (b) forces
   the CPU to issue 320 bind-group+dispatch pairs/frame over-provisioned to a fixed cluster budget it must guess,
   and (c) still needs a GPU copy pass to gather the *visible* clusters' ctx into contiguous 64 KiB windows keyed
   by GPU-computed indices. This is a large regression to remove a broadcast that Lever A already removes for
   free. Not viable.

**Also checked (and rejected) as constant-cache substitutes:** WebGPU has **no push constants** (Dawn's
`chromium_experimental_push_constants` is ~128 B and per-pipeline, not per-cluster) and pipeline-overridable
constants / module `const` are compile-time only — none can carry per-cluster data. (WebGPU spec has no
push-constant surface; Dawn extension docs.)

**Verdict A1:** the *ideal* mechanism, **genuinely unavailable** in WebGPU for a 21 MB GPU-driven per-cluster
working set. Reduction beyond A: N/A (cannot be built). This is the "go up a level" finding: the constant-cache
broadcast is real on Metal; WebGPU simply does not expose enough of it (64 KiB cap + no per-workgroup rebind).

### 2.2 Avenue 2 — Subgroups (`subgroupBroadcast` / `subgroupShuffle`) — **AVAILABLE, but not materially better than A**

**Availability — fully grounded, and usable in THIS codebase:**
- **WebGPU/Dawn:** the `subgroups` feature **shipped in Chrome/Dawn stable in Chrome 134** (Mar 2025) after the
  Chrome 128-131 origin trial (developer.chrome.com/blog/new-in-webgpu-134; chromestatus feature
  5126409856221184). Dawn's Metal backend implements it (Metal 2.1+ simd-scoped permute/broadcast; the M1 Max is
  Metal 3). WGSL enables it with `enable subgroups;` (gpuweb subgroups proposal).
- **three.js r184 (0.184.0):** ships `SubgroupFunctionNode` with TSL nodes `subgroupBroadcast(e,id)`,
  `subgroupBroadcastFirst(e,id)`, `subgroupShuffle(v,id)`, `subgroupElect()`, etc.
  (`node_modules/three/src/nodes/gpgpu/SubgroupFunctionNode.js:191,401`), exported through `Three.TSL.js`. The
  WGSL builder **auto-emits `enable subgroups;`** whenever `renderer.hasFeature('subgroups')`
  (`WGSLNodeBuilder.js:1643-1648`, `1488-1492 enableDirective`), and maps the node method straight through to
  the WGSL builtin (`_getWGSLMethod`).
- **This device already has it.** `Engine.ts:142-147` requests **all adapter features**
  (`requiredFeatures: [...adapter.features, 'timestamp-query']`), so `subgroups` is enabled iff the Metal
  adapter reports it — and it does on Apple Silicon. Independent confirmation it is *live* here:
  `gpu/EnableF16.ts:9` already treats **three's own `enable subgroups;`** as a directive that WILL be present in
  emitted WGSL (it is the reason the f16 patch must be a directive, not an alias). So subgroups is not
  hypothetical in this build.
- **Apple subgroup size = 32** on all shipped Apple Silicon (metal-benchmarks, dougallj/applegpu; gpuweb
  #3950). NOTE: Not queryable before pipeline compile on Metal (subgroups proposal §Metal caveats); WebGPU exposes
  `adapter.info.subgroupMinSize/subgroupMaxSize` but the actual per-pipeline size is fixed at 32 here.

**Mechanism.** Before the per-thread early-outs (workgroup-uniform region, all lanes active), one lane per
subgroup loads each ctx word from the storage buffer, then `subgroupBroadcast(word, 0u)` propagates it to the
other 31 lanes **in registers** — no threadgroup memory, no `workgroupBarrier`. `id = 0u` is a **const-expression**
(required: the subgroups proposal mandates `id` be const for `subgroupBroadcast`; use `subgroupShuffle` for a
dynamic lane — not needed here). Uniformity is fine: the proposal is *permissive* and "all implementations
produce portable results when the workgroup never diverges" — and here the load+broadcast sits in fully-uniform
control flow ahead of any `returnIf`, so it is portable.

**Reduction vs A — the honest arithmetic (this is the crux):**

| variant | global ctx load-issues/cluster | barrier | threadgroup mem | residual unit |
|---|---:|---|---|---|
| current | ~9 435 | no | no | LLC/read-issue (pegged) |
| **Lever A** | **~39** (39 words, 1 lane each, ~3 cache lines) | 1 | ~156 B | threadgroup-load (idle) |
| **Subgroups** | **~312** (39 words x 8 subgroups) | 0 | 0 | shuffle/ALU (idle) |

Subgroups issue **8x more** global ctx loads than A (one fetch per *subgroup*, not per *workgroup*). Both land
below the ~520 vertex-gather floor **for A decisively (39 << 520); for subgroups only marginally (312 ~ 0.6x
floor)** — i.e. under subgroups the ctx broadcast is still ~37 % of all global loads, whereas under A it is
~7 %. So **on the metric that is the bottleneck, A strictly dominates subgroups.** Subgroups' *only* advantages
are (a) no `workgroupBarrier` and (b) no threadgroup-load traffic — but the counters show **neither is a
limiter** (threadgroup-load is not among the pegged units; the profile shows no barrier-stall signature — it is
read-issue + LLC). So those advantages **do not cash out** on this shader. Subgroups would win only in a
different regime (barrier-bound, or threadgroup-capacity-bound) — not this one.

**Parity:** identical — same words, same bit-casts, `subgroupBroadcast(x,0u)` returns lane-0's exact bits =>
byte-identical `xi/yi/dz`. **Register cost:** a few regs for the broadcast temporaries; ALU is idle so
irrelevant. **Feasibility:** high — TSL nodes exist, directive auto-emitted, feature already on.

**Verdict A2:** a legitimate, buildable **barrier-free alternative** to A, but **not materially better than A**
for this issue-rate-bound bottleneck (8x more global loads; its barrier/threadgroup savings target non-limiters).
Consider it only if a post-A re-profile shows the `workgroupBarrier` or threadgroup-load unit has become the new
ceiling (the current counters say it won't).

### 2.3 Avenue 3 — Shrink the broadcast payload — **FREE, stacks into A, but marginal post-A**

**What the projection actually reads (from `NaniteFetch.ts` + `Project.ts`).** After the cluster-uniform
early-outs (voxel `rU(10)==7`, HW-routed `rU(11)==1`), the hot **mesh** path is
`fetchWorldVertByIndex -> explicitWorldByIndex` (`NaniteFetch.ts:515-549`), which consumes only:
`A.xyzw` (4) + `B.xyzw` (4) + `yawSc.cy,sy` (2) via `instTransformPoint`; `channel` (1) to branch
trunk/leaf/grass; and, when `hasWind`, the 10 wind scalars `h0,dirX,dirY,leanBase,swayABase,swayPhase,ph,
branchBase,flutBase,swayXPhase` (`Project.ts:270-281`). Plus `triStart`/`triCount` (2) for gating/fallback and
`isHF` (1) for the branch. **~21 words** — the transform's *hot* set is `A+B+yawSc+channel+wind = 21`.
The other **14** ctx words are for paths the projection never takes: `gx,gz,qxw,oX,oZ,cell` (heightfield
placement — terrain is the per-corner minority via `hfWorld`), `meshId,matClass` (voxel early-out only, already
resolved), `twoSided` (classifier-only), `isDAG` (unused here). So the projection re-broadcasts **~40 % dead
weight**.

**Why it's free and where it stacks.** Lever A's cooperative fill should load **only the ~21 words it uses**,
not all 35 — a straight edit of A's fill loop, **zero extra memory, zero extra pass** (the ClusterCtx buffer
already holds all 35; A just skips 14 of them). This trims A's cooperative global loads 39 -> ~25 and its
shared-mem footprint 156 B -> ~100 B. **A bake-a-compact-payload variant** (emit a 21-word projection-only buffer
from `ClusterCtx`) is NOT worth it: it *adds* ~12.6 MB and a write, to save loads that are already below the
floor. And baking a *fused matrix* is a **loss**, not a win: `A(4)+B(4)+yawSc(2)=10` words already encode the
instance transform more compactly than a 3x4 (12) or 4x4 (16) matrix, and the yaw reconstruction is ALU (idle),
not memory; folding `cam.vp` in would defeat the per-vertex world-space wind and only grow the payload.

**Verdict A3:** **do it, as part of A** (load the ~21-word subset), because it is free; but its standalone value
is small — the words it trims aren't the post-A bottleneck. Multiplies *any* broadcast scheme, doesn't change
the ranking.

### 2.4 Avenue 4 — A broadcast hint on the storage binding — **DOES NOT EXIST**

There is **no** WebGPU/WGSL/Tint/Dawn decoration that makes a uniformly-indexed `var<storage,read>` device load
broadcast. WGSL has `@group/@binding` and access mode `read`/`read_write` only; no "uniform/broadcast" qualifier
(WGSL spec, *Address Spaces* & *Attributes*). `read`-only already applies and does **not** trigger the
Metal `constant` preload — the storage space maps to Metal `device`, which by Apple's own guidance is the
*non-broadcast* space (§2.1). Tint does not promote a `device` load to `constant` even when the index is provably
workgroup-uniform (the profile is the proof: `itemIdx` is `workgroup_id`-derived and uniform, yet the compiler
still issues N loads). The only lever in this family is the **opposite** direction — the robustness `min(idx,
len-1)` **clamps** Tint inserts on every access cost ~2.8 % of ALU (`PROPOSAL` §3.3, "Lever E"); dropping them
(indices are provably in-range) is a free ~2-3 %, but it's clamp-removal, not a broadcast hint, and ALU isn't the
wall. **Verdict A4:** no such feature; nothing to build.

### 2.5 Avenue 5 — Restructure the dispatch so the ctx isn't re-read per thread — **collapses into A + A3**

A **flat per-vertex dispatch** (one thread per unique vert globally, with a vert->cluster map) does **not**
dissolve the broadcast: to keep it efficient you'd sort verts by cluster, so neighbouring threads still share one
cluster's transform and still each need it — the same many-verts-share-one-ctx pattern, now *without* the natural
one-workgroup-per-cluster grouping that makes the cooperative load trivial. **Precomputing the transform product
in `ClusterCtx`** so threads read a smaller result is just avenue 3 (already covered), and its extreme (bake a
full clip matrix) is a memory *loss* (§2.3). The broadcast is **intrinsic** to "a cluster's ~130 verts each need
the cluster transform"; the only ways to cut it are *read-once-and-share* (A / subgroups) or *make-it-smaller*
(A3) — both already on the table — or *constant-cache* (A1, dead). The sole true-dissolution restructure is
keeping the ctx in registers/threadgroup in-pass = **refusion**, which is banned. **Verdict A5:** no new lever;
folds into A + A3 without refusion.

---

## 3. VERDICT & RANKING

**Complete elimination (a true constant-cache / one-fetch-all-lanes hardware broadcast, no barrier, no
threadgroup-load) is NOT achievable in WebGPU + WGSL + Dawn on Apple/Metal for this data.** The only mechanism
that delivers it — the Metal `constant` address space via a WebGPU **uniform** buffer — is blocked by
`maxUniformBufferBindingSize = 64 KiB` (spec default) against a ~21 MB per-cluster working set, and by the
GPU-driven single indirect dispatch's inability to re-window a uniform buffer per workgroup (dynamic offsets are
per-`setBindGroup`, not per-workgroup). Cited, not assumed (§2.1).

**Lever A (threadgroup-shared cooperative load + barrier) is the practical floor — and it is close to the
theoretical one.** It cuts global ctx load-issues ~240x (~9 435 -> ~39/cluster), landing **below the irreducible
~520 vertex-gather floor**. Because the ctx broadcast then sits *beneath* the floor, it stops being the limiter,
and **no available avenue can materially beat A on this bottleneck** — the shader re-limits on vertex gather /
writes / LLC footprint, which is **Lever B's** domain (compacted `projVertBuf`), not a better broadcast.

**Ranked, within the no-refusion constraint:**

1. **Lever A, tightened (LEAD).** Cooperative load into `workgroupArray` + one `workgroupBarrier` (the shipped
   `NaniteVertexCache`/old-`wgcache` pattern), **loading only the ~21 words the projection reads (A3, free)** and
   **vectorised to `vec4<u32>` loads** (39 scalar issues -> ~10 vector issues, also free). Kills the measured
   92 %/96 % limiters; ~39 -> ~10 global ctx issues, far below the vertex floor. This *is* the floor.
2. **Subgroups (ALTERNATIVE, not an upgrade).** Buildable today (Chrome 134+/Dawn-Metal, three r184
   `subgroupBroadcast`, feature already requested in `Engine.ts`), barrier-free, but leaves **8x more** global
   ctx loads than A and saves only non-limiters (barrier, threadgroup-load). **Incremental gain over A ~ 0** for
   this issue-rate-bound case. Reach for it *only* if a post-A re-profile shows the barrier/threadgroup-load is
   the new ceiling — which the current counters do not indicate.
3. **Payload-shrink (A3) + clamp-drop (Lever E).** Free, fold into A; small standalone value (they trim
   non-bottleneck traffic/ALU).
4. **Uniform/constant-cache (A1):** best-in-theory, **unbuildable** here (64 KiB cap + single indirect dispatch).
5. **Storage broadcast hint (A4):** does not exist.
6. **Dispatch reshape (A5):** collapses into A + A3; true dissolution = refusion = banned.

**Incremental gain of the best non-A option over A:** effectively **none** on this bottleneck. A already pushes
the ctx broadcast below the vertex-gather floor; subgroups/payload-shrink move numbers that are no longer the
limiter. **A is the ceiling without refusion, and the reason is precise:** the only thing that beats A's
~1-fetch/word/workgroup is a constant-cache broadcast (~0 amortised via register preload), and WebGPU's 64 KiB
uniform cap + GPU-driven indirect dispatch make that unreachable for a 21 MB per-cluster set.

**Surfaced (out of scope, for the record):** the *only* way to remove the broadcast *entirely* is to stop
parking the ctx in DRAM — keep it in registers/threadgroup computed in-pass = **refusion** into world1, which the
old `wgcache` path did and which is banned by decree because it re-raises world1's registers. So the broadcast is
a structural cost of the deliberate ClusterCtx split; A minimises it, it cannot be zeroed without paying that
register cost back.

---

## 4. Citations

Capability claims are grounded as follows (flagged where I could not runtime-verify):

- **Metal `constant` = broadcast/preload; `device` = per-thread, no preload:** Apple WWDC16 §606 *Advanced Metal
  Shader Optimization* (developer.apple.com/videos/play/wwdc2016/606; transcript asciiwwdc.com/2016/sessions/606
  — "special constant registers ... faster for the ALU"); Apple *Metal Shading Language Specification*, §Address
  Spaces (developer.apple.com/metal/Metal-Shading-Language-Specification.pdf); Apple Tech Talk 111373 *Learn
  performance best practices for Metal shaders*.
- **WGSL `var<uniform>` -> Metal `constant`:** gpuweb issue #2559 *Dynamic indexing in Uniform address space*
  (github.com/gpuweb/gpuweb/issues/2559); Tint/Dawn + Naga behaviour.
- **`maxUniformBufferBindingSize` default 65 536 B:** WebGPU spec *Limits* table (w3.org/TR/webgpu, §3.6.2);
  device census web3dsurvey.com/webgpu/limits/maxUniformBufferBindingSize (65536 dominant tier).
- **Dynamic offsets per `setBindGroup`/dispatch, bind group bound once per indirect dispatch:** WebGPU spec
  *GPUBindingCommandsMixin.setBindGroup* / `GPUBindGroupLayoutEntry.hasDynamicOffset` (w3.org/TR/webgpu).
- **Subgroups shipped Chrome/Dawn 134 stable; feature enabled with `enable subgroups;`:**
  developer.chrome.com/blog/new-in-webgpu-134; chromestatus feature 5126409856221184; gpuweb subgroups proposal
  (github.com/gpuweb/gpuweb/blob/main/proposals/subgroups.md).
- **`subgroupBroadcast` id must be const-expression; `subgroupShuffle` for dynamic id; permissive uniformity
  (portable when workgroup doesn't diverge):** gpuweb subgroups proposal (same URL).
- **Apple Silicon subgroup size = 32; not queryable pre-compile on Metal:** github.com/philipturner/metal-
  benchmarks; dougallj.github.io/applegpu/docs.html; gpuweb issue #3950 *Considerations for subgroups*;
  subgroups proposal §Metal.
- **three.js r184 subgroup TSL nodes + auto-`enable`:** `node_modules/three/src/nodes/gpgpu/
  SubgroupFunctionNode.js` (subgroupBroadcast L191/401, subgroupShuffle L192/412, subgroupElect L168/210);
  `node_modules/three/src/renderers/webgpu/nodes/WGSLNodeBuilder.js:1488-1492,1643-1648`.
- **This device requests all adapter features (so `subgroups` is on iff the adapter exposes it) + three already
  emits `enable subgroups;`:** `src/core/Engine.ts:142-147`; `src/gpu/EnableF16.ts:9`.
- **Cooperative-shared + barrier is a shipped primitive here (Lever A precedent):** `src/nanite/
  NaniteVertexCache.ts:71-88`; `src/nanite/NaniteRaster.ts:881-982` (old `wgcache` = Lever-A-fused-into-raster =
  the banned refusion).

**Could not runtime-verify (flagged):** (i) that this specific M1 Max Dawn/Metal adapter reports `subgroups`
(strongly implied by `Engine.ts` grabbing all features + the `EnableF16.ts` comment treating `enable subgroups;`
as live, but not observed in a capture in this session); (ii) the empirical A-vs-subgroups delta and A's exact
post-fix ceiling — both need a **fresh profile after Lever A lands** (per the standing FRESH-DATA rule); the
ranking above is from the load-issue model + counters in `PROPOSAL-nanProjectVerts.md` §2, not a post-A capture.
