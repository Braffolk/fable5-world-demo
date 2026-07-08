# PROPOSAL — 3× `nanProjectVerts` (the projection whale), ON TOP of Lever 1

**Task #76 continuation. Date 2026-07-08. Target: `src/nanite/raster/Project.ts` (`nanProjectVerts`).**
Explore + propose only — no source edited, nothing committed.

Trace: `/private/tmp/laas_trace-2026-07-08T02-03-38-c000{,-perf}.gputrace`
Results (this trace, `run_all.sh`): `profile-results-20260708-021436/` (POST-Lever-1).
Cross-check (a 55-min-earlier PRE-Lever-1 sibling trace): `profile-results-20260708-014947/`.

---

## 1. PREMISE-AUDIT (fires twice, as required)

### 1a. First firing — right shader / right metric / which Lever-1 state?

**Right shader.** Bin `1b24`, named `nanProjectVerts` by NodeBuffer-100% match; MSL signature confirms it:
`cam.vp * vec4(wv,1)` → `w > 1e-4` gate → `/w` → `(ndc.xy+1)*0.5*(W,H)` → `*256 round` → `i32`, with
the `NEAR_SENTINEL 0x7f800001` (`= 2139095041`, appears 8×) stored in the dz word. This is `projectVert()`
verbatim. Not a look-alike.

**Right metric — NOT the encoder-label ms.** `summary/timing_table.txt` shows
`Dawn_ShaderModule_compute_nanProjectVerts::main = 0.003 ms / 1 invoke`. That is the **encoder-label
artefact** the README warns about (§5), not real time. The trustworthy figure is the PC-sample histogram:
**nanProjectVerts = 17.4 % of sampled GPU time (rank #2), 2,332,945 samples.** (In the PRE-Lever-1 sibling
trace it was **23.0 %, rank #1** — see 1c.)

**TBDR caveat respected.** I do not add per-pass GpuProfiler timestamps (render‖compute overlap makes them
non-additive on Apple). All time claims are PC-sample %GPU or the user's Xcode-timeline hardware counters.

**Which Lever-1 state does the trace reflect? → POST-Lever-1 (confirmed structurally).** The current trace's
`nanProjectVerts` MSL binds **7** storage buffers — including a 7th, `NodeBuffer_285393` (= `gpu.vcompact`) —
and contains **8** NEAR_SENTINEL stores + **16** `*256.0f` rounds. That is exactly the Lever-1 shape:
`isHF` emitPerCorner (3 sentinels) + the `vcCount>0` stride loop `ceil(VCACHE_VERTS/MAX_CLUSTER_TRIS)=2` iters
(2 sentinels) + the `vcCount==0` mesh fallback emitPerCorner (3 sentinels) = 8. The PRE-Lever-1 sibling MSL
binds only **6** buffers (no vcompact) and has **3** sentinels / **6** rounds (pure per-corner). So Lever 1
IS in the measured shader, and **my 3× is relative to this POST-Lever-1 state.** (Workgroup = 255 threads:
the MSL carries `[[max_total_threads_per_threadgroup(255)]]`, i.e. this build ran `bits==8` ⇒
`MAX_CLUSTER_TRIS=255`, `VCACHE_VERTS=382`, `MAX_CLUSTER_VERTS=512`.)

### 1b. Second firing — before any "intrinsic": is the flaw one level up?

The reflex would be "projection is a 58–64-reg transform floor, that's intrinsic." **The counters refute that
outright:** ALU limiter 28 %, ALU util 26 %, F32 util 9 %, F16 util 1 % — **the ALU is nearly idle.** The
transform math is NOT the wall. So the cost is not intrinsic to "projecting a vertex"; it is intrinsic to
**how many memory-load instructions the pass issues**, and that is a *structural* property of the dispatch
(one workgroup/cluster, every thread re-reads the same per-cluster context), not of the arithmetic. That
structure is changeable → not intrinsic. Cleared to proceed.

**Going up one more level (surfaced, not actioned — out of task scope):** `projVertBuf` exists *only* to
bridge three separate passes (Project writes it; Classify + Mid read it). The engine **already has** a
cooperative "transform each unique vert once into **workgroup shared memory**" primitive —
`NaniteVertexCache.ts` (`workgroupArray('vec3', VCACHE_VERTS)` + `workgroupBarrier`). If Project+Classify(+Mid)
were fused, the projected corners could live in threadgroup memory and the **1.13 GB global round-trip would
not exist at all.** The vis-buffer rewrite deliberately split them to cut world1 80→56 regs, and the task
scopes re-fusion out — but the counters say the split's *cost* (a memory-bound pre-pass writing/reading 1.13 GB)
is now the #1 whale, so whether the split is still net-positive is a real open question for a later arc. I
respect the scope and propose within the split below.

---

## 2. GROUNDED DIAGNOSIS — where the traffic goes

### 2a. The counters (user, Xcode timeline, during projection) — the anchor
| counter | value | reading |
|---|---|---|
| kernel occupancy | 53 % | not the constraint (see below) |
| allocated registers | 64 | register-limited occupancy, but **irrelevant** — ALU is idle |
| ALU limiter / util | 28 % / 26 % | **nearly idle** — NOT ALU/register/occupancy bound |
| F32 / F16 util | 9 % / 1 % | idle |
| GPU read | ~200 GB/s | 2× the writes |
| GPU write | ~100 GB/s | writes are secondary |
| **buffer READ limiter** | **92 %** | **THE bottleneck — load-issue throughput pegged** |
| buffer READ util | 40 % | bytes moderate; **issue-slot** pegged ⇒ *many small/cached loads* |
| buffer WRITE limiter / util | 50 % / 28 % | writes real but not the wall |
| MMU limit/util | 40 % | some scattered-gather TLB pressure |
| **last-level-cache util/limit** | **96–97 %** | **THE bottleneck — LLC request-rate saturated** |

Read this precisely: **buffer-READ-limiter 92 % + LLC 96 %, but buffer-READ-*util* only 40 %.** Limiter
pegged while byte-utilisation is moderate ⇒ the wall is the **number of load *instructions issued* to hot
cache lines**, not raw DRAM bandwidth. That is the fingerprint of a **broadcast read** — many threads issuing
loads of the *same* addresses.

### 2b. What issues those loads — the structural cause (per-line + mechanism)

The dispatch is **one workgroup per cluster, 255 threads**. Before any branch, **every one of the 255 threads
decodes the full 35-word per-cluster ctx** (`base = itemIdx*35`, then 35 loads of `clusterCtx[base+i]`,
`Project.ts` `rU`/`rF`, MSL lines ~420–450) — and re-reads `qRaster[itemIdx+1].y` (= `ci`) and
`vcompact[ci*2..+1]`. `itemIdx`/`ci` are **workgroup-uniform**, so all 255 threads load the *identical*
addresses. Load-issue budget per SW cluster/frame:

| stream | buffer | load-issues / cluster | share | nature |
|---|---|---:|---:|---|
| **ctx decode (35 words × 255 threads)** | 316362 | **≈ 8,925** | **~88 %** | **broadcast (same 35 addrs)** |
| qRaster `ci` + itemCount (× 255) | 316154 | ≈ 255 | ~3 % | **broadcast** |
| vcompact `(vMin,count)` (× 255) | 285393 | ≈ 255 | ~3 % | **broadcast** |
| vertex gather (Lever-1: ~130 unique × 4 words) | 285371/285374/285377 | ≈ 520 | ~5 % | scattered (already deduped) |
| **total loads** | | **≈ 9,955** | | |
| projVertBuf stores (~130 × 3) | 316369 | ≈ 390 | (write side) | dense-within-cluster |

**~94 % of the load-issues are broadcast re-reads of workgroup-uniform data** (ctx 88 % + qRaster 3 % +
vcompact 3 %). This is exactly why READ-limiter = 92 % and LLC = 96 % (the same few ctx cache lines serviced
255× per cluster × ~150 K visible clusters ≈ **1.3 billion redundant ctx load-issues/frame**), while READ-util
is only 40 % (few distinct bytes). **The per-line runtime corroborates:** ctx-reads (316362) sum to **8.8 %**
and qRaster (316154) to **4.6 %** of the *whole shader* directly — vs the actual scattered **vertex gather at
only 0.6 %** and **projVertBuf writes at only 2.0 %**. Add the ~17.5 % parked in prologue/entry glue
(`line 0` 8.3 %, `entry` 9.2 %) that on a load-issue-bound shader is where the ctx-load stalls settle.

**Conclusion (bandwidth vs latency vs occupancy):** neither classic bandwidth-bound (util 40 %) nor
occupancy-bound (ALU idle, occ 53 % is a symptom not a cause). It is **load-ISSUE-throughput + LLC-request-rate
bound, and ~94 % of the issues are redundant broadcast reads.** The single highest-leverage fix is to **stop
re-issuing the per-cluster ctx/qRaster/vcompact loads once per thread.**

### 2c. projVertBuf footprint / emptiness (the memory-arc angle)
Flat layout: `PROJ_CLUSTER_CAP(192K) × MAX_CLUSTER_VERTS(512) × 3 u32 = 1.13 GB`, but avg unique verts/cluster
≪ 512 (memory: ~82; even at ~130 for `bits==8`) ⇒ the buffer is **~75–84 % empty.** For *this* shader the write
stream is already dense *within* a cluster's 0..count region (write-once, contiguous slots) — which is why
writes are only 2.0 % per-line and the write-limiter is 50 %, not the wall. The emptiness hurts as (a) a **1.13 GB
footprint** the arc wants gone, and (b) **LLC working-set pressure** — a 1.13 GB region streamed through a
cache whose limiter is already at 96 %. Compaction is therefore a **memory-arc + LLC-footprint + reader-side**
win (§4 Lever B), not primarily a this-shader write-speed win.

### 2d. How many projected verts go unread? (bounds Lever 2)
The classifier (`NaniteRaster.ts` 1273–1293) reads **all 3 corners of every non-skipped tri** to compute
`nearOK` + winding + route. So projection output is **1:1 consumed as the classifier's cull input** — you
cannot skip projecting what the classifier reads. The genuinely-unread projections are only: (i) near-crossing
corners whose xi/yi are garbage-but-unread (only the NEAR_SENTINEL flag is read) — a small tail; (ii) verts in
clusters already skipped before Project (voxel-7, HW-routed slot-11 — **already** `returnIf`'d). Cluster classes
that are entirely culled upstream never reach Project. **This caps Lever 2's upside inside the current
split** (see §4 Lever D).

---

## 3. NONSENSICAL / SUSPECT STATS (with evidence)

1. **The whole ctx read is redundant 255× (the big one).** `clusterCtx[itemIdx*35 + 0..34]` is
   workgroup-uniform yet re-loaded by all 255 threads (MSL 420–450, per-line 8.8 %). ~1.3 B redundant
   ctx load-issues/frame. **This is the 92 % read-limiter / 96 % LLC, essentially by itself.**
2. **qRaster `ci` re-read per thread** (per-line 4.6 %, MSL line 410 `316154[itemIdx+1].y`) and **vcompact
   `(vMin,count)` re-read per thread** (285393) — same broadcast pathology, smaller.
3. **Tint robustness `min(idx, len-1)` clamp on *every* storage access.** MSL `min` helpers cost
   **2.1 % + 0.7 % + … ≈ 2.8 %** (lines 3245/3099/3199) purely for bounds-clamping loads that are provably
   in-range (all indices are `itemIdx*35+const`, `vi*6+const`, etc.). Dead ALU+dependency on the read path.
4. **`285374` per-vertex packed word read 3× for one vertex** (PRE-Lever-1 MSL 316/317/319 re-load the same
   `hfVerts[idx]` for `&8191`, `>>16`, `>>13&7`) — the compiler does not CSE storage loads across the three
   bitfield extracts. Terrain-path only; minor, but a free load-count cut if hoisted to one temp.
5. **~17.5 % parked in `line 0` + `entry` glue** on a shader whose ALU is idle — that is memory-stall
   attribution (the ctx loads at the top), i.e. the same disease as #1, not real prologue work.

---

## 4. PROPOSAL — ranked structural changes (each ~independent, stackable)

> Design constraint honoured everywhere: the projected `xi (i32) | yi (i32) | dz (f32/NEAR_SENTINEL)` records
> the classifier + Mid read back must stay **byte-identical**; any read-side change is spelled out.
> Memory ceiling: every lever is **≤ 0 MB** (three of four *free* memory).

### ★ Lever A (LEAD — the 3× driver): cooperative workgroup-shared ctx / qRaster / vcompact

**(a) Mechanism.** Load the per-cluster invariants **once per workgroup into threadgroup memory**, then decode
from there. Concretely: a `workgroupArray('u32', 35)` for ctx (+ 4 words for `ci`, `itemCount`, `vMin`,
`vCount`); the first 35 lanes cooperatively `shared[i] = clusterCtx[base+i]`; `workgroupBarrier()`; every thread
builds `A/B/yawSc/wind/triStart/...` from `shared[]` instead of from the global buffer. This is the **exact
pattern already shipping in `NaniteVertexCache.ts`** (`workgroupArray` + `workgroupBarrier`) and used in
`ClusterCtx.ts` — proven, not novel plumbing.

**(b) Cost removed.** The measured wall. Buffer-read load-issues/cluster **≈ 9,955 → ≈ 560** (35+4 cooperative
global loads + the ~520 real vertex gathers); the ~8,900 broadcast ctx issues move off the buffer-read/LLC path
onto the **threadgroup-load** unit (which is idle — not among the pegged limiters). Directly collapses
buffer-READ-limiter (92 %) and LLC (96 %).

**(c) Rough gain.** Removes ~88–94 % of the load-issues that constitute the bottleneck. Read-limiter should fall
from 92 % toward the vertex-gather floor (~10–15 %); the shader then re-limits on the *next* ceiling (LLC
footprint / writes / threadgroup-load). Expected **~2–2.7× on this shader alone** — the bulk of the 3×.

**(d) Memory delta.** **≈ 0 VRAM.** Threadgroup memory is on-chip: ~39 × 4 B = **156 B/workgroup** (vs the
~4.6 KB `NaniteVertexCache` already uses — trivially within the 32 KB threadgroup budget). No global allocation.

**(e) Register / occupancy.** ALU is idle so this is not why we do it, but reading ctx just-in-time from
threadgroup memory instead of holding 35 decoded values live can *lower* register pressure (64 → less) →
occupancy *up* from 53 %. Upside, not the objective.

**(f) Parity / bit-identity.** **Zero risk to the records.** Same 35 u32 words, same bit-casts, same
`projectVert()` → identical xi/yi/dz. The only new hazard is **barrier uniformity**: the cooperative load +
`workgroupBarrier` must execute in workgroup-uniform control flow, *before* the per-thread `returnIf`s. The
existing early-outs (`rU(10)==7` voxel, `rU(11)==1` HW-routed, `itemIdx≥cap`) are **cluster-uniform**, so the
structure is: cooperative-load → barrier → read the (now workgroup-shared) skip flags → uniform early-return →
work. `NaniteVertexCache` already does exactly this dance (its comment: "the `vcCount>0` test is UNIFORM ⇒ no
barrier"). No read-side change (Classify/Mid untouched).

**(g) Implementation sketch.** `src/nanite/raster/Project.ts` `kn = Fn(() => {...})`: hoist `itemIdx`
computation, add `const shCtx = workgroupArray('u32', 35)` (+ small companion for ci/itemCount/vMin/vCount),
cooperative fill guarded by `localTri < 35`, `workgroupBarrier()`, then repoint `rU`/`rF` at `shCtx` and `ci`
at the shared word. Import `workgroupArray, workgroupBarrier` from `three/tsl` (as `NaniteVertexCache` does).

---

### ★ Lever B (CO-LEAD — the memory-arc win): prefix-sum-compacted `projVertBuf`

**(a) Mechanism.** Replace the flat `itemIdx * 512` slot base with a **per-cluster exclusive-prefix-sum base**.
A cheap per-frame scan over the `itemCount` (~150 K) visible clusters computes
`clusterBase[itemIdx] = Σ_{j<itemIdx} vertCount[j]`, where `vertCount = vcompact.count` for mesh, `triCount*3`
for terrain / tooWide fallback. Project/Classify/Mid then address `clusterBase[itemIdx] + canonLocal` instead
of `itemIdx*512 + canonLocal`. Slots pack densely; the 512-stride emptiness vanishes.

**(b) Cost removed.** **Footprint: 1.13 GB → ~150–250 MB** (Σcount×3×4 B; ~150 K × ~130 × 12 B ≈ 234 MB, size
to ~256 MB for peak safety). That is the **LLC-footprint** relief the 96 % counter is asking for (the write +
reader streams pass through ~5× less resident memory) and it makes the **reader shaders** (Classify @1283–1289,
Mid @117–122) touch a dense, cache-line-packed region instead of a 512-stride sparse one.

**(c) Rough gain.** On *nanProjectVerts itself*, modest and honest: writes are only 2.0 % per-line and already
dense-within-cluster, so the direct write-speed gain is small; the win is the **LLC working-set** shrink (helps
the 96 % limiter that Lever A doesn't fully clear) — call it **~1.1–1.3×** stacked on Lever A — **plus** a
cross-shader speedup on Classify/Mid (they read projVertBuf) and the **−~900 MB** the memory arc wants.

**(d) Memory delta.** **≈ −900 MB (a large NEGATIVE — the double win the brief wants).** Adds only
`clusterBase` = `itemCount × 4 B` ≈ 768 KB and keeps the vertCount scan buffer (tiny). Net hugely negative.

**(e) Register / occupancy.** Negligible (one extra broadcast load of `clusterBase[itemIdx]` — hoist it into
Lever A's shared block so it costs nothing extra).

**(f) Parity / bit-identity.** The **values** are unchanged — only each cluster's *base offset* moves. Requirement:
Project (write), Classify (`NaniteRaster.ts` 1273), and Mid (`Mid.ts` 99) all currently compute
`recCluster = itemIdx * vertsPerCluster`; **all three must switch to `recCluster = clusterBase[itemIdx]`
reading the same buffer**, or records desync. `canonLocal` (mesh `vi−vBase` ∈ [0,count); terrain
`localTri*3+corner`) is unchanged, so `base+canonLocal` lands on the identical record. **Watch:** the per-cluster
region must reserve `vertCount ≥ max(canonLocal)+1` — for terrain that means `triCount*3` (not `count`), and the
`minU(...,511)` rogue clamp must become `minU(..., vertCount-1)`. Get the scan's per-class count right or you
get cross-cluster corruption (contained, but wrong pixels). Spell it into the scan.

**(g) Implementation sketch.** New tiny compute pre-pass `raster/ProjBase.ts` (or fold into `ClusterCtx`):
emit `vertCount[itemIdx]` then an exclusive scan → `clusterBaseAttr`. Thread it into `buildProject`,
Classify, and Mid via the bundle; replace the three `itemIdx*vertsPerCluster` sites; shrink the
`projVertAttr` allocation to the compacted cap. Cost of the scan: one pass over ~150 K u32 (~a few µs).

---

### Lever C (cheap, modest): close the tooWide fallback

**(a) Mechanism.** Clusters with `uniqueCount > VCACHE_VERTS (382 @bits8)` fall back to the **per-corner** path
(no dedup) — the widest leaf crowns re-fetch/re-project/re-write each shared vert ~3–6×. Extend vcompact
coverage: raise `VCACHE_VERTS` toward `MAX_CLUSTER_VERTS(512)`, or give the stride loop a full-count cap so the
widest crowns also dedup. **(b)** Removes the residual 3–6× vertex-read + projVertBuf-write redundancy on those
clusters. **(c)** Gain bounded by their frame share — vertex gather is only ~0.6 % of the shader post-Lever-1
and avg count (~82–130) ≪ 382, so tooWide is the tail; **quantify from the `populateVCompact` log line**
("`eligible …, tooWide>VCACHE_VERTS N, avg count …`", `GeometryRegistry.ts` ~2357) before investing — likely a
low-single-digit-% win, do it only because it's nearly free. **(d)** Memory ≈ 0 (loop-bound change; the shared
`workgroupArray` in NaniteVertexCache is already sized to VCACHE_VERTS — check its threadgroup budget if you
raise it). **(e)** none. **(f)** identical records (same dedup key, just covered by the fast path). **(g)**
`GeometryRegistry.populateVCompact` threshold + `Project.ts` stride-loop trip count.

### Lever D (bounded — do NOT over-invest): skip cull-doomed / sentinel verts

**(a) Mechanism.** Plan §6 "compacted-survivor" scheme: pre-cull tris (cluster-frustum / backface) *before*
projection and project only surviving verts. **(b)** would cut both reads and writes. **(c) BUT** §2d shows the
classifier reads *every* non-skipped corner as its cull input, and cluster-level culls already happen upstream
(voxel/HW skipped in Project) — so inside the current split the unread fraction is small (near-sentinel tail +
already-skipped classes). The ordering cost (an extra pre-cull pass + a compaction indirection) is real and the
upside is bounded. **Verdict: not worth it as a projection lever** while Lever A/B are on the table; its real
form is *far-field DAG coarsening* (fewer far tris to project at all), which is a different arc
(`base-raster-is-the-bottleneck`), not a `Project.ts` change. Surfaced, deprioritised.

### Lever E (free hygiene): drop robustness clamps on the read path
`min(idx, arrayLength-1)` costs ~2.8 % (§3.3) for bounds we can prove. If Dawn/Tint robustness can be disabled
for these read-only storage bindings (or indices proven in-range), that is a free ~2–3 %. Engine-level knob;
list as a quick add-on to Lever A. Memory 0, parity identical (same in-range values).

---

### Expected stack
Lever A (~2–2.7×, the broadcast-read wall) × Lever B LLC-footprint (~1.1–1.3×) + Lever C/E (~few %) →
**~2.6–3.3×** on `nanProjectVerts`, **plus −~900 MB VRAM** and a cross-shader speedup on Classify/Mid.
Lead with **A** (kills the measured 92 %/96 % limiters), land **B** alongside for the memory arc + LLC
footprint + reader-side, mop up with **C/E**. **Do not** chase registers/ALU/occupancy (idle) and **do not**
re-derive Lever 1.

---

## 5. Blocked on the manual Xcode export / to sharpen the diagnosis
- **Not blocked** on any missing export — the exported `-perf` bundle was present; `run_all.sh` produced full
  per-line runtime + MSL, and the user relayed the Xcode-timeline hardware counters. Diagnosis is grounded.
- **Would sharpen (not required):** a **per-stream bandwidth split** (ctx vs vertex vs projVert-write GB/s) — the
  tool's counter set here is only the 10 *limiter* percentages (`summary/counters.txt`); the per-command
  read/write **Rate** counters and **Threadgroup-load limiter** would let me confirm Lever A's post-fix ceiling
  numerically rather than by the load-issue model. Available only from Xcode's per-encoder counter timeline
  (the user's relayed values already carry the decisive ones: READ-limiter 92 %, LLC 96 %, ALU 28 %).
- **Would confirm Lever C sizing:** the live `populateVCompact` console line (tooWide count / avg unique) —
  read it from a dev run before tuning `VCACHE_VERTS`.
