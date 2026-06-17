# Forest frame 1244 — capture-vs-intent divergence report

Ground truth: WebGPU Inspector capture of `?scene=forest` full-`NaniteFrame` pipe, frame 1244.
Slices: `docs/perf-runs/2026-06-17-webgpu-inspector/slices/`.
Method: 8 per-resource-type subagents compared the captured GPU flow against the intended
architecture (docs NANITE-SPEC `D-N#` / ROADMAP / LOG + `src/nanite` + `src/core` + `src/render`).
This document consolidates, dedups, and ranks their findings.

Capture anchor counts (verified in `_doc0.json` / `summary.md`):
`createCommandEncoder=76, submit=76, beginComputePass=56, beginRenderPass=18` (+2 standalone
TRAA copy encoders = 76 total); `dispatch=56, draw=19, drawIndirect=1, totalTriangles=2001`;
`Buffer=275 / 2212.2 MB`, `Texture=57 / 858.8 MB`, `RenderBundle=248`; one validation error:
*"GPUBuffer was garbage collected without being explicitly destroyed."*

---

## 1. Executive summary — the things that actually matter

The capture **vindicates the core nanite design**: trees are rastered entirely in compute
(`nanRasterWorld1` SW indirect, submit 44 + one `nanHwPass` HW needle draw, submit 46) and
shaded by a **single fullscreen-triangle resolve** (`draw[3]` inside scene pass, submit 61).
The single-pass D-N45/PERF-VB4 world path is intact — there is **no second SW payload pass, no
`kRasterDepth2`, no chunk queue, no rejInst/rejClust buffers**. The user's worry that "multiple
full-res passes render the trees" is **refuted**: the only real indexed geometry in the whole
frame is the 1984-triangle SunSky background dome. The harness's "90% per-pixel, nothing left"
conclusion is correct *about active GPU pixel cost* — but it was **structurally blind** to two
classes of cheaper wins the capture exposes:

1. **76 submits/frame, ZERO batching (the invisible bubble).** Every one of the 56 compute +
   18 render passes is its own `createCommandEncoder → finish → queue.submit`; `multi-buffer
   submits = 0`. This is three.js-normal *per call*, but our code never uses the supported
   `renderer.compute([...])` array form that records many passes into ONE encoder. The
   dependent chains (39-pass cull BFS, 11-pass HZB) serialize with a submit bubble between each
   pass — an estimated ~2.3–6 ms/frame of pure scheduling overhead that is **invisible to both
   the per-pass `GpuProfiler` (it times only beginPass→endPass spans) and `cpu.submitMs` (CPU
   encode wall only)**. This is exactly the gap the prior harness review never looked at. The
   single most batchable cluster — the 11-level HZB (`nanHzbL0..L10`, submits 47–57) — collapses
   `11 → 1` submit with no semantic change.

2. **~1.5 GB of resident-but-dead "full-world" machinery in a terrain-less forest testbed.**
   The forest is supposed to be "200k trees and NOTHING else" (ForestScene header), yet it runs
   the *entire* terrain-genesis + atmosphere bake + impostor/foliage atlas pipeline at boot to
   produce maps it admits are "bound but NEVER sampled." Concretely: **~755 MB of terrain
   synthesis/erosion scratch buffers** (`boundInPasses:[]`, never freed), **~319 MB of bound-
   but-dead terrain heightfield/derived textures** (heightTex/normalTex/biomeTex/fieldsTex/noise
   — bound to the 3 hottest pipelines, every sampler in dead control flow), **201 MB of impostor
   atlases** and **21 MB of foliage cards** (`usedInPasses:[]`), and **~65 terrain/atmo compute
   pipelines + 24 VegLibrary render pipelines** compiled and resident but dispatched/drawn ZERO
   times. The lone GC-without-destroy validation error is the tip of this iceberg. None of this
   is on the per-frame critical path, but it pollutes every VRAM/"what does nanite cost"
   accounting and is the bulk of the object-count explosion that prompted this review.

3. **`compute_autoExposure` runs TWICE per frame** (submits 1 and 2, same pipeline 4530, same
   bindgroup, identical `[1,1,1]` dims) because `NaniteFrame.meter` is wired from **two** callers:
   `ForestScene.ts:217 engine.onUpdate(() => frame.meter(...))` AND `Engine.renderStep`'s own
   `this.post.meter(...)`. On readback frames (`frame % 15 == 0`) this *also* double-issues the
   cull/raster/shadow readbacks and double-writes HUD counters, and exposure adapts at 2× the
   intended rate. A one-line wiring fix.

4. **The BFS runs all 18 traverse iterations though the cut converges ~depth 9** (capture shows
   9 `nanTraverseAB` + 9 `nanTraverseBA`; LOG bw measured convergence at depth ~9 for this view-
   class). ~9 tail traverse + ~9 tail args passes walk an empty frontier — up to ~18 of the 56
   compute submits are post-convergence no-ops.

5. **Several deleted-path remnants survive as live per-frame work**: `depthV` (6.39 MB vis
   buffer) is allocated and write-cleared every frame by `nanVisClear` but read by NOTHING;
   `kRasterArgs` (submit 41) and the `rasterDispatch2` output of `kRasterArgs2` are dead in the
   single-pass world path; the HW needle pass clears+stores a 6.4 MB full-res color target it
   never writes (`colorWrite=false`).

**Bottom line for the perf investigation:** the active-GPU profile is honest, but the *frame
structure* hides a submit-overhead floor the per-pass profiler can't see, and ~1.5 GB of dead
residency that distorts every memory number. The cheap, low-risk wins (HZB batch, kill double-
meter, drop terrain residency in the testbed, BFS early-out) are all **structural**, not pixel-
shader micro-opt — measure them with a **whole-frame wall-clock delta**, not per-pass timestamps.

---

## 2. Cross-cutting themes (ranked by impact × confidence)

### THEME A — One-pass-per-submit: 76 queue boundaries, zero batching `[HIGH]`
**Domains: flow-submits, compute-cull-raster, render-post, bindgroups.**

Verified: `_doc0.json` shows `createCommandEncoder=76 = finish=76 = submit=76`, with
**zero multi-buffer submits**. Breakdown: 2 autoExposure + 1 clearHier + 1 seedRoots + 18 args
+ 18 traverse + 2 rasterArgs + 1 visClear + 1 SW raster + 1 hwArgs + 11 HZB (= 54 nanite compute)
+ 1 HW render + ~13 post/scene + 2 TRAA copies.

Root cause (intent side): three.js `WebGPUBackend.finishCompute` (three.webgpu.js:81359-81367)
and `finishRender` (59496) call `device.queue.submit([encoder.finish()])` on **every**
`renderer.compute()`/`renderer.render()`. Our `dispatch()` helper (`Tsl.ts:224 = renderer.compute(kernel)`)
issues one pass per call. The divergence is that the **supported array form**
`renderer.compute([L0..L10])` — which records many dispatches into ONE encoder (three.webgpu.js:60504)
— is never used. `NaniteHzb.build` (`NaniteHzb.ts:150-152`) already loops kernels; that loop is
the textbook batch candidate.

Why it's invisible to the harness: `GpuProfiler` times only `beginningOfPass→endOfPass`
timestamp spans (per-pass `timestampWrites`, cmd#573); `cpu.submitMs` (`Engine.ts:164`) is CPU
encode wall only. Neither sees the **submit bubble** between passes — the GPU cannot begin submit
N+1 until N's encoder is finished+committed, so dependent chains serialize. On Apple Metal-3 at a
conservative 30–80 µs/boundary, 76 boundaries ≈ **2.3–6 ms/frame** of pure scheduling overhead
nobody has measured.

Concrete batchable clusters (no semantics change, all non-indirect fixed-count dispatches
recorded in order into one encoder):
- **HZB 11 → 1** (submits 47-57; the single biggest, cleanest win).
- `kClearHier + kSeedRoots`; the non-indirect arg/clear kernels where no indirect dispatch sits
  between them. Indirect passes (`kTraverse*`, `kRasterWorld1`) cannot share an array call
  (`compute()` takes one dispatchSize) and must stay separate.
- Target 76 → ~25–35 submits. **Verify with a whole-frame wall delta, NOT per-pass timestamps.**

Framework-bound subset: the ~13 post-tail passes (bloom mip chain, TRAA, aerial RTT, grade) are
three.js PostProcessing-node-scheduled and **cannot** be batched without patching three's
scheduler — flagged for completeness in the 76 total, not as an app lever.

### THEME B — Full-world machinery resident in a terrain-less forest `[HIGH]`
**Domains: buffers, textures, pipelines-shaders, renderbundles.**

The forest testbed builds the entire world-genesis stack purely to satisfy bind-group validity
(`ForestScene.ts:189-207`: `Heightfield.generate` at `:197`, full `PostStack` at `:207`), then
renders zero terrain clusters. The cost is one-time at boot, but steady-state **residency**
distorts every memory number and is the bulk of the object explosion:

| Resident-but-dead block | Size | Capture evidence | Intent |
|---|---|---|---|
| Terrain synthesis/erosion scratch buffers (40, `boundInPasses:[]`) | **~755 MB** (34% of buffers) | 4×67 MB (4096² f32), 2×33.5 MB + 24×16.8 MB (2048²) — ids 4041/4042/4081/4417… all unbound, label:null | `Erosion.ts:79-88` + `HeightSynthesis.ts:28-29` alloc; **no `.destroy()` anywhere** |
| Terrain heightfield/derived **textures** (bound to 3 hot pipelines, never sampled) | **~319 MB** (37% of tex) | id4425 normalTex 134 MB (largest single tex), 4424 67 MB, 4445 67 MB, 4435 33.5 MB, 4050/4051 8.4 MB — all bound in `nanRasterWorld1`, `nanHwPass`, resolve | `hfWorld()` (`NaniteFetch.ts:300-353`) + `buildTerrainShading` (`NaniteResolve.ts:295`) gated on `matClass==0` — dead control flow (no terrain clusters) |
| Octahedral impostor atlases (12× 2048² rgba8 mip12) | **201 MB** (23% of tex) | ids 160/161/170/171/179/180/188/189/197/198/206/207, ALL `usedInPasses:[]` | `VegLibrary.ts:309-338` bakes per species; nanite resolve never instantiates the impostor material |
| Foliage card atlases (5× 1024² rgba8 mip11) | **21 MB** | ids 927/1478/2006/2534/3062, `usedInPasses:[]` | `VegLibrary.ts:170-176`; nanite renders mesh leaf crowns, not cards |
| ~65 terrain/erosion/atmo **compute pipelines** | resident, **0 dispatches** | of 98 ComputePipelines only ~25 dispatched; erosion/fill/flow/atmo (ids 4047-4481) in no submit | `Heightfield.generate` + `SunSky.init` compile them at boot |
| 24 VegLibrary **render pipelines** (`MeshStandardNodeMaterial_18..59`) | resident, **0 draws** | ids 25,41,…3660 — none in the draw list | trees drawn via resolve, not these materials |
| ~182 of 248 mip-bundles on unsampled textures | ~212 MB mip VRAM | rgba8 2048²/1024² with `usedInPasses:[]` → 132+50 bundles | three.js `_mipmapCreateBundles` over boot-baked maps |

Total dead/unsampled: **~755 MB buffers + ~520 MB textures (61% of texture VRAM)**. The single
GC-without-destroy validation error confirms boot scratch is GC'd, not freed. The lean
`?nanitedbg` path (`ForestScene.ts:222`) already proves a 1×1 stub heightTex suffices — the
fullFrame path could mirror it or build at preset `low`.

> Note (not waste): the **1.17 GB of tree geometry** (id4617 verts 894.7 MB = 37.3M verts,
> id4616 indices 278.4 MB = 23.2M tris — the full DAG of 200k×2 bark/leaf instances) IS the
> legitimate, intended testbed content. It is the headline *correct* allocation and embodies the
> unresolved "no far-field cluster floor" the docs call out (N9 impostor/merge target). Likewise
> the 200 MB bark texture arrays (id215/216) ARE sampled by the resolve. These are in Theme B's
> neighborhood but are NOT dead — do not lump them with the leak.

### THEME C — `meter()` double-dispatch (autoExposure ×2, + latent ×2 readbacks) `[MEDIUM]`
**Domains: flow-submits, compute-cull-raster.**

`compute_autoExposure` runs twice at frame head (submit 1 cmd#1 + submit 2 cmd#9, **same pipeline
4530, same bindgroup**, `[1,1,1]`). Cause: `NaniteFrame.meter` (`NaniteFrame.ts:426`) is invoked
from BOTH `ForestScene.ts:217 engine.onUpdate(() => frame.meter(...))` AND `Engine.renderStep:157
this.post.meter(...)` (same handles object). Frame 1244 isn't `%15` so only the double exposure
shows, but on readback frames it double-issues the cull/raster/shadow `getArrayBufferAsync`
readbacks (`NaniteFrame.meter:435-468`) and double-writes HUD counters; exposure feedback (mix
0.07) is applied 2×/frame. The `NaniteView` debug path (`ForestScene.ts:227`) has the identical
shape. One-site fix.

### THEME D — BFS runs full 18 iterations though the cut converges ~depth 9 `[MEDIUM]`
**Domains: flow-submits, compute-cull-raster.**

Capture: submits 5-40 = 9× `(nanArgsAB→nanTraverseAB)` + 9× `(nanArgsBA→nanTraverseBA)`, all
`kTraverse` INDIRECT off resource 4551. `HIER_MAX_DEPTH` defaults 18 (`NaniteCull.ts:209-212`)
and `runPhase1` (`:562-570`) hard-loops p=0..17 regardless of frontier emptiness. LOG bw
(`NANITE-LOG.md:86-88`) explicitly: cut converges at depth ~9 in this bm7-class view, "~9 empty
tail passes," default kept at 18 for the leaf-DAG safe bound. So tail passes ~10-18 walk an empty
frontier — up to **~18 of the 56 compute submits** (9 traverse + 9 args) are no-ops, and this
loop is shared by the shadow culls (×N). Fix options: GPU-side early-out that zeroes the
remaining `traverseDispatch` once frontier hits 0, or lower the camera-path `?hierdepth` to
measured convergence (camera converges earlier than the shadow leaf-DAG bound that forced 18).
Pin the exact depth with a one-frame frontier-count probe before tuning.

### THEME E — Deleted-2-pass-path remnants still doing per-frame work `[LOW-MEDIUM]`
**Domains: buffers, compute-cull-raster, bindgroups, pipelines-shaders.**

D-N45/PERF-VB4 (SPEC:1704-1742, LOG bx) deleted the `depth1→hwDepth→payload` 2-pass world path.
The capture confirms the *passes* are gone, but several **outputs/buffers** survive as live work:

- **`depthV` (id4601, 6.39 MB)** — `boundInPasses:['pass@378']` ONLY (= `nanVisClear`). Its
  siblings payloadV (4602) + visBV (4603) are bound across raster/HW/resolve; depthV is bound
  nowhere else. `makeVisBuffers` (`NaniteRaster.ts:154-169`) still allocates it; `kVisClear`
  (`:283-296`) still atomicStores `0xffffffff` over all 1.6M pixels every frame. The resolve
  reconstructs depth from the election key, not depthV. **6.39 MB dead VRAM + ~1/3 of the
  6237-workgroup clear's writes are dead.** The lone remaining reader is the `?audit` debug
  kernel.
- **`kRasterArgs` (submit 41, cmd#362)** — writes `rasterDispatchAttr`, consumed only by
  `depth1` (`NaniteRaster.ts:1031`, the shadow/deleted path). `world1` (`:1056`) reads only
  `rasterDispatchFull` (written by `kRasterArgs2`). Dead `[1,1,1]` dispatch + one of the 76
  submits in the single-pass world path.
- **`rasterDispatch2`** — `kRasterArgs2` (`NaniteCull.ts:321-328`) still computes the appended-
  range args; nothing dispatches against them (`rasterDispatch2Attr` declared `NaniteRaster.ts:178`,
  never a `dispatchIndirect` source). Dead ALU + dead interface plumbing.

### THEME F — Wasted full-res attachments on side-effect / composite passes `[LOW]`
**Domains: render-post, textures.**

- **HW needle pass** (submit 46, `nanHwPass`): full-res 1101×1450 **rgba8unorm color target
  (6.4 MB)** with `loadOp=clear, storeOp=store` despite `mat.colorWrite=false` (`NaniteRaster.ts:880`,
  frag returns `vec4(0)`). The raster writes vis buffers via fragment storage side-effects; the
  color attachment is a WebGPU structural requirement. **The full-res SIZE is load-bearing** (the
  needle fragments must generate at every covered pixel), but `storeOp` could be `discard` — the
  per-frame clear+store of unwritten pixels is pure waste.
- **Canvas grade pass (submit 58) + RTT/TRAA (60/59)**: full-res `depth24plus` attachments
  (id4711, id4720, ~12.8 MB) auto-attached by three.js to fullscreen-triangle composites with
  `depthTest:false`. Largely three.js RenderPipeline default (mostly framework-normal); reclaimable
  only by constructing the RTs with `depthBuffer:false` like `HalfResMrt.ts:62` already does.

### THEME G — Instance double-count: bark+leaf as separate instances `[LOW]`
**Domain: buffers.**

id4538 instances = 12.8 MB = 400,000×32B and id4539 instanceMesh = 1.6 MB = 400,000×4B — **2×**
the ~200k planted trees, because `ForestScene.ts:126-134` binds the identical transform stream to
BOTH `m.bark` and `m.leaf` as distinct instance ranges (`GeometryRegistry.bindInstances:954-977`).
This is the intended bark+leaf split, but it doubles instance memory (~14.4 vs ~7.2 MB) AND the
seed-root dispatch (`nanSeedRoots` dims `[6250,1,1]` = 400,064 threads). A combined bark+leaf
mesh would halve both. Structural cost the "tree perf" work should know.

---

## 3. Full ranked divergence table

Rank = impact × confidence. "SB" = spec-bug (§4). "FW" = framework-bound (lever limited).

| # | Divergence | Sev | Conf | Theme | Key capture evidence | Key intent |
|---|---|---|---|---|---|---|
| 1 | 76 submits/frame, zero batching; ~2.3-6 ms invisible to profiler | high | high | A | `_doc0`: 76 enc=76 finish=76 submit, multi-buffer=0 | three.webgpu.js:81365/59496; `Tsl.ts:224`; `GpuProfiler` per-pass only |
| 2 | ~755 MB terrain scratch buffers resident, never destroyed | high | high | B | 40 bufs `boundInPasses:[]`, 4×67 MB etc. | `Erosion.ts:79-88`, `HeightSynthesis.ts:28-29`; no `.destroy()` |
| 3 | ~319 MB terrain textures bound to 3 hot pipelines, never sampled | high | high | B | id4425 134 MB normalTex etc., bound in cmd#386 + NM_67 + NM_69 | `NaniteFetch.ts:300-353` / `NaniteResolve.ts:295` gated `matClass==0` |
| 4 | 201 MB impostor atlases resident, `usedInPasses:[]` | high | high | B | 12× 2048² rgba8 mip12, ids 160…207 | `VegLibrary.ts:309-338`; nanite never builds impostor mat |
| 5 | HZB built as 11 separate submits (batchable → 1) | med | high | A | submits 47-57 `nanHzbL0..L10`, each own encoder | `NaniteHzb.ts:150-152` per-level loop; array `compute()` unused |
| 6 | `compute_autoExposure` dispatched 2×/frame (double `meter()`) | med | high | C | submits 1+2, pipe 4530, same bindgroup `[1,1,1]` | `ForestScene.ts:217` + `Engine.renderStep:157` both call meter |
| 7 | BFS runs all 18 traverse passes; converges ~depth 9 | med | high | D | 9 AB + 9 BA traverse, all INDIRECT 4551 | `NaniteCull.ts:209-212/562-570`; LOG bw:86-88 |
| 8 | `depthV` (6.39 MB) cleared every frame, read by nothing | med | high | E | id4601 `boundInPasses:['pass@378']` only | `makeVisBuffers` 154-169, `kVisClear` 283-296; LOG bx |
| 9 | 1.17 GB tree geometry — no far-field cluster floor (N9) | med | high | B(note) | id4617 894.7 MB, id4616 278.4 MB | `GeometryRegistry.build:998-1001`; NaniteCommon.ts:33-39 |
| 10 | ~65 terrain/atmo compute pipelines + 24 veg render pipelines resident, 0 use | med | high | B | pipelines.json: 25/98 dispatched, 0/24 MeshStandard drawn | `Heightfield.generate`+`SunSky.init`; VegLibrary |
| 11 | 21 MB foliage card atlases resident, `usedInPasses:[]` | med | high | B | 5× 1024² rgba8, ids 927…3062 | `VegLibrary.ts:170-176` |
| 12 | HW needle pass stores 6.4 MB color target despite colorWrite=false | low | med | F | submit 46 nanHwPass rgba8 clear/store | `NaniteRaster.ts:880` colorWrite=false; storeOp could discard |
| 13 | `kRasterArgs` (submit 41) dead in single-pass world path | low | high | E | submit 41 cmd#362; world1 reads rasterDispatchFull | `NaniteCull.ts:571` vs `NaniteRaster.ts:1056` |
| 14 | `rasterDispatch2` computed by kRasterArgs2, consumed by nothing | low | high | E | submit 42; only nanRasterWorld1 indirect 4594 dispatches | `NaniteCull.ts:321-328`; `NaniteRaster.ts:178` declared unused |
| 15 | Instance buffers 2× tree count (bark+leaf split) | low | high | G | id4538 12.8 MB = 400k×32B | `ForestScene.ts:126-134`, `GeometryRegistry:954-977` |
| 16 | ~182 of 248 mip-bundles + ~212 MB on unsampled textures | low | med | A/B | rgba8 2048²/1024² `usedInPasses:[]` | three.js `_mipmapCreateBundles` over boot bakes |
| 17 | 200 MB bark arrays at fixed 2048² regardless of trunk coverage | low | med | B(note) | id215/216, bound NM_69 (LIVE) | `BarkSynth.ts:29` BARK_RES hardcoded 2048 |
| 18 | Canvas/RTT/TRAA full-res depth24plus attachments unused | low | med | F | id4711/4720 `usedInPasses:[]` | three.js RenderPipeline default depthBuffer |
| 19 | GPUBuffer GC'd without destroy (leak signal) | low | med | B,E | 1 validation error in statistics.json | boot scratch relies on GC |
| 20 | ~13 post-tail submits never accounted in PERF-4 (FW-bound) | low | med | A | submits 62-74 bloom/TRAA/grade | three PostProcessing nodes; PERF-4 closed |
| 21 | Forest excludes shadow path entirely (csm=null) — coverage gap | med | high | — | zero r32float shadow tex in capture | `NaniteFrame.ts:222` shadowOn=false; ForestScene csm:null |
| 22 | Three compute/HW paths pinned at 10/10 storage-buffer ceiling | med | high | — | layouts 4618, 4636, 4563 = 10 storage each | SPEC:81/600 maxStorageBuffers=10; D-N23 |
| 23 | ~all nanite buffers `label:null` — auditability gap | low | high | — | bg4621 10/11 entries null, etc. | StorageBufferAttribute sets no `.name` |

Item 21 (shadow path absent) and 22 (binding ceiling) are not *divergences* per se — they are
**coverage/architecture facts the perf investigation must not misread**: this frame does NOT
exercise the clipmap/cascade r32float shadow VRAM (so it is not the representative "full pipe" for
shadow memory), and the real 10/10 binding pressure is on `nanRasterWorld1`/HW/`nanTraverse` (NOT
the resolve, which sits comfortably at 8/10 — D-N23's fix is holding, refuting the seeded worry).

---

## 4. Spec-bugs (wrongly specced in BOTH docs and code)

These are the items the user specifically asked for — drift that is wrong in the *intent* itself,
not just stale capture vs code.

1. **Double `meter()` wiring (Theme C).** Both `ForestScene.ts:217` (`onUpdate(frame.meter)`)
   AND `Engine.renderStep:157` (`this.post.meter`) are wired to drive the same auto-exposure +
   readback path. Neither was meant to fire alongside the other — the code architecture itself
   double-schedules it. Confirmed in capture (submits 1+2 identical). `spec_bug: true`.

2. **Dead overflow accounting from the deleted brute-cull path.** `QCHUNK_CAP / CHUNK_CLUSTERS
   / REJ_INST_CAP / REJ_CLUST_CAP` (NaniteCommon) and the `readCounts` overflow checks
   (`NaniteCull.ts:602-608` `over('qChunks'…)`, `over('rejInst'…)`, `over('rejClust'…)`) survive
   the brute-path deletion. The hier BFS *repurposed* those counter slots as frontier-A/B +
   snapshot counts (`:352-359`), so the HUD overflow flag now compares unrelated frontier counts
   against the dead chunk/reject caps — it can mis-fire or mask the REAL ceiling (qRaster /
   frontier vs QRASTER_CAP). No buffer exists (capture confirms no chunk/reject buffers), but the
   constants + checks are stranded in both code and the cull header doc. `spec_bug: true`.

3. **Stale 2-pass comments contradicting the shipped single-pass code.** `NaniteFrame.ts:190-191`
   ("NON-packed two-pass raster (depthV ⇒ HZB…)") and `:396-401` ("ONE depth+payload over the
   NON-packed two-pass raster (depthV written ⇒ HZB…)"), plus `NaniteRaster.ts:194` ("legacy
   depthV/payloadV two-pass + brute encoding"), describe the path D-N45/PERF-VB4 **deleted**.
   The *same file* says single-pass correctly at `:181/403-404`. The capture proves single-pass
   (no depthV writer, no payload pass). This is the drift that **masked the dead-`depthV` finding**
   — a reader following `:191` would re-wire an HZB read against a non-existent exact depthV.
   `spec_bug: true` (it is a documented-intent error, even if no runtime cost).

> Borderline (flagged `spec_bug:false` by the domain agents but worth noting as latent intent
> drift): `kRasterArgs` / `rasterDispatch2` (Theme E) are dead in the single-pass world path but
> the kernels are *also* legitimately used by the shadow/depth1 path, so they are remnants rather
> than mis-spec. The fix is to gate them on the consuming config, not delete outright.

---

## 5. Ruled out / framework-normal (the list is NOT padded)

These looked alarming but are how three.js WebGPURenderer works, or were affirmatively refuted:

- **"248 RenderBundles for 18 passes = churn"** — REFUTED. All 248 are three.js *mipmap-generation*
  bundles (`WebGPUTexturePassUtils._mipmapCreateBundles`), identifiable by the exact degenerate
  descriptor `{colorFormats:["rgba8unorm"]}` with no label/depthStencilFormat/sampleCount. **Zero
  `executeBundles` / zero `createRenderBundleEncoder` in the entire command stream**; the app's
  `src/` never touches bundles. Created once at boot, replayed only on mip regen (not this frame).
  No per-frame, no per-instance churn.
- **"Multiple full-res passes render the trees"** — REFUTED. Trees are compute-rastered
  (`nanRasterWorld1` + `nanHwPass`); the only indexed geometry is the 1984-tri SunSky background
  (`totalTriangles=2001`). The resolve is a single `draw[3]` fullscreen triangle.
- **Per-pass submit on each `renderer.compute()`/`render()`** — three.js r184 behavior, not a
  per-call bug. (The *divergence* is the missing array-batching, Theme A — not that three submits
  per call.)
- **56 separate compute submits / 11 separate HZB pipelines / 18 interleaved `nanArgs` dispatches**
  — the per-iteration arg dispatch and per-level HZB pipeline are *correct design*; only the
  post-convergence BFS tail (Theme D) and the un-batched *encoder* structure (Theme A) are levers.
- **Two raster-arg kernels (`nanRasterArgs` + `nanRasterArgs2`)** — NOT evidence the 2-pass world
  raster survives; both are `[1,1,1]` arg-prep, and the SW raster is genuinely a single pass.
  Only the *dead outputs* (Theme E) are removable.
- **Final grade→canvas submit (58) logged BEFORE its producers (59-74)** — three.js RenderPipeline
  depth-first `updateBefore` lazy nested-pass evaluation. Submit index ≠ execution order in the
  post tail.
- **Aerial RTT / TRAA / TRAA.history each get their own full-res rgba16float RT (~38 MB)** — three.js
  materializes an RT at every node a downstream node texture-samples. Node-graph pass splitting,
  not redundant work. Per PERF-4 these are drain/ALU-bound near-mirages (per-pass timestamps
  overcount ~7× by absorbing the preceding pass's encoder span); real post cost is the half-res AO.
- **12 bloom passes (highpass + separable down/up 551×725→35×46 + comp)** — stock three.js
  BloomNode mip chain; sub-ms true pixel cost.
- **`label:null` on nearly every nanite buffer; `COPY_SRC|COPY_DST|VERTEX|STORAGE` mask** — three.js
  StorageBufferAttribute defaults (no `.name`, always adds copy+vertex usage). (Auditability gap,
  item 23 — cosmetic, not a defect.)
- **41 PipelineLayouts → 13 distinct BGL combos; 157 ShaderModules / 153 unique** — three.js mints
  one layout/module per pipeline-permutation and caches resident; not dedup'd, not a leak.
- **HZB is a single f32 storage buffer (id4560, 2.13 MB), not a texture** — intended NaniteHzb
  layout; its absence from the texture list is correct.
- **Two ~67 MB BFS frontiers + 67 MB qRaster = 201 MB** — documented camera-path cost
  (`NaniteCommon.ts:33-40`, fcap=QRASTER_CAP=8M), sized to stay under WebGPU's 128 MB/binding cap.
- **TSL per-species pipeline explosion (leaf tint 0.24 vs 0.22, LOD-fade 13.9 vs 12.3 baked as
  WGSL literals)** — three.js inlines material scalars as constants → one pipeline per variant.
  A real three.js trap, but only matters on the *draw-hot world path*, not this frame (0 drawn).

---

## 6. What this means for the perf investigation

The harness's verdict — "frame is ~90% per-pixel GPU-bound, nothing left to win" — is **true about
active GPU pixel time and honest within what the per-pass profiler can see.** But the capture's
*structure* exposes wins the per-pass `GpuProfiler` and `cpu.submitMs` are **architecturally blind
to**:

1. **There is a submit-overhead floor the profiler cannot see (Theme A).** 76 one-pass-per-submit
   boundaries with zero batching means dependent chains serialize with a queue bubble between each.
   The per-pass timestamps (begin→end span) and CPU encode wall both *exclude* this. Before
   concluding "nothing left," **measure a whole-frame wall-clock delta** after collapsing the
   HZB 11→1 (the cleanest, zero-risk batch) and the arg/clear clusters. If the estimated 2.3–6 ms
   is even partly real, that is a larger lever than any remaining pixel micro-opt — and it is
   invisible in every number the harness produced. **This is the single most important follow-up.**

2. **The "what does nanite cost" memory accounting is contaminated (Theme B).** ~1.5 GB of the
   capture's footprint is full-world boot machinery the forest never uses (755 MB terrain scratch
   never freed + 520 MB bound/resident-but-dead textures). Any reasoning about nanite VRAM,
   OOM headroom, or "the frame holds 2.2 GB of buffers" is polluted by it. Dropping it in the
   testbed (1×1 stub maps like `?nanitedbg` already uses, or preset `low`, + `.destroy()` of bake
   scratch) doesn't speed the frame but **clears the GC validation error and makes the next
   capture a clean nanite-only measurement** — which the perf work needs to isolate the real cost.

3. **Cheap structural no-ops to remove (Themes C, D, E):** kill the double `meter()` (one wiring
   line → drops 1 submit + halves exposure adaptation + removes latent double-readbacks every 15
   frames); early-out or lower the BFS depth (up to ~18 of 56 compute submits are post-convergence
   no-ops — pin the depth with a one-frame frontier probe first); stop allocating + clearing
   `depthV` in single-pass mode (6.39 MB + ~1/3 of the visClear writes); drop `kRasterArgs` +
   `rasterDispatch2` from the world path. None move the pixel needle, but each removes a submit
   boundary and/or a per-frame dispatch — and they compound with Theme A's batching.

4. **Set up the *right* measurement next time.** This frame has **no shadow path** (csm=null →
   zero clipmap/cascade r32float allocated) and mixes sky+post into the workload, so it is neither
   a clean nanite-only measurement nor a representative full pipe for shadow VRAM. To profile the
   submit-batching win and the shadow path, use a world scene with csm; to isolate nanite raster,
   use the lean `?nanitedbg` view (no sky/post/terrain residency).

5. **The durable structural cost is real and known (item 9):** 1.17 GB of full-DAG tree geometry
   with no far-field cluster floor. That is the N9 impostor/merge target the docs already call out
   — it attacks both the VRAM and the per-pixel cluster flood the harness *did* measure. It is the
   one big number that is *intended* and *not* a quick win.

**Net:** the per-pixel wall is real, but "nothing left" was the per-pass profiler's blind spot
talking. The structure points to a submit-overhead floor (measurable only at whole-frame
granularity) and ~1.5 GB of dead residency — both invisible to the harness, both cheaper to attack
than the pixel shader, and the first (HZB batch + whole-frame timing) is a clean, falsifiable test.
