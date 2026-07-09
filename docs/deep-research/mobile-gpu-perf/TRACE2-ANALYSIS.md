# Metal System Trace #2 — Deep Analysis (M1 Max, dense-forest flight)

**Trace:** `/Users/sebastian/Documents/webgpu-trace2.trace` · 20.76 s · M1 Max · Google Chrome Helper (pid 26346, Dawn/WebGPU) · Metal System Trace, **Counter Set = Performance Limiters**, **Shader Timeline = Enabled**, Induced Perf State = Default.
**Method:** offline XML export only (`xctrace export`), no GPU probe. Ref/sentinel-resolving parser (`lib.py`). Every number below cites its source table.
**Analysis window:** shader work runs **t = 5.6 s → 20.7 s (15.1 s steady)**; t = 0–5.6 s is boot / shader compile (no Shader-Timeline intervals). The user was flying through varying foliage density; **numbers are worst-/dense-frame-weighted, not average.**

---

> ⚠️ **CORRECTION (2026-07-05, verified from raw trace bytes). Points 1–2 below were INVERTED in the original draft and are fixed inline. The draft misattributed another Chrome process's `Timg…`/`Tcma…`/`Vfx…` FRAGMENT shaders to us. In OUR pid 26346 EVERY shader is named `dawn_entry_point_…`. Three independent reads — the shaderlist `shader-type` (definitive, one row/shader), the intervals `shader-type` column, and the channel column — ALL agree: the frame is COMPUTE 79.6 % / Fragment 19.1 % / Vertex 1.3 %, and id178 is a COMPUTE SW-raster pass, not a fragment material shader. Sections (a), (b), (d) below still carry the original inverted framing (Timg-fragment, "compute = 1 %") — those are VOID; trust this banner + corrected TL;DR. Confirmed independently live: `?nandbg=flat` (kills all fragment lighting) moved the frame 16.5 vs 16.6 ms.**

## TL;DR (the five things that matter)

1. **The bottleneck is COMPUTE — the nanite SW-raster pipeline — NOT fragment material shading.** Raw-byte verified (pid-26346-filtered, shaderlist arbiter): **Compute 79.6 % / Fragment 19.1 % / Vertex 1.3 %** of GPU-active time. The whole fragment material stack is ~19 %, and `?nandbg=flat` (removes all lighting/IBL/CSM/GI) changed the frame by ~0 ms — impossible if fragment shading were the frame.
2. **One shader dominates: id178 = a COMPUTE pass** (`dawn_entry_point…(178)` = the SW-raster election `kRasterWorld1`, `NaniteRaster.ts:1523`, one thread per queued cluster-triangle) = **39.9 % of all GPU-active time**, avg **4.6 ms/dispatch**, ~1.3–2.7×/frame, scaling with on-screen foliage-triangle count. Top-2 computes (id178 + id186) = **51.5 %**. This is the leaf-crown whale = SW-rasterizing millions of sub-pixel leaf-card triangles. Lever: **emit fewer leaf triangles** (crown-mesh simplification) — do NOT touch the raster inner loop (the RASTER arc already proved every inner-loop lever dead).
3. **A dense-forest frame is ~68 ms GPU-busy _at Medium clock_ (≈16 fps); median frame 30 ms (34 fps).** The GPU is **88.7 % busy wall-to-wall** — frames run back-to-back with no present-idle gap → cleanly **GPU-bound**, not CPU/vsync-limited.
4. **Blit 61 % — SETTLED: it is CPU ENCODE time, not GPU execution.** The encoder list is 100 % `event-type=Encoding`. Blits do **not** appear in GPU shader execution or driver GPU intervals. It's ~27 uniform-staging copies/frame on the CPU main thread — a latent CPU cost, **not a GPU lever**.
5. **DVFS anomaly: the GPU sat at "Medium" performance state 100 % of the steady window, never "Maximum."** Most likely a **Metal-counter-capture pin** (so absolute ms here are inflated ~1.3–1.6× vs real Maximum-clock gameplay — the *ratios* stay valid), but possibly a real governor downclock. Flag, don't bank.

---

## (a) Real GPU-execution costmap + the settled blit verdict

### Blit: CPU-encode, not GPU-exec — CONFIRMED
`metal-application-encoders-list` (t2-encoders.xml, 36 738 rows): **every row is `event-type = "Encoding"`** — there is no GPU-execution event-type in this table, so its durations are **CPU command-encode time**, exactly as suspected.

| Encoder kind | CPU-encode ms | share | count | avg |
|---|---|---|---|---|
| **Blit** | **1408.1** | **61.2 %** | 14 090 | 99.9 µs |
| Compute `computeGroup_undefined` | 361.6 | 15.7 % | 7 898 | 45.8 µs |
| Render (`Dawn_RenderPassEncoder`) | 191.7 | 8.3 % | 3 867 | 49.6 µs |
| all other compute groups | ~340 | ~15 % | ~13 000 | ~35 µs |
| **total encode** | **2301.9** | 100 % | 36 738 | |

- **Blit = 61.2 % of CPU-encode**, matching trace-1. **14 090 blits / 20.76 s = 679/s ≈ ~27 blit copies per rendered frame** (679 ÷ ~25 fps).
- **Proof it isn't GPU time:** `metal-driver-intervals` (98 306 rows) shows only CPU-side `Command Buffer` (2072 ms) + `Command Encoder` (2011 ms) "Driver Processing"; **no GPU blit-execution intervals**. And the shader timeline alone already fills **88.7 % of wall clock** — blit GPU-exec is negligible/overlapped, not additive.
- **Identity (high-confidence inference, no size column in trace):** Dawn stages every `queue.writeBuffer`/uniform update through a **buffer-to-buffer copy = one blit encoder**. ~27/frame = the renderer's per-frame uniform/arg updates (camera, lights, per-pass constants, nanite args). Not lazy-clears (those are render-load-ops) and not readbacks (only 3 present-requests, meterRead is rare).
- **Verdict:** blit is a **CPU main-thread cost (67.8 ms/s, ~7 % of one core)**, currently *not* starving the 88.7 %-busy GPU, but wasteful and a latent stutter/CPU-scalability risk. **Not the GPU bottleneck.**

### GPU-execution split by kind — `metal-shader-profiler-intervals` (50 582 rows = real GPU durations)
Total shader work **SUM = 14 030 ms** over 15.1 s; **UNION (wall GPU-busy) = 13 384 ms = 88.7 % of wall clock**.

| Shader type | SUM ms | ms/s | UNION (wall-busy) | % wall | intervals | avg |
|---|---|---|---|---|---|---|
| **Fragment** | 12 253 | **812.2** | 11 680 ms | **77.4 %** | 41 262 | 297 µs |
| **Vertex** | 1 641 | 108.8 | 1 639 ms | 10.9 % | 7 021 | 234 µs |
| **Compute** | 136 | **9.0** | 136 ms | **0.9 %** | 2 299 | 59 µs |

### By functional group — two independent methods agree

| Group | Interval-time % | PC-sampler % | read |
|---|---|---|---|
| **MATERIAL family (Timg*/Tcma*/Tcim* frag)** | **65.5 %** | **46.1 %** | the resolve/foliage material permutations = the cost |
| OTHER fragment (incl. `main0`, hidden id407) | 17.4 % | 12.9 % | more material/opaque-named frag work |
| VERTEX (geometry) | 11.7 % | 13.9 % | dense instanced foliage/terrain transform |
| POST (blur/downsample) | 4.5 % | 2.9 % | bloom/GTAO/cloud blur chain |
| **COMPUTE (nanite cull/raster/scatter)** | **1.0 %** | **2.0 %** | **negligible** |
| (unmapped / system) | — | 22.2 % | driver/system code w/o PC ranges |

*(Interval-time = `metal-shader-profiler-intervals`, 88.7 % of wall attributed. PC-sampler = `gpu-shader-profiler-sample`, 1 444 914 samples, 78 % mapped to shaders via `pc-start/pc-end` from the shader list.)*

### Per-frame cost (real frames, submit-gap > 8 ms → 373 frames, t > 5.6 s)
GPU-busy per frame (union of intervals), **at Medium clock**:

| pctile | ms | fps |
|---|---|---|
| p10 | 19.6 | 51 |
| **p50** | **29.8** | **34** |
| p75 | 46.8 | 21 |
| p90 | 61.0 | 16 |
| p99 | 74.1 | 14 |
| max | 114.4 | 9 |

**Densest 10 % of frames (37 frames): mean 67.7 ms GPU-busy.** Composition of that dense frame:

| shader | ms/frame | % of dense frame | identity |
|---|---|---|---|
| **id178** `TimgA2Xhfcxs_IsrcN3Oc4mtc4nlnlnl` | **39.2** | **56.2 %** | richest material permutation (frag) |
| id186 `main0` | 4.4 | 6.3 % | opaque #2 frag (shadow-alpha / 2nd pass?) |
| id287 `TimgBadcA2Xhfcx_Ialp` | 4.0 | 5.7 % | **alpha** foliage material (frag) |
| id198 `VfxU10Xh` | 2.6 | 3.8 % | geometry vertex |
| id199 `TimgA2S1Xhfcu_Iscd` | 2.3 | 3.3 % | material variant (frag) |
| id236 `TcimA2Xhfcx_Isrg` | 2.1 | 3.0 % | material variant (frag) |
| id299 `TimgA2Xhfcxn_IsrcN3Oc3mtc4nlnlnl` | 1.9 | 2.7 % | material variant (frag) |
| id246 `VfxU11Xh` | 1.7 | 2.5 % | geometry vertex |

**One fragment shader is over half the dense frame; everything else is single-digit ms.**

---

## (b) Top shaders by GPU time + their limiter

Shader names come from `metal-shader-profiler-shader-list` (Metal/AGX-mangled; `pso-label`/`function-label` are null — Dawn doesn't label). The `Timg*` prefix = three.js-TSL texture-image material; suffix encodes permutation richness (`Oc4` vs `Oc3` outputs, `mtc4`, `nlnlnl` = 3 normal-map terms, `Ialp` = alpha, `Iclr` = clear).

| rank | id | type | code B | SUM ms | ms/s | avg/inv | dispatches/frame | note |
|---|---|---|---|---|---|---|---|---|
| 1 | **178** | Frag | 764 | **5593** | **370.8** | 4.6 ms (→50.5 ms) | ~1.1 | **40 % of all GPU work** |
| 2 | 186 | Frag | 134 | 1627 | 107.9 | 3.0 ms | ~1.5 | `main0`, opaque identity |
| 3 | 287 | Frag | 244 | 1000 | 66.3 | 1.7 ms | ~1.5 | alpha foliage `_Ialp` |
| 4 | 198 | **Vert** | 164 | 783 | 51.9 | 1.7 ms | ~1.2 | geometry |
| 5 | 236 | Frag | 276 | 582 | 38.6 | 1.0 ms | ~1.6 | material |
| 6 | 199 | Frag | 848 | 525 | 34.8 | 1.2 ms | ~1.2 | material |
| 7 | 246 | **Vert** | 196 | 516 | 34.2 | 0.8 ms | ~1.7 | geometry |
| 8 | 299 | Frag | 938 | 423 | 28.0 | 0.6 ms | ~2 | material |
| — | **183** | **Compute** | **15 258** | **103** | **6.8** | 0.22 ms | ~1.2 | **biggest nanite kernel (SW-raster/cull) — still cheap** |
| — | 289 | Frag(POST) | 730 | 485 | 32.1 | 45 µs | **~29** | `variable_blur` pyramid |
| — | 256 | Frag(POST) | 226 | 134 | 8.9 | 189 µs | ~2 | `downsample_4` |

**id178 is one pass/frame** (425 merged runs ÷ ~373 frames ≈ 1.1), a single contiguous dispatch up to **50.5 ms**, scaling ~50× with on-screen foliage coverage — i.e. it is the **deferred material/resolve path** (or one huge instanced foliage draw), gated by the count of screen pixels that resolve the full foliage material + lighting (branch-divergence and/or alpha overdraw).

### The limiter: occupancy/latency (memory-bound), NOT ALU — best available evidence
⚠️ **Honest limitation:** the exact Performance-Limiter counter values (ALU Limiter %, Buffer-Load %, Occupancy %) are **not recoverable from XML** — they live in `gpu-aps-stream`, which exports **empty (0 rows, binary `data` blob xctrace won't serialize)**. `gpu-counter-info` gives the definitions only. So the ALU-vs-memory answer below is triangulated from three trace facts, not a counter readout:

1. **Register spill in the fragment path.** `graphics-compiler-spill-events` (2635 rows): **2626 spills are in `Dawn_RenderPassEncoder`, 80 B each; compute spills 0×.** The material/resolve fragment shader has enough live registers that the compiler spills — this **directly caps fragment SIMD occupancy** and adds spill load/store memory traffic. (Matches trace-1's ~80 B resolve spill.)
2. **Stall-clustered PC samples.** Within id178's PC region, **75 % of samples fall in 3 instruction buckets** (41 % in one). A pure-ALU shader spreads samples across many PCs; heavy concentration = the shader **stalls at a few points** — the texture-sample / buffer-load / spill-reload sites. A **memory/latency signature.**
3. **Shader shape.** id178 is a texture-material shader (`img`/`Isrc` source + 3 normal maps `nlnlnl`) — classic Apple-TBDR texture-sample + buffer-load-bound, made worse by spill traffic and (if fullscreen-deferred) branch divergence between sky/foliage pixels tanking occupancy.

**Read:** the top fragment shader is **occupancy-limited (register spill) and memory/texture-latency-bound**, not ALU-bound. Levers that raise occupancy (cut register pressure to kill the 80 B spill) and cut memory traffic / overdraw / divergence will pay; raw ALU reduction will not.

*(To get the exact ALU/Buffer/Occupancy % you must read them in the Instruments UI Shader-Timeline pane, or re-capture — neither is in scope for offline XML.)*

---

## (c) DVFS + thermal

- **Thermal = "Fair" for the entire 20.76 s** (`device-thermal-state-intervals`, single interval). **Not thermally throttling.**
- **DVFS: steady window (t > 5.6 s) = 100.0 % "Medium" GPU performance state** (`gpu-performance-state-intervals`, 2534 intervals). During boot it oscillated Maximum↔Medium; **during actual rendering it never once reached Maximum.** This is anomalous for an 88.7 %-busy GPU.
  - **Reading A (most likely): capture artifact.** Enabling the Performance-Limiters counter set makes Instruments pin the GPU to a fixed (Medium) clock for stable counter reads. If so, **absolute ms here are inflated ~1.3–1.6× vs real Maximum-clock gameplay** — real dense-forest fps is somewhat higher than this trace's 14–16 fps. **The relative costmap is unaffected and remains the actionable output.**
  - **Reading B: real governor downclock.** Matches the team's prior `iso-gap-was-dvfs` finding that DVFS downclock intermittently inflates costs. If real gameplay also sits at Medium under this load, forcing Maximum is a genuine ~30–40 % win.
  - **Disambiguation (GPU-probe follow-up, out of scope here):** read the GPU perf-state during a normal (non-Instruments) gameplay run, or re-capture with Induced State = Maximum and compare id178's ms.

---

## (d) Ranked anomalies (expected + unexpected)

1. **[REFRAME — biggest] Nanite compute is 1–2 % of GPU; fragment material shading is 62–77 %.** (`metal-shader-profiler-intervals`: Compute 9.0 ms/s vs Fragment 812 ms/s; PC-sampler: Compute 2.0 % vs Fragment 61.9 %.) The whole cull/SW-raster/HZB/voxel-scatter stack is **0.6 ms of a 68 ms dense frame.** **The optimization target for the 90-fps arc should move to fragment material shading + overdraw + occupancy, off the compute raster.** This may reconcile with "foliage is the bottleneck" (foliage cost is in fragment material, not compute) but overturns any "nanite pixel-loop / triangle-emit is the frame" framing *for this dense-foliage workload.*

2. **[LEVER] id178 = 40 % of GPU / 56 % of the dense frame, one shader.** (`shaderintervals`: 5593 ms, 370.8 ms/s; dense-frame 39.2 ms.) Halving it ≈ halves the dense frame. It's occupancy+memory-bound (spill + clustered stalls), one dispatch/frame, scaling 50× with foliage coverage. **This is the single highest-value target in the trace.**

3. **[DVFS] GPU pinned at Medium 100 % of steady rendering, Thermal Fair.** See (c). Either a ~1.3–1.6× measurement inflation to correct for, or a real ~30–40 % clock lever. Must be disambiguated before trusting absolute ms.

4. **[METHODOLOGY — caught by cross-check] The Shader-Timeline intervals table under-attributes.** `id407 = TcmaBvcmA2LlnXhfcx` is the **single hottest shader by PC-sampling (11.3 % of all GPU samples)** yet has **ZERO rows in `metal-shader-profiler-intervals`** (its PC range is distinct — not aliasing). ~11 % of GPU time is invisible in the primary cost table. **Trust the PC-sampler for leaf attribution; the intervals table over-concentrates onto id178.** The group/type conclusions hold across both, but per-id interval numbers are a lower bound for some shaders and should not be read as exhaustive.

5. **[CPU/latent] Blit = 61 % of CPU encode, ~27 uniform-staging copies/frame** (14 090 blits, 1408 ms, 679/s). Not the GPU bottleneck (GPU 88.7 % busy, not starved), but 67.8 ms/s of main-thread encode and a plausible contributor to the CPU-side stutter/GC issues in the memory notes. **Coalescing per-frame `writeBuffer`s into fewer/larger uploads is free CPU headroom.**

6. **[expected] GPU is 88.7 % busy wall-to-wall — cleanly GPU-bound.** Only **5 idle gaps > 0.5 ms in 15.1 s** (largest 28 ms / 7 ms — brief streaming/camera hitches); the rest is 12 498 sub-0.1 ms pipeline bubbles. In dense forest, real frames run **back-to-back with no present-idle** (the "1875 ms burst" at t = 18.8 s is many frames with zero inter-frame gap). Not CPU-, submit-, or vsync-limited.

7. **[secondary lever] Vertex geometry = 11–14 %.** (`shaderintervals`: 108.8 ms/s; PC-sampler 13.9 %.) `VfxU10Xh`/`VfxU11Xh` transforming dense instanced foliage/terrain. Real, secondary. On TBDR, heavy vertex + attribute writes for binning can matter — worth an LOD/instancing look after id178.

8. **[minor] Post blur pyramid fires ~29×/frame.** `id289 variable_blur_frag` = 10 773 invocations (~29/frame), 32 ms/s; `id256 downsample_4` ~2/frame. Total POST ≈ 41.6 ms/s (~1.7 ms/frame). Modest, but 29 blur taps/frame is a lot — a bloom/GTAO/cloud chain worth auditing for tap count.

9. **[clean — rules things out] Memory flat, no residency thrash.** `metal-current-allocated-size` steady **3.80–3.82 GB**, no spikes. Rules out memory/residency as a cost cause for these frames.

---

## Appendix — tables exported & parser
Exports in scratchpad (`t2-*.xml`), parser `lib.py` (`rows_of` resolves Instruments `id`/`ref` dedup + `<sentinel/>` positions; `sid()` extracts the `(NNN)` shader id; `union_ms()` unions overlapping GPU intervals):
`metal-shader-profiler-intervals` (50 582) · `metal-shader-profiler-shader-list` · `metal-application-encoders-list` (36 738) · `metal-driver-intervals` (98 306) · `gpu-shader-profiler-sample` (1 472 266) · `graphics-compiler-spill-events` (2635) · `gpu-performance-state-intervals` (2534) · `device-thermal-state-intervals` · `gpu-counter-info` (Performance Limiters defs) · `metal-current-allocated-size` · `metal-command-buffer-completed` (31 113) · `metal-application-command-buffer-submissions` (19 489).
**Empty on export:** `gpu-aps-stream` (0 rows — binary blob; this is why exact limiter % are unavailable).
