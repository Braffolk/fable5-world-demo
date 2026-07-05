# Grass/mobile audit — per-theme detailed findings

_Per-agent detailed findings, extracted from the workflow run for cross-session reference._


---

## (agent)

All 7 processed. Manifest:

```
[
  {url:"https://github.com/philipturner/metal-benchmarks/blob/main/README.md",
   savedPath:"/Users/sebastian/IdeaProjects/fable-demo2/docs/mobile-gpu-perf/sources/apple-arch/metal-benchmarks-README.md",
   ok:true, oneLineWhatItContains:"Full raw Philip Turner metal-benchmarks README (859 lines) — Apple GPU microarchitecture: register file, occupancy, SIMD width, fp16 rates, latencies, ALU throughput."},

  {url:"https://dougallj.github.io/applegpu/docs.html",
   savedPath:".../dougallj-applegpu-docs.md",
   ok:true, oneLineWhatItContains:"Apple G13 ISA docs — 128 GPRs/thread (r0-r127, 16-bit low/high halves), 32-wide SIMD-group, fp16 fmadd16, lower regs = higher occupancy."},

  {url:"https://rosenzweig.io/blog/asahi-gpu-part-5.html",
   savedPath:".../rosenzweig-asahi-gpu-part5.md",
   ok:true, oneLineWhatItContains:"Asahi AGX debugging post — TBDR parameter-buffer overflow / partial renders; NO register/fp16/occupancy detail (noted in file)."},

  {url:"https://www.michaelstinkerings.org/apple-m5-gpu-roofline-analysis/",
   savedPath:".../michaelstinkerings-m5-gpu-roofline.md",
   ok:true, oneLineWhatItContains:"M5 roofline — scalar ISA float4->4 scalar (4.75x from scalarizing), needs 8 independent chains (4-cyc FMA latency), fp16 2x via dedicated 16-bit ALUs, register file is occupancy limiter."},

  {url:"https://developer.arm.com/documentation/101863/latest/.../Resource-usage",
   savedPath:".../arm-mali-offline-compiler-resource-usage-STUB.md",
   ok:false, oneLineWhatItContains:"FETCH FAILED — JS-rendered SPA shell only (curl + WebFetch both empty); saved stub with expected content + how to obtain."},

  {url:"https://developer.arm.com/.../the-mali-gpu-an-abstract-machine-part-4---the-bifrost-shader-core",
   savedPath:".../mali-bifrost-shader-core-part4.md",
   ok:true, oneLineWhatItContains:"Bifrost shader core — 64 32-bit regs at max occupancy, quad-vectorization (128-bit/quad), 8x fp16/16x int8 per clock, IDVS."},

  {url:"https://chipsandcheese.com/p/the-snapdragon-x-elites-adreno-igpu",
   savedPath:".../chipsandcheese-snapdragon-x-elite-adreno.md",
   ok:true, oneLineWhatItContains:"Adreno X1 — 96KB/partition register file, 64/128-wide waves, 64-wide FP32 + double-rate FP16, GMEM 3MB, register file limits large-wave occupancy."}
]
```

6 of 7 fetched with full technical content. One failure: the Arm Mali Offline Compiler "Resource usage" page is a client-side-rendered SPA — both curl and WebFetch returned only the empty HTML shell (~13 KB, no register/spill/occupancy text). Saved as a STUB `.md` noting the failure, the expected content (work/uniform registers, stack spilling, 16-bit arithmetic %, occupancy gating), and how to get the real page (headless browser or the `malioc` tool). All other six are saved as clean markdown; the GitHub README was curled verbatim as raw.

The key "judge our shaders" angle is well-covered by the metal-benchmarks README, dougallj docs, and especially the M5 roofline piece (fp16 packing, scalar-decomposition of float4, register-file-as-occupancy-limiter) for the later shader-review step.

---

## (agent)

All 3 sources saved to `docs/mobile-gpu-perf/sources/grass-raymarch/`.

Manifest:

```json
[
  {
    "url": "https://archive.thedatadungeon.com/ghost_of_tsushima_2020/documents/gdc_2021/gdc_2021_procedural_grass_in_got.pdf",
    "savedPath": "/Users/sebastian/IdeaProjects/fable-demo2/docs/mobile-gpu-perf/sources/grass-raymarch/got-procedural-grass.pdf + ghost-of-tsushima-procedural-grass-gdc2021.md",
    "ok": true,
    "oneLineWhatItContains": "GoT GDC2021 deck: grass = GPU per-blade GENERATE (2 compute passes place+cull → indirect-draw Bezier tri-strips, 7/15-vert LODs, clumping, tiny 1D/2D material) — the opposite lane from our per-pixel march; cost scales with visible blades not pixels."
  },
  {
    "url": "https://arxiv.org/pdf/1609.05344",
    "savedPath": "/Users/sebastian/IdeaProjects/fable-demo2/docs/mobile-gpu-perf/sources/grass-raymarch/cloudscapes.pdf + cloudscapes-optimisations-realtime-volumetric-toft2016.md",
    "ok": true,
    "oneLineWhatItContains": "Cloudscapes: cut a per-pixel volume raymarch to 1/16 steps (297.7ms full-128 → 7.5ms half-8-jitter-TAA) via analytical transmittance integration (kills step-length dependence → big steps) + per-frame jittered start offset + TAA; downside = jitter cache-miss tax (+5ms)."
  },
  {
    "url": "https://babylonjs.medium.com/infinite-grassland-ray-traced-rare-semi-uniform-entity-manifolds-...",
    "savedPath": "/Users/sebastian/IdeaProjects/fable-demo2/docs/mobile-gpu-perf/sources/grass-raymarch/babylon-infinite-grassland-semi-uniform-manifolds.md",
    "ok": true,
    "oneLineWhatItContains": "Babylon screen-constant-cost grass: ray + ℤ² 'corridor' of grid cells → analytic ray-vs-paraboloid per blade, zero per-step dependent fetches, cost ∝ pixels only (no per-entity overhead); no concrete ms numbers, proof-of-concept."
  }
]
```

All 3 succeeded. Both PDFs kept alongside their extracted `.md`. Key cross-source finding for the arc: the two non-raymarch sources (GoT per-blade generate, Babylon analytic corridor) argue our +14.5ms per-pixel march is the wrong lane for mobile TBDR, while Cloudscapes gives a concrete 16× step-reduction recipe (analytic integration + jitter + TAA) if we stay in the raymarch lane — each `.md` ends with an explicit "map to our kGrassRay" section.

---

## (agent)

All 5 saved. Manifest:

```json
[
  {
    "url": "https://developer.apple.com/videos/play/tech-talks/10580/",
    "savedPath": "/Users/sebastian/IdeaProjects/fable-demo2/docs/mobile-gpu-perf/sources/apple-wwdc/techtalk-10580-metal-compute-on-macbook-pro.md",
    "ok": true,
    "oneLineWhatItContains": "THE 16%-occupancy-from-register-spilling case study + the recovery levers (16-bit types, shrink stack/threadgroup mem, constant address space, no dynamic stack indexing, maxThreadsPerThreadgroup); diagnosis via Xcode compiler statistics showing spilled bytes."
  },
  {
    "url": "https://developer.apple.com/videos/play/wwdc2020/10603/",
    "savedPath": "/Users/sebastian/IdeaProjects/fable-demo2/docs/mobile-gpu-perf/sources/apple-wwdc/wwdc2020-10603-optimize-metal-with-gpu-counters.md",
    "ok": true,
    "oneLineWhatItContains": "THE profiling path — 150+ GPU counters grouped into limiters (ALU/texture/tile-mem/buffer/LLC/occupancy/HSR), F16 double-rate, AVOID device atomics→threadgroup atomics, occupancy interpretation, Metal Debugger per-draw counters."
  },
  {
    "url": "https://developer.apple.com/videos/play/wwdc2020/10632/",
    "savedPath": "/Users/sebastian/IdeaProjects/fable-demo2/docs/mobile-gpu-perf/sources/apple-wwdc/wwdc2020-10632-optimize-metal-performance-for-apple-silicon-macs.md",
    "ok": true,
    "oneLineWhatItContains": "Richest shader-lever talk: 16-bit types, constant-address-space prefetch into uniform registers, half-literal 'h' suffix, spill avoidance (no runtime-indexed stack arrays, signed indices, vectorizable structs), memoryless targets, tile shaders."
  },
  {
    "url": "https://developer.apple.com/videos/play/wwdc2020/10602/",
    "savedPath": "/Users/sebastian/IdeaProjects/fable-demo2/docs/mobile-gpu-perf/sources/apple-wwdc/wwdc2020-10602-harness-apple-gpus-with-metal.md",
    "ok": true,
    "oneLineWhatItContains": "TBDR fundamentals: load/store actions (Clear>Load), HSR sorting, programmable blending in tile memory, memoryless render targets, imageblocks, tile shaders, GPU-driven rendering (argument/indirect command buffers)."
  },
  {
    "url": "https://metalkit.org/wwdc20-whats-new-in-metal/",
    "savedPath": "/Users/sebastian/IdeaProjects/fable-demo2/docs/mobile-gpu-perf/sources/apple-wwdc/metalkit-wwdc20-whats-new-in-metal.md",
    "ok": true,
    "oneLineWhatItContains": "Blog roundup of WWDC20 Metal changes (Apple Silicon TBDR migration, feature-set detection, position invariance, threadgroup/SIMD sync, etc.); live site was unreachable (curl exit 52) so fetched via Wayback Machine — noted in file header."
  }
]
```

Notes on the levers, flagged Metal-only vs WebGPU/WGSL-reachable:

- **Reachable through WGSL/Dawn:** 16-bit types (WGSL `f16` via `shader-f16` feature — same double-rate on Apple GPU), avoiding device-memory atomics (use `workgroup` atomics/reductions instead), reducing workgroup shared memory, avoiding runtime-indexed private arrays / register spills, minimizing divergence, packing/vectorizing buffer loads, mipmaps/compressed textures.
- **Metal-only (no WebGPU surface):** explicit `constant` address-space uniform-register prefetch, `memoryless` storage / tile memory, programmable blending, tile shaders, imageblocks, parallel render encoders, `[[early_fragment_test]]` control, argument/indirect command buffers. Dawn/WebGPU manages these internally; you cannot address them from WGSL.
- **What a Metal GPU capture of our Chrome/Dawn process could reveal:** Dawn on macOS emits Metal, so Xcode's Metal Debugger / Instruments Metal System Trace can attach to the Chrome GPU process and expose the same 150+ counters against our actual compute kernels — i.e. real **occupancy %, spilled bytes, and the top limiter** for our SW-raster and grass-raycast kernels. That is the direct route to confirm whether our per-pixel loop is ALU-limited (F32 vs F16), tile/threadgroup-memory-exhaustion-limited, or spilling registers at low occupancy exactly like the 10580 case study — measurement we currently only infer.

One failure handled without stopping the rest: metalkit.org was down (connection reset, curl exit 52, even insecure/http/www variants); recovered the full article from the Wayback Machine and flagged the provenance in the file header.

---

## (agent)

All six sources saved. Manifest:

```json
[
  {
    "url": "https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf",
    "savedPath": "docs/mobile-gpu-perf/sources/occupancy-ilp/volkov-gtc2010-better-performance-lower-occupancy.{pdf,md}",
    "ok": true,
    "oneLineWhatItContains": "Volkov's canonical talk: ILP hides latency at LOW occupancy — 84% mem peak at 4% occupancy, issue independent loads before consuming (slide 32), registers are the only fast-enough store; the core 'is it ILP- or occupancy-limited' reference."
  },
  {
    "url": "https://gpuopen.com/learn/optimizing-gpu-occupancy-resource-usage-large-thread-groups/",
    "savedPath": "docs/mobile-gpu-perf/sources/occupancy-ilp/aaltonen-gpuopen-optimizing-occupancy-large-thread-groups.md",
    "ok": true,
    "oneLineWhatItContains": "Aaltonen GCN VGPR-reduction playbook: scalar/raw-buffer loads (0 vs 16 VGPRs for a matrix), fp16 packing, [loop] to shrink live ranges, 32-VGPR budget for dual-group occupancy."
  },
  {
    "url": "https://gpuopen.com/learn/occupancy-explained/",
    "savedPath": "docs/mobile-gpu-perf/sources/occupancy-ilp/gpuopen-occupancy-explained.md",
    "ok": true,
    "oneLineWhatItContains": "RDNA occupancy primer: VGPR/SGPR/LDS limits, theoretical vs measured occupancy, 'peak occupancy != peak performance', spilling ~100x penalty, RGP/RGA/PIX diagnostics."
  },
  {
    "url": "https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-register-pressure-readme/",
    "savedPath": "docs/mobile-gpu-perf/sources/occupancy-ilp/amd-lab-notes-register-pressure.md",
    "ok": true,
    "oneLineWhatItContains": "Measured VGPR-cutting case study (102->96 VGPRs, 4->5 waves): replace pow with mul, shrink variable lifetimes, __restrict__; occupancy is a step function of VGPR count."
  },
  {
    "url": "https://www.sebastianaaltonen.com/blog/no-graphics-api",
    "savedPath": "docs/mobile-gpu-perf/sources/occupancy-ilp/aaltonen-no-graphics-api.md",
    "ok": true,
    "oneLineWhatItContains": "Raw-pointer/wide loads vs texel buffers: RGBA8 texel load = 4 VGPRs vs 1 for packed raw load, 2x throughput/3x lower latency, shorter register lifetime; mobile TBDR notes (thinner on hard SSBO numbers)."
  },
  {
    "url": "https://www.slideshare.net/DICEStudio/future-directions-for-computeforgraphics",
    "savedPath": "docs/mobile-gpu-perf/sources/occupancy-ilp/andersson-dice-future-directions-compute-for-graphics.md",
    "ok": true,
    "oneLineWhatItContains": "DICE talk: mega-shader worst-case static allocation kills occupancy (switch -> occupancy 5->2) — the argument for SPLITTING the raymarch vs deferred-lighting kernel; portability/10x perf-delta caveats."
  }
]
```

All 6 fetched successfully (ok:true), no failures. The Volkov PDF binary is kept alongside its .md.

Key synthesis for the grass-march question (captured in each .md's "Relevance" section):
- **Volkov** is the direct answer that a dependent march loop is the ILP=1 latency-exposed regime; the confident fix is issuing several independent baked-field fetches before consuming any (prefetch), plus unrolling — occupancy is likely NOT the lever until measured.
- **VGPR pressure** is the shared currency across all AMD sources: cut it via packed raw loads (Aaltonen no-graphics-api: 4 VGPRs→1 for the field sample), scalar loads for wave-invariant params, fp16 demotion, shorter live ranges, replacing `pow` with muls.
- **Andersson/DICE** gives the structural occupancy lever: splitting the raymarch vs deferred-lighting mega-shader so the compiler stops allocating worst-case registers for every path.
- Confident-vs-measurement calls are flagged per file (e.g. WebGPU lacks raw pointers/scalar-load intrinsics, so texel→storage-buffer wins need backend VGPR inspection).

---

## (agent)

All 8 sources fetched and saved to `docs/mobile-gpu-perf/sources/visbuffer-mobile/`. Manifest:

| # | URL | savedPath (basename) | ok | Contains |
|---|-----|------|----|----------|
| 1 | developer.arm.com/.../mali-and-unreal-engine-s-nanite | `arm-mali-nanite-mobile.md` | true | Mali TBR + Nanite: SW-vs-HW raster on a tiler, `VK_KHR_shader_image_atomic_int64` required, MinPixelsPerEdgeHW routing table, cluster-culling wins on low-vertex-throughput tilers, atomic/small-dispatch overhead warnings, "compute shading not recommended on Mali", HSR interaction. |
| 2 | tellusim.com/compute-raster | `tellusim-compute-raster.md` | true | Compute vs HW raster tri/sec across GPUs incl Apple M1/A14, Adreno; **64-bit atomics unavailable on mobile+Metal → depth-only 32-bit atomicMax, redundant tri re-intersection needed**; MDI dies on tiled GPUs. |
| 3 | jcgt.org/.../0002/02/04/paper.pdf | `jcgt-burns-hunt-visibility-buffer.md` + `.pdf` | true | The foundational Burns-Hunt vis-buffer paper: **the g-buffer-bandwidth argument** (16-32 B/sample, 46 GB/s example) that makes vis-buffer win on mobile/integrated/bandwidth-limited GPUs; 4-byte tri+instance ID; gains scale with sample rate + cache size. |
| 4 | advances.realtimerendering.com/.../Karis_Nanite | `karis-nanite-siggraph-2021.md` + `.pdf` | true | Primary source: **64-bit atomicMax depth packing (30 depth/27 cluster/7 tri; payload must fit ≤34 bits)**, micropoly + scanline SW rasterizers with inner-loop code, SW/HW per-cluster split, **overdraw is atomic/coverage-bound and "one of many reasons Nanite doesn't perform well with leaves and grass"**, VSMs. |
| 5 | media.gdcvault.com/.../Nanite+GPU+Driven+Materials.pdf | `gdc2024-nanite-gpu-driven-materials.md` + `.pdf` | true | Confirms 64-bit atomic max = 32-bit depth high / 32-bit ID low; depth-EQUALS material-ID culling (Dawn engine); programmable raster (masked opacity) = the aggregate-foliage cost. Deck explicitly excludes raster/cull. |
| 6 | elopezr.com/a-macro-view-of-nanite | `elopezr-macro-view-of-nanite.md` | true | R32G32_UINT (64-bit) vis buffer bit layout, SW-vs-HW split, material-range tile classification, why vis-buffer over g-buffer. |
| 7 | themaister.net/.../modernizing-granites-mesh-rendering | `themaister-modernizing-granite-mesh-rendering.md` | true | **Primitive/micro-poly culling to cut rasterizer pressure** with GLSL code (back-face cross-product + bbox micro-poly reject giving 3.5-10× in dense scenes), benchmarks across NV/AMD, mobile-fallback meshlet design. |
| 8 | sctheblog.com/blog/hair-software-rasterize | `sctheblog-hair-software-rasterize.md` | true | WebGPU SW rasterization: incremental edge-function (A·x+B·y+C) opt, coarse-tile + fine per-pixel two-phase, register pressure notes; same author as the nanite-webgpu u16-depth-packing issue #1. |

Key load-bearing findings for the election-under-dense-overlap question: Nanite depends on a single **64-bit atomicMax** (32-bit depth + 32-bit payload). **WebGPU has no `atomic<u64>`**, so our election must either pack into 32 bits (fewer depth bits → z-fight/leak risk) or split depth-atomic + redundant triangle re-intersection (Tellusim path). The dense-overlap cost is confirmed structural: overdraw from overlapping clusters/aggregates is **atomic-bound (large tris) / coverage-bound (medium)**, and Nanite itself flags leaves/grass as its weakest case with no per-pixel occlusion cull to save it — only prev-frame cluster-granularity HZB. Levers surfaced: per-cluster SW/HW routing (Mali MinPixelsPerEdgeHW), micro-poly/back-face primitive culling (themaister), and depth-EQUALS material-ID culling.

Note: the two Karis/GDC `.md` summaries flag an action item — verify what depth precision and packing OUR `atomicMax` election actually uses in `src/nanite/Tsl.ts` and the NaniteRaster election path; I did not inspect those files (out of scope for this fetch task).

---

## (agent)

All 6 sources fetched and saved to `docs/mobile-gpu-perf/sources/webgpu-wgsl/`. Manifest:

```json
[
  {"url":"https://discourse.threejs.org/t/webgpu-performance-issue/87939",
   "savedPath":".../threejs-forum-webgpu-performance-issue.md","ok":true,
   "oneLineWhatItContains":"Forum thread: WebGPU renderer 4x slower than WebGL on many non-instanced meshes; maintainer Mugen87 pins it on the UBO system's per-render-item cost (ref #30560); fix = instance/batch."},
  {"url":"https://threejsroadmap.com/blog/profiling-webgpu",
   "savedPath":".../threejsroadmap-profiling-webgpu-shaders.md","ok":true,
   "oneLineWhatItContains":"THE profiling recipe: WebGPURenderer({trackTimestamp:true}) + resolveTimestampsAsync(TimestampQuery.COMPUTE/RENDER); returns per-pass-type GPU-time sum since last resolve; warmup 20 + 50 samples + median/p10/p90. (403'd WebFetch; recovered via curl+extract.)"},
  {"url":"https://github.com/mrdoob/three.js/issues/32735",
   "savedPath":".../threejs-issue-32735-compute-pipeline-caching.md","ok":true,
   "oneLineWhatItContains":"Compute pipeline cache keys on ComputeNode instance id, so N nodes sharing identical WGSL compile N pipelines; proposed fix keys on WGSL id + binding layout — DIRECTLY answers 'do we duplicate pipelines across per-level dispatches?'"},
  {"url":"https://github.com/gfx-rs/wgpu/issues/4456",
   "savedPath":".../wgpu-issue-4456-wgsl-to-metal-compile-timeout.md","ok":true,
   "oneLineWhatItContains":"WGSL validates but Apple/Metal MSL compile hangs at pipeline-creation (XPC compiler timeouts) on M1 Max — evidence that WGSL→MSL compile is a real, pathological boot-time cost."},
  {"url":"https://developer.chrome.com/blog/new-in-webgpu-130",
   "savedPath":".../chrome-new-in-webgpu-130.md","ok":true,
   "oneLineWhatItContains":"Chrome 130: Tint IR makes WGSL→MSL translation up to 10x faster (mitigates the compile cost above); plus dual-source-blending feature."},
  {"url":"https://enginearchitecture.realtimerendering.com/downloads/reac2023_modern_mobile_rendering_at_hypehype.pdf",
   "savedPath":".../reac2023-modern-mobile-rendering-at-hypehype.md (+ .pdf kept)","ok":true,
   "oneLineWhatItContains":"Aaltonen HypeHype talk: the mobile-HW-tax slide verbatim ('16KB uniform buffers! SSBOs are slow!', wave intrinsics <10%, no 64-bit ints, avoid complex shaders) + build-all-PSOs-at-startup — the structural case against heavy storage-buffer nanite work on phones."}
]
```

Key findings for the task's four questions:
- **#32735 (duplicate pipelines):** confirmed mechanism — three.js keys the compute cache on ComputeNode instance id, so any place we spawn a distinct node per level/band/tile sharing one kernel compiles a redundant pipeline. Worth auditing our nanite compute-node creation.
- **Aaltonen SSBO tax:** real and named — "SSBOs are slow" on min-spec mobile; we use storage buffers heavily, so this is the structural headwind for the APPLE/MOBILE arc (#72).
- **WGSL→MSL compile cost:** wgpu #4456 shows it can hang at pipeline creation on Apple/Metal; Chrome 130 Tint IR is up to 10x faster — a candidate cause and partial mitigation for boot freezes.
- **Profiling recipe:** concrete and directly usable — `trackTimestamp:true` + `resolveTimestampsAsync(TimestampQuery.COMPUTE|RENDER)`, resolve immediately after a single lane's dispatch to isolate it, warmup 20 + 50 samples + median/p10/p90. This is exactly how to measure grass vs raster vs resolve on this machine.

One recovery note: the profiling blog 403'd to WebFetch but was recovered verbatim via curl + text extraction (all code snippets preserved).

---

## (agent)

_scope: Apple/mobile GPU register file, occupancy model, fp16, scalar ISA, and spill behavior — judged against our grass raymarch, kVoxScatter, and resolve. HEADLINE PREMISE-AUDIT CATCH: the theme's \"fp16 double-rate\" is an M5 property; on M1 Max (Apple 7) fp16 and fp32 FMA are 1:1 (256:256/core-cycle per Turner's table), so the fp16 payoff on OUR target hardware is purely OCCUPANCY + dependent-latency reduction (half the register footprint → more simds/core), NOT raw FLOP doubling. This aligns exactly with the grass file's own self-diagnosis (\"occupancy-bound; 47.6→15.1ms from shrinking live state, not ALU\"). The grass march is the measured #1 Apple cost (+14.5ms) and is a serial ILP=1 dependent chain — the worst latency regime with NO ILP lever available, so occupancy (register/fp16 cuts + fewer steps) is the ONLY dial; the M5 'scalarize float4' fix does not apply to our independent-component vec ops. 32-bit atomicMax election is the forced M1 Nanite path (no 64-bit/M2 hw atomics, unreachable in WebGPU anyway) and our aLoadU pre-gate already minimizes it. Highest-value actionable lever: fp16 the LOCAL march-space state (gated on a live occupancy/A-B capture), mirroring the guide-ctx half-packing already shipped._

- **fp16 "double-rate" does NOT hold on M1 Max — the fp16 win here is OCCUPANCY, not ALU doubling (premise correction)**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: metal-benchmarks-README.md Operations-per-Second table (lines 126-127): M1/A15 F16-FMA=256 and F32-FMA=256 per core-cycle (1:1); ALU-Bottlenecks table (lines 161-198) shows fp16's real advantage is confined to low-occupancy / register-dependent regimes. M5-roofline's 2.0x fp16 is an M5-only property (dedicated 16-bit ALUs), which the digest explicitly warns does not transfer.
  - _ourCode_: N/A — architecture-level, governs how to read every other fp16 finding below
  - _mechanism_: On Apple 7 (M1 Max) fp16 and fp32 FMA run at the SAME 256 ops/core-cycle, so converting math to half buys ZERO raw throughput. What fp16 DOES buy on M1: (a) half the register footprint → more SIMD-groups resident → higher occupancy; (b) lower dependent-op latency (README line 161: back-to-back dependent FMUL penalty 0.84cyc@32b vs 0.56cyc@16b; at 4 simds/core ILP=1, FFMA 11.3cyc→3.9cyc). Both matter ONLY when a kernel is occupancy- or latency-bound — which our grass march self-reports being.
  - _fix_: Frame all fp16 work below as an OCCUPANCY/latency lever, not a FLOP lever; do not expect a 2x speedup, expect a move to a cheaper row of the ALU-bottleneck table. Gate every fp16 change on a live A/B, since the win is regime-dependent.
  - _webgpuFeasible_: partial
  - _impact_: high
  - _confidence_: confident
- **Grass march is self-diagnosed occupancy-bound yet runs entirely fp32 — half-precision the LOCAL march-space state to raise occupancy**
  - _kind_: violation
  - _source_: dougallj-applegpu-docs.md: "Using fewer registers (e.g. by using 16-bit types instead of 32-bit) allows more SIMD-groups to fit in the physical register file (higher occupancy), which improves performance." 128 GPRs/thread cap. metal-benchmarks lines 161/208 (fewer live regs → higher simds/core → cheaper FFMA row).
  - _ourCode_: src/nanite/NaniteGrass.ts:870-1334 (kRay 256-iter loop) — every march var is fp32 .toVar(); the file's own comment (lines 145-147) states "the kernel is occupancy-bound: 47.6→15.1 ms came from shrinking live state, not ALU"
  - _mechanism_: The march carries a very large live fp32 register set per lane (pos, rd, tCur/tEnd/tBest, ca/sa, Slx/Slz/Swx/Swz/Sqx/Sqz, qbx/qbz/az, tile coords, normals, hit params). On M1 each lane's GPRs come from the ~208KB/core file shared across resident simds; halving the footprint of the many [0,1]/small-magnitude locals lets more simds/core resident, which is the ONLY latency-hiding lever available to a serial march (see next finding). Their measured 47.6→15.1ms from shrinking live state IS this lever — fp16 continues it.
  - _fix_: Convert march-space LOCAL quantities to f16: tile-space coords qbx/qbz (∈[0,1]), the swirl ca/sa, offsets offX/offZ (<~2m), the sheared dir ex/ez, azimuth az, the baked normal components, tPar/topEff. KEEP f32 for anything world-anchored: pos.xyz (±155m ring, f16 ~0.06m resolution at 128m is too coarse), tCur/tEnd (absolute t), ground Y. WebGPU: requires `enable f16;` + the `shader-f16` device feature (Metal supports it), but three.js TSL has no first-class scalar half type today — so this is partial/blocked at the TSL layer and may need raw-WGSL FunctionNode for the hot inner block. Gate on live A/B.
  - _webgpuFeasible_: partial
  - _impact_: high
  - _confidence_: speculative
- **The per-ray raymarch is an ILP=1 serial dependent chain — the worst-case latency regime; occupancy is the ONLY lever (you cannot add per-ray ILP)**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: metal-benchmarks-README.md ALU-bottleneck tables: at 4 simds/core FFMA ILP=1 = 11.3 cyc, at 8 simds/core = 5.71 cyc (2x from occupancy alone, zero code change); line 208 "ALU utilization maxes out at 24 simds/core... Apple would rather you spill to device memory than decrease ALU utilization". M5-roofline: FP32 needs 8 INDEPENDENT chains to saturate (4-cyc FMA latency).
  - _ourCode_: src/nanite/NaniteGrass.ts:870 loopUN('gro',0,256,...) — each iteration depends on the previous (tCur advances; fetch→decode→shear→bomb→test is a long dependent chain within an iteration)
  - _mechanism_: A raymarch has ~zero ILP across iterations (step N+1 needs step N's tCur) and a long dependent chain within a step, so it sits at the ILP=1 rows where fp32 FFMA costs 5-11 cyc. The M5-roofline '8 independent chains' fix is UNAVAILABLE — you can't parallelize a serial march. The remaining lever is occupancy: more rays-in-flight (simds/core) hide each ray's latency. This means every grass perf lever must be judged as 'does it raise simds/core?' — which reframes register reduction + fp16 + fewer spills as THE path, and explains why ALU micro-opts inside the loop did nothing (line 99-100 '≤2ms law NOT met').
  - _fix_: PROFILING-TODO first: capture the kRay kernel in Xcode GPU / Instruments and read the actual simds/core (occupancy) + spill counters to confirm which table row we occupy (digest caveat: do NOT assume occupancy collapse, TEST it — Volkov shows 25% can suffice via ILP, but a serial march has no ILP so occupancy is decisive here). If occupancy is <8 simds/core, register reduction (finding above) is high-value; if already high, the cost is intrinsic per-step fetch latency and the lever is fewer STEPS (coarser far-band marching), not fp16.
  - _webgpuFeasible_: yes
  - _impact_: high
  - _confidence_: speculative
- **"Apple spills to device memory rather than lower ALU utilization" — a heavy serial march that spills pays device-memory latency INSIDE its critical path**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: metal-benchmarks-README.md line 208: "ALU utilization maxes out at 24 simds/core. This is also the lowest occupancy you can create by over-allocating registers. Apple would rather you spill to device memory than create chances to decrease ALU utilization."
  - _ourCode_: src/nanite/NaniteGrass.ts:870-1334 — the march + its two nested fetch closures (bakedTexel, the L2 overlay 1335-1460) create a very wide live-var set that can exceed the 128-GPR budget
  - _mechanism_: Because Apple never lets occupancy fall below 24-simds ALU-util by cutting occupancy, an over-register kernel instead SPILLS live vars to device memory. For a bandwidth/ALU-bound kernel that's tolerable, but for our LATENCY-bound serial march every spilled reload injects device-memory latency directly into the dependent chain — compounding the ILP=1 penalty. So register pressure hurts us through spill-latency, NOT through ALU starvation (which is why 'occupancy collapse' won't show as zero-throughput — the digest's warning is correct here).
  - _fix_: PROFILING-TODO: check the kRay Metal shader for spill/fill (stack) traffic in a capture. If present, the fix is to shrink the simultaneously-live set — hoist the L2-overlay branch (NaniteGrass.ts:1335) out of the hot path (it already gates to ≤3 fires/pixel but its locals inflate the loop's register high-water mark even when not taken), and reuse .toVar() slots. fp16 (finding 2) also directly reduces spill pressure.
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: speculative
- **32-bit atomicMax election is the ONLY Nanite-atomic path on M1 Max (no 64-bit atomic, no M2 hw accel) — inherent 2.5x-bw/5x-latency cost; our pre-gate already minimizes it**
  - _kind_: trick-to-steal
  - _source_: metal-benchmarks-README.md Nanite-Atomics section: "Apple GPU only supports 32-bit atomics on pointer values... Nanite can run entirely on 32-bit buffer atomics at a 2.5x bandwidth/5x latency cost. Apple added hw acceleration to the M2 series... The A15/A16 [and M1] do not support Nanite atomics."
  - _ourCode_: src/nanite/NaniteGrass.ts:314-323 emitPx (aLoadU pre-check → atomicMax → conditional atomicStore); src/nanite/NaniteVoxelRaster.ts:383-392 (same world1 pattern: read prevE, gate cand>prevE, atomicMax, winner atomicStore)
  - _mechanism_: M1 Max (Apple 7) has NO 64-bit atomic min/max and NONE of the M2 Nanite-atomic hardware, so our two-word split (depthKey24<<8|id8 via atomicMax on payloadV, then a separate atomicStore of the full id into visBV) is exactly the '32-bit-only' fallback the source describes — carrying its 5x latency penalty per contended pixel. In dense foliage (many blades/bricks contending the same pixel) this atomic latency is a real M1-specific cost. IMPORTANTLY we already do the right thing: the non-atomic aLoadU pre-gate before atomicMax suppresses the majority of atomic writes (a lane whose cand can't beat the current winner never issues the atomic).
  - _fix_: No code change — this is confirmed optimal for the WebGPU constraint (WGSL exposes no 64-bit atomics and no texture atomics, so the M2 fast path is unreachable regardless). KEEP the aLoadU pre-gate everywhere it exists; if any emit path lacks it, add it. Treat atomic contention as a fixed M1 tax and reduce it only by emitting FEWER candidate pixels (coarser far-band, the occlusion pre-gate), never by changing the atomic itself.
  - _webgpuFeasible_: no-metal-only
  - _impact_: medium
  - _confidence_: confident
- **"Avoid float4 / scalarize" does NOT apply to our independent-component vec ops — the M5 trap was a cross-DEPENDENT FMA chain, which we don't have**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: michaelstinkerings-m5-gpu-roofline.md: the 4.75x came from switching cross-DEPENDENT float4 FMA chains to 8 INDEPENDENT scalar chains — i.e. the fix was adding ILP, not deleting vec types. dougallj docs confirm Apple scalarizes vec ops into independent scalar fmadd already.
  - _ourCode_: src/nanite/NaniteGrass.ts / NaniteVoxelRaster.ts — vec2/vec3/vec4 usage is overwhelmingly independent-component (position adds, normal builds, bomb rotations)
  - _mechanism_: WGSL vecN ops on Apple lower to N independent scalar ops that the 4 schedulers can co-issue — that IS good ILP, not the trap. The M5 penalty only appears when the 4 components form a serial dependency (e.g. accumulating a float4 dot into itself across a chain). Our vec math is component-independent, so blindly 'scalarizing' it would gain nothing and lose readability. The genuinely trap-shaped code is the serial raymarch (finding 3), where the dependency is across LOOP ITERATIONS, not across vector lanes — and that can't be vectorized away.
  - _fix_: Do NOT spend effort de-vectorizing vec ops. Redirect that effort to the two real levers: raising march occupancy (fp16/register cuts) and cutting march STEP count. If any hot dependent scalar accumulation exists (none found in the read), THAT would be the place to add independent partial-sum chains.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident
- **kVoxScatter 128-lane workgroup carries a large workgroup-array + per-lane register set that may cap scatter occupancy**
  - _kind_: profiling-todo
  - _source_: metal-benchmarks-README.md: register file ~208KB/core, shared mem ~60KB/core, 128 GPRs/thread; occupancy set by whichever of registers/threadgroup-memory is exhausted first. dougallj: fewer resources → more resident simds.
  - _ourCode_: src/nanite/NaniteVoxelRaster.ts:609-630 — per-cluster workgroupArrays: wgBbX0/Y0/W/H/Cand (5) + wgOccMask + wgBrCx/Cy/Cz/Hf/CellLo/CellHi/CellOk (7 for voxCell) + wgWind(6) + wgVisible; WG_RASTER=128 lanes (=4 SIMD-groups); Phase-B lanes also hold heavy per-pixel election state
  - _mechanism_: With voxCell + voxOccGate + voxWind all default-on, the scatter kernel allocates ~13 workgroupArrays × up to 128 entries plus a wide per-lane register set. On M1 the resident-simd count per core is min(regs, threadgroup-mem) limited; a fat threadgroup footprint can pin the scatter kernel to few simds/core, which for the per-pixel Phase-B election (its own dependent atomic path) reduces latency hiding. Voxel scatter is NOT the measured #1 (grass is), so this is medium/low — but dense-canopy poses are where it bites.
  - _fix_: PROFILING-TODO: capture kVoxScatter occupancy on an M1 at a dense-canopy pose and check whether threadgroup memory or registers is the limiter. If threadgroup-mem-bound, the voxCell 7-array block (only needed for eligible COARSE bricks) is the candidate to shrink — but WGSL workgroup arrays are static-sized, so the only lever is packing (e.g. fold wgBrCx/Cy/Cz into fewer words, store half-packed centers) or a separate specialized kernel for the ray-eligible path. The already-shipped voxRecip (line 257, kills the microcoded int-divide per fragment — Apple has no native int divide) is the correct prior instance of this class; continue that style.
  - _webgpuFeasible_: partial
  - _impact_: medium
  - _confidence_: speculative
- **Guide-ctx already half-packs its fields (grad, top/amp) while keeping ground Y in f32 — the correct precision split; extend it, don't regress it**
  - _kind_: trick-to-steal
  - _source_: dougallj docs (16-bit low/high halves of each GPR, fewer regs → higher occupancy); metal-benchmarks (fp16 shrinks footprint). Confirms half-packing storage is the right move where precision allows.
  - _ourCode_: src/nanite/NaniteGrass.ts:521-523 — guideCtx stores ground as full-word bcF2U(g) (f32 precision for world Y), grad as packHalfU(vec2(dgdx,dgdz)) and (topOut,amp) as packHalfU — half2 for the small-magnitude fields
  - _mechanism_: This is the register/bandwidth lever applied correctly at the STORAGE boundary: gradients (~[-1.4,1.4]) and sward-top/gust-amp (~[0,2]) tolerate f16's ~11-bit mantissa, so packing two per word halves the guide buffer traffic and the march's post-fetch register load; ground Y (absolute world height, needs sub-cm over ±hundreds of m) correctly stays f32. This is exactly the 'fp16 where it packs, f32 where precision demands' discipline the theme asks for — already shipped in the bake path.
  - _fix_: No fix needed — flag as the reference pattern. When applying finding-2 fp16 inside the march, mirror THIS split (local/relative→half, world-absolute→f32). Do not 'optimize' the ground field to half (would sink blades, matching the historical sunken-blade bug at lines 154-155).
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident
- **Resolve fragment is register-heavy (material+sun+CSM+IBL+GI) but measured NOT the bottleneck — presentClasses already applies the occupancy lever there**
  - _kind_: trick-to-steal
  - _source_: metal-benchmarks lines 161/208 (register pressure → occupancy/spill); dougallj (fewer regs → more simds). Our own measurement (grasslean A/B identical, memory 'resolve-wall REFUTED') establishes resolve shading is not the dense-foliage whale.
  - _ourCode_: src/nanite/NaniteResolve.ts:98-104 — presentClasses/RP-1 SKIPS building shading subgraphs of absent material classes, explicitly "strips samples/ALU from the shader ⇒ register pressure/occupancy (the vox-pass 37.5ms-cliff mechanism)"
  - _mechanism_: A monster fullscreen shader that builds every material class's lighting inflates the fragment's register high-water mark → lower occupancy / spills (the cited 37.5ms cliff). Compiling only present classes cuts the live-var set → higher occupancy. This is the SAME occupancy lever as the grass findings, already applied on the resolve. Because resolve is not the dominant Apple cost, fp16-ing its lighting accumulation is low expected value.
  - _fix_: Keep presentClasses on. Do NOT invest in fp16 resolve lighting until a capture shows the resolve fragment (not grass/pixel-loop/post) on the critical path at a dense pose — measured evidence says it isn't. If ever needed, the safe fp16 targets are albedo/normal/lighting accumulators in [0,1], never the reconstructed world position (getViewPosition, f32).
  - _webgpuFeasible_: partial
  - _impact_: low
  - _confidence_: confident

---

## (agent)

_scope: Occupancy vs ILP vs register pressure in the grass raymarch (kRay) — TEST, don't assume_

- **Is kRay occupancy-bound or ILP/latency-bound? — this is UNMEASURED and the whole theme hinges on it**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: Volkov GTC2010 (ILP hides latency at 4-25% occupancy; the march is a single-thread dependent chain = the ILP=1 latency-exposed regime, slides 32/40) + GPUOpen occupancy-explained (measured << theoretical ⇒ dependency/launch-bound, not resource-bound; 'peak occupancy != peak performance').
  - _ourCode_: src/nanite/NaniteGrass.ts:146-147 (dev comment asserts 'kernel is occupancy-bound: 47.6->15.1 ms came from shrinking live state, not ALU') and the march loop at :870-1503
  - _mechanism_: On Apple TBDR the M1-Max symptom is dense-foliage stall, NOT thermal/bandwidth. The kernel comment ALREADY claims occupancy-bound because shrinking live state gave a 3x win. But that 3x is equally consistent with shortening the dependent-ALU chain (Volkov's ILP axis), not just freeing VGPRs. Nobody has read the actual VGPR count / theoretical-vs-measured occupancy / whether the SIMD is latency-stalled vs ALU-saturated. Every downstream lever (split L2, fp16, prefetch) is chosen differently depending on this answer.
  - _fix_: PROFILE FIRST via Xcode Metal GPU capture on the M1-Max build: capture the grassRay compute pass, read the shader's register allocation + occupancy limiter, and read the 'Limited by' / latency bars. Equivalent to compiling the naga-emitted MSL through Metal and reading the pipeline stats. Decision rule (occupancy-explained:39): if measured occupancy is already healthy but perf is poor -> latency-exposed -> ILP/prefetch is the lever; if VGPR count sits just above a wave-count step -> register-reduction is the lever. Do NOT ship any of the below before this reading.
  - _webgpuFeasible_: yes
  - _impact_: high
  - _confidence_: confident
- **The 4th guide-ctx word is WASTED — bake widenT there and delete the per-step grassThin recompute (2 pow + sqrt + divides per accepted texel step)**
  - _kind_: trick-to-steal
  - _source_: AMD lab-notes register-pressure (replace pow/transcendental with precomputed/mul; shrink live ranges) + the guide-field-bake pattern already in this file (move per-step derivation to the O(area) bake).
  - _ourCode_: src/nanite/NaniteGrass.ts:524 (writes uint(0) to ctx word base+3 — dead) vs :924-931 (march recomputes widenT = 1/sqrt(grassThin(distT)) with grassThin containing pow(1.15) AND pow(1.6)) — and :500-506 where widenT is ALREADY computed at bake time for topOff
  - _mechanism_: widenT is a smooth function of distance only, so it is constant-per-texel — exactly what the guide bake exists to fold out of the per-pixel loop. The bake already derives it (line 501-506) and then throws it away; the march re-derives it every accepted texel step, paying two exp/log transcendentals + a sqrt + two divides. On the register-bound hypothesis these transcendentals also inflate VGPR lifetime inside the hot loop. Baking it costs zero (the word is already allocated and zero-filled) and removes ALU + a live temporary from the march.
  - _fix_: In kGuideBake pack widenT (or packHalf(widenT, spare)) into guideCtxW base+3 (NaniteGrass.ts:524). In the march, replace the grassThin(distT)-based widenT block (:924-931) with bcU2F/unpackHalf of cv.w. Texel-center dist vs step-exact distT differ negligibly at 0.84 m grain (same approximation the rest of the guide already makes). Pure WGSL, no constraint issue.
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: confident
- **Layer-2 golden overlay is a large rare branch fused into kRay — DICE worst-case-allocation occupancy killer; split it into its own sparse pass**
  - _kind_: trick-to-steal
  - _source_: Andersson/DICE 'Future Directions' (switch/branch -> compiler allocates worst-case VGPR/LDS across ALL paths -> occupancy 5->2; splitting a mega-shader into specialized kernels restores tight resource bounds).
  - _ourCode_: src/nanite/NaniteGrass.ts:1335-1499 (the entire RAY_LAYER2 block: a second bomb/shear/fetchBand/validate/normal-decode chain with its own live set ca2,sa2,Q2x/z,e2x/z,r2x/z,idT2,top2,gR2... ) fired at most 3x/pixel and only on rd.y<-0.3 downward rays
  - _mechanism_: Even though L2 executes for a tiny minority of pixels/steps (downward rays, holes only, budget 3), its code is compiled into the SAME kernel, so the compiler must reserve its worst-case VGPR footprint for EVERY wave of the common upward-looking dense-sward case that never enters it. On the occupancy-bound hypothesis this directly caps the wave count for the frame's dominant pixels. This is the exact DICE 'occupancy 5->2 for free' anti-pattern.
  - _fix_: After PROFILE confirms kRay's VGPR count sits above a wave-count step (Metal capture), move L2 to a second compute dispatch that runs only over pixels where L1 emitted nothing and rd.y<-0.3 (a compacted/append list, or just a full-screen pass early-out on the L1 election being empty). The main kRay then drops L2's live set and instruction footprint. Gate: A/B whole-frame gpuWall + shotdiff parity (the hole-fill must look identical). Speculative pending the VGPR reading — if kRay is latency-bound not register-bound, this buys little.
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: speculative
- **The march is a dependent chain (ILP=1); the empty-texel DDA-skip sub-chain is prefetchable, the accept path is not**
  - _kind_: profiling-todo
  - _source_: Volkov slide 32/40 ('threads don't stall on memory access, only on data dependency'; issue several independent loads BEFORE consuming any; 84% mem peak at 4% occupancy via bytes/thread). 
  - _ourCode_: src/nanite/NaniteGrass.ts:870-920 (loop: pos -> texel index -> guideCtx4 fetch cv:892 -> vertical reject -> guideMask2 fetch mv:921 -> field fetches f1/f2:957-958 -> texture3D bake smp:1135 -> decide -> advance tCur). The skip advance (:914-919) is a pure deterministic DDA that does NOT depend on any fetch result.
  - _mechanism_: Each march step cannot begin until the previous step's fetch is consumed to decide the advance — the ILP=1 regime Volkov shows leaves the SIMD idle on memory latency (400+ cyc). The near-band cost is dominated by miss-path pixels skipping empty texels (per the file's own note at :151-153). On that skip path the NEXT texel address is computable without the current fetch (the guide DDA is deterministic when the texel is empty), so guideCtx4[next] can be issued before consuming guideCtx4[current] — a software-pipelined 2-deep prefetch that hides ctx-fetch latency exactly as Volkov prescribes. The accept path (dTile-dependent tHit) genuinely cannot be prefetched.
  - _fix_: Prototype a 2-stage software pipeline for the SKIP walk: hoist the next-texel ctx load to issue before the current-texel reject decision, keeping both cv_cur and cv_next live (register-resident staging, Volkov slide 34 'local arrays live in registers'). Measure gpuWall on a look-down dense pose. Only worth it if PROFILE says latency-exposed; if register-bound, the extra live cv_next fights occupancy (the Volkov/Aaltonen tension) — must measure, do not assume.
  - _webgpuFeasible_: partial
  - _impact_: medium
  - _confidence_: speculative
- **Baked raycast tile + guide fields are RGBA textures (4 VGPRs/sample, Aaltonen) — but the raw-buffer fix is BLOCKED by the hardware-filtering requirement**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: Aaltonen 'No Graphics API' (RGBA texel load allocates 4 VGPRs immediately vs 1 for a packed raw load; raw loads 2x throughput / 3x lower latency, shorter register lifetime) + Aaltonen GCN occupancy (typed buffers/textures do NOT support scalar loads).
  - _ourCode_: src/nanite/NaniteGrass.ts:733-737 (bake Data3DTexture LinearFilter, sampled texture3D at :1041/:1135) and :415-419 (guideFieldT1/T2 HalfFloat RGBA LinearFilter, sampled texture() at :957-958). NOTE the ctx/mask ALREADY use the raw-buffer lever correctly: :389-395 uvec4/uvec2 storage buffers, unpacked in-shader via bcU2F/unpackHalfU (:893-895).
  - _mechanism_: Per Aaltonen each vec4 texture sample in the hot loop costs ~4 live VGPRs; on the occupancy-bound hypothesis three such samples/step (T1, T2, bake) is real pressure. The theoretically-correct fix is packed storage-buffer raw loads + in-shader unpack (1 VGPR). BUT the bake tile requires hardware TRILINEAR across x/z/angle (his interpolation, :701-702) and the fields require bilinear; WebGPU storage buffers expose NO hardware filtering, so a manual 3D trilinear would be 8 raw loads + blend ALU per sample — strictly MORE memory ops and MORE VGPRs than the single filtered texel fetch. The register win is real but the filtering requirement inverts it. The devs already applied the lever exactly where filtering is not needed (ctx/mask).
  - _fix_: BLOCKED for the bake tile and fields by the WebGPU no-filtered-storage-buffer constraint — do not pursue the texture->buffer swap there; it would regress. The only reachable variant: if PROFILE shows the fields are the VGPR peak, drop guideFieldT2 to a single packed uint (swirl cos/sin + arc as snorm8x4) sampled NEAREST and reconstruct — but that loses the bilinear continuity the file's grid-quilt post-mortem (:16-20) was fought to get, so likely a quality regression. Record as measured-and-blocked.
  - _webgpuFeasible_: no-metal-only
  - _impact_: low
  - _confidence_: confident
- **'Split raymarch from deferred lighting' (DICE) is ALREADY DONE at the pass level — do not re-recommend it**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: Andersson/DICE (split mega-shader so the compiler stops worst-case-allocating registers for the lighting path in the march kernel).
  - _ourCode_: src/nanite/NaniteGrass.ts:761-769 (resolveRay only TAPS rayNrmTex) and :1522-1527 (kRay writes vec4(nrm,tPar) to a screen StorageTexture); the actual sun/CSM/IBL/GI shading runs in the separate NaniteResolve fullscreen pass, not in kRay.
  - _mechanism_: The single biggest DICE lever — keeping expensive deferred lighting OUT of the march kernel's register footprint — is structurally satisfied by the vis-buffer/deferred-resolve architecture: kRay emits an id+depth+normal and exits; lighting is a downstream pass. So kRay does NOT carry PCSS/GI VGPRs. The confirmatory point: the GRASS_LEAN experiment (:134-141) tried to move lighting into a per-texel bake and was MEASURED no-win, further confirming the resolve shading is not the grass bottleneck (the ~7 ms delta is the emit/march path). The remaining DICE residue is intra-kRay branchiness (Layer-2, see that finding), not lighting.
  - _fix_: No action — confirmed already applied. Redirect any 'split the mega-shader' effort to the Layer-2 split finding, which is the real remaining worst-case-allocation surface inside kRay.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident
- **fp16 demotion of march math — halves VGPRs per Aaltonen, but world-space t/position math is precision-blocked; only tile-local math is a candidate**
  - _kind_: profiling-todo
  - _source_: Aaltonen GCN occupancy ('16-bit registers pack two per VGPR, good for 2D/3D address math'; demote to fp16 where precision allows).
  - _ourCode_: src/nanite/NaniteGrass.ts:836-837 (tCur/tBest f32 accumulators) and tile-local temporaries qbx/qbz/az (:1013-1027), hgt (:1007), offX/offZ (:1008-1009)
  - _mechanism_: Aaltonen: fp16 halves VGPR pressure — directly attacks the occupancy-bound hypothesis. BUT tCur accumulates world-space distance along rays over a +/-161 m ring across up to 256 steps; fp16's ~10-bit mantissa gives ~0.15 m resolution at 161 m — coarser than a blade, so t/ro/rd/world-pos MUST stay f32. The genuinely-local quantities (tile-space uv in [0,1], azimuth, clamped hgt<=0.6 m, sub-meter offsets) tolerate fp16. Realistic VGPR saving is a fraction of the working set, and TSL's f16 emission maturity (requires enable f16 / shader-f16 via Tint) is unverified in this codebase.
  - _fix_: Deferred until PROFILE confirms register-bound. If so, prototype fp16 ONLY on tile-local math (qbx/qbz/az/hgt/offX/offZ/the bomb rotations) leaving all world-space t/position in f32; verify Tint emits packed f16 and that shotdiff is bit-clean. High effort, uncertain payoff — lowest-priority of the register levers.
  - _webgpuFeasible_: partial
  - _impact_: low
  - _confidence_: speculative

---

## (agent)

_scope: Apple official guidance + profiling tooling (WWDC/tech-talks): the actionable levers reduce to (1) run a Metal GPU-capture of the Chrome/Dawn process to MEASURE occupancy/spills/top-limiter on kGrassRay + kVoxScatter — currently only inferred; (2) f16 shader math (double-rate ALU + register relief) is our biggest unused lever but is blocked by TSL having no half type, not by WebGPU; (3) the register-spill/low-occupancy case study (10580) matches the 256-step grass march shape and should be confirmed before acting. The 'avoid device atomics', 'HSR', and 'memoryless tile memory' levers are respectively already-refuted-for-us, already-satisfied-by-architecture, and Metal-only-blocked._

- **Metal GPU-capture of the Chrome/Dawn process is the missing measurement — occupancy% + spilled bytes + top-limiter for kGrassRay and kVoxScatter**
  - _kind_: profiling-todo
  - _source_: WWDC20 10603 'Optimize Metal with GPU counters' (Metal Debugger = all 150+ counters at draw/encoder granularity, per-draw counter mode, NOT thermal-affected) + Tech Talk 10580 (compiler statistics show spilled bytes + occupancy% — that is how the 16% case was diagnosed).
  - _ourCode_: N/A — architecture-level; targets the kernels at src/nanite/NaniteGrass.ts:788 (kGrassRay dispatch line 1539 .compute(Wq*Hq,[256])) and src/nanite/NaniteVoxelRaster.ts:1349-1543 (kVoxScatter election).
  - _mechanism_: Dawn on macOS lowers WebGPU to Metal, so Xcode's Metal Debugger / Instruments 'Metal System Trace' can attach to the Chrome GPU-process and read the real per-kernel occupancy%, spilled-byte count, and the single top limiter (ALU vs buffer vs LLC-atomic vs tile-memory). Every Apple-lever recommendation below is currently INFERRED; this capture is what turns 'grass is ALU/occupancy-bound' from hypothesis into a measured fact and tells us whether f16/register-shrink is even the right lane. Premise-audit passes: the target (grass raymarch, +14.5ms) is the measured #1 Apple cost, not a guess.
  - _fix_: Build a release Chrome/Chromium, launch with --enable-dawn-features + the GPU process made capturable, run our dense-forest scene, and use Xcode > Debug > 'Capture GPU Frame' (or Instruments Metal System Trace) attached to the GPU helper process. Read: (1) kGrassRay occupancy% and spilled bytes; (2) its top limiter group; (3) same for kVoxScatter. This is the single highest-value action in this theme and gates all the others.
  - _webgpuFeasible_: yes
  - _impact_: high
  - _confidence_: confident
- **kGrassRay is a 256-iteration loop with dozens of live f32 vars — the exact shape of the 10580 '16% occupancy from register spilling' case study**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: Tech Talk 10580: kernel stuck at 16% occupancy because spilled bytes + temporary registers exhausted thread memory; recovery = shrink stack/live state, 16-bit types, known max thread count. WWDC20 10603: low occupancy is NOT always a problem — Volkov-style ILP can hide it — so this must be MEASURED not assumed.
  - _ourCode_: src/nanite/NaniteGrass.ts:870 (loopUN 'gro' 0..256) with heavy live state — ro/rd/sdx/sdz/tCur/tBest/tEnd/bodyBest/nrmV/tParV plus per-step pos/txf/tzf/grad/ta/ca/sa/bombF results, all f32/vec (lines 874-1015+).
  - _mechanism_: Apple GPUs allocate registers in blocks; when a kernel's live-var footprint crosses the budget, occupancy drops in whole-block steps and the compiler spills to device memory (a buffer-limiter cost). A 256-step march carrying this many live vec3/vec2 f32 temporaries is precisely what tanked the 10580 kernel to 16%. If confirmed, cutting live state (or f16-ing it) buys occupancy that hides the per-step fetch latency — directly attacking the +14.5ms.
  - _fix_: First the profiling-todo above to read actual occupancy/spill. If spilling: shrink the loop's live set — hoist invariants out of the loop, narrow the wind/arc/normal intermediates to f16 (see f16 finding), and split the rarely-taken bakedTexel() field-fetch block so its temporaries don't inflate the whole-loop register class. The @workgroup_size(256) is already a compile-time constant (line 1539), so the 'known max thread count' lever from 10580 is already satisfied.
  - _webgpuFeasible_: partial
  - _impact_: high
  - _confidence_: speculative
- **No f16 in any shader MATH — the single most-cited Apple lever (double-rate ALU + more registers) is entirely unused, and blocked by our TSL authoring layer not by WebGPU**
  - _kind_: violation
  - _source_: WWDC20 10632 §1 (half/short → fewer registers → higher occupancy + faster arithmetic, conversions free) + 10603 ALU-limiter (16-bit FP = double rate, 32-bit = full rate) + 10580 lever #1 (prefer 16-bit to free registers). Digest caveat noted: M5 roofline gap attributed partly to scalar-ISA float4 decomposition — a compute/ALU cause f16 would help.
  - _ourCode_: src/nanite/NaniteGrass.ts:870-1160 march math is all f32/vec (verified via grep: no f16/half node anywhere in shader math); src/nanite/NaniteResolve.ts fullscreen shade is f32. Device is created WITHOUT requiredFeatures:['shader-f16'] (src/core/Engine.ts:84-88). TSL type table has only float/int/uint (node_modules/three/src/nodes/core/NodeUtils.js:83,332) — no half type.
  - _mechanism_: On Apple GPUs f16 runs at 2× the ALU rate of f32 AND halves register pressure, which raises occupancy — a double win exactly where an ALU-or-occupancy-bound per-pixel march lives. Our whole grass/voxel/resolve stack pays full f32 rate and full register cost. The block is our authoring layer: three.js TSL cannot emit a WGSL `f16`/`half` type today, and we never request the `shader-f16` device feature — so it is NOT reachable through TSL even though WebGPU/Dawn support it.
  - _fix_: Two-step. (1) Add requiredFeatures:['shader-f16'] at src/core/Engine.ts:84 (guarded by adapter.features.has). (2) Because TSL has no half node, the f16 math must go through a raw-WGSL escape (wgslFn / a custom Node) for the march-space intermediates only — wind shear, arc deflection (Sq·h²), swirl cos/sa, baked normal — keeping world POSITIONS in f32 (world coords overflow f16's ~65504 range; that is why position must stay f32). Gate behind the profiling-todo: only worth the raw-WGSL complexity if the capture shows ALU-rate or register-block-occupancy as the limiter.
  - _webgpuFeasible_: partial
  - _impact_: high
  - _confidence_: speculative
- **Vis-buffer election uses DEVICE-memory atomicMax (LLC atomics) everywhere — Apple says avoid; but the workgroup-election alternative was already measured DEAD for voxels and is N/A for grass**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: WWDC20 10603 buffer/LLC limiters: 'avoid device atomics and register spills'; 'refactor device atomics → threadgroup atomics / SIMD-lane ops' (device atomics are stored in the shared LLC).
  - _ourCode_: atomicMax into device visPayloadV at src/nanite/NaniteGrass.ts:318, src/nanite/NaniteRaster.ts:603, src/nanite/NaniteVoxelRaster.ts:1351 & 1543. Note the read-before-atomic guard (prevE at NaniteVoxelRaster.ts:1349, NaniteGrass.ts:316, NaniteRaster.ts:595) already elides the atomic when the candidate can't win.
  - _mechanism_: Device atomics land in the LLC and serialize under contention. Apple's fix (stage in threadgroup memory, flush once) only helps when many lanes contend the SAME pixel. For kVoxScatter that IS the case — but the code comment at NaniteVoxelRaster.ts:385-386 records that the on-chip wgElect/flush-merge was BUILT and REMOVED as measured-dead (the depth-bucketed bin's setup floor lost). For kGrassRay each pixel-lane elects its OWN pixel exactly once (NaniteGrass.ts:318) — zero cross-lane contention, so threadgroup staging is architecturally N/A. So this lever is largely refuted/inapplicable for our two hottest kernels; the prevE guard already captures the cheap win.
  - _fix_: No change recommended for grass (no contention) or voxel (already tested-and-reverted). Do NOT spend effort re-staging these into threadgroup atomics — the setup floor already lost once. Only revisit if the Metal capture shows the LLC-atomic limiter dominating kVoxScatter, in which case the lever to try is the min-pooled footprint-pyramid occlusion cull (already partly present) to cut election attempts, not threadgroup election.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident
- **Guide/mask buffers already pack half-precision and unpack per step — matches Apple 'pack tighter / avoid FP32 buffer inputs' but the value is then re-widened to f32 for the whole march**
  - _kind_: trick-to-steal
  - _source_: WWDC20 10603 buffer-limiter levers: 'pack tighter, smaller types, vectorize loads'; ALU-limiter: 'avoid FP32 texture/buffer inputs'. WWDC20 10632 §4: colocate/vectorize struct fields so loads vectorize.
  - _ourCode_: src/nanite/NaniteGrass.ts:894-895 (unpackHalfU on cv.y/cv.z — grad + sward-top fetched as packed half from guideCtx4, one uvec4 fetch per texel step) and the guideFieldT1/T2 texture fetches at 957-958.
  - _mechanism_: We already do the packed-fetch half of Apple's advice (one vectorized uvec4 load answers the whole texel context instead of many f32 taps — this is the 2026-07-04 rewrite that cut ~12ms). The unrealized part: every packed half is immediately widened to f32 and the rest of the step runs f32. If the loop is ALU/register-bound, keeping grad/top/swirl in half through the reject math compounds with the f16 finding above.
  - _fix_: Keep the packed-half fetch (it is correct and already winning). When/if the raw-WGSL f16 path lands, feed unpackHalfU results straight into half-typed march intermediates instead of f32-widening at NaniteGrass.ts:894-895. Blocked by the same TSL-has-no-half constraint as the f16 finding.
  - _webgpuFeasible_: partial
  - _impact_: medium
  - _confidence_: confident
- **Our opaque vis-buffer already IS the HSR/early-fragment-test win Apple pushes — architecture satisfies the lever, no action**
  - _kind_: trick-to-steal
  - _source_: WWDC20 10602 + 10603 HSR: draw opaque first, front-most-visible tracked before shading, submission-order independent, avoid write-masking; 10632 §7 [[early_fragment_test]] to keep HSR when writing buffers.
  - _ourCode_: src/nanite/NaniteRaster.ts:254-264 (depth-keyed atomicMax election → single winner) and src/nanite/NaniteResolve.ts (deferred pass shades the winner ONCE per pixel).
  - _mechanism_: Apple's HSR eliminates overdraw so each pixel shades once; our compute vis-buffer does its own HSR via atomic depth election and the deferred resolve shades exactly the winner once — structurally equivalent, and it works on Apple's IMR-fallback paths too. The hard constraint (keep opaque vis-buffer + deferred resolve) is already the recommended shape. [[early_fragment_test]] and programmable-blending HSR are Metal-only and moot for us because we don't use the HW raster depth path for shading.
  - _fix_: No change — flagged so the arc does NOT waste effort 'adding HSR'; we already have overdraw≈1 by construction. The only residual is the HW-triangle fallback path (NaniteRaster big-tri queue) which does use HW depth; ensure it writes all attachments / no write-masking if ever profiled.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident
- **memoryless / tile-memory residency for the vis-buffer + HZB + G-buffers is a real TBDR bandwidth win we CANNOT reach from WebGPU**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: WWDC20 10602 + 10632 §5: memoryless storage mode for intermediate attachments, tile-memory residency, programmable blending — no system-memory round-trip. WWDC20 10603 tile-memory limiter + bandwidth counter.
  - _ourCode_: N/A — architecture-level; our visPayloadV/visBV/visDepthV/HZB are full device StorageBuffers/Textures (NaniteRaster.ts:489-559, NaniteHzb.ts), never memoryless.
  - _mechanism_: On Apple TBDR, attachments a pass produces and consumes without persisting can live in on-chip tile memory (memoryless), skipping the system-memory store/load. Our vis-buffer is written by the raster compute pass and read by resolve — a classic candidate. BUT memoryless/tile-shaders/imageblocks are Metal-only and Dawn manages residency internally; WGSL has no surface to declare a storage buffer memoryless. So this potential win is BLOCKED by the WebGPU constraint. Consistent with the measured 'NOT bandwidth-bound' finding — so even if it were reachable, low priority.
  - _fix_: Blocked — no WGSL/Dawn path to declare memoryless or use tile shaders. Do not pursue. Note only so the arc doesn't chase a Metal-only bandwidth lever; our measured limiter is per-pixel work, not the vis-buffer store/load.
  - _webgpuFeasible_: no-metal-only
  - _impact_: low
  - _confidence_: confident
- **No runtime-indexed private stack arrays in the hot kernels — the classic spill trigger is already absent; residual spill risk is live-var COUNT**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: WWDC20 10632 §4 + Tech Talk 10580 lever #4: a stack array indexed by a non-compile-time value spills to memory; use unrollable loops / signed indices / vectorizable structs.
  - _ourCode_: kGrassRay (NaniteGrass.ts:870-1160) and kVoxScatter (NaniteVoxelRaster.ts:1349-1543) use scalar/vec locals only — no dynamically-indexed private arrays (grep found workgroupArrays only in NaniteRaster.ts SW-coop path lines 836-844 and NaniteVertexCache.ts, which are threadgroup memory, not private stack).
  - _mechanism_: The worst Apple spill cause (runtime-indexed private arrays) is NOT something we do in the two hottest kernels, so a targeted 'de-array' refactor has no target there. Any spilling would come from sheer live-var count (the 256-step march), which is the register-shrink/f16 lane, not the array lane. The NaniteRaster SW-coop workgroupArrays (rw0/exy/xy/wh/pay ×128 tris, lines 836-844) are threadgroup memory whose SIZE caps occupancy per 10603 — a separate tile-memory-limiter consideration, but that path (SWCOOP) was already measured dead in the RASTER arc.
  - _fix_: No de-array refactor needed for grass/voxel. For the SW-coop raster path, if ever re-profiled, the ~9 workgroupArrays sized MAX_CLUSTER_TRIS are the threadgroup-memory occupancy limiter to shrink — but that path is off/dead per the closed RASTER arc, so no action now.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident

---

## (agent)

_scope: Cutting the grass raymarch: steps, dependent fetches, per-blade alternatives_

- **The lane question (march vs per-blade-generate) is decided by OUR triangle-emit bottleneck, not the sources' console/desktop bottleneck — and BOTH lanes were already tried**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: GoT GDC2021 (cost ∝ visible blades, per-blade generate + indirect draw) + Babylon (screen-constant cost). Both argue per-pixel march is the wrong mobile lane.
  - _ourCode_: N/A — architecture-level; geo/per-blade lane existed in NaniteGrass and was DELETED 70e4d71 (resurrect 102669c); base raster is TRIANGLE-EMIT-bound per memory base-raster-is-the-bottleneck
  - _mechanism_: The sources' verdict is grounded in THEIR bottleneck (console forward+MSAA, desktop pixel-bound) which is not ours. Our base raster is triangle-EMIT-bound (46M sub-pixel tris, 97% <1px). The lushness law demands ~90 slots/m² × 5-8 fibers to 155m — at range these are massively sub-pixel, so a GoT-style generate pours millions MORE sub-pixel tris + per-pixel blade OVERDRAW into the exact bottleneck we already choke on, plus dense-sward overdraw (each pixel resolves 5-20 overlapping thin blades). The per-pixel march sidesteps triangle-emit entirely (zero tris) and resolves one hit/pixel (zero overdraw) — a structural fit for OUR emit-bound TBDR path. Our own ledger measured the geo lane at +13ms (the cited hwnoemit +1.4ms was the 0-40m near band ONLY, big blades, low count — NOT the full 155m field, so it is not evidence the full generate lane is cheap).
  - _fix_: Do NOT switch lanes on the strength of the sources — the deciding evidence (triangle-emit bottleneck + measured +13ms geo lane) is already in our ledger and cuts the other way for our architecture. Keep the march; attack its per-step dependent-fetch tax (findings below). A forward-SHADE grass (bypass the deferred resolve like water W1) is the one untested variant but saves only the ~1.8ms producer-independent resolve tail, not the dominant march cost, and does not fix overdraw — low ROI for a large rewrite.
  - _webgpuFeasible_: partial
  - _impact_: high
  - _confidence_: confident
- **Per-step baked-tile 3D fetch is Cloudscapes' flagged cache-miss tax; the DELETED closed-form analytic blade (Babylon's exact recipe) avoids it and measured march-only 2.36ms**
  - _kind_: trick-to-steal
  - _source_: Babylon (analytic ray-vs-surface per cell, ZERO dependent texture fetches, params from a hash of the cell origin) + Cloudscapes (jitter's +5ms was purely texture-cache misses — the one downside they flag)
  - _ourCode_: src/nanite/NaniteGrass.ts:1039-1047 fetchBand→texture3D, called :1135; the closed-form blade lived pre-70e4d71 (git 102669c), ledger: 'THE BLADE TEST IS CLOSED-FORM P=A+B·by+C·by²+E·u·W ... march-only 2.36 ms'
  - _mechanism_: On Apple/TBDR the per-in-sward-step trilinear repeat-wrapped 3D-texture fetch (plus 2 field taps + ctx + mask loads) is dependent-fetch LATENCY the GPU must hide with in-flight waves — exactly Cloudscapes' cache-miss penalty, worst on the dense-foliage trigger where every pixel is in-sward. Babylon's answer is the OPPOSITE of a baked tile: a closed-form ray-vs-surface intersection derived from a cell-origin hash — pure ALU, zero fetch. We HAD precisely this (the quadratic blade) and it measured march-only 2.36ms vs the baked-tile full look +14.5.
  - _fix_: Revive the analytic-blade march (resurrect from git 102669c) as the PERF lane. The arc-overhang lushness gap that motivated replacing it with the baked tile was separately solved by the (also-deleted) statistical coverage band — pair analytic-near with statistical-mid rather than the baked tile. Gate on the same grass8-frame gpuWall + lushness shotdiff.
  - _webgpuFeasible_: yes
  - _impact_: high
  - _confidence_: confident
- **The field-bake-to-texture traded ALU for 2 dependent taps/step — 'occupancy-bound' is ASSERTED not tested; the digest says do not assume occupancy collapse**
  - _kind_: profiling-todo
  - _source_: Cloudscapes (cache-miss tax) + digest caveat: Volkov shows 25% occupancy can suffice via ILP; M5 gap was scalar-ISA compute-bound, not latency — TEST occupancy, don't assume it
  - _ourCode_: src/nanite/NaniteGrass.ts:957-958 (guideFieldT1/T2 taps per step); ledger claim 'kernel was OCCUPANCY-bound (vox-cliff), 47.6→15.1 from shrinking live state'
  - _mechanism_: The bake moved 12 hashes+trig off the per-step path (won ~12ms) but ADDED 2 rgba16f dependent taps per in-sward step. If kRay is actually dependent-fetch-LATENCY-bound (not register-occupancy as the ledger assumes), adding taps on TBDR at dpr2 pixel counts is net-negative. The 47.6→15.1 drop from deleting a 15-live-register analytic graph is consistent with occupancy BUT equally with just deleting expensive ALU — the two are not distinguished. This directly governs whether finding 2 (analytic, more registers/ALU, zero fetch) or the baked tile (fewer registers, more fetch) is the right mobile trade.
  - _fix_: Profile to distinguish: webgpu-inspector capture of kRay at the dense-foliage pose, plus a register-pressure probe (artificially inflate live state and watch for a discontinuous cliff = occupancy-bound; smooth degradation = ALU/latency-bound). Decide analytic-vs-baked from the result instead of the assertion.
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: speculative
- **Layer-2 golden-angle overlay is a SECOND full baked-tile fetch chain per qualifying step — the fattest per-step add, and a LOOK feature not a perf one**
  - _kind_: violation
  - _source_: Cloudscapes / Babylon (minimize per-step fetches) — L2 doubles the inner-loop fetch on down-rays
  - _ourCode_: src/nanite/NaniteGrass.ts:1335-1499 (RAY_LAYER2 block), second fetchBand :1378, second maskBitAt :1433
  - _mechanism_: Gated to downward rays + 3/px budget, but each fire duplicates the entire L1 chain (tile fetch + neighbor-mask fetch + hashes + ground/top math). Down-look poses (aerial/oblique) are the dense-foliage trigger, so this concentrates cost exactly where the M1 already struggles. It exists only to fill top-down holes (a look call), not for perf.
  - _fix_: Measure ?grasslayers=1 (L2 off) at the down-look poses on grass8-frame. If the hole-fill can be reproduced by the statistical coverage band or a single density-bump tap instead of a second full march, drop L2. At minimum tighten the 3/px budget.
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: confident
- **Empty-space skip + cell-DDA are ALREADY implemented — Babylon's corridor and Cloudscapes' adaptive step are NOT violations we commit; do not re-chase them**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: Babylon (corridor = bound cells crossed, not fixed steps) + Cloudscapes prior-work #1 (adaptive step / empty-space skip)
  - _ourCode_: src/nanite/NaniteGrass.ts:914-919 (mask==0 terrain-Lipschitz jump over open air), :896-909 (texel DDA + texel-exit t), :1291-1332 (reject → advance to CELL exit, not fixed 0.03m)
  - _mechanism_: The march already DDAs texel→texel and cell→cell (Babylon's corridor) and skips empty/above-sward texels with a conservative-shell Lipschitz jump (Cloudscapes' adaptive step). Both 'obvious' step-count levers are already spent — the residual cost is per-IN-SWARD-step dependent FETCHES (findings 2-4), not wasted empty steps.
  - _fix_: None — verified present. Do not re-attempt corridor/empty-skip; redirect effort to the fetch cost.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident
- **Analytical transmittance integration (Cloudscapes' core big-step enabler) is N/A — grass is opaque hit-finding, not a volume integral; jitter+TAA start-offset also does not apply**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: Cloudscapes §3.1 (analytic transmittance removes step-length dependence → big steps) + §3.2 (per-frame jittered start + TAA for 16× step cut)
  - _ourCode_: src/nanite/NaniteGrass.ts:837-1256 — march finds FIRST OPAQUE HIT (tBest), no transmittance accumulation, no fixed-N sampling
  - _mechanism_: Cloudscapes' 16× win removes a step-length-dependent brightness artifact in a TRANSMITTANCE integral and uses jitter+TAA to recover detail lost by few fixed samples. Our march has neither: it is a to-first-hit DDA on OPAQUE blades (election, first hit wins), so there is no step-length brightness artifact to fix, no free big-step to gain, and no fixed sample count for start-jitter to trade against. Porting either would add cost (jitter = cache thrash) for no structural benefit.
  - _fix_: None — do not port analytic integration or start-jitter to kRay. (TAA is already on for the vis-buffer; the Q-quad decouple below is the applicable 'fewer-samples' analog.)
  - _webgpuFeasible_: no-metal-only
  - _impact_: low
  - _confidence_: confident
- **Q-quad march (1 ray / 2×2 px at dpr≥1.75) IS Babylon's screen-constant-cost lever, already applied — extend to distance-adaptive Q in the far band**
  - _kind_: trick-to-steal
  - _source_: Babylon (cost exclusively per-pixel-weight × pixels covered)
  - _ourCode_: src/nanite/NaniteGrass.ts:781-784 (Q auto=2 at dpr≥1.75, Wq/Hq), :1512-1536 (per-pixel emit fan — silhouettes stay crisp)
  - _mechanism_: Q=2 already quarters ray count at retina and is the single biggest applied perf lever. The FAR band (>~60m) is sub-pixel and already near-splat in look, so it can march at Q=4 with no visible loss, cutting far-band rays another 4×; near band stays Q=1/2 for crisp silhouettes. The emit fan is already per-pixel so nearer true geometry still wins each pixel individually — silhouettes don't degrade with coarser Q.
  - _fix_: Make Q a function of the pixel's election-depth bound (tMax): near→1-2, far→4. The march is already depth-bounded so the plumbing exists; only the dispatch tiling and the fan loop bounds change.
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: speculative
- **GoT's cheap-look tricks (texture-driven density, rounded-normal-from-flat, clumping) are ALL already implemented — not levers**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: GoT GDC2021 (compute place = single texture read for type/density; pixel = rounded normal from flat blade; Voronoi clump facing/color)
  - _ourCode_: src/nanite/NaniteGrass.ts:457-497 (density law baked to occupancy bits from biome/fields/canopy textures), :203-213 (bladeTable precomputed rounded mean normal nm), swirl/hash clump variety
  - _mechanism_: We already place from a texture-authored density law (baked to 64-bit occupancy per texel — GoT's 'read texture, drop empty lanes') and shade with a precomputed rounded mean normal (GoT's flat-blade-shades-round trick). Clump-coherent variety comes from the world-cell hash. The GoT items that would be new levers are all spent.
  - _fix_: None — verified present. The only GoT item NOT ported is per-blade vertex-LOD merging (4→1), which is inapplicable to a per-pixel march (no vertices); the march's thin×widen + splat-beyond-155m is its equivalent.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident

---

## (agent)

_scope: WebGPU/WGSL→Metal gotchas + per-pass profiling on this machine (three.js #32735 duplicate pipelines, Aaltonen SSBO tax, WGSL→MSL compile cost, timestamp-query recipe)_

- **Per-pass GPU timing is ALREADY wired — read these exact labels to attribute grass vs raster vs resolve on THIS M1**
  - _kind_: profiling-todo
  - _source_: threejsroadmap 'Profiling WebGPU Shaders': WebGPURenderer({trackTimestamp:true}) + resolveTimestampsAsync(TimestampQuery.COMPUTE|RENDER), warmup+median+p10/p90.
  - _ourCode_: src/core/Engine.ts:86 (trackTimestamp:true), :234-235 (resolves RENDER+COMPUTE every frame); src/core/GpuProfiler.ts:76-118 (labels each pass by ComputeNode.setName → stats.gpuPasses); src/core/MeasureHarness.ts:183-207 (drain→resolve→collect with GC-stall rejection)
  - _mechanism_: The blog's recipe is the harness we already run. Per-kernel setName gives labelled compute durations, so the Apple cost is decomposable WITHOUT new code: c.grassRay (NaniteGrass.ts:1540) + c.grassGuide (:616) + c.grassLight (:680) = the grass raymarch lane (the measured +14.5ms killer); c.nanRasterWorld1 = the SW per-covered-pixel raster loop; the deferred resolve is a fullscreen RENDER mesh folded into the scene pass (r.screen) so it has no own label — isolate it with the existing ?nores=1 ablation (NaniteFrame.ts:399, baseline−nores = the two resolve passes).
  - _fix_: No build needed. To get the per-pass map on this machine: run MeasureHarness (it already medians samples and rejects GC-inflated frames), or read window.__laas.stats.gpuPasses live. To split grass sub-lanes vs raster vs resolve at a fixed pose, A/B with ?grass=0 (grass off) and ?nores=1 (resolve off). This directly answers the task's 4th question and must run BEFORE acting on any SSBO/pipeline finding below.
  - _webgpuFeasible_: yes
  - _impact_: high
  - _confidence_: confident
- **Shadow clipmap compiles 6 byte-identical SW-raster pipelines (the #32735 duplicate-pipeline pattern, verbatim)**
  - _kind_: violation
  - _source_: three.js #32735: compute pipeline cache keys on ComputeNode instance id, so N nodes sharing identical WGSL compile N pipelines. + wgpu #4456: each WGSL→MSL compile is a real, sometimes-hanging cost at pipeline-creation on Apple/Metal (XPC compiler).
  - _ourCode_: src/nanite/NaniteShadowClip.ts:495-497 — for k in LEVELS(=6, readClipParams :211) call buildNaniteRaster(...) each at makeNaniteCam(SHADOW_MAP=1024, 1024) (:314) with identical flags (shade=false, same disp/wind)
  - _mechanism_: buildNaniteRaster bakes only cam.width/height + build-time flags into WGSL; the per-level camera differs ONLY in uniform values (lv.cam is uniform-backed). All 6 levels therefore emit byte-identical world1/depth/clear/hw-stage WGSL but are 6 distinct ComputeNode ids → 6 separate WGSL→MSL compiles at boot. On Apple the Metal shader compiler runs per distinct pipeline via XPC (wgpu #4456), so this is a 6× multiplier on the shadow raster's boot compile — a contributor to the boot-freeze class the BOOT arc chipped at.
  - _fix_: Build the raster kernel set ONCE and dispatch it per level, swapping the per-level cam/strip/depth UNIFORMS + bind group between dispatches (all already uniform-backed; nothing per-level is baked into the WGSL). three's cache keys on node id, so reuse the SAME node with needsUpdate/bind-group swap rather than 6 fresh builders. Pure boot-time win, runtime unchanged (pipelines are cached after first dispatch). Gate: boot pipeline-creation time on M1, not frame time.
  - _webgpuFeasible_: partial
  - _impact_: medium
  - _confidence_: confident
- **HZB builds up to 16 distinct per-level pipelines; ?pyrfuse collapses them but was judged on runtime, never on Apple boot-compile count**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: wgpu #4456 (per-pipeline MSL compile cost on Apple) + Chrome-130 note (Tint IR up to 10× faster WGSL→MSL) — boot cost scales with distinct-pipeline count. three.js #32735 dedup does NOT help here (WGSL differs per level).
  - _ourCode_: src/nanite/NaniteHzb.ts:139-188 (perLevelCount up to MAX_LEVELS=16 kernels, each with baked info.w/h/offset ⇒ non-identical WGSL); :133-137,189-227 (?pyrfuse fuses the ≤1024-texel tail into ONE static-unrolled kernel, DEFAULT OFF)
  - _mechanism_: Each HZB level bakes its own dims/offset, so the 11-16 kernels are genuinely distinct pipelines (not #32735-dedupable). pyrfuse already exists and reduces this to ~2 pipelines (k=0 + fused tail) via storageBarrier. It's default-off because it measured runtime-NEUTRAL on desktop — but the axis that matters on Apple is pipeline-CREATION time (fewer distinct MSL compiles at boot), which was never the evaluation metric. The camera has 1 HZB; more instances multiply the count.
  - _fix_: Re-benchmark ?pyrfuse=1 on M1 for BOOT pipeline-creation time (not frame time) using the timestamp/boot timers. If it cuts boot compile without a frame-time regression on Apple, flip it default-on for the mobile preset. Feasible today (the fused kernel is already built).
  - _webgpuFeasible_: partial
  - _impact_: low
  - _confidence_: speculative
- **Grass march's hot loop reads guideMask as an SSBO per texel — Aaltonen's 'SSBOs are slow' lands on the measured #1 Apple cost**
  - _kind_: trick-to-steal
  - _source_: Aaltonen REAC2023 mobile-tax slide: 'Mobile: 16KB uniform buffers! SSBOs are slow!' — random storage-buffer access is the mobile/TBDR slow path vs the texture cache.
  - _ourCode_: src/nanite/NaniteGrass.ts:389-393 (guideCtxAttr + guideMaskAttr = StorageBufferAttribute), :156-157 (comment: 'Storage buffers, not textures — uint StorageTexture is mistyped float by the node builder'); the field ctx was ALREADY moved to filterable StorageTextures (mkFieldTex :411), leaving guideMask as the remaining SSBO random-read in the per-pixel raycast (kGrassRay :1540)
  - _mechanism_: kGrassRay walks guide texels and reads guideMask (64-bit occupancy) per step as a dependent SSBO fetch — precisely the dependent-fetch pattern measured as grass +14.5ms gpuWall, and precisely the access Aaltonen names as slow on TBDR. The team already migrated the rgba fields to textures for HW bilinear; the mask is the leftover SSBO because three's node builder mistypes a uint StorageTexture as float. On Apple the texture cache path can beat a device-memory SSBO for this random read.
  - _fix_: Pack guideMask into an rgba32uint (or two rg32uint) texture and read it via textureLoad, bitcasting around the node-builder's float-typing (the same mistype the comment flags is the blocker, not a hardware limit). Then the per-step occupancy fetch hits the texture cache. MEASURE FIRST via finding #1: confirm guideMask fetch is a real share of c.grassRay before converting; A/B mask-as-texture vs mask-as-SSBO on M1.
  - _webgpuFeasible_: partial
  - _impact_: medium
  - _confidence_: speculative
- **The vis buffer SSBO tax is architectural and immovable — do NOT chase 'switch to UBOs'; the actionable SSBO surface is narrow**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: Aaltonen: '16KB uniform buffers! SSBOs are slow!' — but the tax applies to STORAGE traffic that could be UBO/small; the vis-buffer election cannot be either.
  - _ourCode_: src/nanite/NaniteRaster.ts:203-218 (makeVisBuffers: 3× pixelCount u32 StorageBufferAttribute), :599-607 (per-covered-pixel atomicMax election into visPayloadV); small hot read-only data already goes through three uniform nodes (UBOs): cam matrices/level tables via uniformF/uniformArrV4/uniformMat4 (NaniteHzb.ts:113-119, NaniteCull.ts uniforms)
  - _mechanism_: The 3.3M-entry atomic vis buffer is load-bearing under the opaque-vis-buffer / deferred-resolve constraint — it CANNOT become a ≤16KB UBO or a memoryless tile buffer (WebGPU exposes neither imageblocks nor programmable blending). So the biggest SSBO on Apple is immovable. Meanwhile the small hot uniforms already ARE UBOs, so the 'move small data to UBO' half of the advice is already satisfied. This bounds the SSBO-tax opportunity to grass guideMask (finding #4) and possibly qRaster read locality — it stops us wasting the arc chasing a UBO rewrite that the architecture forbids.
  - _fix_: No change to the vis buffer (blocked by the WebGPU no-imageblock/no-memoryless constraint). Confirm via the profiler that small-uniform reads are NOT a hotspot (expected: they're UBOs already). Direct SSBO-tax effort only at guideMask and at qRaster access coherence, not at the vis buffer.
  - _webgpuFeasible_: no-metal-only
  - _impact_: low
  - _confidence_: confident
- **K near-identical kVoxBucketArgs 1-thread kernels are #32735-shaped, but low impact at the default (K=2, lazy-compiled)**
  - _kind_: violation
  - _source_: three.js #32735: N ComputeNodes sharing identical logic → N pipelines.
  - _ourCode_: src/nanite/NaniteCull.ts:831-838 — for b in K build a distinct 1-thread kVoxBucketArgs node differing only by baked bucket index; K default 2 (voxPrev default-on, :431-433) but up to 32 under ?voxf2bk
  - _mechanism_: These K args kernels are near-identical WGSL (only the baked bucket index/attr differs) → K distinct pipelines, the #32735 shape. But WebGPU pipelines compile lazily on first dispatch, so unused ones cost nothing; at the shipped default (voxPrev on ⇒ K=2) only 2 tiny 1-thread pipelines compile. It only bloats to 16-32 compiles under explicit ?voxf2b=1/?voxf2bk. Tiny kernels either way.
  - _fix_: Fold the K args into one kernel over a bucket-index uniform/loop (K is small; a 1-thread K-iteration loop writing K indirect slots via one bound array). Low priority — pursue only if a future large-K voxel retune ships. Confirm with the profiler that c.nanVoxBucketArgs is negligible (expected).
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident
- **Confirm the running Chrome has Tint IR on macOS (up to 10× faster WGSL→MSL) — free mitigation of the boot-compile findings**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: Chrome 130 blog: 'the new version of Tint is up to 10 times faster translating Unity's WGSL to MSL', progressive rollout to macOS Metal. Mitigates wgpu #4456 compile cost.
  - _ourCode_: N/A — architecture-level (affects boot pipeline-creation of NaniteShadowClip.ts:495 rasters, NaniteHzb.ts:139 levels, all buildNaniteRaster/Fn kernels)
  - _mechanism_: Every distinct pipeline (findings #2/#3/#6) pays one WGSL→MSL compile at creation via Apple's XPC compiler (wgpu #4456). Tint's IR backend cuts that per-compile cost up to 10× on macOS Metal. If the dev/test Chrome predates the macOS rollout, boot compile is being measured on the slow translator and the duplicate-pipeline findings look worse than they will ship.
  - _fix_: Check chrome://gpu (Tint IR / Dawn version) and navigator.userAgent ≥ Chrome 130 on the M1 test machine; ensure the profiling/boot baseline is taken on a current Chrome so the pipeline-dedup wins are measured against the fast translator. No code change.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident

---

## (agent)

_scope: Compute vis-buffer raster / atomic depth election on mobile+Apple: OUR election is a WebGPU-forced 32-bit split (24-bit NDC-z depth-key + 8-bit tiebreak via atomicMax into visPayloadV, guarded by a relaxed atomicLoad) plus a winner-gated full-id side buffer (visBV) — NOT Nanite's 64-bit atomicMax (unavailable in WebGPU). Cross-checked against all 8 sources: the election is already well-engineered against this theme's best practices (guard-load contention mitigation, per-cluster SW/HW split, avoidance of the measured '3rd atomic buffer = 3× cliff', single-pass resolve that beats Nanite's per-material depth-EQUALS cull). Premise-audit outcome: the SW pixel loop (7.8ms eye / 18.4ms oblique) is REAL but is A+B·pixels COVERAGE-bound (R²≈1), not atomic-contention-bound — so the theme's contention/packing levers are mostly already-implemented or measured-dead (trihzb −0.5ms, swmax dead, cooperative-bin negative). The one high-value CONFIRM is Tellusim's M1 data (compute 2.30B > HW 1.37B tri/s) validating our all-compute choice against the Mali 'avoid compute' warning. The only remaining real lever is architectural: reduce covered fragments in the near-leaf band (voxelize nearer / non-growing coarsening DAG), which is task #72, not an election-packing change._

- **No atomic<u64>: our election is a 32-bit split (24b depth-key + 8b tiebreak via atomicMax) + a side buffer holding the full 25-bit id**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: Karis SIGGRAPH 2021 + GDC 2024 + elopezr: Nanite's whole SW raster depends on ONE 64-bit InterlockedMax packing 30b depth / 27b cluster / 7b tri (2024: 32/32). WebGPU has no atomic<u64> (Tellusim: '64-bit atomics not available on Mobile and Metal').
  - _ourCode_: src/nanite/NaniteRaster.ts:1262-1272 (cand = depthKey24(cz)<<8 | payload&0xff, then doElection); doElection at 589-608; depthKey24 at 442-443
  - _mechanism_: We cannot do Nanite's single-op depth-test+payload-write. Our world1 packs 24-bit depth + 8-bit id-tiebreak into the 32-bit election word (visPayloadV, atomicMax) and the election WINNER plain-stores the full id into a second buffer (visBV). This is the correct WebGPU realization; it costs one extra gated store vs Nanite's single atomic, but avoids Tellusim's 'redundant triangle re-intersection' path (we keep the id, not re-raster it in the resolve).
  - _fix_: No change — this is the right port of the 64-bit scheme under the WebGPU constraint. The 64-bit fast path is Metal/Vulkan-only (VK_KHR_shader_image_atomic_int64); not exposed in WebGPU.
  - _webgpuFeasible_: no-metal-only
  - _impact_: medium
  - _confidence_: confident
- **Depth precision is 24-bit NDC-z (hyperbolic → fine near), which refutes the Scthe u16 z-fight/leak risk for our trigger case**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: Scthe nanite-webgpu issue #1: 16-bit packed depth risks z-fighting/leaks. Karis: Nanite uses 30 depth bits.
  - _ourCode_: src/nanite/NaniteRaster.ts:442-443 (depthKey24) and cz = barycentric ndc.z at 1206-1209
  - _mechanism_: cz is the interpolated NDC-z (perspective-hyperbolic), so `1-cz` concentrates the 24-bit key's precision NEAR the camera — exactly where dense-foliage overlap (the M1 trigger) lives. 24 bits is 256× finer than a u16 pack; the resolve reconstructs depth from key>>8 and the HZB from key>>16, both sub-pixel. Far foliage (where 24-bit NDC-z goes coarse) is routed to voxels/impostors anyway, so the coarse tail never elects overlapping triangles.
  - _fix_: None needed. If a far-field leaf-vs-leaf z-tie speckle ever shows, the cheapest mitigation is already the architecture (voxelize nearer), not more depth bits.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident
- **The load-bearing packing constraint is measured: a 3rd hot-loop atomic storage buffer is a 3× cliff — we already avoid it by reconstructing depth from the election key**
  - _kind_: trick-to-steal
  - _source_: Tellusim: 32-bit payload forces either fewer depth bits or a redundant intersection. Burns-Hunt: minimize per-sample footprint to stay cache/bandwidth-friendly on integrated/mobile.
  - _ourCode_: src/nanite/NaniteRaster.ts:1266-1270 ('NO depthV write: a 3rd atomic storage buffer in this kernel is the 3× cliff (15-17 ms)'); world1 writes only visPayloadV(+visBV) per fragment, exact depthV only at the kernel-end sink (925)
  - _mechanism_: On the M1's unified memory, each additional per-fragment atomic target multiplies device-memory RMW traffic under dense overlap. We measured that adding an exact per-fragment depthV (the naive '3 buffers: depth, payload, id') triples the world1 cost. The fix already shipped: keep only the 24-bit-depth election word and let the resolve/HZB decode depth from it — 2 atomics/fragment worst case, not 3.
  - _fix_: Already implemented — this is the single most important election decision for Apple and it is correct. Flagging so it is NOT regressed: any future feature that wants exact per-fragment depth in world1 must fold it into the existing election word, never add a 3rd atomic buffer.
  - _webgpuFeasible_: yes
  - _impact_: high
  - _confidence_: confident
- **Relaxed atomicLoad guard before the atomicMax RMW is already implemented — but the pixel loop is coverage-bound, not atomic-contention-bound, so contention is not the lever**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: Tellusim/general SW-raster practice: guard the atomic to skip the RMW for losing fragments. Karis: overdraw is 'atomic-bound for large tris, coverage-bound for medium, setup-bound for small'.
  - _ourCode_: src/nanite/NaniteRaster.ts:601-607 (prevE=atomicLoad; If(cand>prevE){atomicMax; If(cand>wonE) atomicStore}); the ?relect ablation harness at 374-394 / 588-608 measures the election's share
  - _mechanism_: Our tris are slivers (few px each) → Karis's 'medium/coverage-bound' and 'small/setup-bound' regimes, NOT the 'large/atomic-bound' regime. SUBSYSTEM-COSTS confirms frame ≈ A + B·pixels, R²≈1, i.e. the loop scales with COVERED FRAGMENTS (coverage walk + depth interp), and the ?relect ablation showed the election RMW is a minor share. So atomic contention is already mitigated (guard) AND is not the dominant term — chasing it further is dead.
  - _fix_: No change. Do NOT invest in fancier contention schemes (per-tile locks, wave-election); the coverage walk, not the atomic, is B·pixels. The real reducer is fewer covered fragments (overdraw), i.e. architecture-level LOD/voxelization.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident
- **Compute-raster is FASTER than HW on Apple (Tellusim M1: 2.30B vs 1.37B tri/s) — the Mali 'compute shading not recommended' warning does NOT transfer to our target**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: Tellusim compute-raster table: Apple M1 compute 2.30B tri/s vs single-DIP HW 1.37B (A14: 1.02B vs 666M). Arm Mali blog: 'compute shading remains challenging and is not recommended for most situations' (disables framebuffer compression, reduces vertex/fragment overlap).
  - _ourCode_: src/nanite/NaniteRaster.ts (entire world1 SW compute path) + ?swmax split at 106-112
  - _mechanism_: The digest's own caveat says Mali register/architecture claims do not transfer 1:1 to Apple. Tellusim directly measured the opposite on M1/A14: compute rasterization BEATS hardware. Apple's TBDR + strong compute + unified memory means our all-compute vis-buffer is the right call on the exact target GPU — the Mali concern (loss of FB compression / VS-FS overlap) is a Mali-specific structural cost, not an Apple one. This de-risks the whole architecture for the mobile arc.
  - _fix_: None — this validates staying all-compute. If we ever port to Android/Mali specifically, re-evaluate; for M1 Max the choice is measured-correct.
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: confident
- **Per-triangle occlusion cull (the one culling lever the theme surfaces beyond cluster-HZB) is BUILT and measured WEAK on our porous canopy — matches Karis's own 'not worth it'**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: Karis: no per-triangle occlusion cull (re-raster would nullify savings); SW HZB culls clusters not pixels. themaister: two-phase HiZ. Mali: cluster culling is the big win on tilers.
  - _ourCode_: src/nanite/NaniteRaster.ts:283-304 (?trihzb per-tri prev-HZB reject, default off); verdict docs/perf-runs/SUBSYSTEM-COSTS-2026-07-04.md:68 (−13% fragments, only −0.5 ms) and docs/perf-runs/2026-07-04-90fps-arc.md:99-110
  - _mechanism_: On a porous leaf canopy the prev-frame HZB far-window almost always contains a gap, so per-tri occlusion rejects only 13% of fragments for 0.5 ms — Karis predicted exactly this. Our cluster-level prev-frame HZB cull (the Mali 'significant portion of the gain') is the real occlusion reducer and is already in the cull stage.
  - _fix_: Leave ?trihzb opt-in (default off, quality-unvetted stale-HZB disocclusion). Do not promote it; it is a measured 0.5 ms with a disocclusion-hole risk. This lever is CLOSED.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident
- **Back-face cull is done; themaister micro-poly bbox-collapse rejection would NOT transfer, because sub-pixel tris in a compute SW raster already write nothing and setup is measured free**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: themaister: back-face (window-space cross) + micro-poly rejection (reject tri whose bbox floors to one pixel) gives 3.5-10× in dense scenes. But that win is in a MESH/HW path (avoid emitting to the HW rasterizer's quad/primitive setup).
  - _ourCode_: src/nanite/NaniteRaster.ts:134-139 + 949-956 (orientForRaster: back-face accept areaNdc>0 / two-sided handling); no explicit micro-poly bbox-collapse early reject in the SW loop; setup cost rdbg2−rdbg1≈0 per SUBSYSTEM-COSTS notes
  - _mechanism_: themaister's 3.5-10× comes from stopping micro-tris BEFORE the fixed-function rasterizer spins up 2×2 quads + primitive setup. Our SW path has no quad overhead: a sub-pixel triangle whose bbox covers no sample center simply iterates an empty bbox and writes nothing, and per-triangle setup was measured ~free (the pixel loop is 90% of world1). So a micro-poly pre-reject would only trim already-near-zero setup — same weak class as trihzb. Big tris (where quad overhead WOULD matter) are already routed to the HW pipeline by ?swmax, where they are not micro.
  - _fix_: Not worth adding a micro-poly reject to the SW path. Back-face is already done. If profiling ever shows setup > pixel-loop at some pose (it does not today), revisit; blocked-by-measurement, not by WebGPU.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: speculative
- **SW/HW size split (Mali MinPixelsPerEdgeHW ↔ our ?swmax) is implemented and already swept dead — but the Mali-optimal 32px vs our default 16px is a documented Apple re-sweep candidate**
  - _kind_: profiling-todo
  - _source_: Mali blog: MinPixelsPerEdgeHW routing table shows 32px optimal on Mali (24.17/22.72/26.26 ms vs 16px). Karis: SW/HW chosen per-cluster by which is faster; edges <32px SW.
  - _ourCode_: src/nanite/NaniteRaster.ts:106-112 (?swmax, default bbox-extent ≤16px stays SW); noted measured-dead in docs/perf-runs/2026-07-04-mem-boot-raster-arc.md:101 ('swmax ALL measured dead, bit-identity-gated')
  - _mechanism_: The HW/SW split is the primary rasterizer-pressure routing lever on tilers, and Apple's crossover (where HW quad overhead beats SW coverage walk) is a hardware constant that need not equal our 16px default. Our own sweep found it dead, but Mali data shows the optimum is content- and GPU-specific (they land on 32). Since the SW pixel loop is our #2 cost (7.8/18.4 ms), even a small routing shift toward HW for the 8-16px band could matter on M1 Max specifically.
  - _fix_: Re-sweep ?swmax ∈ {8,12,16,24,32} on M1 Max at the dense oblique pose (18.4 ms) with the bit-identity gate. If the prior 'dead' verdict was measured only at eye pose, the oblique/dense pose is the one that would move. Low expected value (prior sweep dead) but cheap and directly Apple-targeted.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: speculative
- **Dense-overlap overdraw is the confirmed structural Nanite weak spot (leaves/grass) — our real response is architectural routing, and it is where the only remaining lever lives**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: Karis: 'riddled with holes... most aggregate geometry like leaves and grass. Overdraw is one of many reasons Nanite doesn't perform well with those'; overdraw is coverage/atomic-bound with no per-pixel occlusion cull. Burns-Hunt: vis-buffer wins scale with sample count but the visibility PASS still pays overdraw.
  - _ourCode_: Architecture-level: grass moved OFF the triangle election to kGrassRay (per-pixel raymarch lane), far foliage to kVoxScatter (NaniteVoxelRaster.ts:383-391 election VERBATIM world1), near-leaf clusters still hit world1 election (NaniteRaster.ts:1262-1272); loderr/τ coarsening verdict docs/perf-runs/SUBSYSTEM-COSTS-2026-07-04.md
  - _mechanism_: The theme confirms our pain is the KNOWN weak case, and the pixel loop being coverage-bound (A+B·pixels) means the only lever that moves it is FEWER covered fragments = less overdraw. We already route grass and far foliage away from the election; the residual is near-leaf-cluster overdraw. Measured coarsening levers are weak (loderr halving visTris = −0.7 ms; mid values WORSE because the aggregate DAG ladder grows geometry → more overdraw). So the honest conclusion: the election machinery is well-tuned; the remaining win is pulling the voxel transition nearer / a real coarsening DAG that reduces fragments without growing aggregate geometry — an architecture task (#72), not an election-packing task.
  - _fix_: Do not seek more election micro-optimizations. Direct effort at reducing covered fragments in the near-leaf band: pull transitionDist nearer (voxelize sooner) or fix the aggregate-DAG-grows-geometry pathology so coarsening actually cuts overdraw. This is the SCAR work already scoped (?scar overdraw counter at NaniteRaster.ts:412-429).
  - _webgpuFeasible_: yes
  - _impact_: high
  - _confidence_: confident
- **We already avoid Nanite's per-material fullscreen-quad + depth-EQUALS material cull — our single-pass deferred resolve shades the winner ONCE, strictly better on mobile bandwidth**
  - _kind_: confirm-or-refute-hypothesis
  - _source_: GDC 2024 + Karis + elopezr: Nanite emits GBuffer via one fullscreen quad PER MATERIAL, culled by a synthetic material-ID depth with depth-EQUALS test (2.08 ms 'Emit GBuffer' in the reference frame). Burns-Hunt: g-buffer bandwidth is the thing to avoid on mobile.
  - _ourCode_: Architecture: deferred RESOLVE is a single fullscreen pass over the vis-buffer winner (RP-4 single-pass resolve, ?respass); the election writes one winner id (visBV), resolve decodes id→cluster→material and shades once (NaniteResolve.ts)
  - _mechanism_: Nanite's depth-EQUALS material trick exists to make its MULTI-pass (per-material) GBuffer emit cheap. We don't have that multi-pass — we shade the winner exactly once in a single compute resolve, which is the cheaper design on a bandwidth-limited M1 (no material-ID depth buffer, no N fullscreen draws, no HTILE). The trick the theme surfaces is therefore N/A because we already skipped the problem it solves — an architecture-aligned 'we're ahead' point, not a lever to steal.
  - _fix_: None. Keep the single-pass resolve. Do NOT adopt per-material passes + material-ID depth culling — it would ADD passes and a buffer we don't need on mobile.
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: confident

---

## (agent)

`docs/mobile-gpu-perf/MASTER-AUDIT.md` written. All file:line anchors verified live against HEAD before inclusion; `CROSS-CHECK.md` folded in (its F2 ctx→texture and #2 toroidal-bake findings are levers 2 and 5).

## TOP LEVERS (ranked, for the +14.5 ms grass frame)

0. **INSTRUMENT FIRST** (CONFIDENT, HIGH) — native Metal GPU capture of the Chrome/Dawn process reading kGrassRay/kVoxScatter occupancy% + spilled bytes + top limiter; plus the already-wired timestamp map (`Engine.ts:86`, labels `grassRay`/`grassGuide`/`grassLight`, `?grass=0`/`?nores=1`/`?grasslayers=1`). Gates everything precision/register-related.
1. **Bake `widenT` into the dead 4th guide-ctx word** (CONFIDENT, MED, SHIP-NOW) — `NaniteGrass.ts:524` writes a dead `uint(0)`; march re-derives `1/sqrt(grassThin(distT))` (2 pow + sqrt + divides) every accepted step at `:926-927` though the bake already has it at `:501`. Free ALU + live-var cut, not gated on the capture.
2. **Guide CTX + MASK: SSBO → texture** (SPECULATIVE, HIGH-MED) — fields already moved to filterable `StorageTexture` and won; ctx (`:389/:892`) + 64-bit mask (`:395/:921/:1121`) are the leftover dynamic-address SSBO loads on the dependent path.
3. **Distance-adaptive far-band Q-quad** (SPECULATIVE, MED) — extend the shipped Q=2 (`:781`) to Q=4 in the sub-pixel far band via the election-depth bound.
4. **Hoist L2 golden-overlay out of `kRay`** into a sparse pass (SPECULATIVE, MED, register-gated) — `:1335-1499` inflates worst-case VGPRs for every lane though it fires rarely.
5. **Toroidal incremental guide bake** (SPECULATIVE, MED) — `kGuideBake` (`:615`) rebakes the whole 384² field/frame; strip-rebake like the shadow clipmap. Cuts `grassGuide`, not the march.
6. **fp16 the march-LOCAL state only** (SPECULATIVE, MED, high effort) — blocked at the TSL layer (no half node; needs `shader-f16` + raw-WGSL FunctionNode). On M1 this is an OCCUPANCY lever, not a FLOP lever.
7. **(USER-GATED, surfaced not silently taken)** revive the analytic closed-form blade lane — measured march-only 2.36 ms vs baked-tile +14.5 ms; conflicts with the user's march-lane directive + lushness law, so it's a user call.

## HYPOTHESIS VERDICT

**MIXED, dominated by dependent-fetch LATENCY that only occupancy can hide — NOT a FLOP/scalar-ISA (M5) problem, and NOT proven to be occupancy *collapse*.** The march is confirmed in code as an ILP≈1 serial dependent-fetch chain (`loopUN 256`, each step's fetch address depends on the prior `tCur` advance), so occupancy is the only latency-hider a serial march has. But the code only *asserts* occupancy-bound: the 47.6→15.1 ms "shrink live state" win is equally consistent with deleting expensive ALU, and nobody has read actual VGPR/occupancy/spill counters. Per the digest's caveats, on M1 Max fp16 is NOT double-rate (256:256 FMA) and our vecs are component-independent, so the M5 "scalarize float4" framing does not transfer. Not bandwidth, not thermal (both measured). **Single settling step:** a Metal GPU capture (Xcode Capture GPU Frame / Instruments Metal System Trace) of the Dawn GPU-process reading kGrassRay occupancy% + spilled bytes + the single top limiter — occupancy-healthy-but-slow ⇒ cut fetches/steps (levers 2–5); spilling / VGPR-above-a-step ⇒ shrink live state (levers 1, 4, 6).
