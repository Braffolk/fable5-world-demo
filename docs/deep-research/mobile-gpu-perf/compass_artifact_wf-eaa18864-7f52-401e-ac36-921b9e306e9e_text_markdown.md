# Deep Technical Sources for a WebGPU Compute-Rasterizer / Visibility-Buffer Renderer Bottlenecked on Apple Silicon & Mobile TBDR GPUs

## TL;DR
- The single closest real-world overlap is **Scthe's nanite-webgpu** — an actual UE5-Nanite-style compute software rasterizer + visibility buffer built in WebGPU/WGSL, whose author documents the exact WGSL limitation you face: *"WebGPU does not support atomic<u64>, so I had to compress the data to fit into 32 bits (u16 for depth, 2*u8 for octahedron-encoded normals)"* and that *"16-bit depth is.. not a great idea. It produces tons of artifacts like z-fighting or leaks."*
- The richest vein of "author was genuinely in this situation" material is Apple-GPU reverse-engineering (Rosenzweig/Asahi, Dougall Johnson, Philip Turner) + Apple's own WWDC TBDR/tile-shading talks + mobile-vendor register-pressure/occupancy guides (Arm Mali, Qualcomm Adreno, Imagination PowerVR).
- The "low occupancy can't hide dependent-fetch latency" hypothesis is directly challenged by Volkov's "Better Performance at Lower Occupancy" (which shows *"Mere 25% occupancy is sufficient"* to hide memory latency via ILP) and refined by Sebastian Aaltonen's compute-occupancy writeups; treat generic desktop-GPU occupancy advice as directional, because Apple's occupancy/register model differs.

## Key Findings
Below, every item has a direct URL, a one-line relevance note, and a HIGH/MEDIUM/LOW rating. Items are grouped by overlap area. The final group is deliberately tangential/high-risk.

---

### GROUP 1 — Compute software rasterizer / visibility buffer (the core architecture)

- https://github.com/Scthe/nanite-webgpu — **THE closest match**: UE5 Nanite reimplemented in WebGPU/WGSL, compute software rasterizer + meshlet LOD; README FAQ states verbatim *"WebGPU does not support atomic<u64>, so I had to compress the data to fit into 32 bits (u16 for depth, 2*u8 for octahedron-encoded normals)"* and warns *"16-bit depth is.. not a great idea. It produces tons of artifacts like z-fighting or leaks."* Author is Marcin Matuszczyk. **HIGH**
- https://scthe.github.io/nanite-webgpu/ — Live demo of the above (Chrome only); lets you toggle the software rasterizer and use the built-in "Profile" button (author warns raw FPS is unreliable due to browser VSync). **HIGH**
- https://github.com/Scthe/nanite-webgpu/issues/1 — Author answers a performance question and lays out the ideal 64-bit atomic depth-election scheme `(depth << 32) | sceneUniqueTriangleId` he could **not** use in WebGPU; notes it's a research project, not tuned for production. **HIGH**
- https://www.sctheblog.com/blog/nanite-report/ — Author's blog pointer post "My thoughts on Nanite after 'Nanite WebGPU'"; the substantive writeup is the GitHub README FAQ, which this links to. **MEDIUM**
- https://www.sctheblog.com/blog/hair-software-rasterize/ — Same author on software-rasterizing hair: sub-pixel primitive rasterization in compute, very close to your dense small-primitive foliage case. **HIGH**
- https://www.sctheblog.com/blog/nanite-materials-notes/ — Same author's notes on Nanite's GPU-driven materials / deferred visibility resolve. **MEDIUM**
- https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf — Brian Karis et al., "Nanite: A Deep Dive" slides; the canonical rationale for compute rasterization of small triangles + 64-bit atomicMax visibility buffer. **HIGH**
- https://advances.realtimerendering.com/s2021/ — SIGGRAPH 2021 Advances course index (Nanite video + related talks, incl. Ghost of Tsushima lighting). **HIGH**
- https://media.gdcvault.com/gdc2024/Slides/GDC+slide+presentations/Nanite+GPU+Driven+Materials.pdf — Nanite GPU-Driven Materials (GDC 2024): states verbatim *"We rasterize the geometry into a 64bit visibility buffer using atomic max (with 32bit depth in the high bits, and 32bit triangle and visible cluster ID in the low bits)"* and describes the deferred material resolve passes. **HIGH**
- https://www.elopezr.com/a-macro-view-of-nanite/ — Code Corsair RenderDoc teardown of Nanite's SW/HW raster split (the ~5M-triangle / ~90%-software-rasterized figures come from this profiling, not from Epic directly). **HIGH**
- https://jcgt.org/published/0002/02/04/paper.pdf — Burns & Hunt, "The Visibility Buffer: A Cache-Friendly Approach to Deferred Shading" (JCGT 2013), the origin technique; explicitly notes G-buffer bandwidth is prohibitive on mobile/integrated GPUs. **HIGH**
- http://diaryofagraphicsprogrammer.blogspot.com/2018/03/triangle-visibility-buffer.html — Wolfgang Engel, "Triangle Visibility Buffer" writeup. **MEDIUM**
- https://media.gdcvault.com/gdceurope2016/presentations/Engel_Wolfgang_Visibility_Buffer.pdf — Engel's GDC Europe 2016 filtered/culled visibility buffer slides. **MEDIUM**
- https://tellusim.com/compute-raster/ — Tellusim "Compute versus Hardware": cross-API/cross-platform compute rasterization benchmarks, explicitly notes 64-bit atomics unavailable on mobile and Metal and the depth-only workaround. **HIGH**
- https://themaister.net/blog/2024/01/17/modernizing-granites-mesh-rendering/ — Themaister (Hans-Kristian Arntzen) on a Nanite-style mesh/visibility renderer: FP32 winding tricks, primitive culling to reduce rasterizer pressure, atomics discussion. **HIGH**
- https://arxiv.org/pdf/2405.13364 — LucidRaster: GPU software rasterizer paper (OIT), useful for compute-raster design tradeoffs. **MEDIUM**

---

### GROUP 2 — Nanite / visibility buffer specifically on mobile / Apple / TBDR

- https://developer.arm.com/community/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/mali-and-unreal-engine-s-nanite-enabling-the-future-of-mobile-graphics — Arm's own writeup running Nanite on Mali TBR (using Arm's internal "Mori" UE demo); states *"64-bit atomics are required to resolve the visibility buffer"* and that `VK_KHR_shader_image_atomic_int64` has been supported by Mali drivers for some time; SW-vs-HW raster comparison on a tiler. **HIGH**
- https://github.com/philipturner/metal-benchmarks — Philip Turner's Apple GPU microarchitecture benchmarks: register file size, occupancy model, FP16-vs-FP32 latency, the crucial *"Apple would rather you spill to device memory than create chances to decrease ALU utilization"* finding (ALU utilization maxes at 24 simds/core). **HIGH**
- https://medium.com/@sarah.hyperdense/understanding-nanite-when-to-use-it-and-when-to-skip-it-b7fcfadc3058 — Practical Nanite platform-support notes incl. Steam Deck sub-20fps in Epic's Matrix City sample; useful for the mobile/handheld collapse framing. **MEDIUM**

---

### GROUP 3 — Apple GPU internals: Apple's own docs & WWDC (TBDR, tile memory, imageblocks, occupancy tooling)

- https://developer.apple.com/videos/play/wwdc2020/10632/ — "Optimize Metal Performance for Apple silicon Macs": shader-core optimization, 16-bit data types, address spaces / constant prefetch, memory-access pitfalls. **HIGH**
- https://developer.apple.com/videos/play/wwdc2020/10602/ — "Harness Apple GPUs with Metal": TBDR, tile memory, imageblocks, tile shading, tiled deferred lighting with light culling. **HIGH**
- https://developer.apple.com/videos/play/wwdc2020/10631/ — "Bring your Metal app to Apple silicon Macs": IMR-vs-TBDR, hidden surface removal, on-chip tile color/depth. **HIGH**
- https://developer.apple.com/videos/play/wwdc2020/10603/ — "Optimize Metal apps and games with GPU counters": the counter/limiter tooling to diagnose register spills, tile-memory and buffer read/write limiters. **HIGH**
- https://developer.apple.com/videos/play/tech-talks/10580/ — "Metal Compute on MacBook Pro" tech talk: walks through a compute kernel stuck at **16% occupancy** caused by register spilling, and how to recover it — near-exact to your occupancy-collapse hypothesis. **HIGH**
- https://developer.apple.com/documentation/metal/tailor-your-apps-for-apple-gpus-and-tile-based-deferred-rendering — Apple docs on imageblocks, tile shaders, raster order groups. **HIGH**
- https://developer.apple.com/videos/play/tech-talks/604/ — "Metal 2 on A11 – Tile Shading": tile size vs tile/fragment complexity tradeoff, imageblock layouts. **HIGH**
- https://metalkit.org/wwdc20-whats-new-in-metal/ — Third-party notes summarizing Apple's performance limiters (tile memory, buffer read/write, register spills, avoid device atomics). **MEDIUM**
- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf — MSL spec: fast-math/FP-contract compiler options, address spaces — the target language your WGSL compiles into. **MEDIUM**

---

### GROUP 4 — Apple GPU reverse-engineering (independent): register file, ISA, latency

- https://dougallj.github.io/applegpu/docs.html — Dougall Johnson's Apple G13 GPU architecture reference: 32-wide SIMD-group, up to 128 GPRs/thread, conditional execution model. **HIGH**
- https://github.com/dougallj/applegpu — The tooling/disassembler/emulator behind the above. **HIGH**
- https://github.com/philipturner/metal-benchmarks/blob/main/README.md — Full README of Turner's microbenchmarks (occupancy caps at 24 simds/core, register-dependency latency penalties, 16-bit advantages specifically at low occupancy). **HIGH**
- https://rosenzweig.io/blog/asahi-gpu-part-5.html — Rosenzweig, "The Apple GPU and the Impossible Bug": clearest explanation of TBDR two-pass vertex/tiler/fragment flow and the tilebuffer; contrasts IMR vs tiler bandwidth. **HIGH**
- https://alyssarosenzweig.ca/blog/asahi-gpu-part-1.html — "Dissecting the Apple M1 GPU, part I": reverse-engineering methodology. **MEDIUM**
- https://alyssarosenzweig.ca/blog/asahi-gpu-part-n.html — "Dissecting the Apple M1 GPU, the end": retrospective on the full driver effort. **MEDIUM**
- https://asahilinux.org/2022/11/tales-of-the-m1-gpu/ — Asahi Linux GPU deep-dive (Alyssa + Dougall + Lina) on driver/firmware internals. **MEDIUM**
- https://www.michaelstinkerings.org/apple-m5-gpu-roofline-analysis/ — Independent M5 GPU roofline analysis: measured **~815 GFLOPS vs ~3,500–4,000 theoretical (a 4–5× gap)**, attributed to a scalar ISA decomposing float4 into 4 scalar instructions — directly relevant to interpreting your compute-bound-vs-latency-bound split. **HIGH**
- https://arxiv.org/pdf/2603.27569 — "Beating vDSP: Radix-8 FFT on Apple Silicon": documents the two-tier register (208 KiB)/threadgroup-memory hierarchy; latency-friendly data-layout ideas. **MEDIUM**

---

### GROUP 5 — Mobile/TBDR vendor guides: register pressure, occupancy, wave/GPR limits

- https://developer.arm.com/documentation/101863/latest/Using-Mali-Offline-Compiler/Performance-analysis/Resource-usage — Mali Offline Compiler docs: register-pressure → stack spilling, 16-bit types to reduce pressure, shrinking variable live ranges. **HIGH**
- https://developer.arm.com/community/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/the-mali-gpu-an-abstract-machine-part-4---the-bifrost-shader-core — Arm "Abstract Machine" Bifrost shader-core: 64 registers, occupancy halving above 32 registers. **HIGH**
- https://developer.arm.com/documentation/102596/0109/Shader-core-functional-units — Mali-G68 counters: the ">32 registers halves peak occupancy" rule quantified as a counter. **MEDIUM**
- https://docs.qualcomm.com/bundle/publicresource/topics/80-78185-2/mobile_best_practices.html — Adreno best practices: GPR/instruction-cache limits, wave-context occupancy, splitting long shaders, GMEM/tile approach for GPU-driven rendering. **HIGH**
- https://docs.qualcomm.com/bundle/publicresource/topics/80-78185-2/overview.html — Adreno overview: FlexRender tiling, LRZ/Early-Z/Fast-Z, UBWC bandwidth compression. **MEDIUM**
- https://developers.meta.com/horizon/blog/unreal-engine-adreno-offline-compiler-meta-quest/ — Meta/Quest: using Adreno Offline Compiler to read register footprint & ALU fiber occupancy on real UE material shaders — a genuine profiling war-story with before/after stats. **HIGH**
- https://docs.imgtec.com/starter-guides/powervr-architecture/html/topics/tile-based-deferred-rendering-index.html — Imagination PowerVR TBDR primer (the TBDR originator): HSR removes opaque overdraw entirely. **HIGH**
- https://docs.imgtec.com/starter-guides/powervr-architecture/html/topics/further-tbdr-details.html — PowerVR deeper TBDR details (tiler, parameter buffer). **MEDIUM**
- https://blog.imaginationtech.com/the-dr-in-tbdr-deferred-rendering-in-rogue/ — PowerVR Rogue deferred-rendering internals blog. **MEDIUM**
- https://chipsandcheese.com/p/the-snapdragon-x-elites-adreno-igpu — Chips and Cheese Adreno X1 deep-dive: register-file capacity vs wide waves, and the explicit finding that Adreno "might struggle to hide latency with shaders that use a lot of registers." **HIGH**
- https://chipsandcheese.com/p/arms-bifrost-architecture-and-the — Chips and Cheese Bifrost/Mali-G52: clause-based scheduling, 64-register/occupancy-halving detail. **MEDIUM**

---

### GROUP 6 — Occupancy vs register pressure vs latency (the working hypothesis)

- https://gpuopen.com/learn/optimizing-gpu-occupancy-resource-usage-large-thread-groups/ — Sebastian Aaltonen (Claybook) on optimizing compute-shader occupancy & register usage with large thread groups; SGPR/VGPR budgeting, scalar loads to cut VGPR pressure. **HIGH**
- https://gpuopen.com/learn/occupancy-explained/ — AMD "Occupancy explained": VGPR→wavefront math, register spilling to scratch, when a few saved registers to gain a wave wins. **HIGH**
- https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-register-pressure-readme/ — AMD register-pressure lab notes: recognizing/mitigating register-pressure & scratch-spill; code-restructuring guidance. **HIGH**
- https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf — Volkov, "Better Performance at Lower Occupancy" (GTC 2010): a 1024-point FFT hits higher performance at ~17% occupancy and concludes *"Mere 25% occupancy is sufficient"* to hide memory latency via ILP — the essential counter-argument that may complicate your occupancy-collapse hypothesis. **HIGH**
- https://www.sebastianaaltonen.com/blog/no-graphics-api — Aaltonen on raw-pointer loads reducing register pressure, mobile SSBO/uniform-buffer performance, and GPU-vendor codegen quirks. **HIGH**
- https://advances.realtimerendering.com/s2023/AaltonenHypeHypeAdvances2023.pdf — Aaltonen (HypeHype) "Modern Mobile Rendering": mobile tile memory, framebuffer fetch, fp16 double-rate math, and Nanite-style V-buffer/software-raster feasibility on mobile. **HIGH**
- https://enginearchitecture.realtimerendering.com/downloads/reac2023_modern_mobile_rendering_at_hypehype.pdf — Companion REAC 2023 deck: *"Mobile: 16KB uniform buffers! SSBOs are slow!"* and 3D-tiled volume-texture layout tips for SDF rendering. **HIGH**
- https://www.slideshare.net/DICEStudio/future-directions-for-computeforgraphics — DICE "Future Directions for Compute-for-Graphics": occupancy, ubershaders vs dynamic dispatch, visibility-buffer takeaways, why static resource allocation forces low occupancy on "worst-case" code paths. **HIGH**

---

### GROUP 7 — Heavy per-pixel loops: grass, raymarch, SDF, volumetrics (foliage cost)

- https://gdcvault.com/play/1027033/Advanced-Graphics-Summit-Procedural-Grass — Eric Wohllaib, "Procedural Grass in Ghost of Tsushima" (GDC 2021): GPU-generated per-blade grass, ~1M blades in ~2ms; the reference grass architecture. **HIGH**
- https://archive.thedatadungeon.com/ghost_of_tsushima_2020/documents/gdc_2021/gdc_2021_procedural_grass_in_got.pdf — The slide deck PDF of the above. **HIGH**
- https://www.youtube.com/watch?v=Ibe1JBF5i5Y — Video of the Ghost of Tsushima procedural grass talk. **MEDIUM**
- https://babylonjs.medium.com/infinite-grassland-ray-traced-rare-semi-uniform-entity-manifolds-for-rendering-countless-discreta-36630d72fc6 — Babylon.js "Infinite Grassland": screen-constant-cost grass where per-pixel shader weight dominates — maps directly to your per-pixel raymarched grass. **HIGH**
- https://arxiv.org/pdf/1609.05344 — Toft/Bowles/Zimmermann, "Optimisations for Real-Time Volumetric Cloudscapes": produces visually similar results with **1/16 the raymarch steps** via jittered offsets + analytical integration + TAA, with real draw-time numbers (297.7ms → 7.5ms). **HIGH**
- https://arxiv.org/pdf/2210.06160 — "RTSDF: Real-time Signed Distance Fields for Soft Shadows": raymarching SDFs, sample-count-vs-cost, combining with shadow maps to skip raymarch where possible. **MEDIUM**

---

### GROUP 8 — WebGPU / WGSL → Metal (Dawn/Tint/wgpu) performance & gotchas

- https://discourse.threejs.org/t/webgpu-performance-issue/87939 — three.js forum: real report of WebGPU running 2-4× slower than WebGL on the same scene, and `forceWebGL` making it 5-10× worse — a genuine three.js/WebGPU perf war-story. **HIGH**
- https://threejsroadmap.com/blog/profiling-webgpu — How to use WebGPU timestamp queries via three.js (`trackTimestamp`, `resolveTimestampsAsync`) to measure per-pass compute/render GPU time; your profiling entry point. **HIGH**
- https://github.com/mrdoob/three.js/issues/32735 — three.js issue: inefficient compute-pipeline caching (one `GPUComputePipeline` per `ComputeNode`) — a concrete TSL/WebGPU perf pitfall for many compute dispatches of the same logic. **HIGH**
- https://github.com/gfx-rs/wgpu/issues/4456 — wgpu issue: WGSL passes validation but the **Metal shader compiler times out/hangs** — concrete WGSL→MSL compilation gotcha on Apple. **HIGH**
- https://developer.chrome.com/blog/new-in-webgpu-130 — Chrome/Dawn: Tint's new IR made WGSL→MSL translation **up to 10× faster** ("Initial tests show that the new version of Tint is up to 10 times faster when translating Unity's WGSL shaders to MSL"); confirms the WGSL→MSL step is a real cost center. **MEDIUM**
- https://blog.maximeheckel.com/posts/field-guide-to-tsl-and-webgpu/ — Maxime Heckel's TSL/WebGPU field guide: undocumented TSL gotchas from porting real shader projects (glass, particles, post-processing). **MEDIUM**
- https://discourse.threejs.org/t/compute-shader-tsl/80277 — three.js forum: raymarching/SDF ping-pong GPGPU in TSL — directly adjacent to your raymarched-grass-in-TSL case. **MEDIUM**

---

### GROUP 9 (HIGH-RISK / DELIBERATELY TANGENTIAL — potentially gold)

- https://arxiv.org/pdf/1803.08601 — "Design Principles for Sparse Matrix Multiplication on the GPU": clean formal treatment of TLP+ILP latency hiding and occupancy — transferable to your dependent-fetch latency problem. **MEDIUM**
- https://arxiv.org/pdf/1509.02308 — "Dissecting GPU Memory Hierarchy through Microbenchmarking": memory-latency measurement methodology; informs latency-bound analysis. **MEDIUM**
- https://arxiv.org/pdf/1907.02894 — "RegDem: Increasing GPU Performance via Shared Memory Register Spilling": deliberately spilling registers to shared memory to raise occupancy — an unusual restructuring strategy for your register-pressure case. **MEDIUM**
- https://arxiv.org/pdf/2204.01287 — "Software Rasterization of 2 Billion Points in Real Time": compute-atomic point rasterization, closely related to your atomic-depth-election rasterizer. **MEDIUM**
- https://arxiv.org/pdf/2505.02017 — "Aokana: A GPU-Driven Voxel Rendering Framework for Open World Games": modern GPU-driven visibility-style pipeline for huge scenes. **LOW**
- https://arxiv.org/pdf/2606.12765 — "Rigel: Reverse-Engineering the Metal 4.1 Tensor Compute Path on the Apple M4 Max GPU": recent Apple-GPU microbenchmarking methodology. **LOW**
- https://forums.macrumors.com/threads/apple-silicon-in-sciences.2374458/page-12 — Philip Turner discussing Apple GPU/driver latency and GPU-driven porting difficulties (forum, but primary voice). **LOW**

## Recommendations
1. **Start with the two sources that share your exact constraints**: Scthe's nanite-webgpu (GROUP 1) for the WebGPU/WGSL atomic<u64> reality and the 32-bit packing workaround, and Arm's "Mali and Nanite" post (GROUP 2) for how the visibility-buffer/compute-raster pipeline actually behaves on a TBDR tiler. These tell you what is architecturally possible before you spend time optimizing.
2. **Instrument before restructuring.** Use WebGPU timestamp queries (threejsroadmap profiling post) to split raymarch / foliage-raster / deferred-lighting costs, then — where you can get native access via a Metal capture — Apple's GPU counters talk (WWDC 10603) and the Metal Compute tech talk (10580) to read occupancy and register-spill directly. **Threshold:** if Xcode/GPU-counter occupancy is well under 50% and "spilled bytes" is nonzero (the 10580 talk shows a real kernel stuck at 16%), register pressure is your primary lever; act on step 3. If occupancy is healthy but throughput is still low, jump to step 4.
3. **Attack register pressure first.** Apply Arm/Adreno/Aaltonen guidance: demote to 16-bit (fp16/half) wherever the raymarch and lighting math tolerate it, shrink live ranges, split the mega-shader (separate the raymarch from deferred lighting into distinct passes), and move wave-invariant data out of per-lane registers. On Mali/Bifrost the concrete cliff is 32 registers (above it, occupancy halves); Apple's model differs — Turner's benchmarks show Apple prefers spilling to lowering ALU utilization — so validate the direction empirically rather than trusting the Mali constant.
4. **Test the occupancy hypothesis explicitly, don't assume it.** Read Volkov and the DICE compute-for-graphics deck: your kernel may be ILP-limited rather than occupancy-limited. Issue several independent baked-field/volume fetches *before* consuming any of them, and unroll the raymarch inner loop; if throughput improves at fixed occupancy, latency is hideable with ILP and chasing raw occupancy would be wasted effort.
5. **Reduce dependent fetches in the grass raymarch.** Borrow the volumetric-cloud playbook (fewer steps + jitter + analytical integration + TAA — 1/16 the steps in the cloudscapes paper) and use latency-friendly data layout (3D-tiled volume textures per Aaltonen) so baked-field fetches hit cache. On unified memory the win is cache residency and access-pattern coherence, not raw bandwidth.
6. **Watch the WGSL→MSL boundary.** The wgpu compiler-hang issue and Chrome's 10× Tint-IR speedup confirm this layer is both fragile and costly; keep shaders within what Tint translates cleanly, and profile pipeline-creation time separately from run time (three.js compute-pipeline-caching issue #32735 shows duplicate pipelines being compiled for identical TSL logic).
7. **Escalation benchmark:** target the specific device tier that collapses (M1/M2 base, or a Steam Deck/Adreno handheld). If, after steps 3–5, native GPU-counter occupancy is above ~50%, spills are eliminated, and you still miss 30 fps, the bottleneck is genuinely memory-latency/bandwidth on unified memory — at that point reduce per-pixel work (lower raymarch step count, tile/imageblock-resident lighting) rather than continuing to chase occupancy.

## Caveats
- **Vendor guides describe their own hardware.** Arm/Adreno/PowerVR register and occupancy numbers (e.g., the 32-register Mali cliff, Adreno GPR limits) do NOT transfer 1:1 to Apple's AGX; Apple's register file and occupancy model are larger/different per Turner and the M5 roofline analysis. Use them as directional, not literal.
- **Apple internals from reverse-engineering are unofficial** — Dougall Johnson, Rosenzweig, and Turner all explicitly state this — and may contain errors or lag current M3/M4/M5 silicon.
- **The M5 roofline gap (~815 vs ~3,500–4,000 GFLOPS) is attributed by that author to scalar-ISA float4 decomposition, not solely memory latency** — a reminder to distinguish compute-bound from latency-bound before concluding "latency can't be hidden."
- **Some sources are desktop-discrete-GPU-centric** (Volkov/CUDA, AMD GPUOpen). The mechanisms transfer, but the tuning constants do not — treat as conceptual.
- **The nanite-webgpu author frames atomic<u64> as a WebGPU-API restriction, not a hardware one, and makes no Apple-specific benchmark claims;** do not read Apple-Silicon numbers into that project. His only hard perf figure is "the FPS tanks 40%" when toggling a culling checkbox, and he notes raw FPS is VSync-limited (use the demo's "Profile" button).
- **The ~5M-triangle / ~90%-software-rasterized Nanite figures come from elopezr's RenderDoc teardown, not from Epic or Arm directly;** Arm's Mali-Nanite demo is its internal "Mori" scene.
- **A few listed items are secondary/aggregator pages** (metalkit.org notes, Medium Nanite explainers, forum threads); they are included only where they summarize or voice primary technical content, and are rated MEDIUM/LOW accordingly.
- **Philip Turner's separate "ue5-nanite-macos" project could not be verified to a working URL within budget and is intentionally omitted;** his verified metal-benchmarks repo is included instead as the primary Apple-GPU-microarchitecture source.