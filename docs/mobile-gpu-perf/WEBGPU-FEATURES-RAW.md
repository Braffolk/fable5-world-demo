# WebGPU feature availability 120-150 — per-area detailed findings

_Per-agent detailed findings, extracted from the workflow run for cross-session reference._


---

## (agent)

- **shader-f16 (WGSL f16 half-precision type; device feature "shader-f16" + WGSL `enable f16;`)**
  - _chromeStatus_: SHIPPED STABLE since Chrome 120 (Dec 2023). Non-flag, non-OT. On Chrome 150 (Jul 2026) it is long-stable and, per three.js's request-all path, already enabled on our M1 Max device. Chrome io24 blog: f16 runs the HuggingFace WebGPU text-embed benchmark ~3x faster than f32 on an M1 Max laptop; io24-part2 cites 28%/41% prefill/decode gains on Llama2-7B M1 Pro. Availability is adapter-conditional (chromestatus 5180552617656320: "Not all GPUs support 16-bit float"; Qualcomm/Adreno historically excluded — gpuweb#5006).
  - _detect_: adapter.features.has('shader-f16')  // on the GPUAdapter, BEFORE requestDevice. Post-device: device.features.has('shader-f16') / renderer.backend.hasFeature('shader-f16'). Our device ALREADY holds it: three r184 WebGPUBackend.js:204-223 iterates Object.values(GPUFeatureName), pushes every adapter.features.has(name) into requiredFeatures, then requestDevice — so shader-f16 is auto-requested whenever the adapter exposes it (no Engine.ts change needed to ENABLE it). Guard our own use with: const f16 = (renderer.backend.device ?? adapter).features.has('shader-f16').
  - _whereApplies_: Register/occupancy relief on Apple TBDR (M1 Max = NO double-rate FLOPs, but half the register/VGPR pressure per value → higher occupancy on our exact bottleneck: dependent-fetch per-pixel loops). Targets, in priority: (1) grass RAYMARCH accumulator/step state in NaniteGrass.ts kGrassRay (the +14.5ms register-heavy march loop, task G-C) — carry march position deltas, fiber half-width, accumulated color/coverage as f16; (2) voxel-brick SCATTER (128-lane workgroup) brick color/normal/spread temporaries; (3) deferred RESOLVE BRDF intermediates (albedo/normal/sun/IBL/GI scratch) in NaniteResolve. NOT the atomicMax election in NaniteRaster (24-bit depth key must stay integer/f32 — precision-critical, do not touch).
  - _threejsTSL_: PARTIALLY reachable, and NOT via plain TSL nodes. three r184 WGSLNodeBuilder.js:1515-1519 has enableShaderF16() which emits the `enable f16;` directive (and enableSubgroupsF16 → 'subgroups-f16'). BUT there is NO f16/half TSL node datatype — every TSL node is f32/i32/u32; you cannot declare an f16 var, vecN<f16>, or f16 arithmetic through the TSL builder. Our kernels are 100% TSL `Fn(()=>{...})().compute(...)` (NaniteRaster.ts:534,543,577,...; NaniteGrass; NaniteResolve) with ZERO raw WGSL (grep: no wgslFn/FunctionNode/CodeNode anywhere in src). So to actually USE f16 we must author the hot inner block as RAW WGSL via a FunctionNode/wgslFn that starts `enable f16;` (or trigger builder.enableShaderF16()) and does its own f16 packing/unpacking at the f32 TSL boundary. That is real surgery — rewrite one hot loop body per kernel in WGSL, keep buffers f32 at the interface. TSL-native f16 would need a three.js upstream node-type addition (does not exist in r184).
  - _fallback_: Feature-detect; if absent, keep the current all-f32 TSL kernels UNCHANGED (they already work). Implementation shape: ship both a TSL-f32 kernel and a raw-WGSL-f16 kernel variant behind `if (device.features.has('shader-f16')) useF16Kernel else useF32Kernel` — Safari/Firefox and any f16-less adapter transparently get the f32 path. No correctness divergence required; f16 only trims precision on scratch math we choose. Zero risk to non-Chrome devices.
  - _usefulToUs_: high
  - _weWereWrong_: true
- **subgroups-f16 (f16 subgroup ops; device feature "subgroups-f16", requires shader-f16 + subgroups)**
  - _chromeStatus_: subgroups shipped stable Chrome 125 (was OT earlier); subgroups-f16 rides on shader-f16 + subgroups and is adapter-conditional. On Chrome 150 both are stable where the adapter supports them; three auto-requests both via the request-all loop. Not present on adapters lacking either base feature.
  - _detect_: adapter.features.has('subgroups') && adapter.features.has('subgroups-f16'). three r184 exposes enableSubgroups()/enableSubgroupsF16() (WGSLNodeBuilder.js:1492/1501) to emit the directives; the TSL subgroup helper nodes exist but again carry no f16 datatype.
  - _whereApplies_: Only marginally: an f16 subgroup reduction could shave the grass-march coverage accumulate or a resolve tile reduction, but our loops are dependent-FETCH-bound (memory latency), not reduction-bound, so subgroup f16 is a second-order lever. List it so it is not re-flagged as 'blocked'; do not spend the arc here before the plain shader-f16 kernel rewrite lands.
  - _threejsTSL_: Reachable only through raw WGSL: enableSubgroupsF16() emits the directive, but subgroupAdd/subgroupBroadcast over f16 values must be hand-written WGSL (same FunctionNode path as shader-f16; no TSL f16 subgroup node exists in r184).
  - _fallback_: Detect both features; absent → use the f32 subgroup path (or no-subgroup TSL path we already run). Non-Chrome/older adapters unaffected.
  - _usefulToUs_: low
  - _weWereWrong_: false

---

## (agent)

- **immediates (var<immediate> / requires immediate_address_space) — WebGPU push-constant-like per-dispatch data**
  - _chromeStatus_: Chrome 150 STABLE (landed across 149-150; enabled by default in stable, no flag/origin-trial). Sources: developer.chrome.com/blog/new-in-webgpu-149-150, webgpu.com/news/chrome-149-150-webgpu-immediates, spec PR gpuweb/gpuweb#5423, chromestatus.com/feature/5199437611794432.
  - _detect_: navigator.gpu.wgslLanguageFeatures.has('immediate_address_space')  // WGSL language-extension test, NOT adapter.features. No requiredFeatures entry needed. Optional larger payload gated by device.limits.maxImmediateSize (guaranteed 64 bytes; may be higher). JS call: passEncoder.setImmediates(byteOffset, ArrayBufferView, srcOffset?).
  - _whereApplies_: Mechanism: replaces a managed uniform-buffer + bind-group update with a pass-encoder setImmediates() call carrying <=64 bytes, read in WGSL as var<immediate>. Fits our SMALL per-dispatch scalars: cull args (tau/minPx/lodNear/lodPow/simBandD, NaniteCull.ts:452-469), per-cascade shadow constants (NaniteShadow.ts:204-205 cullTau/cullMinPx), per-level HZB constants. It targets exactly Aaltonen's 'SSBOs slow on mobile' pain — BUT three.js does NOT put these in SSBOs today; it already uses uniform buffers, so the realizable win on our stack is marginal (saves a bind-group swap / buffer write per dispatch; on Metal/TBDR maps to inline setBytes-style constants, small).
  - _threejsTSL_: NOT reachable via three r184 TSL. No <immediate>/setImmediates node exists; WebGPUBackend never emits setImmediates and TSL uniform() nodes are coalesced into renderer-managed uniform buffers + bind groups. three.js issue #33576 is a heads-up only (no PR, no assignee, no milestone; author says 'wait until available everywhere'). Adoption would require patching three's WebGPUBackend compute/render pass recording — a raw-WGSL FunctionNode alone does NOT help, because the win requires the JS-side setImmediates pass-encoder call which three owns. We already hold renderer.backend.device (Engine.ts:91) but NOT the internally-recorded pass encoders. High plumbing cost for a marginal gain.
  - _fallback_: Keep current TSL uniform()/uniformF() nodes (three-managed uniform buffers) — effectively cost-equivalent on our stack today, so no adoption recommended. If ever adopted, gate the setImmediates path on wgslLanguageFeatures.has('immediate_address_space') and fall back to the uniform-buffer node on Safari/Firefox/older Chrome.
  - _usefulToUs_: low
  - _weWereWrong_: false
- **dynamic buffer offsets (bindGroupLayout hasDynamicOffsets + setBindGroup(offsets), 256-byte-aligned)**
  - _chromeStatus_: Stable since WebGPU launch (Chrome 113); unchanged through 150. Sources: gpuweb/gpuweb#116, developer.chrome.com WebGPU docs.
  - _detect_: No feature flag — core API, always present when navigator.gpu exists. Offsets must be multiples of 256 (validation-enforced).
  - _whereApplies_: The pre-existing, portable alternative to immediates for many small per-dispatch param sets: one big uniform buffer + a rolling per-dispatch dynamic offset instead of N buffers/bind-groups. Applies to the same cull/raster/per-level-shadow constants, with NO Chrome-150 dependency. Included so we don't treat immediates as the only option.
  - _threejsTSL_: NOT exposed through TSL — three.js manages bind groups and does not surface hasDynamicOffsets on node materials/compute; would need WebGPUBackend patching. three's existing uniform coalescing already achieves most of this in practice, so no action recommended.
  - _fallback_: Current uniform() node path; no fallback branch needed (universally available).
  - _usefulToUs_: low
  - _weWereWrong_: false
- **GPUBuffer usable directly as GPUBindingResource (bind a buffer without a {buffer,offset,size} GPUBufferBinding)**
  - _chromeStatus_: Chrome 138 STABLE. Source: developer.chrome.com/blog/new-in-webgpu-138.
  - _detect_: No runtime flag — implied by Chrome >=138; feature-detect indirectly via try/catch on the binding form if needed. Not advertised in adapter.features.
  - _whereApplies_: Ergonomic only (simpler bind-group setup when offset/size are defaults). Zero perf effect on our per-pixel-bound frame; three.js builds bindings internally so it is invisible to us.
  - _threejsTSL_: Internal to three's WebGPUBackend; not a TSL surface. No action.
  - _fallback_: n/a — no adoption.
  - _usefulToUs_: no
  - _weWereWrong_: false
- **GPUBuffer.mapSync() (synchronous buffer mapping, Worker-only)**
  - _chromeStatus_: Chrome 145 — EXPERIMENTAL/prototype, restricted to Worker contexts. Source: developer.chrome.com/blog/new-in-webgpu-145.
  - _detect_: typeof GPUBuffer.prototype.mapSync === 'function' AND running inside a Worker. Experimental — do not depend on it.
  - _whereApplies_: Readback/upload convenience for CPU<->GPU transfer, not our per-dispatch shader-data path. Our hot path is GPU-resident; does not touch the pixel-loop / atomic-election bottleneck.
  - _threejsTSL_: Renderer-level, not a TSL concern. No action.
  - _fallback_: Keep mapAsync-based readback (already used for timestamp/meter reads).
  - _usefulToUs_: no
  - _weWereWrong_: false
- **TRANSIENT_ATTACHMENT texture usage (on-chip tile-memory attachments) — ADJACENT to data-passing, high-value on our TBDR target**
  - _chromeStatus_: Chrome 150 STABLE (usage flag + tightened validation, 149-150). Source: developer.chrome.com/blog/new-in-webgpu-149-150.
  - _detect_: GPUTextureUsage.TRANSIENT_ATTACHMENT !== undefined  // presence of the usage flag. No adapter.features gate. Validation: viewFormats must be [], view usage immutable, cannot be a resolve target.
  - _whereApplies_: Not per-dispatch data-passing, but flagged because it directly targets our Apple-Silicon/TBDR memory bottleneck: intermediate render attachments (e.g. resolve/post depth-stencil or MSAA scratch in NaniteResolve/post) written-then-consumed within a pass can stay in fast on-chip tile memory and never allocate VRAM. ZERO benefit to the compute software rasterizer (vis-buffer is a storage buffer, not an attachment) — applies only to the render-target parts of resolve/post.
  - _threejsTSL_: NOT exposed by three r184 — RenderTarget creation does not surface TRANSIENT_ATTACHMENT usage; would need WebGPUBackend/RenderTarget patching. Most of our frame (storage-buffer vis election) cannot use it anyway.
  - _fallback_: Normal render targets (current). Any adoption gated on the usage-flag presence check + fall back to a regular attachment on Safari/Firefox/older Chrome.
  - _usefulToUs_: medium
  - _weWereWrong_: false

---

## (agent)

- **subgroups (core: subgroupElect / subgroupBallot / subgroupAdd / subgroupMax / subgroupMin / subgroupBroadcast / subgroupBroadcastFirst / subgroupInclusiveAdd / subgroupExclusiveAdd / subgroupShuffle)**
  - _chromeStatus_: STABLE since Chrome 134 (2025-02-26). Went through origin trial in 128-131, shipped stable 134. Fully stable in Chrome 150 (Jul 2026). Subgroup size on all Apple Silicon = 32 (min/max reported via adapter.info.subgroupMinSize / subgroupMaxSize).
  - _detect_: const adapter = await navigator.gpu.requestAdapter(); const hasSG = adapter.features.has('subgroups'); // sizes: adapter.info.subgroupMinSize / adapter.info.subgroupMaxSize. On the DEVICE: renderer.hasFeature('subgroups') (three wraps device.features.has). WGSL requires `enable subgroups;` — three's WGSLNodeBuilder auto-emits it when a subgroup builtin/op is used (WGSLNodeBuilder.js:1643-1646).
  - _whereApplies_: STRONGEST use = QUEUE COMPACTION in cull, NOT election. src/nanite/NaniteCull.ts does one global atomicAdd per surviving thread to append into qRaster (kVoxFanout :655), qVoxRaster, and per-bucket counts (kVoxCount :790, kVoxScatter cursor :865). Replace per-thread atomicAdd with the subgroup-aggregated pattern: b=subgroupBallot(keep); n=countOneBits(b); base=subgroupBroadcastFirst( elect? atomicAdd(counter,n):0 ); slot=base+subgroupExclusiveAdd(u32(keep)) — cutting global-counter atomic traffic ~32x on Apple (subgroup size 32). Clean win, lanes are independent so no structural mismatch. SECONDARY (LOW value, see note): the per-pixel atomicMax ELECTION in NaniteRaster.ts:603 — a subgroup pre-reduce could in principle combine same-pixel candidates before the global atomic, BUT (a) our rasterizer is TRIANGLE-parallel scatter, so lanes in a subgroup target DIFFERENT pixels (no shared-pixel reduction to exploit without a pixel-tiled re-shape), and (b) our own ?relect ablation (docs/perf-runs/2026-07-04-mem-boot-raster-arc.md:85-101, RASTER arc CLOSED 62d607e) already MEASURED election contention as effectively dead — the loop is intrinsic per-covered-pixel work, not atomic contention. So do NOT expect the headline 'slash atomic election contention' payoff; the compaction use is where subgroups actually pay.
  - _threejsTSL_: REACHABLE via three r184 TSL — no raw WGSL needed. Import from 'three/tsl': subgroupElect, subgroupBallot, subgroupAdd, subgroupMax, subgroupMin, subgroupBroadcast, subgroupBroadcastFirst, subgroupInclusiveAdd, subgroupExclusiveAdd, subgroupShuffle (node_modules/three/src/nodes/gpgpu/SubgroupFunctionNode.js:168-196, exported nodeProxy fns :210+). Builtins subgroupSize / invocationSubgroupIndex via builder (WGSLNodeBuilder.js:1353/1366). CRUCIAL: three's WebGPUBackend auto-requests EVERY adapter feature (WebGPUBackend.js:204-223 loops all GPUFeatureName, requests any adapter.features.has) — so 'subgroups' is ALREADY enabled on our device with ZERO changes to src/core/Engine.ts (which passes no requiredFeatures). We just call the TSL nodes; the directive is auto-emitted. Note: countOneBits over the vec4<u32> ballot may need a small helper (sum of 4 lanes) — trivial in TSL.
  - _fallback_: Runtime-branch at build time on renderer.hasFeature('subgroups'): if false, emit today's per-thread atomicAdd compaction path (already the shipped kernel) — identical results, just no aggregation. Gate behind a ?sg flag mirroring the existing ?relect/?swcoop diagnostics so A/B is bit-identity checkable. Absence path = current behavior, so Safari/Firefox/older-Chrome degrade gracefully with no visual change.
  - _usefulToUs_: high
  - _weWereWrong_: true
- **subgroups-f16 (f16 collective ops: subgroupAdd/Max/Min/Broadcast on f16)**
  - _chromeStatus_: STABLE in Chrome (shipped alongside/after the core subgroups + shader-f16 pairing; available and stable by Chrome 150, Jul 2026). Requires BOTH 'shader-f16' AND 'subgroups' features on the device.
  - _detect_: adapter.features.has('subgroups') && adapter.features.has('shader-f16'). WGSL needs `enable f16; enable subgroups; enable subgroups_f16;` (subgroups_f16 directive). three exposes enableSubgroupsF16() (WGSLNodeBuilder.js:1499) but does NOT auto-emit it — must be triggered by using f16-typed subgroup args.
  - _whereApplies_: Only relevant if we run subgroup reductions over f16 payloads. Our compaction counters and election keys are u32/f32, not f16 — no natural fit today. Would only matter if a future grass/voxel reduction operated on packed f16 lighting/coverage accumulators. Not a current lever.
  - _threejsTSL_: Partially reachable: the same subgroupAdd/Max/Min TSL nodes accept f16-typed inputs, and enableSubgroupsF16() exists, but three does not auto-enable the subgroups_f16 directive, so a raw enable directive injection (or a small builder patch) would be needed. Effectively raw-WGSL-adjacent for now.
  - _fallback_: Use f32 subgroup ops (fully supported) — no need for f16 variant. Absence is a non-issue since we have no f16 reduction workload.
  - _usefulToUs_: low
  - _weWereWrong_: false
- **subgroup_id builtin + subgroup_uniformity extension**
  - _chromeStatus_: subgroup_id builtin: STABLE Chrome 144. subgroup_uniformity extension (moves uniformity analysis for subgroup/quad builtins to subgroup scope so more values, incl. subgroup_id, are subgroup-uniform): STABLE Chrome 145. Both stable by Chrome 150 (Jul 2026).
  - _detect_: Gated by the same adapter.features.has('subgroups'). subgroup_id / subgroup_uniformity are WGSL enable-directive/builtin refinements, not separate GPU features — presence follows subgroups. WGSL: `enable subgroups;` then use subgroup_id; add `enable subgroup_uniformity;` to relax uniformity errors.
  - _whereApplies_: subgroup_id (which subgroup within the workgroup) matters only if we do a two-level workgroup+subgroup cooperative reduction (e.g. subgroup-partial then workgroup-combine) in the 128-lane voxel-scatter or a tiled election. Marginal for us now; the single-level subgroup-aggregated atomicAdd (feature #1) captures the win without needing subgroup_id. subgroup_uniformity mainly removes compiler uniformity errors when using these in non-uniform control flow — useful safety valve if the compaction code trips uniformity analysis.
  - _threejsTSL_: subgroup_id reachable via builder.getSubgroupIndex() (WGSLNodeBuilder.js:1379, exposes 'subgroup_id' as subgroupIndex). subgroup_uniformity directive is NOT auto-emitted by three r184 — would need a raw `enable subgroup_uniformity;` injection if the uniformity analyzer rejects our compaction placement.
  - _fallback_: Single-level subgroup reduction (no subgroup_id) or the existing atomicAdd path. If subgroup_uniformity is unavailable, keep subgroup ops in uniform control flow (structure the ballot/prefix at workgroup-uniform points).
  - _usefulToUs_: low
  - _weWereWrong_: false
- **quad operations (quadBroadcast / quadSwapX / quadSwapY / quadSwapDiagonal)**
  - _chromeStatus_: STABLE with core subgroups since Chrome 134; usable in both compute and fragment stages. Stable in Chrome 150 (Jul 2026).
  - _detect_: adapter.features.has('subgroups') (quad ops are part of the subgroups feature). Fragment-stage quad ops always operate over the 2x2 quad; compute needs the subgroups enable directive.
  - _whereApplies_: Quad (2x2) exchange could compute screen-space derivatives / neighbor differences cheaply inside the deferred FRAGMENT resolve (NaniteResolve) — e.g. cross-lane normal/depth deltas for edge or AO terms without extra texture fetches. Minor. No use in the compute rasterizer's scatter (lanes aren't a pixel quad there). Not a priority vs the compaction win.
  - _threejsTSL_: REACHABLE: three r184 exports quadBroadcast, quadSwapX, quadSwapY, quadSwapDiagonal from TSL (SubgroupFunctionNode.js:186-196, 370-455). Works in fragment nodes where the resolve is authored.
  - _fallback_: Compute derivatives via dpdx/dpdy (standard, always available) or extra fetches. Absence changes nothing functionally.
  - _usefulToUs_: low
  - _weWereWrong_: false

---

## (agent)

- **CANONICAL DETECTION RECIPE + where our device is created**
  - _chromeStatus_: N/A — method note. Chrome stable is 149/150 (Jul 2026); 149-150 blog = developer.chrome.com/blog/new-in-webgpu-149-150
  - _detect_: Three layers. (1) GPU FEATURES (device-level caps): const a = await navigator.gpu.requestAdapter({powerPreference:'high-performance'}); a.features is a GPUSupportedFeatures set → a.features.has('shader-f16'|'subgroups'|'timestamp-query'|'timestamp-query-inside-passes'|'dual-source-blending'|'float32-filterable'|'float32-blendable'). Then requestDevice({requiredFeatures:[...]}) — MUST list each or WGSL using it fails validation. (2) WGSL LANGUAGE features (syntax/extensions): navigator.gpu.wgslLanguageFeatures.has('immediate_address_space'|'subgroup_id'|'uniform_buffer_standard_layout'|'subgroup_uniformity') + a matching `requires <name>;` or `enable <name>;` directive in the shader. (3) GROUND TRUTH dump = open webgpureport.org on the target machine — it lists the adapter's actual features/limits set. Our probe already does layer (1): src/core/Diagnostics.ts:65 requestAdapter, :95 `features:[...adapter.features]`.
  - _whereApplies_: OUR DEVICE IS CREATED BY three.js, not our code: src/core/Engine.ts:84 `new WebGPURenderer({trackTimestamp:true, requiredLimits})` → renderer.init() internally calls requestAdapter/requestDevice. We pass requiredLimits (buildRequiredLimits, Diagnostics.ts:36) but NO requiredFeatures — three r184's WebGPUBackend auto-enables the adapter features it knows (that is how timestamp-query gets on). To opt into shader-f16/subgroups we must either (a) confirm three includes them in its auto requiredFeatures list, or (b) pre-create our own GPUDevice with the extra requiredFeatures and hand it to the renderer. Adapter dump already surfaces on window via hooks.diag.features (Engine.ts:114).
  - _threejsTSL_: Layer-1 detection is plain JS we already run. The gap is that three owns requestDevice, so adding a requiredFeature = a renderer-construction change, not a TSL change.
  - _fallback_: Everything below must be gated on adapter.features.has(...) / wgslLanguageFeatures.has(...) with the current code path as the else-branch — Safari/Firefox WebGPU and older Chrome must keep working.
  - _usefulToUs_: high
  - _weWereWrong_: false
- **64-bit buffer atomics (atomic<u64> / proposal atomic_vec2u_min_max, atomic<vec2u>)**
  - _chromeStatus_: NOT SHIPPED as of Chrome 150. No stable, no flag, no origin-trial, no browser-exposed experimental path. It is a spec PROPOSAL: gpuweb #5071 + proposals/atomic-64-min-max.md, status Draft / 'Milestone 2 / waiting for PR'. developer.chrome.com/blog/next-for-webgpu lists 64-bit atomics as still 'under consideration and prioritization' (explicitly citing the Nanite software-raster atomicMax use case). Scope even when it lands is LIMITED: only atomicMin/atomicMax on a vec2u surrogate that backends map to u64, storage buffers only — not full atomic<u64> arithmetic.
  - _detect_: No feature string exists yet — adapter.features.has('...') returns false for every candidate name. When it ships it will be a named GPU feature + a WGSL enable; gate on adapter.features.has(<future-name>) and fall through otherwise.
  - _whereApplies_: This is EXACTLY our vis-buffer election (src/nanite/NaniteRaster.ts). Because there is no u64 atomic we currently PACK depth+id into 32 bits and split the election across two atomics: a 24-bit-Z key atomicMax into visPayloadV, then a winner-conditional atomicStore into visBV (see NaniteRaster.ts:162-198, 254-267 and the `relect` ablation at :374). The comments even name the failure mode this causes — the 'branch-through-trunk' depth/id desync race the packed key was invented to dodge (:197). A real atomicMax(u64 = depth<<32 | id) would collapse the whole two-buffer, race-prone election into ONE op and remove the packed-precision ceiling.
  - _threejsTSL_: Moot until it ships. Would need raw-WGSL FunctionNode anyway (TSL has no 64-bit atomic node) plus a device requiredFeatures change.
  - _fallback_: Our shipped 24-bit-Z + id two-atomic packed election IS the fallback and stays the default. Nothing to change now.
  - _usefulToUs_: high
  - _weWereWrong_: false
- **Texture atomics (R64Uint / TEXTURE_INT64_ATOMIC)**
  - _chromeStatus_: NOT SHIPPED in browser WebGPU as of Chrome 150. Exists in wgpu-native/Rust as Features::TEXTURE_INT64_ATOMIC (R64Uint image, backed by Vulkan VK_EXT_shader_image_atomic_int64 / DX12 SM6.6 / Metal MSL3.1) — gfx-rs #8662 — but it is NOT exposed as a WebGPU/Dawn browser feature. Same proposal orbit as buffer 64-bit atomics; the 'next for WebGPU' blog discusses buffer atomics only.
  - _detect_: No browser feature string; adapter.features.has(...) is false. Not usable from the web platform.
  - _whereApplies_: Would be the alternative Nanite path — atomicMax a packed depth|id directly into an R64Uint visbuffer TEXTURE instead of a storage buffer, letting the resolve sample it. Same election site (NaniteRaster.ts). Lower priority than buffer u64 for us since our visbuffer is already storage-buffer-based.
  - _threejsTSL_: Not reachable (no browser feature, no TSL node).
  - _fallback_: Storage-buffer packed election as today.
  - _usefulToUs_: medium
  - _weWereWrong_: false
- **timestamp-query**
  - _chromeStatus_: STABLE since Chrome 121 (no flag). Core profiling feature.
  - _detect_: adapter.features.has('timestamp-query'); requiredFeatures:['timestamp-query'].
  - _whereApplies_: Already in use and load-bearing for the whole perf arc: Engine.ts:114 `timestampsSupported = diag.features.includes('timestamp-query')`, :115 constructs GpuProfiler, :231-249 resolves per-pass GPU ms; three enables it via trackTimestamp:true (Engine.ts:86). CONFIRMED available and wired. Caveat unchanged: values are the pass-boundary timestampWrites three allocates (GpuProfiler.ts), and the MeasureHarness sub-vsync guard still applies.
  - _threejsTSL_: Fully reachable — three's WebGPURenderer.info.render/compute.timestamp (Engine.ts:241-242).
  - _fallback_: GpuProfiler is null when absent (Engine.ts:184); we fall back to whole-frame wall timing, which our notes already treat as ground truth.
  - _usefulToUs_: high
  - _weWereWrong_: false
- **timestamp-query-inside-passes**
  - _chromeStatus_: STABLE (separate feature from timestamp-query; landed in the Chrome 121-ish window, long stable by 150). Enables encoder.writeTimestamp() at arbitrary points INSIDE a render/compute pass, not just at pass boundaries.
  - _detect_: adapter.features.has('timestamp-query-inside-passes'); requiredFeatures:['timestamp-query-inside-passes']; then computePass.writeTimestamp(querySet, i).
  - _whereApplies_: Would let us time SUB-portions of the single world1 compute pass — e.g. split the ~7.8ms eye / 18.4ms oblique pixel loop (task #62) into coverage-walk vs election vs depth-interp WITHOUT the `relect` build-time ablation hack (NaniteRaster.ts:374). Currently we can only time whole passes, which is why the election split needed an ablation rebuild. Requires bypassing three's pass encoder to call writeTimestamp manually.
  - _threejsTSL_: NOT reachable through TSL/three's timestamp plumbing (three only does boundary timestampWrites); needs raw encoder access to the compute pass.
  - _fallback_: Keep the `relect`/ablation-rebuild attribution method we already use.
  - _usefulToUs_: medium
  - _weWereWrong_: false
- **shader-f16 (WGSL enable f16 — half-precision f16 type)**
  - _chromeStatus_: STABLE since Chrome 120 (developer.chrome.com/blog/new-in-webgpu-120; chromestatus 5180552617656320). Long stable well before 150. NOTE: 'f16 with subgroups' additionally needs BOTH shader-f16 AND subgroups (that combo is the Chrome 149-150 item), but plain f16 has been available since 120.
  - _detect_: adapter.features.has('shader-f16'); requiredFeatures:['shader-f16']; WGSL directive `enable f16;` then use f16/vec4<f16>.
  - _whereApplies_: Our SELF-DECLARED bottleneck is register/occupancy pressure on M1 Max TBDR. f16 for intermediates in the two heaviest per-pixel loops halves register footprint → higher occupancy: (1) the deferred resolve übershader (NaniteResolve — material+sun+CSM+IBL+GI accumulation), (2) the grass raymarch accumulators/step state (src/nanite/NaniteGrass.ts). Also f16 storage for HZB / intermediate G-buffer-ish channels. This is the single most promising NEW-feature lever for the Apple/mobile arc (task #72).
  - _threejsTSL_: NOT natively expressed by three r184 TSL — TSL nodes are f32; there is no f16 type node. Reaching it means raw-WGSL FunctionNode blocks with `enable f16;` for the hot inner math, and ensuring the device requiredFeatures includes 'shader-f16' (three may already auto-enable it if the adapter reports it — verify via window diag.features). Precision/artifact validation required (f16 range ~65504, mantissa 10 bits) — gate behind a param and shotdiff.
  - _fallback_: Current all-f32 shaders remain the default and the else-branch when adapter.features.has('shader-f16') is false (Firefox/older Safari).
  - _usefulToUs_: high
  - _weWereWrong_: true
- **subgroups (WGSL enable subgroups — subgroupMax/Min/Add/Ballot/Broadcast)**
  - _chromeStatus_: STABLE since Chrome 134 (after the Chrome 128-131 origin trial). The old 'chromium-experimental-subgroups' name was REMOVED; the shipping feature is 'subgroups'. WGSL subgroup_id/num_subgroups added Chrome 144; subgroup_uniformity extension Chrome 145.
  - _detect_: adapter.features.has('subgroups'); requiredFeatures:['subgroups']; read adapterInfo.subgroupMinSize/subgroupMaxSize; WGSL uses subgroup built-ins (subgroupMax, subgroupBallot, subgroupBroadcast, etc.). Optional: wgslLanguageFeatures.has('subgroup_id') + `requires subgroup_id;` for subgroup_id/num_subgroups (Chrome 144).
  - _whereApplies_: Directly attacks two named bottlenecks. (1) ELECTION atomic contention under dense overlap (NaniteRaster.ts:374-393 election): pre-reduce the depth key across the subgroup with subgroupMax BEFORE the atomicMax, so only the subgroup-winning lane hits the atomic — cuts atomic RMW traffic under the dense-overlap contention we call out. (2) VOXEL-BRICK SCATTER runs 128-lane workgroups (per the renderer description) — subgroup reductions/ballot replace shared-memory scans for occupancy/compaction. Also grass-tile coherence.
  - _threejsTSL_: NOT reachable through three r184 TSL (no subgroup nodes). Needs raw-WGSL FunctionNode for the election/scatter inner loops + device requiredFeatures:['subgroups']. Subgroup SIZE varies by hardware (Apple 32, others 32/64) so code must be size-agnostic.
  - _fallback_: Plain per-lane atomicMax election / shared-memory scatter (current default) when adapter.features.has('subgroups') is false.
  - _usefulToUs_: high
  - _weWereWrong_: true
- **immediates (WGSL immediate_address_space — push/root constants) + setImmediates()**
  - _chromeStatus_: SHIPPED STABLE in Chrome 149-150 (the headline 149-150 item; webgpu.com/news/chrome-149-150-webgpu-immediates). Small frequently-changing data passed straight to the shader, bypassing UBO creation + bind-group churn.
  - _detect_: navigator.gpu.wgslLanguageFeatures.has('immediate_address_space'); WGSL `requires immediate_address_space;` + a var in the `<immediate>` address space; JS calls pass.setImmediates(...) before the draw/dispatch.
  - _whereApplies_: Per-pass fast-changing scalars we currently ferry through UBOs/uniform nodes — cull/LOD params, per-cascade shadow constants, raster consts, the many URL-param knobs bound each frame. Cheaper than rebuilding a uniform buffer + bind group per pass. Marginal vs our real per-pixel-loop cost, so medium not high.
  - _threejsTSL_: NOT reachable through TSL — three r184 manages uniforms as UBO-backed uniform nodes and has no <immediate> address-space node; would need raw-WGSL + manual pass.setImmediates, i.e. bypassing three's uniform system. Low ROI relative to f16/subgroups.
  - _fallback_: Keep UBO/uniform-node path (works everywhere; Safari/Firefox have no immediates).
  - _usefulToUs_: medium
  - _weWereWrong_: false
- **transient / memoryless attachments (GPUTextureUsage.TRANSIENT_ATTACHMENT)**
  - _chromeStatus_: Shipped, with stricter validation refined in Chrome 149-150. Lets depth-stencil / MSAA render targets stay in on-chip TILE memory and never allocate main VRAM.
  - _detect_: Not an adapter.features flag — it's a GPUTextureUsage bit. Feature-detect by capability/try: create the texture with usage TRANSIENT_ATTACHMENT | RENDER_ATTACHMENT and honor validation rules (no COPY/STORAGE/sampling, load/store must be clear/discard).
  - _whereApplies_: Aimed squarely at TBDR (M1 Max / mobile — task #72): any transient render-attachment (a depth target, MSAA target) becomes memoryless, saving bandwidth+VRAM. BUT our renderer is compute-vis-buffer + storage buffers, not classic render attachments, so applicability is narrow — mainly any HW render-attachment passes (HW election reference NaniteHwRef, or the final resolve target if attachment-based) and shadow depth targets that are cleared each frame.
  - _threejsTSL_: three owns render-target allocation/usages; setting TRANSIENT_ATTACHMENT means a three RenderTarget/usage change, not a TSL change, and only where the target is truly write-then-discard within a frame.
  - _fallback_: Normal VRAM-backed attachments (universal).
  - _usefulToUs_: medium
  - _weWereWrong_: false
- **float32-filterable**
  - _chromeStatus_: STABLE since Chrome 119 (developer.chrome.com/blog/new-in-webgpu-119). Allows linear filtering of r32float/rg32float/rgba32float textures.
  - _detect_: adapter.features.has('float32-filterable'); requiredFeatures:['float32-filterable'].
  - _whereApplies_: Hardware bilinear on our 32-bit-float maps instead of manual 4-tap: terrain heightmap sampling (grass/voxel conform), HZB / depth-derived float targets, GI probe float textures. Removes hand-rolled bilinear in those fetch-heavy paths. three likely already requests it when the adapter reports it (check diag.features) — if a sampler on an rgba32float target uses 'linear', it depends on this being enabled.
  - _threejsTSL_: Reachable — set texture type to float + sampler linear in three; the device feature must be enabled (three auto-includes when available). Confirm via window diag.features rather than assuming.
  - _fallback_: Nearest sampling + manual bilinear in WGSL (current), or use rgba16float which is always filterable.
  - _usefulToUs_: medium
  - _weWereWrong_: false
- **dual-source-blending (WGSL @blend_src)**
  - _chromeStatus_: STABLE since Chrome 130 (chromestatus 5167711051841536).
  - _detect_: adapter.features.has('dual-source-blending'); requiredFeatures:['dual-source-blending']; WGSL @blend_src(0)/@blend_src(1) outputs.
  - _whereApplies_: Dual-source blending is for subpixel/colored-coverage alpha blending (e.g. text, ClearType-style). Our renderer is OPAQUE-ONLY, deferred, no blend stage — no applicable site. Listed only to close the question.
  - _threejsTSL_: N/A — not needed.
  - _fallback_: N/A (opaque renderer).
  - _usefulToUs_: no
  - _weWereWrong_: false
- **subgroup-matrix (Dawn chromium_experimental_subgroup_matrix)**
  - _chromeStatus_: EXPERIMENTAL, Dawn-only, behind the 'chromium_experimental_subgroup_matrix' Dawn toggle (requires the enable-unsafe-webgpu path). NOT a stable/OT web feature by 150. Targets tensor/XMX matmul (ONNX Runtime WebGPU detects it for Intel XMX). Related 'subgroup matrices' is on the 'next for WebGPU' consideration list.
  - _detect_: Only usable when the Dawn experimental toggle is on; would surface as a chromium-experimental feature/WGSL extension. Not detectable in normal stable Chrome.
  - _whereApplies_: Fixed-size matrix-multiply-accumulate hardware — an AI/inference feature. Our renderer has no dense matmul inner loop (we are per-pixel raster + fetch loops), so no application. Excluded.
  - _threejsTSL_: N/A.
  - _fallback_: N/A.
  - _usefulToUs_: no
  - _weWereWrong_: false
- **multi-draw-indirect + bindless (both NOT shipped)**
  - _chromeStatus_: NOT SHIPPED as of Chrome 150 — both are on developer.chrome.com/blog/next-for-webgpu as prioritized-but-future. multi-draw-indirect = one GPU-computed buffer drives many draws; bindless = large unsized descriptor arrays ('highly anticipated, prerequisite for leading-edge rendering'). No stable/flag path yet.
  - _detect_: No feature string yet; adapter.features.has(...) false. Gate + fallback when they land.
  - _whereApplies_: If they ship: bindless would let the resolve übershader index the whole bark/foliage texture-array set without the 24-sampled-texture juggling we raised maxSampledTexturesPerShaderStage to 24 for (Diagnostics.ts:40); multi-draw-indirect mainly helps the HW-election/reference path, less so our compute-scatter core. Track for a future arc; nothing to build now.
  - _threejsTSL_: Not reachable (not shipped; three has no node).
  - _fallback_: Current fixed bind groups + compute scatter.
  - _usefulToUs_: medium
  - _weWereWrong_: false

---

## (agent)

Wrote `/Users/sebastian/IdeaProjects/fable-demo2/docs/mobile-gpu-perf/WEBGPU-FEATURES.md` (full feature table + corrections + ranked candidates + detection pattern). All file:line refs verified against the codebase and three r184 internals.

## CORRECTIONS (amends `MASTER-AUDIT.md:209-215` "BLOCKED BY WEBGPU")

The load-bearing discovery: three r184 `WebGPUBackend.js:204-219` **auto-requests every adapter feature** before `requestDevice`, and `Engine.ts` passes no `requiredFeatures`. So on our M1 Max these features are **already enabled on-device** — the "device lacks the feature (`Engine.ts:86`)" framing in `MASTER-AUDIT.md:156-157/215` is wrong. The real blocker was only *authoring*.

1. **`shader-f16` — NOT blocked. STABLE since Chrome 120, already enabled on our device.** The only gap is that TSL has no f16 node → hot loop must be raw WGSL (`enable f16;`) with f32 buffers at the boundary. Reclassify "blocked-by-WebGPU" → "available; needs raw-WGSL authoring + shotdiff." Honesty caveat (`MASTER-AUDIT.md:36`): M1 Max fp16 is **not double-rate** — the win is register/occupancy, not FLOPs; must be measured.
2. **Subgroups — NOT blocked. STABLE since Chrome 134, enabled on-device, AND reachable from TSL today** (`subgroupBallot/BroadcastFirst/ExclusiveAdd/…` exported from `three/tsl` `SubgroupFunctionNode.js:210+`; directive auto-emits `WGSLNodeBuilder.js:1643-1646`). No raw WGSL, no `Engine.ts` change. Previously treated as unreachable.
3. **Immediates — NOT blocked. STABLE in Chrome 149-150.** Availability corrected, but priority stays LOW (three uses UBOs not SSBOs; `setImmediates` is three-owned, TSL-unreachable → marginal on our stack).
4. Also available, not blocked: **`float32-filterable`** (STABLE 119, likely already auto-enabled), **`timestamp-query-inside-passes`**.
5. Genuinely still unavailable (keep blocked): **64-bit buffer/texture atomics** (proposal only — our packed 2-atomic election stays default), **bindless**, **multi-draw-indirect**, **subgroup-matrix**.

## TOP 3 ADOPT CANDIDATES

**#1 Subgroup-aggregated queue compaction in cull** (HIGH, TSL-native, lowest risk). Replace per-thread global `atomicAdd` append into `qRaster`/vox counters in `src/nanite/NaniteCull.ts` (`kVoxFanout`/`kVoxCount`/`kVoxScatter`, queue `:480+`) with ballot+exclusive-prefix append → ~32× fewer global-counter atomics on Apple (subgroup 32), bit-identical. Guard `renderer.hasFeature('subgroups')`; fallback = shipped `atomicAdd` path behind a `?sg` flag. NOTE: do **not** target the per-pixel election (`NaniteRaster.ts:374`) — `relect` already measured election contention dead (RASTER arc CLOSED `62d607e`).

**#2 f16 scratch in grass raymarch + resolve BRDF** (HIGH, needs raw-WGSL, register-gated). Halves VGPR/value → occupancy on TBDR. Plugs into `NaniteGrass.ts` kGrassRay march state (task G-C #55), then `NaniteResolve` BRDF scratch, then voxel-scatter temporaries — never the election key. Guard `adapter.features.has('shader-f16')`; author hot block as raw-WGSL `FunctionNode` (`enableShaderF16()` `WGSLNodeBuilder.js:1517`), f32 buffers at the boundary, ship both variants. First step: capture to confirm register-bound before the rewrite; shotdiff-gate (demote only tile-local [0,1] scratch, not world-anchored positions).

**#3 TRANSIENT_ATTACHMENT (memoryless) on render-attachment passes** (MED, TBDR bandwidth). Write-then-discard shadow-depth/resolve/post scratch stays in on-chip tile memory, never allocates VRAM. Guard `'TRANSIENT_ATTACHMENT' in GPUTextureUsage`; fallback = normal attachment. Narrow (only render-attachment parts, not the storage-buffer vis-buffer) and needs a three RenderTarget/`WebGPUBackend` patch to surface the usage bit — hence #3.
