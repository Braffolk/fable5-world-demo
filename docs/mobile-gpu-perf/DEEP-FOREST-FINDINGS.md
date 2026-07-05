# Deep-forest full-renderer audit — per-subsystem detailed findings

_Per-agent detailed findings, extracted from the workflow run for cross-session reference._


---

## (agent)

_scope: Vis-buffer CLEAR + buffer layout / bandwidth (c.nanVisClear ~11ms)_  ·  est share: Intrinsic clear cost is ~0.5-1ms of the 20.8ms base, NOT ~11ms. Grounding: the world path clears only 2 full-screen u32 buffers (payloadV + visBV; depthV clear is already skipped, NaniteRaster.ts:524-529). At dpr2 ~3.3M px (comment at :519, alloc size.x*size.y NaniteFrame.ts:219) that is 2 x 3.3M x 4B = 26.4MB of pure store traffic. M1 Max ~400GB/s => ~66us theoretical, <1ms even at 10% efficiency. An 11ms timestamp is ~150x the memset ceiling, so the timestamp is NOT the clear's ALU/bandwidth cost. It is almost certainly a concurrency-serialization / submit-latency artifact: kVisClear is the FIRST dispatch in the coalesced world1 submit (NaniteFrame.ts:610-628) and its GPU-timestamp window absorbs the wait for the PREVIOUS frame's fragment resolve, which reads the very buffers the clear overwrites (WAR hazard, see findings). Confirming needs the no-op ablation (profiling-todo); until then treat the clear's real share as sub-1ms and the 11ms as overlap/stall, per the task's own note that per-pass timestamps overlap and ablation is truth.

- **The ~11ms c.nanVisClear is a concurrency/timestamp artifact, not real clear cost**
  - _kind_: confirm-refute-hypothesis
  - _source_: apple-tbdr-blakecrosley.md:10,31 (render and compute are not different worlds; passes overlap on HW; Apple runs render||compute concurrently) + the task's own note that per-pass timestamps overlap and are non-additive on Apple
  - _ourCode_: src/nanite/NaniteRaster.ts:543-563 (kVisClear) writes visPayloadV+visBV; src/nanite/NaniteResolve.ts:424-425 (previous frame's FRAGMENT resolve reads vis.visBV.ro + vis.payloadV.ro at every pixel)
  - _mechanism_: Memset math falsifies the 11ms: 2 buffers x 3.3M px x 4B = 26.4MB store => ~66us at 400GB/s, <1ms at 10% efficiency; 11ms is 150x impossible for a trivial 2-atomicStore kernel. The real inflation: kVisClear runs on the COMPUTE queue and overwrites payloadV/visBV, but the prior frame's deferred RESOLVE (FRAGMENT queue) reads those same buffers. Apple executes render||compute concurrently, so the driver must serialize the clear behind last frame's resolve (write-after-read hazard on shared buffers) => the clear's timestamp swallows that stall AND destroys the render||compute overlap the architecture is built to exploit.
  - _fix_: Confirm by ablation (see profiling-todo): build-time no-op the payloadV/visBV clears and A/B gpuWall. Expected: base drops far less than 11ms (proving the timestamp lied) OR drops a lot (proving the WAR stall) — either way it disambiguates. Then apply the ping-pong fix below.
  - _webgpuFeasible_: yes
  - _impact_: high
  - _confidence_: speculative
- **Double-buffer (ping-pong) the vis buffers to restore render||compute overlap**
  - _kind_: trick-to-steal
  - _source_: apple-tbdr-blakecrosley.md:10,31 (overlap two phases on HW; render+compute one timeline) — honoring concurrency requires no cross-queue WAR hazard on shared storage
  - _ourCode_: src/nanite/NaniteRaster.ts:203-218 (makeVisBuffers — single instance) + NaniteFrame.ts:219 (one vis created) + NaniteResolve.ts:424-425 (resolve reads same buffers next frame's clear writes)
  - _mechanism_: With two vis buffer sets (A/B) swapped per frame, clear+raster(N+1) write set B while resolve(N) still reads set A. No shared-buffer WAR hazard => the compute clear/raster no longer serializes behind the previous fragment resolve, and Apple can actually overlap them. Removes the stall that inflates c.nanVisClear.
  - _fix_: Allocate 2x NaniteVisBuffers, index by frame parity, pass the current set into world1/resolve. Cost = +26MB device memory (2x8B/px hot pair), trivial vs the 6GB heap. Pure WebGPU (extra StorageBufferAttributes).
  - _webgpuFeasible_: yes
  - _impact_: high
  - _confidence_: speculative
- **Replace the atomicStore compute-clear with native GPUCommandEncoder.clearBuffer (zero-fill DMA)**
  - _kind_: trick-to-steal
  - _source_: arm-mali-guide.md:147 (avoid writing surfaces you can skip) + Aaltonen/Apple bandwidth principle: a driver zero-fill is a DMA fast-path, cheaper than a per-thread atomicStore dispatch
  - _ourCode_: src/nanite/NaniteRaster.ts:548-549 (both world-path clears are to ZERO: packedClear=>payload 0, visBV 0) and NaniteResolve.ts:426 (elect.equal(0) => discard to sky, so 0 IS the required sentinel)
  - _mechanism_: Both hot clears target zero, which exactly matches WebGPU clearBuffer (zero-only). A native buffer clear is a hardware fill with no compute dispatch, no per-thread atomic, no workgroup launch (12900 wgs today). The tiny counter resets (hwQueue/audit/scar at instanceIndex==0) stay as a 1-thread dispatch.
  - _fix_: Emit commandEncoder.clearBuffer(payloadAttr,0,size) + clearBuffer(visBAttr,...) instead of kVisClear's full-screen loop. Feasibility caveat: three r184 WebGPUBackend does not obviously expose the raw GPUCommandEncoder mid-frame; needs a backend reach-through or a three patch. Lower priority than ping-pong since the clear itself is already cheap.
  - _webgpuFeasible_: partial
  - _impact_: low
  - _confidence_: speculative
- **depthV clear already skipped in the world path — do not regress**
  - _kind_: what-we-do-right
  - _source_: Burns-Hunt vis-buffer chosen for LOW bandwidth; task constraint 'no-3rd-atomic-buffer is CORRECT for Apple, do not regress'
  - _ourCode_: src/nanite/NaniteRaster.ts:524-529 (skipDepthClear) + 545 (guarded) + 570 comment ('No exact depthV — a 3rd hot-loop atomic buffer = a 3x cliff')
  - _mechanism_: Only 2 of 3 vis buffers (payloadV+visBV) are cleared/written in the world hot path; the 3.3M-px 0xffffffff depthV clear (a full 13MB store) is elided because nothing in the world path reads depthV (it is shadow/audit-only). This already honors the vis-buffer low-bandwidth rationale. Re-adding a depthV clear or a 3rd hot atomic would be a real regression.
  - _fix_: None — keep skipDepthClear default-on. Any future work must preserve the 2-buffer hot set and the atomicMax-into-payload / plain-store-into-visBV split.
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: confident
- **depthV is allocated full dpr2 but is dead weight in the world path (memory, not bandwidth)**
  - _kind_: violation
  - _source_: arm-mali-guide.md:41,147 (do not keep/write surfaces you do not consume) + Apple bandwidth budget for full-screen storage
  - _ourCode_: src/nanite/NaniteRaster.ts:203-218 (makeVisBuffers allocates depthAttr for every instance) — world path never reads it (:517-518 exhaustive consumer note); only shadow (singlePass=false) and ?audit/?nanprobe consume depthV
  - _mechanism_: depthV = 4B/px x 3.3M = 13.2MB resident that the product world path never touches. It is not bandwidth (skipped clear, no world writes) but it is dead memory during the deep-forest arc's memory pressure. The 12B/px 3-buffer layout is near-minimal otherwise: election key + full id cannot merge into one atomic (WGSL has no atomic64), so the payloadV/visBV split is forced and correct.
  - _fix_: Lazily allocate depthV only when a depthV consumer is active (shadow cascade instance, ?audit, ?nanprobe, rdbg). World-only product runs then hold 8B/px not 12B/px. Pure allocation change; no shader edits. Low perf impact, small memory win.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: speculative
- **Measure the TRUE clear cost: no-op ablation + ping-pong A/B**
  - _kind_: profiling-todo
  - _source_: CLAUDE.md premise-audit + task note 'ablation is the truth' (overlapping per-pass timestamps cannot attribute cost)
  - _ourCode_: src/nanite/NaniteRaster.ts:524-563 (extend the existing build-time skip gate) + NaniteFrame.ts:610-628 (coalesced submit where the clear is the first dispatch)
  - _mechanism_: The 11ms timestamp cannot be trusted (overlaps concurrent raster + absorbs the WAR stall). Two ablations separate the hypotheses: (a) build-time no-op the payloadV/visBV clears (accept 1 dirty frame) and diff honest gpuWall — a tiny delta proves the clear is intrinsically cheap and the 11ms is artifact; (b) implement ping-pong and diff — a large drop proves the WAR concurrency stall was real. Run both at the same dense-forest pose dpr2 with grass OFF (the 20.8ms base).
  - _fix_: Add ?dvclear2=0 build-time gate to skip the two hot clears; add a ?visping flag to swap A/B vis sets. Capture gpuWall A/B/A within one session (cross-boot contamination law).
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: confident

---

## (agent)

_scope: Cull BFS DAG descent + per-cluster prev-HZB occlusion + HZB mip-chain build (deep-forest camera + shadow-shared culls)_  ·  est share: ~2-3 ms of the 20.8 ms base (~10-14%), and it is the SMALLEST of the flagged whales — grounded three ways. (1) Absence-in-the-flagged-list: nanVisClear(11), r.half.mrt(10.5), r.scene(9.6), r.nanHwPass, c.grassRay(4) were the big timestamped passes; the cull passes (nanSeedRoots/nanTraverseAB/BA) and HZB passes (nanHzbL0..15) are ALSO labelled+timestamped yet appear in NONE of them, so each individually sits below ~4 ms. (2) Code reasoning: the only per-pixel-scaled pass in the whole subsystem is HZB level-0 (compute over ceil(W/2)·ceil(H/2) ≈ 835k threads @dpr2 retina, 4 strided u32 loads of visPayloadV each, NaniteHzb.ts:150-164) — plausibly the single largest item at ~0.5-1 ms; every other HZB level is <210k→…→1 threads (negligible). (3) The BFS itself is thread-count-bounded by the frontier, which even in dense forest is orders of magnitude below the per-pixel passes (millions of pixels vs ~10^5-10^6 frontier items across all passes). Two full BFS culls run when shadows are active (camera + NaniteClipCull shared cut), so double the BFS half. Caveat: NO ablation exists (deep-forest-breakdown.md was absent), so the split is SPECULATIVE; the headline verdict — this subsystem is not the drop, do not over-invest — is CONFIDENT.

- **BFS + HZB already coalesced into one submit — do not regress**
  - _kind_: what-we-do-right
  - _source_: MASTER-AUDIT.md 'RECOMMENDED SEQUENCE' + webgpu-wgsl three.js #32735 (per-submit/pipeline overhead on Apple); Karis Nanite GPU-driven single-pass persistent cull
  - _ourCode_: src/nanite/NaniteFrame.ts:582-587 (phase1Batch+fullArgs+voxFanout one dispatchBatchMixed), :620-636 (foldHzb folds hzb.batch() into the raster submit); src/nanite/NaniteHzb.ts:229-234 (build = ONE dispatchBatch over all mip kernels)
  - _mechanism_: Apple/Dawn pays real per-queue.submit drain + barrier cost. The camera cull is ~2+2·(maxDagDepth+2)+1 ≈ 33 dispatches and the HZB is up to 16 dispatches; folding them into single submits (from the header's noted ~38 separate drains) removes dozens of submit boundaries per frame on TBDR. This is the correct structural choice and the reason cull/HZB never surface as flagged passes.
  - _fix_: None — keep coalesce/foldHzb default-on. Any future cull/HZB edit MUST stay inside the batched lists; adding a standalone dispatch/readback mid-frame reintroduces a drain.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident
- **hierDepth = measured maxDagDepth+2 and [64] workgroups — correct occupancy/pass-count choices**
  - _kind_: what-we-do-right
  - _source_: occupancy-interplayoflight.md (fit workgroup to simd multiple; shed empty work) + MASTER-AUDIT.md note that [64] vs [256] is only a capture-gated A/B, shared-mem not a violation on Apple
  - _ourCode_: src/nanite/NaniteFrame.ts:230 (measuredHierDepth = maxDagDepth+2); src/nanite/NaniteCull.ts:337-344 (uses it, not the legacy constant 18), :990/:1173 traverse compute([64]); src/nanite/NaniteHzb.ts:185 compute([64])
  - _mechanism_: [64] = 2 Apple 32-wide simdgroups per workgroup, good residency without over-subscribing threadgroup memory. Using the measured DAG depth (~13) + 2 instead of a fixed 18 sheds ~2×5=10 empty-tail dispatch/barrier pairs per frame per cull (paid twice: camera + shadow). Deep forest = deep DAG, so this pass-count trim matters most exactly here.
  - _fix_: None. Do not raise hierDepth speculatively; if maxDagDepth grows with new aggregate ladders, keep it exact — every extra pass is 2 dispatches × 2 culls.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident
- **HZB builds up to 16 distinct compiled pipelines; the low-level fuser (?pyrfuse) is DEFAULT-OFF**
  - _kind_: violation
  - _source_: webgpu-wgsl three.js #32735 (pipeline caching: every distinct compute pipeline is a separate WGSL→MSL XPC compile on Apple) + MASTER-AUDIT.md:225-228 ('HZB up to 16 distinct ... belongs in the BOOT arc')
  - _ourCode_: src/nanite/NaniteHzb.ts:139-188 (one Fn per level with baked w/h/offset → distinct pipeline), :133 pyrfuse default '1'!=='1'→OFF, :189-227 kFusedTail exists but only wired when pyrfuse on
  - _mechanism_: Each level bakes info.w/h/offset as constants → 16 unique kernels → 16 boot compiles AND 16 compute-pipeline switches per frame. The bottom ~9 levels are ≤1024 texels (near-empty dispatch + barrier each). Frame cost of the switches is minor (~sub-ms), but the boot-compile cost is real on Apple and ties directly to the pipeline-caching issue. This is primarily a BOOT lever, not the 20.8 ms frame drop.
  - _fix_: Flip ?pyrfuse default ON (kFusedTail already bit-identical, NaniteHzb.ts:189-227) to collapse the tail ≤1024-texel levels into ONE single-workgroup kernel — cuts ~8 pipelines + ~8 per-frame dispatch/barrier boundaries. Better still: make the per-level kernel read w/h/offset from the existing uniform `table` (NaniteHzb.ts:113-118) so ALL levels share ONE pipeline (loop levels with storageBarrier is impossible across the full pyramid, but a single parameterised kernel dispatched per level removes 15 compiles). Bench on BOOT pipeline-creation time, not frame.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident
- **HZB level-0 is the only per-pixel-scaled pass in the subsystem — measure its true share before touching anything else**
  - _kind_: profiling-todo
  - _source_: arm-mali-guide.md (route hot strided reads through the texture cache; avoid divergent SSBO addresses) + MASTER-AUDIT.md Phase-0 (instrument via ?nanprobeat + per-pass timestamps already wired)
  - _ourCode_: src/nanite/NaniteHzb.ts:150-164 (level 0: compute over ceil(W/2)·ceil(H/2) threads, 4 clamped strided elemU(visDepthRO) loads + decode each), dispatch :185; probe hook src/nanite/NaniteFrame.ts:638 (?nanprobeat=hzb)
  - _mechanism_: At dpr2 retina (~2268×1473) level-0 is ~835k threads each doing a 2×2-strided read of the u32 visPayloadV storage buffer — the same dynamic-SSBO / strided-access pattern Aaltonen flags as the mobile weakness, and the only piece of my subsystem that scales with the per-pixel count that dominates the frame. Everything below it is <210k→1 threads.
  - _fix_: PROFILE FIRST: run ?nanprobeat=hzb and read the nanHzbL0 timestamp at the dense pose, and A/B by temporarily emitting the pyramid at quarter-res (skip a level). If L0 is >~0.7 ms, the strided visPayloadV read is the target — but it reads the raster's own output buffer, so a texture route means the raster must also write a texture (cross-subsystem; do not do unilaterally). If L0 is <0.5 ms, declare the whole subsystem closed and move effort to clear/resolve/half-mrt.
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: speculative
- **kTraverse is a mega-kernel — worst-case VGPR (frustum+cone+HZB-mat-mul+lodWarp pow) reserved for the cheap descend-majority lanes**
  - _kind_: confirm-refute-hypothesis
  - _source_: occupancy-ilp / MASTER-AUDIT.md lever-4 (Andersson/DICE: a branch's worst-case VGPR+LDS is reserved across ALL lanes; splitting the mega-shader restores occupancy — the '5→2 waves for free' anti-pattern)
  - _ourCode_: src/nanite/NaniteCull.ts:1002-1172 — one kernel: readDag+2 instance vec4 (:1009-1016), lodWarp with a pow (:1039 → :138), then the EMIT branch's frustum 6-plane loop (:1094→:549-559), cone cull (:1118-1141), sphereOccluded = mat4 mul + 4 HZB reads (:1144-1150), vs the ELSE branch that only enqueues children (:1160-1171)
  - _mechanism_: In DEEP forest the vast majority of traverse threads are interior DAG nodes that DESCEND (cheap enqueue-children path); only the frontier's final-level survivors take the expensive emit path. But the compiler reserves the emit path's full VGPR footprint (matrix mul + frustum loop + cone temporaries) for every lane, capping occupancy of the whole traverse by the emit worst-case — the exact anti-pattern that hurt grass's L2 block. Latency-exposed dependent SSBO chain (frontier→instance→dag→cluster) per item compounds it.
  - _fix_: GATED on a Metal capture reading kTraverseAB/BA VGPR + occupancy (same capture the grass audit demands). IF register-bound: the descend path can't be trivially split from emit (BFS needs both in one visit), but the occlusion+cone+frustum could be deferred — emit a RAW candidate to qRaster in the traverse, then run frustum/cone/HZB reject as a separate tight kernel over the emitted list (moves the heavy VGPR out of the deep descend passes). Higher complexity; do NOT build before the capture confirms register-bound.
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: speculative
- **kSeedRoots runs one thread per WORLD instance every frame — O(instanceCount) even for envelope-culled far trees**
  - _kind_: confirm-refute-hypothesis
  - _source_: Karis Nanite GPU-driven (instance cull is per-instance but Nanite pre-buckets via a persistent/hierarchical instance list) + Aaltonen no-graphics-api (process at the right granularity / cull coarse before fine)
  - _ourCode_: src/nanite/NaniteCull.ts:923-990 kSeedRoots compute(instanceCount,[64]); each thread loads 2 instance vec4 + mesh words, does draw-envelope + nearDist + frustum 6-plane + instMinPx size test (:934-980) before appending roots
  - _mechanism_: With WORLD-WIDE placement (200k+ trees, plus grass patches + fartiles) every instance spawns a thread every frame just to be frustum/envelope-rejected. instMinPx (~110px @dpr2, NaniteFrame.ts:212) and the lodDist envelope DO drop far instances before they flood the frontier (good — mitigates the BFS side), but the per-instance TEST itself is still paid for all N. In deep forest N is largest, so this pass grows precisely with the trigger.
  - _fix_: PROFILE: read the nanSeedRoots timestamp at the dense pose. If it is a real share (>~0.5 ms), add a coarse CPU/GPU spatial pre-bucket (grid or BVH over instance AABBs) so seedRoots only visits instances in a few frustum-overlapping cells — Nanite's instance-hierarchy step. If seedRoots is <0.3 ms, refute and leave it. Likely a minor share; verify before building the grid.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: speculative
- **lodWarp evaluates a pow() per frontier item per BFS pass even when the warp is disabled**
  - _kind_: trick-to-steal
  - _source_: occupancy-ilp / MASTER-AUDIT.md lever-1 (replace transcendentals on the hot path with precomputed/cheaper values; Apple pow/sqrt are multi-cycle) — the same 'kill the per-step pow' move applied to grass
  - _ourCode_: src/nanite/NaniteCull.ts:1039-1045 (tauEff=lodWarp per traverse thread) → :125-140 lodWarp does `d.div(scale).pow(pow)` then `scale.greaterThan(0).select(...)` — TSL select evaluates BOTH operands, so the pow runs even when simBandD(scale)=0 (the camera default, NaniteFrame.ts:198)
  - _mechanism_: Every traverse thread, at every DAG level it descends, computes a pow it usually discards (world default simBandD=0 → warp off). A cluster near the frontier is visited ~maxDagDepth times, so it is ~13 discarded pow evals per cluster per frame across two culls. Cheap per-thread but it inflates the mega-kernel's ALU + a live temporary, feeding the occupancy issue above.
  - _fix_: Guard the whole lodWarp at BUILD time: when simBandD is a compile-time-known 0 (no ?simband/?lodnear), skip emitting the lodWarp node entirely and use tau directly (JS-side `if (simBandDisStatic0) tauEff = tau`). When lodPow===1 (linear, the common case), replace the pow with a plain multiply. Bit-identical when the warp is off; removes a transcendental + a live var from the hottest kernel.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident
- **Single-phase prev-frame-HZB occlusion at emit (no two-phase re-traverse) is the right Apple call**
  - _kind_: what-we-do-right
  - _source_: Karis Nanite (two-phase HZB) vs the header's PERF-VB3 note; occupancy (avoid a second full traverse) — MASTER-AUDIT.md 'single-pass' philosophy
  - _ourCode_: src/nanite/NaniteCull.ts:1142-1150 (sphereOccluded only on the visible>0.5 emit path, using cam.prevVp/prevCamPos), guarded so the mat-mul+4 reads never run for the descend majority; NaniteHzb.ts:237-304 conservative test
  - _mechanism_: The classic Nanite two-phase (cull vs prev-HZB, raster, rebuild HZB, re-cull the rejected set, raster again) doubles the traverse + adds a second raster pass. Our single-phase reuses last frame's pyramid at emit only — one BFS, one raster. For a static world (prev-frame occluders valid) this is strictly cheaper on TBDR and is why the cull never dominates. The occlusion test is correctly gated behind the frustum/cone visibility so descend-only lanes skip the matrix mul.
  - _fix_: None — keep single-phase. Do NOT port Nanite's two-phase re-cull to chase a few extra culled clusters; on Apple the second traverse + raster would cost more than the over-draw it saves. The 1-frame-stale occluder is acceptable (static world).
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident

---

## (agent)

_scope: Post stack (GTAO, screen-space bounce, contact shadows, clouds, froxel volumetrics) — DEEP-FOREST case_  ·  est share: ~5-6 ms of the 20.8ms deep-forest base (~25-30%). Grounded on: (1) the eye-pose ablation costmap in memory (post 5.5ms: AO 3.7, clouds+bounce+contact 2.3), plus (2) CODE reasoning that in the deep-forest pose the composition ROTATES toward its worst case — contact shadows (PostStack.ts:504) march full-res on every pixel because dist<240 everywhere, AO (aoFadeNear=90) loses its distance-fade skip because nearly all geometry is <90m, and bounce runs its 8-tap gather on every non-sky pixel — while clouds (Clouds.ts:294 valid gate) nearly vanish because tree hits are nearer than the 1250m slab. Net roughly cancels to ~5-6ms but with heavier contact/AO/bounce and near-zero clouds. This is NOT ablation-confirmed for the forest pose: the per-pass timestamp r.half.mrt ~10.5ms is a wall figure that OVERLAPS the concurrently-running SW raster + grass compute on Apple, so it overstates the isolated cost. A deep-forest-pose ablation (ablate=contact/ao/bounce/clouds/froxels legs, gpuWall delta) is required to firm the number and rank the fixes — filed as a profiling-todo finding.

- **Contact shadows run FULL-RES for the ENTIRE frame when surrounded by trees**
  - _kind_: violation
  - _source_: cloudscapes-optimisations-realtime-volumetric-toft2016.md:49-52 ("render at 1/2 res, reproject/bilateral upsample") + occupancy-interplayoflight.md (process at the right frequency); the same half-res+bilateral pattern we already ship for AO
  - _ourCode_: src/render/PostStack.ts:451-497 (contactNode, SSCS_STEPS=12) applied at :504 (aerialNode.mul(aoFaded).mul(contactNode))
  - _mechanism_: contactNode is evaluated per FULL-RES output pixel; its only skip is dist>=240m. At an elevated vista most pixels are far and skip the 12-step march, but in DEEP FOREST every pixel has a trunk/leaf within 240m so NOTHING skips — the whole frame pays a 12-step depth-buffer march = per pixel ~12 dependent texture(depthTex) fetches + 2 getViewPosition + getScreenPosition. On Apple TBDR dependent full-res texture marches are latency+bandwidth heavy and this is the single post cost that goes to its worst case precisely in the forest pose. First-hit early-out only helps where contact actually hits (lit gaps).
  - _fix_: Move contact into the existing half-res MRT pass as a 4th HalfResEntry (r8/rg16f), then bilateral-upsample it full-res reusing aoFaded's depth guide (the viewZ is already packed per-tap). Quarters the march invocations (~4x on the pixels that ALL pay in forest). Contact is already floored+distance-faded (a soft near-field cue) so half-res softening is within the quality bar. Cheaper fallback: drop SSCS_STEPS 12->8 with the quadratic distribution retained.
  - _webgpuFeasible_: yes
  - _impact_: high
  - _confidence_: confident
- **GTAO temporal reuse is DISABLED — the biggest single post cost never amortized across frames**
  - _kind_: trick-to-steal
  - _source_: cloudscapes-optimisations-realtime-volumetric-toft2016.md:43-45 + :51-52 (accumulate few-sample results over frames via TAA/reproject) + occupancy-interplayoflight.md (process at the right frequency); stock three GTAONode ships temporal-direction rotation which we ported but turned OFF
  - _ourCode_: src/render/Gtao.ts:11-12 ("temporal filtering stays off; _temporalDirection = 0") and :153 (angle has no per-frame rotation term); consumer PostStack.ts:224-234 (6 samples). We ALREADY have exact per-pixel motion vectors (PostStack.ts:547 velReproject) and TRAA history.
  - _mechanism_: AO is the largest post effect (~3.7ms eye-pose costmap) and in deep forest it hits its worst case: nearly all geometry sits inside the 90m full-AO band (PostStack aoFadeNear=90) so the far-fade early-out (Gtao.ts:118, PostStack.ts:443) saves nothing — the full 6-sample horizon march runs on every pixel. AO on static geometry is temporally stable, so rotating the 3-slice direction per frame and accumulating with the motion-vector history lets you halve per-frame samples (6->3) at equal converged quality. Fewer horizon-march samples = fewer of the ALU+dependent-fetch inner iterations, the exact TBDR-expensive part. No dither/noise added (bilateral-reprojected history, not screen-space noise).
  - _fix_: Add a per-frame rotation to the slice angle (Gtao.ts:153, feed frameU like stock _temporalDirection), render AO at 3 samples, add a small history texture in the half-res pass, reproject with velReproject and blend (neighborhood-clamped so disocclusion doesn't ghost). Reuses the analytic reprojection already built for TRAA.
  - _webgpuFeasible_: partial
  - _impact_: high
  - _confidence_: speculative
- **Screen-space bounce does an 8-tap gather on EVERY non-sky pixel for a x0.16 subtle effect**
  - _kind_: violation
  - _source_: occupancy-interplayoflight.md (right frequency / cost vs visible payoff); the effect is self-described "subtle by design" and composited at 0.16
  - _ourCode_: src/render/PostStack.ts:241-268 (bounceLayer, 8-tap loop, 2 texture fetches/tap: depthTex + beauty) composited at :515-517 (mul 0.16)
  - _mechanism_: Half-res but ungated beyond isSky — in deep forest every pixel is non-sky so all 8 taps × (depth + beauty) = 16 dependent texture fetches run per half-res pixel to add a green-on-trunk/warm-on-rock bleed multiplied by 0.16. Cost-to-visible-payoff is poor; the tap count is the lever.
  - _fix_: Drop the gather to 4 taps (the Fibonacci spiral degrades gracefully) — halves its fetch bandwidth for a barely-perceptible change at 0.16 weight. Or fold the depth read to reuse the AO pass's already-fetched half-res viewZ instead of a second depthTex sample per tap.
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: confident
- **Clouds are effectively FREE in deep forest — do not spend cloud-march effort for this pose**
  - _kind_: confirm-refute-hypothesis
  - _source_: cloudscapes-optimisations-realtime-volumetric-toft2016.md (the march is the cost) cross-checked against our slab-intersection gate
  - _ourCode_: src/sky/Clouds.ts:286-294 (tEnter from CLOUD_BOTTOM=1250m slab, valid gate) + PostStack.ts:199-212 (cloudLayer passes maxD=dist for geometry pixels)
  - _mechanism_: For any ray that hits a tree, maxD=dist(~<50m) so tExit=min(slabExit,dist) is small while tEnter(slab entry ~1250m/dir.y) is large => valid=false => the 32-step march (Clouds.ts:319) is skipped entirely. In deep forest almost every pixel hits geometry before the cloud slab, so the marched cloud cost the eye-pose costmap attributed (~part of the 2.3ms) is largely an OPEN-SKY-pose cost, not a forest cost. Residual is simd divergence: scattered sky pixels through leaf gaps force their simdgroup to run the march. HYPOTHESIS: clouds are <1ms in the forest pose.
  - _fix_: Profiling: ablate=clouds A/B at a dense-forest pose. If confirmed <1ms, DO NOT invest in cloud step-count/temporal work for the deep-forest mandate (it pays only at vistas). If divergence cost shows up, gating the half-res cloud entry to only-mostly-sky tiles would recover it, but only worth it if measured.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: speculative
- **Froxel scatter+integrate compute rebuilds the full 921k-voxel grid every frame regardless of view**
  - _kind_: profiling-todo
  - _source_: cloudscapes-optimisations-realtime-volumetric-toft2016.md:51-52 (update a fraction of cells/frame, reproject the rest) + occupancy-interplayoflight.md (right frequency)
  - _ourCode_: src/gpu/passes/Froxels.ts:62-64 (FX160×FY90×FZ64), :102-195 scatterK (each froxel does 5 terrain sampleHeightNearest + canopyAt + clouds.shadowAt + fbm), :199-230 integK, dispatched every frame at :234-240
  - _mechanism_: 921,600 scatter invocations each with 5 dependent height fetches + canopy + cloud-shadow + noise run every frame even though the fog field is temporally near-static (slow wind drift). This is COMPUTE and on Apple runs concurrently with the render wall, so it may be fully hidden — but if the render wall shrinks (from the contact/AO fixes above) this becomes exposed. The froxel apply() (PostStack.ts:308, aerialNode.assign) is a cheap trilinear per pixel and is fine.
  - _fix_: Profiling first: ablate=froxels A/B at a deep-forest pose to see if it's hidden under render. If exposed, reproject the previous-frame integrated grid and re-scatter only a fraction of froxels per frame (Bart-Wronski-style temporal froxel reprojection) or drop FZ 64->48. Only worth building if the ablation shows exposed cost.
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: speculative
- **What we already do right in the post stack — do not re-chase**
  - _kind_: what-we-do-right
  - _source_: cloudscapes-optimisations-realtime-volumetric-toft2016.md (analytic integration + jitter+TAA + half-res are THE tricks) + MASTER-AUDIT "WHAT WE ALREADY DO RIGHT" pattern
  - _ourCode_: src/render/HalfResMrt.ts (clouds+AO+bounce merged into ONE half-res raster, shared depth, one encoder); Gtao.ts:304 + PostStack.ts:222-234 (rg16f AO with viewZ packed for the bilateral guide, single fetch/tap); PostStack.ts:399-446 (joint-bilateral upsample + past-fade early-out); PostStack.ts:471-489 (contact first-hit early-out); Clouds.ts:296-345 (32-step half-res + per-frame jitter + analytic closed-form stepT exp(-σ·seg) + TRAA); PostStack.ts:567 traaPingPong (kills 2 full-res copies/frame)
  - _mechanism_: The three screen-space layers already share one half-res MRT raster over one depth buffer (the single biggest structural win), AO already packs viewZ to avoid re-unprojecting depth in the bilateral, clouds already have cloudscapes' analytic-integration + jitter + TAA (so the 128->8 step trick is largely spent), and every march has an early-out. These are correct Apple/TBDR choices; effort spent re-deriving them is wasted.
  - _fix_: No change — preserve these. Direct new effort at the deep-forest-specific amplifiers (contact full-res, AO no-fade, bounce ungated) instead.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident
- **Deep-forest post-share needs its own ablation legs (eye-pose costmap mis-weights the composition)**
  - _kind_: profiling-todo
  - _source_: occupancy-interplayoflight.md (measure at the real workload) + the project's ablation-is-truth law; the existing costmap was an eye/open pose
  - _ourCode_: PostStack.ts:84 ablate set already supports clouds/ao/bounce/contact; froxels via TerrainScene.ts:445 ablate.has('froxels'); recommend legs at a DEEP-FOREST pose dpr2
  - _mechanism_: The 5.5ms eye-pose costmap (AO 3.7 / clouds+bounce+contact 2.3) does NOT transfer: in deep forest contact goes full-res-all-pixels, AO loses its far-fade skip (all geometry <90m), bounce runs on every non-sky pixel, while clouds nearly vanish. The composition rotates toward contact+AO+bounce. Grounding the fix priority needs a forest-pose ablation, not the vista number.
  - _fix_: Run ablate=contact / ablate=ao / ablate=bounce / ablate=clouds / ablate=froxels one-at-a-time A/B/A at a dense-forest pose (grass already off), gpuWall frame delta. Predicts contact + AO dominate; confirm before building the half-res-contact and temporal-AO changes.
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: confident

---

## (agent)

_scope: Frame orchestration + WebGPU/WGSL cost (pipeline caching, dispatch/submit, SSBO vs UBO, timestamp/readback overhead) — DEEP FOREST base 20.8ms_  ·  est share: ~1.5–3ms of the 20.8ms is cleanly ORCHESTRATION-attributable (not intrinsic per-pixel work). Grounded three ways: (1) the ablation slabs (visClear ~11 / half.mrt ~10.5 / resolve ~9.6, OVERLAPPING) are per-pixel GPU-bound — the arc already established the frame is per-pixel-scaled (dpr1≈9ms), so CPU submit/encode overlaps GPU and is NOT the wall; (2) submit-coalesce already collapsed the frame to ~12–18 queue.submits (mostly render passes that CANNOT share an encoder), so residual submit/pipeline-switch driver overhead is small; (3) the one cleanly-separable orchestration cost is trackTimestamp (2 timestamp writes on every one of ~12–18 passes + a RENDER+COMPUTE resolve/copyBuffer/mapAsync EVERY frame), which I estimate ~0.5–1ms fixed but is UNMEASURED — flagged as the profiling-todo (A/B trackTimestamp on/off). The BIG WebGPU-layer tax — fragment-stage SSBO refetch in the resolve (3× fetchWorldVert/pixel) — is real but baked INTO the 9.6ms resolve slab and not independently attributable; it is structural (vis buffer must be storage). Net: pure orchestration is a MINOR slab; the 20.8ms is dominated by the per-pixel raster/resolve/shadow work owned by the other subsystems.

- **trackTimestamp is unconditionally ON in production + timestamps resolved EVERY frame**
  - _kind_: violation
  - _source_: threejsroadmap-profiling-webgpu-shaders.md: 'Enable (off by default, small cost)' and with trackTimestamp:true three 'automatically injects timestamp writes into begin/end of every compute and render pass it issues'; noisy-measurement section notes readback jitter.
  - _ourCode_: src/core/Engine.ts:86 (trackTimestamp:true, hardcoded, no param gate) + src/core/Engine.ts:231-251 (collectStats resolves TimestampQuery.RENDER + COMPUTE every single frame and runs GpuProfiler.collect)
  - _mechanism_: On Apple/Dawn each of the ~12–18 passes/frame gets a GPU query write at pass begin AND end, and every frame does two resolveTimestampsAsync (a querySet resolve = copyBufferToBuffer + submit + mapAsync). This is fixed per-frame GPU+CPU overhead paid in PRODUCTION where nobody reads the numbers. The profiling doc itself calls it opt-in with a cost.
  - _fix_: Gate trackTimestamp behind a ?gpuprof/?profile URL param (pass true only when profiling) and, when on, resolve on a cadence (every Nth frame) rather than every frame. Fully WGSL/WebGPU-reachable — it is a renderer constructor flag + a resolve-call cadence, no node_modules edit.
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: confident
- **Fragment-stage SSBO refetch: resolve re-reads 3 world verts per pixel from storage buffers**
  - _kind_: confirm-refute-hypothesis
  - _source_: reac2023-modern-mobile-rendering-at-hypehype.md (Aaltonen mobile-tax slide): 'SSBOs are slow!' / 'avoid complex shaders'; the mobile HW is not built for fast generic memory load/store.
  - _ourCode_: src/nanite/NaniteResolve.ts:275 makeFetch in the FRAGMENT resolve + :602-604/:646-648 fetch.fetchWorldVert(ctx,localTri,0..2) — 3 storage-buffer vertex refetches + cluster-DAG/qRaster storage reads per resolved pixel (r.scene ~9.6ms slab)
  - _mechanism_: The deferred resolve reconstructs barycentric attributes by re-fetching the winning triangle's geometry from storage buffers in the fragment stage — random SSBO reads per pixel are the exact mobile-hostile pattern. On Apple the penalty is milder than Mali/Adreno (real cache hierarchy) but still a latency tax inside the 9.6ms resolve. Required by the vis-buffer design (winner stored as an id, not attributes), so it is structural, not a free win.
  - _fix_: Not a pure-orchestration fix (owned by resolve/raster agents): reduce per-pixel storage TRAFFIC (cache the fetched verts across the fetchWorldVert(0/1/2) calls if they re-read shared cluster words; pack hot cluster/vertex data tighter to cut cache-line fetches). CONFIRM first via an A/B that stores interpolated attrs vs refetch. Do NOT move resolve to compute (banned; fragment keeps DCC).
  - _webgpuFeasible_: partial
  - _impact_: medium
  - _confidence_: speculative
- **meterRead fires ~7 buffer readbacks every 15th LIVE frame — periodic hitch candidate**
  - _kind_: confirm-refute-hypothesis
  - _source_: threejsroadmap-profiling-webgpu-shaders.md 'readback jitter' as a noise source; reac2023 Aaltonen 'no per-draw map/unmap' — avoid per-frame staging round-trips.
  - _ourCode_: src/nanite/NaniteFrame.ts:667-673 (meter: every frame%15===0 on the live loop calls meterRead) + :680-694 (meterRead does Promise.all of ~7 readBuffer = getArrayBufferAsync = copyBufferToBuffer + submit + mapAsync each)
  - _mechanism_: On the live rAF loop (meterQuiet only suppresses it inside MeasureHarness), every 15 frames the frame issues ~7 extra staging submits + mapAsync. mapAsync forces the driver to reach a completion point for those buffers; the copies contend with the frame's own submits. At 30fps that is a hitch every ~0.5s — matches the open STUTTER symptom. Async so it does not raise avg, but drives p95/jank.
  - _fix_: On the live loop: stagger to ONE buffer readback per frame (round-robin the 7), drop cadence to every 60 frames, or only run when the HUD/overlay is visible. WebGPU-reachable in our code (NaniteFrame.meter). Verify by A/B: disable the live meterRead and watch p95/frame-time spikes at the deep-forest pose.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: speculative
- **Many near-identical raster pipelines compiled at boot (shadow LEVELS × raster kernel set + ~11 HZB level kernels)**
  - _kind_: violation
  - _source_: threejs-issue-32735: one GPUComputePipeline per ComputeNode instance (no source-dedup); wgpu-issue-4456: WGSL→MSL compile can hang/timeout on Apple M1 Max via XPC; chrome-new-in-webgpu-130: Tint IR 'up to 10x faster translating WGSL to MSL' (so pre-Tint-IR Chrome pays more).
  - _ourCode_: src/nanite/NaniteShadowClip.ts:497 buildNaniteRaster is called once PER shadow clip LEVEL (LEVELS = cfg.levels, ~4-6) → LEVELS full copies of the raster kernel set (kVisClear/kRasterDepth/kHwArgs...) each a distinct pipeline; src/nanite/NaniteHzb.ts:139 builds perLevelCount (~11) distinct level kernels; camera raster adds another set
  - _mechanism_: Dozens of distinct GPUComputePipeline compiles at boot, many with structurally identical logic differing only by baked level w/h/offset and bound buffers. On Apple each is a separate MSL compile via the XPC shader compiler — the frozen-tab / long-boot cost (wgpu #4456). This is a BOOT cost, NOT part of the 20.8ms per-frame base; already partly mitigated by the boot cache. Per-frame it only adds a few extra setPipeline switches (cheap).
  - _fix_: Boot-cost lever, not a 20.8ms lever: share ONE raster/HZB kernel across levels by driving level w/h/offset/cam/vis through UNIFORMS instead of baking them as literals per Fn (so the WGSL source is identical and #32735's proposed bindings-layout cache key would collapse them). Blocked from full dedup unless the #32735 fix lands or we patch _getComputeCacheKey; the uniformization half is doable in our code. Keep relying on the boot cache meanwhile.
  - _webgpuFeasible_: partial
  - _impact_: low
  - _confidence_: confident
- **WHAT WE DO RIGHT — aggressive submit coalescing (do not re-litigate)**
  - _kind_: what-we-do-right
  - _source_: reac2023 Aaltonen: 'batch dynamic uploads per-pass (no per-draw map/unmap)', 'do things at the right frequency', 'no per-draw state tracking'; threejs-issue-32735: renderer.compute(array) 'batches into one encoder'.
  - _ourCode_: src/nanite/Tsl.ts:260/323 dispatchBatch/dispatchBatchMixed; NaniteFrame.ts:323 coalesce default-on folding cull BFS+args+voxFanout+shadowCut into ONE submit (:582-587); NaniteRaster.ts:1971 [kVisClear,kRasterWorld1,grass,kHwArgs] one submit + HZB folded as voxel-submit tail (:1981); NaniteHzb.ts:233 whole mip chain one submit; NaniteShadowClip.ts:1215-1238 P9 per-level pre/post-HW batches
  - _mechanism_: Fewer queue.submit drains = less Dawn/Metal command-buffer commit + XPC overhead per frame; the frame went from ~7 foldable submits to ~2 on the cull side. This is exactly Aaltonen's right-frequency/batching guidance. The residual submits are render passes that legitimately cannot share a compute encoder. Effort here is spent — do not redo.
  - _fix_: None — keep it. If anything, verify coalesce stays default-on in the shipped preset.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident
- **WHAT WE DO RIGHT — hot small data in UBOs, bulk data in storage (the correct mobile split)**
  - _kind_: what-we-do-right
  - _source_: reac2023 Aaltonen: 'Mobile: 16KB uniform buffers! SSBOs are slow!' — keep hot small data in uniform buffers, minimize SSBO random access.
  - _ourCode_: src/nanite/Tsl.ts:357-383 uniformMat4/uniformV3/uniformF/uniformU/uniformArrV4 all wrap three uniform() (=UBO) for per-frame cam matrices (invVp/vp), tau, minPx, lod params, frustum planes; storage() reserved for geometry/vis/queue buffers that MUST be storage (atomic election)
  - _mechanism_: Per-frame scalars/matrices/plane arrays ride 16KB-class uniform buffers, not SSBOs — the exact split Aaltonen prescribes. We are NOT paying the SSBO tax on the hot small uniforms; only the unavoidable bulk buffers (vis, DAG, queues) are storage.
  - _fix_: None — keep it. Any new per-frame scalar should also go through uniformF/uniformU, never a storage slot.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident
- **PROFILING-TODO — bound the true orchestration share on Apple (trackTimestamp A/B + cpu.submitMs)**
  - _kind_: profiling-todo
  - _source_: threejsroadmap-profiling-webgpu-shaders.md recipe: warmup ~20 / collect ~50 / report median + p10/p90; resolveTimestampsAsync sums per-pass-type since last resolve, so isolate a lane by resolving right after it. Recipe is Apple-usable (timestamp-query is an optional feature we already gate on: Engine.ts:114).
  - _ourCode_: src/core/Engine.ts:175-176 already records cpu.updateMs100 / cpu.submitMs100; GpuProfiler + Engine.ts:231-251 already implement per-pass resolve; MeasureHarness meterQuiet path exists (NaniteFrame.ts:667)
  - _mechanism_: Two cheap A/Bs settle whether orchestration is worth chasing: (a) read cpu.submitMs at the deep-forest pose — if it is well under the GPU wall, submit/encode overlaps and orchestration is NOT the slab; (b) boot with trackTimestamp forced off vs on and diff gpuWall to measure the timestamp overhead directly (my ~0.5-1ms estimate is unmeasured). This is the honest gate before spending effort on finding #1.
  - _fix_: Run the two A/Bs on THIS M1 Max at the dense-forest dpr2 pose using the existing GpuProfiler recipe (warmup 20, median of 50). Confirm the in-engine timestamp path returns non-degenerate per-pass numbers on Apple (it is feature-gated, so it may be absent on weaker mobile — handle the missing case, already done at Engine.ts:114).
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: confident

---

## (agent)

_scope: Deferred resolve / per-pixel shading (r.scene ~9.6ms) — NaniteResolve.ts fullscreen FRAGMENT pass that decodes the vis-buffer winner and shades it once (material mux + sun + shadow upsample + ambient), in the DEEP-FOREST case where the dominant covered pixel is voxel-crown foliage._  ·  est share: Estimate: ~4-7 ms of the 20.8 ms grass-off base (roughly 20-35%). Grounding: (a) the r.scene timestamp is ~9.6 ms but overlaps concurrent compute (Apple runs render||compute), so it is an upper bound, not additive — I discount it toward its isolated value; (b) the resolve is a PURE per-covered-pixel fullscreen fragment pass and deep dense forest is near-full coverage (surrounded by crowns), so it scales with the whole framebuffer, consistent with the project's measured pixel-scaling law (dpr1 frame ~9 ms); (c) our own code documents a 37.5 ms r.scene 'cliff' inside a crown driven by the terrain-derivative subgraph inflating register pressure (NaniteResolve.ts:556-561), which confirms the resolve is a top-tier cost specifically in the voxel-crown-dominated case this task targets. I did NOT ablate directly — there is no resolve-off toggle and the deep-forest-breakdown.md the task referenced does not yet exist on disk. The number is therefore grounded on the timestamp clue + the per-pixel-coverage argument + the in-code cliff note, and MUST be pinned by the profiling-todo (default vs ?nandbg=albedo vs ?nandbg=flat vs ?respass=0, grass off, honest gpuWall at a deep-forest pose). Premise-audit fired: the prime suspect I was treating as fixed background is the single-übershader-muxing-7-classes design itself (and RP-4 making it worse in forest), NOT the individual texture taps — hence findings #1/#2 target the shader STRUCTURE (occupancy) one level up, which is where the real lever is.

- **RP-4 single-pass 'both' resolve re-merges the terrain-derivative + all triangle material subgraphs INTO the voxel shader in forest — the exact occupancy-collapse the two-pass split was built to fix**
  - _kind_: confirm-refute-hypothesis
  - _source_: occupancy-interplayoflight.md:21-27 (VGPR count inversely tied to occupancy; >128 VGPR => 0 resident wavefronts => memory stalls exposed) + CROSS-CHECK F3/übershader note (Andersson/DICE: switch/branch shader takes worst-case VGPR across ALL paths; splitting the mega-shader restores tight bounds) + WGSL allocates registers for the WHOLE function so an untaken branch still inflates every lane.
  - _ourCode_: src/nanite/NaniteResolve.ts:1574-1579 (default singlePass='both' when qVoxRasterRO && !gi && csm===null — TRUE in forest) drives buildMat('both'); :562 `if (pass !== 'vox' ...) If(isT, buildTerrainShading)` so 'both' BUILDS terrain; :556-561 the code itself: terrain's implicit-derivative samples are 'the dominant driver of the close-up voxel r.scene cliff (37.5ms inside a crown)' and guarding by pass==='vox' is what strips it.
  - _mechanism_: In forest the resolve is ONE fragment function that muxes terrain(~14 derivative texture taps + fbm + caustics) + rock + bark(TBN + trilinear array taps) + leaf + grass + voxel + procedural-grass and binds all 10 storage buffers. Apple/TBDR occupancy is set by the WORST-CASE register footprint across every branch, even for the >90%-of-pixels voxel path that only executes the voxel decode. The voxel decode is a long DEPENDENT fetch chain (visBV->qVoxRaster->clusters->meshes->voxelBricks->instances); with occupancy collapsed there is no second wavefront to swap to while those loads are in flight, so the latency is fully exposed. RP-4 saved a second payloadV/visBV load + a discard, but at the cost of fattening the shader the dominant tier runs in — the opposite trade for a foliage-saturated frame.
  - _fix_: A/B ?respass=0 (two-pass: the 'vox' shader is stripped of terrain+rock+bark+leaf+grass subgraphs AND the tri-only verts/indices/qRaster bindings) vs the default 'both' at a DEEP-DENSE-FOREST pose at dpr2, grass off, honest gpuWall. If two-pass wins (expected: the vox shader gets a much tighter register footprint => higher occupancy => the voxel-decode latency is hidden), make the forest/foliage config default to two-pass, or gate 'both' on 'is this a foliage-dominated frame'. Pure URL flag + one default flip — no quality change (identity already proven for correctness).
  - _webgpuFeasible_: yes
  - _impact_: high
  - _confidence_: confident
- **Steal Nanite/Burns-Hunt material binning: shade each material class in a SEPARATE tight fragment pass with a material-ID depth-EQUALS reject, instead of one 7-class übershader**
  - _kind_: trick-to-steal
  - _source_: visbuffer-mobile/karis-nanite-siggraph-2021.md + gdc2024-nanite-gpu-driven-materials.md: Nanite emits GBuffer with ONE fullscreen draw per material, using a synthetic Material-ID -> depth value + depth-test EQUAL so non-matching pixels are rejected by fixed-function HW before any PS wave spins up; coarse 64x64 tile + 64-bit mask (portable, no wave intrinsics) snaps absent-material tiles' X to NaN so their PS waves never launch. jcgt-burns-hunt: shade one compute kernel per unique surface material over only the tiles containing it.
  - _ourCode_: src/nanite/NaniteResolve.ts:539-1168 — a single fragmentNode selects terrain/rock/bark/leaf/grass/voxel via isT/isR/isBD/isL/isG/isVox .select() mux; presentClasses (:262-271) only strips classes ABSENT from the whole registry, which in deep forest is none (terrain+bark+leaf+voxel+grass all present) so it buys nothing there.
  - _mechanism_: Separating classes into their own fullscreen fragment passes gives each shader a tight, class-specific register footprint (voxel pass = just the brick decode + wrap lighting; terrain pass = the derivative taps) => each runs at high occupancy for its pixels. The material-ID EQUAL depth reject (writing matClass into a scratch depth target, then depth-test EQUAL per pass) makes the HW skip non-matching pixels with zero PS invocation — cheaper than our per-pixel If(isX)+Discard, and it keeps everything a FRAGMENT pass (DCC preserved). This is the general form of finding #1: two-pass is the crude 2-bin version; per-class binning is the full lever.
  - _fix_: Incremental: (1) ship finding #1 first (cheap). (2) Then split the 'tri' pass itself into terrain-vs-explicit-mesh passes gated by a matClass depth-EQUALS test (terrain's derivative subgraph is the fattest and the terrain pixel-set is screen-coherent in forest — a floor patch). WebGPU-reachable: material-ID-as-depth + depthCompare EQUAL is standard; the 64x64 NaN-tile-cull needs a small classification compute pass writing a per-tile mask, also reachable. Wave-intrinsic 32-bit-mask path is NOT reachable — use the 64x64 portable path.
  - _webgpuFeasible_: partial
  - _impact_: high
  - _confidence_: speculative
- **KEEP the resolve a fullscreen FRAGMENT pass — do NOT move it to compute (Apple DCC guard-rail)**
  - _kind_: what-we-do-right
  - _source_: jcgt-burns-hunt-visibility-buffer.md (vis-buffer is the bandwidth win precisely on integrated/TBDR) + apple-tbdr-blakecrosley.md (framebuffer/tile-memory compression) + arm-mali-guide.md ('compute shading disables bandwidth-saving framebuffer compression; not recommended for most situations'). CROSS-CHECK item 4 flags this as a guard-rail.
  - _ourCode_: src/nanite/NaniteResolve.ts:2-8, 383-385 (NodeMaterial + clip-space fullscreen triangle, fragmentNode); the whole resolve is a Mesh drawn in the scene render pass, shades each vis-buffer winner exactly once (overdraw==1 by construction).
  - _mechanism_: A fragment pass writes through the ROP into a DCC/AFBC-compressed color attachment kept in tile memory; a compute rewrite would decompress and round-trip device memory. Single-shade-the-winner also means the heavy material ALU is paid once per pixel, not per overdrawn fragment — the Nanite deferred-material win. This is correct for Apple; a future 'move resolve to compute for binning' refactor would REGRESS bandwidth. Any material-binning (finding #2) must stay fragment-side.
  - _fix_: No change — protect it. When implementing finding #2's binning, keep every class pass a FRAGMENT draw with depth-EQUALS reject, not a compute shade.
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: confident
- **Forest gi=null already elides the probe-irradiance chain (groundY heightTex tap + SH eval + canopy tap) — do not spend effort 'optimizing GI in forest', it isn't built**
  - _kind_: what-we-do-right
  - _source_: occupancy-interplayoflight.md (fewer live fetches => less exposed latency) — the absence of the whole probe subgraph is register/tap relief already banked.
  - _ourCode_: src/nanite/NaniteResolve.ts:1371 `if (world.gi)` gates the entire probe block (groundY texture tap :1376-1378, world.gi.irradiance SH eval, canopy damping tap :1381); NaniteFrame.ts:381 passes gi: world.gi, which is null in the forest/world scene (also why RP-4 single-pass is legal there).
  - _mechanism_: The probe path is a texture tap + a multi-tap SH irradiance eval per covered pixel; it is compile-time removed in forest, so the ambient in deep forest is only the cheap hemisphere floor (:1403-1417). The dark-forest ambient is essentially free already.
  - _fix_: None. Note for the team: any perf idea about the GI probe does NOT apply to the deep-forest case (only to clearing/world poses where world.gi is set). Focus the resolve effort on the voxel decode + shadow upsample.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident
- **'keep' CSM full-screen sample is already gated to the [0,0] corner pixel (reskeep) — the wasted full-res PCSS-on-empty-maps is already banked**
  - _kind_: what-we-do-right
  - _source_: arm-mali-guide.md: shadow/comparison depth samples are 2x cost; killing a per-pixel comparison sample that always returns 1 is pure win. occupancy-interplayoflight.md: cut fetches that stall.
  - _ourCode_: src/nanite/NaniteResolve.ts:1317-1325 — the three-CSM `keep` PCSS is sampled only on the corner pixel (keepFullU default 0 via reskeep), keeping three's node built (for its cascade fit) while every real pixel skips the ~1 wasted comparison-depth sample; forest csm===null so :395-415's second wp reconstruction (receivedShadowPositionNode) is also NOT built.
  - _mechanism_: Empty black-slate CSM maps => keep==1, so the sample was pure waste; corner-only keeps it alive at 1 pixel of cost. Both the wasted PCSS and a whole second per-pixel world-pos reconstruction are already off in forest.
  - _fix_: None — protect it. Do not re-enable reskeep in the forest path.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident
- **Shadow bilateral upsample does 4 textureLoad taps on an rg32f (FP32) target per covered pixel; pack to a narrower format to halve the tap cost**
  - _kind_: trick-to-steal
  - _source_: arm-mali-guide.md: 'FP32 formats have a 2x cost'; 'if you need higher dynamic range consider packed 32-bit formats (RGB10_A2, RGB9_E5) as an alternative to FP16/FP32 textures.' WWDC-10859: faster FP32/INT32 texture reads on M1 but packed formats still win bandwidth.
  - _ourCode_: src/nanite/NaniteShadowHalf.ts:83-85 tex is FloatType+RGFormat (rg32f: R=shadow 0..1, G=camDist metres); :168-181 upsample loops 4 corners with textureLoad(tex) per full-res covered pixel; called from NaniteResolve.ts:1305-1306 for every non-grass covered pixel (grass takes the lean path).
  - _mechanism_: In deep forest nearly every covered pixel runs 4 FP32 rg texture reads for the bilateral. The shadow factor needs ~8 bits; camDist could be an fp16-range or packed value. A packed rg16f (or r11g11-style / rgba8 with camDist quantized) halves the per-tap format cost on Mali-class TBDR and cuts the half-res texture's bandwidth+footprint. This is the single biggest per-pixel texture cost that lives in r.scene (the shadow EVAL is in r.half.mrt, but the UPSAMPLE taps are in the resolve).
  - _fix_: Change NaniteShadowHalf tex to a packed format: shadow in 8 bits + camDist quantized to fp16-ish in the other channels, or rg16f if half-float storage is accepted (storage textures don't need shader-f16). Keep NearestFilter (textureLoad ignores filter). Verify the bilateral tolerance math still resolves silhouettes after quantizing camDist. Then A/B gpuWall at a deep-forest pose. WGSL-reachable (StorageTexture format is a JS/TS choice, not a shader-f16 feature).
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: speculative
- **Bark is sampled trilinear (LinearMipmapLinear) via explicit .level(lod) = Mali 2x texture cost, and its anisotropy:4 is dead weight**
  - _kind_: violation
  - _source_: arm-mali-guide.md: 'Trilinear, LINEAR_MIPMAP_LINEAR, filtering has a 2x cost'; 'consider 2x bilinear AF in preference to isotropic trilinear'; textureGrad much slower (we correctly avoid it by default).
  - _ourCode_: src/gpu/passes/BarkSynth.ts:292 t.minFilter = LinearMipmapLinearFilter; :293 t.anisotropy = 4 — but NaniteResolve.ts:737 samples with `.level(lod)` (explicit analytic LOD). Explicit LOD ignores anisotropy entirely (dead), and a fractional lod still triggers trilinear inter-mip blend (the 2x).
  - _mechanism_: Every near-trunk bark pixel pays a 2x-cost trilinear array tap (texA always, texB normal-map inside resfar). In deep DENSE forest bark is a minority vs voxel crowns, so the ceiling is modest — but it is a clear format/filter violation with a dead anisotropy setting. LinearMipmapNearest (bilinear, snap to nearest mip) is 1x and, combined with the analytic LOD we already compute, is close in quality (possible faint seam at mip transitions — a quality call).
  - _fix_: Set bark texA/texB minFilter to LinearMipmapNearest and drop anisotropy (it's ignored by .level()). If a mip seam is visible, keep trilinear only on texA and take texB (normal) bilinear. A/B a trunk-heavy pose; low expected forest win but free correctness. WGSL/JS-reachable.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident
- **Material-decode + normal/wrap-lighting ALU runs FP32; fp16 would relieve REGISTERS (the occupancy lever, not a FLOP lever) — BLOCKED at TSL**
  - _kind_: confirm-refute-hypothesis
  - _source_: arm-mali-guide.md (FP16: vectorize f16vec2/4 for packed f16x2) + occupancy-interplayoflight.md (fewer VGPRs => more resident wavefronts) + MASTER-AUDIT (fp16 payoff on our HW is register/occupancy relief, not FLOP doubling).
  - _ourCode_: src/nanite/NaniteResolve.ts:1264-1418 — sunDir/nDotL/wrapped/ambUp/albedo/radiance all float/vec3 (FP32); the voxel bead/normal math :1030-1085 and bark TBN :662-694 are FP32 vec3 chains that dominate the shader's live-register set.
  - _mechanism_: Halving these to fp16 would shrink the resolve's worst-case register footprint => raise occupancy => help hide the voxel-decode fetch latency (same axis as finding #1). It is NOT a FLOP win (Apple isn't ALU-bound here). BLOCKED: three r184 TSL has no half node and the device lacks shader-f16, so it is unreachable at our layer.
  - _fix_: Blocked-unless-raw-WGSL: would require emitting f16 types directly in a hand-written WGSL resolve (bypassing TSL) AND enabling shader-f16 (device lacks it). Not actionable now; record so it isn't re-litigated. The reachable substitute for the SAME occupancy goal is finding #1/#2 (split the shader to shrink the live-register set structurally instead of numerically).
  - _webgpuFeasible_: no-metal-only
  - _impact_: medium
  - _confidence_: speculative
- **PROFILING-TODO: ablate the resolve's true share of the 20.8ms base and capture the 'both' shader's occupancy/VGPR**
  - _kind_: profiling-todo
  - _source_: occupancy-interplayoflight.md:30 (profile for stalls on actual memory reads; occupancy is a proxy — measure) + apple-wwdc Tech-Talk-10580 (compiler statistics show spilled-bytes + occupancy% directly, exactly how the '16% occupancy from spilling' case was diagnosed).
  - _ourCode_: src/nanite/NaniteResolve.ts:1481 `?nandbg=flat` (returns albedo only — skips ALL lighting/shadow-upsample/ambient), :1482 `?nandbg=albedo`, :1574-1579 `?respass=0|1`; NaniteShadowHalf upsample is the resolve-side shadow cost.
  - _mechanism_: The r.scene ~9.6ms timestamp OVERLAPS concurrent compute on Apple, so it is only a clue. The clean ablations: (1) default vs ?nandbg=albedo => isolates lighting+shadow-upsample+ambient cost inside the resolve; (2) default vs ?nandbg=flat => isolates the whole shading tail; (3) ?respass=0 vs default at a deep-forest pose => tests finding #1 directly. All at dpr2, grass OFF, honest gpuWall, same pose.
  - _fix_: Run the three A/Bs above at a DEEP-DENSE-FOREST pose (surrounded by voxel crowns), grass off. Additionally capture the 'both' resolve fragment shader's occupancy% + spilled bytes via the WebGPU/Metal compiler statistics path the grass audit already uses (attach the inspector / MTL compiler stats to the Dawn process). Decision rule: occupancy healthy but slow => latency-exposed => cut the voxel-decode dependent fetches; occupancy collapsed/spilling => ship the shader split (finding #1/#2).
  - _webgpuFeasible_: yes
  - _impact_: high
  - _confidence_: confident

---

## (agent)

_scope: Shadows — toroidal shadow clipmap (NaniteShadowClip.ts) + half-res PCSS eval (NaniteShadowHalf.ts) + shvox2 voxel-crown caster_  ·  est share: ~0.5ms of the ~17.4ms grass-off base (≈3%). GROUNDED in the honest ablation table docs/mobile-gpu-perf/deep-forest-breakdown.md:9,17 — `?nanshadow=0` drops the frame 17.4→16.9ms (whole shadow system = 0.5ms marginal) — and the c.nanShadowHalf timestamp column = 0.2ms (breakdown.md:8). shvox2 crown-caster accounts for essentially all of that 0.5ms (`?shvox2=0` also lands at 16.9, breakdown.md:13,21). CRITICAL PREMISE CORRECTION: the r.half.mrt ~10.5ms the task pointed me at is NOT shadows — it is the POST half-res MRT (cloud march + GTAO + screen-space bounce), tagged at src/render/HalfResMrt.ts:70 and enumerated as post at src/main.ts:33. The shadow half-res eval is the separate c.nanShadowHalf compute pass (0.2ms). So at the STATIC deep-forest base, shadows are NOT a meaningful lever. The real shadow cost is MOTION-gated: strip re-raster is ~8ms p50 MOVING vs 0.8ms still (docs/perf-runs/2026-07-04-90fps-arc.md:68) — i.e. shadows are a STUTTER/motion lever for deep forest, not a static-base lever.

- **PREMISE ERROR: r.half.mrt (10.5ms) is POST, not shadows — the shadow subsystem is ~0.5ms of the static base**
  - _kind_: confirm-refute-hypothesis
  - _source_: Our ablation doc deep-forest-breakdown.md:9,17 (`nanshadow=0` 17.4→16.9ms) + :8 (c.nanShadowHalf=0.2ms); src/main.ts:33 lists half.mrt under the POST chain.
  - _ourCode_: src/render/HalfResMrt.ts:70 (tagGpu(rt,'half.mrt') — cloud/GTAO/bounce); src/nanite/NaniteShadowHalf.ts:140 (the real shadow pass = 'nanShadowHalf')
  - _mechanism_: The task's focus pass (r.half.mrt) is the deferred POST half-res MRT (volumetric clouds + GTAO + bounce), which shares nothing with shadows. The actual shadow-eval pass is c.nanShadowHalf, measured at 0.2ms, and the entire shadow system ablates to 0.5ms at the static deep-forest pose. Chasing shadow taps to cut 'r.half.mrt' would touch the wrong subsystem entirely.
  - _fix_: Re-target: 10.5ms r.half.mrt belongs to the POST audit (GTAO/clouds/bounce), not shadows. For shadows, do NOT invest in the static per-pixel PCSS path — it is already ~0.2ms.
  - _webgpuFeasible_: yes
  - _impact_: high
  - _confidence_: confident
- **The real shadow lever is the MOVING strip re-raster (~8ms p50 walking vs 0.8ms still) + shvox2 — a deep-forest STUTTER cost, invisible in the static base**
  - _kind_: confirm-refute-hypothesis
  - _source_: docs/perf-runs/2026-07-04-90fps-arc.md:68 ('SHADOWS-MOVING ≈ 8ms p50, still-pose 0.8'); shadow-arc.md:156 (toroidal clip shadow bill +5.2 moving).
  - _ourCode_: src/nanite/NaniteShadowClip.ts:1203 rasterLevels (per-level filter+SW depth1+hwDepth+shvox2 caster+kCopy), fired from run():1257 whenever fitLevels returns mask!=0
  - _mechanism_: Deep forest symptom is '<30fps + stutter' under camera motion. Each level whose snapped VP changed re-runs a full cull-filter + SW/HW depth raster + voxel-crown caster + copy. In dense forest a moving camera reveals/occludes many casters per frame → several levels re-raster → the 8ms spike. This is exactly the deep-forest stutter, and it is entirely motion-gated (0.8ms when still), so a STATIC-pose base decomposition (breakdown.md) cannot see it.
  - _fix_: Profile at a MOVING deep-forest pose with `?nanshadow=0` and `?shvox2=0` to split re-raster vs voxel-caster. Levers within the quality bar: (a) coarser far-level caster cut via `?shvoxk` / widen the min-screen-size cull so a moving reveal re-rasters fewer clusters; (b) temporally stagger far-level re-raster (only 1 coarse level per frame) — coarse texels already cache many frames; (c) cap per-frame re-raster level count. All WGSL/CPU-cadence reachable (dispatch gating in run()).
  - _webgpuFeasible_: yes
  - _impact_: high
  - _confidence_: confident
- **shvox2 voxel-crown caster is default-ON and is ~all of the 0.5ms static shadow bill + scales the moving re-raster (the crown-arc's new deep-forest cost)**
  - _kind_: profiling-todo
  - _source_: deep-forest-breakdown.md:13,21 (`shvox2=0` → 16.9 = same as full no-shadow); Karis Nanite/Epic foliage-shadow guidance (small casters → contact shadows, not full maps).
  - _ourCode_: src/nanite/NaniteShadowClip.ts:649 (shVox2 = voxSplat && shvox2!=='0' → DEFAULT ON; the :648 comment 'DEFAULT OFF' is STALE), dispatched per re-rastered level at :1231-1237; seedVoxAllDist default-on at :493
  - _mechanism_: shVox2 atomicMin-splats matClass-7 crown bricks' sun-facing depth into every re-rastered level (seeded at ALL distances). On Apple TBDR the atomic-heavy scatter over dense crown bricks re-runs on every re-raster frame; with seedVoxAllDist ON it also runs on the fine near levels each frame even at rest (why static shadow is 0.5ms not ~0). Under motion it re-scatters crowns for each re-rastered level → the dominant scalable term after the tri strip re-raster.
  - _fix_: Measure `?shvox2=0` and `?shvoxk` at a MOVING pose (the static delta is only 0.5ms). If it dominates moving: (a) restrict seedVoxAllDist to far levels only (near crowns already cast via leaf-mesh/tri depth), (b) coarsen the shadow voxel-τ so the caster reads fewer/bigger bricks, (c) skip the caster on levels served from cache. Reachable via existing `shvoxk`/`shvoxnear` knobs + run() gating.
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: speculative
- **Shadow depth textures are r32float — 15 PCSS taps pay FP32 2x texture-read cost on Mali (and elevated cost on Apple) every eval**
  - _kind_: violation
  - _source_: docs/mobile-gpu-perf/arm-mali-guide.md:90 ('FP32 formats have a 2x cost' for texture sampling).
  - _ourCode_: src/nanite/NaniteShadowClip.ts:321 depthTex = StorageTexture(...FloatType/RedFormat = r32float); read 15×/eval via depthAt():1281 in pcss():1291 (6 blocker + 9 PCF taps)
  - _mechanism_: The stored global sun-axis z is already normalized to [0,1] (z_g = (dot(p,fwd)+D_OFF)/D_RANGE, :1393), so full FP32 range is unnecessary. Each of the 15 taps is a full-width FP32 fetch — 2x the texel-fetch cost of a 16-bit format on Mali and wider data-path traffic on Apple. Also doubles the kCopy write bandwidth.
  - _fix_: Store depth as r16float or unorm16 (z_g is bounded [0,1] → unorm16 gives ~15-bit precision, ample for soft PCSS). Change StorageTexture format at :321 + the bit-pack in kCopy; taps and PCSS math unchanged. WGSL/Three-TSL reachable (r16float/RGFormat are renderable storage formats). LOW static impact (0.2ms eval) but also trims the moving raster copy.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: speculative
- **Half-res eval + copy kernels use 1-D linear dispatch → poor 2-D texture-store/gather cache locality (Mali wants 8×8 square)**
  - _kind_: violation
  - _source_: arm-mali-guide.md:104-106 ('use a square execution dimension, e.g. 8x8, to exploit optimal 2D cache locality'; 'group four adjacent lanes to the same 64-byte cache line').
  - _ourCode_: src/nanite/NaniteShadowHalf.ts:139 kHalf .compute(N,[256]) with idx→(hx=idx%hW, hy=idx/hW); src/nanite/NaniteShadowClip.ts kCopy/kClear use the same flat SHADOW_PIX 1-D grids
  - _mechanism_: A flat 256-lane workgroup is a 256×1 horizontal strip in texture space. The textureStore to the rg32f half-res target and the 4-tap bilateral gather in upsample() then straddle many cache lines per warp instead of a compact 16×16 tile. On Apple/Mali the texture cache is 2-D-tiled, so linear-strip access under-uses each fetched line.
  - _fix_: Remap idx to a Morton/8×8-tiled (hx,hy) inside the kernel (pure ALU on instanceIndex — no API change) so a workgroup covers a square tile. WGSL-reachable. LOW impact given 0.2ms, but free and also helps the shadow-raster copy passes.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: speculative
- **What we do right — DON'T waste effort shrinking the static PCSS: half-res eval, bilateral upsample, toroidal static-caching, manual compare (no HW-compare 2x), NearestFilter**
  - _kind_: what-we-do-right
  - _source_: arm-mali-guide.md:91 (shadow reference/compare depth = 2x cost — we avoid it by manual compare) + :86 (nearest/bilinear full-speed, trilinear 2x — we use NearestFilter); occupancy-interplayoflight.md:30 (don't optimize where there's no stall).
  - _ourCode_: NaniteShadowHalf.ts:111 half-res eval quarters the sample; :149 depth-aware bilateral upsample; NaniteShadowClip.ts:1315 manual textureLoad+lessThan (no comparison sampler); :324 NearestFilter; strip-only re-raster caches static levels (:1144 reRaster gate → 0.8ms still)
  - _mechanism_: The static per-pixel PCSS path is already minimized: manual depth compare dodges Mali's 2x shadow-compare-sampler cost, NearestFilter dodges the trilinear 2x, half-res quarters the 15-tap eval, and the toroidal strip cadence caches unchanged levels so a still camera pays ~0. This is WHY the static shadow bill is only 0.2-0.5ms — further tap-count/eval optimization has near-zero headroom. Spend the budget on the MOVING re-raster instead.
  - _fix_: No change. Explicitly de-prioritize any 'fewer PCSS taps / cheaper eval' work — the ablation proves it cannot pay back more than ~0.2ms. Redirect to the motion-gated re-raster (finding #2/#3).
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: confident

---

## (agent)

_scope: Voxel scatter kVoxScatter (leaves/crowns), 128-lane one-cluster-per-workgroup, 32-bit atomic election — src/nanite/NaniteVoxelRaster.ts_  ·  est share: SPECULATIVE ~3–6 ms of the base. Grounding: the honest ablation (docs/mobile-gpu-perf/deep-forest-breakdown.md) shows base gpuWall 17.4 ms with grass off (the task's "20.8 ms" is that run's *render* pass timestamp, which OVERLAPS compute and is not additive). kVoxScatter is NOT directly isolated there: it lives inside the "compute" bucket (8.72 ms base = cull + SW-tri bark raster + voxel scatter + occ-pyramid + HZB, all coalesced into dispatchVoxel's submit, NaniteFrame.ts:614). The only ablation touching voxels (voxreg=0) *confounds* the scatter with a full-mesh fallback: it makes the frame EXPLODE +54 ms (17.4→71.4, c.nanVisClear 7.7→65.5) because leaves/crowns revert to LOD0 triangle meshes through the SW raster. So the scatter's net contribution is hugely POSITIVE (~-54 ms vs mesh) and its own fill cost is a modest slice of the 8.72 ms compute — the two mesh-foliage marginals (crownlod0 ~3.9 ms + leafmesh ~4.1 ms) account for most of the SW-raster side, leaving the pure voxel fill at roughly 3–6 ms. The clean settling measurement EXISTS but was not run: ?voxrdbg=2 (NaniteVoxelRaster.ts:246,1561) skips Phase-B election so (base − voxrdbg2) gpuWall = the exact voxel per-pixel FILL share. That plus a Metal capture of kVoxScatter occupancy%/spilled-bytes is the profiling-todo below.

- **kVoxScatter is a MEGAKERNEL: 5 default-on feature paths compiled into one Fn inflate every lane's register footprint → Apple spills the per-pixel fill loop to device**
  - _kind_: violation
  - _source_: occupancy-interplayoflight.md:21-27 (VGPR count inversely tied to occupancy; WGSL/compiler allocates registers for the WHOLE function so a rarely-taken path inflates every lane) + metal-benchmarks-README.md:208 ('ALU utilization maxes at 24 simds/core... Apple would rather you SPILL TO DEVICE MEMORY than decrease ALU utilization') + arm-mali-guide.md:102-103 (large workgroups restrict registers/work-item → stack spilling)
  - _ourCode_: src/nanite/NaniteVoxelRaster.ts:571-1564 (single makeVoxScatter Fn) compiling voxOccl pyramid cull (:730-789), voxBocc per-brick occl (:967-1008), voxOccGate 512-projection mask build (:1058-1191), voxCell ray+DDA (:1230-1268 basis, :1401-1549 per-pixel), voxWind (:710-729) — all DEFAULT ON (:206,209,291,304,311)
  - _mechanism_: In DEEP DENSE FOREST the near crowns are fine L0 bricks ≤~6px (area ≤ voxCellMinArea=64, :319-324) so they take the CHEAP flat path — yet every lane still carries the register allocation for the ray/DDA/mask-build/pyramid paths because WGSL registers are allocated for the union of all branches. Apple's documented policy is to spill (device-memory traffic) rather than drop occupancy, so the dominant deep-forest cost — the Phase-B per-footprint-pixel atomic-election fill — runs against a spilled register file. This is the SAME disease our own resolve already fixed for the vox pass (NaniteResolve.ts:556-561: 'inflates register/instruction pressure → collapsed occupancy → close-up voxel cliff 37.5ms').
  - _fix_: Split the megakernel via the existing makeVoxScatter factory + F2B/bucket partition infra: a LEAN flat-scatter kernel (near/dense L0 crown — the deep-forest majority: occlusion cull + flat election only, NO ray/DDA/mask code compiled in) and a separate RAY kernel for the far coarse bricks, dispatched over disjoint qVoxRaster partitions. Build-time gate the voxCell/voxOccGate closures OFF in the near-kernel variant so their registers never allocate. WebGPU-feasible: yes (two Fn instances, same buffers).
  - _webgpuFeasible_: yes
  - _impact_: high
  - _confidence_: speculative
- **voxCell per-cluster ray basis is recomputed redundantly by all 128 lanes (uniform value) — long-lived dead registers on the deep-forest flat path**
  - _kind_: trick-to-steal
  - _source_: shader-opt-persson.md ('keep register lifetime LOW') + occupancy-interplayoflight.md:21-27 (register footprint drives occupancy/spilling)
  - _ourCode_: src/nanite/NaniteVoxelRaster.ts:1230-1268 — rayBaseL/rayDxL/rayDyL (3 vec3), clipP0zw/Db/Dx/Dy (8 floats), roLx/roLy/roLz, plus K/Bx/Cy invVp mat-vec products, all computed per-lane and .toVar()'d live across the whole Phase-B loop; contrast the CORRECT hoist of wgWind to lane 0 + workgroupArray broadcast at :710-729
  - _mechanism_: These ~30+ floats are per-workgroup UNIFORM (A/B/yawSc/cam are per-cluster) but every one of the 128 lanes computes and holds them live, and Metal does no cross-lane CSE. In deep forest the ray path never fires (near = flat), so they are allocated-but-dead registers that inflate the footprint feeding finding #1's spill. The kernel already proves the fix pattern with wgWind (lane-0 compute → shared-memory broadcast → barrier).
  - _fix_: Either (a) hoist the ray basis to lane 0, store in a small workgroupArray, barrier-broadcast (mirror wgWind at :710-729), trading ~30 registers/lane for ~120 bytes shared + one barrier; or (b) only compute it when the workgroup actually contains a ray-eligible brick (a workgroup-shared 'anyRayEligible' flag set in Phase A). Cleanest when combined with the finding-#1 kernel split (the near-kernel simply never emits this block).
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: confident
- **32-bit split atomic election (relaxed-load gate → atomicMax payload → atomicStore id) is the ONLY correct Nanite election on M1 — do not regress**
  - _kind_: what-we-do-right
  - _source_: metal-benchmarks-README.md 'Nanite Atomics' section: Apple 7 (M1/M1 Max) supports ONLY 32-bit pointer atomics; the single 64-bit UInt64 min/max instruction is M2+ (A15/A16/M1 lack it). Nanite-on-32-bit-atomics is a known workaround at '2.5x bandwidth/5x latency cost' — but on M1 there is no 64-bit alternative
  - _ourCode_: src/nanite/NaniteVoxelRaster.ts:1343-1361 electHere: aLoadU(prevE) relaxed load → If(cand>prevE) → atomicMax(visPayloadV) → If(won) atomicStore(visBV, voxIdB); key = depthKey24<<8 | id8 (:1015-1018)
  - _mechanism_: On M1 Max there is no HW 64-bit depth+payload compare, so the split 32-bit payload/id design is forced AND correct. The relaxed-load prevE gate skips the atomic entirely on the common losing case, minimizing the 5x-latency 32-bit atomic traffic — the standard optimization. In deep forest the real atomic cost is contention on visPayloadV from overdraw; the correct lever is REDUCING wins (occlusion culls, finding #4), not touching the election. Matches the system constraint 'no 3rd atomic buffer'.
  - _fix_: None — preserve. Do NOT attempt a 64-bit fused atomic (unavailable on M1) or a 3rd atomic buffer. Bank this so the arc spends no effort re-litigating the election.
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: confident
- **The min-pooled occlusion pyramid + per-block/per-brick cull is the load-bearing deep-forest overdraw killer — do not strip**
  - _kind_: what-we-do-right
  - _source_: karis-nanite-siggraph-2021 (HZB-driven occlusion cull as Nanite's core overdraw defense) + deep-forest-breakdown.md (voxels net -54ms vs mesh fallback)
  - _ourCode_: src/nanite/NaniteVoxelRaster.ts:434-557 (min-pooled pyramid over THIS-frame's triangle election), :730-789 (per-block cull, one global read at projected centre), :967-1008 (voxBocc per-brick), all default ON (voxOccl :404, voxBocc :291)
  - _mechanism_: Deep dense forest = maximal near-canopy occluders. Culling whole blocks/bricks that sit behind the near mesh canopy BEFORE the footprint loop removes the buried-brick fill that would otherwise dominate the atomic-election cost. Code records voxbocc measured eye 36.1→18.9 ms (:286). MIN-pooling + full-pyramid (every texel, no gaps) is what makes it conservative/hole-free on TBDR.
  - _fix_: None — preserve. This is why removing voxels regresses +54ms rather than helping. Any near-kernel split (finding #1) MUST keep the per-block cull.
  - _webgpuFeasible_: yes
  - _impact_: high
  - _confidence_: confident
- **Per-pixel integer div+mod in the footprint loop already replaced by a per-brick reciprocal (Apple has no HW int-divide)**
  - _kind_: what-we-do-right
  - _source_: shader-opt-persson.md ('integer division extremely expensive ~40-48 cycles') — Apple/Metal likewise has no native integer divide
  - _ourCode_: src/nanite/NaniteVoxelRaster.ts:1334-1337 (voxRecip path: ly=floor((localPx+0.5)*invW), lx=localPx-ly*bbW) with invW/invH computed once per brick (:1305-1306); default ON (:257)
  - _mechanism_: Without this, each of tens of millions of footprint fragments/frame paid a microcoded int div+mod. The per-brick reciprocal is loss-exact (fp32 product error ≤ area*2^-23 « the 0.5/bbW rounding margin) and measured -8.8ms at the worst 200k pose. Directly relevant to deep-forest fill.
  - _fix_: None — done. Listed so the arc doesn't re-audit it.
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: confident
- **Phase-A dynamic per-brick indexing into gpu.clusters/gpu.voxelBricks — a Mali violation but Apple-BENIGN (contiguous/strided across the wave)**
  - _kind_: confirm-refute-hypothesis
  - _source_: arm-mali-guide.md:128,185-186 ('avoid dynamic indexing — disables pilot shaders'), :118-120 ('access sequential/overlapping addresses across the quad; do not access divergent addresses') + aaltonen 'scalarize memory access'
  - _ourCode_: src/nanite/NaniteVoxelRaster.ts:642-643 (brickBase/brickCount from clusters), :808-838 (bWordBase=(brickBase+brickLocal)*BRICK_WORDS → per-lane dynamic address into voxelBricks)
  - _mechanism_: The Mali pilot-shader/uniform-register-promotion penalty is a Mali-specific concept Apple does not share. Critically, Phase A is 1 lane = 1 brick with brickLocal contiguous, so lane b reads brick brickBase+b — a SEQUENTIAL, BRICK_WORDS-strided access pattern across the wave, which is exactly the merge-friendly case (mali:118), not divergent. So this is a violation only on Mali, and even there it is the good access shape. On Apple it should be near-optimal.
  - _fix_: Refute for Apple (no change). If the weaker mobile target proves Mali-bound, note the AoS BRICK_WORDS stride puts only ~10 words in a 128B line — an SoA re-pack of the hot brick words (pos/half/occ) would tighten coalescing — but that is a mobile-only, measure-first change, not an Apple lever.
  - _webgpuFeasible_: partial
  - _impact_: low
  - _confidence_: speculative
- **voxOccGate 512-projection mask build runs in Phase A on ONE lane per coarse brick — bounded away from the deep-forest-near case but adds compiled-in register bulk**
  - _kind_: confirm-refute-hypothesis
  - _source_: arm-mali-guide.md:102-103 (per-lane serial work in a large workgroup) + occupancy-interplayoflight.md (long single-lane sequences stall the whole group)
  - _ourCode_: src/nanite/NaniteVoxelRaster.ts:1091-1191 — triple-nested 64-cell loop, each occupied cell projects 8 AABB corners + a 4x4 bucket-mark loop, on the single Phase-A brick-lane; armCond requires dagLevel>0 (:1062) and voxMaskRay skips it for ray-eligible bricks (:1063-1068)
  - _mechanism_: In deep-forest-NEAR the crowns are L0 (dagLevel=0) so the gate does NOT arm — the mask build is a FAR-field cost, and voxMaskRay (default on, :354) further skips it where the ray path consumes. So its RUNTIME cost is small at the target pose. BUT the code is compiled in, contributing to finding #1's register-union footprint on the near flat path.
  - _fix_: Confirm via ?voxocc=0 A/B at the deep-forest pose (isolates the gate). If runtime-negligible there (expected), fold its removal into the finding-#1 near-kernel split so its ~20 live build-vars stop inflating the near path; keep it only in the far/ray kernel.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: speculative
- **PROFILING-TODO: isolate the true kVoxScatter fill share (?voxrdbg=2) and read its occupancy%/spilled-bytes (Metal capture) before any register/kernel-split work**
  - _kind_: profiling-todo
  - _source_: MASTER-AUDIT.md:47,235 (Phase-0: Metal capture of Chrome/Dawn GPU process → kVoxScatter occupancy%/spilled-bytes/top-limiter via Xcode Capture GPU Frame or Instruments; Tech-Talk-10580 compiler statistics show spilled-bytes directly) + occupancy-explained decision rule (spilling ⇒ register-bound ⇒ shrink live state)
  - _ourCode_: src/nanite/NaniteVoxelRaster.ts:246,1561 (?voxrdbg=2 build-time STOP before Phase-B election, already wired); dispatch coalesced in NaniteFrame.ts:614 so no standalone timestamp exists
  - _mechanism_: The deep-forest-breakdown doc ran voxreg=0 (confounded with +54ms mesh fallback), never the clean voxrdbg=2 isolation. (base − voxrdbg2) gpuWall at the deep-forest eye pose gives the exact per-pixel voxel FILL cost = my subsystem's real share of the base. The Metal capture then decides finding #1: if kVoxScatter shows spilled-bytes>0 / occupancy below a wave-count step ⇒ register-bound ⇒ the kernel split (finding #1) + basis hoist (finding #2) are the sanctioned levers; if occupancy is healthy but slow ⇒ atomic/LLC-latency-bound ⇒ double down on occlusion culls instead.
  - _fix_: Run at the deep-forest pose, dpr2: (1) base vs ?voxrdbg=2 gpuWall delta; (2) ?voxcell=0 and ?voxocc=0 A/Bs to size each compiled-in path's runtime; (3) Metal System Trace / Xcode GPU capture on kVoxScatter for occupancy% + spilled bytes; (4) cheap ?WG_RASTER=64 A/B to test whether halving the workgroup frees registers. Ship nothing register-related before (3).
  - _webgpuFeasible_: yes
  - _impact_: high
  - _confidence_: confident

---

## (agent)

_scope: SW compute raster: sub-pixel/sliver tri scanline + atomicMax depth election, HW routing for big/thin tris (trees/trunks/bark), world1 kernel — src/nanite/NaniteRaster.ts_  ·  est share: ≈8ms at a dense eye pose up to ~15–18ms at a dense oblique/surrounded pose — the SINGLE LARGEST slab of the 20.8ms grass-off base. Grounded three ways: (1) direct ablation in the ninety-fps-arc-costmap memory + task #62: "pixel LOOP = 7.8 eye / 18.4 oblique (default−rdbg2)"; (2) the rdbg stage-split finding that the per-covered-pixel loop is ~90% of world1 and per-TRI setup is measured FREE (rdbg2−rdbg1≈0, NaniteRaster.ts:330-337); (3) walked-pixel cost scales with device px (costmap "frame ≈ A + B·px, B≈3.2 eye / 5.0 oblique"), and deep-dense-forest = the surrounded/oblique-and-worse end. CAVEAT: the world1 per-pass GPU timestamp is unusable (reads a bogus constant ~15.2ms at the 120fps rAF cap per the code comment) — I relied on the ablation, not the timestamp. This ~8–18ms EXCLUDES the separately-timed HW leaf-needle pass (r.nanHwPass, additive) and the c.nanVisClear ~11ms (judged a batched-submit overlap artifact, see profiling-todo).

- **32-bit split depth-key election + aLoadU pre-gate + no 3rd atomic buffer is the correct no-atomic<u64> Apple port — do not regress**
  - _kind_: what-we-do-right
  - _source_: Karis SIGGRAPH-2021 (64-bit atomicMax packs depth-high|payload-low; 'payload must fit ≤34 bits or no fast SW raster'); Tellusim (Metal/mobile lack 64-bit image atomics → must split); Aaltonen HypeHype slide 8 ('no 64-bit atomics on mobile GPUs')
  - _ourCode_: NaniteRaster.ts:442-443 (depthKey24), :595-607 (aLoadU guard → If(cand>prev) → atomicMax → winner atomicStore), :1262-1272 ('3rd atomic storage buffer = 3× cliff, 15-17ms')
  - _mechanism_: WGSL has no atomic<u64>; we pack a 24-bit inverted depth key + 8-bit tiebreak into ONE atomicMax word (visPayloadV) and store the full 25-bit id in a side buffer only when the fragment wins. The relaxed atomicLoad pre-gate suppresses the RMW for losing candidates (the vast majority under dense overlap), so the election is a MINOR share of the pixel loop (proven by ?relect). Adding an exact-depth 3rd atomic buffer was measured to triple cost on M1. Most important Apple election decision and it is right.
  - _fix_: No change. Guard-rail: any future exact-per-fragment-depth feature MUST fold into the existing 24-bit election word, never add a 3rd hot-loop atomic buffer.
  - _webgpuFeasible_: yes
  - _impact_: high
  - _confidence_: confident
- **SW compute raster + two-sided leaf re-wind is the correct Apple choice and already halves leaf triangles**
  - _kind_: what-we-do-right
  - _source_: Tellusim: M1 compute-raster 2.30B tri/s > single-DIP HW 1.37B (A14 1.02B > 666M); Karis: 'vast majority SW-rasterized, ~3× faster than HW'; Karis: HW loses on micropoly (4 tris/clock, 2×2 quad waste, SV_PrimitiveID cost)
  - _ourCode_: NaniteRaster.ts:134-140 (orientForRaster), :954-956 (front-faces pass; two-sided leaf back-face RE-WOUND to CCW in place, rastered once)
  - _mechanism_: Trunks single-sided → classic back-face cull (areaNdc>0), the tellusim/themaister 'if(det>=0) return' reject. Leaf crowns two-sided but instead of a reversed-winding duplicate they re-wind whichever side faces the camera to CCW in place → HALF the leaf triangles/clusters, identical pixels. Directly attacks the dense-forest leaf tri count on the SW path where it is cheapest.
  - _fix_: No change. Keep the all-compute vis-buffer + two-sided single-raster; do not move mesh raster to HW or add a reversed-winding leaf duplicate.
  - _webgpuFeasible_: yes
  - _impact_: high
  - _confidence_: confident
- **Micro-poly / sub-pixel rejection is already present AND correctly measured neutral — themaister's 3.5-10× headline does NOT transfer to us**
  - _kind_: confirm-refute-hypothesis
  - _source_: themaister (micro-poly rejection all(notEqual(lo,hi)) = 3.5-10× in dense scenes 'over back-face cull alone'); Karis (small tris → vertex-transform+setup bound)
  - _ourCode_: NaniteRaster.ts:1006-1019 (SAMPLE-MISS 'CuRast' cull: coversSample, exact pixel-centre-in-bbox test) + smallEnough/validBB/area2>0 gates :986-1068
  - _mechanism_: themaister's big number is an RTX3070 63M-tri SETUP-bound scene, so culling a sub-pixel tri's setup pays. OUR per-TRI setup is measured FREE (rdbg2−rdbg1≈0, costmap): cost is the per-walked-pixel LOOP, and a sub-pixel tri already covers ~0-1px so rejecting it earlier saves almost nothing. Our sample-miss cull is exactly this reject and the code comment ('~neutral on world1's already-≤1px cut') matches. Chasing more tri-level culling to cut the pixel loop is a category error.
  - _fix_: None — do NOT invest in tighter primitive/micro-poly culling to reduce the pixel loop; setup is already free. Tri count matters only via OVERDRAW/walked-pixels (see DAG finding), not via setup.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: confident
- **Steal Karis's incremental depth gradient (ZX += GradZ) — we RECOMPUTE cz per walked pixel with int→float + 3 mul + 3 add**
  - _kind_: trick-to-steal
  - _source_: Karis SIGGRAPH-2021 SW rasterizer inner loop carries depth incrementally 'ZX += GradZ.x' (one fp add/pixel), gradient precomputed once per triangle
  - _ourCode_: NaniteRaster.ts:1202-1211 (per-pixel: uw=cw−bias ×3, then cz = toF(uw0)*z0 + toF(uw1)*z1 + toF(uw2)*z2, all mul rcpArea) recomputed for EVERY covered pixel
  - _mechanism_: The dominant cost is the walked-pixel body (setup is free), and depth interp is a large slice of it: 3 int→float + 3 fmul + 2 fadd + 1 fmul per pixel. The numerator N=Σuw_i·z_i changes by CONSTANT dN/dx = sx0·z0+sx1·z1+sx2·z2 per +1px (dN/dy per row) since uw_i is linear in x,y. Precompute dNdx/dNdy once per tri (free — setup untimed), per pixel: N += dNdx; cz = N·rcpArea (1 add + 1 mul). Single-pass world1 has NO cross-pass bit-identity constraint (the unbiased-weight rationale at :1191-1201 was for the OLD two-pass agreement; world1 computes depth once); float drift over ≤17px is sub-ulp vs the 24-bit key (far 1px slivers = 0 accumulation steps = exact).
  - _fix_: In emitFrag (+ swcoop/emitW1): hoist N0=Σuw_i(startX,startY)·z_i, dNdx=Σsx_i·z_i, dNdy=Σsy_i·z_i once per tri; carry N alongside the existing cw edge accumulators; cz = N·rcpArea. Coverage + election unchanged. Flag-gate, A/B c.gpuWall + shotdiff at the deep-dense pose.
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: speculative
- **Extend the row-solve bypass to NARROW slivers (extW≤4, any height) — deep-dense = 'sliver tris', row-span solve does 3 fp32 divides/row for no gain**
  - _kind_: trick-to-steal
  - _source_: Karis: scanline row-solve chosen only when the X interval >4px; micropoly (≤4px X-span) uses the plain per-pixel incremental walk with NO divide. costmap: pixel loop = 'walked-pixel cost of visible SLIVER tris'
  - _ourCode_: NaniteRaster.ts:1158-1183 (per-row: 3 fp32 divides den0..2 + floor/ceil + ~50 ALU for xLo/xHi) runs in the DEFAULT loop; swcoop=2 small-bin that drops it (:1298-1302) gates on SW_SMALL_EXT=4 for BOTH axes (:116)
  - _mechanism_: A tall-narrow sliver (extW≤4, extH>4 — the exact deep-forest bark/needle shape) does NOT qualify for the small-bin, so it pays the 3-divide row-solve on EVERY row while its inner x-span is 1-2px — the solve costs more than the ≤4px it prunes. Karis walks these per-pixel over the narrow bbox with incremental edges (no divide). The prior swcoop=2 'neutral' was a general-pose average; the deep-dense sliver regime was not isolated.
  - _fix_: Widen the row-solve-skip predicate to extW≤4 (X-narrow) regardless of height, walking the narrow bbox with the existing incremental cw edges. Build-time gated (swcoop exists). Re-A/B at the deep-dense surrounded pose specifically. Bit-identity holds (same cw≥0 test).
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: speculative
- **The real generator is OVERDRAW / a DAG that never coarsens — the pixel loop is correctly tapped; the only non-banned big lever is UPSTREAM in cull/DAG**
  - _kind_: confirm-refute-hypothesis
  - _source_: Karis: 'excessive overdraw from large/overlapping clusters, aggregates... riddled with holes describes leaves and grass — overdraw is why Nanite underperforms there'; our memory base-raster-is-the-bottleneck ('DAG never coarsens → 46M tris, 97% sub-pixel') + voxel-lod-only-two-levels-root-cause
  - _ourCode_: NaniteRaster.ts:1146-1286 (intrinsic per-walked-pixel loop) — cost ∝ walked pixels = Σ covered-pixels = overdraw×coverage; the direct px lever (?rscale internal-res) is BANNED by the quality bar
  - _mechanism_: Premise-audit: the pixel-loop cost is intrinsic per-walked-pixel and every IN-loop lever (row-solve, election, swmax, coop, occupancy) was measured dead (RASTER-arc 62d607e). One level up: walked-pixels = overdraw, high in deep forest because the leaf+bark DAG does not coarsen the near/mid field (millions of sub-pixel/sliver tris pile on the same pixels) and cluster occlusion uses prev-frame HZB that porous canopy pokes through. The dissolving lever lives in the CULL/DAG subsystem, not the pixel loop — which is why every in-loop attempt failed.
  - _fix_: Surface to the CULL/DAG audit as owner: aggressive near/mid DAG coarsening for leaf+bark aggregates (shotdiff-gated) to cut walked-pixels without touching resolution. Within raster, only the incremental-depth + narrow-sliver micro-levers remain. Do NOT relitigate rscale (banned) or in-loop micro-opts (dead).
  - _webgpuFeasible_: partial
  - _impact_: high
  - _confidence_: confident
- **Re-measure ?trihzb per-triangle occlusion at the DEEP-DENSE (canopy-wall) pose — it died on POROUS canopy, a different regime**
  - _kind_: confirm-refute-hypothesis
  - _source_: Karis: no per-triangle occlusion cull in Nanite (two-pass conflict + 1-thread/tri divergence) — flags it as a known gap; costmap MEASURED DEAD: '?trihzb fires (−13% frags) but −0.5ms — porous canopy defeats window occlusion'
  - _ourCode_: NaniteRaster.ts:1020-1065 (per-tri nearest-z vs prev-frame HZB 2×2 window over bbox, levels 0-4 tail-mirrored), built + default OFF (:298-304)
  - _mechanism_: trihzb was measured at a porous-canopy pose where gaps let occludees show through the coarse window, so −13% frags bought only −0.5ms. Deep-DENSE forest (surrounded, near canopy = a near WALL) is the opposite regime: many mid/far tris fully behind a solid near occluder, exactly where a per-tri HZB reject of walked-pixels should pay. The dead verdict may not hold in the trigger regime.
  - _fix_: Re-A/B ?trihzb=1 at the deep-dense surrounded pose (freeze=1 to avoid disocclusion holes), measuring c.gpuWall + frag counts; already built + tail-mirrored (dodges the 10-storage-buffer ceiling). If it pays, consider default-on gated on a density heuristic. NB: trihzb+scar combo = pipeline validation error (diag-only).
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: speculative
- **Profile r.nanHwPass — long thin leaf needles route to HW where HW is worst; deep-dense can hit ~876k hwTris**
  - _kind_: profiling-todo
  - _source_: Karis: HW rasterizer loses on thin/micropoly tris (4 tris/clock, 2×2-quad waste, SV_PrimitiveID cost); the whole point of SW raster is to avoid this
  - _ourCode_: NaniteRaster.ts:98-113 (leaf needles exceed the 16px i32-safe SW limit → HW; HW_CAP=2M, comment cites 876k hwTris in a dense stand), :931-943 (near-plane → HW), :1389-1398 (bbox>16px → HW); readHwCount() exists
  - _mechanism_: The SW/HW split (MAX_RASTER_SIZE=16, bounded by i32 edge-term precision <2^26) forces long thin near-camera leaf needles onto the HW vertex-pull path — the exact thin-sliver shape HW handles worst. In deep DENSE forest you are surrounded by near crowns → HW tri count and r.nanHwPass time can balloon, and it is a SEPARATE (additive) pass not in the 7.8/18.4 pixel-loop number.
  - _fix_: At the deep-dense pose, read readHwCount() + isolate r.nanHwPass gpuWall (?swmax sweep to shift the split). If HW is a real slab, fix is upstream: DAG far-crown shedding (two-sided re-wind already halved it) — SW can't take >16px slivers without i64 edge math (not WGSL-reachable). Quantify before acting.
  - _webgpuFeasible_: yes
  - _impact_: medium
  - _confidence_: speculative
- **Atomic cache-line false-sharing (a cluster's 128 adjacent tris → nearby pixels → same 64B lines) is un-actionable AND measured minor — do not re-stage threadgroup atomics**
  - _kind_: confirm-refute-hypothesis
  - _source_: Mali guide: space atomics 64 bytes apart / amortise into a workgroup-L2 atomic; Aaltonen: no 64-bit atomics on mobile
  - _ourCode_: NaniteRaster.ts:589-607 (election atomicMax on visPayloadV[px]); the ?relect ablation (:374-394) measured the election a MINOR share of the pixel loop
  - _mechanism_: One workgroup = one cluster whose 128 triangles are spatially adjacent → they atomicMax to pixels within a few 64-byte lines (false sharing, the Mali violation). BUT the target is a screen PIXEL (inherently 4B apart), not a reduction slot, so the 64B-spacing / L2-amortise trick cannot apply — you cannot pre-combine writes to distinct pixels. The aLoadU pre-gate already suppresses most RMWs, so ?relect proved the election is not the driver (the walk+interp is). The prior on-chip wgElect/flush-merge was built and reverted as measured-dead.
  - _fix_: None — confirmed correct and un-actionable. Do NOT re-stage threadgroup-atomic election or attempt cache-line spacing on per-pixel targets.
  - _webgpuFeasible_: no-metal-only
  - _impact_: low
  - _confidence_: confident
- **The c.nanVisClear ~11ms timestamp is almost certainly a batched-submit OVERLAP artifact, not real clear work — isolate before treating as a slab**
  - _kind_: profiling-todo
  - _source_: task brief ('per-pass timestamps OVERLAP — Apple runs render||compute concurrently, NOT additive; ablation is the truth'); Burns-Hunt (vis-buffer writes are a few bytes/sample — clears are cheap)
  - _ourCode_: NaniteRaster.ts:543-563 (kVisClear, pixelCount threads @[256], already SKIPS the depthV clear via skipDepthClear default-on :524-529); world1 rides the SAME coalesced submit as grass cull+raster
  - _mechanism_: Clearing 2 buffers × pixelCount u32 at dpr2 is ~0.6ms of write traffic at Apple bandwidth, not 11ms. Because kVisClear is folded into the coalesced submit that also holds grass cull + grass raster + world1, its timestamp span plausibly encompasses concurrent compute — the non-additive overlap the brief warns about. Treating 11ms as real clear cost would target an artifact.
  - _fix_: Isolate: A/B ?dvclear=1 (restore depthV clear) vs default, and a clear-only micro-ablation (skip world1) at the deep-dense pose to read the true clear time. Only if it survives isolation is there anything to optimise (depthV is already skipped, so buffer count is minimal). Do not chase otherwise.
  - _webgpuFeasible_: yes
  - _impact_: low
  - _confidence_: speculative

---

## (agent)

Written to `/Users/sebastian/IdeaProjects/fable-demo2/docs/mobile-gpu-perf/DEEP-FOREST-AUDIT.md`.

## Cost model (17.4ms gpuWall base, grass OFF — NOT 20.8ms)

The brief's "20.8ms" is the `render` *pass timestamp* which overlaps the concurrent 8.72ms compute stream; the ablation (`deep-forest-breakdown.md`) says the wall is **17.4ms**. Two premise corrections the ablation forces: `c.nanVisClear ~11ms` is a concurrency artifact (drops 7.73→1.44 when the leaf mesh is removed — it's swallowing the concurrent leaf SW-raster), and `r.half.mrt ~10.5ms` is POST (clouds/GTAO/bounce), not shadows (shadows ablate to 0.5ms).

Marginal costs (overlap, don't sum):
- **Leaf mesh `naniteleaf`: 4.1ms** + **Crown LOD0 mesh `crownlod0`: 3.9ms** → **≈8ms combined = the leaf-crown SW-raster, the whole compute stream**
- Voxel scatter: net **−54ms** vs mesh fallback (own fill ~3–6ms)
- Resolve `r.scene` ~5ms (render stream, overlaps)
- Post ~5ms (rotated toward contact+AO in forest)
- Cull/HZB ~2–3ms; Shadows 0.5ms static (~8ms MOVING = the stutter); vis-clear real <1ms

Two poles: **leaf-mesh raster (compute)** and **resolve+post (render)**, running concurrently. Everything else <3ms.

## The single biggest opportunity

**The near/mid leaf-crown MESH-triangle SW raster (~8ms, the whole compute stream).** Premise-audit one level up: near crowns render as full-density LOD0 leaf triangle meshes (`NaniteCull.ts:1074`) whose DAG never coarsens — millions of sub-pixel leaf tris pile overdraw on the same pixels. Same disease as the voxel/base-raster no-LOD story; every in-loop raster lever was already measured dead because the generator is upstream in the DAG. `?rscale` would dissolve it but is banned. The surviving lever: **push the voxel near-band (`voxnear`) inward** (voxels are net −54ms, strictly cheaper than the mesh they replace) and/or coarsen the leaf-crown DAG, shotdiff-gated. Halving this moves the frame ~4ms — more than any other lever.

## Top 6 levers (ranked by deep-forest impact)

1. **L1 — Push voxel near-band in / coarsen leaf-crown DAG** (the ~8ms whale). HIGH, CONFIRMED slab / SPECULATIVE quality. Measure: `voxnear`+`crownlod0` shotdiff at shipped density.
2. **L2 — Flip forest resolve to two-pass (`respass=0`)**. Single-pass `'both'` merges terrain-derivative + all material subgraphs into the voxel shader → occupancy collapse on the >90% voxel path (`NaniteResolve.ts:1574-1591`, in-code "37.5ms cliff" note at :556-561). HIGH, CONFIDENT, **cheap default-flip**. Measure: `?respass=0` A/B.
3. **L3 — Ping-pong the vis buffers** to kill the cross-queue WAR hazard that serializes clear+raster behind the prior resolve and inflates `nanVisClear`. HIGH-if-real, SPECULATIVE. Measure: no-op-clear A/B, then `?visping` A/B.
4. **L4 — Split the voxel megakernel** into a lean near-scatter kernel (ray/mask/pyramid closures gated off) to stop register spill (`NaniteVoxelRaster.ts:571-1564`). HIGH-if-register-bound, SPECULATIVE — **gated on a Metal capture** (`?voxrdbg=2` first).
5. **L5 — Post: half-res contact shadows + temporal GTAO.** In forest contact marches full-res 12 steps on every pixel (`PostStack.ts:451-504`) and AO loses its far-fade skip. MEDIUM-HIGH; CONFIDENT (contact) / SPECULATIVE (AO temporal). Measure: `ablate=contact`/`ablate=ao` (post has never been ablated at this pose).
6. **L6 — Tame the MOVING shadow strip re-raster + `shvox2`** (~8ms p50 walking vs 0.8ms still) — the stutter symptom. MEDIUM-HIGH on p95. Measure: `?nanshadow=0`/`?shvox2=0` at a MOVING pose.

Instrument-first order: L2 (cheapest, no Xcode) → L3 disambiguation → post forest ablation → voxel fill isolation → moving-pose shadow leg → Metal capture (gates L4) → L1 last to build.
