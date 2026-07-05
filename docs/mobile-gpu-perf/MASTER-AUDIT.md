# MASTER AUDIT — Apple Silicon / mobile TBDR dense-foliage bottleneck

Consolidates 6 theme audits (apple-arch, occupancy-ilp, grass-raymarch, visbuffer-mobile,
webgpu-wgsl, apple-wwdc) + the earlier `CROSS-CHECK.md` (occupancy-interplayoflight,
apple-tbdr-blakecrosley, shader-opt-persson, wwdc-10859, arm-mali, Aaltonen HypeHype 2023).
Task #72. Every lever is grounded in BOTH a source AND our code (file:line, verified live on
`nanite-raster` HEAD 2026-07-05). Tags: **CONFIDENT** = source + code + a measured fact;
**SPECULATIVE** = source + code, win-size unmeasured on Apple.

Measured facts held fixed: grass per-pixel raymarch (`kGrassRay`) is the **#1 Apple cost, eye
+14.5 ms gpuWall**; dense foliage is the trigger; confirmed **NOT thermal, NOT bandwidth**
(1.5× headroom). Every lever below is judged against those. Findings that target the resolve,
SW raster, or voxel scatter are **downgraded** — those were measured NOT to be the grass
bottleneck (grass-lean A/B `NaniteGrass.ts:134-141`; RASTER-arc closed 62d607e).

---

## HYPOTHESIS VERDICT — what limits us, and the one step that settles it

**Verdict: MIXED, dominated by dependent-fetch LATENCY that only occupancy can hide — NOT a
FLOP/scalar-ISA (M5) problem, and NOT proven to be occupancy *collapse*.**

The grass march is structurally the worst possible TBDR shape and this IS confirmed in code:
`NaniteGrass.ts:870` is a `loopUN('gro', 0, 256)` whose every step's fetch **address depends on
the previous step's `tCur` advance** (guide-ctx `:892`, mask `:921/:1121`, two field textures
`:957-958`, 3D tile `:1041`, +4 ground loads on accept `:1086`), and the L2 path `:1335-1499` is
a *second* full fetch-decode chain inline. That is ILP≈1 across iterations plus a long dependent
chain within a step — the latency-exposed regime where fp32 FFMA costs 5–11 cyc and the only
latency-hider is swapping to another resident wavefront (occupancy).

What the evidence does NOT yet distinguish:
- The cited **47.6→15.1 ms** win from "shrinking live state" is consistent with BOTH occupancy
  relief (fewer VGPRs → more resident simds) AND simply deleting expensive per-step ALU. Nobody
  has read the actual VGPR count / occupancy% / spilled bytes. The code *asserts* occupancy-bound
  (`:147`); that is a hypothesis, not a measurement.
- The digest's own caveats bite here: on **M1 Max (Apple 7) fp16 is NOT double-rate** — F16-FMA
  and F32-FMA both run 256 ops/core-cycle (metal-benchmarks lines 126-127), so the "double-rate
  fp16" and "scalar-ISA float4 decomposition" framings are **M5-only and do not transfer**. Our
  vec math is component-independent (good ILP already), so the M5 "scalarize float4" trap does not
  apply. fp16's payoff on OUR hardware is purely register/occupancy relief.
- Volkov shows 25% occupancy can suffice via ILP — but a serial march has **no ILP to add**, so
  occupancy is the decisive axis *if* the limiter is latency. Do NOT assume collapse; test it.

**The single settling step:** a **native Metal GPU capture of the Chrome/Dawn GPU-process**
while our dense-forest scene runs. Dawn lowers WGSL→MSL, so Xcode ▸ Debug ▸ *Capture GPU Frame*
(or Instruments ▸ Metal System Trace) attaches to the GPU helper and reads, for `kGrassRay` and
`kVoxScatter`: (1) **occupancy %**, (2) **spilled bytes**, (3) the **single top limiter** (ALU vs
buffer/LLC-atomic vs latency). Tech-Talk-10580 compiler statistics show spilled-bytes + occupancy
directly — that is exactly how the "16% occupancy from spilling" case was diagnosed. Decision rule
(occupancy-explained): occupancy healthy but perf poor ⇒ **latency-exposed** ⇒ cut fetches/steps
(levers 3–5); VGPR just above a wave-count step or spilling ⇒ **register-bound** ⇒ shrink live
state (levers 2, 4, 6). **Ship nothing precision/register-related before this reading.**

---

## TOP LEVERS (ranked by expected impact on the +14.5 ms grass frame)

### 0 — INSTRUMENT FIRST (gates all of the below) · CONFIDENT · impact HIGH
- **Change:** run the Metal capture above; in parallel read the per-pass timestamp map that is
  **already wired** — `Engine.ts:86` `trackTimestamp:true`; labelled passes `grassRay`
  (`NaniteGrass.ts:1540`), `grassGuide` (`:616`), `grassLight` (`:680`), plus `?grass=0`,
  `?nores=1`, `?grasslayers=1` (`:128`) ablations to split march vs bake vs light vs L2.
- **Source:** apple-wwdc (WWDC20-10603 GPU counters, Tech-Talk-10580 spill/occupancy stats);
  webgpu-wgsl (threejsroadmap profiling recipe — the harness we already run).
- **WebGPU:** yes (timestamps in-engine; capture is a native tool on the Dawn process).
- **First measurement / it IS the measurement:** kGrassRay occupancy%, spilled bytes, top
  limiter; and c.grassRay share of frame at the dense oblique/down-look pose.
- **Why:** every precision/register/fetch lever below is currently INFERRED. Also confirm the
  test Chrome is ≥130 (Tint IR, chrome://gpu) so any boot-compile numbers aren't on the slow
  translator.

### 1 — Bake `widenT` into the dead 4th guide-ctx word · CONFIDENT · impact MED · SHIP-NOW
- **Change:** `NaniteGrass.ts:524` writes `uint(0)` to guide-ctx word `base+3` (dead). The march
  RE-derives `widenT = 1/sqrt(grassThin(distT))` every accepted step (`:926-927`) — `grassThin`
  contains **two `pow` + a `sqrt` + divides** — yet the bake already computes `widenT` at `:501`.
  Pack `widenT` (or `packHalf(widenT, spare)`) into word `base+3`; replace the `:926-927` block
  with an unpack of `cv.w`.
- **Source:** occupancy-ilp (AMD register-pressure: replace transcendentals with precomputed
  values, shrink live ranges) + the file's own guide-bake pattern.
- **WebGPU:** yes — pure WGSL/TSL, the word is already allocated and zero-filled.
- **Expected impact:** removes 2 pow + sqrt + 2 div AND a live temporary from the hot loop every
  accepted step; zero storage cost. Constant-per-texel value baked to the O(area) pass — exactly
  what the guide exists for. **The single most confident, lowest-risk code change; not gated on
  the capture.**
- **First measurement:** c.grassRay A/B at a dense pose + shotdiff parity (texel-center vs
  step-exact dist differ negligibly at 0.84 m grain).

### 2 — Route the per-step guide CTX + MASK through the texture cache (SSBO → texture) · SPECULATIVE (win size) · impact HIGH-MED
- **Change:** the fields ALREADY moved to filterable `StorageTexture` (`guideFieldT1/T2`,
  `:411-422`) and that WON — proof the round-trip pays. The **ctx** (ground/grad/top/amp,
  `guideCtx4` `StorageBufferAttribute` `:389-391`, read per step `:892`) and the **64-bit mask**
  (`guideMask2` `:395`, read per step `:921/:1121`) are the leftover dynamic-address SSBO loads.
  Move ctx to a filterable texture at the ray's world XZ (ground → `r32float`; grad/top/amp →
  `rgba16f`). Keep the mask INTEGER (density law must stay bit-identical) as an `rg32uint` read
  via `textureLoad` (no filtering → no NaN canonicalisation, sidesteps the `:156-158` blocker,
  which is a node-builder uint-typing worry, not a hardware one).
- **Source:** webgpu-wgsl + CROSS-CHECK F2 (Aaltonen: "SSBO loads from dynamic addresses are slow;
  scalarize memory access" — the mobile headline weakness); arm-mali (route hot reads through the
  texture cache, avoid divergent SSBO addresses across the quad — at Q=2 the lanes' `ti` diverge).
- **WebGPU:** partial — reachable; the mask needs the `rg32uint`+`textureLoad`+bitcast dance to
  dodge the TSL float-typing of uint storage textures.
- **Expected impact:** moves the dependent-path loads off the slow dynamic-SSBO path onto Apple's
  spatially-coherent texture unit — the same move that already won for the fields.
- **First measurement:** from the capture, confirm buffer-load / ctx+mask fetch is a real share of
  c.grassRay BEFORE converting; then A/B mask-as-texture vs mask-as-SSBO on M1.

### 3 — Distance-adaptive Q-quad in the far band · SPECULATIVE · impact MED
- **Change:** `Q` (1 ray per Q×Q px) is already auto-2 at dpr≥1.75 (`:781-785`) and is the single
  biggest applied perf lever. Make `Q` a function of the pixel's election-depth bound (`tMax`):
  near → 1–2 (crisp silhouettes), far (>~60 m, already sub-pixel/near-splat) → 4. The per-pixel
  emit fan (`:1512`) keeps nearer true geometry winning each pixel, so silhouettes don't degrade.
- **Source:** grass-raymarch (Babylon: cost is exclusively per-pixel-weight × pixels covered).
- **WebGPU:** yes — the march is already depth-bounded; only dispatch tiling + fan bounds change.
- **Expected impact:** up to 4× fewer far-band rays where the loss is invisible; attacks the
  divergent-deep-trip-count lanes (down-look/aerial = the dense trigger poses).
- **First measurement:** c.grassRay at an aerial/oblique pose, ?grassquad sweep, shotdiff parity
  on the far band.

### 4 — Hoist the L2 golden-overlay out of `kRay` into a sparse second pass · SPECULATIVE (gated on capture) · impact MED
- **Change:** the entire `RAY_LAYER2` block (`:1335-1499`) — a second bomb/shear/fetchBand/decode
  chain with ~20 own live vars — is compiled INTO `kRay`, so the compiler reserves its worst-case
  VGPR footprint for **every** lane, including the dominant upward-looking case that never enters
  it (fires only on `rd.y<-0.3`, budget 3/px). Move it to a second dispatch over just the pixels
  L1 left empty.
- **Source:** occupancy-ilp (Andersson/DICE: switch/branch → worst-case VGPR/LDS across ALL paths;
  splitting a mega-shader restores tight bounds — the "occupancy 5→2 for free" anti-pattern).
- **WebGPU:** yes.
- **Expected impact:** drops L2's live set + instruction footprint from every dominant-case lane
  → higher occupancy for the frame's majority pixels. **Only pays if the capture shows kRay VGPR
  above a wave-count step / spilling.**
- **First measurement:** `?grasslayers=1` (L2 off, `:128`) at down-look poses first — bounds the
  ceiling; then implement the split only if register-bound is confirmed.

### 5 — Toroidal / incremental guide bake (cuts `grassGuide`, not the march) · SPECULATIVE · impact MED
- **Change:** the guide window snaps to a texel grid each frame yet the WHOLE ~384² field is
  rebaked every frame (`kGuideBake` `:424-618`, dispatch `:615`). Make it toroidal like the shadow
  clipmap — rebake only the newly-exposed ring on camera motion + the always-changing wind amp.
- **Source:** CROSS-CHECK #2 (Aaltonen: process at the right frequency / temporal coherency);
  precedent `NaniteShadowClip.ts` (toroidal clipmap, strip re-raster).
- **WebGPU:** yes.
- **Expected impact:** bake pass full-area → thin strip. Note this targets c.grassGuide, a cost
  SEPARATE from the +14.5 ms march — additive, not the main whale.
- **First measurement:** c.grassGuide A/B while walking.

### 6 — fp16 the march-LOCAL state (occupancy lever, NOT a FLOP lever) · SPECULATIVE (gated) · impact MED, high effort
- **Change:** demote to f16 ONLY the tile-local quantities — tile-space `qbx/qbz` (∈[0,1]),
  swirl `ca/sa`, sub-meter `offX/offZ`, azimuth `az`, baked normals, `tPar/topEff`. KEEP f32 for
  everything world-anchored: `pos.xyz` (±155 m ring; f16 ~0.06 m at 128 m is coarser than a
  blade), `tCur/tEnd` (absolute t over 256 steps), ground Y. Mirror the precision split the
  guide-ctx already ships (`:521-523`: ground f32, grad/top/amp half).
- **Source:** apple-arch + apple-wwdc + occupancy-ilp (fewer registers → more resident simds).
  **Reframed by the premise-audit:** on M1 fp16 is NOT double-rate (256:256 FMA) — the win is
  register/occupancy relief + lower dependent-op latency, NOT FLOP doubling. Expect a move to a
  cheaper ALU-bottleneck row, not a 2× speedup.
- **WebGPU:** **partial / blocked at the TSL layer.** three r184 TSL has no half node type
  (`NodeUtils.js`), and the device is created without `requiredFeatures:['shader-f16']`
  (`Engine.ts:86`). Needs `requiredFeatures:['shader-f16']` (guarded by `adapter.features.has`)
  + a raw-WGSL `FunctionNode` escape for the hot inner block. Dawn/Metal support it; our authoring
  layer is the blocker.
- **Expected impact:** further register relief on top of levers 1/4 — only worth the raw-WGSL
  complexity if the capture shows register-block-occupancy or spilling as the limiter.
- **First measurement:** the capture (register-bound?), then Tint-emits-packed-f16 check +
  bit-clean shotdiff.

### 7 — (ARCHITECTURAL, USER-GATED — surfaced, not silently recommended) revive the analytic closed-form blade lane · SPECULATIVE · impact HIGH-if-taken
- **Claim:** the grass-raymarch theme's strongest finding — the DELETED closed-form analytic
  blade (`P = A + B·by + C·by² + E·u·W`, git 102669c) measured **march-only 2.36 ms** vs the
  baked-tile full look **+14.5 ms**, because it has ZERO dependent texture fetches (pure ALU,
  Babylon's exact recipe) and sidesteps levers 2/4/6 entirely.
- **Why it is a USER call, not an engineering pick:** the per-pixel-baked-tile march lane was a
  user directive (geo/hybrid lanes deleted 70e4d71) and the LUSHNESS LAW forbids thinning; the
  arc-overhang gap that motivated replacing the analytic blade was separately solved by the
  (also-deleted) statistical coverage band. Reviving analytic-near + statistical-mid is a lane
  redesign — per project rule, **surface it for the user's call**, do not silently switch.
- **First measurement:** if the user greenlights, resurrect from 102669c behind a flag, A/B
  c.grassRay + lushness shotdiff at shipped density.

---

## WHAT WE ALREADY DO RIGHT (do NOT spend effort here)

- **All-compute vis-buffer is the CORRECT Apple choice** — Tellusim measured M1 compute-raster
  2.30B tri/s > single-DIP HW 1.37B. The Mali "compute shading not recommended" warning is
  Mali-specific (loss of FB compression / VS-FS overlap) and does NOT transfer to Apple.
- **32-bit split election + `aLoadU` pre-gate + no 3rd atomic buffer** — `NaniteRaster.ts:595-607`
  (guard) and `:1267-1268` ("3rd atomic storage buffer = 3× cliff, 15-17 ms"). This is the correct
  no-64-bit-atomic port (WGSL has no atomic<u64>); the single most important Apple election
  decision. Voxel scatter uses it VERBATIM (`NaniteVoxelRaster.ts:383-389`), with the on-chip
  wgElect/flush-merge already built-and-reverted as measured-dead. **Do not regress; any future
  exact-per-fragment-depth feature must fold into the existing election word, never add a buffer.**
- **Single-pass deferred resolve = HSR / overdraw≈1 by construction** — shades the winner once
  (`NaniteResolve.ts`), strictly better on M1 bandwidth than Nanite's per-material fullscreen +
  depth-EQUALS multipass. And it is a **fragment** pass (keeps DCC), not compute — keep it there.
- **`presentClasses` strips absent material classes** → register/occupancy relief
  (`NaniteResolve.ts:99-104`) — the same occupancy lever, already applied where it matters.
- **Guide fields already on filterable textures; guide-ctx already half-packs** grad/top/amp with
  ground kept f32 (`:521-523`) — the correct "fp16 where it packs, f32 where precision demands"
  split. Reference pattern for lever 6.
- **Empty-space skip + texel/cell DDA already implemented** (`:914-919`, `:896-909`) — Babylon's
  corridor + Cloudscapes' adaptive step are already spent; do not re-chase. Residual cost is
  per-IN-SWARD-step fetches, not wasted empty steps.
- **GoT cheap-look tricks already in** — texture-driven density baked to occupancy bits
  (`:457-497`), rounded-normal-from-flat (`:203-213`), clump variety. Not levers.
- **`voxRecip` default-ON** (`NaniteVoxelRaster.ts:257`) — Apple has no HW int-divide; the march
  is already int-div-free. Standing rule: no new per-step `uint .div()/.mod()` on a hot path.

---

## BLOCKED BY WEBGPU (Apple/Metal tricks we cannot reach) + second-best

| Metal trick | Why blocked | Second-best (WGSL-reachable) |
|---|---|---|
| **64-bit `atomicMax`** (M2 Nanite fast path) | WGSL has no `atomic<u64>`, no texture atomics | Already done: 24-bit depth-key + 8-bit tiebreak split (optimal) |
| **memoryless / tile-memory / imageblocks / programmable blending / ROV** | Not exposed in WebGPU; Dawn manages residency; vis-buffer can't be tile-resident | None — but we are measured NOT bandwidth-bound, so the loss is low-priority |
| **`shader-f16` / `half` math** | Dawn/Metal SUPPORT it; the blocker is **three r184 TSL has no half node** + device lacks the feature (`Engine.ts:86`) | `requiredFeatures:['shader-f16']` + raw-WGSL `FunctionNode` for the hot block (lever 6) |
| **Filtered storage buffers** (swap baked 3D tile/fields to raw packed loads) | WebGPU storage buffers have no HW filtering; manual trilinear = 8 loads + blend = strictly WORSE | Keep the filtered textures (already the right call) |
| **`[[early_fragment_test]]` / HW HSR for shading** | We don't use the HW depth path for shading | Our atomic-election vis-buffer already gives overdraw≈1 |

Note: "avoid device atomics → threadgroup atomics" is **inapplicable/refuted** for us — grass
elects each pixel once (zero cross-lane contention) and the voxel wgElect was measured-dead. Do
not re-stage. Also `[64]` vs `[256]` workgroup size (`:1539`) is a cheap A/B if the capture flags
threadgroup-memory/occupancy, but shared memory is NOT a violation on Apple (real fast on-chip
threadgroup mem, unlike Mali) — do not strip the cooperative designs.

Boot-compile (SEPARATE arc, not the frame bottleneck): shadow clip builds 6 byte-identical raster
pipelines (`NaniteShadowClip.ts:312` loop over LEVELS) and HZB up to 16 distinct
(`NaniteHzb.ts`), each a per-pipeline WGSL→MSL XPC compile on Apple. Dedup / `?pyrfuse` re-bench
belongs in the BOOT arc, judged on boot pipeline-creation time — not here.

---

## RECOMMENDED SEQUENCE

- **Phase 0 — instrument (before ANY precision/register code):** Metal GPU capture of the
  Chrome/Dawn process → kGrassRay + kVoxScatter occupancy% / spilled bytes / top limiter; plus the
  in-engine timestamp map (`?grass=0`, `?nores=1`, `?grasslayers=1`). Confirm Chrome ≥130. This
  answers the hypothesis verdict and routes Phases 2–3.
- **Phase 1 — ship-now, capture-independent:** lever 1 (bake `widenT`). Pure ALU + live-var cut,
  no risk, no gate.
- **Phase 2 — IF capture says latency/fetch-bound:** lever 2 (ctx+mask → texture), lever 3
  (distance-adaptive far-band Q).
- **Phase 3 — IF capture says register-bound / spilling:** lever 4 (L2 hoist to sparse pass),
  then lever 6 (fp16 local state via raw-WGSL, highest effort/lowest certainty).
- **Phase 4 — additive bake-pass win (independent):** lever 5 (toroidal incremental guide bake).
- **Surface to user:** lever 7 (analytic-blade lane, measured 2.36 ms) — the biggest lever but a
  lane-redesign that conflicts with the user's march directive + lushness law. User call.

Do NOT invest in: more election micro-opts (coverage-bound, A+B·pixels, R²≈1 — the atomic is a
minor share), `?trihzb` (−0.5 ms, porous-canopy weak), `?swmax` beyond one M1 oblique re-sweep,
de-vectorizing vec ops (M5 trap, N/A to our independent-component vecs), threadgroup-atomic
re-staging (no contention / measured-dead), or memoryless/f16-resolve (blocked / not the whale).
