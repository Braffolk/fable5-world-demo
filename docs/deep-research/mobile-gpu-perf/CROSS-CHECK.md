# Mobile / Apple TBDR best-practices — cross-check vs OUR renderer

Grounded audit: documented Apple/mobile GPU practices vs what `src/nanite/*` actually does.
Every claim cites BOTH the source doc AND our code (file:line). Read against the measured
facts (grass raymarch = the dominant Apple cost at eye +14.5 ms gpuWall; dense foliage =
the trigger; confirmed NOT thermal). Ranked by expected impact on the M1 Max dense-foliage
case; each finding is tagged **CONFIDENT** (grounded in both source + code + a measured fact)
or **SPECULATIVE** (grounded in source + code, but win size unmeasured on Apple).

Sources: `occupancy-interplayoflight.md`, `apple-tbdr-blakecrosley.md`, `shader-opt-persson.md`,
`wwdc-10859-tailor-metal-m1.md`, `arm-mali-guide.md`, `hypehype-advances-2023.pdf` (Aaltonen,
SIGGRAPH 2023 — the closest match: a GPU-driven compute renderer targeting mobile TBDR).

---

## PREMISE-AUDIT (fired before concluding)

The cost I target — the grass per-pixel raymarch — is REAL and in-code documented:
`NaniteGrass.ts:99-100` ("the ≤2 ms law is NOT met yet — eye +14.5 gpuWall … the ray-march
perf pass is the open follow-up") and the perf ledger. It is consistent with the measured
facts (grass = #1 Apple cost; dense foliage = trigger; not thermal). So findings that target
the raymarch are high-value; findings that target the resolve or the SW/voxel raster are
downgraded, because those were **measured NOT to be the grass bottleneck** (the grass-lean A/B
at `NaniteGrass.ts:134-141` refuted the "resolve wall"; the RASTER-CLOSED arc found the raster
loop is intrinsic per-pixel work with all levers dead). I dropped three plausible-looking
findings after reading the code (see "NON-FINDINGS" at the end) — they were opt-in debug paths
or already-fixed, i.e. the setup one level up, not a live violation.

**The one-level-up finding (surfaced, not parked):** all four sources converge that a long,
divergent, **dependent-fetch** per-pixel loop is the single worst fit for a mobile TBDR GPU
(occupancy article: "the enemy is DEPENDENT fetches that stall"; Aaltonen slide 6: SSBO
dynamic loads slow, no wave ops, poor for "memory heavy workloads"). Our grass is the ONLY
foliage type rendered that way. Trunks (SW raster) and leaves (voxel scatter) go through the
**bounded-work election-scatter** path and are "measured levers dead" — i.e. already the
cheaper primitive. Grass being +14.5 ms while the others are tapped-out is therefore
*structural*, not a tuning miss. The per-pixel-march framing was a user directive (the geo/
hybrid lanes were deleted 2026-07-04) and the LUSHNESS LAW forbids thinning — so revisiting
the framing is a **USER call**, not something to silently redesign. Everything in §1–§3 below
is what's actionable *while keeping the raymarch lane*.

---

## 1. VIOLATIONS / WORST-CASES WE HIT (ranked)

### F1 — Grass march = a long DEPENDENT-FETCH loop with divergent trip count  ★ CONFIDENT (the cost) / SPECULATIVE (fix size)
**Practice:** occupancy-interplayoflight.md:30 — "profile for stalls on actual memory reads;
low occupancy is fine IF ILP hides latency; the enemy is DEPENDENT fetches that stall."
Aaltonen slide 6 — mobile GPUs are "not designed for memory-heavy workloads"; slide 5 — "SSBO
loads vs UBO (uniform access fast path) … if you scalarize your memory access patterns you hit
the sweet spot."
**Where:** `NaniteGrass.ts:870` — `loopUN('gro', 0, 256, …)`, a per-pixel march of up to **256
iterations**. Each occupied step issues a chain of fetches whose **address depends on the
previous step's advance** (`tCur` moves by the fetched `top`/`dTile`): guide-ctx load
`guideCtx4.element(ti)` (`:892`), mask load `guideMask2.ro.element(ti)` (`:921`), two field
textures (`:957-958`), a 3D tile fetch (`:1041`), and on accept 4× ground loads
`smoothGroundAt` (`:1086`). The L2 golden-layer path (`:1335-1499`) is a *second* full
fetch-decode inline.
**Mechanism / why Apple:** each fetch stalls until data arrives; because the next address
depends on this result, the compiler cannot hoist the next fetch to hide latency (no ILP). The
only latency-hider left is occupancy (swap to another wavefront) — but the kernel is
register-heavy (F3) so occupancy is low. Worse, the loop's **trip count is data-dependent**:
across a SIMD group, a lane that hits a near blade exits in ~5 steps while a lane looking down
a field marches to `RAY_END=155 m`; the group runs until the LAST lane finishes (`:835`,
`:870`). Dense foliage = every lane marches deep + neighbour rays diverge in trip count ⇒ the
exact "drops to <30 fps under dense foliage" signature.
**Impact: HIGH.** This is the +14.5 ms. It is the umbrella that F2/F3/F4 feed.

### F2 — Per-step dynamic-address STORAGE-BUFFER loads in the march  ★ CONFIDENT (violation) / SPECULATIVE (fix size)
**Practice:** Aaltonen slide 5-6 (the headline mobile weakness: "SSBO loads from dynamic
addresses are still slow … scalarize your memory access patterns"). arm-mali-guide.md:128,184 —
"Avoid dynamic indexing into buffer arrays … it disables pilot shaders" (register-mapped
uniforms). arm-mali:117-120 — access sequential/overlapping addresses across the quad; "do not
access divergent addresses across a thread quad."
**Where:** `NaniteGrass.ts:892` `guideCtx4.element(ti)` and `:921/:1121` `guideMask2.ro.element(ti)`
— `ti` is computed per step from the ray's world XZ, a fully **dynamic** storage-buffer index.
`smoothGroundAt` adds 4 more (`:1086`). Guide ctx/mask are `StorageBufferAttribute`
(`:389-395`), *not* textures — chosen because "uint StorageTexture is mistyped 'float' by the
node builder; float-texture roundtrips could canonicalize mask NaN bit patterns"
(`NaniteGrass.ts:156-158`) — a **three.js TSL typing limitation + a MASK-specific NaN worry**,
not a hardware one.
**Mechanism / why Apple:** the SSBO load path lacks the uniform/texture fast path Apple/mobile
optimise; dynamic indices further defeat the "pilot shader" register promotion. At Q=2
(`:782`, auto-on ≥ dpr1.75) the SIMD lanes are 2-px-apart ray *centres*, so after a few steps
their `ti` diverge ⇒ divergent SSBO addresses across the quad = no load-merging.
**Impact: HIGH-MED.** On the critical dependent path; a top lever for the perf pass.

### F3 — kRay register pressure / live state → collapsed occupancy (the acknowledged bottleneck)  ★ CONFIDENT
**Practice:** occupancy-interplayoflight.md:21-27 — VGPR count is inversely tied to occupancy;
">128 VGPRs ⇒ ZERO wavefronts schedulable." arm-mali:102-103 — "Large workgroup sizes restrict
the number of registers available to each work item … forcing stack memory (spilling)."
**Where:** the `kRay` Fn (`NaniteGrass.ts:788-1539`) is a single ~750-line function with a
very large live-var set (`Slx,Slz,Swx,Swz,Sqx,Sqz,ca,sa,texOx/z,texCx/z,hgt,tanX/Z,ex/ez,az,
HB,…` plus the entire L2 duplicate at `:1335-1499`). Dispatched at workgroup **`[256]`**
(`:1539`). The code itself states the diagnosis: "the kernel is occupancy-bound: 47.6→15.1 ms
came from shrinking live state, not ALU" (`:147`) and did the field-bake specifically "to
collapse its live-register state" (`:858`).
**Mechanism / why Apple:** WGSL allocates registers for the *whole* function, so the rarely-
taken L2 path inflates the register footprint of *every* lane. High registers/lane × 256-lane
group ⇒ few groups resident ⇒ nothing to swap to when F1's dependent fetch stalls. This is the
same failure the resolve already hit and fixed for the vox pass (`NaniteResolve.ts:556-561`:
the derivative-forcing terrain subgraph "inflates register/instruction pressure → collapsed
occupancy → the per-pixel voxel-decode latency chain can't be hidden … dominant driver of the
close-up voxel cliff, 37.5 ms"). Same disease, in grass.
**Impact: MED-HIGH.** The proven medicine (shrink live state) is exactly what already bought
47.6→15.1; there is more of it to take.

### F4 — 3D trilinear baked-tile fetch, once per in-sward step  ★ SPECULATIVE
**Practice:** arm-mali:88-89 — "Trilinear (LINEAR_MIPMAP_LINEAR) has a 2× cost. 3D formats have
a 2× cost." Aaltonen slide 6 — SampleGrad ⅛-rate (a related "fat fetch is expensive" point).
**Where:** `NaniteGrass.ts:1039-1047` `fetchBand` → `texture3D(rayBake.texs[i], vec3(u,v,w))`
on a `Data3DTexture` with `LinearFilter` on all three axes incl. the repeat-wrapped angle
(`:734-737`). Fetched every in-sward step; the L2 path fetches it again (`:1378`).
**Mechanism / why Apple:** a linear-filtered 3D sample interpolates 8 texels across 3 axes —
inherently heavier than a 2D bilinear tap, and it lands on the dependent path. The 3D-ness is
*inherent* to the article's (x, z, angle) tile, so it can't be removed — only possibly made
cheaper (2D-array-with-nearest-angle; see fix). Apple's 3D vs array cost is not documented, so
this is speculative for us specifically.
**Impact: MED-LOW.**

### F5 — Election atomic contention under dense overlap (all three producers)  ★ CONFIDENT (pattern) / SPECULATIVE (that it's material)
**Practice:** arm-mali:136-139 — "space atomics 64 bytes apart to avoid contending on the same
cache line … or amortise by accumulating into a shared (L2) atomic and have one thread push the
global at workgroup end." Aaltonen slide 8 — "64-bit atomics are commonly used in SW
rasterizers … there's no 64-bit atomic support in mobile GPUs."
**Where:** the shared election, three call sites, same shape: grass `emitPx`
(`NaniteGrass.ts:314-323`), SW raster (`NaniteRaster.ts:1351` region), voxel `electHere`
(`NaniteVoxelRaster.ts:1349-1361`): relaxed `atomicLoad` guard → `If(cand > prevE)` →
`atomicMax(visPayloadV[px])` → winner `atomicStore(visBV[px])`. Under dense foliage many
primitives project onto the *same* `px` ⇒ contention on the same cache line.
**Mechanism / why Apple:** this is the **correct no-64-bit-atomic split** Aaltonen prescribes,
so the design is right. The pre-read `If(cand>prevE)` guard already suppresses most atomics
(a losing candidate never RMWs). The arm-mali 64B-spacing / workgroup-L2-amortise trick does
**not** cleanly apply: the target address is a *screen pixel*, not a reduction slot — you
cannot pre-combine writes to distinct pixels, and neighbouring pixels are inherently one cache
line apart. So the violation is real but largely un-actionable without changing the vis-buffer
contract (banned).
**Impact: MED, but mostly un-actionable.** Note it is already mitigated (guard) and the one
truly bad global-atomic — the per-win `atomicAdd` counter — is default-OFF for exactly this
reason (`NaniteVoxelRaster.ts:230-233,1354-1359`; the "close-up 120→30 fps cliff").

### F6 — Resolve shading runs in FP32 where FP16 would halve ALU  ★ SPECULATIVE / mostly BLOCKED
**Practice:** wwdc-10859:22-23 & Aaltonen slide 5 — "double-rate fp16 … ALU over lookups
(fp16)." arm-mali:94,113-114 — mediump/FP16 samplers up to 2× faster; vectorise f16vec2/4.
**Where:** `NaniteResolve.ts` fragment (`:417`+) does material decode + sun + CSM + IBL + GI
entirely in default-precision f32 (grep found zero f16/mediump). Same for the grass march math.
**Mechanism / why Apple:** M1 has double-rate f16; the lighting mix is ALU-heavy and would be a
natural f16 candidate.
**Why it's low / blocked:** (a) the resolve was **measured NOT to be the grass bottleneck**
(`NaniteGrass.ts:134-141`) — this cannot recover the +14.5 ms. (b) **WGSL f16 is blocked in our
stack**: it needs the `shader-f16` device feature + `enable f16;`, and three r184 TSL exposes no
half-precision node type — you cannot express `f16vec2` through TSL today. So this is a
"someday, if TSL gains f16" note, not a lever.
**Impact: LOW (and blocked).**

---

## 2. FIXABLE WITHOUT BREAKING THE PIPELINE?

| # | Concrete WGSL/WebGPU change | Breaks a contract? | Confidence |
|---|---|---|---|
| **F1** | No single fix — it is the sum of F2+F3+F4 plus inherent trip-count divergence. The divergence itself is only reducible by fetching less per step (F2/F4) and marching fewer/shorter rays (Q=2 already halves rays at dpr2; a coarser far band would help but was deleted by user + blocked by lushness law). | Vis-buffer/opaque/resolve all intact. Shorter far band = quality call (banned by lushness law). | — |
| **F2** | Move the **ctx** (ground, gradient, sward-top, gust-amp) from the `StorageBufferAttribute` to a **filterable texture** sampled at the ray's world XZ — exactly as the smooth FIELDS already are (`guideFieldT1/T2`, rgba16f, `:411-422,957-958`), which is *proof the round-trip works and won*. The stated blocker (`:156-158`) is UINT-storage-texture typing + a MASK NaN worry — neither applies to a FLOAT ctx (ground needs f32 → an `r32float`/`rgba32float` ctx texture; grad/top/amp can be an `rgba16f`). Keep only the 64-bit **mask** as a buffer (or an `rg32uint` texture read with `textureLoad`, no filtering → no NaN canonicalisation). Routes the hot per-step load through Apple's texture cache/unit (spatially coherent) instead of the slow dynamic-SSBO path. | No — same data, same march, same election. Mask stays integer so the density law is bit-identical. | Principle CONFIDENT; Apple win size SPECULATIVE — needs an A/B. |
| **F3** | (a) **Hoist the L2 golden-layer** (`:1335-1499`) out of `kRay` into its own tiny pass that only runs on the downward-ray hole pixels L1 left empty, so its ~20 live vars stop inflating every lane's register footprint. (b) Try workgroup **`[256]→[64]`** (`:1539`) — arm-mali:103 baseline; fewer lanes/group can leave more groups resident to hide F1's stalls. NB: this is the workgroup *size*, distinct from the 8×8 *tiling* that was already measured neutral-to-worse (`:786-787`) — do not confuse them. (c) Continue the field-bake medicine: any remaining per-step ALU that is constant-per-texel → bake to the guide. | No. Pure occupancy plumbing; output identical. | CONFIDENT it's the right axis (code admits occupancy-bound); each specific change SPECULATIVE → A/B each. |
| **F4** | Try the baked tile as a **2D texture array** (angle = layer, `LinearFilter` within the layer, nearest across layers) instead of a true `Data3DTexture`, IF the angle axis tolerates nearest-layer stepping (BAKE_ANG=8). Avoids the 3D-format tax on HW that charges it. | No — same fetch semantics, ±angle-interp quality. Verify no banding at 8 slices. | SPECULATIVE (Apple 3D-vs-array cost undocumented). Low priority. |
| **F5** | Leave the election as-is — it is already the correct no-64-bit-atomic split with a pre-read guard, and 64B-spacing can't apply to per-pixel targets. Keep `?voxwrites` and the per-win counter OFF (already default). | Changing it risks the vis-buffer contract (banned). | CONFIDENT: no safe change. |
| **F6** | Blocked until three/TSL exposes f16. Do not pursue now. | — | Blocked. |

---

## 3. TRICKS TO STEAL + WHERE

1. **"ALU over lookups / prefer texture+uniform over dynamic SSBO" (Aaltonen slide 5) → the F2
   ctx→texture move, and a fetch-count budget per march step.** The march's remaining cost is
   fetch-bound and occupancy-bound; the single most on-message steal is to route every per-step
   read through the **texture cache** (spatially coherent, Apple-optimised) and to **minimise
   the fetch COUNT** on the dependent path. This is the same move that already won (fields→
   texture); F2 finishes it for the ctx. *Where:* `NaniteGrass.ts:892,921,1086`.

2. **"Process things at the right frequency / temporal coherency" (Aaltonen slide 13) →
   INCREMENTAL guide bake.** We already exploit this (per-frame O(area) guide bake, 2-pass
   occlusion cull) — validated as correct. The steal: the guide window is snapped to a texel
   grid each frame (`NaniteGrass.ts:1546-1551`) yet the *whole* 384² field is rebaked every
   frame (`kGuideBake`, `:615`). Make it **toroidal** like the shadow clipmap (memory:
   "toroidal clipmap, cz-snap, strip re-raster") — rebake only the newly-exposed ring on camera
   motion + the always-changing wind amp. Cuts the bake pass from full-area to a thin strip.
   *Where:* `kGuideBake` at `NaniteGrass.ts:424-618`; precedent `NaniteShadowClip.ts`.

3. **"No 64-bit atomics → split depth-key `atomicMax` + payload `atomicStore`" (Aaltonen
   slide 8) → we ALREADY do this correctly.** `emitPx`/`electHere` are textbook. No action —
   the steal is *confirmation* that the election is mobile-correct; do not "fix" it.

4. **"Full-screen compute passes bypass framebuffer compression on mobile" (Aaltonen slide 6)
   → we ALREADY avoid this** by keeping the resolve a fullscreen **fragment** pass
   (`NaniteResolve.ts:2-8,417`), which retains DCC. The steal is a **guard-rail**: a tempting
   "combine passes / move resolve to compute" refactor would REGRESS on Apple. Keep the resolve
   in the fragment stage.

5. **"Profile for memory stalls; raise occupancy only if memory-stall-bound" (occupancy
   article:30,42) → the march IS memory-stall-bound**, so lowering registers to raise occupancy
   (F3) is precisely the sanctioned lever here (unlike the general case where higher occupancy
   can hurt via cache contention). This is the theoretical license for F3. *Where:* `kRay`.

6. **Apple has no HW integer divide (validated in our own code) → keep float reciprocals on
   every per-fragment index.** `NaniteVoxelRaster.ts:1332` — "Apple has no HW int-divide so the
   default path pays a microcoded sequence on every fragment"; the `?voxrecip` fix is **already
   default-ON** (`:257`). The steal is a **standing rule**: any new per-pixel/per-step `uint`
   `.div()/.mod()` on a hot path (Persson: "integer division is extremely expensive") must use a
   precomputed float reciprocal. The grass march already avoids int-div in its loop (float math
   + `.floor()`), so it's clean today — keep it that way.

---

## NON-FINDINGS (dropped after reading the code — premise-audit catches)

- **SampleGrad (`.grad()`) in the resolve** — flagged by Aaltonen (⅛-rate) and arm-mali
  (textureGrad slow). But `NaniteResolve.ts:727-734` `.grad()` is behind the **opt-in
  `?nanbark=grad` debug flag**; the DEFAULT bark path is analytic `.level(lod)` = textureLod
  (`:737`), which arm-mali:86 rates full-speed. Not a shipped violation.
- **Resolve as a bandwidth-wasting full-screen compute pass** — it's a **fragment** pass
  (`:2-8,417`) → keeps framebuffer compression. Compliant, not a violation.
- **Workgroup shared-memory usage** (voxel `kVoxScatter` uses ~15 `workgroupArray`s,
  `NaniteVoxelRaster.ts:609-630`; SW raster ~14, `NaniteRaster.ts:836-854`). arm-mali:107-110
  warns against shared memory — but that's **Mali-specific** ("Arm GPUs have no dedicated
  on-chip shared memory; it's cached system RAM"). **Apple has real fast on-chip threadgroup
  memory** (apple-tbdr / wwdc-10859:22 "improved threadgroup-memory perf"). ~7.5 KB of a 32 KB
  budget is fine on M1. Do NOT strip the cooperative workgroup design for Apple.
- **128-lane voxel workgroup register pressure** (`WG_RASTER=128`, `NaniteVoxelRaster.ts:563`)
  — above Mali's 64 baseline, but standard for Apple (4 SIMD-groups); the per-lane state is
  moderate and the raster was measured tapped-out. Low relevance vs the grass march. (A `[64]`
  A/B is cheap if ever revisited, but it is not where the +14.5 ms lives.)
- **Integer divide in the voxel per-pixel loop** — real on Apple, but **already fixed**
  (`?voxrecip` default-ON, `:257`). Listed as a steal (#6), not an open violation.
