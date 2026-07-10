# WebGPU feature availability (Jul 2026) — what actually shipped, and what we can use

**Purpose.** We had been assuming an OLD WebGPU feature set and stamping levers "blocked-by-WebGPU"
(notably `shader-f16`, subgroups, immediates). Chrome **STABLE is 149/150** (Jul 2026). This doc
re-checks what actually shipped across Chrome ~119→150, whether each thing is reachable from our
three r184 TSL stack, and where it plugs into our renderer. It **amends the "BLOCKED BY WEBGPU"
list in `MASTER-AUDIT.md:209-215`** and the DEEP-FOREST breakdown.

**Hard rule (user).** Use a newer feature ONLY where useful AND behind runtime feature-detection
with graceful fallback. Safari/Firefox WebGPU lag Chrome; our device must still work if a feature
is absent. Every adopt path below is `if (has(feature)) fastPath else currentPath`.

**One structural fact that changes everything (verified in source).**
three r184 `WebGPUBackend.js:204-219` loops **every** `GPUFeatureName` and pushes each
`adapter.features.has(name)` into `requiredFeatures` before `requestDevice`. So on our M1 Max, if
the adapter reports `shader-f16` / `subgroups` / `timestamp-query`, **the device already holds them
— with ZERO change to `Engine.ts`** (which passes no `requiredFeatures`). The blocker was never
"the device lacks the feature" (as `MASTER-AUDIT.md:156-157` states); it is purely that **TSL has
no f16 node type**, i.e. an authoring gap, not an availability gap. Subgroups, by contrast, ARE
authorable from TSL today (nodes exist).

---

## 1. Feature table

Status legend: **STABLE** = shipped default-on in Chrome stable · **not-shipped** = proposal/flag/
experimental only · adapter-conditional means even in Chrome it depends on the GPU.

| Feature | Chrome (Jul 2026) | Safari / Firefox | Useful | Where in our renderer | Detection | TSL-reachable? | Fallback |
|---|---|---|---|---|---|---|---|
| **shader-f16** (`enable f16;`) | **STABLE since 120**, adapter-conditional | Safari 26: optional, likely on M-series (detect); FF 141+/145+: detect, not guaranteed | **HIGH** | Register/occupancy relief on TBDR (M1 Max = NOT double-rate FLOPs, but half VGPR/value → higher occupancy on our dependent-fetch loops): grass raymarch state `NaniteGrass.ts` kGrassRay; voxel-scatter temporaries; resolve BRDF scratch `NaniteResolve.ts`. NOT the election key (`NaniteRaster.ts` — precision-critical). | `adapter.features.has('shader-f16')` (device already holds it, auto-requested) | **No native node** — TSL nodes are all f32/i32/u32. Needs raw-WGSL `FunctionNode`/`wgslFn` for the hot block; builder emits directive via `enableShaderF16()` (`WGSLNodeBuilder.js:1517`). f32↔f16 pack at the buffer boundary. | All-f32 TSL kernel (current, unchanged) |
| **subgroups** (`subgroupBallot/Add/Max/Min/Elect/Broadcast/ExclusiveAdd/…`) | **STABLE since 134** (OT 128-131). Apple subgroup size 32 | Safari 26: **absent** (treat so). FF 145: **absent** | **HIGH** | **Best fit = queue COMPACTION in cull**: replace per-thread `atomicAdd` append into `qRaster`/vox counters (`NaniteCull.ts` `kVoxFanout`, `kVoxCount`, `kVoxScatter`; append `:480`+) with ballot-aggregated append → ~32× fewer global-counter atomics on Apple. Weak fit = per-pixel election (`NaniteRaster.ts:374` — lanes hit DIFFERENT pixels; `relect` already proved election contention ~dead). | `adapter.features.has('subgroups')`; sizes via `adapter.info.subgroupMin/MaxSize`; on device `renderer.hasFeature('subgroups')` (`WebGPUBackend.js:2330`) | **YES, native** — import from `three/tsl`: `subgroupBallot, subgroupBroadcastFirst, subgroupExclusiveAdd, subgroupElect, subgroupAdd/Max/Min, …` (`SubgroupFunctionNode.js:210+`); directive auto-emitted (`WGSLNodeBuilder.js:1643-1646`). No raw WGSL needed. | Today's per-thread `atomicAdd` path (bit-identical results) |
| **immediates** (`var<immediate>` / `requires immediate_address_space;` + `pass.setImmediates()`) | **STABLE 149-150** (headline item) | Safari: not shipped · FF: not shipped | LOW | Small per-dispatch scalars we ferry via UBO/uniform nodes: cull tau/minPx (`NaniteCull.ts:456/462`), per-cascade shadow consts (`NaniteShadow.ts:204-205`). But three already uses UBOs (not SSBOs), so the "SSBO-slow-on-mobile" pain doesn't apply → marginal. | `navigator.gpu.wgslLanguageFeatures.has('immediate_address_space')`; `device.limits.maxImmediateSize` (≥64B) | **No** — three owns the pass encoder; no `setImmediates` node. Would need `WebGPUBackend` patch (raw WGSL alone insufficient). | `uniform()` nodes / UBO (current; cost-equivalent here) |
| **subgroups-f16** | STABLE where `shader-f16`+`subgroups` both present | absent (both) | LOW | f16 subgroup reduce; our reduce payloads are u32/f32 → no natural fit. Listed so not re-flagged "blocked". | `has('subgroups') && has('shader-f16')` | raw-WGSL only; `enableSubgroupsF16()` (`:1499`) NOT auto-emitted | f32 subgroup ops |
| **subgroup_id builtin / subgroup_uniformity** | subgroup_id STABLE 144; subgroup_uniformity STABLE 145 | absent | LOW | Only for 2-level workgroup+subgroup reductions; single-level compaction doesn't need it. `subgroup_uniformity` = uniformity-error safety valve if compaction trips the analyzer. | follows `subgroups`; `wgslLanguageFeatures.has('subgroup_id')` | `subgroup_id` via `getSubgroupIndex()` (`:1379`); uniformity directive NOT auto-emitted | single-level reduce / uniform control flow |
| **quad ops** (`quadBroadcast/quadSwapX/Y/Diagonal`) | STABLE with subgroups (134) | absent | LOW | 2×2 cross-lane deltas in the fragment resolve (`NaniteResolve`) for normal/depth/AO w/o extra fetch. Not in compute scatter. | `has('subgroups')` | **YES** — `quadBroadcast/quadSwapX/…` from TSL (`SubgroupFunctionNode.js:186-196`) | `dpdx/dpdy` (always available) |
| **timestamp-query** | **STABLE since 121** | varies | HIGH | Load-bearing for the whole perf arc: `Engine.ts:86` `trackTimestamp:true`, `:114-115` GpuProfiler wiring; per-pass GPU ms. | `has('timestamp-query')` (already wired) | **YES** via `renderer.info.compute/render.timestamp` | GpuProfiler null → whole-frame wall (our ground truth) |
| **timestamp-query-inside-passes** | STABLE (121-ish window) | varies | MED | Time SUB-portions of the single world1 pass (coverage-walk vs election vs interp) without the `relect` ablation rebuild. Needs raw encoder access. | `has('timestamp-query-inside-passes')` | **No** (three only does boundary timestampWrites) | `relect`/ablation-rebuild attribution (current) |
| **float32-filterable** | **STABLE since 119** | varies | MED | HW bilinear on r32/rgba32float instead of manual 4-tap: terrain heightmap (grass/voxel conform), HZB float targets, GI probe floats. | `has('float32-filterable')` (three auto-requests) | **YES** — float texture + `linear` sampler (feature must be enabled; verify via `diag.features`) | nearest + manual bilinear, or use rgba16float (always filterable) |
| **TRANSIENT_ATTACHMENT** (memoryless tile attachments) | **STABLE 149-150** (tighter validation) | Chrome-only | MED | TBDR bandwidth: write-then-discard depth/MSAA scratch stays in on-chip tile memory, never allocates VRAM. **NOT the compute vis-buffer** (storage buffer) — only render-attachment parts of resolve/post/shadow-depth. | `GPUTextureUsage.TRANSIENT_ATTACHMENT !== undefined` | **No** — three RenderTarget doesn't surface the usage bit | normal VRAM attachment |
| **GPUBuffer as bind resource** | STABLE 138 | rolling | no | ergonomic only; internal to three | — | internal | n/a |
| **GPUBuffer.mapSync()** (Worker-only) | 145 **experimental** | absent | no | readback convenience, not our GPU-resident hot path | `typeof GPUBuffer.prototype.mapSync==='function'` + in Worker | renderer-level | mapAsync (current) |
| **dynamic buffer offsets** | STABLE since 113 (core) | universal | LOW | portable alt to immediates (one UBO + 256B-aligned rolling offset). No Chrome-150 dep. | core API, always present | **No** (three manages bind groups) | uniform() coalescing (already ~equivalent) |
| **dual-source-blending** (`@blend_src`) | STABLE 130 | varies | no | subpixel/colored-coverage alpha — we are OPAQUE-ONLY, no site | `has('dual-source-blending')` | n/a | n/a |
| **64-bit buffer atomics** (`atomic<u64>` / `atomic_vec2u_min_max`) | **NOT SHIPPED** (proposal gpuweb#5071, Draft; `next-for-webgpu` "under consideration", cites Nanite atomicMax) | absent | HIGH *(if it lands)* | Our exact vis-buffer election: today we PACK depth+id and split across two atomics (`NaniteRaster.ts:197`, `:254-267`, clear `:548-549`). Real `atomicMax(u64=depth<<32\|id)` collapses it to ONE op, removes the packed-precision ceiling + branch-through-trunk desync race. | no feature string yet → `has(<future>)` false | moot (needs raw WGSL + requiredFeatures when it lands) | **current packed 2-atomic election IS the fallback / default** |
| **texture atomics (R64Uint)** | **NOT SHIPPED** in browser (exists in wgpu-native `TEXTURE_INT64_ATOMIC`) | absent | MED | alt Nanite path: atomicMax into R64Uint vis-TEXTURE. Lower priority — our vis-buffer is storage-buffer already. | no browser feature string | not reachable | storage-buffer packed election |
| **subgroup-matrix** | experimental, Dawn toggle only (unsafe-webgpu) | absent | no | tensor MMA (AI). We have no dense matmul loop. | Dawn toggle only | n/a | n/a |
| **multi-draw-indirect / bindless** | **NOT SHIPPED** (`next-for-webgpu`, prioritized-future) | absent | MED *(future)* | bindless → resolve übershader indexes the whole bark/foliage texture set w/o the 24-sampler juggle (`Diagnostics.ts:40` raised `maxSampledTexturesPerShaderStage` to 24); MDI → HW-election path. | no feature string yet | no node | current fixed bind groups + compute scatter |

**Sources** (cited by the research, verify on-machine via `webgpureport.org` for the real adapter dump):
`developer.chrome.com/blog/new-in-webgpu-119/120/138/145/149-150`, `webgpu.com/news/chrome-149-150-webgpu-immediates`,
`developer.chrome.com/blog/next-for-webgpu`, `chromestatus.com` (5180552617656320 f16, 5167711051841536 dual-src,
5199437611794432 immediates), `gpuweb/gpuweb#5071` (u64 atomics), spec + `wgslLanguageFeatures`, MDN WebGPU compat.

---

## 2. CORRECTIONS — features we wrongly marked "blocked-by-WebGPU"

These directly amend `MASTER-AUDIT.md:209-215` ("BLOCKED BY WEBGPU") and the DEEP-FOREST audit.

1. **`shader-f16` is NOT blocked — it has been STABLE since Chrome 120 and is already enabled on our
   device.** `MASTER-AUDIT.md:215` frames it as "device lacks the feature (`Engine.ts:86`)". That is
   wrong: three `WebGPUBackend.js:204-219` auto-requests every adapter feature, so if our M1 Max
   adapter reports `shader-f16` (Apple Metal supports half natively → it does; confirm via
   `window.hooks.diag.features`), the device **already holds it with no `Engine.ts` change**. The
   real and only blocker is **authoring**: TSL has no f16 node type, so the hot inner loop must be
   written as raw WGSL (`FunctionNode`/`wgslFn` starting `enable f16;`) with f32 buffers at the
   boundary. Reclassify from "blocked-by-WebGPU" to "available; needs raw-WGSL authoring + shotdiff".
   **Honesty caveat (`MASTER-AUDIT.md:36`):** on M1 Max fp16 is **not double-rate** — the win is
   register/occupancy (half the VGPR pressure → higher occupancy on our dependent-fetch loops), NOT
   2× FLOPs. Must be measured (register-bound?), not assumed.

2. **Subgroups are NOT blocked — STABLE since Chrome 134, enabled on our device, AND reachable from
   TSL today.** No raw WGSL, no `Engine.ts` change: `subgroupBallot/BroadcastFirst/ExclusiveAdd/
   Elect/Add/Max/Min` and `quadBroadcast/quadSwap*` are exported from `three/tsl`
   (`SubgroupFunctionNode.js:210+`), and the `enable subgroups;` directive auto-emits when a
   subgroup builtin is used (`WGSLNodeBuilder.js:1643-1646`). This was previously treated as
   unreachable. It is the single most adoptable new lever.

3. **Immediates are NOT blocked — STABLE in Chrome 149-150.** We had marked them unavailable; they
   shipped. BUT the realizable win on our stack is marginal (three uses UBOs not SSBOs, and the
   `setImmediates` pass-encoder call is owned by three, unreachable from TSL). Correct the
   availability claim, keep the priority LOW.

4. **`float32-filterable` (STABLE 119) and `timestamp-query-inside-passes` are available**, not
   blocked — the former likely already auto-enabled (check `diag.features`), the latter needs raw
   encoder access but is a real option instead of the `relect` ablation-rebuild.

5. **Still genuinely NOT available (keep on the blocked list, correctly):** 64-bit buffer/texture
   atomics (proposal only — our packed 2-atomic election stays the default), bindless,
   multi-draw-indirect, subgroup-matrix. These remain future-arc items; do not build against them.

---

## 3. Top adopt candidates, ranked by impact on the Apple/TBDR bottleneck

### #1 — Subgroup-aggregated queue compaction in cull (HIGH, TSL-native, lowest risk)
Our bottleneck is dependent-fetch loops + atomic traffic under dense overlap. The cull compaction
does one **global** `atomicAdd` per surviving lane to append into `qRaster` and the voxel
counters/cursors (`NaniteCull.ts`: `kVoxFanout`, `kVoxCount`, `kVoxScatter`; queue at `:480+`). On
Apple (subgroup size 32) a ballot-aggregated append cuts that global-counter traffic **~32×**, and
lanes are independent so results are bit-identical.

- **Plugs in:** `src/nanite/NaniteCull.ts` — the per-thread `atomicAdd` append sites.
- **Pattern (TSL, from `three/tsl`):**
  ```
  const b    = subgroupBallot(keep);            // vec4<u32>
  const n    = countOneBits(b.x).add(countOneBits(b.y))
                 .add(countOneBits(b.z)).add(countOneBits(b.w));
  const base = subgroupBroadcastFirst(
                 subgroupElect().select(atomicAdd(counter, n), uint(0)));
  const slot = base.add(subgroupExclusiveAdd(uint(keep)));  // per-lane write index
  ```
- **Detection guard:** `renderer.hasFeature('subgroups')` at kernel-build time.
- **Fallback:** the shipped per-thread `atomicAdd` append (identical output). Gate behind a `?sg`
  URL flag mirroring `?relect`/`?swcoop` so A/B is bit-identity checkable. Safari/FF/older-Chrome
  hit the fallback with no visual change.
- **Why #1:** native TSL (no raw-WGSL surgery), zero correctness divergence, attacks real atomic
  traffic. NOTE: do **not** expect a payoff on the per-pixel *election* (`NaniteRaster.ts:374`) — our
  `relect` ablation already measured election contention effectively dead (RASTER arc CLOSED
  `62d607e`); the loop is intrinsic per-covered-pixel work. Compaction is where subgroups pay.

### #2 — f16 scratch in the grass raymarch + resolve BRDF (HIGH, needs raw-WGSL, register-gated)
Halves VGPR/value on the two heaviest per-pixel loops → higher occupancy on M1 Max TBDR (occupancy,
not FLOPs — fp16 is not double-rate here).

- **Plugs in (priority order):** (1) `src/nanite/NaniteGrass.ts` kGrassRay march state — march-space
  position deltas, fiber half-width, accumulated color/coverage as f16 (the +14.5ms register-heavy
  loop, task **G-C #55**); (2) `NaniteResolve` BRDF intermediates (albedo/normal/sun/IBL/GI scratch);
  (3) voxel-scatter brick color/normal temporaries. **NOT** the election key.
- **Detection guard:** `adapter.features.has('shader-f16')` (or `renderer.hasFeature('shader-f16')`) —
  already true on our device.
- **Implementation shape:** author the hot inner block as a raw-WGSL `FunctionNode`/`wgslFn` starting
  `enable f16;` (builder path `enableShaderF16()` `WGSLNodeBuilder.js:1517`); keep the buffer
  interface f32, pack/unpack at the boundary. Ship **both** kernel variants:
  `if (has('shader-f16')) useF16Kernel else useF32Kernel`.
- **Fallback:** the current all-f32 TSL kernel, unchanged.
- **First step (per `MASTER-AUDIT.md:162`):** capture to confirm the loop is register/occupancy-bound
  before spending the raw-WGSL rewrite; then verify Tint emits packed f16 (not f32-promoted); gate
  behind a param + shotdiff (f16 range ~65504, 10-bit mantissa — demote only tile-local ∈[0,1]
  scratch, never world-anchored ±155m positions per `MASTER-AUDIT.md:146-148`).
- **Why #2 not #1:** real surgery (one hot loop body per kernel in WGSL) + precision validation, vs
  subgroup compaction being a native-node drop-in.

### #3 — TRANSIENT_ATTACHMENT (memoryless) on render-attachment passes (MED, TBDR bandwidth)
Directly targets the M1 Max/mobile memory bottleneck: write-then-discard depth/MSAA scratch stays in
on-chip tile memory and never allocates VRAM.

- **Plugs in:** the render-attachment parts only — shadow-depth targets cleared each frame, any
  attachment-based resolve/post scratch. **Not** the compute vis-buffer (storage buffer — no benefit).
- **Detection guard:** `GPUTextureUsage.TRANSIENT_ATTACHMENT !== undefined`.
- **Fallback:** normal VRAM-backed attachment.
- **Blocker:** three r184 RenderTarget doesn't surface the usage bit → needs a `WebGPUBackend`/
  RenderTarget patch, and applies only where the target is truly write-then-discard within a frame.
  Narrow applicability (most of our frame is storage-buffer), hence #3.

---

## 4. Canonical feature-detection + graceful-fallback pattern (for `src/core/Engine.ts`)

**Detection has three layers** (verify ground truth on-machine at `webgpureport.org`):
1. **GPU features** (device caps): `adapter.features.has('shader-f16'|'subgroups'|'timestamp-query'|
   'float32-filterable'|…)` on the adapter *before* `requestDevice`; post-device
   `renderer.hasFeature(name)` (wraps `device.features.has`, `WebGPUBackend.js:2330`).
2. **WGSL language features** (syntax/extensions): `navigator.gpu.wgslLanguageFeatures.has(
   'immediate_address_space'|'subgroup_id'|…)` + the matching `enable`/`requires` directive.
3. **Ground truth dump:** `webgpureport.org` on the target machine.

**We already do layer 1**: `Diagnostics.ts:65` `requestAdapter({powerPreference:'high-performance'})`,
`:95` `features:[...adapter.features]`; surfaced on `window.hooks.diag.features` and consumed at
`Engine.ts:114`.

**The key nuance for enabling features:** we do NOT need to change device creation to *enable*
shader-f16/subgroups — three `WebGPUBackend.js:204-219` already auto-requests every adapter feature,
and `Engine.ts:84` passes no `requiredFeatures`. Only branch our *shaders* on availability. (If we
ever needed a feature three does NOT auto-request, we would pre-create our own `GPUDevice` with the
extra `requiredFeatures` and hand it to the renderer — not needed today.)

**The pattern — detect once at boot, branch kernels at build time:**
```ts
// after renderer.init() (Engine.ts ~:84-115). Single source of truth for capability branches.
const feats = hooks.diag?.features ?? [];            // Diagnostics.ts:95, already populated
const caps = {
  f16:            feats.includes('shader-f16'),        // three already enabled it if present
  subgroups:      feats.includes('subgroups'),
  f32filter:      feats.includes('float32-filterable'),
  tsInside:       feats.includes('timestamp-query-inside-passes'),
  immediates:     navigator.gpu.wgslLanguageFeatures?.has('immediate_address_space') ?? false,
  transientAttach: typeof GPUTextureUsage !== 'undefined'
                     && 'TRANSIENT_ATTACHMENT' in GPUTextureUsage,
};
engine.caps = caps;   // read by NaniteCull / NaniteGrass / NaniteResolve at kernel build

// per-kernel branch (build time), each mirrored by a ?flag for A/B identity checks:
const useSubgroupCompaction = caps.subgroups && param('sg', 1);   // NaniteCull.ts
const useF16Grass           = caps.f16       && param('gf16', 0); // NaniteGrass.ts (opt-in until shotdiff)
// each false → the current shipped f32 / per-thread-atomicAdd path. Bit-identical on absence.
```
**Rules:** (a) every fast path has an `else` = the current shipped kernel; (b) gate each behind a URL
flag (mirror `?relect`/`?swcoop`) so on-device A/B stays bit-identity checkable; (c) never assume —
Safari 26 lacks subgroups/immediates, some Android/Qualcomm adapters lack shader-f16 even on Chrome,
so the runtime `has()` check is mandatory on all browsers; (d) validate f16 precision behind
shotdiff before default-on (demote only tile-local [0,1] scratch, never world-anchored positions).
