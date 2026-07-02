# webgpu-platform deep review (2026-07-02)

Area: the WebGPU/Dawn/Metal platform layer under the renderer — `src/nanite/Tsl.ts`
(codegen + dispatch helpers), device features/limits, workgroup sizes and bind layouts
across every kernel, the atomics idioms, the queue.submit/encoder economics, and the
Metal (AGX, M1 Max) execution model that turns dispatch shapes into milliseconds.
Baselines: post-voxbocc isolated gpuWall eye/oblique/aerial 18.9 / 37.2 / 16.5 (second
run 21.0 / 36.3 / 14.9); live 600-tick p50 16.7 / p95 25.1, live cpu.submit med 1.10 ms
(fresh-voxbocc*.json). All code cites verified at HEAD of `nanite-raster` (6a93dfb +
uncommitted); three r184 cites verified in `node_modules/three` (build + src).

---

## TL;DR

- **The stack requests and receives everything the adapter has.** three r184's backend
  passes ALL adapter features into `requestDevice` (three.webgpu.js:80055-80074) — so
  `subgroups` and `shader-f16` are LIVE on the device (facts of record,
  NANITE-SPEC.md:88-95). TSL r184 has the full subgroup op surface (SubgroupFunctionNode:
  ballot/prefix/broadcast/shuffle/quad) and even emits `enable subgroups;` +
  `@builtin(subgroup_size)` into EVERY compute shader already (three.webgpu.js:76396-76400).
  Nothing in src/ uses any of it yet. f16 is enabled but TSL exposes no f16 storage/ALU
  type — and the frame has no f32-bandwidth hotspot big enough for it to matter (refuted, §Refuted).
- **Mission 2 verdict: doc 10 is CONFIRMED and sharpened.** The voxf2b K=16 penalty
  (+5/+6/+16 ms) is NOT dispatch/barrier overhead (voxwaves bound: 36 extra dispatches +
  3 submits ≈ +0.1 ms at eye ⇒ ≤ ~10 µs each — re-verified from fresh-voxwaves4.json) —
  it is **residency-limited serialization**. New platform quantification: the vox scatter
  workgroup needs ≈6.5 KB of threadgroup memory (measured off the declared arrays,
  NaniteVoxelRaster.ts:526-549,613), and AGX has 32 KiB/core ⇒ **≤4 workgroups/core
  resident ⇒ the whole 32-core M1 Max holds ≤128 WGs of this kernel at once**. An aerial
  F2B stage has 659/16 ≈ 41 WGs = 1.3 WGs/core ⇒ ~5 of ≤16 resident simdgroups/core ⇒
  latency hiding collapses, and each of the 16 barrier-drained stages ends on its longest
  FarTile-head cluster. Law (endorse doc 10): never partition a dispatch below
  (machine residency × a few) threads per stage. Corollary: the same 6.5 KB cap throttles
  the DEFAULT wide dispatch too — a workgroup-memory diet is a real lever (L1).
- **The 32-bit election is sound and near-optimal.** `depthKey24` is monotone (f32 mul of
  non-negative operands is order-preserving), quantization is ~w²/5.0e6 m (0.4 mm at 45 m,
  0.2 m at 1 km) — a mis-election is bounded by one quantum + the id8 tie. The one real
  artifact is the **visBV two-buffer store race** (payload of a strict-but-not-final winner
  can land after the final winner's), already the accepted Class-I exception with the
  `?audit=1` orphan counter watching it (NaniteRaster.ts:350-353,963-975). A two-word
  scheme would ADD traffic and a retry loop; WebGPU has no 64-bit atomics (gpuweb #5071).
  Do not reopen.
- **cpu.submit has a measured encode floor of ~1.1-1.5 ms** (live med 1.10, voxbocc-era
  isolated 1.4-1.5) for ~12 submits / ~67 dispatch-encodes ≈ 15-20 µs per dispatch. The
  churn is structural in three: every `renderer.compute()` = 1 encoder + 1 pass +
  1 queue.submit (WebGPUBackend `finishCompute`), and per dispatch it re-runs
  `nodes/bindings.updateForCompute` + an **uncached** `setBindGroup` (Renderer.js:2767-2777,
  WebGPUBackend.js:1419-1429 — the render path caches bind groups, the compute path does
  not). The 5.2-5.5 ms cpuSubmit of the bead-v2-base-era runs is NOT explained by this
  model — run-state or code drift, one paired re-measure needed (doc 10 Q3, probe P3).
- **Metric-level premise finding: every canonical number carries the timestamp apparatus.**
  `trackTimestamp: true` (Engine.ts:74) makes EVERY render/compute pass carry
  `timestampWrites` (WebGPUBackend initTimestampQuery), and Engine resolves BOTH query
  pools + a mapAsync readback EVERY frame (Engine.ts:219-239). gpuWall itself doesn't use
  them (it's onSubmittedWorkDone) — so the apparatus is severable. Size it once (P4).
- Two S-effort Metal wins found: **storeOp discard for the dead hwRT rgba8 store**
  (three hardcodes Store — NaniteRaster.ts:300, WebGPUBackend.js:606; a ThreePatches-class
  patch fixes it; the TBDR tile flush of 13.4 MB/frame is pure waste) and the already-shipped
  int-divide-avoidance pattern (`?voxrecip`, NaniteVoxelRaster.ts:248-258) which should be
  the template for any future per-pixel div/mod.

---

## Premise audit

**P1 — "The Metal 10-storage-buffer cliff" is a real adapter limit on this machine, not a
spec-default artifact.** `buildRequiredLimits` requests 16 and clamps to adapter max
(Diagnostics.ts:36-52); the probed fact of record is `maxStorageBuffersPerShaderStage = 10`
(NANITE-SPEC.md:81-84, probed 2026-06-12). The cliff has SHAPED the architecture — packed
mega-buffers (F9), the SCAR counters carved into hwQueue's tail to avoid an 11th binding
(NaniteRaster.ts:339-346), the K per-bucket args kernels split precisely so no kernel binds
K dispatch attrs (NaniteCull.ts:644-648,667-681) — but **nothing currently pays a per-frame
cost for it**. It is a design constraint, not a waste line. Re-probe on Chrome majors
(Dawn's Metal argument-buffer work can raise it; P5).

**P2 — three's adapter request is `featureLevel: 'compatibility'`** (three.webgpu.js:
80040-80043). On this desktop Chrome the adapter still carries `core-features-and-limits`,
three requests it with everything else, and `compatibilityMode` resolves false
(three.webgpu.js:80082) — so nothing is degraded HERE. But it means the limits three's
device gets are OUR requested limits (passed through from Diagnostics), not the adapter
defaults — the two-adapter split (probe adapter ≠ render adapter) is only safe because
buildRequiredLimits re-clamps. Filed as a fragility, not a bug.

**P3 — "barriers are cheap" (doc 10) re-verified against the raw JSONs before being used.**
fresh-voxwaves4 vs the F2B-K16 arithmetic: eye 41.1 (f2b ≈ 41.0), aerial 28.1 vs f2b-K16
≈ 32.8 (−4.7, the wave occlusion gain) — the +3 pyramid chains + 3 submits + 36 dispatch
boundaries cost ≤ ~0.1-0.3 ms where their occlusion gain is nil. Bound: ≤ ~10 µs per
barrier-separated dispatch, ≤ ~100 µs per submit, GPU-side. (Metal inserts implicit
barriers around individual dispatches; they are documented-cheap on TBDR.) The claim holds;
everything in §Work model builds on it.

**P4 — the fact pack's "+5/+6/+16" attribution needed the occupancy numbers checked, not
assumed.** Done in §Work model with the threadgroup-memory residency cap (the piece doc 10
did not have). The model reproduces the observed deltas within its error bars at all three
poses; the discriminating probe (K=4 linearity) is still worth running (P1 below).

---

## How it works today

### 1. Codegen + dispatch helpers (Tsl.ts)

`Tsl.ts` is the single casts-live-here shim over @types/three 0.184 TSL gaps — uvec2/uvec4
storage views, uint min/max, ranged loops, bitcasts, atomic views (`sU32Views` returns
rw/ro/atomic views of ONE attribute, Tsl.ts:379-389). Behavior-relevant helpers:

- `dispatch` / `dispatchIndirect` (Tsl.ts:228-231, 367-376): one kernel per
  `renderer.compute()` call ⇒ one encoder + one pass + one submit each.
- `dispatchBatch` (Tsl.ts:241-243): array form ⇒ ONE encoder/pass/submit; WebGPU
  auto-syncs (UAV barrier) between dispatches in a pass, so dependent chains (HZB mips)
  are safe. Array order IS execution order.
- `setIndirectDispatch` + `dispatchBatchMixed` (Tsl.ts:245-306): the submit-coalesce
  enabler. three's array path forwards `dispatchSize=null` to every node; the backend
  falls back to `computeNode.dispatchSize || computeNode.count`
  (node_modules/three/src/renderers/webgpu/WebGPUBackend.js:1431-1435) and takes the
  indirect path when that value `.isIndirectStorageBufferAttribute` (:1439-1446). Tagging
  the attr onto `.dispatchSize` gives per-node tight indirect grids inside one batched
  submit — verified against the r184 source; the 1B-thread baked-grid fallback cannot fire
  while the attr is attached.

**What a `renderer.compute()` call costs on the CPU** (verified,
Renderer.js:2767-2777 + WebGPUBackend build:81230-81367): per CALL — command encoder
create, `beginComputePass` (with `timestampWrites` when trackTimestamp, see P-item W6),
`end`, `queue.submit`. Per NODE in the call — `nodes.updateForCompute` (uniform graph
update), `bindings.updateForCompute` (version-check + writeBuffer for dirty uniforms),
pipeline lookup (cached), `setPipeline` (cached per-pass, three.webgpu.js:78773-78785),
and `setBindGroup` for every bind group **unconditionally** — the compute path has no
`currentBindingGroups` dedup like `_draw` has (build:81272-81280 vs :81397-81407).
Measured floor: ~67 dispatches + ~12 submits ≈ 1.1-1.5 ms ⇒ **~15-20 µs per dispatch-encode**.

### 2. Device features and limits (all verified)

| fact | value | cite |
|---|---|---|
| storage buffers/stage | **10** (probed; the F9 cliff) | NANITE-SPEC.md:81, Diagnostics.ts:38 clamp |
| storage textures/stage | 8 | NANITE-SPEC.md:84 |
| workgroup memory | 32 KiB (= AGX per-core total, see §4) | NANITE-SPEC.md:100 |
| max invocations/WG | 1024; maxComputeWorkgroupsPerDimension 65535 (DISPATCH_ROW split, NaniteCommon.ts:50) | NANITE-SPEC.md:98-100 |
| `subgroups` | PRESENT (Chrome ≥134); three enables the directive in every compute shader | NANITE-SPEC.md:88-90, three.webgpu.js:76396-76400 |
| `shader-f16` | PRESENT on device; TSL has `enableF16()` plumbing but NO f16 storage/var surface | NANITE-SPEC.md:91, three.webgpu.js:76268-76272 |
| 64-bit atomics | ABSENT from WebGPU entirely (proposal gpuweb#5071) | NANITE-SPEC.md:95-97 |
| buffer/binding max | 4 GiB−4 (requested just under, Diagnostics.ts:42-44) | NANITE-SPEC.md:85-87 |

three requests **every feature the adapter reports** (three.webgpu.js:80055-80074), so no
feature-request change is ever needed in src/ — only WGSL/TSL usage.

### 3. Workgroup-size census

| shape | kernels | notes |
|---|---|---|
| 128 lanes (4 simdgroups) | kRasterWorld1 (=MAX_CLUSTER_TRIS, GeometryRegistry.ts:106; NaniteRaster.ts:1015), kVoxScatter (=MAX_BRICKS_PER_CLUSTER, NaniteVoxelRaster.ts:480,1419) | 1 WG = 1 cluster; long serial Phase-B loop inside (critical-path shape) |
| 64 lanes | BFS traverse + fanout + F2B chain (NaniteCull.ts:532,618,638,709), both pyramid chains (NaniteHzb.ts:145, NaniteVoxelRaster.ts:470), instance cull (:805), registry builders | 2 simdgroups — sane default |
| 256 lanes | full-res clears (kVisClear NaniteRaster.ts:408, shadow clears, kClearBins) | bandwidth kernels |
| 1 lane | ~24 args kernels/frame: 2/BFS level ×16 + kRasterArgs(+2) + fanout args (+ under f2b: kVoxRangeArgs, kVoxPrefix, 16× kVoxBucketArgs) | each a full dispatch boundary; bounded cheap by P3's law |

No mis-sized workgroup found: the 128-lane cluster kernels are structural (arrays sized by
the cluster caps), the 64/256 choices are appropriate for AGX's 32-wide simds. The census
finding is not size but SHAPE: both hot kernels put a long serial loop (tris of a cluster /
bricks of a cluster) inside one workgroup — that is what makes stage tails expensive when
occupancy drops (§Work model).

### 4. The AGX execution model (what dispatch shapes cost)

M1 Max: 32 cores, 32-wide simdgroups, **32 KiB threadgroup memory per core**, 208 KiB
register file per core, occupancy starts dropping at ~128 GPRs/thread
([philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks),
[dougallj applegpu docs](https://dougallj.github.io/applegpu/docs.html)); device bandwidth
~400 GB/s. Metal inserts implicit memory barriers around individual compute dispatches
([metalbyexample WebGPU part 1](https://metalbyexample.com/webgpu-part-one/)) — matching
WebGPU's per-dispatch usage-scope model and the measured ≤10 µs bound.

**kVoxScatter's declared workgroup memory at defaults** (voxlod=1, voxcell=1, dither off;
NaniteVoxelRaster.ts:526-549, 613): 5×128 u32 (bbox+cand, 2560 B) + occMask (512 B) +
4×128 f32 voxcell AABB (2048 B) + 3×128 u32 cell words (1536 B) + wgVisible ≈ **6.5-6.7 KB**
⇒ **tg-mem residency ≤4 WGs/core = 512 threads = 16 simdgroups/core** (of the ~24+ a core
can schedule), machine-wide **≤128 resident WGs**. Register pressure compounds it: at
~100 GPRs/thread (plausible for the ray+DDA monolith), 512 threads × 100 × 4 B ≈ 205 KB —
right at the 208 KB file. This is the platform-level explanation candidate for doc 13's
"naive ALU model under-predicts the slope 3-5×" gap.

### 5. Atomics idioms (audit)

- **Election** (world1 NaniteRaster.ts:962-975; vox flat :1196-1215 and ray
  :1266,1394-1403): relaxed load → `cand > prevE` gate → `atomicMax(visPayloadV)` →
  strict-winner `atomicStore(visBV)`. Losers cost 1 load. This is the correct 32-bit
  idiom; UE5 has NO 32-bit vis-buffer recipe to copy (doc 01 §2.6).
- **Precision**: `depthKey24 = uint((1−cz)·16777215)` (NaniteRaster.ts:335-336). f32
  multiply is monotone ⇒ ordering preserved up to ties. Key-step in view depth
  Δw ≈ w²/(n·2²⁴) = w²/5.0e6 m (n=0.3): 0.4 mm @45 m, 4 mm @140 m, 0.2 m @1 km, ~5 m @5 km
  far tiles. Equal-key ties resolve by id8 (largest low-8-bits of the item index wins) —
  wrong-by-≤1-quantum, never torn.
- **HZB decode of the packed key is conservative in the safe direction**: the pyramid
  decodes depth = 1 − (bits>>16)/65535 (NaniteHzb.ts:119-121) — truncating key24 to its
  top 16 bits FLOORS (1−cz) ⇒ decoded depth ≥ true cz ⇒ the occluder reads FARTHER ⇒
  under-culls, never over-culls.
- **visBV race**: two strict winners can interleave so the FINAL visBV payload belongs to
  the non-final key (A max, B max, B store, A store). Intra-dispatch window, rate ∝
  same-pixel concurrent winners, audited by `?audit=1` orphans (NaniteRaster.ts:350-353),
  masked by TAA. Accepted Class-I exception — the vox flat path's brick-constant front-slab
  key (doc 12 W7) is a bigger depth looseness than this.
- **Single-word global counters**: the `?voxwrites` per-win atomicAdd on ONE word was
  measured as a cross-core cache-line ping-pong cliff and is build-time OFF
  (NaniteVoxelRaster.ts:234-241) — the correct Metal-aware default. The append queues
  (kVoxFanout NaniteCull.ts:526-532, hwQueue, frontier counters) are per-thread atomicAdds
  on hot words but live in the ~0.2 ms cull (refuted as a cost).

### 6. Frame dispatch economics (verified against the frame code)

Default forest frame (voxf2b OFF): BFS batch ≈35 dispatches/1 submit (NaniteCull.ts:
998-1010) → kRasterArgs2 (1/1, :1013-1015) → fanout 3 submits (:1042-1046) → world1 batch
[kVisClear, kRasterWorld1, kHwArgs] 1 submit (NaniteRaster.ts:1418-1419) → hwRender render
pass (:1347-1363) → dispatchVoxel: pyramid chain 1 submit + kClearBins + indirect scatter
(NaniteVoxelRaster.ts:1454-1492) → hzb.build 1 submit (NaniteFrame.ts:481, NaniteHzb.ts:
150-155) → post (~2-3 render/compute submits). ≈ 11-13 submits, ~67 dispatches — matches
doc 10's table; W2's 7 foldable submits confirmed at the platform layer (the batching
machinery supports all of them; order constraints documented at each site).

### Mission 2 — the voxf2b +5/+6/+16 ms, from the platform side

Dispatch code (NaniteVoxelRaster.ts:1437-1452, 1481-1486): K=16 bucket kernels
(`?voxf2bk` default 16, NaniteCull.ts:327-328), each tagged indirect, in ONE
`dispatchBatchMixed` submit; all 16 write visPayloadV/visBV ⇒ 15 implicit full drains.

- **Barrier cost per se: ruled out** (P3: ≤10 µs × ~15 boundaries ≪ 1 ms).
- **Aerial (+16)**: 659 clusters → mean 41 WGs/stage = **1.3 WGs/core ≈ 5 resident
  simdgroups/core** (vs ≤16 under the tg-mem cap) ⇒ device-load latency (~150-300 ns)
  almost fully exposed in Phase B's serial brick loop; each stage additionally drains to
  its longest cluster — aerial holds the game's largest footprints (FarTile heads,
  ≤128 bricks × 144-576 px at the τcap 12-24 px sawtooth, doc 14) ⇒ worst-cluster tail
  ~65-200 µs/stage at collapsed hiding. 16 × (underfilled throughput + tail) ≈ 17-24 ms of
  vox time vs 5.7 ms wide ⇒ Δ ≈ +11-18 — brackets the observed +16 (vox share 21.7 ms).
- **Eye (+5)**: 5893 clusters → 368 WGs/stage ≈ 2.9× machine residency ⇒ stages internally
  fill; the cost is 16 stage-drain tails (last partial residency wave + longest WG,
  ~0.2-0.35 ms each) ≈ +3-6. Oblique (+6) sits between.
- The DEFAULT wide dispatch is itself residency-throttled (≤128 WGs resident of 659-9188
  queued) — which is why kernel-time scales so cleanly per-cluster (doc 13's 1.6-2.2 µs
  slope): the machine processes ~128-WG waves back-to-back. Raising residency (L1) attacks
  the slope itself.

Discriminator stays P1 (K=4 linearity). The platform law to write on the wall:
**partitioning a dispatch is safe only while every partition still exceeds machine
residency (~128 WGs of THIS kernel, ~16k threads); below that you pay latency-exposure ×
stage count.**

---

## Waste map

| # | waste | mechanism | est. | cite |
|---|---|---|---|---|
| W1 | vox scatter residency cap | 6.5 KB tg-mem/WG ⇒ 4 WGs/core; ~2.5 KB of it is u32 arrays holding u16-range values | part of the 1.6-2.2 µs/cluster slope; bounded by L1 probe | NaniteVoxelRaster.ts:526-549 |
| W2 | 7 foldable submits + per-dispatch bind churn | 1 encoder+pass+submit per compute() call; uncached setBindGroup per dispatch | ~0.2-0.7 ms GPU + ~0.3-0.5 ms CPU | doc 10 W2; WebGPUBackend.js:1419-1429 |
| W3 | dead hwRT rgba8 tile STORE every frame | three hardcodes storeOp=Store; target never read (clear already skipped) | 13.4 MB/frame ≈ 0.03-0.1 ms | NaniteRaster.ts:290-304,1212; WebGPUBackend.js:606 |
| W4 | timestamp apparatus in every canonical number | timestampWrites on every pass + 2 pool resolves + mapAsync per frame | unknown, likely 0.05-0.3 ms + cpuSubmit noise | Engine.ts:74,219-239 |
| W5 | Phase-B setup ×4 simdgroups | ~15 uniform loads re-issued by each simdgroup per brick (workgroup-uniform b-loop) | ~1-2 ms oblique (doc 13 #3) | NaniteVoxelRaster.ts:1142-1181 |
| W6 | visDepthV clear in the packed path | 3.34M atomicStores nothing reads | ~0.05-0.15 ms | NaniteRaster.ts:391 (doc 10 W1) |
| W7 | fresh batch arrays per frame | world1/dispatchVoxel build new JS arrays per call ⇒ new WeakMap group entries (GC-transient) | µs-level; hygiene only | NaniteRaster.ts:1419, NaniteVoxelRaster.ts:1476-1485 |

---

## Levers

### L1 — platform:vox-tgmem-diet — halve kVoxScatter's workgroup memory (M, IDENTICAL)
Mechanism: pack the u16-range shared arrays into halves — bbX0|bbY0 → one u32 (2×u16;
x ≤ 2267, y ≤ 1472 fit), bbW|bbH → one u32 (≤ BRICK_MAX_EXT=64 after the 128-px span cap),
occMask (16 bits) into the spare half of cand's word or the bbW word's high bits; keep the
voxcell f32 AABB arrays f32 (f16 would NOT be loss-exact — do not touch). ~6.6 →
~3.8-4.2 KB ⇒ residency 4 → 7-8 WGs/core (+75-100% resident simdgroups). Election math,
bbox values, and every painted pixel identical — only the shared-memory ENCODING changes.
Quality: IDENTICAL (integer repack, bit-exact round-trip; shotdiff must be 0).
Expected: eye 0.3-1 / oblique 1-3 / aerial 0.2-0.5 — it attacks the per-cluster slope's
latency-exposure term; error bars honest (register file may become the next cap).
Probe: `?voxtgpack=0` restores the wide layout;
`CONFIG=default LABEL=dr18-tgpack TICKS=0 COOLDOWN_S=45 TREES=200000 npx tsx
tools/probe-fresh-stutter.ts` vs `EXTRA=voxtgpack=0 LABEL=dr18-tgpack-ctl`, + shotdiff.
Effort: M. Risks: unpack ALU in Phase B's hot loop (a shift+mask per read — cheap vs a
memory-latency stall); overlaps doc 13 L3 (subgroup-per-brick + compaction) — land one,
re-measure before the other.

### L2 — platform:coalesce-submits+bind-churn (S/M, IDENTICAL) — endorse doc 10 W2
Platform verification adds: the enabler (Tsl.ts:283-289) is verified against three's
actual fallback lines (WebGPUBackend.js:1431-1446); order-correctness is the array order;
render passes cannot join a compute encoder in three's architecture (each render() is its
own submit) — so 12 → ~6 submits is the floor without patching three. CPU win comes from
fewer encoder/pass/submit creations; the per-dispatch setBindGroup churn stays (it is
per-NODE, not per-submit) — a ThreePatches-class memo of unchanged bind groups per pass is
the optional deeper cut (~0.2-0.4 ms CPU at 67 dispatches). Expected: 0.2-0.5 ms GPU +
0.3-0.5 ms cpuSubmit. Probe: doc 10 C1.

### L3 — platform:sorted-single-dispatch F2B (S, IDENTICAL) — the occupancy-correct F2B
The platform analysis of Mission 2 makes doc 01's L3 the uniquely right F2B consumption:
run the existing counting-sort fanout (kVoxRange→kVoxScatterFan, NaniteCull.ts:599-710 —
kVoxPrefix already publishes whole-list args at :663) and dispatch ONE whole-list scatter
over the reordered queue (NaniteVoxelRaster.ts:1424-1428 path). Near clusters land in low
workgroup indices ⇒ launch-order F2B seeds the prevE guards/bocc — with ZERO residency
loss (single indirect dispatch keeps all ~128 WGs resident). ~30 lines (`?voxsort=1`).
Quality: IDENTICAL (reorder of an order-free atomicMax election).
Expected: oblique −1..−3 / eye 0..−0.5 / aerial ~0. Gate: `?voxwrites` −≥25% AND oblique
med −≥2 ms (doc 01 P-L3). Effort: S.

### L4 — platform:subgroup ops, the SAFE subset (M, IDENTICAL)
Available today: device feature ON, TSL ops present (three.webgpu.js:42516-42541), every
kernel already compiles with `enable subgroups;`. What wave ops buy HERE:
(a) **Phase-B per-brick setup dedup** — the brick loop is workgroup-uniform, so
`subgroupBroadcastFirst` of the ~15 setup words cuts the ×32-lane redundant loads to
×1-per-simdgroup at ZERO shared-memory cost (the subgroup-op alternative to doc 13 L3's
shared-list). (b) **Wave-aggregated appends** (ballot + subgroupExclusiveAdd + one
atomicAdd/wave) for kVoxFanout/hwQueue — sound, but appends live in the ~0.2 ms cull ⇒
non-lever. (c) UE5's BRICK_TRACE_WORK_REDISTRIBUTION (doc 01 §2.7) — **blocked as-is**:
WGSL subgroups give NO reconvergence guarantee, so ballot-driven ring redistribution
inside divergent control flow is not portable-sound WGSL. Only (a) ships.
Quality: IDENTICAL (uniform-control-flow broadcast of values every lane already loads).
Expected: oblique 0.5-1.5 / eye 0.2-0.5 / aerial ~0.1 (same pool as W5/doc 13 L3 —
mutually exclusive accounting). Probe: `?voxsgb=1` behind a flag; smoke-test first that
Tint accepts the op mix in this kernel (P6). Effort: M. Risk: fixed 32-lane assumptions —
read `subgroupSize` builtin, never hardcode.

### L5 — platform:hwrt-store-discard (S, IDENTICAL)
Mechanism: ThreePatches-class hook setting `storeOp:'discard'` on the hwRT color
attachment (mutate the cached render-pass descriptor for that renderContext — the
`colorAttachmentsConfig.storeOp` plumbing exists internally at WebGPUBackend.js:606 but is
not exposed for normal renders). hwRT has no depth buffer (NaniteRaster.ts:1212); the
rgba8 is written by nothing (colorWrite=false) and read by nothing — a discard is
byte-identical downstream and skips the TBDR tile flush.
Expected: 0.03-0.1 ms all poses (13.4 MB/frame @ ~400 GB/s + tile scheduling).
Probe: A/B `?hwdiscard=0/1` + shotdiff 0. Effort: S. Risk: three upgrade fragility — same
class as installFragmentStorageWrites (D-N11), document it next to that patch.

### L6 — platform:timestamp-apparatus A/B (S, measurement-only)
Mechanism: a `?nots=1` boot flag → `trackTimestamp:false` + skip the per-frame
`resolveTimestampsAsync` pair (Engine.ts:74,219-239). gpuWall (onSubmittedWorkDone) and
the live milestone metrics survive; per-pass attribution is lost for that run only.
Expected: unknown — 0-0.3 ms gpuWall + cleaner cpuSubmit; even a null result hardens every
future A/B. Probe: paired same-session run at the 3 poses + one 600-tick live. Effort: S.

### Deferred/bundle-only
- Pyramid-tail fuse / SPD-style single-dispatch pyramid (doc 10's lever; ~0.1-0.3 ms) —
  bundle with any pyramid rework, not standalone.
- HZB/voxOccPyr as r32float/r32uint storage TEXTURES (2D-swizzled locality for the 2×2
  window reads + frees 1-2 storage-buffer slots vs the 10 cap): expected small (~0.1 ms
  class, the tests are off the per-pixel path); only if a pyramid rework happens anyway.
  Note: the ELECTION buffers can never move — WebGPU has no texture atomics.

---

## Refuted (do not rebuild)

- **shader-f16 as a bandwidth lever.** The only f32 storage the hot path streams is the
  HZB (~22 MB/chain-walk ≈ 55 µs at 400 GB/s) — halving it saves ~30 µs and needs
  round-toward-far care to stay conservative; visPayloadV/visBV are u32 ATOMIC words
  (32-bit granularity is mandatory); TSL r184 exposes no f16 storage type anyway (only
  pack/unpack 2x16, which is core WGSL and already available). No target ≥0.1 ms exists.
- **Two-word / "wide" election schemes.** No 64-bit atomics in WebGPU (gpuweb#5071,
  NANITE-SPEC.md:95-97); a depth-word + payload-word protocol adds ≥1 word of traffic per
  win plus a retry loop, to fix a visBV race that is already sub-visible, audited
  (`?audit=1`), and Class-I-accepted. UE5 offers no 32-bit recipe to copy (doc 01 §2.6).
- **Dispatch/barrier overhead as a cost center.** Bounded ≤ ~10 µs/dispatch, ≤ ~100 µs/
  submit (P3). The BFS's ~30 near-empty tail dispatches, both pyramid tails, and the ~24
  one-thread args kernels total ≲0.5 ms — only worth folding as part of L2, never as a
  standalone hunt. The F2B +16 was never barrier cost (Mission 2).
- **Persistent-threads anything.** WebGPU has no forward-progress guarantee; the cull the
  pattern would help is ~0.2 ms (doc 01 non-levers).
- **UE5 wave-ring work redistribution ported literally.** WGSL's subgroup model guarantees
  no reconvergence; the HLSL pattern's correctness assumptions do not transfer. Only the
  uniform-control-flow subset (L4a) is sound.
- **Feature-request changes.** three already requests everything (three.webgpu.js:
  80055-80074); there is nothing to unlock at the device level.

---

## Open questions + proposed serial probes

All `TICKS=0 COOLDOWN_S=45 TREES=200000 npx tsx tools/probe-fresh-stutter.ts` pattern,
run after the day's baselines; [patch] = small src/tools edit first.

**P1 — F2B K-linearity (Mission 2 closer; = doc 10 M2).**
`CONFIG=default EXTRA=voxf2b=1,voxf2bk=4 LABEL=dr18-f2bk4` → aerial ≈ +4-5 ms (residency
model) vs ~+16 flat (fixed overhead — would contradict P3's bound).

**P2 — residency ground truth for L1.** [patch: `?voxtgpack` flag]
A/B per L1. Secondary discriminator for the register-pressure share: doc 13's `?voxcell=0`
compile-variant slope probe (DIAGNOSTIC only). Out-of-band gold standard: one Xcode Metal
GPU capture of nanVoxScatter (occupancy + GPR count counters) — needs the WebGPU-Inspector
→ wgpu-native replay path, tooling session.

**P3 — cpuSubmit 5.3 vs 1.4 same-code paired re-measure (doc 10 Q3).**
Two back-to-back boots of the SAME build, same pose order; if the 5 ms mode reappears,
bisect run-state (thermal vs first-boot shader-cache compile vs code drift). The encode
floor model (15-20 µs/dispatch) only explains the 1.1-1.5 ms mode.

**P4 — timestamp-apparatus severance (L6).** [patch: `?nots=1`]
Paired iso + live run. Also fixes W4 as a metric premise for all future baselines.

**P5 — adapter-limit re-probe on Chrome majors.**
Log `maxStorageBuffersPerShaderStage` from Diagnostics at boot into the probe JSON; if
Dawn's Metal argument-buffer path ever lifts 10 → ≥16, several packing contortions (SCAR
fold, K args-kernel split) become optional.

**P6 — subgroup plumbing smoke test (pre-L4).** [patch: tiny `?sgsmoke=1` kernel]
One 64-lane kernel using subgroupBroadcastFirst + subgroupAdd through the TSL nodes, value
checked via readback — proves Tint/Dawn/TSL agreement on this stack before L4 is built
into the hot kernel.

**Open questions**
1. Is the vox monolith tg-mem-bound or GPR-bound? (L1 vs the doc 13 two-kernel split —
   P2 decides which lands first.)
2. Does Metal serialize ALL dispatches in a pass or only hazard-detected ones on this
   Dawn version? (Academic given P3's bound, but it decides whether hazard-free kernels
   could overlap inside one batched submit — potential free parallelism for the
   cull-vs-shadow overlap pattern, NaniteFrame.ts:446-461.)
3. The live 25.0 ms quantum frames: cpu.submit p95 is 1.5 ms (fresh-voxbocc-milestone
   live) ⇒ the platform layer is NOT the second-quantum driver; the gap lives in GPU work
   (docs 13/14/15 territory). Platform's contribution to the mission is L1+L3 (+L2/L5
   crumbs) ≈ 1.5-4 ms oblique-class, all quality-IDENTICAL.

Sources: [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks),
[dougallj Apple G13 GPU reference](https://dougallj.github.io/applegpu/docs.html),
[Metal by Example — WebGPU for Metal developers](https://metalbyexample.com/webgpu-part-one/),
[Chrome 134 subgroups](https://developer.chrome.com/blog/new-in-webgpu-134),
[gpuweb#5071 64-bit atomics proposal](https://github.com/gpuweb/gpuweb/issues/5071).
