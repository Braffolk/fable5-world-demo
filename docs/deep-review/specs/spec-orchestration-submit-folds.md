# SPEC: orchestration + submit-path folds (W1-W5 of the frame-orchestration review)

**Lever id:** `frame-orchestration:coalesce-submits` + `skip-depthv-clear` + `fuse-pyramid-tails` +
`probe-hygiene` (doc 10, `docs/deep-review/10-frame-orchestration.md` §Waste inventory W1-W5).
**Branch:** `nanite-raster` (HEAD 6a93dfb + uncommitted). **Expected:** ~0.35-0.95 ms GPU per pose
(oblique ≈ **-0.7 ms** mid-estimate) + **-0.5-1.5 ms cpu.submit** (live frames benefit most).
**Quality:** every stage is argued bit-identical or byte-identical below — no conservative-cull
slack is even needed. Stage 4 is measurement-infra only (no frame change).

This spec is self-contained: every edit site was re-verified against the working tree on 2026-07-02.
You do NOT need the review corpus to build it.

---

## 0. Problem + measured motivation

The default forest frame (`?scene=forest&trees=200000&nanite=1&dpr=1.5`, voxActive=true, voxf2b OFF,
shadows/GI null in forest — `src/debug/ForestScene.ts:456-462`) issues **~12-14 queue.submits** per
frame from `NaniteFrame.render()` (`src/nanite/NaniteFrame.ts:425-497`):

| # | call site | submits | content |
|---|-----------|---------|---------|
| 1 | `cull.runPhase1` — NaniteFrame.ts:462 → NaniteCull.ts:1008-1010 | 1 | batched BFS: `[kClearHier, kSeedRoots, (kArgs,kTraverse)×hierDepth, kRasterArgs]` |
| 2 | `cull.syncFullArgs` — NaniteFrame.ts:463 → NaniteCull.ts:1013-1015 | **1** | ONE 1-thread kernel (`kRasterArgs2`) |
| 3 | `cull.runVoxFanout` (non-F2B) — NaniteFrame.ts:468 → NaniteCull.ts:1044-1046 | **3** | `kVoxFanoutArgs` (1 thread); `kVoxFanout` (indirect); `kVoxRasterArgs` (1 thread) |
| 4 | `raster.world1` compute — NaniteRaster.ts:1419 | 1 | `[kVisClear, kRasterWorld1(indirect), kHwArgs]` (already coalesced) |
| 5 | `hwRender` — NaniteRaster.ts:1347-1363 | 1 | render pass (cannot merge with compute encoders) |
| 6 | `voxRaster.dispatchVoxel` — NaniteVoxelRaster.ts:1454-1492 | **3** | voxOccPyr chain (12 kernels, 1 batch, line 1461); `kClearBins` (line 1489 — zeroes ONE debug word); `kVoxScatter` indirect (line 1490) |
| 7 | `hzb.build` — NaniteFrame.ts:481 → NaniteHzb.ts:150-155 | **1** | 12 max-pool kernels, 1 batch |
| 8 | `post.render()` + auto-exposure meter | ~3-4 | untouched by this spec |

Seven of these submits are foldable (bold). Additional per-frame waste:

- **W1 dead depth clear**: `kVisClear` atomic-stores `0xffffffff` into `visDepthV` over all
  3.34 M pixels every frame (`src/nanite/NaniteRaster.ts:391`), but in the world single-pass path
  NOTHING reads or writes `visDepthV` (proof in §2.2) — ~0.05-0.15 ms/frame of pure store traffic.
- **W3 kClearBins**: a whole submit to zero ONE u32 (`NaniteVoxelRaster.ts:309-315`) whose only
  producer is build-time-disabled by default (`voxwrites`, line 241; the `atomicAdd` at 1210-1211
  is not even compiled) — the cleared word is constant 0 either way.
- **W4 pyramid tails**: both half-res pyramids (HZB `NaniteHzb.ts:99-148`, voxOccPyr
  `NaniteVoxelRaster.ts:429-474`) are 12 barrier-separated dispatches each; the bottom ~7 levels
  are ≤864 texels (36×24 → 1×1) — 14 near-empty dispatch+UAV-barrier boundaries per frame.
- **W5 meter contamination (measurement-infra)**: every 15th frame `NaniteFrame.meter`
  (`NaniteFrame.ts:509-519`) launches 4-5 async readbacks (`readCounts`×2, `readHwCount`,
  `readVoxCount`, `readVoxWrites` — each a buffer→staging copy submit + mapAsync). `meter` is
  driven from inside `Engine.renderStep` (`src/core/Engine.ts:157`), which is exactly the timed
  window of `MeasureHarness.measure` (`src/core/MeasureHarness.ts:147-151`) — ~2 of every 32
  isolated samples per pose are contaminated.
- **Harness defects (measurement-infra)**: `capSuspect` (`MeasureHarness.ts:162-169`) compares
  per-pass timestamp totals — the known-unreliable cross-frame spans — against gpuWall, so it fires
  on ~every frame; the probe's rejection fallback (`tools/probe-fresh-stutter.ts:211-212`) then
  uses ALL frames. Measured: capRejects = 15-32 of 16-32 on every run/pose (fresh-final-rested:
  32/32/23 of 32; fresh-voxbocc-milestone: 31/32/29 of 32; fresh-rested-def: 15/16/15 of 16).
  Outlier filtering is dead. Separately, `gpuWallMs` = tSubmit → drain-promise-resolution
  (`MeasureHarness.ts:148-160`), so a GC pause/longtask during the await inflates the sample
  (prime suspect for the rare 70-91 ms "GPU" spikes).

Empirical bound (doc 10 work model, measured via the voxwaves A/B): per-submit overhead
≤ ~100 µs GPU, per-barrier-separated-dispatch ≤ ~10 µs. 7 folded submits + 14 fused tail
boundaries + the dead clear ⇒ **~0.5-1.2 ms GPU per pose, all poses**, plus the CPU-encode share
of 7 submits out of cpuSubmit medians of 1.1-5.4 ms (JSON-verified). This is NOT the -11 ms
oblique lever; it is the trivially-quality-identical floor-sweep that also cleans the measurement
apparatus every future lever depends on.

---

## 1. Design

Four independent stages, each behind its own flag, each measurable alone. All flags land
**default-OFF** (legacy behavior byte-preserved); defaults flip in a follow-up commit after each
stage's gate passes.

### Stage 1 — `?coalesce=1`: fold the 7 submits (W2 + W3)

#### 1a. Cull side: BFS + syncFullArgs + voxel fanout = ONE submit

**Mechanism.** `dispatchBatchMixed` (`src/nanite/Tsl.ts:304-306`) already gives ONE encoder / ONE
compute pass / ONE submit with per-dispatch UAV auto-sync, and `setIndirectDispatch`
(`Tsl.ts:283-289`) lets an indirect kernel keep its tight grid inside a batch (the r184
`dispatchSize`-attach mechanism, verified against `node_modules/three/src/renderers/webgpu/WebGPUBackend.js:1431-1447`:
the per-node fallback `dispatchSize = computeNode.dispatchSize || computeNode.count` fires only
when the outer arg is null, so tagging is also harmless to the legacy explicit
`dispatchIndirect(renderer, k, attr)` path).

**Edits — `src/nanite/NaniteCull.ts`:**

1. Line ~1026 (next to the existing F2B tags at 1024-1026): add
   ```ts
   setIndirectDispatch(kVoxFanout, voxFanoutDispatchAttr);
   ```
   (`kVoxFanout` is the ONLY fanout kernel not yet tagged; `kVoxFanoutArgs`/`kVoxRasterArgs` are
   direct 1-thread kernels and need no tag.)
2. Expose the pieces instead of hard-wiring submits — extend the return object
   (lines 1092-1111) with:
   ```ts
   fullArgsBatch: () => [kRasterArgs2],
   voxFanoutBatch: () => (voxf2b
     ? voxF2bBatch                                   // already one coherent list (1030-1037)
     : [kVoxFanoutArgs, kVoxFanout, kVoxRasterArgs]),
   ```
   Keep `syncFullArgs`/`runVoxFanout` (1013-1048) untouched for the legacy path.
3. Fix the stale comment at 992-995 ("kept out of this batch on purpose … invoked after the
   voxel fan-out") — the frame has called `syncFullArgs` BEFORE `runVoxFanout` since the
   single-phase rewrite (`NaniteFrame.ts:463` vs `:468`); the separation reason no longer exists.

**Edits — `src/nanite/NaniteFrame.ts` (lines 445-469):** read `const coalesce =
params.get('coalesce') === '1';` at build time; in `render()` replace the three calls:

```ts
if (!frozen) {
  if (coalesce) {
    const batch = [
      ...cull.phase1Batch(),        // [kClearHier, kSeedRoots, (args,traverse)×D, kRasterArgs]
      ...cull.fullArgsBatch(),      // [kRasterArgs2]
      ...(voxActive ? cull.voxFanoutBatch() : []),
      ...(shadowCutBatchOrEmpty),   // culloverlap path only, appended LAST as today (line 456)
    ];
    dispatchBatchMixed(renderer, batch);
  } else {
    /* existing lines 453-468 verbatim */
  }
}
```

**RAW/WAW audit of the folded order** (array order IS execution order; each dispatch is its own
usage scope with UAV auto-sync between them — Tsl.ts:296-303):

- `kRasterArgs2` (NaniteCull.ts:482-488) reads `counters[1]` (written by the traverses) and
  `qRasterV.rw[0].y` (written `(n,0)` by `kRasterArgs` at 472-476, one dispatch earlier) → reads
  base=0, writes `(nT,0)` + `rasterDispatch2` + `rasterDispatchFull`. Same values as the
  cross-submit ordering today (queue order == in-pass dispatch order).
- `kVoxFanoutArgs` (504-509) reads `counters[1]`, zeroes `voxCount`, writes `voxFanoutDispatch`.
  No dependency on `kRasterArgs2` (disjoint buffers) — relative order preserved anyway.
- `kVoxFanout` (513-532) indirect-reads `voxFanoutDispatch` (RAW on STORAGE|INDIRECT — the exact
  pattern the BFS batch already relies on for `traverseDispatch`, comment at 986-991), reads
  `qRasterV.ro` entries + `counters[1]`, appends `qVoxRasterV.rw` + `voxCount`.
- `kVoxRasterArgs` (538-542) reads `voxCount` (RAW), writes `qVoxRaster[0]` + `voxRasterDispatch`.
- culloverlap (`NaniteFrame.ts:453-461`, default-off, forest-dormant since csm=null): the shadow
  cut batch writes disjoint buffers (own counters/queues, comment at 446-450) — appending it after
  the fanout is equivalent to today's `[phase1, shadowCut]` + later separate submits.

Same kernels, same dispatch grids (tags preserve indirect sizes), same producer→consumer order,
same barrier semantics ⇒ **bit-identical**.

#### 1b. Raster side: voxOccPyr + (kClearBins) + kVoxScatter + HZB = ONE submit

**Edits — `src/nanite/NaniteVoxelRaster.ts`:**

1. After `kVoxScatter` is built (1424-1429), tag it:
   ```ts
   setIndirectDispatch(kVoxScatter, voxRasterDispatchAttr);
   ```
2. `dispatchVoxel` (1454-1492) gets an optional tail and a coalesced default path:
   ```ts
   const dispatchVoxel = (renderer: Renderer, tail: readonly unknown[] = []): void => {
     if (coalesce && !voxF2bEnabled) {
       dispatchBatchMixed(renderer, [
         ...(voxOccl ? voxPyrKernels : []),
         ...(voxWrites ? [kClearBins] : []),   // W3: dropped when the counter can't increment
         kVoxScatter,
         ...tail,                              // the HZB chain (§1c), possibly empty
       ]);
       return;
     }
     /* existing body 1455-1491 verbatim; if tail.length, dispatchBatch(renderer, tail) at the end */
   };
   ```
   The F2B / voxwaves paths (default-off A/B controls) keep their exact current shape — do NOT
   micro-optimize refuted paths.

**Usage-scope legality:** within one compute pass, each dispatch is its own usage scope. The pyramid
kernels use `visPayloadV.ro` + `voxOccPyr.rw`; `kVoxScatter` uses `visPayloadV.atomic` +
`voxOccPyr.ro`; HZB kernels use `vis.payloadV.ro` + `hzbF.rw`. No single dispatch mixes two views
of one buffer (the N0 same-scope law holds per-dispatch, as today). Cross-dispatch view changes
are legal and auto-synced.

**W3 equivalence:** with `voxwrites` unset the `atomicAdd` producer is never compiled
(NaniteVoxelRaster.ts:241, 1210-1211), so the word is 0 at init and stays 0; clearing it to 0 is
a no-op. The every-15-frame `readVoxWrites` (`NaniteFrame.ts:518` → `NaniteRaster.ts:1458-1459` →
`readWriteCount` 1494-1497) returns 0 both before and after. With `?voxwrites=1` the clear is kept
(first in the batch, before the scatter's adds — same order as today). **Byte-identical either way.**

#### 1c. HZB fold into the same submit

**Edits:**

- `src/nanite/NaniteHzb.ts`: add `batch: () => kernels` to the returned handles (150-155 keep
  `build` as-is — `dispatchBatch(kernels)`).
- `src/nanite/NaniteRaster.ts` `world1` (1418-1424): accept + forward the tail:
  ```ts
  const world1 = (renderer, camera, hzbTail: readonly unknown[] = []): void => {
    dispatchBatchMixed(renderer, [kVisClear, kRasterWorld1, kHwArgs]);
    hwRender(renderer, camera, hwWorld1Mat);
    if (voxRaster) voxRaster.dispatchVoxel(renderer, hzbTail);
    else if (hzbTail.length) dispatchBatch(renderer, hzbTail); // noleaves: same 1 submit as today
  };
  ```
- `src/nanite/NaniteFrame.ts`: compute per-frame
  `const foldHzb = coalesce && !frozen && !probeOn;` and call
  `raster.world1(renderer, engine.camera, foldHzb ? hzb.batch() : [])`; change line 481 to
  `if (!frozen && !foldHzb) hzb.build(renderer);`.
  - `!frozen` preserves the cullfreeze semantics (HZB must NOT rebuild when frozen — line 481).
  - `!probeOn` (`nanprobe=1`, line 299) preserves the exact probe insertion points at 480/482
    (probe-only; keeps the forensics tool's semantics untouched).

**Ordering audit:** today the order is world1-compute → hwRender → voxPyr → clear → scatter →
`raster.scar` (no-op unless `?scar=1`) → `hzb.build`. Folded order runs HZB *before* `raster.scar`.
`kScarCovered` (NaniteRaster.ts:1067-1096) reads `visPayloadV.ro/visBV.ro` + writes scar counters;
HZB kernels read `visPayloadV.ro` + write `hzbF`. Zero shared writes ⇒ the swap cannot change any
value. HZB still runs strictly AFTER `kVoxScatter` in the same pass (UAV-synced), so it pools the
identical post-vox election — same pyramid bits as today. **Bit-identical.**

**Submit count after Stage 1** (default forest): cull(1) + world1(1) + hwRender(1) +
vox+hzb(1) + post(~3-4) ≈ **7-8, down from 12-14**.

### Stage 2 — `?dvclear=0`: skip the dead visDepthV clear (W1)

**Mechanism.** Build-time gate inside `kVisClear` (`src/nanite/NaniteRaster.ts:388-409`):

```ts
const params = new URLSearchParams(window.location.search);
const skipDepthClear =
  singlePass &&
  params.get('dvclear') === '0' &&      // flag (flip default later)
  params.get('nanprobe') !== '1' &&     // probe reads vis.depthV.ro — NaniteFrame.ts:320
  params.get('audit') !== '1' &&        // kAudit reads visDepthV.ro — NaniteRaster.ts:1048
  rdbg === 0;                           // rdbg sinks atomicMin depthV — lines 590/626/657/833
// in kVisClear:
if (!skipDepthClear) atomicStore(visDepthV.atomic.element(instanceIndex), uint(0xffffffff));
```

**Proof that no consumer exists in the gated configuration** (exhaustive `depthV` grep over src/,
2026-07-02):

| consumer | path | why unaffected |
|---|---|---|
| world1 SW election | NaniteRaster.ts:925-975 | comment + code: "NO depthV write" (968-971) — never touches depthV |
| world1 HW fragment | NaniteRaster.ts:1157-1183 | writes payload/visB only; 'depth' pass (1145-1146) belongs to shadow instances with their OWN vis buffers (`NaniteShadow.ts` creates `makeVisBuffers(SHADOW_PIX)` per cascade; NaniteShadowClip likewise) |
| HZB | NaniteFrame.ts:197 `buildNaniteHzb(vis.payloadV.ro, cam, true)` | world HZB is packed — reads payloadV, never depthV (NaniteHzb.ts:62-68,118-121) |
| resolve (frame) | `buildNaniteResolve` | reads the 24-bit election key; "there is no exact depthV" (NaniteResolve.ts:295) — no depthV binding |
| shadowHalf | NaniteShadowHalf.ts:99 | reads `vis.payloadV.ro` only (and is null in forest anyway) |
| raster-internal debug resolve | NaniteRaster.ts:1243,1328 (non-packed branch) | `resolveScene` is only rendered by NaniteView debug scenes, never in frame mode (frame adds `buildNaniteResolve`'s meshes, NaniteFrame.ts:288-294); TSL materials never compiled ⇒ zero cost, zero reads |
| kAudit / nanprobe / rdbg sinks | 1048 / NaniteFrame.ts:320 / 590+ | explicitly kept-cleared by the gate above |
| depth1/kRasterDepth ('depth' mode writes) | NaniteRaster.ts:905-909 | `depth1` is never called in frame mode (frame calls `world1` only, NaniteFrame.ts:475) |

With no reader and no writer, the buffer holds its boot-time zeros forever; skipping the clear
changes no observable value. **Byte-identical frame output.** (Shadow/View raster instances have
`singlePass=false` ⇒ never gated.)

### Stage 3 — `?pyrfuse=1`: fuse both pyramid tails (W4)

**Mechanism.** In each chain, replace the levels with ≤1024 texels (at 2268×1473: levels 5..11,
36×24=864 down to 1×1 — 7 dispatches) with ONE single-workgroup kernel that loops the remaining
levels with `storageBarrier()` between them. `storageBarrier` is exported by three r184
(`node_modules/three/src/Three.TSL.js:520`) and is exactly the storage-address-space control
barrier needed — valid because the fused kernel is ONE workgroup (`.compute(256, [256])`), so the
barrier synchronizes ALL participating invocations.

**Edit — `src/nanite/NaniteHzb.ts` (kernel-build loop at 98-148):**

```ts
const FUSE_MAX_TEXELS = 1024;
const fuseFrom = pyrfuse ? levels.findIndex(l => l.w * l.h <= FUSE_MAX_TEXELS) : -1;
// build per-level kernels only for k < fuseFrom (or all, when disabled), then when enabled:
const kFusedTail = Fn(() => {
  const tid = localX();                       // 256 lanes, ONE workgroup
  for (let k = fuseFrom; k < levelCount; k++) {          // STATIC unroll — uniform control flow
    const info = levels[k]; const src = levels[k - 1];
    const n = info.w * info.h;
    for (let base = 0; base < n; base += 256) {          // static stride loop (≤4 iters at L5)
      const i = uint(base).add(tid);
      If(i.lessThan(uint(n)), () => {
        /* the EXISTING k>0 2×2 max-pool body verbatim (lines 125-143), with
           x=i.mod(lw), y=i.div(lw); reads src level THROUGH hzbF.rw (same-scope law) */
      });
    }
    storageBarrier();   // top level of the Fn body — NEVER inside an If (barrier uniformity)
  }
})().compute(256, [256]);
kernels.push(kFusedTail);   // kernels[] = [L0..L(fuseFrom-1), fused] → build()/batch() unchanged
```

Mirror identically in `src/nanite/NaniteVoxelRaster.ts` for the min-pool chain (429-474): same
shape, `minU` instead of `.max`, `voxOccPyr.rw` (uint) instead of `hzbF.rw` (float), same
1024-texel threshold, flag-shared (`?pyrfuse`).

**Equivalence:** the fused kernel computes the same 2×2 clamped-window min/max over the same
source texels into the same offsets, entirely through the rw view (the same-scope law both chains
already obey — NaniteHzb.ts:126-128, NaniteVoxelRaster.ts:426-428). `storageBarrier()` between
virtual levels gives the same happens-before as today's inter-dispatch UAV sync, restricted to the
one workgroup that is doing all the work. **Bit-identical pyramid contents.**
12 → 6 dispatches per chain (2 chains) and, combined with Stage 1, the whole
pyr+scatter+hzb group is 1 submit with ~12 fewer barrier boundaries.

TSL codegen notes: the level loop and stride loop are JS-static (unrolled at build), so no r184
hoist hazard (no cross-`If` node reuse; the per-level bodies construct fresh nodes exactly like
the existing per-level kernels). No new storage buffers (each fused kernel binds 1-2 — far under
the Metal 10-buffer cliff). No 64-bit atomics anywhere.

### Stage 4 — measurement-infra (NO frame-content change; label all commits `measure-infra:`)

**4a. Meter silence during isolated measurement (W5).**
- `src/core/Engine.ts`: add a public field `meterQuiet = false`.
- `src/nanite/NaniteFrame.ts` `meter` (499-519): keep `post.meter(r)` (auto-exposure IS frame
  content) and the CPU-only counter mirrors (501-507); guard ONLY the readback block:
  `if (engine.meterQuiet || frame === 0 || frame % 15 !== 0 || reading) return;` (line 509).
- `src/core/MeasureHarness.ts` `measure()` (122-184): set `engine.meterQuiet = true` after
  stopping the rAF loop (line 131), restore in the `finally` (180-183).

**4b. Per-frame counters read OUTSIDE the timed window (feeds bimodality probe B1).**
- Factor the readback block of `NaniteFrame.meter` (512-567) into
  `meterRead(r): Promise<Record<string, number>>` (returns the would-be counter assignments);
  expose it on the frame handles and thence `engine.post` (extend the `Engine.post` field type at
  `Engine.ts:34` with optional `meterRead`).
- In `MeasureHarness.measure`, AFTER `tDone` and timestamp resolution (after line 156):
  ```ts
  const post = engine.post as { meterRead?: (r: unknown) => Promise<Record<string, number>> };
  const fresh = post?.meterRead ? await post.meterRead(engine.renderer) : null;
  // merge into the frame's counters snapshot (line 175)
  ```
  The readback submits+maps now run on a drained queue between samples — zero perturbation of
  `gpuWallMs`, and every `MeasuredFrame.counters` carries per-frame `nanite.visClusters` /
  `nanite.voxClusters` (probe B1's discriminator).

**4c. Fix the dead capSuspect (P2) + drain-await GC guard (P3).**
- Replace the per-pass-ghost criterion (MeasureHarness.ts:162-169) as the REJECTION flag with an
  event-loop-lag guard: during `await drain(device)` (line 151) run a 5 ms `setTimeout` ticker
  recording `maxGapMs`; set `capSuspect = maxGapMs > 8` (a GC/longtask pause > 8 ms materially
  inflates the wall sample). Keep the old per-pass comparison as a separate informational field
  `passGhost` (do not reject on it — per-pass timestamps are known cross-frame spans).
- `tools/probe-fresh-stutter.ts:211-212`: keep the >half fallback but `console.warn` loudly when
  it fires (it should now be rare), and emit `capRejects` into the per-pose console line.

Expected effect: capRejects drops from ~30-32/32 to ~0-3/32; the 70-91 ms spike class gets
correctly attributed (flagged frames) instead of polluting p95; medians are unchanged by
construction (they were already computed on gpuWall).

---

## 2. Correctness + quality-equivalence (user law: any visible pixel change = REJECTED)

- **Stage 1** issues the SAME kernels with the SAME dispatch grids in the SAME order; the only
  change is submit granularity. WebGPU's in-pass inter-dispatch UAV sync gives the identical
  happens-before that cross-submit queue order gave. No kernel body changes at all ⇒ bit-identical
  writes to every buffer, including the vis election, the pyramids and all indirect args.
  (kClearBins drop: the cleared word is provably constant-0 when `voxwrites` is unset — §1b.)
- **Stage 2** removes stores to a buffer with zero readers and zero writers in the gated config
  (exhaustive consumer table §2 — re-grep `depthV` before landing to catch drift), with automatic
  fall-back-to-clear under every debug flag that does read it (`nanprobe`/`audit`/`rdbg`).
- **Stage 3** computes the same reductions over the same inputs with an equivalent barrier
  structure ⇒ bit-identical pyramid texels ⇒ identical cull/occlusion decisions.
- **Stage 4** never touches the frame graph; it only silences probe-time readbacks (which are not
  frame content) and fixes harness bookkeeping.
- Empirical backstop for all stages: the probe's per-pose screenshots + `tools/diff.ts`
  (§4 gates). Note the election has benign run-to-run nondeterminism (equal-depth-key atomicMax
  ties), so the gate compares against a ctl-vs-ctl baseline diff, not against literal zero.

---

## 3. Risks + fallbacks

| risk | mitigation / fallback |
|---|---|
| r184 batching quirks: the `dispatchSize`-attach hack (Tsl.ts:274-289) breaks on a three upgrade | mechanism already load-bearing for the BFS batch (NaniteCull.ts:955-961) and world1 (NaniteRaster.ts:1028-1031); tsc/runtime surfaces a change immediately; `?coalesce=0` restores legacy submits |
| mis-ordered RAW inside a folded batch | array order IS execution order (Tsl.ts:300-302); the exact orders in §1a/§1b are normative — copy them verbatim; each stage flag lets you bisect a wrong image in seconds |
| GpuProfiler per-pass attribution coarsens (fewer, bigger compute passes) | accepted: whole-frame `gpuWallMs` is the trusted metric (memory: per-pass timestamps are unreliable anyway); note it in the landing commit |
| fused-tail barrier uniformity (WGSL requires `storageBarrier()` in uniform control flow) | barrier sits at the top level of the static level loop, never inside `If`; guard only the per-texel work. If TSL emits it wrong (validate with `?nanitedbg=hzb` viewer + a getValidationErrors pass), fall back to fusing via N single-level dispatches in one batch (Stage 1 already gives that) and drop Stage 3 |
| skipping depthV clear breaks an overlooked consumer | the gate auto-disables under nanprobe/audit/rdbg; acceptance includes the shot-diff gate; `?dvclear=1` reverts at runtime |
| meterQuiet starves HUD counters during long live probes | quiet is scoped strictly to `MeasureHarness.measure` (try/finally); live loop unaffected |
| folding changes cullfreeze/probe debug semantics | `foldHzb` explicitly excludes `frozen` and `nanprobe` (§1c) |

---

## 4. Staged landing plan + measurement gates

Thermal discipline: all A/Bs same-session or back-to-back with `COOLDOWN_S=45`; dev server on
:5173 (`npx vite --port 5173`). Canonical probe:

```
CONFIG=default TICKS=0 COOLDOWN_S=45 TREES=200000 LABEL=<label> [EXTRA=<flags>] \
  npx tsx tools/probe-fresh-stutter.ts
```

**Land order (each its own commit, flag default-OFF; flip default in a follow-up after its gate):**

| stage | flag | measure | accept iff |
|---|---|---|---|
| 0. ctl pair | — | `LABEL=fold-ctl` then `LABEL=fold-ctl2` (same code, twice) | establishes noise band `N` (max abs median delta across poses) and the ctl-vs-ctl shot-diff baseline `D0` via `npx tsx tools/diff.ts --a shots/fold-ctl-<pose>.png --b shots/fold-ctl2-<pose>.png --thr 12` |
| 1. coalesce | `?coalesce=1` | `EXTRA=coalesce=1 LABEL=fold-coal` vs `LABEL=fold-coal-ctl` | gpuWall median: no pose regresses > max(0.3, N) ms AND (cpuSubmit median improves ≥ 0.3 ms OR any pose gpu improves ≥ 0.2 ms); shot-diff per pose ≤ D0 + 0.02 pp changed-pixels |
| 1L. live check | — | `EXTRA=coalesce=1 TICKS=1200 LABEL=fold-coal-live` vs ctl | live p50/p95 not worse; expect p50 −0.3-1 ms from the submit-encode share |
| 2. dvclear | `?dvclear=0` | `EXTRA=coalesce=1,dvclear=0 LABEL=fold-dv` vs stage-1 build | gpu median −0.05-0.15 ms or neutral, never > N worse; same shot-diff rule; sanity: `?nanprobe=1` still returns coherent depths (probe forces the clear back on) |
| 3. pyrfuse | `?pyrfuse=1` | `EXTRA=coalesce=1,dvclear=0,pyrfuse=1 LABEL=fold-pyr` vs stage-2 build | gpu median improves or neutral (bounded ≤ ~0.3 ms — do not over-read); `?nanitedbg=hzb` viewer visually identical + shot-diff rule; NO WebGPU validation errors |
| 4. infra | none (harness/probe only) | rerun `LABEL=infra-ctl` | capRejects ≤ 3/32 per pose; per-frame counters present in phase-B JSON frames; medians within N of stage-3 run |

**Decision rule overall:** keep any stage that passes its row; park (do NOT delete — surface for
the user's call) any stage that fails after one honest re-measure. Do not chase < N deltas with
re-runs (thermal noise discipline). After all gates: flip the four defaults
(`coalesce=1`, `dvclear=0`, `pyrfuse=1`, infra unconditional) in one commit citing the JSONs, and
re-baseline the canonical numbers (the infra fixes change capRejects semantics, not medians).

**Expected final deltas** (isolated, per doc-10 bounds): eye −0.35-0.95 ms, **oblique −0.35-0.95 ms
(mid ≈ −0.7)**, aerial −0.3-0.85 ms; live cpuSubmit −0.5-1.5 ms; plus every future probe stops
mismeasuring (dead outlier filter fixed, meter contamination gone, per-frame counters available
for the bimodality discrimination B1).
