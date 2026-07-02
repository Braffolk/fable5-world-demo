# frame-orchestration deep review (2026-07-02)

Scope: the full per-frame dispatch graph — order, batching, barriers, indirect dispatches, the
voxOccPyr + HZB pyramid chains, the measurement harness — plus the three special missions
(bimodality, voxf2b +16 ms at aerial, dead/duplicated per-frame work).

All numbers from the session JSONs in the scratchpad (`fresh-*.json`) and the fact pack; code
cites are `path:line` at HEAD of `nanite-raster` (6a93dfb + uncommitted work).

---

## Premise audit

**P1. Two of the fact pack's bimodality suspects do not exist in the forest scene (dead by code).**
`ForestScene.ts:456-462` passes `gi: null, canopyTex: null, csm: null` into `buildNaniteFrame`.
Consequences:
- **ProbeGI never runs** in any canonical forest measurement (`NaniteFrame.ts:70` receives null;
  no gather/publish dispatches). "GI probe updates" is refuted as a bimodality suspect by code
  inspection, not measurement.
- **The entire nanite shadow system never builds**: `NaniteFrame.ts:244`
  `shadowOn = params.get('nanshadow') !== '0' && world.csm !== null` → false. No shadow cull, no
  shadow raster, no shadowHalf (`NaniteFrame.ts:267` needs `shadow`). The measured
  "`nanshadow=0` ⇒ shadows ≈ FREE" attribution is **trivially true — there was nothing to turn
  off**. The SYSTEM paragraph of the fact pack ("resolve … CSM+clip shadows", "ProbeGI updates
  3072 probes/frame") describes the WORLD scene, not the forest test scene. Any budget line that
  assumed shadow/GI headroom in forest numbers is void; conversely, shipping shadows/GI in forest
  later will ADD cost not yet in any baseline.

**P2. The harness's bad-frame rejection is dead: `capRejects` = 30-32 of 32 on every run/pose**
(all four spot-checked JSONs). `probe-fresh-stutter.ts:211-212` falls back to using ALL frames
when >half are suspect, which is always. The `capSuspect` predicate (`MeasureHarness.ts:169`,
per-pass timestamp total vs wall) fires on ~every frame because the per-pass timestamps are the
known-unreliable cross-frame spans. Medians are still honest (gpuWall is used, not timestamps),
but no outlier filtering is happening at all.

**P3. `gpuWallMs` includes event-loop latency.** It is `tSubmit → onSubmittedWorkDone` *promise
resolution* (`MeasureHarness.ts:150-160`). A GC pause or longtask while awaiting the drain
inflates the sample. The rare 70-91 ms oblique spikes (e.g. `fresh-ablate-post3` oblique frames
4 and 15) are prime suspects for this, not GPU work.

**P4. The isolated-frame protocol itself is a live suspect for the bimodality.** Every measured
frame is preceded by a 50 ms `sleep` + full queue drain (`MeasureHarness.ts:143-150`). The M1 Max
GPU power-manages aggressively; every sample starts on a down-clocked GPU and rides the DVFS ramp.
A frequency governor with hysteresis produces exactly the observed signature: metronomic
few-frame limit cycles, amplitude largest where frames are SHORT (aerial ~10-16 ms of work swings
2.3×; eye ~30 ms swings ±10%), and band-onset mid-run as thermal state drifts
(`fresh-final-rested` oblique is STABLE for frames 0-13, then bands appear). This is a
measurement-metric issue one level up from the renderer — it must be discriminated (probe B2
below) before any renderer-side "fix" is attempted.

**P5. GPU pipelining sawtooth cannot explain the isolated bimodality.** Each phase-B frame is
drain-isolated on an empty queue by construction; there is no cross-frame overlap to saw-tooth.
(It can still shape LIVE frame deltas.)

---

## How it works today

Per-frame graph (forest defaults: `voxActive=true`, `voxf2b` OFF, `voxbocc`/`voxoccl` ON,
shadow=null, GI=null), from `NaniteFrame.render()` (`NaniteFrame.ts:425-497`):

| # | call | submit(s) | dispatches inside | notes |
|---|------|-----------|-------------------|-------|
| 1 | `cull.runPhase1` (`NaniteFrame.ts:462`, `NaniteCull.ts:1008-1010`) | 1 | `[kClearHier, kSeedRoots, (kArgs,kTraverse)×hierDepth, kRasterArgs]` ≈ 2+2·16+1 ≈ **35** | one batched compute pass; traverse kernels indirect-tagged (`Tsl.ts:283-289`) |
| 2 | `cull.syncFullArgs` (`NaniteFrame.ts:463`, `NaniteCull.ts:1013-1015`) | 1 | 1 (`kRasterArgs2`, 1 thread) | its own submit "on purpose" (`NaniteCull.ts:993-995`) — the stated reason (runs after fanout) is stale: the frame calls it BEFORE `runVoxFanout` |
| 3 | `cull.runVoxFanout` non-F2B (`NaniteCull.ts:1041-1047`) | **3** | `kVoxFanoutArgs`; `kVoxFanout` (indirect); `kVoxRasterArgs` | three separate submits |
| 4 | `raster.world1` compute (`NaniteRaster.ts:1419`) | 1 | `[kVisClear, kRasterWorld1(indirect), kHwArgs]` | clear owned by this batch |
| 5 | `hwRender` (`NaniteRaster.ts:1347-1363`) | 1 | 1 render pass (indirect draw), dead rgba8 target, clear already skipped (`hwrt` default false, `NaniteRaster.ts:304`) | render + compute can't share an encoder |
| 6 | `voxRaster.dispatchVoxel` (`NaniteVoxelRaster.ts:1454-1492`) | **3** | voxOccPyr chain (12 kernels, one batch); `kClearBins` (1 word!); `kVoxScatter` (indirect) | pyramid MUST follow hwRender (reads post-HW `visPayloadV`) |
| 7 | `raster.scar` (`NaniteFrame.ts:479`) | 0 | no-op unless `?scar=1` | |
| 8 | `hzb.build` (`NaniteFrame.ts:481`, `NaniteHzb.ts:150-155`) | 1 | 12 max-pool kernels, one batch | this frame's depth → NEXT frame's cull occluder |
| 9 | `post.render()` (`NaniteFrame.ts:494`) | ~2-3 | scene pass (resolve mesh + voxMesh) → half-res MRT (GTAO+clouds+bounce) → bilateral → TRAA → bloom; plus `post.meter`'s auto-exposure compute (runs in `Engine.renderStep` at line 157, before render) | |

Total ≈ **11-13 queue.submits, ~67 compute dispatches, 2-4 render passes per frame.**

Pyramid geometry at 2268×1473: both chains are half-res-rooted, 12 levels
(1134×737 → 1×1, ≈1.12 M texels total each). voxOccPyr = min-pool of the packed election key
(pre-vox, conservative keep-on-tie; `NaniteVoxelRaster.ts:399-474`); HZB = max-pool of decoded
depth (post-vox; `NaniteHzb.ts:97-148`). Both dispatch chains run in ONE submit each with
inter-dispatch UAV auto-sync (`Tsl.ts:241-243`).

Every-15th-frame metering: `NaniteFrame.meter` (`NaniteFrame.ts:509-519`) launches 4-5 async
readbacks (`readCounts` ×2 buffers, `readHwCount`, `readVoxCount`, `readVoxWrites`) — each a
buffer→staging copy + mapAsync. In phase B these land inside measured frames (period 15).

Camera/jitter: `cam.update(jitteredCamera())` (`NaniteFrame.ts:440`) re-derives TRAA's Halton
offset each frame so cull+raster project with the same jitter; under `?ablate=taa`
`post.traaNode` is null (`PostStack.ts:543`) and the mirror is a no-op — jitter genuinely frozen.

`cullfreeze=1` (`NaniteFrame.ts:434-445`): freezes cull, HZB build, and fanout after frame 0 but
keeps raster+vox+resolve+post running on the frozen queues — a perfect "constant-input"
discriminator, EXCEPT it freezes at the boot pose (pose changes after freeze render the stale cut).

---

## Work model

- **Dispatch/barrier overhead is TINY on this stack.** Evidence: `voxwaves=4` adds 3 extra
  pyramid rebuilds (36 dispatches), 3 extra submits, and 3 extra full barrier boundaries over
  `voxf2b=1` ctl, at eye where its occlusion gain is ~0 — measured +0.1 ms (41.0→41.1). Bound:
  **≤ ~10 µs per barrier-separated dispatch, ≤ ~100 µs per submit** (GPU side). The BFS batch's
  ~30 near-empty tail dispatches and both pyramid tails (~7 tiny levels each) are therefore
  sub-0.5 ms combined — the "serialized-bubble cost" of the pyramid chains is real but small.
- **What IS expensive is occupancy collapse under serialization** (see Mission 2): splitting a
  parallel job into K barrier-separated stages costs `Σ_k criticalPath(stage_k)` instead of
  `totalWork / GPU_width`. With few, huge workgroups per stage (aerial far-tile clusters), the
  critical path per stage ≈ one cluster's Phase-B footprint loop at <10 % occupancy.
- Pyramid chain cost model: L0 does 4 loads/texel over full-res (3.34 M loads ≈ 13 MB) + writes
  0.84 M; whole chain ≈ 22 MB traffic + 12 dispatch boundaries ⇒ ~0.2-0.4 ms per chain, two
  chains ≈ 0.4-0.8 ms/frame. Consistent with `voxoccl` never showing up as a standalone line.
- CPU encode: `cpuSubmit` med 1.4-5.5 ms across runs (JSONs) for ~12 submits + ~67 dispatch
  encodes. This overlaps GPU in live mode but is serial inside a live frame's critical path when
  the GPU finishes early.

### Mission 2 — why `voxf2b=1` costs +16 ms at aerial (and only +5/+6 at eye/oblique)

Pure overhead is ruled out by the voxwaves bound above (~34 extra tiny dispatches + 2 extra
submits ≈ ≤0.5 ms). The +16 ms is **work-serialization at collapsed occupancy**:

- Aerial vox scatter, wide-parallel (default): ≈ 16.8 − 11.1 (noleaves) ≈ **5.7 ms** as ONE
  indirect dispatch over 659 clusters × 128 lanes ≈ 84 k threads (already marginal occupancy for
  a 32-core M1 Max, but all clusters run concurrently and the wavefront is the max, not the sum).
- Under F2B the same 659 clusters are partitioned into 16 depth slabs dispatched with a full
  UAV barrier between each (`NaniteVoxelRaster.ts:1485`, all buckets touch `visPayloadV`).
  Average 41 WGs ≈ 5 k threads per stage — <10 % occupancy — and at aerial the per-cluster
  footprints are the LARGEST in the game (FarTile heads), so each stage's latency ≈ its worst
  cluster's cooperative footprint loop. 16 × ~1.3 ms ≈ 21 ms ≈ the observed 32.8 − 11.1 ≈
  21.7 ms of vox time (3.8× the wide-parallel 5.7 ms).
- Eye/oblique have 5893/9188 clusters (≈370-570 WGs per bucket) and small footprints, so each
  stage still half-fills the machine → only +5/+6 ms.
- The depth histogram makes aerial worse: looking straight down, most clusters share nearly one
  depth, so a few buckets hold almost everything while ~14 stages run near-empty but still pay
  their barrier + launch.

Orchestration law for this stack: **barriers are cheap; serialized under-filled stages are not.
Never partition a dispatch below (GPU width × a few) threads per stage.** This retro-explains the
F2B/K-pass triangle refutation (eecf046) and the voxwaves null at oblique.

Optional confirmation probe: `voxf2bk=4` at aerial — serialization predicts the penalty scales
~linearly with K (≈ +4-5 ms at K=4); fixed-overhead predicts it stays ~flat.

### Mission 1 — the bimodality: per-frame-variable work sources (complete enumeration)

Sources of frame-to-frame GPU-work variation at a STATIC pose in this codebase:

| # | source | period/shape | status |
|---|--------|--------------|--------|
| A | TRAA Halton jitter → `cam.update` wobbles cull VP sub-pixel → cut flicker (`NaniteFrame.ts:383-418`) | 32-frame Halton | **modulator, not driver** — bands persist under `ablate=taa` (fresh-ablate-post3) |
| B | **HZB occlusion feedback**: frame N's post-vox depth → frame N+1's cull (`NaniteFrame.ts:481`, `NaniteCull` reads prev pyramid). A marginal cluster flips culled↔kept; its presence/absence changes the next pyramid ⇒ limit cycles of small period. The voxbocc brick cull compounds it (reads THIS frame's mesh election, which depends on the cut). | 2-7 frames, pose-dependent | **top renderer-side suspect** — explains lows = over-cull (same mechanism as the proven aerial pose-arrival over-cull ramp: first frames 6.5-11.5 ms then settle ~17), TAA-off persistence, and presence in noleaves (oblique noleaves has 8-9 ms frames among 16-17s) |
| C | worldTime drift: wind sway (raster coverage), cloud march, gust sampling (`Engine.ts:146-147`) | smooth, seconds | refuted by shape (can't produce period-3 metronome) |
| D | every-15th-frame meter readbacks (4-5 staging copies + mapAsync inside the measured window; `NaniteFrame.ts:509`) | 15 | real contaminant (~2 frames per 32-sample run) but wrong period for the bands |
| E | auto-exposure feedback kernel (`PostStack` via `post.meter`) | every frame, constant | not variable |
| F | GI probes / shadow cadence | — | **do not exist in forest** (Premise P1) |
| G | GC/event-loop latency inside the drain await (P3) | sporadic | explains the rare 70-91 ms spikes, not bands |
| H | **DVFS ramp between drain-isolated samples** (P4) | governor hysteresis: few-frame limit cycles | **top harness-side suspect** — explains metronomic period-3 (voxbocc aerial: 6 consecutive clean high→mid→low cycles: 23.4 18.3 13.6 / 23 17.7 12 / 21.3 16.3 12.1 / 22.7 17.9 11.8 / 20.8 16.6 12.6 / 19.2 18.3 11), amplitude ∝ 1/frame-length, and mid-run band onset |

ACF fingerprints (per-frame arrays, first 3 frames trimmed): voxbocc aerial acf(1)=−0.43,
acf(3)=+0.31, acf(6)=+0.44 (period 3); final-rested oblique acf(3)=−0.58, acf(6)=+0.41
(period ~6-7, i.e. runs of 3-4); noleaves poses ~flat ACF with sporadic ~half-cost lows.

**Verdict: two viable hypotheses remain — (B) HZB-feedback cut oscillation and (H) harness DVFS
artifact — and they have opposite consequences.** If H, the bands don't exist live and only the
medians matter (all shipped A/B verdicts stand; they compared medians). If B, there are 2-8 ms
of periodic re-expansion work at oblique/aerial AND a latent correctness issue: the LOW frames
are over-culled frames (content missing for a frame, masked by TAA) — the same disease as the
pose-teleport transient. Discriminating probes below (B1-B3); both are cheap.

### What the model explains

- +16/+6/+5 F2B deltas (above), voxwaves ≈ +0.1 at eye (barrier bound), the aerial arrival ramp
  (stale-HZB over-cull = hypothesis B's mechanism in the large), and why no orchestration line
  item shows up in the pixel-scaling law's fixed terms bigger than ~1 ms (submits+pyramids+tails
  ≈ 0.6-1.3 ms of the eye-13/obl-12/aer-5 fixed cost; the rest is post + base raster).

---

## Waste inventory

| item | what | cost est. | cite |
|------|------|-----------|------|
| W1 | `visDepthV` full-res clear every frame (3.34 M atomic stores) in the packed world path where NOTHING reads depthV (HZB reads payload, resolve packed branch reads payload/visB; only legacy modes + `?nanprobe` use it) | ~0.05-0.15 ms/frame, all poses | `NaniteRaster.ts:391`, buffer alloc `NaniteRaster.ts:179` |
| W2 | 7 foldable queue.submits: syncFullArgs (1), fanout (3→1), dispatchVoxel (3→1, can also absorb hzb.build) | ~0.2-0.7 ms GPU + ~0.5-1.5 ms CPU encode/submit overhead reduction (live frames benefit most) | `NaniteCull.ts:1013,1041-1047`, `NaniteVoxelRaster.ts:1461-1490`, `NaniteFrame.ts:481` |
| W3 | `kClearBins` dispatch+submit every frame to zero ONE debug word whose consumer (`voxwrites`) is default-off | 1 submit | `NaniteVoxelRaster.ts:309-315,1489` |
| W4 | pyramid tails: 2 chains × ~7 levels below 36×24 as separate barrier-separated dispatches | ~0.1-0.3 ms/frame total | `NaniteHzb.ts:99-148`, `NaniteVoxelRaster.ts:431-474` |
| W5 | meter readbacks inside measured frames (period 15) — contaminates ~2 of 32 samples per pose | measurement hygiene | `NaniteFrame.ts:509` |
| W6 | BFS empty-tail passes (hierDepth fixed ≈16; late frontiers empty) — already one submit, indirect 0-size | ≤0.3 ms (bounded by barrier law) | `NaniteCull.ts:998-1006` |
| W7 | hwRT full-res dead rgba8 clear | already fixed (default skipped) | `NaniteRaster.ts:304,1351-1361` |

No large dead-work item exists in orchestration: the big costs are inside the raster/election
kernels (other reviewers' areas). The orchestration-layer recoverable total is ~0.5-1.2 ms GPU
per pose plus CPU-submit reduction — worth taking because it is trivially quality-identical, but
it is not the −11 ms oblique lever.

---

## Levers

### frame-orchestration:coalesce-submits — fold the 7 residual submits (S/M, IDENTICAL)
Mechanism: (a) append `kRasterArgs2` + the non-F2B fanout triple (`kVoxFanoutArgs`,
`kVoxFanout` — tag with `setIndirectDispatch(kVoxFanout, voxFanoutDispatchAttr)` like its F2B
siblings at `NaniteCull.ts:1024-1026` — `kVoxRasterArgs`) onto the BFS batch → cull+fanout = ONE
submit. (b) in `dispatchVoxel`, one `dispatchBatchMixed([...voxPyrKernels, kClearBins,
kVoxScatter])`; drop `kClearBins` entirely when `!voxWrites`. (c) pass the HZB kernel list into
the same post-hwRender submit (order: pyr → scatter → hzb; UAV auto-sync preserves RAW).
Quality: bit-identical (same kernels, same order, same barriers).
Expected: eye 0.2-0.5 ms, oblique 0.2-0.5 ms, aerial 0.2-0.4 ms isolated; larger live-frame CPU
win (cpuSubmit med 1.4-5.5 ms shrinks by the per-submit share).
Discriminator: same-session A/B, `LABEL=coalesce` vs `LABEL=coalesce-ctl`.
Risks: three r184 batching quirks (the `dispatchSize` attach hack, `Tsl.ts:274-289`); mis-ordered
RAW if the array order is wrong — array order IS execution order, keep the documented sequence.

### frame-orchestration:skip-depthv-clear — stop clearing the dead depth buffer (S, IDENTICAL)
Mechanism: build-time gate in `kVisClear` — when `packedClear` (world path) skip the
`visDepthV` store (`NaniteRaster.ts:391`); keep for legacy/probe modes.
Expected: ~0.05-0.15 ms all poses. Discriminator: A/B one flag. Risks: `?nanprobe` reads
`vis.depthV.ro` (`NaniteFrame.ts:320`) — reads stale garbage under the probe; gate the skip off
when `nanprobe=1`.

### frame-orchestration:fuse-pyramid-tails — single-workgroup multi-level reduce (M, IDENTICAL)
Mechanism: for both chains, replace levels ≥5 (≤36×24 texels) with ONE kernel: a single 256-lane
workgroup loops the remaining levels with `workgroupBarrier()` between them (all data fits; the
same-buffer rw-view law already holds). 12 dispatches → 6 per chain.
Expected: 0.1-0.3 ms/frame total (barrier-bound estimate; may be less — bounded above by the
voxwaves law). Discriminator: A/B flag `?pyrfuse=0/1`. Risks: subtle same-scope/wg-barrier
correctness; validate with the HZB viewer (`?nanitedbg=hzb`) + shot-diff.

### frame-orchestration:bimodality-diagnosis — instrument, then fix the real driver (S probe, then M/L fix)
Mechanism (probes B1-B3, exact command lines in the last section): decide between HZB-feedback
oscillation (renderer) and DVFS harness artifact (metric).
- If DVFS (H): change the METRIC — report per-pose medians from back-to-back-burst sampling
  (cooldown only between BURSTS), and re-baseline. No renderer change. Quality: n/a.
- If feedback (B): the fix is the two-pass occlusion already scaffolded in NaniteCull (test
  against THIS frame's HZB in a re-test phase, UE5-style). That kills the oscillation AND the
  pose-teleport over-cull transient AND the masked one-frame disocclusion holes (quality
  IMPROVES). Effort L (it re-orders the frame: cull→raster→hzb→re-cull→raster-2).
Expected IF B is real: oblique −2-4 ms median (the 43-47 band collapses onto 34-39),
aerial −3-5 ms; eye ~0. Expected if H: 0 ms renderer-side, but every future A/B gets ~3× tighter.
Risks: two-pass occlusion touches the hot cull; must keep the conservative direction.

### frame-orchestration:probe-hygiene — meter-silence + counters during phase B (S, measurement-only)
Mechanism: `?meterevery=0` (or a probe-set hook) to suppress the 15-frame readback block during
`measureFrames`; optionally per-frame `readCounts` between drained frames (free of perturbation
there) so per-frame `visClusters/voxClusters` land in each `MeasuredFrame.counters` — this is
also what B1 needs. Expected: cleaner tails; 2/32 samples uncontaminated. Risks: none.

### REJECTED-BY-POLICY (quality-trading; listed for completeness only)
- None orchestration-specific. (`voxf2b`/`voxwaves` are already default-off perf-losers, not
  quality trades; `aggdist`/`voxlodk`/`leafcheap`/dpr knobs belong to other areas and are
  attribution instruments per the user ruling.)

---

## What UE5/prior art does here

- **Two-pass occlusion (main + post pass)**: UE5 Nanite culls against LAST frame's HZB, rasters,
  rebuilds HZB, then RE-TESTS the previously-culled set against THIS frame's HZB and rasters the
  disoccluded remainder. No cross-frame limit cycle is possible because the final visibility each
  frame is consistent with that frame's own depth; the stale-HZB transient we see at pose
  teleport (and hypothesis B's oscillation) is exactly what this removes. Our cull has the
  scaffolding (backlog §4.8).
- **Persistent-threads culling**: UE5's cluster cull is one persistent dispatch with a GPU work
  queue, not a CPU-fixed 16-deep ping-pong ladder — no empty-tail passes, no per-level barrier.
  Our batched BFS is a reasonable WebGPU approximation (WebGPU lacks forward progress
  guarantees); the tail cost is bounded small here.
- **One command list, few submits**: engine-side, an entire Nanite frame encodes into a handful
  of command lists with split barriers; per-submit overhead is treated as a scarcity. Our 11-13
  submits/frame is high by that standard; the coalesce lever closes most of it.
- **HZB build**: UE builds ONE mip chain per frame with a fused multi-mip compute (SPD-style —
  AMD FidelityFX Single Pass Downsampler does the whole pyramid in ONE dispatch with workgroup
  and global-atomic coordination). We build TWO 12-dispatch chains; an SPD-style fused build is
  the end-state of the pyramid-tail lever if it ever matters (>0.5 ms).
- **No fine-grained depth-bucket serialization**: prior art (CudaRaster, Nanite SW raster) keeps
  primitives provably small and lets one wide dispatch + atomics resolve order, exactly the
  conclusion our F2B/waves data re-derived (barriers cheap, under-filled serialized stages ruinous).

---

## Open questions + proposed serial probes

All `TICKS=0 COOLDOWN_S=45 TREES=200000` unless stated; run AFTER the day's baselines per thermal
discipline. Items marked [patch] need the small src/tools edits described (do not edit while a
probe is in flight).

**B1 — feedback-vs-DVFS discriminator #1 (per-frame counters).** [patch: probe hygiene lever —
per-frame `readCounts` in phase B + `?meterevery=0`]
`CONFIG=default LABEL=bimodal-counters TICKS=0 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts`
→ if per-frame `visClusters`/`voxClusters` oscillate in lockstep with the gpu bands ⇒ feedback
(B). If counters are flat while gpu bands persist ⇒ DVFS (H).

**B2 — feedback-vs-DVFS discriminator #2 (cooldown sweep, one boot).** [patch: pass `cooldownMs`
through `measureFrames` in the probe; measure oblique 4× in one page session with
cooldownMs 0/10/50/150]
`CONFIG=default LABEL=bimodal-cooldown npx tsx tools/probe-bimodal-cooldown.ts` (new ~30-line
probe reusing probe-fresh-stutter's boot). Band amplitude scaling with cooldown ⇒ H. Note
cooldown 0 conflates thermal throttle (MeasureHarness.ts:73-76) — compare 10 vs 50 vs 150 first.

**B3 — jitter re-check at the period-3 pose (cheap, no patch).**
`CONFIG=default EXTRA=ablate=taa LABEL=bimodal-aerial-notaa npx tsx tools/probe-fresh-stutter.ts`
→ does voxbocc-aerial's clean period-3 persist with jitter frozen? (ablate-post3 was pre-voxbocc.)

**B4 — constant-input control (cull frozen).** [patch: expose a runtime
`__laasNanite.freezeCull()` hook so the freeze can engage AFTER the pose is set, or accept
boot-pose-only measurement]
`CONFIG=default EXTRA=cullfreeze=1 LABEL=bimodal-frozen …` at the BOOT pose only → raster+vox+
post with a frozen cut: bands gone ⇒ the variable work is cull-driven (B); bands persist ⇒ H
(or resolve/post content drift, which C's shape rules out).

**M2 — F2B serialization confirmation (optional, closes Mission 2 empirically).**
`CONFIG=default EXTRA=voxf2b=1,voxf2bk=4 LABEL=f2bk4 …` → aerial penalty ≈ +4-5 ms (linear in K)
confirms serialization; ~+16 flat would mean fixed overhead (contradicting the voxwaves bound).

**C1 — coalesce A/B (after building the lever).**
`CONFIG=default LABEL=coalesce-ctl …` then `CONFIG=default EXTRA=coalesce=1 LABEL=coalesce …`
(ship behind `?coalesce` first; flip default after the gate).

**Open questions**
1. Is the LIVE frame stream bimodal at a static pose (rAF, no harness)? A 300-tick TICKS-run at
   a STATIC pose (zero motion) would show whether the bands exist outside the harness at all.
2. Where do the 70-91 ms isolated spikes come from — GC in the drain await (P3) or a real GPU
   hiccup? Correlate with `longtasks` (already captured in phase A but not phase B).
3. `cpuSubmit` medians differ 1.4 → 5.5 ms across same-code runs (voxbocc vs bead-v2-base) —
   thermal CPU or code drift between runs? Worth one paired re-measure before trusting any
   CPU-side conclusions.
4. When shadows/GI ship in forest (P1), the whole budget moves — re-baseline then.
