# vox-raster deep review (2026-07-02)

Area: the voxel scatter/election kernels — `src/nanite/NaniteVoxelRaster.ts` (kernel half),
`src/nanite/VoxelBrick.ts`. Post-voxbocc baseline: eye 18.9 / oblique 37.2 / aerial 16.5
(isolated gpuWall med, 200k @ 2268×1473). Oblique is the money pose (needs ≈ −11..13 ms).

## Premise audit

1. **"Vox owns most of oblique foliage (~22 ms)" — CONFIRMED with a sharper shape.**
   Foliage at oblique = 37.2 − 15.4 (noleaves) = 21.8 ms. A linear per-CLUSTER model fits
   every brick-count experiment (all pre-voxbocc, vs bead-v2-base, same-session):

   | run | pose | Δ voxClusters | Δ ms | µs / cluster |
   |---|---|---|---|---|
   | aggdist=60 | eye | −3586 | −10.0 | 2.79 |
   | aggdist=60 | oblique | −4759 | −10.7 | 2.25 |
   | voxlodk=0.7 | eye | −2523 | −3.4 | 1.35 |
   | voxlodk=0.7 | oblique | −4283 | −6.8 | 1.59 |
   | voxlodk=0.85 | oblique | −2233 | −2.1 | 0.94 |

   Slope ≈ 1.6–2.2 µs per emitted vox cluster (≤128-brick workgroup). Predicted oblique vox
   = 9188 × ~1.8 µs ≈ 16 ms + coverage ≈ 20 ms ≈ observed. (JSON: scratchpad fresh-*.json.)

2. **THE central premise finding: the kernel is COUNT-bound, not FILL-bound.** Aerial is
   100% vox-covered screen (shot: voxbocc-aerial.png) at 659 clusters → foliage share
   16.5 − 11.1 = 5.4 ms. Oblique is ~90% covered at 9188 clusters → ~20 ms. Same coverage,
   14× the clusters, ~4× the cost. So "elections for visible bricks (irreducible coverage)"
   ≈ **4–5 ms** at oblique (≈1.2–1.6 ns/px incl. election + ray/DDA), and **~14–16 ms is
   count-scaled overhead** that never reaches a new visible pixel. The fact pack's framing
   "decompose visible-election vs waste" is right, and the answer is lopsided: ~75–80% is
   per-brick/per-cluster work, not per-pixel work.

3. **One level up, named explicitly:** what generates 9188 oblique clusters vs 659 aerial
   clusters at equal coverage is the **45–140 m per-TREE ring** (aggdist=60 diagnostic:
   ring = ~4800 clusters = ~10.7 ms at oblique). Kernel-level cuts below total ~3–6 ms;
   they cannot close −11..13 alone. The structural, quality-PARITY candidate (tiered
   32 m fine-cell ring tiles, 2026-07-02 log §5c) lives in the aggregation area, not here.
   This review still matters: per-brick overhead multiplies whatever emission survives.

4. **Metric checks.** Same-session ordered ratios used throughout; voxClusters counter is
   stable across runs at fixed pose (9188/9186/9185) so cluster deltas are real. The
   oblique bimodality (runs of ~4 frames, ±4 ms) makes single medians ±2 ms soft; all
   deltas quoted are ≥2× that. voxOccPyr build is constant per-frame work (14 fixed
   dispatches) — it is NOT a plausible source of the run-of-4 periodicity.

5. **Fact-pack question "is the K=16 histogram built when f2b is off?" — NO.**
   `runVoxFanout` (NaniteCull.ts:1039-1048) dispatches the histogram chain
   (kVoxRange/kVoxCount/kVoxPrefix/kVoxBucketArgs/kVoxScatterFan) only under `?voxf2b=1`;
   the default path is the plain append fanout (kVoxFanoutArgs + kVoxFanout). The K kernel
   instances in the raster are also build-gated (`if (voxF2bEnabled)`,
   NaniteVoxelRaster.ts:1438). No dead histogram work in the default frame.

## How it works today

One compute workgroup (128 threads = `WG_RASTER` = MAX_BRICKS_PER_CLUSTER) per qVoxRaster
item, indirect-dispatched over the fanned count (NaniteVoxelRaster.ts:1419, 1490).

**Per-cluster prologue** (all lanes, uniform): fetch item + instance A/B + cluster words
(NaniteVoxelRaster.ts:551-583); thread-0 per-BLOCK occlusion test against voxOccPyr
(sphere → mip level → 2×2 min-pooled window, keep-on-tie; :614-665) broadcast via
`wgVisible` + barrier.

**voxOccPyr**: min-pooled pyramid over THIS frame's visPayloadV (mesh winners only — built
before the vox dispatch, :1461), 14-16 levels, one dispatch each (:429-474). Key polarity:
NEARER = LARGER key, min over window = farthest/most-see-through pixel ⇒ conservative,
empty texel (0) always keeps.

**Phase A (1 lane = 1 brick, :675-1063):**
- seed empty records (:679-689);
- decode brick words 5-8 (posXYZ + half) (:695-700), world center via `instTransformPoint`,
  "world AABB" half-extent via `instSphereRadius` (:701-702);
- project the 8 corners of [center ± brWR] with near-plane/NDC-explode straddle detection
  (:737-772); non-straddler → raw clamped bbox + 128 px span cap (:803-812); straddler →
  BRICK_MAX_EXT centre box (:813-836);
- **?voxbocc** (default ON): per-brick front-slab key vs voxOccPyr at the bbox-covering
  level, straddlers exempt (:842-884);
- surviving brick: store bbox + precomputed election key `cand` = depthKey24(frontSlab)<<8
  | id8 (:891-902); ?voxcell records (local AABB + occ words + non-straddle flag,
  :908-920); **occupancy-mask build** (?voxlod): for a coarse (dagLevel>0), armed
  (area ≥ 16 px²), sparse (popcount ≤ 48) brick, loop 64 cells × project 8 corners each →
  4×4 screen-bucket mask (:934-1059).

**Phase B (all lanes, cooperative, :1066-1413):** serial loop over the cluster's bricks;
per brick: read the shared record (~15 loads), 1/bbW + 1/bbH reciprocals (:1161-1162),
voxcell eligibility = non-straddler AND area > ?voxcellmin(64 px²) (:1167-1171); then
stride the brick's own footprint [0, bbW·bbH) across the 128 lanes (:1183).
Per pixel:
- addressing via per-brick reciprocal (loss-exact, ?voxrecip, :1187-1190);
- **ray path** (eligible bricks): relaxed-load early-out `cand > prevE` (:1266-1267), then
  linear-in-NDC local ray (6 FMA, per-cluster bases :1098-1136), slab test vs the local
  cube (:1288-1298), ≤6-step DDA through the 4³ occupancy mask with
  exhaustion-=conservative-hit (:1332-1381), exact per-pixel depth key, guarded
  atomicMax + winner atomicStore of visBV (:1394-1403);
- **flat path** (small/straddler bricks): optional 4×4 occ-mask bucket test (:1240-1252),
  then the world1-verbatim election: relaxed load → gate → atomicMax → winner store
  (:1196-1215).

**Election guard**: yes, a read-before-atomic guard exists at BOTH sites (:1202-1204 flat,
:1266 + :1398 ray) — losing fragments cost one load + compare, no RMW. This is already the
right pattern; the atomic-contention question stays refuted.

**Dispatch** (:1454-1492): default = pyramid chain → kClearBins → ONE indirect scatter.
?voxf2b=1 = K per-bucket kernels near→far in one submit; ?voxwaves=N adds pyramid rebuilds
between waves.

## Work model

Per frame, per pose:

```
voxMs ≈ Nclusters · Cclust                       (block test, bases, Phase-B loop scaffold)
      + Nbricks   · Cbrick                       (Phase A walk + bocc + record + [mask build])
      + Σ_bricks ceil(bboxArea/128)·roundCost    (Phase B rounds; ragged-wave utilization)
      + VisiblePx · Cfill                        (~1.2-1.6 ns/px: guard+ray/DDA+atomic)
      + BuriedPx  · Cguard                       (1 load + cmp when already seeded; FULL ray
                                                  cost when the unordered race runs early)
      + PyrBuild                                  (~0.1-0.3 ms, constant)
```

Measured anchors it explains:
- slope 1.6–2.2 µs/cluster (table above) = the Cclust + avgBricks·Cbrick + bbox-round terms;
- aerial 5.4 ms @ full coverage + 659 clusters ⇒ Cfill·3.34Mpx ≈ 4 ms ⇒ per-pixel is CHEAP;
- voxbocc −17.2 eye / −6.0 oblique: eye's vox field is mesh-hidden (bocc zeroes its
  Phase-B rounds); oblique's canopy IS the vox front surface, little for a mesh-only
  pyramid to kill ⇒ the residual oblique cost is vox-vs-vox + per-brick fixed work;
- voxf2b K16 +6 oblique: 16 barrier-serialized sub-dispatches starve occupancy — ordering
  has value only if bought cheaply (K=2, see levers);
- naive ALU accounting under-predicts the measured slope ~3-5× ⇒ occupancy/latency/
  divergence multiplier on this monolith kernel (open question #3).

## Waste inventory

Ranked, oblique-centric, best-effort quantified:

1. **Wasted occupancy-mask builds (Phase A), ~1–3 ms oblique.** The mask is consumed ONLY
   by the flat path (:1240-1252). Ray-eligible bricks (non-straddler AND area > 64 px²)
   never read it — but the build gate (:938) arms at area ≥ 16 px² with no upper bound, so
   every coarse sparse brick bigger than 64 px² (the τcap=12px far field projects ~144 px²)
   pays up to 64 cells × 8 corner-projections on one divergent lane, then the result is
   dead. Divergence-amplified: one armed lane serializes its 32-lane wave.
2. **Buried-brick full-cost elections from unordered dispatch, ~1–3 ms oblique.** The
   early-out only helps once a nearer key is already written; in an unordered dispatch the
   expected number of "temporary winner" full-cost passes per pixel grows ~ln(stack
   depth). Oblique stacks crowns 3-8 deep; vox-vs-vox is invisible to the mesh-only
   pyramid. (Waves at K16 refuted the EXPENSIVE fix, not the phenomenon.)
3. **Phase-B per-brick setup issued 4× per brick, ~1–2 ms oblique.** The ~15 shared reads
   + 2 fp divides + eligibility logic (:1142-1181) are uniform work executed by all four
   SIMD groups of the workgroup, for every brick — including zero-area (bocc-culled)
   records, which still pay full setup before their zero-trip inner loop.
4. **bbox-vs-silhouette miss pixels on the ray path, ~0.5–1 ms.** The footprint is the
   projected-AABB rect; the rotated-cube silhouette fills ~55-75% of it; miss pixels pay
   ray setup + slab (~25-40 ops). Per-pixel-scaled, so bounded by the aerial anchor.
5. **Phase-B runs the whole per-brick loop even when the BLOCK was culled** (`wgVisible`
   is only consulted in Phase A, :690): a block-culled cluster still loops nBricks × setup
   ×4 groups. Small (block-cull rate at oblique is low) but free to remove.
6. **Micro:** occLo/occHi fetched twice in Phase A (:913-914 and :939-940); per-cluster
   ray/clip bases (:1098-1136) computed by all 4 groups; the DDA's fixed-6 loop keeps
   issuing ~5 ops/iter after `done`.
7. **Pyramid chain**: 14 serialized dispatches every frame, ~0.1–0.3 ms + submit bubbles;
   runs even at aerial (659 clusters). Not worth touching until the big items land.
8. **Straddle path**: costs nothing at canonical poses (voxels start ≥45 m; no camera-in-
   brick). Correct to leave alone.
9. **Correctness nit (not perf):** the "world AABB" half-extent `brWR = instSphereRadius
   (A,B,brHalf,0)` (:702, NaniteCommon.ts:164-167) covers scale+shear but NOT yaw: a
   yaw-rotated cube's world AABB needs h·(|cosθ|+|sinθ|) ≤ h√2 in x/z. At yaw≈45° the
   projected bbox under-covers the true silhouette by up to ~29% linear; the ray path is
   exact only INSIDE the bbox, so brick edges can be clipped (hidden by neighbours in
   practice — and it currently SAVES work). Flag for the quality owner; fixing it costs
   perf, so it should ride along with a lever that pays for it.

## Levers

All quality classes per the ABSOLUTE constraint (identical = conservative-cull correctness
or sub-noise numeric only). Expected ms are savings (eye/oblique/aerial).

### L1 — Skip the wasted occ-mask build for ray-eligible bricks
- **Mechanism:** add to the arm gate (:938) the complement of ray eligibility:
  build only when `straddles==1 OR area ≤ voxCellMinArea OR !voxCell`. Bit-exact output:
  the mask is unread on the ray path.
- **Quality:** identical (provably — dead value elided).
- **Expected:** 0.2 / 1.5 / 0.1 (mask build is per-brick, divergence-amplified; share
  unmeasured — bounded by the L6 attribution probe).
- **Discriminator:** implement behind `?voxmaskray=0` (old behaviour); A/B
  `CONFIG=default EXTRA=voxmaskray=0 LABEL=dr13-maskctl TICKS=0 COOLDOWN_S=45` vs default,
  plus shotdiff (must be pixel-identical).
- **Effort:** S. **Risks:** none beyond the gate condition matching :1167-1171 exactly
  (strict `>` on voxCellMinArea, straddler exemption).

### L2 — Cheap front-to-back: F2B at K=2 (±voxwaves=2) — probe first, zero code
- **Mechanism:** the buried-brick waste (#2) needs only COARSE ordering to collapse: near
  half first seeds the guards, far half then fails at 1 load/px. K16's +6 ms was barrier
  cost, K scales it (single-tree: 25.4 K16 → 10.3 K1); K=2 pays one barrier. voxwaves=2
  additionally rebuilds the pyramid once so the far half's PER-BRICK bocc sees the near
  canopy (brick-granular vox-behind-vox — exactly what voxbocc proved potent vs mesh).
  All flags exist already (`?voxf2b=1&voxf2bk=2[&voxwaves=2]`).
- **Quality:** identical (atomicMax is order-free; bocc conservative).
- **Expected:** 0 / 1.0 / 1.0 — wide error bars (could be net 0; aerial showed the only
  wave gain, −4.7 at K16-cost, so K2 may net it).
- **Discriminator:** `CONFIG=default EXTRA=voxf2b=1,voxf2bk=2 LABEL=dr13-f2bk2 TICKS=0
  COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts`, then
  `EXTRA=voxf2b=1,voxf2bk=2,voxwaves=2 LABEL=dr13-k2w2`.
- **Effort:** S (probe only; K=2 is a param). **Risks:** re-tests a refuted-at-K16 verdict
  — justified by NEW evidence (the K16 measurement conflated mechanism gain with barrier
  cost; §5b of the 2026-07-02 log shows gain existed at aerial even under K16 cost).

### L3 — Phase-B: subgroup-per-brick distribution + live-brick compaction + wgVisible gate
- **Mechanism:** (a) wrap Phase B in `If(wgVisible==1)`; (b) Phase A appends surviving
  brick indices (bbW>0) to a compacted shared list via a workgroup atomic counter; (c)
  Phase B assigns bricks to SIMD groups (`b = sgId; b += 4`) with a 32-lane inner stride,
  so per-brick setup issues once (not ×4) and culled/empty bricks are never visited.
  Elections unchanged — only work distribution.
- **Quality:** identical (same set of (pixel, key, id) elections; unordered anyway).
- **Expected:** 0.3 / 1.5 / 0.1.
- **Discriminator:** flag `?voxsgb=0` reverts; A/B at all poses + shotdiff. Watch eye too:
  bocc-culled bricks (the eye majority) currently pay full setup ×4.
- **Effort:** M. **Risks:** TSL workgroup-atomic ergonomics; subtle: inner stride must
  stay ≤ area bound (same loopU pattern); barrier placement already correct (:1064).

### L4 — Linear-clip corner projection (Phase A ALU cut, also feeds the mask build)
- **Mechanism:** clip(corner) = clipCenter ± h·Mx ± h·My ± h·Mz where M* = VP·(instance
  linear) columns, computed once per cluster. Replaces 8×(instTransformPoint + mat4·vec4)
  per brick with 1 transform + ~24 vector adds; the surviving mask builds' per-cell corner
  projections (:997-1019) linearize the same way (512 transforms → ~64 adds). Expand the
  resulting bbox by +1 px to keep it conservative under fp reassociation.
- **Quality:** identical (conservative bbox; ray path exact inside it; ≤1 px larger rect
  for flat bricks is the same class as today's ceil/floor conservatism — verify by
  shotdiff; if any flip is visible, ship it for RAY-path bricks only, where it is provable).
- **Expected:** 0.2 / 0.7 / 0.05.
- **Discriminator:** flag `?voxlinproj=0`; A/B + shotdiff.
- **Effort:** M. **Risks:** the yaw/shear linearization must match instTransformPoint
  exactly (it is linear+translate, so it does); the +1 px guard must not regress the flat
  path visually (it can only add covered-at-front-slab pixels on cube edges).

### L5 — Early sphere pre-bocc (before the corner walk)
- **Mechanism:** test the brick's bounding sphere (center + √3·brWR) against voxOccPyr
  BEFORE projecting 8 corners; occluded bricks skip the walk entirely. Same conservative
  idiom as the block test. The existing exact-bbox bocc stays for survivors.
- **Quality:** identical (conservative-only drops).
- **Expected:** 0.4 / 0.3 / 0.0 — eye-leaning (that's where bricks are mesh-hidden).
- **Effort:** S/M. **Risks:** sphere radius must over-bound the yaw issue (√3·h·scale·
  (1+lean) does); double pyramid reads for visible bricks (4 extra loads) — net could be
  ~0 at oblique; probe decides.

### Micro (fold into whichever lever touches the lines)
Dedupe the occLo/occHi fetch; compute ray/clip bases on group 0 + broadcast via shared
(or accept — it's per-cluster); break the DDA loop via loop-condition on `done` if TSL
allows. Combined ≲0.3 ms.

### REJECTED-BY-POLICY (quality-trading; listed for completeness only)
- `voxlodk<1` (−6.8 obl at 0.7) — brick coarsening, user-rejected look.
- `aggdist<140` (−10.7 obl at 60) — ring aggregation at coarse cells, user-rejected look.
- `voxcellmin` ↑ (fewer ray-path bricks → big bricks paint solid rects) and
  `voxcellmin` ↓ (was +6/+17/+19 — perf-rejected).
- `voxtaucap` ↑ / `K_FLOOR` ↓ — bigger far bricks; violates "crowns must read as trees".
- `voxdither=1` — see-through stipple kills occlusion (+18.7 historic).

## What UE5/prior art does here

UE5 Nanite proper keeps every SW-rasterized primitive provably small and uses ONE 64-bit
atomic (depth|payload) per pixel — no side-buffer race, no read-guard needed; we emulate
with guard + 32-bit max + winner store, which is the right 32-bit idiom. For foliage UE
increasingly bypasses micro-triangles entirely (Nanite Voxels/"Nanite Foliage",
UE 5.5-5.6 previews): bricks of voxels ray-marched per pixel in cluster order, with
**front-to-back cluster ordering + hierarchical occupancy DAG** so buried voxels are never
visited — i.e., their answer to our #2 waste is ordering baked into the traversal, not
barriers (they exploit HW 64-bit atomics + persistent-thread queues we don't have).
CudaRaster/Laine-Karras is the ancestor of our Phase A/B split (setup vs cooperative
fill); its bin/tile stages were already refuted here (SCAR §6.0). The K=2 wave idea is the
poor-WebGPU's version of their ordered traversal. The prior-art agent should dig into
Nanite Foliage's brick sizing (their bricks are ~4³ mip bricks like ours) and how they
amortize per-brick setup (persistent threads vs one-workgroup-per-cluster).

## Open questions + proposed serial probes

1. **Phase A vs Phase B split** (sizes L1/L3/L4 before building):
   `CONFIG=default EXTRA=voxrdbg=2 LABEL=dr13-phaseA TICKS=0 COOLDOWN_S=45 TREES=200000
   npx tsx tools/probe-fresh-stutter.ts` — (same-session default) − (this) = Phase-B share;
   (this) − (vox disabled via noleaves-style flag) ≈ Phase A + mask build + pyramid.
2. **L2 probes** (zero code): `EXTRA=voxf2b=1,voxf2bk=2 LABEL=dr13-f2bk2`, then
   `EXTRA=voxf2b=1,voxf2bk=2,voxwaves=2 LABEL=dr13-k2w2`, both TICKS=0 COOLDOWN_S=45.
3. **Occlusion-machinery accounting post-voxbocc:** `EXTRA=voxbocc=0 LABEL=dr13-nobocc`
   (re-pin the −6 oblique same-session) and `EXTRA=voxoccl=0 LABEL=dr13-nooccl` (pyramid +
   block + brick tests all off — measures the pyramid's gross cost vs its cull win).
4. **Bricks-per-cluster counter** (converts µs/cluster → ns/brick, sizes L3): add
   `nanite.voxBricks` (sum word7&0xff during fanout) to the HUD counters — S, CPU-only
   change, no GPU cost when HUD off.
5. **Occupancy/register pressure of the monolith** (the 3-5× model-vs-measured gap):
   WebGPU-Inspector capture + Metal shader profiler on nanVoxScatter (follow-up tooling);
   cheap in-repo discriminator: a build-variant that compiles with voxcell path removed
   (`?voxcell=0`, quality-changing, DIAGNOSTIC ONLY) at a fixed pose — if per-cluster
   slope collapses, register pressure from the ray path is the multiplier and a
   two-kernel split (flat-only / ray-only over a compacted item list) is the L-effort fix.
6. **Post-voxbocc slope re-check:** `EXTRA=voxlodk=0.7 LABEL=dr13-slope-postbocc`
   (DIAGNOSTIC, look-rejected flag used as instrument) — re-fit µs/cluster with bocc in
   place so lever forecasts use the current-regime slope.

Bottom line: no single kernel-level lever closes the oblique gap. L1+L2+L3(+L4/L5)
realistically total **3–6 ms at oblique**, all quality-identical. The remaining 6–9 ms at
oblique is generated one level up — the 45–140 m per-tree ring's cluster count (needs the
quality-parity tiered-tile design) — plus the ~5–6 ms post stack owned elsewhere.

## Reconciliation & verification (2026-07-02, post-limit continuation)

The adversarial verify pass died mid-run; this section is the completed RECONCILE+VERIFY
for the vox-raster area. Sibling doc: `05-voxel-raster-runtime.md` (fleet A, same area,
different brief). Method: every load-bearing file:line cite re-read in code
(NaniteVoxelRaster.ts, VoxelBrick.ts, NaniteCull.ts, NaniteCommon.ts, NaniteRaster.ts,
NaniteFrame.ts, Tsl.ts, the vendored UE5 shaders under `docs/perf-runs/Nanite-UE5-shaders/`);
every ≥1 ms measured claim recomputed from the scratchpad `fresh-*.json` medians. Premise
audit fired before every kill below (metric = same-session isolated gpuWall medians;
counter = `nanite.voxClusters` = fanned BLOCK count, NaniteFrame.ts:524 — both sound).

### Verified (mechanism lens) — the load-bearing skeleton is CORRECT

- **The µs/cluster slope table reproduces EXACTLY** from the JSONs vs fresh-bead-v2-base
  (36.0/43.1/16.8): aggdist60 32.3 obl, Δclusters −4759, −10.8 ms → 2.27 µs; voxlodk07
  36.3 obl, −4283, −6.8 → 1.59 µs; voxlodk085 41.1 obl, −2233, −2.0 → 0.90 µs; eye
  aggdist60 −3586/−9.9 → 2.76 µs. COUNT-bound premise **CONFIRMED**.
- **voxbocc deltas confirmed**: fresh-voxbocc 18.8/37.2/16.4 vs base ⇒ −17.2 eye / −5.9
  oblique / −0.4 aerial. **f2b-ctl** 40.9/49.3/32.8 ⇒ +4.9/+6.2/+16.0. **waves4**
  41.1/48.8/28.0 vs f2b-ctl ⇒ +0.2/−0.5/−4.8 (aerial wave gain real). noleaves-now
  16.8/15.2/11.1 (oblique med reproduces as 15.2, not 15.4 — foliage 22.0 ms, immaterial).
- **Doc 05's W2 granularity argument VERIFIED against the JSONs**: fresh-voxwaves4's
  `extra` has NO voxbocc ⇒ only the BLOCK-level test (:614-665) ever saw the rebuilt
  pyramid; the PER-BRICK bocc (:843-884) did not exist in that run. Brick-granular
  vox-behind-vox has genuinely never been measured — the waves-null does not transfer.
- Code cites all check out: kernel factory + WG_RASTER=128 (:480, :488, VoxelBrick.ts:81);
  block test :614-665 (keep-on-tie |0xff, cull iff bNearKey ≤ occK); corners :737-772;
  non-straddler bbox+cap :803-812; straddler box :813-836 (up to 129×129); voxbocc
  :842-884 (straddler-exempt, mask build runs only for bocc SURVIVORS — :885 wraps it);
  record store :891-902; voxcell records :908-920 (occLo/occHi double-fetch :913-914 vs
  :939-940 confirmed); mask arm gate :938 (area ≥ 16, dagLevel>0, popcount ≤ 48 at :951,
  **no upper area bound** — confirmed); Phase-B setup :1142-1181; strict
  `area > voxCellMinArea` (default 64) at :1170; mask consumed ONLY in flatPath
  :1240-1252 — **the ray path never reads it: L1's dead-work claim is PROVEN**; ray
  early-out :1266-1267; DDA :1332-1381 (fixed-6 loop keeps issuing after `done` —
  confirmed); guarded elections :1202-1204 and :1398-1402 (world1-verbatim,
  NaniteRaster.ts:963-967); dispatch :1454-1492; F2B build gate :1438; histogram chain
  only under ?voxf2b (NaniteCull.ts:1030-1048); K clamp [1,32] default 16
  (NaniteCull.ts:327-328); depthKey24 NEARER=LARGER (NaniteRaster.ts:335-336); vox after
  hwRender (NaniteRaster.ts:1423); Tsl batch helpers = ONE submit each (Tsl.ts:241-243,
  :283-288, :304-306). UE5 cites verified in the vendored shaders (MipLevelForRect
  firstbithigh at NaniteHZBCull.ush:54, 4×4 gather :144-158; EarlyDepthTest
  NaniteWritePixel.ush:48-66; two-pass culling NaniteCulling.ush:10-11).
- **The #9 yaw correctness nit is REAL**: `instSphereRadius` (NaniteCommon.ts:164-167)
  = rLocal·scale·(1+|leanX|+|leanZ|) — no yaw term — and :743-747 projects the
  WORLD-axis box [center ± brWR]. A yaw-rotated cube's world AABB needs up to √2·h in
  x/z ⇒ the footprint bbox under-covers up to ~29 % linear at yaw≈45°. Second-order leak:
  the voxbocc front-slab key (:845-846) comes from the same under-sized box, so a brick's
  true nearest point can be nearer than bKey ⇒ a (tiny, depth-scale) conservativeness
  leak in the brick cull too. Stays a ride-along fix (see L4 restatement).

### Corrections (killed or corrected claims)

1. **KILLED (this doc): "voxOccPyr = 14-16 levels / 14 fixed dispatches."** At
   2268×1473 the half-res chain is 1134×737 → 1×1 = **12 levels = 12 dispatches**
   (:399-413 computes it; VOX_PYR_LEVELS=16 is only the uniform-table cap). Doc 05's 12
   is right (~1.12 M texels ≈ 4.5 MB — arithmetic checked). The "constant per-frame
   work, not the periodicity source" conclusion stands unchanged.
2. **KILLED (doc 05 W6): "~10 ms over ~8.5k ring blocks ≈ ~1.2 µs per block."** The
   oblique ring is 9188−4429 = **4,759 blocks at ~2.3 µs each** (8.5k mixes the eye and
   oblique removals). This doc's slope table is the correct one. W6's conclusion (cost
   scales with block count, not final pixels) survives — the unit cost just doubles.
3. **CORRECTED (doc 05 W3): "up to 4× the needed area pooled"** understates its own
   mechanism: the centre-anchored window edge = 2^(ℓ+2) px ∈ [2×, 4×) of the footprint
   diameter per axis ⇒ **4–16× the area** of a rect-aligned minimal cover. Direction and
   lever (exact-rect mip pick) unchanged; magnitude still needs the P3 cull counter.
4. **KILLED-AS-WRITTEN (both docs' L4 formula): "M\* = VP·(instance linear) columns" is
   NOT a refactor of today's math.** Today projects the world-axis box [center ± brWR]
   (:743-747), not the local cube; the docs' formula projects the true rotated-cube
   corners — a DIFFERENT (truer) shape. Restated as two variants below (L4a/L4b) with
   honest quality classes; the original "identical (provably)" phrasing was wrong for
   the formula as given.
5. **KILLED (this doc, open probe #4): "nanite.voxBricks — CPU-only change, no GPU
   cost."** Summing word7&0xff during fanout requires a GPU counter (or a full-buffer
   readback). Subsumed by doc 05's P3 `?voxstats` workgroup-aggregated counter design,
   which is the correct shape (one atomicAdd per WORKGROUP — the per-event global
   atomicAdd is the measured close-up-cliff pathology, :236-241).
6. **Minor**: cluster counts 9188 vs 9185/9187 across docs are both right (different
   runs; counter stable ±3 at fixed pose — re-verified). Aerial 659 (bead-v2-base) vs
   661 (voxbocc): same. Phase-B per-brick shared reads: counted from code = **~13**
   (5 bbox/cand + 1 mask + 7 voxcell) — "5-12" (05) and "~15" (13) both roughly ok.

### Contradictions resolved

- **L2(here) vs L1(05) — same lever, different forecasts** (~1 ms vs −2..−5 oblique).
  Resolution: gross gain = buried-brick fraction of the ~10 ms oblique ring pool
  (UNMEASURED — that's exactly what the probe determines), minus chain cost ≈ one
  barrier + one pyramid rebuild ≈ 0.5–1.0 ms (K16's +6.2 obl ≈ 15 barriers ⇒ ~0.4
  ms/barrier forest-scale; rebuild ~0.3). Reconciled expectation: **oblique −0..−4,
  aerial −0..−3, eye ~0**, wide bars; P1's decision rule (05 §6) governs. Zero code.
- **Straddler path: "leave alone" (here #8) vs L6 lever (05).** Both true: ~0 at
  canonical poses (vox starts ≥45 m — verified no straddlers can arm), but a live
  camera clipping a crown paints up to 129×129 px per brick ⇒ real live-p95 hazard.
  Kept as a deprioritized RISK lever (stress-pose gate), NOT in the canonical sums.
- **Phase-B rebalancing: L3(here, workgroup-local, M) vs L5(05, global queue, L).**
  Not competitors — stages of one attack on the same waste (#3/#5 here, W6 there).
  Stage 1 = wgVisible gate + compacted shared list (+ subgroup distribution if TSL
  allows); stage 2 = global survivor compaction, only if P2/P3 show per-brick fixed
  cost still dominates after stage 1. 05's scar note verified: the deleted flat-domain
  scheme died of a correctness hazard (:1066-1075 header), not measured slowness.
- **Targets**: 05's "oblique ≤21" is the premise-audit budget; this doc's "−11..13 ms"
  matches the live-quantum gap (37.2 → ~24-26). Not a contradiction — different budget
  allocations; the gap statement here is current.
- **05's L3 (single-submit + dead kClearBins) — VERIFIED SAFE**, adopted into canon:
  the counter is incremented only under ?voxwrites (:1210-1212) and reads 0 either way
  (buffer zero-initialized); NaniteFrame's readVoxWrites keeps returning 0. The
  setIndirectDispatch enabler already exists (Tsl.ts:283-288). Default path is
  genuinely 3 submits today (:1461, :1489, :1490).

### SURVIVING LEVER TABLE (canonical, reconciled)

Expected ms = savings (eye / oblique / aerial), same-session isolated gpuWall.

| # | lever (source) | mechanism | eye | obl | aer | quality | probe / gate | effort | conf |
|---|---|---|---|---|---|---|---|---|---|
| V1 | Dead occ-mask-build skip (13-L1) | arm gate (:938) gains the complement of ray eligibility (`straddles==1 OR area ≤ voxcellmin OR !voxcell`); mask is provably unread on the ray path | 0.2 | 1–2 | 0.1 | IDENTICAL (dead value elided) | `?voxmaskray=0` A/B + shotdiff maxDiff=0 | S | med |
| V2 | K=2 F2B ± 1 pyramid rebuild (13-L2 ≡ 05-L1) | coarse near→far ordering seeds the guards; rebuilt pyramid gives PER-BRICK bocc vox-behind-vox on the far slab | ~0 | 0–4 | 0–3 | IDENTICAL (order-free atomicMax; conservative cull) | `?voxf2b=1&voxf2bk=2[&voxwaves=2]` — zero-code P1, decision rule 05 §6 | S probe / M productize | low-med |
| V3 | Exact-rect mip pick + rect-aligned window (05-L2) | UE5 MipLevelForRect idiom for BOTH cull tests (:636-654, :847-874); recovers culls the 4–16×-area over-pool misses | 0–1 | 1–3 | ~0 | IDENTICAL (strictly-more-culls, still conservative) | culled-brick counter Δ (needs P3) + shotdiff maxDiff=0 | S | low |
| V4 | Phase-B distribution, staged (13-L3 → 05-L5) | stage 1: `If(wgVisible)` Phase-B gate + compacted live-brick list (+ subgroup-per-brick); stage 2: global survivor queue + packed indirect Phase B | 0.3 | 1–2 (st.1); 2–5 (st.2, IF P3 says fixed-cost-bound) | 0.1–1.5 | IDENTICAL (same election set) | P2+P3 FIRST (do not build stage 2 blind); `?voxsgb=0` revert + shotdiff | M / L | med |
| V5 | Single-submit dispatchVoxel + drop dead kClearBins (05-L3) | 3 submits → 1 via dispatchBatchMixed + setIndirectDispatch tag; clear only feeds ?voxwrites | 0.2–0.8 | 0.2–0.8 | 0.2–0.8 | IDENTICAL (no math change) | gpuWall + cpu.submit A/B + shotdiff maxDiff=0 | S | high |
| V6 | Linear-clip Phase-A projection, SPLIT (13-L4 ≡ 05-L4, restated) | L4a: shape-preserving basis = brWR·(VP world columns) — pure algebra vs today; L4b (ray-eligible bricks only): true local-corner basis = brHalf·(VP·instance-linear columns) — tighter at yaw≈0 AND fixes the #9 yaw clip; also linearizes surviving mask builds (512 transforms → adds) | 0.2 | 0.7–2.5 | 0.1–1 | L4a: IDENTICAL-with-shotdiff-gate (fp reassoc, ulp); L4b ray-path: IMPROVING (silhouette ⊆ corner-hull bbox, provable); L4b flat-path: RISK — gate = shotdiff + user sign-off | P2 sizes Phase A first; `?voxlinproj=0` revert | M | med |
| V7 | Early sphere pre-bocc (13-L5) | sphere test (√3·brWR — yaw-safe) before the 8-corner walk; eye-leaning | 0.4 | 0.3 | 0 | IDENTICAL (conservative-only) | A/B; accept net-0 oblique risk | S/M | low |
| V8 | Straddler footprint tightening (05-L6) | live-spike guard only; ~0 at canonical poses | ~0 | ~0 | ~0 | RISK — gate = camera-inside-crown crops + user sign-off | forcevox close-up pose gpuWall + crops | S | n/a |
| V9 | Micro bundle (13) | occLo/occHi dedupe; group-0 ray bases; DDA loop break on `done` | — | ≲0.3 | — | IDENTICAL | ride along V1/V6 | S | high |

Honest reconciled oblique sum: **3–7 ms quality-identical** (13 said 3–6, 05 said 4–8;
the overlap is the range). The remaining ~6–9 ms oblique is generated ONE LEVEL UP (the
45–140 m per-tree ring's cluster count — aggregation area, not this kernel), consistent
with both docs and the premise audit.

### Killed claims (one-liners)

- "voxOccPyr is 14-16 levels / 14 dispatches" (13) — it's 12 at canonical res (:399-413).
- "Oblique ring ≈ 8.5k blocks @ ~1.2 µs/block" (05 W6) — 4,759 blocks @ ~2.3 µs (JSON-recomputed).
- "L4 basis projection is a pure refactor / provably identical" (both) — today projects
  the world-axis brWR box, not the local cube; as-written it changes the footprint shape
  (→ split L4a/L4b, flat-path L4b reclassified RISK).
- "nanite.voxBricks counter is CPU-only, no GPU cost" (13 probe #4) — needs a GPU
  counter; superseded by 05-P3's workgroup-aggregated ?voxstats.
- "cull window pools up to 4× the needed area" (05 W3) — 4–16× area (2–4× per axis);
  lever unchanged.
- Oblique foliage pool "21.8 ms" (13 premise 1) — reproduces as 22.0 (noleaves-now
  oblique med = 15.2, not 15.4); immaterial to any verdict.
