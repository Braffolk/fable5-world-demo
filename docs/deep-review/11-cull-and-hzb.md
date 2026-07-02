# cull-hzb deep review (2026-07-02)

Scope: hierarchy traversal (NaniteCull.ts, DagHierarchy.ts), HZB build+sampling
(NaniteHzb.ts), voxOccPyr + block/brick vox occlusion (NaniteVoxelRaster.ts cull half),
shadow shared-cut cull (NaniteClipCull.ts). Post-voxbocc baseline: eye 18.9 / oblique
37.2 / aerial 16.5 (isolated gpuWall med). Oblique gap to target ≈ −11..13 ms.

## Premise audit

1. **"The pyramid is mesh-only at vox-scatter time, so vox-behind-vox is invisible to
   EVERY cull" — HALF WRONG.** There are TWO occlusion structures with different content:
   - `voxOccPyr` (NaniteVoxelRaster.ts:399-474): min-pooled over THIS frame's
     `visPayloadV`, built inside `dispatchVoxel` (NaniteVoxelRaster.ts:1461) — at that
     point `visPayloadV` holds only the SW+HW **mesh** election (world1 order:
     NaniteRaster.ts:1418-1423). Mesh-only, correct claim.
   - The main HZB (NaniteHzb.ts) is built at NaniteFrame.ts:481, **after** `world1`
     **including** `dispatchVoxel` (NaniteRaster.ts:1423) — so the prev-frame HZB the
     DAG traverse tests at emit (NaniteCull.ts:919-925) is **full-content (mesh + vox +
     terrain), one frame stale**. Vox-behind-vox IS visible to the cluster-level cull.
   The practical conclusion survives for a different reason: cluster granularity is
   structurally too coarse at oblique. `sphereOccluded` picks the mip where the whole
   cluster sphere fits one texel (NaniteHzb.ts:175-176) and max-pools a 2×2 window —
   for a 105-brick crown cluster (let alone a 64 m FarTile head) the window spans ≥4×
   the footprint, and any sky/canopy-gap texel (empty ⇒ depth 1.0) forces KEEP. So
   cluster-level occlusion at oblique kills almost nothing; the *brick*-granular test
   (voxbocc) is the effective mechanism, and it is the one that only sees mesh.

2. **The aerial "floor cluster 8.7-11.9 ms" is NOT a reachable steady state — and it is
   not only the pose-arrival ramp.** Per-frame arrays: every run shows 1-3 cheap frames
   right after teleport (stale prev-pose HZB over-culls: fresh-final-rested aerial
   8.7, 8.1 → settle ~17; fresh-ablate-post3: 6.5, 11.5, 9.5 → ~17). But in
   fresh-voxbocc the aerial lows **persist all 32 frames with a ~3-frame period**
   (12.7, 12.1, 23.4 | 18.3, 13.6, 17.1 | 13.6, 23, 17.7 | 12, 21.3, 16.3 | 12.1,
   22.7, 17.9 | 11.8, 20.8, 16.6 …). With TAA ablated (fresh-ablate-post3) the aerial
   lows *vanish* (steady 14-19). So at aerial the periodic CHEAP frames are
   jitter-linked (some Halton phases let the HZB/cut kill more of the 661 huge FarTile
   clusters) — the TRAA-jitter refutation was established at OBLIQUE only and does NOT
   transfer to aerial. Oblique bimodality (runs of ~4, ±5 ms) indeed persists with TAA
   off — still unexplained, and notably *smaller* in the voxbocc run (33.5-41.5) than
   pre-voxbocc (35-39 vs 43-47.4).

3. **The lodWarp "this knob never ships" is the shipped default.** `lodWarp`'s header
   (NaniteCull.ts:104-126) says the distance-banded τ is a SIM that never ships, but
   NaniteFrame.ts:172-177 defaults `simband=6, lodnear=4, lodpow=0.6` — τ_eff ≈ 19 px
   at 100 m, ≈ 34 px at 300 m for trunks/mesh (vox clusters are clamped at
   `voxtaucap=12`, NaniteCull.ts:852-860). This is precisely the "mid-band already too
   coarse" the user flagged; any quality re-investment ("LOD engages farther") lands on
   these three uniforms + voxtaucap. Perf reviews must treat the current cut as
   warp-coarsened, not calibrated-τ.

4. **The perspective `sphereOccluded` is missing the off-screen guard its own ortho
   variant has.** NaniteHzb.ts:187-198 clamps the prev-NDC footprint to edge texels with
   no `|ndc|<1` check; the ortho variant explicitly refuses to occlusion-cull off-map
   casters for exactly this reason (NaniteHzb.ts:244-247). A cluster that passes the
   CURRENT frustum but projects outside the PREV frame's NDC (screen-edge entry during
   pans, any teleport) is tested against clamped edge texels of a stale pyramid ⇒
   spurious cull ⇒ 1-frame pop-in at screen edges / the pose-arrival over-cull ramp.
   This is a live correctness gap in the shipped single-phase scheme, not just a
   transient-measurement nuisance.

5. **Counter caveat:** `nanite.visTris` sums cluster word7&0xff, which for voxel
   clusters is the BRICK count (NaniteCull.ts:931 vs NaniteVoxelRaster.ts:562). At
   oblique, visTris 1.27M ≈ 0.96M bricks (9,185 vox clusters × ~105) + ~0.3M real mesh
   tris. Any "tris cheap at oblique" reasoning based on visTris must subtract this.

6. **Metric sanity:** isolated poses are static, so the stale-HZB *motion* inflation
   (2026-06-26: eye moving 25-28 vs static 17.4, pre-voxbocc) is invisible in every
   number in the fact pack except live p50/p95. Cull-area wins of the two-pass kind pay
   off in the LIVE metric, which is the actual mission metric.

## How it works today

**Per-frame order** (NaniteFrame.ts:440-495): `cam.update` (jittered VP; prevVp saved,
NaniteCommon.ts:108-115) → `cull.runPhase1` (hier BFS, occlusion vs PREV-frame
full-content HZB with prevVp/prevCamPos) → `syncFullArgs` → `runVoxFanout` →
`raster.world1` = [visClear, SW raster, hwArgs] + HW pass + `dispatchVoxel`
(NaniteRaster.ts:1418-1424) → `hzb.build` (NaniteFrame.ts:481; source = final
`visPayloadV` incl. vox) → shadows → resolve/post.

**Traverse** (NaniteCull.ts:722-1011): `kClearHier` → `kSeedRoots` (one thread per
instance; draw envelope far bound :775-779, vox nearDist near bound :786-787, frustum
:790, instMinPx :791-795 — default 0.075·min(W,H) ≈ 110 px diameter, NaniteFrame.ts:187)
→ `(kArgs, kTraverse)×hierDepth` ping-pong (hierDepth = registry.maxDagDepth+2,
NaniteFrame.ts:205), all in ONE batched submit (:998-1010). Per node: project ownError
(:831-835), lodWarp τ_eff (:836-842), voxTauCap clamp (:852-860); CUT ⇒ emit with
frustum + minPx + clipmap hollow + cone + **prev-HZB occlusion at emit only**
(:919-925); else enqueue children — the descend branch (:935-947) applies NO tests
(owner-dedup child links, DagHierarchy.ts:17-24). Vox fanout: qRaster re-scan by
matClass (:513-533); with ?voxf2b the range→count→prefix→scatter pipeline partitions
qVoxRaster into K linear-view-depth slabs (:553-710).

**HZB** (NaniteHzb.ts): half-res L0, 2×2 MAX (classic farthest) chain, one storage
buffer, ~1.11M texels total at 2268×1473; source = packed election key top 16 bits
(depth = 1 − (key>>16)/65535, :119-121; conservative — key truncation makes decoded
depth ≥ true depth). `sphereOccluded` = nearest-sphere-point vs 2×2 max at the
diameter-fits-one-texel level (:158-205); prev VP + prev camPos, current bounds.

**voxOccPyr + vox culls** (NaniteVoxelRaster.ts): min-pooled key pyramid (NEARER=LARGER
key; min over window = farthest/most-see-through pixel ⇒ conservative KEEP on any gap;
empty=0 sentinel forces keep) built fresh each frame pre-scatter (:1461). Block-level
cull: thread 0 tests the cluster sphere's footprint (2×2 at covering level) with
front-slab key, keep-on-tie via `|0xff` + strict-greater election (:588-673). Per-brick
?voxbocc (:838-884): same idiom against each brick's OWN clamped bbox + front-slab key;
straddlers exempt (:844). Both are exactly conservative: bbNearZ = min ndc.z over the 8
corners is the true box minimum (view-z is linear ⇒ extremum at a vertex), the pooled
min is a superset aggregate, and ties cannot win a strict-greater atomicMax. Waves
(?voxwaves=N, needs ?voxf2b): rebuild voxOccPyr between near→far bucket chunks
(:1466-1480) — later waves' block AND brick culls then see earlier waves' vox depth.

**Shadow culls**: clipmap shared-cut runs `buildNaniteCull(..., null, ...)` — NO
occlusion at all (NaniteClipCull.ts:133); `makeOrthoOccluded` (NaniteHzb.ts:217-248) is
dead code (zero call sites). Irrelevant for perf: nanshadow=0 measured shadows ≈ free.

## Work model

- Traverse: O(seed 600k instance-threads + visited nodes ≈ 2-4× emitted clusters
  (23k eye / 11.6k oblique / 0.7k aerial) + hierDepth×2 args kernels). Prior verdict
  ~0.2 ms — **not a lever**, and interior-node occlusion tests (UE does them) would
  only shave traverse time, since emit-time tests already gate every emitted cluster.
- HZB build: 14 dispatches, ~1.11M texel writes ≈ ~0.1 ms. voxOccPyr identical size.
  Measured: voxwaves=4 at eye = voxf2b-ctl +0.1 ms while adding 3 rebuilds + submit
  splits ⇒ **a pyramid rebuild costs ≲0.1-0.3 ms** — rebuilds are cheap; the F2B
  K-bucket barrier chain is what costs (+5/+6/+16 at K=16).
- Vox scatter work the culls gate: oblique ≈ 0.96M bricks × (8 corner projections +
  record) Phase A + footprint loads Phase B; per-pixel losers cost 1 load via the
  relaxed-load / voxcell front-slab early-out (:1266-1267). Oblique foliage total =
  37.2 − 15.4 (noleaves) ≈ **21.8 ms**; eye foliage post-voxbocc ≈ 18.9 − 16.8 ≈
  **2.1 ms** (eye vox is essentially solved; eye's remaining cost is base).
- What the measured numbers say about occlusion mechanisms:
  - brick-vs-mesh (voxbocc): −17.2 eye / −6.0 oblique / 0 aerial — granularity is
    decisive; the block-level cull existed all along and captured ~none of this.
  - block-vs-vox (waves, pre-voxbocc): −0 eye / −0.6 oblique / −4.7 aerial net of the
    rebuilds — block granularity fails against the porous canopy exactly as it failed
    against the mesh.
  - **brick-vs-vox has never been measured** (voxbocc shipped after the waves runs).
    That is the open cell of the 2×2 matrix, and the only in-area path with multi-ms
    oblique potential.

## Waste inventory

1. **Buried far vox behind the vox carpet at oblique** — bricks that pass the mesh-only
   voxOccPyr but lie fully behind nearer vox. Upper bound: the 60-140 m per-tree ring
   ≈ 10 ms at oblique (aggdist=60 diagnostic, includes visible content, pre-voxbocc);
   block-level evidence says small, brick-level unknown. Probe P2/P3 discriminates.
   Best guess 2-6 ms oblique.
2. **Spurious edge culls from clamped prev-NDC sampling** (premise §4): every pan/
   teleport; cost is negative (frames get cheaper) but it is a visible-artifact class
   (1-frame pop-in at screen edges) and it contaminates aerial medians (ramp frames
   6-12 ms inside the 32-frame sample).
3. **Jitter-flicker on FarTile-scale clusters at aerial**: ±9 ms peak-to-peak period-~3
   oscillation in fresh-voxbocc aerial, absent with ablate=taa. Median impact ~1-2 ms;
   p95 impact larger (aerial p95 19.9 vs med 14.6 in final-rested).
4. **Stale-HZB motion inflation** (live only): last measured 2026-06-26 (eye moving
   25-28 vs static 17.4, pre-voxbocc era). Unmeasured today; the two-pass architecture
   is the only fix; affects the mission metric (live p95 33.4) directly.
5. **Non-waste (checked, dead ends):** traverse pass count (measured-depth-sized,
   one submit); fanout re-scans (≤23k entries ×3); HZB/voxOccPyr build cost (~0.1-0.3
   ms each); shadow-cull occlusion absence (shadows ≈ free); makeOrthoOccluded dead
   code (delete or wire, no perf either way); keep-on-tie/straddler exemptions
   (verified exact-conservative, negligible cost).

## Levers

### L1 — Wave-split brick-granular vox-behind-vox (measure-only, flags exist)
- **Mechanism:** `?voxf2b=1&voxf2bk=2&voxwaves=2` — 2 linear-depth slabs, one
  voxOccPyr rebuild between; wave-2's per-BRICK voxbocc (now default-on, unlike the
  earlier waves runs) tests against wave-1's elected vox depth. Pure scheduling: the
  work-item SET is unchanged, deferral + conservative cull ⇒ byte-identical output.
- **Quality class:** identical (loss-exact reorder + conservative cull).
- **Expected:** eye −0..2 (mesh pyramid already covers most), oblique −2..5,
  aerial −1..3; minus K=2 chain cost ≈ +0.6/+0.75/+2 (linear-in-K extrapolation of the
  measured K=16 +5/+6/+16). Net oblique ≈ −2..−4 if brick-vs-vox behaves like
  brick-vs-mesh did.
- **Discriminator:** probes P1-P4 below (control separates chain cost from occlusion
  gain). If K=2 chain cost dominates, a code variant that splits WITHOUT the F2B
  histogram (two indirect dispatches over a depth-partitioned qVoxRaster) is S-effort.
- **Effort:** S (measurement now; small default-flip or dedicated 2-slab split after).
- **Risks:** the aerial K-chain cost may eat the aerial gain (aerial is at target
  anyway); oblique carpet may be genuinely mostly-visible (then this caps at ~−1).

### L2 — Vox two-pass deferral by prev-frame visibility (UE-style, conservative)
- **Mechanism:** in the fanout, partition qVoxRaster by a cluster-sphere test against
  the PREV-frame full-content HZB (already exists, already vox-inclusive): list A =
  "probably visible", list B = "probably occluded". Scatter A → rebuild voxOccPyr
  (now vox-inclusive, current frame) → scatter B, whose block+brick culls now fire
  against real same-frame vox occluders. Nothing is ever dropped by the prev-frame
  test — it only picks the pass — so output is byte-identical even under motion.
  This is UE5's two-pass occlusion mapped onto our vox stack, with "deferral" replacing
  "re-test" (cheaper: no reject recording, no extra storage buffers; the second
  dispatch reads the same qVoxRaster slice).
- **Quality class:** identical.
- **Expected:** oblique −2..6, eye −0..2, aerial −0..2 (better targeting than L1's
  depth split, no K-chain tax; costs 1 rebuild ≈0.1-0.3 ms + 1 extra submit + a 4-tap
  HZB read per fanout thread).
- **Discriminator:** run L1 probes first; if occlusion gain exists at brick-vs-vox but
  the F2B chain eats it, L2 is the build. Direct A/B once built: `?voxtwopass=0/1`.
- **Effort:** M (fanout split kernel + second scatter dispatch + prevVp binding in the
  fanout; ≤10-buffer budget OK — fanout binds 4 today).
- **Risks:** prevVp reprojection quality of the partition (bad partition = no gain, no
  harm); one more sync point in dispatchVoxel.

### L3 — Exact-rect occluder window (tighter voxbocc + sphereOccluded sampling)
- **Mechanism:** both tests take a 2×2 window at the level where the FULL footprint
  fits ONE texel — alignment slack means the pooled window spans up to ~4× the
  footprint per axis. For voxbocc the integer bbox is known exactly: sample the exact
  covering texel rect (≤3×3) one level finer; for sphereOccluded likewise (3×3 at
  level−1). Min/max over a ~4× smaller superset ⇒ strictly more culls, still a
  coverage superset ⇒ still conservative.
- **Quality class:** identical (conservative property preserved by construction).
- **Expected:** oblique −0.5..1.5 (compounds with L1/L2 — matters most when the
  pyramid actually contains vox occluders), eye −0..0.5, aerial ~0. Loads go 4→≤9 per
  brick (oblique ~0.96M bricks ⇒ +~5M loads, ~0.1-0.2 ms).
- **Discriminator:** counter for voxbocc kills (add a debug atomic under ?voxwrites
  idiom) before/after; or straight A/B flag `?voxboccw=2|3`.
- **Effort:** S-M. **Risks:** off-by-one in rect coverage = holes; needs the same
  no-hole eyeball + shotdiff gate voxbocc got.

### L4 — Off-screen guard in perspective sphereOccluded (quality fix, perf ~0)
- **Mechanism:** mirror the ortho variant's guard (NaniteHzb.ts:246): only occlusion-
  cull when the prev-NDC footprint lies inside [−1,1] (or pad by one texel). Removes
  spurious culls of clusters entering at screen edges / after teleports.
- **Quality class:** improving (removes an existing 1-frame pop-in artifact class);
  strictly MORE conservative.
- **Expected:** isolated poses ~0 (steady-state footprints are on-screen); slight
  +cost during fast pans (keeps entering clusters — the correct image). Also cleans
  the aerial pose-arrival ramp contamination of isolated medians.
- **Discriminator:** teleport-frame screenshots (frame 1-2 after pose set) before/
  after; aerial per-frame array should lose the 6-12 ms arrival dip.
- **Effort:** S. **Risks:** none identified; a few extra live clusters at edges.

### L5 — Cluster-level two-phase occlusion (record-not-drop + re-test vs fresh HZB)
- **Mechanism:** the long-known backlog item; scaffolding exists (rejClust buffer slot
  + caps NaniteCull.ts:97-99, comment :743-744, rasterDispatch2/p2Appends plumbing
  :419-425/:482-488). Phase 1 records occlusion-rejected clusters instead of dropping;
  after world1 + fresh hzb.build, re-test rejects vs the CURRENT-frame HZB and raster
  the survivors (rasterDispatch2 path). Eliminates disocclusion holes on motion
  entirely, and unlocks future aggressive phase-1 culling (e.g. prev-frame vox
  occluders at brick level) because phase 2 restores anything wrongly culled.
- **Quality class:** improving (fixes motion holes; static output identical).
- **Expected:** isolated ~0/0/0; live-moving slightly + (it adds back geometry that
  should be there) but licenses the phase-1 tightening that addresses the ~9-11 ms
  stale-HZB motion inflation measured 2026-06-26. Mission-metric (live p95) relevance
  high, isolated-metric relevance nil.
- **Effort:** M (second raster pass ordering: world1 → hzb.build → re-test → raster2 →
  hzb rebuild or accept partial; buffer budget already reserved).
- **Risks:** frame-graph reshuffle (hzb.build currently after everything); second SW
  raster submit bubbles.

### L6 — Aerial jitter-flicker stabilization (diagnose first)
- **Mechanism:** aerial's ±9 ms period-~3 oscillation is TAA-jitter-linked (premise
  §2): borderline HZB occlusion / cut decisions on 661 HUGE FarTile clusters flip with
  sub-pixel VP wobble. Candidate fix: evaluate the emit-time occlusion test with an
  epsilon consistent across jitter phases (pad nearestZ by one HZB-texel depth slack)
  or use the unjittered VP for the occlusion test only (cull decisions become
  jitter-invariant; raster keeps the jitter mirror). Padding direction = keep more ⇒
  conservative.
- **Quality class:** identical (padding only ever keeps more).
- **Expected:** aerial med ~0..−1, aerial p95 −2..4 (kills the expensive phases of the
  oscillation is the hope; possibly it kills the CHEAP frames instead — the data reads
  as jitter-induced over-cull, in which case this stabilizes variance without a med
  win). Oblique: unknown, its bimodality persists without TAA.
- **Discriminator:** P6 below (ablate=taa at aerial, 64 frames) already half-confirms;
  then a `?occleps` flag A/B.
- **Effort:** S probe / S-M fix. **Risks:** mis-padding loses real culls (perf, never
  quality).

### REJECTED-BY-POLICY (listed for completeness)
- **Prev-frame full HZB as a single-phase vox brick cull** (no re-test): would kill
  vox-behind-vox with zero new pyramids, but prev-frame-based DROPPING is not
  conservative under motion (1-frame disocclusion holes at brick scale). Note the
  shipped cluster cull (NaniteCull.ts:919-925) already has exactly this correctness
  class — surfacing, not deciding: if that precedent were ruled acceptable, this
  becomes an S-effort −2..6 oblique lever; under a strict reading of constraint #3 it
  ships only inside L2/L5-style two-pass.
- **simband/lodpow/voxtaucap coarsening** (τ warp already default): further coarsening
  is pure quality-trading; the USER direction is the opposite (spend perf wins to
  soften the warp). Not a lever; it is the quality bar's funding sink.

## What UE5/prior art does here

- **Two-pass occlusion (the main gap):** UE5 renders the previous frame's visible
  set first, builds the HZB from it (current VP, current depth), then tests everything
  else against that fresh HZB and renders the newly-visible remainder; both passes
  feed the final HZB. Result: no disocclusion holes AND tight culling under motion —
  the two things our single-phase prev-VP scheme trades away. Ours differs in that we
  have no per-cluster visibility persistence (instId×ci keying across frames is
  awkward with 600k instances × 157k clusters); L2's "partition + defer" and L5's
  "record + re-test" achieve the same guarantees without persistent visibility bits.
- **Occlusion during traversal:** UE tests HZB per BVH node while descending, pruning
  whole subtrees. We test only at emit; for us that is a traverse-cost optimization
  only (~0.2 ms total) — correctly skipped.
- **Granularity:** UE culls at cluster (~128 tri) granularity everywhere; our vox path
  effectively needed BRICK granularity (voxbocc) because vox clusters are ~105 bricks
  — consistent with UE's lesson that the cull unit must match the occluder porosity.
- **HZB sampling:** UE samples a 4×4 footprint at a finer mip (SPLIT_CULLING path) vs
  our fixed 2×2-at-covering-level — L3 is the port of that idea.

## Open questions + proposed serial probes

All TICKS=0 COOLDOWN_S=45 TREES=200000, ordered (baselines first, candidates after =
thermal bias against candidates). Commas separate EXTRA params (values may not contain
commas; ablate lists use `+`).

1. **P1 control (fresh same-session baseline):**
   `CONFIG=default LABEL=dr-base TICKS=0 COOLDOWN_S=45 TREES=200000 npx tsx tools/probe-fresh-stutter.ts`
2. **P2 K=2 chain-cost control (no waves):**
   `CONFIG=default EXTRA=voxf2b=1,voxf2bk=2 LABEL=dr-k2ctl ... probe-fresh-stutter.ts`
   → isolates the 2-bucket serialization tax (predicted +0.6/+0.75/+2).
3. **P3 the L1 candidate:**
   `CONFIG=default EXTRA=voxf2b=1,voxf2bk=2,voxwaves=2 LABEL=dr-k2w2 ...`
   → (P3 − P2) = brick-granular vox-behind-vox occlusion gain. THE money number.
4. **P4 finer split (only if P3 shows signal):**
   `CONFIG=default EXTRA=voxf2b=1,voxf2bk=4,voxwaves=4 LABEL=dr-k4w4 ...`
5. **P5 cluster-HZB value today (also bimodality arm):**
   `CONFIG=default EXTRA=occl=0 LABEL=dr-occl0 ...`
   → how much the emit-time HZB cull buys per pose now, and whether oblique's ±5 ms
   runs-of-4 survive with the occlusion feedback loop severed.
6. **P6 aerial jitter confirmation:**
   `CONFIG=default EXTRA=ablate=taa LABEL=dr-taaoff-aer FRAMES=64 ...`
   → does aerial's period-~3 oscillation vanish (predicted yes per fresh-ablate-post3)?
7. **P7 vox pyramid attribution (regression sanity, cheap):**
   `CONFIG=default EXTRA=voxoccl=0 LABEL=dr-voxoccl0 ...`
   → expected ≈ +17 eye (pre-voxbocc level); confirms the pyramid+block+brick stack is
   still carrying what we think it carries.

Open questions:
- Is the oblique vox carpet mostly genuinely-visible (L1/L2 cap out ≈ −1) or heavily
  self-occluding at brick scale (−4..6)? P2/P3 answers this; nothing else in this area
  moves oblique multi-ms.
- What is today's stale-HZB motion inflation (the 2026-06-26 ~9-11 ms figure predates
  voxbocc/fartiles)? Needs a live-moving A/B (occl=0 vs default, TICKS=600) before
  investing in L5.
- Oblique runs-of-4 bimodality: not explained by anything in this area's code; P5
  severs the only in-area feedback loop (HZB→cull→HZB). If it persists at occl=0,
  hand the hunt to the pipelining/measurement area.
- Where should the oblique −11..13 come from? Not from cull-hzb alone: the honest
  in-area ceiling is ≈ −3..−7 (L1/L2/L3 compounding). The remainder must come from
  the per-pixel coverage side (vox Phase B fill + resolve + post) or base raster.
