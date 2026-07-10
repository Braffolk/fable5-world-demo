# 00 — MASTER PLAN: locked 60 fps live, quality-neutral (2026-07-02)

**Mission:** locked 60 fps live at 200k trees, 2268×1473 (dpr 1.5), ZERO quality sacrifice.
User law: no optimization that loses visual quality ships; LOD should if anything engage
FARTHER; quality-trading knobs are attribution instruments only.

**This is the synthesis of the 2-fleet deep review** (docs 10–21 canon + 90 premise audit;
01–08 fleet-B siblings are detail only). Reconciliation status: docs 10, 11, 12, 13, 14, 15,
16, 17, 19, 20(+21) carry adversarially verified "Reconciliation & verification" sections —
those are ground truth here. **Doc 18 (webgpu-platform) never got a reconcile pass** — its
levers are marked `[18-unverified]` below and every 18-sourced build step starts with a cite
re-verification. All numbers: isolated gpuWall medians eye/oblique/aerial unless labeled live;
scratchpad = `/private/tmp/claude-501/-Users-sebastian-IdeaProjects-fable-demo2/cc111c9f-86e4-4c1a-be01-9817f021c312/scratchpad/`.

---

## 1. WHERE WE ARE

### 1.1 The measured state (post-voxbocc, 2026-07-02)

| metric | eye | oblique | aerial | source |
|---|---|---|---|---|
| isolated gpuWall med (run 1) | 18.9 | **37.2** | 16.5 | `fresh-voxbocc.json` |
| isolated gpuWall med (run 2, warmer) | 21.0 | **36.3** | 14.9 | `fresh-voxbocc-milestone.json` |
| noleaves (trunks+infra, fartiles OFF) | 16.8 | 15.2–15.4 | 11.1 | `fresh-noleaves-now.json` |
| post stack (ablate delta, same-session) | ~6.7 | ~5.2 | ~0 | doc 16 §Reconciliation |

**LIVE 600-tick milestone** (eye-level glide, `fresh-voxbocc-milestone.json`, recomputed from
raw deltas this pass): p50 **16.7 ms** (= the 60 fps quantum on the 120 Hz panel), avg 17.96,
p95 **25.0**, max 26.0. Slot histogram **{1-slot: 28, 2-slot: 451, 3-slot: 121}**;
**210/600 frames >16.7 ms**, every one of them landing at the next 8.33 ms quantum (25.0).
0 stutters, 0 longtasks, flat heap, live cpu.submit med 1.10 ms (p95 1.5, max 2.2).

### 1.2 The exact remaining gap

Live rAF deltas are hard slot-quantized (mean residual 0.18 ms — doc 20 PA2). "Locked 60"
means **underlying live GPU busy ≤ ~15.5 ms at every pose** (≥1 ms slack under the 16.7 edge,
else the 16.7↔25.0 flap is visible judder). Today ~35% of live eye-glide frames miss the
budget and display at 25.0.

- The live↔isolated ratio r is an **interval, not a constant**: r ∈ (0.65, 0.86], fitted
  ~0.69 at eye this run (docs 90 §2.1, 20 PA2). **r at oblique has never been measured** —
  the live phase never visits the money pose (doc 20 PA1).
- Required isolated oblique = 15.5/r ≈ **18–24 ms** (mid-band r=0.75 → **≤21**; only a pinned
  r ≤ 0.65 licenses the relaxed 24–26 framing).
- From 36.3–37.2, the oblique gap is therefore **−13 ± 3 ms, quality-neutral only**.
  Eye must hold the line (18.9–21.0, boundary-straddling live); aerial passes on median but
  its upper mode (~17–23 in `fresh-voxbocc.json`) would flap live slots — the bimodality must
  be resolved, not just the median.

### 1.3 What is banked (do not re-plan)

- **voxbocc** (per-brick cull vs mesh voxOccPyr): −17.2 eye / −6.0 oblique, shipped
  shot-gated IDENTICAL (verified `NaniteVoxelRaster.ts:838-884`).
- **Connected-bark cap-only** (`0f77edc`): every bark DAG collapses to ~1 root; the
  "46M tris / 97% sub-pixel" era is dead (boot logs `bark lod0 263→root 1`, doc 12).
- **FarTiles + worker splat** (boot 6.55 s + 4.01 s main-emit; 812 tiles / 4.03M bricks /
  138.5 MB / 57,435 clusters), **bead-v2/voxjit** round-normal shading, **voxrecip**,
  **f2b-off default**, **crown-LOD regression fix** (6a93dfb).
- **Live loop hygiene**: CPU is not a lever (cpu.update 0.02, cpu.submit 1.1, 0 GC hitches —
  doc 20 PA3).

### 1.4 What is proven free, absent, or dead (load-bearing corrections)

1. **Shadows and GI are ABSENT, not free.** `ForestScene.ts:456-459` passes
   `gi: null, canopyTex: null, csm: null`; `NaniteFrame.ts:244` then never builds the shadow
   system; `NaniteResolve.ts:237` compiles no shadow sample; ProbeGI exists only in
   TerrainScene (verified this pass). **Every canonical number contains exactly 0 ms of
   shadow/GI work.** `nanshadow=0` A/Bs were no-op-vs-no-op. The whole budget is provisional
   until the user rules on forest-vs-world scope (§3.6); world parity re-adds ~2–6 ms moving
   shadow raster + ~0.5–1.5 ms sampling + ~0.3–1 ms GI/CSM (doc 17).
2. **Base mesh raster ≈ 2–4 ms at oblique** — noleaves OVERSTATES base (it disables fartiles
   so bark extends 140→~230 m and terrain un-occludes). Default-oblique real mesh emit
   ≈ 0.65M tris; `visTris` counts BRICKS for vox clusters (docs 12/15). The oblique gap is
   NOT in base.
3. **The vox scatter is COUNT-bound, not fill-bound**: ~1.6–2.2 µs per emitted vox cluster;
   irreducible visible fill ≈ 4–5 ms oblique; ~14–16 ms scales with cluster/brick count
   (doc 13, slope table reproduced exactly from JSONs).
4. **Leaf decode ≈ free** (leafcheap=all ceiling <1 ms), **lighting math ≈ free at the
   canonical poses** (`nandbg=flat` −1.3), election atomics ≈ free (all reconfirmed doc 19).
5. **The vox LOD ladder truth** (doc 14): the anchorL0/tau formula is DEAD CODE — shipped
   `ownError = brick half-extent × errorK`, errorK=**1** (`VoxelizeCrown.ts:87, :988`),
   K_FLOOR=3 (`:1006-1007`); `voxTauCap=12` (`NaniteCull.ts:317-318`) governs the entire vox
   band; bricks sawtooth 12–24 px. The user-flagged mid-band coarseness is the **tile-L0
   shelf** (94–354 m band, 3 m bricks = 24–45 px at 94–177 m; `ForestScene.ts:317` ftcell
   0.75, `:390` tile nearDist 94) — unreachable by any τ/errorK knob; only finer tile content
   fixes it (§6).
6. **F2B K=16 (+5/+6/+16) was barrier-serialization at collapsed occupancy, not the cost of
   ordering** — ordering itself has never been measured (docs 10/18/19). The voxwaves-null
   was measured at BLOCK granularity pre-voxbocc; **brick-granular vox-behind-vox is the open
   cell of the 2×2 occlusion matrix** and the biggest single quality-identical unknown.

---

## 2. BUDGET LEDGER

Pool-based to avoid double-counting: docs 11/13/15/19 all feed on the SAME buried-vox pool;
docs 13/18/19 share the per-brick-constant pool. Booked = mid estimate of the family ceiling,
oblique ms. Confidence: **M** = measured (same-family precedent measured on this build),
**S** = mechanism-solid (code-verified mechanism, magnitude modeled), **?** = speculative.

| # | Pool (family) | Levers (owner docs) | oblique booked | range | conf | quality |
|---|---|---|---|---|---|---|
| A | Buried-vox ordering/occlusion (vox-behind-vox + tile-behind-per-tree-vox + unordered temporary winners) | zero-code K2+waves probe → tile-wave-occlusion (15-L1) and/or vox two-pass deferral (11-L2 = 19-L3) + `?voxf2bone` (19-L1) + brick sort (19-L8) | **4.0** | 2–8 | S (probe-gated; voxbocc precedent is the same idiom at −6 obl) | IDENTICAL (conservative min-pool, keep-on-tie, deferral-not-drop) |
| B | Per-brick/cluster fixed overhead (count-bound scatter constants) | V1 dead occ-mask skip (13) + V4 Phase-B distribution st.1 (13) + tgmem diet (18-L1 `[18-unverified]`) + subgroup setup broadcast (18-L4, same pool) | **3.0** | 1.5–5 | S (V1 dead-value proof; rest modeled) | IDENTICAL |
| C | 94–140 m double-draw ring | tile-ring-block-nearcull (14) + overlap-near-gate corrected (15-L2) | **2.0** | 1–3 | S (`?ftnear=140` diagnostic bounds it first) | IMPROVING (finer rep shows; seam shotdiff gate + Class-Q crops + user sign-off before default-on) |
| D | Resolve + post, quality-identical | RP-1 tri-class specialization, RP-3 TRAA ping-pong + AO rg16f, RP-4 single-pass resolve, RP-5 reorder (16) | **1.0** | 0.5–1.5 | S | IDENTICAL |
| E | Crumbs: submits, windows, dead work | V5 single-submit dispatchVoxel + dead kClearBins (13/05), coalesce-submits + skip-depthv-clear (10), V3/L9 exact-rect/4×4 occlusion windows (13/19), B2 `?f2b=1`, V7 sphere pre-bocc | **1.0** | 0.5–2 | S/M (V5 high-conf; windows counter-gated) | IDENTICAL |
| F | Base mesh (eye-skewed) | B1 hw1fetch, B3 vcache re-measure (12) | 0.25 (obl) / ~1 eye | 0–1 | S | IDENTICAL |
| — | **Default-path total (IDENTICAL/IMPROVING only)** | | **≈ 11.2** | **7–16** | | |

**Honest sum vs the gap:** booked ≈ **11 ms** against a required **−13 ± 3**. From 36.3–37.2
that lands ~25–26 oblique — hits the target ONLY if the live probe pins r ≤ 0.65.
**We are short ~2–5 ms against the mid-band target (≤21).** The shortfall is covered, in
order of preference, by:

1. **r-pinning** (probe Q3): if live-oblique r ≤ 0.65, the target relaxes to 24–26 and the
   default path suffices — worth up to ~5 ms of ledger by itself (doc 20).
2. **L-tier conditionals** (§3.4): vox kernel bin-split (19-L4, +2.5 obl if the `voxcell=0`
   slope diagnostic confirms register pressure), Phase-B stage-2 global queue (+2–5),
   coarse-occupancy cell-accurate re-bin (14, +2–5 IMPROVING).
3. **RISK shelf with user sign-off** (§3.5): mid-ring merged tiles (+4–8), voxnear 60 (+2–4),
   occupied-bounds shrink (+1.5).
4. **Bimodality resolution** (Q5/Q6): if the HZB-feedback hypothesis (B) is real, two-pass
   occlusion removes 2–4 obl / 3–5 aerial of periodic re-expansion; if DVFS (H), the
   isolated numbers are pessimistic and the real gap shrinks for free.

NO padding: items already counted in A–E are not re-counted in 2–4; the L-tier and RISK
numbers overlap pool B/A respectively and are quoted post-overlap from the reconciled docs.

**Aerial/live-p95 side-budget** (not in the oblique sum): B5 sphereOccluded on-screen gate +
L6 jitter-invariant pad (aerial p95 −2..4, kills pose-arrival over-cull) + mesh two-pass
occlusion (live-moving stale-HZB inflation, last measured ~9–11 ms pre-voxbocc, REMEASURE).

---

## 3. THE ORDERED PLAN

Every step: {mechanism, files, expected ms e/o/a, quality class + gate, measurement gate,
effort, deps}. **Default path = IDENTICAL/IMPROVING only.** RISK lives in §3.5 and ships
nothing without explicit user sign-off. **IMPROVING is still a pixel change (doc 90 §4
Class Q): EVERY IMPROVING lever — pool C / M2f ring nearcull, S4 sphereOccluded gate, M2m
two-pass occlusion, the L-tier IMPROVING items, all of them — requires side-by-side crops +
explicit USER sign-off before default-on; its shotdiff/counter gate licenses the build, not
the ship.** Shotdiff gates: 3 canonical + 2 stress poses, TAA
jitter index pinned, same seed/ToD, per doc 90 §4. Every null result requires an engagement
counter before it may be recorded (tripwire #2). Metal 10-storage-buffer cliff: any lever
adding a binding must fold into an existing buffer (scar precedent `NaniteRaster.ts:339-345`)
— state this in every touched PR.

### Phase 0 — measurement first (S, tools-only edits, no renderer changes)

**M0. Fresh same-session baseline** — every session starts here; cross-era deltas <2 ms are
noise (doc 90 §3.7). Command: probe queue Q1.

**M1. Pose-complete live milestone** (doc 20 lever 1). Add `POSE_PATH` segments to phase A
(`tools/probe-fresh-stutter.ts`, vite-ignored). Pins r at oblique/aerial → converts the
oblique target from an interval to a number. Decision: required iso-oblique =
16.7 × (iso-obl same-run / fitted-live-obl); report immediately if r ≥ 0.8 (budget honesty).
Also: relabel the >33.4 "spike" counter (it sits on the 4-slot edge — doc 20 PA4) and record
measured refresh. Effort S. Deps: none. **Worth up to ~5 ms of ledger.**

**M2. DVFS/duty discriminator** (doc 20 lever 2 = doc 10 B2). Forward `MEASURE_CD_MS` to
`measureFrames({cooldownMs})`, sweep 10/50/200 at oblique. Decision: med moves >10% with duty
⇒ isolated numbers carry an era-correction and the bimodality is a harness artifact (H);
band period constant in wall-time ⇒ H confirmed, the "periodic 4–9 ms cost" leaves the
renderer ledger. Effort S.

**M3. Brick-vs-vox occlusion money probe** (zero code; docs 11 P2/P3 = 13 L2 = 19 P-A).
Q2a/Q2b below. (P3−P2) = brick-granular vox-behind-vox gain, the open 2×2 cell. Decision
rule: gain ≥2 ms obl ⇒ fund pool A builds (tile-wave first if the tile share dominates per
M4; two-pass deferral otherwise); gain <1 ms with engagement confirmed (voxBrickWrites delta)
⇒ pool A caps at ~1–2 and the plan re-weights to pools B/C + L-tier.

**M4. Attribution re-pins** (zero-code + one S flag): post-voxbocc `aggdist=60` (ring share),
`fartiles=0` (tile share + skyline-antenna discrimination, doc 15 probe 3), then the S-effort
`?ftskip=1` boot-skip variant (15 lever 0). Decision: fixes the per-tree-ring vs tile split
that pools A/C and the §6 spends are priced against.

**M5. Bimodality discrimination** (docs 10 B1–B4, 11 P5/P6, 15 L3-gate): `ablate=taa` alone
FRAMES=64 at aerial; `occl=0` FRAMES=120; per-frame counters (`?meterevery=0` + per-frame
readCounts patch). Decision: counters oscillate with bands ⇒ HZB-feedback (B) ⇒ two-pass
occlusion is the fix (IMPROVING); flat counters + duty-scaling ⇒ H ⇒ metric fix only.

**M6. Post singles at eye** (doc 16 amended queue): `ablate=contact`, `ablate=ao` (drops
AO+contact), `ablate=bounce`, `ablate=bloom`, `ablate=taa,bloom` (TAA = pair difference —
`ablate=taa` alone is biased LOW, doc 16 correction 5). Decision: any single >1.5 ms at
oblique becomes a named candidate; <1 ms closed.

### Phase 1 — S-effort builds, all IDENTICAL/IMPROVING (land while probes cycle)

| step | mechanism | files | e/o/a ms | quality + gate | measurement gate | effort | deps |
|---|---|---|---|---|---|---|---|
| S1. V1 dead occ-mask skip | arm gate (`NaniteVoxelRaster.ts:938`) gains the complement of ray eligibility; mask is provably unread on the ray path (`:1240-1252` flat-only — verified this pass) | NaniteVoxelRaster.ts | 0.2 / **1.5** / 0.1 | IDENTICAL (dead value elided) | `?voxmaskray=0` A/B + shotdiff maxDiff=0 | S | — |
| S2. V5 single-submit dispatchVoxel + drop dead kClearBins | 3 submits → 1 via `dispatchBatchMixed`; WRITE_CTR only read under `?voxwrites` (verified `:1454-1492`) | NaniteVoxelRaster.ts | 0.2–0.8 each pose | IDENTICAL | gpuWall + cpu.submit A/B + shotdiff=0 | S | — |
| S3. `?voxf2bone` barrier-free F2B (19-L1) | f2b fanout's counting sort + ONE whole-list dispatch; kVoxPrefix already publishes whole-list args (`NaniteCull.ts:661-663`, verified) — launch-order F2B seeds the prevE guards with ZERO residency loss | NaniteCull.ts, NaniteVoxelRaster.ts | 0.3 / **1–3** / 0.5 | IDENTICAL (order-free atomicMax) | Q4; engagement: `nanite.voxBrickWrites` −≥25% + bucket-range dump; null without engagement ⇒ invalid | S | M3 informs expectations, not blocking |
| S4. B5 sphereOccluded on-screen gate | perspective test lacks the ortho's `\|ndc\|<1` refusal (verified `NaniteHzb.ts:200-204` vs `:246-247`) → edge-straddling frustum-survivors over-cull → pan/teleport pop-in | NaniteHzb.ts (one line) | ~0 / ~0 / ~0 (cleans aerial arrival dips) | IMPROVING (strictly more conservative; fixes a conservative-cull violation) | pan-pose shotdiff + visClusters delta + aerial per-frame array loses the 6–12 ms dip; Class-Q crops + user sign-off before default-on | S | — |
| S5. B2 `?f2b=1` default-flip probe | skip z-interp for provably-losing frags (`NaniteRaster.ts:983-989`); guard already kills losers' RMW | flag only | −0..0.5 each | IDENTICAL (loss-exact) | EXTRA=f2b=1 3 poses + noleaves variant + shotdiff=0; keep off if net-negative | S | — |
| S6. Cut-telemetry (14) | boot print maxExt/cellSize/per-level occ+blocks/T(L) + `?nanitedbg=lod` tint shots — resolves the 2-vs-3-levels band ambiguity, de-risks §6 spends | VoxelizeCrown.ts, ForestScene.ts | 0 | IDENTICAL (instrumentation) | rides any probe | S | — |
| S7. Per-matClass scar counters (12 probe #1) | generalize scar counters (hwQueue-tail fold, zero new bindings) to terrain/bark/leaf/vox frags + owned pixels | NaniteRaster.ts | 0 | IDENTICAL (instrument) | one `EXTRA=scar=1` run retires every §Waste estimate | S | — |
| S8. Probe hygiene (10) | `?meterevery=0` in phase B + per-frame readCounts (feeds M5); skip-depthv-clear (gate `packedClear && !shadowOn && !nanprobe` — doc 10 correction 5) | NaniteFrame.ts, NaniteRaster.ts:391 | 0.05–0.15 | IDENTICAL | A/B flag | S | — |

### Phase 2 — M-effort builds, probe-gated

| step | mechanism | files | e/o/a ms | quality + gate | measurement gate | effort | deps |
|---|---|---|---|---|---|---|---|
| M2a. Tile-wave-occlusion (15-L1) | per-tree vox wave → ONE voxOccPyr rebuild → tile wave; per-brick bocc then sees the 45–140 m vox canopy (proven conservative idiom) | NaniteVoxelRaster.ts | 0.5 / **2–5** / ~0 | IDENTICAL (min-pool, keep-on-tie, straddler-exempt) | `?voxtilewave=1` A/B + shotdiff=0 at 5 poses + tile-reject counter >0 | M | M3 signal + M4 tile share |
| M2b. Vox two-pass deferral (11-L2 = 19-L3) | fanout partitions qVoxRaster by prev-frame full-content HZB (visible → scatter → pyramid rebuild (now vox-inclusive) → deferred scatter); deferral-not-drop ⇒ byte-identical under motion | NaniteCull.ts, NaniteVoxelRaster.ts | 1 / **2–6** / 1 | IDENTICAL | `?voxtwopass` A/B + shotdiff=0 | M | build ONLY the winner of M2a-vs-M2b per M3/M4 (same pool A — never book both) |
| M2c. `[18-unverified]` vox tgmem diet (18-L1) | pack u16-range shared arrays (~6.6→~4 KB/WG) ⇒ residency 4→7-8 WGs/core; attacks the per-cluster slope's latency term. FIRST re-verify doc-18 cites (NaniteVoxelRaster.ts:526-549 array census, NANITE-SPEC limits) | NaniteVoxelRaster.ts | 0.3–1 / **1–3** / 0.2–0.5 | IDENTICAL (integer repack, bit-exact) | `?voxtgpack=0` revert A/B + shotdiff=0 | M | cite re-verify; overlaps V4 — land one, re-measure before the other |
| M2d. V4 Phase-B distribution, stage 1 (13-L3) | `If(wgVisible)` Phase-B gate + compacted live-brick shared list (+ subgroup-per-brick if TSL allows); per-brick setup issues once, culled/empty bricks never visited; V9 micro bundle rides | NaniteVoxelRaster.ts | 0.3 / **1–2** / 0.1–1.5 | IDENTICAL (same election set) | `?voxsgb=0` revert + shotdiff=0; P2/P3 (`?voxrdbg=2`, `?voxstats`) size it first | M | 05-P3 `?voxstats` counter |
| M2e. V3/L9 occlusion windows (13/19) | exact-rect mip pick (UE5 MipLevelForRect) or 4×4-at-finer-mip for sphereOccluded + block + voxbocc; strictly-more-culls, still conservative | NaniteHzb.ts, NaniteVoxelRaster.ts | 0–1 / **0.5–2** / ~0 | IDENTICAL | brick-skip counter +≥20% AND ≥2 ms obl interleaved A/B, else discard; shotdiff=0 | S-M | S7 counters |
| M2f. Ring-block nearcull (14) | per-cluster near cull for fartile heads: drop when provably covered by live per-tree reps (ground-anchor bound `sqrt((dXZ+r+reach)²+camY²) < aggDist` — doc 15 correction 4) | NaniteCull.ts, FarTiles.ts | 0.3 / **1–3** / 0.7 | IMPROVING (removes coarse-outbids-fine defects; 135–145 m seam shotdiff gate) | `?ftnear=140` diagnostic bounds it FIRST (corner holes, diag only); then `?ftneargate=1` + ring-hole shotdiff + 94–140 m band crops + user sign-off before default-on (Class Q — the user actively scrutinizes this band) | S diag / M build | M4 |
| M2g. RP-1 tri-class specialization (16) | strip dead terrain+rock subgraphs from forest tri resolve (register pressure; vox-pass 37.5 ms cliff precedent); carve out noiseA for wind ctx (16 correction 6) | NaniteResolve.ts, ForestScene.ts | 0–1 / 0–0.8 / 0–0.3 | IDENTICAL (classes have zero clusters) | `?resclasses=auto` A/B + shotdiff=0 | M | — |
| M2h. RP-3 TRAA ping-pong + AO rg16f (16) | kill 2 full-res copies/frame (TRAANode fork) + new `rg` HalfResEntry flag (the `red` hook is r8unorm — would break the view-z guide) | three fork, HalfResMrt.ts | 0.3–0.8 each | IDENTICAL | interleaved A/B + shotdiff=0 | M | — |
| M2i. RP-4 single-pass resolve (16) | merge tri+vox passes where the storage-buffer union is exactly 10 (forest) | NaniteResolve.ts | ~0.3 each | IDENTICAL | `?respass=1` + `tools/vcdebug.mjs` silent-death check + shotdiff=0 (the 11th-buffer trap) | M | — |
| M2j. Coalesce-submits (10) | fold syncFullArgs + fanout(3) into the BFS batch; hzb into the post-hwRender submit | NaniteCull.ts, NaniteFrame.ts | 0.2–0.5 iso; **live ~0** (queue never starves — do NOT book live ms) | IDENTICAL | same-session A/B `?coalesce` | S/M | S2 |
| M2k. L6 jitter-invariant occlusion pad (11/19) | pad nearestZ by one HZB texel (padding-only variant — the unjittered-VP variant is NOT conservative alone, 19 correction 8) | NaniteCull.ts/NaniteHzb.ts | 0 / ? / med ~0–1, **aerial p95 −2..4** | IDENTICAL (pad only keeps more) | M5's `ablate=taa` MUST run first; then `?occleps` A/B | S-M | M5 |
| M2l. B1 hw1fetch (12) | HW vertex stage fetches 3 corners, uses 1 (verified `NaniteRaster.ts:1123-1128`); refactor via `fetchWorldVertByIndex`, keep HF window-grid on 3-fetch | NaniteRaster.ts, NaniteFetch.ts | **−0.6 eye** / ~0 / 0 | IDENTICAL (same selected vertex) | `?hw1fetch=1` A/B at eye + shotdiff=0 | S/M | — |
| M2m. Mesh two-pass occlusion (B4/L7/L5-11) | record prev-HZB rejects into qRaster tail (slot-3 counter free) → re-test vs fresh pyramid → append raster; kills 1-frame disocclusion holes + licenses tighter phase-1 | NaniteCull.ts, NaniteRaster.ts | +0.1–0.3 iso; **live-p95 lever** (stale-HZB motion inflation ~9–11 ms measured 2026-06-26 pre-voxbocc — REMEASURE first) | IMPROVING (motion correctness; static bit-identical) | live-moving `occl=0` vs default A/B (TICKS=600) BEFORE building; then static shotdiff=0 + moving hole-pixel A/B + Class-Q crops + user sign-off before default-on | M-L | M1, M5 |

### Phase 3 — L-tier conditionals (build only if the ledger is still short after Phases 1–2)

- **19-L4 raster-bin the vox monolith** (flat-only/ray-only kernel split): 0.5 / **2.5** / 0.3,
  IDENTICAL. Gate: run the P-C diagnostic FIRST (`EXTRA=voxcell=0`, quality-changing
  instrument): per-cluster slope collapse ⇒ register pressure is the 3–5× multiplier ⇒ build.
  Risks: 10-buffer cliff (fold lists into existing buffers). Effort L.
- **V4 stage 2** (global survivor queue + packed indirect Phase B): **2–5 obl**, IDENTICAL.
  Gate: P3 `?voxstats` shows per-brick fixed cost still dominates after stage 1.
- **Coarse-occupancy cell-accurate re-bin** (14 = 04 §4.2): stamp child occupied CELLS not
  whole child boxes; masks ~8× tighter/level, carve gate re-engages. +0.75 eye / **+2–5 obl**,
  IMPROVING (toward mesh ground truth; far-crown crop gate + user sign-off). Gate: P5 boot
  counters (per-level popcount histogram + carve rate); build only if median L2/L3
  popcount > 48. Effort M.
- **occ==0 crown-brick prune** (14 = 04 §4.6): FarTiles-style prune in voxelizeCrown emit.
  **0–1.5 obl**, IMPROVING (phantom sub-2%-coverage fill only). Gate: P5 counter; close if
  <1% of occupied. Effort S.
- **fartiles boot-cache / B1 artifact-cache** (20/21/15): IndexedDB DDC for tree DAGs +
  crowns + fartiles. 0 frame-ms; boot 73.6 → ~17–19 s warm (+B4 trees-only veglib → ~12–16 s).
  IDENTICAL (gate: byte-identical first-frame shot cold vs warm). Not mission-critical;
  triples probe throughput — schedule opportunistically.

### 3.5 RISK shelf — nothing here ships without explicit user sign-off + shotdiff evidence

| lever | obl ms | why RISK | gate |
|---|---|---|---|
| Mid-ring merged tiles 70–140 m (14 = 04 §4.1) | +4–8 | per-tree sway/yaw/scale variation lost; bead crownDir becomes per-tile radial (verified `NaniteResolve.ts:810-826`) | post-voxbocc `aggdist=60` re-pin + ftnear bound FIRST; side-by-side crops + sway A/B video + sign-off |
| voxnear 60 + leaflodk 0.4 (14 = 04 §4.3) | +2–4 | 45–60 m band swaps vox bricks for coarse leaf-ladder levels — the look that got 60/0.25 rejected | 04-P2 A/B + 55–60 m crops + sign-off |
| Occupied-bounds shrink, ray path (19-L2) | +1.5 | only reachable via removing 6-step-exhaustion false fills — far crowns thin slightly | `?voxoshr=0` revert + far-band crops + sign-off |
| Streamed tiered ring tiles (15/06-L3) | +3.5 beyond tile-wave | resample + normal-field shift + stream pop | ONLY if P-A/P-B residual ≥5 ms post tile-wave; needs S1 streaming enablers (21) |
| RP-2 half-res contact shadows (16/08) | +0.6–1.1 (eye +1.5–2.3) | 1–2 px contact detail visibly changes | M6 confirms contact >2 ms eye first; jitter-pinned shotdiff + sign-off |
| Tile-pyramid shell probe (15) | +small | NOT silhouette-identical under voxcell carving | `?voxlodshell=1` + shotdiff gate |
| V8 straddler footprint tightening (13/05) | ~0 canonical (live-spike guard) | camera-inside-crown behavior change | forcevox close-up crops + sign-off |
| Clip-storm-budget (17, world scene) | 0 canonical | deferred coarse shadow level serves a one-snap-stale map | fly-speed shotdiff + teleport flush + p95 histogram |
| rg11b10 RT diet (16/08) | 0.3–0.6 | banding under AgX | dark-scene crops + sign-off |
| B5 progressive far-field boot (20/21) | 0 (boot UX) | visible far-field pop-in transient | controlled reveal + sign-off |

Red list (permanently rejected as perf levers, diagnostics only): `aggdist<140`,
`voxlodk ≥ 0.7`... i.e. `voxlodk<1`, `voxtaucap>12`, `leaflodk<0.4`, `ftcell>0.75`,
`instminpx` raises, `dpr<1.5` (1080p60+upscale = user's sanctioned last resort only).

### 3.6 User rulings needed (surfaced, not parked)

1. **Forest vs world scope for "locked 60"** (doc 17 open Q1): the canonical scene has NO
   shadows/GI/clouds/froxels. Either "forest look is final" (declare on forest numbers) or
   the budget must reserve ~3–8 ms of world-parity headroom and lever
   `?forestshadow=1` (17 lever 1, IDENTICAL instrument) lands to price it.
2. **RISK shelf sign-offs** (§3.5) as their gates produce evidence, **plus Class-Q
   default-on sign-offs for every IMPROVING lever on the default path** (pool C / M2f,
   S4, M2m, L-tier re-bin + occ-prune) — crops precede default-on in all cases.
3. **Quality-up spend order** (§6) once surplus exists.

---

## 4. SERIAL PROBE QUEUE

Rules: one GPU job at a time; `TICKS=0 COOLDOWN_S=45 TREES=200000` unless stated; baselines
FIRST each session (thermal bias against candidates); interleave A/B within one session;
never compare across thermal eras; live effects judged on slot histograms, isolated on
gpuWall medians; every oblique/aerial number reported with its mode structure (p25/p75);
null results need engagement counters. `[patch]` = small tools/src edit before the run.

| # | command | decision rule |
|---|---|---|
| Q1 | `CONFIG=default LABEL=mp-base TICKS=0 COOLDOWN_S=45 TREES=200000 npx tsx tools/probe-fresh-stutter.ts` | fresh same-session baseline; all Q2–Q9 deltas read against this |
| Q2a | `CONFIG=default EXTRA=voxf2b=1,voxf2bk=2 LABEL=mp-k2ctl …` | K=2 chain-cost control (predicted +0.6/+0.75/+2) |
| Q2b | `CONFIG=default EXTRA=voxf2b=1,voxf2bk=2,voxwaves=2 LABEL=mp-k2w2 …` | **(Q2b−Q2a) = brick-vs-vox occlusion gain, THE money number.** ≥2 ms obl ⇒ fund M2a/M2b; <1 ms with engagement ⇒ pool A caps low, re-weight plan |
| Q3 | [patch: POSE_PATH] `CONFIG=default LABEL=mp-live-poses TICKS=900 POSE_PATH=eye,oblique,aerial COOLDOWN_S=60 …` | pins r at oblique/aerial; required iso-obl = 16.7 × iso/live-fit; r≤0.65 ⇒ relax target to 24; r≥0.8 ⇒ tighten toward 18 and tell the user immediately |
| Q4 | [after S3] `CONFIG=default EXTRA=voxf2b=1,voxf2bone=1 LABEL=mp-f2bone …` | UE5-idiom ordering; gate: voxBrickWrites −≥25% + bucket-range dump; obl −≥2 ⇒ productize |
| Q5 | `CONFIG=default EXTRA=ablate=taa FRAMES=64 LABEL=mp-taaoff-aer …` | aerial period-3 vanishes ⇒ jitter-linked ⇒ M2k; persists ⇒ H (DVFS) gains weight |
| Q6 | [patch: MEASURE_CD_MS] `LABEL=mp-duty-cd10/50/200`, oblique, low→high | med moves >10% with duty ⇒ DVFS owns the band; constant wall-period ⇒ delete the phantom periodic target |
| Q7 | `CONFIG=default EXTRA=occl=0 LABEL=mp-occl0 FRAMES=120 …` | bands vanish ⇒ HZB-feedback (B) ⇒ fund two-pass occlusion; also pins today's emit-HZB value per pose |
| Q8 | `CONFIG=default EXTRA=aggdist=60 LABEL=mp-agg60 …` (diagnostic) | post-voxbocc ring-share re-pin (was 10.7 obl pre-voxbocc); prices pools A/C + §6 spends |
| Q9 | `CONFIG=default EXTRA=fartiles=0 LABEL=mp-noft …` (diagnostic) | tile share + skyline-antenna attribution (antennas persist ⇒ ≤140 m coarse bark, not tiles) |
| Q10 | [after S1] `CONFIG=default EXTRA=voxmaskray=0 LABEL=mp-maskctl …` vs default | V1 A/B + shotdiff=0; ship on any positive delta |
| Q11 | post singles at eye: `EXTRA=ablate=contact` / `ablate=ao` / `ablate=bounce` / `ablate=bloom` / `ablate=taa,bloom` | doc 16 rules; TAA = Δ(taa,bloom)−Δ(bloom); >1.5 ms obl ⇒ named candidate |
| Q12+ | candidate A/Bs as builds land: `voxtgpack=0` ctl, `voxtilewave=1`, `voxtwopass`, `occwin=4`, `ftnear=140` diag, `ftneargate=1`, `resclasses=auto`, `respass=1`, `hw1fetch=1`, `f2b=1`, `occleps`, `prof=0` live, `voxcell=0` diag (pre-L4), `voxlodkring=2` (§6), `voxtaucap=8` (§6), `ftcell=0.5 TREES=100000` (§6) | each per its step's gate above |

Cheapest decisive discriminators first: Q2 (zero code), Q5/Q7 (zero code), Q3/Q6 (tools-only
patches). Q8/Q9 are quality-rejected configs used strictly as instruments.

---

## 5. KILL LIST (do not re-propose; one-line evidence each)

**Architecture/premise kills**
- "Shadows/GI measured ≈ FREE" — systems ABSENT: `gi:null/csm:null`
  (ForestScene.ts:456-459 → NaniteFrame.ts:244); `nanshadow=0` was no-op-vs-no-op.
- ProbeGI 128-frame cycle as bimodality suspect — never constructed in forest
  (TerrainScene.ts:113/:120 only); per-frame work constant; observed quasi-period ~6-7.
- "Base is TRIANGLE-EMIT-bound / 46M tris / 97% sub-pixel" — bark 1-root collapse shipped
  (0f77edc); tri-swing disproof (aerial 3.7× fewer tris costs MORE).
- "Trunk far-field DAG coarsening" as an open big rock — already landed (cap-only + 140 m
  bark clamp); residual is 2–10 px roots = quality-barred.
- Atomic-contention-bound raster — refuted (noguard neutral; capture-era memory).
- Shade-binning resolve — decode ≈ free (<1 ms leafcheap ceiling); UE5 anti-lever.
- Tiled raster family (SPEC D-N46, TileBricks port, sort-middle) — measured +11.7/+18.1 ms,
  REMOVED; re-litigation needs new evidence.
- "F2B/ordering is refuted" — K=16 measured barrier-serialization at collapsed occupancy
  (≤10 µs/dispatch bound; residency model doc 18); ordering itself never measured.
- voxwaves-null transfers to brick granularity — it was block-granular, pre-voxbocc; the
  brick cell of the matrix is open (this plan's Q2).
- "Live = 0.65× isolated" as constant — interval (0.65, 0.86]; r unmeasured at oblique.
- 64-bit-election emulation / two-word schemes — no WebGPU 64-bit atomics (gpuweb#5071);
  Epic's own voxel experiment uses OUR 32-bit idiom (ScatterBricks.usf:10-23).
- Doc 02's "UE5's fallback is our architecture" — false: `#error UNKNOWN_ATOMIC_PLATFORM`
  (NaniteWritePixel.ush:33).
- shader-f16 bandwidth lever, persistent threads, literal UE5 wave-ring redistribution —
  refuted/blocked on WGSL (doc 18 §Refuted).
- Per-pixel-loop / cull as base bottleneck — refuted 2026-06-26 (cull ~0.2 ms).
- NDC-z front-to-back "refutation" — was metric rot (far-field bucket compression).

**Mechanism/estimate kills (from the reconcile passes)**
- skip-haze-on-sky (16-L3) — TSL `select` already emits a real branch; waste doesn't exist.
- reskeep in forest — keep block not compiled (`shadowsOn=false`); world-scene lever only.
- `?culloverlap` in forest — requires `shadow?.cullPrepass`; can never fire (csm null).
- Voxel clusters out of qRaster (12-L4) — bail bounded ≲0.1 ms; do not build.
- coalesce-submits as a LIVE win — live queue never starves (cpu.submit 1.1 vs GPU 14+).
- "HZB is 14 levels" — 12 at 2268×1473; hierDepth≈14 is BFS passes, a different number.
- fresh-base-vc* "5.4 ms" runs — vcompact 11th-buffer empty-scene artifact (hwTris=0).
- Doc 06 trunk-antenna mechanism — tile trunk columns cap at ≤2.8 m (FarTilesSplat.ts:197);
  antennas are almost certainly ≤140 m coarse bark (Q9 discriminates).
- Doc 04's FarTiles band map — wrong config read (ftcell is 0.75 not 0.5); tile L0 owns
  94–354 m, 45 px at 94 m; no tile L4 exists.
- "errorK=3 + K_FLOOR=3 shipped" (memory) — code ships errorK=1 (VoxelizeCrown.ts:87).
- anchorL0/tau ladder as the live mechanism — dead config, zero consumers.
- "lodWarp never ships" — simband=6/lodpow=0.6 IS the shipped default (NaniteFrame.ts:172-177).
- f2b "−0.5..−2 ms" — election guard already skips losers' RMW; −0..−0.5.
- Doc 01 L5 adaptive rect walk — sub-floor + inner-loop family measured dead.
- Doc 01 L2's −4 ms finer-window band — over-counted (window can't see vox-behind-vox);
  banded 0.5–2 counter-gated.
- "shelling is silhouette-identical" — false under voxcell carving; RISK class.
- L2 reachMargin=16 m sufficiency (15) — 3D anchor offset breaks it at elevated poses;
  corrected ground-anchor gate required.
- "aerial bimodality is TAA-jitter-linked (established)" — the cited run ablated the ENTIRE
  post stack, pre-voxbocc; downgraded to hypothesis (Q5 decides).
- capSuspect "selection-bias" worry — the filter is DEAD (falls back to all frames); defect
  is no-filtering, not bias.
- "nested shadow texel grids align ⇒ 6-level storm frames" — round-snap flip boundaries
  interleave exactly (proof in 17); storm survives via always-ticking fine levels.
- visTris as a mesh-load metric wherever vox runs — counts BRICKS for vox clusters
  (word7&0xff = brickCount); aerial "44.7k tris" = tile brick count, mesh emit ZERO.

**Policy red list (quality-rejected; instruments only):** aggdist=60 ("massive voxels"),
voxlodk 0.7/0.85, leaflodk<0.4, voxtaucap>12, ftcell>0.75, instminpx raises, dpr<1.5,
voxdither, impostor/billboard distance pop, GTAO/contact/bounce sample cuts, cloud temporal
reprojection.

---

## 6. QUALITY-UP LEDGER — how surplus funds finer-for-longer LOD (doc 14 costing)

User ruling: perf wins are SPENT on quality, LOD engages FARTHER. Funding rule: spend only
after live p95 ≤ 15.5 ms at every measured pose with ≥1 ms slack held in reserve; each spend
is its own probe-gated step and must not re-open the gap. Priority = worst user-visible band
first.

| priority | spend | mechanism | cost e/o/a (ms) | gate / probe |
|---|---|---|---|---|
| 1 | **pertree-ring-octave** `?voxlodkring=2` (14) | scoped errorK=2 for per-tree crown pyramids ONLY (tiles untouched); 45–140 m bricks 6–12 px instead of 12–24; uses resident dead-L0 bricks — NO rebuild; shrinks the 140 m handoff jump | −1 / **−6 (±2)** / −0.5 | new flag A/B vs fresh default; gates: cost vs −6 est, unchanged tile cluster counters, far-band shots. NOT the global knob (`voxlodk=2` = +17.4 obl + extends tile L0 to 707 m — wrong shape) |
| 2 | **tiered-mid-tiles** (14) | second 32 m / 0.375–0.5 m-cell tile tier for the 140–300 m ring — the ONLY fix for the 24–45 px tile-L0 shelf (the worst user-visible coarseness; no τ/errorK knob reaches it) | −0.5 / **−3** / −1 | pre-build de-risk: `TREES=100000 EXTRA=ftcell=0.5` look+cost at the handoff; ring scope mandatory (memory); watch the doc-06/15 "256 MB cliff" (UNVERIFIED on this device — doc 21 probe 4) | 
| 3 | **voxtaucap 12→8** (14 = 04 §4.5) | every vox transition ×1.5 farther (runtime knob ≈ voxlodk 1.5 up to block granularity) | −3 / **−8..−14 (post-voxbocc UNKNOWN)** / ? | 04-P1 A/B + crops 60/100/140 m; ship only if funded and probe says ≤ +4 obl |
| 4 | **SGGX stochastic voxel normal** from the banked SPREAD word (19/02) | shading realism on far crowns, ~0 perf | ~0 | crops at 5 poses vs voxbead-v2 + user sign-off (jitter-stable noise) |
| 5 | **skyline-floater-prune + crownMinY-cap fix** (15) | bake-time connected-component prune of low-w fringe specks; fix the 2 m trunk-stub cap (latent floating-crown gap, ForestScene.ts:336) | ~0 | skyline/140–300 m crops + sign-off |
| 6 | **vox-shadow-splat** (17, world scene only) | far crowns + far-tile heads cast shadows (they cast NONE today) | −0.5..−2 world | after `?forestshadow` lands; peter-panning shotdiff + sign-off on the spend |
| 7 | **junction beauty** (deferred memory) | child-sized collar at branch→trunk junctions | unknown | explicitly user-deferred; re-open on request only |

Bookkeeping: the §2 ledger's surplus at mid estimates (~11 booked vs −13±3 needed) funds
priority 1 only after Q3 pins r; priorities 2–3 wait for Phase-2/3 wins or the RISK shelf.
Every spend re-runs the live milestone (Q3 form) before and after — the quantum math, not
the isolated median, decides whether a spend fits.

---

## Appendix — canonical anchors used by this plan

- Poses: eye [0,2,0] / oblique [0,40,40] pitch −0.35 / aerial (probe-fresh-stutter.ts:37-41).
- projK = cot(27.5°)·1473/2 ≈ 1414.8; 120 Hz panel ⇒ 8.333 ms slots.
- voxf2b K16: +4.9/+6.2/+16.0; voxwaves4 vs f2b: +0.1..0.2/−0.5/−4.7; voxbocc: −17.2/−6.0/−0.4
  (all vs `fresh-bead-v2-base` 36.0/43.1/16.8, recomputed in the reconcile passes).
- aggdist=60 diag: oblique ring ≈ 10.7 ms pre-voxbocc (4,759 blocks @ ~2.25 µs).
- voxlodk 0.7/0.85 diag: −6.8/−2.1 obl (REJECTED look, instruments only).
- errorK 1↔2 rested A/B: 38.5 ↔ 55.9 obl (~×1.7/octave).
- Milestone live: slots {1:28, 2:451, 3:121}, 210/600 >16.7, p95 25.0, cpu.submit 1.10.
- Thermal law: same build measured 55.8 vs 38.7 obl across eras — no cross-era deltas <2 ms.
