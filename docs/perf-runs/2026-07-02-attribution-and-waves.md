# 2026-07-02 (day session) — rested milestone, attribution, bead normals, vox waves

Post-compaction continuation of the 60 fps mission. Entry context: `2026-07-02-CONTINUATION.md`.
User note added this session: far crowns still read BLOCKY — "maybe round normals could help,
unless that kills perf". User also asked to RE-CHECK the backlog logic (late-context plans).

## 1. Rested final milestone (fresh-final-rested.json, machine rested overnight)

- LIVE moving 600 ticks: avg 27.25 p50 25.0 p95 33.4 max 34.3 — **0 frames >100 ms**
  (stutter fix HOLDS), but 584/600 frames >16.7 ⇒ NOT at target. Live sits at 2-4
  vsync periods @120 Hz (25.0 = 3 quanta).
- ISOLATED med/p95: eye 31.2/35.3, oblique 38.8/47.4, aerial 14.6/19.9.
- Gap to locked-60 (needs worst-pose isolated ≈25): oblique −14 ms (−36%), eye −6 ms (−20%).

### Bimodality discovery (per-frame arrays)
- Oblique alternates RUNS of ~4 frames: ~34-39 vs ~43-47.4. Aerial: floor cluster 8.7-11.9
  vs 14-22.8 (2.3× swing!). Eye varies ±3 with no clean split.
- If the periodic cost vanished: aerial med ~10, oblique ~35-36.
- NOT resolve lighting (persists under nandbg=flat). NOT shadow cadence at a static pose
  (shadow-clip levels re-raster only on texel-boundary crossings; static camera ⇒ VP
  bit-identical ⇒ cached). Suspects remaining: TRAA jitter (32-point Halton; jittered VP
  → borderline HZB/cut flicker on BIG far-tile clusters), GI probe updates, voxOccPyr /
  pyramid interactions. UNRESOLVED — check `?traa=0`-equivalent + GI flags next.

## 2. Attribution A/Bs (ordered same-session; candidates later ⇒ thermal bias AGAINST them)

| run | eye | oblique | aerial | verdict |
|---|---|---|---|---|
| default (rested) | 31.2 | 38.8 | 14.6 | baseline |
| `nandbg=flat` (no lighting/GI/backlight) | 29.9 | 39.3 | 17.6 | **lighting ≈ FREE** |
| `nanshadow=0` (NO shadow system at all) | 32.6 | 37.0 | 15.6 | **shadows ≈ FREE (0-2ms)** |
| noleaves (historic, cooler) | 14-17 | 12-14 | 10-13 | foliage = eye +15, obl +25 |
| `ablate=clouds+ao+bounce+bloom+taa` (2026-07-02 04:4x, vs bead-v1 35.5/44.1/17.4) | 29.3 | 38.1 | 17.1 | **post ≈ 6ms eye+oblique, 0 aerial** (mostly TAA+AO; AO self-describes as a near-flat 0.8 cue → tuning lever) |

| `leafcheap=all` (vs bead-v2-base 36.1/43.2/16.8) | 35.2 | 42.4 | 16.3 | **leaf DECODE ≈ FREE (<1ms)** — decode ceiling measured with a real candidate look |

**ATTRIBUTION COMPLETE (2026-07-02): lighting FREE + shadows FREE + leaf decode FREE +
post ≈ 6ms ⇒ the ONLY remaining foliage cost is RASTER COVERAGE (SW-raster walk/election
work). Eye 36.1 ≈ base ~15 + coverage ~15 + post ~6. Shade-binning is fully dead. The
levers are occlusion/emission (voxwaves, voxbocc, HZB quality), not shading.**

### Post-ablate side-findings (per-frame arrays, fresh-ablate-post3.json)
- **Bimodality ≠ TRAA jitter (REFUTED)**: with TAA off the oblique still alternates
  ~33-38 vs ~42-47 in runs of ~4 (+ 70/91ms spikes). Post stack exonerated entirely —
  remaining suspects: GI probe cycle, voxOccPyr/HZB interactions, GPU pipelining sawtooth.
- **Aerial pose-arrival ramp**: first 2-3 isolated frames 6.5-11.5ms then settle ~17 —
  stale-HZB over-cull right after teleport; the milestone's aerial "floor cluster
  8.7-11.9" is likely these transient frames, not a reachable steady state.

**CONCLUSION (recheck of the backlog's §4.6a):** the "shade-binning resolve" premise was
HALF WRONG — the lighting/shadow/GI per-pixel chain costs ~nothing. The frame =
base (~15) + foliage RASTER COVERAGE + foliage MATERIAL DECODE. Note `nandbg=flat` does
NOT test the decode half (albedo path still runs makeCtx + 3× vertex fetch for near
leaves) — that ceiling is measured by the new `?leafcheap=all` lever (pending).

## 3. Far-band blockiness diagnosis (crop of final-rested-oblique)

Zoomed crop shows the blocky look = **piecewise-constant shading per brick**: large flat
same-green quads (one baked normal + one albedo each) + a handful of distinct greens tiling
the far field like camouflage. Silhouettes are fine (voxcell carving works); it's the FLAT
FILL that reads as blocks. ⇒ the user's "round normals" instinct is correct and cheap.

## 4. Shipped this session (flags; all typecheck-clean)

1. **`?voxbead=k`** (default 0.6, NaniteResolve): per-pixel "round normal" — bend the vox
   shading normal toward normalize(wp − brickCenterWorld) (brick center = words 5-7,
   transformed by `instTransformPoint`; far tiles ride identity instances ⇒ pass-through).
   Ball-like within-brick gradients through the existing wrap+ambient. ~3 loads + ~20 ALU,
   vox pixels only. The classic SpeedTree crown-normal trick at brick scale.
2. **`?voxjit=a`** (default 0.12, NaniteResolve): world-anchored per-cell VALUE jitter on
   vox albedo — hash of floor(wp·1.4) (~0.7 m cells, motion-stable, no payload bits).
   Breaks the few-greens tiling.
3. **`?leafcheap=all`** (attribution lever, NaniteResolve): route EVERY mesh-leaf pixel
   through the ?resfar cheap path (tint × quad normal, NO makeCtx/gust/3-vert interp).
   Measured delta vs default = the exact ceiling of ANY leaf-decode optimization
   (incl. shade-binning) with a real candidate look.
4. **`?voxwaves=N`** (with `?voxf2b=1`, NaniteVoxelRaster.dispatchVoxel): chunk the K=16
   F2B bucket dispatches into N near→far WAVES with a voxOccPyr REBUILD between waves.
   This is the fix for the F2B postmortem's own "per-block-cull pre-seed is never
   realized" — the pyramid is mesh-only when the vox scatter runs (built from
   visPayloadV BEFORE dispatchVoxel), so vox-behind-vox (the DOMINANT oblique occlusion)
   was invisible to the block cull. Wave w+1 now tests against the canopy wave w elected.
   ZERO shader changes — the block cull picks up the refreshed pyramid automatically.
5. **`?voxbocc=1`** (NaniteVoxelRaster Phase A): PER-BRICK occlusion — each brick's own
   clamped bbox + front-slab key vs voxOccPyr (exact conservative idiom of the block
   test: min-pool + |0xff keep-on-tie; straddlers exempt). Kills individually-occluded
   bricks inside partially-visible blocks; pairs with voxwaves for brick-level
   vox-behind-vox.

6. **FarTiles WORKER SPLAT** (boot 36-42 s → target ~6 s): the per-tile splat +
   OCC_COVER post-pass extracted THREE-FREE to `FarTilesSplat.ts` (behavior-exact port,
   same accumulation order ⇒ bit-identical), fanned across ≤8 module Workers
   (`FarTiles.worker.ts`, typed-array transfer both ways); dense-emit + pyramid + prep
   stay on main (VoxelizeCrown drags three imports). `buildFarTilesAsync` in FarTiles.ts
   (deterministic tile order = sync path; sync fallback on worker error); ForestScene
   awaits it. Also fixes the USER-reported slow boot for interactive use.
7. **`?ablate=clouds,ao,taa,bloom,bounce`** (pre-existing, rediscovered): the post
   stack (volumetric cloud march, AO, GI bounce, TRAA) runs inside every gpuWall ever
   measured and was NEVER attributed. The pixel-law "fixed" terms (eye 13/obl 12/aer 5)
   may largely be post. `ablate=taa` also freezes the Halton jitter ⇒ discriminates the
   TRAA-jitter cut-flicker theory of the bimodality (visTris swings 4.0-4.9M live).

## 5. Pending measurements (thermal-ordered, all TICKS=0 COOLDOWN_S=45)

1. `LABEL=bead-v1` default config (bead+jit now default-ON; worker splat in) — gates:
   perf-neutrality, far-band screenshot (within-brick gradients? tiling broken?), boot
   time (~102 s → ~65 s expected), fartiles console log shows worker/emit split +
   IDENTICAL brick/cluster counts vs the last sync boot (bit-exactness check).
2. `EXTRA=ablate=clouds,ao,bounce,bloom,taa` — post-stack total (then singles as needed);
   `ablate=taa` alone for the bimodality/jitter theory.
3. `EXTRA=leafcheap=all` — the leaf-decode ceiling (eye pose is the money number).
4. `EXTRA=voxf2b=1` — F2B barrier-cost control at 200k forest scale (the 2026-06-26
   net-loss was measured at a 4k single-tree pose = pathological occupancy).
5. `EXTRA=voxf2b=1,voxwaves=4` — the vox-behind-vox occlusion gain (oblique is the money
   number). Then voxwaves=2 / voxbocc=1 permutations.

## 5b. Big-rock measurements (2026-07-02 04:50-05:3x, vs bead-v2-base 36.1/43.2/16.8)

| run | eye | oblique | aerial | verdict |
|---|---|---|---|---|
| `voxf2b=1` (control) | 41.0 | 49.4 | 32.8 | K=16 bucket chain alone: +5/+6/+16 — brutal, aerial DOUBLES |
| `voxf2b=1,voxwaves=4` | 41.1 | 48.8 | 28.1 | occlusion gain vs control: 0 eye, −0.6 oblique, −4.7 aerial |

**VERDICT: vox-behind-vox occlusion is ≈ DEAD at oblique** (the money pose) — even at
aerial (maximal stacking) the full 4-wave mechanism nets only −4.7, buried by F2B's +16
base cost. Not worth a cheap 2-wave build on this evidence. voxbocc (brick-granular,
no-F2B) still untested — cheap single run, low expectation.

**Premise-audit (go-up-a-level) after the waves null**: errorK already recalibrated
3→2→1 pre-compaction (VoxelizeCrown VOXLOD_CFG comment "2026-07-02b") — the far vox
ladder is at its calibrated coarsest; voxnear/leaflodk band already swept (45/0.4 ≈
60/0.25 neutral). With foliage decode/lighting/shadows all FREE and occlusion dead,
oblique 43.2 decomposes as base(?) + post 6 + foliage-coverage ~23. Even zero foliage
leaves base+post ≈ 20-22 vs the 25 target ⇒ **base (trunk SW raster) must shrink too**
— the unbuilt trunk far-field DAG coarsening (2026-06-26 handoff: 97% sub-pixel emit)
is back as the mandatory big rock. Fresh noleaves re-baseline queued to pin today's
base share before committing to that build.

## 5c. THE RING (2026-07-02 05:3x-06:1x): per-tree vox 60-140m is the oblique+eye whale

Fresh noleaves re-baseline (same era): base+post = eye 16.8 / oblique 15.4 / aerial 11.1
⇒ foliage coverage = 19.3 / 27.8 / 5.7. Counters: default-oblique visTris 1.27M < noleaves
2.1M (canopy HZB-kills trunks ⇒ mesh tris CHEAP at oblique) — the whale is vox election
work (9188 vox clusters at oblique).

Band split by ?aggdist=60 (tiles take the 60-140 ring): eye 26.1 (−10.0) / oblique 32.5
(−10.7) / aerial 16.6 (=). **The ring costs ~10ms at BOTH eye and oblique.** But the look
is UNSHIPPABLE (user: "dogshit, massive voxels" — merged plates + floating edge bricks
at 60m). aggdist=60 was a diagnostic, not a candidate.

Capture-without-the-chunk curve (?voxlodk=K, sub-calibration per-tree coarsening,
crowns stay per-tree; K also coarsens the fartile cut):
| K | eye | oblique | look |
|---|---|---|---|
| 1.0 (baseline) | 36.1 | 43.2 | calibrated (one brick ≈ τ px) |
| 0.85 | ? | ? | pending |
| 0.7 | 32.7 | 36.4 | crowns intact, bricks ~1.43× — DETECTABLE side-by-side, borderline |
| tiles@60 | 26.1 | 32.5 | REJECTED by user |

Remaining design if K-curve unsatisfying: TIERED TILES — 32m/fine-cell tiles for the
60-140 ring (worker splat has boot headroom), keeps aggregation's cluster-count win at
ring-appropriate brick size; floater risk (low-w edge bricks) needs a per-ring OCC_COVER.

## 5d. DIRECTION RESET (user, 06:1x) + the voxbocc bomb

USER RULING: **the quality bar is ABSOLUTE** — no optimization that loses visual quality
ships, ever. voxlodk 0.85/0.7 and aggdist=60 looks ALL REJECTED ("everything except 1.0
looks bad"; "we will not be making optimisations that lose in quality"). K stays 1.0.
Quality-trading knobs are attribution INSTRUMENTS only. Further: LOD levels should if
anything engage FARTHER than now (mid-band already too coarse for the bar) — perf wins
should FUND finer mid/far detail, not the reverse. Also: work autonomously, no questions.

**?voxbocc=1 (quality-IDENTICAL conservative cull): eye 36.1→18.9 (−17.2!!),
oblique 43.2→37.2 (−6.0), aerial 16.5 (=).** The eye whale was the vox field behind the
near mesh canopy at BRICK granularity (block-level cull + waves both missed it — wrong
mechanism at oblique: it was brick-vs-MESH, not block-vs-vox). Shots pixel-equivalent.
DEFAULT ON — committed by the USER themselves (cc73e88) while watching the session.

**NEW BASELINE: eye 18.9 / oblique 37.2 / aerial 16.5. Worst pose = oblique 37.2,
gap to locked-60 ≈ −11..13, all of it to come from QUALITY-NEUTRAL waste.**
Next: ultracode deep review (docs/deep-review/) → ranked quality-neutral path.

### Post-voxbocc LIVE milestone (fresh-voxbocc-milestone.json, 600 ticks, 06:5x warm)
- **live p50 = 16.7ms (= 60fps vsync quantum!), avg 17.96, p90/p95 = 25.0/25.1, max 26.0**
- 210/600 frames >16.7 (all sitting at the NEXT 120Hz quantum, 25.0); 0 >33ms; 0 longtasks; heap flat.
- vs pre-voxbocc rested milestone p50 25.0 / p95 33.4 — a full vsync quantum won across the board.
- Isolated same-run: eye 21.0/27.2, oblique 36.3/44.2, aerial 14.9/19.6 (consistent w/ 18.9/37.2/16.5).
- REMAINING MISSION, exactly: move the 35% of live frames at 25.0 down one quantum ⇒ oblique-ish
  isolated 36.3 → ~24-26, quality-neutral only. That is what docs/deep-review/00-MASTER-PLAN.md must close.

## 6. Notes / hazards

- **NEVER edit src/ while a probe is in flight** — vite full-reloads the connected page
  and the probe dies with "Execution context was destroyed" (killed ablate run #3,
  2026-07-02 04:20). Edits and probes strictly serialized.
- Probe EXTRA can't carry commas; PostStack ?ablate now splits on [,\s+]+ because
  laasUrl %2B-encodes '+' (two wasted runs before the parser fix).
- **bead-v1 → v2 (user: "sharp cones, not blobs")**: v1's per-brick radial field
  (wp−brickCenter) made every brick a bright-tip/dark-flank facet — high-frequency
  contrast, serrated-cone look, AO crevices amplifying. v2 = LOW-frequency crown field:
  horizontal radial from tree origin (A.xyz) + 0.55 up-tilt, 25% residual within-brick
  term. GTAO is exonerated as the CAUSE (it derives normals FROM DEPTH — Gtao.ts, so
  the bent shading normal never feeds AO), it only outlines brick cubes.

- Anthropic classifier outage stalled Bash mid-session; probes were queued behind
  read-only prep. All edits above landed with tsc clean.
- voxwaves adds pyramid-chain dispatches (14 levels, serialized) per extra wave — the
  bubble is the cost to beat; K=16 stays ONE histogram (bucket build is unchanged).
- Aerial/oblique BIMODALITY (§1) is worth its own hunt after the wave experiments — it's
  ~4-9 ms of periodic cost on the two worst poses.
