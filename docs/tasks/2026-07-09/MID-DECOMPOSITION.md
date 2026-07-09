# MID-DECOMPOSITION — why is SW-mid 13.3M at the census worst pose? (2026-07-09)

DIAGNOSIS probe, read-only on source. Decomposes `nanite.midTris` (RAW SW-mid triangle
appends/frame) at the census SW-mid worst pose into {crown-mesh 30-60 m, crown-mesh
0-30 m, bark/trunk + rest} using existing knobs only. 11 sequential fresh boots.

## Verdict (one paragraph)

**Suspect (a) dominates: 73% of mid (8.44M of 11.52M) is crown mesh in the 30-60 m
band, and it is back-loaded — the outermost 15 m (45-60 m) alone carries 5.22M.**
Suspect (b) — bark/tubes that never coarsen — is **refuted for MID**: with the handoff
pulled to 10 m, mid collapses to 48.6k, i.e. bark/trunk/terrain/understory contribute
~0.4% of mid (near trunks live in the HW-cluster path: `clhwTris` ≈ 2.49M, constant
across every mesh-band config). Suspect (c) — 0-30 m resolvable crowns — is secondary
at 3.03M (26%). **BUT the prescribed fix "steeper ladder via errorScale" saturates:**
the `leaflodk` headroom curve bottoms out at **9.46M** (−18%) by K=0.25 because the
crown ladder is too SHALLOW — its coarsest rung is only ~0.37-0.41× of LOD0 tris
(`VegLibrary.ts` CROWN_LOD_BROADLEAF measured [1.0, 0.70, 0.46, 0.41]; CONIFER
[1.0, 0.72, 0.51, 0.37]). errorScale only moves WHERE rungs engage; once every crown
in the band sits at the coarsest rung there is nothing coarser to elect. Reaching
~2-4M needs **coarser rungs to exist** (deeper bake-side ladder: λ-fractions down to
~0.1-0.15, i.e. the crown DAG-coarsen / stochastic-pruning arc) or pulling the
handoff (`voxnear=30` measures 3.08M — inside the target band — at voxel-fidelity cost).

## Raw numbers (avg of 12 counter ticks, static pose, steady state)

Pose `?cam=-595.4,299.1,999.2,2.5755,-0.0061`, internal 1821×1138 = 2.072 Mpx
(canvas 2560×1600, AUTO rscale 0.7115 — bit-matches the census run) for EVERY config.

| config       | midTris    | visTris    | clhwTris  | voxRoutedTris | hwTris  | splatFrags | mid/px |
|--------------|-----------:|-----------:|----------:|--------------:|--------:|-----------:|-------:|
| baseline     | 11,522,347 | 21,532,496 | 2,486,742 |       969,628 | 182,687 |     15,554 |  5.56× |
| crownlod0=1  | 31,517,636 | 92,960,027 | 2,675,662 |     1,027,010 |  29,228 |     62,058 | 15.21× |
| voxnear=45   |  6,305,176 | 13,643,688 | 2,435,359 |     1,079,925 | 155,095 |      6,446 |  3.04× |
| voxnear=30   |  3,081,575 |  8,897,339 | 2,394,694 |     1,160,142 | 145,908 |      1,191 |  1.49× |
| voxnear=10   |     48,614 |  1,794,924 |   251,492 |     1,320,507 |  36,936 |        107 |  0.02× |
| leaflodk=0.15|  9,462,725 | 17,243,639 | 2,024,584 |       971,861 |  89,056 |     13,936 |  4.57× |
| leaflodk=0.25|  9,464,374 | 17,638,828 | 2,198,105 |       977,316 |  97,691 |     13,739 |  4.57× |
| leaflodk=0.4 |  9,643,401 | 18,300,529 | 2,335,162 |       970,569 | 136,754 |     13,790 |  4.65× |
| leaflodk=0.7 | 10,530,606 | 19,787,895 | 2,427,392 |       979,898 | 148,231 |     14,967 |  5.08× |
| leaflodk=2   | 15,117,694 | 27,063,290 | 2,491,693 |       972,730 | 184,691 |     20,824 |  7.30× |
| leaflodk=4   | 19,791,870 | 34,753,909 | 2,492,042 |       968,372 | 183,937 |     35,937 |  9.55× |

`nanite.voxBrickWrites` = 0 in ALL configs (see anomalies). All boots rendered without
`__laas.error`; canvas 2560×1600 / rscale 0.7115 identical everywhere (res premise-check).

## Decomposition of baseline mid = 11.52M

From the voxnear staircase (Δmid when a band's crowns become voxels):

| component                                   | midTris   | share  |
|---------------------------------------------|----------:|-------:|
| crown mesh 45-60 m (baseline − voxnear=45)  | 5,217,171 |  45.3% |
| crown mesh 30-45 m (voxnear=45 − voxnear=30)| 3,223,601 |  28.0% |
| crown mesh 10-30 m (voxnear=30 − voxnear=10)| 3,032,961 |  26.3% |
| bark/trunk + terrain + understory + <10 m   |    48,614 |   0.4% |
| **total (= baseline, exact by construction)** | **11,522,347** | 100% |

- **The band-depth curve is superlinear**: each further 15 m of mesh band costs more
  than the previous (3.03M for 10-30 m → 3.22M for 30-45 m → 5.22M for 45-60 m):
  annulus area/tree-count grows with distance while per-crown tris shrink slower than
  1/d² because the ladder bottoms out (below).
- **Premise-checks that each knob took effect**: voxRoutedTris rose monotonically
  0.97M → 1.08M → 1.16M → 1.32M as voxnear came in (crowns moved mesh→voxel);
  splatFrags fell 15.5k → 6.4k → 1.2k → 0.1k (sub-px mesh tris live in the far band);
  hwTris fell in step. voxnear=10 also dropped clhwTris 2.49M → 0.25M — the handoff
  hands off the WHOLE tree (bark included), which is exactly why the voxnear=10
  residual is the honest bark-in-MID bound (bark contributes ≈0 to mid even at
  baseline, since near trunks route to the HW-cluster path, not SW-mid).
- **crownlod0=1 context**: forcing crowns to LOD0 explodes mid to 31.5M (vis 93M) —
  the shipped crown-LOD ladder already saves ~20M (2.7×) at this pose.

## The leaflodk headroom curve — and why it saturates

Code semantics (premise-audit, confirmed empirically): in BOTH
`src/nanite/BuildAggregateDag.ts` (L77-87) and `src/nanite/BuildCrownLodDag.ts`
(L60-66), **errorScale < 1 pulls coarser rungs NEARER (fewer tris); > 1 pushes detail
farther (more tris)**. The task brief's `leaflodk=2/4` direction was inverted — those
configs measure the fuller-crown ceiling (15.1M / 19.8M). The headroom batch is K<1:

```
leaflodk:   4      2      1(base)  0.7    0.4    0.25   0.15
midTris:  19.79M 15.12M  11.52M  10.53M  9.64M  9.46M  9.46M   ← SATURATES
```

K=0.25 → 0.15 moves mid by 0.02% — the curve is FLAT below ~0.25. Cause (one level
up, in the bake): the crown ladder has only 4 rungs and its coarsest rung still
carries ~0.37-0.41× of LOD0 triangles (`src/vegetation/VegLibrary.ts`
CROWN_LOD_BROADLEAF/CONIFER, measured fractions in the comments; λ floors chosen
spruce-safe). `?leaflodk` (→ `crownLodErrorK`, `crownLodOwnErrors`) only rescales the
ENGAGEMENT distances (CROWN_ENGAGE_FRAC [0.28, 0.56, 0.84] × transitionDist); once
every crown in the band is at rung 3 there is no coarser geometry to elect.
Cross-check: saturated 9.46M / crownlod0 31.5M ≈ 0.30 — right at the coarsest-rung
fraction (mixed with residual splat/HW absorption). **Ladder-steepening alone can
never reach 2-4M; the ladder needs deeper rungs (bake-side), ~λ-fraction 0.1-0.15.**

Note: `?leaflodk` also feeds the aggregate/far-tile ladder (`setAggLodErrorK`,
default 0.4) while the crown ladder's own default is 1.0 — the K<1 runs coarsen both,
so the crown-attributable saturation floor is if anything OPTIMISTIC (some of the
−18% came from the aggregate ladder).

## What DOES reach the 2-4M reference band

| lever                        | midTris at pose | cost/character |
|------------------------------|----------------:|----------------|
| voxnear=30 (handoff at 30 m) |          3.08M  | crowns 30-60 m become voxel bricks — fidelity regression in the most-looked-at band; voxel path has its own no-LOD disease (memory: nanite-voxel-vs-ue5 gap) |
| deeper crown ladder (bake)   | est. ~3-5M      | add rungs below λ-fraction ~0.4 (the stochastic-pruning design `docs/tasks/2026-07-08/DESIGN-crown-lod-stochastic-pruning.md` is exactly this arc); keeps mesh look |
| leaflodk alone               |     9.46M FLOOR | refuted as a standalone fix |

Estimate for the deeper ladder: a rung at ~0.15× engaging where rung-3 engages today
would scale the saturated band contribution (~6.4M sitting at rung ~0.37-0.41) by
~0.4 → ~2.6M band + ~3.0M near band (itself then partially coarsened) ⇒ plausibly
lands 3-5M without moving the handoff.

## Anomalies surfaced

1. **leaflodk direction inverted vs the task brief** — K>1 = MORE tris (code + measured).
   The batch-2 K<1 runs supply the intended headroom curve.
2. **Static baseline 11.52M vs census peak 13.33M (−14%)** — the census number is a
   moving-camera flythrough peak (in-flight streaming/HZB churn + pose rounded to
   0.1 m / 1e-4 rad); static steady-state at the rounded pose reads lower. Same pose,
   same internal res (2.072 Mpx, rscale 0.7115) corroborated — the harness reproduces
   the census within the moving-vs-static gap; all decomposition deltas are against
   the static baseline (apples-to-apples).
3. **`nanite.voxBrickWrites` = 0 in every config** despite voxRoutedTris ~1M — the
   counter is ALSO 0 in the census baseline itself (summary "vox brick" peak = 0), so
   the brick-write readback is not populated in this config lineage (pre-existing,
   not this harness). voxnear premise-checks used voxRoutedTris + splat/hw instead.
4. **visTris ≫ Σ(layers)** (baseline: 21.5M vs mid+clhw+voxR+hw ≈ 15.2M): expected —
   `visTris` is Σ full cluster triCounts at submit; the SW classifier then drops
   backface / off-screen / sub-px tris (HUD.ts L25-37 semantics). The ~6.3M residual
   is conservative-cull waste, worth its own look but out of scope here.

## Reproducibility (harness)

- Headless Playwright via `tools/launch.ts` (`channel:'chromium'`, headless, recipe
  cache `.cache/webgpu-flags.json`), dev server on :5173 (pre-existing, left running).
- Viewport **1280×800 @ deviceScaleFactor 1** + `?dpr=2` → native 2560×1600 → AUTO
  rscale caps to 2.07 Mpx internal — bit-matches `census-baseline.json` meta
  (rscale 0.71151247, mpx 2.072298). Same res every config ⇒ same τ anchor
  (projK ∝ renderHeight) ⇒ raw midTris comparable across configs.
- Base URL: `http://localhost:5173/?scene=world&nanite=1&dpr=2&nanodisp=1&clhwmax=32&grass=0&nanshadow=0&cam=-595.4,299.1,999.2,2.5755,-0.0061`
  (`?cam` pins the camera — `Params.parseCamString` 5-part form verified; no flythrough).
- Per config: fresh boot (voxnear/leaflodk are bootcache-key knobs — DAG rebuilds
  observed, boots up to ~4 min), wait `__laas.ready` (6 min timeout), re-assert pose,
  settle 180 frames, then average counters over 12 samples spaced 20 frames apart
  (readback cadence = 15 frames). Counters from `window.__laas.stats.counters`.
  Sequential boots only; no other GPU load.
- Scratch probes `tools/probe-mid-decomp.ts` / `-2.ts` — deleted after the run.

## POST-DEEPER-LADDER (2026-07-09) — crown ladder 4→6 rungs, CACHE_REV 6

The crown-LOD ladder was deepened **4→6 rungs** (`VegLibrary.ts`: broadleaf coarsest
now λ=0.25 ≈ **0.145×** LOD0; conifer coarsest λ=0.55/μ0.22 ≈ **0.13×** — vs the old
4-rung floors 0.41×/0.37× that SATURATED mid at 9.46M). Re-measured at the SAME census
worst pose, SAME harness, SAME internal res (**1821×1138 = 2.0723 Mpx, rscale
0.71151247 — bit-matches the old baseline** for BOTH configs ⇒ raw midTris directly
comparable). Fresh boots (CACHE_REV 6 rebuilt the crown DAGs; both booted clean, no
`__laas.error`).

| config                  | midTris   | visTris    | clhwTris  | voxRoutedTris | hwTris  | splatFrags | mid/px |
|-------------------------|----------:|-----------:|----------:|--------------:|--------:|-----------:|-------:|
| deeper-ladder baseline  | 6,396,004 | 13,416,213 | 2,441,807 |       978,487 | 154,892 |      4,688 |  3.09× |
| deeper-ladder leaflodk=0.4 | 3,710,555 |  9,027,767 | 2,192,581 |       972,259 |  98,368 |      2,123 |  1.79× |

### Delta analysis

- **Baseline (errorScale=1) 11.52M → 6.40M = −5.13M (−44.5%).** The deeper ladder halves
  SW-mid at the default engagement schedule without touching the handoff — but at
  errorScale=1 it lands **6.40M, ABOVE the 3-5M reference band** (~2× the ~3.3M
  projection). Deepening the rungs alone, at default engagement, does NOT reach the band;
  the extra headroom is real but only fully spent once errorScale pulls the new coarse
  rungs nearer.
- **leaflodk=0.4 9.64M → 3.71M = −5.93M (−61.5%).** This **reaches the 3-5M band (3.71M,
  mid/px 1.79×)**. Decisively — the old 4-rung ladder's K<1 curve SATURATED at 9.46M
  (K=0.25→0.15 moved mid 0.02%); the 6-rung ladder's coarse rungs now exist to elect, so
  the same errorScale=0.4 point drops from 9.64M → 3.71M. **The saturation floor is
  broken** — the errorScale lever is live again below the old wall.
- **Premise-checks that only crowns moved (bark/trunk/voxel paths unchanged):**
  `clhwTris` 2.49M→2.44M (baseline) / 2.19M (K0.4) — the near-trunk HW path is ~constant
  as expected; `voxRoutedTris` 0.970M→0.978M/0.972M — handoff untouched (both configs keep
  voxnear default). `splatFrags` collapsed 15.6k→4.7k→2.1k and `hwTris` 183k→155k→98k —
  coarser far crowns emit far fewer sub-px/HW tris, corroborating the mid drop is
  real crown coarsening, not a re-route. `visTris` 21.5M→13.4M→9.0M tracks in step.

### Verdict

The deeper bake-side ladder does exactly what MID-DECOMPOSITION predicted was needed
("the ladder needs deeper rungs, ~λ-fraction 0.1-0.15"): it **unsaturates** the errorScale
lever. At default engagement mid is 6.40M (−44.5%, short of the band); with a modest
errorScale pull (`leaflodk=0.4`) it lands **3.71M — inside the 3-5M band**, at −61.5% vs
the old K=0.4 and −68% vs the original 11.52M baseline. The ~3.3M single-config projection
was optimistic for errorScale=1 (measured 6.40M there) but is essentially met once the
now-live errorScale headroom is spent. No anomalies: both res checks bit-matched, no boot
failures, DAG rebuilds completed within the boot window.

- Scratch probe `tools/probe-post-ladder.ts` — deleted after the run.
