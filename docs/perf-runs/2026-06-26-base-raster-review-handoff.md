# Base Nanite raster review — HANDOFF / rehydration (2026-06-26, written pre-compaction)

Read this FIRST for the NEXT phase. The voxel-foliage arc is PAUSED (see `2026-06-26-voxel-lod-session-state.md`
for that full history + the structural-gap memory). The NEW target is the **BASE Nanite software compute
rasterizer** (the triangle path + the cull), NOT the voxel raster.

## THE PIVOT (measured, well-grounded — not a guess)
A direct `?noleaves` ablation (render TRUNKS/BARK ONLY — no triangle leaves, no voxel crowns) at the canonical
200k forest, fixed camera, interleaved A/B (0,1,0,1):
- **noleaves (base only): ~26–31 ms (~33 fps).**
- with voxel leaves: ~42–48 ms (~22 fps).
- foliage cost = **~15–16 ms** (consistent both pairs); **base = ~28–30 ms on its own = ~12–14 ms OVER the
  16.6 ms/60 fps budget WITH ZERO FOLIAGE.**
⇒ The base raster (trunk triangles + cull) is the dominant structural problem. Even perfect foliage floors us
at ~28 ms. So: stop optimising voxels; do a VERY in-depth dive of the BASE Nanite compute rasterizer vs the
UE5 reference shaders.

## ⚠️ THE DIRECTIVE FOR THE REVIEW — CLEAN SLATE, NO FED HYPOTHESIS (user's explicit, repeated instruction)
Do NOT push any hypothesis or presumption about WHY the base raster is slow. The exploration must be a clean
slate. The review derives the cost map from its OWN fresh measurement and the structural gaps from its OWN
reading of our code vs UE5 — NOTHING about the cause is seeded. (Feeding a hypothesis has poisoned every
investigation this session; the only ones that worked got apparatus + goal + measured FACTS only.) Below are
the apparatus + measured FACTS (legitimate grounding, not hypotheses) + the constraints. The user wants MAXIMUM
care/thinking when this review is actually authored (next turn, post-compaction).

## MEASURED FACTS that BOUND the review (facts, so it doesn't re-chase dead ends — NOT a cause hypothesis)
- The base raster is ~28–30 ms with 8.6M (eye-level ground_canopy) – 32.6M (aerial) TRUNK/BARK triangles +
  the cull. Target ≤16.6 ms.
- ⚠️ **The per-pass GPU profiler (`stats.passes`) is a HARNESS ARTIFACT for this pipeline** — the indirect-
  dispatched SW raster never appears in it; it inflated `r.scene` to 22 ms with ZERO real-frame effect. TRUST
  `gpuWall` + live-fps; attribute by ABLATION + counters ONLY.
- Lever-1 "SW-raster per-pixel ALU" (rcp-hoist + 2-FMA depth-plane + ≤4px-rect) was IMPLEMENTED + MEASURED →
  **REFUTED**: zero gpuWall delta because Tint→Metal ALREADY hoists the rcp / FMA-contracts the depth. So the
  cost is NOT naive per-triangle ALU. (Don't re-propose it; DO let the review find what IS the cost.)
- HW raster is <3% of the work (hwTris 150–230k vs 8.5–32.6M SW). Not the lever (measured).
- The heavy compute (per the full-pipeline review wxdbsv124) is **`nanTraverseAB` (the DAG-BFS cull) +
  the SW triangle raster (`world1`/`nanRasterWorld1`)** — but the cull-vs-raster SPLIT was NEVER cleanly
  measured (the rdbg/voxrdbg stage-split flags are CONFOUNDED by HZB feedback: gutting a raster stage changes
  next-frame emitted geometry 3–10×). Clean isolation needs geometry-neutral toggles, not rdbg.
- The forest is COMPUTE-BOUND (resolve/post/shadow hide under compute — cutting them does nothing in the forest).
- The trunk/bark has a NORMAL buildDag and coarsens fine; the trunk LOD-τ (NaniteCull τ=3px, `?loderr`,
  `simBandD`/lodWarp default-off) coarsening was flagged but NEVER measured (big-review Lever 2).

## APPARATUS (how to measure — a dev server is RUNNING)
- Dev server ALREADY running at http://localhost:5173 serving the worktree
  `/Users/sebastian/IdeaProjects/fable-demo2/.claude/worktrees/nanite-voxdaglod` (do NOT `npm run dev`; --strictPort).
- Canonical base config (trunks only): `http://localhost:5173/?scene=forest&trees=200000&nanite=1&dpr=1.5&noleaves`
  (add nothing else for pure base; retina 2268×1473 via dpr=1.5). With-foliage baseline: drop `&noleaves`, add
  `&naniteleaf=1&forcevox=all&voxdither=0&voxlodk=8`.
- `?noleaves` (NEW, ForestScene.ts:~86,165,171) = trunks/bark only (leaf head maxDist 0.001 + voxelization
  skipped) — the base-isolation lever for this whole review.
- `window.__laas.measureFrames({frames,warmup,cooldownMs})` → Frame[] (.passes WEAK-hint, .gpuWallMs, .counters,
  .capSuspect=drop). `getPose()` + pose setter ~src/main.ts:145. Reuse worktree harnesses: ab.mjs,
  forest-sweep.mjs, costmap.mjs, shotdiff.mjs.
- ⚠️ **MACHINE LOAD is CONSTANT** (same job, 2 CPU cores, steady ~8h from ~2026-06-26 eve) — a STEADY OFFSET,
  not drift. So MATCHED-condition A/B DELTAS/RATIOS (off vs on, identical pose/frames; or within-boot toggle)
  are RELIABLE; absolute ms is inflated by a constant; DO NOT wait to "cool". Boot once/config, teleport poses
  (200k boot is slow). The load is CPU not GPU → gpuWall is the cleanest metric.

## HARD CONSTRAINTS (TSL r184 → WGSL → Dawn → Metal, Apple M-series)
No 64-bit atomics, no subgroup/wave ops, ~10 storage buffers per stage, NO 3rd atomic storage buffer in one
compute kernel (hard Metal cliff → 15–17ms + broken writes), no fixed-function ROP/HiZ early-Z (build software
equivalents), one-WG-per-cluster is the current raster structure (may be changed if that's the lever). Watch
TSL codegen hazards (build a value inside the conditional subtree that consumes it). GPU measurement STRICTLY
SERIAL (parallel GPU crashes the machine); code/shader-reading agents may parallelise.

## CODE MAP (ours) + UE5 REFERENCE (read-only)
OURS (worktree src/nanite/): NaniteRaster.ts (SW scanline + HW pass, the `world1` single-pass 32-bit
depthKey24<<8|payload8 atomicMax election into visPayloadV + visBV side-store; ~:820-970 inner loop),
NaniteCull.ts (the DAG-BFS cull `nanTraverseAB`, instance/cluster cut, HZB sphereOccluded :~889, τ-LOD :818-831,
kSeedRoots :778), NaniteHzb.ts (HZB build + sphereOccluded), GeometryRegistry.ts (cluster/mesh records, lodDist
draw-envelope), BuildDag.ts/BuildAggregateDag.ts/Clusterize.ts (LOD build), NaniteFrame.ts (per-frame orchestration,
HZB build :437/476, submits). Intent docs: docs/NANITE-SPEC.md, NANITE-LOG.md (read the PERF LEDGER; records what's
measured-marginal — submit-batching, BFS-batching, temporal reuse, kSeedRoots removal, subgroup election; and the
REMOVED tiled/bin rasters D-N45/eecf046/a3a1059 = "frame is COVERAGE-bound, not submit-bound").
UE5 (docs/perf-runs/Nanite-UE5-shaders/, main checkout `/Users/sebastian/IdeaProjects/fable-demo2/docs/...`, 117
files): culling/ (NaniteClusterCulling, NaniteInstanceCulling, NaniteInstanceHierarchyCulling, NaniteHZBCull,
NaniteCullingCommon), compute_rasterizer/, NaniteRasterizer.usf/.ush, NaniteRasterBinning.usf,
NaniteRasterizationCommon.ush, NaniteRasterClear.usf, shared_definitions/, data_streaming/.

## INTENDED REVIEW SHAPE (to author next turn, with max care — still NO fed cause)
Like the wxdbsv124 full-pipeline review but FOCUSED on the base raster + cull, clean-slate:
1. Measure-map (serial GPU): cost-decompose the BASE path (`?noleaves`) by ABLATION + counters across poses —
   split cull (nanTraverseAB) vs SW triangle raster (world1) vs HW pass vs vis-buffer/HZB/clear, using
   GEOMETRY-NEUTRAL toggles (NOT rdbg, which is HZB-confounded). Find what actually owns the ~28-30ms.
2. Shader-review (parallel, NO GPU): per base-raster subsystem, deep UE5-vs-ours structural comparison
   (cull/instancing/hierarchy/HZB; SW rasterizer; clusterize/DAG-build; vis-buffer/election; frame/submit
   structure), file:line both sides, prioritised by the measured cost map.
3. Rank (measured-cost × improvability × quality-neutrality × portability), kill theoretical/cheap-stage ideas.
4. Confirm (serial GPU adversarial A/B) → 5. Fix+validate (behind flag, constant-load A/B + quality-diff).
Embed premise-audit + go-up-a-level VERBATIM; trust gpuWall not passes; serial GPU.

## SHIPPED THIS SESSION (on nanite-raster — do NOT undo) + standing rules
SHIPPED: voxf2b-off (single-tree 42→121fps), voxrecip (200k −18% worst pose), reskeep (general −3.9ms vista,
default-off ?reskeep=0), impostor-skip-under-forcevox, the band-anchor voxel DAG-LOD. nanite-raster HEAD ~1df23d2.
All refuted experiments REVERTED. STANDING RULES (CLAUDE.md + memory): premise-audit + go-up-a-level on every
negative result; NO fed hypothesis to workflows; trust gpuWall/live-fps not the per-pass profiler; serial GPU
only; constant-load matched A/B; verify the USER-OBSERVABLE not "machinery correct"; surface decisions never park;
worktrees NEVER `git add -A` (symlink clobber) — stage specific files; the user runs heavy GPU eyeballs.
