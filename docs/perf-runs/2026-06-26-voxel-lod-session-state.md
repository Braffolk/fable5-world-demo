# Voxel LOD + raster perf — session state (2026-06-26, written for compaction/rehydration)

Read this FIRST to rehydrate. Branch of work = **`nanite-voxdaglod`** (worktree
`/Users/sebastian/IdeaProjects/fable-demo2/.claude/worktrees/nanite-voxdaglod`), off production `nanite-raster`.

## North star
Solid 60fps (p0.05 ≤16.6ms) in the FOREST under a MOVING camera, ZERO *visible* quality loss (methods free,
perceived image must match). The voxel-foliage path is the current frontier.

## The arc this session (newest = most important)

### 1. Structural research — `docs/perf-runs/2026-06-25-nanite-voxel-vs-ours.md` (23-agent workflow)
The ~10x voxel gap vs UE5 is STRUCTURAL (work emitted), not params/HW (HW only ~2-3x; UE voxelizes/rasters on the
GPU). Two portable causes: (Rank-1) no distance-LOD — degenerate always-cut DAG; (Rank-2) solid-AABB
over-election ignoring the baked occupancy mask. Memory: `nanite-voxel-vs-ue5-structural-gap`.

### 2. ROOT CAUSE of every prior DAG-LOD botch (premise-audit win)
The "locked" metric `ownError = cellSize·0.5·2^L` (errorK=0.5) pins the WHOLE LOD ladder into the NEAR field
(finest ~2m, coarsest-of-4 ~17m), BELOW the 35m mesh→voxel handoff → the entire [35,2000]m band rendered at ONE
coarsest level (the "semi-2-levels, no true dropoff" the user kept seeing). Shape (2^L) right, ANCHOR wrong.

### 3. The implement build (committed on `nanite-voxdaglod`, final sha 42f3308; tsc-clean)
Replaced the metric with a BAND-ANCHORED ladder `ownError(L)=anchorL0·errorK·2^L`, `anchorL0=transitionDist·tau/
projK`, 7 levels spanning [35,2000]m, + Rank-2 occupancy MASK gate (not literal DDA — per-pixel DDA fights the
per-brick depth-key + f2b ordering). Default-on. Flags added:
- `?voxlod=0` — old single-level degenerate DAG (Rank-1 off). `?voxocc=0` — old solid-AABB (Rank-2 off).
- `?voxlodk=` — anchor MULTIPLIER, **bigger = finer** (default 1). **User: `voxlodk=8` looks good on the single tree.**
- `?voxlodlevels=` (default 7), `?voxlodsparse=`/`?voxlodshell=` (default off, refuted-theory levers), `?voxdither=1`
  (opaque is default), `?voxao=0` (NEW — flatten the per-voxel N·L sun shading; NOT a perf lever, just look).

### 4. RESULT of eyeballing the build (user)
- ✅ It COARSENS now — true far-field dropoff works for the first time (the anchor fix landed).
- ❌ At voxlodk=1 it was WAY too coarse (big flat rectangles). `voxlodk=8` fixes the look on the single tree.
- ⚠️ Flat solid rectangles = Rank-2 occupancy carve self-disables on dense bricks (OCC_MASK_FULL=48) + OR-merge
  densifies fill 0.17→0.78 (far reads denser/darker). Not yet addressed.

### 5. CPU 600% on voxlod=1 — RESOLVED, red herring
Single tree voxlod=1 briefly hit ~600% renderer CPU. Investigations (agent a78dc0a + workflow wdsm7aysw) proved:
the voxel build is ONCE at boot (no workers in the forest scene; DagWorker is terrain-only), per-frame JS is
byte-identical to baseline. The 600% was a ONE-TIME shader/pipeline compile (bigger voxlod=1 shader) that "stopped"
after compiling. NOT a per-frame cost. Ignore it.

### 6b. UPDATE 2026-06-26 pm — atomic theory REFUTED, premise overturned, cost UNLOCALIZED (workflows wlbc8kgla → wcrjath77)
- `wlbc8kgla` ranked #1 = a single-global-address debug atomicAdd (NaniteVoxelRaster.ts:962, the `voxBrickWrites`
  counter). I gated it behind `?voxwrites` (default OFF; build-time TSL conditional ⇒ byte-identical removal).
  **USER MEASURED: zero delta.** REFUTED. (User's point: atomics have been ~free for a decade; not a 25ms cost.)
- `wcrjath77` (localize workflow) overturned the standing premise: **"the 33ms is in the SW raster loop" was never
  once per-pass measured.** Arithmetic ceiling: ≤~77M iterations (caps + strided loop verified to codegen) ⇒ ~3-5ms
  at any sane per-iter cost. **Work-COUNT cannot be 33ms.** Surviving hypotheses: (a) raster IS home but
  **occupancy-collapse / memory-stall** bound (≤19-36 WGs = ≤~4,600 threads close-up, latency unhidden), NOT work
  count; (b) **cull over-emission** (brick/cluster count secretly jumps close-up); (c) **a different pass**.
- RED-TEAM caught a measurement bug: `voxf2b` defaults ON ⇒ symptom URL dispatches `nanVoxScatterB0..B15` (16 depth
  buckets), NOT bare `nanVoxScatter`. Must sum the scatter FAMILY (`/^c\.nanVoxScatter/`). New suspect: the F2B
  machinery itself (16 serial dispatches + barriers + `nanVoxClearBins` + `nanVoxOccPyr*` occlusion pyramid).
- **PENDING: user runs `window.__laas.measureFrames({frames:21,warmup:10,cooldownMs:50})` FAR vs CLOSE** (snippet
  sums scatter family + occpyr + compute/render totals + voxClusters). Decision table: scatter-family grows +
  voxClusters flat ⇒ Lever 1a (occupancy point-splat, UE5 ScatterVoxels: iterate 64 occ bits not the rectangle,
  splat voxel centers into the existing election @:962-966; footprint-independent; also the forest's ~10x lever).
  voxClusters jumps ⇒ cull. render_total grows ⇒ resolve. compute−family/occpyr grows ⇒ F2B machinery.
  **Do NOT implement any fix until the table localizes the cost.** measureFrames (drain-isolated) NOT the live HUD
  (vsync-ghosts at FAR). Workflow scripts: voxel-33ms-localize (wf_e52609e9-39f).

### 6c. UPDATE 2026-06-26 pm2 — MEASURED: the cliff is r.scene SHADING, not the raster (workflow wm00cqjqk)
USER ran `window.__laas.measureFrames` FAR vs CLOSE (drain-isolated). GROUND TRUTH:
- INSIDE CROWN (~30fps): gpuWall 42.2ms, **r.scene 37.49ms** (render pass), compute_total **0.52ms**,
  *SCATTER_FAMILY **0**, voxClusters 32.
- FAR (~100fps): gpuWall 11.2ms, **r.scene 7.93ms**, compute_total 0.72ms, voxClusters 32.
- ⇒ **The voxel RASTER (all compute) is ≤0.72ms — NEGLIGIBLE. The cliff is `r.scene` = NaniteResolve deferred
  SHADING, 7.93→37.49ms (+29.6ms, ~95% of the delta).** voxClusters=32 at both poses ⇒ NOT cull, NOT more geom;
  only COVERAGE changes. ~30ms to shade one tree's pixels = ~15-30x a normal deferred shade ⇒ a per-pixel
  PATHOLOGY in the voxel resolve.
- **THIS FALSIFIES the 5-month "raster is the bottleneck" framing for this case** (solid-AABB over-election,
  occupancy splat, atomic contention, F2B, occ-pyramid, one-WG-per-cluster, LOD coarsening — ALL ≤0.7ms here).
  The structural-gap memory targeted the RASTER; the single-tree blocker is in SHADING. (Forest may differ —
  needs the same measureFrames read on the real 200k forest; do that before trusting raster levers for it.)
- The resolve = `NaniteResolve.ts` (r.scene). Per-voxel AO/N·L term `voxao` at :241 (prime suspect). Other
  per-pixel suspects: GI/SH eval, sun shadow sample, material/normal decode, a divergent storage refetch of
  brick/voxel data, or a per-pixel loop/ray-march. NOT yet sub-localized.
- **PENDING: A/B at the close pose — toggle voxao/GI/shadow flags one at a time, re-read r.scene** (workflow
  wm00cqjqk = voxel-shading-localize, wf_0c37ebd7-bb1, produces the exact flag set + fix). Quick head-start
  probe: re-run snippet close with `&voxao=0`. Do NOT implement a fix until the A/B pins the contributor.
- measureFrames snippet (sums scatter family + occpyr, drain-isolated, drops capSuspect ghost frames) is in
  the convo / the red-team result of wcrjath77.

### 6d. UPDATE 2026-06-26 pm3 — r.scene cliff SUB-LOCALIZED + free fix shipped (workflow wm00cqjqk)
MECHANISM (proven, red-team-corrected): the +29.6ms r.scene cost = **per-covered-pixel voxel DECODE** — a
~5-6-deep SERIAL dependent storage-read chain (visBV→qVoxRasterRO→clusters.meshId→meshes.matClass→instances/
brickBase→voxelBricks.normal→tint→depthNode re-read; all device-memory `StorageBufferNode<uint>`, no texture
cache, NaniteResolve.ts:357-707) that LOW OCCUPANCY can't hide because the whole terrain shading subgraph was
compiled into the vox pass. Scales with COVERAGE (pixel count), not overdraw (single-winner vis-buffer, no
resolve overdraw). REFUTED by red-team: NOT divergence (close-up a cluster's pixels share voxId→coherent),
NOT demote-fullscreen-doubling (far=7.93 proves non-vox waves terminate), NOT shadow/GI (build-dead in forest:
ForestScene.ts:354-356 gi/csm/canopy=null ⇒ nanshadow/ablate=gi are NO-OPS). r.scene also includes the sky
atmosphere background (SunSky.ts:63) — common-mode, cancels in same-pose A/B.
- **SHIPPED (free, bit-identical): terrain guard** at NaniteResolve.ts:388 — `If(isT,…)` → `if (pass==='tri')
  If(isT,…)` (matches existing ROCK/BARK/LEAF guards :424). Strips the vox shader's only demote-forcing
  texture samples + ~14-sample graph ⇒ occupancy. = RUN 2.
- **PENDING A/B (close pose, guard live):** (1) base [vs orig 37.5 = terrain-guard win]; (2) base+`&voxao=0`
  [vs 1 = the voxao 3-deepest-read sub-chain]. Read snippet logs r.scene/gpuWall.
- **STRUCTURAL ENDGAME (the real fix, gated on A/B): raster-side G-buffer.** Write brick mean-normal(oct rg16)
  + tint(rgba8) at raster time (NaniteVoxelRaster.ts winner-store ~:966; RED-TEAM: brick record NOT in scope
  there — needs a Phase-A stash mirroring wgDensBits :633-634; inherits the atomicStore race, benign). Replace
  NaniteResolve.ts:685-706 chain with 2 coherent textureLoads. ~0.5-1 day. ALSO cuts the 200k forest r.scene.
  UE ref: docs/perf-runs/Nanite-UE5-shaders/extended-all-related-shaders/shading/NaniteExportGBuffer.usf +
  NaniteShadeBinning.usf (decode-once→shade-coherent; we re-decode per pixel = the generating flaw).
- Follow-on: collapse the 2 fullscreen resolve passes (tri −1000 / vox −999) into a coverage-binned compute.
- Workflow: voxel-shading-localize (wf_0c37ebd7-bb1).

### 6. THE EARLIEST FRAMING (workflow `wlbc8kgla`) — SUPERSEDED, see 6b then 6c
Single tree, camera ~2-3m (inside crown), `forcevox=all`: close-up fps 120→30 (~33ms) at **BOTH voxlod=0 AND
voxlod=1**. So it's the **BASE SW voxel raster**, PRE-EXISTING on nanite-raster, NOT the new pyramid/occupancy.
User: it's a code pathology (hundreds of squares should be ~free), it's FUNDAMENTAL, and probably a main thing
tanking the 200k forest. Do NOT hand-wave "big squares close = costly" or "non-issue because close".

VERIFIED base-raster code facts (NaniteVoxelRaster.ts, worktree):
- SW raster compute kernel; ONE workgroup per voxel CLUSTER; WG_RASTER=MAX_BRICKS_PER_CLUSTER=128 lanes.
- Phase A (1 lane/brick): projects brick → footprint AABB clamped to BRICK_MAX_EXT=64 (≤128×128 px/brick), :712-745.
- Phase B (128 lanes): SERIAL loop over the cluster's bricks (:925), each strided across lanes (:942). Per
  footprint pixel: relaxed GLOBAL LOAD `prevE=visPayloadV[px]` for EVERY brick-pixel even on loss (:956); on win
  atomicMax + atomicStore + **debug `atomicAdd(WRITE_CTR)` to ONE global addr (:962)**. `cand` = FLAT per-brick
  depth (:932,955) → billboards, solid AABB (empty corners painted).
- NO hierarchical/tile occlusion reject before the per-pixel loop. Only early-out = the per-pixel relaxed load
  (gates the ATOMIC, not the LOAD). f2b near→far buckets exist (`?voxf2b`).
- Magnitude puzzle: footprints CAPPED at ≤128×128 ⇒ naive fillrate shouldn't be 33ms for ~hundreds of bricks ⇒
  dominant cost UNCERTAIN: (a) low workgroup OCCUPANCY (single tree = few clusters = few WGs, GPU idle), (b) serial
  per-brick loop, (c) per-brick-pixel load fillrate, (d) atomic contention on hot center px, (e) debug atomicAdd,
  (f) far more bricks than "couple hundred". Workflow wlbc8kgla ranks these + gives the decisive measurement + fix.

## OPEN DECISIONS / PENDING (surface, don't park)
1. **band-anchor vs locked cellSize metric**: implementer OVERRODE the locked `cellSize·0.5·2^L`; 2 reviewers
   contest (band-anchor bakes projK at build → resolution drift; global const → large crowns may blob). The
   "too coarse at k=1" result is evidence the band-anchor absolute scale is off. Resolve empirically (voxlodk tune
   vs switch to cellSize metric which self-clamps to ~a-few-px). USER'S CALL.
2. Bake voxlodk=8 as default once the FOREST (not just single tree) is confirmed.
3. OR-densification → far darker/denser (visible quality); Rank-2 self-disable (OCC_MASK_FULL) + ~512 proj/brick
   build cost (maybe net-negative — the ?voxocc=0 A/B). LOD popping unmitigated (no cross-fade).
4. The base-raster fillrate problem (#6) — the active blocker.

## How to test (dev server runs from the WORKTREE, not the main checkout)
```
cd /Users/sebastian/IdeaProjects/fable-demo2/.claude/worktrees/nanite-voxdaglod && npm run dev   # vite :5173
```
- Single tree (see dropoff, fly toward/away): `http://localhost:5173/?scene=forest&trees=1&nanite=1&naniteleaf=1&forcevox=all&voxdither=0&voxlodk=8`
- Forest (real target): `http://localhost:5173/?scene=forest&trees=200000&nanite=1&dpr=1.5&voxdither=0&voxlodk=8`
- A/B flags: `&voxlod=0` (no pyramid), `&voxocc=0` (no occupancy gate), `&voxao=0` (flat shading).
- Counters: `window.__laas.stats.counters['nanite.voxClusters']` (workgroup count), `['nanite.voxBrickWrites']` (election wins).
- Canonical forest perf config: scene=forest trees=200000, retina 2268×1473 (?dpr=1.5), voxdither=0 (opaque).

## Process / standing rules (from CLAUDE.md + memory — agents do NOT inherit these)
- Research workflows: ≥5-6 rigorous stages, NEVER fed a focus area beyond the goal; premise-audit + "go up a level"
  embedded verbatim. Implement workflows: review panel + A/B + keep/drop, premise-audit gate.
- NEVER run heavy GPU/probes myself (thermal + context). User runs measurements; agents do code-only.
- Worktrees: NEVER `git add -A` (symlink clobber); stage specific files. Branch work lives on nanite-voxdaglod.
- Gate on the USER-OBSERVABLE (does far visibly coarsen / does fps recover), not "machinery correct".
- Surface decisions, never park. Premise-audit on every negative result (the anchor bug was found this way).

## Tasks (this session)
- RUNNING: workflow `wlbc8kgla` — base-raster fillrate diagnosis (the active problem).
- DONE: research `wf_9c47edb8` (→ 2026-06-25 doc); implement `w2jgplmxm` (→ 42f3308 on nanite-voxdaglod);
  CPU diagnosis `wdsm7aysw` + agent `a78dc0a` (CPU = one-time compile, resolved); agent `acbba4e1` (voxao flag).
