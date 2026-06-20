# VOXEL PERF BUILD — live state (2026-06-20, written pre-compaction for clean handoff)

The user is on a walk; I run the voxel-perf build AUTONOMOUSLY, no feedback needed. This doc + the
auto-loaded memory index are the post-compact continuation. Read this FIRST, then act.

## THE APPROVED BUILD LIST (all become DEFAULT — no URL param needed for the win)
1. **Front-to-back depth-bucket sorting** — ✅ DONE + MERGED to `nanite-raster` (`4bcb5f7`). Default-on.
2. **Hi-Z per-block occlusion cull** — 🔬 RUNNING (workflow `w4hr1c9lx`, worktree `nanite-voxoccl` @ 4bcb5f7).
3. **DAG-LOD voxels** (coarse far / fine near) — NEXT after occlusion. The one the user can SEE.
   - foundational, folded in: **drop the coverage dither → opaque**; **keep voxels a few px (do NOT shrink)**.

## AUTONOMOUS DIRECTIVES (user's standing orders this session)
- "whatever the result from current task, accept it and merge into nanite-raster" → DONE for f2b.
- Make every win DEFAULT (no URL param). Keep a `?flag=0` debug-disable ONLY for A/B.
- Start the next task on my own. For occlusion: **merge to nanite-raster if it is correct (conservative,
  NO over-cull holes) and a win**; refresh :5201; record the result here.
- Order is fixed: f2b → occlusion → DAG-LOD. Do NOT reorder/park anything.

## CURRENT STATE
- Production `nanite-raster` HEAD = `4bcb5f7` (Merge nanite-voxf2b). Main checkout `/Users/sebastian/IdeaProjects/fable-demo2`.
- `:5201` view server (worktree `nanite-voxel-view`) @ `4bcb5f7`. URLs: forest `?scene=forest&trees=200000&nanite=1`;
  forced-voxel tree `?scene=forest&trees=1&nanite=1&naniteleaf=1&forcevox=all`. (f2b is loss-exact = no visible change.)
- RUNNING: `w4hr1c9lx` Hi-Z occlusion (script `$CLAUDE_JOB_DIR/tmp/voxel-hiz-occlusion.mjs`). Harness re-invokes me on completion.

### WHEN `w4hr1c9lx` COMPLETES (do this):
1. Extract its result (read the .output via python json, NOT the raw transcript). Fields: `audit` (premise-audit
   verdict — if `wrong-lever`/`halted`, it built nothing → relay + reconsider, do NOT force a dead lever),
   `allHold`, `history[].{noHoles,cheaper,notSlower,cullStats}`.
2. If conservative-correct (`noHoles=true`) AND a win (cheaper, not-slower): commit specific files in
   `nanite-voxoccl` (NEVER `git add -A`), merge `nanite-voxoccl` → `nanite-raster`, refresh `:5201`, record here.
   If it over-culls (holes) or net-loses: do the PREMISE-AUDIT (params/metric/upstream — e.g. HZB freshness,
   cull inequality direction, mip level) before concluding; iterate or surface honestly. Default-on, no param.
3. Then TEE UP DAG-LOD (item 3). See below.

## DAG-LOD (item 3) — how to build it RIGHT (the user un-parked this; do NOT park it)
- FINDING (file:line verified, memory `voxel-single-resolution-no-dag-lod`): voxels are SINGLE fixed-resolution —
  `registerVoxelHead` (GeometryRegistry.ts ~1041-1138) attaches a DEGENERATE always-cut DAG (ownError=0,
  childCount=0). The cull's `makeTraverse` (NaniteCull.ts:532-650) ALREADY does error-driven cut — zero kernel
  change needed once voxels carry real per-level error + children.
- 5-STEP: (1) VoxelizeCrown emits a voxel MIP pyramid (2× downsample, OR occupancy, mean normal/density/albedo,
  parent↔child links — clean TREE, each fine block ONE coarse parent). (2) VoxelBrick: NO layout change (per-brick
  half word8 already supports bigger coarse bricks; word7 bits10-15 = LOD level). (3) registerVoxelHead: real
  multi-level DAG — ownError = the SAMPLING half-cell `cellSize·0.5·2^L` (NOT the brick half-extent — that
  under-details), crack-safe parent/child contract (child.parentError==parent.ownError, child.parentSphere==
  parent.ownSphere, parent sphere CONTAINS children; run validateDagHierarchy), roots=coarsest. (4) NaniteCull:
  ZERO kernel change. (5) WorldRegistry: keep transitionDist boundary; budget ~1.33× bricks.
- THE CRITICAL CONSTRAINT (user, emphatic): voxels are a FEW px (UE "near pixel-sized" ≠ 1px). Do NOT shrink.
  CLAMP the coarsening so voxels stay ~a few px, NEVER over-coarsening into solid blobs (the prior nanite-voxlod
  attempt over-coarsened → coarse bricks paint big SOLID footprints → far raster MORE expensive). Tune the error
  metric / cut so the coarsest level is still ~a few px. Machinery exists on branch `nanite-voxlod` (uncommitted)
  + script `$CLAUDE_JOB_DIR/tmp/voxel-dag-lod.mjs` (REVISE: clamp-at-a-few-px, and it must have a PREMISE-AUDIT stage).
- DROP the coverage dither (UE uses stochastic NORMALS + TSR, not coverage dither — sourced, memory
  `voxel-foliage-autonomous-build` reframe). The dither was a workaround for non-pixel-sized voxels.

## KEY TECHNICAL CONTEXT (sourced, do not re-derive or re-guess)
- TWO overdraws: SHADING (already ~1/px via the vis-buffer two-pass resolve — AT THE FLOOR, do NOT chase) vs
  RASTER/DEPTH (how many bricks do the atomicMax election per pixel before nearest wins — THE target).
- UE collapses raster overdraw via OPAQUE + front-to-back depth-buckets + per-pixel SOFTWARE early-Z (Epic doc:
  "sorted into depth buckets, rasterized front to back to gain some benefits of early Z"). We HAVE the early-Z gate
  (NaniteVoxelRaster.ts:567-568) — f2b activated it by ordering.
- N (per-pixel voxel brick overlap) MEASURED ≈ 5.2-6.2 wins/px (LOWER bound) in the distant canopy; ~0 near/eye-level.
  Method: `nanite.voxBrickWrites` ÷ covered foliage pixels, steep-down into distant canopy.
- f2b LESSON (the premise-audit win): bucketing by NDC-z was the BUG (perspective-compresses the far field → whole
  canopy in 1 bucket). LINEAR view-space depth + adaptive [dMin=transitionDist, dMax=lodDist] range fixed it. K=8
  (?voxf2bk). voxBrickWrites −1.4-1.75×, loss-exact, not-slower; win BOUNDED by the weak occlusion cull (item 2).
- gpuWall is THERMALLY NOISY at 200k retina → prefer DETERMINISTIC counters (voxBrickWrites); if timing, use
  interleaved/bracketed thermal control + MIN-over-replays. Canonical: scene=forest trees=200000, retina 2268×1473
  (?dpr=1.5 from 1512×982), voxdither=0 opaque, distant-canopy steep-down for overdraw.

## PROCESS (now enforced — the user patched this in)
- **Repo `CLAUDE.md` (always loaded) holds the "GO UP A LEVEL" premise-audit directive.** On ANY negative/
  disappointing/"can't" result: do NOT vary the method — go UP a level, put the CONTEXT (params, metric, upstream
  structure, the "given") on trial; it's the prime suspect. (This caught the f2b NDC-z bug.)
- **EVERY Workflow I author MUST have a PREMISE-AUDIT stage** (before implement + a re-check before any drop) AND
  embed the GO-UP directive in agent prompts — subagents do NOT inherit CLAUDE.md or memory. The occlusion + DAG-LOD
  workflows must have it. See `docs/perf-runs/NEXT-AGENT-CYCLE-PROMPT.md` §5.
- **NEVER park/drop/reframe/redirect work without surfacing it for the user** (memory `surface-decisions-never-park-silently`).
- GPU: only ONE GPU agent at a time (never parallel — dilutes). Workflow judges visually, I don't view shots.
- Worktrees: NEVER `git add -A` (commits node_modules/.cache symlinks → ff clobbers node_modules); stage specific files.

## FILES / WORKTREES / SERVERS / SCRIPTS
- Worktrees: `nanite-voxoccl` (occlusion, running), `nanite-voxf2b` (f2b, merged), `nanite-voxlod` (DAG-LOD machinery,
  uncommitted, needs clamp-fix), `nanite-voxel-view` (:5201), `nanite-voxel-foliage` (my isolated session cwd).
- Scripts `$CLAUDE_JOB_DIR/tmp` (=/Users/sebastian/.claude/jobs/5f4bc287/tmp): voxel-hiz-occlusion.mjs (running),
  voxel-f2b.mjs (done, linear-adaptive), voxel-dag-lod.mjs (needs clamp-at-a-few-px revision + premise-audit),
  voxel-coverage-temporal-impl.mjs (SUPERSEDED — coverage dither, do NOT use; UE uses normals+TSR).
- Memory index (auto-loaded): voxel-foliage-autonomous-build (build log + backlog: world-scene voxels, NaniteView
  unification, legacy-binding cleanup are queued AFTER the perf build), voxel-single-resolution-no-dag-lod,
  interrogate-constraints-lift-a-level, surface-decisions-never-park-silently, nanite-perf-canonical-config-and-baseline.
- POST-PERF BACKLOG (user-noted, after the build list): (1) enable voxels for the WORLD scene (not just forest);
  (2) update NaniteView debug views to match the main renderer; (3) strip NaniteView-only legacy bindings.
