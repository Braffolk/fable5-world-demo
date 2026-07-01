# FRESH START — Nanite/voxel renderer perf. Handoff to Fable 5.

**Date:** 2026-07-01 · **Branch:** `nanite-raster` (HEAD `6a93dfb`) · **Author of everything below: a
LESS CAPABLE model (Claude Opus 4.x).** Read the next section before anything else.

---

## 0. READ THIS FIRST — who wrote this and what that means for you

You are **Fable 5**, the most capable model Anthropic has. Everything in this repo's nanite/voxel
renderer — the architecture, the shaders, the DAG build, the two-tier mesh/voxel design, the
measurement harnesses, every `docs/perf-runs/*.md`, every `D-N*` decision, this handoff itself —
**was produced by a materially weaker model over many weeks.** That model was good at grinding: it
built apparatus, ran hundreds of A/Bs, and banked some real wins. It was **bad at seeing one level
up.** It repeatedly (a) accepted a flawed premise handed to it, then blamed the implementation as
"not doable"; (b) mistook its own convergence for truth (five agents agreeing on a wrong frame); (c)
optimized inside a problem instead of questioning whether the problem should exist. The project's
own `CLAUDE.md` exists almost entirely to paper over these failure modes. **You should not need those
guardrails, but you should assume they mark exactly the spots where the previous work is weakest.**

**Your explicit mandate (from the user):** take a genuinely fresh look. **You are free — encouraged —
to question the base decisions.** Nothing here is load-bearing scripture:

- The **SCATTER rasterizer** (one-workgroup-per-cluster, `atomicMax` election into a 32-bit
  `depthKey24<<8|id8` word) — the weaker model tried tiled/sort-middle and voxel-bin rasters, measured
  them slower, and REMOVED them. That may have been a bad implementation, a bad measurement, or a
  correct-but-local conclusion. **Reopen it if your judgment says so.**
- The **two-tier design** (triangle mesh near, voxel bricks mid/far) — question whether voxels should
  exist at all. The entire voxel arc is currently PAUSED because a base-path problem dominated.
- The **DAG build** (hand-rolled Garland-Heckbert QEM in `BuildDag.ts`, separate aggregate builder in
  `BuildAggregateDag.ts`) — this is where the real structural bottleneck lives (see §3). The weaker
  model found the cause but shipped only a partial fix.
- The **shaders themselves** (`NaniteRaster.ts`, `NaniteResolve.ts`, `NaniteVoxelRaster.ts` — all TSL,
  compiled TSL→WGSL→Dawn→Metal). Read them against the UE5 reference and trust your own eyes.
- The **measurement conclusions.** Distinguish hard from soft below. Trust the *apparatus* and the
  *raw numbers*; treat every *conclusion* as a hypothesis to re-derive.

Do not, however, blindly re-run experiments the weaker model already measured to a clean negative
(§5 lists them with the numbers) — unless you think the measurement itself was flawed, in which case
say so and redo it properly. The goal is a fresh *frame*, not a fresh *grind*.

---

## 1. The goal (unchanged, and correct)

Solid **60 fps = p0.05 ≤ 16.6 ms frame time**, in the **FOREST scene under a MOVING camera**, with
**zero *visible* quality loss** (optimizations must be perceptually free — the rendered image must
match). This is a procedural WebGPU world (LAAS); the renderer is a from-scratch software Nanite
(compute-shader visibility-buffer rasterizer) living entirely in `src/nanite/`.

**Canonical benchmark config (measure here, nothing else counts):**
`scene=forest`, `trees=200000`, retina **2268×1473** via `?dpr=1.5`. The default probe config
(40k/720p) is NOT representative — the weaker model wasted weeks measuring the wrong load early on.

URL (pure base / trunks only):
`http://localhost:5173/?scene=forest&trees=200000&nanite=1&dpr=1.5&noleaves`
With voxel foliage:
`…&naniteleaf=1&forcevox=all&voxdither=0&voxlodk=8` (drop `&noleaves`).

---

## 2. WHERE THINGS STAND RIGHT NOW (pick up here)

`nanite-raster` HEAD `6a93dfb`, working tree clean. Recently shipped ON THIS BRANCH:

1. **`0f77edc` connected-bark cap-only** — the current frontier win. Trunk/bark geometry now closes
   its open base rings (`capBase` in the tube generator), which lets the QEM DAG actually coarsen the
   far field. Measured forest A/B (200k, dpr=1.5, constant-load matched): **~21–47% gpuWall**
   (eye 29.4→22.2 ms, aerial 13.6→10.8, oblique 37.9→20.2); far-field `visTris` collapse up to 31.6×,
   far-field shotdiff sub-pixel (zero visible loss). See §3 — this is a PARTIAL fix.
2. **`6a93dfb` voxel crown LOD fix** — corrected a regression (offset/over-coarse/collapse-to-one-square
   crowns) from a band-anchored voxel DAG ladder. **⚠️ Its forest perf cost is UNMEASURED** (it holds
   more far bricks). A fresh measure-map must include this.
3. Earlier banked wins (do NOT undo): `voxf2b`-off (single-tree 42→121 fps), `voxrecip` (200k −18%
   worst pose — per-brick float reciprocal, Apple has no HW int-divide), `reskeep` (general-scene
   −3.9 ms vista), AO optimization (~1.5 ms, the whole "post" cost was AO).

**The honest distance-to-goal:** after the connected-bark win, eye-level base is roughly ~22 ms under
a *constant background CPU load* that inflates absolutes ~30–60% (see §4). The real moving-camera
p0.05 gap is estimated **~9–11 ms over budget**. **Nobody has cleanly re-measured the current HEAD**
(cap-only + crown fix together) at the canonical config. **Step 1 for you is a fresh cost-map**, not
trusting the numbers above.

**Separate uncommitted work you should know about:** worktree `nanite-treeconnect` has a *fuller*
connected-geometry rework (real welded junction holes, not just caps) that measured ship-positive but
whose external *beauty* uplift was modest and whose welded-junction *aesthetic* attempt BLOBBED (a
generation bug, deferred by the user — not a limit). What shipped to `nanite-raster` is the cheaper
**cap-only** perf win. Junction beauty is DEFERRED-but-doable, not abandoned.

---

## 3. THE ONE STRUCTURAL FINDING that actually matters (and is only half-fixed)

This is the highest-value thing the weaker model found. **Verify it, then decide how far to push it.**

The renderer is **TRIANGLE-EMIT-bound** in the forest, and voxel foliage was ALSO
**no-distance-LOD-bound** — the *same disease* in two places: **the DAG never coarsens the far field**,
so the GPU rasterizes tens of millions of sub-pixel triangles / bricks that cover ~0 pixels but pay
full per-primitive setup.

- Base/trunk path: counters showed ~46M emitted trunk triangles with **~97% covering zero pixels**
  (0.03 covered-px/tri). Root cause: the trunk DAG's coarsest cluster floored at ~236 tris with NO
  coarser level, because **the tree bark is disjoint interpenetrating tubes with open base rings**,
  and the QEM simplifier locks every open-boundary edge (`BuildDag.ts:393-422`) → collapse stuck. The
  cap-only fix (§2.1) closes those rings → far field collapses ~4–30×. **But the *full* ~10× (down to
  8–32-tri roots) needs a follow-on DAG change** (single-root termination `BuildDag.ts:~1050`
  `active<2`, the 255-tri cap in `GeometryRegistry.ts:~106`). That follow-on is UNBUILT.
- Voxel path: voxels attach a *degenerate always-cut DAG* (`ownError=0`), so bricks are single-fixed-
  resolution. The convergent conclusion across ~10 voxel workflows: **the ONE real voxel lever is
  near-LOD brick-COUNT reduction** (a real MIP: fewer, coarser bricks with distance), which cuts cull
  + scatter + raster simultaneously. Every micro-lever (occupancy-tighten, DDA-discard, software
  early-Z, occluder/fill decouple, per-pixel raster-ALU) was measured dead or subsumed.

**Both reduce to: make the DAG/LOD actually coarsen the far field, capped so it never reads as
rectangles/squares.** This is where I'd point your fresh eyes first — but the framing "coarsen the
existing DAG" is itself the weaker model's frame. You might see a better one (e.g. is the disjoint-tube
generator the thing to fix? should trunks be voxelized far too? is there a unified far-field
representation for both?).

---

## 4. MEASUREMENT APPARATUS — read this or you will draw false conclusions

The weaker model's single biggest recurring error was trusting the wrong metric. The traps are real
and specific:

- **A dev server may already be running at `http://localhost:5173`.** Check before `npm run dev`
  (`--strictPort`). 200k boot is SLOW → boot once per config, teleport the camera between poses.
- **Harness:** `window.__laas.measureFrames({frames, warmup, cooldownMs})` → `Frame[]` with
  `.gpuWallMs`, `.counters`, `.passes` (per-pass), `.capSuspect`, `.refreshMs`. Pose getter/setter near
  `src/main.ts:145`. Reusable `.mjs` harnesses (ab, forest-sweep, costmap, shotdiff) live untracked in
  the worktrees.
- **⚠️ `stats.passes` (per-pass GPU profiler) IS A LIE for this pipeline.** The indirect-dispatched SW
  compute raster and voxel scatter **never appear in it**. It has repeatedly reported a pass costing
  20+ ms that, when gutted, changed the real frame by 0 ms. **Trust `gpuWall` + live fps. Attribute
  cost by ABLATION + deterministic counters ONLY**, never by `passes`.
- **⚠️ `rdbg`/`voxrdbg` stage-split flags are HZB-CONFOUNDED.** Gutting a raster stage changes the next
  frame's emitted geometry 3–10× (the occlusion cull reads last-frame depth). The infamous "per-pixel
  loop is ~90% of the raster" claim came from these flags and was later REFUTED. Clean stage isolation
  needs **geometry-neutral toggles**, not rdbg.
- **⚠️ Background CPU load is (was) CONSTANT** (a steady multi-hour job on 2 cores). It's a steady
  OFFSET, not drift, so **matched-condition A/B deltas/ratios are reliable; absolute ms is inflated by
  a constant.** Confirm whether that load is still present. Prefer within-boot toggles and
  thermal-invariant ratios. `refreshMs` reveals throttle (cool ~33 ms, hot 50–75).
- **GPU measurement must be STRICTLY SERIAL.** Parallel GPU work crashes the machine. Code/shader
  *reading* can parallelize freely; anything touching the GPU cannot.
- **The user runs the heavy GPU eyeballs.** Historically agents did code-only and the user ran probes.
  You have full tools — use judgment — but heed the serial-GPU constraint and gate on the
  **user-observable** ("does the far field visibly coarsen without turning into squares? does fps
  recover?"), not on "the machinery reports correct."

There is a WebGPU-Inspector MCP plugin available (capture/analyze frames) — historically treated as
ground truth over the homegrown harness for capture analysis.

---

## 5. HARD CONSTRAINTS (TSL r184 → WGSL → Dawn → Metal, Apple M-series)

These are platform facts, not preferences — but verify any that block you; some were asserted, not all
re-proven:

- **No 64-bit atomics, no subgroup/wave ops.** (This is why the election packs depth+id into one 32-bit
  word and why there's no wave-based shade-binning.)
- **~10 storage buffers per compute stage** (the resolve is near this ceiling).
- **NO 3rd atomic storage buffer in one compute kernel** — hard Metal cliff (→15–17 ms + broken writes).
  This killed the "keep exact depth as a separate atomic buffer" plan (D-N45); depth had to ride the
  election key. **Re-test this if you need it — it's the single most limiting asserted constraint.**
- **No fixed-function ROP / HiZ / early-Z** — all software-equivalent. (This is why "just add early-Z"
  doesn't obviously pay: a discarded fragment isn't free.)
- **TSL codegen hazards** — build a value *inside* the conditional subtree that consumes it, or codegen
  hoists it.

---

## 6. WHAT'S ALREADY BEEN MEASURED TO A NEGATIVE (don't re-grind these unless you distrust the measure)

Each of these was built and measured by the weaker model. The *conclusion* may be wrong, but re-running
the same experiment the same way will reproduce the same negative. If you reopen one, change the method.

- **Tiled / sort-middle raster** — built, measured **+11.7/+18.1 ms** vs scatter, REMOVED (`eecf046`).
  Conclusion: "the frame is COVERAGE-bound, not submit-bound." (Caveat: may have been an unoptimized
  impl at a degenerate camera.)
- **Voxel-bin raster** — likewise refuted + removed (`a3a1059`).
- **Per-pixel SW-raster ALU micro-opts** (rcp-hoist, 2-FMA depth plane, ≤4px rect) — 0 ms delta.
  Conclusion: Tint→Metal already does these. WGSL confirmed changed yet gpuWall flat.
- **HW raster reclassify** — HW is <3% of the work (hwTris 150–230k vs 8.6–32.6M SW). Not a lever.
- **Cull (`nanTraverseAB`)** — measured ~0.2 ms in the base path, HZB build ~0.05 ms. Earlier the
  weaker model called cull "the never-profiled #2 lever"; that was then REFUTED. (But note it flip-
  flopped on this — worth your own look.)
- **Voxel occupancy-tighten / DDA-discard / software early-Z / occluder-fill decouple** — all measured
  dead or subsumed by near-LOD. The occluder-and-fill are COUPLED (the solid brick AABBs double as the
  occlusion frontier), which defeats naive carving.
- **Atomic contention** — refuted repeatedly ("atomics have been ~free for a decade"); GUARDED == NAIVE
  FreePipe.

---

## 7. CODE MAP

**Ours** (`src/nanite/`, ~19k lines — biggest/hottest first):
- `GeometryRegistry.ts` (2629) — cluster/mesh records, LOD draw-envelope, voxel head registration.
- `VoxelizeCrown.ts` (1503) — crown → voxel bricks; the DAG-LOD ladder + the recent crown fix.
- `NaniteRaster.ts` (1469) — **the SW triangle rasterizer + HW pass.** `world1()` single-pass 32-bit
  `depthKey24<<8|id8` `atomicMax` election into `visPayloadV` + `visBV` side-store. Inner loop ~lines
  820–970. This is the base-path hot shader.
- `BuildDag.ts` (1240) — hand-rolled QEM simplifier; the far-field-coarsening bottleneck lives here
  (open-boundary lock ~393–422, termination ~1050).
- `NaniteVoxelRaster.ts` (1116) — voxel SCATTER raster (`kVoxScatter`), Phase A footprint projection +
  Phase B per-pixel election. The PAUSED voxel frontier.
- `NaniteCull.ts` (1082) — DAG-BFS cull `nanTraverseAB`, instance/cluster cut, HZB `sphereOccluded`,
  τ-LOD, f2b depth buckets.
- `NaniteResolve.ts` (940) — deferred shading of the vis-buffer (the general-scene cost; ~0 in forest).
- `WorldRegistry.ts` (981), `BuildAggregateDag.ts` (621, leaf-crown area-preserving DAG),
  `Clusterize.ts`, `NaniteFrame.ts` (575, per-frame orchestration/submits), `NaniteHzb.ts`,
  `NaniteShadow*.ts`, `NaniteFetch.ts`.
- Scene/test entry: `src/debug/ForestScene.ts` (canonical testbed; `?noleaves`, `?forcevox`,
  `?voxlodk`, trunk `maxDist=2000`), `src/main.ts` (`__laas` harness).

**UE5 reference** (read-only, `docs/perf-runs/Nanite-UE5-shaders/`, 117 files):
`NaniteRasterizer.usf/.ush`, `NaniteRasterizationCommon.ush`, `NaniteRasterBinning.usf`,
`culling/` (Cluster/Instance/HZB/Hierarchy), `compute_rasterizer/`, `shading/` (NaniteExportGBuffer,
NaniteShadeBinning — decode-once→shade-coherent, vs our re-decode-per-pixel). Use it to sanity-check
our structure, not as gospel — UE has 64-bit atomics and a different platform.

---

## 8. THE FULL PAPER TRAIL (read in this order, stop when you have the frame)

Minimize tokens — you likely need only the first two:

1. **`docs/perf-runs/2026-06-26-base-raster-review-handoff.md`** — the base-path pivot + apparatus,
   clean-slate. **Start here.**
2. **The memory file `base-raster-is-the-bottleneck.md`** (in the auto-memory dir) — the corrected,
   measured conclusion (triangle-emit-bound; connected-bark fix; the crux resolved). Denser than #1.
3. `docs/perf-runs/2026-06-26-voxel-lod-session-state.md` — the entire voxel arc (10+ workflows) if you
   choose to reopen voxels. Long; the payload is §6d–6j (convergence on near-LOD).
4. `docs/perf-runs/2026-06-25-nanite-voxel-vs-ours.md` — the structural gap vs UE5 (23-agent study).
5. `docs/NANITE-ROADMAP.md` + `docs/NANITE-SPEC.md` — the task DAG and durable `D-N*` decisions. **The
   ROADMAP's "YOU ARE HERE" banner is STALE** (dated 2026-06-20, still framed around the removed tiled
   raster). Trust the memory + handoff docs over it for current state.
6. `CLAUDE.md` — the project's standing process rules. Written to correct the weaker model's failure
   modes (premise-audit, go-up-a-level, surface-don't-park, never `git add -A` in worktrees). Worth a
   skim so you know the conventions; you may not need the crutches.

---

## 9. RECOMMENDED FIRST MOVES (a suggestion, not a script)

1. **Fresh cost-map at canonical config.** Boot `?noleaves` (base only) and the full config; ablate with
   geometry-neutral toggles; attribute `gpuWall` by counters. Confirm-or-refute for yourself: is the
   current HEAD triangle-emit-bound? where's the moving-camera gap? what did the crown fix cost?
2. **Decide the frame.** Is far-field DAG coarsening (finish the trunk follow-on + build voxel near-LOD)
   the right lever, or is there a better one level up that the weaker model couldn't see? (The whole
   two-tier design, the SCATTER architecture, the disjoint-tube generator, and "voxels at all" are all
   fair game.)
3. **Only then implement**, behind a flag, matched-condition A/B, shotdiff-gated for zero visible loss,
   user eyeballs the observable.

You have far more headroom than the model that wrote this. Use it on the *frame*, not the grind.
Good luck.
