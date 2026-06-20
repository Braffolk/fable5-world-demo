# The nanite forest perf cycle — DETAILED handoff (research × implement loop)

> You are taking over a long-running performance effort on a WebGPU Nanite-style foliage renderer
> (three.js r184 + TSL, branch `nanite-raster`, Apple M-series/Metal). This file is the COMPLETE,
> battle-tested operating manual for the 2-workflow optimization cycle. Read it fully, plus the
> project memory `nanite-perf-canonical-config-and-baseline.md` (the measured ledger) and
> `nanite-perf-capture-method.md`. Everything below is what ACTUALLY WORKED across 3+ cycles, with the
> caveats that cost real time to discover. Use dynamic `Workflow`s for EVERYTHING (the user insists).

---

## 1. THE GOAL (the ONLY thing the research workflow is ever told)

**Whole-frame `p0.95` (the 95th-percentile, i.e. WORST-5%, FRAME TIME in milliseconds) ≤ 16.6 ms** —
solid 60 fps — **in the FOREST scene under a MOVING camera** (a motion track through the canopy, NOT a
static pose). **Zero VISIBLE quality loss** is the only quality bar; the *methods* are fully changeable
(any coverage/depth/occlusion/LOD/aggregation/reconstruction algorithm, large refactors, new data
structures) as long as a MOVING viewer sees the same foliage. Large refactors welcome if the evidence is
strong.

**METRIC NAMING (user-corrected):** for frame TIME in **ms** the "worst-5%" = **p0.95** (95th pct of
time; lower better). For **fps** it would be p0.05. The original handoff said "p0.05 ≤16.6ms" — that was a
mislabel; it means **p0.95 frame time ≤ 16.6 ms**.

---

## 2. ⚠️ THE CANONICAL MEASUREMENT CONFIG (get this wrong and you waste a whole cycle — it happened)

The probes DEFAULT to **1280×720 / dpr 1 / 40k trees**, which is **NON-representative** (~3.6× too few
pixels, 5× too few trees) and reads a falsely-LOW ~20.9 ms. **DO NOT measure there.** The real config:

- `scene=forest`, **`trees=200000`**, **retina backbuffer 2268×1473** — boot viewport `1512×982` + URL
  **`?dpr=1.5`** (`bootPage` hardcodes `deviceScaleFactor:1` in tools/measure.ts; force the pixel ratio
  via the `?dpr` param: Params.ts:48 → Engine.ts:89 `min(devicePixelRatio,1.5)`). **ALWAYS VERIFY**
  `document.querySelector('canvas').width/height == 2268×1473`.
- `freeze=1`, `settle=60`, extra `lodnear=4 simband=6 lodpow=0.6 instminpx=128`, FULL post pipeline.
- Track: `tools/probe-motion.ts` `forest-worst` (121 poses, 3 frames/pose, 6 warmup). It is the canonical
  moving-camera track (alley / horizon-graze / canopy-dive).
- Harness: `tools/measure.ts` `measureActiveGpu` (GPU-bound, drain-isolated, vsync-IMMUNE, rejects
  cap-suspect frames) + `replayMotionTrack` + `percentile`. `cooldownMs=50` (thermally calibrated).
- **THE GATE METRIC is whole-frame `gpuWallMs` p0.95** = `percentile(perPoseWholeFrameGpuWallMs, 0.95)`.
  NOT a single pass. (gpuWallMs is the whole frame regardless of which pass key you pass to measureActiveGpu.)
- 200k boots clean in ~40 s.

**CURRENT MEASURED BASELINE:** whole-frame p0.95 ≈ **43.3–43.8 ms** (gap to 16.6 ≈ **27 ms, 2.6× over**).
The biggest single pass is `c.nanRasterWorld1` (SW raster, ~30.5 ms p0.95 / ~22 ms median). The **render
rollup (~33 ms) > compute rollup (~23 ms)**; they OVERLAP on the GPU so per-pass spans OVERCOUNT — only
whole-frame gpuWall is trustworthy. visClusters ~135k (peak ~232k), hwTris ~3.6M, inst 400k, cpu.submit
~1.9 ms (NOT CPU-bound). Replay paired-|Δ| noise floor ≈ **1.0–1.5 ms** → an A/B win must clear ~1.5–2 ms.

---

## 3. THE CYCLE (loop these; the orchestrator only LAUNCHES + RELAYS)

```
RESEARCH workflow  →  the SINGLE biggest win
      →  IMPLEMENT workflow (design → implement → adversarial review → A/B → keep/drop gate)
      →  relay the measured result + the keep/drop decision to the user
      →  loop (feed the measured result back into the next research's grounding)
```

**⚠️ EVERY workflow (research AND implement) carries a PREMISE-AUDIT stage** (see §5) — a wide step back that
puts the CONTEXT on trial (params, metric, the structure one level up) BEFORE implementing and AGAIN before
any "not-doable/refuted/disappointing" conclusion. A negative result triggers the premise-audit, not a method
tweak. The repo `CLAUDE.md` holds the verbatim "go up a level" directive; embed it in agent prompts (they
don't inherit it).

**Checkpoint the user on every STRATEGIC fork** (which win, which build path, accept a quality budget,
reconsider the goal). They steer direction; you run the machinery. They have repeatedly corrected
direction — relay honestly and ask before big/risky/quality-budgeted builds. **Never park/drop/reframe work
without surfacing it for their call.**

**Merge a KEPT, genuine win UP to `nanite-raster`** — but ASK FIRST. Experimental work lives in worktree
branches; only merge to production on the user's OK (so they see it in git).

---

## 4. THE RESEARCH WORKFLOW (script: `$CLAUDE_JOB_DIR/tmp/research-workflow.mjs` — REUSE IT)

8 adversarial stages, ~26 agents, ~2M tokens/run:
1. **GATHER** — 5 parallel SEARCH angles (academic web-search · engine/repo/talk web-search · mine the
   local prior-art briefs · architectural/first-principles · derive-from-our-code+raw-numbers).
2. **MERGE** — dedup + select ~4 DISTINCT, DIVERSE candidates (logs everything dropped).
3–5. **ANALYZE → CRITICIZE(×2 lenses) → RE-ANALYZE** as a per-candidate *pipeline* (no barrier).
6. **SYNTHESIZE** (barrier) — rank + pick the single biggest win.
7. **RE-CRITICIZE** — a 2-panelist adversarial panel tries to KILL the chosen win.
8. **FINALIZE** — the decision doc; the panel/finalize will SWITCH the pick if the synthesis was wrong
   (it has overridden the f2b pick twice — the panel is load-bearing, keep it).

**ABSOLUTE ANTI-BIAS RULE:** the research is told ONLY the goal + the raw measured grounding + the code +
the external prior-art briefs + web search. **NEVER inject "the bottleneck is X" / "focus on Y" / any
orchestrator hypothesis.** WITHHELD (would bias it): `docs/NANITE-ROADMAP.md`, `NANITE-SPEC.md` (esp.
D-N46), `NANITE-LOG.md`, `docs/perf-runs/prior-art/IDEAS.md`, any `*-REPORT.md`, this file, the memory.
SAFE to feed: the external prior-art briefs `docs/perf-runs/prior-art/{laine2011-cudaraster,curast,freepipe,
lucid,nanite-webgpu,nanite-deepdive,arxiv-2204.01287,extra-0..5}.md` (NOT IDEAS.md), the raw WebGPU-Inspector
capture `docs/perf-runs/2026-06-17-webgpu-inspector/slices/`, and the actual `src/nanite/*` source.

**FEED PRIOR-CYCLE MEASURED RESULTS as grounding** (raw facts, not a hypothesis) so the loop doesn't
re-propose a measured-loser. This is loop hygiene, not bias — research still derives the win independently.
A USER-DIRECTED scoped research round (e.g. "refine resolution") IS allowed — that's the user steering the
topic, not you biasing it.

**MODEL/EFFORT (critical — graphics is too complex for less, it produces garbage):** deep analysis /
synthesis stages MUST be **Opus + `effort:'xhigh'`** (analyze, criticize, reanalyze, synthesize,
re-criticize, finalize). ONLY the web-SEARCH gather stage may drop to `model:'sonnet'`. Merge = Opus high.

**Keep it ~26 agents** (cap candidates ~4, lenses ~2, panel ~2) until the methodology is proven; scale up
only after it works.

---

## 5. THE IMPLEMENT WORKFLOW (one dynamic Workflow per cycle; MEASURE-FIRST)

Shape (adapt per win): **MEASURE-FIRST baseline of the relevant arm → ⚠️ PREMISE-AUDIT → IMPLEMENT (edit
the worktree) → adversarial REVIEW (parallel, multi-lens, NO GPU) → conditional FIX → single-serial-GPU A/B
→ ⚠️ PREMISE-RECHECK (before any drop) → keep/drop GATE.** Scripts used as templates:
`submit-collapse-implement.mjs`, `res-scale-implement.mjs`.

- **⚠️ PREMISE-AUDIT stage (MANDATORY — runs BEFORE implement; and a RE-CHECK before any "refuted/not-doable"
  drop).** Step WIDE back from the task and put the CONTEXT on trial, not the method: are the PARAMS right?
  is the METRIC right (e.g. linear view-depth vs NDC-z, which compresses the far field)? is the structure
  ONE LEVEL UP — the setup that generated this task — the actual flaw? Can the generating context be changed
  instead of out-thinking the problem? **A workflow may NOT conclude "refuted / not-doable / disappointing"
  until this audit has fired and explicitly cleared the params, metric, and upstream structure as sound.** A
  negative result is the TRIGGER to audit the premise, not to vary the method. (Exists because flawed
  premises one level up — wrong depth metric, too-coarse buckets — kept getting misdiagnosed as the
  implementation being undoable, e.g. the f2b NDC-z bucketing that collapsed the far canopy into one bucket.)
  Embed the "go up a level" directive from the repo `CLAUDE.md` VERBATIM in the design/diagnose/verify agent
  prompts — subagents do NOT inherit CLAUDE.md or the project memory; they only get the prompt you write.

- **The A/B is the ONLY GPU stage**, a SINGLE agent, never inside a `parallel()`. Gate on **whole-frame
  p0.95 vs scatter past the ~1.5-2 ms noise floor, at the WORST camera, with parity (no holes, overflow
  counters 0)**.
- **Adversarial review PREDICTS; the A/B MEASURES — you need both.** Review has caught real bugs the A/B
  screenshot missed (e.g. a 1×1 render target starving HW coverage); the A/B has refuted reviewer-passed
  perf hopes. Don't skip either.
- **Quality-BUDGETED changes (resolution, LOD, aggregation): the USER judges perception, NOT an agent.**
  Produce perf numbers + side-by-side captures + a LIVE dev-server link (`npx vite --port <free>` from the
  worktree, give the user `?dpr=…` URLs) and let the user decide the bar. An agent must not declare
  "imperceptible."

---

## 6. ⚠️ ORCHESTRATOR RULES + CRITICAL CAVEATS (each one cost real time to learn)

- **NEVER run heavy GPU yourself** (destroys context). **NEVER overlap two GPU runs** — concurrent perf
  runs don't just risk crashing, they DILUTE the numbers (thermal + scheduler contention). EVERY read-only
  /parallel agent prompt MUST carry a hard banner: do NOT start the dev server / run any probe / harness /
  measure script / touch the GPU. Only the one serial A/B agent measures, alone.
- **DYNAMIC WORKFLOWS for everything** (the user insists; do not hand-fire ad-hoc swarms of Agents for a
  phase — wrap it in a Workflow).
- **WORKTREES (the orchestrator owns edits-isolation):**
  - A bg session CANNOT edit the shared/main checkout (a guard rejects it) — you MUST be in a worktree.
    Create a FRESH worktree off production `nanite-raster` HEAD per implement cycle:
    `git worktree add -b <name> .claude/worktrees/<name> nanite-raster`. Do NOT build a new cycle on a
    dropped-experiment worktree.
  - **Symlink `node_modules` AND `.cache` into the worktree** (`ln -s <main>/node_modules <wt>/node_modules`)
    or vite/playwright won't run.
  - `EnterWorktree({path})` to switch in; `ExitWorktree({action:'keep'})` to leave (parks it). Switching
    between two worktrees via `EnterWorktree({path})` works.
  - **Workflow subagents launch in the MAIN checkout cwd, NOT your worktree** (the Agent tool has no cwd
    override). BUT they CAN edit the worktree **by ABSOLUTE path** (proven cycles 2-3). If an agent's
    `Edit` tool is pinned to another worktree, it falls back to `python3` exact-replace — that works. Tell
    implement agents to edit `<absolute worktree path>/src/...`.
  - The implement workflow should `return {stoppedAt:'implement', impl}` if an agent reports it can't edit,
    so you can apply edits yourself and resume from review.
- **DROP = clean revert with ZERO production risk:** experimental changes live ONLY in worktree branches,
  never merged. (This is WHY the user sees no modified files in their main checkout — explain this.) A KEPT
  win is merged to `nanite-raster` on the user's OK.
- **A/B agent serves the WORKTREE on a FRESH port** (`npx vite --port 5182 --strictPort` from the
  worktree), NOT the main `:5173` (which serves un-edited code). Verify it's serving the edited build.
- **Measurement metric-key trap:** `c.nanRasterWorld1` exists ONLY on the scatter path; tiled/alt paths
  run different-named passes, and `measureActiveGpu` REJECTS frames where the keyed pass is absent
  (gpuWall reads 0). Key on the UNIVERSAL `'compute'` pass (always present) and read whole-frame gpuWall.
- **`?audit` orphan counter is NOT wired into the world1 forest pipe** (a NaniteView-debug-only hook using
  the packed combined raster → reads a vacuous 0). Do NOT trust `orphans==0` as a coverage proof; verify
  parity by LOGIC + screenshot (no holes) + the tiled FLAT_TILE_CAP/cluster overflow tripwire counters.
- **Thermal:** the FIRST hot boot inflates p95 (a single pose can read ~2× hot); discard it or bracket
  arms (A,B,A,B,A) to cancel drift. Cross-boot screenshots are NONDETERMINISTIC (wind + TAA history +
  accepted atomic race + LOD-by-pixel) — a control showed identical-config reboots differ on 95% of pixels.
  Prove zero-loss by LOGIC + a no-holes check + matched-boot MOTION sequences, NOT a cross-boot pixel diff.
- **WORKFLOW SCRIPT GOTCHAS:** (1) Pass grounding by HARDCODING it into the script as a `const` — the
  `args` channel silently surfaced FALSY once (`args.grounding` → fallback fired in all agents, a wasted
  ~4M-token run). Verify propagation by grepping the new run's transcript for a distinctive grounding
  string + confirming the fallback string is absent BEFORE trusting a run. (2) No backticks/`${}` inside a
  template-literal prompt string (they close the literal). (3) `node --check` flags the workflow's
  top-level `return` as a false-positive "Illegal return" — wrap the body in `async function f(){…}` to
  syntax-check. (4) Resume a paused run with `Workflow({scriptPath, resumeFromRunId})`; completed agents
  return cached, so fix broken code on disk first.
- **Don't be hasty to revert a zero-loss change** — keep/drop is the A/B-gated decision, not a snap call.
- **Measure-don't-guess:** the project ethos. Adversarial reasoning is a prediction; the A/B is truth.

---

## 7. THE MEASURED LEDGER (what's been TRIED + MEASURED — do NOT re-propose these as wins)

| Cycle | Win tried | Measured (whole-frame p0.95 @ 200k/retina) | Verdict |
|---|---|---|---|
| 1 | "de-risked TILED raster" (`?tileproto=1`: indirect kSetup + K_BUCKETS 8→2 + overflow tripwire) | tiled **55.2** vs scatter 43.5 (**+11.7 ms**) | DROP |
| 2 | "SUBMIT-COLLAPSE" of the tiled path (~82 submits → ~1 via dispatchBatch + baked per-wave batchBase) | scatter 43.3, tiled 59.3, tiled+collapse **61.4** (**+18.1 ms**) | DROP |
| 3 | "INTERNAL-RESOLUTION scaling" (dpr 1.5→lower) + free `nanHwPass` clear removal (`?hwrt=0`) | dpr 1.5=47(hot;~43-44), 1.3=38.9, 1.2=34.4, 1.0=**31.5** | resolution lever real but INSUFFICIENT; user KEEPS dpr 1.5 (1.0 visibly blurred). hwrt cleanup byte-exact KEEP (~1 ms, in noise) |

**Hard conclusions (measured, do not re-derive):**
- **TILING is REFUTED** — intrinsically +16 ms slower than scatter at retina+200k (the tiled RASTER PASS
  wins by ~2 ms but the FRAME loses on submit/setup/empty-wave overhead). Submit-collapse WORKS
  mechanically (185→66 submits, capture-confirmed) but the raster was never submit-bound. DO NOT re-pursue
  tiling or submit-collapse.
- **f2b front-to-back on the scatter queue is a WASH** — WebGPU gives NO inter-workgroup ordering (one
  indirect dispatch of 134-232k workgroups runs in undefined order), near occluders route to the HW queue
  elected AFTER the SW pass, and the existing relaxed-load guard already captures the atomic-skip.
- **The raster is COVERAGE / overdraw-VOLUME bound**, not submit/atomic/fragment-loop bound (matches the
  Nanite deepdive T2/T6: "reduce overdrawn VOLUME, not the fragment loop").
- **Resolution scaling is SUBLINEAR** (dpr1.0 = 0.44× pixels but only 67% of the time) because of a large
  **RESOLUTION-INDEPENDENT FLOOR** (CPU-submit/compute + the coverage cost) + nanite LOD coarsening with
  dpr. **Even native dpr1.0 = 31.5 ms ≈ 1.9× over 16.6.**
- **THE BIG STRATEGIC FINDING:** NO single lever reaches 16.6 ms at retina+200k. The goal needs a
  multi-lever STACK. MEASURED FACT (feed as grounding, do NOT turn into a directive): a large
  **resolution-independent COST** (~31.5 ms even at native dpr1.0; CPU-submit + compute setup + coverage)
  does not scale with pixels and is the dominant remaining cost. The NEXT unbiased research must RE-DERIVE
  the biggest lever itself from this + the code — do NOT pre-name "the floor", "vcompact", or any technique.

---

## 8. KEY FILES, WORKTREES, SCRIPTS

- Workflow scripts (reuse): `$CLAUDE_JOB_DIR/tmp/{research-workflow, submit-collapse-design,
  submit-collapse-implement, res-scale-implement}.mjs` (a new session's `$CLAUDE_JOB_DIR` differs — re-author
  from this manual + the memory if absent).
- Source: `src/nanite/{NaniteRaster (world1 SW scatter raster + the ?tileproto/?hwrt gates),
  NaniteTileRaster (the parked tiled path), NaniteCull, NaniteResolve, NaniteFrame, NaniteHzb, NaniteFetch,
  NaniteVertexCache, GeometryRegistry, Clusterize, BuildAggregateDag}.ts`. Engine/dpr: `src/core/{Engine,
  Params}.ts`. Post: `src/render/{PostStack, HalfResMrt}.ts`. Probes: `tools/{measure, probe-motion}.ts`.
- Worktrees (parked, off production HEAD `f242f6b`): `.claude/worktrees/nanite-tiled-derisk` (DROPPED
  tiling + submit-collapse), `.claude/worktrees/nanite-res-scale` (the `?hwrt` byte-exact cleanup + the
  resolution-A/B work).
- Prior-art briefs: `docs/perf-runs/prior-art/`. Raw capture: `docs/perf-runs/2026-06-17-webgpu-inspector/`.

---

## 9. CURRENT STATE + NEXT (2026-06-18)

Production `nanite-raster` is UNCHANGED except the merged `nanHwPass` byte-exact cleanup. User decisions:
- **dpr stays 1.5** (crude resolution drop rejected — visibly blurred at 1.0).
- **NEXT (user-directed, in order):** (1) a SCOPED research round to FINE-TUNE the resolution lever — find
  an imperceptible reconstruction that banks the perf WITHOUT the visible blur (temporal upsampling /
  checkerboard / per-pass or variable-rate resolution / dynamic res / FSR-like — render cheaper, reconstruct
  to dpr-1.5 quality). (2) BANK that refined win. (3) Run a NEW, UNBIASED novel-research round (the standard
  8-stage research workflow) — given ONLY the goal + the cycle 1-3 measured grounding (incl. the large
  resolution-independent cost). It DISCOVERS the next biggest win itself. CRITICAL: do NOT pre-lock the
  topic — no "the floor", no "vcompact", no technique handed in. Feed the measurements as facts; let it
  explore. (Orchestrator note: I once pre-named "the floor/vcompact" as the next topic — that is exactly
  the anti-bias violation to avoid.)
- The `nanHwPass` byte-exact cleanup (`?hwrt=0`) is a KEEP and is being merged up to `nanite-raster`.
