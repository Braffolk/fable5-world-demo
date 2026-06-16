# Performance-review workflow — skeleton + the meat we feed it

> Two separate things, deliberately:
> - **The SKELETON** = a *generic*, reusable multi-agent perf-review workflow. Lives in git at
>   **`.claude/workflows/perf-review.js`**, always available, contains **no** project/task specifics.
> - **The MEAT** = the project + task context you *ladle on at run time* via `args` ("impostors are
>   already counted in", `world1`, the operating point, the in-play constraints). §B below is the meat.
>
> Why it exists: a single LLM run repeatedly converged on **plausible-but-wrong "wins"** (the bloom/TAA
> *mirages*, LOG bd–bg). The skeleton defeats that with parallel adversarial review + an agent-external
> measurement gate, and is open-ended enough to reach **novel** levers. Derived by a 26-agent deep-research
> run whose own red-team critic caught a load-bearing error in the first draft — the adversarial layer works.

---

## A. THE SKELETON (generic — `.claude/workflows/perf-review.js`)

Phases: **Measure → Attribute → Confirm → Ideate → Rank → Synthesize.** The skeleton never names a
kernel, flag, or domain object — those arrive as `args`.

**Invoke:**
```js
Workflow({ name: 'perf-review', args: {
  context:       "<the system + how to measure it (probes, metric keys, dev server, hooks)>",
  target:        "<what 'fast enough' means + the HONEST metric to judge it by>",
  investigation: "<what to find; suspected-but-UNCONFIRMED hypotheses — NOT conclusions>",
  constraints:   "<hard feasibility limits the ideas must respect>",
  qualityBars:   "<no-regression floors; what 'noticeable quality loss' means>",
  measure:       "<concrete guidance: probe commands, metric keys, sweeps>",
  // optional: lenses:[...], analysts:5, measureEnabled:true (false ⇒ reasoning-only, no device)
}})
```
(`args` may also be one big string = the whole brief. No args ⇒ it warns and reasons-only.)

**What the skeleton bakes in (generic discipline — the value, none of it task-specific):**

- **Frozen, agent-external grader; proposer never grades its own change** — an independent verifier
  re-measures. (The reward-hacking literature: METR, Sakana CUDA agent, KernelBench.)
- **Ablation/sweeps over raw timers** — timers lie (refresh/vsync caps, overlapping spans, thermal,
  CPU-vs-device boundedness). Prefer noise-immune counters; cross-check.
- **Diagnose before solving; adversarially CONFIRM the diagnosis** — a wrong diagnosis dooms everything.
- **Every idea must attack a CONFIRMED driver, or it's killed** — the systematized guard against chasing
  a non-bottleneck.
- **Structural novelty, not "be creative" prompts** — diverse expertise lenses, "what have we dismissed",
  cross-domain analogy, first-principles "question the architecture", stepping-stone (an idea that moves
  the bottleneck counter survives even before the headline number drops).
- **Mechanism pre-registration; quarantine-then-explain** — a win larger than the cost model predicts is
  held until its mechanism is explained (so real cost-class wins survive, mirages don't).
- **Serialize the device** — measurement parallelism is *tokens only*; never two boots at once.

The full cooled-A/B *commit* campaign (proposer → independent verifier → adversarial skeptic → evidence-bound
judge → change-point ledger) is the heavier sibling; the shipped skeleton is the **diagnose-and-ideate** core
that feeds it. Extend the skeleton, don't fork it per task.

---

## B. THE MEAT (what we ladle on for *this* project) — NOT part of the skeleton

### B1. Project standing context (the reusable brief for this renderer — the "necessary project description")

- **System:** from-scratch Nanite-style GPU-driven virtualized-geometry renderer in a **browser on WebGPU**
  (three.js r184 WebGPURenderer + TSL + hand-written WGSL compute), Apple-silicon GPU (Metal/TBDR),
  deterministic `?seed`, TS strict zero-`any`, zero external geometry libs. Nanite engine self-contained in `src/nanite/`.
- **Measurement harness:** `tools/probe-*.ts` (model new probes on `probe-forestfull.ts` + `tools/launch.ts`'s
  `laasUrl()`/`launchWebGPU()`) drive headless Chromium against the running vite dev server at
  `http://localhost:5173`. Read `window.__laas.stats` (`gpuPasses` / `counters` / `frameMs`). The cap-immune
  metric is the **per-pass GPU timestamp** (e.g. `c.nanRasterWorld1`) — **not** `frameMs` (`= rawDt*1000`,
  vsync-capped at 16.7 ms → blind above 60 fps and under motion).
- **Quality bars (binding):** ≥5M-tri hero crown, no pop within 300 m, no stochastic/dithered density loss,
  deterministic per `?seed`. Quality-affecting levers need **user judge-shots**.
- **Platform constraints:** no 64-bit atomics; documented three.js/Metal **"3rd atomic storage buffer" ~3× cliff**;
  ~10-buffer resolve-bind-group ceiling; no mesh shaders; subgroup/SIMD ops available.

### B2. Current task brief (the forest investigation — the per-run meat, *an example of §A args in action*)

- **Target:** 200k-tree forest blazing fast, no noticeable quality loss, **avg >60 fps during movement**; debug
  (lean) path first, then full pipe.
- **Operating point (verified):** lean vis-buffer = `?nanitedbg=flat`; plain `?scene=forest&nanite=1` = the FULL
  pipe (`nanitedbg=1` is not valid). Canonical flags `trees=40000 lodnear=4 simband=6 lodpow=0.6 instminpx=128`,
  1280×720 → frameMs ~24 ms, `world1` ~15 ms (dominant), ~335k visClusters.
- **The meat that reframes the problem:** with `instminpx=128` the **far field is already impostored**, so the
  cost is the **near/mid crowns** → a far-field cross-instance *merge* (#48) is the **wrong lever**.
  `335k × ~55 ns/cluster ≈ 18 ms ≈ world1` ⇒ per-cluster overhead is the **prime suspect** — *a hypothesis to
  test, not a conclusion* (the only stage split we have, 16/39/40, is from the deleted 2-pass kernel).
- **Calibration cautions:** the `wgcache −11%` figure is from the deleted 2-pass kernel; `?hierdepth=30` is a bad
  negative control (doesn't move `world1`) — use `?instminpx=0` / `?simband=0` / 2× resolution.

This brief is exactly what gets passed as `args` to `perf-review`. **Run-1** (`raster-forge-run1-diagnose`,
script in the session's `workflows/scripts/`) is this brief through the skeleton — diagnosis-first; its confirmed
cost map + idea portfolio will refine B2.

---

## C. Open decisions for the campaign (set before the heavier cooled-A/B sibling)

1. **Primary-metric sign-off:** accept `Δ(per-pass GPU timestamp)` on the lean path as primary (cap-immune,
   motion-safe), validated by a self-ablation, with full-pipe ablation as a directional cross-check?
2. **Autonomy / device budget:** cooled A/B is serial single-GPU (~15–30 min/round, 1–2 h/generation). Unattended
   overnight (bandit-capped, early-stop) vs attended per-generation?
3. **Exploration prior (after the diagnosis):** execute a known lever vs hunt untested ones — Run-1's confirmed
   cost map should largely settle this.

> Skeleton = `.claude/workflows/perf-review.js` (git, permanent). Meat = this doc's §B + whatever the next task needs.
