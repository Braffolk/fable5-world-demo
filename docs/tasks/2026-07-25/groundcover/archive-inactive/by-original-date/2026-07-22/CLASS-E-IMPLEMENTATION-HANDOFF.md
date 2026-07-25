# HANDOFF — Class-E ground cover: measure, compile, gate, implement

Date: 2026-07-22. Branch: `estonia-asset-gen`. You are the **single owner** of
this track, end to end. Supersedes the shell-frame handoff and the
shell-frame gate status (that model is demoted; see §0.3).

> **MATHEMATICAL HOLD / BINDING REPLACEMENT (2026-07-22):** Do not execute this
> handoff's phases as written. `CLASS-E-EXTERIOR-MATHEMATICAL-SPEC.md` is now
> the binding exterior-only replacement, after two independent adversarial
> passes. `CLASS-E-MATHEMATICAL-AUDIT.md` records why the older theory failed.
> The body below is historical planning context only. The replacement proves
> the ideal affine opaque core and a closed-form analytic plume core, but makes
> implementation conditional on structural compiler feasibility, extended
> visibility/interval certification, the flat-ground-compatible no-horizontal-
> axis restriction, an honest minification choice, complete sample/byte cost,
> and fragment-invocation coverage. The user requested mathematics first; no
> compiler, harness, baker, shader, or render-pipeline phase is authorized by
> this handoff until that hold is explicitly lifted.

> **CPU GATE RESULT / RED (2026-07-22):** The hold is not lifted. The accepted
> production Calamagrostis blade+culm endpoints alone give a global lower
> bound of `14` opaque `M x I` fields / `29` two-layer reads at `10 mm`, `9` /
> `19` at `20 mm`, and `5` / `11` even at `50 mm`, omitting the entire flower
> head. See `CLASS-E-CPU-GATE-RESULT.md`. Do not implement this representation;
> its resume condition is new endpoint mathematics, an explicit cost increase,
> or an explicit acceptance of botanical-scale error.

---

## 0. Orientation

### 0.1 Standing rule — follow verbatim (you do NOT inherit `CLAUDE.md`)

> When a problem resists solving, or a result disappoints, do NOT start varying
> your approach to the problem — that stays inside the problem. Go up one level
> first, to the context the problem lives in: the surrounding system, the
> upstream decisions, the goal that made this a problem, and everything you
> have been treating as the fixed environment it sits inside. What you filed
> under "given" is the prime suspect, precisely because you filed it under
> "given" and never looked at it. Ask: what generates this problem? Is that
> context flawed in a way that PRODUCES this failure? Can I feasibly change the
> context instead of out-thinking the problem? Only if the context is genuinely
> fixed, or genuinely sound, drop back down and solve the problem where it sits.
> The trigger is the negative result, NOT your confidence — fire it ESPECIALLY
> when you are sure the limit is real.

A negative or disappointing result is the **trigger to audit the premise**
(params right? metric right? anchor right? the setup one level up?), not a
licence to grind variants. You may not report "refuted / not doable /
disappointing" until that audit has fired and explicitly cleared the setup.

**This rule has already paid twice on this exact track.** The previous agent
ran a reconstruction gate at a direction lattice ~2,600× below the model's own
sizing law and reported the model refuted; the real finding was that the
lattice was wrong. Do not repeat that class of error: **before declaring any
result, verify that the configuration you tested satisfies the law you are
testing.**

### 0.2 Read first, in this order

1. `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-EXACT-REPRESENTATION-THEORY.md` — **the
   specification.** §6 (class-`E` construction + proofs), §7 (extinction
   families for sub-pixel detail), §8 (error bounds), §9 (runtime algebra and
   cost model), §10 (counterexample battery + disclosed limits), §13.3–13.4
   (banded masks for curvature; the compilation corollary).
2. `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-22/GRASS-SHELL-FRAME-FIELD.md` — §14 first (**why
   class `E` was selected: the dimension theorem and the `N_ω` law**), then
   §13 (regime analysis, measured skin depths), then §11 (errata; the mip law
   and metric corrections carry over).
3. `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-22/GRASS-SHELL-FRAME-FIELD-RECONSTRUCTION-BLOCKER.md`
   and `...-RED-BLOCKER.md` — what failed and why, so you do not rebuild it.
4. `docs/tasks/2026-07-25/groundcover/active/GRASS-STATUS-AND-ISSUES.md` §§18–44 — the rejection
   ledger. **Nothing in that list may be resurrected under a new name.**

### 0.3 Where the design stands

Selected architecture: **class `E`** — the community is represented as a
bounded union of affine-sheared, lattice-compatible **parallel-extrusion
families** (arbitrarily detailed periodic 2D masks with per-texel marks;
per-band displaced masks for curvature; family-global horizontal clips) plus
**extinction (volumetric) families** for sub-pixel plume/fuzz. Elevation is
**analytic** (`t = t_⋆ + ρ/s_p`), so each family's stored field is
**3-dimensional** (2 phase + 1 in-plane angle), never 4D.

Demoted: the shell-frame field (4D). Retained only as an optional accelerator
for connected dense mats; **not on your critical path.** Do not spend effort
on it unless a gate below explicitly sends you there.

---

## 1. Contract (user law — non-negotiable)

- **NO runtime geometry of any kind.** No grass meshes, no impostor quads, no
  billboards, no fins/shells, no raised or floating carriers, no duplicated
  terrain. Everything is fragment-shader evaluation of precomputed fields,
  rooted in the real ground chart. "No floating BS."
- **O(1) per pixel.** No ray march, loop, traversal, variable candidate list,
  per-species query, or data-dependent iteration. Family/band counts are
  compile-time constants.
- **Exterior fidelity is the bar.** The camera may not enter the cover volume
  and expect high quality — but every exterior direction (top-down, oblique,
  grazing, exact horizontal) must be high fidelity. No angle may be hidden,
  flattened, clamped, or declared unsupported. Interior = graceful fade only.
- **Low memory.** Working ceiling 250 MB resident for the whole cover system;
  prefer far less. Report measured budgets per community — never assert them.
- **Estonia-native species only.**
- **Variance = tier 1 only** (user decision): two global golden-angle layers +
  the lattice's `D4` variants + smooth world-keyed tint / vigor / ±15% height
  modulation. Do **not** implement runtime cell bombing or super-tile bakes;
  the user decides later if tier 1 is insufficient.
- **Wind = affine tier only**: time-varying, spatially smooth,
  affine-in-height shear (`x' = x + β(x,z,t)(y − g)`), exact by conjugation.
  No per-blade flutter mechanics.

---

## 2. The one distinction you must keep straight

There are **two different comparisons**, and conflating them is the main way
this track can go wrong:

| gate | question | compares |
|---|---|---|
| **Compilation** | does the class-`E` community `𝒢*` look like the real plant? | `𝒢*` vs the authored reference mesh |
| **Reconstruction** | does the runtime algebra reproduce `𝒢*`? | runtime output vs exact ray-traced `𝒢*` |

Reconstruction is expected to be near-exact **by construction** (that is what
§6's proofs assert); it validates the transcription. Compilation carries the
real risk and is the visual question. Score them separately and never let one
mask the other.

---

## 3. PHASE A — measure the angular budget (do this first; it is cheap and it decides everything)

**Why:** §14's law gives `N_ω = 2π/(k·θ_pix) · h/(m₀+h)` using the
*pessimistic* bound `ρ ≤ h/α` (ray traverses the whole band without hitting).
The true requirement uses the measured free path: `Δω ≤ k·θ_pix·D/ρ_p95`. For
dense and tall cover the real `ρ_p95` is far below the traverse bound, and
that is the difference between a ~220 MB budget and a small one.

**Measure** — on the actual accepted meshes, offline, no compilation needed
(the horizontal free path is family-independent):

For each community (`Calamagrostis canescens` h≈1.15 m — the expensive case;
the sparse `Agrostis` stand; the connected `Sphagnum` mat) and for elevations
`{90, 75, 60, 45, 30, 18, 10, 5, 2, 1, 0.1, 0}°` × ≥4 azimuths × dense phase
sampling of the periodic community:

1. `ρ` = **horizontal** distance travelled from band entry to first hit
   (miss ⇒ censored at the band exit; report the censoring rate);
2. emit `ρ` p50/p95/p99 per elevation, and the derived
   `Δω(α) = k·θ_pix·D(α)/ρ_p95(α)` with `D(α) = (m₀+h)/sinα` for the declared
   eye heights, then `N_ω = 2π/min_α Δω(α)`;
3. also emit the same statistics for the **crisp** subset (blades, culms,
   rhizomes, panicle axes) separately from plume/fuzz — fuzz goes to
   extinction families (§7) and must not size the opaque lattice;
4. emit the resulting **per-community byte budget** at candidate phase
   pitches (blade-width-matched, e.g. 128²/192²/256² per 0.52 m tile) and
   record counts, with and without block compression.

**Deliverable:** one content-addressed report + a table of
`(community, N_ω, phase pitch, bytes)`. This is the number the rest of the
track is built on. If any community's honest budget exceeds the 250 MB
ceiling, that is a **reported measurement and a surfaced user decision**
(quality knob `k`, phase pitch, or per-community LOD), never a silent choice
and never a park.

**Also sweep `k`.** The registration tolerance `k` (pixels) was assumed at
2.5 and never measured; it drives the budget quadratically. Emit budgets at
`k ∈ {2.5, 5, 10}` so the user can trade bytes against edge softness with
real numbers.

---

## 4. PHASE B — compile the community into class `E`

The accepted Calamagrostis source is procedural and in-project; its generator
is already partitioned into `270,535` primitive charts with exact family
semantics (foliage, culms, rhizomes, panicle axes, spikelet surfaces, callus
hairs/filaments, anthers — see `GRASS-PROCEDURAL-CHART-FEASIBILITY.md`). This
is a **compiler**, not hand re-authoring.

Per theory-doc §6 and §13.3–13.4:

1. **Assign each primitive** to an (axis class, height band) pair. Axis
   classes `K` and bands `B` are compile-time constants; start from the
   measured lean/tilt distribution of the generator's own primitives, not
   from a guess.
2. **Emit per-band masks** from each primitive's true curve (per-band
   displaced cross-sections ⇒ a curved culm becomes an exact `B`-chord
   polyline; chord residual `∝ 1/B²` — the culm's measured `3.98 mm`
   straight-generator residual must fall below `0.5 px @ 4 m`).
3. **Route sub-pixel populations** (callus hairs, filaments, anthers) to
   extinction families per §7, preserving the §34 band-limited plume result.
4. **Per-texel payload:** authored colour, full 3D normal, species/material
   mark, and (if adopted) rank-`K′` tip records per §13.3.2.
5. **Emit computable residuals per primitive**: axis-class angular
   quantization, chord residual, tip band/texel residual, plume split. These
   are the compilation gate's evidence.

**Compilation gate (visual/authoring):** render `𝒢*` and the reference mesh
from the four cardinal directions + a low-oblique diagnostic (the §34 harness
already does this) and compare silhouettes, colour, and panicle structure.
Also report the residual distributions from (5). If the residuals are small
but the images differ, the assignment (`K`, `B`) is wrong — fix the
assignment, not the model.

---

## 5. PHASE C — reconstruction gate at the measured `N_ω`

Extend the existing offline harness (`tools/groundcover-bake/*`, the
Experiment-024 lineage) to evaluate the §9 runtime algebra against exact ray
tracing **of `𝒢*`**, at the Phase-A lattice.

Runtime model under test (theory doc §6.3, §6.5, §6.6, §7.3, §9):
affine transfer `Â = A_f^{-1}` → in-plane split `(p, s_p, ω)` → slab clip →
address `q_⋆` → one field read per family-band → lift `t_f = t_⋆ + ρ/s_p` →
accept vs clip → composite across families front-to-back (opaque = min of
exact depths; extinction families per §7.3) → attributes from the winning
texel only. Axis-parallel (`s_p = 0`) uses the categorical occupancy chart —
**no epsilon anywhere**.

Metrics (corrected per §11/§13.1 — the old geometry threshold was
mis-specified):

- **transverse** displacement p95, scored against the pixel footprint at the
  hit distance (this is what shows in the image), **not** along-ray;
- **depth** scored separately against a compositing tolerance (terrain,
  water, trees);
- silhouette IoU ≥ `0.97`; composited RGB max-channel p95 ≤ `0.15`; largest
  connected wrong region < `1%` of frame;
- **stability**: unforced predicted class change < `5%` under `1/2.5/4.5 mm`
  camera translations — this is the binding view-consistency criterion;
- explicit checks of the §10 battery: exact horizontal, exact vertical,
  grazing, tile-boundary crossing, two overlapping species, moss beneath
  blades, order swaps;
- measured edge behaviour vs the §8 bounds; measured resident bytes and taps.

Views: the Experiment-024 set plus `{0.1°, 0, 30°, 45°}`, eye heights
`m₀ ∈ [0.4, 1.6] m`, all three communities.

**Sanity check before believing any RED:** confirm the tested lattice
satisfies the Phase-A law, the conditioning gates hold, and no epsilon/clamp
was introduced. A RED at an under-sized configuration is a configuration bug,
not a result.

---

## 6. PHASE D — implementation (only after B and C are GREEN)

- **Cook side (`asset-gen` / bake):** the class-`E` compiler (Phase B) and the
  per-family 2D field baker — for each family, an offline raycast of the
  **infinitely repeated** mask at `(phase, angle)` storing first-entry path,
  normal, colour, mark. Note this is a *2D mask raycast*, not "render the mesh
  from N directions" — do not reuse the shell-frame baker's semantics. Heavy
  work belongs in the cook; the runtime only samples.
- **Runtime:** fragment path only. One field read per family-band at
  analytically computed addresses (all reads independent — addresses never
  depend on fetched data), fixed ALU, fixed select network, two layers via
  global affines (bytes ×1), control field selects the per-type atlas before
  the query (existing root-stable A/B mixture path). Mip law: premultiplied
  colour and normal mips are legal; **never mip the depth/path channel** —
  coarse mips use the extinction (volumetric) semantics of §7.4.
- No new pass, dispatch, barrier, binding growth per species, or screen-sized
  target. Interior fade over the last ~0.5 m of approach.

---

## 7. Verification, tracking, and commit law (project law — you do not inherit it)

- `npm run typecheck` must pass; keep tests lean while algorithms are in flux.
- **Before giving the user ANY live URL**, boot that exact URL (exact manifest,
  exact port) in real Chromium/WebGPU via `npx tsx tools/boot-smoke.ts`. Page,
  console, TSL, WebGPU, binding, pipeline, or command-buffer errors fail the
  boot even if ready fires. A clean boot is **never** visual acceptance.
- Canonical test URL:
  `http://localhost:5173/?scene=world&src=estonia&dataurl=http://localhost:8787&dpr=2`
  (data server serves `asset-gen/data/out` on `:8787`). Terrain-subject review
  links always carry `grass=0`. Give exact-position URLs; never compass
  directions.
- Python runs through `uv` (`uv run`, `uv add/sync`) — never bare pip/venv.
- A fresh **game-only GPU trace** is required after visual acceptance:
  `tools/profile/gputrace.sh` → **ask the user** to do the manual Xcode export
  with "Embed performance data" → `tools/profile/run_all.sh`. Casual HUD
  readouts are not performance results.
- **Commits:** local only, **never push**. One task per commit with an
  explicit file list. Never `git add -A`. `asset-gen/` is committed but in its
  **own** commits, never mixed with `src/` changes. Never commit
  `docs/METAL-PROFILING.md` or `docs/todo-human-only.md`. End every commit
  message with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- **Dual task tracking:** the Claude Code tasklist tool **and** dated files
  under `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-22/`, kept in sync, updated at every terminal
  result so a compaction cannot lose state.
- **Two real end-to-end failures ⇒ park** with a written blocker and an
  objective resume condition — and **surface it to the user**; never park,
  drop, reframe, or redirect silently. One consolidated diagnose+fix cycle per
  failure, not implement→review ping-pong.
- Work autonomously otherwise. Return to the user for: live-URL visual gates,
  destructive/irreversible actions, budgets that exceed the 250 MB ceiling,
  and genuine scope decisions.

---

## 8. Explicitly out of scope

Do not resurrect, under any name: sampled camera-direction lattices / 4D
first-hit atlases, nearest-or-blended direction selection, cross-owner depth
arithmetic or PCF-style depth quantiles, fixed-`K` deep event records, learned
/ low-rank / VQ / codebook codecs of the 4D field, Plücker separator trees or
owner-closure runtimes, per-tile projective taper, runtime candidate lists or
marches, raised terrain shells, and per-species runtime queries. Each is a
measured rejection in the ledger (§0.2 item 4); adding directions, ranks,
layers, iterations, or candidates is not a new idea.
