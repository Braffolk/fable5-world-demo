# CLAUDE.md — always-loaded project instructions

## Go up a level before you grind — STANDING RULE (fires on EVERY negative/disappointing result)

When a problem resists solving, or a result disappoints, do NOT start varying
your approach to the problem — that stays inside the problem and is the move you
already default to. Go up one level first, to the context the problem lives in:
the surrounding system, the upstream decisions, the goal that made this a problem,
and everything you've been treating as the fixed environment it sits inside.

What you filed under "given" is the prime suspect, precisely because you filed it
under "given" and never looked at it. You're working the problem as the figure
and the context as untouchable background. Put the background on trial:

- What generates this problem? What upstream system, decision, goal, or
  assumed-fixed condition created the situation where this problem exists and is
  hard? Name the parts you've been treating as the environment rather than as
  choices you could revisit.

- Is that context flawed in a way that PRODUCES this failure? Not "is my method
  wrong" — is the setup one level up the actual source of why this can't be
  solved or comes out disappointing?

- Can I feasibly change the context? If the generating setup is changeable at
  acceptable complexity cost without negatively affecting other systems, change THAT. The problem dissolves or drops to trivial,
  because you removed what was producing it instead of out-thinking a problem
  that shouldn't have existed.

Only if the context is genuinely fixed, or genuinely sound, drop back down and
solve the problem where it sits.

The trigger is the negative result or the
drop decision, NOT your confidence — fire it ESPECIALLY when you're sure the
limit is real.

## Every Workflow + substantial subagent MUST premise-audit — they do NOT inherit this file or the memory

A Workflow agent / subagent gets ONLY the prompt you write; it does NOT load this file or the project
memory. So the rule above never reaches them unless you put it there. Therefore, when you author ANY
Workflow (or a non-trivial subagent):

1. Add an explicit **PREMISE-AUDIT stage** early — right after the first grounding, BEFORE committing to the
   implementation — AND a **mid-flow re-check** that must fire before any "hard / refuted / not-doable /
   disappointing / can't" conclusion. The stage steps WIDE back from the task and interrogates the context
   it lives in: are the PARAMS right? is the METRIC right (e.g. NDC-z vs linear depth)? is the structure ONE
   LEVEL UP (the setup that generates this task) the actual flaw? Can the generating context be changed
   instead of out-thinking the problem? It applies the "go up a level" rule above to the task itself.
2. Embed the "go up a level" directive (the block above) VERBATIM in the design / diagnose / verify agents'
   prompts — they need it in-context.
3. A negative or disappointing result is the TRIGGER to audit the premise, NOT to start varying the method.
   A workflow may NOT return "refuted / not-doable / disappointing" until the premise-audit has fired and
   explicitly cleared the setup (params, metric, upstream structure) as sound.

This exists because I repeatedly entered with a flawed premise one level up — wrong depth metric, too-coarse
buckets, wrong params — then blamed the implementation as undoable (e.g. the front-to-back "refutation" that
was really just NDC-z bucketing compressing the far field).

## Serious optimization runs on FRESH profiling data — STANDING RULE

Any serious GPU/perf optimization (foliage, shaders, frame time) is grounded in ACTUAL profiling
data of the CURRENT code — not guesswork, not stale numbers. Re-profile after meaningful changes.
The pipeline (full guide: `docs/METAL-PROFILING.md` + `tools/profile/README.md`):

1. `tools/profile/gputrace.sh` → a raw `.gputrace` (headless).
2. ⚠️ MANUAL, in Xcode — open it → let it replay/profile → **File ▸ Export with "Embed performance
   data" ENABLED** → exported bundle. Only Xcode's GPU replay produces timing, so the tools CANNOT
   do this step — **ASK the user to do it** (and to re-export after code changes).
3. `tools/profile/run_all.sh <raw> <exported>` → one results folder: `summary/` rankings +
   `runtime/<shader>.txt` (per-line time) + `runtime/msl/*.metal` (source) + `static/` (occupancy).
   Optimize ONE shader at a time from its files.

Re-profiling is SLOW (capture + manual Xcode export + ~10 min analyze), so do NOT re-run reflexively:
one fresh run per serious optimization push (and after meaningful changes), reused within the session.
Subagents do NOT inherit this file — put the profiling context + the results-folder paths in their prompt.

## Orchestration & model tiering (2026-07-17, supersedes AGENTS.md "Delegation")

The main session is an ORCHESTRATOR. It protects its own context window and delegates
result-producing work through dynamic Workflows (and Agent for one-off reads/digests):

- **Tiering:** judgment calls, architectural thinking, reviews, adversarial verification,
  and complex algorithm design run on **Fable 5 at high effort** (omit `model` in workflow
  `agent()` calls — inherit the session model). Everything simpler — generation runs,
  ingestion, harnesses, mechanical edits, routine fixes, tests — runs on
  **`model: 'opus'`, `effort: 'xhigh'`**. Don't burn Fable tokens on mechanical work;
  don't hand Opus a judgment call. Rescuing a "probably incompatible" dataset into a
  usable one is a judgment task (user law 07-17): Opus may measure, but the
  fit/salvage/method calls are Fable-tier.
- **User judgment cadence (user law 07-17):** ask for visual verdicts sparingly — batch
  review points into one ask; never ask during announced AFK (user announces AFK first).
- **Context economy:** never read a large spec/ledger wholesale into the orchestrator —
  delegate a digest (current one: `docs/tasks/2026-07-13/SPEC-ORCHESTRATOR-DIGEST.md`).
  Have agents return structured/terse results, not file dumps. Update the task ledger at
  every terminal result so compaction never loses state.
- Delegation is ONE level deep: workflow/sub-agents do their own work and never re-delegate.
- Work autonomously; return to the user only for live-URL visual gates, destructive or
  irreversible actions, and genuine scope decisions.

## Efficiency laws (distilled from AGENTS.md after a ~10B-token overrun; full text there)

- **Outcome first.** Every substantial track starts by naming the next artifact a user or
  downstream system can actually inspect, and works backward only through strictly
  necessary dependencies. Commits/schemas/audits/tests are not outcomes.
- **One owner per track**, through a real end-to-end result. No tiny implement→review
  ping-pong; one consolidated diagnose+fix+rerun cycle per failure, one independent review
  at the complete boundary (earlier only for irreversible/destructive/format boundaries).
- **Two real end-to-end failures ⇒ park** (record blocker + objective resume condition in
  KNOWN-ISSUES; never delete reusable work) — unless the capability is uniquely required
  and the reason is recorded. Parking is surfaced to the user, never silent.
- **Parallelize independent candidates** (datasets, methods, integrations) instead of
  serializing behind the hardest one; advance the first that meets the real quality bar.
- **Stop research once the current decision is resolved.** Batch docs/provenance/commits
  at outcome boundaries; don't interrupt the critical path for ceremony.
- Evidence closure, hashes, preregistration, and gates SUPPORT the next visible terrain
  artifact; they are never the product. Supporting work that doesn't unblock the next
  inspectable artifact is scope expansion.

## Project laws that always apply (full versions in AGENTS.md + the spec)

- **All terrain synthesis is cook-side in `asset-gen`;** the browser only streams, decodes,
  samples, morphs, renders packed heights. Canonical grid: LOD0 = 1 m / 2048 m; LOD -1 =
  0.25 m / 512 m; LOD -2 = 0.0625 m / 128 m. No decorative noise, no broad-class style
  lookup, no scene-specific fixes. Source DTM is fallible — evidence-backed correction
  beats exact reconstruction.
- **Real runtime acceptance:** before giving the user ANY live URL, boot that exact URL
  (exact manifest, exact port) via `npx tsx tools/boot-smoke.ts` in real Chromium/WebGPU —
  page/console/TSL/WebGPU/pipeline errors fail the boot even if ready fires. Every boot
  review also eyeballs terrain continuity, materials, trees/plants, grass grounding.
  Give exact-position URLs; NEVER compass directions (UI has no compass). Estonia target
  is `scene=world&src=estonia`. **Terrain review links always carry `grass=0`** (user law
  07-17 — grass hides the mesh); run the all-layer boot separately for surface agreement.
- **Best-fit doctrine (user law 07-17):** when a regime has no perfect candidate, pick the
  best fit and make it work — IF doable without significant effort. This tempers, not
  repeals, the never-fake ethos: corrections still need evidence; bounded weak-evidence
  best-fit beats indefinite abstention.
- **Shaders are performance-critical:** no edit on "should work"; smallest validity fix;
  written perf rationale; re-boot after any shader-path change; traces over casual A/B
  (thermal noise) — see the profiling rule above.
- **Zero external budget:** no paid data/surveys/vendors/hardware unless the user reopens
  budget. Exhaust public/open sources incl. physically-analogous Baltic/Nordic regions.
- **Research standard:** a feature name is never its spec — decompose into real constituent
  phenomena and ground quality-defining work in full primary sources, never abstracts or
  one plausible search hit. No toy proxies for high-fidelity asks.
- Python runs through **`uv`** (`uv run`, `uv add/sync`) — never bare pip/.venv.
- Tests stay lean while algorithms are in flux; a passing test is never runtime acceptance.
- Task ledgers `docs/tasks/2026-07-13/{TASKLIST,KNOWN-ISSUES}.md` stay current enough that
  the user can inspect progress without interrupting. The year is **2026**. Local commits
  at clean checkpoints, ONE task per commit with an explicit file list; **no push**.

## Pointers
- Reality-check pattern + grounding examples: memory `interrogate-constraints-lift-a-level`.
- Never park / drop / reframe / redirect work without surfacing it for the user's call: memory
  `surface-decisions-never-park-silently`.
- Perf-cycle flows (now carry the premise-audit stage): `docs/perf-runs/NEXT-AGENT-CYCLE-PROMPT.md`.
- Microtopography arc: spec `docs/specs/terrain/MICROTOPOGRAPHY.md` (normative), orchestrator
  digest + ledgers in `docs/tasks/2026-07-13/`, engineering rules `AGENTS.md`.
