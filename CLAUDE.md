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

## Pointers
- Reality-check pattern + grounding examples: memory `interrogate-constraints-lift-a-level`.
- Never park / drop / reframe / redirect work without surfacing it for the user's call: memory
  `surface-decisions-never-park-silently`.
- Perf-cycle flows (now carry the premise-audit stage): `docs/perf-runs/NEXT-AGENT-CYCLE-PROMPT.md`.
