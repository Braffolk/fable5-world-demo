# Moss / ground-cover — HARD RESET to RESEARCH phase (user directive, 2026-07-20)

> Written pre-compaction. Post-compact-me: READ THIS FIRST. The geometry-scatter sphagnum was
> REJECTED by the user in the strongest terms. Do NOT resume it. Do NOT jump to implementation.
> This is a RESEARCH phase. The user does not trust the last approach or its execution.

## What was REJECTED (commit 56caf3e — the CARPET geometry moss)
Flat brown "pancakes" of low-poly geometry scattered over the terrain. Concretely wrong:
- Patches FLOAT above the terrain, intersect each other, cast huge BLACK VOIDS underneath,
  disconnected from the substrate (no transition into the wet ground).
- ONE dead "diarrhoea brown" color smeared across hundreds of identical patches.
- Crude, flat, low-poly, repetitive; random spiky low-poly bits poking out of the edges.
- No cushion structure, no fine-scale density, no upright branching, no tiny clustered leaves,
  no wet translucency, no layered depth, no accumulated dead material under living tips, no
  moisture response, no specular variation, no soft self-shadowing, no organic overlap.
- Incoherent scale, obvious procedural-random distribution (the cheapest kind).
This is a PS1-era swamp texture on a cardboard box. Unacceptable. Not photorealism.

## THE MANDATE (what "done" means)
A PHOTOREALISTIC sphagnum moss ground layer in a WebGPU renderer. Real sphagnum:
- Dense, intricate UPRIGHT BRANCHING; tiny clustered leaves; irregular CUSHIONS; wet
  translucency; layered depth; soft interlocking forms; visible VARIATION AT EVERY SCALE.
- COLOR varies: vivid green, yellow-green, ochre, rust, red, pale straw, water-darkened tones
  — by species, moisture, age. NOT one muddy brown.
- Grows as a living carpet integrated INTO the wet substrate, not sitting on top of it.

## DO NOT (anti-patterns the user pre-empted)
- Do NOT do it as scattered discrete GEOMETRY patches (the rejected approach).
- Do NOT "just make grass blades sparser" for density variation — that is the cheap idiot trick
  the user explicitly called out. Density must be a REAL, controllable field.
- Do NOT jump into coding. RESEARCH first. Do NOT copy-adapt one plausible hit; study broadly.

## THE USER'S RUNNING IDEA (their hypothesis — explore + pressure-test, don't blindly obey)
Moss should come from the renderer's PRECOMPUTED GRASS RAYCAST system, not geometry:
  a) OPTIMIZE the hell out of the existing precomputed grass raycast system (it has never been
     gotten fully right — the user cites it as a standing failure; the ≤2ms ray-lane law is UNMET
     per memory `grass-raycast-lane` / `grass-lushness-law`).
  b) EXPAND it into a general GROUND-COVER system able to handle MANY cover types incl. the
     intricate requirements of moss — with REAL density variation, OVERLAY of multiple cover
     types, per-type varying (or absent) WIND effects, etc.
Treat this as the leading direction, but the RESEARCH must confirm/shape HOW (raymarched shells,
volumetric/heightfield ray-marching, parallax/shell texturing with depth, SDF, density fields,
screen-space or precomputed lighting for translucency+self-shadow, etc.), grounded in real
technique — not assumed.

## RESEARCH SOURCES (the user's list — study NOVEL rendering tricks of great graphics programmers)
- https://www.jendrikillner.com/article_database/  (the graphics-programming article database)
- https://advances.realtimerendering.com/  (SIGGRAPH Advances in Real-Time Rendering)
- https://gpuopen.com/learn
- https://www.gamedev.net/  graphics guild
- https://habr.com/en/hubs/gamedev/posts/
- https://80.lv/articles
- Shadertoy (real-time procedural moss/ground/volumetric shaders)
- sizecoding (demoscene — extreme procedural detail cheaply)
- gamedev.ru
- (not exhaustive — cast wide)
Goal: a TECHNIQUE MENU with tradeoffs for a photoreal moss/ground-cover layer that fits a custom
WebGPU nanite-like streaming renderer, plus the grass-raycast-optimization path. Prioritize proven
techniques from named practitioners; capture the actual mechanism + perf + how it adapts here.

## EXISTING CONTEXT to fold in
- Prior UE Nanite-Foliage research (docs/memory this session) was about GEOMETRY aggregation — now
  DE-PRIORITIZED (user is moving away from geometry). Keep only as a contrast/reference.
- The existing grass ray lane: memory `grass-raycast-lane` ("ray lane = THE grass"), `grass-lushness-law`,
  the ≤2ms law UNMET; `src/nanite/**` grass kernels. This is the system to optimize + generalize.
- The CarpetSpec seam (carpet/CarpetTypes, SphagnumCarpet) + the 6 cleanups from 56caf3e: the
  cleanups are keepers; the sphagnum GEOMETRY (Sphagnum.ts) + its scatter wiring are the rejected
  part. Whether the CarpetSpec seam is repurposed for the raycast approach or reverted is a
  DESIGN decision for after the research — do NOT reflexively keep or revert it.

## TRUST / TIERING context
- The user DOES NOT trust the last approach or opus's execution on rendering. Rendering research +
  design + implementation = FABLE tier (the user's law: opus can't do this stuff). Ground every
  quality-defining claim in real primary sources; no toy proxies; no "should work".
- Standing laws still apply: photorealism bar, no fake tricks that aren't geometrically/physically
  honest (BUT legitimate render techniques — raymarching, shells, translucency, precomputed
  lighting — ARE the point here; "no fake" targets fabricated surface noise, not real rendering).

## UPDATE 2026-07-20 (post-compact) — SCOPE BROADENED TO THE WHOLE GRASS+MOSS LANE
User re-scoped after compaction: this is NOT just moss. The GRASS layer fails the bar too. The task is
to evaluate the fit of our existing grass raycast lane as the basis for ONE generalized ground-cover
system (grass + moss + other covers), and produce a research-backed design.
- TWO HARD BARS (both non-negotiable): (1) moss/grass/covers INDISTINGUISHABLE from the real world;
  (2) SIGNIFICANTLY FASTER than now. Reality check: our lane costs ~+14.5 ms eye (≤2 ms law), so today's
  grass is ~7× over budget AND not photoreal. "Impossible" is banned — this is a many-times-solved task.
- THE ALGORITHM IS LOCAL: `docs/legacy/deep-review/grass-raycast.txt` = Aleksandr Sannikov's O(1)
  precomputed-raycast article (RU). Research agents MUST read it IN FULL. Our lane IS this algorithm.
- Sannikov Shadertoys the user pointed at: https://www.shadertoy.com/view/tsVGRd (the main linked one)
  and https://www.shadertoy.com/view/Ds3fWH (a newer one on his account).
- TWO NAMED GRASS DEFECTS to explain + solve: (D1) "it's only grass blades" — real grass is ≥10 species
  in irregularly-SHAPED patches with VARYING patchiness (not per-blade random, not uniform) → need real
  multi-species + spatially-structured density. (D2) jitter/deformation-looking artifacts.
- Our lane files: `src/nanite/grass/NaniteGrass.ts` (~1676 lines) + `src/nanite/build/GrassRayBake.ts`
  (~276), referenced from NaniteFrame/NaniteResolve/NaniteVoxelRaster. Grass memory: grid-lesson
  (world-pos sampling, not per-texel), lushness law (match GroundRing density), +14.5 ms perf state.
- Research agents = **Fable-5 @ xhigh**, read the article in full (user law). Workflow
  `wf_5fdab918-fad` running: Ground(article teardown + our-code teardown) → Research(6 clusters:
  raycast-lineage / species+patch diversity / moss / jitter+temporal / density+overlay+wind /
  shadertoy+demoscene) → Evaluate(GO-UP-A-LEVEL premise-audit + fit verdicts) → Synthesize(design +
  technique menu). Output = a design for USER APPROVAL before any code. Design doc will land at
  `docs/tasks/2026-07-25/groundcover/active/GRASS-MOSS-GROUNDCOVER-DESIGN.md`.

## HARD REQUIREMENT (added 2026-07-20) — SPATIAL GROUND-COVER CONTROL FIELD
> User law, stated as an ADDITIVE guardrail: "it's absolutely critical that things like how dense/fluffy
> this covering is and what even is here can be read in the raycast shader through some buffer that tells
> us what ground covering to expect everywhere and how much of it — e.g. under trees less grass, in swamps
> we get grass/sedge, etc." **DO NOT reduce the requirements to this — EXPAND, don't shrink.** These
> guardrails exist so requirements don't go missing; the floor is below, add whatever realism/perf needs.

The cover shader (the raycast lane or whatever we land on) MUST sample, at every world position, a
GROUND-COVER CONTROL FIELD (a buffer/texture set) that authoritatively answers "what cover is here and
how much" — so cover responds to real context instead of being uniform:
- **WHAT** — cover-type COMPOSITION as a MIX (not single-valued): which of the ≥10 grass species +
  moss/sphagnum + sedge/cotton-grass + lichen + herb/flower + bare are present here.
- **HOW MUCH** — density / coverage fraction / FLUFFINESS / height / vigor, per type, continuously
  varying and REAL (never the forbidden "sparser blades" trick).
- **CONTEXT-DRIVEN** — less+shorter grass and more moss UNDER TREE CANOPY; swamp/bog → sedge + sphagnum
  + cotton-grass; dry/fertile → lush meadow; slope/aspect/moisture/soil/disturbance all modulate.
  Derived COOK-SIDE from REAL cooked layers (canopy height, wetness/hydrology, soil/fertility, biome,
  classId, elevation/slope) — precompute-in-assetgen law — streamed with terrain tiles, sampled cheaply
  in-shader as an EVOLUTION of the current guide/ctx field (which already carries ground/grad/topOff/
  gustAmp + a density mask per texel).
- **PATCH STRUCTURE** — encodes both smooth gradients AND irregular, VARYING-patchiness patches (D1):
  patch shapes and the degree of patchiness themselves vary spatially.
- **OVERLAY + WIND** — multiple co-located cover layers (moss under grass), each with its own density
  and its own wind / no-wind response, all keyed off the field.
- **EXTENSIBLE** — new cover types/species addable without rearchitecting; the field is the single
  source of "what grows where, how much of it, and how it moves."
EXPANSION MANDATE: treat the above as a FLOOR. Add any further control the photoreal+perf goals need —
e.g. per-type color/hue-variation fields, moisture-driven wet/specular, seasonal/age variation, patch
edge/transition control, per-type LOD handoff. NEVER collapse the full photoreal + overlay + density +
wind + species + patch requirement down to "just a density map."

## FIRST ACTIONS post-compact (RESEARCH, not code)
1. Re-read this doc + memory `grass-raycast-lane`, `grass-lushness-law`, `beautification-arc-2026-07`.
2. Launch a FABLE deep-research pass (KEEP IT SMALLER than the 98-agent one — user law) over the
   sources above for: (a) photoreal moss / dense ground-cover real-time rendering techniques, (b)
   ray-marched / shell / heightfield / SDF ground-cover with density fields + translucency +
   self-shadow, (c) grass-raycast-lane optimization techniques. Return a technique menu + tradeoffs.
3. Study the existing grass ray lane code to ground the "optimize + generalize" path.
4. Bring the user a research-backed DESIGN (technique choice + how it optimizes/generalizes the ray
   lane for moss + other covers) for approval BEFORE any implementation.
