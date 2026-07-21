# LAAS Engineering Rules

## Delegation

- Use subagents for substantial independent audits or implementation tracks.
- Every subagent prompt must explicitly say: **Do not spawn subagents.** Subagents may never delegate further.
- When model selection is available, use `sol` with high effort only for genuinely difficult synthesis algorithms and consequential scientific/visual judgment. Use normal effort for generation, ingestion, harnesses, routine fixes, and tests.

## Work Efficiency And Track Triage

- Start every substantial track by naming the next outcome that a user or downstream system can actually inspect. Work backward only through dependencies that are strictly necessary to produce that outcome. A commit, schema, contract, audit, test suite, or successful loader is not an outcome unless it is the requested deliverable.
- Treat browser previews, intermediate cooks, and diagnostic artifacts as pit stops unless the requested goal is specifically that pit stop. Always report progress against the whole product objective and separately state how much of the actual quality/coverage problem is solved.
- Default uncertain-track budget: one coherent owner, at most 60 minutes before the first inspectable artifact, and at most two real end-to-end failures before an explicit park-or-continue decision. Exceed this only when the capability is uniquely required and record the reason before continuing.
- After one failed real attempt, diagnose, fix, and rerun as one consolidated owner cycle. Do not serialize separate diagnosis, proposal, critique, implementation, review, rebinding, and verification agents around each small defect.
- If the task has an explicit token or time budget, maintain a simple remaining-budget check at every major outcome. If completion-rate projection exceeds the budget by more than 2x, stop the current workflow immediately and simplify, parallelize independent alternatives, or park low-value tracks. Do not rationalize the overrun through accumulated supporting work.
- Reuse the existing task ledger instead of repeatedly rereading, restating, or expanding the full spec. Reopen primary sources or large design documents only when a concrete unresolved decision requires them.
- Stop research once it has resolved the current method choice and its implementation-critical unknowns. Additional papers, datasets, contracts, or critics need a stated decision they can change; otherwise they are scope expansion.
- Batch documentation, provenance, and checkpoint updates at meaningful outcome boundaries. Do not interrupt the critical path for a new commit, hash freeze, review, or prose update after every local correction unless the operation is irreversible or the binding is required to execute the next real attempt.
- When several approaches, dependencies, data sources, integrations, or implementations could satisfy a requirement, advance the strongest independent candidates in parallel instead of serializing the project behind the most difficult one.
- Give one agent ownership of one coherent track through a real end-to-end outcome. Optimize the split around independently usable results, not tiny implementation/review fragments or maximum agent activity.
- Prefer the first track that meets the actual quality and correctness requirement. If a track becomes disproportionately difficult, preserve its code, provenance, partial artifacts, measurements, and exact blocker, then drop active focus and advance another candidate. Return only if faster candidates fail or the difficult track supplies a unique required capability.
- Dropping focus never means deleting or reverting reusable work. Avoid future reimplementation by committing a clean checkpoint and recording the objective resume condition.
- Do not alternate implementation and independent review after every small checkpoint. One owner should complete a coherent end-to-end track and exercise it in its real environment; perform one independent review at the complete boundary. Add an earlier review only for destructive publication, security, irreversible schema/format changes, or another genuinely high-cost safety boundary.
- Put explicit go/no-go limits on uncertain tracks before starting: maximum focused engineering time, maximum failed real attempts, acceptance requirement, and fallback path. A new failure mode after that limit normally parks the track rather than expanding its local infrastructure.
- Track effort by user-visible, scientifically usable, or operationally deployable outcomes, not commits, tests, schemas, analyses, or audits. Status reports must separate supporting infrastructure from the requested outcome and state what is actually usable now.
- Keep a short waste/resume record for every parked path in the current task's `KNOWN-ISSUES.md`: effort already spent, reusable result, exact blocker, why focus moved, fallback now active, and objective condition for resuming.
- For evidence acquisition specifically, run candidate datasets as independent tracks from source binding through conversion and directly usable artifacts; never let the hardest corpus monopolize the critical path.

### Recorded Efficiency Failures

- The microtopography work over-invested in repeated Hovi E57 decoder micro-fixes, per-commit reviews, evidence scaffolding, and test/audit loops before producing a morphology input. Two full attempts failed on successive terminal-padding interpretations. The reusable decoder work must be preserved, but Hovi may not monopolize the critical path: run Hovi, Evo, and other scientifically eligible targets as separate tracks and advance whichever yields qualified evidence first.
- The project previously expanded test architecture around provisional synthesis and ingestion internals faster than the visual algorithm matured. Keep only focused stable-contract and reproduced-regression tests until the synthesis method survives real-data and visual evaluation.
- The 2026 microtopography program consumed roughly 10 billion tokens while reaching only about 35% of the whole plan, against an expected whole-task budget near 2 billion. The dominant failure was optimizing for evidence closure, preregistration, hashes, small audits, and fail-closed internal gates while delaying visually inspectable morphology. For the remainder of this program, supporting work must directly unblock the next terrain artifact; one owner performs a consolidated real attempt, and any method that consumes two failed end-to-end attempts without visible morphology is parked unless it is demonstrably irreplaceable.

## Research Standard And Feature Decomposition

- The project target is the best achievable result, not an average implementation, common-case approximation, conventional game shortcut, or minimally working feature. Default engineering instincts toward simplification, familiar patterns, and early implementation are a known risk here and must be actively countered.
- Treat the requested real-world fidelity literally. Do not silently reduce scope to what fits one ordinary implementation pass, the current familiar toolchain, or the cheapest method; surface actual evidence, data, compute, representation, and schedule requirements instead. If the full target needs staged delivery, every stage must preserve the final quality ceiling and be labeled honestly rather than presented as the feature itself.
- A feature name is never its specification. Before proposing or implementing it, expand it maximally into the real constituent phenomena, structures, scales, data dependencies, algorithms, and observable quality criteria that make the real thing convincing.
- Research every consequential constituent independently. If a realistic tree is requested, separately investigate trunk architecture, branch hierarchy, twigs, phyllotaxis and leaf geometry/count/distribution, roots, species and age variation, growth/environment response, bark macrostructure, bark microgeometry, wind behavior, and their representations. Bark or leaves may each require their own substantial multi-paper review.
- Begin difficult or quality-defining work with a broad primary-source literature and implementation audit. Read full papers, supplements, theses when relevant, official repositories, issue discussions, datasets, licenses, and strong competing methods. Do not design from abstracts, secondary summaries, a single paper, remembered techniques, or the first plausible search result.
- Explicitly search for prior work whose authors have already spent years on the exact or adjacent problem. Reuse its validated insights and understand its limitations before inventing a local substitute.
- For major method choices, commission independent evidence-backed proposals and separate adversarial critiques. Critics must check claims against the full primary sources and official code, identify unsupported transfer assumptions, and compare quality ceilings rather than implementation convenience.
- Record the background reasoning, source ledger, rejected alternatives, evidence boundaries, and remaining unknowns in repository task/research documents before implementation. An implementer must be able to reconstruct why the chosen method is capable of the target quality.
- Do not collapse a high-fidelity requirement into a toy proxy: a tree is not a pole and sphere, bark is not generic noise, terrain microtopography is not random bumpiness, and realism is not merely "less triangulated."
- Decomposition does not authorize isolated local perfection that breaks the whole. Reassemble the researched constituents into a coherent system and judge the final user-visible result at its intended scale, density, variance, and context.
- Harnesses, formats, ingestion, and tests may use pragmatic engineering, but they must not silently lower the quality ceiling of the difficult synthesis or rendering algorithm they support.
- Keep production proof and research visual investigation distinct, but do not turn ideal evidence gates into a permanent no-output policy. Independent transfer evidence and perfect gate closure are preferred, not absolute requirements. A best-available method may proceed to a labeled research preview on a defensible "generally should transfer" physical-regime hypothesis, and may eventually reach production when it is the only credible zero-budget choice, its remaining defects are minor and bounded, user-visible quality is good, and the alternative is materially worse or no result. Record the uncertainty and abstain outside the matched regime. Do not relax hard safety/integrity gates for catastrophic relief, water/object/protected leakage, broken seams or hierarchy, corrupt packing, runtime instability, destructive regressions, or obviously repetitive/implausible output.

## Project Budget

- Assume the project has no external budget beyond token usage unless the user explicitly states otherwise. Paid surveys, contractors, vendors, proprietary datasets, cloud-scale services, new hardware, and separately funded side projects are not active implementation paths.
- Before concluding that evidence is unavailable or proposing acquisition spend, exhaust legally usable public/open sources, including official portals, research repositories, raw data, and physically comparable regions in Estonia's neighboring countries. Similarity must be justified by substrate, soil, climate, hydrology, land use, vegetation, process, scale, and acquisition semantics rather than geographic proximity alone.
- Costly alternatives may be documented as deferred context only when materially necessary. Do not present them as open decisions or next steps unless the user explicitly reopens budget.

## Real Runtime Acceptance

- A typecheck, unit test, static inspection, HTTP response, or successful asset cook is not a runtime acceptance test.
- Do not attribute runtime failures to Vite, stale browser cache, or old cached assets without direct reproducing evidence that isolates that mechanism. Investigate buffer lifetime, shader validation, asset identity, and application state first; the user normally validates in an incognito browser.
- After every major runtime task and every runtime-facing fix, boot the exact affected configuration in a real WebGPU Chromium using the existing Playwright/WebGPU launch tooling.
- Before giving the user any live URL, boot that exact URL through readiness, cloud bake, and several settled frames. The agent, not the user, is the first WebGPU/compiler test.
- Capture `pageerror`, console errors, and WebGPU/TSL diagnostics during the whole boot. Any TSL invalid-code message, uncaptured WebGPU error, shader validation error, invalid pipeline/bind group/command buffer, uniform-size violation, or non-uniform-barrier error fails the boot even if `window.__laas.ready` becomes true.
- Do not report a URL as ready until the exact served manifest and exact URL boot cleanly. If it fails, fix it and rerun the complete boot before returning.
- Visual-review instructions must be executable from the UI the user actually has. When the world has no compass, coordinates, map, or direction indicator, never ask the user to travel north/south/east/west or "toward" an unstated target. Provide separate exact camera URLs for each review location, or use an unmistakable visible landmark and a directly observable check.
- Do not imply that a packed coverage area is simultaneously visible when fine levels are camera-windowed or radially morphed. State the actual visible radius and use multiple exact start URLs to inspect spatially separated packed regions.
- Do not solve memory failures by merely raising a ceiling. Account for downstream CPU and GPU allocations and WebGPU binding limits first.
- The Estonia review target is `scene=world&src=estonia`, never `scene=estonia`. Center the Taevaskoda pilot at game coordinates `x=311123.082&z=190723.435` (`58.107506 N, 27.050242 E`) and include the actual asset-server `dataurl`.

## Shader Changes

- Treat every shader and TSL line as performance-critical. Never make a shader edit on the basis that it merely "should work."
- For a mathematically complex shader feature or defect, first extract the governing equations, coordinate systems, domains, invariants, and counterexamples into a working Markdown note. Deliberately remove the renderer, WebGPU, shader-language, and pipeline details from this phase.
- Diagnose and solve that extracted problem as pure mathematics before returning to shader implementation. The mathematical replacement must be explicit enough to test independently; only after its identities and limits are correct may it be mapped back into the render path. Do not let shader plumbing and pipeline debugging obscure an unresolved mathematical model.
- Treat runtime brute force as evidence that the mathematical representation is unresolved. Precomputation may be expensive, but a reconstruction designed for low/mid-range hardware must stay a tiny fixed-cost lookup-and-transform path. If a proposed correctness fix multiplies per-pixel candidates, intersections, samples, passes, memory traffic, or ALU, stop and return to the pure-math derivation unless the user has explicitly approved that measured cost. Do not infer permission to brute-force merely because the user did not explicitly forbid a particular form of it.
- Before changing a shader path, state why the change is necessary and reason explicitly about register pressure, workgroup/warp occupancy, divergence and barrier uniformity, ALU cost, memory traffic and locality, binding count and access mode, synchronization scope, dispatch shape, and downstream passes.
- Preserve optimized shader paths unless a measured constraint requires changing them. Prefer the smallest validity fix over opportunistic rewrites while another feature is being integrated.
- Shader compilation and correct pixels are only the correctness gate. After any shader-path change, re-run the exact real-WebGPU boot. For meaningful shader/performance work, prefer a fresh game-only GPU trace and the established Metal/Xcode analysis pipeline in `docs/METAL-PROFILING.md` and `tools/profile/README.md`; use its per-line runtime, generated Metal, resource, and static-occupancy evidence to qualify the change.
- Do not reflexively demand a one-shot before/after frame-time A/B for every shader edit. Thermal history makes casual A/B results misleading. Use controlled, bracketed/interleaved, cooldown-aware A/B only when a trace cannot answer the performance question, and state the noise/thermal controls.
- GPU trace capture is expensive and its performance-data export requires manual Xcode replay. Reuse one fresh trace within a coherent optimization pass, then re-profile after meaningful accumulated changes rather than after every individual line.
- Keep a written performance rationale for each shader change so a later optimization pass does not have to reconstruct intent from the diff.

## Terrain Architecture

- All microtopography synthesis happens in `asset-gen` and is packed into streamed assets. Browser/runtime synthesis is forbidden.
- Runtime may demand, fetch, decode, cache, interpolate, sample, cull, and render packed data. It may not invent terrain detail.
- Runtime abstractions must describe representation and capability, not provenance. Ordinary packed height rungs should flow through the ordinary terrain path; avoid `micro*`/`synthetic*` special handling in `TerrainField` unless the packed format is structurally different and the distinction is justified.
- The canonical base remains LOD 0 at 1 m texels in 2048 m chunks. Fine rungs are LOD -1 at 0.25 m/512 m and LOD -2 at 0.0625 m/128 m. Do not redefine global LOD 0 as 128 m or create two conflicting base grids.
- Fine coverage must retain valid parent closure and terrain mesh coverage. A region may not publish fine rungs while losing the terrain parents needed to render it.
- Maa-amet 1 m DTM is the preferred ground measurement over DSM, which includes canopy/buildings and must not replace it. The DTM is still a fallible observation: interpolation grids, missing returns, water surfaces, vegetation leakage, and other source defects may require evidence-backed cook-side correction. Do not preserve known errors merely to achieve exact up/downsampling.
- Downsampling consistency, quantization, seams, and determinism are necessary constraints, not the visual objective. The objective is beautiful, credible, non-repeating geometric variance conditioned by landform, soil, substrate, and geology.
- Reject decorative hash noise, random per-chunk sine fields, generic FBM, and scene-specific Taevaskoda hacks. Improvements derived from the Taevaskoda reference must generalize to the same physical terrain class elsewhere.
- Read and verify cited papers themselves before making literature-backed method decisions; do not rely only on research-report summaries.

## Surface Agreement

- Terrain changes must not disable or bypass inherited trees, materials, understory, or grass.
- Terrain mesh, grass roots, trees, plants, collision/probes, normals, and materials must sample the same packed surface and the same LOD-transition policy.
- A release is not visually acceptable if grass floats on a flatter surface, vegetation forms elevated islands, chunks show different synthetic patterns, materials disappear, or parent/fine boundaries are visible.
- Every live boot review must include a quick visual check for terrain continuity, material presence, tree/plant presence, and grass grounding.

## Testing Discipline

- Keep tests lean while architecture and algorithms are changing. Add focused tests for stable contracts, release safety, and reproduced regressions; do not build broad brittle suites around provisional internals.
- Tests never replace the mandatory real WebGPU boot.
- Visual acceptance is user-observable ground-level output, not internal metrics alone.
- When a quality-defining asset algorithm has meaningful intermediate geometry, emit a small set of clearly numbered and labeled PNG diagnostics under a content-addressed `data/work/<domain>/<build>/qa/` directory. Include a machine index binding image hashes, source/recipe hashes, dimensions, and interpretations; do not generate screenshots for routine internals merely to satisfy this rule.

## Python Tooling

- Use `uv` for Python execution, environments, and package management. Run scripts and modules through `uv run`, for example `uv run myscript.py` or `uv run python -m package.module`; do not invoke `.venv/bin/python`, `pip`, or another package manager directly.
- Run dependency changes through the appropriate `uv add`, `uv remove`, `uv sync`, or lock workflow so `pyproject.toml` and `uv.lock` remain the reproducible authority.

## Code Organization

- Do not use `process/` or another generic directory as a catch-all. New code belongs to a domain-owned package with a narrow responsibility and explicit dependency direction.
- Keep source acquisition in `fetch/`, evidence inventory/qualification in `evidence/`, terrain interpretation and reconstruction in `terrain/`, packed serialization/hierarchy materialization in `cook/`, and immutable transaction/publish policy in the release layer.
- For substantial terrain work, prefer cohesive subpackages such as `terrain/repair/` over adding unrelated top-level modules. Keep scientific evidence interpretation separate from encoding and runtime delivery.

## Task Records And Git

- Maintain the current microtopography task records at `docs/tasks/2026-07-13/TASKLIST.md` and `docs/tasks/2026-07-13/KNOWN-ISSUES.md`. The year is **2026**, never 2025.
- Keep those files current enough that the user can inspect progress without interrupting the work.
- Develop risky work in its worktree, then integrate the accepted result into the non-worktree `estonia-asset-gen` branch without overwriting unrelated dirty changes.
- Commit the integrated result locally when complete. Do not push.
- When the user explicitly requests autonomous work while AFK, continue without questions or review pauses. At a requested visual checkpoint, provide a verified working URL and pause only after the mandatory boot passes.
