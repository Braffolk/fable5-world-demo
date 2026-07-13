# LAAS Engineering Rules

## Delegation

- Use subagents for substantial independent audits or implementation tracks.
- Every subagent prompt must explicitly say: **Do not spawn subagents.** Subagents may never delegate further.
- When model selection is available, use `sol` with high effort only for genuinely difficult synthesis algorithms and consequential scientific/visual judgment. Use normal effort for generation, ingestion, harnesses, routine fixes, and tests.

## Real Runtime Acceptance

- A typecheck, unit test, static inspection, HTTP response, or successful asset cook is not a runtime acceptance test.
- After every major runtime task and every runtime-facing fix, boot the exact affected configuration in a real WebGPU Chromium using the existing Playwright/WebGPU launch tooling.
- Before giving the user any live URL, boot that exact URL through readiness, cloud bake, and several settled frames. The agent, not the user, is the first WebGPU/compiler test.
- Capture `pageerror`, console errors, and WebGPU/TSL diagnostics during the whole boot. Any TSL invalid-code message, uncaptured WebGPU error, shader validation error, invalid pipeline/bind group/command buffer, uniform-size violation, or non-uniform-barrier error fails the boot even if `window.__laas.ready` becomes true.
- Do not report a URL as ready until the exact served manifest and exact URL boot cleanly. If it fails, fix it and rerun the complete boot before returning.
- Do not solve memory failures by merely raising a ceiling. Account for downstream CPU and GPU allocations and WebGPU binding limits first.
- The Estonia review target is `scene=world&src=estonia`, never `scene=estonia`. Center the Taevaskoda pilot at game coordinates `x=311123.082&z=190723.435` (`58.107506 N, 27.050242 E`) and include the actual asset-server `dataurl`.

## Shader Changes

- Treat every shader and TSL line as performance-critical. Never make a shader edit on the basis that it merely "should work."
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
- Maa-amet 1 m DTM is authoritative ground elevation. DSM includes canopy/buildings and must not replace the DTM.
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

## Task Records And Git

- Maintain the current microtopography task records at `docs/tasks/2026-07-13/TASKLIST.md` and `docs/tasks/2026-07-13/KNOWN-ISSUES.md`. The year is **2026**, never 2025.
- Keep those files current enough that the user can inspect progress without interrupting the work.
- Develop risky work in its worktree, then integrate the accepted result into the non-worktree `estonia-asset-gen` branch without overwriting unrelated dirty changes.
- Commit the integrated result locally when complete. Do not push.
- When the user explicitly requests autonomous work while AFK, continue without questions or review pauses. At a requested visual checkpoint, provide a verified working URL and pause only after the mandatory boot passes.
