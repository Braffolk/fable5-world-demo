# Apple / mobile-GPU perf research + audit — index

Everything from the 2026-07-05 Apple/mobile deep-forest investigation (task #72/#73). All review
agents' recommendations are saved as md here for cross-session reference.

## ⭐ START HERE (consolidated action plans — read these first)
- **[DEEP-FOREST-AUDIT.md](DEEP-FOREST-AUDIT.md)** — THE main plan. Full-renderer deep-forest audit
  (the real target). Cost model + ranked levers L1–L8 + instrument-first sequence. Corrects the
  framing: base is 17.4ms (not 20.8); the whale is the ~8ms near/mid **leaf-crown MESH raster**;
  `c.nanVisClear`'s big timestamp is an **overlap artifact**, not real cost; shadows are ~0.5ms static.
- **[WEBGPU-FEATURES.md](WEBGPU-FEATURES.md)** — WebGPU/WGSL feature availability Chrome 120→150
  (Jul 2026). Corrects "fp16 blocked": **shader-f16 is available (Chrome 120, our device auto-requests
  it)** — only TSL-blocked (needs raw-WGSL). **subgroups (Chrome 134)** → ~32× atomic cut via
  subgroup queue-compaction in cull. Detection + fallback patterns.
- **[MASTER-AUDIT.md](MASTER-AUDIT.md)** — the grass sub-plan (grass is ~5ms, secondary). Ranked
  grass levers + the user-gated analytic-blade question (2.36ms vs baked tile).
- **[CHANGES.md](CHANGES.md)** — implementation changelog (what's been changed, status, verify).

## Cost data (measured, ground truth)
- **[deep-forest-breakdown.md](deep-forest-breakdown.md)** — honest gpuWall ablation of the 17.4ms
  deep-forest base (shadows ~0.5, leaf-mesh ~4.1, crownlod0 ~3.9, voxels load-bearing −54). ⚠️ Apple
  render‖compute OVERLAP → per-pass timestamps are NOT additive; ablation is truth.

## Per-agent detailed findings (the raw recommendations behind the syntheses)
- **[DEEP-FOREST-FINDINGS.md](DEEP-FOREST-FINDINGS.md)** — 8 subsystem audits (visclear/buffers, SW
  raster, voxel scatter, cull+HZB, resolve, shadows, post, frame/WebGPU), file:line grounded.
- **[WEBGPU-FEATURES-RAW.md](WEBGPU-FEATURES-RAW.md)** — per-area feature research (f16, subgroups,
  immediates, atomics/status/detection).
- **[GRASS-AUDIT-DETAILED.md](GRASS-AUDIT-DETAILED.md)** — the grass/mobile per-theme findings.
- **[CROSS-CHECK.md](CROSS-CHECK.md)** — the first grass audit (grounded grass levers + constraint
  corrections: no FP16 in TSL, keep cooperative workgroups, keep fragment resolve).
- **[RED-TEAM-3-IDEAS.md](RED-TEAM-3-IDEAS.md)** — adversarial red-team of the 3 stutter ideas
  (visibility-reprojection / shading-cache / splatting — all refuted, with the grounded reasons).

## Source material (fetched articles / guides)
- **[compass_artifact_...md](compass_artifact_wf-eaa18864-7f52-401e-ac36-921b9e306e9e_text_markdown.md)**
  — the ~60-source deep-research digest (URLs + relevance + synthesis + caveats).
- occupancy-interplayoflight.md · apple-tbdr-blakecrosley.md · shader-opt-persson.md ·
  wwdc-10859-tailor-metal-m1.md · arm-mali-guide.md · `sources/` (Apple reverse-eng, mobile vendor
  guides, occupancy, visbuffer, WebGPU/WGSL) · hypehype-advances-2023.pdf.

## The one-paragraph state
Deep dense forest on M1 Max is ~17–21ms base @dpr2 (grass off). The whale is the near/mid leaf-crown
MESH-triangle SW raster (~8ms, the whole compute stream) — a DAG that doesn't coarsen the near field
(same disease as voxels/bark). Levers that survive the rscale ban: coarsen/voxel-shift the leaf-crown
band (L1, quality-gated, user call), two-pass resolve to recover übershader occupancy (L2, simple flip),
ping-pong vis buffers to restore render‖compute overlap (L3), voxel megakernel split (L4, Metal-gated),
post half-res (L5), shadow moving-cadence for the stutter (L6). Newly-unblocked: shader-f16 (raw-WGSL)
+ subgroup queue-compaction. Instrument-first; ablate, never trust per-pass timestamps.
