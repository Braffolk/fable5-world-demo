# Microtopography Task List

Last updated: 2026-07-13

## Objective

Ship a visually credible, deterministic, cook-side microtopography pilot at Suur Taevaskoda. The browser only streams and renders packed height data; it never synthesizes terrain detail.

## Current State

- [x] Re-audit the dirty, unpushed implementation and isolate new work in a clean worktree based on `258ddef`.
- [x] Restore the canonical shared grid: height LOD0 remains 1 m with 2048 m chunks; negative height-only rungs provide 0.25 m and 0.0625 m samples.
- [x] Implement signed negative-LOD transport, LAC2 decoding, near-only demand, and packed-height sampling.
- [x] Fix fine-height use by terrain and grass without disabling inherited trees, materials, or vegetation.
- [x] Compile procedural `terrainDispAt` geometry out of format-2 Estonia terrain and grass roots; packed terrain is now the only geometric surface while generated/legacy worlds retain existing behavior.
- [x] Center the hero coverage on `58.107506 N, 27.050242 E` in LOD -1 parent `(607,372)`.
- [x] Select and pin the LUKE Lapinjarvi TLS pilot corpus: k11/k32/k36 calibration and k19 holdout, CC BY 4.0.
- [x] Reject the first TLS ground extraction because vegetation/object returns formed elevated terrain islands.
- [x] Replace extraction with a center-seeded, connected, multi-view ground-manifold method and strict fail-closed gates.
- [x] Rerun corrected extraction on all four real TLS plots and inspect numerical plus rendered surface QA.
- [x] Approve extraction-valid surfaces and build the measured residual bank: 149 calibration patches (k11=42, k32=49, k36=58); k19 excluded as holdout.
- [x] Fix the half-texel synthesis coordinate contract.
- [x] Wire the measured synthesis cook CLI and recipe-kind checks.
- [x] Add bounded production-preview verification and prohibit pilot publication to `latest`.
- [x] Apply direct fine-grid ETAK masks for open water, buildings, and paved roads; deliberately retain unpaved trails.
- [x] Cook the centered 512 m hero closure and materialize an immutable local preview (`5f845ae2...`, manifest `b1c7c44bad667fd7`) with all independent packing/mask gates passing.
- [x] Restrict the Lapinjarvi matrix to exact ETAK forest plus fail-closed unmodified mineral/mesic soil context. Published hero allowance is `53.3685%`; four chunks correctly receive zero residual and unsupported samples remain base-only.
- [x] Replace the fixed 3 m normalized-crossfade quilt with crop-stable equal-power overlap, normalized edge-shape cost, and balanced `top_k=12`; focused calibration diagnostics restore bank RMS/tails and effective use to `140.3/149` patches.
- [x] Replace the inherited per-1 m bubble projection with one AOI-wide smooth mean-null solve plus fail-closed boundary taper. The real 5x5 closure has exact 1 m means (`8.83e-17 m`), zero hard-mask residual, zero shared-overlap error, no `0.747 m` support spike, and stays inside the measured bank amplitude/slope envelope.
- [x] Replace hard finest-level switching with a format-2-only packed-level geomorph shared by terrain, grass, material normals/slopes, and CPU ground probes. The former `14.96 cm` (-2/-1) and `34.89 cm` (-1/0) ownership steps now fade through bounded quintic transition rings.
- [x] Replace the eager raster prepass allocation with a compact projected-record pool and a shared 96 Ki cluster domain. Project/context allocation drops from about `716 MiB` to `133.5 MiB`, with fail-closed observable overflow.
- [x] Add a strict real-Chromium WebGPU boot gate and fix every exposed validation failure: redundant texture uniforms, non-uniform workgroup barriers, writable storage aliases, and mixed access modes in a single synchronization scope.
- [ ] Inspect the supplied Suur Taevaskoda reference viewpoint and categorize generalizable geometry mismatches.
- [x] Correct the premise audit: the main vertical/undercut sandstone wall is not representable by a heightfield. Specify a separate cook-packed structural terrain cluster path with no runtime synthesis.
- [ ] Map the structural terrain contract onto the repository's actual Nanite cluster, manifest, streaming, material, collision, and vegetation systems; implement only after the codebase audit and user-visible height pilot are stable.
- [x] Confirm and locate anomalous pinned-base height returns inside the mapped Ahja river near the reference anchor.
- [ ] Implement and validate a general robust river-surface/base-DTM conditioning pass in the upstream water cook, then decide whether to approve a new pinned base release. Do not patch only the hero coordinates or silently change negative LOD authority.
- [x] Correct the durable spec so raw TLS extraction, diagnostic-only LUKE scope, matrix/event separation, conditional orthophoto use, global projection, non-heightfield structural terrain, executable conditioning requirements, and fixed Taevaskoda review are explicit.
- [x] Materialize and independently verify immutable recipe `090747d74911...`, manifest `eaa0444646b47de1`. All coverage, header, apron, hierarchy, determinism, and decoded hard-mask gates pass; `latest` remains untouched.
- [x] Complete the initial user visual checkpoint at the centered live preview on `:5180` / immutable data endpoint `:8791`; terrain, material, grass, trees, and fine Nanite clusters are present.
- [x] Pass three clean exact-URL WebGPU boots after removing diagnostics, including cloud bake, settled frames, and zero page/console/TSL/WebGPU errors.
- [x] Integrate clean work into the main `estonia-asset-gen` worktree as local commit `0ef75cf` without pushing; superseded root edits remain preserved in a safety stash and were not reapplied.

## Deliberately Deferred

- National or multi-landform rollout.
- Peat, agricultural, carbonate, and pit-mound event synthesis without matching measured exemplars.
- Orthophoto-conditioned morphology until the packed measured pilot is visibly sound.
- Broad test expansion while extractor, generator, and release contracts remain in active change.
