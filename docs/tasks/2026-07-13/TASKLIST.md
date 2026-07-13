# Microtopography Task List

Last updated: 2026-07-13

## Objective

Ship beautiful, physically credible, materially and geomorphologically conditioned cook-side microtopography across Estonia. The browser only streams and renders packed height data; it never synthesizes terrain detail. Suur Taevaskoda is a fixed visual reference for generalizable improvements, not a scene-specific target.

## Current State

### Accepted infrastructure

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
- [x] Inspect the supplied Suur Taevaskoda reference and categorize the representable shore/bank/top-surface failures separately from the vertical/undercut wall.
- [x] Correct the premise audit: the main vertical/undercut sandstone wall is not representable by a heightfield and is excluded from this pass. Any non-heightfield structural solution requires its own future research and user-reviewed spec.
- [x] Confirm and locate anomalous pinned-base height returns inside the mapped Ahja river near the reference anchor.
- [ ] Implement and validate a general robust river-surface/base-DTM conditioning pass in the upstream water cook, then decide whether to approve a new pinned base release. Do not patch only the hero coordinates or silently change negative LOD authority.
- [x] Supersede the earlier spec's false raw-DTM consistency objective and record diagnostic-only LUKE scope, source-repair authority, heightfield limits, executable conditioning, and fixed Taevaskoda review requirements.
- [x] Materialize and independently verify immutable recipe `090747d74911...`, manifest `eaa0444646b47de1`. All coverage, header, apron, hierarchy, determinism, and decoded hard-mask gates pass; `latest` remains untouched.
- [x] Complete the initial user visual checkpoint at the centered live preview on `:5180` / immutable data endpoint `:8791`; terrain, material, grass, trees, and fine Nanite clusters are present.
- [x] Pass three clean exact-URL WebGPU boots after removing diagnostics, including cloud bake, settled frames, and zero page/console/TSL/WebGPU errors.
- [x] Integrate clean work into the main `estonia-asset-gen` worktree as local commit `0ef75cf` without pushing; superseded root edits remain preserved in a safety stash and were not reapplied.

### Rejected synthesis and current research phase

- [x] Reject the LUKE quilt / mean-null residual synthesizer as a production or beauty method after user-visible review. It adds sparse noise-like bumpiness, does not repair 1 m grid shorelines or source defects, lacks Estonia substrate/process grounding, and is not capable evidence for national 6.25 cm detail.
- [x] Retain the proven negative-LOD cook, packing, serving, streaming, decoding, rendering, masks, terrain/grass grounding, and strict real-WebGPU boot infrastructure; disclose the remaining generic tree/boulder/understory/debris packed-surface sampling gap without redesigning the format or adding runtime synthesis.
- [x] Record the repository-wide research and feature-decomposition standard in root `AGENTS.md`.
- [x] Inventory every paper, repository, dataset, and consequential claim cited by `r1.md`, `r2.md`, and the superseded spec; correct citation conflations and report overclaims. The closed v2 ledger has 125 unified records including 21 repositories and exact one-to-one classification of all 127 checksum-manifest artifacts; the uncited Burren dimensions remain explicitly excluded.
- [x] Download every legally accessible full paper and identified repository artifact; record canonical URLs, licenses, hashes, access failures, demonstrated scales, officialness, and review depth without equating a valid tarball with code review. All 127 recorded local artifacts pass the checksum ledger.
- [x] Produce independent learned-model and deterministic/process/exemplar proposals grounded in full primary sources and the explicitly bounded repository evidence; consequential reuse still requires officialness/license/code-review closure.
- [x] Commission separate adversarial critics for each proposal; proposal authors did not critique their own work.
- [x] Expand the synthesis problem into Estonia's actual surface-forming regimes, observation-error correction, covariates, representation limits, and sub-decimeter training/validation requirements (`review/REGIME-PHENOMENON-MATRIX.md`, `review/TARGET-SCALE-DATA-AND-REGIME-AUDIT.md`).
- [x] Audit current and official Estonia-wide conditioning inputs, including source fidelity loss, EGT coverage, and Taevaskoda's mapped ETAK/soil constraints (`review/ESTONIA-CONDITIONING-DATA-AUDIT.md`).
- [x] Audit the actual cook, hierarchy, correction-authority, coverage, verifier, storage, and unchanged-runtime contract (`review/COOK-CONTRACT-AND-HIERARCHY-AUDIT.md`).
- [x] Decide orthophoto acquisition: fetch/version dated RGB/CIR as masked research and likely production conditioning, never as metric height truth; complete the national manifest-size scan before bulk acquisition.
- [x] Write the comparative architecture dossier and source/claim ledger (`review/COMPARATIVE-ARCHITECTURE-DOSSIER.md`, `library/SOURCES.md`, `library/sources.json`).
- [x] Choose one decisive evidence-gated typed-hybrid architecture: deterministic evidence fusion and regime graph, per-regime target-supported specialist bakeoff with valid no-winner, deterministic absolute-surface composition, and unchanged packed-height delivery.
- [x] Add executable regime/evidence, hierarchy/quantization, visual-review, and storage/cost contracts; independently critique the first spec rewrite and correct its water, authority, hierarchy, visual, and staging contradictions.
- [x] Normalize the source ledger to schema v2 and retain the first Stage 1 method contract as an explicitly superseded negative-design record after final audit disproved its Moore ownership test (`library/sources.schema.json`, `review/STAGE1-EXECUTABLE-METHOD-CONTRACT.md`).
- [x] Materialize the strict 26-row regime schema, initial registry, and status manifest; final audit leaves all production morphology unsupported and Moore target evidence insufficient rather than inventing an owner.
- [x] Rewrite `docs/specs/terrain/MICROTOPOGRAPHY.md` from the audited evidence, with one decisive replacement architecture and executable structural, evidence, release, runtime, and visual gates.
- [x] Run independent science/beauty, repository/cook, and standalone-evidence audits of the integrated spec and close every critical/high finding without hiding introduced runtime/release/data dependencies.
- [x] Record the user's zero-external-budget decision: paid Estonia acquisition and a separately funded non-heightfield cliff project are rejected unless explicitly reopened.
- [x] Audit public equivalent target data in Latvia/Lithuania, Finland/Sweden, and Estonia/Taevaskoda through separate primary-source evidence hunts; classify actual qualification rather than nominal resolution.
- [x] Fold qualifying public-data findings and the closed product decisions into the durable spec, move it to `docs/specs/terrain/MICROTOPOGRAPHY.md`, and update every authority reference. Candidate binaries enter the machine ledger only when selected for conversion.
- [x] Check in `PUBLIC-TARGET-QUALIFICATION-PROTOCOL.md` and machine ledger records for the exact selective Hovi Järvselja/Hyytiälä files and exact Taevaskoda ALS epochs before retaining or converting them.
- [x] Freeze the bounded zero-cost conversion inputs: 43 Hovi files totaling 7,241,800,875 bytes, eight official Taevaskoda ALS epochs, and one 351,166,481-byte Biała Góra calibration epoch; do not authorize bulk Biała acquisition.
- [x] Add an independently audited, content-addressed Taevaskoda ALS retention path that binds the frozen selection, source/license snapshots, response provenance, actual bytes, and partial/complete state.
- [ ] Retain and inventory the eight Taevaskoda ALS epochs; keep them `calibration_only` for typed structural repair and never treat them as 6.25 cm morphology truth.
- [ ] Implement Stage 1 Taevaskoda typed structural repair and corrected parent hierarchy, run the bounded real-WebGPU boot, and provide the immutable live URL for visual review.
- [ ] Implement a morphology specialist only after its public target subset passes conversion, support/error/semantics QA, analogue/OOD qualification, and a target-specific preregistration.

## Deliberately Deferred Until The Research Review

- Any further beauty claim or national rollout of the rejected LUKE synthesizer.
- Peat, agriculture, carbonate/alvar, sandstone, shore/coast, fluvial, wetland, forest-event, engineered-ground, and other regime synthesis without matching evidence and exemplars.
- National orthophoto bulk acquisition until the full official manifest is scanned for size and the research ablation proves useful geometric conditioning rather than albedo-to-height leakage.
- Broad test expansion while extractor, generator, and release contracts remain in active change.
- Any non-heightfield cliff representation; it is a separate future research/spec task, not a hidden extension of this heightfield pass.
