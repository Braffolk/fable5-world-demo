# Final Adversarial Critique Of `SPEC-MICROTOPOGRAPHY.md`

**Audit snapshot:** SHA-256 `806936d1c543337c1812211c3029a7a5ba811ea97b3cb108df3c4cbd7a4ccecd`, 2026-07-13
**Scope:** specification review only; this review made no edits to the specification
**Verdict:** **not ready for implementation approval.** The architecture decision is defensible, but the draft still has three implementation-blocking contradictions and several high-severity omissions that can reproduce a technically coherent but visually weak result.

## Critical Findings

### C1. The staged proof cannot produce the submeter shoreline it promises

**Spec:** Sections 14.3 and 18, especially lines 737-745, 994-1000, and 1019-1023.

Section 14.3 says the next proof is one published LOD `-1` parent plus 16 LOD `-2` children. Stage 1 Track B instead publishes only corrected LOD0 and ancestors, while Stage 3 defers those negative rungs until a morphology regime is supported. Stage 1 nevertheless promises to prove that the visible 1 m shoreline staircase is gone at ground level. A 1 m LOD0 terrain mesh cannot display the specified 0.25 m structural shoreline, and typed structural refinement does not require or justify waiting for an unresolved-morphology prior.

**Required correction:** make Track B publish a **structural-only** format-2 negative hierarchy for the existing one-parent coverage: corrected structural LOD `-2`, its decoded-child-derived LOD `-1`, corrected LOD0, and affected ancestors. The unresolved morphology contribution may be identically absent; that is not unsupported synthesis. Stage 3 may later replace the structural-only fine master with a same-regime morphology realization. Remove the requirement that a supported morphology regime must exist before representable source repair is published, and make Sections 14.3 and 18 describe the same proof.

### C2. The water-bed cook order is internally invalid

**Spec:** Sections 10.2 and 14.2, especially lines 480-498 and 718-731.

Section 10.2 correctly treats optical water, submerged bed, shoreline, bank, and escarpment separately. Section 14.2 then derives and publishes the complete height hierarchy before step 12 "recook[s] water/watercover/bed." In the accepted representation, height stores the submerged bed. Changing bed content after deriving LOD0-LOD4 invalidates that hierarchy; leaving bed unchanged while water/shore changes violates the one-snapshot compatibility rule. The phrase "existing waterbed policy" also hides the exact semantic contract a zero-context implementer needs.

**Required correction:** state normatively that packed height stores dry terrain plus submerged bed, `waterY` stores the visible water surface, and `watercover` stores anti-aliased occupancy. Derive a compatible `waterY`/`watercover`/shore/bank/bed solution from one evidence snapshot; compose the bed into structural height **before** canonical fine composition and all parent reduction; then cook matching water products and dependent layers. Any later bed change must invalidate and rederive every affected height parent and apron. Define conservative unknown-bathymetry behavior and prohibit deriving bed relief from optical water appearance.

### C3. The quality-defining algorithms are requirements placeholders, not an implementable specification

**Spec:** Sections 10.1, 12, 13.2-13.3, and Stage 0, especially lines 461-476, 557-628, 647-684, and 970-979.

The source-repair objective does not select an observation operator by repair class, a robust loss, hard-versus-soft constraint priority, discretization, boundary condition, calibration procedure, solver/tolerance, or acceptance threshold. The analysis pyramid says that an implementer must define a filter, phase, transfer function, and boundary extension later. The stochastic contract does not name a PRF, key serialization, coordinate quantization, window lattice, influencing-window enumeration, or fusion quantity. The specialist section names seven families without giving reproducible Stage 1 candidate recipes. Those choices determine whether the output is realistic morphology or another smooth/noisy proxy; they cannot be delegated silently to implementation.

**Required correction:** either add an exact, executable Stage 1 method contract for each of those items, or explicitly limit approval of this document to evidence/schema work and require a second user-reviewed scientific sub-spec before any repair or synthesis algorithm is coded. For every candidate, freeze architecture/configuration, input tensors, physical context, objective, degradation model, train split, inference windows/fusion, stochastic identity, and compute allowance before the bakeoff. For the hierarchy/analysis path, specify coefficients, support, phase, edge/apron handling, and reconstruction relation rather than the instruction to define them later.

## High-Severity Findings

### H1. The file map misstates the current `api.py` contract

**Spec:** Section 19.1, lines 1049-1057.

The current `asset-gen/src/assetgen/process/microtopo/api.py` is an exemplar-bank facade and exposes `synthesize_residual`; it is neither provenance-neutral nor an absolute-surface boundary. Saying to "keep" it contradicts Sections 4.3 and 7 and invites the old residual authority back into the replacement.

**Required correction:** require `api.py` to be replaced or explicitly redesigned around typed absolute `StructuralAuthority`/`SpecialistOutput`/`FinalMaster` records, with no implicit raw-parent projection. Add `cook/layers_cook.py` to the file map because waterbed, slope-derived understory, and debris dependencies are part of the corrected-base cook. Do not characterize either change as preserving the current API.

### H2. Observation evidence and residual authority remain ambiguous

**Spec:** Section 7, lines 210-287.

The mandatory artifact model has no distinct pre-repair `ObservationEvidence` artifact carrying support, interpolation, water, occlusion, object, seam, suspected-error class, confidence, and provenance. Some of those fields appear optionally in `RawObservation`, others only after correction in `StructuralAuthority`. A `SpecialistOutput` may also be an "explicitly declared residual" without defining the authority digest, interpolation/reconstruction operator, or physical analysis band relative to which the residual is measured.

**Required correction:** add a separate immutable `ObservationEvidence` contract consumed by repair. Define every residual as a delta against a named corrected-authority artifact on a named canonical grid, with an exact analysis/reconstruction operator and digest; otherwise require absolute height. Add rejection/abstention reasons, correction confidence, and protected-feature checks as mandatory typed fields rather than prose diagnostics.

### H3. The condition and acquisition contracts are incomplete

**Spec:** Sections 8, 9.2, 15, and 19.3.

The draft does not require the full multiscale causal terrain state from the checklist: aspect, topographic position, contributing area, flow direction, wetness, and distance/side to structures are absent or only implicit. The source-fetch map says to add orthophoto and EGT modules but does not provide the official endpoint/manifest, schema and code-legend validation, CRS, cache layout, snapshot identity, license/attribution, retry/resume, or schema-drift behavior. Orthophoto testing has only a no-image ablation; it lacks shuffled-image, registration/date perturbation, coverage-boundary, canopy, water, shadow, and no-image fallback contracts. Target truth also omits view/incidence geometry and human-QA masks.

**Required correction:** make every national input executable with the acquisition fields above and fail closed on unavailable data or schema drift. Enumerate the required multiscale derived conditions and their support radii/roles. Add a missing-image availability path that cannot mean observed bare ground, and add all optical falsification tests. Extend target records with sensor/view geometry, human QA, moisture/weather/season, and explicit semantic decisions for stable roots, litter, crops, and deadwood.

### H4. The release matrix is not standalone or sufficient to activate a regime

**Spec:** Section 11, lines 520-551.

The spec delegates the real decomposition to another document and its own ten-field row omits required train/development/blind-test split, per-band signal-to-measurement-error evidence, morphology/visual result, packing entropy, cook/serving cost, and explicit failure/abstention behavior. It also does not define the machine-readable schema, eligibility predicates, transition representation, or initial release status of each listed row. This is incompatible with both standalone implementation and the Definition of Done's every-cell claim.

**Required correction:** embed the complete human-readable phenomenon decomposition or a normatively versioned appendix, and define a concrete machine-readable schema containing all 13 checklist fields. Populate an initial row for every mandatory Estonia regime, normally `unsupported`, and define a coverage audit proving that every in-scope cell resolves to a supported row, a target-evidenced effectively smooth result, or an explicit non-heightfield/forbidden domain. No default or nearest-row branch may exist.

### H5. The primary-source trace does not meet the research mandate

**Spec:** Sections 5 and 23, lines 142-172 and 1152-1188.

The evidence table lacks full titles, authors/year, exact demonstrated input/output scales, degradation/data regime, evaluation, code/data/checkpoint status, license/access status, direct finding, limitation, and permitted project transfer. The short link list does not repair that. The normative `sources.json` currently has 104 records but no per-record authors, SHA-256, or access-result fields; 100 records lack a license field, and official repository revisions live outside the machine record. Thus the spec points to a ledger that does not satisfy its own checklist. "Highest-ceiling" diffusion is also an architecture judgment, not a finding demonstrated by a target-scale Estonia paper, and should be labeled as such.

**Required correction:** add the required architecture evidence table to the standalone spec and normalize the machine ledger so artifact hash, authors, license/access, read scope, official repository revision, and dataset/checkpoint availability are machine-traceable per record. Explicitly flag paywalled, request-only, code-only, invalid-HTML, analogue, and unresolved evidence. Require implementers making consequential method judgments to read the relevant full primary papers and official code, and carry the root `AGENTS.md` rule to use `sol` at high effort only for difficult synthesis/scientific judgment.

### H6. Stable grid and hierarchy contracts are underspecified

**Spec:** Sections 4.1, 14.2-14.4, and 16.5.

The draft omits the world anchor (`E=368640`, `N=6635520`), exact half-cell sample-center oracle, mathematical floor parent division, and explicit west/north/northwest neighbor closure for changed aprons. Quantization records maximum error but does not retain the inline half-qscale round-trip requirement. Seam gates check decoded aprons but not decoded gradient/normal continuity. The pilot's shared fine qoffset is not explicitly bounded as a local-only policy.

**Required correction:** state those exact current contracts and make `height_geom.sample_center_en_units` the coordinate oracle. Enumerate all changed-apron neighbor dependencies through LOD4. Require maximum decode error `<= qscale/2` plus the existing documented float32 allowance, and add decoded value/gradient/normal seam gates. State that the current shared qoffset policy is valid only for the bounded one-parent proof; national qoffset domains require a separately approved deterministic seam policy.

### H7. The visual gate can still pass barely visible or non-geometric detail

**Spec:** Section 16.6, lines 883-902.

The review configuration is deferred rather than specified. The actual reference path, dimensions, and hash are absent. The pairwise panel is optional ("where a panel is used") and has no reviewer count, tie handling, statistical procedure, multiple-comparison rule, or fixed cameras. The hard failures do not explicitly require visible geometric silhouettes, contacts, parallax, moving-light response, and survival through the actual terrain DAG. A candidate can therefore satisfy the prose with weak normal changes while still failing the user's "added realistic detail" requirement.

**Required correction:** bind `reference/suur-taevaskoda1.jpg` (currently `1920 x 1278`, SHA-256 `f031ef585dd82779c8e3c8a8059d15aacce11d2e6c886364bee3eb7c2eac382e`) and freeze normal-FOV cameras/paths, comparison order, lighting/material/vegetation state, reviewer protocol, sample size, decision statistic, and thresholds before candidate inspection. Add hard failures for barely visible detail, shading-only detail, static water roughness, and detail that does not survive the real cluster/DAG, contacts, silhouettes, parallax, grounding, and moving light. Explicit user acceptance remains mandatory and may reject any metric pass.

### H8. Storage and serving accounting is materially incomplete

**Spec:** Section 17, lines 934-966.

The draft omits the exact `8,396,802`-byte raw payload, `136.13 MiB` raw parent-closed publication, `23,708,353` transient-support measurement, exact aligned-pilot and rectangular-country chunk counts, and why the old approximately `8.3 TB` figure is unsafe. It calls the rejected LUKE output a "current one-parent measured release," which can be misread as accepted morphology. It also asks for packed bytes but not p50/p95/p99 entropy or per-square-kilometer compute, support/overlap multiplier, retries/checkpoints, scratch bytes, egress, and recook cadence.

**Required correction:** include the audited formulas and exact values: 16,384 LOD `-2` plus 1,024 LOD `-1` chunks for the `16.384 km` square; 6,190,080 plus 386,880 for the current country rectangle; `47,181,413` published and `23,708,353` transient bytes only as rejected-LUKE diagnostics. Separate ideal land area, rectangular closure, aprons, indexes, scratch, replication, and serving. Require p50/p95/p99 packed entropy and complete train/inference/cook/network cost accounting by regime.

### H9. Stage 1 forces an overclaim and does not gate target qualification

**Spec:** Section 18, lines 987-1009.

Stage 1 passes based only on Track B and a forced Track C winner; Track A can fail without failing the stage. The Moore corpus is 68 disconnected foreign peat plots covering only `309.1387 m2`; it cannot validate whole-mire organization, chunk-scale transitions, or an Estonia production owner. Requiring a "clear winner" incentivizes selection even when all candidates fail or the data support only plot-local morphology.

**Required correction:** require Track A to pass its declared qualification gates before any Track C quality conclusion. Predeclare the exact peat phenomena and spatial scales the Moore data can test, and forbid extrapolation to long-range bog layout or Estonia. Treat "no candidate survives" or "target evidence is insufficient" as a valid decisive result, not a failed research stage. A foreign peat winner remains `research` only and may not satisfy the Estonia activation gate.

## Confirmed Compliance

The exact supplied premise-audit quotation is present, including both missing spaces. The draft also correctly preserves zero browser synthesis, ordinary packed absolute heights, LOD0 at `1 m / 2,048 m`, LOD `-1` at `0.25 m / 512 m`, LOD `-2` at `0.0625 m / 128 m`, corrected-parent authority, fail-closed unsupported regimes, and the single-valued heightfield limit. Those decisions should remain unchanged while the findings above are corrected.
