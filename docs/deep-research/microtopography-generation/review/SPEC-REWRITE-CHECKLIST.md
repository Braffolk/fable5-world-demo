# Adversarial Checklist For Rewriting `SPEC-MICROTOPOGRAPHY.md`

**Date:** 2026-07-13
**Purpose:** requirements trace for a from-scratch replacement of
`docs/tasks/2026-07-13/SPEC-MICROTOPOGRAPHY.md`
**Scope:** specification review only. This document does not authorize synthesis,
source-repair, format, runtime, shader, or asset implementation.

## 1. Rewrite Pass Condition

The rewritten spec passes this checklist only if a zero-context implementer can
answer, without consulting chat history:

- what user-visible terrain result is required and what visibly failed before;
- which parts of the existing packed-height infrastructure are frozen;
- which raw observations may be corrected, from what evidence, and how the
  correction becomes the ordinary packed parent hierarchy;
- which physical regimes and scale bands may synthesize detail, which method owns
  each released regime, and what evidence earned that ownership;
- what training/target data, conditions, confidence, provenance, deterministic
  identity, compute, storage, and serving resources are required;
- what exact cook artifacts and hierarchy relations are produced;
- what objective, falsifiable, and user-visible gates stop another technically
  consistent but visually unacceptable result; and
- which stages remain blocked until the user reviews and approves this spec.

The rewrite fails if it is a literature survey, a menu without a recommendation,
an implementation history, an aspirational architecture without executable gates,
or a repackaging of the rejected LUKE quilt.

## 2. Document Status And Scope

- [ ] Title and status say that this is a replacement specification for user
  review, not an implementation-in-progress report.
- [ ] State explicitly that no replacement synthesis code is authorized until the
  user reviews the completed spec.
- [ ] Make the document standalone. Essential constraints, data semantics,
  physical scales, artifact relationships, stages, and gates must appear in the
  spec itself rather than only as links to research notes.
- [ ] Preserve the accepted cook/packing/serving/streaming/decode/render
  infrastructure. The replacement method changes source correction and cooked
  height contents, not the browser's terrain-generation behavior.
- [ ] Scope the next proof to `asset-gen`, its data/model tooling, corrected base
  cooking, hierarchy derivation, verification, and immutable release assembly.
- [ ] Forbid changes to LAC payload bytes, v2 index records, asset serving,
  runtime sampling, `TerrainField`, shaders, grass, vegetation, materials,
  collision, or probes for the next one-parent proof.
- [ ] Allow the minimum `asset-gen` release/verifier changes needed to publish a
  corrected ordinary format-1 base and pin the existing format-2 negative-rung
  overlay to it. Do not misdescribe this as changing only one synthesis function.
- [ ] State that wider than one LOD `-1` parent later requires coverage planner,
  format-2 coverage metadata/audit, and verifier generalization, but no new chunk
  encoding or runtime sampler.
- [ ] Do not require NumPy-only model research. Deep-learning models are allowed.
  Training/inference dependencies must be fully pinned and costed; their accepted
  output still enters the existing `asset-gen` metric-height encoder.
- [ ] Keep tests focused on stable contracts while the scientific method changes.
  Do not prescribe a broad suite around provisional model internals.

## 3. Decisive Architecture The Spec Must Recommend

The rewrite must recommend one architecture, not choose either `r1.md` or `r2.md`
wholesale:

1. **Typed evidence-fused structural reconstruction** produces a corrected,
   uncertainty-bearing single-valued terrain authority from the fallible DTM and
   independent direct evidence.
2. **A corrected packed hierarchy** publishes accepted correction through ordinary
   LOD0 and every affected ancestor. It does not preserve a known error to obtain
   zero raw-DTM downsample error.
3. **A gated mixture of regime specialists** synthesizes only unresolved surface
   morphology. Contextual exemplar, conditional simulation, process/event,
   regressor, GAN, diffusion, and hybrid candidates compete fairly per regime.
4. **One canonical absolute finest surface** is cropped into storage chunks and
   reduced through browser-equivalent decoded children. Runtime ingests the
   result without knowing whether any height was learned or deterministic.
5. **Fail-closed activation** retains corrected structural terrain without an
   invented residual when target evidence is insufficient. There is no generic
   fallback noise or foreign-regime substitution.

- [ ] Name this architecture decisively, for example "evidence-fused corrected
  authority plus gated regime specialists."
- [ ] Explain that the specialist bakeoff is an implementation rule inside the
  selected architecture, not an evasion of the architecture decision.
- [ ] Do not predeclare a universal learned backbone. Neither proposal's evidence
  supports one.
- [ ] Do not predeclare dictionary/process synthesis as the universal backbone.
  The cited deterministic literature also covers only partial regimes.
- [ ] Preserve conditional pixel-space diffusion as a serious high-ceiling
  challenger wherever real target-scale data support it.
- [ ] Require narrow process/event specialists to compete as first-class methods,
  not merely decorate a learned result, where measured causal evidence supports
  them.
- [ ] Do not use the rejected LUKE quilt, sine fixture, or generic interpolation as
  the strongest non-ML comparator.

## 4. Mandatory Premise Audit

The premise-audit section must embed this exact text, including its punctuation,
capitalization, em dash, and the two missing spaces in the supplied mandate:

> "When a problem resists solving, or a result disappoints, do NOT start varying your approach to the problem — that stays inside the problem. Go up one level first, to the context the problem lives in: thesurrounding system, the upstream decisions, the goal that made this a problem, and everything you've been treating as the fixed environment it sits inside. What you filed under 'given' is the prime suspect,precisely because you filed it under 'given' and never looked at it. Put the background on trial: What generates this problem? Is that context flawed in a way that PRODUCES this failure? Can I feasibly change the context instead of out-thinking a problem that shouldn't have existed? Only if the context is genuinely fixed, or genuinely sound, drop back down and solve the problem where it sits."

The old spec silently normalized `thesurrounding` and `suspect,precisely`, so its
quote is not verbatim and must not be copied.

The section must then apply the instruction rather than merely quote it:

- [ ] Put the literal `<=0.1 m` stored heightfield on trial. The actual objective is
  beautiful, credible, non-repeating, close-up terrain geometry with realistic
  variance and correct material/process character.
- [ ] Record the user's settled context: all synthesis is cook-side and fully
  packed. Runtime procedural or learned synthesis is not an option.
- [ ] Reject exact preservation of every raw 1 m DTM cell as a false given. The DTM
  is preferred ground evidence, not infallible truth.
- [ ] Keep hierarchy, seams, quantization, determinism, and downsample consistency
  as delivery constraints around the accepted corrected surface, not the goal.
- [ ] State that a `0.0625 m` sample pitch does not prove recognizable `6.25 cm`
  objects. The minimum sampled wavelength is about `0.125 m`, and triangulation,
  quantization, simplification, target MTF, and geometry topology raise the useful
  feature-size floor.
- [ ] State that a heightfield cannot represent vertical, undercut, overhanging,
  cave, detached-block, or overlapping surfaces. A denser steep ramp does not
  solve the Suur Taevaskoda wall.
- [ ] Separate the product ambition, credible added detail across Estonia, from
  present evidence readiness. National detail is blocked regime-by-regime until
  the required Estonia target evidence exists; it must not be faked to satisfy a
  nominal nationwide checkbox.

## 5. Frozen Repository Contracts To Reverify And State

The rewrite must inspect the current branch immediately before finalizing these
facts. It may cite current paths, but must not turn line numbers or a dirty-worktree
snapshot into timeless architecture.

| Contract | Current value to verify | Primary repository evidence |
|---|---|---|
| World anchor | `E=368640`, `N=6635520`; game `X=E-anchorE`, `Z=anchorN-N` | `asset-gen/config/base.toml`, `asset-gen/src/assetgen/grid.py` |
| Shared base grid | `chunk_m=2048`, `chunk_res=2048`, `lod_step=4`, LODs `0..4` | same files |
| Payload | `2049 x 2049`, including east/south apron, at every height rung | `grid.py`, `height_geom.py` |
| Fine height rungs | LOD `-2`: `0.0625 m`, `128 m`; LOD `-1`: `0.25 m`, `512 m` | `asset-gen/src/assetgen/height_geom.py` |
| Ordinary base | LOD0: `1 m`, `2,048 m`; do not reinterpret it as `128 m` | `base.toml`, `height_geom.py` |
| Parent ratio | one parent has `4 x 4` immediate children; use mathematical floor division | `height_geom.py` |
| Height coordinates | EPSG:3301 horizontal, EH2000 absolute height; exact half-cell sample centers | `grid.py`, `height_geom.py` |
| Height encoding | u16 quant16, 2D delta, deflate; checked no-clipping encode and float32 browser-equivalent decode | `asset-gen/src/assetgen/cook/encode.py`, `chunkio.py` |
| Current fine qscales | LOD `-2`: `0.002 m`; LOD `-1`: `0.005 m`; ordinary LOD0-2: `0.01 m` | `microtopography.toml`, `base.toml`, `release.py` |
| Current pilot coverage | one LOD `-1` parent, 16 published LOD `-2` children, 9 transient east/south/corner support chunks | `height_geom.py`, `release.py`, `micro_verify.py` |
| Existing parent primitive | quantize child, browser-equivalent decode, fixed-order `4 x 4` reduce, quantize parent | `asset-gen/src/assetgen/cook/micro_hierarchy.py` |
| Existing runtime contract | physical negative LODs are representation-based, not generator-based | `RemoteWorldSource.ts`, `PlaneFill.ts`, `TerrainField.ts`, `TerrainMorph.ts` |

- [ ] Use `height_geom.sample_center_en_units` and the tested coordinate contract as
  the oracle. Do not preserve an old prose formula if it differs from current code.
- [ ] Preserve complete parent closure and apron dependency closure. Fine coverage
  may not remove the LOD0/coarse terrain mesh parents needed to render it.
- [ ] State that the currently accepted runtime availability morph is part of the
  frozen consumer path. New correction must make its packed parents agree; the
  cook may not rely on the morph to hide unrelated geometry.
- [ ] Record the accepted infrastructure baseline by current commit only after
  rechecking Git. Do not repeat the old statement that the branch is at `258ddef`,
  has no unpublished commits, or has a particular number of dirty files.
- [ ] Do not preserve the old hard-coded historical base manifest as eternal
  authority. The replacement recipe must bind the newly corrected base manifest.
- [ ] Include an `asset-gen`-only file-boundary map that begins from the actual
  current code: `cook/micro_synth_cook.py`, `cook/pinned_height.py`,
  `cook/micro_hierarchy.py`, `process/microtopo/`, `process/soil.py`,
  `process/water.py`, `cook/layers_cook.py`, `release.py`, and `micro_verify.py`.
  Do not list speculative runtime files as replacement-synthesis work.
- [ ] Explicitly replace the current residual-only `synthesize_residual` and raw
  mean-null authority semantics. Reuse checked encoding, decoded-child reduction,
  release inheritance, and packed transport rather than rewriting them.

## 6. Required Artifact And Authority Model

The spec must define these distinct artifacts and never overload "base," "truth,"
or "residual":

| Artifact | Meaning | Required authority behavior |
|---|---|---|
| Raw observation | Maa- ja Ruumiamet 1 m DTM plus acquisition/source identity | retained unchanged in provenance; not necessarily rendered authority |
| Observation evidence | confidence, support, interpolation, water, occlusion, object, seam, and suspected-error state | determines where observation may be corrected or must be protected |
| Corrected structural authority | evidence-fused single-valued terrain, including reconstructed mapped structures | published through ordinary corrected LOD0 and ancestors |
| Unresolved morphology | plausible detail below reliable observation support, conditioned by a validated physical regime | synthesized only by an approved specialist; may abstain |
| Canonical finest surface | corrected structure plus accepted unresolved morphology as absolute EH2000 height | sole source of published fine chunks and immediate parents |
| Packed hierarchy | quantized/decode-derived immutable runtime artifacts | delivery authority for runtime sampling and transitions |

- [ ] Require every correction to carry class, evidence sources, confidence or
  uncertainty, raw delta, protected-feature checks, and abstention/rejection reason.
- [ ] Require every generated residual to carry regime/subtype, model/process and
  checkpoint/version identity, conditions, support domain, production seed, and
  release status.
- [ ] Forbid a downstream raw-DTM mean-null projection from erasing a justified
  correction.
- [ ] If an internal residual pyramid is used, define it around the corrected
  parent. `fine - interpolate(coarse)` is not automatically a band.

## 7. Typed Source Repair And Corrected Hierarchy

### 7.1 Source-repair responsibilities

- [ ] Treat water interpolation/TIN bridges, grid-stair shorelines, vegetation or
  building leakage, isolated spikes/pits, scan seams, missing returns, object
  contamination, bridged channels, and mapped structural misplacement as distinct
  repair classes.
- [ ] Resolve direct ETAK water, shoreline, river centerline, natural escarpment,
  ditch, road/track, quarry, landform, building, and mapped-boulder constraints
  before unresolved generative synthesis.
- [ ] Use deterministic typed reconstruction for directly evidenced structure.
  Learned assistance may detect or fill ambiguous evidence only with calibrated
  selective risk and an abstention path.
- [ ] Measure false correction of real cliffs, banks, boulders, channels, closed
  bog/glacial/karst forms, and other rare landforms separately from source-error
  removal.
- [ ] Reapply forbidden-domain and protected-structure constraints after every
  learned or stochastic stage so detail cannot recreate a removed water bump or
  move a mapped structure silently.

### 7.2 Water and shore semantics

The spec must define separate authorities and outputs for:

- visible water surface `waterY`;
- anti-aliased water/shore occupancy `watercover`;
- submerged terrain bed stored in height;
- shoreline intersection;
- terrestrial bank and bank shoulder; and
- natural escarpment or cliff warning.

- [ ] Open-water optical surface receives no terrain roughness. Reflections, waves,
  glare, ice, emergent plants, and missing returns never become static height.
- [ ] Unknown bathymetry is conservative and uncertainty-bearing, not inferred from
  orthophoto or arbitrary water gaps.
- [ ] Shore correction removes the visible 1 m staircase without smoothing an
  adjacent real escarpment into a generic ramp.
- [ ] Water surface, watercover, shoreline, bank, bed, and every changed height
  ancestor use one evidence snapshot and compatible cook order.
- [ ] If corrected LOD0 slope changes influence understory/debris suitability,
  recook those dependent assets or prove the old classification is stable. Do not
  create a runtime exception.

### 7.3 Hierarchy publication

- [ ] Generate or reconstruct a canonical `0.25 m` structural support surface over
  every corrected LOD0 payload and its apron dependency closure.
- [ ] Generate one canonical absolute `0.0625 m` finest surface over the published
  fine region plus model and hierarchy support.
- [ ] Crop final LOD `-2` storage chunks from the reconciled master. Never make a
  chunk boundary a generation boundary.
- [ ] Quantize/decode LOD `-2`; derive/quantize/decode LOD `-1` from the decoded
  children; derive/quantize/decode corrected LOD0 from canonical decoded LOD `-1`.
- [ ] Derive every affected LOD1-L4 ancestor and every west/north/northwest neighbor
  whose apron depends on changed child content.
- [ ] Publish a corrected ordinary format-1 base first, inheriting unchanged
  objects, then publish the otherwise unchanged format-2 negative-rung overlay
  pinned to that corrected base hash.
- [ ] Make both publications content-addressed, recipe-bound, resumable only under
  exact dependency identity, completeness-checked, and atomic. Neither cooking nor
  verification may update `latest`; promotion requires the final user checkpoint.
- [ ] Distinguish model-support extent, hierarchy-support extent, correction/base
  closure, and published coverage.
- [ ] Require the corrected authority beneath coverage fade and normal/sampling
  stencils. Storage boundaries may not determine correction taper or physical
  regime transitions.
- [ ] Retain checked no-clipping quantization, half-qscale round-trip evidence,
  browser-equivalent decode, and decoded value/gradient/normal seam gates.
- [ ] Treat the pilot's one shared fine qoffset as a bounded local policy only. A
  national policy must define deterministic qoffset domains and prove decoded seams
  without exceeding the approximately `131 m` span of `0.002 m` u16 quantization.

## 8. Full-Fidelity Estonia Conditioning Contract

Every condition requires physical support scale, source scale, date/vintage,
coverage, uncertainty, availability mask, permitted causal role, and provenance.
The spec must say how each is consumed or explicitly rejected.

- [ ] Maa- ja Ruumiamet 1 m DTM remains the preferred ground measurement. DSM does
  not replace it.
- [ ] Use nDSM and CHM only for above-ground context, occlusion, object leakage, and
  observation confidence.
- [ ] Preserve ETAK vector structures and their types rather than reducing them to
  broad exclusion masks.
- [ ] Preserve Mullastikukaart `Sif1..Sif4`, `Osa1..Osa4`, `Loimis1`, `Loimis2`,
  `Lihtloimis`, `Huumus`, `Kivisus`, and `Boniteet`, including unknowns and source
  typography. Do not reduce physical conditioning to `Sif1` or one hard class.
- [ ] Represent soil mixtures, vertical texture/humus profile, coarse fragments,
  stoniness, drainage/ecohydrology, and coverage confidence compositionally.
- [ ] Consume EGT 1:50,000 bedrock, surficial geology, and geomorphology only under
  an explicit coverage mask. Missing coverage is not a class.
- [ ] Use EGT 1:200,000 bedrock/surficial geology as a broad low-authority national
  family prior. Decode official codes before use; never guess them or emboss its
  polygon edges.
- [ ] Include forest stand composition/age/disturbance, agriculture operation/date,
  drainage, land use, management, and disturbance age where they cause morphology.
- [ ] Include multiscale slope, aspect, curvature, topographic position,
  contributing area, flow direction, wetness, distance and side relative to mapped
  structures, and larger physical-domain context.
- [ ] Mediate coarse map transitions with finer evidence and uncertainty. A
  1:10,000, 1:50,000, or 1:200,000 polygon boundary may never print as a fine
  height seam.
- [ ] Require remove/shuffle, plausible counterfactual, registration perturbation,
  date mismatch, and coverage-boundary ablations. A condition passes only if it
  improves held-out morphology or structural placement without shortcut artifacts.
- [ ] Give every new national source an executable `asset-gen` acquisition contract:
  official endpoint/download manifest, snapshot date, exact schema and code legend,
  CRS, source scale, coverage, hashes, cache layout, license/attribution, retry and
  resume behavior, and fail-closed schema drift. Note that soil is currently
  acquired out of band and orthophoto/EGT acquisition is not yet integrated.

Taevaskoda facts to include as a fixed evidence example, not a scene-specific rule:

- GPS `58.107506 N, 27.050242 E`;
- EPSG:3301 `E 679763.082, N 6444796.565`;
- game `x=311123.082, z=190723.435`;
- nearest natural shoreline escarpment about `5.32 m`;
- Ahja water polygon boundary and explicit shoreline about `1.85 m`;
- river centerline about `15.69 m`;
- soil polygon mixture `LkI` 70 percent and `L(k)I` 30 percent, with layered
  texture and humus fields rather than one class;
- no local EGT 1:50,000 polygon, nearest boundary about `7.05 km`;
- 1:200,000 bedrock fallback `Burtnieki Formation`, a broad
  sandstone/siltstone/clay-interbed family prior only; and
- the diagnosed Ahja DTM/TIN bridge is a source-repair target, not evidence for
  microdetail amplitude.

## 9. Orthophoto Decision

The spec must answer **yes** to fetching and versioning dated RGB/CIR orthophoto
for the research corpus and likely production conditioning, subject to ablation.
It must not answer "optional unless convenient."

- [ ] Orthophoto may locate visible shore, path, drainage, erosion, exposed
  material, crop direction, vehicle track, disturbance, and current land-cover
  evidence at its measured support.
- [ ] Orthophoto is never metric height truth and never direct supervision for the
  `0.0625-0.25 m` band merely because the output grid is finer.
- [ ] Carry image product, sheet, acquisition date, GSD, registration, canopy,
  object, water, shadow, snow/season, seam/cloud, and temporal-mismatch masks.
- [ ] Provide a no-image/low-confidence fallback that does not silently treat zero
  image features as observed bare ground.
- [ ] Require no-image, shuffled-image, registration/date perturbation, canopy,
  water, shadow, and albedo-embossing falsification tests.
- [ ] State current official scale: generally `20-40 cm` national RGB/CIR, with
  `10-16 cm` in dense settlements. Do not repeat `10-20 cm` as the national norm.
- [ ] Bind the official mass-download workbook and per-sheet flight dates. The
  current audit found 2,111 distinct RGB sheets and a provisional newest-RGB cache
  of about `367 GB` compressed before extraction, masks, CIR/NGR, or history.
- [ ] For Taevaskoda sheet `54472`, record the current audited RGB/CIR/NGR bundle at
  about `489.4 MB` compressed, while requiring a fresh manifest check at execution.
- [ ] Require a full manifest-size scan and measured crop/cache policy before a
  national bulk download.

## 10. Target Data, Training, And Release Evidence

### 10.1 State the present evidence honestly

- [ ] State that there are currently zero production-paired Estonia target sites
  and zero audited open target-scale Estonia height surfaces.
- [ ] State that only one of ten audited non-water regime families has any ready
  high-resolution exemplar: foreign peat.
- [ ] State the Moore archive's actual scale: `309.1387 m2` across 68 disconnected
  plots, about `1.89%` of one `128 m x 128 m` LOD `-2` chunk, all one broad material
  family and none in Estonia.
- [ ] Treat LUKE Lapinjarvi and other unclassified forest TLS as raw conversion or
  foreign-analogue infrastructure data, not production ground truth.
- [ ] Treat model-generated, upsampled, interpolated, DSM-contaminated, and
  orthophoto-derived pseudo-height as non-truth.

### 10.2 Define a production target

Every target site must include:

- real post-filter ground support at nominal `<=0.03-0.05 m`, finer than the output
  cell and sufficient for the claimed physical band;
- independently measured horizontal and vertical error distributions relative to
  that regime's relief energy;
- per-cell point support, view/incidence geometry, interpolation, no-data, and
  human-QA masks;
- explicit surface semantics for mineral ground, peat/moss, stable roots, clasts,
  litter, deadwood, crops, water, and objects;
- raw sensor data and immutable processing provenance;
- temporally registered DTM, orthophoto, ETAK, soil, geology, hydrology, land use,
  management, vegetation, moisture, weather, and season records;
- site/campaign identity for whole-site splits; and
- licenses covering data use, model training, checkpoint/model distribution,
  attribution, and redistribution.

- [ ] Require at least three geographically separate Estonian acquisition sites
  before one physical regime emits finest-band production geometry: two
  development sites and one untouched site-level holdout.
- [ ] Call this a minimum go/no-go floor, not proof of sufficiency. Continue adding
  sites until held-out morphology and blind visual results stabilize.
- [ ] Keep Taevaskoda outside training and hyperparameter selection if it remains a
  visual generalization landmark.
- [ ] Use nested spatial acquisition: large contextual surveys for organization,
  drainage, and transitions plus verified target-scale microplots/returns for
  finest-band supervision.
- [ ] Match sensor to regime: controlled SfM for visible bare surfaces; TLS/MLS for
  forest; repeat-state acquisition for agriculture; LiDAR plus controls for wetland
  and shore; oblique 3D capture for cliffs even though invalid heightfield portions
  remain outside this spec's output claims.
- [ ] Allow permissively licensed foreign data for representation pretraining or
  regime-specific initialization only. It cannot be the sole validation for an
  Estonia release regime.
- [ ] Fit synthetic degradation to real paired Maa-amet/target observations by
  vintage, canopy, slope, water, classification, interpolation, and registration.
  Bicubic/average/ideal low-pass degradation remains a baseline only.
- [ ] Split train/validation/test by whole site, geography, and campaign before
  model selection. Random overlapping patch splits are forbidden.

## 11. Required Regime And Phenomenon Coverage

The spec must include a machine-readable release-matrix schema and a human-readable
decomposition at least as complete as `REGIME-PHENOMENON-MATRIX.md`. Broad labels
such as `forest`, `field`, `bog`, `sand`, `shore`, `rock`, or `barren` are not
morphology models.

At minimum, cover or explicitly mark unsupported:

- raised bog; aapa/fen/transitional mire; drained/cut peatland;
- forest pit-and-mound; ordinary forest floor; managed forest/clear-cut;
- ploughed field; seedbed/harrowed/rolled field; pasture/meadow; yard/turf;
- exposed sandy soil; dune/aeolian sand; beach/shoreface; shingle/gravel/cobble;
- till/glacial sediment plain; gravel/esker/outwash exposure;
- carbonate alvar/thin soil; carbonate pavement/karst;
- sandstone outcrop top/slope; limestone/dolostone outcrop top;
- colluvial slope/talus;
- fluvial floodplain; active channel bar/exposed bed; rill/gully/seep/spring;
- saline/coastal wetland; and
- technogenic/disturbed ground, roads/tracks, quarry/spoil, construction, and other
  engineered states.

For each row require:

1. subtype and constituent phenomena, not only a name;
2. physical scale breaks and representable bands;
3. eligibility predicates, mixtures, uncertainty, and transition domain;
4. direct structural constraints and forbidden residual regions;
5. causal conditions and observation-confidence state;
6. target datasets, sites, effective resolution, error, visibility, license, and
   immutable identity;
7. candidate families and frozen fair-comparison configurations;
8. train/validation/blind-test geographic split;
9. signal-to-measurement-error evidence by physical band;
10. morphology and user-visible bakeoff result;
11. packing entropy, cook compute, and serving cost;
12. failure, abstention, and representation-limit behavior; and
13. release state `unsupported`, `research`, `pilot`, or `national`.

- [ ] Require interactions across substrate, hydrology, relief, biological state,
  management, disturbance age, observation state, and representation class.
- [ ] Require event specialists to model measured joint distributions over shape,
  size, density, orientation, spatial interaction, age/decay, and conditions. One
  ideal primitive is not a production event distribution.
- [ ] Forbid a global roughness/diversity/amplitude knob as the feature. Realistic
  within-site and between-site variance, tails, anisotropy, scale breaks, event
  state, and transitions must come from measured conditional distributions.
- [ ] Forbid implicit fallback from an unsupported row to generic mineral, forest,
  or foreign-analogue detail.
- [ ] Require ordinary terrain and transitions in the release set, not only
  spectacular landmarks.

## 12. Fair Specialist Selection And Literature Boundaries

### 12.1 Required bakeoff

- [ ] Compare contextual exemplar/dictionary, conditional geostatistical
  simulation, calibrated process/event, deterministic learned regressor, GAN,
  pixel-space diffusion, regime experts, and physically constrained hybrids where
  relevant to a regime.
- [ ] Use the same target sites, conditions, physical support, split, output
  contract, reasonable parameter/compute allowance, packing gate, and render
  protocol.
- [ ] Compare direct/multiscale and cascaded model factorizations. The `1 m ->
  0.25 m -> 0.0625 m` storage layout does not prove two independent x4 learned
  stages are scientifically optimal.
- [ ] Select an owner only after it wins preregistered per-regime visible and
  measured gates without unacceptable source repair, mode coverage, partition
  invariance, entropy, or cost.
- [ ] Permit shared learned encoders/backbones only where held-out evidence shows
  sharing improves rather than averages away regime morphology.

### 12.2 Claims that must be corrected or bounded

- [ ] DDNM supplies range/null consistency for an explicit linear operator and a
  separate prior. It does not provide a Maa-amet observation model, terrain prior,
  source-error correction, or beauty.
- [ ] MultiDiffusion supplies overlapping pixel-state consensus. It does not prove
  terrain gradient, normal, curvature, drainage, topology, or physical continuity.
- [ ] ControlNet supplies conditioning machinery, not terrain competence or a
  metric-height prior.
- [ ] Guérin 2016 needs similar paired low/high exemplars and does not guarantee
  geomorphological or small-scale coherence. It is a candidate baseline, not a
  demonstrated Estonia backbone.
- [ ] Guérin 2017 cGAN authors macro terrain and delegates final amplification to
  the 2016 sparse method. Do not cite it as learned centimeter amplification.
- [ ] Argudo 2017 is the multi-layer dictionary paper; Argudo 2018 is the
  orthophoto FCN. The latter demonstrated `15 m + 1 m imagery -> 2 m` on selected
  alpine data and reported vegetation/shadow failures.
- [ ] GATA is an old x4 256-pixel research GAN with no released training corpus or
  checkpoint. Its theme mechanism transfers; its weights/evidence do not.
- [ ] Terrain Diffusion/InfiniteDiffusion, StyleDEM, Lochner, TerraFusion, GSRMTL,
  ET-SDE, the lunar bridge, and related learned systems operate at meters to tens
  or hundreds of meters, commonly under synthetic degradation and meter-scale
  errors. They transfer mechanisms, not target-scale priors.
- [ ] GrounDiff is same-grid DSM-to-DTM correction and reports cliff/dense-canopy
  failures. It is source-repair evidence, not unresolved microtopography synthesis.
- [ ] HuHoLa and Ilyasov classify/diagnose existing mire relief. They are not
  generators.
- [ ] Scott/Dodgson pit removal is macro hydrology handling and can destroy valid
  bog, glacial, and karst depressions if generalized.
- [ ] Schott multiscale erosion is a strong specialist for eligible hydraulic
  terrain but has whole-domain dependencies and does not cover non-hydraulic
  Estonia.
- [ ] Grenier/Phasor patterns remain structured procedural noise and reproduce the
  observed oriented-pattern failure if used as general ground detail.
- [ ] Paris/Cordonnier demonstrate geology and true-3D/process importance, but do
  not establish a dense national 6.25 cm heightfield solution.
- [ ] PTRM and macro terrain descriptors do not replace close-up ground-level
  acceptance. Scale-specific diagnostics remain supporting evidence only.
- [ ] Agricultural, peat, forest pit-mound, and rock measurement papers support
  target acquisition and regime-specific distributions. Small foreign plots do
  not provide national priors or Estonia constants.
- [ ] Flag every claim based on inaccessible, request-only, paywalled, secondary,
  analogue, or code-only evidence. Do not convert an unresolved number such as a
  grike width or hummock parameter into a production constant.

### 12.3 Source and claim trace

- [ ] Include an architecture-relevant evidence table in the standalone spec with
  full title, authors/year, DOI or official URL, demonstrated input/output scales,
  data/degradation regime, evaluation, released code/data, direct finding,
  limitation, and permitted project transfer.
- [ ] Point to a machine-readable source ledger that records canonical URL, local
  artifact, SHA-256, license, access result, full-paper read scope, official
  repository revision, dataset/checkpoint availability, and unresolved claims.
- [ ] Cite primary papers, official repositories, and official data records rather
  than `r1.md` or `r2.md` as evidence. Those reports are hypotheses under audit,
  not sources.
- [ ] Preserve citation corrections and quarantined invalid artifacts. A file named
  `.pdf` that contains HTML, a community reimplementation, or an unlicensed
  request-only dataset may not be presented as a validated production dependency.
- [ ] Record why each major rejected alternative fails the quality ceiling or
  evidence boundary, not merely why it is inconvenient to implement.
- [ ] Require any implementer making a consequential scientific/visual method
  judgment to read the relevant full primary sources and official code first and,
  when model selection is available, use `sol` with high effort only for the
  genuinely difficult synthesis/judgment work as required by root `AGENTS.md`.

## 13. Analysis/Synthesis, Domains, And Deterministic Identity

- [ ] Define the exact world-aligned analysis filter, antialiasing, decimation
  phase, synthesis/reconstruction operator, edge/apron handling, and relation to
  corrected parents for every claimed band.
- [ ] Generate coherent physical forms first and then analyze them into packed
  supports. Do not paint independent semantic detail into LOD `-1` and `-2`.
- [ ] Make the accepted absolute finest surface canonical; derive stored parents
  through the existing browser-equivalent decoded-child hierarchy.
- [ ] Define canonical world stochastic identity using a versioned counter-based
  field keyed by absolute world coordinates, regime/expert, model/process version,
  and production seed.
- [ ] Define a fixed support-window lattice, full influencing-window set, exact
  fusion quantity and scheduler for learned sampling, and crop/apron rules.
- [ ] Require output and packed hashes on the locked inference stack to be invariant
  to storage chunking, support-window origin variants, overlap stride variants,
  worker count, job order, batch size, retry/resume, adjacent requests, and a larger
  enclosing AOI.
- [ ] Do not promise bitwise cross-hardware identity unless demonstrated. Released
  packed hashes are runtime authority.
- [ ] Process long-range phenomena over their actual domain, such as connected
  mire, watershed, managed parcel, corridor, stand/event field, or coast reach,
  then crop storage chunks. A fixed local halo is not proof of causal sufficiency.
- [ ] Forbid periodic boundaries and per-chunk random seeds, phases, classes, or
  orientations.
- [ ] Bind every output to source-window hashes, condition dates/versions,
  corrected-authority digest, target/training manifest, model/checkpoint or process
  version, inference environment, stochastic recipe, quantization, and full
  influencing-domain digest.

## 14. Resource, Storage, And Serving Honesty

The spec must include exact formulas, measured pilot facts, and separate estimates
for ideal land area versus the current chunk-snapped rectangular release. It must
not hide accepted zero-runtime-synthesis cost behind "offline."

- [ ] State one raw u16 `2049^2` payload as `8,396,802` bytes before header/codec.
- [ ] State one parent-closed fine publication as 16 LOD `-2` chunks plus one LOD
  `-1` chunk: about `136.13 MiB` raw before transient support.
- [ ] State the rejected LUKE field's measured one-parent published size only as a
  diagnostic compression datum: `47,181,413` bytes, plus `23,708,353` transient
  support bytes. Do not forecast a richer model from it.
- [ ] State the current aligned `16.384 km x 16.384 km` pilot order of magnitude:
  16,384 LOD `-2` plus 1,024 LOD `-1` chunks, about `146.2 GB` raw and about
  `48.3 GB` at the rejected pilot's observed ratio.
- [ ] State the current full-country rectangular AOI audit: 6,190,080 LOD `-2` plus
  386,880 LOD `-1` chunks, about `55.23 TB` raw and about `18.25 TB` only if the
  rejected pilot's compression ratio transferred.
- [ ] Also distinguish the idealized `45,000 km2` land-area lower-order estimate:
  about `23.0 TB` raw LOD `-2` plus `1.44 TB` raw LOD `-1`, before chunk aprons,
  rectangular closure, indexes, replication, and serving.
- [ ] Explain why the old spec's roughly `8.3 TB` country estimate is stale/unsafe.
- [ ] Benchmark accepted-quality outputs by regime and qscale through inference,
  correction, quantization, 2D delta, deflate, verification, and serving. Report
  p50/p95/p99 packed entropy rather than one smooth pilot average.
- [ ] Report model training GPU-hours, inference GPU/CPU-hours per square kilometer,
  host/GPU peak memory, support/overlap multiplier, retries/checkpoints, total
  chunk/object count, scratch/transient bytes, output storage, upload, replication,
  CDN/cache, egress, and recook cadence.
- [ ] Require explicit user acceptance of coverage and cost before national
  production. Expense may be accepted for quality, but may not be omitted.

## 15. Acceptance That Prevents Another Visually Bad Success

### 15.1 Predeclare the review

- [ ] Freeze site-separated target sets, ordinary and spectacular sites, camera
  poses/paths, normal field of view, light/weather/material/vegetation state,
  reviewer blinding, comparison order, failure labels, and decision thresholds
  before inspecting final candidates.
- [ ] Use the real reference at the repository's verified path. The user's prose
  called it `reference/suur-taevaskoda-1.jpg`, while the current repository file and
  review config use `reference/suur-taevaskoda1.jpg`; resolve and hash the actual
  file instead of silently citing a nonexistent path.
- [ ] Keep Taevaskoda a generalization site. No coordinate-, image-, chunk-, or
  camera-specific synthesis behavior is allowed.
- [ ] Grade the single-valued upper ground, forest floor, trail, bank, smooth shore,
  and representable outcrop top. Do not grade the vertical/undercut cathedral-like
  wall as a heightfield synthesis target.

### 15.2 User-visible hard failures

Any one of these rejects a release regardless of internal metrics:

- accepted detail does not visibly affect actual geometric silhouettes, contacts,
  normals, near-ground parallax, grounding, and moving-light response at normal
  field of view, or exists only as material shading;
- detail is barely visible, generic noise, random bumpiness, sine waves, FBM,
  repeating stamps, quilt texture, checkerboards, or support/chunk patterns;
- a 1 m grid remains visible in a shoreline, slope, gradient field, raster-axis
  bias, or moving-light response;
- water contains static roughness or a known DTM bump survives/reappears;
- a real bank, cliff, boulder, channel, depression, ditch, or other protected form
  is flattened or moved without accepted evidence;
- physical character is wrong for substrate, soil profile, hydrology, exposure,
  land management, disturbance, or age;
- ordinary terrain receives gratuitous uniform roughness or rare dramatic forms are
  overexpressed;
- a condition polygon, image/albedo edge, acquisition seam, scanner signature,
  interpolation pattern, or training patch identity appears as relief;
- a storage, model-window, physical-domain, correction-confidence, regime, or LOD
  transition is visible;
- terrain, grass, trees, plants, collision/probes, normals, or materials disagree,
  float, form elevated islands, disappear, or sample a different surface;
- trees, materials, understory, or grass are disabled to make the terrain pass;
- the actual terrain DAG does not retain visibly intended fine features; or
- the result is not clearly preferred to the corrected base at ground level.

### 15.3 Supporting measured gates

- [ ] Separate source-repair precision/recall, correction calibration, and protected
  feature destruction from unresolved-detail quality.
- [ ] Measure one-meter power/harmonics, grid phase, raster-axis gradient bias,
  shoreline stair/curvature, and moving-light visibility.
- [ ] By regime and site, report height, gradient vector, normal, curvature, local
  relief, joint distributions, spatial correlations, spectra, variograms,
  anisotropy, typed object shape/spacing/orientation, hydrologic connectivity,
  tails, precision/coverage, and transition strips.
- [ ] Test repetition, nearest-training-patch similarity, memorization, mode
  coverage, multi-seed research diversity, condition use, and foreign/scanner motif
  transfer.
- [ ] Require per-regime non-inferiority and owner-selection rules. A national
  average may not waive one failed physical regime.
- [ ] Treat metrics as diagnostic. Matching PSD, variogram, descriptor, RMSE, or
  exact hierarchy alone never passes beauty.

### 15.4 Stable cook and runtime gates

- [ ] Independently rederive canonical `-2 -> -1 -> corrected 0 -> 1..4` hierarchy,
  aprons, dependency closure, hashes, finite/range/quantization bounds, and water
  compatibility.
- [ ] Verify deterministic crop/order/resume behavior on the pinned stack.
- [ ] After a candidate release is materialized, boot the exact Estonia URL in real
  WebGPU Chromium through readiness, cloud bake, and settled frames. Static tests,
  a successful cook, and `window.__laas.ready` are insufficient.
- [ ] Use `scene=world&src=estonia`, never `scene=estonia`, include the actual
  `dataurl`, and center review at `x=311123.082&z=190723.435`.
- [ ] Capture page errors, console errors, TSL diagnostics, and uncaptured WebGPU
  errors. Any invalid shader/pipeline/bind group/command buffer, TSL invalid code,
  uniform overflow, alias/access error, or barrier-uniformity failure rejects boot.
- [ ] Visually confirm terrain continuity, material presence, tree/plant/understory
  presence, grass grounding, and motion through negative-rung availability edges.
- [ ] Require explicit user visual approval after the agent has completed the clean
  exact-URL boot. The user is not the WebGPU compiler or first runtime tester.

## 16. Cheapest Proof And Staged Authorization

The rewritten spec must choose and justify one concrete cheapest proof. It should
contain these three separately judged components rather than immediately train or
cook a national model:

1. **Target/evidence gate:** build the versioned registry and paired condition
   stacks for a small number of genuinely different regimes; quantify target-band
   signal against acquisition error and establish untouched site holdouts.
2. **Typed Taevaskoda source-repair gate:** reconstruct water, bed, shoreline,
   bank, and escarpment evidence; remove the diagnosed river artifact; publish the
   corrected ordinary LOD0 and ancestor closure; prove the old 1 m shore/error does
   not return through the runtime transition.
3. **One-regime morphology bakeoff:** select one regime with defensible target-scale
   evidence and compare fair exemplar, simulation/process, regressor, GAN,
   diffusion, and hybrid candidates through the frozen packed-height output.

- [ ] Do not name a regime for the morphology proof merely because Taevaskoda is
  visually important. Choose it from actual target sufficiency, then state the
  choice decisively and explain its transfer limits.
- [ ] Require blind ground-level output, morphology coverage, partition invariance,
  packed entropy, and compute cost for the one-regime decision.
- [ ] Require Stage A/source correction to remove the 1 m lattice and known source
  defect before funding the finest unresolved band at that site.
- [ ] Do not implement a provisional national event catalog, broad national bake,
  or large brittle test suite during the proof.
- [ ] At the end of each major runtime-facing release task, require the exact real
  WebGPU boot. Provide a live URL only after it passes.
- [ ] Require a fresh user checkpoint after the rewritten spec and before any code;
  require another explicit checkpoint before a national data acquisition/storage
  commitment and before publishing `latest`.

## 17. Old-Spec Material That Must Not Survive

| Old material | Required disposition |
|---|---|
| Status says Stage 1/diagnostic Stage 2 implementation is in progress | Replace with specification-for-review status |
| Section 4 branch/worktree forensics at `258ddef`, dirty-file counts, old experimental manifests, failed tests | Remove. Preserve only enduring failure lessons and current verified normative contracts |
| Recommendation: deterministic exemplar plus structural-process hybrid | Remove. It was visually rejected and contradicted by the completed primary-source review |
| `do not adopt PyTorch/diffusion for version 1` | Remove. ML is allowed and diffusion is a serious per-regime challenger |
| Exact raw LOD0 mean preservation and AOI-wide mean-null projection | Remove as authority policy. Projection may target only a corrected parent and must not erase repair |
| LUKE patch bank, quilt placement, top-k, equal-power overlap, fixed event functions, and related thresholds as production synthesis | Remove. Retain LUKE only as rejected/foreign infrastructure evidence and, if useful, a failure control |
| Conservative spline, local projection, quilt, and matrix selector prescribed as the replacement generator | Remove unless independently re-earned as a bakeoff component |
| `zero_detail` caused by lack of current bank while still claiming national detail | Replace with explicit unsupported release state and acquisition blocker |
| Orthophoto optional until an enabled family asks for it | Replace with a positive research/likely-production acquisition decision plus strict evidence boundaries and ablations |
| Three sites per broad family, including foreign analogues, as production readiness | Replace with at least three geographically separate Estonia sites per released regime and continued acquisition until stability |
| Old national `~8.3 TB` fine estimate | Replace with the audited ideal-land and current rectangular-AOI calculations |
| Stale `reference/suur-taevaskoda-1.jpg`/`suur-taevaskoda1.jpg` assumptions | Resolve the actual file, path, dimensions, and hash before freezing the review config |
| Proposed LAC2/index/runtime implementation, normalized finest lattice, terrain morph state machine, memory windows, DAG retention plumbing | Remove from the synthesis rewrite. These are accepted infrastructure, not new work |
| Tree six-column schema, per-record Y redesign, grass hot-path changes, `TerrainField` micro methods | Remove. No runtime/provenance branch is authorized |
| Structural-terrain `enc=4`/`LMP1`, `StructuralBand`, mesh-page pool, collision/BVH, material and vegetation ownership implementation | Remove entirely from this synthesis-only spec. Keep only the heightfield representation limit and a pointer to a future separate spec |
| Stage 0 creates a clean worktree from `258ddef`; Stage 1 reimplements formats/runtime | Remove as completed or obsolete history |
| Stage 3b structural-outcrop implementation | Remove. Non-heightfield work is not part of the replacement height-synthesis proof |
| Large fixed automated-test inventory around provisional internals | Replace with lean stable-contract, reproduced-regression, and exact runtime gates |
| Old "definition of done" includes structural-cluster shipping | Replace with a heightfield-only definition of done and explicit unsupported cliff-face scope |

The rewrite may briefly record why the LUKE output failed: weak/noise-like relief,
water bumpiness, unrepaired 1 m shorelines, preserved source errors, wrong physical
conditioning, and chunk/pattern artifacts. It must not devote a forensic inventory
to code that the replacement method will not reuse.

## 18. Final Spec Self-Audit

Before delivery, the spec author must answer yes to every item:

- [ ] Is the recommendation decisive while leaving per-regime method ownership
  falsifiable?
- [ ] Does every literature-backed claim distinguish direct evidence, transfer,
  and project inference?
- [ ] Can every important number be traced to current code, a full primary source,
  an official data record, or a clearly labeled calculation?
- [ ] Are unresolved/paywalled/request-only sources and licenses identified rather
  than silently filled from summaries?
- [ ] Does the spec maximize the actual physical problem decomposition instead of
  equating terrain with roughness?
- [ ] Are raw observation, correction, unresolved synthesis, and packing authority
  unambiguous?
- [ ] Can a known raw DTM error be corrected without reappearing at LOD0 or a
  coarser transition?
- [ ] Can the generator abstain without inserting generic detail?
- [ ] Are every claimed regime's data, conditions, representation limit,
  specialist selection, and release state explicit?
- [ ] Are orthophoto, soil mixtures/profiles, geology coverage/fallback, ETAK
  structures, and observation confidence first-class inputs?
- [ ] Are model training, inference, deterministic identity, packing entropy,
  storage, serving, and recook costs executable and budgeted?
- [ ] Are the vertical/undercut Taevaskoda wall and all other non-heightfield forms
  excluded from this spec's success claim?
- [ ] Does user-visible ground-level beauty overrule internal consistency metrics?
- [ ] Would the exact failure shown by noise-like relief, random sine orientation,
  different-looking chunks, flat grass, elevated vegetation islands, missing
  materials/trees, 1 m shore stairs, and water bumps fail immediately?
- [ ] Does the next proof make no binary, runtime, or shader change?
- [ ] Is implementation explicitly blocked on user review of the rewritten spec?

If any answer is no, the rewrite is not ready for user review.
