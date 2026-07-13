# Adversarial Critique: Deterministic/Process Track Proposal

**Date:** 2026-07-13
**Reviewed proposal:** `deterministic-process-proposal.md`
**Verdict:** **Reject the proposal's settled learned-national-backbone conclusion. Accept its source audit as a useful, mostly accurate evidence inventory and accept a learned dense model only as one falsifiable regime-specific challenger.**

This is an independent adversarial review, not an implementation authorization. It asks whether the proposal's conclusion follows from the full deterministic/process sources and whether the resulting architecture can credibly deliver beautiful, physically conditioned, non-repeating, 0.0625 m-sampled single-valued terrain across Estonia without changing the accepted packed-height format or adding runtime synthesis.

## Evidence Language

- **Direct evidence** means the cited paper, dataset, or official code demonstrates or states the claim.
- **Project inference** means a reasoned transfer to `laas` that has not been demonstrated by the source.
- **Unsupported transfer** means the proposal treats an inference as settled despite missing target-scale or target-domain evidence.

## Executive Verdict

The deterministic/process audit correctly establishes that no reviewed deterministic paper is a ready Estonia-wide 1 m-to-0.0625 m generator. It correctly rejects the current quilt, generic noise, global pit removal, unrestricted erosion, Paris-style volumetric claims inside a heightfield, and exact preservation of known DTM defects. It also correctly identifies measured exemplars, typed structural correction, geology/soil/hydrology conditions, deterministic coordinate identity, class-specific validation, and narrow process modules as necessary parts of a credible system.

The proposal then makes a non sequitur. Failure to find a demonstrated deterministic national generator is not positive evidence that a learned dense national backbone exists or is the best architecture. None of the reviewed learned systems demonstrates the target either. The proposal's claim that a learned model can smoothly absorb the cross-product of substrate, soil, hydrology, cover, management, disturbance, and landform is a plausible research hypothesis, not a literature result. A network does not eliminate missing combinations; it can hide them behind attractive interpolation and shortcut learning.

The best-supported architecture before experiments is therefore **evidence-fused structural reconstruction plus a gated mixture of regime specialists**, where measured exemplar transfer, calibrated process/event models, deterministic conditional simulation, discriminative regressors, GANs, and diffusion compete on the regimes they can actually support. A shared learned residual model may win several or most regimes. It may not be declared the owner before the target-data, source-repair, hierarchy, national-cost, and rendered-quality gates pass.

## Severity-Ranked Findings

### Critical 1: the learned-backbone conclusion does not follow from the deterministic literature

**Proposal claim.** The lack of a complete deterministic method supports a learned dense conditional model as the national backbone because manual process tables and nearest dictionaries cannot model the full physical cross-product.

**Direct evidence.** Guérin 2016 requires similar target and exemplar terrain and explicitly does not guarantee geomorphological consistency. Argudo 2017 can jointly match elevation, orientation, drainage, vegetation, and soil-like layers, but depends on dictionary variety and can create poor coherence between neighboring patches. Schott 2024 covers hydraulic/thermal amplification and exposes national tiling dependencies. Doane 2024 covers event-driven roughness, not arbitrary surfaces. Moore 2019 measures peat microrelief, not all terrain. These sources establish that the audited deterministic methods are partial.

**Unsupported transfer.** No source demonstrates that one learned model can infer the omitted centimeter distribution for unseen combinations of Estonian geology, surficial material, soil profile, water state, vegetation history, and management. The proposal provides no theorem, target dataset, or held-out experiment showing that continuous conditioning solves combinatorial sparsity or extrapolates physically. The ML sources reviewed elsewhere stop at meter-to-tens-of-meters terrain, synthetic degradation, or different observation tasks. "Learned" changes the interpolation mechanism; it does not manufacture missing evidence.

**Required correction.** The eventual spec must label a learned dense backbone as a candidate hypothesis. Architecture selection must occur per regime after a fair bake-off. A national release may share encoders or a backbone only where held-out evidence shows that sharing improves rather than averages away regime morphology. The recommendation must permit a mixture of learned, exemplar, process, and hybrid experts behind one ordinary cooked-height output contract.

### Critical 2: source reconstruction, unresolved synthesis, and the LOD authority are not closed into one coherent surface

**Proposal claim.** Stage 1 may correct the fallible DTM, Stage 2 adds a residual, Stage 4 derives coarser fine rungs, while LOD 0 remains the existing 1 m authority and the packed/runtime architecture remains unchanged.

**Direct evidence.** ETAK supplies shoreline, river, escarpment, ditch, road, quarry, and other typed structures. At the supplied Taevaskoda position, an explicit Ahja shoreline is 1.85 m away and a natural shoreline escarpment is 5.32 m away. These are independent reasons not to preserve the visible 1 m raster staircase. The proposal also correctly accepts `base_corrected != raw_dtm` where observation confidence is low.

**Contradiction.** If LOD -2/-1 are built around `base_corrected` while the inherited LOD 0 remains the raw, defective DTM, the levels encode different terrain. The existing packed transition can hide some detail amplitude; it cannot make a corrected shoreline, removed water spike, or moved bank identical to its erroneous raw parent. Forcing the fine levels back into the raw LOD 0 null space undoes the correction.

**Required correction.** The eventual spec must define three distinct artifacts: the raw observation, the corrected structural authority, and unresolved synthesized residual. The quality-consistent path is to publish the corrected authority through the ordinary LOD 0 height payload wherever correction is accepted, derive LOD -1 from the decoded accepted LOD -2 master, and verify the complete -2/-1/0 transition against corrected rather than raw authority. This changes payload contents, not the binary format or runtime abstraction. Correction collars at coverage and confidence boundaries must also reach the corrected parent; they cannot exist only in negative LODs.

**Required correction.** Direct typed constraints must be resolved before generative synthesis. A learned model may detect ambiguous artifact candidates or interpolate where evidence is incomplete, but it must have calibrated abstention and may not silently move a mapped bank, destroy a real cliff, or reintroduce relief into a forbidden water region. Source-repair precision, rare-landform destruction, uncertainty calibration, and rendered hierarchy behavior need independent gates from residual beauty.

### Critical 3: the proposed target corpus cannot yet support either a national learned backbone or national deterministic specialists

**Direct evidence.** Moore 2019 provides valuable 1 cm-grid peat DEMs, but the surfaces are natural-neighbor interpolated, smoothed over 3 cm, vegetation-cleaned, and cover small plots. The paper reports that approximately 32 m2 was required to capture 95% of site-level variation at one site and that dominant peat structure occurs mostly over 1-10 m. Pawlik 2024 supplies 0.025 m TLS evidence for forest pit-mound morphology at one Polish granite/Cambisol spruce site. The current Lapinjarvi material is an unlabelled foreign forest TLS-derived bank, not an authoritative Estonia ground DTM. Schott, Guérin, Argudo, Doane, and Paris do not supply a national centimeter corpus.

**Unsupported transfer.** The proposal names six broad Stage-1 classes, then assumes a model can learn dense material morphology inside them. "Mineral forest," "cultivated field," "exposed rock," and "shore/wetland" are not physical regimes at the required fidelity. Each collapses multiple substrates, cover depths, water states, event histories, management states, and surface-forming processes. One foreign site per label is not a conditional distribution.

**Resolution correction.** A nominal 3-5 cm grid is not automatically truth for a 0.0625 m output. Effective horizontal modulation transfer, vertical error, ground visibility, interpolation fraction, smoothing, registration, and temporal change must be measured. A 0.0625 m sample pitch also does not mean recognizable 6.25 cm features: the grid's two-sample Nyquist limit is 12.5 cm, and recognizable irregular objects generally require more support.

**Required correction.** Data sufficiency must precede architecture selection. The spec needs a regime-by-evidence matrix with independent acquisition sites, Estonia/analogue status, effective surface resolution, vertical uncertainty relative to each synthesis band's energy, ground-classification QA, spatial footprint, transfer limits, and untouched site-level tests. A regime does not enter national production from one site, nominal raster resolution, or foreign analogy alone. Because the stated objective is detail across Estonia, missing ordinary regimes are release blockers, not permission to fill them with generic detail.

### Critical 4: national cook, storage, recook, and serving feasibility are absent

The proposal calls the model a national backbone but does not give a national resource model. The accepted zero-runtime-synthesis law makes this an architecture property, not later deployment trivia.

Using a 45,000 km2 planning envelope:

| Quantity | Approximate value before compression/replication |
|---|---:|
| LOD -2 samples per km2 at 0.0625 m | 256,000,000 |
| LOD -2 national samples | 11.52 trillion |
| Raw LOD -2 `u16` height | 23.04 TB |
| 128 m LOD -2 chunks | 2.75 million |
| Raw LOD -1 `u16` height at 0.25 m | 1.44 TB |
| Raw -2 plus -1 | 24.48 TB |

Headers and 2049-sample aprons are not the main cost. Object count, indexes, copies, CDN/storage replication, upload, verification, and retained builds add to it. More importantly, physically rich millimeter-quantized residuals may compress substantially worse than smooth DTM deltas. The current deflate codec's national bytes cannot be inferred from the rejected smooth/quilt pilot.

Pixel-space iterative inference has a similarly unavoidable lower-bound accounting. For Stage B:

```text
time = 45,000 km2 * 256e6 output-pixels/km2
       * overlap_multiplier * model_evaluations
       / measured_output_pixels_per_second
```

This excludes Stage A, condition I/O, corrections, quantization, retries, and verification. "Offline" does not make the product finite or affordable.

**Required correction.** Before a method is called implementable, the spec must require accepted-quality 1 km2 and parent-closed multi-regime benchmarks through the real encoder. Report GPU/CPU time, peak memory, condition-read volume, overlap, sampler steps, checkpoint/retry granularity, packed bytes/km2, chunks/km2, object count, national GPU-hours, storage, upload, serving, and full/partial recook costs. The user may accept a high cost; the design may not conceal it.

### Critical 5: the frozen heightfield cannot satisfy the cliff-facing interpretation of the quality target

**Direct evidence.** Paris 2019 exists because a heightfield cannot represent vertical faces, undercuts, caves, or overhangs. Its strongest results use an implicit volumetric construction and approximately 10 cm meshing; below-10-cm detail is explicitly future work. The official code is a partial recoding with hard-coded authored demonstrations. The supplied Taevaskoda reference includes a near-vertical, locally undercut sandstone wall.

**Unsupported transfer.** A dense learned residual can add ledges and gullies only where the surface remains single-valued as `H(E,N)`. It cannot make the reference wall topologically correct. Higher raster sampling produces a more densely sampled ramp, not an overhang.

**Required correction.** The synthesis-only spec must restrict its claims to single-valued terrain, including the walkable cliff top and representable bank shoulders. It must mark vertical/undercut/cave surfaces unsupported in this pass and suppress unsafe height synthesis there. If Taevaskoda cliff-face equivalence remains an acceptance requirement, the frozen height-only scope is false and structural terrain must be specified separately; that cannot be smuggled into this synthesis hook.

### High 1: deterministic and process approaches are dismissed more broadly than the evidence supports

The audit is correct that no deterministic paper is a complete national backbone. It is not correct to defer almost every process specialist until after a learned backbone exists or to compare the learned candidate mainly with Guérin's released sparsity-one nearest-patch path.

**Direct evidence.** Schott 2024 reports stronger hydraulic coherence than its selected StyleGAN, sparse, and procedural comparisons for its supported landform family. Its whole-map dependence and hydraulic specialization are real limitations, but they make it a regime specialist, not a reason to exclude it from the relevant bake-off. Doane 2024 gives explicit event geometry, production rate, scale, and diffusive decay relationships for pit-mound-like processes; its public code is illustrative rather than production-ready, but the analytical prior is stronger than an arbitrary learned texture. Moore 2019 directly proposes spatially explicit peat landscapes parameterized from measured scale relations; that is not validated as a beauty result, but a measured conditional simulator is a legitimate challenger. Argudo 2017's multilayer contextual matching is stronger than the bare Guérin reference code because it includes joint layers and broader context.

**Project inference.** A deterministic conditional simulator will not automatically be beautiful, and a large rule stack can fail exactly as the audit warns. Conversely, a learned model will not automatically be coherent or physically correct. Both require regime-held-out evidence.

**Required correction.** The fair challenger set must include:

- typed deterministic structural reconstruction with no unresolved residual;
- measured exemplar transfer using both a reproducible Guérin baseline and a contextual/multilayer nonparametric variant;
- calibrated conditional simulation for a regime with measured spatial statistics, such as peat;
- calibrated event/process generation for a regime with event evidence, such as forest pit-mound;
- masked hydraulic amplification for an actually eligible erodible catchment, if target evidence supports that scale;
- shared and regime-expert learned regressors, GANs, and diffusion;
- process/exemplar structure followed by a learned residual;
- learned structure followed by deterministic constraints.

No alternative should be handicapped by using the rejected sine/noise system as its quality representative.

### High 2: the physical phenomenon decomposition is still far too coarse

The proposal correctly asks for factorized conditions but then defines a small class list. That repeats the error at a higher semantic level. A feature label is not a morphology model.

The eventual spec needs an explicit matrix covering at least these distinct families where Estonia evidence shows they exist:

- exposed Devonian sandstone, stratification, weak interbeds, ledge recession, shallow ravines, bank failure, and colluvial/talus transition;
- exposed or shallow carbonate/dolostone, joints, grikes/karren where representable, weathered blocks, and thin-cover transition;
- till with matrix texture, clast abundance/size, boulder expression, cover depth, wetness, and forest disturbance history;
- glaciofluvial sand/gravel, aeolian sand, lacustrine/marine silt/clay, alluvium, and colluvium;
- bog, poor/rich fen, lawn/hummock/hollow, ridge-pool/string-flark hierarchy, drainage state, fire/cutover, and water-table context;
- ordinary mineral forest floor, recent and decayed pit-mound, root-plate legacy, deadwood/soil interaction, forestry ruts, and drainage ditches;
- managed agricultural states such as ploughed, harrowed, rolled, rowed, stubble, wheel-trafficked, drained, fallow, and permanent grassland, with dated direction and event evidence;
- natural river bank, floodplain, bar, wet shore, lake shore, coast/beach, erosion/deposition reach, spring/seep, ditch, and open-water bed/surface distinction;
- paths, unpaved roads, quarries, spoil, construction disturbance, animal/biotic disturbance, frost effects, and ordinary transition ground where supported.

This is not a demand to hand-author one function per bullet. It is the minimum decomposition needed to decide which evidence, representation, model family, scale bands, and failure policy each phenomenon requires. Broad labels such as `forest`, `field`, `shore`, or `rock` cannot own final geometry.

### High 3: physical conditioning is necessary but is not itself a physical model

**Direct evidence.** Mullastikukaart contains multiple soil components and shares, layered textures, humus/peat thickness, and stoniness; the current reduced soil representation discards much of this. EGT 1:50,000 has patchy coverage and does not cover the supplied Taevaskoda point. EGT 1:200,000 covers nationally but is a broad family prior. Orthophoto and ETAK provide finer visible boundaries and mapped structures. None of these coarse maps locates an individual centimeter clast or crack.

**Unsupported transfer.** Feeding all maps to a U-Net does not prove that the model uses them correctly. A high-capacity network can ignore coarse physical channels, shortcut through DTM or orthophoto texture, memorize acquisition geography, or print polygon edges into relief. Counterfactual examples in the proposal are useful but do not establish calibrated behavior on unseen factor combinations.

**Required correction.** Every condition requires an honest support scale, uncertainty, age, availability mask, and permitted causal role. The spec must require removal, shuffle, registration perturbation, temporal mismatch, and physically plausible counterfactual tests on held-out sites. It must compare the full soil mixture/profile against the reduced representation and separately test the 1:50,000 coverage edge and 1:200,000 fallback. A condition passes only if it improves held-out morphology or placement without embossing map boundaries. Unsupported combinations require calibrated abstention or an evidence-backed expert, not an attractive global fallback.

### High 4: water surface, submerged terrain bed, shoreline, and escarpment are conflated

The proposal says connected open water should use a low-dimensional surface model and that shore/water geometry must be credible. That is directionally correct but not specific enough for the existing representation.

**Direct project evidence.** The renderer has a separate packed water-surface field. The terrain height under water is a submerged bed. Existing cook logic derives water coverage and carves a modeled bed; it does not mean the terrain field should become the visible water plane. ETAK supplies water polygons, shoreline lines, river centerlines, and mapped shoreline escarpments. Orthophoto may refine a visible boundary but is not bathymetry. Maa-amet DTM can contain TIN bridges or returns over water.

**Required correction.** The spec must define separate typed outputs and authorities for water surface, terrain bed, shoreline occupancy, bank shoulder, and escarpment. Open-water surface receives no roughness. Unknown bathymetry must be modeled conservatively with explicit uncertainty or remain the accepted bed model; it cannot inherit arbitrary microdetail or be flattened blindly to the water plane. Shore correction must remove 1 m raster stepping without smoothing a real adjacent escarpment into a generic ramp. After synthesis, hard water/structure eligibility rules must be re-applied so the generator cannot recreate the diagnosed bumps.

### High 5: national seams and physical context are not solved by one master surface and a halo

**Direct evidence.** MultiDiffusion provides overlapping pixel-state consensus for a chosen model; it does not guarantee gradient, curvature, drainage, or feature continuity. Schott 2024 explicitly states that drainage and sediment propagation across patches remain a challenge. Argudo 2017 states that patch coherence depends on dictionary variety and that structured features may not join. Doane-style event histories, peat ridge systems, river catchments, and agricultural operations can exceed a 128 or 256 m synthesis window.

**Project inference.** Generating a larger support region and cropping is necessary, but a finite halo is sufficient only for phenomena whose causal support is bounded by that halo or whose larger state is supplied as a consistent condition. A window-local random seed is not a global stochastic field. Results can also change when the requested AOI changes which overlapping windows participate.

**Required correction.** The spec must define a canonical world-aligned output partition, global counter-based stochastic field, fixed support-window lattice, full influencing-window set, exact fusion quantity/scheduler for learned sampling, and crop/apron rules. Recooking the same crop with different job orders, workers, neighboring requests, batch sizes, or enclosing AOIs must produce identical accepted packed bytes on the locked inference stack. Long-range specialists must use their actual domain, such as a watershed, connected mire unit, stand/event field, or managed parcel, and only then crop storage chunks. No periodic boundary is allowed.

### High 6: removing exact raw-DTM fidelity does not remove the need for a defined analysis/synthesis hierarchy

The proposal is correct that exact block-mean fidelity to a defective DTM is not the beauty objective. It still uses terms such as `dense_residual`, `0.25-1 m band`, and `0.0625-0.25 m band` without defining a band-limited transform.

**Technical issue.** `fine - interpolation(coarse)` is a residual, not necessarily a frequency band. Without a specified world-aligned analysis filter, decimation phase, synthesis operator, edge policy, and corrected-parent projection, low-frequency correction can leak into the residual, aliases can preserve the 1 m grid, and independently trained stages can allocate the same form differently.

**Required correction.** Define the actual antialiased analysis/synthesis pyramid around the corrected authority. The finest accepted surface is canonical; stored parents are derived from it through the existing decoded-child hierarchy where required. Projection, if used, targets the corrected parent and preserves justified correction. Audit power and phase at 1 m and its harmonics, axis-aligned gradient bias, shoreline stair counts, and the exact runtime LOD transition. Zero raw-DTM error is neither required nor desirable where repair is accepted.

### High 7: the orthophoto decision is sound, but its role and cost remain underbounded

**Direct evidence.** Argudo 2018 demonstrates that registered orthophoto can improve 15 m-to-2 m terrain inference on selected alpine sites and directly reports failures from vegetation and shadows. Estonia's 20-40 cm RGB/CIR orthophoto can locate visible shore, exposed material, paths, crop direction, drainage, and recent disturbance more finely than broad polygons.

**Evidence boundary.** A 20-40 cm image cannot supervise or place the 0.0625-0.25 m height band directly. Under canopy, water, shadow, snow, objects, and temporal mismatch it may contain no ground evidence. Even in clear ground, albedo edges are not automatically relief.

**Required correction.** Fetch dated RGB and CIR for the research corpus and likely production, but treat it as structural/semantic evidence at its measured support. The spec must include download/cache size, sheet/date/version identity, registration, visibility, canopy/object/water/shadow/season masks, and a no-orthophoto fallback. Stage B may use orthophoto features only when held-out ablation proves geometric benefit without albedo embossing. Orthophoto acquisition is recommended; orthophoto-derived height is rejected.

### Medium 1: event modules need generative distributions, not single ideal primitives

The proposal correctly says event modules require morphology, eligibility, orientation, age, and substrate evidence. The current description still risks reducing a process to one canonical shape.

Doane's derivative-of-Gaussian forms explain roughness evolution and are useful analytical priors, but natural pit-mound forms vary with root plate, tree size/species, fall direction, slope, soil thickness, moisture, substrate, overlap, age, and later disturbance. Moore's peat distributions and spectra similarly do not locate or fully shape a hummock field. Schott's nominal parameters and hardness control are not Estonia constants.

**Required correction.** Every event expert needs a measured joint distribution over geometry, density, orientation, spatial interaction, age/decay, and conditions, plus a placement process and held-out validation. A paper's idealized function may constrain the generator or supply a baseline; it may not become the final event catalog without target calibration.

### Medium 2: the acceptance protocol is directionally correct but has no preregistered decision rule

The proposal asks for rendered comparisons, descriptors, variograms, repetition checks, source-repair overlays, and expert/user review. Those are the right evidence families. "Visibly add plausible structure" and "outperform" remain vulnerable to post-hoc judgment, spectacular-site bias, and average metrics hiding failed regimes.

**Required correction.** Before viewing final candidates, register site-separated test sets, camera paths, lighting/material/vegetation state, reviewer blinding, per-regime non-inferiority requirements, failure classes, and how user preference combines with measured morphology. Report distribution precision and coverage, spatial correlations, tails, nearest-training-patch similarity, multi-seed diversity, boundary strips, water/shore failures, and ordinary as well as spectacular terrain. A single national average cannot waive a failed physical regime.

### Medium 3: deterministic production identity is plausible but not fully specified

The proposal correctly binds production to model/checkpoint/input/seed identities and accepts bitwise identity only on a locked stack. That is a sound boundary. It does not yet define how distributed incremental cooks obtain partition-independent stochastic samples, how numerical changes near quantization thresholds are handled, or how a changed condition sheet invalidates dependent chunks.

**Required correction.** Bind every output to source-window hashes, condition versions/dates, corrected-authority digest, model/process version, global random-field recipe, inference environment, quantization configuration, and full influencing-domain digest. Reordered/retried jobs must encode byte-identically on the supported stack. Released packed hashes are authoritative; cross-hardware reproducibility must not be promised without a demonstrated deterministic path.

## Corrected Evidence Boundaries

| Source | Directly supports | Does not directly support |
|---|---|---|
| Guérin et al. 2016 | Paired low/high exemplar amplification; similar exemplars; compact overlapping reconstruction | Estonia classes, source repair, national coherence, or causal material inference |
| Argudo et al. 2017 | Joint multilayer/contextual exemplar matching and correlated output layers | Centimeter targets, unseen physical combinations, seamless structured networks, or national transfer |
| Argudo et al. 2018 | Registered optical conditioning can improve a selected 15 m-to-2 m task | Native 20 cm imagery as 6.25 cm ground height, canopy/water inference, or Estonia weights |
| Scott and Dodgson 2021 | Hydrology-aware pit handling in macro example synthesis and warnings about filling/breaching artifacts | Universal DTM repair, shoreline de-gridding, or wetland-safe national cleanup |
| Scott and Dodgson 2022 | Terrain-family bias, expert sensitivity, and need for like-for-like evaluation | Close-up game acceptance thresholds or a synthesis winner |
| Paris et al. 2019 | Sparse implicit volumetric geology/erosion features and the importance of strata | Heightfield cliffs, unattended national placement, or dense below-10-cm surface detail |
| Cortial et al. 2020 | Semantic multiscale rule selection and stable hierarchical identity | Material-specific 6.25 cm raster detail or an offline Estonia cook |
| Schott et al. 2024 | Strong hydraulic/thermal multiscale amplification on bounded maps | Non-hydraulic Estonia coverage or solved national patch boundaries |
| Grenier et al. 2024 | Compact slope-controlled procedural ravine patterns | Material-grounded general detail; it remains structured noise |
| Doane et al. 2024 | Event roughness production/decay theory and ideal pit/mound families | Calibrated Estonia event distributions or a complete forest surface generator |
| Moore et al. 2019 | Measured target-scale peat surfaces, distributions, scale relations, and small-site variability | Estonia-wide mire types, national context, or visually validated synthesis by spectra alone |
| Pawlik et al. 2024 | 0.025 m TLS evidence for pit-mound morphology and the inadequacy of 1 m ALS in one regime | Estonia-wide forest parameters or other soil/substrate/stand combinations |
| HuHoLa / Ilyasov 2025-2026 | Hierarchical peat labels and diagnostics on existing measured surfaces | A terrain generator |
| Reviewed learned terrain systems | Learned terrain authoring, amplification, multimodal fusion, and tiling ideas at coarser scales | A demonstrated 1 m-to-0.0625 m Estonia prior |

## Mandatory Changes To The Eventual Spec

1. Replace the settled learned-national-backbone recommendation with a falsifiable regime-specialist bake-off. Preserve a learned dense model as a serious candidate, not the default winner.
2. Define raw observation, corrected structural authority, unresolved residual, and final packed surface as separate artifacts with separate evidence and acceptance.
3. Publish accepted corrections through ordinary LOD 0 in corrected coverage and derive/verify -1/-2 against that authority. Never preserve a known defect solely for raw up/down identity.
4. Specify deterministic typed reconstruction for directly evidenced water, shore, escarpment, road, ditch, quarry, and object structure. Learned correction is limited to ambiguous evidence and must abstain.
5. Build a maximal Estonia regime/phenomenon matrix before model selection. Each row needs process, scale bands, conditions, target evidence, representation limit, generator candidates, transition rules, and release status.
6. Establish a target-data sufficiency gate using effective resolution, vertical error, ground visibility, interpolation/filtering, spatial footprint, site count, geography, date, and condition coverage rather than nominal grid spacing.
7. Add fair deterministic, contextual exemplar, conditional simulation, process, learned-regressor, GAN, diffusion, regime-expert, and hybrid challengers. Do not use the rejected quilt as the quality bar.
8. Define the full-fidelity Estonia condition contract: Mullastikukaart mixtures/profiles, ETAK structures, DTM/nDSM/CHM confidence, EGT scale/coverage/fallback, dated orthophoto, forestry/management state, and multiscale landform/hydrology.
9. Define separate water surface, submerged bed, shoreline, bank, and escarpment behavior. Reapply hard forbidden-region constraints after every synthesis stage.
10. Define the world-aligned analysis/synthesis pyramid, corrected-parent relationship, decimation phase, filters, aprons, and quantization. A residual name is not a band definition.
11. Define canonical world stochastic identity and physical-domain processing so output is invariant to chunking, support-window origin, worker order, AOI enclosure, retry, and adjacent-request sequence on the locked stack.
12. Limit the frozen heightfield claim to single-valued terrain. Explicitly leave vertical/undercut/cave cliff faces unsupported unless a separate structural representation is commissioned.
13. Fetch and version dated RGB/CIR orthophoto for the research corpus and likely production, with visibility and temporal masks. Never treat it as height truth or direct Stage-B supervision without ablation.
14. Benchmark accepted-quality outputs through inference, correction, quantization, delta/deflate packing, verification, and serving. Publish national GPU-hours, packed bytes, object count, storage, upload, and recook estimates before architecture approval.
15. Preregister per-regime visual and measured decision rules, include ordinary terrain, and retain the exact ground-level rendered result as the decisive user-observable gate.
16. Keep the frozen interface promise: all chosen methods return ordinary metric height arrays to the existing cook encoder; no runtime synthesis, provenance-special runtime path, shader change, or binary-format change is part of this method selection.

## Verdict By Component

**Deterministic/process primary-source audit:** accepted with qualifications. Its paper-level limitations are mostly accurate and useful. Its conclusion overreaches where it converts the absence of a deterministic universal generator into support for a learned universal generator and where it understates fair regime-specific deterministic challengers.

**Confidence-aware source correction:** accepted as mandatory, but it must be a separate typed reconstruction problem and must publish a corrected parent authority. "Learned, optimization-based, or hybrid" is not yet an implementable decision.

**Learned dense conditional residual:** conditionally accepted as a high-ceiling challenger. It becomes a production owner only for regimes where true target-scale data and blind held-out rendering show that it beats fair alternatives without shortcutting conditions or destroying source structure.

**Narrow event/process specialists:** accepted as first-class candidates, not merely optional decorations after a learned model. They require measured joint calibration and must remain inside physically eligible domains.

**Orthophoto acquisition:** recommended. Its high-value roles are visible structural correction, regime/event state, exposed-material cues, and agricultural/shore direction. It is not metric ground truth, especially below its pixel support or under canopy, water, shadow, objects, snow, or temporal mismatch.

**Frozen packed-height/runtime interface:** accepted. The critique requires different cooked payload contents and may require corrected ordinary LOD 0 payloads, but no new binary format or runtime synthesis path.

**National readiness:** rejected. No candidate currently passes the target-data, corrected-authority, heightfield-scope, partition-invariance, or national-resource gates.

## Cheapest Decisive Next Proof

Do not train a national diffusion model or implement a catalog of provisional events. Build one research proof with three independently judged parts:

1. **Target/evidence gate:** assemble site-separated, target-scale surfaces and full condition stacks for a small number of genuinely different regimes. Quantify effective band signal versus acquisition error and record a regime-coverage matrix.
2. **Typed source-repair gate:** reconstruct Taevaskoda's water/shore/escarpment and other held-out defect/rare-landform patches with deterministic constraints, then compare calibrated learned assistance only on ambiguous cells. Publish a corrected ordinary LOD 0 in the pilot so the hierarchy is honest.
3. **One-regime morphology bake-off:** choose one regime with defensible target data, not necessarily Taevaskoda, and compare contextual exemplar, calibrated deterministic/process simulation, learned regressor, GAN, diffusion, and hybrids through the frozen cook output. Judge blind ground-level renders, held-out morphology coverage, partition invariance, packed entropy, and cost.

The eventual national architecture should be selected from those results. The current proposal is a strong requirements draft and a useful warning against another procedural-noise failure; it is not evidence that a learned dense backbone has already won.

## Primary Sources And Code Checked

- Guérin et al., *Sparse Representation of Terrains for Procedural Modeling*, CGF 2016, [DOI 10.1111/cgf.12821](https://doi.org/10.1111/cgf.12821), full paper and official MATLAB revision `5b83d65315e9845401df78f6a104c21f9bd473d5`.
- Argudo et al., *Coherent Multi-Layer Landscape Synthesis*, The Visual Computer 2017, [DOI 10.1007/s00371-017-1393-6](https://doi.org/10.1007/s00371-017-1393-6), full paper.
- Argudo, Chica, and Andujar, *Terrain Super-resolution through Aerial Imagery and Fully Convolutional Networks*, CGF 2018, [DOI 10.1111/cgf.13345](https://doi.org/10.1111/cgf.13345), full paper and official Caffe repository revision recorded in `SOURCES-DETERMINISTIC-PROCESS.md`.
- Scott and Dodgson, *Example-Based Terrain Synthesis with Pit Removal*, C&G 2021, [DOI 10.1016/j.cag.2021.06.012](https://doi.org/10.1016/j.cag.2021.06.012), full paper.
- Scott and Dodgson, *Evaluating Realism in Example-Based Terrain Synthesis*, ACM TAP 2022, [DOI 10.1145/3531526](https://doi.org/10.1145/3531526), full paper.
- Paris et al., *Terrain Amplification with Implicit 3D Features*, ACM TOG 2019, [DOI 10.1145/3342765](https://doi.org/10.1145/3342765), full paper and official recoded repository revision `2e7bb3ee79b8d8ffabdeb0958a55c6171f2791fe`.
- Cortial et al., *Real-Time Hyper-Amplification of Planets*, The Visual Computer 2020, [DOI 10.1007/s00371-020-01923-4](https://doi.org/10.1007/s00371-020-01923-4), full paper.
- Schott et al., *Terrain Amplification Using Multi-Scale Erosion*, ACM TOG 2024, [DOI 10.1145/3658200](https://doi.org/10.1145/3658200), full paper and official C++/OpenGL revision `64fe87d57d0ea904f54eb0ec24d19da08bebd737`.
- Grenier et al., *Real-Time Terrain Enhancement with Controlled Procedural Patterns*, CGF 2024, [DOI 10.1111/cgf.14992](https://doi.org/10.1111/cgf.14992), full paper.
- Doane et al., *Topographic Roughness as a Signature of Stochastic Geomorphic Events*, AGU Advances 2024, [DOI 10.1029/2024AV001264](https://doi.org/10.1029/2024AV001264), full paper and official figure-code revision recorded in the source manifest.
- Moore et al., *Assessing the Peatland Hummock-Hollow Classification Framework Using High-Resolution Elevation Models*, Biogeosciences 2019, [DOI 10.5194/bg-16-3491-2019](https://doi.org/10.5194/bg-16-3491-2019), full paper and Zenodo target dataset `2545675`.
- Pawlik et al., *Evaluation of the Hillslope Fine-Scale Morphology under Forest Cover with Pit-Mound Topography*, Geomorphology 2024, [DOI 10.1016/j.geomorph.2024.109283](https://doi.org/10.1016/j.geomorph.2024.109283), corrected proof evidence as recorded in the deterministic source manifest.
- Noumonvi et al., *HuHoLa*, Ecological Modelling 2025, [DOI 10.1016/j.ecolmodel.2025.111212](https://doi.org/10.1016/j.ecolmodel.2025.111212), full paper and official classifier code.
- Ilyasov et al., *UAS-LiDAR Mapping of Bog Microrelief*, Drones 2026, [DOI 10.3390/drones10020121](https://doi.org/10.3390/drones10020121), full paper.
- Argudo et al., *Terrain Descriptors for Computer Graphics*, CGF 2025, [DOI 10.1111/cgf.70080](https://doi.org/10.1111/cgf.70080), full paper and official implementation.
- The ML primary-source boundaries and official revisions recorded in `ROOT-CLAIM-AUDIT.md`, `ML-PAPER-NOTES.md`, and `ML-SOURCES.md` were checked for the learned-backbone comparison; none supplies a demonstrated Estonia centimeter prior.
