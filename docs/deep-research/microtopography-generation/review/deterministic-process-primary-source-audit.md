# Deterministic, Exemplar, And Process Primary-Source Audit

Status: primary-source audit, 2026-07-13. This document evaluates methods; it does not authorize implementation or changes to the accepted fine-rung format, cook, streaming, or runtime.

## Question And Evidence Rules

The actual question is whether deterministic exemplar transfer, procedural rules, or process simulation can generate beautiful and credible geometric variance at 0.0625 m throughout Estonia, conditioned by landform, soil, surficial material, substrate/geology, hydrology, vegetation history, and land use. Preserving every one-meter DTM cell average is not the objective. The DTM is a fallible measurement that can contain interpolation grids, missing-return artifacts, water/shore defects, and vegetation leakage.

The accepted implementation boundary is fixed:

- all synthesis is offline in `asset-gen`;
- the existing LOD -1/-2 packed-height infrastructure remains unchanged;
- runtime synthesis is forbidden;
- no binary-format redesign is proposed here;
- methods must be judged on the rendered ground-level surface, not only RMSE, spectra, or a successful cook.

Each finding below is marked conceptually as one of:

- **Direct evidence:** demonstrated or stated by the primary paper/code.
- **Transfer inference:** a reasoned consequence for this project, not a result measured by the authors.
- **Open evidence gap:** something required by the Estonia target that the source does not establish.

## Executive Finding

No audited deterministic, exemplar, or process method demonstrates the required product. The closest amplification papers operate on a single elevation exemplar family, macro terrain, sparse dramatic features, or one physical process. The centimeter-scale field papers establish that real microrelief is structured and causally regime-specific, but they are classifiers, measurements, or conceptual process models rather than national generators.

The deterministic literature therefore does **not** support replacing the current failed synthesis with a larger handcrafted rule stack or dictionary method as the dense national backbone. It supports three narrower roles:

1. physically grounded condition maps and eligibility masks;
2. explicit event modules where a process has measured morphology, density, orientation, age, and substrate dependence;
3. exemplar/process baselines and diagnostic losses for a learned dense conditional model.

This is not a preference for ML by default. It is the evidence-based result of asking whether the deterministic alternatives have solved dense, centimeter-scale, multi-regime synthesis. They have not.

## Method Decision Matrix

| Method | Demonstrated scale/regime | What it actually contributes | Fatal gap for the national backbone | Decision |
|---|---|---|---|---|
| Guérin 2016 sparse representation | Example-based amplification; paper outputs up to 8192 square | Paired low/high patch transfer with low-resolution conditioning | Requires similar exemplar; code is effectively nearest-atom transfer; no semantic/material model or national coherence | Baseline/component only |
| Argudo 2017 multi-layer dictionary | Multi-layer landscape synthesis at terrain scale | Joint elevation, slope, orientation, drainage, vegetation, and soil-layer exemplars | Soil is an exemplar layer, not inferred geology; nearest-patch transitions and repetition remain | Conditioning concept/baseline only |
| Argudo 2018 orthophoto FCN | 15 m DEM + 1 m ortho to 2 m DEM, alpine sites | Registered imagery can improve structural inference | No centimeter target; shadows/canopy cause false geometry; no credible ground inference under cover | Orthophoto-conditioning evidence only |
| Scott and Dodgson 2021 | 30 km maps at roughly 30 m cells | Multi-resolution pit breaching inside patch optimization | Removes natural closed basins; creates V-shaped/single-pixel artifacts; unrelated to DTM shoreline de-gridding | Restricted hydrology QA only |
| Scott and Dodgson 2022 | Static distant realism survey | Like-for-like evaluation and expert effects | Explicitly not close-up game evaluation; no method wins across terrain families | Acceptance-study guidance only |
| Paris et al. 2019 | Sparse caves, cliffs, karst/hoodoos; approximately 10 cm meshing | Geology fields and sparse implicit 3D construction | Key value is overhangs/caves, which the frozen heightfield cannot represent; sub-10 cm is future work | Conceptual geology/event source only |
| Cortial et al. 2020 | Planet hyper-amplification to roughly 50 cm | Semantic multiscale subdivision rules and stable seeds | Runtime triangular rule graph, generic random fine rules, >2 GB, not 6.25 cm or raster cook | Rule-selection concept only |
| Schott et al. 2024 | Hydraulic/thermal erosion amplification to 8192 square | Multiscale drainage-aware erosion and deposition | Whole-map dependencies, patch-boundary problem, one geomorphic family, thousands of iterations, noise in released code | Narrow erodible-slope module candidate |
| Grenier et al. 2024 | Real-time slope-aligned ravine enhancement | Compact controlled structured pattern field | Explicit structured noise; fails on flat terrain and can reveal linear patterns | Reject as synthesis method |
| Doane et al. 2024 | Stochastic tree-throw roughness theory | Event morphology plus age/diffusive decay by wavelength | Toy/figure code and one event family, not a national surface | Calibrated forest-event prior only |
| Moore 2019 / HuHoLa 2025 / Ilyasov 2026 | Measured/classified peat microrelief | Real hierarchical scales, distributions, centimeter exemplars, class diagnostics | No validated generator; foreign sites and restricted mire types | Peat data/labels/validation only |

## Guérin Et Al. 2016: Sparse Representation Of Terrains

Primary source: Sections 3, 5.3, and 6; official repository revision `5b83d65315e9845401df78f6a104c21f9bd473d5`.

### What the paper demonstrates

**Direct evidence.** The amplification application builds paired dictionaries `(H, L)` from high-resolution exemplar patches and their downsampled versions. It decomposes each low-resolution input neighborhood over `L`, then reuses those coefficients over `H` to construct high-resolution detail. Section 5.3 states that the input and exemplar need to contain similar terrains. Figure 11 shows that increasing the atom radius weakens input control and increases exemplar influence. Section 6 reports outputs up to 8192 square and gives timing for complete and learned dictionaries on a desktop CPU.

The model contains no land-cover, substrate, soil, water, orthophoto, or geological conditioning. Its coherence is the emergent result of overlapping spatial patches, not a physical constraint. The paper acknowledges that geomorphological consistency and small-scale coherence are not guaranteed and that the representation remains a heightfield.

### What the released code actually does

**Direct evidence from code.** `terrain_super_resolution.m` extracts mean-subtracted, radial-mask patches at all valid exemplar locations. The released path invokes OMP with sparsity one, chooses the corresponding high-resolution patch at the same exemplar coordinate, restores the low-resolution patch mean, and overlap-adds the results. It does not release a production K-SVD training/selection pipeline for a national multi-regime library. Borders are replicated.

The practical released baseline is therefore closer to nearest-paired-patch shape transfer than to a rich sparse mixture. This distinction matters: a paper-level dictionary idea cannot be credited with coherence or conditioning absent from the code an implementer can actually audit.

### Estonia transfer judgment

**Transfer inference.** A large, correctly stratified, measured 6.25 cm exemplar library could make paired patch transfer useful inside one narrowly defined regime. It could also provide a nearest-exemplar baseline for a learned model. It cannot decide which morphology belongs on an Estonian pixel without a separate regime/condition system, and hard local selection will repeat distinctive forms or mix incompatible contexts.

**Open evidence gap.** There is no 1 m-to-6.25 cm experiment, no cross-biome national study, no source-error correction, no shoreline model, and no proof that overlapping chunk synthesis remains coherent when hydrology and landforms cross large boundaries.

**Decision.** Do not use this as the dense national backbone.

## Argudo Et Al. 2017: Coherent Multi-Layer Landscape Synthesis

Primary source: full paper, especially method, results, and Section 6.3. No official public source repository was located.

### Citation correction

**Direct evidence.** DOI `10.1007/s00371-017-1393-6` is the multi-layer dictionary/exemplar paper. It is not the orthophoto-guided FCN. The orthophoto paper is Argudo, Chica, and Andujar 2018, DOI `10.1111/cgf.13345`. Any proposal citing the 2017 DOI as a learned RGB-to-height method is factually wrong.

### What it improves

**Direct evidence.** The method forms a multi-resolution, multi-layer dictionary. Local matching can include elevation, slope, orientation, drainage, vegetation, soil/class layers, and a larger contextual neighborhood. The paired high-resolution atom supplies correlated output layers. The paper shows that a context domain two to eight times the patch radius can preserve larger structures better than elevation-only local matching.

This is a materially stronger conditioning idea than Guérin's elevation-only amplification: correlated properties should be synthesized jointly, and selection should see broader context than the output patch.

### What it does not establish

**Direct evidence.** The paper's "soil" is a landscape layer associated with the exemplar and output; it is not evidence that mapped parent material, bedrock, soil horizon, or moisture causally determines unseen centimeter geometry. The algorithm remains nearest matching/replacement with radial blending. Section 6.3 states that local coherence depends on sufficient dictionary variety and identifies sharp transitions when biomes are mixed. Roads and cities cannot necessarily be joined seamlessly.

**Transfer inference.** Scaling this design to Estonia would require dense centimeter exemplars for every relevant cross-product of landform, soil, substrate, hydrology, cover, management, and age. Adding labels does not solve missing exemplars. Hard class boundaries would move the visible seam problem from chunk boundaries to ecological boundaries unless the model learned continuous conditional transitions.

**Open evidence gap.** No 6.25 cm target, no national out-of-distribution test, no robust correction of bad base DTM, and no implementation artifact to evaluate at project scale.

**Decision.** Preserve the joint-condition and broad-context insights. Do not adopt the nearest-dictionary method as the national generator.

## Argudo, Chica, And Andujar 2018: Orthophoto-Guided FCN

Primary source: Sections 3-6; official Caffe repository at packed revision `a02ada6a7806a528e83842b6ae79c22dcc261c62`.

### Direct evidence

The network has separate DEM and orthophoto branches and predicts a height offset added to an upsampled DEM. Experiments use a 15 m input DEM, 1 m orthophoto, and 2 m target DEM on selected Pyrenees/Tyrol alpine terrain, with approximately 400 m training tiles. The objective is Euclidean elevation loss. Reported reconstruction error remains meter-scale.

The paper explicitly shows optical failure modes: vegetation hides ground, and shadows produce height discontinuities. It suggests that future models distinguish terrain from DSM objects. The official tiler emits non-overlapping 200-pixel output tiles with replicated borders; it has no overlap-consensus or halo crop suitable for invisible national seams.

### Estonia transfer judgment

**Transfer inference.** Orthophoto is warranted as a registered structural/class signal because it can reveal shorelines, exposed rock, drainage traces, agricultural direction, bare-soil disturbance, and recent surface boundaries that a one-meter DTM aliases. It cannot be treated as metric ground truth under canopy, water, shadow, snow, or seasonal/temporal mismatch.

**Open evidence gap.** A 7.5x alpine 15 m-to-2 m experiment is not evidence for 16x 1 m-to-0.0625 m generation. The released model is neither trained for Estonia nor a credible source of centimeter material morphology.

**Decision.** Fetch orthophoto for conditioning and correction. Do not reuse this architecture/weights as the final synthesizer without a new target-scale data program.

## Scott And Dodgson 2021: Example-Based Terrain Synthesis With Pit Removal

Primary source: Sections 3-7.

### Direct evidence

The paper correctly warns that closed basins can be natural and that conventional GIS pit filling creates flat planes while breaching can create deep, pixel-width channels. Its proposed integration applies pit removal during each level of a Gaussian-pyramid texture-optimization process instead of as a final post-process. The optimization uses Generalized PatchMatch with translation, rotation, and reflection; the guide weight decreases at finer levels. Scale transforms are rejected as physically implausible, and height-offset matching is rejected because shifted features can lose realism.

The authors prefer breaching but report steep V-shaped valleys and skip breaching at the final pyramid level because single-pixel carving becomes visible. Their unoptimized 1000-square run takes about five minutes, with PatchMatch dominating. Experiments synthesize 30 x 30 km terrain at 1000 square: roughly 30 m cells, not microtopography.

The perceptual study does not establish a universal gain. Pit removal improves artificial or unconstrained cases, while on five real guided fluvial targets it is statistically indistinguishable from the Gain baseline. All five "real" sets are fluvially shaped.

### Estonia transfer judgment

**Transfer inference.** Globally breaching depressions would destroy valid Estonia bog hollows, kettle/glacial depressions, karst, wetland pools, drainage ditches, and other closed forms. The method addresses drainage topology in generated macro terrain; it does not repair the one-meter shoreline stair-step, water interpolation, or localized DTM spikes by itself.

**Decision.** Never apply global pit removal as the synthesis backbone or universal cleanup. Reuse only its warning and possibly a masked hydrology QA step where authoritative hydrography says flow connectivity should exist.

## Scott And Dodgson 2022: Evaluating Realism

Primary source: Sections 3-7.

### Direct evidence

The first experiment shows terrain-feature bias: participants rate some real terrains as more believable than other real terrains. Consequently, methods should be compared on the same terrain/type rather than through a single aggregate ranking. The second experiment finds no universal synthesis winner. Each of four methods can be statistically indistinguishable from real for at least one set and poor elsewhere.

Participants commonly use consistency, detail, and drainage cues: exact or unrelated repetition is implausible; both over-smooth and over-sharp surfaces are implausible; closed or disconnected drainage attracts suspicion. Experts in physical geography, cartography, and image-artifact analysis discriminate better. Visual-arts expertise does not have the same effect.

Section 7 explicitly says the static distant 2D/3D image protocol differs from close-up game environments.

### Estonia transfer judgment

**Transfer inference.** Acceptance must be stratified by physical regime and compared like-for-like with held-out measured exemplars. Aggregate spectrum/RMSE or a distant beauty poll can hide catastrophic class failures. Ground-level interactive rendering is mandatory, with expert review from geomorphology/remote-sensing or image-analysis as well as art direction.

**Decision.** Use this paper to design evaluation, not synthesis.

## Paris Et Al. 2019: Terrain Amplification With Implicit 3D Features

Primary source: Sections 3-9 and official repository revision `2e7bb3ee79b8d8ffabdeb0958a55c6171f2791fe`.

### Direct evidence

The method builds an implicit 3D construction tree. It combines subsurface/geological resistance fields, Poisson-distributed primitives, deformations/faults, shallow erosion using invasion percolation, and shape grammars to create caves, sea cliffs, arches, karst, hoodoos, and goblins. Scenes span about 0.35-6 km and are sparse in volumetric occupancy. Table 1 reports seconds-scale construction/meshing for authored examples and around 10 cm mesh precision.

The paper states that widespread volumetric effects reduce the sparse representation's efficiency. Validation is visual because corresponding real 3D data are scarce. Many results require user editing: the table reports tens to hundreds of edits and several minutes of expert work; the Benagil example is fully hand-authored. Section 9 explicitly identifies details finer than 10 cm as future work because many tiny primitives increase memory and compute cost.

The official repository README says it is a recoding rather than the paper implementation and that results/timings can differ. It omits some optimized meshing and grammar functionality. The code consists of hard-coded C++ demonstrations and marching-cubes OBJ output. The inspected karst setup uses hard-coded strata/noise, break probabilities, radii, altitude, and slope choices; it does not ingest Estonia geology maps into a tiled raster cook.

### Estonia transfer judgment

**Transfer inference.** Subsurface material and strata should condition whether cliffs weather into ledges, blocks, gullies, or smoother mantles. That causal insight is valuable. But the paper's main achievement is truly volumetric: overhangs, arches, and caves. Projecting it into the frozen heightfield discards the contribution the paper was designed to add.

**Open evidence gap.** It does not demonstrate dense surface detail below 10 cm, unsupervised national placement, or no-edit synthesis. A heightfield-only implementation cannot reproduce its showcased geometry.

**Decision.** Do not adopt or port this method in the current synthesis-only pass. Use geology/resistance-field ideas as model conditions and future representation research, not as evidence that the current packed heightfield can generate volumetric cliffs.

## Cortial Et Al. 2020: Real-Time Hyper-Amplification Of Planets

Primary source: Sections 3-9.

### Direct evidence

The method refines a coarse planetary triangular mesh using deterministic dyadic subdivision on the GPU. Vertex/edge semantics include terrain, rivers, lakes, gullies, and sea. Rule parameters include crust elevation, orogeny age, wetness, hills, plateaus, and valley cross-sections. Stable per-vertex seeds keep refinement deterministic.

The demonstrated maximum ground resolution is approximately 50 cm, with about four million triangles, around 25 Hz refinement, and over 2 GB GPU memory. The paper's reference to centimeter precision concerns coordinate/texture precision, not six-centimeter geometric detail. Fine landform rules use generic deterministic random displacements rather than measured substrate-specific morphology. Reported limitations include insufficient river density and temporal slope/shading artifacts. The authors identify data-driven exemplars as future work.

### Estonia transfer judgment

**Transfer inference.** Semantic rule selection, hierarchical conditioning, and coordinate-stable seeds are good cook design principles. The actual method is a runtime GPU triangular rule graph, not the frozen raster fine-rung pipeline. Offline execution would still require a representation and parameterization rewrite and would leave generic random fine detail.

**Decision.** Reject as an implementation candidate. Keep only the semantic hierarchy and deterministic seed concepts.

## Schott Et Al. 2024: Multi-Scale Erosion

Primary source: Sections 4-7 and official repository revision `64fe87d57d0ea904f54eb0ec24d19da08bebd737`.

### Direct evidence

The method repeatedly bicubic-upsamples terrain and applies hydraulic erosion, thermal stabilization, and sediment deposition at each resolution. Elevation distribution is retargeted. Hardness fields and multiscale depression breaching steer the process. Demonstrations include x16/x32 amplification and outputs up to 8192 square.

The paper is candid about scale dependencies: water routing and sediment transport need broad context; independently processed patches do not naturally transmit drainage across boundaries; simple overlap/blending does not solve the dependency. The tested implementation has an 8192-square limit and requires thousands of whole-map iterations. It synthesizes hydraulic/thermal landforms unless additional geological processes are supplied.

The public code is C++/OpenGL, not NumPy/SciPy. The predefined schedule stops at 2048. Inspection found that the bound hardness input is unused in the main erosion shader path, the thermal shader introduces simplex perturbation, and boundary logic is periodic. Those details matter because the repository cannot be treated as a ready deterministic reference for Estonia's nonperiodic national tiles.

### Estonia transfer judgment

**Transfer inference.** This is the strongest deterministic candidate for a narrow erodible-slope or gully module, provided it runs on a coarse watershed context and emits only into physically eligible masks. It is wrong for water, bogs, flat mineral forest, tillage, exposed rock faces, or generic national coverage. Its use of noise to hide grid artifacts also conflicts with the requirement that structure be grounded rather than decorative.

**Decision.** Prototype only as an optional, masked process module after the dense model exists and only if an Estonia class has evidence that this landform family is missing. Do not make it the backbone.

## Grenier Et Al. 2024: Controlled Procedural Patterns

Primary source: full paper, especially results and limitations.

### Direct evidence

The method adds a residual `e = h + r` built with controlled Phasor/structured noise, directed by terrain control maps to resemble slope-aligned erosion patterns. The authors report that flat input lacks the high-frequency cues required for good direction/control, that floodplains and meandering rivers are poor targets, and that linear structures can become visible in flat areas.

### Estonia transfer judgment

The method is an academically improved version of the exact failure mode already observed in the project: visually recognizable oriented procedural fields that vary by chunk or class without explaining physical material structure.

**Decision.** Reject for synthesis. Control maps may inspire conditioning channels, but no structured-noise residual should ship as ground geometry.

## Argudo Et Al. 2025: Terrain Descriptors

Primary source: full paper and official repository revision `03e27f6471771e365d473d03557f8053d21e3ff9`.

### Direct evidence

The work reviews and evaluates terrain descriptors rather than synthesizing terrain. It emphasizes scale dependence and compares curvature, slope, drainage, roughness, and related descriptors. Preliminary artist feedback finds some curvature measures useful on bare rock while roughness measures lack adequate precision for some tasks.

### Estonia transfer judgment

Descriptors are useful as a diagnostic panel and possibly as auxiliary training losses. They must be evaluated at explicit physical support sizes. Matching descriptor histograms does not establish correct placement, causal material structure, continuity, or beauty and must never become the sole objective.

**Decision.** Use for stratified diagnostics, not generation or final acceptance.

## Process And Field Evidence

### Doane Et Al. 2024: stochastic geomorphic events

**Direct evidence.** The model explains wavelength-banded topographic roughness as the superposition of discrete events followed by diffusive decay. Tree-throw pit/mound morphology is represented with a first derivative of a Gaussian; event age and rate affect the spectrum. The paper explicitly notes that one process can dominate a wavelength band and that natural forms and correlated histories can depart from the idealized model.

The official repository is figure/research code with periodic toy domains, hard-coded scales near 0.1 m, and unseeded event placement. It is not a national cook.

**Transfer inference.** Event models are credible only when event rate, orientation, size/shape distribution, age/decay, soil depth, moisture, stand history, and substrate are calibrated. This supports a tree-throw module for eligible forest, not a universal roughness field.

### Pawlik Et Al. 2024: forest pit-mound measurement

**Direct evidence.** The Karkonosze study compares 1 m ALS and 0.025 m TLS surfaces in a spruce forest over granite/Cambisol. The TLS DTM represents pit-mound morphology better; TRI on the 1 m product creates stripes not present in reality. The paper ties form scale to tree height, soil thickness/texture/moisture, elevation, slope, and wind. Pit-mound densities can reach hundreds per hectare and forms can persist centuries to millennia. Pits and mounds differ in moisture, electrical response, organic carbon, and soil fractions.

**Transfer inference.** The paper directly refutes the assumption that the one-meter DTM contains a blurred but otherwise faithful version of all fine structure. It supports conditioning forest events on soils, substrate, stand/disturbance context, and topography. Its Polish granite spruce site does not supply Estonia-wide parameters.

### Moore Et Al. 2019: centimeter peat exemplars

**Direct evidence.** The study publishes 1 cm SfM DEMs from nine northern peatlands. A 3 x 3 smoothing step gives roughly 3 cm support; manual validation reports centimeter-level error. Vascular vegetation is manually removed. Most topographic variance is above roughly 0.6 m wavelength; site distributions often require two or three Gaussian components rather than a binary hummock/hollow split. About 1-10 m scales dominate much of the structure. The Zenodo archive includes the actual DEMs, point clouds, transects, and analysis scripts.

**Transfer inference.** This is valuable measured target-scale data for peat representation learning, conditional statistics, and holdout evaluation. It does not cover Estonia, every mire type, or patterned ridge-pool systems. The paper's suggestion that spectra/fractal parameters can generate surfaces is not a validated visual synthesis result and should not justify generic fractal noise.

### HuHoLa 2025: classification, not generation

**Direct evidence.** HuHoLa fills sinks in a DEM and its inverse and thresholds the resulting height/depth measures to label hummock, hollow, and lawn. The paper reports its best validation around 50 cm aggregation and warns that finer than about 30 cm can fragment forms. The repository implements this classifier.

**Decision.** Use HuHoLa labels or morphology diagnostics on measured/produced peat surfaces at supported aggregation scales. Never describe it as a synthesizer.

### Ilyasov Et Al. 2026: hierarchical bog microrelief

**Direct evidence.** UAS LiDAR is gridded at 0.1 m and normalized to a basal hollow surface. The paper separates ridge/hollow structure at approximately 10-100 m from hummock/depression structure at approximately 0.5-3 m. Its rule thresholds are optimized against 12 field plots and used for biomass upscaling. This is classification/application, not a generator.

**Transfer inference.** Peat synthesis must be hierarchical and conditioned by mire hydrology/type. Hummocks cannot be sprinkled independently as small noise. The Western Siberia thresholds are validation clues, not Estonia constants.

### Nungesser 2003 HOHUM

Only publisher-accessible portions were available. They describe a monthly dynamic model coupling climate, hydrology, Sphagnum growth, peat accumulation, and decomposition, with site validation. Because the full methods were not lawfully available, this audit does not infer an implementable spatial generator from HOHUM. It remains evidence that peat relief is a coupled ecological/hydrological process rather than a texture.

### Cordonnier 2018/2023: geology and glacial processes

**Direct evidence.** These systems model mountain/landform-scale evolution using subsurface geology, erosion resistance, uplift, fluvial action, or glacial erosion. They demonstrate that material and geological structure control landscape form.

**Open evidence gap.** Their scales and targets are not unresolved 1 m-to-6.25 cm surface morphology. Estonia's glacial macroforms are already substantially represented in the DTM; rerunning a mountain/glacier model does not derive modern soil-surface detail. The papers support condition variables, not centimeter weights or parameters.

## Cross-Paper Conclusions

### 1. A deterministic national backbone is unsupported

Every plausible deterministic family fails in a different way:

- local dictionaries need already-matched centimeter exemplars and repeat them;
- procedural rules devolve into generic random patterns unless densely calibrated;
- erosion specializes in hydraulically erodible landforms and needs global context;
- pit removal can destroy valid closed wetland/glacial/karst forms;
- volumetric methods cannot retain their key cliff/cave value in a heightfield;
- field/process studies measure or classify specific regimes but do not synthesize all regimes.

Combining these methods does not automatically close the gaps. It creates a larger hand-authored transition problem whose unsupported regions would still be filled with generic noise.

### 2. The useful deterministic contribution is causal structure

The field and process literature strongly supports these inputs:

- bedrock/lithology and depth to rock;
- Quaternary parent material and soil texture/depth/organic content;
- hydrology, water table, wetness, drainage, and authoritative water geometry;
- slope, curvature, exposure, accumulation area, and landform hierarchy;
- vegetation/stand history and disturbance event likelihood;
- land use and management state, including cultivation direction;
- orthophoto-derived current boundaries and exposed-surface cues with uncertainty masks.

These must be conditions or event eligibility variables, not a table of global amplitude knobs.

### 3. Source correction and synthesis are distinct

Scott's pit-removal analysis and Pawlik's ALS/TLS comparison both show why exact source preservation is wrong, but neither supplies a universal corrector. A credible pipeline must first estimate a corrected structural base and confidence, then synthesize unresolved detail. Authoritative water/shore/road/building data and orthophoto can contradict a bad DTM observation. The final 1 m aggregate may deliberately differ from the source where evidence supports correction.

### 4. Orthophoto acquisition is warranted

Argudo 2018 directly demonstrates that registered optical data can improve terrain inference, while also documenting canopy/shadow failure. For Estonia it is most valuable for visible boundaries, exposed material, agricultural direction, drainage traces, and shoreline correction. It is optional/low-trust under canopy and water, and it never replaces DTM, soil, geology, or measured centimeter targets.

### 5. Visual validation must be class-specific and ground-level

Scott and Dodgson 2022 shows that terrain-family bias invalidates undifferentiated realism comparisons. Argudo 2025 shows descriptor sensitivity is scale-specific. Therefore validation must combine:

- held-out real target surfaces per physical regime;
- descriptors/spectra/variograms at explicit physical scales;
- repetition, transition, drainage, and feature-placement checks;
- exact user-visible ground-level rendered comparison in the accepted runtime;
- experts able to detect geomorphic and image-processing artifacts.

Metrics diagnose. The rendered surface decides.

## Deterministic Track Recommendation

Do not commission a new deterministic dictionary/process-only national generator. Use this track to constrain and test a learned dense conditional synthesizer:

- Guérin/Argudo provide exemplar baselines and the idea of paired multi-layer/context conditioning.
- Cortial provides semantic hierarchy and deterministic coordinate identity.
- Doane/Pawlik support calibrated event modules for tree throw.
- Moore/HuHoLa/Ilyasov provide peat exemplars, labels, and hierarchical diagnostics.
- Schott is a later, optional module for eligible erodible gullies, not current stage one.
- Paris/Cordonnier justify geology/substrate conditioning but do not fit the frozen heightfield as implementations.
- Grenier-style procedural pattern residuals are rejected.

The concrete synthesis proposal is in `deterministic-process-proposal.md`.
