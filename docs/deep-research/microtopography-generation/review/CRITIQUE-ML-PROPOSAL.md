# Adversarial Critique: ML Microtopography Proposal

**Date:** 2026-07-13
**Reviewed proposal:** `ML-SYNTHESIS-PROPOSAL.md`
**Verdict:** **Reject as the complete production architecture; conditionally accept pixel-space conditional diffusion as one learned residual challenger after the critical gates below pass.**

This is an independent critique, not an implementation authorization. It evaluates whether the proposal can credibly deliver beautiful, materially appropriate, non-repeating, 0.0625 m-sampled terrain across Estonia from a fallible 1 m DTM while keeping synthesis in `asset-gen` and leaving the accepted packed-height transport unchanged.

## 1. Bottom line

The proposal is materially better than `r1.md` and the rejected LUKE-quilt implementation. It correctly rejects Stable Diffusion height priors, exact DDNM projection onto a faulty observation, orthophoto-as-displacement, per-chunk generation, and random global roughness. It also correctly makes real target-scale ground data, measured degradation, typed covariates, site-separated testing, and rendered visual review mandatory.

It nevertheless recommends a specific two-stage diffusion architecture before establishing the prerequisites that would make that architecture scientifically or operationally credible. No reviewed paper demonstrates the required transformation, and the locally archived target-scale datasets do not cover the national physical state space. More seriously, the proposal entangles evidence-backed source reconstruction with hallucinated residual synthesis, leaves the corrected 1 m surface inconsistent with the frozen published LOD 0, defines residuals that are not actually band-limited, and gives mutually inconsistent rules for window seeding and overlap consensus.

The best-supported architecture at this point is not "two-stage diffusion owns terrain." It is:

1. deterministic, uncertainty-aware reconstruction of the corrected base wherever direct DTM, ETAK, water, shore, escarpment, and object evidence exists;
2. an explicit parent-consistent residual pyramid derived around that corrected authority;
3. regime-specific process, exemplar, deterministic, GAN, and diffusion challengers for unresolved morphology;
4. a learned generator only in regimes where target-scale data prove that it raises the visible quality ceiling;
5. explicit exclusion of vertical, undercut, and overhanging cliff faces from heightfield claims.

Diffusion may win item 3 for some regimes. The current literature does not establish that it will, and convenience or novelty cannot substitute for the bake-off.

## 2. Severity-ranked findings

### Critical 1: the target corpus does not yet support a national 6.25 cm prior

The proposal acknowledges data insufficiency, but its recommendation still presupposes that a unified terrain-native prior can be trained. That is not established.

- The published terrain ML systems stop at meter-to-tens-of-meters output. Argudo et al. reconstruct 2 m DEMs from synthetic 15 m inputs plus 1 m orthophotos; Liu et al. reconstruct 2 m DEMs from synthetic 8/16/32 m inputs while resampling native 20 cm imagery onto the 2 m grid; GATA's source terrain is normalized around 2 m; Terrain Diffusion's released hierarchy is 30-90 m. These works transfer architecture ideas, not centimeter morphology.
- Moore et al.'s peat data are a valuable true-scale exception, but they are 3.2-10.1 m2 vegetation-clipped plots. The DEM is a 1 cm grid, natural-neighbor interpolated and smoothed with a 3 cm mean filter. Laboratory RMSE is below 1 cm, while field median absolute elevation disagreement is 1.8 cm. The paper itself finds that about 32 m2 was needed to capture 95% of variance at one site and that most microtopographic variance lies at 1-10 m scales. This cannot train or validate a national 128-256 m context model by itself.
- The Lapinjarvi archive is a boreal forest TLS/stem survey: 18 nearby plots, fixed scan stations, acknowledged occlusion, and no delivered authoritative bare-earth DTM. The project's derived patches are useful foreign calibration evidence, not Estonia-wide truth.
- A target sample pitch of 3-5 cm is not automatically adequate training truth for a 6.25 cm output. Five-centimeter source sampling undersamples the output's Nyquist limit, and centimeter vertical error may be comparable with the entire high-frequency signal in smooth regimes. Target modulation transfer, interpolation, filtering, vertical error, ground visibility, and temporal change must be measured per acquisition.
- "Five proof regimes" is not coverage of Estonia. The real state space includes interactions among substrate, surficial deposit, soil profile and mixture, hydrologic position, slope, land use, management event, vegetation/visibility, disturbance, and acquisition regime. A single U-Net can hide unsupported combinations behind plausible-looking averages.

**Required correction:** make the data audit a pre-architecture gate, not Proof 0 after recommending a model. Maintain a regime-by-evidence matrix with independent train, validation, and untouched test sites; measured spatial transfer function; vertical uncertainty relative to each residual band's energy; ground-visibility fraction; acquisition footprint; season/date; and transfer-confidence label. No regime enters Stage B production from one site, an interpolated target, or a foreign analogue alone. Unsupported regimes remain explicitly unsupported rather than receiving generic detail.

### Critical 2: direct source reconstruction is wrongly coupled to generative synthesis

The proposal gives Stage A "most source correction" and jointly trains a correction/confidence head with generative bands. This is unsafe where better evidence already specifies the correction.

At the Taevaskoda review coordinate, ETAK places a natural shoreline escarpment 5.32 m away and the Ahja river boundary and clear shoreline 1.85 m away. These vector constraints directly show that the 1 m grid-shaped shore and bank are not immutable. A generative head should not decide whether or where those structures exist. It should receive a corrected, typed structural base.

Calling a neural head "deterministic" only means repeatable inference for fixed weights and inputs. It does not make its confidence calibrated or its correction evidence-backed. Joint training creates a confound: the model may move the base to make its residual denoising loss easier, then report confidence from the same features. GrounDiff is a warning, not validation: it performs same-grid DSM-to-DTM translation and explicitly fails on abrupt cliffs and in dense vegetation without visible ground. It also states that its probabilistic model cannot give deterministic accuracy guarantees.

Water and shoreline treatment is likewise underspecified. A low-dimensional water surface is appropriate for the water surface, but the packed terrain field may represent a bed or bank under a separate water surface. The proposal does not state which geometric surface is being corrected, nor does it hard-forbid the residual model from reintroducing water bumps after the typed repair.

**Required correction:** split the pipeline into auditable stages.

1. Reconstruct a corrected base deterministically from DTM evidence plus explicit ETAK water, shore, escarpment, ditch, road, quarry, and hard-surface constraints. Carry uncertainty and preserve typed discontinuities; do not indiscriminately smooth an escarpment into a shore ramp.
2. Use nDSM/CHM, orthophoto, and learned models only to detect ambiguous artifact candidates or estimate missing evidence. Require calibrated selective risk and an abstention path. They may not override direct structural constraints silently.
3. Freeze the corrected base before unresolved residual synthesis. Hard-project water and other forbidden regions after every learned stage so a residual model cannot recreate the diagnosed defect.
4. Evaluate source-repair precision, false-positive landform destruction, and calibration independently from residual beauty.

### Critical 3: corrected fine terrain can disagree with the frozen LOD 0 parent

The proposal permits `g_1m != y_1m`, freezes LOD 0 as the existing Maa-amet rung, and says the integration result should be only different fine payload contents. Those three statements cannot all hold without a visible hierarchy disagreement.

If LOD -1/-2 are generated around corrected `g_1m` while LOD 0 still contains raw `y_1m`, the near surface and its parent represent different terrain. A soft observation term does not define their transition and cannot guarantee absence of LOD morph, vegetation, collision, or probe disagreement. Conversely, forcing the fine result to downsample exactly to raw `y_1m` preserves the source defects that correction was introduced to remove.

**Required correction:** choose and document one authority relationship before model work:

- publish the corrected `g_1m` through the ordinary LOD 0 height cook, then derive LOD -1 from LOD -2 and verify the runtime-equivalent transition against corrected LOD 0; or
- keep raw LOD 0 and constrain correction to a transition-safe null space with a demonstrated rendered morph, accepting that many 1 m defects cannot be repaired.

The first option is the quality-consistent one and does not require a new binary format, but it is more than "only different fine payload contents." Parent consistency must target the corrected authority, not the erroneous observation.

### Critical 4: the proposed heightfield cannot satisfy exposed-cliff claims

A `H(E,N)` field stores one elevation per horizontal position. No learned prior can make it represent a vertical wall, undercut ledge, cave, or overhang. Higher sampling only creates a denser steep ramp. This directly affects Suur Taevaskoda and other exposed Estonia cliffs.

The proposal mentions cliff preservation, sandstone ledges, and exposed-rock breaks as though a height residual could cover the class, while simultaneously forbidding format/runtime changes. The current broader spec's structural-cluster concept is outside this proposal's frozen height-only interface. Those are different scopes.

The sample pitch also needs honest language. A 0.0625 m grid is 6.25 cm-sampled terrain, not evidence that coherent 6.25 or 8 cm objects are represented. Two samples are only the Nyquist minimum for a periodic signal; recognizable clasts, narrow cracks, sharp lips, and loose rocks generally need more samples and may not be single-valued.

**Required correction:** scope the ML height proposal to single-valued terrain and conservatively suppress/abstain in detected non-heightfield faces. Do not claim that it solves the Taevaskoda cliff face. Evaluate a structural representation separately, or explicitly leave those faces unsupported. State the minimum feature sizes that the sampled heightfield can actually preserve after terrain triangulation and LOD simplification.

### Critical 5: national cook and storage feasibility is not confronted

The proposal reduces national feasibility to "measure quality per cooked square kilometer." The scale requires an explicit budget before choosing pixel-space diffusion.

Using a 45,000 km2 planning envelope:

- a 6.25 cm field contains about `11.52e12` samples;
- raw uint16 LOD -2 alone is about `23.0 TB` before headers, aprons, indexing, replication, or serving overhead;
- 128 m chunks imply about `2.75 million` LOD -2 chunks;
- raw LOD -1 adds about `1.44 TB`;
- realistic learned residuals may compress materially worse than smooth DTM deltas, so compression must be measured on accepted outputs rather than extrapolated from the base DTM.

Compute is equally unresolved. Stage B's proposed 128-256 m context corresponds to 2048-4096 pixels at 6.25 cm. A 256-pixel Stage B tile sees only 16 m. GrounDiff reports roughly 60 ms per 256x256 reverse step on its GPU; ten non-overlapping steps over one 2048-square chunk would already be roughly 38 seconds before overlap, richer conditioning, Stage A, I/O, or this project's model differences. This is not a performance prediction for the proposed model, but it demonstrates that "offline" does not make the national cost disappear.

**Required correction:** before production architecture selection, benchmark representative accepted-quality models and publish:

- peak host/GPU memory and wall time per square kilometer for both stages;
- context size actually seen at each physical scale;
- overlap multiplier and sampler-step scaling;
- deterministic retry/checkpoint behavior;
- packed bytes per square kilometer after the real encoder;
- national compute, storage, object-count, upload, and serving projections;
- the explicitly funded coverage policy if full national LOD -2 is not intended.

No quality method should be rejected merely for being expensive, but an architecture is not implementable until its accepted cost is known.

### High 1: two x4 model stages are selected by storage layout, not evidence

The fact that the packed hierarchy contains 0.25 m and 0.0625 m rungs does not prove that two x4 generative models are the best statistical factorization. Argudo et al.'s failed cascade shows that generated intermediate data can break cross-modal correlations; it supports testing sampled intermediates and retaining the original observation, but it does not validate this cascade.

A direct multiscale x16 model, a single coarse-to-fine U-Net, a continuous-coordinate model, a learned 0.25 m structure stage plus deterministic/material expert at 6.25 cm, or more than two internal scales could have a higher ceiling. The packed -1 rung can always be derived from the accepted finest field.

**Required correction:** treat two x4 stages as a challenger, not the recommendation. Compare direct/multiscale and cascaded formulations using the same target data, physical context, parameter budget, and packed-output gate. Measure exposure bias, error accumulation, conditional mutual information, and visible morphology, not only per-cell RMSE.

### High 2: the residuals are called bands but are not mathematically band-limited

`r_A = h_025 - U4(g_1m)` and `r_B = h_00625 - U4(h_025)` are residual definitions, not frequency bands. Without a specified antialiased analysis/synthesis pair, they can contain low-frequency drift, aliasing, grid-phase artifacts, and frequencies that the parent cannot represent. A U-Net prediction constrained only by losses can move low frequencies in either residual.

InfiniteDiffusion's relevant transferable mechanism is not simply "Laplacian components": after decoding, it re-extracts the low-frequency component so high-frequency synthesis cannot drift the base. MultiDiffusion does not provide this property.

**Required correction:** specify the exact world-aligned analysis filter, decimation phase, synthesis filter, edge/apron handling, and projection around the corrected parent. Derive the published parent from the accepted finest field or project residuals into the corrected parent's null space after every generative stage. Audit two-dimensional spectra and grid phase; do not optimize exact agreement with the bad raw DTM.

### High 3: the noise and overlap rules are internally inconsistent

The model section asks for a coordinate-seeded global diffusion-noise field, but the cook algorithm derives a seed from each support-window coordinate. Those are not equivalent. Independently seeded overlapping windows see different initial random variables for the same world point. Averaging their denoised states can blur modes or create partition-dependent output.

MultiDiffusion starts from one global latent and applies crop operators to that shared state. Its closed-form step is a per-pixel weighted least-squares average of proposed denoising updates. It does not prove slope, curvature, drainage, or topology continuity, and its authors explicitly state that result quality depends on the reference prior and seed. Averaging predicted noise and averaging predicted next states are also not interchangeable for all samplers, despite the proposal saying "states or noise."

InfiniteDiffusion supplies a more relevant coordinate/query-order mechanism, but it is demonstrated on 30-90 m terrain and image models. Its seed-consistency and cache rules transfer; its learned prior and quality claims do not.

**Required correction:** define one counter-based, coordinate-indexed Gaussian field whose value is invariant to window origin, request order, worker count, and partition. Define exactly which sampler quantity is fused and derive the update for the chosen scheduler. Then require bit-identical locked-stack output under different tilings and orders, plus derivative and hydrology seam tests. Consensus is an inference mechanism, not a physical-consistency proof.

### High 4: the Estonia condition contract is incomplete and currently loses its best soil evidence

The proposal names EstSoil-EH, broad geology, and condition masks, but it does not consume the actual source fidelity now available.

- The national 1:10,000 Mullastikukaart has up to four soil components and weights (`Sif1..4`, `Osa1..4`), layered textures (`Loimis1/2`), humus/peat horizon (`Huumus`), stoniness, and fertility. The current five-plane parser reads only `Sif1`, simplified top texture, stoniness, and fertility. A model trained through that contract would discard physically consequential mixture and vertical-profile information.
- At Taevaskoda the source polygon is a 70/30 soil mixture with layered texture and humus fields. A one-hot class or FiLM label is not an adequate representation.
- EGT 1:50,000 bedrock, surficial, and geomorphology coverage is patchy and absent at Taevaskoda. Missing coverage must not become a class. The 1:200,000 Burtnieki Formation polygon is a broad sandstone/siltstone/clay prior only; it cannot locate outcrop expression or local fractures.
- Categorical polygon boundaries are map artifacts, not 6.25 cm physical discontinuities. One-hot rasterization can create geology/soil seams exactly where the proposal intends to remove chunk seams.

**Required correction:** preserve full soil mixtures and profiles, source typography, map scale, coverage, confidence, and provenance. Encode distributions or compositional mixtures rather than a dominant class. Mediate coarse map transitions using local DTM, ETAK, land cover, orthophoto, and uncertainty; never stamp polygon edges into relief. Require condition-shuffle and polygon-edge falsification tests.

### High 5: orthophoto is useful, but cannot supervise Stage B as implied

Fetching dated RGB/CIR orthophoto is warranted. It can locate sub-meter shore, path, drainage, exposed-material, crop-row, disturbance, and land-cover evidence not present in the DTM. Argudo et al. demonstrate a measurable meter-scale benefit.

The transfer boundary is strict:

- current national imagery is generally 20-40 cm, so it cannot spatially determine the 6.25-25 cm band;
- Argudo's target is 2 m and the paper reports vegetation, shadow, small-rock, and image-noise failures;
- Liu et al. resample 20 cm imagery onto the 2 m target grid, report 1.63-3.27 m DEM RMSE, and achieve only 10.5% mIoU from domain-shifted pseudo-labels;
- under canopy, water, shadow, and vertical/occluded faces, orthophoto does not observe ground;
- annual imagery and multi-year DTM/ETAK/forest/management data can disagree temporally.

**Required correction:** use orthophoto as a dated structural/class condition with explicit visibility, shadow, water, object, registration, and temporal-confidence masks. It may influence the Stage B distribution but not claim to place unseen centimeter features. Prove value by held-out no-image, shuffled-image, date-mismatch, and canopy/water ablations. If it merely embosses albedo edges, reject that fusion path.

### High 6: semantic conditioning is not a physical model

Soil, geology, and land-use labels can alter a learned distribution, but FiLM and counterfactual class swaps do not establish causality or process correctness. A network can learn a texture lookup, ignore weak covariates, or interpolate incompatible morphology into generic bumpiness. Calling the model "physically conditioned" is defensible; calling its output physically modeled is not.

One national U-Net also creates a capacity and imbalance risk. Common smooth agriculture may erase rare outcrops, while class rebalancing may make rare dramatic relief overexpressed. Coarse categorical maps can dominate local evidence or be ignored entirely.

**Required correction:** compare a shared model against regime experts and a mixture-of-experts gate with calibrated interpolation. Use explicit process models where the process is known and measurable: connected water/shore reconstruction, hydraulic/channel structure, peat microform statistics, tillage direction/event state, and other supported families. Require condition influence to match held-out morphology distributions, not only visibly change the image.

### High 7: the challenger set is biased toward learned image models

The mandatory bake-off compares a deterministic FCN, GAN, and diffusion model, with the rejected current synthesis as the only non-learned reference. That cannot establish the best quality ceiling.

Guérin's paired dictionary is limited by exemplar similarity, repetition, and weak geomorphic coherence, but it is a serious measured-exemplar challenger. Schott et al.'s multiscale erosion is limited to hydraulic landforms and whole-map dependencies, but it demonstrates stronger drainage coherence than patch assembly in its supported regime. Deterministic vector-constrained source repair and regime-specific process families may decisively outperform a generic learned model on shores, channels, peat, alvar fractures, agricultural rows, or other typed structures. Hybrid models may outperform every monolithic candidate.

**Required correction:** add at least these fair challengers:

- deterministic ETAK/DTM constrained base plus no learned residual;
- Guérin-style measured exemplar/dictionary amplification with the same target corpus;
- supported process families evaluated only in their valid regimes;
- deterministic corrected base plus learned residual;
- process structure plus learned residual;
- shared versus regime-expert learned models.

Do not handicap alternatives with the rejected sine/noise implementation or compare only average metrics over mixed regimes.

### Medium 1: the proposed objective is a research menu, not an implementable loss

The proposal lists diffusion, height, gradient, curvature, normal, spectral, variogram, confidence, correction, hydrology, and topology terms, but does not define where they are applied in the denoising trajectory, how decoded `x0` is formed, how units are normalized, how masks interact, or how weights avoid contradictory gradients. A differentiable drainage surrogate can itself create smoothing or optimization artifacts.

**Required correction:** define a minimal base objective, add one term at a time, and require site-held-out ablations. Every term needs a physical scale, units, mask, application timestep, expected benefit, failure mode, and removal gate. Diagnostic metrics need not all become training losses.

### Medium 2: confidence and uncertainty need calibration, not another output channel

The proposal requests `w_obs`, correction class probabilities, and uncertainty without defining ground-truth labels, epistemic versus aleatoric uncertainty, calibration, or selective behavior. A softmax score is not confidence.

**Required correction:** use independently labeled artifact/landform cases, reliability diagrams, Brier or proper scoring rules, class-conditional calibration, and risk-coverage curves. Define an abstention threshold before release evaluation. Test geographic and acquisition shift separately.

### Medium 3: the visual and statistical acceptance protocol has no pass thresholds

The chosen diagnostics are useful, but "clear blind visual gains" and "matches class distributions" are not release criteria. A high-capacity generator can also memorize the small target corpus while passing marginal spectra and variograms.

**Required correction:** preregister site-level pass/fail thresholds, reviewer protocol, render paths, and non-inferiority margins for base preservation. Add nearest-neighbor/memorization audits, descriptor precision-and-recall or coverage, multi-seed diversity, spatial cross-correlation, and extreme-event tails. Report every result per site and regime; national averages can hide catastrophic classes.

### Medium 4: absolute coordinate features risk site memorization and analogue failure

World-coordinate noise is necessary for partition-invariant stochasticity. Absolute coordinate Fourier features fed as semantic model input are a separate choice. They can memorize acquisition sites, map seams, or Estonia-specific location shortcuts, and they do not transfer naturally to Baltic/Fennoscandian analogue data in another CRS.

**Required correction:** keep coordinate-indexed random fields separate from semantic inputs. Ablate absolute coordinates. Use physical latitude/climate or regional variables explicitly where scientifically intended, and use local relative coordinates for convolutional geometry.

### Medium 5: repository mechanisms have licensing and reproducibility boundaries

The audited MultiDiffusion, DDNM, IDM, and DEMSR revisions lack clear repository licenses. GATA has component-specific EA terms. Their papers can support independent implementation of ideas, but code must not be copied casually. Terrain Diffusion and Infinite Tensor are MIT-licensed at the audited revisions.

**Required correction:** record paper-derived versus code-derived implementation, retain commit hashes, and complete legal review before reuse. A pinned environment and output hash are good release practices, but deterministic flags alone do not guarantee cross-driver identity; the proposal correctly limits bitwise promises to a locked stack and should retain that limitation.

## 3. Unsupported-transfer ledger

| Source | What it directly establishes | What it does **not** establish for this project |
|---|---|---|
| Argudo et al. 2018 | Multimodal DEM/orthophoto feature fusion can improve a 15 m to 2 m alpine synthetic-SR task; ideal-intermediate cascades can fail | Real 1 m DTM repair, centimeter ground, Estonia transfer, canopy/water inference, or superiority of a two-x4 cascade |
| Liu et al. 2026 | Multitask semantic/image fusion improves a 2 m target under synthetic x4/x8/x16 degradation | Use of native 20 cm image detail to recover ground, target-scale accuracy, physical semantics, or low-error large-factor SR |
| GATA 2019 | A conditional GAN can learn x4 terrain themes and improve local variation | Target data/model availability, real DTM degradation, physical classes, water repair, centimeter output, or national determinism |
| DDNM 2023 | Range/null projection gives exact consistency for a known linear operator and suitable image prior | A correct Maa-amet observation operator, faulty-observation repair, terrain prior, or physical microtopography |
| MultiDiffusion 2023 | Shared-state weighted least-squares fusion reduces image crop seams | Height derivative, drainage, topology, world-partition invariance under independent seeds, or a terrain prior |
| InfiniteDiffusion 2026 | Coordinate/seed-consistent lazy windows and bounded overlap fusion are implementable | Centimeter prior, soil/geology conditioning, or quality under this project's context and hierarchy |
| GrounDiff 2026 | Learned same-grid DSM-to-DTM correction and prior-guided tiling can outperform selected baselines | Unresolved detail synthesis, cliff safety, hidden forest ground, deterministic accuracy, or DTM-to-centimeter SR |
| Moore et al. 2019 | Target-scale peat surface distributions exist and can be measured on prepared plots | National peat coverage, unprepared canopy behavior, mineral terrain, 128 m context, or all-Estonia transfer |
| Schott et al. 2024 | Multiscale process amplification can preserve stronger hydraulic coherence in supported terrain | General Estonia material morphology, patch-independent national processing, or non-hydraulic classes |

## 4. Mandatory falsification experiments

These experiments must be run before the rewritten spec selects diffusion as the production synthesizer. A failed gate rejects the relevant claim; it is not an invitation to weaken the metric after seeing the result.

### F0. Target-data sufficiency and transfer

For every proposed production regime:

1. inventory independent sites, acquisition method, point/image spacing, effective surface resolution, vertical uncertainty, ground visibility, filtered/interpolated fraction, spatial footprint, date, and condition labels;
2. estimate signal energy and acquisition-error energy separately in the 1-0.25 m and 0.25-0.0625 m bands;
3. hold out entire sites and acquisition campaigns before any model or hyperparameter choice;
4. test Estonia-to-analogue and analogue-to-Estonia transfer separately.

**Gate:** Stage B is forbidden where the target's spatial transfer function or error floor does not support the claimed band, where only one site supplies the regime, or where site-held-out morphology cannot be distinguished from acquisition artifacts.

### F1. Deterministic base reconstruction versus learned correction

Compare on held-out artifact and rare-landform sites:

- raw Maa-amet DTM;
- deterministic DTM plus ETAK/nDSM/CHM constrained reconstruction;
- learned correction alone;
- deterministic constraints plus a calibrated learned ambiguous-artifact detector.

Include water bumps, grid shores, scan seams, canopy/building leakage, small channels, true boulders, and true escarpments. At Taevaskoda, use the mapped shoreline and natural shoreline-escarpment as fixed evidence, not training labels invented from the desired render.

**Gate:** learned correction must improve residual source error beyond deterministic reconstruction without a statistically or visually meaningful increase in destroyed cliffs, boulders, channels, or banks. Any water residual after the final hard mask fails.

### F2. Corrected-parent and LOD-transition invariance

Publish a pilot using the chosen corrected authority and run the exact runtime filter/morph path.

**Gate:** LOD -2 to -1 to 0 transitions must be free of visible motion, grounding disagreement, and parent-edge seams while preserving allowed source corrections. The decoded parent relationship must be numerically documented. Passing against raw DTM by undoing a known repair is a failure.

### F3. Model-factorization bake-off

Train with identical data, conditions, physical context, and reasonable compute budgets:

- direct/multiscale finest-field model;
- proposed two-x4 cascade with perfect-intermediate training;
- proposed cascade with sampled-intermediate training;
- deterministic regressor, GAN, diffusion, dictionary, supported process, and hybrid variants.

**Gate:** the two-stage diffusion recommendation survives only if it wins blind ground-level preference and held-out morphology coverage without worse source repair, hierarchy, mode coverage, or unacceptable national cost. Argudo's cascade observation is not itself a pass.

### F4. One-meter-grid leakage

Measure on flat shore, ordinary field, forest, and smooth slope sites:

- two-dimensional power at 1 m and its harmonics;
- gradient orientation bias toward raster axes;
- residual phase relative to 1 m cell boundaries;
- shoreline curvature and stair-step counts;
- blinded top-down and ground-level renders under moving light.

**Gate:** accepted output cannot retain a visible or statistically exceptional 1 m lattice after source repair. Reducing RMSE while preserving the lattice fails.

### F5. Condition use, confounds, and boundary behavior

For soil mixture/profile, geology, hydrology, land use/event, vegetation visibility, and orthophoto:

- remove, shuffle spatially, substitute plausible counterfactuals, and perturb registration/date;
- test 1:50,000 coverage boundaries and the 1:200,000 fallback separately;
- compare full Mullastikukaart mixture/profile features against the current reduced parser;
- test canopy, water, shadow, crop rows, roads, roofs, and seasonal change explicitly.

**Gate:** conditions must improve held-out class morphology or structural placement, not merely alter appearance. Shuffled conditions must not perform similarly. No 1:10,000, 1:50,000, or 1:200,000 polygon edge may appear as fine relief. Orthophoto fusion fails if it embosses albedo/object edges or degrades masked sites.

### F6. Global stochastic field and partition invariance

Cook the same AOI with different support-window origins, overlap strides, worker counts, job orders, batch sizes, and adjacent-request sequences.

**Gate:** the locked inference stack must produce bit-identical cropped heights and packed hashes for every partition/order variant. Slope, curvature, and drainage statistics across former support boundaries must match interior controls. A global coordinate-indexed noise field is mandatory; independently seeded overlap windows fail this gate.

### F7. Realism, variance, memorization, and mode coverage

For each held-out regime:

- conduct blind randomized ground-level and diagnostic rendered comparisons;
- measure joint distributions and spatial correlations of residual height, gradient vector, curvature, local relief, morphology objects, and hydrologic context;
- report distribution precision and recall/coverage rather than only marginal spectra;
- run nearest-neighbor searches against training patches and multi-seed diversity tests;
- test ordinary and extreme sites separately.

**Gate:** output must be preferred to every fair challenger, cover held-out modes without copied training patches, and avoid generic noise, uniform roughness, rare-mode overexpression, or condition drift. Aggregate Estonia metrics cannot waive a failed regime.

### F8. Heightfield capability and abstention

Label single-valued slopes, near-vertical faces, undercuts, caves, overhangs, narrow cracks, and small clasts in the review corpus.

**Gate:** the heightfield proposal may claim only the feature classes it can represent after actual triangulation and simplification. Non-heightfield faces must be masked/abstained or handled by a separate structural representation. A steep ramp that looks unlike the reference cliff fails.

### F9. National cost and packed entropy

On accepted-quality outputs, benchmark 1 km2 and a parent-closed multi-regime pilot through inference, quantization, delta coding, deflate, manifest construction, and asset serving.

**Gate:** project total national GPU-hours, peak memory, packed bytes, object count, storage, upload, and serving cost with measured confidence intervals. The user must explicitly accept the coverage/cost plan before national production. A small smooth pilot's compression ratio cannot stand in for high-entropy national synthesis.

## 5. Required proposal revision

The ML proposal becomes a credible candidate only after these changes:

1. Rename the recommendation as a falsifiable learned-residual hypothesis until F0-F3 pass.
2. Move deterministic, typed base reconstruction before generative synthesis; learned correction is limited to ambiguous evidence and must abstain.
3. Publish the corrected ordinary LOD 0 or otherwise prove a corrected-parent hierarchy. Do not force fine detail back to known raw-DTM errors.
4. Define true world-aligned analysis/synthesis bands and one global coordinate-indexed stochastic field.
5. Replace reduced soil labels with full national soil mixtures/profiles and explicit EGT scale/coverage/fallback semantics.
6. Fetch dated RGB/CIR orthophoto for research and likely production conditioning, but gate it through visibility, object, shadow, water, registration, and temporal masks.
7. Add deterministic, process, dictionary, regime-expert, and hybrid challengers. Do not use the rejected quilt as the non-ML quality bar.
8. Restrict height output claims to single-valued terrain and separately resolve structural cliffs.
9. Produce measured national compute/storage projections before calling the architecture implementable.

## 6. Explicit verdict

**Complete two-stage ML architecture: rejected in its current form.** It has unresolved critical contradictions in data sufficiency, source-repair authority, parent hierarchy, heightfield scope, tiling, and national feasibility.

**Pixel-space conditional diffusion as an experiment: conditionally accepted.** It has a plausible quality ceiling for one-to-many residual morphology if trained on defensible target-scale data and if it beats fair deterministic, process, dictionary, GAN, and hybrid challengers under the same conditions.

**Orthophoto fetch: recommended.** It is high-value structural and semantic evidence at 20-40 cm, especially for shores, exposed ground, paths, drainage, and management traces. It is neither height truth nor sufficient supervision for the 6.25-25 cm band and must be masked or downweighted under canopy, water, shadow, objects, registration error, and temporal mismatch.

**Cheapest valid next action:** do not train the full two-stage model. First build F0 and F1 for a small, site-separated set: full-fidelity condition stacks, deterministic ETAK-constrained corrected bases, and audited target surfaces. Then run a Stage A-only bake-off that includes deterministic/process/hybrid challengers. Stage B is not funded until target-band signal is demonstrably above acquisition error and Stage A removes the observed 1 m lattice without destroying real landforms.

## 7. Primary sources checked

- Argudo, Chica, and Andujar, *Terrain Super-resolution through Aerial Imagery and Fully Convolutional Networks*, CGF 2018, DOI [`10.1111/cgf.13345`](https://doi.org/10.1111/cgf.13345), full paper Sections 3-6.
- Liu et al., *DEM Super-resolution Guided by High-resolution Remote Sensing Images Using Multitask Learning*, IJAEOG 2026, DOI [`10.1016/j.jag.2026.105099`](https://doi.org/10.1016/j.jag.2026.105099), full paper and official DEMSR repository.
- Zhao et al., *Multi-Theme Generative Adversarial Terrain Amplification*, TOG 2019, DOI [`10.1145/3355089.3356553`](https://doi.org/10.1145/3355089.3356553), full text and official GATA repository revision recorded in `ML-SOURCES.md`.
- Wang, Yu, and Zhang, *Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model*, ICLR 2023, [`arXiv:2212.00490`](https://arxiv.org/abs/2212.00490), especially the linear range/null derivation and SR operator.
- Bar-Tal et al., *MultiDiffusion*, ICML 2023, [`arXiv:2302.08113`](https://arxiv.org/abs/2302.08113), especially Equations 3-5, Algorithm 1, and limitations; official repository inspected.
- Goslin et al., *InfiniteDiffusion*, SIGGRAPH 2026, [`arXiv:2512.08309`](https://arxiv.org/abs/2512.08309), full paper and official Terrain Diffusion/Infinite Tensor repositories.
- Dhaouadi et al., *GrounDiff*, WACV 2026, [`arXiv:2511.10391`](https://arxiv.org/abs/2511.10391), full paper and supplement including failure cases and timing.
- Moore et al., *Assessing the peatland hummock-hollow classification framework using high-resolution elevation models*, Biogeosciences 2019, DOI [`10.5194/bg-16-3491-2019`](https://doi.org/10.5194/bg-16-3491-2019), full paper and Zenodo dataset `2545675`.
- Kmoch et al., *EstSoil-EH*, ESSD 2021, DOI [`10.5194/essd-13-83-2021`](https://doi.org/10.5194/essd-13-83-2021), full data-model and uncertainty discussion.
- Schott et al., *Terrain Amplification Using Multi-Scale Erosion*, TOG 2024, DOI [`10.1145/3658200`](https://doi.org/10.1145/3658200), especially comparisons and Section 6.6 limitations.
- Maa- ja Ruumiamet soil map and orthophoto documentation, and Eesti Geoloogiateenistus spatial-data packages and license, as recorded with direct official links and source hashes in `ESTONIA-CONDITIONING-DATA-AUDIT.md`.
- Official local repository revisions and license findings in `ML-SOURCES.md`; code without a license file was treated as non-reusable pending clarification.
