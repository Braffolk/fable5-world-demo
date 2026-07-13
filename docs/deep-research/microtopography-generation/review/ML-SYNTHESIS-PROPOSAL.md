# Proposal: Physically Conditioned Generative Terrain Residuals

Status: ML-track proposal for criticism, 2026-07-13. This is not implementation authorization and does not change the accepted fine-LOD transport architecture.

## Recommendation

Build a terrain-native, pixel-space, conditional residual diffusion system in `asset-gen`, trained to do two coupled jobs:

1. infer a corrected continuous ground surface from the fallible Maa-amet 1 m DTM; and
2. sample physically and geographically conditioned unresolved geometry into the 1 m to 0.25 m and 0.25 m to 0.0625 m bands.

Use two x4 stages because they exactly match the accepted fine LODs:

- **Stage A:** 1 m observation to 0.25 m corrected surface plus 0.25-1 m residual structure.
- **Stage B:** Stage A output to 0.0625 m surface plus 0.0625-0.25 m residual structure.

Stage B must always receive the original 1 m DTM, confidence/correction state, and physical conditions in addition to Stage A. It must be trained on sampled Stage A outputs, or jointly with Stage A, rather than only on perfect 0.25 m ground truth. This follows directly from the failed ideal-intermediate cascade reported by Argudo et al. 2018.

The model emits ordinary float32 height arrays at the existing fine-rung sample coordinates. Existing quantization, chunk packing, manifests, demand, streaming, decoding, and rendering remain unchanged. No runtime synthesis or provenance-specific runtime path is introduced.

## Why This Architecture

The target is intrinsically one-to-many. A 1 m observation cannot specify the locations of every pebble, root mound, erosion notch, hummock, crack, or decimeter ledge. A deterministic regressor trained with L1/L2 will average those possibilities and smooth them; a global procedural field will add variation without the correct material distribution. A conditional generative model can sample a locally plausible realization while staying tied to observed large structure and spatial conditions.

Pixel-space height diffusion is preferred over latent image diffusion because:

- every output value remains a metric elevation throughout inference;
- low/high frequency separation and observation guidance are explicit;
- no RGB VAE has to preserve centimeter-scale slopes and amplitudes;
- height, gradients, curvature, drainage, and class-conditional morphology can be trained directly;
- deterministic output can be fixed by world-coordinate seeds and pinned inference artifacts.

Diffusion is a recommendation, not an exemption from comparison. It must outperform an Argudo/Liu-style residual network and a modernized GATA-style conditional GAN on held-out ground-level visual quality, class-conditional morphology, and failures. If it does not, the cheaper model wins.

## Non-Negotiable Boundaries

- Work is isolated to synthesis and its offline model/data tooling in `asset-gen`.
- LOD 0 remains 1 m in 2,048 m chunks. LOD -1 remains 0.25 m in 512 m chunks. LOD -2 remains 0.0625 m in 128 m chunks.
- The output contract is the existing ordinary height-rung input to the current encoder.
- No binary-format, manifest, server, renderer, shader, `TerrainField`, grass, vegetation, material, or runtime changes are proposed.
- Maa-amet 1 m DTM remains the authoritative observation of ground elevation, but not an infallible surface that must be reproduced exactly.
- Maa-amet DSM must not replace the DTM. DSM/CHM may be used only as conditioning and ground-visibility evidence.
- Orthophoto is a condition, not a heightmap. Visible canopy, roofs, vehicles, water reflections, and shadows must not become ground geometry.
- The system may not use decorative hash noise, generic FBM, random per-chunk fields, or a single global roughness control.
- Taevaskoda is a demanding validation site, not a target for site-specific code, weights, masks, or handcrafted geometry.

## Statistical Problem Formulation

Let:

- `y_1m` be the observed Maa-amet DTM;
- `c` be registered conditions and their confidence;
- `g_1m` be a corrected latent 1 m-scale ground surface;
- `r_A` be the 0.25-1 m residual band;
- `r_B` be the 0.0625-0.25 m residual band;
- `h_025 = U4(g_1m) + r_A`;
- `h_00625 = U4(h_025) + r_B`.

The objective is to model:

`p(g_1m, r_A, r_B | y_1m, c)`

rather than find a single `h` such that `Downsample(h) = y_1m`.

`U4` is a smooth, documented interpolation used only to define the residual target. It is not treated as physical truth. The correction `g_1m - y_1m` is allowed where the observation confidence is low or where independent evidence supports repair.

### Why bands rather than absolute height

Absolute Estonia elevation has a huge low-frequency dynamic range irrelevant to microstructure. Residual bands let the network spend capacity on the target morphology and make amplitude calibration observable. The original DTM and corrected base remain explicit so the residual cannot silently move an entire hill.

## Inputs And Conditions

Conditions are grouped by meaning and encoded separately before multiscale fusion. Missing inputs require explicit availability masks; zero must never ambiguously mean both a valid value and no data.

### Authoritative elevation observation

- Maa-amet 1 m DTM values.
- DTM gradients, curvature, multi-radius relief, aspect, local drainage proxies, and distance to detected discontinuities.
- source tile/vintage/acquisition identifiers where available;
- void, interpolation, water, and suspected outlier masks;
- a calibrated observation-confidence field.

### Visible surface evidence

- registered current orthophoto in native useful bands, including CIR/NIR if available;
- orthophoto acquisition date, sun geometry, and seam/cloud/shadow confidence when available;
- Maa-amet CHM or nDSM as evidence for vegetation/building occlusion, never as ground height;
- land-cover and object masks, with explicit water, forest, wetland, built, road, bare ground, and agricultural states;
- image-to-DTM registration confidence.

Orthophoto acquisition is recommended. The literature shows measurable benefit at meter scales, and imagery carries shoreline, channel, exposed-rock, agricultural, path, drainage, and land-cover structure. It is still insufficient alone: under canopy and water, exact ground is unobserved; visible edges often belong to objects, shadows, or phenology.

### Physical terrain conditions

- EstSoil-EH soil class, texture, peat, drainage, and mapped uncertainty;
- surficial/quaternary deposit class and thickness where available;
- bedrock and lithology class, exposure/outcrop evidence, and map scale/confidence;
- hydrography: waterbody/channel identity, flow direction/order where available, floodplain/wetness, shore distance and side;
- landform class at several radii, not a single per-chunk label;
- land use and management where it changes morphology;
- coarse climate/wetness/freezing conditions only where supported by data.

These maps condition a probability distribution. A coarse geology polygon does not locate a specific 8 cm stone; it changes likely amplitude, anisotropy, shapes, discontinuity frequency, drainage response, and exposure pattern.

## Source Repair Is Part Of The Model

The system must not add detail on top of known bad geometry and call the result realistic. It needs an explicit correction path before or jointly with band synthesis.

### Diagnosed repair classes

- water surfaces with grid ripples, returns, interpolation streaks, or isolated bumps;
- shorelines that stair-step on the 1 m raster;
- canopy/building leakage into the DTM;
- isolated spikes, pits, stripes, scan seams, and no-data interpolation;
- oversmoothed or bridged small channels;
- real cliffs or boulders falsely detected as errors.

### Proposed correction mechanism

Use a deterministic confidence/correction head that predicts:

- corrected low-frequency ground `g_1m` on an internally oversampled support grid;
- observation confidence `w_obs` in `[0, 1]`;
- correction class probabilities for audit;
- uncertainty of the correction.

It is trained jointly with the generative bands where paired data permit, but its output remains inspectable. Independent hydrography and object masks impose typed behavior:

- connected open-water bodies use a hydrologically appropriate low-dimensional surface model, not learned random texture;
- shore geometry is reconstructed from vector/image evidence with a continuous transition, then passed as a fixed condition to the detail model;
- diagnosed artifacts may deviate from the DTM;
- high-confidence exposed ground receives strong observation guidance;
- cliffs retain a separate rare-landform confidence path so the model cannot flatten them as buildings.

This is not exact DDNM projection. The observation term is soft and spatially varying.

## Model Design

### Shared structure

Use a terrain U-Net with:

- pixel-space residual-height input/output;
- antialiased multiscale height and derivative features;
- separate encoders for continuous elevation/observation, optical imagery, categorical physical maps, and confidence/availability masks;
- feature fusion at matching physical receptive fields;
- FiLM or adaptive normalization from physical-class embeddings;
- attention only at coarse feature levels where it adds long-range context without dominating memory;
- explicit world-coordinate Fourier features at scales larger than the output pixel, never as an unsupervised noise source;
- a coordinate-seeded diffusion-noise field as the only stochastic input.

Do not initialize the height backbone from Stable Diffusion. Natural-image weights encode the wrong target statistics. Optical encoders may use pretrained remote-sensing features only after controlled ablation against training from scratch and after proving they do not leak object texture into ground height.

### Stage A

Stage A predicts the corrected 0.25 m surface and the 0.25-1 m band. Its receptive field must cover enough physical area to interpret shores, small channels, hummock fields, erosion lines, cliff context, and DTM artifact patterns. A 128-256 m support window is a reasonable experiment starting point, not a final constant.

Stage A receives the full input set. It owns most source correction because source defects are visible at 1 m and larger scales.

### Stage B

Stage B predicts only the 0.0625-0.25 m band around the sampled Stage A surface. It receives:

- sampled Stage A output and its uncertainty;
- original and corrected 1 m inputs;
- all physical conditions at their actual source resolution;
- native orthophoto features where registration and visibility support them;
- the same world-coordinate seed hierarchy.

Stage B uses smaller physical structures but must retain a large enough context halo to avoid texture-like stationarity. It must learn from real <=6 cm ground exemplars. Upsampling 25 cm truth cannot train this band.

### Joint versus staged training

Start with staged training for debugging and data attribution, then fine-tune jointly or with scheduled sampled intermediates. During Stage B training:

- early batches can use real 0.25 m targets;
- an increasing fraction must use frozen or sampled Stage A predictions;
- the original 1 m input is always retained;
- difficult corrections and class boundaries must be oversampled.

This prevents exposure bias while keeping each band measurable.

## Training Data Program

The training corpus is the highest-risk dependency. No network choice compensates for missing true centimeter ground distributions.

### Required target data

Acquire or derive ground-classified surfaces at <=3-5 cm sampling, with defensible vertical accuracy and uncertainty, from terrestrial laser scanning, low-altitude LiDAR, or SfM only where bare ground is actually visible. Include Estonia wherever possible and geomorphologically close Baltic/Fennoscandian analogs where Estonia coverage is unavailable.

Minimum physical strata for the proof corpus:

- exposed Devonian sandstone, incised riverbank, talus, and forested cliff context around Taevaskoda;
- peat mire hummock/hollow/lawn systems;
- glacial till forest floors and boulder-rich ground;
- limestone/alvar and thin-soil exposed substrate;
- coast, wetland, floodplain, and natural shore transitions;
- agricultural/mineral soil and managed drainage;
- sandy/glaciofluvial landforms and paths;
- representative ordinary forest and open ground, not only spectacular sites.

Each site needs high-resolution ground, the temporally closest DTM and orthophoto products, physical maps, acquisition metadata, and a ground-confidence mask. Spectacular sites alone would bias the national prior toward excessive relief.

### Real degradation model

For sites with paired high-resolution ground and Maa-amet products, estimate how the observed 1 m DTM differs from filtered high-resolution truth as a function of:

- local slope, curvature, discontinuity, and aspect;
- point density and acquisition geometry;
- vegetation and land cover;
- water and wetness;
- ground-classification confidence;
- interpolation/rasterization method and tile boundaries;
- acquisition vintage and registration;
- material/landform class.

Train with a mixture of:

- real aligned Maa-amet/HR pairs;
- HR truth degraded by the measured stochastic observation model;
- deliberately injected diagnosed artifacts, only when their distributions are calibrated from real pairs.

Pure bicubic, nearest, average-pool, or Butterworth degradation is allowed only as a baseline. It must not dominate training.

### Splits

Split by whole acquisition site, geology/soil regime, and acquisition campaign. Patches from one site may not leak across train and test. Maintain:

- held-out Estonia sites for release decisions;
- held-out analog regions to measure geographic transfer;
- rare-class challenge sets for cliffs, water, wet forest, and mapped boundaries;
- a no-orthophoto/poor-registration set.

Random patch splitting would yield falsely optimistic results because adjacent patches share terrain, acquisition errors, vegetation, and physical labels.

## Objectives

The training objective must balance pointwise fidelity with distributional and structural realism. Candidate terms, each justified by an ablation, are:

### Generative objective

- diffusion velocity/noise prediction on normalized residual bands;
- class-balanced sampling so common flat agricultural terrain does not erase rare but important regimes;
- conditional dropout to prevent dependence on any single map and enable missing-data inference.

### Metric ground objectives

- robust height loss on paired visible ground;
- gradient-vector loss, not only gradient magnitude;
- multi-radius curvature and local-relief losses;
- normal-angle loss at rendered-relevant scales;
- band-limited spectral and variogram distribution losses by physical class;
- joint distributions of height residual, slope, curvature, and wetness/drainage context.

### Observation and correction objectives

- confidence-weighted robust likelihood between the re-observed prediction and Maa-amet DTM;
- supervised correction/confidence losses on real paired data;
- zero or appropriately modeled observation pressure on open water and known artifacts;
- conservative correction penalty in high-confidence exposed ground;
- explicit cliff preservation and artifact-vs-landform classification losses.

### Hydro and topology objectives

- connected-water surface constraints;
- channel continuity and local drainage consistency at scales the source can support;
- penalties for synthetic pits, dams, and shore steps;
- no generic erosion loss applied to terrain classes where it is physically wrong.

Do not use RGB VGG/LPIPS loss on height as the main perceptual metric. If a learned terrain feature metric is introduced, train and validate it on human judgments of height/render pairs and publish its failure modes.

## Normalization And Amplitude

Normalize residual bands by robust, class-aware physical scales rather than globally stretching each patch. Per-patch min/max normalization would erase absolute amplitude, making a 5 cm lawn statistically identical to a 1 m boulder field.

Record in meters:

- robust residual scale by band and terrain class;
- vertical datum and registration corrections;
- clipping rates;
- actual band energy retained by quantization;
- correction amplitude and spatial extent.

The decoder must return metric meters before any existing quantization. Class maps may change the distribution; no user-facing global roughness multiplier is the feature.

## National Cook Inference

### Deterministic support windows

For each requested fine-rung region:

1. read the DTM and every condition with a physical context apron larger than the output crop;
2. derive confidence and correction features in world coordinates;
3. derive a deterministic seed from model version, national seed, stage, and integer world support-window coordinates;
4. run overlapping Stage A support windows with per-denoising-step consensus;
5. run Stage B with the Stage A result, original base, and the same condition context;
6. crop only after all neighboring support contributions are reconciled;
7. return the ordinary float height grid to the existing cook encoder.

Generate by support region rather than independently by final 128 m chunk. Chunk borders are storage boundaries, not model boundaries.

### Consensus

Use MultiDiffusion/EarthGen-style weighted fusion of predicted diffusion states or noise at every denoising step. The weight window must go to low confidence near support edges while maintaining a nonzero total weight everywhere. Final-output feathering alone is rejected because it can hide height seams while leaving slope, curvature, drainage, and repeated-pattern seams.

### Determinism contract

Pin and record:

- model architecture and checkpoint hashes;
- training-data manifest and condition-map versions;
- code commit and container/environment hash;
- sampler, step schedule, numerical precision, and random-seed scheme;
- inference backend/device class and deterministic-algorithm flags;
- output asset hashes.

Bitwise cross-hardware identity is promised only for a locked validated inference stack. Released packed assets and their hashes are the authority. The browser remains uninvolved.

## Mandatory Challengers

The first experiment must compare models on the same data and cook adapter:

- **Interpolation/current synthesis baseline:** establishes whether learned output is genuinely better than what exists.
- **Residual FCN:** modern Argudo/Liu-style deterministic multimodal predictor with the same conditions.
- **Conditional GAN:** modernized GATA-style band generator with typed physical embeddings.
- **Conditional diffusion:** the proposed pixel-space band model.

Do not compare a highly tuned diffusion model against intentionally weak baselines. Give each challenger a reasonable parameter/compute budget and evaluate generated samples fairly.

## Acceptance Protocol

### Visual acceptance

Use blind, randomized rendered comparisons at ground level and oblique/top-down diagnostic views. Reviewers must not know the method. Each physical class must be represented. Failures include:

- detail that reads as noise, sine waves, patch texture, or uniform bumpiness;
- 1 m grid shorelines or water bumps;
- repeated features or support-window boundaries;
- material-inappropriate shapes or amplitudes;
- cliffs smoothed as errors or vegetation/object edges embossed as ground;
- flat ordinary terrain made gratuitously rough;
- visible disagreement with the existing materials and vegetation context.

Taevaskoda review uses the real reference only to identify general sandstone/riverbank/forest-floor deficiencies. Any fix must improve held-out terrain of the same class elsewhere.

### Measured held-out acceptance

Report, by physical class and acquisition site:

- robust height, slope, normal, curvature, and local-relief errors against HR truth;
- residual-band power spectra and directional variograms;
- distributions and spatial correlation of mounds, hollows, steps, clasts, rills, and exposed-rock breaks where labels exist;
- hydrologic connectivity, added pits/dams, water planarity, and shoreline smoothness;
- correction precision/recall and correction amplitude for diagnosed source artifacts;
- seam metrics across model support windows and final chunk boundaries;
- sample diversity without condition drift;
- condition sensitivity and counterfactual tests.

### Counterfactual condition tests

At fixed DTM and random seed, change one physically meaningful condition within plausible bounds. The output distribution should change in the expected way without moving protected macrogeometry. Examples:

- peat versus mineral soil changes hummock/hollow morphology;
- till/boulder-rich versus fine sediment changes clast/roughness distribution;
- exposed sandstone versus covered forest soil changes ledge/fracture expression;
- water/shore masks suppress water texture and establish a coherent margin.

This is not a license to render a lookup-table texture. It tests whether conditions actually matter instead of being ignored by the network.

## Cheapest Proof Before Full National Training

The cheapest credible proof is a data-and-model feasibility study, not another full Estonia cook.

### Proof 0: data audit

Before training the production model, assemble and inspect aligned HR ground/base DTM/orthophoto/condition stacks for at least five distinct regimes: Taevaskoda-like sandstone riverbank, peat mire, glacial-till forest, limestone/alvar, and coast/wetland. Quantify actual source-error distributions and determine whether <=6 cm ground is sufficiently observed beneath vegetation.

Stop if the HR target is mostly interpolation, canopy leakage, or temporally mismatched change. A network trained on false ground will reproduce it convincingly.

### Proof 1: Stage A bake-off

Train only 1 m to 0.25 m Stage A on a compact but site-separated corpus. Compare residual FCN, conditional GAN, and conditional diffusion. Include real-pair and synthetic-degradation ablations. Judge held-out rendered output, source repair, shore/water behavior, and class morphology.

Stage A must show clear, blind visual gains and correct at least the measured 1 m grid/source artifacts before funding Stage B.

### Proof 2: Stage B truth test

Train 0.25 m to 0.0625 m only for regimes with genuine <=3-5 cm ground truth. Demonstrate that the added band matches held-out class distributions and is visibly material-specific. If no defensible HR truth exists for a class, do not fabricate confidence; leave that class at Stage A until data are acquired.

### Proof 3: frozen-interface pilot

Run the chosen model through the existing synthesis hook and existing encoder for a small parent-closed Taevaskoda-centered pilot plus several ordinary held-out sites. The only integration result should be different fine height payload contents. Existing runtime acceptance then checks the exact URL, trees/materials/grass agreement, seams, and actual ground-level result.

## Rejected Approaches

- **Exact DDNM projection:** rejects required source correction and assumes the wrong known degradation.
- **Stable Diffusion plus ControlNet on orthophoto:** height is not RGB; the pretrained prior and VAE are inappropriate, and imagery exposes objects rather than hidden ground.
- **Orthophoto displacement:** visible texture, canopy, roofs, shadows, and water are not ground elevation.
- **One national conditional mean regressor:** likely oversmooths the unresolved multimodal band.
- **GATA unchanged:** important baseline, but fixed old architecture, synthetic low-pass degradation, no published target checkpoint/data, and no explicit physical conditions.
- **Published terrain-diffusion weights:** 30-90 m geomorphology is not 6 cm material morphology.
- **Independent per-chunk synthesis plus final blending:** storage-aligned patterns and derivative discontinuities remain.
- **Global noise/roughness/FBM:** cannot express material-specific morphology or repair source observations.
- **Runtime procedural detail:** violates the zero-browser-synthesis law.

## Principal Risks

### Data insufficiency

The highest risk is lack of representative, accurately classified <=6 cm ground. Mitigation is to make the HR-ground audit the first milestone and allow class-specific rollout rather than train on fabricated truth.

### Condition-map scale mismatch

National soil and geology polygons may be much coarser than the output. They may condition distributions but cannot place exact features. Preserve map confidence and test whether the model creates polygon-edge seams.

### False image-to-ground transfer

Orthophoto edges can be canopy, building, shadow, crop rows, reflections, or temporal change. Mitigation is typed visibility/object masks, condition dropout, real held-out tests under canopy/water, and conservative image fusion.

### Rare-landform destruction

Correction models may flatten cliffs, outcrops, boulders, and bank failures as artifacts. Mitigation is rare-class data, explicit uncertainty, multi-radius context, cliff challenge sets, and no correction without confidence evidence.

### Attractive but incorrect hallucination

A generative model can look detailed while violating local material or hydrology. Mitigation is blind class review, HR truth, typed morphology statistics, counterfactual tests, hydro constraints, and challenger comparisons.

### Cook cost

Diffusion can be substantially slower than GAN/CNN inference. Measure quality per cooked square kilometer and sampler-step ablations. Distill or adopt a faster sampler only after the full model defines an accepted quality target.

## Decision Gates

1. **Data gate:** enough defensible HR ground exists for the five proof regimes, including at least one site-separated holdout per regime.
2. **Observation gate:** a measured real degradation model improves held-out correction over bicubic/average-pool training.
3. **Stage A gate:** one learned method visibly and statistically beats interpolation/current synthesis without flattening rare terrain.
4. **Stage B gate:** real <=6 cm targets show material-specific added structure rather than generic noise.
5. **Generalization gate:** Taevaskoda-class improvements repeat on held-out sandstone/riverbank terrain and do not damage ordinary Estonia.
6. **Interface gate:** the chosen synthesis model returns ordinary height arrays through the frozen cook boundary with no format/runtime change.

The current recommendation passes only the literature gate. It does not pass the data or model-quality gates until the proof program is run.
