# ML Literature Audit: Cook-Side Terrain Detail Synthesis

Status: primary-source audit, 2026-07-13.

## Audit Scope

This audit asks one narrow question: which learned methods can support offline generation of credible 0.25 m and 0.0625 m terrain heights from Estonia's fallible 1 m DTM, conditioned by observable surface evidence and the physical terrain class? Existing LOD, binary packing, serving, decoding, and runtime rendering are fixed. The learned system must emit ordinary cooked height samples before the existing quantization and packing step.

The visual objective is not exact reconstruction of the 1 m input. It is plausible added geometry and correction of source artifacts, with distributions that change appropriately with landform, hydrology, soil, surficial deposit, and substrate. Exact downsampling, seam freedom, determinism, and quantization are constraints on that result, not the objective.

### Evidence labels

- **Direct evidence:** an experiment, equation, dataset fact, limitation, or released-code fact stated in the cited paper or repository.
- **Transfer:** a demonstrated mechanism that could be reused, but was not demonstrated on Estonia or at the target scale.
- **Inference:** a proposed implication for this project. It must be validated here.

## Decision Summary

No audited paper or released model solves the required problem end to end. In particular, none demonstrates a real 1 m DTM to 0.0625 m ground-surface transformation conditioned by orthophoto, soil, and geology. The literature does establish useful components:

1. Terrain-specific learned priors can add coherent terrain structure, but published terrain generators stop at roughly 2-5 m outputs or use synthetic degradations.
2. Registered orthophoto can improve DEM inference, but it is an indirect covariate, is unreliable beneath vegetation and water, and has not been shown to determine centimeter ground geometry.
3. Real low/high acquisition pairs materially change the problem; synthetic bicubic or low-pass degradation does not represent sensor, classification, interpolation, canopy leakage, water, or rasterization errors.
4. Diffusion supports conditional multimodal distributions and iterative overlap consensus. It does not supply the terrain prior, the observation model, or physical correctness on its own.
5. Hard inverse-problem projection is inappropriate where the source observation is known to contain errors. Confidence-weighted observation guidance is the defensible transfer.

The resulting recommendation is a terrain-native, pixel-space, conditional residual diffusion model trained on real high-resolution ground exemplars and a measured Maa-amet degradation model, with an inexpensive residual-CNN and GATA-style GAN as mandatory challengers. See `ML-SYNTHESIS-PROPOSAL.md`.

## Terrain-Specific Learned Methods

### Argudo, Chica, and Andujar 2018: orthophoto-guided FCN

Paper: *Terrain Super-resolution through Aerial Imagery and Fully Convolutional Networks*, Computer Graphics Forum 37(2), DOI `10.1111/cgf.13345`.

- **Direct evidence, pp. 1 and 3-4, Sections 3 and 4.1:** the model maps a 15 m DEM plus a 1 m orthophoto to a 2 m DEM. Separate multiresolution RGB and DEM feature pyramids are concatenated and predict a height residual.
- **Direct evidence, Section 4.1:** training uses alpine regions in the Pyrenees and Tyrol. The nominal 2 m target DEM is downsampled synthetically to 15 m; the 1 m orthophoto is registered to it. The study reports about 22,000 training and 11,000 validation tiles and holds out whole regions for testing.
- **Direct evidence, Section 4.2:** the network is initialized from an image segmentation FCN and fine-tuned jointly; training is reported as about 60,000 iterations and 16.5 hours.
- **Direct evidence, Sections 5-6:** the combined model reports roughly 0.75 m RMSE versus roughly 0.98 m without RGB. The resulting effective geometric accuracy is closer to an interpolated 7-10 m DEM than to the nominal 2 m sample pitch.
- **Direct evidence, Section 5:** a 30 m to 15 m to 2 m cascade failed because the generated intermediate DEM did not have the correlations with the real orthophoto that the second model learned from true data.
- **Direct evidence, discussion:** vegetation prevents reliable inference of bare ground from imagery; shadows can create false height discontinuities. The paper explicitly distinguishes DTM reconstruction from visible DSM objects.
- **Direct evidence, official repository:** `fcn-terrains` includes Caffe/Python 3.5 network definitions, trained-model support, preprocessing scripts, tile lists, and render scripts. It is useful as a reproducible architecture baseline, not modern production code.
- **Transfer:** multimodal feature pyramids and residual height prediction are a strong deterministic baseline. Orthophoto should enter as learned context, not be converted directly into displacement.
- **Inference:** a two-stage 1 m to 0.25 m to 0.0625 m cascade must train stage two on stage-one outputs, or train both stages jointly, while retaining the original 1 m observation. Training stage two only on ideal 0.25 m truth would reproduce the paper's cascade failure.
- **Boundary:** this paper does not demonstrate sub-meter output, real DTM degradation, source-error correction, or soil/geology conditioning.

Citation correction: this is not Argudo et al. 2017, *Coherent Multi-Layer Landscape Synthesis*, DOI `10.1007/s00371-017-1393-6`. The 2017 work is an exemplar/dictionary landscape synthesis method. Any report citing the 2017 DOI as the orthophoto FCN is wrong.

### Liu et al. 2026: GSRMTL / GDEMSR

Paper: *DEM Super-resolution Guided by High-resolution Remote Sensing Images Using Multitask Learning*, International Journal of Applied Earth Observation and Geoinformation 146:105099, DOI `10.1016/j.jag.2026.105099`.

- **Direct evidence, Section 4.1, p. 5:** data come from three Ontario projects. The source products are 20 cm orthophotos and 2 m DEMs. Orthophotos are downsampled to 500 x 500 and cropped to 496 x 496 to align with the 2 m DEM grid.
- **Direct evidence, Section 4.1:** 1,409 pairs are used; DRAPE and SWOOP supply 1,154 training pairs and geographically separate SCOOP supplies 255 test pairs.
- **Direct evidence, official repository, Ontario data loader:** the low-resolution DEM is created by bicubic downsampling of the 2 m target at factors x4, x8, and x16. Therefore the tasks reconstruct 2 m from synthetic 8 m, 16 m, and 32 m inputs. The native 20 cm image detail is not preserved as a sub-2 m target cue.
- **Direct evidence, Sections 3-4:** the model uses dual image branches, an auxiliary semantic-segmentation task, attention, and cross-modal feature fusion. The reported training objective is predominantly L1 height loss plus cross-entropy segmentation loss.
- **Direct evidence, Section 4.2:** pseudo-labels are produced by a pretrained UperNet; training uses an RTX 3090, 256-pixel crops, batch size two, 200 epochs, and about 38 hours.
- **Direct evidence, Table 1:** reported RMSE is approximately 1.63 m, 2.23 m, and 3.27 m for x4, x8, and x16 respectively, with favorable slope and aspect metrics relative to the compared models.
- **Direct evidence, ablation/discussion:** semantic segmentation reaches only about 10.5 percent mIoU at x4 because the pseudo-labeler is domain-shifted. The auxiliary branch yields modest rather than transformative DEM gains.
- **Direct evidence, official repository:** commit `7bd0c75d1c4e0e00b9cc74214409af7043db0051` contains project code but no released dataset, checkpoint, or license. The public repository is not a drop-in model artifact.
- **Transfer:** modern multiscale multimodal fusion is a more current baseline than the 2018 FCN. Auxiliary semantics are useful only when labels come from authoritative Estonia maps or a validated local classifier.
- **Inference:** feeding soil, geology, hydrology, land cover, and orthophoto through separate typed encoders is better founded than asking one RGB backbone to discover all terrain classes.
- **Boundary:** despite the title, this experiment does not reconstruct centimeter geometry from 20 cm imagery. It reconstructs a 2 m DEM from synthetically coarsened DEMs, while the imagery is resampled onto the 2 m grid.

### Zhao et al. 2019: GATA

Paper: *Multi-Theme Generative Adversarial Terrain Amplification*, ACM Transactions on Graphics 38(6), DOI `10.1145/3355089.3356553`.

- **Direct evidence, Sections 3-5:** GATA is a conditional adversarial x4 amplifier operating on 256 x 256 terrain patches. It learns a 1,024-dimensional theme embedding and includes a generator, discriminator, and theme encoder.
- **Direct evidence, loss definition:** training combines adversarial, L1, feature-matching, and embedding-consistency losses. The paper reports batch size one, Adam at `2e-4`, and about 1.2 million training steps.
- **Direct evidence, Appendix A.1:** the source terrain is Oregon DOGAMI LiDAR, normalized to 2 m. Approximately 3,000 square kilometers are organized as 1,024 x 1,024 regions and 786 nearby regions supply themes before subdivision into 256-pixel patches.
- **Direct evidence, official repository:** TensorFlow 1.10/Python 3.6 experimental code implements the theme encoder and overlap assembly. It publishes no usable checkpoint or raw training corpus. Preprocessing creates low-resolution inputs using a Butterworth low-pass filter at the same raster resolution, so the degradation is synthetic.
- **Direct evidence, repository license:** the repository contains a multi-component Electronic Arts license; it must not be described casually as MIT.
- **Transfer:** theme embeddings demonstrate that a learned terrain amplifier can vary local detail coherently rather than apply one global roughness function. The encoder can infer a theme from an exemplar outside the training theme set.
- **Inference:** a typed physical-condition embedding should replace a single opaque national theme. It should include categorical and continuous maps for soil, surficial deposit, bedrock/lithology, hydrologic regime, land use, and local landform, plus uncertainty.
- **Boundary:** GATA does not demonstrate real Maa-amet degradation, centimeter-scale output, explicit geology/soil semantics, water correction, or orthophoto guidance. Its fixed style count and old framework make it a challenger, not the proposed production architecture.

### Demiray et al. 2021: D-SRGAN

Paper: *Deep Learning-Based Super-Resolution for Digital Elevation Models*, arXiv `2004.04788`.

- **Direct evidence, data section:** the study uses real North Carolina Floodplain Mapping products delivered at roughly 3 ft and 50 ft resolution. It reports about 590 square kilometers for training and 142 square kilometers for testing.
- **Direct evidence:** high-resolution 1,600 x 1,600 areas and low-resolution 100 x 100 areas are split into 400 x 400 and 25 x 25 patches.
- **Direct evidence, internal inconsistency:** the paper sometimes calls the task x4, but 25 to 400 samples is x16 linear and 50 ft to 3 ft is approximately x16.7. Project decisions must use the actual sampling ratio, not the prose label.
- **Direct evidence, training:** the adversarial loss is combined with pixel MSE at a small adversarial weight, and discriminator-freezing schedules are used over 2,000 epochs.
- **Direct evidence, results:** the reported test MSE is about 0.861 m versus 0.946 m for bicubic and 1.124 m for bilinear interpolation.
- **Direct evidence, discussion:** the network struggles with fine detail, remains oversmoothed, and performs better in flatter terrain than steep terrain.
- **Transfer:** this is important because it uses delivered low/high products rather than generating the entire low-resolution side by bicubic downsampling. It supports measuring real acquisition degradation.
- **Boundary:** the visual quality and conditioning are insufficient for the target. The study does not establish centimeter morphology or Estonia transfer.

### Panangian and Noori 2024: Real-GDSR

Paper: *Real-GDSR: Real-World Guided DSM Super-Resolution*, ISPRS Annals X-2-2024, DOI `10.5194/isprs-annals-X-2-2024-185-2024`.

- **Direct evidence, Section 4.1:** the real paired task uses a 5 m Cartosat-1 DSM, high-resolution Swiss DSM, and registered RGB imagery. The data are urban/built-area DSM, not bare-earth DTM.
- **Direct evidence:** roughly 2,200 256 x 256 patches are reported, with 2,000 for training and 200 for testing.
- **Direct evidence, method:** a local residual refinement network is followed by a learned anisotropic PDE diffusion stage guided by image edges. This `diffusion` is deterministic smoothing, not a generative denoising diffusion model.
- **Direct evidence, method discussion:** the authors deliberately remove strict adjustment to the low-resolution input because the real coarse source lacks information and strict adherence can impede reconstruction.
- **Direct evidence, results:** reported RMSE is about 3.54 m versus 4.10 m for D-SRGAN and 5.59 m for bicubic interpolation. The smoothing stage reduces RMSE but can worsen robust errors and remove intricate details.
- **Transfer:** real-pair training and freedom to correct the observation are directly relevant. Learned edge-aware regularization may help source repair, but cannot be allowed to erase material-specific microgeometry.
- **Boundary:** the method is demonstrated on building-rich DSM at meter-scale errors. Image edges there often correspond to roofs and walls, the opposite of a bare-earth DTM under vegetation.

### Lochner et al. 2023: terrain diffusion

Paper: *Interactive Authoring of Terrain using Diffusion Models*, Computer Graphics Forum, DOI `10.1111/cgf.14941`.

- **Direct evidence, pp. 3-5:** training data combine USGS elevation and registered satellite imagery. Candidate terrain tiles are filtered by learned classifiers to remove recording errors and artificial features; the satellite imagery is used for filtering rather than as a height condition.
- **Direct evidence:** the cascade maps approximately 153 m to 19.1 m to 2.39 m per pixel, two x8 stages and x64 overall. Tiles are 256 x 256 and the final clean terrain corpus is reported as millions of tiles.
- **Direct evidence, Section 4:** super-resolution training uses self-supervised nearest-neighbor x8 downsampling of clean high-resolution terrain, not real paired acquisition products.
- **Direct evidence, architecture/training:** each stage is a time-conditioned U-Net. Terrain style is represented by slope/elevation histograms. Each model is reported as about 72 hours on an A100 with 20 GB memory and batch size eight; sampling uses hundreds of DDIM steps depending on the stage.
- **Direct evidence, evaluation:** a 41-person study evaluates generated terrain preference, not sub-meter super-resolution fidelity. Hydrologic plausibility is assessed with breach volume.
- **Direct evidence, limitations:** explicit elevation constraints can fail to align. An attempted elevation-to-satellite translation is reported as noisy and incoherent.
- **Transfer:** terrain-native pixel diffusion is credible and supports multimodal output. Style conditioning and terrain-domain data filtering are useful.
- **Inference:** train on residual/Laplacian bands around an authoritative base rather than unconditional absolute heights; add explicit, validated spatial conditions instead of a coarse global style histogram.
- **Boundary:** this system stops at 2.39 m, uses synthetic degradation, and provides no evidence for raw 1 m error repair, orthophoto-conditioned height, or centimeter detail.

### Perche et al. 2023: StyleDEM

Paper: *StyleDEM: Generating High-Resolution Terrain Models using StyleGAN*, arXiv `2304.09626`.

- **Direct evidence, data/training:** separate StyleGAN2 models are trained on nominal 5 m French IGN RGE ALTI and 30 m SRTM data at 1,024 x 1,024. Training is reported on four V100 16 GB GPUs for roughly 35 and 75 hours.
- **Direct evidence, method:** a learned encoder maps terrain into the generator latent space. Large areas use overlapping generation, histogram retargeting, and a minimum-error boundary cut.
- **Direct evidence:** a 30 m to 5 m cascade and a x3 super-resolution example are shown.
- **Direct evidence, limitations:** each resolution requires a specific model; input features smaller than about 50 pixels can be ignored; unseen or mixed classes fail; class imbalance affects results.
- **Transfer:** learned style-space control and a terrain-specific generator are relevant challenger ideas.
- **Boundary:** minimum-error cuts on completed tiles are weaker than per-step consensus for height derivatives. StyleGAN image losses and VGG-style perceptual features are not validated for metric height or drainage.

### Goslin et al. 2026: InfiniteDiffusion and terrain-diffusion

Paper: *InfiniteDiffusion: Procedural Terrain Generation with Infinite Tensor Diffusion*, arXiv `2512.08309`; official repositories `terrain-diffusion` and `infinite-tensor`.

- **Direct evidence, Sections 3.2-3.5:** an unbounded integer-coordinate tensor is generated lazily in local windows. Overlapping predictions are accumulated with weights; recursive caches and truncated fusion limit work. Seeds and coordinates make results independent of query order.
- **Direct evidence, Section 3.4:** a small fusion depth approaches the reported MultiDiffusion quality, while deeper fusion eliminates tested boundary artifacts at higher cost.
- **Direct evidence, Section 5:** terrain height is transformed with a signed square root and encoded in low/high Laplacian components. After decoding, the low-frequency component is re-extracted to prevent high-frequency synthesis from drifting the base.
- **Direct evidence, data/scale:** MERIT and ETOPO/climate data support a hierarchy around 23 km, 90 m, and a released 30 m model. Soil and satellite conditions are explicitly future work.
- **Direct evidence, compute:** training is reported as roughly two weeks on a 3090 Ti or one week on a 5090 within 24 GB.
- **Direct evidence, repositories:** `terrain-diffusion` commit `82a0431281f21a6ec3d691a12ee61525de5b0790` and `infinite-tensor` commit `070ca95b2ed7122740e35176113774553d9a67c0` are MIT-licensed.
- **Transfer:** world-coordinate seeds, query-order independence, lazy overlapping support windows, recursive context, and Laplacian band separation transfer directly to deterministic national cooking.
- **Inference:** the accepted fixed LOD grid does not need an infinite runtime tensor. The cook can use the same coordinate-seeded support-window inference to produce normal LOD -1/-2 chunks.
- **Boundary:** the released prior is in the wrong domain and is hundreds to thousands of times coarser than the target. It must not be fine-tuned as if 30-90 m landforms were 6 cm material structure.

### Dhaouadi et al. 2026: GrounDiff

Paper: *GrounDiff: Diffusion-Based Ground Surface Generation from Digital Surface Models*, WACV 2026, arXiv `2511.10391`.

- **Direct evidence, Sections 3.1-3.3:** the task is same-grid DSM-to-DTM translation. The conditional diffusion model predicts a residual and a ground-confidence map; confident ground cells are gated through while occluded cells are reconstructed.
- **Direct evidence, objective:** height L1/L2, gradient-magnitude, and confidence BCE losses are combined.
- **Direct evidence, scalable inference:** `PrioStitch` first estimates a low-resolution global prior, then initializes overlapping high-resolution patches from that prior and blends them spatially.
- **Direct evidence, data:** evaluated datasets include nominal 0.1 m DALES/New Brunswick and 1 m USGS surfaces. Targets beneath objects are partly TIN/interpolation products rather than direct ground observations.
- **Direct evidence, limitations:** abrupt cliffs can be mistaken for structures and smoothed; out-of-distribution rocky mountain terrain performs badly; dense vegetation without ground reference remains difficult. The supplement notes artifacts in interpolated ground truth.
- **Transfer:** explicit observation confidence, correction gating, gradient losses, a global prior, and prior-guided patch inference directly address source defects and large-area coherence.
- **Inference:** confidence must come from acquisition/land-cover/hydro diagnostics and be supervised on held-out Estonia-like data. It must not become a learned excuse to flatten rare cliffs.
- **Boundary:** GrounDiff repairs an observed surface at the same grid; it does not generate new 6 cm morphology. Its cliff failure is especially relevant to Taevaskoda.

### Higo et al. 2025: TerraFusion

Paper: *TerraFusion: Joint Generation of Terrain Geometry and Appearance*, arXiv `2505.04050`.

- **Direct evidence:** the system jointly models height and top-down texture derived from NASADEM and Sentinel-2 at about 25 m samples in a latent generative model.
- **Direct evidence, discussion:** some heightmaps remain unnatural, rivers can be absent, and the resolution is insufficient for practical fine real-world terrain reconstruction.
- **Transfer:** geometry and appearance should share conditions and be checked for contradiction.
- **Boundary:** its weights, latent representation, scale, and objectives are not suitable for metric centimeter height. Jointly generating final imagery is outside the frozen asset-gen height scope.

## General Generative and Inverse-Problem Components

### Wang, Yu, and Zhang 2023: DDNM

Paper: *Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model*, ICLR 2023, arXiv `2212.00490`.

- **Direct evidence, Sections 3-4:** for a known linear observation `y = A x + n`, DDNM decomposes a denoised estimate into a measured range component and an unconstrained null-space component. In the noiseless case it uses `A_dagger y + (I - A_dagger A) x`.
- **Direct evidence, super-resolution experiment:** `A` is synthetic average pooling and `A_dagger` is replication. Experiments use natural-image denoisers on ImageNet and CelebA.
- **Direct evidence:** DDNM+ relaxes the formulation for noise and imperfect measurements.
- **Transfer:** the range/null-space framing is useful for understanding which frequencies are observed. A soft, spatially varying likelihood can preserve trusted large-scale ground while allowing repair elsewhere.
- **Inference:** exact DDNM projection is wrong for this project. Maa-amet's DTM is not known to be a block average of a hidden 6.25 cm surface, and exact projection would preserve water, grid, interpolation, and classification errors.
- **Boundary:** DDNM is neither a terrain prior nor a learned degradation model.

### Bar-Tal et al. 2023: MultiDiffusion

Paper: *MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation*, ICML 2023, arXiv `2302.08113`.

- **Direct evidence, Section 3:** at every denoising step, overlapping crop predictions are reconciled with a global least-squares objective. For simple crop operators, the solution is a weighted average of the overlapping predicted values.
- **Direct evidence, experiments:** panoramas are generated with Stable Diffusion latent crops and dense overlap. The method requires no retraining of the base model.
- **Transfer:** fuse noise or denoised-height predictions during every step in overlapping world-coordinate windows. Do not generate independent final chunks and feather them afterward.
- **Boundary:** overlap consensus inherits all errors and biases of the base prior. It does not enforce terrain gradients, drainage, or physical material behavior by itself.

### Zhang, Rao, and Agrawala 2023: ControlNet

Paper: *Adding Conditional Control to Text-to-Image Diffusion Models*, ICCV 2023, arXiv `2302.05543`.

- **Direct evidence, Sections 3-4:** the pretrained backbone is frozen; trainable cloned blocks receive conditions through zero-initialized 1 x 1 convolutions, avoiding destructive early updates.
- **Direct evidence:** the paper reports about 23 percent additional GPU memory and 34 percent additional training time relative to Stable Diffusion.
- **Direct evidence:** the condition encoder maps a 512-pixel condition to a 64-pixel latent through convolutional blocks; training uses the standard diffusion noise loss.
- **Direct evidence:** examples rely on a vast pretrained image prior. One comparison contrasts a depth model trained on more than 12 million images and A100-scale infrastructure with a ControlNet trained on about 200,000 examples on one 3090 Ti for five days.
- **Transfer:** zero-initialized side encoders can add typed conditions after a strong terrain prior exists.
- **Inference:** do not make image Stable Diffusion the height prior. Terrain conditions should be integrated natively or attached to a terrain-pretrained backbone, and every output must remain metric height.

### Sharma et al. 2024: EarthGen

Paper: *EarthGen: Generating the World from Top-Down Views*, arXiv `2409.01491`.

- **Direct evidence, Sections 3-4:** EarthGen generates RGB top-down imagery, not elevation. It uses separate x4 latent diffusion super-resolution stages across map scales.
- **Direct evidence, tiling:** `Mixture of Diffusers` combines overlapping noise predictions with Gaussian weights at each denoising step; final VAE decodes are also blended.
- **Direct evidence, data:** training uses tens of thousands of Bing map tiles, with the finest RGB scale around 15 cm.
- **Direct evidence, objective:** VAE MSE, KL, and LPIPS losses and Stable Diffusion-derived upscaling are image-domain objectives. Fine features may be hallucinated.
- **Transfer:** two x4 stages match the existing 1 m to 0.25 m to 0.0625 m hierarchy exactly. Overlap should be resolved during denoising, not only after decoding.
- **Boundary:** RGB fidelity, LPIPS, and VAE reconstruction are not evidence for metric height. The model cannot be reused as a DEM generator.

### Gao et al. 2023: Implicit Diffusion Models

Paper: *Implicit Diffusion Models for Continuous Super-Resolution*, arXiv `2303.16491`.

- **Direct evidence:** a coordinate-conditioned implicit decoder enables continuous-scale image super-resolution from a diffusion feature hierarchy. Training and evaluation are on natural images/faces, including fixed-factor supervision and arbitrary-scale queries.
- **Transfer:** implicit coordinates can remove a hard output resolution dependency.
- **Boundary:** the project already has fixed 0.25 m and 0.0625 m grids, so continuous output is unnecessary complexity. The method provides no metric terrain or acquisition evidence.

### Saharia et al. 2021, Li et al. 2021, and Song et al. 2024

Papers: *Image Super-Resolution via Iterative Refinement* (SR3, arXiv `2104.07636`), *SRDiff* (arXiv `2104.14951`), and *ReSample* (arXiv `2307.08123`).

- **Direct evidence:** these works demonstrate conditional diffusion or inverse-problem sampling for natural-image super-resolution. SR3 establishes direct conditional iterative refinement; SRDiff predicts residual image detail; ReSample alternates latent-prior updates and data-consistency steps.
- **Transfer:** residual diffusion and soft data guidance are architecturally relevant.
- **Boundary:** their pretrained weights, perceptual metrics, degradations, and RGB priors do not transfer to height. They are component references, not evidence that diffusion beats a terrain GAN or residual CNN here.

### Kawar et al. 2022 and Chung et al. 2023

Papers: *Denoising Diffusion Restoration Models* (DDRM, arXiv `2201.11793`) and *Diffusion Posterior Sampling for General Noisy Inverse Problems* (DPS, arXiv `2209.14687`).

- **Direct evidence:** DDRM uses an SVD-compatible known linear degradation; DPS adds likelihood-gradient guidance for more general noisy inverse problems.
- **Transfer:** DPS-style likelihood guidance is conceptually closer than exact projection when the observation is uncertain.
- **Inference:** a learned, spatially varying Maa-amet observation likelihood is preferable to a single global `A`. The guidance strength should be high only for trustworthy ground cells and low or zero for water and diagnosed artifacts.
- **Boundary:** neither supplies the terrain prior or validates the observation likelihood.

### Blau and Michaeli 2018

Paper: *The Perception-Distortion Tradeoff*, CVPR 2018, arXiv `1711.06077`.

- **Direct evidence:** estimators optimized only for distortion cannot simultaneously minimize perceptual distribution distance; improving perceptual quality can increase pointwise error.
- **Transfer:** RMSE and exact downsampling cannot be the sole target when the desired result is plausible unresolved detail.
- **Boundary:** this does not excuse arbitrary geometry. Metric, hydrologic, class-conditional, seam, and visual constraints remain mandatory.

### OFTSR and other recent RGB SR methods

Paper: Wang et al. 2024, *OFTSR*, arXiv `2412.09465`, plus natural-image flow/GAN/implicit SR references in the local library.

- **Direct evidence:** these methods improve natural-image texture fidelity with optical-flow, transformer, flow, or adversarial priors under image-domain benchmarks.
- **Transfer:** motion/feature alignment, frequency decomposition, and coarse-to-fine training can inspire modules.
- **Boundary:** they do not establish ground-elevation correctness. Importing a visually successful RGB network without terrain data and terrain losses would optimize the wrong distribution.

## High-Resolution Ground Evidence

### Noumonvi et al. 2025: HuHoLa

Paper: *HuHoLa: A Novel Hummock-Hollow-Lawn Mire Microtopography Modelling Approach*, Ecological Modelling 2025, DOI `10.1016/j.ecolmodel.2025.111212`.

- **Direct evidence:** the study uses centimeter-scale UAV/SfM and LiDAR products at a Swedish mire. Reported morphological ranges include hummocks roughly 20-50 cm and lawns roughly 5-20 cm relative to local wetness structure.
- **Direct evidence:** HuHoLa classifies existing microtopography using DEM and hydrologic proxies; it is not a generative synthesis algorithm.
- **Transfer:** it provides peatland class labels, scale ranges, and potential validation targets. It also shows that morphology statistics change with DEM resolution.
- **Boundary:** one mire cannot define a national peat prior. The ground surface and vegetation/SfM separation must be audited before using it as training truth.

### Lapinjarvi terrestrial laser scanning datasets

Papers/data: Pitkanen et al. 2019 and Luke's 2019 Lapinjarvi TLS data description.

- **Direct evidence:** 18 Finnish forest plots were scanned with a Leica P40 at millimeter-scale point spacing, with reported registration errors around 1-3 mm. The public point clouds are CC BY 4.0.
- **Direct evidence:** the data were collected for tree/stem research. Forest-floor visibility is occluded and existing processing defines ground from low points rather than publishing a ready, authoritative centimeter DTM.
- **Transfer:** these plots are promising raw exemplars for forest-floor morphology after rigorous ground classification and uncertainty labeling.
- **Boundary:** point spacing is not equivalent to ground accuracy or complete ground coverage. They cannot be treated as clean height targets without a separate validation pass.

## What The Literature Does Not Establish

The following claims would exceed the audited evidence:

- A 1 m DTM and 20 cm orthophoto uniquely determine a 6.25 cm bare-earth surface.
- A natural-image diffusion model contains a reusable prior for Estonia ground materials.
- Exact average-pool consistency is faithful to Maa-amet's acquisition and rasterization process.
- Random patch training from one national raster validates transfer to new geology, soil, land use, or acquisition vintages.
- Pixel RMSE, DEM sample pitch, or source round-trip error predicts ground-level rendered beauty.
- A single global roughness, noise spectrum, GAN theme, or diffusion prompt can represent Estonia's material variance.
- Overlap blending alone makes drainage, normals, curvature, or LOD transitions correct.
- Published 2-5 m terrain SR quality extrapolates safely by another 32-80 times in linear resolution.

## Evidence-Backed Design Consequences

1. The main model needs a terrain-native prior. Image Stable Diffusion, ControlNet, DDNM, and RGB SR can contribute mechanisms but not weights or validation.
2. Real high-resolution ground exemplars are the gating input. The missing centimeter band must come from measured ground distributions, not from the 1 m raster alone.
3. The low-resolution training side must reproduce Maa-amet's actual error modes, not only bicubic or average pooling.
4. Orthophoto is worth acquiring as a spatial and semantic condition, not as truth. CIR, CHM, land cover, water, shadows, and ground visibility are needed to prevent visible-object edges from becoming bare-earth relief.
5. Physical maps must be first-class conditions. Soil, peat, surficial deposits, bedrock/lithology, hydrography, landform, land use, and acquisition confidence should condition both residual distribution and correction policy.
6. Source repair precedes or is jointly learned with detail synthesis. Water planes, shore geometry, spikes, pits, canopy leakage, and interpolation grid artifacts require explicit confidence-aware treatment.
7. The two x4 stages should share the original 1 m observation and conditions. Stage two must see stage-one generated outputs during training.
8. National cooking should use coordinate-seeded, overlapping support windows with per-step diffusion consensus and crop only after sufficient context.
9. Deterministic regression and GAN challengers are mandatory. Diffusion is preferred for multimodal class-conditioned detail, but must earn its cost in held-out visual and distributional tests.
10. Acceptance must combine blind rendered review, measured high-resolution holdouts, class-conditional morphology, hydro/topology checks, artifact checks, and correction budgets. No one scalar metric is sufficient.
