# Primary-Source Claim Audit

Status: interim root-agent audit, 2026-07-13. This document records evidence boundaries before architecture selection. It is not a replacement proposal or implementation authorization.

## Target Being Audited

The target is not generic DEM super-resolution and not a less-triangulated interpolation. It is beautiful, realistic, materially and geomorphologically conditioned geometric variance sampled at 0.0625 m across Estonia, cooked and packed offline. The source Maa-amet 1 m DTM is a fallible observation. Water interpolation, missing returns, vegetation leakage, grid artifacts, and other source defects may require correction rather than exact preservation.

The accepted LOD -1/-2 transport, packing, streaming, decoding, and rendering infrastructure is out of scope for this method audit. Runtime synthesis remains forbidden.

## Claims In `r1.md`

### DDNM does not supply a terrain prior or a real-DTM observation model

**Primary evidence:** Wang, Yu, and Zhang, *Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model*, ICLR 2023, Sections 3-4 and Appendix. DDNM projects a denoised estimate into the range and null spaces of an explicit linear degradation operator `A`. Its super-resolution experiments use synthetic average-pooling and pretrained natural-image denoisers on ImageNet/CelebA. The paper lists dependence on an explicit degradation operator and on the pretrained denoiser's distribution; undesirable random outputs and slow sampling are reported limitations.

**Correction:** `r1.md` is mathematically right that block-average consistency can be imposed cheaply if block averaging is the accepted observation operator. It is not evidence that Maa-amet's real 1 m DTM was formed by block averaging a latent 6.25 cm ground surface. Exact projection can preserve the very observation errors that must be corrected. DDNM also delegates null-space realism to its prior; a natural-image prior contains no Estonia soil, lithology, hydrology, or land-use model.

### MultiDiffusion provides overlap consensus, not terrain continuity guarantees

**Primary evidence:** Bar-Tal et al., *MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation*, ICML 2023, Sections 3-4. At each step, overlapping crop predictions are fused by a least-squares consensus whose closed form is weighted pixel averaging. Demonstrations use Stable Diffusion image/latent panoramas and regional controls. The paper states that results inherit failures and biases of the underlying generative model.

**Correction:** overlap consensus is useful tiling machinery, but it does not by itself guarantee continuity of terrain gradients, normals, curvature, drainage, material processes, or hierarchy. It cannot compensate for an unsuitable learned prior.

### ControlNet is a conditioning mechanism, not evidence of terrain competence

**Primary evidence:** Zhang, Rao, and Agrawala, *Adding Conditional Control to Text-to-Image Diffusion Models*, ICCV 2023. The paper demonstrates spatial conditioning of large pretrained text-to-image models using trainable zero-convolution branches.

**Correction:** ControlNet can carry registered DTM, orthophoto, soil, geology, land cover, or masks into a suitable model. It neither creates the terrain prior nor establishes that a natural-image backbone predicts metric height. Its successful small-dataset training claim cannot be transferred without a terrain-specific target dataset and validation.

### Guérin 2017 cGAN does not perform the final amplification

**Primary evidence:** Guérin et al., *Interactive Example-Based Terrain Authoring with Conditional Generative Adversarial Networks*, ACM TOG 2017, Sections 3, 6.2, 7.1, and 7.4. The cGAN produces authored large-scale terrain. The paper explicitly uses Guérin 2016 sparse amplification for the final small-scale detail. Its database is approximately 30 m per cell with 1 m vertical precision, and trained synthesizer scales are bound to training scales.

**Correction:** this paper supports learned terrain authoring and conditioning, not a demonstrated learned 1 m-to-centimeter amplifier.

### Orthophoto-guided FCN evidence stops at 2 m

**Primary evidence:** Argudo, Chica, and Andujar, *Terrain Super-resolution through Aerial Imagery and Fully Convolutional Networks*, CGF 2018, Sections 4-6. The network maps a 15 m DEM plus 1 m orthophoto to a 2 m DEM on selected alpine natural terrain. It produces smoother outputs than competing amplifiers. Vegetation prevents ground inference, shadows create false discontinuities, and the authors propose distinguishing DTM from DSM objects in future work.

**Correction:** the paper establishes that registered imagery can improve coarse terrain inference. It does not establish that 20 cm imagery determines 6.25 cm ground geometry, especially under vegetation or water. Orthophoto is a potentially high-value structural and class covariate, not direct metric ground truth.

### PTRM is a macro-landform study

**Primary evidence:** Rajasekaran et al., *PTRM: Perceived Terrain Realism Metric*, ACM TAP 2022, Sections 3, 5, and 9. Experiments use SRTM at approximately 200 m per pixel; each geomorphon covers roughly 800 x 800 m. The authors warn that scale changes affect results, spatial correlations are omitted, some real terrains appear perceptually unrealistic, and the fixed camera/texture/light regime limits transfer.

**Correction:** PTRM may inform macro-landform studies. It is not an acceptance metric for centimeter-to-meter microtopography and cannot replace ground-level rendered review.

### Published learned terrain systems do not reach the target scale

- **Lochner et al. 2023:** cascaded diffusion maps 153 m to 19.1 m to 2.39 m per pixel. Its 5 x 5 km perceptual study omits amplification for parity. The paper reports failure to align elevation constraints, noisy incoherent satellite textures, and categorical perceptual effects.
- **StyleDEM 2023:** separate StyleGAN models are trained at 30 m and 5 m precision. The fixed 1024-pixel generator requires overlapping patch generation, histogram retargeting, and minimum-error cuts. It is evidence for multi-scale learned style control, not target-scale weights or data.
- **Terrain Diffusion 2026:** the official paper/repository documents 30 m and 90 m per-pixel hierarchy levels. InfiniteDiffusion and seed-consistent tiling may transfer as infrastructure; the released terrain prior is macro-scale.
- **TerraFusion 2025:** jointly models height and texture using 25 m samples derived from NASADEM/Sentinel-2. The authors state that some heightmaps remain unnatural and quality is insufficient for practical real-world terrain modeling.

## Claims In `r2.md`

### Guérin 2016 is not a generally coherent deterministic backbone

**Primary evidence:** Guérin et al., *Sparse Representation of Terrains for Procedural Modeling*, CGF 2016, Sections 3-6. Low-resolution patches are sparsely represented against a paired low/high-resolution dictionary, then reconstructed with high-resolution atoms. The exemplar and target must contain similar terrain. The authors explicitly state that geomorphological consistency and small-scale coherence cannot be guaranteed; observed coherence depends on atom radius being small relative to input landforms. The method is a heightfield and cannot represent cliffs or overhangs.

**Correction:** it remains a useful exemplar baseline and local component. `r2.md` overstates it as a sufficient national backbone.

### `r2.md` conflates two Argudo papers

**Primary evidence:** Argudo et al. 2017, *Coherent Multi-Layer Landscape Synthesis*, *The Visual Computer* 33(6), DOI `10.1007/s00371-017-1393-6`, is a dictionary/exemplar method that adds slope, orientation, drainage, vegetation, and soil layers. Argudo, Chica, and Andujar 2018, CGF 37(2), DOI `10.1111/cgf.13345`, is the orthophoto-guided FCN.

**Correction:** the 2017 DOI cannot be cited as the orthophoto network. Both methods are relevant for different reasons and must be evaluated separately.

### HuHoLa classifies existing mire relief; it does not generate it

**Primary evidence:** Noumonvi et al., *HuHoLa: A Novel Hummock-Hollow-Lawn Mire Microtopography Modelling Approach*, Ecological Modelling 2025. HuHoLa uses sink filling and thresholds to classify hummock, hollow, and lawn forms in an existing DEM, with water-temperature and water-table proxies.

**Correction:** HuHoLa can supply labels, validation, or conditioning logic. Calling it a process/statistical synthesizer is unsupported.

### Current process amplifiers cover restricted phenomena

**Primary evidence:** Schott et al., *Terrain Amplification Using Multi-Scale Erosion*, ACM TOG 2024, Sections 4-7. It combines fluvial erosion, thermal stabilization, sediment deposition, elevation retargeting, and multiscale breaching. It outperforms sparse and procedural alternatives for hydrological coherence on its tested terrain family. The authors report whole-map drainage/sediment dependencies, patch-boundary propagation and blending problems, an 8192-square experimental limit, and outputs restricted to hydraulic landforms unless more geological phenomena are added.

**Primary evidence:** Grenier et al., *Real-Time Terrain Enhancement with Controlled Procedural Patterns*, CGF 2024. It is explicitly structured Phasor noise producing slope-aligned erosion-like ravines. It improves on isotropic noise but remains a compact procedural erosion pattern family.

**Correction:** multiscale erosion is a serious candidate for fluvial/hillslope regimes, not a universal Estonia micro-surface model. Controlled procedural patterns conflict with the project's prohibition on decorative noise when used as a substitute for measured material/process structure.

## Additional Primary Evidence

### GATA contributes learned theme control, but no reusable target-scale model

**Primary evidence:** Zhao et al., *Multi-Theme Generative Adversarial Terrain Amplification*, ACM TOG 2019, and official EA repository commit `0c042f711dae7b6721485edcd6e454739b071a49`. GATA is an `x4`, 256-pixel, conditional adversarial amplifier with learned 1024-dimensional theme embeddings and overlapping patch assembly. The repository is experimental TensorFlow 1.10 code. It publishes neither raw training data nor checkpoints; the README says dataset links are future work. Named preprocessing inputs span approximately 0.5-8 m products.

**Evidence boundary:** learned multi-theme embeddings and the improvement over sparse patch coherence are relevant. The released artifact is not an Estonia model and cannot demonstrate centimeter morphology.

### New optical-guided DEM SR still uses coarse synthetic degradation

**Primary evidence:** Liu et al., *DEM Super-resolution Guided by High-resolution Remote Sensing Images Using Multitask Learning*, IJAEOG 2026, Sections 3-5. GSRMTL trains on 2 m Ontario DEMs and orthophotos resampled onto the DEM grid, with synthetic x4/x8/x16 inputs, L1 height loss, semantic pseudo-labels, and RMSE/slope/aspect evaluation. Reported DEM RMSE is 1.63 m, 2.23 m, and 3.27 m respectively.

**Evidence boundary:** semantic and multimodal fusion are relevant architecture ideas. The experiment discards native 20 cm orthophoto detail by resampling it to the 2 m DEM grid and does not test real 1 m-to-centimeter degradation or perceptual microgeometry.

### GrounDiff addresses observation correction but has known terrain failures

**Primary evidence:** Dhaouadi et al., *GrounDiff: Diffusion-Based Ground Surface Generation from Digital Surface Models*, WACV 2026, official project page and paper. It translates DSM to DTM and uses prior-guided stitching. The authors report failures on abrupt elevation changes, where cliffs resemble building facades and are smoothed, and under dense vegetation without ground references.

**Evidence boundary:** selective learned correction and smooth road reconstruction are relevant to a distinct source-conditioning stage. The model does not synthesize unresolved material microtopography, and its cliff failure is directly material at Taevaskoda.

### The lunar Schrodinger bridge is a useful optical-conditioning challenger, not target-scale evidence

**Primary evidence:** Repasky et al., *Improving Lunar Topography with Deep Learning Schrodinger Bridges*, PSJ 2026 / arXiv `2606.14638`, Sections 3-6. The latent bridge performs synthetic `16x` super-resolution from a mean-downsampled 320 m/pixel lunar DEM to 20 m/pixel, conditioned by rendered optical images. Its 91,708 patches come from the same high-resolution lunar DEM product. On rendered held-out data, single samples have 5-6 m RMS elevation error; averaging 20 samples reduces elevation error while smoothing higher-frequency features. The 19.2 km mosaic uses overlapping 96-pixel patches and final-output weighted averaging. Real NAC imagery produces lower quality and visible mosaicing artifacts. The authors state that clean meter-scale target data do not yet exist in sufficient volume, do not yet compare the method directly with shape-from-shading, and leave higher-resolution work for the future.

**Evidence boundary:** an image-to-image bridge, multiple samples, and uncertainty maps are legitimate challengers for typed ambiguous reconstruction. Synthetic mean degradation, rendered image conditioning, final feathering, a lunar prior, 20 m sampling, and 5-6 m error do not establish Estonia source repair or centimeter morphology. The paper reinforces the target-data and real-degradation gates rather than relaxing them.

### ET-SDE remains conventional DEM restoration at meter scale

**Primary evidence:** Zhang, Zuo, and Li, *Efficient Terrain Stochastic Differential Equations for Multipurpose Digital Elevation Model Restoration*, arXiv `2407.01908`, Sections 4-5. The super-resolution experiments use a 2 m Pyrenees DEM and an ASTER Mount Tai region, synthetic `2x`/`4x` tasks, and random 90/10 patch splits within each region. The Pyrenees experiment reports 2.15 m RMSE for 15 m-to-2 m restoration. The reported Mount Tai `4x` ablation has 4.13 m RMSE. On one RTX 3090, the paper reports 7.95 seconds for a 256-square input and cannot handle multiple large voids.

**Evidence boundary:** a terrain-specific SDE and efficient prior encoder deserve inclusion in the learned challenger set. Within-region random splitting, synthetic degradations, mountain DEMs, meter error, and seconds per small patch do not validate geographic transfer, real Maa-amet errors, target-scale beauty, or a national cook budget.

### EDiffSR and terrain-feature-guided void diffusion solve different data domains

**Primary evidence:** Xiao et al., *EDiffSR*, TGRS 2024 / arXiv `2310.19288`, trains `4x` diffusion super-resolution on 3,000 RGB remote-sensing images from AID using bicubic degradation and evaluates perceptual image metrics on AID/DOTA/DIOR/NWPU. It reports 19.26 seconds for its diffusion inference benchmark and explicitly notes high sampling cost and weak real-degradation adaptability. Zhao et al., *Void Filling of Digital Elevation Models Based on Terrain Feature-Guided Diffusion Model*, RSE 2024, conditions macro DEM inpainting on ridge/valley lines; the public full-paper preview reports 28.91 m MAE and 38.16 m RMSE over ASTER/TanDEM-X tests.

**Evidence boundary:** EDiffSR is RGB synthesis, not height. Terrain-feature line conditioning is relevant to macro void reconstruction, but tens-of-meters error is unrelated to a 1 m-to-0.0625 m terrain distribution. Neither paper supplies target-scale weights, Estonia material conditioning, or a production architecture.

### Target-scale measurement papers strengthen the acquisition program, not a universal generator

**Primary evidence:** Marzahn, Seidel, and Ludwig 2012 measure 6-22 square-meter agricultural plots photogrammetrically on a 2 mm grid with vertical accuracy at or below 2 mm. They separate small-scale seedbed rows/clods from larger wheel tracks and show that both depend on the tillage tool and field state. Verma and Bourke 2019 measure fewer than 10 square-meter weathered Moenkopi sandstone surfaces at sub-millimeter resolution, reporting about 0.5 mm horizontal and 0.3 mm vertical error; their full data are available only by author request. Nakamura et al. 2024 use 0.20-0.63 mm/pixel coastal outcrop models to predict a local terrain-ruggedness index from HSV imagery across three Japanese lithologies, not a height surface; their model underestimates some large cracks despite acceptable aggregate RMSE.

**Evidence boundary:** these papers demonstrate feasible target-scale acquisition and prove that agricultural and rock morphology depend on event state, lithology, and scale. Their plots are small and foreign, their sensing has representation limits, and the Nakamura model predicts ruggedness rather than geometry. They support a versioned acquisition/target registry and optical-condition falsification tests, not direct transfer into national synthesis.

### Actual target-scale exemplar evidence exists, but coverage is regime-specific

- Moore et al. 2019 publish an open SfM DEM corpus for hummock-hollow structure across nine northern peatlands (Zenodo `2545675`).
- Pawlik et al. 2024 compare 1 m ALS and 0.025 m TLS DTMs for forest pit-and-mound relief and show the TLS surface represents the forms better.
- Agricultural roughness literature separates millimeter aggregate roughness, 2-100 mm clods, and 100-300 mm directional tillage/furrow structure. These scales require event and management state, not a single land-cover amplitude.

**Evidence boundary:** these sources can ground particular regimes and validation sets. They do not justify transferring one foreign forest or peat surface across unrelated Estonia substrates.

## Consequences For The Next Spec

These are problem constraints established by the audit, not a method selection:

1. Treat source correction, covariate-conditioned structural inference, and unresolved material/process synthesis as separate learned or deterministic tasks with separate evidence and validation.
2. Treat exact downsampling, hierarchy, seams, quantization, and determinism as representation constraints. They must not force preservation of known source defects or become the visual objective.
3. Require sub-decimeter targets for every claimed sub-decimeter prior. A paper trained at 2-30 m can justify architecture ideas, not centimeter weights, morphology, or acceptance thresholds.
4. Condition on causal or diagnostic factors: substrate/lithology, Quaternary material, soil, hydrology/water table, slope/curvature/drainage, land use and management event, vegetation context, and registered imagery where it adds structure. A single global roughness or style knob is insufficient.
5. Use orthophoto as a registered structural/class signal with explicit shadow, canopy, water, season, and temporal-mismatch handling. Do not treat RGB intensity as height.
6. Preserve direct ground-level visual comparison as the decisive user-observable gate. Spectra, variograms, distributions, terrain descriptors, and perceptual metrics are diagnostic coverage, not substitutes for the rendered result.
7. A national solution must fail closed by regime until its target-scale prior is supported. It must not fill unsupported Estonia with generic noise or a foreign forest matrix merely to claim detail everywhere.
