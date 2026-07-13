# Microtopography Research Library

## Purpose

This directory is the source and artifact ledger for the papers, repositories, datasets, and research claims named in:

- `../r1.md`
- `../r2.md`
- `../../../tasks/2026-07-13/SPEC-MICROTOPOGRAPHY.md`
- the follow-up bibliography requests made on 2026-07-13

This document is an audit, not an architecture proposal. The exhaustive machine-readable record is [`sources.json`](sources.json). SHA-256 hashes for local PDFs, repository snapshots, and the downloaded dataset are in [`metadata/SHA256SUMS`](metadata/SHA256SUMS).

Status terms are deliberately conservative:

- `method_and_results_read`: relevant method, experiments, and limitations were read from the full text.
- `method_read`: the relevant method was read, but the paper was not exhaustively reviewed.
- `abstract_*`: only abstract/metadata and available result excerpts were reviewed.
- `valid_pdf` / `valid_zip` / `validated_tar`: the local artifact passed `pdfinfo`, `unzip -t`, or `tar -tzf`; validity does not imply scientific endorsement.
- `no_local_pdf`: legal full text was found online or access was restricted, but no valid local PDF was obtained.

## Material Corrections

1. **The two Argudo citations were conflated.** Argudo et al. 2017, DOI `10.1007/s00371-017-1393-6`, is *Coherent Multi-Layer Landscape Synthesis*. The orthophoto-guided FCN is Argudo, Chica, and Andujar 2018, DOI `10.1111/cgf.13345`. Both full papers are local and distinct.
2. **Metzger 2023 is not a diffusion-model DSM paper.** It is *Guided Depth Super-Resolution by Deep Anisotropic Diffusion*, an RGB-guided depth method with an adjustment step. It is useful for guided filtering and consistency, not evidence that a learned terrain diffusion prior creates realistic geology-conditioned detail.
3. **HuHoLa and GrounDiff are not generators of new surface richness.** HuHoLa classifies existing 30/50 cm peat DEMs. GrounDiff removes non-ground objects by translating DSM to DTM. Neither solves Estonia-wide decimeter synthesis.
4. **EarthGen generates RGB top-down imagery, not height.** Its impressive 15 cm/px and 1024x results apply to an `R3` RGB map. They cannot be cited as a decimeter heightfield result.
5. **TerraFusion is heightfield generation, but at macro scale.** It trains on NASADEM 30 m and Sentinel-2 10 m, resampled to 25 m. Its 256-pixel samples are upsampled to 512 pixels for the latent model. It does not validate 0.0625 m morphology.
6. **Terrain Diffusion is directly about terrain and infinite random access, but not this scale.** The published model family is approximately 30 m/px and 90 m/px. The tiling/seed design is relevant; the learned prior is not a decimeter prior.
7. **GSRMTL's reported errors are metres.** On GDEMSR it reports RMSE `1.63`, `2.23`, and `3.27 m` at `x4`, `x8`, and `x16`. That supports optical guidance at conventional DEM scales, not 6.25 cm geometric realism.
8. **Downsampling methods do not supply a terrain prior.** DDNM, DPS, PiGDM, DDRM, ReSample, ILVR, and RePaint constrain or sample generic inverse problems. They say nothing about whether invented height is credible for peat, till, sand, limestone, agriculture, forest floor, shoreline, or exposed rock.
9. **PTRM is macro-scale.** Its geomorphon configuration is on the order of 200 m pixels and 800 m neighborhoods. It is not a decimeter acceptance metric.
10. **The Scott papers establish a systems-level realism failure mode.** Local exemplar patches can look plausible while global drainage is impossible. Pit removal worked only when integrated through multiresolution synthesis, not as a final cleanup pass.
11. **EDiffSR is RGB remote-sensing image SR, not elevation SR.** It demonstrates x4 conditional diffusion chiefly under bicubic degradation and RGB perceptual/image metrics. It cannot establish metric height, substrate-conditioned morphology, or 6.25 cm terrain realism.
12. **The terrain-feature-guided diffusion paper fills macro DEM voids.** Zhao et al. 2024 constrains ASTER GDEMv3/TanDEM-X void reconstruction with terrain feature lines and reports errors in tens of metres. It is not evidence for creating sub-metre detail and must not be conflated with the different RSAGAN paper/repository.
13. **Schrodinger-bridge lunar SR is also macro-scale.** Repasky et al. primarily demonstrate 320 m/px to 20 m/px lunar topography, with metre-scale error. Their real NAC case is preliminary and their stated future range is 0.5-3 m, not 0.0625 m.
14. **ATPRK, GWATPRK, and dissever preserve or redistribute coarse support; they do not supply morphology.** They are useful consistency/trend mechanisms for remote-sensing and soil properties. Treating exact coarse agreement as their contribution must not be confused with learning what beautiful, physically credible fine terrain looks like.
15. **Parent material is a useful covariate, not a geometry oracle.** Cahyana et al. show improved soil classification in volcanic Bogor, and Behrens et al. use multiscale DSM context to predict soil texture. Neither establishes a transferable mapping from geology/soil polygons to centimetre-scale surface geometry in Estonia.

## Direct Terrain Generation And Amplification

| Work | Identity / full text | Local artifact | Code | Review and applicability |
|---|---|---|---|---|
| Guérin et al. 2016, sparse terrain representation | [DOI](https://doi.org/10.1111/cgf.12821) | `guerin-2016-sparse-representation-terrains.pdf` | [official MATLAB](https://github.com/eric-guerin/terrain-amplification), pinned snapshot in `repos/` | Method/results read. Direct paired low/high dictionary amplification; exemplar scale, atom coverage, OMP behavior, and overlap blending are central limitations. |
| Argudo et al. 2017, coherent multi-layer landscapes | [DOI](https://doi.org/10.1007/s00371-017-1393-6) | `argudo-2017-coherent-multilayer-landscape.pdf` | No official implementation identified | Method/results read. Joint terrain and semantic-layer synthesis; not the aerial FCN paper. |
| Argudo et al. 2018, aerial-guided terrain SR | [DOI](https://doi.org/10.1111/cgf.13345) | `argudo-2018-orthophoto-fcn-terrain-superresolution.pdf` | No official implementation identified | Method/results read. Paired imagery is useful guidance, but arbitrary RGB edges are not height. |
| Guérin et al. 2017, terrain cGAN | [DOI](https://doi.org/10.1145/3130800.3130804) | `papers/guerin_cgan_2017.pdf` | Community [reimplementation](https://github.com/nanoxas/sketch-to-terrain) only | Method/results read. Authoring from examples/sketches, not georeferenced decimeter reconstruction. |
| Zhao et al. 2019, GATA | [DOI](https://doi.org/10.1145/3355089.3356553) | Full text read online; no valid local PDF | [official EA code](https://github.com/electronicarts/siggraph-asia-2019-gata), pinned snapshot | Direct multi-theme amplification. Uses 256x256 patches, 128 overlap, and theme embeddings. Official code has no checkpoints and uses TensorFlow 1.10. |
| Paris et al. 2019, implicit 3D amplification | [DOI](https://doi.org/10.1145/3342765) | `papers/2019-paris-implicit-3d-doi-10.1145-3342765.pdf` | Local clone under `repos/implicit-volumetric-terrains/` | Method/results read. Adds caves, arches, overhangs, cliffs, and karst in an implicit volume; not representable by an ordinary heightfield alone. |
| Cortial et al. 2020, planetary hyper-amplification | [DOI](https://doi.org/10.1007/s00371-020-01923-4) | `papers/2020-cortial-hyper-amplification-doi-10.1007-s00371-020-01923-4.pdf` | No official repository identified | Method/results read. Real-time procedural planet amplification; runtime context is forbidden here, while some morphology ideas may transfer cook-side. |
| Scott and Dodgson 2021, pit removal | [DOI](https://doi.org/10.1016/j.cag.2021.06.012) | `papers/2021-scott-dodgson-pit-removal-doi-10.1016-j.cag.2021.06.012.pdf` | No code identified | Method/results read. Pit removal must be inside every resolution stage; local texture quality does not ensure plausible drainage. |
| Scott and Dodgson 2022, terrain realism | [DOI](https://doi.org/10.1145/3531526) | `papers/2022-scott-dodgson-realism-doi-10.1145-3531526.pdf` | No code identified | Method/results read. Human realism evaluation supports user-observable acceptance rather than internal metrics alone. |
| Schott et al. 2024, multi-scale erosion | [DOI](https://doi.org/10.1145/3658200) | `schott-2024-terrain-amplification-multiscale-erosion.pdf` | Local clone under `repos/multiscale-erosion/` | Method/results read. Direct coherent process-based amplification; erosion is not a universal soil/substrate model. |
| Grenier et al. 2024, controlled patterns | [DOI](https://doi.org/10.1111/cgf.14992) | `grenier-2024-controlled-procedural-patterns.pdf` | No official repository identified | Method/results read. Controlled ridge/channel patterns are more structural than generic fBm, but the paper is not material-grounded and is designed for runtime enhancement. |
| Lochner et al. 2023, diffusion authoring | [DOI](https://doi.org/10.1111/cgf.14941) | `papers/lochner_diffusion_2023.pdf` | No maintained official release established | Method/results read. Strong authored 256x256 terrain precedent, not Estonia decimeter evidence. |
| Perche et al. 2023, StyleDEM | [arXiv](https://arxiv.org/abs/2304.09626) | `papers/2023-perche-styledem-arxiv-2304.09626.pdf` | No official repository identified | Method/results read. StyleGAN terrain authoring with sketch, variation, style manipulation and super-resolution tools. This is distinct from *Authoring Terrains with Spatialised Style*, DOI `10.1111/cgf.14936`. |
| Goslin 2026, Terrain Diffusion / InfiniteDiffusion | [arXiv](https://arxiv.org/abs/2512.08309) | `papers/2026-goslin-infinitediffusion-arxiv-2512.08309.pdf` | [official code](https://github.com/xandergos/terrain-diffusion), pinned snapshot | Method/results read. Order-independent infinite access is relevant; trained scales near 30/90 m per pixel are not. |
| Higo et al. 2025, TerraFusion | [arXiv](https://arxiv.org/abs/2505.04050) | `papers/2025-higo-terrafusion-arxiv-2505.04050.pdf` | [official code](https://github.com/millennium-nova/terra-fusion), pinned snapshot | Method/results read. Joint height/texture latent diffusion, trained around 25 m output sampling. |
| Sharma et al. 2024, EarthGen | [arXiv](https://arxiv.org/abs/2409.01491) | `papers/2024-sharma-earthgen-arxiv-2409.01491.pdf` | [project](https://earthgen.github.io/) | Method/results read. RGB satellite-imagery synthesis only, not elevation. |
| Argudo et al. 2025, terrain descriptors | [DOI](https://doi.org/10.1111/cgf.70080) | `argudo-2025-terrain-descriptors.pdf` | [official code](https://github.com/oargudo/terrain-descriptors), pinned snapshot | Method/results read. Exemplar description/selection and macro synthesis support, not a decimeter generator. |
| Huftier et al. 2026, iso-contours | [DOI](https://doi.org/10.1111/cgf.70389) | `papers/2026-huftier-terrain-isocontours-doi-10.1111-cgf.70389.pdf` | [official code](https://github.com/Arches-Team/Contours), pinned snapshot | Method/results read. Macro authoring only. |

## DEM, Depth, And Guided Super-resolution

| Work | Identity / local artifact | What was actually demonstrated |
|---|---|---|
| D-SRGAN, Demiray et al. 2021 | [arXiv](https://arxiv.org/abs/2004.04788), `papers/2021-demiray-dsrgan-arxiv-2004.04788.pdf` | GAN DEM SR at conventional DEM scales. It may invent pseudo-detail and has no geology-conditioned decimeter validation. |
| Real-GDSR, Panangian and Bittner 2024 | [DOI](https://doi.org/10.5194/isprs-annals-X-2-2024-185-2024), local PDF in `papers/` | Real low-resolution DSM differs materially from synthetic bicubic degradation. Useful warning against training only on artificial LR/HR pairs. |
| GSRMTL, Liu et al. 2026 | [DOI](https://doi.org/10.1016/j.jag.2026.105099), local PDF and pinned `YuspaceCraft/DEMSR` snapshot | HRSI-guided DEM SR. GDEMSR errors remain 1.63-3.27 m at x4-x16. The repository currently contains limited release material despite the paper's future-release statement. |
| GrounDiff, Dhaouadi et al. 2026 | [arXiv](https://arxiv.org/abs/2511.10391), local PDF | DSM-to-DTM filtering and stitching, not detail generation. |
| Metzger et al. 2023 | [CVPR](https://openaccess.thecvf.com/content/CVPR2023/html/Metzger_Guided_Depth_Super-Resolution_by_Deep_Anisotropic_Diffusion_CVPR_2023_paper.html), local PDF | Deep anisotropic diffusion for RGB-guided depth SR. Not a denoising diffusion model and not terrain DSM synthesis. |
| CDEM, Yao et al. 2024 | [DOI](https://doi.org/10.1016/j.isprsjprs.2024.01.001), no local PDF | Coordinate decoder at x2-x8 on 2, 10, and 30 m DEMs. Continuous querying is not evidence of physically credible newly invented detail. |
| MFSR, Huang et al. 2025 | [DOI](https://doi.org/10.1016/j.jag.2025.104865), no local PDF | Optical and Depth-Anything guidance for DEM SR; conventional benchmark/reconstruction objective. |
| TfaSR, Zhang et al. 2022 | [DOI](https://doi.org/10.1016/j.isprsjprs.2022.04.028), [Figshare code](https://doi.org/10.6084/m9.figshare.19597201) | Terrain-feature-aware deterministic SR, mainly demonstrated around 30 m to 10 m. |
| DSRFB and AFN, Kubade et al. 2020 | `papers/2020-kubade-dsrfb-arxiv-2007.01940.pdf`, `papers/2020-kubade-afn-arxiv-2010.01626.pdf` | RGB-guided depth-map SR, not terrain generation. |
| Implicit Diffusion Models, Gao et al. 2023 | [arXiv](https://arxiv.org/abs/2303.16491), local PDF and `repos/IDM/` | Continuous natural-image SR. The representation idea is relevant; the learned prior is not terrain/material specific. |
| LapSRN, Lai et al. 2017 | [arXiv](https://arxiv.org/abs/1704.03915), local PDF | Natural-image SR whose authors explicitly distinguish reconstruction from hallucination when LR structure is absent. |
| IM2HEIGHT, Mou and Zhu 2018 | [arXiv](https://arxiv.org/abs/1802.10249), local PDF | Monocular height estimation. Bare-earth use is confounded by canopy, buildings, shadows, water, and acquisition artifacts. |
| EDiffSR, Xiao et al. 2024 | [DOI](https://doi.org/10.1109/TGRS.2023.3341437), `papers/2024-xiao-ediffsr-arxiv-2310.19288.pdf`, pinned official repository | Efficient x4 RGB remote-sensing image diffusion SR. Its AID/DOTA/DIOR experiments, bicubic degradation, and perceptual/image metrics provide architecture and degradation cautions only, not metric terrain evidence. The official repository has no license file, so source availability is not reuse permission. |
| ET-SDE, Zhang et al. 2024 | [arXiv](https://arxiv.org/abs/2407.01908), `papers/2024-zhang-et-sde-arxiv-2407.01908.pdf` | Multipurpose DEM restoration for SR, void filling, and denoising. Demonstrations remain x2/x4 with metre-scale error; no official code release or 6.25 cm evidence was found. |
| Terrain feature-guided void diffusion, Zhao et al. 2024 | [DOI](https://doi.org/10.1016/j.rse.2024.114432), publisher abstract/highlights/results only | Macro DEM void filling with feature-line constraints on ASTER GDEMv3/TanDEM-X. Reported MAE is `28.91 +/- 9.45 m` and RMSE `38.16 +/- 13.00 m`; no legal full text or official code was identified. |
| Lunar Schrodinger bridge, Repasky et al. 2026 | [DOI](https://doi.org/10.3847/PSJ/ae6244), `papers/2026-repasky-lunar-schrodinger-bridge-arxiv-2606.14638.pdf` | Latent VAE/ViT bridge with optical context, primarily 320 m/px to 20 m/px. The paper-named GitHub repository was unavailable during audit, so it is not pinned. |
| ATPRK, Wang et al. 2015 | [DOI](https://doi.org/10.1016/j.rse.2015.06.003), publisher abstract/method summary only | Downscales 500 m MODIS to 250 m and preserves coarse spectral support. This is covariate regression/disaggregation, not terrain synthesis, at roughly 4,000 times the target pixel size. |
| GWATPRK, Jin et al. 2018 | [DOI](https://doi.org/10.3390/rs10040579), `papers/2018-jin-gwatprk-doi-10.3390-rs10040579.pdf` | Locally nonstationary regression kriging demonstrated for 25 km to 1 km soil moisture. It supplies no fine terrain distribution. |
| dissever, Malone et al. 2012 | [DOI](https://doi.org/10.1016/j.cageo.2011.08.021), `papers/2012-malone-dissever-doi-10.1016-j.cageo.2011.08.021.pdf`, pinned official package repository | Iterative weighted-GAM downscaling of 1 km soil carbon to 90 m with coarse-average adjustment. Useful for covariate trend redistribution, not microgeometry. |

`r2` uses the names D-SRCNN and D-SRGAN imprecisely around the Demiray line. Reuse the exact title/venue from the primary paper rather than copying the report's acronym assignment.

## Generic Generative And Inverse-problem Methods

These papers supply mechanisms, not a credible Estonia terrain distribution.

| Family | Papers in the local library | Applicability limit |
|---|---|---|
| Measurement/null-space consistency | DDNM (`2023-wang-ddnm...pdf`), DPS (`2023-chung-dps...pdf`), DDRM (`2022-kawar-ddrm...pdf`), ReSample (`2024-song-resample...pdf`), ILVR (`2021-choi-ilvr...pdf`), RePaint (`2022-lugmayr-repaint...pdf`) | Constrains generic image inverse problems; none defines soil-, rock-, geology-, landcover-, or landform-correct geometry. |
| Pseudoinverse and latent posterior sampling | PiGDM [OpenReview](https://openreview.net/forum?id=9_gsMA8MRKQ), PSLD [arXiv](https://arxiv.org/abs/2307.00619) | Full text read online/no local PDF. Latent/pixel consistency is an engineering constraint, not beauty or variance. |
| Stochastic image SR | SRFlow, SR3, SRDiff, OFTSR in `papers/` | Natural-image fidelity/realism and diversity results do not establish heightfield validity. |
| Tiling/fusion | MultiDiffusion (`2023-bartal...pdf`, pinned repo) | Overlapping denoiser fusion does not guarantee height, slope, normal, or hydrological continuity. |
| Conditional architectures | ControlNet, SPADE, T2I-Adapter in `papers/`, ControlNet pinned repo | Conditioning machinery only. A terrain-specific target distribution and supervised interpretation of covariates remain necessary. |
| Continuous decoders | LIIF and LTE in `papers/` | Arbitrary-scale output is not arbitrary-scale information. Deterministic decoders may regress toward smooth conditional means. |
| Diversity regularizers | MSGAN and DSGAN in `papers/` | Generic cGAN anti-collapse methods; they do not create causal terrain classes. |
| Perception-distortion theory | Blau and Michaeli 2018, local PDF | Establishes a frontier for restoration estimators. It neither forbids L2 diagnostics nor selects diffusion as the terrain solution. |
| Perceptual RGB GAN SR | ESRGAN (`papers/2018-wang-esrgan-arxiv-1809.00219.pdf`) and Real-ESRGAN (`papers/2021-wang-real-esrgan-arxiv-2107.10833.pdf`), pinned official repositories | Useful challengers for high-frequency synthesis and real-degradation modeling, but they hallucinate RGB texture without elevation units, geology, soil, drainage, or geometric validity. Their camera/JPEG/noise degradations are not DTM observation physics. |
| Discrete RGB tokens | VQGAN (`papers/2021-esser-vqgan-taming-transformers-arxiv-2012.09841.pdf`), MaskGIT (`papers/2022-chang-maskgit-arxiv-2202.04200.pdf`), and Muse (`papers/2023-chang-muse-arxiv-2301.00704.pdf`) | Scalable token generation mechanisms only. Lossy tokenization can change metric heights; none learns Estonia terrain or enforces coarse height, hydrology, seam, or substrate constraints. VQGAN and MaskGIT repositories are pinned; no official Muse code release was identified. |
| Single-image patch realism | SinGAN (`papers/2019-shaham-singan-arxiv-1905.01164.pdf`), pinned official repository | Learns multiscale statistics from one natural RGB image. SIFID uses Inception features of RGB patches and is not geometry- or process-aware; only the diagnostic idea transfers after redesign around height derivatives and terrain classes. |
| Stationary wavelet texture statistics | Portilla and Simoncelli 2000 (`papers/2000-portilla-simoncelli-texture-model.pdf`), pinned official code | Band/orientation statistics may help diagnose repetition and anisotropy, but stationary texture equivalence cannot distinguish hydrologically or geologically wrong terrain and is not a physical synthesis model. |

## Process, Morphology, And Real Exemplars

| Work | Local/access status | Relevance and hard limit |
|---|---|---|
| Doane et al. 2024, roughness-event evolution | `doane-2024-topographic-roughness-events.pdf` | Method/results read. Offers causal/event-shaped roughness including pit-mound-like couplets; not a universal substrate model. |
| HOHUM, Nungesser 2003 | [DOI](https://doi.org/10.1016/S0304-3800(03)00067-X), no local PDF | Peat hummock/hollow process model. Directly relevant to bog classes, not other terrain. |
| HuHoLa 2025 | `noumonvi-2025-huhola.pdf`, local repo clone | Classifier of existing 30/50 cm DEMs, not synthesis. |
| Moore et al. 2019 | `moore-2019-peatland-hummock-classification.pdf` | Real high-resolution peatland SfM/DEM evidence. Companion Zenodo dataset is local and CC BY 4.0. |
| Korotkov et al. 2026, bog UAS-LiDAR | `papers/2026-korotkov-uas-lidar-bog-microrelief-doi-10.3390-drones10020121.pdf`, CC BY 4.0 | Maps existing microrelief: 100 points/m2, 10 cm ground classification, 0.5-3 m microforms, 30-40 cm hummocks, 4-23 cm classification thresholds. Not a generator. |
| Pawlik et al. 2024 | [DOI](https://doi.org/10.1016/j.geomorph.2024.109283), no legal local PDF found | Directly valuable comparison of 1 m ALS with 0.025 m TLS pit-mound morphology; no open companion dataset was found. |
| Korpela et al. 2020 | [Mires and Peat](https://www.mires-and-peat.net/pages/volumes/map26/map2603.php), no local PDF | Peat microform mapping and an approximately 0.2 m hummock threshold; measurement/classification, not generation. |
| Lode and Leivits 2011 | [DOI](https://doi.org/10.3176/earth.2011.4.04), no local PDF | Estonia-specific evidence for mire microtopography in 1-2 m DTM; no 6.25 cm ground truth. |
| Telbisz et al. 2024 | [DOI](https://doi.org/10.3390/rs16050737), not downloaded | Doline mapping. It does not support the uncited Burren clint/grike dimensions in `r2`. |
| Cordonnier et al. 2018, subsurface geology | `cordonnier-2018-sculpting-mountains-subsurface-geology.pdf` | Strong geology-conditioned morphology precedent at mountain/volumetric scale, not low-relief cover microtopography. |
| Cordonnier et al. 2023, glacial erosion | `cordonnier-2023-glacial-erosion.pdf`, local repository clone | Process-based glacial landscape formation; macro form rather than universal soil microdetail. |
| Cordonnier et al. 2016, tectonic/fluvial | [Eurographics](https://diglib.eg.org/handle/10.1111/cgf12820), no local PDF | Landscape-scale model with weak direct fit to Estonia-wide 6.25 cm synthesis. |
| Tzathas et al. 2024, analytical erosion | [DOI](https://doi.org/10.1111/cgf.15033), no local PDF | Landscape-scale erosion, not a complete microtopography material model. |
| Gujrati et al. 2019 | [DOI](https://doi.org/10.1126/sciadv.aax0847), no local PDF | Broad roughness scaling cannot justify one Hurst exponent across centimetre-to-metre soil process breaks. |
| SCORPAN, McBratney et al. 2003 | [DOI](https://doi.org/10.1016/S0016-7061(03)00223-4), no local PDF | Establishes soil-prediction covariates. It does not map a soil class directly to geometric roughness. |
| Marzahn et al. 2012, dual-scale agricultural roughness | `papers/2012-marzahn-dual-scale-soil-roughness-doi-10.3390-rs4072016.pdf`, CC BY 3.0 | Direct 2 mm DSM measurements separate seedbed and wheel-track regimes, with directional 1.8-2.0 m periodicity and centimetre-scale relief. Strong evidence against one generic noise law; limited to worked agricultural surfaces. |
| Cahyana et al. 2023, Bogor parent material | [DOI](https://doi.org/10.1016/j.geodrs.2023.e00627), abstract/SSRN metadata; download blocked | Parent material improves soil-class prediction in a volcanic Indonesian study. It does not map lithology to centimetre geometry and does not validate transfer to Estonia. |
| Behrens et al. 2014, hyper-scale soil mapping | [DOI](https://doi.org/10.1016/j.geoderma.2013.07.031), publisher abstract/preview only | Multiscale DSM context predicts topsoil texture and can indicate parent-material changes. Correlation used for soil mapping is not a causal surface generator. |
| Nakamura et al. 2024, outcrop roughness | `papers/2024-nakamura-rock-outcrop-roughness-preprint.pdf`, CC BY 4.0 preprint | Millimetre SfM plus X/Y/HSV predicts TRI on three Japanese coastal outcrops; it predicts a roughness scalar rather than elevation and succeeds inconsistently. Optical correlation is a possible cue, not a generator. |
| Verma and Bourke 2019, rock-breakdown SfM | `papers/2019-verma-bourke-rock-breakdown-sfm-doi-10.5194-esurf-7-45-2019.pdf`, CC BY 4.0 | Sub-millimetre DEM acquisition for eight small Moenkopi sandstone outcrops. Valuable exemplar-capture precedent; no generative method or Estonia lithology evidence. |

## Data And Conditioning Sources

| Dataset/source | Local artifact | Access/license | Use and limitation |
|---|---|---|---|
| Moore northern peatland SfM | `datasets/zenodo-2545675-moore-peatland-sfm-v1.zip` | [Zenodo DOI](https://doi.org/10.5281/zenodo.2545675), CC BY 4.0 | Direct real peat microform exemplar/validation data. Archive passed `unzip -t`. The article may cite the concept DOI ending `2545674`; the downloaded version DOI is `2545675`. |
| EstSoil-EH | Paper in `papers/`; pinned supplement repo | [paper](https://doi.org/10.5194/essd-13-83-2021), [data](https://doi.org/10.5281/zenodo.3473289), ODbL 1.0 | National soil/ecohydrological covariates, not fine target geometry. |
| Maa- ja Ruumiamet orthophotos | No local national raster | [official product page](https://geoportaal.maaamet.ee/est/Ruumiandmed/Ortofotod-p99.html), open download with attribution | Nationwide RGB/CIR/NGR at 20-40 cm, with 10-16 cm in dense settlements. Useful visible conditioning, never a ground-height source under canopy, water, shadow, or occluded cliffs. Capture date is required because roughly half the country is refreshed each year. |
| Mullastikukaart | Existing project source copy; no duplicate in this library | [official product page](https://geoportaal.maaamet.ee/est/ruumiandmed/mullastiku-kaart-p33.html) | 1:10,000 national soil polygons over nearly all non-urban/non-water Estonia. Preserve all four soil components/shares, layered texture, humus, stoniness and fertility rather than reducing to `Sif1`. These are priors, not centimetre geometry. |
| EGT 1:50,000 geology/geomorphology | Four packages in `data/egt/` | [official packages](https://gis.egt.ee/geoportaal/ruumiandmed), [CC BY 4.0 license](https://gis.egt.ee/geoportaal/juhendid/Eesti_Geoloogiateenistuse_ruumiandmete_litsents.pdf) | Detailed bedrock, surficial geology and geomorphology where mapped, but patchy. Taevaskoda has no containing 1:50,000 polygon; nearest mapped boundary is about 7.05 km away. Coverage/confidence masks are mandatory. |
| EGT 1:200,000 geology | Two packages in `data/egt/` | Same official source and CC BY 4.0 license | National fallback family prior only. At Taevaskoda the bedrock is Burtnieki Formation (sandstone with siltstone/clay interbeds). Surficial codes require official decoding; polygon edges must not become relief. |
| Lapinjarvi forest TLS | `papers/2019-luke-lapinjarvi-tls-data-description.pdf` and Pitkanen paper | CC BY 4.0 description; see source records | Unclassified forest scans include stems/vegetation. Ground extraction and artifact control are mandatory before treating them as terrain exemplars. |
| AHN4 | No local raster | [OpenTopography DOI](https://doi.org/10.5069/G9CN725M), CC0 metadata | 0.5 m Dutch DTM is a useful low-relief analogue, but still too coarse to supervise 0.0625 m detail directly. |
| Estonia ALS III / RIEGL VQ-1560i | No local point cloud in this audit | [official campaign page](https://geoportaal.maaamet.ee/est/ruumiandmed/korgusandmed/aerolaserskaneerimise-korguspunktid/als-iii-ring-2016-20172020-p625.html), [vendor announcement](https://newsroom.riegl.international/2021/01/25/estonian-land-board-uses-riegl-vq-1560i-airborne-lidar-mapping-system-for-nationwide-coverage/) | Historical parameters corroborate 2.1 points/m2 nationwide and 18 points/m2 urban. Nationwide returns are far sparser than a 6.25 cm grid (256 cells/m2) and cannot be treated as target-scale truth; current government metadata governs format, refresh, classification, and vegetation-leakage semantics. |
| Estonia RIEGL VQ-1460 | No local point cloud in this audit | [official instrument page](https://geoportaal.maaamet.ee/est/ruumiandmed/ortofotod/tootmislugu/aerolaserskanner-riegl-vq-1460-p1031.html), [official 2025 campaign notice](https://maaruum.ee/2025-aasta-aeromoodistamise-hooaeg-alanud) | Official pages place the scanner in use from 2025 and confirm spring 2025 operations with an IGI UrbanMapper-2 camera. They do not substantiate `r2`'s exact “installed in 2024” wording or “UrbanMapper 2 EVO” name. National-flight density is about 8 points/m2, still not a 6.25 cm target surface. |

Pinned EGT package identities:

```text
075d90683faea7cb8694beb89a62d9ccc3e405446a2ae2bec4be10e0d10f950e  data/egt/aluspohi-50k-2026-04-06.zip
435b361972b82f0123e5a2d763e79df4c429d8cd86a1ebd92661d07ed19ef7a7  data/egt/pinnakate-50k-2026-04-06.zip
bec627dd602323c5f3425e48cc5a4358da37483e52781e7aacf65907befb0b75  data/egt/geomorfoloogia-50k-2026-04-06.zip
6f16a4290d92f25c17c9371464d13e5758e93119e24c9e01a12f4c8e0bd44b07  data/egt/geoloogiline-baaskaart-gdb-50k-2026-04-06.zip
8351f5e1ec60a60a8ce377d0c582b8550549ed92e4c0ac5536bc7681a0f057ee  data/egt/aluspohi-200k-2026-04-28.zip
a4bd969e11d6869a34cc44f101050d7a9ab29c7e9aee558efab9f94f01a58f39  data/egt/pinnakate-200k-2026-04-28.zip
```

## Repository Snapshots

All tarballs below passed `tar -tzf`. Exact hashes are in `metadata/SHA256SUMS`.

| Repository | Pinned revision | Local snapshot | Notes |
|---|---|---|---|
| `wyhuai/DDNM` | `00b58eac7843a4c99114fd8fa42da7aa2b6808af` | `repos/github-wyhuai-DDNM-00b58eac.tar.gz` | Official. |
| `xandergos/terrain-diffusion` | `82a0431281f21a6ec3d691a12ee61525de5b0790` | `repos/github-xandergos-terrain-diffusion-82a04312.tar.gz` | Official. |
| `eric-guerin/terrain-amplification` | `5b83d65315e9845401df78f6a104c21f9bd473d5` | `repos/github-eric-guerin-terrain-amplification-5b83d653.tar.gz` | Official MATLAB reference; small repository. |
| `eric-guerin/gradient-terrains` | `68da08b1b75a1e2b5650b5045a62a435072e5874` | `repos/github-eric-guerin-gradient-terrains-68da08b1.tar.gz` | README identifies the Gradient Terrain Authoring code and root LICENSE is MIT; officialness remains unverified and implementation code was not reviewed. |
| `electronicarts/siggraph-asia-2019-gata` | `0c042f711dae7b6721485edcd6e454739b071a49` | `repos/github-electronicarts-siggraph-asia-2019-gata-0c042f71.tar.gz` | Official experimental TensorFlow 1.10 code; no checkpoints. |
| `omerbt/MultiDiffusion` | `69bcdcef437dfdbf48c53624d6bf6f397b5f4894` | `repos/github-omerbt-MultiDiffusion-69bcdcef.tar.gz` | Official. |
| `lllyasviel/ControlNet` | `ed85cd1e25a5ed592f7d8178495b4483de0331bf` | `repos/github-lllyasviel-ControlNet-ed85cd1e.tar.gz` | Official. |
| `millennium-nova/terra-fusion` | `d79c1f64f91ba3ec7f86c8a6b220c19e735d91f4` | `repos/github-millennium-nova-terra-fusion-d79c1f64.tar.gz` | Official. |
| `YuspaceCraft/DEMSR` | `7bd0c75d1c4e0e00b9cc74214409af7043db0051` | `repos/github-YuspaceCraft-DEMSR-7bd0c75d.tar.gz` | Named by GSRMTL; current snapshot is very small/limited and is not a complete code/data release. |
| `oargudo/terrain-descriptors` | `03e27f6471771e365d473d03557f8053d21e3ff9` | `repos/github-oargudo-terrain-descriptors-03e27f64.tar.gz` | Official. |
| `Arches-Team/Contours` | `1e1a2f6301b8d8adc9221321ffc95b8fac78d43c` | `repos/github-Arches-Team-Contours-1e1a2f63.tar.gz` | Official. |
| `LandscapeGeoinformatics/EstSoil-EH_sw_supplement` | `8ff3afafd18e54785c9f4d0781621e1a3c4882be` | `repos/github-LandscapeGeoinformatics-EstSoil-EH-sw-supplement-8ff3afaf.tar.gz` | Official supplement. |
| `nanoxas/sketch-to-terrain` | `fbf33ba544052b542bfdc19c8fb4d749c3e233cc` | `repos/github-nanoxas-sketch-to-terrain-fbf33ba5.tar.gz` | Community reimplementation, not official Guérin 2017 code. |
| `XY-boy/EDiffSR` | `32269df4d9b479adaa111ae3525e7ef753d85a8f` | `repos/github-XY-boy-EDiffSR-32269df4.tar.gz` | Official source snapshot. No license file is present, so source availability does not grant reuse rights. |
| `xinntao/ESRGAN` | `73e9b634cf987f5996ac2dd33f4050922398a921` | `repos/github-xinntao-ESRGAN-73e9b634.tar.gz` | Official, Apache-2.0. |
| `xinntao/Real-ESRGAN` | `a4abfb2979a7bbff3f69f58f58ae324608821e27` | `repos/github-xinntao-Real-ESRGAN-a4abfb29.tar.gz` | Official, BSD-3-Clause. |
| `CompVis/taming-transformers` | `3ba01b241669f5ade541ce990f7650a3b8f65318` | `repos/github-CompVis-taming-transformers-3ba01b24.tar.gz` | Official VQGAN release, MIT. |
| `google-research/maskgit` | `1db23594e1bd328ee78eadcd148a19281cd0f5b8` | `repos/github-google-research-maskgit-1db23594.tar.gz` | Official, Apache-2.0; repository archived and release is oriented around inference/checkpoints rather than complete training. |
| `tamarott/SinGAN` | `df38a4214af95462fa97a613d6ba53eb441509dd` | `repos/github-tamarott-SinGAN-df38a421.tar.gz` | Official, MIT; includes SIFID implementation. |
| `LabForComputationalVision/textureSynth` | `30c3a6c56a249daa292fca18131b968c935eedf9` | `repos/github-LabForComputationalVision-textureSynth-30c3a6c5.tar.gz` | Official Portilla-Simoncelli implementation, MIT. |
| `pierreroudier/dissever` | `37441ee1ccd8ee6c256d1ab4132c81a5acced269` | `repos/github-pierreroudier-dissever-37441ee1.tar.gz` | Official R package source; package metadata declares GPL-2. |

Additional working clones from parallel research tracks are present under `repos/`. The pinned tarballs above are the immutable references in this audit.

## Invalid And Unresolved Items

Two files in `papers/` have a `.pdf` suffix but contain HTML and fail `pdfinfo`:

- `papers/2018-argudo-aerial-terrain-superresolution-doi-10.1111-cgf.13345.pdf`
- `papers/argudo_ortho_fcn_2018.pdf`

They were retained because parallel agents created them and this audit must not delete other work. Use the valid canonical copy `argudo-2018-orthophoto-fcn-terrain-superresolution.pdf` at the library root.

The report mentions for EDiffSR, terrain-feature-guided diffusion void filling, lunar Schrodinger bridges, ATPRK, GWATPRK, dissever, Marzahn, Bogor, Behrens, Nakamura, Verma/Bourke, RIEGL, and the generic RGB image families are now identified and bounded in the tables and `sources.json`. Access limitations are recorded rather than silently upgraded: Zhao 2024, Wang 2015, Cahyana 2023, and Behrens 2014 were not available as legal local full texts; the paper-named lunar repository currently returns not found; and no official code was located for several papers.

One quantitative research claim remains unresolved and is excluded from design evidence: `r2` gives Burren clint areas of `0.4-2.8 m2`, grike depths usually below `2 m`, grike widths of `3.2-6.5 cm`, and karren widths of `5-30 cm` without a citation. Goldie and Cox 2000, *Comparative morphometry of limestone pavements in Switzerland, Britain and Ireland*, is plausibly relevant, but its accessible metadata does not establish those numbers and no legal full text was obtained. The values must not be used until a primary source is supplied.

## Integrity Check

On 2026-07-13:

- every canonical local PDF in the tables above passed `pdfinfo`;
- the two known HTML masquerades failed and are explicitly quarantined by documentation;
- every pinned repository tarball passed `tar -tzf`;
- the Moore Zenodo archive passed `unzip -t`;
- the SHA-256 manifest contains 127 local artifacts, including six EGT source packages, all currently downloaded primary PDFs, nested reference archives, and 21 pinned repository tarballs;
- `sources.json` contains 125 source records and exactly represents all 127 manifest paths and hashes: 23 duplicate copies, three support artifacts, two quarantined HTML masquerades, one newly normalized canonical repository snapshot, and 98 pre-existing canonical artifacts;
- `sources.json` SHA-256 is `40521f07146e7cb87176886cf2845cbd24f604b5b84c441da9652a6265b7d460`, `sources.schema.json` SHA-256 is `8d6911d62602d087839f8aa5eb05cc392cb1b91df8df74ff661e9e1b41604e39`, and `metadata/SHA256SUMS` SHA-256 remains `e0c6dc1d7f67502c9675a12db8fadeb4694744132e1bcaa28ef21d305f35aa77`.
