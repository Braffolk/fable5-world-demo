# Deterministic, Exemplar, And Process Source Manifest

Access date: 2026-07-13. Paths are relative to `docs/deep-research/microtopography-generation/library/`.

This manifest records the primary sources used for the deterministic/process proposal. "Full" means the paper body, methods, results, limitations, figures, and supplement where present were read. It does not mean that a method was validated for Estonia or for 0.0625 m terrain. Those evidence limits are in `../review/deterministic-process-primary-source-audit.md`.

## Terrain Synthesis And Amplification

| Work | Canonical source | Local full text | SHA-256 | Read scope |
|---|---|---|---|---|
| Guérin et al. 2016, *Sparse Representation of Terrains for Procedural Modeling* | [DOI](https://doi.org/10.1111/cgf.12821), [author PDF](https://perso.liris.cnrs.fr/eguerin/download/eg2016.pdf) | `papers/2016-guerin-sparse-terrain-doi-10.1111-cgf.12821.pdf` | `1cd279c92af7a78736973db6acbad4364c095f3e04ccf7e89aae09afc7f83340` | Full paper, especially Sections 3, 5.3, and 6; official MATLAB repository inspected |
| Argudo et al. 2017, *Coherent Multi-Layer Landscape Synthesis* | [DOI](https://doi.org/10.1007/s00371-017-1393-6), [institutional full text](https://upcommons.upc.edu/bitstreams/6c160aef-e5a4-4757-afcb-6dab4836b6db/download) | `papers/2017-argudo-coherent-multilayer-landscape-doi-10.1007-s00371-017-1393-6.pdf` | `6df9b9cf72b0003dd3157188f14436ab07aa8cc8c905e235a638562293ff0a6a` | Full paper. This is the dictionary/multi-layer paper, not the orthophoto FCN |
| Argudo, Chica, and Andujar 2018, *Terrain Super-resolution through Aerial Imagery and Fully Convolutional Networks* | [DOI](https://doi.org/10.1111/cgf.13345) | `argudo-2018-orthophoto-fcn-terrain-superresolution.pdf` | `df62eb57644a03018b5ea8131fd0774cf113292e4366bff34333d8fe3b3f64a5` | Full paper, Sections 3-6; official Caffe repository inspected |
| Paris et al. 2019, *Terrain Amplification with Implicit 3D Features* | [DOI](https://doi.org/10.1145/3342765), [author full text](https://people.cs.uct.ac.za/~jgain/wp-content/papercite-data/pdf/paris2019.pdf) | `papers/2019-paris-implicit-3d-doi-10.1145-3342765.pdf` | `2bf97eae2f6d81cad5fbb2309311bc27440010b4312a159eefc7d81a1553c0a0` | Full paper, Sections 3-9; authors' recoded C++ repository inspected |
| Cortial et al. 2020, *Real-Time Hyper-Amplification of Planets* | [DOI](https://doi.org/10.1007/s00371-020-01923-4), [HAL full text](https://hal.science/hal-02967067v1/document) | `papers/2020-cortial-hyper-amplification-doi-10.1007-s00371-020-01923-4.pdf` | `a8386014e58ba283d1f81d929a4a2360c86e310252c9dae0b9b5fbc81a41200d` | Full paper, Sections 3-9 |
| Scott and Dodgson 2021, *Example-Based Terrain Synthesis with Pit Removal* | [DOI](https://doi.org/10.1016/j.cag.2021.06.012), [author preprint](https://www.neildodgson.com/pubs/CAG2021-pit-removal-final-preprint.pdf) | `papers/2021-scott-dodgson-pit-removal-doi-10.1016-j.cag.2021.06.012.pdf` | `7fdbb8081ef4d8e76d93552596b0e3e7d11c695c949b98bb4eac21e40ff3269f` | Full paper, Sections 3-7. Author host had a certificate hostname mismatch; content and metadata were validated after retrieval |
| Scott and Dodgson 2022, *Evaluating Realism in Example-Based Terrain Synthesis* | [DOI](https://doi.org/10.1145/3531526), [author full text](https://www.neildodgson.com/pubs/TAP2022-evaluating-terrain.pdf) | `papers/2022-scott-dodgson-realism-doi-10.1145-3531526.pdf` | `321c649079f6315c41509bd5366465ddf63a149b00e0050a086a70cd8f824c67` | Full paper, both experiments and Sections 6-7 |
| Schott et al. 2024, *Terrain Amplification Using Multi-Scale Erosion* | [DOI](https://doi.org/10.1145/3658200), [HAL full text](https://hal.science/hal-04565030v1/document) | `papers/2024-schott-multiscale-erosion-doi-10.1145-3658200.pdf` | `91f76e1a920ec3629ad065f927e0ecd1bb6498f67bf2c44217aa817d379290aa` | Full paper, Sections 4-7; official C++/OpenGL implementation inspected |
| Grenier et al. 2024, *Real-Time Terrain Enhancement with Controlled Procedural Patterns* | [DOI](https://doi.org/10.1111/cgf.14992) | `papers/2024-grenier-controlled-procedural-patterns-doi-10.1111-cgf.14992.pdf` | `d317cee806fe836af59479b552c9679d3e9537c8bc88bee66e90d3a36b2b05e1` | Full paper, including limitations |
| Argudo et al. 2025, *Terrain Descriptors for Computer Graphics* | [DOI](https://doi.org/10.1111/cgf.70080) | `papers/2025-argudo-terrain-descriptors-doi-10.1111-cgf.70080.pdf` | `886dc18e4c8521c4ce4e341116cd3f7b48217f284d34252e5a4e35b9cf486579` | Full paper; official descriptor implementation inspected |

## Geology And Geomorphic Processes

| Work | Canonical source | Local full text | SHA-256 | Read scope |
|---|---|---|---|---|
| Cordonnier et al. 2018, *Sculpting Mountains: Interactive Terrain Modeling Based on Subsurface Geology* | [DOI](https://doi.org/10.1109/TVCG.2017.2689022) | `cordonnier-2018-sculpting-mountains-subsurface-geology.pdf` | `22e4ca6a929a4ab60607c8b763852b45191a9b561a89051d907aa8afe4eed064` | Full paper; mountain-scale geology/erosion model |
| Cordonnier et al. 2023, *Forming Terrains by Glacial Erosion* | [DOI](https://doi.org/10.1145/3592422) | `cordonnier-2023-glacial-erosion.pdf` | `710e9eb8bdb6ca34268c2d2dfa16cb113afedff177c0049ab52700ed5f2f7ae8` | Full paper; official repository inspected. The demonstrated scale is glacial landform evolution, not centimeter surface synthesis |
| Doane et al. 2024, *Topographic Roughness as a Signature of Stochastic Geomorphic Events* | [DOI](https://doi.org/10.1029/2024AV001264) | `papers/2024-doane-topographic-roughness-doi-10.1029-2024AV001264.pdf` | `1285ce59f2898a46cbbf370da3166a8f026401cb12010f819941c346f031942d` | Full paper and official figure-code repository |
| Pawlik et al. 2024, *Evaluation of the Hillslope Fine-Scale Morphology under Forest Cover with Pit-Mound Topography* | [DOI](https://doi.org/10.1016/j.geomorph.2024.109283), [author-hosted full text](https://www.researchgate.net/publication/380909495_Evaluation_of_the_hillslope_fine-scale_morphology_under_forest_cover_with_pit-mound_topography_-Integration_of_geomorphometry_geophysical_methods_and_soil_features) | Not archived: publisher is closed and the author-hosted binary endpoint returned HTTP 403 | N/A | Full corrected proof read through the author-hosted page, especially Sections 2.2, 3.1, and discussion |

## Peatland Microrelief

| Work | Canonical source | Local artifact | SHA-256 | Read scope |
|---|---|---|---|---|
| Moore et al. 2019, *Towards Linking Hummock-Hollow Microtopography to Peatland Functional Dynamics* | [DOI](https://doi.org/10.5194/bg-16-3491-2019) | `papers/2019-moore-peatland-hummock-hollow-doi-10.5194-bg-16-3491-2019.pdf` | `b97f86a15f071d59e81ba8e397be2aa67ceb173e1f201ca9b96d0de33d3e34bf` | Full paper |
| Moore et al. 2019 SfM DEM corpus | [Zenodo record](https://doi.org/10.5281/zenodo.2545675) | `datasets/zenodo-2545675-moore-peatland-sfm-v1.zip` | `044413bb87171776b409172d29fbde16341b228dc43b694cffe9f02a88640a67` | Archive inventory inspected: centimeter DEMs, point clouds, transects, and analysis scripts |
| Noumonvi et al. 2025, *HuHoLa* | [DOI](https://doi.org/10.1016/j.ecolmodel.2025.111212) | `papers/2025-noumonvi-huhola-doi-10.1016-j.ecolmodel.2025.111212.pdf` | `a951709a9d7ae41c6942422bada0f6168b8a630e263c40b520e034812bdab01f` | Full paper and official Python package. HuHoLa classifies an existing DEM; it is not a generator |
| Ilyasov et al. 2026, *UAS-LiDAR Mapping of Bog Microrelief Enhances Accuracy of Ground-Layer Phytomass Estimation* | [DOI](https://doi.org/10.3390/drones10020121) | `papers/2026-korotkov-uas-lidar-bog-microrelief-doi-10.3390-drones10020121.pdf` | `9fa9b6c021c4d9c086fb0dce0ec0b315d27402004b4277536c35f84c666083a1` | Full 30-page paper; hierarchical classification and field thresholds, not synthesis |
| Nungesser 2003, *Modelling Microtopography in Boreal Peatlands: Hummocks and Hollows* (HOHUM) | [DOI](https://doi.org/10.1016/S0304-3800(03)00067-X) | Not archived; no lawful open full text was found | N/A | Publisher abstract, introduction, method summary, and conclusion only. No architecture claim below relies on inaccessible details |

## Official Repository Revisions

| Project | Official URL | Local path | Revision | Audit finding |
|---|---|---|---|---|
| Guérin terrain amplification | [github.com/eric-guerin/terrain-amplification](https://github.com/eric-guerin/terrain-amplification) | `repos/terrain-amplification/` | `5b83d65315e9845401df78f6a104c21f9bd473d5` | MATLAB release uses exhaustive paired patches and OMP sparsity one; it is nearest-atom transfer, not the learned-dictionary variant in every paper experiment |
| MultiScaleErosion | [github.com/H-Schott/MultiScaleErosion](https://github.com/H-Schott/MultiScaleErosion) | `repos/multiscale-erosion/` | `64fe87d57d0ea904f54eb0ec24d19da08bebd737` | C++/OpenGL research code. Predefined schedule stops at 2048; thermal shader includes simplex perturbation and periodic boundary behavior |
| Implicit Volumetric Terrains | [github.com/aparis69/Implicit-Volumetric-Terrains](https://github.com/aparis69/Implicit-Volumetric-Terrains) | `repos/implicit-volumetric-terrains/` | `2e7bb3ee79b8d8ffabdeb0958a55c6171f2791fe` | Authors state this is a recoding, not the paper implementation. Hard-coded C++ examples and OBJ meshing; no national raster pipeline |
| Topographic roughness | [github.com/tdoane/TopographicRoughness](https://github.com/tdoane/TopographicRoughness) | `repos/topographic-roughness/` | `11f8a2a146a8366c4f5cb2534f0abc9fa170076b` | Figure/research scripts with toy periodic domains and stochastic event placement, not a production synthesizer |
| HuHoLa | [github.com/bravemaster3/huhola](https://github.com/bravemaster3/huhola) | `repos/huhola/` | `fc46c5a222cb76095ffc75e0d28d5ed869cf705e` | Existing-DEM microform identification package, not generation |
| Glacial erosion | [gitlab.inria.fr/landscapes/glacial-erosion](https://gitlab.inria.fr/landscapes/glacial-erosion) | `repos/glacial-erosion/` | `477376419fcc379e98e4998a00bbbc184b0029bc` | Landform-scale simulation; no demonstrated 1 m-to-0.0625 m surface prior |
| Terrain descriptors | [github.com/oargudo/terrain-descriptors](https://github.com/oargudo/terrain-descriptors) | `repos/terrain-descriptors/` | `03e27f6471771e365d473d03557f8053d21e3ff9` | C++ analysis/evaluation implementation, not a generator |
| Argudo FCN terrains | [gitrepos.virvig.eu/oargudo/fcn-terrains](https://gitrepos.virvig.eu/oargudo/fcn-terrains) | `repos/fcn-terrains/` | `a02ada6a7806a528e83842b6ae79c22dcc261c62` in packed refs | Caffe research code and weights; non-overlapping 200-pixel output tiling with replicated borders |

No official public source repository was found for Argudo 2017, Cortial 2020, Scott and Dodgson 2021/2022, or Grenier 2024 after searches of the papers, author/project pages, and common code hosts. Cortial says source would be released after acceptance, but no attributable release was located.

## Integrity Notes

- `metadata/SHA256SUMS` is the machine-readable checksum inventory for the local library.
- `papers/2018-argudo-aerial-terrain-superresolution-doi-10.1111-cgf.13345.pdf` and `papers/argudo_ortho_fcn_2018.pdf` are HTML error/landing pages despite their extensions. The valid full text is the root-level `argudo-2018-orthophoto-fcn-terrain-superresolution.pdf`.
- Some full texts are duplicated under old and canonical names. They were not deleted while other agents were auditing the shared library.
- A local PDF is evidence that the document was archived, not that its method meets the target. Resolution, terrain regime, data provenance, evaluation protocol, and released-code behavior are audited separately.
