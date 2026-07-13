# ML Source Manifest

Access date: 2026-07-13. Paths are relative to `docs/deep-research/microtopography-generation/library/`.

This manifest distinguishes papers actually read in full from repositories inspected for implementation facts. A local file is not presumed valid because its extension is `.pdf`; the exceptions section records two HTML responses with misleading filenames.

## Core Terrain And DEM Papers

| Work | Canonical source | Local full text | SHA-256 | Read scope |
|---|---|---|---|---|
| Argudo, Chica, Andujar 2018, *Terrain Super-resolution through Aerial Imagery and Fully Convolutional Networks* | [DOI](https://doi.org/10.1111/cgf.13345) | `argudo-2018-orthophoto-fcn-terrain-superresolution.pdf` | `df62eb57644a03018b5ea8131fd0774cf113292e4366bff34333d8fe3b3f64a5` | Full paper, especially Sections 3-6 |
| Liu et al. 2026, *DEM Super-resolution Guided by High-resolution Remote Sensing Images Using Multitask Learning* | [DOI](https://doi.org/10.1016/j.jag.2026.105099) | `papers/2026-liu-gsrmtl-doi-10.1016-j.jag.2026.105099.pdf` | `2460b2612059deed84b2b4f632dfa279b76ee8e74d5b7763077f76f50f7dcccf` | Full paper and code paths for degradation, loss, and Ontario loader |
| Zhao et al. 2019, *Multi-Theme Generative Adversarial Terrain Amplification* | [DOI](https://doi.org/10.1145/3355089.3356553), [author-hosted ResearchGate copy](https://www.researchgate.net/publication/337118022_Multi-theme_generative_adversarial_terrain_amplification) | Not archived; ACM and author-hosted binary endpoints returned HTTP 403, OpenAlex content endpoint returned HTTP 401 | N/A | Full author-hosted text read, including Appendix A; official repository inspected |
| Demiray et al. 2021, *Deep Learning-Based Super-Resolution for Digital Elevation Models* | [arXiv](https://arxiv.org/abs/2004.04788) | `papers/2021-demiray-dsrgan-arxiv-2004.04788.pdf` | `9590ba5872f64cf980dd6dd6a634e666bd9fb1f81a46492e5b5b062efa5f0bfc` | Full paper |
| Panangian and Noori 2024, *Real-GDSR: Real-World Guided DSM Super-Resolution* | [DOI](https://doi.org/10.5194/isprs-annals-X-2-2024-185-2024) | `papers/2024-panangian-real-gdsr-doi-10.5194-isprs-annals-X-2-2024-185-2024.pdf` | `eee1804367e6011134a72c9ad543e8003df9d7c25e950142eca3687029e89675` | Full paper |
| Lochner et al. 2023, *Interactive Authoring of Terrain using Diffusion Models* | [DOI](https://doi.org/10.1111/cgf.14941) | `papers/lochner_diffusion_2023.pdf` | `bfb0ec1673e059a6f9a1945f106559148e188050005e7973df3148880cc08a29` | Full paper |
| Perche et al. 2023, *StyleDEM* | [arXiv](https://arxiv.org/abs/2304.09626) | `papers/2023-perche-styledem-arxiv-2304.09626.pdf` | `3d9b8dfb6445b21c1468f4ef3f0e872a2a561a0f4224ad4af25caee7a496e770` | Full paper |
| Goslin et al. 2026, *InfiniteDiffusion* | [arXiv](https://arxiv.org/abs/2512.08309) | `papers/2026-goslin-infinitediffusion-arxiv-2512.08309.pdf` | `49368e8fb5a3e86edfe3b86d064941f93ba70cc4fdb9c097a4aa9fca50028309` | Full paper and both official repositories |
| Dhaouadi et al. 2026, *GrounDiff* | [arXiv](https://arxiv.org/abs/2511.10391) | `papers/2026-dhaouadi-groundiff-arxiv-2511.10391.pdf` | `76398136f46c4acb3e8c442eec5248d7db96fbdcb0a6029b57c5e3381b94c035` | Full paper including supplement |
| Higo et al. 2025, *TerraFusion* | [arXiv](https://arxiv.org/abs/2505.04050) | `papers/2025-higo-terrafusion-arxiv-2505.04050.pdf` | `7635dc3dad7c8ee813d821aa5953bc98ddc77ff559cc394006720fe4d550b9ba` | Full paper |

## Diffusion, Conditioning, And Tiling Components

| Work | Canonical source | Local full text | SHA-256 | Read scope |
|---|---|---|---|---|
| Wang, Yu, Zhang 2023, DDNM | [arXiv](https://arxiv.org/abs/2212.00490) | `papers/2023-wang-ddnm-arxiv-2212.00490.pdf` | `2eb94d7e487039d3575713586699534a2a775f98b61ba2f5692569391f37fa95` | Full paper, equations and SR operator implementation |
| Bar-Tal et al. 2023, MultiDiffusion | [arXiv](https://arxiv.org/abs/2302.08113) | `papers/2023-bartal-multidiffusion-arxiv-2302.08113.pdf` | `306eb0cffbaba356daa7fb2ae9d692b4fc63bb1b09612372602b7f1fa6dc0616` | Full paper and panorama code |
| Zhang, Rao, Agrawala 2023, ControlNet | [arXiv](https://arxiv.org/abs/2302.05543) | `papers/2023-zhang-controlnet-arxiv-2302.05543.pdf` | `1539c841018cea52b7f0d4a108e6e698d2a5ba2a26c0a665df81d5da79f88170` | Full paper and official repository |
| Sharma et al. 2024, EarthGen | [arXiv](https://arxiv.org/abs/2409.01491) | `papers/2024-sharma-earthgen-arxiv-2409.01491.pdf` | `5a714a15e184ec4d5e7580d32aa64be0f790de0cef5d35eb825cbd4d330d842d` | Full paper, cascade and Mixture-of-Diffusers sections |
| Gao et al. 2023, Implicit Diffusion Models | [arXiv](https://arxiv.org/abs/2303.16491) | `papers/2023-gao-implicit-diffusion-models-arxiv-2303.16491.pdf` | `1ab98a456ec58c0f54dcb71ab893b8d40b12fcdddb1f8cc284dc7d33bfcd3605` | Full paper |
| Saharia et al. 2021, SR3 | [arXiv](https://arxiv.org/abs/2104.07636) | `papers/2021-saharia-sr3-arxiv-2104.07636.pdf` | See repository library checksum inventory | Method and experiments |
| Li et al. 2021, SRDiff | [arXiv](https://arxiv.org/abs/2104.14951) | `papers/2021-li-srdiff-arxiv-2104.14951.pdf` | See repository library checksum inventory | Method and experiments |
| Kawar et al. 2022, DDRM | [arXiv](https://arxiv.org/abs/2201.11793) | `papers/2022-kawar-ddrm-arxiv-2201.11793.pdf` | See repository library checksum inventory | Inverse-problem formulation |
| Chung et al. 2023, DPS | [arXiv](https://arxiv.org/abs/2209.14687) | `papers/2023-chung-dps-arxiv-2209.14687.pdf` | See repository library checksum inventory | Likelihood guidance formulation |
| Song et al. 2024, ReSample | [arXiv](https://arxiv.org/abs/2307.08123) | `papers/2024-song-resample-arxiv-2307.08123.pdf` | See repository library checksum inventory | Latent inverse-problem mechanism |
| Blau and Michaeli 2018 | [arXiv](https://arxiv.org/abs/1711.06077) | `papers/2018-blau-perception-distortion-arxiv-1711.06077.pdf` | See repository library checksum inventory | Full theory paper |
| Wang et al. 2024, OFTSR | [arXiv](https://arxiv.org/abs/2412.09465) | `papers/2024-wang-oftsr-arxiv-2412.09465.pdf` | See repository library checksum inventory | Architecture review; natural-image transfer only |

## Ground-Data And Class Evidence

| Work | Canonical source | Local artifact | SHA-256 | Read scope |
|---|---|---|---|---|
| Noumonvi et al. 2025, HuHoLa | [DOI](https://doi.org/10.1016/j.ecolmodel.2025.111212) | `papers/2025-noumonvi-huhola-doi-10.1016-j.ecolmodel.2025.111212.pdf` | `a951709a9d7ae41c6942422bada0f6168b8a630e263c40b520e034812bdab01f` | Full paper |
| Luke 2019, Lapinjarvi TLS data description | [Dataset documentation](https://etsin.fairdata.fi/dataset/cdd5b545-fcd4-4e88-bc61-39c0c982f0ff) | `papers/2019-luke-lapinjarvi-tls-data-description.pdf` | `5b7b80c192918e6edc35be207988ef90c5500b521c75ab8d959d4ef2c52fa5e1` | Full data description |
| Pitkanen et al. 2019, Lapinjarvi stem/TLS study | [DOI](https://doi.org/10.1016/j.isprsjprs.2018.11.027) | `papers/2019-pitkanen-lapinjarvi-tls-stems-doi-10.1016-j.isprsjprs.2018.11.027.pdf` | See repository library checksum inventory | Full paper |
| Kmoch et al. 2021, EstSoil-EH | [DOI](https://doi.org/10.5194/essd-13-83-2021) | `papers/2021-kmoch-estsoil-eh-doi-10.5194-essd-13-83-2021.pdf` | See repository library checksum inventory | Dataset scope, attributes, resolution, uncertainty |

## Official Repository Revisions

| Project | Official URL | Local path | Revision | License finding |
|---|---|---|---|---|
| GATA | [github.com/EA-GAT/gata](https://github.com/EA-GAT/gata) | `repos/GATA/` | `0c042f711dae7b6721485edcd6e454739b071a49` | Primary EA BSD-like 3-clause terms plus separately attributed bundled components; read the full license before reuse |
| DDNM | [github.com/wyhuai/DDNM](https://github.com/wyhuai/DDNM) | `repos/DDNM/` | `00b58eac7843a4c99114fd8fa42da7aa2b6808af` | No license file found at audited revision; do not copy code without clarification |
| ControlNet | [github.com/lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) | `repos/ControlNet/` | `ed85cd1e25a5ed592f7d8178495b4483de0331bf` | Apache-2.0 |
| MultiDiffusion | [github.com/omerbt/MultiDiffusion](https://github.com/omerbt/MultiDiffusion) | `repos/MultiDiffusion/` | `69bcdcef437dfdbf48c53624d6bf6f397b5f4894` | No license file found at audited revision; treat code as all-rights-reserved unless clarified |
| Terrain Diffusion | [github.com/lineadelucas/terrain-diffusion](https://github.com/lineadelucas/terrain-diffusion) | `repos/terrain-diffusion/` | `82a0431281f21a6ec3d691a12ee61525de5b0790` | MIT |
| Infinite Tensor | [github.com/lineadelucas/infinite-tensor](https://github.com/lineadelucas/infinite-tensor) | `repos/infinite-tensor/` | `070ca95b2ed7122740e35176113774553d9a67c0` | MIT |
| Implicit Diffusion Models | [github.com/Ree1s/IDM](https://github.com/Ree1s/IDM) | `repos/IDM/` | `d224c0ef885168ebd1d19ab4cf4a66a1939ffef0` | No license file found at audited revision |
| DEMSR / GSRMTL | [github.com/YuspaceCraft/DEMSR](https://github.com/YuspaceCraft/DEMSR) | `repos/DEMSR/` | `7bd0c75d1c4e0e00b9cc74214409af7043db0051` | No license file found; no released checkpoint or dataset found |
| Argudo FCN terrains | [gitrepos.virvig.eu/oargudo/fcn-terrains](https://gitrepos.virvig.eu/oargudo/fcn-terrains) | `repos/fcn-terrains/` | Downloaded source archive; no trustworthy Git revision metadata in local copy | No license file found in local archive |

## Invalid And Duplicate Artifacts

- `papers/2018-argudo-aerial-terrain-superresolution-doi-10.1111-cgf.13345.pdf` is HTML, not a PDF. SHA-256: `026a2b234a4c856cfb33379537dd4e5e2bca4580cef93ad5e7d466bb1265822e`.
- `papers/argudo_ortho_fcn_2018.pdf` is also HTML, not a PDF. SHA-256: `5d65e29a140470187a35147c2172f0d099a70e4ae538dfad7cbe1e9629433937`.
- The valid Argudo 2018 full text is `argudo-2018-orthophoto-fcn-terrain-superresolution.pdf` at the library root.
- Several papers appear both under descriptive canonical names and older short names. The manifest names one read artifact; duplicates should be deduplicated only after checking hashes and without deleting another active agent's files.

## Retrieval Gap

The GATA full text was read through the author-hosted ResearchGate page and checked against the official code repository, but a binary PDF could not be archived in this environment: ResearchGate and ACM returned HTTP 403 and OpenAlex's content endpoint returned HTTP 401. This is a reproducibility gap, not a claim that the paper was unread. A later archive pass should add the lawful author/publisher PDF without changing the paper notes unless the binary reveals a discrepancy.
