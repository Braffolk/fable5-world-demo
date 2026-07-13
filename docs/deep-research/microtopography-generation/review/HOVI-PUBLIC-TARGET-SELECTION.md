# Hovi Public Target Selection

**Date:** 2026-07-13
**Scope:** exact zero-purchase artifact selection for the first Hovi 2024 forest-TLS conversion. This freezes a bounded Jarvselja/Hyytiala subset for qualification work. It does not approve any file as target truth, change the canonical microtopography spec, or authorize training.

## 1. Decision

Retain exactly three 16-scan plots for the first conversion sequence:

| Order | Plot | Role | Why selected |
|---|---|---|---|
| 1 | `HY_SPRUCE4` | development, primary | Multi-position 2019 Hyytiala spruce plot. Forest floor is 10.25% vascular, 48% nonvascular, 0.75% lichen, and 41% intact litter in the supplied four-quadrat summary. It is the best selected development analogue for a moss/litter spruce surface while retaining low vascular cover. |
| 2 | `HY_PINE2` | development, condition stress | Multi-position 2019 Hyytiala pine plot. Forest floor is 17.25% vascular, 5.25% nonvascular, 6.25% lichen, and 71.25% intact litter. It tests whether the converter distinguishes a litter/lichen pine floor from a moss/litter spruce floor instead of imposing one surface policy. |
| 3 | `JS_SPRUCE1` | **blind Estonia transfer check** | Multi-position 2020 Jarvselja spruce plot. Forest floor is 22% vascular, 35.75% nonvascular, 0.25% lichen, and 42% intact litter. Stand and floor composition are close enough to `HY_SPRUCE4` for a meaningful same-family transfer check, but it is a different country/site/campaign and must not tune the converter. |

The initial artifact set is 43 files and **7,241,800,875 bytes** (`7.242 GB`, `6.744 GiB`). It contains the three merged 2 cm-thinned LAZ files, their registration diagnostics, all four forest-floor quadrat photographs and six overview/transect photographs per plot, and seven shared metadata artifacts.

This is the smallest set that tests two materially different forest-floor states in development and preserves one multi-position Estonia check. It is not a release corpus. The three plots are only two independent sites and two campaigns; the canonical spec requires at least three geographically independent qualified sites across at least two campaigns before a regime may emit finest-band geometry.

Do **not** initially download all 13 Jarvselja plots, all 28 Hyytiala plots, or any complete 1.95 TiB release. Do **not** promote the first LAZ conversion to target truth. The downsampled clouds were published for quick preview, merge all 16 views, and erase per-view identity.

## 2. Canonical Record And Access Identities

| Field | Canonical value |
|---|---|
| Dataset title | *A spectral-structural characterization of European temperate, hemiboreal and boreal forests: Laboratory and field data* |
| Dataset DOI | [`10.23729/9a8d90cd-73e2-438d-9230-94e10e61adc9`](https://doi.org/10.23729/9a8d90cd-73e2-438d-9230-94e10e61adc9) |
| Fairdata dataset UUID | `ace2a123-00ff-4944-951e-eddbe209b70c` |
| Human record | `https://etsin.fairdata.fi/dataset/ace2a123-00ff-4944-951e-eddbe209b70c` |
| Machine dataset record | `https://metax.fairdata.fi/v3/datasets/ace2a123-00ff-4944-951e-eddbe209b70c` |
| Machine file inventory | `https://metax.fairdata.fi/v3/files?dataset=ace2a123-00ff-4944-951e-eddbe209b70c&limit=5000` |
| Dataset version | version `1`, published revision `1`, issued `2024-04-25`, metadata modified `2024-04-30T07:09:52Z` |
| Whole release | 4,502 files; 2,141,934,002,132 bytes |
| Data paper | [Rautiainen et al. 2024, ESSD 16, 5069-5097](https://doi.org/10.5194/essd-16-5069-2024) |
| Related airborne record | [`10.23729/c6da63dd-f527-4ec9-8401-57c14f77d19f`](https://doi.org/10.23729/c6da63dd-f527-4ec9-8401-57c14f77d19f) |

Fairdata does not expose a permanent public binary URL in each file record. The stable file identity is the tuple `(dataset UUID, file UUID, full pathname, byte size, SHA-256)`. Retrieval uses:

```text
POST https://etsin.fairdata.fi/api/v3/download/authorize
Content-Type: application/json

{"cr_id":"ace2a123-00ff-4944-951e-eddbe209b70c","file":"<full pathname>"}
```

The response is a short-lived signed `download.fairdata.fi` URL. Never check that token into a ledger or script. Check in the stable tuple above and generate a new token at fetch time. Fetch to `.part`, verify byte count and SHA-256 against Metax, then atomically rename.

The official Metax SHA-256 values for the six selected text metadata files downloaded during this audit matched their retrieved bytes exactly. No point-cloud binary was downloaded.

## 3. License And Rights Evidence

The official Fairdata record and the dataset README independently identify [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode) for the **dataset**, not merely the article. Access type is `open`. The Fairdata record names Aalto University as publisher and one rights holder, with CzechGlobe and Charles University also listed as rights holders for the multi-site release.

CC BY 4.0 grants reproduction, sharing, and adaptation for any purpose, including commercial purposes, subject to attribution, license link, and indication of changes. Therefore internal conversion and training use are allowed. Distributed source-derived rasters and other recognizable data derivatives must retain attribution and change notice.

The license does not state whether trained weights are adaptations of the dataset. Record that question separately. Until the project has a stable attribution policy and legal determination for public weights, use:

- `training_use: allowed`
- `data_derivative_distribution: allowed_with_attribution`
- `public_model_or_checkpoint_distribution: legal_review`
- `cooked_height_derivative_distribution: allowed_with_attribution`, provided no other paired source adds stricter terms

Required attribution must name the dataset title, Hovi et al. dataset DOI, Rautiainen et al. article requested by the README, CC BY 4.0, and that LAAS converted/classified/rasterized the source.

## 4. Frozen Initial Artifact Manifest

All sizes and hashes below are from the official Metax v3 file inventory on 2026-07-13. File UUID is the Fairdata file identity. The manifest includes no `Thumbs.db`, hemispherical canopy photography, hyperspectral products, leaf spectra, Czech plots, single-scan plots, or full E57 archives.

| File UUID | Full pathname | Bytes | SHA-256 |
|---|---|---:|---|
| `d310b0f0-419d-458b-9117-6c6b93faaf53` | `/Laboratory_and_field_data/overview.csv` | 8530 | `4c6e50bc9cdf422b007aaaa7068d8bfadfce962e9449cbe1dd505e8da29a6dbd` |
| `bd88966c-3877-4372-9652-506aa19805d2` | `/Laboratory_and_field_data/Forest_inventory/forest_inventory-matureforest_plots.csv` | 239202 | `d48d28aa5e994141426c6c7d36140ad2c1c4fd76573d8a197978d3aebe36ffeb` |
| `7ad4d8c4-7a91-4d54-9543-3d6d801126cb` | `/Laboratory_and_field_data/Forest_inventory/forest_inventory-summary.csv` | 8169 | `c1ac46ed4a86b469251b4709d308c7ce317b11999d2115899427bffb5c0eb81d` |
| `8d15caee-6754-4c73-8baa-bda1546e5c09` | `/Laboratory_and_field_data/Fractional_cover_forest_floor/fractional_cover.csv` | 6463 | `09488e22c936644f530aae2905c805cf8d75b6e6c73f8949a41d5261f8be0df9` |
| `11d4f91a-6aa6-4bcf-9122-6ffc2dcbfda3` | `/Laboratory_and_field_data/Fractional_cover_forest_floor/Photographs/HY_PINE2/HY_PINE2_quadrat1.JPG` | 6883356 | `3bf13e3fce06d3162436a2328f19579efa9aee39fc02370fefe0af01c9599987` |
| `1ec79c9d-15c4-48b3-9a7b-1128f7d82ec8` | `/Laboratory_and_field_data/Fractional_cover_forest_floor/Photographs/HY_PINE2/HY_PINE2_quadrat2.JPG` | 6889821 | `408c10c6a9623aeb2a5df0b97aff29ee7a4baa070baa7ec8f82bd14ead6a957d` |
| `ddf6f827-4d3d-448e-9783-4e893c551b7a` | `/Laboratory_and_field_data/Fractional_cover_forest_floor/Photographs/HY_PINE2/HY_PINE2_quadrat3.JPG` | 6945285 | `dc3021a5fa1b39c76b8b8afed4b6bec21a7ab18ebcbcd2e6f5ce5dc3f502fce1` |
| `ceca1201-01b8-4ca7-8c91-9a23fe865c34` | `/Laboratory_and_field_data/Fractional_cover_forest_floor/Photographs/HY_PINE2/HY_PINE2_quadrat4.JPG` | 6998823 | `8d88186df8aa9b466315f48ca3cfce10c1df7287dbf4c02b0f2e0c8d0d53a4e2` |
| `22f37fcb-88af-4391-b32f-d53eb9983728` | `/Laboratory_and_field_data/Fractional_cover_forest_floor/Photographs/HY_SPRUCE4/HY_SPRUCE4_quadrat1.JPG` | 6873896 | `7fcf1e82cddff9d49a9e424fc5b535064fdf3c14008f7fd2142bdc6028e07c32` |
| `9aa46e07-cea6-4a42-a69c-42825476feef` | `/Laboratory_and_field_data/Fractional_cover_forest_floor/Photographs/HY_SPRUCE4/HY_SPRUCE4_quadrat2.JPG` | 6962730 | `e732ad30248b54f1f6e52ff78e0b0af5d9dad091c0dfcf8ebbd1a084a21329a6` |
| `5be913af-2262-4d27-b302-b47b7861bfc0` | `/Laboratory_and_field_data/Fractional_cover_forest_floor/Photographs/HY_SPRUCE4/HY_SPRUCE4_quadrat3.JPG` | 6635956 | `bc292bc69c1d3ae66ebb88f2a7afefc6afd45bab08e26a82267aa1fe6b9b506d` |
| `065f6e5c-0a9c-4b85-8d32-7fab0ff96725` | `/Laboratory_and_field_data/Fractional_cover_forest_floor/Photographs/HY_SPRUCE4/HY_SPRUCE4_quadrat4.JPG` | 6907631 | `e8383b5b6f21302cca12ea94b45f1cd828528017136441c2e1f946b6032e398a` |
| `d7409682-dd96-4855-b2ae-314c818d861f` | `/Laboratory_and_field_data/Fractional_cover_forest_floor/Photographs/JS_SPRUCE1/JS_SPRUCE1_quadrat1.JPG` | 7216693 | `f4bb097fb34b909dedae8b197e6a60380992bfcaab0b7b648ff607491a9627bc` |
| `6274e8fe-f4ae-473c-848b-5f66f74f388b` | `/Laboratory_and_field_data/Fractional_cover_forest_floor/Photographs/JS_SPRUCE1/JS_SPRUCE1_quadrat2.JPG` | 7029040 | `e0261ca845156d8f5362aaf96b04443bd7722d2e4aa84600bec8c651e2f3d2aa` |
| `78d30249-3a70-499c-a9bc-05d36c44d4cf` | `/Laboratory_and_field_data/Fractional_cover_forest_floor/Photographs/JS_SPRUCE1/JS_SPRUCE1_quadrat3.JPG` | 7071710 | `6155e0bb2abe964dad7efe4ae41535ff02897c766c57691c6cc7a6d8ff1ba3b5` |
| `984ad3f8-d5fd-4364-bc10-9d3e10cdc093` | `/Laboratory_and_field_data/Fractional_cover_forest_floor/Photographs/JS_SPRUCE1/JS_SPRUCE1_quadrat4.JPG` | 7130904 | `e4721147e9120e2c8a467be90532db63ec9a05466a830dbb644dec5eb0749873` |
| `eaeb9d60-ff39-491b-b73b-526bc1b7d264` | `/Laboratory_and_field_data/Overview_photographs/HY_PINE2/HY_PINE2-corner_NE.JPG` | 6722182 | `1ef6ecc4bc95b23bfc4abeb96ca98a176dda6d2f083cbd2c22d67611069f9e9e` |
| `c827040a-b1ad-420b-af97-e787e323fd9d` | `/Laboratory_and_field_data/Overview_photographs/HY_PINE2/HY_PINE2-corner_NW.JPG` | 6899336 | `c842e05e3cae9caa2898dcc4cebea46c5aeda7c9ebe2f3c9f3a153dc407f3ce8` |
| `e4914872-f082-42e7-96fa-74debffd5107` | `/Laboratory_and_field_data/Overview_photographs/HY_PINE2/HY_PINE2-corner_SE.JPG` | 6957981 | `3b1fef43024e08f78ec9ef709297011598198f95f749aa897f8213449491effb` |
| `94da2df1-cb34-43de-81e0-8e8f0b5babaf` | `/Laboratory_and_field_data/Overview_photographs/HY_PINE2/HY_PINE2-corner_SW.JPG` | 6871649 | `ac8cb7846c4b1937b2377e0411081fda4dfdc4bc89cb7ad7f94f16e701fdca08` |
| `05c66196-1f0f-4a3c-b91d-7fd84fbe6070` | `/Laboratory_and_field_data/Overview_photographs/HY_PINE2/HY_PINE2-transect_from_E.JPG` | 6993492 | `11afaf77576f3a4d3b42dd23ee868def859dc873dea2d9d1c6ae34fc58b48dbc` |
| `17dcc69f-5b69-4160-8201-ab8098d61eb0` | `/Laboratory_and_field_data/Overview_photographs/HY_PINE2/HY_PINE2-transect_from_W.JPG` | 6870265 | `7545278c46dc39ff73097021d05840e1eacb39ac4a9a7c21e723cc7342248fd6` |
| `5f1ed433-49d1-4f79-832f-29338fe7083e` | `/Laboratory_and_field_data/Overview_photographs/HY_SPRUCE4/HY_SPRUCE4-corner_NE.JPG` | 6897022 | `889e2d412e14edf6eb9a1992b2579f306aee119ca5cd9ed42ad569dc60e16ca6` |
| `b9c1b065-cfb0-46ed-be66-f675f1c12211` | `/Laboratory_and_field_data/Overview_photographs/HY_SPRUCE4/HY_SPRUCE4-corner_NW.JPG` | 6850667 | `75a11679cd3a624c06f06942bf3366296017f29eb18694159aca54cf4af6fbdb` |
| `f40a324d-1d1b-4564-8fa9-7773d67d3fcc` | `/Laboratory_and_field_data/Overview_photographs/HY_SPRUCE4/HY_SPRUCE4-corner_SE.JPG` | 6903512 | `11e5ae141047c58d4f4a0bd5da4392e1664d507774cd0b440c4ca33f9d44222c` |
| `9f61d166-9a93-4b73-a001-58abd0b7f5ac` | `/Laboratory_and_field_data/Overview_photographs/HY_SPRUCE4/HY_SPRUCE4-corner_SW.JPG` | 6737735 | `a0c401e0ca8032d5b6261bf76b9a4861f39c72e03374a2895da78e4677a32792` |
| `8a13d47d-03d9-4cb6-85db-3711e196affd` | `/Laboratory_and_field_data/Overview_photographs/HY_SPRUCE4/HY_SPRUCE4-transect_from_E.JPG` | 7027360 | `8f012b49b38323bef340586c79bb7e52c584f9b5bb4f0ed99435f761f524f896` |
| `60dbfb47-fddd-4a51-9d2a-468b093f86cb` | `/Laboratory_and_field_data/Overview_photographs/HY_SPRUCE4/HY_SPRUCE4-transect_from_W.JPG` | 6994445 | `114e423a27e28cf84bac98865bf13aeb1a04e54296b6554a27d3aa73caacf80c` |
| `cb8d3034-57ec-4d79-9765-205698db6caa` | `/Laboratory_and_field_data/Overview_photographs/JS_SPRUCE1/JS_SPRUCE1-corner_NE.JPG` | 6300173 | `2f18db0c324c314c9f7217192c46a84cb1127732359f3cfe771011df24576907` |
| `1449d922-9c76-4ca5-966f-a95b57082e1f` | `/Laboratory_and_field_data/Overview_photographs/JS_SPRUCE1/JS_SPRUCE1-corner_NW.JPG` | 6241711 | `e27872696d6d41955ad49c69f253bd3a9ea1791ca7adc405136aac1c393bf576` |
| `6525807b-ebec-45c5-9ab9-0c5cdd51f4c5` | `/Laboratory_and_field_data/Overview_photographs/JS_SPRUCE1/JS_SPRUCE1-corner_SE.JPG` | 6147434 | `5009ba7b1373d9c3f70303bbcc36176d91b40323ad2149934e4785daa3a546c9` |
| `21f10013-cf3b-4988-a5fc-b9727672fb86` | `/Laboratory_and_field_data/Overview_photographs/JS_SPRUCE1/JS_SPRUCE1-corner_SW.JPG` | 6336562 | `2a18129b872bea7eed7ea70a1fababb6df773c973d0178d60c7c0396776d683b` |
| `48701985-5e9b-4b81-a62a-527ef3edf8ca` | `/Laboratory_and_field_data/Overview_photographs/JS_SPRUCE1/JS_SPRUCE1-transect_from_E.JPG` | 6634242 | `e4410ddbb997488219a0b3750648d577f7e8aac893b9ab27b93831495f9af618` |
| `aa1f5032-3599-42ce-8aad-cf22acb6befa` | `/Laboratory_and_field_data/Overview_photographs/JS_SPRUCE1/JS_SPRUCE1-transect_from_W.JPG` | 6233523 | `2e0c7ce28b22d652638ea6530a84eeb82a8d53c21efa51f0c01e4a34faedc916` |
| `9a2ec047-2478-4f3b-b1b1-1d137bbd51a3` | `/Laboratory_and_field_data/README/README.md` | 48765 | `51aa09247336746c093e45499bac6d104652f631df37cd520f3d7556f34423de` |
| `c9fb7871-6296-40b4-9ff5-dfd979ef7031` | `/Laboratory_and_field_data/README/figures/fig2.png` | 89258 | `9ea469ae45e39a75d0e405707a72450817917f0dc923eafe8c6b4e15ffc43464` |
| `4896df8c-35e7-40a3-9936-e8327eb424ce` | `/Laboratory_and_field_data/Terrestrial_laser_scanning/terrestrial_laser_scanning-metadata.csv` | 6047 | `28f0c28d257913b9854960250b8a53f79d549185e6ef4b107da315ba7a556ae9` |
| `b31184d9-dffe-4b13-a1c6-91cac1be2204` | `/Laboratory_and_field_data/Terrestrial_laser_scanning/Point_clouds_downsampled/HY_PINE2-downsampled.laz` | 1775923667 | `d8e28a3142bb3646ebd9dd355b11f9377d9b064bb3277ccd07e4b79be14c93e4` |
| `f4a40d82-2d7a-4f92-a946-ff806a55a096` | `/Laboratory_and_field_data/Terrestrial_laser_scanning/Point_clouds_downsampled/HY_SPRUCE4-downsampled.laz` | 2740487787 | `130778345d4ee5b273ef4abe8fedab4d4e23b20238d50ce34fa1be5b17970852` |
| `616a365b-7270-4c64-8968-ca09316f27dd` | `/Laboratory_and_field_data/Terrestrial_laser_scanning/Point_clouds_downsampled/JS_SPRUCE1-downsampled.laz` | 2520475935 | `410db81b8dfb9cf1af4dd946b0ce9278babdbf59740e07c03196b7ca5d5f9b88` |
| `72692326-5432-4ec7-91ac-c845d2b311e1` | `/Laboratory_and_field_data/Terrestrial_laser_scanning/Registration_diagnostics/HY_PINE2-registration_diagnostics.txt` | 98132 | `70cbcced12dc5488d4922b76f915f9d606a3add342ae2f0ac6637c60288b5afd` |
| `077ac310-f24d-45d2-9ae6-1bc8c6e5104a` | `/Laboratory_and_field_data/Terrestrial_laser_scanning/Registration_diagnostics/HY_SPRUCE4-registration_diagnostics.txt` | 122246 | `56ee8a0cd10ac2d444751724b7d652e33f85fcb1f2a272803d8ac3b5d7c994fc` |
| `c4997d6e-696f-4a4b-a397-dedee52c00af` | `/Laboratory_and_field_data/Terrestrial_laser_scanning/Registration_diagnostics/JS_SPRUCE1-registration_diagnostics.txt` | 121538 | `f94ae8e9bb62646c18110d7e493e2a79c44b40189155b369ea8c0ea10074e4ee` |

### 4.1 Exact byte accounting

| Tranche | Files | Bytes | GiB | Disposition |
|---|---:|---:|---:|---|
| Shared metadata | 7 | 406,434 | 0.0004 | fetch first |
| `HY_PINE2` LAZ, diagnostic, and 10 semantic/context photos | 12 | 1,845,053,989 | 1.7184 | development stress conversion |
| `HY_SPRUCE4` LAZ, diagnostic, and 10 semantic/context photos | 12 | 2,809,400,987 | 2.6165 | primary development conversion |
| `JS_SPRUCE1` LAZ, diagnostic, and 10 semantic/context photos | 12 | 2,586,939,465 | 2.4093 | retain sealed, then blind conversion |
| **Initial total** | **43** | **7,241,800,875** | **6.7445** | approved first-conversion selection |

## 5. Conditional Full-Scan Qualification Tranches

The initial LAZ files are merged and thinned to average 0.02 m spacing. They are sufficient to test parsing, geometric transforms, candidate-surface extraction, support-map plumbing, and obvious contamination. They are insufficient to measure view diversity, occlusion, thinning damage, or per-scan disagreement. Target-truth qualification therefore requires full individual scans for at least development and blind plots.

| Gate order | File UUID | Path | Bytes | SHA-256 | Rule |
|---|---|---|---:|---|---|
| Q1 | `83b09928-c296-4719-982d-0bbd9451405a` | `/Laboratory_and_field_data/Terrestrial_laser_scanning/Point_clouds_full/HY_SPRUCE4-full.zip` | 47,345,435,519 | `008540c96bcbea1eb98128d92df844ee6346c11e1ec99ad626ba3c2e6a0caa1e` | Fetch only if the primary downsampled conversion has enough plausible surface support to justify a view-aware audit. Initial + Q1 is 54,587,236,394 bytes (`50.838 GiB`). |
| Q2 | `10b78236-d890-40eb-862c-446c8dd9aa28` | `/Laboratory_and_field_data/Terrestrial_laser_scanning/Point_clouds_full/JS_SPRUCE1-full.zip` | 47,328,414,906 | `185b191f4f8236215963381d22a706c30ebf180c8f5756d983d26799d5da4ea8` | Fetch only after conversion policy and thresholds are frozen. Initial + Q1 + Q2 is 101,915,651,300 bytes (`94.916 GiB`). Keep blind until the preregistered evaluation runs. |

Do not fetch `HY_PINE2-full.zip` in the first qualification cycle. Its exact record is file UUID `37dbe7d3-e4e1-4e11-9d7e-6fe20acdb8d0`, 46,739,278,738 bytes, SHA-256 `8ab3fb0f4a0f838fa5bb2ff059d95f20674ef15f8750a10f7a662e349b1a67c6`. It becomes eligible only if the merged pine conversion shows a distinct, adequately supported surface family whose full-view audit could change the method decision.

## 6. Campaign And Geographic Split

Use sites and campaigns, not patches, as the independence units.

| Field | `HY_SPRUCE4` | `HY_PINE2` | `JS_SPRUCE1` |
|---|---|---|---|
| `site_id` | `hovi.hyytiala` | `hovi.hyytiala` | `hovi.jarvselja` |
| `campaign_id` | `hovi.hyytiala.2019` | `hovi.hyytiala.2019` | `hovi.jarvselja.2020` |
| country | Finland | Finland | Estonia |
| plot center EPSG:4326 | `24.3134095846, 61.8453841107` | `24.2873093036, 61.8551937506` | `27.3270855675, 58.2740645397` |
| plot center EPSG:25835 | `358599.84, 6859879.617` | `357272.325, 6861028.803` | `519185.78, 6459269.854` |
| TLS date | `2019-07-04` | `2019-07-15` | `2020-06-24` |
| forest-floor photo date | `2019-07-26` | `2019-07-25` | `2020-06-27` |
| scan protocol | 16 scans in 30 x 30 m layout | 16 scans in 30 x 30 m layout | 16 scans in 30 x 30 m layout |
| scan angular pitch | 23 mm at 100 m, approximately 0.23 mrad | same | same |
| split | development | development condition stress | blind Estonia transfer |

The plot centers are geolocated to better than 1 m by TLS-to-airborne matching, but the README warns that absolute horizontal mismatch can be several decimeters. Local TLS z is gravity-aligned; its offset to orthometric or ellipsoidal height was not supplied. Do not interpret the transformed cloud as EH2000 without a separately verified vertical tie.

`HY_SPRUCE4` and `HY_PINE2` are different plots but the same site and campaign. They count as **one** independent site and **one** campaign for release gating. `JS_SPRUCE1` must remain outside training, threshold selection, semantic-policy selection, and converter debugging. Taevaskoda remains excluded from all of those activities.

## 7. Measurements Required Before Target Truth

The initial role of every selected cloud is `raw_candidate`. Promotion requires the canonical target gate and all measurements below.

### 7.1 Geometry and observation support

1. Inspect LAZ/E57 headers and record point count, point format, coordinate scales/offsets, dimensions, return fields, color/intensity fields, scan identities, invalid values, and exact extents.
2. Reconstruct individual E57 scans with published transforms and quantify pairwise stable-surface disagreement by range, incidence, and scan position. Cyclone's 0.9 cm translation tolerance is not terrain error.
3. Measure per-cell unique-view count, angular diversity, nearest measured distance, local point spacing, occlusion, interpolation radius, and scan-edge effects at 0.025, 0.05, 0.0625, 0.125, 0.25, and 1 m.
4. Compare full-scan and 2 cm-thinned merged surfaces to quantify which wavelengths and amplitudes the preview product removes, biases, or aliases.
5. Establish an empirical transfer function or controlled-feature recovery curve over 0.125-1 m. Nominal scan pitch and average point spacing do not satisfy this requirement.
6. Establish independent horizontal and vertical error distributions. Absolute z must be tied to the Estonia/Finnish condition stack's height datum or kept local for residual-only evidence.

### 7.2 Surface semantics

1. Create explicit human-QA labels for mineral soil, stable moss/nonvascular surface, exposed root, clast, intact litter, decomposed litter, deadwood, vascular vegetation, stem, water/wet reflection, built/object, and unknown.
2. State the intended canonical surface for each label. Do not silently mix bare-earth mineral soil with top-of-moss, top-of-litter, roots, or clasts.
3. Use the four quadrat photographs and two transect photographs as sparse semantic evidence only. They are not dense point labels and were acquired 3-22 days after TLS for these plots.
4. Measure semantic confusion on manually labeled point/patch samples and propagate uncertainty into valid/support masks.
5. Forbid lowest-point-only target creation. Lowest returns can be litter, low plant, water artifact, undercut root, or an occlusion outlier.

### 7.3 Physical analogue and condition pairing

1. Pair each plot with the closest-date airborne Hovi artifact, national DTM/ALS, orthophoto, soil profile/map, surficial geology, bedrock, hydrology/wetness, stand history, and disturbance/management evidence.
2. Complete a `physical-analogue-v1` comparison for Hyytiala-to-Estonia transfer: substrate/soil, freeze-thaw and climate, hydrology, organic surface, vegetation, management, process, morphology scale, and observation process.
3. Treat unmatched factors as OOD. Species name and geographic proximity are not sufficient transfer evidence.
4. Keep `JS_SPRUCE1` blind until the converter and all thresholds are frozen. The first Estonia outcome may reject transfer; it may not be used to tune then reported as validation.

### 7.4 Qualification result

Promotion to `exemplar` requires target-valid cells with measured support, semantics, error, and band transfer. Promotion to `production_target` additionally requires paired Estonia inputs and a legally and scientifically valid transfer/release role. The Hovi subset alone cannot satisfy the three-site release minimum even if all three plots pass technical conversion.

## 8. Machine-Ledger Mapping And Unknowns

The checked-in `library/sources.schema.json` is a primary-source ledger, while Section 9.5 of the canonical spec requires a finer target registry. Do not force plot/file facts into one paper-level artifact record. The later implementation needs at least:

1. One dataset source record, proposed ID `dataset.hovi-2024-forest-field`.
2. One linked paper record for DOI `10.5194/essd-16-5069-2024`.
3. One target-registry record per plot/campaign.
4. One artifact record per downloaded file with local cache path and locally verified SHA-256.

The source-ledger facts already resolvable are:

| Schema area | Resolved value |
|---|---|
| `kind` | `dataset` |
| `citation` | official title, seven creators, year 2024, Aalto University dataset, dataset DOI |
| `identifiers.canonical_url` | dataset DOI or Etsin record; related paper and airborne dataset IDs must be linked |
| `access` | `open_full_text_online`/open data; checked 2026-07-13 through official Metax/Etsin |
| `review` | `primary_metadata`; official dataset record, README, CSV schema, and file manifest inspected. This permits artifact selection, not terrain-transfer claims. |
| `licensing` | verified `CC-BY-4.0`, applies to data; training allowed, public weight treatment still requires legal review |
| `software` | `not_applicable` |
| `data_and_checkpoints.data_status` | `open_not_downloaded` for selected point clouds; small audit metadata was fetched only to verify official content/hashes |
| `demonstration` | Leica P40 forest TLS, 16- or single-scan protocols, multi-site European forests; downsampled preview averages 0.02 m spacing |
| `evidence` | direct dense forest observations; prohibited inference is that nominal pitch, co-registration tolerance, or a merged LAZ is target-scale ground truth |

The following machine fields remain unknown and must stay explicit rather than inferred:

- stable repository record IDs until the ledger owner assigns them;
- local artifact paths and local `metadata/SHA256SUMS` entries for all unfetched files;
- exact point count, LAZ/E57 schema, coordinate quantization, attributes, scan IDs, and valid bounds;
- exact plot footprint rather than the README's approximate 35 x 35 m manual crop;
- local-z offset and uncertainty relative to orthometric/EH2000 height;
- independent terrain horizontal/vertical error distributions;
- post-classification measured ground/stable-organic point support;
- effective resolution and transfer function by wavelength/amplitude;
- visibility, interpolation, semantic, support, and human-QA masks;
- mineral soil, peat, root, clast, litter, deadwood, vascular vegetation, and water point semantics;
- exact soil profile, surficial geology, bedrock, hydrology, moisture/weather, disturbance, and management state;
- exact co-temporal airborne/national condition artifact IDs and hashes;
- physical-analogue qualification and condition-stratified OOD bound;
- final role, transfer ceiling, and release qualification result;
- public model/checkpoint legal treatment and attribution implementation.

Until those fields are resolved, set `role=raw_candidate`, `qualification_status=unqualified`, and `transfer_ceiling=none`.

## 9. Explicit Disqualifiers

Reject or defer an artifact from the first conversion when any condition below applies:

- **Single-scan shortcut:** `HY_PINE8` through `HY_PINE13`, `HY_SPRUCE8`, `HY_SPRUCE9`, `JS_BIRCH3`, `JS_MIXED4`, `JS_MIXED5`, and `JS_SPRUCE2` use one central scan. Their small files are attractive for I/O smoke tests but cannot establish multi-view forest-floor visibility or occlusion. They may become later negative controls, never substitutes for the selected multi-position plots.
- **Registration exception:** `JS_MIXED1` missed Cyclone's stated registration tolerance. The authors' visual assessment is useful but not an independent target-error measurement; exclude it from the first qualification.
- **Temporal semantic mismatch:** `HY_BIRCH2`, `HY_BIRCH3`, `HY_SPRUCE3`, and `HY_MIXED2` use 2018 floor measurements with 2019 TLS. Do not use them for the first paired semantic audit.
- **Missing context:** `JS_MIXED2` lacks overview photographs. It can be reconsidered after the semantic protocol works, not in the smallest first set.
- **Merged-LAZ promotion:** any result based only on a 2 cm-thinned merged LAZ remains `raw_candidate` regardless of visual smoothness or raster pitch.
- **Unsupported cell:** fewer than the preregistered unique-view/support requirement, excessive interpolation distance, semantic ambiguity, water/vegetation contamination, or error comparable to claimed band energy invalidates the cell.
- **Datum ambiguity:** absolute heights without a verified z tie cannot supervise corrected EH2000 height.
- **Plot-to-site inflation:** patches or plots from one research site do not become independent sites. Hovi alone cannot meet the three-site minimum for release.
- **Blind leakage:** inspecting `JS_SPRUCE1` output to alter the converter, thresholds, or semantic policy permanently disqualifies it as the blind check for that iteration.
- **Generic forest transfer:** no selected plot authorizes all Estonia forests, peat, till, shore, agriculture, alvar, outcrop, or disturbed ground.

## 10. Retain And Convert Order

1. **Retain/fetch shared metadata first:** the seven artifacts in Section 4, verify exact hashes, and register dataset/license identities.
2. **Retain/fetch all selected semantic/context photos:** they are cheap and are necessary to prevent geometry-only surface labeling.
3. **Convert `HY_SPRUCE4-downsampled.laz` first:** this is the primary parser/transform/support/semantic-mask development artifact.
4. **Convert `HY_PINE2-downsampled.laz` second:** the result must either preserve its different litter/lichen state or abstain; it must not be forced through spruce-tuned rules.
5. **Gate Q1:** if neither development LAZ has plausible measured surface support, stop Hovi conversion and do not spend 47 GB on full scans. If support is plausible, fetch and audit only `HY_SPRUCE4-full.zip` first.
6. **Freeze converter and thresholds:** finish the view-aware Q1 audit, preregister error/support/semantic thresholds, and record all implementation identities before touching blind output.
7. **Convert `JS_SPRUCE1-downsampled.laz` blind:** score the frozen policy without tuning. Failure means Estonia transfer is unsupported, not that the holdout should be absorbed into development.
8. **Gate Q2:** fetch `JS_SPRUCE1-full.zip` only if the blind merged-cloud result is promising enough to justify a full-view qualification. Evaluate once with the frozen protocol.
9. **Final disposition:** retain passing measured cells as `exemplar` or, only after every target/condition/legal gate, `production_target`; retain failures with explicit abstention reasons; never convert uncertainty into decorative height.

This order is decisive: `HY_SPRUCE4` development, `HY_PINE2` stress, `JS_SPRUCE1` blind. Full scans are conditional, one plot at a time. No Hovi-derived owner is released from this first subset alone.
