# Baltic Public Target-Data Hunt

**Date:** 2026-07-13
**Scope:** zero-cost, legally accessible high-resolution terrain or ground-geometry data from Latvia, Lithuania, and relevant cross-Baltic programs
**Purpose:** determine whether neighboring-country data can support Estonia microtopography synthesis without commissioning new surveys

## Conclusion

Public Baltic-region data materially improves the no-budget plan, but it does not eliminate the Estonia evidence requirement.

Two datasets are immediately downloadable with clear reuse rights and are worth ingesting now:

1. **Biała Góra coastal cliff, Poland:** ten raw UAV-LiDAR epochs over approximately 1 km, 27.402 GB compressed, CC BY 4.0. It is strong morphology and temporal-change material for cliffs, mass wasting, till/moraine, and disturbed coastal ground, but it is unclassified and publishes no independent positional-error or ground-support evidence.
2. **ForestSemantic-MS, Finland:** six dense, manually annotated boreal-forest point clouds, 131.1 MB, CC BY 4.0. It is strong forest-floor conversion and semantic-pretraining material, but total point density is not ground density, the released clouds are normalized rather than an absolute surveyed DTM, and no target-scale ground transfer/error measurement is published.

Three Lithuania/Latvia UAV studies are valuable **zero-cost author-request leads** for dunes, coastal cliff/moraine, shoreline, and extracted peat. Their papers establish relevant captures, but no downloadable geometry artifact with a dataset license was found. The Latvia and Lithuania national ALS products are useful free conditioning and source-repair inputs, not 6.25 cm ground truth.

**No candidate in this audit qualifies as production target evidence for the current Estonia activation gate.** Foreign examples cannot replace independent Estonia sites, and none publishes the complete combination of target-scale ground support, bounded error, measured transfer/effective resolution, masks, registration, and redistributable artifacts. The correct no-budget response is to ingest the two open archives, request the three unreleased datasets from their authors, and use them for pretraining/conversion validation while retaining Estonia blind validation.

## Qualification Standard

The finest stored terrain rung is 0.0625 m per sample, or 256 cells/m2, with representable wavelengths beginning near 0.125 m. Nominal raster spacing or total point density alone does not prove that terrain at those scales was measured.

A production target needs all of the following:

- ground-surface semantics, not an unqualified DSM or vegetation-bearing cloud;
- raw returns or a lossless-enough artifact from which support can be audited;
- per-cell or reconstructable ground support and explicit unsupported-area masks;
- independently measured horizontal and vertical error appropriate to the relief being learned;
- effective resolution or transfer evidence, not just output pixel size;
- dates and registration sufficient to distinguish change from acquisition mismatch;
- rights permitting machine-learning use, derived assets, and the intended model/artifact distribution;
- Estonia regime relevance and, for production activation, independent Estonia-site validation.

The report uses these roles:

- **R - raw/research/pretraining:** useful geometry, semantics, or method-validation evidence, but not a production target without further audit or conversion.
- **C - conditioning/source repair:** geographically broad covariate or baseline evidence that cannot resolve target-scale relief.
- **L - lead/request-only:** a relevant capture is documented, but the usable artifact or its rights are not public.
- **X - reject:** the available product does not materially support target-scale ground synthesis.

## Candidate Ledger

| Candidate | Geography and regimes | Public artifact | Ground and quality evidence | Rights | Qualification |
|---|---|---|---|---|---|
| Latvia L\u0122IA national ALS | Latvia; broad coverage of peat, till, agriculture, forests, dunes, shore and disturbed ground | LAS tiles, whole-country 2013-2019; at least 4 points/m2 total and about 1.5 points/m2 ground | Ground class manually corrected; official product card found no independent horizontal/vertical error. A University of Latvia assessment reports approximately 12 cm claimed vertical accuracy and sensitivity to vegetation/terrain | CC BY 4.0 | **C** only; far too sparse and inaccurate for direct 6.25 cm targets |
| Lithuania Lidar_DR_LT | Lithuania; national cross-regime coverage | LAZ tiles by LKS-94 1:2,000 sheet; 2019-2025, at least 6.5 points/m2 overall; 2025 product at least 15 pulses/m2 | Automated LAS classes. Official RMSE limits on hard stable objects: <=30 cm horizontal and <=10 cm vertical, not fine ground transfer | Public access, but machine-readable license/derivative terms were not established from the official metadata; bulk access can require a signed request and may reach 10 TB | **C** only; legal review required before training |
| Pilkosios/Nagliai Dunes multi-epoch SfM | Curonian Spit, Lithuania; active sand dune, grassland, sparse canopy | No public archive found. Five UAV DEM epochs: 3.84-4.02 cm in 2018-2019 and 6.58 cm in 2022 | RTK UAV workflows documented, but no downloadable raw cloud/images/checkpoints, independent error distribution, support masks, or artifact license; output is surface geometry where vegetation is present | Article access does not establish dataset reuse rights | **L**; high-priority author request for bare-sand process morphology |
| Dutchman's Cap coastal cliff | Lithuania; approximately 2 km/3 ha cliff, parabolic dune over moraine, landslides and erosion | No public archive found; paper says data available from authors | 1 cm mesh; detailed photogrammetry GCP RMSE X/Y/Z 1.3/0.9/2.3 cm and combined LiDAR-photogrammetry 4.9/5.6/4.0 cm. Ground/vegetation classification and support remain inadequate; overhangs are not heightfields | Article CC BY does not by itself license unreleased data | **L**, becoming **R** if obtained with rights; valuable cliff/moraine/dune evidence |
| Preila lagoon shoreline | Lithuania; 566.84 m vegetated lagoon shoreline | No point-cloud/DEM archive found; imagery, LiDAR and derived DSM are described | 2024 UAV imagery and LiDAR, approximately 2 cm target GSD, five GCPs; achieved target-scale ground error and vegetation-free ground support are not established | Article CC BY; dataset license not separately established | **L**; shoreline/vegetation conditioning, not ground truth as documented |
| Kaigu extracted peatland | Latvia; 16.4 ha abandoned extraction field with dry/moist bare peat, sedges, reeds, water and ditches | No machine-readable DEM/cloud/raw-image artifact found | 2023 RTK UAV photogrammetry and five RTK-GNSS GCPs documented; no GSD/effective-resolution, error distribution, ground semantics under vegetation, or support masks published | Article CC BY; no independently licensed geometry artifact | **L**; useful peat morphology if authors release raw data and checkpoints |
| Biała Góra cliff UAV LiDAR | Poland, southern Baltic; approximately 1 km coastal cliff, ten epochs in 2022-2023 | Ten ZIP archives containing LAS, RGB and return count; **27,401,865,660 bytes compressed (27.402 GB / 25.520 GiB)** | Public record describes raw unstructured/unclassified clouds and gives no density, independent positional error, GCP/checkpoint record, support mask, or authoritative ground class | Dataset and files CC BY 4.0 | **R**, immediate download; conversion/pretraining and change-consistency evidence, not target truth yet |
| ForestSemantic-MS | Espoonlahti, Finland; six boreal forest plots | Six LAZ files, 131.1 MB total; manually annotated ground, low vegetation, trunks, branches, foliage and woody debris | Very dense total returns, but released total density is not ground support. Preprocessing includes multispectral merge, filtering and coordinate/height normalization; no independent target-scale ground error or transfer measurement | CC BY 4.0 | **R**, immediate download; best forest semantic and ground-conversion corpus found |

## Verified Candidate Notes

### Latvia L\u0122IA National ALS

The Latvian Geospatial Information Agency distributes LAS basic data covering Latvia from the 2013-2019 acquisition cycle. The official page and product card specify at least 4 points/m2 overall, approximately 1.5 points/m2 ground returns, LKS-92 TM / LAS-2000.5, and classes for ground, low/medium/high vegetation and buildings. Ground classification is described as manually corrected. Tiles are available free through the map/list interface under CC BY 4.0.

- Official product page: [L\u0122IA digital elevation-model basic data](https://www.lgia.gov.lv/en/Digit%C4%81lais%20virsmas%20modelis)
- Official product card: [Digital altitude model basic data](https://s3.storage.pub.lvdc.gov.lv/lgia-opendata/pakalpojumu%20kartinas/Digitala_augstuma_modela_pamatdati.pdf)
- Independent assessment lead: [University of Latvia repository record](https://dspace.lu.lv/items/3465c2a3-e66f-4283-b117-f187220a4a44)

At roughly 1.5 ground returns/m2, this is over two orders of magnitude below the 256 samples/m2 output lattice even before occlusion and interpolation are considered. The approximately 12 cm vertical-accuracy figure discussed in the university assessment is also larger than many desired microforms. Use the product to condition broad terrain, detect source anomalies, compare ground filtering, and locate regimes; do not upsample it into target evidence.

### Lithuania Lidar_DR_LT

Lithuania's National Land Service/Geoportal distributes `Lidar_DR_LT` LAZ sheets. The official metadata describes 2019-2025 national coverage, at least 6.5 points/m2 overall, LKS-94 sheet organization, automatic LAS classification, and <=30 cm horizontal and <=10 cm vertical RMSE limits measured on hard stable objects. A 2025 product announcement specifies at least 15 pulses/m2 and additional RIEGL attributes. Those figures remain far below and materially less accurate than the finest relief target.

- Official metadata: [Lidar_DR_LT resource](https://www.geoportal.lt/metadata-catalog/catalog/search/resource/details.page?uuid=%7B3C0D5E47-49CA-4132-88C1-EC3D1E4610B4%7D)
- Open-data catalog: [Lithuanian point-cloud dataset record](https://data.gov.lt/datasets/3387/)
- Bulk-data procedure: [Geoportal large-volume data access](https://www.geoportal.lt/geoportal/subscribe/-/asset_publisher/I0YH9ZsWns4x/content/kaip-gauti-dideles-apimties-duomenis)

The official record establishes access, not a sufficiently explicit ML/derivative license for this use. Confirm legal terms before copying tiles into a training corpus. The product is still useful as a free national geological/landform conditioning layer and as an independent check on Estonia source-repair procedures.

### Pilkosios/Nagliai Active Dunes

Tij\u016bnait\u0117 et al. document five RTK-UAV SfM DEMs of the active Pilkosios Dunes on the Curonian Spit. Reported output spacing is 4.02, 3.94, 3.90, and 3.84 cm for 2018-2019 SenseFly eBee RTK/Pix4D surveys, and 6.58 cm for a 2022 DJI Phantom Pro RTK/Pixpro survey. The landscape includes bare sand, grassland, and sparse canopy, making it unusually relevant to process-shaped aeolian relief.

- Primary paper: [Development of active dunes on the Curonian Spit](https://baltica.gamtc.lt/administravimas/uploads/06__tijunaite__baltica_36_2_657a478130899.pdf)

The paper does not provide a data repository, raw images/clouds, checkpoints, independent error distribution, masks, or dataset license. Output spacing must not be treated as measured resolution. Request the original RTK solution, camera images, dense clouds, DEMs, checkpoints, land-cover masks, and written ML/derived-artifact permission. Bare-sand subareas would be useful pretraining material if these are supplied.

### Dutchman's Cap Coastal Cliff

The Dutchman's Cap/Olandian Hat study covers a roughly 2 km long, approximately 3 ha Lithuanian Baltic cliff system with a parabolic dune over moraine, landslides, and active erosion. The paper combines UAV photogrammetry and LiDAR and reports a 1 cm mesh. Its detailed table reports photogrammetric GCP RMSE of 1.3 cm X, 0.9 cm Y, and 2.3 cm Z, and combined-product RMSE of 4.9 cm X, 5.6 cm Y, and 4.0 cm Z.

- Primary paper and DOI `10.3389/frsen.2025.1397513`: [UAV photogrammetry and LiDAR of Dutchman's Cap](https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2025.1397513/full)

The article states that raw data can be requested, but no public file archive or data-specific license was found. Ask for raw LiDAR, images, trajectories, control/checkpoints, classified products, vegetation masks, dates, CRS and explicit redistribution/training rights. This is excellent non-heightfield cliff and mass-wasting evidence; it cannot directly supervise overhangs in a 2.5D height representation.

### Preila Lagoon Shoreline

The 2024 Preila survey covers 566.84 m of Curonian Lagoon shoreline using Matrice 600 Pro imagery/LiDAR and additional Mavic Air imagery. The workflow targets approximately 2 cm GSD, uses five LKS-94 control points, and produces point clouds, orthomosaics and a DSM.

- Primary paper and DOI `10.3389/frsen.2026.1786848`: [Integrated UAV shoreline monitoring at Preila](https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2026.1786848/full)

No downloadable geometry artifact or data-specific license was found, and the paper describes dense vegetation as a reconstruction constraint. Request the original cloud, imagery, LiDAR returns, GCP/checkpoint observations, water/vegetation masks and permissions. As published, it is a shoreline-monitoring lead rather than vegetation-free terrain evidence.

### Kaigu Extracted Peatland

The Kaigu study surveys 16.4 ha of abandoned Latvian extracted peat in August 2023 using a DJI Phantom RTK and five RTK-GNSS control points. It documents bare dry peat, bare moist peat, sedges, reeds, water and ditches, so it samples a regime missing from the open cliff and forest sets.

- Primary paper and DOI `10.3390/land13020188`: [Remote sensing of an extracted peatland in Latvia](https://www.mdpi.com/2073-445X/13/2/188)

The article's data-availability statement does not expose a machine-readable DEM, cloud, raw imagery or checkpoints. It also does not establish target-scale effective resolution, independent error or ground support below vegetation. Request those artifacts and explicit reuse rights; do not digitize figure rasters as targets.

### Biała Góra Coastal Cliff

The RepOD Dataverse release contains ten UAV-LiDAR captures collected between February 2022 and December 2023 over approximately 1 km of coastal cliff near Mi\u0119dzyzdroje. Files are ZIP-compressed LAS with natural-color RGB and return number. The archive is CC BY 4.0 and directly downloadable. The ten geometry archives plus README total 27,401,865,660 bytes compressed.

- Dataset and DOI `10.18150/BHH1RC`: [Biała Góra UAV LiDAR time series](https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/BHH1RC)
- Conversion code: [SCANDEM-Coast official repository](https://github.com/Baltic-Coastal-Monitoring-Team/SCANDEM)
- Related software paper and DOI `10.1016/j.softx.2025.102270`: [SCANDEM-Coast](https://doi.org/10.1016/j.softx.2025.102270)

The record explicitly leaves the clouds unstructured/unclassified and supplies no density, survey error, GCP/checkpoint data, ground mask or authoritative classification. SCANDEM's SMRF-based result is a conversion hypothesis, not ground truth. First ingest one epoch, inspect CRS and LAS metadata, measure return and ground-candidate support, align repeat epochs, and quantify cross-epoch stable-surface disagreement. Only then decide whether the geometry can supervise particular coastal regimes.

### ForestSemantic-MS

ForestSemantic-MS releases six LAZ plots from multispectral helicopter laser scanning in Espoonlahti, Finland. Points are manually labeled as ground, low vegetation, trunks, branches, foliage, or woody debris. Four plots are designated for training and two for testing. The six files total 131.1 MB and are CC BY 4.0.

- Dataset and DOI `10.5281/zenodo.17172162`: [ForestSemantic-MS](https://zenodo.org/records/17172162)
- Methods paper: [ForestSemantic: multispectral point-cloud segmentation](https://link.springer.com/article/10.1007/s41064-025-00369-4)

The corpus is immediately useful for forest semantic filtering and for evaluating whether conversion preserves woody debris rather than flattening it into terrain. Its very high total return density must not be reported as ground density. The published preprocessing merges wavelengths, filters outliers, applies a cloth-simulation ground stage, and normalizes coordinates/heights; no independent target-scale ground-error or transfer measurement is provided. Retain the manual labels and derive support/confidence masks instead of treating an interpolated DTM as exact truth.

## Regime Coverage And Remaining Gaps

| Regime | Best zero-cost evidence found | Critical unresolved gap |
|---|---|---|
| Forest floor / woody debris | ForestSemantic-MS | Independent ground geometry/error under canopy; ground-support transfer at 6.25 cm |
| Coastal cliff / moraine / landslide | Biała Góra open series; Dutchman's Cap request lead | Authoritative ground classification, checkpoint error, overhang-to-heightfield policy |
| Dunes / bare sand | Pilkosios request lead; Dutchman's Cap request lead | Public raw artifact, error/support evidence, permission |
| Peat / extraction microforms | Kaigu request lead | Public raw artifact; vegetation-free ground and error evidence |
| Vegetated shoreline / fluvial edge | Preila request lead | Water handling, ground below vegetation, target-scale error and masks |
| Till/moraine away from cliffs | National ALS conditioning only | Dense classified public target artifact |
| Agriculture and disturbed ground | National ALS conditioning; no verified dense target corpus | Tillage-state/date, bare-ground survey, repeatability and licensing |
| Alvar/carbonate pavement | No qualifying Latvia/Lithuania public artifact found | Entire target-scale corpus |
| Sandstone outcrop comparable to Taevaskoda | No qualifying neighboring-country public artifact found in this pass | Dense classified geometry and Estonia transfer evidence |

This ledger is evidence of search outcome, not proof that no unindexed institutional dataset exists. It does prove that a proposed candidate cannot be treated as ready merely because a paper shows a centimeter-scale DEM.

## Zero-Budget Acquisition Plan

1. Download ForestSemantic-MS now; preserve original LAZ, licenses, hashes and train/test split.
2. Download one Biała Góra epoch first, audit metadata/support and run a conversion feasibility check; fetch all 27.402 GB only if that audit passes.
3. Send focused data requests to the Pilkosios Dunes, Dutchman's Cap, Preila and Kaigu authors. Ask for raw sensor artifacts, derived clouds/meshes, control and checkpoints, masks/classes, dates/CRS, and explicit permission for ML training and distribution of learned/derived assets.
4. Use Latvia/Lithuania national ALS only as broad conditioning, source-defect comparison, regime localization and classification context. Do not promote interpolated national ALS to decimeter target truth.
5. Register every received artifact with file hashes, exact license text, geometry/error metadata and a target-qualification decision before training.
6. Keep the three-independent-Estonia-site blind-validation requirement. Neighboring-country corpora may pretrain morphology and ground filtering, but activation in Estonia still requires Estonia evidence.

## Decision Impact

The absence of procurement budget does **not** require falling back to procedural noise or abandoning learned synthesis. The open Polish and Finnish artifacts, plus no-cost author requests, provide enough evidence to begin serious foreign-corpus ingestion and conversion work. They do **not** justify claiming measured 6.25 cm Estonia ground truth.

The recommended method architecture should therefore separate:

- broad national conditioning and defect repair;
- foreign morphology/semantic pretraining with explicit domain labels;
- uncertainty-aware conversion from raw returns rather than raster-pixel worship;
- independent Estonia activation and visual validation.

No binary-format or runtime change follows from this audit. It changes only the evidence acquisition and cook-side synthesis-training plan.
