# Nordic Public Target-Scale Data Hunt

**Date:** 2026-07-13
**Scope:** Zero-purchase search for legally accessible, high-resolution terrain and ground-geometry observations in Estonia's neighboring Nordic/Baltic physical domains. This report evaluates whether public Finnish and Swedish data can supply target-scale evidence for the cook-side 0.0625 m microtopography synthesizer. It does not redesign the synthesizer, packed format, or runtime.

## 1. Decision

Public neighboring-country data are worth using, but only as a **stratified raw-observation and exemplar corpus**, not as a turnkey national training target.

The strongest verified acquisitions are:

1. **Hovi et al. 2024 forest TLS:** the highest-priority download because it includes 13 plots at Jarvselja, Estonia, under the same protocol as 28 Finnish Hyytiala plots. Full individual scans and transformations are available, not only a rasterized derivative.
2. **Evo 2024 forest TLS, Finland:** the strongest compact Finnish conversion candidate: 55 leaf-off, nine-station plots with millimetre instrument sampling, height-above-ground and tree IDs, and a manageable 39.3 GiB total release.
3. **FORWARD, Sweden:** the strongest public source for rough glacial/boulder forest, machine disturbance, wheel ruts, and before/after forestry context. Its airborne nominal density is unusually high, but usable ground support must be measured rather than inferred from total density.
4. **Evo 2021 forest TLS, Finland:** a larger 90-plot secondary forest corpus. It has less immediately useful surface metadata than Evo 2024 but expands stand and site variance.
5. **Stordalen 5 cm UAV DSM, Sweden:** a small, cheap peat-morphology exemplar. It is a visible surface in a subarctic permafrost mire, not bare-earth Estonia truth.

These releases require no purchase. Their open licenses allow copying and adaptation with attribution, but none provides a ready, semantically correct 0.0625 m terrain surface. They must pass ground/stable-organic separation, support, uncertainty, and transfer audits before becoming supervision.

They close only part of the evidence gap. No verified open Nordic target-scale corpus was found for Estonian agriculture, Baltic sand and shingle shores, dunes, alvar/carbonate pavements, ordinary fluvial banks, or exposed sandstone/limestone landforms. Request-only papers and commercial demonstrations exist in several of those regimes; they are not public zero-cost dependencies.

## 2. Qualification Standard

The output spacing is 0.0625 m and the smallest representable wavelength is about 0.125 m. A point cloud or raster qualifies as target-scale evidence only if its **actual surface support**, not its file grid or total-return density, can resolve that band.

This report uses these roles:

- **`P` production target:** target-scale measured surface paired to Estonia conditions, with sufficient semantics, support, uncertainty, georeferencing, date, and reuse rights.
- **`E` exemplar:** real target-scale surface useful for morphology or validation, but foreign, unpaired, semantically different, too small, or incompletely characterized.
- **`R` raw conversion candidate:** sufficiently dense point cloud or DSM that may become `E` or `P` only after classification, rasterization, support/error analysis, and human QA.
- **`C` context/calibration:** useful conditioning, masks, parent geometry, or process evidence, but too sparse or otherwise unfit as target heights.
- **`X` rejected/unavailable:** no public artifact, incompatible rights, or inadequate measurement support.

Every `R` source must demonstrate all of the following before height loss is allowed:

1. Surface semantics separating mineral soil, stable moss/peat, exposed roots, clasts, loose litter, deadwood, vascular vegetation, crops, water, snow, and built objects as appropriate.
2. Per-cell measured support, view diversity, occlusion/hole masks, and interpolation distance. Fine rasterization does not create measurements.
3. Vertical and horizontal error relative to independent checks. Instrument accuracy and scan-to-scan registration are not terrain-surface accuracy.
4. A measured transfer function or controlled feature-recovery test across the 0.125-1 m wavelength band.
5. Stable campaign/site IDs, acquisition dates, coordinate systems, and non-overlapping geographic train/validation splits.
6. Pairing to the co-located 1 m source DTM and the soil, surficial geology, bedrock, hydrology, land use, canopy, and orthophoto conditions actually available to the Estonia cook.
7. A recorded attribution and derivative ledger. CC BY permits sharing and adaptation for any purpose with attribution, but does not explicitly define trained-model or model-weight treatment; public model/weight release requires a project legal determination rather than an engineering assumption.

No source below is accepted as `P` without this conversion and qualification work.

## 3. Candidate Matrix

| Priority | Source | Physical analogy | Scale and contents | Rights / transfer | Status |
|---|---|---|---|---|---|
| 1 | [Hovi et al. terrestrial laser scanning point clouds](https://doi.org/10.23729/9a8d90cd-73e2-438d-9230-94e10e61adc9); [data paper](https://doi.org/10.5194/essd-16-5069-2024) | Hemiboreal Estonia and Finnish boreal forest floors; stand and understory variance | 58 plots: 13 Jarvselja, 28 Hyytiala, 17 Czech. Leica P40; 44 mature plots use 16 scans and 14 young plots one scan. Individual full scans plus transformations, merged clouds, 2 cm-thinned LAZ, inventory, plot photos, and forest-floor cover/spectra. Entire release about 1.95 TiB; all 58 thinned LAZ about 94.2 GB decimal. | CC BY 4.0; files selectable individually | **`R+`**. Highest priority, including an Estonia subset; not `P` until terrain semantics, support, and independent surface error are established. |
| 2 | [Evo 2024 TLS, Finland](https://doi.org/10.23729/fd-5a800660-8bd8-35ef-ac9f-ac5c45f7fa77); [data note](https://doi.org/10.14214/sf.24066) | Southern Finnish boreal forest floor across pine, spruce, birch, aspen, age, and closure | 55 plots, each 32 x 32 m; 56,320 m2 nominal. Nine Riegl VZ-400i positions per plot, upright and tilted, leaf-off April-May 2024. 3.5 mm angular sampling at 10 m; 7 mm beam, about 10/14/18 mm footprint at 10/20/30 m. LAZ has returns, reflectance, scan position, height-above-ground, and tree ID. 57 files, about 39.3 GiB. | CC BY 4.0 | **`R+`**. Best compact Finnish raw candidate; `treeid=0` mixes terrain with undergrowth/deadwood and must not be treated as a ground class. |
| 3 | [FORWARD, Sweden](https://doi.org/10.71540/89rs-s553) | Rough till/boulder forest, managed forest floor, clearcut/forwarder disturbance, wheel tracks and rutting | Marrviken and Bjorsjo sites in Vasternorrland. Riegl VUX-120 helicopter LiDAR at approximately 1,500 total points/m2, plus UAV imagery/photogrammetry, forwarder GNSS/IMU/log data, video, and operational context. 1,445 files, about 1.07 TiB. | CC BY 4.0; selective file download is necessary | **`R+`**. Strongest disturbance/till candidate. Total-return density is not ground density; airborne occlusion, classification, cross-date registration, and surface error remain gates. |
| 4 | [Evo 2021 TLS, Finland](https://doi.org/10.23729/4024180e-9462-4937-befa-eb9da622954a) | Boreal forest floors and stand-context variance | 90 radius-25 m plots, roughly 176,700 m2 nominal. Leica RTC360, nine scans per plot, spring 2021, artificial-target registration, local plot coordinates. 90 files, about 100.1 GiB. | CC BY 4.0 | **`R`**. Valuable breadth, but weaker than Evo 2024 because the catalog does not establish terrain classes, surface error, or absolute pairing. |
| 5 | [Stordalen Mire 5 cm UAV product](https://hdl.handle.net/11676.1/xdEqbdjKjwib52XtBLDNaUbh) and [RGB companion](https://hdl.handle.net/11676.1/U4o8KrPkEiKw5RsfiCJZeEgX) | Peat/moss hummock-hollow morphology only | Nominal 0.05 m UAV photogrammetric DSM, acquired 2019-08-16; one 234 MiB ZIP plus orthomosaic. | SITES CC BY 4.0 | **`E?`** after error/MTF audit. It is a visible DSM in a subarctic palsa/thermokarst mire, not bare-earth or an Estonia-equivalent peat distribution. |
| 6 | [Helsinki classified point cloud](https://doi.org/10.5281/zenodo.5578198) | Technogenic, compacted, and urban disturbed ground; classification experiments | 13.2 GB LAS combining Leica RTC360 TLS and DJI Phantom 4 Pro photogrammetry, classified with TerraScan in ETRS-TM35FIN. | CC BY 4.0 | **`R`**, low priority. No published target-scale terrain error, density distribution, fusion uncertainty, or sufficiently explicit class contract. Not a natural-terrain target. |
| 7 | [Getaberget, Aland orthomosaics](https://doi.org/10.5281/zenodo.4719627) | Exposed-rock fracture appearance | 0.0055 m GSD UAV orthomosaics, 16.6 GB, 5-10 ground-control points. | CC BY 4.0 | **`C` only**. No height field, point cloud, DEM, or raw-image reconstruction package; useful only as optical/fracture context. |

`R+` means especially promising raw data, not accepted truth.

## 4. Primary Findings by Dataset

### 4.1 Hovi 2024: first acquisition, and a correction to the prior Estonia inventory

The official release contains 58 forest plots across Hyytiala (Finland), Jarvselja (Estonia), and the Czech Republic. The 13 Jarvselja plots are an important correction to a simplistic statement that no open target-scale Estonia observations exist. There are open, dense Estonia **raw observations**. There is still no ready open Estonia target **height surface** with the semantics, masks, errors, and paired cook conditions required for production supervision.

For 44 mature plots, 16 scans were registered with reported co-registration below 1 cm. Full individual scans and transformations are retained; the convenient merged products were thinned to an average minimum spacing of 2 cm and cover approximately 35 x 35 m for the multi-scan plots. The instrument beam is approximately 6 mm with 0.23 mrad divergence for most scans. Those numbers establish promising acquisition support but not ground availability beneath understory or a terrain error distribution.

The associated forest-floor records are useful: plot photographs, fractional cover, and spectra distinguish broad floor composition. They do not label every point as mineral soil, moss, litter, root, deadwood, or vascular vegetation. The distributed clouds are intentionally unfiltered for forest radiative-transfer work. A terrain converter therefore needs view-aware surface extraction and human-labeled audit patches rather than a lowest-point shortcut.

**Recommended zero-cost action:** download the 13 Jarvselja thinned LAZ files, their metadata/photos/cover tables, and a small number of corresponding full-scan ZIPs. Use Hyytiala as a geographically held-out transfer set under the same acquisition protocol. Do not download the whole 1.95 TiB archive initially.

### 4.2 Evo 2024: best bounded Finnish conversion corpus

The Evo release is unusually suitable for a zero-budget first conversion because its full public footprint is about 39.3 GiB, its 55 plots are uniform 32 x 32 m acquisitions, and the campaign is leaf-off. Nine upright/tilted scan positions reduce, but do not eliminate, forest-floor occlusion. The LAZ schema preserves reflectance, return information, scan-position identity, height above ground, and tree IDs.

The published 3 mm relative positioning at 50 m is an instrument/system performance figure, while the under-3 cm absolute agreement is reported against older mapped tree positions. Neither is an independent forest-floor height error. The supplied height-above-ground was derived after an Axelsson-style ground identification. It is useful evidence but not automatically correct at roots, rocks, moss, depressions, and low plants. `treeid=0` contains all non-tree points rather than a pure terrain class.

**Recommended role:** develop and evaluate the raw-to-surface conversion and surface-support metrics on a stratified subset, then reserve entire plots for Finnish transfer validation. Do not train on height-above-ground as if it were a validated target surface.

### 4.3 FORWARD: uniquely useful for till, boulders, and disturbance

FORWARD covers two Swedish forest-machine research sites. Marrviken is described as rough terrain with large boulders; Bjorsjo is less rough but still contains large boulders. The release joins very dense helicopter LiDAR with UAV data and actual forwarder position, motion, load, and operational records. This makes it the strongest public candidate found for ground deformation, tracks, wheel ruts, boulder exposure, and forestry disturbance rather than merely undisturbed forest floor.

At approximately 1,500 total returns/m2, the naive nominal spacing is about 0.026 m. That figure cannot be used as the surface spacing: canopy returns, repeated views, nonuniform flight geometry, classification, and occlusion determine ground support. The helicopter campaign and machine/UAV campaigns also span different dates, so before/after geometry requires explicit temporal alignment and change logic.

**Recommended role:** selective acquisition of ground-class tiles and matching UAV/machine records for a few rough, smooth, and trafficked strata. Treat it as `R+` until ground-density histograms, held-out check surfaces, classification confusion, and acquisition-date relationships are verified.

### 4.4 Evo 2021: breadth after the first converter works

The 2021 collection offers 90 plots and about 100 GiB of multi-station RTC360 clouds. It should not be the first corpus because Evo 2024 has a stronger published schema and acquisition description. Its value is breadth: after conversion is proven on Jarvselja/Evo 2024, the 90 plots can test whether the surface estimator and learned representation collapse across stand types or scanners.

The plots are distributed in local coordinates and the public catalog does not supply a validated terrain class or terrain-surface accuracy. Absolute pairing to national conditions may require additional site metadata and transforms. Keep it `R`, not `E`, until those are resolved.

### 4.5 Stordalen: useful peat morphology, dangerous transfer

The SITES record calls the product a terrain model in its title but describes it as a photogrammetric **digital surface model**. Its 5 cm pixel size is nominally close to the 6.25 cm output, and the 234 MiB package is cheap to audit. It can test hummock/hollow representation and visible peat/moss surface spectra.

Stordalen is a subarctic permafrost mire with palsa and thermokarst processes that do not define ordinary Estonian bog/fen ownership. Photogrammetry measures the visible canopy/moss surface and can fail in homogeneous, wet, or shadowed regions. The catalog does not establish the effective transfer function, vertical error, or vegetation-removal semantics needed for direct height supervision.

**Recommended role:** an `E?` morphology exemplar and conversion stress test only. Never use it to set Estonia peat frequencies, amplitudes, spatial organization, or class prevalence without Estonia evidence.

## 5. National Products: Context, Not 6.25 cm Targets

### Sweden

[Lantmateriet Laser data Download, forest](https://www.lantmateriet.se/en/geodata/our-products/product-list/laser-data-download-forest/) is open CC0, classified, LAZ, and tiled at 2.5 x 2.5 km, but its nominal density is only 1-2 points/m2. That is roughly 0.7-1 m nominal spacing before ground filtering. It is useful as national parent geometry, masks, georeferencing, and a cross-check, not target-scale morphology.

### Finland

[National Land Survey of Finland open laser scanning data](https://www.maanmittauslaitos.fi/en/maps-and-spatial-data/datasets-and-interfaces/product-descriptions/laser-scanning-data) at 0.5 points/m2 is likewise context only. The 5 points/m2 product is still approximately 0.45 m nominal spacing before filtering, is not an open zero-cost substitute, and carries restrictive [terms of use](https://www.maanmittauslaitos.fi/en/laser-skanning-data/terms-of-use) for high-precision redistribution and processing. It is both technically inadequate for the 0.125 m band and outside this zero-purchase hunt.

National ALS remains useful for aligning foreign exemplars to the same coarse observation process as Estonia. It must not be resampled to 6.25 cm and mislabeled target truth.

## 6. Attractive Near-Misses That Do Not Qualify

| Source | Why it matters | Why it is not a public target dependency |
|---|---|---|
| [Gotland and Faro coastal limestone TLS study](https://doi.org/10.3390/rs15061667) | Exceptional physical analogy for Baltic limestone pavement, slabs, shingle, shore platforms, and sea stacks; multiple TLS stations with millimetre-class instrument accuracy | Paper says raw data are available on request; no public deposit, manifest, data license, or guaranteed access was verified. **`X` until deposited and licensed.** |
| [Oulanka multispectral high-density LiDAR preprint](https://arxiv.org/abs/2603.22230) | Approximately 500-1,600 total points/m2 from three wavelengths over sand, gravel, vegetation, forest floor, and water; highly relevant fluvial morphology and semantic classification | No public data DOI, download, or reuse license was found. Project pages direct users to contact the team. **`X` for current planning.** |
| HuHoLa Swedish mire 3 cm SfM survey ([paper](https://doi.org/10.1016/j.ecolmodel.2025.111212), [code](https://github.com/bravemaster3/huhola)) | Relevant mire microform resolution study | Full 3 cm survey is request-only; checked repository samples are coarser and no repository data-license contract was established. **`C` now.** |
| Ystad bathymetric UAV LiDAR demonstrations | Relevant sandy Baltic beach/topobathymetric process | Commercial case study; no open corpus or license. **`X`.** |
| Repeated Oulanka river TLS publications | Relevant erosion/deposition and bank-change method | No stable public raw deposit and derivative license found. **`X`.** |

A polite no-cost data request may eventually succeed, but the implementation plan must not depend on it. Requested files must enter the same license, semantics, uncertainty, and support audit as public files.

## 7. Regime Coverage and Remaining Gaps

| Estonia-relevant regime | Best verified public neighbor evidence | Current conclusion |
|---|---|---|
| Hemiboreal/boreal forest floor | Hovi Jarvselja/Hyytiala; Evo 2024/2021 | Strong raw observation base. Still requires ground/stable-organic semantics and transfer qualification. |
| Pit-mound, roots, deadwood, understory interaction | Hovi and Evo TLS | Morphology may be present, but no explicit labels or complete visible-surface contract. Raw conversion only. |
| Glacial till, stony/boulder forest | FORWARD; possibly Evo sites after geology pairing | Strongest public raw candidate, especially FORWARD. Substrate/soil pairing remains unresolved. |
| Forestry disturbance, tracks, wheel ruts | FORWARD | Strong public candidate with operational context; temporal and ground-class audit required. |
| Peat bog/fen hummock-hollow | Stordalen DSM; forest TLS may contain wet plots | One cheap exemplar with major permafrost/DSM transfer limitations. Estonia ownership remains unsupported. |
| Agriculture and field operations | None verified | Open Nordic target-scale geometry gap. National ALS and orthophoto are conditions, not target relief. |
| Baltic sand beach, dune, shingle, shallow-water edge | None verified | Open target-scale gap. Public commercial examples and request-only studies do not qualify. |
| Alvar/carbonate pavement and limestone coast | Gotland/Faro request-only TLS | Excellent analogy, no public licensed artifact. Gap remains. |
| Fluvial banks, bars, floodplain microrelief | Oulanka high-density data unavailable | Gap remains despite a strong demonstrated acquisition. |
| Exposed bedrock/outcrop fracture geometry | Getaberget optical only | No public neighboring target heights verified. Optical conditioning only. |
| Urban/technogenic disturbed ground | Helsinki classified cloud | Limited raw conversion candidate; not natural-terrain evidence. |

The available corpus therefore supports a serious forest/till/disturbance workstream and a limited peat experiment. It does **not** justify claiming credible added detail on every Estonian surface class. Unsupported regimes need either later public discoveries, a general model whose transfer is validated on real Estonia observations, or an explicit release limitation. They must not be silently filled with generic noise.

## 8. Zero-Purchase Acquisition Plan

No paid survey, commercial license, or commissioned dataset is proposed.

### Phase 1: smallest decisive download

1. Pull Hovi metadata, cover/spectra tables, all 13 Jarvselja thinned LAZ files, and 3-4 corresponding full-scan packages spanning materially different forest floors.
2. Pull 6-10 Evo 2024 plots stratified by species mixture, age/closure, and site condition rather than the entire release.
3. Pull a few FORWARD ground/UAV subsets spanning boulder-rich undisturbed ground, less-rough ground, and trafficked/rutted ground.
4. Pull the 234 MiB Stordalen DSM/orthomosaic pair.
5. Record file-level checksums, source version, DOI, license, campaign/site/plot ID, coordinates, date, instrument, and transformations before conversion.

### Phase 2: qualification before expansion

For each family, produce hand-inspected audit patches with explicit semantic labels and measured support. Compare the converted surface to the original scans from held-out views and to independently selected check features. Measure recovery by wavelength and relief amplitude, not only point RMSE. Reject cells that are interpolated across occlusion, water, vegetation, or uncertain surface identity.

Expansion to all Evo/Hovi/FORWARD files is justified only if the smallest subset demonstrates usable 0.125-1 m morphology and a defensible surface contract. This prevents terabyte-scale download and storage from becoming a substitute for evidence quality.

### Phase 3: train/validation boundaries

- Keep entire sites and campaigns held out; never split overlapping patches randomly.
- Jarvselja is the only verified neighboring corpus here that is actually in Estonia. Reserve enough complete Jarvselja plots for untouched Estonia validation.
- Use Hyytiala and Evo to test protocol/scanner/domain transfer, not to define Estonia prevalence.
- Use FORWARD only for strata supported by its measured terrain and disturbance context.
- Use Stordalen only for representation/morphology tests unless Estonia peat observations validate transfer.
- Preserve source attribution through derived rasters, training manifests, checkpoints, and cooked evaluation artifacts.

## 9. Final Recommendation

Proceed immediately with a **small, selective, zero-cost raw-data qualification sprint** centered on Hovi Jarvselja, Evo 2024, and FORWARD. These are not side projects; they directly test whether public neighboring-country observations can supervise the difficult 6.25 cm band without purchasing new surveys.

Do not wait for a perfect pan-Nordic corpus, but do not misrepresent the result. The public evidence is strong enough to improve forest floor, stony till, and forestry-disturbance synthesis. It is not strong enough to own all of Estonia's agriculture, peat, shore, dune, carbonate, fluvial, and outcrop distributions. The generator must retain explicit evidence boundaries, and every visual claim must be validated against held-out measured surfaces rather than downsampling consistency or attractive noise.

## 10. Primary Source Ledger

| Source | Primary record inspected | Key use |
|---|---|---|
| Hovi et al. forest TLS | [Fairdata DOI](https://doi.org/10.23729/9a8d90cd-73e2-438d-9230-94e10e61adc9), [ESSD data paper](https://doi.org/10.5194/essd-16-5069-2024) | Exact Jarvselja Estonia + comparable Hyytiala raw TLS |
| Evo 2024 | [Fairdata DOI](https://doi.org/10.23729/fd-5a800660-8bd8-35ef-ac9f-ac5c45f7fa77), [Silva Fennica data note](https://doi.org/10.14214/sf.24066) | Compact, well-documented Finnish forest TLS |
| Evo 2021 | [Fairdata DOI](https://doi.org/10.23729/4024180e-9462-4937-befa-eb9da622954a) | Larger Finnish TLS breadth |
| FORWARD | [Swedish National Data Service DOI](https://doi.org/10.71540/89rs-s553) | Dense forest, boulders, and forestry disturbance |
| Stordalen | [SITES handle](https://hdl.handle.net/11676.1/xdEqbdjKjwib52XtBLDNaUbh) | 5 cm peat visible-surface exemplar |
| Swedish national ALS | [Lantmateriet product page](https://www.lantmateriet.se/en/geodata/our-products/product-list/laser-data-download-forest/) | Context/parent geometry only |
| Finnish national ALS | [NLS product page](https://www.maanmittauslaitos.fi/en/maps-and-spatial-data/datasets-and-interfaces/product-descriptions/laser-scanning-data), [high-density terms](https://www.maanmittauslaitos.fi/en/laser-skanning-data/terms-of-use) | Context and explicit rejection as target truth |
| Helsinki cloud | [Zenodo DOI](https://doi.org/10.5281/zenodo.5578198) | Technogenic conversion candidate |
| Getaberget orthomosaics | [Zenodo DOI](https://doi.org/10.5281/zenodo.4719627) | Optical fracture context, not height truth |
| Gotland/Faro limestone | [Remote Sensing paper](https://doi.org/10.3390/rs15061667) | High-value request-only near-miss |
| Oulanka multispectral LiDAR | [arXiv preprint](https://arxiv.org/abs/2603.22230) | High-value unavailable fluvial near-miss |
