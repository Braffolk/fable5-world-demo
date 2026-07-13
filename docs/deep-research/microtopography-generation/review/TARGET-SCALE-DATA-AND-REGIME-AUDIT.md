# Target-Scale Data and Estonia Regime Audit

**Date:** 2026-07-13
**Scope:** Training and validation evidence for a learned, cook-side Estonia microtopography generator at 0.0625 m output spacing. This audit does not redesign the generator, packed format, streaming hierarchy, or runtime.

## 1. Decision

The current corpus cannot train or validate a national, beauty-grade Estonia microtopography model. It can support architecture development, foreign-domain representation pretraining, a peat-only morphology experiment, and conversion-pipeline research. It cannot support the claim that the generated 0.0625 m surface is realistic for every Estonian soil, substrate, landform, forest floor, shore, agricultural state, or disturbance regime.

The quantified result is stark:

- **Production-paired Estonia target sites:** 0.
- **Open target-scale Estonia height surfaces found:** 0.
- **Audited physical release regimes with any ready target-scale height exemplar:** 1 of 10, peat only, and that exemplar is foreign and unpaired.
- **Audited physical release regimes without a ready target-scale height exemplar:** 9 of 10.
- **Ready open high-resolution surface area in the cited/local corpus:** 309.1387 m2 of peat surface in 68 disconnected plots from the Moore archive.
- **Equivalent 0.0625 m cells:** about 79,140, versus 4,194,304 cells in one 128 m x 128 m LOD -2 chunk.
- **Equivalent area relative to one LOD -2 chunk:** 1.89%, all in one material family and none in Estonia.

The model architecture may be selected provisionally, but national Stage B training must be data-gated. Unsupported regimes must not be filled with decorative noise or mislabeled foreign DSM data. If the product requirement remains credible added detail across all Estonia, target acquisition is a release dependency rather than an optional future improvement.

This conclusion agrees with the highest-risk finding in both the [ML proposal](ML-SYNTHESIS-PROPOSAL.md) and the [deterministic/process proposal](deterministic-process-proposal.md), but is stricter because it checks the actual downloadable products, licenses, sensing modalities, plot extents, and surface semantics rather than treating a paper's nominal resolution as training readiness.

## 2. What Counts as Target Truth

The output grid is 0.0625 m, or 256 cells/m2. A raster with 0.1 m pixels contains only 100 cells/m2 and cannot directly supervise the 0.0625 m residual band. A point cloud with 1,000 total points/m2 is not automatically a 0.0316 m bare-ground surface: vegetation, repeated returns, view clustering, occlusion, classification, interpolation, and registration control the actual ground support.

The proposals' `<=3-5 cm` acquisition gate is appropriate for Stage B. It is an engineering gate, not a claim that every feature smaller than 5 cm is recoverable. At 0.0625 m output spacing, the smallest representable wavelength is approximately 0.125 m, and measurement support still needs to be finer than the output cell so the raster is not mostly interpolation.

A **production-supervised target** must have all of the following:

1. Real measured surface support at `<=0.03-0.05 m` nominal spacing after ground filtering, not merely a fine export grid.
2. A reported or independently measured vertical-error distribution small enough to distinguish the class's target relief amplitudes.
3. Per-cell support and confidence masks. Interpolated holes are not truth and receive no height loss.
4. Explicit surface semantics: mineral soil, moss/peat surface, exposed stable roots, clasts/boulders, loose litter, deadwood, crops, water, and built objects cannot be silently mixed.
5. Vegetation removal or classification appropriate to the desired surface. Optical SfM sees the visible top surface; it does not see ground beneath vascular vegetation.
6. Registration to the temporally closest source DTM, orthophoto, structural vectors, soil, surficial geology, bedrock, hydrology, land use, and acquisition metadata.
7. A reuse license that permits the intended training and model distribution.
8. Site identity sufficient to split entire locations and campaigns. Overlapping random patches are not independent validation.

The audit uses four evidence roles:

- **P, production target:** target-scale surface plus paired Estonia inputs, uncertainty, semantics, and usable license.
- **E, high-resolution exemplar:** real target-scale surface, but foreign, unpaired, too small, or otherwise insufficient for production supervision.
- **R, raw conversion candidate:** dense point cloud or DSM that may become an exemplar only after classification, error analysis, rasterization, and human QA.
- **C, calibration/protocol evidence:** morphology statistics, class labels, coarser surfaces, unavailable data, or a demonstrated acquisition method. It cannot be used as Stage B height truth.

No audited source qualifies as `P`.

## 3. Primary Cited Evidence

### 3.1 Moore peatland SfM archive: real target-scale data, but only 309 m2 of one family

[Moore et al. 2019](https://doi.org/10.5194/bg-16-3491-2019) and its open [Zenodo dataset](https://doi.org/10.5281/zenodo.2545675) are the strongest target-scale source in the current library.

| Property | Verified finding |
|---|---|
| Regime | Northern bog and poor-fen moss microtopography; not mineral soil, forest floor, agriculture, rock, coast, or road |
| Geography | Canada, United States, and Sweden; nine plot-analysis peatlands plus the Red Earth Creek campaign, no Estonia |
| Modality | Ground photogrammetry / SfM |
| Surface preparation | All vascular vegetation manually clipped with scissors and pruners to expose the moss surface |
| Point density | 3-59 points/cm2, or 30,000-590,000 points/m2 |
| DEM | 0.01 m grid, natural-neighbor interpolation, then a 0.03 m x 0.03 m mean filter |
| Accuracy | Laboratory RMSE below 0.01 m in x/y/z; elevation median absolute difference 0.004 m in laboratory validation and 0.018 m in field validation |
| Plots | 50 Red Earth Creek plots plus 18 plot-analysis DEMs; 68 disconnected plots total |
| Area | Archive-derived finite-cell area 192.7204 m2 for the 50 Red Earth Creek DEMs and 116.4183 m2 for the 18 other DEMs; 309.1387 m2 total |
| License/access | Open, CC BY 4.0, locally archived |
| Role | `E`: high-resolution peat exemplar and site-held-out calibration source |

The paper reports that an average aggregate of 32 m2 from ten randomly located plots captured about 95% of the elevation variance at one unpatterned peatland. That is a within-site sampling result, not evidence that 32 m2 trains a spatial generator, covers patterned bogs, covers all peatland types, or generalizes to Estonia. The archive's plots are mostly only a few metres across, so they cannot teach 128 m context, transitions, drainage organization, or class boundaries by themselves.

The target is also a moss surface after vascular clipping, not a conventional bare-earth DTM. That is useful if the intended terrain surface includes stable moss/peat microforms, but the semantic distinction must remain explicit.

### 3.2 HuHoLa: a classifier and request-only 3 cm survey, not an open target corpus

[Noumonvi et al. 2025](https://doi.org/10.1016/j.ecolmodel.2025.111212) used four northern Swedish mire sites. The core landscape DEMs came from airborne LiDAR at 20 points/m2 per channel and were produced at 0.30 m and 0.50 m. A DJI Phantom 4 SfM survey produced a 0.03 m DEM for resolution sensitivity tests.

The paper does not establish a downloadable, licensed 0.03 m training corpus with target area, vertical accuracy, ground-confidence masks, and paired source inputs. It states that the full data, including UAV images, are available on request. The [repository](https://github.com/bravemaster3/huhola) contains code and sample rasters, but the checked sample DEMs are 0.50 m, including the nominal 100 m/150 m examples. The repository also warns that inputs finer than 0.30 m become fragmented. No repository license file was found in the audited snapshot.

Role: `C` now; potentially `R` if the full SfM data, terms, accuracy report, and masks are obtained.

### 3.3 Korotkov/Ilyasov bog LiDAR: broad coverage but coarser than Stage B truth

[Korotkov et al. 2026](https://doi.org/10.3390/drones10020121) surveyed 4.64 km2 of western Siberian ombrotrophic bog using UAS LiDAR at about 100 total points/m2. Processing classified `ground` and `low points` at 0.10 m resolution, and the published final microrelief product is 0.09 m. The data are available only on request.

This source is valuable for 0.5-3 m microform organization, 0.30-0.40 m hummock heights, and 0.04-0.23 m class thresholds. Its total density is below the 256 output cells/m2 before ground filtering, its classified support is 0.10 m, and the paper does not provide target-scale vertical-error evidence. It is not 0.0625 m truth.

Role: `C` for peat statistics, labels, and landscape organization.

### 3.4 Pawlik forest pit-mound DTM: relevant 0.025 m result, no training artifact

[Pawlik et al. 2024](https://doi.org/10.1016/j.geomorph.2024.109283) directly compared a 1 m ALS DTM with a 0.025 m TLS DTM at a pit-mound forest hillslope in the Karkonosze Mountains, Poland. It demonstrates that 1 m products lose or distort pit-mound morphology and that a target-scale TLS DTM materially changes geomorphometric interpretation.

No open companion DTM or point cloud, reuse license, complete target area, or paired source package was found. One specialized site also cannot establish distributions for Estonian till, sand, peat forest, different stand histories, root systems, animal disturbance, or soil depth.

Role: `C` for acquisition need and forest morphology evaluation, not training.

### 3.5 Lapinjarvi TLS: strong raw Nordic candidate, not bare-ground truth

The official [LUKE Lapinjarvi dataset](https://opendata.luke.fi/en/dataset/urn-nbn-fi-att-6a42126a-33f0-4a25-84b8-71fadc2374c8) contains 18 southern Finnish boreal forest plots under CC BY 4.0.

| Property | Verified finding |
|---|---|
| Modality | Leica P40 multi-station TLS, five stations per plot |
| Instrument sampling | 3.1 mm point spacing at 10 m |
| Registration | 1-3 mm co-registration error |
| Distributed cloud | Cropped to 12.5 m radius and decimated to at most one point per scan in a 5 mm cube |
| Intended reliable core | 9 m radius due to scan layout and occlusion limits |
| Nominal area | 4,580.4 m2 over 18 reliable 9 m-radius cores; 8,835.7 m2 over the distributed 12.5 m-radius extents |
| Conditions | Productive non-drained forest; plot metadata include mineral soil versus spruce mire, fertility, structure, and tree inventory |
| Missing target work | No supplied bare-ground DTM, no ground semantic labels, no per-cell support, no forest-floor occlusion analysis, and no manual ground validation |
| Role | `R` |

The 1-3 mm figure is scan co-registration, not forest-floor elevation accuracy. Lowest-point filtering would mix ground, litter, low vegetation, roots, deadwood, and occlusion-driven holes. This source is worth converting and auditing, but it must not be declared truth before that work.

### 3.6 Existing learned-terrain datasets do not solve target acquisition

The terrain and DEM papers in [SOURCES.md](../library/SOURCES.md) and [ML-SOURCES.md](ML-SOURCES.md) provide architectures, losses, priors, or coarse training data. They do not provide a 0.0625 m Estonia ground distribution:

- GrounDiff's nominal 0.10 m source is DSM-to-DTM filtering and can contain TIN/interpolation targets; its other source is 1 m.
- GSRMTL uses 0.20 m imagery with a 2 m DEM target and reports metre-scale errors.
- Argudo's aerial-guided terrain SR, GATA, D-SRGAN, StyleDEM, Terrain Diffusion, TerraFusion, and related terrain generators operate at metre to tens-of-metres terrain scales or on authored synthetic terrains.
- AHN4 at 0.50 m, Maa-amet DTM at 1 m, orthophoto at 0.20-0.40 m, and the national soil/geology vectors are conditions or Stage A evidence, not Stage B targets.
- Diffusion inverse-problem papers constrain observations. They do not supply missing soil-, substrate-, or process-conditioned target data.

## 4. Additional Open Candidates Found

These sources were not enough to reverse the decision, but they materially improve the acquisition and pretraining inventory.

| Source | Geometry, access, and license | Target-scale issue | Permitted role |
|---|---|---|---|
| [Harmonized European forest TLS collection](https://doi.org/10.5281/zenodo.18670608) | 121 plots from five campaigns; each 50 m x 50 m; registered, filtered, 0.01 m-subsampled point clouds; 302,500 m2 nominal; CC BY 4.0 | Forest-structure clouds, not bare-ground DTMs; no ground semantic class, per-cell support, or validated forest-floor error in the record | `R`; valuable large-scale forest conversion candidate |
| [NIBIO MLS](https://doi.org/10.5281/zenodo.12754726) | 16 Norwegian forest plots of about 250 m2, approximately 4,000 m2 total; manually labeled ground/vegetation/deadwood/stems | MLS/SLAM surface error and ground raster quality are not established; repository declares AGPL/GPL terms and derivative-sharing obligations that require legal review before model training | `R` for technical experiments; not production until license and error gates pass |
| [Point Beach, Wisconsin, August 2020](https://doi.org/10.5069/G9X63K4B) | Coastal SfM, 0.17 km2, 117.3M points, 698 total points/m2, 0.10 m raster, CC BY 4.0 | Total nominal spacing is about 0.038 m, but cloud is unclassified, raster is coarser than output, water/vegetation contamination and vertical uncertainty are unresolved | `R` for beach/coastal sand after QA; not direct target now |
| [Virginia Tech StREAM Lab 2026](https://doi.org/10.5069/G9NZ85W7) | Leaf-off UAS LiDAR, 0.28 km2, 1,295 total points/m2, 100.3M classified ground points, 0.10 m DTM, CC BY 4.0 | Average ground density is about 359 points/m2, nominally 0.053 m spacing if uniform; it is not uniform, the released DTM is 0.10 m, and stream bathymetry is interpolated | `R` for fluvial/floodplain conversion; interpolation and water must be masked |
| [Panamint Valley SfM surfaces](https://doi.org/10.5069/G9PC30M4) | 20 arid alluvial DSMs, 3.2 km2 total, downloadable 0.02-0.05 m rasters, CC BY 4.0 | DSM rather than classified ground; arid alluvium and fault scarps are not Baltic glaciofluvial/till surfaces; no target-scale vertical error in the record | Geometry representation pretraining and algorithm stress tests only |
| [Sky Valley SfM](https://doi.org/10.5069/G9BV7DSK) | 0.16 km2, 1,159 points/m2, 0.02-0.03 m DSM, CC0 | Arid tectonic surface, unclassified DSM, no Estonia condition pairing | Representation pretraining only |
| [Bedrock cliffs, southern California](https://doi.org/10.5069/G96H4FKG) | Exposed rock SfM, 0.01 km2, rasters from 0.001-0.03 m, CC0 | Different lithology/climate; steep true-3D surfaces and patchy soil do not map cleanly to a heightfield; no production vertical-error statement | Rock fracture/roughness calibration and representation pretraining |
| [Lauterbrunnen rockwalls/talus TLS](https://doi.org/10.5069/G9GQ6VX7) | 0.56 km2, 232.8M points, 418 points/m2 | Unclassified true-3D cloud, Alpine material/process mismatch, no provided reuse license, no stated target error | Inspection only until written reuse rights; morphology calibration at most |
| [Day Creek headwater SfM](https://doi.org/10.5069/G9F47MBX) | 0.18 km2, 169 points/m2, CC0 | Reported 1-sigma cloud difference 0.21 m and no vegetation filtering; target error exceeds the desired band | Explicit negative example: density/resolution does not imply truth |
| [McKittrick carbonate cliffs](https://doi.org/10.5069/G99C6VM6) | 0.84 km2, 40 points/m2, CC0 | Reported 1-sigma cloud difference 0.96 m, holes at overhangs, no vegetation filtering | Stratigraphic visualization only, not microtopography truth |

The open forest collection is the largest promising raw source. It can reduce the cost of learning point-cloud cleanup, surface representations, and generic forest morphology. It does not remove the need for Estonia ground classification and holdouts. Likewise, the OpenTopography sources show that public centimetre-grid DSMs exist, but their scale label cannot override wrong material, wrong climate, unmodeled vegetation, missing uncertainty, or true-3D geometry that the heightfield cannot represent.

## 5. Regime-by-Regime Estonia Gap

The regimes below are not mutually exclusive. A till forest, cultivated glaciofluvial soil, peat woodland, limestone coast, and sandstone riverbank require mixtures of substrate, soil, hydrology, land use, vegetation, and disturbance conditions. A single winning class label would erase exactly the variance the model is meant to create.

### 5.1 Mineral soils by texture, layering, stoniness, and humus

**Available condition evidence:** The national 1:10,000 soil map and EstSoil-EH provide soil mixtures, layered texture, humus, stoniness, fertility, and ecohydrological properties. The [conditioning audit](ESTONIA-CONDITIONING-DATA-AUDIT.md) shows that the current cook discards important mixture and layer fields. These maps are priors, not surface geometry.

**Target evidence:** None of the cited target-scale datasets pairs centimetre ground surfaces with Estonian soil profile, texture, stoniness, humus, moisture, and land use. Lapinjarvi supplies coarse forest site/fertility categories but no validated ground DTM or detailed soil profile. The agricultural papers below demonstrate texture effects at plot scale, not national target coverage.

**Gap:** 0 Estonia target sites; 0 ready foreign target corpora spanning the needed mineral-soil attributes.

**Mandatory acquisition:** Stratified Estonian mineral-soil plots covering the materially different source combinations, with lab/field soil profile metadata and seasonal moisture state. The strata should be chosen from actual national mixtures and prevalence, not one plot per simplified soil code.

### 5.2 Peat bog and fen microforms

**Available evidence:** Moore provides 309.1387 m2 of target-scale moss surfaces; HuHoLa provides class/process diagnostics and a request-only 0.03 m survey; Korotkov provides 4.64 km2 of 0.09-0.10 m organization and microform thresholds.

**Gap:** 0 Estonia target sites. Moore lacks patterned ridge-pool systems and is too spatially fragmented for landscape organization. Foreign bog/poor-fen data do not establish Estonian bog, fen, aapa-like, drained, restored, wooded, or peat-extraction states.

**Permitted use now:** Peat representation pretraining, Moore site-held-out morphology metrics, and classifier/label development. Do not random-split Moore plots from the same site.

**Mandatory acquisition:** Estonia mire sites spanning hydrologic type, drainage/restoration state, wooded versus open surface, ridge/hollow/lawn/string/flark organization, and water-table season. SfM is allowed only where vascular cover is removed or naturally sparse; otherwise use dense multi-return LiDAR/TLS with manual surface QA.

### 5.3 Till and moraine

**Available evidence:** Pawlik proves that 1 m ALS loses forest pit-mound form. Nordic/European forest TLS clouds may contain till surfaces, but they lack verified substrate pairing and release-ready ground rasters.

**Gap:** 0 target surfaces explicitly labeled as Estonian/Baltic till or moraine with clast abundance, soil depth, moisture, land use, and stand/disturbance history.

**Mandatory acquisition:** Estonian till/moraine sites across smooth and hummocky relief, fine- versus coarse-rich till, boulder/stone abundance, forest/open/agricultural cover, and drainage. Stable clasts and boulders must be retained as surface geometry rather than filtered as outliers.

### 5.4 Glaciofluvial and sandy terrain

**Available evidence:** Point Beach provides a coastal-sand conversion candidate. Panamint and Sky Valley provide large centimetre DSMs of arid alluvium. None represents Baltic glaciofluvial deposits, humid dunes, pine-forest sand, eskers, outwash, or Estonian management.

**Gap:** 0 class-matched target surfaces. The large arid datasets must not dominate amplitude, channel form, clast distribution, or weathering statistics.

**Mandatory acquisition:** Estonian esker/outwash/dune/sandy-plain sites with exposed, vegetated, forested, tracked, and eroded states. Use the foreign DSMs only to initialize geometric encoders or test tiling, followed by Estonia fine-tuning and held-out validation.

### 5.5 Limestone, alvar, and karst

**Available evidence:** The southern California rock source resolves fractures, and McKittrick includes carbonate cliffs, but the latter has metre-scale registration error. Getaberget's Finnish 0.55 cm orthomosaics are imagery only, not elevation. None is an alvar pavement target.

**Gap:** 0 Estonia target sites and 0 verified foreign alvar/low-relief limestone-pavement height corpora. True karst cavities and overhangs are not representable by the current heightfield.

**Mandatory acquisition:** Estonian exposed/shallow limestone and alvar plots across fracture, clint/grike, thin-soil, vegetation, dissolution, and coast states. Acquire true 3D data, then derive a 2.5D surface with explicit overhang/cavity confidence instead of projecting multiple elevations into arbitrary spikes.

### 5.6 Sandstone, clay, and silt cliffs and talus

**Available evidence:** Open rockwall, cliff, and talus clouds demonstrate acquisition techniques and true-3D morphology. They have incompatible lithology/climate, incomplete licensing, or inadequate vertical accuracy. The EGT 1:200,000 fallback identifies Burtnieki Formation sandstone with siltstone/clay interbeds at Taevaskoda, but it is a broad condition only.

**Gap:** 0 Estonia target sites. There is no measured target distribution for erodible Devonian sandstone ledges, seepage, bank undercut, clay/silt interbeds, collapse blocks, or talus. The known 1 m Taevaskoda DTM artifacts further prevent using the source as self-supervision.

**Mandatory acquisition:** Oblique TLS plus low-altitude photogrammetry or UAS LiDAR at several Estonian cliffs and banks, including top, face, toe, talus, vegetation, seepage, and adjacent river edge. The 3D survey must be converted to heightfield-valid surface with visibility and overhang masks. Suur Taevaskoda remains an untouched visual/generalization site rather than a hand-authored training patch.

### 5.7 Fluvial banks, shorelines, and floodplains

**Available evidence:** The Virginia StREAM Lab has dense classified ground and repeated leaf-off surveys; Point Beach has dense coastal SfM. Both release 0.10 m rasters. The Virginia DTM explicitly interpolates stream bathymetry.

**Gap:** 0 Estonia target sites and no ready `<=0.05 m` paired bank/floodplain target. Water, emergent vegetation, saturated sediment, undercut banks, and seasonal change are major failure modes.

**Mandatory acquisition:** Estonian riverbanks, natural and engineered shorelines, floodplains, springs/seepage, erosion/deposition states, and representative coast types. Survey in leaf-off/low-water conditions where possible. Water and interpolated bathymetry remain invalid target cells unless independently measured. ETAK shores, rivers, escarpments, and ditches constrain structure but do not supply fine shape.

### 5.8 Agricultural roughness, tillage, and ditches

The primary literature proves that this class is structured rather than generic noise:

- [Gilliot et al. 2017](https://doi.org/10.1016/j.compag.2017.01.010) produced 0.001 m DEMs at 32 measurement sites across four tillage levels and measured about 1.5 mm positional/elevation accuracy on artificial models. The artificial-model error is not field-target accuracy, and no open licensed DEM corpus was located.
- [Herodowicz-Mleczak et al. 2022](https://doi.org/10.1029/2021MS002578) generated 0.001 m GeoTIFF DEMs for 126 test areas across 18 Polish fields and five tillage tools. It found tillage method most important for macro roughness and soil texture important for micro roughness. An open target download and reuse license were not established.
- [Azizi et al. 2021](https://doi.org/10.3390/s21134386) collected 156 stereo pairs over sandy-loam soil and three tools. Data are request-only, and the evaluation is not sufficient to treat the depth maps as metric target truth.

**Gap:** 0 Estonia target sites and 0 open production-ready agricultural DEM corpus. The available studies are protocol and causal-conditioning evidence.

**Mandatory acquisition:** Estonian fields paired by soil texture, moisture, tillage implement, direction, time since operation/rain, seedbed/crop state, compaction, wheel tracks, headlands, and drainage ditches. Because this geometry is ephemeral, source orthophoto and DTM dates must be recorded; an old target must not supervise a newer field state.

### 5.9 Forest pit-mound, roots, animals, and low disturbance

**Available evidence:** Pawlik supplies one 0.025 m scientific result without data. Lapinjarvi supplies 18 Nordic raw plots. The harmonized forest collection supplies 121 large European raw plots. NIBIO supplies approximately 4,000 m2 of manually labeled Norwegian MLS ground but has unresolved metric accuracy and strong-copyleft licensing.

**Gap:** 0 release-ready forest-floor targets and 0 Estonia target sites. The current sources were designed primarily for stems/canopy/segmentation, not forest-floor truth. No audited corpus pairs pit-mound age/orientation, roots, animal disturbance, deadwood policy, stand history, soil, substrate, moisture, and slope.

**Mandatory acquisition:** Multi-position TLS or accurately registered MLS in Estonian forest regimes, with nested manual checks and site history. Acquisition must define whether stable exposed roots, stumps, coarse deadwood, loose litter, and animal diggings belong in the heightfield, a separate object layer, or the invalid mask. Lowest-point filtering alone is forbidden as target creation.

### 5.10 Roads and human disturbance

**Available evidence:** ETAK and orthophoto locate roads, paths, quarries, ditches, and visible tracks. No cited target-scale corpus covers Estonian paved roads, gravel roads, forest tracks, trails, cut/fill slopes, quarries, drainage, rutting, maintenance, and abandonment states.

**Gap:** 0 target sites and no class-matched ready foreign source.

**Mandatory acquisition:** Representative Estonian road/path/quarry/disturbance profiles stratified by construction material, use, moisture, maintenance, rutting, drainage, and adjacent soil. Paved surfaces should remain structurally smooth except for measured grade/crown/damage; they must not receive natural-soil noise.

### 5.11 Water exclusions

Open water is not a microtopography training regime. Optical texture, waves, glare, ice, emergent plants, and LiDAR returns must never become static terrain displacement.

Water contributes:

- a hard invalid/exclusion mask for height learning;
- a shoreline and bank structural constraint;
- optional independently surveyed bathymetry, explicitly labeled and separated from interpolated water gaps.

The current national water/shore vectors and orthophoto are adequate conditions, not height truth. The synthetic surface must preserve water connectivity and bank continuity without inventing bottom topography from reflections.

## 6. What Can Be Trained Before Acquisition

### 6.1 Safe work now

1. Build a versioned high-resolution registry, surface-semantic schema, license ledger, point-support masks, and site-level split machinery.
2. Convert and audit Lapinjarvi and a small subset of the open forest collection without calling the result truth until manual checks pass.
3. Train generic geometric representations or denoisers on permissively licensed foreign surfaces, with source identity and physical regime retained.
4. Run a peat-only Stage B proof using Moore, with whole peatlands held out and no claim of Estonia production readiness.
5. Prototype Stage A/source-repair using coarser paired data, structural vectors, and real degradation studies. Release-quality repair still needs Estonia high-resolution pairs around known DTM defects.
6. Compare diffusion, GAN, and deterministic challengers on exactly the same evidence. A model family cannot compensate for a missing target distribution.

### 6.2 Unsafe work

- Upsampling Maa-amet 1 m DTM, AHN 0.5 m, HuHoLa 0.3/0.5 m, or a 0.1 m DTM and labeling it 0.0625 m truth.
- Treating point-cloud density, export pixel size, scan co-registration, or photogrammetric reprojection error as ground-elevation accuracy.
- Training on DSMs without object/vegetation/water masks.
- Using orthophoto edges as pseudo-height.
- Random patch splitting within one acquisition site.
- Letting the national class imbalance make common smooth terrain erase rare but important structures.
- Filling unsupported classes with generic noise and calling the model national.
- Training on model-generated pseudo-targets and then using agreement with those targets as validation.

## 7. Mandatory Estonia Acquisition Program

No paper establishes a universal number of plots that is sufficient for a national generative model. This audit therefore separates a minimum go/no-go gate from a claim of sufficiency.

### 7.1 Minimum release gate per regime

Before a physical regime can emit Stage B geometry, require at least **three geographically separate Estonian acquisition sites**: two development sites and one untouched site-level holdout. This is a project minimum, not a literature-derived sufficiency theorem. Training continues to add independent sites until regime-specific held-out errors, morphology distributions, and blind visual ratings stop improving materially; patch count from the same site does not substitute for new sites.

One site must not be split across train and validation, even if it yields thousands of patches. Acquisition campaign and sensor should also be separated where the available count permits. Taevaskoda remains outside training and hyperparameter selection if it is to be a meaningful generalization target.

With ten non-water regime families, the absolute logical floor is 30 Estonia site-regime acquisitions. Mixed sites can cover more than one regime only when each component has enough valid measured area and explicit labels. This floor is deliberately not called sufficient; the heterogeneous mineral-soil, forest, agricultural, and shoreline families will almost certainly require more independent states.

### 7.2 Nested spatial design

The Moore archive demonstrates why many tiny plots are not a spatial corpus. Each site should combine:

- a contiguous context survey large enough to include landform organization, transitions, drainage, and the model's actual condition halo, using dense UAS LiDAR or photogrammetry where the surface is visible; and
- nested `<=0.03-0.05 m` verified microplots or dense ground returns for Stage B supervision, using TLS, close-range SfM, MLS, or low-altitude LiDAR appropriate to the cover.

The context survey may be coarser than Stage B if every Stage B loss is masked to genuinely supported fine cells. Disconnected 2-10 m microplots can teach local morphology, but cannot alone teach 128 m-scale arrangement or class transitions.

### 7.3 Required paired record

Every target site must retain:

1. Raw sensor data and immutable processing provenance.
2. Target surface, point-support count, incidence/view geometry, interpolation mask, and human-QA mask.
3. Independent control/check measurements and horizontal/vertical error distributions.
4. Surface semantic labels and explicit include/exclude policy.
5. Temporally closest Maa-amet DTM source sheet/vintage and raw 1 m samples.
6. Dated RGB/CIR orthophoto and canopy/visibility evidence.
7. ETAK water, shore, escarpment, road, ditch, quarry, wetland, peat-field, land-cover, and mapped-boulder features.
8. Full soil mixture, layered texture, humus, stoniness, fertility, and ecohydrology fields.
9. EGT 1:50,000 geology/geomorphology with coverage confidence and 1:200,000 national fallback.
10. Land use, management/disturbance history, vegetation/stand state, moisture/water-table state, weather, and acquisition season.
11. License, attribution, redistribution, and trained-model rights.

The [Estonia conditioning-data audit](ESTONIA-CONDITIONING-DATA-AUDIT.md) already establishes that these conditions exist at different support scales. Coarse polygons select a distribution; they must not print their boundaries into centimetre geometry.

### 7.4 Sensor rules

- **Open bare soil, fields, exposed peat, rock, and beaches:** close-range or very-low-altitude SfM is acceptable with metric control, stable illumination, and explicit vegetation/water masks.
- **Forest and scrub:** multi-position TLS or well-controlled MLS is primary. UAS LiDAR is useful for context only where actual ground-return density and error pass the cell-level gate.
- **Cliffs and talus:** combine oblique TLS/SfM with nadir context. Preserve the true 3D source and record where a single-valued heightfield projection is invalid.
- **Wetland and shore:** LiDAR plus ground checks during suitable water conditions; never infer submerged ground from an empty return or optical reflection.
- **Agriculture:** acquire repeated states because tillage and rainfall change the target. Pair operations, soil moisture, and imagery dates.

### 7.5 Quality acceptance

A converted surface is accepted only when:

- error is measured against independent controls, not inferred from export resolution;
- confidence is reported per cell and by regime;
- no-data and interpolation remain distinguishable;
- visual inspection catches vegetation islands, lowest-point pits, scanner shadows, alignment bands, water spikes, and edge extrapolation;
- target-area statistics exclude invalid cells;
- the same processing applied to a different site does not require hand-tuned scene fixes.

## 8. Simulation, Weak Supervision, and Domain Adaptation

### 8.1 Simulated proxy data

Process simulations and authored examples can expand known variation only after calibration to real target surfaces. They are appropriate for:

- pretraining an encoder or denoiser;
- rare-event augmentation after real amplitude, orientation, spacing, decay, and co-occurrence distributions are measured;
- controlled counterfactuals for condition sensitivity;
- testing invariance, tiling, and measurement-consistency code.

They are not target truth and may not dominate any release regime. Generic fBm, hash noise, sine fields, and uncalibrated erosion are excluded.

### 8.2 Synthetic degradation

Fine surfaces may be degraded to train super-resolution only after the degradation operator is fitted to real paired fine/Maa-amet observations by acquisition vintage, canopy, slope, water, and classifier confidence. Bicubic, average-pool, or ideal low-pass pairs are baselines. Real-GDSR's central warning applies: a mathematically downsampled target does not reproduce the errors of a real elevation product.

### 8.3 Weak labels

Orthophoto, ETAK, soil, geology, hydrology, and land-use layers can supervise classes, boundaries, exclusions, and causal conditions. They cannot supervise exact height. Orthophoto is useful for visible crop rows, erosion, tracks, shorelines, exposed material, and recent disturbance only under a visibility/date mask.

Point clouds can provide weak height labels only where point support and semantic confidence are high. Interpolated cells, canopy gaps, water, and true-3D conflicts are masked, not pseudo-labeled.

### 8.4 Foreign-domain pretraining and adaptation

Foreign data should retain country, site, sensor, substrate, soil, climate, vegetation, and processing identity. Use it for representation pretraining or class-specific initialization, then fine-tune on Estonia. A foreign site can never be the sole held-out validation for an Estonia release class.

Domain adaptation passes only if an untouched Estonia site shows:

- improved blind ground-level realism;
- correct class-conditioned height, slope, normal, curvature, local-relief, spacing, orientation, and spectrum distributions;
- no imported arid, Alpine, tropical, or scanner-specific motifs;
- no loss of source repair, shoreline continuity, vegetation grounding, or water exclusion.

Self-training on Estonia predictions is allowed only as consistency regularization against measured conditions. Predictions never become ground truth.

## 9. Held-Out Validation Design

Maintain four distinct validation roles:

1. **Foreign method validation:** Moore peatlands and selected permissively licensed raw conversions test whether the pipeline can learn measured morphology at all.
2. **Estonia regime validation:** complete untouched Estonia sites for every enabled physical regime.
3. **Sensor/campaign validation:** where possible, hold out an acquisition campaign or sensor to detect scanner/processing signatures.
4. **Visual generalization landmarks:** Taevaskoda and other recognizable sites remain outside training, tuning, and scene-specific correction.

Report results per regime and site, not only nationally averaged:

- valid target area and point support;
- height, slope, normal, curvature, and local-relief error distributions;
- multiscale spectra/variograms and typed object statistics such as hummock, clod, rut, pit-mound, clast, ledge, and channel spacing/orientation;
- source-DTM correction separated from added residual quality;
- repetition and seam detection;
- blind rendered ground-level preference with identical camera, light, materials, and vegetation;
- failure maps for water, canopy, cliffs, roads, transitions, and low-confidence conditioning.

A visually attractive foreign-style hallucination that fails Estonia target distributions does not pass. Conversely, low pointwise error with flat, repetitive, or material-incorrect ground does not pass.

## 10. Consequences for the Learned Backbone

1. **A national model is not data-ready.** Current evidence cannot support a single model that adds correct fine detail everywhere in Estonia.
2. **Peat is the only immediate Stage B morphology proof.** Even there, Moore is foreign, tiny, and unpaired, so the result is an algorithm proof rather than production.
3. **Forest is the best raw-data conversion opportunity.** Lapinjarvi, the harmonized European collection, and NIBIO can establish whether reliable forest-floor surfaces can be extracted, but none is production truth as downloaded.
4. **Fluvial/coastal and rock sources are useful pretraining candidates.** Their raw geometry and permissive licenses do not make their physical domains Estonian.
5. **Agriculture has strong causal literature but no open ready corpus.** Its acquisition can follow proven millimetre photogrammetric protocols, with Estonian texture, tillage, moisture, and dates.
6. **Condition maps are adequate to design acquisition strata.** They are not substitutes for targets.
7. **Class-specific activation is mandatory.** Stage B may emit a regime only after that regime passes Estonia site-held-out data and visual gates. If full national activation is a hard product requirement, all ten acquisition gaps are blockers.

The decisive next action is not another synthesis heuristic. It is a small, versioned target registry plus a three-regime acquisition/processing pilot: one exposed easy surface, one peat surface, and one forest surface. That pilot must prove metric control, surface semantics, ground classification, pairing to Maa-amet conditions, legal reuse, and site-held-out training before scaling to the full regime matrix.

## 11. Source Notes

The local source ledger is [library/SOURCES.md](../library/SOURCES.md), with deterministic/process-specific notes in [SOURCES-DETERMINISTIC-PROCESS.md](../library/SOURCES-DETERMINISTIC-PROCESS.md). The Moore area totals in this audit were calculated from finite `z` cells in the 50 individual Red Earth Creek MAT files and the 18 named surfaces in `DEMs.mat`, using the archive's 0.01 m grid. Bounding boxes and NaN cells were not counted as valid target area.

Article open access does not automatically grant reuse rights to unarchived request-only datasets. Repository code licenses also do not automatically cover separately supplied imagery or point clouds. Every acquired source therefore needs an explicit dataset-level license record before entering production training.
