# Estonia Conditioning-Data Audit

**Status:** evidence audit in progress; not an architecture recommendation
**Date checked:** 2026-07-13
**Scope:** cook-side inputs capable of conditioning or correcting a national 0.0625 m packed terrain result

## 1. Why this audit exists

The rejected LUKE quilt treated an unclassified foreign surface exemplar and a few broad masks as though they were enough to generate convincing Estonian microtopography. They are not. A national generator must distinguish the physical regimes that create surface form: exposed or shallow bedrock, surficial deposit, soil profile, geomorphic process, hydrologic position, land use and management, vegetation, and mapped structural constraints such as banks, cliffs, shores, ditches, roads, and quarries.

These inputs are conditioning evidence, not centimeter geometry truth. A 1:200,000 geology polygon cannot place a 6 cm stone. It can select or weight a scientifically compatible generative family. Conversely, a 20 cm orthophoto can locate visible boundaries and management traces but cannot directly recover ground height under canopy or resolve an occluded cliff face.

## 2. Inputs already present in `asset-gen`

| Source | Native/current representation | Useful information | Current limitation |
|---|---:|---|---|
| Maa- ja Ruumiamet DTM | 1 m GeoTIFF | measured bare-earth authority | fallible observation; water, vegetation leakage, interpolation, holes, and grid-shaped boundaries require correction |
| Maa- ja Ruumiamet nDSM | 1 m GeoTIFF | above-ground height | not ground geometry; useful for confidence/occlusion and vegetation context |
| Maa- ja Ruumiamet CHM | source sheets, sampled by cook | canopy height/cover | not soil or landform evidence |
| ETAK | national GeoPackage | shores, rivers, cliffs/slopes, land cover, roads, ditches, quarries, mapped boulders | the rejected synthesizer used broad exclusion masks and did not reconstruct these structures |
| Mullastikukaart | 1:10,000 national vector polygons | soil mixtures, layered texture, humus, stoniness, fertility | current five-plane cook discards important source fields and polygon mixture weights |
| Forest registry | WFS polygons | stand composition/age context | vegetation evidence, not ground morphology by itself |

The CLI currently advertises `assetgen fetch --only soil` in an error message, but `fetch --only` accepts only `elevation`, `country`, `etak`, and `wfs`; soil acquisition is presently out-of-band. Orthophoto and EGT geology acquisition are not implemented.

## 3. Soil fidelity loss in the current cook

[Maa- ja Ruumiamet describes Mullastikukaart](https://geoportaal.maaamet.ee/est/ruumiandmed/mullastiku-kaart-p33.html) as a 1:10,000 open vector database covering nearly all Estonia except dense settlements, water, and soil-less islets. The downloaded SHP has 766,627 features and these relevant fields:

- `Sif1..Sif4` and `Osa1..Osa4`: up to four soil components and their shares;
- `Loimis1`, `Loimis2`: layered full texture descriptions;
- `Lihtloimis`: simplified texture;
- `Huumus`: humus or peat-horizon thickness;
- `Kivisus`: stones over 20 cm in the upper 30 cm;
- `Boniteet`: potential fertility after drainage/improvement.

Current `process/soil.py` reads only `Sif1`, `Lihtloimis`, `Kivisus`, and `Boniteet`. It discards mixture components and weights, both full layered-texture fields, and humus thickness. It also reduces `Lihtloimis` to the surface component before `/`. This representation is inadequate as the sole soil contract for a high-fidelity generator.

At the supplied Suur Taevaskoda point (`58.107506 N, 27.050242 E`, EPSG:3301 `679763.082, 6444796.565`), the source polygon contains:

```text
Siffer      LkI;L(k)I
Sif1/Osa1   LkI / 70
Sif2/Osa2   L(k)I / 30
Loimis1     l90-100/ls₁
Lihtloimis  l/ls
Huumus      0/15-20 2₁/6-8
Boniteet    37
```

The production decoder must preserve the source typography and meaning rather than flattening subscripts into ambiguous ordinary digits.

## 4. ETAK structural evidence at Taevaskoda

ETAK contains higher-value structural constraints than the rejected synthesis used:

- `E_102_nolv_j`: slope/escarpment lines, including `Looduslik järsak` and whether the feature is a shoreline escarpment;
- `E_103_pinnavorm_j/p`: mapped landform features where available;
- `E_203_vooluveekogu_a/j`: river polygons and centerlines;
- `E_204_kaldajoon_j`: explicit shoreline lines and shoreline type;
- land-cover, wetland, peat-field, road, ditch, quarry, and boulder layers.

At the supplied GPS point, source-space distances are:

| ETAK feature | Distance | Classification |
|---|---:|---|
| nearest `E_102_nolv_j` | 5.32 m | natural escarpment; shoreline escarpment = yes |
| Ahja `E_203_vooluveekogu_a` polygon | 1.85 m | Ahja river boundary |
| nearest `E_204_kaldajoon_j` | 1.85 m | clear shoreline |
| Ahja `E_203_vooluveekogu_j` centerline | 15.69 m | river |

This is direct evidence that the 1 m grid-shaped shore and cliff boundary need not be accepted as immutable. The cook can use mapped vector constraints plus DTM evidence to reconstruct a smooth, stable base boundary before generating smaller-scale detail. The exact reconstruction method remains a research/design question; the requirement is not.

## 5. Orthophoto availability and evidence boundary

[Maa- ja Ruumiamet orthophotos](https://geoportaal.maaamet.ee/est/Ruumiandmed/Ortofotod-p99.html) cover the country at 20-40 cm pixels, with 10-16 cm imagery in dense settlements. RGB, CIR, and forestry NGR products are available as open downloads with attribution. Roughly half the country is refreshed each year, so acquisition must retain capture date and reconcile temporal mismatch with DTM, ETAK, forestry, and agricultural state.

An orthophoto fetch is warranted for research and likely for production conditioning because it can supply sub-meter boundary, exposed-material, drainage, erosion, crop-row, vehicle-track, and recent-disturbance evidence absent from polygon maps. It is not a height source and cannot justify geometry under forest canopy, water, shadow, or an occluded/vertical face. The final architecture must prove that orthophoto conditioning improves held-out geometry and rendered beauty rather than merely copying albedo edges into height.

### 5.1 Acquisition inventory and cost evidence

The agency's [mass-download instructions](https://geoportaal.maaamet.ee/est/abi-ja-juhised/andmed/kuidas-on-voimalik-palju-kaardilehti-korraga-alla-laadida/86) publish `tomba_etak_avaandmed.xlsx` because orthophoto filenames contain non-derivable flight dates. The workbook downloaded on 2026-07-13 is 2,515,402 bytes with SHA-256 `fbef1433eff6116174cd1bbeca6e6550e093eeed794c14d583af77e03970b4b5`. Its `Ortofotod 2016-2025` sheet contains 12,052 RGB archive records for 2,111 distinct 1:10,000 sheets. Selecting the newest named flight per sheet yields 1,253 sheets from 2025, 534 from 2024, 55 from 2023, 253 from 2022, 6 from 2021, and 10 from 2020. A production fetch therefore needs a per-sheet acquisition manifest; it cannot infer one national date.

The Taevaskoda coordinate is in sheet `54472`. Official HTTP metadata gives these compressed archive sizes:

| Product | Latest listed acquisition | Compressed bytes |
|---|---:|---:|
| RGB GeoTIFF | 2025-07-18 | 207,351,332 |
| CIR GeoTIFF | 2024-05-22 | 133,805,748 |
| NGR GeoTIFF | 2024-05-22 | 148,206,299 |

The three-sheet pilot fetch is therefore about 489.4 MB before extraction and derived confidence/visibility masks. An evenly spaced 11-sheet-ID sample of newest RGB archives ranged from 71.6 MB to 248.8 MB and averaged 173.8 MB. Multiplying that sample mean by 2,111 sheets gives a provisional 367 GB compressed national RGB cache. This is a planning estimate, not a production budget: it excludes GeoTIFF expansion, historical acquisitions, CIR/NGR, retries, checksums, masks, and any duplicated working set. The spec must require a complete manifest-size scan and a measured crop/cache policy before national acquisition.

## 6. EGT geology and geomorphology

[Eesti Geoloogiateenistus publishes](https://gis.egt.ee/geoportaal/ruumiandmed) CC BY 4.0 vector packages for 1:50,000 bedrock, surficial deposits, and geomorphology, plus 1:200,000 national bedrock and surficial maps. [The license](https://gis.egt.ee/geoportaal/juhendid/Eesti_Geoloogiateenistuse_ruumiandmete_litsents.pdf) explicitly requires scale limitations and attribution to be retained.

### 6.1 1:50,000 source package

The downloaded 2026-04-06 packages contain:

| Layer family | Principal layers | Features / meaning |
|---|---|---:|
| bedrock | outcrop polygons, escarpments, contours, faults, valleys | 1,132 outcrop polygons; 777 escarpment lines; 3,224 bedrock-elevation contours |
| surficial geology | deposit polygons and thickness contours | 21,547 polygons; 12,530 thickness contours |
| geomorphology | landform polygons, lines, axes, points | 3,544 polygons; 335 lines; 775 axes; 3,149 points |

It is not complete national coverage. At the supplied Taevaskoda point there is no containing 1:50,000 bedrock, surficial-deposit, or geomorphology polygon; the nearest mapped polygon boundary is approximately 7.05 km away. Every use must carry an explicit coverage/confidence mask. Unmapped must never be silently interpreted as a geological class.

### 6.2 1:200,000 national fallback

The 2026-04-28 1:200,000 bedrock and surficial packages do cover the Taevaskoda point. The bedrock polygon is `Burtnieki kihistu / Burtnieki Formation`, index `2130100`. EGT's [geological base-map guide](https://www.egt.ee/media/85/download) defines it as sandstone with siltstone and clay interbeds. The surficial layer contains the point but uses coded `lito200`/`genees200` values; those codes must be decoded through the official schema or guide before implementation, never guessed.

The 1:200,000 fallback is a broad family prior only. It must not introduce polygon-edge geometry or pretend to locate local outcrops. Finer ETAK, soil, DTM-derived, orthophoto, and exemplar evidence controls local expression.

## 7. Mandatory data-contract requirements for proposals

Every candidate generator proposal must specify how it consumes or explicitly rejects each of these inputs:

1. corrected 1 m base surface plus a per-cell source-confidence/error mask;
2. ETAK hard and soft structures: water/shore, escarpment, ditch/road, quarry, land cover, wetland, peat field, and mapped boulders;
3. full soil mixture and vertical-profile fields, not only `Sif1`;
4. 1:50,000 EGT geology/geomorphology with a coverage mask;
5. 1:200,000 national geological fallback with low spatial authority;
6. dated RGB/CIR orthophoto evidence with shadow, canopy, water, and temporal-confidence masks;
7. DTM/nDSM/CHM agreement and disagreement as measurement-confidence evidence;
8. land-use and management state for agricultural and disturbed surfaces;
9. process-derived terrain descriptors at multiple scales, without using descriptors as a substitute for user-visible acceptance;
10. exemplar provenance, physical class, spatial scale, acquisition method, and transfer-confidence label.

Class transitions must be spatially coherent and physically mediated. Rasterizing old coarse polygon edges directly into 6.25 cm relief would replace one artifact with another.

## 8. Source archive and identity

Downloaded under `docs/deep-research/microtopography-generation/library/data/egt/`:

```text
075d90683faea7cb8694beb89a62d9ccc3e405446a2ae2bec4be10e0d10f950e  aluspohi-50k-2026-04-06.zip
bec627dd602323c5f3425e48cc5a4358da37483e52781e7aacf65907befb0b75  geomorfoloogia-50k-2026-04-06.zip
435b361972b82f0123e5a2d763e79df4c429d8cd86a1ebd92661d07ed19ef7a7  pinnakate-50k-2026-04-06.zip
6f16a4290d92f25c17c9371464d13e5758e93119e24c9e01a12f4c8e0bd44b07  geoloogiline-baaskaart-gdb-50k-2026-04-06.zip
8351f5e1ec60a60a8ce377d0c582b8550549ed92e4c0ac5536bc7681a0f057ee  aluspohi-200k-2026-04-28.zip
a4bd969e11d6869a34cc44f101050d7a9ab29c7e9aee558efab9f94f01a58f39  pinnakate-200k-2026-04-28.zip
```

The large source archive is a local research cache, not yet a proposed Git artifact or production fetch contract.
