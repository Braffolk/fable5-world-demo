# Taevaskoda and Estonia Public 3D Scan Hunt

**Audit date:** 2026-07-13
**Scope:** existing, no-new-acquisition sources for Suur Taevaskoda, the Ahja
valley, and Estonian terrain at useful geometric resolution
**Decision context:** there is no budget for a new survey campaign or a separate
non-heightfield cliff project. This audit therefore separates genuinely open data
from material that may be recoverable through a no-fee author permission request.

## Executive finding

The user's recollection is correct. Suur Taevaskoda has been photogrammetrically
surveyed at least twice:

1. Marko Kohv photographed and published a true-3D model in 2015.
2. Polina Shlykova and Marko Kohv made new drone and terrestrial surveys in 2024,
   built separate and merged models, and reprocessed the 2015 photographs for a
   change comparison.

The geometry itself is **not presently an open download**. The 2015 Sketchfab
model is view-only, explicitly non-downloadable, and has no license. The University
of Tartu repository exposes only Shlykova's thesis PDF, not the 2015/2024 images,
Pix4D projects, point clouds, or meshes. The thesis license does not silently license
those absent research data.

This changes the practical conclusion, but not by pretending the scans are open:

- the first action should be a no-fee data-and-license request to the authors, not
  a paid resurvey;
- exact-site public Maa- ja Ruumiamet raw ALS is immediately downloadable and
  legally usable for source repair, but is far too sparse for 6.25 cm morphology;
- other recoverable Estonian university campaigns, especially Selisoo bog, may
  supply high-density native examples if their authors release the files;
- several downloadable CC Attribution Estonian outcrop meshes can exercise a
  cliff-data pipeline, but their lithology and unreported errors disqualify them as
  Taevaskoda sandstone truth;
- no catalogued, openly downloadable Estonia-wide target-scale terrain corpus was
  found. Public national ALS materially reduces structural uncertainty; it does not
  remove the morphology evidence gap.

No new survey should be proposed while these access-recovery leads remain
unexhausted. The separate non-heightfield cliff project remains rejected. Existing
cliff scans may be preserved as research evidence without starting that project.

## Qualification labels

| Label | Meaning |
|---|---|
| **O** | Open download with verified rights broad enough for derivatives and commercial use |
| **V** | Publicly viewable, but no lawful geometry download or derivative right was verified |
| **R** | Research product is verified to exist and is a reasonable no-fee permission-request target, but is not currently an open data download |
| **S** | Can support source/structural repair after its measurement errors and classes are audited |
| **H** | Can condition or validate heightfield-valid morphology at the project's scale after qualification |
| **C** | Relevant only to a possible future true-3D cliff representation or its acquisition/processing pipeline |
| **I** | Visual/process reference only; cannot be used as geometry truth |

Point count, mesh triangle count, export pixel size, image GSD, and coordinate
precision are not treated as surface accuracy. A source receives **H** only when
actual surface support, semantics, registration, and error are adequate or can be
qualified from the raw files.

## 1. Exact Suur Taevaskoda campaigns

### 1.1 Shlykova and Kohv 2024 photogrammetry

**Record:** Polina Shlykova, *Suure Taevaskoja liivakivipaljandi kaardistamine
fotogramm-meetria abil*, University of Tartu bachelor's thesis, 2024,
[handle 10062/100019](https://hdl.handle.net/10062/100019),
[repository item](https://dspace.ut.ee/items/4b9cffa2-65c4-4de0-b383-d146efedb6bf).

**Verified site and extent:** the main Suur Taevaskoda Burtnieki Formation
sandstone wall on the right bank of the Ahja. The thesis describes the outcrop as
nearly 150 m long and 15-20 m high. Its published GCP table spans approximately
`E 679736.636-679794.297`, `N 6444825.785-6444851.769`, and
`Z 37.774-40.260 m` in L-EST97/EH2000-scale coordinates. This is the requested
landmark, not a same-name proxy.

**Acquisition and geometry:** field acquisition was on 2024-04-26, after snowmelt
and before leaves obstructed the wall.

- DJI Mavic 3M, 20 MP RGB camera, supplied the aerial/top views.
- Pentax KP, 24.3 MP, 24 mm lens, supplied terrestrial/oblique views.
- Five markers and four 1 m rods were placed along the river bank.
- Marker coordinates were measured with a Topcon Hiper 5 GNSS. The receiver
  reported `1-3 cm` horizontal and vertical precision during measurement. This is
  a receiver estimate, not an independent checkpoint RMSE.
- Pix4Dmapper 4.8.4 produced a drone model, a terrestrial model, and a merged
  model. The work also produced a fourth model by reprocessing 2015 photographs.
- The merged 2024 product is a true-3D wall model. Point density, source-file size,
  final mesh spacing, independent checkpoints, and per-surface completeness are
  not reported in the public text.

**Comparison processing:** for the 2015/2024 comparison, trees, an old debris
mound, and non-overlapping areas were manually segmented away. A statistical
outlier filter used six neighbors and a `3 sigma` cutoff; clouds were fine-registered
with ICP at 75% overlap and compared with M3C2. The thesis explicitly treats the
comparison models as having about `5 cm` precision and warns that residual
vegetation produces false apparent deposition. That is not a complete ground or
vegetation semantic classification.

**Public files and size:** a direct DSpace REST audit found one file in the
`ORIGINAL` bundle: `Shlykova_Polina.pdf`, `1,822,127` bytes. No source photographs,
Pix4D project/report, dense cloud, mesh, GCP machine-readable file, or M3C2 output
is attached. The thesis says that the models *can* be uploaded for public study;
it does not identify an upload.

**Rights:** repository metadata says `CC BY-NC-ND 3.0 Estonia`; the thesis's signed
license page says `CC BY-NC-ND 4.0`. This inconsistency is immaterial to the scan
files because neither license grants rights in files the repository does not
publish and whose ownership is not specified. `ND` would also prohibit deriving a
production geometry asset from the thesis itself. Written data-owner permission or
a new explicit data license is required.

**Qualification:** **R, conditional S/H/C**. If released with raw data and adequate
rights, this is the strongest exact-site source for the wall, toe, cliff shoulder,
and adjacent bank. It is not qualified now. Heightfield use must retain only
single-valued, sufficiently supported top/shoulder/bank regions; faces, caves, and
undercuts remain true-3D. Suur Taevaskoda must remain outside morphology training
if it continues to be the fixed generalization review site.

### 1.2 Kohv 2015 public Sketchfab model

**Record:** Marko Kohv, [*Suur Taevaskoda - Devonian outcrop*](https://sketchfab.com/3d-models/suur-taevaskoda-devonian-outcrop-6866b96716ad42be8ce243e645b1f042),
published 2015-02-23, Sketchfab UID
`6866b96716ad42be8ce243e645b1f042`.

**Verified facts from the creator record and API:** Burtnieki Formation, Ahja river;
`185 m` length, `25.2 m` maximum height; `103` photographs; public viewer mesh of
`1,000,000` triangles and `504,356` vertices. It is true-3D, not a heightfield.
No point density, image GSD, control survey, coordinate reference system,
registration error, vegetation class, or source archive size is published.

Shlykova's thesis establishes an important limitation: the 2015 photographs lacked
the 2024 rods and markers, so her 2015 reconstruction was scaled/georeferenced from
carved letters and fracture corners visible in both epochs, then registered to the
2024 cloud. The Sketchfab upload itself must not be assumed to have 2024-quality
absolute registration.

**Access and rights:** the live Sketchfab v3 API reports
`isDownloadable=false`, `downloadCount=0`, an empty license object, and no archive.
Viewer access is not a geometry license. Scraping the viewer is not an acceptable
substitute.

**Qualification:** **V/I; R for the creator's original archive**. The viewer is
useful for confirming wall-scale forms and planning access requests. It cannot be
used as training geometry or shipped geometry without the creator's release.

## 2. Closely related sandstone scans

These records prove that the same University of Tartu creator has a broader
Devonian-outcrop collection. They are valuable leads for a no-fee collection-level
request, not only a request for one landmark.

| Model | Exact geology/site | Public representation | Access/rights | Qualification |
|---|---|---:|---|---|
| [Väike Taevaskoda](https://sketchfab.com/3d-models/none-dd5731d1f7564397bc2608fc9288aa15), UID `dd5731d1f7564397bc2608fc9288aa15` | Burtnieki Formation, Ahja river, `71.5 m` long, `11.5 m` high; 45 photos | 518,439 faces; 262,429 vertices | Non-downloadable; no license | **V/I, R**; exact substrate and neighboring site, potentially more valuable for generalization than training on Suur |
| [Härma müür](https://sketchfab.com/3d-models/none-81c1729244d9432592f510246495d874), UID `81c1729244d9432592f510246495d874` | Gauja Formation sandstone, Piusa river; `115 m` wide, `18.3 m` high; drone plus handheld images | 797,150 faces; 402,412 vertices | Non-downloadable; no license | **V/I, R**; similar southern Estonian sandstone, but not Burtnieki |
| [Härma lower outcrop](https://sketchfab.com/3d-models/none-271047d7e6f0454d8aa0f5617ad784a0), UID `271047d7e6f0454d8aa0f5617ad784a0` | Gauja Formation sandstone, Piusa river; about `13 m` high | 173,508 faces; 87,760 vertices | Non-downloadable; no license | **V/I, R** |
| [Veczemju Cliffs](https://sketchfab.com/3d-models/none-065fafac46914fe5b55d3beb6b0f117e), UID `065fafac46914fe5b55d3beb6b0f117e` | Northern Latvia near Salacgriva; **Burtnieki Formation** sandstone under marine/eolian sand; caves and alcoves; `127 m` long, `3.8 m` high | 635,644 faces; 322,089 vertices | Non-downloadable; no license | **V/I, R**; unusually strong neighboring-country exact-formation lead, but no open geometry |

None publishes a ground/vegetation class, independent error, raw point cloud, or
effective surface resolution. Uploaded face counts are decimation/export facts,
not accuracy evidence.

## 3. Exact-site open national ALS

### 3.1 Taevaskoda tile `444679`

The requested coordinate `E 679763.082, N 6444796.565` falls in official 1:2,000
LiDAR tile **`444679`**, extent
`[679000,680000) x [6444000,6445000)` in EPSG:3301.

The [official height-data download service](https://geoportaal.maaruum.ee/est/ruumiandmed/korgusandmed/laadi-korgusandmed-alla-p614.html)
currently exposes these raw epochs for that exact square:

- normal/spring mapping: `2011`, `2015`, `2019`, `2023`;
- low flight: `2016`;
- forestry/summer mapping: `2017`, `2021`, `2024`.

The current normal file has the canonical endpoint:

```text
https://geoportaal.maaamet.ee/index.php?lang_id=1&plugin_act=otsing&kaardiruut=444679&andmetyyp=lidar_laz_tava&dl=1&f=444679_2023_tava.laz&page_id=614
```

It was downloaded and header-audited rather than inferred from the 1 m DTM:

| Property | Verified value |
|---|---:|
| file | `444679_2023_tava.laz` |
| bytes | `68,066,160` |
| SHA-256 | `9c50c123f14841c717d0d123d2d08061b87a51baf6d806ef269fabf1036a9fb7` |
| format | LAS 1.4, compressed point format 8, 46-byte logical records |
| point count | `5,234,018` total points |
| tile-average total density | `5.234 points/m2` |
| density-equivalent mean spacing | about `0.437 m` |
| coordinate scale | `0.01 m` on X/Y/Z |
| bounds | X `679000.00-679999.99`; Y `6444000.00-6444999.99`; Z `37.25-168.81 m` |
| header production date | day 325 of 2023 |

Coordinate storage precision is not measurement accuracy. Total density is not
bare-ground density, and density is nonuniform near trees, water, and a steep wall.
The raw cloud nevertheless provides information discarded by the gridded DTM:
return order, time, scan geometry, RGB/NIR color, original 3D positions, classes,
overlap flags, and multiple epochs.

The [official ALS III documentation](https://geoportaal.maaamet.ee/est/ruumiandmed/korgusandmed/aerolaserskaneerimise-korguspunktid/als-iii-ring-2016-20172020-p625.html)
defines class 2 ground, class 5 first/intermediate vegetation returns, class 6
automatically classified buildings, classes 7/18 low/high noise, class 9 water,
class 17 second-level road structures, and keypoint/overlap bits. Water and bridge
classes use ETAK boundaries, and the agency warns that overlap points should usually
be excluded. These labels are useful evidence, not infallible truth.

**Rights:** the [Maa- ja Ruumiamet open-data license dated 2025-01-01](https://geoportaal.maaruum.ee/avaandmete-litsents)
explicitly permits free commercial or noncommercial use, derivatives, combination
with products/services, and redistribution. Publication/redistribution must identify
the agency, dataset, and age/extract date and include the license text or link.

**Qualification:** **O/S**, not **H**. The epochs should be used immediately to
audit the false Ahja river ramp, vegetation leakage, water returns, flight-strip
agreement, and cliff/top/shore observations before repairing the 1 m surface. A
5.234 points/m2 total cloud cannot supervise 256 output cells/m2 at 6.25 cm; it has
about one total return for every 49 output cells before ground filtering. Top-down
airborne geometry also leaves cliff-face shadows and cannot reliably resolve
undercuts.

### 3.2 Country-wide raw ALS

This is a real, country-wide, legally usable resource, not a proposed acquisition.
The portal provides 1 km2 LAZ tiles, historical epochs since 2008, unlimited batch
download instructions, and approximately one-quarter-country refresh per year.

Published campaign densities must be kept campaign-specific:

- Riegl VQ-1560i-era nominal totals were 18 points/m2 over settlements,
  3.5 points/m2 for spring image-priority mapping, 2.1 points/m2 for national
  mapping, and 0.8 points/m2 for summer forestry mapping.
- The agency's 2025 Riegl VQ-1460 description gives up to 80 points/m2 in
  settlements, 8 points/m2 in spring, and 2 points/m2 in summer.
- The agency also reports that 2025 GNSS interference prevented ALS production
  over the eastern half of the country. The exact Taevaskoda index consequently has
  no 2025 file; its latest normal and forestry files are 2023 and 2024 respectively.

Even the new 80 points/m2 settlement product averages only 0.3125 return per
6.25 cm output cell before ground filtering and is geographically concentrated in
settlements. It improves source repair, feature detection, and conditioning; it is
not national decimeter morphology truth.

## 4. Other Estonian high-density campaigns

### 4.1 Selisoo bog multi-temporal SfM

**Record:** Marko Kohv, Edgar Sepp, and Lii Vammus, *Assessing multitemporal
water-level changes with UAV-based photogrammetry*, *The Photogrammetric Record*
32 (2017), [DOI 10.1111/phor.12214](https://doi.org/10.1111/phor.12214).

**Site/extent:** two raised-bog test areas in Selisoo, northeastern Estonia. The
common analyzed extents were `14.8 ha` west and `11.6 ha` east. Each was surveyed
in September 2015, May 2016, and September 2016.

**Data:** Sony A5100, 24 MP, 16 mm lens, nadir flights at 100 m in 2015 and 80 m in
2016; 198-221 images per flight; 5-12 RTK GCPs, average 8. Six dense clouds were
produced, normally `200-250 million` points each or `700-800 points/m2`. The final
orthophotos were 2.5 cm GSD and the released analysis surfaces were 10 cm DEMs.
The dense surface was automatically split into ground and vegetation in PhotoScan
using maximum angle 40 degrees, maximum distance 0.5 m, and maximum cell size
0.5 m, then only the ground class was meshed as a heightfield.

**Error and semantics:** the RTK system's ideal accuracy was stated as 2 cm
horizontal/5 cm vertical, while manual pool-bank measurements were estimated at
1-2 cm. The terrain is a living Sphagnum surface with ridge/pool organization, not
bare mineral earth. The authors explicitly report tree-stump/noise artifacts and
poor water surfaces. Therefore `ground` here means the reconstructed bog surface
after an algorithmic classification, not vegetation-free DTM truth.

**Access/rights:** no raw photographs, point clouds, DEM package, or data license
was found in the paper/repository records audited for this report. The clouds are
verified to have existed and the author is the same Marko Kohv who owns the
Taevaskoda collection. They are a high-priority permission-request target.

**Qualification:** **R; conditional H for raised-bog surface organization**. If the
six raw campaigns, GCPs, and rights are released, they could supply native Estonian
ridge/hollow/pool examples and sensor-repeat evidence. They cannot be accepted from
the 10 cm export alone, cannot describe forested peat ground, and require water,
trees, and weakly supported cells to be masked.

### 4.2 Pakri coast digital twin

**Record:** Tiina Harak, *Pakri poolsaare rannikust digitaalkaksiku loomine ja
rannikumuutuste analüüs*, University of Tartu master's thesis, 2024,
[handle 10062/100042](https://hdl.handle.net/10062/100042).

The study verifies multi-temporal drone photogrammetric models of the Pakri cliff
and coast from 2015-2024, processed with Agisoft Metashape and CloudCompare. The
repository publishes the thesis, not a licensed raw cloud/model corpus. Pakri's
Cambrian/Ordovician sandstone-argillite-limestone cliff and marine forcing do not
match Taevaskoda's Burtnieki fluvial sandstone.

**Qualification:** **R/C/I**, not **H** for Taevaskoda morphology. If released, the
raw epochs could inform general cliff change-processing and occlusion/error audits.

### 4.3 EMU UAV accuracy campaign

**Record:** Karmo Siim and Oscar Rahu, *Mehitamata õhusõidukiga pildistatud
aerofotodest valmistatud ortofotomosaiikide ja 3D-punktipilvede täpsust mõjutavad
faktorid*, Estonian University of Life Sciences, 2019,
[handle 10492/5065](http://hdl.handle.net/10492/5065).

Trimble ZX5 flights over the EMU Metsamaja area in Tartu tested height, speed, and
image overlap. Reported best-control RMSE values are roughly 1-2.1 cm by axis for
particular flight settings. The repository attaches the thesis and review, not the
underlying point clouds or a data license. The site is a campus experiment rather
than a representative terrain-regime corpus.

**Qualification:** **R/I**. It is evidence that sub-decimeter Estonian UAV products
exist in university archives and a possible camera/error-method source, not a
morphology target.

## 5. Immediately downloadable Estonian meshes

The [Marko Kohv Sketchfab collection](https://sketchfab.com/markokohv/models)
contains four models marked downloadable under `CC Attribution`. Three are Estonian
terrain/outcrop-scale objects:

| Model | Geometry and advertised archive | Limitation | Qualification |
|---|---|---|---|
| [Panga Pank](https://sketchfab.com/3d-models/none-3c4676182dc64a019b3e788e3502e5f1), UID `3c4676182dc64a019b3e788e3502e5f1` | Saaremaa Silurian limestone/dolostone cliff; 145 drone photos; `300 m` long, `16.5 m` high; 1M-face/506,032-vertex viewer mesh; API advertises 68.5 MB GLB and 60.8 MB source archive | No control/error, point semantics, raw photos, or bare-ground classification; wrong lithology | **O/C/I** |
| [NE Estonia oil-shale quarry outcrop](https://sketchfab.com/3d-models/none-4eccf1de60c343f1be24a1ca3e9a809d), UID `4eccf1de60c343f1be24a1ca3e9a809d` | Ordovician-Devonian contact; about `17.5 m` high; DJI Phantom Vision+; 500,000-face viewer mesh; API advertises 31.4 MB GLB and 28.5 MB source | Artificial quarry exposure, wrong lithology; no error/registration | **O/C/I** |
| [Kukruse waste-rock mound](https://sketchfab.com/3d-models/none-64ad301de3514aa5991d9c9d83523ba9), UID `64ad301de3514aa5991d9c9d83523ba9` | NE Estonia, `40 m` relative height; drone at 50-70 m; 1M-face viewer mesh; API advertises 70.1 MB GLB and 72.8 MB source | Anthropogenic, internally burning/subsiding mound; not natural microtopography | **O/I** |

Before use, the download page must be archived with the exact license URI/version,
because the API label alone says only `CC Attribution`. These decimated meshes can
test ingestion, 3D normal/curvature extraction, and cliff masks. They must not set
Estonian soil/sandstone morphology distributions.

## 6. Catalog and portal negatives

Exact-name and geographic searches were made across University of Tartu DSpace,
Estonian University of Life Sciences DSpace, DataCite-indexed records, Zenodo,
PANGAEA, OpenTopography, Figshare, the Maa- ja Ruumiamet portals, EGT pages, and
general scholarly/project search. As of the audit date:

- no DOI/data record exposes the 2015 or 2024 Suur Taevaskoda raw images, point
  clouds, Pix4D/PhotoScan project, or mesh under a usable data license;
- no OpenTopography/PANGAEA record was found for a Taevaskoda or Ahja dense local
  survey;
- no open Estonia-wide TLS/UAS-LiDAR terrain corpus at 6.25 cm support was found;
- many theses prove that local dense surveys exist, but their repositories usually
  attach only the document and review;
- Google Earth/photorealistic viewers and Sketchfab viewer access are not lawful
  data downloads and were not counted as open sources.

This is a bounded catalog finding, not proof that no laboratory, contractor, or
park authority holds additional unpublished files.

## 7. Zero-budget action order

### P0: request the Kohv/Shlykova collection once

Make one narrow University of Tartu request covering Suur 2015, Suur 2024, Väike
Taevaskoda, Härma, and the Selisoo campaigns. Request:

- original images and camera metadata;
- Pix4D/PhotoScan project and processing-quality report;
- full dense cloud before viewer decimation, mesh, and textures;
- GCP/checkpoint table, coordinate reference system, camera calibration, and
  registration/M3C2 reports;
- masks or retained classifications for vegetation, water, low support, and
  manually removed material;
- written ownership confirmation and a license explicitly allowing commercial
  derivatives and ML training/evaluation, preferably CC BY 4.0;
- permission to redistribute only derived statistics/geometry if the raw source
  cannot be republished.

No payment, new fieldwork, or vague collaboration should be proposed. A refusal or
non-response leaves these sources at **V/R**, not **O**.

### P0: exploit the already-open exact-site ALS

Download and inventory all eight `444679` epochs. Use normal/spring and forestry
epochs jointly to distinguish repeatable ground from vegetation/water/strip
artifacts and to repair the existing DTM source. Preserve raw point support,
classification, time, overlap, and campaign metadata. This is immediately lawful
and directly addresses the known Ahja artifact.

### P1: qualify, do not assume

If any university archive is released, run the spec's target-data qualification
before viewing it as morphology truth: independent checkpoints, surface-support
maps, vegetation/water masks, multi-view completeness, registration uncertainty,
and heightfield-validity masks. Do not infer accuracy from a dense cloud or a
million-triangle viewer model.

### P2: use open mismatched cliffs only as tooling evidence

The Panga and quarry downloads can validate true-3D import, decimation diagnostics,
surface descriptors, and mask extraction at no acquisition cost. They cannot justify
Burtnieki sandstone synthesis or reopen the rejected non-heightfield cliff project.

## Final qualification matrix

| Source | Open now | Structural repair | 6.25 cm heightfield morphology | Future cliff research | Visual reference |
|---|---:|---:|---:|---:|---:|
| Suur Taevaskoda 2024 raw survey | No; permission lead | Strong if released | Conditional for top/shoulder/bank; keep out of training while it is the blind landmark | Strong if released | Strong |
| Suur Taevaskoda 2015 Sketchfab | Viewer only | No | No | Viewer/process reference only | Strong |
| Väike Taevaskoda/Härma source archives | No; permission leads | Conditional | Conditional after error/support audit | Strong exact/near material leads | Strong |
| Exact tile `444679` 2023 plus seven epochs | **Yes** | **Strong and immediate** | No; far too sparse | Coarse/top-down context only | Moderate |
| Country-wide Maa- ja Ruumiamet ALS | **Yes** | **Strong national input** | No national target-scale supervision | Context only | Moderate |
| Selisoo six dense SfM campaigns | No; permission lead | Conditional | Strong raised-bog candidate if raw support and living-surface semantics pass | No | Strong |
| Pakri 2015-2024 campaigns | No; permission lead | Conditional locally | Wrong regime | Conditional method evidence | Strong |
| CC Attribution Panga/quarry meshes | **Yes** | No | Wrong regime and no error truth | Tooling/mask evidence only | Strong |

The defensible no-budget course is therefore: recover existing author-held data,
fully use public raw ALS for structural correction, and abstain from claiming
target-scale morphology where no qualified source becomes available.
