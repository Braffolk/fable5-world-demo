# Taevaskoda ALS Selection For Stage 1 Structural Repair

**Status:** executable acquisition and evidence-role decision
**Date:** 2026-07-13
**Scope:** official public airborne-laser-scanning files for tile `444679` only
**Stage:** Stage 1 Track B structural repair, not morphology synthesis or training

## 1. Decision

Acquire and preserve all eight official epochs for tile `444679`. Use
`444679_2023_tava.laz` as the **primary current geometric observation**, but only
cell-by-cell where qualified class-2, non-overlap, non-water support passes the
checks in this document. It is selected because it is the newest normal-mapping
epoch, its exact bytes and header have already been audited, and it preserves raw
returns and acquisition attributes that the 1 m DTM discarded.

Do not treat an entire epoch as truth. Use the remaining seven files as independent
or complementary observations:

- `2011`, `2015`, and `2019` normal-mapping files corroborate stable dry ground and
  reveal temporal change, classification mistakes, and scan-strip disagreement;
- `2016` low-flight data is a potentially stronger local observation, but it cannot
  become authority until its actual density, geometry, classes, and acquisition
  date are audited;
- `2017`, `2021`, and `2024` forestry/summer files provide complementary leaf-on,
  canopy, occlusion, water, and change evidence; their newer year does not make
  their ground estimate more authoritative than the 2023 normal file.

When the 2023 observation fails locally, select a qualified older observation with
an explicit observation date and temporal-uncertainty flag. Never fill a failed
area by averaging epochs. Near the river and actively eroding cliff, disagreement
may be real change and must not be forced into a static consensus.

These files are **structural-repair evidence only**. Their total and bare-ground
support are insufficient for 6.25 cm morphology, and no file in this set may be a
fine-detail training target.

## 2. Spatial Binding

The Stage 1 review coordinate is `E 679763.082, N 6444796.565` in EPSG:3301. It
falls inside official 1:2,000 LiDAR tile `444679`:

```text
CRS:       EPSG:3301
X extent:  [679000, 680000)
Y extent:  [6444000, 6445000)
Area:      1,000 m x 1,000 m
```

The canonical discovery service is:

```text
https://geoportaal.maaruum.ee/est/ruumiandmed/korgusandmed/laadi-korgusandmed-alla-p614.html
```

The exact official search that exposed the file set is:

```text
https://geoportaal.maaamet.ee/index.php?lang_id=1&plugin_act=otsing&page_id=614&kaardiruut=444679&andmetyyp=lidar_laz_tava
```

The host-name difference is an agency/portal routing fact. The acquisition ledger
must record both the requested URL and final response URL after redirects.

## 3. Exact File Set

The year in a filename is the published campaign year. It is not an exact flight
date. Exact acquisition time remains unknown until adjusted GPS time and campaign
metadata are decoded with an era-correct rule. The 2023 LAS header production day
is a file-production fact, not the flight date.

| Role | Published file | Canonical type | Campaign year | Exact acquisition date | Bytes | SHA-256 |
|---|---|---|---:|---|---:|---|
| corroboration | `444679_2011_tava.laz` | `lidar_laz_tava` | 2011 | unknown | unknown | unknown |
| corroboration | `444679_2015_tava.laz` | `lidar_laz_tava` | 2015 | unknown | unknown | unknown |
| conditional corroboration | `444679_2016_madal.laz` | `lidar_laz_madal` | 2016 | unknown | unknown | unknown |
| complementary | `444679_2017_mets.laz` | `lidar_laz_mets` | 2017 | unknown | unknown | unknown |
| corroboration | `444679_2019_tava.laz` | `lidar_laz_tava` | 2019 | unknown | unknown | unknown |
| complementary | `444679_2021_mets.laz` | `lidar_laz_mets` | 2021 | unknown | unknown | unknown |
| **primary candidate** | `444679_2023_tava.laz` | `lidar_laz_tava` | 2023 | unknown | **68,066,160** | **`9c50c123f14841c717d0d123d2d08061b87a51baf6d806ef269fabf1036a9fb7`** |
| complementary/change | `444679_2024_mets.laz` | `lidar_laz_mets` | 2024 | unknown | unknown | unknown |

The canonical URL for each row is exactly:

```text
https://geoportaal.maaamet.ee/index.php?lang_id=1&plugin_act=otsing&kaardiruut=444679&andmetyyp={canonical_type}&dl=1&f={published_file}&page_id=614
```

Resolved URLs are therefore:

```text
https://geoportaal.maaamet.ee/index.php?lang_id=1&plugin_act=otsing&kaardiruut=444679&andmetyyp=lidar_laz_tava&dl=1&f=444679_2011_tava.laz&page_id=614
https://geoportaal.maaamet.ee/index.php?lang_id=1&plugin_act=otsing&kaardiruut=444679&andmetyyp=lidar_laz_tava&dl=1&f=444679_2015_tava.laz&page_id=614
https://geoportaal.maaamet.ee/index.php?lang_id=1&plugin_act=otsing&kaardiruut=444679&andmetyyp=lidar_laz_madal&dl=1&f=444679_2016_madal.laz&page_id=614
https://geoportaal.maaamet.ee/index.php?lang_id=1&plugin_act=otsing&kaardiruut=444679&andmetyyp=lidar_laz_mets&dl=1&f=444679_2017_mets.laz&page_id=614
https://geoportaal.maaamet.ee/index.php?lang_id=1&plugin_act=otsing&kaardiruut=444679&andmetyyp=lidar_laz_tava&dl=1&f=444679_2019_tava.laz&page_id=614
https://geoportaal.maaamet.ee/index.php?lang_id=1&plugin_act=otsing&kaardiruut=444679&andmetyyp=lidar_laz_mets&dl=1&f=444679_2021_mets.laz&page_id=614
https://geoportaal.maaamet.ee/index.php?lang_id=1&plugin_act=otsing&kaardiruut=444679&andmetyyp=lidar_laz_tava&dl=1&f=444679_2023_tava.laz&page_id=614
https://geoportaal.maaamet.ee/index.php?lang_id=1&plugin_act=otsing&kaardiruut=444679&andmetyyp=lidar_laz_mets&dl=1&f=444679_2024_mets.laz&page_id=614
```

### 3.1 Verified 2023 facts

The prior byte/header audit established:

| Property | Verified value |
|---|---:|
| format | LAS 1.4, LAZ-compressed point format 8 |
| logical point-record length | 46 bytes |
| point count | 5,234,018 |
| tile-average total density | 5.234 points/m2 |
| density-equivalent spacing | approximately 0.437 m |
| coordinate scale | 0.01 m on X, Y, and Z |
| X bounds | 679000.00 to 679999.99 |
| Y bounds | 6444000.00 to 6444999.99 |
| Z bounds | 37.25 to 168.81 m |
| header production date | day 325 of 2023 |
| producer strings | `LAStools ...`; `lascolor (230123) commercial` |

Coordinate scale is storage precision, not measurement accuracy. Total density is
not ground density. RGB/NIR values were added from orthophoto and are not an
independent geometric observation.

## 4. Rights And Attribution

The controlling public terms are the Maa- ja Ruumiamet open-data license dated
2025-01-01:

```text
https://geoportaal.maaruum.ee/avaandmete-litsents
```

The reviewed terms allow free commercial and noncommercial use, modification,
combination with products or services, derivative production, and redistribution.
Publication or redistribution must identify:

- Maa- ja Ruumiamet as provider;
- the dataset, here raw airborne-laser-scanning height points for tile `444679`;
- the data age or extraction date;
- the license text or link.

Every derivative manifest and published asset recipe must carry those fields. The
license page itself must be snapshotted and hashed at acquisition time. Do not
infer a standard SPDX identifier unless a legal mapping is separately recorded;
the source ledger should preserve the license's official name and URL verbatim.

## 5. Evidence Roles And Failure Risks

### 5.1 Point acceptance

For 2017-and-later files, begin with the official ALS III semantics:

- class 2 is ground candidate evidence;
- class 5 is first/intermediate vegetation and is never ground merely because it is
  low;
- class 6 is automatically classified building evidence;
- classes 7 and 18 are low/high noise;
- class 9 is water;
- class 17 is a second-level road structure;
- overlap and key-point bits must be retained explicitly.

Only non-overlap class-2 points are default dry-ground candidates. Retain overlap
points for flight-strip agreement diagnostics, but do not silently mix them into
the accepted surface. Any other class requires an explicit reviewed conversion
rule and cannot become ground through a generic "last return" heuristic.

The post-2017 class map must not be applied to 2011, 2015, or 2016. Their LAS
versions, point formats, class vocabulary, and GPS-time convention must be derived
from the actual headers and matching official campaign documentation. Unknown
semantics fail closed.

### 5.2 Water and shoreline

Class 9 does not represent the river bed. It may become visible-water-surface
evidence only if its reach is locally coherent, its acquisition state is compatible
with the dated map/imagery evidence, and strip/return geometry does not indicate a
spurious interpolation. Otherwise retain it only as a water mask observation.

Ground-class points near the Ahja River are not automatically terrain. Water and
bridge labels use ETAK boundaries, so classification and ETAK geometry share a
semantic dependency and are not independent votes. A class-2 ramp across water,
the two reported river bumps, or a TIN bridge must be rejected from bed authority.
Stage 1 keeps `waterY`, terrain bed, shoreline, bank side, and water coverage as
separate fields under the canonical spec.

ETAK remains the Stage 1 continuous shoreline authority. ALS may diagnose support,
bank side, local vertical structure, and likely source errors; it cannot move the
shoreline outside the spec's dated optical/ETAK challenger gate.

### 5.3 Vegetation and objects

Leaf-on forestry acquisitions can improve evidence about canopy and occlusion
while reducing usable bare-ground support. They are complementary, not a ground
truth majority vote. Low vegetation, roots, fallen trees, and dense bank growth can
leak into class 2; steep sandstone faces can be omitted or misclassified because
airborne line of sight and ground classifiers are biased toward horizontal terrain.

The 1 m DTM may have been derived from some of these same measurements. Agreement
between the DTM and a contributing ALS epoch is provenance-correlated, not
independent validation. Treat earlier flight epochs as independent measurements
only after checking that they are distinct acquisitions; shared agency classifiers,
ETAK masks, and processing software remain shared systematic risks.

### 5.4 Temporal change

The cliff, bank, channel, vegetation, and human disturbance can change between
2011 and 2024. Establish a vertical/strip bias on stable, dry, well-supported ground
away from the river, cliff edge, structures, and recent disturbance. After that
calibration:

- repeatable geometry supports a stable-ground correction;
- an isolated epoch anomaly supports rejection or uncertainty, not averaging;
- coherent spatial change across later observations is recorded as real-change
  evidence;
- ambiguous disagreement becomes abstention.

Do not use the Taevaskoda epochs to tune morphology. Exact-site evidence remains
outside training, hyperparameter selection, and candidate selection.

## 6. Qualification And Selection Algorithm

Apply this deterministic order independently to each repair-support cell and
protected feature:

1. Decode the epoch with its validated era-specific LAS and time semantics.
2. Preserve every source attribute before classification or cropping.
3. Reject invalid/noise points and separate overlap points from candidate support.
4. Build support diagnostics by class, return order, scanner channel, flight strip,
   incidence/scan angle, GPS time, and local slope.
5. Calibrate per-epoch and per-strip vertical offsets on stable dry reference areas;
   record the correction and uncertainty without altering raw coordinates.
6. Test 2023 class-2 non-overlap points for local support, residual distribution,
   cliff/water/object risk, and agreement with independent epochs.
7. Accept 2023 only where those tests pass. Record it as the selected observation,
   not as repaired truth.
8. If 2023 fails, evaluate the other normal epochs and the audited 2016 low-flight
   data, ranking them by measured local support and error rather than filename or
   age. Select one qualified observation or an evidence-backed structural fit; do
   not take an untyped mean.
9. Use forestry epochs to challenge vegetation leakage, expose occlusion, and flag
   change. Promote their ground points only when the same full qualification passes.
10. If no epoch supplies defensible support, set explicit unknown/abstention and let
    typed ETAK/optical structural authority handle only what its own evidence permits.

The accepted repair must preserve counterevidence for real banks, shoulders,
escarpments, cliff tops, channels, and small stable landforms. "Smooth" is never a
valid correction class by itself.

## 7. Immutable Provenance

Each downloaded epoch requires an immutable acquisition record with at least:

- tile ID, published filename, canonical data type, campaign year, and intended
  evidence role;
- requested URL, redirect chain, final URL, UTC fetch start/end, HTTP status,
  `Content-Disposition`, `Content-Length`, `ETag`, `Last-Modified`, and byte-range
  behavior where present;
- exact byte count and SHA-256 of the original LAZ;
- source-index URL, index query time, and a hash/snapshot of the returned file list;
- license URL, license effective date, snapshot path/hash, extraction date, and
  exact attribution string;
- LAS version, compressed/raw point format, logical record length, global encoding,
  creation day/year, system/software strings, scales, offsets, bounds, point count,
  return histogram, VLR/EVLR inventory and hashes, CRS GeoKeys/WKT, and all extra
  dimensions;
- class and classification-flag counts; scanner-channel, point-source-ID, overlap,
  scan-angle, return, GPS-time, RGB/NIR, and user-data ranges and missingness;
- GPS-time convention, raw time range, conversion method/version, leap-second table
  identity, derived acquisition interval, and confidence;
- AOI/collar geometry in EPSG:3301, its hash, every coordinate-transform pipeline,
  and the PROJ database/version used;
- decoder/converter name, version, source revision, environment lock, command or
  pipeline JSON, deterministic ordering rule, and output hashes/counts;
- per-epoch qualification state, rejection reasons, selected cells/features, and
  hashes of all masks and evidence artifacts.

Source facts belong in immutable `RawObservation`/`ObservationEvidence`. Repair may
consume them but cannot rewrite classifications, coordinates, times, or support to
hide an accepted correction.

## 8. Retention And Conversion Order

Use this order so convenience derivatives never replace evidence:

1. Download to `.part`; validate expected filename/content type and reject HTML.
2. Hash and byte-count the complete response before any conversion.
3. Atomically retain the original LAZ under a path keyed by its SHA-256. Make it
   read-only to the cook workflow.
4. Emit a small header/VLR/EVLR and dimension-inventory JSON tied to the original
   digest.
5. Stream-decode all attributes and emit deterministic per-class/support statistics.
6. Derive an AOI plus declared collar subset only after the complete original is
   retained. Preserve source point order or store an explicit original-point index.
7. If COPC/LAZ reordering is used for spatial access, treat it as a derivative and
   preserve all dimensions, point counts, converter identity, and source mapping.
8. Produce epoch-specific semantic point sets and support/error masks. Never destroy
   excluded points; retain their reason codes.
9. Register epochs without mutating raw coordinates; store transforms and residuals
   as separate artifacts.
10. Rasterize only the typed evidence fields required by `ObservationEvidence`.
    Retain counts, support footprint, interpolation state, uncertainty, source epoch,
    and selected-point references per field.

Retain all eight original 1 km files for this pilot. Their sizes are bounded enough
for selective exact-site acquisition; no nationwide multi-epoch mirror is implied.

## 9. Machine Ledger Contract

The existing `sources.schema.json` is source-ledger version 2.0.0. It requires every
listed artifact to have a non-null 64-hex SHA-256 linked to
`metadata/SHA256SUMS`. Therefore an indexed-but-unacquired file must **not** be
inserted into `artifacts` with a fake or null hash.

Add a future site-specific `official_dataset` record with ID such as
`maaruumiamet-als-444679-taevaskoda` only when the ledger is deliberately updated.
Relate it to:

- `maaruumiamet-als-iii-2016-2020` for post-2017 acquisition/class metadata;
- the official height-data download service record, creating it if still absent;
- an official Maa- ja Ruumiamet open-data-license record, creating it if still
  absent.

Until each file is retained in the repository evidence library and its checksum is
entered in `metadata/SHA256SUMS`, use `data_status=open_not_downloaded` and an empty
artifact list. The already audited 2023 digest is a known external acquisition fact;
it does not become a ledger artifact until a canonical local path exists and its
bytes are reverified there.

Maintain a companion epoch manifest, for example
`review/contracts/taevaskoda-als-444679.epochs.json`, with one row per file and these
machine fields:

```text
schema_version
site_id, tile_id, crs, tile_bounds
epoch_id, campaign_kind, campaign_year, acquisition_start, acquisition_end
published_filename, canonical_type, request_url, redirect_chain, final_url
index_checked_at, download_state, fetched_at, http_metadata
byte_length, sha256, canonical_local_path
license_record_id, license_snapshot_sha256, attribution
las_version, point_format, point_record_length, point_count
scales, offsets, bounds, crs_wkt_sha256, vlr_inventory_sha256
dimension_inventory, return_histogram, class_histogram, flag_histogram
gps_time_convention, gps_time_range, time_conversion_recipe_sha256
decoder_recipe_sha256, conversion_recipe_sha256, derived_artifact_ids
role, qualification_state, qualification_recipe_sha256
accepted_support_artifact_id, rejection_reason_codes, notes
```

Nullable unknowns are permitted in this companion manifest only when accompanied by
an explicit state. Use the state progression:

```text
indexed -> downloaded -> byte_validated -> metadata_validated -> qualified
                                                       \-> rejected
```

No `qualified` row may contain unknown acquisition semantics, missing source hash,
unknown CRS, unmapped classification semantics, or unresolved rights.

## 10. Stage 1 Deliverable Gate

The eight-epoch ALS acquisition is complete only when:

- all eight original LAZ files have immutable hashes, sizes, exact URLs, and rights;
- era-correct class/time semantics and complete dimension inventories are recorded;
- 2023 primary support and every fallback are selected by the algorithm above;
- water, vegetation, overlap, cliff, object, scan-strip, and temporal-change risks
  appear in `ObservationEvidence` with explicit unknown/abstention states;
- the repaired Ahja water surface/bed, shoreline, bank/shoulder, escarpment, and two
  diagnosed river bumps have typed evidence provenance;
- no ALS-derived 6.25 cm morphology is emitted or claimed;
- the structural repair is evaluated separately from later morphology and leaves
  real protected features intact.

This selection closes the exact-site evidence choice. It does not authorize a
generic smoother, an exact DTM-preservation objective, or a Taevaskoda-specific
morphology rule.
