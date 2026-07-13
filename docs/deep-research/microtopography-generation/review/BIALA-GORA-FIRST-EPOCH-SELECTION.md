# Biała Góra First-Epoch Selection

**Status:** frozen metadata selection; binary not downloaded
**Selection date:** 2026-07-13
**Role:** `calibration_only`
**Bulk acquisition authorized:** no

## Selection

Select exactly this official RepOD file for the first bounded audit:

| Field | Frozen value |
|---|---|
| Dataset | *UAV LiDAR point clouds from the DJI Zenmuse L1 sensor on the cliff coast.* |
| Dataset DOI | [`10.18150/BHH1RC`](https://doi.org/10.18150/BHH1RC) |
| Repository/version | RepOD, dataset version `1.0`, published 2025-01-24 |
| Creator | Jakub Śledziowski, University of Szczecin, ORCID `0000-0001-9359-7907` |
| Epoch/file | `2022-02-27.zip` |
| RepOD data-file ID | `54447` |
| Canonical file record | [`fileId=54447`, dataset version 1.0](https://repod.icm.edu.pl/file.xhtml?fileId=54447&version=1.0) |
| Official metadata API | [RepOD dataset API](https://repod.icm.edu.pl/api/datasets/:persistentId/?persistentId=doi%3A10.18150%2FBHH1RC) |
| Media/contents | `application/zip`; repository description `LAS (0,6 GB)` |
| Exact compressed size | `351,166,481` bytes (`334.90 MiB`) |
| Published checksum | MD5 `c7a0df4c052ebfabcc40729c319f3614` |
| Access restriction | `restricted=false` |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode) |

RepOD does not publish a SHA-256 for this file. A future explicitly authorized
download must first verify the published MD5, then compute and record a local
SHA-256. This selection record does not authorize that download.

## Selection Rule

`2022-02-27.zip` is selected because it is the smallest complete dated LiDAR epoch
in the official version-1.0 manifest and is also the earliest campaign. It therefore
minimizes initial transfer/storage cost while testing the oldest temporal endpoint.
The choice was made from metadata before inspecting geometry; visual attractiveness
or apparent roughness did not influence it.

The remaining nine epoch archives, totaling most of the 27.402 GB release, remain
unselected. Passing this audit would justify only a separately frozen next-file
decision, not a bulk download.

## Allowed Audit

If a later step authorizes the single file download, its role remains
`calibration_only`. The audit may measure:

- archive integrity, LAS schema, CRS/header metadata, attributes and return fields;
- total and spatially varying point support, view/flight geometry if reconstructable,
  and candidate heightfield-valid coverage;
- feasibility of semantic classification, support masks, cliff/overhang masks and
  cross-referencing to the published campaign description;
- storage, decompression and conversion working-set cost.

It may not:

- treat raw, lowest, vendor-classified, gridded or interpolated points as target truth;
- train or select a microtopography synthesis owner;
- authorize `B1` or `B2` target status, Estonia transfer, pilot packing or national use;
- infer density, accuracy, surface semantics or effective resolution from file size;
- trigger acquisition of any other Biała Góra archive.

The repository supplies no independent positional-error distribution, checkpoint
record, authoritative ground semantics, per-cell support mask, or target-band
transfer evidence. Those absences keep the file at `calibration_only` regardless of
how detailed it looks.

## License And Attribution

RepOD applies file-level **Creative Commons Attribution 4.0** to data-file `54447`.
Any retained copy or derived calibration artifact must preserve this attribution:

> Śledziowski, Jakub (2025), “UAV LiDAR point clouds from the DJI Zenmuse L1 sensor
> on the cliff coast.”, RepOD, V1, DOI 10.18150/BHH1RC, file
> `2022-02-27.zip`, CC BY 4.0.

Record the access date, link the license, retain the original creator and DOI, and
identify modifications when publishing a derived artifact. This license record does
not change the scientific role from calibration-only.

## Pre-Download Gate

Before any binary transfer, a manifest must bind the values above and additionally
record `max_network_bytes`, retry allowance, `max_cache_bytes`, actual free space,
minimum reserved free space, download destination, exact access UTC, and the code
used to verify MD5 and compute SHA-256. Any official metadata drift, byte-size
change, checksum change, license change, or file-ID change stops the transfer and
requires a new selection revision.

**Current decision:** metadata frozen; zero archive bytes acquired; no further
Biała Góra action authorized.
