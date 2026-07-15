"""Frozen ZIP offsets and transfer ranges for selected Hinsberger members."""
from __future__ import annotations

from dataclasses import dataclass

from ..contract import (
    ARCHIVE_BYTES,
    ARCHIVE_MD5,
    ARCHIVE_NAME,
    DATASET_DOI,
    LICENSE_SPDX,
    LICENSE_URL,
    PUBLISHER_FILE_ID,
    member_by_path,
)

DOWNLOAD_URL = "https://ndownloader.figshare.com/files/43218330"

API_RESPONSE_SHA256 = "2129853ef317ddb82d11c12c7704bce3dbd01b3f0ef563d829b23a07872e800b"
CENTRAL_INVENTORY_SHA256 = "329433922e018f64b5d320355fff5be97cff52f1152243c0f89241fdd7c1d9e8"
PARTIAL_TRANSFER_SHA256 = "b366dc8da1afbf766b55cb37e85fa56ab9ed41032657182ccf9c586a6fee2515"
CENTRAL_MEMBER_COUNT = 28


@dataclass(frozen=True)
class SelectedMember:
    path: str
    local_header_offset: int
    local_header_bytes: int
    data_start: int
    compressed_bytes: int
    uncompressed_bytes: int
    crc32: int

    @property
    def data_end(self) -> int:
        return self.data_start + self.compressed_bytes - 1


def _member(
    path: str,
    local_header_offset: int,
    local_header_bytes: int,
    data_start: int,
) -> SelectedMember:
    common = member_by_path()[path]
    return SelectedMember(
        path,
        local_header_offset,
        local_header_bytes,
        data_start,
        common.compressed_bytes,
        common.uncompressed_bytes,
        common.crc32,
    )


SELECTED_MEMBERS = (
    _member(
        "Hinsberger_aerial_survey_data/Field4_Field5/Field4_5_dem.tif",
        13_081_499_621,
        90,
        13_081_499_711,
    ),
    _member(
        "Hinsberger_aerial_survey_data/Field4_Field5/Field4_5_orthomosaic.tif",
        13_222_622_506,
        98,
        13_222_622_604,
    ),
    _member(
        "Hinsberger_aerial_survey_data/Field7/Field7_dem.tif",
        15_692_716_386,
        81,
        15_692_716_467,
    ),
    _member(
        "Hinsberger_aerial_survey_data/Field7/Field7_orthomosaic.tif",
        15_881_205_811,
        89,
        15_881_205_900,
    ),
    _member(
        "Hinsberger_aerial_survey_data/Field8_Field9/Field8_9_dem.tif",
        17_847_729_271,
        90,
        17_847_729_361,
    ),
    _member(
        "Hinsberger_aerial_survey_data/Field8_Field9/Field8_9_orthomosaic.tif",
        18_160_555_812,
        98,
        18_160_555_910,
    ),
)

# These are the only compressed-data gaps absent from the frozen partial transfer.
MISSING_RANGES = (
    (13_530_933_650, 13_957_664_268),
    (14_057_860_621, 14_472_610_439),
    (15_915_064_188, 16_368_002_038),
    (16_469_599_223, 16_893_045_873),
    (17_009_196_146, 17_418_089_708),
    (17_511_322_861, 17_847_729_196),
    (17_847_733_367, 17_943_133_541),
    (18_353_823_078, 18_799_290_847),
    (18_865_793_504, 19_327_448_665),
    (19_400_992_346, 19_855_606_483),
    (19_927_356_116, 20_383_760_253),
)

HEADER_PROBE_RANGE = (17_847_729_271, 17_847_733_366)
SELECTED_COMPRESSED_BYTES = sum(member.compressed_bytes for member in SELECTED_MEMBERS)
MISSING_COMPRESSED_BYTES = sum(end - start + 1 for start, end in MISSING_RANGES)
NEW_SELECTED_BYTES_IN_HEADER_PROBE = 4_006
REUSED_SELECTED_COMPRESSED_BYTES = (
    SELECTED_COMPRESSED_BYTES - MISSING_COMPRESSED_BYTES - NEW_SELECTED_BYTES_IN_HEADER_PROBE
)
