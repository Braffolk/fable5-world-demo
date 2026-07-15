"""Frozen publisher identity and remote ZIP inventory for Hinsberger evidence."""
from __future__ import annotations

from dataclasses import dataclass


DATASET_DOI = "10.6084/m9.figshare.24592338.v1"
PUBLISHER_FILE_ID = 43_218_330
ARCHIVE_NAME = "Hinsberger_aerial_survey_data.zip"
ARCHIVE_BYTES = 20_383_764_301
ARCHIVE_MD5 = "b7729bd73319e1860b98437df685c3a4"
LICENSE_SPDX = "CC-BY-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


@dataclass(frozen=True)
class ArchiveMember:
    path: str
    compressed_bytes: int
    uncompressed_bytes: int
    crc32: int
    is_dir: bool = False


@dataclass(frozen=True)
class SurveyPair:
    survey_id: str
    dem_member: str
    orthomosaic_member: str


def _directory(path: str) -> ArchiveMember:
    return ArchiveMember(path, 0, 0, 0, True)


# Exact Figshare central-directory inventory acquired before the raw download.
ARCHIVE_MEMBERS = (
    _directory("Hinsberger_aerial_survey_data/"),
    _directory("Hinsberger_aerial_survey_data/Field1/"),
    _directory("Hinsberger_aerial_survey_data/Field10/"),
    ArchiveMember("Hinsberger_aerial_survey_data/Field10/field10_dem.tif", 331_423_225, 1_885_771_868, 0xD063E458),
    ArchiveMember("Hinsberger_aerial_survey_data/Field10/field10_orthomosaic.tif", 3_123_548_288, 3_214_954_429, 0xB131B2AC),
    _directory("Hinsberger_aerial_survey_data/Field11/"),
    ArchiveMember("Hinsberger_aerial_survey_data/Field11/field11_dem.tif", 251_128_120, 1_882_231_627, 0xC27F58DD),
    ArchiveMember("Hinsberger_aerial_survey_data/Field11/field11_orthomosaic.tif", 2_446_260_086, 2_526_502_966, 0x43AA7C18),
    ArchiveMember("Hinsberger_aerial_survey_data/Field1/Field1_dem.tif", 213_272_250, 1_248_113_051, 0x4E35027B),
    ArchiveMember("Hinsberger_aerial_survey_data/Field1/Field1_orthomosaic.tif", 1_943_600_588, 2_029_114_451, 0x6639ACD9),
    _directory("Hinsberger_aerial_survey_data/Field2/"),
    ArchiveMember("Hinsberger_aerial_survey_data/Field2/Field2_dem.tif", 291_007_738, 1_015_144_092, 0xCACE520D),
    ArchiveMember("Hinsberger_aerial_survey_data/Field2/Field2_orthomosaic.tif", 1_325_563_743, 1_370_058_237, 0x1B674877),
    _directory("Hinsberger_aerial_survey_data/Field3/"),
    ArchiveMember("Hinsberger_aerial_survey_data/Field3/Field3_dem.tif", 427_860_716, 3_193_103_101, 0x0807D658),
    ArchiveMember("Hinsberger_aerial_survey_data/Field3/Field3_orthomosaic.tif", 2_727_833_538, 2_810_430_299, 0xB1F44AB8),
    _directory("Hinsberger_aerial_survey_data/Field4_Field5/"),
    ArchiveMember("Hinsberger_aerial_survey_data/Field4_Field5/Field4_5_dem.tif", 141_122_795, 999_670_971, 0x8B967C0E),
    ArchiveMember("Hinsberger_aerial_survey_data/Field4_Field5/Field4_5_orthomosaic.tif", 1_287_877_681, 1_319_113_873, 0x9FE17CF7),
    _directory("Hinsberger_aerial_survey_data/Field6/"),
    ArchiveMember("Hinsberger_aerial_survey_data/Field6/Field6_dem.tif", 108_467_743, 676_499_611, 0x9C8B1FE7),
    ArchiveMember("Hinsberger_aerial_survey_data/Field6/Field6_orthomosaic.tif", 1_073_748_054, 1_111_955_733, 0x37FCF0A2),
    _directory("Hinsberger_aerial_survey_data/Field7/"),
    ArchiveMember("Hinsberger_aerial_survey_data/Field7/Field7_dem.tif", 188_489_344, 1_763_024_011, 0xA8F1AB17),
    ArchiveMember("Hinsberger_aerial_survey_data/Field7/Field7_orthomosaic.tif", 1_966_523_297, 2_038_568_917, 0xE39794EA),
    _directory("Hinsberger_aerial_survey_data/Field8_Field9/"),
    ArchiveMember("Hinsberger_aerial_survey_data/Field8_Field9/Field8_9_dem.tif", 312_826_451, 1_349_066_524, 0x89D244FE),
    ArchiveMember("Hinsberger_aerial_survey_data/Field8_Field9/Field8_9_orthomosaic.tif", 2_223_204_344, 2_294_651_505, 0x8685172B),
)

SELECTED_SURVEYS = (
    SurveyPair(
        "Field4_5",
        "Hinsberger_aerial_survey_data/Field4_Field5/Field4_5_dem.tif",
        "Hinsberger_aerial_survey_data/Field4_Field5/Field4_5_orthomosaic.tif",
    ),
    SurveyPair(
        "Field7",
        "Hinsberger_aerial_survey_data/Field7/Field7_dem.tif",
        "Hinsberger_aerial_survey_data/Field7/Field7_orthomosaic.tif",
    ),
    SurveyPair(
        "Field8_9",
        "Hinsberger_aerial_survey_data/Field8_Field9/Field8_9_dem.tif",
        "Hinsberger_aerial_survey_data/Field8_Field9/Field8_9_orthomosaic.tif",
    ),
)

ANALYSIS_CELL_M = 0.0625
ANALYSIS_HALO_M = 1.875
ANALYSIS_CORE_M = 32.0
F4_COEFFICIENTS = (1, 3, 6, 10, 12, 12, 10, 6, 3, 1)


def member_by_path() -> dict[str, ArchiveMember]:
    return {member.path: member for member in ARCHIVE_MEMBERS}
