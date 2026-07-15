"""Frozen Alessio V3 source identities and portable morphology selection."""
from __future__ import annotations

from dataclasses import dataclass


DATASET_DOI = "10.17632/6gjn2fmz86.3"
DATASET_VERSION = 3
ARCHIVE_NAME = "Rill_Data_GIS_supplement.mpk"
ARCHIVE_FILE_ID = "1317bf5a-a2f2-4fee-b067-316e6245c407"
ARCHIVE_CONTENT_ID = "0e7b23bf-7177-4744-ab13-0b3371549546"
ARCHIVE_BYTES = 9_040_075_751
ARCHIVE_SHA256 = "61d11524df35353c9339896c7831c9f107a41e7afb87706a4009b8111a110fbc"
LICENSE_SPDX = "CC-BY-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/legalcode"
PUBLISHER_NOTICE = "Further permission may be required for content identified as third-party."

INVENTORY_MEMBER_COUNT = 1_106
INVENTORY_PATHS_SHA256 = "b366b57eba781e276b395fa726ff6686cbc7a3ec98617591f2f771d14a144c7b"
INVENTORY_VERBOSE_SHA256 = "722840d34cd78dc4800c44ed09731aaa8fca6a43a2e769dc296832def86e341a"
PORTABLE_SELECTION_COUNT = 966
PORTABLE_SELECTION_SHA256 = "d40646691038fec732d44c60e9ae01a574fb36e09824865fcb555ad51a13c2e8"
MORPHOLOGY_SELECTION_COUNT = 644
MORPHOLOGY_SELECTION_BYTES = 12_156_440_444


@dataclass(frozen=True)
class Workbook:
    filename: str
    file_id: str
    bytes: int
    sha256: str
    role: str


WORKBOOKS = (
    Workbook(
        "Rill Meaurements and Data.xlsx",
        "550eec89-e0b3-4819-bb78-faa7f4c44c6b",
        338_958,
        "476597deb123d3c0353ee5eaa97ab6947769ea3e072184e857bb02966fecc377",
        "rill width/depth/length and mapped measurement binding",
    ),
    Workbook(
        "S1.2 Infiltration and Runoff Calculations.xlsx",
        "13a1848c-d896-45a8-8114-5a1a1060f914",
        115_238,
        "479a4424b9fed16440c57af07664bd66315d589b8002f51aa461356367418501",
        "runoff condition descriptors",
    ),
    Workbook(
        "S2.5 Sediment budget of a primary rill.xlsx",
        "29350c06-926b-4143-82be-dfce38267d5a",
        82_244,
        "2a6c25bced52fac2d16ecbc5c8c88ca62f2086ab9e51dfa64c65a4cb4d9d000a",
        "primary-rill sediment and morphology context",
    ),
    Workbook(
        "Hydrographs for Fig 9.xlsx",
        "de1c26b4-03c6-49cb-a444-c5f9582d4416",
        41_067,
        "afe31391593d544f7d0b4a5d1a6b16228b547e751f741b5e62d79b96ca12681a",
        "hydrograph condition context",
    ),
)


DEM_DATASETS = (
    "commondata/raster_data/dem_3",
    "commondata/raster_data/dem_4",
    "commondata/raster_data/dem_ss",
    "commondata/raster_data/dem_ss2",
    "commondata/raster_data/sfm_dem",
    "commondata/raster_data2/NW_Corner_DEM.tif",
)

FLOW_DATASETS = {
    "bvc": "commondata/raster_data3/bv_fa.tif",
    "csc": "commondata/raster_data3/cs_fa.tif",
    "hsc": "commondata/raster_data3/hs_fa",
    "oc": "commondata/raster_data3/oc_fa",
    "rc": "commondata/raster_data3/rc_fa",
    "syc": "commondata/raster_data3/syc_fa",
    "regional": "commondata/raster_data/frzn_fa",
}

RILL_VECTORS = {
    "bvc": "commondata/bvc_map/Rills_bvc.shp",
    "csc": "commondata/csc_map/Rills_csc.shp",
    "hsc": "commondata/hsc_map/Rills_hsc.shp",
    "oc": "commondata/oc_map/Rills_OC.shp",
    "rc": "commondata/rc_map/Rills_rc.shp",
    "syc": "commondata/syc_map/25_rills_syc.shp",
}

BEDROCK_VECTORS = {
    "bvc": "commondata/bvc_map/bedrock_bvc.shp",
    "csc": "commondata/csc_map/bedrock_csc.shp",
    "hsc": "commondata/hsc_map/bedrock_hsc.shp",
    "oc": "commondata/oc_map/bedrock_ss_oc.shp",
    "rc": "commondata/rc_map/bedrock_rc.shp",
    "syc": "commondata/syc_map/Bedrock.shp",
}

VEGETATION_VECTORS = {
    "bvc": "commondata/bvc_map/veg_bvc.shp",
    "csc": "commondata/csc_map/vegetation_csc.shp",
    "hsc": "commondata/hsc_map/veg_hsc.shp",
    "oc": "commondata/oc_map/veg_ss_oc.shp",
    "rc": "commondata/rc_map/veg_rc.shp",
    "syc": "commondata/syc_map/Vegetation.shp",
}

GEOLOGY_CONTACTS = (
    "commondata/geology_contacts/Coldw_Cozyd.shp",
    "commondata/geology_contacts/Juncal_Mat.shp",
    "commondata/geology_contacts/Matilija_Cd.shp",
)

WATERSHEDS = (
    "commondata/watershed_boundaries/Buena_Vista.shp",
    "commondata/watershed_boundaries/Cold_Springs.shp",
    "commondata/watershed_boundaries/hot_springs.shp",
    "commondata/watershed_boundaries/Oak_creek.shp",
    "commondata/watershed_boundaries/Romero.shp",
    "commondata/watershed_boundaries/san_ysidro.shp",
)

TRANSECT_LOCATIONS = (
    "commondata/rill_mapping/transect_locations.shp",
    "commondata/rill_mapping/transect_locations_ss.shp",
)
TRANSECT_LINES_GDB = "v106/rill_transect_lines.gdb"


def selected_member(path: str) -> bool:
    raster_prefixes = (
        "commondata/raster_data/dem_3",
        "commondata/raster_data/dem_4",
        "commondata/raster_data/dem_ss",
        "commondata/raster_data/sfm_dem",
        "commondata/raster_data/frzn_fa",
        "commondata/raster_data2/NW_Corner_DEM",
        "commondata/raster_data3/bv_fa",
        "commondata/raster_data3/cs_fa",
        "commondata/raster_data3/hs_fa",
        "commondata/raster_data3/oc_fa",
        "commondata/raster_data3/rc_fa",
        "commondata/raster_data3/syc_fa",
    )
    vector_directories = (
        "bvc_map",
        "csc_map",
        "hsc_map",
        "oc_map",
        "rc_map",
        "syc_map",
        "geology_contacts",
        "watershed_boundaries",
    )
    return (
        path.startswith(raster_prefixes)
        or any(path.startswith(f"commondata/{directory}/") for directory in vector_directories)
        or path.startswith("commondata/rill_mapping/transect_locations")
        or path.startswith(f"{TRANSECT_LINES_GDB}/")
    )


FORBIDDEN_MEMBER_SUBSTRINGS = (
    "ortho",
    "montecito_mosaic_nad83utm11",
    "commondata/raster_data1/nw corner.tif",
    ".avi",
)
