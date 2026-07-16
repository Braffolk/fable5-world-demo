"""Categorical EGT geology planes for runtime surface-material selection.

The source polygons are regional map evidence. They identify the mapped surficial
material/process and the bedrock formation beneath it; they do not assert that the
bedrock is exposed unless the surficial map explicitly says so.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyogrio.raw
import rasterio.features
import rasterio.transform
import shapely

from ..config import ASSET_GEN_ROOT

_EGT_ROOT = (
    ASSET_GEN_ROOT.parent
    / "docs"
    / "deep-research"
    / "microtopography-generation"
    / "library"
    / "data"
    / "egt"
)
SURFICIAL_SOURCE = _EGT_ROOT / "pinnakate-200k" / "q_avamus_a_200t.shp"
BEDROCK_SOURCE = _EGT_ROOT / "aluspohi-200k" / "AP_Avamus_a_200T.shp"

# Plane R: bedrock/exposed-rock family.
BEDROCK_UNKNOWN = 0
BEDROCK_SANDSTONE = 1
BEDROCK_CARBONATE = 2
BEDROCK_OTHER = 3

# Plane G: surficial material family.
SURFICIAL_UNKNOWN = 0
SURFICIAL_SAND = 1
SURFICIAL_TILL = 2
SURFICIAL_GRAVEL = 3
SURFICIAL_PEAT = 4
SURFICIAL_OTHER = 5

# Plane B: genesis/process family.
PROCESS_UNKNOWN = 0
PROCESS_FLUVIAL = 1
PROCESS_LACUSTRINE = 2
PROCESS_GLACIOFLUVIAL = 3
PROCESS_GLACIOLACUSTRINE = 4
PROCESS_GLACIAL = 5
PROCESS_MARINE = 6
PROCESS_COLLUVIAL = 7
PROCESS_PEAT = 8
PROCESS_ANTHROPOGENIC = 9
PROCESS_AEOLIAN = 10
PROCESS_WATER = 11
PROCESS_BEDROCK_EXPOSURE = 12

# Plane A: independent authority/known flags.
COVERAGE_AUTHORITATIVE = 1 << 0
COVERAGE_BEDROCK_KNOWN = 1 << 1
COVERAGE_SURFICIAL_KNOWN = 1 << 2
COVERAGE_PROCESS_KNOWN = 1 << 3
COVERAGE_MAPPED_BEDROCK_EXPOSURE = 1 << 4
COVERAGE_SCALE_50K = 1 << 5

# Official EGT lito200 domain -> stable render family. Codes remain documented in
# manifest.py; 997/998 deliberately stay unknown rather than becoming "other".
SURFICIAL_FAMILY_BY_CODE = {
    10: SURFICIAL_OTHER,  # silt
    20: SURFICIAL_OTHER,  # lake lime
    30: SURFICIAL_GRAVEL,
    40: SURFICIAL_SAND,
    50: SURFICIAL_TILL,
    60: SURFICIAL_OTHER,  # mud
    70: SURFICIAL_OTHER,
    80: SURFICIAL_OTHER,  # clay
    90: SURFICIAL_PEAT,
    100: SURFICIAL_OTHER,  # anthropogenic sediment
    30000: SURFICIAL_OTHER,  # crystalline-basement exposure
}

# Official EGT genees200 domain -> stable process family.
PROCESS_FAMILY_BY_CODE = {
    10: PROCESS_FLUVIAL,
    20: PROCESS_LACUSTRINE,
    30: PROCESS_GLACIOFLUVIAL,
    40: PROCESS_GLACIOLACUSTRINE,
    50: PROCESS_GLACIAL,
    60: PROCESS_MARINE,
    70: PROCESS_COLLUVIAL,
    80: PROCESS_PEAT,
    90: PROCESS_ANTHROPOGENIC,
    100: PROCESS_AEOLIAN,
    112: PROCESS_WATER,
}

# National 1:200k bedrock formation index -> lithologic family. Mixed marl/clay,
# shale, crystalline, diapir, and underspecified systems remain OTHER. The target
# Devonian formations (Burtnieki/Arukula) are explicit rather than inferred from
# coordinates. This is a palette contract, not an assertion of surface exposure.
_SANDSTONE_FORMATIONS = {
    2110400,  # Amata
    2120200,  # Gauja
    2130100,  # Burtnieki
    2130200,  # Arukula
    2140100,  # Parnu
    2440100,  # Leetse
    2510100,  # Kallavere
    2610200,  # Ulgase
    2630400,  # Tiskre
}
_CARBONATE_FORMATIONS = {
    2110100, 2110200, 2110300,  # Daugava, Dubniki, Plavinas
    2210100, 2210200, 2220100, 2230100, 2240100, 2250200, 2250300,
    2260100, 2270100, 2270200, 2270300, 2270500, 2270600, 2310200,
    2410300, 2410400, 2410800, 2410900, 2411200, 2411300, 2411500,
    2411800, 2411900, 2412000, 2412100, 2412600, 2420100, 2420300,
    2420500, 2420900, 2421200, 2421350, 2430100,
}
BEDROCK_FAMILY_BY_INDEX = {
    **{index: BEDROCK_SANDSTONE for index in _SANDSTONE_FORMATIONS},
    **{index: BEDROCK_CARBONATE for index in _CARBONATE_FORMATIONS},
    **{
        index: BEDROCK_OTHER
        for index in {
            2130300, 2421000, 2440400, 2500001, 2630500, 2640200,
            3100000, 4000000, 5000000,
        }
    },
}


def _read_polygons(path: Path, bbox: tuple[float, float, float, float], fields: list[str]):
    if not path.is_file():
        raise FileNotFoundError(f"retained EGT source is absent: {path}")
    metadata, _, geometry_wkb, columns = pyogrio.raw.read(
        path, bbox=bbox, columns=fields, return_fids=True
    )
    if geometry_wkb is None or [str(value) for value in metadata["fields"]] != fields:
        raise ValueError(f"EGT source schema differs: {path}")
    geometries = shapely.from_wkb(geometry_wkb)
    return geometries, {field: columns[i] for i, field in enumerate(fields)}


def _burn(
    geometries,
    values,
    *,
    shape: tuple[int, int],
    transform,
) -> np.ndarray:
    pairs = [
        (geometry, int(value))
        for geometry, value in zip(geometries, values, strict=True)
        if geometry is not None and not geometry.is_empty and int(value) != 0
    ]
    return rasterio.features.rasterize(
        pairs,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )


def rasterize_geology(
    window_en: tuple[float, float, float, float, float],
) -> list[np.ndarray]:
    """Return [bedrock, surficial, process, coverage] u8 planes at ``window_en``."""
    e_min, n_min, e_max, n_max, texel = window_en
    cols = round((e_max - e_min) / texel)
    rows = round((n_max - n_min) / texel)
    shape = (rows, cols)
    transform = rasterio.transform.from_origin(e_min, n_max, texel, texel)
    bbox = (e_min, n_min, e_max, n_max)

    bed_geoms, bed_data = _read_polygons(BEDROCK_SOURCE, bbox, ["indeks"])
    bed_values = np.asarray(
        [BEDROCK_FAMILY_BY_INDEX.get(int(value), BEDROCK_UNKNOWN) for value in bed_data["indeks"]],
        dtype=np.uint8,
    )
    bedrock = _burn(bed_geoms, bed_values, shape=shape, transform=transform)
    bed_coverage = _burn(
        bed_geoms, np.ones(len(bed_geoms), dtype=np.uint8), shape=shape, transform=transform
    )

    surf_geoms, surf_data = _read_polygons(
        SURFICIAL_SOURCE, bbox, ["lito200", "genees200"]
    )
    raw_lithology = np.asarray(surf_data["lito200"], dtype=np.int32)
    raw_process = np.asarray(surf_data["genees200"], dtype=np.int32)
    exposure = raw_lithology == 20000
    surf_values = np.asarray(
        [SURFICIAL_FAMILY_BY_CODE.get(int(value), SURFICIAL_UNKNOWN) for value in raw_lithology],
        dtype=np.uint8,
    )
    process_values = np.asarray(
        [PROCESS_FAMILY_BY_CODE.get(int(value), PROCESS_UNKNOWN) for value in raw_process],
        dtype=np.uint8,
    )
    process_values[exposure] = PROCESS_BEDROCK_EXPOSURE
    surficial = _burn(surf_geoms, surf_values, shape=shape, transform=transform)
    process = _burn(surf_geoms, process_values, shape=shape, transform=transform)
    surf_coverage = _burn(
        surf_geoms, np.ones(len(surf_geoms), dtype=np.uint8), shape=shape, transform=transform
    )
    exposure_plane = _burn(
        surf_geoms, exposure.astype(np.uint8), shape=shape, transform=transform
    )

    coverage = np.zeros(shape, dtype=np.uint8)
    coverage[(bed_coverage | surf_coverage) != 0] |= COVERAGE_AUTHORITATIVE
    coverage[bedrock != BEDROCK_UNKNOWN] |= COVERAGE_BEDROCK_KNOWN
    known_surf_codes = set(SURFICIAL_FAMILY_BY_CODE) | {20000}
    surf_known_values = np.asarray(
        [int(value) in known_surf_codes for value in raw_lithology], dtype=np.uint8
    )
    surf_known = _burn(surf_geoms, surf_known_values, shape=shape, transform=transform)
    coverage[surf_known != 0] |= COVERAGE_SURFICIAL_KNOWN
    coverage[process != PROCESS_UNKNOWN] |= COVERAGE_PROCESS_KNOWN
    coverage[exposure_plane != 0] |= COVERAGE_MAPPED_BEDROCK_EXPOSURE
    return [bedrock, surficial, process, coverage]
