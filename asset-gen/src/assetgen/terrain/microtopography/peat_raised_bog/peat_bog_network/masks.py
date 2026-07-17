"""ETAK-derived mire coverage, hard-exclusion and open-water masks.

Self-contained pyogrio reads (no v1/v2 code path). Rasterised at an arbitrary pitch over
an integer bbox; rows north-to-south. Open water and hard objects stay relief-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyogrio.raw
import rasterio.features
import rasterio.transform
import shapely

MIRE_LAYER = "E_306_margala_a"
DITCH_TYPES = {20, 40, 50}

HARD_AREA_LAYERS = {
    "forest": "E_305_puittaimestik_a",
    "sea": "E_201_meri_a",
    "lake": "E_202_seisuveekogu_a",
    "flowing_water": "E_203_vooluveekogu_a",
    "cut_peat": "E_307_turbavali_a",
    "road_area": "E_501_tee_a",
    "building": "E_401_hoone_ka",
    "high_object": "E_402_korgrajatis_p",
    "other_object_area": "E_403_muu_rajatis_ka",
    "other_object_point": "E_403_muu_rajatis_p",
}
WATER_AREA_LAYERS = ("sea", "lake", "flowing_water")


@dataclass(frozen=True)
class MaskSet:
    mire_coverage: np.ndarray
    hard_exclusion: np.ndarray
    open_water: np.ndarray
    authority: np.ndarray
    bbox_en: tuple[int, int, int, int]
    pitch_m: float


def _read_layer(
    etak: Path, layer: str, bbox: tuple[int, int, int, int], columns: tuple[str, ...]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    metadata, _fids, wkbs, values = pyogrio.raw.read(
        etak, layer=layer, bbox=bbox, columns=list(columns), return_fids=True
    )
    if str(metadata.get("crs")) != "EPSG:3301":
        raise ValueError(f"ETAK layer {layer} is not EPSG:3301")
    if wkbs is None:
        return np.asarray([], dtype=object), {}
    geometries = np.asarray(
        [shapely.force_2d(shapely.from_wkb(bytes(w))) for w in wkbs], dtype=object
    )
    return geometries, dict(zip(metadata.get("fields", ()), values, strict=True))


def _rasterize(geometries, shape, transform) -> np.ndarray:
    rows = [(g, 1) for g in geometries if g is not None and not g.is_empty]
    if not rows:
        return np.zeros(shape, dtype=bool)
    return rasterio.features.rasterize(
        rows, out_shape=shape, transform=transform, fill=0, all_touched=True, dtype="uint8"
    ).astype(bool)


def _buffer_lines(geometries, widths, min_half, selected=None):
    out = []
    for i, g in enumerate(geometries):
        if selected is not None and not bool(selected[i]):
            continue
        try:
            half = max(min_half, 0.5 * float(widths[i]))
        except (TypeError, ValueError, KeyError):
            half = min_half
        out.append(g.buffer(half, cap_style="flat"))
    return out


def build(
    etak: Path,
    mire_geometry: shapely.Geometry,
    bbox_en: tuple[int, int, int, int],
    pitch_m: float,
) -> MaskSet:
    width = int(round((bbox_en[2] - bbox_en[0]) / pitch_m))
    height = int(round((bbox_en[3] - bbox_en[1]) / pitch_m))
    shape = (height, width)
    transform = rasterio.transform.from_origin(bbox_en[0], bbox_en[3], pitch_m, pitch_m)

    mire = _rasterize([mire_geometry], shape, transform)
    area_geoms: dict[str, np.ndarray] = {}
    for name, layer in HARD_AREA_LAYERS.items():
        geoms, _attrs = _read_layer(etak, layer, bbox_en, ("etak_id",))
        area_geoms[name] = geoms

    water_lines, water_attrs = _read_layer(
        etak, "E_203_vooluveekogu_j", bbox_en, ("tyyp", "laius")
    )
    line_types = water_attrs.get("tyyp", np.full(len(water_lines), -1))
    ditch_sel = np.asarray(
        [v is not None and int(v) in DITCH_TYPES for v in line_types], dtype=bool
    )
    ditches = _buffer_lines(water_lines, water_attrs.get("laius"), 1.0, ditch_sel)
    road_lines, road_attrs = _read_layer(etak, "E_501_tee_j", bbox_en, ("laius",))
    roads = [*area_geoms["road_area"], *_buffer_lines(road_lines, road_attrs.get("laius"), 1.5)]

    water = [g for name in WATER_AREA_LAYERS for g in area_geoms[name]]
    open_water = _rasterize([*water, *_buffer_lines(water_lines, water_attrs.get("laius"), 1.0)], shape, transform)
    ditch = _rasterize(ditches, shape, transform)
    forest = _rasterize(area_geoms["forest"], shape, transform)
    cut_peat = _rasterize(area_geoms["cut_peat"], shape, transform)
    road = _rasterize(roads, shape, transform)
    building = _rasterize(area_geoms["building"], shape, transform)
    objects = _rasterize(
        [*area_geoms["high_object"], *area_geoms["other_object_area"], *area_geoms["other_object_point"]],
        shape,
        transform,
    )
    hard = forest | open_water | ditch | cut_peat | road | building | objects
    authority = mire & ~hard
    return MaskSet(
        mire_coverage=mire,
        hard_exclusion=hard,
        open_water=open_water | ditch,
        authority=authority,
        bbox_en=bbox_en,
        pitch_m=pitch_m,
    )
