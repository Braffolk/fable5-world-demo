"""Frozen, condition-selected Estonia dune pilot and conservative exclusions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyogrio.raw
import rasterio.features
import rasterio.transform
import shapely
from shapely.geometry import Point, box
from shapely.ops import nearest_points


AEOLIAN_FID = 4659
AEOLIAN_LITHOLOGY = 40
AEOLIAN_GENESIS = 100
CORE_CENTER_EN = (394368.0, 6533632.0)
CORE_METERS = 128
SOLVE_HALO_METERS = 4
DTM_SHEET = "61391"


@dataclass(frozen=True)
class PilotConditions:
    core_bbox: tuple[int, int, int, int]
    support_bbox: tuple[int, int, int, int]
    authority_fine: np.ndarray
    forest_fine: np.ndarray
    hard_exclusion_fine: np.ndarray
    protected_fine: np.ndarray
    shore_tangent_en: tuple[float, float]
    shore_normal_inland_en: tuple[float, float]
    nearest_sea_en: tuple[float, float]
    evidence: dict


def _read(
    path: Path,
    *,
    layer: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    columns: tuple[str, ...] = (),
) -> tuple[np.ndarray, list[shapely.Geometry], dict[str, np.ndarray]]:
    metadata, fids, wkbs, values = pyogrio.raw.read(
        path,
        layer=layer,
        bbox=bbox,
        columns=list(columns),
        return_fids=True,
    )
    if str(metadata.get("crs")) != "EPSG:3301" or fids is None or wkbs is None:
        raise ValueError(f"invalid EPSG:3301 vector source: {path}, layer={layer}")
    geometries = [
        shapely.force_2d(shapely.from_wkb(bytes(value))) for value in wkbs
    ]
    return (
        np.asarray(fids),
        geometries,
        dict(zip(metadata.get("fields", ()), values, strict=True)),
    )

def _rasterize(
    geometries: Iterable[shapely.Geometry],
    shape: tuple[int, int],
    transform: rasterio.Affine,
) -> np.ndarray:
    rows = [(geometry, 1) for geometry in geometries if not geometry.is_empty]
    if not rows:
        return np.zeros(shape, dtype=bool)
    return rasterio.features.rasterize(
        rows,
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)


def _buffer_lines(
    geometries: list[shapely.Geometry],
    widths: np.ndarray | None,
    minimum_half_width_m: float,
) -> list[shapely.Geometry]:
    result: list[shapely.Geometry] = []
    for index, geometry in enumerate(geometries):
        width = None if widths is None else widths[index]
        try:
            half_width = max(minimum_half_width_m, 0.5 * float(width))
        except (TypeError, ValueError):
            half_width = minimum_half_width_m
        result.append(geometry.buffer(half_width, cap_style="flat"))
    return result


def load_pilot_conditions(
    *,
    etak: Path,
    geology_200k: Path,
    geology_50k: Path,
) -> PilotConditions:
    """Verify the appearance-independent pilot and rasterize its authority.

    Selection rule frozen before reading the pilot DTM: largest coastal 1:200k
    aeolian-sand polygon with a 192 m interior, then the nearest-sea canonical
    128 m single-sheet window that is fully forested and free of mapped hard
    surfaces. The exact FID and center are retained so source updates fail loud.
    """
    cx, cy = CORE_CENTER_EN
    half = CORE_METERS // 2
    core_bbox = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    support_bbox = tuple(
        value + delta
        for value, delta in zip(core_bbox, (-SOLVE_HALO_METERS,) * 2 + (SOLVE_HALO_METERS,) * 2)
    )
    query_bbox = (core_bbox[0] - 1600, core_bbox[1] - 1600, core_bbox[2] + 1600, core_bbox[3] + 1600)

    fids, geometries, attributes = _read(
        geology_200k,
        bbox=query_bbox,
        columns=("lito200", "genees200"),
    )
    matches = np.flatnonzero(fids == AEOLIAN_FID)
    if len(matches) != 1:
        raise ValueError("frozen aeolian source feature is missing or duplicated")
    index = int(matches[0])
    aeolian = geometries[index]
    if (
        int(attributes["lito200"][index]) != AEOLIAN_LITHOLOGY
        or int(attributes["genees200"][index]) != AEOLIAN_GENESIS
        or not aeolian.covers(box(*support_bbox))
    ):
        raise ValueError("frozen pilot is no longer inside mapped aeolian sand")

    _, fifty_geometries, fifty_attributes = _read(
        geology_50k,
        bbox=core_bbox,
        columns=("kood", "stratigr", "litoloogia"),
    )
    fifty_cover = shapely.union_all(fifty_geometries) if fifty_geometries else shapely.GeometryCollection()
    if not fifty_cover.covers(box(*core_bbox)):
        raise ValueError("pilot no longer has complete 1:50k surficial coverage")

    _, forest_geometries, forest_attributes = _read(
        etak,
        layer="E_305_puittaimestik_a",
        bbox=core_bbox,
        columns=("tyyp", "vajalik_t"),
    )
    forest = [
        geometry
        for geometry, kind in zip(forest_geometries, forest_attributes["tyyp"], strict=True)
        if int(kind) == 10
    ]
    forest_union = shapely.union_all(forest) if forest else shapely.GeometryCollection()
    if not forest_union.covers(box(*core_bbox)):
        raise ValueError("pilot is no longer fully mapped forest")

    hard: list[shapely.Geometry] = []
    protected: list[shapely.Geometry] = []
    dynamic: list[shapely.Geometry] = []
    counts: dict[str, int] = {}
    polygon_layers = (
        "E_201_meri_a",
        "E_202_seisuveekogu_a",
        "E_203_vooluveekogu_a",
        "E_401_hoone_ka",
        "E_403_muu_rajatis_ka",
    )
    for layer in polygon_layers:
        _, layer_geometries, layer_attributes = _read(
            etak,
            layer=layer,
            bbox=core_bbox,
            columns=("vajalik_t",),
        )
        counts[layer] = len(layer_geometries)
        hard.extend(layer_geometries)
        statuses = layer_attributes.get("vajalik_t", np.asarray([], dtype=object))
        dynamic.extend(
            geometry
            for geometry, status in zip(layer_geometries, statuses, strict=True)
            if status is not None and str(status) != "Korras"
        )

    _, road_areas, road_area_attributes = _read(
        etak, layer="E_501_tee_a", bbox=core_bbox, columns=("vajalik_t",)
    )
    _, road_lines, road_line_attributes = _read(
        etak,
        layer="E_501_tee_j",
        bbox=core_bbox,
        columns=("laius", "vajalik_t"),
    )
    roads = [
        *road_areas,
        *_buffer_lines(road_lines, road_line_attributes.get("laius"), 1.5),
    ]
    hard.extend(roads)
    counts["E_501_tee_a"] = len(road_areas)
    counts["E_501_tee_j"] = len(road_lines)
    for geometries_part, attributes_part in (
        (road_areas, road_area_attributes),
        (road_lines, road_line_attributes),
    ):
        statuses = attributes_part.get("vajalik_t", np.asarray([], dtype=object))
        dynamic.extend(
            geometry
            for geometry, status in zip(geometries_part, statuses, strict=True)
            if status is not None and str(status) != "Korras"
        )

    _, slope_lines, slope_attributes = _read(
        etak,
        layer="E_102_nolv_j",
        bbox=core_bbox,
        columns=("vajalik_t",),
    )
    protected.extend(geometry.buffer(2.0, cap_style="flat") for geometry in slope_lines)
    statuses = slope_attributes.get("vajalik_t", np.asarray([], dtype=object))
    dynamic.extend(
        geometry
        for geometry, status in zip(slope_lines, statuses, strict=True)
        if status is not None and str(status) != "Korras"
    )
    counts["E_102_nolv_j"] = len(slope_lines)
    protected.extend(dynamic)

    fine_shape = (CORE_METERS * 4, CORE_METERS * 4)
    transform = rasterio.transform.from_origin(
        core_bbox[0], core_bbox[3], 0.25, 0.25
    )
    aeolian_fine = _rasterize([aeolian], fine_shape, transform)
    forest_fine = _rasterize(forest, fine_shape, transform)
    fifty_fine = _rasterize(fifty_geometries, fine_shape, transform)
    hard_fine = _rasterize(hard, fine_shape, transform)
    protected_fine = _rasterize(protected, fine_shape, transform)
    authority = aeolian_fine & forest_fine & fifty_fine & ~hard_fine & ~protected_fine
    if not np.all(authority):
        raise ValueError("frozen pilot core now intersects a hard, protected, or unknown cell")

    _, sea_geometries, _ = _read(
        etak,
        layer="E_201_meri_a",
        bbox=query_bbox,
    )
    sea = shapely.union_all(sea_geometries)
    center = Point(cx, cy)
    nearest = nearest_points(center, sea)[1]
    inward = np.asarray((cx - nearest.x, cy - nearest.y), dtype=np.float64)
    distance = float(np.linalg.norm(inward))
    if not np.isfinite(distance) or not (500.0 <= distance <= 1000.0):
        raise ValueError(f"pilot shoreline context changed: {distance:.3f} m")
    inward /= distance
    tangent = np.asarray((-inward[1], inward[0]), dtype=np.float64)

    fifty_codes = sorted(
        {
            (int(kood), int(stratigr), int(lithology))
            for kood, stratigr, lithology in zip(
                fifty_attributes["kood"],
                fifty_attributes["stratigr"],
                fifty_attributes["litoloogia"],
                strict=True,
            )
        }
    )
    return PilotConditions(
        core_bbox=core_bbox,
        support_bbox=support_bbox,
        authority_fine=authority,
        forest_fine=forest_fine,
        hard_exclusion_fine=hard_fine,
        protected_fine=protected_fine,
        shore_tangent_en=(float(tangent[0]), float(tangent[1])),
        shore_normal_inland_en=(float(inward[0]), float(inward[1])),
        nearest_sea_en=(float(nearest.x), float(nearest.y)),
        evidence={
            "selectionRule": (
                "largest coastal 1:200k aeolian-sand polygon with a 192 m "
                "interior; nearest-sea canonical 128 m single-sheet window "
                "with full forest coverage and no mapped hard surface"
            ),
            "appearanceUsedForSelection": False,
            "aeolianSourceFid": AEOLIAN_FID,
            "lito200": AEOLIAN_LITHOLOGY,
            "genees200": AEOLIAN_GENESIS,
            "aeolianFeatureAreaM2": float(aeolian.area),
            "forestFraction": float(np.mean(forest_fine)),
            "geology50kCodes": [list(item) for item in fifty_codes],
            "shoreDistanceM": distance,
            "nearestSeaEn": [float(nearest.x), float(nearest.y)],
            "shoreTangentEn": [float(tangent[0]), float(tangent[1])],
            "shoreNormalInlandEn": [float(inward[0]), float(inward[1])],
            "etakFeatureCounts": counts,
            "hardExcludedFineCells": int(np.count_nonzero(hard_fine)),
            "protectedFineCells": int(np.count_nonzero(protected_fine)),
            "authorityFineCells": int(np.count_nonzero(authority)),
        },
    )
