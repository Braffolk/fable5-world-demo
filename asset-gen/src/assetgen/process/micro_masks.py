"""Direct fine-grid hard masks for cooked microtopography residuals."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio.features
import rasterio.transform

from .etak_read import etak_gpkg, read_layer_window
from .landcover import load_rules, rasterize_classes
from .soil import (
    UNKNOWN,
    boniteet_id,
    normalize_code,
    soil_shp,
    soil_type_id,
    texture_ids,
)

WATER_LAYERS = (
    "E_201_meri_a",
    "E_202_seisuveekogu_a",
    "E_203_vooluveekogu_a",
)
BUILDING_LAYER = "E_401_hoone_ka"
ROAD_LAYER = "E_501_tee_j"
PAVED_ROAD_SURFACES = frozenset((10, 30))  # Pusikate, Kivikate
SLOPE_LAYER = "E_102_nolv_j"
FOREST_CLASS = 1
CLIFF_CONTEXT_BUFFER_M = 2.0  # Half the 4 m measured-patch width.
MESIC_BONITEET_RANGE = (30, 50)
# Unmodified automorphic mineral soils. This is a broad safety classification,
# not a claim that LUKE's foreign analogue has any matching soil taxonomy.
SUPPORTED_MINERAL_SOIL_TYPES = frozenset((*range(1, 16), *range(17, 21)))
PEAT_TEXTURES = frozenset(range(9, 15))


@dataclass(frozen=True)
class MicroHardMask:
    allowed: np.ndarray
    water: np.ndarray
    building: np.ndarray
    paved_road: np.ndarray

    def evidence(self) -> dict:
        cells = int(self.allowed.size)
        return {
            "cells": cells,
            "allowedCells": int(np.count_nonzero(self.allowed)),
            "waterCells": int(np.count_nonzero(self.water)),
            "buildingCells": int(np.count_nonzero(self.building)),
            "pavedRoadCells": int(np.count_nonzero(self.paved_road)),
        }


@dataclass(frozen=True)
class MicroMorphologyMask:
    allowed: np.ndarray
    forest: np.ndarray
    compatible_soil: np.ndarray
    water: np.ndarray
    building: np.ndarray
    paved_road: np.ndarray
    slope_cliff_context: np.ndarray
    unknown_soil: np.ndarray
    unsupported_soil_context: np.ndarray
    drained_soil: np.ndarray

    def evidence(self) -> dict:
        cells = int(self.allowed.size)
        count = np.count_nonzero
        reason_cells = {
            "nonForest": int(count(~self.forest)),
            "incompatibleOrUnsupportedSoil": int(count(~self.compatible_soil)),
            "unknownSoil": int(count(self.unknown_soil)),
            "peatWetModifiedOrGenericSoil": int(count(self.unsupported_soil_context)),
            "drainedSoil": int(count(self.drained_soil)),
            "slopeCliffContext": int(count(self.slope_cliff_context)),
            "water": int(count(self.water)),
            "building": int(count(self.building)),
            "pavedRoad": int(count(self.paved_road)),
        }
        return {
            "cells": cells,
            "allowedCells": int(count(self.allowed)),
            "allowedFraction": float(count(self.allowed) / cells),
            "forestCells": int(count(self.forest)),
            "compatibleSoilCells": int(count(self.compatible_soil)),
            "unknownSoilCells": int(count(self.unknown_soil)),
            "unsupportedSoilContextCells": int(count(self.unsupported_soil_context)),
            "drainedSoilCells": int(count(self.drained_soil)),
            "nonForestCells": int(count(~self.forest)),
            "slopeCliffContextCells": int(count(self.slope_cliff_context)),
            "waterCells": int(count(self.water)),
            "buildingCells": int(count(self.building)),
            "pavedRoadCells": int(count(self.paved_road)),
            "rejectedReasonCells": reason_cells,
            "rejectedReasonFractions": {
                name: float(value / cells) for name, value in reason_cells.items()
            },
        }


def _axis(values: np.ndarray, name: str) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite one-dimensional axis")
    delta = np.diff(values)
    step = float(abs(delta[0]))
    if step <= 0 or not np.allclose(np.abs(delta), step, rtol=0, atol=1e-9):
        raise ValueError(f"{name} must be uniformly spaced")
    return float(values.min() - 0.5 * step), float(values.max() + 0.5 * step), values.size


def _rasterize(geometries: list, shape: tuple[int, int], transform) -> np.ndarray:
    valid = [geometry for geometry in geometries if geometry is not None and not geometry.is_empty]
    if not valid:
        return np.zeros(shape, dtype=bool)
    return rasterio.features.rasterize(
        [(geometry, 1) for geometry in valid],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)


def rasterize_micro_hard_mask(
    east_1d: np.ndarray,
    north_1d: np.ndarray,
) -> MicroHardMask:
    """Rasterize ETAK hard surfaces directly at fine sample centers.

    North must descend so array rows retain the cook's north-to-south convention.
    Unpaved roads and mapped footpaths are intentionally not hard masks.
    """
    e_min, e_max, cols = _axis(east_1d, "east_1d")
    n_min, n_max, rows = _axis(north_1d, "north_1d")
    if float(north_1d[0]) < float(north_1d[-1]):
        raise ValueError("north_1d must descend for direct ETAK rasterization")
    texel = (e_max - e_min) / cols
    if not np.isclose((n_max - n_min) / rows, texel, rtol=0, atol=1e-9):
        raise ValueError("east and north axes must use the same texel size")

    bbox = (e_min, n_min, e_max, n_max)
    transform = rasterio.transform.from_origin(e_min, n_max, texel, texel)
    source = etak_gpkg()

    water_geometries: list = []
    for layer in WATER_LAYERS:
        geometries, _ = read_layer_window(source, layer, bbox)
        water_geometries.extend(geometries)
    building_geometries, _ = read_layer_window(source, BUILDING_LAYER, bbox)
    road_geometries, road_fields = read_layer_window(
        source, ROAD_LAYER, bbox, fields=["teekate", "laius"]
    )
    paved_geometries = []
    surfaces = road_fields.get("teekate", ())
    widths = road_fields.get("laius", ())
    for geometry, surface, width in zip(road_geometries, surfaces, widths):
        if (
            geometry is None
            or surface is None
            or not np.isfinite(surface)
            or int(surface) not in PAVED_ROAD_SURFACES
        ):
            continue
        width_m = float(width) if width is not None and np.isfinite(width) else 0.0
        paved_geometries.append(geometry.buffer(max(0.5 * width_m, 0.5)))

    shape = (rows, cols)
    water = _rasterize(water_geometries, shape, transform)
    building = _rasterize(list(building_geometries), shape, transform)
    paved_road = _rasterize(paved_geometries, shape, transform)
    allowed = ~(water | building | paved_road)
    return MicroHardMask(allowed, water, building, paved_road)


def _compose_morphology_mask(
    landcover: np.ndarray,
    soil_type: np.ndarray,
    texture_core: np.ndarray,
    boniteet: np.ndarray,
    drained_soil: np.ndarray,
    hard: MicroHardMask,
    slope_cliff_context: np.ndarray,
) -> MicroMorphologyMask:
    shape = hard.allowed.shape
    arrays = (
        landcover, soil_type, texture_core, boniteet, drained_soil,
        slope_cliff_context,
    )
    if any(np.asarray(value).shape != shape for value in arrays):
        raise ValueError("morphology selector inputs must share the hard-mask shape")
    forest = np.asarray(landcover) == FOREST_CLASS
    known_type = (soil_type != 0) & (soil_type != UNKNOWN)
    known_texture = texture_core != UNKNOWN
    peat_texture = np.isin(texture_core, tuple(PEAT_TEXTURES))
    supported_mineral = np.isin(soil_type, tuple(SUPPORTED_MINERAL_SOIL_TYPES))
    fertility = (boniteet >= MESIC_BONITEET_RANGE[0]) & (
        boniteet <= MESIC_BONITEET_RANGE[1]
    )
    compatible_soil = (
        known_type
        & known_texture
        & ~peat_texture
        & supported_mineral
        & fertility
        & ~np.asarray(drained_soil, dtype=bool)
    )
    unknown_soil = ~known_type | ~known_texture | (boniteet == 0)
    unsupported_soil_context = (known_type & ~supported_mineral) | peat_texture
    allowed = (
        forest
        & compatible_soil
        & hard.allowed
        & ~np.asarray(slope_cliff_context, dtype=bool)
    )
    return MicroMorphologyMask(
        allowed=allowed,
        forest=forest,
        compatible_soil=compatible_soil,
        water=hard.water,
        building=hard.building,
        paved_road=hard.paved_road,
        slope_cliff_context=np.asarray(slope_cliff_context, dtype=bool),
        unknown_soil=unknown_soil,
        unsupported_soil_context=unsupported_soil_context,
        drained_soil=np.asarray(drained_soil, dtype=bool),
    )


def rasterize_micro_morphology_mask(
    east_1d: np.ndarray,
    north_1d: np.ndarray,
) -> MicroMorphologyMask:
    """Select the only context supported by the Stage-2 foreign analogue.

    This is deliberately fail-closed. It does not infer a geology or texture match:
    LUKE metadata supports only a productive, non-drained mineral forest context.
    """
    e_min, e_max, cols = _axis(east_1d, "east_1d")
    n_min, n_max, rows = _axis(north_1d, "north_1d")
    if float(north_1d[0]) < float(north_1d[-1]):
        raise ValueError("north_1d must descend for direct morphology rasterization")
    texel = (e_max - e_min) / cols
    if not np.isclose((n_max - n_min) / rows, texel, rtol=0, atol=1e-9):
        raise ValueError("east and north axes must use the same texel size")
    window = (e_min, n_min, e_max, n_max, texel)
    bbox = window[:4]
    shape = (rows, cols)
    transform = rasterio.transform.from_origin(e_min, n_max, texel, texel)

    landcover = rasterize_classes(load_rules(), window)
    soil_geometries, soil_fields = read_layer_window(
        soil_shp(),
        "Mullakaart",
        bbox,
        fields=["Sif1", "Lihtloimis", "Boniteet"],
    )
    soil_shapes: list[list[tuple]] = [[], [], [], []]
    for index, geometry in enumerate(soil_geometries):
        if geometry is None or geometry.is_empty:
            continue
        soil_id = soil_type_id(soil_fields["Sif1"][index])
        texture_core, _ = texture_ids(soil_fields["Lihtloimis"][index])
        fertility = boniteet_id(soil_fields["Boniteet"][index])
        raw_code = soil_fields["Sif1"][index]
        drained = int(
            raw_code is not None and normalize_code(str(raw_code)).lower().endswith("d")
        )
        for shapes, value in zip(
            soil_shapes, (soil_id, texture_core, fertility, drained), strict=True
        ):
            shapes.append((geometry, value))
    soil_planes = []
    for shapes in soil_shapes:
        plane = np.zeros(shape, dtype=np.uint8)
        if shapes:
            rasterio.features.rasterize(shapes, out=plane, transform=transform)
        soil_planes.append(plane)

    slope_geometries, _ = read_layer_window(etak_gpkg(), SLOPE_LAYER, bbox)
    slope_context = _rasterize(
        [
            geometry.buffer(CLIFF_CONTEXT_BUFFER_M)
            for geometry in slope_geometries
            if geometry is not None and not geometry.is_empty
        ],
        shape,
        transform,
    )
    hard = rasterize_micro_hard_mask(east_1d, north_1d)
    return _compose_morphology_mask(
        landcover, *soil_planes, hard, slope_context
    )
