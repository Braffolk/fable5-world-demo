"""Exact LOD -2 flowing-water ownership and bed reevaluation."""
from __future__ import annotations

import numpy as np
import shapely

from ...config import GridConfig
from ...height_geom import (
    UNITS_PER_METER,
    chunk_origin_en_units,
    texel_units,
)
from .hierarchy import FineReevaluationResult, FineSurfaceWindow
from .model import FlowProfile, WaterSamples
from .water import evaluate_flowing_water
from .closure import (
    MIXED_AUTHORITY_ADJACENCY,
    MIXED_AUTHORITY_CLOSURE_VERSION,
    MIXED_AUTHORITY_COLLAR_SAMPLES,
    close_mixed_water_surface,
)


_FINE_LOD = -2
_ROW_BATCH = 64
FINE_WATER_CLOSURE_VERSION = MIXED_AUTHORITY_CLOSURE_VERSION
FINE_WATER_COLLAR_SAMPLES = MIXED_AUTHORITY_COLLAR_SAMPLES * 4
FINE_WATER_ADJACENCY = MIXED_AUTHORITY_ADJACENCY

def _validate_authority(mapped_water_polygon, qualified_water_polygon, centerline) -> None:
    for name, geometry in (
        ("mapped_water_polygon", mapped_water_polygon),
        ("qualified_water_polygon", qualified_water_polygon),
        ("centerline", centerline),
    ):
        if geometry is None or geometry.is_empty or not geometry.is_valid:
            raise ValueError(f"{name} must be nonempty and valid")
    if mapped_water_polygon.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError("mapped water must be polygonal")
    if qualified_water_polygon.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError("qualified water must be polygonal")
    if not mapped_water_polygon.covers(qualified_water_polygon):
        raise ValueError("qualified water territory must stay inside mapped water")
    if centerline.geom_type != "LineString" or not centerline.is_simple:
        raise ValueError("centerline must be one simple unbranched LineString")


def reevaluate_fine_flowing_water(
    context: FineSurfaceWindow,
    *,
    grid: GridConfig,
    mapped_water_polygon,
    qualified_water_polygon,
    centerline,
    profile: FlowProfile,
) -> FineReevaluationResult:
    """Reclassify and evaluate a packed-bed surface at exact 0.0625 m centers.

    Only samples supported by both qualified geometry and the qualified profile
    own the conservative-bed correction. A one-metre mixed-authority collar
    closes that correction to the independent baseline before mapped abstention;
    actual dry shoreline is not a taper seed. Rows are batched solely to bound
    temporary GEOS point storage.
    """
    if not isinstance(context, FineSurfaceWindow):
        raise TypeError("context must be FineSurfaceWindow")
    if context.chunk.lod != _FINE_LOD or texel_units(context.chunk.lod) != 2:
        raise ValueError("fine flowing-water reevaluation requires LOD -2")
    if not isinstance(profile, FlowProfile):
        raise TypeError("profile must be one qualified FlowProfile")
    _validate_authority(mapped_water_polygon, qualified_water_polygon, centerline)
    if profile.station_m[0] < 0.0 or profile.station_m[-1] > centerline.length:
        raise ValueError("profile station domain leaves the qualified centerline")

    rows, cols = context.baseline_height.shape
    halo = context.halo_samples
    if halo != FINE_WATER_COLLAR_SAMPLES:
        raise ValueError("fine water closure and surface-window halos differ")
    core_rows = core_cols = context.core_side
    origin_e, origin_n = chunk_origin_en_units(grid, context.chunk)
    step = texel_units(context.chunk.lod)
    half = step // 2
    east = (
        origin_e + np.arange(-halo, core_cols + halo, dtype=np.int64) * step + half
    ).astype(np.float64) / UNITS_PER_METER
    north_first = (origin_n + halo * step - half) / UNITS_PER_METER
    north_last = (
        origin_n - (core_rows + halo - 1) * step - half
    ) / UNITS_PER_METER

    sample_envelope = shapely.box(east[0], north_last, east[-1], north_first)
    if not mapped_water_polygon.intersects(sample_envelope):
        empty = np.zeros((core_rows, core_cols), dtype=np.bool_)
        return FineReevaluationResult(
            context.baseline_height[context.core_slice], empty, empty
        )

    mapped_wet = np.zeros((rows, cols), dtype=np.bool_)
    support = np.zeros((rows, cols), dtype=np.bool_)
    water_y = np.full((rows, cols), np.nan, dtype=np.float64)
    bed_y = np.full((rows, cols), np.nan, dtype=np.float64)
    for row_start in range(0, rows, _ROW_BATCH):
        row_stop = min(row_start + _ROW_BATCH, rows)
        row_index = np.arange(row_start - halo, row_stop - halo, dtype=np.int64)
        north = (origin_n - row_index * step - half).astype(np.float64)
        north /= UNITS_PER_METER
        sample_e = np.broadcast_to(
            east, (row_stop - row_start, cols)
        ).ravel()
        sample_n = np.broadcast_to(
            north[:, None], (row_stop - row_start, cols)
        ).ravel()
        points = shapely.points(sample_e, sample_n)
        batch_mapped = np.asarray(
            shapely.covers(mapped_water_polygon, points), dtype=np.bool_
        )
        evaluated = evaluate_flowing_water(
            easting=sample_e,
            northing=sample_n,
            water_polygon=qualified_water_polygon,
            centerline=centerline,
            profile=profile,
        )
        if np.any(evaluated.wet & ~batch_mapped):
            raise ValueError("qualified fine samples leave mapped water ownership")

        batch_shape = (row_stop - row_start, cols)
        target = slice(row_start, row_stop)
        mapped_wet[target] = batch_mapped.reshape(batch_shape)
        support[target] = evaluated.supported.reshape(batch_shape)
        water_y[target] = evaluated.water_y.reshape(batch_shape)
        bed_y[target] = evaluated.bed_y.reshape(batch_shape)

    water = WaterSamples(
        wet=mapped_wet.ravel(),
        supported=support.ravel(),
        water_y=water_y.ravel(),
        bed_y=bed_y.ravel(),
    )
    # WaterSamples owns immutable copies; release the four rasterization buffers
    # before allocating closure weights and output at production resolution.
    del mapped_wet, support, water_y, bed_y
    baseline_valid = np.ones(context.baseline_height.shape, dtype=np.bool_)
    closed = close_mixed_water_surface(
        context.baseline_height,
        baseline_valid,
        water,
        collar_samples=FINE_WATER_COLLAR_SAMPLES,
    )
    removed = closed.removed_support
    if (
        np.any(removed[0, :])
        or np.any(removed[-1, :])
        or np.any(removed[:, 0])
        or np.any(removed[:, -1])
    ):
        raise ValueError("fine mixed-authority abstention reached the real halo edge")

    crop = context.core_slice
    height = np.ascontiguousarray(closed.height[crop])
    authority = np.ascontiguousarray(closed.authority[crop])
    abstained = np.ascontiguousarray(closed.abstained[crop])
    baseline_core = context.baseline_height[crop]

    unowned = ~authority
    if not np.array_equal(height[unowned], baseline_core[unowned]):
        raise AssertionError("fine water reevaluation leaked outside qualified authority")
    if np.any(authority & abstained):
        raise AssertionError("fine water authority and abstention overlap")
    return FineReevaluationResult(height, authority, abstained)
