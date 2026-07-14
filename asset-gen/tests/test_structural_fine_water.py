import numpy as np
from shapely.geometry import LineString, box

from assetgen.config import GridConfig
from assetgen.height_geom import HeightChunkId
from assetgen.terrain.repair.baseline import BaselineTile
from assetgen.terrain.repair.closure import (
    close_mixed_water_authority,
    close_mixed_water_surface,
)
from assetgen.terrain.repair.fine_water import reevaluate_fine_flowing_water
from assetgen.terrain.repair.hierarchy import FineSurfaceWindow
from assetgen.terrain.repair.model import FlowOrientation, FlowProfile, WaterSamples


GRID = GridConfig(368640, 6635520, 2048, 4, (0, 1, 2, 3, 4), 2048)


def _water(
    wet: np.ndarray,
    support: np.ndarray,
    water_y: np.ndarray,
    bed_y: np.ndarray,
) -> WaterSamples:
    water = np.where(support, water_y, np.nan)
    bed = np.where(support, bed_y, np.nan)
    return WaterSamples(wet.ravel(), support.ravel(), water.ravel(), bed.ravel())


def test_fine_surface_window_has_the_exact_production_halo_contract() -> None:
    scalar = np.array(72.0, dtype=np.float64)
    structural = np.broadcast_to(scalar, (2081, 2081))
    baseline = np.broadcast_to(scalar, (2081, 2081))
    window = FineSurfaceWindow(HeightChunkId(-2, 2416, 1488), structural, baseline)

    assert window.halo_samples == 16
    assert window.core_side == 2049
    assert window.core_slice == (slice(16, 2065), slice(16, 2065))
    assert window.structural_height.shape == (2081, 2081)
    assert not window.structural_height.flags.writeable


def test_shared_closure_removes_the_same_unsafe_support_at_coarse_and_fine() -> None:
    coarse_shape = (9, 9)
    baseline = np.full(coarse_shape, 9.0, dtype=np.float64)
    valid = np.ones(coarse_shape, dtype=np.bool_)
    wet = np.ones(coarse_shape, dtype=np.bool_)
    support = np.ones(coarse_shape, dtype=np.bool_)
    water_y = np.full(coarse_shape, 10.0, dtype=np.float64)
    bed_y = np.full(coarse_shape, 9.5, dtype=np.float64)
    bed_y[4, 4] = 10.25
    samples = _water(wet, support, water_y, bed_y)

    coarse = close_mixed_water_authority(
        BaselineTile(baseline, valid), samples, collar_samples=4
    )
    direct = close_mixed_water_surface(
        baseline, valid, samples, collar_samples=4
    )
    np.testing.assert_array_equal(coarse.removed_support, direct.removed_support)

    fine = close_mixed_water_surface(
        np.repeat(np.repeat(baseline, 4, axis=0), 4, axis=1),
        np.repeat(np.repeat(valid, 4, axis=0), 4, axis=1),
        _water(
            np.repeat(np.repeat(wet, 4, axis=0), 4, axis=1),
            np.repeat(np.repeat(support, 4, axis=0), 4, axis=1),
            np.repeat(np.repeat(water_y, 4, axis=0), 4, axis=1),
            np.repeat(np.repeat(bed_y, 4, axis=0), 4, axis=1),
        ),
        collar_samples=16,
    )
    fine_removed = fine.removed_support.reshape(9, 4, 9, 4).any(axis=(1, 3))
    np.testing.assert_array_equal(fine_removed, coarse.removed_support)
    assert coarse.removed_support[4, 4]


def test_shared_closure_restores_every_dry_and_abstained_sample_exactly() -> None:
    row, col = np.mgrid[:17, :17]
    baseline = 91.0 + row * 0.003 + col * 0.007
    valid = np.ones_like(baseline, dtype=np.bool_)
    wet = (col >= 3) & (col <= 13)
    support = wet & (col <= 9)
    water_y = np.full_like(baseline, 100.0)
    bed_y = np.full_like(baseline, 92.0)

    result = close_mixed_water_surface(
        baseline,
        valid,
        _water(wet, support, water_y, bed_y),
        collar_samples=4,
    )
    unowned = ~result.authority
    np.testing.assert_array_equal(result.height[unowned], baseline[unowned])
    np.testing.assert_array_equal(result.abstained, wet & ~result.authority)
    np.testing.assert_array_equal(
        result.height[result.authority & (col <= 5)],
        bed_y[result.authority & (col <= 5)],
    )


def test_fine_evaluator_closes_expanded_window_before_cropping() -> None:
    chunk = HeightChunkId(-2, 0, 0)
    baseline = np.full((49, 49), 9.5, dtype=np.float64)
    structural = np.full((49, 49), 8.0, dtype=np.float64)
    window = FineSurfaceWindow(chunk, structural, baseline)
    e0, n0 = 368640.0, 6635520.0
    mapped = box(e0 - 0.5, n0 - 1.0, e0 + 1.25, n0 + 0.5)
    qualified = box(e0 - 0.5, n0 - 1.0, e0 + 0.75, n0 + 0.5)
    centerline = LineString([(e0 - 0.5, n0 - 0.25), (e0 + 0.75, n0 - 0.25)])
    profile = FlowProfile(
        reach_id="fixture-reach",
        epoch_id="fixture-epoch",
        station_m=np.array([0.0, centerline.length]),
        observation_y=np.array([10.0, 10.0]),
        accepted=np.array([True, True]),
        water_y=np.array([10.0, 10.0]),
        orientation=FlowOrientation.FORWARD,
        longest_missing_span_m=0.0,
        source_artifact_sha256="1" * 64,
    )

    result = reevaluate_fine_flowing_water(
        window,
        grid=GRID,
        mapped_water_polygon=mapped,
        qualified_water_polygon=qualified,
        centerline=centerline,
        profile=profile,
    )
    assert result.height.shape == (17, 17)
    baseline_core = baseline[window.core_slice]
    np.testing.assert_array_equal(
        result.height[~result.authority], baseline_core[~result.authority]
    )
    assert result.authority.any()
    assert result.abstained.any()
    assert not np.any(result.authority & result.abstained)
