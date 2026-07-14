import numpy as np
from shapely.geometry import LineString, box

from assetgen.cook.pyramid import assemble_finer, blocks16, wet_majority
from assetgen.grid import ChunkId
from assetgen.terrain.repair.baseline import SampleGrid
from assetgen.terrain.repair.model import FlowOrientation, FlowProfile
from assetgen.terrain.repair.water_layer import (
    correct_flowing_water_layer,
    reduce_water_lod1,
)


def _profile() -> FlowProfile:
    stations = np.array([6.0, 18.0], dtype=np.float64)
    water = np.array([12.0, 10.0], dtype=np.float64)
    return FlowProfile(
        reach_id="generic-reach",
        epoch_id="qualified-epoch",
        station_m=stations,
        observation_y=water,
        accepted=np.ones(2, dtype=bool),
        water_y=water,
        orientation=FlowOrientation.FORWARD,
        longest_missing_span_m=0.0,
        source_artifact_sha256="ab" * 32,
    )


def test_corrects_only_qualified_wet_with_c0_taper_and_reduces_like_water_policy() -> None:
    grid = SampleGrid(center_e=1.0, center_n=9.0, texel_m=2.0, rows=10, cols=12)
    baseline = np.full((10, 12), np.nan, dtype=np.float32)
    baseline[1:4, :] = 20.0
    mapped = box(0.0, 2.0, 24.0, 8.0)
    qualified = box(6.0, 2.0, 18.0, 8.0)
    centerline = LineString([(0.0, 5.0), (24.0, 5.0)])

    first = correct_flowing_water_layer(
        baseline_water_y=baseline,
        grid=grid,
        mapped_water_polygon=mapped,
        qualified_water_polygon=qualified,
        centerline=centerline,
        profile=_profile(),
        collar_samples=2,
    )
    second = correct_flowing_water_layer(
        baseline_water_y=baseline,
        grid=grid,
        mapped_water_polygon=mapped,
        qualified_water_polygon=qualified,
        centerline=centerline,
        profile=_profile(),
        collar_samples=2,
    )
    assert first.evidence == second.evidence
    np.testing.assert_array_equal(first.water_y, second.water_y)
    np.testing.assert_array_equal(first.water_y[~first.authority], baseline[~first.authority])
    assert np.isnan(first.water_y[4:]).all()
    assert np.all(first.water_y[first.authority] <= 20.0)
    assert np.all(first.water_y[first.authority] >= 10.0)
    assert first.evidence.maximum_boundary_correction_m == 0.0
    assert first.evidence.zero_weight_boundary_samples > 0
    assert first.evidence.tapered_samples > 0
    assert first.evidence.full_profile_samples > 0
    assert not first.authority.flags.writeable

    parent = ChunkId(0, 0, 1)
    core = 8

    def load(chunk: ChunkId) -> np.ndarray:
        row, col = np.mgrid[: core + 1, : core + 1]
        values = 30.0 + (chunk.cz * core + row) * 0.1 + (chunk.cx * core + col) * 0.01
        values[(row + col) % 5 == 0] = np.nan
        return values.astype(np.float32)

    reduced = reduce_water_lod1(parent, load, core_res=core)
    assert reduced is not None
    expected_assembled = assemble_finer(
        lambda chunk: [load(chunk)], parent, core, fills=[np.nan]
    )
    assert expected_assembled is not None
    expected = wet_majority(blocks16(expected_assembled[0]))
    np.testing.assert_allclose(reduced.water_y, expected, rtol=0.0, atol=4e-6)
    assert len(reduced.dependencies) == 25
    assert len(reduced.present_dependencies) == 25
