import hashlib

import numpy as np
from shapely.geometry import LineString, box

from assetgen.config import GridConfig
from assetgen.grid import ChunkId
from assetgen.terrain.repair.model import FlowOrientation, FlowProfile
from assetgen.terrain.repair.water_halo import (
    DecodedWaterHaloChunk,
    WaterHaloSource,
    correct_flowing_water_chunk_halo,
)


GRID = GridConfig(0, 0, 16, 4, (0, 1), 16)


def test_real_halo_closes_correction_to_abstention_beyond_payload_edge() -> None:
    target = ChunkId(0, 0, 0)
    records: dict[ChunkId, DecodedWaterHaloChunk] = {}
    for dz in (-1, 0, 1):
        for dx in (-1, 0, 1):
            chunk = ChunkId(dx, dz, 0)
            present = dx >= 0 and dz >= 0
            source = (
                WaterHaloSource.INHERITED
                if (dx, dz) == (0, 0)
                else WaterHaloSource.CORRECTED
                if present
                else WaterHaloSource.ABSENT
            )
            water = np.full((9, 9), 20.0, dtype=np.float32) if present else None
            digest = hashlib.sha256(f"halo:{source}:{chunk}".encode()).hexdigest()
            records[chunk] = DecodedWaterHaloChunk(chunk, source, water, digest)

    qualified = box(0.0, -18.0, 18.0, 0.0)
    mapped = box(0.0, -28.0, 28.0, 0.0)
    centerline = LineString([(0.0, -9.0), (18.0, -9.0)])
    profile = FlowProfile(
        reach_id="generic-cross-chunk-reach",
        epoch_id="qualified-epoch",
        station_m=np.array([0.0, centerline.length]),
        observation_y=np.array([10.0, 10.0]),
        accepted=np.array([True, True]),
        water_y=np.array([10.0, 10.0]),
        orientation=FlowOrientation.FORWARD,
        longest_missing_span_m=0.0,
        source_artifact_sha256="1" * 64,
    )
    corrected = correct_flowing_water_chunk_halo(
        chunk=target,
        grid=GRID,
        load_baseline=records.__getitem__,
        mapped_water_polygon=mapped,
        qualified_water_polygon=qualified,
        centerline=centerline,
        profile=profile,
    )

    layer = corrected.layer
    assert layer.authority.all()
    np.testing.assert_array_equal(layer.water_y[:, 8], 20.0)
    np.testing.assert_array_equal(layer.water_y[8, :], 20.0)
    np.testing.assert_array_equal(layer.water_y[0, 0], 10.0)
    np.testing.assert_array_equal(layer.water_y[4, 4], 10.0)
    assert layer.evidence.maximum_boundary_correction_m == 0.0
    assert layer.evidence.zero_weight_boundary_samples > 0
    assert len(corrected.dependencies) == 9
    assert {source for _, source, _ in corrected.dependencies} == {
        WaterHaloSource.INHERITED,
        WaterHaloSource.CORRECTED,
        WaterHaloSource.ABSENT,
    }

    absent_records = {
        chunk: DecodedWaterHaloChunk(
            chunk,
            WaterHaloSource.ABSENT,
            None,
            hashlib.sha256(f"absent:{chunk}".encode()).hexdigest(),
        )
        for chunk in records
    }
    absent = correct_flowing_water_chunk_halo(
        chunk=target,
        grid=GRID,
        load_baseline=absent_records.__getitem__,
        mapped_water_polygon=mapped,
        qualified_water_polygon=qualified,
        centerline=centerline,
        profile=profile,
    )
    assert np.isnan(absent.layer.water_y).all()
    assert not absent.layer.authority.any()
    assert absent.layer.abstained.any()
    assert absent.dependency_sha256 != corrected.dependency_sha256
    assert (
        absent.layer.evidence.recipe_sha256
        != corrected.layer.evidence.recipe_sha256
    )
