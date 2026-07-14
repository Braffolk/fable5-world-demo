import hashlib
import zlib

import numpy as np
from shapely.geometry import LineString, box

from assetgen.config import EncodeConfig, GridConfig
from assetgen.cook.chunkio import read_chunk
from assetgen.cook.encode import decode_quant16, undelta2d
from assetgen.grid import ChunkId
from assetgen.terrain.repair.baseline import SampleGrid
from assetgen.terrain.repair.model import FlowOrientation, FlowProfile
from assetgen.terrain.repair.water_layer import (
    correct_flowing_water_layer,
    reduce_water_lod1,
)
from assetgen.terrain.repair.water_pack import (
    AbsentWaterChunk,
    DecodedWaterDependency,
    PackedWaterChunk,
    pack_corrected_water_lod0,
    pack_reduced_water_lod1,
)


GRID = GridConfig(0, 0, 16, 4, (0, 1), 16)
ENCODE = EncodeConfig("deflate", 0.01, 0.01, 19, 1)


def _profile() -> FlowProfile:
    return FlowProfile(
        reach_id="generic-reach",
        epoch_id="qualified-epoch",
        station_m=np.array([4.0, 12.0]),
        observation_y=np.array([19.5, 19.4]),
        accepted=np.array([True, True]),
        water_y=np.array([19.5, 19.4]),
        orientation=FlowOrientation.FORWARD,
        longest_missing_span_m=0.0,
        source_artifact_sha256="1" * 64,
    )


def _corrected(baseline: np.ndarray):
    return correct_flowing_water_layer(
        baseline_water_y=baseline,
        grid=SampleGrid(1.0, -1.0, 2.0, 9, 9),
        mapped_water_polygon=box(0.0, -8.0, 16.0, -2.0),
        qualified_water_polygon=box(4.0, -8.0, 12.0, -2.0),
        centerline=LineString([(0.0, -5.0), (16.0, -5.0)]),
        profile=_profile(),
        collar_samples=1,
    )


def test_lod0_preserves_reserved_dry_and_inherited_unmodified_wet_codes(tmp_path) -> None:
    inherited_offset = 5.0
    wire_scale = float(np.float32(ENCODE.water_qscale))
    inherited_codes = np.zeros((9, 9), dtype=np.uint16)
    inherited_codes[1:4] = 1500 + np.arange(9, dtype=np.uint16)
    baseline = inherited_codes.astype(np.float32) * wire_scale + inherited_offset
    baseline[inherited_codes == 0] = np.nan
    corrected = _corrected(baseline)
    chunk = ChunkId(0, 0, 0)

    packed = pack_corrected_water_lod0(
        corrected,
        chunk=chunk,
        grid=GRID,
        encode=ENCODE,
        inherited_artifact_sha256="a" * 64,
        inherited_qoffset=inherited_offset,
        inherited_qscale=wire_scale,
    )
    assert isinstance(packed, PackedWaterChunk)
    assert packed.meta.layer == "water"
    assert packed.meta.enc == 1
    assert packed.meta.qoffset == inherited_offset
    assert packed.meta.qscale == wire_scale
    assert (packed.meta.origin_e, packed.meta.origin_n) == (0.0, 0.0)

    delta = np.frombuffer(zlib.decompress(packed.payload), dtype=np.uint16).reshape(9, 9)
    packed_codes = undelta2d(delta)
    np.testing.assert_array_equal(packed_codes[~np.isfinite(corrected.water_y)], 0)
    inherited_unmodified = np.isfinite(baseline) & ~corrected.authority
    np.testing.assert_array_equal(
        packed_codes[inherited_unmodified], inherited_codes[inherited_unmodified]
    )
    decoded = decode_quant16(
        ENCODE, packed.payload, packed.meta.res, packed.meta.qoffset, packed.meta.qscale
    )
    assert packed.max_roundtrip_error_m <= packed.roundtrip_limit_m
    assert np.isfinite(decoded[packed_codes != 0]).all()

    path = tmp_path / "water.lac"
    packed.write_lac1(path)
    meta, payload = read_chunk(path)
    assert meta == packed.meta
    assert payload == packed.payload
    assert hashlib.sha256(path.read_bytes()).hexdigest() == packed.artifact_sha256

    absent = pack_corrected_water_lod0(
        _corrected(np.full((9, 9), np.nan, dtype=np.float32)),
        chunk=chunk,
        grid=GRID,
        encode=ENCODE,
        inherited_artifact_sha256="b" * 64,
        inherited_qoffset=None,
        inherited_qscale=None,
    )
    assert isinstance(absent, AbsentWaterChunk)
    assert absent.reason == "all_dry"
    assert absent.dry_samples == 81


def test_lod1_is_fixed_decoded_child_reduction_with_bound_dependencies() -> None:
    parent = ChunkId(0, 0, 1)
    dependencies: dict[ChunkId, DecodedWaterDependency] = {}
    for dz in range(5):
        for dx in range(5):
            chunk = ChunkId(dx, dz, 0)
            row, col = np.mgrid[:9, :9]
            water = (30.0 + dz + dx * 0.1 + row * 0.01 + col * 0.001).astype(
                np.float32
            )
            water[(row + col + dx + dz) % 5 == 0] = np.nan
            digest = hashlib.sha256(f"decoded-water:{chunk}".encode()).hexdigest()
            dependencies[chunk] = DecodedWaterDependency(water, digest)

    packed = pack_reduced_water_lod1(
        parent=parent,
        grid=GRID,
        encode=ENCODE,
        load_dependency=dependencies.__getitem__,
        evidence_sha256="c" * 64,
    )
    assert isinstance(packed, PackedWaterChunk)
    expected = reduce_water_lod1(
        parent,
        lambda chunk: dependencies[chunk].water_y,
        core_res=8,
    )
    assert expected is not None
    assert packed.source_values_sha256 == hashlib.sha256(
        np.ascontiguousarray(expected.water_y, dtype="<f8").tobytes()
    ).hexdigest()
    assert len(packed.dependencies) == 25
    assert dict(packed.dependencies) == {
        chunk: dependency.artifact_sha256
        for chunk, dependency in dependencies.items()
    }
    repeated = pack_reduced_water_lod1(
        parent=parent,
        grid=GRID,
        encode=ENCODE,
        load_dependency=dependencies.__getitem__,
        evidence_sha256="c" * 64,
    )
    assert isinstance(repeated, PackedWaterChunk)
    assert repeated.artifact_sha256 == packed.artifact_sha256
    assert repeated.dependency_sha256 == packed.dependency_sha256
