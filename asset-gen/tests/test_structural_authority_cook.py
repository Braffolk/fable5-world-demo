import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

from assetgen.config import GridConfig
from assetgen.height_geom import HeightChunkId
from assetgen.terrain.repair.baseline import BaselineTile
from assetgen.terrain.repair.closure import close_mixed_water_authority
from assetgen.terrain.repair import cook as authority_cook
from assetgen.terrain.repair.cook import (
    AUTHORITY_EVIDENCE_BINDING_VERSION,
    AuthorityEvidenceBinding,
    cook_structural_authority,
)
from assetgen.terrain.repair.model import FlowOrientation, FlowProfile, WaterSamples
from assetgen.terrain.repair.pinned_baseline import (
    PINNED_BASELINE_VERSION,
    PinnedBaselineReleaseIdentity,
    PinnedHeightDependency,
    ReconstructedBaselineTile,
)
from assetgen.terrain.repair.storage import (
    decode_baseline_tile,
    decode_structural_tile,
    encode_baseline_tile,
    load_baseline_tile_input,
    write_baseline_tile,
)


GRID = GridConfig(
    anchor_e=0,
    anchor_n=0,
    chunk_m=2048,
    lod_step=4,
    lods=(0, 1, 2, 3, 4),
    chunk_res=2048,
)
CHUNK = HeightChunkId(-2, 0, 0)
EVIDENCE_SHA256 = "ab" * 32
CAMPAIGN_CONTENT_SHA256 = "cd" * 32
EVIDENCE_BINDING = AuthorityEvidenceBinding(
    profile_qualification_sha256=EVIDENCE_SHA256,
    campaign_content_sha256=CAMPAIGN_CONTENT_SHA256,
)


class _PinnedPlanarBaseline:
    release_identity = PinnedBaselineReleaseIdentity(
        manifest_sha256="11" * 32,
        height_index_relative_path="index/height.bin",
        height_index_bytes=21,
        height_index_sha256="22" * 32,
    )

    def __init__(self, offset: float = 100.0) -> None:
        self.offset = offset

    def reconstruct(
        self, chunk: HeightChunkId, *, halo_samples: int = 0
    ) -> ReconstructedBaselineTile:
        side = 512 + 2 * halo_samples
        west = chunk.cx * 128.0 - halo_samples * 0.25
        north = -chunk.cz * 128.0 + halo_samples * 0.25
        east = west + 0.125 + np.arange(side, dtype=np.float64) * 0.25
        northing = north - 0.125 - np.arange(side, dtype=np.float64) * 0.25
        height = self.offset + 0.01 * east[None, :] + 0.02 * northing[:, None]
        canonical = np.ascontiguousarray(height, dtype="<f4")
        digest = hashlib.sha256(canonical.tobytes()).hexdigest()
        dependency = PinnedHeightDependency(
            chunk=HeightChunkId(0, 0, 0),
            content_relative_path="c/height/0/0_0.fake.bin",
            bytes=123,
            sha256="33" * 32,
            index_hash64=0x3333333333333333,
            decoded_sha256="44" * 32,
            qoffset=0.0,
            qscale=0.01,
            flags=1,
        )
        return ReconstructedBaselineTile(
            chunk=chunk,
            tile=BaselineTile(height, np.ones(height.shape, dtype=bool)),
            manifest_sha256=self.release_identity.manifest_sha256,
            dependencies=(dependency,),
            dependency_root_sha256="55" * 32,
            authority_sha256=digest,
            maximum_mean_error_m=0.0,
            mean_error_limit_m=1e-12,
            reconstruction_version=PINNED_BASELINE_VERSION,
        )


PINNED_BASELINE = _PinnedPlanarBaseline()


def _write_planar_dtm(path: Path) -> None:
    east = -0.5 + np.arange(130, dtype=np.float64)
    north = 0.5 - np.arange(130, dtype=np.float64)
    height = 100.0 + 0.01 * east[None, :] + 0.02 * north[:, None]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=130,
        height=130,
        count=1,
        dtype="float32",
        crs="EPSG:3301",
        transform=from_origin(-1.0, 1.0, 1.0, 1.0),
    ) as dataset:
        dataset.write(height.astype(np.float32), 1)


def _profile(start: float = 0.0, end: float = 112.0) -> FlowProfile:
    stations = np.array([start, end], dtype=np.float64)
    water = np.array([90.0, 89.0], dtype=np.float64)
    return FlowProfile(
        reach_id="test-reach",
        epoch_id="test-epoch",
        station_m=stations,
        observation_y=water,
        accepted=np.ones(2, dtype=bool),
        water_y=water,
        orientation=FlowOrientation.FORWARD,
        longest_missing_span_m=0.0,
        source_artifact_sha256=EVIDENCE_SHA256,
    )


def test_cooks_real_dtm_dry_exactly_and_wet_as_forbidden(tmp_path: Path) -> None:
    dtm = tmp_path / "dtm.tif"
    _write_planar_dtm(dtm)
    output = tmp_path / "authority"
    polygon = box(20.0, -120.0, 30.0, -8.0)
    centerline = LineString([(25.0, -120.0), (25.0, -8.0)])

    first = cook_structural_authority(
        grid=GRID,
        dtm_sources=[dtm],
        mapped_water_polygon=polygon,
        qualified_water_polygon=polygon,
        centerline=centerline,
        profile=_profile(),
        requested_tiles=[CHUNK],
        evidence_binding=EVIDENCE_BINDING,
        output_root=output,
        baseline_source=PINNED_BASELINE,
    )
    second = cook_structural_authority(
        grid=GRID,
        dtm_sources=[dtm],
        mapped_water_polygon=polygon,
        qualified_water_polygon=polygon,
        centerline=centerline,
        profile=_profile(),
        requested_tiles=[CHUNK],
        evidence_binding=EVIDENCE_BINDING,
        output_root=output,
        baseline_source=PINNED_BASELINE,
    )

    assert first.recipe_sha256 == second.recipe_sha256
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.artifacts == second.artifacts
    manifest = json.loads(first.manifest_path.read_text())
    assert manifest["format"] == 2
    assert manifest["morphology"] == "absent"
    assert manifest["tileAlignmentLod"] == -2
    assert manifest["tileTexelMeters"] == 0.25
    assert manifest["evidenceSha256"] == CAMPAIGN_CONTENT_SHA256
    assert manifest["evidenceBinding"] == {
        "version": AUTHORITY_EVIDENCE_BINDING_VERSION,
        "profileQualificationSha256": EVIDENCE_SHA256,
        "campaignContentSha256": CAMPAIGN_CONTENT_SHA256,
    }
    assert manifest["baselineAuthority"]["release"] == {
        "manifestSha256": "11" * 32,
        "heightIndexPath": "index/height.bin",
        "heightIndexBytes": 21,
        "heightIndexSha256": "22" * 32,
        "reconstructionVersion": PINNED_BASELINE_VERSION,
    }
    baseline_identity = manifest["baselineAuthority"]["tiles"][0]
    assert baseline_identity["key"] == [-2, 0, 0]
    assert baseline_identity["dependencies"][0]["decoded_sha256"] == "44" * 32
    manifest_tile = manifest["tiles"][0]
    assert manifest_tile["baselinePath"] == first.artifacts[0].baseline_relative_path
    assert manifest_tile["baselineBytes"] == first.artifacts[0].baseline_bytes
    assert manifest_tile["baselineSha256"] == first.artifacts[0].baseline_sha256

    artifact_path = output / first.artifacts[0].relative_path
    tile = decode_structural_tile(artifact_path.read_bytes())
    baseline_input = load_baseline_tile_input(output, first.artifacts[0])
    baseline_tile = baseline_input.tile
    assert baseline_input.chunk == CHUNK
    assert baseline_input.artifact_sha256 == first.artifacts[0].baseline_sha256
    east = 0.125 + np.arange(512, dtype=np.float64) * 0.25
    north = -0.125 - np.arange(512, dtype=np.float64) * 0.25
    wet = (
        (east[None, :] >= 20.0)
        & (east[None, :] <= 30.0)
        & (north[:, None] >= -120.0)
        & (north[:, None] <= -8.0)
    )
    expected_dry = 100.0 + 0.01 * east[None, :] + 0.02 * north[:, None]

    np.testing.assert_array_equal(tile.forbidden_morphology, wet)
    np.testing.assert_array_equal(tile.unknown_bathymetry, wet)
    assert tile.valid.all()
    assert baseline_tile.valid.all()
    np.testing.assert_allclose(
        baseline_tile.height, expected_dry, rtol=0.0, atol=8e-6
    )
    np.testing.assert_array_equal(
        tile.height[~tile.unknown_bathymetry],
        baseline_tile.height[~tile.unknown_bathymetry],
    )
    np.testing.assert_allclose(
        tile.height[~wet], expected_dry[~wet], rtol=0.0, atol=8e-6
    )
    assert np.max(tile.height[wet]) <= 90.0
    assert not np.array_equal(tile.height[wet], expected_dry[wet])


def test_baseline_storage_is_deterministic_and_preserves_invalid_nan(
    tmp_path: Path,
) -> None:
    row, col = np.mgrid[:512, :512]
    height = 17.0 + row * 0.001 + col * 0.002
    valid = np.ones((512, 512), dtype=bool)
    valid[7, 11] = False
    height[~valid] = np.nan
    tile = BaselineTile(height, valid)

    first_payload = encode_baseline_tile(tile)
    second_payload = encode_baseline_tile(tile)
    assert first_payload == second_payload
    decoded = decode_baseline_tile(first_payload)
    np.testing.assert_array_equal(decoded.valid, valid)
    assert np.isnan(decoded.height[7, 11])
    np.testing.assert_allclose(decoded.height[valid], height[valid], rtol=0.0, atol=3e-6)

    artifact = write_baseline_tile(tmp_path, CHUNK, tile)
    loaded = load_baseline_tile_input(tmp_path, artifact)
    assert loaded.chunk == CHUNK
    assert loaded.artifact_sha256 == artifact.sha256
    np.testing.assert_array_equal(loaded.tile.valid, valid)
    np.testing.assert_array_equal(loaded.tile.height, decoded.height)


def test_disjoint_mapped_water_uses_exact_dry_tile_fast_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dtm = tmp_path / "dtm.tif"
    _write_planar_dtm(dtm)
    polygon = box(1_000.0, -120.0, 1_010.0, -8.0)
    centerline = LineString([(1_005.0, -120.0), (1_005.0, -8.0)])

    def unexpected_points(*_args, **_kwargs):
        raise AssertionError("disjoint dry tile constructed a Shapely point grid")

    monkeypatch.setattr("assetgen.terrain.repair.cook.shapely.points", unexpected_points)
    pinned = _PinnedPlanarBaseline(offset=500.0)
    result = cook_structural_authority(
        grid=GRID,
        dtm_sources=[dtm],
        mapped_water_polygon=polygon,
        qualified_water_polygon=polygon,
        centerline=centerline,
        profile=_profile(),
        requested_tiles=[CHUNK],
        evidence_binding=EVIDENCE_BINDING,
        output_root=tmp_path / "dry-authority",
        baseline_source=pinned,
    )
    artifact = result.artifacts[0]
    structural = decode_structural_tile(
        (tmp_path / "dry-authority" / artifact.relative_path).read_bytes()
    )
    baseline = load_baseline_tile_input(tmp_path / "dry-authority", artifact).tile
    assert not structural.unknown_bathymetry.any()
    assert not structural.forbidden_morphology.any()
    np.testing.assert_array_equal(structural.height, baseline.height)
    np.testing.assert_array_equal(structural.valid, baseline.valid)
    assert float(np.min(baseline.height)) > 490.0


def test_unsupported_wet_samples_preserve_baseline_and_forbid_morphology(tmp_path: Path) -> None:
    dtm = tmp_path / "dtm.tif"
    _write_planar_dtm(dtm)
    output = tmp_path / "authority"

    mapped = box(20.0, -120.0, 30.0, -8.0)
    qualified = box(20.0, -98.0, 30.0, -28.0)
    result = cook_structural_authority(
        grid=GRID,
        dtm_sources=[dtm],
        mapped_water_polygon=mapped,
        qualified_water_polygon=qualified,
        centerline=LineString([(25.0, -98.0), (25.0, -28.0)]),
        profile=_profile(0.0, 70.0),
        requested_tiles=[CHUNK],
        evidence_binding=EVIDENCE_BINDING,
        output_root=output,
        baseline_source=PINNED_BASELINE,
    )
    tile = decode_structural_tile((output / result.artifacts[0].relative_path).read_bytes())
    east = 0.125 + np.arange(512, dtype=np.float64) * 0.25
    north = -0.125 - np.arange(512, dtype=np.float64) * 0.25
    mapped_wet = (
        (east[None, :] >= 20.0)
        & (east[None, :] <= 30.0)
        & (north[:, None] >= -120.0)
        & (north[:, None] <= -8.0)
    )
    qualified_wet = mapped_wet & (north[:, None] >= -98.0) & (north[:, None] <= -28.0)
    abstained = mapped_wet & ~qualified_wet
    baseline = 100.0 + 0.01 * east[None, :] + 0.02 * north[:, None]
    np.testing.assert_array_equal(tile.forbidden_morphology, mapped_wet)
    assert not tile.unknown_bathymetry.any()
    np.testing.assert_allclose(
        tile.height[abstained], baseline[abstained], rtol=0.0, atol=8e-6
    )
    np.testing.assert_allclose(
        tile.height[qualified_wet], baseline[qualified_wet], rtol=0.0, atol=8e-6
    )


def test_large_abstained_discontinuity_has_exact_zero_correction_boundary() -> None:
    rows, cols = 48, 160
    wet = np.ones((rows, cols), dtype=bool)
    supported = np.zeros((rows, cols), dtype=bool)
    supported[:, :128] = True
    baseline_height = np.full((rows, cols), 9.8, dtype=np.float64)
    baseline_height[:, 128:] = 1_000.0
    baseline_height[:, :120] = 12.0
    water_y = np.full((rows, cols), np.nan, dtype=np.float64)
    bed_y = np.full((rows, cols), np.nan, dtype=np.float64)
    water_y[supported] = 10.0
    bed_y[supported] = 9.9

    closure = close_mixed_water_authority(
        BaselineTile(baseline_height, np.ones((rows, cols), dtype=bool)),
        WaterSamples(
            wet.ravel(),
            supported.ravel(),
            water_y.ravel(),
            bed_y.ravel(),
        ),
    )
    tile = closure.tile

    np.testing.assert_array_equal(tile.height[:, 128:], baseline_height[:, 128:])
    np.testing.assert_array_equal(tile.height[:, 127], baseline_height[:, 127])
    np.testing.assert_array_equal(tile.height[:, :120], bed_y[:, :120])
    np.testing.assert_array_equal(tile.unknown_bathymetry, supported)
    assert np.all(tile.height[supported] <= bed_y[supported] + 1e-12)
    np.testing.assert_array_equal(
        tile.forbidden_morphology & ~tile.unknown_bathymetry,
        wet & ~supported,
    )


def test_interrupted_pair_cook_resumes_without_rebuilding_and_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dtm = tmp_path / "dtm.tif"
    _write_planar_dtm(dtm)
    output = tmp_path / "resumable"
    chunks = (CHUNK, HeightChunkId(-2, 1, 0))
    mapped = box(20.0, -120.0, 30.0, -8.0)
    centerline = LineString([(25.0, -120.0), (25.0, -8.0)])
    original = authority_cook._build_tile
    calls: list[HeightChunkId] = []

    def interrupt_second(**kwargs):
        calls.append(kwargs["chunk"])
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return original(**kwargs)

    monkeypatch.setattr(authority_cook, "_build_tile", interrupt_second)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        cook_structural_authority(
            grid=GRID,
            dtm_sources=[dtm],
            mapped_water_polygon=mapped,
            qualified_water_polygon=mapped,
            centerline=centerline,
            profile=_profile(),
            requested_tiles=chunks,
            evidence_binding=EVIDENCE_BINDING,
            output_root=output,
            baseline_source=PINNED_BASELINE,
        )
    assert not (output / "manifest.json").exists()
    completed = output / "completion" / "0_0.json"
    assert completed.is_file()

    resumed_calls: list[HeightChunkId] = []

    def track_resume(**kwargs):
        resumed_calls.append(kwargs["chunk"])
        return original(**kwargs)

    monkeypatch.setattr(authority_cook, "_build_tile", track_resume)
    result = cook_structural_authority(
        grid=GRID,
        dtm_sources=[dtm],
        mapped_water_polygon=mapped,
        qualified_water_polygon=mapped,
        centerline=centerline,
        profile=_profile(),
        requested_tiles=chunks,
        evidence_binding=EVIDENCE_BINDING,
        output_root=output,
        baseline_source=PINNED_BASELINE,
    )
    assert resumed_calls == [HeightChunkId(-2, 1, 0)]
    assert len(result.artifacts) == 2

    first_artifact = output / result.artifacts[0].relative_path
    first_artifact.write_bytes(first_artifact.read_bytes() + b"tampered")
    resumed_calls.clear()
    with pytest.raises(ValueError, match="structural artifact differs"):
        cook_structural_authority(
            grid=GRID,
            dtm_sources=[dtm],
            mapped_water_polygon=mapped,
            qualified_water_polygon=mapped,
            centerline=centerline,
            profile=_profile(),
            requested_tiles=chunks,
            evidence_binding=EVIDENCE_BINDING,
            output_root=output,
            baseline_source=PINNED_BASELINE,
        )
    assert resumed_calls == []
