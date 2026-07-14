import hashlib
from pathlib import Path

import numpy as np

from assetgen.config import EncodeConfig, GridConfig
from assetgen.cook.chunkio import read_chunk_v2
from assetgen.cook.encode import decode_quant16, encode_quant16_checked
from assetgen.height_geom import HeightChunkId
from assetgen.terrain.repair.baseline import BaselineTile
from assetgen.terrain.repair.hierarchy import (
    AuthorityTileInput,
    BaselineTileInput,
    FineReevaluationResult,
    FineSurfaceWindow,
    HierarchyLayout,
    _raster_support,
    assemble_fine_surface_window,
    build_structural_hierarchy,
    reduce_decoded_children,
)
from assetgen.terrain.repair.model import StructuralTile
from assetgen.terrain.repair.plan import plan_structural_repair


GRID = GridConfig(368640, 6635520, 2048, 4, (0, 1, 2, 3, 4), 2048)
ENCODE = EncodeConfig("deflate", 0.01, 0.01, 19, 1)


def test_structural_hierarchy_is_seamless_and_decoded_child_derived(
    tmp_path: Path,
) -> None:
    layout = HierarchyLayout(authority_core_res=8)
    authority = HeightChunkId(0, 0, 0)
    review = HeightChunkId(-1, 1, 1)
    plan = plan_structural_repair(authority, review)

    def baseline_values(chunk: HeightChunkId) -> np.ndarray:
        row, col = np.mgrid[:8, :8]
        global_row = chunk.cz * 8 + row
        global_col = chunk.cx * 8 + col
        return 42.0 + global_col * 0.017 + global_row * 0.031

    def load(chunk: HeightChunkId) -> AuthorityTileInput:
        height = baseline_values(chunk)
        mask = np.ones_like(height, dtype=bool)
        tile = StructuralTile(height, mask, ~mask, ~mask)
        digest = hashlib.sha256(f"{chunk}".encode()).hexdigest()
        return AuthorityTileInput(chunk, tile, digest)

    def load_baseline(chunk: HeightChunkId) -> BaselineTileInput:
        height = baseline_values(chunk)
        tile = BaselineTile(height, np.ones_like(height, dtype=bool))
        digest = hashlib.sha256(f"baseline:{chunk}".encode()).hexdigest()
        return BaselineTileInput(chunk, tile, digest)

    def preserve_baseline(context: FineSurfaceWindow) -> FineReevaluationResult:
        baseline = context.baseline_height[context.core_slice]
        shape = baseline.shape
        return FineReevaluationResult(
            baseline,
            np.zeros(shape, dtype=bool),
            np.zeros(shape, dtype=bool),
        )

    hierarchy = build_structural_hierarchy(
        grid=GRID,
        encode=ENCODE,
        plan=plan,
        load_authority=load,
        load_baseline=load_baseline,
        reevaluate_fine_surface=preserve_baseline,
        layout=layout,
    )
    assert len(hierarchy.fine) == 25
    assert hierarchy.metrics.max_decoded_seam_m == 0.0
    assert hierarchy.metrics.max_fine_roundtrip_error_m <= 0.00101
    assert hierarchy.metrics.fine_reevaluation_applied
    assert hierarchy.metrics.max_post_reevaluation_unowned_delta_m == 0.0
    assert len(hierarchy.metrics.parent_dependencies) == 25
    assert dict(hierarchy.metrics.parent_dependencies) == {
        item.chunk: item.decoded_values_sha256 for item in hierarchy.fine
    }
    assert hierarchy.metrics.authority_dependencies
    assert hierarchy.metrics.baseline_dependencies
    assert all(item.meta.qscale == np.float32(0.002) for item in hierarchy.fine)
    assert len({item.meta.qoffset for item in hierarchy.fine}) == 1
    expected_authority_min = min(
        float(np.min(baseline_values(item.chunk))) for item in hierarchy.fine
    )
    assert hierarchy.metrics.fine_domain_authority_min_m == expected_authority_min
    assert hierarchy.metrics.fine_shared_qoffset == np.floor(
        expected_authority_min - 2.0
    )

    packed = {item.chunk: item for item in hierarchy.fine}

    def decoded(chunk: HeightChunkId) -> np.ndarray:
        item = packed[chunk]
        return decode_quant16(
            ENCODE, item.payload, item.meta.res, item.meta.qoffset, item.meta.qscale
        )

    derived = reduce_decoded_children(
        review, decoded, core_res=layout.fine_core_res
    ).values
    parent = hierarchy.parent
    expected_payload, expected_offset, expected_scale = encode_quant16_checked(
        ENCODE, derived, 0.005
    )
    assert parent.payload == expected_payload
    assert parent.meta.qoffset == expected_offset
    assert parent.meta.qscale == expected_scale
    parent_path = tmp_path / "parent.lac"
    parent.write_lac2(parent_path)
    written_meta, written_payload = read_chunk_v2(parent_path)
    assert written_meta == parent.meta
    assert written_payload == parent.payload
    parent_decoded = decode_quant16(
        ENCODE,
        parent.payload,
        parent.meta.res,
        parent.meta.qoffset,
        parent.meta.qscale,
    )
    np.testing.assert_allclose(parent_decoded, derived, atol=parent.roundtrip_limit_m)


def test_real_parent_support_coordinates_and_adjacent_fine_overlap() -> None:
    layout = HierarchyLayout(authority_core_res=4)

    def height(chunk: HeightChunkId) -> np.ndarray:
        row, col = np.mgrid[:4, :4]
        global_row = chunk.cz * 4 + row
        global_col = chunk.cx * 4 + col
        return 60.0 + global_row * 0.013 + global_col * 0.029

    support = _raster_support(HeightChunkId(-2, 0, 0), height, layout)
    assert support.shape == (17, 17)
    assert support[0, 0] == 60.0 + (-6) * 0.013 + (-6) * 0.029
    assert support[-1, -1] == 60.0 + 10 * 0.013 + 10 * 0.029

    first = assemble_fine_surface_window(
        HeightChunkId(-2, 0, 0),
        load_structural_height=height,
        load_baseline_height=height,
        layout=layout,
    )
    east = assemble_fine_surface_window(
        HeightChunkId(-2, 1, 0),
        load_structural_height=height,
        load_baseline_height=height,
        layout=layout,
    )
    assert first.structural_height.shape == (49, 49)
    first_core = first.structural_height[first.core_slice]
    east_core = east.structural_height[east.core_slice]
    np.testing.assert_array_equal(first_core[:, -1], east_core[:, 0])


def test_independent_baseline_removes_dry_and_abstained_bed_ringing() -> None:
    layout = HierarchyLayout(authority_core_res=8)
    authority = HeightChunkId(0, 0, 0)
    review = HeightChunkId(-1, 1, 1)
    plan = plan_structural_repair(authority, review)
    supported_start = 40
    supported_stop = 42
    abstained_stop = 44
    baseline_revision = ["v1"]

    def load_baseline(chunk: HeightChunkId) -> BaselineTileInput:
        height = np.full((8, 8), 100.0, dtype=np.float64)
        tile = BaselineTile(height, np.ones_like(height, dtype=bool))
        digest = hashlib.sha256(
            f"baseline-ring:{baseline_revision[0]}:{chunk}".encode()
        ).hexdigest()
        return BaselineTileInput(chunk, tile, digest)

    def load_structural(chunk: HeightChunkId) -> AuthorityTileInput:
        _, col = np.mgrid[:8, :8]
        global_col = chunk.cx * 8 + col
        supported = (global_col >= supported_start) & (global_col < supported_stop)
        abstained = (global_col >= supported_stop) & (global_col < abstained_stop)
        supported = np.broadcast_to(supported, (8, 8)).copy()
        abstained = np.broadcast_to(abstained, (8, 8)).copy()
        height = np.full((8, 8), 100.0, dtype=np.float64)
        height[supported] = 93.0
        valid = np.ones_like(height, dtype=bool)
        tile = StructuralTile(height, valid, supported, supported | abstained)
        digest = hashlib.sha256(f"structural-ring:{chunk}".encode()).hexdigest()
        return AuthorityTileInput(chunk, tile, digest)

    observed_pre_restore: list[float] = []

    def restore_unowned(context: FineSurfaceWindow) -> FineReevaluationResult:
        fine_col = context.chunk.cx * layout.fine_core_res + np.arange(
            layout.fine_core_res + 1
        )
        authority_1d = (fine_col >= supported_start * 4) & (
            fine_col < supported_stop * 4
        )
        abstained_1d = (fine_col >= supported_stop * 4) & (
            fine_col < abstained_stop * 4
        )
        structural = context.structural_height[context.core_slice]
        baseline = context.baseline_height[context.core_slice]
        authority_mask = np.broadcast_to(authority_1d, structural.shape).copy()
        abstained_mask = np.broadcast_to(abstained_1d, structural.shape).copy()
        observed_pre_restore.append(
            float(
                np.max(
                    np.abs(
                            structural[~authority_mask]
                            - baseline[~authority_mask]
                    ),
                    initial=0.0,
                )
            )
        )
        height = structural.copy()
        height[~authority_mask] = baseline[~authority_mask]
        return FineReevaluationResult(height, authority_mask, abstained_mask)

    def build():
        return build_structural_hierarchy(
            grid=GRID,
            encode=ENCODE,
            plan=plan,
            load_authority=load_structural,
            load_baseline=load_baseline,
            reevaluate_fine_surface=restore_unowned,
            layout=layout,
        )

    first = build()
    second = build()
    assert max(observed_pre_restore) > 0.1
    assert first.metrics.max_pre_reevaluation_unowned_delta_m > 0.1
    assert first.metrics.max_post_reevaluation_unowned_delta_m == 0.0
    assert first.metrics.fine_authority_samples > 0
    assert first.metrics.fine_abstained_samples > 0
    assert first.metrics.baseline_dependency_merkle_root == (
        second.metrics.baseline_dependency_merkle_root
    )
    assert [item.dependency_merkle_root for item in first.fine] == [
        item.dependency_merkle_root for item in second.fine
    ]
    baseline_revision[0] = "v2"
    rebound = build()
    assert first.metrics.baseline_dependency_merkle_root != (
        rebound.metrics.baseline_dependency_merkle_root
    )
    assert [item.dependency_merkle_root for item in first.fine] != [
        item.dependency_merkle_root for item in rebound.fine
    ]
    assert [item.source_values_sha256 for item in first.fine] == [
        item.source_values_sha256 for item in rebound.fine
    ]

    for item in first.fine:
        decoded = decode_quant16(
            ENCODE,
            item.payload,
            item.meta.res,
            item.meta.qoffset,
            item.meta.qscale,
        )
        fine_col = item.chunk.cx * layout.fine_core_res + np.arange(decoded.shape[1])
        owned = (fine_col >= supported_start * 4) & (fine_col < supported_stop * 4)
        unowned = np.broadcast_to(~owned, decoded.shape)
        np.testing.assert_array_equal(decoded[unowned], np.float32(100.0))
