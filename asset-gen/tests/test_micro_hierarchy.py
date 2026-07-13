from pathlib import Path

import numpy as np
import pytest

from assetgen.cook.micro_hierarchy import (
    assemble_parent_source_memmap,
    box_mean4_striped,
    dependency_merkle_root,
)
from assetgen.height_geom import HeightChunkId, HeroCoverage, children_of


def _coverage() -> HeroCoverage:
    parent = HeightChunkId(-1, -2, 3)
    fine_cx0, fine_cz0 = parent.cx * 4, parent.cz * 4
    support = tuple(
        [HeightChunkId(-2, fine_cx0 + 4, fine_cz0 + dz) for dz in range(4)]
        + [HeightChunkId(-2, fine_cx0 + dx, fine_cz0 + 4) for dx in range(4)]
        + [HeightChunkId(-2, fine_cx0 + 4, fine_cz0 + 4)]
    )
    return HeroCoverage(parent, children_of(parent), support, HeightChunkId(0, -1, 0))


def test_assemble_and_reduce_uses_all_support_samples(tmp_path: Path) -> None:
    coverage = _coverage()
    core_res, factor = 8, 4

    def load(chunk: HeightChunkId) -> np.ndarray:
        # A global affine field catches swaps, gaps, and use of duplicate aprons.
        x0 = (chunk.cx - coverage.parent.cx * 4) * core_res
        z0 = (chunk.cz - coverage.parent.cz * 4) * core_res
        z, x = np.mgrid[: core_res + 1, : core_res + 1]
        return (1000.0 + (x0 + x) * 3.0 + (z0 + z) * 11.0).astype(np.float32)

    mosaic = assemble_parent_source_memmap(
        tmp_path / "parent.f32", coverage, load, core_res=core_res, factor=factor
    )
    side = 4 * core_res + factor
    z, x = np.mgrid[:side, :side]
    expected = (1000.0 + x * 3.0 + z * 11.0).astype(np.float32)
    np.testing.assert_array_equal(mosaic, expected)
    parent = box_mean4_striped(mosaic, factor=factor, stripe_rows=3)
    np.testing.assert_array_equal(
        parent,
        expected.reshape(side // factor, factor, side // factor, factor).mean(axis=(1, 3)),
    )


def test_assembly_rejects_missing_or_unexpected_support(tmp_path: Path) -> None:
    coverage = _coverage()
    bad = HeroCoverage(
        coverage.parent,
        coverage.published_fine,
        coverage.transient_support[:-1],
        coverage.authority_lod0,
    )
    with pytest.raises(ValueError, match="invalid transient support"):
        assemble_parent_source_memmap(
            tmp_path / "bad.f32",
            bad,
            lambda _: np.zeros((9, 9), dtype=np.float32),
            core_res=8,
        )


def test_dependency_merkle_is_order_independent_and_key_bound() -> None:
    a = (HeightChunkId(-2, 1, 2), "01" * 32)
    b = (HeightChunkId(-2, 2, 2), "02" * 32)
    assert dependency_merkle_root([a, b]) == dependency_merkle_root([b, a])
    assert dependency_merkle_root([a, b]) != dependency_merkle_root(
        [a, (HeightChunkId(-2, 3, 2), b[1])]
    )
    with pytest.raises(ValueError, match="duplicate"):
        dependency_merkle_root([a, a])
    with pytest.raises(ValueError, match="SHA-256"):
        dependency_merkle_root([(a[0], "00")])
