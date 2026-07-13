import pytest

from assetgen.config import load_base
from assetgen.height_geom import (
    HeightChunkId,
    children_of,
    chunk_origin_en_units,
    footprint_m,
    parent_of,
    plan_hero,
    sample_center_en_units,
    texel_m,
)

GRID = load_base().grid


def test_signed_height_physical_geometry():
    assert [texel_m(lod) for lod in (-2, -1, 0, 1, 4)] == [0.0625, 0.25, 1.0, 4.0, 256.0]
    assert [footprint_m(GRID, lod) for lod in (-2, -1, 0)] == [128.0, 512.0, 2048.0]


@pytest.mark.parametrize("lod", [-2, -1, 0, 1, 4])
@pytest.mark.parametrize("cx,cz", [(0, 0), (3, 7), (-1, -1), (-17, 9)])
def test_apron_center_equals_neighbor_core(lod, cx, cz):
    chunk = HeightChunkId(lod, cx, cz)
    east = HeightChunkId(lod, cx + 1, cz)
    south = HeightChunkId(lod, cx, cz + 1)
    corner = HeightChunkId(lod, cx + 1, cz + 1)
    assert sample_center_en_units(GRID, chunk, 2048, 0) == sample_center_en_units(GRID, east, 0, 0)
    assert sample_center_en_units(GRID, chunk, 0, 2048) == sample_center_en_units(GRID, south, 0, 0)
    assert sample_center_en_units(GRID, chunk, 2048, 2048) == sample_center_en_units(GRID, corner, 0, 0)


def test_parent_floor_division_and_children_for_negative_coordinates():
    assert parent_of(HeightChunkId(-2, -1, -1)) == HeightChunkId(-1, -1, -1)
    assert parent_of(HeightChunkId(-2, -4, -4)) == HeightChunkId(-1, -1, -1)
    parent = HeightChunkId(-1, -3, 2)
    children = children_of(parent)
    assert len(children) == 16
    assert all(parent_of(child) == parent for child in children)
    assert children[0] == HeightChunkId(-2, -12, 8)
    assert children[-1] == HeightChunkId(-2, -9, 11)


def test_hero_plan_has_exact_published_and_support_sets():
    plan = plan_hero(-3, 2)
    assert plan.parent == HeightChunkId(-1, -3, 2)
    assert plan.authority_lod0 == HeightChunkId(0, -1, 0)
    assert len(plan.published_fine) == 16
    assert len(plan.transient_support) == 9
    assert not set(plan.published_fine) & set(plan.transient_support)
    assert all(parent_of(child) == plan.parent for child in plan.published_fine)
    assert plan.transient_support[-1] == HeightChunkId(-2, -8, 12)


def test_sample_centers_are_exact_half_texel_units():
    chunk = HeightChunkId(-2, 0, 0)
    origin_e, origin_n = chunk_origin_en_units(GRID, chunk)
    e, n = sample_center_en_units(GRID, chunk, 0, 0)
    assert (e - origin_e, origin_n - n) == (1, 1)  # 1/32 m = half of 0.0625 m


def test_invalid_height_lod_rejected():
    with pytest.raises(ValueError):
        texel_m(-3)
    with pytest.raises(ValueError):
        texel_m(5)
