import pytest

from assetgen.config import load_base
from assetgen.grid import (
    ChunkId,
    chunk_bounds_en,
    chunk_id_for_en,
    chunk_raster_window_en,
    chunks_covering_bbox_en,
    en_from_game,
    game_from_en,
    lod_footprint,
    lod_texel,
    snap_bbox_to_chunks_en,
)

GRID = load_base().grid

TAEVASKOJA_E, TAEVASKOJA_N = 679928, 6443064


def test_anchor_is_chunk_aligned():
    assert GRID.anchor_e % GRID.chunk_m == 0
    assert GRID.anchor_n % GRID.chunk_m == 0


def test_game_en_round_trip():
    x, z = game_from_en(GRID, TAEVASKOJA_E, TAEVASKOJA_N)
    assert x > 0 and z > 0, "Estonia must be in the +x/+z quadrant"
    e, n = en_from_game(GRID, x, z)
    assert (e, n) == (TAEVASKOJA_E, TAEVASKOJA_N)


def test_lod_scaling():
    assert lod_texel(GRID, 0) == 1
    assert lod_texel(GRID, 2) == 16
    assert lod_footprint(GRID, 0) == 2048
    assert lod_footprint(GRID, 3) == 2048 * 64


def test_chunk_contains_its_point():
    for lod in GRID.lods:
        c = chunk_id_for_en(GRID, TAEVASKOJA_E, TAEVASKOJA_N, lod)
        e0, n0, e1, n1 = chunk_bounds_en(GRID, c)
        assert e0 <= TAEVASKOJA_E < e1
        assert n0 < TAEVASKOJA_N <= n1
        assert (e1 - e0) == lod_footprint(GRID, lod)


def test_raster_window_has_one_texel_apron():
    c = chunk_id_for_en(GRID, TAEVASKOJA_E, TAEVASKOJA_N, 0)
    e0, n0, e1, n1, t = chunk_raster_window_en(GRID, c)
    b = chunk_bounds_en(GRID, c)
    assert t == 1
    assert (e1 - e0) == GRID.chunk_m + t
    assert (n1 - n0) == GRID.chunk_m + t
    assert e0 == b[0] and n1 == b[3], "window shares the chunk's NW corner"
    # (chunk_res+1) texels of size t fill the window exactly
    assert (e1 - e0) // t == GRID.chunk_res + 1


def test_apron_overlaps_east_neighbor():
    c = ChunkId(100, 200, 0)
    e = ChunkId(101, 200, 0)
    cw = chunk_raster_window_en(GRID, c)
    ew = chunk_raster_window_en(GRID, e)
    # c's last texel column center == e's first texel column center
    t = cw[4]
    c_last_center = cw[0] + (GRID.chunk_res + 0.5) * t
    e_first_center = ew[0] + 0.5 * t
    assert c_last_center == e_first_center


def test_cover_counts():
    # 16x16 km at LOD0 (2048 m chunks): snapped cover is 9x9 when not grid-aligned
    bbox = (TAEVASKOJA_E - 8000, TAEVASKOJA_N - 8000, TAEVASKOJA_E + 8000, TAEVASKOJA_N + 8000)
    ids = chunks_covering_bbox_en(GRID, bbox, 0)
    assert len(ids) in (64, 72, 81)
    assert len(set((c.cx, c.cz) for c in ids)) == len(ids)


def test_cover_exact_alignment_no_spill():
    # a bbox exactly equal to one chunk must return exactly that chunk
    c = ChunkId(50, 60, 1)
    ids = chunks_covering_bbox_en(GRID, chunk_bounds_en(GRID, c), 1)
    assert ids == [c]


def test_snap_bbox():
    bbox = (TAEVASKOJA_E - 100, TAEVASKOJA_N - 100, TAEVASKOJA_E + 100, TAEVASKOJA_N + 100)
    s = snap_bbox_to_chunks_en(GRID, bbox, 0)
    assert s[0] % GRID.chunk_m == GRID.anchor_e % GRID.chunk_m == 0
    assert s[2] - s[0] >= 200 and (s[2] - s[0]) % GRID.chunk_m == 0
    assert s[0] <= bbox[0] and s[1] <= bbox[1] and s[2] >= bbox[2] and s[3] >= bbox[3]
