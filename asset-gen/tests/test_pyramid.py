import numpy as np

from assetgen.cook.pyramid import (
    assemble_finer,
    blocks16,
    majority_u8,
    mean_u8,
    weighted_mean_u8,
    wet_majority,
)
from assetgen.grid import ChunkId

RNG = np.random.default_rng(11)


def test_blocks16_groups_4x4_neighborhoods():
    a = np.arange(64, dtype=np.uint8).reshape(8, 8)
    b = blocks16(a)
    assert b.shape == (2, 2, 16)
    # block (0,0) = rows 0..3 x cols 0..3
    assert sorted(b[0, 0]) == sorted(a[:4, :4].ravel())
    assert sorted(b[1, 0]) == sorted(a[4:, :4].ravel())


def test_majority_picks_dominant_class():
    block = np.full((1, 1, 16), 3, dtype=np.uint8)
    block[0, 0, :5] = 1
    assert majority_u8(block)[0, 0] == 3


def test_majority_tie_breaks_to_higher_class():
    block = np.zeros((1, 1, 16), dtype=np.uint8)
    block[0, 0, :8] = 1   # grass-ish
    block[0, 0, 8:] = 13  # sea — must win the 8:8 coastline tie
    assert majority_u8(block)[0, 0] == 13


def test_mean_u8_rounds():
    block = np.zeros((1, 1, 16), dtype=np.uint8)
    block[0, 0, :8] = 255
    assert mean_u8(block)[0, 0] == 128


def test_weighted_mean_ignores_zero_weight_texels():
    h = np.zeros((1, 1, 16), dtype=np.uint8)
    w = np.zeros((1, 1, 16), dtype=np.uint8)
    h[0, 0, :4] = 20  # 20 m trees on the only covered quarter
    w[0, 0, :4] = 255
    assert weighted_mean_u8(h, w)[0, 0] == 20
    assert weighted_mean_u8(h, np.zeros_like(w))[0, 0] == 0


def test_wet_majority_levels_and_dry():
    block = np.full((1, 2, 16), np.nan, dtype=np.float32)
    block[0, 0, :8] = 60.0  # exactly half wet -> tie leans wet
    block[0, 1, :7] = 60.0  # minority wet -> dry
    out = wet_majority(block)
    assert out[0, 0] == 60.0 and np.isnan(out[0, 1])


def _tile(v: int, n: int) -> list[np.ndarray]:
    return [np.full((n + 1, n + 1), v, dtype=np.uint8)]


def test_assemble_finer_fills_pastes_and_replicates():
    n = 4
    present = {(0, 0): 7, (3, 3): 9, (4, 0): 5}  # interior x2 + a real east strip chunk

    def read(c: ChunkId) -> list[np.ndarray] | None:
        assert c.lod == 1
        v = present.get((c.cx, c.cz))
        return _tile(v, n) if v is not None else None

    big = assemble_finer(read, ChunkId(0, 0, 2), n, fills=[0])[0]
    assert big.shape == (4 * n + 4, 4 * n + 4)
    assert (big[:n, :n] == 7).all()          # present interior chunk
    assert (big[n : 2 * n, :n] == 0).all()   # missing interior chunk keeps fill
    assert (big[3 * n : 4 * n, 3 * n : 4 * n] == 9).all()
    assert (big[:n, 4 * n :] == 5).all()     # real east strip chunk pasted
    assert (big[3 * n : 4 * n, 4 * n :] == 9).all()  # missing east strip -> edge-replicated
    assert (big[4 * n :, 3 * n : 4 * n] == 9).all()  # missing south strip -> edge-replicated


def test_assemble_finer_none_when_no_interior_data():
    assert assemble_finer(lambda c: None, ChunkId(0, 0, 1), 4, fills=[0]) is None
