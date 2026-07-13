import numpy as np

from assetgen.config import GridConfig
from assetgen.cook.pinned_height import assemble_lod0_cell_window
from assetgen.grid import ChunkId


def test_cell_window_crosses_east_and_south_boundaries_without_aprons() -> None:
    grid = GridConfig(100, 5000, 2048, 4, (0, 1), 2048)

    def load(chunk: ChunkId) -> np.ndarray:
        origin_e = grid.anchor_e + chunk.cx * grid.chunk_m
        origin_n = grid.anchor_n - chunk.cz * grid.chunk_m
        row, col = np.mgrid[:2049, :2049]
        values = origin_e + col + (origin_n - row) * 10000
        result = values.astype(np.float32)
        # Poison duplicate aprons; assembly must use adjacent cores instead.
        result[-1, :] = -1e20
        result[:, -1] = -1e20
        return result

    e_min = grid.anchor_e + 2046
    n_max = grid.anchor_n - 2046
    actual = assemble_lod0_cell_window(grid, e_min, n_max, 5, 5, load)
    row, col = np.mgrid[:5, :5]
    expected = (e_min + col + (n_max - row) * 10000).astype(np.float32)
    np.testing.assert_array_equal(actual, expected)
