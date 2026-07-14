import math

import numpy as np
import pytest

from assetgen.terrain.repair import prolong_structural_4x


PHASES = (-3.0 / 8.0, -1.0 / 8.0, 1.0 / 8.0, 3.0 / 8.0)


def _keys_kernel(distance: float) -> float:
    x = abs(distance)
    if x < 1.0:
        return 1.5 * x**3 - 2.5 * x**2 + 1.0
    if x < 2.0:
        return -0.5 * x**3 + 2.5 * x**2 - 4.0 * x + 2.0
    return 0.0


def _scalar_interpolation(source: np.ndarray, row: int, col: int) -> np.ndarray:
    result = np.empty((4, 4), dtype=np.float64)
    for fine_row, row_phase in enumerate(PHASES):
        for fine_col, col_phase in enumerate(PHASES):
            value = 0.0
            for source_row in range(row - 2, row + 3):
                north_weight = _keys_kernel(row + row_phase - source_row)
                for source_col in range(col - 2, col + 3):
                    east_weight = _keys_kernel(col + col_phase - source_col)
                    value += source[source_row, source_col] * north_weight * east_weight
            result[fine_row, fine_col] = value
    return result


def _restore_mean(interpolated: np.ndarray, parent: float) -> np.ndarray:
    u = np.array(((0.5 / 4.0), (1.5 / 4.0), (2.5 / 4.0), (3.5 / 4.0)))
    bubble = u**3 * (1.0 - u) ** 3
    bubble /= bubble.mean()
    return interpolated + (parent - interpolated.mean()) * np.outer(bubble, bubble)


def test_matches_independent_keys_and_bubble_reference() -> None:
    rows, cols = np.mgrid[:9, :10]
    source = 31.0 + rows**2 * 0.13 + cols**3 * 0.017 + rows * cols * 0.021

    actual = prolong_structural_4x(
        source, parent_rows=(3, 6), parent_cols=(2, 7)
    ).reshape(3, 4, 5, 4)

    for output_row, source_row in enumerate(range(3, 6)):
        for output_col, source_col in enumerate(range(2, 7)):
            expected = _restore_mean(
                _scalar_interpolation(source, source_row, source_col),
                source[source_row, source_col],
            )
            np.testing.assert_allclose(
                actual[output_row, :, output_col, :], expected, rtol=0.0, atol=2e-13
            )


def test_preserves_each_parent_mean_and_reproduces_affine_surface() -> None:
    rows, cols = np.mgrid[:11, :12]
    source = 100.0 + 2.5 * rows - 0.75 * cols
    fine = prolong_structural_4x(source, parent_rows=(2, 9), parent_cols=(3, 10))
    blocks = fine.reshape(7, 4, 7, 4)

    np.testing.assert_allclose(
        blocks.mean(axis=(1, 3)), source[2:9, 3:10], rtol=0.0, atol=3e-14
    )
    for parent_row in range(7):
        for row_phase, phase_y in enumerate(PHASES):
            for parent_col in range(7):
                for col_phase, phase_x in enumerate(PHASES):
                    expected = (
                        100.0
                        + 2.5 * (parent_row + 2 + phase_y)
                        - 0.75 * (parent_col + 3 + phase_x)
                    )
                    assert math.isclose(
                        blocks[parent_row, row_phase, parent_col, col_phase],
                        expected,
                        rel_tol=0.0,
                        abs_tol=3e-14,
                    )


def test_tiled_crops_are_identical_to_one_larger_crop() -> None:
    rng = np.random.default_rng(982134)
    source = rng.normal(size=(14, 17))
    complete = prolong_structural_4x(
        source, parent_rows=(3, 11), parent_cols=(4, 13)
    )
    northwest = prolong_structural_4x(
        source, parent_rows=(3, 7), parent_cols=(4, 9)
    )
    northeast = prolong_structural_4x(
        source, parent_rows=(3, 7), parent_cols=(9, 13)
    )
    southwest = prolong_structural_4x(
        source, parent_rows=(7, 11), parent_cols=(4, 9)
    )
    southeast = prolong_structural_4x(
        source, parent_rows=(7, 11), parent_cols=(9, 13)
    )

    tiled = np.block([[northwest, northeast], [southwest, southeast]])
    np.testing.assert_array_equal(tiled, complete)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"parent_rows": (1, 3), "parent_cols": (2, 4)}, "two parent support"),
        ({"parent_rows": (2, 4), "parent_cols": (2, 7)}, "two parent support"),
        ({"parent_rows": (3, 3), "parent_cols": (2, 4)}, "non-empty"),
        ({"parent_rows": (2.5, 4), "parent_cols": (2, 4)}, "integers"),
    ],
)
def test_rejects_invalid_or_unsupported_crop(kwargs: dict, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        prolong_structural_4x(np.zeros((8, 8)), **kwargs)


def test_rejects_nonfinite_values_in_required_support() -> None:
    source = np.zeros((9, 9))
    source[1, 1] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        prolong_structural_4x(source, parent_rows=(3, 6), parent_cols=(3, 6))
