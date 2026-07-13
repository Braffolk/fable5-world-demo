import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from assetgen.pilots.taevaskoda_structural import _close_coverage_sliver


def test_coverage_sliver_closure_is_ulp_bounded_and_fails_closed() -> None:
    x, y = 679544.0, 6445025.0
    line = LineString([(x, y), (x + 10.0, y)])
    epsilon = 2.0 * np.spacing(y)
    slivered = Polygon(
        [
            (x + epsilon, y),
            (x, y - 2.0),
            (x + 10.0, y - 2.0),
            (x + 10.0, y + 2.0),
            (x, y + 2.0),
        ]
    )
    result = _close_coverage_sliver(slivered, line, 16)
    assert result.geometry.covers(line)
    assert result.hausdorff_displacement_m <= result.tolerance_m

    visible_gap = Polygon(
        [
            (x + 0.001, y),
            (x + 0.001, y - 2.0),
            (x + 10.0, y - 2.0),
            (x + 10.0, y + 2.0),
            (x + 0.001, y + 2.0),
        ]
    )
    with pytest.raises(ValueError, match="beyond ULP tolerance"):
        _close_coverage_sliver(visible_gap, line, 16)
