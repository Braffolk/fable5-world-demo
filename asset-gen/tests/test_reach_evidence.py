import numpy as np
from shapely.geometry import LineString, Polygon, box

from assetgen.evidence.reach import cap_water_polygon_to_centerline


def test_caps_local_channel_without_cutting_meander_crossing_endpoint_normal() -> None:
    line = LineString([(0, 8), (0, 2), (10, 2), (10, 8), (3, 8), (3, 4)])
    water = line.buffer(1.0, cap_style="flat", join_style="round").union(box(-1, 8, 1, 12))
    capped = cap_water_polygon_to_centerline(water, line)
    assert capped.covers(line)
    assert not capped.covers(box(-0.5, 9, 0.5, 10).centroid)


def test_cap_is_deterministic_and_rejects_uncovered_centerline() -> None:
    line = LineString([(0, 0), (10, 0)])
    water = box(-5, -2, 15, 2)
    a = cap_water_polygon_to_centerline(water, line)
    b = cap_water_polygon_to_centerline(water, line)
    assert a.equals_exact(b, tolerance=0.0)
    assert np.allclose(a.bounds, (0.0, -2.0, 10.0, 2.0))

    try:
        cap_water_polygon_to_centerline(box(0, -2, 5, 2), line)
    except ValueError as error:
        assert "covering" in str(error)
    else:
        raise AssertionError("uncovered centerline was accepted")
