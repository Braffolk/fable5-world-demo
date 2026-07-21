import numpy as np

from assetgen.groundcover.control_field import GroundCoverId, derive_control_planes


def _window(e0: float = 0.0) -> tuple[float, float, float, float, float]:
    return (e0, 0.0, e0 + 8.0, 8.0, 2.0)


def test_community_mix_and_wetland_floor() -> None:
    community = np.array([[0, 5, 6, 7]] * 4, dtype=np.uint8)
    density = np.array([[0, 220, 190, 180]] * 4, dtype=np.uint8)
    soil_type = np.zeros_like(community)
    tex_core = np.zeros_like(community)
    canopy = np.zeros_like(community)
    a, b, _, _, blend, vigor, moisture, proximity = derive_control_planes(
        community, density, soil_type, tex_core, canopy, _window()
    )
    assert a[0].tolist() == [GroundCoverId.BARE, GroundCoverId.MOSS, GroundCoverId.SEDGE, GroundCoverId.GRASS]
    assert b[0].tolist() == [GroundCoverId.BARE, GroundCoverId.SEDGE, GroundCoverId.MOSS, GroundCoverId.FORB]
    assert blend[0].tolist() == [0, 88, 96, 86]
    assert vigor[0].tolist() == density[0].tolist()
    assert moisture[0, 1] >= 229 and moisture[0, 2] >= 229
    assert not proximity.any()


def test_control_field_is_deterministic_and_apron_seam_matches() -> None:
    shape = (4, 4)
    community = np.full(shape, 7, dtype=np.uint8)
    density = np.full(shape, 200, dtype=np.uint8)
    zeros = np.zeros(shape, dtype=np.uint8)
    left = derive_control_planes(community, density, zeros, zeros, zeros, _window(0.0))
    again = derive_control_planes(community, density, zeros, zeros, zeros, _window(0.0))
    right = derive_control_planes(community, density, zeros, zeros, zeros, _window(6.0))
    for got, expected in zip(left, again):
        np.testing.assert_array_equal(got, expected)
    # left x=3 and right x=0 have the same 7 m east-coordinate sample.
    np.testing.assert_array_equal(left[2][:, -1], right[2][:, 0])
    np.testing.assert_array_equal(left[3][:, -1], right[3][:, 0])


def test_unknown_community_fails_closed() -> None:
    bad = np.full((4, 4), 77, dtype=np.uint8)
    zeros = np.zeros_like(bad)
    try:
        derive_control_planes(bad, zeros, zeros, zeros, zeros, _window())
    except ValueError as exc:
        assert "unmapped community ids [77]" in str(exc)
    else:
        raise AssertionError("unknown community must fail closed")
