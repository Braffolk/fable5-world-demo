import hashlib

import numpy as np

from assetgen.groundcover.control_field import (
    GroundCoverId,
    GroundCoverProfileId,
    derive_control_planes,
)


def _window(e0: float = 0.0) -> tuple[float, float, float, float, float]:
    return (e0, 0.0, e0 + 8.0, 8.0, 2.0)


def test_community_mix_and_wetland_floor() -> None:
    community = np.array([[0, 5, 6, 7]] * 4, dtype=np.uint8)
    density = np.array([[0, 220, 190, 180]] * 4, dtype=np.uint8)
    soil_type = np.zeros_like(community)
    tex_core = np.zeros_like(community)
    canopy = np.zeros_like(community)
    a, b, _, _, blend, vigor, moisture, proximity, mask_lo, mask_hi, profile_a, profile_b = derive_control_planes(
        community, density, soil_type, tex_core, canopy, _window()
    )
    assert a[0].tolist() == [GroundCoverId.BARE, GroundCoverId.MOSS, GroundCoverId.SEDGE, GroundCoverId.GRASS]
    assert b[0].tolist() == [GroundCoverId.BARE, GroundCoverId.SEDGE, GroundCoverId.MOSS, GroundCoverId.FORB]
    assert blend[0].tolist() == [0, 88, 96, 86]
    assert vigor[0].tolist() == density[0].tolist()
    assert moisture[0, 1] >= 229 and moisture[0, 2] >= 229
    assert not proximity.any()
    assert int(mask_hi[0, 0]) == 0  # bare stays no-query even at the conservative rim
    assert profile_a[0].tolist() == [
        255,
        GroundCoverProfileId.SPHAGNUM_CAPILLIFOLIUM,
        GroundCoverProfileId.CAREX_CESPITOSA,
        GroundCoverProfileId.AGROSTIS_CAPILLARIS,
    ]
    assert profile_b[0].tolist() == [
        255,
        GroundCoverProfileId.ERIOPHORUM_VAGINATUM,
        GroundCoverProfileId.SPHAGNUM_CAPILLIFOLIUM,
        GroundCoverProfileId.OXALIS_ACETOSELLA,
    ]
    assert mask_lo[0, 0] == 0  # bare stays no-query even at the conservative rim


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


def test_candidate_mask_closes_over_four_metre_root_reach() -> None:
    community = np.full((9, 9), 7, dtype=np.uint8)  # grass/forb
    community[4, 6] = 2  # moss/shrub, exactly 4 m east
    density = np.full_like(community, 200)
    zeros = np.zeros_like(community)
    planes = derive_control_planes(community, density, zeros, zeros, zeros, (0, 0, 18, 18, 2))
    mask = planes[8].astype(np.uint16) | (planes[9].astype(np.uint16) << np.uint16(8))
    profile_a, profile_b = planes[10], planes[11]
    expected = 0
    for profile_id in np.unique(np.concatenate((profile_a[2:7, 2:7], profile_b[2:7, 2:7]))):
        if profile_id != 0xFF:
            expected |= 1 << int(profile_id)
    assert int(mask[4, 4]) == expected
    expected_west = 0
    for profile_id in np.unique(np.concatenate((profile_a[2:7, 1:6], profile_b[2:7, 1:6]))):
        if profile_id != 0xFF:
            expected_west |= 1 << int(profile_id)
    assert int(mask[4, 3]) == expected_west


def test_first_eight_v1_planes_are_byte_frozen() -> None:
    shape = (8, 8)
    y, x = np.indices(shape)
    community = ((x + 3 * y) % 10).astype(np.uint8)
    density = ((x * 37 + y * 19 + 41) % 256).astype(np.uint8)
    density[(x + 2 * y) % 7 == 0] = 0
    soil = np.zeros(shape, dtype=np.uint8)
    tex_core = ((2 * x + y) % 15).astype(np.uint8)
    canopy = ((31 * x + 17 * y) % 256).astype(np.uint8)
    planes = derive_control_planes(
        community,
        density,
        soil,
        tex_core,
        canopy,
        (330000.0, 6400000.0, 330016.0, 6400016.0, 2.0),
    )
    digest = hashlib.sha256(b"".join(plane.tobytes() for plane in planes[:8])).hexdigest()
    assert digest == "2985032252ab60bb8b640454544a271db97957830d740b9e1bdf73bfd6dc55ea"


def test_exact_profile_facies_reaches_all_twelve_native_ids() -> None:
    shape = (144, 128)
    community = np.empty(shape, dtype=np.uint8)
    for band, community_id in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9]):
        community[band * 16:(band + 1) * 16] = community_id
    density = np.full(shape, 205, dtype=np.uint8)
    density[:, ::7] = 145
    soil = np.zeros(shape, dtype=np.uint8)
    tex_core = np.full(shape, 4, dtype=np.uint8)
    tex_core[0:16] = 1       # dry sandy heath
    tex_core[16:48] = 3      # mineral forest floor
    tex_core[64:96] = 12     # raised bog + fen peat
    tex_core[112:128] = 13   # drained peat
    tex_core[128:144] = 1    # dry sandy scrub
    canopy = np.zeros(shape, dtype=np.uint8)
    canopy[16:64] = 184      # bilberry/herb forest
    canopy[112:128] = 150    # drained-peat forest
    canopy[128:144] = 180    # partially closed scrub
    planes = derive_control_planes(
        community,
        density,
        soil,
        tex_core,
        canopy,
        (330000.0, 6400000.0, 330256.0, 6400288.0, 2.0),
    )
    ids = set(np.unique(np.concatenate((planes[10].ravel(), planes[11].ravel()))).tolist())
    ids.discard(0xFF)
    assert ids == set(range(len(GroundCoverProfileId)))

    compatible = {
        GroundCoverId.GRASS: {0, 1, 2},
        GroundCoverId.MOSS: {5, 6},
        GroundCoverId.SEDGE: {3, 4},
        GroundCoverId.LICHEN: {7},
        GroundCoverId.FORB: {8, 9},
        GroundCoverId.DWARF_SHRUB: {10, 11},
    }
    for type_plane, profile_plane in ((planes[0], planes[10]), (planes[1], planes[11])):
        for cover_id, valid_profiles in compatible.items():
            selected = set(np.unique(profile_plane[type_plane == cover_id]).tolist())
            assert selected <= valid_profiles

    # Hard ecological anchors: the correlated fields may vary facies within a
    # class, but they must not invert these community-defining associations.
    assert np.all(planes[10][0:16] == GroundCoverProfileId.CLADONIA_RANGIFERINA)
    assert np.all(planes[11][0:16] == GroundCoverProfileId.CALLUNA_VULGARIS)
    assert np.all(planes[10][16:32] == GroundCoverProfileId.PLEUROZIUM_SCHREBERI)
    assert np.all(planes[11][16:32] == GroundCoverProfileId.VACCINIUM_MYRTILLUS)
    assert np.all(planes[10][64:80] == GroundCoverProfileId.SPHAGNUM_CAPILLIFOLIUM)
    assert np.all(planes[11][64:80] == GroundCoverProfileId.ERIOPHORUM_VAGINATUM)
    assert np.all(planes[10][80:96] == GroundCoverProfileId.CAREX_CESPITOSA)
    assert np.all(planes[11][80:96] == GroundCoverProfileId.SPHAGNUM_CAPILLIFOLIUM)


def test_profile_facies_matches_at_shared_world_samples() -> None:
    shape = (8, 8)
    community = np.full(shape, 3, dtype=np.uint8)
    density = np.full(shape, 190, dtype=np.uint8)
    soil = np.zeros(shape, dtype=np.uint8)
    tex_core = np.full(shape, 4, dtype=np.uint8)
    canopy = np.full(shape, 170, dtype=np.uint8)
    left = derive_control_planes(community, density, soil, tex_core, canopy, (0, 0, 16, 16, 2))
    right = derive_control_planes(community, density, soil, tex_core, canopy, (14, 0, 30, 16, 2))
    # The last left and first right columns both sample world easting 15 m.
    np.testing.assert_array_equal(left[10][:, -1], right[10][:, 0])
    np.testing.assert_array_equal(left[11][:, -1], right[11][:, 0])
