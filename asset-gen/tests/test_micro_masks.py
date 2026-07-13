import numpy as np

from assetgen.process.micro_masks import MicroHardMask, _compose_morphology_mask


def test_morphology_selector_is_fail_closed_and_reports_reasons():
    landcover = np.array([[1, 2, 1, 1], [1, 1, 1, 1]], dtype=np.uint8)
    soil_type = np.array([[9, 9, 54, 9], [0, 16, 9, 9]], dtype=np.uint8)
    texture = np.array([[0, 0, 9, 0], [0, 1, 255, 0]], dtype=np.uint8)
    boniteet = np.array([[40, 40, 40, 20], [40, 40, 40, 40]], dtype=np.uint8)
    water = np.zeros((2, 4), dtype=bool)
    water[1, 3] = True
    empty = np.zeros_like(water)
    hard = MicroHardMask(~water, water, empty, empty)
    slope = np.zeros_like(water)
    slope[1, 2] = True
    drained = np.zeros_like(water)
    drained[0, 1] = True

    mask = _compose_morphology_mask(
        landcover, soil_type, texture, boniteet, drained, hard, slope
    )

    assert np.array_equal(
        mask.allowed,
        np.array([[True, False, False, False], [False, False, False, False]]),
    )
    evidence = mask.evidence()
    assert evidence["allowedCells"] == 1
    assert evidence["waterCells"] == 1
    assert evidence["slopeCliffContextCells"] == 1
    assert evidence["drainedSoilCells"] == 1
    assert evidence["unknownSoilCells"] == 2
    assert evidence["rejectedReasonCells"]["nonForest"] == 1


def test_morphology_selector_rejects_shape_mismatch():
    shape = (2, 2)
    empty = np.zeros(shape, dtype=bool)
    hard = MicroHardMask(~empty, empty, empty, empty)
    with np.testing.assert_raises_regex(ValueError, "share the hard-mask shape"):
        _compose_morphology_mask(
            np.zeros((1, 2), dtype=np.uint8),
            np.zeros(shape, dtype=np.uint8),
            np.zeros(shape, dtype=np.uint8),
            np.zeros(shape, dtype=np.uint8),
            np.zeros(shape, dtype=bool),
            hard,
            empty,
        )
