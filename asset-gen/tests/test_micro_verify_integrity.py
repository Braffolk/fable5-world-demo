import numpy as np

from assetgen.micro_verify import (
    _decoded_authority_mean_error,
    _decoded_rejected_residual,
)


def test_decoded_authority_and_rejected_residual_are_data_derived():
    authority = np.array([[2.0, 4.0], [6.0, 8.0]])
    decoded = np.repeat(np.repeat(authority, 2, axis=0), 2, axis=1)
    assert _decoded_authority_mean_error(decoded, authority, factor=2) == 0.0

    allowed = np.ones(decoded.shape, dtype=bool)
    allowed[0, 0] = False
    count, maximum = _decoded_rejected_residual(decoded, decoded.copy(), allowed)
    assert (count, maximum) == (1, 0.0)

    tampered = decoded.copy()
    tampered[0, 0] += 0.01
    assert _decoded_authority_mean_error(tampered, authority, factor=2) > 0.0
    count, maximum = _decoded_rejected_residual(tampered, decoded, allowed)
    assert count == 1
    assert np.isclose(maximum, 0.01)
