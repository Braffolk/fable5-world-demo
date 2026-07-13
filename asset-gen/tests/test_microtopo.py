import hashlib
import json

import numpy as np
import pytest

from assetgen.process.microtopo import (
    PatchBank,
    build_residual_bank,
    load_exemplar_bank,
    synthesize_measured,
    synthesize_residual,
)
from assetgen.process.microtopo.model import GroundSurface
from assetgen.process.microtopo.preprocess import (
    GroundExtractionConfig,
    _center_connected_support,
)
from assetgen.process.microtopo.synthesis import (
    _blend_window,
    _pairwise_normalized_error,
)


def _bank() -> PatchBank:
    y, x = np.mgrid[:16, :16]
    patches = np.stack([
        (x + y) * 0.001,
        (x - y) * 0.001 + 0.01,
        np.sin(x * 0.4) * 0.004 + np.cos(y * 0.3) * 0.003,
    ])
    return PatchBank(
        patches_m=patches,
        quality=np.array([0.9, 0.8, 0.7]),
        source_index=np.array([0, 0, 1]),
        source_ids=("k11", "k32"),
        texel_m=0.0625,
        overlap_cells=4,
        provenance={},
    )


def test_synthesis_is_finite_deterministic_and_crop_stable():
    bank = _bank()
    center = 0.5 * bank.texel_m
    whole = synthesize_measured(
        bank, origin_e_m=center, origin_n_m=center, shape=(80, 96), seed=17
    )
    again = synthesize_measured(
        bank, origin_e_m=center, origin_n_m=center, shape=(80, 96), seed=17
    )
    crop = synthesize_measured(
        bank, origin_e_m=(16.5) * bank.texel_m, origin_n_m=(12.5) * bank.texel_m,
        shape=(41, 37), seed=17,
    )
    assert np.isfinite(whole).all()
    assert np.array_equal(whole, again)
    assert np.array_equal(crop, whole[12:53, 16:53])


def test_overlap_blend_avoids_patch_boundary_steps():
    bank = _bank()
    center = 0.5 * bank.texel_m
    out = synthesize_measured(
        bank, origin_e_m=center, origin_n_m=center, shape=(96, 96), seed=8
    )
    jump = np.maximum(np.abs(np.diff(out, axis=0)), np.abs(np.diff(out, axis=1)).T).max()
    raw_range = np.ptp(bank.patches_m)
    assert jump < raw_range * 0.5


def test_overlap_window_is_equal_power_at_the_placement_stride():
    patch_cells = 64
    overlap = 16
    step = patch_cells - overlap
    window = _blend_window(patch_cells, overlap)
    energy = np.zeros((step * 4, step * 4), dtype=np.float64)
    for y0 in range(-step, energy.shape[0], step):
        for x0 in range(-step, energy.shape[1], step):
            y1, x1 = y0 + patch_cells, x0 + patch_cells
            oy0, ox0 = max(0, y0), max(0, x0)
            oy1, ox1 = min(energy.shape[0], y1), min(energy.shape[1], x1)
            py0, px0 = oy0 - y0, ox0 - x0
            energy[oy0:oy1, ox0:ox1] += window[
                py0 : py0 + oy1 - oy0, px0 : px0 + ox1 - ox0
            ] ** 2
    assert np.allclose(energy, 1.0, rtol=0, atol=1e-15)


def test_edge_error_compares_relative_shape_not_absolute_energy():
    a = np.array([[[0.0, 1.0], [2.0, 3.0]]])
    b = np.array([[[0.2, 0.9], [2.1, 2.8]]])
    base = _pairwise_normalized_error(a, b)
    scaled = _pairwise_normalized_error(a * 20.0, b * 20.0)
    assert np.allclose(base, scaled, rtol=0, atol=1e-15)


def test_coordinate_facade_supports_descending_north_axis():
    bank = _bank()
    east = (np.arange(30) + 0.5) * bank.texel_m
    north = (np.arange(25, -1, -1) + 0.5) * bank.texel_m
    descending = synthesize_residual(bank, east, north, seed=4)
    ascending = synthesize_residual(bank, east, north[::-1], seed=4)
    assert descending.dtype == np.float64
    assert np.allclose(descending, ascending[::-1], atol=1e-15)


def test_bank_refuses_holdout_and_unreviewed_surfaces():
    shape = (64, 64)
    surface = GroundSurface(
        elevation_m=np.zeros(shape), measured=np.ones(shape, bool), confidence=np.ones(shape),
        origin_x_m=0, origin_y_m=0, texel_m=0.0625, source_id="k19", role="holdout",
        provenance={}, qa_status="approved",
    )
    with pytest.raises(ValueError, match="holdout"):
        build_residual_bank([surface], patch_cells=32, overlap_cells=8, stride_cells=16)


def test_bank_fills_only_small_unsupported_residual_holes():
    shape = (64, 64)
    elevation = np.zeros(shape)
    elevation[31, 31] = np.nan
    surface = GroundSurface(
        elevation_m=elevation, measured=np.isfinite(elevation), confidence=np.ones(shape),
        origin_x_m=0, origin_y_m=0, texel_m=0.0625, source_id="k11",
        role="calibration", provenance={}, qa_status="approved",
    )
    bank = build_residual_bank(
        [surface], patch_cells=32, overlap_cells=8, stride_cells=16,
        coarse_factor=16, min_direct_fraction=0.99,
    )
    assert np.isfinite(bank.patches_m).all()
    assert bank.provenance["max_fill_distance_cells"] == 2.0

    blocked = elevation.copy()
    blocked[26:37, 26:37] = np.nan
    surface = GroundSurface(
        elevation_m=blocked, measured=np.isfinite(blocked), confidence=np.ones(shape),
        origin_x_m=0, origin_y_m=0, texel_m=0.0625, source_id="k11",
        role="calibration", provenance={}, qa_status="approved",
    )
    with pytest.raises(ValueError, match="no residual patches"):
        build_residual_bank(
            [surface], patch_cells=64, overlap_cells=8, stride_cells=16,
            coarse_factor=16, min_valid_fraction=0.9, min_direct_fraction=0.9,
        )


def test_manifest_loader_verifies_bound_bank_bytes(tmp_path):
    bank_path = tmp_path / "bank.npz"
    _bank().save(bank_path)
    digest = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "format": 1, "bankFile": "bank.npz", "bankSha256": digest,
    }))
    assert load_exemplar_bank(manifest).patches_m.shape == (3, 16, 16)
    manifest.write_text(json.dumps({
        "format": 1, "bankFile": "bank.npz", "bankSha256": "0" * 64,
    }))
    with pytest.raises(ValueError, match="SHA-256"):
        load_exemplar_bank(manifest)


def test_center_connected_support_rejects_elevated_object_island():
    candidate = np.zeros((64, 64), dtype=np.float64)
    candidate[8:16, 45:53] = np.nan
    candidate[10:14, 47:51] = 7.0
    views = np.full((64, 64), 3, dtype=np.uint8)
    spread = np.full((64, 64), 0.01)
    support, confidence = _center_connected_support(
        candidate, views, spread, GroundExtractionConfig()
    )
    assert np.isnan(support[10:14, 47:51]).all()
    assert np.nanmax(support) == 0.0
    assert confidence[32, 32] > 0
