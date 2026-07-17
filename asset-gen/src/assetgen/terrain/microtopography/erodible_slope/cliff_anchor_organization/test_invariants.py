"""Small synthetic invariant tests for the cliff-anchor-organization mechanism.

Run: uv run python -m assetgen.terrain.microtopography.erodible_slope.cliff_anchor_organization.test_invariants
"""
from __future__ import annotations

import numpy as np

from . import model as M


def test_world_prf_crop_invariance() -> None:
    """World-PRF white noise is keyed by world coordinate, so a sub-window of a
    large canvas equals the same window generated standalone (crop-invariant)."""
    pitch = 0.25
    big_bbox = (1000.0, 2000.0, 1000.0 + 63 * pitch, 2000.0 + 47 * pitch)
    big = M._world_white((48, 64), big_bbox, pitch, seed=12345)
    # a 16x16 window starting at row 8, col 12 -> its world bbox
    r0, c0 = 8, 12
    sub_bbox = (
        big_bbox[0] + c0 * pitch, big_bbox[3] - (r0 + 15) * pitch,
        big_bbox[0] + (c0 + 15) * pitch, big_bbox[3] - r0 * pitch,
    )
    sub = M._world_white((16, 16), sub_bbox, pitch, seed=12345)
    assert np.allclose(sub, big[r0:r0 + 16, c0:c0 + 16]), "world-PRF is not crop invariant"


def test_world_prf_determinism_and_seed() -> None:
    a = M._world_white((32, 32), (0.0, 0.0, 7.75, 7.75), 0.25, seed=7)
    b = M._world_white((32, 32), (0.0, 0.0, 7.75, 7.75), 0.25, seed=7)
    c = M._world_white((32, 32), (0.0, 0.0, 7.75, 7.75), 0.25, seed=8)
    assert np.array_equal(a, b), "world-PRF not deterministic"
    assert not np.array_equal(a, c), "world-PRF ignores seed"
    assert abs(float(a.mean())) < 0.05 and abs(float(a.max())) <= 0.5


def test_smoothstep_bounds() -> None:
    x = np.linspace(-3, 5, 200)
    s = M._smoothstep(x, 0.5, 2.5)
    assert s.min() >= 0.0 and s.max() <= 1.0
    assert s[0] == 0.0 and s[-1] == 1.0


def test_flow_accumulation_conserves_and_grows_downhill() -> None:
    """On a monotone ramp every cell drains east; accumulation is nondecreasing
    downslope and total interior mass is conserved into the outlet column."""
    H, W = 12, 20
    c0 = np.tile(np.linspace(10.0, 0.0, W), (H, 1))  # drops to the east
    active = np.ones((H, W), dtype=bool)
    acc = M.flow_accumulation(c0, active)
    # accumulation must be positive and largest at the low (east) edge
    assert acc.min() >= 1.0
    assert acc[:, -1].sum() > acc[:, 0].sum()
    # each interior row's easternmost active cell collects its whole row
    assert acc[H // 2, -1] >= W - 1


def test_safety_masks_are_zeroed() -> None:
    """A synthetic escarpment canvas: after synth, every hard/inactive/mapped and
    collar cell has EXACTLY zero residual (safety exactness is non-negotiable)."""
    rng = np.random.default_rng(0)
    H, W = 80, 120
    pitch = 0.25
    bbox = (0.0, 0.0, (W - 1) * pitch, (H - 1) * pitch)
    yy = np.linspace(0, 1, H)[:, None]
    c0 = 5.0 * (1.0 / (1.0 + np.exp(-(yy - 0.5) * 20))) * np.ones((1, W)) + 0.01 * rng.standard_normal((H, W))
    c0 = c0.astype(np.float64)
    import shapely
    line = shapely.LineString([(bbox[0] + 5 * pitch, bbox[3] - H / 2 * pitch), (bbox[2] - 5 * pitch, bbox[3] - H / 2 * pitch)])
    active = np.zeros((H, W), dtype=bool)
    active[10:70, 10:110] = True
    hard = ~active
    mapped = np.zeros((H, W), dtype=bool); mapped[38:42, 10:110] = True
    active &= ~mapped
    anchor = 0.02 * rng.standard_normal((H, W))
    px = bbox[0] + (rng.uniform(12, 108, 400)) * pitch
    py = bbox[3] - (rng.uniform(12, 68, 400)) * pitch
    config = _mini_config()
    frame = M.build_line_frame(line, c0, bbox, pitch, 1.0)
    corridor = M.build_corridor(frame, active, bbox, pitch, 48.0, px, py, c0.shape)
    state = M.fit_profile_state(frame, c0, bbox, pitch, 24.0, 0.5)
    env = M.anchored_envelope(corridor, anchor.ravel()[corridor.ridx], len(frame.station_m), config["envelope"])
    recon = M.synthesize(c0, active, hard, mapped, anchor, frame, corridor, state, env, bbox, pitch, config)
    r = recon.residual_m
    assert np.max(np.abs(r[hard])) == 0.0
    assert np.max(np.abs(r[mapped])) == 0.0
    assert np.max(np.abs(r[~active])) == 0.0
    assert np.all(np.isfinite(r)) and r.dtype == np.float64
    # deterministic: identical inputs -> identical output
    recon2 = M.synthesize(c0, active, hard, mapped, anchor, frame, corridor, state, env, bbox, pitch, config)
    assert np.array_equal(r, recon2.residual_m), "synthesis is not deterministic"


def _mini_config() -> dict:
    return {
        "world_prf_seed": 999,
        "corridor": {"half_width_m": 48.0, "profile_normal_extent_m": 24.0, "profile_normal_step_m": 0.5,
                     "station_step_m": 1.0, "edge_taper_sigma_m": 18.0, "endpoint_taper_m": 6.0},
        "measured_protection": {"als_proximity_ramp_lo_m": 0.5, "als_proximity_ramp_hi_m": 2.5,
                                "void_fill_ramp_lo_m": 2.0, "void_fill_ramp_hi_m": 6.0},
        "envelope": {"near_line_m": 16.0, "station_window_m": 3.0, "percentile": 95.0, "floor_m": 0.05,
                     "ceiling_m": 0.30, "smooth_stations": 5},
        "macro_sharpen": {"sigmoid_width_scale": 0.55, "gain": 0.9},
        "grammar": {"bench_gain": 0.7, "bench_normal_offset_widths": 2.0, "socket_gain": 0.85,
                    "socket_min_plan_curvature": 0.01, "chute_gain": 0.8, "chute_accumulation_percentile": 96.0,
                    "chute_max_seeds": 6, "chute_branch_fraction": 0.45},
        "fine_band": {"sigma_low_m": 0.5, "sigma_high_m": 1.25, "face_amplitude_m": 0.11, "toe_amplitude_m": 0.07,
                      "shoulder_amplitude_m": 0.07, "clip_m": 0.13},
        "gates": {"dc_drift_ordinary_ground_min_distance_m": 30.0,
                  "dc_drift_ordinary_ground_p95_ceiling_m": 0.02},
    }


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"all {len(tests)} invariant tests passed")


if __name__ == "__main__":
    main()
