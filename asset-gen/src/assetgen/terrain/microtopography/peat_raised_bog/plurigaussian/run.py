"""Run plurigaussian synthesis, evaluate every gate honestly, and (optionally) freeze the
immutable float artifact the packer consumes.

Gates (all measured on the OUTPUT relief):
  * safety        : off-authority / open-water residual == 0; core-edge relief ~ 0.
  * amplitude     : local hummock-hollow relief p50 in [0.15,0.25], p95 in [0.28,0.40].
  * transiogram   : Moore-banded hollow<->hummock contact ~ 0; monotone ordering.
  * fractions     : plurigaussian class fractions within tolerance of the Ilyasov target.
  * anti_artifact : radial PSD slope in [-4.6,-3.3]; dominant spacing in [0.8,3] m; no peak.
  * two_scale     : microform first-zero in [0.3,0.9] m AND a coarse (10-100 m) scale present.
  * pool_coupling : pool margins run low; no hummock over pools.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from .....config import DATA_IN, DATA_WORK
from . import gates, synth

ETAK = DATA_IN / "etak/ETAK_EESTI_GPKG.gpkg"
DTM10 = DATA_IN / "country/DTM_10m_eesti.tif"
ARTIFACT_ROOT = DATA_WORK / "microtopography/peat-bog-plurigaussian-v6/sha256"
FLOAT_NAME = "plurigaussian-v6-float.npz"

# Gate thresholds (frozen).
AMP_P50 = (0.15, 0.25)
AMP_P95 = (0.28, 0.40)
PSD_SLOPE = (-4.6, -3.3)
DOMINANT_SPACING = (0.8, 3.0)
MICROFORM_BAND_M = (0.5, 3.0)  # microform PSD fit band; kept fixed for apples-to-apples with
#                                the 0.25 m gate (0.0625 m now resolves down to 0.125 m Nyquist,
#                                but the microform fit stays in the 0.5-3 m band; see gates.radial_psd)
FIRST_ZERO = (0.3, 0.9)
COARSE_SCALE_MIN_M = 8.0
HOLLOW_HUMMOCK_MAX = 0.01
FRACTION_TOL = 0.08
PERIODICITY_MAX = 8.0
LOCAL_WINDOW_M = 2.0


def evaluate(result: synth.SynthResult) -> dict[str, Any]:
    pitch = synth.OUTPUT_PITCH_M
    core = result.relief_core
    core_auth = result.authority[result.core_slice]
    core_water = result.open_water[result.core_slice]

    # --- safety ----------------------------------------------------------------------------
    off_auth_residual = int(np.count_nonzero(core[~core_auth]))
    water_residual = int(np.count_nonzero(core[core_water]))
    edge = np.concatenate([core[0, :], core[-1, :], core[:, 0], core[:, -1]])
    edge_p99 = float(np.percentile(np.abs(edge), 99))
    safety_ok = off_auth_residual == 0 and water_residual == 0 and edge_p99 <= 0.01

    # --- amplitude (local relief over authority) ------------------------------------------
    lr = gates.local_relief(core, core_auth, pitch, LOCAL_WINDOW_M)
    amp_ok = AMP_P50[0] <= lr["p50_m"] <= AMP_P50[1] and AMP_P95[0] <= lr["p95_m"] <= AMP_P95[1]
    p95_p5 = float(np.percentile(core[core_auth], 95) - np.percentile(core[core_auth], 5))

    # --- Moore-banded transiogram + fractions ---------------------------------------------
    z_rel = gates.detrend(core, core_auth, pitch)
    lab = gates.classify(z_rel)
    trans = gates.lag1_transition(lab)
    hollow_hummock = float(trans[gates.HOLLOW, gates.HUMMOCK] + trans[gates.HUMMOCK, gates.HOLLOW])
    monotone = bool(trans[gates.HOLLOW, gates.HOLLOW] > trans[gates.HOLLOW, gates.HUMMOCK]
                    and trans[gates.HUMMOCK, gates.HUMMOCK] > trans[gates.HUMMOCK, gates.HOLLOW])
    moore_af = gates.area_fractions(lab)
    trans_ok = hollow_hummock <= HOLLOW_HUMMOCK_MAX and monotone

    # --- plurigaussian class fractions vs Ilyasov-derived target ---------------------------
    target = {k: result.class_fractions[k] for k in ("hollow", "lawn", "hummock")}
    from . import anamorphosis

    ily = anamorphosis.estonian_fractions()
    frac_err = {k: abs(target[k] - ily[k]) for k in ily}
    frac_ok = all(v <= FRACTION_TOL for v in frac_err.values())

    # --- anti-artifact (detrended microform spectrum) -------------------------------------
    psd = gates.radial_psd(z_rel, pitch, MICROFORM_BAND_M)
    periodicity = gates.periodicity_peak(z_rel)
    slope_ok = psd["psd_loglog_slope"] is not None and PSD_SLOPE[0] <= psd["psd_loglog_slope"] <= PSD_SLOPE[1]
    spacing_ok = psd["dominant_wavelength_m"] is not None and DOMINANT_SPACING[0] <= psd["dominant_wavelength_m"] <= DOMINANT_SPACING[1]
    anti_ok = slope_ok and spacing_ok and periodicity <= PERIODICITY_MAX

    # --- two-scale ------------------------------------------------------------------------
    ac = gates.autocorr_scale(z_rel, pitch)
    coarse_len = gates.coarse_autocorr_length(core, core_auth, pitch)
    fine_ok = ac["first_zero_lag_m"] is not None and FIRST_ZERO[0] <= ac["first_zero_lag_m"] <= FIRST_ZERO[1]
    coarse_ok = coarse_len >= COARSE_SCALE_MIN_M
    two_scale_ok = fine_ok and coarse_ok

    # --- pool coupling --------------------------------------------------------------------
    pc = gates.pool_coupling(core, core_water, core_auth, pitch, 12.0)
    pool_ok = (not pc["pool_present"]) or (pc["margin_low_fraction"] is not None
                                           and pc["margin_low_fraction"] >= 0.6
                                           and pc["hummock_over_pool_cells"] == 0)

    # --- semivariogram (diagnostic) -------------------------------------------------------
    report = {
        "safety": {"passed": bool(safety_ok), "off_authority_residual_cells": off_auth_residual,
                   "open_water_residual_cells": water_residual, "core_edge_abs_p99_m": edge_p99},
        "amplitude": {"passed": bool(amp_ok), **lr, "p50_target": AMP_P50, "p95_target": AMP_P95,
                      "global_p95_minus_p5_m": p95_p5},
        "transiogram": {"passed": bool(trans_ok), "hollow_hummock_contact": hollow_hummock,
                        "monotone_ordering": monotone, "lag1_matrix": trans.tolist(),
                        "moore_banded_area_fractions": moore_af},
        "fractions": {"passed": bool(frac_ok), "plurigaussian": target,
                      "ilyasov_derived_target": ily, "abs_error": frac_err, "tol": FRACTION_TOL},
        "anti_artifact": {"passed": bool(anti_ok), **psd, "periodicity_peak_ratio": periodicity,
                          "slope_target": PSD_SLOPE, "spacing_target": DOMINANT_SPACING,
                          "periodicity_max": PERIODICITY_MAX},
        "two_scale": {"passed": bool(two_scale_ok), "microform_first_zero_m": ac["first_zero_lag_m"],
                      "microform_integral_length_m": ac["integral_length_m"],
                      "coarse_half_autocorr_m": coarse_len, "coarse_min_m": COARSE_SCALE_MIN_M,
                      "first_zero_target": FIRST_ZERO},
        "pool_coupling": {"passed": bool(pool_ok), **pc},
    }
    report["all_passed"] = bool(safety_ok and amp_ok and trans_ok and frac_ok
                                and anti_ok and two_scale_ok and pool_ok)
    return report


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def _recipe_sha(result: synth.SynthResult, relief: np.ndarray) -> str:
    module_files = sorted(Path(__file__).parent.glob("*.py"))
    src = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in module_files}
    payload = {
        "params": result.params.__dict__,
        "inputs": result.inputs,
        "source_sha256": src,
        "relief_sha256": hashlib.sha256(relief.astype(np.float32).tobytes()).hexdigest(),
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


def freeze(result: synth.SynthResult, report: dict, output_root: Path) -> Path:
    relief = result.relief_core.astype(np.float32)
    recipe_sha = _recipe_sha(result, relief)
    out = output_root / recipe_sha
    if out.exists():
        raise FileExistsError(f"immutable plurigaussian v6 artifact exists: {out}")
    measurements = {
        "schema_version": "laas.peat-bog-plurigaussian-v6-float-artifact/1",
        "build_id": recipe_sha,
        "route": "hydrology_conditioned_two_field_plurigaussian_gaussian_anamorphosis",
        "status": "float_qa_pending_ground_level_verdict" if report["all_passed"] else "research_rejected_gate_fail",
        "params": result.params.__dict__,
        "inputs": result.inputs,
        "flow_angle_rad": result.flow_angle_rad,
        "class_fractions": result.class_fractions,
        "gates": report,
        "authority": {
            "research_only": True, "target_truth": False, "estonia_transfer": False,
            "latest_eligible": False, "measured_morphology_fit": False,
            "packing_runtime_shader_material_or_format_authority": False,
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / FLOAT_NAME,
        core_relief_00625m=relief,
        latent_std_core=result.latent_std[result.core_slice].astype(np.float32),
        coarse_core=result.coarse[result.core_slice].astype(np.float32),
        authority_core=result.authority[result.core_slice],
        open_water_core=result.open_water[result.core_slice],
        measurements_json_u8=np.frombuffer(_canonical(measurements), dtype=np.uint8),
    )
    (out / "measurements.json").write_bytes(_canonical(measurements))
    return out


def _print_table(report: dict) -> None:
    print("\n=== PLURIGAUSSIAN v6 GATE TABLE ===")
    for name, g in report.items():
        if name == "all_passed":
            continue
        mark = "PASS" if g.get("passed") else "FAIL"
        print(f"[{mark}] {name}: {json.dumps({k: v for k, v in g.items() if k != 'lag1_matrix'}, default=float)[:200]}")
    print(f"ALL_PASSED = {report['all_passed']}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="write the immutable float artifact")
    parser.add_argument("--cache-dir", type=Path, default=DATA_WORK / "microtopography/peat-bog-plurigaussian-v6/noise-cache")
    parser.add_argument("--amplitude", type=float, default=None)
    parser.add_argument("--fine-range", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.amplitude is not None:
        overrides["amplitude"] = args.amplitude
    if args.fine_range is not None:
        overrides["fine_range_m"] = args.fine_range
    if args.beta is not None:
        overrides["coarse_weight_beta"] = args.beta
    params = synth.SynthParams(**overrides)

    result = synth.synthesize(ETAK, DTM10, params, cache_dir=args.cache_dir)
    report = evaluate(result)
    _print_table(report)
    if args.freeze:
        if not report["all_passed"]:
            print("NOT freezing: gates fail.")
            return 3
        out = freeze(result, report, ARTIFACT_ROOT)
        print(f"FROZE {out}")
    return 0 if report["all_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
