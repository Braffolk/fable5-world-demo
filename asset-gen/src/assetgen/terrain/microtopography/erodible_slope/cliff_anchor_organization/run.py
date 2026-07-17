"""Materialize the Development-A cliff-anchor-organization float candidate.

STOP after FLOAT + gate evaluation + QA. No packing/publishing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from .....config import ASSET_GEN_ROOT, DATA_WORK
from ...coastal_escarpment.als_tgv.evidence import load_evidence, _target_line
from . import model as M
from .qa import write_qa

_SCHEMA = "laas.erodible-slope-cliff-anchor-organization-config/1"
_ARTIFACT_ROOT = DATA_WORK / "microtopography" / "cliff-anchor-organization-v1"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_json_default).encode("ascii")


def _sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            d.update(block)
    return d.hexdigest()


def _identity(path: Path, logical: str | None = None) -> dict[str, Any]:
    if logical is None:
        logical = str(path.resolve().relative_to(ASSET_GEN_ROOT.parent))
    return {"path": logical, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _bind(row: dict[str, Any]) -> Path:
    path = ASSET_GEN_ROOT.parent / row["path"]
    if _sha256_file(path) != row["sha256"]:
        raise ValueError(f"bound SHA-256 differs: {path}")
    return path


def _write_npy(path: Path, value: np.ndarray) -> None:
    with path.open("wb") as dst:
        np.lib.format.write_array(dst, np.asarray(value, dtype="<f4", order="C"), allow_pickle=False)


def _read_config(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema_version") != _SCHEMA:
        raise ValueError("unsupported cliff-anchor-organization config")
    if doc.get("status") != "development_only" or doc.get("target_etak_id") != 1826743:
        raise ValueError("Development-A boundary differs")
    laws = doc.get("hard_laws", {})
    if laws.get("biala_amplitudes_forbidden") is not True or laws.get("taevaskoda_sentinel_opened") is not False:
        raise ValueError("research/sentinel boundary differs")
    if laws.get("sandstone_carbonate_abstained") is not True:
        raise ValueError("sandstone/carbonate abstention differs")
    return doc


# --------------------------------------------------------------------------- #
# gates                                                                         #
# --------------------------------------------------------------------------- #
def _holdout(ev, field: np.ndarray, block_m: float = 8.0) -> dict[str, Any]:
    b = ev.bbox_en; pitch = ev.pitch_m
    pr = ndimage.map_coordinates(field, [(b[3] - ev.point_y) / pitch, (ev.point_x - b[0]) / pitch], order=1, mode="nearest")
    bx = np.floor((ev.point_x - b[0]) / block_m).astype(np.int64)
    by = np.floor((ev.point_y - b[1]) / block_m).astype(np.int64)
    bid = by * 10000 + bx
    res: dict[str, Any] = {}
    for side, name in ((1, "high"), (-1, "low")):
        sel = ev.point_holdout & (ev.point_side == side)
        rows = []
        for ident in np.unique(bid[sel]):
            bb = sel & (bid == ident)
            if int(np.count_nonzero(bb)) >= 5:
                rows.append(abs(float(np.median(ev.point_residual_m[bb] - pr[bb]))))
        res[name] = {"block_median_mae_m": float(np.mean(rows)) if rows else float("inf"), "qualified_blocks": len(rows)}
    return res


def _connectedness(ev, frame, state, after: np.ndarray, config: dict) -> dict[str, Any]:
    b = ev.bbox_en; pitch = ev.pitch_m
    g = config["gates"]["connectedness"]
    min_relief = float(g["coherent_face_min_relief_m"])
    ncent = np.arange(-24.0, 24.01, 0.5)
    nS = len(frame.station_m)

    def sample(px, py):
        return ndimage.map_coordinates(after, [(b[3] - py) / pitch, (px - b[0]) / pitch], order=1, mode="nearest")

    coherent = np.zeros(nS, dtype=bool)
    supported = np.zeros(nS, dtype=bool)
    active = ev.active
    for s in range(nS):
        # supported if the line here sits inside the active/material corridor
        rr = int(np.clip(round((b[3] - frame.sy[s]) / pitch), 0, active.shape[0] - 1))
        cc = int(np.clip(round((frame.sx[s] - b[0]) / pitch), 0, active.shape[1] - 1))
        r0, r1 = max(rr - 40, 0), min(rr + 41, active.shape[0])
        c0b, c1b = max(cc - 40, 0), min(cc + 41, active.shape[1])
        supported[s] = np.count_nonzero(active[r0:r1, c0b:c1b]) > 200
        prof = sample(frame.sx[s] + ncent * frame.nx[s], frame.sy[s] + ncent * frame.ny[s])
        crest = np.median(prof[ncent >= 14.0]); toe = np.median(prof[ncent <= -14.0])
        face_h = crest - toe
        face_zone = (ncent >= state.center[s] - 1.6 * state.width[s]) & (ncent <= state.center[s] + 1.6 * state.width[s])
        if np.count_nonzero(face_zone) >= 4:
            d = np.diff(prof[face_zone])
            monotone = float(np.mean(d >= -0.02))  # nondecreasing toe->crest, small tolerance
        else:
            monotone = 0.0
        coherent[s] = (face_h > min_relief) and (monotone > 0.9)
    step = frame.station_m[1] - frame.station_m[0]
    sup = supported
    frac = float(np.count_nonzero(coherent & sup)) / max(int(np.count_nonzero(sup)), 1)
    # longest run of supported-but-incoherent
    gap = 0; mx = 0
    for s in range(nS):
        if sup[s] and not coherent[s]:
            gap += 1; mx = max(mx, gap)
        else:
            gap = 0
    return {
        "supported_stations": int(np.count_nonzero(sup)),
        "coherent_fraction": frac,
        "max_crest_break_gap_m": mx * step,
        "pass": frac >= float(g["min_supported_coherent_fraction"]) and mx * step <= float(g["max_crest_break_gap_m"]),
    }


def _fine_band_metrics(recon, config) -> dict[str, Any]:
    """Measure the band-limited fine component where it is EXPRESSED (proximity>0.5,
    i.e. between measurements). Intent per brief: energy WITHIN the Biala source-band
    envelope for matching form types (an upper ceiling) and band-limited (no broadband
    speckle) — there is no lower bound on synthetic texture energy."""
    fine = recon.fine_m
    expressed = recon.proximity > 0.5
    def p95(member):
        v = np.abs(fine[(member > 0.3) & expressed])
        return float(np.percentile(v, 95)) if v.size else 0.0
    band_all = np.abs(fine[(fine != 0.0) & expressed])
    return {
        "region": "expressed (als proximity > 0.5)",
        "face_p95_m": p95(recon.face_member), "toe_p95_m": p95(recon.toe_member),
        "shoulder_p95_m": p95(recon.shoulder_member),
        "band_p95_m": float(np.percentile(band_all, 95)) if band_all.size else 0.0,
    }


def _failure_mode_metrics(ev, frame, state, recon, config) -> dict[str, Any]:
    b = ev.bbox_en; pitch = ev.pitch_m; residual = recon.residual_m
    nS = len(frame.station_m); step = frame.station_m[1] - frame.station_m[0]
    # residual sampled on the face centerline per station
    face_line = ndimage.map_coordinates(
        residual,
        [(b[3] - (frame.sy + state.center * frame.ny)) / pitch, (frame.sx + state.center * frame.nx - 0.0 - b[0]) / pitch],
        order=1, mode="nearest",
    )
    amp = np.abs(face_line)
    endL = float(config["corridor"]["endpoint_taper_m"])
    end = (frame.station_m < endL) | (frame.station_m > frame.length_m - endL)
    interior = ~end
    end_ratio = float(np.mean(amp[end]) / max(np.mean(amp[interior]), 1e-9)) if np.any(interior) else 0.0
    # along-line rib spectrum: dominant non-DC power fraction
    sig = face_line - np.mean(face_line)
    if np.any(sig):
        spec = np.abs(np.fft.rfft(sig))
        spec[0] = 0.0
        rib_ratio = float(spec.max() / (np.sum(spec) + 1e-12))
    else:
        rib_ratio = 0.0
    fm = config["gates"]["failure_modes"]
    return {
        "endpoint_cap_ratio": end_ratio,
        "endpoint_cap_pass": end_ratio <= float(fm["endpoint_cap_ratio_ceiling"]),
        "normal_rib_spectral_ratio": rib_ratio,
        "normal_rib_pass": rib_ratio <= float(fm["normal_rib_spectral_ratio_ceiling"]),
    }


def run(config_path: Path) -> Path:
    config_payload = config_path.read_bytes()
    config = _read_config(config_path)

    ev_row = config["evidence_config"]
    ev_config_path = _bind(ev_row)
    ev_config = json.loads(ev_config_path.read_text(encoding="utf-8"))
    ev = load_evidence(ev_config)
    c0 = ev.c0_m.astype(np.float64)
    b = ev.bbox_en; pitch = ev.pitch_m

    anchor_path = _bind(config["measured_anchor_artifact"]["residual"])
    anchor_residual = np.load(anchor_path).astype(np.float64)
    if anchor_residual.shape != c0.shape:
        raise ValueError("anchor residual shape differs from solve canvas")
    _bind(config["measured_anchor_artifact"]["metrics"])
    biala_path = _bind(config["connective_grammar_artifact"]["npz"])
    with np.load(biala_path, allow_pickle=False) as bz:  # bound as evidence of the transferred grammar source
        _ = bz["macro_component_m"].shape

    line = _target_line(ASSET_GEN_ROOT.parent / ev_config["inputs"]["etak"]["path"], int(config["target_etak_id"]))
    corr = config["corridor"]
    frame = M.build_line_frame(line, c0, b, pitch, float(corr["station_step_m"]))
    corridor = M.build_corridor(frame, ev.active, b, pitch, float(corr["half_width_m"]), ev.point_x, ev.point_y, c0.shape)
    state = M.fit_profile_state(frame, c0, b, pitch, float(corr["profile_normal_extent_m"]), float(corr["profile_normal_step_m"]))
    envelope = M.anchored_envelope(corridor, anchor_residual.ravel()[corridor.ridx], len(frame.station_m), config["envelope"])

    recon = M.synthesize(
        c0, ev.active, ev.hard_zero, ev.mapped_face, anchor_residual,
        frame, corridor, state, envelope, b, pitch, config,
    )
    residual = recon.residual_m
    reconstructed = c0 + residual

    # ---------------- gates ----------------
    gcfg = config["gates"]
    # safety exactness
    collar = int(round(float(ev_config["outer_zero_collar_m"]) / pitch))
    collar_vals = np.concatenate([residual[:collar].ravel(), residual[-collar:].ravel(), residual[:, :collar].ravel(), residual[:, -collar:].ravel()])
    safety = {
        "hard_zero_max_abs_m": float(np.max(np.abs(residual[ev.hard_zero]))) if np.any(ev.hard_zero) else 0.0,
        "inactive_max_abs_m": float(np.max(np.abs(residual[~ev.active]))) if np.any(~ev.active) else 0.0,
        "mapped_face_max_abs_m": float(np.max(np.abs(residual[ev.mapped_face]))) if np.any(ev.mapped_face) else 0.0,
        "collar_max_abs_m": float(np.max(np.abs(collar_vals))) if collar_vals.size else 0.0,
    }
    safety["pass"] = all(v == 0.0 for k, v in safety.items() if k.endswith("_max_abs_m"))

    # dc-drift ordinary ground: guard SYNTHESIS-ADDED drift (residual - accepted
    # anchor), NOT the accepted holdout-verified measured relief. The brief's
    # DC-drift guard targets the +7 m synthesis failure mode, "not macro relief".
    og = recon.ordinary_ground
    added = residual - anchor_residual
    og_added_p95 = float(np.percentile(np.abs(added[og]), 95)) if np.any(og) else 0.0
    og_added_max = float(np.max(np.abs(added[og]))) if np.any(og) else 0.0
    og_total_p95 = float(np.percentile(np.abs(residual[og]), 95)) if np.any(og) else 0.0
    dc = {"ordinary_ground_cells": int(np.count_nonzero(og)),
          "synthesis_added_p95_abs_m": og_added_p95, "synthesis_added_max_abs_m": og_added_max,
          "total_incl_accepted_anchor_p95_abs_m": og_total_p95,
          "ceiling_m": float(gcfg["dc_drift_ordinary_ground_p95_ceiling_m"]),
          "pass": og_added_p95 <= float(gcfg["dc_drift_ordinary_ground_p95_ceiling_m"])}

    # anchor fidelity
    af = gcfg["anchor_fidelity"]; tol = float(af["tolerance_m"])
    mine = _holdout(ev, residual); a5 = _holdout(ev, anchor_residual); c0h = _holdout(ev, np.zeros_like(residual))
    # Compare candidate to the accepted ALS/TGV field recomputed under the IDENTICAL
    # in-run holdout protocol (apples-to-apples); the frozen a5d101_* constants come
    # from a5d101's own metrics.json under a different implementation and are kept only
    # as a reference. No-regression = candidate <= a5d101 (in-run) + tol, each side.
    anchor = {"candidate": mine, "a5d101_in_run": a5, "a5d101_frozen_reference": {
        "high": float(af["a5d101_candidate_high_mae_m"]), "low": float(af["a5d101_candidate_low_mae_m"])}, "c0": c0h}
    anchor["high_pass"] = mine["high"]["block_median_mae_m"] <= a5["high"]["block_median_mae_m"] + tol
    anchor["low_pass"] = mine["low"]["block_median_mae_m"] <= a5["low"]["block_median_mae_m"] + tol
    anchor["beats_c0"] = (mine["high"]["block_median_mae_m"] < c0h["high"]["block_median_mae_m"]) and (
        mine["low"]["block_median_mae_m"] < c0h["low"]["block_median_mae_m"])
    anchor["pass"] = anchor["high_pass"] and anchor["low_pass"] and (anchor["beats_c0"] or not bool(af["require_beats_c0"]))

    connect = _connectedness(ev, frame, state, reconstructed, config)
    fineb = _fine_band_metrics(recon, config)
    fbg = gcfg["fine_band_biala_envelope"]
    fineb["pass"] = (
        fineb["face_p95_m"] <= float(fbg["face_p95_ceiling_m"]) and
        fineb["toe_p95_m"] <= float(fbg["toe_p95_ceiling_m"]) and
        fineb["shoulder_p95_m"] <= float(fbg["shoulder_p95_ceiling_m"]) and
        fineb["band_p95_m"] <= float(fbg["band_p95_upper_m"])
    )
    fail = _failure_mode_metrics(ev, frame, state, recon, config)
    fail["pass"] = fail["endpoint_cap_pass"] and fail["normal_rib_pass"]

    # deviation diagnostic (not a gate)
    act = ev.active & ~ev.hard_zero & ~ev.mapped_face
    av = residual[act]
    deviation = {
        "p50_abs_m": float(np.percentile(np.abs(av), 50)), "p95_abs_m": float(np.percentile(np.abs(av), 95)),
        "max_abs_m": float(np.max(np.abs(av))), "changed_over_2cm_fraction": float(np.mean(np.abs(av) > 0.02)),
    }

    all_pass = all([safety["pass"], dc["pass"], anchor["pass"], connect["pass"], fineb["pass"], fail["pass"]])
    status = "development_float_candidate_accepted" if all_pass else "development_float_candidate_rejected"

    measurements = {
        "status": status, "overall_pass": all_pass,
        "spatial": {"bbox_en": list(b), "pitch_m": pitch, "shape": list(residual.shape),
                    "active_nodes": int(np.count_nonzero(ev.active)), "corridor_cells": int(len(corridor.ridx)),
                    "line_length_m": frame.length_m},
        "gates": {"safety_exact": safety, "dc_drift": dc, "anchor_fidelity": anchor,
                  "connectedness": connect, "fine_band": fineb, "failure_modes": fail},
        "deviation_diagnostic": deviation,
        "grammar": recon.diagnostics,
        "provenance": {
            "measured_anchor_sha256": config["measured_anchor_artifact"]["residual"]["sha256"],
            "connective_grammar_sha256": config["connective_grammar_artifact"]["npz"]["sha256"],
            "evidence_config_sha256": ev_row["sha256"], "world_prf_seed": int(config["world_prf_seed"]),
        },
        "credit": {"development_a_only": True, "packing": False, "browser": False, "production": False,
                   "sandstone_carbonate": False, "sentinel_opened": False},
    }

    with tempfile.TemporaryDirectory(prefix="laas-cliff-anchor-") as tmp:
        staging = Path(tmp) / "artifact"
        staging.mkdir()
        _write_npy(staging / "reconstruction-f32.npy", reconstructed)
        _write_npy(staging / "residual-f32.npy", residual)
        (staging / "measurements.json").write_bytes(_canonical(measurements) + b"\n")
        content = hashlib.sha256()
        content.update(hashlib.sha256(config_payload).digest())
        content.update((staging / "reconstruction-f32.npy").read_bytes())
        content.update((staging / "residual-f32.npy").read_bytes())
        content.update((staging / "measurements.json").read_bytes())
        build_id = content.hexdigest()
        final = _ARTIFACT_ROOT / "sha256" / build_id
        if final.exists():
            shutil.rmtree(staging)
            return final / "manifest.json"

        interp = write_qa(staging / "qa", ev, recon, anchor_residual, state, envelope, corridor, measurements["gates"], config)
        qa_rows = []
        for name, meaning in interp.items():
            path = staging / "qa" / name
            with Image.open(path) as image:
                dims = list(image.size)
            qa_rows.append({**_identity(path, f"qa/{name}"), "dimensions": dims, "interpretation": meaning})
        (staging / "qa" / "index.json").write_bytes(_canonical({"images": qa_rows}) + b"\n")

        src = [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("qa.py")]
        manifest = {
            "schema_version": "laas.erodible-slope-cliff-anchor-organization-artifact/1",
            "build_id": build_id, "state": status,
            "config": _identity(config_path),
            "sources": {
                "evidence_config": _identity(ev_config_path),
                "measured_anchor_residual": _identity(anchor_path),
                "connective_grammar_npz": _identity(biala_path),
            },
            "implementation": [_identity(p) for p in src],
            "outputs": {
                "reconstruction": _identity(staging / "reconstruction-f32.npy", "reconstruction-f32.npy"),
                "residual": _identity(staging / "residual-f32.npy", "residual-f32.npy"),
                "measurements": _identity(staging / "measurements.json", "measurements.json"),
                "qa_index": _identity(staging / "qa" / "index.json", "qa/index.json"),
            },
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "pid": os.getpid()},
        }
        (staging / "manifest.json").write_bytes(_canonical(manifest) + b"\n")
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
    return final / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
