"""Materialize one agriculture v2 development FLOAT preview + gates + QA.

Whole-domain world-coordinate model (partition-invariant); the dev tile (2426,1498)
is a crop for direct v1 comparison, the boundary window straddles the parcel edge for
the taper/boundary QA, and the cross-parcel AOI supplies the direction-variation gate.
STOP after FLOAT + gates + QA. No packing, no base cook, no publish.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ....config import ASSET_GEN_ROOT, DATA_WORK
from . import forms, gates, qa
from .geometry import ParcelField, canonical_axes, load_hard_and_ditch, sample_base
from .model import load_config, sha256_file

CONFIG = ASSET_GEN_ROOT / "config/microtopography/agriculture-v2-cultivated-v1.json"
OUTPUT_ROOT = DATA_WORK / "microtopography/agriculture-network-v2/sha256"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def _build_window(bbox, texel, source, parcel_field, config):
    east, north = canonical_axes(tuple(bbox), texel)
    base = sample_base(source["path"], source["lod0_cx"], source["lod0_cz"], east, north)
    e0, n0, e1, n1 = (int(v) for v in (bbox[0], bbox[1], bbox[2], bbox[3]))
    hard, ditch = load_hard_and_ditch(
        ASSET_GEN_ROOT / config.raw["sources"]["etak"]["path"], (e0, n0, e1, n1), east, north)
    syn, clod_stats = forms.synthesize(east, north, base, parcel_field, config, hard, ditch)
    return syn, clod_stats


def _partition_overlap(dev: forms.Synthesis, win: forms.Synthesis) -> dict:
    """Byte-identical check on the coincident dev-tile / boundary-window overlap."""
    def match(a, b):
        ai = {round(float(v), 6): i for i, v in enumerate(a)}
        pairs = [(ai[round(float(v), 6)], j) for j, v in enumerate(b) if round(float(v), 6) in ai]
        return pairs
    col = match(dev.east, win.east)
    row = match(dev.north, win.north)
    if not col or not row:
        return {"overlap_samples": 0, "byte_identical": False, "note": "no coincident overlap"}
    dcols, wcols = zip(*col)
    drows, wrows = zip(*row)
    d = dev.height[np.ix_(list(drows), list(dcols))]
    w = win.height[np.ix_(list(wrows), list(wcols))]
    da = dev.added[np.ix_(list(drows), list(dcols))]
    wa = win.added[np.ix_(list(wrows), list(wcols))]
    return {
        "overlap_samples": int(d.size),
        "height_byte_identical": bool(np.array_equal(d, w)),
        "added_byte_identical": bool(np.array_equal(da, wa)),
        "max_abs_height_diff": float(np.max(np.abs(d - w))),
        "byte_identical": bool(np.array_equal(d, w) and np.array_equal(da, wa)),
    }


def run(config_path: Path = CONFIG, output_root: Path = OUTPUT_ROOT) -> Path:
    config = load_config(config_path)
    verified = config.verify_sources()
    site = config.site
    texel = config.texel_m
    etak = ASSET_GEN_ROOT / config.raw["sources"]["etak"]["path"]
    source = config.raw["sources"]["corrected_base"]

    aoi = tuple(int(v) for v in site["cross_parcel_aoi_bbox_en"])
    parcel_field = ParcelField(etak, config.raw["sources"]["etak"]["arable_layer"], aoi)
    dev_pid = int(site["dev_parcel_etak_id"])
    dev_theta = parcel_field.infos[dev_pid].theta_rad

    dev, dev_clods = _build_window(site["dev_tile_bbox_en"], texel, source, parcel_field, config)
    win, win_clods = _build_window(site["boundary_window_bbox_en"], texel, source, parcel_field, config)

    partition = _partition_overlap(dev, win)
    gate_results = gates.evaluate(dev, parcel_field, config, dev_pid, dev_theta,
                                  partition["byte_identical"])
    taper = gates.taper_from_window(win, config)
    gate_results["form_free_taper"] = {**taper, "pass": bool(taper["passed"])}

    all_pass = bool(
        gate_results["anti_corduroy"]["pass"]
        and gate_results["scale_break"]["pass"]
        and gate_results["safety_exactness"]["passed"]
        and gate_results["partition_invariance"]["pass"]
        and gate_results["form_free_taper"]["pass"]
    )

    impl_sha = config.implementation_sha256()
    recipe = {
        "config": {"path": str(config_path.relative_to(ASSET_GEN_ROOT.parent)),
                   "sha256": config.config_sha256},
        "implementation_sha256": impl_sha,
        "sources": verified,
        "site": site,
    }
    recipe_sha = hashlib.sha256(_canonical(recipe)).hexdigest()
    output = output_root / recipe_sha
    if output.exists():
        raise FileExistsError(f"immutable agriculture v2 artifact exists: {output}")

    spectrum = gate_results["scale_break"]
    measurements = {
        "schema_version": "laas.agriculture-v2-float-artifact/1",
        "build_id": recipe_sha,
        "status": "float_qa_pending_visual_verdict" if all_pass else "research_rejected_gate_fail",
        "recipe": recipe,
        "marzahn_facts_used": config.marzahn,
        "operation_unknown_mixture": config.mixture,
        "soil_texture": config.raw["soil_texture"],
        "dev_tile": {
            "bbox_en": site["dev_tile_bbox_en"], "shape": list(dev.height.shape),
            "dev_parcel_row_direction_deg": float(np.degrees(dev_theta) % 180.0),
            "added_relief_rms_m": float(np.sqrt(np.mean(dev.added[dev.interior] ** 2))) if dev.interior.any() else 0.0,
            "clods": dev_clods,
        },
        "boundary_window": {"bbox_en": site["boundary_window_bbox_en"],
                            "shape": list(win.height.shape), "clods": win_clods},
        "gates": gate_results,
        "partition_overlap": partition,
        "all_gates_pass": all_pass,
        "authority": {"research_only": True, "target_truth": False,
                      "packing_runtime_material_or_format_authority": False},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pid": os.getpid()},
    }

    with tempfile.TemporaryDirectory(prefix="laas-agri-v2-") as tmp:
        staging = Path(tmp) / "artifact"
        (staging / "qa").mkdir(parents=True)
        images = qa.render(staging / "qa", dev, win, parcel_field, dev_theta, spectrum, gate_results)
        # trim bulky spectrum arrays out of the human-readable measurements copy
        for k in ("small_scale_wavelength_m", "small_scale_power", "large_scale_wavelength_m",
                  "large_scale_power", "combined_wavelength_m", "combined_power"):
            gate_results["scale_break"].pop(k, None)
        np.savez_compressed(
            staging / "agriculture-v2-float.npz",
            measurements_json_u8=np.frombuffer(_canonical(measurements), dtype=np.uint8),
            dev_height=dev.height.astype(np.float32),
            dev_added=dev.added.astype(np.float32),
            dev_rows=dev.rows.astype(np.float32),
            dev_tracks=dev.tracks.astype(np.float32),
            dev_clods=dev.clods.astype(np.float32),
            dev_ids=dev.ids.astype(np.int32),
            window_height=win.height.astype(np.float32),
            window_added=win.added.astype(np.float32),
            window_ids=win.ids.astype(np.int32),
        )
        (staging / "qa" / "index.json").write_bytes(_canonical(
            {"schema_version": "laas.qa-image-index/1", "recipe_sha256": recipe_sha, "images": images}))
        (staging / "measurements.json").write_bytes(_canonical(measurements))
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, output)
    return output


def main() -> int:
    output = run()
    m = json.loads((output / "measurements.json").read_bytes())
    print(output)
    print(json.dumps(m["gates"], indent=2, sort_keys=True)[:4000])
    print("ALL_GATES_PASS", m["all_gates_pass"])
    return 0 if m["all_gates_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
