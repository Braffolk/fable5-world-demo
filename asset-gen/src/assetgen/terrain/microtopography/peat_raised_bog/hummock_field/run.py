"""Materialize the immutable bog v5 organic hummock-hollow FLOAT + gates + ground QA.

Deterministic float64. STOP after float + gates + QA; packing is a separate step
(network_preview.py) gated on the ground-level verdict. Reuses the shared v3 utilities
(masks, hydrology, prf) -- NOT the rejected reaction-diffusion network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pyogrio.raw
import shapely

from .....config import ASSET_GEN_ROOT, DATA_WORK
from ..peat_bog_network import hydrology, masks
from . import field as field_mod
from . import gates as gates_mod
from . import qa as qa_mod

REPO = ASSET_GEN_ROOT.parent
CONFIG = ASSET_GEN_ROOT / "config/microtopography/peat-raised-bog/bundle-preregistration-bog-hummock-v5.json"
OUTPUT_ROOT = DATA_WORK / "microtopography/peat-bog-hummock-v5/sha256"
MIRE_LAYER = "E_306_margala_a"
FLOAT_NAME = "hummock-v5-float.npz"
PITCH = 0.25


def _sha_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            d.update(b)
    return d.hexdigest()


def _canonical(v: Any) -> bytes:
    return (json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _load_mire(etak: Path, member_ids: list[int], geometry_sha256: str) -> shapely.Geometry:
    where = " OR ".join(f"etak_id = {int(v)}" for v in member_ids)
    md, _fids, wkbs, values = pyogrio.raw.read(
        etak, layer=MIRE_LAYER, where=where, columns=["etak_id"], return_fids=True
    )
    if str(md.get("crs")) != "EPSG:3301" or wkbs is None:
        raise ValueError("mire geometry unavailable")
    geometry = shapely.union_all([shapely.force_2d(shapely.from_wkb(bytes(w))) for w in wkbs])
    if hashlib.sha256(geometry.wkb).hexdigest() != geometry_sha256:
        raise ValueError("mire geometry changed")
    return geometry


def run(config_path: Path = CONFIG, output_root: Path = OUTPUT_ROOT) -> Path:
    config = json.loads(config_path.read_bytes())
    if config.get("status") != "development_float_qa_only":
        raise ValueError("unexpected preregistration status")
    for name, decl in config["inputs"].items():
        p = REPO / decl["path"]
        if _sha_file(p) != decl["sha256"]:
            raise ValueError(f"bound input changed: {name}")
    tgt = config["target"]
    thr = config["gate_thresholds"]
    etak = REPO / config["inputs"]["etak"]["path"]
    dtm10 = REPO / config["inputs"]["hydrology_dtm10m"]["path"]
    halo = tuple(int(v) for v in tgt["halo_bounds_en"])
    core_slice = (slice(*tgt["core_slice_in_halo"][0]), slice(*tgt["core_slice_in_halo"][1]))

    mire = _load_mire(etak, tgt["member_etak_ids"], tgt["mire_geometry_sha256"])
    ms = masks.build(etak, mire, halo, PITCH)

    hy = hydrology.compute(dtm10, tuple(tgt["whole_mire_bounds_en"]), halo, PITCH)
    slope_core = hy.slope_mag[core_slice]
    slope_p95 = float(np.percentile(slope_core, 95))

    hf = field_mod.synthesize(
        halo_bbox=halo, core_slice=core_slice, pitch=PITCH,
        authority_halo=ms.authority, open_water_halo=ms.open_water, slope_core_p95=slope_p95,
    )
    result = gates_mod.evaluate(
        hf.relief_core, hf.authority_core, hf.open_water_core, hf.pool_dist_core, PITCH, thr,
    )

    # ---------------------------------------------------------------- artifact identity
    relief = np.ascontiguousarray(hf.relief_core, dtype=np.float64)
    build_id = hashlib.sha256(b"bog-hummock-v5\0" + relief.tobytes()).hexdigest()
    root = output_root / build_id
    (root / "qa").mkdir(parents=True, exist_ok=True)

    np.savez(
        root / FLOAT_NAME,
        core_relief_025m=relief,
        core_authority_025m=hf.authority_core,
        core_open_water_025m=hf.open_water_core,
        core_pool_dist_025m=hf.pool_dist_core,
    )

    # ---------------------------------------------------------------- ground-level QA
    qa_dir = root / "qa"
    qa_mod.hillshade(relief, PITCH, qa_dir / "hillshade_topdown.png")
    qa_mod.render_oblique(relief, hf.open_water_core, PITCH, qa_dir / "oblique_A_look_north.png",
                          eye_xyz=(64, 4, 1.6), look_xyz=(64, 120, -0.2), sun_az_deg=100, sun_el_deg=13)
    qa_mod.render_oblique(relief, hf.open_water_core, PITCH, qa_dir / "oblique_B_diagonal.png",
                          eye_xyz=(20, 20, 1.4), look_xyz=(110, 105, -0.3), sun_az_deg=150, sun_el_deg=12)
    qa_mod.render_oblique(np.zeros_like(relief), hf.open_water_core, PITCH, qa_dir / "before_base_flat.png",
                          eye_xyz=(64, 4, 1.6), look_xyz=(64, 120, -0.2), sun_az_deg=100, sun_el_deg=13)
    qa_mod.render_oblique(relief, hf.open_water_core, PITCH, qa_dir / "after_hummock.png",
                          eye_xyz=(64, 4, 1.6), look_xyz=(64, 120, -0.2), sun_az_deg=100, sun_el_deg=13)

    measurements = {
        "schema": "laas.peat-bog-hummock-v5-measurements/1",
        "buildId": build_id,
        "floatName": FLOAT_NAME,
        "mechanism": config["mechanism"],
        "real_morphometry": config["real_morphometry"],
        "target": tgt,
        "prf_person": config["prf_person"],
        "site_measured": {
            "core_slope_p95_m_per_m": slope_p95,
            "core_open_water_cells": int(hf.open_water_core.sum()),
            "core_authority_cells": int(hf.authority_core.sum()),
            "hummock_site_count": hf.site_count,
        },
        "gates": result,
        "field_meta": hf.meta,
        "inputs": config["inputs"],
        "source_base_manifest": config["source_base_manifest"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    (root / "measurements.json").write_bytes(_canonical(measurements))
    print(_canonical({"buildId": build_id, "passed": result["passed"], "root": root.as_posix()}).decode(), end="")
    return root


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = ap.parse_args()
    run(args.config, args.output_root)


if __name__ == "__main__":
    _main()
