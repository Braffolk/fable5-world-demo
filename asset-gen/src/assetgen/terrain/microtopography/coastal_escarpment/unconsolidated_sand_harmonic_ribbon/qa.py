"""Exactly four FLOAT-review images for the injective harmonic ribbon."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..unconsolidated_sand_curvilinear_strip.evidence import Evidence
from ..unconsolidated_sand_curvilinear_strip.qa import (
    _close_crop,
    _hillshade,
    _scalar,
    _sheet,
    _signed,
)
from .domain import HarmonicRibbon
from .model import Result


def _domain_scalar(value: np.ndarray, mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    visible = np.where(mask, value, 0.0)
    rgb = _scalar(visible, color)
    rgb[~mask] = np.asarray([0.94, 0.92, 0.85])
    return rgb


def _ownership(evidence: Evidence, ribbon: HarmonicRibbon) -> np.ndarray:
    rgb = _hillshade(evidence.c0_solve_m)
    rgb = rgb * (1.0 - 0.50 * ribbon.mask[..., None]) + np.asarray([0.12, 0.52, 0.44]) * 0.50 * ribbon.mask[..., None]
    boundary = ribbon.boundary
    rgb[boundary] = np.asarray([0.93, 0.33, 0.10])
    rgb[evidence.hard_solve] *= 0.30
    return rgb


def _jacobian(ribbon: HarmonicRibbon) -> np.ndarray:
    gy_u, gx_u = np.gradient(np.nan_to_num(ribbon.u), 0.25)
    gy_v, gx_v = np.gradient(np.nan_to_num(ribbon.v), 0.25)
    return np.abs(gx_u * gy_v - gy_u * gx_v) * ribbon.mask


def write_qa(
    directory: Path,
    evidence: Evidence,
    ribbon: HarmonicRibbon,
    result: Result,
    recipe_sha256: str,
    source_sha256s: list[str],
) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    c0_close = _close_crop(evidence.c0_solve_m, ribbon.mask)
    c1_close = _close_crop(result.solve_absolute_m, ribbon.mask)
    ownership_close = _close_crop(_ownership(evidence, ribbon), ribbon.mask)
    delta_close = _close_crop(result.solve_delta_m, ribbon.mask)
    images = [
        (
            "01-harmonic-domain-injectivity.png",
            _sheet(
                "01 / Proven non-folding harmonic ribbon",
                "One simple physical disk mapped to a convex rectangle; no closest-line ownership.",
                [
                    ("physical ribbon and exact boundary", _ownership(evidence, ribbon)),
                    ("harmonic along-ribbon u", _domain_scalar(ribbon.u, ribbon.mask, (0.10, 0.37, 0.69))),
                    ("harmonic toe-to-crest v", _domain_scalar(ribbon.v, ribbon.mask, (0.10, 0.55, 0.35))),
                    ("absolute UV Jacobian", _domain_scalar(_jacobian(ribbon), ribbon.mask, (0.72, 0.28, 0.08))),
                ],
            ),
            "Physical ribbon, harmonic coordinates, and non-folding Jacobian evidence.",
        ),
        (
            "02-common-light-c0-c1-macro.png",
            _sheet(
                "02 / Common-light harmonic-ribbon macro verdict",
                "Same light, elevation palette, and scale. FLOAT only; no runtime or material rendering.",
                [
                    ("accepted C0", _hillshade(evidence.c0_solve_m)),
                    ("harmonic-ribbon C1", _hillshade(result.solve_absolute_m)),
                    ("signed C1-C0", _signed(result.solve_delta_m)),
                    ("absolute change", _scalar(np.abs(result.solve_delta_m), (0.68, 0.25, 0.09))),
                ],
            ),
            "Whole-canvas common-light C0/C1 macro comparison.",
        ),
        (
            "03-shoulder-bench-recess-rill-toe-closeup.png",
            _sheet(
                "03 / Shoulder-bench-recess-rill-toe closeup",
                "Bounds derive from the injective physical ribbon, not a hand-selected coordinate.",
                [
                    ("C0", _hillshade(c0_close)),
                    ("C1", _hillshade(c1_close)),
                    ("non-folding ribbon ownership", ownership_close),
                    ("signed local change", _signed(delta_close)),
                ],
            ),
            "Close inspection of macro shoulder, bench, recess, connected rill, and toe forms.",
        ),
        (
            "04-uv-paths-transport-and-micro.png",
            _sheet(
                "04 / Measured UV paths, conservative transport, and micro continuation",
                "Paths solve measured valley cost from headcut to toe; fine band is generated, not truth.",
                [
                    ("measured residual in harmonic UV", _signed(result.measured_residual_uv_m)),
                    ("connected headcut-to-toe path field", _scalar(result.path_field_uv, (0.07, 0.31, 0.68))),
                    ("erosion blue / deposition red", _signed(result.deposition_m - result.erosion_m)),
                    ("0.0625-0.5 m subordinate continuation", _signed(result.micro_master_m)),
                ],
            ),
            "Measured UV residual, connected erosion paths, conservative deposition, and fine continuation.",
        ),
    ]
    rows = []
    for filename, image, interpretation in images:
        path = directory / filename
        image.save(path, optimize=True)
        payload = path.read_bytes()
        rows.append(
            {
                "path": filename,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "dimensions_px": [1800, 1400],
                "interpretation": interpretation,
            }
        )
    index = {
        "schema_version": "laas.unconsolidated-sand-harmonic-ribbon-qa/1",
        "recipe_sha256": recipe_sha256,
        "source_sha256s": sorted(source_sha256s),
        "image_count": 4,
        "images": rows,
    }
    (directory / "index.json").write_text(
        json.dumps(index, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    return index
