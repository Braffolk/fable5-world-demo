"""Hash-bound Development-A evidence on the exact west 128 m canvas."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .....config import ASSET_GEN_ROOT
from ...erodible_slope.morphodynamics.structural_base import (
    FINE_CANVAS_BBOX_EN,
    load_development_a_structural_base,
)
from ..als_tgv.evidence import _sample_domain


TARGET_BBOX_EN = (680448.0, 6444416.0, 680576.0, 6444544.0)
TARGET_GAME_CENTER = (311872.0, 191040.0)
SOLVE_PITCH_M = 0.25
MASTER_PITCH_M = 0.0625


@dataclass(frozen=True)
class Evidence:
    c0_master_m: np.ndarray
    c0_solve_m: np.ndarray
    hard_master: np.ndarray
    hard_solve: np.ndarray
    mapped_face_solve: np.ndarray
    reference_m: np.ndarray
    reference_confidence: np.ndarray
    network_strength: np.ndarray
    source_reconstruction_m: np.ndarray
    source_fine_m: np.ndarray
    source_support: np.ndarray
    source_identities: dict[str, dict[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bound_path(row: dict[str, Any]) -> Path:
    path = ASSET_GEN_ROOT.parent / row["path"]
    if not path.is_file():
        raise ValueError(f"bound evidence is absent: {path}")
    if path.stat().st_size != int(row["bytes"]) or _sha256(path) != row["sha256"]:
        raise ValueError(f"bound evidence differs: {path}")
    return path


def _crop(array: np.ndarray, pitch_m: float, source_bbox: tuple[float, float, float, float]) -> np.ndarray:
    west = int(round((TARGET_BBOX_EN[0] - source_bbox[0]) / pitch_m))
    east = int(round((TARGET_BBOX_EN[2] - source_bbox[0]) / pitch_m))
    north = int(round((source_bbox[3] - TARGET_BBOX_EN[3]) / pitch_m))
    south = int(round((source_bbox[3] - TARGET_BBOX_EN[1]) / pitch_m))
    return np.asarray(array[north : south + 1, west : east + 1])


def _grid(pitch_m: float) -> tuple[np.ndarray, np.ndarray]:
    count = int(round(128.0 / pitch_m)) + 1
    return (
        TARGET_BBOX_EN[0] + np.arange(count, dtype=np.float64) * pitch_m,
        TARGET_BBOX_EN[3] - np.arange(count, dtype=np.float64) * pitch_m,
    )


def load(config: dict[str, Any]) -> Evidence:
    if tuple(float(v) for v in config["target"]["bbox_epsg3301_m"]) != TARGET_BBOX_EN:
        raise ValueError("target bbox differs from the bounded west 128 m canvas")
    if tuple(float(v) for v in config["target"]["game_center_xz_m"]) != TARGET_GAME_CENTER:
        raise ValueError("game center differs")
    paths = {name: bound_path(row) for name, row in config["inputs"].items()}
    closure = json.loads(paths["domain_closure"].read_text(encoding="utf-8"))
    if not (
        closure.get("state") == "closure_pass"
        and closure.get("solver_use_authorized") is True
        and closure.get("canonical_role") == "solver_authorized_condition_authority"
    ):
        raise ValueError("Development-A condition domain is not solver-authorized")
    structural_manifest = json.loads(paths["structural_manifest"].read_text(encoding="utf-8"))
    if structural_manifest.get("build_id") != "ed91840fac7839642c18590eb697c43e914d6dce03c2eb4f58b7eaffaa05ef47":
        raise ValueError("ALS structural evidence identity differs")
    biala_manifest = json.loads(paths["biala_manifest"].read_text(encoding="utf-8"))
    if not (
        biala_manifest.get("build_id") == "9d27b52da864ed51d4e0118545f2537bc9b91c7aba0ef7f5fd63e765ba144cca"
        and biala_manifest.get("qualification", {}).get("source_domain_capacity_only") is True
        and biala_manifest.get("qualification", {}).get("estonia_transfer_authorized") is False
    ):
        raise ValueError("Biala source-domain evidence boundary differs")

    base = load_development_a_structural_base()
    if base.bbox_en != FINE_CANVAS_BBOX_EN or base.texel_m != MASTER_PITCH_M:
        raise ValueError("accepted C0 structural canvas differs")
    c0_master = _crop(base.c0_height_m, MASTER_PITCH_M, base.bbox_en).astype(np.float64)
    structural_hard = _crop(
        base.forbidden_morphology | base.unknown_bathymetry,
        MASTER_PITCH_M,
        base.bbox_en,
    )
    if c0_master.shape != (2049, 2049):
        raise ValueError("absolute master must be the inclusive 0.0625 m target canvas")

    domain_bbox = tuple(float(v) for v in closure["recipe"]["canonical_bbox_en"])
    domain = np.load(paths["domain_npz"], allow_pickle=False)
    master_x, master_y = _grid(MASTER_PITCH_M)
    solve_x, solve_y = _grid(SOLVE_PITCH_M)

    def masks(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        return {
            name: _sample_domain(domain[name], domain_bbox, x, y) >= 0.5
            for name in (
                "valid", "water", "object", "non_heightfield",
                "protected_structure", "unknown", "target_feature",
            )
        }

    master_masks = masks(master_x, master_y)
    solve_masks = masks(solve_x, solve_y)
    lithology_master = _sample_domain(domain["geology_lithology_code"], domain_bbox, master_x, master_y)
    genesis_master = _sample_domain(domain["geology_genesis_code"], domain_bbox, master_x, master_y)
    if np.any(lithology_master != 40) or np.any(genesis_master != 40):
        raise ValueError("canvas is not wholly 40 Liiv / 40 Jaajarvesetted")

    def combine(sampled: dict[str, np.ndarray], structural: np.ndarray) -> np.ndarray:
        return (
            ~sampled["valid"] | sampled["water"] | sampled["object"]
            | sampled["non_heightfield"] | sampled["protected_structure"]
            | sampled["unknown"] | structural
        )

    hard_master = combine(master_masks, structural_hard)
    c0_solve = c0_master[::4, ::4].astype(np.float64)
    hard_solve = combine(solve_masks, structural_hard[::4, ::4])
    collar_solve = int(round(float(config["solver"]["outer_collar_m"]) / SOLVE_PITCH_M))
    collar_master = int(round(float(config["solver"]["outer_collar_m"]) / MASTER_PITCH_M))
    for hard, collar in ((hard_solve, collar_solve), (hard_master, collar_master)):
        hard[:collar] = True
        hard[-collar:] = True
        hard[:, :collar] = True
        hard[:, -collar:] = True

    structural_config = json.loads(paths["structural_config"].read_text(encoding="utf-8"))
    reference_bbox = tuple(float(v) for v in structural_config["canvas_bbox_en"])
    reference = _crop(np.load(paths["reference_surface"], allow_pickle=False), 1.0, reference_bbox)
    confidence = _crop(np.load(paths["reference_confidence"], allow_pickle=False), 1.0, reference_bbox)
    network = _crop(np.load(paths["network_strength"], allow_pickle=False), 1.0, reference_bbox)
    if reference.shape != (129, 129) or confidence.shape != reference.shape:
        raise ValueError("measured ALS graph lattice differs")

    with np.load(paths["biala_capacity"], allow_pickle=False) as source:
        core = np.s_[64:320, 64:320]
        source_reconstruction = source["reconstruction_m"][core].astype(np.float64)
        source_fine = source["fine_component_m"][core].astype(np.float64)
        source_support = source["direct_support"][core].astype(bool)
    fill = float(np.nanmedian(source_reconstruction[source_support]))
    source_reconstruction = np.where(source_support, source_reconstruction, fill)
    source_fine = np.where(source_support, source_fine, 0.0)

    identities = {
        name: {
            "path": config["inputs"][name]["path"],
            "bytes": path.stat().st_size,
            "sha256": config["inputs"][name]["sha256"],
        }
        for name, path in paths.items()
    }
    return Evidence(
        c0_master_m=c0_master,
        c0_solve_m=c0_solve,
        hard_master=hard_master,
        hard_solve=hard_solve,
        mapped_face_solve=solve_masks["target_feature"],
        reference_m=reference.astype(np.float64),
        reference_confidence=confidence.astype(np.float64),
        network_strength=network.astype(np.float64),
        source_reconstruction_m=source_reconstruction,
        source_fine_m=source_fine,
        source_support=source_support,
        source_identities=identities,
    )
