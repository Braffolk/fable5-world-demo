"""Strict evidence binding for the bounded curvilinear-strip attempt."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from shapely import wkb

from .....config import ASSET_GEN_ROOT
from ...erodible_slope.morphodynamics.structural_base import (
    FINE_CANVAS_BBOX_EN,
    load_development_a_structural_base,
)
from ..als_tgv.evidence import _sample_domain


TARGET_BBOX_EN = (680448.0, 6444416.0, 680576.0, 6444544.0)
MASTER_PITCH_M = 0.0625
SOLVE_PITCH_M = 0.25


@dataclass(frozen=True)
class Evidence:
    c0_master_m: np.ndarray
    c0_solve_m: np.ndarray
    hard_master: np.ndarray
    hard_solve: np.ndarray
    mapped_face_solve: np.ndarray
    reference_m: np.ndarray
    reference_confidence: np.ndarray
    etak_line: Any
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
    if not path.is_file() or path.stat().st_size != int(row["bytes"]) or _sha256(path) != row["sha256"]:
        raise ValueError(f"bound evidence differs: {path}")
    return path


def _grid(pitch_m: float) -> tuple[np.ndarray, np.ndarray]:
    count = int(round(128.0 / pitch_m)) + 1
    return (
        TARGET_BBOX_EN[0] + np.arange(count, dtype=np.float64) * pitch_m,
        TARGET_BBOX_EN[3] - np.arange(count, dtype=np.float64) * pitch_m,
    )


def _crop(array: np.ndarray, pitch_m: float, bbox: tuple[float, float, float, float]) -> np.ndarray:
    c0 = int(round((TARGET_BBOX_EN[0] - bbox[0]) / pitch_m))
    c1 = int(round((TARGET_BBOX_EN[2] - bbox[0]) / pitch_m))
    r0 = int(round((bbox[3] - TARGET_BBOX_EN[3]) / pitch_m))
    r1 = int(round((bbox[3] - TARGET_BBOX_EN[1]) / pitch_m))
    return np.asarray(array[r0 : r1 + 1, c0 : c1 + 1])


def _target_line(path: Path, target_id: int):
    document = json.loads(path.read_text(encoding="utf-8"))
    records = [row for row in document["records"] if row["attributes"].get("etak_id") == target_id]
    if len(records) != 1:
        raise ValueError("ETAK target is not unique")
    line = wkb.loads(bytes.fromhex(records[0]["geometry"]["wkb_hex"]))
    if line.geom_type != "LineString" or not line.is_simple:
        raise ValueError("ETAK target is not a simple line")
    from shapely.geometry import box

    clipped = line.intersection(box(*TARGET_BBOX_EN))
    if clipped.geom_type != "LineString" or clipped.length < 80.0:
        raise ValueError("target canvas lacks one adequate topology chain")
    return clipped


def load(config: dict[str, Any]) -> Evidence:
    if tuple(config["target"]["bbox_epsg3301_m"]) != TARGET_BBOX_EN or config["target"]["etak_id"] != 1826743:
        raise ValueError("target identity differs")
    paths = {name: bound_path(row) for name, row in config["inputs"].items()}
    closure = json.loads(paths["domain_closure"].read_text(encoding="utf-8"))
    if not (
        closure.get("state") == "closure_pass"
        and closure.get("solver_use_authorized") is True
        and closure.get("canonical_role") == "solver_authorized_condition_authority"
    ):
        raise ValueError("condition domain is not solver-authorized")
    structural = json.loads(paths["structural_manifest"].read_text(encoding="utf-8"))
    if structural.get("build_id") != "ed91840fac7839642c18590eb697c43e914d6dce03c2eb4f58b7eaffaa05ef47":
        raise ValueError("ALS structural artifact differs")
    biala = json.loads(paths["biala_manifest"].read_text(encoding="utf-8"))
    if not (
        biala.get("build_id") == "9d27b52da864ed51d4e0118545f2537bc9b91c7aba0ef7f5fd63e765ba144cca"
        and biala["qualification"]["source_domain_capacity_only"] is True
        and biala["qualification"]["estonia_transfer_authorized"] is False
    ):
        raise ValueError("Biala evidence boundary differs")

    base = load_development_a_structural_base()
    if base.bbox_en != FINE_CANVAS_BBOX_EN or base.texel_m != MASTER_PITCH_M:
        raise ValueError("accepted structural base differs")
    c0_master = _crop(base.c0_height_m, MASTER_PITCH_M, base.bbox_en).astype(np.float64)
    structural_hard = _crop(base.forbidden_morphology | base.unknown_bathymetry, MASTER_PITCH_M, base.bbox_en)
    if c0_master.shape != (2049, 2049):
        raise ValueError("master lattice differs")

    domain_bbox = tuple(float(v) for v in closure["recipe"]["canonical_bbox_en"])
    domain = np.load(paths["domain_npz"], allow_pickle=False)

    def sampled_masks(pitch: float) -> dict[str, np.ndarray]:
        x, y = _grid(pitch)
        return {
            name: _sample_domain(domain[name], domain_bbox, x, y) >= 0.5
            for name in ("valid", "water", "object", "non_heightfield", "protected_structure", "unknown", "target_feature")
        }

    master_masks = sampled_masks(MASTER_PITCH_M)
    solve_masks = sampled_masks(SOLVE_PITCH_M)
    master_x, master_y = _grid(MASTER_PITCH_M)
    lithology = _sample_domain(domain["geology_lithology_code"], domain_bbox, master_x, master_y)
    genesis = _sample_domain(domain["geology_genesis_code"], domain_bbox, master_x, master_y)
    if np.any(lithology != 40) or np.any(genesis != 40):
        raise ValueError("target is not wholly 40 Liiv / 40 Jaajarvesetted")

    def hard(masks: dict[str, np.ndarray], structural_mask: np.ndarray) -> np.ndarray:
        return (
            ~masks["valid"] | masks["water"] | masks["object"] | masks["non_heightfield"]
            | masks["protected_structure"] | masks["unknown"] | structural_mask
        )

    hard_master = hard(master_masks, structural_hard)
    hard_solve = hard(solve_masks, structural_hard[::4, ::4])
    for mask, pitch in ((hard_master, MASTER_PITCH_M), (hard_solve, SOLVE_PITCH_M)):
        collar = int(round(float(config["strip"]["outer_collar_m"]) / pitch))
        mask[:collar] = True
        mask[-collar:] = True
        mask[:, :collar] = True
        mask[:, -collar:] = True

    structural_config = json.loads(paths["structural_config"].read_text(encoding="utf-8"))
    reference_bbox = tuple(float(v) for v in structural_config["canvas_bbox_en"])
    reference = _crop(np.load(paths["reference_surface"], allow_pickle=False), 1.0, reference_bbox)
    confidence = _crop(np.load(paths["reference_confidence"], allow_pickle=False), 1.0, reference_bbox)
    if reference.shape != (129, 129):
        raise ValueError("ALS reference crop differs")
    with np.load(paths["biala_capacity"], allow_pickle=False) as source:
        core = np.s_[64:320, 64:320]
        source_fine = source["fine_component_m"][core].astype(np.float64)
        source_support = source["direct_support"][core].astype(bool)
    source_fine = np.where(source_support, source_fine, 0.0)

    identities = {
        name: {"path": config["inputs"][name]["path"], "bytes": path.stat().st_size, "sha256": config["inputs"][name]["sha256"]}
        for name, path in paths.items()
    }
    return Evidence(
        c0_master_m=c0_master,
        c0_solve_m=c0_master[::4, ::4].copy(),
        hard_master=hard_master,
        hard_solve=hard_solve,
        mapped_face_solve=solve_masks["target_feature"],
        reference_m=reference.astype(np.float64),
        reference_confidence=confidence.astype(np.float64),
        etak_line=_target_line(paths["etak"], 1826743),
        source_fine_m=source_fine,
        source_support=source_support,
        source_identities=identities,
    )
