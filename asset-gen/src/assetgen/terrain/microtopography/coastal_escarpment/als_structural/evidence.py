"""Bound Development-A ALS, C0, masks, and point-quality fields."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import laspy
import numpy as np
from scipy import ndimage

from ..als_tgv.evidence import _bound, _grid, _line_coordinates, _sample_domain, _target_line
from ...erodible_slope.morphodynamics.structural_base import (
    FINE_CANVAS_BBOX_EN,
    load_development_a_structural_base,
)


@dataclass(frozen=True)
class StructuralInputs:
    bbox_en: tuple[float, float, float, float]
    solve_pitch_m: float
    reference_pitch_m: float
    solve_x: np.ndarray
    solve_y: np.ndarray
    reference_x: np.ndarray
    reference_y: np.ndarray
    c0_solve_m: np.ndarray
    c0_reference_m: np.ndarray
    hard_solve: np.ndarray
    hard_reference: np.ndarray
    safe_reference: np.ndarray
    mapped_face_solve: np.ndarray
    mapped_face_reference: np.ndarray
    point_x: np.ndarray
    point_y: np.ndarray
    point_z: np.ndarray
    point_c0_m: np.ndarray
    point_scan_angle_degrees: np.ndarray
    point_last_return: np.ndarray
    point_overlap: np.ndarray
    point_side: np.ndarray
    point_holdout: np.ndarray
    source_identities: dict[str, dict[str, Any]]


def load_inputs(config: dict[str, Any]) -> StructuralInputs:
    paths = {name: _bound(row) for name, row in config["inputs"].items()}
    import json

    closure = json.loads(paths["domain_closure"].read_text(encoding="utf-8"))
    if not (
        closure.get("state") == "closure_pass"
        and closure.get("solver_use_authorized") is True
        and closure.get("canonical_role") == "solver_authorized_condition_authority"
    ):
        raise ValueError("domain closure is not solver-authorized")
    domain_bbox = tuple(float(v) for v in closure["recipe"]["canonical_bbox_en"])
    domain = np.load(paths["domain_npz"], allow_pickle=False)
    expected_face = int(config["gates"]["mapped_face_source_cells"])
    face_count = int(np.count_nonzero(domain["target_feature"]))
    face_nonheight = int(np.count_nonzero(domain["target_feature"] & domain["non_heightfield"]))
    if (face_count, face_nonheight) != (expected_face, expected_face):
        raise ValueError("mapped face/non-heightfield authority differs")

    bbox = tuple(float(v) for v in config["canvas_bbox_en"])
    if bbox != FINE_CANVAS_BBOX_EN:
        raise ValueError("canvas differs from accepted structural authority")
    solve_pitch = float(config["solve_pitch_m"])
    reference_pitch = float(config["reference_pitch_m"])
    solve_x, solve_y = _grid(bbox, solve_pitch)
    reference_x, reference_y = _grid(bbox, reference_pitch)
    base = load_development_a_structural_base()
    solve_stride = int(round(solve_pitch / base.texel_m))
    reference_stride = int(round(reference_pitch / base.texel_m))
    c0_solve = np.asarray(base.c0_height_m[::solve_stride, ::solve_stride], dtype=np.float32)
    c0_reference = np.asarray(base.c0_height_m[::reference_stride, ::reference_stride], dtype=np.float32)
    if c0_solve.shape != (len(solve_y), len(solve_x)) or c0_reference.shape != (len(reference_y), len(reference_x)):
        raise ValueError("accepted C0 lattice shape differs")

    def masks_at(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        return {
            name: _sample_domain(domain[name], domain_bbox, x, y) >= 0.5
            for name in (
                "valid", "water", "object", "non_heightfield", "protected_structure",
                "unknown", "target_feature",
            )
        }

    solve_masks = masks_at(solve_x, solve_y)
    reference_masks = masks_at(reference_x, reference_y)
    material = (
        (domain["soil_feature_index"] >= 0)
        & (domain["geology_lithology_code"] > 0)
        & (domain["geology_genesis_code"] > 0)
    )
    material_solve = _sample_domain(material, domain_bbox, solve_x, solve_y) >= 0.5
    material_reference = _sample_domain(material, domain_bbox, reference_x, reference_y) >= 0.5
    structural_hard_solve = (
        base.forbidden_morphology[::solve_stride, ::solve_stride]
        | base.unknown_bathymetry[::solve_stride, ::solve_stride]
    )
    structural_hard_reference = (
        base.forbidden_morphology[::reference_stride, ::reference_stride]
        | base.unknown_bathymetry[::reference_stride, ::reference_stride]
    )

    def hard(masks: dict[str, np.ndarray], structural: np.ndarray) -> np.ndarray:
        return (
            ~masks["valid"] | masks["water"] | masks["object"] | masks["non_heightfield"]
            | masks["protected_structure"] | masks["unknown"] | structural
        )

    hard_solve = hard(solve_masks, structural_hard_solve)
    hard_reference = hard(reference_masks, structural_hard_reference)
    collar = int(round(float(config["outer_zero_collar_m"]) / solve_pitch))
    hard_solve[:collar] = True
    hard_solve[-collar:] = True
    hard_solve[:, :collar] = True
    hard_solve[:, -collar:] = True
    reference_collar = int(round(float(config["outer_zero_collar_m"]) / reference_pitch))
    hard_reference[:reference_collar] = True
    hard_reference[-reference_collar:] = True
    hard_reference[:, :reference_collar] = True
    hard_reference[:, -reference_collar:] = True
    safe_reference = ~hard_reference & material_reference

    points = laspy.read(paths["als"])
    qualification = config["als_qualification"]
    selected = (
        (points.classification == int(qualification["classification"]))
        & ~np.asarray(points.withheld, dtype=bool)
        & ~np.asarray(points.synthetic, dtype=bool)
        & (points.x >= bbox[0]) & (points.x <= bbox[2])
        & (points.y >= bbox[1]) & (points.y <= bbox[3])
    )
    px = np.asarray(points.x[selected], dtype=np.float64)
    py = np.asarray(points.y[selected], dtype=np.float64)
    pz = np.asarray(points.z[selected], dtype=np.float64)
    scan_angle = np.asarray(points.scan_angle[selected], dtype=np.float64)
    # LAS 1.4 point formats 6-10 encode scan angle in 0.006-degree units.
    if int(points.header.point_format.id) >= 6:
        scan_angle *= 0.006
    last_return = np.asarray(
        points.return_number[selected] == points.number_of_returns[selected], dtype=bool
    )
    overlap = np.asarray(points.overlap[selected], dtype=bool)
    solve_rows = (bbox[3] - py) / solve_pitch
    solve_cols = (px - bbox[0]) / solve_pitch
    point_c0 = ndimage.map_coordinates(c0_solve, [solve_rows, solve_cols], order=1, mode="nearest")
    reference_rows = np.clip(np.rint((bbox[3] - py) / reference_pitch).astype(np.int64), 0, len(reference_y) - 1)
    reference_cols = np.clip(np.rint((px - bbox[0]) / reference_pitch).astype(np.int64), 0, len(reference_x) - 1)
    qualified = (
        safe_reference[reference_rows, reference_cols]
        & (np.abs(pz - point_c0) <= float(qualification["maximum_reference_residual_m"]))
    )
    px, py, pz, point_c0, scan_angle, last_return, overlap = (
        value[qualified]
        for value in (px, py, pz, point_c0, scan_angle, last_return, overlap)
    )
    line = _target_line(paths["etak"], int(config["target_etak_id"]))
    side, _ = _line_coordinates(line, px, py, c0_solve, bbox, solve_pitch)
    blocks = (
        np.floor((px - bbox[0]) / float(config["holdout_block_m"])).astype(np.int64)
        + np.floor((py - bbox[1]) / float(config["holdout_block_m"])).astype(np.int64)
    )
    holdout = blocks % int(config["holdout_modulus"]) == int(config["holdout_remainder"])
    identities = {
        name: {"path": row["path"], "bytes": paths[name].stat().st_size, "sha256": row["sha256"]}
        for name, row in config["inputs"].items()
    }
    return StructuralInputs(
        bbox_en=bbox,
        solve_pitch_m=solve_pitch,
        reference_pitch_m=reference_pitch,
        solve_x=solve_x,
        solve_y=solve_y,
        reference_x=reference_x,
        reference_y=reference_y,
        c0_solve_m=c0_solve,
        c0_reference_m=c0_reference,
        hard_solve=hard_solve,
        hard_reference=hard_reference,
        safe_reference=safe_reference,
        mapped_face_solve=solve_masks["target_feature"],
        mapped_face_reference=reference_masks["target_feature"],
        point_x=px,
        point_y=py,
        point_z=pz,
        point_c0_m=point_c0,
        point_scan_angle_degrees=scan_angle,
        point_last_return=last_return,
        point_overlap=overlap,
        point_side=np.where(side >= 0.0, 1, -1).astype(np.int8),
        point_holdout=holdout,
        source_identities=identities,
    )
