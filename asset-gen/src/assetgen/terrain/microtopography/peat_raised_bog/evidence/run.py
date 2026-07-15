"""Content-addressed Moore whole-form descriptor capacity checkpoint."""
from __future__ import annotations

import hashlib
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import scipy

from .....config import ASSET_GEN_ROOT, DATA_WORK
from .contracts import (
    CapacityContract,
    canonical_json,
    load_capacity_contract,
    sha256_file,
)
from .forms import extract_whole_forms
from .huhola import (
    CLASS_CELL_M,
    FILL_THRESHOLD_M,
    SOURCE_CELL_M,
    classify_plot,
)
from .qa import image_record, render_capacity_qa


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    payload = canonical_json(value)
    with temporary.open("wb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _recipe(contract: CapacityContract) -> dict[str, Any]:
    module_root = Path(__file__).parent
    return {
        "schema_version": "moore-whole-form-capacity-recipe/1",
        "checkpoint": "peat-raised-bog-moore-whole-form-capacity-v1",
        "preregistration": {
            "path": contract.prereg_path.relative_to(ASSET_GEN_ROOT.parent).as_posix(),
            "sha256": contract.prereg_sha256,
            "bundle_id": contract.prereg["bundle_id"],
            "preregistration_id": contract.prereg["preregistration_id"],
        },
        "moore_materialization": {
            "path": contract.materialization_manifest_path.relative_to(
                ASSET_GEN_ROOT.parent
            ).as_posix(),
            "sha256": contract.materialization_manifest_sha256,
            "build_id": contract.materialization["recipe_id"],
            "archive_sha256": contract.materialization["archive_sha256"],
        },
        "classification": {
            "source_cell_m": SOURCE_CELL_M,
            "classification_cell_m": CLASS_CELL_M,
            "aggregation": "fixed 50x50 float64 area mean; all 2500 finite cells required",
            "fill": "deterministic 8-neighbor priority flood; no flat fixing",
            "hhdh": "(filled_dem-dem)-(filled_inverted_dem-inverted_dem)",
            "three_class_threshold_m": FILL_THRESHOLD_M,
            "nested_feature_rule": "larger of positive hollow fill and positive hummock fill wins",
            "connectivity": 8,
            "boundary_erosion_cells": 1,
            "boundary_component_credit": False,
            "padding_interpolation_reflection_credit": False,
        },
        "marks": {
            "footprint": "typed 8-connected 0.50 m component",
            "axes": "area-conserving equivalent ellipse from cell-area-corrected covariance",
            "orientation": "defined only for anisotropic forms; major axis east-counterclockwise with deterministic half-plane sign",
            "asymmetry": "unweighted major-axis coordinate skewness",
            "relative_relief": "native measured source height relative to adjacent typed-cell median",
            "shoulder_profile": "five equal-count native-sample strata ordered boundary-to-core by Euclidean depth",
            "spacing": "within-plot nearest centroid and equivalent-radius edge gap, with explicit right censoring",
            "adjacency": "distinct 8-neighbor typed components",
        },
        "gate": contract.prereg["descriptor_capacity_gate"],
        "implementation": {
            "modules": {
                path.name: sha256_file(path)
                for path in sorted(module_root.glob("*.py"))
            },
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pillow": PIL.__version__,
            "huhola_implementation_sha256": contract.huhola_implementation_sha256,
        },
        "authority": {
            "descriptive_forms_only": True,
            "height_supervision": False,
            "pixelwise_training": False,
            "synthesis": False,
            "estonia_transfer": False,
            "production": False,
            "preview": False,
        },
    }


def _assert_complete_marks(forms: list[dict[str, Any]]) -> None:
    required = {
        "topographic_sign",
        "class",
        "relative_relief_m",
        "footprint_area_m2",
        "major_axis_m",
        "minor_axis_m",
        "anisotropy_ratio",
        "orientation_defined",
        "orientation_rad_east_ccw",
        "major_axis_asymmetry_skewness",
        "shoulder_profile_boundary_to_core_quintile_mean_m",
        "nearest_same_class_observed",
        "nearest_same_class_centroid_m",
        "nearest_same_class_edge_m",
        "nearest_cross_class_observed",
        "nearest_cross_class_centroid_m",
        "nearest_cross_class_edge_m",
        "adjacent_component_count",
        "adjacent_classes",
        "spacing_censor_radius_m",
    }
    for form in forms:
        missing = required - set(form)
        if missing:
            raise AssertionError(f"incomplete whole-form marks for {form['form_id']}: {sorted(missing)}")
        shoulder = form["shoulder_profile_boundary_to_core_quintile_mean_m"]
        if len(shoulder) != 5 or not np.all(np.isfinite(shoulder)):
            raise AssertionError(f"invalid shoulder profile for {form['form_id']}")
        orientation_defined = bool(form["orientation_defined"])
        orientation = form["orientation_rad_east_ccw"]
        if orientation_defined != (orientation is not None):
            raise AssertionError(f"invalid orientation support mark for {form['form_id']}")
        for prefix in ("same_class", "cross_class"):
            observed = bool(form[f"nearest_{prefix}_observed"])
            values = (
                form[f"nearest_{prefix}_centroid_m"],
                form[f"nearest_{prefix}_edge_m"],
                form[f"nearest_{prefix}_form_id"],
            )
            if observed != all(value is not None for value in values):
                raise AssertionError(f"invalid censored spacing mark for {form['form_id']}")


def _compute_capacity(contract: CapacityContract, *, log=print) -> dict[str, Any]:
    """Derive the decisive capacity result directly from the pinned Moore sources."""
    all_forms: list[dict[str, Any]] = []
    plot_rows: list[dict[str, Any]] = []
    representatives: dict[str, tuple[str, Any, list[dict[str, Any]]]] = {}
    representative_scores: dict[str, tuple[int, int, str]] = {}
    for index, row in enumerate(contract.materialization["plots"]):
        with np.load(contract.materialization_root / row["path"], allow_pickle=False) as arrays:
            required = {
                "source_height_after_demsmooth_m_f64",
                "source_finite",
                "source_x_centers_m_f64",
                "source_y_centers_m_f64",
            }
            if not required.issubset(arrays.files):
                raise ValueError(f"materialized Moore plot lacks source arrays: {row['path']}")
            source_height = arrays["source_height_after_demsmooth_m_f64"].copy()
            source_valid = arrays["source_finite"].copy()
            source_x = arrays["source_x_centers_m_f64"].copy()
            source_y = arrays["source_y_centers_m_f64"].copy()
        if list(source_height.shape) != row["source_shape_yx"]:
            raise ValueError(f"materialized Moore source shape changed: {row['path']}")
        classified = classify_plot(source_height, source_valid, source_x, source_y)
        forms, plot_summary = extract_whole_forms(
            group_id=row["group_id"],
            plot_id=row["plot_id"],
            classified=classified,
            source_height_m=source_height,
            source_valid=source_valid,
        )
        all_forms.extend(forms)
        plot_summary["source_artifact_path"] = row["path"]
        plot_summary["source_artifact_sha256"] = row["sha256"]
        plot_rows.append(plot_summary)
        score = (
            len(forms),
            int(np.count_nonzero(classified.valid)),
            str(row["plot_id"]),
        )
        if score > representative_scores.get(row["group_id"], (-1, -1, "")):
            representative_scores[row["group_id"]] = score
            representatives[row["group_id"]] = (row["plot_id"], classified, forms)
        log(
            f"Moore capacity {index + 1}/68: {row['group_id']}/{row['plot_id']} "
            f"whole={len(forms)} valid_50cm={plot_summary['classification_cells']}"
        )

    _assert_complete_marks(all_forms)
    group_order = list(dict.fromkeys(row["group_id"] for row in contract.materialization["plots"]))
    group_rows: list[dict[str, Any]] = []
    for group_id in group_order:
        group_forms = [form for form in all_forms if form["group_id"] == group_id]
        group_rows.append(
            {
                "group_id": group_id,
                "plot_count": sum(row["group_id"] == group_id for row in plot_rows),
                "whole_form_count": len(group_forms),
                "whole_form_count_by_class": {
                    class_name: sum(form["class"] == class_name for form in group_forms)
                    for class_name in ("lawn", "hollow", "hummock")
                },
                "boundary_rejected_component_count": sum(
                    sum(row["boundary_rejected_components_by_class"].values())
                    for row in plot_rows
                    if row["group_id"] == group_id
                ),
            }
        )
    gate_config = contract.prereg["descriptor_capacity_gate"]
    minimum_forms = int(gate_config["minimum_forms"])
    minimum_groups = int(gate_config["minimum_geographic_groups"])
    used_groups = sum(row["whole_form_count"] > 0 for row in group_rows)
    result = "pass" if len(all_forms) >= minimum_forms and used_groups >= minimum_groups else "reject"
    failures = []
    if len(all_forms) < minimum_forms:
        failures.append("WHOLE_FORM_COUNT_BELOW_200")
    if used_groups < minimum_groups:
        failures.append("GEOGRAPHIC_GROUP_COUNT_BELOW_6")
    return {
        "forms": all_forms,
        "plots": plot_rows,
        "groups": group_rows,
        "representatives": representatives,
        "minimum_forms": minimum_forms,
        "minimum_groups": minimum_groups,
        "observed_forms": len(all_forms),
        "observed_groups": used_groups,
        "result": result,
        "failures": failures,
        "failure_action": gate_config["failure_action"] if failures else None,
    }


def _write_expected_artifact(
    root: Path,
    *,
    contract: CapacityContract,
    recipe: dict[str, Any],
    recipe_id: str,
    capacity: dict[str, Any],
) -> None:
    """Write the sole canonical artifact implied by the recipe and pinned sources."""
    recipe_bytes = canonical_json(recipe)
    if hashlib.sha256(recipe_bytes).hexdigest() != recipe_id:
        raise AssertionError("Moore form-capacity recipe identity is not canonical")

    forms_document = {
        "schema_version": "moore-whole-form-descriptors/1",
        "recipe_id": recipe_id,
        "joint_marks_complete": True,
        "spacing_absence_is_right_censored": True,
        "plots": capacity["plots"],
        "groups": capacity["groups"],
        "forms": capacity["forms"],
    }
    _atomic_json(root / "recipe.json", recipe)
    _atomic_json(root / "forms.json", forms_document)
    qa_path = root / "qa" / "02_moore_form_capacity_and_groups.png"
    render_capacity_qa(
        qa_path,
        recipe_id=recipe_id,
        result=capacity["result"],
        group_rows=capacity["groups"],
        representative_plots=capacity["representatives"],
        minimum_forms=capacity["minimum_forms"],
        minimum_groups=capacity["minimum_groups"],
    )
    qa = [
        image_record(
            qa_path,
            "Whole HuHoLa form capacity by independent Moore geographic group; dark plot margins are excluded and white rings are retained components.",
        )
    ]
    qa_index = {
        "schema_version": "peat-raised-bog-capacity-qa/1",
        "recipe_id": recipe_id,
        "preregistration_sha256": contract.prereg_sha256,
        "moore_materialization_manifest_sha256": contract.materialization_manifest_sha256,
        "source_recipe_sha256": sha256_file(root / "recipe.json"),
        "authority": recipe["authority"],
        "images": qa,
    }
    _atomic_json(root / "qa" / "index.json", qa_index)
    manifest = {
        "schema_version": "moore-whole-form-capacity/1",
        "status": "complete",
        "result": capacity["result"],
        "failure_action": capacity["failure_action"],
        "failures": capacity["failures"],
        "recipe_id": recipe_id,
        "recipe": {
            "path": "recipe.json",
            "sha256": sha256_file(root / "recipe.json"),
            "bytes": (root / "recipe.json").stat().st_size,
        },
        "forms": {
            "path": "forms.json",
            "sha256": sha256_file(root / "forms.json"),
            "bytes": (root / "forms.json").stat().st_size,
        },
        "qa_index": {
            "path": "qa/index.json",
            "sha256": sha256_file(root / "qa" / "index.json"),
            "bytes": (root / "qa" / "index.json").stat().st_size,
        },
        "qa": qa,
        "gate": {
            "minimum_forms": capacity["minimum_forms"],
            "observed_forms": capacity["observed_forms"],
            "minimum_geographic_groups": capacity["minimum_groups"],
            "observed_geographic_groups": capacity["observed_groups"],
            "joint_marks_complete": True,
            "passed": not capacity["failures"],
        },
        "groups": capacity["groups"],
        "authority": recipe["authority"],
    }
    _atomic_json(root / "manifest.json", manifest)


def _verify_existing(
    manifest_path: Path,
    *,
    contract: CapacityContract,
    recipe: dict[str, Any],
    recipe_id: str,
    capacity: dict[str, Any],
) -> Path:
    """Rebuild the expected artifact and require byte-identical replay."""
    with tempfile.TemporaryDirectory(prefix=f".{recipe_id}.replay-") as replay_directory:
        expected_root = Path(replay_directory)
        _write_expected_artifact(
            expected_root,
            contract=contract,
            recipe=recipe,
            recipe_id=recipe_id,
            capacity=capacity,
        )
        actual_root = manifest_path.parent
        for relative in (
            Path("recipe.json"),
            Path("forms.json"),
            Path("qa/index.json"),
            Path("qa/02_moore_form_capacity_and_groups.png"),
            Path("manifest.json"),
        ):
            actual = actual_root / relative
            expected = expected_root / relative
            if not actual.is_file() or actual.read_bytes() != expected.read_bytes():
                raise ValueError(
                    f"existing Moore form-capacity artifact is not canonical: {relative}"
                )
    return manifest_path


def qualify_moore_form_capacity(output_root: Path | None = None, *, log=print) -> Path:
    contract = load_capacity_contract()
    recipe = _recipe(contract)
    recipe_id = hashlib.sha256(canonical_json(recipe)).hexdigest()
    capacity = _compute_capacity(contract, log=log)
    base = output_root or (
        DATA_WORK
        / "microtopography"
        / "peat-raised-bog"
        / "moore-form-capacity"
        / "sha256"
    )
    final_root = base / recipe_id
    manifest_path = final_root / "manifest.json"
    if manifest_path.exists():
        return _verify_existing(
            manifest_path,
            contract=contract,
            recipe=recipe,
            recipe_id=recipe_id,
            capacity=capacity,
        )
    temporary = base / f".{recipe_id}.preparing-{os.getpid()}"
    if temporary.exists() or final_root.exists():
        raise ValueError("incomplete Moore form-capacity transaction requires inspection")
    temporary.mkdir(parents=True)
    _write_expected_artifact(
        temporary,
        contract=contract,
        recipe=recipe,
        recipe_id=recipe_id,
        capacity=capacity,
    )
    temporary.replace(final_root)
    log(
        f"Moore whole-form capacity complete: {recipe_id}; result={capacity['result']}; "
        f"forms={capacity['observed_forms']}; groups={capacity['observed_groups']}"
    )
    return final_root / "manifest.json"


if __name__ == "__main__":
    print(qualify_moore_form_capacity())
