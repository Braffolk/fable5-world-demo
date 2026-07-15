"""Content-addressed v2 Moore edge-corrected evidence materialization."""
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
from ..evidence.huhola import CLASS_CELL_M, FILL_THRESHOLD_M, classify_plot
from .contracts import (
    CAPACITY_MANIFEST_SHA256,
    CLASSIFIER_SHA256,
    PREREG_SHA256,
    EvidenceV2Contract,
    canonical_json,
    load_evidence_v2_contract,
    sha256_file,
)
from .qa import image_record, render_evidence_qa
from .statistics import (
    CLASS_NAMES,
    MINIMUM_SUPPORT_GROUPS,
    QUANTILE_PROBABILITIES,
    PlotEvidence,
    d4_offsets,
    estimate,
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as target:
        target.write(canonical_json(value))
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _recipe(contract: EvidenceV2Contract) -> dict[str, Any]:
    module_root = Path(__file__).parent
    return {
        "schema_version": "peat-raised-bog-edge-corrected-evidence-recipe/2",
        "checkpoint": "peat-raised-bog-v2-six-group-evidence-gate",
        "bindings": {
            "preregistration": {
                "path": contract.prereg_path.relative_to(ASSET_GEN_ROOT.parent).as_posix(),
                "sha256": PREREG_SHA256,
            },
            "corrected_capacity_artifact": {
                "path": contract.capacity_path.relative_to(ASSET_GEN_ROOT.parent).as_posix(),
                "sha256": CAPACITY_MANIFEST_SHA256,
                "recipe_id": contract.capacity["recipe_id"],
                "result": contract.capacity["result"],
            },
            "moore_materialization": {
                "path": contract.source.materialization_manifest_path.relative_to(
                    ASSET_GEN_ROOT.parent
                ).as_posix(),
                "sha256": contract.source.materialization_manifest_sha256,
                "recipe_id": contract.source.materialization["recipe_id"],
            },
            "exact_classifier": {
                "path": contract.classifier_path.relative_to(ASSET_GEN_ROOT.parent).as_posix(),
                "sha256": CLASSIFIER_SHA256,
            },
        },
        "estimator": {
            "classification_cell_m": CLASS_CELL_M,
            "classes": list(CLASS_NAMES),
            "huhola_threshold_m": FILL_THRESHOLD_M,
            "orbits": "all integer lattice offsets with 0 < dx^2+dy^2 <= 16, canonical major>=minor>=0",
            "d4": "all unique sign, axis-exchange translations; source and target must both be exact valid cells",
            "ordered_transition": "N(i at source,j at target) / sum_j N(i at source,j at target)",
            "indicator_semivariance": "0.5 E[(I_i(target)-I_i(source))(I_j(target)-I_j(source))]",
            "class_fraction": "pooled cells within group then equal-weight groups",
            "relative_height": "height minus its plot valid-cell median, then class-conditional within group",
            "hhdh_margin": "nonnegative distance to the exact active HuHoLa threshold or nested-feature decision",
            "quantile_probabilities": QUANTILE_PROBABILITIES.tolist(),
            "minimum_support_groups": MINIMUM_SUPPORT_GROUPS,
            "leave_one_group_out": "each of eight folds must retain required axial orbits using six of seven training groups",
            "required_axial_orbits_lattice": [[1, 0], [2, 0]],
        },
        "implementation": {
            "modules": {
                path.name: sha256_file(path)
                for path in sorted(module_root.glob("*.py"))
            },
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pillow": PIL.__version__,
        },
        "authority": {
            "descriptive_edge_corrected_evidence_only": True,
            "threshold_tuning": False,
            "whole_form_reinterpretation": False,
            "height_supervision": False,
            "pixelwise_training": False,
            "synthesis": False,
            "estonia_transfer": False,
            "production": False,
            "preview": False,
        },
    }


def _json_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _nested(values: np.ndarray) -> Any:
    if values.ndim == 0:
        return _json_float(float(values)) if np.issubdtype(values.dtype, np.floating) else int(values)
    return [_nested(value) for value in values]


def _load_plots(contract: EvidenceV2Contract, *, log=print) -> tuple[list[PlotEvidence], tuple[str, ...]]:
    result = []
    groups: list[str] = []
    for index, row in enumerate(contract.source.materialization["plots"]):
        if row["group_id"] not in groups:
            groups.append(row["group_id"])
        path = contract.source.materialization_root / row["path"]
        with np.load(path, allow_pickle=False) as arrays:
            classified = classify_plot(
                arrays["source_height_after_demsmooth_m_f64"],
                arrays["source_finite"],
                arrays["source_x_centers_m_f64"],
                arrays["source_y_centers_m_f64"],
            )
        result.append(PlotEvidence(row["group_id"], row["plot_id"], classified))
        log(f"Moore v2 evidence {index + 1}/68: {row['group_id']}/{row['plot_id']}")
    if len(groups) != 8:
        raise ValueError("Moore geographic-group count changed")
    return result, tuple(groups)


def _document(recipe_id: str, groups: tuple[str, ...], result: dict[str, Any]) -> dict[str, Any]:
    orbits = result["orbits"]
    return {
        "schema_version": "peat-raised-bog-edge-corrected-evidence/2",
        "recipe_id": recipe_id,
        "result": result["result"],
        "failures": result["failures"],
        "classes": list(CLASS_NAMES),
        "groups": list(groups),
        "orbits": [list(orbit) for orbit in orbits],
        "orbit_distance_m": [0.5 * float(np.hypot(*orbit)) for orbit in orbits],
        "d4_offsets_dy_dx": [
            [list(offset) for offset in d4_offsets(*orbit)] for orbit in orbits
        ],
        "quantile_probabilities": QUANTILE_PROBABILITIES.tolist(),
        "group_valid_cell_count": _nested(result["valid_counts"]),
        "group_class_cell_count": _nested(result["class_counts"]),
        "group_class_fractions": _nested(result["class_fractions"]),
        "group_balanced_class_fractions": _nested(result["group_balanced_class_fractions"]),
        "group_orbit_valid_endpoint_denominators": _nested(result["endpoint_denominators"]),
        "group_orbit_ordered_pair_counts": _nested(result["pair_counts"]),
        "group_orbit_transition_denominators": _nested(result["transition_denominators"]),
        "group_orbit_ordered_transition_probabilities": _nested(result["transitions"]),
        "group_orbit_ordered_joint_probabilities": _nested(result["joint"]),
        "group_orbit_indicator_auto_cross_semivariances": _nested(result["semivariance"]),
        "group_balanced_ordered_transition_probabilities": _nested(
            result["group_balanced_transitions"]
        ),
        "group_balanced_indicator_auto_cross_semivariances": _nested(
            result["group_balanced_semivariance"]
        ),
        "orbit_ordered_pair_support_group_count": _nested(result["support_by_pair"]),
        "retained_orbits_full_eight_groups": [
            list(orbits[index]) for index in np.flatnonzero(result["retained"])
        ],
        "class_conditional_support_group_count": _nested(result["marginal_support"]),
        "class_conditional_group_quantiles": result["margins"],
        "group_balanced_class_conditional_quantiles": result["group_balanced_quantiles"],
        "leave_one_geographic_group_out": result["folds"],
        "gate": {
            "minimum_support_groups": MINIMUM_SUPPORT_GROUPS,
            "required_axial_orbits_lattice": [[1, 0], [2, 0]],
            "all_eight_leave_one_group_out_folds_required": True,
            "passed": result["result"] == "pass",
            "failure_action": "reject_the_exact_v2_recipe_without_parameter_tuning_or_method_fallback",
        },
        "authority": {
            "evidence_only": True,
            "training": False,
            "synthesis": False,
            "estonia_transfer": False,
            "production": False,
            "preview": False,
        },
    }


def _write(root: Path, *, recipe: dict[str, Any], recipe_id: str, document: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    if hashlib.sha256(canonical_json(recipe)).hexdigest() != recipe_id:
        raise AssertionError("v2 evidence recipe identity is not canonical")
    _atomic_json(root / "recipe.json", recipe)
    _atomic_json(root / "evidence.json", document)
    qa_path = root / "qa" / "01_support_transitions_radial_behavior.png"
    render_evidence_qa(qa_path, recipe_id=recipe_id, document=document, arrays=arrays)
    qa = image_record(
        qa_path,
        "Group class support, the frozen six-group ordered-transition gate, and observed radial same-class behavior; missing typed curves expose rejection rather than being treated as zero.",
    )
    qa["path"] = f"qa/{qa['path']}"
    manifest = {
        "schema_version": "peat-raised-bog-edge-corrected-evidence-manifest/2",
        "status": "complete",
        "result": document["result"],
        "failures": document["failures"],
        "recipe_id": recipe_id,
        "recipe": {"path": "recipe.json", "sha256": sha256_file(root / "recipe.json")},
        "evidence": {"path": "evidence.json", "sha256": sha256_file(root / "evidence.json")},
        "qa": qa,
        "authority": document["authority"],
    }
    _atomic_json(root / "manifest.json", manifest)


def _verify(root: Path, *, recipe: dict[str, Any], recipe_id: str, document: dict[str, Any], arrays: dict[str, np.ndarray]) -> Path:
    with tempfile.TemporaryDirectory(prefix=f".{recipe_id}.replay-") as directory:
        expected = Path(directory)
        _write(expected, recipe=recipe, recipe_id=recipe_id, document=document, arrays=arrays)
        for relative in (
            Path("recipe.json"),
            Path("evidence.json"),
            Path("qa/01_support_transitions_radial_behavior.png"),
            Path("manifest.json"),
        ):
            if not (root / relative).is_file() or (root / relative).read_bytes() != (expected / relative).read_bytes():
                raise ValueError(f"existing v2 evidence artifact is not canonical: {relative}")
    return root / "manifest.json"


def qualify_edge_corrected_evidence(output_root: Path | None = None, *, log=print) -> Path:
    contract = load_evidence_v2_contract()
    recipe = _recipe(contract)
    recipe_id = hashlib.sha256(canonical_json(recipe)).hexdigest()
    plots, groups = _load_plots(contract, log=log)
    result = estimate(plots, groups)
    document = _document(recipe_id, groups, result)
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    base = output_root or (
        DATA_WORK / "microtopography" / "peat-raised-bog" / "edge-corrected-evidence-v2" / "sha256"
    )
    final_root = base / recipe_id
    if (final_root / "manifest.json").exists():
        return _verify(final_root, recipe=recipe, recipe_id=recipe_id, document=document, arrays=arrays)
    temporary = base / f".{recipe_id}.preparing-{os.getpid()}"
    if temporary.exists() or final_root.exists():
        raise ValueError("incomplete v2 evidence transaction requires inspection")
    temporary.mkdir(parents=True)
    _write(temporary, recipe=recipe, recipe_id=recipe_id, document=document, arrays=arrays)
    temporary.replace(final_root)
    log(f"Moore v2 evidence complete: {recipe_id}; result={result['result']}")
    return final_root / "manifest.json"
