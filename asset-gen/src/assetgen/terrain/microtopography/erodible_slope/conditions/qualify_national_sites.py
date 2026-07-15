"""Qualify the bounded retained national slope candidates without imagery reads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .....config import ASSET_GEN_ROOT, DATA_IN, DATA_WORK
from .materialize import SiteSpec, _site_artifacts
from .qualify_sites import (
    _CONDITIONS,
    _FORBIDDEN_ETAK_IDS,
    _PROCESS_CONFIG,
    _array_identity,
    _bound_path,
    _evaluate_chunks,
    _identity,
    _read_arrays,
    _target_geometry,
)

_SELECTION_CONFIG = ASSET_GEN_ROOT / "config/evidence/orthophoto-development-sheets-v1.json"
_PRIOR_OPPORTUNITY_RECIPE = (
    DATA_WORK
    / "microtopography/erodible-slope/site-selection/sha256/"
    "07d0e5d40ee4ff9813c4a63c54ded5021ce42dad656ebb41c0bf37580d2eb084/"
    "selection-proposal.json"
)
_SEALED_OOD2_ETAK_ID = 9688702
_CANDIDATES = (
    {
        "etak_id": 1812065,
        "sheet": "65901",
        "outlet_etak_id": 2410694,
        "dtm": DATA_IN / "dem_1m/65901_dtm_1m.tif",
    },
    {
        "etak_id": 1812320,
        "sheet": "63944",
        "outlet_etak_id": 9737242,
        "dtm": DATA_IN / "dem_1m/63944_dtm_1m.tif",
    },
    {
        "etak_id": 1812321,
        "sheet": "63944",
        "outlet_etak_id": 2211505,
        "dtm": DATA_IN / "dem_1m/63944_dtm_1m.tif",
    },
)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _load_condition_authorities() -> tuple[dict[str, Any], dict[str, Any], Path]:
    bundle = json.loads(_CONDITIONS.read_text(encoding="utf-8"))
    inputs = bundle["recipe"]["inputs"]
    authority_path = _bound_path(inputs["accepted_structural_authority"])
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    return bundle, inputs, authority_path


def _positive_control(
    bundle: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    site = next(row for row in bundle["sites"] if row["site_id"] == "development_a")
    geometry, identity = _target_geometry(1826743)
    evaluations = _evaluate_chunks(
        geometry,
        _read_arrays(site["arrays"]),
        _read_arrays(site["enlarged_domain_arrays"]),
        json.loads(_bound_path(site["soil_window"]).read_text(encoding="utf-8")),
        json.loads(
            _bound_path(site["enlarged_domain"]["soil_window"]).read_text(
                encoding="utf-8"
            )
        ),
        site["bbox_en"],
        site["enlarged_domain"]["bbox_en"],
        config,
    )
    passing = [row for row in evaluations if row["passed"]]
    if not passing:
        raise RuntimeError("Development A positive control no longer passes")
    return {"identity": identity, "passing_chunk_evaluations": passing}


def _retained_orthophoto_identities(sheet: str) -> dict[str, Any]:
    # These JSON manifests bind the RGB/CIR observations without opening image pixels.
    root = DATA_IN / "orthophoto"
    result = {
        "sheet_retention": _identity(root / f"development-{sheet}/retained.json")
    }
    for product in ("rgb", "cir"):
        retention = root / product / sheet / "retained.json"
        payload = json.loads(retention.read_text(encoding="utf-8"))
        extraction = root / payload["extraction"]["manifest"]
        result[product] = {
            "retention": _identity(retention),
            "extraction_manifest": _identity(extraction),
            "capture_date": payload["captureDate"],
            "image_pixels_accessed": False,
        }
    return result


def _condition_candidate(
    candidate: dict[str, Any],
    *,
    inputs: dict[str, Any],
    authority_path: Path,
    authority: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    etak_id = int(candidate["etak_id"])
    geometry, identity = _target_geometry(etak_id)
    midpoint = geometry.interpolate(float(geometry.length) * 0.5)
    site_id = f"candidate_{etak_id}"
    result = _site_artifacts(
        SiteSpec(
            site_id=site_id,
            role="morphology_development",
            target_etak_id=etak_id,
            target_e=float(midpoint.x),
            target_n=float(midpoint.y),
            sheet=str(candidate["sheet"]),
            target_outlet_ids=(int(candidate["outlet_etak_id"]),),
        ),
        authority_root=authority_path.parent,
        authority_manifest=authority,
        accepted_materialization_path=_bound_path(
            inputs["accepted_corrected_materialization"]
        ),
        domain_snapshot_path=_bound_path(inputs["egt_domain_snapshot"]),
        forbidden_etak_ids=_FORBIDDEN_ETAK_IDS,
        official_dtm_path=Path(candidate["dtm"]),
    )
    facts = result["facts"]
    obsolete_blocker = f"{site_id}_target_material_support_incomplete"
    active_blockers = [
        blocker for blocker in facts["blockers"] if blocker != obsolete_blocker
    ]
    evaluations: list[dict[str, Any]] = []
    if not active_blockers:
        evaluations = _evaluate_chunks(
            geometry,
            result["arrays"],
            result["enlarged_domain_arrays"],
            json.loads(_bound_path(facts["soil_window"]).read_text(encoding="utf-8")),
            json.loads(
                _bound_path(facts["enlarged_domain"]["soil_window"]).read_text(
                    encoding="utf-8"
                )
            ),
            facts["bbox_en"],
            facts["enlarged_domain"]["bbox_en"],
            config,
        )
    passing = next((row for row in evaluations if row["passed"]), None)
    return {
        "identity": identity,
        "sheet": candidate["sheet"],
        "outlet_etak_id": candidate["outlet_etak_id"],
        "official_dtm": _identity(Path(candidate["dtm"])),
        "orthophoto_observation_identities": _retained_orthophoto_identities(
            str(candidate["sheet"])
        ),
        "condition": {
            "bbox_en": facts["bbox_en"],
            "enlarged_bbox_en": facts["enlarged_domain"]["bbox_en"],
            "soil_window": facts["soil_window"],
            "enlarged_soil_window": facts["enlarged_domain"]["soil_window"],
            "geology_window": facts["geology_window"],
            "enlarged_geology_window": facts["enlarged_domain"]["geology_window"],
            "normal_arrays": _array_identity(result["arrays"]),
            "enlarged_arrays": _array_identity(result["enlarged_domain_arrays"]),
            "reported_blockers": facts["blockers"],
            "ignored_obsolete_target_line_material_blocker": (
                obsolete_blocker in facts["blockers"]
            ),
            "active_blockers": active_blockers,
            "target_material_support": facts["target_material_support"],
        },
        "chunk_evaluations": evaluations,
        "passed": passing is not None,
        "selected_chunk": passing,
    }


def qualify_national_sites(output_root: Path | None = None) -> Path:
    selection_config = json.loads(_SELECTION_CONFIG.read_text(encoding="utf-8"))
    excluded = {int(value) for value in selection_config["excludedEtakIds"]}
    if _SEALED_OOD2_ETAK_ID not in excluded:
        raise ValueError("sealed OOD2 is not source-excluded by the frozen selection")
    if any(int(row["etak_id"]) in excluded for row in _CANDIDATES):
        raise ValueError("frozen candidate set includes a source-excluded ETAK feature")

    bundle, inputs, authority_path = _load_condition_authorities()
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    config = json.loads(_PROCESS_CONFIG.read_text(encoding="utf-8"))
    control = _positive_control(bundle, config)
    evaluations = []
    selected = None
    for candidate in _CANDIDATES:
        evaluated = _condition_candidate(
            candidate,
            inputs=inputs,
            authority_path=authority_path,
            authority=authority,
            config=config,
        )
        evaluations.append(evaluated)
        if evaluated["passed"]:
            selected = evaluated
            break

    if selected is not None:
        raise RuntimeError(
            "a frozen national candidate unexpectedly passed; publish a full condition "
            "bundle rather than a no-pass report"
        )
    if len(evaluations) != len(_CANDIDATES):
        raise RuntimeError("bounded candidate order was not exhausted")

    implementation = Path(__file__)
    recipe = {
        "schema_version": "laas.erodible-slope-national-opportunity-selection/1.recipe",
        "search_order": [int(row["etak_id"]) for row in _CANDIDATES],
        "stop_policy": "stop at first passing candidate; otherwise exhaust only this frozen list",
        "sources": {
            "selection_config": _identity(_SELECTION_CONFIG),
            "prior_corrected_opportunity_recipe": _identity(_PRIOR_OPPORTUNITY_RECIPE),
            "historical_condition_bundle": _identity(_CONDITIONS),
            "process_config": _identity(_PROCESS_CONFIG),
            "etak": _identity(DATA_IN / "etak/ETAK_EESTI_GPKG.gpkg"),
            "accepted_structural_authority": _identity(authority_path),
            "accepted_corrected_materialization": _identity(
                _bound_path(inputs["accepted_corrected_materialization"])
            ),
            "egt_domain_snapshot": _identity(_bound_path(inputs["egt_domain_snapshot"])),
            "materializer": _identity(implementation.with_name("materialize.py")),
            "opportunity": _identity(implementation.with_name("opportunity.py")),
            "qualification_gate": _identity(implementation.with_name("qualify_sites.py")),
            "national_qualification": _identity(implementation),
        },
        "sealed_ood2": {
            "etak_id": _SEALED_OOD2_ETAK_ID,
            "source_excluded": True,
            "accessed": False,
        },
        "imagery_policy": "bind retained RGB/CIR identities only; never open or rank image pixels",
        "candidate_evaluations": evaluations,
        "positive_control": control,
    }
    recipe_sha256 = _canonical_sha256(recipe)
    artifact = {
        "schema_version": "laas.erodible-slope-national-opportunity-no-pass/1",
        "recipe_sha256": recipe_sha256,
        "status": "bounded_frozen_candidate_set_exhausted_without_pass",
        "development_replacement": None,
        "candidate_evaluations": evaluations,
        "positive_control": control,
        "conclusion": (
            "Keep Development A as the only qualified morphology-development site. "
            "Do not weaken gates or expand this frozen search implicitly."
        ),
        "recipe": recipe,
    }
    root = output_root or (
        DATA_WORK / "microtopography/erodible-slope/site-selection/sha256"
    )
    output = root / recipe_sha256 / "selection-report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    os.replace(temporary, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(qualify_national_sites(args.output_root))


if __name__ == "__main__":
    main()
