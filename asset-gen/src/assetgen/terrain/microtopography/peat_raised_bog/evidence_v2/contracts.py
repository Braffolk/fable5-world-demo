"""Fail-closed bindings for the preregistered v2 Moore evidence gate."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .....config import ASSET_GEN_ROOT
from ..evidence.contracts import CapacityContract, load_capacity_contract


REPOSITORY_ROOT = ASSET_GEN_ROOT.parent
PREREG_RELATIVE = Path(
    "asset-gen/config/microtopography/peat-raised-bog/bundle-preregistration-v2.json"
)
PREREG_SHA256 = "431b269d49f7e85457a531cc669a3422e8eee077a96958d9beb9dc7c42da0d35"
CAPACITY_RELATIVE = Path(
    "asset-gen/data/work/microtopography/peat-raised-bog/moore-form-capacity/sha256/"
    "cc811bc0425a9762f9afc978a65039f673069e6632db0cf9361ed379d7738c83/manifest.json"
)
CAPACITY_BUILD_ID = "cc811bc0425a9762f9afc978a65039f673069e6632db0cf9361ed379d7738c83"
CAPACITY_MANIFEST_SHA256 = "ecf647306241496a4d831223ffd6cfe0881f8b07607d416e9cd0b06def0d3cd9"
CLASSIFIER_SHA256 = "38f6e8697eccebeafa76ac52cd5dd7d53b7973f03dc54549f48479dd59fa586c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


@dataclass(frozen=True)
class EvidenceV2Contract:
    prereg_path: Path
    prereg: dict[str, Any]
    capacity_path: Path
    capacity: dict[str, Any]
    capacity_recipe: dict[str, Any]
    source: CapacityContract
    classifier_path: Path


def _require_v2(value: dict[str, Any]) -> None:
    if value.get("schema_version") != "laas.peat-raised-bog-r0-research-bundle-preregistration/2":
        raise ValueError("raised-bog v2 preregistration schema changed")
    if value.get("preregistration_id") != "open-intact-nonforested-edge-corrected-plurigaussian-v2":
        raise ValueError("raised-bog v2 method changed")
    history = value.get("immutable_history", {}).get("moore_whole_form_capacity", {})
    if history.get("build_id") != CAPACITY_BUILD_ID or history.get("result") != "reject":
        raise ValueError("v2 no longer preserves the rejected whole-form history")
    evidence = value.get("evidence_estimation", {})
    expected = {
        "classification_cell_m": 0.5,
        "classes": ["hollow", "lawn", "hummock"],
        "independent_unit": "geographic_group",
        "group_weighting": "each of the eight Moore geographic groups receives equal weight",
        "candidate_offset_orbits": "all integer 0.5 m lattice offsets with 0 < dx^2 + dy^2 <= 16 collapsed by sign swap and axis exchange",
        "support_rule": "retain an orbit only when every class-pair denominator is nonzero in at least six geographic groups; require retained 0.5 m and 1.0 m axial orbits or reject",
    }
    for key, expected_value in expected.items():
        if evidence.get(key) != expected_value:
            raise ValueError(f"v2 evidence contract changed: {key}")
    if evidence.get("fit", {}).get("uncertainty") != (
        "report eight leave-one-group-out fits and an eight-group bootstrap envelope; no pixel bootstrap"
    ):
        raise ValueError("v2 geographic-group fold law changed")
    authority = value.get("authority", {})
    if authority.get("height_supervision") is not False or authority.get("pixelwise_training") is not False:
        raise ValueError("v2 evidence unexpectedly grants supervision or training authority")


def load_evidence_v2_contract() -> EvidenceV2Contract:
    prereg_path = REPOSITORY_ROOT / PREREG_RELATIVE
    if sha256_file(prereg_path) != PREREG_SHA256:
        raise ValueError("raised-bog v2 preregistration hash changed")
    prereg = json.loads(prereg_path.read_bytes())
    _require_v2(prereg)

    capacity_path = REPOSITORY_ROOT / CAPACITY_RELATIVE
    if sha256_file(capacity_path) != CAPACITY_MANIFEST_SHA256:
        raise ValueError("corrected Moore capacity artifact changed")
    capacity = json.loads(capacity_path.read_bytes())
    if (
        capacity.get("recipe_id") != CAPACITY_BUILD_ID
        or capacity.get("status") != "complete"
        or capacity.get("result") != "reject"
    ):
        raise ValueError("corrected Moore capacity identity or result changed")
    recipe_path = capacity_path.parent / capacity["recipe"]["path"]
    if sha256_file(recipe_path) != capacity["recipe"]["sha256"]:
        raise ValueError("corrected Moore capacity recipe changed")
    capacity_recipe = json.loads(recipe_path.read_bytes())

    source = load_capacity_contract()
    materialization = capacity_recipe.get("moore_materialization", {})
    if (
        materialization.get("build_id") != source.materialization["recipe_id"]
        or materialization.get("sha256") != source.materialization_manifest_sha256
    ):
        raise ValueError("capacity artifact and source materialization disagree")
    classifier_path = Path(__file__).parents[1] / "evidence" / "huhola.py"
    if sha256_file(classifier_path) != CLASSIFIER_SHA256:
        raise ValueError("exact local HuHoLa classifier changed")
    if capacity_recipe.get("implementation", {}).get("modules", {}).get("huhola.py") != CLASSIFIER_SHA256:
        raise ValueError("capacity artifact did not bind the exact local classifier")
    return EvidenceV2Contract(
        prereg_path=prereg_path,
        prereg=prereg,
        capacity_path=capacity_path,
        capacity=capacity,
        capacity_recipe=capacity_recipe,
        source=source,
        classifier_path=classifier_path,
    )
