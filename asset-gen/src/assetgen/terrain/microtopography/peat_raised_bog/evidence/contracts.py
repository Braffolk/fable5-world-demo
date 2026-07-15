"""Fail-closed bindings for the Moore whole-form capacity checkpoint."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .....config import ASSET_GEN_ROOT


REPOSITORY_ROOT = ASSET_GEN_ROOT.parent
PREREG_RELATIVE = Path(
    "asset-gen/config/microtopography/peat-raised-bog/bundle-preregistration-v1.json"
)
PREREG_SHA256 = "21d972723f7c2641ce6a9fb7c15e171e85f3378dc83863cc93cb833be94bf115"
MATERIALIZATION_BUILD_ID = (
    "5bdc602d32bcd0c6d685e811d804c615104de4f0b2831eae0e6033653b0d6bdc"
)
MATERIALIZATION_MANIFEST_SHA256 = (
    "13b7a18304ae7e8fe0db786e7e6bb9a199f481dc1f6b95911ed65d6f6068df16"
)
HUHOLA_IMPLEMENTATION_SHA256 = (
    "77283068c4862e50b0950940c2105919bbb7c412ad462a54d0c7bf96daead6ac"
)


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
class CapacityContract:
    prereg_path: Path
    prereg_sha256: str
    prereg: dict[str, Any]
    materialization_root: Path
    materialization_manifest_path: Path
    materialization_manifest_sha256: str
    materialization: dict[str, Any]
    huhola_implementation_path: Path
    huhola_implementation_sha256: str


def _require_exact_preregistration(value: dict[str, Any]) -> None:
    if value.get("schema_version") != "laas.peat-raised-bog-r0-research-bundle-preregistration/1":
        raise ValueError("raised-bog preregistration schema changed")
    if value.get("bundle_id") != "peat-raised-bog-r0-research-bundle/1":
        raise ValueError("raised-bog preregistration identity changed")
    if value.get("preregistration_id") != "open-intact-nonforested-hummock-lawn-hollow-v1":
        raise ValueError("raised-bog preregistration selection changed")
    gate = value.get("descriptor_capacity_gate", {})
    if gate.get("minimum_forms") != 200 or gate.get("minimum_geographic_groups") != 6:
        raise ValueError("Moore capacity thresholds changed")
    if gate.get("no_padding_interpolation_or_boundary_reflection_credit") is not True:
        raise ValueError("Moore boundary-support law changed")
    expected_marks = (
        "sign and class",
        "relative relief",
        "footprint area and axes",
        "anisotropy orientation and asymmetry",
        "shoulder profile",
        "same-class and cross-class spacing adjacency",
    )
    if tuple(gate.get("required_joint_marks", ())) != expected_marks:
        raise ValueError("Moore joint-mark contract changed")
    authority = value.get("authority", {})
    if authority.get("height_supervision") is not False or authority.get("training") is not False:
        raise ValueError("capacity checkpoint unexpectedly authorizes training")
    if value.get("retained_evidence", {}).get("huhola", {}).get("generator_role_forbidden") is not True:
        raise ValueError("HuHoLa generator prohibition changed")


def load_capacity_contract() -> CapacityContract:
    prereg_path = REPOSITORY_ROOT / PREREG_RELATIVE
    prereg_sha = sha256_file(prereg_path)
    if prereg_sha != PREREG_SHA256:
        raise ValueError(f"raised-bog preregistration hash changed: {prereg_sha}")
    prereg = json.loads(prereg_path.read_bytes())
    _require_exact_preregistration(prereg)

    retained = prereg["retained_evidence"]["moore_materialization"]
    root = REPOSITORY_ROOT / retained["root"]
    manifest_path = REPOSITORY_ROOT / retained["manifest"]["path"]
    if root.name != MATERIALIZATION_BUILD_ID or manifest_path.parent != root:
        raise ValueError("Moore materialization root changed")
    manifest_sha = sha256_file(manifest_path)
    if (
        manifest_sha != MATERIALIZATION_MANIFEST_SHA256
        or manifest_sha != retained["manifest"]["sha256"]
    ):
        raise ValueError("Moore materialization manifest changed")
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get("recipe_id") != MATERIALIZATION_BUILD_ID or manifest.get("status") != "complete":
        raise ValueError("Moore materialization identity or completion state changed")
    inventory = manifest.get("inventory", {})
    expected_inventory = {
        "grid_count": 68,
        "geographic_group_count": 8,
        "finite_cell_count": 3_091_387,
    }
    for field, expected in expected_inventory.items():
        if inventory.get(field) != expected:
            raise ValueError(f"Moore materialization inventory changed: {field}")
    if abs(float(inventory.get("finite_area_m2", 0.0)) - 309.1387) > 1e-10:
        raise ValueError("Moore materialization finite area changed")
    if manifest.get("training_authorized") is not False or manifest.get("publication_authorized") is not False:
        raise ValueError("Moore materialization unexpectedly grants authority")
    plots = manifest.get("plots", [])
    if len(plots) != 68 or len({(row["group_id"], row["plot_id"]) for row in plots}) != 68:
        raise ValueError("Moore materialized plot inventory changed")
    for row in plots:
        path = root / row["path"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"Moore plot artifact changed: {row['path']}")

    huhola = prereg["retained_evidence"]["huhola"]
    implementation = REPOSITORY_ROOT / huhola["repository_path"] / "huhola/microtopography.py"
    implementation_sha = sha256_file(implementation)
    if implementation_sha != HUHOLA_IMPLEMENTATION_SHA256:
        raise ValueError("pinned HuHoLa implementation changed")
    paper = REPOSITORY_ROOT / huhola["paper_path"]
    if sha256_file(paper) != huhola["paper_sha256"]:
        raise ValueError("pinned HuHoLa paper changed")
    return CapacityContract(
        prereg_path=prereg_path,
        prereg_sha256=prereg_sha,
        prereg=prereg,
        materialization_root=root,
        materialization_manifest_path=manifest_path,
        materialization_manifest_sha256=manifest_sha,
        materialization=manifest,
        huhola_implementation_path=implementation,
        huhola_implementation_sha256=implementation_sha,
    )
