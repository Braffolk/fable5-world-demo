"""Load and fail-closed validate the frozen Moore M0 machine design."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DESIGN_RELATIVE = Path(
    "docs/deep-research/microtopography-generation/review/contracts/"
    "moore-m0-research-design.2026-07-15.json"
)
DESIGN_SCHEMA = "moore-m0-research-design/1.0.0"
DESIGN_ID = "moore-m0-joint-b1-derived-b2.2026-07-15"
ARCHIVE_SHA256 = "044413bb87171776b409172d29fbde16341b228dc43b694cffe9f02a88640a67"


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
class FrozenDesign:
    repository_root: Path
    path: Path
    sha256: str
    value: dict[str, Any]
    archive_path: Path
    normative_sha256: dict[str, str]

    @property
    def group_members(self) -> dict[str, tuple[str, ...]]:
        result: dict[str, tuple[str, ...]] = {}
        for row in self.value["geographic_groups"]:
            members = row.get("members", row.get("dems_mat_fields", []))
            result[row["id"]] = tuple(members)
        return result


def load_frozen_design(repository_root: Path) -> FrozenDesign:
    root = repository_root.resolve()
    path = root / DESIGN_RELATIVE
    encoded = path.read_bytes()
    design = json.loads(encoded)
    if design.get("schema_version") != DESIGN_SCHEMA or design.get("design_id") != DESIGN_ID:
        raise ValueError("Moore M0 design schema or identity changed")
    if design["first_executable_checkpoint"].get("model_training_allowed") is not False:
        raise ValueError("materialization checkpoint unexpectedly authorizes training")
    if design["decision"].get("candidate_training_currently_authorized", False) is not False:
        raise ValueError("Moore source checkpoint unexpectedly authorizes candidate training")
    if design["decision"].get("b1_only_fallback_allowed") is not False:
        raise ValueError("Moore design unexpectedly allows a B1-only fallback")

    normative: dict[str, str] = {}
    rows = [design["normative_bindings"]["spec_after_activation"]]
    rows.extend(design["normative_bindings"]["stable_protocols"])
    for row in rows:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("normative path escapes repository")
        actual = sha256_file(root / relative)
        if actual != row["sha256"]:
            raise ValueError(f"normative binding changed: {relative}: {actual}")
        normative[relative.as_posix()] = actual

    archive_row = design["source"]["archive"]
    archive_relative = Path(archive_row["path"])
    if archive_relative.is_absolute() or ".." in archive_relative.parts:
        raise ValueError("archive path escapes repository")
    archive_path = root / archive_relative
    if archive_path.stat().st_size != archive_row["bytes"]:
        raise ValueError("Moore archive size changed")
    if sha256_file(archive_path) != archive_row["sha256"] or archive_row["sha256"] != ARCHIVE_SHA256:
        raise ValueError("Moore archive hash changed")
    if len(design["geographic_groups"]) != 8:
        raise ValueError("Moore design must bind exactly eight geographic groups")
    return FrozenDesign(
        repository_root=root,
        path=path,
        sha256=hashlib.sha256(encoded).hexdigest(),
        value=design,
        archive_path=archive_path,
        normative_sha256=normative,
    )
