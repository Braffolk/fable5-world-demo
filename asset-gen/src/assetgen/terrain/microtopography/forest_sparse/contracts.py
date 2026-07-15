"""Machine contract and fail-closed source binding for the forest sparse screen."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .frozen_anchor import EXPECTED_IMPLEMENTATION_SHA256, EXPECTED_SEMANTIC_SHA256


SEMANTIC_KEYS = (
    "schema",
    "purpose",
    "authority",
    "primary_method",
    "surfaces",
    "algorithm",
    "rejection_gates",
    "human_gate_workflow",
    "budget",
    "environment",
    "source_bindings",
)
FROZEN_ANCHOR_PATH = (
    "asset-gen/src/assetgen/terrain/microtopography/forest_sparse/frozen_anchor.py"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def semantic_projection(raw: dict[str, Any]) -> dict[str, Any]:
    """Normative method content; implementation/output bookkeeping is excluded."""
    missing = [key for key in SEMANTIC_KEYS if key not in raw]
    if missing:
        raise ValueError(f"semantic config keys missing: {missing}")
    return {key: raw[key] for key in SEMANTIC_KEYS}


def semantic_sha256(raw: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(semantic_projection(raw))).hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


@dataclass(frozen=True)
class SurfaceContract:
    source_id: str
    role: str
    path: Path
    sha256: str
    shape: tuple[int, int]
    texel_m: float
    origin_m: tuple[float, float]


@dataclass(frozen=True)
class ScreenConfig:
    path: Path
    root: Path
    raw: dict[str, Any]
    construction: tuple[SurfaceContract, ...]
    evaluation: SurfaceContract

    @property
    def config_sha256(self) -> str:
        return sha256_file(self.path)


def _project_root(config_path: Path) -> Path:
    for parent in config_path.resolve().parents:
        if (parent / "asset-gen" / "pyproject.toml").is_file() and (parent / "docs").is_dir():
            return parent
    raise ValueError(f"cannot locate project root from {config_path}")


def _surface(root: Path, record: dict[str, Any]) -> SurfaceContract:
    return SurfaceContract(
        source_id=str(record["source_id"]),
        role=str(record["role"]),
        path=root / str(record["path"]),
        sha256=str(record["sha256"]),
        shape=(int(record["shape"][0]), int(record["shape"][1])),
        texel_m=float(record["texel_m"]),
        origin_m=(float(record["origin_m"][0]), float(record["origin_m"][1])),
    )


def load_config(path: Path) -> ScreenConfig:
    resolved = path.resolve()
    raw = _load_json_object(resolved)
    if raw.get("schema") != "forest-sparse-ideal-r0-screen/1":
        raise ValueError("unsupported forest sparse screen schema")
    actual_semantic = semantic_sha256(raw)
    if actual_semantic != EXPECTED_SEMANTIC_SHA256:
        raise ValueError(
            "forest sparse semantic anchor mismatch: "
            f"expected {EXPECTED_SEMANTIC_SHA256}, got {actual_semantic}"
        )
    anchor = raw.get("semantic_anchor", {})
    if anchor.get("sha256") != EXPECTED_SEMANTIC_SHA256:
        raise ValueError("config semantic_anchor does not match code anchor")
    if anchor.get("included_top_level_keys") != list(SEMANTIC_KEYS):
        raise ValueError("config semantic projection key list changed")
    root = _project_root(resolved)
    construction = tuple(_surface(root, item) for item in raw["surfaces"]["construction"])
    evaluation = _surface(root, raw["surfaces"]["evaluation_only"])
    if tuple(item.source_id for item in construction) != ("k11", "k32", "k36"):
        raise ValueError("construction order must remain k11, k32, k36")
    if any(item.role != "construction" for item in construction):
        raise ValueError("all construction surfaces must have construction role")
    if evaluation.source_id != "k19" or evaluation.role != "evaluation_only":
        raise ValueError("k19 must remain the sole evaluation-only surface")
    fixed = raw["algorithm"]
    expected = {
        "fine_texel_m": 0.0625,
        "coarse_texel_m": 1.0,
        "factor": 16,
        "low_patch_cells": 8,
        "low_stride_cells": 2,
        "sparsity": 1,
    }
    if any(fixed.get(key) != value for key, value in expected.items()):
        raise ValueError(f"frozen algorithm constants changed; expected {expected}")
    return ScreenConfig(resolved, root, raw, construction, evaluation)


def verify_static_closure(config: ScreenConfig) -> dict[str, str]:
    """Verify hashes without importing or opening any NPZ member arrays."""
    records: dict[str, str] = {}
    implementation = config.raw["implementation"]
    if implementation.get("package") != "assetgen.terrain.microtopography.forest_sparse":
        raise ValueError("implementation package binding changed")
    if implementation.get("trust_root_path") != FROZEN_ANCHOR_PATH:
        raise ValueError("implementation trust-root path changed")
    implementation_bindings = implementation.get("files")
    if not isinstance(implementation_bindings, list):
        raise ValueError("implementation files binding must be a list")
    by_path: dict[str, str] = {}
    for binding in implementation_bindings:
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError("implementation binding schema is not exact")
        relative = str(binding["path"])
        if relative in by_path:
            raise ValueError(f"duplicate implementation binding: {relative}")
        by_path[relative] = str(binding["sha256"])

    expected_paths = set(EXPECTED_IMPLEMENTATION_SHA256) | {FROZEN_ANCHOR_PATH}
    if set(by_path) != expected_paths:
        raise ValueError("implementation bindings have an addition or omission")
    package_dir = config.root / FROZEN_ANCHOR_PATH
    actual_package_files = {
        path.relative_to(config.root).as_posix()
        for path in package_dir.parent.glob("*.py")
        if path.is_file() and not path.is_symlink()
    }
    if actual_package_files != expected_paths:
        raise ValueError("outcome-producing package files have an addition or omission")
    for relative, anchored_sha256 in EXPECTED_IMPLEMENTATION_SHA256.items():
        if by_path[relative] != anchored_sha256:
            raise ValueError(f"config implementation identity differs from trust root: {relative}")

    bindings = list(config.raw["source_bindings"]) + implementation_bindings
    bindings += [
        {"path": item.path.relative_to(config.root).as_posix(), "sha256": item.sha256}
        for item in (*config.construction, config.evaluation)
    ]
    for binding in bindings:
        relative = str(binding["path"])
        expected = str(binding["sha256"])
        actual = sha256_file(config.root / relative)
        if actual != expected:
            raise ValueError(f"hash mismatch for {relative}: expected {expected}, got {actual}")
        if relative in records:
            raise ValueError(f"duplicate closure binding: {relative}")
        records[relative] = actual
    return records
