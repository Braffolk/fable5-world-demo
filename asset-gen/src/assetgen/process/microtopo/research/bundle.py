"""Load and hash-bind the frozen forest research design and retained evidence."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONTRACT_RELATIVE = Path(
    "docs/deep-research/microtopography-generation/review/contracts/"
    "forest-floor-research-design.2026-07-14.json"
)
CONTRACT_SCHEMA = "microtopography-forest-floor-research-design/1.0.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


@dataclass(frozen=True)
class FrozenBundle:
    repository_root: Path
    contract_path: Path
    contract_sha256: str
    contract: dict[str, Any]
    inputs: tuple[tuple[Path, str], ...]
    current_input_sha256: tuple[tuple[Path, str], ...]
    tracking_input_sha256: tuple[tuple[Path, str], ...]

    @property
    def bundle_identity(self) -> str:
        return hashlib.sha256(
            canonical_json(
                {
                    "schema": "forest-weak-research-bundle/1",
                    "contract": self.contract_sha256,
                    "inputs": [[path.relative_to(self.repository_root).as_posix(), digest] for path, digest in self.inputs],
                }
            )
        ).hexdigest()


def load_frozen_bundle(repository_root: Path) -> FrozenBundle:
    root = repository_root.resolve()
    contract_path = root / CONTRACT_RELATIVE
    encoded = contract_path.read_bytes()
    contract = json.loads(encoded)
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("forest research design schema changed")
    if contract.get("contract_id") != "forest-floor-hovi-evo-forestsemantic-2026-07-14":
        raise ValueError("forest research design identity changed")
    inputs: list[tuple[Path, str]] = []
    for row in contract.get("normative_inputs", []):
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("normative input escapes repository")
        path = root / relative
        actual = sha256_file(path)
        if actual != row["sha256"]:
            raise ValueError(f"normative input changed: {relative}: {actual}")
        inputs.append((path, row["sha256"]))
    if len(inputs) != 13:
        raise ValueError("forest research design must bind exactly thirteen normative inputs")
    tracking = []
    for row in contract.get("tracking_inputs", []):
        relative = Path(row["path"])
        if row.get("authority") != "nonnormative_mutable_progress_record":
            raise ValueError("tracking input claims unexpected authority")
        path = root / relative
        tracking.append((path, sha256_file(path)))
    return FrozenBundle(
        repository_root=root,
        contract_path=contract_path,
        contract_sha256=hashlib.sha256(encoded).hexdigest(),
        contract=contract,
        inputs=tuple(inputs),
        current_input_sha256=tuple((path, sha256_file(path)) for path, _ in inputs),
        tracking_input_sha256=tuple(tracking),
    )
