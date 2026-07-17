"""Frozen condition contract + source verification for agriculture v2."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ....config import ASSET_GEN_ROOT

_SCHEMA = "laas.agriculture-v2-cultivated.conditions/1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class V2Config:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.raw: dict[str, Any] = json.loads(self.path.read_text())
        if self.raw.get("schema_version") != _SCHEMA:
            raise ValueError("unsupported agriculture v2 condition schema")
        if self.raw.get("status") != "development_float_qa_only":
            raise ValueError("agriculture v2 config must be development_float_qa_only")
        self.config_sha256 = sha256_file(self.path)

    # -- convenience accessors -------------------------------------------------
    @property
    def site(self) -> dict[str, Any]:
        return self.raw["site"]

    @property
    def params(self) -> dict[str, Any]:
        return self.raw["params"]

    @property
    def gates(self) -> dict[str, Any]:
        return self.raw["gates"]

    @property
    def mixture(self) -> dict[str, Any]:
        return self.raw["operation_unknown_mixture"]

    @property
    def marzahn(self) -> dict[str, Any]:
        return self.raw["marzahn_dual_scale_facts"]

    @property
    def texel_m(self) -> float:
        return float(self.site["texel_m"])

    def verify_sources(self) -> dict[str, dict[str, Any]]:
        """Confirm every bound source file matches its frozen sha256; fail closed."""
        verified: dict[str, dict[str, Any]] = {}
        for name, source in self.raw["sources"].items():
            resolved = ASSET_GEN_ROOT / source["path"]
            actual = sha256_file(resolved)
            if actual != source["sha256"]:
                raise ValueError(
                    f"bound agriculture v2 source changed: {name} {source['path']}"
                )
            verified[name] = {
                "path": source["path"],
                "sha256": source["sha256"],
                "role": source.get("role"),
            }
        return verified

    def implementation_sha256(self) -> str:
        digest = hashlib.sha256()
        root = Path(__file__).parent
        for path in sorted(root.glob("*.py")):
            digest.update(path.name.encode("ascii") + b"\0")
            digest.update(path.read_bytes())
        return digest.hexdigest()


def load_config(path: Path) -> V2Config:
    return V2Config(path)
