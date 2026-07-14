"""Fail-closed binding for the metadata-only E57 reader wheel."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ....config import ASSET_GEN_ROOT


E57_VERSION = "0.2.3"
E57_WHEEL_FILENAME = "e57-0.2.3-cp312-cp312-macosx_11_0_arm64.whl"
E57_WHEEL_SHA256 = "6560e33463d0f9852b468e10db2071d0e1019f5c5cb56e2b5ec7d9fccc8c237d"
E57_WHEEL_BYTES = 394_268
E57_WHEEL_TAG = "cp312-cp312-macosx_11_0_arm64"
E57_INIT_SHA256 = "ceb8d9fcdaafbb3efab882e0dbe5690599a9cce434bdf76379cc51bf7e96f83b"
E57_EXTENSION_FILENAME = "e57.cpython-312-darwin.so"
E57_EXTENSION_SHA256 = "2ab56d9990cf7a52272a838d98a988a65733dce15085747a4adcc1210cf3f1b2"
E57_WHEEL_METADATA_SHA256 = (
    "f88bca52207a71934ca2a802e6552d7a014bfeee5e3c2a6008a9652024839516"
)
E57_SBOM_SHA256 = "8888380d38de6f9d405506254f484eec8b2e74d199e874138edbe0c52ec1c9d5"
E57_RUST_VERSION = "0.11.13"
E57_RUST_CHECKSUM = "fcfee41a50fbd70278c70cc477b8671ffe03951cba028cc2c057777a9ee6a3cc"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class E57Runtime:
    raw_xml: Callable[[str], str]
    binding: dict


def validate_e57_runtime(
    *,
    pyproject_bytes: bytes | None = None,
    lock_bytes: bytes | None = None,
) -> E57Runtime:
    """Return only the pinned raw-XML callable after verifying its wheel."""
    if (
        sys.implementation.name != "cpython"
        or sys.version_info[:2] != (3, 12)
        or sys.platform != "darwin"
        or platform.machine() != "arm64"
    ):
        raise RuntimeError(
            "Hovi metadata inventory requires CPython 3.12 on macOS arm64"
        )

    pyproject_path = ASSET_GEN_ROOT / "pyproject.toml"
    pyproject = tomllib.loads(
        (
            pyproject_path.read_bytes()
            if pyproject_bytes is None
            else pyproject_bytes
        ).decode("utf-8", errors="strict")
    )
    dependencies = pyproject.get("project", {}).get("dependencies", [])
    if "e57==0.2.3" not in dependencies:
        raise RuntimeError("pyproject no longer pins e57==0.2.3 exactly")

    lock_path = ASSET_GEN_ROOT / "uv.lock"
    lock = tomllib.loads(
        (lock_path.read_bytes() if lock_bytes is None else lock_bytes).decode(
            "utf-8",
            errors="strict",
        )
    )
    packages = [package for package in lock.get("package", []) if package.get("name") == "e57"]
    if len(packages) != 1 or packages[0].get("version") != E57_VERSION:
        raise RuntimeError("uv.lock no longer resolves exactly e57==0.2.3")
    target_wheel = {
        "url": (
            "https://files.pythonhosted.org/packages/7e/cd/"
            "5039370a7806f6b579d448358742b1a49a78e0408f650ebdaba7ad740c32/"
            f"{E57_WHEEL_FILENAME}"
        ),
        "hash": f"sha256:{E57_WHEEL_SHA256}",
        "size": E57_WHEEL_BYTES,
    }
    if target_wheel not in packages[0].get("wheels", []):
        raise RuntimeError("uv.lock lost the authorized CPython 3.12 macOS-arm64 wheel")

    distribution = importlib.metadata.distribution("e57")
    if distribution.version != E57_VERSION:
        raise RuntimeError("installed e57 distribution version drifted")
    package_root = Path(distribution.locate_file("e57"))
    init_path = package_root / "__init__.py"
    extension_path = package_root / E57_EXTENSION_FILENAME
    wheel_metadata_path = Path(distribution.locate_file("e57-0.2.3.dist-info/WHEEL"))
    sbom_path = Path(
        distribution.locate_file(
            "e57-0.2.3.dist-info/sboms/e57-python.cyclonedx.json"
        )
    )
    expected_hashes = {
        init_path: E57_INIT_SHA256,
        extension_path: E57_EXTENSION_SHA256,
        wheel_metadata_path: E57_WHEEL_METADATA_SHA256,
        sbom_path: E57_SBOM_SHA256,
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"installed e57 wheel file drifted: {path.name}")
    wheel_metadata = wheel_metadata_path.read_text(encoding="utf-8")
    if f"Tag: {E57_WHEEL_TAG}\n" not in wheel_metadata:
        raise RuntimeError("installed e57 wheel tag is not the authorized platform tag")
    sbom = json.loads(sbom_path.read_bytes())
    rust_components = [
        component
        for component in sbom.get("components", [])
        if component.get("name") == "e57"
        and component.get("version") == E57_RUST_VERSION
    ]
    if len(rust_components) != 1 or {
        "alg": "SHA-256",
        "content": E57_RUST_CHECKSUM,
    } not in rust_components[0].get("hashes", []):
        raise RuntimeError("installed e57 wheel SBOM lost its pinned Rust reader")

    import e57

    raw_xml = getattr(e57, "raw_xml", None)
    if not callable(raw_xml) or Path(e57.__file__) != init_path:
        raise RuntimeError("installed e57 module does not expose the pinned raw_xml callable")
    binding = {
        "python": {
            "implementation": sys.implementation.name,
            "version": platform.python_version(),
            "platform": sys.platform,
            "machine": platform.machine(),
        },
        "e57Python": {
            "version": E57_VERSION,
            "wheel": E57_WHEEL_FILENAME,
            "wheelBytes": E57_WHEEL_BYTES,
            "wheelSha256": E57_WHEEL_SHA256,
            "wheelTag": E57_WHEEL_TAG,
            "initSha256": E57_INIT_SHA256,
            "extension": E57_EXTENSION_FILENAME,
            "extensionSha256": E57_EXTENSION_SHA256,
            "wheelMetadataSha256": E57_WHEEL_METADATA_SHA256,
            "sbomSha256": E57_SBOM_SHA256,
        },
        "e57Rust": {
            "version": E57_RUST_VERSION,
            "crateChecksum": E57_RUST_CHECKSUM,
        },
        "numpy": {"version": importlib.metadata.version("numpy")},
    }
    return E57Runtime(raw_xml=raw_xml, binding=binding)
