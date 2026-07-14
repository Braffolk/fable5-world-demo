"""Complete reproducibility identity for research-only numerical artifacts."""
from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path

from .bundle import sha256_file


def numerical_environment_identity(repository_root: Path) -> dict:
    assetgen_root = repository_root / "asset-gen"
    uv = subprocess.run(
        ("uv", "--version"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "build": list(platform.python_build()),
            "compiler": platform.python_compiler(),
            "executable": sys.executable,
            "executable_sha256": sha256_file(Path(sys.executable)),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "scipy",
                "joblib",
                "scikit-learn",
                "laspy",
                "pillow",
            )
        },
        "uv": uv,
        "uv_lock_sha256": sha256_file(assetgen_root / "uv.lock"),
        "pyproject_sha256": sha256_file(assetgen_root / "pyproject.toml"),
    }
