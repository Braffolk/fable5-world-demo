"""Content-derived identity for the Stage-1 microtopography fixture recipe."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import sys

from .config import ASSET_GEN_ROOT

RECIPE_ID = "laas.micro.fixture.recipe.v1"
SYNTHESIS_RECIPE_ID = "laas.micro.measured-synthesis.recipe.v1"
_SOURCE_PATHS = (
    "src/assetgen/cook/chunkio.py",
    "src/assetgen/cook/encode.py",
    "src/assetgen/cook/micro_fixture_cook.py",
    "src/assetgen/cook/micro_hierarchy.py",
    "src/assetgen/cook/pinned_height.py",
    "src/assetgen/height_geom.py",
    "src/assetgen/micro_config.py",
    "src/assetgen/process/micro_masks.py",
    "src/assetgen/process/micro_fixture.py",
)

_SYNTHESIS_SOURCE_PATHS = (
    "src/assetgen/cook/chunkio.py",
    "src/assetgen/cook/encode.py",
    "src/assetgen/cook/micro_hierarchy.py",
    "src/assetgen/cook/micro_synth_cook.py",
    "src/assetgen/cook/pinned_height.py",
    "src/assetgen/height_geom.py",
    "src/assetgen/micro_config.py",
    "src/assetgen/process/etak_read.py",
    "src/assetgen/process/landcover.py",
    "src/assetgen/process/micro_masks.py",
    "src/assetgen/process/micro_fixture.py",
    "src/assetgen/process/soil.py",
    "src/assetgen/process/microtopo/__init__.py",
    "src/assetgen/process/microtopo/api.py",
    "src/assetgen/process/microtopo/model.py",
    "src/assetgen/process/microtopo/preprocess.py",
    "src/assetgen/process/microtopo/projection.py",
    "src/assetgen/process/microtopo/synthesis.py",
)


def _synthesis_environment_identity(asset_gen_root: Path) -> dict:
    """Bind the locked environment and numerical runtime used by the cook."""
    import numpy as np
    import scipy

    def numerical_backend(module) -> dict:
        config = getattr(module.__config__, "CONFIG", {})
        dependencies = config.get("Build Dependencies", {})
        machine = config.get("Machine Information", {}).get("host", {})
        return {
            "machine": {
                key: machine.get(key) for key in ("cpu", "family", "endian", "system")
            },
            "blas": {
                key: dependencies.get("blas", {}).get(key)
                for key in ("name", "version", "has ilp64")
            },
            "lapack": {
                key: dependencies.get("lapack", {}).get(key)
                for key in ("name", "version", "has ilp64")
            },
        }

    return {
        "lockSha256": {
            "pyproject.toml": _sha256_file(asset_gen_root / "pyproject.toml"),
            "uv.lock": _sha256_file(asset_gen_root / "uv.lock"),
        },
        "runtime": {
            "pythonImplementation": platform.python_implementation(),
            "pythonVersion": platform.python_version(),
            "pythonCacheTag": sys.implementation.cache_tag,
            "platformSystem": platform.system(),
            "platformMachine": platform.machine(),
            "numpyVersion": np.__version__,
            "numpyBackend": numerical_backend(np),
            "scipyVersion": scipy.__version__,
            "scipyBackend": numerical_backend(scipy),
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_micro_fixture_recipe(
    parent_cx: int,
    parent_cz: int,
    base_manifest_path: Path,
    *,
    asset_gen_root: Path = ASSET_GEN_ROOT,
) -> tuple[str, dict]:
    """Bind code, configs, AOI, base manifest, and every inherited index."""
    manifest = json.loads(base_manifest_path.read_bytes())
    indexes = {
        layer: _sha256_file(base_manifest_path.parent / meta["index"])
        for layer, meta in sorted(manifest["layers"].items())
    }
    inputs = {
        "id": RECIPE_ID,
        "parent": [-1, parent_cx, parent_cz],
        "baseManifestSha256": _sha256_file(base_manifest_path),
        "baseIndexSha256": indexes,
        "configSha256": {
            "base.toml": _sha256_file(asset_gen_root / "config" / "base.toml"),
            "microtopography.toml": _sha256_file(
                asset_gen_root / "config" / "microtopography.toml"
            ),
        },
        "sourceSha256": {
            relative: _sha256_file(asset_gen_root / relative) for relative in _SOURCE_PATHS
        },
    }
    blob = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(RECIPE_ID.encode() + b"\0" + blob).hexdigest(), inputs


def derive_micro_synthesis_recipe(
    parent_cx: int,
    parent_cz: int,
    base_manifest_path: Path,
    exemplar_manifest_path: Path,
    *,
    asset_gen_root: Path = ASSET_GEN_ROOT,
) -> tuple[str, dict]:
    """Bind the measured bank, generator, base release and exact proof window."""
    manifest = json.loads(base_manifest_path.read_bytes())
    indexes = {
        layer: _sha256_file(base_manifest_path.parent / meta["index"])
        for layer, meta in sorted(manifest["layers"].items())
    }
    exemplar = json.loads(exemplar_manifest_path.read_bytes())
    bank_relative = exemplar.get("bankFile")
    bank_sha256 = exemplar.get("bankSha256")
    if not isinstance(bank_relative, str) or not isinstance(bank_sha256, str):
        raise ValueError("exemplar manifest must bind bankFile and bankSha256")
    bank_path = exemplar_manifest_path.parent / bank_relative
    actual_bank_sha256 = _sha256_file(bank_path)
    if actual_bank_sha256 != bank_sha256:
        raise ValueError("exemplar bank differs from its manifest SHA-256")
    etak_sources = sorted((asset_gen_root / "data" / "in" / "etak").glob("*.gpkg"))
    if len(etak_sources) != 1:
        raise ValueError("measured synthesis recipe requires exactly one ETAK GeoPackage")
    soil_root = asset_gen_root / "data" / "in" / "soil" / "mullakaart"
    soil_sources = sorted(soil_root.glob("Mullakaart.*"))
    required_soil_suffixes = {".shp", ".shx", ".dbf", ".prj", ".cpg"}
    present_soil_suffixes = {path.suffix.lower() for path in soil_sources}
    missing_soil_suffixes = required_soil_suffixes - present_soil_suffixes
    if missing_soil_suffixes:
        raise ValueError(
            "measured synthesis recipe lacks Mullakaart components: "
            + ", ".join(sorted(missing_soil_suffixes))
        )
    inputs = {
        "id": SYNTHESIS_RECIPE_ID,
        "parent": [-1, parent_cx, parent_cz],
        "baseManifestSha256": _sha256_file(base_manifest_path),
        "baseIndexSha256": indexes,
        "exemplarManifestSha256": _sha256_file(exemplar_manifest_path),
        "exemplarBankSha256": actual_bank_sha256,
        "etakGpkgPath": str(etak_sources[0].relative_to(asset_gen_root)),
        "etakGpkgSha256": _sha256_file(etak_sources[0]),
        "soilSourceSha256": {
            str(path.relative_to(asset_gen_root)): _sha256_file(path)
            for path in soil_sources
        },
        "configSha256": {
            "base.toml": _sha256_file(asset_gen_root / "config" / "base.toml"),
            "landcover-classes.toml": _sha256_file(
                asset_gen_root / "config" / "landcover-classes.toml"
            ),
            "microtopography.toml": _sha256_file(
                asset_gen_root / "config" / "microtopography.toml"
            ),
            "microtopography-exemplars.json": _sha256_file(
                asset_gen_root / "config" / "microtopography-exemplars.json"
            ),
            "soil-texture.toml": _sha256_file(
                asset_gen_root / "config" / "soil-texture.toml"
            ),
            "soil-types.toml": _sha256_file(
                asset_gen_root / "config" / "soil-types.toml"
            ),
        },
        "sourceSha256": {
            relative: _sha256_file(asset_gen_root / relative)
            for relative in _SYNTHESIS_SOURCE_PATHS
        },
        "environment": _synthesis_environment_identity(asset_gen_root),
    }
    blob = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(SYNTHESIS_RECIPE_ID.encode() + b"\0" + blob).hexdigest()
    return digest, inputs
