"""Strict loader for a hash-bound Development A morphodynamics recipe."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .....config import ASSET_GEN_ROOT
from .state import FixedNestedRecipe, MorphodynamicsConfig
from .structural_base import (
    ACCEPTED_AUTHORITY_RECIPE_SHA256,
    ACCEPTED_AUTHORITY_RELATIVE_PATH,
    ACCEPTED_AUTHORITY_SHA256,
    CANONICAL_BBOX_EN,
    CONTROL_BBOX_EN,
    FINE_CANVAS_BBOX_EN,
    FINE_CANVAS_SHAPE,
    FINE_TEXEL_M,
    EAST_OUTPUT_SAMPLE_SLICE,
    REQUIRED_AUTHORITY_CHUNKS,
    StructuralFineCanvas,
    WEST_OUTPUT_SAMPLE_SLICE,
)

RECIPE_SCHEMA = "laas.erodible-slope-morphodynamics-development-a/1"
RECIPE_ID = "development-a-continuous-morphodynamics-v1"
INFLUENCE_GUARD_M = 32.0
MAXIMUM_RSS_BYTES = 8 * 1024**3
MAXIMUM_WORKSPACE_BYTES = 12 * 1024**3

_REPOSITORY_ROOT = ASSET_GEN_ROOT.parent
_IDENTITY_KEYS = {"path", "bytes", "sha256"}
_TOP_LEVEL_KEYS = {
    "schema",
    "recipeId",
    "bindings",
    "spatial",
    "c0Sha256",
    "morphodynamics",
    "morphodynamicsSha256",
    "implementation",
    "environment",
}

_FIXED_BINDINGS = {
    "conditionBundle": (
        "asset-gen/data/work/microtopography/erodible-slope/conditions/sha256/"
        "10a5ea53b007f3e6e2e878ccebd70a9712716042139ca35178310655fcb85b8b/"
        "bundle.json",
        "9d417818c90b7e8e4db757a70cd2703df5a2e4045ff54ada9248f700bede38f9",
    ),
    "evaluationPlan": (
        "asset-gen/config/microtopography/erodible-slope/"
        "r0-development-a-two-crop-v1.json",
        "8fcb20f442d1ee925cb9cc5b9bf67035f16337dcf94c40b9f84db0a820d6f7ce",
    ),
    "domainClosure": (
        "asset-gen/data/work/microtopography/erodible-slope/conditions/"
        "domain-closure/sha256/"
        "e316add66854bd25fd13dcbf07c017950ebec97b18f0485820367a7ae774a0ac/"
        "manifest.json",
        "c2ea6bf89f85421295ebb65bf4b20eac8a99a858ea9e9922dfdcdb6fc5483600",
    ),
    "processConfig": (
        "asset-gen/config/microtopography/erodible-slope/solver-c1-v1.json",
        "c45b420f28b25b59584c72ddacfa0a259ecd9cc9a652b7057a049856764637b9",
    ),
    "structuralAuthority": (
        f"asset-gen/{ACCEPTED_AUTHORITY_RELATIVE_PATH.as_posix()}",
        ACCEPTED_AUTHORITY_SHA256,
    ),
    "correctedTransaction": (
        "asset-gen/data/work/terrain-repair/taevaskoda-ahja-stage1-transaction/"
        "009fd3d2be07d465199e6330ac9d2fc5a8fc7a16bbb5df0f55e5030ce962d3d1/"
        "materialization/transaction.json",
        "90aba35212f577126031b8f913dbb469902ae791bf9bd783ffc9a4f729c334ff",
    ),
}

_IMPLEMENTATION_PATHS = frozenset(
    {
        "asset-gen/src/assetgen/config.py",
        "asset-gen/src/assetgen/grid.py",
        "asset-gen/src/assetgen/height_geom.py",
        "asset-gen/src/assetgen/terrain/microtopography/erodible_slope/solver/load.py",
        "asset-gen/src/assetgen/terrain/microtopography/erodible_slope/solver/model.py",
        "asset-gen/src/assetgen/terrain/microtopography/erodible_slope/"
        "morphodynamics/assemble.py",
        "asset-gen/src/assetgen/terrain/microtopography/erodible_slope/"
        "morphodynamics/artifact.py",
        "asset-gen/src/assetgen/terrain/microtopography/erodible_slope/"
        "morphodynamics/state.py",
        "asset-gen/src/assetgen/terrain/microtopography/erodible_slope/"
        "morphodynamics/material_field.py",
        "asset-gen/src/assetgen/terrain/microtopography/erodible_slope/"
        "morphodynamics/hydrology.py",
        "asset-gen/src/assetgen/terrain/microtopography/erodible_slope/"
        "morphodynamics/sediment.py",
        "asset-gen/src/assetgen/terrain/microtopography/erodible_slope/"
        "morphodynamics/relaxation.py",
        "asset-gen/src/assetgen/terrain/microtopography/erodible_slope/"
        "morphodynamics/solver.py",
        "asset-gen/src/assetgen/terrain/microtopography/erodible_slope/"
        "morphodynamics/run.py",
        "asset-gen/src/assetgen/terrain/microtopography/erodible_slope/"
        "morphodynamics/qa.py",
        "asset-gen/src/assetgen/terrain/microtopography/erodible_slope/"
        "morphodynamics/structural_base.py",
        "asset-gen/src/assetgen/terrain/microtopography/erodible_slope/"
        "morphodynamics/recipe.py",
        "asset-gen/src/assetgen/terrain/repair/baseline.py",
        "asset-gen/src/assetgen/terrain/repair/model.py",
        "asset-gen/src/assetgen/terrain/repair/storage.py",
        "asset-gen/src/assetgen/terrain/repair/prolong.py",
    }
)

_SPATIAL_CONTRACT = {
    "canonicalBboxEn": list(CANONICAL_BBOX_EN),
    "controlBboxEn": list(CONTROL_BBOX_EN),
    "outputChunks": [[-2, 2436, 1492], [-2, 2437, 1492]],
    "outputBboxEn": [680448.0, 6444416.0, 680704.0, 6444544.0],
    "fineCanvasBboxEn": list(FINE_CANVAS_BBOX_EN),
    "fineTexelMeters": FINE_TEXEL_M,
    "fineCanvasShape": list(FINE_CANVAS_SHAPE),
    "stitchedOutputShape": [2049, 4097],
    "stitchedOutputCanvasSlice": [[512, 2561], [512, 4609]],
    "westOutputCanvasSlice": [[512, 2561], [512, 2561]],
    "eastOutputCanvasSlice": [[512, 2561], [2560, 4609]],
    "sharedCanvasColumn": 2560,
    "requiredAuthorityChunks": [list(value) for value in REQUIRED_AUTHORITY_CHUNKS],
    "influenceGuardMeters": INFLUENCE_GUARD_M,
    "maximumRssBytes": MAXIMUM_RSS_BYTES,
    "maximumWorkspaceBytes": MAXIMUM_WORKSPACE_BYTES,
}

_MORPHODYNAMICS_CONTRACT = {
    "event_duration_s": 7200.0,
    "rainfall_depth_m": 0.04,
    "material_seed": 26071517,
    "material_halo_m": 4.0,
    "sand_correlation_m": 0.5,
    "till_correlation_m": 1.0,
    "sand_log_std": 0.30,
    "till_log_std": 0.18,
    "material_factor_min": 0.5,
    "material_factor_max": 2.0,
    "maturity_growth_s": 900.0,
    "maturity_heal_s": 86400.0,
    "maturity_exponent": 1.5,
    "maturity_floor": 0.15,
    "deposition_slope_ceiling": 0.12,
    "transport_kg_per_joule": 0.00018,
    "deposit_bulk_density_kg_m3": 1450.0,
    "water_density_kg_m3": 998.0,
    "gravity_m_s2": 9.80665,
    "manning_n_s_m13": 0.09,
    "repose_sand_gradient": 0.70,
    "repose_till_gradient": 0.95,
    "sediment_relative_tolerance": 2e-5,
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return _sha256(payload)


def _require_sha256(value: object, *, role: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{role} must be a SHA-256 hex digest")
    try:
        if len(bytes.fromhex(value)) != 32 or value != value.lower():
            raise ValueError
    except ValueError as error:
        raise ValueError(f"{role} must be a lowercase SHA-256 hex digest") from error
    return value


def _require_exact(value: object, expected: object, *, role: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(value, dict) or set(value) != set(expected):
            raise ValueError(f"{role} has missing or unknown fields")
        for key, expected_value in expected.items():
            _require_exact(value[key], expected_value, role=f"{role}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(value, list) or len(value) != len(expected):
            raise ValueError(f"{role} has the wrong list contract")
        for index, (item, expected_item) in enumerate(zip(value, expected)):
            _require_exact(item, expected_item, role=f"{role}[{index}]")
        return
    if isinstance(expected, bool):
        if not isinstance(value, bool) or value is not expected:
            raise ValueError(f"{role} differs from the frozen value")
        return
    if isinstance(expected, int):
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ValueError(f"{role} differs from the frozen integer")
        return
    if isinstance(expected, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{role} must be numeric")
        if float(value) != expected:
            raise ValueError(f"{role} differs from the frozen numeric value")
        return
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{role} differs from the frozen value")


def _safe_repository_path(relative_path: object) -> Path:
    if not isinstance(relative_path, str):
        raise ValueError("bound artifact path must be a string")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("bound artifact path escapes the repository")
    root = _REPOSITORY_ROOT.resolve()
    result = (_REPOSITORY_ROOT / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError("bound artifact path escapes the repository")
    return result


@dataclass(frozen=True)
class ArtifactIdentity:
    relative_path: str
    bytes: int
    sha256: str

    @property
    def repository_path(self) -> Path:
        path = _safe_repository_path(self.relative_path)
        payload = path.read_bytes()
        if len(payload) != self.bytes or _sha256(payload) != self.sha256:
            raise ValueError(f"bound artifact changed after recipe load: {path}")
        return path


def _load_identity(value: object, *, role: str) -> ArtifactIdentity:
    if not isinstance(value, dict) or set(value) != _IDENTITY_KEYS:
        raise ValueError(f"{role} has missing or unknown identity fields")
    relative_path = value["path"]
    byte_count = value["bytes"]
    digest = _require_sha256(value["sha256"], role=f"{role}.sha256")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise ValueError(f"{role}.bytes must be a nonnegative integer")
    path = _safe_repository_path(relative_path)
    payload = path.read_bytes()
    if len(payload) != byte_count:
        raise ValueError(f"{role} byte count differs from its binding")
    if _sha256(payload) != digest:
        raise ValueError(f"{role} SHA-256 differs from its binding")
    return ArtifactIdentity(relative_path, byte_count, digest)


@dataclass(frozen=True)
class FrozenInputBindings:
    condition_bundle: ArtifactIdentity
    evaluation_plan: ArtifactIdentity
    domain_closure: ArtifactIdentity
    process_config: ArtifactIdentity
    structural_authority: ArtifactIdentity
    corrected_transaction: ArtifactIdentity
    structural_authority_recipe_sha256: str


def _load_bindings(value: object) -> FrozenInputBindings:
    expected_keys = set(_FIXED_BINDINGS) | {"structuralAuthorityRecipeSha256"}
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("bindings has missing or unknown fields")
    if (
        _require_sha256(
            value["structuralAuthorityRecipeSha256"],
            role="bindings.structuralAuthorityRecipeSha256",
        )
        != ACCEPTED_AUTHORITY_RECIPE_SHA256
    ):
        raise ValueError("structural-authority recipe identity differs")
    identities: dict[str, ArtifactIdentity] = {}
    for name, (expected_path, expected_sha256) in _FIXED_BINDINGS.items():
        identity = _load_identity(value[name], role=f"bindings.{name}")
        if identity.relative_path != expected_path or identity.sha256 != expected_sha256:
            raise ValueError(f"bindings.{name} is not the accepted authority")
        identities[name] = identity
    return FrozenInputBindings(
        condition_bundle=identities["conditionBundle"],
        evaluation_plan=identities["evaluationPlan"],
        domain_closure=identities["domainClosure"],
        process_config=identities["processConfig"],
        structural_authority=identities["structuralAuthority"],
        corrected_transaction=identities["correctedTransaction"],
        structural_authority_recipe_sha256=ACCEPTED_AUTHORITY_RECIPE_SHA256,
    )


def _load_implementation(value: object) -> tuple[str, tuple[ArtifactIdentity, ...]]:
    if not isinstance(value, dict) or set(value) != {"sourceRevision", "files"}:
        raise ValueError("implementation has missing or unknown fields")
    revision = value["sourceRevision"]
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("implementation.sourceRevision must be a Git object id")
    try:
        bytes.fromhex(revision)
    except ValueError as error:
        raise ValueError("implementation.sourceRevision is not hexadecimal") from error
    rows = value["files"]
    if not isinstance(rows, list):
        raise ValueError("implementation.files must be a list")
    identities = tuple(
        _load_identity(row, role=f"implementation.files[{index}]")
        for index, row in enumerate(rows)
    )
    paths = [identity.relative_path for identity in identities]
    if len(paths) != len(set(paths)):
        raise ValueError("implementation.files repeats a path")
    if set(paths) != _IMPLEMENTATION_PATHS:
        missing = sorted(_IMPLEMENTATION_PATHS - set(paths))
        unknown = sorted(set(paths) - _IMPLEMENTATION_PATHS)
        raise ValueError(
            f"implementation file inventory differs; missing={missing}, unknown={unknown}"
        )
    return revision, identities


def _load_environment(value: object) -> ArtifactIdentity:
    expected_keys = {
        "uvLock",
        "pythonVersion",
        "numpyVersion",
        "scipyVersion",
        "byteorder",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("environment has missing or unknown fields")
    uv_lock = _load_identity(value["uvLock"], role="environment.uvLock")
    if uv_lock.relative_path != "asset-gen/uv.lock":
        raise ValueError("environment.uvLock must bind asset-gen/uv.lock")
    expected = {
        "pythonVersion": platform.python_version(),
        "numpyVersion": importlib.metadata.version("numpy"),
        "scipyVersion": importlib.metadata.version("scipy"),
        "byteorder": sys.byteorder,
    }
    for key, expected_value in expected.items():
        if value[key] != expected_value:
            raise ValueError(f"environment.{key} differs from the running environment")
    return uv_lock


def _load_config(value: object, digest: object) -> MorphodynamicsConfig:
    _require_exact(value, _MORPHODYNAMICS_CONTRACT, role="morphodynamics")
    expected_digest = _require_sha256(digest, role="morphodynamicsSha256")
    if _canonical_sha256(value) != expected_digest:
        raise ValueError("morphodynamics object differs from its SHA-256 binding")
    if not isinstance(value, dict):
        raise AssertionError("exact config validation did not preserve an object")
    config = MorphodynamicsConfig(**value)
    expected = MorphodynamicsConfig(**_MORPHODYNAMICS_CONTRACT)
    if config != expected:
        raise ValueError("constructed MorphodynamicsConfig differs from the recipe")
    return config


@dataclass(frozen=True)
class FrozenMorphodynamicsRecipe:
    path: Path
    sha256: str
    c0_sha256: str
    config: MorphodynamicsConfig
    bindings: FrozenInputBindings
    fixed_nested: FixedNestedRecipe
    spatial_sha256: str
    source_revision: str
    implementation_files: tuple[ArtifactIdentity, ...]
    uv_lock: ArtifactIdentity


def load_frozen_recipe(
    path: Path,
    *,
    expected_sha256: str,
    structural_base: StructuralFineCanvas,
) -> FrozenMorphodynamicsRecipe:
    """Load a complete recipe without accepting implicit values or substitutions."""
    expected_recipe_sha256 = _require_sha256(
        expected_sha256, role="expected recipe SHA-256"
    )
    payload = path.read_bytes()
    digest = _sha256(payload)
    if digest != expected_recipe_sha256:
        raise ValueError("recipe SHA-256 differs from the caller's binding")
    document = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(document, dict) or set(document) != _TOP_LEVEL_KEYS:
        raise ValueError("recipe has missing or unknown top-level fields")
    if document["schema"] != RECIPE_SCHEMA or document["recipeId"] != RECIPE_ID:
        raise ValueError("recipe schema or identity differs")

    bindings = _load_bindings(document["bindings"])
    _require_exact(document["spatial"], _SPATIAL_CONTRACT, role="spatial")
    spatial_sha256 = _canonical_sha256(document["spatial"])
    c0_sha256 = _require_sha256(document["c0Sha256"], role="c0Sha256")
    if (
        structural_base.authority_manifest_sha256 != ACCEPTED_AUTHORITY_SHA256
        or structural_base.authority_recipe_sha256
        != ACCEPTED_AUTHORITY_RECIPE_SHA256
        or structural_base.bbox_en != FINE_CANVAS_BBOX_EN
        or structural_base.texel_m != FINE_TEXEL_M
        or structural_base.c0_height_m.shape != FINE_CANVAS_SHAPE
        or structural_base.c0_sha256 != c0_sha256
    ):
        raise ValueError("recipe C0 identity differs from the loaded structural base")
    config = _load_config(
        document["morphodynamics"], document["morphodynamicsSha256"]
    )
    source_revision, implementation_files = _load_implementation(
        document["implementation"]
    )
    uv_lock = _load_environment(document["environment"])
    fixed_nested = FixedNestedRecipe(
        fine_canvas_bbox_en=FINE_CANVAS_BBOX_EN,
        fine_texel_m=FINE_TEXEL_M,
        influence_guard_m=INFLUENCE_GUARD_M,
        output_windows=(
            WEST_OUTPUT_SAMPLE_SLICE,
            EAST_OUTPUT_SAMPLE_SLICE,
        ),
        expected_node_shape=FINE_CANVAS_SHAPE,
        maximum_rss_bytes=MAXIMUM_RSS_BYTES,
        maximum_workspace_bytes=MAXIMUM_WORKSPACE_BYTES,
    )
    return FrozenMorphodynamicsRecipe(
        path=path.resolve(),
        sha256=digest,
        c0_sha256=c0_sha256,
        config=config,
        bindings=bindings,
        fixed_nested=fixed_nested,
        spatial_sha256=spatial_sha256,
        source_revision=source_revision,
        implementation_files=implementation_files,
        uv_lock=uv_lock,
    )
