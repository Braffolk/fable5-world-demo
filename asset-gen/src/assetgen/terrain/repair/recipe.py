"""Content-derived identity for one structural-repair fine overlay."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...config import ASSET_GEN_ROOT, BaseConfig
from ...cook.micro_hierarchy import BOX_MEAN_REDUCER_VERSION
from ...height_geom import HeightChunkId
from .hierarchy import FINE_QSCALE_M
from .plan import StructuralRepairPlan
from .storage import AUTHORITY_MANIFEST_FORMAT, baseline_tile_contract


RECIPE_ID = "laas.terrain.structural-repair-overlay.recipe.v1"
RECIPE_KIND = "structural-repair-overlay-v1"
RELEASE_DISPOSITION = "preview-only"
EXPECTATION_FORMAT = 1

_DEFAULT_SOURCE_PATHS = (
    "src/assetgen/cook/chunkio.py",
    "src/assetgen/cook/encode.py",
    "src/assetgen/cook/micro_hierarchy.py",
    "src/assetgen/cook/pinned_height.py",
    "src/assetgen/cook/pyramid.py",
    "src/assetgen/config.py",
    "src/assetgen/evidence/als_water.py",
    "src/assetgen/evidence/reach.py",
    "src/assetgen/grid.py",
    "src/assetgen/height_geom.py",
    "src/assetgen/pilots/taevaskoda_authority.py",
    "src/assetgen/pilots/taevaskoda_structural.py",
    "src/assetgen/release.py",
    "src/assetgen/terrain/repair/authority_inventory.py",
    "src/assetgen/terrain/repair/baseline.py",
    "src/assetgen/terrain/repair/base_transaction.py",
    "src/assetgen/terrain/repair/closure.py",
    "src/assetgen/terrain/repair/components.py",
    "src/assetgen/terrain/repair/cook.py",
    "src/assetgen/terrain/repair/fine_water.py",
    "src/assetgen/terrain/repair/hierarchy.py",
    "src/assetgen/terrain/repair/materialize.py",
    "src/assetgen/terrain/repair/model.py",
    "src/assetgen/terrain/repair/plan.py",
    "src/assetgen/terrain/repair/pinned_baseline.py",
    "src/assetgen/terrain/repair/prolong.py",
    "src/assetgen/terrain/repair/recipe.py",
    "src/assetgen/terrain/repair/storage.py",
    "src/assetgen/terrain/repair/stage1_transaction.py",
    "src/assetgen/terrain/repair/water.py",
    "src/assetgen/terrain/repair/water_halo.py",
    "src/assetgen/terrain/repair/water_layer.py",
    "src/assetgen/terrain/repair/water_pack.py",
    "src/assetgen/terrain/repair/water_source.py",
    "src/assetgen/terrain/repair/verify.py",
)


@dataclass(frozen=True)
class StructuralRepairRecipe:
    sha256: str
    inputs: dict[str, Any]


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_child(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe relative artifact path: {relative!r}")
    root_resolved = root.resolve()
    result = (root / rel).resolve()
    if result != root_resolved and root_resolved not in result.parents:
        raise ValueError(f"artifact path leaves its manifest root: {relative!r}")
    return result


def _validated_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    try:
        if len(value) != 64 or len(bytes.fromhex(value)) != 32:
            raise ValueError
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest") from error
    return value


def _key(chunk: HeightChunkId) -> list[int]:
    return [chunk.lod, chunk.cx, chunk.cz]


def _artifact_hashes(
    artifacts: Mapping[str, Path], name: str
) -> dict[str, dict[str, Any]]:
    if not artifacts:
        raise ValueError(f"{name} artifacts must not be empty")
    result: dict[str, dict[str, Any]] = {}
    for logical_name, raw_path in sorted(artifacts.items()):
        if (
            not isinstance(logical_name, str)
            or not logical_name
            or logical_name.strip() != logical_name
            or logical_name in result
        ):
            raise ValueError(f"invalid {name} artifact name {logical_name!r}")
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"missing {name} artifact {logical_name}: {path}")
        result[logical_name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return result


def _manifest_indexes(path: Path, document: dict[str, Any]) -> dict[str, str]:
    layers = document.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError("corrected base manifest has no layers")
    indexes: dict[str, str] = {}
    for layer, metadata in sorted(layers.items()):
        if not isinstance(metadata, dict) or not isinstance(metadata.get("index"), str):
            raise ValueError(f"corrected base layer {layer!r} has no index")
        index_path = _safe_child(path.parent, metadata["index"])
        if not index_path.is_file():
            raise ValueError(f"corrected base index is missing: {index_path}")
        digest = _sha256_file(index_path)
        declared = metadata.get("indexSha256")
        if declared is not None and declared != digest:
            raise ValueError(f"corrected base index SHA-256 mismatch: {layer}")
        indexes[layer] = digest
    return indexes


def _validate_base(
    base: BaseConfig,
    manifest_path: Path,
    verification_path: Path,
) -> dict[str, Any]:
    manifest_blob = manifest_path.read_bytes()
    manifest = json.loads(manifest_blob)
    if manifest.get("format") != 1:
        raise ValueError("structural repair requires a corrected format-1 base")
    grid = {
        "anchorE": int(manifest["anchor"]["e"]),
        "anchorN": int(manifest["anchor"]["n"]),
        "chunkMeters": int(manifest["chunkMeters"]),
        "chunkRes": int(manifest["chunkRes"]),
        "lodStep": int(manifest["lodStep"]),
    }
    expected_grid = {
        "anchorE": base.grid.anchor_e,
        "anchorN": base.grid.anchor_n,
        "chunkMeters": base.grid.chunk_m,
        "chunkRes": base.grid.chunk_res,
        "lodStep": base.grid.lod_step,
    }
    if grid != expected_grid or manifest.get("codec") != base.encode.codec:
        raise ValueError("corrected base grid/codec differs from the structural recipe")
    manifest_sha = _sha256_bytes(manifest_blob)
    verification_blob = verification_path.read_bytes()
    verification = json.loads(verification_blob)
    if verification.get("passed") is not True:
        raise ValueError("corrected base verification has not passed")
    if verification.get("manifestSha256") != manifest_sha:
        raise ValueError("corrected base verification names a different manifest")
    return {
        "manifestSha256": manifest_sha,
        "indexSha256": _manifest_indexes(manifest_path, manifest),
        "verificationSha256": _sha256_bytes(verification_blob),
    }


def _validate_authority(
    path: Path, plan: StructuralRepairPlan
) -> dict[str, Any]:
    blob = path.read_bytes()
    document = json.loads(blob)
    if (
        document.get("format") != AUTHORITY_MANIFEST_FORMAT
        or document.get("role") != "structural_authority_0.25m"
        or document.get("morphology") != "absent"
        or document.get("tileAlignmentLod") != -2
        or document.get("tileTexelMeters") != 0.25
        or document.get("tileCoreResolution") != 512
        or document.get("baselineContract") != baseline_tile_contract()
    ):
        raise ValueError("invalid structural authority manifest contract")
    recipe_sha = _validated_digest(document.get("recipeSha256"), "authority recipe")
    evidence_sha = _validated_digest(document.get("evidenceSha256"), "authority evidence")
    rows = document.get("tiles")
    if not isinstance(rows, list):
        raise ValueError("structural authority manifest has no tile inventory")
    expected = {tuple(_key(chunk)) for chunk in plan.authority_support}
    actual: set[tuple[int, int, int]] = set()
    structural_paths: set[Path] = set()
    baseline_paths: set[Path] = set()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid structural authority tile row")
        raw_key = row.get("key")
        if (
            not isinstance(raw_key, list)
            or len(raw_key) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_key)
        ):
            raise ValueError("invalid structural authority tile key")
        key = tuple(raw_key)
        if key in actual:
            raise ValueError(f"duplicate structural authority tile {key}")
        actual.add(key)
        relative = row.get("path")
        baseline_relative = row.get("baselinePath")
        if not isinstance(relative, str) or not isinstance(baseline_relative, str):
            raise ValueError(f"structural authority pair {key} has no paired paths")
        artifact = _safe_child(path.parent, relative)
        baseline_artifact = _safe_child(path.parent, baseline_relative)
        if artifact in structural_paths or baseline_artifact in baseline_paths:
            raise ValueError("structural authority pairs alias an artifact within one role")
        structural_paths.add(artifact)
        baseline_paths.add(baseline_artifact)
        if not artifact.is_file() or not baseline_artifact.is_file():
            raise ValueError(f"structural authority pair is missing: {key}")
        structural_size = artifact.stat().st_size
        baseline_size = baseline_artifact.stat().st_size
        digest = _sha256_file(artifact)
        baseline_digest = _sha256_file(baseline_artifact)
        if (
            row.get("sha256") != digest
            or row.get("bytes") != structural_size
            or row.get("baselineSha256") != baseline_digest
            or row.get("baselineBytes") != baseline_size
        ):
            raise ValueError(f"structural authority pair identity mismatch: {key}")
        normalized.append(
            {
                "key": list(key),
                "path": relative,
                "bytes": structural_size,
                "sha256": digest,
                "baselinePath": baseline_relative,
                "baselineBytes": baseline_size,
                "baselineSha256": baseline_digest,
            }
        )
    if structural_paths & baseline_paths:
        raise ValueError("structural and baseline authority inventories alias artifacts")
    if actual != expected:
        raise ValueError(
            "structural authority tile set differs from the frozen source support; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    normalized.sort(key=lambda row: tuple(row["key"]))
    structural = [
        {
            "key": row["key"],
            "path": row["path"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in normalized
    ]
    baseline = [
        {
            "key": row["key"],
            "path": row["baselinePath"],
            "bytes": row["baselineBytes"],
            "sha256": row["baselineSha256"],
        }
        for row in normalized
    ]
    return {
        "manifestSha256": _sha256_bytes(blob),
        "recipeSha256": recipe_sha,
        "evidenceSha256": evidence_sha,
        "pairInventorySha256": _sha256_bytes(_json_bytes(normalized)),
        "structuralTileInventorySha256": _sha256_bytes(_json_bytes(structural)),
        "baselineTileInventorySha256": _sha256_bytes(_json_bytes(baseline)),
        "pairCount": len(normalized),
        "structuralTileCount": len(structural),
        "baselineTileCount": len(baseline),
        "structuralBytes": sum(row["bytes"] for row in structural),
        "baselineBytes": sum(row["bytes"] for row in baseline),
    }


def _validate_campaign_bindings(
    campaign_path: Path,
    pilot_path: Path,
    profile: Mapping[str, Path],
    geometry: Mapping[str, Path],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    campaign_sha = _sha256_file(campaign_path)
    campaign = json.loads(campaign_path.read_bytes())
    pilot = json.loads(pilot_path.read_bytes())
    if campaign.get("format") != 1 or pilot.get("format") != 1:
        raise ValueError("campaign and pilot manifests must use format 1")
    profile_hashes = _artifact_hashes(profile, "profile")
    geometry_hashes = _artifact_hashes(geometry, "geometry")
    if pilot.get("campaignMetadataSha256") != campaign_sha:
        raise ValueError("pilot names a different campaign manifest")
    declared_profile = campaign.get("arraysNpzSha256")
    if declared_profile not in {item["sha256"] for item in profile_hashes.values()}:
        raise ValueError("campaign profile arrays are absent from the bound profile artifacts")
    if pilot.get("campaignArraysSha256") != declared_profile:
        raise ValueError("pilot names different campaign profile arrays")
    campaign_content = _validated_digest(
        pilot.get("campaignContentSha256"), "campaign content evidence"
    )
    qualification = _validated_digest(
        campaign.get("campaign", {}).get("qualificationSha256"),
        "campaign qualification evidence",
    )
    declared_geometry = pilot.get("geometrySha256")
    if not isinstance(declared_geometry, dict):
        raise ValueError("pilot has no geometry identity map")
    for name, digest in declared_geometry.items():
        actual = geometry_hashes.get(name)
        if actual is None or actual["sha256"] != digest:
            raise ValueError(f"pilot geometry identity mismatch: {name}")
    return profile_hashes, geometry_hashes, {
        "contentSha256": campaign_content,
        "qualificationSha256": qualification,
        "metadataSha256": campaign_sha,
        "arraysSha256": declared_profile,
    }


def _environment_identity(asset_gen_root: Path) -> dict[str, Any]:
    import numpy as np
    import scipy

    def numerical_backend(module) -> dict[str, Any]:
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
            name: _sha256_file(asset_gen_root / name)
            for name in ("pyproject.toml", "uv.lock")
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


def _validate_pilot_config_binding(
    pilot_path: Path, config_artifacts: Mapping[str, Path]
) -> None:
    pilot = json.loads(pilot_path.read_bytes())
    expected = pilot.get("configCanonicalSha256")
    _validated_digest(expected, "pilot canonical config")
    canonical_digests = set()
    for path in config_artifacts.values():
        candidate = Path(path)
        if candidate.suffix.lower() != ".json":
            continue
        raw = json.loads(candidate.read_bytes())
        canonical_digests.add(_sha256_bytes(_json_bytes(raw)))
    if expected not in canonical_digests:
        raise ValueError("pilot canonical config is absent from the bound config artifacts")


def _source_hashes(
    asset_gen_root: Path, source_paths: Sequence[str]
) -> dict[str, str]:
    if not source_paths or len(set(source_paths)) != len(source_paths):
        raise ValueError("source path closure must be nonempty and unique")
    result: dict[str, str] = {}
    for relative in sorted(source_paths):
        path = _safe_child(asset_gen_root, relative)
        if not path.is_file():
            raise ValueError(f"recipe source is missing: {relative}")
        result[relative] = _sha256_file(path)
    return result


def derive_structural_repair_recipe(
    *,
    base: BaseConfig,
    plan: StructuralRepairPlan,
    corrected_base_manifest_path: Path,
    corrected_base_verification_path: Path,
    authority_manifest_path: Path,
    campaign_manifest_path: Path,
    pilot_manifest_path: Path,
    profile_artifacts: Mapping[str, Path],
    config_artifacts: Mapping[str, Path],
    geometry_artifacts: Mapping[str, Path],
    dtm_artifacts: Mapping[str, Path],
    asset_gen_root: Path = ASSET_GEN_ROOT,
    source_paths: Sequence[str] = _DEFAULT_SOURCE_PATHS,
) -> StructuralRepairRecipe:
    """Derive, rather than accept, the complete structural overlay identity."""
    base_identity = _validate_base(
        base, Path(corrected_base_manifest_path), Path(corrected_base_verification_path)
    )
    authority_identity = _validate_authority(Path(authority_manifest_path), plan)
    profile_hashes, geometry_hashes, campaign_evidence = _validate_campaign_bindings(
        Path(campaign_manifest_path),
        Path(pilot_manifest_path),
        profile_artifacts,
        geometry_artifacts,
    )
    _validate_pilot_config_binding(Path(pilot_manifest_path), config_artifacts)
    campaign_hashes = _artifact_hashes(
        {
            "campaign.json": Path(campaign_manifest_path),
            "pilot.json": Path(pilot_manifest_path),
        },
        "campaign",
    )
    grid = {
        "anchorE": base.grid.anchor_e,
        "anchorN": base.grid.anchor_n,
        "chunkMeters": base.grid.chunk_m,
        "chunkRes": base.grid.chunk_res,
        "lodStep": base.grid.lod_step,
    }
    inputs = {
        "id": RECIPE_ID,
        "recipeKind": RECIPE_KIND,
        "releaseDisposition": RELEASE_DISPOSITION,
        "correctedBase": base_identity,
        "structuralAuthority": authority_identity,
        "campaignSha256": campaign_hashes,
        "campaignEvidence": campaign_evidence,
        "configSha256": _artifact_hashes(config_artifacts, "config"),
        "geometrySha256": geometry_hashes,
        "profileSha256": profile_hashes,
        "dtmSha256": _artifact_hashes(dtm_artifacts, "DTM"),
        "plan": {
            "correctedLod0": [_key(chunk) for chunk in plan.corrected_lod0],
            "authority": _key(plan.authority_lod0),
            "reviewParent": _key(plan.review_parent),
            "publishedFine": [_key(chunk) for chunk in plan.published_lod2],
            "transientFineSupport": [
                _key(chunk) for chunk in plan.review_lod2_support
            ],
            "lod1Reducer": [_key(chunk) for chunk in plan.lod1_reducer],
            "lod2Reducer": [_key(chunk) for chunk in plan.lod2_reducer],
            "authoritySupport": [_key(chunk) for chunk in plan.authority_support],
            "expectedPublished": [
                ["height", *(_key(chunk))]
                for chunk in (*plan.published_lod2, plan.review_parent)
            ],
        },
        "packing": {
            "grid": grid,
            "codec": base.encode.codec,
            "parentReducer": BOX_MEAN_REDUCER_VERSION,
            "qscaleMeters": {
                "lod-2": FINE_QSCALE_M,
                "lod-1": 0.005,
                **{
                    f"lod{lod}": base.encode.height_qscale_for(lod)
                    for lod in range(5)
                },
            },
        },
        "sourceSha256": _source_hashes(Path(asset_gen_root), source_paths),
        "environment": _environment_identity(Path(asset_gen_root)),
    }
    digest = _sha256_bytes(RECIPE_ID.encode() + b"\0" + _json_bytes(inputs))
    return StructuralRepairRecipe(sha256=digest, inputs=inputs)


def expectation_document(recipe: StructuralRepairRecipe) -> dict[str, Any]:
    """Return the release-facing subset while retaining the complete recipe inputs."""
    expected = _sha256_bytes(RECIPE_ID.encode() + b"\0" + _json_bytes(recipe.inputs))
    if recipe.sha256 != expected:
        raise ValueError("structural repair recipe digest was not derived from its inputs")
    plan = recipe.inputs["plan"]
    return {
        "format": EXPECTATION_FORMAT,
        "recipeKind": RECIPE_KIND,
        "recipeSha256": recipe.sha256,
        "releaseDisposition": RELEASE_DISPOSITION,
        "baseManifestSha256": recipe.inputs["correctedBase"]["manifestSha256"],
        "recipeInputs": recipe.inputs,
        "grid": recipe.inputs["packing"]["grid"],
        "parent": plan["reviewParent"],
        "publishedFine": plan["publishedFine"],
        "transientSupport": plan["transientFineSupport"],
        "authority": plan["authority"],
        "expectedPublished": plan["expectedPublished"],
    }


def freeze_structural_repair_expectation(
    build_root: Path, recipe: StructuralRepairRecipe
) -> Path:
    """Freeze one immutable expectation before any release chunk is staged."""
    chunks_root = build_root / "chunks"
    if chunks_root.exists() and any(path.is_file() for path in chunks_root.rglob("*")):
        raise ValueError("structural expectation must be frozen before staged chunks exist")
    document = expectation_document(recipe)
    blob = (json.dumps(document, indent=1, sort_keys=True) + "\n").encode()
    path = build_root / "expectation.json"
    digest_path = build_root / "expectation.sha256"
    build_root.mkdir(parents=True, exist_ok=True)
    for destination, payload in (
        (path, blob),
        (digest_path, (_sha256_bytes(blob) + "\n").encode()),
    ):
        if destination.exists():
            if destination.read_bytes() != payload:
                raise ValueError(f"immutable expectation differs: {destination}")
            continue
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def load_structural_repair_expectation(
    path: Path, expected_recipe_sha256: str
) -> dict[str, Any]:
    """Validate both frozen-file identity and the content-derived recipe identity."""
    blob = path.read_bytes()
    if _sha256_bytes(blob) != path.with_name("expectation.sha256").read_text().strip():
        raise ValueError("structural expectation differs from expectation.sha256")
    document = json.loads(blob)
    if (
        document.get("format") != EXPECTATION_FORMAT
        or document.get("recipeKind") != RECIPE_KIND
        or document.get("releaseDisposition") != RELEASE_DISPOSITION
        or document.get("recipeSha256") != expected_recipe_sha256
    ):
        raise ValueError("structural expectation identity mismatch")
    inputs = document.get("recipeInputs")
    actual_recipe = _sha256_bytes(RECIPE_ID.encode() + b"\0" + _json_bytes(inputs))
    if actual_recipe != expected_recipe_sha256:
        raise ValueError("structural expectation recipe inputs do not derive its digest")
    if document != expectation_document(
        StructuralRepairRecipe(sha256=actual_recipe, inputs=inputs)
    ):
        raise ValueError("structural expectation duplicates disagree with recipe inputs")
    return document
