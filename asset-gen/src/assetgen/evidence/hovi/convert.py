"""Immutable first conversion of one unsealed Hovi development point cloud."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Callable

import laspy
import numpy as np
from PIL import Image

from ...config import CONFIG_DIR, DATA_WORK
from .qa import render_support_qa
from .records import (
    HoviSelection,
    RetainedSelection,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from .support import SUPPORT_RESOLUTIONS_M, inventory_observed_support

_CONVERTER_SCHEMA = "hovi-observation-support-converter/1.0.0"
_INVENTORY_SCHEMA = "hovi-observation-support-inventory/1.0.0"
_MANIFEST_SCHEMA = "hovi-observation-support-build/1.0.0"
_QA_SCHEMA = "hovi-observation-support-qa/1.0.0"


def _environment() -> dict[str, str]:
    return {
        "numpy": np.__version__,
        "laspy": laspy.__version__,
        "lazrs": importlib.metadata.version("lazrs"),
        "pillow": importlib.metadata.version("pillow"),
    }


def _atomic_write(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as target:
        target.write(encoded)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _store_content_file(build_root: Path, source: Path, filename: str) -> tuple[Path, str]:
    digest = sha256_file(source)
    destination = build_root / "artifacts" / "sha256" / digest / filename
    if destination.exists():
        if (
            destination.stat().st_size != source.stat().st_size
            or sha256_file(destination) != digest
        ):
            raise ValueError(f"corrupt Hovi content-addressed artifact: {destination}")
        source.unlink()
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
    return destination, digest


def _store_content_bytes(
    build_root: Path, encoded: bytes, filename: str
) -> tuple[Path, str]:
    digest = sha256_bytes(encoded)
    destination = build_root / "artifacts" / "sha256" / digest / filename
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise ValueError(f"corrupt Hovi content-addressed artifact: {destination}")
    else:
        _atomic_write(destination, encoded)
    return destination, digest


def _artifact_ref(build_root: Path, path: Path, digest: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(build_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest,
    }


def _verify_ref(build_root: Path, reference: dict[str, Any]) -> None:
    relative = reference.get("path")
    digest = reference.get("sha256")
    byte_count = reference.get("bytes")
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or not isinstance(digest, str)
        or len(digest) != 64
        or not isinstance(byte_count, int)
        or byte_count <= 0
    ):
        raise ValueError("invalid Hovi immutable artifact reference")
    path = (build_root / relative).resolve()
    if (
        not path.is_relative_to(build_root.resolve())
        or not path.is_file()
        or path.stat().st_size != byte_count
        or sha256_file(path) != digest
    ):
        raise ValueError(f"Hovi immutable artifact failed verification: {relative}")


def _reuse_complete_build(
    build_root: Path, *, build_id: str, recipe_sha256: str
) -> Path | None:
    manifest_path = build_root / "manifest.json"
    if not manifest_path.exists():
        return None
    raw = json.loads(manifest_path.read_bytes())
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != _MANIFEST_SCHEMA
        or raw.get("status") != "complete"
        or raw.get("build_id") != build_id
        or raw.get("recipe_sha256") != recipe_sha256
    ):
        raise ValueError(f"existing Hovi build has conflicting identity: {manifest_path}")
    recipe_path = build_root / "recipe.json"
    if not recipe_path.is_file() or sha256_file(recipe_path) != recipe_sha256:
        raise ValueError("existing Hovi build has a missing or corrupt recipe")
    for label in ("inventory", "support_npz", "qa_index"):
        reference = raw.get(label)
        if not isinstance(reference, dict):
            raise ValueError(f"existing Hovi build lacks {label}")
        _verify_ref(build_root, reference)
    qa_reference = raw["qa_index"]
    qa_path = build_root / qa_reference["path"]
    qa = json.loads(qa_path.read_bytes())
    if not isinstance(qa, dict) or qa.get("build_id") != build_id:
        raise ValueError("existing Hovi build has an invalid QA index")
    images = qa.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("existing Hovi build has no QA images")
    for reference in images:
        if not isinstance(reference, dict):
            raise ValueError("existing Hovi build has an invalid QA image reference")
        _verify_ref(build_root, reference)
    return manifest_path


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            with archive.open(info, mode="w", force_zip64=True) as member:
                np.lib.format.write_array(member, arrays[name], allow_pickle=False)
    with path.open("rb") as source:
        os.fsync(source.fileno())


def _plot_identity(plot: Any) -> dict[str, Any]:
    return {
        "plot_id": plot.plot_id,
        "selection_role": plot.role,
        "site_id": plot.site_id,
        "campaign_id": plot.campaign_id,
        "nominal_layout_m": list(plot.nominal_layout_m),
    }


def convert_hovi_plot(
    retained_path: Path,
    plot_id: str,
    *,
    selection_path: Path = CONFIG_DIR / "hovi-public-targets.json",
    work_root: Path = DATA_WORK,
    chunk_points: int = 1_000_000,
    log: Callable[[str], None] = print,
) -> Path:
    """Inventory one development cloud; never classify or reconstruct a ground surface."""
    selection = HoviSelection.load(selection_path)
    plot = selection.development_plot(plot_id)
    selected_artifact = plot.geometry_artifact()
    retained = RetainedSelection.load(retained_path, selection=selection)
    retained_artifact = retained.artifact_for(selected_artifact)
    if (
        retained_artifact.plot_id != plot.plot_id
        or retained_artifact.tranche != "hy-spruce4-geometry"
    ):
        raise ValueError("Hovi retained geometry does not belong to the requested plot")

    log(f"verifying retained Hovi source bytes: {retained_artifact.local_path.name}")
    actual_source_sha256 = sha256_file(retained_artifact.local_path)
    if actual_source_sha256 != selected_artifact.sha256:
        raise ValueError("Hovi retained geometry differs from the frozen source tuple")

    recipe = {
        "schema_version": _CONVERTER_SCHEMA,
        "selection": {
            "schema_version": selection.schema_version,
            "id": selection.selection_id,
            "config_sha256": selection.sha256,
        },
        "retention_id": retained.retention_id,
        "plot": _plot_identity(plot),
        "source_artifact": {
            "file_id": selected_artifact.file_id,
            "path": selected_artifact.source_path,
            "bytes": selected_artifact.bytes,
            "sha256": selected_artifact.sha256,
            "kind": selected_artifact.kind,
        },
        "support": {
            "resolutions_m": list(SUPPORT_RESOLUTIONS_M),
            "extent_policy": "global-grid-aligned-las-header-xy-envelope-inclusive-max",
            "point_count": "all decoded finite XYZ points within declared LAS extents",
            "nearest_distance": (
                "minimum planar distance from cell center to a decoded point assigned to "
                "that same half-open cell; unavailable for empty cells"
            ),
        },
        "environment": _environment(),
    }
    recipe_bytes = canonical_json_bytes(recipe)
    build_id = sha256_bytes(recipe_bytes)
    build_root = work_root / "microtopography" / "hovi" / build_id
    existing = _reuse_complete_build(
        build_root, build_id=build_id, recipe_sha256=build_id
    )
    if existing is not None:
        log(f"reused complete Hovi support build {build_id}")
        return existing

    build_root.mkdir(parents=True, exist_ok=True)
    _atomic_write(build_root / "recipe.json", recipe_bytes)
    inventory = inventory_observed_support(
        retained_artifact.local_path, chunk_points=chunk_points, log=log
    )
    arrays = inventory.arrays()
    temporary_npz = build_root / ".support.npz.part"
    _write_npz(temporary_npz, arrays)
    support_path, support_sha256 = _store_content_file(
        build_root, temporary_npz, "support.npz"
    )

    array_schema = {
        name: {"dtype": str(array.dtype), "shape": list(array.shape)}
        for name, array in arrays.items()
    }
    limitations = {
        "scan_identity": {
            "status": "unavailable",
            "reason": "the selected source is a merged, thinned LAZ without scan identity",
        },
        "unique_view_count": {"status": "unavailable", "reason": "scan identity is absent"},
        "angular_diversity": {"status": "unavailable", "reason": "scan identity is absent"},
        "occlusion": {
            "status": "unavailable",
            "reason": "merged observed points do not encode unobserved rays or visibility",
        },
        "independent_error": {
            "status": "unavailable",
            "reason": "merged returns cannot be separated into independent observations",
        },
        "point_spacing": {
            "status": "unavailable",
            "reason": (
                "vertical structure and merged scans make raw XY nearest-neighbor spacing an "
                "invalid proxy for independent ground measurement spacing"
            ),
        },
    }
    inventory_record = {
        "schema_version": _INVENTORY_SCHEMA,
        "build_id": build_id,
        "scientific_disposition": {
            "role": "raw_candidate",
            "qualification_status": "unqualified",
            "target_truth": False,
            "transfer_ceiling": "none",
            "synthesis_authorized": False,
        },
        "selection": {
            "id": selection.selection_id,
            "config_sha256": selection.sha256,
        },
        "retention": {
            "retention_id": retained.retention_id,
        },
        "plot": _plot_identity(plot),
        "source_artifact": {
            "file_id": selected_artifact.file_id,
            "path": selected_artifact.source_path,
            "kind": selected_artifact.kind,
            "bytes": selected_artifact.bytes,
            "sha256": actual_source_sha256,
            "retained_relative_path": retained_artifact.local_path.relative_to(
                retained.path.parent
            ).as_posix(),
        },
        "interpretation_boundary": {
            "lowest_point_ground_extraction": "not_performed",
            "point_class_semantics": "not_inferred",
            "morphology_synthesis": "not_performed",
            "support_meaning": (
                "raw all-return observation support only; occupied cells do not establish "
                "forest-floor or ground support"
            ),
        },
        "limitations": limitations,
        "header": inventory.header,
        "point_counts": inventory.point_counts,
        "observed_extents": inventory.observed_extents,
        "invalids": inventory.invalids,
        "categorical_counts_without_semantic_relabeling": inventory.categorical_counts,
        "support_grids": [grid.metadata() for grid in inventory.grids],
        "support_npz": {
            **_artifact_ref(build_root, support_path, support_sha256),
            "arrays": array_schema,
        },
        "environment": recipe["environment"],
    }
    inventory_bytes = canonical_json_bytes(inventory_record)
    inventory_path, inventory_sha256 = _store_content_bytes(
        build_root, inventory_bytes, "inventory.json"
    )

    qa_dir = build_root / "qa"
    qa_images = render_support_qa(inventory.grids, qa_dir, plot_id=plot.plot_id)
    qa_records = []
    for path, interpretation in qa_images:
        digest = sha256_file(path)
        with Image.open(path) as image:
            dimensions = list(image.size)
        qa_records.append(
            {
                "path": path.relative_to(build_root).as_posix(),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "dimensions_xy": dimensions,
                "interpretation": interpretation,
            }
        )
    qa_index = {
        "schema_version": _QA_SCHEMA,
        "build_id": build_id,
        "recipe_sha256": build_id,
        "source_sha256": actual_source_sha256,
        "inventory_sha256": inventory_sha256,
        "support_npz_sha256": support_sha256,
        "images": qa_records,
    }
    qa_index_bytes = canonical_json_bytes(qa_index)
    qa_index_path = qa_dir / "index.json"
    _atomic_write(qa_index_path, qa_index_bytes)
    _atomic_write(
        qa_index_path.with_suffix(".json.sha256"),
        (sha256_bytes(qa_index_bytes) + "\n").encode("ascii"),
    )

    manifest = {
        "schema_version": _MANIFEST_SCHEMA,
        "status": "complete",
        "build_id": build_id,
        "recipe_sha256": build_id,
        "plot_id": plot.plot_id,
        "scientific_role": "raw_candidate",
        "qualification_status": "unqualified",
        "transfer_ceiling": "none",
        "synthesis_authorized": False,
        "inventory": _artifact_ref(build_root, inventory_path, inventory_sha256),
        "support_npz": _artifact_ref(build_root, support_path, support_sha256),
        "qa_index": _artifact_ref(
            build_root, qa_index_path, sha256_bytes(qa_index_bytes)
        ),
    }
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path = build_root / "manifest.json"
    _atomic_write(manifest_path, manifest_bytes)
    _atomic_write(
        manifest_path.with_suffix(".json.sha256"),
        (sha256_bytes(manifest_bytes) + "\n").encode("ascii"),
    )
    log(f"completed Hovi raw-support inventory {build_id}")
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory one unsealed Hovi development plot without ground extraction."
    )
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--plot", required=True, help="Named unsealed development plot")
    parser.add_argument(
        "--selection",
        type=Path,
        default=CONFIG_DIR / "hovi-public-targets.json",
    )
    parser.add_argument("--work-root", type=Path, default=DATA_WORK)
    parser.add_argument("--chunk-points", type=int, default=1_000_000)
    args = parser.parse_args()
    path = convert_hovi_plot(
        args.retained,
        args.plot,
        selection_path=args.selection,
        work_root=args.work_root,
        chunk_points=args.chunk_points,
    )
    print(f"artifact manifest: {path}")


if __name__ == "__main__":
    _main()
