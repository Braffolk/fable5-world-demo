"""Immutable content-addressed materialization of a bound morphodynamic result."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .assemble import BoundDevelopmentA
from .qa import render_diagnostics
from .solver import MorphodynamicsResult
from .structural_base import (
    EAST_OUTPUT_SAMPLE_SLICE,
    FINE_CANVAS_BBOX_EN,
    FINE_TEXEL_M,
    OUTPUT_SAMPLE_SLICE,
    WEST_OUTPUT_SAMPLE_SLICE,
)

_MANIFEST_SCHEMA = "laas.erodible-slope-morphodynamics-artifact/1"


@dataclass(frozen=True)
class MorphodynamicsArtifact:
    root: Path
    manifest_path: Path
    content_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(document: object) -> bytes:
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write_bytes(root: Path, relative_path: str, payload: bytes) -> dict[str, Any]:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    return {
        "path": relative_path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_float(
    root: Path,
    relative_path: str,
    values: np.ndarray,
    *,
    role: str,
    bbox_en: tuple[float, float, float, float],
    canvas_slice: tuple[slice, slice],
) -> dict[str, Any]:
    array = np.asarray(values, dtype="<f8", order="C")
    if not np.isfinite(array).all():
        raise ValueError(f"{role} contains nonfinite values")
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        np.lib.format.write_array(output, array, allow_pickle=False)
        output.flush()
        os.fsync(output.fileno())
    return {
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "role": role,
        "dtype": "<f8",
        "shape": list(array.shape),
        "bboxEn": list(bbox_en),
        "canvasSlice": [
            [canvas_slice[0].start, canvas_slice[0].stop],
            [canvas_slice[1].start, canvas_slice[1].stop],
        ],
        "texelMeters": FINE_TEXEL_M,
        "samplePhase": {
            "firstCenterEastOffsetMeters": 0.5 * FINE_TEXEL_M,
            "firstCenterSouthOffsetMeters": 0.5 * FINE_TEXEL_M,
            "terminalRowAndColumn": "south/east neighbor first sample apron",
        },
    }


def _float_fields(
    temporary: Path,
    result: MorphodynamicsResult,
) -> list[dict[str, Any]]:
    windows = (
        (
            "stitched",
            OUTPUT_SAMPLE_SLICE,
            (680448.0, 6444416.0, 680704.0, 6444544.0),
        ),
        (
            "west",
            WEST_OUTPUT_SAMPLE_SLICE,
            (680448.0, 6444416.0, 680576.0, 6444544.0),
        ),
        (
            "east",
            EAST_OUTPUT_SAMPLE_SLICE,
            (680576.0, 6444416.0, 680704.0, 6444544.0),
        ),
    )
    records: list[dict[str, Any]] = []
    for window_name, window, bbox in windows:
        c0 = result.authority.c0_node_m[window]
        c1 = result.c1_node_m[window]
        for field_name, values in (
            ("c0", c0),
            ("c1", c1),
            ("delta", c1 - c0),
        ):
            records.append(
                _write_float(
                    temporary,
                    f"float/{window_name}-{field_name}.npy",
                    values,
                    role=f"{window_name}_{field_name}",
                    bbox_en=bbox,
                    canvas_slice=window,
                )
            )
    return records


def _verify_existing(
    root: Path,
    manifest_payload: bytes,
    document: dict[str, Any],
) -> MorphodynamicsArtifact:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.read_bytes() != manifest_payload:
        raise ValueError(f"immutable morphodynamics manifest differs: {root}")
    expected_files = {"manifest.json"}
    for record in document["artifacts"]:
        path = root / record["path"]
        expected_files.add(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or _sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"immutable morphodynamics artifact differs: {path}")
    actual_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("immutable morphodynamics artifact inventory differs")
    return MorphodynamicsArtifact(
        root=root,
        manifest_path=manifest_path,
        content_sha256=document["contentSha256"],
    )


def materialize_bound_result(
    bound: BoundDevelopmentA,
    result: MorphodynamicsResult,
    output_root: Path,
) -> MorphodynamicsArtifact:
    """Atomically create or byte-verify one fully bound float and QA artifact."""
    if result.authority is not bound.authority:
        raise ValueError("result authority is not the bound recipe authority")
    if result.c1_node_m.shape != bound.recipe.fixed_nested.expected_node_shape:
        raise ValueError("result C1 differs from the fixed inclusive node shape")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".tmp-{os.getpid()}-{uuid.uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        artifacts = _float_fields(temporary, result)
        metrics_document = {
            "schema": "laas.erodible-slope-morphodynamics-metrics/1",
            "recipeSha256": bound.recipe.sha256,
            "boundSemanticSha256": bound.semantic_sha256,
            "c0Sha256": bound.recipe.c0_sha256,
            "summary": asdict(result.summary),
            "sedimentEvent": asdict(result.sediment_event),
            "relaxationEvent": asdict(result.relaxation_event),
        }
        artifacts.append(
            _write_bytes(
                temporary,
                "metrics.json",
                _canonical_json(metrics_document),
            )
        )
        qa_records: list[dict[str, Any]] = []
        for image in render_diagnostics(result):
            identity = _write_bytes(temporary, image.relative_path, image.payload)
            qa_records.append({**identity, "interpretation": image.interpretation})
            artifacts.append(identity)
        qa_index = {
            "schema": "laas.erodible-slope-morphodynamics-qa/1",
            "recipeSha256": bound.recipe.sha256,
            "boundSemanticSha256": bound.semantic_sha256,
            "sourceRecipeSha256": bound.recipe.sha256,
            "sourceC0Sha256": bound.recipe.c0_sha256,
            "fineCanvasBboxEn": list(FINE_CANVAS_BBOX_EN),
            "images": qa_records,
        }
        artifacts.append(
            _write_bytes(temporary, "qa/index.json", _canonical_json(qa_index))
        )
        artifacts.sort(key=lambda value: value["path"])
        identity_document = {
            "schema": _MANIFEST_SCHEMA,
            "recipeSha256": bound.recipe.sha256,
            "boundSemanticSha256": bound.semantic_sha256,
            "c0Sha256": bound.recipe.c0_sha256,
            "spatialSha256": bound.recipe.spatial_sha256,
            "sourceRevision": bound.recipe.source_revision,
            "uvLockSha256": bound.recipe.uv_lock.sha256,
            "conditionBundleSha256": bound.recipe.bindings.condition_bundle.sha256,
            "evaluationPlanSha256": bound.recipe.bindings.evaluation_plan.sha256,
            "domainClosureSha256": bound.recipe.bindings.domain_closure.sha256,
            "processConfigSha256": bound.recipe.bindings.process_config.sha256,
            "structuralAuthoritySha256": (
                bound.recipe.bindings.structural_authority.sha256
            ),
            "artifacts": artifacts,
        }
        content_sha256 = hashlib.sha256(_canonical_json(identity_document)).hexdigest()
        document = {**identity_document, "contentSha256": content_sha256}
        manifest_payload = _canonical_json(document)
        _write_bytes(temporary, "manifest.json", manifest_payload)
        destination_parent = output_root / "sha256"
        destination_parent.mkdir(parents=True, exist_ok=True)
        destination = destination_parent / content_sha256
        if destination.exists():
            existing = _verify_existing(destination, manifest_payload, document)
            shutil.rmtree(temporary)
            return existing
        try:
            temporary.replace(destination)
        except FileExistsError:
            existing = _verify_existing(destination, manifest_payload, document)
            shutil.rmtree(temporary)
            return existing
        return _verify_existing(destination, manifest_payload, document)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
