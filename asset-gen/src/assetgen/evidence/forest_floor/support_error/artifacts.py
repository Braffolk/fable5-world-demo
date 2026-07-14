"""Content-addressed publication for the forest-floor support/error decision."""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .audit import canonical_json_bytes, sha256_file
from .render import render_evidence_ladder


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())


def _reference(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def publish_support_error_decision(
    decision: dict[str, Any],
    output_root: Path,
) -> Path:
    payload = canonical_json_bytes(decision)
    build_id = hashlib.sha256(payload).hexdigest()
    final = output_root.resolve() / "sha256" / build_id
    if final.exists():
        manifest = final / "manifest.json"
        if manifest.is_file():
            return manifest
        raise FileExistsError(f"Incomplete forest-floor qualification build exists: {final}")

    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=final.parent))
    try:
        decision_path = staging / "support-error-decision.json"
        _write_bytes(decision_path, payload)
        qa_path = staging / "01_support_error_evidence_ladder.png"
        render_evidence_ladder(decision, qa_path)
        with qa_path.open("rb") as source:
            os.fsync(source.fileno())
        manifest_value = {
            "schema_version": "forest-floor-support-error-build/1.0.0",
            "build_id": build_id,
            "status": "complete_fail_closed",
            "decision": decision["decision"],
            "qualification": {
                "qualified_target_B1": False,
                "qualified_target_B2": False,
                "synthesis_authorized": False,
            },
            "artifacts": {
                "machine_decision": _reference(decision_path),
                "qa_png": {
                    **_reference(qa_path),
                    "dimensions_px": [1500, 1009],
                    "interpretation": (
                        "Evidence ladder separating retained raw geometry from missing semantic, "
                        "independent-control, repeatability, and B1/B2 qualification gates."
                    ),
                },
            },
        }
        manifest_path = staging / "manifest.json"
        _write_bytes(manifest_path, canonical_json_bytes(manifest_value))
        os.replace(staging, final)
        return final / "manifest.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
