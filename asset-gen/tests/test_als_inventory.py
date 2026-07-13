from __future__ import annotations

import hashlib
import json
from pathlib import Path

import laspy
import numpy as np
import pytest

from assetgen.evidence import als_inventory


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_selection(
    data_in: Path,
    *,
    selection_id: str = "adjacent-evidence",
    scientific_role: str | None = "profile_spatial_closure_structural_repair",
) -> tuple[Path, dict]:
    root = data_in / "public" / "als" / selection_id
    snapshot_data = (
        json.dumps(
            {
                "id": selection_id,
                "stage": "stage1-structural-repair",
                "morphology_target": False,
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()
    snapshot_digest = _sha256(snapshot_data)
    snapshot = root / "snapshots" / "selection" / snapshot_digest / "selection.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(snapshot_data)

    staging = root / "point.las"
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    points = laspy.LasData(header)
    points.x = np.array([679_100.0, 679_100.25])
    points.y = np.array([6_444_100.0, 6_444_100.25])
    points.z = np.array([37.5, 37.6])
    points.classification = np.array([9, 9], dtype=np.uint8)
    points.write(staging)
    artifact_data = staging.read_bytes()
    artifact_digest = _sha256(artifact_data)
    artifact = root / "raw" / "sha256" / artifact_digest / "point.las"
    artifact.parent.mkdir(parents=True)
    staging.replace(artifact)
    artifact.with_suffix(".las.sha256").write_text(artifact_digest + "\n", encoding="ascii")

    provenance_data = b'{"status":200,"source":"official-fixture"}\n'
    provenance = root / "provenance" / "http" / "point.las.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_bytes(provenance_data)
    retained = {
        "format": 1,
        "selectionId": selection_id,
        "complete": True,
        "missingFiles": [],
        "morphologyTarget": False,
        "selectionSnapshot": snapshot.relative_to(root).as_posix(),
        "selectionSnapshotSha256": snapshot_digest,
        "artifacts": [
            {
                "year": 2019,
                "type": "lidar_las_tava",
                "filename": artifact.name,
                "role": "primary_profile_spatial_closure",
                "relativePath": artifact.relative_to(root).as_posix(),
                "bytes": len(artifact_data),
                "sha256": artifact_digest,
                "provenance": provenance.relative_to(root).as_posix(),
                "provenanceSha256": _sha256(provenance_data),
            }
        ],
    }
    if scientific_role is not None:
        retained["scientificRole"] = scientific_role
    retained_path = root / "retained.json"
    retained_path.write_text(
        json.dumps(retained, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return retained_path, retained


def test_explicit_inventory_uses_its_own_selection_root_and_retained_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_in = tmp_path / "data" / "in"
    retained_path, retained = _write_selection(data_in)
    retained_digest = _sha256(retained_path.read_bytes())
    monkeypatch.setattr(als_inventory, "DATA_IN", data_in)

    output = als_inventory.inventory_taevaskoda_als(
        retained_path, log=lambda _message: None
    )
    inventory = json.loads(output.read_bytes())

    assert output == retained_path.parent / "inventory" / retained_digest / "inventory.json"
    assert inventory["selectionId"] == retained["selectionId"]
    assert inventory["scientificRole"] == retained["scientificRole"]
    assert inventory["scientificRoleSource"] == "retained"
    assert inventory["morphologyTarget"] is False
    assert inventory["retainedManifestSha256"] == retained_digest
    assert inventory["artifacts"][0]["header"]["pointCount"] == 2
    assert inventory["artifacts"][0]["counts"]["classification"] == {"9": 2}


def test_missing_role_requires_a_validated_structural_selection_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_in = tmp_path / "data" / "in"
    retained_path, _ = _write_selection(data_in, scientific_role=None)
    monkeypatch.setattr(als_inventory, "DATA_IN", data_in)

    output = als_inventory.inventory_taevaskoda_als(
        retained_path, log=lambda _message: None
    )
    inventory = json.loads(output.read_bytes())

    assert inventory["scientificRole"] == "calibration_only_structural_repair"
    assert inventory["scientificRoleSource"] == "validated_selection_fallback"


@pytest.mark.parametrize("mutation", ["bytes", "sha256", "relativePath", "provenanceSha256"])
def test_explicit_inventory_rejects_mutated_retained_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    data_in = tmp_path / "data" / "in"
    retained_path, retained = _write_selection(data_in)
    monkeypatch.setattr(als_inventory, "DATA_IN", data_in)
    artifact = retained["artifacts"][0]
    if mutation == "bytes":
        artifact[mutation] += 1
    elif mutation == "relativePath":
        artifact[mutation] = "../point.las"
    else:
        artifact[mutation] = "0" * 64
    retained_path.write_text(
        json.dumps(retained, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError):
        als_inventory.inventory_taevaskoda_als(retained_path, log=lambda _message: None)


def test_default_inventory_keeps_primary_eight_epoch_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_in = tmp_path / "data" / "in"
    _write_selection(
        data_in,
        selection_id="taevaskoda-als-444679-stage1",
        scientific_role=None,
    )
    monkeypatch.setattr(als_inventory, "DATA_IN", data_in)

    with pytest.raises(ValueError, match="Taevaskoda ALS retention manifest"):
        als_inventory.inventory_taevaskoda_als(log=lambda _message: None)
