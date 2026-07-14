import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from assetgen.height_geom import HeightChunkId
from assetgen.terrain.repair.plan import plan_structural_repair
from assetgen.terrain.repair.stage1_transaction import (
    _run_staging,
    _validate_water_transaction,
)


def _artifact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_staging_exports_only_review_overlay_and_replacement_candidates(
    tmp_path: Path,
) -> None:
    plan = plan_structural_repair(
        (HeightChunkId(0, 3, 0), HeightChunkId(0, 4, 0)),
        HeightChunkId(-1, 12, 0),
    )
    config = SimpleNamespace(plan=plan)
    materialization_root = tmp_path / "materialization"
    materialization_path = materialization_root / "transaction.json"
    materialization_path.parent.mkdir(parents=True)
    materialization_path.write_text("{}")
    fine = []
    for chunk in plan.published_lod2:
        stem = f"{chunk.cx}_{chunk.cz}"
        height = materialization_root / "chunks" / "height" / "-2" / f"{stem}.lac2"
        sidecar = height.with_suffix(".json")
        ownership = materialization_root / "ownership" / "-2" / f"{stem}.mask"
        _artifact(height, f"height:{stem}".encode())
        _artifact(sidecar, f"sidecar:{stem}".encode())
        _artifact(ownership, f"ownership:{stem}".encode())
        fine.append(
            {
                "key": [chunk.lod, chunk.cx, chunk.cz],
                "path": height.relative_to(materialization_root).as_posix(),
                "sidecar": sidecar.relative_to(materialization_root).as_posix(),
                "ownership": {
                    "path": ownership.relative_to(materialization_root).as_posix()
                },
            }
        )
    parent_height = (
        materialization_root
        / "chunks"
        / "height"
        / "-1"
        / f"{plan.review_parent.cx}_{plan.review_parent.cz}.lac2"
    )
    parent_sidecar = parent_height.with_suffix(".json")
    _artifact(parent_height, b"parent-height")
    _artifact(parent_sidecar, b"parent-sidecar")
    materialization = {
        "fine": fine,
        "parents": [
            {
                "key": [
                    plan.review_parent.lod,
                    plan.review_parent.cx,
                    plan.review_parent.cz,
                ],
                "path": parent_height.relative_to(materialization_root).as_posix(),
                "sidecar": parent_sidecar.relative_to(materialization_root).as_posix(),
                "ownership": None,
            }
        ],
    }

    base_root = tmp_path / "corrected-base"
    base_plan_path = base_root / "resolved-hierarchy-plan.json"
    base_rows = []
    for lod in (0, 1):
        path = base_root / "chunks" / str(lod) / f"0_0_{lod}.lac"
        _artifact(path, f"base:{lod}".encode())
        base_rows.append(
            {
                "chunk": [lod, 0, 0],
                "relative_path": path.relative_to(base_root).as_posix(),
            }
        )
    _artifact(base_plan_path, b"{}")
    base_plan = {"artifacts": base_rows}

    water_root = tmp_path / "corrected-water"
    water_path = water_root / "transaction.json"
    water_rows = []
    for lod, cx in ((0, 3), (0, 4), (1, 0), (1, 1)):
        disposition = "replacement" if cx == 0 else "remove-inherited"
        suffix = ".lac" if disposition == "replacement" else ".json"
        path = water_root / "chunks" / str(lod) / f"{cx}_0{suffix}"
        _artifact(path, f"water:{lod}:{cx}".encode())
        water_rows.append(
            {
                "key": [lod, cx, 0],
                "path": path.relative_to(water_root).as_posix(),
                "disposition": disposition,
            }
        )
    _artifact(water_path, b"{}")
    water = {"artifacts": water_rows}

    result_path, result_sha, document = _run_staging(
        root=tmp_path,
        transaction_sha256="ab" * 32,
        config=config,
        materialization_path=materialization_path,
        materialization=materialization,
        base_plan_path=base_plan_path,
        base_plan=base_plan,
        water_path=water_path,
        water=water,
    )

    assert document["negativeOverlayCount"] == 17
    assert document["negativeOverlayFileCount"] == 50
    assert document["correctedHeightCandidateCount"] == 2
    assert document["correctedWaterCandidateCount"] == 4
    assert len(document["files"]) == 56
    assert not (tmp_path / "stage/negative-overlay/height/0").exists()
    repeated_path, repeated_sha, repeated = _run_staging(
        root=tmp_path,
        transaction_sha256="ab" * 32,
        config=config,
        materialization_path=materialization_path,
        materialization=materialization,
        base_plan_path=base_plan_path,
        base_plan=base_plan,
        water_path=water_path,
        water=water,
    )
    assert repeated_path == result_path
    assert repeated_sha == result_sha
    assert repeated == document


def test_corrected_water_inventory_binds_exact_halos_children_and_files(
    tmp_path: Path,
) -> None:
    plan = plan_structural_repair(
        (HeightChunkId(0, 3, 0), HeightChunkId(0, 4, 0)),
        HeightChunkId(-1, 12, 0),
    )
    config = SimpleNamespace(plan=plan)
    digest = "cd" * 32
    artifacts = []
    for lod, cx in ((0, 3), (0, 4), (1, 0), (1, 1)):
        path = tmp_path / "chunks" / str(lod) / f"{cx}_0.lac"
        payload = f"water:{lod}:{cx}".encode()
        _artifact(path, payload)
        source_values = np.full((5, 5), np.nan, dtype="<f8")
        source_path = tmp_path / "evidence" / f"source-{lod}-{cx}.npy"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(source_path, source_values, allow_pickle=False)
        evidence_path = tmp_path / "evidence" / f"evidence-{lod}-{cx}.json"
        _artifact(evidence_path, b"{}")
        dependencies = (
            [[lod, cx, 0]]
            if lod == 0
            else [
                [0, cx * 4 + dx, dz]
                for dz in range(5)
                for dx in range(5)
            ]
        )
        artifacts.append(
            {
                "key": [lod, cx, 0],
                "disposition": "replacement",
                "path": path.relative_to(tmp_path).as_posix(),
                "bytes": len(payload),
                "fileSha256": hashlib.sha256(payload).hexdigest(),
                "sourceValuesSha256": hashlib.sha256(
                    source_values.tobytes()
                ).hexdigest(),
                "sourceValues": {
                    "path": source_path.relative_to(tmp_path).as_posix(),
                    "bytes": source_path.stat().st_size,
                    "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                },
                "evidenceSha256": hashlib.sha256(b"{}").hexdigest(),
                "evidence": {
                    "path": evidence_path.relative_to(tmp_path).as_posix(),
                    "bytes": evidence_path.stat().st_size,
                    "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                },
                "dependencies": [
                    {"key": key, "artifactSha256": digest}
                    for key in dependencies
                ],
            }
        )
    document = {
        "format": 1,
        "transactionVersion": "corrected-structural-water-transaction/1",
        "waterCoreResolution": 4,
        "halo": [
            {
                "key": [0, cx, 0],
                "dependencies": [
                    {
                        "key": [0, cx + dx, dz],
                        "source": "inherited",
                        "artifactSha256": digest,
                    }
                    for dz in (-1, 0, 1)
                    for dx in (-1, 0, 1)
                ],
            }
            for cx in (3, 4)
        ],
        "artifacts": artifacts,
    }

    _validate_water_transaction(document, tmp_path, config)
    (tmp_path / artifacts[0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="transaction artifact differs"):
        _validate_water_transaction(document, tmp_path, config)
