import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from assetgen.config import EncodeConfig
from assetgen.height_geom import HeightChunkId
from assetgen.release import publish_build
from assetgen.terrain.repair.verify import (
    _array_sha256,
    _verify_ownership,
    _verify_water,
    inherited_absent_water_sha256,
    load_verifier_inputs,
)
from assetgen.terrain.repair.materialize import _write_fine_ownership


ENCODE = EncodeConfig("deflate", 0.01, 0.01, 19, 1)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(root: Path, path: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _npy(path: Path, values: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values, allow_pickle=False)
    return path


def test_frozen_verifier_input_map_requires_every_hashed_artifact(tmp_path: Path) -> None:
    recipe = "1" * 64
    names = (
        "authority",
        "materializer",
        "correctedBasePlan",
        "correctedBaseVerification",
        "waterTransaction",
        "scientificClosure",
    )
    artifacts = {}
    for name in names:
        path = tmp_path / "inputs" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'{{"name":"{name}"}}\n')
        artifacts[name] = _entry(tmp_path, path)
    document = {"format": 1, "recipeSha256": recipe, "artifacts": artifacts}
    path = tmp_path / "structural-verify-inputs.json"
    path.write_text(json.dumps(document, sort_keys=True) + "\n")

    loaded, digest = load_verifier_inputs(tmp_path, recipe)
    assert loaded == document
    assert digest == _sha(path)

    (tmp_path / artifacts["authority"]["path"]).write_text("tampered")
    with pytest.raises(ValueError, match="authority input integrity"):
        load_verifier_inputs(tmp_path, recipe)


def _ownership_fixture(tmp_path: Path):
    chunk = HeightChunkId(-2, 3, 4)
    shape = (2049, 2049)
    baseline = np.full(shape, 12.0, dtype="<f4")
    values = baseline.copy()
    authority = np.zeros(shape, dtype=np.uint8)
    abstained = np.zeros(shape, dtype=np.uint8)
    authority[8:12, 12:16] = 1
    values[authority.astype(bool)] = 11.5
    abstained[20:24, 20:24] = 1
    artifact = _write_fine_ownership(
        tmp_path, chunk, authority.astype(bool), abstained.astype(bool), "e" * 64
    )
    sidecar = {
        "ownership": {
            "closureSha256": "e" * 64,
            "format": "TOM1/1",
            "path": artifact.relative_path,
            "bytes": artifact.bytes,
            "containerSha256": artifact.container_sha256,
            "authoritySha256": artifact.authority_sha256,
            "abstentionSha256": artifact.abstention_sha256,
            "authoritySamples": artifact.authority_samples,
            "abstainedSamples": artifact.abstained_samples,
        }
    }
    coarse_authority = np.zeros((512, 512), dtype=bool)
    coarse_forbidden = np.zeros((512, 512), dtype=bool)
    coarse_authority[2, 3] = True
    coarse_forbidden[2, 3] = True
    coarse_forbidden[5, 5] = True
    return chunk, values, sidecar, authority.astype(bool), abstained.astype(bool), baseline


def test_ownership_verifier_restores_baseline_and_forbids_fine_reauthorization(
    tmp_path: Path,
) -> None:
    chunk, values, sidecar, authority, abstained, baseline = _ownership_fixture(tmp_path)
    expected_authority = authority.copy()
    rerasterize = lambda _chunk: (expected_authority, abstained, baseline)
    evidence = _verify_ownership(
        tmp_path, {chunk: (values, sidecar)}, {}, "e" * 64, rerasterize
    )
    assert evidence["authoritySamples"] == 16
    assert evidence["abstainedSamples"] == 16

    authority[100, 100] = 1
    bad_root = tmp_path / "bad"
    bad_artifact = _write_fine_ownership(
        bad_root, chunk, authority.astype(bool), abstained.astype(bool), "e" * 64
    )
    bad_sidecar = {
        "ownership": {
            "closureSha256": "e" * 64,
            "format": "TOM1/1",
            "path": bad_artifact.relative_path,
            "bytes": bad_artifact.bytes,
            "containerSha256": bad_artifact.container_sha256,
            "authoritySha256": bad_artifact.authority_sha256,
            "abstentionSha256": bad_artifact.abstention_sha256,
            "authoritySamples": bad_artifact.authority_samples,
            "abstainedSamples": bad_artifact.abstained_samples,
        }
    }
    with pytest.raises(ValueError, match="invalid contract"):
        _verify_ownership(
            bad_root, {chunk: (values, bad_sidecar)}, {}, "e" * 64, rerasterize
        )

    values[100, 100] += np.float32(0.01)
    with pytest.raises(ValueError, match="did not exactly restore"):
        _verify_ownership(
            tmp_path, {chunk: (values, sidecar)}, {}, "e" * 64, rerasterize
        )


def test_all_dry_water_transaction_binds_fixed_absent_dependencies(tmp_path: Path) -> None:
    recipe = "2" * 64
    base_manifest = tmp_path / "base" / "manifest.json"
    water_index = tmp_path / "base" / "index" / "water.bin"
    water_index.parent.mkdir(parents=True)
    water_index.write_bytes(b"")
    base_manifest.write_text(
        json.dumps(
            {
                "format": 1,
                "chunkMeters": 8,
                "layers": {
                    "water": {
                        "index": "index/water.bin",
                        "indexSha256": _sha(water_index),
                    }
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    base_sha = _sha(base_manifest)
    parent = HeightChunkId(1, 1, 2)
    dependencies = [
        HeightChunkId(0, parent.cx * 4 + dx, parent.cz * 4 + dz)
        for dz in range(5)
        for dx in range(5)
    ]
    dependency_rows = [
        {"key": [chunk.lod, chunk.cx, chunk.cz], "artifactSha256": inherited_absent_water_sha256(chunk, base_sha)}
        for chunk in dependencies
    ]
    dependency_digest = hashlib.sha256(b"laas.structural-water.dependencies.v1\0")
    for chunk, row in zip(dependencies, dependency_rows, strict=True):
        dependency_digest.update(struct.pack("<Bii", chunk.lod, chunk.cx, chunk.cz))
        dependency_digest.update(bytes.fromhex(row["artifactSha256"]))
    dependency_sha = dependency_digest.hexdigest()
    source = np.full((5, 5), np.nan, dtype="<f8")
    source_path = _npy(tmp_path / "water" / "source.npy", source)
    evidence_sha = "c" * 64
    evidence_document = {"semanticSha256": evidence_sha}
    evidence_path = tmp_path / "water" / "evidence.json"
    evidence_path.write_text(json.dumps(evidence_document, sort_keys=True))
    source_sha = _array_sha256(source, "<f8")
    tombstone_hash = hashlib.sha256()
    tombstone_hash.update(b"laas.structural-water.absent.v1\0")
    tombstone_hash.update(struct.pack("<Bii", parent.lod, parent.cx, parent.cz))
    tombstone_hash.update(bytes.fromhex(source_sha))
    tombstone_hash.update(bytes.fromhex(evidence_sha))
    tombstone_hash.update(bytes.fromhex(dependency_sha))
    tombstone_document = {
        "format": 1,
        "tombstoneSha256": tombstone_hash.hexdigest(),
        "reason": "all_dry",
    }
    tombstone_path = tmp_path / "water" / "tombstone.json"
    tombstone_path.write_text(json.dumps(tombstone_document, sort_keys=True) + "\n")
    row = {
        "key": [parent.lod, parent.cx, parent.cz],
        "disposition": "remove-inherited",
        "artifactSha256": tombstone_hash.hexdigest(),
        "path": tombstone_path.name,
        "bytes": tombstone_path.stat().st_size,
        "fileSha256": _sha(tombstone_path),
        "sourceValues": _entry(tmp_path / "water", source_path),
        "sourceValuesSha256": source_sha,
        "evidence": _entry(tmp_path / "water", evidence_path),
        "evidenceSha256": evidence_sha,
        "dependencies": dependency_rows,
        "dependencySha256": dependency_sha,
        "drySamples": 25,
    }
    halo_dependencies = [
        {
            "key": item["key"],
            "source": "explicit_absent",
            "artifactSha256": item["artifactSha256"],
        }
        for item in dependency_rows[:9]
    ]
    halo_digest = hashlib.sha256(b"laas.structural-water.halo-dependencies.v1\0")
    for item in halo_dependencies:
        lod, cx, cz = item["key"]
        encoded_source = item["source"].encode()
        halo_digest.update(struct.pack("<BiiB", lod, cx, cz, len(encoded_source)))
        halo_digest.update(encoded_source)
        halo_digest.update(bytes.fromhex(item["artifactSha256"]))
    transaction = {
        "format": 1,
        "transactionVersion": "corrected-structural-water-transaction/1",
        "sourceManifestSha256": base_sha,
        "halo": [{"key": [0, 4, 8], "dependencies": halo_dependencies, "dependencySha256": halo_digest.hexdigest()}],
        "artifacts": [row],
    }
    transaction_path = tmp_path / "water" / "transaction.json"
    transaction_path.write_text(json.dumps(transaction, sort_keys=True) + "\n")

    present, tombstones, evidence = _verify_water(
        transaction_path,
        _sha(transaction_path),
        recipe,
        tmp_path,
        base_manifest,
        tmp_path / "content",
        ENCODE,
        base_sha,
    )
    assert not present
    assert tombstones == {("water", 1, 1, 2)}
    assert evidence["transactionSha256"] == _sha(transaction_path)

    row["dependencies"][0]["artifactSha256"] = "f" * 64
    transaction_path.write_text(json.dumps(transaction, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="dependency identity mismatch"):
        _verify_water(
            transaction_path,
            _sha(transaction_path),
            recipe,
            tmp_path,
            base_manifest,
            tmp_path / "content",
            ENCODE,
            base_sha,
        )


def test_structural_overlay_can_never_update_latest(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "out"
    out.mkdir()
    latest = out / "latest.json"
    latest.write_text('{"manifest":"old"}\n')
    monkeypatch.setattr(
        "assetgen.release._load_plan",
        lambda *_args, **_kwargs: (
            tmp_path,
            {"microRecipeKind": "structural-repair-overlay-v1"},
            b"plan",
        ),
    )
    with pytest.raises(ValueError, match="preview-only"):
        publish_build("0" * 64, tmp_path, out)
    assert latest.read_text() == '{"manifest":"old"}\n'
