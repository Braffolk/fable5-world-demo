import hashlib
import json
from pathlib import Path

import pytest

from assetgen.config import load_base
from assetgen.cook.micro_hierarchy import BOX_MEAN_REDUCER_VERSION
from assetgen.height_geom import HeightChunkId
from assetgen.terrain.repair.plan import plan_structural_repair
from assetgen.terrain.repair.recipe import (
    RECIPE_KIND,
    RELEASE_DISPOSITION,
    derive_structural_repair_recipe,
    freeze_structural_repair_expectation,
    load_structural_repair_expectation,
)
from assetgen.terrain.repair.storage import baseline_tile_contract


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _fixture(tmp_path: Path):
    base = load_base()
    plan = plan_structural_repair(
        (HeightChunkId(0, 2, 3), HeightChunkId(0, 3, 3)),
        HeightChunkId(-1, 9, 12),
    )

    base_root = tmp_path / "corrected-base"
    height_index = _write(base_root / "index" / "height.bin", b"height-index")
    trees_index = _write(base_root / "index" / "trees.bin", b"trees-index")
    manifest = {
        "format": 1,
        "anchor": {"e": base.grid.anchor_e, "n": base.grid.anchor_n},
        "chunkMeters": base.grid.chunk_m,
        "chunkRes": base.grid.chunk_res,
        "lodStep": base.grid.lod_step,
        "codec": base.encode.codec,
        "layers": {
            "height": {
                "index": "index/height.bin",
                "indexSha256": _sha(height_index),
            },
            "trees": {
                "index": "index/trees.bin",
                "indexSha256": _sha(trees_index),
            },
        },
    }
    manifest_path = _write(
        base_root / "manifest.json",
        (json.dumps(manifest, sort_keys=True) + "\n").encode(),
    )
    verification_path = _write(
        tmp_path / "corrected-base-verify.json",
        (
            json.dumps(
                {"format": 1, "passed": True, "manifestSha256": _sha(manifest_path)},
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )

    authority_root = tmp_path / "authority"
    rows = []
    for chunk in plan.authority_support:
        artifact = _write(
            authority_root / "tiles" / f"{chunk.cx}_{chunk.cz}.npz",
            f"{chunk.lod}/{chunk.cx}/{chunk.cz}".encode(),
        )
        baseline = _write(
            authority_root / "baseline" / f"{chunk.cx}_{chunk.cz}.npz",
            f"baseline/{chunk.lod}/{chunk.cx}/{chunk.cz}".encode(),
        )
        rows.append(
            {
                "key": [chunk.lod, chunk.cx, chunk.cz],
                "path": artifact.relative_to(authority_root).as_posix(),
                "bytes": artifact.stat().st_size,
                "sha256": _sha(artifact),
                "baselinePath": baseline.relative_to(authority_root).as_posix(),
                "baselineBytes": baseline.stat().st_size,
                "baselineSha256": _sha(baseline),
            }
        )
    authority = {
        "format": 2,
        "role": "structural_authority_0.25m",
        "morphology": "absent",
        "recipeSha256": "a" * 64,
        "evidenceSha256": "b" * 64,
        "tileAlignmentLod": -2,
        "tileTexelMeters": 0.25,
        "tileCoreResolution": 512,
        "baselineContract": baseline_tile_contract(),
        "tiles": rows,
    }
    authority_path = _write(
        authority_root / "manifest.json",
        (json.dumps(authority, sort_keys=True) + "\n").encode(),
    )

    campaign_root = tmp_path / "campaign"
    profile_path = _write(campaign_root / "campaign.npz", b"qualified-profile")
    geometry = {
        name: _write(campaign_root / name, name.encode())
        for name in (
            "downstream-centerline.wkb",
            "downstream-water-cap.wkb",
            "full-centerline.wkb",
            "full-water-cap.wkb",
        )
    }
    campaign_path = _write(
        campaign_root / "campaign.json",
        (
            json.dumps(
                {
                    "format": 1,
                    "arraysNpzSha256": _sha(profile_path),
                    "campaign": {"qualificationSha256": "c" * 64},
                },
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )
    pilot_path = _write(
        campaign_root / "pilot.json",
        (
            json.dumps(
                {
                    "format": 1,
                    "campaignMetadataSha256": _sha(campaign_path),
                    "campaignArraysSha256": _sha(profile_path),
                    "campaignContentSha256": "d" * 64,
                    "configCanonicalSha256": "pending",
                    "geometrySha256": {
                        name: _sha(path) for name, path in geometry.items()
                    },
                },
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )

    config_path = _write(
        tmp_path / "config.json",
        (json.dumps({"format": 1, "pilot": "test"}, indent=2) + "\n").encode(),
    )
    pilot_document = json.loads(pilot_path.read_text())
    canonical_config = (
        json.dumps(
            json.loads(config_path.read_text()), sort_keys=True, separators=(",", ":")
        )
        + "\n"
    ).encode()
    pilot_document["configCanonicalSha256"] = hashlib.sha256(
        canonical_config
    ).hexdigest()
    pilot_path.write_text(json.dumps(pilot_document, sort_keys=True) + "\n")
    dtm_path = _write(tmp_path / "dtm.tif", b"dtm")
    source_root = tmp_path / "asset-gen"
    _write(source_root / "pyproject.toml", b"project")
    _write(source_root / "uv.lock", b"lock")
    _write(source_root / "src" / "algorithm.py", b"algorithm")
    arguments = {
        "base": base,
        "plan": plan,
        "corrected_base_manifest_path": manifest_path,
        "corrected_base_verification_path": verification_path,
        "authority_manifest_path": authority_path,
        "campaign_manifest_path": campaign_path,
        "pilot_manifest_path": pilot_path,
        "profile_artifacts": {"campaign.npz": profile_path},
        "config_artifacts": {"terrain-repair.json": config_path},
        "geometry_artifacts": geometry,
        "dtm_artifacts": {"dtm.tif": dtm_path},
        "asset_gen_root": source_root,
        "source_paths": ("src/algorithm.py",),
    }
    return arguments, dtm_path, plan


def test_recipe_is_content_derived_and_binds_complete_plan(tmp_path: Path) -> None:
    arguments, dtm_path, plan = _fixture(tmp_path)
    first = derive_structural_repair_recipe(**arguments)
    second = derive_structural_repair_recipe(**arguments)

    assert first == second
    assert first.inputs["recipeKind"] == RECIPE_KIND
    assert first.inputs["releaseDisposition"] == RELEASE_DISPOSITION
    assert first.inputs["packing"]["parentReducer"] == BOX_MEAN_REDUCER_VERSION
    assert len(first.inputs["plan"]["publishedFine"]) == 16
    assert len(first.inputs["plan"]["transientFineSupport"]) == 9
    assert len(first.inputs["plan"]["correctedLod0"]) == 2
    assert len(first.inputs["plan"]["lod1Reducer"]) == 45
    assert len(first.inputs["plan"]["lod2Reducer"]) == 777
    assert len(first.inputs["plan"]["authoritySupport"]) == 897
    assert first.inputs["structuralAuthority"]["pairCount"] == 897
    assert first.inputs["structuralAuthority"]["structuralTileCount"] == 897
    assert first.inputs["structuralAuthority"]["baselineTileCount"] == 897
    assert first.inputs["plan"]["authority"] == [
        plan.authority_lod0.lod,
        plan.authority_lod0.cx,
        plan.authority_lod0.cz,
    ]

    dtm_path.write_bytes(b"different-dtm")
    changed = derive_structural_repair_recipe(**arguments)
    assert changed.sha256 != first.sha256


def test_freeze_and_load_rederive_identity_and_reject_staged_chunks(
    tmp_path: Path,
) -> None:
    arguments, _, _ = _fixture(tmp_path)
    recipe = derive_structural_repair_recipe(**arguments)
    build_root = tmp_path / "build"
    path = freeze_structural_repair_expectation(build_root, recipe)

    loaded = load_structural_repair_expectation(path, recipe.sha256)
    assert loaded["recipeSha256"] == recipe.sha256
    assert loaded["releaseDisposition"] == "preview-only"
    assert freeze_structural_repair_expectation(build_root, recipe) == path

    _write(build_root / "chunks" / "height" / "-2" / "0_0.lac", b"staged")
    with pytest.raises(ValueError, match="before staged chunks"):
        freeze_structural_repair_expectation(build_root, recipe)


def test_recipe_rejects_unverified_base_and_incomplete_authority(
    tmp_path: Path,
) -> None:
    arguments, _, _ = _fixture(tmp_path)
    verification = arguments["corrected_base_verification_path"]
    document = json.loads(verification.read_text())
    document["passed"] = False
    verification.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="has not passed"):
        derive_structural_repair_recipe(**arguments)

    document["passed"] = True
    verification.write_text(json.dumps(document))
    authority = arguments["authority_manifest_path"]
    authority_document = json.loads(authority.read_text())
    authority_document["tiles"].pop()
    authority.write_text(json.dumps(authority_document))
    with pytest.raises(ValueError, match="tile set differs"):
        derive_structural_repair_recipe(**arguments)


def test_recipe_rejects_missing_tampered_and_aliased_baselines(
    tmp_path: Path,
) -> None:
    arguments, _, _ = _fixture(tmp_path)
    manifest_path = arguments["authority_manifest_path"]
    pristine = manifest_path.read_bytes()
    document = json.loads(pristine)
    first = document["tiles"][0]
    second = document["tiles"][1]
    root = manifest_path.parent
    baseline_path = root / first["baselinePath"]
    baseline_payload = baseline_path.read_bytes()

    baseline_path.unlink()
    with pytest.raises(ValueError, match="pair is missing"):
        derive_structural_repair_recipe(**arguments)

    baseline_path.write_bytes(baseline_payload + b"tampered")
    with pytest.raises(ValueError, match="pair identity mismatch"):
        derive_structural_repair_recipe(**arguments)

    baseline_path.write_bytes(baseline_payload)
    second["baselinePath"] = first["baselinePath"]
    second["baselineBytes"] = first["baselineBytes"]
    second["baselineSha256"] = first["baselineSha256"]
    manifest_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="alias an artifact"):
        derive_structural_repair_recipe(**arguments)
