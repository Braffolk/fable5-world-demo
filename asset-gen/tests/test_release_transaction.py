import hashlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from assetgen.cli import main
from assetgen.config import load_base
from assetgen.cook.chunkio import ChunkMeta, write_chunk, write_chunk_v2
from assetgen.height_geom import plan_hero
from assetgen.release import (
    _measured_synthesis_cook_revision,
    audit_base_release,
    create_build_plan,
    create_micro_expectation,
    materialize_preview,
    publish_build,
)

BUILD = "a" * 64


def test_measured_release_revision_must_match_cook_evidence(tmp_path):
    build = "b" * 64
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "micro-synthesis-cook.json").write_text(json.dumps({
        "format": 1,
        "recipeSha256": build,
        "cook": "exemplar-driven-production-pilot-v1",
        "cookRevision": 3,
    }))

    assert _measured_synthesis_cook_revision(tmp_path, build, 3) == 3
    with pytest.raises(ValueError, match="differs from measured-synthesis evidence"):
        _measured_synthesis_cook_revision(tmp_path, build, 2)


def _stage_chunk(
    work_root: Path,
    build: str = BUILD,
    *,
    layer: str = "height",
    lod: int = 0,
    cx: int = 0,
    cz: int = 0,
    payload: bytes = b"synthetic-payload",
    container_version: int = 1,
    qscale: float | None = None,
) -> Path:
    base = load_base()
    footprint = base.grid.chunk_m * base.grid.lod_step**lod
    path = work_root / "builds" / build / "chunks" / layer / str(lod) / f"{cx}_{cz}.lac"
    writer = write_chunk if container_version == 1 else write_chunk_v2
    writer(
        path,
        ChunkMeta(
            layer=layer,
            lod=lod,
            enc=1 if layer == "height" else 2,
            cx=cx,
            cz=cz,
            res=2049,
            count=0,
            origin_e=base.grid.anchor_e + cx * footprint,
            origin_n=base.grid.anchor_n - cz * footprint,
            qoffset=0.0,
            qscale=qscale if qscale is not None else (0.002 if lod == -2 else 0.005 if lod == -1 else 0.01),
        ),
        payload,
    )
    return path


def _plan_one(tmp_path: Path) -> tuple[Path, Path, Path]:
    work = tmp_path / "work"
    out = tmp_path / "out"
    _stage_chunk(work)
    plan = create_build_plan(load_base(), BUILD, 7, work_root=work)
    return work, out, plan


def test_transaction_builds_preview_then_publishes_latest(tmp_path):
    work, out, _ = _plan_one(tmp_path)
    preview = materialize_preview(BUILD, work_root=work, out_root=out)

    assert preview.is_file()
    assert not (out / "latest.json").exists()
    assert (work / "builds" / BUILD / "verify.json").is_file()
    assert (work / "builds" / BUILD / "COMPLETE").is_file()

    published = publish_build(BUILD, work_root=work, out_root=out)
    latest = json.loads((out / "latest.json").read_text())
    assert published == out / latest["manifest"]
    manifest_sha = hashlib.sha256(published.read_bytes()).hexdigest()
    audit = audit_base_release(published, manifest_sha, out)
    assert audit.chunk_count == 1


def test_incomplete_build_cannot_publish_or_touch_latest(tmp_path):
    work, out, _ = _plan_one(tmp_path)
    (out / "latest.json").parent.mkdir(parents=True)
    (out / "latest.json").write_text('{"manifest":"old"}\n')

    with pytest.raises(FileNotFoundError):
        publish_build(BUILD, work_root=work, out_root=out)
    assert json.loads((out / "latest.json").read_text()) == {"manifest": "old"}


def test_unplanned_chunk_rejects_build(tmp_path):
    work, out, _ = _plan_one(tmp_path)
    _stage_chunk(work, cx=1)

    with pytest.raises(ValueError, match="staged key set mismatch"):
        materialize_preview(BUILD, work_root=work, out_root=out)
    assert not (out / "latest.json").exists()


def test_chunk_changed_after_plan_rejects_build(tmp_path):
    work, out, _ = _plan_one(tmp_path)
    chunk = work / "builds" / BUILD / "chunks" / "height" / "0" / "0_0.lac"
    blob = bytearray(chunk.read_bytes())
    blob[-1] ^= 0xFF
    chunk.write_bytes(blob)

    with pytest.raises(ValueError, match="changed after planning"):
        materialize_preview(BUILD, work_root=work, out_root=out)


def test_existing_content_address_collision_is_rejected(tmp_path):
    work, out, plan_path = _plan_one(tmp_path)
    entry = json.loads(plan_path.read_text())["chunks"][0]
    dest = out / "c" / "height" / "0" / f"0_0.{entry['sha256'][:8]}.bin"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"not-the-planned-content")

    with pytest.raises(ValueError, match="content-address collision"):
        materialize_preview(BUILD, work_root=work, out_root=out)


def test_tampered_verified_preview_cannot_publish(tmp_path):
    work, out, _ = _plan_one(tmp_path)
    preview = materialize_preview(BUILD, work_root=work, out_root=out)
    preview.write_bytes(preview.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        publish_build(BUILD, work_root=work, out_root=out)
    assert not (out / "latest.json").exists()


def test_plan_is_immutable(tmp_path):
    work, _, plan = _plan_one(tmp_path)
    original = plan.read_bytes()
    plan.write_bytes(b"{}\n")

    with pytest.raises(ValueError, match="immutable file already exists"):
        create_build_plan(load_base(), BUILD, 7, work_root=work)
    plan.write_bytes(original)


def test_bad_header_origin_is_rejected_during_plan(tmp_path):
    work = tmp_path / "work"
    path = _stage_chunk(work)
    meta = ChunkMeta("height", 0, 1, 0, 0, 17, 0, 123.0, 456.0, 0.0, 0.01)
    write_chunk(path, meta, b"payload")

    with pytest.raises(ValueError, match="header origin mismatch"):
        create_build_plan(load_base(), BUILD, 1, work_root=work)


def test_pinned_base_is_inherited_and_overlay_replaces_exact_key(tmp_path):
    work = tmp_path / "work"
    out = tmp_path / "out"
    base_build = "b" * 64
    overlay_build = "c" * 64
    _stage_chunk(work, base_build, payload=b"base-height")
    create_build_plan(load_base(), base_build, 1, work_root=work)
    materialize_preview(base_build, work_root=work, out_root=out)
    base_manifest = publish_build(base_build, work_root=work, out_root=out)
    base_sha = hashlib.sha256(base_manifest.read_bytes()).hexdigest()

    _stage_chunk(work, overlay_build, payload=b"replacement-height")
    _stage_chunk(work, overlay_build, cx=1, payload=b"new-height")
    create_build_plan(
        load_base(), overlay_build, 2, work_root=work,
        base_manifest_path=base_manifest, base_manifest_sha256=base_sha, base_out_root=out,
    )
    preview = materialize_preview(overlay_build, work_root=work, out_root=out)
    audit = audit_base_release(preview, hashlib.sha256(preview.read_bytes()).hexdigest(), out)

    assert audit.chunk_count == 2
    manifest = json.loads(preview.read_text())
    assert manifest["layers"]["height"]["count"] == 2
    plan = json.loads((work / "builds" / overlay_build / "plan.json").read_text())
    assert len(plan["baseRelease"]["chunks"]) == 1


def test_format2_plan_is_complete_but_publication_requires_real_verifier(tmp_path, monkeypatch):
    work = tmp_path / "work"
    out = tmp_path / "out"
    base_build = "1" * 64
    overlay_build = "2" * 64
    for lod in range(5):
        _stage_chunk(work, base_build, lod=lod, payload=f"base-lod-{lod}".encode())
    create_build_plan(load_base(), base_build, 1, work_root=work)
    materialize_preview(base_build, work_root=work, out_root=out)
    base_manifest = publish_build(base_build, work_root=work, out_root=out)
    base_sha = hashlib.sha256(base_manifest.read_bytes()).hexdigest()
    monkeypatch.setattr("assetgen.release.MICRO_V1_BASE_SHA256", base_sha)
    monkeypatch.setattr(
        "assetgen.release.derive_micro_fixture_recipe",
        lambda *_args, **_kwargs: (overlay_build, {"test": True}),
    )

    hero = plan_hero(0, 0)
    expectation = create_micro_expectation(
        load_base(), overlay_build, 0, 0, base_manifest, base_sha,
        work_root=work, base_out_root=out,
    )
    frozen_expectation = json.loads(expectation.read_text())
    assert len(frozen_expectation["expectedPublished"]) == 17
    assert len(frozen_expectation["transientSupport"]) == 9
    for chunk in hero.published_fine:
        _stage_chunk(
            work, overlay_build, lod=chunk.lod, cx=chunk.cx, cz=chunk.cz,
            payload=f"micro-{chunk.cx}-{chunk.cz}".encode(), container_version=2,
        )
    _stage_chunk(
        work, overlay_build, lod=-1, payload=b"micro-parent", container_version=2,
    )
    plan_path = create_build_plan(
        load_base(),
        overlay_build,
        2,
        work_root=work,
        base_manifest_path=base_manifest,
        base_manifest_sha256=base_sha,
        base_out_root=out,
        manifest_format=2,
        micro_parent=(0, 0),
    )
    plan = json.loads(plan_path.read_text())
    assert all(entry["containerVersion"] == 2 for entry in plan["chunks"])
    assert plan["baseRelease"]["chunks"][0]["containerVersion"] == 1
    assert plan["microRecipeKind"] == "retention-fixture"

    with pytest.raises(ValueError, match="independent micro verification failed"):
        materialize_preview(overlay_build, work_root=work, out_root=out)
    assert not (work / "builds" / overlay_build / "COMPLETE").exists()
    assert json.loads((out / "latest.json").read_text())["manifest"] == base_manifest.relative_to(out).as_posix()


def test_format1_rejects_lac2_chunk(tmp_path):
    work = tmp_path / "work"
    _stage_chunk(work, lod=-1, container_version=2)

    with pytest.raises(ValueError, match="format 1 may contain only LAC1"):
        create_build_plan(load_base(), BUILD, 1, work_root=work)


def test_tampered_base_snapshot_rejects_build(tmp_path):
    work = tmp_path / "work"
    out = tmp_path / "out"
    base_build = "d" * 64
    overlay_build = "e" * 64
    _stage_chunk(work, base_build)
    create_build_plan(load_base(), base_build, 1, work_root=work)
    materialize_preview(base_build, work_root=work, out_root=out)
    base_manifest = publish_build(base_build, work_root=work, out_root=out)
    base_sha = hashlib.sha256(base_manifest.read_bytes()).hexdigest()

    _stage_chunk(work, overlay_build, cx=1)
    create_build_plan(
        load_base(), overlay_build, 2, work_root=work,
        base_manifest_path=base_manifest, base_manifest_sha256=base_sha, base_out_root=out,
    )
    snapshot = work / "builds" / overlay_build / "inputs" / "base" / "index" / "height.bin"
    snapshot.write_bytes(snapshot.read_bytes() + b"x")

    with pytest.raises(ValueError, match="base index changed"):
        materialize_preview(overlay_build, work_root=work, out_root=out)


def test_legacy_manifest_command_is_disabled():
    result = CliRunner().invoke(main, ["manifest"])
    assert result.exit_code != 0
    assert "legacy manifest publication is disabled" in result.output


def test_measured_synthesis_pilot_cannot_publish_or_touch_latest(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (out / "latest.json").write_text('{"manifest":"old"}\n')
    monkeypatch.setattr(
        "assetgen.release._load_plan",
        lambda *_args, **_kwargs: (
            tmp_path,
            {"microRecipeKind": "measured-synthesis-pilot"},
            b"plan",
        ),
    )

    with pytest.raises(ValueError, match="immutable-preview-only"):
        publish_build(BUILD, work_root=tmp_path / "work", out_root=out)
    assert json.loads((out / "latest.json").read_text()) == {"manifest": "old"}


def test_production_pilot_cli_commands_are_wired():
    runner = CliRunner()
    assert runner.invoke(main, ["micro-exemplars-prepare", "--help"]).exit_code == 0
    assert runner.invoke(main, ["micro-synthesis-cook", "--help"]).exit_code == 0
