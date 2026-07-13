import json
from pathlib import Path

from assetgen.micro_recipe import (
    _SYNTHESIS_SOURCE_PATHS,
    derive_micro_fixture_recipe,
    derive_micro_synthesis_recipe,
)


def _fixture(root: Path) -> Path:
    (root / "config").mkdir(parents=True)
    (root / "config" / "base.toml").write_text("base")
    (root / "config" / "microtopography.toml").write_text("micro")
    for relative in (
        "src/assetgen/cook/chunkio.py",
        "src/assetgen/cook/encode.py",
        "src/assetgen/cook/micro_fixture_cook.py",
        "src/assetgen/cook/micro_hierarchy.py",
        "src/assetgen/cook/pinned_height.py",
        "src/assetgen/height_geom.py",
        "src/assetgen/micro_config.py",
        "src/assetgen/process/micro_masks.py",
        "src/assetgen/process/micro_fixture.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    release = root / "release"
    (release / "index").mkdir(parents=True)
    (release / "index" / "height.bin").write_bytes(b"height-index")
    manifest = release / "manifest.json"
    manifest.write_text(json.dumps({"layers": {"height": {"index": "index/height.bin"}}}))
    return manifest


def test_recipe_binds_parent_source_config_manifest_and_indexes(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    first, inputs = derive_micro_fixture_recipe(2, 3, manifest, asset_gen_root=tmp_path)
    assert first == derive_micro_fixture_recipe(2, 3, manifest, asset_gen_root=tmp_path)[0]
    assert inputs["parent"] == [-1, 2, 3]
    (tmp_path / "release" / "index" / "height.bin").write_bytes(b"changed")
    assert first != derive_micro_fixture_recipe(2, 3, manifest, asset_gen_root=tmp_path)[0]
    assert first != derive_micro_fixture_recipe(3, 3, manifest, asset_gen_root=tmp_path)[0]


def test_synthesis_recipe_binds_locked_numerical_environment(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    for relative in _SYNTHESIS_SOURCE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    for name in (
        "landcover-classes.toml",
        "microtopography-exemplars.json",
        "soil-texture.toml",
        "soil-types.toml",
    ):
        (tmp_path / "config" / name).write_text(name)
    (tmp_path / "pyproject.toml").write_text("project")
    (tmp_path / "uv.lock").write_text("locked")
    etak = tmp_path / "data" / "in" / "etak" / "source.gpkg"
    etak.parent.mkdir(parents=True)
    etak.write_bytes(b"etak")
    soil = tmp_path / "data" / "in" / "soil" / "mullakaart"
    soil.mkdir(parents=True)
    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        (soil / f"Mullakaart{suffix}").write_bytes(suffix.encode())
    exemplar_root = tmp_path / "exemplars"
    exemplar_root.mkdir()
    bank = exemplar_root / "bank.npz"
    bank.write_bytes(b"bank")
    import hashlib

    exemplar = exemplar_root / "manifest.json"
    exemplar.write_text(json.dumps({
        "bankFile": bank.name,
        "bankSha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
    }))

    first, inputs = derive_micro_synthesis_recipe(
        2, 3, manifest, exemplar, asset_gen_root=tmp_path
    )
    assert inputs["environment"]["runtime"]["numpyVersion"]
    assert inputs["environment"]["runtime"]["scipyVersion"]
    (tmp_path / "uv.lock").write_text("changed")
    assert first != derive_micro_synthesis_recipe(
        2, 3, manifest, exemplar, asset_gen_root=tmp_path
    )[0]
