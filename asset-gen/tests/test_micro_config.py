import tomllib
from pathlib import Path

import pytest

from assetgen.config import CONFIG_DIR, load_aoi, load_base
from assetgen.height_geom import HeightChunkId, children_of
from assetgen.micro_config import load_micro_config


def test_micro_recipe_is_frozen_to_signed_128m_geometry():
    base = load_base()
    cfg = load_micro_config(base)

    assert cfg.fine_lod == -2
    assert cfg.parent_lod == -1
    assert cfg.fine_texel_m == 0.0625
    assert cfg.qscale_candidates_m == (0.001, 0.002, 0.005, 0.01)
    assert cfg.allow_latest is False


def test_hero_aoi_is_exactly_one_parent_and_sixteen_fine_chunks():
    aoi = load_aoi("micro-hero-taevaskoja")
    assert aoi.bbox_en() == (679424, 6444544, 679936, 6445056)
    children = children_of(HeightChunkId(-1, 607, 372))
    assert len(children) == 16
    assert (children[0].cx, children[0].cz) == (2428, 1488)
    assert (children[-1].cx, children[-1].cz) == (2431, 1491)


def test_micro_config_rejects_geometry_drift(tmp_path):
    source = CONFIG_DIR / "microtopography.toml"
    raw = tomllib.loads(source.read_text())
    text = source.read_text().replace("fine_texel_m = 0.0625", "fine_texel_m = 0.125")
    path = tmp_path / "micro.toml"
    path.write_text(text)

    with pytest.raises(ValueError, match="fine texel"):
        load_micro_config(load_base(), path)
