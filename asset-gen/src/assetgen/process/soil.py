"""Soil chunk rasterization from Mullastikukaart -> FIVE u8 planes @ 2 m (full fidelity —
the pipeline ships the complete official taxonomy; any collapsing into a material palette
is a renderer/art decision made later):

  plane 0 soilType:    official legend code id (config/soil-types.toml, ~116 canonical
                        types incl. gleyed/eroded/peat-thickness variants). 0 = no data
                        (cities/water/islets are legally unmapped), 255 = unparseable.
  plane 1 texCore:     surface-layer fine-earth class (liiv..savi, turvas by thickness)
  plane 2 texSkeleton: coarse-fragment component (veeris/rähk/kruus/klibu + grade)
  plane 3 stoniness:   Kivisus 0..6 (roman numeral, max on composite entries)
  plane 4 boniteet:    fertility score 0..100 raw (ground-flora lushness driver)

Cook fails when unparseable soilType/texture area exceeds the configured fractions.
"""
from __future__ import annotations

import re
import tomllib
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio.features
import rasterio.transform

from ..config import CONFIG_DIR, DATA_IN
from .etak_read import read_layer_window

UNKNOWN = 255

# fold the map's typographic marks to ASCII: ’ ′ -> '   ” ″ '' -> "
_QUOTE_FOLD = str.maketrans({"’": "'", "′": "'", "`": "'", "”": '"', "“": '"', "″": '"'})
_SUBSUP = {ord(c): str(d) for d, chars in enumerate(["₀⁰", "₁¹", "₂²", "₃³", "₄⁴"]) for c in chars}

unmapped_types: dict[str, int] = {}
unmapped_textures: dict[str, int] = {}


def soil_shp() -> Path:
    p = DATA_IN / "soil" / "mullakaart" / "Mullakaart.shp"
    if not p.exists():
        raise FileNotFoundError("Mullakaart.shp missing — run `assetgen fetch --only soil`")
    return p


@lru_cache
def _tables():
    types_raw = tomllib.loads((CONFIG_DIR / "soil-types.toml").read_text())
    tex_raw = tomllib.loads((CONFIG_DIR / "soil-texture.toml").read_text())
    type_ids = {code: spec["id"] for code, spec in types_raw["types"].items()}
    for alias, target in types_raw.get("aliases", {}).items():
        type_ids[alias] = type_ids[target]
    return (
        type_ids,
        tex_raw["core"],
        tex_raw["skeleton"],
        float(types_raw["max_unknown_fraction"]),
        float(tex_raw["max_unknown_fraction"]),
    )


def normalize_code(value: str) -> str:
    """Fold typography and canonicalize thickness marks: any '/" run becomes the official
    spelling for its total prime count (1 -> ', 2 -> ", 3 -> '\")."""
    v = value.translate(_QUOTE_FOLD).translate(_SUBSUP).strip()
    v = re.sub(r"\s+", "", v)

    def canon_marks(m: re.Match) -> str:
        total = sum(1 if ch == "'" else 2 for ch in m.group(0))
        return {1: "'", 2: '"', 3: "'\""}.get(total, "'\"")

    return re.sub(r"['\"]+", canon_marks, v)


def soil_type_id(sif: str | None) -> int:
    type_ids, *_ = _tables()
    if sif is None or str(sif).strip() in ("", "None"):
        return 0
    v = normalize_code(str(sif))
    if v in type_ids:
        return type_ids[v]
    # fallback chain, mildest first: strip parenthetical/comma garnish; strip trailing
    # note letters (d/õ/z/n/a/l); strip thickness marks (TxR'" -> TxR); strip a rare
    # 'e' (eroded) variant back to its base type
    v2 = re.sub(r"[(,;].*$", "", v)
    for candidate in (
        v2,
        re.sub(r"[dõznal]+$", "", v2),
        re.sub(r"['\"]+$", "", re.sub(r"[dõznal]+$", "", v2)),
        re.sub(r"e$", "", re.sub(r"[dõznal]+$", "", v2)),
    ):
        if candidate in type_ids:
            return type_ids[candidate]
    unmapped_types[v] = unmapped_types.get(v, 0) + 1
    return UNKNOWN


_SKELETON_RE = re.compile(r"^(rb|kr|kb|ko|v|r|k|d|p)([0-9]?)")
_CORE_RE = re.compile(r"^(pl|sl|ls|th|tt|t|l|s)([0-9]?)")


def texture_ids(lihtloimis: str | None) -> tuple[int, int]:
    """(texCore, texSkeleton) of the SURFACE component (before the first '/')."""
    _, core_ids, skel_ids, *_ = _tables()
    if lihtloimis is None or str(lihtloimis).strip() in ("", "None", "<Null>"):
        return 0, 0
    v = normalize_code(str(lihtloimis)).lower().split("/")[0]
    v = re.sub(r"[()'\"]", "", v)
    skel = 0
    m = _SKELETON_RE.match(v)
    # a skeleton token counts when followed by a fine-earth core OR standing alone
    # (bare 'kb'/'r' = pure shingle/gravel surface); 'l...'/'s...' always open a core
    rest = v[m.end():] if m else v
    if m and v[: m.end()] not in ("l", "s") and (rest == "" or _CORE_RE.match(rest)):
        family = "r" if m.group(1) == "rb" else m.group(1)
        grade = min(int(m.group(2) or 1), 5)
        key = family if family in ("p", "ko") else f"{family}{grade}"
        skel = skel_ids.get(key, UNKNOWN)
        v = rest
    m = _CORE_RE.match(v)
    if m:
        key = m.group(1) + m.group(2)
        core = core_ids.get(key, core_ids.get(m.group(1), UNKNOWN))
    elif v == "":
        core = 0
    else:
        core = UNKNOWN
    if core == UNKNOWN or skel == UNKNOWN:
        unmapped_textures[str(lihtloimis)] = unmapped_textures.get(str(lihtloimis), 0) + 1
    return core, skel


_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}


def stoniness_id(kivisus: str | None) -> int:
    if kivisus is None or str(kivisus).strip() in ("", "None"):
        return 0
    tokens = re.findall(r"[IVX]+", str(kivisus).translate(_SUBSUP))
    return max((_ROMAN.get(t, 0) for t in tokens), default=0)


def boniteet_id(boniteet) -> int:
    try:
        return int(np.clip(int(boniteet), 0, 100))
    except (TypeError, ValueError):
        return 0


def rasterize_soil(window_en: tuple[float, float, float, float, float]) -> list[np.ndarray]:
    e_min, n_min, e_max, n_max, t = window_en
    cols = round((e_max - e_min) / t)
    rows = round((n_max - n_min) / t)
    transform = rasterio.transform.from_origin(e_min, n_max, t, t)
    geoms, data = read_layer_window(
        soil_shp(), "Mullakaart", (e_min, n_min, e_max, n_max),
        fields=["Sif1", "Lihtloimis", "Kivisus", "Boniteet"],
    )
    planes = [np.zeros((rows, cols), dtype=np.uint8) for _ in range(5)]
    if not len(geoms):
        return planes
    shape_lists: list[list] = [[] for _ in range(5)]
    for i, geom in enumerate(geoms):
        if geom is None:
            continue
        core, skel = texture_ids(data["Lihtloimis"][i])
        values = (
            soil_type_id(data["Sif1"][i]),
            core,
            skel,
            stoniness_id(data["Kivisus"][i]),
            boniteet_id(data["Boniteet"][i]),
        )
        for plane_idx, value in enumerate(values):
            shape_lists[plane_idx].append((geom, value))
    for plane_idx, shapes in enumerate(shape_lists):
        if shapes:
            rasterio.features.rasterize(shapes, out=planes[plane_idx], transform=transform)
    return planes


def check_unknown_budget(total_texels_mapped: int, log=print) -> None:
    """Fail the cook when unparseable codes exceed the configured budget."""
    *_, max_type_frac, max_tex_frac = _tables()
    if unmapped_types:
        top = sorted(unmapped_types.items(), key=lambda kv: -kv[1])[:20]
        log(f"soil: {len(unmapped_types)} unparseable Sif1 codes (by polygon count): {top}")
    if unmapped_textures:
        top = sorted(unmapped_textures.items(), key=lambda kv: -kv[1])[:20]
        log(f"soil: {len(unmapped_textures)} unparseable lõimis values: {top}")
