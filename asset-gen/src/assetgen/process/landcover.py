"""Land-cover chunk rasterization: ETAK kõlvikud -> classId u8, CHM -> vegDensity u8.

Rasters are 2 m/texel at LOD0 footprint: (chunk_m/2 + 1)^2 = 1025^2 incl. the shared apron.
Class mapping comes from config/landcover-classes.toml and FAILS on unmapped tyyp values.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio.features
import rasterio.transform

from ..config import CONFIG_DIR, DATA_IN
from .etak_read import etak_gpkg, read_layer_window
from .mosaic import RasterStack

BIOME_TEXEL = 2.0


@dataclass(frozen=True)
class LandcoverRules:
    palette: dict[str, int]
    # paint order preserved: (layer_name, tyyp->classId or None, default classId or None)
    layers: list[tuple[str, dict[str, int] | None, int | None]]


def load_rules() -> LandcoverRules:
    raw = tomllib.loads((CONFIG_DIR / "landcover-classes.toml").read_text())
    palette = dict(raw["palette"])
    layers = []
    for spec in raw["layer"]:
        tyyp_map = (
            {str(k): palette[v] for k, v in spec["tyyp"].items()} if "tyyp" in spec else None
        )
        default = palette[spec["default"]] if "default" in spec else None
        layers.append((spec["name"], tyyp_map, default))
    return LandcoverRules(palette, layers)


def _window_geometry(window_en: tuple[float, float, float, float, float]) -> tuple:
    e_min, n_min, e_max, n_max, t = window_en
    cols = round((e_max - e_min) / t)
    rows = round((n_max - n_min) / t)
    transform = rasterio.transform.from_origin(e_min, n_max, t, t)
    return rows, cols, transform


def rasterize_classes(
    rules: LandcoverRules, window_en: tuple[float, float, float, float, float]
) -> np.ndarray:
    """Paint kõlvik classes into a u8 raster; later layer blocks overwrite earlier ones."""
    rows, cols, transform = _window_geometry(window_en)
    out = np.zeros((rows, cols), dtype=np.uint8)
    gpkg = etak_gpkg()
    bbox = (window_en[0], window_en[1], window_en[2], window_en[3])
    for layer, tyyp_map, default in rules.layers:
        geoms, cols_data = read_layer_window(gpkg, layer, bbox, fields=["tyyp"])
        if len(geoms) == 0:
            continue
        shapes = []
        for i, geom in enumerate(geoms):
            if geom is None:
                continue
            if tyyp_map is not None:
                tyyp = str(cols_data["tyyp"][i])
                if tyyp not in tyyp_map:
                    raise KeyError(
                        f"unmapped ETAK class: layer {layer} tyyp={tyyp!r} — "
                        f"add it to config/landcover-classes.toml"
                    )
                value = tyyp_map[tyyp]
            else:
                assert default is not None
                value = default
            shapes.append((geom, value))
        if shapes:
            rasterio.features.rasterize(
                shapes, out=out, transform=transform, default_value=0
            )
    return out


def _box_mean_u8(binary: np.ndarray, radius: int) -> np.ndarray:
    """Integral-image box mean of a 0/1 array -> u8 0..255."""
    pad = np.pad(binary.astype(np.float32), radius + 1)
    ii = pad.cumsum(0).cumsum(1)
    k = 2 * radius + 1
    s = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
    mean = s[: binary.shape[0], : binary.shape[1]] / (k * k)
    return np.clip(mean * 255.0, 0, 255).astype(np.uint8)


def veg_density(window_en: tuple[float, float, float, float, float]) -> np.ndarray:
    """Canopy-cover fraction (u8) from the CHM: share of >=2 m canopy in a ~14 m window."""
    chm_paths = sorted((DATA_IN / "chm").glob("*.tif"))
    if not chm_paths:
        raise FileNotFoundError("no CHM rasters — run `assetgen fetch --only elevation`")
    stack = RasterStack(chm_paths)
    e_min, n_min, e_max, n_max, t = window_en
    chm = stack.read_window(e_min, n_min, e_max, n_max, t)
    canopy = (np.nan_to_num(chm, nan=0.0) >= 2.0).astype(np.uint8)
    return _box_mean_u8(canopy, radius=3)  # 7 texels * 2 m = 14 m window
