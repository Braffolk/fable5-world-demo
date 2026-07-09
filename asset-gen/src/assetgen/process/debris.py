"""Ground-debris field: soil stoniness + forest leaf-type + land cover -> surfaceClass +
density (2 m). Client scatters stones/litter/deadwood; we ship the field only."""
from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio.features
import rasterio.transform
import shapely

from ..config import CONFIG_DIR, DATA_IN


@dataclass(frozen=True)
class DebrisClasses:
    id_by_name: dict[str, int]
    base_density: dict[int, float]
    scree_threshold: int
    stony_threshold: int
    dictionary: dict[int, dict]


@lru_cache
def load_debris() -> DebrisClasses:
    raw = tomllib.loads((CONFIG_DIR / "debris-classes.toml").read_text())
    cls = raw["classes"]
    return DebrisClasses(
        id_by_name={n: s["id"] for n, s in cls.items()},
        base_density={s["id"]: float(s["base_density"]) for s in cls.values()},
        scree_threshold=raw["stoniness"]["scree_threshold"],
        stony_threshold=raw["stoniness"]["stony_threshold"],
        dictionary={s["id"]: {"name": s["name"], "base_density": s["base_density"],
                              "palette": s["palette"]} for s in cls.values()},
    )


def _stand_leaf_raster(window_en, rows, cols, transform) -> np.ndarray:
    """0 = no stand, 1 = conifer-dominant, 2 = broadleaf-dominant (from stand species)."""
    from .trees import load_species_map

    smap = load_species_map()
    leaf_of = {d["code"]: d["leaf"] for d in smap.dictionary.values()}
    files = sorted((DATA_IN / "wfs" / "metsaregister").glob("*eraldis.*.ndjson"))
    shapes = []
    e_min, n_min, e_max, n_max, _ = window_en
    for f in files:
        for line in f.open():
            feat = json.loads(line)
            g = feat.get("geometry")
            if not g:
                continue
            leaf = leaf_of.get(feat["properties"].get("peapuuliik_kood"))
            shapes.append((shapely.geometry.shape(g), 1 if leaf == "conifer" else 2 if leaf == "broadleaf" else 0))
    if not shapes:
        return np.zeros((rows, cols), dtype=np.uint8)
    return rasterio.features.rasterize(shapes, out_shape=(rows, cols), transform=transform).astype(np.uint8)


def rasterize_debris(
    window_en, classid_plane: np.ndarray, stoniness_plane: np.ndarray
) -> list[np.ndarray]:
    """(surfaceClass u8, density u8). classid = biome plane, stoniness = soil plane, 2 m grid."""
    d = load_debris()
    e_min, n_min, e_max, n_max, t = window_en
    rows = round((n_max - n_min) / t)
    cols = round((e_max - e_min) / t)
    transform = rasterio.transform.from_origin(e_min, n_max, t, t)

    surface = np.zeros((rows, cols), dtype=np.uint8)
    forest = np.isin(classid_plane, [1, 2])

    # litter under forest by dominant leaf type
    leaf = _stand_leaf_raster(window_en, rows, cols, transform)
    surface[forest & (leaf == 0)] = d.id_by_name["mixed_litter"]
    surface[forest & (leaf == 1)] = d.id_by_name["needle_litter"]
    surface[forest & (leaf == 2)] = d.id_by_name["leaf_litter"]

    # wetland / sandy / stony overrides from land cover + soil
    surface[np.isin(classid_plane, [8, 9, 10])] = d.id_by_name["wetland_debris"]
    surface[classid_plane == 5] = d.id_by_name["sandy"]  # sand
    surface[stoniness_plane >= d.stony_threshold] = d.id_by_name["stony"]
    surface[stoniness_plane >= d.scree_threshold] = d.id_by_name["scree"]

    base = np.array([d.base_density.get(int(c), 0.0) for c in range(256)])
    dens = base[surface] / max(base.max(), 1e-6)
    density = np.clip(dens * 255.0, 0, 255).astype(np.uint8)
    density[surface == 0] = 0
    return [surface, density]
