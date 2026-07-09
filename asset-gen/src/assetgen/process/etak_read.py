"""Windowed vector reads from the ETAK GeoPackage (and other OGR sources) as shapely geoms."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyogrio
import shapely

from ..config import DATA_IN


def etak_gpkg() -> Path:
    gpkgs = sorted((DATA_IN / "etak").glob("*.gpkg"))
    if not gpkgs:
        raise FileNotFoundError("ETAK GeoPackage missing — run `assetgen fetch --only etak`")
    return gpkgs[0]


def read_layer_window(
    source: Path,
    layer: str,
    bbox_en: tuple[float, float, float, float],
    fields: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(geometries, columns) intersecting bbox. Geometries are shapely objects."""
    meta, _, geom_wkb, fdata = pyogrio.raw.read(
        source, layer=layer, bbox=tuple(float(v) for v in bbox_en), columns=fields
    )
    geoms = shapely.from_wkb(geom_wkb) if len(geom_wkb) else np.empty(0, dtype=object)
    cols = dict(zip(meta["fields"], fdata))
    return geoms, cols
