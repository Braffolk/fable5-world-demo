"""Priority-ordered raster stack -> chunk-aligned window reads.

Sources are tried in order (finest first); texels still NaN after a source are filled by
the next one. LOD0 reads from the 1 m sheets are exact pixel copies (nearest resampling on
an integer-aligned window); coarser LODs average-downsample. Everything stays in EPSG:3301.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds


@dataclass
class _Source:
    path: Path
    bounds: tuple[float, float, float, float]  # (left, bottom, right, top)
    res: float
    nodata: float | None


class RasterStack:
    def __init__(self, source_paths: list[Path]):
        self.sources: list[_Source] = []
        for p in source_paths:
            with rasterio.open(p) as ds:
                b = ds.bounds
                self.sources.append(
                    _Source(p, (b.left, b.bottom, b.right, b.top), float(ds.res[0]), ds.nodata)
                )

    def read_window(
        self, e_min: float, n_min: float, e_max: float, n_max: float, texel: float
    ) -> np.ndarray:
        """Read (rows, cols) float32 with NaN where no source has data.

        Row 0 = northern edge (row order == +gameZ). Texel (i, j) centers at
        (e_min + (i+0.5)*texel, n_max - (j+0.5)*texel).
        """
        cols = round((e_max - e_min) / texel)
        rows = round((n_max - n_min) / texel)
        out = np.full((rows, cols), np.nan, dtype=np.float32)
        for src in self.sources:
            if (
                src.bounds[2] <= e_min or src.bounds[0] >= e_max
                or src.bounds[3] <= n_min or src.bounds[1] >= n_max
            ):
                continue
            if not np.isnan(out).any():
                break
            with rasterio.open(src.path) as ds:
                window = from_bounds(e_min, n_min, e_max, n_max, transform=ds.transform)
                resampling = Resampling.nearest if src.res >= texel else Resampling.average
                fill = src.nodata if src.nodata is not None else np.nan
                data = ds.read(
                    1,
                    window=window,
                    out_shape=(rows, cols),
                    resampling=resampling,
                    boundless=True,
                    fill_value=fill,
                ).astype(np.float32)
            if src.nodata is not None:
                data[data == np.float32(src.nodata)] = np.nan
            mask = np.isnan(out) & ~np.isnan(data)
            out[mask] = data[mask]
        return out


def dem_sources(data_in: Path) -> list[Path]:
    """Height source priority: 1 m DTM sheets, then the whole-country 10 m DTM."""
    fine = sorted((data_in / "dem_1m").glob("*_dtm_1m.tif"))
    country = sorted((data_in / "country").glob("DTM_*_eesti.tif"))
    return fine + country
