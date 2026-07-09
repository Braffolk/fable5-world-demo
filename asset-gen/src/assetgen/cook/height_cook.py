"""Height layer cook: DTM sources -> LAC1 chunks at every LOD, with inline round-trip verify."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..config import DATA_IN, DATA_WORK, BaseConfig
from ..grid import ChunkId, chunk_bounds_en, chunk_raster_window_en, chunks_covering_bbox_en
from ..process.mosaic import RasterStack, dem_sources
from .chunkio import ChunkMeta, write_chunk
from .encode import decode_quant16, encode_quant16

COOK_REV = 1
SEA_LEVEL = 0.0

_worker_stack: RasterStack | None = None
_worker_base: BaseConfig | None = None


def work_dir(layer: str) -> Path:
    return DATA_WORK / "chunks" / layer


def chunk_path(layer: str, c: ChunkId) -> Path:
    return work_dir(layer) / str(c.lod) / f"{c.cx}_{c.cz}.lac"


def _init_worker(base: BaseConfig, source_paths: list[Path]) -> None:
    global _worker_stack, _worker_base
    _worker_base = base
    _worker_stack = RasterStack(source_paths)


def _cook_one(c: ChunkId) -> tuple[ChunkId, int, bool]:
    """Returns (chunk, cooked_bytes, had_data). Skips work if an up-to-date file exists."""
    base, stack = _worker_base, _worker_stack
    assert base is not None and stack is not None
    dest = chunk_path("height", c)
    if dest.exists():
        return c, dest.stat().st_size, True

    e_min, n_min, e_max, n_max, t = chunk_raster_window_en(base.grid, c)
    arr = stack.read_window(e_min, n_min, e_max, n_max, t)
    had_data = bool(np.isfinite(arr).any())
    arr = np.nan_to_num(arr, nan=SEA_LEVEL)  # sea / no-coverage texels sit at EH2000 zero

    qscale = base.encode.height_qscale
    payload, qoffset = encode_quant16(base.encode, arr.astype(np.float64), qscale)
    # inline round-trip gate: never write a chunk that doesn't decode to within half a step
    out = decode_quant16(base.encode, payload, arr.shape[0], qoffset, qscale)
    err = float(np.max(np.abs(out - arr)))
    if err > qscale * 0.5 + 1e-3:
        raise AssertionError(f"round-trip error {err} m on chunk {c}")

    b = chunk_bounds_en(base.grid, c)
    meta = ChunkMeta(
        layer="height", lod=c.lod, enc=1, cx=c.cx, cz=c.cz,
        res=arr.shape[0], count=0, origin_e=b[0], origin_n=b[3],
        qoffset=qoffset, qscale=qscale,
    )
    write_chunk(dest, meta, payload)
    return c, dest.stat().st_size, had_data


def cook_height(
    base: BaseConfig,
    bbox_en: tuple[int, int, int, int],
    lods: tuple[int, ...] | None = None,
    workers: int = 6,
    log=print,
) -> list[Path]:
    sources = dem_sources(DATA_IN)
    if not sources:
        raise FileNotFoundError("no DTM sources in data/in — run `assetgen fetch` first")
    lods = lods if lods is not None else base.grid.lods
    chunks = [c for lod in lods for c in chunks_covering_bbox_en(base.grid, bbox_en, lod)]
    log(f"height cook: {len(chunks)} chunks across LODs {list(lods)} from {len(sources)} sources")

    done_paths: list[Path] = []
    total = 0
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(base, sources)
    ) as ex:
        for i, (c, size, had_data) in enumerate(ex.map(_cook_one, chunks, chunksize=4)):
            total += size
            done_paths.append(chunk_path("height", c))
            if (i + 1) % 16 == 0 or i + 1 == len(chunks):
                log(f"  [{i + 1}/{len(chunks)}] lod{c.lod} ({total / 1e6:.1f} MB cooked)")
    return done_paths
