"""Post-cook verification: coverage, decode round-trip vs sources, chunk-edge seams."""
from __future__ import annotations

import random

import numpy as np

from .config import DATA_IN, DATA_WORK, BaseConfig
from .cook.chunkio import read_chunk
from .cook.encode import decode_quant16
from .grid import ChunkId, chunk_raster_window_en, chunks_covering_bbox_en
from .process.mosaic import RasterStack, dem_sources


def _load_height(base: BaseConfig, c: ChunkId) -> np.ndarray | None:
    p = DATA_WORK / "chunks" / "height" / str(c.lod) / f"{c.cx}_{c.cz}.lac"
    if not p.exists():
        return None
    meta, payload = read_chunk(p)  # crc validated inside
    return decode_quant16(base.encode, payload, meta.res, meta.qoffset, meta.qscale)


def verify_height(base: BaseConfig, bbox_en, sample_source_checks: int = 8, log=print) -> bool:
    ok = True
    rng = random.Random(1)
    all_ids: list[ChunkId] = []
    for lod in base.grid.lods:
        ids = chunks_covering_bbox_en(base.grid, bbox_en, lod)
        missing = [
            c for c in ids
            if not (DATA_WORK / "chunks" / "height" / str(c.lod) / f"{c.cx}_{c.cz}.lac").exists()
        ]
        if missing:
            ok = False
            log(f"FAIL coverage lod{lod}: {len(missing)}/{len(ids)} chunks missing, e.g. {missing[0]}")
        else:
            log(f"ok coverage lod{lod}: {len(ids)} chunks present")
        all_ids += ids

    # decode every chunk (crc + structural), seam-check east neighbors at LOD0
    lod0 = [c for c in all_ids if c.lod == 0]
    seam_worst = 0.0
    for c in lod0:
        arr = _load_height(base, c)
        east = _load_height(base, ChunkId(c.cx + 1, c.cz, 0))
        if arr is None or east is None:
            continue
        # c's apron column (i == chunk_res) duplicates east's first column
        d = float(np.max(np.abs(arr[:, -1] - east[: arr.shape[0], 0])))
        seam_worst = max(seam_worst, d)
    seam_tol = base.encode.height_qscale + 1e-4  # two half-steps from independent qoffsets
    if seam_worst > seam_tol:
        ok = False
        log(f"FAIL seams: worst apron mismatch {seam_worst:.4f} m > {seam_tol}")
    else:
        log(f"ok seams: worst apron mismatch {seam_worst:.4f} m across {len(lod0)} LOD0 chunks")

    # spot-check decoded chunks against a fresh source-mosaic read
    stack = RasterStack(dem_sources(DATA_IN))
    for c in rng.sample(lod0, min(sample_source_checks, len(lod0))):
        arr = _load_height(base, c)
        e0, n0, e1, n1, t = chunk_raster_window_en(base.grid, c)
        src = np.nan_to_num(stack.read_window(e0, n0, e1, n1, t), nan=0.0)
        err = float(np.max(np.abs(arr - src)))
        if err > base.encode.height_qscale * 0.5 + 2e-3:
            ok = False
            log(f"FAIL source check {c}: max err {err:.4f} m")
        else:
            log(f"ok source check {c}: max err {err:.4f} m")
    return ok
