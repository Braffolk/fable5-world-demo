"""Tree-instance layer cook: nDSM+stands -> per-chunk SoA records (enc 3)."""
from __future__ import annotations

import numpy as np

from ..config import BaseConfig
from ..grid import chunk_bounds_en, chunks_covering_bbox_en
from ..process import trees as T
from .chunkio import ChunkMeta, write_chunk
from .encode import decode_records, encode_records
from .height_cook import chunk_path

# column dtypes, in wire order (must match manifest LAYER_DOC["trees"]["columns"])
TREE_DTYPES = ["u2", "u2", "u1", "u1", "u1"]


def cook_trees(base: BaseConfig, bbox_en, log=print) -> None:
    smap = T.load_species_map()
    ndsm = T.ndsm_stack()
    log("  loading Metsaregister stands...")
    stands = T._load_stands(bbox_en)
    log(f"  {len(stands[0])} stands loaded")

    chunks = chunks_covering_bbox_en(base.grid, bbox_en, 0)
    total_trees = 0
    ha_per_chunk = (base.grid.chunk_m / 100.0) ** 2
    for i, c in enumerate(chunks):
        dest = chunk_path("trees", c)
        if dest.exists():
            continue
        cols = T.derive_trees_for_chunk(
            chunk_bounds_en(base.grid, c), ndsm, stands, smap, base.grid.chunk_m
        )
        columns = [cols.x, cols.z, cols.species, cols.scale, cols.variant]
        payload = encode_records(base.encode, columns)
        # round-trip gate
        back = decode_records(base.encode, payload, len(cols), TREE_DTYPES)
        assert all(np.array_equal(a, b) for a, b in zip(columns, back))
        b = chunk_bounds_en(base.grid, c)
        meta = ChunkMeta(
            layer="trees", lod=0, enc=3, cx=c.cx, cz=c.cz, res=0, count=len(cols),
            origin_e=b[0], origin_n=b[3], qoffset=0.0, qscale=base.grid.chunk_m / 65535.0,
        )
        write_chunk(dest, meta, payload)
        total_trees += len(cols)
        if (i + 1) % 8 == 0 or i + 1 == len(chunks):
            log(f"  trees [{i + 1}/{len(chunks)}] {total_trees} trees "
                f"({total_trees / ((i + 1) * ha_per_chunk):.0f}/ha avg)")

    if T.unmapped_species:
        log(f"  trees: unmapped species codes: {dict(sorted(T.unmapped_species.items(), key=lambda kv: -kv[1]))}")
    forest_density = total_trees / (len(chunks) * ha_per_chunk)
    log(f"  trees: {total_trees} total, {forest_density:.0f}/ha over whole AOI")
