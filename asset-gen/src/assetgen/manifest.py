"""Build the streaming manifest + per-layer binary indexes and the S3-ready out/ tree.

Layout (data/out mirrors the S3 bucket):
    latest.json                      { "manifest": "m/<hash16>/manifest.json" }
    m/<hash16>/manifest.json         immutable, self-hash-named
    m/<hash16>/index/<layer>.bin     sorted records: <B lod><i cx><i cz><I size><Q hash64>
    c/<layer>/<lod>/<cx>_<cz>.<hash8>.bin   immutable content-hash-named LAC1 chunks

Chunk absence in an index is authoritative (e.g. all-dry water chunks are simply omitted).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
from collections import defaultdict
from pathlib import Path

from .config import DATA_OUT, DATA_WORK, BaseConfig

MANIFEST_FORMAT = 1
INDEX_RECORD = struct.Struct("<BiiIQ")


def _species_dictionary() -> dict[int, dict]:
    from .process.trees import load_species_map

    return load_species_map().dictionary


def _understory_dictionary() -> dict[int, dict]:
    from .process.understory import load_communities

    return load_communities().dictionary


def _debris_dictionary() -> dict[int, dict]:
    from .process.debris import load_debris

    return load_debris().dictionary

LAYER_DOC = {
    "height": {"enc": 1, "semantics": "u16 heights, meters EH2000; texel(i,j) center at origin+(i+0.5)*t"},
    "biome": {"enc": 2, "texelMeters": 2, "planes": ["classId", "vegDensity"],
               "semantics": "land-cover class (see config/landcover-classes.toml palette) + canopy fraction"},
    "water": {"enc": 1, "texelMeters": 2,
               "semantics": "waterY surface elevation on wet texels; quantized value 0 = DRY - "
               "client substitutes (own bed height - 2.0 m); absent chunk = all dry"},
    "soil": {"enc": 2, "texelMeters": 2,
              "planes": ["soilType", "texCore", "texSkeleton", "stoniness", "boniteet"],
              "semantics": "full Mullastikukaart taxonomy (config/soil-types.toml + soil-texture.toml); 0 = no data, 255 = unparseable"},
    "trees": {
        "enc": 3,
        "columns": [["x", "u16"], ["z", "u16"], ["species", "u8"], ["scale", "u8"],
                     ["variant", "u8"]],
        "semantics": "chunk-local SoA, Morton-sorted; 7 B/tree. x,z = position "
        "(step chunkMeters/65535); species = id into speciesMap; scale = crownHeight/refHeight*64; "
        "variant = art seed. NO y — client grounds tree on rendered terrain; NO yaw/lean — "
        "client hashes (cx,cz,x,z) for cosmetic orientation.",
    },
    "understory": {"enc": 2, "texelMeters": 2, "planes": ["communityId", "density"],
                    "semantics": "ground-flora SCATTER FIELD (see understoryMap): client scatters "
                    "the community's plant palette at density/255 * base_density plants/m^2"},
    "debris": {"enc": 2, "texelMeters": 2, "planes": ["surfaceClass", "density"],
                "semantics": "debris/litter SCATTER FIELD (see debrisMap): client scatters "
                "stones/deadwood/litter from the class palette at density/255 * base_density"},
    "boulders": {"enc": 3, "columns": [["x", "u16"], ["z", "u16"], ["kind", "u8"], ["size", "u8"],
                  ["variant", "u8"]],
                  "semantics": "real ETAK-mapped boulders; kind 0=single 1=pile; size ~cm/40; "
                  "NO y (client grounds); absent chunk = none"},
}


def _hash_file(path: Path) -> bytes:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.digest()


def build_release(base: BaseConfig, cook_rev: int, log=print) -> Path:
    """Collect every cooked chunk in data/work/chunks into a content-addressed release."""
    chunks_root = DATA_WORK / "chunks"
    out_c = DATA_OUT / "c"
    layers: dict[str, dict] = {}
    index_blobs: dict[str, bytes] = {}

    for layer_dir in sorted(p for p in chunks_root.iterdir() if p.is_dir()):
        layer = layer_dir.name
        records = []
        total_bytes = 0
        lods = set()
        for lac in sorted(layer_dir.glob("*/*.lac")):
            lod = int(lac.parent.name)
            cx, cz = (int(v) for v in lac.stem.split("_"))
            digest = _hash_file(lac)
            h8 = digest[:4].hex()
            dest = out_c / layer / str(lod) / f"{cx}_{cz}.{h8}.bin"
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(lac, dest)
            size = lac.stat().st_size
            records.append((lod, cx, cz, size, int.from_bytes(digest[:8], "big")))
            total_bytes += size
            lods.add(lod)
        if not records:
            continue
        records.sort()
        index_blobs[layer] = b"".join(INDEX_RECORD.pack(*r) for r in records)
        layers[layer] = {
            **LAYER_DOC.get(layer, {}),
            "lods": sorted(lods),
            "count": len(records),
            "bytes": total_bytes,
            "index": f"index/{layer}.bin",
        }
        log(f"  {layer}: {len(records)} chunks, {total_bytes / 1e6:.1f} MB")

    g = base.grid
    manifest = {
        "format": MANIFEST_FORMAT,
        "cookRev": cook_rev,
        "crs": "EPSG:3301",
        "attribution": base.attribution,
        "anchor": {"e": g.anchor_e, "n": g.anchor_n},
        "axes": {
            "x": "east: gameX = E - anchor.e",
            "z": "south: gameZ = anchor.n - N (raster row order == +z)",
            "y": "EH2000 meters (sea = 0)",
        },
        "chunkMeters": g.chunk_m,
        "chunkRes": g.chunk_res,
        "lodStep": g.lod_step,
        "texelConvention": "texel (i,j) of chunk (cx,cz,lod) centers at "
        "(anchor.e + cx*F + (i+0.5)*t, anchor.n - cz*F - (j+0.5)*t), t = lodStep^lod, "
        "F = chunkMeters*t; far row/col (i or j == chunkRes) duplicates the east/south neighbor",
        "codec": base.encode.codec,
        "container": "LAC1 v1 (56-byte LE header + compressed payload; see chunkio.py)",
        "speciesMap": _species_dictionary(),
        "understoryMap": _understory_dictionary(),
        "debrisMap": _debris_dictionary(),
        "layers": layers,
    }
    blob = json.dumps(manifest, indent=1, sort_keys=True).encode()
    mhash = hashlib.sha256(blob).hexdigest()[:16]
    mdir = DATA_OUT / "m" / mhash
    (mdir / "index").mkdir(parents=True, exist_ok=True)
    (mdir / "manifest.json").write_bytes(blob)
    for layer, idx in index_blobs.items():
        (mdir / "index" / f"{layer}.bin").write_bytes(idx)
    (DATA_OUT / "latest.json").write_text(
        json.dumps({"manifest": f"m/{mhash}/manifest.json"}) + "\n"
    )
    log(f"release m/{mhash}: {sum(v['bytes'] for v in layers.values()) / 1e6:.1f} MB across {len(layers)} layers")
    return mdir / "manifest.json"
