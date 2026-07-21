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
    "height": {"enc": 1, "semantics": "u16 heights, meters EH2000; texel(i,j) center at origin+(i+0.5)*t. "
                "chunk header flags bit 0 (=1) means the submerged bed is carved under the water mask "
                "(#104): wet texels store height − depth so streamed water gets real depth, not z-fight"},
    "biome": {"enc": 2, "texelMeters": 2, "planes": ["classId", "vegDensity"],
               "semantics": "land-cover class (see config/landcover-classes.toml palette) + TOTAL "
               "vegetation cover (#106): max(canopy cover, herbaceous ground cover) so open "
               "grassland/meadow/field/yard/fen read as vegetated, not bare soil (pilot = max(CHM "
               ">=2 m canopy fraction, per-class herb cover); far floor = max(ETAK forest/shrub "
               "canopy fraction, per-class herb cover)); texel = texelMeters * lodStep^lod m. "
               "LOD >= 1 reduces the finer rung per 4x4 block: classId = MAJORITY (ties -> higher "
               "classId, so water/sea/wetland win mixed coast texels), vegDensity = mean; texels "
               "beyond cooked coverage count as none(0)"},
    "water": {"enc": 1, "texelMeters": 2,
               "semantics": "waterY surface elevation on wet texels; quantized value 0 = DRY - "
               "client substitutes (own bed height - 2.0 m); absent chunk = all dry; "
               "texel = texelMeters * lodStep^lod m. LOD1 texel is wet iff >= 8 of its 16 finer "
               "texels are wet (ties lean wet), level = mean of the wet levels"},
    "watercover": {"enc": 2, "texelMeters": 2, "planes": ["coverage"],
                    "semantics": "anti-aliased water-area fraction * 255 (#114 smooth shoreline). "
                    "8x-supersampled rasterization of the SAME ETAK polygons as the water layer "
                    "(rivers E_203_a + lakes E_202 + sea E_201), box-downsampled to the texel. "
                    "255 = fully wet, 0 = fully dry, mid = sub-texel shore coverage; sample it "
                    "bilinearly and threshold/feather the water edge instead of the binary water "
                    "mask. texel = texelMeters * lodStep^lod m; LOD1 = mean of the 4x4 finer "
                    "coverage. absent chunk = no water. The submerged bed (#104) is carved into "
                    "the height layer under this same extent (height chunk flags bit 0 = carved)"},
    "canopy": {"enc": 2, "texelMeters": 2, "planes": ["heightM", "cover"],
                "semantics": "far-forest canopy from the summer CHM; LODs 1-4 ONLY (no LOD0 - near "
                "canopy derives from tree records); texel = texelMeters * lodStep^lod m. heightM = mean "
                "canopy height in meters over canopy area (CHM >= 2 m), u8 clamp 0-255, 0 where cover 0; "
                "cover = canopy-cover fraction * 255. Coarser rungs: heightM cover-weighted mean, cover "
                "plain mean. cover 0 = treeless OR unmeasured; absent chunk = no canopy data"},
    "soil": {"enc": 2, "texelMeters": 2,
              "planes": ["soilType", "texCore", "texSkeleton", "stoniness", "boniteet"],
              "semantics": "full Mullastikukaart taxonomy (config/soil-types.toml + soil-texture.toml); 0 = no data, 255 = unparseable"},
    "geology": {
        "enc": 2,
        "texelMeters": 2,
        "planes": ["bedrockFamily", "surficialFamily", "processFamily", "coverageFlags"],
        "semantics": (
            "optional categorical EGT geology; nearest sampling only. bedrockFamily: 0 unknown, "
            "1 sandstone, 2 carbonate, 3 other. surficialFamily: 0 unknown/not-applicable, "
            "1 unconsolidated sand (lito200=40), 2 till/moraine (50), 3 gravel/outwash (30), "
            "4 peat (90), 5 other (10/20/60/70/80/100/30000). processFamily: 0 unknown, "
            "1 fluvial (genees200=10), 2 lacustrine (20), 3 glaciofluvial (30), "
            "4 glaciolacustrine (40), 5 glacial (50), 6 marine (60), 7 colluvial (70), "
            "8 peat-forming (80), 9 anthropogenic (90), 10 aeolian (100), 11 water (112), "
            "12 mapped bedrock exposure (lito200=20000). coverageFlags: bit0 authoritative "
            "polygon coverage, bit1 bedrock family known, bit2 surficial class known, bit3 process "
            "known, bit4 mapped bedrock exposure, bit5 1:50k source. Current national cook uses "
            "retained 1:200k EGT surficial and bedrock polygons, so bit5 is clear. Bedrock is a "
            "subsurface formation prior unless bit4 is set; coarse polygons never place an exact cliff."
        ),
    },
    "groundcover": {
        "enc": 2,
        "texelMeters": 2,
        "planes": [
            "typeA", "typeB", "clumpLo", "clumpHi",
            "blend", "vigor", "moisture", "canopyProximity",
        ],
        "semantics": (
            "ground-cover control field v1; one logical authority stored as two rgba8 "
            "runtime carriers. typeA/typeB are categorical GroundCoverId values; blend is "
            "the typeB fraction; vigor is continuous cover/height potential; clumpLo/Hi "
            "form a stable cook-side patch id; moisture and canopyProximity are continuous. "
            "The v1 ecology is derived from the already-cooked understory community, soil "
            "wetness, and CHM evidence. It establishes the generic two-type contract; it is "
            "not yet the final >=10 native-species facies cook. LOD0 only, 2 m texels."
        ),
    },
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
                    "semantics": "ground-flora SCATTER GUIDANCE (see understoryMap), NOT a 1:1 stamp. "
                    "density already cut by slope/soil-wetness/texture/stoniness/fertility suitability "
                    "and polygon edges domain-warped + noise-broken (seamless across chunks). Client "
                    "scatters the community palette at density/255 * base_density plants/m^2, MUST jitter "
                    "per plant and SHOULD blend palettes across the class neighborhood (fuzzy ecotones)"},
    "debris": {"enc": 2, "texelMeters": 2, "planes": ["surfaceClass", "density"],
                "semantics": "debris/litter SCATTER GUIDANCE (see debrisMap), NOT a 1:1 stamp. Same "
                "suitability + edge-softening as understory (litter thins on slopes, stone/scree exposed "
                "on steep/thin/stony ground). Client scatters stones/deadwood/litter, jittered + blended"},
    "boulders": {"enc": 3, "columns": [["x", "u16"], ["z", "u16"], ["kind", "u8"], ["size", "u8"],
                  ["variant", "u8"]],
                  "semantics": "real ETAK-mapped boulders; kind 0=single 1=pile (ETAK tyyp 10/20); "
                  "size = height_m * 40, so meters = size/40 (clamp 0.2..6.4 m) — from ETAK korgus "
                  "where surveyed, else default 1 m single / 2 m pile (korgus is 0 for ~97% of "
                  "features, so size 40/80 usually means 'default'); NO y (client grounds); "
                  "absent chunk = none"},
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
