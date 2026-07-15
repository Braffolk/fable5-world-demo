"""Recipe-addressed, transactional release construction.

The legacy manifest command publishes whatever happens to be in data/work/chunks.
This module deliberately has no path from that shared directory: a build is planned
from, verified against, and materialized from one isolated build root.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DATA_OUT, DATA_WORK, BaseConfig, EncodeConfig
from .cook.chunkio import ChunkMeta, read_chunk, read_chunk_any
from .height_geom import HeightChunkId, children_of, parent_of, plan_hero
from .manifest import (
    LAYER_DOC,
    _debris_dictionary,
    _species_dictionary,
    _understory_dictionary,
)
from .micro_recipe import derive_micro_fixture_recipe, derive_micro_synthesis_recipe
from .micro_verify import (
    VERIFIER_ID,
    VERIFIER_CAN_AUTHORIZE_RELEASE,
    evidence_sha256,
    transient_merkle_root,
    verify_micro_fixture,
    verify_micro_synthesis,
    verifier_source_sha256,
)

PLAN_FORMAT = 1
VERIFY_FORMAT = 1
INDEX_RECORD_V1 = struct.Struct("<BiiIQ")
INDEX_RECORD_V2 = struct.Struct("<biiIQ")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
MICRO_VERIFY_GATES = (
    "coverage",
    "parentClosure",
    "headers",
    "decodedHierarchy",
    "aprons",
    "determinism",
    "hardMasks",
)
MICRO_V1_BASE_SHA256 = "708478a57c2118eaa618867e87cc74a35e20c615999b60d7e4177b595ef5495a"


@dataclass(frozen=True)
class IndexRecord:
    lod: int
    cx: int
    cz: int
    size: int
    hash64: int

    @property
    def key(self) -> tuple[int, int, int]:
        return self.lod, self.cx, self.cz


@dataclass(frozen=True)
class VerifiedBaseRelease:
    manifest_path: Path
    manifest_sha256: str
    chunk_count: int
    total_bytes: int


@dataclass(frozen=True)
class CorrectedFormat1BaseRelease:
    manifest_path: Path
    manifest_sha256: str
    verification_path: Path
    verification_sha256: str


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=1, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _validate_digest(build_digest: str) -> None:
    if not _DIGEST_RE.fullmatch(build_digest):
        raise ValueError("build digest must be exactly 64 lowercase hexadecimal characters")


def _write_fsynced(path: Path, data: bytes) -> None:
    with path.open("wb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_create_or_verify(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"immutable file already exists with different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        _write_fsynced(tmp, data)
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"concurrent immutable-file conflict: {path}")
        _fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_replace(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _write_fsynced(tmp, data)
    tmp.replace(path)
    _fsync_dir(path.parent)


def _safe_child(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe relative path: {relative!r}")
    root_resolved = root.resolve()
    result = (root / rel).resolve()
    if result != root_resolved and root_resolved not in result.parents:
        raise ValueError(f"path escapes root: {relative!r}")
    return result


def read_v1_index(path: Path) -> tuple[IndexRecord, ...]:
    blob = path.read_bytes()
    if len(blob) % INDEX_RECORD_V1.size:
        raise ValueError(f"{path}: index length is not a multiple of {INDEX_RECORD_V1.size}")
    records = tuple(IndexRecord(*INDEX_RECORD_V1.unpack_from(blob, off)) for off in range(0, len(blob), INDEX_RECORD_V1.size))
    keys = [r.key for r in records]
    if keys != sorted(keys):
        raise ValueError(f"{path}: index records are not sorted")
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate chunk key")
    return records


def read_v2_index(path: Path) -> tuple[IndexRecord, ...]:
    blob = path.read_bytes()
    if len(blob) % INDEX_RECORD_V2.size:
        raise ValueError(f"{path}: v2 index length is not a multiple of {INDEX_RECORD_V2.size}")
    records = tuple(IndexRecord(*INDEX_RECORD_V2.unpack_from(blob, off)) for off in range(0, len(blob), INDEX_RECORD_V2.size))
    keys = [r.key for r in records]
    if keys != sorted(keys):
        raise ValueError(f"{path}: v2 index records are not sorted")
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate v2 chunk key")
    return records


def _content_path(out_root: Path, layer: str, rec: IndexRecord) -> Path:
    hash8 = ((rec.hash64 >> 32) & 0xFFFFFFFF).to_bytes(4, "big").hex()
    return out_root / "c" / layer / str(rec.lod) / f"{rec.cx}_{rec.cz}.{hash8}.bin"


def _validate_header(meta: ChunkMeta, layer: str, lod: int, cx: int, cz: int, grid: dict[str, Any]) -> None:
    if (meta.layer, meta.lod, meta.cx, meta.cz) != (layer, lod, cx, cz):
        raise ValueError(
            f"header/path mismatch: header {(meta.layer, meta.lod, meta.cx, meta.cz)} "
            f"!= path {(layer, lod, cx, cz)}"
        )
    footprint = grid["chunkMeters"] * grid["lodStep"] ** lod
    expected_e = grid["anchorE"] + cx * footprint
    expected_n = grid["anchorN"] - cz * footprint
    if meta.origin_e != expected_e or meta.origin_n != expected_n:
        raise ValueError(
            f"header origin mismatch for {(layer, lod, cx, cz)}: "
            f"{(meta.origin_e, meta.origin_n)} != {(expected_e, expected_n)}"
        )


def _validate_format2_height_header(meta: ChunkMeta, container_version: int) -> None:
    if meta.layer != "height" or not -2 <= meta.lod <= 4:
        raise ValueError("format-2 v1 overlays support only height LOD -2..4")
    if meta.enc != 1 or meta.res != 2049 or meta.count != 0:
        raise ValueError("format-2 height chunks require enc=1, res=2049, count=0")
    if meta.lod < 0 and container_version != 2:
        raise ValueError("negative height LOD requires LAC2")
    expected_qscale = 0.002 if meta.lod == -2 else 0.005 if meta.lod == -1 else None
    if expected_qscale is not None and abs(meta.qscale - expected_qscale) > 1e-8:
        raise ValueError(
            f"height LOD {meta.lod} qscale {meta.qscale} does not match {expected_qscale}"
        )


def _validate_format2_overlay_header(
    meta: ChunkMeta, container_version: int, recipe_kind: str | None
) -> None:
    if meta.layer == "height":
        _validate_format2_height_header(meta, container_version)
        return
    if (
        recipe_kind != "structural-repair-overlay-v1"
        or meta.layer != "water"
        or meta.lod not in (0, 1)
        or container_version != 1
        or meta.enc != 1
        or meta.count != 0
    ):
        raise ValueError(f"unsupported format-2 overlay header: {meta}")


def _manifest_grid(manifest: dict[str, Any]) -> dict[str, int]:
    if manifest.get("format") == 1:
        return {
            "anchorE": int(manifest["anchor"]["e"]),
            "anchorN": int(manifest["anchor"]["n"]),
            "chunkMeters": int(manifest["chunkMeters"]),
            "chunkRes": int(manifest["chunkRes"]),
            "lodStep": int(manifest["lodStep"]),
        }
    if manifest.get("format") == 2:
        grid = manifest["grid"]
        return {
            "anchorE": int(grid["anchorE"]),
            "anchorN": int(grid["anchorN"]),
            "chunkMeters": int(grid["chunkMeters"]),
            "chunkRes": int(grid["chunkRes"]),
            "lodStep": int(grid["lodStep"]),
        }
    raise ValueError(f"unsupported manifest format {manifest.get('format')!r}")


def audit_release(
    manifest_path: Path,
    expected_sha256: str,
    out_root: Path = DATA_OUT,
) -> VerifiedBaseRelease:
    manifest_blob = manifest_path.read_bytes()
    actual_manifest_sha = _sha256_bytes(manifest_blob)
    if actual_manifest_sha != expected_sha256:
        raise ValueError(
            f"manifest SHA-256 mismatch: {actual_manifest_sha} != {expected_sha256}"
        )
    manifest = json.loads(manifest_blob)
    manifest_format = int(manifest["format"])
    if manifest_format == 2:
        if manifest.get("containers") != ["LAC1", "LAC2"]:
            raise ValueError("format-2 release must declare LAC1 and LAC2 containers")
        build = manifest.get("build", {})
        if build.get("verified") is not True or not _DIGEST_RE.fullmatch(build.get("microVerifySha256", "")):
            raise ValueError("format-2 release is not bound to a passed micro verification")
    grid = _manifest_grid(manifest)
    chunk_count = 0
    total_bytes = 0
    records_by_layer: dict[str, tuple[IndexRecord, ...]] = {}
    manifest_dir = manifest_path.parent
    for layer, layer_meta in sorted(manifest["layers"].items()):
        index_path = _safe_child(manifest_dir, layer_meta["index"])
        if manifest_format == 2 and "indexSha256" not in layer_meta:
            raise ValueError(f"{layer}: format-2 index lacks SHA-256 identity")
        if "indexSha256" in layer_meta and _sha256_file(index_path) != layer_meta["indexSha256"]:
            raise ValueError(f"{layer}: index SHA-256 mismatch")
        records = read_v1_index(index_path) if manifest_format == 1 else read_v2_index(index_path)
        records_by_layer[layer] = records
        if len(records) != int(layer_meta["count"]):
            raise ValueError(f"{layer}: index count mismatch")
        if sorted({r.lod for r in records}) != list(layer_meta["lods"]):
            raise ValueError(f"{layer}: index LOD set mismatch")
        if sum(r.size for r in records) != int(layer_meta["bytes"]):
            raise ValueError(f"{layer}: index byte total mismatch")
        for rec in records:
            content_path = _content_path(out_root, layer, rec)
            if not content_path.is_file():
                raise ValueError(f"missing content object: {content_path}")
            if content_path.stat().st_size != rec.size:
                raise ValueError(f"content size mismatch: {content_path}")
            digest = bytes.fromhex(_sha256_file(content_path))
            if int.from_bytes(digest[:8], "big") != rec.hash64:
                raise ValueError(f"content hash mismatch: {content_path}")
            container_version, meta, _ = read_chunk_any(content_path)
            if manifest_format == 1 and container_version != 1:
                raise ValueError(f"format-1 release references non-LAC1 content: {content_path}")
            if rec.lod < 0 and container_version != 2:
                raise ValueError(f"negative LOD must use LAC2: {content_path}")
            if manifest_format == 2 and layer == "height":
                _validate_format2_height_header(meta, container_version)
            _validate_header(meta, layer, rec.lod, rec.cx, rec.cz, grid)
            chunk_count += 1
            total_bytes += rec.size
    if manifest_format == 2:
        height_records = records_by_layer.get("height", ())
        height_keys = {record.key for record in height_records}
        coverage = manifest.get("coverage", {}).get("height")
        if not isinstance(coverage, dict):
            raise ValueError("format-2 release lacks declared height coverage")
        parent = HeightChunkId(*coverage["parent"])
        authority = HeightChunkId(*coverage["authority"])
        fine = tuple(HeightChunkId(*key) for key in coverage["publishedFine"])
        if parent.lod != -1 or authority != parent_of(parent):
            raise ValueError("format-2 height coverage has an invalid parent/authority chain")
        if set(fine) != set(children_of(parent)):
            raise ValueError("format-2 fine coverage is not the complete 4x4 parent rectangle")
        declared_negative = {(chunk.lod, chunk.cx, chunk.cz) for chunk in (*fine, parent)}
        indexed_negative = {key for key in height_keys if key[0] < 0}
        if declared_negative != indexed_negative or (authority.lod, authority.cx, authority.cz) not in height_keys:
            raise ValueError("format-2 indexed height keys do not match declared coverage")
        ancestor = authority
        required_floor = set()
        while ancestor.lod < 4:
            ancestor = parent_of(ancestor)
            required_floor.add((ancestor.lod, ancestor.cx, ancestor.cz))
        if not required_floor.issubset(height_keys):
            raise ValueError(
                f"format-2 release lacks authority ancestors: {sorted(required_floor - height_keys)}"
            )
    return VerifiedBaseRelease(manifest_path, actual_manifest_sha, chunk_count, total_bytes)


def audit_base_release(
    manifest_path: Path,
    expected_sha256: str,
    out_root: Path = DATA_OUT,
) -> VerifiedBaseRelease:
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get("format") != 1:
        raise ValueError("base release must be manifest format 1")
    grid = _manifest_grid(manifest)
    if (grid["chunkMeters"], grid["chunkRes"], grid["lodStep"]) != (2048, 2048, 4):
        raise ValueError(f"base release uses a noncanonical grid: {grid}")
    return audit_release(manifest_path, expected_sha256, out_root)


def materialize_corrected_format1_base(
    *,
    base: BaseConfig,
    recipe_sha256: str,
    pinned_manifest_path: Path,
    pinned_manifest_sha256: str,
    content_root: Path,
    release_root: Path,
    replacements: dict[tuple[str, int, int, int], Path],
    tombstones: tuple[tuple[str, int, int, int], ...] = (),
) -> CorrectedFormat1BaseRelease:
    """Merge sparse LAC1 replacements into a pinned format-1 base, fail closed."""
    _validate_digest(recipe_sha256)
    audit_base_release(pinned_manifest_path, pinned_manifest_sha256, content_root)
    grid = {
        "anchorE": base.grid.anchor_e,
        "anchorN": base.grid.anchor_n,
        "chunkMeters": base.grid.chunk_m,
        "chunkRes": base.grid.chunk_res,
        "lodStep": base.grid.lod_step,
    }
    base_release = _snapshot_base_release(
        release_root,
        pinned_manifest_path,
        pinned_manifest_sha256,
        content_root,
        grid,
    )
    entries: list[dict[str, Any]] = []
    for key, path in sorted(replacements.items()):
        layer, lod, cx, cz = key
        container_version, meta, _ = read_chunk_any(path)
        if container_version != 1 or lod < 0:
            raise ValueError(f"corrected format-1 replacement is not LAC1: {path}")
        _validate_header(meta, layer, lod, cx, cz, grid)
        digest = _sha256_file(path)
        entries.append(
            {
                "layer": layer,
                "lod": lod,
                "cx": cx,
                "cz": cz,
                "relativePath": path.as_posix(),
                "size": path.stat().st_size,
                "sha256": digest,
                "enc": meta.enc,
                "res": meta.res,
                "count": meta.count,
                "flags": meta.flags,
                "qoffset": meta.qoffset,
                "qscale": meta.qscale,
                "containerVersion": container_version,
                "source": "corrected-format1",
            }
        )
        destination = content_root / "c" / layer / str(lod) / f"{cx}_{cz}.{digest[:8]}.bin"
        _install_content(path, destination, digest)
    if not entries:
        raise ValueError("corrected format-1 base has no replacements")
    normalized_tombstones = [list(key) for key in sorted(tombstones)]
    if set(replacements).intersection(tombstones):
        raise ValueError("corrected format-1 key is both replacement and tombstone")
    plan = {
        "manifestFormat": 1,
        "cookRev": int(json.loads(pinned_manifest_path.read_bytes())["cookRev"]) + 1,
        "grid": grid,
        "codec": base_release["codec"],
        "attribution": base_release["attribution"],
        "dictionaries": base_release["dictionaries"],
        "layerSchemas": base_release["layerSchemas"],
        "baseRelease": base_release,
        "recipeSha256": recipe_sha256,
        "microRecipeKind": None,
        "chunks": entries,
        "tombstones": normalized_tombstones,
    }
    manifest, indexes = _manifest_from_plan(plan)
    manifest_blob = _json_bytes(manifest)
    manifest_sha = _sha256_bytes(manifest_blob)
    destination = release_root / "m" / manifest_sha[:16]
    manifest_path = destination / "manifest.json"
    _atomic_create_or_verify(manifest_path, manifest_blob)
    for layer, payload in indexes.items():
        _atomic_create_or_verify(destination / "index" / f"{layer}.bin", payload)
    audit = audit_base_release(manifest_path, manifest_sha, content_root)
    _validate_overlay_content(plan, content_root)
    verification = {
        "format": 1,
        "role": "corrected-format1-base-verification",
        "recipeSha256": recipe_sha256,
        "pinnedManifestSha256": pinned_manifest_sha256,
        "manifestSha256": manifest_sha,
        "replacementCount": len(entries),
        "tombstoneCount": len(normalized_tombstones),
        "chunkCount": audit.chunk_count,
        "totalBytes": audit.total_bytes,
        "passed": True,
    }
    verification_blob = _json_bytes(verification)
    verification_path = release_root / "verification.json"
    _atomic_create_or_verify(verification_path, verification_blob)
    return CorrectedFormat1BaseRelease(
        manifest_path,
        manifest_sha,
        verification_path,
        _sha256_bytes(verification_blob),
    )


def _snapshot_base_release(
    build_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    base_out_root: Path,
    expected_grid: dict[str, Any],
) -> dict[str, Any]:
    manifest_blob = manifest_path.read_bytes()
    manifest = json.loads(manifest_blob)
    base_grid = {
        "anchorE": int(manifest["anchor"]["e"]),
        "anchorN": int(manifest["anchor"]["n"]),
        "chunkMeters": int(manifest["chunkMeters"]),
        "chunkRes": int(manifest["chunkRes"]),
        "lodStep": int(manifest["lodStep"]),
    }
    if base_grid != expected_grid:
        raise ValueError(f"base release grid differs from build grid: {base_grid} != {expected_grid}")

    snapshot_root = build_root / "inputs" / "base"
    _atomic_create_or_verify(snapshot_root / "manifest.json", manifest_blob)
    chunks: list[dict[str, Any]] = []
    index_sha256: dict[str, str] = {}
    layer_schemas: dict[str, Any] = {}
    dynamic_fields = {"lods", "count", "bytes", "index"}
    for layer, layer_meta in sorted(manifest["layers"].items()):
        index_path = _safe_child(manifest_path.parent, layer_meta["index"])
        index_blob = index_path.read_bytes()
        _atomic_create_or_verify(snapshot_root / "index" / f"{layer}.bin", index_blob)
        index_sha256[layer] = _sha256_bytes(index_blob)
        layer_schemas[layer] = {k: v for k, v in layer_meta.items() if k not in dynamic_fields}
        for rec in read_v1_index(index_path):
            content_path = _content_path(base_out_root, layer, rec)
            digest = _sha256_file(content_path)
            meta = read_chunk(content_path)[0]
            chunks.append(
                {
                    "layer": layer,
                    "lod": rec.lod,
                    "cx": rec.cx,
                    "cz": rec.cz,
                    "contentRelativePath": content_path.relative_to(base_out_root).as_posix(),
                    "size": rec.size,
                    "sha256": digest,
                    "enc": meta.enc,
                    "res": meta.res,
                    "count": meta.count,
                    "flags": meta.flags,
                    "qoffset": meta.qoffset,
                    "qscale": meta.qscale,
                    "containerVersion": 1,
                    "source": "base",
                }
            )
    chunks.sort(key=lambda entry: (entry["layer"], entry["lod"], entry["cx"], entry["cz"]))
    try:
        manifest_reference = manifest_path.resolve().relative_to(base_out_root.resolve()).as_posix()
    except ValueError:
        manifest_reference = str(manifest_path.resolve())
    return {
        "manifestSha256": manifest_sha256,
        "manifestReference": manifest_reference,
        "sourceRoot": str(base_out_root.resolve()),
        "snapshotManifestSha256": _sha256_bytes(manifest_blob),
        "indexSha256": index_sha256,
        "chunks": chunks,
        "layerSchemas": layer_schemas,
        "dictionaries": {
            key: manifest.get(key, {})
            for key in ("speciesMap", "understoryMap", "debrisMap")
        },
        "attribution": manifest["attribution"],
        "codec": manifest["codec"],
    }


def _staged_chunk_entries(chunks_root: Path, grid: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, int]] = set()
    for path in sorted(p for p in chunks_root.rglob("*") if p.is_file()):
        rel = path.relative_to(chunks_root)
        if len(rel.parts) != 3 or path.suffix != ".lac":
            raise ValueError(f"unexpected file in isolated chunk staging root: {path}")
    for path in sorted(chunks_root.glob("*/*/*.lac")):
        try:
            layer = path.parent.parent.name
            lod = int(path.parent.name)
            cx, cz = (int(v) for v in path.stem.split("_"))
        except ValueError as exc:
            raise ValueError(f"invalid staged chunk path: {path}") from exc
        key = layer, lod, cx, cz
        if key in seen:
            raise ValueError(f"duplicate staged chunk key: {key}")
        seen.add(key)
        container_version, meta, _ = read_chunk_any(path)
        if lod < 0 and container_version != 2:
            raise ValueError(f"negative LOD staged chunk must use LAC2: {path}")
        _validate_header(meta, layer, lod, cx, cz, grid)
        entries.append(
            {
                "layer": layer,
                "lod": lod,
                "cx": cx,
                "cz": cz,
                "relativePath": path.relative_to(chunks_root.parent).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
                "enc": meta.enc,
                "res": meta.res,
                "count": meta.count,
                "flags": meta.flags,
                "qoffset": meta.qoffset,
                "qscale": meta.qscale,
                "containerVersion": container_version,
                "source": "overlay",
            }
        )
    if not entries:
        raise ValueError(f"no staged chunks found under {chunks_root}")
    return entries


def create_micro_expectation(
    base: BaseConfig,
    build_digest: str,
    parent_cx: int,
    parent_cz: int,
    base_manifest_path: Path,
    base_manifest_sha256: str,
    work_root: Path = DATA_WORK,
    base_out_root: Path = DATA_OUT,
    exemplar_manifest_path: Path | None = None,
) -> Path:
    """Freeze independently derived Stage-1 coverage before the cook writes chunks."""
    _validate_digest(build_digest)
    if base_manifest_sha256 != MICRO_V1_BASE_SHA256:
        raise ValueError("micro v1 base is not the explicitly approved manifest identity")
    audit_base_release(base_manifest_path, base_manifest_sha256, base_out_root)
    if exemplar_manifest_path is None:
        derived_digest, recipe_inputs = derive_micro_fixture_recipe(
            parent_cx, parent_cz, base_manifest_path
        )
        recipe_kind = "retention-fixture"
    else:
        derived_digest, recipe_inputs = derive_micro_synthesis_recipe(
            parent_cx, parent_cz, base_manifest_path, exemplar_manifest_path
        )
        recipe_kind = "measured-synthesis-pilot"
    if build_digest != derived_digest:
        raise ValueError(
            f"build digest is not the derived micro recipe: {build_digest} != {derived_digest}"
        )
    build_root = work_root / "builds" / build_digest
    chunks_root = build_root / "chunks"
    if chunks_root.exists() and any(path.is_file() for path in chunks_root.rglob("*")):
        raise ValueError("micro expectation must be frozen before any staged chunk exists")
    hero = plan_hero(parent_cx, parent_cz)
    grid = {
        "anchorE": base.grid.anchor_e,
        "anchorN": base.grid.anchor_n,
        "chunkMeters": base.grid.chunk_m,
        "chunkRes": base.grid.chunk_res,
        "lodStep": base.grid.lod_step,
    }
    expectation = {
        "format": 1,
        "recipeKind": recipe_kind,
        "recipeSha256": build_digest,
        "baseManifestSha256": base_manifest_sha256,
        "recipeInputs": recipe_inputs,
        "grid": grid,
        "parent": [hero.parent.lod, hero.parent.cx, hero.parent.cz],
        "publishedFine": [[c.lod, c.cx, c.cz] for c in hero.published_fine],
        "transientSupport": [[c.lod, c.cx, c.cz] for c in hero.transient_support],
        "authority": [hero.authority_lod0.lod, hero.authority_lod0.cx, hero.authority_lod0.cz],
        "expectedPublished": [
            ["height", c.lod, c.cx, c.cz] for c in (*hero.published_fine, hero.parent)
        ],
        "verifier": {
            "id": VERIFIER_ID,
            "sourceSha256": verifier_source_sha256(),
        },
    }
    blob = _json_bytes(expectation)
    path = build_root / "expectation.json"
    _atomic_create_or_verify(path, blob)
    _atomic_create_or_verify(
        build_root / "expectation.sha256", (_sha256_bytes(blob) + "\n").encode()
    )
    return path


def _load_micro_expectation(build_root: Path, build_digest: str) -> dict[str, Any]:
    blob = (build_root / "expectation.json").read_bytes()
    expected_sha = (build_root / "expectation.sha256").read_text().strip()
    if _sha256_bytes(blob) != expected_sha:
        raise ValueError("micro expectation does not match expectation.sha256")
    expectation = json.loads(blob)
    if expectation.get("format") != 1 or expectation.get("recipeSha256") != build_digest:
        raise ValueError("micro expectation identity mismatch")
    return expectation


def _measured_synthesis_cook_revision(
    build_root: Path,
    build_digest: str,
    requested_revision: int,
) -> int:
    """Derive the release revision from the recipe-bound synthesis evidence."""
    path = build_root / "evidence" / "micro-synthesis-cook.json"
    evidence = json.loads(path.read_bytes())
    revision = evidence.get("cookRevision")
    if (
        evidence.get("format") != 1
        or evidence.get("recipeSha256") != build_digest
        or evidence.get("cook") != "exemplar-driven-production-pilot-v1"
        or isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 1
    ):
        raise ValueError("invalid measured-synthesis cook revision evidence")
    if requested_revision != revision:
        raise ValueError(
            "release cook revision differs from measured-synthesis evidence: "
            f"{requested_revision} != {revision}"
        )
    return revision


def create_build_plan(
    base: BaseConfig,
    build_digest: str,
    cook_rev: int,
    work_root: Path = DATA_WORK,
    base_manifest_path: Path | None = None,
    base_manifest_sha256: str | None = None,
    base_out_root: Path = DATA_OUT,
    manifest_format: int = 1,
    micro_parent: tuple[int, int] | None = None,
) -> Path:
    if manifest_format not in (1, 2):
        raise ValueError("manifest format must be 1 or 2")
    _validate_digest(build_digest)
    build_root = work_root / "builds" / build_digest
    chunks_root = build_root / "chunks"
    shared_root = (work_root / "chunks").resolve()
    if chunks_root.resolve() == shared_root:
        raise ValueError("transactional plans may not use the shared data/work/chunks directory")
    g = base.grid
    grid = {
        "anchorE": g.anchor_e,
        "anchorN": g.anchor_n,
        "chunkMeters": g.chunk_m,
        "chunkRes": g.chunk_res,
        "lodStep": g.lod_step,
    }
    entries = _staged_chunk_entries(chunks_root, grid)
    if manifest_format == 1 and any(entry["containerVersion"] != 1 for entry in entries):
        raise ValueError("manifest format 1 may contain only LAC1 chunks")
    micro_coverage = None
    micro_recipe_kind = None
    structural_verifier_inputs_sha256 = None
    planned_tombstones: list[list[Any]] = []
    effective_cook_rev = cook_rev
    if manifest_format == 2:
        if base_manifest_path is None or base_manifest_sha256 is None:
            raise ValueError("manifest format 2 requires an explicitly pinned base release")
        if base.encode.codec != "deflate":
            raise ValueError("manifest format 2 requires the deflate codec")
        expectation = _load_micro_expectation(build_root, build_digest)
        micro_recipe_kind = expectation.get("recipeKind")
        if micro_recipe_kind not in (
            "retention-fixture",
            "measured-synthesis-pilot",
            "structural-repair-overlay-v1",
            "research-microtopography-preview-v1",
            "research-microtopography-generalization-preview-v1",
        ):
            raise ValueError(f"unsupported micro recipe kind {micro_recipe_kind!r}")
        if (
            micro_recipe_kind
            not in (
                "structural-repair-overlay-v1",
                "research-microtopography-preview-v1",
                "research-microtopography-generalization-preview-v1",
            )
            and base_manifest_sha256 != MICRO_V1_BASE_SHA256
        ):
            raise ValueError("manifest format 2 base is not the approved micro v1 release")
        if expectation["baseManifestSha256"] != base_manifest_sha256 or expectation["grid"] != grid:
            raise ValueError("micro expectation does not match the pinned base/grid")
        if micro_recipe_kind == "structural-repair-overlay-v1":
            from .terrain.repair.verify import (
                VERIFIER_ID as STRUCTURAL_VERIFIER_ID,
                load_verifier_inputs,
                verifier_source_sha256 as structural_verifier_source_sha256,
            )

            verifier_inputs, structural_verifier_inputs_sha256 = load_verifier_inputs(
                build_root, build_digest
            )
            water_entry = verifier_inputs["artifacts"]["waterTransaction"]
            water_path = _safe_child(build_root, water_entry["path"])
            water_transaction = json.loads(water_path.read_bytes())
            if _sha256_file(water_path) != water_entry["sha256"]:
                raise ValueError("corrected-water transaction changed before planning")
            for row in water_transaction.get("artifacts", []):
                key = ["water", *row.get("key", [])]
                if row.get("disposition") == "remove-inherited":
                    planned_tombstones.append(key)
                elif row.get("disposition") != "replacement":
                    raise ValueError("corrected-water transaction contains an invalid state")
            planned_verifier = {
                "id": STRUCTURAL_VERIFIER_ID,
                "sourceSha256": structural_verifier_source_sha256(),
            }
        elif micro_recipe_kind == "research-microtopography-preview-v1":
            from .terrain.microtopography.forest_exemplar.preview_verify import (
                VERIFIER_ID as FOREST_PREVIEW_VERIFIER_ID,
                verifier_source_sha256 as forest_preview_verifier_source_sha256,
            )

            planned_verifier = {
                "id": FOREST_PREVIEW_VERIFIER_ID,
                "sourceSha256": forest_preview_verifier_source_sha256(),
            }
            if expectation.get("verifier") != planned_verifier:
                raise ValueError("forest preview expectation names a different verifier")
        elif micro_recipe_kind == "research-microtopography-generalization-preview-v1":
            from .terrain.microtopography.forest_exemplar.generalization_preview_verify import (
                VERIFIER_ID as FOREST_GENERALIZATION_VERIFIER_ID,
                verifier_source_sha256 as forest_generalization_verifier_source_sha256,
            )

            planned_verifier = {
                "id": FOREST_GENERALIZATION_VERIFIER_ID,
                "sourceSha256": forest_generalization_verifier_source_sha256(),
            }
            if expectation.get("verifier") != planned_verifier:
                raise ValueError(
                    "forest generalization preview expectation names a different verifier"
                )
        else:
            planned_verifier = {
                "id": VERIFIER_ID,
                "sourceSha256": verifier_source_sha256(),
            }
            if expectation.get("verifier") != planned_verifier:
                raise ValueError("micro expectation names a different verifier implementation")
        expected = {tuple(key) for key in expectation["expectedPublished"]}
        if micro_recipe_kind == "structural-repair-overlay-v1":
            expected.update(
                ("water", *row["key"])
                for row in water_transaction["artifacts"]
                if row.get("disposition") == "replacement"
            )
        actual = {(e["layer"], e["lod"], e["cx"], e["cz"]) for e in entries}
        if actual != expected:
            raise ValueError(
                f"micro overlay key set mismatch; missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        for entry in entries:
            _validate_format2_overlay_header(
                ChunkMeta(
                    entry["layer"], entry["lod"], entry["enc"], entry["cx"], entry["cz"],
                    entry["res"], entry["count"], 0.0, 0.0, entry["qoffset"],
                    entry["qscale"], entry["flags"],
                ),
                entry["containerVersion"],
                micro_recipe_kind,
            )
        if micro_parent is not None and expectation["parent"][1:] != list(micro_parent):
            raise ValueError("requested micro parent differs from the frozen expectation")
        micro_coverage = {
            key: expectation[key]
            for key in ("parent", "publishedFine", "transientSupport", "authority")
        }
        if micro_recipe_kind == "measured-synthesis-pilot":
            effective_cook_rev = _measured_synthesis_cook_revision(
                build_root, build_digest, cook_rev
            )
    base_release = None
    dictionaries = {
        "speciesMap": _species_dictionary(),
        "understoryMap": _understory_dictionary(),
        "debrisMap": _debris_dictionary(),
    }
    layer_schemas: dict[str, Any] = {}
    if (base_manifest_path is None) != (base_manifest_sha256 is None):
        raise ValueError("base manifest path and SHA-256 must be provided together")
    if base_manifest_path is not None and base_manifest_sha256 is not None:
        audit_base_release(base_manifest_path, base_manifest_sha256, base_out_root)
        base_release = _snapshot_base_release(
            build_root, base_manifest_path, base_manifest_sha256, base_out_root, grid
        )
        layer_schemas.update(base_release["layerSchemas"])
        dictionaries = base_release["dictionaries"]
        if manifest_format == 2:
            if base_release["codec"] != "deflate":
                raise ValueError("format-2 base release must use deflate")
            authority = tuple(micro_coverage["authority"])
            base_keys = {
                (e["lod"], e["cx"], e["cz"])
                for e in base_release["chunks"]
                if e["layer"] == "height"
            }
            if authority not in base_keys:
                raise ValueError(f"base release lacks required LOD0 authority chunk {authority}")
    for layer in sorted({entry["layer"] for entry in entries}):
        layer_schemas.setdefault(layer, LAYER_DOC.get(layer, {}))
    plan = {
        "format": PLAN_FORMAT,
        "recipeSha256": build_digest,
        "manifestFormat": manifest_format,
        "cookRev": effective_cook_rev,
        "grid": grid,
        "codec": base.encode.codec,
        "attribution": base_release["attribution"] if base_release else base.attribution,
        "dictionaries": dictionaries,
        "layerSchemas": layer_schemas,
        "baseRelease": base_release,
        "microCoverage": micro_coverage,
        "microRecipeKind": micro_recipe_kind,
        "microVerifier": planned_verifier if manifest_format == 2 else None,
        **(
            {
                "structuralVerifierInputsSha256": structural_verifier_inputs_sha256,
                "tombstones": sorted(planned_tombstones),
            }
            if micro_recipe_kind == "structural-repair-overlay-v1"
            else {}
        ),
        "chunks": entries,
    }
    plan_blob = _json_bytes(plan)
    plan_path = build_root / "plan.json"
    _atomic_create_or_verify(plan_path, plan_blob)
    _atomic_create_or_verify(build_root / "plan.sha256", (_sha256_bytes(plan_blob) + "\n").encode())
    return plan_path


def _load_plan(build_digest: str, work_root: Path) -> tuple[Path, dict[str, Any], bytes]:
    _validate_digest(build_digest)
    build_root = work_root / "builds" / build_digest
    plan_path = build_root / "plan.json"
    plan_blob = plan_path.read_bytes()
    expected = (build_root / "plan.sha256").read_text().strip()
    if _sha256_bytes(plan_blob) != expected:
        raise ValueError("plan.json does not match plan.sha256")
    plan = json.loads(plan_blob)
    if plan.get("format") != PLAN_FORMAT or plan.get("recipeSha256") != build_digest:
        raise ValueError("plan identity mismatch")
    return build_root, plan, plan_blob


def _overlay_set_sha256(plan: dict[str, Any]) -> str:
    identity = [
        {
            "key": [entry["layer"], entry["lod"], entry["cx"], entry["cz"]],
            "size": entry["size"],
            "sha256": entry["sha256"],
        }
        for entry in sorted(
            plan["chunks"], key=lambda e: (e["layer"], e["lod"], e["cx"], e["cz"])
        )
    ]
    return _sha256_bytes(_json_bytes(identity))


def micro_verification_binding(
    build_digest: str,
    work_root: Path = DATA_WORK,
) -> dict[str, Any]:
    """Identity fields an independent micro verifier must bind and sign off."""
    _, plan, plan_blob = _load_plan(build_digest, work_root)
    if plan["manifestFormat"] != 2:
        raise ValueError("micro verification applies only to manifest format 2")
    return {
        "format": 1,
        "recipeSha256": build_digest,
        "planSha256": _sha256_bytes(plan_blob),
        "overlaySetSha256": _overlay_set_sha256(plan),
    }


def _require_micro_verification(
    build_root: Path,
    plan: dict[str, Any],
    plan_blob: bytes,
) -> tuple[dict[str, Any], str] | None:
    if plan["manifestFormat"] != 2:
        return None
    recipe_kind = plan.get("microRecipeKind")
    if recipe_kind == "structural-repair-overlay-v1":
        from .terrain.repair.verify import (
            GATES as STRUCTURAL_VERIFY_GATES,
            VERIFIER_ID as STRUCTURAL_VERIFIER_ID,
            verify_structural_repair,
        )

        base_release = plan.get("baseRelease") or {}
        try:
            report = verify_structural_repair(
                plan["recipeSha256"],
                build_root / "inputs" / "base" / "manifest.json",
                Path(base_release["sourceRoot"]),
                build_root.parent.parent,
                encode=EncodeConfig(plan["codec"], 0.01, 0.01, 19, 1),
            )
        except Exception as exc:
            raise ValueError(f"independent structural repair verification failed: {exc}") from exc
        path = build_root / "structural-verify.json"
        blob = path.read_bytes()
        expected = {
            "format": 1,
            "recipeKind": recipe_kind,
            "recipeSha256": plan["recipeSha256"],
            "releaseDisposition": "preview-only",
            "planSha256": _sha256_bytes(plan_blob),
            "overlaySetSha256": _overlay_set_sha256(plan),
            "verifier": STRUCTURAL_VERIFIER_ID,
        }
        for key, value in expected.items():
            if report.get(key) != value:
                raise ValueError(f"structural verification {key} does not match the frozen plan")
        planned_verifier = plan.get("microVerifier") or {}
        if (
            report.get("verifier") != planned_verifier.get("id")
            or report.get("verifierSourceSha256") != planned_verifier.get("sourceSha256")
        ):
            raise ValueError("structural verification was not produced by the planned verifier")
        gates = report.get("gates")
        if not isinstance(gates, dict) or set(gates) != set(STRUCTURAL_VERIFY_GATES):
            raise ValueError("structural verification does not contain every hard gate")
        for gate in STRUCTURAL_VERIFY_GATES:
            result = gates[gate]
            if (
                not isinstance(result, dict)
                or result.get("passed") is not True
                or "evidence" not in result
                or result.get("evidenceSha256") != evidence_sha256(result["evidence"])
            ):
                raise ValueError(f"structural verification gate lacks valid evidence: {gate}")
        if report.get("passed") is not True:
            raise ValueError("structural verification has not passed")
        return report, _sha256_bytes(blob)
    if recipe_kind == "research-microtopography-preview-v1":
        from .terrain.microtopography.forest_exemplar.preview_verify import (
            verify_irregular_forest_preview,
        )

        verifier = verify_irregular_forest_preview
    elif recipe_kind == "research-microtopography-generalization-preview-v1":
        from .terrain.microtopography.forest_exemplar.generalization_preview_verify import (
            verify_generalization_preview,
        )

        verifier = verify_generalization_preview
    else:
        verifier = {
            "retention-fixture": verify_micro_fixture,
            "measured-synthesis-pilot": verify_micro_synthesis,
        }.get(recipe_kind)
    if not VERIFIER_CAN_AUTHORIZE_RELEASE:
        raise ValueError(
            "micro release authorization is closed until the verifier recomputes "
            "hierarchy, apron, mask, determinism, and transient-parent evidence"
        )
    base_release = plan.get("baseRelease") or {}
    if verifier is None:
        raise ValueError(f"unsupported micro verification recipe kind {recipe_kind!r}")
    try:
        verifier(
            plan["recipeSha256"],
            build_root / "inputs" / "base" / "manifest.json",
            Path(base_release["sourceRoot"]),
            build_root.parent.parent,
        )
    except Exception as exc:
        raise ValueError(f"independent micro verification failed: {exc}") from exc
    path = build_root / "micro-verify.json"
    blob = path.read_bytes()
    report = json.loads(blob)
    expected = {
        "format": 1,
        "recipeSha256": plan["recipeSha256"],
        "planSha256": _sha256_bytes(plan_blob),
        "overlaySetSha256": _overlay_set_sha256(plan),
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"micro verification {key} does not match the frozen plan")
    planned_verifier = plan.get("microVerifier") or {}
    if (
        report.get("verifier") != planned_verifier.get("id")
        or report.get("verifierSourceSha256") != planned_verifier.get("sourceSha256")
    ):
        raise ValueError("micro verification was not produced by the current verifier implementation")
    gates = report.get("gates")
    if not isinstance(gates, dict) or set(gates) != set(MICRO_VERIFY_GATES):
        raise ValueError("micro verification does not pass every required hard gate")
    for gate in MICRO_VERIFY_GATES:
        result = gates[gate]
        evidence_valid = (
            isinstance(result, dict)
            and "evidence" in result
            and result.get("evidenceSha256") == evidence_sha256(result["evidence"])
        )
        preview_mask_na = (
            gate == "hardMasks"
            and (report.get("fixtureOnly") is True or report.get("pilotOnly") is True)
            and result.get("status") == "notApplicable"
            and result.get("applicable") is False
        )
        if not evidence_valid or (result.get("passed") is not True and not preview_mask_na):
            raise ValueError(f"micro verification gate lacks valid evidence: {gate}")
    artifacts = report.get("transientArtifacts")
    if not isinstance(artifacts, list):
        raise ValueError("micro verification lacks transient support artifacts")
    expected_support = {tuple(key) for key in plan["microCoverage"]["transientSupport"]}
    artifact_keys = {tuple(item.get("key", ())) for item in artifacts}
    if artifact_keys != expected_support:
        raise ValueError("micro verification transient support set differs from the frozen plan")
    if any(not _DIGEST_RE.fullmatch(item.get("sha256", "")) for item in artifacts):
        raise ValueError("micro verification has an invalid transient artifact digest")
    if report.get("transientSupportMerkleRoot") != transient_merkle_root(artifacts):
        raise ValueError("micro verification transient support Merkle root mismatch")
    return report, _sha256_bytes(blob)


def _validate_staged_plan(build_root: Path, plan: dict[str, Any]) -> list[Path]:
    chunks_root = build_root / "chunks"
    invalid = [
        path
        for path in chunks_root.rglob("*")
        if path.is_file()
        and (len(path.relative_to(chunks_root).parts) != 3 or path.suffix != ".lac")
    ]
    if invalid:
        raise ValueError(f"unexpected files in isolated chunk staging root: {sorted(invalid)}")
    planned_rel = {entry["relativePath"] for entry in plan["chunks"]}
    actual_rel = {
        path.relative_to(build_root).as_posix()
        for path in chunks_root.glob("*/*/*.lac")
    }
    if actual_rel != planned_rel:
        missing = sorted(planned_rel - actual_rel)
        unexpected = sorted(actual_rel - planned_rel)
        raise ValueError(f"staged key set mismatch; missing={missing}, unexpected={unexpected}")
    paths: list[Path] = []
    for entry in plan["chunks"]:
        path = _safe_child(build_root, entry["relativePath"])
        if path.stat().st_size != entry["size"] or _sha256_file(path) != entry["sha256"]:
            raise ValueError(f"staged chunk changed after planning: {path}")
        container_version, meta, _ = read_chunk_any(path)
        if container_version != entry["containerVersion"]:
            raise ValueError(f"staged container version changed after planning: {path}")
        _validate_header(meta, entry["layer"], entry["lod"], entry["cx"], entry["cz"], plan["grid"])
        if plan["manifestFormat"] == 2:
            _validate_format2_overlay_header(
                meta, container_version, plan.get("microRecipeKind")
            )
        paths.append(path)
    return paths


def _validate_base_plan(build_root: Path, plan: dict[str, Any], out_root: Path) -> None:
    base_release = plan.get("baseRelease")
    if not base_release:
        return
    snapshot_root = build_root / "inputs" / "base"
    manifest_blob = (snapshot_root / "manifest.json").read_bytes()
    if _sha256_bytes(manifest_blob) != base_release["snapshotManifestSha256"]:
        raise ValueError("snapshotted base manifest changed after planning")
    for layer, expected in base_release["indexSha256"].items():
        if _sha256_file(snapshot_root / "index" / f"{layer}.bin") != expected:
            raise ValueError(f"snapshotted base index changed after planning: {layer}")
    for entry in base_release["chunks"]:
        path = _safe_child(out_root, entry["contentRelativePath"])
        if path.stat().st_size != entry["size"] or _sha256_file(path) != entry["sha256"]:
            raise ValueError(f"inherited content changed after planning: {path}")
        container_version, meta, _ = read_chunk_any(path)
        if container_version != entry["containerVersion"]:
            raise ValueError(f"inherited container version changed after planning: {path}")
        _validate_header(meta, entry["layer"], entry["lod"], entry["cx"], entry["cz"], plan["grid"])


def _validate_overlay_content(plan: dict[str, Any], out_root: Path) -> None:
    for entry in plan["chunks"]:
        digest = entry["sha256"]
        path = out_root / "c" / entry["layer"] / str(entry["lod"]) / (
            f"{entry['cx']}_{entry['cz']}.{digest[:8]}.bin"
        )
        if path.stat().st_size != entry["size"] or _sha256_file(path) != digest:
            raise ValueError(f"published overlay content mismatch: {path}")


def _install_content(src: Path, dest: Path, expected_sha256: str) -> None:
    if dest.exists():
        if _sha256_file(dest) != expected_sha256:
            raise ValueError(f"content-address collision: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(src, tmp)
        with tmp.open("rb") as copied:
            os.fsync(copied.fileno())
        if _sha256_file(tmp) != expected_sha256:
            raise ValueError(f"copied content hash mismatch: {src}")
        try:
            os.link(tmp, dest)
        except FileExistsError:
            if _sha256_file(dest) != expected_sha256:
                raise ValueError(f"concurrent content-address collision: {dest}")
        _fsync_dir(dest.parent)
    finally:
        tmp.unlink(missing_ok=True)


def _manifest_from_plan(
    plan: dict[str, Any],
    micro_verify_sha256: str | None = None,
    fixture_only: bool = False,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    fixture_only = fixture_only or plan.get("microRecipeKind") == "retention-fixture"
    pilot_only = plan.get("microRecipeKind") in (
        "measured-synthesis-pilot",
        "research-microtopography-preview-v1",
    )
    structural_only = plan.get("microRecipeKind") == "structural-repair-overlay-v1"
    manifest_format = int(plan["manifestFormat"])
    if manifest_format not in (1, 2):
        raise ValueError(f"unsupported planned manifest format {manifest_format}")
    combined: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for entry in (plan.get("baseRelease") or {}).get("chunks", []):
        combined[(entry["layer"], entry["lod"], entry["cx"], entry["cz"])] = entry
    for raw_key in plan.get("tombstones", []):
        if (
            not isinstance(raw_key, list)
            or len(raw_key) != 4
            or raw_key[0] != "water"
            or raw_key[1] not in (0, 1)
        ):
            raise ValueError(f"invalid structural water tombstone: {raw_key!r}")
        key = tuple(raw_key)
        if key not in combined:
            raise ValueError(f"structural water tombstone does not remove an inherited key: {raw_key}")
        del combined[key]
    for entry in plan["chunks"]:
        combined[(entry["layer"], entry["lod"], entry["cx"], entry["cz"])] = entry
    by_layer: dict[str, list[dict[str, Any]]] = {}
    for entry in combined.values():
        by_layer.setdefault(entry["layer"], []).append(entry)
    layers: dict[str, Any] = {}
    indexes: dict[str, bytes] = {}
    for layer, entries in sorted(by_layer.items()):
        entries.sort(key=lambda e: (e["lod"], e["cx"], e["cz"]))
        records = []
        for entry in entries:
            if entry["lod"] < 0 and entry["layer"] != "height":
                raise ValueError(f"negative LOD is height-only: {entry}")
            if entry["lod"] < 0 and entry["containerVersion"] != 2:
                raise ValueError(f"negative LOD must use LAC2: {entry}")
            if manifest_format == 1 and (entry["lod"] < 0 or entry["containerVersion"] != 1):
                raise ValueError("manifest format 1 supports only nonnegative LAC1 chunks")
            digest = bytes.fromhex(entry["sha256"])
            record_struct = INDEX_RECORD_V1 if manifest_format == 1 else INDEX_RECORD_V2
            records.append(
                record_struct.pack(
                    entry["lod"], entry["cx"], entry["cz"], entry["size"], int.from_bytes(digest[:8], "big")
                )
            )
        indexes[layer] = b"".join(records)
        layers[layer] = {
            **plan["layerSchemas"].get(layer, {}),
            "lods": sorted({entry["lod"] for entry in entries}),
            "count": len(entries),
            "bytes": sum(entry["size"] for entry in entries),
            "index": f"index/{layer}.bin",
            "indexSha256": _sha256_bytes(indexes[layer]),
        }
    grid = plan["grid"]
    common = {
        "format": manifest_format,
        "cookRev": plan["cookRev"],
        "crs": "EPSG:3301",
        "attribution": plan["attribution"],
        "axes": {
            "x": "east: gameX = E - anchor.e",
            "z": "south: gameZ = anchor.n - N (raster row order == +z)",
            "y": "EH2000 meters (sea = 0)",
        },
        "codec": plan["codec"],
        **plan["dictionaries"],
        "layers": layers,
    }
    if manifest_format == 1:
        manifest = {
            **common,
            "anchor": {"e": grid["anchorE"], "n": grid["anchorN"]},
            "chunkMeters": grid["chunkMeters"],
            "chunkRes": grid["chunkRes"],
            "lodStep": grid["lodStep"],
            "texelConvention": "texel (i,j) of chunk (cx,cz,lod) centers at "
            "(anchor.e + cx*F + (i+0.5)*t, anchor.n - cz*F - (j+0.5)*t), t = lodStep^lod, "
            "F = chunkMeters*t; far row/col (i or j == chunkRes) duplicates the east/south neighbor",
            "container": "LAC1 v1 (56-byte LE header + compressed payload; see chunkio.py)",
            "build": {"recipeSha256": plan["recipeSha256"]},
        }
        return manifest, indexes

    if "height" in layers:
        height_lods = layers["height"]["lods"]
        layers["height"].update(
            {
                "baseTexelMeters": 1.0,
                "finestLod": min(height_lods),
                "authorityLod": 0,
                "synthesis": (
                    "calibrated-retention-fixture-v1"
                    if fixture_only
                    else "accepted-irregular-forest-research-preview-v1"
                    if plan.get("microRecipeKind") == "research-microtopography-preview-v1"
                    else "accepted-forest-generalization-research-preview-v1"
                    if plan.get("microRecipeKind")
                    == "research-microtopography-generalization-preview-v1"
                    else "measured-synthesis-pilot-v1"
                    if pilot_only
                    else "structural-repair-overlay-v1"
                    if structural_only
                    else "microtopography-v1"
                ),
            }
        )
    base_release = plan.get("baseRelease")
    manifest = {
        **common,
        "containers": ["LAC1", "LAC2"],
        "grid": {
            "anchorE": grid["anchorE"],
            "anchorN": grid["anchorN"],
            "chunkMeters": grid["chunkMeters"],
            "chunkRes": grid["chunkRes"],
            "lodStep": grid["lodStep"],
        },
        "texelConvention": "texel = baseTexelMeters * lodStep^lod; footprint = "
        "grid.chunkMeters * lodStep^lod; far row/col duplicates the east/south neighbor",
        "build": {
            "recipeSha256": plan["recipeSha256"],
            "generatorVersion": 1,
            "verified": micro_verify_sha256 is not None,
            **({"fixtureOnly": True} if fixture_only else {}),
            **({"pilotOnly": True} if pilot_only else {}),
            **(
                {
                    "baseManifest": base_release["manifestReference"],
                    "baseManifestSha256": base_release["manifestSha256"],
                }
                if base_release
                else {}
            ),
            **({"microVerifySha256": micro_verify_sha256} if micro_verify_sha256 else {}),
        },
        "coverage": {"height": plan["microCoverage"]},
    }
    return manifest, indexes


def materialize_preview(
    build_digest: str,
    work_root: Path = DATA_WORK,
    out_root: Path = DATA_OUT,
) -> Path:
    build_root, plan, plan_blob = _load_plan(build_digest, work_root)
    paths = _validate_staged_plan(build_root, plan)
    micro_verification = _require_micro_verification(build_root, plan, plan_blob)
    base_source_root = Path((plan.get("baseRelease") or {}).get("sourceRoot", out_root))
    _validate_base_plan(build_root, plan, base_source_root)
    for entry in (plan.get("baseRelease") or {}).get("chunks", []):
        src = _safe_child(base_source_root, entry["contentRelativePath"])
        dest = _safe_child(out_root, entry["contentRelativePath"])
        _install_content(src, dest, entry["sha256"])
    for entry, src in zip(plan["chunks"], paths, strict=True):
        digest = entry["sha256"]
        dest = out_root / "c" / entry["layer"] / str(entry["lod"]) / (
            f"{entry['cx']}_{entry['cz']}.{digest[:8]}.bin"
        )
        _install_content(src, dest, digest)

    manifest, indexes = _manifest_from_plan(
        plan,
        micro_verification[1] if micro_verification is not None else None,
        bool(micro_verification and micro_verification[0].get("fixtureOnly")),
    )
    manifest_blob = _json_bytes(manifest)
    manifest_sha = _sha256_bytes(manifest_blob)
    manifest_hash = manifest_sha[:16]
    preview_dir = out_root / "builds" / build_digest / "m" / manifest_hash
    _atomic_create_or_verify(preview_dir / "manifest.json", manifest_blob)
    for layer, blob in indexes.items():
        _atomic_create_or_verify(preview_dir / "index" / f"{layer}.bin", blob)

    audit = audit_release(preview_dir / "manifest.json", manifest_sha, out_root)
    _validate_overlay_content(plan, out_root)
    verify = {
        "format": VERIFY_FORMAT,
        "recipeSha256": build_digest,
        "planSha256": _sha256_bytes(plan_blob),
        "manifestSha256": manifest_sha,
        "chunkCount": audit.chunk_count,
        "totalBytes": audit.total_bytes,
        "passed": True,
        **(
            {"microVerifySha256": micro_verification[1]}
            if micro_verification is not None
            else {}
        ),
    }
    verify_blob = _json_bytes(verify)
    _atomic_create_or_verify(build_root / "verify.json", verify_blob)
    complete = {
        "verifySha256": _sha256_bytes(verify_blob),
        "manifestSha256": manifest_sha,
        "previewManifest": (preview_dir / "manifest.json").relative_to(out_root).as_posix(),
        **(
            {"microVerifySha256": micro_verification[1]}
            if micro_verification is not None
            else {}
        ),
    }
    _atomic_create_or_verify(build_root / "COMPLETE", _json_bytes(complete))
    return preview_dir / "manifest.json"


def _load_complete(build_digest: str, work_root: Path, out_root: Path) -> tuple[dict[str, Any], Path]:
    build_root, plan, plan_blob = _load_plan(build_digest, work_root)
    complete_blob = (build_root / "COMPLETE").read_bytes()
    complete = json.loads(complete_blob)
    verify_blob = (build_root / "verify.json").read_bytes()
    if _sha256_bytes(verify_blob) != complete["verifySha256"]:
        raise ValueError("verify.json does not match COMPLETE")
    verify = json.loads(verify_blob)
    if (
        verify.get("format") != VERIFY_FORMAT
        or verify.get("recipeSha256") != build_digest
        or not verify.get("passed")
        or verify.get("planSha256") != _sha256_bytes(plan_blob)
        or verify.get("manifestSha256") != complete.get("manifestSha256")
    ):
        raise ValueError("verification report is not valid for this plan")
    micro_verification = _require_micro_verification(build_root, plan, plan_blob)
    if micro_verification is not None and verify.get("microVerifySha256") != micro_verification[1]:
        raise ValueError("verification report is not bound to micro-verify.json")
    preview_manifest = _safe_child(out_root, complete["previewManifest"])
    audit_release(preview_manifest, complete["manifestSha256"], out_root)
    expected_manifest, expected_indexes = _manifest_from_plan(
        plan, micro_verification[1] if micro_verification is not None else None
    )
    if preview_manifest.read_bytes() != _json_bytes(expected_manifest):
        raise ValueError("preview manifest does not equal the frozen plan artifacts")
    for layer, expected_index in expected_indexes.items():
        if (preview_manifest.parent / "index" / f"{layer}.bin").read_bytes() != expected_index:
            raise ValueError(f"preview index does not equal the frozen plan artifacts: {layer}")
    _validate_staged_plan(build_root, plan)
    _validate_base_plan(build_root, plan, out_root)
    _validate_overlay_content(plan, out_root)
    return complete, preview_manifest


def publish_build(
    build_digest: str,
    work_root: Path = DATA_WORK,
    out_root: Path = DATA_OUT,
) -> Path:
    _, plan, _ = _load_plan(build_digest, work_root)
    if plan.get("microRecipeKind") in (
        "measured-synthesis-pilot",
        "research-microtopography-preview-v1",
    ):
        raise ValueError("measured-synthesis pilot is immutable-preview-only and cannot update latest")
    if plan.get("microRecipeKind") == "structural-repair-overlay-v1":
        raise ValueError("structural repair overlay is preview-only and cannot update latest")
    complete, preview_manifest = _load_complete(build_digest, work_root, out_root)
    source_dir = preview_manifest.parent
    manifest_hash = complete["manifestSha256"][:16]
    dest_dir = out_root / "m" / manifest_hash
    for src in sorted(path for path in source_dir.rglob("*") if path.is_file()):
        rel = src.relative_to(source_dir)
        _atomic_create_or_verify(dest_dir / rel, src.read_bytes())
    dest_manifest = dest_dir / "manifest.json"
    audit_release(dest_manifest, complete["manifestSha256"], out_root)
    _atomic_replace(
        out_root / "latest.json",
        _json_bytes({"manifest": f"m/{manifest_hash}/manifest.json"}),
    )
    return dest_manifest
