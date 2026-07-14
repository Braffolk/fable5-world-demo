"""Independent release verifier for structural terrain repair overlays.

The verifier intentionally consumes only frozen files.  It does not call the
authority cook, the fine reevaluator, or the corrected-base transaction builder;
those implementations are evidence under test rather than verifier dependencies.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from contextlib import contextmanager
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ...config import EncodeConfig, load_base
from ...cook.chunkio import read_chunk, read_chunk_any, read_chunk_v2
from ...cook.encode import (
    decode_quant16,
    encode_quant16_checked,
    quantize16_checked,
)
from ...cook.micro_hierarchy import (
    BOX_MEAN_REDUCER_VERSION,
    box_mean_fixed,
    dependency_merkle_root,
)
from ...height_geom import HeightChunkId, children_of, parent_of
from ...grid import ChunkId
from .hierarchy import reduce_decoded_children
from .materialize import MATERIALIZER_VERSION, decode_fine_ownership
from .recipe import RECIPE_ID, RECIPE_KIND, load_structural_repair_expectation
from .storage import (
    AUTHORITY_MANIFEST_FORMAT,
    decode_baseline_tile,
    decode_structural_tile,
)
from .water_layer import reduce_water_lod1


VERIFIER_ID = "laas.terrain.structural-repair-overlay.verifier.v1"
INPUT_FORMAT = 1
REPORT_FORMAT = 1
GATES = (
    "frozenInputs",
    "authority",
    "materializedHierarchy",
    "ownershipRestoration",
    "correctedBase",
    "correctedWater",
    "publishedOverlay",
)
_INDEX_V1 = struct.Struct("<BiiIQ")
_DIGEST_LENGTH = 64
_INHERITED_ABSENT_WATER_DOMAIN = b"laas.structural-water.inherited-absent.v1\0"
OWNERSHIP_CLOSURE_VERSION = "halo-aware-shared-water-closure/2"
OWNERSHIP_RASTERIZER_ID = "laas.terrain.shared-water-closure-raster.v1"
_CORRECTED_BASE_PLAN_FORMAT = 2
_CORRECTED_ANCESTOR_SPLICE = (
    "masked-child-reduction-splice-preserve-independent-dtm/1"
)
_CORRECTED_QOFFSET_POLICY = (
    "preserve-inherited-or-retune-full-payload-outside-bitexact/1"
)
_CORRECTED_MASK_FORMAT = "packed-bits-row-major-msb/1"
_CORRECTED_WINDOW_FORMAT = "row-run-half-open-json/1"
_CORRECTED_OUTSIDE_PROOF = "decoded-f32-outside-mask-bitexact/1"
_CORRECTED_QUANTIZER_DOMAIN = (
    "final-full-payload-core-and-east-south-southeast-aprons"
)
_CORRECTED_MASK_MAGIC = b"TAM1"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=1, sort_keys=True) + "\n").encode()


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    try:
        if len(value) != _DIGEST_LENGTH or len(bytes.fromhex(value)) != 32:
            raise ValueError
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest") from error
    return value


def _safe_child(root: Path, relative: Any, name: str) -> Path:
    if not isinstance(relative, str):
        raise ValueError(f"{name} path must be relative")
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe {name} path: {relative!r}")
    root = root.resolve()
    result = (root / rel).resolve()
    if result != root and root not in result.parents:
        raise ValueError(f"{name} path leaves its frozen root")
    return result


def _checked_file(root: Path, entry: Any, name: str) -> Path:
    if not isinstance(entry, dict):
        raise ValueError(f"structural verifier lacks {name} input")
    path = _safe_child(root, entry.get("path"), name)
    expected = _digest(entry.get("sha256"), f"{name} sha256")
    size = entry.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"{name} bytes must be a nonnegative integer")
    if not path.is_file() or path.stat().st_size != size or _sha256_file(path) != expected:
        raise ValueError(f"frozen {name} input integrity mismatch: {path}")
    return path


@contextmanager
def _content_addressed_snapshot(path: Path, content_sha256: str):
    """Present a relocated verifier snapshot under its checked content address."""
    content_sha = _digest(content_sha256, "snapshot content sha256")
    if path.parent.name == content_sha:
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="laas-structural-verify-") as temporary:
        alias = Path(temporary) / content_sha
        alias.symlink_to(path.parent.resolve(), target_is_directory=True)
        yield alias / path.name


def _chunk(raw: Any, name: str) -> HeightChunkId:
    if (
        not isinstance(raw, list)
        or len(raw) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw)
    ):
        raise ValueError(f"invalid {name} chunk key")
    return HeightChunkId(*raw)


def _key(chunk: HeightChunkId) -> list[int]:
    return [chunk.lod, chunk.cx, chunk.cz]


def _array_sha256(values: np.ndarray, dtype: str) -> str:
    array = np.array(values, dtype=np.dtype(dtype), copy=True)
    if np.issubdtype(array.dtype, np.floating):
        array[np.isnan(array)] = np.nan
    return _sha256_bytes(np.ascontiguousarray(array).tobytes())


def _gate(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": True,
        "evidence": evidence,
        "evidenceSha256": _sha256_bytes(_canonical_json_bytes(evidence)),
    }


def verifier_source_sha256() -> str:
    return _sha256_file(Path(__file__))


def ownership_rasterizer_source_sha256() -> str:
    return _sha256_file(Path(__file__).with_name("shared_closure.py"))


def inherited_absent_water_sha256(
    chunk: HeightChunkId, base_manifest_sha256: str
) -> str:
    """Identity used when a fixed water dependency is absent from the base index."""
    if chunk.lod != 0:
        raise ValueError("only LOD0 inherited water can use an absent identity")
    digest = hashlib.sha256(_INHERITED_ABSENT_WATER_DOMAIN)
    digest.update(bytes.fromhex(_digest(base_manifest_sha256, "base manifest sha256")))
    digest.update(struct.pack("<Bii", chunk.lod, chunk.cx, chunk.cz))
    return digest.hexdigest()


def load_verifier_inputs(build_root: Path, recipe_sha256: str) -> tuple[dict[str, Any], str]:
    """Validate the small, immutable path/hash map supplied by orchestration."""
    path = Path(build_root) / "structural-verify-inputs.json"
    blob = path.read_bytes()
    document = json.loads(blob)
    if (
        document.get("format") != INPUT_FORMAT
        or document.get("recipeSha256") != recipe_sha256
        or set(document.get("artifacts", {}))
        != {
            "authority",
            "materializer",
            "correctedBasePlan",
            "correctedBaseVerification",
            "waterTransaction",
            "scientificClosure",
        }
    ):
        raise ValueError("structural verifier input map is incomplete or has the wrong identity")
    for name, entry in sorted(document["artifacts"].items()):
        _checked_file(Path(build_root), entry, name)
    return document, _sha256_bytes(blob)


def _validate_scientific_closure_manifest(
    path: Path,
    *,
    authority_sha256: str,
    campaign_evidence: dict[str, str],
) -> dict[str, Any]:
    document = json.loads(path.read_bytes())
    if (
        document.get("format") != 1
        or document.get("closureVersion") != OWNERSHIP_CLOSURE_VERSION
        or document.get("authorityManifestSha256") != authority_sha256
        or document.get("campaignContentSha256")
        != campaign_evidence.get("contentSha256")
        or document.get("campaignQualificationSha256")
        != campaign_evidence.get("qualificationSha256")
        or document.get("fineCoreSide") != 2049
        or document.get("haloSamples") != 16
        or document.get("expandedSide") != 2081
        or set(document.get("inputs", {}))
        != {
            "mappedWaterWkb",
            "qualifiedWaterWkb",
            "centerlineWkb",
            "profileMetadata",
            "profileArrays",
        }
    ):
        raise ValueError("scientific closure manifest has the wrong 2081-halo contract")
    for name, entry in document["inputs"].items():
        _checked_file(path.parent, entry, f"scientific closure {name}")
    return document


def _verify_authority(
    path: Path, expected_sha256: str, expected: dict[str, Any]
) -> tuple[dict[HeightChunkId, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, Any]]:
    if _sha256_file(path) != _digest(expected_sha256, "authority manifest sha256"):
        raise ValueError("authority manifest differs from the frozen verifier input")
    document = json.loads(path.read_bytes())
    if (
        document.get("format") != AUTHORITY_MANIFEST_FORMAT
        or document.get("role") != "structural_authority_0.25m"
        or document.get("morphology") != "absent"
        or document.get("tileAlignmentLod") != -2
        or document.get("tileCoreResolution") != 512
    ):
        raise ValueError("authority manifest contract mismatch")
    if _sha256_file(path) != expected.get("manifestSha256"):
        raise ValueError("recipe names a different authority manifest")
    campaign_evidence = _digest(
        document.get("evidenceSha256"), "authority campaign-content evidence"
    )
    if campaign_evidence != expected.get("evidenceSha256"):
        raise ValueError("authority campaign-content evidence differs from the recipe")
    rows = document.get("tiles")
    if not isinstance(rows, list) or len(rows) != expected.get("pairCount"):
        raise ValueError("authority pair count differs from the recipe")
    result: dict[HeightChunkId, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    structural_bytes = baseline_bytes = 0
    for row in rows:
        chunk = _chunk(row.get("key"), "authority")
        if chunk in result or chunk.lod != -2:
            raise ValueError("authority tile inventory is duplicate or has the wrong LOD")
        structural_path = _safe_child(path.parent, row.get("path"), "authority tile")
        baseline_path = _safe_child(path.parent, row.get("baselinePath"), "baseline tile")
        structural_payload = structural_path.read_bytes()
        baseline_payload = baseline_path.read_bytes()
        if (
            len(structural_payload) != row.get("bytes")
            or _sha256_bytes(structural_payload) != row.get("sha256")
            or len(baseline_payload) != row.get("baselineBytes")
            or _sha256_bytes(baseline_payload) != row.get("baselineSha256")
        ):
            raise ValueError(f"authority pair integrity mismatch: {chunk}")
        structural = decode_structural_tile(structural_payload)
        baseline = decode_baseline_tile(baseline_payload)
        if (
            not structural.valid.all()
            or not baseline.valid.all()
            or np.any(structural.unknown_bathymetry & ~structural.forbidden_morphology)
        ):
            raise ValueError(f"authority masks are unresolved or contradictory: {chunk}")
        result[chunk] = (
            np.asarray(baseline.height, dtype=np.float32),
            np.asarray(structural.unknown_bathymetry, dtype=bool),
            np.asarray(structural.forbidden_morphology, dtype=bool),
        )
        structural_bytes += len(structural_payload)
        baseline_bytes += len(baseline_payload)
    return result, {
        "manifestSha256": _sha256_file(path),
        "pairCount": len(result),
        "structuralBytes": structural_bytes,
        "baselineBytes": baseline_bytes,
        "campaignContentEvidenceSha256": campaign_evidence,
        "canonicalBaseManifestSha256": document["baselineAuthority"]["release"][
            "manifestSha256"
        ],
    }


def _read_height_artifact(
    root: Path,
    row: dict[str, Any],
    encode: EncodeConfig,
) -> tuple[HeightChunkId, np.ndarray, dict[str, Any], float, float]:
    chunk = _chunk(row.get("key"), "materialized height")
    path = _safe_child(root, row.get("path"), "materialized height")
    if path.stat().st_size != row.get("bytes") or _sha256_file(path) != row.get(
        "containerSha256"
    ):
        raise ValueError(f"materialized container integrity mismatch: {chunk}")
    meta, payload = read_chunk_v2(path)
    if (meta.layer, meta.lod, meta.cx, meta.cz, meta.enc, meta.count) != (
        "height",
        chunk.lod,
        chunk.cx,
        chunk.cz,
        1,
        0,
    ):
        raise ValueError(f"materialized height header mismatch: {chunk}")
    values = decode_quant16(encode, payload, meta.res, meta.qoffset, meta.qscale)
    if _array_sha256(values, "<f4") != row.get("decodedValuesSha256"):
        raise ValueError(f"materialized decoded hash mismatch: {chunk}")
    sidecar_path = _safe_child(root, row.get("sidecar"), "materialized sidecar")
    sidecar = json.loads(sidecar_path.read_bytes())
    if (
        sidecar.get("key") != _key(chunk)
        or sidecar.get("path") != row.get("path")
        or sidecar.get("containerSha256") != row.get("containerSha256")
        or sidecar.get("payloadSha256") != _sha256_bytes(payload)
        or sidecar.get("decodedValuesSha256") != row.get("decodedValuesSha256")
        or sidecar.get("qoffset") != meta.qoffset
        or sidecar.get("qscale") != meta.qscale
        or sidecar.get("dependencyMerkleRoot") != row.get("dependencyMerkleRoot")
    ):
        raise ValueError(f"materialized sidecar disagrees with transaction: {chunk}")
    dependencies = tuple(
        (_chunk(item.get("key"), "height dependency"), _digest(item.get("sha256"), "dependency"))
        for item in sidecar.get("dependencies", [])
    )
    if dependency_merkle_root(dependencies) != sidecar.get("dependencyMerkleRoot"):
        raise ValueError(f"materialized dependency root mismatch: {chunk}")
    return chunk, values, sidecar, meta.qoffset, meta.qscale


def _verify_fixed_reduction(
    *,
    chunk: HeightChunkId,
    values: np.ndarray,
    sidecar: dict[str, Any],
    qoffset: float,
    qscale: float,
    reduced_values: np.ndarray,
    required_qscale: float,
    encode: EncodeConfig,
    name: str,
) -> None:
    """Prove the artifact is the canonical quantized fixed-child reduction."""
    source_sha = _array_sha256(reduced_values, "<f8")
    if source_sha != sidecar.get("sourceValuesSha256"):
        raise ValueError(f"{name} source reduction hash mismatch: {chunk}")

    expected_payload, expected_qoffset, expected_qscale = encode_quant16_checked(
        encode,
        reduced_values,
        required_qscale,
        qoffset=None,
    )
    if qoffset != expected_qoffset or qscale != expected_qscale:
        raise ValueError(f"{name} is not the canonical quantized reduction: {chunk}")

    # Compression level is not part of the verifier's recipe contract. The
    # actual payload is hash-bound above; compare the canonical decoded codes.
    expected_values = decode_quant16(
        encode,
        expected_payload,
        reduced_values.shape[0],
        expected_qoffset,
        expected_qscale,
    )
    if not np.array_equal(values, expected_values):
        raise ValueError(f"{name} decoded reduction mismatch: {chunk}")


def _verify_ownership(
    root: Path,
    fine: dict[HeightChunkId, tuple[np.ndarray, dict[str, Any]]],
    authority: dict[HeightChunkId, tuple[np.ndarray, np.ndarray, np.ndarray]],
    closure_sha256: str,
    rerasterize: Callable[[HeightChunkId], tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    owned = abstained = restored = 0
    for chunk, (values, sidecar) in fine.items():
        ownership = sidecar.get("ownership")
        if (
            not isinstance(ownership, dict)
            or ownership.get("closureSha256") != closure_sha256
        ):
            raise ValueError(f"fine artifact lacks independently checkable ownership: {chunk}")
        ownership_path = _safe_child(root, ownership.get("path"), "fine ownership")
        payload = ownership_path.read_bytes()
        if (
            ownership.get("format") != "TOM1/1"
            or len(payload) != ownership.get("bytes")
            or _sha256_bytes(payload) != ownership.get("containerSha256")
        ):
            raise ValueError(f"fine ownership container integrity mismatch: {chunk}")
        authority_bool, abstained_bool = decode_fine_ownership(payload)
        expected_authority, expected_abstained, baseline = rerasterize(chunk)
        if (
            authority_bool.shape != values.shape
            or abstained_bool.shape != values.shape
            or baseline.shape != values.shape
            or not np.array_equal(authority_bool, expected_authority)
            or not np.array_equal(abstained_bool, expected_abstained)
            or _array_sha256(authority_bool, "u1")
            != ownership.get("authoritySha256")
            or _array_sha256(abstained_bool, "u1")
            != ownership.get("abstentionSha256")
            or int(np.count_nonzero(authority_bool))
            != ownership.get("authoritySamples")
            or int(np.count_nonzero(abstained_bool))
            != ownership.get("abstainedSamples")
        ):
            raise ValueError(f"fine ownership artifacts have an invalid contract: {chunk}")
        baseline_codes, baseline_qoffset, baseline_qscale = quantize16_checked(
            baseline,
            sidecar["qscale"],
            qoffset=sidecar["qoffset"],
        )
        quantized_baseline = (
            baseline_codes.astype(np.float32) * baseline_qscale + baseline_qoffset
        )
        restore_mask = ~authority_bool
        if not np.array_equal(
            values[restore_mask], quantized_baseline[restore_mask]
        ):
            raise ValueError(
                f"dry/abstained samples did not restore the quantized baseline: {chunk}"
            )
        if not np.array_equal(
            values[abstained_bool], quantized_baseline[abstained_bool]
        ):
            raise ValueError(f"abstained samples differ from quantized baseline: {chunk}")
        owned += int(np.count_nonzero(authority_bool))
        abstained += int(np.count_nonzero(abstained_bool))
        restored += int(np.count_nonzero(restore_mask))
    return {"fineChunks": len(fine), "authoritySamples": owned, "abstainedSamples": abstained, "restoredSamples": restored}


def _verify_materializer(
    path: Path,
    expected_sha256: str,
    recipe_sha256: str,
    authority_sha256: str,
    base_sha256: str,
    encode: EncodeConfig,
    authority: dict[HeightChunkId, tuple[np.ndarray, np.ndarray, np.ndarray]],
    campaign_evidence: dict[str, str],
    rerasterize: Callable[[HeightChunkId], tuple[np.ndarray, np.ndarray, np.ndarray]],
    scientific_closure_sha256: str,
) -> tuple[dict[HeightChunkId, tuple[np.ndarray, dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    if _sha256_file(path) != _digest(expected_sha256, "materializer sha256"):
        raise ValueError("materializer transaction differs from the frozen input")
    root = path.parent
    document = json.loads(path.read_bytes())
    if (
        document.get("format") != 1
        or document.get("materializerVersion") != MATERIALIZER_VERSION
        or document.get("recipeSha256") != recipe_sha256
        or document.get("authorityManifestSha256") != authority_sha256
        or document.get("canonicalBaseManifestSha256") != base_sha256
    ):
        raise ValueError("materializer transaction identity mismatch")
    closure = document.get("ownershipClosure")
    if (
        not isinstance(closure, dict)
        or closure.get("version") != OWNERSHIP_CLOSURE_VERSION
        or closure.get("authorityManifestSha256") != authority_sha256
        or closure.get("campaignContentSha256")
        != campaign_evidence.get("contentSha256")
        or closure.get("campaignQualificationSha256")
        != campaign_evidence.get("qualificationSha256")
        or closure.get("scientificClosureSha256") != scientific_closure_sha256
        or closure.get("rasterizer")
        != {
            "id": OWNERSHIP_RASTERIZER_ID,
            "sourceSha256": ownership_rasterizer_source_sha256(),
        }
    ):
        raise ValueError(
            "materializer lacks the independently rerasterized shared closure"
        )
    closure_sha = _sha256_bytes(_canonical_json_bytes(closure))
    collections: dict[str, dict[HeightChunkId, tuple[np.ndarray, dict[str, Any]]]] = {}
    wires: dict[str, dict[HeightChunkId, tuple[float, float]]] = {}
    for name in ("fine", "parents", "correctedLod0"):
        rows = document.get(name)
        if not isinstance(rows, list):
            raise ValueError(f"materializer transaction lacks {name}")
        decoded: dict[HeightChunkId, tuple[np.ndarray, dict[str, Any]]] = {}
        wire: dict[HeightChunkId, tuple[float, float]] = {}
        for row in rows:
            chunk, values, sidecar, qoffset, qscale = (
                _read_height_artifact(root, row, encode)
            )
            if chunk in decoded:
                raise ValueError(f"duplicate materialized {name} key: {chunk}")
            decoded[chunk] = (values, sidecar)
            wire[chunk] = (qoffset, qscale)
        collections[name] = decoded
        wires[name] = wire
    fine = collections["fine"]
    parents = collections["parents"]
    corrected = collections["correctedLod0"]

    for chunk, (values, sidecar) in parents.items():
        reduced = reduce_decoded_children(chunk, lambda child: fine[child][0], core_res=values.shape[0] - 1)
        qoffset, qscale = wires["parents"][chunk]
        _verify_fixed_reduction(
            chunk=chunk,
            values=values,
            sidecar=sidecar,
            qoffset=qoffset,
            qscale=qscale,
            reduced_values=reduced.values,
            required_qscale=0.005,
            encode=encode,
            name="LOD-1",
        )
    for chunk, (values, sidecar) in corrected.items():
        reduced = reduce_decoded_children(chunk, lambda child: parents[child][0], core_res=values.shape[0] - 1)
        qoffset, qscale = wires["correctedLod0"][chunk]
        _verify_fixed_reduction(
            chunk=chunk,
            values=values,
            sidecar=sidecar,
            qoffset=qoffset,
            qscale=qscale,
            reduced_values=reduced.values,
            required_qscale=encode.height_qscale_for(0),
            encode=encode,
            name="LOD0 candidate",
        )

    published = tuple(_chunk(value, "published overlay") for value in document.get("publishedOverlay", []))
    expected_published = tuple(sorted((*children_of(published[-1]), published[-1]))) if published else ()
    if len(published) != 17 or set(published) != set(expected_published):
        raise ValueError("materializer published set is not one complete 16+1 overlay")
    ownership = _verify_ownership(root, fine, authority, closure_sha, rerasterize)
    ownership["closureSha256"] = closure_sha
    return fine, {
        "transactionSha256": _sha256_file(path),
        "fineChunks": len(fine),
        "parentChunks": len(parents),
        "correctedLod0Chunks": len(corrected),
        "published": [_key(chunk) for chunk in published],
        "reducerVersion": BOX_MEAN_REDUCER_VERSION,
    }, ownership


def _read_index(path: Path) -> dict[HeightChunkId, tuple[int, int]]:
    blob = path.read_bytes()
    if len(blob) % _INDEX_V1.size:
        raise ValueError("corrected base index is truncated")
    records: dict[HeightChunkId, tuple[int, int]] = {}
    for offset in range(0, len(blob), _INDEX_V1.size):
        lod, cx, cz, size, hash64 = _INDEX_V1.unpack_from(blob, offset)
        chunk = HeightChunkId(lod, cx, cz)
        if chunk in records:
            raise ValueError("corrected base height index contains a duplicate key")
        records[chunk] = (size, hash64)
    return records


def _content_path(root: Path, layer: str, chunk: HeightChunkId, hash64: int) -> Path:
    hash8 = ((hash64 >> 32) & 0xFFFFFFFF).to_bytes(4, "big").hex()
    return root / "c" / layer / str(chunk.lod) / f"{chunk.cx}_{chunk.cz}.{hash8}.bin"


def _corrected_mask(root: Path, entry: Any, name: str) -> np.ndarray:
    path = _checked_file(root, entry, name)
    blob = path.read_bytes()
    if (
        entry.get("format") != _CORRECTED_MASK_FORMAT
        or len(blob) < 12
        or blob[:4] != _CORRECTED_MASK_MAGIC
    ):
        raise ValueError(f"{name} has an invalid masked-splice format")
    rows, cols = struct.unpack_from("<II", blob, 4)
    bit_count = rows * cols
    packed_bytes = (bit_count + 7) // 8
    if (
        rows != entry.get("rows")
        or cols != entry.get("cols")
        or len(blob) != 12 + packed_bytes
    ):
        raise ValueError(f"{name} dimensions differ from its inventory")
    bits = np.unpackbits(np.frombuffer(blob, dtype=np.uint8, offset=12), bitorder="big")
    if np.any(bits[bit_count:]):
        raise ValueError(f"{name} has nonzero padding bits")
    mask = bits[:bit_count].reshape(rows, cols).astype(np.bool_)
    if int(np.count_nonzero(mask)) != entry.get("sampleCount"):
        raise ValueError(f"{name} sample count differs from its inventory")
    return mask


def _corrected_row_runs(mask: np.ndarray) -> list[list[int]]:
    runs: list[list[int]] = []
    for row in range(mask.shape[0]):
        padded = np.pad(mask[row].astype(np.int8), (1, 1))
        edges = np.flatnonzero(np.diff(padded))
        for index in range(0, len(edges), 2):
            runs.append([row, int(edges[index]), int(edges[index + 1])])
    return runs


def _verify_corrected_windows(
    root: Path, entry: Any, mask: np.ndarray, name: str
) -> None:
    path = _checked_file(root, entry, name)
    try:
        runs = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not canonical row-run JSON") from error
    expected = _corrected_row_runs(mask)
    if (
        entry.get("format") != _CORRECTED_WINDOW_FORMAT
        or entry.get("runCount") != len(expected)
        or runs != expected
        or path.read_bytes()
        != (json.dumps(expected, separators=(",", ":")) + "\n").encode()
    ):
        raise ValueError(f"{name} does not reconstruct the exact affected mask")


def _corrected_selected_sha(values: np.ndarray, select: np.ndarray) -> str:
    data = np.asarray(values, dtype="<f4")
    mask = np.asarray(select, dtype=bool)
    digest = hashlib.sha256()
    digest.update(struct.pack("<IIQ", data.shape[0], data.shape[1], int(mask.sum())))
    for row in range(data.shape[0]):
        digest.update(
            np.ascontiguousarray(data[row, mask[row]], dtype="<f4").tobytes()
        )
    return digest.hexdigest()


def _content_path_from_sha(
    root: Path, layer: str, chunk: HeightChunkId, artifact_sha256: str
) -> Path:
    digest = _digest(artifact_sha256, f"{layer} inherited artifact sha256")
    return root / "c" / layer / str(chunk.lod) / f"{chunk.cx}_{chunk.cz}.{digest[:8]}.bin"


def _verify_corrected_base(
    manifest_path: Path,
    content_root: Path,
    verification_path: Path,
    resolved_path: Path,
    expected_resolved_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    manifest_blob = manifest_path.read_bytes()
    manifest_sha = _sha256_bytes(manifest_blob)
    if manifest_sha != expected_manifest_sha256:
        raise ValueError("corrected base manifest differs from the structural recipe")
    manifest = json.loads(manifest_blob)
    if manifest.get("format") != 1:
        raise ValueError("structural corrected base must retain format 1")
    verification = json.loads(verification_path.read_bytes())
    if verification.get("passed") is not True or verification.get("manifestSha256") != manifest_sha:
        raise ValueError("corrected base verification is absent or names another manifest")
    if _sha256_file(resolved_path) != expected_resolved_sha256:
        raise ValueError("resolved corrected-base plan differs from the frozen input")
    resolved = json.loads(resolved_path.read_bytes())
    if (
        resolved.get("format") != _CORRECTED_BASE_PLAN_FORMAT
        or resolved.get("reducerVersion") != BOX_MEAN_REDUCER_VERSION
        or resolved.get("ancestorSpliceVersion") != _CORRECTED_ANCESTOR_SPLICE
        or resolved.get("qoffsetPolicy") != _CORRECTED_QOFFSET_POLICY
        or not isinstance(resolved.get("artifacts"), list)
    ):
        raise ValueError("resolved corrected-base plan has the wrong contract")
    height_meta = manifest.get("layers", {}).get("height", {})
    index_path = _safe_child(manifest_path.parent, height_meta.get("index"), "corrected height index")
    if _sha256_file(index_path) != height_meta.get("indexSha256"):
        raise ValueError("corrected height index hash mismatch")
    index = _read_index(index_path)
    rows_by_chunk: dict[HeightChunkId, dict[str, Any]] = {}
    corrected_masks: dict[HeightChunkId, str] = {}
    for entry in resolved.get("correctedMaskSha256", ()):
        chunk = _chunk(entry.get("chunk"), "corrected LOD0 mask")
        if chunk in corrected_masks:
            raise ValueError("corrected LOD0 mask inventory contains a duplicate key")
        corrected_masks[chunk] = _digest(
            entry.get("sha256"), "corrected LOD0 mask sha256"
        )
    if set(corrected_masks) != {
        _chunk(value, "corrected LOD0") for value in resolved.get("correctedLod0", ())
    }:
        raise ValueError("corrected LOD0 mask inventory differs from corrected cores")
    for row in resolved["artifacts"]:
        chunk = _chunk(row.get("chunk"), "corrected base artifact")
        if chunk in rows_by_chunk:
            raise ValueError("resolved corrected-base artifacts contain a duplicate key")
        rows_by_chunk[chunk] = row

    wire_encode = EncodeConfig(manifest["codec"], 0.01, 0.01, 19, 1)

    def decoded_row(chunk: HeightChunkId) -> tuple[Any, np.ndarray]:
        row = rows_by_chunk[chunk]
        size, hash64 = index[chunk]
        content = _content_path(content_root, "height", chunk, hash64)
        if content.stat().st_size != size:
            raise ValueError(f"corrected base content size mismatch: {chunk}")
        meta, payload = read_chunk(content)
        return meta, decode_quant16(
            wire_encode, payload, meta.res, meta.qoffset, meta.qscale
        )

    checked = masked_samples = outside_samples = 0
    for chunk, row in sorted(rows_by_chunk.items()):
        chunk = _chunk(row.get("chunk"), "corrected base artifact")
        if chunk not in index:
            raise ValueError(f"corrected base index omits resolved artifact: {chunk}")
        size, hash64 = index[chunk]
        content = _content_path(content_root, "height", chunk, hash64)
        expected_sha = _digest(row.get("artifact_sha256"), "corrected artifact sha256")
        if content.stat().st_size != size or _sha256_file(content) != expected_sha:
            raise ValueError(f"corrected base content differs from resolved artifact: {chunk}")
        meta, payload = read_chunk(content)
        if (meta.layer, meta.lod, meta.cx, meta.cz, meta.enc) != (
            "height", chunk.lod, chunk.cx, chunk.cz, 1
        ) or _sha256_bytes(payload) != row.get("payload_sha256"):
            raise ValueError(f"corrected base header/payload mismatch: {chunk}")
        decoded = decode_quant16(
            wire_encode, payload, meta.res, meta.qoffset, meta.qscale
        )
        if (
            _array_sha256(decoded, "<f4") != row.get("decoded_values_sha256")
            or _array_sha256(decoded[:-1, :-1], "<f4")
            != row.get("decoded_core_sha256")
        ):
            raise ValueError(f"corrected base decoded identity mismatch: {chunk}")
        if (
            row.get("maskedSpliceVersion") != _CORRECTED_ANCESTOR_SPLICE
            or row.get("quantizerDomain") != _CORRECTED_QUANTIZER_DOMAIN
        ):
            raise ValueError(f"corrected base masked-splice policy mismatch: {chunk}")
        affected = _corrected_mask(
            resolved_path.parent, row.get("affectedMask"), f"affected mask {chunk}"
        )
        core_mask = _corrected_mask(
            resolved_path.parent,
            row.get("affectedCoreMask"),
            f"affected core mask {chunk}",
        )
        if (
            affected.shape != decoded.shape
            or core_mask.shape != decoded[:-1, :-1].shape
            or not np.array_equal(core_mask, affected[:-1, :-1])
        ):
            raise ValueError(f"corrected base affected masks disagree: {chunk}")
        _verify_corrected_windows(
            resolved_path.parent,
            row.get("affectedWindows"),
            affected,
            f"affected windows {chunk}",
        )
        if chunk.lod == 0 and core_mask.any():
            if corrected_masks.get(chunk) != row["affectedCoreMask"].get("sha256"):
                raise ValueError(f"corrected LOD0 input mask identity mismatch: {chunk}")

        inherited_dependencies = [
            item
            for item in row.get("dependencies", ())
            if item.get("role") == "inherited-independent-dtm-core"
        ]
        if len(inherited_dependencies) != 1:
            raise ValueError(f"corrected artifact lacks one inherited identity: {chunk}")
        inherited_dependency = inherited_dependencies[0]
        if _chunk(inherited_dependency.get("chunk"), "inherited dependency") != chunk:
            raise ValueError(f"corrected artifact inherited identity names another key: {chunk}")
        inherited_sha = _digest(
            inherited_dependency.get("artifactSha256"),
            "corrected inherited artifact sha256",
        )
        inherited_path = _content_path_from_sha(
            content_root, "height", chunk, inherited_sha
        )
        if _sha256_file(inherited_path) != inherited_sha:
            raise ValueError(f"corrected inherited artifact integrity mismatch: {chunk}")
        inherited_meta, inherited_payload = read_chunk(inherited_path)
        inherited = decode_quant16(
            wire_encode,
            inherited_payload,
            inherited_meta.res,
            inherited_meta.qoffset,
            inherited_meta.qscale,
        )
        if (
            (inherited_meta.layer, inherited_meta.lod, inherited_meta.cx, inherited_meta.cz)
            != ("height", chunk.lod, chunk.cx, chunk.cz)
            or inherited.shape != decoded.shape
            or _array_sha256(inherited[:-1, :-1], "<f4")
            != row.get("inheritedDecodedCoreSha256")
            or inherited_dependency.get("decodedCoreSha256")
            != row.get("inheritedDecodedCoreSha256")
        ):
            raise ValueError(f"corrected inherited decoded identity mismatch: {chunk}")

        outside = ~affected
        proof = row.get("outsideProof")
        inherited_outside_sha = _corrected_selected_sha(inherited, outside)
        output_outside_sha = _corrected_selected_sha(decoded, outside)
        if (
            not isinstance(proof, dict)
            or proof.get("version") != _CORRECTED_OUTSIDE_PROOF
            or proof.get("sampleCount") != int(np.count_nonzero(outside))
            or proof.get("inheritedDecodedSha256") != inherited_outside_sha
            or proof.get("outputDecodedSha256") != output_outside_sha
            or proof.get("bitExact") is not True
            or inherited_outside_sha != output_outside_sha
            or not np.array_equal(decoded[outside], inherited[outside])
        ):
            raise ValueError(f"corrected outside-mask proof mismatch: {chunk}")
        if meta.qoffset != row.get("qoffset") or meta.qscale != row.get("qscale"):
            raise ValueError(f"corrected quantizer inventory mismatch: {chunk}")
        if row.get("inheritedQuantizerPreserved") is True:
            output_codes = np.rint(
                (decoded.astype(np.float64) - meta.qoffset) / meta.qscale
            ).astype(np.uint16)
            inherited_codes = np.rint(
                (inherited.astype(np.float64) - inherited_meta.qoffset)
                / inherited_meta.qscale
            ).astype(np.uint16)
            if (
                meta.qoffset != inherited_meta.qoffset
                or meta.qscale != inherited_meta.qscale
                or not np.array_equal(output_codes[outside], inherited_codes[outside])
            ):
                raise ValueError(f"preserved corrected quantizer changed outside codes: {chunk}")
        elif row.get("inheritedQuantizerPreserved") is not False:
            raise ValueError(f"corrected quantizer disposition is absent: {chunk}")

        if row.get("role") == "masked-apron-only-replacement":
            if (
                row.get("core_codes_preserved") is not True
                or core_mask.any()
                or row.get("decoded_core_sha256")
                != row.get("inheritedDecodedCoreSha256")
            ):
                raise ValueError(
                    f"apron-only replacement changed inherited quantizer/core: {chunk}"
                )
        masked_samples += int(np.count_nonzero(affected))
        outside_samples += int(np.count_nonzero(outside))
        checked += 1

    # Reconstruct the full payload mask from core ownership and affected neighbor
    # boundaries, then independently rederive every coarse masked child window.
    for chunk, row in sorted(rows_by_chunk.items()):
        meta, decoded = decoded_row(chunk)
        affected = _corrected_mask(
            resolved_path.parent, row["affectedMask"], f"affected mask {chunk}"
        )
        core_mask = _corrected_mask(
            resolved_path.parent,
            row["affectedCoreMask"],
            f"affected core mask {chunk}",
        )
        expected_payload = np.zeros_like(affected)
        expected_payload[:-1, :-1] = core_mask
        east = HeightChunkId(chunk.lod, chunk.cx + 1, chunk.cz)
        south = HeightChunkId(chunk.lod, chunk.cx, chunk.cz + 1)
        southeast = HeightChunkId(chunk.lod, chunk.cx + 1, chunk.cz + 1)
        if east in rows_by_chunk:
            east_mask = _corrected_mask(
                resolved_path.parent,
                rows_by_chunk[east]["affectedCoreMask"],
                f"affected core mask {east}",
            )
            expected_payload[:-1, -1] = east_mask[:, 0]
            _, east_values = decoded_row(east)
            selected = expected_payload[:-1, -1]
            expected_codes = np.rint(
                (east_values[:-1, 0][selected].astype(np.float64) - meta.qoffset)
                / meta.qscale
            ).astype(np.uint16)
            actual_codes = np.rint(
                (decoded[:-1, -1][selected].astype(np.float64) - meta.qoffset)
                / meta.qscale
            ).astype(np.uint16)
            if not np.array_equal(actual_codes, expected_codes):
                raise ValueError(f"corrected east apron differs from affected neighbor: {chunk}")
        if south in rows_by_chunk:
            south_mask = _corrected_mask(
                resolved_path.parent,
                rows_by_chunk[south]["affectedCoreMask"],
                f"affected core mask {south}",
            )
            expected_payload[-1, :-1] = south_mask[0, :]
            _, south_values = decoded_row(south)
            selected = expected_payload[-1, :-1]
            expected_codes = np.rint(
                (south_values[0, :-1][selected].astype(np.float64) - meta.qoffset)
                / meta.qscale
            ).astype(np.uint16)
            actual_codes = np.rint(
                (decoded[-1, :-1][selected].astype(np.float64) - meta.qoffset)
                / meta.qscale
            ).astype(np.uint16)
            if not np.array_equal(actual_codes, expected_codes):
                raise ValueError(f"corrected south apron differs from affected neighbor: {chunk}")
        if southeast in rows_by_chunk:
            southeast_mask = _corrected_mask(
                resolved_path.parent,
                rows_by_chunk[southeast]["affectedCoreMask"],
                f"affected core mask {southeast}",
            )
            expected_payload[-1, -1] = southeast_mask[0, 0]
            if expected_payload[-1, -1]:
                _, southeast_values = decoded_row(southeast)
                expected_code = int(
                    np.rint(
                        (float(southeast_values[0, 0]) - meta.qoffset) / meta.qscale
                    )
                )
                actual_code = int(
                    np.rint((float(decoded[-1, -1]) - meta.qoffset) / meta.qscale)
                )
                if actual_code != expected_code:
                    raise ValueError(f"corrected southeast apron differs: {chunk}")
        if not np.array_equal(affected, expected_payload):
            raise ValueError(f"corrected affected payload mask is not neighbor-derived: {chunk}")

        child_dependencies = [
            item
            for item in row.get("dependencies", ())
            if item.get("role") == "decoded-child-masked-window"
        ]
        if chunk.lod == 0 or not core_mask.any():
            if child_dependencies:
                raise ValueError(f"corrected artifact has impossible child dependencies: {chunk}")
            continue
        if not child_dependencies:
            raise ValueError(f"affected corrected ancestor lacks child windows: {chunk}")
        core_res = meta.res - 1
        quarter = core_res // 4
        expected_core_mask = np.zeros((core_res, core_res), dtype=bool)
        parent_codes = np.rint(
            (decoded[:-1, :-1].astype(np.float64) - meta.qoffset) / meta.qscale
        ).astype(np.uint16)
        for dependency in child_dependencies:
            child = _chunk(dependency.get("chunk"), "corrected child dependency")
            if child not in rows_by_chunk or parent_of(child) != chunk:
                raise ValueError(f"corrected child dependency is outside its parent: {chunk}")
            child_row = rows_by_chunk[child]
            if (
                dependency.get("artifactSha256") != child_row.get("artifact_sha256")
                or dependency.get("decodedCoreSha256")
                != child_row.get("decoded_core_sha256")
            ):
                raise ValueError(f"corrected child dependency identity mismatch: {chunk}")
            _, child_values = decoded_row(child)
            child_mask = _corrected_mask(
                resolved_path.parent,
                child_row["affectedCoreMask"],
                f"affected core mask {child}",
            )
            reduced_mask = child_mask.reshape(quarter, 4, quarter, 4).any(
                axis=(1, 3)
            )
            reduced_values = box_mean_fixed(child_values[:-1, :-1], factor=4)
            dx = child.cx - chunk.cx * 4
            dz = child.cz - chunk.cz * 4
            if dx not in range(4) or dz not in range(4):
                raise ValueError(f"corrected child has an invalid quadrant: {child}")
            rows = slice(dz * quarter, (dz + 1) * quarter)
            cols = slice(dx * quarter, (dx + 1) * quarter)
            expected_core_mask[rows, cols] |= reduced_mask
            expected_codes = np.rint(
                (reduced_values[reduced_mask] - meta.qoffset) / meta.qscale
            ).astype(np.uint16)
            if not np.array_equal(parent_codes[rows, cols][reduced_mask], expected_codes):
                raise ValueError(f"corrected ancestor is not a masked child reduction: {chunk}")
        if not np.array_equal(core_mask, expected_core_mask):
            raise ValueError(f"corrected ancestor mask is not the child any-window: {chunk}")
    return {
        "manifestSha256": manifest_sha,
        "verificationSha256": _sha256_file(verification_path),
        "resolvedPlanSha256": _sha256_file(resolved_path),
        "resolvedArtifacts": checked,
        "maskedSamples": masked_samples,
        "outsideSamples": outside_samples,
        "format": 1,
    }


def _verify_water(
    path: Path,
    expected_sha256: str,
    recipe_sha256: str,
    build_root: Path,
    base_manifest_path: Path,
    content_root: Path,
    encode: EncodeConfig,
    source_manifest_sha256: str,
) -> tuple[set[tuple[str, int, int, int]], set[tuple[str, int, int, int]], dict[str, Any]]:
    if _sha256_file(path) != _digest(expected_sha256, "water transaction sha256"):
        raise ValueError("water transaction differs from the frozen input")
    document = json.loads(path.read_bytes())
    if (
        document.get("format") != 1
        or document.get("transactionVersion")
        != "corrected-structural-water-transaction/1"
        or document.get("sourceManifestSha256") != source_manifest_sha256
        or not isinstance(document.get("artifacts"), list)
    ):
        raise ValueError("corrected-water transaction contract mismatch")
    base_manifest = json.loads(base_manifest_path.read_bytes())
    water_core_res = int(base_manifest["chunkMeters"]) // 2
    base_water = base_manifest.get("layers", {}).get("water")
    base_records: dict[HeightChunkId, tuple[int, int]] = {}
    if isinstance(base_water, dict):
        base_index = _safe_child(base_manifest_path.parent, base_water.get("index"), "base water index")
        if _sha256_file(base_index) != base_water.get("indexSha256"):
            raise ValueError("corrected base water index hash mismatch")
        base_records = _read_index(base_index)
    halos = document.get("halo")
    if not isinstance(halos, list):
        raise ValueError("corrected-water transaction lacks halo inventories")
    for halo in halos:
        halo_chunk = _chunk(halo.get("key"), "water halo")
        dependencies = halo.get("dependencies")
        if (
            halo_chunk.lod != 0
            or not isinstance(dependencies, list)
            or len(dependencies) != 9
            or len({tuple(item.get("key", ())) for item in dependencies}) != 9
        ):
            raise ValueError("corrected-water halo dependency closure mismatch")
        digest = hashlib.sha256(b"laas.structural-water.halo-dependencies.v1\0")
        for dependency in dependencies:
            key = _chunk(dependency.get("key"), "water halo dependency")
            source_kind = dependency.get("source")
            artifact_sha = _digest(
                dependency.get("artifactSha256"), "water halo artifact sha256"
            )
            if source_kind not in {"inherited", "corrected", "explicit_absent"}:
                raise ValueError("corrected-water halo has an invalid source role")
            encoded_source = source_kind.encode()
            digest.update(
                struct.pack("<BiiB", key.lod, key.cx, key.cz, len(encoded_source))
            )
            digest.update(encoded_source)
            digest.update(bytes.fromhex(artifact_sha))
        if digest.hexdigest() != halo.get("dependencySha256"):
            raise ValueError("corrected-water halo dependency root mismatch")

    present: set[tuple[str, int, int, int]] = set()
    tombstones: set[tuple[str, int, int, int]] = set()
    decoded: dict[HeightChunkId, np.ndarray] = {}
    artifact_id: dict[HeightChunkId, str] = {}
    rows_by_chunk: dict[HeightChunkId, dict[str, Any]] = {}
    for row in document["artifacts"]:
        chunk = _chunk(row.get("key"), "corrected water")
        key = ("water", chunk.lod, chunk.cx, chunk.cz)
        if key in present or key in tombstones or chunk.lod not in (0, 1):
            raise ValueError("corrected-water keys are duplicate or outside LOD0/1")
        source_path = _checked_file(path.parent, row.get("sourceValues"), "water source values")
        source_values = np.load(source_path, allow_pickle=False)
        if source_values.dtype != np.dtype("<f8") or source_values.shape != (
            water_core_res + 1,
            water_core_res + 1,
        ) or np.isinf(source_values).any():
            raise ValueError(f"water source values have an invalid contract: {chunk}")
        source_sha = _array_sha256(source_values, "<f8")
        if source_sha != row.get("sourceValuesSha256"):
            raise ValueError(f"water source values hash mismatch: {chunk}")
        evidence_path = _checked_file(path.parent, row.get("evidence"), "water correction evidence")
        evidence_document = json.loads(evidence_path.read_bytes())
        evidence_identity = evidence_document.get("semanticSha256")
        if evidence_identity is None:
            evidence_identity = _sha256_bytes(
                json.dumps(evidence_document, sort_keys=True, separators=(",", ":")).encode()
            )
        if evidence_identity != row.get("evidenceSha256"):
            raise ValueError(f"water correction evidence identity mismatch: {chunk}")
        if row.get("disposition") == "remove-inherited":
            tombstone = _digest(row.get("artifactSha256"), "water tombstone sha256")
            source_sha = _digest(row.get("sourceValuesSha256"), "water source values sha256")
            evidence_sha = _digest(row.get("evidenceSha256"), "water evidence sha256")
            dependency_sha = _digest(row.get("dependencySha256"), "water dependency sha256")
            identity = hashlib.sha256()
            identity.update(b"laas.structural-water.absent.v1\0")
            identity.update(struct.pack("<Bii", chunk.lod, chunk.cx, chunk.cz))
            identity.update(bytes.fromhex(source_sha))
            identity.update(bytes.fromhex(evidence_sha))
            identity.update(bytes.fromhex(dependency_sha))
            tombstone_path = _safe_child(path.parent, row.get("path"), "water tombstone")
            tombstone_document = json.loads(tombstone_path.read_bytes())
            if (
                tombstone_path.stat().st_size != row.get("bytes")
                or _sha256_file(tombstone_path) != row.get("fileSha256")
                or tombstone != identity.hexdigest()
                or tombstone_document.get("reason") != "all_dry"
                or tombstone_document.get("tombstoneSha256") != tombstone
            ):
                raise ValueError(f"corrected-water tombstone identity mismatch: {chunk}")
            if np.isfinite(source_values).any():
                raise ValueError(f"absent water has wet source samples: {chunk}")
            artifact_id[chunk] = tombstone
            rows_by_chunk[chunk] = row
            tombstones.add(key)
            continue
        if row.get("disposition") != "replacement":
            raise ValueError("corrected-water artifact has an invalid state")
        artifact_path = _safe_child(path.parent, row.get("path"), "corrected water chunk")
        artifact_sha = _digest(row.get("artifactSha256"), "water artifact sha256")
        if (
            artifact_path.stat().st_size != row.get("bytes")
            or _sha256_file(artifact_path) != artifact_sha
            or row.get("fileSha256") != artifact_sha
        ):
            raise ValueError(f"corrected-water container integrity mismatch: {chunk}")
        version, meta, payload = read_chunk_any(artifact_path)
        if version != 1 or (meta.layer, meta.lod, meta.cx, meta.cz, meta.enc) != (
            "water", chunk.lod, chunk.cx, chunk.cz, 1
        ):
            raise ValueError(f"corrected-water header mismatch: {chunk}")
        values = decode_quant16(encode, payload, meta.res, meta.qoffset, meta.qscale)
        codes = np.rint((values.astype(np.float64) - meta.qoffset) / meta.qscale).astype(np.uint16)
        values[codes == 0] = np.nan
        if _array_sha256(values, "<f4") != row.get("decodedValuesSha256"):
            raise ValueError(f"corrected-water decoded hash mismatch: {chunk}")
        wet = np.isfinite(source_values)
        if not np.array_equal(np.isfinite(values), wet):
            raise ValueError(f"corrected-water wet/dry mask changed in packing: {chunk}")
        peak = np.float32(np.max(np.abs(values[wet]), initial=0.0))
        limit = meta.qscale * 0.5 + 2.0 * abs(float(np.spacing(peak)))
        error = float(np.max(np.abs(values[wet].astype(np.float64) - source_values[wet]), initial=0.0))
        if error > limit:
            raise ValueError(f"corrected-water roundtrip bound mismatch: {chunk}")
        _digest(row.get("sourceValuesSha256"), "water source values sha256")
        _digest(row.get("evidenceSha256"), "water evidence sha256")
        dependencies = row.get("dependencies")
        if not isinstance(dependencies, list) or not dependencies:
            raise ValueError(f"corrected-water dependencies are absent: {chunk}")
        for dependency in dependencies:
            _chunk(dependency.get("key"), "water dependency")
            _digest(dependency.get("artifactSha256"), "water dependency sha256")
        decoded[chunk] = values
        artifact_id[chunk] = artifact_sha
        rows_by_chunk[chunk] = row
        present.add(key)

    base_manifest_sha = source_manifest_sha256

    def inherited(dependency: HeightChunkId) -> tuple[np.ndarray | None, str]:
        record = base_records.get(dependency)
        if record is None:
            return None, inherited_absent_water_sha256(dependency, base_manifest_sha)
        size, hash64 = record
        source = _content_path(content_root, "water", dependency, hash64)
        if source.stat().st_size != size:
            raise ValueError(f"inherited water size mismatch: {dependency}")
        full_sha = _sha256_file(source)
        meta, payload = read_chunk(source)
        values = decode_quant16(encode, payload, meta.res, meta.qoffset, meta.qscale)
        codes = np.rint((values.astype(np.float64) - meta.qoffset) / meta.qscale).astype(np.uint16)
        values[codes == 0] = np.nan
        return values, full_sha

    def loaded(dependency: HeightChunkId) -> tuple[np.ndarray | None, str]:
        if dependency in rows_by_chunk:
            return decoded.get(dependency), artifact_id[dependency]
        return inherited(dependency)

    for chunk, row in rows_by_chunk.items():
        dependencies = tuple(
            (_chunk(item["key"], "water dependency"), item["artifactSha256"])
            for item in row["dependencies"]
        )
        ordered = tuple(sorted(dependencies, key=lambda item: (item[0].lod, item[0].cz, item[0].cx)))
        dependency_digest = hashlib.sha256(b"laas.structural-water.dependencies.v1\0")
        for dependency, declared_sha in ordered:
            _, actual_sha = (
                inherited(dependency)
                if chunk.lod == 0 and dependency == chunk
                else loaded(dependency)
            )
            if actual_sha != declared_sha:
                raise ValueError(f"water dependency identity mismatch: {chunk} <- {dependency}")
            dependency_digest.update(struct.pack("<Bii", dependency.lod, dependency.cx, dependency.cz))
            dependency_digest.update(bytes.fromhex(declared_sha))
        if dependency_digest.hexdigest() != row.get("dependencySha256"):
            raise ValueError(f"water dependency root mismatch: {chunk}")
        if chunk.lod == 0:
            if len(dependencies) != 1 or dependencies[0][0] != chunk:
                raise ValueError(f"corrected LOD0 water lacks its inherited identity: {chunk}")
            continue
        expected_dependencies = tuple(
            HeightChunkId(0, chunk.cx * 4 + dx, chunk.cz * 4 + dz)
            for dz in range(5)
            for dx in range(5)
        )
        if tuple(dependency for dependency, _ in dependencies) != expected_dependencies:
            raise ValueError(f"LOD1 water dependency closure mismatch: {chunk}")
        reduced = reduce_water_lod1(
            ChunkId(chunk.cx, chunk.cz, chunk.lod),
            lambda key: loaded(HeightChunkId(key.lod, key.cx, key.cz))[0],
            core_res=water_core_res,
        )
        source = (
            np.full((water_core_res + 1, water_core_res + 1), np.nan, dtype=np.float64)
            if reduced is None
            else reduced.water_y
        )
        if _array_sha256(source, "<f8") != row.get("sourceValuesSha256"):
            raise ValueError(f"LOD1 water is not the fixed decoded-child reduction: {chunk}")
    return present, tombstones, {
        "transactionSha256": _sha256_file(path),
        "present": [list(key) for key in sorted(present)],
        "tombstones": [list(key) for key in sorted(tombstones)],
    }


def _overlay_set_sha256(plan: dict[str, Any]) -> str:
    identity = [
        {
            "key": [entry["layer"], entry["lod"], entry["cx"], entry["cz"]],
            "size": entry["size"],
            "sha256": entry["sha256"],
        }
        for entry in sorted(plan["chunks"], key=lambda item: (item["layer"], item["lod"], item["cx"], item["cz"]))
    ]
    return _sha256_bytes(_json_bytes(identity))


def verify_structural_repair(
    recipe_sha256: str,
    corrected_base_manifest_path: Path,
    content_root: Path,
    work_root: Path,
    *,
    encode: EncodeConfig,
) -> dict[str, Any]:
    """Reopen every frozen transaction and authorize only a preview overlay."""
    recipe_sha256 = _digest(recipe_sha256, "recipe sha256")
    build_root = Path(work_root) / "builds" / recipe_sha256
    expectation = load_structural_repair_expectation(
        build_root / "expectation.json", recipe_sha256
    )
    if expectation.get("recipeKind") != RECIPE_KIND or expectation.get("releaseDisposition") != "preview-only":
        raise ValueError("structural release expectation is not preview-only")
    plan_blob = (build_root / "plan.json").read_bytes()
    if _sha256_bytes(plan_blob) != (build_root / "plan.sha256").read_text().strip():
        raise ValueError("release plan differs from plan.sha256")
    plan = json.loads(plan_blob)
    inputs, inputs_sha = load_verifier_inputs(build_root, recipe_sha256)
    if plan.get("structuralVerifierInputsSha256") != inputs_sha:
        raise ValueError("release plan is not bound to the structural verifier inputs")
    artifacts = inputs["artifacts"]
    authority_path = _checked_file(build_root, artifacts["authority"], "authority")
    materializer_path = _checked_file(build_root, artifacts["materializer"], "materializer")
    resolved_path = _checked_file(build_root, artifacts["correctedBasePlan"], "correctedBasePlan")
    corrected_verify_path = _checked_file(build_root, artifacts["correctedBaseVerification"], "correctedBaseVerification")
    water_path = _checked_file(build_root, artifacts["waterTransaction"], "waterTransaction")
    scientific_path = _checked_file(
        build_root, artifacts["scientificClosure"], "scientificClosure"
    )
    try:
        from .authority_inventory import StructuralAuthorityInventory
        from .fine_water import reevaluate_fine_flowing_water
        from .hierarchy import assemble_fine_surface_window
        from .shared_closure import load_shared_closure_manifest
    except ImportError as error:
        raise ValueError(
            "independent expanded-halo fine closure rerasterizer is unavailable"
        ) from error

    authority, authority_evidence = _verify_authority(
        authority_path,
        artifacts["authority"]["sha256"],
        expectation["recipeInputs"]["structuralAuthority"],
    )
    scientific_document = _validate_scientific_closure_manifest(
        scientific_path,
        authority_sha256=authority_evidence["manifestSha256"],
        campaign_evidence=expectation["recipeInputs"]["campaignEvidence"],
    )
    with _content_addressed_snapshot(
        scientific_path, scientific_document["contentSha256"]
    ) as scientific_raster_path:
        shared_inputs = load_shared_closure_manifest(scientific_raster_path)
        if _sha256_file(authority_path) != shared_inputs.authority_manifest_sha256:
            raise ValueError("shared closure names another authority manifest")
        authority_support = frozenset(shared_inputs.authority_support)
        inventory = StructuralAuthorityInventory(
            manifest_path=authority_path,
            manifest_sha256=shared_inputs.authority_manifest_sha256,
            expected_chunks=shared_inputs.authority_support,
            cache_tiles=12,
        )
        grid = load_base().grid

        def rerasterize(
            chunk: HeightChunkId,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if chunk not in authority_support:
                raise ValueError(
                    "fine rerasterization chunk leaves declared authority support"
                )
            window = assemble_fine_surface_window(
                chunk,
                load_structural_height=lambda dependency: inventory.load(
                    dependency
                ).structural.height,
                load_baseline_height=lambda dependency: inventory.load_canonical(
                    dependency
                ).tile.height,
            )
            result = reevaluate_fine_flowing_water(
                window,
                grid=grid,
                mapped_water_polygon=shared_inputs.mapped_water,
                qualified_water_polygon=shared_inputs.qualified_water,
                centerline=shared_inputs.centerline,
                profile=shared_inputs.profile,
            )
            rerasterized = (
                result.authority,
                result.abstained,
                np.ascontiguousarray(window.baseline_height[window.core_slice]),
            )
            if not isinstance(rerasterized, tuple) or len(rerasterized) != 3:
                raise ValueError(
                    "independent fine closure returned an invalid contract"
                )
            return rerasterized

        fine, hierarchy_evidence, ownership_evidence = _verify_materializer(
            materializer_path,
            artifacts["materializer"]["sha256"],
            recipe_sha256,
            authority_evidence["manifestSha256"],
            authority_evidence["canonicalBaseManifestSha256"],
            encode,
            authority,
            expectation["recipeInputs"]["campaignEvidence"],
            rerasterize,
            artifacts["scientificClosure"]["sha256"],
        )
    base_evidence = _verify_corrected_base(
        Path(corrected_base_manifest_path),
        Path(content_root),
        corrected_verify_path,
        resolved_path,
        artifacts["correctedBasePlan"]["sha256"],
        expectation["baseManifestSha256"],
    )
    water_present, water_tombstones, water_evidence = _verify_water(
        water_path,
        artifacts["waterTransaction"]["sha256"],
        recipe_sha256,
        build_root,
        Path(corrected_base_manifest_path),
        Path(content_root),
        encode,
        authority_evidence["canonicalBaseManifestSha256"],
    )
    actual = {(entry["layer"], entry["lod"], entry["cx"], entry["cz"]) for entry in plan["chunks"]}
    expected_height = {tuple(value) for value in expectation["expectedPublished"]}
    if actual != expected_height | water_present or set(map(tuple, plan.get("tombstones", []))) != water_tombstones:
        raise ValueError("published structural overlay differs from verified height/water artifacts")
    if hierarchy_evidence["published"] != [value[1:] for value in expectation["expectedPublished"]]:
        raise ValueError("published 16+1 height overlay differs from the expectation")

    gates = {
        "frozenInputs": _gate({"inputsSha256": inputs_sha, "expectationSha256": _sha256_file(build_root / "expectation.json")}),
        "authority": _gate(authority_evidence),
        "materializedHierarchy": _gate(hierarchy_evidence),
        "ownershipRestoration": _gate(ownership_evidence),
        "correctedBase": _gate(base_evidence),
        "correctedWater": _gate(water_evidence),
        "publishedOverlay": _gate({"heightChunks": 17, "waterChunks": len(water_present), "waterTombstones": len(water_tombstones)}),
    }
    report = {
        "format": REPORT_FORMAT,
        "verifier": VERIFIER_ID,
        "verifierSourceSha256": verifier_source_sha256(),
        "recipeKind": RECIPE_KIND,
        "recipeSha256": recipe_sha256,
        "releaseDisposition": "preview-only",
        "planSha256": _sha256_bytes(plan_blob),
        "overlaySetSha256": _overlay_set_sha256(plan),
        "gates": gates,
        "passed": True,
    }
    blob = _json_bytes(report)
    destination = build_root / "structural-verify.json"
    if destination.exists() and destination.read_bytes() != blob:
        raise ValueError("immutable structural verification report differs")
    if not destination.exists():
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        temporary.write_bytes(blob)
        temporary.replace(destination)
    return report
