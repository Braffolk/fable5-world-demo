"""Restartable Stage-1 structural-repair transaction for the frozen Ahja pilot."""
from __future__ import annotations

import functools
import hashlib
import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ...config import ASSET_GEN_ROOT, DATA_WORK, load_base
from ...cook.chunkio import read_chunk_v2
from ...cook.encode import decode_quant16
from ...grid import ChunkId
from ...height_geom import HeightChunkId, parent_of
from ...pilots.taevaskoda_authority import (
    _verify_inputs,
    load_authority_config,
    run_authority_cook,
)
from ...pilots import taevaskoda_authority as authority_pilot
from ...release import (
    create_build_plan,
    materialize_corrected_format1_base,
    materialize_preview,
)
from . import base_transaction as corrected_base_policy
from . import hierarchy as structural_hierarchy
from . import shared_closure
from .authority_inventory import StructuralAuthorityInventory
from .base_transaction import (
    AuditedFormat1HeightSource,
    CorrectedLod0Core,
    build_corrected_base_transaction,
)
from .fine_water import reevaluate_fine_flowing_water
from .materialize import (
    MATERIALIZER_VERSION,
    decode_fine_ownership,
    materialize_structural_hierarchy,
)
from .recipe import (
    derive_structural_repair_recipe,
    freeze_structural_repair_expectation,
)
from .water_halo import HaloCorrectedWaterLayer, correct_flowing_water_chunk_halo
from .water_pack import (
    AbsentWaterChunk,
    DecodedWaterDependency,
    PackedWaterChunk,
    WaterPackResult,
    pack_corrected_water_lod0,
    pack_reduced_water_lod1,
)
from .water_source import AuditedFormat1WaterSource


STAGE1_TRANSACTION_VERSION = "taevaskoda-ahja-stage1-transaction/1"
_TRANSACTION_DOMAIN = b"laas.terrain.repair.taevaskoda-stage1-transaction.v1\0"
_PARENT_WATER_EVIDENCE_DOMAIN = b"laas.terrain.repair.corrected-water-parent.v1\0"
_DEFAULT_CONFIG = (
    ASSET_GEN_ROOT / "config/terrain-repair/taevaskoda-ahja-authority-stage1.json"
)
_REQUIRED_AUTHORITY_EVIDENCE_BINDING = "qualification-and-campaign-content/1"
_REQUIRED_SHARED_CLOSURE = "halo-aware-shared-water-closure/2"
_REQUIRED_SHARED_RASTERIZER = "laas.terrain.shared-water-closure-raster.v1"
_REQUIRED_QOFFSET_POLICY = (
    "preserve-inherited-or-retune-full-payload-outside-bitexact/1"
)
_REQUIRED_ANCESTOR_SPLICE = (
    "masked-child-reduction-splice-preserve-independent-dtm/1"
)
_REQUIRED_FINE_WINDOW = "real-expanded-structural-baseline-halo-16/1"
_REQUIRED_OUTSIDE_PROOF = "decoded-f32-outside-mask-bitexact/1"
_REQUIRED_QUANTIZER_DOMAIN = (
    "final-full-payload-core-and-east-south-southeast-aprons"
)
_IDENTITY_SOURCES = (
    "config/base.toml",
    "src/assetgen/pilots/taevaskoda_authority.py",
    "src/assetgen/terrain/repair/authority_inventory.py",
    "src/assetgen/terrain/repair/base_transaction.py",
    "src/assetgen/terrain/repair/closure.py",
    "src/assetgen/terrain/repair/fine_water.py",
    "src/assetgen/terrain/repair/hierarchy.py",
    "src/assetgen/terrain/repair/materialize.py",
    "src/assetgen/terrain/repair/shared_closure.py",
    "src/assetgen/terrain/repair/stage1_transaction.py",
    "src/assetgen/terrain/repair/water_halo.py",
    "src/assetgen/terrain/repair/water_layer.py",
    "src/assetgen/terrain/repair/water_pack.py",
    "src/assetgen/terrain/repair/water_source.py",
)


def _require_scientific_preconditions() -> None:
    required = {
        "authority evidence binding": (
            getattr(authority_pilot, "AUTHORITY_EVIDENCE_BINDING_VERSION", None),
            _REQUIRED_AUTHORITY_EVIDENCE_BINDING,
        ),
        "shared coarse/fine closure": (
            getattr(shared_closure, "SHARED_WATER_CLOSURE_VERSION", None),
            _REQUIRED_SHARED_CLOSURE,
        ),
        "shared closure rasterizer": (
            getattr(shared_closure, "SHARED_WATER_RASTERIZER_ID", None),
            _REQUIRED_SHARED_RASTERIZER,
        ),
        "expanded fine surface window": (
            getattr(structural_hierarchy, "FINE_SURFACE_WINDOW_VERSION", None),
            _REQUIRED_FINE_WINDOW,
        ),
        "independent fine closure rerasterizer": (
            callable(
                getattr(
                    shared_closure,
                    "rerasterize_fine_shared_closure_for_verification",
                    None,
                )
            ),
            True,
        ),
        "corrected full-payload qoffset": (
            getattr(corrected_base_policy, "CORRECTED_QOFFSET_VERSION", None),
            _REQUIRED_QOFFSET_POLICY,
        ),
        "corrected ancestor splice": (
            getattr(corrected_base_policy, "CORRECTED_ANCESTOR_SPLICE_VERSION", None),
            _REQUIRED_ANCESTOR_SPLICE,
        ),
        "corrected outside-mask proof": (
            getattr(corrected_base_policy, "OUTSIDE_PROOF_VERSION", None),
            _REQUIRED_OUTSIDE_PROOF,
        ),
        "corrected quantizer domain": (
            getattr(corrected_base_policy, "QUANTIZER_DOMAIN", None),
            _REQUIRED_QUANTIZER_DOMAIN,
        ),
    }
    mismatches = [
        f"{name}={actual!r}, required {expected!r}"
        for name, (actual, expected) in required.items()
        if actual != expected
    ]
    if mismatches:
        raise RuntimeError(
            "Stage-1 cook is blocked until audited scientific preconditions land: "
            + "; ".join(mismatches)
        )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(document: object) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _key(chunk: HeightChunkId | ChunkId) -> list[int]:
    if isinstance(chunk, HeightChunkId):
        return [chunk.lod, chunk.cx, chunk.cz]
    return [chunk.lod, chunk.cx, chunk.cz]


def _water_key(key: list[int] | tuple[int, int, int]) -> ChunkId:
    lod, cx, cz = (int(value) for value in key)
    return ChunkId(cx, cz, lod)


def _safe_child(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe transaction-relative path: {relative!r}")
    return root / path


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable transaction artifact differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError(f"concurrent immutable transaction conflict: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _write_marker(root: Path, name: str, transaction_sha256: str, body: dict) -> Path:
    marker = {
        "format": 1,
        "transactionVersion": STAGE1_TRANSACTION_VERSION,
        "transactionSha256": transaction_sha256,
        "step": name,
        **body,
    }
    path = root / "steps" / f"{name}.json"
    _write_immutable(path, _canonical_json(marker))
    return path


def _existing_marker(root: Path, name: str, transaction_sha256: str) -> dict | None:
    path = root / "steps" / f"{name}.json"
    if not path.exists():
        return None
    document = json.loads(path.read_bytes())
    if (
        document.get("format") != 1
        or document.get("transactionVersion") != STAGE1_TRANSACTION_VERSION
        or document.get("transactionSha256") != transaction_sha256
        or document.get("step") != name
    ):
        raise ValueError(f"restart marker identity differs: {path}")
    return document


def _checked_manifest(root: Path, relative: str, sha256: str) -> tuple[Path, dict]:
    path = _safe_child(root, relative)
    if _sha256_file(path) != sha256:
        raise ValueError(f"transaction manifest integrity mismatch: {path}")
    return path, json.loads(path.read_bytes())


def _transaction_identity(config, authority_path: Path) -> tuple[dict, str]:
    authority_sha = _sha256_file(authority_path)
    source_sha = {
        relative: _sha256_file(ASSET_GEN_ROOT / relative)
        for relative in _IDENTITY_SOURCES
    }
    identity = {
        "version": STAGE1_TRANSACTION_VERSION,
        "authorityConfigSha256": config.canonical_sha256,
        "authorityOrchestrationSha256": authority_sha,
        "pinnedBaseManifestSha256": config.raw["pinnedBase"]["manifestSha256"],
        "sourceSha256": source_sha,
        "closure": {
            "authoritySupport": [_key(chunk) for chunk in config.plan.authority_support],
            "fineReducer": [_key(chunk) for chunk in config.plan.lod2_reducer],
            "parentReducer": [_key(chunk) for chunk in config.plan.lod1_reducer],
            "correctedLod0": [_key(chunk) for chunk in config.plan.corrected_lod0],
            "publishedOverlay": [
                *[_key(chunk) for chunk in config.plan.published_lod2],
                _key(config.plan.review_parent),
            ],
        },
    }
    digest = hashlib.sha256(_TRANSACTION_DOMAIN + _canonical_json(identity)).hexdigest()
    return identity, digest


def _authority_manifest(authority_path: Path) -> tuple[Path, str, dict]:
    document = json.loads(authority_path.read_bytes())
    authority = document.get("authority", {})
    manifest = authority_path.parent / authority["manifest"]
    sha256 = authority["manifestSha256"]
    if _sha256_file(manifest) != sha256:
        raise ValueError("authority orchestration references a changed manifest")
    return manifest, sha256, document


def _run_scientific_closure(
    *,
    root: Path,
    transaction_sha256: str,
    config,
    inputs,
    authority_manifest_sha256: str,
) -> tuple[Path, str]:
    name = "02-scientific-closure"
    marker = _existing_marker(root, name, transaction_sha256)
    if marker is not None:
        path, document = _checked_manifest(
            root, marker["manifest"], marker["manifestSha256"]
        )
        loaded = shared_closure.load_shared_closure_manifest(path)
        if (
            loaded.authority_manifest_sha256 != authority_manifest_sha256
            or loaded.campaign_content_sha256
            != config.raw["campaign"]["contentSha256"]
            or loaded.campaign_qualification_sha256
            != inputs.profile.source_artifact_sha256
            or loaded.authority_support != config.plan.authority_support
            or document.get("contentSha256") != path.parent.name
        ):
            raise ValueError("scientific closure manifest differs from frozen Stage-1 inputs")
        return path, marker["manifestSha256"]
    result = shared_closure.write_shared_closure_manifest(
        output_root=root / "scientific-closure",
        authority_manifest_sha256=authority_manifest_sha256,
        campaign_content_sha256=config.raw["campaign"]["contentSha256"],
        campaign_qualification_sha256=inputs.profile.source_artifact_sha256,
        authority_support=config.plan.authority_support,
        mapped_water_polygon=inputs.mapped_water,
        qualified_water_polygon=inputs.qualified_water,
        centerline=inputs.centerline,
        profile=inputs.profile,
    )
    _write_marker(
        root,
        name,
        transaction_sha256,
        {
            "manifest": result.path.relative_to(root).as_posix(),
            "manifestSha256": result.manifest_sha256,
            "contentSha256": result.content_sha256,
        },
    )
    return result.path, result.manifest_sha256


def _validate_materialization(document: dict, config, artifact_root: Path | None = None) -> None:
    expected_overlay = [
        *[_key(chunk) for chunk in config.plan.published_lod2],
        _key(config.plan.review_parent),
    ]
    if (
        document.get("format") != 1
        or document.get("materializerVersion") != MATERIALIZER_VERSION
        or len(document.get("fine", ())) != 777
        or len(document.get("parents", ())) != 45
        or [row.get("key") for row in document.get("correctedLod0", ())]
        != [_key(chunk) for chunk in config.plan.corrected_lod0]
        or document.get("publishedOverlay") != expected_overlay
    ):
        raise ValueError("materialization manifest differs from the frozen Stage-1 closure")
    for row in document["fine"]:
        ownership = row.get("ownership")
        if (
            not isinstance(ownership, dict)
            or ownership.get("authoritySamples", -1) < 0
            or ownership.get("abstainedSamples", -1) < 0
            or not ownership.get("authoritySha256")
            or not ownership.get("abstentionSha256")
            or not ownership.get("containerSha256")
        ):
            raise ValueError("fine materialization lacks verifier-grade ownership evidence")
    if artifact_root is not None:
        for row in (
            *document["fine"],
            *document["parents"],
            *document["correctedLod0"],
        ):
            artifact = _safe_child(artifact_root, row["path"])
            sidecar = _safe_child(artifact_root, row["sidecar"])
            if artifact.stat().st_size != row["bytes"] or not sidecar.is_file():
                raise ValueError(f"materialization artifact inventory is incomplete: {artifact}")
            ownership = row.get("ownership")
            if ownership is not None:
                mask = _safe_child(artifact_root, ownership["path"])
                if mask.stat().st_size != ownership["bytes"]:
                    raise ValueError(f"materialization ownership inventory is incomplete: {mask}")


def _run_materialization(
    *,
    root: Path,
    transaction_sha256: str,
    config,
    inputs,
    authority_manifest_path: Path,
    authority_manifest_sha256: str,
    scientific_closure_sha256: str,
) -> tuple[Path, str, dict]:
    name = "03-materialization"
    marker = _existing_marker(root, name, transaction_sha256)
    if marker is not None:
        path, document = _checked_manifest(
            root, marker["manifest"], marker["manifestSha256"]
        )
        _validate_materialization(document, config, path.parent)
        return path, marker["manifestSha256"], document
    base = load_base()
    inventory = StructuralAuthorityInventory(
        manifest_path=authority_manifest_path,
        manifest_sha256=authority_manifest_sha256,
        expected_chunks=config.plan.authority_support,
        cache_tiles=12,
    )
    reevaluate = functools.partial(
        reevaluate_fine_flowing_water,
        grid=base.grid,
        mapped_water_polygon=inputs.mapped_water,
        qualified_water_polygon=inputs.qualified_water,
        centerline=inputs.centerline,
        profile=inputs.profile,
    )
    ownership_closure = {
        "version": shared_closure.SHARED_WATER_CLOSURE_VERSION,
        "authorityManifestSha256": authority_manifest_sha256,
        "campaignContentSha256": config.raw["campaign"]["contentSha256"],
        "campaignQualificationSha256": inputs.profile.source_artifact_sha256,
        "scientificClosureSha256": scientific_closure_sha256,
        "rasterizer": {
            "id": shared_closure.SHARED_WATER_RASTERIZER_ID,
            "sourceSha256": _sha256_file(Path(shared_closure.__file__)),
        },
    }
    result = materialize_structural_hierarchy(
        grid=base.grid,
        encode=base.encode,
        plan=config.plan,
        recipe_sha256=transaction_sha256,
        authority_manifest_sha256=authority_manifest_sha256,
        canonical_base_manifest_sha256=inventory.base_manifest_sha256,
        load_pair=inventory.load,
        load_canonical_baseline=inventory.load_canonical,
        reevaluate_fine_surface=reevaluate,
        ownership_closure=ownership_closure,
        output_root=root / "materialization",
        pair_cache_tiles=12,
        canonical_cache_tiles=12,
    )
    document = json.loads(result.manifest_path.read_bytes())
    _validate_materialization(document, config, result.manifest_path.parent)
    relative = result.manifest_path.relative_to(root).as_posix()
    _write_marker(
        root,
        name,
        transaction_sha256,
        {
            "manifest": relative,
            "manifestSha256": result.manifest_sha256,
            "fineCount": 777,
            "parentCount": 45,
            "correctedLod0Count": 2,
        },
    )
    return result.manifest_path, result.manifest_sha256, document


def _decode_materialized_lod0(
    materialization_root: Path,
    document: dict,
    encode,
) -> dict[HeightChunkId, CorrectedLod0Core]:
    corrected_values: dict[HeightChunkId, np.ndarray] = {}
    for row in document["correctedLod0"]:
        chunk = HeightChunkId(*row["key"])
        path = _safe_child(materialization_root, row["path"])
        if path.stat().st_size != row["bytes"] or _sha256_file(path) != row["containerSha256"]:
            raise ValueError(f"materialized LOD0 candidate integrity mismatch: {path}")
        meta, payload = read_chunk_v2(path)
        if (meta.layer, meta.lod, meta.cx, meta.cz, meta.res) != (
            "height",
            chunk.lod,
            chunk.cx,
            chunk.cz,
            2049,
        ):
            raise ValueError(f"materialized LOD0 candidate header mismatch: {path}")
        decoded = decode_quant16(encode, payload, meta.res, meta.qoffset, meta.qscale)
        if _sha256_bytes(np.ascontiguousarray(decoded, dtype="<f4").tobytes()) != row[
            "decodedValuesSha256"
        ]:
            raise ValueError(f"materialized LOD0 decoded hash mismatch: {path}")
        corrected_values[chunk] = decoded[:-1, :-1]

    affected = {
        chunk: np.zeros(values.shape, dtype=np.bool_)
        for chunk, values in corrected_values.items()
    }
    for row in document["fine"]:
        fine = HeightChunkId(*row["key"])
        ownership = row.get("ownership")
        if not isinstance(ownership, dict):
            raise ValueError(f"materialized fine chunk lacks ownership: {fine}")
        path = _safe_child(materialization_root, ownership["path"])
        payload = path.read_bytes()
        if (
            len(payload) != ownership.get("bytes")
            or _sha256_bytes(payload) != ownership.get("containerSha256")
        ):
            raise ValueError(f"materialized fine ownership integrity mismatch: {path}")
        authority, abstained = decode_fine_ownership(payload)
        if (
            _sha256_bytes(np.ascontiguousarray(authority, dtype="u1").tobytes())
            != ownership.get("authoritySha256")
            or _sha256_bytes(np.ascontiguousarray(abstained, dtype="u1").tobytes())
            != ownership.get("abstentionSha256")
            or int(np.count_nonzero(authority))
            != ownership.get("authoritySamples")
            or int(np.count_nonzero(abstained))
            != ownership.get("abstainedSamples")
        ):
            raise ValueError(f"materialized fine ownership identity mismatch: {path}")
        lod0 = parent_of(parent_of(fine))
        target = affected.get(lod0)
        if target is None:
            continue
        core = authority[:-1, :-1]
        if core.shape != (2048, 2048):
            raise ValueError(f"materialized fine ownership has the wrong shape: {fine}")
        reduced = core.reshape(128, 16, 128, 16).any(axis=(1, 3))
        dx = fine.cx - lod0.cx * 16
        dz = fine.cz - lod0.cz * 16
        if dx not in range(16) or dz not in range(16):
            raise AssertionError(f"invalid LOD-2 to LOD0 relation: {fine} -> {lod0}")
        target[
            dz * 128 : (dz + 1) * 128,
            dx * 128 : (dx + 1) * 128,
        ] |= reduced

    result: dict[HeightChunkId, CorrectedLod0Core] = {}
    for chunk, values in corrected_values.items():
        mask = affected[chunk]
        if not mask.any():
            raise ValueError(f"corrected LOD0 core has no qualified authority: {chunk}")
        result[chunk] = CorrectedLod0Core(values=values, affected_mask=mask)
    return result


def _validate_base_plan(
    document: dict, config, plan_root: Path | None = None
) -> None:
    if (
        document.get("format") != 2
        or document.get("correctedLod0")
        != [_key(chunk) for chunk in config.plan.corrected_lod0]
        or [rung.get("lod") for rung in document.get("rungs", ())] != [0, 1, 2, 3, 4]
        or document.get("ancestorSpliceVersion") != _REQUIRED_ANCESTOR_SPLICE
        or document.get("qoffsetPolicy") != _REQUIRED_QOFFSET_POLICY
        or not document.get("artifacts")
    ):
        raise ValueError("corrected-base plan differs from the frozen Stage-1 contract")
    for artifact in document["artifacts"]:
        affected = artifact.get("affectedMask")
        outside = artifact.get("outsideProof")
        if (
            artifact.get("maskedSpliceVersion") != _REQUIRED_ANCESTOR_SPLICE
            or not isinstance(affected, dict)
            or affected.get("sampleCount", -1) < 0
            or not isinstance(artifact.get("affectedWindows"), dict)
            or not artifact["affectedWindows"].get("sha256")
            or not artifact.get("inheritedDecodedCoreSha256")
            or artifact.get("quantizerDomain") != _REQUIRED_QUANTIZER_DOMAIN
            or not isinstance(outside, dict)
            or outside.get("version") != _REQUIRED_OUTSIDE_PROOF
            or outside.get("sampleCount", -1) < 0
            or outside.get("bitExact") is not True
            or not outside.get("inheritedDecodedSha256")
            or outside.get("inheritedDecodedSha256")
            != outside.get("outputDecodedSha256")
        ):
            raise ValueError("corrected-base artifact lacks masked-splice proof")
        if plan_root is not None:
            for name, entry in (
                ("affected mask", affected),
                ("affected core mask", artifact.get("affectedCoreMask")),
                ("affected windows", artifact.get("affectedWindows")),
            ):
                if not isinstance(entry, dict):
                    raise ValueError(f"corrected-base {name} inventory is absent")
                path = _safe_child(plan_root, entry.get("path"))
                if (
                    path.stat().st_size != entry.get("bytes")
                    or _sha256_file(path) != entry.get("sha256")
                ):
                    raise ValueError(f"corrected-base {name} differs: {path}")


def _run_corrected_base(
    *,
    root: Path,
    transaction_sha256: str,
    config,
    materialization_path: Path,
    materialization: dict,
) -> tuple[Path, str, dict]:
    name = "04-corrected-base"
    marker = _existing_marker(root, name, transaction_sha256)
    if marker is not None:
        path, document = _checked_manifest(root, marker["plan"], marker["planSha256"])
        _validate_base_plan(document, config, path.parent)
        return path, marker["planSha256"], document
    base = load_base()
    pinned = config.raw["pinnedBase"]
    source = AuditedFormat1HeightSource(
        manifest_path=config.pinned_manifest,
        manifest_sha256=pinned["manifestSha256"],
        content_root=config.pinned_content_root,
        encode=base.encode,
        cache_chunks=int(pinned["cacheChunks"]),
    )
    corrected = _decode_materialized_lod0(
        materialization_path.parent, materialization, base.encode
    )
    result = build_corrected_base_transaction(
        source=source,
        staging_root=root / "corrected-base",
        corrected_lod0=corrected,
        grid=base.grid,
        encode=base.encode,
        maximum_lod=4,
    )
    document = json.loads(result.plan_path.read_bytes())
    _validate_base_plan(document, config, result.plan_path.parent)
    relative = result.plan_path.relative_to(root).as_posix()
    _write_marker(
        root,
        name,
        transaction_sha256,
        {
            "plan": relative,
            "planSha256": result.plan_sha256,
            "replacementCount": len(result.plan.artifacts),
            "maximumLod": 4,
        },
    )
    return result.plan_path, result.plan_sha256, document


@dataclass(frozen=True)
class _WaterArtifact:
    chunk: ChunkId
    disposition: str
    relative_path: str
    bytes: int
    file_sha256: str
    artifact_sha256: str
    source_values_sha256: str
    decoded_values_sha256: str | None
    evidence_sha256: str
    dependency_sha256: str
    dependencies: tuple[tuple[ChunkId, str], ...]
    wet_samples: int
    dry_samples: int
    source_values_relative_path: str
    source_values_bytes: int
    source_values_file_sha256: str
    evidence_relative_path: str
    evidence_bytes: int
    evidence_file_sha256: str
    payload_sha256: str | None
    max_roundtrip_error_m: float | None
    roundtrip_limit_m: float | None

    def to_json(self) -> dict:
        return {
            "key": _key(self.chunk),
            "disposition": self.disposition,
            "path": self.relative_path,
            "bytes": self.bytes,
            "fileSha256": self.file_sha256,
            "artifactSha256": self.artifact_sha256,
            "sourceValuesSha256": self.source_values_sha256,
            "decodedValuesSha256": self.decoded_values_sha256,
            "evidenceSha256": self.evidence_sha256,
            "sourceValues": {
                "path": self.source_values_relative_path,
                "bytes": self.source_values_bytes,
                "sha256": self.source_values_file_sha256,
            },
            "evidence": {
                "path": self.evidence_relative_path,
                "bytes": self.evidence_bytes,
                "sha256": self.evidence_file_sha256,
            },
            "dependencySha256": self.dependency_sha256,
            "dependencies": [
                {"key": _key(chunk), "artifactSha256": sha256}
                for chunk, sha256 in self.dependencies
            ],
            "wetSamples": self.wet_samples,
            "drySamples": self.dry_samples,
            "payloadSha256": self.payload_sha256,
            "maxRoundtripErrorM": self.max_roundtrip_error_m,
            "roundtripLimitM": self.roundtrip_limit_m,
        }


def _array_sha256(values: np.ndarray, dtype: str) -> str:
    array = np.array(values, dtype=np.dtype(dtype), copy=True)
    if np.issubdtype(array.dtype, np.floating):
        array[np.isnan(array)] = np.nan
    return _sha256_bytes(np.ascontiguousarray(array).tobytes())


def _npy_bytes(values: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, np.asarray(values, dtype="<f8"), allow_pickle=False)
    return output.getvalue()


def _write_water_result(
    root: Path, result: WaterPackResult, *, evidence_document: dict
) -> _WaterArtifact:
    source_sha = _array_sha256(result.source_water_y, "<f8")
    if source_sha != result.source_values_sha256:
        raise ValueError("water pack result does not retain its canonical source values")
    source_payload = _npy_bytes(result.source_water_y)
    source_relative = (
        Path("evidence")
        / "source-values"
        / str(result.chunk.lod)
        / f"{result.chunk.cx}_{result.chunk.cz}.{source_sha}.npy"
    )
    _write_immutable(root / source_relative, source_payload)
    evidence_payload = json.dumps(
        evidence_document, sort_keys=True, separators=(",", ":")
    ).encode()
    if _sha256_bytes(evidence_payload) != result.evidence_sha256:
        raise ValueError("water evidence document differs from the packed evidence identity")
    evidence_relative = (
        Path("evidence")
        / "correction"
        / str(result.chunk.lod)
        / f"{result.chunk.cx}_{result.chunk.cz}.{result.evidence_sha256}.json"
    )
    _write_immutable(root / evidence_relative, evidence_payload)
    evidence_fields = {
        "source_values_relative_path": source_relative.as_posix(),
        "source_values_bytes": len(source_payload),
        "source_values_file_sha256": _sha256_bytes(source_payload),
        "evidence_relative_path": evidence_relative.as_posix(),
        "evidence_bytes": len(evidence_payload),
        "evidence_file_sha256": _sha256_bytes(evidence_payload),
    }
    if isinstance(result, PackedWaterChunk):
        relative = (
            Path("chunks")
            / "water"
            / str(result.chunk.lod)
            / f"{result.chunk.cx}_{result.chunk.cz}.{result.artifact_sha256}.lac"
        )
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        candidate = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        result.write_lac1(candidate)
        payload = candidate.read_bytes()
        if _sha256_bytes(payload) != result.artifact_sha256:
            candidate.unlink(missing_ok=True)
            raise AssertionError("packed water container hash differs from its identity")
        if path.exists():
            if path.read_bytes() != payload:
                candidate.unlink(missing_ok=True)
                raise ValueError(f"immutable corrected water differs: {path}")
            candidate.unlink()
        else:
            candidate.replace(path)
        return _WaterArtifact(
            chunk=result.chunk,
            disposition="replacement",
            relative_path=relative.as_posix(),
            bytes=len(payload),
            file_sha256=result.artifact_sha256,
            artifact_sha256=result.artifact_sha256,
            source_values_sha256=result.source_values_sha256,
            decoded_values_sha256=result.decoded_values_sha256,
            evidence_sha256=result.evidence_sha256,
            dependency_sha256=result.dependency_sha256,
            dependencies=result.dependencies,
            wet_samples=result.wet_samples,
            dry_samples=result.dry_samples,
            payload_sha256=result.payload_sha256,
            max_roundtrip_error_m=result.max_roundtrip_error_m,
            roundtrip_limit_m=result.roundtrip_limit_m,
            **evidence_fields,
        )
    document = {
        "format": 1,
        "role": "all-dry-water-removal-candidate",
        "key": _key(result.chunk),
        "tombstoneSha256": result.tombstone_sha256,
        "sourceValuesSha256": result.source_values_sha256,
        "evidenceSha256": result.evidence_sha256,
        "dependencySha256": result.dependency_sha256,
        "dependencies": [
            {"key": _key(chunk), "artifactSha256": sha256}
            for chunk, sha256 in result.dependencies
        ],
        "drySamples": result.dry_samples,
        "reason": result.reason,
    }
    payload = _canonical_json(document)
    relative = (
        Path("tombstones")
        / "water"
        / str(result.chunk.lod)
        / f"{result.chunk.cx}_{result.chunk.cz}.{result.tombstone_sha256}.json"
    )
    _write_immutable(root / relative, payload)
    return _WaterArtifact(
        chunk=result.chunk,
        disposition="remove-inherited",
        relative_path=relative.as_posix(),
        bytes=len(payload),
        file_sha256=_sha256_bytes(payload),
        artifact_sha256=result.tombstone_sha256,
        source_values_sha256=result.source_values_sha256,
        decoded_values_sha256=None,
        evidence_sha256=result.evidence_sha256,
        dependency_sha256=result.dependency_sha256,
        dependencies=result.dependencies,
        wet_samples=0,
        dry_samples=result.dry_samples,
        payload_sha256=None,
        max_roundtrip_error_m=None,
        roundtrip_limit_m=None,
        **evidence_fields,
    )


def _decoded_packed_water(result: PackedWaterChunk, encode) -> np.ndarray:
    decoded = decode_quant16(
        encode, result.payload, result.meta.res, result.meta.qoffset, result.meta.qscale
    ).astype(np.float32, copy=True)
    codes = np.rint(
        (decoded.astype(np.float64) - float(result.meta.qoffset))
        / float(result.meta.qscale)
    )
    decoded[codes == 0.0] = np.nan
    return decoded


def _parent_water_evidence(transaction_sha256: str, parent: ChunkId) -> tuple[dict, str]:
    identity = hashlib.sha256(_PARENT_WATER_EVIDENCE_DOMAIN)
    identity.update(bytes.fromhex(transaction_sha256))
    identity.update(bytes((parent.lod,)))
    identity.update(parent.cx.to_bytes(4, "little", signed=True))
    identity.update(parent.cz.to_bytes(4, "little", signed=True))
    document = {
        "format": 1,
        "role": "decoded-child-lod1-water-reduction",
        "transactionSha256": transaction_sha256,
        "key": _key(parent),
        "identitySha256": identity.hexdigest(),
    }
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return document, _sha256_bytes(payload)


def _run_corrected_water(
    *,
    root: Path,
    transaction_sha256: str,
    config,
    inputs,
) -> tuple[Path, str, dict]:
    name = "05-corrected-water"
    marker = _existing_marker(root, name, transaction_sha256)
    if marker is not None:
        path, document = _checked_manifest(
            root, marker["manifest"], marker["manifestSha256"]
        )
        _validate_water_transaction(document, path.parent, config)
        return path, marker["manifestSha256"], document
    base = load_base()
    pinned = config.raw["pinnedBase"]
    source = AuditedFormat1WaterSource(
        manifest_path=config.pinned_manifest,
        manifest_sha256=pinned["manifestSha256"],
        content_root=config.pinned_content_root,
        grid=base.grid,
        encode=base.encode,
        cache_chunks=int(pinned["cacheChunks"]),
        audit=False,
    )
    water_root = root / "corrected-water"
    lod0_results: dict[ChunkId, WaterPackResult] = {}
    halo_evidence: dict[ChunkId, HaloCorrectedWaterLayer] = {}
    records: list[_WaterArtifact] = []
    for height_chunk in config.plan.corrected_lod0:
        chunk = ChunkId(height_chunk.cx, height_chunk.cz, 0)
        corrected = correct_flowing_water_chunk_halo(
            chunk=chunk,
            grid=base.grid,
            load_baseline=source.load_halo,
            mapped_water_polygon=inputs.mapped_water,
            qualified_water_polygon=inputs.qualified_water,
            centerline=inputs.centerline,
            profile=inputs.profile,
        )
        inherited = source.load(chunk)
        inherited_sha = (
            source.absent_sha256(chunk)
            if inherited is None
            else inherited.artifact_sha256
        )
        result = pack_corrected_water_lod0(
            corrected.layer,
            chunk=chunk,
            grid=base.grid,
            encode=base.encode,
            inherited_artifact_sha256=inherited_sha,
            inherited_qoffset=None if inherited is None else inherited.meta.qoffset,
            inherited_qscale=None if inherited is None else inherited.meta.qscale,
        )
        lod0_results[chunk] = result
        halo_evidence[chunk] = corrected
        records.append(
            _write_water_result(
                water_root,
                result,
                evidence_document=asdict(corrected.layer.evidence),
            )
        )

    corrected_dependencies: dict[ChunkId, DecodedWaterDependency] = {}
    for chunk, result in lod0_results.items():
        if isinstance(result, PackedWaterChunk):
            corrected_dependencies[chunk] = DecodedWaterDependency(
                _decoded_packed_water(result, base.encode), result.artifact_sha256
            )
        else:
            corrected_dependencies[chunk] = DecodedWaterDependency(
                None, result.tombstone_sha256
            )

    parents = tuple(
        sorted(
            {
                ChunkId(parent.cx, parent.cz, parent.lod)
                for parent in (
                    parent_of(chunk) for chunk in config.plan.corrected_lod0
                )
            },
            key=lambda chunk: (chunk.lod, chunk.cz, chunk.cx),
        )
    )

    def load_child(chunk: ChunkId) -> DecodedWaterDependency:
        return corrected_dependencies.get(chunk) or source.load_dependency(chunk)

    for parent in parents:
        inherited = source.load(parent)
        evidence_document, evidence_sha256 = _parent_water_evidence(
            transaction_sha256, parent
        )
        result = pack_reduced_water_lod1(
            parent=parent,
            grid=base.grid,
            encode=base.encode,
            load_dependency=load_child,
            evidence_sha256=evidence_sha256,
            inherited_qoffset=None if inherited is None else inherited.meta.qoffset,
            inherited_qscale=None if inherited is None else inherited.meta.qscale,
        )
        records.append(
            _write_water_result(
                water_root, result, evidence_document=evidence_document
            )
        )

    ordered = tuple(sorted(records, key=lambda item: (item.chunk.lod, item.chunk.cz, item.chunk.cx)))
    document = {
        "format": 1,
        "transactionVersion": "corrected-structural-water-transaction/1",
        "transactionSha256": transaction_sha256,
        "recipeSha256": transaction_sha256,
        "waterCoreResolution": base.grid.chunk_m // 2,
        "sourceManifestSha256": source.manifest_sha256,
        "boundedMemory": {
            "inheritedDecodedChunkCache": source.cache_capacity,
            "correctedLod0HeldForParentReduction": len(lod0_results),
            "completeWaterHierarchyHeldInMemory": False,
        },
        "halo": [
            {
                "key": _key(chunk),
                "dependencySha256": corrected.dependency_sha256,
                "dependencies": [
                    {
                        "key": _key(dependency),
                        "source": source_kind.value,
                        "artifactSha256": sha256,
                    }
                    for dependency, source_kind, sha256 in corrected.dependencies
                ],
            }
            for chunk, corrected in sorted(
                halo_evidence.items(), key=lambda item: (item[0].cz, item[0].cx)
            )
        ],
        "artifacts": [record.to_json() for record in ordered],
    }
    manifest_path = water_root / "transaction.json"
    _write_immutable(manifest_path, _canonical_json(document))
    _validate_water_transaction(document, water_root, config)
    manifest_sha = _sha256_file(manifest_path)
    _write_marker(
        root,
        name,
        transaction_sha256,
        {
            "manifest": manifest_path.relative_to(root).as_posix(),
            "manifestSha256": manifest_sha,
            "lod0Count": 2,
            "lod1Count": 2,
        },
    )
    return manifest_path, manifest_sha, document


def _validate_water_transaction(document: dict, root: Path, config) -> None:
    lod0 = [ChunkId(chunk.cx, chunk.cz, 0) for chunk in config.plan.corrected_lod0]
    lod1 = sorted(
        {
            ChunkId(parent.cx, parent.cz, parent.lod)
            for parent in (parent_of(chunk) for chunk in config.plan.corrected_lod0)
        },
        key=lambda chunk: (chunk.lod, chunk.cz, chunk.cx),
    )
    expected = sorted((*lod0, *lod1), key=lambda chunk: (chunk.lod, chunk.cz, chunk.cx))
    rows = document.get("artifacts")
    halos = document.get("halo")
    if (
        document.get("format") != 1
        or document.get("transactionVersion")
        != "corrected-structural-water-transaction/1"
        or not isinstance(rows, list)
        or [_water_key(row.get("key", ())) for row in rows] != expected
        or not isinstance(halos, list)
        or [_water_key(row.get("key", ())) for row in halos] != lod0
    ):
        raise ValueError("corrected-water transaction key closure differs")
    for halo in halos:
        dependencies = halo.get("dependencies")
        if (
            not isinstance(dependencies, list)
            or len(dependencies) != 9
            or len({tuple(row.get("key", ())) for row in dependencies}) != 9
            or any(
                row.get("source")
                not in {"inherited", "corrected", "explicit_absent"}
                for row in dependencies
            )
        ):
            raise ValueError("corrected-water halo dependency inventory differs")
    for row in rows:
        chunk = _water_key(row["key"])
        dependencies = row.get("dependencies")
        expected_dependencies = 1 if chunk.lod == 0 else 25
        if (
            row.get("disposition") not in {"replacement", "remove-inherited"}
            or not isinstance(dependencies, list)
            or len(dependencies) != expected_dependencies
            or len({tuple(item.get("key", ())) for item in dependencies})
            != expected_dependencies
        ):
            raise ValueError("corrected-water artifact dependency inventory differs")
        source_values_path = _safe_child(root, row.get("sourceValues", {}).get("path"))
        evidence_path = _safe_child(root, row.get("evidence", {}).get("path"))
        source_entry = row["sourceValues"]
        evidence_entry = row["evidence"]
        if (
            source_values_path.stat().st_size != source_entry.get("bytes")
            or _sha256_file(source_values_path) != source_entry.get("sha256")
            or evidence_path.stat().st_size != evidence_entry.get("bytes")
            or _sha256_file(evidence_path) != evidence_entry.get("sha256")
            or evidence_entry.get("sha256") != row.get("evidenceSha256")
        ):
            raise ValueError("corrected-water evidence artifact inventory differs")
        source_values = np.load(source_values_path, allow_pickle=False)
        expected_shape = (document.get("waterCoreResolution", -1) + 1,) * 2
        if (
            source_values.dtype != np.dtype("<f8")
            or source_values.shape != expected_shape
            or np.isinf(source_values).any()
            or _array_sha256(source_values, "<f8")
            != row.get("sourceValuesSha256")
        ):
            raise ValueError("corrected-water source-values evidence differs")
        path = _safe_child(root, row["path"])
        if (
            path.stat().st_size != row["bytes"]
            or _sha256_file(path) != row["fileSha256"]
        ):
            raise ValueError(f"corrected-water transaction artifact differs: {path}")


def _stage_file(
    *, root: Path, source: Path, destination: Path, role: str, key: list[int] | None
) -> dict:
    source_sha = _sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256_file(destination) != source_sha:
            raise ValueError(f"immutable staged candidate differs: {destination}")
    else:
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        os.link(source, temporary)
        temporary.replace(destination)
    return {
        "role": role,
        "key": key,
        "source": source.relative_to(root).as_posix(),
        "path": destination.relative_to(root).as_posix(),
        "bytes": source.stat().st_size,
        "sha256": source_sha,
    }


def _run_staging(
    *,
    root: Path,
    transaction_sha256: str,
    config,
    materialization_path: Path,
    materialization: dict,
    base_plan_path: Path,
    base_plan: dict,
    water_path: Path,
    water: dict,
) -> tuple[Path, str, dict]:
    name = "06-staging"
    marker = _existing_marker(root, name, transaction_sha256)
    if marker is not None:
        path, document = _checked_manifest(
            root, marker["inventory"], marker["inventorySha256"]
        )
        if (
            document.get("negativeOverlayCount") != 17
            or document.get("correctedHeightCandidateCount") != len(base_plan["artifacts"])
            or document.get("correctedWaterCandidateCount") != 4
        ):
            raise ValueError("staged inventory differs from the frozen Stage-1 contract")
        for row in document.get("files", ()):
            candidate = _safe_child(root, row["path"])
            if (
                candidate.stat().st_size != row["bytes"]
                or _sha256_file(candidate) != row["sha256"]
            ):
                raise ValueError(f"staged candidate inventory differs: {candidate}")
        return path, marker["inventorySha256"], document

    stage_root = root / "stage"
    staged: list[dict] = []
    fine_by_key = {tuple(row["key"]): row for row in materialization["fine"]}
    parent_by_key = {tuple(row["key"]): row for row in materialization["parents"]}
    overlay_keys = [
        *[_key(chunk) for chunk in config.plan.published_lod2],
        _key(config.plan.review_parent),
    ]
    for key in overlay_keys:
        row = fine_by_key.get(tuple(key)) or parent_by_key.get(tuple(key))
        if row is None:
            raise ValueError(f"materialization lacks published overlay {key}")
        source = _safe_child(materialization_path.parent, row["path"])
        destination = (
            stage_root / "negative-overlay" / "height" / str(key[0]) / source.name
        )
        staged.append(
            _stage_file(
                root=root,
                source=source,
                destination=destination,
                role="negative-height-overlay",
                key=key,
            )
        )
        sidecar = _safe_child(materialization_path.parent, row["sidecar"])
        staged.append(
            _stage_file(
                root=root,
                source=sidecar,
                destination=destination.with_suffix(".json"),
                role="negative-height-sidecar",
                key=key,
            )
        )
        ownership = row.get("ownership")
        if key[0] == -2:
            if not isinstance(ownership, dict):
                raise ValueError(f"published fine overlay lacks ownership evidence: {key}")
            mask = _safe_child(materialization_path.parent, ownership["path"])
            staged.append(
                _stage_file(
                    root=root,
                    source=mask,
                    destination=(
                        stage_root
                        / "negative-overlay"
                        / "ownership"
                        / str(key[0])
                        / mask.name
                    ),
                    role="negative-height-ownership",
                    key=key,
                )
            )

    for artifact in base_plan["artifacts"]:
        key = artifact["chunk"]
        source = _safe_child(base_plan_path.parent, artifact["relative_path"])
        staged.append(
            _stage_file(
                root=root,
                source=source,
                destination=(
                    stage_root
                    / "corrected-format1"
                    / "height"
                    / str(key[0])
                    / source.name
                ),
                role="corrected-format1-height-candidate",
                key=key,
            )
        )

    for artifact in water["artifacts"]:
        key = artifact["key"]
        source = _safe_child(water_path.parent, artifact["path"])
        extension = ".lac" if artifact["disposition"] == "replacement" else ".tombstone.json"
        destination = (
            stage_root
            / "corrected-format1"
            / "water"
            / str(key[0])
            / f"{key[1]}_{key[2]}{extension}"
        )
        staged.append(
            _stage_file(
                root=root,
                source=source,
                destination=destination,
                role="corrected-format1-water-candidate",
                key=key,
            )
        )

    document = {
        "format": 1,
        "transactionVersion": STAGE1_TRANSACTION_VERSION,
        "transactionSha256": transaction_sha256,
        "negativeOverlayCount": len(overlay_keys),
        "negativeOverlayFileCount": 17 * 2 + 16,
        "correctedHeightCandidateCount": len(base_plan["artifacts"]),
        "correctedWaterCandidateCount": len(water["artifacts"]),
        "files": staged,
    }
    inventory_path = stage_root / "inventory.json"
    _write_immutable(inventory_path, _canonical_json(document))
    inventory_sha = _sha256_file(inventory_path)
    _write_marker(
        root,
        name,
        transaction_sha256,
        {
            "inventory": inventory_path.relative_to(root).as_posix(),
            "inventorySha256": inventory_sha,
            "negativeOverlayCount": 17,
            "correctedHeightCandidateCount": len(base_plan["artifacts"]),
            "correctedWaterCandidateCount": 4,
        },
    )
    return inventory_path, inventory_sha, document


def _entry(root: Path, path: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _link_tree_immutable(
    source: Path, destination: Path, *, skip: frozenset[str] = frozenset()
) -> None:
    for path in sorted(candidate for candidate in source.rglob("*") if candidate.is_file()):
        relative = path.relative_to(source).as_posix()
        if relative in skip:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != path.stat().st_size or _sha256_file(target) != _sha256_file(path):
                raise ValueError(f"immutable verifier snapshot differs: {target}")
            continue
        os.link(path, target)


def _run_corrected_base_release(
    *,
    root: Path,
    transaction_sha256: str,
    config,
    stage: dict,
) -> tuple[Path, str, Path, str]:
    name = "07-corrected-format1-release"
    marker = _existing_marker(root, name, transaction_sha256)
    if marker is not None:
        manifest = _safe_child(root, marker["manifest"])
        verification = _safe_child(root, marker["verification"])
        if (
            _sha256_file(manifest) != marker["manifestSha256"]
            or _sha256_file(verification) != marker["verificationSha256"]
        ):
            raise ValueError("corrected format-1 release changed after completion")
        return (
            manifest,
            marker["manifestSha256"],
            verification,
            marker["verificationSha256"],
        )
    replacements: dict[tuple[str, int, int, int], Path] = {}
    for row in stage["files"]:
        if row["role"] != "corrected-format1-height-candidate":
            continue
        lod, cx, cz = row["key"]
        replacements[("height", lod, cx, cz)] = _safe_child(root, row["path"])
    result = materialize_corrected_format1_base(
        base=load_base(),
        recipe_sha256=transaction_sha256,
        pinned_manifest_path=config.pinned_manifest,
        pinned_manifest_sha256=config.raw["pinnedBase"]["manifestSha256"],
        content_root=config.pinned_content_root,
        release_root=root / "corrected-format1-release",
        replacements=replacements,
    )
    _write_marker(
        root,
        name,
        transaction_sha256,
        {
            "manifest": result.manifest_path.relative_to(root).as_posix(),
            "manifestSha256": result.manifest_sha256,
            "verification": result.verification_path.relative_to(root).as_posix(),
            "verificationSha256": result.verification_sha256,
            "replacementCount": len(replacements),
        },
    )
    return (
        result.manifest_path,
        result.manifest_sha256,
        result.verification_path,
        result.verification_sha256,
    )


def _run_preview_release(
    *,
    root: Path,
    work_root: Path,
    config,
    inputs,
    authority_manifest_path: Path,
    scientific_path: Path,
    materialization_path: Path,
    base_plan_path: Path,
    water_path: Path,
    water: dict,
    stage: dict,
    corrected_manifest_path: Path,
    corrected_manifest_sha256: str,
    corrected_verification_path: Path,
) -> tuple[str, Path]:
    campaign_root = config.campaign_root
    pilot_config = (ASSET_GEN_ROOT / inputs.pilot["configPath"]).resolve()
    recipe = derive_structural_repair_recipe(
        base=load_base(),
        plan=config.plan,
        corrected_base_manifest_path=corrected_manifest_path,
        corrected_base_verification_path=corrected_verification_path,
        authority_manifest_path=authority_manifest_path,
        campaign_manifest_path=campaign_root / "campaign.json",
        pilot_manifest_path=campaign_root / "pilot.json",
        profile_artifacts={"campaign.npz": campaign_root / "campaign.npz"},
        config_artifacts={
            "pilot-config": pilot_config,
            "authority-config": config.path,
        },
        geometry_artifacts={
            name: campaign_root / name for name in inputs.pilot["geometrySha256"]
        },
        dtm_artifacts={path.name: path for path in config.dtm_sources},
    )
    build_root = Path(work_root) / "builds" / recipe.sha256
    freeze_structural_repair_expectation(build_root, recipe)

    snapshots = build_root / "inputs" / "structural"
    authority_snapshot = snapshots / "authority"
    materializer_snapshot = snapshots / "materializer"
    base_plan_snapshot = snapshots / "corrected-base-plan"
    water_snapshot = snapshots / "water"
    scientific_snapshot = snapshots / "scientific"
    _link_tree_immutable(authority_manifest_path.parent, authority_snapshot)
    _link_tree_immutable(
        materialization_path.parent,
        materializer_snapshot,
        skip=frozenset({materialization_path.name}),
    )
    materializer_document = json.loads(materialization_path.read_bytes())
    materializer_document["recipeSha256"] = recipe.sha256
    materializer_manifest = materializer_snapshot / materialization_path.name
    _write_immutable(materializer_manifest, _canonical_json(materializer_document))
    _link_tree_immutable(base_plan_path.parent, base_plan_snapshot)
    _link_tree_immutable(water_path.parent, water_snapshot)
    _link_tree_immutable(scientific_path.parent, scientific_snapshot)
    corrected_verify_snapshot = snapshots / "corrected-base-verification.json"
    _write_immutable(corrected_verify_snapshot, corrected_verification_path.read_bytes())

    artifacts = {
        "authority": _entry(
            build_root, authority_snapshot / authority_manifest_path.name
        ),
        "materializer": _entry(build_root, materializer_manifest),
        "correctedBasePlan": _entry(
            build_root, base_plan_snapshot / base_plan_path.name
        ),
        "correctedBaseVerification": _entry(
            build_root, corrected_verify_snapshot
        ),
        "waterTransaction": _entry(
            build_root, water_snapshot / water_path.name
        ),
        "scientificClosure": _entry(
            build_root, scientific_snapshot / scientific_path.name
        ),
    }
    verifier_inputs = {
        "format": 1,
        "recipeSha256": recipe.sha256,
        "artifacts": artifacts,
    }
    _write_immutable(
        build_root / "structural-verify-inputs.json",
        _canonical_json(verifier_inputs),
    )

    staged_water = {
        tuple(row["key"]): row["disposition"] for row in water["artifacts"]
    }
    for row in stage["files"]:
        role = row["role"]
        if role == "negative-height-overlay":
            layer = "height"
        elif role == "corrected-format1-water-candidate" and staged_water[
            tuple(row["key"])
        ] == "replacement":
            layer = "water"
        else:
            continue
        lod, cx, cz = row["key"]
        source = _safe_child(root, row["path"])
        destination = build_root / "chunks" / layer / str(lod) / f"{cx}_{cz}.lac"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if _sha256_file(destination) != row["sha256"]:
                raise ValueError(f"immutable preview overlay differs: {destination}")
        else:
            os.link(source, destination)

    corrected_document = json.loads(corrected_manifest_path.read_bytes())
    create_build_plan(
        load_base(),
        recipe.sha256,
        int(corrected_document["cookRev"]),
        work_root=work_root,
        base_manifest_path=corrected_manifest_path,
        base_manifest_sha256=corrected_manifest_sha256,
        base_out_root=config.pinned_content_root,
        manifest_format=2,
    )
    preview = materialize_preview(
        recipe.sha256,
        work_root=work_root,
        out_root=config.pinned_content_root,
    )
    return recipe.sha256, preview


@dataclass(frozen=True)
class Stage1Transaction:
    root: Path
    transaction_sha256: str
    manifest_path: Path
    manifest_sha256: str
    release_ready: bool = False


def run_stage1_transaction(
    config_path: Path = _DEFAULT_CONFIG,
    *,
    work_root: Path = DATA_WORK,
) -> Stage1Transaction:
    """Build every immutable pre-release Stage-1 artifact, resuming at step markers."""
    config = load_authority_config(config_path)
    _require_scientific_preconditions()
    authority_path = run_authority_cook(config_path, work_root=work_root)
    authority_manifest_path, authority_manifest_sha, authority_document = (
        _authority_manifest(authority_path)
    )
    inputs = _verify_inputs(config)
    identity, transaction_sha = _transaction_identity(config, authority_path)
    root = (
        Path(work_root)
        / "terrain-repair"
        / "taevaskoda-ahja-stage1-transaction"
        / transaction_sha
    )
    root.mkdir(parents=True, exist_ok=True)
    _write_immutable(root / "input-identity.json", _canonical_json(identity))
    _write_marker(
        root,
        "01-authority",
        transaction_sha,
        {
            "orchestration": os.path.relpath(authority_path, root),
            "orchestrationSha256": _sha256_file(authority_path),
            "manifest": os.path.relpath(authority_manifest_path, root),
            "manifestSha256": authority_manifest_sha,
            "pairCount": authority_document["authority"]["pairCount"],
        },
    )
    scientific_path, scientific_sha = _run_scientific_closure(
        root=root,
        transaction_sha256=transaction_sha,
        config=config,
        inputs=inputs,
        authority_manifest_sha256=authority_manifest_sha,
    )
    materialization_path, materialization_sha, materialization = _run_materialization(
        root=root,
        transaction_sha256=transaction_sha,
        config=config,
        inputs=inputs,
        authority_manifest_path=authority_manifest_path,
        authority_manifest_sha256=authority_manifest_sha,
        scientific_closure_sha256=scientific_sha,
    )
    base_plan_path, base_plan_sha, base_plan = _run_corrected_base(
        root=root,
        transaction_sha256=transaction_sha,
        config=config,
        materialization_path=materialization_path,
        materialization=materialization,
    )
    water_path, water_sha, water = _run_corrected_water(
        root=root,
        transaction_sha256=transaction_sha,
        config=config,
        inputs=inputs,
    )
    stage_path, stage_sha, stage = _run_staging(
        root=root,
        transaction_sha256=transaction_sha,
        config=config,
        materialization_path=materialization_path,
        materialization=materialization,
        base_plan_path=base_plan_path,
        base_plan=base_plan,
        water_path=water_path,
        water=water,
    )
    (
        corrected_manifest_path,
        corrected_manifest_sha,
        corrected_verification_path,
        corrected_verification_sha,
    ) = _run_corrected_base_release(
        root=root,
        transaction_sha256=transaction_sha,
        config=config,
        stage=stage,
    )
    preview_recipe_sha, preview_manifest_path = _run_preview_release(
        root=root,
        work_root=Path(work_root),
        config=config,
        inputs=inputs,
        authority_manifest_path=authority_manifest_path,
        scientific_path=scientific_path,
        materialization_path=materialization_path,
        base_plan_path=base_plan_path,
        water_path=water_path,
        water=water,
        stage=stage,
        corrected_manifest_path=corrected_manifest_path,
        corrected_manifest_sha256=corrected_manifest_sha,
        corrected_verification_path=corrected_verification_path,
    )
    manifest = {
        "format": 1,
        "transactionVersion": STAGE1_TRANSACTION_VERSION,
        "transactionSha256": transaction_sha,
        "inputIdentity": "input-identity.json",
        "authorityOrchestration": os.path.relpath(authority_path, root),
        "authorityOrchestrationSha256": _sha256_file(authority_path),
        "authorityManifest": os.path.relpath(authority_manifest_path, root),
        "authorityManifestSha256": authority_manifest_sha,
        "scientificClosure": scientific_path.relative_to(root).as_posix(),
        "scientificClosureSha256": scientific_sha,
        "materialization": materialization_path.relative_to(root).as_posix(),
        "materializationSha256": materialization_sha,
        "correctedBasePlan": base_plan_path.relative_to(root).as_posix(),
        "correctedBasePlanSha256": base_plan_sha,
        "correctedWater": water_path.relative_to(root).as_posix(),
        "correctedWaterSha256": water_sha,
        "stagedInventory": stage_path.relative_to(root).as_posix(),
        "stagedInventorySha256": stage_sha,
        "correctedBaseManifest": corrected_manifest_path.relative_to(root).as_posix(),
        "correctedBaseManifestSha256": corrected_manifest_sha,
        "correctedBaseVerification": corrected_verification_path.relative_to(root).as_posix(),
        "correctedBaseVerificationSha256": corrected_verification_sha,
        "previewRecipeSha256": preview_recipe_sha,
        "previewManifest": preview_manifest_path.relative_to(
            config.pinned_content_root
        ).as_posix(),
        "previewManifestSha256": _sha256_file(preview_manifest_path),
        "completionStatus": "verified-preview-release",
        "releaseReady": True,
    }
    manifest_path = root / "transaction.json"
    _write_immutable(manifest_path, _canonical_json(manifest))
    return Stage1Transaction(
        root=root,
        transaction_sha256=transaction_sha,
        manifest_path=manifest_path,
        manifest_sha256=_sha256_file(manifest_path),
        release_ready=True,
    )
