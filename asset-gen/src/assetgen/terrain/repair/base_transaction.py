"""Masked corrected format-1 height hierarchy transaction.

Format-1 LOD1-4 height chunks are independent DTM cooks, not a child-derived
pyramid.  A sparse qualified repair therefore splices decoded-child reductions
only into transitively affected parent samples.  Every other decoded sample is
inherited bit-for-bit, including unrelated terrain inside the same coarse chunk.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ...config import EncodeConfig, GridConfig
from ...cook.chunkio import ChunkMeta, read_chunk, write_chunk
from ...cook.encode import decode_quant16, encode_quantized16
from ...cook.micro_hierarchy import BOX_MEAN_REDUCER_VERSION, box_mean_fixed
from ...height_geom import HeightChunkId, parent_of
from ...release import IndexRecord, audit_base_release, read_v1_index


TRANSACTION_VERSION = "corrected-format1-height-transaction/2"
PLAN_FORMAT = 2
CORRECTED_ANCESTOR_SPLICE_VERSION = (
    "masked-child-reduction-splice-preserve-independent-dtm/1"
)
CORRECTED_QOFFSET_VERSION = (
    "preserve-inherited-or-retune-full-payload-outside-bitexact/1"
)
QOFFSET_POLICY = CORRECTED_QOFFSET_VERSION
MASK_FORMAT = "packed-bits-row-major-msb/1"
WINDOW_FORMAT = "row-run-half-open-json/1"
OUTSIDE_PROOF_VERSION = "decoded-f32-outside-mask-bitexact/1"
QUANTIZER_DOMAIN = "final-full-payload-core-and-east-south-southeast-aprons"
_MASK_MAGIC = b"TAM1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray, dtype: str) -> str:
    canonical = np.ascontiguousarray(values, dtype=dtype)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _chunk_key(chunk: HeightChunkId) -> tuple[int, int, int]:
    return chunk.lod, chunk.cx, chunk.cz


def _chunk_json(chunk: HeightChunkId) -> list[int]:
    return [chunk.lod, chunk.cx, chunk.cz]


def _canonical_chunks(chunks: set[HeightChunkId]) -> tuple[HeightChunkId, ...]:
    return tuple(sorted(chunks, key=_chunk_key))


def _immutable_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(f".{path.name}.{os.getpid()}.candidate")
    candidate.write_bytes(payload)
    if path.exists():
        if path.read_bytes() != payload:
            candidate.unlink(missing_ok=True)
            raise ValueError(f"immutable transaction artifact differs: {path}")
        candidate.unlink()
    else:
        candidate.replace(path)


@dataclass(frozen=True)
class CorrectedLod0Core:
    """Qualified LOD0 target values and their exact authority mask."""

    values: np.ndarray
    affected_mask: np.ndarray


@dataclass(frozen=True)
class HeightDependency:
    chunk: HeightChunkId
    artifact_sha256: str
    decoded_core_sha256: str
    role: str


@dataclass(frozen=True)
class InheritedHeightChunk:
    chunk: HeightChunkId
    meta: ChunkMeta
    codes: np.ndarray
    decoded: np.ndarray
    artifact_sha256: str
    payload_sha256: str
    decoded_core_sha256: str


class InheritedHeightSource(Protocol):
    manifest_sha256: str

    def contains(self, chunk: HeightChunkId) -> bool: ...

    def load(self, chunk: HeightChunkId) -> InheritedHeightChunk: ...


class AuditedFormat1HeightSource:
    """Integrity-audited format-1 height reader with a bounded decoded LRU."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        manifest_sha256: str,
        content_root: Path,
        encode: EncodeConfig,
        cache_chunks: int = 4,
    ) -> None:
        if isinstance(cache_chunks, bool) or not isinstance(cache_chunks, int):
            raise TypeError("cache_chunks must be an integer")
        if cache_chunks < 1:
            raise ValueError("cache_chunks must be positive")
        audit_base_release(manifest_path, manifest_sha256, content_root)
        manifest = json.loads(manifest_path.read_bytes())
        if manifest.get("format") != 1 or manifest.get("codec") != encode.codec:
            raise ValueError("corrected base requires a codec-compatible format-1 release")
        index_path = manifest_path.parent / manifest["layers"]["height"]["index"]
        self._records = {record.key: record for record in read_v1_index(index_path)}
        self._content_root = Path(content_root)
        self._encode = encode
        self._cache_chunks = cache_chunks
        self._cache: OrderedDict[HeightChunkId, InheritedHeightChunk] = OrderedDict()
        self.manifest_sha256 = manifest_sha256

    @property
    def cached_chunk_count(self) -> int:
        return len(self._cache)

    @property
    def cache_capacity(self) -> int:
        return self._cache_chunks

    def contains(self, chunk: HeightChunkId) -> bool:
        return _chunk_key(chunk) in self._records

    def _content_path(self, record: IndexRecord) -> Path:
        hash8 = ((record.hash64 >> 32) & 0xFFFFFFFF).to_bytes(4, "big").hex()
        return (
            self._content_root
            / "c"
            / "height"
            / str(record.lod)
            / f"{record.cx}_{record.cz}.{hash8}.bin"
        )

    def load(self, chunk: HeightChunkId) -> InheritedHeightChunk:
        cached = self._cache.pop(chunk, None)
        if cached is not None:
            self._cache[chunk] = cached
            return cached
        try:
            record = self._records[_chunk_key(chunk)]
        except KeyError as error:
            raise ValueError(f"inherited format-1 base lacks {chunk}") from error
        path = self._content_path(record)
        if not path.is_file() or path.stat().st_size != record.size:
            raise ValueError(f"inherited height content size mismatch: {path}")
        artifact_sha256 = _sha256_file(path)
        if int.from_bytes(bytes.fromhex(artifact_sha256)[:8], "big") != record.hash64:
            raise ValueError(f"inherited height content hash mismatch: {path}")
        meta, payload = read_chunk(path)
        if (
            meta.layer != "height"
            or meta.enc != 1
            or (meta.lod, meta.cx, meta.cz) != _chunk_key(chunk)
            or meta.res < 2
            or meta.count != 0
        ):
            raise ValueError(f"inherited height header mismatch: {path}")
        decoded = decode_quant16(
            self._encode, payload, meta.res, meta.qoffset, meta.qscale
        )
        codes64 = np.rint(
            (decoded.astype(np.float64) - float(meta.qoffset)) / float(meta.qscale)
        )
        if float(codes64.min()) < 0 or float(codes64.max()) > 65535:
            raise ValueError(f"cannot recover inherited quantized codes: {path}")
        codes = codes64.astype(np.uint16)
        recovered = codes.astype(np.float32) * meta.qscale + meta.qoffset
        if not np.array_equal(recovered, decoded):
            raise ValueError(f"inherited decoded codes are not recoverable exactly: {path}")
        result = InheritedHeightChunk(
            chunk=chunk,
            meta=meta,
            codes=codes,
            decoded=decoded,
            artifact_sha256=artifact_sha256,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            decoded_core_sha256=_array_sha256(decoded[:-1, :-1], "<f4"),
        )
        self._cache[chunk] = result
        while len(self._cache) > self._cache_chunks:
            self._cache.popitem(last=False)
        return result


@dataclass(frozen=True)
class MaskIdentity:
    format: str
    relative_path: str
    bytes: int
    sha256: str
    rows: int
    cols: int
    sample_count: int


@dataclass(frozen=True)
class WindowIdentity:
    format: str
    relative_path: str
    bytes: int
    sha256: str
    run_count: int


@dataclass(frozen=True)
class OutsideProof:
    version: str
    sample_count: int
    inherited_decoded_sha256: str
    output_decoded_sha256: str
    bit_exact: bool


@dataclass(frozen=True)
class StagedHeightArtifact:
    chunk: HeightChunkId
    role: str
    relative_path: str
    bytes: int
    artifact_sha256: str
    payload_sha256: str
    source_core_sha256: str
    decoded_values_sha256: str
    decoded_core_sha256: str
    qoffset: float
    qscale: float
    inherited_qoffset: float
    inherited_qscale: float
    core_codes_preserved: bool
    maximum_apron_error_m: float
    dependencies: tuple[HeightDependency, ...]
    masked_splice_version: str
    affected_mask: MaskIdentity
    affected_core_mask: MaskIdentity
    affected_windows: WindowIdentity
    inherited_decoded_core_sha256: str
    quantizer_domain: str
    outside_proof: OutsideProof
    inherited_quantizer_preserved: bool


@dataclass(frozen=True)
class ResolvedRung:
    lod: int
    changed_cores: tuple[HeightChunkId, ...]
    apron_candidates: tuple[HeightChunkId, ...]
    absent_candidates: tuple[HeightChunkId, ...]
    replacements: tuple[HeightChunkId, ...]
    promotions: tuple[HeightChunkId, ...]
    retuned_quantizers: tuple[HeightChunkId, ...]


@dataclass(frozen=True)
class ResolvedGrid:
    anchor_e: int
    anchor_n: int
    chunk_m: int
    chunk_res: int
    lod_step: int


@dataclass(frozen=True)
class ResolvedHierarchyPlan:
    format: int
    transaction_version: str
    source_manifest_sha256: str
    reducer_version: str
    ancestor_splice_version: str
    qoffset_policy: str
    grid: ResolvedGrid
    corrected_lod0: tuple[HeightChunkId, ...]
    corrected_core_sha256: tuple[tuple[HeightChunkId, str], ...]
    corrected_mask_sha256: tuple[tuple[HeightChunkId, str], ...]
    rungs: tuple[ResolvedRung, ...]
    artifacts: tuple[StagedHeightArtifact, ...]
    maximum_resident_decoded_chunks: int

    def to_json_dict(self) -> dict:
        def dependency(value: HeightDependency) -> dict:
            return {
                "chunk": _chunk_json(value.chunk),
                "artifactSha256": value.artifact_sha256,
                "decodedCoreSha256": value.decoded_core_sha256,
                "role": value.role,
            }

        def mask(value: MaskIdentity) -> dict:
            return {
                "format": value.format,
                "path": value.relative_path,
                "bytes": value.bytes,
                "sha256": value.sha256,
                "rows": value.rows,
                "cols": value.cols,
                "sampleCount": value.sample_count,
            }

        def windows(value: WindowIdentity) -> dict:
            return {
                "format": value.format,
                "path": value.relative_path,
                "bytes": value.bytes,
                "sha256": value.sha256,
                "runCount": value.run_count,
            }

        def artifact(value: StagedHeightArtifact) -> dict:
            excluded = {
                "chunk",
                "dependencies",
                "masked_splice_version",
                "affected_mask",
                "affected_core_mask",
                "affected_windows",
                "inherited_decoded_core_sha256",
                "quantizer_domain",
                "outside_proof",
                "inherited_quantizer_preserved",
            }
            base = {key: item for key, item in asdict(value).items() if key not in excluded}
            proof = value.outside_proof
            return {
                **base,
                "chunk": _chunk_json(value.chunk),
                "dependencies": [dependency(item) for item in value.dependencies],
                "maskedSpliceVersion": value.masked_splice_version,
                "affectedMask": mask(value.affected_mask),
                "affectedCoreMask": mask(value.affected_core_mask),
                "affectedWindows": windows(value.affected_windows),
                "inheritedDecodedCoreSha256": value.inherited_decoded_core_sha256,
                "quantizerDomain": value.quantizer_domain,
                "outsideProof": {
                    "version": proof.version,
                    "sampleCount": proof.sample_count,
                    "inheritedDecodedSha256": proof.inherited_decoded_sha256,
                    "outputDecodedSha256": proof.output_decoded_sha256,
                    "bitExact": proof.bit_exact,
                },
                "inheritedQuantizerPreserved": value.inherited_quantizer_preserved,
            }

        return {
            "format": self.format,
            "transactionVersion": self.transaction_version,
            "sourceManifestSha256": self.source_manifest_sha256,
            "reducerVersion": self.reducer_version,
            "ancestorSpliceVersion": self.ancestor_splice_version,
            "qoffsetPolicy": self.qoffset_policy,
            "grid": {
                "anchorE": self.grid.anchor_e,
                "anchorN": self.grid.anchor_n,
                "chunkMeters": self.grid.chunk_m,
                "chunkRes": self.grid.chunk_res,
                "lodStep": self.grid.lod_step,
            },
            "correctedLod0": [_chunk_json(value) for value in self.corrected_lod0],
            "correctedCoreSha256": [
                {"chunk": _chunk_json(chunk), "sha256": digest}
                for chunk, digest in self.corrected_core_sha256
            ],
            "correctedMaskSha256": [
                {"chunk": _chunk_json(chunk), "sha256": digest}
                for chunk, digest in self.corrected_mask_sha256
            ],
            "rungs": [
                {
                    "lod": rung.lod,
                    "changedCores": [_chunk_json(value) for value in rung.changed_cores],
                    "apronCandidates": [_chunk_json(value) for value in rung.apron_candidates],
                    "absentCandidates": [_chunk_json(value) for value in rung.absent_candidates],
                    "replacements": [_chunk_json(value) for value in rung.replacements],
                    "promotions": [_chunk_json(value) for value in rung.promotions],
                    "retunedQuantizers": [_chunk_json(value) for value in rung.retuned_quantizers],
                }
                for rung in self.rungs
            ],
            "artifacts": [artifact(value) for value in self.artifacts],
            "memory": {
                "maximumResidentDecodedChunks": self.maximum_resident_decoded_chunks,
                "maskedCoreStorage": "disk-backed-npy",
            },
        }


@dataclass(frozen=True)
class CorrectedBaseTransaction:
    plan: ResolvedHierarchyPlan
    plan_path: Path
    plan_sha256: str


@dataclass(frozen=True)
class _CoreState:
    chunk: HeightChunkId
    decoded_core_path: Path
    affected_core_mask_path: Path
    decoded_core_sha256: str
    artifact_sha256: str


@dataclass(frozen=True)
class _DesiredCore:
    values: np.ndarray
    affected: np.ndarray
    source_core_sha256: str
    dependencies: tuple[HeightDependency, ...]
    role: str


def _write_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as output:
        np.save(output, values, allow_pickle=False)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _mask_blob(mask: np.ndarray) -> bytes:
    values = np.ascontiguousarray(mask, dtype=np.bool_)
    rows, cols = values.shape
    return _MASK_MAGIC + struct.pack("<II", rows, cols) + np.packbits(
        values.reshape(-1), bitorder="big"
    ).tobytes()


def _row_runs(mask: np.ndarray) -> list[list[int]]:
    values = np.asarray(mask, dtype=bool)
    runs: list[list[int]] = []
    for row in range(values.shape[0]):
        padded = np.pad(values[row].astype(np.int8), (1, 1))
        edges = np.flatnonzero(np.diff(padded))
        for index in range(0, len(edges), 2):
            runs.append([row, int(edges[index]), int(edges[index + 1])])
    return runs


def _write_mask_identity(root: Path, name: str, mask: np.ndarray) -> MaskIdentity:
    values = np.ascontiguousarray(mask, dtype=bool)
    path = root / "masks" / f"{name}.tam"
    payload = _mask_blob(values)
    _immutable_write(path, payload)
    return MaskIdentity(
        MASK_FORMAT,
        path.relative_to(root).as_posix(),
        len(payload),
        hashlib.sha256(payload).hexdigest(),
        values.shape[0],
        values.shape[1],
        int(np.count_nonzero(values)),
    )


def _write_window_identity(root: Path, name: str, mask: np.ndarray) -> WindowIdentity:
    runs = _row_runs(mask)
    payload = (json.dumps(runs, separators=(",", ":")) + "\n").encode()
    path = root / "windows" / f"{name}.json"
    _immutable_write(path, payload)
    return WindowIdentity(
        WINDOW_FORMAT,
        path.relative_to(root).as_posix(),
        len(payload),
        hashlib.sha256(payload).hexdigest(),
        len(runs),
    )


def _selected_offset(values: np.ndarray, qscale: float) -> float:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    qoffset = float(np.floor(minimum - 1.0))
    codes = np.rint((np.asarray([minimum, maximum]) - qoffset) / qscale)
    if float(codes.min()) < 0 or float(codes.max()) > 65535:
        raise OverflowError(
            f"height domain {minimum}..{maximum} cannot fit qscale {qscale}"
        )
    if float(np.float32(qoffset)) != qoffset:
        raise ValueError("selected qoffset is not exactly representable as float32")
    return qoffset


def _quantize_values(values: np.ndarray, qoffset: float, qscale: float) -> np.ndarray | None:
    codes = np.rint((np.asarray(values, dtype=np.float64) - qoffset) / qscale)
    if not np.isfinite(codes).all() or float(codes.min()) < 0 or float(codes.max()) > 65535:
        return None
    return codes.astype(np.uint16)


def _selected_sha(values: np.ndarray, select: np.ndarray) -> str:
    data = np.asarray(values, dtype="<f4")
    mask = np.asarray(select, dtype=bool)
    digest = hashlib.sha256()
    digest.update(struct.pack("<IIQ", data.shape[0], data.shape[1], int(mask.sum())))
    for row in range(data.shape[0]):
        digest.update(np.ascontiguousarray(data[row, mask[row]], dtype="<f4").tobytes())
    return digest.hexdigest()


def _candidate_closure(changed: set[HeightChunkId]) -> set[HeightChunkId]:
    return {
        HeightChunkId(chunk.lod, chunk.cx + dx, chunk.cz + dz)
        for chunk in changed
        for dx, dz in ((0, 0), (-1, 0), (0, -1), (-1, -1))
    }


def _origin(grid: GridConfig, chunk: HeightChunkId) -> tuple[float, float]:
    footprint = grid.chunk_m * grid.lod_step**chunk.lod
    return (
        float(grid.anchor_e + chunk.cx * footprint),
        float(grid.anchor_n - chunk.cz * footprint),
    )


def _load_state_core(state: _CoreState) -> np.ndarray:
    return np.load(state.decoded_core_path, mmap_mode="r", allow_pickle=False)


def _load_state_mask(state: _CoreState) -> np.ndarray:
    return np.load(state.affected_core_mask_path, mmap_mode="r", allow_pickle=False)


def _state_dependency(state: _CoreState) -> HeightDependency:
    return HeightDependency(
        state.chunk,
        state.artifact_sha256,
        state.decoded_core_sha256,
        "decoded-child-masked-window",
    )


def _payload_target(
    *,
    chunk: HeightChunkId,
    desired: _DesiredCore | None,
    states: Mapping[HeightChunkId, _CoreState],
    inherited: InheritedHeightChunk,
) -> tuple[np.ndarray, np.ndarray]:
    core_res = inherited.meta.res - 1
    target = np.array(inherited.decoded, dtype=np.float64, copy=True)
    affected = np.zeros(target.shape, dtype=bool)
    if desired is not None:
        target[:-1, :-1][desired.affected] = desired.values[desired.affected]
        affected[:-1, :-1] = desired.affected

    east = states.get(HeightChunkId(chunk.lod, chunk.cx + 1, chunk.cz))
    if east is not None:
        east_mask = np.asarray(_load_state_mask(east))[:, 0]
        if east_mask.any():
            target[:-1, -1][east_mask] = np.asarray(_load_state_core(east))[:, 0][east_mask]
            affected[:-1, -1] = east_mask
    south = states.get(HeightChunkId(chunk.lod, chunk.cx, chunk.cz + 1))
    if south is not None:
        south_mask = np.asarray(_load_state_mask(south))[0, :]
        if south_mask.any():
            target[-1, :-1][south_mask] = np.asarray(_load_state_core(south))[0, :][south_mask]
            affected[-1, :-1] = south_mask
    southeast = states.get(HeightChunkId(chunk.lod, chunk.cx + 1, chunk.cz + 1))
    if southeast is not None and bool(np.asarray(_load_state_mask(southeast))[0, 0]):
        target[-1, -1] = float(np.asarray(_load_state_core(southeast))[0, 0])
        affected[-1, -1] = True
    if target.shape != (core_res + 1, core_res + 1):
        raise AssertionError("payload target shape changed")
    return target, affected


def _encode_masked_payload(
    *,
    chunk: HeightChunkId,
    target: np.ndarray,
    affected: np.ndarray,
    inherited: InheritedHeightChunk,
) -> tuple[np.ndarray, np.ndarray, float, bool, OutsideProof]:
    qscale = float(inherited.meta.qscale)
    qoffset = float(inherited.meta.qoffset)
    changed_codes = _quantize_values(target[affected], qoffset, qscale)
    preserved = changed_codes is not None
    if preserved:
        codes = np.array(inherited.codes, copy=True)
        codes[affected] = changed_codes
    else:
        qoffset = _selected_offset(target, qscale)
        full_codes = _quantize_values(target, qoffset, qscale)
        if full_codes is None:
            raise OverflowError(f"final corrected payload cannot fit quantizer on {chunk}")
        codes = full_codes
    decoded = codes.astype(np.float32) * qscale + qoffset
    outside = ~affected
    inherited_sha = _selected_sha(inherited.decoded, outside)
    output_sha = _selected_sha(decoded, outside)
    exact = np.array_equal(decoded[outside], inherited.decoded[outside])
    if not exact:
        raise ValueError(
            f"retuned quantizer changes decoded terrain outside the affected mask on {chunk}"
        )
    proof = OutsideProof(
        OUTSIDE_PROOF_VERSION,
        int(np.count_nonzero(outside)),
        inherited_sha,
        output_sha,
        True,
    )
    return codes, decoded, qoffset, preserved, proof


class _StagedStore:
    def __init__(self, root: Path, encode: EncodeConfig, core_res: int) -> None:
        self.root = root
        self.encode = encode
        self.core_res = core_res
        self.artifacts: dict[HeightChunkId, StagedHeightArtifact] = {}

    def write(
        self,
        *,
        chunk: HeightChunkId,
        inherited: InheritedHeightChunk,
        desired: _DesiredCore | None,
        target: np.ndarray,
        affected: np.ndarray,
        grid: GridConfig,
    ) -> tuple[StagedHeightArtifact, _CoreState]:
        codes, decoded, qoffset, quantizer_preserved, proof = _encode_masked_payload(
            chunk=chunk, target=target, affected=affected, inherited=inherited
        )
        qscale = float(inherited.meta.qscale)
        origin_e, origin_n = _origin(grid, chunk)
        meta = ChunkMeta(
            layer="height",
            lod=chunk.lod,
            enc=1,
            cx=chunk.cx,
            cz=chunk.cz,
            res=self.core_res + 1,
            count=0,
            origin_e=origin_e,
            origin_n=origin_n,
            qoffset=qoffset,
            qscale=qscale,
            flags=inherited.meta.flags,
        )
        payload = encode_quantized16(self.encode, codes)
        path = self.root / "chunks" / str(chunk.lod) / f"{chunk.cx}_{chunk.cz}.lac"
        candidate = path.with_name(f".{path.name}.{os.getpid()}.candidate")
        write_chunk(candidate, meta, payload)
        if path.exists():
            if path.read_bytes() != candidate.read_bytes():
                candidate.unlink(missing_ok=True)
                raise ValueError(f"immutable corrected chunk differs: {path}")
            candidate.unlink()
        else:
            candidate.replace(path)

        name = f"{chunk.lod}_{chunk.cx}_{chunk.cz}"
        core_mask = np.ascontiguousarray(affected[:-1, :-1])
        affected_identity = _write_mask_identity(self.root, name, affected)
        core_identity = _write_mask_identity(self.root, f"{name}.core", core_mask)
        window_identity = _write_window_identity(self.root, name, affected)
        decoded_core_path = self.root / "decoded-core" / str(chunk.lod) / f"{chunk.cx}_{chunk.cz}.npy"
        core_mask_path = self.root / "core-mask" / str(chunk.lod) / f"{chunk.cx}_{chunk.cz}.npy"
        _write_npy(decoded_core_path, np.ascontiguousarray(decoded[:-1, :-1], dtype="<f4"))
        _write_npy(core_mask_path, core_mask)

        dependencies = desired.dependencies if desired is not None else ()
        inherited_dependency = HeightDependency(
            chunk,
            inherited.artifact_sha256,
            inherited.decoded_core_sha256,
            "inherited-independent-dtm-core",
        )
        dependencies = tuple(
            sorted(
                (inherited_dependency, *dependencies),
                key=lambda item: (_chunk_key(item.chunk), item.role),
            )
        )
        apron_mask = affected.copy()
        apron_mask[:-1, :-1] = False
        apron_error = (
            float(np.max(np.abs(decoded[apron_mask] - target[apron_mask])))
            if apron_mask.any()
            else 0.0
        )
        allowance = qscale * 0.5 + 2.0 * abs(
            float(np.spacing(np.float32(np.max(np.abs(decoded)))))
        )
        if apron_error > allowance:
            raise AssertionError(
                f"decoded apron error {apron_error} exceeds {allowance} on {chunk}"
            )
        artifact_sha256 = _sha256_file(path)
        decoded_core_sha256 = _array_sha256(decoded[:-1, :-1], "<f4")
        artifact = StagedHeightArtifact(
            chunk=chunk,
            role=desired.role if desired is not None else "masked-apron-only-replacement",
            relative_path=path.relative_to(self.root).as_posix(),
            bytes=path.stat().st_size,
            artifact_sha256=artifact_sha256,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            source_core_sha256=(
                desired.source_core_sha256
                if desired is not None
                else inherited.decoded_core_sha256
            ),
            decoded_values_sha256=_array_sha256(decoded, "<f4"),
            decoded_core_sha256=decoded_core_sha256,
            qoffset=qoffset,
            qscale=qscale,
            inherited_qoffset=float(inherited.meta.qoffset),
            inherited_qscale=float(inherited.meta.qscale),
            core_codes_preserved=desired is None,
            maximum_apron_error_m=apron_error,
            dependencies=dependencies,
            masked_splice_version=CORRECTED_ANCESTOR_SPLICE_VERSION,
            affected_mask=affected_identity,
            affected_core_mask=core_identity,
            affected_windows=window_identity,
            inherited_decoded_core_sha256=inherited.decoded_core_sha256,
            quantizer_domain=QUANTIZER_DOMAIN,
            outside_proof=proof,
            inherited_quantizer_preserved=quantizer_preserved,
        )
        self.artifacts[chunk] = artifact
        return artifact, _CoreState(
            chunk,
            decoded_core_path,
            core_mask_path,
            decoded_core_sha256,
            artifact_sha256,
        )


def _resolve_rung(
    *,
    lod: int,
    desired_cores: Mapping[HeightChunkId, _DesiredCore],
    source: InheritedHeightSource,
    store: _StagedStore,
    grid: GridConfig,
) -> tuple[ResolvedRung, dict[HeightChunkId, _CoreState]]:
    changed = set(desired_cores)
    candidate_set = _candidate_closure(changed)
    absent = {chunk for chunk in candidate_set if not source.contains(chunk)}
    if changed & absent:
        raise ValueError(f"corrected hierarchy chunks are absent: {sorted(changed & absent)}")
    candidates = candidate_set - absent
    states: dict[HeightChunkId, _CoreState] = {}
    replacements: set[HeightChunkId] = set()
    retuned: set[HeightChunkId] = set()

    # Aprons depend only on east/south/southeast cores, so reverse raster order
    # makes every dependency final before its west/north owner is encoded.
    for chunk in sorted(candidates, key=lambda item: (item.cz, item.cx), reverse=True):
        inherited = source.load(chunk)
        desired = desired_cores.get(chunk)
        target, affected = _payload_target(
            chunk=chunk, desired=desired, states=states, inherited=inherited
        )
        if not affected.any():
            continue
        artifact, state = store.write(
            chunk=chunk,
            inherited=inherited,
            desired=desired,
            target=target,
            affected=affected,
            grid=grid,
        )
        states[chunk] = state
        replacements.add(chunk)
        if not artifact.inherited_quantizer_preserved:
            retuned.add(chunk)

    missing_states = changed - set(states)
    if missing_states:
        raise AssertionError(f"changed cores were not staged: {sorted(missing_states)}")
    return (
        ResolvedRung(
            lod=lod,
            changed_cores=_canonical_chunks(changed),
            apron_candidates=_canonical_chunks(candidates),
            absent_candidates=_canonical_chunks(absent),
            replacements=_canonical_chunks(replacements),
            promotions=(),
            retuned_quantizers=_canonical_chunks(retuned),
        ),
        {chunk: states[chunk] for chunk in changed},
    )


def _derive_parent_desired(
    *,
    child_states: Mapping[HeightChunkId, _CoreState],
    source: InheritedHeightSource,
    core_res: int,
) -> dict[HeightChunkId, _DesiredCore]:
    if core_res % 4:
        raise ValueError("parent reduction requires a four-way divisible core")
    quarter = core_res // 4
    parents: dict[HeightChunkId, tuple[np.ndarray, np.ndarray, list[HeightDependency]]] = {}
    for child in sorted(child_states, key=_chunk_key):
        state = child_states[child]
        parent = parent_of(child)
        if not source.contains(parent):
            raise ValueError(f"inherited base lacks required ancestor {parent}")
        if parent not in parents:
            inherited = source.load(parent)
            parents[parent] = (
                np.array(inherited.decoded[:-1, :-1], dtype=np.float64, copy=True),
                np.zeros((core_res, core_res), dtype=bool),
                [],
            )
        values, affected, dependencies = parents[parent]
        dx = child.cx - parent.cx * 4
        dz = child.cz - parent.cz * 4
        if dx not in range(4) or dz not in range(4):
            raise AssertionError(f"invalid child-parent relation: {child} -> {parent}")
        child_values = np.asarray(_load_state_core(state))
        child_mask = np.asarray(_load_state_mask(state), dtype=bool)
        reduced_values = box_mean_fixed(child_values, factor=4)
        reduced_mask = child_mask.reshape(quarter, 4, quarter, 4).any(axis=(1, 3))
        row = slice(dz * quarter, (dz + 1) * quarter)
        col = slice(dx * quarter, (dx + 1) * quarter)
        region = values[row, col]
        region[reduced_mask] = reduced_values[reduced_mask]
        affected[row, col] |= reduced_mask
        dependencies.append(_state_dependency(state))

    result: dict[HeightChunkId, _DesiredCore] = {}
    for parent, (values, affected, dependencies) in parents.items():
        if not affected.any():
            raise AssertionError(f"parent {parent} has no transitive affected window")
        result[parent] = _DesiredCore(
            values,
            affected,
            _array_sha256(values, "<f8"),
            tuple(sorted(dependencies, key=lambda item: _chunk_key(item.chunk))),
            "masked-decoded-child-ancestor-splice",
        )
    return result


def build_corrected_base_transaction(
    *,
    source: InheritedHeightSource,
    staging_root: Path,
    corrected_lod0: Mapping[HeightChunkId, CorrectedLod0Core],
    grid: GridConfig,
    encode: EncodeConfig,
    maximum_lod: int = 4,
) -> CorrectedBaseTransaction:
    """Build a sparse, immutable corrected format-1 closure.

    ``affected_mask`` is the sole authority boundary. Values outside it are
    ignored at LOD0, and each coarser rung receives only the transitively
    affected 4x4 decoded-child reductions spliced into its independent DTM core.
    """
    if not corrected_lod0:
        raise ValueError("corrected transaction requires at least one LOD0 core")
    if maximum_lod not in range(0, 5):
        raise ValueError("format-1 corrected hierarchy supports maximum LOD 0..4")
    if grid.lod_step != 4 or grid.chunk_res < 4 or grid.chunk_res % 4:
        raise ValueError("corrected hierarchy requires a four-way divisible grid")
    staging_root = Path(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    store = _StagedStore(staging_root, encode, grid.chunk_res)
    desired_cores: dict[HeightChunkId, _DesiredCore] = {}
    corrected_hashes: list[tuple[HeightChunkId, str]] = []
    corrected_mask_hashes: list[tuple[HeightChunkId, str]] = []
    expected_shape = (grid.chunk_res, grid.chunk_res)
    for chunk in sorted(corrected_lod0, key=_chunk_key):
        if chunk.lod != 0 or not source.contains(chunk):
            raise ValueError(f"invalid or absent corrected LOD0 chunk {chunk}")
        correction = corrected_lod0[chunk]
        if not isinstance(correction, CorrectedLod0Core):
            raise TypeError(
                "corrected_lod0 values must be CorrectedLod0Core with an exact authority mask"
            )
        values = np.asarray(correction.values)
        affected = np.asarray(correction.affected_mask)
        if values.shape != expected_shape or affected.shape != expected_shape:
            raise ValueError(f"corrected core {chunk} must have shape {expected_shape}")
        if affected.dtype != np.bool_:
            raise TypeError(f"corrected core {chunk} affected_mask must be bool")
        if not affected.any():
            raise ValueError(f"corrected core {chunk} has an empty affected mask")
        if not np.isfinite(values[affected]).all():
            raise ValueError(f"corrected core {chunk} has nonfinite affected values")
        inherited = source.load(chunk)
        target = np.array(inherited.decoded[:-1, :-1], dtype=np.float64, copy=True)
        target[affected] = values[affected]
        value_sha = _array_sha256(values, "<f8")
        mask_sha = hashlib.sha256(_mask_blob(affected)).hexdigest()
        corrected_hashes.append((chunk, value_sha))
        corrected_mask_hashes.append((chunk, mask_sha))
        desired_cores[chunk] = _DesiredCore(
            target,
            np.array(affected, copy=True),
            value_sha,
            (),
            "qualified-masked-corrected-core",
        )

    rungs: list[ResolvedRung] = []
    for lod in range(maximum_lod + 1):
        rung, changed_states = _resolve_rung(
            lod=lod,
            desired_cores=desired_cores,
            source=source,
            store=store,
            grid=grid,
        )
        rungs.append(rung)
        if lod == maximum_lod:
            break
        desired_cores = _derive_parent_desired(
            child_states=changed_states,
            source=source,
            core_res=grid.chunk_res,
        )

    plan = ResolvedHierarchyPlan(
        format=PLAN_FORMAT,
        transaction_version=TRANSACTION_VERSION,
        source_manifest_sha256=source.manifest_sha256,
        reducer_version=BOX_MEAN_REDUCER_VERSION,
        ancestor_splice_version=CORRECTED_ANCESTOR_SPLICE_VERSION,
        qoffset_policy=QOFFSET_POLICY,
        grid=ResolvedGrid(
            anchor_e=grid.anchor_e,
            anchor_n=grid.anchor_n,
            chunk_m=grid.chunk_m,
            chunk_res=grid.chunk_res,
            lod_step=grid.lod_step,
        ),
        corrected_lod0=tuple(chunk for chunk, _ in corrected_hashes),
        corrected_core_sha256=tuple(corrected_hashes),
        corrected_mask_sha256=tuple(corrected_mask_hashes),
        rungs=tuple(rungs),
        artifacts=tuple(store.artifacts[chunk] for chunk in sorted(store.artifacts, key=_chunk_key)),
        maximum_resident_decoded_chunks=(
            source.cache_capacity if isinstance(source, AuditedFormat1HeightSource) else 0
        ),
    )
    plan_bytes = (json.dumps(plan.to_json_dict(), indent=2, sort_keys=True) + "\n").encode()
    plan_path = staging_root / "resolved-hierarchy-plan.json"
    _immutable_write(plan_path, plan_bytes)
    return CorrectedBaseTransaction(plan, plan_path, hashlib.sha256(plan_bytes).hexdigest())
