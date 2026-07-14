import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from assetgen.config import EncodeConfig, GridConfig
from assetgen.cook.chunkio import ChunkMeta, read_chunk
from assetgen.cook.encode import decode_quant16
from assetgen.height_geom import HeightChunkId
from assetgen.terrain.repair.base_transaction import (
    CORRECTED_ANCESTOR_SPLICE_VERSION,
    CORRECTED_QOFFSET_VERSION,
    CorrectedLod0Core,
    InheritedHeightChunk,
    build_corrected_base_transaction,
)


GRID = GridConfig(0, 0, 8, 4, (0, 1, 2, 3, 4), 8)
ENCODE = EncodeConfig("deflate", 0.01, 0.01, 19, 1)


class _SyntheticFormat1Source:
    manifest_sha256 = "7" * 64

    def __init__(
        self,
        *,
        qoffset: float = 0.0,
        independent_lods: bool = True,
    ) -> None:
        self._chunks: dict[HeightChunkId, InheritedHeightChunk] = {}
        for lod in range(5):
            for cz in range(-8, 9):
                for cx in range(-8, 9):
                    chunk = HeightChunkId(lod, cx, cz)
                    level = 10.0 + lod if independent_lods else 10.0
                    code = int(round((level - qoffset) / 0.01))
                    if code < 0:
                        raise ValueError("fixture base does not fit its quantizer")
                    codes = np.full((9, 9), code, np.uint16)
                    if chunk == HeightChunkId(4, 0, 0):
                        codes[7, 7] = int(round((55.0 - qoffset) / 0.01))
                    decoded = codes.astype(np.float32) * np.float32(0.01) + qoffset
                    artifact_sha = hashlib.sha256(f"lac:{lod}:{cx}:{cz}".encode()).hexdigest()
                    core_sha = hashlib.sha256(decoded[:-1, :-1].tobytes()).hexdigest()
                    self._chunks[chunk] = InheritedHeightChunk(
                        chunk=chunk,
                        meta=ChunkMeta(
                            layer="height",
                            lod=lod,
                            enc=1,
                            cx=cx,
                            cz=cz,
                            res=9,
                            count=0,
                            origin_e=float(cx * 8 * 4**lod),
                            origin_n=float(-cz * 8 * 4**lod),
                            qoffset=qoffset,
                            qscale=float(np.float32(0.01)),
                            flags=1,
                        ),
                        codes=codes,
                        decoded=decoded,
                        artifact_sha256=artifact_sha,
                        payload_sha256=hashlib.sha256(
                            f"payload:{lod}:{cx}:{cz}".encode()
                        ).hexdigest(),
                        decoded_core_sha256=core_sha,
                    )

    def contains(self, chunk: HeightChunkId) -> bool:
        return chunk in self._chunks

    def load(self, chunk: HeightChunkId) -> InheritedHeightChunk:
        return self._chunks[chunk]


def _decode_artifact(root: Path, relative_path: str) -> np.ndarray:
    meta, payload = read_chunk(root / relative_path)
    return decode_quant16(ENCODE, payload, meta.res, meta.qoffset, meta.qscale)


def test_masked_splice_preserves_independent_coarse_terrain_and_aprons(tmp_path) -> None:
    source = _SyntheticFormat1Source()
    corrected_chunk = HeightChunkId(0, 1, 1)
    values = np.full((8, 8), -1000.0, dtype=np.float32)
    values[:2, :2] = 9.75
    mask = np.zeros((8, 8), dtype=bool)
    mask[:2, :2] = True
    root = tmp_path / "first"

    transaction = build_corrected_base_transaction(
        source=source,
        staging_root=root,
        corrected_lod0={corrected_chunk: CorrectedLod0Core(values, mask)},
        grid=GRID,
        encode=ENCODE,
    )

    artifacts = {artifact.chunk: artifact for artifact in transaction.plan.artifacts}
    lod0 = transaction.plan.rungs[0]
    west = HeightChunkId(0, 0, 1)
    north = HeightChunkId(0, 1, 0)
    northwest = HeightChunkId(0, 0, 0)
    assert set(lod0.apron_candidates) == {corrected_chunk, west, north, northwest}
    assert set(lod0.replacements) == {corrected_chunk, west, north, northwest}
    for chunk in (west, north, northwest):
        artifact = artifacts[chunk]
        assert artifact.role == "masked-apron-only-replacement"
        assert artifact.core_codes_preserved
        decoded = _decode_artifact(root, artifact.relative_path)
        np.testing.assert_array_equal(decoded[:-1, :-1], source.load(chunk).decoded[:-1, :-1])
    np.testing.assert_allclose(
        _decode_artifact(root, artifacts[west].relative_path)[:2, -1],
        9.75,
        atol=0.005001,
    )

    # Each independently cooked coarse core changes only the transitively
    # affected reducer window; a distant discontinuity in the same LOD4 core is
    # a generic protected sentinel, not a place-specific exception.
    for lod in range(1, 5):
        chunk = HeightChunkId(lod, 0, 0)
        output = _decode_artifact(root, artifacts[chunk].relative_path)[:-1, :-1]
        affected = np.load(root / f"core-mask/{lod}/0_0.npy", allow_pickle=False)
        inherited = source.load(chunk).decoded[:-1, :-1]
        assert affected.any()
        np.testing.assert_array_equal(output[~affected], inherited[~affected])
    lod4 = _decode_artifact(root, artifacts[HeightChunkId(4, 0, 0)].relative_path)
    assert lod4[7, 7].tobytes() == source.load(HeightChunkId(4, 0, 0)).decoded[7, 7].tobytes()

    plan = json.loads(transaction.plan_path.read_bytes())
    row = next(item for item in plan["artifacts"] if item["chunk"] == [4, 0, 0])
    assert plan["ancestorSpliceVersion"] == CORRECTED_ANCESTOR_SPLICE_VERSION
    assert plan["qoffsetPolicy"] == CORRECTED_QOFFSET_VERSION
    assert row["affectedMask"]["sampleCount"] > 0
    assert row["affectedWindows"]["runCount"] > 0
    assert row["inheritedDecodedCoreSha256"] == source.load(HeightChunkId(4, 0, 0)).decoded_core_sha256
    assert row["quantizerDomain"] == "final-full-payload-core-and-east-south-southeast-aprons"
    assert row["outsideProof"]["bitExact"] is True
    assert row["outsideProof"]["inheritedDecodedSha256"] == row["outsideProof"]["outputDecodedSha256"]

    repeated = build_corrected_base_transaction(
        source=source,
        staging_root=tmp_path / "second",
        corrected_lod0={corrected_chunk: CorrectedLod0Core(values, mask)},
        grid=GRID,
        encode=ENCODE,
    )
    assert repeated.plan_sha256 == transaction.plan_sha256
    assert [artifact.artifact_sha256 for artifact in repeated.plan.artifacts] == [
        artifact.artifact_sha256 for artifact in transaction.plan.artifacts
    ]


def test_retune_uses_full_payload_only_when_outside_decodes_bit_exact(tmp_path) -> None:
    source = _SyntheticFormat1Source(qoffset=10.0, independent_lods=False)
    chunk = HeightChunkId(0, 1, 1)
    values = np.full((8, 8), 9.75, dtype=np.float32)
    mask = np.zeros((8, 8), dtype=bool)
    mask[3:5, 3:5] = True

    transaction = build_corrected_base_transaction(
        source=source,
        staging_root=tmp_path / "retuned",
        corrected_lod0={chunk: CorrectedLod0Core(values, mask)},
        grid=GRID,
        encode=ENCODE,
        maximum_lod=0,
    )

    assert transaction.plan.rungs[0].retuned_quantizers == (chunk,)
    artifact = next(item for item in transaction.plan.artifacts if item.chunk == chunk)
    assert not artifact.inherited_quantizer_preserved
    assert artifact.outside_proof.bit_exact
    output = _decode_artifact(tmp_path / "retuned", artifact.relative_path)
    np.testing.assert_array_equal(output[:-1, :-1][~mask], source.load(chunk).decoded[:-1, :-1][~mask])


def test_retune_fails_closed_when_quantizer_phase_would_move_unmasked_values(tmp_path) -> None:
    source = _SyntheticFormat1Source(qoffset=10.003, independent_lods=False)
    chunk = HeightChunkId(0, 1, 1)
    values = np.full((8, 8), 9.75, dtype=np.float32)
    mask = np.zeros((8, 8), dtype=bool)
    mask[3:5, 3:5] = True

    with pytest.raises(ValueError, match="outside the affected mask"):
        build_corrected_base_transaction(
            source=source,
            staging_root=tmp_path / "fail-closed",
            corrected_lod0={chunk: CorrectedLod0Core(values, mask)},
            grid=GRID,
            encode=ENCODE,
            maximum_lod=0,
        )


def test_untyped_or_empty_authority_is_rejected(tmp_path) -> None:
    source = _SyntheticFormat1Source()
    chunk = HeightChunkId(0, 1, 1)
    values = np.full((8, 8), 9.75, dtype=np.float32)
    with pytest.raises(TypeError, match="exact authority mask"):
        build_corrected_base_transaction(
            source=source,
            staging_root=tmp_path / "untyped",
            corrected_lod0={chunk: values},  # type: ignore[dict-item]
            grid=GRID,
            encode=ENCODE,
            maximum_lod=0,
        )
    with pytest.raises(ValueError, match="empty affected mask"):
        build_corrected_base_transaction(
            source=source,
            staging_root=tmp_path / "empty",
            corrected_lod0={chunk: CorrectedLod0Core(values, np.zeros((8, 8), bool))},
            grid=GRID,
            encode=ENCODE,
            maximum_lod=0,
        )
