import hashlib
import json
from pathlib import Path

import numpy as np

from assetgen.config import load_base
from assetgen.cook.chunkio import ChunkMeta, write_chunk
from assetgen.cook.encode import encode_quant16_checked
from assetgen.height_geom import HeightChunkId
from assetgen.release import INDEX_RECORD_V1
from assetgen.terrain.repair.pinned_baseline import PinnedDecodedBaseline


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pinned_release(root: Path) -> tuple[Path, str, Path]:
    base = load_base()
    content_root = root / "content"
    records: list[tuple[int, int, int, int, int]] = []
    rows = np.arange(2049, dtype=np.float32)[:, None]
    cols = np.arange(2049, dtype=np.float32)[None, :]

    for cx, cz in ((0, 0), (0, 1), (1, 0), (1, 1)):
        # One continuous affine surface makes any wrong cross-chunk coordinate visible.
        values = (
            np.float32(100.0)
            + np.float32(0.01) * (np.float32(cx * 2048) + cols)
            + np.float32(0.02) * (np.float32(cz * 2048) + rows)
        )
        payload, qoffset, qscale = encode_quant16_checked(
            base.encode, values, requested_qscale=0.01
        )
        staging = root / f"height-{cx}-{cz}.lac"
        write_chunk(
            staging,
            ChunkMeta(
                layer="height",
                lod=0,
                enc=1,
                cx=cx,
                cz=cz,
                res=2049,
                count=0,
                origin_e=float(base.grid.anchor_e + cx * 2048),
                origin_n=float(base.grid.anchor_n - cz * 2048),
                qoffset=qoffset,
                qscale=qscale,
            ),
            payload,
        )
        digest = _sha256(staging)
        hash64 = int.from_bytes(bytes.fromhex(digest)[:8], "big")
        destination = (
            content_root
            / "c"
            / "height"
            / "0"
            / f"{cx}_{cz}.{digest[:8]}.bin"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(destination)
        records.append((0, cx, cz, destination.stat().st_size, hash64))

    manifest_dir = root / "release"
    index_path = manifest_dir / "index" / "height.bin"
    index_path.parent.mkdir(parents=True)
    index_blob = b"".join(
        INDEX_RECORD_V1.pack(*record) for record in sorted(records)
    )
    index_path.write_bytes(index_blob)
    manifest = {
        "format": 1,
        "anchor": {"e": base.grid.anchor_e, "n": base.grid.anchor_n},
        "chunkMeters": 2048,
        "chunkRes": 2048,
        "lodStep": 4,
        "codec": base.encode.codec,
        "layers": {
            "height": {
                "lods": [0],
                "count": len(records),
                "bytes": sum(record[3] for record in records),
                "index": "index/height.bin",
                "indexSha256": hashlib.sha256(index_blob).hexdigest(),
            }
        },
    }
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path, _sha256(manifest_path), content_root


def test_reconstructs_across_real_lac1_chunk_corner_with_bounded_cache(
    tmp_path: Path,
) -> None:
    base = load_base()
    manifest_path, manifest_sha256, content_root = _write_pinned_release(tmp_path)
    reader = PinnedDecodedBaseline(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        content_root=content_root,
        encode=base.encode,
        cache_chunks=2,
    )

    # LOD-2 (16,16) starts at the southeast corner of LOD0 (0,0), so its
    # two-cell Keys halo necessarily reads all four neighboring LOD0 chunks.
    result = reader.reconstruct(HeightChunkId(-2, 16, 16))
    collared = reader.reconstruct(HeightChunkId(-2, 16, 16), halo_samples=4)

    assert result.tile.height.shape == (512, 512)
    assert collared.tile.height.shape == (520, 520)
    np.testing.assert_array_equal(collared.tile.height[4:-4, 4:-4], result.tile.height)
    assert result.tile.valid.all()
    assert result.maximum_mean_error_m <= result.mean_error_limit_m
    assert reader.cached_chunk_count == 2
    assert [dependency.chunk for dependency in result.dependencies] == [
        HeightChunkId(0, 0, 0),
        HeightChunkId(0, 1, 0),
        HeightChunkId(0, 0, 1),
        HeightChunkId(0, 1, 1),
    ]
    assert all(len(dependency.sha256) == 64 for dependency in result.dependencies)
    assert all(
        len(dependency.decoded_sha256) == 64 for dependency in result.dependencies
    )
    assert len(result.dependency_root_sha256) == 64
    assert result.manifest_sha256 == manifest_sha256

    blocks = result.tile.height.reshape(128, 4, 128, 4)
    expected_rows = np.arange(128, dtype=np.float64)[:, None] + 2048.0
    expected_cols = np.arange(128, dtype=np.float64)[None, :] + 2048.0
    expected_authority = 100.0 + 0.01 * expected_cols + 0.02 * expected_rows
    np.testing.assert_allclose(
        blocks.mean(axis=(1, 3)), expected_authority, rtol=0.0, atol=2.0e-5
    )

    # The first fine center is 3/8 of a source cell northwest of its owner.
    first_index = 2048.0 - 3.0 / 8.0
    expected_first = 100.0 + 0.01 * first_index + 0.02 * first_index
    assert abs(result.tile.height[0, 0] - expected_first) <= 3.0e-5
