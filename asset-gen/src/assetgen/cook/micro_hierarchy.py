"""Decoded-artifact hierarchy primitives for format-2 fine height.

The parent is derived from browser-equivalent decoded child values.  Synthesis
code is intentionally absent: this module is the packing/verifier boundary used
by both the Stage-1 fixture and the eventual production morphology generator.
"""
from __future__ import annotations

import hashlib
import os
import struct
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from ..height_geom import HeightChunkId, HeroCoverage

_MERKLE_LEAF = b"laas.micro.dependencies.leaf.v1\0"
_MERKLE_NODE = b"laas.micro.dependencies.node.v1\0"
_MERKLE_EMPTY = b"laas.micro.dependencies.empty.v1\0"
BOX_MEAN_REDUCER_VERSION = "reshape-mean-float64/1"


def dependency_merkle_root(
    dependencies: Iterable[tuple[HeightChunkId, str]],
) -> str:
    """Hash a unique chunk->artifact-SHA map independently of input ordering."""
    ordered = sorted(dependencies, key=lambda item: item[0])
    if len({chunk for chunk, _ in ordered}) != len(ordered):
        raise ValueError("dependency list contains duplicate chunk keys")
    level: list[bytes] = []
    for chunk, artifact_sha in ordered:
        try:
            digest = bytes.fromhex(artifact_sha)
        except ValueError as exc:
            raise ValueError(f"invalid artifact SHA for {chunk}") from exc
        if len(digest) != hashlib.sha256().digest_size:
            raise ValueError(f"artifact SHA for {chunk} is not SHA-256")
        identity = struct.pack("<bii", chunk.lod, chunk.cx, chunk.cz)
        level.append(hashlib.sha256(_MERKLE_LEAF + identity + digest).digest())
    if not level:
        return hashlib.sha256(_MERKLE_EMPTY).hexdigest()
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(_MERKLE_NODE + level[i] + level[i + 1]).digest()
            for i in range(0, len(level), 2)
        ]
    return level[0].hex()


def assemble_parent_source_memmap(
    path: Path,
    coverage: HeroCoverage,
    load_decoded: Callable[[HeightChunkId], np.ndarray],
    *,
    core_res: int = 2048,
    factor: int = 4,
) -> np.memmap:
    """Assemble the 4x4 child cores and the factor-wide southeast apron.

    Raster rows increase southward and columns increase eastward.  Each decoded
    chunk must include its one-sample apron, but only the first ``core_res``
    samples belong to its core.  The adjacent support chunks supply the four
    samples needed to derive the final parent sample, not merely the duplicated
    boundary sample.
    """
    if core_res < 1 or factor < 1:
        raise ValueError("core_res and factor must be positive")
    fine_cx0 = coverage.parent.cx * 4
    fine_cz0 = coverage.parent.cz * 4
    expected_published = {
        HeightChunkId(coverage.parent.lod - 1, fine_cx0 + dx, fine_cz0 + dz)
        for dz in range(4)
        for dx in range(4)
    }
    if set(coverage.published_fine) != expected_published:
        raise ValueError("coverage does not contain the parent's exact 4x4 children")

    corner = HeightChunkId(coverage.parent.lod - 1, fine_cx0 + 4, fine_cz0 + 4)
    expected_support = {
        HeightChunkId(coverage.parent.lod - 1, fine_cx0 + 4, fine_cz0 + dz)
        for dz in range(4)
    } | {
        HeightChunkId(coverage.parent.lod - 1, fine_cx0 + dx, fine_cz0 + 4)
        for dx in range(4)
    } | {corner}
    support = set(coverage.transient_support)
    if support != expected_support:
        missing = sorted(expected_support - support)
        extra = sorted(support - expected_support)
        raise ValueError(f"invalid transient support; missing={missing}, extra={extra}")

    expected_shape = (core_res + 1, core_res + 1)
    side = 4 * core_res + factor
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    mosaic = np.memmap(temporary, dtype=np.float32, mode="w+", shape=(side, side))

    def decoded(chunk: HeightChunkId) -> np.ndarray:
        values = np.asarray(load_decoded(chunk))
        if values.shape != expected_shape or values.dtype != np.float32:
            raise ValueError(
                f"decoded {chunk} must be float32 {expected_shape}, got "
                f"{values.dtype} {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"decoded {chunk} contains nonfinite height")
        return values

    try:
        for chunk in sorted(coverage.published_fine):
            dx, dz = chunk.cx - fine_cx0, chunk.cz - fine_cz0
            values = decoded(chunk)
            y0, x0 = dz * core_res, dx * core_res
            mosaic[y0 : y0 + core_res, x0 : x0 + core_res] = values[:core_res, :core_res]

        for dz in range(4):
            chunk = HeightChunkId(coverage.parent.lod - 1, fine_cx0 + 4, fine_cz0 + dz)
            values = decoded(chunk)
            y0 = dz * core_res
            mosaic[y0 : y0 + core_res, 4 * core_res :] = values[:core_res, :factor]
        for dx in range(4):
            chunk = HeightChunkId(coverage.parent.lod - 1, fine_cx0 + dx, fine_cz0 + 4)
            values = decoded(chunk)
            x0 = dx * core_res
            mosaic[4 * core_res :, x0 : x0 + core_res] = values[:factor, :core_res]
        mosaic[4 * core_res :, 4 * core_res :] = decoded(corner)[:factor, :factor]
        mosaic.flush()
        temporary.replace(path)
    except BaseException:
        del mosaic
        temporary.unlink(missing_ok=True)
        raise
    return mosaic


def box_mean_fixed(source: np.ndarray, *, factor: int = 4) -> np.ndarray:
    """Apply the versioned reshape/mean reducer to a factor-divisible raster."""
    values = np.asarray(source)
    if values.ndim != 2 or factor < 1:
        raise ValueError("box-mean source must be 2D and factor must be positive")
    rows, cols = values.shape
    if rows % factor or cols % factor:
        raise ValueError("box-mean source dimensions must be divisible by factor")
    return values.reshape(rows // factor, factor, cols // factor, factor).mean(
        axis=(1, 3), dtype=np.float64
    )


def box_mean4_striped(
    source: np.ndarray,
    *,
    factor: int = 4,
    stripe_rows: int = 64,
    output: np.ndarray | None = None,
) -> np.ndarray:
    """Reduce complete factor-square cells without materializing a 4-D mosaic."""
    values = np.asarray(source)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("parent source must be a square 2D raster")
    if factor < 1 or values.shape[0] % factor:
        raise ValueError("parent source side must be divisible by factor")
    if stripe_rows < 1:
        raise ValueError("stripe_rows must be positive")
    parent_res = values.shape[0] // factor
    if output is None:
        result = np.empty((parent_res, parent_res), dtype=np.float64)
    else:
        result = np.asarray(output)
        if result.shape != (parent_res, parent_res) or result.dtype != np.float64:
            raise ValueError("parent output must be a correctly sized float64 raster")
    for row0 in range(0, parent_res, stripe_rows):
        row1 = min(row0 + stripe_rows, parent_res)
        block = values[row0 * factor : row1 * factor, :]
        result[row0:row1] = box_mean_fixed(block, factor=factor)
    return result
