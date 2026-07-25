#!/usr/bin/env python3
"""Measure low-rank first-event reconstruction on a periodic GCRP/v4 atlas.

This is an offline representation gate.  It never changes or approximates the
runtime asset.  The report answers whether a fixed-rank direct line-field codec
is even plausible before any shader implementation is considered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np


RANKS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)


def u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def f32(data: bytes, offset: int) -> float:
    return struct.unpack_from("<f", data, offset)[0]


def parse_profile(path: Path) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if data[:4] != b"GCRP" or u32(data, 4) != 4:
        raise ValueError("rank audit requires a GCRP/v4 profile")

    stored_w = u32(data, 12)
    stored_h = u32(data, 16)
    columns = u32(data, 20)
    rows = u32(data, 24)
    slice_count = u32(data, 28)
    texel_bytes = u32(data, 32)
    payload_offset = u32(data, 36)
    interior_w = u32(data, 44)
    interior_h = u32(data, 48)
    gutter = u32(data, 52)
    top_h = f32(data, 56)
    tile_size_x = f32(data, 68)
    tile_size_z = f32(data, 72)
    header_bytes = u32(data, 76)
    if texel_bytes != 8 or stored_w != interior_w + 2 * gutter or stored_h != interior_h + 2 * gutter:
        raise ValueError("non-canonical GCRP/v4 layout")

    atlas_w = stored_w * columns
    atlas_h = stored_h * rows
    payload_count = atlas_w * atlas_h * 4
    texels = np.frombuffer(
        data,
        dtype="<u2",
        count=payload_count,
        offset=payload_offset,
    ).reshape(atlas_h, atlas_w, 4)

    events = np.empty((slice_count, interior_h, interior_w, 4), dtype=np.uint16)
    directions = np.empty((slice_count, 5), dtype=np.float64)
    for index in range(slice_count):
        tile_x = (index % columns) * stored_w + gutter
        tile_y = (index // columns) * stored_h + gutter
        events[index] = texels[tile_y : tile_y + interior_h, tile_x : tile_x + interior_w]
        directions[index] = struct.unpack_from("<5f", data, header_bytes + index * 64)

    metadata: dict[str, object] = {
        "source": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "sliceCount": slice_count,
        "interior": [interior_w, interior_h],
        "topH": top_h,
        "tileSize": [tile_size_x, tile_size_z],
        "directions": directions.tolist(),
    }
    return metadata, events, directions


def centered_roll(events: np.ndarray, directions: np.ndarray, top_h: float, tile_size: float) -> np.ndarray:
    """Move the stored top-plane phase to the botanical middle plane.

    Integer texel rolls intentionally avoid inventing filtered events.  The
    residual sub-texel phase is reported as a quantisation bound.
    """

    result = np.empty_like(events)
    height, width = events.shape[1:3]
    reference_drop = top_h * 0.5
    for index, direction in enumerate(directions):
        dx, dy, dz = direction[:3]
        shift_x = (dx / -dy) * reference_drop / tile_size * width
        shift_z = (dz / -dy) * reference_drop / tile_size * height
        # Atlas V increases as world Z decreases.
        result[index] = np.roll(
            events[index],
            shift=(-int(round(shift_z)), int(round(shift_x))),
            axis=(0, 1),
        )
    return result


def eigensystem(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    gram = matrix @ matrix.T
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    total = float(values.sum())
    return values, vectors, total


def reconstruction(vectors: np.ndarray, matrix: np.ndarray, rank: int) -> np.ndarray:
    basis = vectors[:, :rank]
    return basis @ (basis.T @ matrix)


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=float)
    parser.add_argument("--no-center", action="store_true")
    args = parser.parse_args()

    metadata, raw_events, directions = parse_profile(args.source.resolve())
    events = raw_events if args.no_center else centered_roll(
        raw_events,
        directions,
        float(metadata["topH"]),
        args.tile_size or float(metadata["tileSize"][0]),
    )

    slices = events.shape[0]
    samples = events.shape[1] * events.shape[2]
    hit = events[..., 0] != 65535
    coverage = hit.reshape(slices, samples).astype(np.float64)

    depth_u = events[..., 0].astype(np.float64) / 65534.0
    depth_u[~hit] = 0.0
    t_min = directions[:, 3, None, None]
    t_span = (directions[:, 4] - directions[:, 3])[:, None, None]
    ray_t = t_min + depth_u * t_span
    hit_y = float(metadata["topH"]) + directions[:, 1, None, None] * ray_t
    hit_y[~hit] = 0.0
    premul_height = hit_y.reshape(slices, samples)

    coverage_values, coverage_vectors, coverage_total = eigensystem(coverage)
    joint = np.concatenate((coverage, premul_height / float(metadata["topH"])), axis=1)
    joint_values, joint_vectors, joint_total = eigensystem(joint)

    reports: list[dict[str, object]] = []
    truth_hit = coverage > 0.5
    for requested_rank in RANKS:
        rank = min(requested_rank, slices)
        coverage_hat = reconstruction(joint_vectors, coverage, rank)
        height_hat = reconstruction(joint_vectors, premul_height, rank)
        predicted_hit = coverage_hat >= 0.5
        true_positive = predicted_hit & truth_hit
        false_positive = predicted_hit & ~truth_hit
        false_negative = ~predicted_hit & truth_hit
        agreement = int(np.count_nonzero(predicted_hit == truth_hit))

        denom = np.maximum(coverage_hat, 1e-6)
        decoded_y = height_hat / denom
        height_error = np.abs(decoded_y[true_positive] - premul_height[true_positive])
        quantiles = (
            np.quantile(height_error, (0.5, 0.95, 0.99)).tolist()
            if height_error.size
            else [0.0, 0.0, 0.0]
        )
        reports.append(
            {
                "rank": rank,
                "minimumSpatialTextureReadsRGBA": math.ceil(rank / 4),
                "jointEnergy": ratio(float(joint_values[:rank].sum()), joint_total),
                "coverageEnergy": ratio(float(coverage_values[:rank].sum()), coverage_total),
                "hitAgreement": ratio(agreement, truth_hit.size),
                "hitPrecision": ratio(int(np.count_nonzero(true_positive)), int(np.count_nonzero(true_positive | false_positive))),
                "hitRecall": ratio(int(np.count_nonzero(true_positive)), int(np.count_nonzero(true_positive | false_negative))),
                "falseHits": int(np.count_nonzero(false_positive)),
                "missedHits": int(np.count_nonzero(false_negative)),
                "hitHeightErrorMetresP50P95P99": quantiles,
                "spatialBasisBytesRGBA16F": math.ceil(rank / 4) * samples * 8,
            }
        )

    output = {
        **metadata,
        "phase": "top" if args.no_center else "integer-texel botanical-middle",
        "field": "joint coverage and premultiplied world hit height",
        "warning": "This is an optimistic exterior first-event lower bound; it excludes colour, normal, mark, continuous angular holdout, camera-inside origin phase, and deep events.",
        "ranks": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
