"""Absolute-coordinate deterministic synthesis from direct measured patches."""

from __future__ import annotations

import hashlib

import numpy as np

from .model import PatchBank


def _hash64(*values: int, seed: int) -> int:
    h = hashlib.blake2b(digest_size=8, person=b"laas-micro-v1")
    h.update((int(seed) & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little"))
    for value in values:
        h.update((int(value) & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little"))
    return int.from_bytes(h.digest(), "little")


def _stable_pool(bank: PatchBank) -> np.ndarray:
    fingerprints = np.asarray([
        _hash64(i, int(bank.source_index[i]), seed=0) for i in range(len(bank.patches_m))
    ], dtype=np.uint64)
    # Evidence quality fixes the candidate order only; every accepted patch remains eligible.
    return np.lexsort((fingerprints, -np.asarray(bank.quality)))


def _base_patch_position(pool: np.ndarray, tx: int, ty: int, seed: int) -> int:
    return int(_hash64(tx, ty, seed=seed) % len(pool))


def _pairwise_normalized_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Symmetric edge-shape error without preferring low-amplitude patches.

    Raw MSE makes smooth patches globally cheaper even when a rough pair has the
    same relative mismatch. Normalizing by the measured energy of both strips
    keeps the cost dimensionless and bounded while retaining amplitude mismatch
    as evidence.
    """
    a = a.reshape(a.shape[0], -1).astype(np.float64, copy=False)
    b = b.reshape(b.shape[0], -1).astype(np.float64, copy=False)
    sum_a2 = np.sum(a * a, axis=1)[:, None]
    sum_b2 = np.sum(b * b, axis=1)[None, :]
    distance = np.maximum(
        sum_a2 + sum_b2 - 2.0 * (a @ b.T),
        0.0,
    )
    energy = sum_a2 + sum_b2
    return np.divide(
        distance,
        energy,
        out=np.zeros_like(distance),
        where=energy > np.finfo(np.float64).eps,
    )


def _edge_cost_tables(bank: PatchBank, pool: np.ndarray) -> np.ndarray:
    patches = bank.patches_m[pool]
    overlap = bank.overlap_cells
    return np.stack([
        _pairwise_normalized_error(patches[:, :, :overlap], patches[:, :, -overlap:]),
        _pairwise_normalized_error(patches[:, :, -overlap:], patches[:, :, :overlap]),
        _pairwise_normalized_error(patches[:, :overlap, :], patches[:, -overlap:, :]),
        _pairwise_normalized_error(patches[:, -overlap:, :], patches[:, :overlap, :]),
    ])


def _select_patch(
    pool: np.ndarray,
    source_by_position: np.ndarray,
    edge_cost: np.ndarray,
    tx: int,
    ty: int,
    seed: int,
    top_k: int,
) -> int:
    """Choose against coordinate-stable neighbor proposals, independent of request extent."""
    neighbor_coords = ((tx - 1, ty), (tx + 1, ty), (tx, ty - 1), (tx, ty + 1))
    neighbor_positions = [
        _base_patch_position(pool, x, y, seed) for x, y in neighbor_coords
    ]
    costs = sum(edge_cost[side, :, pos] for side, pos in enumerate(neighbor_positions))
    tie = np.asarray([_hash64(tx, ty, int(pid), seed=seed) for pid in pool], dtype=np.uint64)
    ranked = np.lexsort((tie, costs))
    count = min(max(1, top_k), len(ranked))
    # Round-robin the best edge-compatible patch from each calibration source. This
    # prevents a particularly smooth plot from erasing the measured between-plot variance.
    by_source = {
        int(source): ranked[source_by_position[ranked] == source]
        for source in np.unique(source_by_position)
    }
    balanced: list[int] = []
    depth = 0
    while len(balanced) < count:
        available = [values[depth] for values in by_source.values() if depth < len(values)]
        if not available:
            break
        available.sort(key=lambda pos: (costs[pos], tie[pos]))
        balanced.extend(available[: count - len(balanced)])
        depth += 1
    chosen_rank = _hash64(tx, ty, 0x544F504B, seed=seed) % len(balanced)
    return int(pool[balanced[chosen_rank]])


def _blend_window(patch_cells: int, overlap: int) -> np.ndarray:
    axis = np.ones(patch_cells, dtype=np.float64)
    phase = (np.arange(overlap, dtype=np.float64) + 0.5) / overlap
    # Complementary sine/cosine ramps are an equal-power crossfade: squared
    # contributor weights sum to one through every overlap. This preserves the
    # measured variance instead of printing a low-energy quilt at the stride.
    ramp = np.sin(0.5 * np.pi * phase)
    axis[:overlap] = ramp
    axis[-overlap:] = ramp[::-1]
    return axis[:, None] * axis[None, :]


def synthesize_measured(
    bank: PatchBank,
    *,
    origin_e_m: float,
    origin_n_m: float,
    shape: tuple[int, int],
    seed: int = 0,
    top_k: int = 12,
) -> np.ndarray:
    """Synthesize an arbitrary aligned grid by overlap-blending unchanged TLS patches.

    The same absolute sample receives bit-identical output regardless of requested crop.
    Origins are sample-center coordinates on the bank's global texel lattice; no
    resampling or amplitude normalization occurs here.
    """
    bank.validate()
    rows, cols = map(int, shape)
    if rows <= 0 or cols <= 0:
        raise ValueError("shape must be positive")
    e_cell = int(round(origin_e_m / bank.texel_m - 0.5))
    n_cell = int(round(origin_n_m / bank.texel_m - 0.5))
    if not np.isclose((e_cell + 0.5) * bank.texel_m, origin_e_m, atol=1e-8):
        raise ValueError("origin_e_m is not aligned to the exemplar sample-center lattice")
    if not np.isclose((n_cell + 0.5) * bank.texel_m, origin_n_m, atol=1e-8):
        raise ValueError("origin_n_m is not aligned to the exemplar sample-center lattice")

    p = bank.patch_cells
    step = p - bank.overlap_cells
    tx0 = (e_cell - p + step) // step
    tx1 = (e_cell + cols - 1) // step
    ty0 = (n_cell - p + step) // step
    ty1 = (n_cell + rows - 1) // step
    pool = _stable_pool(bank)
    source_by_position = np.asarray(bank.source_index)[pool]
    edge_cost = _edge_cost_tables(bank, pool)
    window = _blend_window(p, bank.overlap_cells)
    accum = np.zeros((rows, cols), dtype=np.float64)
    weight_energy = np.zeros((rows, cols), dtype=np.float64)

    for ty in range(ty0, ty1 + 1):
        patch_n = ty * step
        out_y0 = max(0, patch_n - n_cell)
        out_y1 = min(rows, patch_n + p - n_cell)
        py0 = out_y0 + n_cell - patch_n
        py1 = py0 + out_y1 - out_y0
        for tx in range(tx0, tx1 + 1):
            patch_e = tx * step
            out_x0 = max(0, patch_e - e_cell)
            out_x1 = min(cols, patch_e + p - e_cell)
            px0 = out_x0 + e_cell - patch_e
            px1 = px0 + out_x1 - out_x0
            patch_id = _select_patch(
                pool, source_by_position, edge_cost, tx, ty, seed, top_k
            )
            w = window[py0:py1, px0:px1]
            accum[out_y0:out_y1, out_x0:out_x1] += bank.patches_m[patch_id, py0:py1, px0:px1] * w
            weight_energy[out_y0:out_y1, out_x0:out_x1] += w * w
    if np.any(weight_energy <= 0):
        raise RuntimeError("internal patch coverage failure")
    return accum / np.sqrt(weight_energy)
