"""Bounded diagnostic grids derived from unqualified candidate hypotheses."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..candidate import CellHypothesisSet
from ..candidate.manifest import SHARD_M


UNOBSERVED = 0
SOLE_RAW_CANDIDATE = 1
AMBIGUOUS_CANDIDATES = 2
HYPOTHESIS_OVERFLOW = 3


@dataclass(frozen=True)
class CandidateQaGrid:
    shard_x: int
    shard_y: int
    resolution_m: float
    scan_balanced_height_m: np.ndarray
    hypothesis_count: np.ndarray
    evidence_state: np.ndarray
    unique_scan_count: np.ndarray
    maximum_pairwise_view_angle_deg: np.ndarray
    descriptive_inter_scan_mad_m: np.ndarray
    unknown_reason_counts: tuple[tuple[str, int], ...]

    @property
    def unambiguous_mask(self) -> np.ndarray:
        return self.evidence_state == SOLE_RAW_CANDIDATE


def collect_candidate_qa_grid(
    cells: Iterable[CellHypothesisSet],
    *,
    shard_x: int,
    shard_y: int,
    resolution_m: float,
) -> CandidateQaGrid:
    """Collect one complete 4 m shard without selecting or interpolating a surface."""
    cells_per_side = round(SHARD_M / resolution_m)
    if (
        cells_per_side <= 0
        or not np.isclose(
            cells_per_side * resolution_m,
            SHARD_M,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError("candidate QA resolution must exactly partition one 4 m shard")
    shape = (cells_per_side, cells_per_side)
    seen = np.zeros(shape, dtype=np.bool_)
    height = np.full(shape, np.nan, dtype=np.float64)
    hypothesis_count = np.zeros(shape, dtype=np.uint16)
    state = np.full(shape, 255, dtype=np.uint8)
    unique_scans = np.zeros(shape, dtype=np.uint8)
    view_angle = np.full(shape, np.nan, dtype=np.float64)
    disagreement = np.full(shape, np.nan, dtype=np.float64)
    unknown_reasons: Counter[str] = Counter()

    for cell in cells:
        if (
            cell.shard_x != shard_x
            or cell.shard_y != shard_y
            or cell.resolution_m != resolution_m
            or not 0 <= cell.cell_x < cells_per_side
            or not 0 <= cell.cell_y < cells_per_side
        ):
            raise ValueError("candidate QA cell lies outside its selected shard/resolution")
        if (
            cell.qualification_status != "raw_candidate_unqualified"
            or cell.surface_claim
            or cell.target_truth
            or cell.synthesis_authorized
            or cell.interpolation_state != "not_performed"
        ):
            raise PermissionError("candidate QA input crossed its unqualified evidence boundary")
        key = (cell.cell_y, cell.cell_x)
        if seen[key]:
            raise ValueError("candidate QA input repeats a cell")
        seen[key] = True
        count = len(cell.hypotheses)
        if count > np.iinfo(np.uint16).max:
            raise ValueError("candidate QA hypothesis count exceeds its diagnostic ABI")
        hypothesis_count[key] = count
        unique_scans[key] = cell.unique_scan_count
        unknown_reasons.update(cell.unknown_reasons)
        if cell.observation_state == "unobserved":
            if count != 0:
                raise ValueError("unobserved candidate QA cell contains hypotheses")
            state[key] = UNOBSERVED
        elif cell.observation_state == "hypothesis_overflow":
            if count != 0:
                raise ValueError("overflow candidate QA cell contains retained hypotheses")
            state[key] = HYPOTHESIS_OVERFLOW
        elif cell.observation_state == "observed" and count == 1:
            hypothesis = cell.hypotheses[0]
            if hypothesis.hypothesis_index != 0:
                raise ValueError("sole candidate hypothesis has a nonzero index")
            state[key] = SOLE_RAW_CANDIDATE
            height[key] = hypothesis.scan_balanced_z_m
            disagreement[key] = hypothesis.descriptive_inter_scan_mad_m
            if hypothesis.maximum_pairwise_view_angle_deg is not None:
                view_angle[key] = hypothesis.maximum_pairwise_view_angle_deg
        elif cell.observation_state == "observed" and count > 1:
            state[key] = AMBIGUOUS_CANDIDATES
        else:
            raise ValueError("candidate QA observed cell has no retained hypothesis")

    if not np.all(seen) or np.any(state == 255):
        raise ValueError("candidate QA input does not cover the complete selected shard")
    return CandidateQaGrid(
        shard_x=shard_x,
        shard_y=shard_y,
        resolution_m=resolution_m,
        scan_balanced_height_m=height,
        hypothesis_count=hypothesis_count,
        evidence_state=state,
        unique_scan_count=unique_scans,
        maximum_pairwise_view_angle_deg=view_angle,
        descriptive_inter_scan_mad_m=disagreement,
        unknown_reason_counts=tuple(sorted(unknown_reasons.items())),
    )
