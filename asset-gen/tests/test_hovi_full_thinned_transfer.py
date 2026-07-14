from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from assetgen.evidence.hovi.full_scan.transfer.artifacts import write_machine_product
from assetgen.evidence.hovi.full_scan.transfer.binding import HeldFileBinding
from assetgen.evidence.hovi.full_scan.transfer.model import (
    CommonGridPlan,
    MultiscaleObservables,
    comparison_metrics,
)


def test_common_grid_computes_only_paired_raw_observables() -> None:
    plan = CommonGridPlan(0.0, 0.0, 2.0, 1.0, (0.5, 1.0))
    full = MultiscaleObservables(plan)
    full.update(
        np.asarray([0.1, 0.49, 0.51, 1.9]),
        np.asarray([0.1, 0.49, 0.1, 0.9]),
    )
    thinned = MultiscaleObservables(plan)
    thinned.update(np.asarray([0.1, 0.51]), np.asarray([0.1, 0.1]))

    metrics = comparison_metrics(full, thinned)
    fine = metrics["levels"][0]
    assert fine["full"]["point_count_sum"] == 4
    assert fine["thinned"]["point_count_sum"] == 2
    assert fine["paired_numeric_cells"]["both_occupied"] == 2
    assert fine["paired_numeric_cells"]["full_only_occupied"] == 1
    assert metrics["observable_boundary"]["frame_equivalence_claim"] is False

    arrays = full.arrays("full")
    assert arrays["full_point_count__0p5m"].tolist() == [[2, 1, 0, 0], [0, 0, 0, 1]]
    assert arrays["full_occupied__0p5m"].dtype == np.bool_
    assert np.isclose(
        arrays["full_nearest_in_cell_center_distance_m__0p5m"][0, 0],
        np.hypot(0.1 - 0.25, 0.1 - 0.25),
    )


def test_held_binding_rejects_in_place_mutation(tmp_path: Path) -> None:
    path = tmp_path / "source.bin"
    path.write_bytes(b"frozen-source")
    os.chmod(path, 0o644)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    binding = HeldFileBinding(
        path,
        expected_sha256=digest,
        expected_bytes=path.stat().st_size,
        expected_mode=0o644,
        label="test source",
    )
    try:
        path.write_bytes(b"mutated-data!")
        with pytest.raises(ValueError, match="changed during comparison"):
            binding.verify_unchanged()
    finally:
        binding.close(verify=False)


def test_machine_product_binds_metrics_and_observable_arrays(tmp_path: Path) -> None:
    output = tmp_path / "comparison"
    manifest_path = write_machine_product(
        output,
        metrics={"schema_version": "test", "levels": []},
        arrays={"full_point_count__1m": np.asarray([[3]], dtype=np.uint64)},
        sources={"frame_equivalence_established": False},
    )
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["status"] == "complete_unqualified_numeric_comparison"
    assert manifest["qa_pngs_generated"] is False
    with np.load(output / "observables.npz", allow_pickle=False) as arrays:
        assert arrays["full_point_count__1m"].tolist() == [[3]]
