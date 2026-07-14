"""Published full-scan candidate evidence to content-addressed visual QA."""
from __future__ import annotations

from pathlib import Path

from .....config import DATA_WORK
from ..candidate import (
    CandidateSurfaceHypothesisGenerator,
    HypothesisConfig,
    MultiscaleShardAccumulator,
    SpatialObservationReader,
    load_verified_spatial_manifest,
)
from .grid import collect_candidate_qa_grid
from .render import publish_candidate_qa


DEFAULT_CANDIDATE_QA_ROOT = DATA_WORK / "hovi-full-scan-candidate-evidence"


def build_published_candidate_qa(
    manifest_path: Path,
    *,
    shard_x: int,
    shard_y: int,
    resolution_m: float,
    hypothesis_config: HypothesisConfig,
    workspace_root: Path,
    output_root: Path = DEFAULT_CANDIDATE_QA_ROOT,
    reader_batch_records: int = 1 << 18,
    memory_ceiling_bytes: int = 1 << 30,
    temp_ceiling_bytes: int = 32 << 30,
) -> Path:
    """Render one explicit shard from a verified published materialization."""
    manifest = load_verified_spatial_manifest(manifest_path)
    reader = SpatialObservationReader(manifest)
    accumulator = MultiscaleShardAccumulator(
        manifest,
        shard_x,
        shard_y,
        workspace_root=workspace_root,
        memory_ceiling_bytes=memory_ceiling_bytes,
        temp_ceiling_bytes=temp_ceiling_bytes,
    )
    try:
        for batch in reader.iter_spatial_shard(
            shard_x,
            shard_y,
            batch_records=reader_batch_records,
        ):
            accumulator.add(batch)
        evidence = accumulator.finalize()
    except BaseException:
        accumulator.abort()
        raise
    with evidence:
        generator = CandidateSurfaceHypothesisGenerator(hypothesis_config)
        grid = collect_candidate_qa_grid(
            generator.iter_resolution(evidence, resolution_m),
            shard_x=shard_x,
            shard_y=shard_y,
            resolution_m=resolution_m,
        )
    return publish_candidate_qa(
        grid,
        output_root=output_root,
        manifest=manifest,
        hypothesis_config=hypothesis_config,
    )
