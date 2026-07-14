"""CLI for one explicitly selected Hovi candidate-evidence QA shard."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..candidate import HypothesisConfig
from .pipeline import DEFAULT_CANDIDATE_QA_ROOT, build_published_candidate_qa


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_CANDIDATE_QA_ROOT)
    parser.add_argument("--shard-x", type=int, required=True)
    parser.add_argument("--shard-y", type=int, required=True)
    parser.add_argument("--resolution-m", type=float, required=True)
    parser.add_argument("--vertical-bin-m", type=float, required=True)
    parser.add_argument("--smoothing-sigma-m", type=float, required=True)
    parser.add_argument("--mode-union-resolution-m", type=float, required=True)
    parser.add_argument("--max-vertical-bins", type=int, required=True)
    parser.add_argument("--max-hypotheses-per-cell", type=int, required=True)
    parser.add_argument("--reader-batch-records", type=int, default=1 << 18)
    parser.add_argument("--group-batch-records", type=int, default=1 << 15)
    parser.add_argument("--level-block-records", type=int, default=1 << 15)
    parser.add_argument("--memory-ceiling-bytes", type=int, default=1 << 30)
    parser.add_argument("--temp-ceiling-bytes", type=int, default=32 << 30)
    parser.add_argument("--peak-live-memory-ceiling-bytes", type=int, default=1 << 30)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = HypothesisConfig(
        vertical_bin_m=args.vertical_bin_m,
        smoothing_sigma_m=args.smoothing_sigma_m,
        mode_union_resolution_m=args.mode_union_resolution_m,
        max_vertical_bins=args.max_vertical_bins,
        max_hypotheses_per_cell=args.max_hypotheses_per_cell,
        group_batch_records=args.group_batch_records,
        level_block_records=args.level_block_records,
        peak_live_memory_ceiling_bytes=args.peak_live_memory_ceiling_bytes,
    )
    index_path = build_published_candidate_qa(
        args.manifest,
        shard_x=args.shard_x,
        shard_y=args.shard_y,
        resolution_m=args.resolution_m,
        hypothesis_config=config,
        workspace_root=args.workspace_root,
        output_root=args.output_root,
        reader_batch_records=args.reader_batch_records,
        memory_ceiling_bytes=args.memory_ceiling_bytes,
        temp_ceiling_bytes=args.temp_ceiling_bytes,
    )
    print(index_path)


if __name__ == "__main__":
    main()
