"""Command-line entry point for the non-authorizing A/C development candidate."""
from __future__ import annotations

import argparse
from pathlib import Path

from .artifact import materialize_candidate
from .load import load_condition_bundle
from .model import load_process_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/microtopography/erodible-slope/solver-c1-v1.json"),
    )
    parser.add_argument("--output-parent", type=Path)
    args = parser.parse_args()
    config = load_process_config(args.config)
    domains, enlarged, crops, collars, condition_sha256 = load_condition_bundle(
        args.conditions, config
    )
    manifest = materialize_candidate(
        domains=domains,
        enlarged_domains=enlarged,
        crops_en=crops,
        collars_m=collars,
        config=config,
        config_path=args.config,
        condition_bundle_path=args.conditions,
        condition_bundle_sha256=condition_sha256,
        output_parent=args.output_parent,
    )
    print(manifest)


if __name__ == "__main__":
    main()
