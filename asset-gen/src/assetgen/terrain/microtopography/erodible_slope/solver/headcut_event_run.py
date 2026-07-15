"""CLI for the single frozen Development-A headcut-event evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

from .headcut_event import load_headcut_event_config
from .headcut_event_artifact import materialize_headcut_event_evaluation
from .load import load_crop_evaluation_plan
from .model import load_process_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/microtopography/erodible-slope/r0-headcut-event-v1.json"),
    )
    parser.add_argument("--output-parent", type=Path)
    args = parser.parse_args()
    asset_gen_root = Path(__file__).resolve().parents[6]
    repo_root = asset_gen_root.parent
    config_path = args.config if args.config.is_absolute() else asset_gen_root / args.config
    event_config = load_headcut_event_config(config_path)
    bindings = event_config.values["bindings"]
    process_config_path = repo_root / bindings["process_config"]["path"]
    evaluation_plan_path = repo_root / bindings["evaluation_plan"]["path"]
    condition_bundle_path = repo_root / bindings["condition_bundle"]["path"]
    process_config = load_process_config(process_config_path)
    evaluations, _, _, domain, enlarged_domain = load_crop_evaluation_plan(
        evaluation_plan_path,
        condition_bundle_path=condition_bundle_path,
        config=process_config,
    )
    manifest = materialize_headcut_event_evaluation(
        domain=domain,
        enlarged_domain=enlarged_domain,
        evaluations=evaluations,
        process_config=process_config,
        event_config=event_config,
        event_config_path=config_path,
        repo_root=repo_root,
        output_parent=args.output_parent,
    )
    print(manifest)


if __name__ == "__main__":
    main()
