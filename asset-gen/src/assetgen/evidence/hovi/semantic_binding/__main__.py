"""CLI for the bounded Hovi semantic-scope binding."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import build_hovi_semantic_binding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--photo-index", type=Path, required=True)
    parser.add_argument("--materialization-manifest", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=None)
    args = parser.parse_args()
    kwargs = {
        "retained_path": args.retained,
        "photo_index_path": args.photo_index,
        "materialization_manifest_path": args.materialization_manifest,
    }
    if args.work_root is not None:
        kwargs["work_root"] = args.work_root
    print(build_hovi_semantic_binding(**kwargs))


if __name__ == "__main__":
    main()
