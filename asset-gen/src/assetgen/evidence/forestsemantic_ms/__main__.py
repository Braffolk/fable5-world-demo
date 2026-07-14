from __future__ import annotations

import argparse
from pathlib import Path

from ...config import DATA_IN
from ...fetch.forestsemantic_ms import build_forestsemantic_plan
from .candidate import build_forestsemantic_evidence


def main() -> None:
    _, retention_id = build_forestsemantic_plan()
    parser = argparse.ArgumentParser(description="Build ForestSemantic-MS semantic evidence.")
    parser.add_argument(
        "--retained",
        type=Path,
        default=DATA_IN / "evidence" / "forestsemantic_ms" / retention_id / "retained.json",
    )
    args = parser.parse_args()
    print(build_forestsemantic_evidence(args.retained))


if __name__ == "__main__":
    main()
