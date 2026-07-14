from __future__ import annotations

import argparse
from pathlib import Path

from .audit import build_support_error_decision
from .artifacts import publish_support_error_decision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed B1/B2 support/error qualification of retained forest-floor evidence"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    decision = build_support_error_decision(arguments.repo_root)
    print(publish_support_error_decision(decision, arguments.output_root))


if __name__ == "__main__":
    main()
