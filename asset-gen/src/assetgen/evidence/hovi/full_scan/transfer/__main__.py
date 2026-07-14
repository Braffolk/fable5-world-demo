from __future__ import annotations

import argparse
from pathlib import Path

from .compare import compare_hovi_full_vs_thinned


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full and thinned Hovi raw support")
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--thinned-laz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-points", type=int, default=262_144)
    parser.add_argument("--memory-ceiling-bytes", type=int, default=512 << 20)
    arguments = parser.parse_args()
    product = compare_hovi_full_vs_thinned(
        arguments.full_manifest,
        arguments.thinned_laz,
        batch_points=arguments.batch_points,
        memory_ceiling_bytes=arguments.memory_ceiling_bytes,
    )
    print(product.write(arguments.output_dir))


if __name__ == "__main__":
    main()
