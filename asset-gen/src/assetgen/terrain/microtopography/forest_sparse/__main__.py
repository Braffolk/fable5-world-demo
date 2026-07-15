"""CLI for static verification and the separated human-gated screen stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_EXPECTED_PACKAGE_ENTRIES = frozenset(
    {
        "__init__.py",
        "__main__.py",
        "contracts.py",
        "frozen_anchor.py",
        "operators.py",
        "qa.py",
        "screen.py",
    }
)


def _scan_package_before_sibling_import() -> Path:
    package_dir = Path(__file__).resolve().parent
    entries = {entry.name: entry for entry in package_dir.iterdir()}
    if set(entries) != _EXPECTED_PACKAGE_ENTRIES:
        raise RuntimeError(
            "forest_sparse import-shadow closure failed: "
            f"missing={sorted(_EXPECTED_PACKAGE_ENTRIES-set(entries))}, "
            f"unexpected={sorted(set(entries)-_EXPECTED_PACKAGE_ENTRIES)}"
        )
    for name, path in entries.items():
        if path.is_symlink() or not path.is_file() or path.suffix != ".py":
            raise RuntimeError(f"forest_sparse package entry is not an anchored source: {name}")
    return package_dir


_PACKAGE_DIR = _scan_package_before_sibling_import()

from .contracts import load_config, verify_static_closure
from .frozen_anchor import EXPECTED_EXTERNAL_MODULE_ORIGINS, EXPECTED_IMPLEMENTATION_SHA256
from .screen import finalize_evaluation, run_construction, run_evaluation


def _project_root() -> Path:
    for parent in _PACKAGE_DIR.parents:
        if (parent / "asset-gen" / "pyproject.toml").is_file() and (parent / "docs").is_dir():
            return parent
    raise RuntimeError("cannot locate project root for forest_sparse import closure")


def _verify_import_origins() -> None:
    root = _project_root()
    package = __package__
    if package is None:
        raise RuntimeError("forest_sparse must execute as a package module")
    anchored = dict(EXPECTED_IMPLEMENTATION_SHA256)
    anchored[
        "asset-gen/src/assetgen/terrain/microtopography/forest_sparse/frozen_anchor.py"
    ] = "self-sealed-by-config-and-commit"
    for relative in anchored:
        name = Path(relative).name
        logical_name = package if name == "__init__.py" else f"{package}.{Path(name).stem}"
        module = sys.modules.get(__name__ if name == "__main__.py" else logical_name)
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        expected = (root / relative).resolve()
        if origin is None or Path(origin).resolve() != expected:
            raise RuntimeError(
                f"forest_sparse module origin mismatch for {logical_name}: "
                f"expected {expected}, got {origin}"
            )
    for module_name, relative in EXPECTED_EXTERNAL_MODULE_ORIGINS.items():
        module = sys.modules.get(module_name)
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        expected = (root / relative).resolve()
        if origin is None or Path(origin).resolve() != expected:
            raise RuntimeError(
                f"bound external module origin mismatch for {module_name}: "
                f"expected {expected}, got {origin}"
            )


_verify_import_origins()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-static", action="store_true")
    mode.add_argument("--run-construction", action="store_true")
    mode.add_argument("--run-evaluation", action="store_true")
    mode.add_argument("--finalize-evaluation", action="store_true")
    parser.add_argument("--construction-manifest", type=Path)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--visual-approval", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.verify_static:
        print(
            json.dumps(
                {
                    "config_sha256": config.config_sha256,
                    "closure": verify_static_closure(config),
                },
                sort_keys=True,
            )
        )
        return
    if args.run_construction:
        print(run_construction(config))
        return
    if args.run_evaluation:
        if args.construction_manifest is None or args.visual_approval is None:
            parser.error("--run-evaluation requires --construction-manifest and --visual-approval")
        print(
            run_evaluation(
                config,
                construction_manifest=args.construction_manifest,
                visual_approval=args.visual_approval,
            )
        )
        return
    if args.evaluation_manifest is None or args.visual_approval is None:
        parser.error("--finalize-evaluation requires --evaluation-manifest and --visual-approval")
    print(
        finalize_evaluation(
            config,
            evaluation_manifest=args.evaluation_manifest,
            visual_approval=args.visual_approval,
        )
    )


if __name__ == "__main__":
    main()
