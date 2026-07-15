"""Frozen Guérin-style forest-floor sparse-representation capacity screen."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ScreenConfig",
    "finalize_evaluation",
    "load_config",
    "run_construction",
    "run_evaluation",
    "verify_static_closure",
]

_EXPORT_MODULES = {
    "ScreenConfig": ".contracts",
    "load_config": ".contracts",
    "verify_static_closure": ".contracts",
    "finalize_evaluation": ".screen",
    "run_construction": ".screen",
    "run_evaluation": ".screen",
}


def __getattr__(name: str) -> Any:
    """Keep package import inert so the CLI can inspect siblings before import."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
