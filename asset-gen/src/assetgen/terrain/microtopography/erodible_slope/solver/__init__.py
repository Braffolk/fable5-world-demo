"""Whole-domain erodible-slope process solver."""

from .model import ProcessConfig, SlopeDomain
from .process import ProcessResult, solve_process

__all__ = ["ProcessConfig", "ProcessResult", "SlopeDomain", "solve_process"]
