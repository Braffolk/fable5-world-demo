"""Authoritative geological conditioning domains."""

from .domains import EgtSurficialDomains, load_egt_surficial_domains
from .extract import extract_egt_surficial_chunk, extract_egt_surficial_window

__all__ = [
    "EgtSurficialDomains",
    "extract_egt_surficial_chunk",
    "extract_egt_surficial_window",
    "load_egt_surficial_domains",
]
