"""Read the fixed OPARA sample without expanding its multi-gigabyte payload."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


_ROOT = "B_sample_plot"
GCP_MEMBER = (
    f"{_ROOT}/III_plot_0_raw/gcp_scale/2021-07-21/"
    "2021-07-21_GCP_Coordinates.txt"
)
PROTOCOL_README_MEMBER = (
    f"{_ROOT}/III_plot_0_raw/protocol_fieldwork/2021-07-21/read.me"
)
PARAMETER_MEMBER = (
    f"{_ROOT}/III_plot_0_raw/protocol_fieldwork/2021-07-21/parameter.csv"
)
PROCESSED_README_MEMBER = f"{_ROOT}/III_plot_1_processed/read.me"
TIMELAPSE_README_MEMBER = (
    f"{_ROOT}/III_plot_1_processed/sfm_timelapse/2021-07-21/"
    "timelapse/read.me"
)
LOG_SUMMARY_MEMBER = (
    f"{_ROOT}/III_plot_1_processed/sfm_timelapse/log_summary.txt"
)

QUALIFICATION_MEMBERS = (
    GCP_MEMBER,
    PROTOCOL_README_MEMBER,
    PARAMETER_MEMBER,
    PROCESSED_README_MEMBER,
    TIMELAPSE_README_MEMBER,
    LOG_SUMMARY_MEMBER,
)


def _bsdtar() -> str:
    executable = shutil.which("bsdtar")
    if executable is None:
        raise RuntimeError("OPARA probe requires the system bsdtar executable")
    return executable


def inventory(archive: Path) -> tuple[str, ...]:
    result = subprocess.run(
        [_bsdtar(), "-tf", archive],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    members = tuple(line for line in result.stdout.splitlines() if line)
    if any(
        Path(name).is_absolute() or ".." in Path(name).parts for name in members
    ):
        raise ValueError("OPARA archive contains an unsafe member path")
    if len(members) != len(set(members)):
        raise ValueError("OPARA archive contains duplicate member paths")
    return members


def extract_qualification_records(archive: Path) -> dict[str, bytes]:
    """Stream one solid archive pass and retain only the six small records."""
    with tempfile.TemporaryDirectory(prefix="opara-1038-probe-") as temporary:
        root = Path(temporary)
        subprocess.run(
            [_bsdtar(), "-xf", archive, "-C", root, *QUALIFICATION_MEMBERS],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        records = {}
        for member in QUALIFICATION_MEMBERS:
            path = root / member
            if not path.is_file():
                raise ValueError(f"OPARA qualification record was not extracted: {member}")
            records[member] = path.read_bytes()
        return records
