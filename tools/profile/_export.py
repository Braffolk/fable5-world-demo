#!/usr/bin/env python3
"""_export.py — locate the `*.gpuprofiler_raw` perf stream in an Xcode-exported `.gputrace`,
and give ONE precise, actionable error when it's missing.

WHY this exists (verified 2026-07-09, Xcode 26.6 / macOS 26.4):
An Xcode "Export" of a .gputrace can produce TWO different bundle shapes:

  • PROFILED export  = store0 + index + metadata + (gpu.quicklook-preview)
                       + **NAME.gputrace.gpuprofiler_raw**  (the perf stream, GBs)
                       + **thumbnails_encoder/**             (per-encoder thumbs)
    → has timings / counters / per-line PC samples.  runtime/timing/counters WORK.

  • CAPTURE-ONLY export = store0 + index + metadata + (gpu.quicklook-preview)   ONLY
    → NO perf data at all.  It is produced when you Export WITHOUT "Embed performance
      data" (or before/without running the GPU profiler on Replay).

The perf stream is ALWAYS a standalone `*.gpuprofiler_raw` FILE — it is NEVER packed
inside `store0`. `store0` is only the capture (shader `program_source` + resource dumps),
which is why a capture-only `store0` can still be 1–2 GB and even grow across captures.
Verified: a profiled export's `store0` contains ZERO profiler markers (GPRWCNTR /
`Havested Binaries` / `Program Address Mappings` / `ShaderProfilerData`); every one of
those lives in the sibling `*.gpuprofiler_raw`. So there is nothing to "carve" out of a
capture-only store0 — the data was simply not exported.
"""
import glob
import os

REEXPORT_HINT = (
    "This is a CAPTURE-ONLY export (store0 + index + metadata only) — it carries NO\n"
    "  performance data (no timings, counters, or per-line samples). The perf stream is a\n"
    "  standalone *.gpuprofiler_raw FILE and is NEVER inside store0, so nothing can be\n"
    "  recovered from this bundle.\n"
    "  To get perf data, in Xcode:\n"
    "    1. Open the RAW .gputrace.\n"
    "    2. Run the GPU profiler — press Replay / Debug ▸ 'Profile GPU Trace' so the\n"
    "       counters + Shaders tab actually populate (this is the step that generates perf).\n"
    "    3. File ▸ Export…  with  \"Embed performance data\"  ENABLED.\n"
    "  A correct profiled export contains a  *.gpuprofiler_raw  file (GBs) AND a\n"
    "  thumbnails_encoder/ folder next to store0.  See docs/METAL-PROFILING.md."
)


def find_gpuprofiler_raw(path):
    """Return path to the *.gpuprofiler_raw FILE, or None if this bundle has none.
    Accepts an exported .gputrace DIRECTORY or the .gpuprofiler_raw file itself."""
    if not os.path.isdir(path):
        return path if os.path.exists(path) else None
    hits = [x for x in glob.glob(os.path.join(path, "*.gpuprofiler_raw")) if os.path.isfile(x)]
    return hits[0] if hits else None


def is_capture_only_export(path):
    """True when `path` is an exported .gputrace DIRECTORY that has store0 but no perf stream."""
    return (os.path.isdir(path)
            and os.path.isfile(os.path.join(path, "store0"))
            and find_gpuprofiler_raw(path) is None)


def require_gpuprofiler_raw(path, tool="profile"):
    """Return the *.gpuprofiler_raw path, or raise SystemExit with a precise, actionable
    message. Distinguishes the capture-only-export case from a plain missing bundle."""
    raw = find_gpuprofiler_raw(path)
    if raw:
        return raw
    if is_capture_only_export(path):
        raise SystemExit(f"[{tool}] no *.gpuprofiler_raw in {path}.\n  {REEXPORT_HINT}")
    raise SystemExit(
        f"[{tool}] no *.gpuprofiler_raw FILE found in {path} — pass an EXPORTED profiled\n"
        f"  .gputrace bundle (see docs/METAL-PROFILING.md), not a raw capture.")
