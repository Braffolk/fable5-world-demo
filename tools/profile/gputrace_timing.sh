#!/usr/bin/env bash
#
# gputrace_timing.sh — HEADLESS per-kernel GPU timing (the frame timeline) from a
# captured raw trace + its Xcode-exported profile, via tmc/gputrace.
#
# WHY the two inputs: a raw `.gputrace` (our gputrace.sh) has the command stream but
# no timing; Xcode's "Profile GPU Trace → Export" produces a *profile-only* bundle
# whose `*.gpuprofiler_raw` FILE is an NSKeyedArchiver plist of the counters/timing.
# tmc/gputrace wants that plist as `<trace>/<x>.gpuprofiler_raw/streamData` (a
# DIRECTORY named …gpuprofiler_raw containing a file `streamData`). This script
# grafts the exported plist into the raw trace in that layout (symlink, no copy),
# then runs `gputrace timing` → real per-kernel ms / avg / p50 / p95 / %total.
#
# INSTALL the analyzer once:  go install github.com/tmc/gputrace/cmd/gputrace@latest
#
# USAGE:
#   tools/profile/gputrace_timing.sh <raw.gputrace> <exported.gputrace | streamData.plist> [CMD]
#     CMD defaults to `timing`; can be any profiler-data command (stats, kernels,
#     timeline, pprof …). Extra gputrace flags: put them after CMD.
#
# EXAMPLE:
#   tools/profile/gputrace_timing.sh /tmp/laas_trace-*.gputrace /tmp/exported.gputrace
#   tools/profile/gputrace_timing.sh /tmp/raw.gputrace /tmp/exported.gputrace timeline -o /tmp/t.json
#
# NOTE: per-LINE shader cost is NOT reliable here — our shaders are Dawn-transpiled
# MSL (all named `main0`, no .metal source files), so gputrace's shader-source /
# pprof -list can't map lines. Read per-line in Xcode's Shaders tab. See README.md §5.
set -euo pipefail

RAW="${1:?usage: gputrace_timing.sh <raw.gputrace> <exported.gputrace|plist> [cmd ...]}"
EXP="${2:?need the exported .gputrace bundle or its *.gpuprofiler_raw plist}"
shift 2 || true
CMD="${1:-timing}"; [ $# -gt 0 ] && shift || true

GPUTRACE="$(command -v gputrace || echo "$HOME/go/bin/gputrace")"
[ -x "$GPUTRACE" ] || { echo "[gputrace_timing] gputrace not found — run: go install github.com/tmc/gputrace/cmd/gputrace@latest" >&2; exit 1; }
[ -d "$RAW" ] && [ -e "$RAW/capture" -o -e "$RAW/unsorted-capture" ] || { echo "[gputrace_timing] $RAW is not a raw .gputrace (no capture stream)" >&2; exit 1; }

# locate the exported plist (Xcode export puts it as a FILE ending in .gpuprofiler_raw).
# The perf stream is ALWAYS a standalone *.gpuprofiler_raw FILE — never inside store0.
if [ -d "$EXP" ]; then
  BPLIST="$(find "$EXP" -maxdepth 1 -name '*.gpuprofiler_raw' -type f | head -1)"
else
  BPLIST="$EXP"
fi
if [ -z "${BPLIST:-}" ] || [ ! -e "$BPLIST" ]; then
  if [ -d "$EXP" ] && [ -f "$EXP/store0" ]; then
    cat >&2 <<EOF
[gputrace_timing] no *.gpuprofiler_raw in $EXP.
  This is a CAPTURE-ONLY export (store0 + index only) — it carries NO performance data.
  The perf stream is a standalone *.gpuprofiler_raw FILE and is NEVER inside store0.
  Re-export from Xcode: open the RAW .gputrace, run the GPU profiler (Replay /
  Debug ▸ 'Profile GPU Trace' so counters populate), then File ▸ Export… with
  "Embed performance data" ENABLED. A correct profiled export contains a
  *.gpuprofiler_raw file (GBs) AND a thumbnails_encoder/ folder. See docs/METAL-PROFILING.md.
EOF
  else
    echo "[gputrace_timing] no *.gpuprofiler_raw plist found in $EXP (pass an EXPORTED profiled .gputrace)" >&2
  fi
  exit 1
fi

# graft into the tmc/gputrace layout: <raw>/profile.gpuprofiler_raw/streamData
PDIR="$RAW/profile.gpuprofiler_raw"
mkdir -p "$PDIR"
ln -sf "$BPLIST" "$PDIR/streamData"
echo "[gputrace_timing] grafted $(basename "$BPLIST") → $PDIR/streamData" >&2

exec "$GPUTRACE" "$CMD" "$RAW" "$@"
