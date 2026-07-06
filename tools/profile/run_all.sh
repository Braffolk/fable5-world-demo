#!/usr/bin/env bash
# run_all.sh — FULL profiling dump into one structured folder, built for one-shader-at-a-time
# optimization: summaries to pick a target, per-shader detail (nothing truncated), full .metal source.
# Verified 2026-07-06 (macOS 26.4 / Xcode 26.6, M-series).
#
# Usage:
#   tools/profile/run_all.sh <raw.gputrace> [exported.gputrace] [OUTDIR]
#     <raw.gputrace>       capture from gputrace.sh (has `capture` + MSL files)
#     [exported.gputrace]  Xcode "Profile GPU Trace → Replay → Export" bundle (has *.gpuprofiler_raw +
#                          store0). If omitted, timing/runtime steps are skipped.
#     [OUTDIR]             default: ./profile-results-<timestamp>
#
# Layout produced:
#   INDEX.md                    navigation + top rankings inline
#   summary/                    whole-frame overviews (single files, full detail)
#     structure.txt/.json         pass counts, VRAM map, compute-pipeline inventory
#     device.txt / counters.txt / report.txt   GPU config, 31-counter glossary, capture inventory
#     kernels_by_ms.txt           per-kernel GPU ms, FULL names, sorted (from timing.json)
#     timing.json / timing_table.txt   tmc/gputrace raw JSON + its table
#   runtime/                    per-shader RUNTIME time-per-line (ALL lines, full source) + _ranking.txt
#     msl/<NNN_name>.metal        full Metal source per shader (line N == breakdown line N)
#   static/                     per-shader STATIC per-line cost (spills/regs/ALU, ALL lines) + _ranking.txt
set -uo pipefail

RAW="${1:-}"; [[ -z "$RAW" || ! -e "$RAW" ]] && { echo "usage: $0 <raw.gputrace> [exported.gputrace] [OUTDIR]" >&2; exit 2; }
EXP="${2:-}"; [[ -n "${EXP:-}" && ! -e "$EXP" ]] && { echo "exported bundle not found: $EXP (skipping profiled steps)" >&2; EXP=""; }
TS="$(date +%Y%m%d-%H%M%S)"; OUT="${3:-$PWD/profile-results-$TS}"
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT/summary" "$OUT/runtime" "$OUT/static" || { echo "cannot create $OUT" >&2; exit 1; }
ERR="$OUT/summary/_errors.log"; : >"$ERR"
echo "profile run-all"; echo "  raw:      $RAW"; echo "  exported: ${EXP:-<none>}"; echo "  output →  $OUT"; echo

run() { local out="$1"; shift; printf '  … %-30s' "$out"; if "$@" >"$OUT/$out" 2>>"$ERR"; then echo ok; else echo "FAILED (summary/_errors.log)"; fi; }

# ---- summaries (whole-frame; full detail, no truncation) ----
run summary/structure.txt     "$DIR/trace_static.py" "$RAW" --top 100000
run summary/structure.json    "$DIR/trace_static.py" "$RAW" --json
if [[ -n "$EXP" ]]; then
  run summary/device.txt      "$DIR/profile_report.py" "$EXP" device
  run summary/report.txt      "$DIR/profile_report.py" "$EXP" summary
  run summary/counters.txt    "$DIR/profile_report.py" "$EXP" counters
  # per-kernel timing: full JSON (full names) + the tmc table + a full-name sorted table
  run summary/timing_table.txt "$DIR/gputrace_timing.sh" "$RAW" "$EXP" timing --json "$OUT/summary/timing.json"
  if [[ -s "$OUT/summary/timing.json" ]]; then
    printf '  … %-30s' "summary/kernels_by_ms.txt"
    python3 - "$OUT/summary/timing.json" >"$OUT/summary/kernels_by_ms.txt" 2>>"$ERR" <<'PY' && echo ok || echo "FAILED"
import json,sys
d=json.load(open(sys.argv[1]))
ks=sorted(d.get("kernel_timings",[]), key=lambda k:-k.get("total_duration",0))
tot=d.get("total_duration",1) or 1
print(f"# per-kernel GPU timing (tmc/gputrace), FULL names, sorted by total ms.  total_duration={tot/1e6:.2f} ms")
print(f"# NOTE: tmc/gputrace groups by ENCODER label; real shader kernels = computeGroup_* / ShaderModule_* (grep those).")
print(f"{'total_ms':>9} {'%tot':>6} {'invokes':>8} {'avg_us':>8} {'p95_us':>8}  name")
for k in ks:
    print(f"{k.get('total_duration',0)/1e6:>9.3f} {k.get('percent_of_total',0):>5.1f}% {k.get('invocation_count',0):>8} "
          f"{k.get('avg_duration',0)/1e3:>8.1f} {k.get('p95_duration',0)/1e3:>8.1f}  {k.get('name','')}")
PY
  fi
fi

# ---- per-shader STATIC (raw trace): one file per shader, all lines, full source ----
run static/_run.log           "$DIR/perline_remarks.py" "$RAW" --outdir "$OUT/static"

# ---- per-shader RUNTIME (exported): one file per shader, all lines, full source + msl/*.metal ----
if [[ -n "$EXP" ]]; then
  run runtime/_run.log        "$DIR/runtime_perline.py" "$EXP" --src --outdir "$OUT/runtime"
fi

# ---- INDEX ----
{
  echo "# Profile results — $TS"
  echo
  echo "raw: \`$RAW\`  ·  exported: \`${EXP:-<none>}\`"
  echo
  echo "## How to use (one shader at a time)"
  echo "1. Pick a target from a ranking: \`summary/kernels_by_ms.txt\` (GPU ms), \`runtime/_ranking.txt\` (%GPU per-line),"
  echo "   or \`static/_ranking.txt\` (occupancy cost = spills/temp-regs)."
  echo "2. Open that shader's files: \`runtime/<NNN>_<name>.txt\` (where time is spent, per line + source),"
  echo "   \`static/<NNN>_<name>.txt\` (why: spills/registers/ALU per line), \`runtime/msl/<NNN>_<name>.metal\` (full source)."
  echo "3. Everything is un-truncated; grep/scroll freely."
  echo
  echo "## Folders"
  echo '```'
  echo "summary/    whole-frame overviews (device, counters, VRAM/structure, per-kernel ms)"
  echo "runtime/    per-shader RUNTIME time-per-line (= Xcode Shaders tab) + _ranking.txt"
  echo "runtime/msl/  full Metal source per shader (file line N == breakdown line N)"
  echo "static/     per-shader STATIC per-line cost (spills / temp-registers / ALU) + _ranking.txt"
  echo '```'
  echo
  echo "## Top GPU-time shaders (runtime)"
  echo '```'
  [[ -f "$OUT/runtime/_ranking.txt" ]] && grep -v '^#' "$OUT/runtime/_ranking.txt" | sed '/^$/d' | head -16 || echo "(runtime step skipped)"
  echo '```'
  echo "## Top occupancy-cost shaders (static)"
  echo '```'
  [[ -f "$OUT/static/_ranking.txt" ]] && grep -v '^#' "$OUT/static/_ranking.txt" | sed '/^$/d' | head -16 || echo "(none)"
  echo '```'
} >"$OUT/INDEX.md" 2>>"$ERR"

echo; echo "done → $OUT"
echo "  summary/  $(ls "$OUT/summary" 2>/dev/null | wc -l | tr -d ' ') files"
echo "  runtime/  $(ls "$OUT/runtime"/*.txt 2>/dev/null | wc -l | tr -d ' ') shaders,  $(ls "$OUT/runtime/msl"/*.metal 2>/dev/null | wc -l | tr -d ' ') .metal"
echo "  static/   $(ls "$OUT/static"/*.txt 2>/dev/null | wc -l | tr -d ' ') shaders"
[[ -s "$ERR" ]] && echo "  (stderr/progress in summary/_errors.log)"
echo "  open $OUT/INDEX.md"
