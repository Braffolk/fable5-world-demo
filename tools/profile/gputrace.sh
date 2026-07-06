#!/usr/bin/env bash
#
# gputrace.sh — capture a GAME-ONLY Metal `.gputrace` (Xcode Metal-debugger replay
# bundle) using the ?profile=1 two-device split.
#
# HOW IT WORKS
#   `?profile=1` loads the whole world on a throwaway device labelled `laas-loading`,
#   then — after buildScene — swaps to a FRESH device labelled `laas-render` and runs
#   the game loop there (see src/core/ProfileBoot.ts). Dawn's `DAWN_TRACE_*` records a
#   device from creation to destruction and has NO start/stop hook, so filtering it to
#   `laas-render` yields a trace of ONLY the game: the ~11 GB of boot GPU compute
#   (heightfield/erosion/flow/bark bakes) ran on the discarded `laas-loading` device
#   and is excluded by construction.
#
# FLOW
#   launch Chrome (Dawn tracing armed) → wait for the swap (`[profile] swapped` on
#   stderr) → let the game run for CAPTURE seconds → quit Chrome. Destroying the
#   `laas-render` device on shutdown finalizes the `.gputrace`. Open it in Xcode.
#
# USAGE
#   tools/profile/gputrace.sh
#   CAPTURE=12 OUT=/tmp/laas_trace tools/profile/gputrace.sh
#   URL='http://localhost:5173/?scene=world&nanite=1&dpr=2&profile=1&grass=0' tools/profile/gputrace.sh
#
# Then analyze the result with the sibling tools (see tools/profile/README.md):
#   tools/profile/trace_static.py  <trace.gputrace>          # structure + VRAM (raw)
#   tools/profile/profile_report.py <exported.gputrace>      # counters/device (profiled export)
#
# REQUIRES the dev server on :5173 (npm run dev) and a Chrome whose Dawn backend
# honours DAWN_TRACE (Google Chrome stable does on macOS).
set -euo pipefail

URL="${URL:-http://localhost:5173/?scene=world&nanite=1&dpr=2&profile=1}"
OUT="${OUT:-/tmp/laas_trace}"                 # DAWN_TRACE_FILE_BASE (trace path prefix)
CAPTURE="${CAPTURE:-10}"                       # seconds of game to record after the swap
READY_TIMEOUT="${READY_TIMEOUT:-300}"          # cold boot can be ~65-100 s
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
PROFILE_DIR="${PROFILE_DIR:-/tmp/chrome-metal}"

# profile=1 is what makes the trace game-only — warn loudly if it's missing.
case "$URL" in
  *profile=1*) ;;
  *) echo "[gputrace] WARNING: URL has no profile=1 — the trace WILL include boot (11+ GB)." >&2 ;;
esac

LOG="$(mktemp -t gputrace-chrome-log)"
CHROME_PID=""
cleanup() {
  [ -n "$CHROME_PID" ] && kill "$CHROME_PID" 2>/dev/null || true
  pkill -f "user-data-dir=${PROFILE_DIR}" 2>/dev/null || true
  rm -f "$LOG" 2>/dev/null || true
}
trap cleanup EXIT

pkill -f "user-data-dir=${PROFILE_DIR}" 2>/dev/null || true   # clear a leftover instance
rm -rf "${OUT}"* 2>/dev/null || true                          # stale trace(s)

echo "[gputrace] launching Chrome — Dawn tracing armed, DEVICE_FILTER=laas-render"
DAWN_TRACE_FILE_BASE="${OUT}" \
DAWN_TRACE_DEVICE_FILTER=laas-render \
MTL_CAPTURE_ENABLED=1 \
  "${CHROME}" \
    --disable-gpu-sandbox --user-data-dir="${PROFILE_DIR}" \
    --disable-features=SkiaGraphite \
    --no-first-run --no-default-browser-check \
    --enable-logging=stderr --log-level=0 \
    --enable-dawn-features=use_user_defined_labels_in_backend,disable_symbol_renaming \
    "${URL}" >/dev/null 2>"$LOG" &
CHROME_PID=$!

echo "[gputrace] waiting for the render-device swap ('[profile] swapped…') — cold boot is slow…"
waited=0
until grep -qF '[profile] swapped' "$LOG" 2>/dev/null; do
  kill -0 "$CHROME_PID" 2>/dev/null || { echo "[gputrace] Chrome exited early:"; tail -n 12 "$LOG" 2>/dev/null || true; exit 1; }
  sleep 0.2; waited=$((waited + 1))
  (( waited > READY_TIMEOUT * 5 )) && { echo "[gputrace] TIMEOUT — no swap (dev server on :5173? URL has profile=1?):"; tail -n 12 "$LOG" 2>/dev/null || true; exit 1; }
done
grep -F '[profile] swapped' "$LOG" | tail -n 1 | sed 's/^/[gputrace]   /'

echo "[gputrace] swap done — recording ${CAPTURE}s of the game on laas-render…"
sleep "${CAPTURE}"

echo "[gputrace] quitting Chrome to finalize the trace (destroys laas-render)…"
kill "$CHROME_PID" 2>/dev/null || true         # SIGTERM → graceful GPU-device teardown
CHROME_PID=""
sleep 3                                         # let Dawn flush the trace to disk

echo "[gputrace] done. Trace written under: ${OUT}*"
ls -ld "${OUT}"* 2>/dev/null | sed 's/^/[gputrace]   /' || \
  echo "[gputrace]   (nothing at ${OUT}* — does this Chrome build honour DAWN_TRACE? see docs/METAL-PROFILING.md)"
echo "[gputrace] open it in Xcode ▸ File ▸ Open, or drag onto Xcode; the labelled passes come from --enable-dawn-features above."
