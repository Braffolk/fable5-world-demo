# `tools/profile/` — Metal GPU trace tooling (LLM guide)

Reusable scripts + reference for getting perf signal out of this game's GPU work
on Apple Silicon, from the terminal. Written for a future coding agent: run a
script, read the LLM-friendly output, know where to look next. Verified
2026-07-06 on macOS 26.4 / Xcode 26.6, M-series (32-core) Max.

```
tools/profile/
  run_all.sh           run EVERY tool below on a trace pair → all outputs in one folder  ◀ start here
  gputrace.sh          capture a GAME-ONLY .gputrace (via ?profile=1 two-device split)
  trace_static.py      raw .gputrace → pipelines, pass counts, VRAM map      (fast, no deps)
  profile_report.py    EXPORTED profiled .gputrace → device, inventory, counter glossary
  gputrace_timing.sh   raw + exported → HEADLESS per-kernel GPU ms (via tmc/gputrace)  ★
  perline_remarks.py   raw .gputrace → HEADLESS per-SHADER + per-SOURCE-LINE STATIC cost  ★★
                       (spilled bytes, temp registers, ALU, per-line instr — the M1 levers)
  runtime_perline.py   EXPORTED profiled .gputrace → HEADLESS per-SOURCE-LINE RUNTIME TIME  ★★★
                       (PC samples × PAB load-map × Mach-O DWARF; = Xcode Shaders tab, per shader)
                       --src adds shader NAME/TYPE + the MSL source text per line (via _srcmap.py)
  _srcmap.py           helper: store0 (zlib) → MSL source + Dawn names; bridges binaryUniqueId→name
  perinstr_hotness.py  exported → per-instruction PC histogram (coarse; prefer perline_remarks)
  gpuprofiler-raw-format.md   reverse-engineering notes for the binary formats (READ §8, §8c)
  README.md            this file
```

---

## 0. What each artifact holds (read this first)

| Artifact | How | Holds | Timing/counters? |
|---|---|---|---|
| **raw `.gputrace`** | `gputrace.sh` (Dawn) or Xcode capture | command structure + resource dumps | **no** (computed on replay) |
| **exported `.gputrace`** | ⚠️ MANUAL: open raw in Xcode → replay/profile → **File ▸ Export with "Embed performance data" ON** | above **+ `*.gpuprofiler_raw`** (counters/timings/per-line samples) **+ `store0`** (shader source) | **yes**, but in proprietary binary streams |
| Instruments `.trace` | `xctrace record --template 'Metal System Trace'` | system GPU timeline | yes → `xctrace export` XML/CSV (headless) |

A `.gputrace` is a **directory bundle**. The numbers you see in Xcode come from
*replaying* it on the GPU; they are not stored in a raw capture.

## 1. The workflow — capture → MANUAL Xcode export → analyze

```bash
# 1. CAPTURE a game-only raw trace (headless; excludes the ~11 GB of boot bakes — see §7)
tools/profile/gputrace.sh                     # → /tmp/laas_trace-*.gputrace  (structure only, NO timing)

# 2. ⚠️ MANUAL, in Xcode — the tools CANNOT do this (a raw trace has no timing):
#    open /tmp/laas_trace-*.gputrace in Xcode  →  let it replay/profile the frame  →
#    File ▸ Export…  with  "Embed performance data"  ENABLED   →  e.g. /tmp/exported.gputrace
#    (that adds the *.gpuprofiler_raw counters/timings + store0 shader source the tools read)
#    Exact clicks + why: docs/METAL-PROFILING.md.

# 3. ANALYZE everything into one folder (headless):
tools/profile/run_all.sh /tmp/laas_trace-*.gputrace /tmp/exported.gputrace
#    → ./profile-results-<timestamp>/  — open its INDEX.md
```

### What `run_all.sh` produces (built for one-shader-at-a-time optimization)
```
INDEX.md                    navigation + top rankings inline
summary/                    whole-frame overviews — read these to PICK a target
  kernels_by_ms.txt           per-kernel GPU ms, FULL names, sorted
  structure.txt/.json         pass counts, VRAM map, compute-pipeline inventory
  device.txt counters.txt report.txt   GPU config, 31-counter glossary, capture inventory
runtime/                    ONE file per shader: RUNTIME time-per-line (= Xcode Shaders tab), ALL lines + source
  001_<name>.txt … _ranking.txt        ranked by %GPU; the `bin` column is the unique id (name~binid on fuzzy dups)
  msl/001_<name>.metal …               full Metal source per shader; file line N == breakdown line N
static/                     ONE file per shader: STATIC per-line cost (spills / temp-regs / ALU) + _ranking.txt
```
Nothing is truncated — all lines, full source, full kernel names. Pick your granularity: `summary/`
ranking → a shader's `runtime/`+`static/` file → its full `msl/*.metal`.

Individual tools (if you don't want the whole dump) are in §3–§5c. `trace_static.py` works on the raw
trace alone (step 1); everything else needs the **exported** bundle from step 2.

## 2. `gputrace.sh` — capture game-only

`?profile=1` loads the world on a throwaway `laas-loading` device, then swaps to a
fresh `laas-render` device for the game loop; `DAWN_TRACE_DEVICE_FILTER=laas-render`
therefore records only game frames (mechanism: `src/core/ProfileBoot.ts`, deep
dive in `docs/METAL-PROFILING.md`). Needs the dev server on :5173 + Google Chrome.

```bash
tools/profile/gputrace.sh
CAPTURE=12 OUT=/tmp/laas_trace tools/profile/gputrace.sh
URL='http://localhost:5173/?scene=world&nanite=1&dpr=2&profile=1&grass=0' tools/profile/gputrace.sh
```

## 3. `trace_static.py` — structure + memory (raw trace, fast)

No timing (a raw trace has none). Emits: render/compute/blit pass counts, embedded
MSL library count, GPU **memory footprint + biggest resources** (great for the MEM
arc), and the labelled **compute-pipeline inventory**. `--json`, `--top N`.
```bash
tools/profile/trace_static.py trace.gputrace --top 20
tools/profile/trace_static.py trace.gputrace --json
```

## 4. `profile_report.py` — the profiled export

Reads the `*.gpuprofiler_raw` inside an exported bundle (a 2–3 GB `NSKeyedArchiver`
plist). First run parses it (~30–90 s, RAM ≈ 3× size) and writes a small
`*.summary.json` sidecar; later runs are instant.
```bash
tools/profile/profile_report.py exported.gputrace              # summary (device, inventory, #counters)
tools/profile/profile_report.py exported.gputrace counters     # 31-counter glossary (name/type/desc/sampled)
tools/profile/profile_report.py exported.gputrace device       # GPU: gen/cores/frags/GPs
tools/profile/profile_report.py exported.gputrace json         # everything, as JSON
tools/profile/profile_report.py exported.gputrace counters --json
```
The **counter glossary** is the payoff for interpreting the GUI: it prints every
GPU counter Xcode sampled (ALU/Buffer/Texture/MMU/LLC **Limiters** and
**Utilizations**, Compute/Fragment/Vertex **Occupancy**, F16/F32 util, GPU
read/write **Bandwidth**, Partial Renders) with Apple's own descriptions — so you
can read "ALU Limiter 100%" in the GUI and know it means *ALU-throughput bound*.

## 5. Exact per-kernel ms — HEADLESS via `tmc/gputrace` (verified)

`github.com/tmc/gputrace` (MIT Go CLI) decodes the profiler streams. Install once:
```bash
go install github.com/tmc/gputrace/cmd/gputrace@latest      # → ~/go/bin/gputrace
```
Its profiler commands want the exported plist as `<trace>/<x>.gpuprofiler_raw/streamData`
(a DIRECTORY), whereas Xcode's *Export* writes a `.gpuprofiler_raw` FILE. `gputrace_timing.sh`
grafts one into the other (symlink) and runs the timing:
```bash
tools/profile/gputrace_timing.sh <raw.gputrace> <exported.gputrace>     # → per-kernel ms table
# → Total Duration, then per kernel: Invokes | Total ms | Avg/Min/Max/P50/P95 µs | %Total
```
**This is the headless frame timeline** — real per-kernel GPU milliseconds, exactly what the
perf arc needed. Structure commands run on the raw trace alone (no graft):
```bash
gputrace stats <raw.gputrace>      # 1007 encoders, 1702 dispatches, 753 kernels, 3.84 GB
gputrace kernels <raw.gputrace>    # kernel ↔ pipeline ↔ dispatch counts (our Dawn labels)
gputrace timeline <raw.gputrace> -o t.json   # Perfetto/Chrome timeline
```
⚠️ `timing` attributes time to encoder labels, so `Dawn_TextureView` / `Dawn_Buffer_*` show up
as "kernels" (they're pass/resource labels); our real kernels are the `computeGroup_nan*` /
`ShaderModule_*` rows — grep for those.

### 5a. Per-SHADER + per-LINE cost — HEADLESS via `perline_remarks.py`  ★★
The right per-line signal is **structural, from the compiler** — not runtime PC sampling
(too coarse to resolve lines; sample "hotness" was a dead end). When the Apple-GPU compiler
(`agc`) builds each shader it emits a **telemetry block + LLVM optimization remarks** keyed
to `program_source` lines; those land in hex-named bplists in the raw `.gputrace`, and the
MSL source lands in library-hash files. `perline_remarks.py` joins them:
```bash
tools/profile/perline_remarks.py <raw.gputrace>          # ranked per shader
tools/profile/perline_remarks.py <raw.gputrace> --src    # + the MSL source of each hot line
tools/profile/perline_remarks.py <raw.gputrace> --json
```
Per shader (named via capture proximity, dedup'd across pipeline variants) it prints the
**occupancy levers**: `Spilled bytes`, `Temporary register count` (the Apple-GPU occupancy
driver), ALU/FP32/FP16 counts — then per source line: machine-instruction count, register
**spills**, unroll factor, + the MSL. Real hit: `compute_atmoMultiScatter` = 224 spilled
bytes / 124 temp registers / 17681 ALU, spilling at its main function. **Levers = spilled
bytes > 0 and high temp-register count** (kill Apple-GPU occupancy) + high per-line instr.

This is STATIC cost (register pressure/occupancy — properties of the machine code the compiler
measured per line). For actual per-line *time-spent* (what Xcode's Shaders tab shows), use
`runtime_perline.py` (§5c) — also headless. `gputrace shader-source`/`pprof --source-lines` do NOT
work here (Dawn shaders are all `main0`, no external `.metal`; their per-line is a static heuristic).

### 5c. Per-SOURCE-LINE RUNTIME TIME — HEADLESS via `runtime_perline.py`  ★★★
The other half of §5a: not *structural* cost but *time actually spent* per source line — the
statistical PC-sampling Xcode's Shaders tab shows, reconstructed offline (full RE in
gpuprofiler-raw-format.md §8c). Runs on the EXPORTED profiled `.gputrace` (needs the
`*.gpuprofiler_raw`); requires `dwarfdump` (Xcode CLT). No live GPU.
```bash
tools/profile/runtime_perline.py <exported.gputrace>            # per shader: hot source lines by %GPU-time
tools/profile/runtime_perline.py <exported.gputrace> --src      # + shader NAME/TYPE + the MSL source of each line
tools/profile/runtime_perline.py <exported.gputrace> --lines 12 --both --json
```
It joins three streams in the raw: **GPRWCNTR PC samples** (per-source `ShaderProfilerData` blobs)
× the **Program Address Buffer** (per-command `mappedAddress`→binary load-map) × the **Havested
Mach-O DWARF** (offset→program_source line + function). Output: each shader (binaryUniqueId + type)
by % of sampled GPU time, then its hottest source lines with the enclosing DWARF function. **line 0
= compiler glue / register spills / prologue** — a line-0-heavy shader is spill/occupancy-bound
(cross-check `perline_remarks.py`). ~2 min (parses the 2.7 GB export + dwarfdumps ~160 shaders).
⚠️ Was long stuck at "15% / unsolved"; the fix was the per-command PAB base + right stream (GPRWCNTR,
not the `usc` map) + range-containment (not a single global base). Ground truth = eyeball a hot line in Xcode.

**`--src` (name + type + source text, via `_srcmap.py`):** the compiled Mach-O has NO shader name —
only the Dawn capture does. `--src` decompresses the export's **`store0`** (zlib, ~5.5 GB in RAM,
+~90 s) to get the MSL `program_source` blocks + `Dawn_ShaderModule_*` labels, then bridges
`binaryUniqueId → block` (DWARF **barrier-call_line verification**, which separates same-family
siblings like nanRasterWorld1 vs nanRasterDepth; fuzzy fallback) → `block → name` (shared
`NodeBuffer` IDs + shader type) → aligns the block to DWARF line numbers → prints the source text
per hot line. **Reliable: source+type; exact name for barrier-verified compute** (the hot SW-raster
family) + distinctive shaders; **best-effort (confidence shown)** otherwise. Verified vs Xcode on
`nanRasterWorld1`: barrier line Xcode 3.52% ≈ tool 3.34% (±1 line: PC-sampling puts the stall on the
next instruction). Fully-exact universal naming would need parsing the Dawn capture object-graph.

### 5b. Quick `pass,ms` without any trace
Our `GpuProfiler` (`src/core/GpuProfiler.ts`, `Engine.resolveTimestampsAsync`). ⚠️ Apple
render‖compute overlap → per-pass timestamps are *not additive*; directional only
(memory `mobile-gpu-apple-arc`). Use `gputrace timing` above for trustworthy per-kernel ms.

## 6. The macOS 27 agentic CLI — what to download

Apple's WWDC26 "agentic Metal tools" suite. **None are on macOS 26.x** (confirmed:
`command not found`; absent from Xcode 26.0–26.6 release notes). They require
**macOS 27 + Xcode 27** (beta as of 2026-07; developer.apple.com/download).
`metalperftrace` is Apple-stated macOS 27. Once on macOS 27 they're bundled with
Xcode — no separate component.

| tool | does | trace |
|---|---|---|
| `gpucapture` | attach to a process, capture on demand | → `.gputrace` |
| `gpudebug` | REPL to navigate/inspect/profile a capture; `--json` | `.gputrace` |
| `metalperftrace` | collect perf traces + human summaries; `--json` | `.atrc` |

```bash
# gpudebug agentic form (macOS 27): point it at man gpudebug + a .gputrace
gpudebug --oneshot -t trace.gputrace -c "go commands/cb0/re0/draw0" -c "info pipeline"
gpudebug -t trace.gputrace -c "list"     # → reusable "Session <id>"; then: gpudebug -s <id> -c "next"
metalperftrace collect /tmp --last 5h ; metalperftrace overview t.atrc --aggregate --json
```
On macOS 26, the shader **compiler** toolchain *is* installable and cryptex-mounts
(`xcodebuild -downloadComponent MetalToolchain`; find it via
`dirname $(xcrun -f metal-source)`) — useful for static shader ISA
(`metal-objdump`, `applegpu-nt`) but it is **not** a trace reader.

## 7. Gotchas
- A Dawn trace has `captured_frames_count=1` — **no frame delimiters**; the whole
  run is one concatenated stream. Segment by swapchain writes if you need frames.
- Export only carries perf data if **Profile GPU Trace** was on during Replay; a
  raw capture has no `*.gpuprofiler_raw`.
- Dawn labels reach the trace only with `--enable-dawn-features=use_user_defined_labels_in_backend`
  (which `gputrace.sh` sets) — that's why `trace_static.py` can name the kernels.

Sources: developer.apple.com/metal/tools/ · .../xcode/debugging-with-interactive-command-line-tools
· .../xcode/replaying-a-gpu-trace-file · WWDC26 s.388 · Xcode 26.0/26.6 release notes.
