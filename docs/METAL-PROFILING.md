# Metal GPU trace of the game (game-only `.gputrace`)

Goal: a replayable Xcode Metal-debugger `.gputrace` of the **steady-state game frame**,
without the ~11 GB of boot GPU compute (heightfield/erosion/flow/bark bakes) drowning it.

## The mechanism — `?profile=1` two-device split

Dawn's `DAWN_TRACE_*` records a WebGPU device from **creation to destruction** and has
**no start/stop hook**, so a normal single-device run always captures boot + game together.

`?profile=1` fixes this structurally (see `src/core/ProfileBoot.ts`):

1. The whole world loads on a throwaway device labelled **`laas-loading`** — this is where
   the expensive bakes run.
2. After `buildScene`, we swap to a **fresh** device labelled **`laas-render`** and run the
   game loop there. The ~500 MiB of GPU-only bake textures are read back and re-uploaded;
   the nanite mega-buffers re-upload for free (they are CPU-backed, kept alive by the
   auto-set `noreleasemirrors`); the render graph (post + nanite frame + water) is *built*
   on the render device, not rebuilt.
3. `DAWN_TRACE_DEVICE_FILTER=laas-render` therefore records **only the game** — the boot
   compute ran on the discarded `laas-loading` device and is excluded by construction.

The normal path (no `?profile`) is untouched and still single-device.

## The full pipeline — 3 steps (one is MANUAL)

```
1. CAPTURE  (headless)  gputrace.sh              → raw .gputrace   (structure + resources, NO timing)
2. EXPORT   (⚠ MANUAL, Xcode — tools CANNOT do this step)
            open raw .gputrace in Xcode → let it replay/profile → File ▸ Export
            with "Embed performance data" ENABLED
                                                 → exported .gputrace (adds *.gpuprofiler_raw + store0)
3. ANALYZE  (headless)  run_all.sh <raw> <exported>  → one folder of per-shader perf + source
```

⚠️ **Step 2 is required and cannot be automated.** A raw Dawn `.gputrace` has **no timing** — the
numbers come from *replaying* it on the GPU, which only Xcode's Metal debugger does. The analysis
tools (`run_all.sh`, `runtime_perline.py`, `profile_report.py`, `gputrace_timing.sh`) all need the
**exported** bundle; without "Embed performance data" you get an export with no counters and the
tools have nothing to read. See "Step 2 — Xcode export" below. Analysis is documented in
**`tools/profile/README.md`**; this doc covers steps 1–2.

## `tools/profile/gputrace.sh` — one command

```
tools/profile/gputrace.sh
# or tune:
CAPTURE=12 tools/profile/gputrace.sh

URL='http://localhost:5173/?scene=world&nanite=1&dpr=2&nanodisp=1&clhw=1&clhwmax=32&profile=1&grass=0&nanshadow=0&ksplit=1&fp16w=1' CAPTURE=2 tools/profile/gputrace.sh
```

It launches Chrome with Dawn tracing armed, waits for the render-device swap
(`[profile] swapped…` on Chrome's stderr), records `CAPTURE` seconds of the game, then
quits Chrome — destroying `laas-render` finalizes the `.gputrace`. Open the result under
`$OUT*` (default `/tmp/laas_trace*`) in Xcode's Metal debugger.

Knobs: `URL`, `OUT` (`DAWN_TRACE_FILE_BASE`, default `/tmp/laas_trace`), `CAPTURE` (seconds,
default 10), `READY_TIMEOUT`, `CHROME`, `PROFILE_DIR`.

Requires the dev server on :5173 (`npm run dev`) and Google Chrome (its Dawn backend
honours `DAWN_TRACE`). The `--enable-dawn-features=use_user_defined_labels_in_backend,disable_symbol_renaming`
flags surface our named buffers/textures/kernels/passes in the capture.

### The bare command (what the script runs)

```
DAWN_TRACE_FILE_BASE=/tmp/laas_trace \
DAWN_TRACE_DEVICE_FILTER=laas-render \
MTL_CAPTURE_ENABLED=1 \
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --disable-gpu-sandbox --user-data-dir=/tmp/chrome-metal \
  --disable-features=SkiaGraphite --no-first-run --no-default-browser-check \
  --enable-dawn-features=use_user_defined_labels_in_backend,disable_symbol_renaming \
  'http://localhost:5173/?scene=world&nanite=1&dpr=2&clhw=1&profile=1'
```

Then close Chrome to finalize the trace. The device filter (`laas-render`) is the only
thing making it game-only — drop `?profile=1` and it captures boot too (the 11 GB path).

## Step 2 — Xcode export **with performance data** (manual, required for analysis)

The capture from step 1 is structure-only. To get timings / counters / per-line shader cost, a
human must replay it in Xcode and export it **with performance data embedded**:

1. **Open** the raw `.gputrace` (e.g. `/tmp/laas_trace-*.gputrace`) in **Xcode** (double-click, or
   Xcode ▸ Open). It loads in the Metal debugger.
2. **Let Xcode generate the profile** — it replays the frame on the GPU to gather counters/timings.
   If it doesn't start automatically, use the GPU-frame **"Profile"** action (the per-encoder GPU
   timeline / Shaders tab populating = the profile is ready). Give it a few seconds to finish.
3. **File ▸ Export…**, and in the export dialog **enable "Embed performance data"**, then save.
   This writes a *second* `.gputrace` bundle that contains a `*.gpuprofiler_raw` (the profiled
   counters/timings) **and** `store0` (the shader `program_source`). That second bundle is the
   `<exported.gputrace>` the analysis tools consume.

Then, headless: `tools/profile/run_all.sh <raw.gputrace> <exported.gputrace>` → a results folder
(summary + per-shader runtime/static breakdowns + full `.metal` source). See `tools/profile/README.md`.

⚠️ If you export **without** "Embed performance data", the bundle has no `*.gpuprofiler_raw` and the
runtime/timing/counter tools will report nothing to read. The raw-only tools (`trace_static.py`,
structure/VRAM) still work on step-1's capture alone.

## Notes / caveats

- There is a small unavoidable prefix on `laas-render` before steady state: the texture
  writeback + the cheap re-heal bakes (sky LUTs, IBL, GI warm, cloud noise) run on the
  render device right after the swap. Navigate a few frames in to reach a representative
  game frame. The *expensive* bakes are excluded.
- Peak memory is higher during a profile run (both devices are briefly live before
  `laas-loading` is destroyed). Fine on an M1 Max; close other GPU apps if tight.
- If `$OUT*` is empty after a run, this Chrome build may not honour `DAWN_TRACE` — check
  `chrome://gpu` shows Dawn/Metal, and that you launched Google Chrome (not the headless
  shell).
- `.gputrace` ≠ `.trace`: `xctrace --template "Metal System Trace"` produces a `.trace`
  (Instruments *timeline*), not the replayable per-encoder `.gputrace` this doc is about.
