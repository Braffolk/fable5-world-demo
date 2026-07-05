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

## `tools/gputrace.sh` — one command

```
tools/gputrace.sh
# or tune:
CAPTURE=12 tools/gputrace.sh
URL='http://localhost:5173/?scene=world&nanite=1&dpr=2&profile=1&grass=0' tools/gputrace.sh
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
  'http://localhost:5173/?scene=world&nanite=1&dpr=2&profile=1'
```

Then close Chrome to finalize the trace. The device filter (`laas-render`) is the only
thing making it game-only — drop `?profile=1` and it captures boot too (the 11 GB path).

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
