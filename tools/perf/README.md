# tools/perf — throttle-resistant A/B frame-time harness

`interleaved_ab.mjs` compares two URLs (arm A vs arm B) by frame-time p50/p95, sampled in-page via rAF deltas in one headless WebGPU Chromium session.

**Why interleaved:** the M1 Max thermally drifts during perf work (same build+pose read 30→51ms p95 sequentially). Alternating A,B,A,B,… makes both arms share the drift; the reported **median per-cycle delta (B−A)** cancels it to first order. Per-reading `pmset -g therm` CPU_Speed_Limit is recorded and throttled readings flagged.

Example (canonical worst-pose, A/B'ing a flag):

```sh
node tools/perf/interleaved_ab.mjs \
  --base "http://localhost:5173/?scene=world&nanite=1&dpr=2&nanodisp=1&clhwmax=32&grass=0&nanshadow=0&ksplit=1&fp16w=1&ctxsm=1&cam=-582.1,302.4,1006.1,2.5692,-0.0077" \
  --alt  "http://localhost:5173/?scene=world&nanite=1&dpr=2&nanodisp=1&clhwmax=32&grass=0&nanshadow=0&ksplit=1&fp16w=1&ctxsm=1&cam=-582.1,302.4,1006.1,2.5692,-0.0077&middz=1" \
  --cycles 3 --warmup 8 --sample 10
```

Needs the dev server (`npm run dev`, port 5173). Add `--headed` if headless WebGPU fails. Results: table + verdict on stdout, JSON at `tools/perf/last_ab.json`.
