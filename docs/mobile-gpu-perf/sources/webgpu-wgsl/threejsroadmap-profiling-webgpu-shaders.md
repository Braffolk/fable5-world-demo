# What's Actually Slow? Profiling WebGPU Shaders in Three.js

Source: https://threejsroadmap.com/blog/profiling-webgpu — Dan Greenheck, June 1, 2026.
(Fetched via curl + text extraction; original 403s to WebFetch.)

## The problem with frame time
CPU frame time (`performance.now()` deltas) rolls JS + all GPU work into one figure and,
crucially, **is pinned to the display refresh rate while you're inside budget** — it reads
~16.6 ms at 60 Hz whether the GPU spent 2 ms or 14 ms, hiding your true headroom. To find
what one pass costs you must measure the GPU's own clock.

## Timestamp queries in three.js
WebGPU exposes a `timestamp-query` device feature: GPU writes its clock at pass begin and
end; the difference is elapsed GPU time. **Optional feature** — most desktop adapters have
it; **some mobile and software adapters don't** (handle the missing case). three.js wraps
querySet creation / resolve / readback for you. Two pieces: a renderer flag + a resolve call.

### Enable (off by default, small cost)
```javascript
import * as THREE from 'three/webgpu';
const renderer = new THREE.WebGPURenderer({ trackTimestamp: true });
await renderer.init(); // adapter/device acquired here; feature requested here
```
With `trackTimestamp: true`, three.js requests the feature and **automatically injects
timestamp writes into begin/end of every compute and render pass it issues**.

### Resolve
```javascript
import { TimestampQuery } from 'three/webgpu';
renderer.compute(update);
const ms = await renderer.resolveTimestampsAsync(TimestampQuery.COMPUTE);
console.log(`compute pass: ${ms.toFixed(3)} ms`);
```
- `TimestampQuery.COMPUTE` covers compute passes; `TimestampQuery.RENDER` covers render passes.
- Async because the value must read back from GPU after the work finishes.
- **Returned number is the SUM of all passes since the last resolve.** To time ONE pass,
  resolve after a single dispatch.
- It's **GPU time, not wall-clock** — excludes JS queueing overhead and queue-wait time.
  Exactly the isolated cost of the pass.

## Time one pass in isolation
```javascript
async function timePass(pass) {
  renderer.compute(pass);
  return renderer.resolveTimestampsAsync(TimestampQuery.COMPUTE);
}
```
Run it twice → two different numbers. A single reading is close to meaningless.

## Handling noisy measurements
Noise sources: GPU idles at low clock and **only boosts under sustained load (first several
passes run slower until clock stabilizes)**; caches warm over the same window; OS/other tabs
steal GPU time; readback jitter.

Recipe: **warm up ~20 passes and discard**, then **collect ~50 samples, report the MEDIAN**
(shrugs off GC/OS spikes; average gets dragged up). Keep **p10/p50/p90** to see distribution tightness.

```javascript
async function measurePass(pass, { warmup = 20, measure = 50 } = {}) {
  for (let i = 0; i < warmup; i++) {                    // warm-up, discard
    renderer.compute(pass);
    await renderer.resolveTimestampsAsync(TimestampQuery.COMPUTE);
  }
  const samples = [];
  for (let i = 0; i < measure; i++) {                   // measure, keep all
    renderer.compute(pass);
    samples.push(await renderer.resolveTimestampsAsync(TimestampQuery.COMPUTE));
  }
  samples.sort((a, b) => a - b);
  const pct = (p) => samples[Math.floor((samples.length - 1) * p)];
  return { p10: pct(0.1), p50: pct(0.5), p90: pct(0.9) };
}
```
20-warmup / 50-measure is a reasonable start. If >~25% variance between consecutive runs,
widen the window — the noise floor is dominated by **readback variance**, not real change.

## Relevance to us — THE profiling recipe for this machine
This is exactly how to measure **grass vs raster vs resolve** separately on THIS machine:
construct WebGPURenderer with `trackTimestamp:true`, then use `resolveTimestampsAsync`
around each pass (or `TimestampQuery.RENDER` for render passes). Because the sum resolves
per-pass-type since last resolve, isolate a lane by resolving immediately after just that
lane's dispatch/draw. Warm up + median + p10/p90 matches our existing p50/p95 discipline.
