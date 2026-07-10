# three.js #32735 — Inefficient compute pipeline caching in WebGPU

Source: https://github.com/mrdoob/three.js/issues/32735
Author: DomenicoBruzzese, Jan 12 2026. Status: Open. Labels: Enhancement, WebGPURenderer.

## The bug
The WebGPU compute pipeline cache keys on the **ComputeNode instance id**, not on the
shader source + binding layout:
```javascript
_getComputeCacheKey( computeNode, stageCompute ) {
    return computeNode.id + ',' + stageCompute.id;
}
```
So running the **same compute shader logic across N objects with different I/O buffers**
means N distinct `ComputeNode` instances → **N separately compiled `GPUComputePipeline`
objects**. Running the same shader over 1,000 objects compiles 1,000 pipelines —
substantially wasteful for complex shaders.

## Proposed fix
Key on the WGSL source id (`stageCompute.id` already indexes the WGSL) + a binding layout key:
```javascript
_getComputeCacheKey( stageCompute, bindings ) {
    return stageCompute.id + ',' + this.backend.getComputeBindingsLayoutKey( bindings );
}
```
This lets you define a TSL shader once, instantiate with different input buffers, batch via
`renderer.compute([node1, node2, ...])`, and get **one pipeline + N dispatches in one encoder**.

## Current limitations noted
- A single reusable ComputeNode with `.needsUpdate = true` can't batch multiple different-input calls.
- One giant consolidated buffer risks memory exhaustion.
- `renderer.compute(arrayOfNodes)` batches into one encoder but **still doesn't dedupe pipeline compiles**.

## Relevance to us — DIRECT
The task's core question: **do WE spawn many compute nodes for the same logic (per-level
dispatches)?** If our nanite cull/raster/resolve/grass creates a distinct ComputeNode per
LOD level / per band / per tile that all share identical WGSL, this cache-key bug means each
one triggers a **duplicate WGSL→MSL pipeline compile** — a boot-time hang (see wgpu #4456 /
Tint) AND redundant runtime state. Audit for per-level/per-band ComputeNode instances sharing
one kernel; if found, reuse a single node or await the three.js fix / patch the cache key.
