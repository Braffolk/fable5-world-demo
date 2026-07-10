# WebGPU Performance Issue — three.js Forum (discourse #87939)

Source: https://discourse.threejs.org/t/webgpu-performance-issue/87939

## Original question — martinp (Nov 5, 2025)

WebGPURenderer significantly underperforms WebGLRenderer. Benchmark on M4 Pro MacBook,
Chrome, uncapped FPS, V-Sync disabled, MeshBasicMaterial cubes in a small area, three.js r181:

| Cube Count | WebGPU (forceWebGL) | WebGPU | WebGL |
|-----------|---------------------|--------|-------|
| 5k  | 50 fps  | 140 fps | 350 fps |
| 10k | 14 fps  | 60 fps  | 130 fps |
| 50k | 1-2 fps | 3-6 fps | 40 fps  |

## Key answers

- **dubois**: WebGPU renderer still experimental. Drawing thousands of individual meshes
  is slow; with **instancing** you can draw millions.
- **phil_crowther**: use WebGPU-specific features (compute shaders), TSL; implementation
  still in development, current three.js emphasis is on increasing performance.
- **Mugen87 (maintainer) — the critical finding**: references GitHub issue **#30560**.
  Same scene of **thousands of non-instanced meshes runs 60 FPS on WebGL but drops to
  15 FPS on WebGPU**. Root cause: **the UBO system has severe performance issues with
  many render items**. Workaround: **use instancing and batching whenever possible**.
  Team is actively working on the UBO organization.

## Relevance to us
Per-object draw items are expensive on the three.js WebGPU path due to UBO handling.
Our nanite path is largely compute/instanced, but any per-object node graph work
(many small draws/uniform updates) pays this tax. Batch/instance aggressively.
