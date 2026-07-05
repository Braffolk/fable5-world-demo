<!-- source: https://developer.apple.com/videos/play/wwdc2020/10603/ -->
<!-- Apple WWDC20 video. Transcript content extracted via WebFetch. -->

# Optimize Metal apps and games with GPU counters (WWDC20 10603)

Speaker: Guillem Vinals Gangolells, Metal Ecosystem. THE profiling talk — 150+ GPU counters organized into limiter groups.

## Apple GPU architecture (context for the counters)
- Unified memory (CPU+GPU share System Memory, no dedicated VRAM). TBDR.
- Two phases: **Tiling** (vertex, primitive binning into tiles) then **Rendering** (rasterize, HSR visibility, fragment shade, store).
- Per-core: Shader Core (ALU), Texture Unit (TPU), Pixel Backend, dedicated **Tile Memory** pool, L1 caches for ALU+TPU; shared **Last Level Cache** across cores.
- Peak-rate hierarchy: **Tile Memory > GPU LLC > System Memory (DRAM)**.

## The performance limiters (the counter-driven diagnosis path)

### ALU limiter
- Throughput: **16-bit FP = double rate; 32-bit FP = full rate; 32-bit int/complex = half rate or less.**
- SIMD = 32 threads, one program counter. Divergent branches: ALL threads execute ALL paths, masked lanes still burn cycles.
- Levers: approximations/LUTs; **F16 over F32**; avoid implicit conversions; avoid FP32 texture/buffer inputs; compile `-ffast-math`.
- Warning: 100% ALU-limited ≠ efficient — could be 50% utilization if FP32-only.

### Texture read (TPU) / write (Pixel Backend) limiters
- 128-bit formats (RGBA32Float): quarter-rate sampling. Aniso costs rate.
- Levers: mipmaps; lower aniso; compressed formats (**ASTC** for assets, lossless for runtime textures); small pixel sizes; watch MSAA; avoid divergent writes (Pixel Backend is separate HW optimized for coherent writes).

### Tile Memory load/store limiter
- Backs threadgroup memory, imageblocks, programmable-blending color attachments.
- Levers: **reduce threadgroup atomics** (use threadgroup parallel reductions or SIMD-lane ops instead); align to 16 bytes; reorder access patterns.

### Buffer read/write limiter (device memory via Shader Core)
- Address spaces: `device` (RW), `constant` (RO).
- Levers: pack tighter; smaller types; vectorize loads/stores; **avoid device atomics and register spills**; offload some work to textures (different cache).

### GPU Last Level Cache limiter
- Shared, caches texture+buffer, **stores device atomics**.
- Levers: fix texture/buffer limiters first; shrink working set; **refactor device atomics → threadgroup atomics**; improve spatial/temporal locality.

### Fragment input interpolation limiter
- Fixed-function full precision. Only lever: remove unused vertex attributes.

## Memory bandwidth counter
- Measures System Memory ↔ GPU. Load only what the pass needs; store only what future passes need; texture compression (ASTC + lossless) is critical.

## Occupancy analysis (compute / vertex / fragment)
- **Low occupancy is NOT always a problem** (low vertex occupancy fine if fragment occupancy suffices; low occupancy fine if resources fully used).
- Causes of low occupancy: shaders **exhausted internal resources (Tile/Threadgroup memory)**; threads finishing faster than GPU can create new ones; small render area / small compute grid.
- Method: correlate occupancy with other counters; check resource exhaustion; analyze dispatch patterns.

## Hidden Surface Removal (HSR) efficiency
- Overdraw ratio = Fragment Shader invocations / Pixels Stored. Counters: pixels rasterized, FS invocations, pixels stored, pre-Z test fails.
- Sort meshes: opaque → alpha-test/discard/depth-feedback → translucent. Don't interleave opaque with non-opaque, nor opaque with different write masks.

## Tools
- **Metal System Trace (Instruments)**: overview, top-limiter track; affected by thermals.
- **Metal Debugger (Xcode)**: all 150+ counters at encoder/draw granularity, per-draw counter mode, bound-resources; NOT affected by thermals.

## Case study: Respawnables Heroes (iPad Pro)
- Start ~12.82ms GPU, ALU-limited (deferred) + texture-sampler (post). Fixes: raise FP16 utilization; RGBA16Float cubemap shared→private storage (enables lossless compression); ASTC on assets. Result: steady 120 FPS.
