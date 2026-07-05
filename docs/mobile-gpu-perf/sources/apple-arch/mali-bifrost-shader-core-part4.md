# The Mali GPU: An Abstract Machine, Part 4 — The Bifrost Shader Core

Source: https://developer.arm.com/community/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/the-mali-gpu-an-abstract-machine-part-4---the-bifrost-shader-core
Author: Peter Harris

## Bifrost Shader Core Architecture

### Execution Engines and Quad-Vectorization
Three execution engines in Mali-G71, each with a composite arithmetic pipeline. Quad-vectorization: "Threads are grouped into bundles of four, called a quad, and each quad fills the width of a 128-bit data processing unit." Individual threads perceive scalar 32-bit operations.

### Register File and Thread Occupancy
"The Mali-G71 provides 64x 32-bit registers while still allowing the maximum thread occupancy of the GPU." (Improvement over Midgard — removes the thread-count vs register-resources trade-off.)

### Sub-Word Data Types
Native int8/int16/fp16. "A single 128-bit maths unit can therefore perform 8x fp16/int16 operations per clock cycle, or 16x int8 operations per clock cycle."

## Data Processing Units
- **Load/Store**: 16KB L1 per core; accesses across a thread quad optimized to reduce unique cache requests.
- **Varying unit**: interpolates 128-bits per quad per clock; a mediump (fp16) vec4 takes two cycles per four thread quad.
- **Texture unit**: one bilinear filtered texel per clock; DEPTH_COMPONENT16/24 is now single-cycle (double vs Midgard).
- **ZS/Blend**: tile-memory depth/stencil, blend, pixel-local-storage/framebuffer-fetch.

## Index-Driven Vertex Shading (IDVS)
Splits vertex shading: (1) position shading before tiling/culling, (2) varying shading only for non-culled primitives — eliminates redundant compute/bandwidth.

## GPU-Level
Scales 1–32 cores. ~one 32-bit pixel per core per clock; L2 typically 64KB per shader core.
