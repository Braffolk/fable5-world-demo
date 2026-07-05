# Mali and Nanite: Technical Deep Dive on Mobile GPU-Driven Rendering

Source: https://developer.arm.com/community/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/mali-and-unreal-engine-s-nanite-enabling-the-future-of-mobile-graphics

## Executive Summary

Mali's Tile-Based Rendering (TBR) architecture fully supports Nanite's GPU-driven rendering pipeline.

## Hardware Requirements for Nanite on Mali

- **Dispatch and draw indirect** - standard Vulkan features enabling GPU-driven techniques
- **VK_KHR_shader_image_atomic_int64** - long supported by Mali drivers
- **VK_KHR_compute_shader_derivatives** - recently added support improving texture quality in compute shading

**Driver requirement:** R49 or newer. Older drivers need shader workarounds due to bugs in Nanite bit operations.

## Nanite Pipeline Breakdown

1. Render non-Nanite geometry via traditional vertex-fragment pipeline
2. GPU-based cluster culling using previous Z-buffer; LoD selection
3. Classify and allocate visible clusters into rasterization bins
4. Build visibility buffer with 64-bit atomics:
   - Small triangles → compute-based rasterizer
   - Large triangles → fixed-function hardware
5. Re-run against current Z-buffer (minimal work as most triangles already rasterized)
6. Classify and allocate visible pixels into shading bins
7. Shade via compute shaders

## Mali Tile-Based Rendering Interactions

### Performance Data (Mori Demo)

30 FPS maintaining 6M+ visible triangles. Proxy geometry tests:

| Triangles/Mesh | Max Rendered | Time w/o Nanite | Time w/ Nanite | FPS Gain |
|---|---|---|---|---|
| 30K | 1.2M | 29.67ms (33.71 FPS) | 23.83ms (41.96 FPS) | +24% |
| 76K | 2.2M | 37.40ms (26.74 FPS) | 22.75ms (43.96 FPS) | +64% |
| 634K | 18M | 199.36ms (5.02 FPS) | 26.16ms (38.23 FPS) | **+661%** |

### Visibility Buffer & Cluster Culling Efficiency (5M-triangle scene)

| Metric | No Nanite | HW Rasterizer Only | HW + Compute Rasterizer |
|---|---|---|---|
| Time (ms) | 0.25 | 0.10 (60% reduction) | 0.04 (84% reduction) |
| GPU Active Cycles | 100.0% | 39.5% | 15.7% |
| Visible Primitives | 100.0% | 49.7% | 0.31% |
| Input Primitives | 100.0% | 17.33% | 0.04% |
| Overdraw (milli-threads) | 0.183 | 0.112 | 0.106 |

**Key insight:** "A significant portion of Nanite's performance gain comes from cluster culling." TBR GPUs benefit particularly because they have relatively low vertex throughput.

## Hardware vs. Compute Rasterization

### Compute Shader Rasterizer Advantages
- **Micro-triangle inefficiency:** Hardware rasterizers struggle with small/thin triangles; compute handles these efficiently
- **Quad overdraw:** Micro-triangles not covering full fragment quads generate discarded fragments; compute eliminates this waste
- **Thin triangle spanning:** Long thin triangles spanning multiple tiling bins incur significant costs on TBR; Nanite clustering replaces these with well-formed triangles

MinPixelsPerEdgeHW tuning (controls triangle routing HW vs compute):

| Triangles/Mesh | 8px | 16px | 32px | 64px |
|---|---|---|---|---|
| 30K | 25.63ms | 24.59ms | 24.17ms | 24.62ms |
| 76K | 24.28ms | 23.39ms | 22.72ms | 23.65ms |
| 634K | 29.57ms | 28.13ms | 26.26ms | 26.95ms |

**Critical Mali advantage:** Fragment and non-fragment work can overlap. Fine-tuning the HW/compute balance produces meaningful improvements.

## Resolution Scaling with Nanite

Nanite LoD selection ties geometry cost to resolution:

| Triangles/Mesh | 80% Resolution | 50% Resolution | Improvement |
|---|---|---|---|
| 30K | 23.83ms (41.96 FPS) | 17.99ms (55.57 FPS) | 32% |
| 76K | 22.75ms (43.96 FPS) | 17.93ms (55.78 FPS) | 21% |
| 634K | 26.16ms (38.23 FPS) | 20.48ms (48.84 FPS) | 22% |

Works well with Arm Accuracy Super Resolution (ASR) and Neural Super Sampling (NSS).

## Mali-Specific Optimization Challenges

### Atomic Operations
Mali has long supported atomics, but workloads experience suboptimal performance if atomics not used efficiently. Profile via Streamline.

### Empty/Small Compute Dispatches
GPU-driven pipelines issue many small/empty dispatches; on Mali these introduce significant overhead. Design shaders for meaningful work per dispatch.

### Compute Shading Limitations
- Disables bandwidth-saving techniques like framebuffer compression
- Reduces overlap opportunities between fragment and vertex processing
- Compute shader derivatives now supported but challenging
- "Compute shading remains a challenging area and is not recommended for most situations"

### Virtual Shadow Maps
Mali-G1 lacks hardware clip distance support; demo uses default shadow maps instead.

## Hidden Surface Removal (HSR) Interaction
Recent Mali GPUs include HSR hardware fragment pre-pass. HSR may reduce value of Z-buffer culling, but still performs work to determine whether fragments from culled triangles are visible.
