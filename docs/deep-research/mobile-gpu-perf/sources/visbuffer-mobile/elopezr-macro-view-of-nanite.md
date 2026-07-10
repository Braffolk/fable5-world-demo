# A Macro View of Nanite (elopezr)

Source: https://www.elopezr.com/a-macro-view-of-nanite/

## Overview
Nanite decouples geometry rendering from material evaluation. ~5M triangles/frame, >90% software-rasterized.

## Pipeline Stages

### Nanite::CullRasterize
- **Instance Culling:** GPU frustum+occlusion culling. Nanite.Views buffer for frustum; HZB from previous frame, forward-projected. Both visible and occluded instances written to separate buffers.
- **Persistent Culling:** fixed-count compute threads output visible cluster counts by type (compute vs HW raster) into MainRasterizeArgsSWHW buffer.
- **Clustering/LOD:** clusters (patches). Clusters appear/disappear across LOD with seamless stitching.

#### Rasterization Strategy
**Hardware:** 3,333 instances × 384 vertices. Cluster = 384 vertices = 128 triangles (multiples of 32/64 for wavefront efficiency).
**Compute:** 34,821 compute groups, 128 threads/group (one thread per triangle), ~5M triangles. Selection based on triangle size relative to pixels.
**Post Pass:** secondary raster using BuildPreviousOccluderHZB (HZB + Z-prepass depth).

### Visibility Buffer
**R32G32_UINT texture (64-bit per pixel):**

| Field | Bits | Content |
|-------|------|---------|
| R [31:7] | 25 bits | Cluster ID |
| R [6:0] | 7 bits | Triangle ID |
| G | 32 bits | Depth |

Rationale: every pixel's material evaluated exactly once; no occluded resources accessed. Rather than storing barycentrics, derive quantities by intersecting triangle with camera ray and recomputing/interpolating vertex data on-the-fly. Max ~2^32 triangles via cluster+triangle IDs.

### Nanite::EmitDepthTargets
Outputs Depth, Motion Vectors, Material Depth (unique depth per material ID, in depth-stencil for Early Z), Nanite Mask.

### Nanite::BasePass
- **Material Classification:** compute analyzes fullscreen vis buffer → 20×12 tile (240 texel) Material Range texture (R32G32_UINT); each tile = 64×64 px region, encodes material ID range present.
- **GBuffer Emission:** one fullscreen quad drawcall per material ID. Quad samples Material Range; if material absent in tile, x-coord set to NaN → quad discarded. Material ID = 14 bits (16,384 max materials). Material depth test EQUAL for rapid rejection; stencil marks Nanite vs regular geometry.

## Key Design Decisions
- **Vis buffer vs G-buffer:** decouples geometry from materials, delays material eval until visibility confirmed, eliminating texture/resource access for occluded pixels.
- **HZB occlusion:** forward-project previous frame HZB for conservative occlusion.
- **SW vs HW raster:** >90% compute raster, addressing HW small-triangle scheduling/occupancy issues.
