# Compute Shader Rasterization vs Hardware Rasterization (Tellusim)

Source: https://tellusim.com/compute-raster/

## Summary
Tellusim compared compute-based software rasterization against hardware rasterization across GPUs. "Compute shader rasterization outperforms dedicated mesh shader and multi-draw indirect approaches" on most platforms.

## Test Configuration
498,990 64×128 Meshlets, back-face culling only. Meshlet data fits compute rasterization well (independent, tightly-packed vertex/index data → one workgroup per meshlet, one triangle per thread). Tests: Single Draw Indirect Pass (32-bit indices), Mesh Shader, MultiDrawIndirect/ICB/loop, Compute Shader (depth-only).

## Performance Results (Triangles Per Second)

| GPU | Single DIP | Mesh Shader | MDI/ICB/Loop | Compute Shader |
|-----|-----------|-------------|--------------|----------------|
| GeForce 2080 Ti | 12.05 B | 12.57 B | 12.63 B | **17.26 B** |
| GeForce 1060 M | 3.86 B | 3.90 B | 4.55 B | — |
| Radeon 6700 XT | 14.73 B | 4.38 B | 3.63 B | **16.74 B** |
| Radeon RX 5600M | 4.87 B | 1.11 B | 7.57 B | — |
| Radeon Vega 56 (macOS) | 2.40 B | 796.0 M | — | **3.17 B** |
| Apple M1 (macOS) | 1.37 B | 739.0 M | — | **2.30 B** |
| Apple A14 (iOS) | 666.1 M | 475.0 M | — | **1.02 B** |
| Intel UHD Graphics | 680.0 M | 396.5 M | 556.1 M | — |
| Adreno 660 (Android) | 565.2 M | 31.17 M | 497.3 M | — |

## Key Technical Findings

### Atomic Operations Constraints (CRITICAL for us)
Implementation "limited to depth-only mode because 64-bit atomics are not available on Mobile devices and Metal." Quote: "a compute shader extension allowing atomically to write payloads into the image will change everything. Without it, we are limited to 32-bit payload data and must perform a redundant triangle intersection."

Proposed extension:
```
uint imageAtomicPayloadMax(gimage2D atomic_image, gimage2D payload_image,
  ivec2 P, uint atomic_data, gvec4 payload_data);
```

### Depth Packing Strategy
Current implementation uses 32-bit atomics with `atomicMax` for depth compare:
`imageAtomicMax(out_surface, ivec2(...), floatBitsToUint(z))`
This restricts payload to a single 32-bit value → separate triangle intersection needed rather than writing depth+visibility/material IDs simultaneously.

### Tile-Based Architecture Issues
"MultiDrawIndirect doesn't work well on mobile because of the tile-based rendering" — dramatic degradation on Adreno 660 (31.17M vs 565.2M tri/sec for single DIP).

## Conclusions
- "Single shader type is better than 14 dedicated shader types"
- Everything (including raytracing) implementable on compute
- Back-face early reject: `if(det >= 0.0f) return`; per-pixel depth via atomics.
