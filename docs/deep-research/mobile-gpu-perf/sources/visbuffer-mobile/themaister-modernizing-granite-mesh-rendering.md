# Modernizing Granite's Mesh Rendering (themaister)

Source: https://themaister.net/blog/2024/01/17/modernizing-granites-mesh-rendering/

## Overview
Rewrite of Granite's mesh rendering to leverage mesh shaders while keeping fallbacks for hardware lacking support.

## Core Design Requirements
- **Reasonable Fallbacks:** avoids hard-requiring VK_EXT_mesh_shader. Meshlet concept benefits MDI paths and direct draws on mobile lacking MDI.
- **Nanite's pitfall:** vis buffers with per-primitive varyings in fragment shaders require geometry shaders or vertex duplication → ~50% penalty; "5-15% FPS loss" in games like Immortals of Aveum.

## Meshlet Format
256-primitive meshlets base unit (lower culling overhead, efficient MDI batching). Sublets in 8×32 groupings for HW-specific specialization.

### Stream Encoding
- Index buffers: 5-bit encoding, 15 bits/primitive
- Positions: 3×16-bit SINT shared exponent
- UVs: 2×16-bit SINT
- Normal/Tangent: 4×8-bit SNORM octahedral
- Tightly-packed bits with 64-bit shifts (not bitplane), balancing NVIDIA/AMD.

## Culling Pipeline

### Back-face Culling (window-space cross product, adaptive precision)
```glsl
bool cull_triangle(vec2 a, vec2 b, vec2 c) {
  precise vec2 ab = b - a;
  precise vec2 ac = c - a;
  precise float pos_area = ab.y * ac.x;
  precise float neg_area = ab.x * ac.y;
  bool active_primitive;
  if (abs(pos_area) < 16777216.0)
    active_primitive = pos_area > neg_area;
  else
    active_primitive = pos_area >= neg_area;
  return active_primitive;
}
```

### Micro-poly Rejection (reject primitives whose bbox smaller than pixel grid)
```glsl
if (active_primitive) {
    const int SUBPIXEL_BITS = 8;
    vec2 lo = floor(ldexp(min(min(a, b), c), ivec2(-SUBPIXEL_BITS)));
    vec2 hi = floor(ldexp(max(max(a, b), c), ivec2(-SUBPIXEL_BITS)));
    active_primitive = all(notEqual(lo, hi));
}
```

## Performance Benchmarks

### NVIDIA RTX 3070 (63.59M triangles)
- vkCmdDrawIndexed baseline: **5.5 ms**
- Frustum culling: **4.3 ms**
- MDI with back-face cull: **3.9 ms**
- Meshlet (decoded): **4.0 ms**
- Per-primitive culling: **3.3 ms**
- Micro-poly rejection: **1.9 ms**
- Vertex ID passthrough: **1.0 ms**

Vertex ID optimization uses VK_KHR_fragment_shader_barycentrics to export only IDs + transform indices; fragment shaders fetch attributes directly. Shifts pressure from export buffers to compute.

### AMD Steam Deck (RDNA2)
- RADV culling: 9.6 ms; MDI: 8.9 ms; Wave32 32/32 meshlet: 9.3 ms. Deck prefers smaller meshlets.

### AMD RX 7600 (RDNA3)
- Wave32 64/64 (encoded): 2.5 ms; Wave64 64/64 (decoded): 2.2 ms; proprietary decoded: 2.1 ms. Prefers larger meshlets.

## Occlusion Culling
Two-phase visibility (Phase 1 previously-visible, Phase 2 test new against HiZ). Conservative sphere-to-screen projection; HiZ mip selected via findMSB so only 2×2 sampling needed.

## Key Insights
1. Vendor divergence: format must adapt to 32/64/128/256 primitive meshlet preferences.
2. Compression vs decode-speed trade-off.
3. **Micro-poly dominance:** in dense scenes, micro-poly rejection gives 3.5-10× gains over back-face culling alone.
4. Task shaders skepticism (payload limits, vendor inefficiency); prefer indirect mesh dispatch.
