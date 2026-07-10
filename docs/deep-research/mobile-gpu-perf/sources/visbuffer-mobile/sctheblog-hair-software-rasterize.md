# Software Hair Rasterization in WebGPU (Scthe blog)

Source: https://www.sctheblog.com/blog/hair-software-rasterize/

## Overview
Software rasterization for real-time hair in WebGPU, building on UE5 Nanite and Frostbite hair. Converts strand data to rasterized pixels with order-independent transparency. (Note: same author as the nanite-webgpu issue #1 referenced by the user re: u16 depth packing / z-fighting risk in WebGPU with no atomic<u64>.)

## Projecting Hair as Billboards
Camera-facing billboards, width = fiberRadius. Per strand point: tangent toward next point, bitangent = cross(tangent, towardsCameraVector), offset ±fiberRadius, apply view-proj. Calculations MUST be in world space (view-space towardsCamera=vec3(0,0,1) fails — sign inconsistencies across strands).

## Edge Function (Shoelace)
`f = (xb - xa)(yp - ya) - (yb - ya)(xp - xa)`. Triangle: all 3 edge functions consistent sign; quad: all 4.

### A·x + B·y + C incremental form
```
A = ya - yb
B = xb - xa
C = xa·yb - ya·xb
```
Add A per pixel horizontally, B per row. ~7% faster for triangles. Quads need 4 sets (12 floats) → register-heavy; HairFinePass uses it, HairTilesPass does not. Quad raster ~4× more register-intensive than triangle.

## Segment-Space Coordinates
v (long axis): project pixel onto segment line, distance/length, saturate. u (short axis): width at start perpendicular to tangent, project end width via scalar projection, interpolate, project pixel onto side edge, distance/width, saturate.

Alpha:
```wgsl
var alpha = 1.0 - abs(interpolationWidth * 2.0 - 1.0);
if (HAIR_USE_ALPHA_QUADRATIC) { alpha = sqrt(alpha); }
alpha = saturate(alpha * HAIR_ALPHA_MULTIPLIER);
```

## Per-Pixel Linked Lists & Tile-Based Processing
Two-phase transparency:
- **Phase 1 (Coarse):** list of hair segments per tile. Memory: tileCount × avgHairSegmentsPerTile × sizeof(HairSegmentId).
- **Phase 2 (Fine):** each workgroup one tile from global queue, one thread per pixel. First thread broadcasts segment data, others apply edge tests. Buffer: numWorkgroups × tileSize² × avgSamplesPerPixel × sizeof(HairSample). Sort+shade in fine rasterizer. Avoids pre-allocating fixed memory tied to resolution (helps high-DPI).

## Key Technical Claims
- 16 points per strand typical (short hair 3).
- Half-pixel offset for single-sample rasterization: vec2f(0.5, 0.5); top-left rule.
- Working code: github.com/Scthe/frostbitten-hair-webgpu.
