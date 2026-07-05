# The Visibility Buffer: A Cache-Friendly Approach to Deferred Shading

Burns & Hunt, Intel Labs. JCGT Vol. 2, No. 2, 2013. Source PDF: jcgt-visibility-buffer.pdf
Original: https://jcgt.org/published/0002/02/04/paper.pdf

This is the FOUNDATIONAL vis-buffer paper. Key relevance to us: the bandwidth argument for choosing vis-buffer over g-buffer, which is MOST pronounced exactly on bandwidth-limited / mobile / integrated GPUs.

## Abstract / core thesis
Forward rendering shades in triangle-submission order → over-shading (occluded fragments wastefully shaded). Deferred shading (g-buffer) avoids over-shading but has two deficits:
1. **The g-buffer is large — often 20+ bytes per visibility sample. Bandwidth to read/write large g-buffers can be prohibitive on mobile or integrated GPUs.** (THE mobile argument.)
2. Separation of shading and visibility is incomplete — surface attributes eagerly computed on occluded fragments, wasting texture bandwidth.

Proposal: replace the g-buffer with a **visibility buffer** storing only a triangle index + instance ID per sample, encoded in **as few as four bytes** (eight bytes for large/tessellated scenes). Deferred pass accesses triangle data with this index to compute barycentrics and interpolate vertex data. Minimizing the memory footprint reduces the working set → improved performance on bandwidth-limited GPU platforms, especially high-resolution workloads.

## The bandwidth numbers (the crux)
- G-buffer typically **16 to 32 bytes per visibility sample** in optimized high-quality systems.
- Example: 4-megapixel display, 4× 24-byte samples/pixel at 60 Hz → **46 GB/s** of bandwidth just for the uncompressed g-buffer write + read, one lighting pass. Thus AA or resolution is often sacrificed on economical hardware.
- Vis-buffer at 1080p 8× sampling = **64 MB** vs **398 MB** for a 24-byte-per-sample g-buffer.
- Cost paid: recomputing barycentrics, vertex attribute interpolation, tex-coord differentials, and re-running vertex shaders at pixel frequency. But "except for very high-bandwidth discrete GPU platforms, bandwidth is the constraining resource for mainstream graphics workloads, especially on power-sensitive micro-architectures." Compute-bandwidth ratio keeps growing → sensible tradeoff.
- Memory accesses (loading tri/vertex data for visible triangles) are **highly coherent in screen-space → high cache hit rates even if cache much smaller than total working set.**

## The visibility buffer format
- Each sample stores a **4-byte integer encoding a triangle id + instance id.** For scenes too large or tessellated, **8 bytes** per sample.
- Deferred pass: load visible vertices, transform, shade, intersect (ray-tri), interpolate into screen-space, illuminate. Reference g-buffer for comparison is 24 bytes (Listing 1): half4 position (8B), half2 normal (4B), unorm84 diffuse (4B), unorm84 specular_exp (4B), uint triangleID (4B).

## Pipeline (three phases)
1. **Visibility pass:** render all geometry, record depth + primitive id into buffer. No other data written/computed (cheap, like a z-prepass).
2. **Worklist pass:** build tile worklists (tiles = 16×8 pixels). Read vis buffer, build list of tile-shader pair records; sort by shader, store packed tile ids with per-shader offset array. Cheap — ≤10% of frame.
3. **Shading pass:** one compute kernel dispatch **per unique surface material**, ranging over tiles containing that material. Does vertex gather, vertex shade, attribute interpolation, surface shading, illumination, multisample blend, final color write.

Shading-pass pseudocode (Fig 3): loadVisSample → decode instanceID + triID → bail if shaderID != this shader → load indices → load vertices → ray-tri intersect + interp barycentrics → ShadeVertex/ShadeSurface → light loop. Loads are coherent, shared across many pixels depending on triangle size.

## Results
- Scenes: Warehouse (559k tris, 24 lights/px), Ecosystem (431k unique / 1.74M rasterized tris — many small triangles, 8 lights), Sphere Cloud (760 unique tris instanced 200×, 32 lights). All 1080p.
- GPUs (Table 2, memory hierarchy): AMD HD 5750 (40kB/256kB, 73.6 GB/s), Nvidia 560 Ti (384kB/512kB, 128 GB/s), Intel HD 4000 (256kB/8MB, 25.6 GB/s), **Intel Iris Pro 5200 (512kB/8MB/128MB eDRAM, 25.6 GB/s)**.
- **Little/no benefit at only 2M visibility samples; advantage grows with higher sample rates, particularly on bandwidth-constrained Intel platforms.** Iris Pro (128MB eDRAM cache) shows the biggest speedups.
- Ecosystem (many small tris) challenges the approach — depends on spatial locality of visible geometry during shading; smallest-cache platforms never benefit there, integrated parts still do.
- Sphere-cloud (mesh fits in cache on all platforms) → outperforms on all HW → demonstrates impact of geometric working-set size.
- Fig 6 breakdown (8× MSAA): vis pass far cheaper than g-buffer pass (like z-prepass); shading pass sometimes outperforms g-buffer lighting despite compute overhead. Worklist gen is the small black segment.
- "In none of our scenes were frame rates sensitive to the addition of math instructions in the shaders, indicating we are NOT compute-bound." (bandwidth is the limiter.)

## Related-work notes relevant to tilers
- PowerVR (Imagination) low-power GPUs already do a similar tiled deferred strategy in HW/driver: resolve visibility for a tile before interpolating attributes / running the pixel shader; rasterized-triangle references stored in a tile-sized HW **tag buffer**. This paper = a full-screen application-layer implementation of that strategy on DX11-class GPUs.
- Tile size: 16×8. Tiles smaller than 64 px underperformed (small workgroups undersubscribe GPUs).

## Takeaways for OUR mobile arc
- The vis-buffer choice is VALIDATED specifically for bandwidth-limited/integrated/mobile GPUs — g-buffer bandwidth (46 GB/s example) is the thing we avoid. We are aligned.
- Gains scale with sample count / resolution and with cache size (eDRAM/TBDR on-chip tile memory is favorable).
- The redundant recompute (ray-tri intersect, vertex re-transform, barycentrics) is the accepted price; it is compute we trade for bandwidth we don't have on mobile.
