# Nanite GPU Driven Materials — Graham Wihlidal, Epic (GDC 2024)

Source PDF: gdc2024-nanite-gpu-driven-materials.pdf (168 slides). Original: https://media.gdcvault.com/gdc2024/Slides/GDC+slide+presentations/Nanite+GPU+Driven+Materials.pdf

Scope note: this talk EXPLICITLY does NOT cover cluster build, culling, or the rasterizer (those are in the 2021 Deep Dive and HPG 2022 keynote). It covers the GPU-driven MATERIAL pipeline across three eras: Initial (UE5.0 2022), Programmable Raster (UE5.1 2022), Latest (UE5.4 2024). Extracted for the load-bearing atomic/depth-packing confirmation + material-culling levers relevant to our election and deferred shading.

## Material Pipeline Overview — the atomic-max vis buffer (CONFIRMS 2021, with 2024 bit split)
- Geometry virtualized into a SINGLE GPU buffer → whole scene rasterized in a single draw-indirect (HW path) + a single async dispatch-indirect (SW path).
- **Rasterize geometry into a 64-bit visibility buffer using atomic max:**
  - **32-bit Depth in the high bits**
  - **32-bit Triangle / Visible Cluster ID in the low bits**
  - (Note: the 2021 Deep Dive quoted 30 depth / 27 cluster / 7 tri; by 2024 the framing is a clean 32/32 split. Either way it is a single 64-bit `InterlockedMax`/atomicMax = depth-test + payload-write in one op. WebGPU has NO atomic<u64>, so a port must compromise depth bits into 32 or use a redundant intersection — the core constraint for our election.)
- GBuffer still required for rendering (material graph eval). Design goal: keep existing GBuffer/deferred logic "fully unaware of Nanite or the visibility buffer." Non-Nanite renders into GBuffer as usual; Nanite populates GBuffer from the VBuffer via several full-screen passes with "plenty of tricks."

## Initial Material Pipeline (UE5.0)
- **Full-screen pass per unique material**, each draw assigned a unique depth value; **depth-EQUALS testing** shades the GBuffer with the right material. Materials fully decoupled from geometry → even 10,000 different meshes sharing one material = a single full-screen draw.
- Based on Eidos' Dawn Engine ("Deferred+ Next Gen Culling and Rendering for Dawn Engine") with improvements.
- Improvement to depth testing: want depth-equals to quickly reject pixels not covered by a material, but NOT writing actual scene depth → a synthetic **material-ID depth** is written (Material ID → depth value), so the depth-test HW does the material culling for free.
- Demo material counts (before GPU culling): Lumen in the Land of Nanite ~500; Valley of the Ancients ~2000; City Sample (Matrix Awakens) ~5000, millions of instances.

## Later eras (Programmable Raster UE5.1, Latest UE5.4) — headlines
- **Programmable Raster (5.1):** artists can control the rasterizer from material graphs (e.g. world-position offset, masked/alpha-test, pixel-depth offset during the vis-buffer raster). Relevant because masked foliage (leaves/grass) needs to evaluate opacity DURING rasterization, which reintroduces material work into the otherwise fixed-function raster — a known cost for aggregate geometry.
- **Latest (5.4):** further evolution toward compute-based material classification / shading (moving off the full-screen-quad-per-material toward tile/compute binning), continuing the direction flagged in the 2021 Deep Dive ("this is an area being heavily reworked... will likely switch to completely compute in the future").

## Takeaways for us
1. Confirms the load-bearing fact: Nanite's vis buffer = one 64-bit atomic max, 32-bit depth high / 32-bit ID low. Our WebGPU election cannot do atomic<u64> and must document its depth-precision compromise (Tsl.ts / NaniteRaster).
2. Material culling via depth-EQUALS on a synthetic material-ID depth is the portable trick (no wave intrinsics needed) — cheaper than per-pixel material tests.
3. Programmable raster (masked opacity in the raster pass) is exactly what makes leaves/grass expensive — material evaluation leaks back into rasterization for aggregate geometry.
