# Apple Silicon TBDR — what app developers actually get (reference copy)
Source: https://blakecrosley.com/blog/apple-silicon-tbdr  (derives from Apple's "Tailor your apps for Apple GPUs and TBDR" docs)
Fetched 2026-07-05 (WebFetch extraction, faithful).

---

## What TBDR is
GPU splits the render target into TILES, processes each tile on separate cores (often parallel), and DEFERS shading until all geometry for that tile is evaluated → shades only visible primitives. Contrasts with immediate-mode (IM), which processes primitives regardless of visibility.

**Tile memory:** bandwidth many× faster than device memory, latency many× lower, energy significantly less. Two render passes can overlap on HW (final stages of pass N to tile memory while vertex stage of pass N+1 starts).

## Imageblocks — custom per-pixel data in tile memory
2D structured image data in local memory (width, height, pixel depth); each pixel = multiple addressable components (e.g. albedo+specular+normal in one struct).
- available to kernel AND fragment functions
- PERSIST for the tile's lifetime, across draws/dispatches
- fragment shaders see only their fragment's imageblock data; compute threads access the whole imageblock
- keeps multi-stage pipelines (deferred shading, SS effects, custom blend) in local memory instead of round-tripping device memory → big per-frame win.

## Tile shaders — render + compute in one pass
Compute/fragment functions executing as part of a render pass; compute + save data to tile memory persistent between passes. Sidesteps the traditional split (separate passes that communicate via device memory). Metal 4: tile shaders pair with unified `MTL4ComputeCommandEncoder` — collapsing render-vs-compute boundaries the Apple HW doesn't need.

## Raster order groups — ordering concurrent fragment threads
Metal guarantees the APPEARANCE of draw-order blending, but fragment shaders run concurrently → custom blend against another triangle's result = race. ROGs synchronize only threads targeting the same pixel/sample (annotate memory pointers). Recent Apple GPUs: sync individual imageblock channels + threadgroup memory, multiple order groups for finer sync.
Deferred-shading example: group1 = g-buffer fields (albedo/normal/depth), group2 = accumulated lighting → coalesces both phases into ONE pass; non-conflicting reads run concurrently.

## MSAA that tracks unique samples per pixel (A11+)
HW tracks whether a pixel has a primitive edge; runs per-sample blend only when necessary. Overlapping edges: A11+ blends only for the unique colors (2× not 3×). An opaque triangle on top collapses the pixel to ONE color. Apps can extend with a tile shader that resolves opaque samples before translucent blend — resolve stays in tile memory.

## What it means for architecture
1. **Tile memory is the budget** — imageblocks/tile-shaders/ROGs/sample-coverage all exist to keep work in tile memory and out of device memory (faster, cooler).
2. **Render and compute are not different worlds** — multi-phase algorithms run inside one render pass (imageblock persistence + tile shaders).
3. **Concurrency is the default; ordering is opt-in** — ROGs are how you say "this RMW depends on order"; default is unordered concurrency.

## FAQ highlights
- Imageblock vs threadgroup memory: imageblock = structured 2D image data w/ addressable slices; threadgroup = flat scratch allocation.
- Apple has NOT published silicon-level details (cluster counts, ALU widths, raster specifics) per GPU gen — treat non-developer.apple.com numbers as unverified.
