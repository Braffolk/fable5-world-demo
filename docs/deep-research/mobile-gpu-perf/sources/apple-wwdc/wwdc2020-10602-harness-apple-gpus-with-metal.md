<!-- source: https://developer.apple.com/videos/play/wwdc2020/10602/ -->
<!-- Apple WWDC20 video "Harness Apple GPUs with Metal". Transcript content extracted via WebFetch. -->

# Harness Apple GPUs with Metal (WWDC20 10602)

Speaker: Guillem Viñals, Metal Ecosystem. TBDR architecture fundamentals + how to exploit tile memory.

## TBDR
Two-phase: Tiling (vertex + primitive binning) then Rendering (per-tile pixel processing). On-chip Tile Memory eliminates dedicated VRAM. Each tile processed independently.

## Load / store actions (critical bandwidth lever)
- Load: only load what you need; prefer **Clear over Load** where possible (saves color/depth/stencil transfers).
- Store: only store necessary data (e.g. just the main color attachment); avoid unnecessary write-back to main memory.

## Hidden Surface Removal
Front-most visible layer tracked per pixel before fragment shading; pixel-perfect, submission-order independent. Sort opaque → alpha-test/discard → translucent; don't interleave opaque/non-opaque or differing write masks.

## Programmable blending
Fragment shaders read pixel data directly from Tile Memory (no dedicated blend unit; blending always in Tile Memory). Lets you merge multiple passes into one — deferred lighting, fog, custom blends.

## Memoryless render targets
`memoryless` storage mode for intermediate textures (e.g. G-buffers) — no allocation in system memory, pairs with programmable blending.

## Efficient MSAA
Edge tracking: non-edge pixels blend per-pixel, edge pixels per-sample. Samples live in Tile Memory, resolved on flush. Use memoryless MSAA texture + Resolve action; never load/store MSAA samples.

## Modern Apple GPU (A11+) features
- **Imageblocks**: 2D structure in Tile Memory; load/store whole image data in one op.
- **Tile shaders**: compute kernels dispatched mid-render-pass to access imageblocks, interleaved with draws in submission order, auto-synchronized. Use for tiled deferred light culling.
- **Imageblock sample coverage control**: custom MSAA resolve (HDR color, linear depth).
- **GPU-driven rendering**: Argument Buffers (scene data on GPU) + Indirect Command Buffers (GPU encodes its own draws) — removes CPU-GPU sync points / readbacks.

## Deferred-rendering optimized recipe
Programmable blending (G-buffer in Tile Memory) + memoryless targets + tile-shader light culling → merge 3+ passes into 1; big footprint/bandwidth win.
