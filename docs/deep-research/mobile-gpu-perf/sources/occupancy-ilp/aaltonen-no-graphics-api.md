# No Graphics API — Sebastian Aaltonen (blog)

URL: https://www.sebastianaaltonen.com/blog/no-graphics-api

Argues for a pointer-based compute-first model. Relevant parts: raw-pointer/scalar loads vs typed/texel buffers, VGPR register pressure, and mobile TBDR memory behaviour.

## Raw pointers & wide loads
- Replace opaque buffer descriptors with 64-bit pointer semantics.
- **"Wide loads are significantly faster, especially if the struct contains narrow 8- and 16-bit fields"** — GPUs are ALU-dense with big register files, but their **memory paths are relatively slow** vs CPUs.
- Wide load + unpack in-shader beats texel buffers for compact data: **raw buffer loads have up to 2× higher throughput and up to 3× lower latency than texel buffers.**

## VGPR register pressure (the key claim for the grass march)
- Texel buffers cause **"register bloat"**: an `RGBA8_UNORM` load to `vec4` **allocates four vector registers immediately.**
- By contrast, `uint8x4` raw data **consumes just a single 32-bit register** when using wide raw loads with in-shader unpacking.
- **"The register lifetime is much shorter"** with pointer-based loads.
- 64-bit pointer raw loads directly from VRAM into groupshared memory further reduce register bloat.

## Mobile GPUs / TBDR / SSBOs
- Framebuffer fetch on mobile gives pixel shaders **direct low-latency read+write to previously rasterized pixels** (vs traditional blending).
- **SoA data layout causes significantly more cache misses for non-linear index lookups.**
- Meshlet binning is a poor fit on mobile: tile size is commonly 16×16 to 64×64 px, making meshlets too coarse a primitive for binning.
- (The post is light on hard SSBO-latency numbers for mobile; the concrete numbers above are the transferable ones. Aaltonen's broader well-known claim that SSBOs/structured buffers are comparatively slow on mobile TBDR appears elsewhere; this post's evidence is the texel-buffer register-bloat and wide-load throughput/latency figures.)

## Relevance to grass march
- **Confident lever:** if the baked field is sampled as a typed texture returning `vec4`, that's up to 4 VGPRs per live sample. Storing the field as a **packed raw buffer** (e.g. `uint8x4` / `uint` bit-packed) and unpacking after a **wide raw load** cuts to ~1 VGPR/sample AND shortens register lifetime — directly attacking the VGPR pressure that caps ILP/occupancy for the march.
- **Confident:** prefer narrow packed fields + unpack; up to 2× throughput / 3× lower latency vs texel-buffer equivalents matters most for a latency-exposed march.
- **Needs measurement on WebGPU:** WebGPU has no raw pointers; storage buffers are the closest analog to raw loads and textures are the "typed" path. Whether the texel→storage-buffer swap yields the register-bloat win depends on the backend (naga/tint → Metal/SPIR-V) — must inspect emitted VGPR counts.
