<!-- source: https://developer.apple.com/videos/play/wwdc2020/10632/ -->
<!-- Apple WWDC20 video "Optimize Metal Performance for Apple silicon Macs". Transcript content extracted via WebFetch. -->

# Optimize Metal Performance for Apple silicon Macs (WWDC20 10632)

The single richest talk for the OFFICIAL shader-level levers. TBDR context.

## 1. 16-bit data types
`half`/`short` instead of `float`/`int` when possible: fewer registers → higher shader-core occupancy → faster arithmetic; conversions typically free. Use small types for indices/positions that fit in 16 bits.

## 2. Address spaces — constant-buffer prefetch into uniform registers
- `device` = RW, unlimited, unique-per-thread access.
- `constant` = RO, limited, same data reused across threads.
- **Constant data is preloaded into "uniform" registers IF offset is known at compile time AND array size is known at compile time.** Encapsulate a fixed-size array in a struct (`LightInfo lights[MAX_LIGHTS]`) passed as `constant` so the compiler can prefetch — vs `device LightArray&` (unknown size) which cannot preload.

## 3. Half-precision literals
Metal promotes to the highest-precision operand. `half r = a + b - 2 + 5;` promotes to float. Use `h` suffix: `half r = a + b - 2h + 5h;` stays half.

## 4. Memory access / register-spill avoidance
- **Avoid stack-allocated arrays indexed by a runtime value** — may spill. Prefer loops that the compiler can unroll at compile time.
- Use **signed** loop-index types (unsigned needs wrapping semantics; signed allows vectorization).
- Colocate struct fields (or use vector types like `float2 AC`) so loads vectorize.

## 5. Tile memory & TBDR
- Minimize system-memory bandwidth via load/store actions.
- **`memoryless` storage mode** for attachments that don't need to persist (`descriptor.storageMode = .memoryless`).
- Programmable blending reads current pixel from Tile Memory — no system-memory round-trip.

## 6. Pass optimization
- Merge adjacent passes to same attachments.
- Use `makeParallelRenderCommandEncoder` (sub-encoders combine into one pass) instead of separate command buffers.
- Avoid attachment ping-ponging (up to 8 color attachments / MRT).

## 7. HSR
- Draw opaque first. Use `[[early_fragment_test]]` to keep HSR when a fragment function writes buffers/textures. Avoid write masking. Write ALL pass attachments even if some phases don't use them (avoid unintentional write masking).

## 8. Tile shaders
Per-tile compute inside a render pass (`renderPassDesc.tileWidth/tileHeight/threadgroupMemoryLength`); access imageblock + threadgroup memory without flushing tile memory to system memory.
