# Ideas parked 2026-07-09 (user-flagged for the shader-opt phase)

## Bitmask voxel-DDA (kVoxScatter per-pixel ray march) — potentially critical when shader opt reaches the voxel raster

The per-pixel brick march is Amanatides-Woo 3D-DDA over a 4×4×4 = 64-cell grid whose occupancy is TWO u32
words (`cOccLo`/`cOccHi`, VoxelBrick words 0/1). Because the whole grid fits in 64 bits, the loop can be
replaced with **set intersection in bit operations**: build (or look up) the ray's crossed-cell set as a
64-bit mask, then

```
hitMask = (rayMaskLo & occLo, rayMaskHi & occHi)
hit     = hitMask != 0
firstHit = firstSetBitAlongRayOrder(hitMask)   // → cell index → entry t for the depth key
```

— no per-step marching at all. Construction options for the ray mask: a small direction-bucketed LUT
(the 4³ line rasterization per (entry cell, quantized direction) is precomputable), or incremental mask
building with shifts per axis step (still loop-free per-plane: OR 3-4 shifted slabs).

**Why parked:** the current fixed loop is ≤10 iterations of ~8 ALU with early-exit (typical 2-4 steps —
hit or exit). Not worth the complexity until a fresh profile shows the DDA hot. **When to revive:** the
voxel-raster shader-opt pass — if per-line profiling shows the march loop dominating kVoxScatter, this
removes the loop-carried dependency chain entirely (the current march is serial: each step depends on the
previous min-t compare), which is exactly the kind of latency chain that hurts on Apple TBDR at low
occupancy. The firstSetBit-in-ray-order needs care (bit order ≠ march order; may need the LUT to store
masks in march order per direction octant, or ctz on a re-permuted mask).

Context: the 6-step exhaustion-ACCEPT bug (view-dependent "crossing planes", fixed 2026-07-09) is what
put eyes on this loop; the fix raised the bound to the true worst case (10) with exit-reject.
