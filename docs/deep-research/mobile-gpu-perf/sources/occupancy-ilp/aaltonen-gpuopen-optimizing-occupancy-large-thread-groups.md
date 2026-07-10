# Optimizing GPU Occupancy and Resource Usage with Large Thread Groups — Sebastian Aaltonen (Second Order), GPUOpen

URL: https://gpuopen.com/learn/optimizing-gpu-occupancy-resource-usage-large-thread-groups/

Compute-shader occupancy on AMD GCN. Directly relevant to the VGPR-pressure / scalar-load levers for the grass march.

## GCN Compute Unit
- 4 SIMDs, 64 KiB register file each (256 KiB total). **65,536 VGPRs per CU.** 512–800 SGPRs. 64 KiB LDS. **Max 10 waves/SIMD** (wave = 64 threads).

## Budget to fit two 1024-thread groups per CU
| Resource | Limit | Per-thread budget |
|---|---|---|
| VGPRs | 65,536 | **32 VGPRs max** |
| LDS | 64 KiB | 32 KiB/group |
| Waves | 40 (4/SIMD) | 16 waves/1024 threads |

- 40 VGPRs/thread → only ONE group fits: **40% occupancy** (4 vs 10 waves), 50% LDS idle, 37.5% register file wasted.
- 32 VGPRs/thread → TWO groups run simultaneously → better latency hiding, async barrier hits improve instruction mix. **"50% performance boost" documented.**

## VGPR-reduction techniques (the load-bearing part for the grass march)
- **Scalar loads:** load wave-invariant data via `ByteAddressBuffer` (raw buffer) → generates SCALAR loads → stored in SGPRs at **~64× less storage than vector loads**. `SV_GroupID` is wave-invariant. **Typed buffers/textures do NOT support scalar loads.** Example: a 4×4 view/proj matrix costs 16 VGPRs via typed loads but **0 VGPRs** via scalar loads from `ByteAddressBuffer`.
- **Data reduction:** drop unused W; 4×3 matrices instead of 4×4 for affine.
- **Bit-packing:** pack 2D int coords 16+16 bits (GCN single-cycle bitfield extract/insert); pack 32 bools into one VGPR.
- **Booleans:** GCN stores `bool` in a 64-bit SGPR (1 bit/lane), zero VGPR cost — don't emulate with int/float. Store flags in sign bits of always-positive floats (free via `abs()`/`saturate()`). `countbits()` for prefix sums.
- **Control flow:** use `[loop]` attribute to prevent unrolling so loop-counter-dependent loads stay inside the loop body → **reduces VGPR lifetime** by confining data to loop scope.
- **16-bit registers:** GCN3+ packs two 16-bit values per VGPR; Vega doubles fp16 throughput. Good for 2D/3D address math and post filters.
- **LDS as VGPR-pressure relief:** load redundant data once to LDS; temporarily spill registers to LDS during peak VGPR usage.

## When large thread groups are worth it
- Worth it: neighborhood processing (post filters, TAA, blur), multi-pass with LDS between passes, physics island solvers.
- Avoid if no LDS need — a 64–256 thread group is fine.
- Neighborhood border overhead: 8×8 = 56%, 16×16 = 27%, 32×32 = 13% (2D); 4³ = 70%, 8³ = 48% (3D). Larger groups cut boundary redundancy.

## Takeaway
"Hitting the goal of 32 VGPRs is hard" for complex shaders, but reaching it enables dual-group execution and large gains.

## Relevance to grass march
- **Note the tension with Volkov:** this article optimizes for HIGH occupancy (more waves). Volkov shows low occupancy + high ILP also works. Both agree the real currency is **VGPR pressure** and **register-resident working set**.
- **Confident lever:** move wave-invariant march params (camera, wind, field metadata) to raw-buffer **scalar loads** → frees VGPRs → either raises occupancy or leaves headroom for ILP prefetch. Note WebGPU exposes this only partially (storage buffers ≈ raw loads; no explicit scalar-load intrinsic), so this needs measurement on the actual backend.
- **Confident:** demote march math to **fp16** where precision allows (address/step math) → halves VGPR pressure.
- **Confident:** `[loop]` / short live ranges to shrink VGPR lifetime.
