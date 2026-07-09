# The Snapdragon X Elite's Adreno iGPU

Source: https://chipsandcheese.com/p/the-snapdragon-x-elites-adreno-igpu
Author: Chester Lam, Chips and Cheese, 2024-07-04

## Overview
Adreno X1 (Adreno 741) = scaled-out Adreno 730. Up to 1.5 GHz (vs 730's 900 MHz).

## Shader Processor Architecture
- SPs contain two uSPTPs (Micro Shader Processor Texture Processors). Similar to AMD WGP / Nvidia Maxwell-Pascal SM.
- Per scheduler partition: **64-wide FP32 unit**; **FP16 at double rate (2x)**; eight special function units.

### Register File
- Adreno X1: **192 KB per uSPTP (96 KB per scheduler partition)**
- Adreno 730 baseline: 64 KB per partition → 50% increase
- Wave-granular allocation: "each register allocated by the compiler takes up a wave-wide portion of the register file."

### Wave Size and Occupancy
Large waves (**64-wide or 128-wide**). Larger waves increase register pressure; with 96 KB/partition, register file capacity can constrain occupancy for register-heavy shaders. Max occupancy undisclosed.

## Memory Hierarchy
- **L1 texture cache**: 2 KB per uSPTP, ~960 GB/s; texture-only (compute bypasses to next level).
- **Cluster cache (new)**: 128 KB per cluster (3 clusters = 384 KB), latency 56.62 ns.
- **L2**: shared, texture + compute.
- **SLC**: 6 MB, ~211 GB/s.
- 128-bit LPDDR5X, up to 64 GB.

## Local Memory (GMEM)
3 MB on-chip GMEM (vs 730's 2 MB): tiled render buffer + LDS-equivalent + render cache.
- float4 loads ~L1 texture bandwidth (~960 GB/s), below AMD/Intel.
- Per-kernel limit 32 KB local memory; GPU-wide 384 KB concurrent.

## Compute
- **1536 FP32 units**; 1.25 GHz tested (up to 1.5 GHz).
- FP16 double-rate keeps pace with Intel/AMD.
- INT32 add poor via Vulkan; INT64 mediocre-to-poor; no FP64.

## Trade-offs
Strengths: wide EUs, flexible GMEM, DRAM bandwidth, FP16. Weaknesses: low cache bandwidth/mediocre latency, register file limits large-wave occupancy, 2 KB texture-only L1, poor 64-bit int. Cache architecture tuned for "DX11 era where pixel shader work dominates."
