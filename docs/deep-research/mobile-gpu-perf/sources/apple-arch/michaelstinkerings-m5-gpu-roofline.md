# Apple M5 GPU Roofline Analysis

Source: https://www.michaelstinkerings.org/apple-m5-gpu-roofline-analysis/

## Scalar ISA Architecture

The M5 GPU employs a **scalar instruction set** where "each register stores one 32-bit value per thread." A `float4` operation decomposes into **4 separate scalar `fmadd` instructions** rather than one wide SIMD op.

**Key evidence:** Switching from `float4` cross-dependent FMA chains to scalar floats with 8 independent accumulator chains, throughput jumped from 791 GFLOPS to 3,760 GFLOPS — a **4.75x improvement**. Confirms the compiler lowers vector types into sequential scalar operations.

## Roofline Results

| Metric | Standard (float4) | Optimized (scalar) | FP16 (scalar 32-chain) |
|--------|-------------------|-------------------|------------------------|
| Bandwidth ceiling | 122 GB/s | 122 GB/s | 122 GB/s |
| Compute peak | 815 GFLOPS | 3,849 GFLOPS | 6,004 GFLOPS |
| Ridge point | ~6.5 F/B | ~31 F/B | ~49 F/B |

Standard (float4) methodology measures only **22% of hardware capability** because float4 serialization prevents accessing the full ALU pipeline.

## Instruction-Level Parallelism (ILP)

The M5 requires **8 independent scalar accumulator chains** to saturate FP32 FMA throughput → implies a **4-cycle FMA latency**. Using 16 chains caused 5% throughput reduction from register pressure — the register file becomes the occupancy bottleneck before additional ILP helps.

## FP16 Double Rate

FP16 executes at double the rate of FP32 via **more dedicated 16-bit ALUs** (not the same ALUs running faster). Vector `half4` ops achieve nearly perfect **2.0x speedup** (1.96x measured); scalar FP16 with sufficient ILP reaches only **1.59x** over scalar FP32 — the double-rate pipeline needs 2× the independent instructions to stay saturated, hitting register file limits first.

## GPU Specs (derived)

- 10 cores, **128 ALU lanes per core**
- Peak clock 1,578 MHz
- Theoretical FP32 peak 4,040 GFLOPS
- Measured scalar FP32 peak 3,814 GFLOPS (94.4% utilization)
- GPU power 18.2W sustained

## The float4 Performance Trap

A compute-bound `float4` kernel hits ~800 GFLOPS. Restructured for scalar float with 8 independent chains → 3,849 GFLOPS — a **4.7x difference**.
- AI 8–31 F/B: standard kernels compute-bound; optimized remain bandwidth-bound → 2–5x higher throughput
- AI >31 F/B: both compute-bound, optimized achieves 4.9x higher ceiling

## Graphics: TBDR

- Fill rate (RGBA8, FMA=0): M5 **1,284 GPixels/s**; 128× overdraw writes to tile memory only; final flush ~2% overhead
- R8 vs RGBA8 tile saturation (FMA=128): R8 sustains 1,067 GPixels/s while RGBA8 drops to 91 — an **11.7x gap**: tile memory bandwidth (not DRAM) becomes the bottleneck for fat G-buffers
- Texture caching: coherent access shows zero degradation 16 KB → 64 MB via 32 MB SLC

## Critical developer insight

Unoptimized Metal using `float4` captures only ~22% of M5 compute. Restructure for scalar with 8+ independent accumulators to unlock 4–5× in compute-bound regimes. FP16 packed (half4) gives near-2x. Register file is the occupancy limiter.
