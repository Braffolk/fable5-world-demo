# Occupancy Explained — GPUOpen (AMD RDNA)

URL: https://gpuopen.com/learn/occupancy-explained/

## Definition
Occupancy = ratio of assigned wavefronts to max available slots. RDNA2+: each SIMD has 16 wavefront slots (8 assigned = 50% occupancy).

## Hardware
- Work executes in wavefronts (32 or 64 threads). A SIMD executes one wavefront/cycle but **switches between assigned wavefronts to hide memory latency** — while one waits for data, another does ALU.
- Wave slots/SIMD: RDNA1 = 20, RDNA2/RDNA3 = 16.
- Three occupancy-limiting resources: **VGPRs** (per-SIMD, e.g. RX 7900 XTX = 1536 VGPRs/SIMD), **SGPRs** (rarely limiting on RDNA), **LDS** (shared, must be on same WGP).

## Theoretical vs measured
- **Theoretical** occupancy = compile-time, from shader resource needs.
- **Measured** occupancy can be lower due to: too few total waves to fill SIMDs; launch-rate limits (can't schedule new waves faster than old retire); dependencies/barriers preventing overlap.

## Latency hiding
Memory latency (hundreds of cycles) hidden by switching wavefronts IF other waves have ALU work ready. RGP shows hidden latency as green portion of latency bars.

## Key performance caveat
**"Peak occupancy does not always mean peak performance."** Higher occupancy can increase cache thrashing when accesses cluster temporally; lower occupancy can improve perf by cutting cache pressure. (Same message as Volkov.)

## Optimization approach
- Low theoretical occupancy → reduce VGPR/LDS pressure or adjust threadgroup size.
- High theoretical but low measured → ensure enough waves generated; increase per-wave work to amortize launch overhead; for pixel shaders reduce VS→PS interpolants / improve LOD.

## Register spilling
Compiler can spill registers to memory to keep occupancy, but memory carries **~100× latency penalty** vs registers. Check pipeline state tab in RGP.

## Tools
- Radeon GPU Profiler (RGP): pipeline tab = theoretical occupancy + limiting resource; wavefront occupancy tab = measured over time.
- Radeon GPU Analyzer (RGA): VGPR pressure per instruction.
- PIX + AMD plugin: WaveOccupancyLimiters counters identify the constraining resource.

## Barriers
Compute shaders: up to 16 barriers/SIMD-pair; a threadgroup spanning 2+ wavefronts needs a barrier at sync points — can bottleneck in extremes.

## Relevance to grass march
- Confirms the diagnostic path: **measure whether occupancy is limited by VGPRs, LDS, or wave count**, and whether measured << theoretical (→ dependency/launch-bound, not resource-bound). If measured occupancy is already fine but perf is poor, the loop is latency-exposed → the fix is ILP (prefetch), not more occupancy.
