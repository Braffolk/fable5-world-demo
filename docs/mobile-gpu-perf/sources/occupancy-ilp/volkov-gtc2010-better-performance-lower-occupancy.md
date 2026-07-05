# Better Performance at Lower Occupancy — Vasily Volkov (UC Berkeley), GTC 2010

Source PDF: `volkov-gtc2010-better-performance-lower-occupancy.pdf` (75 slides). URL: https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf

**This is the canonical "occupancy is not the metric — ILP hides latency too" reference.** Directly relevant to the grass-march question: *is the march ILP-limited or occupancy-limited?*

## Thesis (slides 1–4)
- Common advice: run more threads/multiprocessor and more threads/block, on the premise "this is the only way to hide latencies." **This is a fallacy.**
- **Faster codes run at LOWER occupancy** (slide 3, measured):
  - SGEMM: CUBLAS 1.1 (512 threads/block, 67% occ, 128 Gflop/s) → CUBLAS 2.0 (64 threads/block, 33% occ, 204 Gflop/s). **8× smaller blocks, 2× lower occupancy, 1.6× higher performance.**
  - 1024-pt FFT: CUFFT 2.2 (256 threads, 33% occ, 45 Gflop/s) → CUFFT 2.3 (64 threads, 17% occ, 93 Gflop/s). **4× smaller, 2× lower occupancy, 2× higher perf.**
  - "Maximizing occupancy, you may lose performance."
- Two named fallacies: (1) "multithreading is the only way to hide latency on GPU" — **No, ILP is another way.** (2) "shared memory is as fast as registers" — **No.**

## Part I — Hide ARITHMETIC latency using fewer threads (slides 6–26)
- **Latency ≈ 20 cycles for arithmetic, 400+ cycles for memory.** Can't start a *dependent* op until it completes; can hide by overlapping *independent* ops.
- Latency ≠ throughput. Throughput = ops completed/cycle (arithmetic ~480 ops/cycle at 1.3 Tflop/s; memory ~32 ops/cycle at 177 GB/s).
- **Little's law: Needed parallelism = Latency × Throughput.**
- Arithmetic parallelism needed per SM: G80–GT200 ≈192, GF100 ≈576, GF104 ≈864 ops/SM. "Not enough operations in flight = idle cycles."
- **Two ways to supply that parallelism:**
  - **TLP (thread-level):** many threads, one independent op each.
  - **ILP (instruction-level):** independent instructions *within a single thread*.
  - "You can use BOTH ILP and TLP on GPU."
- On G80: get ~100% peak at 25% occupancy with no ILP; **or at 8% occupancy if 3 independent ops per thread.** On GF104 you *must* use ILP to exceed 66% of peak (48 cores/SM, 2 warp schedulers → must dual-issue).
- **Empirical (GTX480), `a = a*b + c` loop, vary ILP by adding independent accumulators:**
  - ILP=1 (no ILP): need **576 threads/SM** for 100% utilization.
  - ILP=2 (`a=...; d=...`): need **320 threads**.
  - ILP=3: need **256 threads**.
  - ILP=4: need **192 threads**. Doesn't scale past ILP=4.
  - Summary: latency can be hidden EITHER by thread parallelism (at ILP=1) OR by instruction parallelism (at fixed 12.5% occupancy, saturating by ILP≈4).
- Refuted CUDA Best Practices claim ("need ≥192 threads on cc1.x / 384 on cc2.0 to hide arithmetic latency"): **No — doable with 64 threads/SM on G80–GT200 and 192 on GF100** via ILP.

## Part II — Hide MEMORY latency using fewer threads (slides 27–42)
- Same formula. Memory: latency <800 cycles, throughput <177 GB/s → **need ~100 KB in flight** to hide memory latency (less if compute-bound).
- **Three ways to keep 100 KB in flight:** multithreading; **instruction parallelism = more fetches per thread**; bit-level parallelism = wider (64/128-bit) fetches.
- Do more work per thread → need fewer threads: fetch 4 B/thread → need 25,000 threads; fetch 100 B/thread → need 1,000 threads.
- **KEY MECHANISM (slide 32):** issue multiple independent loads BEFORE consuming any. `float a0=src[i]; float a1=src[i+stride]; dst[i]=a0; dst[i+stride]=a1;` — the second *load* does **not** stall; **"threads don't stall on memory access — only on data dependency."** The stall happens at first *use* of the loaded value.
- **Empirical (GTX480 copy kernel), fraction of 177.4 GB/s peak vs occupancy, varying floats/thread:**
  - 1 float/thread: needs high occupancy (85% of peak only at 100% occupancy).
  - 2 floats/thread: "can get away with lower occupancy."
  - 4 floats/thread: **25% occupancy is sufficient** for ~85% of peak.
  - 8 float4/thread: **87% of pin bandwidth at only 8% occupancy.**
  - 14 float4/thread: **84% of peak at 4% occupancy.**
  - Two ways to hide memory latency (slide 40): raise occupancy (at 4 B/thread) OR raise bytes/thread (at 4% occupancy) — both reach ~84% of peak.
- Refuted: "Low occupancy always interferes with hiding memory latency" — **No, 84% of peak at 4% occupancy** (above the 71% cudaMemcpy achieves). Refuted "more warps needed when compute:memory ratio low" — **No, 87% of memory peak with only 4 warps/SM.**
- Note (slide 34): **local arrays are allocated in registers if possible** — so per-thread arrays of fetched values become the register-resident staging that enables the many-in-flight loads.

## Part III — Run FASTER by using fewer threads (slides 43–50)
- 32768 registers/SM: fewer threads → **more registers per thread** (GF100: 20 regs at 100% occ vs 63 at 33% occ = 3×; GT200: 16 at 100% vs ~128 at 12.5% = 8×).
- **"Only registers are fast enough to get the peak."** `a*b+c` = 2 flops, 12 B in / 4 B out → needs **8.1 TB/s** to sustain 1.3 Tflop/s. Registers can; shared memory can't (shared ≈1.3 TB/s = 6× slower than registers on Fermi, ≥3× before Fermi).
- Bandwidth hierarchy: global 177 GB/s → shared 1.3 TB/s (7.6×) → registers ≥8 TB/s (6×).
- Refuted CUDA guide ("shared memory as fast as a register with no bank conflicts") — **No, shared memory BW ≥6× lower than register BW on Fermi.**
- **"Running fast may require LOW occupancy":** must use registers to run near peak → the larger the bandwidth gap, the more data must come from registers → needs many registers → **low occupancy. Accomplished by computing multiple outputs per thread** (more data local to thread in registers, fewer shared-memory accesses, more parallel work per thread).
- Tesla→Fermi regression: shared:arith gap widened (G80 16 banks:8 procs = 2:1; GF100 32:32 = 1:1; GF104 32:48 = 2:3) yet Fermi *restricted* max registers/thread (~128 on GT200 → ~64 on Fermi).

## Part IV/V — Case studies (slides 51–75)
- Matrix multiply (SDK 3.1, GTX480): baseline 137 Gflop/s → larger matrices 240 → BLOCK_SIZE 32 + `#pragma unroll` 242 → further register-blocking (compute multiple outputs/thread) pushes higher. Removing `-maxrregcount 32` matters later. FFT case study similar.

## Application to the grass march (my read)
- The march loop is a **single-thread sequential dependent chain** (step → sample field → decide → step). That is exactly the **ILP=1 / dependent-chain** regime Volkov shows leaves the SM idle waiting on latency.
- **Concrete, confident restructuring:** issue several *independent* baked-field fetches BEFORE consuming any — prefetch the next N march samples' texture loads, then consume. This is slide 32 verbatim ("threads don't stall on memory access, only on data dependency"). Should hide texture latency without needing more threads/occupancy.
- **Confident:** unrolling the inner loop + keeping the working set in registers (local arrays → registers) supplies ILP.
- **Needs measurement:** whether the march is memory-latency-bound (then prefetch wins big) vs ALU/dependent-arithmetic-bound (then need independent arithmetic accumulators, ILP≈4 saturates). Volkov's whole point: don't assume occupancy is the lever — measure whether latency is already hidden by ILP.
