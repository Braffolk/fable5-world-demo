# Register Pressure in AMD CDNA2 GPUs — AMD Lab Notes (GPUOpen)

URL: https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-register-pressure-readme/

## Core concepts
- Occupancy = max wavefronts that can run concurrently on a CU. Higher occupancy typically masks memory latency (not guaranteed).
- VGPRs = non-uniform per-lane data (Vector ALU); SGPRs = uniform-across-wavefront data (Scalar ALU).
- CDNA2 physical limit: **32 wavefronts/CU** (4 EUs × 8 waves).

## VGPR → occupancy thresholds (CDNA2)
- ≤96 VGPRs → 5+ waves/SIMD.
- 97–102 VGPRs → 4 waves/SIMD.
- Higher → progressively fewer waves. (Occupancy is a step function of VGPR count.)

## Register spilling
When demand exceeds capacity, compiler spills to scratch (global-memory backed) → trades occupancy for slow access. Register allocation is NP-hard → heuristic.

## Optimization strategies (measured case study: kernel starting at 102 VGPRs, 4 waves/SIMD)
1. **Remove math-function calls:** `pow(var,2.0)` → `var*var` cut 102 → 100 VGPRs (compiler inlines device functions).
2. **Shrink variable lifetimes:** move a definition closer to first use → 100 → **96 VGPRs, occupancy 4 → 5 waves/SIMD.** (Defining early allocates the register early.)
3. **`__restrict__`** on reused pointers → SGPRs 98 → 78, no occupancy loss (aliasing assumptions let compiler reuse).
4. Others: set `launch_bounds` for accurate allocation; avoid stack-allocated arrays (default to scratch); minimize kernel size / call depth; control unroll pragmas to balance latency vs register pressure.

Net: 102 VGPRs/4 waves → 96 VGPRs/5 waves via inlining + lifetime shrinking + restrict.

## Caveat
Results reproduce only on CDNA2 + ROCm 5.4; compiler heuristics change across versions.

## Relevance to grass march
- **Concrete, transferable levers** (compiler-agnostic in spirit): (1) replace `pow`/transcendental in the march with mul chains; (2) **shrink live ranges** — declare march temporaries at point of use, not at loop top; (3) mark buffer pointers non-aliasing where the backend supports it. Each can drop VGPR count across an occupancy step boundary.
- Occupancy being a **step function** of VGPRs means small reductions can be worthless OR unlock a whole extra wave — must measure the actual VGPR count and where the step boundaries sit (WebGPU: inspect via Radeon GPU Analyzer on the emitted shader, or naga/tint output → offline compiler).
