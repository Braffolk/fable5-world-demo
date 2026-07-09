# Future Directions for Compute-for-Graphics — DICE (Johan Andersson), SlideShare

URL: https://www.slideshare.net/DICEStudio/future-directions-for-computeforgraphics

## Problems
- GPU fixed-function doesn't scale with FLOPS growth → imbalanced resource allocation across passes. Power efficiency is the limiting design factor.

## Static resource allocation & occupancy (the load-bearing part)
- **"GPUs need to know how many resources a shader uses before launch. Can only launch a warp if all resources are available."**
- **Occupancy example:** shaderA = 40 registers + 12 shared memory → **Occupancy 5** (5 warps). Wrap two shader paths in a `switch` statement → same resources but **Occupancy 2** (2 warps).
- Cause: the compiler must allocate for the **worst-case resource use across all branches**, even when the taken path uses fewer. **→ Splitting a mega-shader into separate kernels avoids paying worst-case register/LDS for every path.**

## Dynamic allocation proposal
- Advocates hardware that allocates resources dynamically inside warps rather than statically pre-launch. NVIDIA Volta cited as progress ("more dynamic resource allocation and scheduling", "less sensitive" to worst-case paths).

## Nested parallelism
- Quadtree GPU dispatch shows idle GPU periods = missed async-compute opportunity. Constraints today: baked kernel sizes (occupancy risk), threads doing useless work without early termination, no sync across recursive calls.

## Language issues
- "Writing fast and portable compute shaders is just not possible right now." SIMD width varies across HW/kernels. "Warp-synchronous programming is not spec compliant." "No legal way to 'wait' for another thread outside your group." **"Performance delta can be 10× or more."**
- Proposed: explicit-SIMD `parallel_foreach` with nested parallelism; real pointers, bindless textures, consistent CPU/GPU struct layout, low-latency bidirectional submission with user-space signaling + atomics.

## Relevance to grass march
- **Directly actionable, confident:** the "switch → occupancy 5 → 2" example is the strongest argument for **splitting the mega-shader**. If the grass kernel does raymarch AND deferred lighting AND branchy per-band variants in one shader, the compiler allocates worst-case VGPR/LDS for ALL paths, capping occupancy for every wave. **Split raymarch from deferred lighting (and collapse branches into separate specialized kernels/pipelines).** Each sub-kernel gets tighter resource bounds → higher occupancy.
- This is the occupancy-side complement to Volkov's ILP side: mega-shader worst-case allocation is a *structural* occupancy killer that no amount of ILP fixes; kernel splitting is the lever. **Confident this is worth testing; the magnitude needs measurement (compare emitted VGPR/occupancy of combined vs split kernels).**
