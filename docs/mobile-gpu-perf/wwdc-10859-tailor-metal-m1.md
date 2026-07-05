# Tailor your Metal apps for Apple M1 — WWDC Tech Talk 10859 (reference notes)
Source: https://nonstrict.eu/wwdcindex/tech-talks/10859/  (Apple, 2020, 25:30, Luc Semeria, GPU software team)
Fetched 2026-07-05 (WebFetch extraction of the index page — summary, not full transcript).

Four headline optimization techniques: **tile shading, memoryless render targets, programmable blending, sparse texturing.**

---

## Metal features on M1
- Full GPU-driven pipeline: Tier-2 indirect argument buffers; nested argument buffers up to 500,000 textures / 1,000 samplers.
- Indirect command buffers (render + compute) with GPU encoding.
- **Barycentric coordinates + pre-clipped primitive IDs → enabling VISIBILITY BUFFER rendering.**  ← directly relevant to us
- MSAA arrays, layered rendering, F32 MSAA resolve.
- Raster order groups for order-independent transparency.
- Ray tracing + function pointers.
- Sparse textures with access counters (new on Mac).
- BC1-7 compressed formats; FP32 filtering; advanced clamp modes.

## GPU perf improvements
- Faster F32 ops; improved threadgroup-memory perf in compute + tile shaders.
- Better ALU utilization via improved instruction scheduling.
- Faster function calls + stack access.
- FP16 texture filtering for HDR; faster FP32/INT32 texture reads.
- Advanced compression HW for bandwidth savings.
- MULTIPLE concurrent compute workloads in parallel with geometry/fragment.

## Tile shading
TBDR lets compute-like shaders access full tile memory during fragment processing → single-pass rendering without system-memory round-trips → "significant" bandwidth reduction.

## Memoryless render targets
Attachments backed ONLY by on-chip tile memory (not system memory) — for MSAA textures, depth/stencil not needed post-render, temporary intermediate attachments. Reduces footprint + bandwidth.

## Programmable blending
Fragment shaders read pixel data directly from tile memory → merge multiple render passes into one; consistency via implicit Metal sync/barriers.

## Sparse textures
GPU-based texture streaming w/ fixed memory budgets; access counters show region usage; tile-level (not MIP) residency. Demo: <half the memory of traditional streaming at equal/superior quality.
