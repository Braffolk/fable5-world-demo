# Low-Level Shader Optimization for Next-Gen and DX11 — Emil Persson (reference notes)
Source: https://www.slideshare.net/slideshow/lowlevel-shader-optimization-for-nextgen-and-dx11-by-emil-persson/35658006
Fetched 2026-07-05. ⚠️ PARTIAL — WebFetch would not reproduce the deck verbatim (copyright); key points only. For the full slide-by-slide deck + assembly listings, open the URL. (AMD/GCN-oriented, but the principles are general.)

---

## Optimization principles
- "You get what you write" — the compiler cannot change operation semantics.
- Write code in MAD (multiply-add) form.
- Separate scalar and vector work; look inside function internals.

## Hardware mapping
- `rcp() rsqrt() sqrt() exp2() log2()` + trig map to HW.
- Inverse trig is expensive.

## GCN architecture specifics
- Scalar/vector instruction split with independent execution.
- Full-rate: float add/sub/mul/mad/fma, integer ops.
- ¼-rate: transcendentals, doubles, 32-bit multiply.
- Integer division is extremely expensive (~40–48 cycles).

## Practical optimizations
- `mul24()` → 4× over 32-bit multiply (where operands fit 24 bits).
- Prefer `Load()` over `Sample()` when you don't need filtering.
- Keep register lifetime LOW; use `nointerpolation` on constants.
- AVOID register indexing (dynamic index into an array of registers) — implement manual selection trees instead.
- Bypass ROPs with COMPUTE shaders for bandwidth-bound workloads.

## Anti-patterns
- Unnecessary `pow(color, 2.2)`, denorm-handling bugs, poor loop unrolling.
