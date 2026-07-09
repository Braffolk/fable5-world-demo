# Modern Mobile Rendering @ HypeHype — Sebastian Aaltonen (REAC 2023)

Source PDF: reac2023_modern_mobile_rendering_at_hypehype.pdf (kept in this folder).
https://enginearchitecture.realtimerendering.com/downloads/reac2023_modern_mobile_rendering_at_hypehype.pdf

Extracted the sections relevant to: **mobile storage-buffer tax, shader compile cost, and
the structural reasons Nanite-class techniques struggle on phones.** (Talk covers Research,
Design of a minimal cross-platform Metal/Vulkan/WebGPU abstraction, and Implementation of
fast object lifetime tracking. The load-bearing slide for us is the mobile-HW-tax slide.)

## THE mobile hardware tax slide — "Why can't we have Nanite / MM:Dreams on a phone?" (verbatim)
Comparing Nintendo Switch vs bottom-50% mobile phones (peak flops ~200 GFLOP/s and mem
bandwidth ~20 GB/s are in the same ballpark). Nvidia GPU arch is designed for compute
(CUDA, AZDO); mobile is NOT:

- **Fast generic memory load/store** — **Mobile: 16KB uniform buffers! SSBOs are slow!**
- **Fast & big groupshared memory** — Mobile: small or emulated.
- **Fast local/global atomics and wave intrinsics** — **Mobile: wave intrinsic support <10%.**
- **Big register files and big generic caches** — **Mobile: avoid complex shaders.**
- **64-bit atomics** — **Mobile: no 64-bit integers at all!**
- Modern PC: 3D tiling layout for volume textures (big deal for SDF rendering) — absent on mobile.
- 50%+ of mobile phones: designed to run existing GLES 3.0 games efficiently, nothing more.

Prior art referenced: GPU-driven rendering (SIGGRAPH 2015), SDF ray-tracing / Claybook (GDC 2018).

Target audience/min-spec (HypeHype): Android 95% have Vulkan 1.0 + Android 9; 2GB RAM
(1.4GB usable); min-spec devices ~6–7 years old (Mali-G/Bifrost, Adreno 500, PowerVR 8000
Rogue; Apple A9/A8X). Buy the min-spec device of every GPU vendor and measure the important
gfx-API features on each in a small test app before committing.

## Design section (their abstraction)
- **Minimal platform abstraction**: thin low-level gfx wrapper across Vulkan/Metal/WebGPU;
  find common feature set; trim deprecated stuff (transform feedback, strips/fans, geometry
  shaders, HW tessellation). Single set of shaders (GLSL → SPIRV-Cross cross-compile).
- **Do things at the right frequency/granularity** (temporal coherence — ~90% of data
  unchanged frame-to-frame): build **all PSOs at app startup**; create one **bind group per
  material at level load** (changing material = a single Vulkan/Metal command); upload
  persistent data once + delta-update; batch dynamic uploads per-pass (no per-draw map/unmap);
  no per-draw state tracking.
- "Zero" extra API overhead pitfalls to avoid: fine-grained inputs/state/data copies,
  resource state tracking/shadow state, PSO + bind-group hash-table lookups, software cmd buffers.

## Implementation section
- **Object lifetime = arrays, not smart pointers/RAII** (per-object alloc scatters cache,
  refcount = 2× atomics, RAII destructor races). One big allocation per type; array index is
  the handle.
- **Pools + handles**: typed pool array, per-slot generation counter (bumped on free), freelist
  stack of unused indices; Handle = index + generation (32/64-bit POD); `pool.get<T>(handle)`
  compares generations → null on mismatch. Weak-reference semantics, no mutex/callbacks.
- **Hot/cold split inside the pool** (SoA): hot data (rendering-only) kept dense; cold aux
  data (size/format/ptr/allocator) in a parallel array, same handle/index. Avoids the
  usability-vs-perf compromise.
- **C++20 designated struct initializers** for resource descs (default values, clean syntax);
  custom span with `const&&` to force temporaries safely.

## Relevance to us — the mobile lever list (task focus)
1. **"SSBOs are slow" on mobile — we use storage buffers HEAVILY.** This is a real,
   named mobile tax. On TBDR/Mali/Adreno/Apple our nanite storage-buffer-driven pipeline is
   swimming against the hardware. Consider: uniform buffers (≤16KB!) for hot small data,
   minimize SSBO random access, avoid 64-bit atomics (none on mobile) and wave intrinsics
   (<10% support), keep shaders simple. This is the structural headwind for the APPLE/MOBILE arc.
2. **Build all PSOs at startup / dedupe** aligns with three.js #32735 + wgpu #4456: our per-level
   compute nodes likely over-create pipelines → boot compile cost.
3. Their whole thesis: min-spec mobile is designed for GLES 3.0-era workloads; Nanite-class
   work is structurally hostile — measure on real min-spec devices, don't extrapolate from desktop.
