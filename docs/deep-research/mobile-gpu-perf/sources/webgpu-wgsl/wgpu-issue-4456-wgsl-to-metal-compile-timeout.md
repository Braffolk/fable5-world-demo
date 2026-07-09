# wgpu #4456 — WGSL passes validation but MSL (Metal) compile times out / hangs

Source: https://github.com/gfx-rs/wgpu/issues/4456
Reporter: @superdump, Dec 26 2021. Labels: naga back-end, Metal, bug. Milestone: WebGPU Spec V1.

## The problem
A WGSL shader **validates fine but the Metal shader compiler hangs indefinitely** during
`new_render_pipeline_state` on macOS. Triggered after refactoring the shader to pass all
bindings as function arguments (making binding/function definition order irrelevant).

## Symptoms
- Repeated `Compiler failed with XPC_ERROR_CONNECTION_INTERRUPTED` messages.
- Retries at ~50–60 s intervals.
- Eventually device validation fails: *"Internal error in VERTEX | FRAGMENT |
  VERTEX_FRAGMENT shader: new_render_pipeline_state: 'Compiler encountered an internal error'"*.
- App panics: `wgpu error: Validation Error ... Compiler encountered an internal error`.

## Environment
macOS 12.1, Apple M1 Max (Metal), Xcode 13.2.1, naga 0.8.0, wgpu 0.12.0.

## Relevance to us — mobile/Apple GPU + boot cost
Confirms that on Apple/Metal, **WGSL→MSL translation is a real, sometimes pathological cost
happening at pipeline-creation time** (the Apple driver's shader compiler, invoked via XPC).
Combined with three.js #32735 (duplicate pipelines) and the Chrome-130 note that Tint's new
IR is "up to 10× faster translating Unity's WGSL to MSL", this is a strong candidate for our
boot-time hangs / frozen tabs: many pipelines × slow per-pipeline MSL compile. Levers: reduce
distinct pipeline count (dedupe compute nodes), keep shaders simpler on mobile (Aaltonen:
"avoid complex shaders"), and cache/pre-warm pipelines. Newer Chrome/Tint reduces per-compile cost.
