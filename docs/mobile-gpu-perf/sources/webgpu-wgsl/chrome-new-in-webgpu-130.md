# New in WebGPU (Chrome 130)

Source: https://developer.chrome.com/blog/new-in-webgpu-130

## Shader compilation performance on Metal (RELEVANT)
Tint now has an intermediate representation (IR) between the AST and the Metal backend writer.
**"Initial tests show that the new version of Tint is up to 10 times faster when translating
Unity's WGSL shaders to MSL."** Available on Android and ChromeOS; progressive rollout to
macOS Metal devices. → Directly reduces the WGSL→MSL compile cost implicated in wgpu #4456
and our boot pipeline compiles.

## Dual-source blending
New `"dual-source-blending"` feature: combine two fragment outputs into one framebuffer
(Porter-Duff). WGSL `@blend_src` at `@location(0)`; factors `src1`, `one-minus-src1`,
`src1-alpha`, `one-minus-src1-alpha`.
```javascript
const adapter = await navigator.gpu.requestAdapter();
if (!adapter.features.has("dual-source-blending")) throw new Error("unavailable");
const device = await adapter.requestDevice({ requiredFeatures: ["dual-source-blending"] });
```
```wgsl
enable dual_source_blending;
struct FragOut {
  @location(0) @blend_src(0) color : vec4f,
  @location(0) @blend_src(1) blend : vec4f,
}
@fragment fn main() -> FragOut { /* ... */ }
```

## Other
- `GPUAdapter.requestAdapterInfo()` deprecated → use sync `adapter.info`.
- Dawn C API renames (`WGPUShaderSourceWGSL`, `WGPUSurfaceSource*`, etc.).
- `depthWriteEnabled` now `WGPUOptionalBool` (true/false/undefined).

## Relevance to us
The **Tint 10× MSL speedup** is the load-bearing item for boot/pipeline-compile cost on
Apple. Confirm our Chrome is recent enough to have it on macOS. Nothing here is a direct
profiling API (that's the timestamp-query post), but it corroborates the compile-cost thread.
