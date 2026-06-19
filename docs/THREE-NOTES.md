# three.js 0.184.0 — verified API notes (append as discovered)

> RULE: before using an unfamiliar three/TSL API, verify it against `node_modules/three`
> (impl) and `node_modules/@types/three` (what tsc accepts). Record findings here.

## Package layout
- `three` ships **no TS types**; we use `@types/three@0.184.1`.
- Imports: `three/webgpu` (renderer + node materials + storage attrs + `TSL` namespace),
  `three/tsl` (flat TSL function re-exports), `three/addons/*` → `examples/jsm/*`.
- Core classes (Vector3 etc.): import from `three` — but note `three/webgpu` re-exports core too.
  **Convention: import core math/geometry from `three`, renderer/nodes from `three/webgpu`, TSL fns
  from `three/tsl`.** (three.webgpu.js includes core; bundler dedupes — fine with Vite.)

## Verified exports (0.184.0)
- `three/webgpu`: `WebGPURenderer`, `PostProcessing`, `PMREMGenerator`, `QuadMesh`,
  `NodeMaterial`, `MeshStandardNodeMaterial`, `MeshPhysicalNodeMaterial`, `SpriteNodeMaterial`,
  `StorageBufferAttribute`, `StorageInstancedBufferAttribute`, `IndirectStorageBufferAttribute`,
  `StorageTexture`, `Storage3DTexture`, `RenderTarget3D`, `TimestampQuery`.
- `three/tsl` (subset we checked): `Fn, If, Loop, Break, Continue, Return, Switch`,
  `uniform, uniformArray, storage, storageTexture, textureStore, texture3D, instancedArray,
  attributeArray, instanceIndex, vertexIndex, drawIndex, time, deltaTime, velocity, hash, range,
  atomicAdd, atomicStore, workgroupBarrier, wgslFn, mrt, pass, varying, positionLocal,
  positionWorld, normalWorld, uv, vec2/3/4, ivec/uvec*, reflector`, tone-mapping fns
  (`agxToneMapping`, `acesFilmicToneMapping`), shadow filters (`PCFShadowFilter`,
  `PCFSoftShadowFilter`, `VSMShadowFilter`, …).
- Addons (`three/addons/...`): `csm/CSMShadowNode.js` (WebGPU CSM), `tsl/display/`:
  `GTAONode, TRAANode, BloomNode, DepthOfFieldNode, SSGINode, SSRNode, SSAAPassNode, FXAANode,
  SMAANode, TAAUNode, DenoiseNode, GodraysNode, Lut3DNode, ChromaticAberrationNode…`

## Compute idioms (verified from types)
- Buffers: `const buf = instancedArray(count, 'vec4')` → `StorageBufferNode`;
  `buf.element(i)` read/write inside `Fn`; also `attributeArray` for non-instanced.
- Kernel: `const k = Fn(() => { ... })().compute(count, [64])` (NodeElements `.compute(count,
  workgroupSize?)`); run with `await renderer.computeAsync(k)` or sync queue `renderer.compute(k)`.
- **Indirect dispatch supported**: `computeAsync(node, IndirectStorageBufferAttribute)`.
- Storage textures: `new StorageTexture(w,h)`; write in kernel via
  `textureStore(tex, uvCoordIntNode, vec4Node)`; sample elsewhere via `texture(tex, uv)`.
  `Storage3DTexture(w,h,d)` exists for 3D (clouds/froxels/probes).
- `ComputeNode.onInit(({renderer}) => …)`, `.setName()` for GPU timestamp labels.

## Open questions (verify when reached)
- Reversed-Z / depth format default in WebGPURenderer 0.184 (for 4 km range) — check
  `renderer.depthBuffer`/camera near-far handling + logarithmicDepthBuffer option.
- `renderer.compute()` sync variant exists? (we saw `computeAsync`; check `compute`.)
- PMREMGenerator on WebGPU: `fromScene` availability/perf.
- CSMShadowNode usage pattern (examples/jsm/csm/CSMShadowNode.js) + custom shadow filter hook.
- TRAANode vs manual TAA; whether TRAA works with custom post chain & MRT.
- Readback: `renderer.getArrayBufferAsync(StorageBufferAttribute)` — confirm name.
- `hash(instanceIndex)` TSL — distribution quality; fine for jitter.

## Gotchas (append-only)
- `three/package.json` has no `./package.json` export — read version via fs, not require.
- @types/three Fn typing: `Fn(fn)` returns callable; calling with no args then `.compute()` —
  typed via `FnNode`/NodeElements; if tsc complains about `Fn(() => {...})()` use explicit
  zero-arg tuple generic or `Fn<[]>`.

## Phase 7 (perf) findings — three 0.184 WebGPU internals

- **Timestamp pools**: every render context / `renderer.compute()` CALL
  allocates a query pair keyed `r:<frameCalls>:<ctxId>:f<frame>` /
  `c:...`; `resolveTimestampsAsync` computes PER-UID durations into
  `backend.timestampQueryPool[type].timestamps` (a Map that three never
  clears — prune it or it grows forever) and only reports the sum. The
  2048-query pool resets its write index ONLY on resolve — resolve every
  frame or attribution dies after ~10 frames. Compute arrays get ONE uid
  for the whole array (the DataMap keys on the array instance).
- **Per-pass GPU timestamps on Apple are encoder wall spans** including
  waits on prior passes. Sums match wall only when serialized; individual
  values inflate for dependency-stalled passes (bloom bright/h0 showed
  4-6.6 ms each; ablating the whole chain moved wall fps ~0). Rank with
  them, verify with fps + ablation.
- **`@builtin(position)` is NOT invariant by default**: a depth-prepass
  (depthFunc EQUAL) needs `@invariant` or Metal fuses position math
  differently across pipelines (last-ulp depth mismatch = shaded pass
  drops out). three has no API; patch the WGSL builder prototype obtained
  from `backend.createNodeBuilder(...)` (the class is not exported and
  `three/src/...` imports load a SECOND module instance — patching that
  does nothing).
- **BundleGroup (static) is not production-ready here**: it records
  whatever pipelines exist at first render (async shader compiles ⇒
  objects silently missing forever), children encode in TRAVERSAL order
  (renderOrder ignored inside), and per-cascade shadow cameras lost the
  caster layer filtering (every cascade rendered the full veg = GPU 2×).
- **CSMShadowNode cascade caching**: `lwLight.shadow.autoUpdate=false` +
  scheduled `needsUpdate=true` works (ShadowNode.updateBefore contract),
  but the light pose must freeze WITH the map — CSM updateBefore refits
  texel-snapped centers per frame; override it (CsmCached.ts mirrors the
  loop; extents are rotation-invariant, set in updateFrustums only).
- **ShadowMap RTs all share texture.name 'ShadowMap'** and RenderTarget
  has NO id field — distinguish cascades by RT object identity (WeakMap).
- **VelocityNode is blind to shader displacement**: it projects raw
  `positionLocal` through model matrices (VelocityNode.js setup), so the
  velocity MRT is GARBAGE for anything positioned by a custom positionNode
  (CDLOD morph, instanced veg) — reads |v|~0.5-1 NDC with a static camera
  and world. Any consumer (TRAA) silently rejects history there. Either
  supply per-material velocity or feed analytic camera reprojection.
- **TRAANode samples its velocityNode exactly once** —
  `velocityNode.load(closestPositionTexel)` in the resolve — so a
  duck-typed `{ load: (texel) => vec4 }` is a legitimate seam for custom
  velocity (constructor arg only stored; the internal `_velocityNode`
  jitter handshake uses the global `velocity` node independently).
- **getViewPosition/getScreenPosition flip v internally** (uv is top-left
  origin, NDC y-up — PostProcessingUtils.js): a hand-rolled forward
  projection paired with getViewPosition MUST flip y back
  (`uv.y.oneMinus()`) or reprojection comes out vertically MIRRORED
  (symptom: zero-error stripe on the mirror axis, ?skyveldbg).
- **PassTextureNode.size() on an MRT attachment returned 0** at least for
  the velocity attachment (NaN uvs downstream) — use `screenSize` when the
  pass renders at drawing-buffer resolution.
- **TRAA + RenderPipeline jitter handshake**: setup wires
  onBeforeRenderPipeline→setViewOffset / onAfterRenderPipeline→
  clearViewOffset, so the scene camera is UNJITTERED between frames —
  uniforms copied outside the pipeline render never carry jitter.
- **Camera matrixWorld freshness**: mutating camera pose outside render
  leaves matrixWorld stale until the renderer's updateMatrixWorld; copies
  made in update callbacks must force `camera.updateMatrixWorld()` (also
  refreshes matrixWorldInverse) or read one-frame-old matrices.
- **GTAONode horizon math degenerates at distance/grazing** (stock 0.184
  carries both; fixed in our port, src/render/Gtao.ts): (1) once the
  world-space radius projects below one depth texel, samples land on the
  center's OWN texel, pass the |Δz| thickness test with quantization-
  dominated directions (normalize(≈0)) and drive cosHorizons → 1 = "fully
  occluded" → AO crushes to 0 on far grazing surfaces (flat fields near
  the horizon, grazing water). Reject same-texel samples. (2) f32:
  dot(viewDir, normalize(δ)) can read 1+ε → sqrt(1−cos²) = NaN; clamp the
  horizon cosines.
- **Joint-bilateral upsamplers must handle weight collapse explicitly**:
  with w = exp2(−k·|Δz|) taps, grazing slopes make EVERY tap reject (a
  half-res texel near the horizon spans tens of meters of view depth) and
  acc/ε fabricates 0 — rendered as a black horizon band after the AO
  multiply. Gate on wsum: full bilateral above a small support threshold
  (bit-exact on healthy pixels), plain tap-average fallback below it. A
  global additive weight floor is NOT equivalent — it perturbs the blend
  on every partially-weighted pixel (printed a ~1% wash on a hero trunk).

## queue.submit() granularity + GPU inter-kernel concurrency (CRITICAL — 2026-06-19)

**The single most overlooked perf lever in this project** (it never came up across ~30M tokens of
optimization). WebGPU has no async-compute queue (exactly one `GPUQueue` per device), so it was
assumed nothing could run concurrently. WRONG: GPUs + Dawn (Chromium's WebGPU impl) overlap
independent dispatches **automatically — but ONLY when they share one command buffer / submit.**

**How three.js r184 maps calls → submits (verify in `WebGPUBackend.js`):**
- Each `renderer.compute(node)` and each `renderer.render(scene)` opens its OWN command encoder and
  fires its OWN `queue.submit()`. They are NOT batched with each other.
- `clear()`, `copyTextureToTexture()`, `copyFramebufferToTexture()` each fire their own extra submit.
- `beginRender` opens one encoder for the whole render context; `finishRender` submits it — so all
  render passes inside ONE `render()` ride a single command buffer. But `compute()` then `render()`
  = 2 submits; N separate `compute()` calls = N submits.
- **THE BATCHING LEVER:** `compute()` accepts an ARRAY. `renderer.compute([a,b,c])` →
  `Array.isArray(computeNodes) ? computeNodes : [computeNodes]` → opens ONE encoder via
  `beginCompute`, dispatches every node, submits ONCE in `finishCompute`. So `compute([a,b,c])` = 1
  submit; three separate `compute()` calls = 3 submits. Our `dispatchBatch` helper (`src/.../Tsl.ts:241-243`)
  uses this array form and is the only place we currently batch.

**Why it matters (the real cost — NOT just CPU submit overhead):**
Within ONE submit, Dawn can OVERLAP two dispatches iff (a) they do not write the same resource
(read-only sharing is fine) and (b) neither's output is the other's input. If one writes what the
other reads → they serialize (Dawn inserts the UAV barrier). Across SEPARATE submits the GPU drains
between every dispatch → ALL inter-kernel concurrency + latency hiding is forfeit. This is the only
"async-ish" parallelism WebGPU offers, and maximally-isolated submits give it zero headroom.

**Rules:**
- Batch INDEPENDENT dispatches into one `compute([...])` → Dawn overlaps them (the real GPU win).
- Batch DEPENDENT chains into one `compute([...])` too → still skips the per-submit drain; in-pass
  UAV barriers + preserved array order keep it correct. Net: far fewer submits, no race.
- NEVER co-batch a write+read / write+write on the same resource OUT OF dependency order. Same-order
  dependent chains ARE safe (dispatchBatch preserves order + auto-barriers).
- Measured state (2026-06-19 audit): frame fired ~45 submits static / ~92 moving, nearly all isolated
  (only `[kClearHier,kSeedRoots]` + the HZB mip chain batched). The cull BFS (~36 submits) and shadow
  BFS (~36) dominate and are strict producer-consumer chains → each collapses to ~1 submit, zero race.
