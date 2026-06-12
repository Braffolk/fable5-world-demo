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
