# A Deep Dive into Nanite Virtualized Geometry — Karis, Stubbe, Wihlidal (SIGGRAPH 2021 Advances)

Source PDF: karis-nanite-siggraph-2021.pdf. Original: https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf

THE primary-source for Nanite's SW rasterizer + atomic-depth packing. Focused extract on the levers relevant to our election under dense overlap.

## Visibility buffer format + 64-bit atomic depth packing (THE crux)
Nanite has NO ROP or depth-test hardware when it forgoes HW rasterization, but still needs a Z-buffer. It does NOT lock tiles (many triangles may be in flight to one tile/pixel; locking is unacceptable). Instead:

**Use 64-bit atomics — a global-image `InterlockedMax` (atomicMax) to the visibility buffer.** The 64-bit integer packs:

| Bits | 30 | 27 | 7 |
|------|-----|-----|----|
| Field | **Depth** (high bits) | Visible cluster index | Triangle index |

- Depth in the HIGH bits gives the depth test for free (max of packed int = nearest/farthest depending on convention). Payload (cluster idx + tri idx) in the LOW bits.
- **"The payload needs to be small enough to pack in 34 bits or less. Without that we wouldn't be able to do fast software rasterization."** (34 bits payload + 30 bits depth = 64.)
- This is what makes the vis buffer's power real: the whole write is `pack(depth, payload)` then one `InterlockedMax`.

### Relevance to us (WebGPU has NO atomic<u64>)
Nanite's whole scheme DEPENDS on a single 64-bit atomicMax = depth-test + payload-write in one uncontended op. WebGPU lacks atomic<u64>, so a WebGPU port must either (a) pack depth+id into 32 bits (fewer depth bits → z-fighting/leak risk, per Scthe nanite-webgpu issue #1), or (b) use a separate depth atomic + redundant triangle re-intersection to resolve the payload (the Tellusim path). Nanite uses 30 depth bits; a 32-bit WebGPU pack must steal from that. Our election's depth precision + packing is the thing to check against Tsl.ts / NaniteRaster.

## Micropoly software rasterizer (the SW path)
- 128-triangle clusters → threadgroup size 128. Structurally like a mesh shader; shares vertex work via groupshared with no post-transform cache.
- Phase 1: 1 thread per vertex — transform position, store in groupshared (loop max 2 for up to 256 verts/cluster).
- Phase 2: 1 thread per triangle — fetch indices, fetch transformed positions from groupshared, compute edge equations + depth gradient, compute screen bounding rect, iterate pixels in rect; if inside all 3 edges, `WritePixel` = pack depth with payload and atomicMax to the screen.
- "All the fancy hierarchical tiling and stamps thrown out. A super basic half-space rasterizer, instruction-level micro-optimized." Inner loop is barely anything: a couple ALU + the atomic max. "Very few iterations expected so we shouldn't add fixed overhead trying to reduce it."

Inner loop (incremental edge functions, integer add per pixel):
```
for( uint y = MinPixel.y; y < MaxPixel.y; y++ ) {
  float CX0 = CY0; CX1 = CY1; CX2 = CY2; float ZX = ZY;
  for( uint x = MinPixel.x; x < MaxPixel.x; x++ ) {
    if( min3( CX0, CX1, CX2 ) >= 0 )
      WritePixel( PixelValue, uint2(x,y), ZX );
    CX0 -= Edge01.y; CX1 -= Edge12.y; CX2 -= Edge20.y; ZX += GradZ.x;
  }
  CY0 += Edge01.x; CY1 += Edge12.x; CY2 += Edge20.x; ZY += GradZ.y;
}
```

## Scanline SW rasterizer (for medium triangles)
Clusters with edges **< 32 pixels long are SW-rasterized.** For the bigger end, per-pixel edge tests waste work (best case half pixels covered, worst none). Solve for the X interval that passes instead of testing every pixel:
```
float3 InvEdge012 = Edge012 == 0 ? 1e8 : rcp(Edge012);
...
float3 CrossX = float3(CY0,CY1,CY2) * InvEdge012;
float x0 = ceil( max3(MinX) ); float x1 = min3(MaxX);
for( float x = x0; x <= x1; x++ ) { WritePixel(...); ZX += GradZ.x; }
```
Not exact fixed-point anymore (has a divide) but no issues found. **Scanline chosen if the X loop is >4 pixels for any triangle in the wave.** (Author later notes: distributing work for larger tris tested and NOT a win — HW rasterizer already wins by then; no middle ground.)

## SW vs HW selection
- **Choose SW or HW per CLUSTER based on which is faster. Vast majority are SW-rasterized in the demos.**
- Big triangles / clipping cases → HW rasterizer (it's good at that).
- **HW rasterizer also uses 64-bit atomic writes to the UAV** — it binds NO color/depth targets, atomic-writes exactly like SW raster. This lets SW and HW passes async-overlap (couldn't if HW used depth-test hardware + separate render target).
- **Cracks between SW and HW clusters:** avoided because DirectX has a strict rasterization-rules spec; following it exactly matches HW → no pixel cracks.
- Reported result: **SW rasterization ~3× faster than HW on average** (vs their fastest primitive-shader impl); even more for pure micropoly.

## Why HW rasterizers lose on micropoly
- Optimized for large triangles → run wide over PIXELS. Nanite has many triangles with few pixels each → wants to run wide over TRIANGLES.
- Modern GPUs set up 4 tris/clock max; outputting SV_PrimitiveID (needed for vis buffer) makes it worse.
- HW does binning, tile serialization for depth/ROP, 2x2 quad output, clipping, general VS+PS scheduling — all wasteful for tiny tris.

## Rasterizer overdraw (dense-overlap cost — DIRECTLY relevant to our grass/foliage)
- No per-triangle culling; no HW HiZ culling pixels; SW HZB is from the PREVIOUS frame, culls CLUSTERS not pixels (resolution based on cluster screen size).
- **Excessive overdraw from: large clusters, overlapping clusters, aggregates, fast motion.** Overdraw expense by triangle size:
  - Small tris → vertex-transform + triangle-setup bound
  - Medium tris → pixel coverage-test bound
  - Large tris → **atomic bound**
- "Riddled with holes... describes most aggregate geometry cases like leaves and grass. **Overdraw is one of many reasons Nanite doesn't perform as well with those.**" (Confirms our grass/foliage pain is the known Nanite weak spot.)
- No per-triangle occlusion culling because of the two-pass occlusion (any cluster with an occlusion-culled triangle would need re-rasterizing, nullifying savings); also a 1-thread-per-triangle divergence issue.
- **Reliance on previous-frame depth for occlusion culling is one of Nanite's biggest deficiencies.**

## Two-pass culling dataflow
Main pass: Instance Culling (prev-frame HZB + transforms) → Persistent hierarchical/cluster culling (LOD + visibility) → SW/HW rasterizer → Build HZB (current frame). Post pass: re-test occluded instances/nodes/clusters against current-frame HZB + current transforms → rasterize → Build HZB for next frame → material passes. Frustum + LOD culling only done in the first pass (occlusion-independent).

## Deferred material evaluation
- VisBuffer decode: VisibleCluster → InstanceID,ClusterID; ClusterID+TriangleID → MaterialSlotID; InstanceID+MaterialSlotID → MaterialID.
- **Material culling exploits depth-test hardware: Material ID → depth value (Material Depth buffer).** Full-screen quad per material, depth test = EQUAL → only matching-ID pixels drawn. CS outputs standard depth + material depth + HTILE (HiZ accel). Idea from Dawn engine.
- Coarse tile classification: instead of full-screen quad, an 8×4 grid of tiles per material (wave-ops path) culled by a **32-bit mask** built when material depth was; **portable no-wave-intrinsic path uses a 64×64 grid + 64-bit mask** (can alias bins, defeated occasionally, still culls by depth). Tiles killed in vertex shader by snapping X to NaN → no PS waves spin up. Rect primitives used where available to avoid diagonal overshade (PC APIs don't support consistently; consoles do).
- UV derivatives: pixel quads span triangles (good — avoids quad overdraw) but also span depth discontinuities/UV seams/objects (bad → huge finite-diff mips). Fix = **analytic derivatives** propagated through material node graph via chain rule, fall back to finite diff, sample with SampleGrad. **<2% overhead** (only affects tex-sampling ops; virtual texturing already uses SampleGrad).

## Visibility buffer imposters (far LOD / tiny instances)
- 12×12 view directions in atlas (octahedral mapped, dithered quantization); 12×12 pixels per direction; orthogonal projection fit to mesh AABB.
- **Stores 8:8 Depth + TriangleID from the root cluster (40.5KB per mesh, always resident).** Injected directly into the screen vis buffer → supports material remapping, non-uniform scale. Ray-march to adjust parallax between directions (few steps due to small parallax). Drawn directly from instance culling pass. Quality-loss pop noticeable when many copies of same mesh adjacent.

## Pipeline / performance numbers (reference)
- Main pass: instances pre-cull 896,322 → post-cull 3,668; cluster candidates 1,536,794; visible clusters SW 184,828 / HW 6,686. Total rasterized: 199,420 clusters / **25,041,711 triangles** / 19,851,262 vertices. (Naive UE4 path would rasterize >1 billion; Nanite holds ~25M consistently.)
- Performance @ ~2496×1404 upsampled to 4K (TAAU): **~2.5ms to draw entire VisBuffer** (all culling+raster, near-zero CPU), **~2ms deferred material pass** (VisBuffer→GBuffer, 1 draw per material).
  - Nanite::CullRasterize breakdown: Clear VisBuffer 66us; Main InstanceCull 108us; Main ClusterCull 406us; **Main Rasterize 1148us**; BuildHZB 99us; Post InstanceCull 125us; Post ClusterCull 102us; Post Rasterize 183us.
  - BasePass: DepthExport 217us; **Emit GBuffer 2084us.**

## Shadows (relevant to our shadow arc)
- Ray tracing rejected: DXR not flexible enough for LOD logic / triangle format / no partial BVH updates; more shadow rays than primary (>1 light/pixel). Want a raster solution to leverage existing work. HW triangle formats + BLAS are 3-7× the size of Nanite's memory format.
- **Virtual Shadow Maps: 16k×16k shadow maps everywhere** (spot 1× projection, point 6× cube, directional Nx clipmaps). Pick mip where 1 texel = 1 pixel; only render shadow pixels that are visible; Nanite culled+LODed to required detail. Don't even allocate memory for unsampled shadow space.

## Key takeaways for OUR election under dense overlap
1. Depth+ID packing: Nanite = 30 depth / 27 cluster / 7 tri in a 64-bit atomicMax. WebGPU has no atomic<u64> → we pay either depth precision (z-fight risk) or a redundant intersection. Verify what OUR atomicMax election packs.
2. Under dense overlap (grass/foliage) the SW raster is **atomic-bound for large tris, coverage-bound for medium, vertex/setup-bound for small** — and this is exactly where Nanite is admittedly weakest ("leaves and grass"). Overdraw from overlapping clusters/aggregates is the cost, and there's no per-triangle/per-pixel occlusion cull to save it — only cluster-granularity HZB from the previous frame.
3. Culling levers to cut rasterizer pressure: cluster-level HZB cull (prev frame), micropoly rejection is implicit (few pixels), SW/HW split at 32px edges / >4px scanline threshold. Per-cluster SW-vs-HW choice is the main routing lever (cf. Mali MinPixelsPerEdgeHW).
