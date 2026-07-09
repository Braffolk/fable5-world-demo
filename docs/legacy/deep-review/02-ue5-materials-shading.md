> Reconciled into 19-ue5-and-prior-art.md §Reconciliation (2026-07-02); this doc kept for detail. NOTE: §2.1's "UE5 fallback = depth-only InterlockedMax + separate payload store (OUR architecture)" was KILLED by the reconciliation — shipping UE5 has `#error UNKNOWN_ATOMIC_PLATFORM` (NaniteWritePixel.ush:33); only the experimental Voxel/ScatterBricks.usf uses our idiom.

# 02 — UE5 Nanite materials/shading/decode vs our resolve (reference playbook)

Reader 02 of the whole-pipeline deep review. Slice: UE5's vis-buffer → material pipeline
(NaniteShadeBinning.usf, NaniteDataDecode.ush, NaniteAttributeDecode.ush, NaniteVertexFactory.ush,
NaniteVertexDeformation.ush, NaniteTranslucency*, NaniteTranscode.usf, NaniteStreamOut.usf — all in
docs/perf-runs/Nanite-UE5-shaders/) mapped against our `src/nanite/NaniteResolve.ts`. Premise audit
(docs/deep-review/90-premise-audit.md) read first; all numbers from the 2026-07-02 measured table.

## 1. TL;DR

- **This stage is measured ~FREE on our side** (nandbg=flat ≈ baseline; nanshadow=0 ≈ baseline;
  leafcheap=all −0.9/−0.8/−0.5, under the 2ms apparatus floor). Shade binning is NOT our lever — say
  it plainly: UE5 bins because it has thousands of arbitrary artist materials; we have six trivial
  classes in one graph. **Do not build binning, quad-shading, or VRS scatter.**
- Only frame-ms candidate here: flip `?reskeep=0` default (drop the full-screen empty-CSM `keep`
  sample + its second per-pixel wp reconstruction) — Class I by the keep≡1 argument, expected
  0–1.5ms eye/oblique, one interleaved probe decides (§6 P-A).
- Best steal is quality, not ms: UE5 shades voxels with a **stochastic SGGX normal + TAA**
  (NaniteVertexFactory.ush:898–921). We already bank a SPREAD word per brick — the principled
  replacement for voxbead/voxjit. Class Q, ~0 perf cost on vox pixels.
- Reference playbook documented for future work: WPO distance-gating (wind), transcode/streaming
  (boot), StreamOut (GPU mesh extraction), translucency = HW-raster forward, never through the
  vis-buffer election (future water).

## 2. How UE5's vis-buffer → material pipeline works

### 2.1 The vis-buffer contract

UE5 packs a 64-bit pixel: depth in the high 32, `((VisibleClusterIndex+1)<<8)|TriIndex` in the low
32 (PackVisPixelX/UnpackVisPixel, NaniteDataDecode.ush:890–911), elected with a 64-bit
`InterlockedMax` where available (NaniteWritePixel.ush:28–31 — the fallback is depth-only
InterlockedMax + separate payload store, i.e. OUR architecture). 8 bits of tri index vs our 8-bit
id8 tiebreak — same budget, different position.

After raster, a fullscreen PS (`EmitSceneDepthPS`, NaniteExportGBuffer.usf:70–157) converts the
vis-buffer to a real HW depth buffer AND writes a per-pixel **ShadingMask**: a uint texture packing
a 14-bit *shading bin* + decal/shadow/lighting-channel/VRS bits (PackShadingMask,
NaniteDataDecode.ush:833–850). The bin comes from per-triangle material lookup: each cluster
carries material ranges — a fast path of ≤3 inline ranges decoded from one dword
(NaniteDataDecode.ush:561–588) or a slow table walk (NaniteAttributeDecode.ush:244–263) — then a
per-primitive material-slot table maps material index → {TriangleShadingBin, VoxelShadingBin,
CurveShadingBin, RasterBin, FallbackRasterBin} (NaniteAttributeDecode.ush:312–336, 401–410).

### 2.2 Shade binning: count → reserve → scatter (NaniteShadeBinning.usf)

Purpose: turn "each pixel has one of ~thousands of unique material shaders" into "one indirect CS
dispatch per material over a dense pixel list". Three phases over 8×8-pixel tiles of 2×2 quads,
Z-order-swizzled threads (ShadingBinBuildCS, NaniteShadeBinning.usf:941–954):

1. **COUNT**: each thread owns a 2×2 quad, loads 4 ShadingMask values (BinShadingQuad:716–843),
   then a wave-scalarization loop (`BinScalarization`, :336–395) repeatedly votes one bin
   (groupshared `InterlockedMax` or `WaveReadLaneFirst`), processes all lanes matching it, marks
   them done — converting divergent per-pixel bins into a serial-over-unique-bins scalar loop with
   one `InterlockedAdd` per wave per bin (FCountPixelsTask:397–418).
2. **RESERVE** (ShadingBinReserveCS:958–1044): one thread per bin allocates its range in one big
   `OutShadingBinData` blob and writes per-bin indirect dispatch args
   (`DivideAndRoundUp(BinPixelCount, COMPUTE_MATERIAL_GROUP_SIZE)`, :1038).
3. **SCATTER**: re-classify and write packed pixel coords (PackShadingPixel — 2×~14-bit coords +
   VRS shift + write mask in 32 bits, :184–202) into the bin's range. Full 8×8 tiles pack from the
   range's front, loose quads from the back (FScatterPixelsTask/AllocateElements:257–334, 420–525)
   so full-tile shading groups get perfectly coherent pixels.

Two shading modes per material, chosen by a material flag `bNoDerivativeOps` (:826–839): pixel
binning (no helper lanes) vs **quad binning** for materials needing ddx/ddy — quads carry helper
pixels and the stats pass counts them as waste (TotalHelperCount, :700–706). VRS folds 2×2 coarse
pixels of the same bin into one shaded value scattered through a write mask
(UpdateVRSActiveAndWriteMasks, :220–239). There is even a CMask export path to keep fast-clear
metadata coherent (:889–938).

The material pass itself (`GetMaterialPixelParameters`, NaniteVertexFactory.ush:1240–1251) then
runs per bin as a compute dispatch that re-reads VisBuffer64 per listed pixel.

**Why this exists**: with arbitrary material graphs, one uber-shader would compile the union of
all materials (register pressure = worst case for every pixel) and diverge per pixel. Binning buys
per-dispatch specialization + pixel-list coherence. That premise does NOT hold for us (§3).

### 2.3 Attribute decode economy

Per material-pass pixel, UE5 does the FULL decode chain (FetchNaniteMaterialPixelParameters,
NaniteVertexFactory.ush:1187–1231 → _Inner:1101–1184):

- Unpack vis pixel → `GetVisibleCluster` (8–16B packed, NaniteDataDecode.ush:431–447) →
  `GetCluster` (8×float4 SOA header loads, :652–687).
- Triangle indices from a bitstream: base index + two 5-bit deltas (DecodeTriangleIndices,
  :780–799). Positions bit-packed per cluster (GetClusterPosition/DecodePosition, :924–951).
- `GetRawAttributeData<3>` (NaniteAttributeDecode.ush:742–888) decodes **all three vertices
  simultaneously** — the header comment (:740–741) notes batching the three bitstream readers
  generates better code than three independent decodes. Octahedral normals (UnpackNormal,
  :168–176), tangent as angle-around-normal (UnpackTangentX, :567–581) or implicit from UV deltas
  (DecodeImplicitTangents, :670–722), custom low-bit float UVs (DecodeUVFloat, :148–158), color as
  delta-vs-min.
- Vertices are transformed to clip and barycentrics are computed **analytically** from the three
  clip positions with screen-space derivatives carried as duals (CalculateBarycentrics,
  NaniteVertexFactory.ush:1043–1098; Lerp over TDual). Attributes get exact ddx/ddy with no quad
  communication — that is how UE5 sidesteps the "undefined derivatives in non-uniform control
  flow" problem we solved with the analytic bark mip LOD (NaniteResolve.ts:568–578).
- The pixel's world position is NOT interpolated: it is reconstructed from SvPosition × inverse
  projection (NaniteVertexFactory.ush:872–877) — exactly our depth-reconstruct idiom
  (NaniteResolve.ts:365–371).

Our equivalent (three `fetchWorldVert` + `readVertex` per rock/bark/leaf pixel,
NaniteResolve.ts:482–508/523–545/703–716; makeCtx gust samples NaniteFetch.ts:147–255) is
architecturally the same shape and **measured ~free** (leafcheap=all delta under noise), so decode
economy is a non-lever for us today. Their batching trick is worth remembering only if a future
material class makes decode hot.

### 2.4 Voxel shading (their answer to our flat-brick look)

UE5 reconstructs the voxel hit from SvPosition: unproject to local, floor to voxel grid, brick
occupancy popcount → vertex index (ReconstructVoxel, NaniteVertexDeformation.ush:507–589; brick
decode NaniteDataDecode.ush:801–820 — same occLo/occHi popcount idiom as our VoxelBrick). Then, in
the material pass, voxel pixels get a **stochastic normal drawn from an SGGX distribution**:
per-brick anisotropy in Color.a, per-pixel-per-frame noise, `SGGX_DvisSample` of the
visible-normal distribution, TAA integrates the result (NaniteVertexFactory.ush:898–921). UV
derivatives for voxels are faked from view-space pixel size (:1157–1181, "VOXELTODO" clamp).

Map to ours: we shade voxels with the baked per-brick mean normal + bead-field bend + value jitter
(NaniteResolve.ts:782–868) and carry a SPREAD word per brick explicitly reserved "for a future
SGGX fallback" (NaniteResolve.ts:763–764; BRICK_WORDS layout GeometryRegistry.ts:91). UE5's scheme
is the principled version of the bead/jit hacks: crowns get within-brick normal variance that
converges under TRAA instead of a hand-tuned radial field. Class Q lever, §4.2.

### 2.5 Vertex deformation (WPO) architecture

Deformation is a fixed stack applied to LOCAL verts before world transform: skinning (bitstream
bone influences, DeformLocalNaniteVertex, NaniteVertexDeformation.ush:91–171), spline deform
(:150–168), then material WPO evaluated inside the material pass on the 3 fetched verts
(TransformNaniteVerts, NaniteVertexFactory.ush:339–457) — current AND previous frame for velocity.
Three details worth stealing:

1. **WPO is gated twice**: per-cluster culling flag `NANITE_CULLING_FLAG_ENABLE_WPO`
   (NaniteVertexFactory.ush:1140–1143) AND per-primitive
   `PRIMITIVE_SCENE_DATA_FLAG_EVALUATE_WORLD_POSITION_OFFSET` (:344–351) — UE5 turns WPO off by
   distance per primitive and the material pass respects it. Our wind is structurally the same
   (makeFetch windOn shared by raster+resolve so positions stay bit-identical,
   NaniteResolve.ts:222–227) and the cheap-leaf path already skips gust work beyond ~36m
   (NaniteResolve.ts:737–754); a distance gate on the RASTER side's wind ALU is the transferable
   idea (base/raster reader's territory — resolve side is free).
2. **Velocity**: fixed-function velocity is written in the depth-export pass unless WPO forces it
   into the material pass (NaniteExportGBuffer.usf:124–141; CalculateNaniteVelocity,
   NaniteVertexDeformation.ush:334–504). If we ever feed TRAA real motion vectors for foliage,
   this is the template.
3. **Occupancy trick**: the prev-frame bone loop is intentionally split into a second loop that
   REFETCHES influences — "well worth it in practice" to shorten the live-register chain
   (NaniteVertexDeformation.ush:458–461). Same disease we fixed by splitting the vox resolve pass
   when the terrain subgraph collapsed vox-pass occupancy (NaniteResolve.ts:439–445). Generic
   lesson for any hot TSL kernel: split latency chains even at refetch cost.

### 2.6 Streaming / transcode (their boot & memory model)

Nanite meshes live as compressed disk pages; the GPU transcodes on stream-in (NaniteTranscode.usf):
strip-index decompression with per-dword prefix bookkeeping (UnpackStripIndices, :200–298), vertex
attributes as zigzag deltas decoded with **wave prefix sums** (UnpackZigZagDeltas, :160–189), and a
second parent-dependent pass that copies "ref" vertices from already-resident parent pages —
deduplicating vertices across LOD levels at rest (TranscodePageParentDependent, :596–692;
BuildRefTable, :365–396). StreamOut (NaniteStreamOut.usf:29–41, 107–153) is the reverse: pick a cut
at a fixed error, count, allocate, write raw VB/IB — how UE5 builds RT BLASes from Nanite data.

Map to ours: we re-cluster/voxelize on CPU workers every boot (FarTiles worker-pool splat;
DagWorker). The transferable idea is *persist the built DAG/brick pages as compressed binary
(zigzag/delta) and GPU-transcode at boot* — boot-time lever only (frame loop untouched), effort L.
StreamOut is the template if we ever need GPU-extracted merged far meshes or physics proxies.

### 2.7 Translucency

Translucent Nanite bins are **hardware-raster only** — the args transcode states it outright
("only support HW raster (currently). SW counts are ignored", NaniteTranslucency.usf:22–34) and the
factory forward-renders per raster bin with material VS/PS (NaniteTranslucencyFactory.ush:80–106).
Lesson for future water/translucent foliage: a single-election vis buffer cannot hold multiple
fragments per pixel; UE5 does not try. Route translucents through an ordinary forward pass over
the nanite depth, not through the atomic election.

## 3. Why shade-binning is NOT our lever (measured + structural)

Measured (2026-07-02 table): `nandbg=flat` (no lighting/GI) ≈ baseline; `nanshadow=0` ≈ baseline;
`leafcheap=all` (every mesh-leaf pixel through the cheap path — the exact ceiling of any leaf-decode
optimization) = 35.2/42.4/16.3 vs 36.1/43.2/16.8, i.e. <1ms everywhere, under the ±1.5–2ms
apparatus floor (premise audit §3.7). The resolve stage — attribute decode, material mux, lighting,
shadow receive — is not where the frame lives. The frame lives in base raster + foliage coverage
(premise audit §2.4 pools: oblique base ~10.3, foliage ~21.8, post ~5.1).

Structural reason it stays free: our "materials" are six fixed classes (terrain/rock/bark/deadwood/
leaf/vox — matClass mux NaniteResolve.ts:418–422, 881–891) inside ONE fragment graph, each a small
analytic shader with 0–2 texture arrays; branches are screen-coherent (a wave is almost always all-
terrain or all-canopy). UE5's count/reserve/scatter exists to make *thousands of incoherent artist
materials* dispatchable; porting it buys us three extra compute passes + a pixel-list round-trip to
solve a problem we do not have.

We already run the degenerate 2-bin version of it: the tri/vox Discard partition
(NaniteResolve.ts:305–320) exists for the Metal 10-storage-buffer ceiling, and the one real
shading-stage regression we ever measured (37.5ms close-up cliff) was fixed by *stripping the
terrain subgraph out of the vox pass* — a specialization win, not a binning-granularity win
(NaniteResolve.ts:439–445). If a future class (water, grass) bloats the graph, the escape hatch is
another Discard-partitioned fullscreen pass per class family — not UE5's machinery.

Cost of our partition scheme, for the record: both passes read `payloadV` for every screen pixel
and the loser pass discards (NaniteResolve.ts:359–391) ⇒ ~2 fullscreen u32 reads + fragment
launches ≈ 3.34Mpx × 8B ≈ 27MB traffic, well under 0.1ms-equivalent on M1 Max — consistent with
the measured-free attribution. Not worth a pixel-list scatter.

## 4. Waste map + levers (ranked)

### 4.1 Lever: flip `?reskeep=0` default — kill the full-screen empty-CSM `keep` sample

- **Mechanism**: with our nanite shadow active, the resolve still multiplies in three's CSM factor
  `keep` per covered pixel. Three's cascade maps are empty in the black slate (all casters render
  castShadow=false), so every tap returns lit ⇒ keep≡1 ⇒ the whole full-res cascade-select+PCF
  sample chain is dead work, and it is NOT quartered by the half-res shadow path
  (NaniteResolve.ts:275–291). Referencing the CSM node also forces `receivedShadowPositionNode` — a
  SECOND per-pixel wp reconstruction (NaniteResolve.ts:330–350). The runtime corner-pixel gate
  (`keepFullU`, NaniteResolve.ts:946–954) already exists precisely for a thermal-invariant
  within-boot A/B; the default is still keep-on (`keepOn = q.get('reskeep') !== '0'`,
  NaniteResolve.ts:286–291).
- **Files**: NaniteResolve.ts (default flip), NaniteFrame.ts (explicit cascade-fit drive — the
  comment at NaniteResolve.ts:283–285 says the fit consumer must be driven explicitly).
- **Expected ms**: eye 0–1.5, oblique 0–1.5, aerial ~0, live ≤1 slot-fraction. Honest: the whole
  stage measured ~free, so this may be 0; it is the only item in the stage with a per-pixel chain
  long enough to matter at all.
- **Quality**: Class I (keep≡1 argument + empty-map invariant), gated by shotdiff-0 at the 5 poses.
- **Measurement gate**: probe P-A (§6) — within-boot `setKeepFull(0|1)` toggle, med gpuWall.
- **Effort**: S. **UE5 analog**: none — they never sample a shadow term the producer didn't fill.

### 4.2 Lever: SGGX stochastic voxel normal from the banked SPREAD word

- **Mechanism**: replace/augment the bead-field + value-jitter voxel look
  (NaniteResolve.ts:803–868) with UE5's scheme: build a tangent basis around the brick mean normal,
  draw a visible-normal sample per pixel per frame from an SGGX lobe whose roughness comes from the
  brick SPREAD word, let TRAA integrate (NaniteVertexFactory.ush:898–921). Within-brick and
  frame-to-frame variance where today there is a piecewise-constant plate.
- **Files**: NaniteResolve.ts vox branch; VoxelBrick.ts (SPREAD decode already exists);
  VoxelizeCrown.ts already bakes spread.
- **Expected ms**: ~0 (a dozen ALU on vox-pass pixels only; no new bindings — stays under the
  10-buffer ceiling since voxelBricks is already bound in the vox pass).
- **Quality**: Class Q — pixels change, claimed better. Gate: side-by-side crops + user sign-off,
  flag-gated (?voxsggx=k) with voxbead as fallback. Risk: shimmer if TRAA history rejects foliage;
  the noise must be jitter-sequence-stable.
- **Effort**: M. **UE5 analog**: this IS the UE5 solution.

### 4.3 Non-levers, stated so nobody builds them

- **Shade binning / pixel-list scatter / per-bin dispatch**: targets a measured-free stage; premise
  (arbitrary materials) absent. DEAD for perf; revisit only if material-class count grows past what
  Discard-partitioned passes handle.
- **Quad-shading / VRS coarse-pixel folding at resolve**: Class R information reduction aimed at a
  free pass. DEAD.
- **Attribute decode compression (batched 3-vert bitstream readers, octahedral normals)**: our
  decode is measured free; keep in the drawer for a future hot class only.
- **Fetch-side micro-economy (BRICK_WORDS=9 repack)**: premise audit §3.3 already parks it —
  nothing here changes that.

### 4.4 Boot/future playbook (no frame ms, documented for the master plan)

- **Transcode-style boot**: persist built DAG/brick pages (zigzag/delta + wave-prefix GPU decode,
  NaniteTranscode.usf:160–189) instead of re-building at boot; parent-dependent vertex refs
  (:596–692) are the pattern if our mip-pyramid DAG ever dedups verts across levels. Effort L,
  boot-time only.
- **StreamOut**: GPU cut-selection + VB/IB writeout (NaniteStreamOut.usf:29–41) — template for
  merged-far-mesh extraction or physics proxies.
- **Translucency**: forward HW pass over nanite depth; never through the election
  (NaniteTranslucency.usf:22–34). Template for water.
- **Velocity export**: EmitSceneDepth-style fixed-function velocity (NaniteExportGBuffer.usf:
  124–141) if TRAA foliage ghosting ever needs real motion vectors.

## 5. Refuted / rejected for this stage (do not retry)

- Shade-binning as our frame model — half-wrong per the 2026-07-02 attribution (resolve lighting +
  shadows ≈ FREE); this doc adds the structural half: the binning premise (material heterogeneity)
  does not exist here. Consistent with docs/perf-runs/2026-07-02-attribution-and-waves.md.
- Leaf-decode cheapening beyond ?resfar — ceiling measured by leafcheap=all at <1ms; closed.
- Per-pixel resolve pass consolidation (merging tri+vox passes back) — blocked by the Metal
  10-storage-buffer cliff AND the vox-pass occupancy cliff it fixed (NaniteResolve.ts:305–320,
  439–445); the "waste" of the second fullscreen launch is ~0.1ms-order. Closed.
- No contradiction found with prior docs; this doc CONFIRMS the attribution verdict and premise
  audit §2.4 (this stage's pool ≈ 0 of the oblique −16.2).

## 6. Open questions + serial GPU probes wanted

- **P-A (the one perf probe this stage requests)**: within-boot keep A/B. Boot default config,
  settle, then `window.__laasNanite.setKeepFull(1)` → 64-frame isolated med, `setKeepFull(0)` →
  64-frame isolated med, repeated ×2 in reversed order (thermal rule), at oblique and eye.
  Engagement counter: the flipped uniform is the mechanism (code path is unconditional).
  Decision: Δ ≥ 1ms med at either pose ⇒ flip `reskeep` default + explicit cascade-fit drive +
  shotdiff-0 gate; Δ < 1ms ⇒ close the lever permanently and record the number.
- **P-B (crops only, no gpuWall)**: ?voxsggx prototype crops at the 3 canonical poses + the two
  stress poses vs voxbead-v2 for user sign-off (Class Q gate). Only after P-A since it touches the
  same file.
- Open question for the raster readers (not this stage): UE5 gates WPO per cluster by distance
  (NaniteVertexFactory.ush:1140–1143) — does our raster's per-vertex wind ALU at 45–140m survive a
  same-shape distance gate bit-identically at the vox handoff? (Resolve side already does this via
  resfar.)
- Open question for the streaming/boot owner: is boot re-voxelization (FarTiles splat ~6s
  post-worker-pool) worth a transcode-style persistent cache, or is 6s acceptable? Decision is
  product, not perf.
