# SPEC — Rock/Stone Library Replacement: SDF-composed, Estonia-authentic (approved 2026-07-10)

Repo branch `estonia-asset-gen`. Implementer must embed the go-up-a-level rule +
EXCISION LAW + no-noise-tricks law verbatim in any subagent prompt.

**⛔ USER LAW (verbatim-critical): rocks stay MESHES.** No shader-based rock solution —
no runtime SDF raymarching, no per-frame deformation, no rock "sub renderer". Generation
may use a compute shader or plain JS; the runtime renders ordinary nanite meshes. This
spec complies: CPU f64 worker bake → Surface Nets meshes → standard registerMesh+DAG;
shading = the existing NaniteResolve material path only.

**User defaults locked at plan approval:** 12 MB registry ceiling; EtakErratic hero tier
now; `?rockdnorm` detail-normal deferred (ship v1 without).

## A. Diagnosis of the current rocks (anchors verified)

Both the geometry source AND the shading port are structurally at fault; fixing either
alone will not land.

1. **Geometry is star-convex by construction.** src/vegetation/RockBuilder.ts:190-218 —
   fieldR(dir) returns a single radius per unit direction; every vertex is
   dir · r · squash (:220-226). No overhang, no true concavity, no saddle can exist.
   "Fracture cuts" are radial clamps (:206-216) producing cone-sections aimed at the
   origin, not planar cleavage. Every rock is a lumpy convex potato.
2. **Tri budgets make displacement invisible.** VegLibrary.ts:541-542,625-628 —
   Boulder/Slab r1 = icosphere detail 4 (5,120 tris), r2 = 1,280; StoneL 1,280/320;
   StoneM 320/80; StoneS 80 tris. Micro amplitude 0.01-0.022 (RockBuilder.ts:47-76) is
   far below vertex spacing → smooth blobs. The old material's premise — "Geometric
   normals carry the meso detail (displaced mesh)" (render/VegMaterials.ts:112) — is
   dead at these budgets.
3. **The nanite resolve port lost all variation.** NaniteResolve.ts:187-217 rockShade is
   a defaults-only port: hardcoded tone (0.285,0.255,0.215), moss fixed at the
   0.25-default gate. Per-pool tone/moss options (pale talus vs dark mossy forest rock,
   VegLibrary.ts:535-540,615-624) are discarded — WorldRegistry.ts:1005 "matParam low
   byte = bark texture-array slice (rock ignores it)". Every rock in the nanite path is
   the same gray-brown.
4. **Zero meso normal detail.** NaniteResolve.ts:594-599 — shading normal is the
   barycentric smooth vertex normal. All "detail" is albedo value-noise painted on a
   smooth ball → plastic.
5. **Variant poverty.** 4 variants per class from one preset + yaw-only rotation
   (Scatter.ts:685; instance B = yaw/leanX/leanZ per GeometryRegistry.ts:18).

## B. Chosen architecture

**A seeded SDF-composed mesh library, meshed by Surface Nets at boot in the existing
worker pool, cached in BootCache, registered through the existing registerMesh + DAG
path, instanced at scatter sites + ETAK boulder sites.** Per-vertex baked maps
(curvature/AO/strata in the existing vdata word); shading stays runtime 3D-procedural —
**no textures, no triplanar, 0 texture VRAM**.

Justifications vs alternatives:
- SDF field composition is the only portable option giving true overhangs/concavities/
  planar cleavage as REAL geometry (law-compliant). gl-rock/noisy-hull rejected — that
  is what we have.
- **Surface Nets** over Marching Cubes/MDC: indexed, watertight, near-DAG-ready, simple
  CPU typed-array code. MC emits unindexed soup needing weld; MDC is paper-only debt for
  angular cleavage we get cheaper (hard smax plane cuts in the field produce sharp edges
  Surface Nets captures at our grid res).
- **CPU bake in workers** over WGSL compute: the mesh DAG path is already CPU/worker
  (DagWorkerTypes.ts:28-40 MeshDagReq); a GPU field-eval adds a readback + WGSL mesher
  for no boot-time win at our library size. f32-safety moot (CPU is f64).
- **No triplanar** — verified: rock shading already samples valueNoise3(wp)/fbm3(wp)
  (3D domain, NaniteResolve.ts:194-212); triplanar exists to project 2D textures we
  don't use. (Digest risk dissolved one level up: choose 3D-procedural + per-vertex
  bakes and the UV problem never exists.)
- **Slot-compatible classes** — keep VegClass.Boulder/Slab/StoneL/StoneM/StoneS (18-22,
  Scatter.ts:70-77) and idF = cls·8+variant: scatter kernels, classPolicy
  (WorldRegistry.ts:237), clsMaxDist, cull, raster all untouched.

### Library table (counts × res × tris × VRAM)

Registry vertex = 24 B (VERT_WORDS=6), index tri = 12 B, cluster rec 32 B + DAG sidecar
48 B per 128-tri cluster. DAG total ≈ 2× LOD0 tris.

| Class | Archetype mix (4 variants) | Grid res | LOD0 tris/variant | LOD0 total |
|---|---|---|---|---|
| Boulder (18) | 2× graniteErratic, 2× fieldstone | 64³ | ~14k | 56k |
| Slab (19) | 2× flatCobble(large), 2× graniteErratic(low) | 64³ | ~12k | 48k |
| StoneL (20) | 2× angularShard (talus), 2× fieldstone (streambed) | 48³ | ~7k | 28k |
| StoneM (21) | 2× flatCobble, 2× pebble | 32³ | ~2.5k | 10k |
| StoneS (22) | 4× pebble | 24³ | ~1k | 4k |
| EtakErratic (new head, ETAK-only) | 2× graniteErratic hero | 96³ | ~30k | 60k |

LOD0 ≈ 206k tris; with DAG ≈ 410k tris / ~215k verts.
**VRAM: verts 5.2 + indices 4.9 + cluster/DAG recs ~0.6 ≈ 10.7 MB; ceiling 12 MB
registry delta; textures 0 MB.** (Current rock library ≈ 60k tris ≈ 1.5 MB ⇒ net ≈
+9 MB.) If p95/VRAM gates fail: grid res −1 notch per tier (−45% tris) BEFORE any other
lever.

## C. Field composition per archetype

Fields are f(p) → signed distance-ish in object space, unit nominal radius, CPU f64.
Ops: smin/smax (polynomial), sdEllipsoid approx. Deterministic from Rng
(src/core/Seed.ts), one stream per (archetype, variant).

```
base(p) = smin over 2-4 ellipsoids Ei (centers ±0.35r, per-archetype aspect), k = 0.25r

fbmsdf(p, d, octaves, amp)            // IQ fbm-SDF — stays a valid SDF
  s = 1
  for o in 0..octaves-1:
    n = sdSphereGrid(p / s) * s        // min of 8 corner spheres, rad = 0.5*s*hash(cell)
    n = smax(n, d - 0.1*s, 0.3*s)      // clip against inflated host
    d = smin(n, d, 0.3*s)              // union: lumps AND bites
    p = rot(p, R_o); s *= 0.5
  return d

cracks(p, d)  = d + crackDepth · (1 − smoothstep(0, w, worleyF2mF1(p·freq)))   // engraved relief
strata(p, d)  = d + a · plateauWave(dot(p, axis)·freq + phase)                  // stepped ledges
cutPlane(d,p) = smax(d, dot(p, n) − off, kEdge)                                 // cleavage facet
```

| Archetype | Recipe |
|---|---|
| graniteErratic | base 3-4 ellipsoids; 2-4 cutPlane with LARGE kEdge=0.12r (weathered rounded facets); fbmsdf 3 oct amp 0.10r; cracks freq 2.5/r depth 0.03r width 0.08; bottom flattened smax(d, −(p.y+0.55r), 0.2r) (part-buried sit) |
| fieldstone | base 2-3 ellipsoids k=0.35r (very rounded); fbmsdf 2 oct amp 0.05r; no cuts/cracks |
| angularShard | intersection of 5-8 half-spaces through a ball (smax chain, kEdge=0.03r) ∩ ellipsoid; fbmsdf 2 oct 0.035r so facets aren't dead-flat |
| flatCobble | ellipsoid squash y=0.35-0.5, smin one side-lobe; strata a=0.02r tilted ~0.15 rad; fbmsdf 1 oct 0.03r |
| pebble | single ellipsoid, fbmsdf 1 oct 0.025r |

Border guarantee: field clamped ≥ +ε on the outermost grid shell → closed surface always.

### Per-vertex bakes (into existing vdata 4×u8, packed by geometryToSource,
WorldRegistry.ts:173-183; interpolated through DAG collapse per DAG_VERT_STRIDE layout)
- x = curvature, remapped [−c,+c]→[0,1] (6-tap field Laplacian) — convex edges vs hollows
- y = archetype flow coord (strata phase, or worley F1 crack proximity)
- z = AO × upness: field occlusion at p + n·{0.15, 0.4, 1.0}r × max(n.y,0) — moss/lichen
  openness (Dorsey-lineage geometric mask)
- w = cavity AO (resolve already applies as ao, NaniteResolve.ts:603)

## D. Mesher + manifold step

Naive Surface Nets (~150 LOC typed arrays): one vertex per sign-change cell at the
field-zero centroid, quad per sign-change edge → 2 tris; indexed + watertight by
construction. Vertex normal = normalized field gradient (central differences) — the TRUE
field normal, carries detail at exactly vertex scale.

Validation in the bake (throw on fail): every edge shared by exactly 2 tris
(open-address edge audit like Clusterize.ts:44-81); zero-area tris dropped; vertex count
< 65k per variant. Rare Surface-Nets pinch is tolerated downstream: Clusterize retires
3rd+ tri on an edge (:78-81), BuildDag position-welds (:28-33) — no MDC needed.

## E. Weathering — decision

v1: NONE as a separate pass (smin/smax rounding + fbmsdf clip dominate for glacial/
fluvial Estonian stone). v2 REVIVE CONDITION: erratics read "CSG-ish" at the R2 visual
gate → Goblins-style curvature-directed erosion (~8 Jacobi iterations,
d' = d + λ·clamp(κ,0,κmax) on the narrow band) before meshing — same worker job.
Surface for the user when the condition fires; don't build speculatively, don't
silently park.

## F. Material — rockShadeV2 in the resolve

Replaces rockShade (NaniteResolve.ts:187-217). Same signature + matParam input; stays
diffuse-only 3D-procedural (no textures). Verified feasible: wp/nrm/vdata/matParam all
available in the rock branch (:578-604; matParam readable from mesh word 7 as bark does
at :611).

- **Palette via matParam low byte** (the #1 shading fix): 8-entry const palette
  {tone vec3, mossAmt, lichenTint} — granite-gray, pale granite (talus), red-brown
  rapakivi, dark diabase, mossy-forest. WorldRegistry.ts:1007 sets matParam per pool
  from the new library descriptor.
- **Granite speckle**: two-scale quantized valueNoise3(wp·{24, 90}) → light quartz/
  feldspar + dark mica grains mixed into tone, amplitude gated by vdata.x (worn convex
  edges lighter). Albedo texture, not fake relief — legal.
- **Crack/strata darkening** from vdata.y; **cavity dirt** from vdata.x concave end.
- **Moss/lichen**: moss = smoothstep(vdata.z) × world moisture-ish fbm (keep existing
  :212-214 structure) × palette mossAmt; lichen = pale patches on convex, exposed
  (vdata.x high, vdata.z mid) — geometric masks only.
- Keep the streak/dust logic shape (:206-210), drive steepness with the (now detailed)
  field normal.
- NO screen-space noise, no dither, no detail-normal in v1 (user deferred ?rockdnorm).

Old-path rockMaterial (VegMaterials.ts:114-171): replaced by minimal rockMaterialV2
mirroring the same palette; port nothing else.

## G. Bake location + caching + boot budget

- **Where:** src/vegetation/RockGen.ts (new; pure typed-arrays, node-runnable like
  Clusterize/BuildDag). Executed via a new worker job kind 'rock' in DagWorkerTypes.ts
  (params+seed in, {positions,normals,vdata,indices} transferables out), fanned across
  the existing prepareWorldVeg pool (WorldRegistry.ts:741-752 pattern).
- **Cache:** new BootCache store 'rocks', keyed per the DDC model (BootCache.ts:1-40):
  FNV-1a of RockGen.ts?raw + seed + per-tier res/count params + CACHE_REV bump. Cached
  value = the meshed arrays. Downstream DAG builds already cache via 'dags'.
- **Boot estimate:** coarse-to-fine narrow band (eval ¼-res grid, refine only cells with
  |d| < cellDiag) → ~1/8 full-grid evals. 96³ hero ≈ 0.3-0.6 s; 22 variants over the
  pool ≈ 2-4 s cold, ~0 warm. State in PR.
- All f64 CPU; runtime shaders unchanged f32; no shader-f16 dependency.

## H. Wiring + ETAK instancing

**Generated-world scatter:** unchanged — classes 18-22 keep kernels, context-keyed
variants (Scatter.ts:657-698, 781-808), scale/sink logic. Only the library content
behind each idF changes (VegLibrary.ts pool construction). Variant semantics preserved:
v0/1 = pale/talus context, v2/3 = dark/mossy context.

**ETAK boulders** (record per asset-gen/src/assetgen/process/boulders.py: x,z u16
chunk-local; kind u8 0=single/1=pile; size u8 = clamp(h·40, 8, 255); variant u8; layer
id 7). No client reader exists yet — define the interface now, wire to streaming at S9:

```ts
// src/vegetation/EtakBoulders.ts (new)
ingestEtakBoulders(records, chunkOriginXZ, chunkM, heightAt): { perId: Map<idF, {a,b}[]> }
```
- size_m = size/40. kind=0 single → class by size: <0.45 m StoneM, <1.2 StoneL,
  <2.5 Boulder, ≥2.5 EtakErratic (new head beside the pools, matClass 'rock',
  classPolicy gains its class id). variant&3 → library variant; yaw = pcg hash;
  scale = size_m / nominalRadius(class); y = height sample − sink (reuse the
  scale·0.28·bed law, Scatter.ts:682-684).
- kind=1 pile → deterministic composition, not a merged mesh: 6-14 fieldstone/flatCobble
  instances (seeded by variant) on a ring+top layout of footprint ≈ max(1.5, 1.2·size_m)
  m, each sunk 25-35%, mutual overlap allowed (reads as contact). Peytavie rejected v1.
- Output merges into the same perId partition maps before registration
  (WorldRegistry.ts:863-891) → standard bindInstances path. Dev fixture:
  `?etakrocks=fixture` hand-written record array for R3 verification pre-streaming.

## I. Full excision list (EXCISION LAW: no shims, no tombstones)

| Target | Action |
|---|---|
| src/vegetation/RockBuilder.ts (entire: icosphere, ROCK_PRESETS, fieldR, buildRock) | DELETE; replaced by RockGen.ts |
| VegLibrary.ts:528-571 rockPools + :573-660 stoneClasses bodies | REPLACE with new-library pool construction (same cls/variant/idF contract) |
| VegMaterials.ts:114-171 rockMaterial | REPLACE with rockMaterialV2 |
| NaniteResolve.ts:187-217 rockShade + call :600 | REPLACE with rockShadeV2 (matParam-aware) |
| WorldRegistry.ts:1005 comment "(rock ignores it)" | update — rock now consumes matParam |
| GalleryScene.ts:45,215,244,261,298,304 buildRock exhibits incl. cliffFace preset | REPLACE with new archetype exhibits (gallery = the visual gate surface) |
| Stale preset comments (Scatter.ts:788-789, VegLibrary.ts:611-624) | reword to archetype names |

Post-excision: `grep -rn 'buildRock\|ROCK_PRESETS\|RockPreset\|rockMaterial\b' src/` →
zero hits.

## J. Staged execution + gates

Worst-pose smoke: `?scene=world&cam=-582.1,302.4,1006.1,2.5692,-0.0077`. Perf by p95
only via tools/perf/interleaved_ab.mjs (cooled). Each stage ends `npx tsc --noEmit`.

- **R0 — RockGen core.** SDF ops + archetypes + narrow-band Surface Nets + vdata bakes +
  tools/probe-rockgen.ts (node, pattern of probe-clusterize): manifold edge audit, tri
  counts within ±25% of table, determinism (same seed → byte-identical arrays).
  Gate: probe green.
- **R1 — Boot integration.** Worker job 'rock', BootCache 'rocks' store, VegLibrary
  pools swapped, EtakErratic head registered (0 instances until R3). Gate: tsc + cold
  and warm boot smoke at worst pose (no holes, counters sane, warm uses cache —
  byte-identical first frame cold vs warm).
- **R2 — Shading.** rockShadeV2 + matParam palette + rockMaterialV2; GalleryScene
  exhibits updated. Gate: gallery + worst-pose screenshots for USER VISUAL SIGN-OFF
  (async: post screenshots, proceed, revisit on feedback). If erratics read CSG-ish →
  §E revive condition fires (surface it).
- **R3 — ETAK interface.** ingestEtakBoulders + pile composer + ?etakrocks=fixture.
  Gate: fixture renders grounded singles + piles at correct sizes.
- **R4 — Excision + perf.** §I executed; VRAM delta reported; interleaved A/B old-vs-new
  at worst pose. Gate: p95 ≤ +3% (else grid res −1 notch and re-measure); registry
  delta ≤ 12 MB; grep-clean. Commit per repo law.
