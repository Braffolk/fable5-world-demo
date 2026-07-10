# SPEC — Bark rework: geometric relief + CPU-baked height-field texture (approved 2026-07-10)

Deliverable owner: implementer agent, executing verbatim. Repo branch `estonia-asset-gen`.
Grounded in the procedural-bark research digest (docs/deep-research/procedural-bark) +
verified code investigation; two digest claims were refuted against the actual code
(no `parallaxOcclusion` node exists in three r184 — only the material-TBN-bound
`parallaxUV` helper, unusable in the vis-buffer resolve; and the species list is
spruce/pine/beech/birch/karst/snag, not oak/aspen/alder).

**User defaults locked at plan approval:** 512²×6 texture array (8.4 MB); POM
degradation ladder pre-approved; beech/birch stay near-smooth (botanically faithful).

## 0. Laws for the implementer (embed in every subagent prompt — they do NOT inherit CLAUDE.md)

- Go up a level (verbatim): "When a problem resists solving, or a result disappoints, do
  NOT start varying your approach to the problem — go up one level first, to the context
  the problem lives in: the surrounding system, the upstream decisions, the goal that
  made this a problem, and everything you've been treating as the fixed environment it
  sits inside. What you filed under 'given' is the prime suspect."
- NO noise/dither/stochastic tricks to fake detail. Noise as height-field *topology
  source* is fine; noise to hide flatness is banned.
- Judge perf by p95 only; interleaved A/B via tools/perf/interleaved_ab.mjs; never bench
  with anything heavy running.
- No systemic knobs. Dead code goes COMPLETELY, including comments — no shims, no
  tombstones.
- shader-f16 not guaranteed; all GPU-visible data f32/u8-safe (the bake is CPU TS).
- NEVER commit docs/METAL-PROFILING.md, docs/todo-human-only.md, asset-gen/; never
  `git add -A`.

## 1. Diagnosis — why current bark is "horrendous" (anchors verified 2026-07-10)

The texture recipe is the LEAST of it. The generating context — UV scale, dead height
channel, zero geometry, LOD gates — produces most of the badness:

1. **Zero geometric relief.** Trunk silhouette is a smooth low-poly cylinder:
   `ringsForLevel` gives level-0 trunks 14 radial segs (src/vegetation/TubeMesh.ts:307),
   multiplied by `HERO_DIETS.barkK` — beech hero trunks are 7-sided prisms
   (src/vegetation/VegLibrary.ts:176, barkK 0.5). No displacement anywhere; relief exists
   only as a normal map.
2. **The baked height channel is never consumed.** BarkSynth bakes height into texB.w
   (src/gpu/passes/BarkSynth.ts:202) but NaniteResolve reads only texB.xy for the normal
   perturb (src/nanite/shade/NaniteResolve.ts:742-754). No POM, no displacement.
   Roughness is also baked then ignored (resolve is diffuse-only, NaniteResolve.ts:611).
3. **Tiling period ≈ 0.5 m.** UVs tie the tile to `barkRepeats`: u = k/seg·uRepeats,
   v = vAlong/(2π·baseR)·uRepeats (TubeMesh.ts:469-470). Spruce (barkRepeats 5,
   baseR ≈ 0.4 m): one 2048² tile spans ~0.5 m around AND along — the identical pattern
   repeats ~50× up a 25 m trunk; worley "plates" are miniature (~3 cm × 12 cm).
4. **Mip policy over-blurs.** Analytic LOD uses the conservative min world-per-texel axis
   (NaniteResolve.ts:670); explicit .level() disables the requested anisotropy=4 —
   grazing trunks (the common case) smear. `?nanbark=grad` is documented NaN-broken
   (NaniteResolve.ts:265).
5. **Mid-field is flat-lit.** Normal map + moss gated off beyond `?resfar` 60 m
   (NaniteResolve.ts:319,727,753) — beyond that a trunk is a smooth vertex-normal
   cylinder with tinted albedo.
6. **~535 MB VRAM for this.** All bark maps are 2048² rgba8unorm StorageTextures with
   full mips (~22.4 MB each): 6 per-layer 2D pairs (12 textures ≈ 268 MB,
   VegLibrary.ts:251-255) + the 6-slice array pair (≈ 268 MB, BarkSynth.ts:284-332).
   The 12 per-layer maps are consumed by ZERO live code: PoolPart.make has no callers
   anywhere in src/ (grep-verified; GalleryScene bakes its own).
7. **Recipe:** worley anisotropy comes only from lattice frequency [16,4] with weak warp
   0.12 (BarkSynth.ts:207-231) — furrows read as a regular rectangular grid; no
   ridge-interior detail gating; fissures don't chain into a connected furrow network.

Species reality: spruce, pine, beech, birch, karst(-gnarl), snag
(src/vegetation/Species.ts, barkLayer 0-5). Deadwood (logs/stumps/branches/shrubs)
shares layers 2 and 5 (VegLibrary.ts:411,472).

## 2. Decision — method

**Ship Method A now** (anisotropic fracture-network height field, properly done).
**Defer Method B** (Lefebvre/Neyret growth-strip fracture, EGSR 2002 — no public source,
a full solver reimplementation; REVIVE CONDITION: A's furrow topology still reads
"cellular" after S3 tuning — surface for the user, don't silently park). **Cut Method C**
(reaction-diffusion lichen — appearance-only; moss/mottle already exist as world-space
albedo content in the resolve).

**Frequency-band split (the no-double-count contract):**

| band | wavelength | carrier | who computes it |
|---|---|---|---|
| macro (furrows, plates, buttress ripple) | ≥ 6 cm | real mesh displacement at TreeBuilder time + recomputed vertex normals | BarkField.macroHeight() on CPU |
| meso (crack walls, flake edges) | 6 mm – 6 cm | POM from texture R + normal map BA | same field, meso bands, CPU-baked |
| micro (grain) | < 6 mm | normal map only | CPU bake |
| cavity/AO + albedo mix | all | texture G; albedo derived in-shader | CPU bake (sees full H) |

**One implementation, CPU TypeScript.** The bake moves off the GPU entirely: a single TS
module evaluates the field for BOTH texture texels and vertex displacement — no dual
GLSL/JS drift, no storage-texture format constraints, no readback; packed bytes ride
BootCache so warm boots pay ~0. Cold ≈ 0.5-1.5 s for 6×512² (measure; chunk with
yieldIfDue).

## 3. New module: `src/vegetation/BarkField.ts`

Pure functions, no three.js imports. Periodic in u (wraps at 1 tile), continuous in v.

```ts
export interface BarkFieldParams {
  id: string;            // species id, for logging
  tileW: number;         // world metres per texture tile (both axes at base radius)
  cellsU: number;        // fracture cells per tile, around
  cellsV: number;        // fracture cells per tile, along
  aspect: number;        // worley metric anisotropy (>1 = vertical furrows, <1 = horizontal)
  warp: number;          // domain-warp amplitude (breaks the lattice grid)
  fissureW: number;      // F2−F1 width of the crack band
  macroAmp: number;      // metres — geometric displacement amplitude at reference radius
  mesoAmp: number;       // metres — POM height scale
  plateRound: number; ridgeOct: number; micro: number;
  lenticels: 0 | 1; twist: number;     // grain spiral (v-shear per u), snag/karst
  deep: [number,number,number]; high: [number,number,number]; mottle: number;
}
export const BARK_FIELDS: readonly BarkFieldParams[]; // index == barkLayer 0..5

// full field (bake): [0,1] height + cavity in one pass per texel
export function barkSample(p, u, v): { hMacro: number; hMeso: number; cavity: number };
// displacement (geometry): macro band only, with finite-diff derivatives
export function barkMacro(p, u, v): { h: number; dhdu: number; dhdv: number };
```

Field construction (port pnoise/pworley from BarkSynth to TS — small; delete the TSL
originals in S4):
1. q = (u·cellsU, v·cellsV) + warp · fbm2(...) — Quilez domain warp, wrapped lattice.
2. Anisotropic worley with metric d² = (Δu·aspect)² + Δv² → F1, F2.
   fissure = smoothstep(0, fissureW, F2−F1) = the connected furrow network (vertical
   chains when aspect > 1 because cell boundaries align along v).
3. plate = F1-based dome · plateRound; ridge-interior fine fbm MULTIPLIED by fissure
   (detail only on plates, not in cracks).
4. hMacro = lowFreqUndulation + plate·fissure^0.65 − (1−fissure)·1.0, normalized [0,1].
5. hMeso = the ≥4×-frequency components of (3) + crack-wall micro-lines; micro grain
   rides the normal only.
6. cavity = multi-tap horizon AO over hMacro+hMeso (4-6 taps, bake-time only) — also the
   albedo mix factor; birch lenticels stamp cavity ≈ 0.05 + small hMeso ridge (dark rough
   dashes — physically motivated, no extra albedo channel).
7. twist: shear v += twist·u·tileW before (1) — spiral weathered grain for snag/karst.

**Per-species table (initial values — tune at S3 screenshots):**

| layer | species | tileW | cellsU×V | aspect | warp | fissureW | macroAmp | mesoAmp | twist | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | spruce | 1.4 | 10×2 | 5.0 | 0.35 | 0.30 | 0.030 | 0.008 | 0 | narrow vertical ridge/furrow |
| 1 | pine | 1.8 | 5×3 | 1.6 | 0.45 | 0.42 | 0.045 | 0.010 | 0 | plate mosaic, deep crevices |
| 2 | beech | 2.0 | — | — | — | — | 0.006 | 0.002 | 0 | fbm "muscle" only, no fissure net |
| 3 | birch | 1.6 | 3×6 | 0.25 | 0.25 | 0.55 | 0.008 | 0.003 | 0 | horizontal peel bands + lenticels |
| 4 | karst | 1.2 | 8×2 | 4.0 | 0.90 | 0.35 | 0.060 | 0.010 | 0.15 | twisted deep ridges |
| 5 | snag | 1.5 | 9×1 | 8.0 | 0.30 | 0.25 | 0.040 | 0.008 | 0.20 | long spiral splits |

deep/high/mottle carry over from BARK_TABLE (BarkSynth.ts:118-155) — the palette was
never the problem.

## 4. Bake, packing, caching — VRAM budget

ONE texture replaces four channels' worth of maps. CPU-baked, CPU-mipped (box filter —
deterministic), uploaded as a DataArrayTexture (sampled-only: no storage-format
constraint).

| item | format / size | MB |
|---|---|---|
| barkFieldArray — R=hMeso, G=cavity/albedoT, BA=meso+micro normal xy | rgba8unorm 512²×6, full mips, RepeatWrapping, trilinear | **8.4** |
| registry verts/DAG delta from displacement (§5) | ~+50k verts × 24 B × DAG | ~+2.5 |
| cluster records delta (~+1.3k × 80 B) | | ~+0.1 |
| **new total** | | **≈ 11** |
| **deleted** (12× 2048² 2D + 2× 2048²×6 arrays, all mipped rgba8) | | **≈ −535** |

Albedo/roughness derived in-shader (≈10 ALU on bark pixels): albedo =
mix(deep[layer], high[layer], G), mottle as world-space valueNoise3(wp·0.35) (albedo
content, world-anchored — kills mottle tiling; same family as existing rock/moss
shading). Roughness not carried (resolve is diffuse-only, verified). deep/high reach the
resolve as a uniformArray of 12 vec3 (verify uniformArray in r184 TSL; fallback: pack u8
colors into two spare mesh words — implementer's call, prefer uniformArray).

**BootCache:** store packed level-0 + mip bytes per layer, keyed as usual. Add to
src/nanite/world/BootCache.ts: `import srcBarkField from '../../vegetation/BarkField.ts?raw'`
AND `import srcTubeMesh from '../../vegetation/TubeMesh.ts?raw'` into SRC_HASH (line 81 —
TubeMesh is currently ABSENT and displacement lives there; without this the DAG cache
serves stale smooth trunks). Bump CACHE_REV 8→9. Gate: byte-identical first frame cold
vs warm.

buildVegLibrary order already works: bark bakes at progress 0.2 before trees grow at 0.3
(VegLibrary.ts:244-267).

## 5. Geometry — macro displacement (the silhouette requirement)

**Where:** meshBranch ring loop, src/vegetation/TubeMesh.ts:443-474 (junction-aware
path; legacy tubeForBranch at :126 is ?nojunctions-A/B-only — leave undisplaced). Thread
`relief?: { field: BarkFieldParams }` from buildTree (TreeBuilder.ts:171-178) →
tubesForSkeleton → meshBranch. Only the lod0 head gets relief; r1/r2 discrete rings
(WorldRegistry.ts:1012) stay smooth — they own far distances where 5 cm is sub-pixel.

**What:**
- Radial offset: rr += ampFor(r) · (hMacro(uQ, vQ) − 0.5) · 2, ampFor(r) =
  macroAmp · min(1, r/0.25), skipped entirely when r < 0.06 m (thin/young bark smooth —
  biologically right, stops branch-count explosion; only trunks + thick primaries).
- **uQ, vQ are the f16-quantized uv** (same quantization as GeometryRegistry packing,
  packHalf at GeometryRegistry.ts:531) — evaluate displacement at the value the shader
  will interpolate, so groove geometry and cavity darkening can never drift apart.
- Normals: perturb the analytic ring normal with the field gradient:
  n' = normalize(n − T·dhdu·k − B·dhdv·k), k = ampFor(r)·cellsPerMetre; T = ring tangent,
  B = branch dir. No mesh-wide normal recompute.
- Seam: k=0 and k=seg duplicates get identical displaced positions (field periodic in u;
  frac(uRepeats)=0) — the closed-manifold/QEM-collapse invariant (TubeMesh.ts:311-330)
  untouched.

**Density (replaces fixed 14-seg + barkK diet):** for displaced rings,
segsAround = clamp(round(2π·r / 0.05), 14, 64); insert interpolated rings along the
branch to reach spacing ≈ 0.2 m below 25% height, 0.5 m to 60%, 1.0 m above (lerp
pts/radii/dirs, renormalize). Undisplaced branches keep ringsForLevel unchanged.

**Deltas (estimate, 20 m spruce trunk, r≈0.4):** ~42 rings × 44 segs ≈ 1.9k verts /
3.7k tris vs ~450 today → per pool +3-6k tris (bark heads are 30-80k ⇒ +5-10%).
World: +~1.3k clusters (noise), +~2.5 MB registry VRAM. QEM DAG collapses relief once
error is sub-pixel; shadows raster the same clusters — displaced silhouettes in shadows
for free. MAX_CLUSTER_TRIS=128 unaffected.

## 6. UV remap + tiling strategy (no noise overlays)

Replace barkRepeats mapping (TubeMesh.ts:469-470) with world-proportional tiles:
- uRepeats = max(1, round(2π·baseR / tileW)) — integer (seam law), per branch from the
  species tileW; EXCISE SpeciesParams.barkRepeats.
- v = vAlong / tileW — one tile ≈ tileW metres both ways at base radius (square texels;
  taper compresses u toward the top — features narrow with the trunk, plausible + free).

Repetition control, structural only:
1. Vertical tiling of a v-continuous furrow network is nearly invisible (fissure chains
   cross the tile border by construction); period rises 0.5 → 1.4-2.0 m.
2. Per-variant uv phase: u += variant·0.25 + hash(seed)·frac, v += hash(seed), applied in
   TreeBuilder (geometry-side, keeps displacement+texture aligned).
3. Per-instance: existing yaw + slotHash warm/cool/value tint (NaniteResolve.ts:761-770).
4. v-magnitude drops ~4× (v_max ≈ 12 vs ~50) ⇒ f16-uv snapping ≤0.8% of a tile (≤16 mm);
   the §5 quantized-eval makes even that exact.

## 7. Resolve changes (src/nanite/shade/NaniteResolve.ts, isBD branch :613-774)

**Decode swap (S1):** ResolveWorld.barkTexA/barkTexB → single barkTex; sample once at
.depth(layer).level(lod); albedo = mix(deep[layer], high[layer], G) (+ world-space
mottle), ao = G·0.7+0.3, normal from BA exactly as today's texB path. Deadwood keeps its
dim/moss/rot chain on the derived albedo. Delete the sqrt-decode (tex.rgb·tex.rgb) — no
albedo channels exist anymore.

**POM (S3), hand-rolled — no library node exists:** inside the existing detailNear
structure, three distance tiers:
- **< 30 m — POM.** Tangent frame T/Bi/gnrm and per-axis world-per-uv (|Traw|, |Braw|)
  already exist (:649-670). Sketch:

```ts
const Vw   = normalize(camPos.sub(wp));
const vTS  = vec3(dot(Vw,T), dot(Vw,Bi), dot(Vw,gnrm).max(0.1));
const sUV  = vec2(float(mesoAmp).div(Traw.length()), float(mesoAmp).div(Braw.length()));
const dUV  = vTS.xy.div(vTS.z).mul(sUV);          // full-height parallax offset
const NSTEP = 8;                                   // fixed march + 4 bisection refines
const uvP = uvv.add(dUV).toVar(); const hCur = float(1).toVar();
Loop(NSTEP, () => { /* step uv back by dUV/NSTEP, hCur -= 1/NSTEP,
     break when tapR(uvP) >= hCur */ });           // all taps .depth(layer).level(lod)
// 4 halving refines, then uvP feeds the single full decode tap above
```
  All taps use the precomputed explicit lod (uv is in non-uniform control flow — same
  reason the current path does; :662-666). No depth-write offset: depthNode
  (:1446-1451) stays untouched — cm-scale meso doesn't warrant per-pixel depth surgery;
  the macro band is real geometry, already correct in depth/shadows.
- **30-60 m** — one tap: normal map + AO + albedo (today's near path, minus POM). Moss
  keeps its 60 m gate.
- **> 60 m** — one coarse-mip tap for albedo/AO, geometric normal. Mid-field flatness is
  now covered by displaced geometry normals, not the texture.

**Cost (honest, measure):** POM ≈ 10-13 rgba8 taps of one well-cached 512² slice on bark
pixels < 30 m only — worst pose is trunk-heavy, expect ≈ +0.15-0.4 ms @dpr2. Gate §9.
Known residual (documented, not fixed here): explicit .level() still disables aniso and
keeps min-axis blur at grazing angles; acceptable because vertical furrow content
survives isotropic mips far better than the old isotropic plates.

**Debug bisects:** keep ?nanbark=const|lN; DELETE ?nanbark=grad (NaN-broken, :265); add
?nanbark=h (raw height vis) and ?nanpom=0 (skip POM tier).

## 8. Excision (complete removals, S4 — grep-proof each)

| item | anchor | action |
|---|---|---|
| src/gpu/passes/BarkSynth.ts (whole file: GPU bake, BARK_TABLE, BARK_RES, both array bakes) | — | delete; BARK_FIELDS + BARK_TEX_RES in BarkField.ts replace it |
| per-layer 2D bakes + barks Map + barkOf + VegLib.barks | VegLibrary.ts:246-261,188,707 | delete (≈268 MB back) |
| PoolPart.make + all material-factory wiring | VegLibrary.ts:49,284-293,411,456,489,514,639… | delete field + closures (zero callers, grep-verified); then delete each render/VegMaterials.ts factory that reaches zero callers (barkTexturedMaterial, likely deadwoodMaterial/rockMaterial/flowerMaterial/foliageMaterial — re-verify each; GalleryScene is the remaining consumer to port) |
| SpeciesParams.barkRepeats | Species.ts:56,106,…, VegTypes.ts | replace with the field's tileW (§6) |
| HERO_DIETS.*.barkK + hero.barkK plumbing | VegLibrary.ts:172-180, TreeBuilder.ts:168 | delete — density is radius/relief-driven now |
| ?nanbark=grad path + uvAt/rayDir helpers | NaniteResolve.ts:672-711 | delete |
| ResolveWorld.barkTexA/barkTexB dual binding | NaniteResolve.ts:121-122,616, ForestScene.ts:518-519, TerrainScene.ts:219 | single barkTex |
| GalleryScene bark usage | GalleryScene.ts:156,187,272,360,420 | port to thin MeshStandardNodeMaterial sampling the new array at fixed layer |

Post-excision verify: `grep -rn "BarkSynth\|barkRepeats\|barkTexA\|barkK" src` → only
legitimate hits, documented.

## 9. Staged execution (each stage: commit only after gates)

Canonical URLs: world `?scene=world`, worst pose `&cam=-582.1,302.4,1006.1,2.5692,-0.0077`,
perf flags dpr=2&nanodisp=1&grass=0&nanshadow=0&ksplit=1&fp16w=1. Screenshots via
Playwright recipe (persistent context; rerun once on first-load vite fatal). p95 via
tools/perf/interleaved_ab.mjs (?fly=1 restarts t=0). Counter noise ±5-7%.

- **B1 — BarkField + CPU bake + decode swap** (no POM, no displacement). New
  BarkField.ts, bake+pack+mip+upload in VegLibrary, BootCache entry + SRC_HASH additions
  + CACHE_REV 9, resolve single-texture decode, §8 texture-side excisions.
  Gates: tsc; worst-pose smoke; screenshot ×6 species (gallery); p95 ±2% of baseline;
  cold-vs-warm first frame byte-identical; VRAM delta confirmed (≈ −535 MB).
- **B2 — UV remap + displacement + density** (§5 + §6). Gates: tsc; smoke; closeup +
  60 m screenshots per species (silhouette visibly non-cylindrical on spruce/pine/karst/
  snag; beech/birch subtly rippled); printed cluster/vert deltas ≈ §5 estimates (±50%);
  DAG cache invalidated exactly once then warm; p95 budget +0.3 ms — else premise-audit
  before touching amplitudes (suspect the density rule first).
- **B3 — POM + tier gates + param tuning** (§7). Gates: grazing-angle closeups;
  ?nanpom=0 A/B — POM budget +0.4 ms p95 worst pose; if exceeded, PRE-APPROVED fallback
  order: gate 30→20 m, then steps 8→6, then ship geometric+normal-only.
- **B4 — full excision + Gallery port + final proof** (§8). Gates: tsc; grep-proof;
  cold+warm boot timings; final six-species screenshot sheet + worst-pose p95 A/B vs
  pre-B1 baseline in ONE interleaved session.

Verification rule every stage: user-observable output (screenshot), not just tsc; any
disappointing visual/perf result triggers the go-up-a-level audit BEFORE
parameter-twiddling.
