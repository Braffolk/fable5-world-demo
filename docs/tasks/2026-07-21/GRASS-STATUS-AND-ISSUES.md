# Grass rendering — status & issues (2026-07-21)

Factual record only. **No hypotheses about causes are included** (by user directive).
This documents: the overarching goal, the original issues we set out to fix, the one
piece of work considered done (O(1)), every change made during this session, every new
issue observed after those changes, and the debug mode that was attempted and botched.

Work was **stopped by the user** at the original handoff point. It resumed later on
2026-07-21; the continuation and its verified checkpoint are recorded in §9.

---

## 1. Overarching goal (user, start of session)

- Evaluate the fit of the existing grass raycast lane against Aleksandr Sannikov's
  precomputed-raycast technique (gamedev.ru article `docs/deep-research/grass/`,
  shadertoy `tsVGRd`, and his 2023 flow-map update; all read in full this session).
- Quality bar (user, verbatim intent): the moss/grass/ground-cover layer must be
  **INDISTINGUISHABLE from how it appears in the real world**. The current grass did
  not meet the bar either.
- **Significantly faster** than the current grass (the ray lane measured ~10 ms).
- Real grass is **not only blades**: it is ≥10 species distributed in **patches** (not
  random per-blade), i.e. a ground-cover system (grass + moss + overlays).
- **Control field (hard requirement, not to be reduced):** the raycast shader must read
  a spatial ground-cover field telling it *what cover is here and how much* everywhere
  (under trees less grass, swamps get sedge, etc.).
- **Estonia-native species only.**
- **O(1)** — the single-fetch speedup from the article (no per-pixel step loop).
- Remove the previously-generated geometry sphagnum moss ("poop piles").

## 2. Original issues we set out to fix

### 2a. The four original issues (user, verbatim)

> - there seems to be some issue with "expected grass sparsity" being applied very odd.
>   like in taevaskoda, there is a dirty (could be sand, idk) area that from some distance
>   in air, idk, lets say about 50m from it, renders expectedly. some sparse grassy places,
>   most of it bare. now the area BEHIND it in forest where i would expect grass, its grass.
>   so thats, lets say, easily 150m away. all grass. BUT when i fly in towards this
>   sandy/dirty area, suddenly it gets a thick area of grass when im close enough. idk, lets
>   say smth like 15-40m. hard to tell exactly. guessing
> - the cells that the grass is made out of, work as some sort of clippers. meaning if a bit
>   of grass is trying to extend outwards, into the neighboring one, it cant, its clipped.
>   its not seamless. it has clear cell like seams/clipping effect going on
> - the direction of grass makes zero sense. it doesnt feel like grass is coming out of
>   ground. sometimes it feels like its coming out from some side, sometimes from top
> - another cellines bug. same cells that i described before, but when going more into top
>   down view, their centers get emptied. like only their edges are grassy looking, centers
>   are empty. but in a more oblique view, it gets the grassy stuff (even if wrong and all
>   fucked looking, still does get it)

Labels used elsewhere in this doc: **C** = sparsity-applied-oddly (issue 1),
**D** = cells-clip (issue 2), **E** = direction-makes-no-sense (issue 3),
**F** = top-down-centres-empty (issue 4).

Also stated as grass problems earlier in the session (part of the §1 goal): **A** grass is
only blades — real grass is ≥10 species in irregular patches; **B** jitter / deformation
artifacts.

### 2b. Additional detail the user added during debugging

- **Reference bar (important).** In the gamedev.ru (Sannikov) implementation **video**, the
  camera is rotated around the grass, and **top-down AND side views show NO artifacts at
  any angle** — it all looks like high-quality **geometry**, when in fact it is not real
  geometry. Our implementation shows artifacts at those same angles (holes / empty centres
  / clipping) that his does not. This is the quality target the grass is being measured
  against.
- **G. Not rooted.** The core of issue E, restated: the grass is not rooted in the ground —
  it does not grow out of the ground.
- **H. Banding / stretching.** Close up, at various band distances from the camera, the
  grass starts stretching out in bands.
- **I. "Vertically flipped" drips** (user screenshot): on a hillside near a sandy/bare
  boundary, grass appeared to hang downward like drips.
- **J. Sliding / inner-cell warping.** When moving on an oblique angle, the grass appears to
  slide slightly faster than the camera; described as inner-cell warping; slight, and it
  "stops at some point" (not the whole thing sliding at once).
- **Black grass in the debug view.** ALL of the grass renders black; with the debug mode it
  gains only red/green (later magenta) **outlines** — the black interior remains. The user
  confirms this black is grass (see §6).

## 3. What was completed (the only item considered good/done)

- **O(1) single fetch (commit `f9238ed`).** The per-pixel 256-step DDA march
  (`loopUN('gro', 0, 256)`) in `src/nanite/grass/NaniteGrass.ts` was deleted and replaced
  with the article's single terrain-anchored LUT fetch: the vis-buffer election depth
  gives point O, the guide texel's ground plane reconstructs the sward-top entry E, and
  one trilinear fetch of the boot-baked (x, z-in-tile, angle) LUT gives the path, with
  `|OB| = |OA|/cos α`. Reported measurement at the time: grassRay ~15–20 ms → ~0.3–0.5 ms,
  flat across camera poses.
- This is the only change the user considers a genuine improvement.
- Also before this arc: the rejected geometry sphagnum moss was excised (commit
  `c46878e`).

## 4. Changes made during this session (beyond O(1))

All of the following are in the working tree. Only `f9238ed` and `2ab69cf` were committed;
everything after `2ab69cf` is uncommitted (see §6).

In `src/nanite/grass/NaniteGrass.ts`:

- **Anti-tiling hex 3-tap texture-bombing** (commit `2ab69cf`), replacing the previous
  per-cell single-tap swirl "bomb". Three hex-lattice nodes, each with an integer-hashed
  (pcg2d) rotation + phase, triangle-weight blended, with variance preservation toward a
  baked mean-R (`meanR` added to `GrassRayBake.ts`). Knob `?grasshex=N` (default 2).
- **World-anchored the tile lattice** (commit `2ab69cf`). Added CPU-computed uniforms
  `uOTx`/`uOTz` (exact-integer guide origin in tile units, reduced mod 4096 on the CPU).
  The reconstruction now uses `worldTile = phE/PITCH + uOT`.
- **Bilinear entry-side (E) ground** (commit `2ab69cf`): `gPE` changed from a nearest,
  per-guide-texel faceted plane to a bilinear 4-corner sample.
- **Bilinear O-side ground + gradient** (uncommitted). Added a `bguide(rc)` helper
  returning bilinear `{ground, grad}`; `groundO`/`gradO`/`hO`/`dhdt` and the `gOB`
  plane-agreement term now use it instead of nearest per-texel values. Removed the now-dead
  `texOrg`, `texCO*`, `txfO`/`tzfO`, `cvO` locals.
- **Guide fields sampled at O** (uncommitted): the `guv` used for the density gate + wind
  fields (`guideFieldT1/2/3`) was moved from the walked-back entry E's position to O's
  position.
- **Removed the static random per-tile lean/arc** (uncommitted): `SlxB/SlzB` (linear lean)
  and `a1xB/a1zB` (quadratic arc) in the guide bake are now gated to 0 via a new knob
  `?grassrandlean=K` (default 0). Wind lean (`SwxB/SwzB`) is unchanged.

In `src/nanite/shade/NaniteResolve.ts`:

- **Reduced the terrain-normal pull** on the proc-grass normal from `0.5→0.85` (near→far)
  to `0.18→0.65`, via a new knob `?grassnrmpull=near` (default 0.18). Intended so the
  blade's own normal drives near-field shading.
- **Removed the dead legacy mesh-grass shading path** (matClass 5) via a subagent
  (`−79` lines). Verified: no geometry produces matClass 5; boot PASS after removal; grass
  rendered identically. Stale "legacy-grass / matClass 1-5" comments updated to "1-4" in
  `NaniteResolve.ts` and one comment in `NaniteFrame.ts`.
- **The `?grassdbg=tip` debug mode** — see §5. It is currently left in the file in its
  last (magenta) state; per the stop directive it was not reverted.

## 5. New issues observed AFTER the above changes

Reported by the user while testing, in order:

- After the O(1) + anti-tiling changes: the visible **grid was replaced by a
  voronoi-looking pattern**; clipping "99% gone" but some **dark seams at rare oblique
  angles**; the voronoi cells have **empty middles** (from a slightly-top-down angle;
  oblique does not show it); the **voronoi pattern reset to a new one on every camera
  position move** (not angle); a **green halo** appeared toward one direction when standing
  and looking straight down.
- The **dark seams** were reported to follow an actual **square grid**, independent of the
  voronoi pattern.
- After world-anchoring + bilinear O-side + density-at-O: user reported **"nothing
  changed / nothing improved"** for the named issues; the voronoi-reset and dark-square-
  seams were reported fixed.
- After reducing the terrain-normal pull: user reported the edge **clipping became more
  visible** again ("still 100% happening, clipping at edges of cells").
- On a hillside near a sandy/bare boundary: grass appeared **vertically flipped / hanging
  down like drips** (user screenshot).
- Persistent through all changes: **grass is not rooted / does not grow from the ground**;
  **stretches in bands**; user states these were **not improved** by any of the changes.

Separately (not a grass-code issue): the Estonia world briefly rendered as **cubes with no
trees**. This was traced to a wrong cooked data build being hosted; fixed by repointing
`asset-gen/data/out/latest.json` from `m/b7afac38c167af79` to `m/fffd3771349f27c7`
(cook_rev 4). This is a data-file (gitignored) change, not a shader change.

## 6. The debug mode (`?grassdbg=tip`) — historical attempts and correction

**Intended behaviour:** color grass by its tip parameter so rootedness could be judged —
base (`t=0`) = red, tip (`t=1`) = green. If red sits on the ground line the grass is
rooted.

**Historical sequence before the handoff and the observation after each:**

1. In the proc-grass resolve block, set `albedo = mix(red, green, t)` (lit path). —
   Observed (user): grass showed red/green **outlines** with a **black interior**; the AO
   was more visible than the grass. (This version was lit, so shadows applied.)
2. Added an unlit early return `if (grassdbg==='tip') return vec4(albedo,1)` at the
   debug-override point. — Observed (user): **no change; still black inside.**
3. Rewrote it self-contained at the debug-override point using `gpTip` (the tip param the
   proc-grass block writes) with `isGP` selecting grass; **every non-grass pixel forced to
   gray (0.2)**. — Observed (my screenshots): terrain and voxel trees rendered gray; near
   grass rendered as red/orange/green **speckles**; large **black masses** remained (a
   ground-level shot at the Taevaskoja spawn and a turned-around forest shot).
4. Forced **solid magenta** for every `isGP` grass pixel (no `gpTip`, no mix), gray
   elsewhere. — Observed (user + my screenshot): grass clumps **remained black**, now with
   **magenta outlines only**. Terrain gray, voxel-trees gray, water blue/teal. The user
   confirmed: it is grass, the black is still there, only the outlines are magenta.

**Correction after the handoff:**

- The black pixels were the same procedural grass geometry. Procedural grass owns election
  ids with `0xc0000000 | body`; `isGP` detects those ids in the ordinary resolve, and the
  procedural block forces `matClass=255`. The earlier statement that the black interiors
  were outside the resolve-owned grass pixels was false.
- The earlier fartile statement was also false in this context. Fartile voxel records and
  procedural grass use the shared visibility/resolve pipeline; neither explains a second
  black grass renderer.
- The pass-wide constant return in the old debug attempt dead-stripped graph inputs that
  the Three/WebGPU material still expected and produced `null.constructor` material-build
  failures. The repaired mode keeps the ordinary `lit` graph alive and replaces it only
  inside `If(isGP)`.
- Dense false-colour interiors stayed intact through the aerial composite and became black
  only in the later temporal/grading chain. Debug mode now terminates at that proven-clean
  aerial composite; production output is unchanged.
- Verified exact localhost boots at the close Taevaskoda pose now render every procedural
  grass pixel red→green, retain normal colours for terrain/trees/water, and contain no
  black grass interior (`/tmp/grass-exact-default-tip.png`, frame 190).

**Debug tooling added (scratchpad only, not in repo):**

- `scratchpad/shot_ground.mjs` — boots, reads the spawn pose via `window.__laas.getPose()`,
  sets a pose via `setPose` (args: pitch, dy, yaw, dx, dz), settles, screenshots at dpr2.
  Estonia spawn pose observed: `p = [311123, 96.378, 190723], yaw 0, pitch -0.18, fov 55`.

## 7. Code state at the original stop point (historical; superseded by §9)

**Committed this session:**

| commit    | summary |
|-----------|---------|
| `c46878e` | Excise rejected geometry sphagnum moss (before this arc) |
| `f9238ed` | Grass TRUE O(1): kill the 256-step march, single terrain-anchored LUT fetch |
| `2ab69cf` | Grass anti-tiling: world-anchored hex texture-bombing + bilinear entry ground |

**Uncommitted (working tree) — relevant to grass:**

- `src/nanite/grass/NaniteGrass.ts` — bilinear O-side, density-at-O, random-lean-off knob.
- `src/nanite/shade/NaniteResolve.ts` — dead mesh-grass path removed; terrain-normal pull
  reduced; `grassdbg=tip` debug left in its magenta state.
- `src/nanite/frame/NaniteFrame.ts` — one comment text change ("matClass 1-5" → "1-4").

**Uncommitted — unrelated / pre-existing (not touched for grass this session):**

- `asset-gen/.../forest_exemplar/adjacent.py`, `docs/METAL-PROFILING.md`,
  `docs/tasks/2026-07-13/KNOWN-ISSUES.md`, `src/core/ProfileBoot.ts` (older harness fix),
  and various untracked docs/data.

**Knobs / URLs in play:**

- Test URL (Estonia, dpr2): `http://localhost:5173/?scene=world&src=estonia&dataurl=http://localhost:8787&dpr=2`
- Debug: append `&grassdbg=tip` (currently forces magenta on grass, gray elsewhere).
- Grass knobs: `grasshex`, `grassrandlean`, `grassnrmpull`, plus pre-existing `grassbakres`,
  `grassbakang`, `grassbakw`, `grassbakt`, `grassbakn`, `grassquad`, `grasssway`,
  `grassrayend`, `grassdbg=flatres|raysetup`.
- Data server must serve `asset-gen/data/out` on `:8787`; `latest.json` must point at
  `m/fffd3771349f27c7`.

## 8. Status of the original issues

| # | issue | status |
|---|-------|--------|
| A | only blades / no species / no patches | not addressed |
| B | jitter / deformation | not addressed |
| C | density/sparsity applied oddly (Taevaskoja sand) | implementation corrected; user acceptance pending |
| D | cells clip (seams) | implementation corrected; user acceptance pending |
| E | direction makes no sense (side/top) | implementation corrected; user acceptance pending |
| F | top-down cell centres empty | exact-default near-nadir checkpoint clean; user acceptance pending |
| G | not rooted / doesn't grow from ground | repaired tip checkpoint clean; user acceptance pending |
| H | stretching in bands | distance-dependent geometry removed; user acceptance pending |
| I | vertically-flipped drips | base-space rooting implemented; user acceptance pending |
| J | sliding / inner-cell warping | camera-relative transforms and 2×2 replication removed; user acceptance pending |
| — | O(1) single fetch | **done** (`f9238ed`) |
| — | debug mode `grassdbg=tip` | **repaired and WebGPU-verified** (`52df93d`) |

## 9. Continuation checkpoint (`52df93d`, 2026-07-21)

The following changes are committed in `52df93d` without restoring any per-pixel march.
The runtime remains fixed-cost O(1): a bounded number of predetermined texture candidates,
no ray-step loop, and no distance- or density-dependent iteration.

### Geometry and density corrections

- Removed camera-distance thinning from the guide mask, removed the extra 8–14 m scruff
  population, and removed the 50–90 m camera-distance blade-height growth. The physical
  sward height is now camera-independent.
- Replaced the three-node hex blend of unrelated first-hit depths/normals with the article's
  geometry-safe composition: two globally continuous layers separated by golden-ratio·π.
  The second layer has an incommensurate physical scale; the nearer complete record wins.
- Density tiers no longer interpolate visibility records. A nearest-texel root id is checked
  with the exact same `pcg2d` keep threshold used by the CPU bake; the shader selects the
  complete denser record or the nested sparser fallback.
- Exact `d=0` origin occupancy is preserved categorically instead of being blurred into a
  small positive depth. This removes the division singularity for vertical/near-vertical
  rays and retains filled top-down blade footprints.
- Production bake defaults are back on the article's geometrically exact parallel-extrusion
  case: `shiftK=0`, `thickK=0`, `arcK=0`. Those controlled-error extensions remain explicit
  query knobs. Coverage now comes from six actual fibers per layer, not distance inflation.

### Rooting, slope, normals, and motion

- Implemented the exact non-orthogonal derivative for `x=P+F(h), y=g(P)+h`:
  `dh/dt=(rd.y-m·rd.xz)/(1-m·F'(h))` and `dP/dt=rd.xz-F'(h)·dh/dt`.
- Reconstructs the blade base coordinate `P=x-F(h)` at B, samples ground beneath P, derives
  the stable body/root cell from P, and computes tip position from physical height `h`.
- The CPU precompute stores the face actually entered. An exact origin/top hit stores an up
  normal. The later per-cell shading-only normal twist and per-cell tip hash were removed.
- Removed DPR-driven 2×2 ray replication and its copied depth/normal. Every output pixel now
  runs its own fixed-cost lookup, removing that screen-space source of sliding/warping.

### Verification evidence

- `npm run typecheck`: pass.
- Exact real-WebGPU localhost close/oblique production boot: pass at frame 179,
  `cam=311123,47,190723,0,-0.12`.
- Exact-default near-nadir production boot: pass at frame 136,
  `cam=311123,120,190703,0,-1.45`; dense grass stays filled while the sandy density-field
  clearing remains bare.
- Repaired `grassdbg=tip` close boot: pass at frame 190; complete red-base→green-tip grass,
  ordinary scene colours retained, no black grass interiors.
- Timestamp-query harness at DPR 2: median `c.grassRay = 0.39 ms` over seven samples at the
  Taevaskoda pose. This preserves the achieved speedup versus the former ~15–20 ms march.

This checkpoint corrects the implementation-side causes and passes the recorded automated
and visual inspections. Final issue closure remains subject to the user's live motion/angle
acceptance; the multi-species/moss groundcover work begins from this committed grass base.

## 10. Generic ground-cover carrier checkpoint (2026-07-21)

The first post-grass checkpoint is an end-to-end, optional `groundcover` layer. It is a
control carrier, not yet the promised multi-species/moss geometry result: production still
uses the validated grass LUT for every non-bare type. Its inspectable purpose is to prove
that cooked cover authority survives the complete streaming and O(1) visibility path before
the precomputed geometry becomes type-keyed.

### Frozen carrier contract

- One logical LAC1 layer (`groundcover`, id 11), LOD0 at 2 m, carries eight u8 planes:
  `[typeA,typeB,clumpLo,clumpHi]` categorically and
  `[blend,vigor,moisture,canopyProximity]` continuously.
- Runtime splits it into nearest- and linear-filtered rgba8 carriers. Old manifests omit the
  layer and compile to canonical grass id 0 with the prior procedural density.
- The camera-window is 512², not the general u8 1024²: at 2 m it has a 512 m half-extent
  around the grass lane's 155 m radius. The two retained carriers therefore cost 2 MB total,
  keep StreamBrain at 126.3 MB under its 128 MB ledger, and expose no less rendered area.
- Guide-record words 2–3 carry the packed control because they are already included in the
  first `uvec4` record loads. Words 6–7 mirror the values for the approved external 32-byte
  layout, but the ray kernel does not fetch that tail. The earlier design note calling only
  words 6–7 "free" is stale after the O(1) rebuild; reading them would add memory traffic.
- Cover election happens only after base-space root recovery, using the nearest complete
  control record plus the cook-side clump id. The six low body-id bits carry `GroundCoverId`;
  legacy grass remains exactly zero. There is no screen-noise type decision.

### Cook-side v1 scope

- `GroundCoverId`: grass 0, moss 1, sedge 2, lichen 3, forb 4, dwarf shrub 5, bare 63.
- The v1 cook maps the existing cooked understory communities to functional two-type mixes,
  derives vigor from the existing suitability-cut density, moisture from soil wetness,
  canopy proximity from CHM cover/distance, and a world-seamless domain-warped clump id.
- This is deliberately not labeled as the final >=10 Estonia-native species/facies cook.
  It establishes the representation and produces visible ecological patches while the
  native-species GPU baker and type-keyed geometry atlas are implemented next.

### Shader-performance rationale

- Ray candidate count, fixed texture candidates, dispatch shape, barriers, synchronization,
  and binding count are unchanged. No march and no shader loop were introduced.
- The retired nested 8×8 guide occupancy loop was removed; it had no surviving mask consumer.
  Its density output is now the analytic mean of the four bilinear corner probabilities.
- Ground-cover texture taps occur once per guide texel, not per pixel. Ray-time type decoding
  reuses already loaded guide records. Moving the control read from entry record to root
  record adds no storage load and shortens two register live ranges.

### Verification

- Taevaskoda cook: 64/64 control chunks; immutable overlay recipe
  `5ec67099fa2b5415d6d0f6ccf3331884df5f2122ae039e7c87a27389af20721a`, manifest SHA-256
  `14414ab1606ed57587db09582be26be6bf12823b64a77100b4faec8197258681`.
- `npm run typecheck`: pass. Ground-cover cook tests: 3/3 pass. Focused TerrainField and
  RemoteWorldSource tests pass. `PlaneFill.micro.test.ts` retains one unrelated stale
  expectation (1536 vs the pre-existing 2560 finest micro-height window).
- Exact localhost production boot at `cam=311123,47,190723,0,-0.12`, DPR2: pass with no
  page error, TSL diagnostic, shader validation error, or uncaptured WebGPU error.
  TerrainField is 151.8 MB; retained brain planes are 126.3 MB. Seven-sample medians before
  the window correction were `c.grassRay=0.46 ms`, `c.grassGuide=0.13 ms`; the final boot
  reported `c.grassRay=0.39 ms`, `c.grassGuide=0.13 ms`.
- Exact `groundcoverdbg=type` boot after root anchoring: pass at frame 84. The coloured
  patches are the same procedural geometry records that production shades as grass; the
  debug is not a far-tile/pixel-ownership surrogate.

## 11. Type-keyed runtime seam and moss-baker checkpoint (2026-07-21)

This section is the current continuation boundary after the committed carrier. It does not
claim a finished species palette or finished moss.

### Runtime seam now exercised

- A bounded root-stable A/B query replaces shell-point type selection. Each cooked candidate
  is queried with its own height/deformation/atlas profile, recovered to a world-space base,
  re-sampled against the root control, and accepted only when that root elects the same type.
  Pure `A == B` patches issue one candidate; actual mixtures issue two fixed candidates and
  keep the nearer valid surface. Both calls are TSL graph expansion, not a WGSL loop.
- Categorical type and clump carriers remain nearest. The already-bound guide T3 now carries
  continuous `[vigor, blend, moisture, canopyProximity]`; no interpolated type id is possible.
- The atlas packer gives every profile private last/first angular guard slices, preventing
  trilinear angle wrap from crossing into the next type. Four density-tier bindings are
  retained; no ray binding was added. Focused deterministic/nesting, ellipse, cap-normal,
  seam-pack, and incompatible-layout tests pass (5/5).
- Production resolve now decodes the 6-bit cover id into a 64-entry indexed material table.
  Grass preserves its prior colour/AO/translucency values; the five functional forms have
  distinct provisional parameters. These are functional-form materials, not native-species
  provenance claims.
- Corrected streamed-coordinate identity from `floor(relative + gfx)` to
  `floor(relative) + gfx`; the former loses fine-cell identity at Estonia-scale floats.

The rejected intermediate elections are useful regression evidence: guide-shell election
made metre-scale rectangular type quilts; fine-cell IGN produced diagonal planted rows; a
fine-cell integer hash removed the rows but still changed one physical object across
pixels/views. The root-stable query is the first version whose type-debug image consists of
whole coherent tufts without those patterns.

### Performance and WebGPU evidence

- Exact localhost production/type boots pass with no page, TSL, WebGPU validation, binding,
  pipeline, command-buffer, or uniform diagnostics.
- A deliberately mixture-heavy DPR2 view measured median `c.grassRay = 0.59 ms` over seven
  timestamp samples, versus the validated all-grass checkpoint's `0.39 ms`. The fixed second
  type query therefore costs about `0.20 ms` in the exercised worst-case mix and remains below
  1 ms; pure patches retain the coherent single-query branch.

### Parked analytic moss-cap path (two failed real attempts)

- Reusable result: the CPU reference baker can analytically intersect an upper ellipsoid cap
  at a bounded set of ray-elevation slices and pack rounded upward normals. Eight logarithmic
  elevation blocks plus the other functional forms produce a 130-slice, four-tier atlas
  (about 8.1 MiB). The exact atlas boots successfully in real WebGPU.
- Real attempt 1, actual cooked moss component: full production boot passed at
  `cam=318703,53.2,186089,0,-0.16`, packed ground `50.97 m`. The full scene obscured the
  cushion shape and still read as irregular generic green cover, so it was not accepted.
- Real attempt 2, isolated terrain+ray geometry at a stronger moss/sedge point
  (`cam=310227,67,189511,0,-0.72`, packed ground `64.59 m`): both production and
  `groundcoverdbg=type` boots passed, but the moss pixels themselves showed rectangular strips
  and square holes. Filling moss roots instead of applying grass's runtime cell thinning
  reduced holes but did not remove the 64² tile's visible spatial/cell quantization.
- Exact blocker: this boot-time analytic fixture is a 64² stratified union of caps, not the
  approved arbitrary-mesh GPU bake, baked horizon/AO/thickness payload, or native Sphagnum
  profile. Further radius/spread tuning would lower the final quality ceiling and would repeat
  the project's recorded two-failure pattern.
- Active fallback: build the approved offline GPU raster-projection baker. Keep the analytic
  cap and its focused tests as a numerical oracle only. Resume CPU-cap visual tuning only if
  the GPU raster path cannot produce a deterministic first-hit atlas after its own bounded
  two-attempt end-to-end cycle, or if it exposes a representation bug shared by both paths.

O(1) remains non-negotiable throughout: none of the above adds a per-frame march, ray-step
loop, distance-dependent iteration, or shader barrier.

## 12. Exact-profile carrier, GPU baker, and first botanical profile (2026-07-21)

This supersedes §11's continuation boundary. The analytic cap remains an oracle only; it is
not used for production Sphagnum.

### Exact cooked authority

- The v2 carrier stores exact native profile ids, not only the six functional ids. The fixed
  palette is: *Agrostis capillaris*, *Avenella flexuosa*, *Calamagrostis canescens*, *Carex
  cespitosa*, *Eriophorum vaginatum*, *Sphagnum capillifolium*, *Pleurozium schreberi*,
  *Cladonia rangiferina*, *Oxalis acetosella*, *Maianthemum bifolium*, *Vaccinium myrtillus*,
  and *Calluna vulgaris* (ids 0–11 respectively).
- A third rgba8 carrier supplies the 12-bit root-neighbour closure mask. The ray graph expands
  twelve fixed guards at TypeScript build time; cooked cells normally enable only their local
  A/B pair. There is no WGSL loop or runtime iteration over the palette.
- The exact-profile Taevaskoda recook completed 64/64 chunks. Recipe:
  `4b60720f863a860c91d313c33455514c3e0129e919237e62934ca395351d7112`;
  locally published manifest: `m/b75736653a54055a/manifest.json` (cook revision 7).

### Periodic arbitrary-mesh GPU baker

- `GCRP/v2` defines a periodic top-XZ first-hit profile: 16 azimuths × four elevations
  (15°, 35°, 55°, 75°), 64² interior texels with a one-pixel wrapped gutter, an 8×8 slice
  atlas, normalized first-hit depth, octahedral normal, and coverage.
- The GPU path rasterizes submitted indexed triangles into `depth32float`; it is not tied to
  caps, fibers, or another analytic primitive. CPU ray tests are verification oracles only.
- The first accepted botanical artifact is an original deterministic procedural *Sphagnum
  capillifolium* periodic carpet: 25,397 vertices, 41,184 triangles, 922 primary branches,
  287 branch forks, 88 non-grid star capitula, and exact periodic seams. Artifact SHA-256:
  `cb61dd42c6265067f0be9a320d763da928b1e65a60ae3ebb359f49b3137a9959`.
- Two Apple-Metal GPU bakes were byte-identical. The strict selected-pixel comparison reported
  zero ownership mismatches, RMS depth error `2.08e-7`, maximum depth error `2.87e-6`, and
  minimum normal dot `0.9999867`.

### Runtime and black-tip correction

- Production *S. capillifolium* uses the GCRP profile through two globally continuous
  incommensurate geometry layers. Each layer performs four fixed direction-corner taps;
  the nearer complete hit wins. This adds one filterable atlas binding and eight bounded taps
  only for a queried Sphagnum candidate. Dispatch, barriers, synchronization, and workgroup
  shape are unchanged.
- `rgba16unorm` profile bytes are converted once during loading to `rgba16float`. This retains
  the 8-byte/texel GPU footprint and supplies the filterable float binding on WebGPU devices
  that do not expose optional `rgba16unorm` sampling.
- The periodic lookup now carries the winning authored profile-space hit elevation into the
  ray output. It no longer substitutes the generic blade-height reconstruction for a connected
  carpet. This is fixed scalar ALU only: no fetch, binding, branch loop, or march was added.
- Exact DPR2 localhost production and `grassdbg=tip` boots both pass at the moss-heavy pose
  `cam=310227,67,189511,0,-0.72`. The unlit debug shows every procedural-cover pixel on the
  red-base→green-tip ramp; there are no black procedural-cover interiors. The canonical
  Taevaskoda production gate at `cam=311123,47,190723,0,-0.12` also passes.
- Static gates at this boundary: TypeScript typecheck pass, 25 focused TypeScript tests pass,
  four focused cook tests pass, and `git diff --check` pass.

### Current boundary

- The system carries all twelve exact identities but has one accepted authored periodic
  geometry profile so far. The other eleven identities still use analytic integration
  fixtures and therefore do **not** satisfy the approved ≥10-species visual deliverable.
- Loading one 2D texture per profile would scale the bind group with species count. The active
  next step is one packed multi-profile atlas/binding plus growth-form-specific GPU-baked
  periodic assets for the remaining native palette.

## 13. Calamagrostis single-species acceptance and reconstruction proof (2026-07-21)

- `?grassprofile=2` remains the only active visual-acceptance view. Multi-profile code and
  assets are preserved but must not be re-enabled until this view is accepted.
- The accepted source mesh is 1.15 m tall with 2,049,985 vertices and 2,171,134 triangles.
  Its four-direction QA is under
  `data/work/groundcover-mesh-review/6e3dfab54aa214c75e395e8be733cdc08aa0979b8bcb2360a11a46c8cc82aa4e/qa/2-calamagrostis-canescens`.
  The compact upward-branching, fluffy panicle source shape is committed at `d075c79`.
- The full shader-independent reconstruction derivation and counterexamples are preserved in
  `GRASS-RAY-RECONSTRUCTION-MATH.md`; nineteen CPU properties now cover the affine inverse
  basis, Sannikov projected distance, constant-translation shell equivalence, camera-inside
  successor semantics and direction sign, inverse-transpose normals, grazing mixed-chart
  failure, different-triangle failure, angular owner-closure failure, and exact bounded
  alternatives. A fixed 1.176 m value is only this asset's current top height, never a generic
  ground-cover height contract.
- The current single-species hardware shell is temporary exact-O acceptance scaffolding. It is
  depth-only, and root density rather than shell-origin density owns visible coverage. It must
  not survive as a fixed-height generic-groundcover architecture or appear as a visible
  floating terrain copy. User boundary-flight review confirmed that its side plane is visible;
  the duplicate translated mesh is therefore an active defect, not accepted scaffolding.
- The exact shell-free replacement is proved: intersect the unchanged carrier terrain from the
  camera translated by the negative envelope vector. The hit parameter, primitive owner,
  barycentrics, clipping, silhouette, and projected depth are identical to the translated
  surface. The result must carry the shifted terrain owner/chart; ordinary scene depth and
  ordinary ground ownership are not substitutes.
- The two remaining projection classes are solved as math contracts. Camera-inside visibility
  is the next ordered surface event after the camera phase and retains the forward pixel ray;
  deriving direction from a behind-camera top point reverses it. Distant top/side drift cannot
  contain a distance term under an exact affine chart. The current 16x4 direction carrier
  filters arbitrary-mesh first hits and clamps elevations below 15 degrees, neither of which is
  exact. Exterior rendering needs a certified four-dimensional first-owner closure; inside
  rendering needs a sparse five-dimensional successor-owner closure. Both use fixed candidate
  counts, exact live triangle tests, and a hard offline memory/rejection bound—never a dense
  unbounded 5D atlas, runtime march, or data-dependent loop.
- The no-memory-blowup constraint is quantified against the accepted asset. Its canonical
  4,260,096-texel owner grid costs 17.0 MB for one 32-bit owner per sample and references
  445,227 distinct triangles. A dense four-owner closure would cost 68.2 MB for ids alone and
  is explicitly rejected. The math contract instead uses a bounded indirection field plus
  sparse fixed-width owner/event pages and compact referenced geometry, with hard bake gates
  for candidate count, subdivision depth, page count, geometry bytes, and total bytes.
- A five-degree canonical elevation row is approved for the grazing domain. It removes the
  current direct 5-to-15-degree clamp and serves as an additional closure seed; it is not used
  to justify scalar interpolation between arbitrary-mesh owners.
- GCRP/v4 vertex RGB is now transcoded once at load into a filterable premultiplied RGBA8
  first-hit colour atlas. The compute shader samples four cache-local colour records only
  after the winning hit, packs full RGB888 plus the four-bit exact profile id, and resolve uses
  that authored colour rather than the functional green palette. Default acceptance no longer
  uploads the much larger owner, vertex, and triangle GPU tables; `?grassowner=1` retains them
  only for its explicit diagnostic.
- The exact acceptance URL booted cleanly through frame 113 at DPR2 with no page, TSL, WebGPU,
  binding, pipeline, or command-buffer diagnostic. The captured view visibly contains the
  authored pale pink-brown panicles above green culms/leaves. The added shader work is one
  texture binding and four winner-only taps; dispatch shape, barriers, synchronization, and
  the fixed query count are unchanged. The casual HUD readout is not a performance result; a
  fresh game-only GPU trace is still required after visual acceptance.

O(1) remains intact: no per-frame ray march, WGSL loop, distance-dependent iteration, or new
screen-sized colour target was introduced.

## 14. Shell-free exterior reconstruction checkpoint (2026-07-21)

- The scene-visible translated terrain shell has been removed from the implementation. The
  query now fetches unchanged terrain vertices and applies `VP * vec4(0,H,0,0)` only to their
  homogeneous projection. Algebraically this is the proved camera-translation identity; no
  world vertex is raised by `H`, and the query renders only into an offscreen target.
- Heightfield visibility conservatively includes both the ordinary and shifted-camera frusta.
  Ordinary-view HZB rejection is disabled only for those heightfield clusters because it cannot
  prove shifted-view occlusion. Ordinary camera distance still owns DAG LOD, while objects and
  every non-heightfield decision retain the existing cull path.
- Native depth election and an `rg16f` octahedral triangle chart share one render target, so the
  queried depth and local terrain derivative come from the same winning terrain fragment. The
  old mixed state—translated entry depth combined with a separately filtered guide slope—is no
  longer used by the isolated Calamagrostis profile.
- The ordinary hardware pass reuses this colour target with colour/depth writes disabled; only
  a full-resolution `depth32float` attachment is newly resident. The query adds one conservative
  terrain draw and fixed per-fragment derivative/octahedral ALU. It adds no bind group, compute
  workgroup, barrier, ray step, data-dependent branch count, or shader loop. A fresh game-only
  GPU trace remains required after visual acceptance.
- `npm run typecheck` passes, all 23 shader-independent reconstruction properties pass, and the
  exact DPR2 localhost acceptance URL passed real WebGPU Chromium through frame 167 with no
  page, TSL, shader-validation, binding, pipeline, command-buffer, or uncaptured WebGPU error.
- User flight review confirmed that the translated-mesh removal is regression-free and top-down
  views remain strong. It also rejected the remaining low-oblique result: distant plants collapse
  into striped relief, and a world-cover boundary still presents as a plane with only rare hints
  of lower plant structure.
- The pure-math comparison identifies two independent causes. Reusing the 15-degree projected
  path at a five-degree live elevation compresses all source height below `H` by
  `tan(5 deg)/tan(15 deg)=0.3265`; a true ground hit is reconstructed about `0.792 m` above ground,
  and the limit toward horizontal is literally the top plane. Separately, querying the fully
  populated periodic first hit and then rejecting a bare world root is not the first eligible
  hit: it loses later covered roots and can reject the nearer anti-layer before considering a
  valid farther layer.
- This checkpoint therefore does **not** claim the finite-profile reconstruction is accepted.
  The exact next carrier is the certified predicate-filtered owner/successor closure: live
  ray/triangle intersection and barycentric colour/normal, fixed statically expanded candidate
  count, finite horizon, sparse pages, and hard subdivision/page/geometry/total-byte gates. The
  five-degree row is only a closure seed. Camera-inside adds signed origin phase to the same
  successor relation. The current v4 owner-token range cannot represent the new row or closure.
