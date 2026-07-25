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
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-RAY-RECONSTRUCTION-MATH.md`; nineteen CPU properties now cover the affine inverse
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
- An actual-asset lower-bound probe now covers all 3,145,728 canonical 4D cells. Corner-owner
  union is median 6, p95 12, p99 15, maximum 16; 1,908,893 cells already exceed four. This is
  before live-only interior owners, the five-degree domain, or predicate-filtered successors.
  The result rules out the current four-owner diagnostic and further rules out treating a blind
  dense `K` array as the memory-safe answer; successor-aware adaptive sparse certification is
  the active feasibility track.

## 15. Runtime-closure rejection and cheap projective reset (2026-07-21)

- The fixed-candidate owner/successor runtime proposed at the end of §14 is rejected. Although
  bounded and loop-free, it replaces a tiny precomputed lookup with many live triangle tests and
  is therefore a brute-force response to an over-strengthened mathematical requirement. No
  runtime or shader implementation of that proposal was made.
- The conceptual mistake was requiring continuum-exact reproduction of the arbitrary source
  mesh under arbitrary root deletion and camera origin. Sannikov's mesh-baked result is a direct,
  discretised finite-mesh ray impostor. It is exact at baked rays and approximate between them;
  the published low/mid-range architecture does not recover source topology at runtime.
- The active derivation is `GRASS-CHEAP-PROJECTIVE-RAY-MATH.md`. A finite-mesh row must retain its
  full view-conditioned distance. The current loader instead converts it to horizontal projected
  path and the runtime relifts it with another elevation—an identity which is exact only for a
  parallel extrusion. This is the direct mathematical source of the grazing top-sheet collapse.
- The minimum corrected carrier is one complete vertical-drop record selected from a direction
  domain which includes the grazing guard and vertical centre. A stronger one-read record stores
  the selected geometric plane covector; the live ray then intersects that plane with one dot
  denominator and one division. Three-read reciprocal-slope interpolation is an optional bounded
  refinement and is exact on a shared plane without source triangles.
- The CPU math harness now contains 27 passing properties, including one-record plane
  reconstruction, reciprocal depth in Cartesian slope space, and perspective-correct affine
  attributes. TypeScript type checking and whitespace checks pass.
- The first actual-asset surrogate comparison is recorded in
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-RAY-SURROGATE-EVALUATION.md`. Across 53,248 noncanonical live rays, none of the current
  projected relift, nearest canonical distance, nearest vertical drop, one-face plane, or
  three-read reciprocal-slope forms is acceptable on the 16-azimuth support. The one-face plane
  path is rejected: every accepted extrapolation left its microscopic source triangle.
  Phase-only median error is millimetric for categorical vertical drop, while the 11.25-degree
  half-bin azimuth offset produces metre-scale median grazing error; directional support is the
  dominant current discretisation failure.
- One consolidated second fixed-read experiment is active before this sampled-field track is
  either selected or parked: exact phase-reprojected two-read lookup and denser azimuth support
  under a fixed atlas-byte budget. It still contains no live triangle, candidate list, march, or
  data-dependent loop.
- The abandoned closure census/format files are preserved as research evidence only. Resume that
  path solely if the user explicitly approves measured live candidate cost; absence of an
  explicit prohibition is never permission to brute-force.
- The second sampled-field attempt is also complete and parked. One/two-read phase reprojection
  and byte-matched 32/64-azimuth layouts retain roughly metre-scale median and multi-metre p95
  disagreement against the unchanged mesh. Another local interpolation, predictor iteration, or
  plane heuristic is not authorized by that result.
- A direct Pluecker line-space classifier is mathematically exact but fails the low/mid-range
  gate on the accepted asset. One current-scale 4D cell contains at least 202 first owners plus
  miss on a sparse `5^4` sample, requiring at least eight serial binary decisions before
  separator geometry. After one 4D subdivision every subcell still exceeds a four-decision tail.
  The hierarchy/page path is parked; it must not be disguised as constant-time cheap work.
- A tiny union of exact affine/projective mask sweeps is likewise rejected for near-field
  Calamagrostis. The source has 270,535 independently posed primitives; four constructor laws do
  not turn them into four shared sweeps. Direct procedural provenance remains useful to an
  offline baker, but not as a runtime family decomposition.
- The primary-source author comments now establish the intended later-mesh approximation: the
  3D texture deliberately omits elevation, bakes the mesh under a fixed-angle assumption, and
  compensates that angle as if the grass were vertical. The published flattening/avoid-the-pole
  workaround is not an acceptable free-camera contract here.
- The surviving cheap theorem is an affine conjugacy of one canonical first-hit field. Relative
  to carrier plane `n*x=k`, `T(x)=x+(d/(n*d)-c/(n*c))*(n*x-k)` maps every canonical ray to one live
  ray, preserves first-owner order for the coherently deformed complete source mesh, and gives an
  exact inverse-transpose normal update. It remains one record plus tiny fixed ALU and does not
  require the source mesh to be an extrusion.
- That theorem is exact for the view-dependent deformed mesh, not the unchanged mesh, and has a
  real carrier-tangent pole. The active math-only gate is a finite complementary chart/carrier
  cover which handles every promised camera direction, selects exactly one chart directly,
  produces no visible raised terrain copy, and has subpixel chart-boundary discrepancy. No
  runtime or shader implementation begins before this is proved or rejected.
- A tiny complementary all-view cover is now rejected for the generic asymmetric mesh. Three
  carrier normals are the algebraic minimum to avoid a hemisphere pole, but their independently
  deformed images cannot agree on overlap boundaries without symmetry the Calamagrostis does not
  have. A projected-distance/height-distance switch on one carrier merely chooses rank collapse
  versus unbounded stretch at the horizon.
- Perspective adds one further correction: the per-ray inverse-transpose is not the Jacobian of
  the stitched pinhole-camera warp because the live direction varies across carrier points. The
  exact fixed-cost Jacobian is recorded in `GRASS-CHEAP-PROJECTIVE-RAY-MATH.md` §12.1. A single
  fixed-elevation/narrow-cone contract is now rejected: it would merely rename Sannikov's old
  flatten-and-avoid workaround, which cannot govern a free camera or the accepted tall plant.
- Sannikov's 2024 follow-up comments supply the active cheap correction. His arbitrary-model
  experiment used 16 directions, moved the depth-zero plane from the top to the approximate
  middle so ray scale changed from `[0,1]` to `[-0.5,0.5]`, and states that depth must use a
  PCF-like interpolation rather than linear blending. His grass experiment used a
  `128x128x4` light field for a repeating 25 cm tile and explicitly targeted accurate
  interpolation from very few directions; he rejected the suggestion that the method is only
  for top-down rather than first-person views.
- The active sampled-record correction is centred ray phase plus one categorical complete-record
  sample. Because the current GCRP is physically indexed at the top plane, its selected canonical
  query is `q_c=q+(d_xz/d_y-c_xz/c_y)*(y0-H)`; sampling at the middle crossing directly would be
  wrong unless the slices are recentered once during bake/load. Depth, coverage, normal, colour,
  and owner follow one record and unrelated owners are never numerically averaged. Its normal is
  transported by `m_d~m_c-e_y*((d/d_y-c/c_y)*m_c)`. This reduces atlas traffic and register live
  ranges; it adds no binding, pass, barrier, loop, march, or candidate list.
- The fixed `1.176 m` translated-terrain query has been completely removed: its target, indirect
  args, shifted projection, duplicate terrain draw, cull expansion, and runtime call no longer
  exist. Per-profile `H`, minimax `y0=(ymin+ymax)/2`, and selected hit
  height are reconstructed in one coherent local chart from the real carrier datum:
  `tB=T+(yh-hT)/(J^-1*d)_h`. This permits distinct overlapping botanical bands without a raised
  terrain copy or per-profile shell. The exact no-carrier horizontal case remains open and may
  not be hidden with a clamp or a restored envelope.
- The exact horizontal carrier case remains explicit work. A 5-degree view may reduce the
  low-oblique mismatch, but no camera angle may be hidden, flattened away, or declared outside
  the visual contract.
- The isolated acceptance path now recentres every baked slice once at load, stores each complete
  depth/normal/coverage event in one exact 8-byte `rg32uint` record (two packed 16-bit fields per
  channel), selects one categorical angular footprint corner, and uses the identical address for
  winner colour. It remains one `textureLoad` per stage with no extra binding, pass, barrier,
  march, runtime loop, candidate list, or atlas-byte increase. The shader-independent reference
  has 35 passing properties and the focused packed-recentre test passes.
- The exact DPR2 localhost acceptance URL passed real WebGPU Chromium through frame 124 after 60
  settled frames with no page, TSL, shader-validation, binding, pipeline, command-buffer, or
  uncaptured WebGPU error. Visual acceptance of the low-oblique result remains pending.

## 16. Centred-categorical visual rejection and active defects (2026-07-22)

The first centred-reference/categorical runtime checkpoint is **not accepted**. User flight review
on the exact isolated `?grassprofile=2` view found some added side detail, but reported all of the
following current defects:

- thin elongated streaks run through the plants, especially in oblique views;
- distant/side detail remains fuzzy enough that individual geometry is difficult to judge;
- on a slight slope, a height/deformation transition appears at a camera-relative distance and
  follows the camera while walking, with adjacent regions reading slightly lower/higher and some
  poses visibly warping;
- the pale violet/pink fuzzy panicles disappear beyond a camera-relative distance while green stems
  remain, then reappear when the camera approaches;
- a subtle temporal shimmer resembling millions of tiny particles moving over the grass is a
  separate **legacy** defect. It predates both this O(1) reconstruction and the present rework and
  must not be used to explain the new streaks or camera-relative band.

The authored population density is **not a bug**. A sparser isolated view was requested only as a
temporary, world-stable inspection aid so one plant silhouette can be judged. It may not change the
accepted source morphology, production density law, ray reconstruction, or multi-profile work.

The first confirmed mathematical cause is the current angular PCF approximation. At roughly
`0.52/256 = 2.03 mm` spatial intervals it hashes the centred phase texel and independently selects
one of the two azimuth and two elevation rows. Each row is a coherent complete first-hit field, but
the runtime result splices different view-deformed fields at every 2 mm phase cell. If two selected
records disagree in height by `Delta y`, their grazing displacement is
`|Delta x| = |Delta y| cot(alpha)`; only `0.20 m` disagreement becomes `2.29 m` at five degrees.
Hit/miss disagreement becomes broken streaks, and the selected normal/colour reproduce the same
fuzz. Because centred phase itself moves with `cot(alpha)`, the mosaic is also highly view-sensitive.

This rules out spatially hashed angular selection as the accepted one-read filter. With one record
and no connectivity/owner closure, angular choice must either be phase-independent (one coherent
canonical field with an angular switch) or it splices fields. Coherent nearest/MAP angle is the
lowest-cost diagnostic and deletes the hash ALU, but its angular Voronoi boundary can still form a
camera-relative ring/pop and therefore is not presumed to be the final solution. A literal
four-record categorical weighted median is the principled fixed-footprint PCF fallback, but its
extra reads/registers require an explicit performance decision and it still cannot manufacture
continuous owner correspondence. The active task remains a qualitatively coherent cheap angular
representation/filter, not another stochastic dither, arithmetic depth blend, live triangle test,
candidate search, loop, or march.

The carrier/distance audit is still active. In addition to the angular-row mechanism above, current
code retains an old camera-radius density fade over `0.9R..R` (`139.5..155 m`) despite the stated
camera-independent physical-density contract. That fade is a separate known camera-locked term; it
must be removed or proved irrelevant rather than confused with the nearer angular band.

## 17. Orthogonal-carrier runtime rejection (2026-07-22)

The selected-direction orthogonal-carrier runtime attempt is **rejected** and is not a visual
checkpoint. User review found:

- obvious triangular/fan regions in oblique view, widening away from their camera-front apex;
- increasing stretch with viewing distance;
- a far, high-oblique result whose repeated botanical detail reads as the wrong plant/view being
  stretched rather than as the Calamagrostis geometry;
- the field remains patterned and view-quantized even where individual coloured panicles survive.

The exact angular cause is now proved. Phase-independent nearest-direction selection partitions the
camera frustum into angular Voronoi cones. Their world width grows as `r * DeltaTheta`, so disagreement
between adjacent baked first-hit fields becomes the observed widening wedges. This is the coarse form
of the prior 2.03 mm hashed splice, not its cure. Nearest/MAP one-record angular selection is rejected.

The attempted orthogonal carrier also moved its reference point with each query datum. That violates
oriented-line invariance: shifting the query origin along the same ray changes the atlas address in
general, so the per-ray maps cannot be the projection of one coherent surface. The fixed-carrier
normal formula is consequently not the Jacobian of that moving construction.

Work has returned to math only. No further shader/render/WebGPU change is permitted until the next
representation proves oriented-line invariance, pinhole integrability, continuous angular
reconstruction, bounded grazing coordinates, and the unchanged O(1)/no-loop/no-march/no-candidate/
no-memory-blow-up cost contract. The failed code is preserved for diagnosis and has not been promoted
or committed as an accepted result.

## 18. Direct-field codec feasibility (2026-07-22)

- Paper-grade prior-art and experiment provenance is now centralised in
  `docs/deep-research/grass/GROUNDCOVER-LINE-FIELD-SOURCE-AND-EXPERIMENT-LEDGER.md`.
  It records the exact borrowed Sannikov/VDM/GDM claims separately from LAAS
  derivations, source URLs/local files, immutable Calamagrostis hash, commands,
  output hashes, measured results, and rejection boundaries.
- A first shader-independent actual-asset codec gate rejects ordinary linear/
  SVD reconstruction. Even though the test omits colour, normal, material mark,
  continuous angular holdout, inside-origin phase, and deep events, rank 4 has
  only 74.25% hit agreement with 0.257/0.536 m p50/p95 hit-height error; rank 32
  still needs at least eight RGBA spatial reads for 91.87% agreement and
  0.104/0.368 m error; rank 48 needs twelve reads for 96.83% agreement. Only
  fitting all 64 existing directions is exact on those same samples.
- The report is
  `data/work/groundcover-line-field/2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c/qa/001-linear-rank-gate.json`
  (SHA-256 `138dcaf40119452d6ade0283ac63028585edb22aa66e785a55e4bdc4f73d1464`).
- Linear/tensor depth averaging is therefore not an active runtime candidate.
  The remaining measured tracks are a categorical complete-record codebook,
  the actual deep successor count over declared finite horizons, and the direct
  categorical angular-density/byte bound. None changes the shader until it
  proves fixed one/few-read cost and categorical geometric accuracy.

## 19. Event-measure codec rejection (2026-07-22)

- A shader-independent block/VQ audit now evaluates an honest band-limited
  record on the immutable Calamagrostis v4 source: coverage plus premultiplied
  first and second ray-depth moments, authored colour, and normal moment. Its
  single-event depth is explicitly a moment-matched representative with a
  recorded unresolved sigma, never an exact triangle/owner claim.
- The strongest fixed-filterable layout is a half-density, ray-aligned index-two
  periodic lattice with three direct reads. It would occupy `43,467,840` bytes
  at the corrected 81 directions versus the current `51,121,152`-byte
  geometry-plus-colour baseline, so it clears the byte-funding gate.
- It fails the geometric gate: p95 representative ray-depth error is `0.434 m`,
  unresolved ray-depth sigma is `0.722 m`, representative height error is
  `0.173 m`, and representative-normal error is `53.42 degrees`. Coverage IoU
  is `0.9458`; the plausible static colour image does not erase those tails.
- A 51,204-codeword VQ is smaller, but its integer index cannot be filtered.
  Nearest decoding still has `0.343 m` p95 depth error, `0.363 m` p95 unresolved
  sigma, and `62.9 degrees` p95 normal error; bilinear decoding would cost eight
  dependent reads. Nonlinear conditional and square block codecs are worse.
- The track is therefore rejected for crisp near-field geometry and no runtime
  shader change follows. Tool, full metrics, source/recipe/script hashes, and
  four numbered PNG comparisons are preserved under
  `data/work/groundcover-event-measure-codec/1856ac68e85012ed36538c593c1c02e6b0023ea02281b0f5b5d7582126d6208c/`.
  A radiance-like distant use may be reconsidered separately only if it cannot
  leak into the accepted geometric range.

## 20. Literal PCF depth-filter rejection (2026-07-22)

- The actual Calamagrostis mesh was evaluated with two concrete fixed-footprint
  readings of the unpublished "PCF-like" hint: a four-corner front depth
  quartile and a four-corner weighted median. No shader change was made.
- Both filters leave hit agreement/recall exactly at `65.68%/96.25%`; they use
  the same coverage support and therefore cannot repair the wrong line-space
  event. Their world-position p95/p99 errors worsen from the projected
  baseline's `4.137/7.060 m` to `4.769/7.588 m` and `4.643/7.542 m`.
- The four filtered reads (16 effective source texels), compare/select sorting
  network, and added live ranges have no accepted quality return. This path is
  rejected before runtime integration and may not be described as Sannikov's
  unpublished interpolation.
- Tool, command, immutable source hash, preserved report, output hash, and
  interpretation boundary are recorded as Experiment 006 in
  `docs/deep-research/grass/GROUNDCOVER-LINE-FIELD-SOURCE-AND-EXPERIMENT-LEDGER.md`.

## 21. Shell-free top-entry correction is necessary but visually insufficient (2026-07-22)

- After the physical raised carrier was removed, the isolated rigid-profile
  specialization still forced `dtE=0` and `tE=tScene`. That queried the centred
  field from the terrain/scene datum instead of the unique algebraic botanical
  top entry and violated the exterior line-coordinate contract.
- The specialization now uses the already-computed
  `dtE=(qTop-qO)/kGround` and `tE=tScene+dtE`. This restores the proved
  line-origin-invariant exterior entry without rendering or translating any
  plane. It adds no texture read, binding, buffer, memory, pass, barrier,
  dispatch, loop, march, candidate, or runtime branch; only the existing
  divide/add survive dead-code elimination in this specialization.
- TypeScript type checking passes and all 37 pure ray-math tests pass. Two
  unrelated/pre-existing loader tests currently fail in the combined loader
  suite (`filterable half-float` fixture expectation and real Sphagnum payload
  expectation); this one-line shader correction does not touch that loader.
- User review on the live exact isolated URL reports **no visual improvement**:
  the stretching and wrong perspective remain. Therefore this correction is
  retained as necessary coordinate hygiene but is explicitly rejected as a
  visual solution. The dominant defect remains the angular/visibility
  representation, not the exterior datum.

## 22. Fixed-K successor-record rejection (2026-07-22)

- Effort spent: one actual-mesh offline gate over `3,232` oriented lines and
  `519,359` exact surface events, including exact horizontal, signed sub-degree
  directions, exterior first hits, and camera-inside successors.
- Reusable result: the executable, exact cost table, complete metrics, hashes,
  and proof that an omitted categorical event owns a camera-origin interval on
  which a K-atom CDF must return the wrong successor. The full report is
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-FIXED-K-EVENT-RECORD-FEASIBILITY.md`; provenance
  is Experiment 007 in the central grass source ledger.
- Exact blocker: K=8 retains at most `1.924%` of sampled event topology. Its
  favourable endpoint record has only `56.363%` camera-height exact-event
  agreement with `0.401 m` p95 world-height error, while a complete 81-direction
  atlas costs `345,067,776` bytes. At `0.1` degrees exact-event agreement falls
  to `29.05%` downward and `2.87%` upward.
- Why focus moved: K=2 is the only complete record that fits the resident cap,
  and only after an optimistic half-phase lattice; its geometry fails. K=4/8
  exceed memory before spatial/angular interpolation and multi-species overlap.
- Active fallback: continue the line-space mathematical representation search;
  do not integrate this codec and do not replace it with runtime traversal,
  marching, loops, larger candidate lists, extra passes, or distance-dependent
  work.
- Objective resume condition: a representation must encode omitted-event
  successors without one atom per event, pass exact-horizontal and signed
  sub-degree actual-mesh tests, carry winning colour/normal/species attributes,
  and prove `<=51,121,152` resident bytes plus `<=4` fixed loads before shader
  work resumes.

## 23. Independent-marginal transfer field rejection (2026-07-22)

- User review reconfirmed that the live isolated profile still has the dominant
  widening/stretch/wrong-perspective defect. The shell-free top-entry correction
  in Section 21 therefore remains coordinate hygiene only; it did not improve
  the accepted image criterion.
- The proposed footprint-transfer codec was stopped **before fitting or runtime
  integration**. Its first truth artifact stored per-beam coverage plus three
  independently depth-sorted conditional moments. Those marginals discard
  micro-ray identity, marked botanical owner, and correspondence between camera
  frames. Two different connected image fields can have identical records, so
  a decoder could pass every aggregate error threshold while reproducing the
  exact camera-centred fans, distance stretch, wrong-view sheets, and shimmer.
- The required mathematical object is a coupled marked transport
  `G(ell, eta, xi)`: fixed oriented line `ell`, physical successor phase `eta`,
  and a frame-invariant sample/event label `xi`. Moving a datum along the same
  exterior line must preserve the decoded world event; moving a physical camera
  inside the cover may change it only after crossing that event, and then only
  to a later marked event. This coupling is now a hard prerequisite, not an
  optional diagnostic.
- The rejected trainer was also structurally unidentified: `23,744` training
  beams could touch at most `189,952` of `2,359,296` proposed spatial texels,
  leaving at least `91.949%` at random initialisation. Its blindly averaged mips
  did not represent larger-footprint truth, and its clamped octahedral angular
  square did not identify spherical quotient edges. These issues independently
  forbid treating a rank result as evidence.
- The preserved adversarial report is
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-LINE-INVARIANT-TRANSFER-FIELD-AUDIT.md`. The expensive actual-mesh beam
  tensors remain useful as marginal/radiance evidence, but they are not a crisp
  geometry acceptance truth and will not be promoted to the shader.
- Active parallel math tracks are now: (1) a directly coupled, chart-seam-safe
  marked line transport with explicit same-line/pinhole sequences and a hard
  byte/read gate; and (2) a procedural botanical-chart feasibility audit which
  tests whether stable blade/culm/panicle provenance can supply one coherent
  fixed-cost chart without live triangle candidates. Neither authorizes a
  shader change, loop, march, traversal, candidate list, shell, or memory-cap
  increase.

## 24. Six-chart direct categorical line-atlas rejection (2026-07-22)

- The strongest dense exact exterior alternative was evaluated in the correct
  origin-invariant coordinates: six signed dominant-axis charts addressed by
  two plane intersections, equivalently `(Q_a,Q_b,r_a,r_b)` with both slopes
  bounded by one. It stores one nearest complete marked event; it has no
  camera-centred angle address and does not create an unbounded camera-range
  wedge merely by choosing a different datum on the same line.
- The actual-source budget uses Calamagrostis GCRP/v4 SHA-256
  `2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c`,
  its `0.51999998 m` square tile, `1.175222625 m` botanical height, and the
  declared `155 m` grazing horizon. It optimistically adapts horizontal slope
  density to the exact in-slab/finite-horizon axis travel and charges no
  integer-grid waste, gutter, mip, palette, metadata, or categorical-boundary
  cost.
- Under the entire `51,121,152`-byte cap, a strict four-byte geometry-only
  record permits only a `10.205 cm` per-coordinate / `14.432 cm` transverse
  line bound. A minimum plausible six-byte event permits only
  `11.294 / 15.972 cm` before its palette. Four bytes cannot simultaneously
  carry line depth, normal, authored appearance, and botanical mark.
- Those are only geometric line-perturbation bounds. At thin occlusion
  boundaries, nearest categorical event depth and owner error remain unbounded;
  the direct atlas also does not encode the genuinely five-dimensional
  camera-inside successor. It is therefore decisively rejected rather than
  fitted or integrated.
- Tool:
  `tools/groundcover-bake/analyze-oriented-line-atlas-budget.ts`, SHA-256
  `66c599ab747b8f30dc9ce239a33c07c5331953f41f3b8602956d07594bf6eecc`.
  Report:
  `data/work/groundcover-oriented-line-atlas-budget/2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c/43ef4e75634df73b5fc83a904b7588f09384fad5d264f2110af81d89db7f6f65/report.json`,
  SHA-256
  `4884b151a9281d60c346385cebebec0cb9e4cd7e3cf37d95d24c845866863d71`.
  The source/recipe/tool/result and explicit rejection are Experiment 010 in
  the central grass source ledger.
- Combined with the actual-mesh adaptive atlas lower bound (Experiment 008),
  this leaves **no currently validated coupled representation to integrate**.
  Resume only for a new analytic event-sharing mechanism with a complete
  `<=4`-read, `<=256`-FMA, `<=51,121,152`-byte proof. No runtime file was
  changed by this experiment.

## 25. Procedural botanical-chart carrier rejection (2026-07-22)

- Effort spent: one deterministic actual-source feasibility pass that rebuilt
  the accepted Calamagrostis generator, verified zero packed-index mismatches,
  partitioned all `2,171,134` triangles by generator semantics/topology, and
  audited angular disocclusion, same-line successors, exact chart evaluation,
  current metadata, and resident cost. No runtime or shader file was changed.
- Reusable result: `270,535` real primitive-instance charts, all contiguous in
  the source triangle stream, with exact family counts for foliage, culms,
  rhizomes, panicle axes, spikelet surfaces, callus hairs/filaments, and
  anthers. This is useful offline provenance and is not triangle-ID relabelling.
  The full report is `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-PROCEDURAL-CHART-FEASIBILITY.md`; source provenance
  and immutable hashes are Experiment 011 in the central grass ledger.
- Exact blocker: with a proposed 81-direction field, the exact live winner
  chart appears among four surrounding baked chart tokens for only
  `0.837320574%` of truth hits; the exact local triangle appears for `0%`.
  Therefore chart-local coordinates or an intra-cell separator cannot recover
  the disoccluded live surface. Along one oriented camera path, charts/line are
  p95 `481`, chart runs are p95 `723`, and periodic horizontal runs reach
  `4,182`; chart identity does not encode inside-origin/copy successor phase.
- Cost blocker: eight bytes per atlas token at 81 directions consumes
  `43,133,472` bytes and leaves only `29.5255` bytes/chart. A 32-byte descriptor
  already exceeds the resident cap, while the exact procedural ribbons/tubes
  are nonlinear or piecewise and require a local-panel/root choice forbidden by
  the fixed-work contract.
- Why focus moved: the partition answered the provenance question positively
  but failed the visibility question decisively. Adding fixed chart candidates,
  analytic per-chart root searches, or per-species fields would turn the failed
  representation into forbidden brute force.
- Preserved bounded recook: if a future representation can use provenance
  correctly, instrument the four procedural append functions, store chart ID in
  the currently zero fourth triangle `u32`, append a versioned recipe/local
  panel table, and carry community/species/population/root/periodic-copy marks.
  This recook is deliberately not performed after the gate failure.
- Active fallback: seek a new coupled analytic event-sharing representation;
  offline-union overlapping species and moss into one marked community truth.
  Do not query one field per species and do not introduce runtime traversal,
  marching, loops, candidates, extra passes, distance-dependent work, or a
  larger memory cap.
- Objective resume condition: a new proof must recover absent disocclusions and
  same-line marked successors, evaluate one selected primitive in fixed closed
  form, and remain within `<=51,121,152` bytes and one/few fixed reads before
  this chart metadata path is reopened.

## 26. One-read 32-bit exterior light-field fallback rejection (2026-07-22)

- Effort spent: one bounded actual-asset gate with two concrete cap-filling
  layouts, 9,216 structured exterior camera rays per layout, exact packed
  record simulation, fixed-plane line-invariance tests, coherent fan
  measurement, panicle class checks, millimetric camera translations, and two
  content-addressed QA sheets. No shader or runtime file was changed.
- Reusable representation: one categorical `u32` holds 1-bit hit/coverage,
  11-bit vertical drop, 5+5-bit oct normal, and a 10-bit direct RGB343
  material/colour class. Its fixed exterior path is two botanical Y-plane
  intersections, periodic phase/slope quantisation, one nearest texture read,
  decode, and live-ray height relift. No arithmetic record blend, owner-table
  read, loop, march, traversal, candidate, species query, or extra pass exists.
- Concrete allocations: phase-rich P255/D193 uses `50,989,828` bytes; angle-rich
  P220/D257 uses `50,663,952` bytes, both including wrapped gutters. Phase
  texels remain `2.04/2.36 mm`; direction rings allocate most sectors to large
  grazing slopes.
- Positive invariant: 3,072 same-line datum shifts per allocation produced zero
  address mismatches and at most `1.07e-14 m` fixed-plane intersection drift.
  Therefore the failure is not camera-centred carrier deformation.
- Exact blocker: aggregate world-position p50/p95 is `1.557/7.989 m` and
  `1.605/9.535 m`. At the supported 5.5-degree sweep it remains
  `1.597/7.808 m` and `1.176/7.482 m`, even though hit/miss agreement is about
  99%; the predicted event belongs to a different view line.
- Coherent-fan blocker: connected same-angular-bin regions with at least 1 m
  position error are p50/max `6.377/12.338 m` and `5.396/11.514 m` wide at 5.5
  degrees. At 10 degrees they remain `3.255/6.120 m` and `2.074/6.322 m`.
  This explicitly reproduces coherent wrong-view/stretch sectors rather than
  only lowering aggregate pixel accuracy.
- Appearance blocker: panicle-vs-vegetative retention is only `59.82/59.34%`,
  exact panicle family about 18%, and `39.23/36.24%` of stable-truth semantic
  comparisons acquire an unforced predicted family change under 1--4.5 mm
  camera translations.
- Why focus moved: doubling/tripling the direction count within the full
  resident cap changes the sector partition but cannot supply disoccluded
  first-event identity. It is not a visually plausible exterior fallback.
- Camera boundary: this top-entry field still does not encode pointed-line
  inside successors. Exact horizontal lines have no finite intersection with
  the two Y planes. Neither limitation is hidden or declared solved.
- Full report and QA:
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-PACKED-LIGHT-FIELD-FALLBACK-GATE.md`; immutable provenance is
  Experiment 012 in the central grass ledger; artifact root is
  `data/work/groundcover-packed-light-field/2ed57f59d86e8376/a3ce1c6b881bcc30/`.
- Decision: reject and park both allocations. Resume only for a qualitatively
  new fixed-cost event-sharing representation that removes connected
  wrong-view sectors and supplies camera-inside/horizontal semantics. Do not
  add live reads, candidates, loops, marches, per-species fields, passes, or
  memory.

## 27. Coupled band-limited three-sheet rejection (2026-07-22)

- One final compact coupled visibility candidate was taken through a bounded
  actual-Calamagrostis gate and stopped without touching the runtime. Nine
  persistent exact micro-rays were reduced to three scalar-depth quantile
  sheets, but every stored atom remained one actual authored event with its
  colour/normal/material/botanical mark; unrelated owners were never averaged.
- The complete proposed 64-bit layout fits: 71 directions over an independently
  retraced `258..1` footprint pyramid cost `50,511,104` bytes and leave
  `610,048` bytes under the cap. Live work was bounded at three reads and less
  than 192 FMAs with no loop, march, traversal, candidate list, shell, extra
  pass, or per-species query. Cost was not the blocker.
- Exact transported-line right-censoring passed all 738 uncrossed sheet
  comparisons, with `6.55e-14 m` p95 world-event motion. This confirms that the
  labelled-micro-line construction itself is correct and reusable.
- Global camera coupling failed decisively. On fully populated `17x17`
  persistent-sensor camera/pixel patches, locally preferred sheet permutations
  have nontrivial loop holonomy on `645/963 = 66.978193%` of plaquettes. At 5
  degrees the two finest-footprint phases fail at `75.41%/83.98%`; four- and
  eight-times wider footprints remain nonintegrable at `46.77%/52.45%`.
  Conditional three-medoid depth `W1` is also `0.601579 m` p95 before packing.
- The panicle check is non-vacuous: 60 records contain 540 exact panicle hits,
  and the targeted records are 93.75% panicle by hit mass. Exact-atom medoids
  preserve sampled panicle mass, but that positive mark result cannot repair
  the failed geometric coupling.
- Interpretation: scalar depth rank can couple two one-dimensional marginals,
  but it is not a path-independent surface identity over a 2D camera/pixel
  patch. No global three-sheet decoder can satisfy all local continuations; it
  must tear, swap, or stretch somewhere. More rank/filtering/direction slices
  would spend cost without repairing that topological failure.
- Full report:
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-COUPLED-BANDLIMITED-EVENT-SHEET-FEASIBILITY.md`. Artifact:
  `data/work/groundcover-coupled-quantile-sheet-gate/2ed57f59d86e8376/b70facc20d553407/`.
  Source/tool/configuration/result/QA hashes and prior-art versus LAAS boundaries
  are Experiment 013 in the central source ledger.
- Decision: **reject and park after one attempt.** No runtime asset or shader
  implements it. Resume only for a path-independent global marked-event label
  or equivalent analytic visibility-sharing law that also handles arbitrary
  inside-origin successors inside the unchanged memory/read/ALU limits.
  Multi-species grass, other cover, and moss remain one offline-composed marked
  community truth, never one live query per species.

## 28. Fixed plenoptic cascade rejection (2026-07-22)

- A qualitatively separate fixed sampling cascade was taken through its single
  allowed actual-Calamagrostis attempt and stopped without any runtime or
  shader edit. Six levels trade phase resolution (`255²` down to `18²`) for
  angular support (16 up to 1,600 directions); wrapped gutters and optimistic
  eight-byte complete marked records total `47,187,808` bytes. Live work would
  have been exactly one selected record read.
- Fixed two-Y-plane line coordinates pass origin invariance, but the sampling
  bound does not: address displacement contains
  `sqrt(2) DeltaQ/2 + 2r sin(DeltaTheta/2)`. Pixel footprint and the angular
  term both scale with range, so coarsening phase at distance cannot buy away
  the required angular rate. The far level still has `8.705/13.564 m`
  p50/p95 `r DeltaTheta`.
- The exact-mesh 3x3 common-top directional-microbundle gate covers 2,688
  camera-centre pixels at
  `0.1/1/5/15/35/75/90` degrees and four millimetric camera translations.
  Surrounding rays pivot at the centre top-plane intersection, so this is an
  optimistic directional stress test rather than an exact pinhole-cone claim;
  the analytic angular bound and horizontal/inside failures are independent of
  that simplification.
  Before record quantisation, representative-event position error is
  `1.738/84.344 m` p50/p95, hit/miss disagreement is `21.763%`, panicle-mass
  p95 error is `0.778`, and stable truth suffers `38.727%` unforced predicted
  class changes.
- Connected >=25 cm wrong-view regions remain `24.955 m` p95 at 1 degree and
  `10.212 m` at 5 degrees. Near/top views also fail crispness: 75/90 degrees
  have `0.161/0.532 m` p95 event error. The candidate is rejected as a distant
  fallback too; it loses colour/coverage/panicle mass rather than merely
  filtering harmless detail.
- Exact horizontal is singular in the two-Y-plane chart and camera-inside
  successor phase is absent. Terrain depth from the representative event is
  wrong whenever the category names another event; a radiance-only variant
  would collapse back to a textured carrier plane. Offline union remains the
  only valid one-query route for overlapping species/moss.
- Full math, source boundary, hashes, metrics, and numbered QA are in
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-PLENOPTIC-SAMPLING-CASCADE-GATE.md`; Experiment 014 in the central
  source ledger records publication provenance. Artifact root:
  `data/work/groundcover-plenoptic-cascade/2ed57f59d86e8376/51129ab7d93cd865/`.
- Decision: reject and park after one actual-source attempt. Do not add level
  blends, reads, direction candidates, loops, marches, species queries, passes,
  or memory. Resume only for a path-independent marked event/successor law
  whose angular support is not a dense independent direction lattice.

## 29. Four affine botanical-slab rejection (2026-07-22)

- One qualitatively different fixed-cost geometry class was proved and tested
  without touching the runtime: four clipped affine images of arbitrarily
  detailed periodic 2D mask extrusions, with the upper two bands dedicated to
  the panicle-bearing height regime.
- The math is exact for that class. After inverse affine transport, finite slab
  clipping and `t=t_in+rho/||w_q||` remove live elevation continuously. Exact
  parallel-axis/boundary limits, closed caps, arbitrary inside-origin
  successors, inverse-transpose normals, and the periodic lattice action are
  all recorded in `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-FOUR-AFFINE-SLAB-EXTRUSION-FEASIBILITY.md`.
- A non-affine projective taper was rejected at the quotient proof, not hidden
  in the implementation. It conjugates canonical periods into
  position-dependent projective motions instead of Euclidean ground
  translations. Resetting the taper per tile destroys the one-query
  cross-copy first-hit property. Only affine lean/skew is exactly periodic.
- The complete fixed cost barely fits: four eight-byte records over four
  `220x220`, 24-direction, wrapped/mipped fields cost `50,466,816` bytes; four
  predetermined reads and less than 256 FMAs suffice. There is one texture
  array binding and no runtime loop, march, traversal, candidate tail, shell
  pass, distance-dependent work, or per-species multiplier.
- Geometry fails decisively on the accepted `2,171,134`-triangle source.
  Conservative `512x512` per-band masks produce silhouette IoU only
  `0.1718..0.2167`, overfill `78.07..81.83%`, and optimistic owner recall only
  `62.12..70.28%`. Panicle retention falls to `48.23%` at 0.1 degrees and
  `47.88%` at 1 degree. At 90 degrees, apparent `99.05%` retention is bought by
  artificial horizontal cap hits on `31.86%` of candidate pixels.
- The numbered QA images visibly show the failure: the crisp source plant on
  the left becomes filled axial planks on the right, with horizontal band
  sheets and lost upward branch/hair structure. More compact masks trade that
  overfill for still more source loss; more bands become a forbidden
  near-linear shell stack.
- Full report, tool, source, recipe, result, and PNG hashes are Experiment 015
  in the central grass source ledger. Artifact root:
  `data/work/groundcover-four-slab-extrusion/2ed57f59d86e8376/db4432b7634eb595/`.
- Decision: **reject and park after one attempt.** No runtime asset or shader
  implements it. Resume only for a new analytic event-sharing geometry that
  does not fill height intervals, proves Euclidean periodicity and camera-inside
  successors, and remains within the unchanged four-read/256-FMA/memory cap.
  Multi-species grass, moss, and other cover still require one offline-unioned
  marked community, never one live query per species.

## 30. Oracle angular event-flow rejection (2026-07-22)

- A genuinely different event-sharing mechanism was tested without changing
  the runtime or shader. The gate grants a hypothetical inverse-flow decoder
  each exact target world point, primitive, periodic copy, and direction for
  free, then asks whether that point is first-visible in any two/four
  neighbouring baked directions. Failure is therefore categorical
  disocclusion, not bad flow estimation, blended depth, or quantisation.
- The actual-source domain contains 700 structured `5x5` pinhole rays at
  `0.1/1/5/15/35/75/90` degrees, two azimuths and two phases, plus exact
  horizontal pointed-line successors from within-profile camera heights.
  Source spacings are `22.5/11.25/5/2/1/0.5` degrees; panicle and vegetative
  owners are measured separately.
- No spacing reaches the 95% contract. At 22.5 degrees, any-four exterior
  exact-event visibility is 87.74%, with panicles 88.97%, vegetative geometry
  85.31%, and camera-inside horizontal successors 62.5%. At 0.5 degrees,
  panicles reach 95.73%, but vegetative visibility is only 77.62% and
  horizontal-inside visibility is 0%. Oracle holes form connected regions up
  to 13 of 25 pixels, matching the coherent wrong-view patches seen live.
- Storage closes the other escape. A highly optimistic fixed two/three-read
  format charges only one 4-byte inverse-flow coordinate and complete 8-byte
  residual births, with topology metadata free. The only fitting spacing,
  22.5 degrees, fails visibility. At 11.25 degrees the lower bound is already
  `53,349,163` bytes, above the `51,121,152`-byte cap; finer grids explode.
- Full proof boundary, primary-source ancestry, metrics, tool/hash provenance,
  and numbered QA are in `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-ANGULAR-EVENT-FLOW-CEILING.md`; Experiment 016
  records it in the central source ledger. Artifact root:
  `data/work/groundcover-angular-event-flow/2ed57f59d86e8376/44420268b58db8a1/`.
- Decision: **reject and park after one attempt; do not build the codec.** Lin
  and Shum explains the neighbouring-ray/depth-assisted reconstruction, but
  even oracle correspondence cannot invent disoccluded authored events. Resume
  only for a path-independent visibility-sharing law which represents exact
  horizontal and arbitrary camera-inside successors without a dense angular
  complete-record lattice. Offline-unioned multi-species/moss remains the only
  one-query overlap contract.

## 31. Finite swept botanical-fibre rejection (2026-07-22)

- A non-slab authoring class was reduced to exact mathematics before any visual
  fitting or runtime work: up to four affine families of finite swept tubes,
  ribbons, stems, leaves, branchlets, hairs, and spikelets, each with detailed
  transverse masks and element-specific axial intervals.
- The exact side-event equation is
  `t_n=rho_n/||w_q||`, accepted only when
  `h_0+(w_h/||w_q||)rho_n in I_n`. Individual finite intervals avoid the
  filled-plank defect of the four slabs, but they make the first eligible 3D
  event an origin/elevation-conditioned successor rank rather than the first
  2D mask crossing.
- The fixed-rank theorem fails constructively. A horizontal ray at the height
  of a later strand rejects any number of earlier projected crossings whose
  intervals do not contain that height. Correctness therefore requires either
  the rejected common filled interval, an unbounded successor/candidate tail,
  or a five-dimensional field over transverse phase, direction, axial origin,
  and axial slope.
- Restoring even one coordinate with two coarse bins doubles the barely fitting
  four-family slab allocation to `100,933,632` bytes. Merely adding quantised
  interval endpoints to an optimistic complete record raises the same layout
  to at least `63,083,520` bytes and still cannot select the absent next event.
- Exact horizontal and camera-inside are the decisive cases, not hidden poles.
  The accepted source has 270,535 finite procedural charts, including 20,969
  panicle-axis tubes and 203,258 hairs/filaments; its actual line census reaches
  p95/max `1,077/4,481` events and `723/4,182` chart runs. Four constant affine
  generator directions also cannot retain the curved, golden-angle recursive
  plume without obvious direction quantisation or filled intervals.
- Full proof, immutable source hashes, prior-art boundary, cost, and resume
  condition are in `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-FINITE-SWEPT-FIBRE-FEASIBILITY.md`; Experiment 017 is
  in the central source ledger. No PNG was generated because the arbitrary
  horizontal/camera-inside hard contract fails before the class is visually
  viable. No runtime or shader file was changed.
- Decision: **reject and park at the math gate.** Do not add endpoint
  candidates, bands, shells, per-species fields, loops, traversal, or memory.
  Resume only for an analytic finite-endpoint class with a proved eligible
  successor rank at most four or an equivalent fixed-cost origin-aware selector.

## 32. Shell-free live reconstruction remains rejected (2026-07-22)

- The exact runtime URL was booted cleanly in real WebGPU Chromium after the
  fixed `1.176 m` carrier/query/pass had been removed completely. This rules
  out shader validation, bind-group failure, or the deleted duplicate surface
  as the explanation for the remaining image.
- Two shell-free mappings were checked on the same isolated
  `grassprofile=2` Calamagrostis asset. The centred categorical mapping still
  produced camera-front widening triangles, distance-growing stretch, and the
  wrong apparent view. The four-view coverage-unmixed projected-path baseline
  removed none of the underlying perspective error and reverted toward a
  striped/textured carpet at oblique range.
- User visual acceptance explicitly reports **no improvement** in the
  stretching or weird perspective. Both live mappings are rejected. A clean
  boot is supporting evidence only and is not visual acceptance.
- The failure is the line/event map, not a tunable colour, density, height,
  terrain, or post-process parameter: a neighbouring line's categorical first
  event is selected and then relifted onto the live ray. Relifting that wrong
  event turns angular disagreement into the observed widening sectors.
- Do not spend another runtime attempt on nearest-angle selection, arithmetic
  depth interpolation, local infinite-plane extrapolation, PCF depth order
  statistics, or another centred carrier. Those families already fail the
  actual-source mathematical gates above. The next shader edit requires a
  qualitatively different, fixed-read line representation which first passes
  camera/direction continuity outside the renderer.
- A subsequent end-to-end coordinate audit found one independent live bug in
  the winner-only attribute re-query. Depth used the solved botanical top entry
  `tE`, while isolated-profile normal and authored colour explicitly reset to
  `tScene`. On flat ground the two phase addresses differ by `H*cot(alpha)`:
  `13.4477 m / 25.86 tiles` at 5 degrees and `4.3908 m / 8.44 tiles` at 15
  degrees for this profile. Thus pale panicles could acquire the normal/colour
  of an unrelated repeated green plant even when depth election itself stayed
  unchanged.
- The correction reuses `bestDtEUnbounded` and `tScene+bestDtE` for the
  isolated winner re-query. It adds no texture read, binding, pass, dispatch,
  barrier, loop, candidate, or new control divergence. Typecheck and two exact
  real-WebGPU boots passed without console, shader, bind-group, or validation
  errors. The pale authored material is visibly attached to the elected field
  again. This is a retained correctness fix, not acceptance of the remaining
  depth/silhouette reconstruction.

## 33. Periodic cell-exit factorisation retained; primary codec rejected (2026-07-22)

- Centered depth is proved to be an invertible shear of the same
  four-dimensional exterior oriented-line coordinates. It reduces numeric
  range but does not restore a disoccluded owner or the fifth inside-origin
  coordinate. A two-plane/epipolar warp has the same boundary: it is exact on
  proved same-owner support and cannot create a target event hidden in all
  fixed source views.
- Horizontal periodicity does give one exact new identity. Split a pointed ray
  at its first canonical X/Z cell or botanical Y-slab exit. The successor is
  the first event in that finite local segment, or on a miss one
  boundary-entry four-dimensional suffix event. Both queries can be evaluated
  at fixed work and selected branchlessly; there is no cell walk, march, event
  tail, hardware grass geometry, shell, or species loop.
- A concrete rank-four local-plus-boundary allocation was costed at four fixed
  reads, `42,080,048` resident bytes (leaving `9,041,104` bytes), and 72 dense
  FMA-equivalent operations before transforms. This is a cost envelope, not a
  passed codec.
- The first actual-source support gate rejects it as the primary visual
  solution before fitting. Across the immutable 174,720-ray origin-aware
  validation truth, only 16,145 rays (`9.24%`) and `15.18%` of true hits resolve
  before first exit. At exact horizontal, 0.1 degrees, and 1 degree only
  `10.29%`, `10.30%`, and `10.71%` of true hits are local. Roughly 90% of the
  failing low-oblique image therefore still requires the unsolved long-range
  boundary field.
- The split reaches 100% of vertical true hits, confirming the derivation, but
  vertical/top-down is already the visually good regime. Training rank-four
  factors would mostly retry the rejected exterior carrier without a new
  angular/disocclusion mechanism, so no misleading local-only PNG or runtime
  change was made.
- Full math, measured table, cost, tool and report hashes are in
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-CELL-EXIT-POINTED-LINE-FACTORIZATION.md`. Artifact:
  `data/work/groundcover-cell-exit-support/2ed57f59d86e8376/ae91218c866d05bc/`.
- Decision: **preserve the exact identity for future camera-inside composition,
  but reject and park its rank-four pair as the next carrier.** Resume only
  after an independently successful fixed-read boundary/exterior representation
  exists. Offline-unioned multi-species/moss remains compatible and mandatory.

## 34. Hybrid band-limited Calamagrostis source gate (2026-07-22)

- The accepted high-detail morphology is no longer treated as one homogeneous
  opaque first-event soup. One exact production shoot was isolated without a
  second botanical recipe: factoring the shoot author left the complete
  six-shoot production mesh byte-identical at SHA-256
  `37b0cf1d33f632b5cde3dd60410ad987eec2e977661d0bc804dd6f2bee7054c0`.
  The isolated source is 333,520 vertices / 353,245 triangles, SHA-256
  `8cd69c2a6c61043c2861cad3a5ec06fee6dd4346a6722fd01fcc43468071ce5d`.
- The authored palette supplies a non-heuristic source partition. Green opaque
  blades, culm and basal structure have `G >= R`; reproductive rachis, glumes,
  lemmas, callus hairs and anthers have `R > G`. Majority vertex classification
  assigns 375 triangles to explicit offline structural geometry and 352,870
  triangles to the plume. The shared weld triangle follows the two plume
  vertices, so the split covers every source triangle without overlap or loss.
- The plume is compiled offline into view-conditioned premultiplied
  radiance/coverage. For `N` subpixel samples in one output footprint, the
  stored coverage and premultiplied colour are
  `A=(1/N) sum_i h_i` and `C_p=(1/N) sum_i h_i C_i`, where `h_i` is the opaque
  microscopic first-hit indicator. This removes individual 0.12--0.22 mm hair
  ownership from the later carrier while retaining their aggregate fluffy edge,
  holes, pale callus material, purple-pink glumes/anthers, and authored branch
  outline. It is a source representation change, not a depth filter applied to
  a wrong live ray.
- A real WebGPU Chromium render produced four full-colour transparent cardinal
  views plus an 18-degree low-oblique perspective diagnostic at 3,072 square,
  then resolved them to 1,536 square with explicit premultiplied-alpha Lanczos3.
  The resulting plant-height pitch is 0.636 mm/pixel; 44--51k fractional-alpha
  pixels in each cardinal complete view demonstrate that the microscopic
  silhouette has become continuous coverage rather than binary owner noise.
  Browser console, page error, uncaptured WebGPU, and renderer diagnostics were
  empty.
- Direct inspection of all four cardinal images and the low-oblique image shows
  a crisp, correctly pink/cream fluffy panicle, green structural foliage, no
  camera-front widening triangles, and no distance/angle stretch. The adjacent
  `*-plume-rgba.png` files are the actual compiled reproductive layer;
  `006-low-oblique-structure-rgba.png` proves the complementary opaque source.
  This passes the source-bandwidth/silhouette gate only. It does **not** claim
  that the still-rejected live line reconstruction has been fixed.
- Content-addressed artifact and full machine index:
  `data/work/groundcover-bandlimited-botanical-source/968e74302585c74af46f59fadb6c9ff541e4afb1bd3adc42602bf29c5e2ed87f/`.
  `qa/001..004-*-complete-rgba.png` are the four primary full-colour alpha
  views, `qa/005-low-oblique-complete-rgba.png` is the perspective gate, and
  `qa/007-cardinal-complete-checkerboard.png` is the human contact sheet.
- No runtime shader, asset binding, draw, dispatch, pass, sample, loop, march,
  register live range, synchronization, or GPU allocation changed. The intended
  integration boundary remains one tiny fixed-cost precomputed-ray query over
  an offline-composed community. Overlapping species and moss must be combined
  before this compilation; they must never multiply the live query by species.

## 35. Hybrid structural/deep-transfer mathematics admitted for one fit (2026-07-22)

- The user's latest visual verdict remains **no improvement**: the live field
  still has distance-growing stretch and the wrong oblique perspective. No
  current runtime mapping is accepted, and this section does not reinterpret
  that failure as an inherent artifact or tune around it.
- The next mathematical candidate changes the source visibility object rather
  than adding live directions or reprojecting another microscopic owner. The
  accepted GCRP/v4 source has 113 foliage/culm/rhizome charts (2,400 triangles)
  that remain opaque structure. Its other 270,422 charts / 2,168,734 triangles
  remain at full authored detail offline but compose into fractional panicle,
  spikelet, hair and anther transfer.
- “Opaque structure” is only an offline delta-interaction classification. It
  does not retain grass meshes, triangles, or a second structural query in the
  runtime; the live result remains one precomputed transfer field.
- A complete deterministic segment reduces exactly to eight scalars
  `(T,Q,C_rgb,N_xyz)`, with `Q=int T(s) ds`. Adjacent records compose as
  `(T1*T2, Q1+T1*Q2, C1+T1*C2, N1+T1*N2)`, and the same record elects the
  conditional first-interaction depth `mu=(Q-L*T)/(1-T)`. A lone opaque leaf
  reconstructs its exact depth; plume colour, normal and depth share one
  interaction measure, so there is no independent winner-attribute re-query.
- Exact horizontal and camera-inside rays use the proved first-cell split, but
  the local and boundary features feed one joint decoder which also receives
  `e/155`. This preserves the real pointed-origin and remaining-horizon
  coordinates. It assumes no periodic antiderivative, infinite line, raised
  carrier, grazing clamp, five-degree substitute, or division by vertical ray
  slope.
- The candidate retains the previously costed allocation: `42,080,048` resident
  bytes, four fixed filtered reads, and a proposed hard ceiling of 192
  FMA-equivalent operations. It adds no loop, march, traversal, event/species
  candidates, shell, pass, binding or synchronization.
- This is an admitted hypothesis, not a solved codec. The algebra removes the
  plume's microscopic event-count lower bound, but rank-four bandwidth,
  structural silhouettes, plume conditional variance, full camera-path
  continuity and panicle retention are not yet measured. Permit one bounded
  actual-source fit; one widening sector, wrong-view sheet, field-edge plane,
  panicle dropout or camera-relative band parks it without increasing cost.
- Full equations, costs, acceptance tests and primary-source/contribution
  boundaries are in `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-HYBRID-DEEP-TRANSFER-FACTORIZATION.md`; Experiment
  019 is recorded in the central grass source ledger. No runtime shader was
  changed by this mathematics track.

## 36. Hybrid affine head rejected before fit; exact low-oblique truth preserved (2026-07-22)

- The live visual verdict remains unchanged: oblique/distant stretching and
  wrong perspective are **not improved**. The hybrid experiment did not touch
  the live shader and does not claim otherwise.
- The original `(T,Q)` FP16 target was rejected as numerically ill-conditioned.
  Recovering sparse interaction depth via `Q-LT` subtracts two horizon-scale
  values. The preserved truth instead stores direct opacity `A` and normalized
  premultiplied first moment `m=M/L`, with exact composition
  `A12=A1+(1-A1)A2` and
  `M12=M1+(1-A1)(M2+L1*A2)`. Segment ownership is half-open `[0,L)` and a
  botanical Y-slab exit has an explicit terminal mask rather than addressing a
  periodic side-boundary factor.
- The exact isolated Calamagrostis source was traced at the declared 18-degree
  low-oblique camera path: 17 frames, `64x64` pixels, eight persistent micro-rays
  per pixel, 69,632 macro records and 557,056 stable labelled micro-ray records.
  The source remains 333,520 vertices / 353,245 triangles; full authored
  green/pink colour, structure/plume coverage, first-interaction depth, and
  conditional variance are emitted as numbered PNG truth panels.
- The sampled factor addresses themselves are adequately constrained
  (`2.788` training equations per active scalar; `96.51%` of held-out records
  have all corners observed). This does **not** rescue the decoder. The proposed
  affine head `h(b,r)=p(b)+c*r` has identically zero mixed cross-difference
  between boundary state and remaining distance. Exact finite-horizon suffix
  transfer, for example `T(b,r)=exp(-kappa(b)r)`, has nonzero mixed
  cross-difference whenever two boundary states have different extinction.
  Therefore this head cannot represent the required field even with perfect
  training data.
- A second independent blocker exists at the current live interface: the grass
  resolve carries opaque depth/body plus normal/tip and writes final alpha one.
  Fractional `A` cannot survive that interface without a new compositing or
  stochastic-coverage contract. Neither was silently added.
- Decision: **stop before fitting and reject this affine head.** No extra rank,
  read, layer, loop, march, stochastic coverage path, or memorising codec was
  introduced. Truth, persistent labels, machine metrics, and the explicit
  stopped-reconstruction panel are preserved at
  `data/work/groundcover-hybrid-transfer/8cd69c2a6c61043c2861cad3a5ec06fee6dd4346a6722fd01fcc43468071ce5d/63a512106a67f5cc44b90d655b178f09f50e36cd2a28488c9c47d30d5a238554/`.
  Tool SHA-256 is
  `d2f8f9876233f7c19d91c801f381b558af42432c67cc3494b51878e15c1bfb51`;
  report SHA-256 is
  `8f97bfc2c77b190fcf524712f32792a18a414602ccb013d385a79ae15d7c01fd`.

## 37. Endpoint-conditioned grouped rank eight is a measured NO-GO (2026-07-22)

- The current live stretching/wrong-perspective image remains rejected and
  unchanged. This track made no runtime or shader edit.
- Exact isolated Calamagrostis truth now spans the full pointed endpoint query
  `(u,v,y,phi,theta,L)`. Its four-micro-ray footprint, half-open ownership,
  nonlinear endpoint map, face-forward normals, contiguous coordinate
  holdouts, exact horizontal/vertical rays, endpoint sequences, camera-inside
  censor pairs, and preserved 18-degree path are machine-bound in the manifest.
  Endpoint/censor gates use `A` and physical `M=Lm`, not `m` alone.
- An optimistic `256x256` sampled-matrix oracle independently fits the best
  rank-eight SVD to A and M with fixed W rows versus E columns. Even this looser
  problem has only `0.4193` coverage IoU, `0.2445` A p95 error, `0.2015 m` M
  p95 error, `3.391 m` conditional-distance p95 error, and `0.9210` plume
  recall. Rank eight retains only `59.99%` of A energy. Its raw optimum violates
  A/M physicality tens of thousands of times; metrics were not clamped.
- The fairness-corrected continuous `W(u,v,phi)*E(y,theta,L)` rank-eight model
  enforces `0<=m<=A`, premultiplied colour bounds, and `|N|<=A`. It still fails
  general held-out truth at `0.2655` IoU / `6.764 m` conditional-depth p95 /
  `0.4341` plume recall. The exact 18-degree path is `0.4639` IoU with
  `4.543 m` conditional-depth p95; its numbered reconstruction is a dark blurred
  sheet with pale panicles largely missing. Censor A/M p95 change errors are
  `0.1323/0.1263 m`, and exact-vertical phi invariance has `0.708` maximum
  predicted disagreement despite zero truth disagreement.
- Decision: **reject and park rank eight.** Since the continuous gate and the
  stronger optimistic SVD lower bound both fail, the `212x212x48` W and
  `64x193x20` E tables were not sampled or quantized. No rank/read increase,
  table growth, loop, march, pass, memory addition, or shader integration is
  authorised. Truth, model, metrics, and QA are preserved under
  `data/work/groundcover-endpoint-gdm/8cd69c2a6c61043c2861cad3a5ec06fee6dd4346a6722fd01fcc43468071ce5d/0526e98735b745f3074212ac07e1ff8d6e5c456061283b0d443e897955362207/`.

## 38. Closed-form projective inverse is a measured NO-GO (2026-07-22)

- The current live stretching and wrong perspective remain rejected and
  unchanged. This was pure mathematics plus an offline actual-source gate; no
  shader/runtime path was integrated.
- For top-plane live phase `q`, slopes `s,s_i`, and vertical drop `h`, the exact
  correspondence sign is `x=q+(s-s_i)h`. A canonical geometric normal gives
  `grad D_i=n_xz/(n_y-n_xz dot s_i)`, so one Sherman--Morrison plane inverse is
  `h_p=h_0(n_y-n_xz dot s_i)/(n_y-n_xz dot s)` and
  `x_1=q+(s-s_i)h_p`. A second complete event is accepted only if
  `|x_1-q-(s-s_i)h_1|<=epsilon`; colour, normal, coverage, and mark remain
  coupled to that event rather than an infinite plane.
- The corrected gate repeats the immutable isolated 333,520-vertex / 353,245-
  triangle Calamagrostis source on its actual 0.52 m lattice. An identical-
  direction control reproduces all 868 hits and 476 misses exactly, proving the
  sign and implementation. Held-out truth contains 21,504 rays at two half-bin
  azimuths and `0.1/1/5/10/25/45/65/82.5` degrees.
- The strongest allowed K=4 path updates the normal once and uses three records
  per direction: twelve fixed loads, no loop/march/traversal. The exact target
  is visible in at least one of four source directions on 94.790% of hits, but
  the two closed-form updates reach it on only 4.147% and the selector returns
  the exact first event on only 3.722% (`5.375%` panicle, `0.782%` structure).
  Of 792 strict residual-valid selected hits, 197 are farther self-consistent
  events; line residual does not certify firstness. K=8/one-update target
  visibility is 97.905%, yet exact recall is only 0.344%.
- Exact horizontal has no finite top-plane slope/phase; arbitrary inside-origin
  visibility remains a pointed-line successor. The exact first-cell prefix does
  not supply its boundary-height suffix field. The candidate therefore fails
  independently in both held-out exterior convergence and horizontal/inside
  domain coverage.
- Decision: **reject and park.** Do not add another inverse iteration, direction
  candidate, looser residual, live triangle, or runtime search. Full derivation,
  resource accounting, source-versus-basin diagnosis, provenance, and QA are in
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-PROJECTIVE-INVERSE-CORRESPONDENCE-GATE.md`; artifact:
  `data/work/groundcover-projective-inverse-correspondence/8cd69c2a6c61043c/2967319b3f513252/`.

## 39. Six-coordinate all-pairs plane field is a measured NO-GO (2026-07-22)

- The user's live verdict remains **unchanged stretching and wrong
  perspective**. This experiment made no runtime or shader edit.
- The frozen alternative used all 15 unordered pairs of
  `(u,v,y,phi,theta,L)`, one rank-four Hadamard product, a fourteen-term
  quadratic lift, and an affine eight-output head. Its proposed runtime layout
  was one `512x512x15 RGBA16F` array with complete mips: `41,943,512` resident
  bytes including the f32 head, 15 fixed filtered samples, and 178
  FMA-equivalent decoder operations before physical output decode. It added no
  loop, march, pass, search, or shell. The layout and gates were frozen before
  fitting.
- This borrows only the all-`d choose 2` bilinear-plane/Hadamard-product
  precedent from the primary K-Planes paper. Six-coordinate pointed finite-ray
  transfer, botanical moments, packing, decoder, and falsifiers are local
  hypotheses. HexPlane is recorded only as adjacent plane-factor evidence.
- After 80 epochs the continuous field itself fails: general validation is
  `0.1838` coverage IoU, `0.5572` A p95 error, `6.046 m` conditional-depth p95,
  `0.2345` colour RMSE, `139.03 degrees` normal p95, and `0.6047` plume recall.
  It cannot even fit training records (`0.1933` IoU). Exact horizontal depth
  p95 is `36.994 m`; exact vertical IoU and plume recall are both zero.
- The unseen 18-degree path is `0.4897` IoU / `2.718 m` conditional-depth p95.
  Direct QA inspection shows broad horizontal wrong-view sheets instead of the
  source panicle and blades. This is the prohibited artifact family, not a
  small metric miss.
- Decision: **reject before f16/table sampling.** Encoding a known-wrong
  continuous function cannot be rescued by hardware bilinear filtering. No
  tables or runtime integration were produced. Full equations, source boundary
  and result are in `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-PAIRWISE-PLANE-TRANSFER-GATE.md`; report and numbered
  QA live under
  `data/work/groundcover-endpoint-pair-planes/8cd69c2a6c61043c2861cad3a5ec06fee6dd4346a6722fd01fcc43468071ce5d/6517639e8eda451e356c71dad6f1938795d13e3723de48650e5d600749efc21d/fit-width64-epochs80/`.
- Fixed overlapping grass species, litter, and moss still require no live
  species loop or coordinate: their marked geometry is unioned offline before
  tracing. Only a live-changing ecological mixture needs another state
  coordinate or asset; mark/material payload remains a later explicit gate.

## 40. Fixed-read primary-source fallback audit is a precise negative (2026-07-22)

- The latest user verdict is definitive: the live low-oblique stretching and
  wrong perspective are **not improved**. The current runtime remains a
  rejected reference, not a solution or partial acceptance.
- A bounded primary-source audit checked the exact adjacent mechanisms requested:
  Sannikov's follow-ups, VDM/GDM, surface and cube-surface light fields,
  lumigraph reconstruction, deep shadow/opacity maps, Fur PRT, shell/fin fur,
  cone-traced fibres, and near/far fur reflectance. No shader/runtime file was
  changed.
- No source supplies a materially different small fixed-read interpolation or
  compression mechanism. GDM/VDM use the direct origin/view field but
  reconstruct a dense or low-rank scalar already more favourably falsified by
  the exact-source rank-eight oracle. Surface light fields factorise radiance
  only after a live triangular mesh has elected visibility. Cube-surface light
  fields store an exterior 4D RGB field (108 GB raw in the authors' declared
  `M=64,N=512` example); nearest aliases and the fixed 4x4 interpolation
  explicitly blurs edges, while depth, normal, mark, and camera-inside
  successor remain absent. Deep and fur methods precompute lighting or
  transmittance while retaining explicit fibres, shells, fins, or layers.
- Sannikov's centered signed depth remains valid conditioning and his rejection
  of linear depth blending remains important. The actual PCF-like comparison,
  winner, and attached-attribute rule are not public, so no algorithm may be
  invented and attributed to him from the label or opaque effect node.
- Decision: **do not run another renamed fit.** A new actual-source gate needs
  either the missing author rule, a published nonlinear owner/event
  correspondence for the full pointed-ray domain, or a new LAAS mathematical
  invariant with bytes/reads/ALU frozen first. Increasing directions, ranks,
  blended rays, terms, layers, or candidates is not such an input. Full costs,
  source links, and paper-ready provenance boundaries are recorded in
  `docs/deep-research/grass/FIXED-READ-NOVEL-VIEW-PRIMARY-SOURCE-NEGATIVE-BOUNDARY.md`.

## 41. Frame-global pinhole surface/light field is a measured NO-GO (2026-07-22)

- The user's new runtime verdict is definitive: distance-growing stretch and
  wrong perspective are **not improved**. The live path remains rejected. This
  gate changed no runtime or shader file.
- The bounded alternative selected one to four canonical directions once from
  the camera optical axis and used those same fields and weights for every
  pixel. Each field used one genuine fixed botanical mid-plane and the exact
  pinhole map
  `F_C(Q)=Q+(Q-C)(n dot c)sigma(Q)/(k-n dot C)`. Only coverage,
  premultiplied radiance, and the first depth moment were globally combined;
  no owners or per-pixel direction candidates were blended or elected.
- The algebra is exact for the surface defined by a canonical scalar event
  sheet, but that sheet is not the actual first surface seen from the held-out
  camera. Its error is transported as
  `delta F=(Q-C)(n dot c)delta sigma/(k-n dot C)`: it radiates from the camera
  and grows with carrier distance. The f64 CPU QA visibly reproduces widening
  V-shaped streaks, wrong perspective, ghost panicles, and path mismatch
  without WebGPU, rasterisation, packed depth, or shader arithmetic.
- On the exact periodic 353,245-triangle Calamagrostis source, narrow 18-degree
  coverage IoU is `0.1777`, ghost alpha `0.6533`, panicle recall `0.2356`, and
  depth p95 `3.294 m`. Wide oblique IoU is `0.1102` with `27.399 m` depth p95;
  2-degree grazing IoU is `0.2154` with `26.184 m` p95. A seven-frame path with
  `2 mm` steps has `0.09638` premultiplied delta error against a `0.03` limit.
- Exact horizontal/camera-inside also fails independently: the fixed carrier
  has `k-n dot C=0`, no unique horizontal carrier point, and no pointed-line
  successor after an arbitrary inside origin. Adding another carrier would
  remove a coordinate pole but not create the missing shared first-event
  invariant demonstrated by the exterior failures.
- The rejected layout nevertheless fits its resource freeze: four complete
  field samples, estimated `224` FMA-equivalent operations including the exact
  stitched normal, and `43,133,472` resident bytes. There is no loop, march,
  search, candidate list, shell, extra pass, barrier, or per-species query.
- Decision: **reject and park after the single allowed actual-source attempt.**
  Full equations, results, cost accounting, provenance, and image hashes are in
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-FRAME-GLOBAL-PINHOLE-LIGHT-FIELD-GATE.md`; artifact:
  `data/work/groundcover-frame-global-pinhole-field/8cd69c2a6c61043c/3de6240997bff07d/`.

## 42. Four-view exact hit-height carry is a measured NO-GO (2026-07-22)

- The active live stretching/wrong-perspective defect remains unresolved and
  unchanged. This track made no runtime or shader edit.
- Current and `b8e0a10` `GroundCoverProfiles.ts` and `NaniteGrass.ts` were read
  in full. The old path combined a translated/raised-terrain query with
  four-view projected-path relift. The current path removed that shell/query,
  defaults to a centred categorical record, and retains the old four-view
  formula only as a dormant branch.
- The strongest cheap replacement was derived without renderer assumptions.
  A canonical record is addressed through the real botanical middle plane;
  its stored hit height `y_i` is placed on the exact live ray by
  `t_i=(H-y_i)/(-d_y)`. This identity is exact and uses neither horizontal-path
  relift nor a duplicated terrain surface. It does not, however, create
  correspondence between unrelated canonical and live first events.
- An exact canonical decode control on the actual checked-in GCRP passes:
  `0.99523` silhouette IoU, `0.0000331 m` depth p95, `0.98321` exact owner,
  and `0.97847` attached triangle. Held-out half-bin views then show the real
  failure. The true live owner occurs among the four records on only `0.3279%`
  of true hits. Height blending has `0.63864` IoU / `8.466 m` depth p95 and
  fabricates depths/mixes owners on `64.54%` of predicted hits. Even a
  truth-assisted oracle over the same four records reaches only `0.93569` IoU,
  `5.993 m` depth p95, and `0.337%` attached-triangle agreement.
- Two-millimetre motion p95 error remains metres at grazing/oblique angles.
  Near-inside oracle successor recall is `0.72917` with `2.049 m` depth p95;
  exact-horizontal middle-plane rays have 48/64 real successors and zero
  representable top-height predictions. These are independent information
  deficits, not WebGPU, depth-format, interpolation, or precision bugs.
- Frozen implementable cost was four eight-byte complete-record reads, one
  winner-colour read, at most five compare/select stages and one division; no
  loop, march, candidate list, pass, binding, barrier, dispatch, or species
  query. It is rejected for geometry, not expense. Do not enable the dormant
  four-view branch or add another direction row as a claimed fix.
- Full proof, metrics, provenance, and implementation boundary are in
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-FOUR-VIEW-HIT-HEIGHT-GATE.md`; Experiment 023 in the central ledger;
  artifact:
  `data/work/groundcover-four-view-height-gate/2ed57f59d86e8376/8d1465c218cd38a3/`.

## 43. Literal four-read radiance field changes the artifact but is a measured NO-GO (2026-07-22)

- The live stretching/wrong-perspective defect remains unresolved and
  unchanged. This was an offline actual-source image/geometry gate; no shader
  or runtime file changed.
- The exact exterior live ray was mapped to fixed-top-plane phase plus
  direction. Four angular nodes were sampled with hardware-equivalent periodic
  phase bilinear filtering, then only premultiplied authored RGB and coverage
  were combined. Depth, owner, normal, and reconstructed position never entered
  colour. Therefore decoded-event stretching is absent by construction; the
  measured residual is direct image-field error rather than a depth-relift
  failure.
- Four allocations were charged at the existing `51,121,152`-byte acceptance
  cap. The strongest uses 207-square phase, azimuthal rows at
  `1/5/15/35/60/85` degrees, and one shared azimuth-independent 90-degree pole:
  97 directions, `50,844,684` bytes, four filtered RGBA reads, about 28 scalar
  lerp FMAs, one binding, no loop/march/search/candidate/pass/shell/depth read or
  species multiplier.
- Low-angle support is necessary: versus current `15/35/55/75`, exact 5- and
  1-degree rows raise silhouette IoU from `0.87240/0.83854` to
  `0.99479/1.00000` and panicle retention from `0.39867/0.42373` to
  `0.94020/1.00000`. It is not sufficient. Corresponding RGB p95 errors remain
  `0.49280/0.52031`; broad connected error regions occupy `30.47/25.65%` of a
  frame.
- Aggregate strongest-lattice IoU is `0.85167`, composited max-channel RGB
  p50/p95 is `0.21020/0.54714`, 44.01% of pixels have fractional coverage, and
  one connected high-error region reaches 59.51% of a frame. QA shows broad
  washed wrong-view cross-fades rather than bounded silhouette blur. Even the
  exact shared 90-degree row has only `0.26556` IoU, isolating discontinuous
  thin-coverage phase filtering from angular spacing.
- Camera shifts of `1/2.5/4.5 mm` retain unforced predicted family-change rates
  of `20.97/45.15/41.75/26.39/36.12%` at `90/18/10/5/1` degrees. Separately
  scored depth choices remain metre-wrong: conditional same-live-ray position
  p95 is `6.083 m` aggregate, `7.165 m` at 5 degrees, and `7.310 m` at 1 degree;
  even an offline hit-aware sixteen-point oracle is `6.737 m` aggregate p95.
- Decision: **reject as the complete grass representation.** Preserve the
  genuine artifact distinction and the requirement for low-angle rows plus a
  shared vertical pole. Do not attach terrain/carrier depth, blend scalar
  depth, or increase samples/memory. Full equations, QA, temporal metrics,
  radiance/depth boundary, and hashes are in
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-DIRECT-RADIANCE-FIELD-GATE.md`; Experiment 024 in the central ledger;
  artifact:
  `data/work/groundcover-direct-radiance-field/2ed57f59d86e8376/83b63ea1cb677fa4/`.

## 44. Support-bounded canonical planes remove fans only by removing grass (2026-07-22)

- The active live stretching/wrong-perspective defect remains unresolved and
  unchanged. This was a pure-math and offline actual-GCRP gate; no runtime or
  shader file was edited.
- Each of four nearest-phase angular records reconstructs its exact source-
  triangle plane, hit point, and conservative in-triangle radius. The exact
  live ray intersects all four planes and a fixed minimum selects the nearest
  positive candidate only when it remains inside both the radius and source
  triangle. Owners are never blended.
- The frozen primary pack is eight bytes: hit-Y16, oct-normal XY16+16, and a
  floor-quantised support16 over 64 mm. Four reads plus one winner RGBA8 colour
  read cost 36 bytes; estimated ALU is 120--150 FMA-equivalent operations plus
  four divisions. No loop, march, traversal, candidate table, shell, pass,
  barrier, dispatch, or species query is added.
- A literal canonical record-centre control proves the method and packing:
  f64 ideal exact-owner recall is `90.339%`; packed recall is `90.078%`, with
  ordinary accepted p95 position error `14.53 micrometres`. The packed result
  is within `0.261` percentage points of ideal.
- All held-out views fail completely. Across 3,920 rays / 1,979 truth hits at
  grazing, oblique, high, top, and exact half-bin views, both the f64-ideal and
  packed four-record methods return zero hits: `0%` exact owner, panicle, and
  structure recall. A separately costed sixteen-read `2x2` phase footprint
  also returns zero.
- The reason is measured, not threshold-dependent: median certified source-
  triangle support is `0.1283 mm`, while median live plane displacement is
  `0.3246 m`, or `2,308.7x` support. Microscopic tessellation triangles are not
  view-stable botanical charts.
- Practical verdict: conservative support reduces connected false-positive fan
  width to exactly zero and produces no wrong-owner geometry, but false
  negatives are 100% of truth. They can be encoded honestly as zero
  premultiplied coverage without relocating geometry; visually that is an
  empty grass field, not a usable filtered silhouette. Grazing omissions form
  one connected component spanning the full 0.52 m tile.
- Decision: **reject and park.** Do not loosen support (which restores the
  infinite-plane stretch), add phase taps, or integrate the record. Full math,
  packing, category/motion metrics, visual QA interpretation, provenance, and
  resume condition are in
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-SUPPORT-BOUNDED-CANONICAL-PLANE-GATE.md`; Experiment 025 in the central
  ledger; artifact:
  `data/work/groundcover-support-bounded-plane-record/2ed57f59d86e8376/5428b274dc9fed8f/`.

## 45. Shell-frame field: dense premise confirmed, sparse reconstruction rejected (2026-07-22)

- No runtime or shader path changed. The live grass remains on its current
  implementation; the shell-frame field is parked before implementation.
- Section 12's correction to the first residual audit was valid. The actual
  connected `Sphagnum capillifolium` carpet has 100% top-down interception;
  raw crisp residual p95 is about 10.6--12.5 mm through 5° and falls to
  3.4--4.0 mm at 1°. Every per-view shell was smoothed under the exact
  mean-plane-chart E2 invariant `|grad o_i| max|s-s_i| <= 0.5`.
- The decisive sparse/open `Agrostis capillaris` reconstruction then used the
  fixed 97-node Experiment-024 lattice and four separately costed variants:
  6-read no-realignment, 7-read winner-only, strict 9-read, and 12-read
  second-step ablation. Radiance alone is blended; geometry/owner/mark remain
  categorical.
- Strict-9 is RED by a large margin: IoU `0.61974`, RGB max-channel p95
  `0.36095`, largest connected wrong region `44.53%`, crisp winner geometry
  p95 `12.88498 m`, and per-view unforced class changes `19.66--52.52%`.
  Required values were respectively `>=0.97`, `<=0.15`, `<1%`, `<=0.05 m`,
  and `<5%`.
- Correct crest-vs-interior stratification does not rescue the proposal. The
  outer crest scores IoU/RGB/geometry p95 `0.59585/0.365/13.998 m`; in-band
  content scores `0.63567/0.365/12.723 m`. At 1° the filled binary silhouette
  reaches `0.99870` IoU, but RGB p95 remains `0.26905`, geometry p95
  `15.459 m`, and class changes `52.23%`.
- The failure is information-theoretic at the record level, not filtering or
  denominator instability. Sparse first hits remain multimodal inside a shell
  footprint; residual correction often moves to a different owner/periodic
  successor instead of refining one Lipschitz event sheet. No-realignment is
  actually better in aggregate IoU/RGB (`0.62839/0.35118`) than strict-9
  (`0.61974/0.36095`), and the second step worsens them to
  `0.58276/0.37124`. The proposed `O(Delta^2 sigma)` bound therefore does not
  govern sparse cover.
- Cost passes: 97 directions use an estimated 21.62 MB raw / 6.18 MB
  compressed, below the 50.84 MB baseline. Reconstruction quality, not cost,
  blocks the route.
- Decision: **reject and park the one-record shell-frame field.** Resume only
  with upstream math for a fixed-size multi-stratum record plus a loop-free,
  non-oracle live stratum selector and accepted fixed cost. Do not add angular
  rows, correction iterations, depth blending, a correctness filter, or a
  runtime mesh.
- Dense artifact:
  `data/work/groundcover-shell-frame-field/2f1dca02ed141e8f/718dd44345aba66f/`.
  Final reconstruction artifact:
  `data/work/groundcover-shell-frame-field/2f1dca02ed141e8f/cca44e1d11aa283e/`.
  Full blocker and resume condition:
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-22/GRASS-SHELL-FRAME-FIELD-RECONSTRUCTION-BLOCKER.md`.

## 46. Class-E proposal was re-derived before implementation; corrected exterior specification is conditional (2026-07-22)

- Per user direction, this phase was pure mathematics only. No CPU/GPU
  harness, shader, baker, renderer, asset, or runtime path was changed. Camera
  inside plant matter is entirely outside the quality contract and may fade;
  no successor machinery remains.
- The inherited class-E handoff is mathematically held. Its ideal opaque core
  survives only in a narrower form: a fixed union of globally affine,
  periodic 2D-mask extrusion fields with finite analytic ray-valid intervals
  has exact exterior first contact through `t=t_a+rho/s_p`, a categorical
  family-axis pole, and a fixed minimum network.
- Finite-axis caps repair the old infinite-chord definition, but exactly
  horizontal finite structural axes remain outside the selected
  terrain-independent flat-ground-compatible periodic subclass. They require
  a measured upward-tilt residual, genuinely sub-pixel analytic treatment, or
  a new proved finite-axis structure. The compiler may not assume `4–7`
  directions; the accepted plant's weighted covering/field-count curve is the
  gate.
- Raw angular-slice depth still creates a distance-stable screen wedge for a
  horizontal exterior camera inside an empty part of the botanical slab. The
  corrected fixed-cost decode stores one categorical hit point plus canonical
  boundary normal and re-intersects the live ray with that tangent. This is
  exact for a certified straight compiled boundary and second-order for a
  smooth one; it is the full phase-aware form of Sannikov's cosine correction.
  It adds no candidate/read, but only applies in cells certified to preserve
  intrinsic boundary identity **and** final clip/support interval category.
- Sub-pixel plume/fuzz no longer uses the false `V(q,omega)` total-transfer
  record. A finite nonnegative analytic spectral medium has an exact closed-
  form optical-depth integral for every direction and opaque cutoff. Different
  colours are exact only with shared colour or a fixed ordered segment
  decomposition; source approximation is gated in maximal partial-ray
  integral norm.
- Curved terrain, spatial wind, smooth world-keyed geometric height, arbitrary
  root-based control fields, and opaque minification are not silently called
  exact. Each now has either an explicit affine/interval domain, a deterministic
  bound, or a stated no-go/unresolved gate. Independent per-field opacity mips
  are proved insufficient because they lose overlap correlation; a joint
  analytic coarse medium is the leading deterministic candidate but remains a
  measured approximation.
- The provisional cost equation counts two layers, every field/interval,
  control, material, plume, and minification access. With nine total samples,
  four one-sample opaque fields fit only in the illustrative near-field case
  where every other term totals one; the accepted plant has not yet proved
  that budget. Resident bytes likewise include all interval/control/plume/
  minification data under the 250 MB ceiling.
- Binding documents:
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-22/CLASS-E-MATHEMATICAL-AUDIT.md` (why the inherited
  theory is RED) and
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-22/CLASS-E-EXTERIOR-MATHEMATICAL-SPEC.md` (corrected
  exterior-only theorem, bounds, counterexamples, cost equations, and gate
  order). The old `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-EXACT-REPRESENTATION-THEORY.md` construction is marked
  superseded.
- Decision: **no implementation yet.** The next usable outcome, once the user
  authorizes leaving pure mathematics, is a structural compiler report for
  accepted Calamagrostis plus dense low/moss cover. It must produce the
  fidelity/tap/byte Pareto frontier before any shader transcription.

## 47. Class-E CPU gate is structurally RED before atlas or shader work (2026-07-22)

- The renderer-free affine equations pass seven independent CPU checks,
  including exact horizontal rays, the categorical pole, finite affine lift,
  tangent-line correction, fixed union ordering, root-local wind, and the
  analytic Fourier plume integral. The failure is not in that algebra.
- Cook-only generator sidecars now retain `270,541` exact Calamagrostis
  primitive recipes and `1,405` Sphagnum recipes. Both source meshes remain
  byte-identical, and every triangle has exactly one recipe attribution.
- The binding structural result is a global cap-plane lower bound for the
  proposed `G_f=A_f(M_f x I_f)` class. On flat ground, one field supplies only
  two horizontal cap planes. For the accepted production Calamagrostis's
  `102` blades plus `6` culms, omitting the entire reproductive head, the
  minimum is `14` fields / `29` optimistic two-layer reads at `10 mm`, `9` /
  `19` at `20 mm`, and `5` / `11` at `50 mm`. The target is nine total reads.
- This is not an over-strict-threshold failure. The isolated-shoot
  fixed-catalogue lower bound reaches `4` fields / `9` reads only at `100 mm`
  error, while its constructive grouping still needs `10` / `21`. `100 mm`
  is many authored blade widths and about one tenth of plant height.
- Sphagnum stems/cores fit a vertical direction, but its `1,228`
  near-horizontal curved branches are only medium candidates; no credit is
  taken without a partial-ray analytic-medium gate. Tall grass already blocks
  the representation independently.
- Decision: **do not implement the Class-E `M x I` runtime.** Return to the
  mathematics of phase-dependent finite endpoints. A successor must preserve
  the later eligible event when a nearer projected mask event fails its
  interval, while staying O(1), loop-free, geometry-free, and within the fixed
  read/memory budget. Details and the paper-designer handoff are in
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-22/CLASS-E-CPU-GATE-RESULT.md`; final artifact:
  `data/work/class-e-structural-feasibility/093984f242f07f37/8da8f737e15dd2ca/`.

## 48. Lateral relay core is constructive; fitted two-layer residual is RED and parked before runtime (2026-07-23)

- No shader, WebGPU, renderer, or runtime file changed. The strict offline
  F4 relay compiler produced `379` continuously height-certified arc edges
  (`315` foliage, `64` purple/brown reproductive) under the eight core-read
  two-layer budget. Its CPU tessellation is truth-query machinery, not a
  runtime mesh proposal.
- The second exterior CPU attempt used both Tier-1 affine populations on both
  the accepted Calamagrostis reference and candidate, `K=2` residual modes per
  layer (`K_total=4`), exact perspective bundles at `1/2/4/8/16/32 m`, full
  periodic phases, millimetric translations, exact horizontal rays, all-event
  prefix checks, and `4x4 -> 8x8` convergence.
- The fitted candidate is decisively RED: near-vertical IoU `0.278–0.306`, RGB
  p95 `0.421–0.553`, connected wrong region `66.7–72.2%`, prefix-alpha p95
  effectively `1.0`. A certified 4 m exact-horizontal source/core air corridor
  becomes candidate alpha `1`.
- For this fitted coefficient set, the positivity floor alone proves
  `tau>=3.62847`, hence alpha `>=0.97344`, on that corridor. This is independent
  of renderer details or quadrature. The `8x8` pass also preserves the serious
  low-angle early-core/depth errors, so they are not manufactured by a `1/16`
  sample atom.
- The broader K2+K2 class is not yet refuted. The residual was optimized per
  layer and then unioned, not optimized directly against the joint two-layer
  final transfer. Moreover the legal vertical notch
  `a[1-cos(lambda(y-y0))]^2` shows that one empty horizontal height cannot
  support a coefficient-independent horizontal/top-down no-go theorem.
- Decision: **park before runtime under the two-real-failure rule.** Resume
  only with a global/near-global direct joint two-layer K2+K2 minimax fit over
  the finite frequency catalogue, including horizontal and partial-cutoff
  constraints, or with positive-measure multi-height/phase emptiness evidence
  sufficient for a bandlimited Remez/Turan lower bound. Do not run another
  heuristic fit or begin shader transcription.
- Full report:
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-22/GRASS-LATERAL-CORE-RESIDUAL-CPU-GATE.md`.
  Strict network artifact:
  `data/work/groundcover-relay-arc-network/37b0cf1d33f632b5/6219c2153f9523c7/9917249acc582ac1/`.
  Exterior artifact:
  `data/work/groundcover-relay-exterior-ray-gate/37b0cf1d33f632b5/9d993b97802dc47d/a8ca42a9b2901f1b/`.

## 49. Remaining K2+K2 question reduced to a finite certified optimization (2026-07-23)

- No new fit or renderer/runtime work was run. The remaining joint-residual
  question is now specified as a discrete-frequency plus compact continuous-
  coefficient minimax problem over the complete two-layer final hybrid.
- Global positivity is exact rather than texel-sampled: horizontally
  independent phases use necessary-and-sufficient second-order cones; one-
  phase harmonic pairs use Fejer-Riesz Gram SDPs; rank-one-horizontal and
  vertical-only pairs use a certified compact-domain positivity separator,
  retaining legal vertical-notch candidates.
- The frequency catalogue is provably finite from the pixel law only when the
  residual includes a declared continuous pixel-footprint filter/analytic mode
  attenuation and a frequency-independent soft-density ceiling. A centre-ray-
  only evaluation or arbitrarily tall high-frequency spikes have no uniform
  cutoff. Even with those bounds, literal 1 m resolution permits roughly
  `586` horizontal and `2650` half-wave vertical indices, so the current
  171-mode catalogue is an explicit low-band authoring restriction, not a
  consequence of visibility alone.
- Near-globality requires frequency-support branch-and-bound plus interval
  envelopes for the exponential transfer, conditional convex colour solves,
  exact positivity certificates, and a complete `[lower,upper]` objective
  interval. Correlation may order nodes but never prune them.
- A decisive corpus begins with positive-measure, interval-certified
  multi-height/phase horizontal-air cones paired with cover-positive and
  coloured-head boxes, low-oblique core counterexamples, millimetric
  translations, and all event-adjacent prefixes. A certified separation oracle
  adds whole violating boxes until RED or exhaustive validation.
- Any nonnegative finite-Fourier two-layer residual that has exactly zero
  opacity on an open family of air rays must vanish identically: zero line
  integrals create an open zero-volume, affine pullback preserves openness,
  and the analytic identity theorem closes each layer. Visual tolerance makes
  the quantitative Remez/Turan version—not exact zero—the binding route.
- A global support certificate over the full billion-frequency 1 m box is
  computationally intractable (`Theta(P^4)` raw pair product). The practical
  choice is an accepted low-band residual, a structural Remez bound that
  bypasses enumeration, or a new polynomial-cost multiresolution support
  theorem; heuristic pruning cannot stand in for any of these.
- Solver design, catalogue/tail conditions, lower-bound ladder, estimated
  offline cost, and go/no-go rules are in
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-23/GRASS-JOINT-K2K2-GLOBAL-OPTIMIZATION.md`.

## 50. Inverse-authored F4/F6 ideal fields are RED and parked before runtime (2026-07-23)

- No shader, WebGPU, renderer, or runtime path changed. Both attempts used an
  exact continuous CPU ray oracle and identical physical filtering for source
  and candidate.
- F4 removed the camera-direction wedge mechanism and preserved a certified
  4 m exact-horizontal air corridor, but its field-global vertical sweeps
  produced top-view IoU near `0.316`, oblique RGB p95 near `0.45`, and metres
  of false early depth. A proposed three-cohort F5 was rejected before another
  run: at least one reproductive band would be `136.8 mm` tall while source
  glumes/hairs span about `4--6 mm`, forcing `25--55x` false persistence.
- The complementary F6 attempt spent four additional prospective reads on two
  exact horizontal head masks across two Tier-1 populations: six fields times
  two populations plus one control read, or `13` fixed reads. It improved the
  one-metre vertical result to IoU `0.389`, RGB p95 `0.470`, and first-hit depth
  p95 `0.179 m`, proving that head-height localisation matters.
- F6 nevertheless remained decisively RED. At 18/10/5/1 degrees and 1 m, RGB
  p95 was respectively `0.453/0.399/0.338/0.328`, while first-hit depth p95 was
  `2.821/3.717/6.781/8.562 m`. Low-angle IoU saturates only because both fields
  are opaque; the candidate is selecting wrong early periodic successors.
  The QA images remain visibly cross-laminated and banded.
- The extra reads therefore do not earn their cost. The field-global ruled
  interval plus sparse sheet basis is parked under the two-real-attempt rule;
  it will not be tuned with more cohorts, masks, filters, or runtime work.
- Active pure-math work moves to a genuinely different structural-core plus
  sub-pixel-plume transport split. Read count may rise modestly above nine only
  when each added access buys a missing invariant and the complete path stays
  fixed-cost, coherent, low-memory, and suitable for low/mid-range hardware.
- Full derivation, metrics, hashes, QA paths, waste record, and resume condition:
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-23/GRASS-INVERSE-AUTHORED-FOUR-FIELD-MATH.md`. F6 artifact:
  `data/work/groundcover-complementary-f6-ideal-field-gate/37b0cf1d33f632b5/bc8b67ff7684bdf2/2e17a0c43761211a/`.

## 51. Fixed-rank interval relay is a theorem-level NO-GO (2026-07-23)

- A packed four-event record has exact conditional decode algebra, but finite
  axial intervals do not imply four-relay completeness. Exact-horizontal
  exterior air-gap rays give an open-set `K+1` counterexample for every finite
  `K`; the field-axis pole needs a separate axial-successor structure.
- The proposed 32 live event lanes are semantically the forbidden candidate
  list even when unrolled. At useful phase resolution their descriptors and
  payload also exceed the resident ceiling.
- Retain only a narrow optional tip refinement over an independently correct
  carrier. It cannot repair F4/F6 because their carrier is itself wrong.
- Full proof: `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-23/GRASS-PACKED-INTERVAL-RELAY-MATH.md`.

## 52. Spectral whole-head replacement is RED; microscopic plume remains valid (2026-07-23)

- Direct finite-prefix integration is resonance-free and exact for an authored
  nonnegative shared-colour medium at every direction, air-gap origin and
  opaque cutoff, including two affine Tier-1 layers.
- It cannot carry the full resolved Calamagrostis head at small fixed cost.
  A finite trigonometric density has no positive-width air corridors; one
  10 cm macro localiser is already about 225 2D modes, while stable 5/2 mm
  edges require roughly 841/4761 robust grid modes per stratum. Interleaved
  violet, cream and green owners also prevent exact small coloured transfer,
  and a medium has no categorical first depth/normal for resolved glumes.
- Surviving scope is common-colour microscopic hairs or a measured far-field
  LOD after minification. It is not the near-field head carrier.
- Full proof: `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-23/GRASS-STRATIFIED-SPECTRAL-HEAD-MATH.md`.

## 53. Exterior visibility complex is exact in principle but outside the frozen runtime (2026-07-23)

- Exterior air gaps retain the fifth origin coordinate; top/side boundary
  charts cannot reduce pointed visibility to an unpointed oriented line.
- A fixed-depth unrolled DAG is O(1), but it is a dependent traversal. Existing
  incomplete layouts measure 219.5--357.5 MB, and realistic exact-predicate
  queries are estimated around 17--27 serial reads. A highly shared integrated
  5D DAG is not information-theoretically disproved, but no measured artifact
  fits the no-traversal/read/memory contract.
- The corrected boundary and exact resume conditions are in
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-23/GRASS-EXTERIOR-VISIBILITY-COMPLEX-MATH.md`.

## 54. Complete unconditional arbitrary-soup contract is internally inconsistent (2026-07-23)

- Arbitrary soups can encode an unbounded number `M` of independently visible,
  near-field resolved bits. Any representation promising fixed-error fidelity
  needs at least `M` bits; no fixed 250 MB state can provide that guarantee for
  unbounded `M`.
- Independently, `M` separated opaque intervals on one line give `M` exterior
  air-gap origins with different successors. No universal fixed-rank record
  supplies them without traversal/candidate enumeration or an unbounded field.
- Therefore arbitrary-soup guaranteed fidelity, the fixed approximately
  9--14-read no-traversal query, and fixed memory cannot all be unconditional.
  The strongest honest product contract keeps arbitrary soup as compiler input
  but makes source fidelity a compulsory rate-distortion gate into a bounded
  exact runtime class.
- This conclusion was the boundary under the former camera domain. The user
  then allowed a lane to fade while the camera is inside the guarded box
  volume containing the plant/overlap component. Section 55 records the
  resulting constructive state reduction; the finite-state byte bound remains
  binding.
- Consolidated proof and contract:
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-23/GRASS-COMPLETE-MODEL-BOUNDARY.md`.

## 55. Guarded box-boundary transfer is the complete active model (2026-07-23)

- The user clarified that the quality contract may fade a plant lane whenever
  the camera lies inside the physical box volume containing it—not only while
  intersecting literal triangle matter. This selects the missing upstream
  relaxation without excluding ordinary air gaps between separate carrier
  footprints.
- Each motion lane is compiled into a stable guarded common-slab carrier
  `U=P x I`. Clipping the ray to `I` and querying the pointed periodic
  first-passage field `rho_P(q_a,omega)` gives the first carrier entry exactly,
  including arbitrary footprint gaps, exact horizontal rays, and a categorical
  vertical pole. An outgoing half-open convention prevents reselecting the
  just-exited component.
- At the first carrier boundary `q`, the complete forward community is one
  four-dimensional transfer `T(q,d)`. It already sees through an empty first
  box to every later periodic copy. Thus no live successor, event tail, march,
  candidate list, or traversal remains. An origin inside a carrier fades that
  lane; optional union-mask exit then strict re-entry is still fixed work.
- Resolved transfer cells keep one categorical plane/copy/material/root token
  and re-intersect the exact live ray. Uncertified final cells are legal only
  as explicitly physically filtered premultiplied appearance with conservative
  depth intervals/moments. Cross-owner depth, normal, mark, or species blending
  remains forbidden.
- The selected finite codec is a direct certified `4^4` macrobrick field:
  uniform regular/MISS bricks, lossless topology-codeword plus deduplicated
  palette-set bricks, and filtered VQ bricks only below the declared physical
  resolution. Mips are baked independently from the oracle. The carrier uses
  the same categorical boundary-feature rule; interpolated first-passage
  distance across successor changes is forbidden.
- Concrete budget: the illustrative six-chart allocation has `6,291,456`
  finest bricks; eight-byte headers cost `50.33 MB` base / `67.11 MB` with the
  contemplated mips. A 4,096-pattern byte topology dictionary costs `1.05 MB`,
  all 2.17M source triangles as 16-byte charts cost `34.72 MB`, and 65,536
  32-byte mixed atoms cost `2.10 MB`. About `145 MB` remains for deduplicated
  palette sets, the complete carrier, page tables, and auxiliaries. The actual
  palette/carrier entropy is a mandatory cook gate; no arbitrary-soup theorem
  can guarantee every input fits 250 MB.
- Fixed cost is `4--6` reads per arbitrary-soup lane. Two Tier-1 affine grass
  lanes reuse the same atlas bytes and cost `8--12` reads; control plus one
  final winning-root terrain/deformation correction targets a packed
  `<=14`-read production schedule. Moss hummock form/material stays in packed
  terrain at zero transfer lanes; resolved static moss is a supported third
  lane but can reach `13--18` reads and is not silently called free.
- Curved terrain and root-wise `+-15%` height/gust variation are not called
  affine. A cooked local terrain chart elects the canonical owner; the real
  winning root, plane, stiffness, tint and scale are then evaluated once and
  re-intersected exactly. Continued firstness is certified against every later
  event using the terrain/deformation displacement, incidence, order, and
  barycentric margins. Remaining ambiguity is accepted only when its whole
  world band fits the live pixel footprint and its RGB error passes.
- Species count does not set runtime cost. Every ecological community state is
  a jointly baked marked union; categorical species/stiffness follows the
  winner. Independent nonlinear/phase motion laws—not species names—create
  extra lanes. Per-root geometry/species toggles after the lookup are invalid
  because removing a winner can reveal an unbounded successor.
- Invocation is one query for every potentially intersecting screen ray in an
  existing full-screen resolve/fragment path. Scene depth is only the opaque
  cutoff; it may not suppress grass against sky. No grass mesh, quad,
  billboard, shell, raised terrain copy, floating carrier raster, or new grass
  geometry pass is part of the model.
- Binding documents:
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-23/GRASS-BOX-BOUNDARY-TRANSFER-MATH.md`,
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-23/GRASS-BOUNDARY-TRANSFER-FINITE-CODEC.md`, and
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-23/GRASS-BOX-TERRAIN-MOTION-CLOSURE.md`.
- Runtime remains held only for the first actual-community CPU gate: fade
  volume, categorical carrier no-skip, regular/mixed certification, palette
  entropy, exact bytes/reads, terrain/deformation stability, silhouette/RGB/
  geometry error, seams, and millimetric camera translation. One consolidated
  diagnose/fix cycle is permitted if red; no shader experiment precedes green.

## 56. The continuous reduction is green; the finite codec is RED (2026-07-23)

- The actual accepted Calamagrostis source confirms the periodic
  common-slab identity: 192/192 deterministic rays agree with direct periodic
  triangle truth, with maximum composed hit error `1.95e-14 m`. This validates
  the continuous carrier plus whole-forward boundary transfer, not its finite
  encoding.
- The active 2.5 cm guarded six-shoot union covers 100% of the 0.52 m tile and
  creates one 1.20022 m-high carrier lane. While the camera is at a height
  inside that periodic lane, the entire grass layer is in the permitted fade
  domain. A finite ecological patch still needs a canonical outer side-entry
  factorization; the cap-only gate did not test it.
- The proposed `4^4` macrobrick codec is rejected. Every sampled view/tier was
  categorically mixed; the finest `Q512` cell is about 1.016 mm wide while the
  frozen closest-exterior pixel footprint at the 2.5 cm guard is about
  13.6 micrometres. Such cells are visibly resolved and cannot legally be
  renamed filtered atoms. Dense physical-footprint sampling is orders of
  magnitude over the byte cap before direction and payload are charged.
- The gate's point-centre versus four point-corner RGB/coverage score is only
  an interpolation diagnostic, not a valid final filtered-image metric. The
  next representation must be justified against the correlated camera-pixel
  integral and scored with that same integral; this correction does not make
  the rejected codec green.
- No runtime, shader, geometry, carrier plane, pass, loop, march, or candidate
  path is authorized. Resume requires both an exact finite-patch side entry
  and a new compression theorem/model predicting `<=250 MiB` before a held-out
  filtered actual-source cook. Repacking or merely increasing the current 4D
  atlas does not qualify.
- Physical filtering also exposes a fifth *filter-state* dependency even
  though central-ray visibility is four-dimensional. Cameras on the same
  exterior line share `(q,d)` but have different standoff and therefore
  different correlated pixel footprints. Production must either store the 4D
  field at the smallest supported footprint and filter it with fixed live
  work, or store and charge a finite standoff/covariance family. One filtered
  value per `(q,d)` is not complete for arbitrary exterior origins.
- Binding gate:
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-23/GRASS-BOX-BOUNDARY-TRANSFER-CPU-GATE.md`.

## 57. Filtered-codec verdict corrected to INVALID / UNDECIDED (2026-07-23)

- Section 56's finite-codec RED is superseded as a production decision. The
  categorical point-ray census remains evidence of a discontinuous/highly
  mixed field, but it did not construct the correlated physical pixel target,
  fit a filtered codec, or score held-out decoded pixels.
- The first shared-origin follow-up accidentally treated a `6x4` diagnostic
  image as the whole sensor. Its pixels were `8--56x` wider than the frozen
  `60 degrees / 1920 = 0.03125 degree` production pixel. Correcting only that
  premise reduced unconverged truth pixels from 70.0% to 37.2%, coverage-step
  p95 from 0.125 to 0.00781, and RGB-step p95 from 0.0677 to 0.0303. Its prior
  sampled RED was therefore false-worsened.
- The production-pixel rerun still cannot decide the codec. Its finite side
  decode disagreed with truth phase/horizon; chart wrapping was wrong;
  filter-family selection used oracle subrays rather than the analytic live
  Jacobian; the charged 16-byte atom was never quantized/decoded; fit and
  score sets overlapped; temporal error was not excess-over-truth; and total
  camera-depth moments lacked the entry/transfer cross term.
- Final result: point-ray factorization GREEN (`192/192`, maximum error
  `1.95e-14 m`); finite ecological patch and filtered full-domain codec
  **UNDECIDED**; runtime **not authorized**. No shader/runtime file was changed.
- Pure math also proves that exact physical pixel filtering adds unavoidable
  standoff/shear state: cameras on the same central exterior ray share `(q,d)`
  but integrate different 2D planes through the 4D light field. The minimum
  honest practical contract is bounded screen-space error with a measured
  finite Jacobian/filter family; no exact fixed-state identity exists for
  arbitrary soup.
- The two actual codec attempts are parked. Resume only with the validity
  defects fixed as one coherent full-domain cook, converged shared-origin
  truth, disjoint held-out cameras/addresses, physical packed decode, complete
  finite-patch closure, and exact `<=250 MiB` / `<=14`-read accounting.

## 58. GBC2-K6 is the active finite-codec proposal (2026-07-23)

- The user explicitly reopened the representation question and requested a
  concrete codec. The selected candidate is **GBC2-K6**: a guarded boundary
  field whose ordinary cells are certified MISS, one-owner REGULAR, or exact
  two-outcome CUT2, with one fixed six-pair-plane filtered terminal for
  genuinely multiway, chart-seam, pole, or copy-overflow cells.
- The central new result is that an opaque triangle's support edges and every
  pairwise depth-order separator are bilinear in the four boundary-ray
  coordinates. A CUT2 descriptor therefore stores only its two source tokens;
  the decoder regenerates the exact event polynomial, evaluates its tangent at
  the live ray, projects the live camera-pixel Jacobian through it, and
  integrates the resulting two-region pixel cut in closed form. Camera
  standoff changes the Jacobian and fixed ALU, not atlas dimensionality.
- The cap is one direct `128 x 128 x 32 x 16` grid of 64-bit `RG32Uint`
  descriptors. A 32-byte source payload holds triangle geometry, colour,
  normal, and categorical species/material. There is no page traversal,
  candidate list, retry, march, loop, per-species query, or runtime geometry.
- Complex cells use K6: six `448 x 448 x RGBA16F` coordinate-pair planes with
  jointly fitted scale levels and one fixed shallow head. Live pair-projected
  gradients select the fitted physical-pixel scale; analytic ray coordinate
  and Jacobian invariants enter the head without reads. K6 is an explicit
  held-out empirical terminal, not falsely described as categorical geometry
  or as a proof for every arbitrary soup.
- Per lane the schedules are `1/3/5/7` logical reads for
  MISS/REGULAR/CUT2/KPLANE. Two Tier-1 lanes plus control cost at most `15`
  logical reads. The fifteenth is declared rather than hidden; it is the one
  extra read the user allowed when justified by fixed multiway coverage.
  Anisotropic physical sampler traffic, register pressure, occupancy, and
  payload locality remain mandatory measured performance gates.
- The fifteenth read includes the one control record that supplies the local
  terrain/carrier chart and both Tier-1 affine fields. There is no hidden
  winner-root read: terrain curvature and smooth world-keyed tint/vigor/height
  variation must fit the certified local affine carrier or force cook-side
  carrier subdivision within the byte cap. A variant retaining a dependent
  real-root correction is honestly a `16`-read worst case.
- The accepted-source budget is `64.00 MiB` cap descriptors + `66.26 MiB`
  source payload + at most `13 MiB` K6 + `16--64 MiB` simultaneously resident
  finite-side/corner state. The intended complete total is `195--230 MiB` and
  the hard publication ceiling remains `250 MiB`, including all control,
  metadata, alignment, and padding. Both golden-angle layers reuse these
  bytes.
- No shader/runtime path is authorized yet. The next artifact is one coherent
  actual-source offline gate on Calamagrostis and a dense low community:
  certify/census the four modes and copy range, fit K6 on converged
  production-pixel truth, score disjoint held-out exterior cameras and live
  Jacobians including all angles/standoffs/seams/translations/two-layer order
  swaps, and report exact packed bytes plus logical and physical traffic.
- Binding proposal:
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-23/GRASS-RAY-SPACE-CUT-CELL-CODEC.md`.

## 59. Exhaustive runtime rewrite is fail-closed after the honest v4 gate (2026-07-23)

- The complete active isolated-Calamagrostis path was audited line by line
  against the exterior boundary-transfer and finite-codec mathematics. The
  terrain-coloured mosaic was not a colour or resolution defect: the retired
  v2 path blended appearance from unrelated histories, attached one unrelated
  categorical depth/normal, and lifted it through a scene-dependent terrain
  plane. The conformance report is
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-23/GRASS-GBR4-RUNTIME-CONFORMANCE-AUDIT.md`.
- The v2 full-screen overlay implementation is removed from the active grass,
  frame, and resolve path, including its three screen-sized handoff textures
  and compute/resolve plumbing. The ordinary multi-species path is unchanged.
- The replacement code now has strict `GCC1` carrier and `GBR4/v4` loaders, a
  fixed-cost exterior query, tagged MISS/REGULAR/MIXED semantics, live
  winner-plane intersection, physical-footprint selection, exact horizontal
  and vertical cases, and no loop, march, candidate list, runtime cover
  geometry, floating plane, or screen-sized cover handoff.
- The actual Calamagrostis v4 asset is deliberately `REFERENCE_RED`, and the
  loader exposes `runtimeBindAllowed=false`. Its 1,536 coarse records are all
  physically filtered MIXED; only 8/4096 tested fine cells received a
  continuum REGULAR certificate. A dense table at that measured scale would
  require about `1.729e18` addresses (roughly 6 EiB before payloads). Binding
  the small reference would therefore reproduce a coloured sheet under a more
  honest format. Details and the paper-designer handoff are in
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-23/GRASS-GBR4-V4-PRODUCTION-RED-BLOCKER.md`.
- Visible binding is consequently fail-closed. `grassprofile=2` now renders no
  ground cover instead of the invalid mosaic; a future status flip also stops
  until the fixed-chart resource adapter and real-WebGPU gate exist, so neither
  the publication bit nor missing plumbing can be bypassed accidentally.
- Verification: repository typecheck passes; six focused carrier/codec/truth
  regressions pass; exact real-WebGPU boots pass for both the isolated RED URL
  and the ordinary multi-species URL, with no page, TSL, shader, bind-group, or
  WebGPU validation errors in those active graphs. Because no GREEN resource
  can legally bind, the new direct boundary-query shader graph remains gated
  and is not claimed WebGPU-compiled. The isolated boot proves removal of the
  broken image, not visual grass acceptance.
- Resume condition: an adaptive macrobrick/VQ cook must cover every unresolved
  physical-footprint cell, keep REGULAR payloads lossless, pass held-out image,
  depth, seam, and 1--4.5 mm translation gates, and remain within 250 MiB and
  the fixed small-read contract. Until then there is intentionally no visual
  review URL for the replacement.

## 60. Isolated Calamagrostis recovery and standing-height correction (2026-07-23)

- Section 59 remains the verdict for the RED `GBR4/v4` boundary resource; it is
  not bound.  After that result, `grassprofile=2` was deliberately restored as
  an isolated visual-recovery lane using the accepted high-resolution standalone
  Calamagrostis GCRP/v4 source.  This is not the rejected multi-species renderer
  and not a claim that the boundary codec became GREEN.
- The renderer/profile/carrier files were split into responsibility-owned modules
  in commit `24027f1`.  The old multi-species path remains separately available
  and unchanged while the single-species lane is judged.  The recovery result and
  its honest scope are recorded in
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-23/GRASSPROFILE2-ONE-HOUR-RESULT.md`.
- User review established a new positive gate: at roughly 10 m height and oblique
  view, authored shape finally read correctly.  It also established the remaining
  failure: standing-height exterior rays still stretched/warped despite the eye
  being above the 1.1765 m plant box.
- Pure math identified the cause in the live code.  The standalone loader replaced
  baked 3D depth by horizontal travel, then divided by the live horizontal ray
  component.  For a 9-degree live ray clamped to a 15-degree row this reconstructs
  a true ground hit about 0.48 m above ground.  The derivation and replacement are
  frozen in `docs/tasks/2026-07-25/groundcover/active/GRASSPROFILE2-STANDING-EXTERIOR-MATH.md`.
- The isolated lane now preserves full per-node 3D depth, performs one fixed
  epipolar readdress per four lattice corners, elects one corrected geometry
  record categorically, and filters only corrected premultiplied colour.  Four
  initial coupled records + four corrected records + four corrected colour reads
  replace the old four depth + four normal + four colour reads: still 12 fixed
  reads, no loop, march, candidate list, pass, barrier, binding, or runtime cover
  geometry.
- Typecheck, 42 focused contracts, and exact real-WebGPU boots at standing shallow,
  standing downward, and 10 m control poses pass without browser/TSL/WebGPU errors.
  Captures remove the old camera-distance height sheet and retain the accepted
  medium-height structure.  Status is **awaiting close live-flight acceptance**:
  residual categorical owner wedges/phase jumps at low grazing cannot be certified
  from one still and remain the explicit visual risk of the 15-degree lattice floor.

## 61. Angular-ring fingerprint, vertical pole fix, and chart-consensus RED (2026-07-24)

- The three reported camera-centred circles are now quantitatively identified
  as categorical elevation-row transitions.  The `15/35/55/75 degree` lattice
  elects different rows at `25/45/65 degrees`; at a `1.7 m` eye those meet flat
  ground at `3.65/1.70/0.79 m`.  The user's additional observation that each
  ring keeps constant screen size while its world radius grows during ascent
  is the exact angular-cell signature `r=h/tan(beta)`, not an ordinary
  world-distance LOD.
- The undefined nadir was a separate data-domain defect: above `75 degrees`
  the old path reused sixteen tilted views while azimuth became undefined.
  The isolated Calamagrostis bake now appends one azimuth-independent vertical
  pole slice.  The cap is a continuous three-node slope barycentric with one
  forced-ineligible dummy; it preserves the current fixed
  `4 initial + 4 corrected + 1 winner-colour = 9` reads and adds no binding,
  pass, loop, march, candidate list, or runtime geometry.
- Installed GCRP/v4 hash:
  `59cebe81bbf7b7a70a1ed2aeb99d9d20f8405dff40cdb5a8936679cb2f6c6c1e`;
  parsed layout: `65` slices, pole index `64`, `13 x 5` atlas.  Typecheck and
  ten focused contracts pass.  The exact Estonia `grassprofile=2` URL completed
  ninety settled real-WebGPU frames without page, TSL, shader, bind-group,
  command-buffer, or uncaptured WebGPU errors.  User visual acceptance is
  pending and is scoped to the close top-down pole transition; standing rings
  are not claimed fixed by this checkpoint.
- The proposed same-chart consensus for removing standing rings is decisively
  RED on exact epipolarly paired v4 records: shared exact triangle/copy mass is
  `29.26%` at `25 degrees`, `36.98%` at the critical `45 degrees`, `38.43%` at
  `65 degrees`, and `33.51%` aggregate, versus frozen gates
  `35/60/35/50%`.  Even assigning every invalid source sample a favourable
  match raises the critical result only to `40.54%`.  No integer repack or
  shader consensus was implemented.  Resume only with a cook-certified coarser
  affine chart identity that clears the same epipolar thresholds.

## 62. Dedicated profile-2 renderer and sampled-centre plane correction (2026-07-24)

- The active `grassprofile=2` renderer is now completely outside the deprecated
  `LegacyPeriodic*` implementation.  It has dedicated contracts, builder,
  325-line screen query, and 439-line profile sampler; only the terrain/control
  guide and final material decode are neutral shared modules.  `NaniteFrame`
  routes the singular `calamagrostis-preview` binding directly to this path.
  The old GCAR path is still available separately and is now multi-only.
- Candidate E fixes a proved complete-record mismatch.  A nearest depth record
  belongs to the baked texel-centre ray `Q(q)`, but the old reconstruction
  attached that depth to continuous `q`, and then treated an interpolated
  shading normal as the triangle plane.  At a `1.44 mm` half-texel residual,
  the false live depth reaches `14.4 cm` at incidence `0.01` and `1.44 m` at
  `0.001`, directly producing view-direction wedges and translation swim.
- The new sampler computes the integer spatial texel once, fetches its exact
  atlas centre, and reconstructs from that same unwrapped centre at R0 and R1.
  The winner colour reuses the elected R1 centre.  The bake stores one flat
  geometric face normal per source triangle in the existing normal channels.
  Runtime stays exactly `4 R0 + 4 R1 + 1 colour = 9` profile operations, with
  no added read, byte, binding, pass, loop, march, candidate, or geometry.
- The first full cook rejected fragment-derivative face normals as
  non-deterministic on the 2.17M-triangle source.  The consolidated correction
  derives the face normal directly from the triangle's indexed positions and
  passes it flat; two complete Apple Metal-3 bakes then matched at
  `e3e0a4175b151b89a1a4bb58089aed2e809da337a04ce024240be8decdacb0be`.
  The installed file remains exactly `119462112` bytes: zero memory growth.
- Typecheck, 43 focused profile/math contracts, and diff checks pass.  The
  exact Estonia `grassprofile=2` URL completed 126 rendered real-WebGPU frames
  without page, TSL, shader, bind-group, command-buffer, or uncaptured WebGPU
  errors.  User flight review is the remaining gate.  This candidate primarily
  claims flat/affine stretch and camera-motion swim; it does not claim the
  independent curved-terrain direction-cell drift or the distant
  radiance-minification grain is already solved.
- User flight result: **only a slight improvement**.  Candidate E is retained
  as a proved record-consistency correction, but it did not materially close
  the dominant field-wide failures.  Major low-oblique/uphill stretch, wrong
  apparent view, translation swim, early fuzz/grain/directional particle crawl,
  incomplete geometry, and incoherent cover edges remain active.  No further
  shader change is authorised from this result; the track has returned to the
  pure reconstruction equations under the unchanged nine-read/zero-memory-growth
  contract.

## 63. Face-covariant dependent address (Candidate F, 2026-07-24)

- The surviving R1 equation was proved to intersect the live ray with the R0
  hit's horizontal constant-height plane.  It ignored both the snapped R0
  origin and the recorded geometric face plane, making it the wrong inverse for
  the predominantly vertical/tilted Calamagrostis surfaces.
- The dedicated sampler now computes the exact R0 face-plane/live-ray
  intersection and constructs the canonical R1 ray through that point.
  Parallel, behind-entry, and outside-source-slab hypotheses are excluded
  before categorical election and receive only a finite dummy fetch address.
- Runtime remains exactly `4 R0 + 4 R1 + 1 colour = 9` profile operations with
  no byte, binding, pass, loop, march, candidate, geometry, or legacy-path
  change.  R0 plane decoding is local to each node to constrain register live
  ranges.
- Typecheck, 43 focused contracts, diff checks, and 125 exact real-WebGPU
  frames pass without browser, TSL, shader, bind-group, command-buffer, or
  uncaptured WebGPU errors.  User flight review is the visual gate; Candidate F
  claims stretch/wrong-orientation/swim correction, not event closure or
  distant radiance minification.
- User flight result: Candidate F gives a real near-field stability gain.
  Individual nearby plume heads remain more coherent and can be followed during
  camera motion, so the face-covariant correction is retained.  The overall
  image remains similar, however: distant fuzz/stretch from only a few metres
  onward is unchanged, and looking across terrain whose cover sits slightly
  above the eye still produces the stronger warped/sliced reconstruction.
  This closes the two-attempt local-reconstruction cycle.  Further coefficient
  tuning is parked; the next work must separately close projected-footprint
  minification and curved-terrain chart drift without increasing the nine reads
  or resident profile bytes.
- Additional live defect: plume colour/extent is not owner-coherent in the
  perceived geometry.  Calamagrostis plume heads should retain a roughly stable
  fluffy violet extent, yet some views show violet on thin stem/leaf-like
  fragments while other leaves/branches disappear.  Code inspection confirms
  the geometry and colour atlases use the same slice, guarded texel, owner token,
  Z convention, and Candidate-E winner centre; there is no independent colour
  address in the active path.  The leading mechanism is therefore a plume-owned
  infinite face plane being reconstructed outside its finite triangle support,
  together with a missing leaf successor—not an arbitrary tint variation.
  This joins incomplete geometry and broken side silhouettes under the existing
  event/support-closure blocker unless a later owner diagnostic disproves it.
- The existing `grassdbg=flatres` live diagnostic held both authored colour and
  shading normal constant while leaving categorical visibility unchanged.  It
  passed 118 exact real-WebGPU frames, and the user confirmed the same
  one-direction “millions of tiny particles” stream remains.  Therefore colour
  radiance and face-lighting alias are not the dominant source of that artifact.
  Candidate S0/colour scale-stack is rejected as the next fix: it could smooth
  chroma but would leave the binding complaint intact.  The remaining source is
  binary hit/miss and owner/support turnover in the categorical geometry path.
- A second live diagnostic retained `flatres` but disabled TRAA (and therefore
  the mirrored Nanite camera jitter).  The user initially reported a reduction,
  but the subsequent AO-only comparison exposed a contrast confound: disabling
  AO made terrain and the constant green grass too similar for the moving grain
  to be judged.  A deliberately invalid full-resolution-shadow URL rendered the
  terrain grey after a 25-vs-24 sampled-texture validation failure; against that
  contrast the same crawl was plainly visible with AO disabled.  The inference
  that AO caused the grass artifact is therefore withdrawn.  AO and cast shadow
  make the position/visibility turnover easier to see; ordinary cast shadows are
  stable on terrain, while the same underlying grass instability remains visible
  without AO.  The invalid render is diagnostic evidence only and is not a valid
  runtime gate or shareable configuration.

## 64. Candidate G RED; Candidate H background-preserving review (2026-07-24)

- Candidate G (`GRASSPROFILE2-STANDING-EXTERIOR-MATH.md` Section 28) was rejected
  before implementation.  Filtered coverage followed by an opaque threshold has
  an unavoidable veil/erosion/stipple trilemma: a constant threshold turns a
  half-covered checkerboard into a sheet or a hole, while a varying threshold
  retains the moving-particle turnover.  No Candidate-G code or asset change was
  made.  Section 28.6 records the counterexample and exact blocker.
- The invalid `shalfres=0` diagnostic exceeded the requested fragment-stage
  sampled-texture limit (`25 > 24`) and consequently rendered terrain grey.  The
  grey output is not a valid runtime result, but its contrast exposed the same
  moving grass crawl while AO was disabled.  The earlier inference that AO caused
  the artifact is withdrawn: AO and cast shadow reveal the unstable grass samples;
  they do not generate them.  Raising the device limit to 48 is not an accepted
  fix because the renderer must retain the low/mid-range adapter contract.
- Candidate H is now mathematics/dataflow only in Section 29.  It goes up one
  level from the failed opaque assumption and preserves the real background sample
  so filtered coverage can use the exact overlay equation
  `C = A*C_grass + (1-A)*C_background` rather than a binary threshold.
- Its proposed profile payload remains nine physical profile texels and drops from
  `49.515 MiB` to `48.750 MiB`; the existing 64-bit ray texture is repacked and
  `depthV` remains untouched.  Fractional pixels preserve the existing background
  id/depth and set one reserved payload tie bit; exact opaque grass keeps the
  current foreground election.  The active split resolve reuses its existing
  third pass as a premultiplied grass overlay, while the no-GI merged path mixes
  internally, so neither topology gains a pass.  These byte/fetch claims are not
  yet an implementation licence: the seven-bit tie reservation, post-voxel query
  order, binding ceilings, fractional background-depth convention, and remaining
  categorical angular boundary must pass external review first.
- Actual runtime state since the previous external review: Candidate F alone was
  transcribed as approved.  It improved nearby plume identity/stability but did
  not materially improve distant fuzz, stretch, or uphill discontinuities.  No
  runtime/profile code has changed after F; Candidate G was killed on mathematics,
  and Candidate H remains unimplemented.
- External review verdict on H: **REVISE, architecture conditionally approved**.
  The background-preserving overlay, reserved-bit ordering, split/merged resolve
  topologies, fractional background-depth convention, eligibility-independent
  atom-1 carrier, and nine-read/48.750-MiB budget survive review.  Binding
  corrections are now frozen in Section 29: all solid and exact-grass writers
  mask the tie byte to seven bits; empty backgrounds remain an early-out; near
  records use depth12 + oct8,8 + class4; non-finite empty-atom planes fail cook;
  pass builders assert the default texture/storage binding ceilings; and the
  moving exact/fractional depth frontier is an explicit visual gate.
- Runtime implementation was gated on two cook-side representation measurements,
  whose thresholds were recorded before either result: a coverage-weighted joint
  16-medoid palette gate over plume and stem/leaf semantics, and an
  epipolar/world-corresponded angular-boundary continuity gate at every
  25/45/65-degree row boundary and azimuth midpoint.  Any RED primary metric is
  reported as a
  codec/carrier failure and prevents shader transcription; it is not worked
  around by another tap, more memory, threshold noise, or a higher adapter limit.
- Both measurements are complete.  The joint sixteen-entry palette is GREEN
  with large margin (`DeltaE00` overall p95 `0.741`, p99 `1.807`; plume p95
  `0`; stem/leaf p95 `0.782`; largest connected error region `0.0271%`).
  Palette compression is not the blocker.
- The one-read maximum-angular scale carrier is decisively RED at every row and
  azimuth boundary at both spatial scales.  The critical 45-degree/sigma-16
  result is `0.4706/0.6196` coverage p95/p99, `0.3012/0.4107` premultiplied-RGB
  p95/p99, `129.24/165.84` transverse pixels p95/p99 at 2 m, and a `91.14%`
  connected exceedance.  The least-bad raw carrier family still measures
  `5.67` pixels p95 under the 10 m pixel footprint against `0.75`; this is not
  threshold-edge noise.
- A false-RED audit confirms that the analyzer implements H's frozen atom codec,
  exact world/epipolar pairing, unwrapped nearest target centre, both directed
  pairings, and no equal-address fallback.  The gate measures the raw carrier,
  not a falsely claimed full rho-weighted composite.  That is nevertheless the
  frozen Section-29.3 requirement; at exact-MISS sites with one covered filtered
  contributor the weight cancels from representative position entirely.  This
  RED is scoped to the declared unit-square scale carrier, not every possible
  compact positive kernel.
- Candidate H's background-preserving overlay remains a reusable mathematical
  result, but its categorical angular carrier is parked before runtime.  No
  shader/profile asset was changed.  Resume only with a same-read/same-byte
  angular carrier whose boundary continuity is structural, then require both
  the same frozen continuity gate and a preregistered fidelity gate against all
  65 per-direction filtered truth fields.  This closes the constant-field hole:
  continuity without fidelity is not progress.  A direction-analytic vertical
  extinction profile is the first constructive candidate, but its arbitrary-soup
  fidelity at 32 bits per scale is an empirical codec obligation, not yet a
  theorem.  No extra tap, memory, third scale, filter widening, or noisy
  threshold follows this RED.

## 65. Candidate K/K2 RED; angular first-hit field remains the blocker (2026-07-24)

- Candidate K removed Candidate H's unnecessary filtered grass surface and
  represented only the positive unresolved foreground measure `(A,A*C)` over
  the real background.  Its direct `RGBA4444` stored-node codec is genuinely
  GREEN and its one-read allocation is below budget.  This supporting result
  does not establish arbitrary-angle reconstruction.
- The first held-out attempt accidentally retained radius-4/radius-16 filters
  while reducing the phase-page resolution.  That changed their physical
  support and is invalid evidence.  The corrected oracle preserves the
  accepted 256-grid footprint (`4,16 -> 1,4` at resolution 64), raycasts the
  actual periodic GCRP triangle soup, and includes `0.05--0.5` degree grazing
  directions.
- Corrected middle-reference result: **RED in unlimited precision** for
  `212/224` held-out direction/scale cases.  Worst p95 coverage is `0.667`
  against `0.08`, premultiplied RGB is `0.490` against `0.06`, and the largest
  connected exceedance is `70.1%` against `1%`.  Stored angular vertices remain
  exact/GREEN while sector midpoints fail; the result is therefore not an
  RGBA4444, atlas-resolution, or source-raycaster failure.  At `0.05--0.5`
  degrees coverage tends opaque but RGB p95 remains about `0.27`, so the
  assumed fine-scale horizon appearance limit also fails.
- Mechanism: first-hit appearance is a four-dimensional light field.  A middle
  reference plane removes parallax only for surfaces at that height.  The
  accepted plant spans the full cover box, so visibility events shear by
  `(X_y-h_ref) Delta(d_xz/|d_y|)` and cannot be recovered by bilinearly blending
  four independently raycast corner views at one phase address.  The correction
  needed is a different representation, not more packing precision.
- Candidate K2's nested parent pair proves exact `C0` equality at footprint-LOD
  thresholds, but does not make ordinary nearest phase-cell crossings
  continuous.  Its first real base/translation audit is also RED: enforcing a
  bit-identical coarse parent destroys the already-GREEN fine filtered field
  (`A p95 0.1306`, premultiplied RGB p95 `0.0995` on the budget-exact audit),
  while retaining the fine endpoint breaks the claimed LOD identity or budget.
  K2 is parked; its 9-read/40.244-MiB arithmetic remains reusable context only.
- No runtime, shader, resolve, or installed profile asset changed for K/K2.
  Candidate F remains the active runtime.  The reusable result is Candidate H's
  real-background fractional compositing protocol.  Resume requires a
  conforming spatial-and-angular transfer representation, not another
  categorical or same-phase four-corner carrier.  A targeted angular
  convergence audit is the one permitted premise check before choosing that
  next representation.
- That premise audit is complete and parks K unconditionally within its own
  topology.  At source-equivalent `sigma4`, `108/112` held-outs are unlimited-
  precision RED; at `sigma16`, `104/112` are RED.  Halving the angular cell
  width successively through `1/8` does not approach the frozen limits:
  normalized p95 remains `7.02` (azimuth) / `4.17` (elevation) for sigma4 and
  `3.22` / `1.47` for sigma16.  Uniform eightfold refinement in both angular
  axes already implies `8192` cell pages and roughly `2 GiB` in K's duplicated
  layout, yet remains RED.  Middle-plane versus top-plane ablation is mixed
  (`4.999` versus `4.356` median normalized error); reference-height tuning is
  not the cure.  The resume condition is therefore a conforming coupled
  phase-direction field, now specified as Candidate L, not a denser K lattice.

## 66. Candidates L/M/N RED; aligned filtered atoms are the final bounded ablation (2026-07-24)

- Candidate L tested the strongest conforming sampled-direction continuation:
  a true four-dimensional phase/direction simplex with shared vertices.  It is
  continuous, but the real Calamagrostis field varies `24--156` times faster
  than the admissible spatial/directional rate.  The targeted exact-BVH gate is
  RED for same-phase K, simplex L, and the stronger nine-vertex tensor ceiling
  alike (`7.0982`, `7.2126`, and `7.1790` worst normalized p95).  Candidate L
  is parked; adding sampled directions would require gigabytes and would still
  preserve the wrong amplified-view sectors.
- Candidate M removed sampled elevation and fitted four positive
  line-conditioned analytic transfer modes.  Its periodic-interior fidelity
  subgate is decisively RED before any runtime transcription: unquantized and
  packed are both `0/176` GREEN, with worst coverage p95 `0.8482` versus `0.08`,
  premultiplied RGB p95 `0.5176` versus `0.06`, and connected exceedance
  `91.58%` versus `1%`.  Exact training phases and azimuths remain RED
  (`0.8102/0.5033/0.9075`), proving that neither interpolation nor UNORM8 caused
  the broad-wash failure.  No modes, bytes, reads, sampled elevations, or
  decoder were added after RED.
- Candidate M's math audit separately caught a false top-entry premise for
  side/uphill views, missing finite-support integration, a non-exact colour
  quotient, unaccounted minification means, and an unproved near/far handoff.
  Those issues are preserved in its audit, but repairing them cannot rescue the
  already-failed easiest periodic interior.  M is parked.
- Candidate N tested four cubic
  Bernstein height moments with four separately parallaxed world-anchored C0
  phase fields; elevation, exact horizontal/vertical limits, side/uphill entry,
  and footprint contraction are analytic.  Its filtered output is only positive
  premultiplied measure over the real background—never a fake grass surface.
  The resolved record adds a finite owner-support micro-mask and incidence bound.
  The complete query schedule is `8/9/9/8` loads for resolved/B1/B2/unresolved,
  includes both affine variance layers, and totals `35.0 MiB` resident.
- N is decisively RED after the one permitted premise correction.  The first
  fitter projected an unconstrained solution into the nonnegative cone after
  fitting; the corrected fit solves in that cone.  The conforming result still
  collapses the source bands into a flat beige wash even at trained directions:
  exact-trained sigma-4 coverage/RGB p95 are `0.5198/0.3502` with `79.39%`
  connected failure, held-out sigma-4 is `0.4946/0.3238/83.37%`, and held-out
  sigma-16 is `0.3750/0.2607/65.04%`, against `0.08/0.06/1%`.  Four-bit packing
  is immaterial.  The handoff algebra itself is exactly continuous at
  `8/9/9/8` reads, but bridge fidelity was correctly not fitted after the
  stronger full field failed.  N is parked; resume requires removal of the
  one-point height-moment quadrature, not more moments, tuning, bytes, or reads.
- The only bounded successor still being measured combines two already-RED
  ingredients in one previously unmeasured way: four neighbouring angular
  nodes of Candidate-H's physically filtered positive atoms, each seed-read,
  epipolarly re-addressed by its representative height, reread, and then blended
  only in premultiplied colour/coverage.  Unlimited precision costs eight atom
  reads and emits no fake depth/owner/normal.  Shell-frame reconstruction and H
  each refute one premise separately, so this is an ablation-sized gate, not an
  open-ended track.  A RED result closes it; a GREEN result must still prove C0
  spatial packing, the one-read bridge, both affine layers, and the resident
  budget before Fable review or runtime work.

## 67. Candidate O RED; representative-height alignment route closed (2026-07-24)

- Candidate O measured the final bounded ablation from Section 66: four
  neighbouring angular nodes of Candidate H's positive filtered atoms, each
  seed-read, epipolarly re-addressed by its representative height, reread, and
  blended only in premultiplied colour/coverage.  It emits no fake depth,
  owner, normal, or surface.
- The corrected oracle proves its own chart transcription.  All `32/32` exact
  stored-node controls are GREEN at numerical zero, including atlas
  orientation, middle-plane signs, angular endpoints, and seed/corrected
  identity.  The failure is therefore not a coordinate, sign, packing, or
  implementation defect.
- Held-out reconstruction is decisively RED: `0/48` direction/scale cases
  pass; worst coverage p95/p99 is `0.6053/0.8796`, premultiplied-RGB p95/p99
  is `0.4576/0.6413`, and connected exceedance is `58.56%`.  Only `9/48`
  translation cases pass.  Sigma-16 remains far outside the limits
  (`0.3781` coverage p95, `0.2970` RGB p95, `12.76%` connected).  The
  separate one-read bridge is `0/48` GREEN and a deterministic two-height
  counterexample is also RED.
- This is the predicted missing-height-strata failure: one representative
  height cannot align multiple independently varying plant heights.  More
  packing precision, another correction, more blur, or a runtime trial cannot
  restore the omitted phase shifts.  Candidate O is parked with no runtime or
  shader change.  The durable result is
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-CANDIDATE-O-ALIGNED-POSITIVE-ATOM-GATE.md` and artifact
  `data/work/groundcover-candidate-o-aligned-atoms/e3e0a4175b151b89/b3b7abee02162ee4/`.
- The remaining active design question is a genuinely coupled phase-direction
  codec whose phase and angular continuity are structural and whose complete
  two-layer query, handoff, resident memory, and fidelity fit the existing
  ceilings.  A sampled-view blend, representative-depth warp, fixed height
  moment set, boundary-zero private event, or tiny direction-analytic latent
  is not a candidate under a new name.

## 68. Candidate P fixed-mesh capacity RED; same-count direction-partition audit only (2026-07-24)

- Candidate P is a jointly continuous canonical event-function field: a
  `64 x 64` periodic phase finite element and a conforming angular finite
  element over `240` real hemisphere triangles in the existing `256` record
  allocation.  Its proposed runtime cost remains three physical reads per
  affine layer and `16 MiB` for the P atlas; it adds no geometry, marching,
  loop, candidate list, or per-species work.  No runtime code was changed.
- The final corrected unlimited-precision gate optimizes the phase
  coefficients and scores exact phase vertices, phase interiors, held-out
  directions, and `1--4.5 mm` translation differences.  The unrestricted
  angular phase control is `72/72` GREEN statically and `1151/1152` GREEN in
  translation.  Its only miss is a sigma-4, `4.5 mm` standing-case coverage
  p95 of `0.06874` against `0.06`; worst static coverage/RGB p95 is only
  `0.04119/0.02665`.  This establishes that the phase grid is basically
  adequate and localizes the remaining pressure upstream in angular capacity.
- The preserved favourable joint affine-rank-two oracle is materially RED:
  `32/72` static and `1062/1152` translation cases pass, with worst coverage
  p95 `0.13442`, RGB p95 `0.10996`, translation coverage p95 `0.09799`, and
  largest connected exceedance `0.00073`.  Failures concentrate in standing
  cells: sigma-16 static fidelity and sigma-4 motion at `4.5 mm`.
- A coupled least-squares solve converged numerically (normal residual about
  `8e-7`) and reduced average loss, but worsened the binding tail metrics; it
  is optimizer evidence, not an accepted codec.  Sigma-4-first lifted rank
  two is also RED.  Independent per-scale fitting did not construct a better
  witness and, being a theoretical superset, cannot turn that numerical miss
  into a no-go theorem.
- Durable evidence is
  `data/work/groundcover-candidate-p-rank2-bound/e3e0a4175b151b89/21a11246991582a6/`
  (report SHA-256
  `51e6b64bece429c73ed3aada3db7afe89ad753e8ba2e04a0e6cd9af406bd34a1`)
  with filtered truth at
  `data/work/groundcover-candidate-p-filtered-truth/e3e0a4175b151b89/a1ff5b80bf4e007d/manifest.json`.
- Quantisation, chords, bridge, packing, shaders, and runtime remain
  unauthorised; Candidate F is still the live renderer.  The sole active
  upstream audit may redistribute the same `256` angular record slots into a
  conforming, analytically indexed nonuniform partition.  It may not add
  cells, reads, bytes, events, runtime lookup taps, searches, or loops.  After
  two fixed subdivision failures this route parks.

## 69. Candidate P 2x adaptive-direction gate RED; rank-two refinement parked (2026-07-24)

- The adaptive audit first removed a potential false result: cell location,
  barycentrics, held-out generation, and exact truth now all use the identical
  logical-polar `(elevation, unwrapped azimuth)` chart.  A conforming 2x local
  refinement is analytically locatable with fixed comparisons/floors/selects,
  uses only `186` real triangles in its conservative witness, and changes no
  read or resident byte.  A stronger global 2x witness can retain the original
  grid at `250/256` records.
- The corrected 2x gate is decisively RED: `0/144` standing static cases and
  `1887/2304` translations pass; worst coverage/RGB p95 is
  `0.135882/0.110044`, translation coverage/RGB p95 is
  `0.109981/0.081773`, and the largest connected exceedance is `0.001709`.
  Halving both angular dimensions therefore is not a sufficient remedy.
- The unrestricted-angular phase control remains statically strong
  (`144/144`) but exposes a separate translation tail of `0.092438` coverage
  p95.  The grazing phase-only control is completely GREEN.  Thus neither a
  chart mismatch nor the standing cell's first subdivision accounts for the
  active response complexity.
- The result does **not** prove that affine rank two cannot converge under
  arbitrarily fine angular subdivision.  The preregistered `1.834x` non-narrow
  stop rule and the track's two-failure limit prevent a 4x run; the documented
  zero-byte `250`-cell 4x topology remains a separately authorisable resume
  path, not an active claim.
- Evidence is
  `data/work/groundcover-candidate-p-adaptive-direction/e3e0a4175b151b89/73102281018064a4/report.json`
  (SHA-256
  `26e1cbfbff65cdf824ab057df0a867ab009965999d3dee0583bfbdf04edac031`).
  The durable derivation and exact horizon contract are in
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-ADAPTIVE-DIRECTION-LATTICE-MATH.md`.
- No quantisation, chord fit, bridge, runtime, or shader work followed.  The
  next upstream audit changes the record's local response dimension, not the
  sampling rate: Candidate Q uses palette-compressed positive states to fit
  three scale-continuous sheets/four regions in the same `128` bits and same
  three reads.  It receives one favourable rank-three capacity gate before any
  nonlinear or runtime work.

## 70. Candidate Q favourable rank-three capacity RED; event-sheet track parked (2026-07-24)

- Candidate Q proposed three ordered scale-continuous event sheets and four
  positive palette states in the same `128` bits, reads, and resident memory as
  Candidate P.  Its math audit found an additional structural condition:
  shared sheet weights do not by themselves make the decoded measure C0 when
  incident cells carry different positive states; complete decoded edge
  responses must match.  The measured oracle deliberately ignored this and
  was therefore strictly favourable.
- The one bounded run reused the original and corrected logical-polar exact
  truth with zero new rays.  Its independent `R4` affine-rank-three endpoint
  oracle is decisively RED: `72/252` static and `1727/4032` translation cases
  pass; worst coverage/RGB p95 is `0.320523/0.458205`, connected exceedance is
  `15.8447%`, and translation coverage/RGB p95 is `0.227268/0.688661`.
  Failures are the standing cases in both datasets; every grazing static case
  fits.
- The shared-coordinate `R8` rank-three diagnostic is `0/252` GREEN
  statically.  The unrestricted-angular phase control remains `252/252` GREEN
  statically and `3979/4032` in translation, so the failure is angular
  response capacity, not a phase-grid or chart transcription defect.
- The favourable oracle already grants arbitrary independent endpoint
  coordinates and discards positivity, ordering, canonical-edge, packing, and
  palette constraints.  A nonlinear ordered-sheet fit cannot repair this
  missing dimension.  Candidate Q is parked; no sheet fit, quantisation,
  runtime, or shader work followed.
- Evidence:
  `data/work/groundcover-candidate-q-rank3-capacity/e3e0a4175b151b89/2060286bc07c3dc7/report.json`
  (SHA-256
  `5e5ec2abe57b4118a8744ed2fe1d6aa1309ef6e2dfc0fb1ddf342b576672455e`).
  The run took `1098.45 s`; total track time was `1739.84 s`, inside the
  `3600 s` limit.  Candidate F remains the live renderer.
