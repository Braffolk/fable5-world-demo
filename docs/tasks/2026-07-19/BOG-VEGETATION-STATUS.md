# Bog vegetation arc — status (2026-07-19)

TLDR of today's bog work. Done section is terse; "not done yet" has the detail.

## DONE (committed)
- **8 bog plant meshes** built by Fable, grounded in real reference, user-approved: cotton-grass,
  heather, Labrador tea, bog rosemary, cranberry, cloudberry, bog pine (stunted), bog birch (bog form).
  Self-contained modules in `src/vegetation/bog/`. — commit `8a9f574`
- **Preview harness** `tools/veg-preview.ts` (headless WebGPU turntable → PNGs for isolated QA). — `e833286`
- **Understory wiring**: VegClass 32–37, VegLibrary pools, classPolicy sets, ScatterMap keywords
  (split off the generic BushPink collapse), SpeciesMap `SK`→Birch fix. tsc + boot clean. Generated
  world byte-identical (underK untouched). — commit `12775b0`
- **Height sanity check**: all 8 within real-world ranges even after 1.3× sizeFor. No change needed.
- **CARPET layer architecture** proposed + user-approved (name = CARPET; declarative `CarpetSpec`
  + generalize 3 gates, no new subsystem). Doc: `scratchpad/bog-veg/GROUNDCOVER-LAYER-ARCH.md`.
- **Moss/UE Nanite-Foliage research** recovered + summarized (voxel-aggregate far + sparse near-field
  opaque cushions + terrain tint; never per-cushion the whole carpet).

## NOT DONE YET (with detail)
1. **Per-part blossom/berry color** — task #27, IN FLIGHT. World leaf shader packs ONE tint/pool and
   uses vdata.x only as hue-jitter → flowers/berries render muted foliage green (cranberry not red,
   cloudberry not amber, heather not purple, cotton-grass head not white, etc.). Fix = clean 2nd
   blossom color on the leaf pool + a vdata.x select in `NaniteResolve.ts`, no regression on non-
   flowering pools. Also fixes the existing muted forest flowers (#113). Then commit.
2. **Cook-side placement + 256 m test map** — task #20 (blocked on #27 committing; overlaps files).
   - Plants only scatter where cooked bog **community-5** cells are, currently sparse (~2/255) and not
     at the default review camera → a good first look needs the density raise, not just the wiring.
   - `understory-communities.toml`: raise bog density + microtopography-driven placement (hummock vs
     lawn vs pool-margin via cooked relief/wetness) + add `bog_rosemary`/`andromeda` token (its class
     + keyword are already wired, dormant until the token exists).
   - `trees.py`: NEW deterministic scatter sampler over the classId-8 bog mask (placement today is
     nDSM-peak-only → flat bog gets zero trees). Sparse stunted pine interior + downy birch margins,
     low `ref_height` so cooked scale reads stunted.
   - RENDERER prep (clean, no hacks): decouple bog-tree VegClass from TREE_SPECIES index (tree block
     0–15 full; BOG_PINE/BOG_BIRCH need classes ≥32 routed as trees via classPolicy/windProfile),
     then wire the tree pools. **Tighten bog-tree seed variance** (birch 28k→140k tris / to 6.17 m on
     unlucky seeds) so they stay firmly stunted + cheap.
   - species-map.toml bog-pine entry aligned to renderer SpeciesMap. Re-cook. Then a **256 m walkable
     test map** in the same mire → boot → send user the URL (with measured cook duration).
3. **CARPET layer + sphagnum (sammal)** — tasks #21/#26 (after cook-side). 2 Fable cushion/hummock
   meshes → opus implements the seam (declarative `CarpetSpec` + generalize 3 gates: unify the
   mirrored voxel-sibling gate → `voxelFarClass`; leaf reg uses `policy.channel` incl `'rigid'`
   non-wind; per-class `voxNear` handoff) + **all 6 opportunistic cleanups** (VEG_CLASS_COUNT const;
   mirrored gates; scatterGrid options object; CLASS_NAME colocate w/ enum; GroundCover.ts→Debris.ts;
   unify clsMaxDist defaults). Mandatory fly-over gate (2.4 m patch lattice = grass grid-from-altitude
   trap). Far = voxel-aggregate, near = sparse opaque cushions, beyond = terrain tint.

## PARKED behind the bog arc (user ordering)
- Veg-quality fixes: #22 sparse generic world shrubs, #23 pine needle tri-clumping, #24 ferns rebuild.
- #28 all-plants tri-budget audit — END of backlog, only after ALL bog work incl. sphagnum done AND
  user visual-confirmed.
- Tree-gen rework — `docs/tasks/2026-07-20/TREE-GEN-REWORK.md` (natural placement + age variation);
  after the bog arc, before other-regime microtopography.

## Working notes
- Live env: data server `:8790` (bog preview manifest), vite `:5199`. Bog preview manifest sha
  `f38a967ac83e1e4f` (build d07a78c5…). Ground URL base in `scratchpad/bog-veg/` specs.
- All mesh/geometry work = Fable only (user law). Briefs must NOT prescribe shape — give species +
  reject observations, model owns geometry. Leaf detail = real geometry, never a gradient blob.
- Full per-species specs + integration notes: `scratchpad/bog-veg/`.
