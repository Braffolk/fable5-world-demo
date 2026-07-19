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

## Open questions for the user (do NOT act without their call)
- **"Labrador tea" naming vs native status:** user flagged it as non-native/weird for Estonia and wants
  it removed from usage — but the modeled species is **Rhododendron tomentosum / Ledum palustre**
  (Estonian *sookail*), which IS native to Estonian raised bogs (a defining raba dwarf shrub). The
  misleading part is the common name ("Labrador tea" = the North American *R. groenlandicum*). Likely a
  RENAME (marsh Labrador tea / wild rosemary / sookail), not a removal. Confirm before removing a
  genuinely-native plant. Kept in the current wind/scale/grounding fixes meanwhile. (task tracker #32.)

## In-world bug batch (2026-07-19/20, user AFK — autonomous, all FABLE; opus NOT trusted here)
Reported after walking the populated bog preview. Fable diagnosis (IN-WORLD-DIAGNOSIS.md) root-caused all
six with cited code + the exact engine sway function; fixes dispatched file-disjoint. RESULTS:
- **Scale + grounding (2,3) DONE — commit `e6263b2`.** Per-class sizeFor curves (real-size) + sink
  0.005-0.01 (was a flat 3cm burying a 5cm cranberry). Zero perf, same instance count.
- **Wind (1a flex + 1b shrub-param flag) + sookail leaves DONE — commit `5720ff8`** (+ wind-preview
  tooling `e00087b`). Root: flex authored 0..1 per sub-object but engine sway uses flex as ~the whole
  amplitude for <0.5m plants → connected parts tore (cotton head detached/stretched). Re-authored flex
  as a plant-global monotone field anchored at each part's attach point; rigid sub-objects one constant
  flex. 1b: leaf channel hardcoded TREE wind params → shrub stems 1.8x faster than their leaves; added a
  free MESH_FLAG_SHRUB_WIND, trees byte-identical. Sookail leaves narrowed (were too wide = American look).
  Verified via a new `veg-preview --wind` animated filmstrip (before: heads fly off; after: coherent) +
  in-world burst. Cosmetics 6.1/6.2 folded in.
- **LOD ladder (4=5) DONE — commit `f692573`.** The "3D shadow-ray" speckle = mm-scale geometry rendered
  single-LOD → sub-pixel stochastic pixels. New BogLod.ts gives the 6 leaf pools a 4-rung prune-and-preserve
  ladder (λ 1.0/0.55/0.32/0.18, survivors widen ×1/λ) wired as pool.leaf.buildLadder like trees; LOD0
  byte-identical, net perf win. (Definitive moving-view speckle confirmation = a user visual gate.)
  NOTE: the LOD agent stalled in a boot-retry loop; orchestrator stopped it, verified the code (sound
  prune-and-preserve, tsc clean, LOD0 unchanged, 90fps), and committed.
ALL FIVE in-world fixes committed. Bog plants are colored, real-sized, grounded, wind-coherent, LOD'd.
Deferred (scope/perf, surfaced to user): clump/patch density (the "reads like a carpet" lever — costs
instances; ties to UNDER_STEP 2.4m ceiling), the fake deadwood LOGS on open bog (→ TREE-GEN-REWORK #3).

## NEXT (autonomous): moss / CARPET (#21/#26)
After LOD commits: sphagnum cushion mesh(es) (Fable) → CARPET seam + 6 cleanups (per GROUNDCOVER-LAYER-
ARCH.md) → fly-over gate. User note: moss + plants must COEXIST (plants grow IN the sphagnum), not
compete for scatter slots.

## Working notes
- Live env: data server `:8790` (bog preview manifest), vite `:5199`. Bog preview manifest sha
  `f38a967ac83e1e4f` (build d07a78c5…). Ground URL base in `scratchpad/bog-veg/` specs.
- All mesh/geometry work = Fable only (user law). Briefs must NOT prescribe shape — give species +
  reject observations, model owns geometry. Leaf detail = real geometry, never a gradient blob.
- Full per-species specs + integration notes: `scratchpad/bog-veg/`.
