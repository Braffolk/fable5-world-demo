# Tree generation rework — two problems (saved 2026-07-19 for pickup ~2026-07-20)

> User law (2026-07-19, 22:15): after the current bog vegetation arc (incl. sphagnum/sammal)
> is done and visually confirmed, and BEFORE returning to microtopography on other regimes,
> do a quick pass on tree generation. It's "basically dogshit right now." Two distinct problems
> below. This doc captures the problems + known root-cause context; the design is open.

## Problem 1 — Unnatural placement / artificial single-species adult forests
**Symptom (user):** trees mostly follow strict large polygons, producing weird artificial forests
of a SINGLE tree type where every tree is an adult — no smaller/younger trees mixed in, no natural
species variation. It all looks a bit weird/uniform.

**Known root-cause context (from the trees.py placement map, this session):**
- Cook-side `asset-gen/src/assetgen/process/trees.py`: trees are detected as **nDSM canopy local-
  maxima** (Popescu-style variable-window peak finder) gated by a forest/shrub landcover mask
  (`FOREST_CLASSES = {1,2}`), with `MIN_TREE_H = 3.0`.
- **Species** is assigned by point-in-polygon join against **Metsaregister stand polygons**
  (`_assign_species`): the whole stand gets its DOMINANT species (`peapuuliik_kood`). → every tree
  in a stand becomes the same species = the "artificial single-type forest" the user sees. Trees
  outside stands get a coarse hash prior (pine/birch/spruce).
- Because detection is **nDSM-peak-driven**, only large canopy trees register → sub-canopy /
  younger / smaller trees are simply never placed → "all adults."

**Direction (open — design when we get to it):**
- Break the per-stand uniform-species assignment: use the stand's species MIXTURE / a natural
  intra-stand mix, not just the single dominant, so a stand reads as a mixed forest.
- Add sub-canopy / smaller trees BETWEEN the nDSM peaks (a natural understory-tree scatter), not
  only at canopy maxima — so forests have size structure, not just a mono-layer of adults.
- Keep it evidence-backed + deterministic (hash-driven, content-hash-stable chunks; keep the
  Morton sort). Natural placement, not decorative noise.

## Problem 2 — No tree AGE variation (all adults)
**Symptom (user):** no variation in actual tree age — they're all adults. We should have age
variants per tree already (yes — most trees have them).

**Known root-cause context:**
- The renderer already has an **age-form / ontogeny system**: `src/vegetation/AgeForm.ts`
  (`TREE_AGE_STAGES = 4`, `ageStageVariant(scale, dither)`), `Skeleton.ts` reshapes proportions by
  `inst.age`. VegLibrary bakes **4 age forms per species** (juvenile→veteran). So the VARIANTS EXIST.
- The cook drives age via the per-tree `scale` byte = `round(height / ref_height * 64)`, i.e. age
  is proxied from real nDSM height. Since only tall canopy peaks are detected (Problem 1), the scale
  bytes cluster high → the age-form selector `ageStageVariant` mostly picks mature forms → "all
  adults." The machinery is there; the cook just never feeds it a spread of ages.

**Direction (open):**
- Feed a realistic DISTRIBUTION of ages/sizes into placement (younger/smaller trees get lower
  `scale` bytes → the existing AgeForm juvenile/broadening/self-pruning forms render). Couples with
  Problem 1's sub-canopy trees (young trees ARE the small ones between the peaks).
- Verify the age-form variants actually exist + look right for the common Estonian species before
  relying on them (larch/oak confirmed have distinct forms; check the rest).

## Problem 3 — Dead / fallen / decaying trees — REPLACE the current fake "logs" with a PROPER system
**Ask (user, emphatic):** the current deadwood is unacceptable and must be REMOVED, not reused. The
existing `Log`/`Stump`/`Branch` debris pool is nonsensical — **featureless branchless cylinders (no
branches, no root plate, no decay), reading as human-cut piles of junk. NOTHING like a fallen, decaying
tree.** The defect is the PRIMITIVE ITSELF, not a lack of variance — a branchless cylinder is not a
tree, and no amount of random scaling/jittering makes it one (that would be the exact WRONG fix). A
"log" and an actual fallen tree are two completely different worlds. Real variety comes for FREE from
using actual different trees (real species/age/branch geometry), not from perturbing a placeholder. Rip out the current logs and build
a PROPER dead/fallen/decaying-tree system.

**What a proper fallen/decaying tree actually is (the target):**
- A REAL tree that has died/fallen — a full trunk WITH its branches, at the tree's real size/species
  (varied, not one scale), often with a **root plate / uprooted base** (windthrow) or a broken/snapped
  top, boughs still attached, lying at a natural angle, partly sunk into the ground.
- **Decay STAGES**: fresh-fallen (bark on, branches intact) → rotting (bark sloughing, limbs breaking,
  moss/fungi) → collapsing/crumbling (soft, moss-covered, sinking into the litter). Snags (standing
  dead) and broken stumps are part of the same continuum.
- Integrated with the LIVING tree gen: a fallen tree relates to the standing forest (its species/size
  come from the same population; it can be a former stand member), not a decoration scattered blindly.

**Direction (open — this is a real build, not a placement tweak):**
- Generate fallen/decaying trees from the SAME tree builder as living trees (reuse trunk+branch
  geometry), then apply a "felled/decayed" transform: lay it over, add root plate or snapped top,
  strip/break limbs per decay stage, add moss/rot material. NOT a standalone cylinder pool.
- Include standing snags (bleached dead pine especially in bogs) + broken stumps as decay states.
- Remove the old `Log`/`Stump`/`Branch` cylinder pool once the proper system replaces it (user
  excision law: dead code + assets go completely, no shims/tombstones). Check what still references it.
- Keep it evidence-reasonable + deterministic; deadwood is real forest structure, defining to the look.
- Mesh geometry = Fable (user law); this is procedural-modelling-heavy.

## Notes
- Both problems are primarily COOK-SIDE (`trees.py` placement + `scale`/species assignment), with
  the renderer age-form machinery already in place. Likely no/minimal renderer changes.
- Scope: "quick look" per the user — natural placement + age variation, not a full forestry sim.
- Do AFTER the bog arc is visually confirmed; do BEFORE other-regime microtopography work.
- Related: the bog-tree seed-variance / tri-outlier tightening (birch 28k→140k tris across seeds)
  flagged in the bog arc — a general "trees vary wildly in tri cost per seed" concern that the
  all-plants tri audit (task #28) will also surface.
