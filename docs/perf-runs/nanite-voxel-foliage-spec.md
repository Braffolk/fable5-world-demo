# SPEC: 2-TIER FOLIAGE (mesh near + voxel mid→far) for the LAAS Nanite renderer

Status: DESIGN / DE-RISKING SPEC. No code lands until the Stage 0 kill-gates pass.
This is **not** a brochure. The conclusion "voxels are worth pursuing" was already
reached; this spec's job is to design the real thing AND surface where it dies. Every
load-bearing claim is anchored to `file:line` in the actual codebase. Where the original
draft mis-cited a constraint, this revision states the correction inline and re-derives.

All paths are relative to `/Users/sebastian/IdeaProjects/fable-demo2`.

---

## 0. HOW TO READ THIS / WHAT CHANGED FROM THE DRAFT

An adversarial 6-lens audit found the draft's *architecture* sound but its *stated
mechanism* wrong on several load-bearing details. This spec folds in every fix. The
corrections that an implementer MUST internalize before reading further:

1. **There is no free cluster-flag bit.** word7 = `triCount(0-7) | flags(8-9) | LOD level(10-15) | handle(16-31)` (`GeometryRegistry.ts:1231-1233`, verbatim comment + encode). The flags field is exactly 2 bits and both are taken (`CLUSTER_FLAG_HEIGHTFIELD=1`, `CLUSTER_FLAG_DAG=2`, `GeometryRegistry.ts:162-164`). The draft's `CLUSTER_FLAG_VOXEL=4` lands in bit 10 = LOD-level bit 0. **Resolution: do NOT carry the voxel decision on a cluster flag. Carry it on the MESH record** (a voxel sibling mesh with `matClass=voxel`), and derive "is voxel" from `matClass`, which the resolve already reads for free.
2. **`matClass=5` is `grass`, not free.** The enum is contiguous `terrain:0 … leaf:4, grass:5, debris:6` (`GeometryRegistry.ts:129-137`). matClass is byte 1 of mesh word6, an 8-bit field (`(w6>>>8)&0xff`, `GeometryRegistry.ts:448 / NaniteFetch.ts:701`), so **use `voxel:7`**.
3. **Depth-sort/bucketing is a FIRST-CLASS SPECCED COMPONENT — and the whole point of voxels.** The voxel-brick bin MUST rasterize bricks **front-to-back with a per-pixel early-depth SKIP** so an occluded brick does **not** pay the `atomicMax` election. **The win is FEWER BRICK WRITES than the overlapping leaf-cluster triangles that band would otherwise rasterize — it skips OCCLUDED brick rasterizations. This is NOT a fixed 1-write/pixel model:** a voxel may cover ONE or SEVERAL pixels (we ship deliberately COARSER-than-UE voxels, §3), so the saving is the occlusion-skip, not a per-pixel write bound. It is **not** "unbudgeted" and it is **not** the existing triangle `f2b` flag (which is unrelated — see §0.bis below). It is a concretely-designed pass (§6) that **PORTS the depth-bucketing mechanism we already built, MEASURED, and DELETED** in the `?tileproto` sort-middle TILED raster (`src/nanite/NaniteTileRaster.ts`, removed in commit `eecf046`, recoverable via `git show dfe6518:src/nanite/NaniteTileRaster.ts`; the B3/B1 work landed in commits `b037f4d`/`43e4aa1`/`fed6771`). That prototype rasterized triangles with: a per-entry depth bucket packed in the high bits of a flat per-tile list (`BUCKET_SHIFT=28`, `K_BUCKETS=8`, `KBITS=3` = top 3 bits of `depthKey24`), **K near→far passes** over the flat list (`bk = K-1 … 0`, larger depthKey = nearer = higher bucket), and a **per-pixel early-out that reads the current winner BEFORE the z-interp** (`nearKey > prevE` ⇒ process, else skip the dominant depth work). We re-use that exact mechanism, **with bricks as the binned primitive instead of triangles** (§6). For voxels the early-out is far stronger than it ever was for triangles, because bricks are **solid primitives that DO occlude each other** — see §0.bis. **THE SCAR: read §6.0 FIRST.** That same bin→K-pass mechanism *lost the frame* when we measured it on triangles, by a hard floor in the BIN/SETUP half. The entire voxel net-win rests on ONE unproven inequality. §6.0 leads with it; do not start implementing until you have internalized it.
4. **`matClass=5`-style routing is unnecessary.** Because the voxel sibling head carries its own `matClass` in `gpu.meshes`, the resolve's existing matClass derive (`NaniteResolve.ts:286-290`) routes voxels for free. The cull does NOT need a separate `qVoxel` queue *for the matClass test*; it only needs separate emission *if* we want a separate raster bin (we do — §6). See §4/§6 for the binding-cost tradeoff.
5. **The bit budget has exactly 1 portable spare bit.** At the default 128-tri cap the triangle id uses 30 of 32 bits (2 spare); at `?clustertris=256` it uses 31 (1 spare) (`NaniteCommon.ts` / `GeometryRegistry.ts:108-121`). The voxel namespace bit must be **bit31** (the only bit free at BOTH caps).
6. **THE COARSE one-sample-per-brick PATH IS THE HARD DEFAULT (not "recommended").** One `voxCz` per brick (AABB front-slab depth), brick-granular election id. It is load-bearing for THREE gates simultaneously — see §6.4. The per-cell ray-march refine opens a new precision-dependent shimmer surface and ships ONLY if KG-0b proves the coarse path too blobby, and ONLY with an explicit error bound.

### 0.bis — Why depth-skip lives ON THE VOXEL BIN ONLY (and NOT on the triangle raster)

The depth-skip is an overdraw win **only for solid voxel bricks**, and we do **not** test it on the triangle raster — there is nothing there to early-out. The existing `world1` triangle raster renders **extremely sparse, sub-pixel foliage triangles that do not occlude each other** (a leaf quad is ~1 px and the canopy is mostly holes). A front-to-back early-Z on those triangles has almost no fragment behind another fragment to skip; that is exactly why `?f2b=1` is OFF by default on `world1`, and the live measurement confirms it: **`?f2b=1` on world1 = WASH** (raster 10.55→11.01 ms; unordered scatter ⇒ the early-out's read+gate cost > its savings — `docs/perf-runs/2026-06-18-tiled-maximize.md:64`). This spec therefore **does NOT propose testing f2b on the mesh raster anywhere** (the old "0c cheapest-disproof" gate that did so is **DELETED**, see §11 — its replacement is the per-fragment voxel-vs-triangle estimate in §6.0/Stage-0a). The depth-skip only pays off against **opaque voxel bricks that genuinely overdraw each other** (whether a brick covers one or several pixels), which is the entire reason the mid→far band is voxelized in the first place. So: the depth-bucketed K-pass front-to-back is a property **of the voxel bin (§6)**, ported from the recovered tileproto mechanism, and the win is **first-class and designed** — but **conditional**, see §6.0.

---

## 1. GOAL + MEASURED GROUNDING (with honest noise)

**Goal:** 60 fps (16.6 ms). **Now:** ~43.8 ms p0.95 at 2268×1473 retina in the forest
(200k trees) under a moving camera. Gap ≈ 27.2 ms. Treat all numbers as big guidelines.
**Every perf gate in this spec re-baselines against this 43.8 ms p0.95** at canonical config —
there is **no separate f2b baseline** (f2b on world1 is moot, §0.bis).

**The overdraw signal (real, noisy):** at `instMinPx` 768–1024 the frame hits 90+ fps.
Per `NaniteCull.ts:398-402`, `instMinPx` is a per-instance screen **diameter** cull:
`sizePx = 2·projK·R / dist`, `returnIf sizePx < instMinPx`. With `projK = cotHalfFov·uH·0.5`
(`NaniteCull.ts:247`; fov=55, retina uH=1473 ⇒ projK ≈ 1415) and crown R ≈ 3.5 m, `instMinPx`
768 ⇒ cull at ~13 m, 1024 ⇒ ~10 m. So that lever does **not** carve out a tidy 90 m band —
**it deletes the entire forest past ~10–13 m**, replacing its cost with ZERO. This proves the
cost is overdraw from many small mid/far clusters, but it bounds the band cost **from below
only** (against zero), and says nothing about whether the *same* band rasterized as voxel
bricks is cheaper than as triangles. That comparison is unmeasured. → **Risk #2 / Stage 0.**

**The default operating point matters.** The shipping default `instMinPx = round(0.075·min(W,H))`
(`NaniteFrame.ts:181`) ≈ 110 px diameter ⇒ instances already culled at ~90 m for R≈3.5 m. So the
43.8 ms baseline and the 90+ fps datapoint are at *very different* cull settings. The voxelizable
band is precisely the distance range **between the default ~110 px cull edge (~90 m) and the
768 px lever (~13 m)** — and the cost distribution inside it is the thing Stage 0a must measure,
because the default already sheds the far end.

**Where the 60% cost lives (forest diagnosis).** The frame is **coverage / per-pixel-election
bound**: `docs/perf-runs/2026-06-16-forest-raster-diagnosis.md:17` attributes ~90% of the raster
to per-pixel coverage/atomic-election. The deferred vis-buffer did **not** make `atomicMax` cheap —
the election **IS** the cost. This is *good news for the thesis*: the early-skip targets exactly
this cost center (it skips the per-pixel election of occluded bricks). It is *bad news* in that the
bin that BUILDS the ordering is itself a coverage-class cost (§6.0).

**Second motivation (independent of perf):** the simplified crown-mesh LODs look bad even at MID
distance; voxels may preserve sparse-foliage silhouette better there. This is a *quality* claim
that survives even if the perf win is small — but it is gated by Risk #1 (the same crowns can
*bloat to a blob* if voxelized while leaves are still resolvable).

---

## 2. THE 2-TIER ARCHITECTURE

- **Tier 1 — FULL-DETAIL MESH (near):** the existing nanite mesh-cluster vis-buffer path
  (`raster.world1` + `NaniteResolve`), **UNCHANGED**. Leaves render two-sided via the per-mesh
  `MESH_FLAG_TWO_SIDED` (`GeometryRegistry.ts:160`) + in-place re-winding + camera-ward normal
  flip in the resolve (`NaniteResolve.ts:568-569`).
- **Tier 2 — VOXELIZED (mid → very-far, to the horizon):** the NEW subsystem. UE5.7
  Nanite-Voxels-style brick representation of the bounded crown palette, **rasterized** (not
  ray-marched — that is what dodges the 64-bit-atomic SVO dependency), reusing the existing
  32-bit `depthKey24<<8|id8` election. Goes ALL THE WAY OUT. The hope we are speccing is that
  voxels-all-the-way is sufficient.
  **WE USE VOXELS CLOSER TO THE CAMERA THAN UE5 DOES, and DELIBERATELY COARSER.** UE5 confines
  voxels to the far-field / sub-pixel regime and targets "1 voxel ≈ 1 px." We **reject that target**:
  under the WebGPU/browser raster budget we accept **a voxel covering SEVERAL pixels** (less detail
  on purpose) and **pull the mesh→voxel transition NEARER**. Both the voxel detail and the
  transition distance are **TUNEABLE knobs** with browser-budget defaults found by testing, NOT
  fixed constants (§3.2.bis). The #1 fine-shape risk (§3) is therefore **confronted by the
  detail/transition knobs + the conifer one-crown quality spike (KG-0b)**, NOT dodged by pushing
  voxels out past where the instance is already culled.
- **Tier 3 — IMPOSTORS:** NOTED FALLBACK ONLY (§13). Not designed here.

The two tiers are bound to the **same instance stream** (one transform per tree). Tier 1 is a
`leaf` mesh head; Tier 2 is a sibling `voxel` mesh head over the same instances, with the leaf
head's draw distance lowered to the transition and the voxel head's set to `TREE_GEO_FAR`.

---

## 3. THE MESH→VOXEL TRANSITION + VOXEL DETAIL (tuneable knobs; the fine-shape crux, Risk #1)

This is the **#1 risk** and the load-bearing knob set. Voxels preserve only the **aggregate
silhouette**; a sub-voxel leaf either **bloats** (binary occupancy fills the cell → canopy turns to
a blob) or **vanishes** (cell missed → holes). So at *resolvable* mid distance voxels can look WORSE
than mesh. The whole viability hinges on the two knobs below: **voxel DETAIL** (cell size /
bricks-per-cluster) and **TRANSITION DISTANCE**. Neither is a fixed constant; **neither targets
"1 voxel = 1 pixel."**

### 3.1 The rule (UE-confirmed) and why the draft's runtime formula was circular

UE switches per-cluster when `voxelError < triangleError` — i.e. once triangles are sub-pixel
(`research-analyses.txt`, §3). We do **not** copy UE's "voxel cell ≤ 1 px" sub-pixel target. The
draft proposed a runtime test `brickCellPx = projK·(2·s.radius/voxelGridDim)/distC`. The audit found
two fatal problems with it:
- **`voxelGridDim` is not in scope and not in the cluster record.** `readCluster` exposes only
  sphere/cone/triStart/triCount/flags/meshId (`GeometryRegistry.ts:396-413`); word6/word7 are full.
- **`s.radius` is the CLUSTER radius, not the leaf-feature size.** Gating on `cluster-radius/gridDim`
  collapses into the existing `minPx` screen-radius cull (`NaniteCull.ts:462`) rescaled by a constant.

**So both values become OFFLINE/uniform KNOBS, not a live cull math path** (§3.2 / §3.2.bis).

### 3.2 The corrected derivation — STARTING DEFAULTS for the two knobs, evaluated at build time

The bloat/vanish failure is driven by the **minimum leaf feature width**, not the crown radius.
Confirmed feature dims (`Species.ts`):

| species | leaf len (norm) | leaf width (norm) | crown scale | real blade size |
|---|---|---|---|---|
| beech (broadleaf, densest) | 1.0 | **0.42** | [0.16, 0.24] (`:151,157`) | ~0.16–0.24 m long, **~0.07–0.10 m wide** |
| oak/maple/etc broadleaf | 1.0 | 0.5–0.55 | [0.10, 0.16] (`:204,210`) | ~0.1 m long, **~0.05–0.09 m wide** |
| **spruce** (needled) | 0.1 | **0.024** | [0.22, 0.35] (`:53`) | needle **~2.4 cm wide**, single-quad (`LeafMesh.ts:128-167`) |
| **pine** (needled) | 0.21 | **0.018** | [0.26, 0.42] (`:104`) | needle **~1.8 cm wide**, single-quad |

A blade ~0.07 m wide, or a needle ~0.02 m wide, is **sub-cell** for any affordable voxel cell
(a 16³ grid over a 3.5 m crown = ~0.22 m cells; even crownDiam/32 ≈ 0.11 m). So at the distance
where the *cell* is 1 px the *blade* is already far sub-pixel — which is exactly the regime where
aggregation is invisible and correct, and *nearer* than that is the danger zone.

**Closer-than-UE STARTING DEFAULTS (these are seeds for the §3.2.bis tuning, NOT fixed targets).**
Because we deliberately voxelize NEARER than UE, we are willing to push the transition into the
resolvable-leaf danger zone — so we **buy back fidelity by making the grid FINER**, so a leaf spans
*several* cells rather than one (multi-voxel leaf → silhouette survives instead of bloating to a
blob). The DEFAULT transition is set so that **a CELL is ~1 px AND a leaf is ≥2–3 cells across** at
that distance. Concretely (beech, featureWorld ≈ 0.07 m, crownDiam ≈ 5 m, projK ≈ 1415):
- a leaf must span ≥2–3 cells ⇒ `cellWorld ≤ featureWorld/2.5 ≈ 0.028 m` ⇒ `voxelGridDim ≥ 5/0.028 ≈ 180`,
  i.e. an effective grid in the **128³-effective class** for the whole crown (vs UE's far-field 16–32³).
  Per the §5.3 brick cap this is realised as **finer (≤128) clusters each carrying a ≤128-brick
  4×4×4 block**, NOT one 256³ block per crown — the fine grid is distributed across the cluster fan-out.
- the cell-≈1px clause then sets the DEFAULT `transitionDist ≈ projK · cellWorld / 1px ≈ 1415 · 0.028 ≈ 40 m`
  — i.e. **NEAR field (tens of metres), inside the default `instMinPx` envelope**, exactly where
  the perf overdraw lives, NOT the 225–440 m far-field band a coarse grid would give.
- **Conifer (the worst case):** needle ≈ 0.02 m is finer than even a 0.028 m cell, so the
  "leaf ≥2–3 cells" clause cannot be met without an unaffordable grid; here the SURVIVAL mechanism
  is **mandatory fractional-density voxelization** (§5.4) so the needle becomes a low-density
  several-cell smear instead of a hole or a blob. This is precisely what KG-0b (§3.4) validates on
  spruce/pine at the chosen near `transitionDist`.

The DEFAULT transition distance is the **nearer** of (cell-≈1px) and (leaf-≥2–3-cells-resolvable):

```
featureWorld   = min leaf width (per species, table above)
cellWorld      = crownDiameter / voxelGridDim     (voxelGridDim = brick-grid edge count, a per-crown BUILD KNOB, §3.2.bis)
# STARTING DEFAULT only — choose voxelGridDim so a leaf spans ≥2–3 cells (cellWorld ≤ featureWorld/2.5),
#   then place the transition where the CELL is ~1px. The SHIPPED values come from tuning (§3.2.bis):
#   ratchet voxelGridDim COARSER until just before KG-0b breaks. NO "1 voxel = 1 px" target anywhere.
transitionDist = projK · cellWorld / 1px           # DEFAULT ~40 m for beech, inside instMinPx — a knob
```

The runtime cull does **not** evaluate any `brickCellPx`; it uses the existing **per-mesh
draw-envelope distance** (`lodDist`, `NaniteCull.ts:391-394` / `setMaxDistance`) — the leaf head's
max draw distance is lowered to `transitionDist`, the voxel head's `lodDist` starts at
`transitionDist`. **The transition is a per-mesh distance handoff, identical in mechanism to the
existing LOD envelope, not a new per-cluster math path.** This sidesteps the "voxelGridDim has no
home in the cluster record" blocker entirely.

### 3.2.bis BOTH KNOBS ARE TUNEABLE UNIFORMS WITH BROWSER-BUDGET DEFAULTS — NOT a "1 voxel = 1 px" target

The two values above are **knobs found by TESTING, with browser-budget DEFAULTS — not hardcoded
constants and NOT UE's sub-pixel "1 voxel ≈ 1 pixel" goal.** We **explicitly reject** that goal:
under the WebGPU/browser raster budget we deliberately accept **COARSER voxels — a voxel may cover
SEVERAL pixels** (less detail on purpose), and place the transition **NEARER** than UE (UE =
far-field only). The two knobs:

- **`voxelGridDim` (voxel DETAIL = cell size = bricks-per-cluster)** — a per-crown BUILD knob
  (palette-indexed table on the voxel sibling mesh, exposed as a tuneable, e.g. `?voxgrid=` / config
  override). It is per-crown (broadleaf vs conifer differ). **DEFAULT = the COARSEST grid that still
  passes KG-0b** (the §3.4 conifer one-crown spike), NOT the finest the math allows. The §3.2
  formulae compute a *starting default* (128³-effective class, distributed across ≤128 finer
  clusters each carrying a ≤128-brick 4×4×4 block — NOT one 256³ block/crown); tuning then ratchets
  it **coarser** until just before quality breaks. The procedure: **find the coarsest detail still
  acceptable at the chosen NEAR transition.**
- **`transitionDist` (the mesh→voxel handoff distance)** — a per-mesh draw-envelope uniform
  (`setMaxDistance`, §3.2). **DEFAULT = NEARER than UE** (≈40 m for beech, inside the `instMinPx`
  envelope), swept in Stage 5 (`?voxnear=` / config).

**Detail couples to the raster (important).** The recommended/HARD-default DETAIL setting is the
**COARSE one-sample-per-brick** path (§6.4): one `voxCz` per brick (= AABB-slab/ray-entry NDC depth)
makes the election id **brick-granular (~7 bits)**, dissolves the 13-bit per-cell collision
(§4.5/KG-ID-BUDGET), drops `qVoxelRO` toward a 9-buffer resolve (§4.6/KG-BUFFER), and makes
`brick.nearestZ` exact (the AABB front-slab) so the early-skip proof holds (§6.3). The 64-cell
occupancy is then BUILD/raster-only, not resolve-visible. Refine to per-cell ray-march **only if
KG-0b demands finer detail**, and only with an explicit depth-quantization error bound (§4.5b).

These knobs are how Risk #1 (fine-shape) is MANAGED rather than dodged. If NO `(voxelGridDim,
transitionDist)` setting looks acceptable at a useful (near-enough) distance before the band overlaps
where `instMinPx` already culls, **that is the kill-gate** (§3.4 / KG-0b). Nowhere does this spec fix
a "1 voxel = 1 pixel" target.

### 3.3 The numeric window — why coarse far-field fails and the closer-than-UE FINE-default we ship

A COARSE far-field grid (the UE-style choice we reject) puts the band uselessly far:
- For **beech** (featureWorld ≈ 0.07 m, crownDiam ≈ 5 m): 16³ ⇒ cellWorld ≈ 0.31 m ⇒
  transitionDist ≈ **440 m**; 32³ ⇒ ~225 m; 64³ ⇒ ~110 m.
- For **spruce** (featureWorld ≈ 0.024 m): feature is sub-cell at any of these ⇒ same ~225–440 m, and
  the needle would **bloat to a blob** without fractional density.

**Why coarse fails — and why we DON'T default to it:** the DAG already collapses crown detail toward
its root by `tau≈3 px` (`NaniteFrame.ts:152`, cut at `NaniteCull.ts:450`) and the default `instMinPx
≈ 110 px` culls the whole instance at ~90 m. So a **coarse** voxel band would sit ENTIRELY past where
the instance is already culled ⇒ zero win. UE can live there because UE only wants sub-pixel
far-field aggregation; **we do not.**

**The closer-than-UE default (correction 2):** the DEFAULT detail pulls the transition NEAR by going
FINE. A grid in the **128³-effective class** (distributed across ≤128 finer clusters, §5.3) gives
cellWorld ≈ 0.028 m ⇒ **transitionDist ≈ 40 m for beech** — inside the `instMinPx ~110 px` / ~90 m
envelope and squarely in the overdraw band (90 m → ~13 m at the 768 px lever, §1). The price — more
bricks/cluster, smaller cells, more brick memory (§5.3) AND a wider doubled-work cross-fade band in
the highest-overdraw zone (§6.5 / §7.3) — is the deliberate spend, and the bloat/vanish risk is
**confronted** by fractional density + the conifer KG-0b spike, **not** dodged by retreating far. The
single remaining fine-shape gate is KG-0b: at the chosen near `transitionDist` + tuned grid, does the
worst-case conifer crown survive (no blob/bald) and read ≥ the simplified mesh-LOD? If yes, the win
is near; if no, ratchet the transition out one notch (coarser) and re-judge, killing only if it never
passes before the band overlaps the cull edge.

### 3.4 KILL-CRITERION (Risk #1 — KG-0b, the single fine-shape kill)

At the candidate `transitionDist` for the chosen `voxelGridDim`, the voxel crown must look
**better-or-equal to the simplified mesh-LOD it replaces** AND **thin leaves must survive (no blob,
no bald)** in side-by-side judge-shots. The spike **MUST test SPRUCE/PINE FIRST** (single-quad
needles, the worst case AND among the most numerous species, `Species.ts:53,104,307`), **before**
the ID-bit fork is locked (§4.5) — the worst-case needle is what forces fractional-density
voxelization and the grid choice the brick budget depends on. **Testing the easy broadleaf (beech
0.07 m) and shipping the verdict for the hard conifer is the optimistic move FORBIDDEN in writing.**
The closer-than-UE fine grid keeps the transition NEAR (inside `instMinPx`), so the "band past the
cull edge ⇒ ZERO win" failure is engineered against — but if even the finest affordable grid cannot
pass KG-0b before the transition must be pushed out past the cull edge, the win is **ZERO** → **KILL
for our content.** (Stage 0b.)

---

## 4. VOXEL DATA STRUCTURES + PER-STAGE BUFFER ACCOUNTING (≤10)

### 4.1 No cluster flag. The voxel decision rides the MESH record.

Because there is no free cluster-flag bit (§0.1), voxel clusters are simply the clusters of a
**voxel sibling mesh** whose mesh-record `matClass = voxel(7)`. The resolve already derives
matClass from `gpu.clusters[ci].word7>>16 → meshId → gpu.meshes[meshId].word6` (`NaniteResolve.ts:286-290`).
Nothing in the cluster record changes. (Optional: if a per-cluster brick decode needs `brickBase`,
reinterpret word6=`triStart`→`brickBase` and word7-lowbyte=`triCount`→`brickCount(≤128, fits u8)`
**only on voxel meshes**, leaving the cluster mega-buffer size/stride untouched. Safe because voxel
clusters are never fed to the triangle raster.)

### 4.2 New buffer `gpu.voxelBricks` (the 11th RegistryGpu field)

`RegistryGpu` has exactly 10 fields today: verts, hfVerts, indices, clusters, meshes, instances,
instanceMesh, dag, vcompact, dagLinks (`GeometryRegistry.ts:783-808`). Add `voxelBricks:
StorageBufferNode<'uint'>` as the 11th. **It must NOT be bound on every stage** — only the voxel
raster permutation and (a slice of it) the resolve. The DAG sidecar (`DAG_WORDS=12`, kept separate
specifically "to keep the cut kernel ≤10 storage bindings (F9)", `GeometryRegistry.ts:78`) is the
precedent that the ceiling is managed by NOT binding everything everywhere.

### 4.3 Brick encoding

4×4×4 = 64-cell occupancy. **No u64 in WGSL r184** ⇒ `2×u32` (lo/hi), bitwise ops on the two words.
Per occupied cell: density 1 B + albedo RGB 3 B + a normal descriptor. **The normal descriptor size
is DECIDED in Stage 0b BEFORE the buffer is sized (§5.4):** mean-normal (oct snorm2x16, 4 B) if the
cheap path ships, or SGGX (6 B) if the stochastic fallback is needed. Sizing the frozen `addLate`
reservation to the *winner* avoids baking dead bytes.

Two layout forks (a real decision, not yet made):
- (a) per-brick occupancy `2×u32` + packed per-occupied-cell attribute stream (compact, needs a
  prefix-sum at build);
- (b) per-brick fixed block: occupancy `2×u32` + 64 attribute slots (simpler addressing, more
  memory). A 128-brick cluster ≈ 2–2.5 KB at 4-byte attrs (per-cell budget pending §5.4).

### 4.4 Resolve-visible subset (helps the resolve ceiling)

Occupancy is **raster-only** (resolve never re-tests it). Per-voxel **color rides the existing leaf
per-species tint** (mesh word7 `matParam`, `NaniteResolve.ts:544-560` / `WorldRegistry.ts:426
packLeafTint`) — UE per-voxel color is UNCONFIRMED, and reusing matParam saves a buffer. Resolve then
needs only **one** new buffer slice (the normal descriptor + any metadata, packed into a single
buffer). **With the coarse one-sample-per-brick default, color is brick-mean and even the per-cell
attribute stream is raster-only.**

### 4.5 ID / payload bit budget (tightest shared raster+resolve constraint)

The election word `visPayloadV` is full: `depthKey24(24) | id-tiebreak(8)` (`NaniteRaster.ts:810-813`).
The full id is plain-stored into `visBV` by the winner (`NaniteRaster.ts:830`). Triangle id =
`itemIdx(23) << CLUSTER_TRI_BITS | localTri(7@128 / 8@256)` = **30 bits @128-cap (2 spare) / 31
@256-cap (1 spare)**. **The only bit free at BOTH caps is bit31** — the voxel marker. The voxel id
must fit **≤30 bits** to survive the 256-cap.

`128 bricks × 64 cells = 8192 cells = 13 bits`, which collides with a 23-bit `voxItemIdx` inside
30 bits. **FORK (decide in Stage 1, AFTER 0b fixes the grid):**
- (a) **Cap voxels-per-cluster ≤512** (coarser bricks, ≤8 bricks/cluster) so 9 bits suffice ⇒
  `voxItemIdx(21) | cell(9)` ≤ 30 bits. Simple, deterministic resolve decode.
- (b) **Elect at BRICK granularity** (~7-bit brickIdx) and have the resolve re-derive the winning
  cell from the reconstructed `wp` vs the brick origin. This is a **new precision-dependent failure
  surface**: `wp` is reconstructed from the 24-bit election depth key quantized to 1/16777215 of the
  depth range (`NaniteResolve.ts:273-280`); at the transition where a cell projects ≤1 px the depth
  band can straddle multiple cells ⇒ the re-derived cell is non-deterministic frame-to-frame ⇒
  **shimmer**. **No error bound is offered; (b) is taken ONLY if (a)'s coarser bricks fail KG-0b, and
  ONLY with that error bound written first.**

**Coupling to §6.4 (THE DEFAULT, load-bearing for THREE gates):** the **coarse one-sample-per-brick**
raster path (§6.4) is the **HARD Stage-1/2 default** (correction #6, not "recommended start"). With
it, per-cell occupancy is *unused at the only distance voxels are allowed*, the id is **brick-granular
(~7 bits, trivially <30)**, the KG-ID-BUDGET fork **disappears**, `qVoxelRO` drops (resolve → 9), and
`brick.nearestZ` is **exact** so the early-skip proof (§6.3) holds without a depth-quantization
hazard. The dataStructures bake this in: occupancy is build/raster-only, id is brick-granular.

### 4.6 Per-stage storage-buffer accounting

Textures do NOT count against the ~10 storage-buffer/stage ceiling (heights/GI are textures:
`NaniteResolve.ts:636-638`; ProbeGI `irradiance()` reads a `Storage3DTexture`, `ProbeGI.ts:369-371`
— **0 storage buffers**, verified; flag a re-verify gate if a future SH-buffer read is added).

**RESOLVE FRAGMENT STAGE — today = 8 storage buffers** (grep-confirmed):
`vis.payloadV` (`:269`), `vis.visBV` (`:268`), `cull.qRasterRO` (`:189,284`), `gpu.clusters` (`:286`),
`gpu.meshes` (`:287`), + `gpu.verts` + `gpu.indices` + `gpu.instances` via `makeFetch(…, false)`
(bindHfVerts=false, `:206` / `NaniteFetch.ts:147-171`). NOT bound: hfVerts/dag/vcompact/dagLinks.

After voxels — **with the coarse one-sample-per-brick default the TARGET is 9** (the preferred path):
- **+9** `gpu.voxelBricks` resolve view (normal descriptor only; occupancy raster-only; color via
  matParam → NO color buffer). `qVoxelRO` is **DROPPED** because the brick-granular id packs
  `(instId, brickIdx)` directly in the ≤30 id bits, so the resolve decodes them from `pRaw` with no
  queue lookup, and the voxel sibling's matClass already comes via `gpu.meshes`.
- **= 9 of 10. Margin of 1.**

If the per-cell refine is ever forced (KG-0b only):
- **+10** `cull.qVoxelRO` (to recover the winning brick's cell). **= 10 of 10, ZERO margin.**

**Honesty:** WGSL binds every *referenced* buffer in the fragment shader regardless of the `isV`
branch, so a single resolve pays for `voxelBricks` (and `qVoxelRO` if present) on **every pixel
including pure-terrain frames**, and any future debug/audit buffer slipped in silently busts 10. The
9-buffer target is contingent on THREE unproven packing assumptions (matParam color, raster-only
occupancy, single metadata buffer) AND the brick-granular id. **FALLBACK (make this the Stage-2
DEFAULT until the assumptions are proven by binding audit):** a **separate voxel-resolve fullscreen
pass** keyed on bit31 (two draws, each <10). This is the guaranteed-safe escape.

**RASTER STAGE (tighter). Why a SEPARATE voxel permutation has headroom:** world1's fetch set is
`gpu.verts/hfVerts/indices/clusters/meshes/instances` via makeCtx/fetchWorldVert (`NaniteFetch.ts:147-171`)
plus `visDepthV/payloadV/visBV` + `qRasterRO` + `hwQueue` (+`auditV`). The voxel bin **reads bricks,
not triangle verts/indices/hfVerts** — so it is a separate permutation that **drops
verts/indices/hfVerts (frees 3)**. This is *why UE bins voxels separately.* Per §6.6 the two voxel
kernels split by access type (the tileproto lesson, `NaniteTileRaster.ts:223-255`):
- **`kVoxBin`** binds: `gpu.clusters, gpu.meshes, gpu.instances, gpu.voxelBricks, qVoxRaster (in),
  atomicBuf (tile-count cursors), dataBuf (flat list out)` = **7**.
- **`kRasterVox`** binds: `gpu.voxelBricks, qVoxRaster, atomicBuf (ro counts), dataBuf (ro flat list),
  visPayloadV, visBV` = **6** (NO `hwQueue` — bricks are small, no HW spill; NO exact `depthV` —
  world1 already avoids it, `NaniteRaster.ts:825-829`).
- Both **≤10 with margin (~8 worst case)**. **Plausible but UNVERIFIED until written** — a Stage-2
  hard binding-audit gate, not an assumption.

**CULL — do NOT touch kTraverse.** The cut (`NaniteCull.ts:450`) lives **inside** `makeTraverse`
(`kTraverseAB/BA`, `:418-543`), the live hierarchical pass, **deliberately pinned at ≤10 buffers** by
recycling counter slots 2/3 and 0/4 "so kTraverse stays ≤10 storage buffers WITH the HZB occlusion
read" (`NaniteCull.ts:353-359`). Adding a `qVoxel` buffer+counter there **busts the ceiling and there
are no free counter slots in hier mode.** → Routing is a per-mesh distance handoff (§3.2), so the cut
emits voxel clusters into the SAME `qRaster` as triangles. A **tiny post-traverse FAN-OUT pass** then
reads emitted `qRaster`, tests the cluster's mesh matClass, and fans voxel entries into `qVoxRaster`
— a **2–3-binding kernel** (`qRasterRO` in, `qVoxRaster` out + counter), NOT inside kTraverse. **This
second pass is the PRIMARY design, not a fallback.** Its cost (one re-scan of up to QRASTER_CAP
entries) is a Stage-0a/Stage-3 line item, not free.

---

## 5. OFFLINE BUILD PIPELINE (palette voxelization, error-metric, normal distribution)

### 5.1 Target = the bounded palette (memory thesis SURVIVES)

Trees instance from a small palette; per-tree geometry is NOT unique — instances carry only transform
`A=(x,y,z,scale), B=(yaw,leanX,leanZ,idF)` (`GeometryRegistry.ts:18`) + a per-vert hue jitter. Crowns
are deterministic from `seed.rng(label)` (`VegLibrary.ts:145,247`). Palette size: **5 leafy species**
(6 `TREE_SPECIES`, `Species.ts:307`, minus SNAG `foliage:null` `:298`) **× `TREE_VARIANTS=4`**
(`Scatter.ts:82`) = **≤20 crowns**, + ~3 shrubs + ferns ×4 + flowers (`VegLibrary.ts:344-448`) ≈
**~36 unique meshes**. Per-tree-unique voxelization is **400 GB–12 TB** (`research-analyses.txt §6`)
and is **avoided**. **Verdict: bounded, ~tens of MB — but the exact number is unquantified until §5.3
runs.**

### 5.2 Build hook

In `buildVegLibrary` after `pool.leaf` is produced (`VegLibrary.ts:241-262` builds the crown;
`WorldRegistry.ts:415-432` registers it). Add `registerVoxelMesh(brickSource, 'voxel'(=7), {same
matParam / twoSided / aggregate})` bound to the SAME instance stream; `setMaxDistance(voxelHead,
TREE_GEO_FAR)` (`WorldRegistry.ts:429`) and **lower the leaf head's max distance to `transitionDist`**
(§3.2). Voxelize at **FULL anchor density** (`skel.anchors`), NOT the `meshAnchorTarget` render diet
(`VegLibrary.ts:259`, e.g. beech 2200, birch 4000) — so thin leaves are not pre-decimated.

### 5.3 Brick budget — QUANTIFY BEFORE RESERVING (addLate freezes at build) — HARD Stage-1 precondition

`addLate` caps freeze at `build()` (`GeometryRegistry.ts:915-921, 1020-1028`); under-reservation
throws, over-reservation wastes resident memory. **`LateBudget` has NO `bricks` field** (it is
verts/hfVerts/tris/clusters/meshes/instances, `:917-922`) — a `bricks` field + buffer-sizing path
**must be ADDED, and the brick-granular normal-descriptor size (mean 4 B vs SGGX 6 B) must be DECIDED
in Stage 0b BEFORE this reservation freezes** so no dead bytes are baked in. Before reserving: Stage 1
must emit the per-crown cluster count in the levels **above** the transition (DAG stats exist:
`lod0Clusters/totalClusters/levels/roots`, `BuildAggregateDag.ts:613-618`), multiply by the §5.4
per-cluster byte size, and report total MB per palette. The §0.3 "11th buffer fits" only argues the
*binding-count* axis; the **SIZE axis is unproven until this number exists.**

**The per-coarse-cluster fit proof (HARD Stage-1 gate, not an assumption):** a whole-crown coarse DAG
cluster aggregates thousands of leaves; **PROVE it fits ≤128 4×4×4 bricks (= a 16–32 voxel cube) at a
cell size that passes KG-0b** — or flag finer-clusters (more voxel clusters → re-run the budget) or
coarser-cells (worse silhouette → KG-0b risk) as an **open risk that re-runs the budget**. State the
per-coarse-cluster `voxelGridDim` explicitly.

### 5.4 Voxelizer (offline, cost-tolerant)

1. **Dominant-axis ortho rasterization** (X/Y/Z by `max|N.component|`). **WGSL BLOCKER:** no geometry
   shader, no HW conservative raster (`research-analyses.txt §2/4`) ⇒ emulate via software triangle
   dilation (~0.5 voxel in clip) in a compute prepass OR multi-axis (3/6-dir) MSAA union. Offline so
   tolerable, but **fragile/slow — the dominant build-quality risk** (a hole baked into a thin-leaf
   crown is a permanent artifact, not a runtime tunable).
2. **Thin-leaf hole-free:** 6-separating (gap-free) or 26-separating (conservative, no holes, more
   bloat) Akenine-Möller SAT tri/box test. **Needles** (single-quad, `LeafMesh.ts:128-167`):
   conservative **capsule voxelization** (Dynamic Line Sets, `research-analyses.txt §4c`).
3. **Fractional opacity, NOT binary** (the bloat/vanish fix, MANDATORY for needles): supersample
   (MSAA 8–16× or multi-axis union), store `density = covered/total`; density-weighted color =
   `Σ(color·cov)/Σcov`. **Our leaves are OPAQUE modeled geometry, no alpha mask** → the VoxelPipe
   alpha-test hook is **UNNEEDED** — a real simplification vs UE.
4. **Per-voxel normal descriptor:** for the **mean-normal cheap path** (recommended, §7) store one
   oct-encoded MEAN normal per occupied cell (or per BRICK in the coarse default). For the **SGGX
   fallback** store one SGGX microflake (6 B) per cell. **CAUTION (bend-normal variance trap):**
   `MeshGrower.bendNormals` (`TubeMesh.ts:51`) bends per-vert normals toward the crown sphere — reuse
   the BENT normals as the MEAN but fit any SGGX VARIANCE from RAW pre-bend normals or the spread is
   wrong. Moot if the mean path ships.

### 5.5 Error-metric / transition flag (set OFFLINE)

Per palette crown, compare the simplified-mesh error vs the voxel-representation error at candidate
distances and set the crown's `transitionDist` (§3.2). The runtime reads only the per-mesh draw
distance — **no live voxel-vs-tri error compute** (the cull has no budget for it).

### 5.6 Build cost + cache (was glossed)

The leaf aggregate DAG ALONE already costs ~0.8 s/crown synchronously at boot and
`WorldRegistry.ts:725-726` warns it is **already at a budget threshold**. Voxelization +
supersampled conservative raster + (optional) SGGX fit is **added on top**. **State the added
per-crown ms and the thread** (main / worker / time-sliced). **There is NO foliage cache today** —
the IndexedDB `DagCache` is terrain-only (`DagCache.ts`); leaf DAGs rebuild every boot. **Spec a
persistent voxel cache** (copy the DagCache IndexedDB + version pattern, keyed on seed+crown+version)
or every cold boot pays voxelization on top of the ~16 s leaf-DAG build.

---

## 6. RUNTIME RASTER INTEGRATION (voxel-brick bin, depth-bucketed front-to-back, reusing the 32-bit election)

### 6.0 LEAD WITH THE SCAR — the entire net-win rests on ONE unproven inequality

**Read this before §6.1.** The depth-bucket K-pass below is ported from the `?tileproto` sort-middle
TILED raster — **a mechanism we already built, MEASURED, and DELETED.** The scar (do not bury it):

- `eecf046` commit body: the `?tileproto` tiled raster measured **+11.7 / +18.1 ms slower** than the
  scatter `world1` path; "the frame is **coverage-bound, not submit-bound**."
- `docs/perf-runs/2026-06-18-tiled-maximize.md:57-65` quantifies WHY: the tiled **RASTER half
  actually WON** (8.85 vs 10.55 ms p50) — but the **BIN/SETUP half** (sort-middle transform →
  xtri-store → re-read, which scatter fuses away) cost **8.65 ms and LOST THE FRAME** (tiled 25.1 vs
  world1 17.7 ms).

**`kVoxBin` IS that setup half** (project AABB + bucket + scatter into the flat per-tile list). So:

> **The whole voxel net-win rests on ONE inequality: `brick-bin cost ≪ saved elections`.** It is true
> ONLY IF (i) the brick count per tile is **far below** the 388k-cluster triangle cut, AND (ii) the
> bricks overdraw enough that the skip pays. (i) is *plausible* — the coarse brick band has far fewer
> primitives than the triangle cut — and it is **the one thing that can save this**. But it is
> **ASSERTED, never measured**, and it is **the precise cost that already sank the only working
> prototype of this mechanism.** This is not a logic hole (KG-NET-WIN guards it correctly); it is an
> **unmeasured bet against directly adverse prior evidence.**

**Consequence for the plan:** the **FIRST Stage-0a measurement** is a back-of-envelope (or throwaway
fixed-distance spike) of **fragments-per-crown(voxel, coarse 1-sample-per-brick) vs
fragments-per-crown(triangle)** at the transition — the *cheapest disproof of the real thesis*. It
**replaces the deleted f2b "0c" gate** as the genuine cheapest disproof. If brick fragments don't
beat triangle fragments by a margin that survives the bin cost, STOP before any raster code.

What is NOT at risk (do not re-litigate): the early-skip is **correct and loss-exact** (§6.3), it is
**fully TSL-r184-expressible** (§6.7), and both raster permutations **fit ≤10 buffers** (§4.6/§6.6).
The bet is purely the bin-cost-vs-saved-elections magnitude.

### 6.1 Depth-bucketed front-to-back is a DESIGNED, FIRST-CLASS component (the whole point)

The voxel-brick bin **rasterizes bricks front-to-back with a per-pixel early-depth SKIP** so an
occluded brick does NOT pay the `atomicMax` election. This is the overdraw win and **the entire
reason to voxelize** the mid→far band. The win is **FEWER brick writes than the overlapping
leaf-cluster triangles** that the same band would otherwise rasterize: unlike the sparse sub-pixel
foliage *triangles* (which do not occlude each other — §0.bis, why `f2b` on `world1` is a WASH and
the old "0c" gate is DELETED), voxel bricks are **solid primitives that genuinely overdraw each
other**, so the depth-skip has real fragments to eliminate. **A voxel covers ONE OR SEVERAL pixels —
this is NOT a fixed 1-write/pixel model;** the saving is that occluded BRICK rasterizations are
skipped entirely, not that every visible brick collapses to a single pixel write. (The prior-art
itself — `IDEAS.md` B3 — warns the triangle early-out "may exceed savings on ~1px tris"; bricks are
**not** ~1px sparse tris, so there are real fragments to eliminate. That distinction is this
mechanism's strongest, most-defensible point — but it is bounded by §6.0.)

We do **not** re-invent the mechanism — we **PORT it from the prototype** (the sort-middle TILED
raster, `git show dfe6518:src/nanite/NaniteTileRaster.ts`; depth-bucket work in `b037f4d` (B3
ordering), `43e4aa1` (B1 on-chip per-tile depth election), `fed6771` (B1 tiled base)). The production
`world1` triangle path is **unchanged** (Tier 1 near-field); the depth-skip is a property of the NEW
`kVoxBin`/`kRasterVox` bin only.

### 6.2 `kVoxBin` — the BINNING pass (ported from tileproto `kSetup`, `dfe6518:NaniteTileRaster.ts:281-456`)

**One thread = one BRICK work-item** read from `qVoxRaster` (NOT a triangle from `qRasterRO`). Per
brick:
1. **Prime the brick's cluster ctx** via the existing wgcache broadcast (mirror `kSetup`'s
   `primeCtx`, `:290`).
2. **Project the brick AABB** — its 8 corners through `cam.vp` — take the screen bbox + the
   **NEAREST NDC z** across corners.
3. **BUCKET KEY** (verbatim from tileproto `:410-415`):
   `bucket = min(depthKey24(nearestNdcZ) >> (24 - KBITS), K_BUCKETS-1)` with **`K_BUCKETS=8`,
   `KBITS=3`** (`:118-119`) — the top 3 bits of the 24-bit `depthKey24`; **LARGER depthKey = NEARER =
   HIGHER bucket.**
4. **SCATTER** (verbatim `:416,422-435`): `entry = payload | (bucket << BUCKET_SHIFT)`,
   **`BUCKET_SHIFT=28`** (`:124`), where `payload` = the **brick work-item index** into `qVoxRaster`
   (this replaces tileproto's `payload = itemIdx<<CLUSTER_TRI_BITS | localTri`). Scatter that single
   **1-word** entry into **each 16×16 tile the AABB bbox overlaps**, via the per-tile `atomicAdd`
   cursor into a flat per-tile list (`FLAT_TILE_CAP` slots, one list per tile; **atomicBuf** counts
   vs **dataBuf** list, split by access type, `:233-255`).

**The SAME build-time bit-budget assert applies** (`:146-159`): `QVOX_CAP-1` must be `< 1<<28`
(268M — trivially true for brick work-items), and `BUCKET_SHIFT + KBITS ≤ 32`.

**NO per-bucket cap** — a dense single-bucket tile uses the full flat headroom, so the tileproto's
~10% dense-canopy per-bucket drop **cannot fire** (`:108-117`). The brick's full attribute block
(occupancy `2×u32` + AABB origin/extent + normal descriptor) is read **ONCE** in the fine pass via
the work-item index; the flat list carries only the 1-word bucket-tagged entry.

**NEW pre-seed (over the verbatim port).** After `kRasterWorld1`+`hwRender` have run, the near-field
triangle winners are already in global `visPayloadV`. `kRasterVox` **pre-seeds its on-chip
`wgElect[lpx]` from `visPayloadV[px]` at tile entry** (before the K loop) so bricks occluded by
NEAR-FIELD leaves also early-skip — extending the occlusion-skip across the tri/voxel tier boundary.
(This turns the §6.6 "dispatch AFTER hwRender" correctness requirement into an optimization.)

### 6.3 `kRasterVox` — K near→far passes + the per-pixel early-depth SKIP (ported from `kRasterTiled :460-680`)

One **workgroup per 16×16 tile**, running the on-chip election.

**On-chip election state** (verbatim tileproto `:483-491`): `wgElect = workgroupArray('uint', 256)`
with `bufferType='atomic<u32>'` (packed `depthKey24<<8|id8`) + a plain `wgId = workgroupArray('uint',
256)` (full voxId payload). 2 × 256 × 4 B = **2 KB workgroup memory**. Cooperative init: the 64
threads strided-loop over the 256 entries (`loopU(local, 256, …, WG)` — **NOT subgroups**), zero
both, then `workgroupBarrier`. **THEN the §6.2 pre-seed:** copy each `visPayloadV[px]` into
`wgElect[lpx]` so the near-field triangle election is already on-chip.

**K passes** (verbatim `:498-518`): `loopU(0, K_BUCKETS, bi => { bk = (K_BUCKETS-1) - bi; … })`, so
`bk` walks **7 (NEAREST) … 0 (FARTHEST)**. Each pass re-walks the **single flat per-tile list** (1-word
entries; re-read K times — accepted bandwidth, the brick attr block read once when its bucket fires)
and processes **ONLY** entries whose `eBucket == bk` (`eBucket = entry >> BUCKET_SHIFT`). **Each brick
is processed EXACTLY ONCE, on its own bucket pass — ordering only reorders work, never the result.**

**Per-brick precompute** (verbatim `:539,607-617`):
```
nearKey = depthKey24(brick.nearestZ) << 8 | 0xff   // brick's NEAREST-possible election key
                                                   //  (AABB front-slab depth; 0xff = max id tiebreak)
```
`nearKey` is a **strict UPPER BOUND on `cand` at EVERY pixel the brick covers.** Per covered pixel:
```
prevE = aLoadU(wgElect[lpx])            // on-chip winner, read BEFORE deriving voxCz (== NaniteRaster.ts:821 pattern)
if (nearKey > prevE) {                  // OCCLUSION SKIP — else skip the whole voxCz interp + depthKey + atomicMax
  voxCz = AABB-slab / ray-entry NDC depth at pixel center      // §6.4
  cand  = depthKey24(voxCz) << 8 | (voxId & 0xff)
  if (cand > prevE) {
    wonE = atomicMax(wgElect[lpx], cand)
    if (cand > wonE) wgId[lpx] = voxId   // workgroup-scoped, full payload
  }
}
```

**HOW DEPTH COMMITS BETWEEN PASSES so far bricks see near depth:** the election is **on-chip** in
`wgElect[256]/wgId[256]`. Each bucket pass's `atomicMax` writes commit into `wgElect` **immediately**
and are visible to the NEXT (farther) pass's `nearKey > prevE` read with **NO global round-trip and
NO barrier needed between bucket passes** (same-workgroup same-pixel atomic ordering on the shared
array is sufficient; the only `workgroupBarrier`s are the cooperative init before the K loop and the
flush after, `:491,660`). So a far brick in bucket `bk` reads the committed nearer winners from
buckets `bk+1 … K-1` (and the pre-seeded near-field tris) and early-skips. **That IS the overdraw
win:** by the time a far/occluded brick is processed, its pixels fail `nearKey > prevE` and pay
nothing — FEWER brick writes than the overlapping leaf-cluster triangles, because occluded brick
rasterizations are skipped **wholesale**; a voxel may cover one OR several pixels (coarser-than-UE),
the skip operates per-pixel-of-the-brick, not a 1-write/pixel model.

**CORRECTNESS PROOF (cannot wrongly drop a near brick — verified, could not break it).** The election
is order-independent `atomicMax`; the bucket ordering only REORDERS which bricks process when, never
which one finally wins, so it is **loss-EXACT regardless of K** (tileproto invariant `:114,498-504`).
The skip discards a **provable LOSER**: `cand ≤ nearKey` at every pixel (cand uses this brick's actual
per-pixel `voxCz` which is `≥ brick.nearestZ` in NDC, so its `depthKey ≤` the nearest depthKey, and
the id byte `≤ 0xff`), so if `nearKey ≤ prevE` then `cand ≤ prevE` for sure ⇒ `atomicMax(prevE,
cand)` would not change the winner ⇒ skipping is **identical** to running it. A NEARER brick
processed LATER has a HIGHER `nearKey` and a HIGHER `cand`, so its `nearKey > prevE` passes and it
correctly wins. **This holds ONLY IF `brick.nearestZ` is a true conservative lower-NDC-depth bound —
which the coarse one-sample-per-brick default makes EXACT** (the AABB front-slab). A per-cell
ray-march refine re-opens the 24-bit depth-quantization cell-straddle (§4.5b) and IS a real temporal
hazard — correctly defaulted-OFF and ship-gated on an explicit error bound.

**FLUSH (verbatim `:659-680`):** after all K passes, ONE `workgroupBarrier`, then a single flush loop
MERGES `wgElect/wgId` into global `visPayloadV/visBV` via **`atomicMax` (NOT overwrite)** — preserving
any nearer near-field-tri or prior-wave winner. (Because we pre-seeded from `visPayloadV`, the flush
is a strict no-op for pixels where the near-field tri already won.)

### 6.4 `voxCz` derivation — the COARSE one-sample-per-brick path is the HARD DEFAULT

`voxCz` has **no triangle to barycentric-interpolate** (world1 gets cz from a 3-vert z-interp). Define
`voxCz` = the **AABB-slab / ray-entry NDC depth at the pixel center** — plain float box-front math, no
barycentric. **DEFAULT (HARD, correction #6): ONE `voxCz` per brick** (the AABB front-slab NDC
depth). This single default is load-bearing for **THREE separate gates**:
1. **KG-ID-BUDGET** — the election id is **brick-granular (~7 bits, trivially <30)**, dissolving the
   13-bit-cell × 23-bit-itemIdx collision in 30 bits (§4.5).
2. **KG-BUFFER** — `qVoxelRO` is dropped (the id packs `(instId, brickIdx)` directly), so the resolve
   lands at **9 of 10**, not 10 (§4.6).
3. **The §6.3 skip proof** — `brick.nearestZ` is **exact** (the AABB front-slab), so the `nearKey`
   upper bound and loss-exact skip hold **WITHOUT** the 24-bit depth-quantization cell-straddle.

**Refine to per-cell ray-march ONLY if KG-0b proves the coarse path too blobby — and ONLY with an
explicit depth-quantization error bound written first** (a real box/ray intersection per pixel
re-opens the §4.5b straddle, a new precision-dependent shimmer surface). Mark it as such in the code.

### 6.5 Depth-tie hazard at the transition (Risk #3 — the cross-TIER tie the K-pass does NOT solve)

A `depthKey24` tie degrades to a valid-but-maybe-wrong pick — sparse speckle, never a torn id. The
voxel bin's front-to-back K-pass + pre-seed removes the **intra-voxel** overdraw contest (occluded
bricks never write) AND lets near-field tris occlude far bricks. But it does NOT order a near-**BRICK**
against a **coplanar near-TRI** in the SAME `visPayloadV`: the flush-merge `atomicMax` preserves
whichever has the higher packed key, which at a depth tie is the **arbitrary 8-bit id tiebreak** ⇒
**per-pixel shimmer under motion** in the thin overlap window only. The election cannot fix this
**cross-tier** tie. **The cull-side cross-fade/dither band (§7.3) is therefore a Stage-3 HARD
DEPENDENCY**, but its job is narrow: only the thin mesh↔voxel overlap window. **TENSION (§6.5.bis):
the NEARER-than-UE transition puts that band in the highest-overdraw zone, and it DOUBLES work there —
directly fighting the win.**

### 6.5.bis The NEARER-transition ↔ CROSS-FADE-DOUBLE-WORK tension (must be resolved, not glossed)

Correction #2 pulls the transition NEAR (~40 m, inside the ~90 m `instMinPx` envelope). The §7.3
cross-fade band emits each transition-band cluster to **BOTH** the leaf mesh AND the voxel mesh — so
in the band you pay **the triangles AND the bricks AND the bin AND the K-pass.** Because the
transition is deliberately near, and the band must be wide enough to hide a WPO-sway→rigid-static pop
(§8, no TSR-Thin-Geometry analog), **the doubled-work band lands exactly where cluster density and
overdraw are HIGHEST** — strictly worse than a far-field UE-style placement would be. This is a
**genuine tension between correction #2 (pull transition nearer) and the perf thesis.**

**RESOLUTION (a quantified, measured gate — Stage-3):** state the **max cross-fade band width (in
metres) that still leaves a net win at 43.8 ms**, and make it a measured gate. If hiding the
WPO-sway→rigid-static pop needs a band **wider** than that budget, **KG-TEMPORAL + KG-NET-WIN kill
jointly.** The band width, the coverage ramp curve, and the per-cluster hash are **unspecified today**
(§7.3 admits this) — the doubled-work cost **cannot be bounded until they are**, so they are Stage-3
line items with a hard width budget, not free.

### 6.6 Kernel + thread model + INSERTION POINT + buffer split

Add a **`'vox'` mode** to `rasterKernel(mode)` — today `'depth' | 'combined' | 'world1'`
(`NaniteRaster.ts:337`) — the voxel bin is the 4th branch alongside the `mode === 'world1'` election
block (`:800-833`). `const kRasterVox = rasterKernel('vox')` next to `kRasterWorld1` (`:884`). The
voxel bin is a **two-kernel pass**, mirroring tileproto's `kSetup` (bin) → `kRasterTiled` (K-pass
fine raster) split, with buffers split by access type (`NaniteTileRaster.ts:223-255`): **`atomicBuf`**
(per-tile counts + stats cursors) vs **`dataBuf`** (the flat per-tile list).

**INSERTION POINT (correctness-relevant).** The two voxel kernels dispatch in `world1()` **right after
`hwRender`** — `world1()` is the arrow fn at `NaniteRaster.ts:1188-1192` (`dispatchIndirect(kRasterWorld1)
→ dispatch(kHwArgs) → hwRender`). **Insert `kVoxBin` then `kRasterVox` after `:1190` (after
`hwRender`)** so the SW + HW near-field triangle election is already in global `visPayloadV` and
pre-seeds the on-chip array (§6.2). Use the existing `split2D` / `wgLinear(DISPATCH_ROW)` 2D-split
dispatch (`NaniteCull.ts:265-275`).

**Bounded-batching** of the cut (`BATCH_CLUSTERS` waves, tileproto `:706-712`) carries over **only if
QVOX exceeds the fixed flat-buffer bound**; for the coarse brick band it likely fits **one wave** (far
fewer bricks than the 388k-cluster tri cut). Flag the flat-cap (`FLAT_TILE_CAP=8192`, overflow drops a
brick silently — not mis-elected) and the wave count as Stage-1/Stage-2 line items, not free.

The brick decode is broadcast via the workgroup-shared cache (mirror the `world1` wgcache); `voxId`
carries **bit31=1** (voxel marker) + low ≤30 bits per the §4.5 brick-granular default.

### 6.7 TSL r184 expressibility (YES — every piece, verbatim from the validated tileproto)

- **Election:** reuses the existing 32-bit `depthKey24<<8|id8` `atomicMax` on `visPayloadV` +
  `atomicStore visBV` (`NaniteRaster.ts:810-833`) — the **32-bit packed election IS the substitute for
  the absent 64-bit atomic**; the SVO ray-march path (which would need u64) is explicitly NOT built.
- **On-chip per-tile election:** `workgroupArray('uint', 256)` with `bufferType='atomic<u32>'` for
  `wgElect` + a plain `workgroupArray` for `wgId`, `atomicStore/atomicMax/aLoadU` on workgroup-scoped
  atomics + `workgroupBarrier` — all verbatim in the recovered `kRasterTiled`
  (`NaniteTileRaster.ts:483-491,667-680`), which compiled and ran. **NO subgroup ops anywhere** (the
  cooperative init/flush use a plain 64-thread strided loop, `loopU(local,256,…,WG)`).
- **Binning:** per-brick AABB project + screen-bbox `loopI`-footprint scatter with `atomicAdd`
  cursors into a flat storage buffer — the `kSetup` pattern (`:262-456`).
- **Brick AABB occupancy `2×u32`** (no u64) — bitwise ops on two u32.
- **Dispatch** via `split2D` / `wgLinear(DISPATCH_ROW)` 2D-split (`NaniteCull.ts:265-275`).
- **Metal TBDR:** the global flush `atomicMax` into a flat storage buffer is the same TBDR-friendly
  pattern world1 already uses; no new hazard.
- **Buffer ceiling** respected (≤8 raster, ≤9/10 resolve, §4.6).

Confirmed: nothing requires a feature beyond what world1 + the validated tileproto already used. The
`voxCz` AABB-slab depth is plain float math (no barycentric).

### 6.8 Net-win is unestablished until measured (perf honesty — see §6.0)

`instMinPx`→90 fps proves "deleting the band is cheap" (replaces it with ZERO), NOT "voxel-band <
triangle-band." The valid comparison is **voxel-band-GPU (depth-bucketed K-pass) vs the
triangle-band-GPU it replaces**, after build + brick-project + **bin (the §6.0 scar)** + K-pass
footprint-loop + bit31-election + matClass-7 resolve branch + the cross-fade double-work (§6.5.bis).
Note the triangle band has **no f2b lever to compare against** (f2b on world1 is a WASH, §0.bis) — so
the comparison is the depth-skipped voxel bin vs the plain triangle band, and the voxel bin's whole
advantage is that its bricks DO occlude and the K-pass skips them. Measure against the real **43.8 ms**
baseline via whole-frame `gpuWall` (NOT per-pass timestamps — per the capture-method memory).

---

## 7. RESOLVE / STOCHASTIC-NORMAL SHADING

### 7.1 Branch order (corrected — bit31 test FIRST)

The resolve currently reads `cull.qRasterRO.element(itemIdx+1)` **unconditionally** at
`NaniteResolve.ts:284`, before matClass. A voxel id is **not** a triangle itemIdx, so indexing
qRasterRO with it reads garbage and matClass derived from it is garbage. **The bit31 voxel test MUST
precede the qRasterRO read.** Spec:
```
isV = (pRaw >> 31) & 1
if (isV) { decode voxel (instId, brickIdx) from pRaw low bits [+ voxelBricks normal slice]; matClass = voxel }
else     { item = qRasterRO[itemIdx+1]; ci = item.y; matClass = mesh(ci).matClass }   // existing path, unchanged
```
With the coarse brick-granular default, `(instId, brickIdx)` decode directly from `pRaw` — **no
`qVoxelRO` lookup** (§4.6, resolve at 9).

### 7.2 Voxel shading branch (NOT a new pass, single-resolve design)

Add `voxel:7` to the enum and an `isV` block + mux entries at `NaniteResolve.ts:574-584`:
1. **Reuse the SAME `wp`** from the election key (`NaniteResolve.ts:273-280`) — zero new depth math.
2. Decode normal descriptor from the new resolve-bound `voxelBricks` slice (brick-mean in the coarse
   default).
3. **CHEAP PATH FIRST — shade the MEAN normal, no stochastic draw.** Our lighting is **DIFFUSE-ONLY**
   (`NaniteResolve.ts:522` "NO specular"; `:586-668` lambert+ambient+probe GI). UE stores a
   *distribution* specifically for **specular aggregation**, which we do not have — so a single mean
   normal is very likely sufficient, sidestepping the stochastic noise AND most of the normal-noise
   TAA hazard. **Evaluate this first.**
4. **FALLBACK (only if mean-normal canopies read flat):** stochastic per-pixel SGGX draw hashed on a
   **TAA-stable seed** `hash(screenCoordinate, frameIndex)`. Changes the build output (6-B SGGX from
   RAW pre-bend normals, §5.4), the resolve buffer layout, AND re-introduces the stochastic TAA
   hazard — so the mean-vs-SGGX decision is made in **Stage 0b** (decidable on the one-crown spike)
   and the brick payload sized to the winner BEFORE the addLate reservation freezes (§5.3).
5. Either way **flip camera-ward** via the existing leaf idiom (`NaniteResolve.ts:568-569`); voxels
   are inherently two-sided (occupancy has no winding).
6. **Albedo via matParam** (`NaniteResolve.ts:544-560`) — no per-voxel color buffer. Flow through the
   SHARED lighting (`:586-668`) + `depthNode` unchanged.

### 7.3 Cross-fade / dither (Risk #3, in the cull — a HARD Stage-3 dependency with a WIDTH BUDGET)

In a distance window `[d_lo, d_hi]` around `transitionDist`, emit a cluster to **BOTH** the leaf mesh
(Tier 1) and the voxel mesh (Tier 2) with a **per-cluster** (NOT per-pixel — per-pixel dither + TAA =
classic dither-ghosting) **stable-hash dither coverage** that ramps over distance. **This DOUBLES work
in the band**, in the highest-overdraw zone (§6.5.bis), directly fighting the overdraw reduction that
is the entire perf justification. **The band width, the coverage ramp curve, and the per-cluster hash
are NOT yet specified** — they are the only defense against the binary mesh↔voxel flip popping AND the
cross-tier depth tie (§6.5), and they must be A/B'd under the moving-camera capture. **Quantify a max
band-width budget (metres) that still nets a win at 43.8 ms (§6.5.bis), and gate on it in Stage 3:**
if the band must be wider than that budget to hide the pop, that is a joint **KG-TEMPORAL + KG-NET-WIN
kill.**

---

## 8. ANIMATION (transform inheritance)

Voxels are baked **rest-pose** (no per-frame re-voxelization — confirmed UE, `research-analyses.txt
§3`). The brick set inherits the per-instance **rigid transform** (instance record B,
`GeometryRegistry.ts:18`). **No per-voxel wind.** Acceptable ONLY where sub-pixel sway projects < 1 px,
which the transition distance guarantees by construction (UE auto-disables animation below a
screen-size gate, `research-analyses.txt §3`). **The mesh→voxel boundary is a WPO→static handoff** —
a temporal discontinuity (the leaf head sways, the voxel head is rigid), covered by the Risk #3
cross-fade band. UE documents this WPO-on-voxels handoff as incomplete/experimental; we have **no
TSR-Thin-Geometry analog** to hide it. **OPEN RISK** (KG-TEMPORAL).

---

## 9. MEMORY PLAN

- **Palette-only:** ~36 deterministic crowns (§5.1), reused by all 200k trees via transform
  instancing. Per-tree-unique (400 GB–12 TB) avoided.
- **Brick storage:** bounded ~tens of MB **IF only coarse clusters carry bricks** (the aggregate DAG
  collapses toward root before the transition). **EXACT number unquantified until §5.3 runs the
  per-crown cluster count** + the per-coarse-cluster ≤128-brick fit proof — a HARD Stage-1 gate, not
  an assumption.
- **`addLate` needs a new `bricks` field** + buffer-sizing path (`GeometryRegistry.ts:917-922`); the
  reservation FREEZES at `build()`, so the §5.3 count AND the Stage-0b mean-vs-SGGX size decision must
  precede it. `RegistryGpu` grows to an **11th field** (`voxelBricks`, `:783-808`).
- **Payload sizing:** size the brick attribute to the mean-vs-SGGX **winner** (decided Stage 0b) to
  avoid baking dead bytes into the frozen reservation.
- **Cache:** spec a persistent voxel cache (DagCache pattern) or pay voxelization every cold boot.

---

## 10. CONSTRAINTS HANDLING (TSL r184 / buffer ceiling)

- **NO 64-bit atomics:** we **rasterize bricks** and reuse the existing 32-bit `depthKey24<<8|id8`
  election (`NaniteRaster.ts:810-833`). The 64-bit dependency in the literature is the SVO ray-march
  path, which we do NOT build. ✅ expressible.
- **NO mesh shaders / NO subgroups:** the voxel raster is a plain compute kernel (per-brick footprint
  loop + cooperative strided init/flush, NOT subgroup reductions). ✅
- **~10 storage buffers / stage:** raster fits via the separate permutation that drops tri-only
  buffers (≈7–8, §4.6) — UNVERIFIED until written (Stage-2 hard gate). Resolve fits at **9** (coarse
  brick-granular id drops `qVoxelRO`) / 10 (per-cell refine) — with the **two-pass voxel-resolve as
  the guaranteed-safe DEFAULT** until the three packing assumptions are proven. kTraverse is **not
  touched** (routing is a per-mesh distance handoff + a tiny post-traverse fan-out pass, §4.6).
- **Metal TBDR:** the per-pixel election is already TBDR-friendly; the voxel bin adds no new pattern.
- **`?clustertris` cap:** voxel id namespacing uses **bit31** (free at both 128 and 256 caps); voxel
  id ≤ 30 bits (§4.5).

---

## 11. STAGED IMPLEMENTATION PLAN (small, SCREENSHOT-VERIFIED stages — front-loads the existential risks)

**Effort + method (no months-long roadmap).** The realistic build is **~4 HOURS of LLM/implementation
work for the bulk of the staged subsystem + ~1 day of optimization/fixes** on top, run as a
**dynamic-workflow STAGED implementation where every stage is verified by a SCREENSHOT/CAPTURE taken
at that stage.** Each stage below **names the shot/capture that proves it** — these ARE the per-stage
verification artifacts (a judge-shot, a banded-counter HUD readout, a fixed-distance render, a
moving-camera capture), not calendar milestones. Stage-0 de-risk is hours. **Baseline for every perf
gate = the real 43.8 ms p0.95** at canonical config (2268×1473, 200k trees, moving camera). There is
**no separate f2b baseline** — f2b on the triangle raster is a WASH (§0.bis), so the old "0c
cheapest-disproof" gate is **DELETED** and replaced by the per-fragment voxel-vs-triangle estimate in
0a.

### STAGE 0 — DE-RISK (no subsystem code; both gates must pass; ~hours)

**0a. THE CHEAPEST DISPROOF FIRST — fragments-per-crown(voxel) vs (triangle) + band cost-share
(Risk #2 / §6.0).** **FIRST measurement (replaces the deleted f2b gate):** a back-of-envelope (or
throwaway fixed-distance spike) of **fragments-per-crown(voxel, coarse 1-sample-per-brick) vs
fragments-per-crown(triangle)** at the transition — this is the cheapest disproof of the real thesis
(the §6.0 tileproto bin-cost scar: the bin/setup half is a hard floor that already lost the frame by
7.4 ms). THEN add a distance-banded counter (bucket `distC` at emit; counters surface to HUD,
`NaniteCull.ts:520-522` / `NaniteFrame.ts:459-462`) for clusters/tris/fragments PAST the candidate
`transitionDist`, and cross-check with an `instMinPx` sweep via WebGPU-Inspector whole-frame `gpuWall`.
**SHOT:** the banded-counter HUD readout + the per-fragment ratio number. **GATE:** voxel-band-fragments
< triangle-band-fragments by a margin that survives build + bin + resolve + cross-fade overhead, AND
band > ~40% of frame. **KILL:** if voxels don't beat triangles per-fragment, STOP before any raster —
this is the cheapest place to disprove the thesis.

**0b. ONE-CROWN QUALITY SPIKE on the WORST-CASE CONIFER (Risk #1 — the single fine-shape kill).**
Offline-voxelize one crown **at the WORST case: SPRUCE or PINE FIRST** (single-quad ~0.02 m needles,
`Species.ts:53,104`, `LeafMesh.ts:128-167`) — **NOT beech; testing the easy broadleaf and shipping
the verdict for the hard conifer is the optimistic move forbidden in writing** — with
**fractional-density** voxelization, at the candidate `transitionDist` for the chosen `voxelGridDim`.
Render voxel-crown vs the **exact simplified DAG cluster the cut emits there** (build a tiny
extraction tool — the cut emits a DAG cluster, not a named LOD). **SHOT:** side-by-side judge-shots
(voxel crown | simplified mesh-LOD). **GATE:** (i) voxel crown ≥ the simplified mesh-LOD AND (ii)
needles SURVIVE (no blob, no bald) AND (iii) the transition sits NEARER than where `instMinPx` already
culls. Also spike the bend-normal variance trap and **DECIDE mean-vs-SGGX here** (sizes the brick
payload before §5.3 freezes the reservation). **KILL:** if worse, ratchet the transition farther
(coarser); if it never passes before the band overlaps `instMinPx` culling, the win is ZERO → STOP.

Both 0a/0b are cheap (hours), use existing counters/capture, need NO raster/resolve/registry surgery,
and each produces a SHOT before proceeding. **Do NOT build until both pass.**

### STAGE 1 — REGISTRY + BUILD SCAFFOLD + FIXED-DISTANCE NET-WIN (~1 h)

`gpu.voxelBricks` (11th buffer) + **`addLate.bricks` field + buffer-sizing path (HARD precondition,
freezes at build, §5.3)** sized to the Stage-0b mean-vs-SGGX winner; `voxel:7` matClass; voxel sibling
head (`registerVoxelMesh`); the offline palette voxelizer + normal fit; voxel cache. **Emit the §5.3
per-crown cluster-count-above-transition × byte-size and PROVE the ≤128-brick per-coarse-cluster fit**
(or flag finer-clusters/coarser-cells as an open risk that re-runs the budget). **Pull KG-NET-WIN
forward:** render ONE voxelized crown band at a FIXED distance and measure GPU-wall delta vs the same
band as triangles. **SHOT:** the voxel crown rendering at the fixed distance + the per-crown brick
count + total MB printout. **ACCEPTANCE:** ~36 crowns voxelize within boot budget (state ms + thread);
bricks upload; budget reported; fixed-distance band is net-neutral-or-better. **KILL:** neutral/regress
at fixed distance → STOP before cull/resolve surgery.

### STAGE 2 — RASTER BIN (kVoxBin + kRasterVox) + RESOLVE BRANCH (~1.5 h)

`kRasterVox` permutation (drop tri-only buffers, bind `voxelBricks`); **`kVoxBin` (§6.2) + the K
near→far passes + per-pixel early-depth skip + pre-seed (§6.3)**, BUCKET_SHIFT=28 / K_BUCKETS=8 /
KBITS=3; bit31 id namespacing (brick-granular, coarse one-sample default, §6.4); matClass-7 resolve
branch (mean-normal first, bit31 test BEFORE qRasterRO read). Dispatch the two voxel kernels **after
`hwRender` at `NaniteRaster.ts:1190`** (§6.6). **SHOT:** side-by-side capture of voxel crowns at fixed
distance vs the triangle band, + the **per-pixel write-count** overlay (the occlusion-skip should show
FEWER brick writes than the overlapping triangles). **ACCEPTANCE:** voxel crowns render correctly;
**BINDING AUDIT confirms ≤10 on raster AND resolve** (else the two-pass voxel-resolve fallback is wired
and becomes default). **MEASURE (KG-NET-WIN signal, §6.0/§6.8):** write count + GPU-wall delta vs the
triangle band, against 43.8 ms.

### STAGE 3 — CULL ROUTING + TRANSITION + CROSS-FADE (cross-fade is a HARD dependency; ~1 h)

Per-mesh distance handoff (leaf head maxDist→transition, voxel head→TREE_GEO_FAR) + the tiny
post-traverse fan-out pass into `qVoxRaster` + the **per-cluster cross-fade/dither band** (required vs
the un-depth-ordered cross-tier shimmer, §6.5) **with the §6.5.bis max band-width budget enforced**.
**SHOT:** full-forest moving-camera CAPTURE at canonical config (per the capture-method memory) + the
frame p0.95 readout. **ACCEPTANCE:** clusters transition mesh→voxel at `transitionDist`; full forest
renders. **GATE (KG-NET-WIN):** net GPU win vs the **43.8 ms** Stage-0 baseline, **INCLUDING the
cross-fade double-work (§6.5.bis) and the fan-out re-scan.** Neutral/regress after debugging overdraw,
OR the band must exceed its width budget → KILL.

### STAGE 4 — TEMPORAL STABILIZATION (~0.5 h + part of the +1 day)

Moving-camera dolly through the band; mean-vs-stochastic A/B; WPO→static handoff. **SHOT:**
moving-camera capture through the transition (judge by capture, not stills). **GATE (KG-TEMPORAL):**
no material popping/shimmer/ghosting at the boundary under motion; if the per-cluster flip crackles
and the dither band cannot hide it within its width budget (no TSR-Thin-Geometry here), joint
KG-TEMPORAL + KG-NET-WIN kill.

### STAGE 5 — TUNE + SHIP-OR-KILL (the bulk of the +1 day)

Sweep `voxelGridDim` (`?voxgrid=`), `transitionDist` (`?voxnear=`), and cross-fade band width vs
quality+perf — finding the **coarsest detail + nearest transition** that still passes KG-0b and nets a
win. **SHOT:** the sweep matrix (a grid of judge-shots + p0.95 per setting). **Final gate:** does the
2-tier system close a material fraction of the 27.2 ms gap WITHOUT quality regression? Yes → ship.
Real-but-small + marginal quality → document, consider Tier-3 impostors (§13).

---

## 12. OPEN RISKS (each with a kill-gate)

0. **THE TILEPROTO BIN-COST SCAR (PRIMARY existential risk, above fine-shape — §6.0).** The same
   bin→K-pass mechanism measured **+11.7/+18.1 ms slower** and was DELETED (`eecf046`) because the
   bin/setup half is a hard floor (8.65 ms setup ⇒ tiled 25.1 vs world1 17.7,
   `docs/perf-runs/2026-06-18-tiled-maximize.md:57-65`). The entire voxel net-win rests on ONE
   unproven inequality — **brick-bin cost ≪ saved elections** — true ONLY IF brick count per tile is
   far below the 388k-cluster tri cut AND bricks overdraw enough. **KILL-GATE: Stage-0a per-fragment
   voxel-vs-triangle estimate FIRST** (the cheapest disproof, replacing the deleted f2b gate).
1. **RISK #1 — FINE-SHAPE FIDELITY (the #1 quality risk).** Sub-cell blades (~0.07 m) and needles
   (~0.02 m, `Species.ts:53,104,157`) bloat or vanish if voxelized while resolvable; the nearer-than-UE
   transition deliberately pushes INTO this danger zone. **MITIGATION:** the two tuneable knobs
   (finer grid + nearer transition, §3.2.bis) + mandatory fractional-density voxelization (§5.4).
   **KILL-GATE KG-0b:** the WORST-CASE CONIFER one-crown spike (spruce/pine FIRST, before the ID-bit
   fork) must beat the simplified mesh-LOD AND survive AND sit nearer than the cull edge. If no
   `(voxelGridDim, transitionDist)` passes, the win is ZERO → KILL.
2. **RISK #2 — NET-WIN UNMEASURED (KG-0a/KG-NET-WIN).** `instMinPx`→90 fps only proves deleting the
   band is cheap vs ZERO, not that the depth-skipped voxel bin beats the triangle band. **MITIGATION:**
   Stage 0a per-fragment estimate + banded counter; Stage 1/2/3 measured GPU-wall deltas vs 43.8 ms.
   **KILL-GATE KG-NET-WIN:** net win must clear build + brick-project + bin (§6.0) + K-pass +
   bit31-election + matClass-7 resolve branch + cross-fade double-work (§6.5.bis), whole-frame
   `gpuWall`.
3. **RISK #3 — CROSS-TIER DEPTH TIE / TEMPORAL (KG-TEMPORAL).** The K-pass orders bricks AGAINST EACH
   OTHER (and the pre-seed lets near tris occlude far bricks) but does NOT order a near-BRICK against a
   coplanar near-TRI in the same `visPayloadV` ⇒ arbitrary 8-bit id tiebreak ⇒ shimmer in the thin
   overlap window (§6.5). The per-cluster cross-fade/dither band (Stage-3 HARD dep, §7.3) is the only
   defense and it DOUBLES work in the highest-overdraw zone (§6.5.bis), fighting the win; WPO→static
   handoff (§8) adds a discontinuity with no TSR-Thin-Geometry analog. **KILL-GATE KG-TEMPORAL +
   KG-NET-WIN jointly** (Stage 4 capture + the §6.5.bis band-width budget).
4. **RISK #4 — 10-BUFFER CEILING (KG-BUFFER).** Resolve targets **9** single-pass (8 today,
   grep-confirmed) via the coarse brick-granular id dropping `qVoxelRO`; **10 with ZERO margin** if the
   per-cell refine is ever forced. Contingent on three unproven packing assumptions (matParam color,
   raster-only occupancy, single metadata buffer); the ~7–8 raster figure is plausible but UNVERIFIED
   until kVoxBin/kRasterVox are written. **MITIGATION + FALLBACK (Stage-2 default until proven):**
   two-pass bit31-keyed voxel-resolve. **KILL-GATE KG-BUFFER:** Stage-2 binding audit proves ≤10 on
   both stages, or the two-pass fallback is wired.
5. **RISK #5 — ID BIT BUDGET (KG-ID-BUDGET).** Only bit31 is free at both caps; 128×64=8192 = 13 bits
   collides with 23-bit voxItemIdx in 30 bits. **The coarse one-sample-per-brick path (the HARD
   default, §6.4) makes the id brick-granular (~7 bits) and DISSOLVES this fork.** Only if KG-0b forces
   per-cell detail: cap ≤512 voxels/cluster OR brick-granular election + resolve cell re-derive — the
   latter a NEW precision-dependent shimmer surface (24-bit depth quantization straddles cells at the
   ~1 px transition, §4.5b) that needs an explicit error bound before it ships.
6. **RISK #6 — BIN/SCATTER COST + FLAT-LIST OVERFLOW.** `kVoxBin` re-uses the tileproto flat per-tile
   list (`FLAT_TILE_CAP=8192`) with NO per-bucket cap; an overflow brick is silently NOT rasterized
   (not mis-elected). The per-crown brick count + per-tile worst-case occupancy + the bounded-batching
   wave count are unquantified until §5.3. **KILL-GATE: folded into Stage-1/Stage-2 line items** (flag
   the flat-cap + wave count, not free).
7. **RISK #7 — CONSERVATIVE OFFLINE VOXELIZATION ON WGSL (folded into KG-0b).** No geometry shader, no
   HW conservative raster; emulate via software dilation or multi-axis MSAA union. A hole baked into a
   thin-leaf crown is a **permanent artifact**. Plus the bend-normal variance trap (`TubeMesh.ts:51`)
   if SGGX ships (moot if mean-normal passes 0b). Build-time stacks on the ~16 s leaf-DAG boot; a
   persistent voxel cache (DagCache pattern) is required or every cold boot re-voxelizes. **KILL-GATE:
   folded into KG-0b** (needles hole-free) + Stage-1 boot-budget acceptance.
8. **RISK #8 — ANIMATION HANDOFF.** Rest-pose rigid bricks; the mesh→voxel boundary is a WPO→static
   handoff (§8). Covered by Risk #3 / KG-TEMPORAL.
9. **RISK #9 — `voxCz` DERIVATION AMBIGUITY.** world1 gets cz from a 3-vert barycentric interp; the
   voxel bin has none ⇒ `voxCz` = AABB-slab/ray-entry NDC depth at pixel center. **The coarse
   one-sample-per-brick start makes `brick.nearestZ` EXACT** (the AABB front-slab), so the bucket key
   and nearKey upper bound are sound; a per-cell refine needs a real ray/box intersection per pixel and
   re-opens the depth-quantization straddle. **Keep the coarse path unless KG-0b forces otherwise.**
10. **UNCERTAINTY HONESTY.** The third-party UE perf number (62→119 fps) is whole-frame vs Nanite-MESH
    of the SAME asset, NOT vs an optimized impostor baseline — indicative only. Transition px values
    are DERIVED from projK + leaf dims, NOT measured — they are TUNEABLE knobs (uniforms, §3.2.bis)
    with browser-budget defaults that Stage 0b/Stage 5 tune. The overdraw win is the **occlusion-skip**
    delivered by the FIRST-CLASS, designed depth-bucketed K-pass (§6, ported from the recovered
    tileproto) — **fewer brick writes than the overlapping triangles**, NOT a fixed 1-write/pixel — but
    its magnitude is **the §6.0 bet**, unmeasured until KG-NET-WIN (Stage 2/3).

---

## 13. FUTURE / FALLBACK — TIER-3 IMPOSTORS (noted, NOT designed)

IF voxels prove inadequate at very-far (cost still too high, OR the silhouette degrades past the brick
resolution that fits memory), a billboard-impostor very-far tier (~16×16 px) slots in EXACTLY where
`instMinPx` currently culls whole instances (`NaniteCull.ts:402`) — a **per-instance** (not
per-cluster) far-field swap beyond the voxel band. The foliage atlas capture already exists
(`captureFoliageAtlas`, `VegLibrary.ts:176`), so the source art is partly in place.

**OPEN ALPHA-TESTING QUESTION (the reason it is not designed here):** impostors are **alpha-masked
cards**, re-introducing the per-pixel alpha-test overdraw that voxels were built to KILL, and the
vis-buffer election is **opaque-only** (`depthKey24<<8|id8`, no coverage). Keep the 2-tier focus;
revisit only if Stage 5 shows voxels-all-the-way is insufficient.

---

## 14. DECISION LOG / WHAT WE ARE NOT DOING AND WHY

- **NOT a cluster flag for "is voxel."** word7 has no free flag bit (`GeometryRegistry.ts:1231-1233`).
  We carry the decision on the voxel sibling MESH (`matClass=voxel(7)`), which the resolve reads free.
- **NOT `matClass=5`.** That is `grass` (`GeometryRegistry.ts:135`). We use `voxel:7`.
- **NOT a "1 voxel = 1 pixel" target.** UE's sub-pixel goal is rejected; voxel DETAIL and TRANSITION
  DISTANCE are tuneable knobs with COARSER browser-budget defaults, found by testing (§3.2.bis).
- **NOT testing f2b on the triangle raster.** It is a measured WASH (`docs/perf-runs/2026-06-18-tiled-maximize.md:64`);
  sparse sub-pixel foliage tris don't occlude. The old "0c" gate is DELETED; its replacement is the
  per-fragment voxel-vs-triangle estimate (Stage 0a / §6.0).
- **NOT a fresh depth-bucket mechanism.** We PORT the recovered tileproto (`dfe6518:NaniteTileRaster.ts`),
  with bricks as the binned primitive — AND we lead with its measured failure scar (§6.0).
- **NOT routing inside kTraverse.** Pinned ≤10 buffers by counter-slot recycling (`NaniteCull.ts:353-359`).
  Voxel clusters emit into the SAME qRaster, then a tiny post-traverse fan-out pass into qVoxRaster.
- **NOT an SVO ray-march.** That is where the 64-bit-atomic dependency lives. We rasterize bricks and
  reuse the 32-bit election.
- **NOT per-tree-unique voxelization.** 400 GB–12 TB; we voxelize the ~36-crown palette and instance.
- **NOT a per-cell election by default.** The coarse one-sample-per-brick path is the HARD default
  (§6.4); per-cell ray-march ships only if KG-0b forces it AND with an explicit error bound.
- **NOT a stochastic per-voxel normal by default.** Lighting is diffuse-only (`NaniteResolve.ts:522`);
  mean normal is the cheap path. SGGX is a Stage-0b-decided fallback.
- **NOT designing impostors.** §13 notes where they slot in and the open alpha-test question only.
- **NOT trusting per-pass GPU timestamps.** Per the capture-method memory, we trust whole-frame `gpuWall`.

---

### Appendix: verified file:line anchors (spot-checked against the live tree + recovered tileproto)

- word7 layout + encode: `GeometryRegistry.ts:1231-1233`; decode `:396-413`; GPU fetch `NaniteFetch.ts:147-155`
- cluster flags (both bits taken): `GeometryRegistry.ts:162-164`
- matClass enum (5=grass, 6=debris): `GeometryRegistry.ts:129-137`; byte in mesh w6: `:448` / `NaniteFetch.ts:701`
- mesh flags (TWO_SIDED=16): `GeometryRegistry.ts:149-160`
- projK + cull cut + minPx + instMinPx: `NaniteCull.ts:247, 398-402, 450, 459-462, 515-522`
- per-mesh draw envelope (lodDist / setMaxDistance): `NaniteCull.ts:391-394`
- kTraverse counter-slot recycling (≤10): `NaniteCull.ts:353-359`; 2D-split dispatch `:265-275`
- 32-bit election + f2b default-off: `NaniteRaster.ts:337, 800-833, 841-844`; read-before-write `:810-830`, prevE `:821`
- world1 dispatch order (SW → kHwArgs → hwRender), insertion point: `NaniteRaster.ts:1188-1192` (insert voxel kernels after `:1190`)
- instMinPx default ~110 px: `NaniteFrame.ts:181`
- resolve buffers (8 today) + qRasterRO unconditional read + wp reconstruct + leaf branch:
  `NaniteResolve.ts:189, 206, 268-290, 273-280, 522, 525-570, 574-584, 586-668, 636-638`
- ProbeGI binds 0 storage buffers (texture3D read): `ProbeGI.ts:369-371`
- RegistryGpu = 10 fields: `GeometryRegistry.ts:783-808`; addLate (no bricks): `:915-922, 1020-1028`
- aggregate DAG cluster stats: `BuildAggregateDag.ts:613-618`
- leaf registration sibling insertion: `WorldRegistry.ts:415-432`; packLeafTint `:426`; build-budget warn `:725-726`
- palette: `Species.ts:53,104,151,157,204,210,298,307`; TREE_VARIANTS=4 `Scatter.ts:82`; VegLibrary `:145,176,241-262,344-448`
- needles single-quad: `LeafMesh.ts:128-167`; bendNormals: `TubeMesh.ts:51`
- **RECOVERED TILEPROTO (`git show dfe6518:src/nanite/NaniteTileRaster.ts`), ported into §6:**
  - constants K_BUCKETS=8 / KBITS=3 / BUCKET_SHIFT=28 / FLAT_TILE_CAP=8192 + bit-budget assert: `:108-159`
  - kSetup bin (primeCtx, AABB project, bucket key, scatter, atomicBuf/dataBuf split): `:281-456`; split lesson `:223-255`
  - kRasterTiled on-chip election (workgroupArray + atomic<u32> + cooperative init): `:483-491`
  - K near→far passes (bk=K-1..0, eBucket==bk, exactly-once): `:498-518`
  - per-pixel early-out (nearKey>prevE BEFORE z-interp): `:539,607-617`; loss-exact invariant `:114,498-504`
  - flush-MERGE into global visPayloadV/visBV via atomicMax (not overwrite): `:659-680`
  - bounded-batching waves (BATCH_CLUSTERS): `:706-712`
- **THE SCAR:** `eecf046` commit body (+11.7/+18.1 ms, coverage-bound, deleted);
  `docs/perf-runs/2026-06-18-tiled-maximize.md:57-65` (setup 8.65 ms floor: tiled 25.1 vs world1 17.7);
  f2b-on-world1 WASH `:64`; forest 60% coverage/election-bound `docs/perf-runs/2026-06-16-forest-raster-diagnosis.md:17`
- research mechanics (UE error-switch, 128 4×4×4 bricks, SGGX 6 B, 400 GB–12 TB, build cost):
  `research-analyses.txt §2,3,4,5,6,7`
