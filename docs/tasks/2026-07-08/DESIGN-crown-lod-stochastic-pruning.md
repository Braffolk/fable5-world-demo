# DESIGN — Close-by crown mesh LOD by Cook stochastic pruning, adapted to our anchor/spray/card geometry

Status: **DESIGN + RESEARCH ONLY. Nothing implemented. Implementation is gated on user review of this document.**
Author task: re-enable *close-by* crown-LOD **meshing** with a new simplification algorithm that (a) looks **nearly
identical** to the full mesh up close and (b) structurally cannot produce the old failures — **spruce super-long
spikes** and **leaves welded into MASSIVE merged leaves**. Shadows keep using voxels; the voxel/shadow path is
**unchanged**.

Primary source: Cook, Halstead, Planck & Ryu, *Stochastic Simplification of Aggregate Detail*, ACM TOG 26(3)
Art. 79 / SIGGRAPH 2007 (`docs/mobile-gpu-perf/1275808.1276476 (1).pdf`, 8 pp., read in full incl. the appendix).
Secondary: `docs/mobile-gpu-perf/deep-research-crown-mesh-simplification.md` (our synthesis; the PDF is used below to
fill and correct its two admitted gaps — the exact selection method and the exact temporal fade).

---

## 0. What "close-by crown-LOD meshing is disabled" actually means (the whale, grounded)

The near crown is the perf/memory whale (memory index: `nanProjectVerts` ≈ 30% of frame, `projVertBuf` ≈ 1.13–1.18 GB;
`nanMidRaster` ≈ 14.5%). It is disabled at exactly one place:

- **Mesh→voxel handoff = 60 m.** `DEFAULT_TRANSITION_DIST = 60` (`src/nanite/WorldRegistry.ts:99`, `?voxnear`). Inside
  0–60 m the crown is the **real leaf/needle mesh** (`foliageMesh`, `foliageMode:'hybrid'`, full anchor density
  `leafAnchorTarget = 4000` → effectively all anchors — `src/vegetation/VegLibrary.ts:176,266–281`). Beyond 60 m it is
  the **voxel crown** (`VoxelizeCrown.ts`), which also owns **all crown shadows** (leaf mesh is `castShadow:false`,
  `src/vegetation/VegLibrary.ts:292`).
- **`?crownlod0` (DEFAULT ON, user mandate 2026-07-04)** forces every leaf cluster that still has DAG children to
  **DESCEND regardless of screen error**; only true LOD0 leaves (`childCount==0`) emit
  (`src/nanite/NaniteCull.ts:1183–1194`, `LEAF_MATCLASS = 4` at `:683`). Net effect: **the entire 0–60 m band renders
  the FULL LOD0 leaf mesh.** The crown's simplified DAG levels render **nowhere** for the camera.
- Consequently the crown DAG is built **`maxLevels:1`** ("crown aggregate LOD0-only… dead post-crownlod0",
  `src/nanite/WorldRegistry.ts:596–621, :1335`; `src/nanite/BootCache.ts:38–39`) — the coarse levels were dropped as
  pure build+mem waste **because they looked bad**, not because they were unwanted.

So "re-enable close-by crown mesh LOD" = **let leaf clusters emit at coarser-than-LOD0 levels inside 0–60 m** (relax
`forceDescend`), **once** those coarser levels are near-identical to LOD0. The `BUILD-LOG-overnight.md:8` states the same
lever from the memory side: *"The real projVertBuf memory lever is fewer/coarser crown clusters."* Fewer near-crown
verts directly shrinks `projVertBuf`, `nanProjectVerts`, and `nanMidRaster`. That is the win.

**Crucial nuance discovered in the code:** we already ship a Cook-style prune-and-grow builder —
`src/nanite/BuildAggregateDag.ts` (island keep-by-hash + √(1/λ) survivor grow about the island centroid). It builds and
renders; it is simply **tuned for the far band and currently switched off near-band**, and it has two properties that
make it wrong for *close-by* fidelity (welds the baked soup; uses surface-area not visible-area growth; uniform-grows
connected needle sprays → the spruce-spike mechanism). This design keeps its DAG *scaffolding* (the crack-free cut
records the raster depends on) and replaces its *pruning strategy* with an anchor-graph-baked, interior-biased,
visible-area-preserving one tuned for the near band. Details in §5–§6.

---

## 1. PREMISE-AUDIT #1 — is Cook stochastic pruning the right fit for OUR real-time baked-LOD pipeline?

The paper is **Pixar OFFLINE film rendering** (RenderMan, point-sampled surfaces, elements streamed from a
priority-ordered file at render time, λ driven by screen size **and motion blur / depth of field**; pp. 79-1..79-2,
79-6 Figs 13–16). Before adopting it I put the "given" on trial: does the offline→real-time-baked adaptation hold, or
am I importing a film technique into a game pipeline where it breaks?

**It holds. Evidence, strongest first:**

1. **We already run a Cook adaptation in production.** `BuildAggregateDag.ts` is literally Cook: keep a low-hash prefix
   of leaf islands until kept area ≥ `targetRatio`·total, then grow survivors by `g = √(totalArea/keptArea) = √(1/λ)`
   about the island centroid (`:275–331`). It builds three-free, node-runnable, deterministic, and rides the existing
   flat `kClusterCull` cut. The offline→real-time question is therefore already answered *in this repo* — the only open
   part is near-band tuning.
2. **SpeedTree ships exactly this in real time** (synthesis §1, §6): "fewer and bigger leaf instances… survivors scaled
   up," implemented as a vertex format carrying a base position + a 3D-LOD value lerped on a CPU LOD scalar — smooth,
   animatable, stays a mesh, never merges cards. UE5 "Preserve Area" and meshoptimizer `SimplifyPrune` are the same
   family. The technique's real-time descendants are mature.
3. **The offline-only parts do not transfer, and we do not need them.** The paper's render-time file read (pp. 79-3
   Fig 6) and its motion-blur/DOF λ terms (λ = λ_size·λ_blur, Eq 1, p. 79-3; Figs 14–16) are the *offline* couplings.
   Our replacement is the **DAG**: bake discrete LOD levels each pruning a deterministic subset, and select the level
   by the per-cluster screen-error **cut** (`emit when projK·err/d ≤ τ`). This is precisely the Hulm/SpeedTree lesson
   the synthesis flags (§5, §7): **bake LODs from the procedural graph, not by decimating the baked mesh.** Our foliage
   *is* procedurally generated from anchors, so this is native, not a port.

**The ONE genuine divergence — surfaced, not hand-waved:** Cook's aggressiveness rests on an assumption that fails
close-up. The paper is explicit (p. 79-4 §3.3: the survivor widening "is not noticeable in practice because the
elements are so small that their shapes are not discernible"; p. 79-7 §5: "The elements must be small enough that
individual element shapes are not more important than the shape of the whole"). **Close-by, our leaves and needles are
individually resolvable** — the exact regime where Cook warns simplification "cannot be as aggressive." This is **not a
refutation**; it is a boundary condition. It means the *near* band is the **shallow end** of the λ continuum (λ near 1,
gentle scaling), and only as the crown recedes toward the 60 m voxel handoff (elements shrinking below a pixel) does λ
drop and the full Cook compensation engage. Premise CLEARED, with the binding constraint: **near-band λ must stay high
and near-band survivor growth must stay sub-perceptual.** §2's Audit #2 shows this is achievable rather than a wall.

---

## 2. The paper's algorithm, precisely — with the details our synthesis could not quote

Symbol table (Fig 4, p. 79-3): `a` element surface area · `B` bbox size in pixels · `b` = B scaled so simplification
begins at b=1 · `c` element color · `D` depth complexity · `h` value of b at which λ=½ · `k` elements sampled per pixel
· `N` elements in the object · `r` = a/V · `s` area-scaling correction · `t` transition-region half-width · `V` visible
area · `x` element position in priority order · `α` variance-reduction scale · `λ` fraction of elements included.

Five aspects (§3, p. 79-3): (1) detail level, (2) rendering priority, (3) area preservation, (4) contrast preservation,
(5) smooth animation.

### 2.1 Detail level — λ(distance) (§3.1, p. 79-3)
`λ = λ_size · λ_blur` (Eq 1). With `b = B/B₀` (B₀ = bbox size at which simplification begins — "where the shapes of
individual elements are no longer discernible"; `hB₀` = size at which λ_size=½): **`λ_size = b^(log_h ½)`** (Eq 2, valid
for b<1; λ_size=1 for b≥1). At `h=½`, λ_size=b and the *average number of included elements per pixel is constant*.
**`h` must never exceed ½** (else elements-per-pixel would *drop* as the object shrinks, forcing survivors to grow to
preserve area — wrong). Blur regions can simplify more (λ_blur, "exact formula is not critical").
The **appendix sample code** (p. 79-7) pins the defaults: `h = 0.4`, `k_max = 121`, transition `trans = 0.1`, and
`L = (b≥1) ? 1 : pow(b, log(0.5)/log(h))` — i.e. λ computed exactly as Eq 2.

### 2.2 Rendering priority — the SELECTION method (§3.2, p. 79-3) — GAP #1 in the synthesis, now filled
The synthesis admitted it could not quote whether selection is stratified/importance-weighted. **The paper's base
method is neither — it is a de-correlated RANDOM order:**
- "Assign a random number to each element, then sort the elements by their random numbers. This is usually sufficient
  in practice." Elements are excluded **in a consistent order** so animation is temporally stable.
- The order **must NOT correlate with geometric position, size, surface normal, or color** — "excluding elements from
  top to bottom would be objectionable."
- **Optional stratification:** "Poisson disk sampling or stratified sampling can be used in such a way that similar
  elements (geometrically close, with similar normals) are not close to each other in the rendering priority order;
  this spreads out the visual effects of simplification during animation and allows somewhat more aggressive
  simplification." So stratification's role is to avoid removing a whole spatial clump at once — **not** to bias toward
  interior elements.
- **There is NO silhouette/interior bias in base Cook.** Biasing pruning toward occluded interior elements is a
  *SpeedTree "Jumble"* extension and is named only as **future work** in the paper (p. 79-7 §5: local characteristics
  "such as… proximity to the silhouette edge could be stored in a volume texture," and "instead of determining the
  priority order randomly, we could use an error metric… start with the element closest to the mean and successively
  select the one that gives the best approximation to the statistics"). **We rely on this extension (§5.4); the doc is
  explicit that it goes beyond the base paper.**
- Elements are stored **in priority order** so only the first λN need be read (Fig 6, p. 79-4) — the offline analogue of
  our per-level baked subset.

### 2.3 Area preservation — surface area vs VISIBLE area (§3.3, p. 79-4) — the crux for close-by fidelity
- Surface-area preservation: excluding drops total area from `Na` to `λNa`; scale survivors by `s` with `(λN)(as)=Na`
  ⇒ **`s = 1/λ`** (Eq 3). Linearly that is **√(1/λ) per dimension**; "depending on the type of element, this can be
  done by scaling in one or two dimensions" (their leaf example scales **leaf width**, and the widening is
  imperceptible *because elements are sub-pixel*).
- **But what must be preserved is VISIBLE area, not surface area.** The ratio changes because pruning changes depth
  complexity. With `r = a/V` and `D = Na/V = rN`: visible fraction after n elements is **`1 − (1−r)^n`** (Eq 4);
  simplified with survivors of area `as` it is `1 − (1−rs)^{λn}` (Eq 5). Equating and solving:
  **`s = (1 − (1−r)^{1/λ}) / r`** (Eq 6). Binomial expansion (Eq 7) ⇒ `s ≈ 1/λ` only when λ ≫ r; Fig 8 shows the
  correction only matters "when λ is small" — i.e. **for a high-depth-complexity object, pruning occluded elements
  barely reduces visible area, so survivors need far LESS than 1/λ growth.** This is the single most important formula
  for us and Audit #2 turns on it.

### 2.4 Contrast preservation — α = √λ, and WHEN it matters (§3.4, p. 79-4..79-5) — GAP #2 partly, now filled
Fewer samples/pixel ⇒ higher pixel variance (contrast) by the CLT. Pull each survivor's color toward the population
mean: **`c′_i = c̄ + α(c_i − c̄)`** (Eq 13), which reduces element variance to `α²σ²` (Eqs 14–17). Matching simplified to
unsimplified pixel variance needs `α² = k_simplified/k_unsimplified` (Eq 21); since visible elements/pixel ∝ 1/size,
**`α = √λ`** (Eq 22). With a renderer sample cap `k_max` (e.g. 121):
**`α = √( min(λk₁/b, k_max) / min(k₁/b, k_max) )`** (Eq 23).
**The nuance the synthesis lacked:** per Fig 10 and the text, **contrast correction is a MIDDLE-distance effect only.**
When the object is **close, there is no simplification → α=1** (no-op); when **far, > k_max elements land in each pixel
in both cases → α=1** (no-op). The maximum contrast error is at `b = k₁/k_max`, and it is larger for smaller λ. Two more
points: `c̄` must be computed over the **final shaded** colors, and **each element TYPE gets its own independent
correction** ("if there are… leaves and grass, each type needs its own"; also matters when color correlates with
position, e.g. greener tips vs older interior leaves).

### 2.5 Smooth animation — the exact TEMPORAL FADE (§3.5 + appendix, p. 79-5, 79-7) — GAP #2 remainder, now filled
The synthesis could not quote the fade rule. It is: elements do **not** pop; over a transition band of half-width
**`t = 0.1`** around the current λ, a survivor's **area lerps continuously from its full (scaled) value down to 0** as
its priority position `x` crosses from `λ−t` to `λ+t` (Fig 7b). "The excluded elements gradually fade… by gradually
making the elements either more transparent or smaller." Worked example (p. 79-6): at λ=0.3, elements with x ≤ λ−t=0.2
are fully enlarged by 1/λ=10/3; x > λ+t=0.4 are fully excluded; between, area ramps to 0; the **total surviving area
stays constant** across the fade (the area under each curve in Fig 7b is invariant). The appendix computes exactly this
per element: `aLerp = (x < λ−t) ? 1 : (x < λ+t) ? (λ+t−x)/(2t) : 0`, then `scaleAreaBy(a)` and `scaleContrastBy(α)`.
Because λ = b^(log_h ½) is continuous in distance, the fade is continuous in distance — the anti-popping mechanism.

### 2.6 Scope & limits the paper states (§5, p. 79-7)
Designed for **aggregate** detail (snow, spray, dust, feathers, hair, insect swarms — and plants). **Not** for element
detail (removing surface elements makes holes). Elements must be roughly similar in appearance; the more their size /
normals / color vary, the less aggressive λ can be. Shiny/narrow-highlight surfaces resist aggressive λ (remedy: gentler
h). Results (Figs 1, 11–16): a 320k-leaf plant went 409.3 MB → 3.4 MB and 239.3 s → 1.7 s; a 240M-hair swarm dropped
hair count 88% / memory 84% / time 75%. Shadows had **no popping artifacts because preserving area also preserves each
element's probability of being in shadow** (directly relevant to our voxel-shadow split — see §6.5).

---

## 3. Our crown geometry — the KEY QUESTIONS answered from the code

### (a) Are foliage elements discrete cards/sprays with per-anchor PIVOTS? — YES, cleanly.
The growth grammar emits **`LeafAnchor { pos, quat, scale, hue, age }`** (`src/vegetation/VegTypes.ts:162–170`), one run
of anchors per anchor-level branch (`src/vegetation/Skeleton.ts:225–285`). Every foliage element is built **from** an
anchor via a transform composed at `anchor.pos`:
- **Broadleaf** `buildLeafCluster` (`src/vegetation/LeafMesh.ts:171–196`): fans `clusterSize` (beech 2–3, karst 2–4)
  **independent** leaves, each `_m.compose(anchor.pos, qr, s)` with `s = anchor.scale·(0.8..1.25)`. Leaves are
  **disconnected** (no shared verts) → natural element = **one leaf**, pivot = its rooted base at `anchor.pos`.
- **Conifer** `buildSprayAt` (`:199–210`) → `buildNeedleSpray` (`:85–168`): one **connected** stem-strip +
  30 (spruce) / 88 (pine) single-quad needles, all in the anchor frame `_m.compose(anchor.pos, anchor.quat, 1)`.
  Natural element = **one spray**, pivot = `anchor.pos`.
- **Cards** `buildFoliageCards` (`src/vegetation/FoliageCards.ts:261–321`): one atlas card per anchor, marched from
  `a.pos` along the anchor's `out`, size `s = a.scale·sizeK`. Pivot = `a.pos`.

**The pivot problem the synthesis worried about (baked jitter losing the pivot) does not exist for us**, because we
**bake LODs from the anchor graph, not by decimating a baked mesh.** Per-leaf/per-needle jitter is generated at build
time by a forked RNG *relative to `anchor.pos`*; we never have to recover a pivot from welded positions — we regenerate
survivors from the anchor with a modified `anchor.scale`. This is the Hulm/SpeedTree lesson realized natively and is the
foundation of the whole design.

### (b) Exactly why did the old paths spike/merge? — the position-weld + quadric-optimal-target mechanism.
Two paths have this disease; both are structurally excluded by §5.7.

- **QEM `buildDag.ts` (the classic failure).** It welds a group's triangles into a soup by **quantized position**
  (`weldEps = 1e-5`, `src/nanite/BuildDag.ts:6–7, 27–30, 353–373`), locks only edges used by ≠2 soup triangles
  (`:437–465`), then edge-collapses interior edges to the **Garland–Heckbert quadric-optimal point** (`:493–529+`).
  On disconnected leaf/needle soup this fails two ways at once:
  1. **MERGE:** in a dense crown, verts of *different* leaves/needles fall within `weldEps` and **fuse**. A boundary
     edge (locked) becomes an *interior* edge spanning two formerly-distinct leaves → QEM collapses it → the two leaves
     weld into one **giant merged leaf**. Boundary-locking prevents *cross-group* cracks but does nothing about
     *inter-leaf* welds *inside* a group.
  2. **SPIKE:** thin needle/leaf quads are near-coplanar and near-degenerate, so their area-weighted quadric is
     near-singular → the quadric-optimal collapse target flies **far** from both endpoints, shooting a long thin sliver
     out of the crown. On spruce's thin single-quad needles this is the **super-long spike**.
- **The existing aggregate `buildAggregateDag.ts`, pushed close (a subtler failure).** It avoids QEM but (i) welds the
  baked soup and finds connected-component **islands** (`:191–223`) — a **needle spray is one connected island** (stem
  + needles share verts), so it grows the **whole spray uniformly** about its centroid by up to `growMax = 2.5×`
  (`:303–331`), which **elongates needles and stem** → a milder spruce-spike; and (ii) it uses **surface-area**
  growth `g = √(totalArea/keptArea)` (Eq 3), which **over-grows** dense crowns (Eq 6 says they need far less), producing
  the "fewer, bigger" look that reads as merged up close. This is why its coarse levels look bad near-band and were
  disabled (`maxLevels:1`, `?crownlod0`). Fixable, and §5–§6 fix it.

### (c) Is `HeroDiet`'s stride-and-enlarge already the Cook prune+scale? — Yes in spirit; three flaws for near-band.
`TreeBuilder.ts:82–108` (card LODs): `stride = ceil(anchors.length / target)`, keep `i % stride === 0`, enlarge
survivors `sizeK *= min(sizeCap, √stride·0.9 + 0.12)` — the comment even says *"≈ sqrt(stride) keeps painted coverage."*
That **is** Cook with λ = 1/stride and √(1/λ) growth. But:
1. **Selection is a regular `i % stride`** over branch-traversal order (spatially clustered) — **position-correlated**,
   exactly what Cook §3.2 forbids; it can drop whole small twigs. Base Cook wants a **de-correlated** key.
2. **`sizeCap` (1.9 / 3.1) breaks area preservation** at large stride — survivors under-grow → crown thins. (For
   near-band that under-grow is actually *good* per Eq 6, but it must be principled, not a clamp.)
3. It applies to the **card LOD1/2** path, and there is **no contrast pull and no temporal fade**. The near-crown
   **mesh** path (`meshAnchorTarget`) uses the same regular stride but is run at full density (`leafAnchorTarget=4000`),
   so it does not prune near-band at all.
**Verdict:** reuse the stride-and-enlarge *foundation* but (a) swap the key to de-correlated + interior-biased, (b) use
visible-area growth (Eq 6) instead of a size-cap, (c) apply it to the **mesh** path across 0–60 m, (d) add fade.

### (d) How does √λ contrast map to our atlas + vdata? — as a per-anchor VDATA hue scale; and near-band it's a no-op.
`foliageColor` is a per-species base + per-anchor hue jitter carried in **`vdata.x` scaled by `hueVar`**
(`src/vegetation/FoliageCards.ts:118–133`; `LeafMesh` writes `hue` into vdata.x per vertex, `:62–64,118–119,162–165`).
So the population mean `c̄` = species base, and each element's deviation = its `vdata.x·hueVar`. **Cook's √λ pull = scale
`vdata.x` by √λ at bake** (pull each element's hue toward the species mean) — a trivial per-vertex bake-time edit on the
**mesh** path, **no texel/atlas edit** (the sqrt-encoded atlas is the *card* path, out of scope). **But per §2.4 this
is a middle-distance effect: near-band λ≈1 ⇒ √λ≈1 ⇒ negligible.** So near-band we **skip** contrast correction; it is
only worth wiring for the 30–60 m sub-band as λ drops (and even there, per-type — needles and leaves separately).

### (e) Where does close-by mesh-LOD live, and which band? — 0–60 m mesh; voxels beyond.
Near band **0–60 m** = mesh crown, currently full LOD0 everywhere (§0). **This design coarsens INSIDE 0–60 m** and hands
off to voxels at 60 m unchanged. Target sub-bands (illustrative, to be tuned/profiled):
`0–~15 m` λ≈1 (near-identical, essentially LOD0); `~15–~40 m` λ ramps ~1→~0.5; `~40–60 m` λ ~0.5→~0.3 meeting the voxel.
Voxel path (`VoxelizeCrown.ts`), shadows, and the 60 m handoff are **untouched**.

---

## 4. PREMISE-AUDIT #2 — "close-by can't be nearly identical AND simplified" — put on trial before concluding

The tempting wall: "up close, leaves/needles are resolvable, so scaling survivors by √(1/λ) (e.g. 1.41× at λ=0.5) makes
visibly bigger leaves — you cannot both simplify and look identical." Per CLAUDE.md I do **not** accept this; I go up a
level and interrogate the metric.

**The premise's flaw is the metric — surface area (Eq 3) vs visible area (Eq 6).** The wall assumes survivors must grow
by 1/λ. But that is *surface*-area preservation. What the eye sees is *visible* area, and Eq 6 says for a
**high-depth-complexity** object the required growth is far smaller: `s = (1−(1−r)^{1/λ})/r`, with `s → 1` as depth
complexity rises. A dense crown is exactly high-depth-complexity — many leaves stacked front-to-back per pixel; the
visible silhouette is a **thin shell**, and everything behind it is **occluded overdraw** that costs projection + raster
+ memory (the whale) while contributing ≈0 to the image.

**Therefore the resolution — and it is a genuine dissolve, not a workaround:**
1. **Prune the occluded interior, keep the silhouette shell.** Pruning occluded elements barely changes visible area
   (Eq 4 saturates), so by Eq 6 **survivors need almost no growth** (`s ≈ 1`) → close-by leaves/needles stay at native
   size → **nearly identical**. The win comes from cutting *depth complexity*, which is invisible, not silhouette,
   which is not.
2. **Near-band λ stays high** (§1's boundary): keep most shell elements; only the deep interior thins hard. Even where
   growth is applied it is √(1/λ) with λ≈0.8–1 → 1.05–1.12× — below the perceptual floor for a 0.02 m needle or a
   fanned leaf at ≥10 m.
3. **The existing builder's over-grow is a fixable flaw, not a limit:** `buildAggregateDag` uses Eq 3 (`g =
   √(totalArea/keptArea)`, `:307`) — the crude law that *does* over-grow. Switching to Eq 6 (needs a per-crown depth
   proxy `r`, cheaply estimable from crown coverage at bake) is the concrete fix.

**Conclusion:** "nearly identical close-by" is achievable. It requires biasing pruning to **interior/occluded** elements
(the SpeedTree "Jumble" / paper-§5 extension, NOT base Cook) and using **visible-area** growth. Premise CLEARED; the
design in §5 is built on exactly these two levers. Where a specific detail still cannot be guaranteed, it is surfaced in
§7, not papered over (notably: elements right at the silhouette that the shell test misclassifies; and true occlusion
being view-dependent while our proxy is static).

---

## 5. The adapted design — "Crown LOD by anchor-graph stochastic pruning"

One-line: **at bake, generate the near-crown mesh as a ladder of LOD levels, each pruning a de-correlated,
interior-biased subset of WHOLE anchors (leaves for broadleaf, sprays for conifer), regenerated from the anchor at
native or minimally-grown size; feed the ladder to the crown DAG as pre-pruned levels so the existing crack-free cut
selects the right level by distance; never weld, never edge-collapse.**

### 5.1 Species split (from §3a geometry)
- **Broadleaf (beech/birch/karst).** Element = **one leaf** (or optionally one anchor's 2–4-leaf cluster for coarser
  granularity). Prune whole leaves. Survivor growth about the leaf's rooted base (`anchor.pos`) — Cook broadleaf recipe,
  minimal per §4.
- **Conifer (spruce/pine).** Element = **one needle spray** (the connected stem+needles unit). Prune whole sprays.
  **NEVER dissolve individual needles** (QEM spike) and **NEVER uniform-scale a spray** (elongates needles → the
  aggregate-DAG spruce spike). If any area compensation is needed in the mid sub-band, **widen only** (scale the spray's
  bough-plane footprint / needle *width*, `nw` in `LeafMesh.ts:131`, never `len`) — mirroring the grass `'widen'` mode
  (`BuildAggregateDag.ts:64–72,321–323`) which exists precisely to avoid "multi-metre blade monsters." Near-band,
  prefer no growth at all + a gentler λ (spruce gentlest of all species — it was the worst offender).

### 5.2 Detail level λ(distance) (Cook §3.1)
Map Eq 2 onto our cut. Bake N discrete levels with λ_L (e.g. λ = 1, 0.7, 0.5, 0.35). Each level's DAG `ownError` is set
so its cut lands at the intended distance (the machinery already exists: `AGG_LOD_CFG.errorK`/`?leaflodk`,
`BuildAggregateDag.ts:77–93`, scales the reported error so the cut engages nearer/farther). Choose per-level error so
the ladder spans 0–60 m and each distance octave descends one level, meeting the voxel at 60 m. Keep `h ≤ ½`
equivalence: elements-per-pixel non-increasing as the crown shrinks.

### 5.3 Rendering priority — de-correlated key (Cook §3.2), NOT the regular stride
Per element, key = **hash(quantized element centroid ⊕ seed)** — exactly `hashOf` in `BuildAggregateDag.ts:278–293`,
which is already de-correlated from position and seed-deterministic (satisfies §3.2 and our determinism/bootcache law).
This replaces `TreeBuilder`'s position-correlated `i % stride` (§3c flaw 1). **Nested/monotone keep-sets:** a level's
kept set must be a **superset** of the next-coarser level's, so distance transitions only ever *remove* elements
(prerequisite for the temporal fade and for crack-free DAG parenting). Achieved by keeping the **low-key prefix** and
lowering the λ cutoff per level (the aggregate builder's "keep low-hash prefix until area target" already yields nested
sets, `:294–301`).

### 5.4 Silhouette / interior bias — the SpeedTree "Jumble" extension (explicitly beyond base Cook — §2.2)
This is the core enabler of §4, and I state plainly it is **not in the base paper** — base Cook is pure de-correlated
random; biasing is SpeedTree "Jumble" + the paper's §5 future work. We fold an **interiority weight** into the key so
occluded interior elements sort *first* (pruned first) and silhouette-shell elements sort *last* (kept):
- Cheap static proxy computable at bake from data we already have: **radial shell fraction** `ρ = |anchor.pos −
  crownCenter| / crownRadius` (crownCenter/Radius from `Skeleton.growSkeleton`, `src/vegetation/Skeleton.ts:332–351`),
  plus **`anchor.age`** (already ≈ interiority: inner/lower crown is older, `Skeleton.ts:282`). Keep-priority ↑ with ρ
  (shell) and freshness; prune-priority ↑ with low ρ (interior) and age.
- Effective key: `sortKey = hash(centroid) · w(ρ, age)` (or bucket by an interiority tier, de-correlated-random within a
  tier — the §2.2 stratification, so we neither strip a whole clump nor carve a hollow shell).
- **This keeps the silhouette + apparent density constant while cutting invisible depth complexity** — the whale win and
  the fidelity guarantee are the same mechanism.

### 5.5 Area preservation — visible area (Eq 6), not surface area (Eq 3)
Replace `BuildAggregateDag`'s `g = √(totalArea/keptArea)` (`:307`) with `s = (1−(1−r)^{1/λ})/r` (Eq 6), `r = a/V` per
crown. Estimate `V` (visible/silhouette area) and depth complexity `D = rN` at bake from a cheap coverage pass (render
the crown, or use the alpha-derived `D` the paper suggests, p. 79-4). For dense crowns Eq 6 yields `s ≈ 1` → survivors
essentially native size (§4). Apply the affine **about each element's own pivot** (`anchor.pos`), never a shared
centroid, so survivors do not "fly away" (the SpeedTree/Polycount pivot lesson, synthesis §1) — our anchor pivots make
this exact. Broadleaf: uniform (or width-biased) scale; conifer: width-only or none (§5.1).

### 5.6 Contrast preservation — √λ vdata pull, near-band no-op (Cook §3.4; §3d)
Near-band: **skip** (λ≈1 ⇒ α≈1, §2.4). For the 30–60 m sub-band optionally scale `vdata.x` (hue) by `α=√λ` per element
**per type** (needles vs leaves separately, §2.4) at bake. Cheap, no atlas edit. Listed as a phase-3 refinement, not a
phase-1 requirement — its value for our look is marginal near-band and should be A/B'd before shipping.

### 5.7 The structural NO-SPIKE / NO-MERGE guarantee (contrast with §3b)
Precisely why this cannot reproduce the old failures:
1. **We never weld across elements and never edge-collapse.** There is **no `weldEps` position-weld** fusing distinct
   leaves (the QEM/aggregate MERGE mechanism, §3b) and **no quadric-optimal collapse target** (the SPIKE mechanism).
2. **Survivors are regenerated from the anchor graph**, so a survivor's vertices are **bit-identical** to LOD0's (or a
   clean affine about its own pivot). **No vertex is ever synthesized *between* two elements.** An element is
   present-at-native (optionally gently pivot-scaled) or absent — atomic, like the aggregate builder's per-island
   keep/drop (`BuildAggregateDag.ts:22–26` crack-note) but **without the weld step that risks cross-element fusion.**
3. **Conifer sprays are pruned as whole units and never uniformly scaled** (§5.1), so needles cannot elongate → no
   spruce spikes even in the limit.
This is **strictly stronger** than the shipped `buildAggregateDag`, because pruning in **anchor space** (before meshing)
removes the weld/island step entirely — the one place cross-element fusion and spray-elongation could still enter.

### 5.8 Pipeline integration + expected reduction
- **Generate (vegetation space).** Extend `buildTree`'s foliage-mesh loop (`TreeBuilder.ts:109–129`) to emit the
  `foliageMesh` **ladder**: for each level, compute the kept anchor set (§5.3–5.4) and regenerate
  `buildLeafCluster`/`buildSprayAt` for survivors with the level's per-anchor scale (§5.5). Attach interiority to
  `LeafAnchor` in `Skeleton.ts` (or derive it in `TreeBuilder` from `pos`/`age`). Determinism preserved (seed-hash key);
  **bootcache `SRC_HASH` must include the new params** (`BootCache.ts`).
- **Feed the DAG.** Two options (surfaced in §7): **(A, preferred)** hand the pre-pruned ladder to the crown DAG so the
  builder's own weld/island/grow is a **passthrough** (`targetRatio=1`, grow=1) and pruning lives in vegetation space —
  cleanest, keeps the crack-free cut records the raster depends on; **(B)** keep `buildAggregateDag` doing the pruning
  but feed it **anchor/island metadata** so it prunes by anchor (not welded island) and skips the weld — smaller code
  change, retains a residual weld. Either way, **re-enable `maxLevels>1`** (`WorldRegistry.ts:621,1335`) and **relax
  `?crownlod0`'s `forceDescend`** so leaf clusters may emit at coarser levels **inside** the mesh band
  (`NaniteCull.ts:1183–1194`) — the single switch that turns the whole thing on.
- **Fewer verts → the whale.** Pruned crown levels feed the same `projVertBuf`/cluster pipeline (`DAG_VERT_STRIDE=12`,
  `GeometryRegistry.ts:97`; `LEAF_MATCLASS=4`). Expected: near-band interior thin of ~2–4× on occluded depth complexity
  (λ≈0.25–0.5 interior, shell kept) → roughly **halve-to-quarter the near-crown vert/cluster count** → large cut in
  `projVertBuf` (1.13 GB), `nanProjectVerts` (30%), `nanMidRaster` (14.5%) at near-identical silhouette. **These are
  projections; per the FRESH-PROFILING standing rule, implementation must re-profile the current code and confirm on
  the canonical world scene — do not ship on these estimates.**
- **Band + handoff.** Coarsen 0–60 m only; hand to voxels at 60 m; **shadows/voxels unchanged** (leaf mesh stays
  `castShadow:false`; voxel crown owns shadows). Cook's own result that area-preservation preserves shadow probability
  (§2.6) means the mesh↔voxel shadow seam is not disturbed by mesh-side pruning.

---

## 6. Appearance validation plan (the user's "nearly identical," gated on the eyeball per project law)
1. **Per-species side-by-side, static**, at 5 / 10 / 20 m: new LOD vs full LOD0 (`?crownlod0=1` vs the re-enabled path).
   Gate: identical silhouette + apparent density, **no spikes, no giant merged leaves.** **Spruce FIRST** (worst
   offender historically; confirm zero elongation).
2. **Moving camera 0 → 60 m**, per species: **no popping** at level transitions; smooth into the voxel handoff. If pop
   is visible, that is the §7-risk-1 temporal-fade trigger, not a tuning failure.
3. **Hollow-shell check** (SpeedTree "Jumble" failure): orbit a pruned near crown — interior bias must not carve a
   visibly empty core at grazing/looking-up angles.
4. **Contrast/brightness A/B** (only if §5.6 is wired): pruned-with vs pruned-without √λ pull at 30–60 m; keep it only if
   the eye prefers it.
5. **Fresh profile** on the canonical world scene (`?scene=world&nanite=1&dpr=2&…`, memory task76 URL) confirming the
   `projVertBuf`/`nanProjectVerts`/`nanMidRaster` drop — capture → **manual Xcode export** → `run_all.sh` per the
   profiling law. Verify no regression in oblique/aerial poses.

---

## 7. Risks & open questions (surfaced honestly — not resolved here)
1. **Temporal fade is the biggest new piece.** The DAG gives *discrete* levels; Cook's continuous size-lerp (§2.5) needs
   a **per-element size lerp in the shader** keyed on distance-to-cut (SpeedTree's 3D-position+3D-LOD vertex lerp). Open:
   is discrete-level + the existing distance dither good enough near-band (small λ steps), or is the shader fade
   required to kill popping? Phase-4 candidate; may gate "nearly identical while moving."
2. **Graph-bake (A) vs mesh-prune (B) integration (§5.8).** (A) is cleaner (exact pivots, zero weld) but touches the
   vegetation builders and the ladder plumbing; (B) reuses `buildAggregateDag` but keeps a residual weld (small
   cross-element-fusion risk remains). **Needs a user/architecture call.**
3. **Conifer needle elongation** is the central spruce risk. The rule (prune whole sprays, never uniform-scale, widen-
   only if anything) is sound, but the exact per-species λ ramp and whether *any* mid-band widening is acceptable up
   close is unproven — must be eyeballed (validation §6.1, spruce first).
4. **Visible-area growth (Eq 6) needs a per-crown depth proxy `r = a/V`.** Estimating `V`/`D` at bake (coverage pass or
   alpha-derived D) adds a build step; if skipped we fall back to Eq 3 with a cap (the current over-grow). Open: cheapest
   faithful estimate of `r`.
5. **Interior-bias proxy is static; true occlusion is view-dependent.** Radial-shell + age is cheap but will
   misclassify some silhouette-adjacent-but-deep elements. Open: is the static proxy enough, or do we need a bake-time
   occlusion/coverage pass (Neubert view-optimized / paper §5 error-metric ordering)? Elements *exactly at* the
   silhouette that the shell test drops are the most visible failure mode.
6. **Per-species tuning + determinism + cache.** λ ramps, interior weight, growMax per species must stay
   seed-deterministic and be folded into `bootcache SRC_HASH`; a miss silently serves stale crowns.
7. **Contrast pull value is marginal near-band** (§5.6) — do not over-invest; A/B before shipping it at all.

---

## 8. Phased implementation sketch (AFTER user review — nothing below is to be built yet)
- **Phase 0 — this design review.** Resolve open questions 1, 2, 4, 5 with the user before coding.
- **Phase 1 — anchor-graph LOD ladder (vegetation space).** `src/vegetation/Skeleton.ts` (attach interiority ρ to
  `LeafAnchor`), `src/vegetation/LeafMesh.ts` (accept a keep-set + per-anchor scale in `buildLeafCluster`/`buildSprayAt`),
  `src/vegetation/TreeBuilder.ts` (emit the `foliageMesh` ladder with de-correlated interior-biased keep-sets, §5.3–5.5),
  `src/vegetation/VegLibrary.ts` (drive levels; keep full density at L0). Nested keep-sets; seed-deterministic.
- **Phase 2 — feed the DAG + flip the switch.** `src/nanite/WorldRegistry.ts` (build the crown DAG from the ladder;
  `maxLevels>1`), `src/nanite/BuildAggregateDag.ts` **or** a thin new `BuildCrownLodDag.ts` (passthrough option A / anchor
  metadata option B), `src/nanite/NaniteCull.ts` (relax `crownLod0` `forceDescend` so crowns may emit coarser inside
  0–60 m), `src/nanite/BootCache.ts` (`SRC_HASH`). Reuse `AGG_LOD_CFG.errorK`/`?leaflodk` to place cuts across 0–60 m.
- **Phase 3 — visible-area growth (Eq 6) + optional √λ vdata contrast (per type) + optional mid-band width-widen.**
  In the ladder baker.
- **Phase 4 (optional) — shader temporal fade** (per-element size lerp about pivot, keyed on distance-to-cut). Leaf
  material / `src/nanite/raster/*` + a vdata fade channel.
- **Phase 5 — validation + fresh profile** (§6): spruce first, `?crownlod0` A/B, re-profile the whale on the world scene.

## References
- Cook, Halstead, Planck, Ryu, *Stochastic Simplification of Aggregate Detail*, ACM TOG 26(3) Art. 79, SIGGRAPH 2007 —
  `docs/mobile-gpu-perf/1275808.1276476 (1).pdf`. Cited above by section/page/equation (§3.1–3.5, Eqs 1–23, Figs 4–16,
  appendix sample code).
- Synthesis: `docs/mobile-gpu-perf/deep-research-crown-mesh-simplification.md` (SpeedTree, meshoptimizer, UE5, Lacewell,
  Hulm, FSA/PLU; §2.2 and §2.5 above fill its two admitted gaps from the PDF).
- Code (all paths absolute-from-repo-root): `src/vegetation/{Skeleton,LeafMesh,FoliageCards,TreeBuilder,VegTypes,Species,VegLibrary}.ts`;
  `src/nanite/{BuildDag,BuildAggregateDag,DagHierarchy,VoxelizeCrown,WorldRegistry,NaniteCull,GeometryRegistry,BootCache}.ts`.
</content>
</invoke>
