# State of the Art & Shipped Techniques: Fast LOD Simplification of Tree-Crown / Foliage Card Meshes

## TL;DR
- **The single winning idea across research and shipped games is "prune-and-preserve," not edge-collapse:** remove whole cards/leaves stochastically or by importance, then compensate by scaling up survivors (area preservation) and reducing per-element contrast (color preservation). This keeps card SIZE, spatial DENSITY, and silhouette constant while cutting count — and it structurally cannot produce giant merged leaves or cross-gap spikes. The canonical source is Cook, Halstead, Planck & Ryu, "Stochastic Simplification of Aggregate Detail" (Pixar, ACM TOG 26(3) Art. 79, SIGGRAPH 2007), and it is exactly what SpeedTree ships.
- **QEM/edge-collapse (meshoptimizer, Blender Decimate collapse, classic Nanite) genuinely fails on card soup** because it assumes a connected manifold; on disconnected quads it welds cards across gaps or gets "stuck." The fixes used in practice are: per-card operations (leaf-collapse with area constraint — the FSA/PLU family), component-aware pruning (`meshopt_SimplifyPrune`), or abandoning triangle-decimation for voxel/impostor representations (UE5 Nanite Voxels).
- **Species split is real and simple:** broadleaf = stochastic card pruning + survivor scaling (SpeedTree "leaves shrink away, survivors grow + jumble"); conifer needle/branch sprigs = keep as instanced sprig geometry and drop whole sprigs / reduce branch spine segments by volume, never dissolve needles individually.

---

## Key Findings

1. **Stochastic pruning is the "solid gold" method and it explicitly beats the brightness/coverage trap.** Cook et al. keep a survival fraction λ of elements, scale each surviving element's linear size by √(1/λ) (so per-element area grows by 1/λ, preserving total coverage), and pull each texel's color toward the element mean by c′ = c̄ + √λ·(c − c̄) to hold aggregate contrast and mean color constant. This is the precise, named answer to "why naive removal looks darker/brighter/thinner."
2. **SpeedTree ships exactly this** for leaves ("fewer and bigger leaf instances," survivors scaled up via a 3D-position + 3D-LOD vertex lerp), and uses a different strategy for branches/fronds (remove by volume/area, shrink outlying geometry). The "Grow Scale" + "Jumble" parameters exist specifically to preserve silhouette/fullness and avoid the hollow-shell artifact.
3. **QEM on soup is a documented failure**, both in the foliage-simplification literature ("general simplification methods do not properly work with isolated polygons") and in meshoptimizer/Nanite practice (simplifier gets "stuck," disconnected triangles, blobbing).
4. **UE5's answer (2025–26) abandons triangle decimation for aggregate foliage:** Nanite Voxels voxelize disconnected pieces to preserve silhouette at distance, plus Nanite Assemblies (micro-instanced twigs/needles) and Nanite Skinning (bone-based wind instead of WPO). Explicitly the "disconnected pieces lose their shape" problem being solved — but the mid representation is voxels, not cards.
5. **Billboard clouds are the mid-LOD bridge**, and the foliage-specific variant (Lacewell et al.) was built precisely because "traditional triangle reduction methods break down for foliage with its small, disconnected pieces of geometry." Octahedral impostors are the modern shipped form for the far band.
6. **The FSA/PLU/HUO family** (Remolar 2002, Zhang & Blaise 2003, Deng 2009, and later viewpoint-driven variants) is the academic line that does per-leaf collapse WITH an explicit area-preservation constraint and Hausdorff-distance similarity — a card-decimation approach that, unlike QEM, keeps leaves leaf-sized. But it has documented drawbacks (burrs, new leaf sometimes smaller than originals).

---

## Details

### 1. Foliage/vegetation LOD as shipped in real games/engines

**SpeedTree — the reference shipped method (SOLID GOLD).**
- Runtime LOD docs (https://docs9.speedtree.com/sdk/doku.php?id=level-of-detail): "SpeedTree's leaf LOD is based on using fewer and bigger leaf instances as the camera moves away from the tree. Some percentage of the leaf instances are gradually shrunk away and the remaining instances are scaled up." Implemented as a vertex format carrying both a 3D position and a 3D LOD value, lerped on a CPU-supplied LOD value — so it's smooth, animatable, and stays a mesh. Relevance: **high** — this is literally the shipped answer to the exact problem, and it never merges cards.
- Modeler LOD docs (https://docs8.speedtree.com/modeler/doku.php?id=lod): branches/fronds are removed "based on their size and how 'hidden' they are"; leaves shrink away while survivors grow ("Grow Scale"), and "Jumble" adds noise to removal to avoid the interior being "carved out, leaving just a shell." Explicitly names silhouette/fullness preservation. Relevance: **high**.
- Known limitation (Polycount, https://polycount.com/discussion/232518/speed-tree-cards-moves-away-in-lods): if leaf pivots/anchors are placed wrong, the scale-up "looks like the tree is exploding and cards fly away." The survivor-scaling trick depends on correct per-card anchor points. Relevance: **medium** (failure mode + fix).

**Horizon Zero Dawn (Guerrilla, Decima) — GDC 2018 "Between Tech and Art: The Vegetation of Horizon Zero Dawn."**
- GDC Vault: https://www.gdcvault.com/play/1025530/ ; free video: https://www.youtube.com/watch?v=wavnKZNSYqU ; slide transcription: https://docslib.org/doc/8059789/the-vegetation-of-horizon-zero-dawn
- Relevant detail: trees have three motion levels (whole-tree bend, branch, leaf) driven by a global wind force field, with baked-in per-vertex data (bend/rigidity, distance-to-trunk, distance-to-branch, baked AO). This is the animatable-mesh wind pipeline your LOD output must feed. Stronger on placement/wind/shading than on the crown-decimation algorithm specifically. Relevance: **medium**.

**Ghost of Tsushima (Sucker Punch) — GDC 2021.**
- Grass is GPU-generated Bezier blades (not cards); trees/shrubs are joint-rigged with separate trunk/branch controls and wind from 2D Perlin noise, with billboard trees only in the far background. PS Blog: https://blog.playstation.com/2021/01/12/how-stunning-visual-effects-bring-ghost-of-tsushima-to-life/ ; GDC procedural grass PDF: https://archive.thedatadungeon.com/ghost_of_tsushima_2020/documents/gdc_2021/gdc_2021_procedural_grass_in_got.pdf . Relevance: **low–medium** (excellent for the wind/animation model, not for crown-card decimation).

**Unreal Engine 5 Nanite Foliage (2025–26, Experimental).**
- Docs: https://dev.epicgames.com/documentation/unreal-engine/nanite-foliage and https://dev.epicgames.com/documentation/unreal-engine/nanite-assemblies
- Directly names the problem: "Nanite Voxels can preserve the general silhouette of aggregate geometry (disconnected pieces) during simplification at distance." Prior Nanite mitigated foliage thinning by "adding surface area back into surviving triangles when the simplification process would eliminate them" (the "Preserve Area" flag) — an area-preservation echo of Cook. Assemblies micro-instance twigs/needles ("down to individual pine needles"); Nanite Skinning does wind via a bone hierarchy so cluster bounds stay tight, unlike WPO. Caveat: experimental, full release estimated mid-2026 by Epic; the mid representation becomes voxels, not animatable cards, partially diverging from your "stays a card mesh" constraint. Relevance: **high** (names the failure mode, ships a solution) with the voxel caveat.
- "Preserve Area" is imperfect: devs report leaves still disappear at distance and "even Preserve Area won't help much" for individually-meshed tiny leaves (https://medium.com/@shinsoj/notes-on-foliage-in-unreal-5-3522b6eb159f). Relevance: **medium** (known limits).

### 2. Stochastic pruning / stochastic simplification of aggregate detail (SOLID GOLD)

**Cook, R.L., Halstead, J., Planck, M., Ryu, D. — "Stochastic Simplification of Aggregate Detail," ACM Transactions on Graphics 26(3), Article 79 (July 2007), 8 pp. / SIGGRAPH 2007 / Pixar Tech Memo #06-05a. DOI 10.1145/1275808.1276476.**
- Landing/abstract: https://graphics.pixar.com/library/StochasticSimplification/ ; ACM: https://dl.acm.org/doi/10.1145/1275808.1276476 . Abstract (ACM DL): "Scenes are rendered by randomly selecting a subset of the geometric elements and altering those elements statistically to preserve the overall appearance of the scene."
- **How they beat the brightness/coverage trap (the crucial detail):** keep survival fraction λ; scale each surviving element's linear size by √(1/λ) so total rendered area/coverage is unchanged; and adjust each surviving element's texel colors as c′ = c̄ + √λ·(c − c̄) (c̄ = mean color) so aggregate mean color AND contrast stay constant. Without the area scaling the crown looks thinner and more background shows through; without the √λ contrast pull, aggregate variance rises (fewer, larger samples per pixel) and it looks wrong. Verbatim reproduction of both formulas (Hasselgren et al. 2021, arXiv:2104.03989, Appendix F): "we stochastically remove 90% (λ=0.1) of the leaves… and adjust the element area of the reduced model by scaling each leaf uniformly with a factor √(1/λ). Finally, the contrast of each leaf texture is adjusted by modifying the color of its texels as c_i′ = c̄ + √λ·(c_i − c̄), where c̄ is the average color of the texture." Corroboration: https://pmc.ncbi.nlm.nih.gov/articles/PMC7512728/ ("reduced down to a certain fraction… the remaining geometry is scaled so that the total area of rendered surfaces is equal to the original area"). The amount of simplification varies with screen size, motion blur, and depth of field, with continuous fades to avoid popping. Relevance: **high — this is the core paper.**

**Neubert, Pharr et al. — optimized/view-optimized pruning (2011).**
- Improves Cook with model-specific geometry-reduction and optimized scaling functions, introduces Precision/Recall as a rendering-quality metric to predict better scaling values, and adds view-optimized (occlusion-aware) selection instead of purely stochastic. Overview: https://kit.academia.edu/BorisNeubert . Relevance: **high** (directly refines the size/coverage scaling that causes the brightness bias).

**GPU / engine adaptations.** Castelló et al. "view-dependent pruning for real-time rendering of trees" builds a GPU pruning order on Cook & Halstead; alpha-to-coverage is the standard companion for the many transparent card texels (GPU Gems 3 Ch. 4, https://developer.nvidia.com/gpugems/gpugems3/part-i-geometry/chapter-4-next-generation-speedtree-rendering). Relevance: **medium**.

### 3. Card/cluster decimation and billboard clouds

**Lacewell, Edwards, Shirley, Thompson — "Stochastic Billboard Clouds for Interactive Foliage Rendering," Journal of Graphics Tools 11(1), 2006 (SOLID GOLD for the mid band).**
- PDF: https://azslide.com/download/stochastic-billboard-clouds-for-interactive-foliage-rendering_5a3edfac1723dde562efc736.html ; abstract: https://www.tandfonline.com/doi/abs/10.1080/2151237X.2006.10129213
- Explicitly: "Billboard clouds offer an alternative to traditional triangle reduction methods, which break down for foliage with its small, disconnected pieces of geometry." Stochastic search for a set of billboards + smooth LOD transitions. Demonstrated on a **spruce (conifer)**: base mesh 20,610 triangles at 60 FPS → BC1 of 78 quads (156 triangles) with two 2K×2K textures (N=600, ε=0.04) at ~300 FPS; a coarser BC2 of 43 quads (86 triangles) with a 1K×1K texture (N=300, ε=0.10). Relevance: **high** (names the failure; handles conifer; mid-LOD).

**Décoret, Durand, Sillion, Dorsey — "Billboard Clouds for Extreme Model Simplification," SIGGRAPH 2003.**
- https://maverick.inria.fr/Publications/2003/DDSD03/ ; PDF https://graphics.cs.yale.edu/sites/default/files/bc03_0.pdf . Key property: "does not require connectivity information" — greedy plane selection over a density function in plane space. Foundational but aimed at extreme/flat simplification. The 2004 "Hybrid Billboard Clouds" poster (https://history.siggraph.org/wp-content/uploads/2023/01/2004-Poster-030-Bromberg-martin_Hybrid-Billboard-Clouds-for-Model-Simplification.pdf) explicitly shows QEM leaving "jarring cracks" and merging leaves on a tree, and proposes a hybrid mesh+billboard fix. Relevance: **high** for connectivity-free clustering; **medium** as an animatable mid-LOD.

**Octahedral impostors (modern shipped far-LOD).** Shaderbits octahedral impostors (https://www.shaderbits.com/blog/octahedral-impostors) — superior to flat billboards because they match the 3D mesh from many angles including top-down. George Hulm's Houdini tree suite uses octahedral hemisphere impostors as the final stage. Relevance: **medium** (far band, not animatable mid-mesh).

### 4. Disconnected-geometry-aware simplification (avoiding cross-component bridging)

**meshoptimizer** (https://github.com/zeux/meshoptimizer, https://meshoptimizer.org/v1.html):
- `meshopt_SimplifyPrune` (verbatim, meshoptimizer.org v1.0) "removes individual small disconnected parts of the mesh regardless of their topology, and that process is integrated with the main simplification flow"; its `target_error` "controls the cutoff of component radius… (e.g., 1e-2f will remove components under 1%)." This is the soup-safe direction — drop whole cards, don't merge. `vertex_lock`/`LockBorder` prevent moving boundary vertices; `meshopt_partitionClusters` "will not merge disconnected clusters." `simplifySloppy` ignores topology for far LODs.
- Documented QEM failure on soup: docs warn "For meshes with inconsistent topology or many seams, such as faceted meshes, it can result in simplifier getting 'stuck' and not being able to simplify the mesh fully. Therefore it's critical that identical vertices are 'welded' together." Discussion #750 (https://github.com/zeux/meshoptimizer/discussions/750): with disjoint triangles "at higher DAG levels all meshlets are disconnected single triangles" and "the only solution seems to be randomly removing triangles (!)" — an inadvertent rediscovery that stochastic pruning is what soup needs. Relevance: **high** (open-source; names the exact failure and the workable modes).

**Appearance-Preserving Simplification (Cohen, Olano, Manocha, SIGGRAPH 1998)** and **Simplification Envelopes (Cohen et al. 1996)** — classic appearance/error-bounded methods; useful conceptual background but manifold-oriented. Relevance: **low–medium**.

### 5. Silhouette- and density-preserving foliage decimation that stays a mesh

**George Hulm — "Tree Suite" Houdini pipeline (SOLID GOLD practitioner example).**
- https://b00merang.artstation.com/blog/Anjj/tree-suite-part-2-seeing-the-wood-for-the-trees
- Explicitly rejects mesh decimation: "in order to preserve the apparent density and silhouette of the model from afar, it was important to retain all of the branches and leaf cards right up until the Imposter transition… I took a completely different route to traditional mesh decimation." LODs are generated procedurally from the node graph (per-node edge-loop count, mesh-reduction %, curve-simplification %, polygonal-vs-planar branch toggle), and leaf cards are constrained to never leave the confines of their triangle. 4-stage example ~80k → ~40k → ~6.5k tris → octahedral impostor, in-game at 60fps, across 5 species. Relevance: **high** (density+silhouette preservation, animatable card mesh, fast bake, multi-species).

**Hasselgren, Munkberg, Lehtinen, Aittala, Laine — "Appearance-Driven Automatic 3D Model Simplification," EGSR 2021 (NVIDIA).**
- https://arxiv.org/abs/2104.03989 ; code https://github.com/NVlabs/nvdiffmodeling . Differentiable-rendering, image-loss-driven joint optimization of mesh + materials; explicitly compares against and can initialize from billboard clouds / stochastic simplification on the Disney Moana foliage asset. Produces a mesh at ~0.4% triangle count matching appearance. Caveat: heavier than a "minutes per tree" bake and needs a good initial mesh/UVs. Relevance: **high** (appearance/coverage-preserving, mesh output) with a build-cost caveat.

### 6. Species-aware: conifer sprigs vs broadleaf cards

- **Broadleaf:** stochastic card pruning + survivor scaling (Cook 2007; SpeedTree leaf LOD; FSA/PLU leaf-collapse with area constraint).
- **Conifer/spruce:** treat needle/branch sprigs as instanced units and remove whole sprigs / reduce branch-spine segment counts by volume, rather than dissolving needles. SpeedTree branch/frond LOD removes nodes "based on size and how hidden," reducing spine segments (https://docs.speedtree.com/doku.php?id=lod). Lacewell's stochastic billboard clouds were demoed specifically on a spruce. UE Nanite handles "down to individual pine needles" via Assemblies + voxels. Practitioner note (https://medium.com/@shinsoj/notes-on-foliage-in-unreal-5-3522b6eb159f): "Artists often have some doubts about conifer trees, but they work just fine… focus on the overall shape." Deng et al. 2009 gave two HUO-based algorithms — one for broad leaves (quads) and one for thin/needle leaves (lines). Relevance: **high** (explicit per-species treatment).

### 7. Open-source tools and their limits

- **meshoptimizer:** see §4. Best soup path = `SimplifyPrune` + locks + sloppy; do NOT expect plain `meshopt_simplify` (QEM) to behave on card soup.
- **Blender Decimate:** "Collapse" (QEM) produces overlapping faces/slivers and merges across gaps on foliage (bug reports T37121, T47998); "Planar" is for CAD flats and is slow/wrong on organic meshes (T69860). Use vertex-group masking / Delimit to restrict, but there is no foliage-aware mode. Docs: https://docs.blender.org/manual/en/latest/modeling/modifiers/generate/decimate.html . Relevance: **medium** (shows why generic tools fail).
- **FSA/PLU/HUO academic line:** Remolar et al. Foliage Simplification Algorithm — per the survey (Gasch, Remolar, Chover & Rebollo, *Entropy* 20(4):213, 2018, https://www.mdpi.com/1099-4300/20/4/213 ; PMC7512728): "In 2002, Remolar et al. proposed the first method for foliage simplification that deals with polygons, the foliage simplification algorithm (FSA). Two leaves disappear and are replaced by a new one that preserves an area similar to the original ones… uses an error function… taking the Hausdorff distance between two leaves and their planarity into account." PLU = Zhang & Blaise 2003 (Progressive Leaves Union); Deng et al. 2009 split broad-vs-thin leaves. The WSCG paper (Remolar & Chover et al., "View-Dependent Multiresolution Model for Foliage," *Journal of WSCG* 11(1):370–378, 2003, http://wscg.zcu.cz/wscg2003/Papers_2003/J23.pdf) states "the unit of information managed by this scheme is the leaf, four vertices determining two triangles" — i.e., the simplification unit is the card, not the vertex. Documented drawbacks: FSA's new leaf can be smaller than originals (area not always kept) and produces "burrs." Relevance: **high** (directly on-problem) but older and not drop-in for a modern engine.
- **Procedural generators (Sapling/tree-gen, SpeedTree):** best practice is to bake LODs from the generative graph (like SpeedTree/Hulm) rather than decimate the output mesh — you keep per-card identity, size, and anchors. Relevance: **medium–high**.

---

## Recommendations

**Stage 1 — Build the mid-LOD by stochastic pruning + survivor compensation (do this first).**
- Implement Cook et al. per species: choose survival fraction λ per LOD band (target 2–8× reduction → λ ≈ 0.5 down to ≈ 0.125). Scale surviving cards' linear size by √(1/λ) and pull card-texture texels toward their mean by c′ = c̄ + √λ·(c − c̄). Highest-value, lowest-risk step; it directly preserves size, density, silhouette, coverage, and average brightness, and it is structurally incapable of creating giant leaves or cross-gap spikes.
- Anchor each card at its own pivot (SpeedTree's lesson) so scaling doesn't make cards "fly away." Keep both base and LOD positions in the vertex format and lerp for smooth transitions.

**Stage 2 — Species-specialize.**
- Broadleaf: prune individual leaf cards.
- Conifer: prune whole needle/branch SPRIGS as units and additionally reduce branch-spine segment counts by volume; never dissolve individual needles. Use Neubert's PR-based scaling to auto-tune the grow factor.

**Stage 3 — Choose the decimation backend deliberately.**
- If using meshoptimizer, use `meshopt_SimplifyPrune` + `vertex_lock`/`LockBorder`; treat each card as a locked component. Do NOT rely on plain QEM `meshopt_simplify`. Blender Decimate-Collapse is unsuitable for the crown (fine for the trunk).
- Prefer baking LODs from the procedural graph (SpeedTree/Houdini style) over decimating the final mesh whenever the source is procedural.

**Stage 4 — Beyond the mid band.**
- For the far band, transition to a foliage billboard cloud (Lacewell) or octahedral impostor, not a flat billboard.
- If on UE5 and able to accept Experimental status, evaluate Nanite Voxels + Assemblies + Skinning as an alternative that solves the disconnected-silhouette problem natively — but note the mid representation is voxels and wind moves to bones, which changes your animation pipeline.

**Benchmarks/thresholds that change the plan:**
- If pruned crowns look too "holed" at mid distance → raise λ and increase survivor grow factor, or bias pruning toward interior/occluded cards (SpeedTree "Jumble"/weight; Neubert view-optimized selection).
- If aggregate looks darker/lighter after pruning → your area (√(1/λ)) or contrast (√λ) compensation is off; verify both are applied.
- If build time must stay in minutes/tree → stay with analytic prune-and-scale; only reach for differentiable optimization (Hasselgren) if you need best-in-class appearance matching and can afford per-asset optimization time.

## Caveats
- The exact Cook et al. (2007) full-text PDF is currently gated (Pixar link redirects to a landing page; ACM/docplayer block automated fetch). The two core formulas (area scale √(1/λ) and contrast c′ = c̄ + √λ·(c − c̄)) are quoted verbatim from Hasselgren et al. 2021 Appendix F and corroborated by the MDPI/PMC survey; the paper's precise wording on stratified "systematic vs. random" selection and the exact temporal-fade mechanism could not be quoted from an open copy and should be verified against the original if that detail matters.
- UE5 Nanite Foliage is Experimental (full release estimated mid-2026 by Epic); its voxel mid-LOD and bone-based wind diverge from a "stays an animatable card mesh" requirement.
- Several game talks (Horizon, Tsushima) are excellent on wind/animation and placement but comparatively thin on the specific crown-card decimation algorithm; treat them as pipeline context, not as the decimation recipe.
- The FSA/PLU/HUO line is directly on-problem but older; expect to reimplement rather than drop in, and mind FSA's documented "new leaf smaller than originals" and "burr" artifacts.
