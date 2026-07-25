# Grass reconstruction mathematics: exploration summary and stop boundary

Date: 2026-07-22  
Status: **exploration stopped; no comprehensive reconstruction has been proved**  
Active visual verdict: **the live low-oblique grass still stretches and presents the wrong perspective**

This file consolidates the mathematical work that followed the rejected live
reconstruction. It exists so that the next effort does not repeat the same
approximations, mistake a CPU gate for visual progress, or lose the few results
that are actually proved.

The raised `1.176 m` hardware-raster terrain copy is gone and is not part of any
candidate below. None of the offline gates in this summary fixed the live
shader. The runtime remains the rejected reference implementation.

## 1. Why these explorations were done

The visible failure was not a colour or ordinary filtering defect. At low
oblique angles, plants became widening triangles, long streaks, radial fans,
flat sheets, and distance-growing wrong-view patterns. Top-down views were much
better. That evidence pointed at the ray reconstruction itself.

The work therefore tried to answer one question before touching WebGPU again:

> Can an arbitrary live ray obtain one coherent first interaction from a very
> small fixed number of precomputed records, without marching, looping,
> traversing geometry, adding a raised shell, or increasing resident memory
> beyond the existing approximately 51.1 MB cap?

CPU gates were used because they can isolate this mathematical question from
Three.js, WebGPU, depth formats, terrain rasterisation, and shader precision.
They traced the accepted source mesh and evaluated the exact formula that a
prospective fixed-cost shader would use. In principle this was the correct role
for an offline gate: falsify bad mathematics before paying for another live
shader attempt.

In practice, the track continued through too many approximation families after
the central information deficit was already clear. The activity became a
sequence of CPU codec/reconstruction probes instead of producing the requested
comprehensive derivation. That was the wrong balance. The negative results are
retained below, but **no further fit or shader experiment is authorised merely
because it is another four-read approximation**.

## 2. Non-negotiable target

The required representation must satisfy all of the following together:

1. O(1) fragment work: no ray march, loop, traversal, variable candidate tail,
   or per-species loop.
2. No actual grass mesh rendering. Offline geometry may be baked into textures
   or other fixed records.
3. Correct viewing from top-down, low oblique, grazing, horizontal, vertical,
   and camera-inside positions. No angle may be hidden or flattened away.
4. One coherent first event: hit/miss, depth, normal, authored colour, and
   material/mark must come from the same interaction.
5. Stable line anchoring: moving the camera along one ray without crossing the
   selected surface must not move that surface or create camera-centred bands.
6. Multi-species grass, flowers, moss, litter, and overlap are composed offline
   as one marked geometric union. They do not become runtime layers or queries.
7. Little performance impact, no memory blow-up, and no resurrection of the
   raised terrain copy.
8. The current isolated `Calamagrostis canescens` view stays active until its
   geometry, projection, colour, scale, and silhouettes are correct.

## 3. What the primary source actually proves

Sannikov's 2019 article proves an exact result for a much narrower geometry
class than the accepted plant mesh. The basic bake raycasts an infinitely
repeating two-dimensional mask and stores path length and normal in
`(phaseX, phaseY, angle)`. Extruding that mask along one common direction gives
a three-dimensional prism field. For that extrusion symmetry, the omitted
view dimension is reconstructed analytically:

\[
 |OB|=\frac{|OA|}{\cos\alpha}.
\]

This is exact because all geometry shares the same extrusion direction. An
affine, possibly oblique TBN change of basis preserves lines and therefore
preserves the proof.

The article explicitly says that non-parallel fibres, variable cross-sections,
and changing fibre direction are approximations and can look unnatural at some
angles. Sannikov's later public comments add useful implementation facts - a
centred signed depth range, few directions, and a PCF-like depth interpolation
- but do not publish the categorical winner/correspondence rule. Those comments
cannot be promoted into an algorithm that is not actually specified.

The important boundary is therefore:

- one common extrusion or affine image of it has an analytic dimension
  reduction;
- the accepted branched, tapered, fluffy Calamagrostis mesh and future moss do
  not have that global symmetry;
- adding a few elevation views does not by itself recreate the missing
  first-event correspondence.

Sources and claim boundaries are indexed in:

- `docs/deep-research/grass/SANNIKOV-ALGORITHM-AND-RUNTIME-MISMATCH-AUDIT.md`
- `docs/deep-research/grass/SANNIKOV-LATER-FEW-DIRECTION-METHOD-CHOICE.md`
- `docs/deep-research/grass/SANNIKOV-2024-YOUTUBE-COMMENTS.md`
- `docs/deep-research/grass/FIXED-READ-NOVEL-VIEW-PRIMARY-SOURCE-NEGATIVE-BOUNDARY.md`
- `docs/deep-research/grass/GROUNDCOVER-LINE-FIELD-SOURCE-AND-EXPERIMENT-LEDGER.md`

## 4. The mathematical object that is actually required

For a bounded periodic ground-cover community, an exterior oriented line has
four degrees of freedom. A convenient chart is two periodic phase coordinates
plus two direction coordinates. An arbitrary origin inside the botanical
volume adds the position along that line, because the query must return the
**next** interaction rather than the exterior first interaction. The generic
successor field is therefore five-dimensional:

\[
 F(o,d)=\operatorname*{first}_{t\geq0}\{o+td\in\mathcal G\},
 \qquad o=(x\bmod\Lambda,y,z\bmod\Lambda),\ d\in S^2.
\]

An exterior boundary chart can remove the along-line origin coordinate only
while the camera is outside and before any earlier event has been crossed. It
does not solve camera-inside visibility. Exact horizontal rays also have no
finite intersection with a horizontal reference plane, so a top/middle-plane
parameterisation needs a separate nonsingular chart rather than an epsilon
division.

The first-hit function is not an ordinary smooth scalar field. It is a
categorical visibility partition. Within one visible surface chart, depth is a
smooth or rational function of the ray. At silhouettes, disocclusions, and
occlusion-order changes, the owner changes discontinuously. Consequently:

\[
 \text{interpolate/elect depth without owner correspondence}
 \;\not\Rightarrow\;
 \text{interpolate/elect a surface}.
\]

This is the common reason the rejected methods below produced stretched sheets
or blurred plants.

## 5. Results that are genuinely retained

### 5.1 Stable reference coordinates are necessary

A ray must be addressed at a fixed botanical reference, not at a
camera-dependent orthogonal carrier. Camera-centred carriers create radial
sectors because their phase datum moves with the camera. A centred depth-zero
plane is useful for numerical range, but it is only a datum choice; it does not
create missing angular information.

### 5.2 Terrain entry must use the actual terrain graph

The older visually better checkpoint intersected the translated terrain graph
`g(x,z)+H`. A replacement that used one tangent at the ordinary terrain hit is
identical only on affine terrain. On curved terrain its top-entry height error
has the leading term

\[
 -\tfrac12\Delta x^T\operatorname{Hess}(g)\Delta x,
\]

which grows at low oblique angles. This explains one terrain-coupled component
of the old error. It does **not** justify a raised raster shell; the correct
query must reconstruct the actual graph intersection mathematically or use an
equivalent packed terrain root.

### 5.3 Exact hit-height carry is not event correspondence

For a live ray `d`, a canonical record with hit height `y_i` can be placed at
that height on the live ray exactly:

\[
 t_i=\frac{H-y_i}{-d_y}.
\]

But equality of height does not prove that the resulting point lies on the
canonical owner triangle, that it lies on the live owner, or that it is first.
This exact identity is retained only as coordinate algebra.

### 5.4 Attributes must stay coupled

Coverage, depth, normal, colour, and material mark must be baked/elected as one
event. Averaging unrelated first depths and then independently sampling colour
is invalid. Premultiplied colour is correct for a declared filtered ray
distribution, but its mean depth is not automatically a visible surface.

### 5.5 Species and moss belong in the offline truth union

For fixed ecological communities,

\[
 \mathcal G=\mathcal G_{grass,1}\cup\cdots\cup
 \mathcal G_{flower}\cup\mathcal G_{moss}\cup\mathcal G_{litter}
\]

is traced before reduction. This provides correct mutual occlusion with one
runtime query. Arbitrary live-changing Boolean mixtures would require another
state coordinate or another asset and are not silently included.

## 6. Rejected reconstruction families and why they were tried

### 6.1 Smooth four-view depth/colour interpolation

**Why tried:** it was the cheapest continuation of the existing atlas and
seemed consistent with Sannikov's few-direction comments.

**Why rejected:** the four neighbouring rays generally hit unrelated surface
events. Smooth blending suppresses hard angular cones but creates depths and
normals belonging to no surface. The live result was the observed flat,
stretched, wrong-perspective grass.

### 6.2 Camera-centred categorical selection

**Why tried:** selecting one complete record avoids numerically mixing owners
and keeps a fixed four-read cost.

**Why rejected:** the moving carrier violates line-origin invariance. The
selected angular regions become camera-centred radial cones, and the chosen
canonical event still need not cover the live ray.

### 6.3 Scalar PCF-like depth quantiles

**Why tried:** later Sannikov comments explicitly mention PCF-like depth
interpolation, and a four-input compare/select network is cheap and loop-free.

**Why rejected:** quartile/median selection changes which scalar wins but cannot
repair the support or owner correspondence of the four inputs. The measured
coverage support and metre-scale world-position errors remain wrong. This is
an LAAS interpretation of the public hint, not Sannikov's unpublished rule.

### 6.4 Plane, tangent, Jacobian, and projective inverse transport

**Why tried:** once a canonical event stays on one smooth surface, its tangent
plane gives a cheap closed-form live-ray intersection. This is the strongest
plausible analytic correction and avoids brute-force marching.

**Why rejected:** the premise fails across ordinary angular cells. The target
live surface is rarely the canonical first owner, and local self-consistency
does not prove firstness. On the actual source, the corrected K=4 projective
inverse returned the exact first event on only `3.722%` of held-out hits;
K=8 target visibility was high but exact recall after the update was only
`0.344%`. Further iterations or candidates would be the prohibited brute-force
growth of the same family.

Files:

- `GRASS-PROJECTIVE-INVERSE-CORRESPONDENCE-GATE.md`
- `GRASS-CHEAP-PROJECTIVE-RAY-MATH.md`
- `GRASS-RAY-RECONSTRUCTION-MATH.md`

### 6.5 Support-bounded canonical planes

**Why tried:** clipping a transported tangent plane to its actual source
triangle support should remove the infinite widening sheets while retaining the
exact plane intersection.

**Why rejected:** it removes the false sheets by rejecting essentially all
useful interpolation. Median certified support is `0.1283 mm`, while median
held-out candidate displacement is `0.3246 m` (`2309x` larger). Four-read and
even sixteen-read variants return zero held-out hits. Loosening support simply
restores the infinite-plane artifact.

File: `GRASS-SUPPORT-BOUNDED-CANONICAL-PLANE-GATE.md`.

### 6.6 Exact four-view hit-height reconstruction

**Why tried:** it uses the correct fixed middle-plane address and the exact
height carry above, costs only four records plus a winner-colour read, and
avoids the removed shell.

**Why rejected:** on held-out half-bin views the true live owner is present
among the four records for only `0.3279%` of true hits. Height blending gives
`0.63864` IoU and `8.466 m` depth p95. Even a truth-assisted oracle over those
same four candidates gives `5.993 m` p95 and only `0.337%` attached-triangle
agreement. Horizontal and camera-inside queries fail independently.

File: `GRASS-FOUR-VIEW-HIT-HEIGHT-GATE.md`.

### 6.7 Frame-global pinhole reconstruction

**Why tried:** selecting one canonical field for the whole camera removes
per-pixel direction cones, and a pinhole map from one event sheet to the frame
is algebraically exact for that sheet.

**Why rejected:** the canonical sheet is not the held-out camera's actual first
surface. Its sheet error is multiplied by camera distance, directly producing
the observed widening radial streaks. The CPU reconstruction reproduced the
same artifact family without any shader or terrain pipeline.

File: `GRASS-FRAME-GLOBAL-PINHOLE-LIGHT-FIELD-GATE.md`.

### 6.8 Direct nearest direction lattices and packed light fields

**Why tried:** storing the complete event per sampled ray avoids inventing
mixed events. Nearest selection also cannot geometrically stretch a record.

**Why rejected:** the full directional lattice required for the accepted
spatial resolution exceeds the resident cap, while capped direction counts
alias and pop. One-read 32-bit light-field allocations lose too much spatial
or angular fidelity and still omit generic camera-inside successors.

Files:

- `GRASS-DIRECTION-LATTICE-FEASIBILITY.md`
- `GRASS-PACKED-LIGHT-FIELD-FALLBACK-GATE.md`
- `GRASS-PLENOPTIC-SAMPLING-CASCADE-GATE.md`

### 6.9 Direct premultiplied radiance fields

**Why tried:** radiance/coverage interpolation cannot create a false geometric
plane because it does not relift blended depth. It was a clean test of whether
the remaining problem could be treated as a light-field image reconstruction.

**Why rejected:** it exchanges stretching for broad wrong-view cross-fades.
The strongest capped lattice has silhouette IoU `0.852`, RGB max-channel p95
`0.547`, and `44.0%` fractional coverage. Added `5 degree` and `1 degree` rows
retain panicles but do not make the field crisp. Separate depth elections still
have metre-scale errors.

File: `GRASS-DIRECT-RADIANCE-FIELD-GATE.md`.

### 6.10 Fixed-K event lists and event-sheet moments

**Why tried:** a small deep record might retain several occlusion modes while
remaining fixed cost, and band-limited moments are mathematically legal to
prefilter.

**Why rejected:** arbitrary repeated botanical lines do not have a small
uniform successor bound. Collapsing multimodal first-hit distributions to a
mean or three sheets places events in empty space and produces broad depth and
normal tails. Increasing K scales reads/output work and becomes the candidate
enumeration the task forbids.

Files:

- `GRASS-FIXED-K-EVENT-RECORD-FEASIBILITY.md`
- `GRASS-COUPLED-BANDLIMITED-EVENT-SHEET-FEASIBILITY.md`
- `GRASS-ANGULAR-EVENT-FLOW-CEILING.md`

### 6.11 Affine slabs, swept fibres, and procedural botanical charts

**Why tried:** Sannikov's exact theorem applies to extrusions, so decomposing
the plant into a tiny union of line-preserving botanical families could extend
the exact method without a full five-dimensional field.

**Why rejected:** the accepted plant cannot be covered by a tiny fixed union
without filling empty intervals, dropping taper/branch support, or introducing
an unbounded successor/candidate tail. General non-affine taper does not
preserve rays. Procedural chart identifiers also do not solve first-visible
chart election.

Files:

- `GRASS-FOUR-AFFINE-SLAB-EXTRUSION-FEASIBILITY.md`
- `GRASS-FINITE-SWEPT-FIBRE-FEASIBILITY.md`
- `GRASS-PROCEDURAL-CHART-FEASIBILITY.md`

### 6.12 Cell-exit and endpoint-conditioned transfer factorisations

**Why tried:** a periodic cell has a finite exit, so factoring one-cell transfer
from repeated continuation could avoid a 155 m direct field and naturally
express camera-inside suffixes.

**What is retained:** the associative cell-transfer identity and half-open
endpoint/censoring rules are exact.

**Why the proposed codecs were rejected:** a finite first-cell record does not
close the unknown continuation without another exterior successor field. The
tested affine head cannot represent endpoint-dependent extinction; the grouped
rank-eight model and even a more favourable rank-eight SVD oracle fail badly.
The latter retains only `59.99%` of coverage energy and produces a dark blurred
sheet with missing panicles.

Files:

- `GRASS-CELL-EXIT-POINTED-LINE-FACTORIZATION.md`
- `GRASS-ENDPOINT-CONDITIONED-GDM-TRANSFER.md`
- `GRASS-HYBRID-DEEP-TRANSFER-FACTORIZATION.md`

### 6.13 Pairwise-plane and other low-rank learned fields

**Why tried:** factorised fields are the obvious way to encode the complete
five-dimensional pointed-ray function within four reads and the memory cap.
They also offered a fixed dense decoder with no runtime traversal.

**Why rejected:** the actual visibility partition has sharp coupled
discontinuities that these smooth low-rank products do not capture. The
all-pairs rank-four model fails even training records and produces broad
wrong-view sheets. Prior one-volume origin-aware decoding also failed.

The final stopped experiment used four RGBA16F trilinear factors over the five
pointed-ray variables:

\[
 (x,z,y),\quad(x,\phi,\theta),\quad(z,\phi,\theta),\quad(y,\phi,\theta),
\]

followed by a fixed `19 -> 12 -> 2` ReLU decoder. This covering includes every
pairwise variable interaction, uses four logical reads, `252` FMA-equivalent
operations, and `8,389,140` resident bytes at resolution 64. It was attempted
because it was a materially stronger full-domain factorisation than the failed
single spatial volume, not because CPU fitting itself was the goal.

It decisively fails held-out actual-mesh rays: IoU `0.72854`, hit-depth p95
`14.376 m`, top-down `75 degree` IoU `0.10487`, and QA IoU `0.55489`. Training
loss continues falling while held-out topology fails, so a resolution-96 retry
was not run. The problem is not merely table size; smooth feature interpolation
does not supply categorical visibility correspondence.

Artifact:
`data/work/groundcover-origin-aware-tetrafield/0d2f141e28deb881/9cf21b2582581670/r64-e60/report.json`
with SHA-256
`169eb52f2a2b6af7c87924e930b53ed9de9d22cd579985fb142e1edd2f8246b6`.

Related file: `GRASS-PAIRWISE-PLANE-TRANSFER-GATE.md`.

### 6.14 Owner-conditioned codebooks

**Why tried:** if exact event records repeated heavily, a two-read dictionary
could preserve categories without interpolation.

**Why rejected as the central answer:** exact hit-record reuse is only `1.81%`;
most savings come from collapsing misses. The dictionary does not add the
missing all-direction field, and its dependent second read has poor locality.
It remains a possible final packing pass only after a correct field exists.

## 7. Why the CPU work did not amount to the requested solution

The gates established useful impossibility boundaries, but they did not derive
the missing categorical visibility operator. The recurring pattern was:

1. start from four or a few unrelated sampled rays;
2. transport, blend, elect, regress, or factorise their scalar records;
3. hope that the result remains the live first surface.

Actual-source tests repeatedly disproved step 3. Running more fits within that
pattern was no longer mathematical progress. It was a search over codecs whose
most important invariant - first-visible surface correspondence - had not been
constructed.

The correct stopping conclusion is not that O(1) is impossible. It is that the
work so far has **not** found the additional structure that makes the generic
five-dimensional categorical successor cheap. Sannikov supplies that structure
for one common extrusion family. The accepted arbitrary plant/moss target needs
either another exact structure or an explicitly bounded categorical
point-location representation; it cannot be obtained by renaming interpolation.

## 8. The only legitimate next mathematical questions

No shader implementation or new CPU fit should begin until one of these is
answered in equations with a frozen resource bound:

1. **Exact geometry class:** Can the authored ground-cover community be built
   as a line-preserving transformation or a genuinely tiny fixed union of
   extrusion families while retaining the accepted Calamagrostis and moss
   silhouettes? If yes, prove that live rays remain rays and prove the fixed
   first-event composition rule.
2. **Categorical point location:** Can the full pointed-ray visibility
   partition be encoded in a fixed-depth, fixed-read structure that first
   selects one surface chart and then evaluates that chart's exact/rational ray
   intersection? The proof must include memory, worst-case depth, boundary
   filtering, horizontal/vertical charts, and camera-inside successor order.
3. **Missing author rule:** Is Sannikov's later PCF-like categorical rule
   available in code, captures, or a primary explanation? If found, transcribe
   the complete record, comparison, and attribute-coupling semantics before
   claiming transfer to arbitrary meshes.

A proposal that merely increases directions, rank, network width, layers,
candidates, or iterations does not answer any of these questions.

## 9. Resume gate

The mathematics track resumes only when there is a written construction that:

- identifies the exact ray-space invariance or categorical partition being
  represented;
- proves why one selected record is the live first interaction rather than a
  plausible depth;
- covers exterior, horizontal, vertical, and inside-origin queries;
- keeps overlapping species/moss in one offline union;
- freezes reads, bytes, ALU, and branch structure before evaluation;
- contains no loop, march, traversal, raised shell, or hidden view restriction.

Only then may one bounded exact-source CPU gate check the derivation. Only a
passing gate may lead back to the shader and mandatory real-WebGPU boot.

