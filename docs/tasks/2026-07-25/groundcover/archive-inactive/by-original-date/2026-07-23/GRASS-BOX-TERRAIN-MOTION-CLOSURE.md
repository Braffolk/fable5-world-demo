# Terrain, motion, control, and invocation closure for the box-boundary field

**Date:** 2026-07-23
**Scope:** pure mathematics and product contract; no renderer or shader change
**Parent construction:** `GRASS-BOX-BOUNDARY-TRANSFER-MATH.md`

## 1. Binding verdict

The guarded common-slab plus boundary-transfer construction is exact for one
periodic community under one invertible affine map.  A curved heightfield,
root-by-root height variation, a spatially varying wind field, or a control
field which changes the set of plants from root to root is **not** one affine
map.  Calling any of those cases "exact by conjugation" would be false.

The practical product construction is therefore:

1. retain exact carrier entry and exact boundary transfer in a local affine
   ground chart;
2. make every resolved record name its categorical source triangle and root;
3. sample the real packed terrain and smooth deformation controls at that root,
   transform that one plane, and reintersect it with the live ray exactly;
4. certify first-event stability with a cook-computed displacement/order
   margin; and
5. use a physically filtered mixed record only when the remaining ambiguity is
   smaller than the live pixel footprint.  A resolved exterior ambiguity is a
   failed cook/gate, never a hidden fade, clamp, or fabricated surface.

Thus the arbitrary triangle soup is still a general cook input.  The finite
low-memory output is conditional only on measurable terrain/deformation
bandwidth and the ordinary physically filtered quality gate.  It is not
conditional on the soup belonging to a special botanical geometry class.

## 2. Rooted world map

Let `g(p)` be the packed terrain height at horizontal root position
`p in R^2`.  Tall plants are world-up by default; a source point `(xi,h)` in
motion lane `l` maps to

\[
 X_{p,l}(\xi,h,t)=
 \begin{bmatrix}p\\g(p)\end{bmatrix}
 +
 \begin{bmatrix}
   R_l & b_l(p,t)\\
   0   & a_l(p,t)
 \end{bmatrix}
 \begin{bmatrix}\xi\\h\end{bmatrix}.
\]

`R_l` contains the lane-global golden-angle/D4 placement transform,
`b_l h` is root-fixing affine wind shear, and `a_l` is height/vigor scale.
An authored per-root/species stiffness `k_j` may multiply `b_l` for every
primitive of owner `j`; it is carried by the categorical winner and included
in the all-event displacement certificate. It does not create a per-species
query. Non-affine within-blade flutter remains outside this tier.
The carrier guard is baked for the complete declared ranges of `a_l` and
`b_l`, so every permitted pose remains inside its fade box.

For constant `b_l` and `a_l`, this is an affine conjugation after terrain is
made affine.  It is exact: inverse-transform the ray, query the unchanged
field, transform the winning plane and normal forward.  Roots remain fixed
because the shear displacement is zero at `h=0`.

Normal-aligning every plant to a varying terrain normal is allowed only as the
bounded extension in Section 4.  The binding inexpensive grass path is
world-up plus exact root height.  Moss hummock form belongs in packed terrain
microtopography, so this choice does not flatten moss landform.

## 3. Local ground chart and exact winner correction

At a ray anchor `a`, write

\[
 g(a+u)=g(a)+m_a\cdot u+e_a(u),\qquad m_a=\nabla g(a).
\]

The affine chart

\[
 C_a(u,h)=(a+u,\ g(a)+m_a\cdot u+h)
\]

maps a world ray to a straight canonical ray, so the common-slab entry and the
four-dimensional boundary transfer are used without approximation *inside
that chart*.  The binding terrain quantity is the directly certifiable chart
remainder

\[
 E_T(a,L)=\sup_{\|u\|\le L}
 |g(a+u)-g(a)-m_a\cdot u|.
\]

For a `C2` patch with packed-height/interpolation error `epsilon_g` and
curvature bound `||Hess g|| <= kappa`, the useful sufficient bound is

\[
 \boxed{|e_a(u)|\le E_T(L)=\epsilon_g+\tfrac12\kappa L^2.}
\]

Every regular transfer record stores the winner's root offset/copy token,
source plane `(n,c)`, triangle-domain chart, and coupled attribute chart.
After election, evaluate `g(p_w)` and the lane
controls at the real winner root `p_w`, transform the plane by the rooted map
of Section 2, and intersect that transformed plane with the original world
ray.  The live barycentric/domain test and attributes use that same transformed
triangle chart.  This final position is exact for the elected source triangle
(up to the packed terrain/control quantisation); no terrain-plane depth is
retained and no attribute from another owner is introduced.

This correction does not by itself prove that the same triangle would have
won after all roots were moved.  That is the purpose of the stability
certificate below.

## 4. Stability certificate and honest error law

For a translated world-up triangle `j`, a vertical root error `delta y_j`
changes its ray parameter to first order by

\[
 |\delta t_j|\le
 \lambda_j |\delta y_j|,
 \qquad
 \lambda_j={|n_{j,y}|\over |n_j\cdot d|}.
\]

Parallel/incidence-boundary cases have unbounded `lambda`; they are mixed
cells, not regular cells.  If plants follow terrain normal, add the conservative
point displacement `H kappa L` for rotation over maximum botanical height
`H`.  Smooth root-wise deformation contributes

\[
 E_D(L)\le H\bigl(G_a+G_b\bigr)L,
 \quad
 G_a=\|\nabla a_l\|_\infty,
 \quad
 G_b=\|\nabla b_l\|_∞.
\]

Use

\[
 E_X(L)=E_T(L)+E_D(L)
\]

for world-up plants, or

\[
 E_X^{normal}(L)=E_X(L)+H\kappa L
\]

for normal-aligned plants.  The baker has the complete exact event sequence
offline.  For canonical winner `1`, define the robust order margin against
**every** later event

\[
 M=\min_{j>1}\left[(t_j-t_1)
 -\lambda_1E_X(L_1)-\lambda_jE_X(L_j)\right].
\]

The categorical winner is order-stable when `M>0`.  Checking only the second
canonical event is insufficient because a later nearly parallel surface may
have a much larger `lambda_j`.  This all-event certification is cook work; it
does not store or traverse a live candidate list.  A second regularity test
requires the transformed hit to retain positive barycentric/edge margin after
the same maximum point displacement.  Cells which can leave the winner
triangle through an edge are mixed even when plane-depth order alone is
stable.

The record also stores/certifies its horizontal interaction reach `L`: the
largest root/point displacement able to affect firstness in that cell before
the finite horizon.  This is a maximum for a crisp proof, not a p95.  Cells
passing the inequality retain one real owner and receive the exact winner-root
correction of Section 3.

Cells failing it are subdivided to the maximum physically useful resolution.
At the last level they may become a filtered record only if their complete
projected ambiguity band fits the declared pixel kernel.  For hit distance
`D` and angular pixel pitch `theta_pix`, let

\[
 w_{pix}(D)=2D\tan(\theta_{pix}/2)\simeq D\theta_{pix}.
\]

The mixed record is admissible only when both its oracle-measured colour error
passes the frozen threshold and its conservative world band `W_cell` obeys

\[
 \boxed{W_{cell}\le k_{pix}w_{pix}(D).}
\]

This is the legitimate distance relaxation: a fixed centimetre-scale
uncertainty occupies fewer pixels as `1/D`.  Nothing is excused merely for
being far away.  In particular, if `L` itself grows so quickly that
`E_T(L)` grows faster than the footprint, the cell stays red.  A near-field
red cell cannot be converted to a fade unless the camera is actually inside
the declared guarded carrier box.

## 5. Binding terrain and anchor contract

The production carrier uses the actual packed terrain field and its measured
per-grade `E_T(a,L)` envelope; `(epsilon_g,kappa)` is the compact smooth-patch
certificate, not an assumption that a piecewise packed field has a classical
Hessian across every cell edge.  Cover is disabled by the cook on cliffs or
discontinuities for which no accepted chart bound exists; this is an
ecological placement decision, not a runtime correctness filter.

Ground charts and their guarded carriers are fixed cooked world regions, not
camera-dependent boxes.  Roots assigned to chart `c` use its fixed
`(a_c,m_c)` and a slab/footprint guard enlarged for the certified terrain,
wind, and height displacement.  The camera-box fade test is therefore against
a stable physical guarded region containing those plants.  A live anchor only
selects a cooked chart; it never moves the carrier with the camera.

The whole-forward transfer may use one chart only while its certified event
reach stays inside that chart's eroded interior.  A ray whose possible first
event crosses a chart edge is a transition cell.  It needs a jointly cooked
finite chart-transition state or may become filtered only after its full
ambiguity band is sub-pixel.  This is the same closure rule as ecological
community boundaries in Section 7.  It prevents a local tangent plane from
silently being extrapolated across a hill, ridge, or arbitrary number of
terrain patches.

Each full-screen ray selects one cooked affine chart in fixed work:

1. when a real scene/terrain endpoint exists inside the cover horizon, its
   horizontal coordinate is the preferred anchor;
2. otherwise use the packed ground sample under the camera and one analytic
   intersection with that chart's guarded slab; one packed sample at the
   predicted horizontal coordinate supplies the final cooked chart id;
3. exact horizontal/tangent rays use the camera chart directly and are judged
   by the same finite `L`, curvature, and pixel-footprint certificates.  No
   epsilon tilt or hidden angle clamp is permitted.

This is at most a fixed predictor and one correction sample, never a terrain
march.  Scene depth is only the opaque cutoff `t_scene`; it is not a required
query origin and must not suppress a sky pixel which can see a cover box.

The chart choice itself is included in the offline gate: translation sequences
must not cause unforced anchor/chart changes, and all sky/no-depth, exact
horizontal, uphill, downhill, ridge, and slope-tangent views are explicit
cases.  Failure means the portable local-chart realization is insufficient
for that terrain grade; the honest fallback is a terrain-specific cooked
transfer/correction asset or no high-detail cover on that grade, not more live
steps.

## 6. Wind and smooth height modulation

The exact wind tier before categorical election is lane-global affine shear:

\[
 b_l(p,t)=b_l(t),\qquad a_l(p,t)=a_l(t).
\]

Two Tier-1 populations may have different phases `b_0(t),b_1(t)` and are
queried independently. This gives coherent lean with roots fixed and has no
pose-atlas memory multiplier. Per-species/root affine stiffness is applied to
the finally elected plane and remains firstness-safe only where Section 4's
all-event certificate covers its complete range. Per-blade flutter and
nonlinear bending remain out of the geometry contract; high-frequency shimmer
may affect shading only.

The requested smooth world-keyed `+-15%` height/vigor field is retained as a
root-wise affine scale.  It is exact for the finally elected plane because
`a_l(p_w,t)` is sampled at its real root.  Its possible effect on *which*
triangle wins is bounded by `H G_a L` and certified by Section 4.  The same
rule permits a low-frequency gust field through `H G_b L`; it is bounded, not
called exact.  Choose wavelengths/amplitudes cook-side so all resolved
near-field cells pass.  No extra motion lane is required for these smooth
fields.

Tint which does not change opacity/geometry is simply evaluated at the winner
root and is exact shading data.  A vigor value which removes plants, changes
coverage, or switches species is not a tint: it changes firstness and must be
compiled into an authored community state as Section 7 requires.

## 7. Community control, species, moss, and true lane count

The whole-forward transfer sees the complete future community.  Therefore an
arbitrary per-root control toggle cannot be applied after the lookup: removing
the stored winner may expose an unbounded successor.  The binding control
contract is:

- a **community state** is a jointly baked soup containing every overlapping
  species in that state; species/material remains a categorical winner mark;
- the cooked control field selects one complete state before the query;
- that state must be stable over the record's certified interaction reach
  `L`; ecological transition bands are either jointly baked finite transition
  states or physically filtered only when their measured band is sub-pixel;
- D4/root variants which change geometry obey the same rule.  They are baked
  into the community/supertile pattern or selected over patches wider than the
  certified reach.  Cross-owner depth/normal/mark blending is never used.

Many species therefore cost no extra query when they share a motion law and
are union-baked in one community.  Independent **motion laws**, not species,
set the lane count:

1. two golden-angle Tier-1 above-ground populations with their two wind phases:
   **two dynamic lanes**;
2. moss hummock shape in terrain and moss colour in the terrain material:
   **zero additional transfer lanes**;
3. if resolved static moss shoots/capitula or another genuinely static
   protruding cover must occlude the dynamic plants: **one static lane**.

Thus the binding baseline is two lanes.  The highest-fidelity grass-plus-static-
moss configuration is three.  Grass, sedge, reed, forbs, flowers, and multiple
species may coexist inside the two dynamic soups without per-species work,
provided each owner's affine stiffness lies inside the certified lane range.
A genuinely non-affine or independently phased motion law requires another
lane; it cannot be smuggled into a mark. Elect the
canonical winner across lanes first, use the robust cross-lane margin, and
then fetch real terrain/control only at the final winning root; losers do not
need root reads. Including control, the usually shared anchor sample, and that
one dependent winner-root sample, the macrobrick codec's honest two-lane
envelope is approximately `10--14` reads on the packed common path and can
reach `12--16` on a sky/no-depth ray if its anchor predictor cannot be packed
with existing terrain data. The production format is green only when its
actual worst schedule is `<=14`; this is a packing gate, not permission to
omit the sky case. A third geometric static lane is approximately `13--18`,
which is why moss-in-terrain/material is the binding low-end configuration
rather than pretending the third lane is free.

## 8. Invocation coverage without grass geometry

The visibility query is invoked once for every output pixel in an existing
full-screen resolve/fragment path.  It reconstructs the camera ray, performs
the fixed carrier/transfer queries, compares the returned world depth with the
real scene cutoff, and composites the nearer result.  This is `O(screen area)`
and `O(1)` per pixel, independent of grass count.  It requires no grass mesh,
quad, billboard, shell, raised terrain copy, floating plane, indirect draw, or
grass-specific raster pass.

Full-screen invocation is load-bearing.  Reusing only ordinary terrain
fragments cannot cover blade silhouettes which project beyond the terrain;
requiring a same-pixel scene depth misses grass against sky or in front of a
distant object.  Conversely, a conservative early-out is legal only if a
cooked mask proves that the complete ray segment before `t_scene` contains no
carrier.  Scene miss by itself is not that proof.

The query may be scheduled in an already existing full-screen resolve rather
than as a new compute pass.  That scheduling is an implementation choice; the
mathematical coverage requirement is simply one invocation for every camera
ray which may intersect a declared carrier.

## 9. CPU gate additions before runtime work

The first common-slab/boundary-transfer CPU gate must additionally report:

1. measured packed-terrain `epsilon_g` and curvature envelopes by accepted
   terrain grade;
2. per-cell interaction reach `L`, incidence factors, first/second-event
   clearance, and the fraction passing the stability inequality;
3. unresolved projected band width in pixels over distance and every required
   view, including exact horizontal and sky/no-depth cases;
4. error from the fixed anchor rule under millimetre camera translations;
5. sweeps of the actual `+-15%` height field and maximum wind-gradient
   contract, not only constant affines;
6. community-state boundary cases and jointly baked species/moss occlusion;
7. fade volume introduced by guarded common slabs; and
8. the complete `2`-lane and optional `3`-lane read/byte budgets.

Green means resolved cells are categorical and stable, mixed cells are
genuinely sub-pixel under the declared kernel, and the total atlas/correction
working set remains within 250 MB.  If the first end-to-end run is red, one
premise audit may correct a bad anchor, bound, or unit.  A second red run parks
the model with the exact failing terrain grade / cell class; it does not add a
march, candidate list, hidden carrier, or per-species loop.

## 10. What is exact and what is bounded

| Component | Guarantee |
|---|---|
| guarded common-slab entry on affine ground | exact |
| 4D whole-community boundary transfer | exact continuous oracle |
| categorical winner plane/root reconstruction | exact after real root transform |
| constant per-lane wind/height affine | exact conjugation |
| two Tier-1 layers | exact fixed min/composition of two lane queries |
| multiple species sharing a lane | exact jointly baked occlusion |
| curved terrain firstness | cook-certified by `E_T`, reach, incidence, and order margin |
| smooth spatial wind/height | bounded by `H(G_a+G_b)L`, then certified |
| final visibility discontinuities | physically filtered only below the live pixel footprint |
| arbitrary per-root species/geometry toggles | unsupported; compile finite community states |
| moss hummock landform | exact packed terrain path; optional static shoots use one lane |
| screen coverage | exact ray-domain coverage through full-screen invocation; no grass geometry |

## 11. Finite ecological-patch side entry

### 11.1 The missing domain

The periodic botanical footprint fills its `0.52 m` source tile.  That is good
inside an active community, but it means the periodic carrier by itself is
`R^2 x I`: at eye height there is no side boundary.  The side boundary belongs
to the finite cooked ecological control patch.

Resolve the existing root-stable A/B field cook-side to a categorical
**community-state field**

\[
 c(x)\in\{0,1,\ldots,S\},
 \]

on its declared world lattice (`0` means no geometric cover).  "Root-stable"
means that the hash/tie decision is a deterministic function of the cooked
world cell, not of camera, frame, or screen position.  Geometry-affecting
fine-root toggles are part of the jointly baked community state; they cannot
be applied after a first-hit lookup.  Let

\[
 M=\{x:c(x)\ne0\}.
\]

Within one fixed affine terrain chart and one shared conservative above-ground
height interval `I=[h-,h+]`, the finite carrier is

\[
 U_M=M\times I.
\]

All dynamic above-ground states in this carrier group use the same `I`, padded
for their complete permitted height/wind range and the chart remainder.  A
short state merely contains empty air.  This common interval is load-bearing:
if every state has an unrelated vertical interval, rejecting a vertically
ineligible first patch and finding the next eligible patch is again a generic
successor query.  Moss landform/material stays in terrain; a genuinely separate
static height group pays a separate fixed carrier query.

### 11.2 Exact continuous factorisation on an affine chart

Transform the world ray into the fixed chart and clip it to `I`, giving the
forward interval `[t_a,t_b]`.  Put

\[
 s=\|d_{xz}\|,\qquad q_a=o_{xz}+t_a d_{xz}.
\]

For `s>0`, set `omega=d_xz/s` and define the pointed finite-mask first passage

\[
 \rho_M(q,\omega)=
 \inf\{\ell\ge0:q+\ell\omega\in M\}.
\]

Then

\[
 \boxed{t_M=t_a+\rho_M(q_a,\omega)/s}
\]

is the exact first side/cap entry, accepted only if `t_M<=t_b` and within the
finite world horizon.  If `q_a in M`, `rho_M=0`: an origin above/below the slab
enters through a cap; an origin already in the guarded physical carrier uses
the declared box fade.  A camera at eye height outside `M` receives the desired
positive side-entry distance.

For `s=0`, horizontal phase is constant.  Membership at `q_a` gives the
appropriate cap; non-membership is an exact miss.  There is no epsilon tilt or
division by a small horizontal speed.  The cooked mask uses half-open cell
ownership and a fixed edge/vertex total order.  An exact line on a patch edge
therefore has one categorical convention; a tangential touch which never
enters the owned half-open interior is a miss.  These pole and tie conventions
are shared by bake and runtime.

This is a complete `O(1)` *continuous oracle*.  It retains the outside origin's
two-dimensional phase; replacing it with an unpointed line field would repeat
the already-proved gap error.

### 11.3 Store a boundary owner, not sampled distance

For the cooked grid/polygon mask, let each directed boundary primitive `e`
have supporting line

\[
 a_e\cdot x=b_e
\]

and a finite span.  Partition `(q_x,q_z,omega)` by the identity of the first
entered primitive.  A regular table cell stores only the categorical primitive
token.  For the live direction, reconstruct

\[
 \boxed{\ell_e={b_e-a_e\cdot q\over a_e\cdot\omega}}
\]

and validate the token's span/sign convention.  Thus angular/spatial sampling
selects an owner but never transports a sampled distance to a different ray.
Within a regular visibility cell the entry is exact at every angle, including
horizontal world rays.  There is no `range * Delta-angle` stretch.

The token also identifies the cell immediately on the active side.  Its
already-packed community-state record supplies the jointly baked A/B state;
state ids, normals, depths, and botanical marks are never interpolated across
the patch boundary.

Cells in which the first boundary primitive changes are true 3D visibility-
complex boundaries.  Subdivide them to the declared physical footprint.  A
last-level mixed cell is legal only when exact pinhole subray integration shows
that the complete patch-edge band is sub-pixel and passes the premultiplied RGB
and coverage thresholds.  Otherwise it needs a direct correction token or the
cook is red.  Nearest-angle selection without this mixed-cell rule is
forbidden even though the live line reintersection is exact.

### 11.4 Direct streamed codec and byte law

Use a direct, non-traversing two-level layout:

1. a world-streamed page table over the existing control lattice;
2. pages only for cells in the finite-horizon dilation of a patch boundary;
3. an `R32Uint` regular token table over `(page-local x,z, base angle)`;
4. at most one fixed perfect-hash correction access for mixed base cells; and
5. the existing packed control read at the exact inside entry cell.

Page absence is an exact `MISS` only when the cook proves that the complete
forward horizon contains no active cell.  A scalar distance-to-patch field may
provide this proof, but an ordinary signed-distance sample cannot replace
directional first passage.

Let `N_band` be the number of resident control cells in the horizon dilation
of patch boundaries, `N_omega` the base angular slots, `b_0=4` bytes the regular
token, `C` the number of correction records, and `b_c` their complete key plus
payload bytes.  The charged resident cost is

\[
 \boxed{B_{side}=b_0N_{band}N_\omega+b_cC+B_{page}.}
\]

This is charged inside, not in addition to, the `250 MB` whole-cover ceiling:

\[
 B_{side}+B_{botanical}+B_{payload}+B_{control}+B_{terrain}+B_{work}
 \le 250\ \mathrm{MB}.
\]

At the current `512^2` camera window on the `2 m` ground-cover lattice, a
dense worst-case `R32Uint` table costs `16/32/64 MiB` for
`N_omega=16/32/64` before corrections.  This is only an accounting example,
not an authorization to pick 32 angles: token-change rate and filtered error
choose the resolution.  Sparse large patches reduce `N_band`; a checkerboard
mask makes it the whole window.  The packed country store is charged by the
same boundary-band pages, so a scheme which fits resident VRAM but creates an
unacceptable country-scale asset is also red.

All regular accesses are coherent.  Only the optional correction/token payload
is indirect.  The live arithmetic is one chart/slab clip, one token fetch, one
line intersection, one control fetch, and fixed accepts/selects.  There is no
walk, loop, per-patch candidate, or geometry.

### 11.5 Why arbitrary masks have no universal low-memory guarantee

The three-dimensional field `rho_M(q,omega)` is the necessary pointed state,
not an implementation accident.  An arbitrary `n x n` binary mask already
contains `n^2` independent bits.  More strongly, masks with alternating thin
components can make a different boundary segment first for every origin cell
and angular slot.  Their direct first-owner table has
`Omega(N_q N_omega)` independent categorical labels.  Polygonal masks with
holes have a first-segment visibility complex of quadratic worst-case
complexity in boundary count.  No fixed 250 MB representation can encode this
family for unbounded extent/complexity, and an SDF does not change the result:
finding a ray zero of an arbitrary SDF requires traversal/marching.

Therefore the factorisation works for every finite mask as a continuous oracle,
but **low memory is a measured compiler property, not an unconditional theorem
for arbitrary control masks**.  The necessary packed/authoring contract is:

- finite `R=155 m` (or the finally frozen cover horizon);
- categorical half-open control cells with geometry-affecting A/B decisions
  resolved to finite jointly baked community states;
- one shared conservative slab per queried motion/height group;
- fixed terrain charts whose remainder guard contains all assigned plants;
- a measured cap on boundary-dilation area, boundary primitives per horizon
  disk, and first-owner angular changes per page;
- last-level mixed bands demonstrably below the live pixel footprint; and
- `B_side` plus the complete botanical codec below the shared byte/read gates.

Large ecological polygons satisfy this naturally; adversarial salt-and-pepper
or sub-cell state toggles may not.  If the actual control field fails, the
permitted upstream fixes are to cook geometry-affecting variation into a
finite community/supertile, merge sub-resolution patch islands according to a
declared physical filter, or allocate a measured larger side-entry asset inside
the same ceiling.  Runtime marching and silent patch omission are not fixes.

### 11.6 Terrain curvature and chart boundaries

Within a fixed chart, enlarge the vertical guard by the certified
`E_T(a,L)` remainder from Section 3 and enlarge horizontal patch footprints for
the complete affine wind/height displacement.  The side-entry line remains a
straight 2D mask query; the winning botanical root/plane receives the exact
terrain correction and all-event order certificate already specified.

At a terrain-chart boundary, the first side token also carries the categorical
chart id.  The record is regular only if its complete carrier/botanical reach
stays within that chart's eroded interior.  Otherwise it is a jointly cooked
chart-transition record or a last-level filtered cell.  If different charts
cannot share one conservative slab without an unacceptable false-fade volume,
they are different carrier groups and consume independent fixed queries; the
compiler may not reject one vertically ineligible entry and search successors
live.

### 11.7 Entry is solved; finite-patch botanical succession needs one more certificate

`rho_M` solves first carrier entry, not by itself the clipping of an infinite
periodic botanical transfer to a finite patch.  At the returned side boundary,
use an edge-conditioned community transfer (straight edge/corner and ordered
state pair are finite cook categories).  A regular record is accepted only
when its elected botanical hit occurs before the exact forward patch exit.

If the first patch interval contains no botanical hit but a later disjoint
patch can contain one, finding that later hit is a successor problem.  For an
arbitrary mask and arbitrary soup, eliminating it requires the combined field

\[
 \Psi(q,d)=\text{first botanical event over the complete finite control field},
\]

which is four-dimensional at the side boundary/world phase and is not the
cheap 3D side-entry table.  The low-memory construction therefore requires a
**first-entry closure certificate** per regular cell: either the edge-
conditioned transfer hits before forward patch exit, or the cook proves no
later patch can contribute before the horizon.  Thin/tangent intervals may
terminate as filtered only when their complete contribution is sub-pixel.

This certificate is essential for the claim of general final visibility.  If
it fails at resolved scale, the honest options are a world-specific combined
`Psi` asset which passes its own rate gate, a bounded extra carrier group/query
explicitly approved by cost, or parking the finite-patch model.  Repeating
`rho_M`/`T` until something hits would be the forbidden march under another
name.

### 11.8 Additional CPU gate outputs

Before runtime transcription, the finite-patch gate must emit:

1. exact side-entry agreement for random and adversarial outside origins,
   exact horizontal/vertical/tangent rays, vertices, finite horizon, and camera
   translations;
2. regular/mixed/correction fractions and first-boundary token-change counts;
3. `N_band`, page occupancy, correction count, resident bytes, and packed
   country bytes for `16/32/64` base-angle ablations;
4. live-reintersected entry error and filtered patch-edge RGB/coverage error;
5. false-fade volume from the shared terrain/deformation slab;
6. edge-conditioned hit-before-exit and first-entry-closure pass rates; and
7. the final combined read/byte ledger with the botanical macrobrick codec.

The side-entry method is green only if the actual Estonia control field—not a
filled periodic tile—passes those measurements.
