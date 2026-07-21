# Precomputed-ray ground-cover reconstruction math

Status: math-only reconstruction contract, 2026-07-21. This note deliberately
contains no Three.js, TSL, WGSL, render-pass, or binding decisions. The executable
reference is `tools/groundcover-bake/GroundCoverRayMath.ts`, with finite tests in
`GroundCoverRayMath.test.ts`.

## 1. Scope and source interpretation

The relevant source figures are only `img-000` through `img-003` extracted from
Sannikov's archived GameDev.ru PDF:

1. a periodic two-dimensional mask;
2. its parallel three-dimensional extrusion;
3. the two-dimensional precomputed ray from `O` to `A`;
4. the lifted three-dimensional view ray from `O` to `B`.

The blue plane in the fourth figure is the outer/top envelope. Sannikov's `O` is
the view ray's intersection with that envelope. It is not a later intersection
with the underlying ground. This note names a later ground hit `T` to prevent the
two points from being conflated again.

The source identity

\[
|OB|=\frac{|OA|}{\cos\alpha}
\]

says that `OA` is the in-plane projection of the same segment `OB`. In a skewed
or scaled carrier this must be evaluated in profile coordinates. A world-space
horizontal cosine is only the orthonormal special case.

The linked Tenth Planet follow-up supplies the second important distinction:
tangent vectors and normal covectors transform differently when the basis is
not orthogonal. For ray coordinates we use the inverse of the full derivative
basis. For normals we use its inverse transpose.

## 2. Exact affine theorem

Let profile coordinates be

\[
q=(u,h,v)^T
\]

and let one fixed carrier chart map them to world space by

\[
x=b+Jq, \qquad J=\frac{\partial x}{\partial q}.
\]

The columns of `J` are the actual, unnormalised partial derivatives. They may be
scaled, skewed, and non-orthogonal. `J` must be invertible. Define

\[
K=J^{-1}.
\]

For a world ray

\[
r(t)=c+t d,
\]

the exact profile-space ray is

\[
q(t)=K(r(t)-b), \qquad \dot q=Kd.
\]

Because `J` is constant inside an affine chart, `q(t)` is a straight ray. Define

\[
s=\|\dot q\|, \qquad n=\frac{\dot q}{s}.
\]

If the precompute for direction `n` returns a full three-dimensional profile
distance `tau`, the world-ray parameter of the hit is

\[
t_B=t_O+\frac{\tau}{s}.
\]

This is the complete inverse for a full 3D profile bake. It is valid for a
normalised or non-normalised world ray; the units of `t` simply follow `d`.

### 2.1 Exact extrusion form

For geometry invariant along the profile height axis, the precompute only needs
the in-plane velocity

\[
\dot q_{uv}=(\dot u,\dot v), \qquad
s_{uv}=\|\dot q_{uv}\|.
\]

If the bake returns projected in-plane distance `rho=|OA|`, then

\[
t_B=t_O+\frac{\rho}{s_{uv}}.
\]

In an orthonormal profile with a unit world ray, `s_uv=cos(alpha)`, producing the
article's displayed formula. In a non-orthogonal or non-uniformly scaled chart,
`s_uv` is the correct generalized denominator. Computing a world-horizontal
cosine and separately applying scale or slope is not equivalent.

### 2.2 Exact normal transform

A baked profile normal is a covector. For profile normal `n_q`, the corresponding
world normal is

\[
n_x=\operatorname{normalize}(J^{-T}n_q)
   =\operatorname{normalize}(K^Tn_q).
\]

Forward-transforming it with `J`, or dotting against individually normalized
"TBN" axes, is only correct for an orthonormal basis.

## 3. Exact top-entry reconstruction inside one affine chart

Assume a known ground point

\[
T=r(t_T)
\]

has profile height `h_T`, and that the same affine chart remains valid back to
the top plane `h=H`. Because profile height is linear along the ray,

\[
t_O=t_T+\frac{H-h_T}{\dot q_h}.
\]

For a downward ray, `dot(q)_h` is normally negative, so `O` is earlier than `T`.
If `T` is the chart's actual ground point, set `h_T=0` by definition. Do not
re-sample a second surface representation to estimate it.

The result is exact under three simultaneous conditions:

- `T`, the height coordinate, and `dot(q)_h` all use the same chart;
- that chart is affine and valid for the complete segment from `T` to `O`;
- no clamp replaces a parallel or near-parallel ray with a fabricated slope.

This is the mathematically valid ground-to-top reconstruction. The former code
did not meet its premises.

## 4. Heightfield specialization

For a planar terrain chart

\[
g(P)=g_0+m\cdot(P-P_0), \qquad m=(g_x,g_z),
\]

and vertical profile height

\[
q=(x,\ y-g(x,z),\ z),
\]

the exact inverse ray derivative is

\[
\dot q=(d_x,\ d_y-m\cdot d_{xz},\ d_z).
\]

For constant parallel lean `l=(l_x,l_z)`, use

\[
x=u+l_xh,\qquad
y=g_u u+h+g_vv,\qquad
z=v+l_zh.
\]

Then

\[
\Delta=1-g_ul_x-g_vl_z,
\]

\[
\dot h=\frac{d_y-g_ud_x-g_vd_z}{\Delta},
\]

\[
\dot u=d_x-l_x\dot h,\qquad
\dot v=d_z-l_z\dot h.
\]

These are components of `J^-1 d`, not a heuristic correction.

A nonlinear bend `F(h)` makes `J` depend on height. A straight world ray then
becomes a curved profile-space ray. Freezing `J^-1 d` at one height is not an
exact inversion of an arbitrary 3D mesh. The isolated rigid Calamagrostis path
has deformation disabled, so this is not the cause of its earlier gross
angle-dependent failure.

## 5. The former ground-to-top defect

The pre-envelope path combined:

- `T` from the rasterized terrain geometry;
- ground height at `T` from a separate bilinear guide field;
- ground gradient from a separate central-difference/filtered field.

Those values do not describe one coordinate chart. Let their height mismatch be
`delta`, their slope mismatch be `delta_m`, and define

\[
k=d_y-m\cdot d_{xz}.
\]

The same-chart shift is

\[
\Delta t=\frac{H}{k},
\]

while the mixed-chart operation is

\[
\widetilde{\Delta t}
=\frac{H+\delta}{k-\delta_m\cdot d_{xz}}.
\]

Their exact difference is

\[
\widetilde{\Delta t}-\Delta t
=
\frac{\delta k+H\,\delta_m\cdot d_{xz}}
{k\,(k-\delta_m\cdot d_{xz})}.
\]

Height disagreement is amplified approximately by `1/|k|`; slope disagreement
is amplified approximately by `1/|k|^2`. This directly predicts the observed
signature: nearly correct from above, increasingly stretched and radially
displaced toward grazing views.

The executable negative control uses an 8 mm height disagreement and a slope
disagreement of 0.008. At five degrees it already moves the reconstructed entry
by more than one metre. This is not a depth-format issue and not an inherent
artifact of precomputed rays. It is the result of using incompatible charts.

Clamping `k` to an invented epsilon or clamping `t_O` to a near distance changes
the ray. It cannot repair the coordinate map and can create camera-radial wedges.

## 6. Why the ground-hit triangle is not always enough

Using the exact raster triangle at `T` removes the mixed-chart error, but it is
only globally exact when that triangle's affine chart remains valid through
`O`. At grazing angles, `O` can lie over another terrain triangle.

The finite reference includes a continuous piecewise-planar counterexample:

\[
g(x)=
\begin{cases}
0,&x\le 0,\\
0.4x,&x>0.
\end{cases}
\]

A ten-degree ray hits the ground at `x=1` on the sloped piece. For `H=1.176 m`,
the real top-envelope intersection lies over the flat neighbour. Extending the
ground-hit plane backward satisfies the wrong plane and differs from the real
entry by more than two metres along the ray.

More generally, two terrain surfaces may share the same value and derivative at
`T` while differing earlier along the ray. They therefore provide identical
local inputs but have different `O`. No formula consuming only `T` and its local
jet can distinguish them.

The exact mathematical contract is consequently:

> Supply the first top-envelope intersection `O` and the winning affine chart,
> or supply an exactly equivalent intersection oracle.

This statement does **not** require a particular hardware pass. Valid suppliers
include a rasterized envelope, an equivalent depth-plus-primitive record, or an
analytic acceleration structure with a proven fixed candidate bound. Sannikov's
source implementation obtains `O` by rendering the outer envelope. Our earlier
claim that hardware rasterization itself was required was unjustified; the
required datum is the exact `O`, not one particular mechanism that produces it.

For one known bilinear ground patch,

\[
g(x,z)=a+bx+cz+dxz,
\]

substitution of `x(t),z(t)` produces a quadratic and can be solved exactly in
closed form. The returned root must still be checked against that patch's
domain. A piecewise surface requires determining the winning patch; silently
extending the ground-hit patch is not exact.

### 6.1 Exact constant-translation replacement theorem

The rigid ground-cover envelope has additional structure that removes the need
to construct a second translated surface. Let the carrier terrain be an
arbitrary set `S` (it may be piecewise, finite, non-planar, or a triangle soup),
let `v` be one constant vector, and define its translated envelope

\[
S_v=S+v=\{p+v\mid p\in S\}.
\]

For the view ray

\[
r_C(t)=C+td,
\]

translation by `-v` gives the exact equivalence

\[
C+td\in S_v
\quad\Longleftrightarrow\quad
(C-v)+td\in S.
\]

The parameter sets on the two sides are identical, not merely approximately
related. It follows immediately that all of the following are preserved:

- the first positive ray parameter and the complete ordered intersection set;
- miss/hit classification and finite-domain silhouette coverage;
- primitive owner, barycentric coordinates, edge ties, and coplanar intervals;
- grazing and parallel degeneracies;
- front/back facing and normals, because a translation changes neither.

For the current rigid profile, `v=(0,H,0)`. The exact shell-free query is thus:

1. leave the terrain `S` unchanged;
2. translate the ray/camera origin downward to `C-v` without changing `d`;
3. query the first terrain hit there;
4. retain the same parameter `t`, or map its point back by adding `v`.

No terrain chart needs to be predicted from the ordinary ground hit. The
shifted ray itself elects the correct chart, including a chart which is hidden
or absent from the ordinary view. The transform is one constant subtraction;
it introduces no ray march, iterative solve, or loop.

The exact result is therefore the tuple `(t, primitive owner, barycentrics)`,
not depth alone. Translation preserves the primitive id and barycentric
coordinates, so the translated primitive supplies the same affine chart at
`O`. Any later profile-space basis must use that owner/chart. Retaining exact
`O` depth but substituting a separately filtered guide gradient would recreate
a smaller mixed-chart error and does not satisfy the theorem.

This equivalence is also exact under perspective projection. With common camera
rotation `R`, view transform `V_C`, homogeneous translation `T_v`, and the same
projection `P`,

\[
V_C T_v = V_{C-v},
\]

and therefore

\[
P V_C (X+v)=P V_{C-v}X.
\]

The paired points have identical clip coordinates, NDC, nonlinear depth,
near/far clipping, projected triangle coverage, and silhouette edges. An exact
depth supplier may consequently query the original terrain from the translated
camera; no depth conversion is required.

This is **not** a transformation of the ordinary depth image from `C`. The
ordinary and shifted rays are parallel distinct lines whenever `v` is not
parallel to `d`. They may hit different primitives, and the shifted ray may hit
terrain where the ordinary ray has no terrain hit. Nor can one general
screen-space homography recover the result for non-planar terrain: translation
creates depth-dependent parallax and disocclusions.

The query must contain the carrier terrain only. If unrelated occluders `Q` are
included, translating the whole query would compare `S` with `Q-v`, not the
desired original-scene occlusion. Ordinary scene depth remains a separate upper
bound on the final botanical hit.

The theorem generalizes to any invertible affine map `F(X)=AX+b`:

\[
r(t)\in F(S)
\quad\Longleftrightarrow\quad
A^{-1}(C-b)+tA^{-1}d\in S.
\]

Translation is the special case `A=I`. A constant lean/top displacement is
still one translation. A spatially varying nonlinear displacement generally
maps a straight ray to a curve and has no single shifted-camera equivalent.

Finally, the surface equivalence remains true when the camera lies below the
envelope, but a *top-entry first-hit botanical atlas* then has a different
semantic problem: its stored first hit after the top entry can lie behind the
camera while a later visible hit lies ahead. For arbitrary finite 3D plants, a
single first-hit record cannot distinguish that case from a profile with no
later hit. Camera-inside correctness therefore requires an inside-origin query
or a bounded ordered-hit/interval carrier; it is separate from terrain-envelope
reconstruction.

### 6.2 Camera-inside visibility is a successor query

Fix one oriented profile-space line and parameterize it from the top reference
by `s`. Let its complete ordered surface-event set be

\[
I_\ell=\{s_1<s_2<\cdots<s_N\}.
\]

If the camera is at line coordinate `s_C`, the first visible event is exactly

\[
\sigma(\ell,s_C)=\min\{s_i\in I_\ell\mid s_i\ge s_C\}.
\]

The visible distance is `sigma-s_C`, divided by profile speed if `s` is a
normalised profile distance. A top-entry atlas stores only `s_1`; it answers the
successor only while `s_C<=s_1`. Clamping a negative result to zero, discarding
it, or taking its absolute value cannot recover a later event.

This has a strict information lower bound. Put `N` disjoint opaque sheets on one
line. Camera positions between successive sheets have `N` distinct successors.
Any carrier retaining only a fixed prefix `K<N` fails after event `K`. First and
last are also insufficient: `{1,4}` and `{1,3,4}` have identical endpoints but
different successors from `s_C=2`. For closed solids, parity answers only
inside/outside; it does not give the distance or owner of the next boundary.
Open blade sheets make parity still less applicable.

There is consequently no universal small `K` for arbitrary finite meshes. The
minimal exact abstract representation is either:

- the complete ordered event-owner list per oriented line, followed by a
  successor selection; or
- the successor owner itself as a function of ray origin and direction.

The second form has five degrees of freedom: an oriented line is four-dimensional
and the origin position along that line adds one scalar. A compact implementation
may partition this five-dimensional domain and store at most a certified fixed
`K` possible successor `(triangle, periodic-copy)` owners per cell. Runtime then
performs `K` statically expanded live ray/triangle tests, accepts `t>=0`, and
elects the minimum. This stores owner ids rather than replicated depth, normal,
or colour payloads. An equivalent sparse form stores ordered event-owner runs in
the four-dimensional line domain plus an origin-height slab index.

Exactness requires an offline closure proof, including owners on cell boundaries
and ties. A cell whose possible-successor set exceeds `K` must subdivide, raise
`K`, or reject the asset. A finite mesh and finite forward render interval make
the candidate set finite. An infinite periodic tiling without a finite horizon
does not: the number of crossed copies is unbounded as elevation approaches
horizontal. `t=0`, edge ties, and coplanar rays need explicit categorical
ownership rules rather than epsilon clamps.

Sannikov's parallel infinite extrusion avoids this extra origin dimension:
occupancy is invariant along profile height, so moving the camera along that
axis does not change the two-dimensional successor problem. Arbitrary finite 3D
botanical meshes do not have that invariance.

There is also a direction-sign rule which cannot be inferred from the envelope
point. Since

\[
O-C=t_Od,
\]

normalising `O-C` gives `sign(t_O)d`. When the camera is inside and the chosen
top reference is behind it, `t_O<0` and that construction reverses the ray. The
profile direction must always remain the forward pixel ray `d`; `-t_O` is used
only as the successor threshold `s_C`. A hit with camera-space `t<0` remains
behind the pinhole camera and may not be projected as a visible point.

For a downward live ray from inside the cover band, the required top reference
is the nearest envelope crossing behind the camera. The translation theorem
still supplies it without a duplicate surface: query the unchanged carrier from
`C-v` in direction `-d`, obtain positive distance `u`, and set `t_O=-u`. This is
the backward half of one signed line-intersection oracle. A forward-only depth
query cannot return it. Full arbitrary-direction camera-inside rendering also
requires the successor carrier's direction domain to cover those directions;
the current top-down-only hemisphere is not such a domain.

## 7. Arbitrary mesh profiles and angular reconstruction

The article's exact base class is a parallel extrusion. Its first-hit projected
path is two-dimensional, and elevation is recovered exactly by the projected
speed formula.

An arbitrary 3D plant mesh is different. First-hit triangle ownership varies
with both origin and direction and changes discontinuously at silhouettes and
occlusion boundaries. Linear interpolation of scalar depths or complete hit
records from different owners is not a geometric inverse.

For one known geometric plane with normal `eta`, a canonical record provides

\[
p_i=q_0+d_i\tau_i.
\]

The exact live-direction plane intersection is

\[
\tau_{live}
=\frac{\eta\cdot(p_i-q_0)}{\eta\cdot d_{live}}.
\]

The property test confirms that this rational reprojection matches the direct
plane intersection while linear directional depth blending does not. It is only
a complete mesh hit when:

- the live intersection remains inside the stored triangle; and
- that triangle remains the nearest owner.

A safe in-triangle radius proves only the first condition. A nearest owner from
one or four sampled directions does not guarantee that the true live owner was
recalled.

A certifiable fixed-cost arbitrary-mesh carrier therefore needs an offline
owner closure:

1. Subdivide each spatial/directional domain.
2. Store a conservative set of at most fixed `K` possible first-hit triangles.
3. Reject or further subdivide any domain whose closure exceeds `K`.
4. At runtime, perform exactly `K` statically expanded ray/triangle tests and
   elect the nearest valid hit.

That is O(1), contains no runtime march, and has an offline-checkable exactness
condition. The bounded sparse storage contract is fixed below; whether a
particular asset satisfies it is an offline certification result. Scalar
interpolation cannot be labelled exact in its place.

### 7.1 Distance and apparent-orientation invariance

Let the exact shifted-envelope query elect carrier chart `i` at `O`, with affine
profile map `x=b_i+J_i q`. For a live world direction `d`, all directional and
orientation quantities are

\[
a=J_i^{-1}d,\qquad \hat a=\frac{a}{\lVert a\rVert},\qquad
n_x=\operatorname{normalize}(J_i^{-T}n_q).
\]

None contains `t_O` or camera-to-cover distance. Translating the camera along
the same viewing line leaves `d`, `J_i^-1 d`, the profile direction, and the
inverse-transpose normal unchanged. It only adds the opposite translation to
the ray parameter of the same hit. Exact perspective makes a fixed object
smaller with range; it does not rotate a top into a side.

Therefore an apparent orientation that correlates with distance is not a new
perspective correction term. It proves that at least one supposedly invariant
input changed. The possible mathematical causes are:

- the shifted-envelope owner/chart changed, or the ordinary ground owner was
  substituted for it;
- the direction was derived from an unnormalised distance-bearing displacement;
- filtering, LOD, quantisation, or pixel footprint selected different owners;
- scalar directional interpolation fabricated a hit between different owners;
- a non-affine profile transform was treated as one constant affine basis.

The shifted owner is load-bearing. On affine ground
`g(x,z)=g_0+m_x x+m_z z`,

\[
J^{-1}d=(d_x,\ d_y-m_xd_x-m_zd_z,\ d_z),
\]

and a profile top normal transforms to

\[
J^{-T}e_h=(-m_x,1,-m_z).
\]

Using a different terrain owner's slope corrupts both ray elevation and normal;
a sufficiently wrong slope can literally turn a top-facing record toward a
side-facing world direction even when entry depth is exact.

Within one fixed geometric plane `eta dot q=c`, live depth is rational,

\[
\lambda(a)=\frac{c}{\eta\cdot a}.
\]

Its relative sensitivity is approximately
`delta(lambda)/lambda=-(eta dot delta(a))/(eta dot a)`, so angular errors amplify
at grazing incidence but contain no absolute-range term. Reciprocal depth is
linear in an unnormalised direction only while origin and owner plane are fixed.
That identity cannot justify filtering across triangle owners.

In particular, the owners at the two or four sampled directions surrounding a
live angle are not a conservative owner closure. A narrow foreground triangle
may win strictly between samples while every canonical sample sees the same
background triangle. Exact arbitrary-mesh angular reconstruction must therefore
store a certified owner closure and intersect its fixed candidates with the
live ray. After election, shading normals and authored colour come from live
barycentrics on that winning triangle, followed by the inverse-transpose normal
map. Blending sampled normals is not the same operation.

### 7.2 The current finite-direction carrier and the exact replacement

The committed Calamagrostis asset has a `256x256` periodic origin lattice and 64
canonical directions: 16 azimuths times elevations `15, 35, 55, 75` degrees.
Its ordinary carrier converts each canonical 3D first hit into inverse projected
path and filters the four surrounding direction records. Live elevation outside
the stored range is clamped to the endpoint.

That projected-path filtering is exact for Sannikov's parallel extrusion,
because the two-dimensional path is independent of elevation and the live lift
is exactly `rho/||P J^-1 d||`. It is not an exact identity for an arbitrary
finite 3D plant. There, projected first-hit path changes with elevation and
triangle ownership. Even a horizontal plane one metre below the origin exposes
the range error: reusing a 15-degree projected path for a five-degree live ray
produces roughly `3.75 m` instead of the exact `11.47 m`. No final colour or
normal adjustment can repair that positional error.

The exact fixed-cost replacement is a **certified successor-owner closure**,
split so camera-inside support does not make the ordinary exterior carrier
dense:

1. Exterior/top-entry domains use a four-dimensional closure over periodic top
   phase and live direction. Each cell stores at most fixed `K_ext` triangle/copy
   owners that can be first anywhere in the cell.
2. Camera-inside domains add signed origin phase/height and store a sparse
   five-dimensional closure of at most fixed `K_in` possible successors.
3. Runtime intersects the live forward ray with every stored candidate using a
   statically fixed comparison network, rejects `t<0` and out-of-triangle hits,
   and elects the minimum. There is no depth interpolation, ray march, or
   data-dependent iteration.
4. The elected triangle supplies live barycentrics, authored colour, and local
   normal; the exact shifted terrain owner supplies `J`, and the normal uses
   `J^-T`.

This is a mathematical contract, not permission for a dense 5D texture. Cells
store compact owner/copy ids and share the existing immutable triangle data.
Adaptive subdivision is performed offline and must be certified conservatively.
The runtime lookup depth and candidate counts are fixed constants. A hard memory
budget is part of acceptance: if a cell cannot meet `K_ext`/`K_in` within the
fixed subdivision depth and memory cap, that bake is rejected rather than
silently approximated or allowed to grow without bound.

The closure is exact only inside an explicit finite render horizon and declared
direction domain. This is necessary, not a quality shortcut: periodic copy count
is unbounded as a downward ray approaches horizontal over an infinite horizon.
The domain must cover every angle the renderer promises; clamping an uncovered
angle to the nearest baked direction is not reconstruction.

A new five-degree canonical elevation row is approved as an additional seed for
the grazing domain. It eliminates the current direct 5-to-15-degree endpoint
substitution and should reduce closure page complexity. It does not replace the
closure: the continuum between canonical rows still uses certified candidate
owners and exact live intersections, never blended scalar path as proof.

The current asset also shows why the word *sparse* is mandatory. Its guarded
canonical atlas has 4,260,096 texels. One packed owner id per texel is already
17,040,384 bytes. Across those canonical samples there are 762,368 distinct
`(triangle,copy)` owners and 445,227 distinct source triangles. Restricting the
existing packed geometry table to only those canonically visible triangles and
their referenced vertices would still be about 21.4 MB. A blind dense `K=4`
owner array would consume about 68.2 MB for ids alone and is rejected by this
design.

The accepted storage form is therefore one bounded indirection field into
sparse fixed-width owner pages, or the equivalent sparse ordered-event runs plus
origin-height slab references. Its exact byte formula is

\[
B=B_{index}+4K\,N_{pages}+B_{referenced\ geometry},
\]

before format alignment. `K`, maximum subdivision depth, page count, referenced
geometry bytes, and total `B` are hard bake gates. The existing owner counts are
only a sizing observation, not a closure proof: triangles visible only between
canonical samples must also be included by certification.

### 7.3 Grazing elevation clamp collapses finite geometry into the top plane

The observed distant low-oblique sheet has a direct mathematical signature. Let
the profile top be `H`, let a canonical downward ray at elevation `beta` hit a
finite three-dimensional plant at height `h_beta`, and let `rho_beta` be its
horizontal projected path. Then

\[
\rho_\beta=\frac{H-h_\beta}{\tan\beta}.
\]

If a lower live elevation `alpha` is clamped to that canonical record and the
stored projected path is merely lifted with the live direction, the reconstructed
height is

\[
\widetilde h_\alpha
=H-\rho_\beta\tan\alpha
=H-(H-h_\beta)\frac{\tan\alpha}{\tan\beta}.
\]

Thus the entire vertical extent below `H` is compressed by the factor

\[
c=\frac{\tan\alpha}{\tan\beta}.
\]

At the current five-degree live view and fifteen-degree lowest stored row,
`c=0.3264`. For `H=1.1765 m`, an actual ground-level hit is reconstructed at
approximately `0.7925 m`, and an actual `0.2 m` hit at approximately `0.858 m`.
As `alpha` approaches zero, every finite-mesh hit converges to `H` regardless of
its true height. The limiting image is literally a textured top sheet. This
explains all three linked observations without any distance-dependent projection
term:

- distant ground rays in one perspective image approach the horizon and therefore
  have smaller `alpha` than foreground rays;
- the apparent plant depth collapses toward the top envelope in those pixels;
- only occasional records selected from another angular/ownership cell retain a
  hint of lower structure, producing stripes rather than coherent plants.

Adding a five-degree canonical row removes the particularly large `5 -> 15`
collapse only at that row. It cannot make the continuum exact. Between rows, a
finite mesh still has piecewise-rational depth

\[
\tau(n)=\frac{c_i}{\eta_i\cdot n}
\]

while owner `i` remains valid, followed by a discontinuous minimum when ownership
changes. Linear interpolation of depth, inverse depth, projected path, colour, or
normal is not this function. The exact runtime operation remains live intersection
of a certified owner closure.

The same objection applies before angular reconstruction. For fixed direction,
first-hit distance as a function of top-plane phase `(u,v)` is also a lower
envelope of triangle-plane intersections with discontinuities at silhouettes and
owner changes. Bilinearly filtering the four neighbouring first-hit records can
therefore fabricate a surface which no source triangle contains. A closure cell
must cover both continuous phase and direction; neither axis may use scalar record
interpolation as its correctness proof.

### 7.4 World cover selection is a predicate-filtered successor query

Let the complete ordered events of the fully populated periodic profile on one
forward line be

\[
E_\ell=\{(s_1,o_1),(s_2,o_2),\ldots\},
\]

where owner `o_i` includes triangle, periodic copy/root, and geometry layer. Let
`A(o_i)` say whether that root is active in the cooked world cover field. The
actual visible event is

\[
\sigma_A(\ell,s_C)
=\min\{s_i\mid s_i\ge s_C\ \land\ A(o_i)\}.
\]

This is a successor query even when the camera is above the top envelope. Querying
only `(s_1,o_1)`, rejecting it when `A(o_1)=false`, and returning miss is wrong
whenever a later active plant exists. It clips away the sides of a covered patch.
The same order matters across the two anti-tiling layers: the minimum must be taken
between each layer's first **eligible** event. Taking the unfiltered minimum first
and rejecting its owner can discard a valid farther event from the other layer.

The exact fixed-cost closure must therefore be certified against the declared
eligibility family, not merely against the first owner of the fully populated
profile. For a ray-domain cell `D` and allowed predicate family `mathcal A`, its
required candidate set is

\[
C(D)=\bigcup_{(q,n)\in D,\ A\in\mathcal{A}}
\operatorname{owner}(\sigma_A(q,n)).
\]

The bake may accept a cell only when `|C(D)| <= K`. Runtime intersects all `K`
statically expanded triangle/copy candidates with the exact live ray, rejects
behind-camera, out-of-triangle, out-of-horizon, and inactive-root candidates, then
elects the minimum. Layer identity participates in that one election. This is
loop-free O(1); it is not a first-hit texture followed by a coverage filter.

For arbitrary masks and an unbounded periodic horizon, `C(D)` need not be bounded.
Exactness therefore requires a declared finite horizon and a restricted/certified
world-predicate family. If the fixed `K`, subdivision-depth, and memory gates cannot
certify it, the asset or coverage representation must be rejected rather than
silently clipping its patch sides. Camera-inside support adds the signed origin
coordinate to this same predicate-filtered successor relation; it is not a separate
approximation.

### 7.5 The envelope owner is not automatically the botanical owner chart

The shifted terrain query supplies the exact top-envelope point and the terrain
primitive beneath that point. That chart is sufficient to establish `O`; it is
not automatically the affine transform of a plant hit metres farther along a
grazing ray.

For an upright discrete plant rooted at horizontal coordinate `p`, the natural
map of a local source point `(xi,h)` is

\[
X_{p}(\xi,h)=(p+\xi,\ g(p)+h),
\]

or a root-specific affine transform `B_p+A_p q` when the plant is aligned to a
root frame. The correct transform belongs to the elected periodic copy/root. If
instead the whole profile is draped continuously as

\[
X(u,h,v)=(u,\ g(u,v)+h,\ v),
\]

then non-affine terrain makes a straight world ray curved in profile coordinates.
Freezing the terrain derivative at `O` is exact only while the same affine terrain
chart remains valid over the complete segment to `B`.

Consequently the owner closure must name not only a source triangle and periodic
copy but also its root transform. Runtime transforms that triangle with the
root's exact ground datum and intersects it with the live world ray. Alternatively,
a closure for a continuously draped profile must be certified over the allowed
terrain-transform family. A post-hit ground-range check cannot repair a hit that
was selected with the wrong transform; it can only reject it.

## 8. Multiple heights and overlapping cover

`H=1.176 m` is the current Calamagrostis asset's top coordinate. It is not a
generic ground-cover height.

There are two mathematically coherent multi-profile arrangements:

- Each profile `i` has its own exact envelope entry `O_i` at height `H_i`. For a
  constant vertical envelope this is exactly the unchanged terrain queried from
  `C-(0,H_i,0)`. Its fixed candidate then competes by reconstructed world depth.
- Every profile is baked from one shared conservative height `H_*` at or above
  all geometry. Shorter species contain empty air between `H_*` and their real
  top. All candidates begin at the exact terrain query from `C-(0,H_*,0)` and
  elect the nearest hit.

The second arrangement does not make every plant `H_*` tall; it only shares a
ray origin. Overlap is the minimum positive reconstructed hit across the fixed
candidate set. Either arrangement remains O(1) when the candidate count is
statically bounded.

A shared constant envelope can remain exact even when wind moves plant geometry
horizontally; it need only be a conservative carrier above every permitted
state. For upright profiles rooted at horizontal coordinate `p`, write a local
plant point (including any wind state) as horizontal offset `xi` and vertical
offset `y`. Its world position is

\[
(p+\xi,\ g(p)+y).
\]

The translated terrain `g(x)+H_*` lies above every such point exactly when

\[
H_*\ge
\sup_{p,\xi,y}\left(y+g(p)-g(p+\xi)\right).
\]

Thus spatially varying wind does not force a spatially varying entry surface if
one constant conservative `H_*` covers its complete horizontal/vertical motion
set and the profile bake retains the resulting empty air. If the terrain is
`L`-Lipschitz over the allowed horizontal reach, the simpler sufficient bound is

\[
H_*\ge\sup_{\xi,y}\left(y+L\lVert\xi\rVert\right).
\]

This bound concerns the common entry carrier only. The profile-space transform
must still represent the actual wind deformation of the botanical geometry.

## 9. Singular and boundary cases

- If full profile speed `||J^-1 d||` is zero, the chart or ray is invalid.
- If projected speed is zero and projected path is zero, the origin occupancy
  hit is exactly zero and must remain categorical.
- If projected speed is zero and projected path is nonzero, the projected
  inverse is undefined; inventing an epsilon produces unbounded geometry.
- If the ray is parallel to the top envelope, entry is undefined or non-unique.
- A basis with `det(J)=0` is not a coordinate chart.
- Directional and spatial interpolation may cross first-hit ownership. Coverage
  unmixing prevents a miss sentinel from contaminating a hit, but it does not
  make mixed owners one surface.

## 10. Executable proof boundary

The CPU reference currently proves twenty-three finite properties:

1. full inverse-basis reconstruction under a skewed affine map;
2. projected-path reconstruction using profile projected speed;
3. exact same-chart ground-to-top entry;
4. exact first-hit/owner/barycentric preservation under a constant translated
   envelope versus a shifted ray origin;
5. exact recovery of a translated-surface silhouette where the ordinary ground
   ray has no hit;
6. insufficiency of one top-entry first-hit record for arbitrary
   camera-inside visibility;
7. successor-event semantics for arbitrary camera origins, including `t=0`;
8. failure of a peeled-hit prefix below the line's depth complexity;
9. preservation of the forward pixel direction when a camera-inside top entry
   lies behind the camera;
10. grazing amplification from a mixed height/gradient chart;
11. insufficiency of value plus gradient on curved terrain;
12. failure when `T` and `O` belong to different terrain pieces;
13. closed-form intersection on one known bilinear patch;
14. exact plane reprojection versus incorrect linear depth interpolation;
15. invariance of profile direction, hit point, and apparent orientation along
    one view ray;
16. failure of surrounding sampled directions to form an arbitrary-mesh owner
    closure;
17. failure of surrounding sampled origin texels to form an arbitrary-mesh
    owner closure;
18. invalidity of clamping a grazing 3D-profile direction to the lowest stored
    elevation;
19. exact vertical-range collapse caused by lifting that clamped projected path
    with a lower live elevation;
20. predicate-filtered successor semantics at a world-cover boundary;
21. necessity of applying root eligibility before anti-layer depth election;
22. inverse-transpose normal transformation under shear;
23. finite categorical handling of vertical origin occupancy.

Run it with:

```sh
npx tsx --test tools/groundcover-bake/GroundCoverRayMath.test.ts
```

The test is deliberately independent of WebGPU, depth packing, texture formats,
Three.js, and shader compilation. Returning to the shader is permitted only
after an implementation choice supplies the exact entry/chart contract above
and preserves these equations without clamps or coordinate substitutions.
