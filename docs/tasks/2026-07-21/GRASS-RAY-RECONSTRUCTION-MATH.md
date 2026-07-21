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
condition. Whether its memory/performance budget is acceptable is a later design
decision; scalar interpolation cannot be labelled exact in its place.

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

The CPU reference currently proves thirteen finite properties:

1. full inverse-basis reconstruction under a skewed affine map;
2. projected-path reconstruction using profile projected speed;
3. exact same-chart ground-to-top entry;
4. exact first-hit/owner/barycentric preservation under a constant translated
   envelope versus a shifted ray origin;
5. exact recovery of a translated-surface silhouette where the ordinary ground
   ray has no hit;
6. insufficiency of one top-entry first-hit record for arbitrary
   camera-inside visibility;
7. grazing amplification from a mixed height/gradient chart;
8. insufficiency of value plus gradient on curved terrain;
9. failure when `T` and `O` belong to different terrain pieces;
10. closed-form intersection on one known bilinear patch;
11. exact plane reprojection versus incorrect linear depth interpolation;
12. inverse-transpose normal transformation under shear;
13. finite categorical handling of vertical origin occupancy.

Run it with:

```sh
npx tsx --test tools/groundcover-bake/GroundCoverRayMath.test.ts
```

The test is deliberately independent of WebGPU, depth packing, texture formats,
Three.js, and shader compilation. Returning to the shader is permitted only
after an implementation choice supplies the exact entry/chart contract above
and preserves these equations without clamps or coordinate substitutions.
