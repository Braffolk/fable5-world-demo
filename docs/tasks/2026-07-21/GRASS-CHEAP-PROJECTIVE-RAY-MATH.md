# Cheap projective reconstruction for precomputed ground cover

Status: active pure-mathematics design, 2026-07-21. This note supersedes the
runtime owner-closure proposal in `GRASS-RAY-RECONSTRUCTION-MATH.md`. It contains
no render-pipeline or shader implementation decision.

## 1. Corrected objective

The runtime target is Sannikov's low/mid-range architecture:

- expensive source ray casting is offline;
- runtime performs one, or at most a very small fixed number of, coherent
  texture reads plus a bounded coordinate transform;
- runtime work is independent of source triangle and blade count;
- there is no ray march, candidate expansion, triangle loop, or source-mesh
  table in the live reconstruction.

The earlier owner-closure analysis silently changed the problem into continuum-
exact reproduction of an arbitrary source mesh under arbitrary independently
enabled roots and arbitrary camera origins. That is stronger than the published
method and stronger than the visual target. Its fixed live triangle tests are a
bounded brute-force fallback, not the accepted runtime direction.

For a finite sampled atlas, the right correctness claim is:

1. exact at baked rays, up to quantisation;
2. projectively exact between samples while the represented surface plane is
   unchanged;
3. bounded discretisation error at visibility/owner discontinuities;
4. never fabricate large-scale geometry by applying an identity outside its
   mathematical domain.

## 2. The published extrusion theorem

Let profile coordinates be

\[
q=(u,h,v)^T,
\qquad
X=b+Jq,
\]

where the columns of `J` are the honest, unnormalised derivative basis. Let the
periodic two-dimensional mask lie in `(u,v)` and be invariant along `h`. For the
world ray

\[
r(t)=O+td,
\]

define its profile derivative

\[
a=J^{-1}d=(a_u,a_h,a_v)^T,
\qquad
p=(a_u,a_v),
\qquad
s_p=\lVert p\rVert,
\qquad
\omega=p/s_p.
\]

The precomputed mask ray returns the in-plane first-hit path `rho` in direction
`omega`. Because occupancy is invariant along `h`, the exact world parameter is

\[
\boxed{t_B=\frac{\rho}{s_p}}.
\]

For an orthonormal profile and unit world ray, `s_p=cos(alpha)`, yielding the
article's displayed identity

\[
|OB|=\frac{|OA|}{\cos\alpha}.
\]

This is a projection identity for a parallel extrusion. It is not an angular
reconstruction identity for a finite three-dimensional plant.

## 3. Finite-mesh ray field

Let the top plane be `y=H`. After the affine inverse from Section 2, write the
profile derivative as

\[
a=J^{-1}d.
\]

Parameterise every downward profile ray by top phase

\[
q=(q_x,q_z),
\]

horizontal slope

\[
s=\frac{a_{xz}}{-a_h},
\]

and vertical drop `h=H-y`:

\[
\boxed{Q(q,s,h)=(q_x+s_xh,\ H-h,\ q_z+s_zh)}.
\]

The finite-mesh first-hit field is the four-dimensional function

\[
D(q,s)=\inf\{h\ge 0\mid Q(q,s,h)\in G\}.
\]

A direct finite-mesh atlas stores a discretisation of the bounded vertical-drop
field `D`, together with normal, colour, material, and validity. Runtime samples
the complete view-conditioned record and reconstructs

\[
h=D(q,s),
\qquad
q_B=Q(q,s,h),
\qquad
t_B=\frac{h}{-a_h}.
\]

The corresponding world hit is

\[
B=b+Jq_B.
\]

At the exact baked slope, full distance `tau` along unit profile direction
`a_hat` is equivalent:

\[
h=-\widehat a_h\tau,
\qquad
t_B=\frac{\tau}{\lVert a\rVert}.
\]

That equivalence does **not** permit applying `tau_i` from one canonical slope to
a different live slope. `tau_i` is metric distance along the canonical ray. The
ray-space field's stored dependent coordinate is `h_i`; categorical selection
must carry that vertical drop to the live line and use

\[
t_{live}=\frac{h_i}{-a_{live,h}}.
\]

Unlike projected-path relifting or canonical-`tau` reuse, this preserves the
sampled botanical height and cannot algebraically collapse every record toward
the top plane as live elevation decreases.

The finite mesh requires both direction coordinates. Its elevation-specific
distance must remain elevation-specific.

## 4. The current category error

The current finite-mesh path converts each baked full distance `tau_i` into a
horizontal projected path

\[
\rho_i=\tau_i\lVert \widehat a_{i,xz}\rVert,
\]

then lifts that path using a different live elevation:

\[
\widetilde\tau
=\frac{\rho_i}{\lVert \widehat a_{live,xz}\rVert}.
\]

That operation deliberately discards the finite mesh's elevation-specific hit
height and treats it as an extrusion. If a canonical elevation `beta` hit is
used for a lower live elevation `alpha`, reconstructed height becomes

\[
\widetilde h_\alpha
=H-(H-h_\beta)\frac{\tan\alpha}{\tan\beta}.
\]

Consequently every hit collapses toward `H` as `alpha` approaches zero. This is
the observed distant top sheet. The failure is not an inherent limitation of a
precomputed finite-mesh ray field. It is the result of applying the extrusion
theorem to non-extruded geometry.

The minimum source-faithful correction is therefore:

1. retain vertical drop `h=-a_hat_h*tau` for every elevation row;
2. cover the declared view domain, including the grazing guard row;
3. select a complete direction-conditioned record;
4. reconstruct the live line with that record's bounded vertical drop, without
   projected-path relifting or canonical-distance reuse.

## 5. One-read projective plane reconstruction

The direct record can be made more accurate between angular samples without any
live triangle test.

Let the baked hit lie on the geometric plane

\[
n\cdot q=c.
\]

Define the top point and vertical-drop ray derivative

\[
O(q)=(q_x,H,q_z),
\qquad
v(s)=(s_x,-1,s_z).
\]

Define the plane numerator relative to the top origin

\[
\kappa(q)=c-n\cdot O(q).
\]

The exact intersection with the represented plane is

\[
\boxed{h(q,s)=\frac{\kappa(q)}{n\cdot v(s)}}.
\]

At a baked slope `s_i` and drop `h_i`, the record can obtain the numerator as

\[
\kappa(q)=h_i\,n\cdot v(s_i).
\]

Thus one atlas record containing `(kappa,n)` replaces a stored scalar distance
with a plane covector. Runtime is one texture read, one denominator dot product,
and one division. It is exact for the represented source triangle plane for any
live direction for which that plane remains the visible owner.

This is the projective generalisation of `OA/cos(alpha)`. For an extruded side
plane, `n_y=0`; changing elevation changes only the projected speed in the
denominator, reducing the equation to Sannikov's identity.

The atlas is therefore best understood as an offline owner-plane selector. The
runtime does not recover or intersect a triangle. It intersects the single plane
record already selected by the precomputed ray field.

### 5.1 Categorical texel selection with exact phase correction

Let the atlas select one complete record at top-phase texel centre `q_i`, storing

\[
\kappa_i=c-n\cdot O(q_i).
\]

For the live phase `q`, the same plane's numerator is recovered exactly by

\[
\boxed{
\kappa(q)=
\kappa_i-n_x(q_x-q_{i,x})-n_z(q_z-q_{i,z}).
}
\]

The live drop is then

\[
\boxed{
h(q,s)=
\frac{
\kappa_i-n_{xz}\cdot(q-q_i)
}{n_{xz}\cdot s-n_y}.
}
\]

Thus nearest categorical selection in the complete four-dimensional ray atlas
does not quantise the reconstructed plane to the texel centre. It uses the texel
only to select one owner-plane chart, then evaluates that chart at the exact live
phase and direction. Runtime remains one read plus bounded scalar ALU.

This is preferable to filtering records across unknown owners. At a `256x256`
phase lattice over a `0.52 m` tile, the point-location uncertainty band is about
`2.03 mm` wide before angular discretisation; it does not justify interpolating
unrelated surfaces into a large sheet.

### 5.2 Spatial filtering identity within one owner

For one fixed plane, `n` is constant and

\[
\kappa(q)=c-n_xq_x-n_yH-n_zq_z
\]

is affine in top phase. Bilinear phase filtering of a consistently oriented
`(kappa,n)` record is therefore exact while all contributing samples represent
the same plane. It is safe only when the bake proves that shared ownership; the
default complete-record selector remains categorical. Direct bilinear depth
filtering does not have this property for changing direction.

### 5.3 Equivalent depth-gradient form

For a canonical slope `s_i`, let `h_i=D_i(q)` and

\[
g_i=\nabla_qD_i(q).
\]

The exact top-phase shift of a canonical ray through the live hit is

\[
q_i=q+(s-s_i)h.
\]

For one planar depth field,

\[
D_i(q+\delta q)=h_i+g_i\cdot\delta q.
\]

Substitution gives the equivalent closed form

\[
\boxed{h(s)=\frac{h_i}{1-g_i\cdot(s-s_i)}}.
\]

This is an exact line-plane solve, not a first-order angular approximation.
The denominator's zero is the real parallel-plane pole; it must be handled
categorically, never replaced by an epsilon clamp.

The gradient must come from the geometric hit plane. A smooth shading normal is
not a depth gradient.

### 5.4 Actual-asset boundary of the one-plane theorem

The theorem is correct, but nearest canonical record selection does not satisfy
its owner premise for the accepted Calamagrostis mesh. In the first 53,248-ray
offline comparison, all 27,999 accepted live plane intersections lay outside the
microscopic source triangle which supplied the record; 2,984 more left the
botanical height band. The representation has millions of tiny triangles, so a
16-azimuth categorical record almost never owns the neighbouring live ray.

Therefore nearest-record plane extrapolation is rejected for this asset. A plane
chart becomes usable only if direct line-space point location first selects the
correct visibility cell, as in Section 8.1. The algebra must not be used as a
substitute for that selection.

## 6. Exact slope interpolation on one plane

If retaining only depth is preferable, there is a three-read projective form.
For one plane at fixed phase `q`, reciprocal vertical drop is affine in slope:

\[
\frac1{h(q,s)}
=\frac{n_xs_x-n_y+n_zs_z}
{c-n_xq_x-n_yH-n_zq_z}.
\]

Choose three canonical slopes forming a triangle around the live slope and
barycentric weights satisfying

\[
\sum_i\lambda_i=1,
\qquad
s=\sum_i\lambda_i s_i.
\]

Then

\[
\boxed{
h(q,s)=\left(\sum_i\frac{\lambda_i}{h(q,s_i)}\right)^{-1}
}.
\]

For an affine surface attribute `a`, perspective-correct interpolation is

\[
\boxed{
a(q,s)=
\frac{\sum_i\lambda_i\,a_i/h_i}
{\sum_i\lambda_i/h_i}
}.
\]

This is exact while the three records lie on the same triangle plane. It uses
three coherent, spatially filtered atlas reads and no source geometry. It is a
fallback if the one-read plane record is not sufficiently stable at visibility
changes, not permission to expand runtime work with geometry complexity.

## 7. Direction coordinates

Linear interpolation in azimuth/elevation does not reproduce a live ray slope
and is therefore not projectively exact. Canonical directions must be embedded
in Cartesian slope space

\[
s=(s_x,s_z)=\frac{d_{xz}}{-d_y}.
\]

Use a fixed triangulation of this plane. Barycentric weights reproduce `s`
exactly and make reciprocal-depth interpolation exact for a shared plane.

The direction domain needs:

- a centre vertex `s=0` for the vertical view;
- enough angular sectors for visibility discretisation;
- a grazing guard ring whose polygon encloses the complete promised slope disk.

For a promised minimum elevation `alpha_min`, the disk radius is

\[
r_{max}=\cot\alpha_{min}.
\]

An inscribed finite azimuth ring at exactly `r_max` does not cover directions
between its vertices. With `N` uniform sectors, a circumscribing guard ring uses

\[
r_{guard}=\frac{r_{max}}{\cos(\pi/N)}.
\]

This avoids endpoint clamping and extrapolating the most sensitive grazing
rays.

### 7.1 Coupled phase-direction sampling bound

The same-point identity

\[
q_i=q+(s-s_i)h
\]

gives a direct angular sampling bound. On a slope ring of radius `r`, `N`
uniform azimuth samples have a worst nearest-sample slope difference

\[
\lVert\Delta s\rVert
=2r\sin\left(\frac{\pi}{2N}\right).
\]

At vertical drop `h`, this is the unwrapped phase shear

\[
\boxed{
\lVert\Delta q\rVert
=2hr\sin\left(\frac{\pi}{2N}\right).
}
\]

For `h=1.175 m`, five-degree `r=cot(5 deg)=11.43`, and the current `N=16`,
the worst shear is about `2.63 m`, more than five complete `0.52 m` periodic
tiles.

The profile is periodic, so the lookup phase is `Delta q mod T`; translating by
an exact integer tile is not itself an error. The significance of the unwrapped
bound is frequency: one current angular half-bin winds the correspondence around
the phase torus more than five times over the botanical height range. Small
changes of hit height can therefore select effectively unrelated phases and
owners. This is consistent with the measured direction-only instability; the
raw `2.63 m` is not asserted to be the final world-position error.

To keep the displacement below a chosen phase tolerance `epsilon`,

\[
\boxed{
N\ge
\frac{\pi}
{2\arcsin\!\left(\epsilon/(2hr)\right)}.
}
\]

At the same worst-case height and slope, the sufficient no-full-wrap condition
`epsilon=0.52 m` requires about 82 azimuth samples. A `0.10 m` unwrapped-shear
bound requires about 422. These are sufficient sampling conditions, not
necessary ones: periodic coincidences and phase reprojection can do better. They
do prove that uncorrected 16-azimuth categorical lookup is deeply aliased over
the declared grazing domain.

The atlas budget must consequently be allocated across the four-dimensional
ray field rather than maximizing phase resolution alone. Reducing phase
resolution while increasing slope-space density can retain the same bytes and
runtime read count. The actual-asset comparison, screen-space footprint, and
cache-locality gate choose that allocation; source-triangle exactness is not the
metric.

## 8. Discretisation and visibility boundaries

No finite atlas is continuum-exact at every arbitrary-mesh visibility boundary.
That is a discretisation issue, not a reason to upload and re-intersect source
triangles.

Exact within the represented owner plane:

- full-distance reconstruction at baked rays;
- plane-covector reconstruction;
- depth-gradient reprojection;
- reciprocal-slope interpolation;
- perspective-correct affine attributes.

Approximate and controlled by bake resolution/filter policy:

- a live-only owner between angular samples;
- disocclusion and silhouettes;
- phase filtering across different owners or miss records;
- non-affine texture colour inside a triangle;
- record selection at an angular cell boundary.

The cheap policy is to improve the offline ray field and its parameterisation:
more appropriate direction samples, categorical hit/miss handling, complete-
record selection, projective plane records, and depth-aware mip/filter rules.
The rejected policy is to reconstruct source topology per pixel.

### 8.1 Direct line-space point location

If nearest categorical record selection leaves visible angular boundaries, the
mathematical replacement is direct point location in the precomputed visibility
field, not candidate intersection.

Represent an oriented line by Pluecker coordinates `(d,m)`, with

\[
m=o\times d,
\qquad
d\cdot m=0.
\]

For another oriented line `(e,m_E)`, define the reciprocal product

\[
\boxed{
\Omega((d,m),(e,m_E))=d\cdot m_E+e\cdot m
}.
\]

`Omega=0` exactly when the two projective lines intersect or are coplanar. The
three reciprocal-product signs against a triangle's oriented edges are its exact
line-sidedness predicates.

For a static triangle mesh, first-owner transitions occur when either:

1. the ray-plane hit crosses a triangle edge; or
2. the depth order of two triangle planes changes.

The first boundary is `Omega(L,E_edge)=0`. The second occurs when the query line
meets the intersection line of the two planes and is also one reciprocal-product
boundary. Thus first-owner identity is piecewise constant in four-dimensional
line space, separated by projective hyperplanes.

An offline visibility-complex bake can exploit this directly:

- ordinary ray-space cells store one plane/surface chart;
- a boundary cell stores one or a very shallow fixed decision tree of separator
  forms and leaf chart ids;
- runtime evaluates the fixed separator tree and selects exactly one chart;
- runtime intersects only that selected plane.

This is one/few-read direct point location. It does not test a list of triangles
or choose the minimum. A cell exceeding the declared tiny separator depth is
subdivided offline. If the resulting field fails fixed depth, byte, or locality
gates, the bake fails instead of moving source geometry work into the shader.

The first experiment remains nearest complete-record selection because it is the
smallest faithful correction. Direct point location is the next exact boundary
tool if categorical angular quantisation is visible.

### 8.2 Exact affine attributes without source geometry

Let an authored triangle attribute be affine on its plane:

\[
a(q)=a_i+C_i(q-q_i).
\]

The selected precomputed surface chart may store `a_i` and the two in-plane
derivatives `C_i`. After the one plane intersection, runtime evaluates this
chart at the live point. This gives exact triangle-interpolated UV, unnormalised
vertex RGB, or another affine attribute without a triangle id, barycentric test,
or vertex-buffer access. Normalisation and authored texture lookup happen after
the affine transport.

If the atlas stores only the canonical hit colour, colour remains an ordinary
sampled light-field approximation even when geometry uses exact plane transport.

## 9. Separate domains: world support and camera-inside views

Two earlier requirements genuinely add information, but neither justifies live
triangle brute force.

### 9.1 World support

Arbitrary independent post-hit deletion of periodic roots changes a first-hit
query into a successor query. That representation is not part of Sannikov's base
theorem. A low-cost ground-cover system should instead select a precomputed
density/profile/support variant before the ray-field lookup, or define support
through the carrier domain. It must not solve an unnecessarily general Boolean
root-deletion problem with live geometry.

The precise variant/support construction remains open and must be solved as a
precomputed-field design before generic ground cover is enabled. It is not a
blocker for correcting the isolated fully populated Calamagrostis ray field.

### 9.2 Camera inside finite cover

A top-first finite-mesh atlas does not contain the next event after an arbitrary
inside origin. The cheap representation, if this domain is required, is another
precomputed lookup dimension or a compact precomputed successor/deep-ray record,
not source triangles.

For normal player-height rendering, an exterior-only profile may be an honest
declared domain. Free-camera debugging below `H` must not silently redefine the
production representation.

## 10. Performance contract

Accepted runtime families:

- one complete-record lookup plus `J^-1 d` and a projective scalar solve;
- one lookup plus one fixed corrective lookup if a measured visual defect
  requires phase reprojection;
- at most one slope-space triangle's three coherent reads when that materially
  improves quality over the one-record form.

Rejected runtime families:

- per-pixel source triangle or root candidate arrays;
- live ray/triangle tests whose count follows geometric depth complexity;
- ray marching or data-dependent traversal;
- dense arbitrary successor volumes added without a separately justified memory
  and locality design;
- extra passes that compensate for unresolved ray mathematics.

The first implementation experiment is selected only after the fixed-read
actual-asset comparisons pass. The current 16-azimuth carrier does not: its
half-bin direction error dominates phase error by orders of magnitude, and
nearest plane extrapolation is invalid. The final sampled-field comparison must
therefore test exact phase reprojection and denser angular support under a fixed
atlas-byte budget. Direct line-space point location remains the analytical
fallback if sampled categorical boundaries cannot meet the quality bar.
