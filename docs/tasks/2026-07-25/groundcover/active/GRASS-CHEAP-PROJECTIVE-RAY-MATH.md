# Cheap projective reconstruction for precomputed ground cover

Status: active pure-mathematics design, 2026-07-21. This note supersedes the
runtime owner-closure proposal in `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-RAY-RECONSTRUCTION-MATH.md`. It contains
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

## 11. Primary-source correction: the omitted dimension is intentional

The later author comments remove an ambiguity in the article. A complete first-
hit field for an arbitrary static mesh is four-dimensional: two carrier
coordinates and two view-direction coordinates. Sannikov explicitly says that
his three-dimensional texture omits the angle above the horizon, assumes the
grass is vertical, and derives the missing angular dependence as for vertical
grass. He also says that the Maya-mesh version bakes the mesh as if viewed from
one fixed angle and compensates the resulting artifacts in the shader.

This changes the target representation. It is **not** a compressed exact point
locator for the unchanged arbitrary mesh. It is a one-read view-conditioned
impostor whose missing dimension is supplied by a coherent geometric transform.
The published production workaround of flattening a patch so the camera does
not look along the singular grain is evidence about the representation's pole;
it is not an acceptable contract for this project. A free camera must be
covered without flattening the accepted Calamagrostis source or forbidding a
view.

The source comments are in the article discussion, especially replies 34, 37,
40, and 43:

- <https://gamedev.ru/code/forum/?id=246320&page=3>
- <https://gamedev.ru/code/articles/grass_raycast>

## 12. Exact affine conjugacy of one canonical ray field

There is nevertheless a strong exact theorem behind the fixed-angle
approximation. It applies to an arbitrary source mesh; it does not require the
source itself to be an extrusion or a small union of sweeps.

Let the carrier plane be

\[
\Pi=\{x\mid n\cdot x=k\}.
\]

Let `c` be one canonical ray direction and `d` one live direction in the same
carrier hemisphere:

\[
n\cdot c\ne0,\qquad n\cdot d\ne0,\qquad
\frac{n\cdot c}{n\cdot d}>0.
\]

For every carrier point `O` in `Pi`, the canonical bake stores the first event
of

\[
r_c(\tau)=O+c\tau
\]

against an arbitrary mesh `G`. Define

\[
w=\frac d{n\cdot d}-\frac c{n\cdot c}
\]

and the affine map

\[
\boxed{
T_{c\rightarrow d}(x)=x+w(n\cdot x-k).
}
\]

Because `n dot w=0`, its linear part

\[
M=I+wn^T
\]

has determinant one and inverse `M^-1=I-wn^T`. It fixes the complete carrier
plane pointwise. More importantly,

\[
\boxed{
T_{c\rightarrow d}(O+c\tau)
=O+d\tau\frac{n\cdot c}{n\cdot d}.
}
\]

The positive scalar factor preserves order along every ray. Consequently the
canonical first hit, primitive owner, miss state, barycentric coordinates, and
attached material record are exactly the live first event of the coherently
deformed mesh

\[
G_d=T_{c\rightarrow d}(G).
\]

If the canonical atlas stores `tau_c`, the complete live solve is

\[
\boxed{
\tau_d=\tau_c\frac{n\cdot c}{n\cdot d}.
}
\]

No source triangle, plane extrapolation, owner interpolation, second sample, or
candidate minimum is involved. The canonical phase is unchanged because the
map fixes `O`. A source normal covector `m_c` transforms exactly as the
following expression when `d` is constant over the chart, in particular for an
orthographic view:

\[
\boxed{
m_d\propto M^{-T}m_c
=m_c-n(w\cdot m_c).
}
\]

This is a dot product, a vector multiply-add, and normalisation after the one
record fetch. Colour and other material attributes remain attached to the same
canonical event.

### 12.1 Perspective stitching changes the normal Jacobian

In a pinhole camera `d` varies with carrier point `O`. Per-ray order is still
preserved, but the per-ray rank-one matrix is not the Jacobian of the complete
stitched image warp away from the carrier.

For a general scaled conjugacy

\[
T_C(O,t)=O+t\,g(O),
\qquad g(O)=\lambda(O)d(O),
\]

recover carrier point and canonical parameter from `x=O+tc` with

\[
P_c=I-\frac{cn^T}{n\cdot c},
\qquad dO=P_c\,dx,
\qquad dt=\frac{n^Tdx}{n\cdot c}.
\]

The exact stitched Jacobian is

\[
\boxed{
J_C=\left[I+t\,Dg(O)\right]P_c
+g(O)\frac{n^T}{n\cdot c}.
}
\]

For unit perspective direction

\[
d(O)=\frac{O-C}{\lVert O-C\rVert},
\qquad
Dd=\frac{I-dd^T}{\lVert O-C\rVert}.
\]

The height-preserving scale used above is

\[
\lambda(O)=\frac{n\cdot c}{n\cdot d(O)},
\qquad
g(O)=\frac{(n\cdot c)d(O)}{n\cdot d(O)}.
\]

Its `Dg` follows by the quotient rule. Exact normals for the stitched
perspective deformation use

\[
\boxed{m_{live}\propto J_C^{-T}m_c.}
\]

Using only the cheaper per-ray inverse transpose is exact at the carrier and
in the orthographic limit; away from those cases it is another approximation.
This distinction must be evaluated before a shader change. It is a fixed
Jacobian calculation, not permission to add neighbouring samples or a normal
reconstruction filter.

The theorem also explains why the arbitrary-mesh evaluator's metre-scale
direction error is not by itself a visual rejection of this representation.
That evaluator compared the reconstructed point to the unchanged `G`; the
theorem renders `G_d`. The correct acceptance test is coherence and projected
appearance of `G_d`, plus a bound on its discrepancy from `G`, not equality of
their world hit points.

## 13. What the conjugacy does and does not solve

The result is exact and one-read, but only inside one carrier chart.

It solves:

- arbitrary source-mesh complexity at the canonical view;
- coherent motion of every source surface between views;
- first-hit ownership and miss preservation for the deformed mesh;
- exact transformed normals and stable attached colour;
- the absence of mixed-owner sheets created by interpolating unrelated depth
  records.

It does not solve:

- equality to the unchanged source mesh between canonical views;
- the pole `n dot d=0` where the live ray is parallel to the carrier;
- raster coverage for a ray which sees cover but never meets that carrier;
- chart-transition error between different canonical directions or carrier
  planes.

A projected-distance formula is not a proof that the pole disappeared. At a
carrier-tangent live direction, any map which fixes that carrier and sends a
transverse canonical ray onto the tangent live ray becomes rank-deficient. It
can collapse the cover onto the carrier while remaining finite numerically.
That is another plane artifact, not a valid complementary chart.

More generally, let a plane-fixing map send `c` to `lambda*d`. Its determinant
is

\[
\boxed{
\det M_\lambda
=\lambda\frac{n\cdot d}{n\cdot c}.
}
\]

At `n dot d=0`, a bounded projected-distance scale makes this determinant zero;
the cover collapses onto the carrier. Keeping the determinant nonzero requires
`lambda` to diverge like `1/(n dot d)`, which is the unbounded height-preserving
stretch. Switching between the two scalar formulas does not constitute a
nonsingular chart atlas.

The required all-view extension must therefore prove all of the following
before implementation:

1. a finite set of complementary carrier/canonical charts covers every promised
   camera direction with a certified denominator lower bound;
2. runtime selects exactly one chart directly, with no traversal or candidate
   testing;
3. the selected carrier supplies every cover pixel without a visible raised
   terrain copy;
4. adjacent charts agree at their boundary or have a measured subpixel
   projected discrepancy at the closest accepted range;
5. the selected chart still costs one coherent atlas record and tiny fixed ALU.

Flattening the plant, forbidding the singular view, or silently relying on a
camera that always intersects one ground carrier fails this contract.

Three differently oriented carrier normals are the minimum algebraic cover of
a hemisphere: two normals always share a perpendicular direction. Three can
keep the selected carrier dot product bounded away from zero, but this only
removes the pole. For a generic asymmetric mesh their three deformed images do
not agree on chart boundaries. Hard selection creates a spatial or temporal
seam; blending costs another record and still blends different owners. A tiny
three-chart pole cover is therefore not yet a visual solution.

## 14. Rejected structural alternatives

Two independent exact alternatives have now failed the low/mid-range gate on
the accepted asset.

### 14.1 Tiny affine/projective sweep union

One finite affine or projective sweep of an arbitrarily detailed 2D mask has an
exact one-read solve. It can share one taper, lean, convergence law, and axial
interval. The accepted Calamagrostis contains hundreds of independently born,
curved, tapering, and terminating structures. Its culm alone has about `3.98
mm` maximum residual from its best straight generator, already above a `0.5 px`
bound at 4 m. Four to six global sweeps cannot preserve the accepted morphology.
This remains a possible certified far LOD, not the near-field answer.

### 14.2 Tiny Pluecker separator tail

Direct line-space point location is exact in principle, but a bounded actual-
asset probe rejects a tiny separator tree. One current atlas-scale 4D cell had
202 distinct first owners plus miss on a sparse `5^4` interior lattice, an
information-only lower bound of eight binary decisions before geometric
separability. After one uniform subdivision in all four coordinates, every one
of the sixteen subcells still exceeded four decisions and the worst still
needed at least six.

The resulting serial dependent reads and boundary-page memory violate the
target architecture. Adaptive refinement could eventually work as an arbitrary
visibility data structure, but it is not the tiny one-read reconstruction being
sought and is parked.

## 15. The view restriction is rejected, not promoted to a contract

The sampled arbitrary-mesh field, tiny sweep union, runtime candidate closure,
and shallow line-space classifier are all parked. The affine-conjugate canonical
field from Section 12 remains a useful theorem and diagnostic, but a single
fixed-elevation carrier is not the production contract. Its pole and its
view-dependent deformation cannot be excused by flattening the plant or by
declaring a camera direction unsupported.

The required visual domain is the complete free-camera exterior domain used by
the scene, including nadir, low oblique, and first-person/grazing inspection.
The source Calamagrostis morphology is retained. An internal representation may
be approximate between baked rays, but it must be judged over that complete
domain and it may not define correctness by avoiding the angle where the
approximation fails.

## 16. 2024 follow-up: centred light-field coordinates and categorical depth

Sannikov's newer public experiment supplies two implementation-critical
corrections which are more relevant than the old flattening workaround:

- the newer arbitrary-model experiment used 16 baked directions and explicitly
  identified stretch as bake-direction/view-direction mismatch;
- he moved the depth-zero plane from the top of the mesh to approximately its
  middle, changing the ray-scale interval from `[0, 1]` to `[-0.5, 0.5]` and
  materially reducing parallax error;
- he states that depth must not be linearly blended and suggests a PCF-like
  interpolation instead;
- the grass experiment used a `128 x 128 x 4` light field for a repeating
  `25 cm x 25 cm` tile, with the explicit goal of obtaining accurate view
  interpolation from as few directions as possible;
- when asked whether the method is limited to top-down views and unsuitable for
  first-person rendering, he rejected that restriction.

Primary sources:

- <https://www.youtube.com/watch?v=aBPwnzUtfkA>
- <https://www.youtube.com/watch?v=b4jpSx5FA7c>

These comments do not prove continuum-exact unchanged-mesh visibility. They do
give the correct cheap approximation contract: centre the ray coordinate,
retain complete direction-conditioned records, and filter categorical hit
events rather than numerically averaging unrelated depths and owners.

Let the botanical band be `y_min <= y <= y_max`, with `H=y_max`, and choose the
minimax central reference plane

\[
y_0=\frac{y_{min}+y_{max}}2.
\]

`H/2` is only the special case `y_min=0`. The eventual generic profile format
must carry `y_min`, `y_max`, or `y_0`; one global Calamagrostis constant is not
part of the contract.

For a live top-plane ray

\[
r(t)=O_H+t d,\qquad (O_H)_y=H,\qquad d_y<0,
\]

its reference-plane phase is

\[
\boxed{
Q=O_H+d\frac{y_0-H}{d_y}.
}
\]

The existing GCRP is indexed by its **top-plane** origin, not by `Q`. After one
baked direction `c` has been selected categorically, its atlas address must be

\[
\boxed{
O_c
=Q-c\frac{y_0-H}{c_y}
=O_H+\left(\frac{d}{d_y}-\frac{c}{c_y}\right)(y_0-H).
}
\]

Sampling the current atlas at `Q.xz` directly is wrong. An equivalent and
cheaper implementation may recenter every slice once while loading/baking, so
the runtime atlas is physically indexed by `Q.xz` and no per-depth-sample
canonical-direction phase correction remains.

For the baked direction `c`, its top-plane record `tau_c` has hit height

\[
y_h=H+\tau_c c_y.
\]

The signed centred scale is

\[
\sigma_c=\frac{y_h-y_0}{c_y},
\]

and the direction-independent record is simply normalized hit height

\[
\boxed{h=y_h/H.}
\]

Reconstruction on the live ray is

\[
\boxed{
\tau_d=\frac{H-y_h}{-d_y},\qquad
P_d=O_H+\tau_d d.
}
\]

The important change is not the last division; it is that the atlas is
addressed by the ray's phase `Q.xz` at the middle plane. Direction mismatch can
then displace a represented point by at most half the botanical height before
the next view record is selected, rather than by the complete top-to-ground
height.

The matching constant-direction normal transport is not optional. With

\[
w=\frac d{d_y}-\frac c{c_y},
\]

the centred affine conjugacy gives

\[
\boxed{m_d\propto m_c-e_y(w\cdot m_c).}
\]

Keeping `m_c` unchanged would leave a view-dependent orientation error even if
the hit point were reconstructed correctly. Section 12.1 remains the stricter
perspective-stitched normal Jacobian; the expression above is exact for one
constant direction and the orthographic limit.

Depth, normal, colour, coverage, and material owner form one categorical event.
They must never be bilinearly averaged across different view owners. A one-tap
PCF-like filter selects one complete record from the fixed spatial/angular
filter footprint using deterministic subpixel coverage; all attributes follow
that record. This is fixed-cost stochastic/categorical filtering, not a
candidate search:

1. compute the ordinary spatial and angular filter weights;
2. use a deterministic well-distributed threshold to choose one footprint
   corner before the texture read;
3. fetch exactly one complete record;
4. reconstruct depth from its hit height and use its attached normal/colour.

The expectation equals the filtered categorical distribution, while no pixel
contains an invented average owner or a depth sheet between two plants. The
threshold must be stable in the ray/ground domain and sufficiently distributed
to avoid a new visible grid. A later temporal pattern is permitted only if the
existing temporal resolve proves it converges without shimmer.

Centred coordinates bound but do not erase angular mismatch. Their horizontal
error is

\[
\lVert\Delta xz\rVert
\leq \max_y|y-y_0|
\left\lVert\frac{d_{xz}}{d_y}-\frac{c_{xz}}{c_y}\right\rVert.
\]

The current 15-degree lowest row is therefore not silently accepted for a
5-degree live view. A 5-degree row is a legitimate fixed-memory refinement if
the grazing visual gate still fails; it is not a substitute for record-safe
sampling or an excuse to clamp an uncovered angle.

### 16.1 Runtime cost gate

For the isolated high-resolution profile this change must reduce, not increase,
the hot path:

- one atlas record instead of four angular records per layer;
- one selected normal and one selected colour record instead of four of each;
- no new binding, pass, dispatch, barrier, candidate list, or loop;
- the same atlas footprint and phase locality;
- a small fixed amount of phase-shift, hash, and categorical-select ALU;
- shorter live ranges because four records and four weighted sums disappear.

The exact horizon remains a separate carrier-coverage problem. Adding a `5`
degree baked elevation can reduce the low-oblique mismatch but does not by
itself prove the exact-horizontal case. The visual implementation may proceed
on the current free-camera scene only after it makes no angle unavailable and
the grazing path is explicitly exercised; it may not relabel a hidden angle as
an accepted limitation.

## 17. Removing the translated-height carrier without mixing charts

The current isolated implementation still constructs a second terrain draw and
projects it as `S+(0,H,0)`. Although its vertices are sourced from unchanged
terrain, its raster coverage is exactly the constant-height duplicate which the
user sees as a hovering sheet when reconstruction collapses toward `H`. It also
cannot be the generic contract: overlapping profiles have distinct botanical
bands and no single `1.176 m` datum represents them.

The cheap replacement is the article's carrier-coordinate operation performed
from one coherent local botanical chart, not a second world surface. Let `T` be
the ray parameter at the available carrier datum and let

\[
q_h(T)=h_T,\qquad \dot q_h=(J^{-1}d)_h.
\]

The live top and centred-reference parameters are then

\[
\boxed{
t_H=T+\frac{H-h_T}{\dot q_h},\qquad
t_0=T+\frac{y_0-h_T}{\dot q_h}.
}
\]

For the isolated rigid profile, `J` is the exact same local ground chart used
to define `h_T`; no separately sampled height may be combined with a raster
point from another chart. In the heightfield specialization

\[
\dot q_h=d_y-g_xd_x-g_zd_z.
\]

The selected record carries `y_h`, so the final world-ray parameter can also be
written directly as

\[
\boxed{t_B=T+\frac{y_h-h_T}{\dot q_h}.}
\]

This is one division and fixed affine coordinate work per profile. It supports
different and overlapping profile heights because `H`, `y_0`, and `y_h` belong
to the selected profile, while all candidates share the real carrier datum.
It adds no raster geometry, no extra terrain pass, no march, and no candidate
search.

The premises must remain explicit. A local affine chart is exact for its plane;
it is not an information-theoretic oracle for every piecewise/nonlinear terrain
surface crossed before `T`. Likewise an exactly horizontal ray with no carrier
intersection cannot be reconstructed from a nonexistent `T`. These are open
carrier-domain questions, not permission to restore a constant raised surface,
hide a camera angle, clamp the denominator, or add brute force.

## 18. Selected-direction orthogonal carrier: removing the live horizon pole

The first no-envelope runtime did not implement the open carrier premise above.
It forced

\[
\dot q_h=\min(d_y-g_xd_x-g_zd_z,-10^{-3})
\]

and then clamped the resulting top parameter to the camera near plane. For a
locally tangent or upward ray this fabricates a large negative displacement and
then collapses every result to the same near sheet. The false sheet is therefore
an algebraic consequence of the clamp, not an inherent limitation of removing
the translated terrain copy.

After selecting one canonical unit direction `c`, use the plane through the
botanical reference point `X_0` whose normal is that same direction:

\[
\Pi_c=\{x\mid c\cdot(x-X_0)=0\}.
\]

For any point `A` on the live ray with unit direction `d`, its intersection with
that carrier is

\[
\boxed{
t_Q=-\frac{c\cdot(A-X_0)}{c\cdot d},\qquad Q=A+t_Qd.
}
\]

The existing centred atlas can be reused without rebaking so long as `c_y` is
nonzero. Its address is the point where the canonical line through `Q` crosses
the stored horizontal reference plane `y=y_0`:

\[
\boxed{
X_c=Q+c\frac{y_0-Q_y}{c_y}.
}
\]

The recentered slice at `X_c.xz` returns the categorical canonical hit height
`y_h`. Its signed canonical coordinate from `Q` is

\[
s_c=\frac{y_h-Q_y}{c_y}.
\]

The live hit parameter measured from `A` is then

\[
\boxed{
t_h=t_Q+\frac{s_c}{c\cdot d}.
}
\]

This is the affine conjugacy with carrier normal `c`. It is exact for the same
coherently deformed complete canonical field as Section 12 and preserves
forward order whenever `c dot d > 0`. Nearest-direction selection bounds that
denominator by the cosine of the direction cell's angular radius. In
particular, an exactly horizontal live direction is no longer a pole: a nearby
five- or fifteen-degree canonical direction still has a large positive dot
product with it. Only the atlas conversion through `y=y_0` uses `c_y`, which is
an offline lattice property rather than the live view denominator.

The inverse-transpose normal is

\[
\boxed{
m_d\propto m_c-c\left[\left(\frac d{c\cdot d}-c\right)\cdot m_c\right].
}
\]

The first categorical experiment also made the angular selector depend on the
2.03 mm phase texel. That spliced distinct coherent fields as
`F_{S(Q)}(Q)` and created the observed grazing streaks. A one-record selector
must instead be phase-independent. The lowest-cost diagnostic is the
maximum-weight/nearest angular corner:

\[
a^*=a_0+\mathbf1[\mu_a\geq 1/2],\qquad
e^*=e_0+\mathbf1[\mu_e\geq 1/2].
\]

Every local ray bundle then evaluates one field `F_{c*(d)}(Q)`. This removes the
spatial splice but cannot make a coarse angular Voronoi boundary continuous;
the lowest five-degree row remains a legitimate support refinement. A
four-record categorical median is the literal PCF fallback, but is not enabled
without an explicit performance decision.

This carrier change adds no texture, atlas byte, binding, pass, dispatch,
barrier, loop, march, or candidate. It replaces the phase hash and live
horizontal-carrier divisions with fixed dot/FMA work and one bounded reciprocal.
The independent `139.5..155 m` camera-radius density fade and hard range cutoff
remain separate camera-locked terms and are not justified by this derivation.

## 19. Rejection of nearest angular cones and a moving carrier

The first visual experiment based on Section 18 is rejected. It produced
camera-front wedges which widen toward the foreground, increasingly stretched
botanical records with range, and a far-oblique field which visibly reads as
the wrong view of the plant. These are direct signatures of two mathematical
mistakes, not rendering noise.

First, phase-independent nearest angular selection partitions pinhole rays into
angular Voronoi cones. In one angular coordinate, with baked directions
`theta_i`, the switch occurs on

\[
\theta(d)=\frac{\theta_i+\theta_{i+1}}2.
\]

That locus is a ray/plane through the camera. On a surface at range `r`, one
angular cell has world width

\[
\boxed{w(r)\simeq r\,\Delta\theta.}
\]

If neighbouring canonical first-hit fields disagree, the disagreement is
therefore not a small local pop: it fills a camera-centred triangular sector
which grows linearly with distance. The 2.03 mm stochastic selector spliced the
same fields into fine grazing streaks; nearest selection merely enlarged those
splices into the observed wedges. A one-record nearest/MAP angular selector is
therefore rejected for the accepted visual contract.

Second, the experiment instantiated `X_0` separately for every query datum `A`
as its vertical projection onto `y=y_0`. That invalidates the fixed-carrier
premise of Section 18. A necessary line-space invariance is that moving the
query origin along the same oriented ray must not change the atlas address. Let

\[
A'=A+\lambda d,\qquad X_0(A)=(A_x,y_0,A_z).
\]

For the moving carrier, the horizontal-reference address changes by

\[
\frac{dX_c}{d\lambda}
=\left(1-\frac{c_yd_y}{c\cdot d}\right)
\left(d-c\frac{d_y}{c_y}\right),
\]

which is nonzero in general. Thus two origins on the exact same ray can fetch
different canonical records. There is no single reconstructed surface behind
those per-query conjugacies, and the fixed-carrier inverse-transpose normal is
not their Jacobian. By contrast, a genuinely fixed `X_0` gives
`t_Q'=t_Q-lambda`, keeps `Q` and `X_c` invariant, and preserves the world hit.

The visual wording that “the wrong thing is being stretched” is mathematically
accurate: within each nearest-direction cone, the method displays an affine-
conjugate image of one neighbouring canonical view field, not the unchanged
mesh's live first-hit field. Increasing carrier algebra cannot repair that
missing angular visibility information.

The next representation must satisfy all of these before another shader edit:

1. **oriented-line invariance:** atlas address and world hit are unchanged when
   the query origin moves along one ray;
2. **pinhole integrability:** neighbouring pixel rays sample one coherent
   surface/light field rather than independent per-ray deformations;
3. **continuous angular reconstruction:** no camera-centred nearest-direction
   cones, hashed world-phase splices, or arithmetic averages of unrelated hit
   owners;
4. **bounded grazing coordinates:** direction and phase charts may not contain
   a live `1/d_y` pole;
5. **fixed low cost:** O(1), no march, loop, live triangle/candidate closure,
   extra terrain shell, or memory blow-up.

No runtime correction is authorized until a pure-math representation meets
those conditions. The failed implementation is retained only as evidence.

## 20. Executable coherence contract

The author comment in Section 11 is evidence about a missing coordinate, not an
algorithm. The following is the contract a one-record reconstruction must
actually implement.

### 20.1 A per-query carrier offset is not a carrier

Keep one canonical direction `c` fixed for this derivation and write the offset
of the alleged orthogonal carrier as

\[
k(A,d)=c\cdot X_0(A,d).
\]

The point selected by Section 18 is

\[
\boxed{
Q(A,d)=A+d\frac{k(A,d)-c\cdot A}{c\cdot d}.
}
\]

Replace the ray datum by another point on the exact same oriented line,
`A_lambda=A+lambda*d`. Direct subtraction gives

\[
\boxed{
Q(A_\lambda,d)-Q(A,d)
=d\frac{k(A_\lambda,d)-k(A,d)}{c\cdot d}.
}
\]

Therefore `Q`, its atlas address, and the reconstructed world hit are invariant
to the arbitrary choice of ray datum if and only if `k` is constant along that
oriented line. This is the first mandatory property test.

Line invariance alone is not enough to recover the affine theorem. Let a
pinhole bundle be parameterised by `(u,v)`. Since

\[
c\cdot Q(u,v)=k(u,v),
\]

differentiation gives

\[
c\cdot Q_u=k_u,\qquad c\cdot Q_v=k_v.
\]

For `c` to be the normal of one carrier plane, both tangent derivatives must be
annihilated by `c`. Hence

\[
\boxed{k_u=k_v=0.}
\]

On every connected carrier chart, `k` is one constant. This is necessary and
sufficient for all `Q(u,v)` to lie on the same plane with normal `c`. Allowing
`X_0` to vary while keeping only its projection `c dot X_0` fixed is harmless;
allowing that scalar projection to vary is not.

The rejected experiment used

\[
X_0(A)=(A_x,y_0,A_z).
\]

For this choice,

\[
\frac{dk}{d\lambda}
=c_xd_x+c_zd_z
=c\cdot d-c_yd_y,
\]

which is nonzero for a generic ray. The resulting reference-plane address has
the compact derivative

\[
\boxed{
\frac{dX_c}{d\lambda}
=\left(1-\frac{c_yd_y}{c\cdot d}\right)
\left(d-c\frac{d_y}{c_y}\right),
}
\]

also nonzero in general.

A concrete counterexample needs no renderer. In two dimensions choose
`c=(1,1)/sqrt(2)`, `d=(1,0)`, `y_0=0`, and let the canonical atlas alternate hit
and miss every half-period in `x`. With `A=(A_x,A_y)` and
`X_0(A)=(A_x,0)`, Section 18 gives

\[
Q=(A_x-A_y,A_y),\qquad X_c=(A_x-2A_y,0).
\]

Replacing `A` by `A+(T/2)d` therefore moves `X_c` by `T/2`, so the exact
same oriented ray changes from hit to miss. No static or coherently deformed
surface can have that property.

### 20.2 Sufficient construction for one coherent pinhole field

Let the carrier be genuinely fixed:

\[
\Pi=\{Q\mid n\cdot Q=k\},
\]

and let one canonical atlas field define the canonical surface graph

\[
P_c(Q)=Q+c\,\sigma(Q),\qquad n\cdot c\ne0.
\]

For a camera centre `C`, set

\[
b=k-n\cdot C.
\]

Assume `b` is nonzero and has one sign on the declared camera domain. The
unique height-preserving pinhole conjugate of that complete field is

\[
\boxed{
F_C(Q)
=Q+(Q-C)\frac{(n\cdot c)\sigma(Q)}{k-n\cdot C}.
}
\]

`F_C(Q)` lies on the live ray through `C` and `Q`, uses one value
`sigma(Q)`, and is one explicit surface map shared by all neighbouring pixels.
It is not a collection of unrelated per-ray affine maps.

Choose a fixed carrier basis `E=[e_1,e_2]` and write `Q(u)=Q_0+Eu`. With

\[
\gamma=\frac{n\cdot c}{k-n\cdot C},\qquad a(u)=\gamma\sigma(u),
\]

the exact stitched differential is

\[
\boxed{
D_uF_C=(1+a)E+(Q-C)\,\gamma(\nabla_u\sigma)^T.
}
\]

This differential, not the constant-direction inverse transpose, owns the
pinhole normal. The mapping is a regular visible surface wherever the carrier
projection is regular and

\[
\boxed{1+a(u)\ne0.}
\]

It varies continuously with the camera while the same `Pi`, `c`, and atlas
field remain selected. No query datum other than the oriented ray may alter its
source address.

This gives direct mathematical tests:

1. replacing `A` by `A+lambda*d` leaves `Q`, the atlas address, and `F_C(Q)`
   bit-identical in the real-number reference;
2. finite differences of neighbouring `F_C` values agree with `D_uF_C`;
3. the two columns of `D_uF_C` retain rank two over every hit record;
4. a small camera perturbation changes `F_C` continuously while the chart is
   unchanged;
5. no denominator is clamped and no miss is converted into a hit.

### 20.3 Necessary and sufficient global form

A fixed-cost one-record method is globally coherent over neighbouring rays and
camera motion exactly when it can be written using:

- one camera-independent two-dimensional label manifold `U` and record field
  `R:U -> records` for each coherent surface chart;
- a line-origin-invariant query map `chi_C` from each live oriented ray to one
  label `u`;
- a single-valued map `F_C(u)` which lies on that live ray, is continuous in
  `(C,u)` away from true visibility boundaries, and has rank two in `u`;
- on every overlap of two charts, a camera-independent transition
  `psi_ij` satisfying

\[
\boxed{
F_{C,i}(u)=F_{C,j}(\psi_{ij}(u))
}
\]

  with the same hit/miss state, material event, and orientation.

These conditions are sufficient because `F_C(U)` is then the required
piecewise surface. They are necessary because a coherent surface itself
supplies the labels, incidence map, differential, and overlap transitions.

Nearest-direction selection does not meet this contract for independent
arbitrary-mesh first-hit slices. A per-pixel selector makes screen-space cones;
a per-frame selector removes the spatial cone but still pops under camera
motion. Either is valid only if the two selected fields obey the overlap
identity above. The accepted Calamagrostis slices do not: they contain different
first owners and misses.

### 20.4 The minimum all-direction data contract

The author comment implies a strict fork:

1. a three-dimensional atlas is exact only after imposing a structural model
   which analytically removes one direction coordinate, such as a parallel
   extrusion; or
2. an arbitrary finite mesh needs its complete four-dimensional exterior
   first-hit field: two coordinates for an oriented line's transverse position
   and two for its direction.

No carrier algebra turns case 2 into case 1. The cheap all-direction
representation is therefore a direct four-dimensional oriented-line field,
not another per-ray deformation of a neighbouring angular slice.

A bounded coordinate system is obtained with six dominant-axis charts. For a
unit direction `d`, choose the signed index `j` of its largest-magnitude
component. In that chart, intersect the oriented line with the fixed plane
`x_j=k_j`:

\[
\boxed{
Q=A+d\frac{k_j-A_j}{d_j}.
}
\]

If `a,b` are the other two axes, use line coordinates

\[
\boxed{
\ell_j=(Q_a,Q_b,d_a/d_j,d_b/d_j).
}
\]

They are unchanged by `A -> A+lambda*d`, and dominant-axis selection guarantees

\[
\boxed{|d_j|\ge 1/\sqrt{3}.}
\]

Thus no live horizon clamp exists. The six charts describe the same oriented
lines; their overlap maps are analytic, and both sides must bake the identical
first event. A record storing the hit parameter `s_h` reconstructs

\[
\boxed{
B=Q+s_h\left(e_j+\frac{d_a}{d_j}e_a+\frac{d_b}{d_j}e_b\right).
}
\]

This is still O(1): one direct four-dimensional record lookup packed into an
ordinary texture layout, followed by fixed coordinate arithmetic. It introduces
no ray march, loop, candidate set, live triangle, or depth-complexity-dependent
work. Finite sampling still needs an explicit offline error and byte budget;
increasing it without that bound is not authorised.

This exterior field assumes a first event from outside a bounded botanical
volume. If the camera origin may lie inside a finite profile, the next event
depends additionally on the origin's signed phase along the oriented line. That
is a fifth coordinate or an explicitly bounded successor record. It cannot be
recovered from an exterior four-dimensional record by changing a distance
formula.

The cheapest valid next experiment is therefore unambiguous:

- for a coherence-only diagnostic, use one genuinely fixed carrier and one
  canonical field with the `F_C` construction in Section 20.2 over its certified
  domain;
- for the complete free-camera arbitrary-mesh contract, retain the current
  one-read architecture but make the data a fixed-chart four-dimensional line
  field as above, with exact overlap equality and a fixed total-record budget;
- do not implement another moving carrier, per-ray angular selector, phase hash,
  depth average, or candidate closure.

## 21. Periodic all-angle synthesis: the one-event obstruction

The dominant-axis construction gives bounded coordinates, but it does not
reduce the information carried by the field. Two further distinctions close
the remaining apparent shortcuts.

First, an exterior first-hit query for a bounded three-dimensional population
is a function of an oriented line and is therefore four-dimensional. A
successor query from an origin inside an infinitely repeated population is a
function of a **pointed** oriented line. If `s_0` is the origin's signed phase
on line `L`, the required event is

\[
E^+(L,s_0)=\inf\{s>s_0\mid r_L(s)\in G\}.
\]

Changing `s_0` can change the owner while leaving `L` fixed. It is a fifth
coordinate unless one record carries an ordered deep-ray representation from
which the successor can be selected. At an exactly horizontal direction an
infinite periodic field has no exterior top entry; periodicity does not remove
`s_0`. An unbounded line can contain unboundedly many repeated events, so an
exact tiny deep record exists only after declaring a finite ray horizon and
proving a tiny maximum event count on that horizon.

Second, explicit band-limiting does not turn a discontinuous first-hit field
into one opaque event. Let a line-space reconstruction footprint contain
complete events `e_r` with filter weights `p_r`. The exact filtered object is
the event measure

\[
\boxed{\mu_L=\sum_r p_r\,\delta_{e_r}.}
\]

When two distinct owners or depths have nonzero weight, no single complete
record equals `mu_L`. A one-event decoder has only three possibilities:

1. average depths or owner attributes, creating a surface which belongs to no
   source event;
2. choose one event deterministically, producing a categorical switch surface
   such as the rejected camera-centred cones or phase splices;
3. choose one event stochastically, which is unbiased only in expectation and
   presents single-frame noise or temporal shimmer.

A median or another quantile is still case 2. In particular it deletes a sparse
panicle whenever that event owns less than the chosen quantile mass. A
premultiplied colour/opacity average represents filtered radiance, but no longer
supplies the categorical depth, normal, material owner, or opaque overlap
required by the geometry election. This is the precise reason a PCF-like
filter can antialias visibility but cannot make one filtered depth record an
exact geometric event.

### 21.1 Candidate reductions

The remaining proposed reductions consequently have strict scopes:

- A three-dimensional sampler is exact only when a genuine symmetry removes
  one line coordinate. Parallel extrusion does this. The accepted
  Calamagrostis does not: its terminating leaves, curved culms, and branching
  panicles have independent births, deaths, and occlusions. A recursive
  constructor is not an intersection symmetry and still requires traversal or
  a precomputed visibility field.
- A generalized displacement map is one coherent surface sheet and can satisfy
  line-origin invariance and pinhole integrability. It cannot expose a surface
  hidden behind that sheet. Exact disocclusion therefore requires a layered
  displacement/deep-image representation whose layer count is at least the
  relevant visibility depth complexity. Cost and storage are linear in that
  count.
- Fourier, tensor, vector-quantised, or neural decoders can compress a sampled
  four-dimensional field, but do not lower its dimension or remove the event
  measure above. They are admissible only after an offline actual-asset test
  certifies a fixed decode cost, byte bound, hit/miss error, geometric error,
  and boundary behaviour. No such small-rank certificate currently exists.
- A finite deep occupancy or successor mask replaces a march with a fixed
  broadword operation, but exact depth, normal, colour, and material still need
  an event attached to every retained crossing. For an infinite horizontal
  periodic ray its required extent is unbounded; for a finite horizon its
  required bit/event count must be measured rather than assumed.

These are representation statements, not invitations to test another shader
heuristic.

### 21.2 Exact overlap has one cheap form

For separately baked species fields with first-hit parameters `t_i`, the exact
opaque union is

\[
\boxed{t_{\cup}=\min_i t_i.}
\]

The minimum is fixed-cost only for a fixed small number of simultaneously
queried fields, and it requires one complete result per field. There is no
sublinear exact operator which recovers that minimum without either reading
those results or having precomputed their union. Runtime enable/disable masks
also turn the query into the successor problem; supporting every subset by
precomposition requires up to `2^S` states.

The copy-count-compatible solution is therefore to bake one **marked composite
population** per ecologically valid mixture: grass blades, culms, panicles,
moss, and other simultaneous cover participate in one visibility field, and
the winning event carries its species/material mark. Hardware repetition then
still renders one or arbitrarily many population copies for the same query
cost. A finite catalogue of authored mixture states may be selected before the
lookup; arbitrary live Boolean composition is outside the one-field contract.

### 21.3 Strongest surviving representation and its gate

The only representation not rejected by the preceding proofs is a
**precomposed marked deep four-dimensional line field**:

1. use the six dominant-axis charts of Section 20.4, with exact analytic overlap
   transitions and no direction pole;
2. bake the complete marked composite population, not independent species
   depths which are blended later;
3. store either the exact selected event in an analytically certified
   visibility cell, or a fixed `K`-event filtered/deep record representing the
   line-space event measure and camera-inside successors over a declared finite
   horizon;
4. address the record directly and resolve its fixed record with a bounded
   operator. No traversal, source primitive, variable list, or per-copy work is
   permitted.

This is a data model, not yet an implementation choice. It is exact only when
the line-space cell partition and deep event sequence are complete. It is an
honest band-limited model only when the footprint kernel, retained event mass,
depth error, silhouette error, and discarded transmittance are explicit and
certified. Dominant-axis overlap records must agree on hit/miss, world hit,
orientation, and mark. Moving the query datum along an exterior line must not
change its address or hit; moving the physical inside origin may change only
the successor selection within the same deep record. Every retained event
sheet must also carry one source-surface label and obey the angular
correspondence cocycle and the camera-independent chart-overlap identity;
otherwise interpolation of the deep record is not one pinhole-integrable
field.

The required offline feasibility test is therefore the minimum filtered event
support/deep count `K` and the minimum fixed-size codec over the actual
Calamagrostis population at the closest accepted screen footprint and complete
direction domain. It must also include exact horizontal rays and the declared
camera-inside horizon. If `K`, the fixed decode, or the total bytes do not fit
the low/mid-range budget, then no currently proposed exact or honestly
band-limited representation satisfies all constraints. The consequence is to
relax memory/read count, camera-inside/horizon, or near-field fidelity
explicitly—not to return to nearest cones, moving carriers, depth blends,
runtime candidates, or marching.

## 22. Correspondence-aware angular flow audit

**Decision:** correspondence flow is rejected as a distinct fixed-read
representation for the accepted Calamagrostis field. Exact forward flow is
cheap, but it is already completely determined by the stored hit depth. The
missing operation is inverse visibility correspondence. Storing that inverse
and its disocclusion branches reproduces the complete four-dimensional
first-hit field; not storing it requires an inverse solve, splat, candidate
election, or approximation already rejected by the visual evidence.

### 22.1 Exact line-invariant flow

Work in one dominant-axis chart from Section 20.4. Normalise the line direction
so its dominant component is one:

\[
v(r)=e_j+r_a e_a+r_b e_b,
\qquad r=(r_a,r_b).
\]

For a fixed surface point `P`, let

\[
\lambda(P)=P_j-k_j.
\]

The carrier phase of the oriented line through `P` with slope `r` is

\[
\boxed{
q_r(P)=
\begin{pmatrix}P_a\\P_b\end{pmatrix}
-r\lambda(P).
}
\]

Hence the exact same-point correspondence between two directions is

\[
\boxed{
q_{r_1}(P)=q_{r_0}(P)+(r_0-r_1)\lambda(P).
}
\]

Both `q_r` and `r` are unchanged when the query datum moves along its oriented
line, so this flow satisfies the required origin invariance. It also exposes an
important redundancy: a two-component angular flow vector contains no new
geometry once the hit parameter `lambda` is known.

Let the first-hit field at `r_0` be `D_0(q)`. Its exact forward warp toward a
live direction `r` is

\[
\boxed{
M_{0\rightarrow r}(q_0)
=q_0+(r_0-r)D_0(q_0).
}
\]

This is the correspondence identity evaluated by the earlier phase-reprojection
experiment. Adding a separately baked forward-flow texture cannot make it more
exact; it merely stores the same scalar depth twice.

### 22.2 Pull reconstruction requires the missing inverse

A live pixel supplies target phase `q`, not source phase `q_0`. Pull
reconstruction therefore needs

\[
\boxed{q=M_{0\rightarrow r}(q_0)}
\]

solved for `q_0`. Its Jacobian is

\[
DM_{0\rightarrow r}
=I+(r_0-r)(\nabla D_0)^T,
\]

and the matrix-determinant lemma gives

\[
\boxed{
\det DM_{0\rightarrow r}
=1+\nabla D_0\cdot(r_0-r).
}
\]

On one retained planar surface this map is affine and has a closed-form inverse
while that determinant remains nonzero. This is exactly the already-evaluated
one-plane theorem. At silhouettes the determinant reaches zero; at occlusions
the map can be globally many-to-one even when its local determinant is nonzero;
at disocclusions the target phase has no source preimage. An open, overlapping
grass mesh necessarily contains all three cases.

Consequently a one-read source flow has only three possible consumers:

1. iterate or search for `q_0`;
2. forward-splat source events and perform a depth election; or
3. fetch a precomputed target-to-source inverse map.

The first two violate the fixed cheap architecture. The third inverse map is a
function of target phase and both live direction coordinates. It is therefore
four-dimensional. If the inverse record also carries the resulting event, it
is the direct four-dimensional first-hit atlas. If it carries only a pointer or
source coordinate, it adds a second read and a second four-dimensional field
without reducing the first-hit information.

### 22.3 Pinhole integrability and angular cocycles

The same conclusion follows without referring to a sampling lattice. Let
`P(u)` parameterise one static source surface and define its projection into the
dominant-axis carrier by

\[
G_r(u)=P_{ab}(u)-r\,[P_j(u)-k_j].
\]

On a visible branch the inverse correspondence is `u=Psi(q,r)` and must obey

\[
\boxed{q=G_r(\Psi(q,r)).}
\]

Where `D_uG_r` is invertible, differentiation gives

\[
\boxed{
D_q\Psi=(D_uG_r)^{-1},
\qquad
\frac{\partial\Psi}{\partial r_m}
=(D_uG_r)^{-1}e_m\lambda(\Psi).
}
\]

These are the pinhole-integrability equations for correspondence flow. Exact
pairwise flows derived from one `P(u)` satisfy, on their common visible domain,

\[
\boxed{
\Phi_{0\rightarrow2}
=\Phi_{1\rightarrow2}\circ\Phi_{0\rightarrow1}.
}
\]

Composition around an angular cell returns the same surface label. Independently
quantised edge flows generally violate this cocycle and produce camera-motion
swimming even before visibility changes are considered.

Across dominant-axis line charts, the additional overlap requirement is

\[
\boxed{
\Psi_j(\ell_j)
=\Psi_k(T_{jk}\ell_j),
}
\]

where `T_jk` is the analytic coordinate change for the same oriented line. The
hit point, signed event order, coverage, material, and orientation must match.
Chart-local optical flows with different categorical owners do not satisfy this
merely because both produce finite depths.

### 22.4 Exact disocclusion is the first-owner field

For one direction cell, same-surface flow is only a partial map. The exact
query also needs a categorical branch function

\[
\mathcal O(q,r)=\text{first visible surface label at }(q,r).
\]

Its discontinuities are the silhouettes and depth-order boundaries of the
source geometry. A surface hidden at all sampled corner directions can become
visible inside the direction cell, so endpoint flow plus endpoint masks is not
a conservative branch set. Exact categorical disocclusion data is therefore
not a binary hole mask; it is the complete piecewise-constant owner field in
four-dimensional line space.

Once `mathcal O(q,r)` is known, the corresponding surface chart can evaluate
the exact hit. But directly storing `mathcal O` and the chart payload is the
four-dimensional first-hit atlas. Analytically selecting it is line-space point
location. The actual-asset separator probe already rejected a tiny classifier:
one current-scale cell contained at least 202 owners plus miss on a sparse
`5^4` sample, and after one uniform subdivision every subcell still required
more than four decisions, with a worst lower bound of six.

The actual phase-reprojection experiment reaches the same conclusion from a
different direction. Among 7,637 rays for which the predictor returned a hit,
the reprojected categorical record had a different triangle owner in all 7,637
cases and 1,992 became misses. Adjacent triangles could in principle be grouped
into authored continuous surface charts, so triangle-id disagreement alone is
not an impossibility proof. The hit-to-miss changes are genuine disocclusions,
and no authored grouping removes the need for `mathcal O`.

### 22.5 Fixed-read and memory bound

The corrected five-ring plus vertical direction lattice has 81 direction nodes.
With 16 azimuth sectors, its natural triangular angular complex contains

\[
16+4(2\cdot16)=144
\]

angular triangles. At `256 x 256` phase resolution, the existing eight-byte
complete events occupy `43,133,472` bytes including gutters.

A literal correspondence layout needs two two-component flows from one angular
triangle vertex to the other two. Even at two signed 16-bit components per flow,
that is eight bytes per phase sample:

\[
144\cdot256^2\cdot8
=75,497,472\ \text{bytes}
\]

for flow alone, or approximately `113 MiB` together with the event field before
boundary labels, disocclusion payloads, gutters, colour, or normals. Even one
four-byte flow vector per triangle produces about `77 MiB` together with the
events. Both exceed the `51,121,152`-byte resident-profile cap.

This explicit-flow count is not claimed as an information-theoretic lower
bound, because forward flow can be regenerated from depth. The storage which
cannot be removed is categorical inverse visibility. The measured corner-owner
lower bound covers `3,145,728` current four-dimensional cells:

- `1,908,893` cells exceed four owners;
- `772,572` exceed eight;
- `144,225` exceed twelve.

An explicit per-cell branch-list representation therefore needs at least

\[
5(1,908,893)+4(772,572)+4(144,225)
=13,211,653
\]

owner incidences even while ignoring every cell with at most four owners and
every live-only interior owner. At four bytes per incidence that is
`52,846,612` bytes for ids alone, already above the resident cap before any
flow, event, index, or surface data. Cross-cell page sharing could reduce this
particular layout, but no measured sharing bound exists, and using a decision
structure to recover the branch reintroduces the already-rejected point-location
path.

### 22.6 Conclusion

Correspondence flow is useful offline for explaining or compressing stable
same-surface regions. It is not the missing cheap angular reconstruction:

- forward correspondence is already the stored depth under a coordinate
  change;
- inverse correspondence is the four-dimensional live first-hit map;
- exact disocclusion is the four-dimensional categorical owner field;
- enforcing pinhole and chart coherence requires a common surface label and
  cocycle, not independently filtered depth images;
- the actual field's branch complexity defeats a tiny boundary record, while a
  literal flow/branch layout exceeds the fixed memory cap.

This track is rejected unless a future offline artifact demonstrates, on the
accepted mesh, a compressed inverse-visibility field which simultaneously fits
the existing resident cap, uses a fixed one/few direct reads, preserves the
integrability and chart-overlap identities above, and clears the existing
hit/miss and position-tail measurements. No shader experiment follows from the
present correspondence proposal.

## 23. Owner-conditioned codebook audit on the actual v4 field

The preceding memory bound leaves one legitimate question: perhaps the complete
categorical first-hit records have enough exact or near-exact repetition that an
offline codebook can store them much more cheaply. This is a compression audit,
not a substitute contract. It cannot recover an omitted angular dimension; it
only asks how cheaply an already-complete sampled field can be retained.

The executable audit is
`tools/groundcover-bake/analyze-line-field-codebook.ts`. It is hash-bound to the
actual Calamagrostis v4 source:

```text
src/assets/groundcover/calamagrostis-canescens.gcrp
sha256 2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c
118,663,280 bytes
```

The field has `4,260,096` stored texels, `1,965,536` hits, `2,294,560`
misses, and `762,368` exact packed owner tokens. Of those owners, `596,936`
occur once and the most frequent occurs only `279` times. Coverage is binary;
owner-versus-coverage disagreement is zero.

### 23.1 Construction and preserved invariants

Every nonlinear cluster key contains the **exact packed owner token** plus a
quantised cell of `(depth, octNormalX, octNormalY)`. A codeword centroid is
formed only within that key. Consequently, for every tested configuration:

- hit/miss agreement is exactly `1.0`;
- packed owner/category agreement is exactly `1.0`;
- unrelated owners sharing a codeword is exactly zero;
- owner and hit/miss discontinuities cannot move;
- runtime work is one direct `r32uint` index read followed by one direct
  dependent codeword read, with no search, candidate list, loop, or march.

The practical fixed layout is four bytes of index per stored texel and a
12-byte storage-buffer codeword containing owner plus the eight-byte event.
The uncompressed complete baseline is the same 12-byte record inline, or
`51,121,152` bytes. A 16-byte texture codeword was evaluated separately because
its alignment materially weakens the result. The ideal 21-bit packed index is
reported only as a lower-bound variant: records crossing a word boundary need
another index read, so it is not the fixed index-plus-record path.

### 23.2 Measured result

| configuration | codewords incl. miss | fixed buffer bytes | reduction | depth p95 / max | normal p95 / max |
|---|---:|---:|---:|---:|---:|
| `d10-n6` | 1,602,874 | 36,274,872 | 29.04% | 1.179 / 3.052 mm | 0.161 / 3.359 deg |
| `d12-n8` | 1,841,999 | 39,144,372 | 23.43% | 0.0438 / 0.694 mm | 0.00411 / 0.756 deg |
| `d14-n10` | 1,907,115 | 39,925,764 | 21.90% | 0 / 0.139 mm | 0 / 0.185 deg |
| exact `d16-n16` | 1,929,868 | 40,198,800 | 21.37% | 0 / 0 | 0 / 0 |

The apparently useful exact reduction is mostly empty-space coding. Collapsing
all `2,294,560` misses to one record removes approximately `27.5 MB`, but the
four-byte index field then costs `17.0 MB`. Among hit records, exact deduplication
reduces `1,965,536` records to `1,929,867`: only `35,669` records, or `1.81%`.
The near-exact `d12-n8` result reduces the hit codewords by `123,538`, or
`6.28%`, at the errors shown above. It saves only `1,054,428` additional bytes
over the exact fixed-buffer codebook.

The texture-codeword layouts are weaker: `d12-n8` occupies `46,512,368` bytes
(`9.02%` reduction), while the exact layout occupies `47,918,272` bytes
(`6.27%` reduction). The exact codebook has `1,894,767` hit codewords used only
once. Only `45.53%` of spatial neighbour edges and `34.13%` of angular neighbour
edges reuse a codeword, so the dependent second read cannot be assumed to have
good locality.

Almost every interior hit (`1,931,799` of `1,935,453`) lies on a four-neighbour
exact-owner boundary, and every interior hit changes exact owner against at
least one adjacent angular sample. This is expected for triangle/copy-level
ownership at the present tessellation and means owner-boundary-versus-interior
error plots are not a useful discriminator here. The categorical result is
stronger and simpler: ownership is preserved exactly by construction.

### 23.3 Decision

Owner-conditioned codebooks are rejected as the central representation answer:

1. they do not supply the missing all-direction categorical field;
2. exact hit-record repetition is only `1.81%` on the accepted asset;
3. the strongest near-exact fixed path saves `23.43%` overall, but only about
   `1.05 MB` beyond exact coding, while adding a dependent low-locality read;
4. compared with the current geometry-only eight-byte event field, even the
   compressed complete record remains larger; its valid comparison is only
   against an event-plus-owner complete field;
5. adding new direction rows changes both the records and ownership distribution,
   so no compression ratio may be extrapolated without rerunning the audit.

The technique remains admissible as a final offline packing pass **after** an
all-direction representation independently satisfies the mathematical
contract, but it does not change which representation is required. The complete
machine-readable metrics, including elevation rows, worst slices, owner-frequency
classes, and `16 x 16` phase error maps, are emitted at:

```text
data/work/groundcover-line-field-codebook/
  2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c/
  metrics.json
```

The deterministic metrics file has SHA-256
`2307682534546f37bff29065694e3726c7c04e3a7d9fe8f0fc281000b6c9f1a0`.

## 24. Band-limited event-measure compression gate

Triangle ownership is unnecessarily strict for a genuinely band-limited
carrier, but replacing it requires an honest measure rather than calling an
average an exact surface. For one spatial filter footprint, the tested record is

\[
\boxed{
\mathcal E=
\left(C,\;C\,\mathbb E[u],\;C\,\mathbb E[u^2],\;
C\,\mathbb E[\mathrm{RGB}],\;C\,\mathbb E[n]\right),
}
\]

where `u` is the slice-normalised ray parameter. These moments are linear under
prefiltering. If a consumer insists on one event, the only moment-matched
reduction is

\[
\hat u=\frac{C\mathbb E[u]}{C},\qquad
\sigma_t=(t_{max}-t_{min})
\sqrt{\frac{C\mathbb E[u^2]}{C}-\hat u^2}.
\]

`sigma_t` is unresolved line-depth variation, not blade thickness and not an
exact owner confidence interval.

The strongest filterable half-density layout uses, for each direction, an
index-two periodic lattice with basis

\[
B=[2v,w],\qquad |\det(v,w)|=1,
\]

where the small primitive integer vector `v` follows the projected ray. Its
`128 x 256` torus has a closed-form affine address and retains full bandwidth
across the projected ray. A concrete 16-byte block uses three filterable reads
for half-float scalar moments, premultiplied colour, and the linear normal
moment. With gutters it occupies `34,344,960` bytes for 64 directions and an
estimated `43,467,840` bytes for the corrected 81-direction lattice, below the
current `51,121,152`-byte geometry-plus-colour budget.

The storage gate passes but the geometry gate fails on the actual Calamagrostis
field. Across `1,935,453` interior truth hits, the index-two carrier has:

- representative ray-depth error p50/p95/p99 of
  `0.00151 / 0.434 / 1.151 m`;
- unresolved ray-depth sigma p50/p95/p99 of
  `0.0225 / 0.722 / 1.362 m`;
- representative height error p95 `0.173 m`;
- representative normal error p50/p95 `8.54 / 53.42 degrees`;
- coverage IoU `0.9458` and premultiplied-colour RMS `0.08915`.

The long tail is not packing noise: it is the multimodal first-hit distribution
inside even a two-sample ray-aligned footprint. A nonlinear conditional `2 x 2`
record is worse because interpolation no longer preserves the measure. Square
`2 x 2` and `4 x 4` filters further increase the p95 depth error to `0.866 m`
and `1.094 m`.

A 51,204-entry VQ reduces the projected 81-direction footprint to about
`6.25 MB`, but its categorical index is nearest-only. Filtered decoding would
require four indices plus four dependent records, not the accepted one/few-read
path. Nearest decoding still has `0.343 m` p95 representative-depth error,
`0.363 m` p95 unresolved sigma, `62.9 degrees` p95 representative-normal error,
and `0.8988` coverage IoU.

Therefore the event-measure codec is rejected as the crisp geometric carrier.
It may remain useful for a deliberately radiance-like distant appearance, but
it cannot fund missing angular rows for the accepted near-field first-hit
reconstruction by collapsing its multimodal events to one depth or normal.
The reproducible metrics and numbered QA are rooted at artifact
`1856ac68e85012ed36538c593c1c02e6b0023ea02281b0f5b5d7582126d6208c`.

## 25. Literal PCF depth selection does not restore correspondence

Let the four spatially filtered angular-corner events be `(h_i,c_i,w_i)`, with
vertical drop `h_i`, coverage `c_i`, and bilinear angular coefficient `w_i`.
The tested PCF-like interpretations form masses `m_i=c_iw_i`, sort by `h_i`,
and select either the front quartile or weighted median of that four-event
measure. A shader could express this as a fixed compare/select network; no
runtime loop or search is needed.

This operation cannot alter the coverage support

\[
 C=\sum_i m_i,
\]

so every false hit or miss caused by angular correspondence remains. It also
selects a depth from one corner without proving that the selected owner covers
the live line. The actual-asset result confirms the algebraic limitation:
both quantiles retain `65.68%` hit agreement and `96.25%` recall. Front
quartile changes world p50/p95/p99 error from the projected-relift baseline
`0.924/4.137/7.060 m` to `0.874/4.769/7.588 m`; weighted median produces
`0.928/4.643/7.542 m`.

Therefore a categorical scalar depth quantile is not the missing angular
correspondence operator. Four filtered reads plus a sorting network are
rejected before shader integration. The result is an LAAS interpretation gate,
not a claim about the unpublished Sannikov filter. Full provenance is Experiment
006 in the central source/experiment ledger.

## 26. Exact four-view hit-height carry does not preserve an event

The strongest form of the cheap four-view proposal was tested separately from
the old projected-path relift.  With real top-plane live origin `O_H`, middle-
plane phase `Q`, and canonical direction `c_i`, use

\[
Q=O_H+d\frac{y_0-H}{d_y},\qquad
O_i=Q-c_i\frac{y_0-H}{(c_i)_y}.
\]

If the canonical record returns hit height
`y_i=H+(c_i)_y tau_i`, its exact live-line placement is

\[
\boxed{t_i=\frac{H-y_i}{-d_y}},\qquad P_i=O_H+t_i d.
\]

This identity is correct.  The invalid implication is instead

\[
y(P_i)=y_i\quad\not\Rightarrow\quad
P_i\in T_{m_i},\;m_i=m_d,\;t_i=t_d.
\]

The canonical first event belongs to another oriented line.  Preserving its
height does not transport its owner triangle or firstness to the live line.
Blending several `y_i` creates a point on no attached owner in general;
selecting one complete record avoids an invented mixed owner but can only
switch between the same unrelated events.

The actual checked-in GCRP confirms that this is not decoder error.  An exact
canonical control reaches `0.99523` silhouette IoU and `0.0000331 m` depth
p95.  On held-out half-bin rays the true live owner appears among the four
records for only `0.3279%` of true hits.  Height blending is `0.63864` IoU with
`8.466 m` depth p95; even a truth-assisted selector over the four records is
`0.93569` IoU with `5.993 m` p95 and `0.337%` attached-triangle agreement.
The exact-horizontal division is undefined, and top-entry first records omit
camera-inside successors.

Therefore the middle-plane address and exact-height placement remain reusable
identities, but four existing canonical events are not an implementation-ready
live event field.  Full derivation and gates are in
`docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-FOUR-VIEW-HIT-HEIGHT-GATE.md`; this is Experiment 023 in the central
ledger.
