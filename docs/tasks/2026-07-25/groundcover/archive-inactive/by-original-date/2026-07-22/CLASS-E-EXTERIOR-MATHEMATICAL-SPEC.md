# Class-E ground cover — corrected exterior mathematical specification

Date: 2026-07-22
Status: **replacement mathematics; implementation remains held until the
finite compiler and sampled-field gates in Sections 11–13 are green**
Scope: mathematics only. This document does not authorize a baker, shader,
render-pipeline change, CPU harness, or WebGPU experiment.

## 0. Result and boundary

This document replaces the mathematical contract inherited by
`CLASS-E-IMPLEMENTATION-HANDOFF.md`. It keeps the useful core of class `E`,
removes claims that were false, and states every remaining approximation as an
approximation.

The resulting architecture has two mathematically different parts:

1. **Opaque structure:** a finite union of globally affine,
   lattice-compatible, parallel-extrusion fields. Its ideal continuous
   exterior first contact is exact with one direction-analytic first-passage
   query per frozen field and a fixed minimum network.
2. **Sub-pixel plume/fuzz:** a small finite analytic extinction field whose
   optical depth is integrated in closed form. It is not a sampled 3D
   `V(q, omega)` transfer record and does not pretend that moments can preserve
   occlusion through opaque blades.

The quality contract is **exterior only**. A camera actually inside plant
matter may fade the ground-cover result. There is no camera-inside successor
theorem, data structure, or quality requirement in this specification.

The corrected model is not yet proved feasible for the accepted tall
Calamagrostis community under the small tap budget. The decisive open question
is whether that community can be compiled into sufficiently few
ownership-compatible affine fields while retaining its botanical silhouette.
That is a structural authoring/compiler question, not a reconstruction
question. It must be answered before implementation.

### 0.1 Classification used throughout

Every claim below carries one of four meanings:

- **Exact:** follows algebraically for the ideal continuous compiled object.
- **Certified approximation:** has an explicit deterministic error bound once
  named constants are measured or conservatively bounded.
- **Measured acceptance:** no uniform theorem is available at visibility
  discontinuities; a specified whole-image or temporal metric is binding.
- **Rejected:** incompatible with the exterior/O(1)/low-cost contract unless
  the upstream representation is changed.

These classes may not be exchanged. In particular, a good visual metric does
not make an approximate identity exact, and an exact reconstruction theorem
does not prove that the compiled plant resembles the authored plant.

## 1. Frozen product contract

The representation must ultimately satisfy all of the following:

- no runtime grass mesh, quad, billboard, shell, raised carrier, or duplicate
  terrain;
- fixed work per covered pixel: no march, loop, traversal, data-dependent
  candidate list, or per-species query;
- every exterior direction, including exact horizontal and vertical
  directions, remains in the quality domain;
- camera-inside-plant quality is out of scope and may fade;
- two global anti-tiling layers, `D4` variants, Estonia-native multi-species
  marks, overlapping cover types, and moss are representable;
- wind is restricted to height-proportional affine shear; per-blade flutter is
  out of scope;
- total resident ground-cover data is at most `250,000,000` bytes;
- the provisional small-cost target is at most `9` cover reads per pixel.
  This is a feasibility target, not a theorem. Exceeding it is a user-visible
  scope decision, not something an implementation may silently do.

There are two error domains:

1. **world-space:** all exterior queries receive the stated world-space or
   categorical guarantees;
2. **pixel-space:** projection bounds additionally require a declared positive
   camera-to-hit distance `D_min`, determined by the renderer near plane and
   closest supported exterior review distance.

`D_min` is not a fade distance. Fade begins only when the camera is actually
inside plant matter.

Before any finite budget is evaluated, freeze:

```text
R_max       finite ray horizon
D_min       closest distance for pixel-space guarantees
theta_pix   conservative angular pixel footprint, including DPR
T_max       provisional cover-read ceiling (= 9)
B_max       resident-byte ceiling (= 250,000,000)
L           anti-tiling layer count (= 2)
tie rule    deterministic equality/owner convention
```

## 2. Four objects that must never be conflated

Let:

- `G_ref` be the accepted authored reference community;
- `G*` be the ideal continuous community compiled into the restricted class;
- `R*` be the ideal continuous first-passage/analytic-medium representation of
  `G*`; and
- `R_h` be its finite stored approximation.

There are three independent comparisons:

```text
compilation:      G_ref  versus G*
ideal algebra:    exact ray query of G* versus R*
finite sampling:  R* versus R_h
```

Only the second comparison is exact by construction. A perfect shader for a
bad `G*` is still a bad plant. A perfect `G*` queried through an insufficient
finite texture is still unstable. The gates in Section 13 score these stages
separately.

## 3. Exact opaque exterior core

### 3.1 One compiled field

Use canonical coordinates `(q, h) in R^2 x R`. For opaque query field `f`,
let:

- `M_f subset R^2` be a closed mask, periodic under a rank-two lattice
  `LambdaHat_f`;
- `A_f(z) = L_f z + c_f` be an invertible affine map from canonical to world
  space;
- canonical transverse lattice translations have rank-two horizontal
  projections, so one bake repeats over a 2D ground chart:

  ```text
  rank {pi_xz L_f(lambdaHat,0)} = 2.
  ```

- a generic field must also work on a locally flat ground chart. There the two
  transverse columns are horizontal, so invertibility requires the world
  extrusion axis `a_f=L_f(0,0,1)` to have nonzero vertical component. For
  terrain-independent finite conditioning the compiler therefore freezes
  `|e_y dot a_f|>=a_min>0` after normalizing the axis.

- the field be valid only on a known finite union of ray-parameter intervals,
  as formalized in Section 6; and
- `R_max` be the finite horizon.

The fundamental finite-axis solid uses a canonical axial interval
`I_f=[h_f^-,h_f^+]`:

```text
G_f = A_f(M_f x I_f).
```

Equivalently, it clips the infinite extrusion by two affine world planes on
which `e_h dot A_f^-1 x` is constant. This makes the compiled chord finite.
However, it does **not** provide a terrain-independent exactly horizontal axis
while retaining rank-two repetition on a flat chart: if the two transverse
columns and the axis were all horizontal, `L_f` would be singular. A
world-horizontal height slab has the additional failure that it cannot bound
a horizontal axis at all.

Additional terrain, top-envelope, or community-support constraints may clip
the field further, but they must satisfy the interval criterion in Section 6.

The mask may contain arbitrarily many primitives and species marks. Runtime
cost depends on the number of compiled fields, not on primitive or species
count.

### 3.2 Ray reduction

For the world ray

```text
r(t) = o + t d,    ||d|| = 1,    t >= 0,
```

write its inverse affine image as

```text
A_f^-1 r(t) = (q_0 + t p, h_0 + t eta),
s_p = ||p||.
```

Intersect the field's analytic validity envelope and `[0,R_max]`. For one
valid interval write it as `J=[t_a,t_b]`. At entry,

```text
q_a = q_0 + t_a p.
```

For `s_p>0`, define the in-plane direction and ideal first-passage function

```text
omega = p / s_p,
rho_M(q, omega) = inf {ell >= 0 : q + ell omega in M_f}.
```

The stored finite field needs only distinguish paths up to the fixed constant

```text
rho_cap,f
  >= sup_(||d||=1, valid J) ||pi_q L_f^-1 d|| |J|
  <= R_max sup_(||d||=1) ||pi_q L_f^-1 d||.
```

A true first passage beyond that cap is equivalent to MISS for the bounded
world ray. An infinite directional corridor is also MISS. This cap is derived
from the frozen horizon and field transform; it is not an empirical free-path
quantile.

The field candidate is

```text
t_f = t_a + rho_M(q_a, omega) / s_p,
```

accepted iff `t_f<=t_b`.

For the categorical pole `s_p=0`, the projected point never moves. The answer
is exactly `t_a` if `q_a in M_f`, otherwise MISS. No epsilon substitutes for
this pole.

### 3.3 Exactness theorem

For `t in [t_a,t_b]`,

```text
q_0 + t p = q_a + s_p (t-t_a) omega.
```

Therefore membership of `r(t)` in the affine extrusion is exactly membership
of the projected 2D ray in `M_f`. The smallest projected path lifts to the
smallest world-ray parameter by `1/s_p`. The interval acceptance test is exact
because it is performed in world-ray parameter.

> **Opaque exterior theorem.** For an origin exterior to `G_f`, the ideal
> continuous field plus the analytic lift returns the exact first closed-set
> contact with `G_f` for every direction, including exact horizontal and
> vertical directions.

This theorem counts tangency as contact. If rendering needs only strict
outside-to-inside crossings, the first-passage field must store that oriented
predicate instead; it may not be recovered by interpolating unsigned depths.

There are distinct event types, and validity intervals are maximal components
created by physical axial/envelope/support constraints before intersection
with the forward cutoff. If `q_a` is outside `M_f`, a finite positive `rho` is
a side-boundary event and its normal/mark belongs to that categorical
mask-boundary record. For `rho=0`, classify in this order:

- if `t_a=0` and `q_a in boundary M_f`, this is a side event or the frozen
  side/cap tie, not a forward-cutoff cap;
- if `t_a=0`, `q_a in interior M_f`, and every physical interval constraint is
  interior, the origin is inside plant matter and the query is outside the
  quality contract;
- if `t_a>0` and `q_a in interior M_f`, the hit is the physical constraint that
  created interval entry: axial bottom/top cap, terrain/top envelope, or
  pointwise-support boundary; and
- if `q_a in boundary M_f` at a physical entry, use the frozen coincident
  constraint rule.

A physical cap normal is the inverse-transpose gradient of its active
constraint; for `y=e^+(q)`, it is proportional to `(-grad e^+(q),1)`.
Owner/material comes from a single-valued occupancy/mark field at `q_a`.

### 3.4 Fixed union and overlap

For a finite frozen field set `F`, with the ray origin exterior to the union,

```text
t_hit(union_f G_f) = min_f t_hit(G_f).
```

A fixed comparison network therefore gives exact exterior ordering, including
order swaps between overlapping representable species and moss. Geometry,
normal, material, and species mark all come categorically from the winning
field/event. They are never arithmetically blended across owners.

The work is

```text
O(L * sum_f J_f),
```

where layer count `L`, field count, and maximum validity-interval count `J_f`
are compile-time constants. This is true O(1); it does not by itself prove the
constant is affordable.

### 3.5 Periodicity and two layers

If `lambdaHat in LambdaHat_f`, then

```text
A_f(q+lambdaHat,h) = A_f(q,h) + lambda_world.
```

Thus the intrinsic toroidal mask and its first-passage field are exactly
reusable under arbitrarily many lattice translations. The complete world
solid is periodic only if every axial/envelope/support constraint transforms
compatibly under the same lattice. A nonperiodic terrain or control envelope
reuses the periodic mask but makes the clipped world solid nonperiodic. Each
globally affine anti-tiling layer can still be queried separately and the two
layer results min-composed.

A golden-angle rotation generally makes the two layer lattices
incommensurate. Their union is then quasiperiodic, not periodic under one
shared lattice. Nothing in this specification relies on a nonexistent common
period.

## 4. What banded masks really represent

A mask displaced between height bands does not create a diagonal chord. It
creates a union of vertical or affine-extruded stair-step slabs. The compiled
solid is exact only for that stair-step object. An actual chord instead uses
its own affine axis and canonical axial cap as defined in Section 3.1.

Exactly horizontal finite structural chords are outside the selected
**terrain-independent, flat-ground-compatible periodic subclass**. A general
affine field can keep a horizontal axis only by allowing at least one lattice
translation to drift vertically, which does not supply reusable flat-ground
cover. Such chords must be deliberately tilted into `|a_y|>=a_min` with the
resulting compiler residual, routed to the analytic medium only when genuinely
sub-pixel, or represented by a new flat-ground-compatible finite-axis
structure with its own proof. The accepted Calamagrostis happens to branch
predominantly upward, but that botanical observation is not a theorem and does
not waive this gate.

To represent a curved or tilted crisp primitive, the compiler must partition
it into actual affine chord pieces. Each piece belongs to a field whose
extrusion axis and clip interval match that chord. Multiple chord pieces may
share one field only when their affine map, attribute layout, and ownership
semantics are compatible.

For source segment `i,s`, let:

- `ell_is` be its chord length;
- `a_is` its source tangent direction;
- `ahat_z` the chosen catalogue axis;
- `kappa_is` a curvature bound;
- `e_shape` a cross-section/mask approximation bound; and
- `e_tip` a cap/tip approximation bound; and
- `e_endpoint` a root-anchoring/finite-segment endpoint mismatch bound.

For the displayed axis term, axes are unoriented, the fitted finite segment is
centred optimally, and angular difference is at most `pi/2`. Any root-anchored
endpoint mismatch is an additional endpoint term. Under that convention, a
conservative primitive residual is

```text
e_is <= kappa_is ell_is^2 / 8
      + (ell_is / 2) sin angle(a_is, ahat_z)
      + e_shape + e_tip + e_endpoint.
```

This is a compiler bound, not a runtime correction. The compiler must choose
segmentation, axis catalogue, banding, and field grouping jointly.

The minimum required axis count at tolerance `epsilon` is a weighted covering
number:

```text
K(epsilon) = N_cover({a_is},
                     d_is(a,ahat)=(ell_is/2) sin angle(a,ahat),
                     epsilon after curvature/shape/tip residuals).
```

There is no theorem that Calamagrostis needs only `4–7` axes. That number must
come from the accepted asset and tolerance, or be rejected.

## 5. Exact sub-pixel plume/fuzz without a false 3D transfer record

### 5.1 Why the old volume field is removed

A record indexed only by `(q,omega)` cannot, for all ray elevations and all
opaque cut depths, encode the transfer of a finite-height 3D medium. Elevation
changes path length and vertical density; a nearer opaque event changes the
integration endpoint. A representative depth, moments, or opacity lump does
not preserve those dependencies or opaque interleaving.

The replacement makes direction analytic and integrates the actual finite
model over the actual clipped ray interval.

### 5.2 Finite analytic medium

For plume stratum `j`, define canonical position `z=(q,h)` and a finite
nonnegative spectral extinction density

```text
kappa_j(z,d)
  = sum_r g_jr(d)
      [kappa_jr0
       + 2 Re sum_m kappa_jrm exp(i xi_jrm dot z)].
```

Requirements:

- horizontal components of `xi` lie on the reciprocal lattice, so the field
  is exactly periodic;
- vertical components are finite Fourier modes, or a separately integrated
  finite polynomial basis, inside an analytic clip slab;
- each `g_jr(d)>=0` is a closed-form directional response constant along one
  straight ray;
- nonnegativity of `kappa_j` is certified, preferably by constructing it as a
  finite sum of squared trigonometric amplitudes;
- each overlapping optical medium has one constant source colour `c_j`, or
  overlapping differently named components share the same `c_j` and may be
  merged;
- coloured crisp glumes, anthers, and axes remain opaque fields. The medium is
  reserved for genuinely sub-pixel, approximately constant-colour fuzz.

### 5.3 Exact closed-form optical depth

Let the ray in canonical coordinates be

```text
z(t) = z_0 + t v.
```

Intersect the plume slab, forward ray, `R_max`, and the nearest opaque/scene
cutoff `t_o`, producing `[a,b]`. Let

```text
Delta = b-a,    m=(a+b)/2.
```

For one exponential mode,

```text
integral_a^b exp(i xi dot z(t)) dt
  = Delta exp(i xi dot z(m)) sinc((Delta/2) xi dot v),
```

where `sinc(0)=1`. Consequently the optical depth is exactly

```text
tau_j = sum_r g_jr(d)
          [kappa_jr0 Delta
           + 2 Re sum_m
               kappa_jrm Delta
               exp(i xi_jrm dot z(m))
               sinc((Delta/2) xi_jrm dot v)].
```

This identity is valid for vertical, oblique, grazing, and exact horizontal
rays, and for every opaque cutoff. It has no sampled direction and no grazing
epsilon.

For constant source colour,

```text
T_j = exp(-tau_j),
C_j = c_j (1-T_j).
```

Ordered medium segments compose front to back using

```text
(T,C) tensor (T',C') = (T T', C + T C').
```

Spatial nonoverlap alone is insufficient: one ray can encounter two separated
pieces of red stratum around a green stratum, and coloured transfer composition
is noncommutative. Exact coloured composition therefore requires one of:

- all interleaving/overlapping media share one source colour and are merged;
- strata are globally depth-orderable and each ray meets each in one interval;
  or
- a compile-time-bounded ordered interval decomposition is supplied, and every
  segment is sorted with a fixed comparison network and composed in ray order.

Overlapping differently coloured media with spatially varying extinction
ratios satisfy none of these automatically. They must be re-authored, given a
shared colour, supplied with the bounded ordered decomposition, or explicitly
accepted as an approximation.

### 5.4 Approximation and budget boundary

A finite trigonometric field cannot have exact nonzero compact support. The
compiler must therefore certify the **maximal partial-ray** integral error
over the entire promised exterior-ray domain:

```text
sup_ray sup_[a,b]subset I_ray
  |integral_a^b (kappa_true-kappa_Q) dt| <= epsilon_tau.
```

The subinterval supremum includes every prefix created by an opaque cutoff,
support clip, transition stratum, and the horizon. A whole-line norm is not
enough because positive and negative fit residuals can cancel only until a
nearer opaque event truncates the ray.

Then, for constant colour,

```text
|alpha_true-alpha_Q| <= epsilon_tau,
||C_true-C_Q|| <= ||c|| epsilon_tau.
```

An `L2` density fit is insufficient because tiny density bias can accumulate
over a long grazing ray. The maximal partial-ray norm, including every
subinterval of the 155 m horizon if that remains `R_max`, is binding.

The number of modes `Q` and strata `S` is a compile-time constant. It adds no
texture samples only when coefficients are true constants/uniforms; otherwise
their access is charged in Section 11. In either case it adds fixed
ALU/register pressure. Values such as `Q=8–16` and `S<=2` are hypotheses to
gate, not accepted facts.

An optically thin, spatially varying-colour extension may approximate colour
by `C_thin=integral kappa(t)c(t)dt`, neglecting internal attenuation, only when
`tau<=tau_max` and source-colour components lie in `[0,c_max]`. Its
conservative max-channel colour error is

```text
||C_exact-C_thin||_infinity
  <= c_max tau_max (1-exp(-tau_max)) ~= O(tau_max^2).
```

## 6. The interval theorem for terrain, height, and control support

### 6.1 General exact statement

Let a canonical extrusion be vertically restricted by envelopes

```text
G = {(q,y): q in M, e^-(q) <= y <= e^+(q)}.
```

Along a ray define the validity set

```text
I_e = {t in [0,R_max] : e^-(q(t)) <= y(t) <= e^+(q(t))}.
```

If `I_e` is one analytically known interval `[t_a,t_b]`, Section 3's query is
exact. If `I_e` is a union of at most a compile-time constant `J_e` of known
intervals, one query per interval followed by a fixed minimum is exact.

> **Interval criterion.** A uniformly bounded number of ray-valid intervals
> with exact endpoints is sufficient for exact O(1) reconstruction. It is also
> necessary for the particular strategy that enumerates validity intervals.
> A different representation could instead supply an equivalent fixed-cost
> first-valid-event oracle; this document does not claim that interval
> enumeration is the only possible data structure.

The criterion is intentionally representation-level. Merely calling a function
smooth does not provide its ray intervals.

### 6.2 Exact terrain cases

Planar ground and planar top envelopes give linear inequalities in `t`, hence
one exact interval. A finite convex support cell with `H` frozen half-planes
also gives one interval using fixed slab clipping. A fixed finite union of
such cells is exact if the maximum number of ray intervals remains frozen and
small.

These are proved exact terrain/control cases. They are not claimed to classify
every possible fixed-cost oracle.

### 6.3 Why arbitrary curved terrain is not exact here

For ground `y=g(q)`, a horizontal ray has relative height derivative

```text
f'(t) = -grad g(q(t)) dot v.
```

This can change sign arbitrarily often. Even a curvature bound alone does not
bound the number of intersections: `g(x)=a sin(Nx)` can keep `a N^2` bounded
while `N` grows and `a` shrinks, producing arbitrarily many intervals. At a
tangent root, arbitrarily small height error can cause arbitrarily large root
movement or change the interval count.

Therefore arbitrary curved terrain plus every horizontal/grazing direction
cannot be made exact by one local affine chart and one query. The admissible
choices are upstream:

1. compile terrain/support into a small certified affine interval cover;
2. accept a bounded local-affine approximation away from tangency; or
3. use a different fixed-cost representation that explicitly supplies the
   first valid interval.

A shader-side root search or variable list is rejected.

### 6.4 Certified local-affine terrain approximation

On a chart of horizontal radius `R`, suppose

```text
||Hessian g|| <= kappa_g,
delta_g <= kappa_g R^2 / 2.
```

The entire pre-hit projected segment must remain inside that chart:

```text
R >= sup_(t in [t_a,t_hit]) ||q(t)-q_anchor||.
```

For a certified MISS, replace `t_hit` by the end of the entire valid interval.

Let the ray/envelope incidence be

```text
nu = inf |d_y - grad g dot d_xz| > 0,
```

and at the relevant mask boundary let

```text
mu = |n_M dot omega| > 0.
```

Then freezing the terrain to its tangent plane gives, to first order,

```text
|delta t_envelope| <= delta_g / nu,
|delta q_entry| <= ||d_xz|| delta_g / nu,
|delta rho| <= ||delta q_entry|| / mu,
|delta t_hit| <= (delta_g/nu) (1 + 1/mu),
```

plus second-order terms controlled by reach. If the plant chart follows the
terrain normal, a height-`h` point additionally incurs orientation error of
order

```text
h C_n kappa_g R.
```

This is not uniform at `nu=0` or `mu=0`; those are terrain and mask
silhouettes and require the categorical/silhouette acceptance of Section 9.
Even away from tangency, owner/firstness additionally needs the whole-ray
visibility-cell clearance in Section 9.3; `nu,mu>0` alone do not exclude an
earlier near-missed component.
The relevant `R` is the actual horizontal search/free path, not merely the
terrain texel size and not an unconditioned p95 if the quality contract covers
the tail.

## 7. Wind and height variation

### 7.1 Exact global affine motion

Wind is authored in plant/root-local coordinates, before terrain placement.
For a plant rooted at horizontal coordinate `q`, local height `h`, terrain
height `g(q)`, spatially constant shear `beta_t`, and height scale `s_t`, the
correct root-fixed world embedding is the map

```text
X_(g,beta,s)(q,h)
  = (q + beta_t h, g(q) + s_t h),    s_t>0.
```

This placement fixes every root exactly:
`X(q,0)=(q,g(q))`. It is not a global world shear about `y=0`, which would move
roots by an amount depending on terrain elevation.

If `g` is affine, `X` is affine, can be absorbed into `A_f`, and Section 3 is
exact. Time dependence is allowed because each frame still uses one affine
map. If `g` is nonlinear, `X` is nonlinear even when `beta_t` is spatially
constant. Its inverse maps a straight world ray to a curved canonical path.
The issue is therefore not that wind was applied after terrain; it is that no
single global affine chart can both follow nonplanar roots and preserve every
straight ray.

The correct cheap approximation freezes terrain to its local affine tangent
`g_a` and queries `X_(g_a,beta,s)`. For the same authored `(q,h)`, the direct
world-space modelling difference is only

```text
X_(g,beta,s)(q,h) - X_(g_a,beta,s)(q,h)
  = (0, g(q)-g_a(q), 0),
||delta X|| <= kappa_g R^2/2.
```

Wind does not introduce an extra direct displacement term when placed
root-locally; it changes the affine chart's conditioning and the incidence by
which this terrain error affects first-hit depth and ownership. The full hit
error therefore uses Section 6.4's incidence and visibility margins.

If `beta_t=0`, curved vertical attachment can be exact when Section 6 supplies
exact envelope intervals because the projected path remains straight. If
`beta_t!=0`, an interval oracle alone is insufficient: exactness needs the
local-affine terrain approximation or a new fixed-cost curved-path
first-passage oracle.

On that affine-ground domain, exact tier-1 transforms therefore include:

- one global shear/height scale per layer;
- global `D4` or other lattice-compatible affine variants; and
- colour/tint/AO/wetness attributes evaluated at the exact winning **opaque**
  hit.

A spatial tint applied to plume changes its source colour along the ray and is
not covered by the constant-colour transfer theorem. Two transformed plume
layers remain exact when their overlapping densities share one source colour
and are merged (with the rotated modes counted in `Q`). Different per-layer or
world-varying plume colours require ordered nonoverlap or the explicitly
optically-thin approximation of Section 5.4.

`D4` is exact here only as one global lattice symmetry applied to a whole
queried layer, or when a finite `D4` arrangement is already authored inside
the periodic mask. A world-keyed, independently chosen `D4` transform per
lattice cell is piecewise affine; a ray can cross arbitrarily many choices and
the base first-passage field no longer identifies the first valid event. That
variant needs a separately proved bounded support oracle and is not silently
included in tier 1.

### 7.2 Spatially varying wind is not an affine conjugation

The field

```text
(q,h) -> (q + beta(q,t) h, h)
```

is nonlinear whenever `beta` varies spatially. Freezing it locally at `q_0`
is an approximation. If `beta` is Lipschitz with constant `L_beta`, then over
horizontal search radius `R` and plant height `h`, the transverse deformation
error is bounded by

```text
delta_wind <= h L_beta R.
```

If this field is treated as a deformation rather than only an error model, it
must also be nonfolding. A sufficient local condition is

```text
det(I + h grad beta(q,t)) > 0,
```

with the stronger easy-to-check condition `h ||grad beta|| < 1`.

Piecewise global shears are exact inside each piece but cannot join
continuously unless adjacent shears agree; a smooth transition is nonlinear
again. Consequently spatial gust variation requires either a certified error
gate using this bound, or explicit fixed analytic support states. It may not
be labelled exact.

### 7.3 Smooth world-keyed height variation

A spatial height scale `s(q)` changes the top envelope to

```text
e^+(q)=g(q)+s(q)h.
```

It is exact only when its ray-valid set satisfies the interval theorem. A
general smooth/hash/control texture does not. Freezing `s` at a root or chart
anchor gives the height error

```text
delta_h <= h L_s R
```

for Lipschitz constant `L_s`, followed by the incidence amplification in
Section 6.4. Applying the modulation after selecting a hit is not exact: it
can create/remove eligible events and change ownership.

The user-requested smooth world-keyed `+/-15%` geometric modulation therefore
remains a **certified approximation requirement**, unless the authoring system
restricts it to a fixed interval-solvable basis. Periodic band-authored height
variation is exact for its compiled stair/chord solid, but is not the same
thing as unrestricted smooth world-keyed variation.

## 8. Community control, multiple species, and moss

Species identity is an attribute of a winning event, not a runtime query
dimension. A field may contain any number of primitives/species that share its
affine direction, clip/interval semantics, owner layout, and attribute layout.

There are two different control semantics.

**Pointwise support** defines

```text
G_j = {(q,y): q in M_j intersect C_j,
                 e_j^-(q)<=y<=e_j^+(q)}.
```

Exact selection follows Section 6: clip the ray to every one of a fixed small
number of support/envelope intervals, query the field in each interval, and
take the fixed categorical minimum. This can cut a blade or head at a
community boundary; that cut is part of the compiled `G*`, not a reconstruction
artifact.

**Whole-primitive/root support** includes a complete primitive according to
its botanical root. The extrusion coordinate `q` labels a material line or
cross-sectional point, not necessarily that root. Testing a root id/offset
after the first hit is not exact, because arbitrarily many root-invalid hits
can precede the first valid one. Root support is therefore exact only when the
root-conditioned primitives are compiled into the mask/support state before
first passage, field splits keep ownership compatible, or a separately proved
fixed-cost first-valid-root oracle is supplied.

An arbitrary streamed A/B control texture does not supply these intervals and
therefore cannot guarantee exact first eligible contact with fixed work.
Permitted exact forms are fixed convex/affine supports, explicit finite
transition states under explicit pointwise semantics, root-aware precompiled
states, or another fixed-cost first-valid-event oracle. Simply
blending A and B geometry records is forbidden. Querying un-clipped A and B
and choosing afterward is also wrong because the nearer event may lie outside
its owner support.

Moss can be a short dense opaque field plus an analytic constant-colour fuzz
stratum. It overlaps taller species through the same fixed minimum/cutoff
algebra; no per-moss or per-species loop appears.

## 9. Finite sampled first-passage fields

### 9.1 Domain and quantizer

For one opaque field, the stored ideal function has domain

```text
rho_f : T^2 x S^1 -> [0,rho_cap] union {MISS}.
```

It is three-dimensional: two intrinsic phase coordinates and one in-plane
angle. Live elevation remains analytic through `s_p` and the interval.

Let the canonical lattice basis be `B=[b_1 b_2]`. With a `P_u x P_v` phase
grid and `N_omega` uniform directions, the exact nearest-cell phase radius is

```text
delta_q
  = 1/2 max_(sigma_1,sigma_2 in {-1,+1})
      ||sigma_1 b_1/P_u + sigma_2 b_2/P_v||,
delta_theta = pi/N_omega.
```

The familiar square-root expression using lattice-vector lengths is valid
only when `b_1` and `b_2` are orthogonal.

Records are categorical. Depth, hit/miss, owner, mark, and normal from
different samples are never linearly blended. Filtered appearance may be
integrated separately over a declared pixel footprint, but it may not invent
an averaged geometry event.

### 9.2 Unconditional un-clipped-extrusion proximity theorem

Suppose the selected stored record is a HIT at

```text
x_i = q_i + rho_i omega_i in M.
```

The live reconstructed projected point is

```text
x_hat = q + rho_i omega.
```

Then

```text
dist(x_hat,M)
  <= ||q-q_i|| + rho_i ||omega-omega_i||
  <= delta_q + 2 rho_i sin(delta_theta/2).
```

After the field's transverse affine map, with operator norm `kappa_f`,

```text
E_extrusion
  <= kappa_f [delta_q + 2 rho_i sin(delta_theta/2)].
```

This bounds distance to the infinite affine extrusion
`A_f(M_f x R)`, not necessarily to the clipped solid `G_f`: the nearby mask
point can lie outside an active axial, terrain, or control interval. A bound to
`G_f` additionally needs interval-cap clearance and support compatibility.
Even then, this does **not** prove that the live ray hits, that this is the live
first event, that the owner is unchanged, or that ray depth is close.

### 9.3 Same-event local theorem

At a unique regular first hit with unit boundary normal `n`, define

```text
gamma = |n dot omega| > 0,
```

and let `r_reach` be a local reach/curvature radius. A positive depth gap after
the hit is not enough to preserve firstness: a nearby perturbed ray may hit a
component that the sampled ray narrowly missed before reaching this branch.

First require a **whole visibility-cell certificate**. With the product metric
defined in Section 9.4, either

```text
dist_U(u_i,D_M) > r_h,
```

or an equivalent swept-ray clearance certificate must prove that every query
in the quantizer cell has the same intrinsic mask hit/MISS, first boundary
segment, and owner throughout the entire projected path before the event. Only
inside such a certified cell may local depth analysis begin. Set

```text
e_0 = delta_q + 2 rho delta_theta_sin,
delta_theta_sin = sin(delta_theta/2).
```

Implicit differentiation of the boundary equation gives

```text
delta rho
  = - n dot (delta q + rho delta omega) / (n dot omega)
    + O(e_0^2/r_reach).
```

Under a lower incidence bound `gamma>=gamma_0>0`, sufficient reach/curvature
control, and a cell small enough for the implicit-function neighbourhood, the
same branch persists. Reusing raw sampled path gives the first-order behaviour

```text
|delta rho_raw| = O(e_0/gamma_0),
|delta t_raw|   = O(e_0/(gamma_0 s_p)).
```

The `1/gamma` and `1/s_p` factors are real. They may not be deleted by calling
`rho delta_theta` a depth bound.

The fixed-cost corrected record stores the sampled hit

```text
x_i = q_i + rho_i omega_i
```

and its canonical side-boundary normal `n_i`. For the live query, intersect its
ray with the stored tangent line:

```text
rho_hat = n_i dot (x_i-q) / (n_i dot omega).
```

When phase is unchanged (`q=q_i`), this reduces to

```text
rho_hat = rho_i (n_i dot omega_i)/(n_i dot omega),
```

the precise cosine-ratio correction behind the article's `|OB|=|OA|/cos
alpha` construction. The full numerator is required when phase also changes.

This is **exact** throughout a certified same-segment visibility cell when the
compiled boundary segment is straight and `n_i` is exact. For a `C^2` boundary
with reach `r_reach`, Taylor expansion cancels the first-order phase/angular
term and gives

```text
|rho_hat-rho_live| <= C(gamma_0) e_0^2/r_reach
```

for a sufficiently small certified cell. Normal encoding error `delta_n`
adds a term of order `rho delta_n/gamma_0` and must be budgeted. Segment-end,
owner, hit/miss, and silhouette cells remain categorical edge cells; the
tangent formula may not be used to claim correctness there.

Intrinsic segment stability is not yet final clipped-solid stability. Let

```text
S(u,chi) = s_p (t_b-t_a)
```

be the projected length of the live validity interval. A side hit additionally
needs `0<rho_live<S`. The whole cell must have a positive interval margin from
both endpoints, or be classified as an interval-edge cell. Changes in entry
constraint, support/control state, and cap/side type are certified in the
extended category of Section 9.4.

The record therefore contains a categorical `TANGENT_SAFE` bit, or an
equivalent deterministic atlas classification, which is true only when both
intrinsic visibility and interval margins are certified. Unsafe cells use the
declared raw categorical/coverage fallback and carry its measured error; they
do not borrow the tangent theorem. The bit and fallback payload count in
`bytes_per_record`; any extra fallback sample counts in `r_f`.

This one-record tangent reintersection, rather than raw `rho_i` reuse, is the
mathematical reason a practical class-E lattice may be much smaller without
reintroducing first-order stretching. It adds fixed ALU and no candidate, but
its extra normal/record bits count in Section 11.

The correction proves geometry, not arbitrary sampled appearance. Owner and
species mark remain exact only when constant on the certified boundary
segment. A varying attribute `a(x)` must either be evaluated analytically at
the corrected live point, use a fixed boundary chart supplied by the record,
or carry a sampling bound such as

```text
||a(x_i)-a(x_live)|| <= L_a ||x_i-x_live||.
```

Using the sample's colour at a different corrected point without one of these
conditions is an appearance approximation.

### 9.4 Visibility discontinuities and silhouettes

At `gamma=0`, at owner ties, and at hit/miss changes, no uniform categorical
depth theorem exists. Split intrinsic and final categories. The intrinsic mask
category is

```text
C_M : U=T^2 x S^1
      -> {intrinsic MISS or exact mask segment/owner},
```

with discontinuity set `D_M`. Two hits on different edges of the same blade
are different intrinsic categories for tangent reconstruction.

Final clipped-solid category also depends on

```text
chi = (t_a,t_b, active entry/exit constraints,
       support/control state, s_p),
C_G : (u,chi) -> {final MISS, side, cap, owner/mark},
```

with extended discontinuity set `D_G`. The same `(q,omega)` can be a side hit,
MISS, or cap when `chi` changes. An atlas certificate on `U` proves only
intrinsic segment identity; interval-edge/cap/support correctness must also be
certified against `D_G` or by explicit positive interval margins.

Freeze a dimensionally valid product metric on `U`, for example

```text
d_U(u,u') = max(||q-q'||/lambda_q,
                wrap_angle(theta,theta')/lambda_theta),
```

with declared phase and angular scales. For nearest-cell quantizer `Q_h`, let
`r_h=sup_u d_U(u,Q_h(u))`. Then

```text
{u : C_M(Q_h(u)) != C_M(u)} subset D_M^(+r_h),
```

the `r_h`-neighbourhood of the discontinuity set.

For camera-to-query map `Phi`, wrong pixels lie in

```text
Phi^-1(D_M^(+r_h)).
```

The analogous statement for final geometry uses the extended camera map into
`(u,chi)` and `D_G`. These are the correct organizing results for silhouettes,
interval edges, and owner changes. Their projected size and connected shape
must be measured or geometrically certified. A p95 free path cannot prove them
small.

The pole `s_p=0` uses mask membership. For near-pole rays, categorical errors
are confined to the swept mask-boundary neighbourhood, but depth within that
neighbourhood can be large; it is scored as a silhouette/owner event, not
hidden with an epsilon.

### 9.5 Raw-record grazing bounds and the remaining angular wedge

For an orthonormal vertical slab of height `h`, camera clearance `c>0` above
the slab, and elevation `alpha in (0,pi/2)`, a hit with horizontal path `rho`
satisfies

```text
D(rho,alpha) = c/sin(alpha) + rho/cos(alpha),
rho <= h cot(alpha),
rho/D <= h cos(alpha)/(c+h) <= h/(c+h).
```

Thus, **only for this above-slab geometry**, a sufficient raw-record angular
extrusion-proximity condition is

```text
2 kappa_f sin(delta_theta/2) h/(c+h)
  <= k_support theta_pix.
```

Phase quantization separately requires

```text
kappa_f delta_q <= k_phase theta_pix D_min.
```

The complete exterior domain also contains a camera in empty air inside the
botanical slab. For an exact horizontal ray through such a gap, `D` can be
approximately `rho`; in the orthonormal case the general bound is only

```text
rho/D <= 1,
2 kappa_f sin(delta_theta/2) <= k_support theta_pix.
```

For a general affine field replace `1` by
`sigma_q,f=sup_(||d||=1)||pi_q L_f^-1 d||`; explicitly,

```text
E_angular/D
  <= 2 kappa_f sigma_q,f sin(delta_theta/2).
```

Raw angular sampling therefore **does** make a world displacement proportional
to distance and a distance-stable screen wedge. The above-slab formula merely
scales its coefficient when `rho/D` is provably small.

This is why raw path reuse can require an impractically large `N_omega`. The
tangent reintersection of Section 9.3 cancels that first-order wedge on
certified straight/smooth same-boundary cells. It does not remove categorical
wrong-owner/silhouette sausages, so no small direction count is assumed before
the compiled visibility cells are certified.

### 9.6 Pixel footprints, minification, and the old shimmer

The preceding results concern one mathematical screen ray. They do not by
themselves antialias a pixel footprint. Let `Pi` be the distribution of rays
inside a pixel/filter footprint. The exact filtered result is the integral of
the **complete categorical first-event/compositing result** over `Pi`, not the
result of averaging path, normal, or owner records before comparison.

For two subrays, two opaque fields can each have coverage `1/2` while their
union coverage is either `1/2` (the same subray is covered) or `1` (opposite
subrays are covered). Separate coarse per-field alpha values are identical in
both cases. Therefore independently mipping fields and composing their average
opacities loses cross-field correlation and is not exact.

Likewise, exact filtering against an arbitrary opaque cutoff `t_o` requires a
coverage/colour distribution as a function of cutoff. A single averaged
depth, finite moments, or a representative owner cannot reproduce every
possible cutoff.

There is also a simple finite-sample no-go result. For any deterministic
filter that evaluates only a fixed finite set of subrays, construct two masks
that agree on every sampled subray but differ on an unsampled open subset of
the footprint. The algorithm receives identical data while the exact pixel
integrals differ. Thus exact filtering of arbitrary masks requires either a
joint preintegrated footprint representation, analytic restrictions on the
mask, or unbounded sampling. It cannot be recovered by a clever average of a
fixed few categorical hits.

The binding rules are therefore:

- ordinary linear mips of `rho`, normal, mark, or owner are forbidden;
- near-field categorical geometry is queried at a declared sample ray and
  antialiasing belongs to the screen resolve/sample distribution;
- any coarse jointly baked coverage/radiance level is a separate approximate
  representation of the **joint community**, not an exact per-field dual;
- any opaque-to-extinction distance transition carries a measured correlation
  and temporal-stability error; and
- deterministic shimmer/noise under camera translation is part of Gate D,
  not dismissed as an unrelated legacy effect.

This specification does not yet select the coarse-distance antialiasing
mechanism. A mechanism that needs more reads, a new pass, or per-pixel
candidates must be charged to the same fixed budget and approved before
implementation.

Two fixed-cost routes remain mathematically honest:

1. One subray drawn from the declared footprint distribution is an unbiased
   estimator of filtered radiance,
   `E[L(ray_Xi)]=integral L(ray)dPi`, but has nonzero variance. Stable temporal
   accumulation can reduce that variance; motion/reset shimmer remains a
   measured property, never an exact single-frame claim.
2. Crisp microgeometry may transition, only once its projected scale is
   sub-pixel, to a **joint-community** analytic extinction fit of Section 5.
   This is direction-analytic and fixed-cost, but converting correlated opaque
   cover into Beer--Lambert transfer is an explicit approximation. It must be
   fit/gated on filtered transfer and partial cutoffs, not on independent
   per-field densities.

The second route is the stronger deterministic candidate under the current
low-end contract, but remains unselected until its joint transfer, colour, and
temporal errors are below the visual thresholds at fixed `Q`.

## 10. Invocation-support condition

The mathematics can return a grass hit only on a pixel where some runtime
fragment invocation executes the query. A terrain-only carrier is not
automatically sufficient: an exact horizontal or upward ray may intersect a
plant silhouette but never intersect the terrain behind it.

Therefore the implementation interface has a separate necessary condition:

> Every promised exterior screen ray that can hit `G*` before opaque scene
> geometry must receive at least one suitable invocation, without introducing
> forbidden grass geometry. Multiple invocations are acceptable only with
> deterministic categorical depth/composition semantics.

This might be satisfied by an already existing full-coverage fragment path or
another existing scene pass. The mathematics does not assume which. If the
current pipeline cannot satisfy it without a new carrier/pass, that is a
runtime architecture decision to surface to the user, not a shader detail and
not evidence against the class-E intersection algebra.

## 11. Compiler optimization and feasibility equations

### 11.1 Field identity and ownership

An opaque query field is identified by the tuple

```text
(affine axis/chart,
 clip or interval semantics,
 ownership-compatible group,
 attribute layout,
 sampled resolution/catalogue).
```

Projected ownership at a first boundary must be single-valued under the
chosen tie convention. Conflicting coincident owners split fields or receive
one explicit authoring priority; they are never numerically averaged.

The compiler jointly optimizes:

```text
primitive segmentation
axis catalogue
field grouping
clip/band intervals
phase and angle resolution
plume modes/strata
```

subject to compilation fidelity, sampled-field error, tap cost, ALU cost, and
resident bytes. Optimizing axes before band/field grouping can produce a
catalogue that is geometrically small but runtime-infeasible.

### 11.2 Tap equation

Let the provisional `T_max=9` metric mean texture/buffer samples, not abstract
arithmetic and not all physical cache transactions. Let `r_f` be the number of
packed record samples required for one query of opaque field `f`, including any
separate categorical pole/attribute sample that cannot be packed. Then

```text
T_cover
  = L sum_f J_f r_f
    + T_interval + T_control + T_plume + T_material
    + T_minify
  <= T_max.
```

An analytic interval oracle costs zero samples only if all its data are true
constants/uniforms or already available inputs. The same applies to spectral
plume coefficients: their texture/buffer access contributes to `T_plume`; if
held as constants/uniform data they instead contribute explicitly to fixed
ALU, register pressure, and constant-memory traffic.

In the illustrative near-field lower-bound case `L=2`, `J_f=r_f=1`, and all
other terms totaling one sample, `T_max=9` permits at most four opaque fields.
This is not a recommendation or a final budget; it is the arithmetic
feasibility boundary before unresolved minification cost. If the accepted plant
needs five axis/band fields per layer, the proposed `9`-sample target is already
false before shader work.

Species do not multiply reads when they share fields. Independent directions,
clips, or overlapping ownership structures do.

### 11.3 Memory equation

Without assuming compression, resident stored bytes are

```text
B_total
  = sum_profiles sum_fields sum_levels
      P_u P_v N_omega bytes_per_record
    + B_gutters + B_tables + B_interval
    + B_control + B_plume_coeff + B_material + B_minify
  <= B_max.
```

Two transformed layers share the same immutable atlases and therefore do not
double bytes, but they do double queries. Distinct community profiles do
multiply residency unless streamed mutually exclusively. Block compression
may be credited only after the exact record format and measured quality are
known.

Because class `E` samples one angular coordinate, its leading angular bytes
scale linearly with `1/delta_theta`, not quadratically as a two-angle frame
field does.

Until Section 9.6 selects a minification mechanism, `T_minify` and `B_minify`
are unknown. Gate B can report only the near-field/base lower bound; the final
resource total is not known until the minification candidate is fixed and
Gate E is evaluated.

### 11.4 No pre-compiler free-path budget

The directional free-path distribution belongs to the compiled masks and
field grouping. It cannot be measured meaningfully on the arbitrary source
mesh and then used to size fields that do not yet exist. Budget order is:

1. structurally compile candidate `G*`;
2. measure/certify each resulting `rho_f`, reach, incidence, owner gaps, and
   silhouette discontinuity set;
3. derive its phase/angle sampling and record layout;
4. reject or accept the joint compiler result under taps and bytes.

## 12. Exact, bounded, and unresolved feature table

| feature | status under this specification | condition |
|---|---|---|
| ideal opaque first contact | exact | globally affine field + known finite ray intervals |
| exact horizontal/vertical view rays | exact in ideal field | continuous in-plane query + categorical family-axis pole + exact interval |
| exactly horizontal finite plant axis | outside selected flat-ground-compatible periodic subclass | tilt with residual, sub-pixel medium, or new proved structure |
| intrinsic periodic mask/field | exact | lattice-compatible affine field |
| complete world-solid periodicity | conditional | every envelope/support must share the lattice |
| two global rotated layers | exact per layer | separate query/min; union is generally quasiperiodic |
| multi-species overlap | exact | species encoded as marks; fixed compatible field set |
| crisp owner/normal/attributes | exact in ideal field; conditional when sampled | categorical winner; constant/analytic boundary attributes or Lipschitz bound |
| finite sampled geometry | bounded locally / measured at discontinuities | Sections 9.2–9.4 |
| pixel-footprint/minified opaque appearance | unresolved measured approximation | no linear geometry mips; preserve joint-field correlation |
| plume optical depth | exact for compiled analytic medium | finite basis, analytic clips, actual opaque cutoff |
| coloured plume composition | exact only with ordered segments | shared colour, global ordering, or fixed ordered interval decomposition |
| plume vs reference | certified approximation | maximal partial-ray line-integral norm |
| planar terrain | exact | affine chart and interval |
| curved terrain | bounded away from tangency | curvature, radius, incidence bounds |
| arbitrary curved terrain at all grazing rays | rejected in current representation | needs fixed interval oracle/cover |
| global height-proportional wind | exact on affine ground | one spatially constant affine shear per layer; curved attachment is approximate |
| spatially varying wind | certified approximation | `h L_beta R` plus visibility gate |
| global height scale | exact on affine ground | affine scale; curved root attachment inherits Section 6 |
| smooth world-keyed height scale | certified approximation or restricted basis | interval theorem / `h L_s R` |
| arbitrary streamed community controls | unresolved/rejected as exact | needs fixed support intervals or authored transition states |
| whole-primitive/root community controls | conditional | root-aware precompiled masks or fixed first-valid-root oracle |
| camera inside plant | out of scope | fade; no successor machinery |
| runtime screen coverage | required interface gate | every possible grass-hit pixel receives an invocation |

## 13. Binding gate order

No implementation phase begins merely because the ideal opaque theorem is
correct. The following order is mandatory.

### Gate A — freeze the actual contract

Freeze `R_max`, `D_min`, `theta_pix`, `T_max`, `B_max`, `a_min`, the product
metric scales `(lambda_q,lambda_theta)`, tie semantics, layer count,
exact-vs-approximate terrain/wind/height/control choices, minification
acceptance semantics and reserved cost, and the screen-invocation assumption.
The exact minification route may remain open through the base compiler gate,
but must be selected before final Gate E. If any frozen quantity is unknown,
report it; do not insert a convenient inherited number.

### Gate B — structural compiler feasibility

Compile the accepted tall Calamagrostis and a dense low/moss community into
actual affine chord fields. Report:

- axis covering curve `K(epsilon)`;
- residual/coverage of primitives forced away from an exactly horizontal axis;
- primitive residual distribution and maximum crisp residual;
- number of fields after band, support, owner, and attribute compatibility;
- explicit pointwise-versus-root community-support semantics;
- visibility-cell coverage and the fraction using exact straight-segment
  tangent reintersection versus curved approximations;
- near-field/base `T_cover` lower bound for two layers, with minification cost
  explicitly still unknown until a Section 9.6 route is selected;
- plume basis/ordered-segment count and maximal partial-ray integral residual;
- which tier-1 height/wind/control terms are exact and which use bounds.

This gate is RED if botanical fidelity requires more fields/reads than the
frozen cost, if the plume line-integral tolerance is not met at fixed analytic
cost, or if a required support cannot be expressed with bounded intervals.

### Gate C — compilation fidelity (`G_ref` versus `G*`)

Score the actual reference and compiled object from the entire exterior view
domain, including top-down, oblique, grazing, exact horizontal, all azimuths,
community edges, species overlap, and moss under blades. Crisp silhouette,
topology, colour ownership, and plume are scored separately. A good aggregate
image metric may not hide loss of the accepted Calamagrostis head or branch
structure.

### Gate D — finite sampled representation (`R*` versus `R_h`)

For the exact fields produced by Gate B, certify regular same-event regions
with Section 9.3 and score discontinuities with:

- intrinsic `C_M` and extended clipped-solid `C_G` disagreement, including
  interval-edge/cap changes;
- `TANGENT_SAFE` coverage and fallback error;
- hit/miss and species-mark disagreement;
- silhouette IoU and maximum/connected wrong-region size;
- categorical owner stability;
- winner world-position error on regular crisp content;
- RGB/alpha error for filtered appearance;
- unforced class changes under `1–4.5 mm` camera translations;
- distant minification and shimmer under continuous camera motion, with no
  independent per-field opacity composition presented as exact;
- exact horizontal, vertical, tile-boundary, owner-order-swap, and near-pole
  counterexamples.

The sampled lattice must satisfy the law being tested. Testing a configuration
below its own derived resolution cannot refute the representation.

### Gate E — resource and invocation feasibility

After selecting the minification route, verify the uncompressed and compressed
byte equations including `B_minify`, the complete sample equation including
`T_minify`, fixed analytic plume/minification ALU, and the existence of a
fragment invocation for every promised grass-hit pixel. This is still a
mathematical/architecture gate; it does not authorize shader work.

### Gate F — only then transcribe into baker and runtime

Only after Gates A–E are green may the equations be mapped to implementation.
Shader work must remain a transcription of this specification. Any proposed
runtime fix that adds candidates, samples, dependent searches, loops, passes,
or geometry reopens the mathematics instead of silently expanding cost.

## 14. Go/no-go verdict

The corrected specification is internally constructive but **conditionally
feasible**:

- The ideal opaque exterior algebra is exact.
- The finite analytic plume algebra is exact for its compiled medium and has
  an honest source-approximation norm.
- Finite sampling has a valid raw extrusion-proximity bound, an exact
  tangent-line correction on certified same-segment cells, and an explicit
  categorical discontinuity boundary. Raw angular slices still make a
  distance-stable screen wedge for in-slab horizontal rays; no small
  `N_omega` is assumed until the tangent/categorical cells are certified.
- Terrain, wind, height variance, and community control are exact only inside
  the interval/affine restrictions stated here; outside them they carry named
  bounds or remain rejected.
- Exactly horizontal finite structural axes are outside the terrain-independent
  periodic class and need a measured tilt residual or a new proved structure.
- The accepted tall plant has not yet been shown to fit at most four one-sample
  opaque fields per layer when all other cover samples total one under a
  `9`-sample/two-layer target.
- Stable minified opaque appearance is unresolved; independent per-field
  opacity mips are proved insufficient in general.
- Full exterior screen coverage still requires a compatible fragment
  invocation path.

Therefore the correct next outcome is not a shader preview. It is a structural
compiler report that either produces a botanically faithful `G*` within the
frozen field/tap/byte limits, or gives the precise Pareto frontier between
fidelity and cost for user decision. Until that report is green, the runtime
implementation hold remains active.

## 15. Provenance

This specification was derived from:

- `GRASS-EXACT-REPRESENTATION-THEORY.md`, retaining only its valid affine
  direction-analytic opaque core;
- `CLASS-E-MATHEMATICAL-AUDIT.md`, including its counterexamples and corrected
  bounds;
- the shell-frame failure and direction-amplification analyses recorded in
  `GRASS-SHELL-FRAME-FIELD*.md` and `GRASS-STATUS-AND-ISSUES.md`;
- Sannikov's precomputed grass method as documented in
  `docs/deep-research/grass/`, for the core idea of removing live elevation
  from sampled direction by analytic vertical/extrusion structure; and
- the independent LAAS derivations for the interval theorem, local
  same-event bound, quantized silhouette sausage, weighted compiler covering
  problem, and finite analytic plume integral.

No theorem here is attributed to an external author unless the source states
it. The new interval, compiler, sampling, and spectral-medium results are
project derivations and must be cited as such if this work becomes a paper.

## 16. Post-specification CPU gate result (2026-07-22)

Gate B is **RED** for the selected `G_f=A_f(M_f x I_f)` opaque class. The
accepted production Calamagrostis blades and culms alone require at least
`14` fields / `29` optimistic two-layer reads at `10 mm`, `9` / `19` at
`20 mm`, and `5` / `11` even at `50 mm`, before the reproductive head is
represented. This is a catalogue-independent cap-plane lower bound: one
finite product field supplies only two horizontal world cap planes.

This is not repaired by slightly relaxing a perceptual threshold. The
fixed-catalogue lower bound first reaches the nominal four-field/nine-read
boundary only near `100 mm` compilation error, where the constructive result
still requires `10` fields / `21` reads. That error is botanically destructive.

The implementation hold therefore remains active. The exact ideal ray
reduction is retained as reusable mathematics, but a successor representation
must encode phase-dependent finite endpoints at fixed cost without losing the
next eligible event. Full result, source bindings, threshold sweep, proof
boundary, commands, and hashes are in `CLASS-E-CPU-GATE-RESULT.md`.
