# Ground cover lateral-core + analytic-residual mathematics

Date: 2026-07-22
Status: **CPU candidate RED and track parked; ideal core remains constructive, full K2+K2 class unresolved**
Scope: exterior camera views. A camera entering plant matter may fade. Air gaps
inside the overall cover slab remain ordinary exterior queries.

## 0. Result

The failed finite-endpoint product representation cannot be repaired by a
slightly better endpoint table. There is an information-dimensional
obstruction: arbitrary local endpoints require either the missing pointed-ray
coordinates, an unbounded successor search, or true axial invariance with
shared clips.

The selected bounded-quality successor is therefore a hybrid authored object,
not another purported exact arbitrary-mesh codec:

```text
compiled ground cover = cap-free lateral core + analytic soft residual
```

- The ideal lateral core contains only source-contained affine structures
  whose side entries are reconstructed exactly on the live ray. Its finite
  atlas obeys the explicit certified/unsafe-cell bounds in Section 3.4. It
  never emits an axial cap.
- Omitted caps, curved tails, minor leaves, reproductive fuzz, and other
  structures that do not fit the fixed core budget become one jointly authored
  analytic transfer residual. They are not silently discarded and are not
  treated as categorical geometry.
- The residual is evaluated from the live ray in closed form. It has no camera
  direction lattice and therefore cannot generate the old `r DeltaTheta`
  widening sectors.
- Two global anti-tiling transforms remain. With at most four core fields, the
  geometric path costs at most eight complete-record reads. One existing
  control/material read leaves the total at nine. Residual coefficients are a
  tiny fixed constant block and add no texture reads.

The continuous core query and the optical-depth/moment evaluation of the
declared residual density are exact. Finite core-atlas sampling, mixed-colour
transfer ordering, curved-terrain linearization, and the offline conversion
from the accepted plant are explicitly bounded approximations. All must pass a
distance-aware visible-transfer gate before runtime work begins. The two
permitted exterior CPU attempts did not pass; Section 8.6 records the measured
RED, the exact limit of that verdict, and the objective resume condition.

## 1. Frozen contract

The successor must retain all of the following:

- `O(1)` fixed work per pixel;
- no loop, march, runtime traversal, candidate list, per-copy work, or
  per-species work;
- no runtime grass mesh, quad, billboard, shell, raised carrier, or floating
  terrain copy;
- no sampled camera-elevation coordinate and no hidden/clamped view direction;
- correct live-ray treatment of horizontal and vertical directions;
- two global affine anti-tiling transforms unless a later explicit user
  decision selects the one-layer quality trade;
- at most approximately nine cover reads and low resident memory;
- categorical core colour, normal, material, and species marks;
- cook-side joint composition of overlapping species and moss;
- root-local height-proportional affine wind;
- no claim of camera-inside quality once the camera enters plant matter.

The accepted arbitrary Calamagrostis mesh remains the visual reference. It is
not required to remain the runtime representation.

## 2. Why exact local endpoints are impossible here

### 2.1 Pointed projected line

For one affine family, parameterize a live ray by projected distance `ell`:

```text
q(ell) = q0 + ell omega,
h(ell) = h0 + k ell,
k = eta / s_p.
```

Along the projected line, let the mask have ordered occupied components

```text
C_1, C_2, ..., C_N
```

and let component `C_i` carry an independent finite axial eligibility interval
`I_i`. The exact first contact is the first component for which

```text
there exists ell in C_i with h0 + k ell in I_i.
```

### 2.2 Fixed-prefix counterexample

For every fixed `K`, choose `K` nearer projected components whose intervals
exclude `h0+k ell`, followed by component `K+1` whose interval includes it.
Any record that exposes only the first `K` projected events returns a miss or
the wrong owner even though event `K+1` is the true first eligible contact.

The construction works with `k=0` by assigning disjoint height intervals and
varying the exterior origin height `h0`. It also works for top-entry exterior
rays by fixing `h0` and assigning intervals so the winner visits arbitrarily
many disjoint slope ranges as `k` varies. The origin may remain in air
throughout; fading cameras inside plant matter does not remove the example.

### 2.3 Endpoint-successor trilemma

In the computation model used here—a bounded number of fixed-size sampled
records plus fixed algebra, with no content-dependent traversal—for a
detail-closed source class with arbitrary component-local endpoints, exact
exterior first contact requires at least one of:

1. a pointed-ray field addressed by `(q0[2], omega, h0, k)`;
2. an ordered successor structure whose inspected depth grows with content;
3. a source restriction that gives true axial invariance or a fixed finite set
   of globally shared clip intervals.

The first is a five-dimensional pointed-ray field. Even an optimistic
`128^2 x 32 x 16 x 16` eight-byte field is `1 GiB` before mips, complete
attributes, multiple communities, or interpolation safety. Sampling its
categorical slope/origin axes recreates coherent winner sectors.

The second violates the fixed-cost/no-candidate contract. The third is the
only inexpensive exact option and is precisely the restricted affine class
whose cap count failed the production-plant CPU gate.

### 2.4 Why common proposed escapes do not change the theorem

- A per-texel interval merely moves `I_i` into the first record. Rejecting that
  record still hides an arbitrarily later eligible component.
- Height bands have different entry phases
  `q_b=q0+p t_b`; they therefore require different addresses/reads. Packing
  scalar band values into one vector does not make those addresses equal.
- A normalized-height warp makes world lines curved in the new coordinate.
  If the warp preserves lines, projective geometry reduces it to a projective
  map with common plane caps. If it does not preserve lines, the missing curve
  parameters replace the missing endpoint coordinates.
- A bounded-degree inverse bounds roots for one primitive, not election over
  an unbounded number of projected components.
- Fixed-`K` events, learned selectors, and direction bins approximate the
  priority function but do not make it fixed-complexity or direction-analytic.

This closes exact arbitrary local endpoints under the retained contract. The
remaining freedom is an explicit approximation of the authored source.

## 3. Exact cap-free lateral core

### 3.1 Source-contained field

Let `S_i` be a crisp source surface. For core field `f`, choose:

- an invertible affine map `A_f(q,h)` compatible with the periodic ground
  chart;
- one true shared interval `I_f=[h_f^-,h_f^+]`;
- owner-marked, periodic, closed one-dimensional sets `Gamma_if` in the phase
  plane, each a finite union of piecewise-regular arcs whose endpoints are
  included, satisfying

  ```text
  A_f(Gamma_if x I_f) subseteq N_epsilon(S_i);
  ```

- the ownership-compatible curve union `Gamma_f=union_i Gamma_if`.

`Closed` here is a set-theoretic condition, not a demand that every component
be a closed loop or bound a filled planar region. A finite line segment with
its endpoints included is a closed subset of the phase plane. Its product
with `I_f` is an ordinary finite ribbon sheet: it has one-dimensional boundary
edges but no two-dimensional axial cap. Tubes may use closed curves; blade
faces and relay fragments may use arcs. The runtime first-passage problem is
intersection with this curve set, so no binary solid occupancy is required.

The interval is genuinely field-global. A primitive cannot slide, recenter,
or rescale `I_f` locally. If `nu_f` is the canonical height covector (the
third row of the inverse linear map), its actual source eligibility is the
overlap of `I_f` with the source patch's measured `nu_f dot x` range.

Ground periodicity makes this restriction stronger, not weaker. Let the two
independent ground-lattice translations be `lambda_1,lambda_2`. A finite
interval can be invariant under every repeated copy only if

```text
nu_f dot lambda_1 = nu_f dot lambda_2 = 0.
```

Thus `nu_f` is gravity-height-like for a horizontal ground lattice. The
extrusion vector `A_f e_h` may be a sheared upright direction such as
`(beta_x,1,beta_z)`, but it cannot be exactly horizontal: invertibility
requires `nu_f dot (A_f e_h)=1`. Compiler searches that give every ribbon its
own width interval, or admit a horizontal width-axis with a finite local cap,
do not instantiate this representation and cannot be used as evidence for it.

Without loss of the selected gravity-up contract, use the structured chart

```text
A_beta(q,h) = (q_x+beta_x h, h, q_z+beta_z h),
A_beta^-1(x,y,z) = (x-beta_x y, z-beta_z y, y).
```

For a source sweep with endpoint difference `(dx,dy,dz)`, `dy!=0`, its
compatible shear is `beta_s=(dx/dy,dz/dy)` and its true axial range is the
global world/root-local height interval between the endpoint heights. A field
`(beta_f,I_f)` may retain that patch only when its one `I_f` is contained in
the patch's axial range. Its retained area fraction is the band-height fraction
and its transverse shear error is bounded by

```text
0.5 |I_f| ||beta_s-beta_f||
```

plus the patch's certified curvature/taper/frame remainder. This is the
binding model for a same-primitive constructive compiler. It is a useful lower
bound, but it is not the strongest source-contained compiler.

### 3.1.1 Owner-compatible relay templates

The product theorem does not require one template boundary component to stay
attached to one source-triangle identity. It requires every emitted point to
remain close to a compatible source surface. Dense grass can therefore use a
relay template that follows different nearby blades/glumes at different
heights without changing species/material.

For one categorical owner/mark class `m`, define the source slice in field
coordinates

```text
C_m(h) = {q : A_beta_f(q,h) lies on source surface S_m}.
```

For tolerance `epsilon`, its persistent compatible corridor over one true
global band is

```text
P_m(beta_f,I_f,epsilon)
  = intersection_(h in I_f) {q : dist(q,C_m(h)) <= epsilon
                                  and normal/mark constraints hold}.
```

Choose a periodic compact curve set

```text
Gamma_m subseteq P_m.
```

Components may be finite rectifiable piecewise-`C1` arcs with their endpoints
included, or loops. They do not require artificial closing connectors: an arc
times `I_f` is a ruled sheet with boundary, represented by exactly the same
first-passage oracle. The curve must be a seam-consistent closed set on the
phase torus; this is not permission for a half-open dangling raster fragment.
Given such a curve, every point of
`A_beta_f(Gamma_m x I_f)` is within `epsilon` of some compatible source point,
even when the nearest source primitive changes with height. This is a direct
proof of the same one-sided lateral certificate; it does not reintroduce filled
local caps, successor search, or a runtime identity list.

Position alone is insufficient. `Gamma_m` has one transported boundary normal
and height-independent categorical payload, so the compiler uses an oriented
corridor in `(q,allowed normal cone,payload)` and requires

```text
n_Gamma(q) in intersection_(h in I_f) N_allowed(q,h).
```

Species/material, side orientation, tint class, and every stored categorical
mark must likewise remain compatible across the relay. Individual recipe or
triangle identity need not. A declared global analytic height ramp is allowed;
arbitrary relay-specific colour changes with height would require the missing
axial coordinate and are not.

The certificate is set-wise, so it does not impose a one-to-one surface-area
budget on each source triangle. Such a budget would contradict the relay: one
persistent curve is specifically allowed to hand off among many short source
triangles with height. Density inflation is controlled at the compatible
payload/band and first-visible-transfer levels instead: emitted sheet area is
reported separately, source correspondence is deduplicated over the whole
path neighbourhood, and the exterior ray gate binds silhouette/opacity/gap
agreement. Per-triangle identity or capacity may be diagnostic provenance,
but cannot be a rejection rule unless it is shown to control a visible metric.
For unit phase-curve tangent and corresponding unit phase normal `n_q`, the
exact structured-chart area element is

```text
dA = sqrt(1 + (beta_f dot n_q)^2) ds dh,
```

so even diagnostic emitted-area accounting includes this shear Jacobian.

The intersection is over continuous height. The compiler splits at all
critical slice heights—triangle vertices, births/deaths, tips, nearly
horizontal faces, and payload/normal discontinuities—and uses conservative
swept-triangle/interval bounds between them. A finite sample grid plus a
generic Lipschitz guess is not a certificate.

Primitive relays may manufacture different topology or bridge a real gap, so
retained first-visible agreement, connected-gap topology, and grazing
silhouettes remain measured gates. This is the correct next structural gate if
the same-primitive compiler is sparse: reauthoring a visually close persistent
surface is the slight quality relaxation, whereas increasing taps/modes is not.

### 3.1.2 Finite constructive certificate for a relay

The continuous intersection above does not require an unbounded height sample.
For one transformed source triangle, split its height range at its three vertex
heights.  On every resulting open interval the two triangle edges cut by a
horizontal plane are fixed, so both endpoints of the slice segment are affine
functions of `h`.  If their horizontal derivatives have norms at most `L_T`,
the segment moves by at most

```text
d_H(segment(h),segment(h')) <= L_T |h-h'|
```

in Hausdorff distance.  After splitting at every vertex, payload, and normal
event, take the union only of compatible triangles that remain active across
the whole subinterval.  The union then has the same bound with
`L=max_T L_T`, and distance to that union is `L`-Lipschitz in height.

For a phase-grid cell with centre `q_c`, phase radius `r_q`, one height
subinterval of width `Delta h`, and a sample at its midpoint `h_c`, the
following is therefore a sufficient continuous certificate:

```text
dist(q_c,C_m(h_c)) + r_q + 0.5 L Delta h <= epsilon.
```

It proves every phase point in that cell remains inside the positional
corridor for every height in the subinterval.  Intersect these certified masks
over all event intervals in `I_f`.  Nearly horizontal triangles have an
unbounded or very large `L_T`; they cannot silently make this certificate
green and are either isolated into their true short interval or routed to the
residual.

Normal compatibility uses the same construction rather than a post-hoc
average. For each fixed offline candidate normal cone, restrict `C_m` to
triangles whose transported source normal and categorical payload are
compatible with that cone, then apply the inequality above. A curve segment
may use the cone only where its own transported sheet normal lies in it. The
corridor is eroded by the phase-cell radius, curve-simplification error, and
the finite-atlas minimum feature radius before curve extraction.

Finally extract periodic compact piecewise-regular arcs or loops and verify
every segment and included endpoint against its certified normal/payload mask.
No loop/homology or filled-interior constraint is imposed on an arc. Endpoints,
crossings, and branches receive frozen categorical ownership and sidedness;
incompatible crossings are split between fields. This is a constructive
sufficient test: a passing curve really is a continuous relay sheet. Its
failure is not by itself an impossibility proof because the cell/cone
discretization is conservative; a decisive structural rejection additionally
needs the position-only optimistic superset to be negligible. A constructive
pass still proceeds to the exterior first-visible gate because proximity does
not preserve gap topology.

Normals and marks on that lateral boundary must also satisfy their declared
compatibility bounds. Requiring the whole solid product to lie inside `S_i`
would be meaningless for a zero-thickness authored ribbon; only the emitted
lateral surface requires the certificate. For a genuinely volumetric tube,
the compiler may use the stronger inner-solid certificate.

Each anti-tiling layer is certified against its own transformed authored
population target. It is invalid to prove one layer against the source and
then assume that a rotated/shifted second population inherits the same
owner/visibility correspondence. For every retained boundary patch the cook
records an owner-preserving source correspondence and measures first-visible
agreement over the exterior-ray gate, including grazing directions. The
continuous theorem below is exact for the **compiled lateral set**; proximity
alone is not misreported as exact source-mesh firstness.

The boundary certificate is deliberately one-sided. A core may omit source
material but may not widen, relocate, or invent a crisp foreground surface
beyond `epsilon`. Root attachments selected for the core must remain attached;
they cannot use the relaxed free-tip rule.

### 3.2 Lateral-only query

For world ray `r(t)=o+td`, invert the affine map:

```text
A_f^-1 r(t) = (q0+t p, h0+t eta).
```

Intersect `[0,R_max]`, the real scene cutoff, and `h in I_f`, producing
`[t_a,t_b]`. Put `q_a=q0+t_a p`.

For `s_p=||p||>0`, query the next marked curve crossing

```text
rho_Gamma,f(q_a,p/s_p)
  = inf {ell>=0 : q_a + ell p/s_p in Gamma_f}.
```

There is no inside/outside mask state and no cap-start special case. Starting
on a curve gives `rho=0`; otherwise the oracle returns the first later arc or
loop intersection. Store the winning two-sided sheet normal and owner/mark
categorically and return

```text
t_f = t_a + rho_Gamma,f / s_p
```

iff `t_f<=t_b`.

For `s_p=0`, the ideal rule is `q_a in Gamma_f ? t_a : MISS`. If `t_a=0` and
the origin is already on the sheet, that is the excluded camera-on-plant case.
The pole chart can encode the same record's zero-distance sentinel; it needs no
separate occupancy field. Because this is exact coincidence with a
zero-thickness sheet, the finite filtered gate also reports the ablation that
routes it to the residual rather than thickening it into a false cap.

### 3.3 Lateral exactness theorem

For `s_p>0`, let

```text
E = {ell in [0,s_p(t_b-t_a)] : q_a+ell omega in Gamma_f}.
```

`Gamma_f` is closed and the tested interval is compact, so `E` is closed. If
it is nonempty, it has a minimum. The map
`ell=s_p(t-t_a)` is strictly increasing, hence
`t_a+min(E)/s_p` is exactly the first world-ray contact with the compiled
sheet inside the true global height interval. If `E` is empty, the sheet is a
miss. This proof neither refers to a filled mask nor requires an arc to close
into a loop.

Every returned point is on the live world ray, lies on a marked ruled sheet
`A_f(Gamma_f x I_f)`, and—by the curve certificate—lies within the
declared tolerance of its assigned source surface with a compatible mark and
normal. No returned point lies on a filled field cap. Therefore this query
cannot create:

- a horizontal or tilted cap plane;
- a floating copy of the terrain;
- camera-front widening caused by relifting an old-view hit;
- an interval-ineligible first event that hides a later event; or
- a wrong species/normal produced by arithmetic blending.

Taking the categorical minimum over a fixed core field set returns the exact
first member of the compiled lateral interaction set. A cap-facing or omitted
source ray is intentionally delegated to the residual rather than falsely
reported as crisp geometry.

### 3.4 Finite sampled curve field

The theorem above concerns the ideal continuous boundary-crossing function.
One runtime read means a finite sampled atlas, so unconditional firstness at
silhouettes, tangencies, owner changes, and component changes is not implied.

The stored domain remains only

```text
(live phase q, live in-plane direction omega),
```

not camera elevation. Each texel is a complete categorical curve-crossing
record. No separate inside/outside read, entry/exit state, or doubled
occupancy field is hidden in the cost.

Tangent reintersection is exact only for a straight curve segment. A cell may
be certified for reintersection only if, over its entire footprint, the same
first arc segment, owner, normal/sidedness, and endpoint branch win;
`|n dot omega|>=gamma_0>0`; the hit retains both an axial-slab margin and an
arc-endpoint/arclength margin larger than its lifted error; and local curve
curvature has a certified tangent remainder `E_curve`. The curvature
contribution to crossing-distance error is at most `E_curve/gamma_0`.
Tangencies, endpoints, crossings, branches, owner/component changes, hit/miss
transitions, and cells without either margin are unsafe categorical cells, not
exact records. The atlas may not extend a sampled segment's supporting line
past its real endpoint.

For phase radius `delta_q`, in-plane angular radius `delta_omega`, local
first-boundary reach `rho`, and transverse affine norm `kappa_f`, support
displacement is bounded by

```text
kappa_f (delta_q + rho delta_omega)
```

away from visibility discontinuities. Total retained-core/source displacement
is `epsilon_source+epsilon_atlas`, with the latter including this support term
and the curvature/incidence remainder. Certified cells have a local free-path
bound, not a camera-range `r DeltaTheta` term. Unsafe categorical cells can
still form full-contrast constant-screen angular seams; dense angular sweeps,
connected-region size, and millimetric temporal changes are binding gates
there. The implementation may not call a sampled atlas exact merely because
the continuous field is exact.

## 4. Direction-analytic residual

### 4.1 Authored meaning

The residual is the appearance contribution deliberately not represented by
the lateral core: cap-facing blade area, curved/tapered tails, minor foliage,
panicle fuzz, moss fuzz, and other transfer content selected by the compiler.

It is an authored joint transfer approximation to the accepted source, not a
claim that the opaque source triangles were physically a homogeneous volume.
Its fit is judged against filtered reference coverage/RGB and every partial
ray cutoff used by compositing.

For a declared filtered ray footprint with opaque-reference coverage
`A_ref(a,b)<1`, define the transfer target

```text
tau_ref(a,b) = -log(1-A_ref(a,b)).
```

Coverage indistinguishable from one would demand unbounded optical depth and
is not residual material: it must be retained by the crisp core or reported as
a compilation failure. In finite-precision fitting, the maximum admitted
residual coverage and corresponding finite `tau_max` are frozen explicitly;
they are not obtained by silently clamping an opaque landmark into fuzz.

### 4.2 One plane-free joint residual

Multiple hard height strata would multiply clipping, colour composition, and
mode/window work, and their boundaries could reproduce the plane artifact
being removed. The selected default is therefore one joint soft residual per
compiled community.

For anti-tiling layer `l`, work in its root-local rest coordinates. Let
`H=[z^-,z^+]`, `u=(z-z^-)/(z^+-z^-)`, and use the zero-at-boundary quadratic
window

```text
w(u) = 4u(1-u),       0<=u<=1.
```

The extinction and premultiplied source-colour densities are

```text
kappa_l(x)
  = 1_(z in H) w(u)
    (a_0 + 2 Re sum_(j=1..K_l) c_j exp(i(k_j dot x + phi_j))),

j_l,rgb(x)
  = 1_(z in H) w(u)
    (b_0,rgb + 2 Re sum_(j=1..K_l) d_j,rgb exp(i(k_j dot x + phi_j))).
```

The window makes both densities continuous and zero at the two support planes;
there is no residual opacity sheet at either plane. Horizontal wavevectors
belong to the reciprocal authoring lattice; vertical components may provide
additional within-window variation.

Positivity and `0<=j_rgb<=kappa` are certified offline. A convenient positive
authoring form is `kappa_base=|psi|^2`, expanded once before the live-mode
count is charged. The total live count is the actual sum after positivity
expansion and after evaluating both anti-tiling layers:

```text
K_live = sum_l K_l.
```

It is not multiplied invisibly by species, colours, or cover types. Those are
jointly fitted into `j_rgb` offline. Recognition-critical purple glumes,
anthers, white petals, and other opaque landmarks remain categorical core
content rather than being trusted to the soft mixture.

### 4.3 Exact partial-ray integrals of the compiled residual

Define world-length density by pullback,

```text
kappa_world(x) = kappa_rest(A_l^-1 x),
```

so integration remains with respect to unit-world-ray parameter `dt` and no
unwritten affine path-length Jacobian exists. Explicitly inverse-transform the
live ray as `o_hat=A_l^-1 o`, `d_hat=L_l^-1 d`. Intersect it with `H`,
`[0,R_max]`, the scene cutoff, and the nearest opaque-core hit, producing one
interval `[a,b]`.

Without the quadratic window, one complex mode has

```text
T = b-a,
m = (a+b)/2,
lambda_j = k_j dot d_hat,
psi_j = k_j dot (o_hat+m d_hat) + phi_j,
z_j = lambda_j T/2,
I_0,j = T exp(i psi_j) sinc(z_j),
sinc(z)=sin(z)/z, sinc(0)=1.
```

Because `w(u(t))` is quadratic in `t`, its weighted integral is a fixed linear
combination of

```text
I_n,j = (-i)^n partial^n I_0,j / partial lambda_j^n,
n in {0,1,2}.
```

The opacity centroid additionally uses `I_3,j`. All derivatives have stable
series at `z=0`; they are exact poles of the same analytic functions, not
direction clamps. Constant terms use ordinary polynomial antiderivatives.
Explicitly, if `u(t)=u_0+nu t` on the clipped ray, then

```text
w(u(t)) = 4[(u_0-u_0^2) + nu(1-2u_0)t - nu^2 t^2].
```

Thus the window introduces no sampling dimension, numerical integration, or
runtime iteration: it is exactly three fixed raw moments for opacity/RGB and
one additional raw moment for the optional centroid.

Applying those fixed moments to `a,c` yields optical depth `tau_l`; applying
the identical moments to `b,d` yields the unattenuated RGB source integral
`J_l,rgb`. The two transformed-layer results add:

```text
tau = tau_0 + tau_1,
J_rgb = J_0,rgb + J_1,rgb.
```

The corresponding opacity centroid is a soft-transfer statistic with a
declared spread. It is never inserted into categorical core ordering as if it
were a first-hit surface.

The integrated density gradient may reuse the same phases and moments, but it
must include both terms

```text
grad_hat(kappa) = w grad_hat(f) + f w'(u) grad_hat(u),
grad_world(kappa) = L_l^(-T) grad_hat(kappa).
```

The first multiplies spatial coefficients by `i k_j`; the second uses the
linear `w'` moments and adds no new phase evaluation. Omitting either the
window-gradient term or inverse-transpose transport is invalid. This remains
a soft-transfer normal with charged fixed ALU, not a categorical triangle
normal.

### 4.4 Joint colour approximation

Define

```text
c_bar = J_rgb / tau,
alpha = 1-exp(-tau),
C_approx = c_bar alpha.
```

This is exact for a constant-colour compiled residual. For internally mixed
colours, it is the declared opacity-weighted mean-colour approximation. If the
per-channel colour range is `Delta_c`, its internal-ordering error obeys

```text
|C_exact-C_approx|
  <= Delta_c tau(1-exp(-tau))/4
  <= Delta_c tau^2/4.
```

Therefore a high-contrast mixed residual must remain optically thin. Dense or
recognition-critical coloured structure moves to the categorical core or the
candidate fails; it may not be averaged into green. This is the deliberate
quality concession that avoids extra per-colour strata, reads, and mode
multiplication.

### 4.5 Partial-cutoff stability

Let `tau_ref(a,b)` be the filtered reference residual's optical depth/coverage
transform and `tau_fit(a,b)` the analytic fit. If the offline gate proves

```text
sup_(declared exterior domain, all a<=b) |tau_fit(a,b)-tau_ref(a,b)| <= E_tau,
```

then, because `exp(-x)` is one-Lipschitz for nonnegative `x`,

```text
|T_fit-T_ref| <= E_tau.
```

Coverage alone does not certify colour. Let `cbar_fit=J_fit/tau_fit` and let
`cbar_ref` be the reference residual's opacity-weighted mean colour on the
same cutoff interval. If colours lie in `[0,1]` and

```text
sup |cbar_fit-cbar_ref| <= E_c,
```

then the difference between their mean-colour transfer approximations is at
most `E_c+E_tau` per channel. Add the Section 4.4 internal-ordering bounds for
the fit and reference to bound their actual premultiplied RGB. Thus every
partial cutoff is certified in both optical depth and colour; an excellent
coverage fit cannot hide a disappearing purple head.

A finite ray set reports a measured maximum, not this supremum. A certified
claim additionally needs a Lipschitz/interval remainder covering the spaces
between sampled rays and cutoffs.

For the actual opaque-mesh compilation gate, the fitted object is the
**complete positive hybrid transfer**, never a numerically subtracted
"source minus core" field.  On one infinitesimal exterior ray, let `t_s` be
the first source hit, `t_c` the first compiled-core hit, and `b` a physically
reachable prefix cutoff measured from the camera/slab entry.  Integrate the
positive residual only to `min(b,t_c)`.  The candidate alpha is

```text
A_H(b) = 1-exp(-tau(b))       when t_c > b,
         1                    when t_c <= b,
```

and its premultiplied colour is

```text
C_H(b) = C_R(b)                                      when t_c > b,
         C_R(t_c) + exp(-tau(t_c)) c_core             when t_c <= b.
```

This complete result is compared directly with the opaque-source transfer
and only then averaged over the identical perspective pixel-ray bundle.
Optical depth, alpha, or colour are never subtracted to manufacture a
residual target.  In particular, a positive residual cannot repair an early
or false core hit.  If `t_c<t_s`, every cutoff in `(t_c,t_s)` has candidate
alpha one and source alpha zero; that interval is an analytic infeasibility
lower bound for the compiled hybrid.  It is reported, not automatically turned
into a visual RED: along-ray displacement is scored separately from transverse
screen displacement, and an early hit within the declared crisp compositing
allowance (currently `5 cm` p95) is the permitted small quality relaxation.
A source miss paired with a core hit, or an early-core region exceeding that
depth allowance over visible screen area, remains binding.

The opaque-source filtered alpha as a function of cutoff is the depth CDF

```text
A_S(b) = measure { subrays u : t_s(u) <= b }.
```

A smooth residual produces a continuous function of `b`.  If omitted opaque
geometry creates a CDF jump of mass `m` and no categorical core event
reproduces it, every continuous residual has uniform cutoff error at least
`m/2`.  A jump covering visible screen area and separated beyond the declared
depth/compositing tolerance is therefore a representation failure; a truly
subpixel event or a microscopic along-ray interval within that allowance is
judged only after the common anisotropic footprint removes the corresponding
visible degree of freedom.  The gate tests camera-prefix cutoffs immediately
before and after source/core events, but reports depth and image-transfer error
separately.  It does not demand arbitrary suffix intervals that discard an
earlier opaque hit, and it never assigns infinite `tau_ref` to an individual
opaque subray.

Discrete `N`-subray quadrature creates artificial CDF jumps of `1/N`.  Any
supremum-over-cutoff or atom lower-bound claim must demonstrate bundle
convergence or carry a conservative quadrature remainder.  Positivity
`0<=j<=kappa` and the declared residual optical-depth ceiling are checked over
all admitted oblique and horizontal paths; a modest vertical density may
otherwise become falsely opaque through the `1/sin(elevation)` grazing path
length.

The residual formula remains valid when the origin lies inside its soft
support: use `a=0` after the ordinary forward/slab clipping. The camera-inside
fade applies only when the camera actually enters crisp plant matter; it is
not a license to fade ordinary air gaps merely because their height lies
inside the overall cover slab.

## 5. What the quality relaxation actually means

### 5.1 No distance-amplified reconstruction error

Ideal-core hits are computed on the live ray. Residual integrals use the live
ray analytically. Neither path samples camera elevation. Consequently the
continuous model has no mechanism for a direction-cell error to create a
world sector of width `r DeltaTheta`.

The finite core atlas still samples in-plane direction. Certified
same-boundary cells inherit the local bound in Section 3.4; unsafe categorical
cells can retain constant-screen owner/hit/miss seams. Those seams are neither
handwaved as subpixel nor accepted as inherent: the actual atlas must pass
dense angular, connected-region, and translation gates at its frozen
resolution and memory. Failure returns to the representation rather than
adding a filter.

Any core/source discrepancy is a static world-space authoring discrepancy.
For transverse displacement `epsilon` viewed at distance `D`, its angular
size is

```text
atan(epsilon/D)
```

and its approximate pixel size is `epsilon/(D theta_pix)`. It shrinks with
distance. Residual error is bounded directly in filtered coverage/RGB by
`E_tau`; it does not relift into a false surface.

### 5.2 Pixel-scale law

At `55 degrees / 2160 physical pixels`,

```text
theta_pix approximately 0.000444 rad.
```

For a feature whose **projected transverse width in the current view** is
`w_perp`, one pixel occurs near `D=w_perp/theta_pix`. The often-useful `3 mm`
and `6 mm` examples give `6.8 m` and `13.5 m` only when their full width is
actually transverse to the view. Edge-on ribbons can become subpixel much
sooner; a broad panicle silhouette can remain resolvable much farther away.
Beyond the measured feature's subpixel distance, preserving every microscopic
endpoint is the wrong metric: the reference and candidate must be filtered to
the identical anisotropic pixel footprint and compared by joint coverage and
premultiplied colour. This does not waive coherent gaps, silhouette drift,
colour loss, temporal shimmer, or any artifact of constant screen-space size.

The implementation of that statement is not a scalar `world_error/D` test.
Let `e_x,e_y` be the camera's horizontal and vertical screen axes at a live
ray, and let `theta_x,theta_y` be the corresponding radians per physical
pixel.  The first-order screen displacement of a world-space error `delta`
at range `D` is

```text
J_D delta = ((delta dot e_x)/(D theta_x),
             (delta dot e_y)/(D theta_y)).
```

`||J_D delta||` is the displacement in physical pixels.  In the plane normal
to the live ray, the one-pixel world footprint is the parallelogram generated
by `D theta_x e_x` and `D theta_y e_y`.  On a locally planar source surface
with normal `n`, the corresponding hit-point differentials are instead

```text
a_x = D theta_x (e_x - d (n dot e_x)/(n dot d)),
a_y = D theta_y (e_y - d (n dot e_y)/(n dot d)).
```

This is the ordinary ray-plane differential and it retains the grazing-angle
stretch rather than pretending a ground footprint is isotropic.  At a
silhouette, where `n dot d` tends to zero and this local chart becomes
singular, the binding object is the common screen-space ray bundle itself;
source and candidate coverage are integrated over identical pixel samples or
an equivalent conservative footprint.  The test never divides by a clamped
incidence to make a silhouette disappear.

Every reference/candidate comparison uses that same anisotropic footprint
(with the actual reconstruction-filter support applied on top of it); it does
not replace it with an isotropic disk or the feature's nominal authored width.
A silhouette is filtered principally across its projected normal, so a
ribbon's long tangent does not falsely keep an edge-on width resolvable.
Conversely, a broad panicle or connected gap whose projected normal extent
spans several pixels does not become acceptable merely because its individual
glumes are small.

This gives two deliberately different distance decisions:

1. a bounded static world-space compilation or atlas error is allowed to fall
   below the common filtered footprint as `||J_D delta||` decreases;
2. an address/owner discontinuity that sweeps a region of fixed angular width,
   a repeating wedge, or a temporally unstable winner is measured directly in
   screen pixels and does **not** gain a `1/D` waiver.

The CPU gate therefore emits both world-space error and filtered screen-space
error at every tested range.  It may relax microscopic geometry only where the
source itself has lost that visible degree of freedom under the same filter.

The residual has an additional exact distance-aware simplification.  For a
local parallel-ray footprint spanned by world vectors `a_x,a_y`, a Fourier
mode `exp(i k dot x)` is multiplied by the box-filter transfer

```text
H(k) = sinc(0.5 k dot a_x) sinc(0.5 k dot a_y).
```

For the true perspective ray bundle those spans vary along the interval; the
CPU reference integrates that bundle directly.  A runtime near/mid/far
specialization may omit a mode only when a conservative bound over its whole
admitted ray interval proves

```text
2 |c_j| integral w(t) |H_t(k_j)| dt <= E_lod,j.
```

The sum of omitted `E_lod,j` is charged to the residual transfer budget.  Thus
`K_live=4/2/0` distance specializations can save distant ALU without pretending
that distance erases still-visible content: a broad or low-frequency mode is
kept until its own anisotropic filtered contribution is small.  The choice is
a fixed coherent specialization, not a march or a content-dependent candidate
loop.

Nearer than the feature's sub-pixel distance, exposed blade/culm silhouettes
remain crisp obligations. They cannot be moved `50–100 mm` merely because the
old field-count bound demands it.

### 5.3 One-sided free-tip relaxation

For a genuinely tapered free tip with local width

```text
w(s) <= gamma s
```

measured inward from the tip, removing a length `delta` removes at most

```text
A_removed <= gamma delta^2 / 2
```

in its own projection plane. This quadratic area law is a legitimate soft
quality metric after pixel filtering. It does not apply to a root or branch
attachment: removing an attached interval costs `O(w delta)` area and may
change topology. Attachments and recognition-critical landmarks must be kept
in the core or represented visibly by the residual.

### 5.4 Population fidelity, not mesh identity

The compiler is allowed to abandon one-to-one correspondence with all
`270,541` source recipes. It is not allowed to abandon species recognition,
the green/purple/cream palette, tall-versus-low stature, head envelope,
attachment cues, or overlap behaviour.

The quality claim, if the gate passes, is therefore:

```text
high-fidelity species/community appearance under exterior flight,
not exact reproduction of every source triangle or endpoint.
```

## 6. Fixed resource model

Let `F` be the number of lateral core fields and `L=2` the anti-tiling layers.
Each core lookup must return a complete packed candidate record, or its
additional attribute read is charged explicitly.

```text
T_cover = 2 F + T_control/material <= 9.
```

The primary Pareto sweep is `F in {2,3,4}`. With one existing control/material
read, `F=4` exactly reaches nine reads. The same `F` atlases are reused by both
layer transforms, so layers do not duplicate resident bytes.

Residual coefficients occupy only a small constant/uniform block. They add no
texture, buffer, pass, dispatch, barrier, or synchronization. They do add
fixed ALU and special-function work. Freeze the total retained spectral mode
count before implementation and reject quality rather than increasing it past
the measured low-end performance envelope. The first CPU gate must report the
quality curve for `K_live in {0,1,2,3,4}`; a later GPU trace decides whether
even four modes are affordable. No result may call unmeasured transcendental
cost free.

The active periodic chart is structured, not a generic 3x3 affine inverse.
For layer `l`, rotate/translate the horizontal origin and direction once, then
each field needs only

```text
q0_lf = R_l o_xz - beta_f o_y + delta_l,
p_lf  = R_l d_xz - beta_f d_y,
h0=o_y, eta=d_y.
```

Thus `h0,eta`, the layer rotation, and part of clipping are shared; each field
adds four shear multiply-adds before normalization/addressing. The earlier
generic-affine estimates (`112–256` scalar operations for the core) are
conservative non-binding upper bounds, not the selected implementation cost.
The quadratic-window residual still adds fixed polynomial moments,
exponentials, and up to `K_live` phase evaluations. The CPU/compiler gate must
emit the exact expanded structured count, and the eventual GPU trace remains
binding. `K_live>4`, hidden per-colour evaluations, or replicating modes after
positivity expansion is forbidden; a quality miss is reported instead of
buying correctness with unapproved shader work.

Memory remains the sampled first-passage storage for at most four core fields,
their mips/records, and existing control data. The residual adds negligible
resident memory. Exact byte accounting remains a gate because record packing,
minification, and gutters are not yet frozen.

## 7. Many species, cover types, moss, wind, and terrain

- Species are marks and source components inside the fixed core fields. The
  compiler partitions by affine compatibility and ownership, not by a runtime
  species loop.
- Residual densities from multiple species are compiled offline into the one
  joint extinction/RGB field. Strongly different interleaved colours remain a
  measured optically-thin approximation unless moved to categorical core
  structure.
- Moss hummock shape belongs to packed terrain microtopography. Crisp moss
  landmarks may occupy a low core cohort; only its fine fuzz belongs to the
  joint residual.
- White flowers, coloured heads, and other recognition-critical non-green
  structures may not be averaged into foliage. They require a categorical
  core mark; adding separately evaluated colour strata is not an uncounted
  escape hatch.
- Height-proportional wind is applied in root-local rest space. Invert the
  same globally affine layer transform on the live ray before both the core
  and residual queries. This is an exact affine conjugation.
- Curved terrain uses the bounded chart contract below; "locally flat" by
  itself is not a proof and is especially unsafe for grazing rays.

### 7.1 Bounded curved-terrain contract

Let the same packed, seam-consistent terrain surface used by vegetation roots
be a `C1` height field `g` on an admitted chart, with Hessian norm bounded by
`kappa_T`. For field shear `beta_f`, the query-independent authored embedding
for this candidate is

```text
Phi_f(s,h) = (s_x+beta_f.x h, g(s)+h, s_z+beta_f.z h).
```

Here `s` is world-horizontal root/phase and `h` is root-local gravity height.
Ground cover is not rotated independently with each terrain normal. Its root
phase and shear are query-independent and therefore cannot drift between
views. The live query receives a terrain anchor `x_*` from that same
surface query; it may not choose a camera-cell centre or a discontinuous
nearest tile. The local affine query is the tangent approximation to `Phi_f` at
`x_*`. Different rays may have different anchors only because every such
affine map is compared to this one fixed world-space `Phi_f`, never treated as a
different plant embedding.

The approximation must have a finite certified interaction horizon. Define:

```text
R_core = largest admitted first lateral-boundary reach of any core field,
R_res  = smallest reach after which either
         (a) accumulated foreground optical depth >= -log(E_tail), or
         (b) all omitted residual optical depth <= E_tail,
R_chart = max(R_core,R_res).
```

These are source/cook certificates, not runtime searches. A core atlas record
beyond `R_core` is a miss. The analytic residual may truncate at `R_res` only
when one of the two stated transfer bounds is true for every admitted ray;
otherwise this candidate fails rather than silently assuming that grazing
content is nearby.

On the disk of radius `R_chart`, the tangent-chart height discrepancy obeys

```text
E_height <= 0.5 kappa_T R_chart^2.
```

Because `Phi_f` uses gravity-up placement, there is no unaccounted
`H_cover kappa_T R` terrain-normal frame-rotation term and no horizontal phase
error. The gate projects the vertical bound through the actual live view and
pixel footprint, including grazing incidence; it also measures the difference
between two admitted anchor approximations to the same `Phi_f` (bounded
conservatively by twice the single-chart error) and checks root agreement at
packed-terrain seams. If the packed surface cannot provide a continuous
anchor over this radius, or if the curvature error is multi-pixel where the
plant feature remains resolvable, the terrain version is red. No neighbouring
chart loop, chart blend, floating carrier, or extra geometry may be introduced
to hide that result. A future terrain-normal botanical model would require the
omitted frame-rotation/top-displacement proof explicitly; it is not smuggled
into this candidate.

## 8. CPU gate required before implementation

### 8.1 Candidates

Compile and score, on the accepted Calamagrostis and a dense low/moss
community:

1. run the cheap same-primitive/global-band construction as a lower bound;
2. if that construction is sparse, compile the Section 3.1.1 persistent
   owner-compatible slice corridors for `F=2,3,4`, then combine them with
   `K_live=0..4` analytic residual modes;
3. an ablation with omitted residual, proving what the analytic component
   actually contributes;
4. a one-geometric-layer `F<=8` alternative only as an explicit Pareto point:
   it spends the same nine reads on more crisp cohorts while sacrificing the
   second quasiperiodic geometric population. World-keyed tint/vigor is not
   misreported as equivalent geometry variance.

The third candidate is not the default and requires an explicit user quality
decision if it wins.

### 8.2 Non-negotiable artifact gates

- no returned crisp cap event;
- no false crisp point outside the certified source tolerance;
- no camera-centred fan, widening sector, floating plane, field-edge plane, or
  distance-growing stretch;
- exact horizontal and vertical formulas remain finite;
- no categorical owner/normal/mark interpolation;
- no colour disappearance of the reproductive head with distance;
- millimetric camera translations do not create a coherent class-change band;
- residual partial-cutoff error is tested before and behind opaque core hits.
- finite-atlas sweeps include unsafe categorical cells, incidence margins,
  terrain-curvature extrema, and the same-arc/endpoint/slab-margin
  certificates.

### 8.3 Distance-aware quality report

Report, rather than hide behind one scalar:

- near-field crisp first-hit coverage retained by the lateral core;
- silhouette IoU and connected omission-region size;
- RGB max-channel p95, split into foliage, reproductive colour, and moss;
- recognition-landmark displacement and attachment preservation;
- residual `E_tau`, `E_c`, premultiplied RGB error, centroid/spread error, and
  optically thin colour-interleaving bound;
- the same metrics after filtering both source and candidate to the exact
  pixel footprint at increasing distances;
- unforced class-change rate under `1–4.5 mm` translations;
- reads, complete atlas bytes, coefficient bytes, and fixed operation count.

The starting quality target remains the existing `IoU>=0.97`, RGB p95
`<=0.15`, connected wrong regions `<1%`, and class change `<5%`, but the gate
must also emit the complete Pareto curve. If a miss is confined to already
sub-pixel residual content, the pixel-filtered metric governs. If it is a
multi-pixel near-field landmark, calling the threshold too strict is not an
acceptable diagnosis.

### 8.4 Decision rule

- If an `F<=4`, `K_live<=4` two-layer candidate passes, proceed to a baker and
  runtime transcription of exactly this math.
- If only the one-layer `F<=8` candidate passes, surface the loss of
  quasiperiodic geometry variance to the user before implementation.
- If both fail because multi-pixel crisp structure remains residual, the
  simultaneous nine-read/two-layer/no-pointed-field contract is incompatible
  with this plant. Do not add modes, reads, events, or direction cells. Record
  the measured blocker and return to the representation/quality decision.

### 8.5 Attempt budget

The structural/global-band pre-gate is prerequisite evidence, not an
end-to-end success. If it is not decisively red, run one optimistic full-ray
attempt using exact retained source surfaces plus the joint 3D residual. After
one diagnosis/fix rerun, a remaining red result parks this hybrid. Only an
optimistic green justifies constructing the finite curve atlas. That atlas
then receives one complete angular/terrain/translation attempt and one
diagnosis/fix rerun. A second red parks the track with its artifacts and exact
resume condition. No sequence of local filters or extra infrastructure may
extend those limits silently.

### 8.6 Terminal CPU result and exact theory boundary (2026-07-23)

The strict compiler produced `379` continuously height-certified phase-arc
edges (`315` foliage and `64` purple/brown reproductive) across four global
height fields.  The second exterior attempt evaluated the real accepted
Calamagrostis source and that core under both Tier-1 affine populations, with
two positive residual modes per layer (`K_total=4` after expansion), identical
perspective bundles at `1,2,4,8,16,32 m`, full periodic-phase stratification,
millimetre translations, exact horizontal queries, all-event prefix checks,
and `4x4 -> 8x8` convergence.

That **fitted coefficient set** is decisively RED:

```text
top-down silhouette IoU       0.278--0.306
RGB max-channel p95           0.421--0.553
largest wrong phase region    66.7--72.2%
fitted prefix-alpha p95       approximately 1
```

At one certified source/core-empty height and phase, all four cardinal
horizontal rays remain empty for `4 m`, while the fitted residual is opaque.
For the selected coefficients, the bracket has the global positivity floor
`kappa_min=0.46075846498` and the quadratic window at `u=7/16` is `63/64`, so

```text
tau >= 4 * 2 * (63/64) * 0.46075846498 = 3.62847291172,
alpha >= 1-exp(-tau) = 0.97344329210.
```

This is an exact renderer-independent rejection of the fitted candidate.
It is **not** a coefficient-independent rejection of every admissible
two-layer `K2+K2` residual. Attempt 2 optimized one layer and then applied the
result to both affine populations rather than solving the joint final-union
minimax problem. More decisively, arbitrary vertical frequencies admit the
legal nonnegative two-mode notch

```text
P(y) = a [1-cos(lambda(y-y0))]^2
     = a [3/2 - 2 cos(lambda(y-y0))
              + 1/2 cos(2 lambda(y-y0))].
```

It vanishes on the entire horizontal plane `y=y0` while retaining positive
top-down optical depth elsewhere. Near that plane,

```text
[1-cos(lambda delta_y)]^2 <= lambda^4 delta_y^4 / 4,
```

so the escape remains quartically small for a finite millimetric pixel bundle.
Consequently one empty horizontal cross plus the measured mean top-down
coverage cannot support a general no-go theorem; the strongest universal
horizontal-opacity lower bound from those premises is zero.

The track is parked under Section 8.5.  Its objective resume condition is
either:

1. a globally or near-globally optimal **direct joint** final-two-layer
   `K2+K2` minimax fit over a frozen finite reciprocal-frequency catalogue,
   with positivity, `0<=j<=kappa`, exact-horizontal air queries, and all
   source/core event-adjacent prefix constraints included; or
2. positive-measure empty constraints over multiple heights and phases plus a
   frozen vertical bandlimit/coefficient bound sufficient for a quantitative
   Remez/Turan lower bound.

A third heuristic frequency ranking is not a resume condition. Runtime, atlas,
shader, and WebGPU work remain unauthorized. Full metrics and hashes are in
`GRASS-LATERAL-CORE-RESIDUAL-CPU-GATE.md` and
`data/work/groundcover-relay-exterior-ray-gate/37b0cf1d33f632b5/9d993b97802dc47d/a8ca42a9b2901f1b/`.

## 9. Provenance and contribution boundary

This note builds on:

- Sannikov's precomputed repeating-ray objective and the known vertical-
  extrusion elevation reduction;
- the local `GRASS-EXACT-REPRESENTATION-THEORY.md` amplification dichotomy;
- the cap/field lower bound in `CLASS-E-CPU-GATE-RESULT.md`;
- the actual-source fixed-`K`, angular-flow, coupled-sheet, and shell-frame
  rejection records under `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/`;
- the accepted structural/plume visual-source gate in
  `GRASS-STATUS-AND-ISSUES.md` Section 34; and
- standard closed-form integration of finite complex exponentials along an
  affine line.

The endpoint-successor trilemma, compact-arc ruled-sheet generalization,
cap-free source-contained lateral query, fixed-cost core/residual
decomposition, partial-cutoff transfer bound, and distance-aware compilation
gate are LAAS-local derivations. If the resulting system succeeds, these are
the parts that require a precise source ledger and independent review before
any publication claim.
