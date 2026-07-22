# Class-E ground cover — mathematical audit before implementation

Date: 2026-07-22
Status: **RED as currently specified; a narrower opaque exterior theorem is
proved and reusable**
Scope: pure mathematics only. No shader, renderer, WebGPU, CPU harness, bake,
or runtime implementation is evaluated or authorized by this note.

## 0. Verdict

The class-E proposal contains one strong and useful result:

> A finite union of globally affine, lattice-compatible parallel-extrusion
> solids can be queried for its exact **exterior first contact** with one ideal
> continuous 2D-mask first-passage field with 3D domain per frozen family/band,
> an analytic
> `t = t_star + rho / s_p` lift, and a fixed minimum network.

That result is mathematically sound, including exact horizontal directions,
the axis-parallel pole, periodic cell crossings, and exterior overlap between
representable families.

The larger theorem in `GRASS-EXACT-REPRESENTATION-THEORY.md` is not sound as
written. The following failures affect the required **exterior** contract and
are structural, not implementation details:

1. a 3D record `V(q, omega)` cannot encode finite-slab volumetric transfer for
   all elevations or arbitrary opaque cut depths;
2. `rho Delta_omega` is a distance-to-support heuristic, not a bound on live
   first-event depth, owner, coverage, or hit/miss correctness;
3. the proposed Phase A measures free paths before the family masks whose free
   paths determine the budget exist;
4. displaced masks alone form stair-step prisms, not chord-following fibres;
5. the claimed small axis catalogue has no covering-number proof;
6. spatially varying terrain charts, wind, height modulation, and community
   control fields are not global affine conjugations and therefore do not
   preserve the reduction exactly; and
7. the claimed necessity/"sole surviving architecture" classification is not
   proved by the stated hypotheses.

Separately, the theory advertises exact camera-inside succession even though
the user contract explicitly excludes camera-inside quality and permits a
fade. That overclaim should simply be deleted; it is **not** a design blocker
and no replacement successor mechanism is required.

Therefore **do not begin the handoff's Phase A, B, C, or D under the current
specification**. The proposal is salvageable only after its contract is split
into:

- an exact, globally affine, opaque exterior core;
- explicitly bounded authoring/sampling approximations; and
- a separate plume representation whose dimensional requirements are stated
  honestly.

## 1. Provenance and audit boundary

The proposal audited here is defined by:

- `docs/tasks/2026-07-22/CLASS-E-IMPLEMENTATION-HANDOFF.md`;
- `docs/tasks/2026-07-21/GRASS-EXACT-REPRESENTATION-THEORY.md`, especially
  Sections 5–10, 12, and 13.3–13.4;
- `docs/tasks/2026-07-22/GRASS-SHELL-FRAME-FIELD.md`, especially Sections 11,
  13, and 14; and
- the negative-result ledger in
  `docs/tasks/2026-07-21/GRASS-STATUS-AND-ISSUES.md` Sections 18–44.

The proofs, counterexamples, and corrections below are LAAS derivations from
those definitions. They are not claims attributed to Sannikov or to the
external papers in the grass source ledger.

The audit deliberately distinguishes four objects:

1. the ideal continuous compiled solid `G*`;
2. the ideal continuous first-passage function of `G*`;
3. a finite sampled texture approximating that function; and
4. the authored reference plant from which `G*` is compiled.

Exact reconstruction of (1) from (2) does not prove that (3) approximates (2),
or that (1) resembles (4).

## 2. Precise opaque-family definition

Use canonical coordinates `(q, h) in R^2 x R`. Let:

- `M_f subset R^2` be a closed mask periodic under a rank-two lattice
  `LambdaHat_f`;
- `A_f(x) = L_f x + c_f` be invertible and affine;
- the two canonical transverse lattice vectors map bijectively to the world
  horizontal lattice:

  ```text
  L_f (lambdaHat, 0) = (lambda, 0_y),
  lambdaHat in LambdaHat_f, lambda in Lambda;
  ```

- `I_f = [b_f^-, b_f^+]` be a world-horizontal clip slab; and
- `R_max` be the finite world-ray horizon.

The actual closed family solid is

```text
G_f = A_f(M_f x R) intersect { x : b_f^- <= e_y dot x <= b_f^+ }.
```

This definition is important. A mask and an axis do not by themselves define
a finite plant; the clipped **solid** is the mathematical object whose first
contact is reconstructed.

For a world ray

```text
r(t) = o + t d,  ||d|| = 1,  t >= 0,
```

write its inverse-affine image as

```text
A_f^-1 r(t) = (q_0 + t p, h_0 + t eta),
s_p = ||p||.
```

Intersect the world slab and `[0, R_max]` to obtain the accepted parameter
interval `J_f = [t_star, t_max]`. An empty interval is a miss.

## 3. The theorem that actually survives

### 3.1 Ideal continuous field

For `s_p > 0`, let `omega = p / s_p` and define

```text
rho_M(q, omega) = inf { ell >= 0 : q + ell omega in M }.
```

At slab/horizon entry,

```text
q_star = q_0 + t_star p.
```

The candidate is

```text
t_f = t_star + rho_M(q_star, omega) / s_p,
```

accepted iff `t_f <= t_max`.

### 3.2 Proof of exterior first-contact exactness

For every `t in J_f`,

```text
q_0 + t p
  = q_star + (t - t_star) p
  = q_star + s_p (t - t_star) omega.
```

Membership of `r(t)` in the affine extrusion is therefore equivalent to
membership of the projected 2D ray in `M_f`. The first nonnegative projected
path `rho` lifts to the first world parameter by division by `s_p`. The slab
and horizon acceptance test is exact because both are linear intervals in
`t`.

For `s_p = 0`, the projected point is constant. The exact answer is `t_star`
iff `q_star in M_f`; otherwise it is MISS. This is a categorical pole, not a
limit implemented with an epsilon.

Thus:

> **Correct opaque-family theorem.** From an origin exterior to `G_f`, the
> ideal continuous field and analytic lift return the exact first closed-set
> contact of the ray with `G_f`, including exact horizontal and vertical world
> directions.

If tangential contact should not count, replace closed-set contact by an
oriented outside-to-inside crossing predicate. The current `inf` definition
counts tangencies.

### 3.3 Exterior union theorem

For a finite frozen family set and `o` exterior to the union,

```text
inf { t : r(t) in union_f G_f }
  = min_f inf { t : r(t) in G_f }.
```

Consequently, exact exterior firstness, order swaps, and overlap between
representable opaque families survive a fixed minimum/select network.

The result is O(1) only after hard upper bounds on layers, families, and bands
are frozen:

```text
runtime work = O(L * sum_f B_f), with L, f, and B_f compile-time constants.
```

That is genuine constant work per pixel. It is not evidence that the constant
is small or that it stays fixed as the fidelity target tightens.

### 3.4 Periodicity theorem

Under the precise lattice condition in Section 2,

```text
A_f(q + lambdaHat, h) = A_f(q, h) + lambda.
```

The horizontal slab is invariant under `lambda`, so each family is exactly
periodic and one toroidal first-passage function covers arbitrarily many
copies.

Two globally affine anti-tiling layers can each be queried exactly and
min-composed. A golden-angle rotated layer generally has an incommensurate
lattice, so the union is usually quasiperiodic, not periodic under one common
`Lambda`. This does not break the per-layer query, but the common-periodicity
claim must be removed.

### 3.5 Fixed horizon correction

The bake cannot use the query-dependent cap `R_max s_p`, because `s_p` is not
an atlas coordinate. A sufficient fixed canonical cap is

```text
rho_cap,f >= R_max * sup_{||d||=1} || P_uv L_f^-1 d ||.
```

At runtime, accept only when

```text
rho <= s_p (t_max - t_star).
```

This keeps the ideal field independent of the missing elevation coordinate
while preserving the declared finite world horizon.

## 4. Exactness claims that fail even before sampling

### 4.1 The defined field is not a successor field

**Scope note:** camera-inside rendering is not required and this section is
not part of the implementation no-go. It exists only to remove an inherited
false theorem so a later document does not accidentally restore that scope.

Let the one-dimensional mask, extended trivially in the second coordinate, be

```text
M = ([0,1] union [2,3]) mod 4,
omega = +e_u.
```

Immediately after entering the first interval, take `q = 0.1`. The proposal's
definition gives

```text
rho_M(0.1, omega) = 0
```

because `q in M`. The next entry is at `u = 2`, distance `1.9`. Therefore the
claim that the same `rho` returns the next entry "just past" a hit is false.

The identity

```text
rho(q + epsilon omega, omega) = rho(q, omega) - epsilon
```

is valid only for `0 <= epsilon < rho(q, omega)`, while the origin is still
strictly before the first contact. It proves exterior line-datum invariance,
not post-contact succession.

The current user contract excludes cameras inside the cover volume, so the
clean correction is to delete all interior/successor claims. If a future
contract needs the next surface crossing, the field must instead store a
one-sided next-boundary function such as

```text
rho_boundary(q, omega)
  = inf { ell > 0 : q + ell omega crosses boundary(M) }.
```

Even then, successor-of-each-family followed by `min` is not generally the
successor of a union for an origin already inside another overlapping solid.

### 4.2 Tangencies, corners, and tied marks need a convention

The closed-set `inf` reports tangent touches. At side/cap corners and exact
equal-depth family ties, the geometric normal or categorical mark may be
non-unique. A half-open ownership rule and deterministic tie priority are part
of the mathematical marked set; they cannot be left to texture or shader
accident.

### 4.3 The claimed necessity classification is unproved

Lemma 5.2 states that a line-preserving one-parameter affine group commuting
with the horizontal lattice and having straight orbits must be conjugate to a
fixed-axis translation. Its stated hypotheses are insufficient.

With `y` vertical and `(x,z)` horizontal, consider

```text
F_t(x,y,z) = (x + t y, y, z + t).
```

It is a one-parameter affine group. It commutes with every horizontal `x,z`
translation. Every point orbit is a straight line, but its direction
`(y,0,1)` depends on `y`. Its affine generator has a nonzero nilpotent linear
part, so affine conjugation cannot turn it into a pure translation generator.

Stronger hypotheses—free action by globally parallel orbits and a quotient
that maps every query line to a line—would recover the extrusion
classification. The current proof does not.

This does not weaken the sufficiency theorem in Section 3. It only removes the
claim that no other exact invariant could exist.

## 5. Finite texture sampling: the correct mathematics

The continuous field is not the runtime texture. Arbitrarily detailed masks
produce discontinuous first-event fields, so finite sampling needs a separate
theory.

### 5.1 A valid support-proximity lemma

Suppose a live query `(q, omega)` uses a stored sample `(q_i, omega_i)` whose
record is a hit at path `rho_i`. The sampled mask point

```text
x_i = q_i + rho_i omega_i
```

belongs to `M`. The reconstructed live projected point is

```text
x_hat = q + rho_i omega.
```

Therefore

```text
dist(x_hat, M)
  <= ||x_hat - x_i||
  <= ||q - q_i|| + rho_i ||omega - omega_i||.
```

After affine transport, multiply by the transverse operator norm of `L_f`.
This is a useful and fully general statement: a reported quantized hit lies in
a local tube around some actual compiled geometry.

It does **not** prove that the live ray truly hits, that the same owner wins,
or that the live first-event depth is close.

### 5.2 Event depth and owner need visibility conditioning

Let a unique regular first hit satisfy `F(x) = 0`, with unit boundary normal
`n`, and write a small angular change as `delta omega`. Implicit
differentiation gives

```text
delta rho
  = - n dot (delta q + rho delta omega) / (n dot omega).
```

The hit-point Jacobian is

```text
delta x
  = [I - omega n^T / (n dot omega)]
    (delta q + rho delta omega).
```

Hence a same-event local bound requires at least:

1. a unique first hit;
2. positive boundary reach;
3. transversality `|n dot omega| >= gamma > 0`; and
4. a positive depth/owner margin to the next competing event.

Then, locally,

```text
||delta x||
  <= C(gamma) (||delta q|| + rho ||delta omega||)
     + higher-order terms.
```

At a silhouette `gamma = 0`. With two separated mask components, an
arbitrarily small direction change can replace a near first hit with a far
one or MISS. There is no global first-event/owner bound of the form
`rho Delta_omega`.

### 5.3 The proposal misuses the coherence lemma in phase

The 1-Lipschitz identity holds only for a phase displacement parallel to the
same ray, `delta q = epsilon omega`, before the hit. A general 2D texel error
has a transverse component and can cross a silhouette or owner boundary.
Therefore nearest-phase error is not universally "subtexel" in event depth or
owner space.

The support-proximity lemma remains valid. Event correctness requires the
conditioning in Section 5.2 or a band-limited coverage metric.

### 5.4 What a defensible angular law can claim

Let nearest angular sampling have maximum angular error `delta_theta_max`, and
let nearest phase sampling have maximum transverse error `delta_q_max`. A
support-tube target can use

```text
E_support
  = ||L_transverse||
    (delta_q_max + rho delta_theta_max).
```

Require

```text
E_support <= k theta_pix D
```

at the reconstructed distance `D`. This controls proximity to compiled
geometry, not categorical firstness.

For the special orthonormal horizontal-slab case, put the camera clearance
`c > 0` above the slab top, slab height `h`, and downward elevation `alpha`.
Assume `0 < alpha < pi/2`, a vertical extrusion in an orthonormal frame, and
entry through the top slab. For an in-slab projected path `rho`,

```text
D(rho, alpha) = c / sin(alpha) + rho / cos(alpha),
rho <= h cot(alpha).
```

The coupled ratio obeys

```text
rho / D <= h cos(alpha) / (c + h) <= h / (c + h).
```

The exact-horizontal and exact-vertical cases use their separate categorical
or limiting charts. The inequality proves the grazing cancellation for the
**support tube**. If
`Delta_omega` denotes full sample spacing, nearest sampling has
`delta_theta_max = Delta_omega/2`; if it denotes cell radius, it does not.
The factor-of-two convention must be frozen explicitly.

This cancellation does not prove owner, hit/miss, or depth accuracy, and it
does not include the phase term, affine condition number, terrain curvature,
or minimum camera clearance.

### 5.5 A p95 free path is not a fidelity theorem

A p95 scalar permits a coherent catastrophic four-percent stripe, while the
declared connected-wrong-region limit is below one percent. A binding sampled
gate needs, per family and view regime:

- hit/miss and mark disagreement;
- largest connected disagreement region;
- p95 **and** p99 support, depth, and colour errors;
- worst tested direction;
- transversality/visibility-margin diagnostics; and
- stability under millimetric camera motion.

For a statistical angular budget, measure the joint amplification variable,
not a ratio of unrelated quantiles. A local same-branch example is

```text
Z = rho / (gamma D theta_pix).
```

Even `Z` does not cover silhouettes; those need filtered coverage and
connected-region gates.

### 5.6 Memory scaling correction

For one sampled angular coordinate,

```text
bytes
  = sum_f P_u,f P_v,f N_omega,f B_f bytes_per_record,f.
```

With phase pitch and record format fixed, `N_omega proportional 1/k`, so class-E
bytes scale linearly with `1/k`, not quadratically. Quadratic scaling belongs
to a representation sampling two angular coordinates unless another
explicit rule also changes phase resolution.

The quoted 32-slice illustrative allocation cannot justify a 128–730-slice
budget. For scale, one `256^2 x 730 x 8-byte x 4-band` family is about 1.53 GB
(1.43 GiB) before gutters, mips, metadata, or compression. Categorical marks and
non-filterable path data may not be assigned an assumed block-compression
ratio without an actual format/error contract.

## 6. Compilation mathematics

### 6.1 The handoff's Phase A is ordered incorrectly

`rho_f` belongs to a specific compiled tuple `(A_f, M_f, band_f)`. The
arbitrary source mesh has no family-independent class-E free path. Different
axis assignments and band masks produce different first passages.

The global community first hit also cannot size every constituent family: a
far candidate from one family can become the visible winner where another
family disappears or order swaps.

Therefore a binding direction/memory budget cannot be measured before at
least a provisional structural compilation defines the family masks. A
source-mesh free path can be reported only as a non-binding proxy.

### 6.2 Axis classes require a covering-number curve

For a straight primitive of length `L_i`, approximating unit axis `a_i` by a
class axis `a_hat` has best-centered transverse Hausdorff error at least

```text
e_axis,i >= (L_i / 2) sin angle(a_i, a_hat).
```

Thus tolerance `epsilon_i` requires

```text
angle(a_i, a_hat) <= asin(2 epsilon_i / L_i).
```

Natural population lean variance does not reduce an individual silhouette
error. The compiler must emit the weighted directional covering curve

```text
K(epsilon)
  = N_cover({a_i}, d_i),
d_i(a_i, a_hat) = (L_i/2) sin angle(a_i, a_hat).
```

If directions fill a two-dimensional cone, the worst-case covering count is
quadratic in inverse angular tolerance; if they lie on a one-dimensional
ring, it is linear. There is currently no proof that `K = 4..7` meets the
plant-fidelity target.

### 6.3 Displaced masks are not chords

A mask translated once per horizontal band and extruded under the same axis
forms a staircase of parallel prisms. It does not follow a chord within the
band.

For a centre curve `c(y)`, a true chord band `[y_i, y_{i+1}]` requires a
band-specific shear

```text
v_i = (c(y_{i+1}) - c(y_i)) / (y_{i+1} - y_i)
```

and geometry

```text
G_i = { (x,y) :
        y_i <= y <= y_{i+1},
        x - c(y_i) - v_i (y-y_i) in M_i }.
```

Adjacent bands join only if their endpoint cross-sections agree. If `v_i` is
quantized to a family axis `v_hat_i`, an additional band error is bounded by

```text
||v_i - v_hat_i|| Delta_y
```

(or half that under a best-centered placement, with endpoint mismatch moved
to both ends).

For `c in C^2`, the standard uniform chord bound is

```text
||c - c_chord||_infinity
  <= ||c''||_infinity Delta_y^2 / 8.
```

The `1/B^2` statement is therefore conditional on bounded curvature and the
actual chord construction. Dividing one measured whole-span residual by 16
does not prove the `B=4` residual when curvature is localized. Width, taper,
twist, cross-section, and normal variation require their own residuals.

### 6.4 Quantized tops are exact only for the quantized solid

Nested masks

```text
M_b = { q : tau(q) >= b }
```

over fixed slabs exactly reconstruct a stepped-top compiled solid when every
band is queried and min-composed. The required `B` is empirical.

A fixed `K'` list of 2D events with per-event top tests is not exact: choose
the first `K'` crossings to fail the height test and crossing `K'+1` to pass.
A fallback band can be exact for a different quantized target, but it may
select a different owner relative to the per-texel-tip target. The proposal
must not call this both bounded-rank exactness and never-wrong-owner.

Each band also needs its coupled boundary attributes. `B` path channels alone
are insufficient if normals, colour, or marks differ at the selected band
event; either store a complete record per band or perform an analytically
addressed attribute read after election.

### 6.5 Shared family masks need single-valued ownership

Putting many species into one same-axis mask is exact only when the visible
boundary mark and attributes are single-valued at each mask phase/band. If two
same-axis fibres with different marks overlap in projected phase, their
extruded solids merge and one texel cannot retain both axial orders.

The compiler must either:

- prove projected interiors disjoint;
- merge only identical ownership semantics;
- split conflicts into separate families/bands; or
- report a categorical compilation approximation.

Species need not appear as a runtime loop variable, but adding species can
increase mask resolution, conflict count, family count, bands, and resident
bytes. “No per-species query” is valid; “species add zero cost” is not a
general theorem.

## 7. Volumetric families: dimensional counterproof

### 7.1 What is correct

For an axially invariant density and `s_p > 0`, put

```text
ell = s_p (t - t_star),
K(L) = integral_0^L kappa(q_star + ell omega) d ell.
```

Then

```text
tau(t) = K(s_p(t-t_star)) / s_p.
```

The `1/s_p` change of variables is exact.

The correct premultiplied colour for a projected segment of length `L` is

```text
C(L,s_p)
  = (1/s_p) integral_0^L
      c(q_star + ell omega)
      kappa(q_star + ell omega)
      exp[-K(ell)/s_p]
    d ell.
```

The proposal's record suppresses both the finite endpoint `L` and the
`s_p`-dependent attenuation.

### 7.2 One 3D volume record cannot serve all elevations

Take a vertical slab `0 <= y <= H`, constant density `kappa_0 > 0`, constant
colour, and rays entering at the same `q_star` with the same in-plane
direction `omega` but elevation `alpha`:

```text
d_alpha = (cos(alpha) omega, sin(alpha)).
```

Every ray addresses the same proposed `V(q_star, omega)`, yet

```text
Delta_t(alpha) = H / |sin(alpha)|,
A(alpha) = 1 - exp[-kappa_0 H / |sin(alpha)|].
```

The correct opacity changes with elevation, while the proposal's stored total
`(A,C,mu,sigma)` record is identical. Therefore that record cannot be exact.
A different restricted model could store `kappa_0` and evaluate this
particular formula from live `alpha`; the counterexample rejects the proposed
baked total, not every possible analytic medium.

For nonconstant density, the projected endpoint

```text
L = s_p (t_max - t_star)
```

is independently necessary. A density patch beyond `L_1` but before `L_2`
distinguishes two queries with identical `(q, omega)`.

For unrestricted spatially varying `kappa` and `c`, exact transfer requires
the cumulative functions `K(q,omega,L)` and the corresponding colour-prefix
transfer; no fixed finite moment record determines them. For a full horizontal
slab entered from outside, elevation determines both `s_p` and projected
endpoint `L`, so one added elevation/end coordinate suffices for total
transfer:

```text
V(q, omega, endpoint/elevation),
```

a 4D field, or a sharply restricted analytic density class. With an arbitrary
opaque cutoff, the cutoff endpoint `L_o` is independent and requires an
additional prefix coordinate unless the analytic class is closed under that
prefix query.

### 7.3 Total moments do not compose exactly through opaque events

An opaque event at projected cutoff `L_o` needs the **prefix** transfer up to
`L_o`, not the total opacity, mean, and variance of the whole plume.

Finite moments do not determine a cumulative distribution. For example, on
`[0,2]` the interaction measures

```text
mu_1 = 0.5 delta_(1-sqrt(0.5)) + 0.5 delta_(1+sqrt(0.5)),
mu_2 = 0.25 delta_0 + 0.5 delta_1 + 0.25 delta_2
```

have the same total mass, mean `1`, and variance `0.5`, but different mass in
front of a cutoff at `0.5`. They therefore composite differently with an
opaque blade at that depth.

A red concentration in front of a black blade and blue concentration behind
it gives an immediate visual counterexample: ordering one total lump before
the blade wrongly includes blue; ordering it after wrongly removes red. The
error can be order one.

Exact arbitrary opaque/volume interpenetration requires a cumulative prefix
operator such as

```text
P(q, omega, elevation, L_o),
```

or an authored non-interpenetration restriction. One or two lumps are bounded
approximations, not exact composition.

### 7.4 Per-family volumetric mips lose cross-family correlation

Take a pixel with two equal subrays and two opaque families, each with coarse
alpha `1/2`.

- If both cover the same subray, true union coverage is `1/2`.
- If they cover complementary subrays, true union coverage is `1`.

The separate per-family coarse records are identical in both scenes.
Independent medium composition returns `3/4` for both. Therefore
opaque-to-volume mip duality is not exact across independently filtered
families without joint correlation data.

### 7.5 Honest plume options

The current exact theorem must remove arbitrary volumetric families. A later
design may choose one of these mathematically honest routes:

1. a 4D finite-segment transfer field for total exterior plume transfer, plus
   another prefix/cut coordinate if arbitrary opaque straddles remain;
2. a fixed finite analytic density basis with a closed-form line integral;
3. opaque class-E microfamilies for plume structure; or
4. an explicitly approximate volume model with bounded density/colour,
   authored non-overlap or depth slabs, and a screen-space error gate.

Only option 4 resembles the current one-read proposal, and it must be labelled
an approximation rather than part of the exact reconstruction theorem.

## 8. Position-dependent transforms break exact conjugation

### 8.1 Terrain

A spatially varying ground chart maps a world ray to a curve, not a line, in
canonical coordinates. One global first-passage field is exact only on a
global affine terrain chart.

For terrain `y = g(x,z)`, a local affine tangent approximation over horizontal
offset `r` has vertical remainder

```text
|delta g| <= 0.5 kappa_T r^2.
```

The corresponding ray-parameter error is locally

```text
|delta t|
  approximately |delta g|
  / |d_y - grad(g) dot d_xz|,
```

which diverges at terrain-tangent directions. The relevant `r` is the cover
query's horizontal travel/free path, not merely one guide-texel width. A
piecewise chart also needs an overlap/seam rule; two different affine maps
cannot agree on an open overlap.

Terrain attachment can therefore be a measured local approximation, but it
has a curvature, incidence, range, and chart-transition error term. It is not
in the exact tier.

### 8.2 Wind

The proposed wind map is

```text
W(x,z,y)
  = ((x,z) + beta(x,z,t) (y-g(x,z)), y).
```

It is affine only when `beta` is spatially constant and `g` is affine over the
query domain. A spatially varying `beta` bends the inverse image of a world
ray. Smoothness does not restore line preservation.

- A globally constant time-varying shear per layer is exact by conjugation.
- Spatial gust variation is approximate, with a displacement scale beginning
  at `h ||grad beta|| r` before incidence/visibility amplification.

The exact and approximate wind tiers must be named separately.

### 8.3 Vigor and height modulation

Tint modulation after event selection does not move geometry and can remain a
cheap world-keyed attribute variation. Spatially varying `+/-15%` height or
vigor changes family clips per root and is not a global affine transform. It
has the same finite-end/first-eligible-event problem as per-element axial
intervals unless compiled into discrete authored states or admitted as a
bounded approximation.

### 8.4 Community control-field boundaries

Selecting one atlas before the query is exact only when that community is
constant over every possible first-hit segment. If the starting/root region
is type A but the ray enters neighboring type B before any A hit, an A-only
query returns a farther hit or MISS.

Root-stable selection prevents key flicker at one root; it does not prove
cross-boundary firstness. Exact boundaries require one of:

- a precompiled boundary/supertile community;
- a profile constant over the query horizon;
- boundary/control support compiled into every queried field; or
- analytically clip-able support regions with a fixed complete region
  candidate set.

Plainly querying periodic A and B fields and taking their minimum is not
sufficient: each field would still return copies inside cells controlled by
the other profile. Complete support-aware candidates add per-state work. This
issue is separate from overlapping species already compiled into one
homogeneous community mask.

## 9. Corrected mathematical contract

### 9.1 Exact tier

The following theorem is defensible:

> Let `G*` be a finite union of globally affine,
> lattice-compatible extrusion solids, each clipped by finitely many global
> horizontal slabs, with single-valued axially invariant or closed-form
> attributes. Let the ray origin be exterior to `G*`. Given ideal continuous
> first-contact fields, analytic affine projection, the categorical
> axis-parallel chart, `t = t_star + rho/s_p`, and fixed min composition return
> the exact first closed-set contact for every ray direction and every periodic
> cell crossing. Runtime work is fixed once the family/band/layer catalogue is
> frozen.

This theorem includes no arbitrary finite texture sampling, no arbitrary
volume, no curved terrain chart, no spatial wind field, no spatially varying
community choice, and no camera-inside successor claim.

### 9.2 Bounded-approximation tier

The following may be admitted only with explicit residuals and gates:

- finite phase/angular texture sampling;
- axis-class quantization;
- chord/band approximation, width/taper/twist, and quantized tips;
- local terrain tangents;
- spatial gust and height/vigor modulation;
- volume/lump plume rendering;
- coarse per-family coverage mips; and
- community boundary transitions.

None may be listed under “what has no error term.”

### 9.3 Claims to delete or rewrite

Before implementation, remove or demote:

- exact camera-inside succession from the occupancy-sentinel field;
- exact arbitrary volume transfer from one `V(q,omega)` read;
- exact volume/opaque straddle composition from moments/lumps;
- exact per-family opaque-to-volume mip composition;
- global first-event error `<= rho Delta_omega`;
- general subtexel phase error from Lemma 6.2;
- family-independent pre-compilation free paths;
- exact chords from displaced masks alone;
- never-wrong-owner fixed-rank tips;
- exact spatially varying wind/terrain/height modulation;
- exact one-atlas control-field boundaries; and
- the “sole surviving architecture” necessity verdict.

## 10. Corrected go/no-go order

The original Phase A-first order is invalid because its binding free paths do
not exist until family masks exist. A mathematically defensible order is:

1. **Freeze the admissible class.** Exterior domain and minimum camera
   clearance; global affine exact tier; maximum families, bands, layers,
   records, and taps; ownership conflict rule; plume model; terrain/wind and
   control-boundary approximation contracts.
2. **Structural compilation only.** Produce enough candidate `(A_f,M_f,B_f)`
   structure to calculate the weighted axis-covering curve, true per-band
   chord/surface residuals, ownership conflicts, and actual nonempty
   family-band count. This is not runtime implementation.
3. **Budget the actual compiled masks.** Measure per-family phase/angular
   categorical error, support tube, depth, filtered silhouettes, connected
   wrong regions, and temporal stability. A free-path proxy alone is not the
   gate.
4. **Compute real resident bytes and fixed work.** Use the actual record
   layout, gutters, non-mippable channels, formats, and any demonstrated
   compression. Compare the result with the hard low/mid-end tap and memory
   ceiling.
5. **Only then test transcription.** Reconstruction against the exact
   compiled target is useful only after the revised mathematics and budget are
   green.

## 11. Final go/no-go

**NO-GO for implementation from the current handoff.** This is not a rejection
of affine-extrusion ground cover. It is a rejection of promoting the present
assembled theorem to code before its false exactness and budget claims are
removed.

**GO for a revised mathematics pass** centred on the proved opaque exterior
core. The next spec must answer, before any harness or shader work:

1. What exact finite family/band class is allowed, and what axis-covering
   bound does the accepted plant require?
2. Is plume/fuzz an approximate non-straddling volume, a higher-dimensional
   prefix field, or additional opaque families?
3. What minimum exterior clearance and terrain/wind approximation domain are
   promised?
4. How are spatial community boundaries represented without hiding a
   per-state firstness cost?
5. What sampled-field metric controls coverage and categorical stability in
   addition to the valid support-proximity bound?

Until those are fixed, a measured red or green result would be ambiguous: it
would test an unstated approximation, not the proposal's claimed mathematics.
