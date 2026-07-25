# Direct joint two-layer K2+K2 residual: globally auditable solver design

Date: 2026-07-23  
Status: **PURE-MATH SOLVER SPECIFICATION — no fit, shader, or runtime work authorized**  
Parent track: `GRASS-LATERAL-CORE-RESIDUAL-CPU-GATE.md`

## 0. Decision this document must make

The strict F4 lateral core is constructive. The last fitted residual is
decisively red, but it was fitted to one layer and then duplicated. That does
not answer the remaining question:

> Does any directly optimized, positive, jointly coloured two-layer residual
> with two modes in each affine layer reproduce the complete final
> core-plus-residual transfer within the accepted exterior-view errors?

The next admissible computation is therefore not another correlation-ranked
fit. It is one finite global or certified-near-global optimization with an
explicit model-error budget. Its output must be either:

1. a feasible coefficient/frequency assignment whose objective is below all
   gates by more than the numerical/model certificate; or
2. a lower bound above a gate, proving that no member of the frozen catalogue
   can pass; or
3. an honest unresolved interval `[lower,upper]` and the exact condition needed
   to close it.

Runtime remains fixed at F4 core reads in two layers plus `K_0+K_1=4` analytic
modes. No march, loop, candidate search, sampled camera direction, per-species
work, or runtime geometry is introduced.

The formulation is globally auditable for a frozen finite catalogue. It is
**not computationally credible over the entire literal 1 m pixel-resolvable
frequency box**: that box has billions of frequencies before pair selection.
A practical certificate therefore needs either an explicitly accepted low-
band residual class (the current catalogue has 171 frequencies) or a
frequency-independent structural lower bound that avoids support enumeration.

## 1. Frozen final-hybrid model

### 1.1 Layer coordinates

The two Tier-1 affine maps are fixed before optimization:

```text
x_l = T_l x + delta_l,       l in {0,1}.
```

They are applied identically to the accepted source, strict core, and
residual. Layer 0 is the identity. Layer 1 is the accepted golden-angle
rotation, scale, and phase. The optimizer may not move those transforms.

Let the shared vertical slab be `H=[y-,y+]`,

```text
u=(y-y-)/(y+-y-),       w(u)=4u(1-u).
```

### 1.2 Discrete frequencies and continuous coefficients

Layer `l` chooses an unordered pair of nonzero frequency indices
`nu_l1,nu_l2` from a frozen catalogue `C`, with an explicit inactive sentinel
for a one- or zero-mode solution. Its rest-space fields are

```text
kappa_l(x_l)
  = 1_H w(u) [a_l + 2 Re sum_(j=1..2) c_lj exp(i k_nu_lj dot x_l)],

j_l,r(x_l)
  = 1_H w(u) [b_l,r + 2 Re sum_(j=1..2) d_lj,r exp(i k_nu_lj dot x_l)],
```

for colour channels `r in {R,G,B}`. The variables are:

```text
nu_lj                         discrete frequency choices
a_l,b_l,r                     real constants
c_lj,d_lj,r                   complex coefficients
```

The live-mode charge is exactly `2+2=4` after layer expansion. Frequencies,
opacity coefficients, and colour coefficients are optimized jointly against
the final two-layer union. There is no per-layer marginal target.

### 1.3 Exact partial-ray moments

For every source ray sample `s`, prefix cutoff `b`, layer `l`, and catalogue
frequency `nu`, precompute the exact quadratic-window moment

```text
M_sblnu = integral_[entry,min(b,t_core)]
           w(u_l(t)) exp(i k_nu dot x_l(t)) dt.
```

`M_sbl0` denotes the real constant moment. These are the closed-form sinc and
first-two-derivative moments already derived in the parent mathematics. Exact
horizontal and vertical rays use their analytic limits, never epsilons.

The two-layer optical depth and unattenuated colour moment are linear in the
continuous coefficients:

```text
tau_sb = sum_l [a_l M_sbl0
                + 2 Re sum_j c_lj M_sbl,nu_lj],

J_sb,r = sum_l [b_l,r M_sbl0
                + 2 Re sum_j d_lj,r M_sbl,nu_lj].
```

### 1.4 Complete final-hybrid prefix map

Let `chi_sb` say that the nearest strict-core event occurs no later than the
prefix, and let `g_sb` be that categorical core colour. Define the stable
function

```text
s(tau) = (1-exp(-tau))/tau,       s(0)=1.
```

The exact model being optimized is frozen as

```text
A_sb = chi_sb ? 1 : 1-exp(-tau_sb),

C_sb,r = s(tau_sb) J_sb,r
         + chi_sb exp(-tau_sb) g_sb,r.
```

For a physical pixel bundle `p` with fixed quadrature or certified continuous
footprint weights `omega_s`, compare

```text
A_p = sum_s omega_s A_sb,
C_p = sum_s omega_s C_sb
```

directly to the complete opaque-source prefix transfer. There is never a
`source minus core` target. Every source event, core event, slab entry/exit,
and real scene cutoff contributes immediately-before and immediately-after
prefixes. Event quantiles alone are not a certificate.

The formula above is an authored effective-transfer definition. It is not
silently described as the exact radiative-transfer solution for spatially
varying `j/kappa`. Changing it requires a new mathematical contract.

## 2. Exact positivity and colour constraints

The solver enforces, over the entire infinite repeated field,

```text
0 <= kappa_l <= kappa_max,
0 <= j_l,r <= kappa_l.
```

`kappa_max` is a measured/frozen soft-residual density ceiling. It is needed
both for a compact coefficient domain and for the spectral-tail proof in
Section 5. A ray-integrated `tau_cap` alone is not a substitute: without a
pointwise or coefficient envelope, increasingly high frequencies can form
increasingly tall, narrow nonnegative spikes whose integrated effect does not
vanish uniformly.

Point samples are not a proof. Frequency pairs split into three exact cases.

### 2.1 Independent phase pair

If the **horizontal projections** of `k_1,k_2` are linearly independent, their
two phases range independently over the two-torus at every fixed height. For

```text
p=a+2 Re(c1 z1+c2 z2)
```

global nonnegativity is equivalent—not merely sufficient—to

```text
a >= 2(|c1|+|c2|).
```

This is a second-order-cone constraint. Apply the same exact cone to `j_r`
and `kappa-j_r`:

```text
b_r >= 2(|d_1r|+|d_2r|),
a-b_r >= 2(|c1-d_1r|+|c2-d_2r|).
```

Apply it once more to `kappa_max-kappa` for the upper density bound.

### 2.2 One common full-circle phase

If the two 3D frequencies are integer harmonics of one base vector whose phase
ranges over the full circle on the authored domain, their phases are not
independent. The amplitude bound would wrongly exclude legal fields such as

```text
a[1-cos(lambda(y-y0))]^2.
```

Use exact univariate trigonometric-polynomial positivity instead. Introduce a
small Hermitian positive-semidefinite Gram matrix `Q` whose diagonal sums
match the constant and whose offset-diagonal sums match the two Fourier
coefficients (Fejer-Riesz/Toeplitz form). Apply it independently to `kappa`,
`j_r`, and `kappa-j_r`. With two selected harmonics these are tiny fixed SDPs,
not sampled positivity tests.

### 2.3 Rank-one horizontal projection or vertical-only pair

The remaining case is important in practice: the horizontal projections are
collinear (possibly zero), while different vertical components make the 3D
frequencies non-collinear. The selected failed fit itself had one horizontal
frequency with two different vertical indices. Its reachable phases form a
cylinder or finite strip, not the whole two-torus, so neither Section 2.1's
SOC nor Section 2.2's one-phase Gram matrix is exact.

Treat positivity as the convex semi-infinite constraint

```text
p(theta,y) >= 0     for (theta,y) in S1 x [y-,y+]
```

(or only `[y-,y+]` for vertical-only modes). A certified separation oracle
finds the global minimum of this two-mode trigonometric polynomial over the
compact domain by interval root isolation/branch-and-bound. Violating points
add exact linear coefficient cuts. Final feasibility is certified either by
an interval lower bound over the whole cylinder or by a low-degree
trigonometric interval-SOS certificate. Apply the same oracle to `kappa`,
`kappa_max-kappa`, `j_r`, and `kappa-j_r`.

The feasible coefficient set remains convex because it is an intersection of
linear half-spaces indexed by the compact domain, even though its exact cone
is not the simple SOC. Duplicate/opposite frequencies are canonicalized before
solving. A zero-horizontal, nonzero-vertical mode is legal. This is essential:
a vertical notch is the known counterexample to the false single-height no-go
argument.

A factorization such as `|psi|^2` may be used only after expanding it and
charging every resulting Fourier mode to `K_total`. It is not a way to hide a
four- or eight-mode density behind two generator symbols.

## 3. Compact coefficient domain

Global optimization needs a compact continuous domain. An arbitrary numeric
coefficient clamp is not acceptable. Compactness comes from the authored
residual transfer contract:

1. no-core source prefixes admitted to the soft part of the final hybrid have
   a frozen maximum residual opacity `A_res,max<1`; opaque landmarks must be
   core failures;
2. therefore define
   `tau_cap=-log(1-min(A_res,max+epsilon_A,1-T_floor))` with a declared positive
   transmission floor `T_floor`—never by silently clamping an opaque target;
3. freeze the measured pointwise soft-density ceiling `kappa_max`; Fourier
   coefficients are then bounded independently of frequency (in particular,
   each coefficient magnitude is no larger than the field's uniform norm);
4. every admitted full/prefix ray satisfies `0<=tau<=tau_cap` unless a
   separately named saturated-tail class is proved visually irrelevant;
5. positive-measure transparent-air and cover-positive boxes in Section 6,
   together with the exact positivity cones, bound every constant and mode
   coefficient. For a bandlimited polynomial this bound is made explicit by
   the interval/Remez certificate, rather than assumed.

Colour compactness follows from `0<=j_r<=kappa`. The solver report emits the
derived coefficient box and the exact constraint that produced every face of
it.

## 4. Objective and certificate-relevant errors

For each pixel/prefix row, use premultiplied quantities and define

```text
e_A,p   = |A_p-A_ref,p|,
e_C,p   = max_r |C_p,r-C_ref,p,r|.
```

The primary minimax objective is

```text
z = max_p {e_A,p/epsilon_A,p,
           e_C,p/0.15}.
```

`epsilon_A,p` is fixed before solving from the silhouette margin. For
reference pixels whose filtered alpha is separated from the 0.5 decision by
`gamma_p`, requiring `e_A,p<gamma_p` preserves its class. Ambiguous edge
pixels are scored continuously and cannot manufacture a binary failure.

The following remain separate, binding gates:

- silhouette IoU `>=0.97` and largest connected wrong region `<1%` on the
  validation images;
- paired source-MISS/core-HIT false coverage;
- paired early-core depth excess beyond the declared `5 cm` compositing
  tolerance;
- categorical payload/colour errors for purple and cream landmarks;
- unforced class change `<5%` under `1/2.5/4.5 mm` translations.

The residual cannot move a strict-core event. Therefore

```text
A_p >= core_hit_fraction_p
```

is an immediate coefficient-independent alpha lower bound. Core false
coverage and unacceptable categorical depth are evaluated before solving.
One certified violating row is sufficient for RED. Conversely, passing the
small optimization corpus is not GREEN; the complete exterior validation set
still binds.

## 5. Can the pixel/distance law make the frequency set finite?

### 5.1 The answer is conditional

Yes, for a declared continuous pixel-footprint filter, a frequency-independent
`kappa_max`/coefficient envelope, and a nonzero error budget. No, not from
distance labels alone, not for an unfiltered centre-ray evaluation, and not
when arbitrarily tall high-frequency density spikes remain legal.

For a mode `k`, let `H_s(t,k)` be the exact average of
`exp(i k dot delta x)` over the physical pixel aperture at path location `t`.
Its maximum possible omitted contribution is bounded by

```text
B(k) = 2 C_coeff max_s integral w(t) |H_s(t,k)| dt.
```

For a two-dimensional continuous angular aperture, split any `k` into the
component along the centre ray and its transverse component. At least one has
magnitude `>=|k|/sqrt(2)`:

- a large axial component gives an integration-by-parts/sinc bound `O(1/|k|)`;
- a large transverse component gives a footprint bound
  `O(log(1+|k| R theta_pix)/(|k| theta_pix))` after integrating from the ray
  origin through the finite interaction horizon.

The quadratic window is zero at the slab boundaries and has bounded
derivatives, so it does not spoil that decay. Hence `B(k)->0` uniformly. Once
`B(k)<=eta_mode`, deleting that mode changes every admitted opacity/RGB row by
at most its charged `eta_mode`. The catalogue

```text
C_eta = {nu on the frozen reciprocal/vertical lattice : B(k_nu)>eta_mode}
```

is finite.

If runtime evaluates only a centre ray, a mode perpendicular to that ray has
no transverse attenuation; arbitrarily high-frequency notches remain
available and no uniform pixel-law cutoff follows. A finite global claim
therefore requires the already proposed analytic footprint attenuation/LOD,
or an explicit authoring bandlimit accepted as a quality restriction.

### 5.2 Distance-aware does not mean cheaply enumerable

At `55 degrees / 2160 px`, `theta_pix` is about `0.000444`. At `1 m`, one
physical pixel is about `0.444 mm`; a literal Nyquist wavelength is about
`0.888 mm`. For the `0.52 m` tile this permits horizontal indices on the order
of

```text
M_xz ~= 0.52/0.000888 ~= 586.
```

For a `1.176 m` vertical slab, a `pi n/H` basis permits vertical indices on
the order of `2H/0.000888 ~= 2650`. The resulting raw 3D catalogue has billions
of entries. It is finite but not directly enumerable and proves that “the
pixel law makes it finite” is not by itself a practical solver plan.

The currently explored authoring catalogue

```text
mx,mz in [-3,3],       my in [0,3]
```

has `171` canonical nonzero frequencies. That is tractable enough for a
certificate track, but it is an explicit low-band authoring restriction, not
something implied by 1 m pixel resolution. The final report must show both:

1. the result inside this declared low-band class; and
2. the charged tail/quality uncertainty between that class and the much
   larger pixel-resolvable class.

If the user will not accept a low-band residual restriction, the response-
quotient/branch strategy below must cover the larger `C_eta`; simple pair
enumeration is infeasible.

## 6. Smallest decisive source dataset

### 6.1 Positive-measure air is required

An isolated horizontal centre ray is insufficient. A legal K2 field can place
a quartic vertical notch at exactly that height. Build certified open sets

```text
E_i = Q_i x Y_i x Omega_i
```

such that, for every origin phase/height in `Q_i x Y_i` and every ray in the
full physical pixel cone `Omega_i`, both transformed source layers and all F4
core fields miss to a fixed finite cutoff. Certification uses swept-cone BVH
distance/interval bounds, not point rays. Emit the world/phase/height/angular
measure of every box.

For a frozen vertical degree `N_y`, use enough separated height intervals and
total vertical measure that a trigonometric Remez/Turan inequality gives a
numerically useful bound. Exact zero on one open interval already forces an
analytic polynomial to vanish identically; the positive-measure inequality is
needed because the visual target permits small nonzero error. If the computed
Remez constant is too weak, the dataset is not decisive and must say so.

Pair those air boxes with certified cover-positive boxes at comparable
heights/phases whose filtered reference alpha or coloured transfer has a
strict lower bound. This creates the actual low-frequency tradeoff; empty
constraints alone can always be met by the zero residual.

### 6.2 Analytic no-open-gap theorem

There is a stronger exact structural statement. A finite Fourier residual in
each affine layer is real analytic inside the open slab. The sum of the two
pulled-back densities is nonnegative. Suppose an open set of exterior origins
and directions has exact zero residual opacity over a nonzero ray interval.
Every such line integral of the nonnegative total density is zero, so the
total density is zero pointwise on the swept open world volume. Each layer is
nonnegative, hence each layer is separately zero there. An invertible affine
pullback maps that volume to an open rest-space volume; the real-analytic
identity theorem then makes each layer identically zero throughout the
connected slab interior.

Therefore a nonzero finite-spectral residual cannot reproduce **exactly** an
open family of transparent air corridors and also provide transfer elsewhere.
This applies to any finite mode count and to the two incommensurate affine
layers; it is not a K2 fitting accident.

It is not by itself a visual RED, because the accepted quality allows small
nonzero opacity. Its quantitative version is precisely the bandlimited
Remez/Turan bound above: measured positive air-set volume and a frozen
frequency degree convert the allowed air opacity into an upper bound on
transfer elsewhere. This structural route is potentially much cheaper and
stronger than enumerating the full physically visible frequency box.

### 6.3 Counterexample corpus, then separation

The smallest useful initial corpus is approximately `48–96` physical pixel
bundles, not another full rendered frame:

- positive-measure horizontal-air cones in four cardinal directions, spread
  across lower/middle/upper height intervals and at least two phase regions;
- matched top-down source-hit and source-miss phase boxes at `1 m`, paired at
  `32 m` through their actual larger footprint;
- low-oblique `1,5,18 degree` bundles containing the existing core-depth/CDF
  counterexamples;
- purple and cream head-prefix bundles whose recognition colour survives the
  reference filter;
- the same rows translated by `1,2.5,4.5 mm`;
- every relevant source/core event-adjacent prefix for those bundles.

The exact count is chosen by an auditable cutting-plane procedure:

1. solve the finite global problem on the current corpus;
2. run a certified interval separation oracle over phase, exterior direction,
   distance, height, and prefix to find the largest unrepresented violation;
3. add that entire certified box, not just its centre sample;
4. repeat until the lower bound is RED or no violation above the remaining
   certificate budget exists.

This is not iterative runtime work. It is an offline proof construction. A
RED subset is decisive. A GREEN claim additionally requires the complete
cardinal/all-elevation validation images and connected-region metrics.

## 7. Global/near-global solver

### 7.1 Outer discrete problem

For `P=|C|`, naive separate pair enumeration costs

```text
choose(P,2)^2.
```

At `P=171` this is about `2.11e8` two-layer pair assignments, too large for a
per-assignment nonlinear solve. Use a certified branch-and-bound tree over
frequency supports:

1. precompute each frequency's complex moment-response vector over all
   corpus rays/prefixes;
2. canonicalize conjugates, layer symmetries, inactive slots, and collinear
   harmonic cases;
3. quotient two frequencies only when interval arithmetic proves that their
   response vectors differ by at most `eta_quotient/C_coeff` on every certified
   box; charge `eta_quotient` to the final gap;
4. for a partial support node, compute a lower bound after relaxing every
   unchosen mode to the span or interval hull of all remaining response
   vectors;
5. prune only when that lower bound exceeds the best feasible upper bound
   minus the complete certificate budget.

Fast FFT/correlation scores may order nodes, but may never prune one.

### 7.2 Fixed-support continuous problem

For a fixed K2+K2 allocation, `tau` and `J` are linear in coefficients. The
nonlinearity is confined to `exp(-tau)` and `s(tau)`.

- Branch only the opacity coefficients. On every coefficient box, interval
  arithmetic gives a `tau` interval for each ray.
- Use tangent/secant convex envelopes of `exp(-tau)` and `s(tau)` over that
  interval to obtain rigorous lower bounds.
- Conditional on a fixed opacity point—or on an interval relaxation of
  `s(tau)`—the colour minimax problem is a small SOCP/SDP because `J` is
  linear and the exact positivity cones are convex.
- Split the opacity box with the largest contribution to the objective-bound
  gap. Evaluate feasible points with a local nonlinear solver only to improve
  the upper bound; local convergence is never the certificate.

The continuous dimension is small: per layer, five real opacity variables
and fifteen real colour variables before Gram auxiliaries. Colour elimination
leaves at most ten branched opacity variables for K2+K2.

### 7.3 Certificate

The immutable solver artifact contains:

- source/core/network/frequency-catalogue/dataset hashes;
- all exact or interval-enclosed moment vectors;
- positivity case and cone/Gram certificate per support;
- the incumbent feasible coefficients and recomputed upper bound;
- every open branch node, its coefficient/frequency domain, and lower bound;
- every pruned node and the inequality that pruned it;
- spectral-tail, response-quotient, ray-footprint, quadrature, and floating-
  point interval budgets;
- final global interval `[L,U]` for `z`.

If

```text
U-L <= delta_solver
```

then the result is near-global inside the frozen catalogue with total error

```text
delta_total = delta_solver + eta_mode + eta_quotient
              + eta_footprint + eta_interval.
```

`L>1+delta_total` is a certified RED. `U<1-delta_total` permits full exterior
validation; it is not by itself visual acceptance. An interval overlapping
one is unresolved.

## 8. Lower bounds to run before branch-and-bound

Apply these in order:

1. **Core floor:** `A>=core_hit_fraction`; source-MISS/core-HIT and categorical
   depth errors are coefficient-independent.
2. **Independent-ray oracle:** allow every ray/prefix an unrelated nonnegative
   `tau,J`. Failure here refutes any shared field.
3. **Shared-prefix monotonicity:** along one ray, optical depth is monotone in
   cutoff. Violating source targets give a representation-independent lower
   bound.
4. **Air/cover Remez bound:** use the measured positive air-set volume and
   frozen `(M_xz,N_y)` to bound how much a positive bandlimited field can rise
   on the cover-positive set while remaining below the air tolerance.
5. **Unconstrained dictionary span:** drop positivity and K2 sparsity but keep
   the full catalogue span. Failure refutes every K2+K2 member.
6. **K2+K2 support branch-and-bound:** only if all cheaper relaxations survive.

These are proof-strength filters, not alternative models. Passing one merely
allows the next.

## 9. Offline cost and go/no-go limits

For the current `P=171` catalogue and a `48–96` bundle corpus with all event
prefixes (roughly `500–2,000` rows):

- exact moment generation is seconds to minutes and `O(P rows)` storage;
- the coefficient-independent and span relaxations should finish in minutes;
- a response-quotiented support tree with small SOCP/SDP subproblems is
  expected to take hours on one workstation if pruning is effective;
- the raw `2.11e8` pair-product is not an acceptable fallback. If safe pruning
  does not reduce the live support nodes by at least roughly three orders of
  magnitude, park and report the remaining bound instead of running for days;
- the literal pixel-resolvable billion-frequency catalogue is not feasible
  without an explicit lower authoring bandlimit or a mathematically certified
  multiresolution frequency-domain branch bound.

This last point is a hard computational verdict, not cautionary wording. If
the catalogue has `P` frequencies, separate K2 choices cost
`choose(P,2)^2=Theta(P^4)` before continuous optimization. For a billion-scale
`P` this is astronomically outside the local/no-budget project even if almost
all nodes are eventually pruned. Near-range responses are not known to fall
into a tiny number of certified equivalence classes. The only credible routes
are:

1. user-accepted low-band authoring plus a complete certificate inside it;
2. the analytic/Remez structural bound, which bypasses frequency selection; or
3. a new proven multiresolution support bound whose cost is polynomial in
   bandlimit rather than in the pair-product catalogue.

Go/no-go before implementation:

- **RED:** a cheap lower bound or final `L` exceeds a gate after all certificate
  budgets;
- **GREEN FOR FULL CPU VALIDATION:** a feasible `U` clears every corpus gate,
  `U-L` and spectral/quotient errors are small enough that the clearance
  remains, and the separation oracle finds no omitted violation;
- **PARK:** the global interval overlaps the gate after the focused solver
  budget, the Remez air set has insufficient positive measure, or the required
  catalogue is too large without a user-approved bandlimit.

There is no shader step from an unresolved or merely locally optimized result.
