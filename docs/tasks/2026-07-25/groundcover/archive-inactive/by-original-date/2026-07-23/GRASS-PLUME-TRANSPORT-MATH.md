# Direction-analytic transport for the soft plume only

**Date:** 2026-07-23

**Scope:** pure mathematics; no renderer, shader, WebGPU, or runtime change

**Status:** bounded periodic endpoint potentials are **refuted**. A direct
finite-spectrum prefix integral is mathematically sound, fixed-cost, and
resonance-free, but is viable only after a stricter source split and an
actual-source mode-count gate. It is not yet an implementation authorization.

## 1. Question and promised domain

The query is only for the genuinely soft, sub-pixel part of a panicle or
similar ground-cover fuzz. Assume that a separate structural representation
owns every recognition-bearing blade, culm, panicle axis, glume, lemma,
spikelet body, and coloured anther.

Let

\[
  D=\mathbb T^2_\Lambda\times [y_-,y_+]
\]

be one horizontally periodic community cell with a finite vertical support.
For a world ray, analytic slab clipping, the forward-ray condition, the scene
or opaque-core depth, and `R_max` produce a finite interval `[a,b]`. The plume
query must return its transfer on every such prefix, including exact vertical,
exact horizontal, and a camera in an air gap inside the height slab. A camera
actually inside authored plant matter may use the separately permitted fade.

The runtime restrictions are fixed work, no march, no traversal, no
data-dependent candidate set, no direction-bin election, and no runtime grass
geometry. Extra fixed work is acceptable only when it buys a stated invariant.

## 2. Why the periodic transport-potential route cannot be repaired

It is enough to consider one horizontally periodic Fourier mode

\[
  \kappa_k(q)=\widehat\kappa_k e^{i k\cdot q},
  \qquad k\ne0,
\]

and a canonical ray `q(t)=q_0+v t`. A periodic endpoint potential would solve

\[
  v\cdot\nabla P_v=\kappa_k,
\]

so its Fourier coefficient must be

\[
  \widehat P_{v,k}={\widehat\kappa_k\over i k\cdot v}.
\]

This is not a removable numerical inconvenience. On the great circle
`k dot v = 0`, the mode is constant along the ray and the cohomological
equation has no bounded periodic solution. Integrating around a periodic orbit
would make a bounded function increase by a nonzero constant on every turn.

### Theorem 2.1 (endpoint-potential no-go)

No bounded periodic endpoint field `P_v`, continuous for every direction, can
represent the finite integral of a nonconstant periodic density as

\[
  I(a,b)=P_v(q(b))-P_v(q(a))
\]

for every finite interval and direction.

**Proof.** Choose a nonzero mode `k` and a direction `v` with `k dot v=0`.
Along that ray the true contribution is

\[
  (b-a)\widehat\kappa_k e^{i k\cdot q_0},
\]

which grows linearly with interval length. The same Fourier mode at the two
endpoints has identical phase, so the endpoint difference is zero. Equivalently,
the required potential coefficient has the pole above. QED.

Adding a finite number of bounded endpoint potentials does not change the
argument: every endpoint difference of the `k` mode vanishes at its resonant
directions. A special mean term repairs only `k=0`; every nonzero mode has its
own resonance great circle.

For a horizon `R`, the angular transition around resonance has characteristic
width

\[
  \delta\theta_k\asymp {1\over |k|R}.
\]

For the current `Lambda=0.52 m` and `R=155 m`, even the fundamental reciprocal
mode has `|k|R \approx 1873`. An angular expansion of the potential therefore
needs order on that scale before higher spatial modes are considered. Softness
reduces Fourier amplitudes; it does not remove the small divisor.

This refutes the hoped-for construction made from a few bounded 3D potential
textures sampled at two endpoints. More endpoint reads do not cure it.

## 3. The finite segment itself has no resonance pole

The no-go is specific to factoring the answer through bounded endpoint
potentials. The actual finite integral is entire at resonance. With

\[
  \Delta=b-a,\qquad m=(a+b)/2,\qquad \omega=k\cdot v,
\]

one mode integrates exactly as

\[
\boxed{
  \int_a^b \widehat\kappa_k e^{i k\cdot(q_0+vt)}dt
  =\widehat\kappa_k\,\Delta\,
   e^{i k\cdot(q_0+vm)}
   \operatorname{sinc}\!\left({\omega\Delta\over2}\right)
}
\]

with `sinc(0)=1`. The apparent `1/omega` singularity has cancelled before
evaluation. At exact resonance the result is exactly the constant mode value
times `Delta`; at high axial frequency it receives the expected cancellation.

This identity, rather than a direction-sampled transfer texture or a periodic
potential, is the usable mathematical primitive.

## 4. Constructive finite-spectrum plume

Author one joint soft-fuzz medium per ecological community as a finite,
nonnegative spectral field inside analytic vertical clips:

\[
 \kappa(z)=
 \sum_{r=1}^{S}g_r(d)
 \left[
   a_{r0}+2\operatorname{Re}
   \sum_{j=1}^{Q_r} a_{rj}e^{i\xi_{rj}\cdot z}
 \right],
 \qquad z=(q,y).
\]

The horizontal components of every `xi` are reciprocal-lattice vectors.
Vertical localization may use vertical Fourier components inside one clip, or
a compile-time-bounded set of nonoverlapping polynomial/top-hat bands. The
directional fibre response `g_r(d)>=0` is a closed-form constant on one ray.
Nonnegativity must be certified, for example by a sum-of-squares
trigonometric construction; clamping a signed fit is not a proof.

Each mode uses the boxed formula after intersecting its clip with the actual
prefix `[a,b]`. Therefore:

- exact horizontal and vertical rays are ordinary cases;
- an arbitrary opaque-core or scene cutoff is exact;
- the origin height is not discarded;
- no camera direction is sampled or elected;
- no spatial event is relifted onto the live ray; and
- affine Tier-1 layer transforms and affine-in-height wind are exact inverse
  changes of variables: transform both the origin and ray direction, then use
  the same identity.

The complete optical depth is the sum across the fixed strata and two Tier-1
layers,

\[
  \tau=\sum_{\ell,r}\tau_{\ell r},
  \qquad A=1-e^{-\tau}.
\]

No species loop is present: all microscopic fuzz in the selected community is
compiled into the coefficients before runtime.

## 5. Source split forced by exact coloured composition

Different source colours do not generally commute through extinction. In
cumulative optical-depth coordinates, exact colour is

\[
 C=\int_0^\tau c(u)e^{-u}\,du.
\]

Knowing total optical depths or unattenuated colour integrals per species does
not determine this value when their spatial order changes. Exponentiating a
sum of coloured densities does not close into a finite set of independent
per-colour totals.

The exact small representation therefore requires one of the following:

1. all mutually interleaving plume density has one source colour;
2. differently coloured strata are globally nonoverlapping and have a fixed
   ray-orderable interval decomposition; or
3. colour is explicitly approximate in an optically thin regime.

The current author makes the strongest option practical. The true microscopic
soft population is the `hair-filament` family: callus hairs are approximately
`0.12--0.22 mm` wide and use an almost common pale cream colour. Panicle axes,
glumes, lemmas, spikelet bodies, anthers, and their filaments are not routed to
this medium; they remain in the structural core. Their violet, brown, green,
and cream recognition colours therefore remain categorical instead of being
averaged into fog.

For future species, pale hair/cotton fuzz may join the same joint medium.
Microscopic detail with a materially different colour either remains in the
core or uses a globally ordered second stratum. This is independent of source
species count.

If an optically thin coloured extension is deliberately accepted, using

\[
 C_{thin}=\int c(t)\kappa(t)dt
\]

has the conservative max-channel bound

\[
 \lVert C-C_{thin}\rVert_\infty
 \le c_{max}\tau_{max}(1-e^{-\tau_{max}})
 \le c_{max}\tau_{max}^2.
\]

That approximation is not acceptable on grazing rays unless the compiled
medium independently certifies the same `tau_max` there.

## 6. Prefix-tail bound and the real mode-count gate

Let `kappa_Q` be the retained finite spectrum and `e=kappa-kappa_Q`. The
correct error metric is not density L2 error or one complete-ray error. Every
opaque event can stop the integration at a different prefix, so the binding
quantity is

\[
\boxed{
 E_{prefix}=\sup_{ray}\ \sup_{[a,b]\subset I_{ray}}
 \left|\int_a^b e(z(t))dt\right|.
}
\]

Then Beer--Lambert is 1-Lipschitz in nonnegative optical depth:

\[
 |A-A_Q|\le E_{prefix},
 \qquad \lVert C-C_Q\rVert\le \lVert c\rVert E_{prefix}
\]

for the shared-colour construction.

For omitted Fourier coefficients, a direction-uniform sufficient bound is

\[
 E_{prefix}\le
 R_{max}\sum_{k\notin K_Q}|\widehat e_k|.
\]

This bound is sharp in its scaling: every omitted mode has a resonant ray on
which oscillatory cancellation disappears. A claim that grazing or distance
automatically removes the tail is therefore invalid. Pixel filtering may
multiply coefficients by a known footprint response, but only a strictly
positive promised minimum footprint can improve the uniform bound. With
arbitrarily close exterior cameras that minimum is zero.

Define the active mode set at opacity tolerance `epsilon_tau` by the measured
partial-ray contribution, not by coefficient magnitude alone:

\[
 K_{active}(\epsilon_\tau)=
 \left\{k:\sup_{ray,prefix}
   \left|\int_{prefix}\widehat\kappa_k e^{ik\cdot z(t)}dt\right|
   >\epsilon_k\right\},
 \qquad \sum_k\epsilon_k\le\epsilon_\tau.
\]

The finite-spectrum candidate is viable only if the actual callus-hair-only
source fits the accepted `epsilon_tau` with a small `|K_active|`. Values such
as `Q=8--16` are hypotheses, not conclusions.

### Air-corridor consequence

A nonzero finite trigonometric polynomial is real analytic, so a nonnegative
one cannot be exactly zero on an open 3D air region without vanishing
identically. Approximate leakage is amplified by a long clear ray. A density
floor `eta` over length `R` creates

\[
 A_{false}=1-e^{-\eta R}.
\]

To keep false alpha below `epsilon_alpha` at `R_max`, it is necessary that

\[
 \eta\le{-\log(1-\epsilon_\alpha)\over R_{max}}.
\]

For `epsilon_alpha=0.01` and `R_max=155 m`, this is about
`6.48e-5 m^-1`. This is why the actual-source gate must contain
positive-measure horizontal air corridors and every partial cutoff. A broad
low-order fog fit can look plausible top-down and still fail catastrophically
at grazing.

## 7. Pixel footprint and closest-distance contract

The exact centre-ray identity does not by itself filter a pixel. For a known
linear pixel filter, each Fourier mode is multiplied by that filter's transfer
function. A Gaussian canonical footprint with covariance `Sigma` contributes

\[
 H_\Pi(\xi)=e^{-\frac12\xi^T\Sigma\xi}.
\]

Perspective footprints grow along the ray, so exact bundle integration has an
additional analytic window (Gaussian angular filters lead to a complex-error-
function integral). Using one midpoint footprint is an approximation and must
be charged to the same prefix norm.

There is no nonzero geometry scale that remains sub-pixel for an exterior
camera allowed to approach arbitrarily closely. Consequently the authoring
boundary must be explicit:

- detail whose projected scale can exceed the accepted fuzz footprint at
  `D_min` belongs to the structural core;
- only the residual after that transfer is eligible for the analytic medium;
- below `D_min`, either the structural level is used or the already permitted
  near/inside fade owns the transition.

Distance can and should reduce retained visible spectrum through the known
pixel filter, but it cannot excuse an unfiltered near-field mismatch.

## 8. Runtime-shape accounting

If the reciprocal frequencies and coefficients are selected with the
community and held in constants/uniform storage, plume evaluation adds **zero
filtered texture samples**. It adds fixed coefficient loads and one unrolled
analytic mode evaluation for every retained mode and Tier-1 layer. The memory
is tiny:

```text
rough scalar coefficient count
  = S * (1 + 2Q) * (extinction plus any permitted shared-colour data).
```

Even `S=2,Q=16` is kilobytes rather than an atlas-scale allocation. It does not
justify a large `Q`: phase, `sinc`, coefficient accumulation, and their live
registers are the real cost. Two affine Tier-1 layers double that arithmetic.

A phasor texture can trade sine/cosine arithmetic for packed RGBA reads, but
it does not reduce the number of independent active modes or cure resonance.
Such reads are defensible only after the measured mode set is small. The
overall sample count may therefore rise modestly above nine if, for example,
two packed phasor reads replace expensive phase evaluation and the GPU trace
supports that trade. The reads buy phase evaluation; they do not buy missing
geometry or an angular approximation.

The plume should not invent an opaque representative depth. Given the nearest
opaque-core/scene depth `t_o`, integrate only to `t_o` and composite

\[
 C_{out}=C_{plume}+e^{-\tau}C_{opaque}.
\]

This is exact for the shared-colour medium and avoids a floating plume plane.
A pipeline that only accepts one opaque depth event cannot express this
transfer; mapping `A>0` to a mean depth is not a mathematical replacement.

## 9. Decision

The bounded periodic-potential proposal is decisively parked. Its resonance
is structural and a few more texture reads cannot repair it.

The strongest fixed-cost successor for soft plume is the direct finite-mode
prefix integral, with the following binding changes upstream:

1. restrict plume to nearly common-colour microscopic callus hairs/fuzz;
2. keep all coloured and recognition-bearing reproductive geometry in the
   structural core;
3. compile the species union into one joint nonnegative spectrum;
4. fit and certify maximal partial-ray error, including open horizontal air
   corridors, exact poles, opaque cutoffs, and the declared pixel footprint;
5. reject before runtime if the active spectrum exceeds the measured
   low/mid-GPU arithmetic/register budget.

This construction is O(1), direction analytic, exact at every angle for its
authored finite medium, prefix-correct under opaque overlap, and independent
of species count. What remains unknown is empirical but sharply bounded:
whether the callus-hair-only source admits a sufficiently small spectrum. If
it does not, the minimal further change is not more directions or endpoint
potentials; it is moving more plume silhouette into the exact structural core
until the residual satisfies the partial-ray spectral bound.
