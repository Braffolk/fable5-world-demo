# Candidate M mathematical audit

Date: 2026-07-24  
Scope: `GRASSPROFILE2-CANDIDATE-M-LINE-CONDITIONED-ANALYTIC-TRANSFER.md` only  
Verdict: **REVISE before the offline gate; no runtime authorisation**

Candidate M contains a real dimensional reduction rather than a disguised sampled
4D light field: phase remains two-dimensional, only horizontal azimuth is sampled,
and elevation enters the closed form through the live slope `lambda`.  Its pole
limit, positivity of the amplitude interpolant, prism load count, and stated base
payload arithmetic are sound under the conditions below.  Several current claims,
however, are stronger than the mathematics establishes.

## Required corrections

### R1. Top-plane entry does not cover every exterior view

The Section 2 parameterisation is exact only when the forward ray enters the active
cover through `h=H`, so that `t_top >= 0` and the active segment begins at `Q=(q,H)`.
"Camera outside the cover box" is not sufficient.  A camera horizontally outside a
finite support region but vertically between `0` and `H` can see the cover through a
side boundary.  The same case occurs in the local terrain frame when rising terrain
puts the destination cover top above a standing camera.  Then `t_top < 0`, `Q` is
behind the camera, and integrating from `rho=0` includes a non-visible segment.

This is directly relevant to the active uphill and cover-edge complaints.  Before
the gate the spec must either:

1. prove that every quality-contracted query has forward top entry in the chosen
   local-affine chart, or
2. define a forward side-entry construction and show how its phase, line amplitude,
   pole/grazing limits, and fixed-cost lookup reduce to the stored field.

Fading a camera *inside* a cover box does not remove a camera outside a finite
horizontal support and looking through its side.

### R2. Finite world support is absent from the transfer integral

The closed form integrates a continuous interval `[0,R]` of the periodic community.
A separate world support value at the background point cannot preserve a cover edge,
a hole, or multiple enter/exit intervals along that ray.  Multiplying `kappa` by an
arbitrary support field destroys the stated exponential primitive; merely shortening
`R` works only for one known contiguous interval.

The document must define exactly how support intersects the ray, what quantities are
already available without a new profile read, and why the resulting transfer remains
fixed-cost and continuous.  Until then, the claims that large holes are retained and
that Candidate M can produce coherent rooted side silhouettes at cover boundaries are
unproved.

### R3. The colour expression is not the exact emission law of the stated medium

For the stated spatially varying modes, the exact premultiplied source is

```text
P = integral T(rho) sum_k kappa_k(rho) c_k d rho / s_p,
T(rho) = exp(-integral_0^rho sum_k kappa_k(r) d r / s_p).
```

In general this is **not**

```text
(1-exp(-sum tau_k)) * sum(tau_k c_k) / sum(tau_k).
```

Those coincide when the colour mixture is constant along the line (all mode
profiles are proportional), but Candidate M deliberately permits different `b_k`,
`gamma_k`, and base/residual ratios.  The current formula may remain as the
definition of a positive fitted colour surrogate, but it must not be called exact
radiative transfer or exact emission for the Section 3 `kappa_k` field.  Its error,
including plume colour migrating onto stem support, belongs explicitly in the fit
gate.

Also replace the `max(tau,tau_min)` quotient by the analytic continuous form

```text
P = phi(tau) sum_k tau_k c_k,
phi(tau) = -expm1(-tau)/tau,  phi(0)=1.
```

The existing clamp is continuous but is not the claimed law for
`0 < tau < tau_min` and suppresses colour quadratically there.

### R4. The footprint contraction mean has no accounted source

Section 6 requires eight azimuth-dependent phase means
`Bbar_k(omega), Abar_k(omega)`.  They cannot be recovered from the four local
tetrahedral loads.  The proposal currently assigns them no packing, read, binding,
or interpolation rule, so the 4/8-load schedules and "no new binding" claim are not
closed.

The gate spec must choose one concrete option and include it in byte/load accounting:

- a genuinely azimuth-independent profile constant (with the resulting fidelity
  tested),
- an analytic low-order azimuth model carried in existing profile constants, or
- an explicitly packed/interpolated table and its actual access cost.

Whichever is chosen must share the same periodic seam and pole convention as the
amplitude field.  This correction is required before using the minification law to
claim that particle crawl is structurally removed.

### R5. The support mask bounds phase dilation, not visible depth error

The 5x5 bit says only that the centre record's owner-visible region intersects a
microcell.  Accepting every point in that cell dilates that region by at most one
microcell in phase (`~0.4 mm` per axis, `~0.57 mm` diagonally at the stated rate),
provided the runtime first proves that the reprojected address is in the same atlas
texel as the fetched record.  It does **not** bound the along-ray depth or colour
error when another surface owns most of that microcell; that error can be much
larger at grazing incidence.

The spec must make snap/address agreement, half-open boundary convention, MISS
encoding (`mask=0` is the apparent available convention), and false-accept semantics
explicit.  The proposed false-accept/false-reject gate is necessary and must measure
rendered depth/owner error as well as millimetres of phase dilation.

### R6. The claimed resolved/bridge `C0` handoff is conditional, not proved

A cubic weight has matching scalar endpoint weights; it does not make two different
endpoint functions equal.  The bridge uses four R0 records, whereas the resolved
path may obtain a different eligible owner from corrected R1 records.  Even on the
same plane, two records can have different support masks.  Thus "R0 and R1 are
identical" is true only of the solved point/class under the same certified face, not
of eligibility or election in general.

In addition, the fractional bridge preserves background id/depth while the opaque
resolved endpoint writes grass id/depth.  That depth/id frontier is not made `C0` by
colour weights.  Rename the handoff claim as an empirical gate and score colour,
coverage, depth/id frontier stability, and temporal motion separately.  A failure
there must not be reported as a coefficient-fit failure.

### R7. The prism and seam construction needs one binding indexing definition

The mathematical count is correct: `triangle x interval` admits three conforming
tetrahedra, and sorting the interval coordinate among the phase triangle's two
ordered cumulative barycentrics produces four nonnegative successive-difference
weights and four product vertices.  Four 64-bit loads per layer is therefore exact.

For implementation equivalence, the spec still needs to record:

- the exact cumulative-coordinate ordering and monotone-path vertex table for all
  three tetrahedra;
- one deterministic equality/tie convention shared by adjacent cells; and
- whether 64 azimuth entries are modulo-identified (the stated 8.000 MiB) or a seam
  vertex is physically duplicated (65 entries, 8.125 MiB).

"Duplicated/identified" is not one layout.  `C0` at the seam is proved only after
one is selected and the endpoint bits are required equal.

### R8. Cost/divergence language and final two-layer schedule need correction

The base allocation arithmetic is correct:

```text
65 * 256^2 * 8 bytes = 32.500 MiB resolved
128^2 * 64 * 8 bytes = 8.000 MiB transfer
total                         = 40.500 MiB
```

This excludes the unaccounted means in R4 and any support data in R2.  The one-layer
query shapes are also arithmetically correct: resolved `4+4=8`, bridge `4+4=8`, and
unresolved `4`.  Two unresolved affine layers cost eight loads while sharing bytes.

However, footprint-selected resolved/bridge/unresolved branches vary across pixels;
they are not "uniform profile/branch selection" and can diverge within a quad/wave.
The implementation must demonstrate that inactive shader branches do not execute
their texture loads.  Further, the final tier-1 two-layer bridge has no <=9-load
schedule yet (`4 R0 + 2*4 transfer = 12`).  The current proposal is therefore a
single-layer Calamagrostis gate, not a completed final two-layer architecture.

## Verified mathematics

- **Units:** with `rho,R` in metres, `b_k,gamma_k,B_k,A_k` all have units `m^-1`;
  `lambda`, `s_p`, and `v` are dimensionless; `E` has units metres; every `tau_k`
  is dimensionless.
- **Vertical pole:** for `R=s_p Delta_t`, the base limit in Section 4 is correct,
  including `gamma=0`, and every bounded residual contribution is `O(s_p)` and
  vanishes.  Bit-identical `B` vertices across azimuth are sufficient to make the
  pole azimuth-independent.
- **Grazing:** for a forward top-entry ray with a finite continuous cutoff, the
  formula has no artificial `1/v` singularity.  Continuity to the declared exact
  horizontal miss additionally requires `R -> 0` as the top-plane intersection
  passes beyond that cutoff.  It is not established for the side-entry cases in R1.
- **Positivity:** nonnegative decoded vertex amplitudes and tetrahedral barycentrics
  preserve nonnegativity.  The footprint contraction is also positive because it
  is a convex interpolation with `0 <= w <= 1`.
- **No hidden 4D atlas:** elevation is analytic, not a sampled coordinate.  Sampling
  and conformingly interpolating azimuth does not by itself recreate the rejected
  elevation-by-azimuth chart field.
- **Packing:** `depth12 + oct8+8 + class4 = 32` bits and `mask25 + reserved7 = 32`
  bits.  The stated `RG32Uint` resolved record is internally consistent.

## Visual-claim assessment against the active complaints

If the fit succeeds, the absence of a filtered owner/depth/plane and the continuous
phase/azimuth interpolant plausibly attack view-direction stretching, categorical
swim, old row/azimuth rings, and part of the directional particle turnover.  The
micro-mask could separately reduce resolved long-plane streaks.

The proposal does not yet justify all sixteen complaints:

- uphill and side-edge correctness are blocked by R1/R2;
- a four-mode well-mixed positive surrogate can become a fog/sheet or lose plant
  silhouettes, and the incorrect exact-emission claim currently hides plume/stem
  ordering loss (R3);
- nearby missing thin detail may worsen through micro-mask false rejection (R5);
- unresolved lighting is only named as a statistical NDF, with no fitted parameters
  or fidelity metric for the normals/lighting complaint; and
- terrain curvature is not covered by the exact local-affine conjugation.

These are legitimate empirical risks rather than proof that the representation is
RED.  The proposed image/metric gate is the right place to decide them after R1--R8
make the tested model and its resource contract unambiguous.  Candidate M should not
be implemented, nor described as a complete unconditional exterior solution, before
those corrections and the resulting fit/handoff gate are green.
