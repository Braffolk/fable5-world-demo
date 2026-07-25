# Candidate H angular-continuity premeasurement

Date: 2026-07-24  
Status at threshold freeze: **not run; no result inspected**  
Scope: cook-side measurement only; no runtime, shader, asset, tap, or memory change

## 1. Decision being measured

Candidate H keeps one categorical maximum-angular scale read.  At every
elevation-row or azimuth-midplane election boundary, that read changes atlas
view discontinuously.  The carrier is admissible only if its compact filtered
atoms describe the same world-space bundle closely enough on both sides that
the switch is below the review pixel footprint.  Same-address comparison is
forbidden because adjacent view charts do not assign the same world line to the
same top-plane address.

The measurement uses the accepted current Calamagrostis GCRP/v4 source and the
two H atom scales

```text
sigma1 = 4 source texels
sigma2 = 16 source texels.
```

The frozen positive periodic kernel for this attempt is a unit-weight toroidal
square of radius `sigma`, i.e. `(2 sigma+1)^2` texels.  Coverage and
premultiplied RGB are arithmetic kernel measures; representative depth is the
covered-sample median, then quantised through H's ten-bit per-slice atom depth.
This definition is fixed before the result and is the definition any later H
cook would have to reproduce if the gate were GREEN.

## 2. World correspondence

For source slice `a`, source atom top origin `q_a`, direction `c_a`, and
representative depth `tau_a`, form

```text
P_a = (q_a.x,H,q_a.z) + tau_a c_a.
```

Address target slice `b` by the exact top-plane reprojection of that point:

```text
q_b = P_a.xz - ((P_a.y-H)/c_b.y)c_b.xz.
```

Fetch the target atom at the nearest unwrapped texel centre `Q_b(q_b)` and form

```text
P_b = (Q_b.x,H,Q_b.z) + tau_b c_b.
```

Every pair is evaluated in both directions, `a -> b` and `b -> a`.  Equal
source/target texel coordinates are never used as correspondence.

The mandatory boundaries are:

- aligned adjacent elevation rows at `25`, `45`, and `65` degrees, for all
  sixteen azimuths;
- every one of the sixteen azimuth midplanes, in every regular elevation row
  (`15`, `35`, `55`, and `75` degrees).

The live direction used to split representative displacement into longitudinal
and transverse parts is the exact row or azimuth boundary direction, not one of
the two stored slice directions.

## 3. Frozen metrics and thresholds

One epipolar direction is defined wherever its source atom has `A_a>1/255`;
there is no world point with which to reproject an empty source atom.  Coverage
and premultiplied-colour metrics use that directed source support, and running
every pair in both directions supplies both halves of the union without a
same-address fallback.  Representative displacement is evaluated where the
reprojected target also has `A_b>1/255`.

```text
coverage:       abs(A_a-A_b)
premul RGB:     max_channel(abs(A_a C_a-A_b C_b)) in linear authored RGB
longitudinal:   abs((P_b-P_a) dot d_live), reported separately
transverse:     length((P_b-P_a)-d_live*((P_b-P_a) dot d_live))
depth code:     abs(tau_b-tau_a), reported separately
```

Transverse displacement is divided by the live vertical pixel footprint

```text
pixel(R) = 2 R tan(55 degrees/2) / 1440
```

at exterior distances `R in {2 m,5 m,10 m}`.

The externally reviewed GREEN limits, frozen verbatim before measurement, are:

```text
abs(delta A):             p95 <= 0.08, p99 <= 0.20
premul RGB max channel:   p95 <= 0.06, p99 <= 0.15
transverse displacement:  p95 <= 0.75 px, p99 <= 1.5 px
                           independently at 2 m, 5 m, and 10 m
connected exceedance:     no connected region >= 1% of evaluated support
```

For the connected-region test, a sample exceeds when any p99 point limit is
crossed: `abs(delta A)>0.20`, premultiplied RGB delta `>0.15`, or transverse
displacement `>1.5 px` at the binding `2 m` distance.  Connectivity is
four-neighbour and toroidal.  The one-percent denominator is that directed
pair's evaluated source support.  The maximum over every required directed pair
is binding.

Colour-palette class change is reported only as a secondary diagnostic.  It
cannot turn a RED primary metric GREEN and is not itself a blocker because the
actual H palette has not yet been fitted.

## 4. Frozen verdict rule and artifacts

Each elevation boundary and each azimuth-row family is reported separately at
both filter scales, as are the aggregate elevation and azimuth families.  A
single RED primary metric or connected-region violation blocks Candidate H
before runtime implementation.  Thresholds, scales, kernel, support masks,
pairing, and pixel convention may not be tuned after seeing the result.

The analyzer must emit a content-addressed report plus numbered, labelled PNG
heatmaps and a machine index containing source/recipe/analyzer/image hashes,
dimensions, and interpretations under
`data/work/groundcover-candidate-h-angular-continuity/`.
