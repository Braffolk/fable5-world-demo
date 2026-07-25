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

## 5. Measured result

**Verdict: RED.  Candidate H is blocked before runtime implementation.**

The accepted source was measured, not a proxy:

```text
source SHA-256     e3e0a4175b151b89a1a4bb58089aed2e809da337a04ce024240be8decdacb0be
source bytes       119462112
regular views      64 plus the vertical pole
interior grid      256 x 256
recipe SHA-256     cf0ec24e866884940f7c2dc4a2cd3ca125e977a879a29746a0cda1444a974531
analyzer SHA-256   dd0efc9c94dafa12b8df44ceb72d3bdbd4628bd7dcf796e07921900ee845a446
report SHA-256     168980c3389b22746fde690f91fac0079238d93cac3be4eb75c49921d79cf9b4
```

Every primary metric failed at every required boundary family and both scales:

| Scale | Boundary family | `|delta A|` p95/p99 | premul RGB p95/p99 | transverse px at 2 m p95/p99 | largest connected exceedance | Verdict |
|---:|---|---:|---:|---:|---:|---|
| 4 | elevation 25 | 0.7059 / 0.8627 | 0.4838 / 0.5680 | 350.63 / 411.53 | 88.56% | RED |
| 4 | elevation 45 | 0.6314 / 0.8275 | 0.4480 / 0.5712 | 130.28 / 173.67 | 75.58% | RED |
| 4 | elevation 65 | 0.5804 / 0.8275 | 0.4449 / 0.6123 | 100.48 / 126.28 | 65.75% | RED |
| 4 | azimuth at row 15 | 0.5451 / 0.7059 | 0.4426 / 0.5214 | 375.31 / 438.81 | 79.97% | RED |
| 4 | azimuth at row 35 | 0.6902 / 0.8431 | 0.4785 / 0.5717 | 141.09 / 164.53 | 78.96% | RED |
| 4 | azimuth at row 55 | 0.6510 / 0.8510 | 0.4825 / 0.6066 | 70.16 / 86.94 | 64.73% | RED |
| 4 | azimuth at row 75 | 0.5059 / 0.8392 | 0.3996 / 0.6388 | 26.30 / 34.05 | 23.04% | RED |
| 16 | elevation 25 | 0.5725 / 0.7137 | 0.3852 / 0.4751 | 340.22 / 380.93 | 96.31% | RED |
| 16 | elevation 45 | 0.4706 / 0.6196 | 0.3012 / 0.4107 | 129.24 / 165.84 | 91.14% | RED |
| 16 | elevation 65 | 0.4078 / 0.5608 | 0.2875 / 0.3960 | 103.00 / 125.17 | 88.27% | RED |
| 16 | azimuth at row 15 | 0.3686 / 0.5020 | 0.3118 / 0.3998 | 323.25 / 380.83 | 93.08% | RED |
| 16 | azimuth at row 35 | 0.4745 / 0.6000 | 0.3260 / 0.4100 | 137.09 / 153.61 | 91.40% | RED |
| 16 | azimuth at row 55 | 0.4392 / 0.5804 | 0.3091 / 0.4000 | 68.66 / 81.39 | 81.82% | RED |
| 16 | azimuth at row 75 | 0.3176 / 0.4902 | 0.2514 / 0.3870 | 28.36 / 33.72 | 48.10% | RED |

The least-bad family, the 75-degree azimuth switch at `sigma=16`, still misses
coverage p95 by `3.97x`, premultiplied-RGB p95 by `4.19x`, transverse p95 by
`37.8x`, and the connected-region ceiling by `48.1x`.  The primary 45-degree
row boundary misses those same limits by approximately `5.88x`, `5.02x`,
`172.3x`, and `91.1x`.  This is not a threshold-edge failure and cannot be
repaired by quantisation tuning.

The secondary sixteen-class palette changes on roughly `57--68%` of the
world-corresponded samples in representative reported families.  That is not
used to manufacture the RED verdict; coverage, colour, world displacement, and
connected regions already fail independently.

## 6. Interpretation and blocker

Positive spatial filtering stabilises each direction chart internally, but it
does not make independently baked first-hit views the same world-space measure.
At a max-angular switch, neighbouring charts still describe different visible
surface bundles.  Their world-corresponded coverage, radiance, and representative
points differ coherently over large parts of the tile.  A categorical view
switch would therefore retain the rejected fixed-angle rings/crawl even if its
fractional background compositing were otherwise correct.

The blocker is specifically Candidate H's **one categorical maximum-direction
scale read**.  It is not the background-compositing equation, and it does not
authorise more reads, a third scale, wider filtering, threshold tuning, or a
runtime attempt.  Resume only with a same-read angular carrier whose continuity
is structural rather than assumed—for example one precomputed atom already
defined at the live angular cell rather than a categorical choice between two
independent view atoms—and premeasure that carrier against the same frozen gate.

No runtime/shader file or accepted asset was edited or replaced by this gate.

## 7. Commands and artifacts

Exact command:

```text
uv run --project asset-gen python tools/groundcover-bake/analyze_candidate_h_angular_continuity.py \
  --threshold-note docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-CANDIDATE-H-ANGULAR-CONTINUITY-THRESHOLDS-FROZEN.md
```

Analyzer:

```text
tools/groundcover-bake/analyze_candidate_h_angular_continuity.py
```

The immutable pre-result threshold file is
`GRASSPROFILE2-CANDIDATE-H-ANGULAR-CONTINUITY-THRESHOLDS-FROZEN.md`; its
`2699cf49...` hash is the threshold-note hash bound by the recipe.  The present
file appends the measured result and therefore intentionally has a different
post-result hash.

Content-addressed result:

```text
data/work/groundcover-candidate-h-angular-continuity/
  e3e0a4175b151b89/
    cf0ec24e86688494/
      SUMMARY.md
      report.json
      qa/
        001-elevation-25-sigma4.png
        ...
        014-azimuth-75-sigma16.png
        index.json
```

The QA index binds every numbered PNG to its SHA-256, dimensions, source,
recipe, analyzer, report, and interpretation.  The root `index.json` points to
this immutable result.
