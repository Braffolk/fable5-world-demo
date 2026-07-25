# Candidate O: eight-read aligned positive atoms — measured result

Date: 2026-07-24  
Status: **RED and parked; no runtime authorisation**

## 1. Question and frozen construction

Candidate O was the one decision-changing experiment left by
`GRASSPROFILE2-EIGHT-READ-ALIGNED-ATOM-AUDIT.md`.  It tested whether Candidate
H's physically filtered positive atoms become a sufficiently faithful
arbitrary-direction carrier after one representative-height epipolar
realignment per angular node.

The run reused the accepted `2,171,134`-triangle Calamagrostis GCRP and H's
existing `256 x 256` atoms at source-filter radii `sigma=4` and `sigma=16`:

```text
atom = (UNORM8 coverage, premultiplied linear authored RGB,
        covered-sample median ray depth quantised to 10 bits)
```

For each live middle-plane phase and direction inside H's stored `15--75`
degree elevation domain, the oracle performed exactly:

1. select the four surrounding H angular nodes;
2. read each node's seed atom;
3. turn its representative ray depth into a vertical height;
4. compute the exact flat local-affine middle-plane epipolar correspondence;
5. reread the **same node's** atom at the corrected address; and
6. bilinearly blend only corrected coverage and premultiplied RGB.

No representative depth, point, owner, plane, or normal was blended or
output.  This is four `seed + corrected` logical atom accesses, hence eight
logical reads.  The one-read maximum-weight H bridge was scored separately and
was not allowed to borrow the aligned result.

The corrected address used the exact geometric construction.  With
`s=d_xz/(-d_y)`, live middle-plane intercept `q`, node slope `s_i`, and seed
representative height `hbar_i`, the node's corrected middle-plane address was

```text
q_i* = q + (h_ref-hbar_i)(s-s_i).
```

It then converted `q_i*` back to the same node's top-plane atlas.  At
`s=s_i`, this is bit-identically the seed address.  All `32` exact stored-node
controls were numerical zero and GREEN, proving the address signs, chart
conversion, atlas orientation, and blend endpoint identity in the conforming
run.

## 2. Frozen evaluation

Exact-BVH positive-measure truth was freshly rendered on a `128 x 128`
middle-plane phase page and physically filtered at radii `2/8`, preserving the
canonical `sigma=4/16` widths.  Held-out elevations were `24`, `45`, and `65`
degrees.  The eight separated azimuths were the Candidate-K sector sites and
midpoints

```text
0, 11.25, 67.5, 78.75, 157.5, 168.75, 247.5, 258.75 degrees.
```

Every direction was evaluated at both scales.  A `4.5 mm` world-phase
translation used the frozen excess-change definition: codec change minus real
truth change.  The old positive-measure limits remained binding:

```text
coverage p95/p99             <= 0.08 / 0.20
premultiplied RGB p95/p99    <= 0.06 / 0.15
connected exceedance         < 1%
translation p95              <= 0.06, connected < 1%
```

A deterministic two-height, two-colour counterexample was also run.  It uses
two independently varying structures at `0.20 m` and `0.90 m`, so a single
representative height cannot align both except at stored directions.

## 3. Result

**Candidate O is decisively RED.**  None of the `48` held-out
direction/scale cases passed.  Only `9/48` translation cases passed, while all
exact-node identity controls passed.

```text
metric                              worst measured       limit
aligned coverage p95               0.605294             0.08
aligned coverage p99               0.879641             0.20
aligned premul-RGB p95              0.457583             0.06
aligned premul-RGB p99              0.641305             0.15
aligned connected exceedance       58.5571%             <1%
translation coverage p95           0.317647             0.06
translation premul-RGB p95          0.223979             0.06
translation connected exceedance   1.2329%              <1%
```

More filtering reduces but does not approach the threshold:

```text
scale    coverage p95   RGB p95   connected   translation A/RGB p95
sigma4      0.605294    0.457583    58.5571%      0.317647 / 0.223979
sigma16     0.378126    0.296956    12.7625%      0.124271 / 0.099822
```

The separate one-read bridge is also RED in all `48/48` cases; its worst
coverage p95 is `0.804706`, RGB p95 `0.559991`, and connected exceedance
`73.0103%`.

The synthetic two-height counterexample independently fails with coverage p95
`0.379569`, RGB p95 `0.187650`, and a `2.34375%` connected exceedance.  This
matches the analytic missing-stratum theorem: the seed median chooses one
phase shift, while different plant heights require different shifts.  Positive
blending prevents a fake depth surface but cannot restore the missing
height-conditioned measures.  The QA comparison visibly shows plume and green
structure smeared into the wrong phase rather than merely quantised.

H's four stored elevation rows also have no conforming four-node cell below
`15` or above `75` degrees.  Those required exterior directions were left as
an explicit domain failure; they were not clamped, flattened, or hidden.

## 4. Decision and durable artifacts

Candidate O is parked.  Per the frozen stop rule, packing, another correction
step, more blur, or a runtime trial is not authorised: none supplies the
missing height strata, and the unlimited-precision positive-measure result is
already far outside every threshold.

```text
data/work/groundcover-candidate-o-aligned-atoms/
  e3e0a4175b151b89/b3b7abee02162ee4/
    report.json
    SUMMARY.md
    qa/index.json
    qa/001..005-*.png
```

Source SHA-256:
`e3e0a4175b151b89a1a4bb58089aed2e809da337a04ce024240be8decdacb0be`

Recipe SHA-256:
`b3b7abee02162ee4947938ac81c1886ebe2c12183a14df8c7247600645137393`

Report SHA-256:
`9df38a214bb0dc47a988b30d13a4f19958667db0e81076748c7a8e1f5ba7929e`

The numbered QA panels are `exact-BVH truth | aligned-positive-atom
prediction | max-channel error`; the machine index binds every image hash,
dimensions, source, recipe, and report.
