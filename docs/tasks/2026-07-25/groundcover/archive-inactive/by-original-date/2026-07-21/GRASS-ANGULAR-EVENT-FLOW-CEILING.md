# Angular event-flow oracle ceiling

Date: 2026-07-22  
Decision: **reject and park after one actual-source attempt**  
Runtime/shader changes: none

## Question

Can neighbouring baked directions share exact Calamagrostis surface events by
an optical/depth-flow-like correspondence, avoiding both scalar depth blending
and a dense independent complete-event atlas?

This gate is deliberately more favourable than any implementable codec. For
each exact target first hit it grants the reconstruction the exact world point,
triangle, periodic copy, and direction for free. It projects that point onto
two azimuth-bracketing or four azimuth/elevation-corner source lines and traces
the accepted periodic mesh again. A source line succeeds only when that exact
point is its first visible event. Thus failure is a real disocclusion, not
flow-estimation error, depth quantisation, filtering, or wrong interpolation.

## Immutable source and domain

- Source: `src/assets/groundcover/calamagrostis-canescens.gcrp`.
- SHA-256:
  `2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c`.
- GCRP/v4 decoded f64 oracle: `2,049,985` vertices and `2,171,134`
  triangles.
- Angular spacings: `22.5, 11.25, 5, 2, 1, 0.5` degrees.
- Target elevations: `0.1, 1, 5, 15, 35, 75, 90` degrees, plus an exact
  horizontal camera-inside successor gate.
- Structured target sampling: 28 `5x5` pinhole grids from two azimuths and two
  camera phases, with a 0.06-degree pixel pitch; 700 target rays, 424 actual
  hits. Exact-horizontal tests use two phases, two azimuths, four within-profile
  heights, a 155 m horizon, and a 1 cm same-line camera translation.
- A strict primitive match requires the same triangle and periodic copy. The
  more optimistic event/chart test permits an edge-tie owner change but
  requires the neighbouring first hit to be the same world surface point
  within 20 micrometres.

Exact horizontal is not removed, flattened, or assigned a top-plane phase. It
is treated as the pointed-line successor problem it actually is.

## Result

No spacing reaches the required 95% event visibility for exterior,
panicle-bearing, vegetative, and exact-horizontal camera-inside queries.

| spacing | any-four exterior event | panicle | vegetative | horizontal inside | optimistic delta + births |
|---:|---:|---:|---:|---:|---:|
| 22.5° | 87.74% | 88.97% | 85.31% | 62.50% | 13.47 MiB |
| 11.25° | 89.15% | 93.59% | 80.42% | 37.50% | 50.88 MiB |
| 5° | 82.08% | 90.04% | 66.43% | 25.00% | 285.44 MiB |
| 2° | 84.43% | 90.75% | 72.03% | 12.50% | 1,717.65 MiB |
| 1° | 87.50% | 94.31% | 74.13% | 12.50% | 6,547.56 MiB |
| 0.5° | 89.62% | 95.73% | 77.62% | 0.00% | 25,298.68 MiB |

The non-monotone exterior numbers are expected for visibility rather than a
smooth signal: changing the source direction can reveal or hide unrelated
surfaces before the oracle point. At 0.5 degrees the panicle subset finally
passes 95%, but vegetative events remain at 77.62% and exact-horizontal
inside successors at 0%. This is precisely the categorical disocclusion that
numeric depth or colour interpolation cannot repair.

The `5x5` grids also contain connected oracle failure regions. Across tested
spacings the maximum horizontal hole is 2--4 pixels and the largest connected
hole is 6--13 of 25 pixels. These are coherent wrong-view patches, not isolated
noise that a categorical selector can make disappear.

## Fixed-cost storage ceiling

For completeness, the report charges an eight-byte complete event record at
`258x258` phase resolution over a hemisphere direction count
`ceil(2 pi / DeltaTheta^2)`. The deliberately optimistic delta format stores:

1. one eight-byte base record field;
2. one packed four-byte inverse-flow coordinate per direction/phase;
3. one eight-byte complete record only for the measured residual-birth
   fraction.

It omits topology, chart, validity, and flow-regularisation metadata. Covered
queries would need two fixed reads and births three, so the read limit is not
the rejection. At 11.25 degrees the optimistic lower bound is already
`53,349,163` bytes, above the `51,121,152`-byte resident cap; every finer
spacing exceeds it by a rapidly growing factor. The only spacing that fits,
22.5 degrees, misses 12.26% of exterior events and 37.5% of horizontal-inside
events even with oracle correspondence.

## Prior art and LAAS contribution boundary

- Lin and Shum, *A Geometric Analysis of Light Field Rendering*, IJCV 58(2),
  2004, supplies geometry-assisted neighbouring-ray reconstruction and the
  occlusion/sampling limits of four-ray light-field interpolation.
- Chai, Tong, Chan, and Shum, *Plenoptic Sampling*, SIGGRAPH 2000, supplies the
  spatial/angular sampling framework.
- Sannikov supplies the repeating-copy O(1) objective and the vertical-grass
  missing-elevation approximation, but not this oracle gate or a published
  all-angle correspondence codec.
- LAAS-original work here is the exact authored-event visibility oracle on the
  accepted periodic GCRP/v4 mesh, the primitive versus exact-point identity
  split, botanical subset accounting, pointed-line camera-inside gate,
  connected disocclusion measurement, and complete-record residual-birth
  storage ceiling.

The primary-source links and fuller transfer assessment are preserved in
`../../deep-research/grass/FUR-PRECOMPUTED-VISIBILITY-TRANSFER-AUDIT.md`.

## Rejection and resume condition

Reject inverse angular flow as the complete all-angle carrier and do not build
its codec. An implementable estimator cannot exceed the oracle which already
knows the answer. More directions eventually spend an independent dense
angular lattice and still do not encode the pointed-line origin needed by
exact-horizontal/camera-inside successors.

Resume only if a new path-independent analytic visibility-sharing law can
create disoccluded events without an independent complete record per angular
cell, while also proving exact horizontal and arbitrary free-space successors.
Overlapping species and moss remain one offline-unioned marked community;
runtime work must never multiply by species.

## Reproduction and artifacts

```bash
node --import tsx tools/groundcover-bake/analyze-angular-event-flow-ceiling.ts
```

- Tool SHA-256:
  `c714678a440aa3a303fa116ef69e9acbf41e4faa930b483af824164d8fc9d1a9`.
- Artifact root:
  `data/work/groundcover-angular-event-flow/2ed57f59d86e8376/44420268b58db8a1/`.
- `metrics.json` SHA-256:
  `86957eac1c282463423ef38b7d54dcae49a1852175e097c19010f1e88cff0c75`.
- QA index SHA-256:
  `e60e65afb0cbf41d0cc0302f8116a08d7b4a3a2b5c2a4e673ffb3eab47c7967a`.
- QA colours: green is a recovered vegetative event, pale is a recovered
  panicle event, red is an oracle disocclusion, and grey is a target miss.
