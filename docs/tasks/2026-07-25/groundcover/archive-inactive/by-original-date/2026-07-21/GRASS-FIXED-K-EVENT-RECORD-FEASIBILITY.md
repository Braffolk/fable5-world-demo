# Fixed-K line-event record feasibility for Calamagrostis

Date: 2026-07-22

## Decision

**Reject a fixed `K=2`, `K=4`, or `K=8` event-distribution / quantile-CDF
record as the exact all-angle geometric carrier.** The rejection is already
decisive with exact per-line, unquantized, actual-mesh atoms. Practical spatial
pooling, angular interpolation, depth quantization, and multi-species overlap
can only add error.

This result does not authorize a runtime event traversal, larger candidate
list, ray march, loop, extra pass, or distance-dependent search. It says that
this particular tiny-deep-record representation cannot satisfy the existing
fixed-cost contract.

## Mathematical gate

Let one oriented line contain the ordered categorical surface events

\[
E=\{(s_i,a_i)\}_{i=0}^{N-1},\qquad s_i<s_{i+1},
\]

where `a_i` carries the visible surface attributes. A camera at line coordinate
`s` requires the successor

\[
S_E(s)=\min\{(s_i,a_i):s_i>s\}.
\]

A fixed-K atom record retains a subset `R` of at most K events. Every omitted
event `i` owns a non-empty predecessor interval on which `S_E(s)=i` but
`S_R(s)` is either a later event or a miss. Interpolating a CDF cannot recreate
the omitted event's categorical normal, colour, botanical class, or species.
Consequently, even an oracle subset has the event-uniform exact bound

\[
\frac{\sum_{\text{lines}}\min(K,N)}
     {\sum_{\text{lines}}N}.
\]

The evaluated endpoint-CDF record is deliberately favourable: it keeps the
exact first and last events and uses inclusive, evenly spaced event-rank
quantiles between them. The separate oracle is even more favourable: for each
line and each query family it may choose a different K-event subset, while
forcing retention of the exterior first event.

## Actual-mesh experiment

The source of truth is the accepted GCRP/v4 Calamagrostis mesh, not its baked
first-hit atlas:

- asset: `src/assets/groundcover/calamagrostis-canescens.gcrp`;
- SHA-256:
  `2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c`;
- `2,049,985` vertices and `2,171,134` triangles;
- `3,232` sampled oriented lines and `519,359` deduplicated analytic surface
  events;
- `4x4` periodic phases, eight azimuths, exact horizontal, and signed
  `0.1/1/5/15/35/75/90` degree elevations;
- horizontal horizon `155 m`;
- thirteen camera-inside height queries and 32 metric-uniform queries per
  applicable line;
- shared-edge/coincident hits merge within `1e-8 m` on the unit ray.

The largest sampled line contained `5,275` events. One `0.1` degree line
through the finite botanical height contained as many as `2,657` events. Thus
the difficulty is not restricted to an infinite abstract horizon.

### Measured reconstruction

| K | event-topology oracle ceiling | non-empty lines with N<=K | endpoint camera-inside exact event | endpoint camera-inside p95 height error | endpoint metric-uniform exact event | endpoint metric-uniform p95 height error | per-family oracle metric exact event |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.561% | 16.743% | 44.881% | 0.851 m | 27.261% | 0.743 m | 43.552% |
| 4 | 1.044% | 23.127% | 49.382% | 0.621 m | 30.644% | 0.513 m | 56.458% |
| 8 | 1.924% | 28.860% | 56.363% | 0.401 m | 36.607% | 0.313 m | 67.693% |

For K=8, endpoint camera-inside palette-class agreement is only `74.733%`.
Exact panicle-event recall is `62.267%` and exact stem/leaf-event recall is
`54.589%`. The coarser panicle-vs-green class can look deceptively better
because one unrelated panicle event may replace another; it does not preserve
the visible first surface or its depth.

At the critical `0.1` degree rows, K=8 endpoint-CDF camera-height exact-event
agreement is `29.05%` downward and `2.87%` upward. The corresponding p95 world
height errors are `0.430 m` and `0.762 m`. Higher-angle success therefore
cannot rescue the required arbitrary-angle contract.

Coverage alone is misleading. The endpoint record retains the last event, so
it obtains 100% sampled hit recall while often jumping over hundreds of metres
of line-space events to the wrong surface.

## Exact cost model

A favourable complete atom is eight bytes:

- depth `u16`;
- octahedral normal `2xu8`;
- authored colour `RGB8`;
- one byte for valid state plus a seven-bit botanical/species class.

One 16-byte logical load carries two atoms, so K=2/4/8 require respectively
one/two/four fixed loads and two/four/eight statically unrolled compare-select
lanes. There is no runtime loop in this cost model.

| K | complete bytes/line | 81-dir dense bytes | 81-dir index-two bytes | fixed logical reads |
|---:|---:|---:|---:|---:|
| 2 | 16 | 86,266,944 | 43,467,840 | 1 |
| 4 | 32 | 172,533,888 | 86,935,680 | 2 |
| 8 | 64 | 345,067,776 | 173,871,360 | 4 |

The resident-profile ceiling is `51,121,152` bytes. Only K=2 fits, and only
after the optimistic index-two spatial reduction which already halves the
line-phase samples before measuring interpolation damage. K=2 is geometrically
unacceptable. K=4 can fit the dense budget only as a depth-only two-byte atom,
which discards colour, normal, botanical/species identity, and therefore cannot
represent overlapping multi-species cover. K=8 exceeds the budget even as a
dense depth-only record (`86,266,944` bytes at 81 directions).

## Reproduction and immutable outputs

Command:

```text
npx tsx tools/groundcover-bake/analyze-fixed-k-event-record.ts
```

Tool:
`tools/groundcover-bake/analyze-fixed-k-event-record.ts`, SHA-256
`74328e2cd193e022be82bbddde97588afdfc4869617e2d6a6e3834150c2db456`.

Artifact root:
`data/work/groundcover-fixed-k-event-gate/2ed57f59d86e8376/9e53f33e1c30daab/`.

- recipe SHA-256:
  `9e53f33e1c30daabba67a630a55fe466e1a961564c9b5c1d6512fd21e238f675`;
- `metrics.json` SHA-256:
  `0a6c8dc85c23f724c3b6ab549fe514fc7b11669f6188cab71fdf8c426449975c`;
- `index.json` SHA-256:
  `3e676674711e9eb1fa33ac6aa6ede726917c3df2c68ea86eeb8477e80254124d`.

The central provenance record is
`docs/deep-research/grass/GROUNDCOVER-LINE-FIELD-SOURCE-AND-EXPERIMENT-LEDGER.md`.
The successor definition and information-dimensionality argument are developed
in `docs/tasks/2026-07-25/groundcover/active/GRASS-CHEAP-PROJECTIVE-RAY-MATH.md`; the full-event
census used to formulate this gate is recorded in the same ledger. No external
source is being credited with this fixed-K rejection: it is a new result from
the accepted LAAS asset and the explicitly stated successor model.

## Resume condition

Resume a finite event-record path only if a new mathematical representation
encodes the omitted-event successor map without one atom per categorical event,
passes exact horizontal and signed sub-degree actual-mesh queries, carries the
winning colour/normal/species record, and demonstrates the complete
`<=51,121,152`-byte / `<=4`-load cost before any shader integration.

