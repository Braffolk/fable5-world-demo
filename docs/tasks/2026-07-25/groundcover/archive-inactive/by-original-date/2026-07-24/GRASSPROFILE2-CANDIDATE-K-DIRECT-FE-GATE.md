# Candidate K direct finite-element carrier: stored-node gate

Date: 2026-07-24  
Status: **GREEN for stored-node spatial reduction and quantisation; held-out angular gate still required**  
Scope: mathematics/cook-side codec only; no runtime, shader, or installed asset changed

## Codec measured

Candidate K removes representative grass depth from the unresolved carrier.
Fractional pixels retain the real background depth/id, and footprint `rho` is
derived from the cover-entry Jacobian.  The filtered carrier therefore stores
only coverage and premultiplied authored colour.

One angular-cell texel is `RGBA32Uint`.  Its four channels are the cell's four
shared angular vertices.  In every channel, the low and high 16-bit halves are
the `sigma=4` and `sigma=16` vertices:

```text
vertex16 = A4 | premul-R4 | premul-G4 | premul-B4
```

The cook quantises each angular vertex once and copies those exact bits into
all incident cell pages.  Ordinary row/azimuth cells use bilinear non-negative
partition weights.  A pole-cap cell repeats the one canonical pole vertex in
both pole corners.  Consequently adjacent cells evaluate bit-identical traces
on every shared edge, and the pole limit is azimuth-independent: angular `C0`
continuity is exact rather than a measured approximation.

This gate tests the other loss source: reduction of each accepted 256-square
direction field to the candidate page resolution followed by direct
premultiplied RGBA4444 quantisation.  Coverage and premultiplied RGB are
integrated with an exact separable area kernel.  Reconstruction samples the
candidate page at every original phase centre.  Every one of the 65 accepted
direction nodes is scored separately at both spatial scales.

## Frozen limits and result

```text
absolute coverage error:              p95 <= 0.08, p99 <= 0.20
premultiplied RGB max-channel error:   p95 <= 0.06, p99 <= 0.15
largest toroidal connected p99 region: < 1% of evaluated support
```

All four proposed memory configurations are GREEN:

| Pages | Resolution | Scale memory | Scale | Worst `A` p95/p99 | Worst premul RGB p95/p99 | Max connected | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 180 | 31.641 MiB | 4 | 0.0667 / 0.0863 | 0.0495 / 0.0604 | 0.000% | GREEN |
| 64 | 180 | 31.641 MiB | 16 | 0.0353 / 0.0431 | 0.0357 / 0.0390 | 0.000% | GREEN |
| 80 | 160 | 31.250 MiB | 4 | 0.0706 / 0.0902 | 0.0511 / 0.0627 | 0.000% | GREEN |
| 80 | 160 | 31.250 MiB | 16 | 0.0353 / 0.0431 | 0.0360 / 0.0398 | 0.000% | GREEN |
| 96 | 144 | 30.375 MiB | 4 | 0.0745 / 0.0980 | 0.0540 / 0.0709 | 0.000% | GREEN |
| 96 | 144 | 30.375 MiB | 16 | 0.0392 / 0.0471 | 0.0367 / 0.0409 | 0.000% | GREEN |
| 128 | 128 | 32.000 MiB | 4 | 0.0706 / 0.0902 | 0.0543 / 0.0682 | 0.000% | GREEN |
| 128 | 128 | 32.000 MiB | 16 | 0.0392 / 0.0431 | 0.0368 / 0.0397 | 0.000% | GREEN |

The direct premultiplied channels are load-bearing.  The rejected 128-square
`A5 + height7 + palette4` carrier had worst premultiplied-RGB p95 `0.1560`
and representative-position p95 `0.4733 m` at `sigma=4`; the event-changing
depth field cannot survive a 2:1 spatial collapse.  K works because fractional
depth is not needed and colour is stored as the positive filtered moment,
rather than reconstructed from a categorical palette owner.

## What is and is not proved

This gate proves:

- the direct carrier fits the unchanged scale-memory ceiling;
- its stored-node phase reduction and quantisation meet the frozen fidelity
  limits independently for every accepted direction;
- premultiplied physicality is retained (`0 <= RGB <= A`) because equal-step
  monotone quantisation and convex angular interpolation preserve it;
- canonical shared vertices give exact angular continuity at all old row,
  azimuth, and pole boundaries;
- the scale read remains one physical `RGBA32Uint` operation and supplies both
  scales.

It does **not** yet prove:

- fidelity at directions inside the angular cells.  Extra direct-truth bakes at
  row and azimuth midpoints must gate 64/80/96/128-cell topologies before the
  page count is frozen;
- the grazing/horizon construction.  A finite exterior-angle contract cannot
  silently clamp directions below the old 15-degree row.  Additional pages
  must encode an explicit low-angle ring/fringe or a separately proved
  horizon-limit chart;
- temporal stability under spatial page-cell crossings.  The integer atlas is
  nearest/unfilterable, so a 1--4.5 mm translated-camera sequence remains a
  required offline and live gate even though its source fields are filtered;
- exact unresolved lighting.  The admissible no-data approximation is an
  axisymmetric two-sided foliage distribution about the wind-sheared growth
  axis; it must not masquerade as an exact face normal.

Numeric `A=1` in a filtered vertex remains fractional/background-depth mode.
Only the categorical exact branch may write foreground grass depth.

## Reproducible artifacts

Analyzer:

```text
tools/groundcover-bake/analyze_candidate_k_direct_fe.py
```

Content-addressed result:

```text
data/work/groundcover-candidate-k-direct-fe/
  e3e0a4175b151b89/
    6e8ba84eda853dab/
      SUMMARY.md
      report.json
      qa/
        001-64pages-180px.png
        002-80pages-160px.png
        003-96pages-144px.png
        004-128pages-128px.png
        index.json
```

Hashes:

```text
source   e3e0a4175b151b89a1a4bb58089aed2e809da337a04ce024240be8decdacb0be
recipe   6e8ba84eda853dab734242f1a61f942aee51e3ed2113bfbc24bc981a285d834a
analyzer ed8df3ff9b5d22486ec80081798567053331d18ad0cb4c340031d5f8a415895a
report   58ad8de8b8cc7c6c5a6f812bdf66d672b6a4ba6849ec81bf94b13cd35c234e04
```

Exact command:

```text
env UV_CACHE_DIR=/tmp/laas-uv-cache uv run --project asset-gen python tools/groundcover-bake/analyze_candidate_k_direct_fe.py
```

