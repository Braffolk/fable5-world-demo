# Candidate K held-out angular FE — RED blocker

Date: 2026-07-24  
Status: **PARKED after one full screen plus one premise-audit cycle**  
Scope: offline mathematics/oracle only; no runtime or shader edits

## Outcome

Candidate K's one-read direct `RGBA4444` appearance codec is rejected for the
accepted Calamagrostis community.  The direct packing is adequate at stored
vertices.  The angular field between vertices is not.

The corrected full `64 x 64` phase screen used the physical equivalents of the
accepted 256-page kernels (`sigma 4 -> radius 1`, `sigma 16 -> radius 4`), the
botanical middle reference plane `h_ref=0.49255 m`, all 129 global angular
vertices, and 112 held-out directions from 0.05 through 89 degrees.  Results:

```text
                         sigma 4             sigma 16
held-out cases           112                 112
unlimited FE failures    108                 104
RGBA4444 failures        108                 108
worst coverage p95       0.667               0.438     (limit 0.08)
worst RGB p95            0.490               0.323     (limit 0.06)
worst connected region   70.1%               24.2%     (limit <1%)
stored-node packing RED  0 / 129             0 / 129
```

The internal control is decisive: at 0.25 degrees, sector-edge directions that
are actual stored vertices reproduce exactly in unlimited precision, while the
sector midpoints fail.  This exonerates the exact periodic BVH, field-address
construction, filtering, and source colour decode.  It isolates angular
interpolation.

The horizon premise also fails as an appearance field.  At
`0.05/0.1/0.2/0.25/0.5` degrees coverage is nearly opaque and similar, but
premultiplied-RGB p95 remains about `0.27`.  The deterministic periodic soup
does not supply one usable shallow-angle appearance value merely because
coverage converges.

## Premise-audit cycle

The single permitted diagnose/fix cycle raycast fresh endpoint and midpoint
truth while independently halving azimuth or elevation cell widths to
`1, 1/2, 1/4, 1/8` of the Candidate-K cells.  It used worst and median cells in
eight elevation families, both physical kernels, and no quantisation.

Worst normalized p95 (`max(A_p95/.08, RGB_p95/.06)`):

```text
axis / scale        current     1/2       1/4       1/8
azimuth / sigma 4     9.72      8.33      7.64      7.02
azimuth / sigma 16    6.40      5.08      3.84      3.22
elevation / sigma 4   8.05      6.25      4.86      4.17
elevation / sigma 16  5.00      3.40      2.12      1.47
```

No family-wide case is green at one-eighth width.  A uniform atlas with both
axes refined by eight already has

```text
128 * 8 * 8 = 8192 pages
8192 * 128^2 * 16 B = 2048 MiB for the scale atlas alone,
```

sixty-four times Candidate K's 32 MiB scale allocation, and the measured field
is still RED.  Log-slope extrapolations from the last two samples range from
five to twenty-seven total halvings and are too unstable to claim as an exact
requirement; they only reinforce that a dense uniform angular grid is not a
credible rescue.  The measured 2 GiB still-RED point is the defensible bound.

## Middle-plane premise audit

A small, matched middle-versus-top reference-plane ablation covered horizon,
standing, 30--55 degree, and pole-cap cells on both axes and scales:

```text
reference        median normalized p95   mean       worst
middle plane            4.999            5.013      9.722
top plane               4.356            4.658      7.927
```

The centered plane helps some cap cases and hurts others; it is neither a
uniform improvement nor a solution to the visibility event structure.  This
does not invalidate centered phase for other representations.  It rejects the
claim that centered phase makes this four-corner direct appearance FE viable.

## Invalidated diagnostics

Early 4/16-square smokes and an interrupted 128-square run incorrectly treated
`4/16` as radii at the test resolution.  They changed physical footprint and
were discarded before any verdict.  The corrected recipes record source and
effective radii explicitly.  A later corrected 128 run was intentionally
stopped after the massive, internally controlled 64 RED; the premise audit was
more informative than repeating the same failure at four times the rays.

## Artifacts and provenance

- Full corrected screen:
  `data/work/groundcover-candidate-k-heldout-fe/e3e0a4175b151b89/60a8e06b5f69e390/`
- Convergence and reference-plane audit:
  `data/work/groundcover-candidate-k-angular-convergence/e3e0a4175b151b89/18fafa8984a3c8ac/`
- QA index:
  `data/work/groundcover-candidate-k-angular-convergence/e3e0a4175b151b89/18fafa8984a3c8ac/qa/index.json`
- Frozen thresholds:
  `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-CANDIDATE-K-HELDOUT-THRESHOLDS-FROZEN.md`
- Source SHA-256:
  `e3e0a4175b151b89a1a4bb58089aed2e809da337a04ce024240be8decdacb0be`

## Exact blocker and resume condition

Blocker: the filtered premultiplied appearance of the accepted arbitrary
triangle soup contains coherent angular visibility events that a four-corner
bilinear field does not approximate at the 128-page budget.  RGBA4444 is not
the cause, centered phase is not the cure, and more uniform angular pages miss
the memory contract by orders of magnitude before the measured error is green.

Resume only with a mathematically different **direction-analytic or
event-adaptive** representation that demonstrates, before runtime work, all of:

1. structural continuity in direction,
2. held-out appearance fidelity against the exact periodic soup,
3. one small fixed-cost read or an explicitly re-authorised cost,
4. a measured resident allocation near the current 32 MiB scale budget.

Do not resume by changing thresholds, adding uniform rings, increasing taps,
or increasing memory.
