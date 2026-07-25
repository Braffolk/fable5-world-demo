# Candidate L targeted 4D rate gate — frozen before result

Date: 2026-07-24  
Status: preregistered; no result when written

## Purpose

This is the smallest real-GCRP test which can justify a larger Candidate-L
cook.  It asks whether a conforming spatial-triangle × slope-triangle field,
compressed to one five-vertex 4-simplex, follows epipolar shear materially
better than Candidate K's same-phase angular interpolation.

It is not a codec or runtime test.  All values are unlimited precision.

## Frozen source and physical truth

- source: `src/assets/groundcover/calamagrostis-canescens.gcrp`;
- exact periodic first hit through `TriangleBvh` and
  `periodicNearestSuccessor`;
- phase truth resolution: `128 x 128`;
- reference height: `0.49255 m`;
- positive premultiplied `(P_r,P_g,P_b,A)` only;
- physical square supports: exactly `9/256` and `33/256` of the periodic tile
  per axis, matching the accepted canonical radius-4 and radius-16 boxes.
  Fractional edge-cell weights on the 128 grid preserve those widths exactly;
  integer-radius substitution is forbidden.

## Frozen representative direction cells

These are the prior angular-convergence audit's worst representatives for the
two primary live regimes:

```text
grazing-worst:   elevation [0.25, 2] deg, azimuth [67.5, 90] deg
standing-worst:  elevation [10, 18] deg, azimuth [247.5, 270] deg
```

Each angular quadrilateral is split on the low-left to high-right diagonal.
Interpolation coordinates are barycentric in slope
`s=d_xz/(-d_y)`, not in degrees.  Each direction triangle is evaluated at its
centroid, three asymmetric interior quarter sites, and its shared-diagonal
midpoint.  Stored vertices are excluded from the fidelity verdict.

## Frozen spatial rate and sites

The candidate spatial torus has `64 x 64` vertices, embedded exactly in the
truth page at a two-truth-texel pitch.  Every square is split on the
low-left to high-right diagonal.  All `128 x 128` truth sites are
evaluated, including spatial vertices, interiors, square edges, and triangle
diagonals.  Product-simplex internal boundaries also receive a numerical
`+/- epsilon` continuity check.

The initial 256-grid execution was stopped before any result after the
measured shallow-page time projected beyond the one-hour targeted-track cap.
This resolution change was made before observing any interpolation score; the
physical supports, FE rate, direction cells, and thresholds are unchanged.

## Frozen predictors

1. `K-same-q`: Candidate-K four-corner angular bilinear interpolation at the
   exact held-out phase.  This deliberately grants K perfect spatial access;
   it is a stronger baseline than its live nearest-phase field.
2. `L-simplex5`: the five shared product vertices selected by the standard
   staircase triangulation of spatial triangle × slope triangle.
3. `tensor9`: spatial-triangle interpolation at all three direction vertices,
   followed by direction-triangle interpolation.  This nine-product-vertex
   result is a rate ceiling, not a runtime candidate.

## Frozen thresholds

For every representative cell, direction triangle, held-out direction, and
physical support independently:

```text
coverage absolute error:       p95 <= .08, p99 <= .20
premul RGB max-channel error:   p95 <= .06, p99 <= .15
connected p99 exceedance:       < 1%
```

Candidate L is **GREEN for expansion** only if:

1. every `L-simplex5` case passes all absolute limits;
2. its worst normalized p95 is at most `0.75` of `K-same-q`'s worst normalized
   p95 (at least 25% material improvement); and
3. its worst normalized p95 is at most `1.25` times `tensor9`'s, unless both
   are already below `1.0`.

The product-simplex continuity residual must be `<= 1e-5` per component.

If `tensor9` is RED, the selected spatial/direction rate or reference chart is
insufficient; simplex packing is not blamed.  If `tensor9` is GREEN and
`L-simplex5` is RED, five-vertex simplex compression is the blocker.  If both
are GREEN but the material-improvement condition fails, Candidate K's failure
is not epipolar coupling at this tested rate and Candidate L is not expanded.

No result authorises a runtime edit, more than nine reads, or more than
`48.75 MiB`.
