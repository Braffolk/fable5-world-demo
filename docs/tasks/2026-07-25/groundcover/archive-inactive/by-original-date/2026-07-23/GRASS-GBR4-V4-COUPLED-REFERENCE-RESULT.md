# GBR4/v4 coupled boundary-transfer reference result

**Date:** 2026-07-23  
**Status:** **RED for visual production; GREEN for the versioned record,
loader, bounded-chart, fixed-read, and sparse continuum-certificate plumbing.**

This is the finite-codec result produced after the complete runtime
conformance audit.  It does not bind a new visible grass path.

## Inspectable artifact

- stable container:
  `src/assets/groundcover/calamagrostis-canescens.reference-v4.gbr4`
- stable SHA-256:
  `53f4e14d966392c56ba7e2092014853a782155f7de9b81ea68b15464deaaa854`
- content-addressed result:
  `data/work/groundcover-gbr4-v4-coupled-reference/2ed57f59d86e8376/144b0d0f4a075647/`
- container bytes: `57,600`
- exact GPU upload bytes: `56,448`
- active runtime binding: **none**

The loader parses this real Calamagrostis asset and constructs the complete
upload views.  Two focused tests pass, including corruption of a sparse
regular correction into an out-of-range payload.

## What v4 changes

The old v2 record numerically mixed appearance across unrelated histories and
then assigned that mixture one nearest categorical depth and normal.  v4
makes that state impossible:

- every address is tagged `INVALID`, `CERTIFIED_MISS`,
  `CERTIFIED_REGULAR`, or `FILTERED_MIXED`;
- a regular payload couples source plane, root, periodic copy, triangle,
  primitive, material/species, deformation class, colour, and source
  support/incidence/order margins;
- a mixed payload contains only premultiplied appearance/coverage, a
  conservative depth interval plus depth moments, normal moments, and colour
  moments; it has no categorical surface;
- six boundary-face charts include cap, side, and horizontal addresses;
- direction uses the bounded Lambert equal-area closed hemisphere disk;
- `0.025`, `0.25`, `4`, and `32 m` standoff footprints are independently
  integrated from shared-origin pinhole rays;
- a correction hit is one perfect-hash read plus five coupled regular payload
  reads; a miss is one hash read, one base tag, and four mixed payload reads:
  at most **six terminal reads** for this reference asset, with no loop,
  march, candidate list, or runtime geometry.

The regular correction is not based on corners.  For the complete product
cell the offline certifier bounds boundary position and Lambert direction,
bounds the candidate plane intersection and triangle-domain support, then
enumerates every periodic-copy triangle whose AABB overlaps the complete
pre-hit swept capsule.  It publishes the cell only when every competitor has
strictly positive clearance.

## Measured regular result

The permitted diagnosis/fix cycle first tested `B=512`, `D=8191` and found
zero certified regular cells.  The physical-pixel-scale correction lattice
was then tightened to:

- boundary resolution `B=4096`;
- Lambert-disk direction resolution `D=65535`;
- the closest `0.025 m` footprint; and
- a deterministic `64 x 64` upper-face/vertical subset.

Results:

- tested continuum cells: `4,096`;
- center rays with a source hit: `785`;
- certified regular cells: **`8`**;
- regular / tested: `0.1953125%`;
- regular / center-hit: `1.0191083%`.

This proves that the regular record class is constructive and nonempty on the
accepted `2,171,134`-triangle source.  It is not a whole-domain regular
fraction estimate.

## Why visual production is RED

The coarse direct base has `1,536` cells and all remain honestly mixed.  Those
cells are much larger than the smallest supported pixel footprint, so they
cannot legally terminate the unresolved near-field discontinuities.  The
sparse correction experiment found only eight strict regular cells in the
tested physical-resolution subset.

A dense direct allocation at the demonstrated physical certificate scale
would require

```text
4 levels * 6 faces * 4096^2 boundary cells * 65535^2 direction cells
= 1,729,329,480,754,790,400 cells.
```

Four address bytes alone would be about `6 EiB`, before payloads, carrier,
terrain charts, or mips.  This is not remotely close to the `250 MiB` ceiling.
The result therefore rejects a dense tensor at the physical footprint; it
does **not** reject the continuous boundary-transfer identity.

The objective resume condition for this codec is a complete adaptive
macrobrick/VQ cook which covers every unresolved cell, retains the certified
regular payloads losslessly, independently integrates its footprint levels,
and passes the held-out image/depth/seam/mm-translation gates with total
resident bytes at or below `250 MiB`.  Until that exists, this v4 asset is a
loadable semantic and certifier reference only and must not be exposed as
grass.

The remaining whole-product gaps are also explicit:

- no finite ecological-patch `rho_M`/first-entry closure asset;
- no positive terrain/deformation continued-firstness margin (the eight
  source-flat regular records carry zero terrain margin);
- no converged mixed-cell quadrature or held-out visual gate; and
- no runtime binding.
