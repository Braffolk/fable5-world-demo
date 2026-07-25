# GBR4/v4 production RED blocker

**Date:** 2026-07-23  
**Decision:** park the dense/sparse certified-macrobrick production fit; retain
the loader, cooker, certifier, real asset, QA, and measurements.

The complete result and artifact paths are in
`GRASS-GBR4-V4-COUPLED-REFERENCE-RESULT.md`.

## Exact mathematics implemented

- Six guarded box-boundary face charts with one frozen lowest-face-id tie.
- A bounded Lambert equal-area closed-hemisphere disk, including side and
  horizontal addresses.
- Four independently integrated shared-origin pinhole footprint levels.
- Tagged `INVALID`, certified `MISS`, certified `REGULAR`, and physically
  filtered `MIXED` records.  Mixed records contain premultiplied appearance,
  a conservative depth interval, depth/normal/material moments, and no fake
  surface.
- A continuum regular-cell sufficient certificate: bound the complete
  boundary-position and direction product cell, bound analytic winner-plane
  intersection/support/incidence, and exclude every earlier periodic-copy
  competitor from the complete swept pre-hit capsule through the BVH.
- A fixed perfect-hash sparse correction read.  A hit reads five coupled
  plane/root/copy/primitive/material/certificate texels; a miss reads one base
  tag and four mixed texels.  Both schedules cost six terminal reads and add
  no loop, march, candidate list, or runtime geometry.

## Terminal measurement

At the closest physical footprint, using `4096^2` boundary and `65535^2`
direction correction cells, the deterministic upper-face/vertical gate found:

- `8 / 4096` tested cells regular: **0.1953125%**;
- `8 / 785` center-hit cells regular: **1.0191083%**.

The actual coarse asset contains `1,536` filtered-mixed base records and eight
continuum-certified regular corrections.  The all-mixed coarse field is not
visually bindable: its cells are far larger than the smallest physical pixel
footprint, so unresolved plant edges cannot legally terminate there.
The container itself carries publication enum `REFERENCE_RED` and the strict
loader exposes `runtimeBindAllowed=false`; this is not delegated to an
external report or filename convention.

Sparse corrections do not close the visible domain.  A dense table at the
demonstrated physical certificate scale would contain
`1,729,329,480,754,790,400` cells; its four-byte addresses alone are roughly
`6 EiB`, before payloads, carrier data, terrain closure, or mips.  The measured
regular occupancy is also too low to claim that sparse lossless corrections
fit the remaining domain.  Binding the 56 KiB reference would therefore
recreate the invalid coloured-sheet result under a more honest file format.

## Reusable result and resume condition

Retained:

- strict v4 loader and exact GPU-byte accounting;
- real 57,600-byte Calamagrostis reference asset;
- six-face/Lambert/four-footprint cooker;
- conservative periodic continuum certifier and sparse perfect-hash table;
- focused corruption/loader tests and content-addressed QA.

Resume only when a complete adaptive macrobrick/VQ cook covers every
unresolved physical-footprint cell, keeps certified regular payloads lossless,
and passes the held-out image/depth/seam/1--4.5 mm translation gates with all
carrier, payload, mip, and upload bytes at or below 250 MiB.  The coarse
all-mixed reference is never a fallback visual implementation.
