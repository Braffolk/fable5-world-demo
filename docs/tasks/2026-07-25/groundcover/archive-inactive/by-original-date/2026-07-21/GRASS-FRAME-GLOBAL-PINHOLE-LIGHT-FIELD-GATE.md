# Frame-global pinhole surface/light-field gate

Date: 2026-07-22  
Decision: **reject and park after one actual-source image attempt**  
Runtime/shader changes: none

## Question

Can the camera-front wedges produced by a per-pixel angular selector be removed
without search by choosing at most four canonical directions once per frame,
projectively deforming every selected canonical first-event sheet through one
fixed botanical mid-plane, and globally blending only premultiplied measures?

This is deliberately stronger than another per-pixel angular cone. Every pixel
uses the same direction indices and weights. There is no per-pixel owner vote,
candidate list, march, loop, shell, or extra pass.

## Exact map that was tested

Let the single fixed carrier be

\[
\Pi=\{Q:n\cdot Q=k\}.
\]

For one canonical direction `c`, its exterior first-event field is written as

\[
P_c(Q)=Q+c\,\sigma_c(Q).
\]

For a pinhole camera at `C`, set `b=k-n dot C`. The unique point on the camera
line through `Q` whose carrier-normal coordinate equals that of `P_c(Q)` is

\[
\boxed{
F_C(Q)=Q+(Q-C)\frac{(n\cdot c)\sigma_c(Q)}{k-n\cdot C}
}.
\]

Indeed, with `a=(n dot c)sigma/b`,

\[
F_C(Q)-C=(1+a)(Q-C)
\]

is on the pinhole ray, while

\[
n\cdot F_C(Q)=k+(n\cdot c)\sigma_c(Q)=n\cdot P_c(Q).
\]

The implementation uses this identity directly. A live unit ray intersects the
mid-plane at distance `t_Q`; the reconstructed distance is exactly
`t=t_Q(1+a)`. No approximate view/projection matrix or screen-space
reprojection occurs in the gate.

For carrier basis `E`, `gamma=(n dot c)/(k-n dot C)`, and carrier gradient
`grad sigma`, the exact stitched differential is

\[
\boxed{
D_uF_C=(1+a)E+(Q-C)\,\gamma(\nabla\sigma)^T.
}
\]

Normals were intentionally not displayed by this gate. Any future use of this
map that displays normals must obtain them from the cross product of the two
columns above, or the equivalent inverse-transpose. A per-ray rank-one normal
is not the stitched surface normal.

## The decisive distinction: exact projection, wrong surface

The equations above are exact for the **surface defined by the chosen
canonical scalar sheet**. They do not prove that this sheet is the surface
visible from `C`. Equality to actual-source truth would require the canonical
first-event chart and the live first-event chart to retain the same finite
surface across the whole camera footprint. Dense grass violates that condition
at ordinary occlusions, disocclusions, blade crossings, panicle hairs, and tile
successor changes.

If the canonical sheet has scalar event error `delta sigma`, its reconstructed
position error contains

\[
\delta F_C=(Q-C)\frac{n\cdot c}{k-n\cdot C}\,\delta\sigma.
\]

Thus a coherent canonical error is explicitly directed along rays radiating
from the camera and grows with `|Q-C|`. This predicts the observed widening
V-shaped streaks and distance-growing wrong perspective. The streaks are not a
WebGPU, raster, depth-format, or floating-shell defect: they occur in an f64
CPU rendering of the isolated exact source.

Selecting directions once per frame successfully removes angular Voronoi
boundaries in screen space. It cannot make two unrelated first-event sheets
agree. Global interpolation of their coverage and premultiplied radiance merely
superposes incompatible silhouettes. It avoids illegal owner averaging, but
it does not restore first-event topology.

## Frozen gate

- Immutable isolated production Calamagrostis SHA-256:
  `8cd69c2a6c61043c2861cad3a5ec06fee6dd4346a6722fd01fcc43468071ce5d`.
- Actual authored 333,520-vertex / 353,245-triangle source, repeated on its
  `0.52 m` lattice.
- One fixed mid-plane at `y=0.49255296619984323 m`.
- Canonical support: 16 azimuths, `5/15/35/55/75` degree rings, plus a vertical
  singleton; one to four directions selected from the optical axis once per
  camera.
- Static truth/reconstruction/error images: narrow 18-degree, wide-FOV
  18-degree, 2-degree grazing exterior, 82.5-degree top-down, and exact
  horizontal camera-inside.
- Motion truth/reconstruction images: seven 18-degree frames separated by
  `2 mm` lateral camera motion.
- Continuous f64 canonical events and exact periodic BVH truth. Atlas
  quantisation, filtering, packed depth, packed normals, and shader arithmetic
  cannot be blamed for the result.
- One end-to-end attempt. No correction/rerun was made after seeing the images.

The acceptance requirements were coverage IoU at least `0.90`, panicle recall
at least `0.90`, ghost alpha at most `0.05`, depth p95 at most `0.02 m`, largest
connected ghost at most `1%` of the image, and mean premultiplied motion-delta
error at most `0.03`.

## Results

| camera | coverage IoU | ghost alpha | panicle recall | depth error p50 / p95 | largest connected ghost |
|---|---:|---:|---:|---:|---:|
| narrow 18 degrees | 0.1777 | 0.6533 | 0.2356 | 0.746 / 3.294 m | 2.639% |
| wide 18 degrees | 0.1102 | 0.7702 | 0.0930 | 1.481 / 27.399 m | 1.603% |
| grazing 2 degrees | 0.2154 | 0.5877 | 0.2024 | 2.038 / 26.184 m | 1.139% |
| top-down 82.5 degrees | 0.0757 | 0.8622 | 0.0029 | 0.250 / 0.841 m | 0.062% |

The seven-frame `2 mm` path has mean premultiplied delta error `0.09638`, over
three times its `0.03` limit. Direct inspection of every numbered image agrees
with the measurements:

- the narrow reconstruction retains a central plant fragment but turns other
  blades and panicles into camera-radial V shapes;
- the wide and grazing cases become long converging streak fields rather than
  coherent plants;
- global weighting overlays wrong sheets instead of retaining fluffy panicle
  silhouettes;
- the first/middle/last path reconstruction moves a different V-shaped field
  than the actual source.

## Horizontal and camera-inside boundary

When the camera lies on the carrier, `k-n dot C=0`. An exact-horizontal ray
also has either no carrier intersection or an entire line of them, rather than
one finite `Q`. More fundamentally, an exterior first-event sheet does not
encode the pointed-line successor after an arbitrary inside origin. The gate
therefore marks exact-horizontal camera-inside as `FAIL_BY_DOMAIN`; it is not
clamped to five degrees, flattened, or hidden.

A second dominant-axis carrier could remove this coordinate pole. It would not
repair the exterior failures above: each new carrier still needs a compatible
first-event chart, and overlap must return the same marked event. Adding a
carrier without that invariant only moves the wrong-sheet boundary.

## Fixed resource accounting

The rejected proposal would fit the frozen performance budget:

- 81 wrapped `258x258` eight-byte complete-record fields:
  `43,133,472` resident bytes, below the `51,121,152`-byte cap;
- at most four complete filtered field samples per pixel;
- estimated ceiling `224` FMA-equivalent operations including the exact
  stitched normal;
- one texture-array binding and sequential accumulation;
- no data-dependent loop, march, candidate election, live triangle, shell,
  extra pass, dispatch, barrier, or species query.

The method is rejected for surface correspondence and domain coverage, not for
cost. Increasing directions, samples, carriers, or per-pixel selection would
leave this bounded experiment and is not authorised by its result.

Overlapping grass species, litter, and moss would still be unioned into one
marked community before baking. They must not multiply runtime queries. This
gate supplies no accepted reconstruction for that future community.

## Artifact and provenance

- Tool:
  `tools/groundcover-bake/analyze-frame-global-pinhole-light-field.ts`, SHA-256
  `c5e6d05f0c20657af89b8b0e0db27b1e17b6bb1de3086159fb122fb67554d3f7`.
- Artifact:
  `data/work/groundcover-frame-global-pinhole-field/8cd69c2a6c61043c/3de6240997bff07d/`.
- Report SHA-256:
  `0b51aa25c77ec4a97c94288bb82c93a70cbceb6bda5fb277a8bb029a88067ccd`.
- `qa/001..005` are labelled actual-source truth / reconstruction / absolute
  error panels. `qa/006..007` are truth and reconstruction at the first,
  middle, and last millimetric path frames. `qa/index.json` binds dimensions,
  interpretations, and image hashes.

Prior-art boundary:

- Lin and Shum (2004), *A Geometric Analysis of Light Field Rendering*,
  supplies the geometry-assisted light-field reconstruction context and the
  visibility/disocclusion boundary.
- Sannikov's published grass work supplies the broader fixed-cost precomputed
  ray-field target, not this frame-global pinhole map or blending rule.
- The fixed-mid-plane conjugacy, exact stitched Jacobian, frame-global angular
  contract, actual-source periodic image gate, connected stretch/ghost metrics,
  millimetric path test, cost freeze, and diagnosis above are LAAS-original.

## Decision

**Reject and park.** The exact map coherently projects a canonical sheet, but
that sheet is not the actual surface visible along held-out live rays. The
result directly recreates the user's prohibited distance-growing stretch and
wrong perspective even with perfect f64 events. No runtime integration,
direction increase, per-pixel selector, filter, correction pass, or second
attempt follows.
