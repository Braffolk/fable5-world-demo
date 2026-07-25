# One-read 32-bit exterior light-field fallback gate

Date: 2026-07-22  
Decision: **reject and park both concrete allocations**  
Scope: actual accepted Calamagrostis asset, offline bake/truth simulation only;
no runtime or shader file was changed.

## Inspectable outcome and question

This is the bounded pragmatic check of the strongest simple discretized
light-field fallback: pack each already-baked first hit into one categorical
`u32`, recover half the former bytes per record, and spend almost the entire
`51,121,152`-byte cap on denser direction support without destroying the
millimetric phase lattice.

The test asks whether this very cheap exterior field is at least visually
plausible when its address is mathematically correct. It deliberately does
**not** use camera-centred carrier deformation. The live line is intersected
with two fixed botanical Y planes; their upper XZ intersection supplies
periodic phase and their difference supplies slope. One nearest category is
read and its vertical drop is relifted along the exact live ray.

Both allocations preserve line-datum invariance exactly. Both nevertheless
produce multi-metre connected wrong-view sectors, lose panicle classes, and
change predicted material under millimetric camera motion. Denser discrete
angles do not make the one-record event field a plausible fallback.

## Immutable source, tool, and artifacts

- Source: `src/assets/groundcover/calamagrostis-canescens.gcrp`, GCRP/v4,
  `118,663,280` bytes, `2,171,134` triangles, SHA-256
  `2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c`.
- Evaluator:
  `tools/groundcover-bake/analyze-packed-light-field-fallback.ts`, SHA-256
  `9fb5bbf6591b4fbd3000c7d8fa6814ef81f0b986894baac3c3e88d2bfc0bb0c0`.
- Recipe SHA-256:
  `a3ce1c6b881bcc30aa4bb7990d8c76ef097a6a2969634c4fb5942b3bc165aded`.
- Artifact root:
  `data/work/groundcover-packed-light-field/2ed57f59d86e8376/a3ce1c6b881bcc30/`.
- `metrics.json` SHA-256:
  `35bb85fcd61d0672a3243694101f364a17fe31b63c9d90004d83fed9970856fc`.
- `index.json` SHA-256:
  `c2e22a06b50744f89d499f6797afaa5c0749fce0540b920bd7413cdfa57ce9aa`.

Reproduce with:

```sh
node --import tsx tools/groundcover-bake/analyze-packed-light-field-fallback.ts
```

The two numbered QA contact sheets are:

1. `qa/001-phase-rich-p255-d193-truth-prediction-error.png`, SHA-256
   `d0633e0e2c69a59f3325cac70489edb0ccbe27e91b9f9bb32ad875f5a858c23e`;
2. `qa/002-angle-rich-p220-d257-truth-prediction-error.png`, SHA-256
   `1326c96fc7189209ea5455dd44ea449ba0edd7d4ccafb7a185ab9124b1071445`.

Rows follow the six camera sweeps below. Each row shows exact authored-colour
truth, decoded `RGB343` prediction, then error class: green below 5 cm, yellow
below 25 cm, orange below 1 m, red at or above 1 m, and magenta for hit/miss
disagreement.

## Exact 32-bit record and fixed runtime work

One complete categorical record is:

| field | bits | meaning |
|---|---:|---|
| hit/coverage | 1 | binary first-event presence |
| vertical drop | 11 | `topH - hitY` over the botanical slab |
| oct normal X/Y | 5 + 5 | one decoded unit normal |
| material/colour class | 10 | direct `RGB343`, 1,024 categorical values |

There is no arithmetic blending of depth, normal, colour, coverage, or owner.
There is no auxiliary owner, triangle, geometry, or palette-table read.

The proposed fixed live path is one upper/lower-plane intersection, periodic
XZ wrap, slope radius/angle, one radial-band and sector quantisation, **one
nearest `u32` texture read**, decode, and one division by the live vertical ray
speed. The 8/9 ring-count choice is a compile-time fixed select, not a runtime
search. The offline evaluator loops and BVH are bake/truth machinery only.

No runtime loop, march, traversal, candidate list, species query, extra pass,
or distance-dependent operation is proposed.

## Two concrete cap-filling allocations

Directions are polar samples in the two-plane slope coordinates
`(dx/-dy,dz/-dy)`, not uniform camera angles. Rings cover exterior downward
elevations from 5 to 90 degrees. Ring sectors grow with slope radius so low
oblique views receive most directions.

| allocation | phase | directions | ring sectors | stored records | bytes | cap left |
|---|---:|---:|---|---:|---:|---:|
| phase-rich | 255² | 193 | `3,9,15,21,27,33,39,45` + vertical | 12,747,457 | 50,989,828 | 131,324 |
| angle-rich | 220² | 257 | `3,9,16,22,28,35,41,48,54` + vertical | 12,665,988 | 50,663,952 | 457,200 |

The one-texel wrapped gutter is included. Phase texels are `2.039 mm` and
`2.364 mm`. Measured phase-coordinate p50 errors are only `0.807 mm` and
`0.942 mm`; the allocations really do retain high phase resolution.

## Actual-asset camera sweeps

The exact decoded-u16 mesh and periodic copies supply analytic truth. Six
pinhole exterior sweeps each use four frames translated by
`0, 1, 2.5, 4.5 mm`, with a `24 x 16` ray bundle per frame:

- 2.5 degrees, deliberately outside the 5-degree finite slope domain;
- 5.5 degrees grazing;
- 10 degrees low oblique;
- 25 degrees oblique;
- 55 degrees high;
- 80 degrees near top-down.

There are 9,216 rays per allocation. A reconstructed true positive is compared
with truth on the **same exact live ray**; its Euclidean world-position error
is therefore not a proxy based on a different camera or carrier.

### Positive result: the line address is correct

For 3,072 independent datum shifts per allocation, the same oriented line was
reparameterised from points `-0.37`, `+0.19`, and `+1.31 m` along it. Recomputed
upper/lower plane intersections drifted by at most
`1.0695e-14 m` and produced **zero atlas-address mismatches**.

This rules out camera-centred address deformation as the cause of the failures
below. It is reusable evidence for a future representation: fixed two-plane
line coordinates preserve exterior line identity.

## Decisive geometry and coherent-fan failure

Aggregate true-positive world-position error is already fatal:

| allocation | agreement | recall | position p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| phase-rich P255/D193 | 75.16% | 76.80% | 1.557 m | 7.989 m | 16.851 m |
| angle-rich P220/D257 | 77.50% | 78.56% | 1.605 m | 9.535 m | 18.838 m |

The out-of-domain 2.5-degree row is not needed for rejection. Inside the
declared 5-degree boundary:

| sweep | phase-rich p50 / p95 | angle-rich p50 / p95 |
|---|---:|---:|
| 5.5 degrees | 1.597 / 7.808 m | 1.176 / 7.482 m |
| 10 degrees | 1.462 / 4.739 m | 1.527 / 5.111 m |
| 25 degrees | 1.113 / 2.066 m | 0.927 / 1.966 m |

The 5.5-degree hit/miss agreements are deceptively high: `99.09%` and
`98.50%`. The record usually says “grass exists,” but identifies a first event
whose height and appearance belong to a different view line. This is precisely
why coverage alone can look numerically good while the image becomes a
stretched sheet.

To measure that structure explicitly, an error fan is a 4-connected camera
region whose pixels select the same canonical angular bin and whose predicted
world positions exceed an error threshold. Its width is the diagonal of those
predicted positions' world-space bounding box. At the 1 m threshold:

| sweep | phase-rich fan p50 / max | angle-rich fan p50 / max |
|---|---:|---:|
| 5.5 degrees | 6.377 / 12.338 m | 5.396 / 11.514 m |
| 10 degrees | 3.255 / 6.120 m | 2.074 / 6.322 m |
| all sweeps | 2.217 / 27.581 m | 2.074 / 29.402 m |

These are coherent wrong-view sectors, not isolated noisy pixels. The denser
257-direction allocation changes their partition but does not remove them.
Therefore it explicitly fails the connected-fan/wrong-perspective criterion.

## Panicle colour/class and camera-motion failure

Across truth panicle hits:

| allocation | panicle vs vegetative retained | exact semantic family | exact RGB343 class |
|---|---:|---:|---:|
| phase-rich | 59.82% | 18.26% | 18.10% |
| angle-rich | 59.34% | 18.75% | 18.53% |

At 25 degrees, broad panicle retention collapses to `2.65%` and `3.09%`, with
only `1.03%` exact family/class retention for both allocations. The sampled
record is therefore not merely a slightly displaced purple head; it commonly
selects vegetative geometry or a miss in the same place.

Across adjacent 1--4.5 mm camera translations, truth retained its semantic
family in 1,871 comparable pixel pairs. The prediction changed semantic family
without that truth change in `734` phase-rich pairs (`39.23%`) and `678`
angle-rich pairs (`36.24%`). This categorical instability is a direct shimmer
and distance-disappearance risk.

The QA sheets visually agree with the metrics: large red same-bin sectors and
magenta hit/miss regions replace the fine authored panicle/foliage pattern.

## Camera-inside, exact horizontal, and multiple cover types

This field stores only the exterior first event from the fixed top plane. It
does **not** store pointed-line origin phase or an ordered successor field.
Camera-inside rendering therefore remains unsolved. An exactly horizontal line
is parallel to both Y reference planes and has no finite top-entry address; it
belongs to the separate side-entry/pointed-line problem. The 2.5-degree row
also shows that clamping beyond a finite slope domain is catastrophic
(`14.061/16.969 m` p95 for the two allocations).

The 10-bit class can name many material appearances, but it does not encode
species/population/root eligibility. Valid multi-species and moss use still
requires one offline-unioned marked community truth. Querying one live field
per species or blending unrelated winners remains invalid. Since the
single-species event field already fails, unioning more overlapping cover can
only add disocclusions.

## Provenance and publication boundary

The prior-art ancestry is recorded centrally:

- Sannikov supplies the copy-count-independent repeating precomputed-field
  objective and evidence that later categorical/PCF-like corrections existed;
- View-Dependent Displacement Mapping supplies fixed-cost direct
  view-conditioning and treats silhouettes as a discontinuous signal;
- Generalized Displacement Maps supplies the origin-aware five-dimensional
  query semantics and periodic-neighbour bake.

The two-plane slope lattice, concrete cap-filling allocations, packed record,
line-datum audit, connected angular-bin fan metric, camera sweeps, and negative
result here are **LAAS derivations and measurements**, not claims made by those
sources. Immutable source, tool, recipe, metric, and image hashes above preserve
that distinction for a future paper.

## Park/resume decision

Reject both 32-bit one-record lattices. Do not implement either in the shader.
They prove that correct fixed line coordinates plus roughly 200--250 discrete
directions are still not a coherent visibility representation for this thin,
occluding Calamagrostis population.

Preserve the exact 32-bit layout, slope allocations, line-invariance proof,
camera truth, fan metric, and QA images as reusable lower-bound evidence.
Resume only if a qualitatively new fixed-cost event-sharing representation
eliminates the measured connected sectors while proving the camera-inside and
horizontal contract. Failure does not authorize more runtime reads, candidate
lists, loops, marching, per-species queries, extra passes, or a larger memory
cap.
