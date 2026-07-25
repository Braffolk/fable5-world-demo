# Calamagrostis direct direction-lattice feasibility

**Date:** 2026-07-22  
**Status:** measured offline lower bound; no shader or runtime change  
**Constraint retained:** one nearest complete record at runtime; no march, loop, or candidate enumeration

## Outcome

At a fixed `256 x 256` phase plane, there is **no direction-node count that makes this direct nearest categorical 4D representation acceptable** for the current Calamagrostis mesh. Direction density eventually ceases to be the dominant error, but nearest phase snapping then fixes the asymptote at only `95.2474%` hit/miss agreement, `2.324 m` aggregate p95 world-position error, and `2.754 m` grazing p95. That limit was measured with the **exact live direction**, so adding direction nodes cannot cross it.

The direction-only lower bound is also much denser than a screen-angle argument suggests. With exact phase:

- a `0.0048828125 deg` adaptive spherical spacing reaches `99.5117%` aggregate hit agreement, but grazing p95 is still `0.535 m` and grazing p99 is `4.096 m`;
- a `0.00244140625 deg` spacing reaches `99.7396%` aggregate agreement and `6.18 mm` grazing p95, but grazing p99 is still `2.532 m` and exact owner agreement is only `75.31%`;
- that latter lattice has `3,400,246,224` direction nodes and costs `1,782,708,292,288,512` raw bytes (`1.583 PiB`) before gutters, mips, container overhead, or any additional payload.

Thus `0.00244140625 deg` / `3.40 billion` directions is only a **lenient direction-only p95 knee**, not an artifact-free result. The fixed-`256` direct 4D lattice has no finite direction-density answer because its measured phase-only limit already fails.

This conclusion is specific to **nearest categorical sampling of a direct phase/direction lattice**. It is not a claim that artifacts are inherent to Sannikov's method, the current precomputed-ray method, or O(1) rendering generally.

## Bound source and evaluator

Profile:

- file: `src/assets/groundcover/calamagrostis-canescens.gcrp`
- container: `GCRP/v4`
- bytes: `118,663,280`
- SHA-256: `2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c`
- decoded actual mesh: `2,171,134` triangles
- tile: `0.51999998 m x 0.51999998 m`
- top-to-minimum vertical horizon: `1.175222625 m`
- `256`-phase texel width: `2.03125 mm`

Offline evaluator:

- `tools/groundcover-bake/DirectionLatticeEvaluator.ts`, SHA-256 `989d64d77dc5dc22218bca4b0ff28cd8690c1941ad592d38430c6d8d293a5a04`
- `tools/groundcover-bake/run-direction-lattice-evaluator.ts`, SHA-256 `eaf08e232eea31ca5c5f69431be758333bd1821ab6b4a70a775dd1a7b2767ebb`
- analytic truth: f64 Moller-Trumbore first hit on the decoded-u16 actual mesh using the existing `TriangleBvh`, across every periodic copy intersected before the mesh-bottom horizon
- grazing acceleration: exact dominant-axis copy-interval traversal, offline only
- reference cross-check: `7/7` agreements with the established rectangle enumerator, including `1.1`, `1.5`, `2.25`, and `3.25` degree rays; the main sweeps also recorded `16/16`

The evaluated one-record reconstruction stores canonical vertical drop `h_i=-d_i.y t_i`, preserves hit/miss as categorical, and evaluates the live ray at `t=h_i/(-d.y)`. The full result snaps both phase coordinates to their `256`-plane texel centres. The oracle-phase result snaps direction only. The phase-only limit uses exact live direction and snaps phase only.

No runtime shader path, loop, march, or candidate set was added or evaluated.

## Held-out domain

The common set contains `3,072` analytic live rays:

- `24` deterministic low-discrepancy phase points, none chosen from phase-texel centres;
- `8` held-out azimuths per elevation;
- elevations `1.1, 1.5, 2.25, 3.25, 4.5, 6, 8, 10.5, 13.5, 15, 20, 27.5, 37.5, 52.5, 70, 85` degrees;
- `1,920/3,072` rays are in the `0-15` degree grazing band.

The 24 held phase points snap by p50 `0.801 mm`, p95 `1.160 mm`, and maximum `1.286 mm`. Exact `0 deg` is excluded: a ray with zero vertical speed never reaches the finite mesh-bottom horizon, so its horizontal periodic reach is unbounded. The measured grazing domain begins at `1.1 deg` rather than hiding that singularity in an arbitrary finite cutoff.

World-position quantiles include only true-positive pairs. Hit/miss agreement separately counts false positives and false negatives. Owner agreement requires the same source triangle and periodic tile copy.

## Adaptive spherical direction curve

Adaptive spherical lattices use uniform elevation steps and reduce azimuth count toward vertical in proportion to `cos(elevation)`. Raw bytes are exactly

`direction nodes * 256 * 256 * 8`.

There is no gutter, compression, mip, container, or alignment charge in the table.

| nominal step | directions | raw packed bytes | conservative cell at 155 m | exact-phase hit | exact-phase p50 / p95 / p99 | exact-phase grazing hit | grazing p50 / p95 / p99 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `5 deg` | 858 | 449,839,104 (`0.419 GiB`) | 27.12 m | 81.93% | 1.353 / 7.649 / 15.067 m | 92.45% | 1.640 / 8.178 / 15.624 m |
| `1.25 deg` | 13,156 | 6,897,532,928 (`6.424 GiB`) | 6.764 m | 84.99% | 0.594 / 5.776 / 9.887 m | 92.19% | 0.789 / 6.174 / 10.340 m |
| `0.3125 deg` | 208,254 | 109,185,073,152 (`101.69 GiB`) | 1.691 m | 88.74% | 0.108 / 4.435 / 8.277 m | 94.01% | 0.271 / 4.834 / 8.809 m |
| `0.078125 deg` | 3,323,327 | 1,742,380,466,176 (`1.584 TiB`) | 0.423 m | 93.95% | 9.67 mm / 3.395 / 6.595 m | 95.47% | 17.78 mm / 3.865 / 6.852 m |
| `0.01953125 deg` | 53,138,910 | 27,860,092,846,080 (`25.34 TiB`) | 0.106 m | 97.75% | 1.54 mm / 1.684 / 5.075 m | 97.76% | 2.90 mm / 2.131 / 5.669 m |
| `0.009765625 deg` | 212,532,592 | 111,428,287,594,496 (`101.34 TiB`) | 52.84 mm | 98.50% | 0.65 mm / 0.849 / 4.148 m | 98.65% | 1.34 mm / 1.179 / 4.677 m |
| `0.0048828125 deg` | 850,084,509 | 445,689,107,054,592 (`405.35 TiB`) | 26.42 mm | 99.51% | 0.28 mm / 41.98 mm / 3.335 m | 99.53% | 0.60 mm / 0.535 / 4.096 m |
| `0.00244140625 deg` | 3,400,246,224 | 1,782,708,292,288,512 (`1.583 PiB`) | 13.21 mm | 99.74% | 0.13 mm / 4.13 mm / 2.138 m | 99.79% | 0.25 mm / 6.18 mm / 2.532 m |

The p50 falls smoothly, but that is not the artifact criterion. Event-boundary crossings keep p95 and p99 large until extraordinarily fine cells. At the densest sample, only `79.39%` of aggregate true positives and `75.31%` of grazing true positives retain the exact first owner, explaining the remaining metre-scale p99 tail.

Uniform-elevation/constant-azimuth lattices were also measured. They spend about 1.6 times as many nodes as the adaptive spherical family at fine spacing and do not remove the tail sooner. For example, `0.00244140625 deg` costs `5,375,508,481` directions versus `3,400,246,224`; its exact-phase aggregate hit agreement is `99.7070%` versus `99.7396%`.

## Fixed 256-phase limit

The exact-direction phase-only limit is the relevant asymptote for every direction lattice in this representation:

| band | rays | hit agreement | precision / recall | exact owner among TP | position p50 / p95 / p99 / max |
|---|---:|---:|---:|---:|---:|
| all | 3,072 | **95.2474%** | 96.997% / 96.748% | 26.23% | 1.03 mm / **2.324 m** / 5.702 m / 15.317 m |
| grazing `1.1-15 deg` | 1,920 | **97.2917%** | 98.587% / 98.587% | 22.38% | 1.07 mm / **2.754 m** / 5.904 m / 15.317 m |
| middle `20-37.5 deg` | 576 | 89.4097% | 92.582% / 89.655% | 36.86% | 0.86 mm / 0.652 m / 2.168 m / 3.210 m |
| steep `52.5-85 deg` | 576 | 94.2708% | 87.662% / 90.604% | 53.33% | 0.95 mm / 55.85 mm / 0.753 m / 0.852 m |

The dense full-4D measurements converge around this floor. At `0.00244140625 deg`, the adaptive lattice with phase snapping gives `95.1497%` aggregate hit agreement and `2.569 m` p95, versus the exact-direction limit's `95.2474%` and `2.324 m`. Direction nodes have ceased to be the governing variable.

The small p50 and huge tail are not contradictory. Most sub-texel phase shifts retain the same event and move it slightly; a minority cross a thin first-event boundary and select a different blade or a much deeper periodic event. A nearest categorical sample has no continuity mechanism across that boundary.

## Nonuniform slope/elevation result

The slope family spaces rings uniformly in `cot(elevation)`, deliberately spending far more elevation nodes near grazing while adapting azimuth count toward vertical.

| slope step at vertical | directions | raw bytes | exact-phase all hit / p95 | exact-phase grazing hit / p95 |
|---:|---:|---:|---:|---:|
| `5 deg` | 46,460 | 24,358,420,480 (`22.69 GiB`) | 82.85% / 6.424 m | 92.66% / 6.935 m |
| `2.5 deg` | 186,013 | 97,524,383,744 (`90.83 GiB`) | 85.03% / 5.769 m | 93.28% / 6.277 m |
| `1.25 deg` | 743,915 | 390,025,707,520 (`363.24 GiB`) | 86.65% / 5.050 m | 94.27% / 5.427 m |
| `0.625 deg` | 2,974,767 | 1,559,634,640,896 (`1.419 TiB`) | 88.09% / 4.278 m | 93.85% / 4.683 m |
| `0.3125 deg` | 11,896,167 | 6,237,017,604,096 (`5.673 TiB`) | 90.43% / 3.579 m | 95.10% / 4.022 m |

At comparable node counts, slope spacing is consistently worse than adaptive spherical spacing. Examples:

- `0.625 deg` slope: 2.97 million nodes, 93.85% grazing agreement, 4.683 m grazing p95;
- `0.078125 deg` adaptive spherical: 3.32 million nodes, 95.47%, 3.865 m;
- `0.3125 deg` slope: 11.90 million nodes, 95.10%, 4.022 m;
- `0.0390625 deg` adaptive spherical: 13.29 million nodes, 96.93%, 3.051 m.

Uniform-slope rings do reduce grazing median error early, but they over-invest in elevation while azimuthal and event-boundary errors remain. They do not lower the node or byte requirement.

## Screen-space/angular bound at 155 m

For the live `55 deg` vertical FOV, a `1080` CSS-pixel-high viewport at DPR 2 has centre-pixel angular width

`2 atan(tan(55 deg / 2) / 2160) = 0.02761694 deg`,

or a `74.71 mm` footprint at `155 m`.

Covering the `1-90 deg` direction hemisphere with half-pixel spherical caps has an area-only lower bound of `33,832,787` direction nodes. Even this unrealistically efficient cover costs `17,738,124,230,656` raw bytes (`16.13 TiB`) at `256 x 256 x 8` bytes. A realizable lattice needs more.

That screen-angle bound is insufficient for first-event stability:

- `212.5 million` adaptive nodes give a conservative `0.707`-pixel cell at 155 m, yet exact-phase grazing p95 is `1.179 m`;
- `850.1 million` nodes give `0.354` pixel, yet grazing p95 is `0.535 m`;
- `3.40 billion` nodes give `0.177` pixel before grazing p95 reaches `6.18 mm`, while p99 remains `2.532 m`.

Screen-cell width bounds ray-direction displacement; they do not bound discontinuous first-event identity. A thin blade boundary can replace the first event with a much deeper event under a subpixel angular perturbation.

## Decision

For the actual Calamagrostis mesh, a direct nearest complete-event `phaseX x phaseZ x direction` lattice with `256 x 256` phase resolution is rejected as a generic production representation:

1. the infinite-direction-density phase limit is already objectionable;
2. the direction-only p95 knee costs roughly `3.40 billion` directions and `1.583 PiB` raw;
3. p99 and owner stability remain objectionable even at that density;
4. nonuniform slope spacing does not change the conclusion;
5. the one-pixel angular storage lower bound understates the event-stability requirement by orders of magnitude.

This is a representation feasibility result, not a request to sacrifice O(1), add a runtime loop, or accept the current artifacts.

## Reproduction

The complete sweep can be reproduced with the current evaluator in one command:

```bash
npx tsx tools/groundcover-bake/run-direction-lattice-evaluator.ts \
  --phase-count 24 \
  --azimuths-per-elevation 8 \
  --elevations 1.1,1.5,2.25,3.25,4.5,6,8,10.5,13.5,15,20,27.5,37.5,52.5,70,85 \
  --steps 5,2.5,1.25,0.625,0.3125,0.15625,0.078125,0.0390625,0.01953125,0.009765625,0.0048828125,0.00244140625 \
  --slope-steps 5,2.5,1.25,0.625,0.3125 \
  --phase-resolution 256 \
  --bytes-per-record 8 \
  --radius 155 \
  --fov 55 \
  --css-height 1080 \
  --dpr 2 \
  --source src/assets/groundcover/calamagrostis-canescens.gcrp \
  --output /tmp/calamagrostis-direction-lattice.json \
  --quiet
```

Executed measurements were split into coarse, subdegree, microdegree, and exact-direction phase-limit runs over the identical held-out rays. The runner records the complete `process.argv`, source byte count, and source SHA-256 in every JSON output.
