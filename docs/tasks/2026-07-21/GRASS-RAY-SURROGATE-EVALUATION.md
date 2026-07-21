# Calamagrostis fixed-read ray-surrogate evaluation

Date: 2026-07-21. This is an offline mathematical evaluation only. It changes
no runtime or shader code and proposes no runtime march, loop, BVH, or owner
enumeration.

## Evaluated carrier and domain

- Source: `src/assets/groundcover/calamagrostis-canescens.gcrp`, GCRP/v4,
  118,663,280 bytes, SHA-256
  `2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c`.
- Truth: nearest analytic hit of the decoded-u16 2,171,134-triangle mesh across
  all periodic copies, using the existing typed BVH.
- Dense comparison: 16x16 noncanonical phases, 16 half-bin azimuths, elevations
  `5, 7.5, 10, 12.5, 15, 20, 25, 35, 45, 55, 65, 75, 82.5` degrees: 53,248
  live rays.
- The shipped `15/35/55/75` degree rows remain unchanged. The evaluator also
  made a sparse hypothetical exact-mesh 5-degree row and vertical singleton on
  every touched point of the same 256-square phase lattice: 17,408 generated
  canonical records. Thus the decisive 5-to-15-degree result evaluates the
  intended first domain correction, not the old 15-degree clamp.
- Dense trace time: 90.55 seconds. Output:
  `/tmp/calamagrostis-ray-surrogate-16x16-v2.json`.

The canonical-coordinate sanity gate used exact texel centres and canonical
directions. Across 256 rays, all methods had 100% hit/miss agreement and about
18 micrometres median world-position error. Only two plane points fell outside
their decoded triangles at quantized silhouette edges. This rules out an atlas
address, gutter, direction-order, or truth-BVH mismatch as the source of the
between-sample errors below.

## Fixed-read methods

| key | fixed shader work | evaluated reconstruction |
|---|---:|---|
| A projected relift | 4 filtered reads / 16 effective texels | current coverage-unmixed inverse projected path, relifted by live horizontal speed |
| B_tau nearest tau | 1 nearest read | nearest complete canonical 3D distance `tau_i` applied unchanged to the live ray; retained as a negative control |
| B_h nearest vertical drop | 1 nearest read | nearest complete `h_i=(-d_i.y)tau_i`, then `tau=h_i/(-d_live.y)` |
| C face plane | 1 nearest read | plane through the stored hit with the winning source face normal; two dot products and one division |
| D reciprocal slope triangle | 3 filtered reads / 12 effective texels | coverage-unmixed reciprocal vertical drop over one Cartesian-slope triangle, then one reciprocal |

C assumes the geometric face normal is packed with the depth record. The
evaluator derives it from the v4 owner table offline; a runtime geometry-table
read is not included in its one-read budget. D assumes reciprocal vertical drop
is packed directly; deriving it from ordinary depth adds three reciprocals.

## Dense aggregate result

Errors use true-positive rays only. `h` is absolute vertical-drop error; `pos`
is Euclidean world-position error, equal to absolute ray-distance error here.
Agreement, precision, and recall separately retain false hits and missed hits.

| method | agreement | hit precision | hit recall | h p50 / p95 / p99 (m) | pos p50 / p95 / p99 (m) |
|---|---:|---:|---:|---:|---:|
| A projected relift | 65.68% | 64.44% | 96.25% | 0.247 / 0.786 / 0.979 | 0.924 / 4.137 / 7.060 |
| B_tau nearest tau | 69.18% | 75.37% | 72.67% | 0.252 / 0.884 / 1.113 | 1.195 / 6.052 / 9.061 |
| B_h nearest vertical drop | 69.18% | 75.37% | 72.67% | **0.232 / 0.849 / 0.967** | **1.142 / 5.545 / 8.279** |
| C face plane | 67.11% | 76.11% | 66.31% | 0.243 / 0.836 / 0.963 | 1.176 / 5.510 / 8.370 |
| D reciprocal slope triangle | 65.95% | 64.70% | 95.87% | 0.233 / 0.781 / 0.910 | 0.925 / 4.312 / 7.149 |

B_h does fix the specific metric error in B_tau: its vertical-drop p50/p95/p99
improve by `20/35/146 mm`, and its position p50/p95/p99 improve by
`53/507/782 mm`. It does not fix categorical first-owner displacement. The
one-read method still misses 8,781 truth hits, fabricates 7,630 hits, and has a
1.142 m median position error.

C encountered zero denominator poles at the declared `1e-4` threshold, but
2,984 intersections landed outside the vertical band. More decisively, all
27,999 accepted plane intersections were outside the finite winning source
triangle. The plane algebra is valid; its assumed owner correspondence is not.
D had zero reciprocal poles and zero out-of-band values after the fresh 5-degree
and vertical rows closed the slope domain. It only marginally improves A's
vertical error and slightly worsens A's position tail and hit recall.

## Low-elevation result

These are the decisive ground-oblique rows. Each cell is `vertical h p50/p95 ;
world position p50/p95`, in metres. The separate height numbers prevent grazing
horizontal displacement from hiding whether vertical collapse was corrected.

| elevation | A projected relift | B_h nearest vertical drop | C face plane | D reciprocal triangle |
|---:|---:|---:|---:|---:|
| 5 | 0.131/0.668 ; 1.501/7.659 | 0.138/0.741 ; 1.581/8.507 | 0.141/0.742 ; 1.617/8.510 | 0.134/0.669 ; 1.543/7.678 |
| 7.5 | 0.182/0.731 ; 1.395/5.602 | 0.176/0.815 ; 1.348/6.245 | 0.223/0.814 ; 1.711/6.232 | 0.181/0.755 ; 1.386/5.780 |
| 10 | 0.233/0.746 ; 1.341/4.294 | 0.296/0.866 ; 1.702/4.987 | 0.264/0.842 ; 1.523/4.847 | 0.235/0.783 ; 1.354/4.508 |
| 12.5 | 0.279/0.760 ; 1.288/3.511 | 0.328/0.853 ; 1.515/3.939 | 0.301/0.844 ; 1.389/3.900 | 0.277/0.776 ; 1.278/3.584 |
| 15 | 0.307/0.789 ; 1.186/3.048 | 0.333/0.861 ; 1.285/3.327 | 0.330/0.857 ; 1.276/3.313 | 0.309/0.790 ; 1.194/3.053 |

At 5 degrees almost every ray intersects the dense periodic stand, so agreement
alone is misleading: A and D exceed 99% agreement while their median hit
position is still wrong by about 1.5 m and p95 is wrong by about 7.7 m.

## Phase-versus-direction decomposition

Two 8,192-ray controls isolated one coordinate at a time at 5 and 15 degrees:

| perturbation | A pos p50 at 5 / 15 deg | A h p50 at 5 / 15 deg |
|---|---:|---:|
| noncanonical phase, canonical direction | 0.108 / 0.044 m | 0.009 / 0.011 m |
| canonical phase, half-bin direction | 1.597 / 1.208 m | 0.139 / 0.313 m |

For B_h, phase-only median position error is about `1.8/1.5 mm`; direction-only
is `1.618/1.345 m`. Spatial quantization is not the median failure. The shipped
16-azimuth lattice's 11.25-degree worst-case directional offset dominates the
low-oblique result, even after adding the exact 5-degree row.

Phase-only tails remain large at silhouettes and owner changes (A position p95
3.60 m at 5 degrees), so denser angles alone are not a proof of continuum
visibility. They are, however, the immediate discretization variable that a
fixed-read visual experiment must change; swapping among A/B_h/C/D on the same
16-azimuth support cannot meet the geometry-quality target.

## First-pass selection

None of the tested one-to-four-read surrogates is acceptable for the requested
artifact-free geometry result on the current 16-azimuth support.

- Keep A as the least-bad reference: strongest hit recall and the best or tied
  low-elevation errors. D costs three reads and does not materially beat it.
- Keep B_h as the smallest honest one-read diagnostic. It is mathematically the
  correct vertical-drop chart and improves B_tau, but its owner/direction error
  is far above the visual tolerance.
- Reject C. Its infinite-plane result almost never remains on the baked finite
  face between sampled rays.
- Do not choose stochastic categorical direction selection as a repair. It
  samples the same displaced one-record support as B_h, preserves rather than
  removes its owner error, and converts deterministic error into spatial or
  temporal noise.
- The final fixed-read experiment below increases canonical azimuth density
  while retaining the exact 5-degree row and vertical singleton, and tests one
  exact-identity phase reprojection. Acceptance still requires exact-mesh
  evaluation because owner/silhouette tails do not vanish merely from denser
  sampling.

Reproduce the dense report with:

```sh
node --import tsx tools/groundcover-bake/run-ray-surrogate-evaluator.ts \
  --phase-grid 16 --azimuth-count 16 \
  --output /tmp/calamagrostis-ray-surrogate-16x16-v2.json
```

## Second and final sampled-field attempt

This is the bounded follow-up to the first pass. It adds no runtime/shader
implementation. All proposals still have a fixed number of categorical reads;
there is no live march, loop, triangle access, candidate list, or owner search.

The final run used 8x8 phases, 16 half-bin azimuths, and the same 13 elevations:
13,312 primary rays. It also evaluated statistically useful independent N=32
and N=64 half-bin direction sets. Primary tracing took 30.97 seconds. The
denser probes took 8.11 and 15.39 seconds respectively. Output:
`/tmp/calamagrostis-ray-surrogate-final.json`.

### E1 and E2: two-read canonical phase reprojection

For one nearest canonical slope `s_i`, E1 performs exactly two nearest,
categorical reads:

1. Read `h0=D_i(q)` at the live phase.
2. Apply the exact same-hit identity `q_i=q+(s-s_i)h0`, read
   `h1=D_i(q_i)`, and reconstruct the live distance as `h1/(-d_live.y)`.

E2 uses those same two samples and no third read. It treats the local map as
affine and solves its fixed point in closed form:
`h*=h0^2/(2h0-h1)`. Neither method filters or blends owners.

| method | agreement | hit precision | hit recall | h p50 / p95 / p99 (m) | pos p50 / p95 / p99 (m) |
|---|---:|---:|---:|---:|---:|
| B_h one-read baseline | 68.90% | 75.76% | 71.65% | 0.220 / 0.834 / 0.962 | 1.139 / 5.313 / 8.228 |
| E1 one reprojection | 68.06% | 83.86% | 58.63% | 0.202 / 0.845 / 0.979 | 1.192 / 5.896 / 8.740 |
| E2 affine fixed point | 60.04% | 84.71% | 41.63% | 0.208 / 0.787 / 0.934 | 1.143 / 5.182 / 7.906 |

E1 improves B_h median vertical error by only 18 mm, while reducing recall by
13.0 percentage points and worsening median and p95 position error. Of 7,637
predictor-hit rays, the reprojected categorical record had a different owner
in all 7,637 cases; 1,992 became misses. E2 rejects another 1,676 rays as
out-of-band affine solutions. Its smaller true-positive tail is therefore
selection loss, not recovered geometry.

E1 remains especially poor in the decisive oblique rows. Each cell is
`agreement / recall ; vertical h p50/p95 ; world position p50/p95`.

| elevation | E1 result |
|---:|---:|
| 5 | 97.27% / 97.84% ; 0.129/0.758 m ; 1.483/8.701 m |
| 7.5 | 94.24% / 97.18% ; 0.180/0.811 m ; 1.380/6.216 m |
| 10 | 62.21% / 64.68% ; 0.323/0.889 m ; 1.861/5.118 m |
| 12.5 | 62.89% / 66.82% ; 0.260/0.845 m ; 1.200/3.902 m |
| 15 | 59.18% / 64.24% ; 0.332/0.898 m ; 1.284/3.468 m |

The algebraic identity is exact for a retained first surface. The result shows
that a one-step predictor does not retain that surface on this discontinuous
first-hit chart; it does not establish an inherent limitation of Sannikov's
O(1) method or of later artifact fixes.

### Denser azimuth rings at byte-matched storage

The independent probes use exact-mesh-generated categorical vertical-drop
records at their own worst half-bin azimuths. N=32 uses P=161; N=64 uses P=113.
These phase resolutions keep total stored bytes within 0.5% of the current
34,080,768-byte N=16/P=256/four-ring atlas while adding the 5-degree ring and
vertical singleton.

| lattice | rays | agreement | precision | recall | h p50 / p95 (m) | pos p50 / p95 (m) | sampled records | stored records / bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N=32, P=161 | 6,656 | 69.11% | 75.96% | 71.99% | 0.239 / 0.852 | 1.142 / 5.720 | 2,496 | 4,277,609 / 34,220,872 |
| N=64, P=113 | 13,312 | 69.76% | 75.74% | 73.62% | 0.189 / 0.833 | 0.993 / 5.443 | 4,832 | 4,245,225 / 33,961,800 |

N=64 lowers median position error modestly, but does not materially improve
categorical agreement or the multi-metre p95 tail. At 5 degrees its position
p50/p95 is 1.461/8.730 m. Trading phase density for azimuth density at fixed
memory therefore does not produce crisp geometry.

### Atlas accounting

All corrected-grid rows below contain five azimuth rings
(`5/15/35/55/75` degrees), one vertical singleton, one wrapped gutter texel on
each side, and eight bytes per categorical record. Counts are exact.

| azimuths | phase P | directions | interior records | stored records | stored bytes |
|---:|---:|---:|---:|---:|---:|
| 16 | 64 | 81 | 331,776 | 352,836 | 2,822,688 |
| 16 | 128 | 81 | 1,327,104 | 1,368,900 | 10,951,200 |
| 16 | 256 | 81 | 5,308,416 | 5,391,684 | 43,133,472 |
| 32 | 64 | 161 | 659,456 | 701,316 | 5,610,528 |
| 32 | 128 | 161 | 2,637,824 | 2,720,900 | 21,767,200 |
| 32 | 256 | 161 | 10,551,296 | 10,716,804 | 85,734,432 |
| 64 | 64 | 321 | 1,314,816 | 1,398,276 | 11,186,208 |
| 64 | 128 | 321 | 5,259,264 | 5,424,900 | 43,399,200 |
| 64 | 256 | 321 | 21,037,056 | 21,367,044 | 170,936,352 |

For comparison, the current four-ring N=16/P=256 layout has 64 directions,
4,194,304 interior records, 4,260,096 stored records, and 34,080,768 stored
bytes. The byte-matched corrected layouts are N=32/P=161 at 34,220,872 bytes
(`+0.41%`) and N=64/P=113 at 33,961,800 bytes (`-0.35%`).

## Final decision: park this sampled-field reconstruction track

The second real attempt does not materially fix geometry without memory
growth. E1/E2 lose owner/hit continuity and worsen useful position accuracy;
N=32/N=64 at byte-matched storage leave approximately 1 m median and 5.4--5.7 m
p95 position errors. No runtime or shader change from this track is warranted.

Do not add another interpolation, predictor iteration, local plane, or sampled
owner heuristic to this path. Resume it only if new evidence supplies a
qualitatively different fixed-read representation that preserves first-surface
continuity, or if an exact-mesh evaluation of a materially different baked
support clears both categorical and position-tail acceptance. This park is
narrow: it rejects the evaluated surrogate family, not the O(1) precomputed-ray
method or the known artifact-free target.

Reproduce the final report with:

```sh
node --import tsx tools/groundcover-bake/run-ray-surrogate-evaluator.ts \
  --phase-grid 8 --azimuth-count 16 --denser-phase-grid 4 \
  --output /tmp/calamagrostis-ray-surrogate-final.json
```
