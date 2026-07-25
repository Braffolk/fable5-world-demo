# Candidate L conforming 4D FE — slope-rate RED blocker

Date: 2026-07-24  
Status: **RED / PARKED before packing or a full 4D cook**  
Scope: algebra and reduction of existing exact first-hit pages; no new BVH cook,
runtime, or shader edits

## Decision

No tested Candidate-L slope mesh under `V_D=257/513/769` has a plausible
non-blurred 2--5 m band.  The direction rate misses the representation's own
sufficient phase-shear law by one to two orders of magnitude, while increasing
`V_D` enough to help removes the spatial and footprint-level capacity needed
by that band.  The compactified horizon fringe has unbounded slope diameter
and its required appearance convergence is already empirically RED.

Candidate L therefore stops before packing, a conforming 4D cook, or runtime
work.  This is a rate/allocation failure, not a four-bit-code failure.

## Inputs and physical scale

The reduction decoded the 65 current exact first-hit pages of the accepted
Calamagrostis GCRP/v4 and reconstructed hit height as

```text
y = topH - (-d_y) * first_hit_depth.
```

The fitted reference is `h_ref=0.49255 m`; the relevant phase-shear lever is
`R=|y-h_ref|`.  The repeating tile is `0.52 m` at `256 x 256`, so the accepted
physical filter radii are

```text
sigma 4:   r = 0.008125 m
sigma 16:  r = 0.032500 m.
```

Actual hit residuals are not small:

```text
elevation   R p95 across azimuths (min ... max)    largest R p99
15 deg      0.568 ... 0.581 m                     0.635 m
35 deg      0.533 ... 0.543 m                     0.630 m
55 deg      0.529 ... 0.535 m                     0.631 m
75 deg      0.529 ... 0.534 m                     0.635 m
90 deg      0.536 m                               0.633 m
```

For each physical scale the gate also formed toroidal footprint-local mean and
maximum residual fields.  The rate below uses the p95 local maximum, so it is
conditioned on the actual positive footprint rather than the global
`max_X |X_y-h_ref|` bound.

## Structured slope meshes

For every exact vertex budget, the gate enumerated all compatible factorizations

```text
V_D = 1 pole + A azimuth vertices * N radial rings
```

for `A in {8,16,32,64,128,256}` and three radial laws:

1. uniform elevation,
2. uniform `log(1+|s|)`,
3. uniform compact coordinate `|s|/(1+|s|)`.

For every measured elevation/azimuth, it computed radial and tangential slope
edges, their simplex diameter `Delta_s`, cell anisotropy, and

```text
rate = R_p95 * Delta_s / r.
```

The target regime is `rate <= 1`.  Values much greater than one move a measured
first-hit event by several filter radii inside one direction simplex.

Best single mesh over all measured directions:

```text
V_D   best structured mesh                 worst rate sigma4   sigma16
257   A16 x R16, compact slope                    156.42          41.50
513   A32 x R16, compact slope                    114.36          30.34
769   A32 x R24, log slope                         93.35          24.77
```

This conclusion is not an artifact of choosing one compromise mesh.  If a
different tested mesh is allowed independently at every elevation, the best
rates at 15 degrees are still

```text
V_D=257: 154.37 / 41.18
V_D=513: 112.86 / 30.10
V_D=769:  92.13 / 24.57       (sigma4 / sigma16).
```

That is the standing/low-oblique end of the 2--5 m quality band and is the
binding failure.

## Elevation and azimuth anisotropy

For the best global `V_D=513` mesh, the measured per-elevation rate including
the actual azimuthal residual spread is:

```text
elevation  cell anisotropy   sigma4 p95/max     sigma16 p95/max
15 deg          1.55          114.20 / 114.36     30.31 / 30.34
35 deg          1.16           33.34 /  33.35      9.59 /  9.60
55 deg          1.16           15.12 /  15.13      4.41 /  4.42
75 deg          1.57            7.82 /   7.83      2.23 /  2.23
90 deg          5.10            4.37 /   4.37      1.24 /  1.24
```

Azimuthal variation of `R` is small compared with the cell-rate deficit; no
fortunate plant orientation closes it.  Rebalancing radial versus tangential
edges changes which elevation is worst but never creates an all-elevation
GREEN mesh.

## Compactified horizon fringe

The structured meshes explicitly end their finite slope domain at `0.25`
degrees (`|s|=cot(0.25 deg)=229.18`).  The compactified final cell covers
`0 < elevation < 0.25 deg`, hence has infinite slope diameter.  For any
non-zero residual its sufficient rate is infinite.

This could only be harmless if the footprint appearance converged to one
phase-independent fringe value.  Candidate K's exact held-out sequence at
`0.05/0.1/0.2/0.25/0.5` degrees rejected that premise: coverage becomes nearly
opaque but premultiplied RGB remains direction/phase dependent at roughly
`0.27` p95.  Candidate L has no valid compactified fringe under the current
arbitrary-soup contract.

## `V_Q`, `V_D`, and `L` memory frontier

The exact law is

```text
B_scale = 4 * V_Q * V_D * L bytes <= 32.5 MiB.
```

Avoiding spatial blur at sigma 4 requires phase spacing no coarser than
`8.125 mm`, hence at least `V_Q=64^2` on the 0.52 m tile.  At that minimum:

```text
V_D   maximum L under cap   allocation at maximum L
257          7                    28.109 MiB
513          4                    32.063 MiB
769          2                    24.031 MiB
```

Some complete reference choices:

```text
V_D   V_Q       L      MiB      phase spacing
257   64^2      6      24.094   8.125 mm
257   72^2      6      30.494   7.222 mm
257   80^2      5      31.372   6.500 mm
513   64^2      4      32.063   8.125 mm
513   72^2      3      30.434   7.222 mm
769   64^2      2      24.031   8.125 mm
769   72^2      2      30.415   7.222 mm
```

Thus the only budget with the originally contemplated six footprint intervals
and non-blurred sigma-4 phase is `V_D=257`, whose measured slope rate is the
worst (`156.4`).  Raising direction rate to 769 leaves only two intervals and
still misses the shear condition by `93.4x` at sigma 4.

## Provenance

- Machine report and full per-slice/per-mesh table:
  `data/work/groundcover-candidate-l-slope-rate/e3e0a4175b151b89/288db08ddf91c583/report.json`
- QA and hash index:
  `data/work/groundcover-candidate-l-slope-rate/e3e0a4175b151b89/288db08ddf91c583/qa/`
- Reduction tool:
  `tools/groundcover-bake/analyze_candidate_l_slope_rate.py`
- Source SHA-256:
  `e3e0a4175b151b89a1a4bb58089aed2e809da337a04ce024240be8decdacb0be`

## Claim boundary and resume condition

`R Delta_s/r <= 1` is a sufficient sizing law, not an impossibility theorem
for every conceivable basis.  The RED conclusion is narrower and actionable:
none of Candidate L's proposed structured slope-product FE allocations is a
plausible use of the fixed budget, and the measured K convergence agrees with
that rate diagnosis.

Resume only with a representation that removes the phase shear analytically
or encodes angular visibility events adaptively without a dense `V_Q * V_D`
product.  Do not resume Candidate L by lowering `V_Q`, dropping the sigma-4
2--5 m band, clamping the horizon, raising memory, or proceeding directly to
packing.
