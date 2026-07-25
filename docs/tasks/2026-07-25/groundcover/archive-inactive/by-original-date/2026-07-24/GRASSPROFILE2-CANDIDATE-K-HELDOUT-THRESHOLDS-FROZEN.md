# Candidate K held-out angular gate — frozen protocol

Date: 2026-07-24  
Status: frozen before the exact-BVH run; offline only

## Claim under test

Candidate K stores only the unresolved foreground measure
`(premultiplied linear RGB, coverage)`.  Fractional pixels retain the real
background depth and id.  This gate therefore contains no representative
position, depth, normal, owner, or surface-plane metric.

The angular finite element has sixteen periodic azimuth sectors, eight radial
bands, and one duplicated pole.  The preregistered non-pole node elevations
are

```text
0.25, 2, 5, 10, 18, 30, 55, 75 degrees; pole = 90 degrees.
```

This deliberately spends most angular resolution below 30 degrees, where the
standing and grazing failures occur.  The stored `0.25` degree node is the
finite operational horizon-limit sample; an exactly horizontal exterior ray
above the cover box does not query the field.  Calling a literal zero-elevation
top-plane MISS the limit would be mathematically wrong for infinitely repeated
open cover.

## Coordinates and ablation

For a surface point `X`, descending ray direction `d`, and reference height
`h_ref`, the periodic field address is

```text
r = X.xz - ((X.y - h_ref) / d.y) * d.xz.
```

Equivalently, `r` is the oriented line's intersection with `y=h_ref`.  The
primary construction uses the cook-fitted botanical middle plane
`h_ref=0.49255 m`.  The old top plane `h_ref=topH` is an ablation, not a
fallback.  Both use the same topology, truth raycaster, phase resolution,
filters, directions, packing, and score code.

## Truth and finite-element protocol

- Source: the current accepted Calamagrostis GCRP/v4, with its SHA-256 recorded
  by the run.
- Exact truth: nearest forward hit in the infinitely periodic triangle soup,
  using the existing periodic BVH oracle and barycentric authored colour.
- Phase grid: full `128 x 128` texel-centre grid on the chosen reference plane.
- Spatial measures: toroidal square positive kernels with radii `sigma=4` and
  `sigma=16` on the canonical `256 x 256` source page, applied to binary
  coverage and hit RGB times coverage.  A lower-resolution diagnostic must
  preserve physical width (`r_R=sigma*R/256`); therefore the binding 128 page
  uses radii `2` and `8`, and the 64-page screen uses `1` and `4`.
- Horizon-limit premise: `0.05, 0.1, 0.2, 0.25, 0.5` degrees are scored
  separately.  The `0.25` node may govern shallower directions only if this
  sequence passes; periodic repetition alone does not prove a unique
  directional limit.
- Other held-out elevation interiors: `1, 3.5, 7.5, 14, 24, 45, 65, 82, 89`
  degrees.
- Held-out azimuths: sector edges and interiors in four separated sectors, so
  the test does not assume rotational symmetry of the authored community.
- Stored-node/corner directions are evaluated separately to isolate packing
  error.
- Unlimited-precision FE is scored first.  RGBA4444 is then applied once to
  every shared global vertex before the identical FE interpolation.

Every direction is binding; a pooled percentile cannot turn a failed direction
green.  Connected regions use four-neighbour toroidal phase connectivity and
are divided by the full phase page.

## Frozen limits

For both `sigma=4` and `sigma=16`, every held-out direction must meet:

```text
coverage absolute error:        p95 <= 0.08, p99 <= 0.20
premul RGB max-channel error:    p95 <= 0.06, p99 <= 0.15
largest connected joint p99 exceedance region: < 1% of the phase page
```

The joint exceedance mask is `coverage_error > 0.20 OR rgb_error > 0.15`.
The report must expose, separately:

1. unlimited-precision FE error (topology/interpolation loss),
2. RGBA4444 total error,
3. RGBA4444 minus unlimited-precision error (packing increment),
4. middle-plane versus top-plane coordinate results.

If unlimited-precision FE is red, quantisation cannot rescue the topology and
Candidate K parks.  If unlimited precision is green but RGBA4444 is red, only
the packing is rejected.  No runtime or shader edit is authorised by this
gate.

## Invalidated pre-run diagnostics

The first 4- and 16-square smoke invocations, and an interrupted 128-square
invocation, mistakenly treated `4/16` as radii at the test resolution.  They
changed the physical footprint (at 128, doubled it) and are invalid.  They were
discarded before any Candidate-K verdict.  The corrected tool records both the
canonical source radii and effective test-page radii in its recipe.
