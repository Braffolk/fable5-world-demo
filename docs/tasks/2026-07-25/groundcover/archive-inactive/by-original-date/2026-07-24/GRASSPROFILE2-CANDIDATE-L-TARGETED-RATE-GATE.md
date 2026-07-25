# Candidate L targeted 4D rate gate

Date: 2026-07-24  
Status: **RED / do not expand to a full cook or runtime implementation**

## Question and frozen experiment

This gate tested the strongest small experiment that could rescue Candidate L
from its prior slope-rate warning.  It used the real Calamagrostis GCRP and
exact periodic BVH first-hit truth, physically fixed positive-measure filter
supports, unlimited-precision values, and fresh held-out interiors and shared
boundaries in the two previously measured worst direction cells.

The comparison was intentionally favourable to the old Candidate K baseline:

1. `K-same-q`: four angular corners, with exact live phase access;
2. `L-simplex5`: the proposed conforming five-vertex 4-simplex over spatial
   triangle x slope triangle;
3. `tensor9`: all nine vertices of the same spatial-triangle x
   direction-triangle product, used only as an approximation ceiling.

The thresholds, sample sites, physical supports, predictors, and interpretation
rules were frozen before any interpolation result in
`GRASSPROFILE2-CANDIDATE-L-TARGETED-RATE-THRESHOLDS-FROZEN.md`.

## Result

Candidate L is decisively RED.

```text
predictor       green cases   worst normalized p95   coverage p95   RGB p95   connected
K same-q          5 / 40             7.0982              .5386        .4259      .5341
L simplex5        0 / 40             7.2126              .5139        .4328      .5386
tensor9           0 / 40             7.1790              .5112        .4307      .5787
```

The frozen comparison ratios were:

```text
L / K worst normalized p95:       1.0161   (required <= 0.75)
L / tensor9 worst normalized p95: 1.0047   (required <= 1.25)
simplex boundary residual:        2.95e-9  (required <= 1e-5)
```

The conforming assembly itself is correct: the numerical boundary residual is
over three thousand times below its limit.  The failure is fidelity, not a
simplex crack, quantization error, or packing error.

At the standing cell, every predictor fails strongly at both physical scales.
For the smaller support, Candidate L reaches coverage p95 `.5139`, RGB p95
`.4328`, and a `.5386` connected wrong region.  The nine-vertex tensor ceiling
is essentially identical.  At the grazing cell, coverage is already saturated
and therefore has zero error in these cases, but colour remains far outside the
limit: Candidate L RGB p95 reaches `.3340` at the smaller support and `.2755`
at the larger support.

## Mathematical diagnosis

The five-vertex compression is **not** the blocker.  The full nine-vertex
tensor interpolant fails almost exactly like the five-vertex simplex.  Nor is
Candidate K's main failure explained by its omission of the spatial vertices:
adding the conforming spatial triangle gives no material improvement over the
strong same-phase K baseline.

The blocker is the selected local rate/reference chart itself.  Across these
direction cells, the exact filtered field contains visibility and colour
variation that a piecewise-affine interpolant on this spatial/directional
lattice does not resolve.  Conformity removes cell-boundary jumps, but it
cannot reconstruct events absent from the vertex samples.  This empirically
confirms the earlier phase-shear sizing result rather than rescuing it.

Therefore a denser Candidate-L product mesh would be required.  The existing
allocation analysis already shows that the required slope density is
incompatible with the fixed memory budget while retaining the necessary
spatial rate and footprint levels.  Candidate L must not proceed by lowering
spatial quality, clamping shallow directions, adding reads, or raising memory.

## Decision and resume condition

Do not run the giant Candidate-L cook.  Do not pack it.  Do not edit the
runtime or shader for it.  No additional local Candidate-L variant is
authorised by this result.

Resume only if a new representation removes the phase shear analytically or
encodes the relevant visibility events without a dense `V_Q * V_D` product.
That representation must be derived as mathematics first and receive its own
frozen rate/fidelity gate.

## Provenance

- machine report:
  `data/work/groundcover-candidate-l-targeted-fe/e3e0a4175b151b89/1931b9fb2d09903b/report.json`
- QA directory:
  `data/work/groundcover-candidate-l-targeted-fe/e3e0a4175b151b89/1931b9fb2d09903b/qa/`
- gate:
  `tools/groundcover-bake/gate-candidate-l-targeted-4d-fe.ts`
- source SHA-256:
  `e3e0a4175b151b89a1a4bb58089aed2e809da337a04ce024240be8decdacb0be`
- gate SHA-256:
  `d83cd589f4e81ebc15efba20a10fe2a2c0b0b84619c1a5bf98f42eba6546d3a4`
- frozen-threshold SHA-256:
  `9a286d7a1ba7ad01ed36351156025e8d28af8472f3b19f01d13cf505a584c352`
- report SHA-256:
  `bb77c86b5c150e4087be4a1681f8187d103dff91c3bfeeef363f7ad9a5613be4`

