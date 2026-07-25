# Candidate H — fixed 16-entry Calamagrostis palette gate

Status: thresholds and scoring protocol frozen before the first palette fit or
metric observation.

## Bound source

- Accepted actual Calamagrostis GCRP/v4 bake:
  `data/work/groundcover-gpu-bake-estonian-graminoids-v4/2-calamagrostis-canescens/2ed57f59d86e8376/calamagrostis-canescens-periodic-profile.gcrp`.
- The GCRP first-hit owner atlas supplies the visible-sample population and its
  binary baked-coverage weights across all 64 baked direction slices.
- The deterministic production fixture's `primitiveRecipes` sidecar supplies
  triangle semantics. Its indexed-mesh SHA-256 must equal the source mesh hash
  recorded by the accepted bake before scoring is allowed.
- Plume means `hair-filament`, reproductive `lanceolate-surface` roles
  (`glume`, `lemma`, `anther`), and reproductive tube roles (`panicle-axis`,
  `spikelet-axis`, `anther-filament`). Stem/leaf means every remaining authored
  primitive. No post-hoc RGB threshold may define either subset.

## Frozen fit

- Decode each covered first hit to linear-sRGB using its owner triangle and
  ray/triangle barycentrics. Convert linear-sRGB to CIE Lab using the D65
  reference white and score with CIEDE2000.
- Aggregate identical decoded RGB triplets with their baked-coverage counts.
- Fit one deterministic joint 16-medoid palette over plume and stem/leaf
  samples. The two semantic subsets never receive separate palettes.
- Objective: minimum coverage-weighted sum of CIEDE2000 distances. Use
  deterministic BUILD plus exhaustive improving PAM swaps until no improving
  swap remains; deterministic alternate starts may be evaluated, with the
  lowest objective selected and lexicographic RGB as the final tie-break.

## Frozen gates

All percentiles are coverage-weighted.

- Overall Delta-E-00 p95 <= 2.5.
- Overall Delta-E-00 p99 <= 5.0.
- Violet plume semantic subset p95 <= 3.0.
- Green stem/leaf semantic subset p95 <= 3.0.
- For every baked slice, the largest toroidally connected four-neighbour
  component whose Delta-E-00 is greater than 5.0 must occupy less than 1% of
  that slice's covered texels. The reported gate value is the maximum of this
  fraction over all slices.

No threshold, subset, connectivity rule, source binding, or fitting objective
may change after the first result is observed. A RED result is reported as
RED; it is not tuned into GREEN.

## Result — GREEN

Command:

```text
NODE_OPTIONS=--max-old-space-size=4096 npx tsx tools/groundcover-bake/gate-candidate-h-palette16.ts
```

Content-addressed result:

`data/work/groundcover-candidate-h-palette16/2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c/dd4b05a99c5ba516fa2c70d4e35b260ca7c58fdd1714a1e226c04cbcf8b910f4/qa/`

- Population: 1,935,453 covered first hits; 617 occupied linear-RGB8 colours.
- Overall Delta-E-00 p95 = 0.74134 (gate 2.5).
- Overall Delta-E-00 p99 = 1.80697 (gate 5.0).
- Plume Delta-E-00 p95 = 0 (gate 3.0).
- Stem/leaf Delta-E-00 p95 = 0.78248 (gate 3.0).
- Maximum per-slice connected bad-support fraction = 0.0002709 (gate 0.01).
- Deterministic palette16 weighted mean Delta-E-00 = 0.24121.

The implementation verified every triangle index against both the accepted
source transport and the GCRP embedded triangle table before applying the
procedural semantic ranges. The numbered mosaic was inspected after the run;
the quantized side preserves the visible green structural and violet/plume
separation without a coherent error region.

## Required artifacts

Write under a content-addressed
`data/work/groundcover-candidate-h-palette16/<source>/<recipe>/qa/` directory:

1. `001-original-vs-quantized-palette-mosaic.png`
2. `002-deltae00-error-diagnostics.png`
3. `003-connected-error-support.png`
4. `index.json`, binding source/recipe hashes, image hashes and dimensions,
   palette, metrics, pass/fail results, and interpretations.
