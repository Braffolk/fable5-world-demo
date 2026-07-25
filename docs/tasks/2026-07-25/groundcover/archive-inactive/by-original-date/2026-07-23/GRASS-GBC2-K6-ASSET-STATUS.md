# GBC2-K6 production asset status

**Date:** 2026-07-23  
**Track:** accepted Calamagrostis GCRP/v4 to runtime GBC2/v1  
**Runtime/shader changes:** none in this track

## Current inspectable outcome

- Stable asset: `src/assets/groundcover/calamagrostis-canescens.gbc2`
- Content-addressed report:
  `data/work/groundcover-gbc2-k6-assets/2ed57f59d86e8376/97e5f44b22241920/report.json`
- Asset SHA-256:
  `b02e9e0cea4e3b225a92841a7a8011496cd2171790f7dcf9c765ea8017a027a2`
- Current bytes: `137,638,144` (`131.262 MiB`).
- Loader state: structurally consumable GBC2/v1, `k6Present=0`, therefore not
  render-ready as a complete GBC2-K6 transfer.

The container contains the requested `2064 x 2064 RGBA32Uint` direct
descriptor atlas and all `2,171,134` physical 32-byte triangle payloads.
Descriptor census from sixteen neighbouring owner-atlas corners:

| mode | cells |
|---|---:|
| MISS | 631,359 |
| REGULAR candidate | 0 |
| CUT2 candidate | 428,129 |
| KPLANE | 3,200,608 |

REGULAR/CUT2 candidate records carry modeKind bit 4 (`CENSUS_ONLY`). The
runtime must route them to K6 until a continuous separator compiler clears
that bit. This prevents a discrete corner census from being presented as a
continuum certificate.

## Frozen binary contract

The first 256 bytes follow the loader-owned layout: magic at 0; version at 4;
header size at 8; profile id at 12; descriptor fields at 16--32; payload fields
at 36--48; K6 presence/offset/bytes/size/layers/mips at 52--72; head offset/
bytes at 76--80; tile origin/size at 84--96; source bounds at 100--120; source
SHA-256 at 128. All sections are 256-byte aligned.

K6 storage order is mip-major, then six contiguous layers, then texel,
RGBA16F. Pair order is `(u,v)`, `(u,a)`, `(u,e)`, `(v,a)`, `(v,e)`, `(a,e)`.
The fixed head is loader-compatible float32, row-major
`A[4x14], a[4], B[18x4], b[18]`: exactly 150 floats / 600 bytes before section
padding.

## Shared-origin cap truth and bounded fit result

The previously missing exterior truth now exists for one honest production
terminal/level:

- manifest:
  `data/work/groundcover-gbc2-k6-cap-truth/2ed57f59d86e8376/e6a9a96614253e1e/manifest.json`;
- domain: top-cap entry, `5..90` degrees elevation, `4 m` camera standoff;
- pixel: `60 degrees / 1920` full angular width, `4x4` quadrature;
- samples: `4,096` train plus `1,024` held-out validation pixels;
- every pixel has one exterior pinhole origin and every subray independently
  intersects the top carrier before tracing the infinitely repeated accepted
  `2,171,134`-triangle source;
- cook cost: `81,920` exact mesh rays, `21.8 s` on the generating machine;
- observed truth is nontrivial: median total coverage `0.375`, p95 `1.0`,
  conditional first-hit depth p95 `4.58 m`, maximum `11.31 m`.

The bounded fit against the then-frozen loader contract is:

`data/work/groundcover-gbc2-k6-fit/2ed57f59d86e8376/a36aa1c9b242e55d/report.json`

It trained the six `448x448 RGBA16F` pair planes and the frozen
`product -> phi14 -> SiLU rank4 -> raw18` head for 4,000 steps, then scored
only the quantised tables through the prospective bilinear decoder. It is
decisively **RED**:

| held-out metric | required | measured |
|---|---:|---:|
| silhouette IoU | `>=0.97` | `0.4860` |
| coverage absolute p95 | `<=0.15` | `0.9870` |
| premultiplied RGB max-channel p95 | `<=0.15` | `0.6343` |
| conditional mean depth p95 | `<=0.05 m` | `4.5588 m` |
| conditional normal p95 | `<=20 deg` | `157.79 deg` |

This is not a table-quantisation-only miss: the sampled training table itself
reached only IoU `0.8324`, RGB p95 `0.3125`, and depth p95 `4.4796 m`.
The stable production asset was not mutated: it remains exactly
`137,638,144` bytes with SHA-256
`b02e9e0cea4e3b225a92841a7a8011496cd2171790f7dcf9c765ea8017a027a2`
and `k6Present=0`.

### Premise audit

The truth cook matches the intended exterior common-carrier setup and the
fit matches the frozen binary decoder. However, that decoder is not the K6
head described in Section 10 of
`GRASS-RAY-SPACE-CUT-CELL-CODEC.md`. Section 10 says the six samples remain
`24` features and the shallow head additionally receives analytic `z` and
pixel-Jacobian invariants. It explicitly identifies a `rank-four product plus
quadratic head` as the rejected earlier pointed-ray experiment. The frozen
loader contract used by this fit instead multiplies six four-channel factors,
lifts that four-vector to `phi14`, and passes it through rank four with no
analytic address or footprint inputs. The bounded fit has therefore measured
the frozen loader bottleneck faithfully, but that bottleneck contradicts the
selected mathematical proposal.

That one permitted diagnose/fix rerun has now also completed. It kept all 24
sampled channels independent, used 18 primary sparse direct accumulators,
assigned the six spare channels by loss sensitivity on a 512-record subset of
the training split, and added a rank-two residual over all 24 channels. The
runtime arithmetic is `24 + 48 + 36 = 108` FMAs, with the same six texture
reads. `z` was already present in the six pair coordinates; `J` was constant
for this one fitted footprint level, so adding a constant affine gate could
not increase expressivity.

Corrected-head report:

`data/work/groundcover-gbc2-k6-sparse-fit/2ed57f59d86e8376/71fd627d8473f8b3/report.json`

| held-out metric | required | corrected measured |
|---|---:|---:|
| silhouette IoU | `>=0.97` | `0.5976` |
| coverage absolute p95 | `<=0.15` | `0.8607` |
| premultiplied RGB max-channel p95 | `<=0.15` | `0.5633` |
| conditional mean depth p95 | `<=0.05 m` | `2.5263 m` |
| conditional normal p95 | `<=20 deg` | `138.63 deg` |

The continuous generator is almost equally red (IoU `0.6000`, RGB p95
`0.5544`, depth p95 `2.7797 m`), and the sampled training set is also red
(IoU `0.6009`, RGB p95 `0.3840`, depth p95 `2.1193 m`). Therefore RGBA16F
quantisation and bilinear table interpolation are not the binding failure.
The corrected low-order pairwise/sparse terminal cannot represent this
filtered arbitrary-soup visibility field under the fixed small-head envelope.

The two permitted end-to-end fits are exhausted. **GBC2-K6 is parked for this
accepted arbitrary triangle soup.** Resume only with a mathematically distinct
O(1) terminal whose additional information capacity is derived and budgeted
before fitting—not more steps, looser thresholds, more runtime candidates, or
the same pair field under a renamed head.

## Separately named K6-CP12 candidate

One mathematically distinct candidate was subsequently authorized and tested:
**K6-CP12**, a complementary-pair tensor rather than an additive/product head.
The six signed RGBA planes create twelve genuine four-coordinate rank
features, channelwise:

`uv * ae`, `ua * ve`, `ue * va`.

A dense linear `12 -> 18` head costs `216` FMAs. Reads (`6`) and table bytes
are unchanged. There is no hidden layer or runtime candidate work.

Full 7,000-step report:

`data/work/groundcover-gbc2-k6-cp12-fit/2ed57f59d86e8376/c33e8d4e709c4d19/report.json`

The factors emerged after the initial plateau around step 2,750. Logged batch
loss then progressed through `3.77296 @ 3000`, `3.57184 @ 4000`,
`3.38653 @ 5250`, `3.19850 @ 6500`, with the best logged value
`2.94160 @ 6750` and final `2.99833 @ 7000`. This is converged enough to
exclude the three-step smoke artifact as evidence.

K6-CP12 is also decisively **RED**:

| held-out metric | required | CP12 sampled measured |
|---|---:|---:|
| silhouette IoU | `>=0.97` | `0.5810` |
| coverage absolute p95 | `<=0.15` | `0.8720` |
| premultiplied RGB max-channel p95 | `<=0.15` | `0.5002` |
| conditional mean depth p95 | `<=0.05 m` | `5.5066 m` |
| conditional normal p95 | `<=20 deg` | `136.83 deg` |

The continuous generator is similarly red (IoU `0.5738`, RGB p95 `0.5009`),
so table quantisation/interpolation again is not the blocker.

Post-gate six-bin truth-tensor SVD diagnostics show substantial residual rank
in every complementary matricization:

| matricization | rank-4 residual median / max | rank-12 residual median / max |
|---|---:|---:|
| `uv | ae` | `0.4297 / 0.7596` | `0.2543 / 0.4497` |
| `ua | ve` | `0.4241 / 0.7577` | `0.2514 / 0.4611` |
| `ue | va` | `0.4195 / 0.7724` | `0.2451 / 0.4500` |

Those are approximate relative Frobenius residuals of the fully binned truth
tensor (occupied-bin fraction and per-output spectra are in the report). They
confirm that four ranks in each complementary family are far below the
measured field complexity. K6-CP12 is parked. No mip/head was appended.

## Earlier truth blocker (resolved for one cap level only)

No pre-existing truth artifact was admissible for this fit:

- `groundcover-transfer-field` and `groundcover-origin-aware-codec` encode an
  interior-origin five-dimensional state; the current exterior-box contract
  deliberately removes that state.
- the existing box-boundary filtered-codec experiment kept its small truth in
  memory and records six known setup invalidities; it cannot supervise a
  production asset.
- the existing GCRP owner/geometry atlas has only the old sampled directions
  and is not shared-origin production-pinhole truth. Fitting K6 to it would
  reproduce the very wrong-view/discrete-angle field K6 is meant to replace.

The new cook resolves that blocker only for one top-cap level. A complete
asset still requires shared-origin truth over all fitted footprint levels,
the exact-horizontal/non-top charts, carrier seams, and disjoint translated
camera families. K6 may be appended only after the reconciled head passes the
held-out gate in that complete declared domain.

## Resource projection

- Current descriptor + triangle payload container: `137,638,144` bytes.
- Six complete `448^2 RGBA16F` fitted level stacks: `12,844,752` bytes.
- Head plus alignment budget: at most `786,736` bytes.
- Complete cap artifact before external control/non-top pages:
  `151,269,632` bytes (`144.261 MiB`).
