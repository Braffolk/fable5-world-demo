# K6-CP12 complementary-pair tensor red blocker

**Date:** 2026-07-23  
**Candidate:** K6-CP12  
**Verdict:** `RED_K6_CP12_FIT`; stable production asset unchanged

## Candidate

K6-CP12 preserves the existing six pair-plane reads and `448x448 RGBA16F`
memory, but replaces the parked low-order heads with actual four-dimensional
rank products:

- four signed ranks from `uv * ae`;
- four signed ranks from `ua * ve`;
- four signed ranks from `ue * va`;
- one dense linear `12 -> 18` output head (`216` FMAs), then the constrained
  two-stratum moment transform.

There is no hidden network at runtime, no extra texture, no candidates, and no
loop. The offline plane generator is not part of the runtime cost.

## Full fit

Immutable truth:

`data/work/groundcover-gbc2-k6-cap-truth/2ed57f59d86e8376/e6a9a96614253e1e/manifest.json`

Full report:

`data/work/groundcover-gbc2-k6-cp12-fit/2ed57f59d86e8376/c33e8d4e709c4d19/report.json`

The 7,000-step MPS fit did not remain at its early constant solution: ranks
emerged near step 2,750, loss fell from about `5.2` to `3.77 @ 3000`,
`3.39 @ 5250`, `3.20 @ 6500`, best logged `2.9416 @ 6750`, and final
`2.9983 @ 7000`.

Held-out sampled-table result:

- IoU `0.5810` against `0.97`;
- coverage p95 `0.8720` against `0.15`;
- RGB p95 `0.5002` against `0.15`;
- mean-depth p95 `5.5066 m` against `0.05 m`;
- normal p95 `136.83 deg` against `20 deg`.

Continuous-generator metrics are almost the same, ruling out RGBA16F and
bilinear sampling as the principal error.

## Rank diagnosis

After the red gate, all 5,120 records were binned into a `6^4` truth tensor.
Empty bins were filled with the occupied-bin mean; the report records the
occupied fraction and all 18 output spectra. Relative Frobenius tail energy:

| partition | rank 4 median / max | rank 12 median / max |
|---|---:|---:|
| `uv | ae` | `0.4297 / 0.7596` | `0.2543 / 0.4497` |
| `ua | ve` | `0.4241 / 0.7577` | `0.2514 / 0.4611` |
| `ue | va` | `0.4195 / 0.7724` | `0.2451 / 0.4500` |

The accepted soup's filtered visibility field has substantial tail rank in
every complementary partition. Four ranks per partition cannot meet the
quality contract; even twelve ranks in any one matricization leave roughly
`25%` median and `45%` worst relative residual in this conservative binned
diagnostic.

## Park / resume condition

K6-CP12 is parked. The stable GBC2 remains SHA-256
`b02e9e0cea4e3b225a92841a7a8011496cd2171790f7dcf9c765ea8017a027a2`,
`137,638,144` bytes, `k6Present=0`.

Resume only with a representation whose information-capacity increase follows
from these spectra and still has explicit fixed read/FMA/memory accounting.
Increasing optimizer steps or changing the output loss cannot erase the
measured rank tail.
