# GBC2-K6 cap fit red blocker

**Date:** 2026-07-23  
**Source:** accepted Calamagrostis GCRP/v4 SHA-256
`2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c`  
**Verdict:** `RED_K6_FIT`; production asset deliberately unchanged

## Why this experiment was run

The GBC2 container already had direct descriptors and the complete physical
triangle payload, but `3,200,608` KPLANE cells plus `428,129` census-only CUT2
candidates could not render without a fitted K6 terminal. Earlier truth sets
were pointed-ray or interior-origin fields and could not supervise the
exterior shared-origin transfer. This experiment supplied the smallest honest
missing dependency: one top-cap terminal at one real production pixel scale.

## What was actually tested

Truth artifact:

`data/work/groundcover-gbc2-k6-cap-truth/2ed57f59d86e8376/e6a9a96614253e1e/manifest.json`

- exterior camera origin, `4 m` from the top carrier;
- central address `(u,v,azimuth,elevation)` with elevation `5..90` degrees;
- one `60 deg / 1920` pinhole pixel per record;
- sixteen shared-origin subrays, each with its own exact top-carrier entry;
- exact periodic BVH first-hit truth against the accepted `2.17 M`-triangle
  mesh;
- deterministic two-depth-stratum filtered moments;
- disjoint `4,096` train / `1,024` validation records.

Fit artifact:

`data/work/groundcover-gbc2-k6-fit/2ed57f59d86e8376/a36aa1c9b242e55d/report.json`

- six `448x448xRGBA16F` pair planes in `uv,ua,ue,va,ve,ae` order;
- runtime-equivalent bilinear sampling of the quantised tables;
- then-frozen factor product, `phi14`, rank-four SiLU head, constrained
  two-stratum decoder;
- 4,000 MPS optimisation steps, fixed seed `23072026`.

Held-out results: IoU `0.4860`, coverage p95 `0.9870`, RGB p95 `0.6343`,
mean-depth p95 `4.5588 m`, normal p95 `157.79 deg`. The miss is orders of
magnitude beyond threshold uncertainty.

## Exact blocker found by the premise audit

The fit faithfully implements the frozen loader algebra, but that algebra is
not the selected K6 mathematics in the governing codec document. Section 10
keeps all six RGBA samples as 24 head features and also supplies analytic
address and footprint-Jacobian invariants. It explicitly calls the rank-four
product family rejected. The current loader contract collapsed 24 sampled
features to four multiplicative features before the head and omitted `z/J`.

Therefore this result rejects the frozen product/rank-four terminal on the
measured cap level. It does not yet reject the distinct 24-feature plus
analytic-`z/J` K6 proposal.

## Diagnose/fix cycle and final park

The single allowed corrected-head cycle was run after this contradiction was
found. It used:

- all 24 plane channels independently (no multiplicative collapse);
- 18 primary one-feature/one-output direct routes;
- six spare direct routes chosen by validation sensitivity on a 512-record
  subset of the training file only, leaving the 1,024 final records untouched;
- a global `24 -> SiLU rank2 -> 18` residual;
- exactly 108 decoder FMAs and the same six reads;
- no `J` gate because `J` is fixed for this single footprint level.

Report:

`data/work/groundcover-gbc2-k6-sparse-fit/2ed57f59d86e8376/71fd627d8473f8b3/report.json`

The corrected sampled tables remain red: IoU `0.5976`, coverage p95 `0.8607`,
RGB p95 `0.5633`, depth p95 `2.5263 m`, normal p95 `138.63 deg`. Its
continuous generator and training scores are also red, ruling out f16
quantisation or table interpolation as the main cause.

This is the second end-to-end red. The codec is now parked for the accepted
arbitrary soup, and no K6 bytes were appended.

## Objective resume condition

Propose a mathematically distinct O(1) terminal that:

1. has more relevant information capacity than a sum of six pair functions
   plus a rank-two residual, with that claim derived before training;
2. remains fixed-cost, O(1), low-memory, and does not add runtime candidates,
   loops, marching, passes, or geometry;
3. has exact read/FMA/residency accounting and a shared cook/loader layout;
4. is first falsified on the immutable truth split above; and
5. passes the existing held-out thresholds before any byte is appended.

More optimiser steps, threshold relaxation, or a wider version of the same
pairwise sparse head are not resume conditions.
