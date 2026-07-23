# Guarded common-slab boundary-transfer CPU gate

**Date:** 2026-07-23
**Source:** accepted production *Calamagrostis canescens*, SHA-256
`2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c`
**Status:** **INVALID / FULL DOMAIN UNDECIDED; do not implement the runtime
representation.** The periodic interior point-ray factorization is
numerically confirmed. No finite-codec RED or GREEN in this file is valid as
a final filtered-image decision: the first used point-ray diagnostics, the
second treated a six-pixel diagnostic as the whole sensor, and the
production-pixel rerun exposed additional setup defects before codec scoring
could be certified.

## Historical shared-origin attempt (superseded)

The earlier centre/corner finite-codec verdict is superseded. It lacked a
shared pinhole origin, per-subray carrier entry, standoff-dependent footprint
covariance, and an actually fitted codec. Its `192/192` point-ray
factorization result remains valid because that test did not use the bad
filter metric.

The corrected harness and immutable artifact are:

- `tools/groundcover-bake/analyze-box-boundary-filtered-codec.ts`;
- `data/work/groundcover-box-boundary-filtered-codec/`
  `2ed57f59d86e8376/644a4bf0b74b66a6/`.

It is reproduced with:

```sh
npx tsx tools/groundcover-bake/analyze-box-boundary-filtered-codec.ts
```

Five explicit exterior camera families are used: top, 10-degree oblique,
1-degree grazing, horizontal finite-patch side, and oblique finite-patch
side, plus 1 mm and 4.5 mm translations. Every pixel subray shares its
pinhole origin, obtains its own exact top-cap or ecological-patch side `q`,
and accumulates premultiplied barycentric colour, coverage, the first two
transfer-depth moments, and barycentric normal moments.

Integration is frozen at `4x4 -> 8x8`, escalating once to `16x16`. The tested
representation stores one finest point field and filters it live. Subray
address covariance in `(q0,q1,azimuth,elevation)` selects one of four
principal-axis families and radius `{0,1,2,4}` cells; every family performs
two symmetric reads from the same charged finest table. There are no hidden
prefiltered levels.

The frozen Q128/A64/E32 cap plus canonical-D4-side address grid contains
67,108,864 cells. A real 64-entry VQ was fitted and decoded on the 242 unique
cells addressed by the frozen cameras. Exact projected storage is:

- 67,108,864 one-byte indices;
- 1,024 bytes of codebook;
- 4,194,304 bytes of R32 carrier field;
- **71,304,192 bytes = 68.001 MiB total**.

The projected fixed cost is five reads per affine lane—one carrier, two VQ
indices, two dependent codebook records—or **11 reads** for two Tier-1 lanes
plus control.

The codec is sampled-red on the base cameras:

- premultiplied RGB max-channel p95: **0.321**;
- coverage absolute p95: **0.469**;
- silhouette IoU: **0.805**;
- conditional transfer-depth p95: **13.226 m**;
- normal-moment L2 p95: **0.823**.

However, truth integration itself did not converge sufficiently. 137/180
pixels escalated and 126/180 (**70%**) still failed at least one declared
`8x8 -> 16x16` tolerance. Final-step RGB p95 was 0.0677, coverage p95 0.125,
depth p95 0.496 m, and normal-moment p95 0.201. These numbers were initially
treated as enough to reject that frozen candidate. The production-pixel audit
below supersedes even that narrow conclusion: the diagnostic pixel footprint
was wrong and the rerun exposed additional validity defects.

Only the 242 addressed cells were fitted. Full-grid index bytes are charged,
but codebook error outside the explicit cameras remains unmeasured. The
corrected conclusion is:

> **HISTORICAL SAMPLED RED; SUPERSEDED BY INVALID / FULL DOMAIN UNDECIDED;
> IMPLEMENTATION NOT AUTHORIZED.**

Resume only with a converged physical-pixel truth cook and a full-domain—not
addressed-subset—fit/decode under the same `<=14` reads and `<=250 MiB`.
Increasing quadrature or fitting the remaining field is not another tuning
variant in this attempt; it is explicitly deferred work for a future
representation/cook decision.

## Production-pixel audit: final validity status

The historical run above distributed each authored camera FOV over the
`6x4` diagnostic samples. One “pixel” therefore subtended roughly
`0.25--1.75` degrees rather than the frozen production
`60 degrees / 1920 = 0.03125` degrees. That was a premise error, not a codec
failure.

The parameter-only rerun uses a virtual production-resolution sensor for
each broad camera FOV and selects the same `6x4` stratified pixel centres
across it. Quadrature remains inside each selected production-sized pixel;
camera origins, central directions, view coverage, codec, thresholds, and
resource accounting remain unchanged.

Artifact:

`data/work/groundcover-box-boundary-filtered-codec/`
`2ed57f59d86e8376/cb0bf3cd0467a279/`

The corrected angular footprint materially changed the truth behaviour:

- 16x16 escalations fell from 137/180 to 89/180;
- unconverged pixels fell from 126/180 (**70.0%**) to 67/180 (**37.2%**);
- final-step coverage p95 fell from 0.125 to 0.00781;
- final-step RGB p95 fell from 0.0677 to 0.0303.

This confirms that the historical sampled RED was false-worsened by the
sensor parameterization. The rerun's own `SAMPLED_RED` field is also
superseded and must not be used. A post-run validity audit found:

1. the side `pointAtCell` chart does not reproduce the finite-patch truth
   phase and exact exit horizon;
2. covariance wrapping is chart-dependent—cap `q0,q1` are periodic, side only
   `q0` is—but the tested generic rule wrapped axes 0 and 2;
3. filter-family selection used oracle quadrature subray addresses rather
   than the required analytic central-ray Jacobian fixed ALU;
4. the nominal 16-byte centroid was evaluated as floating point rather than
   quantized and decoded from the charged format;
5. fit and score address sets were not disjoint, and translations did not
   score excess decoded change relative to truth; and
6. geometry stored relative transfer-depth moments without the total
   camera-depth cross term needed for compositing-depth scoring.

The harness now emits **INVALID_UNDECIDED** directly in its reproducible
report and binds the six invalidating findings there; it cannot regenerate a
misleading sampled-RED verdict. No codec tuning was performed. The only
retained conclusions are the production pixel parameter, the material
reduction in quadrature instability after correcting it, and the independent
exact `192/192` point-ray factorization. The preceding
`2eb7e84aa7bb425d` artifact and its bound `validity-addendum.json` remain the
immutable audit trail for the correction.

## Why this gate was run

The relaxed quality domain allows a ground-cover lane to fade while the
camera is inside a box containing the plant. That permits the arbitrary soup
to be enclosed in a guarded common slab `U=P x I`. The proposed state
reduction is then:

1. obtain the exact first carrier point `q` from a periodic 2D first-passage
   field `rho_P`;
2. query a four-dimensional whole-forward transfer `T(q,d)`; and
3. reconstruct the categorical winner plane analytically, using a filtered
   appearance record only in unresolved sub-pixel cells.

The CPU gate asks two separate questions which must not be conflated:

- is the common-slab factorization exact on the real triangle soup; and
- can its 4D transfer be encoded with a tiny fixed read path, acceptable
  filtered error, and at most 250 MiB?

## Reproduction and immutable output

Harness:

`tools/groundcover-bake/analyze-box-boundary-transfer.ts`

Command:

```sh
npx tsx tools/groundcover-bake/analyze-box-boundary-transfer.ts
```

Corrected run:

`data/work/groundcover-box-boundary-transfer/2ed57f59d86e8376/1d712142fe92f96d/`

It contains `report.json` and numbered QA PNGs plus their machine hash index
under `qa/`. No runtime or shader file is read as an implementation target or
modified by the harness. BVH traversal and every loop are offline truth work.

## Carrier measured on the accepted plant

Six containing boxes were derived by assigning each actual source triangle to
the nearest authored tuft centre in periodic XZ, then enclosing every assigned
vertex. Guards of 1, 2.5, and 5 cm were measured. At the active 2.5 cm guard:

- the periodic footprint union covers **100% of the 0.52 m tile**;
- the footprint has zero internal periodic perimeter and one overlap-connected
  component;
- the common slab is `[-0.0237049, 1.1765176] m`, height **1.20022 m**;
- common-slab fade volume is about **1.144x** the union of the six tight-box
  volumes.

This is a substantial visual contract: while the camera height is inside that
slab, every position in the active periodic patch lies inside the lane carrier
and the whole lane may fade. It is not selective per-shoot fade.

## Exact periodic-interior identity: green

The gate compares direct periodic analytic triangle truth against:

`exact periodic rectangle-union entry + whole-forward transfer from q`.

Across 192 deterministic exterior/edge-case rays, including 8 vertical and 64
horizontal cases:

- carrier-entry agreement: **192/192**;
- composed winner agreement: **192/192**;
- carrier entry error: exactly **0** in the sample;
- composed hit-distance error p95: `1.998e-15 m`;
- maximum composed hit-distance error: `1.954e-14 m`.

This confirms the algebra for the infinitely periodic interior. It does not
confirm the finite ecological-patch side boundary. Because `P` covers the
whole periodic tile, this run has only top/bottom cap charts and zero internal
side blocks. Exact horizontal rays entering a finite grassy patch from outside
need a separate canonical side-entry chart (finite D4 edge orientation,
boundary phase/height, and direction) or an equally exact outer control-mask
first-passage field. Until that is specified and tested, the complete world
model is mathematically incomplete even though the periodic identity passes.

## Finite codec: red

The sampled `4^4` macroblocks were not sparse:

- modal correction fraction: **50.66%**;
- mean categorical owners per block: **94.06**;
- canonical macroblock-pattern unique fraction: **100%** in the deterministic
  sample;
- worst-view mixed-cell fraction: **100%** at every tested tier.

The first run accidentally treated a centre-label codebook as if it represented
mixed cells and produced a false 66.9 MiB green. The single permitted fix cycle
corrected the accounting: each final mixed cell now carries an optimistic
12-byte filtered colour/depth/moment payload, and a tier must also pass a
held-out centre-from-four-corners quality check.

Corrected projections:

| Tier | Sparse/hash | Macroblock/codebook | RGB p95 worst view | Coverage p95 | Verdict |
|---|---:|---:|---:|---:|---|
| Q128 / A32 / E16 | 259.3 MiB | 258.9 MiB | 0.683 | 1.0 | RED |
| Q256 / A64 / E32 | 4128 MiB | 4120.5 MiB | 0.635 | 1.0 | RED |
| Q512 / A128 / E64 | 65877 MiB | 65757 MiB | 0.548 | 1.0 | RED |

The table's quality check compares a point-ray centre with four point-ray cell
corners. It is useful evidence that the field is not interpolable at those
cell sizes, but it is **not** the final physically filtered target required by
the model. A valid rate-distortion gate must integrate both source and decoded
output through the same correlated camera-pixel cone and score held-out
camera configurations. Consequently the numeric RGB/coverage columns cannot,
on their own, prove that every filtered codec is red.

The tested tiers nevertheless fail a more basic legality check. At the frozen
`60 degree / 1920 pixel` projection, one pixel subtends about
`5.454e-4 rad`. The active top guard is only `0.025 m`, so an exterior camera
arbitrarily close to the carrier boundary can resolve a footprint at the
nearest source geometry of approximately

`0.025 * 5.454e-4 = 1.364e-5 m`, or **13.6 micrometres**.

The finest tested spatial cell is `0.52 / 512 = 1.016 mm`, about **74.5x**
wider; its angular cells are also tens of pixel angles wide. Since the gate
measured those cells as 100% mixed, they are not allowed to become final
filtered atoms: they remain visibly resolved domains under the declared
closest exterior view. Reaching the physical footprint by a dense atlas would
already require about `38,136` spatial samples per tile axis before charging
direction, depth, colour, mips, or side charts. One four-byte cap for one
direction at that resolution is about **5.55 GiB**. This is not asserted as a
lower bound on every possible compressor, but it proves that the tested dense
tiers and their current final-mixed-cell interpretation are not valid.

Relaxing the byte ceiling slightly or adding a couple of reads therefore does
not make this codec implementation-ready. A different representation would
need an independently proved compression of the actual *correlated,
physically filtered* 4D signal, not another resolution increase or categorical
macroblock repack.

The measured fixed lookup cost itself remains attractive—about 4--5 reads per
affine lane and 9--11 for two Tier-1 layers including control. The blocker is
state entropy/bytes and filtered error, not runtime traversal.

## Park / resume condition

Do not build this representation in the shader. Resume only after a new
mathematical representation supplies both:

1. an exact finite ecological-patch side-entry factorization without runtime
   traversal or candidate enumeration; and
2. an independently justified reduction of the mixed 4D transfer state, with
   a concrete lower-byte codec that predicts `<=250 MiB` and is then scored on
   held-out *camera-pixel integrals* at RGB p95 `<=0.15` / coverage p95
   `<=0.25` before another actual-source cook.

Repacking the same categorical centre field, increasing atlas resolution, or
adding view candidates does not meet the resume condition.
