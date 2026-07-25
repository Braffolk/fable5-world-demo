# GrassProfile2 event-bubble same-read audit

Date: 2026-07-24  
Status: **NO-GO; do not implement**

## Question

Could Candidate K's continuous angular finite element be rescued without an
extra physical profile read or additional resident memory by putting one (or,
if packing allowed it, two) analytic soft visibility events inside every
phase/angular cell?  The proposed correction was required to vanish on all
angular-cell edges so continuity would be structural rather than obtained by
filtering or threshold tuning.

This note considers only that representation.  It does not alter a shader,
runtime path, bake, or test harness.

## Strongest form of the proposed construction

Triangulate each angular cell and let `lambda_0, lambda_1, lambda_2` be its
barycentric coordinates.  For positive premultiplied appearance measure
`M=(A C_r,A C_g,A C_b,A)`, the base finite element is

```text
M_FE(lambda) = sum_i lambda_i M_i.
```

The normalized cubic triangle bubble is

```text
B(lambda) = 27 lambda_0 lambda_1 lambda_2,
```

so `0 <= B <= 1` and `B=0` on every triangle edge.  One adaptive soft
half-plane event can be written as

```text
S(lambda) = sigmoid(k (a_0 lambda_0 + a_1 lambda_1 + c)),
M(lambda) = (1 - B S) M_FE(lambda) + B S M_E.
```

With nonnegative endpoint measures this remains a nonnegative
premultiplied measure.  Since `B=0` on every shared edge, adjacent triangles
agree there regardless of their private event parameters.  It is therefore an
exact angular-C0 construction.  It adds fixed ALU only and no loop, march,
candidate list, or dependent read.

## Best possible 128-bit packing

Candidate K spends all `128` bits on four angular corners at two spatial
scales:

```text
4 corners * 2 scales * RGBA4 (16 bits) = 128 bits.
```

Triangulation frees one corner:

```text
3 corners * 2 scales * RGBA4 = 96 bits.
```

The strongest plausible single-event use of the remaining `32` bits is:

```text
event target at scale 0: RGBA3 = 12 bits
event target at scale 1: RGBA3 = 12 bits
shared line angle + offset:        8 bits
total:                           128 bits.
```

Sharpness must be fixed globally and event strength must be expressed through
the target measure and bubble.  There is no room for a second independently
located event.  Reducing the base to RGBA3 could create room, but Candidate K
only established the stored-node packing gate for RGBA4; a three-bit channel
has a `1/7` endpoint spacing and would reopen a packing failure before testing
the event model.  It is not a justified escape hatch.

At 16 bytes per cell, the exact memory limit permits

```text
64 angular pages * 128^2 phase cells * 16 B = 16.000 MiB.
```

Together with the fixed `32.5 MiB` exact atlas this is `48.5 MiB`, leaving only
`0.25 MiB` under the `48.75 MiB` combined ceiling.  Candidate K's stored-node
gate previously showed that phase resolution 128 with RGBA4 can be acceptable;
the arithmetic itself is therefore not the blocker.

## Fixed query cost, including two affine layers

One record contains both spatial-scale endpoints, so footprint interpolation
does not require another physical texture read.  A legal staged schedule could
have been:

```text
resolved exact zone:                   8 exact reads
exact -> layer-1 bridge:               8 exact + 1 filtered = 9 reads
layer-1 -> layer-2 bridge:             1 + 2 filtered = 3 reads
fully filtered zone:                   2 filtered reads
```

Thus two global affine layers do not by themselves violate the nine-read cap.
The event-bubble candidate fails for fidelity, not because of reads, ALU, or
memory arithmetic.

## Decisive edge theorem

Let `e` be any angular-cell edge.  The required structural-continuity condition
gives

```text
B|_e = 0,
therefore M|_e = M_FE|_e.
```

This remains true for any number of private bubble-gated events, any event
sharpness, any target measure, and any cook-side fit.  Consequently:

> If the base finite element exceeds a frozen fidelity threshold at even one
> held-out point on an angular-cell edge, no cell-private boundary-zero event
> correction can make that point pass.

Candidate K already contains exactly that falsifier.  Its angular lattice has
the stored elevation ring `0.25 degrees` and azimuth vertices every `22.5
degrees`.  Therefore every direction

```text
elevation = 0.25 degrees,
azimuth = (j + 1/2) * 22.5 degrees
```

lies in the interior of a shared ring edge.  A triangle mesh over the same
ring/wedge lattice must retain that shared edge; changing the cell diagonal
does not move it.

The unlimited-precision Candidate-K held-out report measured four such edge
midpoints at the physical scale-4 kernel.  Coverage happens to be saturated,
but premultiplied-RGB p95 and connected wrong-region fractions are:

```text
azimuth     RGB p95       connected wrong region
 11.25       0.25425             11.84%
 78.75       0.25511             11.13%
168.75       0.26456             13.09%
258.75       0.27173             12.74%
```

The frozen limits are RGB p95 `<= 0.06` and connected wrong region `< 1%`.
These edge errors are about `4.2--4.5x` the RGB limit and `11--13x` the
connected-region limit.  The scale-16 kernel is also RED at every listed edge
midpoint (`RGB p95 0.168--0.186`, connected `6.6--10.5%`).

Because the proposed correction is identically zero at those sample sites,
its output is bit-for-bit the already-failed unlimited base FE there.  No fit
or quantizer can change this result.  This is a direct measured contradiction,
not an extrapolation from Candidate K's poor convergence rate.

## Why prior K/L convergence does not rescue it

Candidate K's independent axis-refinement audit remained RED after reducing
both angular widths to one eighth.  Candidate L's conforming 4D simplex and
even its full nine-vertex tensor ceiling reproduced essentially the same
failure.  Those results established that shared-vertex conformity and denser
piecewise-affine interpolation do not encode the missing visibility/colour
events.

The bubble proposal correctly targets that missing interior event class, but
its structural-C0 mechanism forbids it from acting on shared edges.  The edge
falsifier above is therefore stricter than the convergence result: even a
perfectly fitted interior event cannot repair all exterior directions.

Allowing the event to remain nonzero on an edge would require the two incident
cells to share the complete edge-event parameters.  That is a different
representation.  Under one independent 128-bit cell read it cannot be made
canonical without either duplicating additional shared edge state, losing the
already-full base/event packing, or adding a separate shared-edge read.  All
three violate this candidate's frozen premise.

## Secondary continuity limitation

The construction proves C0 only in angle.  Candidate K addresses one phase
cell per read.  A fully continuous phase-and-angle product field would need
shared phase vertices as well as shared angular vertices.  Even the smallest
triangle-product base needs

```text
3 phase vertices * 3 angular vertices * 2 scales * 16 bits = 288 bits
```

before event parameters.  A conforming 4D simplex still needs

```text
5 vertices * 2 scales * 16 bits = 160 bits
```

before events.  Candidate L paid multiple reads for precisely this product.
Thus the one-read 128-bit event bubble also cannot provide structural camera-
translation continuity; at best it retains Candidate K's phase-cell behavior.
This is not needed for the decisive RED verdict, but prevents treating the
codec as a complete continuous solution even if the edge data had passed.

## Verdict

**NO-GO.**  Do not implement or run a new gate.  The existing unlimited-
precision Candidate-K held-out artifact is already an exact gate for the
proposed representation at shared edges, where its correction is required to
be zero.  One event fits only by using coarse RGBA3 event targets and cannot
change the measured RED samples; two events do not fit at all.  The path also
does not solve phase continuity.

Resume only for a materially different codec that makes visibility-event state
canonical on shared angular edges and shared in phase while still proving its
one-read, no-memory-increase packing.  Renaming cell-private events or changing
their sigmoid/bubble shape cannot satisfy that condition.

## Evidence

- Candidate-K held-out report:
  `data/work/groundcover-candidate-k-heldout-fe/e3e0a4175b151b89/60a8e06b5f69e390/report.json`
- Candidate-K convergence report:
  `data/work/groundcover-candidate-k-angular-convergence/e3e0a4175b151b89/18fafa8984a3c8ac/report.json`
- Candidate-L targeted gate:
  `data/work/groundcover-candidate-l-targeted-fe/e3e0a4175b151b89/1931b9fb2d09903b/report.json`

