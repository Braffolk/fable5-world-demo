# Eight-read aligned filtered atoms — prior-art and mathematical audit

Date: 2026-07-24  
Verdict: **not a cleanly new representation; not yet decisively refuted in this exact
combination; one small existing-artifact gate is justified before any cook or runtime work**

## 1. What has already been tried

The proposal combines two previously measured mechanisms.

### 1.1 The seed-depth epipolar reread is the shell-frame corrector

`GRASS-SHELL-FRAME-FIELD.md` stores one representative residual event per view,
uses it to move along the epipolar line, rereads the corrected event, and blends
aligned premultiplied radiance.  Its strict reconstruction was measured in
`GRASS-SHELL-FRAME-FIELD-RECONSTRUCTION-BLOCKER.md`.

The fixed-97 strict result was strongly RED:

```text
silhouette IoU                 0.61974       required >= 0.97
RGB max-channel p95            0.36095       required <= 0.15
connected wrong region         44.531%       required < 1%
crisp geometry p95             12.88498 m    required <= 0.05 m
class change                   19.66--52.52% required < 5%
```

The `6/7/9/12`-read ablation did not converge with more corrections; strict
9-read radiance was slightly worse than the no-realignment result.  The diagnosed
failure was that one representative event is not a locally smooth event sheet in
sparse multi-height cover.  The seed and corrected addresses frequently name
different owners/successors.

### 1.2 The proposed physically filtered atoms are Candidate H's atoms

Candidate H cooked kernel-filtered coverage, premultiplied RGB, and covered-sample
median representative depth at `sigma=4` and `sigma=16`, then paired adjacent views
by exact epipolar reprojection.  Its measured gate is
`GRASSPROFILE2-CANDIDATE-H-ANGULAR-CONTINUITY-GATE.md`.

Every boundary family was RED.  At the 45-degree boundary and `sigma=16`:

```text
coverage difference p95        0.4706
premul-RGB difference p95       0.3012
transverse discrepancy at 2 m  129.24 pixels
connected exceedance            91.14%
```

Thus spatial filtering plus one representative depth did not make neighbouring
view atoms describe one common bundle after epipolar correspondence.

### 1.3 What was not run exactly

Candidate H selected one angular atom categorically.  Candidate K blended four
positive atoms but did not perform this representative-depth reread.  Candidate L
used a conforming phase-direction finite element, not a seed-dependent depth warp.
The shell gate used three direction nodes and one categorical event per texel, not
the proposed four physically filtered positive atoms.

Therefore it is too strong to say that the **exact combination** has already been
measured.  Its two load-bearing premises have, however, each failed independently
and by large margins.  It is an ablation-sized hypothesis, not a new open-ended
track.

## 2. Exactness and continuity limits

Let one stored node contain a positive measure `M_i(q)=(A_i,P_i)` and one
representative height `hbar_i(q)`.  In slope coordinates, the proposal evaluates
approximately

```text
q_i* = q + (hbar_i(q)-h_ref)(s-s_i),
Mhat(q,s) = sum_i w_i(s) M_i(q_i*).
```

At an exact baked direction `s=s_i`, `q_i*=q`; the node is reproduced exactly if
the seed and corrected accesses use the same spatial sampling convention.  The
warp is also exact for a single constant-depth sheet.

It is not exact for a multi-height filtered footprint.  For two components at
heights `h1 != h2`, the correct live measure contains

```text
M1(q+(h1-h_ref) Delta_s) + M2(q+(h2-h_ref) Delta_s),
```

whereas one representative depth produces

```text
M1(q+(hbar-h_ref) Delta_s) + M2(q+(hbar-h_ref) Delta_s).
```

No choice of one `hbar` makes these equal for arbitrary `M1,M2`.  This is the same
missing-stratum theorem exposed by the shell reconstruction.  Positive radiance
blending avoids inventing a geometric surface, but it does not restore the missing
height-conditioned phase shifts; it produces doubled/softened appearance when the
components disagree.

Angular `C0` is conditional.  With a conforming four-node angular cell, shared node
bits, and weights that agree on every shared edge/pole, the angular blend is `C0`
provided each shared node computes the same `q_i*` independently of the incident
cell.  Spatial `C0` additionally requires continuous sampling of both the seed
depth and corrected measure.  Point-loaded `RG32Uint` records are piecewise
constant and do **not** satisfy that condition.  Manual bilinear interpolation
would multiply the physical texel loads; a filterable same-byte representation and
its exact packing must therefore be exhibited rather than assumed.

MISS atoms also require a finite, continuous, eligibility-independent depth carrier.
Coverage zero does not make a NaN or discontinuous corrected address harmless: that
address may reread covered mass.  This is Candidate H's already-recorded atom-1
requirement.

## 3. Load, handoff, and memory claims

Four nodes times `(seed + corrected)` is exactly eight logical atom accesses.  The
`32.5 MiB + <=16.25 MiB = <=48.75 MiB` resident arithmetic is possible in the
abstract.  It is not yet a complete codec accounting because continuity, depth,
coverage, and direct premultiplied RGB all need to coexist in the stated 64-bit
record with a declared filter/access mode.

The claimed `8/9/9/8` handoff is not a continuity proof.  If the one-read bridge is
Candidate H's maximum-direction atom, its angular discontinuity has already failed
decisively.  If it is instead a packed `C0` cell record, the packing and equality of
both endpoints must be specified.  Smooth scalar weights cannot make unequal exact,
bridge, and unresolved functions equal at their joins.  Background-depth versus
exact-grass-depth switching remains a separate frontier gate.

Likewise, two physical filter scales need either a shared continuous scale basis or
an explicit scale handoff.  Selecting between two independent atom atlases by
footprint simply recreates a camera-centred ring.  The memory sum alone does not
solve this.

## 4. Decisive minimal gate

Do not cook a new codec.  Reuse the accepted Calamagrostis exact-BVH truth and the
existing Candidate-H `sigma=4/16` atoms.  In unlimited precision, for the already
frozen Candidate-K/L held-out directions and 4.5 mm translations:

1. select the four surrounding angular nodes with a conforming shared-node cell;
2. read each node's existing representative depth at the live phase;
3. perform the exact slope-chart epipolar correction;
4. reread that node's existing positive `(A,P)` atom;
5. blend the four corrected positive measures; and
6. compare to held-out exact filtered truth.

Use the existing limits without retuning:

```text
coverage p95/p99              <= 0.08 / 0.20
premul-RGB p95/p99            <= 0.06 / 0.15
largest connected exceedance  < 1%
4.5 mm translation p95        <= 0.06, connected < 1%
```

Also test exact stored nodes, all angular shared boundaries/pole, and a synthetic
two-height counterexample.  Report the one-read bridge separately; it may not borrow
the four-node result.

If unlimited-precision aligned atoms are RED, the result closes this successor:
packing, another Newton step, more spatial blur, or runtime inspection cannot supply
the missing depth strata.  If GREEN, only then specify the 64-bit filterable codec,
scale handoff, bridge endpoint identities, and actual physical-load accounting.

## 5. Bottom line

This proposal removes fake unresolved geometry and is mathematically safer than the
old shell output, but it does **not** remove the shell model's single-depth
correspondence assumption.  Prior evidence makes RED likely, not logically certain.
The exact four-node aligned-positive-measure oracle above is the one remaining
decision-changing experiment; anything larger would repeat already parked work.
