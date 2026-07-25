# Coupled phase--direction functional codebook audit

Date: 2026-07-24  
Verdict: **ordinary angular-FE prototype records are already RED; the indirection
is a valid same-read factorisation but only an event-adaptive, shared-edge prototype
record is mathematically new**

## 1. Proposed field and exact continuity condition

Let the periodic phase torus be triangulated.  Each canonical phase vertex `v`
stores one prototype id `p(v)`.  For the current phase triangle with barycentrics
`lambda_i`, decode the three complete prototype functions and return

```text
M(q,d,sigma) = sum_(i=0)^2 lambda_i(q) F_(p_i)(d,sigma),
M = (premultiplied RGB, coverage).
```

This is structurally `C0` in phase if and only if the id of every phase vertex is
cooked once and duplicated bit-identically into every incident phase cell and torus
seam.  Prototype clustering does not break that proof: an id is categorical only at
a mesh vertex; decoded positive measures, not ids, are interpolated.

It is structurally `C0` in direction if and only if every prototype's incident
direction-cell records have bit-identical restrictions on their shared angular edge,
the azimuth seam is canonical, and the pole has one azimuth-independent limit.  If
both conditions hold, a phase-weighted sum of the prototype functions is jointly
`C0` in phase and direction.  Two positive filtered-scale endpoints in the same
record can also be blended continuously in footprint.

Thus the topology is valid.  The question is the approximation space carried by
each `F_p`.

## 2. Ordinary FE is Candidate L's measured tensor ceiling

Suppose one prototype direction-cell record stores the three angular-vertex measures
for two filtered scales and evaluates them affinely.  If every phase vertex is given
its own prototype, then

```text
sum_i lambda_i(q) [sum_j mu_j(d) M_(i,j)]
```

is exactly the `3 phase vertices x 3 direction vertices = 9`-vertex tensor-product
finite element.  Prototype sharing can only identify/quantise some rows of this
tensor; it cannot enlarge the function space.

Candidate L measured this exact unlimited-precision `tensor9` ceiling on the real
Calamagrostis positive-measure truth.  It was RED in `40/40` targeted cases, with
worst coverage p95 `0.5112`, premultiplied-RGB p95 `0.4307`, and connected error
`0.5787`.  It was essentially identical to the five-vertex simplex and the strong
same-phase Candidate-K baseline.  Candidate K's independent refinement audit was
still RED after eightfold angular refinement, while the corresponding uniform field
already cost about `2 GiB`.

Therefore the ordinary-FE functional codebook is conclusively RED without another
harness.  Its dependent reads improve traffic relative to nine direct product
vertices; they do not repair visibility events absent between the angular vertices.

## 3. Relationship to earlier latent and codebook results

- The earlier linear/SVD latent gate reached only `91.87%` hit agreement at rank 32
  with `0.368 m` p95 hit-height error and needed eight RGBA spatial reads; only all
  64 directions were exact.  The proposed categorical phase prototype is nonlinear
  and may use one unique prototype per phase vertex, so that rank result is not a
  direct packing no-go.  It does, however, reject replacing the complete prototype
  function by another small linear latent basis.
- The owner-conditioned codebook found only `1.81%` exact hit-record deduplication
  and poor dependent-read locality.  This proposal codes filtered positive response
  functions rather than triangle owners, and the byte budget permits unique ids, so
  the owner repetition percentage is not a theorem against it.  It remains a warning
  that useful prototype sharing and cache locality must be measured, not assumed.
- Candidate J's 8-bit VQ symbols were RED, while Candidate K's direct per-node
  `RGBA4444` symbols were GREEN.  A functional codebook cannot cite J as a packing
  proof; direct positive state values must retain at least the already-gated quality
  or receive a new packing gate.
- The event-bubble audit is binding: a cell-private correction that vanishes on
  every shared angular edge cannot repair Candidate K's already-RED shared-edge
  midpoints.  Event state must be canonical and nonzero on shared edges.

## 4. Exact one-binding packing bound

Use one interior-only `RGBA32Uint` texture/array for both id records and prototype
records.  This is important: a separate `RG32Uint` id texture plus prototype texture
would add the binding which the proposal forbids.

For a `128 x 128` phase torus, store one square-cell id texel containing its four
canonical vertex ids; the selected phase triangle uses three of them.  With at most
`P=16384=2^14` prototypes, four 14-bit ids need only 56 of the 128 bits.

Use Candidate J's 64 conforming direction cells.  One 128-bit prototype texel per
`(prototype,direction cell)` gives

```text
prototype table: 16384 * 64 * 16 B = 16.000 MiB
phase id field:     128^2 * 16 B    =  0.250 MiB
filtered total:                         16.250 MiB
resolved atlas:                         32.500 MiB
combined:                               48.750 MiB
```

This is an exact capacity witness: every phase vertex can have a unique complete
response, so compression is not required for correctness.  Two affine layers reuse
the same records and cost `2 * (1 id + 3 prototype) = 8` physical texture loads.
The three prototype accesses may be far apart and have materially worse locality than
the direct atlas; low/mid-end performance remains a trace risk.

A triangle-id layout with two 16-byte id records per phase square would cost
`0.500 MiB` and exceed the target at `P=16384`.  Likewise, gutters invalidate the
bound.  The four-id square record, modulo seams, and one shared binding are therefore
load-bearing.

For an ordinary affine angular triangle, direct two-scale `RGBA4444` vertices need

```text
3 angular vertices * 2 scales * 16 bits = 96 bits,
```

so they fit but remain Candidate-L-equivalent and RED.

## 5. A genuinely new same-read event-adaptive record

There is no theorem that every nonlinear 128-bit local decoder is impossible.  A
concrete record family which escapes the ordinary-FE and boundary-zero-bubble no-gos
is a **canonical shared-edge band record**.

One packing witness is:

```text
three positive RGBA4444 region states at scale 1:  48 bits
three positive RGBA4444 region states at scale 2:  48 bits
shared edge-pair/topology and orientation:           3 bits
four 5-bit canonical chord/edge crossing positions: 20 bits
two 3-bit transition widths:                         6 bits
ordering/reserved:                                   3 bits
total:                                              128 bits
```

The two non-crossing chords connect the same selected pair of triangle edges and
partition the angular cell into three ordered bands.  Fixed `smoothstep` transitions
form nonnegative weights over the three stored positive measures; no output geometry,
owner, loop, search, or extra read is introduced.

This construction is direction-`C0` only under a strong cook invariant: whenever a
band meets a shared angular edge, the two incident records for that prototype must
duplicate the complete quantised edge trace--crossing positions, transition widths,
state ordering, and the state bits themselves.  At the pole every incident record
must reduce to one canonical state.  Private per-cell fitting is forbidden.  These
constraints let an event correction remain nonzero on a shared edge, so the measured
Candidate-K edge falsifier does not apply.

The witness is deliberately narrow.  It represents at most two ordered visibility
transitions in one coarse angular cell, assumes the transition geometry can be
continued consistently through the direction mesh, and replaces smooth affine
variation by three locally constant positive states.  Branching, crossing, closed,
or more numerous events do not fit.  Prior shell/owner-complexity measurements make
those limitations serious.  The witness proves only that "same read + structural
C0 + one small event family" is packable, not that Calamagrostis fits it.

An enriched FE with private interior bubbles is already refuted.  A quadratic FE
with all three shared edge midpoints would require at least

```text
(3 vertices + 3 edge values) * 2 scales * 16 bits = 192 bits
```

before an interior event.  It does not fit one 128-bit record.  Any proposed decoder
more general than the band witness must publish an equally explicit bit layout and
shared-edge identity proof.

## 6. Remaining scale and fidelity obligations

Two stored footprints do not by themselves define the large-footprint limit.  A
smooth interpolation between `sigma=4` and `sigma=16` is `C0`, but holding the
`sigma=16` function indefinitely preserves the moving far-field phase noise that the
active work is meant to remove.  The complete proposal still needs a continuous
contraction to the direction-conditioned torus mean, or another same-record limit,
with its constants and bytes accounted.  It may not select the two scale fields
categorically and recreate a camera-centred ring.

The event-adaptive record also needs two separately frozen results:

1. **unlimited functional fit:** with unique prototype per phase vertex, can the
   canonical shared-edge two-band family meet held-out coverage/RGB/connected and
   4.5-mm translation thresholds at both physical footprints and the far limit?
2. **prototype compression:** only after (1), reduce `P` if desired and report added
   error plus dependent-read locality.  Compression may not obscure failure of the
   event family itself.

The direct truth must include Candidate K's RED shared ring-edge midpoints,
Candidate L's two targeted worst cells, pole/seam limits, and deliberately branching
or crossing angular events.  If unique prototypes and unlimited state precision are
RED, clustering, quantisation, another id layout, or runtime inspection cannot help.

## 7. Verdict

The coupled functional indirection is **genuinely useful as a read-factorisation**:
it can evaluate three complete phase-vertex response functions in four accesses per
layer, remain jointly `C0`, use one binding, and fit the byte ceiling even without
prototype sharing.

It is **not** a fidelity solution when each prototype is an ordinary angular FE;
that form is algebraically Candidate L's measured-RED tensor9.  Progress requires an
event-adaptive prototype record whose shared-edge traces are canonical.  The 128-bit
two-band witness shows one such constrained family is packable, but no prior result
establishes that its two events per cell are sufficient.  That sufficiency--not the
id indirection--is the only decision-changing gate left for this proposal.

