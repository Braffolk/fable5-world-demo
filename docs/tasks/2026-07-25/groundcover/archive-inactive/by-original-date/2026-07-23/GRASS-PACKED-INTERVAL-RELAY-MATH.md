# Rank-packed interval relay mathematics

Date: 2026-07-23  
Status: **NO-GO as the all-angle structural carrier; conditionally useful only
as a bounded-rank refinement with a certified non-crisp overflow path**

This note audits one proposed successor to the parked F4/F6 ruled-field basis.
It is deliberately renderer-independent.  No runtime or shader code was
changed and no broad CPU gate was run.

## 1. Proposed representation and verdict

For each of two Tier-1 population transforms, query four affine structural
fields.  One descriptor read per `(population,field)` returns the first
`K=4` crossings of the field's intrinsic periodic 2D curve network.  Crossing
`k` is packed as

```text
rho16 | hMin8 | hMax8
```

so the four crossings occupy one 128-bit texel.  Live algebra reconstructs
the axial coordinate at every crossing, rejects crossings outside their
finite interval, and elects the nearest valid one among the resulting 32
lanes.  One dependent read at `(winning field, winning texel, winning rank)`
fetches the coupled categorical payload.  One control read gives a nominal
total of ten texture reads.  A separate finite-spectrum prefix integral is
intended to carry common-colour microscopic hairs without a texture read.

The construction has a valid conditional decoder, derived in Section 2, but
it does **not** supply the condition that makes that decoder complete.  For
arbitrary exterior air-gap origins, no finite `K` suffices.  Exact horizontal
rays give the shortest counterexample.  Exact field-axis rays require a
different axial-successor chart which the record does not contain.  Moreover,
the 32 event lanes are a fixed candidate list in precisely the semantic sense
forbidden by the current contract: the extra lanes exist solely to enumerate
and reject possible surface events.

The proposal therefore does not advance to an implementation gate.  It may
be retained only in the narrower role stated in Section 10: an optional
rank-`K` tip refinement over an independently correct banded structural
carrier, with every overflow proven to be non-crisp and routed to a filtered
medium.  It cannot itself be the carrier.

## 2. Exact conditional algebra

The interval coordinate must be the field's canonical axial coordinate, not
unqualified world `y`.  Let `A_f` be the field-to-rest affine map, let
`pi` select its two intrinsic mask coordinates, and let `eta` select its
axial coordinate.  Apply the inverse population/wind transform and `A_f^-1`
to the live ray first:

```text
x(t)       = x_a + (t-t_a) v
q(t)       = pi(x(t))  = q_a + (t-t_a) p
zeta(t)    = eta(x(t)) = zeta_a + (t-t_a) a
s          = |p|,       omega = p/s.
```

`[t_a,t_b]` is the exact forward intersection with the field's declared
global support and the scene horizon.  For `s>0`, an ideal descriptor at
`(q_a,omega)` stores the ordered transverse curve crossings

```text
0 <= rho_1 < rho_2 < ... < rho_K
```

and the finite axial interval `J_k=[l_k,u_k]` attached to each crossing.
The corresponding world-ray candidate is

```text
t_k       = t_a + rho_k/s
zeta_k    = zeta_a + a rho_k/s
valid_k   = (t_k <= t_b) and (zeta_k in J_k)
            and the crossing's declared entry/side predicate.
```

The nearest valid `t_k` is exact **if and only if** the descriptor is complete
for the query.  Across fields and populations, the minimum of exact
per-field answers is the exact union answer, and fetching all attributes from
the elected `(texel,rank)` preserves owner, colour, normal, and mark coupling.
No cross-owner arithmetic is permitted.

Define the required property formally.  If the complete ordered intrinsic
crossing list is `(rho_j,J_j)_(j>=1)`, a compiled field is **`K`-relay
complete** on a query domain `Q` when

```text
for every query in Q:
  if any j is live-eligible, then some live-eligible j <= K.
```

Only this property turns the packed record into a correct MISS/first-hit
oracle.  Finite intervals, periodicity, four fields, two populations, and
offline source knowledge do not imply it.

The correct affine formulation also shows why storing `world y` intervals is
not general.  A tilted/sheared field's finite coordinate is `zeta`; under the
allowed height-proportional affine wind, `zeta` remains exact only after the
inverse live affine transform.  Comparing world `y` directly would reject or
accept the wrong part of tilted fields.

## 3. Fixed-rank incompleteness theorem

**Theorem 3.1 (no universal finite relay rank).**  For every finite `K`, there
is a periodic finite-interval field and an open set of exterior pointed rays
whose first eligible crossing has rank `K+1`.

*Construction.*  Choose one transverse intrinsic line and place `K+1`
disjoint curve arcs across it in order.  Give the first `K` arcs axial
intervals which exclude the ray's axial coordinate, and give arc `K+1` an
interval which includes it.  Choose the origin in air before the first arc.
All inequalities may be strict, so small perturbations of origin and
direction preserve both crossing order and interval eligibility.  The
failure therefore occupies an open set, not a measure-zero degeneracy.  The
packed record returns MISS while the true first hit is crossing `K+1`.  This
is Lemma 5.4 of `GRASS-EXACT-REPRESENTATION-THEORY.md` applied to the proposed
record layout.  QED.

This theorem is stronger than the generic observation that a line can cross
many blades.  It applies even when the live origin is outside all plant
matter, hence inside the promised exterior domain.  Fading an origin actually
inside a plant does not remove the counterexample.

### 3.1 Exact horizontal rays are the binding special case

For a vertical structural field and a world-horizontal ray, `a=0`; the live
axial coordinate is constant:

```text
zeta_k = zeta_a for every crossing k.
```

Place the first four projected arcs at other heights and the fifth at
`zeta_a`.  A `K=4` record deterministically rejects all stored entries and
misses the fifth.  In a periodic field, an inactive projected network may
generate arbitrarily many rejected crossings before the next active feature.
If no feature is active at that height, the same four rejected entries must
instead mean a true MISS.  The descriptor contains no fact capable of
distinguishing those two tails.

Thus exact horizontal handling is not obtained merely by avoiding a divide by
`d_y`.  It requires either:

1. a global proof of `K`-relay completeness at every height and phase;
2. a query coordinate carrying height/slope into the bake, restoring the
   dimensions the extrusion reduction removed; or
3. a different exact carrier whose masks are already height-valid.

Option 1 defines a much smaller authored class.  Option 2 recreates the
large-dimensional field.  Option 3 is the banded/family construction the
relay was intended to avoid.

### 3.2 Overlapping species cannot be free under the rank bound

Jointly cooking species into the same networks keeps the *read count* free of
a per-species factor only after per-field completeness has been established.
It does not preserve the rank certificate.  Adding an overlapping species can
insert any number of interval-ineligible projected arcs before an existing
eligible arc, raising its rank beyond four without changing its visibility.

Therefore `K`-relay completeness is not closed under species union.  Every
authored community mixture would need a new whole-union certificate.  A
claim that arbitrary future species or moss can overlap at zero cost is false
for this representation; only a finite catalogue of separately certified
community unions is supportable.

## 4. The field-axis pole is not represented

At `s=|p|=0`, `omega` is undefined and the projected query is stationary.
The transverse crossing list contains no answer.  The exact query is instead
an axial successor problem at fixed phase:

```text
first interval boundary after zeta_0 among all elements whose footprint
contains q_a, in the sign of axial travel.
```

For an exterior origin in an air gap between finite intervals this is still a
required query.  An arbitrary phase can have an arbitrarily deep stack of
disjoint or overlapping species intervals.  Four transverse crossings do not
encode that stack, and reserving four axial slots merely repeats the same
fixed-rank counterexample.

The exact pole can be categorical only if the compiled class separately
proves a bounded axial-stack depth and provides a phase-indexed axial
successor chart (for both travel signs), or if each field has one
field-global interval so ordinary occupancy plus a cap suffices.  The latter
is the F4 persistence model.  Switching a nonzero cone to the pole chart would
hide near-axis directions and is not exact.  An epsilon is likewise forbidden.

Consequently the proposed ten reads do not include the data needed by its
own required pole case.

## 5. Quantization is not uniformly bounded

### 5.1 Eight-bit interval endpoints

For one normalization span `H`, uniform 8-bit endpoints have bin width

```text
Delta_h = H/255.
```

With `H=1.15--1.176 m`, `Delta_h=4.51--4.61 mm`; round-to-nearest endpoint
error is `2.25--2.31 mm`.  Existing source measurements put p90 axial spans
at approximately `6.38 mm` for panicle axes, `5.36 mm` for glumes, and
`4.01 mm` for hairs.  These are only `1.38`, `1.16`, and `0.87` bins over a
whole-plant normalization.  Intervals can collapse, swap, or change
eligibility at exactly the recognition-critical endpoints.

Outward rounding avoids false negatives but expands each endpoint by almost
one bin, inflating a short feature by up to `9.2 mm`.  Nearest rounding keeps
smaller geometric error but can omit the first eligible event.  Either error
is categorical: one flipped eligibility test may expose a successor metres
away, so endpoint millimetres do not bound first-hit depth without a separate
free-path/rank bound.

Per-field local normalization can improve these numbers, but it does not
repair Theorem 3.1.  It also forces the global interval normalization to be
part of the field contract and payload decode.

### 5.2 Sixteen-bit crossing distance

If `rho16` is linear over the declared `155 m` intrinsic horizon, its step is
`2.37 mm` and round-to-nearest error is about `1.18 mm`.  The lifted ray error
is

```text
|delta t| = |delta rho|/s.
```

It is therefore unbounded as the ray approaches the field axis.  This is not
fixed by defining the exact `s=0` case: the punctured neighbourhood still has
arbitrarily large lift amplification.  More importantly, genuine near-axis
hits have `rho=O(s)` and eventually fall below one fixed `rho` bin.

A shorter per-field distance cap improves precision only by categorically
turning longer valid free paths into MISS.  Half-float or logarithmic encoding
trades absolute for relative error and does not establish a uniform all-angle
surface bound.  A second near-axis chart could repair conditioning only if it
also stores the axial successor described in Section 4, adding data and a new
completeness proof.

## 6. Atlas filtering and categorical ownership

The descriptor is categorical.  Adjacent phase/direction texels can differ in
crossing owner, crossing count, order, and interval.  Linear filtering of
`rho`, endpoints, or ranks creates events and intervals belonging to no
surface.  Depth, normal, mark, and colour must never be interpolated across
those owners.

Accordingly:

- descriptor fetches must use exact integer/nearest semantics;
- descriptor and dependent payload must use the same texel, direction slice,
  LOD, and rank convention;
- the payload read must be winner-only;
- ordinary averaged mipmaps are invalid; every mip must be recooked either as
  a categorical ray-bundle event representation or as an explicitly filtered
  extinction/coverage representation.

Nearest categorical sampling leaves phase and intrinsic-angle quantization.
For a valid extrusion family its angular displacement is local,
approximately `rho_free Delta_omega`, rather than camera-distance-amplified,
but it still requires a measured free-path lattice law.  Increasing `K` does
not reduce this error.  At coarse footprints one rank list cannot represent
all rays in the footprint; a separate opaque-to-medium mip law is still
required for structural content, not only for hairs.

The dependent payload lookup is coherent mathematically, but it is a true
dependent memory operation: it cannot begin until all descriptor candidates
have elected a winner.  The nominal read count alone understates that latency.

## 7. The 32 lanes are a candidate list

The current contract forbids candidate lists in addition to loops and
marching.  Compile-time unrolling proves constant asymptotic work, but it does
not change what the work is.  Every packed lane is a possible geometric event;
the decoder enumerates 32 such events, runs eligibility predicates, and takes
their minimum.  The extra ranks exist solely because an earlier event may be
rejected and a later candidate tried.

This differs from taking the minimum of one exact answer from each independent
geometric family.  In the latter, each lane is a closed-form component answer.
Here each lane is an incomplete search prefix.  It is exactly a bounded
candidate list and is the fixed-width brute-force repair warned against in
the shader rules.

Even if the user separately waived the semantic prohibition, the hardware
shape is not equivalent to ten ordinary small reads: eight 128-bit integer
loads feed 32 interval reconstructions and comparisons, followed by a
dependent payload load.  Issuing all descriptor reads together exposes at
least 32 packed words plus decoded temporaries; serial unpacking reduces live
registers but serializes work before the dependent read.  Register pressure,
texture bandwidth, latency hiding, and occupancy would require measurement.
The present user statement that nine reads may fluctuate does not authorize a
32-event candidate network.

## 8. Resident-memory arithmetic

Let `P^2` be phase resolution, `N_omega` intrinsic direction slices, `F=4`
stored fields (the second population reuses them), `K=4`, and let a complete
payload atom require an optimistic `B_p=8` bytes for colour, octahedral normal,
mark/material, and flags.  Before mips:

```text
descriptor bytes = F P^2 N_omega * 16
payload bytes    = F P^2 N_omega * K * B_p
total            = 192 P^2 N_omega bytes.
```

At `N_omega=32`:

| phase grid | descriptors | 8-byte payloads | total | with 4/3 mip factor |
|---:|---:|---:|---:|---:|
| `128^2` | 32 MiB | 64 MiB | 96 MiB | 128 MiB |
| `192^2` | 72 MiB | 144 MiB | 216 MiB | 288 MiB |
| `256^2` | 128 MiB | 256 MiB | 384 MiB | 512 MiB |

Even an implausible four-byte *complete* payload leaves the `256^2 x 32`
case at 256 MiB before mips.  The 128-bit integer descriptors are not eligible
for ordinary lossy block compression while preserving ranks, distances, and
interval tests.  Payload colour may compress separately, but categorical
marks and normals still need an owner-preserving representation.

Thus the only nominally comfortable row under the 250 MB working ceiling is
`128^2 x 32` (or a similarly reduced nonuniform allocation).  Over a `0.52 m`
tile its phase pitch is `4.06 mm`, already comparable to the `3--6 mm` crisp
blade/glume scale and much coarser than head microstructure.  Sparse fields may
also require more than 32 direction slices under the measured
`Delta_omega <= epsilon/rho_p95` law.  The proposed format therefore spends
its memory margin before proving either phase or angular adequacy.

This accounting excludes the control field, finite-spectrum constants,
community palette residency, custom coarse mips, and any required pole chart.

## 9. Can Calamagrostis be reauthored into rank four?

In principle a specially authored field can be certified `K`-relay complete.
The exact authoring constraint is severe:

```text
For every phase, intrinsic direction, exterior axial origin, and axial slope,
no eligible crossing may have four or more ineligible predecessors in that
field.
```

It must also hold after both population transforms, at both axial travel
signs, at exact horizontal and near-axis directions, and after every supported
species/community union.  This is a global visibility constraint, not a local
per-primitive fitting error.

The accepted Calamagrostis statistics point in the opposite direction.  Its
reproductive range spans about `410.5 mm`, while recognition-critical axes and
glumes have p90 vertical spans of only about `4--6 mm`.  Putting even half of
that head range into one or two networks makes most projected arcs inactive at
any fixed horizontal height.  The ratio is tens of inactive height cohorts
per active cohort before spatial overlap and the second population are added.
This is not a formal asset-specific lower bound on rank because projected
ordering also matters, but it makes `K=4` an unsupported assumption, not a
plausible certificate.

There are only three systematic ways to force the property:

1. split networks into many narrow height/axis fields;
2. extend intervals so most early crossings become eligible; or
3. delete/merge finite features until projected inactive crossings disappear.

The first restores the field/read explosion measured by the Class-E CPU gate.
The second restores F4's false vertical persistence and early periodic
successors.  The third removes the fluffy finite head structure whose fidelity
is the present objective.  Adding more packed ranks moves these boundaries but
does not alter the theorem; the required `K` is a measured whole-community
quantity and is not closed under multi-species overlap.

## 10. Narrow corrected use

There is one mathematically honest role for a small packed rank tail:

1. Start with an independently correct, direction-analytic structural carrier
   whose MISS and pole semantics do not depend on the rank tail (for example,
   exact banded masks with field-global eligibility).
2. Treat up to `K` per-texel finite-tip records as a local silhouette
   refinement only.
3. Prove offline that every overflow belongs exclusively to content below the
   physical pixel-resolution threshold over the declared view domain.
4. Route that overflow to a nonnegative filtered medium with declared colour
   and depth-spread bounds; never return MISS or a wrong crisp owner because a
   tail entry was omitted.
5. Keep coloured glumes, anthers, axes, blade tips, and any other resolvable
   categorical feature in the exact carrier.

This is the bounded-rank refinement already foreshadowed in
`GRASS-EXACT-REPRESENTATION-THEORY.md` Section 13.3.2.  It cannot repair the
parked F4/F6 carrier because their fallback geometry is itself the source of
the false periodic winners.  A new exact fallback would have to be found
first.

The zero-read finite-spectrum hair integral does not change that conclusion.
It can approximate genuinely sub-pixel, common-colour extinction after the
structural head is present.  A finite nonnegative analytic field cannot retain
exact open air gaps unless it vanishes there by construction, and it cannot
carry the missing categorical purple/cream owners.  It is a soft residual,
not a successor oracle.

## 11. Decision and objective resume condition

**Do not implement the rank-packed interval relay as proposed.**  It fails
four independent obligations:

1. `K=4` has no all-exterior completeness proof and is generically false,
   with an exact-horizontal open-set counterexample;
2. the exact field-axis pole requires a different axial-successor record;
3. fixed endpoint/distance precision has categorical and near-axis unbounded
   error modes; and
4. the 32-lane enumeration is a forbidden candidate list, while useful
   phase/direction resolutions exceed the low-memory target.

Resume this specific route only if all of the following become true:

- the user explicitly authorizes a fixed event-candidate network rather than
  only a small fluctuation in texture reads;
- a compiler emits a machine-checkable `K`-relay-completeness certificate over
  every exterior phase, direction, origin height/slope, population, and
  supported species union;
- a separate exact pole/near-pole chart and its byte/read cost are supplied;
- measured `P`, `N_omega`, quantization, custom mip, and payload bytes fit the
  working ceiling; and
- every rank overflow is proven non-crisp and has a bounded filtered fallback.

Without those conditions, increasing `K`, endpoint bits, or read count is not
a breakthrough.  It is a larger finite search prefix of a successor problem
whose depth remains unbounded.

