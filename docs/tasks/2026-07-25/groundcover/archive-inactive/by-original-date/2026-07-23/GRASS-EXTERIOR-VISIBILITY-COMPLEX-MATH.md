# Exterior categorical visibility-complex mathematics

**Date:** 2026-07-23
**Status:** **Top/side boundary reduction: NO-GO. Integrated 5D visibility
complex: mathematically valid but budget-UNPROVED and not implementation-ready.**
Exact point location can be padded to fixed depth and is technically `O(1)`,
but exterior air gaps retain the fifth origin coordinate. The only actual
pointed-line page probe is incomplete in elevation, does not filter origins
inside closed matter, and already consumes `219.5 MB` optimistic / `357.5 MB`
explicit at a six-decision tail. It proves that the measured page layout is
not the answer; it does **not** prove an information-theoretic byte no-go
against every unbuilt succinct 5D DAG. No runtime implementation is authorized
by the available evidence.

This is a pure mathematics and data-structure audit. It changes no shader,
runtime path, asset, or renderer.

## 1. Question and exact verdict

The route under review is the strongest exact alternative to sampled view
interpolation:

1. compile the complete marked community, including overlapping species and
   moss, into its visibility complex offline;
2. locate the live pointed ray in that complex with a fixed, unrolled decision
   program;
3. read one categorical first-surface record and evaluate its analytic surface
   chart;
4. perform no march, loop, candidate list, triangle election, or per-species
   runtime work.

There are two verdicts, which must not be conflated.

1. **Mathematical semantics: valid.** A finite-horizon polyhedral community has
   a finite semi-algebraic pointed-ray visibility complex. A padded decision
   DAG of globally bounded depth is fixed work and is not a traversal loop.
2. **The proposed 4D top/side reduction: rejected by theorem.** Literal
   exterior-to-matter includes open air gaps between blades. Those origins
   retain the pointed-ray phase, making the domain 5D.
3. **A complete integrated 5D DAG: unresolved, not disproved.** Every measured
   explicit layout is over budget and the compact layouts omit required data,
   but the existing artifacts do not measure an exterior-only, continuous-
   elevation, globally shared DAG. Claiming a universal 250 MB no-go would
   exceed the evidence.

This is not a theorem that ray shooting is impossible on all hardware. It is a
decisive rejection of the **boundary-chart dimension reduction** and of the
**measured page implementations**. The fully integrated exact 5D route remains
an expensive unproved possibility, not a solution.

## 2. Domain: exterior to matter is not exterior to the botanical slab

Let the periodically repeated, marked community be the closed set

\[
\mathcal G\subset
(\mathbb R^2/\Lambda)\times[y_{\min},y_{\max}],
\qquad \Lambda=T\mathbb Z^2,
\]

where the mark contains the species/material identity and all coupled surface
attributes. The finite query horizon is

\[
R=155\ {\rm m}.
\]

The promised exterior query domain is

\[
\boxed{
\mathcal Q_{\rm ext}
=\{(o,d):o\notin\mathcal G,\ d\in S^2\}
}
\]

with the first event

\[
N(o,d)=\min\{t\in(0,R]:o+td\in\partial\mathcal G
\text{ and is a rendered surface event}\}.
\]

If the set is empty the answer is categorical `MISS`.

The contract may fade when `o` is literally inside plant matter. It **does not**
fade a camera in the open air between blades merely because its height lies
between `y_min` and `y_max`. Therefore removing `o in int(G)` removes only a
small closed-solid subset of the pointed-ray domain. It does not remove the
origin coordinate from the open set

\[
\big((\mathbb R^2/\Lambda)\times[y_{\min},y_{\max}]\big)
\setminus\mathcal G.
\]

Modulo horizontal periodicity, `o` has three coordinates and `d` has two.
Thus `Q_ext` contains a five-dimensional open subset.

## 3. Nonsingular line and pointed-line charts

Represent an oriented line by Pluecker coordinates

\[
L=(d,m),\qquad m=o\times d,\qquad d\cdot m=0.
\]

The space of oriented lines in `R^3` is four-dimensional. For storage, choose
the dominant component `j` of `d`; then `|d_j|>=1/sqrt(3)`. Intersect the line
with a fixed chart plane `x_j=k_j`:

\[
Q=o+d\frac{k_j-o_j}{d_j},
\qquad
\ell_j=(Q_a,Q_b,d_a/d_j,d_b/d_j).
\]

The six dominant-axis charts cover vertical, horizontal, and oblique lines
without an epsilon or a pole.

A pointed ray adds the signed origin phase `sigma` on the line:

\[
(L,\sigma),\qquad o=w(L)+\sigma d.
\]

The chart is therefore `(ell_j,sigma)`, five-dimensional. The pole problem is
solved; the information problem is not.

## 4. Why top, bottom, or side boundary charts do not remove `sigma`

### 4.1 The boundary reduction where it really applies

If the origin is outside an enclosing botanical volume and lies before every
event on its ray, the ray has a canonical entry point `q` on the enclosing
boundary. Boundary position contributes two coordinates and direction two;
the exterior first-event field is four-dimensional. For a camera above the
slab and a downward ray this is the ordinary top-plane address.

This is exact and useful for that subdomain.

### 4.2 Air-gap counterexample

Fix one oriented line `L` with ordered entering events

\[
t_1<t_2<\cdots<t_M.
\]

Choose two origins in different free intervals,

\[
\sigma_1\in(t_{k-1},t_k),
\qquad
\sigma_2\in(t_k,t_{k+1}),
\]

and choose them outside the closed matter. Both cameras are exterior under the
user's contract. They have the same oriented line, the same top/bottom/side
entry record, and the same direction. Nevertheless,

\[
N(L,\sigma_1)=t_k,
\qquad
N(L,\sigma_2)=t_{k+1}.
\]

Hence no function of a boundary-line chart alone can answer both.

> **Exterior air-gap theorem.** Removing origins inside matter does not reduce
> the pointed-ray successor field to an unpointed exterior field whenever an
> admissible line contains at least two separated entering events. The origin
> phase remains a necessary coordinate on open exterior sets.

The accepted community contains thousands of such events on measured lines,
so the hypothesis is not marginal.

### 4.3 Why a two-sided hybrid does not help

A top/bottom or front/back pair gives the first event from each outer end. It
does not give the successor rank of an origin in a middle gap. A reverse-ray
query from the camera would find the previous event, but that reverse query is
itself the same 5D exterior ray-shooting problem.

The only exact hybrid is therefore

\[
\text{outside enclosing volume}:F_4(\ell),
\qquad
\text{air gap inside enclosing volume}:F_5(\ell,\sigma).
\]

The 5D branch is an open, visually important domain and dominates the hard
case. Dispatching to it does not reduce its storage or decision complexity.

For the infinitely periodic ideal community there is also no horizontal side
boundary. A side chart exists only after introducing finite world support; an
air-gap origin inside that support still requires the same successor rank.

## 5. Exact visibility-complex construction

For a fixed oriented line cell, write the ordered event functions as

\[
t_1(\ell)<t_2(\ell)<\cdots<t_M(\ell).
\]

Within a cell on which event topology and order are constant, the pointed
visibility cells are

\[
C_i=\{(\ell,\sigma):t_{i-1}(\ell)<\sigma<t_i(\ell)\},
\]

and leaf `i` returns the chart/copy record of `t_i`.

For a triangle mesh, boundaries are semi-algebraic:

- triangle-edge incidence is a reciprocal-product zero

  \[
  \Omega(L,E)=d\cdot m_E+e\cdot m=0;
  \]

- depth-order changes occur when the query line crosses the intersection line
  of two surface planes;
- pointed successor changes occur at `sigma=t_i(ell)`.

An exact offline construction may therefore build an adaptive arrangement or
binary space partition. A leaf stores one coupled owner/copy/chart token. The
selected chart evaluates the live hit point, normal, colour, and mark. Species
and moss are already one offline marked union, so source species count does not
appear in the query.

### 5.1 Fixed unrolling is `O(1)` but not automatically cheap

Let `D` be the maximum decision depth frozen for one compiled community and
horizon. Pad every path to `D` with no-op nodes. Runtime executes exactly `D`
predicate steps and one payload selection.

This has:

- no loop;
- no data-dependent termination;
- no candidate list;
- no march;
- no cost proportional to the number of periodic copies or species at runtime.

It is therefore asymptotically `O(1)`. But an asset-dependent fixed maximum can
still be a long chain of dependent memory reads. The asymptotic label does not
erase latency, bandwidth, register live ranges, divergence, or storage.

## 6. Decision-depth bounds

### 6.0 Universal generic-community no-go

No one fixed decision depth and byte table can be exact for an unrestricted
future catalogue of community geometry. For any declared binary depth `D`, put
`2^D+1` pairwise disjoint open cards across one generic (non-lattice-periodic)
line segment inside one fundamental community. Every gap origin is exterior,
and successive gaps require different next-card records. The successor
function has more than `2^D` leaves, contradicting depth `D`. Independent
marks on the cards also give a linear storage lower bound.

Thus a generic ground-cover **shader algebra** may be fixed, but each authored
offline community needs its own hard depth/byte bake gate. Offline species
union does not provide a universal source-complexity bound.

### 6.1 What the line-event census does and does not prove

On one fixed line with `F` **inequivalent exterior free intervals** having
distinct successors, a binary decision tree with depth `D` has at most `2^D`
leaves, so

\[
\boxed{D\ge\lceil\log_2 F\rceil.}
\]

The qualification "inequivalent" is essential. Pointed origins related by a
horizontal lattice translation represent the same local query. Their world
hit translation is reconstructed analytically; they do not need separate
leaves.

The measured 155 m raw surface-event census on 384 actual-source lines is:

| quantity | p50 | p95 | maximum |
|---|---:|---:|---:|
| raw surface events | 10 | 1,077 | 4,481 |
| distinct charts | 5 | 481 | 692 |
| chart runs | 5 | 723 | 4,182 |

Those values reject a tiny literal deep-event record. They are **not** by
themselves decision-depth lower bounds for the exterior periodic quotient:

1. intervals inside closed tubes may fade and must be removed;
2. triangle events internal to another solid need not bound an exterior cell;
3. an exactly horizontal lattice-aligned line repeats the same local successor
   pattern every tile.

The last point invalidates a tempting but wrong inference. The preserved
horizontal worst query has `3,588 = 12 x 299` events. Its first period is ten
open panels plus the two sides of one tube, but translating the origin by one
`0.52 m` lattice period produces the same local query. It proves that origin
phase matters *within one period*; it does not prove 3,588 distinct leaves.

At `0.1 degrees`, raw p95/max event counts reach roughly `1,372/1,377` and the
vertical drift prevents the exact horizontal lattice equivalence. Even there,
the preserved artifact did not classify union-exterior free intervals, so
turning `1,377` into a formal `ceil(log2(...))` lower bound would again outrun
the data.

**Honest consequence:** the fifth coordinate is proved necessary, but the
existing line census does not certify its exterior-only binary decision depth.
A fresh exterior-union interval classification would be required for that
number.

### 6.2 Unpointed-line ownership is independently nontrivial

The earlier 4D owner probe found one current-scale cell with at least `202`
owners plus miss, hence

\[
\lceil\log_2 203\rceil=8
\]

binary decisions before separator geometry. After one full 4D subdivision,
every sampled subcell still exceeded four decisions and the worst required at
least six.

A factorized construction which first locates an unpointed line cell and then
performs predecessor search adds the unknown exterior-fibre depth to this
eight-decision floor. An integrated 5D tree need not add the two depths.

Thus the only certified information-depth floor carried into this audit is
eight from the unpointed exterior subset. The sampled pointed-page probe sees
as many as thirteen decisions, but because it includes unclassified solid
origins and omits continuous elevation, thirteen is an engineering observation,
not the exterior-only theorem.

### 6.3 Reads, predicates, and divergence

The most optimistic binary-node runtime assumes one dependent read contains
the exact predicate and both child addresses. It costs

\[
D\ \text{serial node reads}+1\ \text{leaf/payload read}.
\]

The certified eight-decision exterior subset gives **9 serial reads** including
payload in this idealized model. The sampled pointed pages suggest paths up to
14 reads, but do not certify that maximum on the corrected domain. Therefore
the user's willingness to move modestly above nine genuinely reopens the read
count; reads alone do not refute an integrated 5D DAG.

That count is optimistic. A Pluecker separator has five independent
homogeneous coefficients after removing scale, plus child addresses. An exact
32-bit representation does not fit an ordinary 16-byte node. If the node
stores only a global predicate id, every level adds a dependent coefficient
read; even the certified `D=8` path approaches `2D+1 = 17` serial reads.
Computing the separator from two surface records needs at least as much
traffic. A successful format must show how exact predicates and child topology
fit the same read, rather than charging only abstract decision depth.

Fetching all path predicates independently is not generally possible: the
predicate used at depth `k+1` depends on the first `k` outcomes. Fetching the
entire depth-`D` tree to remove dependency reads as many as `2^D-1` nodes. An
oblivious small-hyperplane classifier is a different, much stronger geometric
assumption; no evidence says the source's visibility cells share one such
global separator set.

Pixels in one wave may take different children at every categorical boundary.
Predicated unrolling avoids control-flow deadlock, but memory addresses still
diverge. The exact sampled field has only `34.13%` angular-neighbour codeword
reuse, so coherent node locality cannot be assumed.

## 7. Measured storage, recomputed under the exterior-air-gap contract

All decimal byte counts below are deliberate; the project cap should state
whether `MB` or `MiB` before any implementation decision.

### 7.1 The 4D corner-incidence census is already large

The accepted `256 x 256 x 16 x 4` atlas defines `3,145,728` current 4D cells.
The sixteen-corner owner-union histogram contains:

- `18,573,759` unique owner/cell incidences;
- median `6`, p95 `12`, p99 `15`, maximum `16` owners per cell;
- `1,908,893` cells above four owners;
- `772,572` above eight;
- `144,225` above twelve;
- `2,981,297` nonempty cells.

For a separate binary tree in every current cell, the minimum full binary-tree
node count is

\[
18,573,759-2,981,297=15,592,462.
\]

This gives the following **sampled-layout lower accounting**:

| leaf token | node token | 4-byte cell index | bytes before predicates/payload |
|---:|---:|---:|---:|
| 4 B | 4 B | yes | 149,247,796 |
| 8 B | 4 B | yes | 223,542,832 |
| 8 B | 8 B | yes | 285,912,680 |

The 4-byte leaf can only be an indirection; exact triangle/copy/attribute data
must live elsewhere. The census is also only a corner lower bound: it omits
owners appearing strictly inside a cell, the pointed origin dimension,
horizontal continuation, and unsampled directions.

DAG sharing could reduce duplicated topology across coarse-cell boundaries.
The table is therefore not an information-theoretic lower bound for every DAG.
But sharing does not reduce decision depth, and the actual exact-record audit
found only `1.81%` exact hit-record deduplication. There is no measured DAG
sharing factor capable of closing the missing-dimension gap.

### 7.2 The actual adaptive pointed-line probe

The preserved artifact is:

```text
data/work/groundcover-adaptive-line-cell-probe/
  2ed57f59d86e8376/0f4abaf3b1b6eddb/report.json

report sha256
c51195163febbd3d32576273ccae9f0e08723afbe0a0abddbea49939f4efc69b
```

It traced `173,502` rays in 30 deterministic coarse pages and projected 6,784
pages. With a six-decision separator tail:

- only `17/30` sampled pages passed within measured refinement;
- `13/30` still failed;
- unresolved cells retained up to `22.815 m` sampled depth span;
- projected optimistic bytes were `219,462,656`;
- projected explicit bytes were `357,457,664`.

The optimistic projection assumes free global chart payload and compact
integer tokens. The explicit projection charges 12-byte event leaves and
16-byte separators.

The probe did not classify and discard origins lying inside the small closed
tube volumes. Consequently its byte totals are evidence for the full sampled
pointed domain, not a formal lower bound for the exterior-only subset. Most of
the source consists of zero-thickness open ribbons, lanceolate panels, and hair
panels, for which every non-surface origin remains exterior; the exact
horizontal lower bound in Section 6.1 isolates those explicitly. A future
exterior-only bake would still be required before claiming a final byte lower
bound.

### 7.3 A critical limitation of that probe

The report is not a complete 5D probe. Every named elevation
`0, 0.1, 1, 5, 15, 35, 75, 90 degrees` is a **fixed slice**. Inside a
nonvertical page it varies `(origin x, origin y, origin z, azimuth)` but not
elevation. Therefore it measures 4D slices through the pointed domain, not the
continuous 5D complex between those slices.

At dyadic refinement level three, adding the missing elevation coordinate
changes one failed page's uniform one-owner fallback from

\[
2^{4\cdot3}=4,096
\quad\text{to}\quad
2^{5\cdot3}=32,768
\]

subcells, an eightfold increase. Replaying the report's own page weights with
8-byte owner/copy leaves gives approximately `957.8 MB` for the corresponding
full-5D failed-page projection, before exact separator coefficients, surface
attributes, and continuum certification.

This `957.8 MB` is layout-specific, not a universal lower bound on every
possible DAG. Its decisive meaning is narrower: the preserved `219.5 MB`
number cannot be cited as evidence that the complete 5D field fits.

### 7.4 Allowing a deeper fixed tail still does not produce a fit

Re-evaluating the stored sampled pages, without tracing new rays, gives the
following compact bookkeeping:

\[
B=b_pN_p+
b_l\sum_c\max(1,n_c)+
b_n\sum_c\bigl(\max(1,n_c)-1\bigr),
\]

where `n_c` is the sampled owner count of refined subcell `c`, `b_l` is leaf
bytes, `b_n` node bytes, and `b_p=8` is the page descriptor. Failing pages use
the report's own next-level one-owner fallback. This is layout accounting, not
a new trace or a continuum claim.

| binary tail depth | sampled pages passing by level 2 | 4 B leaf + 4 B node | 8 B owner/copy + 4 B node | 8 B + 8 B |
|---:|---:|---:|---:|---:|
| 8 | 30/30 | 208.4 MB | 315.1 MB | 416.7 MB |
| 13 | 30/30 | 158.3 MB | 237.4 MB | 316.5 MB |

The apparently smaller depth-13 layout selects coarser pages and spends more
decisions instead of more subdivision. It is still only the fixed-elevation
sample, not the continuum. Its `237.4 MB` case leaves almost no room under a
250 MB envelope for:

- exact separator coefficients;
- the page/root index;
- complete chart/copy records;
- normals, colour, UV/mark data;
- the missing elevation coordinate;
- any bake or runtime scratch.

The source GCRP itself is `118,663,280` bytes. Even the earlier restricted
canonically-visible geometry estimate was about `21.4 MB`; continuum-exact
owners can include triangles absent from canonical samples. Thus the compact
sampled topology cannot plausibly be both the working set and the complete
resident representation under 250 MB.

## 8. Why block compression and DAG sharing do not change the verdict

### 8.1 Hardware block compression

Standard fixed-rate texture block compression is lossy and targets normalized
colour channels. Corrupting a predicate id, child pointer, owner, hit/miss bit,
or separator coefficient changes a categorical cell and is not exact.

A custom lossless block codec can reduce bytes only by decoding variable
blocks. Exact random access then needs extra index reads and enough fixed ALU
to decode the worst block; variable probing or a dictionary walk is another
traversal. The measured categorical field is also hostile to local reuse:
`98.19%` of exact hit records survive deduplication and angular-neighbour exact
reuse is only `34.13%`.

Colour/normal payloads may still use appropriate block compression after the
categorical event is known. That does not compress the missing point-location
function.

### 8.2 Decision-DAG sharing

DAG sharing is legitimate and should be used by any future bake. It can share:

- identical leaf surface records;
- identical separator predicates;
- identical suffix subtrees;
- miss leaves.

But it cannot:

- merge inequivalent event copies which return different local hit records;
- merge categorical hit and miss cells;
- make a fixed-elevation probe certify elevation;
- remove the origin coordinate from exterior air gaps.

The current source has `270,535` independently authored procedural charts and
`2,171,134` triangles. Its exact hit records and angular neighbours show little
measured repetition. A large miraculous suffix-sharing factor is possible in
logic but is not evidence-backed, and no existing artifact measures it.

## 9. Exact horizontal, vertical, periodic, and overlap cases

### 9.1 Vertical

The dominant-axis line charts represent `d=(0,plus_or_minus1,0)` exactly.
There is no slope infinity. A vertical pointed ray in an air gap still uses its
origin phase, but finite vertical extent limits its event count.

### 9.2 Horizontal

Exactly horizontal is also ordinary in the dominant-axis atlas. If the origin
is above the botanical slab, the exact answer is miss. If the origin lies in
an exterior air gap inside the slab, the periodic first-passage problem is
real. The measured 155 m line with `4,481` events proves that it cannot be
replaced by one top record or a tiny successor tail.

Finite horizon makes the answer finite; it does not make its complexity tiny.
Increasing `R` changes the required maximum depth/storage, so any fixed bound
is a community-and-horizon bake contract, not a universal constant.

### 9.3 Periodic copies

Periodicity lets all world copies share source chart payload. It does not make
successor rank periodic along an arbitrary finite ray: the ray phase advances
through the torus, and the first eligible copy depends on origin and direction.
The selected leaf must encode or reconstruct the winning copy offset.

### 9.4 Overlapping species and moss

Let

\[
\mathcal G=\bigcup_s\mathcal G_s
\]

be the offline marked union of grass species, flowers, litter, and moss. The
visibility complex of the union returns the exact minimum event and attached
mark. This satisfies zero per-species runtime work.

However, the complexity of a union never decreases monotonically in general:
new components add silhouettes, order boundaries, and air-gap successor
events. Arbitrary runtime species subsets would require separate compiled
communities or a new subset dimension. The single-species Calamagrostis is
already over budget in every explicit measured layout, so offline union is
semantically correct but does not rescue those layouts.

## 10. Source reauthoring boundary

Exact-preserving changes may merge coplanar, same-attribute tessellation panels
or share repeated payloads. They do not remove curved terminations, order
changes, or repeated copy events. Procedural grouping already reduces the
source to `270,535` botanical charts, yet measured lines still have p95/max
`723/4,182` chart runs.

Materially lowering the visibility complex requires changing the represented
set, for example:

- replacing subpixel plume hairs with a filtered extinction/radiance field;
- compiling blades into a small direction-analytic invariant family;
- authoring a bounded-complexity structural proxy.

Those may preserve **perceptual** fidelity at declared footprints, and they are
reasonable research directions. They do not preserve the exact categorical
first-surface field of the accepted arbitrary mesh. Once such a proxy is
chosen, its analytic invariance—not a generic visibility complex—is doing the
important compression. It belongs to the structural-core/filtered-plume or
class-E research tracks and requires its own visual/geometry gate.

## 11. Performance interpretation

For the exact route, the best mathematically honest runtime statements are:

- **ALU:** one exact separator evaluation per decision; the certified
  unpointed exterior subset has an eight-bit information floor, while the
  uncorrected pointed-page sample observes up to thirteen;
- **serial reads:** idealized certified floor 9 including payload; sampled
  pointed paths suggest up to 14; indirect exact separator coefficients can
  raise these toward 17–27;
- **total reads:** equal to serial reads for an ordinary BSP path; making them
  independent requires fetching an exponentially larger tree or proving an
  unsupported oblivious separator set;
- **divergence:** branchless/predicated execution is possible, but addresses
  diverge categorically across neighbouring pixels;
- **locality:** poor at owner boundaries; the measured angular exact-record
  neighbour reuse is 34.13%;
- **resident bytes:** sampled compact slice topology 158–237 MB at depth 13,
  before complete 5D coverage and payload; exact sampled explicit depth-6
  layout 357.5 MB;
- **working bytes:** source plus even compact sampled topology already exceeds
  250 MB before arrangement construction scratch.

The user's permission for a compelling variant to move a couple of reads above
nine **does** remove the old read-count-only rejection. The remaining blocker
is that no complete exterior-only 5D DAG has measured its total nodes, exact
predicate representation, serial reads, or payload bytes. The known layouts
are red; the entire representation family is not mathematically disproved.

## 12. Final decision and objective resume condition

### Decision

**Park implementation of exact exterior visibility-complex point location for
the accepted Calamagrostis and generic multi-species ground cover.** The 4D
boundary reduction is refuted. The integrated 5D route is an unmeasured
high-cost possibility, not a proved no-go and not yet the sought breakthrough.

The decisive reason is not that a fixed-depth DAG would be a forbidden loop;
it would not. The decisive reasons are:

1. exterior air-gap origins preserve the fifth coordinate;
2. top/side boundary charts provably cannot recover successor rank;
3. the raw event and pointed-page measurements used by earlier rejections are
   not clean exterior-only decision-depth certificates;
4. exact separator coefficients and child topology are absent from the
   optimistic read accounting;
5. the only pointed-line page probe omits continuous elevation and is already
   219.5/357.5 MB at a six-decision tail;
6. compact deeper sampled layouts leave no demonstrated room for predicates,
   complete payload, the missing dimension, or source attributes under 250 MB;
7. overlapping species/moss increase the same offline complex.

### Resume only if all of the following exist

1. a complete **five-dimensional** bake over continuous elevation and all
   exterior air-gap origins, not fixed direction slices;
2. continuum certification of every leaf or an explicitly bounded filtered
   measure at true subpixel cells;
3. exact horizontal and vertical batteries at `R=155 m`;
4. a measured maximum total and serial read count, including separator
   coefficients and payload—not just tree depth;
5. actual resident and peak working bytes including page roots, DAG nodes,
   predicate tables, chart/copy payloads, attributes, and scratch;
6. demonstrated fit under an explicitly approved memory/performance envelope;
7. the same gate on an offline union containing at least overlapping grass and
   moss.

Without that artifact, further separator tuning, uniform subdivision, or
compression speculation is grinding the same disproved premise.

## 13. Evidence ledger

### Accepted source

```text
src/assets/groundcover/calamagrostis-canescens.gcrp
sha256 2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c
bytes 118,663,280
triangles 2,171,134
procedural charts 270,535
```

### Pointed-line page probe

```text
data/work/groundcover-adaptive-line-cell-probe/
  2ed57f59d86e8376/0f4abaf3b1b6eddb/report.json
report sha256 c51195163febbd3d32576273ccae9f0e08723afbe0a0abddbea49939f4efc69b
tool sha256   1681c0845a921455673a4b7e49098eb02201052816c2a8cc47039bc8c30d216e
```

### Procedural chart and same-line census

```text
data/work/groundcover-procedural-chart-feasibility/
  2ed57f59d86e8376/cc8609cd49533b5b/metrics.json
metrics sha256 fac690a8e28218bd5429efad7e12ac3689c9e38fd11fcd30e6cd7af324f691ae
tool sha256    c715feb7d9d6eb5ccdeb6916f03ce5afc1458a5093b8cfd94519ac91cc761af0
```

### Prior derivations used

- `docs/tasks/2026-07-25/groundcover/active/GRASS-CHEAP-PROJECTIVE-RAY-MATH.md`
  §§8.1, 14.2, 22.4–22.5, 23.
- `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-EXACT-REPRESENTATION-THEORY.md`
  §§1–3 and 13.2.
- `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-ORIENTED-LINE-EXTERIOR-INSIDE-DERIVATION.md`
  §§2–6.
- `docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-PROCEDURAL-CHART-FEASIBILITY.md`.
