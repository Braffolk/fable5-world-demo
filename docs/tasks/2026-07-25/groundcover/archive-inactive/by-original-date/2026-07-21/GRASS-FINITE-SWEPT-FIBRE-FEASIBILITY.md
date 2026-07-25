# Finite swept botanical fibres: exactness boundary and rejection

Date: 2026-07-22  
Decision: **reject before visual fitting**  
Scope: pure mathematics plus immutable actual-source topology/provenance; no
runtime asset, shader, render pass, or browser path was changed.

## 1. Candidate and why it is not the four filled slabs

The candidate was the strongest obvious way to retain Sannikov's projected
ray solve while using the procedural author's freedom rather than the accepted
triangle soup. Partition the community into at most four affine botanical
families. In family `i`, an inverse affine chart turns every culm, leaf strip,
panicle axis, pedicel, hair, and spikelet carrier into a finite sweep parallel
to one constant generator. An arbitrarily detailed periodic two-dimensional
mask supplies the transverse cross-sections. Each connected transverse event
also carries its own finite generator interval, taper/cross-section record,
normal, colour, and complete botanical/community mark.

This is **not** the rejected four-slab representation. A slab occupies every
mask point for one whole shared height band. The finite-fibre candidate instead
keeps separate short intervals for individual branchlets, hairs, leaves, and
stems, so it does not deliberately fill the empty space between them. Swept
ribbons and polygonal tubes are included: their constant-generator side panels
reduce to marked finite intervals over transverse boundary curves. End caps
are separate interval endpoint events.

The hoped-for live path was one predetermined complete-record lookup per
family, one affine projected-path reconstruction, then a fixed four-way nearest
reduction. That would have been at most four reads and below 256
FMA-equivalent operations, with no loop, march, traversal, candidate list,
shell, pass, or species multiplier.

## 2. Exact projected-path equation

In one inverse affine chart write the pointed live ray as

\[
q(t)=q_0+t w_q,\qquad h(t)=h_0+t w_h,
\]

where `q` is the two-dimensional transverse coordinate and `h` is the sweep
coordinate. For `s=||w_q||>0`, a directed transverse successor field returns
ordered mask-boundary events

\[
e_n=(\rho_n,I_n,m_n),\qquad
0\leq\rho_1<\rho_2<\cdots,
\]

where `I_n=[a_n,b_n]` is that strand's finite axial eligibility interval and
`m_n` is its complete authored mark. The exact live parameter of transverse
event `n` is still Sannikov's inexpensive scale:

\[
t_n=\frac{\rho_n}{s}.
\]

But finite botanical geometry adds the necessary eligibility test

\[
\boxed{h_0+\frac{w_h}{s}\rho_n\in I_n.}
\]

Consequently the exact side hit is

\[
\boxed{n_* = \min\{n:\ h_0+k\rho_n\in I_n\},\qquad
t_*=\rho_{n_*}/s,\quad k=w_h/s.}
\]

The filled extrusion theorem is the special case `I_n=R` (or one common
clipping band handled before the successor lookup). Only in that special case
is `n_*=1` independent of live elevation. Giving each botanical element its
real finite endpoints removes the overfill but also removes the collapse.

## 3. Why one record per family cannot be exact

Take two projected crossings with `rho_1<rho_2` and distinct finite intervals.
There is a supported line `h(rho)=h_0+k rho` for which

\[
h(\rho_1)\notin I_1,\qquad h(\rho_2)\in I_2.
\]

For example, a horizontal ray (`k=0`) at the second strand's height suffices
whenever the two height intervals are disjoint. The first transverse successor
is then not a 3D hit and the second is. Repeating the construction with `N`
preceding intervals makes the eligible successor rank `N+1`. A periodic
community and an offline union of overlapping cover can contain arbitrarily
many such crossings, so no fixed first/fourth successor theorem exists.

There are only three ways to remove this counterexample:

1. expand every `I_n` to the common band, which is exactly the rejected filled
   extrusion/slab and destroys the branch/hair voids;
2. store and test a successor tail, which is a runtime candidate list and has
   unbounded required rank; or
3. precompute the already eligible successor as a function of `(q,theta,h_0,k)`.

The last field has five continuous coordinates: two transverse phase
coordinates, transverse direction, live axial origin, and live axial slope.
Sannikov's parallel-extrusion symmetry removes `(h_0,k)` only when membership
is invariant along the generator. Finite tops, bottoms, tapering branchlets,
and camera-inside origins require them back. Even an unrealistically coarse
two-bin discretisation of only one restored coordinate doubles the barely
fitting four-family slab allocation from `50,466,816` to `100,933,632` bytes;
exact continuous conditioning is not a finite version of that estimate.

Packing endpoints into the returned record does not solve selection. It only
lets the shader discover that the nearest projected crossing is invalid, after
which the absent next crossing is still required. It also breaks the existing
record budget: a complete depth/normal/colour/mark record already consumes the
optimistic eight bytes used by the slab gate. Charging only ten bytes after
adding quantised endpoints costs `63,083,520` bytes at the same four-family,
`222x222`, 24-direction, `4/3`-footprint layout, above the unchanged
`51,121,152`-byte cap.

## 4. Exact limiting directions and camera-inside result

- `w_h=0` is not singular algebraically; it is the decisive horizontal-height
  eligibility case. The result depends on `h_0`, which the 2D successor field
  deliberately does not contain.
- `w_q=0` has no transverse successor. Exact finite fibres require choosing the
  nearest forward cap among all intervals covering the fixed transverse phase.
  Overlapping stems, leaves, and species again form an origin-dependent ordered
  successor set, not one global cap.
- Moving a camera forward on one oriented line changes which interval endpoint
  or side event is the next successor. A complete exterior record cannot answer
  every inside origin by relabelling its distance; right-censoring the ordered
  event set is required.

Thus this representation does not merely have a grazing approximation. Its
missing coordinates are exercised exactly at horizontal view and by arbitrary
camera-inside motion, both hard requirements.

## 5. Why four affine strand directions cannot preserve the source

The immutable accepted source is
`src/assets/groundcover/calamagrostis-canescens.gcrp`, SHA-256
`2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c`,
with `2,049,985` vertices and `2,171,134` triangles. Its deterministic source
generator is `tools/groundcover-bake/EstonianGraminoids.ts`, SHA-256
`b430c03c4c8d24e545923bd20fbeecf10525889df07c53dc62d299549c0531e9`.

The bound procedural-chart census already proves that the finite intervals are
not a hypothetical corner case:

| authored component | independent finite charts |
|---|---:|
| foliage ribbons | 102 |
| culm tubes | 6 |
| rhizome tubes | 5 |
| panicle-axis tubes (rachises, recursive branches, pedicels, spikelet axes) | 20,969 |
| spikelet surfaces | 18,478 |
| callus hairs or filaments | 203,258 |
| anther surfaces | 27,717 |

The panicle generator uses 22 primary axes per shoot, golden-angle azimuths,
jittered upward directions, recursively changing directions, curved
four-section polylines, variable radii, finite pedicels, and finite hairs.
Those elements cannot be recooked into four constant affine generator
directions without either visibly quantising the fluffy upward plume or
projecting their union into filled intervals. The accepted actual-mesh line
census independently contains p95 `1,077` / max `4,481` surface events and p95
`723` / max `4,182` chart runs on one line. Four first-crossing records are not
an origin-aware successor representation for this topology.

This source evidence is used only after the analytic exactness boundary. It is
not presented as a new silhouette score. A numbered PNG fit was intentionally
not produced: the candidate fails the arbitrary-horizontal and camera-inside
mathematical contract before it becomes a viable visual representation. A
pretty exterior view could not change that hard rejection, and generating one
would repeat the project's recorded harness-before-method failure.

## 6. Cost and shader implications

Had the first-crossing invariant held, the path would have used four
predetermined reads, one array binding, four inverse-affine projected-path
solves, inverse-transpose normal transport, and a branchless nearest reduction.
Register pressure would have been four decoded candidates plus the current
minimum; no workgroup barrier, synchronization, dispatch, downstream pass, or
runtime species count would change.

It does not hold. Correcting it requires unbounded successor traversal or a
five-dimensional origin/elevation field. Both violate the stated runtime and
resident-memory contract. Therefore no shader performance rationale can
authorize implementation, and no runtime code was touched.

## 7. Provenance boundary and resume condition

The public GameDev.ru article PDF, SHA-256
`f470d9b13f161c20dbd06bd25675853d0a573c775141ae1db20abd39a31ed7eb`,
supplies Sannikov's projected-distance scaling for geometry invariant along the
ignored extrusion dimension. The interval-decorated fibre construction,
eligibility equation, restored-coordinate proof, fixed-rank counterexample,
camera-inside analysis, and budget are LAAS derivations. The source component
and line-event counts come from the previously bound actual-source procedural
chart experiment; they are not claims made by Sannikov.

Decision: **reject and park after the mathematical gate.** Resume only if a new
analytic geometry class preserves finite branch endpoints and panicle voids
while proving that its exact eligible successor has bounded rank at most four,
or supplies an equivalent closed-form origin-aware selector under the same
four-read, 256-FMA, and resident-memory limits. Do not convert this result into
more height bands, endpoint candidates, a shell stack, a per-species query, or
runtime traversal.
